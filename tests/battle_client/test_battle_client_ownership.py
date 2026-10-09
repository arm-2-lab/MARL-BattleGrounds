"""Check caller-owned methods and resource lifetimes in the public BattleClient.

Callbacks keep the launch thread and its exact context. Borrowed descriptors and
storage are not copied. Match evidence binds without closing launch resources;
failed or cancelled work cannot change the accepted prefix or run again silently.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import get_ident, local
from typing import TYPE_CHECKING, Never, SupportsIndex, cast

import jax
import numpy as np
import pytest

if TYPE_CHECKING:
    from marl_battlegrounds._battle_client.service import DebuggerService

from marl_battlegrounds.evaluation.host_evidence import (
    DecisionContext,
    HostRun,
    current_run,
)


@dataclass
class _Observer:
    name: str
    events: list[str]
    fail_close: bool = False

    @contextmanager
    def decision(self, context: DecisionContext) -> Generator[None]:
        self.events.append(f"decision:{self.name}")
        try:
            yield
        except BaseException:
            self.events.append(f"abandoned:{self.name}")
            raise
        else:
            self.events.append(f"accepted:{self.name}")

    def flush(self) -> None:
        self.events.append(f"flush:{self.name}")

    def completed(self, episode_ids: frozenset[int]) -> None:
        self.events.append(f"completed:{self.name}")

    def close(self) -> None:
        self.events.append(f"close:{self.name}")
        if self.fail_close:
            raise RuntimeError(f"Cannot close {self.name}")


@pytest.mark.parametrize("body_fails", [False, True])
def test_evidence_bind_restores_parent_without_closing_observers(
    tmp_path: Path, body_fails: bool
) -> None:
    events: list[str] = []
    outer = HostRun("outer", "test", "pass", tmp_path, {})
    inner = HostRun("inner", "test", "match", tmp_path, {})
    outer.register("outer", lambda: _Observer("outer", events))
    inner.register("inner", lambda: _Observer("inner", events))
    previous = current_run()
    with outer.scope():
        assert current_run() is outer
        with (
            (
                pytest.raises(ValueError, match="Callback failed")
                if body_fails
                else nullcontext()
            ),
            inner.bind(),
        ):
            assert current_run() is inner
            if body_fails:
                raise ValueError("Callback failed")
        assert current_run() is outer
        assert events == []
    assert current_run() is previous
    assert events == ["close:outer"]
    inner.close()
    assert events == ["close:outer", "close:inner"]


def test_evidence_bind_skips_device_work_and_preserves_all_cleanup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jax

    events: list[str] = []
    run = HostRun("run", "test", "match", tmp_path, {})
    run.register("first", lambda: _Observer("first", events, fail_close=True))
    run.register("second", lambda: _Observer("second", events, fail_close=True))

    def forbidden_transfer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Evidence binding must not transfer device data.")

    monkeypatch.setattr(jax, "device_get", forbidden_transfer)
    previous = current_run()
    with run.bind():
        assert current_run() is run
    assert current_run() is previous
    assert events == []
    with pytest.raises(ExceptionGroup) as failure:
        run.close()
    assert events == ["close:first", "close:second"]
    assert [str(error) for error in failure.value.exceptions] == [
        "Cannot close first",
        "Cannot close second",
    ]


class _Opaque:
    def close(self) -> None:
        raise AssertionError("Borrowed objects must not be closed directly.")

    def __copy__(self) -> object:
        raise AssertionError("Borrowed leaves must not be copied.")

    def __deepcopy__(self, _memo: object) -> object:
        raise AssertionError("Borrowed leaves must not be deep-copied.")

    def __reduce_ex__(self, _protocol: SupportsIndex) -> Never:
        raise AssertionError("Borrowed leaves must not be serialized.")

    def __array__(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("Opaque leaves must not become arrays.")


def test_borrowed_system_and_storage_survive_turns_and_restart() -> None:
    from tests.visual_debugger_fixtures import debugger_test_launch_specification

    from marl_battlegrounds._battle_client import control
    from marl_battlegrounds._battle_client.setup import prepare_battle_setup
    from marl_battlegrounds.evaluation.policy_execution import (
        PolicyTree,
        System,
        SystemInput,
        SystemOutput,
    )
    from marl_battlegrounds.policies.actor import ActorAction

    opaque = _Opaque()
    weights = np.zeros(3, np.float32)
    variables = {"opaque": opaque, "weights": weights}
    initializations: list[object] = []
    decisions: list[int] = []

    def initialize(
        actual: PolicyTree, inputs: SystemInput, _key: jax.Array
    ) -> list[dict[str, object]]:
        assert actual is variables
        assert actual["opaque"] is opaque
        assert actual["weights"] is weights
        initializations.append(actual)
        return [{"opaque": opaque, "count": 0} for _ in inputs.valid]

    def apply(
        actual: PolicyTree, memory: PolicyTree, inputs: SystemInput, _key: jax.Array
    ) -> SystemOutput:
        assert actual is variables
        assert actual["opaque"] is opaque
        assert actual["weights"] is weights
        assert memory[0]["opaque"] is opaque
        count = int(memory[0]["count"])
        decisions.append(count)
        zeros = cast(jax.Array, np.zeros(inputs.active_mask.shape, np.int32))
        return SystemOutput(
            ActorAction(zeros, zeros, zeros), [{"opaque": opaque, "count": count + 1}]
        )

    method = System(
        "Borrowed", apply, variables=variables, init=initialize, execution="host"
    )
    setup = prepare_battle_setup(
        map_id=0, team_a_roster=["warrior"], team_b_roster=["warrior"]
    )
    session = control.create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=debugger_test_launch_specification(setup.seed),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="system:borrowed",
        team_b_controller="manual",
        execution_information_mode="shared_obs",
        systems=(method, None),
        borrow_systems=True,
    )
    for _ in range(2):
        assert session.borrow_systems
        assert session.systems is not None and session.systems[0] is method
        assert session.systems[0].variables is variables
        session = control.submit_interactive(session)
    restarted = control.reset_session(session)
    assert restarted.borrow_systems
    assert restarted.systems is not None and restarted.systems[0] is method
    assert restarted.systems[0].variables is variables
    control.submit_interactive(restarted)
    assert len(initializations) == 2
    assert decisions == [0, 1, 0]
    np.testing.assert_array_equal(weights, np.zeros(3, np.float32))


def _install_launch_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: Callable[[DebuggerService], None],
) -> list[str]:
    from marl_battlegrounds._battle_client import launch
    from marl_battlegrounds.evaluation.models import CodeRevisionV2
    from marl_battlegrounds.viewer import server

    events: list[str] = []
    assets = Path(__file__).resolve().parents[2] / "web" / "visual_debugger"

    class Coordinator:
        product_kind = "combat_debugger"

    class FakeServer:
        expected_origin = "http://127.0.0.1:12345"
        capability_token = "ownership-test"
        coordinator = Coordinator()

        def __init__(self, service: DebuggerService) -> None:
            self.service = service

        def __enter__(self) -> FakeServer:
            events.append("enter")
            return self

        def __exit__(self, _kind: object, _error: object, _traceback: object) -> None:
            events.append("close")

        def serve_forever(self, *, poll_interval: float) -> None:
            assert poll_interval > 0
            events.append("serve")
            script(self.service)

        def shutdown(self) -> None:
            events.append("shutdown")

    def create_server(
        service: DebuggerService, **_kwargs: object
    ) -> server.DebuggerHTTPServer:
        return cast("server.DebuggerHTTPServer", FakeServer(service))

    def files(name: str) -> Path:
        assert name == "marl_battlegrounds.viewer"
        return tmp_path / "installed-viewer"

    @contextmanager
    def as_file(_resource: object) -> Generator[Path]:
        yield assets

    monkeypatch.setattr(
        launch,
        "discover_code_revision_v2",
        lambda: CodeRevisionV2(package_version="0.0.0"),
    )
    monkeypatch.setattr(launch, "files", files)
    monkeypatch.setattr(launch, "as_file", as_file)
    monkeypatch.setattr(server, "create_server", create_server)
    return events


@pytest.mark.parametrize("through_factory", [False, True])
def test_public_launch_preserves_caller_context_leases_and_method_across_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, through_factory: bool
) -> None:
    from tests.test_visual_debugger_systems import (
        _finish,  # pyright: ignore[reportPrivateUsage]
        _send,  # pyright: ignore[reportPrivateUsage]
    )

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds._battle_client import methods
    from marl_battlegrounds._battle_client.protocol import (
        ConfirmDiscardAndReplaceCommandV1,
        KeyboardCommandV1,
        ResetCommandV1,
        SetViewCommandV1,
    )
    from marl_battlegrounds.evaluation.policy_execution import (
        PolicyTree,
        System,
        SystemInput,
        SystemOutput,
    )
    from marl_battlegrounds.policies.actor import ActorAction

    owner = get_ident()
    thread_values = local()
    marker = object()
    thread_values.marker = marker
    context = ContextVar("ownership-test", default="before")
    token = context.set("caller")
    token_unused = True
    queued_context = ContextVar("queued-ownership-test", default="before")
    queued_token = queued_context.set("caller")
    queued_token_unused = True
    leases = 0
    lease_entries: list[tuple[bool, HostRun | None]] = []
    loads: list[str] = []
    initializations: list[int] = []
    decisions: list[int] = []
    decision_runs: list[HostRun] = []
    evidence_events: list[str] = []
    opaque = _Opaque()
    variables = {"opaque": opaque, "weights": np.zeros(2, np.float32)}

    def check_owner() -> None:
        assert get_ident() == owner
        assert thread_values.marker is marker

    def initialize(
        actual: PolicyTree, inputs: SystemInput, _key: jax.Array
    ) -> list[int]:
        nonlocal token_unused
        check_owner()
        assert actual is variables
        if token_unused:
            context.reset(token)
            token_unused = False
        assert context.get() == "before"
        initializations.append(get_ident())
        assert leases >= 2
        return [0 for _ in inputs.valid]

    def apply(
        actual: PolicyTree, memory: PolicyTree, inputs: SystemInput, _key: jax.Array
    ) -> SystemOutput:
        nonlocal queued_token_unused
        check_owner()
        if queued_token_unused:
            queued_context.reset(queued_token)
            queued_token_unused = False
        assert queued_context.get() == "before"
        assert context.get() == "before"
        assert actual is variables
        run = current_run()
        assert run is not None
        decision_runs.append(run)
        decisions.append(memory[0])
        zeros = cast(jax.Array, np.zeros(inputs.active_mask.shape, np.int32))
        return SystemOutput(ActorAction(zeros, zeros, zeros), [memory[0] + 1])

    @contextmanager
    def scope(recording: bool) -> Generator[None]:
        nonlocal leases
        check_owner()
        run = current_run()
        lease_entries.append((recording, run))
        if run is not None:
            run.register("ownership", lambda: _Observer(run.pass_id, evidence_events))
        leases += 1
        try:
            yield
        finally:
            check_owner()
            leases -= 1

    method = System(
        "Borrowed",
        apply,
        variables=variables,
        init=initialize,
        execution="host",
        resource_scope=scope,
    )

    source = "ownership_fixture:make" if through_factory else method
    expected_loads = ["ownership_fixture:make"] if through_factory else []

    def load(source: str) -> System:
        check_owner()
        loads.append(source)
        return method

    def run_commands(service: DebuggerService) -> None:
        assert get_ident() != owner
        assert service.session.systems is not None
        assert service.session.systems[0] is service.session.systems[1] is method
        for _ in range(2):
            _send(service, KeyboardCommandV1(key="Enter"))
            _finish(service)
        assert decisions == [0, 0, 1, 1]
        _send(service, SetViewCommandV1(view_mode="pov"))
        service.current_frame()
        service.current_presentation()
        assert decisions == [0, 0, 1, 1]
        assert loads == expected_loads
        _send(service, ConfirmDiscardAndReplaceCommandV1(replacement=ResetCommandV1()))
        _finish(service)
        assert service.session.current_evaluation_frame.frame_index == 0
        assert service.session.systems is not None
        assert service.session.systems[0] is service.session.systems[1] is method
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service)
        assert service.close_recording_for_keyboard_interrupt().saved

    server_events = _install_launch_server(monkeypatch, tmp_path, run_commands)
    monkeypatch.setattr(methods, "load_method", load)
    try:
        with scope(False):
            assert (
                marl_bgs.battle_client(
                    map_id=0,
                    team_a=source,
                    team_b=source,
                    team_a_roster=["warrior"],
                    team_b_roster=["warrior"],
                    max_steps=10,
                    output_dir=tmp_path / "recordings",
                    open_browser=False,
                )
                == 0
            )
            assert leases == 1
    finally:
        if token_unused:
            context.reset(token)
        if queued_token_unused:
            queued_context.reset(queued_token)
    assert not queued_token_unused
    assert queued_context.get() == "before"
    assert leases == 0
    assert context.get() == "before"
    assert current_run() is None
    assert len(initializations) == 4
    assert decisions == [0, 0, 1, 1, 0, 0]
    assert loads == expected_loads
    assert [recording for recording, run in lease_entries if run is None] == [
        False,
        False,
    ]
    assert len({id(run) for run in decision_runs[:4]}) == 1
    assert len({id(run) for run in decision_runs[4:]}) == 1
    assert decision_runs[0] is not decision_runs[-1]
    assert sum(value.startswith("accepted:") for value in evidence_events) == 3
    assert sum(value.startswith("close:") for value in evidence_events) == 2
    assert server_events[0:2] == ["enter", "serve"]
    assert server_events[-1] == "close"


@pytest.mark.parametrize("failure_at", ["initialize", "server"])
def test_public_launch_failures_release_scopes_on_the_caller_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.policy_execution import (
        PolicyTree,
        System,
        SystemInput,
        SystemOutput,
    )
    from marl_battlegrounds.policies.actor import ActorAction
    from marl_battlegrounds.viewer import server

    owner = get_ident()
    active: list[int] = []
    entered: list[int] = []
    exited: list[int] = []
    initializations: list[int] = []

    @contextmanager
    def scope(_recording: bool) -> Generator[None]:
        assert get_ident() == owner
        lease = len(entered)
        entered.append(lease)
        active.append(lease)
        try:
            yield
        finally:
            assert get_ident() == owner
            assert active.pop() == lease
            exited.append(lease)

    def initialize(
        _variables: PolicyTree, inputs: SystemInput, _key: jax.Array
    ) -> list[int]:
        assert get_ident() == owner
        initializations.append(get_ident())
        if failure_at == "initialize":
            raise RuntimeError("Initialization failed")
        return [0 for _ in inputs.valid]

    def apply(
        _variables: PolicyTree,
        memory: PolicyTree,
        inputs: SystemInput,
        _key: jax.Array,
    ) -> SystemOutput:
        zeros = cast(jax.Array, np.zeros(inputs.active_mask.shape, np.int32))
        return SystemOutput(ActorAction(zeros, zeros, zeros), memory)

    def forbidden_serving(_service: DebuggerService) -> None:
        raise AssertionError("A failed launch must never start serving.")

    def cannot_bind(*_args: object, **_kwargs: object) -> Never:
        raise OSError("Server bind failed")

    _install_launch_server(monkeypatch, tmp_path, forbidden_serving)
    monkeypatch.setattr(server, "create_server", cannot_bind)
    method = System(
        "Borrowed",
        apply,
        init=initialize,
        execution="host",
        resource_scope=scope,
    )
    expected = (
        "Initialization failed" if failure_at == "initialize" else "Server bind failed"
    )
    with pytest.raises((RuntimeError, OSError), match=expected):
        marl_bgs.battle_client(
            map_id=0,
            team_a=method,
            team_b="manual",
            team_a_roster=["warrior"],
            team_b_roster=["warrior"],
            open_browser=False,
            capture_replay=False,
        )
    assert initializations == [owner]
    assert active == []
    assert entered
    assert exited == list(reversed(entered))
    assert current_run() is None


def _exception_messages(error: BaseException) -> set[str]:
    seen: set[int] = set()
    messages: set[str] = set()

    def visit(current: BaseException) -> None:
        if id(current) in seen:
            return
        seen.add(id(current))
        messages.add(str(current))
        if isinstance(current, BaseExceptionGroup):
            for child in cast("BaseExceptionGroup[BaseException]", current).exceptions:
                visit(child)
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                visit(linked)

    visit(error)
    return messages


def test_public_launch_keeps_initialization_and_all_lease_cleanup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.policy_execution import (
        PolicyTree,
        System,
        SystemInput,
        SystemOutput,
    )

    owner = get_ident()
    closed: list[str] = []

    def make_method(label: str, *, fails_initialization: bool) -> System:
        @contextmanager
        def scope(_recording: bool) -> Generator[None]:
            assert get_ident() == owner
            try:
                yield
            finally:
                assert get_ident() == owner
                closed.append(label)
                raise RuntimeError(f"{label} cleanup failed")

        def initialize(
            _variables: PolicyTree, inputs: SystemInput, _key: jax.Array
        ) -> list[int]:
            assert get_ident() == owner
            if fails_initialization:
                raise ValueError("Second initialization failed")
            return [0 for _ in inputs.valid]

        def forbidden_apply(
            _variables: PolicyTree,
            _memory: PolicyTree,
            _inputs: SystemInput,
            _key: jax.Array,
        ) -> SystemOutput:
            raise AssertionError("Failed setup must not run a decision.")

        return System(
            label,
            forbidden_apply,
            init=initialize,
            execution="host",
            resource_scope=scope,
        )

    def forbidden_serving(_service: DebuggerService) -> None:
        raise AssertionError("Failed setup must not start serving.")

    _install_launch_server(monkeypatch, tmp_path, forbidden_serving)
    first = make_method("First", fails_initialization=False)
    second = make_method("Second", fails_initialization=True)
    with pytest.raises(Exception) as failure:
        marl_bgs.battle_client(
            map_id=0,
            team_a=first,
            team_b=second,
            team_a_roster=["warrior"],
            team_b_roster=["warrior"],
            open_browser=False,
            capture_replay=False,
        )
    assert closed == ["Second", "First"]
    assert {
        "Second initialization failed",
        "Second cleanup failed",
        "First cleanup failed",
    } <= _exception_messages(failure.value)
    assert current_run() is None
