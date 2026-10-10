"""Check server-owned Run, Pause, turn adoption and caller-context execution.

Run never owns a Manual slot, and old same-episode Pause requests stop scheduling
without cancelling the accepted in-flight turn. No browser finish request is
needed. An idle pump does no inference or display work, and failures stop play.
"""

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import replace
from threading import Event, Thread, get_ident
from time import sleep
from uuid import uuid4

import jax
import pytest
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.battle_client.test_battle_client_worker import _method, _service
from tests.helpers.visual_debugger_fixtures import debugger_test_launch_specification
from tests.visual_debugger.test_visual_debugger_systems import _create, _send

from marl_battlegrounds._battle_client import control
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.protocol import (
    ApiErrorV2,
    CancelSystemCommandV1,
    CommandRequestV1,
    CommandResponseV2,
    KeyboardCommandV1,
    PauseCommandV1,
    ResetCommandV1,
    RosterSelectionCommandV1,
    RunCommandV1,
)
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds._battle_client.system_worker import SystemWorker
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.viewer.presentation_protocol import (
    LiveOracleAuthorizedPresentationFrameV1,
)


def _autonomous(
    method: System, *, max_steps: int | None = None
) -> tuple[DebuggerService, SystemWorker]:
    session = control.create_session(
        get_scenario("arena_5v5")
        if max_steps is None
        else prepare_battle_setup(map_id=0, max_steps=max_steps).scenario,
        seed=7,
        evaluation_launch_specification=debugger_test_launch_specification(7),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="system:first",
        team_b_controller="system:second",
        systems=(method, method),
        execution_information_mode="shared_obs",
        borrow_systems=True,
    )
    return _service(session)


def _until(service: DebuggerService, predicate: Callable[[], bool]) -> None:
    while not predicate():
        if not service.pump():
            sleep(0.001)


def _pause(service: DebuggerService, *, revision: int | None = None) -> object:
    frame = service.current_frame()
    return service.apply_command(
        CommandRequestV1(
            client_id="other-tab",
            command_id=uuid4().hex,
            base_revision=frame.revision if revision is None else revision,
            command=PauseCommandV1(
                session_id=frame.session_id, run_generation=frame.run_generation
            ),
        )
    ).payload


def test_start_paused_run_and_pause_without_browser_finish() -> None:
    calls: list[str] = []
    service, worker = _autonomous(_method(calls))
    try:
        assert service.current_frame().run_state == "paused"
        initial = len(calls)
        assert not service.pump()
        assert len(calls) == initial
        first_revision = service.revision
        _send(service, RunCommandV1())
        assert service.current_frame().run_state == "running"
        _until(
            service, lambda: service.session.current_evaluation_frame.frame_index >= 2
        )
        assert not isinstance(_pause(service, revision=first_revision), ApiErrorV2)
        _until(service, lambda: worker.operation is None)
        accepted = service.session.current_evaluation_frame.frame_index
        assert accepted >= 2
        assert service.current_frame().run_state == "paused"
        count = len(calls)
        assert not service.pump()
        assert not service.pump()
        assert len(calls) == count
        _send(service, KeyboardCommandV1(key="Enter"))
        _until(service, lambda: worker.operation is None)
        assert service.session.current_evaluation_frame.frame_index == accepted + 1
        assert service.current_frame().run_state == "paused"
    finally:
        service.close()


def test_pause_allows_old_revision_only_for_the_same_episode() -> None:
    service, worker = _autonomous(_method([]))
    try:
        frame = service.current_frame()
        run = CommandRequestV1(
            client_id="tab", command_id="run", base_revision=0, command=RunCommandV1()
        )
        service.apply_command(run)
        revision = service.revision
        service.apply_command(run)
        assert service.revision == revision
        assert isinstance(
            service.apply_command(
                run.model_copy(update={"command_id": "stale"})
            ).payload,
            ApiErrorV2,
        )
        assert isinstance(_pause(service, revision=revision + 1), ApiErrorV2)
        assert service.current_frame().run_state == "running"
        assert not isinstance(_pause(service, revision=0), ApiErrorV2)
        _send(service, ResetCommandV1())
        _until(service, lambda: worker.operation is None)
        assert service.session.run_generation == 1
        assert service.current_frame().run_state == "paused"
        _send(service, RunCommandV1())
        old = CommandRequestV1(
            client_id="old-tab",
            command_id="old",
            base_revision=0,
            command=PauseCommandV1(session_id=frame.session_id, run_generation=0),
        )
        assert isinstance(service.apply_command(old).payload, ApiErrorV2)
        wrong_session = old.model_copy(
            update={
                "command_id": "wrong",
                "command": PauseCommandV1(session_id="other-session", run_generation=1),
            }
        )
        assert isinstance(service.apply_command(wrong_session).payload, ApiErrorV2)
        assert service.current_frame().run_state == "running"
    finally:
        service.close()


def test_slow_turn_keeps_caller_context_and_pause_can_take_service_lock() -> None:
    calls: list[str] = []
    started, released = Event(), Event()
    context: ContextVar[str] = ContextVar("run-test")
    token = context.set("original")
    owner = get_ident()
    method = _method(calls)
    apply = method.apply
    blocked = False

    def slow(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> SystemOutput:
        nonlocal blocked
        assert get_ident() == owner
        assert context.get() == "original"
        if not blocked:
            blocked = True
            started.set()
            # Pause must reach the service while this host call is blocked.
            assert released.wait(10)
        return apply(variables, memory, inputs, key)

    service, worker = _autonomous(replace(method, apply=slow))
    failures: list[BaseException] = []

    def stop_from_tab() -> None:
        try:
            started.wait()
            assert not isinstance(_pause(service, revision=0), ApiErrorV2)
        except BaseException as exc:
            failures.append(exc)
        finally:
            released.set()

    thread = Thread(target=stop_from_tab)
    try:
        _send(service, RunCommandV1())
        thread.start()
        _until(
            service, lambda: service.session.current_evaluation_frame.frame_index == 1
        )
        _until(service, lambda: worker.operation is None)
        thread.join()
        assert not failures
        assert calls.count("apply") == 2
        assert not service.pump()
        assert service.current_frame().run_state == "paused"
    finally:
        released.set()
        thread.join()
        service.close()
        context.reset(token)


def test_manual_slots_disallow_run_even_when_the_only_manual_agent_is_dead() -> None:
    setup = prepare_battle_setup(scenario_id=1)
    _config, state = setup.scenario.build_scenario()
    dead_slot = next(slot for slot in range(5) if not bool(state.alive_mask[slot]))
    choices = tuple("manual" if slot == dead_slot else "random" for slot in range(5))
    setup = prepare_battle_setup(scenario_id=1, team_a=choices)
    resolved = resolve_controllers(setup)
    session = control.create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=debugger_test_launch_specification(setup.seed),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller=resolved.controllers[0],
        team_b_controller=resolved.controllers[1],
        systems=resolved.systems,
        manual_slots=resolved.manual_slots,
        slot_labels=resolved.slot_labels,
        execution_information_mode="shared_obs",
    )
    assert session.manual_slots == (dead_slot,)
    assert not bool(session.state.alive_mask[dead_slot])
    service = DebuggerService(
        session, view_mode="researcher", preset="analysis", include_stress=False
    )
    try:
        result = _send(service, RunCommandV1())
        assert isinstance(result.payload, CommandResponseV2)
        assert result.payload.notice is not None and "Manual" in result.payload.notice
        assert service.current_frame().run_state == "paused"
        assert not service.pump()
    finally:
        service.close()


def test_failed_method_stops_run_without_reusing_uncertain_state() -> None:
    method = _method([])
    calls = 0

    def fail(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> SystemOutput:
        nonlocal calls
        calls += 1
        raise RuntimeError("Opaque method mutated then failed")

    service, worker = _autonomous(replace(method, apply=fail))
    try:
        _send(service, RunCommandV1())
        _until(service, lambda: service.current_frame().run_state == "paused")
        assert calls == 1
        assert worker.failed
        assert service.current_frame().fresh_launch_required
        assert service.session.current_evaluation_frame.frame_index == 0
        _send(service, RunCommandV1())
        assert service.current_frame().run_state == "paused"
        _until(service, lambda: worker.operation is None)
        assert not service.pump()
        assert calls == 1
        assert service.current_frame().fresh_launch_required
        assert service.current_snapshot().frame == service.current_frame()
    finally:
        service.close()


def test_owned_legacy_no_shared_random_uses_the_same_pump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _create(None, None, other="random_valid")
    session = control.set_combat_configuration(
        session,
        team_a_controller="random_valid",
        team_b_controller="random_valid",
        execution_information_mode="no_shared_obs",
    )
    assert session.environment_state is None
    worker = SystemWorker()
    worker.install(session, directory=None)
    service = DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        system_worker=worker,
    )
    owner = get_ident()
    threads: list[int] = []
    original = control._build_configured_joint_action

    def checked(current: DebuggerSession) -> Action:
        threads.append(get_ident())
        return original(current)

    monkeypatch.setattr(control, "_build_configured_joint_action", checked)
    try:
        _send(service, RunCommandV1())
        _until(
            service, lambda: service.session.current_evaluation_frame.frame_index >= 1
        )
        _pause(service)
        _until(service, lambda: worker.operation is None)
        assert threads and all(thread != owner for thread in threads)
    finally:
        service.close()
        for future in worker.shutdown_futures:
            future.result()


def test_terminal_stops_run_without_extra_method_calls() -> None:
    calls: list[str] = []
    service, worker = _autonomous(_method(calls), max_steps=3)
    try:
        _send(service, RunCommandV1())
        _until(
            service, lambda: service.session.episode_sealed and worker.operation is None
        )
        assert service.session.current_evaluation_frame.frame_index == 3
        assert calls.count("apply") == 6
        assert service.current_frame().run_state == "paused"
        assert not service.pump()
        _send(service, RunCommandV1())
        assert service.current_frame().run_state == "paused"
        assert calls.count("apply") == 6
    finally:
        service.close()


def test_cancellation_stops_run_and_idle_pumps_build_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    service, worker = _autonomous(_method(calls))
    try:
        _send(service, RunCommandV1())
        assert service.pump()
        operation = worker.operation
        assert operation is not None
        _send(service, CancelSystemCommandV1(operation_id=operation.identifier))
        _until(service, lambda: worker.operation is None)
        assert calls.count("apply") == 0
        assert service.current_frame().frame_index == 0
        assert service.current_frame().run_state == "paused"

        def forbidden(**kwargs: object) -> object:
            raise AssertionError("An idle pump must reuse the accepted frame")

        monkeypatch.setattr(service, "_build_frame", forbidden)
        for _ in range(100):
            assert not service.pump()
        assert calls.count("apply") == 0
    finally:
        service.close()


def test_failed_acceptance_does_not_spin_or_repeat_a_method() -> None:
    calls: list[str] = []
    service, worker = _autonomous(_method(calls))
    installation = worker.active
    assert installation is not None
    original = installation.write_details

    def fail_record() -> None:
        raise OSError("Accepted record cannot be written")

    try:
        _send(service, RunCommandV1())
        service.pump()
        service.pump()
        assert service.session.current_evaluation_frame.frame_index == 1
        installation.write_details = fail_record
        service.pump()
        frame = service.current_frame()
        assert (
            frame.system_operation is not None
            and frame.system_operation.state == "failed"
        )
        assert frame.run_state == "paused"
        assert frame.fresh_launch_required
        revision = frame.revision
        for _ in range(20):
            assert not service.pump()
        assert service.revision == revision
        assert calls.count("apply") == 2
        assert service.current_snapshot().frame == frame
        accepted = service.session
        _send(service, RosterSelectionCommandV1(role="control", global_slot=5))
        assert service.session.controlled_global_slot == 5
        assert (
            service.session.raw_continuation_identity
            is accepted.raw_continuation_identity
        )
        assert service.session.pending_actions == accepted.pending_actions
        assert service.current_frame().fresh_launch_required
        presentation = service.current_snapshot().presentation
        assert isinstance(presentation, LiveOracleAuthorizedPresentationFrameV1)
        assert presentation.fresh_launch_required
        _send(service, RunCommandV1())
        assert service.current_frame().run_state == "paused"
    finally:
        installation.write_details = original
        with pytest.raises(BaseExceptionGroup, match="cleanup failed"):
            service.close()
