"""Check caller-queue adoption, cancellation, restart and bounded cleanup ownership.

Only drained work may call a method. HTTP-style commands can inspect or cancel
while that callback is blocked. An abandoned or failed callback never advances
the accepted match and fences reuse of its possibly changed borrowed object.
Accepted-turn evidence must reach disk before any save or close can hide a miss.
"""

# pyright: reportPrivateUsage=false

import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from typing import NoReturn, cast

import jax
import jax.numpy as jnp
import pytest
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.test_visual_debugger_systems import _send
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client import control, launch
from marl_battlegrounds._battle_client import service as service_module
from marl_battlegrounds._battle_client.input import InputDispatchResult
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.protocol import (
    CancelSystemCommandV1,
    CommandRequestV1,
    DebuggerCommandV1,
    FinishSystemCommandV1,
    KeyboardCommandV1,
    ResetCommandV1,
    ScenarioSwitchCommandV1,
    SetCombatConfigurationCommandV1,
    SetViewCommandV1,
)
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds._battle_client.system_worker import SystemWorker
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.policies.actor import ActorAction


def _session(method: System, *, borrow_systems: bool = True) -> DebuggerSession:
    return control.create_session(
        get_scenario("arena_5v5"),
        seed=7,
        evaluation_launch_specification=debugger_test_launch_specification(7),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="system:counter",
        team_b_controller="manual",
        systems=(method, None),
        execution_information_mode="shared_obs",
        borrow_systems=borrow_systems,
    )


def _method(calls: list[str], *, fail_restart: bool = False) -> System:
    def initialize(
        variables: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> list[int]:
        calls.append("init")
        if fail_restart and calls.count("init") > 1:
            raise RuntimeError("Initializer changed its opaque state")
        return [0 for _ in inputs.valid]

    def apply(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> SystemOutput:
        calls.append("apply")
        zeros = jnp.zeros(inputs.active_mask.shape, jnp.int32)
        return SystemOutput(
            ActorAction(zeros, zeros, zeros), [value + 1 for value in memory]
        )

    return System("Borrowed counter", apply, init=initialize, execution="host")


def _service(
    session: DebuggerSession, directory: Path | None = None
) -> tuple[DebuggerService, SystemWorker]:
    worker = SystemWorker(caller_thread=True)
    worker.install(session, directory=directory)
    service = DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        system_worker=worker,
    )
    return service, worker


def _finish(service: DebuggerService, worker: SystemWorker) -> None:
    while worker.operation is not None:
        while service.drain_system_work():
            pass
        operation = worker.operation
        assert operation is not None
        _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
        assert not service.faulted


def test_queue_accepts_once_and_writes_turn_evidence_before_save_or_close(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    session = _session(_method(calls))
    service, worker = _service(session, tmp_path)
    try:
        frame = service.current_frame()
        request = CommandRequestV1(
            client_id="caller-test",
            base_revision=frame.revision,
            command_id="one-turn",
            command=KeyboardCommandV1(key="Enter"),
        )
        service.apply_command(request)
        service.apply_command(request)
        assert calls == ["init"]
        assert service.session is session
        assert service.drain_system_work()
        assert calls == ["init", "apply"]
        assert service.session is session
        _finish(service, worker)
        service.apply_command(request)
        assert calls == ["init", "apply"]
        assert service.session.current_evaluation_frame.frame_index == 1
        details = json.loads((tmp_path / "session.json").read_text())
        assert details["accepted_turns"] == 1
        assert details["saved_prefixes"] == []
        assert worker.active is not None and worker.active.decision is None
    finally:
        service.close()


def test_borrowed_restart_keeps_both_adapters_and_resets_only_match_memory() -> None:
    calls: list[str] = []
    service, worker = _service(_session(_method(calls)))
    try:
        pair = service.session.systems
        first_installation = worker.active
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service, worker)
        old_memory = service.session.system_memory
        _send(service, ResetCommandV1())
        assert calls == ["init", "apply"]
        _finish(service, worker)
        assert service.session.systems is pair
        assert service.session.system_memory is not old_memory
        assert service.session.system_memory is not None
        assert service.session.system_memory.team_a == [0]
        assert service.session.run_generation == 1
        assert calls == ["init", "apply", "init"]
        assert first_installation is not None and first_installation.closed
    finally:
        service.close()


def test_slow_cancel_keeps_http_access_and_fences_borrowed_restart() -> None:
    started, release = Event(), Event()
    calls: list[str] = []
    method = _method(calls)
    original_apply = method.apply

    def slow(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> SystemOutput:
        started.set()
        release.wait()
        return original_apply(variables, memory, inputs, key)

    method = System(method.name, slow, init=method.init, execution="host")
    service, worker = _service(_session(method))
    session = service.session
    _send(service, KeyboardCommandV1(key="Enter"))
    errors: list[BaseException] = []

    def browser() -> None:
        started.wait()
        try:
            assert service.current_frame().frame_index == 0
            operation = worker.operation
            assert operation is not None
            _send(service, CancelSystemCommandV1(operation_id=operation.identifier))
        except BaseException as error:
            errors.append(error)
        finally:
            release.set()

    thread = Thread(target=browser)
    thread.start()
    try:
        assert service.drain_system_work()
        thread.join()
        assert not errors
        _finish(service, worker)
        assert worker.failed
        assert service.session is session
        _send(service, ResetCommandV1())
        _send(service, KeyboardCommandV1(key="Enter"))
        assert worker.operation is None
        assert calls == ["init", "apply"]
        assert service.session.system_memory is not None
        assert service.session.system_memory.team_a == [0]
    finally:
        release.set()
        thread.join()
        service.close()


def test_cancel_before_drain_calls_no_method_and_drains_owned_cleanup() -> None:
    calls: list[str] = []
    service, worker = _service(_session(_method(calls)))
    try:
        _send(service, KeyboardCommandV1(key="Enter"))
        operation = worker.operation
        assert operation is not None
        _send(service, CancelSystemCommandV1(operation_id=operation.identifier))
        _finish(service, worker)
        assert calls == ["init"]
        assert service.session.current_evaluation_frame.frame_index == 0
        assert worker.active is None
    finally:
        service.close()


def test_failed_restart_keeps_accepted_memory_and_forbids_retry() -> None:
    calls: list[str] = []
    service, worker = _service(_session(_method(calls, fail_restart=True)))
    accepted = service.session
    try:
        _send(service, ResetCommandV1())
        _finish(service, worker)
        assert worker.failed
        assert service.session is accepted
        _send(service, ResetCommandV1())
        assert worker.operation is None
        assert calls == ["init", "init"]
    finally:
        service.close()


@pytest.mark.parametrize(
    "command",
    [
        ScenarioSwitchCommandV1(scenario_name="interactive"),
        SetCombatConfigurationCommandV1(
            team_a_controller="manual",
            team_b_controller="manual",
            execution_information_mode="shared_obs",
        ),
    ],
)
def test_borrowed_assignments_reject_replacement_before_callback(
    command: DebuggerCommandV1,
) -> None:
    calls: list[str] = []
    service, worker = _service(_session(_method(calls)))
    try:
        session = service.session
        result = _send(service, command)
        assert result.outcome == "response"
        assert service.session is session
        assert worker.operation is None
        assert calls == ["init"]
    finally:
        service.close()


def test_successful_recording_futures_are_pruned_but_failures_remain(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    service, worker = _service(_session(_method(calls)), tmp_path)
    installation = worker.active
    assert installation is not None
    for index in range(25):
        worker.record_saved_prefix(
            installation=installation,
            replay_id=str(index),
            path=tmp_path / str(index),
            transitions=index,
        )
        assert len(worker.recording_futures) == 1
        assert worker.drain()
        assert worker.recording_futures == []
    assert len(cast(list[object], installation.metadata["saved_prefixes"])) == 25

    def fail() -> None:
        raise OSError("Disk failed")

    installation.write_details = fail
    worker.record_saved_prefix(
        installation=installation,
        replay_id="failed",
        path=tmp_path / "failed",
        transitions=26,
    )
    assert worker.drain()
    assert len(worker.recording_futures) == 1
    assert isinstance(worker.recording_futures[0].exception(), OSError)
    with pytest.raises(BaseExceptionGroup, match="cleanup failed"):
        service.close()
    assert len(worker.recording_futures) == 1


def test_drain_rejects_another_thread_and_ctrl_c_reaches_original_caller() -> None:
    calls: list[str] = []
    session = _session(_method(calls))
    service, worker = _service(session)
    errors: list[BaseException] = []

    def wrong_thread() -> None:
        try:
            worker.drain()
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)

    def interrupted() -> NoReturn:
        raise KeyboardInterrupt

    worker.decide(session, interrupted)
    with pytest.raises(KeyboardInterrupt):
        worker.drain()
    service.close()
    assert worker.failed
    assert calls == ["init"]
    assert service.session is session


@pytest.mark.parametrize("borrowed", [False, True])
def test_continuation_cannot_change_method_ownership(borrowed: bool) -> None:
    session = _session(_method([]), borrow_systems=borrowed)
    with pytest.raises(ValueError, match="raw continuation identity"):
        replace(session, borrow_systems=not borrowed)


@pytest.mark.parametrize("forged_root", [False, True])
def test_fixed_service_accepts_evaluator_root_but_rejects_replacement(
    monkeypatch: pytest.MonkeyPatch, forged_root: bool
) -> None:
    calls: list[str] = []
    session = control.create_session(
        prepare_battle_setup(scenario_id=1).scenario,
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="system:counter",
        team_b_controller="reactive_tdm",
        systems=(_method(calls), None),
        execution_information_mode="shared_obs",
        borrow_systems=True,
    )
    service, worker = _service(session)
    if forged_root:
        original = service_module.dispatch_command

        def forged(*args: object, **kwargs: object) -> InputDispatchResult:
            result = original(*args, **kwargs)  # pyright: ignore[reportArgumentType]
            successor = replace(
                result.session, key=jax.random.key(0), raw_continuation_identity=None
            )
            return replace(
                result,
                session=successor,
                raw_continuation_identity=successor.raw_continuation_identity,
            )

        monkeypatch.setattr(service_module, "dispatch_command", forged)
    try:
        for index in range(1 if forged_root else 3):
            _send(service, KeyboardCommandV1(key="Enter"))
            assert worker.drain()
            operation = worker.operation
            assert operation is not None
            _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
            if forged_root:
                assert service.faulted
                assert service.session is session
            else:
                _finish(service, worker)
                assert service.session.current_evaluation_frame.frame_index == index + 1
                assert (
                    service.session.current_evaluation_frame.simulator_step_count
                    == 296 + index
                )
                assert service.session.key is session.key
    finally:
        service.close()


def test_failed_install_cannot_retry_a_changed_borrowed_resource() -> None:
    calls: list[str] = []
    method = _method(calls)

    @contextmanager
    def failed_scope(recording: bool) -> Generator[None]:
        calls.append("scope")
        if calls:
            raise RuntimeError("Resource setup changed its opaque state")
        yield

    method = System(
        method.name,
        method.apply,
        init=method.init,
        execution="host",
        resource_scope=failed_scope,
    )
    session = _session(method)
    worker = SystemWorker(caller_thread=True)
    with pytest.raises(RuntimeError, match="opaque state"):
        worker.install(session, directory=None)
    with pytest.raises(RuntimeError, match="unused worker"):
        worker.install(session, directory=None)
    assert calls == ["init", "scope"]
    worker.close()
    assert not worker.drain()


def test_failed_borrowed_apply_preserves_saved_prefix_and_fences_all_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    opaque = {"mutated": False}
    original = _method(calls)

    def apply(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> SystemOutput:
        if calls.count("apply"):
            opaque["mutated"] = True
            calls.append("failed")
            raise RuntimeError("Opaque method changed before failure")
        return original.apply(variables, memory, inputs, key)

    method = System(original.name, apply, init=original.init, execution="host")

    def browser(service: DebuggerService, **kwargs: object) -> int:
        worker = service._system_worker
        assert worker is not None
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service, worker)
        accepted = service.session
        assert accepted.current_evaluation_frame.frame_index == 1
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service, worker)
        assert worker.failed and opaque["mutated"]
        assert service.session is accepted
        assert service.recording_status is not None
        assert service.recording_status.lifecycle == "saved"
        assert service.recording_status.captured_transition_count == 1
        recorder = service._recorder
        assert recorder is not None and recorder.verified_loaded_bundle is not None
        assert len(recorder.verified_loaded_bundle.replay.transitions) == 1
        previous_calls = tuple(calls)
        _send(service, KeyboardCommandV1(key="Enter"))
        _send(service, ResetCommandV1())
        assert tuple(calls) == previous_calls
        assert worker.operation is None
        assert service.session is accepted
        return 0

    monkeypatch.setattr(launch, "serve_browser_debugger", browser)
    assert (
        launch._launch_setup(
            prepare_battle_setup(map_id=0, team_a=method),
            port=0,
            open_browser=False,
            output_dir=tmp_path,
            capture_replay=True,
            asset_root=tmp_path,
        )
        == 0
    )
    assert calls == ["init", "apply", "failed"]


def test_shared_borrowed_registration_is_read_once_across_install_turn_and_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registrations: list[object] = []
    original = control.normalize_system_registration

    def registered(
        value: object, *, phase: str, frozen: bool = False
    ) -> tuple[str, dict[str, object]]:
        registrations.append(value)
        return original(value, phase=phase, frozen=frozen)

    monkeypatch.setattr(control, "normalize_system_registration", registered)
    method = _method([])
    session = control.create_session(
        get_scenario("arena_5v5"),
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
    assert registrations == [method]
    assert session.system_registrations is not None
    service, worker = _service(session)
    try:
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service, worker)
        _send(service, ResetCommandV1())
        _finish(service, worker)
        assert registrations == [method]
        assert service.session.system_registrations is session.system_registrations
        assert worker.active is not None
        recorded = cast(dict[str, object], worker.active.metadata["systems"])
        assert recorded["0"] is recorded["1"] is session.system_registrations[0][1]
        with pytest.raises(ValueError, match="raw continuation identity"):
            replace(service.session, system_registrations=None)
        with pytest.raises(ValueError, match="setup records must match"):
            replace(
                service.session,
                system_registrations=None,
                raw_continuation_identity=None,
            )
    finally:
        service.close()


@pytest.mark.parametrize("identifier", range(1, 9))
def test_every_fixed_scenario_reaches_terminal_through_real_service(
    identifier: int,
) -> None:
    calls: list[str] = []
    setup = prepare_battle_setup(scenario_id=identifier)
    session = control.create_session(
        setup.scenario,
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="system:counter",
        team_b_controller=cast(str, setup.team_b),
        systems=(_method(calls), None),
        execution_information_mode="shared_obs",
        borrow_systems=True,
    )
    initial_tick = session.current_evaluation_frame.simulator_step_count
    service, worker = _service(session)
    try:
        assert service.current_presentation().outcome == "response"
        for turn in range(session.evaluation_context.expected_horizon):
            assert not service.session.episode_sealed
            _send(service, KeyboardCommandV1(key="Enter"))
            _finish(service, worker)
            assert not service.faulted
            frame = service.session.current_evaluation_frame
            assert frame.frame_index == turn + 1
            assert frame.simulator_step_count == initial_tick + turn + 1
            assert service.session.team_b_controller == setup.team_b
            assert service.current_presentation().outcome == "response"
            if service.session.episode_sealed:
                break
        assert service.session.episode_sealed
        assert (
            calls
            == ["init"]
            + ["apply"] * service.session.current_evaluation_frame.frame_index
        )
        assert service.session.key is session.key
    finally:
        service.close()


def test_worker_status_reuses_exact_frame_while_view_and_step_build_new_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    service, worker = _service(_session(_method(calls)))
    original = service._build_frame
    builds: list[object] = []

    def counted(**kwargs: object) -> object:
        builds.append(service.session)
        return original(**kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(service, "_build_frame", counted)
    try:
        _send(service, KeyboardCommandV1(key="Enter"))
        assert builds == []
        assert service.current_frame() == original()
        status = service.current_frame().system_operation
        assert status is not None and status.state == "thinking"
        _send(service, SetViewCommandV1(view_mode="pov"))
        assert len(builds) == 2
        assert service.current_frame() == original()
        operation = worker.operation
        assert operation is not None
        _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
        assert len(builds) == 2
        assert service.current_frame() == original()
        assert worker.drain()
        _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
        assert len(builds) == 4
        assert service.current_frame() == original()
        assert service.current_frame().frame_index == 1
        status = service.current_frame().system_operation
        assert status is not None and status.state == "finishing"
        _send(service, SetViewCommandV1(view_mode="researcher"))
        assert len(builds) == 5
        assert service.current_frame() == original()
        assert worker.drain()
        _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
        assert len(builds) == 5
        assert service.current_frame() == original()
        assert service.current_frame().system_operation is None
        assert calls == ["init", "apply"]
    finally:
        service.close()


def test_async_evidence_failure_does_not_break_inspection_before_status_poll() -> None:
    service, worker = _service(_session(_method([])))
    _send(service, KeyboardCommandV1(key="Enter"))
    assert worker.drain()
    operation = worker.operation
    assert operation is not None
    _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
    accepted = service.session
    assert accepted.current_evaluation_frame.frame_index == 1
    installation = worker.active
    assert installation is not None

    def failed_record() -> None:
        raise OSError("Accepted turn metadata write failed")

    installation.write_details = failed_record
    assert worker.drain()
    published = service.current_frame()
    assert published.system_operation is not None
    assert published.system_operation.state == "finishing"
    assert service.current_presentation().outcome == "response"
    assert service.session is accepted
    assert not service.faulted
    _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
    updated = service.current_frame()
    assert updated.system_operation is not None
    assert updated.system_operation.state == "failed"
    assert service.current_presentation().outcome == "response"
    assert service.session is accepted
    with pytest.raises(BaseExceptionGroup, match="cleanup failed"):
        service.close()
