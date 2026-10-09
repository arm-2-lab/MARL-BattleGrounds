"""Check retained component traces at the real live-worker acceptance boundary.

Every accepted turn keeps its existing PolicyTrace fields once, including Manual
-1 rows. Cancelled work writes no row, disabled capture performs no trace transfer
or write, and evidence failures preserve the accepted game and replay save path.
"""

# pyright: reportPrivateUsage=false

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import IO, Any

import jax
import pytest
from tests.battle_client.test_battle_client_mixed_control import _host, _session
from tests.battle_client.test_battle_client_worker import _finish, _service
from tests.test_visual_debugger_systems import _send

from marl_battlegrounds._battle_client import launch, system_worker
from marl_battlegrounds._battle_client.protocol import (
    CancelSystemCommandV1,
    FinishAndReviewCommandV1,
    FinishSystemCommandV1,
    KeyboardCommandV1,
    ResetCommandV1,
)
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
)


def _varying(records: list[dict[str, Any]]) -> System:
    original = _host(records, parts=8)

    def choose(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> SystemOutput:
        output = original.apply(variables, memory, inputs, key)
        assert output.policy_ids is not None
        return SystemOutput(
            output.actions,
            output.next_memory,
            policy_ids=output.policy_ids * 0 + memory[0],
        )

    return System(
        original.name,
        choose,
        init=original.init,
        execution="host",
        components=original.components,
    )


def test_every_accepted_component_choice_is_retained_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    records: list[dict[str, Any]] = []
    method = _varying(records)
    session = _session(
        prepare_battle_setup(
            map_id=0,
            team_a=["manual", method, method, method, method],
            team_b=method,
        )
    )
    service, worker = _service(session, tmp_path)
    timings: list[float] = []
    original_write = system_worker.SystemInstallation.write_policy_trace

    def timed_write(
        installation: system_worker.SystemInstallation, accepted: object
    ) -> None:
        start = perf_counter()
        original_write(installation, accepted)  # pyright: ignore[reportArgumentType]
        timings.append(perf_counter() - start)

    monkeypatch.setattr(
        system_worker.SystemInstallation, "write_policy_trace", timed_write
    )
    try:
        for _ in range(2):
            _send(service, KeyboardCommandV1(key="Enter"))
            operation = worker.operation
            assert operation is not None
            _finish(service, worker)
            _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
        rows = [
            json.loads(line)
            for line in (tmp_path / "policy_trace.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 2
        assert [row["accepted_frame_index"] for row in rows] == [1, 2]
        assert [row["decision_step"] for row in rows] == [[0], [1]]
        assert [row["policy_ids"][0][0] for row in rows] == [-1, -1]
        assert [row["policy_ids"][0][5] for row in rows] == [0, 1]
        assert all(row["valid"] == [True] for row in rows)
        details = json.loads((tmp_path / "session.json").read_text())
        assert details["policy_trace_file"] == "policy_trace.jsonl"
        declared = details["systems"]["0"]["components"]
        assert [declared[row["policy_ids"][0][1]]["name"] for row in rows] == [
            "Host part 0",
            "Host part 1",
        ]
        assert len(timings) == 2
        record_property("trace_append_seconds", json.dumps(timings))
        record_property(
            "trace_jsonl_bytes", (tmp_path / "policy_trace.jsonl").stat().st_size
        )
        assert details["accepted_turns"] == 2
        assert details["last_policy_trace"] == {
            key: value
            for key, value in rows[-1].items()
            if key != "accepted_frame_index"
        }
        assert sum(record["phase"] == "apply" for record in records) == 4
    finally:
        service.close()


@pytest.mark.parametrize("choose_first", [False, True])
def test_cancelled_decision_has_no_accepted_trace(
    tmp_path: Path, choose_first: bool
) -> None:
    records: list[dict[str, Any]] = []
    session = _session(prepare_battle_setup(map_id=0, team_a=_varying(records)))
    service, worker = _service(session, tmp_path)
    try:
        _send(service, KeyboardCommandV1(key="Enter"))
        operation = worker.operation
        assert operation is not None
        if choose_first:
            assert worker.drain()
        _send(service, CancelSystemCommandV1(operation_id=operation.identifier))
        _finish(service, worker)
        assert service.session is session
        assert not (tmp_path / "policy_trace.jsonl").exists()
        assert (
            json.loads((tmp_path / "session.json").read_text())["accepted_turns"] == 0
        )
    finally:
        service.close()


def test_disabled_trace_capture_skips_transfer_and_file_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(prepare_battle_setup(map_id=0, team_a=_varying([])))
    service, worker = _service(session)
    try:
        _send(service, KeyboardCommandV1(key="Enter"))
        operation = worker.operation
        assert operation is not None
        assert worker.drain()
        _send(service, FinishSystemCommandV1(operation_id=operation.identifier))

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("Disabled capture must skip this work")

        with monkeypatch.context() as patch:
            patch.setattr(system_worker.jax, "device_get", forbidden)
            patch.setattr(Path, "open", forbidden)
            assert worker.drain()
        _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
        assert worker.active is not None
        assert "policy_trace_file" not in worker.active.metadata
        assert "last_policy_trace" not in worker.active.metadata
    finally:
        service.close()


def test_trace_append_failure_keeps_accepted_game_saveable_and_fences_play(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records: list[dict[str, Any]] = []
    method = _varying(records)

    def browser(service: DebuggerService, **kwargs: object) -> int:
        worker = service._system_worker
        assert worker is not None
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service, worker)
        installation = worker.active
        assert installation is not None
        assert installation.directory is not None
        trace_path = installation.directory / "policy_trace.jsonl"
        original_open = Path.open

        def fail_append(
            path: Path,
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> IO[Any]:
            if path == trace_path and mode == "a":
                raise OSError("Trace append failed")
            return original_open(path, mode, buffering, encoding, errors, newline)

        monkeypatch.setattr(Path, "open", fail_append)
        _send(service, KeyboardCommandV1(key="Enter"))
        operation = worker.operation
        assert operation is not None and worker.drain()
        _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
        accepted = service.session
        assert accepted.current_evaluation_frame.frame_index == 2
        assert worker.drain()
        assert service.current_presentation().outcome == "response"
        _send(service, FinishSystemCommandV1(operation_id=operation.identifier))
        assert service.current_presentation().outcome == "response"
        previous_calls = len(records)
        _send(service, KeyboardCommandV1(key="Enter"))
        _send(service, ResetCommandV1())
        assert len(records) == previous_calls
        assert service.session is accepted
        _send(service, FinishAndReviewCommandV1())
        assert service.recording_status is not None
        assert service.recording_status.lifecycle == "saved"
        recorder = service._recorder
        assert recorder is not None and recorder.verified_loaded_bundle is not None
        assert len(recorder.verified_loaded_bundle.replay.transitions) == 2
        assert len(trace_path.read_text().splitlines()) == 1
        while service.drain_system_work():
            pass
        return 0

    monkeypatch.setattr(launch, "serve_browser_debugger", browser)
    with pytest.raises(BaseExceptionGroup, match="cleanup failed"):
        launch._launch_setup(
            prepare_battle_setup(map_id=0, team_a=method),
            port=0,
            open_browser=False,
            output_dir=tmp_path,
            capture_replay=True,
            asset_root=tmp_path,
        )


def test_legacy_installation_does_not_advertise_a_missing_system_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(prepare_battle_setup(map_id=0))
    worker = system_worker.SystemWorker()
    installation = system_worker.SystemInstallation(directory=tmp_path)
    try:
        worker._describe(installation, session)
        assert "policy_trace_file" not in installation.metadata

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("A legacy session has no System trace to write")

        with monkeypatch.context() as patch:
            patch.setattr(system_worker.jax, "device_get", forbidden)
            patch.setattr(Path, "open", forbidden)
            installation.write_policy_trace(session)
        assert not (tmp_path / "policy_trace.jsonl").exists()
    finally:
        installation.close()
        worker.close()
