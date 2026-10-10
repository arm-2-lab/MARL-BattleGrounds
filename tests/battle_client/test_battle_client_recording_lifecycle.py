"""Check saved live matches, atomic replacements, and exact episode evidence.

Saves keep the accepted live frame and never open Viewer by themselves. A new
match starts only after a successful revision-matched save or explicit discard.
Saved files remain distinct, failed saves retain their bytes, and worker traces
stay joined to the episode that actually accepted each mixed action.
"""

# pyright: reportPrivateUsage=false

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
from pydantic import ValidationError
from tests.battle_client.test_battle_client_policy_trace import _varying
from tests.battle_client.test_battle_client_worker import _finish, _method
from tests.visual_debugger.test_visual_debugger_service import (
    _close_recording_in_lifecycle,
    _recording_service,
    _request,
)
from tests.visual_debugger.test_visual_debugger_systems import _send

from marl_battlegrounds._battle_client import launch, replay_recorder
from marl_battlegrounds._battle_client import service as service_module
from marl_battlegrounds._battle_client.protocol import (
    CommandResponseV2,
    ConfirmDiscardAndReplaceCommandV1,
    FinishAndReviewCommandV1,
    KeyboardCommandV1,
    ResetCommandV1,
    RetrySaveCommandV1,
    ReviewReplayCommandV1,
    RosterSelectionCommandV1,
    SaveAndReplaceCommandV1,
    SaveAsCommandV1,
)
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds.evaluation.models import EvaluationTransitionV1
from marl_battlegrounds.evaluation.replay_io import ReplaySaveError, load_replay


@pytest.mark.parametrize("turns", [0, 1])
def test_finish_review_restart_keeps_files_and_seals_turns(
    tmp_path: Path, turns: int
) -> None:
    service, first = _recording_service(tmp_path)
    for _ in range(turns):
        _send(service, KeyboardCommandV1(key="Enter"))
    accepted = service.session
    saved = _send(service, FinishAndReviewCommandV1())
    assert saved.replay_handoff is None
    assert first.lifecycle == "saved" and not first.status.restart_fenced
    assert service.session is accepted
    assert first.retained_frame_count == turns + 1
    assert first.saved_bundle is not None
    path = first.saved_bundle.replay_path
    original = path.read_bytes()
    _send(service, KeyboardCommandV1(key="Enter"))
    assert service.session is accepted
    _send(service, RosterSelectionCommandV1(role="control", global_slot=1))
    assert service.session.current_evaluation_frame is accepted.current_evaluation_frame
    for _ in range(2):
        reviewed = _send(service, ReviewReplayCommandV1())
        assert reviewed.replay_handoff is not None
        assert first.lifecycle == "saved"
    _send(service, ResetCommandV1())
    second = service._recorder
    assert second is not None and second is not first
    assert first.lifecycle == "saved" and path.read_bytes() == original
    assert second.retained_frame_count == 1 and second.validated_transition_count == 0
    assert service.session.run_generation == accepted.run_generation + 1
    _send(service, KeyboardCommandV1(key="Enter"))
    _send(service, FinishAndReviewCommandV1())
    assert second.saved_bundle is not None
    assert second.saved_bundle.replay_path != path
    assert len(load_replay(second.saved_bundle.replay_path).replay.transitions) == 1
    assert path.read_bytes() == original
    service.close()


def test_atomic_save_replace_is_revision_checked_and_duplicate_safe(
    tmp_path: Path,
) -> None:
    service, first = _recording_service(tmp_path)
    _send(service, KeyboardCommandV1(key="Enter"))
    accepted = service.session
    request = _request(
        "save-replace",
        base_revision=service.revision,
        command=SaveAndReplaceCommandV1(replacement=ResetCommandV1()),
    )
    result = service.apply_command(request)
    second = service._recorder
    assert isinstance(result.payload, CommandResponseV2)
    assert result.replay_handoff is None
    assert first.lifecycle == "saved"
    assert service.session.run_generation == accepted.run_generation + 1
    duplicate = service.apply_command(request).payload
    assert isinstance(duplicate, CommandResponseV2) and duplicate.result == "duplicate"
    stale = service.apply_command(request.model_copy(update={"command_id": "stale"}))
    assert stale.outcome == "stale_revision"
    assert second is not None
    assert service._recorder is second and second.lifecycle == "recording"
    assert len(list(tmp_path.glob("*.marlbg-replay.json"))) == 1
    service.close()


@pytest.mark.parametrize("recovery", ["retry", "save_as", "discard"])
def test_failed_save_never_replaces_and_recovery_never_replays_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recovery: str
) -> None:
    service, first = _recording_service(tmp_path)
    _send(service, KeyboardCommandV1(key="Enter"))
    accepted = service.session
    actual = replay_recorder.publish_prepared_replay

    def fail(*args: object, **kwargs: object) -> object:
        raise ReplaySaveError("temporary_write_failed", path=None, detail="Injected")

    monkeypatch.setattr(replay_recorder, "publish_prepared_replay", fail)
    _send(service, SaveAndReplaceCommandV1(replacement=ResetCommandV1()))
    prepared = first.prepared_replay
    assert first.lifecycle == "persistence_failed" and first.status.discard_available
    assert service.session is accepted and service._recorder is first
    assert first.validated_transition_count == 1
    monkeypatch.setattr(replay_recorder, "publish_prepared_replay", actual)
    if recovery == "discard":
        _send(service, ConfirmDiscardAndReplaceCommandV1(replacement=ResetCommandV1()))
        assert first.lifecycle == "discarded" and service.session is not accepted
        assert not list(tmp_path.glob("*.marlbg-replay.json"))
    else:
        _send(
            service,
            RetrySaveCommandV1()
            if recovery == "retry"
            else SaveAsCommandV1(file_name="recovered.marlbg-replay.json"),
        )
        assert service.session is accepted and service._recorder is first
        assert first.lifecycle == "saved" and first.prepared_replay is prepared
        _send(service, ResetCommandV1())
        assert service.session.run_generation == accepted.run_generation + 1
    service.close()


@pytest.mark.parametrize("stage", ["reset", "destination"])
def test_saved_replacement_failure_retains_old_frame_and_permits_explicit_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    service, first = _recording_service(tmp_path)
    _send(service, KeyboardCommandV1(key="Enter"))
    accepted = service.session
    with monkeypatch.context() as patch:
        if stage == "reset":
            actual = service_module.dispatch_command

            def fail(*args: object, **kwargs: object) -> object:
                if isinstance(args[1], ResetCommandV1):
                    raise RuntimeError("Injected reset failure")
                return actual(*args, **kwargs)  # pyright: ignore[reportArgumentType]

            patch.setattr(service_module, "dispatch_command", fail)
        else:

            def fail_destination(*args: object, **kwargs: object) -> object:
                raise RuntimeError("Injected destination failure")

            patch.setattr(
                replay_recorder, "preflight_replay_destination", fail_destination
            )
        _send(service, SaveAndReplaceCommandV1(replacement=ResetCommandV1()))
    assert first.saved_bundle is not None
    assert first.lifecycle == "saved" and first.saved_bundle.replay_path.is_file()
    assert service.session is accepted and service._recorder is first
    assert service.current_frame().recording == first.status and not service.faulted
    _send(service, RetrySaveCommandV1())
    assert service.session is accepted
    _send(service, ResetCommandV1())
    assert service.session.run_generation == accepted.run_generation + 1
    service.close()


@pytest.mark.parametrize(
    "name", ["../bad.marlbg-replay.json", "/tmp/bad.marlbg-replay.json", "bad.json"]
)
def test_save_as_rejects_unsafe_browser_names(name: str) -> None:
    with pytest.raises(ValidationError):
        SaveAsCommandV1(file_name=name)


def test_two_saved_mixed_episodes_keep_exact_trace_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    records: list[dict[str, Any]] = []
    method = _varying(records)
    evidence: list[dict[str, object]] = []

    def browser(service: DebuggerService, **kwargs: object) -> int:
        worker = service._system_worker
        assert worker is not None
        methods = service.session.systems
        identities = []
        for count in (2, 1):
            installation = worker.active
            recorder = service._recorder
            assert installation is not None and recorder is not None
            assert installation.directory is not None
            accepted_transitions: list[EvaluationTransitionV1] = []
            for _ in range(count):
                _send(service, KeyboardCommandV1(key="Enter"))
                _finish(service, worker)
                transition = service.session.incoming_evaluation_view
                assert transition is not None
                accepted_transitions.append(transition.transition)
            accepted = service.session
            start = perf_counter()
            if count == 2:
                _send(service, SaveAndReplaceCommandV1(replacement=ResetCommandV1()))
            else:
                _send(service, FinishAndReviewCommandV1())
            save_seconds = perf_counter() - start
            saved = recorder.saved_bundle
            assert saved is not None
            for _ in range(5):
                service._acknowledge_saved_system_prefix()
            if count == 2:
                _finish(service, worker)
                assert service.session.systems is methods
                assert service.session.run_generation == accepted.run_generation + 1
            while service.drain_system_work():
                pass
            details = json.loads((installation.directory / "session.json").read_text())
            trace = [
                json.loads(line)
                for line in (installation.directory / "policy_trace.jsonl")
                .read_text()
                .splitlines()
            ]
            replay = load_replay(saved.replay_path).replay
            identities.append(replay.artifact_id)
            assert len(replay.frames) == count + 1
            assert replay.transitions == tuple(accepted_transitions)
            assert (
                details["accepted_turns"]
                == len(trace)
                == len(replay.transitions)
                == count
            )
            assert details["saved_prefixes"] == [
                {
                    "replay_id": replay.artifact_id,
                    "path": str(saved.replay_path),
                    "transitions": count,
                }
            ]
            assert [row["accepted_frame_index"] for row in trace] == list(
                range(1, count + 1)
            )
            assert [row["decision_step"] for row in trace] == [
                [index] for index in range(count)
            ]
            assert [row["policy_ids"][0][0] for row in trace] == [-1] * count
            components = details["systems"]["0"]["components"]
            assert [components[row["policy_ids"][0][1]]["name"] for row in trace] == [
                f"Host part {index}" for index in range(count)
            ]
            assert [row["policy_ids"][0][5] for row in trace] == list(range(count))
            assert recorder.current_frame == accepted.current_evaluation_frame
            assert len(installation.saved_prefix_requests) == 1
            evidence.append(
                {
                    "transitions": count,
                    "save_seconds": save_seconds,
                    "replay_bytes": saved.replay_path.stat().st_size,
                    "trace_bytes": (installation.directory / "policy_trace.jsonl")
                    .stat()
                    .st_size,
                }
            )
        assert identities[0] != identities[1]
        assert sum(row["phase"] == "init" for row in records) == 4
        assert sum(row["phase"] == "apply" for row in records) == 6
        assert [row["memory"] for row in records if row["phase"] == "apply"] == [
            [0],
            [0],
            [1],
            [1],
            [0],
            [0],
        ]
        assert len(worker.recording_futures) <= 1
        return 0

    monkeypatch.setattr(launch, "serve_browser_debugger", browser)
    launch._launch_setup(
        prepare_battle_setup(
            map_id=0, team_a=["manual", method, method, method, method], team_b=method
        ),
        port=0,
        open_browser=False,
        output_dir=tmp_path,
        capture_replay=True,
        asset_root=tmp_path,
    )
    record_property("recording_cost", json.dumps(evidence))


def test_failed_opaque_restart_preserves_saved_prefix_and_fresh_launch_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    method = _method(calls, fail_restart=True)

    def browser(service: DebuggerService, **kwargs: object) -> int:
        worker = service._system_worker
        assert worker is not None
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service, worker)
        accepted, recorder = service.session, service._recorder
        _send(service, SaveAndReplaceCommandV1(replacement=ResetCommandV1()))
        _finish(service, worker)
        assert service.session is accepted and service._recorder is recorder
        assert recorder is not None and recorder.saved_bundle is not None
        assert (
            recorder.lifecycle == "saved"
            and recorder.saved_bundle.replay_path.is_file()
        )
        assert service.current_frame().fresh_launch_required
        previous = list(calls)
        _send(service, ResetCommandV1())
        _send(service, RetrySaveCommandV1())
        assert calls == previous
        return 0

    monkeypatch.setattr(launch, "serve_browser_debugger", browser)
    launch._launch_setup(
        prepare_battle_setup(map_id=0, team_a=method, team_b="manual"),
        port=0,
        open_browser=False,
        output_dir=tmp_path,
        capture_replay=True,
        asset_root=tmp_path,
    )


def test_terminal_save_restarts_with_fresh_frame_zero(tmp_path: Path) -> None:
    service, first = _recording_service(tmp_path, maximum_episode_steps=1)
    _send(service, KeyboardCommandV1(key="Enter"))
    assert first.lifecycle == "saved" and first.validated_transition_count == 1
    old_episode = service.session.evaluation_context.identity.episode_id
    _send(service, ResetCommandV1())
    assert service.session.evaluation_context.identity.episode_id != old_episode
    assert service.session.current_evaluation_frame.frame_index == 0
    assert service._recorder is not first
    assert service.recording_status is not None
    assert service.recording_status.lifecycle == "recording"
    service.close()


@pytest.mark.parametrize("boundary", ["target_exists", "exclusive_link"])
def test_real_publication_failure_keeps_prefix_for_save_as(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    from marl_battlegrounds.evaluation import replay_io

    service, first = _recording_service(tmp_path)
    _send(service, KeyboardCommandV1(key="Enter"))
    accepted = service.session
    with monkeypatch.context() as patch:
        if boundary == "target_exists":
            (tmp_path / "episode.marlbg-replay.json").write_text("Existing file")
        else:

            def fail_link(*args: object, **kwargs: object) -> object:
                raise OSError("Injected exclusive publication failure")

            patch.setattr(replay_io.os, "link", fail_link)
        _send(service, SaveAndReplaceCommandV1(replacement=ResetCommandV1()))
    assert first.lifecycle == "persistence_failed" and service.session is accepted
    prepared = first.prepared_replay
    assert prepared is not None
    _send(service, SaveAsCommandV1(file_name="safe-copy.marlbg-replay.json"))
    assert first.lifecycle == "saved" and first.prepared_replay is prepared
    assert first.saved_bundle is not None
    assert load_replay(first.saved_bundle.replay_path).replay == prepared.replay
    assert service.session is accepted
    if boundary == "target_exists":
        assert (tmp_path / "episode.marlbg-replay.json").read_text() == "Existing file"
    service.close()


def test_open_reuses_verified_service_until_viewer_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.viewer.replay_protocol import (
        ReplayCommandRequestV1,
        ReplayExitCommandV1,
    )

    service, first = _recording_service(tmp_path)
    _send(service, FinishAndReviewCommandV1())
    viewer = _send(service, ReviewReplayCommandV1()).replay_handoff
    assert viewer is not None

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("Reopening this replay must not rebuild the Viewer")

    with monkeypatch.context() as patch:
        patch.setattr(service_module, "ReplayViewerService", forbidden)
        assert _send(service, ReviewReplayCommandV1()).replay_handoff is viewer
    frame = viewer.current_frame()
    viewer.apply_command(
        ReplayCommandRequestV1(
            client_id="viewer",
            command_id="exit",
            base_revision=frame.revision,
            command=ReplayExitCommandV1(),
        )
    )
    reopened = _send(service, ReviewReplayCommandV1()).replay_handoff
    assert reopened is not None and reopened is not viewer
    assert first.lifecycle == "saved"
    _send(service, ResetCommandV1())
    assert service._replay_handoff_cache is None
    service.close()


def test_keyboard_interrupt_joins_saved_prefix_before_worker_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method = _varying([])

    def browser(service: DebuggerService, **kwargs: object) -> int:
        worker = service._system_worker
        assert worker is not None
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service, worker)
        installation, recorder = worker.active, service._recorder
        assert installation is not None and installation.directory is not None
        assert recorder is not None
        assert service.close_recording_for_keyboard_interrupt().saved
        assert service.close_recording_for_keyboard_interrupt().saved
        while service.drain_system_work():
            pass
        details = json.loads((installation.directory / "session.json").read_text())
        assert recorder.saved_bundle is not None
        replay = load_replay(recorder.saved_bundle.replay_path).replay
        assert details["saved_prefixes"] == [
            {
                "replay_id": replay.artifact_id,
                "path": str(recorder.saved_bundle.replay_path),
                "transitions": 1,
            }
        ]
        assert len(installation.saved_prefix_requests) == 1
        return 0

    monkeypatch.setattr(launch, "serve_browser_debugger", browser)
    launch._launch_setup(
        prepare_battle_setup(map_id=0, team_a=method),
        port=0,
        open_browser=False,
        output_dir=tmp_path,
        capture_replay=True,
        asset_root=tmp_path,
    )


@pytest.mark.parametrize("lifecycle", ["persistence_failed", "saved", "reviewing"])
def test_closed_pov_recording_allows_focus_without_changing_drafts_or_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lifecycle: str
) -> None:
    service, recorder = _recording_service(tmp_path, view_mode="pov")
    _close_recording_in_lifecycle(service, recorder, lifecycle, monkeypatch)
    accepted = service.session
    _send(service, RosterSelectionCommandV1(role="control", global_slot=1))
    assert service.session.controlled_global_slot == 1
    assert service.session.current_evaluation_frame is accepted.current_evaluation_frame
    assert service.session.pending_actions == accepted.pending_actions
    assert recorder.lifecycle == lifecycle and recorder.validated_transition_count == 0
    service.close()
