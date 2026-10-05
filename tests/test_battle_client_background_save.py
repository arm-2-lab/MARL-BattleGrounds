"""Check the game-over background save, the kept Replay Viewer and the URL flush.

Contracts checked:

- With ``save_in_background=True`` the step that ends the game returns at once
  with recording lifecycle "sealed" (the page says "Saving Replay…"); a view
  switch still answers while the save runs; the save then installs exactly
  one new revision whose recording says "saved", and the file loads back.
- Exit and a Discard-and-restart sent while that save runs wait for it: Exit
  then shuts down with the replay saved, and the Discard restarts the match
  without discarding the saved replay. ``close`` also waits for the save.
- Ctrl-C before the first tick writes no replay file.
- The recorder's trusted prepare path writes the same bytes as the strict
  ``PreparedReplay`` check.
- With ``keep_viewer_after_exit=True`` an opened Replay Viewer keeps serving
  after the live Exit, ``release_kept_viewer`` hands it over, and
  ``serve_until_closed`` returns once that Viewer shuts down.
- ``serve_debugger_server`` flushes its URL line before it starts serving, so
  a launcher reading through a pipe sees it at once.
"""

# pyright: reportPrivateUsage=false
import io
from collections.abc import Generator
from http.client import HTTPConnection
from pathlib import Path
from threading import Event, Thread

import pytest
from tests.test_visual_debugger_service import _recording_service, _request

from marl_battlegrounds._battle_client.protocol import (
    CommandResponseV2,
    ConfirmDiscardAndReplaceCommandV1,
    ExitCommandV1,
    FinishAndReviewCommandV1,
    KeyboardCommandV1,
    ResetCommandV1,
    ReviewReplayCommandV1,
    SetViewCommandV1,
)
from marl_battlegrounds._battle_client.recording import (
    DebuggerRecordingCloseCauseV1,
    DebuggerRecordingPublicationOutcomeV1,
)
from marl_battlegrounds._battle_client.recording_coordinator import (
    RecordingDebuggerCoordinator,
)
from marl_battlegrounds._battle_client.replay_recorder import DebuggerReplayRecorder
from marl_battlegrounds._battle_client.service import (
    DebuggerService,
    ServiceCommandResult,
)
from marl_battlegrounds.evaluation.replay_io import PreparedReplay, load_replay
from marl_battlegrounds.viewer import server as server_module
from marl_battlegrounds.viewer.server import DebuggerHTTPServer, create_server

_ASSET_ROOT = Path(__file__).resolve().parents[1] / "web" / "visual_debugger"


@pytest.fixture
def save_gate(monkeypatch: pytest.MonkeyPatch) -> Generator[Event]:
    gate = Event()
    original = DebuggerReplayRecorder.finalize_and_save

    def gated(
        recorder: DebuggerReplayRecorder,
        close_cause: DebuggerRecordingCloseCauseV1,
        *,
        failure_reason: str | None = None,
    ) -> DebuggerRecordingPublicationOutcomeV1:
        gate.wait()
        return original(recorder, close_cause, failure_reason=failure_reason)

    monkeypatch.setattr(DebuggerReplayRecorder, "finalize_and_save", gated)
    yield gate
    # Never leave a save thread blocked, even after a failed assertion.
    gate.set()


def _background_service(
    tmp_path: Path,
) -> tuple[DebuggerService, DebuggerReplayRecorder]:
    plain, recorder = _recording_service(tmp_path, maximum_episode_steps=1)
    service = DebuggerService(
        plain.session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        session_id="background-save-test",
        recorder=recorder,
        save_in_background=True,
    )
    return service, recorder


def _send(
    service: DebuggerService, command: object, command_id: str
) -> ServiceCommandResult:
    return service.apply_command(
        _request(command_id, base_revision=service.revision, command=command)
    )


def _end_game(service: DebuggerService) -> ServiceCommandResult:
    return _send(service, KeyboardCommandV1(key="Enter"), "last-turn")


def test_game_over_answers_at_once_and_saves_in_one_later_revision(
    tmp_path: Path, save_gate: Event
) -> None:
    service, recorder = _background_service(tmp_path)
    ended = _end_game(service)
    assert isinstance(ended.payload, CommandResponseV2)
    assert ended.payload.result == "applied"
    assert ended.payload.frame.terminal.is_sealed
    assert ended.payload.frame.recording is not None
    assert ended.payload.frame.recording.lifecycle == "sealed"
    assert service.recording_status == ended.payload.frame.recording
    job = service._background_save
    assert job is not None and job.thread is not None and job.thread.is_alive()

    view = _send(service, SetViewCommandV1(view_mode="pov"), "view-while-saving")
    assert isinstance(view.payload, CommandResponseV2)
    assert view.payload.frame.recording is not None
    assert view.payload.frame.recording.lifecycle == "sealed"
    revision_while_saving = service.revision

    save_gate.set()
    job.thread.join()
    assert service._background_save is None
    assert service.revision == revision_while_saving + 1
    saved_frame = service.current_frame()
    assert saved_frame.revision == service.revision
    assert saved_frame.recording == recorder.status
    assert recorder.lifecycle == "saved" and recorder.saved_bundle is not None
    loaded = load_replay(recorder.saved_bundle.replay_path)
    assert loaded.replay == recorder.verified_loaded_bundle.replay  # type: ignore[union-attr]
    assert len(loaded.replay.transitions) == 1
    service.close()


@pytest.mark.parametrize("command_kind", ["exit", "discard"])
def test_recording_commands_wait_for_the_running_save(
    tmp_path: Path, save_gate: Event, command_kind: str
) -> None:
    service, recorder = _background_service(tmp_path)
    _end_game(service)
    command = (
        ExitCommandV1()
        if command_kind == "exit"
        else ConfirmDiscardAndReplaceCommandV1(replacement=ResetCommandV1())
    )
    results: list[ServiceCommandResult] = []
    sender = Thread(
        target=lambda: results.append(_send(service, command, "during-save"))
    )
    sender.start()
    # This wait is meant to expire: the command must still be waiting for the save.
    sender.join(0.5)
    assert sender.is_alive() and not results
    save_gate.set()
    sender.join()
    (result,) = results
    assert isinstance(result.payload, CommandResponseV2)
    assert recorder.lifecycle == "saved" and recorder.saved_bundle is not None
    assert recorder.saved_bundle.replay_path.is_file()
    if command_kind == "exit":
        assert result.payload.result == "shutdown_scheduled"
        assert result.shutdown_requested
        assert result.payload.frame.recording == recorder.status
    else:
        assert result.payload.result == "applied"
        assert service._recorder is not recorder
        assert service.session.current_evaluation_frame.frame_index == 0
    service.close()


def test_close_waits_for_the_running_save(tmp_path: Path, save_gate: Event) -> None:
    service, recorder = _background_service(tmp_path)
    _end_game(service)
    closer = Thread(target=service.close)
    closer.start()
    # This wait is meant to expire: close must still be waiting for the save.
    closer.join(0.5)
    assert closer.is_alive()
    save_gate.set()
    closer.join()
    assert recorder.lifecycle == "saved"
    assert service._background_save is None


def test_ctrl_c_before_the_first_tick_writes_no_replay(tmp_path: Path) -> None:
    service, recorder = _recording_service(tmp_path)
    closed = service.close_recording_for_keyboard_interrupt()
    assert closed.saved
    assert closed.message == "No tick was played, so no replay was saved."
    assert recorder.lifecycle == "recording"
    assert not list(tmp_path.glob("*.marlbg-replay.json"))
    service.close()


def test_trusted_prepare_writes_the_strict_bytes(tmp_path: Path) -> None:
    service, recorder = _recording_service(tmp_path)
    _send(service, KeyboardCommandV1(key="Enter"), "one-turn")
    _send(service, FinishAndReviewCommandV1(), "finish")
    assert recorder.replay is not None and recorder.prepared_replay is not None
    assert recorder.saved_bundle is not None
    strict = PreparedReplay(recorder.replay)
    assert recorder.prepared_replay.replay_json_bytes == strict.replay_json_bytes
    assert recorder.saved_bundle.replay_path.read_bytes() == strict.replay_json_bytes
    service.close()


def test_opened_viewer_outlives_exit_until_its_own_shutdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    service, _recorder = _recording_service(tmp_path)
    _send(service, KeyboardCommandV1(key="Enter"), "one-turn")
    _send(service, FinishAndReviewCommandV1(), "finish")
    coordinator = RecordingDebuggerCoordinator(
        service, asset_root=_ASSET_ROOT, keep_viewer_after_exit=True
    )
    opened = coordinator.apply_command(
        _request(
            "open", base_revision=service.revision, command=ReviewReplayCommandV1()
        )
    )
    assert isinstance(opened.payload, CommandResponseV2)
    assert opened.payload.replay_url is not None
    exited = coordinator.apply_command(
        _request("exit", base_revision=service.revision, command=ExitCommandV1())
    )
    assert exited.shutdown_requested
    kept = coordinator.release_kept_viewer()
    coordinator.close()
    service.close()
    assert kept is not None
    viewer = kept._server
    connection = HTTPConnection("127.0.0.1", viewer.server_port)
    connection.request(
        "GET", "/api/frame", headers={"X-MARL-Debugger-Token": viewer.capability_token}
    )
    assert connection.getresponse().status == 200
    connection.close()

    viewer.request_shutdown()
    assert kept.serve_until_closed() == 0
    assert not kept._thread.is_alive()
    assert viewer.socket.fileno() == -1
    printed = capsys.readouterr().out
    assert kept.url in printed
    assert "Replay Viewer stopped." in printed


class _RecordingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def write(self, text: str) -> int:
        self.events.append(text)
        return super().write(text)

    def flush(self) -> None:
        self.events.append("<flush>")
        super().flush()


def test_url_line_is_flushed_before_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _recorder = _recording_service(tmp_path)
    stream = _RecordingStream()
    served: list[list[str]] = []

    def serve_once(server: DebuggerHTTPServer, *, poll_interval: float) -> None:
        del server, poll_interval
        served.append(list(stream.events))
        raise KeyboardInterrupt

    monkeypatch.setattr(server_module.sys, "stdout", stream)
    monkeypatch.setattr(DebuggerHTTPServer, "serve_forever", serve_once)
    server = create_server(service, asset_root=_ASSET_ROOT, port=0)
    server_module.serve_debugger_server(server, open_browser=False)
    (before_serving,) = served
    url_index = next(i for i, text in enumerate(before_serving) if "/#token=" in text)
    assert "<flush>" in before_serving[url_index + 1 :]
    service.close()
