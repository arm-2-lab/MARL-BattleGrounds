"""Check that Exit before the first tick writes no replay file.

Contracts checked:

- Exit sent while the recording is open with zero captured ticks schedules
  shutdown at once with the notice "No tick was played, so no replay was
  saved.", leaves the recording status unchanged (still "recording", zero
  ticks) and writes no replay file, the same rule as Ctrl-C and Restart.
- Exit after one played tick still saves the game so far before it shuts
  down, so the zero-tick rule never drops a real game.
"""

# pyright: reportPrivateUsage=false
from pathlib import Path

from tests.visual_debugger.test_visual_debugger_service import (
    _recording_service,
    _request,
)

from marl_battlegrounds._battle_client.protocol import (
    CommandResponseV2,
    ExitCommandV1,
    KeyboardCommandV1,
)


def _replay_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.marlbg-replay.json"))


def test_exit_before_the_first_tick_saves_no_replay(tmp_path: Path) -> None:
    service, recorder = _recording_service(tmp_path)
    before = recorder.status
    assert before.lifecycle == "recording"
    assert before.captured_transition_count == 0

    exited = service.apply_command(
        _request("exit-at-tick-zero", base_revision=0, command=ExitCommandV1())
    )

    assert isinstance(exited.payload, CommandResponseV2)
    assert exited.payload.result == "shutdown_scheduled"
    assert exited.payload.notice == "No tick was played, so no replay was saved."
    assert exited.shutdown_requested
    assert service.shutting_down
    assert exited.payload.frame.recording == before
    assert recorder.lifecycle == "recording"
    assert recorder.saved_bundle is None
    assert _replay_files(tmp_path) == []
    service.close()
    assert _replay_files(tmp_path) == []


def test_exit_after_one_tick_still_saves_the_game(tmp_path: Path) -> None:
    service, recorder = _recording_service(tmp_path)
    played = service.apply_command(
        _request("one-turn", base_revision=0, command=KeyboardCommandV1(key="Enter"))
    )
    assert isinstance(played.payload, CommandResponseV2)
    assert recorder.validated_transition_count == 1

    exited = service.apply_command(
        _request(
            "exit-after-one-tick",
            base_revision=service.revision,
            command=ExitCommandV1(),
        )
    )

    assert isinstance(exited.payload, CommandResponseV2)
    assert exited.payload.result == "shutdown_scheduled"
    assert exited.shutdown_requested
    assert recorder.lifecycle == "saved"
    assert recorder.saved_bundle is not None
    assert _replay_files(tmp_path) == [recorder.saved_bundle.replay_path]
    service.close()
