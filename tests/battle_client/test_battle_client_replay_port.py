"""Check the BattleClient's chosen Open Replay port.

Contracts checked:

- ``battle_client`` refuses a ``replay_port`` that is not an integer from 0
  through 65535, or that equals a nonzero ``port``, before any setup, and
  passes a valid one on to the launch.
- With a fixed ``replay_port``, Open Replay serves the Replay Viewer on exactly
  that loopback port, and the returned address names it.
- Opening a different replay on a fixed port stops the old Viewer first and
  serves the new one on the same port, with the new token.

The default, 0, keeps today's free-port behavior; the BattleClient and
recording coordinator tests cover it unchanged.
"""

# pyright: reportPrivateUsage=false
import socket
from http.client import HTTPConnection
from pathlib import Path
from typing import cast

import pytest
from tests.visual_debugger.test_visual_debugger_service import (
    _recording_service,
    _request,
)

import marl_battlegrounds._battle_client.launch as launch
from marl_battlegrounds._battle_client.protocol import (
    CommandResponseV2,
    FinishAndReviewCommandV1,
    ReviewReplayCommandV1,
)
from marl_battlegrounds._battle_client.recording_coordinator import (
    RecordingDebuggerCoordinator,
)
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds.viewer.replay_service import ReplayViewerService
from marl_battlegrounds.viewer.server import DebuggerHTTPServer

_ASSET_ROOT = Path(__file__).resolve().parents[2] / "web" / "visual_debugger"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return cast(int, probe.getsockname()[1])


def _finished_service(folder: Path, scenario_name: str) -> DebuggerService:
    folder.mkdir()
    service, _recorder = _recording_service(folder, scenario_name)
    service.apply_command(
        _request(
            "finish", base_revision=service.revision, command=FinishAndReviewCommandV1()
        )
    )
    return service


def _open_replay(coordinator: RecordingDebuggerCoordinator, command_id: str) -> str:
    result = coordinator.apply_command(
        _request(
            command_id,
            base_revision=coordinator.service.revision,
            command=ReviewReplayCommandV1(),
        )
    )
    assert isinstance(result.payload, CommandResponseV2)
    assert result.payload.replay_url is not None
    return result.payload.replay_url


def _frame_status(server: DebuggerHTTPServer, port: int) -> int:
    connection = HTTPConnection("127.0.0.1", port)
    try:
        connection.request(
            "GET",
            "/api/frame",
            headers={"X-MARL-Debugger-Token": server.capability_token},
        )
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("replay_port", "port"),
    [(-1, 0), (65536, 0), (True, 0), ("8767", 0), (8767, 8767)],
)
def test_battle_client_refuses_bad_replay_ports_before_setup(
    monkeypatch: pytest.MonkeyPatch, replay_port: object, port: int
) -> None:
    def forbidden_setup(**_kwargs: object) -> None:
        raise AssertionError("A bad replay_port must fail before setup.")

    monkeypatch.setattr(launch, "prepare_battle_setup", forbidden_setup)
    with pytest.raises(ValueError, match="replay_port"):
        launch.battle_client(map_id=0, port=port, replay_port=cast(int, replay_port))


def test_battle_client_passes_the_replay_port_to_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launches: list[dict[str, object]] = []

    def serve(_setup: object, **kwargs: object) -> int:
        launches.append(kwargs)
        return 0

    def prepare(**_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(launch, "prepare_battle_setup", prepare)
    monkeypatch.setattr(launch, "_launch_setup", serve)
    launch.battle_client(
        map_id=0, port=8765, replay_port=8767, output_dir=tmp_path / "records"
    )
    assert [(item["port"], item["replay_port"]) for item in launches] == [(8765, 8767)]


def test_fixed_replay_port_serves_open_replay_there(tmp_path: Path) -> None:
    port = _free_port()
    service = _finished_service(tmp_path / "game", "arena_5v5")
    coordinator = RecordingDebuggerCoordinator(
        service, asset_root=_ASSET_ROOT, replay_port=port
    )
    try:
        url = _open_replay(coordinator, "open")
        server = coordinator._viewer_server
        assert server is not None
        assert server.server_port == port
        assert url == f"http://127.0.0.1:{port}/#token={server.capability_token}"
        assert _frame_status(server, port) == 200
    finally:
        coordinator.close()
        service.close()


def test_a_different_replay_replaces_the_viewer_on_the_same_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _free_port()
    service = _finished_service(tmp_path / "first", "arena_5v5")
    other = _finished_service(tmp_path / "second", "basic_support")
    coordinator = RecordingDebuggerCoordinator(
        service, asset_root=_ASSET_ROOT, replay_port=port
    )
    try:
        _open_replay(coordinator, "open-first")
        old_server, old_thread = coordinator._viewer_server, coordinator._viewer_thread
        assert old_server is not None and old_thread is not None
        next_viewer = other._build_replay_handoff()
        original = DebuggerService._build_replay_handoff

        def next_handoff(owner: DebuggerService) -> ReplayViewerService:
            return next_viewer if owner is service else original(owner)

        monkeypatch.setattr(DebuggerService, "_build_replay_handoff", next_handoff)
        url = _open_replay(coordinator, "open-second")
        new_server = coordinator._viewer_server
        assert new_server is not None and new_server is not old_server
        assert not old_thread.is_alive()
        assert old_server.socket.fileno() == -1
        assert new_server.server_port == port
        assert new_server.capability_token != old_server.capability_token
        assert url == f"http://127.0.0.1:{port}/#token={new_server.capability_token}"
        assert _frame_status(new_server, port) == 200
        assert coordinator._viewer_replay_id == next_viewer.replay_id
    finally:
        coordinator.close()
        service.close()
        other.close()
