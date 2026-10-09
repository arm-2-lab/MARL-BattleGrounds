"""Check the Rune Scrubber timeline marks the Replay Viewer serves per audience.

ReplayViewerService.timeline_marks returns one ReplayTimelineMarksV1 for the
current audience with exactly the fields schema_version, marks_kind, audience,
recipient_public_agent_id, final_frame_index and ticks (no fight list: nothing
groups ticks into fights any more). Oracle View marks equal the recorded falls,
Ultimates and returns; Agent POV marks equal exactly what that recipient's
Agent POV visual events show (SharedObs and NoSharedObs),
so another team's unseen returns are left out; the bytes are cached per
audience. replay_http_binding offers the marks, so real launches (the
standalone Replay Viewer and the BattleClient's Open Replay child) serve them.
GET /api/replay/timeline-marks needs the launch token, answers with the current
audience's JSON, and stays 404 when a binding offers no marks.

ReplayViewerService.prepare_agent_pov builds every agent's Agent POV work up
front: in a SharedObs replay every frame's source rows for every agent, and for
every agent its marks (byte for byte like a request) and its Champion Streaks
and Herald Cards tables to the last frame, plus Oracle View's tables; a second
call changes nothing, and a later Agent POV request returns the cached bytes. In a
NoSharedObs replay it also builds every agent's projection. A piece that fails
its validation makes it return False and leaves the rest on demand. A server
made with page_ready (create_server) holds a GET of its page until the event
is set while its API answers at once, and wait_for_quiet_requests waits while
any request of the process runs, or returns False once its longest wait runs
out. prepare_agent_pov calls its between before every small piece (at least
one per source row) with unchanged results. The BattleClient's Open Replay
(RecordingDebuggerCoordinator._open_viewer) returns its child's address at
once, holds that child's page while a background thread prepares the service
with a _GiveWayToRequests pause between pieces, then lets the page open; a
second Open Replay reuses the child. That pause waits for quiet requests for at
most WAIT_SECONDS, then lets the prepare work for WORK_SECONDS without waiting.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Condition, Event, Thread
from typing import Any, cast

import pytest

from marl_battlegrounds._battle_client import recording_coordinator
from marl_battlegrounds._battle_client.recording_coordinator import (
    RecordingDebuggerCoordinator,
)
from marl_battlegrounds.evaluation.models import (
    AbilityActivatedEventV1,
    AgentDiedEventV1,
    AgentRespawnedEventV1,
)
from marl_battlegrounds.evaluation.replay_io import load_replay
from marl_battlegrounds.rendering.authorized_presentation import (
    AgentPovVisualIncomingAgentRespawnedEventV1,
    ReplayIncomingAbilityActivatedEventV1,
    ReplayIncomingAgentDiedEventV1,
)
from marl_battlegrounds.viewer import server as server_module
from marl_battlegrounds.viewer.launch import replay_http_binding
from marl_battlegrounds.viewer.presentation_protocol import (
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
)
from marl_battlegrounds.viewer.replay_protocol import ReplayCommandRequestV1
from marl_battlegrounds.viewer.replay_service import ReplayViewerService
from marl_battlegrounds.viewer.server import create_server, wait_for_quiet_requests

_ROOT = Path(__file__).resolve().parents[2]
_SHARED_REPLAY = (
    _ROOT
    / "tests"
    / "fixtures"
    / "historical_scenario_v4"
    / "scenario_1.marlbg-replay.json"
)
_NO_SHARED_REPLAY = (
    _ROOT / "examples" / "replays" / "v1" / "death-respawn-shield.marlbg-replay.json"
)
_ASSET_ROOT = _ROOT / "web" / "visual_debugger"
_TOKEN = "timeline-marks-token"
_ROUTE = "/api/replay/timeline-marks"


def _next(service: ReplayViewerService, index: int) -> None:
    service.apply_command(
        ReplayCommandRequestV1.model_validate(
            {
                "client_id": "marks",
                "command_id": f"next-{index}",
                "base_revision": service.revision,
                "command": {"command_type": "next_frame"},
            }
        )
    )


def _marks(service: ReplayViewerService) -> dict[str, Any]:
    return json.loads(service.timeline_marks())


def _command(service: ReplayViewerService, name: str, command: dict[str, Any]) -> None:
    service.apply_command(
        ReplayCommandRequestV1.model_validate(
            {
                "client_id": "marks",
                "command_id": name,
                "base_revision": service.revision,
                "command": command,
            }
        )
    )


def _tick_ids(
    marks: dict[str, Any],
) -> dict[int, tuple[list[str], list[str], list[str]]]:
    return {
        tick["frame_index"]: (
            sorted(row["public_agent_id"] for row in tick["falls"]),
            [row["public_agent_id"] for row in tick["ultimates"]],
            sorted(row["public_agent_id"] for row in tick["returns"]),
        )
        for tick in marks["ticks"]
    }


@pytest.mark.parametrize("replay_path", (_SHARED_REPLAY, _NO_SHARED_REPLAY))
def test_oracle_marks_equal_the_recorded_events(replay_path: Path) -> None:
    loaded = load_replay(replay_path)
    replay = loaded.replay
    roster = replay.header.context.roster
    service = ReplayViewerService(loaded, view_mode="researcher")
    marks = _marks(service)
    assert set(marks) == {
        "schema_version",
        "marks_kind",
        "audience",
        "recipient_public_agent_id",
        "final_frame_index",
        "ticks",
    }
    assert marks["audience"] == "oracle"
    assert marks["recipient_public_agent_id"] is None
    assert marks["final_frame_index"] == len(replay.transitions)
    expected: dict[int, tuple[list[str], list[str], list[str]]] = {}
    for index, transition in enumerate(replay.transitions, start=1):
        falls = sorted(
            roster[event.recipient_global_slot].public_agent_id
            for event in transition.events
            if isinstance(event, AgentDiedEventV1)
        )
        ultimates = [
            roster[event.source_global_slot].public_agent_id
            for event in transition.events
            if isinstance(event, AbilityActivatedEventV1)
            and event.ability_component == "ultimate"
        ]
        returns = sorted(
            roster[event.agent_global_slot].public_agent_id
            for event in transition.events
            if isinstance(event, AgentRespawnedEventV1)
        )
        if falls or ultimates or returns:
            expected[index] = (falls, ultimates, returns)
    assert expected
    assert _tick_ids(marks) == expected
    assert service.timeline_marks() is service.timeline_marks()


def _seen_in_presentations(
    replay_path: Path, recipient: int
) -> dict[int, tuple[list[str], list[str], list[str]]]:
    loaded = load_replay(replay_path)
    service = ReplayViewerService(loaded, view_mode="pov", pov_global_slot=recipient)
    seen: dict[int, tuple[list[str], list[str], list[str]]] = {}
    for index in range(1, len(loaded.replay.frames)):
        _next(service, index)
        payload = service.current_presentation().payload
        assert isinstance(
            payload,
            (
                ReplaySharedObsAuthorizedPresentationFrameV1,
                ReplayNoSharedObsAuthorizedPresentationFrameV1,
            ),
        )
        assert payload.visual_events is not None
        events = payload.visual_events.events
        falls = sorted(
            event.recipient_anchor.public_agent_id
            for event in events
            if isinstance(event, ReplayIncomingAgentDiedEventV1)
        )
        ultimates = [
            event.source_anchor.public_agent_id
            for event in events
            if isinstance(event, ReplayIncomingAbilityActivatedEventV1)
            and event.ability_component == "ultimate"
        ]
        returns = sorted(
            event.agent_anchor.public_agent_id
            for event in events
            if isinstance(event, AgentPovVisualIncomingAgentRespawnedEventV1)
        )
        if falls or ultimates or returns:
            seen[index] = (falls, ultimates, returns)
    return seen


@pytest.mark.parametrize(
    ("replay_path", "recipient"),
    (
        (_SHARED_REPLAY, 0),
        (_SHARED_REPLAY, 5),
        (_NO_SHARED_REPLAY, 0),
        (_NO_SHARED_REPLAY, 5),
    ),
)
def test_agent_pov_marks_equal_what_that_audience_saw(
    replay_path: Path, recipient: int
) -> None:
    loaded = load_replay(replay_path)
    service = ReplayViewerService(loaded, view_mode="pov", pov_global_slot=recipient)
    marks = _marks(service)
    assert marks["audience"] == "agent_pov"
    assert marks["recipient_public_agent_id"] == (
        loaded.replay.header.context.roster[recipient].public_agent_id
    )
    assert _tick_ids(marks) == _seen_in_presentations(replay_path, recipient)
    oracle = _marks(ReplayViewerService(loaded, view_mode="researcher"))
    # Agent POV never holds a mark that Oracle View lacks.
    for frame, (falls, ultimates, returns) in _tick_ids(marks).items():
        oracle_falls, oracle_ultimates, oracle_returns = _tick_ids(oracle)[frame]
        assert set(falls) <= set(oracle_falls)
        assert set(ultimates) <= set(oracle_ultimates)
        assert set(returns) <= set(oracle_returns)


def test_another_teams_unseen_returns_are_left_out() -> None:
    loaded = load_replay(_NO_SHARED_REPLAY)
    oracle = _tick_ids(_marks(ReplayViewerService(loaded, view_mode="researcher")))
    watcher = _tick_ids(
        _marks(ReplayViewerService(loaded, view_mode="pov", pov_global_slot=0))
    )
    returner = _tick_ids(
        _marks(ReplayViewerService(loaded, view_mode="pov", pov_global_slot=5))
    )
    return_frames = [frame for frame, marks in oracle.items() if marks[2]]
    assert return_frames
    assert all(frame in returner and returner[frame][2] for frame in return_frames)
    assert all(frame not in watcher or not watcher[frame][2] for frame in return_frames)


def _get(port: int, *, token: str | None, route: str = _ROUTE) -> tuple[int, bytes]:
    connection = HTTPConnection("127.0.0.1", port, timeout=None)
    headers = {} if token is None else {"X-MARL-Debugger-Token": token}
    connection.request("GET", route, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, payload


@pytest.mark.parametrize("offer_marks", (True, False))
def test_timeline_marks_route_needs_the_token_and_serves_the_audience(
    offer_marks: bool,
) -> None:
    service = ReplayViewerService(load_replay(_SHARED_REPLAY), view_mode="researcher")
    binding = replay_http_binding(service)
    assert binding.timeline_marks == service.timeline_marks
    if not offer_marks:
        binding = replace(binding, timeline_marks=None)
    server = create_server(
        service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=binding,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _ = _get(server.server_port, token=None)
        assert status == (
            HTTPStatus.UNAUTHORIZED if offer_marks else HTTPStatus.NOT_FOUND
        )
        status, payload = _get(server.server_port, token=_TOKEN)
        if offer_marks:
            assert status == HTTPStatus.OK
            assert json.loads(payload) == _marks(service)
            assert json.loads(payload)["audience"] == "oracle"
        else:
            assert status == HTTPStatus.NOT_FOUND
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    with pytest.raises(ValueError, match="timeline marks require"):
        replace(binding, timeline_marks=cast(Any, "not callable"))


def test_prepare_builds_every_agent_like_a_request() -> None:
    loaded = load_replay(_SHARED_REPLAY)
    service = ReplayViewerService(loaded, view_mode="researcher")
    pauses: list[None] = []
    assert service.prepare_agent_pov(between=lambda: pauses.append(None))
    slots = service._active_slots
    frame_count = len(loaded.replay.frames)
    assert len(pauses) > frame_count * len(slots)
    assert set(service._source_materials) == {
        (frame, slot) for frame in range(frame_count) for slot in slots
    }
    for table in (service._champion_streak_tables, service._match_moment_tables):
        assert set(table) == {None, *slots}
        assert all(len(row.reached) == frame_count for row in table.values())
    reference = ReplayViewerService(loaded, view_mode="researcher")
    _command(reference, "pov", {"command_type": "set_view", "view_mode": "pov"})
    for slot in slots:
        _command(
            reference,
            f"agent-{slot}",
            {"command_type": "set_pov_actor", "global_slot": slot},
        )
        assert reference._pov_global_slot == slot
        assert service._timeline_marks_bytes[("agent_pov", slot)] == (
            reference.timeline_marks()
        )
    built = dict(service._timeline_marks_bytes)
    assert service.prepare_agent_pov()
    assert service._timeline_marks_bytes == built
    _command(service, "pov", {"command_type": "set_view", "view_mode": "pov"})
    recipient = service._pov_global_slot
    assert service.timeline_marks() is built[("agent_pov", recipient)]


def test_no_shared_obs_prepare_builds_every_agents_projection() -> None:
    loaded = load_replay(_NO_SHARED_REPLAY)
    service = ReplayViewerService(loaded, view_mode="pov", pov_global_slot=0)
    assert set(service._pov_cache) == {0}
    assert service.prepare_agent_pov()
    slots = service._active_slots
    assert set(service._pov_cache) == set(slots)
    for slot in slots:
        assert service._timeline_marks_bytes[("agent_pov", slot)] == (
            ReplayViewerService(
                loaded, view_mode="pov", pov_global_slot=slot
            ).timeline_marks()
        )
        assert len(service._champion_streak_tables[slot].reached) == len(
            loaded.replay.frames
        )


def test_a_failing_prepare_leaves_the_rest_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ReplayViewerService(load_replay(_SHARED_REPLAY), view_mode="researcher")

    def broken(*_args: object, **_kwargs: object) -> bytes:
        raise ValueError("a recorded scene failed its validation")

    monkeypatch.setattr(service, "_agent_pov_marks_bytes", broken)
    assert not service.prepare_agent_pov()
    assert not service._agent_pov_prepared
    assert service._timeline_marks_bytes == {}


def test_a_held_page_waits_while_the_api_answers() -> None:
    service = ReplayViewerService(load_replay(_SHARED_REPLAY), view_mode="researcher")
    waiting = Event()

    class _Hold(Event):
        def wait(self, timeout: float | None = None) -> bool:
            waiting.set()
            return super().wait(timeout)

    hold = _Hold()
    server = create_server(
        service,
        asset_root=_ASSET_ROOT,
        port=0,
        capability_token=_TOKEN,
        coordinator=replay_http_binding(service),
        page_ready=hold,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    answers: list[int] = []
    page = Thread(
        target=lambda: answers.append(
            _get(server.server_port, token=None, route="/")[0]
        )
    )
    try:
        page.start()
        waiting.wait()
        assert _get(server.server_port, token=_TOKEN)[0] == HTTPStatus.OK
        # The page request is still inside the hold: it can answer only after set.
        assert answers == []
        hold.set()
        page.join()
        assert answers == [HTTPStatus.OK]
    finally:
        hold.set()
        server.shutdown()
        server.server_close()
        thread.join()


def test_quiet_wait_waits_while_a_request_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiting = Event()

    class _Activity(Condition):
        def wait(self, timeout: float | None = None) -> bool:
            waiting.set()
            return super().wait(timeout)

    monkeypatch.setattr(server_module, "_REQUEST_ACTIVITY", _Activity())
    done: list[bool] = []
    waiter = Thread(
        target=lambda: (wait_for_quiet_requests(quiet_seconds=0), done.append(True))
    )
    with server_module._counted_request():
        waiter.start()
        waiting.wait()
        # The waiter is inside the wait: it can return only after the request.
        assert done == []
    waiter.join()
    assert done == [True]
    with server_module._counted_request():
        assert not wait_for_quiet_requests(quiet_seconds=0, longest_wait_seconds=0)
    assert wait_for_quiet_requests(quiet_seconds=0, longest_wait_seconds=0)


def test_give_way_waits_then_opens_a_work_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    quiet = [False]
    waits: list[float | None] = []

    def fake_wait(
        quiet_seconds: float = 0.15, *, longest_wait_seconds: float | None = None
    ) -> bool:
        del quiet_seconds
        waits.append(longest_wait_seconds)
        return quiet[0]

    monkeypatch.setattr(recording_coordinator, "monotonic", lambda: clock[0])
    monkeypatch.setattr(recording_coordinator, "wait_for_quiet_requests", fake_wait)
    give_way = recording_coordinator._GiveWayToRequests()
    give_way()
    assert waits == [give_way.WAIT_SECONDS]
    # Requests kept coming, so the next pieces run for WORK_SECONDS unpaused.
    clock[0] = give_way.WORK_SECONDS / 2
    give_way()
    assert len(waits) == 1
    clock[0] = give_way.WORK_SECONDS
    quiet[0] = True
    give_way()
    give_way()
    assert len(waits) == 3


def test_open_replay_answers_at_once_and_holds_only_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ReplayViewerService(load_replay(_SHARED_REPLAY), view_mode="researcher")
    real_prepare = service.prepare_agent_pov
    release = Event()
    given: list[object] = []

    def held_prepare(*, between: Callable[[], None] | None = None) -> bool:
        given.append(between)
        release.wait()
        return real_prepare(between=between)

    monkeypatch.setattr(service, "prepare_agent_pov", held_prepare)
    coordinator = object.__new__(RecordingDebuggerCoordinator)
    coordinator._closed = False
    coordinator._viewer_server = None
    coordinator._viewer_thread = None
    coordinator._viewer_replay_id = None
    coordinator._replay_port = 0
    coordinator._asset_root = _ASSET_ROOT
    # Returns while the prepare is still held.
    url = coordinator._open_viewer(service)
    server = coordinator._viewer_server
    thread = coordinator._viewer_thread
    assert server is not None
    assert thread is not None
    try:
        page_ready = server.page_ready
        assert page_ready is not None
        assert not page_ready.is_set()
        assert url.startswith(server.expected_origin)
        status, _ = _get(server.server_port, token=server.capability_token)
        assert status == HTTPStatus.OK
        release.set()
        page_ready.wait()
        assert [type(pause) for pause in given] == [
            recording_coordinator._GiveWayToRequests
        ]
        assert service._agent_pov_prepared
        assert {("agent_pov", slot) for slot in service._active_slots} <= set(
            service._timeline_marks_bytes
        )
        assert _get(server.server_port, token=None, route="/")[0] == HTTPStatus.OK
        assert coordinator._open_viewer(service) == url
    finally:
        release.set()
        RecordingDebuggerCoordinator._stop_viewer(server, thread)
