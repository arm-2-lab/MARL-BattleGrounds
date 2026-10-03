"""Check the Saga Chronicle timeline marks the Replay Viewer serves per audience.

build_replay_fight_chapters_v1: a fall or a damaging Ultimate (Charge,
Freezing Trap, Crippling Poison) opens a fight; Burst and Holy Word: Salvation
only extend an open fight; a fight grows across at most 2 quiet ticks and a
third quiet tick starts a new one; returns never touch a fight; counts cover
the whole span; unordered ticks are refused. ReplayViewerService.timeline_marks
returns one ReplayTimelineMarksV1 for the current audience: Oracle View marks
equal the recorded falls, Ultimates and returns; Agent POV marks equal exactly
what that recipient's Agent POV visual events show (SharedObs and NoSharedObs),
so another team's unseen returns are left out; the bytes are cached per
audience. replay_http_binding offers the marks, so real launches (the
standalone Replay Viewer and the BattleClient's Open Replay child) serve them.
GET /api/replay/timeline-marks needs the launch token, answers with the current
audience's JSON, and stays 404 when a binding offers no marks.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from dataclasses import replace
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Any, cast

import pytest

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
from marl_battlegrounds.viewer.launch import replay_http_binding
from marl_battlegrounds.viewer.presentation_protocol import (
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
)
from marl_battlegrounds.viewer.replay_protocol import ReplayCommandRequestV1
from marl_battlegrounds.viewer.replay_service import (
    FIGHT_MERGE_QUIET_TICKS,
    ReplayTimelineAgentMarkV1,
    ReplayTimelineFallMarkV1,
    ReplayTimelineTickMarksV1,
    ReplayTimelineUltimateMarkV1,
    ReplayViewerService,
    build_replay_fight_chapters_v1,
)
from marl_battlegrounds.viewer.server import create_server

_ROOT = Path(__file__).resolve().parents[1]
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


def _agent(slot: int, class_id: int = 1) -> dict[str, Any]:
    return {
        "public_agent_id": f"agent-{slot}",
        "team_id": 1 if slot < 5 else 2,
        "class_id": class_id,
    }


def _tick(
    frame: int,
    *,
    falls: tuple[int, ...] = (),
    ultimates: tuple[tuple[int, int], ...] = (),
    returns: tuple[int, ...] = (),
) -> ReplayTimelineTickMarksV1:
    return ReplayTimelineTickMarksV1(
        frame_index=frame,
        falls=tuple(
            ReplayTimelineFallMarkV1(**_agent(slot), home_fall=False) for slot in falls
        ),
        ultimates=tuple(
            ReplayTimelineUltimateMarkV1(
                **_agent(slot, class_id), damaging=class_id in (2, 3, 4)
            )
            for slot, class_id in ultimates
        ),
        returns=tuple(ReplayTimelineAgentMarkV1(**_agent(slot)) for slot in returns),
    )


def _spans(
    ticks: tuple[ReplayTimelineTickMarksV1, ...],
) -> list[tuple[int, int, int, int]]:
    return [
        (
            fight.first_frame_index,
            fight.last_frame_index,
            fight.ultimate_count,
            fight.fallen_count,
        )
        for fight in build_replay_fight_chapters_v1(ticks)
    ]


def test_fight_rule_opens_extends_and_merges_across_two_quiet_ticks() -> None:
    assert FIGHT_MERGE_QUIET_TICKS == 2
    # A fall at 3 and another after 2 quiet ticks (6) stay one fight; after 3
    # quiet ticks (10) a new fight starts.
    assert _spans(
        (_tick(3, falls=(1,)), _tick(6, falls=(6,)), _tick(10, falls=(2,)))
    ) == [
        (3, 6, 0, 2),
        (10, 10, 0, 1),
    ]
    # Charge (Warrior), Freezing Trap (Hunter) and Crippling Poison (Rogue) open.
    for class_id in (2, 3, 4):
        assert _spans((_tick(5, ultimates=((1, class_id),)),)) == [(5, 5, 1, 0)]
    # Burst (Mage) and Holy Word: Salvation (Priest) never open a fight...
    assert _spans((_tick(4, ultimates=((0, 1),)), _tick(5, ultimates=((2, 5),)))) == []
    # ...but extend an open one within 2 quiet ticks, and not beyond.
    assert _spans(
        (
            _tick(3, falls=(1,)),
            _tick(6, ultimates=((0, 1),)),
            _tick(9, ultimates=((2, 5),)),
            _tick(13, ultimates=((7, 5),)),
        )
    ) == [(3, 9, 2, 1)]
    # Returns never open or extend a fight, but a later fall can merge over them.
    assert _spans((_tick(2, returns=(1,)), _tick(4, returns=(6,)))) == []
    assert _spans(
        (_tick(2, falls=(1,)), _tick(4, returns=(6,)), _tick(5, falls=(7,)))
    ) == [(2, 5, 0, 2)]
    with pytest.raises(ValueError, match="increasing frame order"):
        build_replay_fight_chapters_v1((_tick(5, falls=(1,)), _tick(5, falls=(2,))))


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


def _get(port: int, *, token: str | None) -> tuple[int, bytes]:
    connection = HTTPConnection("127.0.0.1", port, timeout=None)
    headers = {} if token is None else {"X-MARL-Debugger-Token": token}
    connection.request("GET", _ROUTE, headers=headers)
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
