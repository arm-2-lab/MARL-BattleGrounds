"""Check the match facts the beat clock holds back: old scores, returns, home falls.

MatchSummaryV1.previous_scores is each team's score before the tick (equal to
scores on frame zero and on ticks with no score); returned_agent_ids names the
champions that return on the tick; MatchDeathV1.home_fall is True only for a
death that started in the victim's own Red Zone and so scored double. The rule
has one owner, evaluation.events.team_deathmatch_home_fall_slots_v1, used by
the score check itself: two ordinary falls in one tick, one home fall plus one
ordinary fall, a Team B home fall and the swapped spawn ends all agree with the
recorded score edges. Without the start frame, previous scores come from the
score-change events and home_fall stays False. Old payloads without the new
fields still validate; impossible values are refused. In replay presentations
every audience gets the same global match facts; the Oracle agent_died rows
carry home_fall; an Agent POV agent_died row, and so its flag, exists only for
a fall that audience saw, and Agent POV visual events hold only the returns
that team saw while the global summary names every return.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from tests.evaluation.test_evaluation_events import (
    _FALSE_10,
    _frame,
    _neutral_facts,
    _red_zone_context,
    _replace_item,
    _snapshot,
)

import marl_battlegrounds.rendering.authorized_presentation as authorized_module
import marl_battlegrounds.viewer.presentation as presentation_module
from marl_battlegrounds.evaluation.events import (
    decode_evaluation_events_v1,
    team_deathmatch_home_fall_slots_v1,
)
from marl_battlegrounds.evaluation.models import (
    AgentRespawnedEventV1,
    EvaluationEpisodeContext,
    EvaluationEventV1,
    EvaluationFrameV1,
)
from marl_battlegrounds.evaluation.replay_io import load_replay
from marl_battlegrounds.rendering.authorized_presentation import (
    AgentPovVisualIncomingAgentRespawnedEventV1,
    ReplayIncomingAgentAnchorV1,
    ReplayIncomingAgentDiedEventV1,
)
from marl_battlegrounds.rendering.scene import AgentDiedEventV2, VisualEventBatchV2
from marl_battlegrounds.viewer.match_summary import (
    MatchDeathV1,
    MatchSummaryV1,
    build_match_summary_v1,
)
from marl_battlegrounds.viewer.presentation_protocol import (
    ReplayOracleAuthorizedPresentationFrameV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
)
from marl_battlegrounds.viewer.replay_protocol import ReplayCommandRequestV1
from marl_battlegrounds.viewer.replay_service import ReplayViewerService

_SCENARIO_REPLAY = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "historical_scenario_v4"
    / "scenario_1.marlbg-replay.json"
)
_POSITIONS = tuple((float(slot), 0.0) for slot in range(10))


def _death_tick(
    context: EvaluationEpisodeContext,
    victims: dict[int, float],
    scores: tuple[int, int],
) -> tuple[EvaluationFrameV1, EvaluationFrameV1, tuple[EvaluationEventV1, ...]]:
    # Width 20: Team A's strip is x <= 5 and Team B's x >= 15 (swapped: reverse).
    facts = _neutral_facts()
    dead = _FALSE_10
    positions = _POSITIONS
    alive = (True,) * 10
    for slot, x in victims.items():
        dead = _replace_item(dead, slot, True)
        positions = _replace_item(positions, slot, (x, 0.0))
        alive = _replace_item(alive, slot, False)
    facts = facts.model_copy(
        update={
            "death_facts": facts.death_facts.model_copy(
                update={"is_newly_dead_by_recipient": dead}
            )
        }
    )
    start = _frame(3, 7, _snapshot(agent_positions=positions))
    successor = _frame(
        4,
        8,
        _snapshot(
            agent_positions=positions,
            alive_mask=alive,
            team_deathmatch_scores=scores,
        ),
    )
    return (
        start,
        successor,
        decode_evaluation_events_v1(context, start, facts, successor),
    )


@pytest.mark.parametrize(
    ("swap_banks", "victims", "scores", "home"),
    (
        pytest.param(False, {0: 8.0, 1: 9.0}, (0, 2), {}, id="two-ordinary-falls"),
        pytest.param(
            False, {0: 4.0, 1: 8.0}, (0, 3), {0}, id="one-home-one-ordinary-fall"
        ),
        pytest.param(False, {0: 5.0}, (0, 2), {0}, id="home-on-the-boundary"),
        pytest.param(False, {5: 16.0, 6: 14.0}, (3, 0), {5}, id="team-b-home-fall"),
        pytest.param(
            True, {0: 16.0, 1: 4.0}, (0, 3), {0}, id="swapped-ends-home-and-invader"
        ),
        pytest.param(True, {5: 4.0}, (2, 0), {5}, id="swapped-ends-team-b-home"),
    ),
)
def test_home_falls_match_the_score_rule_on_both_spawn_ends(
    swap_banks: bool,
    victims: dict[int, float],
    scores: tuple[int, int],
    home: set[int],
) -> None:
    context = _red_zone_context("v4", 5.0, swap_banks)
    start, successor, events = _death_tick(context, victims, scores)
    summary = build_match_summary_v1(
        context,
        successor,
        events,
        transition_start_frame=start,
    )
    assert summary.previous_scores == (0, 0)
    assert summary.scores == scores
    assert {death.public_agent_id: death.home_fall for death in summary.deaths} == {
        context.roster[slot].public_agent_id: slot in home for slot in sorted(victims)
    }
    assert summary.returned_agent_ids == ()
    assert team_deathmatch_home_fall_slots_v1(
        context,
        start.snapshot.agent_positions,
        tuple(slot in victims for slot in range(10)),
    ) == frozenset(home)
    # Without the start frame the score edge still gives the previous scores,
    # while the start positions (and so home falls) stay unknown.
    unknown = build_match_summary_v1(context, successor, events)
    assert unknown.previous_scores == (0, 0)
    assert not any(death.home_fall for death in unknown.deaths)


@pytest.mark.parametrize(("version", "depth"), (("v1", 0.0), ("v3", 0.0), ("v4", 0.0)))
def test_records_without_an_active_red_zone_never_mark_a_home_fall(
    version: str, depth: float
) -> None:
    context = _red_zone_context(version, depth, False)
    start, successor, events = _death_tick(context, {0: 1.0, 1: 2.0}, (0, 2))
    summary = build_match_summary_v1(
        context,
        successor,
        events,
        transition_start_frame=start,
    )
    assert [death.home_fall for death in summary.deaths] == [False, False]
    assert summary.previous_scores == (0, 0)


def test_start_frame_must_be_the_frame_just_before() -> None:
    context = _red_zone_context("v4", 5.0, False)
    _start, successor, events = _death_tick(context, {0: 8.0}, (0, 1))
    with pytest.raises(ValueError, match="frame before it"):
        build_match_summary_v1(
            context,
            successor,
            events,
            transition_start_frame=successor,
        )


def test_old_payloads_validate_and_impossible_values_are_refused() -> None:
    old = MatchSummaryV1.model_validate(
        {
            "schema_version": 1,
            "episode_id": "old",
            "source_frame_index": 3,
            "simulator_step_count": 3,
            "task_mode": 1,
            "score_threshold": 20,
            "scores": [2, 1],
            "outcome": "in_progress",
            "teams": [
                {
                    "team_id": 1,
                    "display_name": "A",
                    "policy_ids": ["a"],
                    "checkpoint_digests": [],
                },
                {
                    "team_id": 2,
                    "display_name": "B",
                    "policy_ids": ["b"],
                    "checkpoint_digests": [],
                },
            ],
            "deaths": [{"public_agent_id": "x", "team_id": 2, "class_id": 1}],
        },
        strict=False,
    )
    assert old.previous_scores is None
    assert old.returned_agent_ids == ()
    assert old.deaths[0].home_fall is False
    assert MatchDeathV1(public_agent_id="x", team_id=1, class_id=2).home_fall is False
    with pytest.raises(ValidationError, match="cannot go down"):
        old.model_validate(
            {**old.model_dump(), "previous_scores": (3, 1)}, strict=False
        )
    with pytest.raises(ValidationError, match="each returning agent once"):
        old.model_validate(
            {**old.model_dump(), "returned_agent_ids": ("a", "a")}, strict=False
        )
    with pytest.raises(ValidationError, match="cannot claim TDM scores"):
        old.model_validate(
            {
                **old.model_dump(),
                "task_mode": 0,
                "score_threshold": 0,
                "scores": (0, 0),
                "outcome": "not_applicable",
                "deaths": (
                    {
                        "public_agent_id": "x",
                        "team_id": 2,
                        "class_id": 1,
                        "home_fall": True,
                    },
                ),
            },
            strict=False,
        )
    anchor = ReplayIncomingAgentAnchorV1(
        presentation_key="oracle_" + "0" * 64,
        public_agent_id="x",
        phase="successor",
        position=(1.0, 1.0),
    )
    with pytest.raises(ValueError, match="home_fall must be an exact Python bool"):
        ReplayIncomingAgentDiedEventV1(
            event_id="e",
            ordinal=0,
            phase_rank=90,
            event_kind="agent_died",
            recipient_anchor=anchor,
            home_fall=cast(bool, 1),
        )


def _step(service: ReplayViewerService, index: int) -> None:
    service.apply_command(
        ReplayCommandRequestV1.model_validate(
            {
                "client_id": "beats",
                "command_id": f"next-{index}",
                "base_revision": service.revision,
                "command": {"command_type": "next_frame"},
            }
        )
    )


def _oracle(service: ReplayViewerService) -> ReplayOracleAuthorizedPresentationFrameV1:
    payload = service.current_presentation().payload
    assert isinstance(payload, ReplayOracleAuthorizedPresentationFrameV1)
    return payload


def _shared_pov(
    service: ReplayViewerService,
) -> ReplaySharedObsAuthorizedPresentationFrameV1:
    payload = service.current_presentation().payload
    assert isinstance(payload, ReplaySharedObsAuthorizedPresentationFrameV1)
    return payload


def test_replay_presentations_carry_old_scores_and_returns_for_all() -> None:
    loaded = load_replay(_SCENARIO_REPLAY)
    replay = loaded.replay
    context = replay.header.context
    oracle = ReplayViewerService(loaded, view_mode="researcher")
    pov = ReplayViewerService(loaded, view_mode="pov", pov_global_slot=0)
    returns_seen = 0
    for index in range(1, len(replay.frames)):
        _step(oracle, index)
        _step(pov, index)
        events = replay.transitions[index - 1].events
        expected_returns = tuple(
            context.roster[slot].public_agent_id
            for slot in sorted(
                {
                    event.agent_global_slot
                    for event in events
                    if isinstance(event, AgentRespawnedEventV1)
                }
            )
        )
        returns_seen += len(expected_returns)
        for match in (_oracle(oracle).match_summary, _shared_pov(pov).match_summary):
            assert match is not None
            assert match.previous_scores == (
                replay.frames[index - 1].snapshot.team_deathmatch_scores
            )
            assert match.scores == replay.frames[index].snapshot.team_deathmatch_scores
            assert match.returned_agent_ids == expected_returns
    assert returns_seen > 0


def test_agent_died_rows_carry_home_fall_and_agent_pov_only_for_seen_falls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This historical record has no Red Zone rule, so every fall is ordinary;
    # force every death to count as a home fall to see the flag travel.
    def every_death(
        _context: EvaluationEpisodeContext, batch: VisualEventBatchV2
    ) -> frozenset[int]:
        return frozenset(
            event.recipient_global_slot
            for event in batch.events
            if isinstance(event, AgentDiedEventV2)
        )

    monkeypatch.setattr(
        authorized_module, "visual_batch_home_fall_slots_v1", every_death
    )
    monkeypatch.setattr(
        presentation_module, "visual_batch_home_fall_slots_v1", every_death
    )
    loaded = load_replay(_SCENARIO_REPLAY)
    replay = loaded.replay
    oracle = ReplayViewerService(loaded, view_mode="researcher")
    oracle_falls: set[str] = set()
    for index in range(1, len(replay.frames)):
        _step(oracle, index)
        payload = _oracle(oracle)
        assert payload.latest_events is not None
        assert payload.match_summary is not None
        died = [
            event
            for event in payload.latest_events.events
            if isinstance(event, ReplayIncomingAgentDiedEventV1)
        ]
        assert all(event.home_fall for event in died)
        oracle_falls |= {event.recipient_anchor.public_agent_id for event in died}
        # The match summary keeps its own rule owner: no Red Zone, no home fall.
        assert not any(death.home_fall for death in payload.match_summary.deaths)
    assert oracle_falls
    for recipient in (0, 5):
        pov = ReplayViewerService(loaded, view_mode="pov", pov_global_slot=recipient)
        seen_falls: set[str] = set()
        seen_returns: set[str] = set()
        global_returns: set[str] = set()
        for index in range(1, len(replay.frames)):
            _step(pov, index)
            payload = _shared_pov(pov)
            assert payload.visual_events is not None
            assert payload.match_summary is not None
            events = payload.visual_events.events
            died = [
                event
                for event in events
                if isinstance(event, ReplayIncomingAgentDiedEventV1)
            ]
            # Rows exist only for seen falls, and only those rows carry the flag.
            assert all(event.home_fall for event in died)
            assert all(
                isinstance(event, ReplayIncomingAgentDiedEventV1)
                for event in events
                if hasattr(event, "home_fall")
            )
            seen_falls |= {event.recipient_anchor.public_agent_id for event in died}
            seen_returns |= {
                event.agent_anchor.public_agent_id
                for event in events
                if isinstance(event, AgentPovVisualIncomingAgentRespawnedEventV1)
            }
            global_returns |= set(payload.match_summary.returned_agent_ids)
        assert seen_falls <= oracle_falls
        # Each team sees only its own returns here; the global summary has all.
        team = replay.header.context.roster[recipient].configured_team_id
        assert seen_returns == {
            row.public_agent_id
            for row in replay.header.context.roster
            if row.public_agent_id in global_returns and row.configured_team_id == team
        }
        assert seen_returns < global_returns
