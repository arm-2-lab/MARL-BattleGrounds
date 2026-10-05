"""Check the Momentum switch's host fact: each team's score 10 ticks ago.

MatchSummaryV1.scores_ten_ticks_ago is the [Team A, Team B] score at the frame
MOMENTUM_TICKS (10) frames before the shown one, or at frame zero in a younger
game. The host owns this history; the browser keeps none. Contracts checked:

- The record: old payloads without the field read None; a value above the
  score before this tick, or a combat diagnostic claiming points, is refused.
- build_match_summary_v1 copies the caller's value, keeps None by default and
  refuses a frame-zero value other than the frame's own scores. Over a
  recorded game, the Replay Viewer rule (frames[max(0, i - 10)]) is accepted on
  every frame.
- The live session (DebuggerSession.recent_scores): from the Vision Moment's
  "fall" start, the staged Charge scores on tick 1; that point is recent (the
  score 10 ticks ago stays 2) through frame 10 and stops being recent on frame
  11. The tail never holds more than 10 frames; Reset empties it; a tail that
  is too long, shows a falling score or has the wrong shape is refused.
- Live presentations: Oracle View, SharedObs Agent POV and NoSharedObs Agent
  POV all carry the session's value in match_summary, so every view shows the
  same public fact.
- Replay Viewer presentations: a replay saved from that live game, shown in
  Oracle View and Agent POV, carries the recorded scores of frame
  max(0, i - 10) on every frame i, before and after a seek, so the Replay
  Viewer and the live service give the same fact.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import pytest
from pydantic import ValidationError
from scripts.dev.visual_debugger.scenarios import vision_moment_scenario
from tests.test_evaluation_pov import _runtime_provenance
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client.control import (
    create_session,
    make_neutral_joint_action,
    reset_session,
    submit_joint_action,
)
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation.replay_io import LoadedReplay, load_replay
from marl_battlegrounds.evaluation.replay_v4 import build_replay_v4
from marl_battlegrounds.viewer.match_summary import (
    MOMENTUM_TICKS,
    MatchSummaryV1,
    build_match_summary_v1,
)
from marl_battlegrounds.viewer.replay_protocol import (
    ReplayAbsoluteSeekCommandV1,
    ReplayCommandRequestV1,
)
from marl_battlegrounds.viewer.replay_service import ReplayViewerService

_SCENARIO_REPLAY = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "historical_scenario_v4"
    / "scenario_1.marlbg-replay.json"
)


def _session(mode: str) -> DebuggerSession:
    return create_session(
        vision_moment_scenario("fall"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="manual",
        team_b_controller="manual",
        execution_information_mode=mode,  # pyright: ignore[reportArgumentType]
        manual_slots=tuple(range(10)),
        slot_labels=("Manual",) * 10,
    )


def _charge(session: DebuggerSession) -> DebuggerSession:
    # The Warrior (slot 1) Charges the Team B Mage (target 6), who falls.
    zeros = jnp.zeros((10,), dtype=jnp.int32)
    action = Action(
        move=zeros,
        select_target=zeros.at[1].set(6),
        use_ultimate=zeros.at[1].set(1),
    )
    return submit_joint_action(
        session, action, submission_kind="interactive", report_actor_slots=()
    )


def _still(session: DebuggerSession) -> DebuggerSession:
    return submit_joint_action(
        session,
        make_neutral_joint_action(),
        submission_kind="interactive",
        report_actor_slots=(),
    )


@pytest.fixture(scope="module")
def fall_game() -> tuple[DebuggerSession, ...]:
    sessions = [_session("shared_obs")]
    sessions.append(_charge(sessions[0]))
    for _ in range(MOMENTUM_TICKS):
        sessions.append(_still(sessions[-1]))
    return tuple(sessions)


def _summary(session: DebuggerSession, view_mode: str) -> MatchSummaryV1:
    service = DebuggerService(
        session,
        view_mode=view_mode,  # pyright: ignore[reportArgumentType]
        preset="analysis",
        include_stress=False,
    )
    try:
        payload = service.current_presentation().payload
    finally:
        service.close()
    summary = getattr(payload, "match_summary", None)
    assert isinstance(summary, MatchSummaryV1)
    return summary


def test_match_summary_carries_the_field_and_refuses_impossible_values() -> None:
    loaded = load_replay(_SCENARIO_REPLAY)
    context = loaded.replay.header.context
    frames = loaded.replay.frames
    transitions = loaded.replay.transitions
    first = build_match_summary_v1(context, frames[0])
    assert first.scores_ten_ticks_ago is None
    with pytest.raises(ValueError, match="on frame zero"):
        build_match_summary_v1(context, frames[0], scores_ten_ticks_ago=(0, 0))
    scores = [tuple(frame.snapshot.team_deathmatch_scores) for frame in frames]
    assert len(set(scores)) > 1
    summary = first
    for index, frame in enumerate(frames):
        earlier = frames[max(0, index - MOMENTUM_TICKS)].snapshot
        summary = build_match_summary_v1(
            context,
            frame,
            () if index == 0 else transitions[index - 1].events,
            transition_start_frame=None if index == 0 else frames[index - 1],
            scores_ten_ticks_ago=earlier.team_deathmatch_scores,
        )
        assert summary.scores_ten_ticks_ago == scores[max(0, index - MOMENTUM_TICKS)]
    old = MatchSummaryV1.model_validate(
        {
            key: value
            for key, value in summary.model_dump().items()
            if key != "scores_ten_ticks_ago"
        },
        strict=False,
    )
    assert old.scores_ten_ticks_ago is None
    above = (summary.scores[0] + 1, summary.scores[1])
    with pytest.raises(ValidationError, match="over ten ticks"):
        MatchSummaryV1.model_validate(
            {**summary.model_dump(), "scores_ten_ticks_ago": above}, strict=False
        )
    with pytest.raises(ValidationError, match="over ten ticks"):
        MatchSummaryV1.model_validate(
            {
                **summary.model_dump(),
                "previous_scores": (0, 0),
                "scores_ten_ticks_ago": (1, 0),
            },
            strict=False,
        )
    # A diagnostic's scores are (0, 0), so any point ten ticks ago is above them.
    with pytest.raises(ValidationError, match="over ten ticks"):
        MatchSummaryV1.model_validate(
            {
                **summary.model_dump(),
                "task_mode": 0,
                "score_threshold": 0,
                "scores": (0, 0),
                "previous_scores": None,
                "outcome": "not_applicable",
                "deaths": (),
                "scores_ten_ticks_ago": (1, 0),
            },
            strict=False,
        )


def test_live_session_keeps_ten_ticks_of_scores_and_reset_clears_them(
    fall_game: tuple[DebuggerSession, ...],
) -> None:
    start, scored, *_ = fall_game
    assert start.current_evaluation_frame.snapshot.team_deathmatch_scores == (2, 4)
    assert scored.current_evaluation_frame.snapshot.team_deathmatch_scores == (3, 4)
    for session in fall_game:
        index = session.current_evaluation_frame.frame_index
        assert len(session.recent_scores) == min(MOMENTUM_TICKS, index)
        expected = (2, 4) if index <= MOMENTUM_TICKS else (3, 4)
        assert session.scores_ten_ticks_ago == expected
    assert reset_session(fall_game[-1]).recent_scores == ()
    with pytest.raises(ValueError, match="more frames"):
        replace(scored, recent_scores=((2, 4), (2, 4)))
    with pytest.raises(ValueError, match="going down"):
        replace(scored, recent_scores=((2, 5),))
    with pytest.raises(TypeError, match="pairs"):
        replace(scored, recent_scores=((2, -1),))


def test_every_live_view_carries_the_same_scores_ten_ticks_ago(
    fall_game: tuple[DebuggerSession, ...],
) -> None:
    for session in (fall_game[1], fall_game[-1]):
        oracle = _summary(session, "researcher")
        pov = _summary(session, "pov")
        assert oracle.scores_ten_ticks_ago == session.scores_ten_ticks_ago
        assert pov.scores_ten_ticks_ago == session.scores_ten_ticks_ago
        assert pov.scores == oracle.scores
    no_shared = _charge(_session("no_shared_obs"))
    assert _summary(no_shared, "pov").scores_ten_ticks_ago == (2, 4)
    assert _summary(no_shared, "researcher").scores_ten_ticks_ago == (2, 4)


def _replay_of(sessions: tuple[DebuggerSession, ...]) -> LoadedReplay:
    context = sessions[-1].evaluation_context
    provenance = _runtime_provenance().model_copy(
        update={"package_version": context.code_revision.package_version}
    )
    return LoadedReplay(
        build_replay_v4(
            context,
            [session.current_evaluation_frame for session in sessions],
            [
                session.incoming_evaluation_view.transition
                for session in sessions[1:]
                if session.incoming_evaluation_view is not None
            ],
            runtime_provenance=provenance,
        )
    )


def _replay_summary(service: ReplayViewerService) -> MatchSummaryV1:
    summary = getattr(service.current_presentation().payload, "match_summary", None)
    assert isinstance(summary, MatchSummaryV1)
    return summary


def test_replay_viewer_reads_the_recorded_scores_ten_frames_back(
    fall_game: tuple[DebuggerSession, ...],
) -> None:
    loaded = _replay_of(fall_game)
    frames = loaded.replay.frames
    assert len(frames) == MOMENTUM_TICKS + 2
    for view_mode in ("researcher", "pov"):
        service = ReplayViewerService(loaded, view_mode=view_mode)
        for index in (0, 1, MOMENTUM_TICKS, MOMENTUM_TICKS + 1, 1):
            if index != service.current_frame().cursor.frame_index:
                result = service.apply_command(
                    ReplayCommandRequestV1(
                        client_id="momentum-test",
                        command_id=f"seek-{view_mode}-{index}-{service.revision}",
                        base_revision=service.revision,
                        command=ReplayAbsoluteSeekCommandV1(frame_index=index),
                    )
                )
                assert result.outcome == "response"
            summary = _replay_summary(service)
            earlier = frames[max(0, index - MOMENTUM_TICKS)].snapshot
            assert summary.scores_ten_ticks_ago == tuple(earlier.team_deathmatch_scores)
            # The same game in the live service gives the same public fact.
            assert summary.scores_ten_ticks_ago == fall_game[index].scores_ten_ticks_ago
    # The scoring tick stays recent through frame 10 and leaves on frame 11.
    assert fall_game[MOMENTUM_TICKS].scores_ten_ticks_ago == (2, 4)
    assert fall_game[MOMENTUM_TICKS + 1].scores_ten_ticks_ago == (3, 4)
