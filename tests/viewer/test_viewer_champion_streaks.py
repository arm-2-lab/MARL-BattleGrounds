"""Check the Champion Streaks host fact (viewer.champion_streaks) in both servers.

A champion reaches a streak on the tick it takes part in its 5th, 10th, 15th
or 20th (every 5th) takedown without falling. The host counts; the browser
only draws the frame field ``champion_streaks``. Contracts checked:

- The counting rule: kill credit is shared (two sources on one fall both
  gain), a fall resets the fallen champion, a source that falls on the same
  tick gains nothing, every multiple of 5 (5, 10, 20) is announced once (a
  jump from 4 to 9 reports 5; 7 to 9 reports nothing), a record accepts 20 and
  refuses 6 and 8, and impossible counts, ticks and records are refused.
- Live service: the Charge of the Vision Moment's "fall" start counts one
  takedown for the Warrior in Oracle View. Under NoSharedObs a recipient that
  did not see the blow counts nothing (an unseen takedown never counts in Agent
  POV) while the Warrior's own view counts it; under SharedObs every recipient
  saw it. Restart clears every count. Each live view's presentation carries
  only its own row: Oracle View the Oracle row, Agent POV the recipient's row.
- Replay Viewer service: on a replay saved from the same live game, in both
  observation modes, Oracle View and every recipient's Agent POV count exactly
  what the live session counted on every frame. A first request at the last
  frame (a seek) and a new service on the same replay (a page reload) give the
  same tables as counting frame by frame. On the recorded scenario fixture the
  Team B Mage reaches two takedowns in Oracle View and in the Agent POV that saw
  them, and every presentation carries an empty field when nobody reached 5.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import pytest
from pydantic import ValidationError
from scripts.dev.visual_debugger.scenarios import vision_moment_scenario
from tests.evaluation.test_evaluation_pov import _runtime_provenance
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
from marl_battlegrounds.viewer.champion_streaks import (
    NO_TAKEDOWNS,
    ZERO_COUNTS,
    ChampionStreakV1,
    ChampionTakedownsV1,
    TickTakedownsV1,
    advance_takedown_counts_v1,
)
from marl_battlegrounds.viewer.replay_service import ReplayViewerService

_SCENARIO_REPLAY = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "historical_scenario_v4"
    / "scenario_1.marlbg-replay.json"
)


def _tick(falls: set[int], takedowns: set[tuple[int, int]]) -> TickTakedownsV1:
    return TickTakedownsV1(falls=frozenset(falls), takedowns=frozenset(takedowns))


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


def _game(mode: str) -> tuple[DebuggerSession, ...]:
    # The Warrior (slot 1) Charges the Team B Mage (slot 6), who falls; then two
    # quiet ticks.
    zeros = jnp.zeros((10,), dtype=jnp.int32)
    charge = Action(
        move=zeros,
        select_target=zeros.at[1].set(6),
        use_ultimate=zeros.at[1].set(1),
    )
    sessions = [_session(mode)]
    for action in (charge, make_neutral_joint_action(), make_neutral_joint_action()):
        sessions.append(
            submit_joint_action(
                sessions[-1],
                action,
                submission_kind="interactive",
                report_actor_slots=(),
            )
        )
    return tuple(sessions)


@pytest.fixture(scope="module")
def games() -> dict[str, tuple[DebuggerSession, ...]]:
    return {mode: _game(mode) for mode in ("shared_obs", "no_shared_obs")}


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


def _replay_counts(
    service: ReplayViewerService, recipient: int | None, frame_index: int
) -> tuple[int, ...]:
    pov_entry = (
        service._pov_entry(recipient, cache=service._pov_cache)
        if recipient is not None
        and service._context.execution_information_mode == "no_shared_obs"
        else None
    )
    service._champion_streaks(
        frame_index, recipient_global_slot=recipient, pov_entry=pov_entry
    )
    table = service._champion_streak_tables[recipient]
    assert len(table.reached) == frame_index + 1
    return table.counts


def test_counting_rule_shares_credit_resets_on_a_fall_and_announces_every_fifth() -> (
    None
):
    counts, reached = advance_takedown_counts_v1(
        ZERO_COUNTS, _tick({6}, {(1, 6), (2, 6)})
    )
    assert counts == (0, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    assert reached == ()
    # The Warrior falls on the same tick it strikes: it gains nothing and resets.
    counts, _ = advance_takedown_counts_v1(counts, _tick({1, 7}, {(1, 7), (2, 7)}))
    assert counts == (0, 0, 2, 0, 0, 0, 0, 0, 0, 0)
    counts = (0, 0, 4, 0, 0, 0, 0, 0, 0, 0)
    counts, reached = advance_takedown_counts_v1(counts, _tick({7}, {(2, 7)}))
    assert (counts[2], reached) == (5, ((2, 5),))
    counts, reached = advance_takedown_counts_v1(counts, _tick({8}, {(2, 8)}))
    assert (counts[2], reached) == (6, ())
    counts, reached = advance_takedown_counts_v1(
        (0, 0, 7, 0, 0, 0, 0, 0, 0, 0), _tick({6, 8}, {(2, 6), (2, 8)})
    )
    assert (counts[2], reached) == (9, ())
    counts, reached = advance_takedown_counts_v1(
        (0, 0, 9, 0, 0, 0, 0, 0, 0, 0), _tick({6, 8}, {(2, 6), (2, 8)})
    )
    assert (counts[2], reached) == (11, ((2, 10),))
    counts, reached = advance_takedown_counts_v1(
        (0, 0, 19, 0, 0, 0, 0, 0, 0, 0), _tick({6}, {(2, 6)})
    )
    assert (counts[2], reached) == (20, ((2, 20),))
    counts, reached = advance_takedown_counts_v1(
        (0, 0, 4, 0, 0, 0, 0, 0, 0, 0),
        _tick({5, 6, 7, 8, 9}, {(2, slot) for slot in range(5, 10)}),
    )
    assert (counts[2], reached) == (9, ((2, 5),))
    # A tick this view saw no fall on changes nothing.
    assert advance_takedown_counts_v1(counts, NO_TAKEDOWNS) == (counts, ())


def test_tick_and_state_records_refuse_impossible_values() -> None:
    with pytest.raises(ValueError, match="join another"):
        _tick({6}, {(1, 7)})
    with pytest.raises(ValueError, match="join another"):
        _tick({6}, {(6, 6)})
    with pytest.raises(ValueError, match="0-9"):
        _tick({10}, set())
    with pytest.raises(ValueError, match="ten whole"):
        advance_takedown_counts_v1((0,) * 9, NO_TAKEDOWNS)
    with pytest.raises(ValueError, match="join its champion"):
        ChampionTakedownsV1(oracle_reached=((1, 5),))
    with pytest.raises(ValueError, match="one row per"):
        ChampionTakedownsV1(agent_pov_counts=(ZERO_COUNTS,))
    with pytest.raises(ValidationError):
        ChampionStreakV1(public_agent_id="a", team_id=1, class_id=2, takedowns=6)
    with pytest.raises(ValidationError):
        ChampionStreakV1(public_agent_id="a", team_id=1, class_id=2, takedowns=8)
    assert (
        ChampionStreakV1(
            public_agent_id="a", team_id=1, class_id=2, takedowns=20
        ).takedowns
        == 20
    )


def test_live_agent_pov_counts_only_what_the_team_saw(
    games: dict[str, tuple[DebuggerSession, ...]],
) -> None:
    for mode, sessions in games.items():
        start, charged, *_ = sessions
        assert start.champion_takedowns == ChampionTakedownsV1()
        takedowns = charged.champion_takedowns
        assert takedowns.oracle_counts == (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
        assert takedowns.agent_pov_counts[1] == takedowns.oracle_counts
        unseen = [
            recipient
            for recipient, counts in enumerate(takedowns.agent_pov_counts)
            if counts == ZERO_COUNTS
        ]
        if mode == "shared_obs":
            assert unseen == []
        else:
            # Recipient 0 did not see the Charge land, so it counts nothing.
            assert 0 in unseen
        # Quiet ticks keep every count; restart clears them.
        assert sessions[-1].champion_takedowns.oracle_counts == takedowns.oracle_counts
        assert reset_session(sessions[-1]).champion_takedowns == ChampionTakedownsV1()
        with pytest.raises(ValueError, match="frame zero"):
            replace(start, champion_takedowns=takedowns)


def test_each_live_view_gets_only_its_own_row(
    games: dict[str, tuple[DebuggerSession, ...]],
) -> None:
    charged = games["shared_obs"][1]
    counts = charged.champion_takedowns
    warrior = (0, 5, 0, 0, 0, 0, 0, 0, 0, 0)
    crafted = replace(
        charged,
        controlled_global_slot=5,
        champion_takedowns=ChampionTakedownsV1(
            oracle_counts=warrior,
            oracle_reached=((1, 5),),
            agent_pov_counts=tuple(
                warrior if recipient == 5 else row
                for recipient, row in enumerate(counts.agent_pov_counts)
            ),
            agent_pov_reached=tuple(
                ((1, 5),) if recipient == 5 else () for recipient in range(10)
            ),
        ),
    )
    expected = (
        ChampionStreakV1(
            public_agent_id=charged.evaluation_context.roster[1].public_agent_id,
            team_id=1,
            class_id=2,
            takedowns=5,
        ),
    )
    for view_mode, focus, streaks in (
        ("researcher", 5, expected),
        ("pov", 5, expected),
        ("pov", 0, ()),
    ):
        service = DebuggerService(
            replace(crafted, controlled_global_slot=focus),
            view_mode=view_mode,  # pyright: ignore[reportArgumentType]
            preset="analysis",
            include_stress=False,
        )
        try:
            payload = service.current_presentation().payload
        finally:
            service.close()
        assert getattr(payload, "champion_streaks", None) == streaks


def test_replay_viewer_counts_match_the_live_game_after_seeks_and_reloads(
    games: dict[str, tuple[DebuggerSession, ...]],
) -> None:
    for sessions in games.values():
        loaded = _replay_of(sessions)
        last = len(sessions) - 1
        active = [
            row.global_slot
            for row in sessions[0].evaluation_context.roster
            if row.configured_active
        ]
        sequential = ReplayViewerService(loaded)
        for recipient in (None, *active):
            for index, session in enumerate(sessions):
                expected = (
                    session.champion_takedowns.oracle_counts
                    if recipient is None
                    else session.champion_takedowns.agent_pov_counts[recipient]
                )
                assert _replay_counts(sequential, recipient, index) == expected
        # A seek straight to the last frame, and a new service on the same replay
        # (a page reload), count exactly the same tables.
        for reloaded in (ReplayViewerService(loaded), ReplayViewerService(loaded)):
            for recipient in (None, *active):
                assert _replay_counts(reloaded, recipient, last) == (
                    _replay_counts(sequential, recipient, last)
                )
                assert (
                    reloaded._champion_streak_tables[recipient].reached
                    == sequential._champion_streak_tables[recipient].reached
                )


def test_replay_viewer_scenario_fixture_counts_each_view() -> None:
    loaded = load_replay(_SCENARIO_REPLAY)
    service = ReplayViewerService(loaded)
    two = (0, 0, 0, 0, 0, 2, 0, 0, 0, 0)
    assert _replay_counts(service, None, 5) == two
    assert _replay_counts(service, 5, 5) == two
    for view_mode in ("researcher", "pov"):
        viewer = ReplayViewerService(loaded, view_mode=view_mode, initial_frame_index=2)
        payload = viewer.current_presentation().payload
        assert getattr(payload, "champion_streaks", None) == ()
