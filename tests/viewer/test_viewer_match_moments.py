"""Check the Herald Cards history facts (viewer.match_moments) in both servers.

The host owns the facts the browser cannot hold: whether the match began at
tick 0 with 0-0, the largest earlier lead, each view's unanswered takedown
runs and each view's rescues. The browser only draws the frame field
``match_moments``. Contracts checked:

- Runs: a team gains one per enemy fall it saw, any loss of its own (a trade
  included) starts it again, every multiple of 5 (5, 10) is announced once (3
  and 8 are not), a jump from 2 to 5 reports 5, a run record refuses 3 and 8,
  and impossible runs are refused.
- History: the first transition sets the fresh start from its start frame,
  the largest lead grows from each tick's start scores, and each view keeps
  its own runs and rescues; the frame field names each team once in runs and
  each champion once in rescues.
- Rescue, live service, the Vision Moment's "saved" start (your Warrior's
  Charge would fell the Team B Mage and the Team B Priest's Holy Word:
  Salvation keeps it alive): Oracle View names the rescue by Ultimate; each
  Agent POV names it only when that team saw the Mage at both ends, the
  Priest's heal and every blow; the presentation carries only its own view's
  facts, and the protocol refuses a rescue of a champion the view did not see
  at both ends of the tick. Restart clears the history.
- Replay Viewer service: on a replay saved from the same live game, Oracle
  View and every recipient's Agent POV give exactly the live facts on every
  frame, after a seek to the last frame and in a new service.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import pytest
from pydantic import ValidationError
from scripts.dev.visual_debugger.scenarios import vision_moment_scenario
from tests.evaluation.test_evaluation_pov import _runtime_provenance
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client.control import (
    create_session,
    reset_session,
    submit_joint_action,
)
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation.replay_io import LoadedReplay
from marl_battlegrounds.evaluation.replay_v4 import build_replay_v4
from marl_battlegrounds.viewer.match_moments import (
    NO_RUNS,
    MatchMomentFactsV1,
    MatchMomentHistoryV1,
    RescueV1,
    TeamRunV1,
    advance_team_runs_v1,
)
from marl_battlegrounds.viewer.presentation_protocol import (
    LiveNoSharedObsAuthorizedPresentationFrameV1,
)
from marl_battlegrounds.viewer.replay_service import ReplayViewerService

_MAGE_B = 5
_PRIEST_B = 9


def _session(mode: str) -> DebuggerSession:
    return create_session(
        vision_moment_scenario("saved"),
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


def _saved_game(mode: str) -> tuple[DebuggerSession, DebuggerSession]:
    # Your Warrior (slot 1) Charges the Team B Mage (target action 6); the Team
    # B Priest (slot 9) casts Holy Word: Salvation on it (ally action 1).
    zeros = jnp.zeros((10,), dtype=jnp.int32)
    action = Action(
        move=zeros,
        select_target=zeros.at[1].set(6).at[_PRIEST_B].set(1),
        use_ultimate=zeros.at[1].set(1).at[_PRIEST_B].set(1),
    )
    start = _session(mode)
    saved = submit_joint_action(
        start, action, submission_kind="interactive", report_actor_slots=()
    )
    return start, saved


@pytest.fixture(scope="module")
def games() -> dict[str, tuple[DebuggerSession, DebuggerSession]]:
    return {mode: _saved_game(mode) for mode in ("shared_obs", "no_shared_obs")}


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


def test_runs_count_seen_enemy_falls_and_restart_on_a_loss(
    games: dict[str, tuple[DebuggerSession, DebuggerSession]],
) -> None:
    context = games["shared_obs"][0].evaluation_context
    runs, reached = advance_team_runs_v1(NO_RUNS, context, frozenset({5, 6}))
    assert (runs, reached) == ((2, 0), ())
    runs, reached = advance_team_runs_v1(runs, context, frozenset({7}))
    assert (runs, reached) == ((3, 0), ())
    # A trade starts both counts again.
    assert advance_team_runs_v1((4, 2), context, frozenset({0, 8})) == ((0, 0), ())
    # A jump from 2 to 5 reports 5; Team B's run grows the same way.
    assert advance_team_runs_v1((2, 7), context, frozenset({5, 6, 7})) == (
        (5, 0),
        ((1, 5),),
    )
    assert advance_team_runs_v1((0, 7), context, frozenset({0})) == ((0, 8), ())
    assert advance_team_runs_v1((0, 9), context, frozenset({0})) == (
        (0, 10),
        ((2, 10),),
    )
    with pytest.raises(ValidationError):
        TeamRunV1(team_id=1, takedowns=3)
    with pytest.raises(ValidationError):
        TeamRunV1(team_id=2, takedowns=8)
    assert advance_team_runs_v1((4, 4), context, frozenset()) == ((4, 4), ())
    with pytest.raises(ValueError, match="two whole"):
        advance_team_runs_v1((0, -1), context, frozenset())
    with pytest.raises(ValueError, match="0-9"):
        advance_team_runs_v1(NO_RUNS, context, frozenset({10}))


def test_history_sets_the_fresh_start_grows_the_lead_and_keeps_each_view(
    games: dict[str, tuple[DebuggerSession, DebuggerSession]],
) -> None:
    context = games["shared_obs"][0].evaluation_context
    quiet = (frozenset[int](),) * 10
    none = ((),) * 10
    history = MatchMomentHistoryV1().advance(
        context,
        start_frame_index=0,
        start_simulator_step_count=0,
        start_scores=(0, 0),
        oracle_falls=frozenset({5}),
        oracle_rescues=((1, False),),
        agent_pov_falls=(frozenset({5}), *quiet[1:]),
        agent_pov_rescues=none,
    )
    assert history.fresh_start
    assert history.facts(context, None).rescues[0].by_ultimate is False
    assert history.facts(context, 3).rescues == ()
    later = history.advance(
        context,
        start_frame_index=1,
        start_simulator_step_count=1,
        start_scores=(4, 1),
        oracle_falls=frozenset({6, 7, 8, 9}),
        oracle_rescues=(),
        agent_pov_falls=quiet,
        agent_pov_rescues=none,
    )
    oracle = later.facts(context, None)
    assert (oracle.fresh_start, oracle.largest_lead_before) == (True, 3)
    assert oracle.runs == (TeamRunV1(team_id=1, takedowns=5),)
    assert later.facts(context, 0).runs == ()
    assert later.agent_pov_runs[0] == (1, 0)
    # A match that began anywhere but tick 0 at 0-0 is not a fresh start.
    scenario = MatchMomentHistoryV1().advance(
        context,
        start_frame_index=0,
        start_simulator_step_count=91,
        start_scores=(2, 4),
        oracle_falls=frozenset(),
        oracle_rescues=(),
        agent_pov_falls=quiet,
        agent_pov_rescues=none,
    )
    assert (scenario.fresh_start, scenario.largest_lead) == (False, 2)
    with pytest.raises(ValueError, match="one entry per"):
        scenario.advance(
            context,
            start_frame_index=1,
            start_simulator_step_count=92,
            start_scores=(2, 4),
            oracle_falls=frozenset(),
            oracle_rescues=(),
            agent_pov_falls=quiet[:9],
            agent_pov_rescues=none,
        )
    with pytest.raises(ValueError, match="one row per"):
        MatchMomentHistoryV1(agent_pov_runs=(NO_RUNS,))
    with pytest.raises(ValidationError):
        MatchMomentFactsV1(
            fresh_start=False,
            largest_lead_before=0,
            runs=(
                TeamRunV1(team_id=1, takedowns=5),
                TeamRunV1(team_id=1, takedowns=10),
            ),
            rescues=(),
        )


def test_live_rescue_is_named_by_ultimate_in_each_view_that_saw_it_whole(
    games: dict[str, tuple[DebuggerSession, DebuggerSession]],
) -> None:
    for mode, (start, saved) in games.items():
        assert start.match_moments == MatchMomentHistoryV1()
        history = saved.match_moments
        assert history.oracle_rescues == ((_MAGE_B, True),)
        assert (history.fresh_start, history.largest_lead) == (False, 2)
        context = saved.evaluation_context
        mage_id = context.roster[_MAGE_B].public_agent_id
        oracle = history.facts(context, None)
        assert oracle.rescues == (
            RescueV1(public_agent_id=mage_id, team_id=2, class_id=1, by_ultimate=True),
        )
        # The Mage itself saw the Charge, the heal and its own life.
        assert history.agent_pov_rescues[_MAGE_B] == ((_MAGE_B, True),)
        if mode == "shared_obs":
            # Shared sight: every recipient saw the whole rescue.
            assert all(row == ((_MAGE_B, True),) for row in history.agent_pov_rescues)
        else:
            # The Priest did not see your Warrior before its Charge, and your
            # Warrior did not see the Priest: neither view names the rescue.
            assert history.agent_pov_rescues[_PRIEST_B] == ()
            assert history.agent_pov_rescues[1] == ()
        # Each live presentation carries only its own view's facts.
        for view_mode, focus, expected in (
            ("researcher", 1, oracle),
            ("pov", _PRIEST_B, history.facts(context, _PRIEST_B)),
            ("pov", 1, history.facts(context, 1)),
        ):
            service = DebuggerService(
                replace(saved, controlled_global_slot=focus),
                view_mode=view_mode,  # pyright: ignore[reportArgumentType]
                preset="analysis",
                include_stress=False,
            )
            try:
                payload = service.current_presentation().payload
            finally:
                service.close()
            assert getattr(payload, "match_moments", None) == expected
        assert reset_session(saved).match_moments == MatchMomentHistoryV1()
        with pytest.raises(ValueError, match="frame zero"):
            replace(start, match_moments=history)


def test_the_protocol_refuses_a_rescue_the_view_did_not_see_at_both_ends(
    games: dict[str, tuple[DebuggerSession, DebuggerSession]],
) -> None:
    _, saved = games["no_shared_obs"]
    context = saved.evaluation_context
    unseen = [
        slot
        for slot in range(10)
        if context.roster[slot].configured_active
        and saved.match_moments.agent_pov_rescues[slot] == ()
    ]
    assert unseen, "some Agent POV recipient did not see the rescue whole"
    service = DebuggerService(
        replace(saved, controlled_global_slot=unseen[0]),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )
    try:
        payload = service.current_presentation().payload
    finally:
        service.close()
    assert isinstance(payload, LiveNoSharedObsAuthorizedPresentationFrameV1)
    assert payload.visual_events is not None
    trajectories = payload.visual_events.agent_phase_trajectories
    seen = {
        row.agent_public_agent_id
        for row in trajectories
        if row.transition_start is not None and row.successor is not None
    }
    outsider = next(row for row in context.roster if row.public_agent_id not in seen)
    forged = MatchMomentFactsV1(
        fresh_start=False,
        largest_lead_before=2,
        runs=(),
        rescues=(
            RescueV1(
                public_agent_id=outsider.public_agent_id,
                team_id=outsider.configured_team_id,  # pyright: ignore[reportArgumentType]
                class_id=outsider.class_id,  # pyright: ignore[reportArgumentType]
                by_ultimate=False,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="both ends of the tick"):
        type(payload).model_validate(
            {**dict(payload), "match_moments": forged}  # pyright: ignore[reportArgumentType]
        )


def test_replay_viewer_facts_match_the_live_game_after_seeks_and_reloads(
    games: dict[str, tuple[DebuggerSession, DebuggerSession]],
) -> None:
    for sessions in games.values():
        loaded = _replay_of(sessions)
        context = sessions[0].evaluation_context
        active = [row.global_slot for row in context.roster if row.configured_active]
        mode = context.execution_information_mode

        def facts(
            service: ReplayViewerService,
            recipient: int | None,
            frame: int,
            mode: str = mode,
        ) -> MatchMomentFactsV1:
            entry = (
                service._pov_entry(recipient, cache=service._pov_cache)
                if recipient is not None and mode == "no_shared_obs"
                else None
            )
            return service._match_moments(
                frame, recipient_global_slot=recipient, pov_entry=entry
            )

        sequential = ReplayViewerService(loaded)
        for recipient in (None, *active):
            for index, session in enumerate(sessions):
                expected = session.match_moments.facts(context, recipient)
                assert facts(sequential, recipient, index) == expected
        reloaded = ReplayViewerService(loaded)
        for recipient in (None, *active):
            assert facts(reloaded, recipient, 1) == facts(sequential, recipient, 1)
