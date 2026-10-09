"""Check the game-over Honors host fact (viewer.match_summary) in both servers.

The Verdict card's Honors and Match Summary rows are counted by the host,
per view, from the recorded history. Contracts checked:

- One transition's totals: the Vision Moment's "fall" Charge gives the
  Warrior 1 Kill Contribution, 1 accepted Ultimate and its full damage
  (damage beyond the victim's last life counts), and the fallen Mage 1 death.
- Picking Honors: each honor names every champion with the best rounded
  value (ties), an honor nobody earned is left out, unknown kill credit drops
  Kill Contributions, amounts round half up, labels come from the metric
  catalog, every champion is listed when no team is given and only that
  team's own champions when one is. Impossible records are refused.
- Live service: the game's last turn ("last_tick" start) gives Honors over
  every champion in Oracle View and in Agent POV for either team's recipient
  (the game is over, so nothing is left to hide); a frame before game over
  carries none.
- Replay Viewer service: the replay saved from the same game gives the same
  Honors on its last frame in both views, and none on other frames; one
  team's Honors can still be picked.
- Protocol: an Agent POV frame refuses Honors of only the other team, and a
  frame whose own match summary does not show a finished game (in progress,
  no result, or no summary at all) refuses Honors in both views, so a
  mid-game Agent POV frame can never carry every champion's totals. One
  team's Honors list only that team's champions.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import replace
from typing import cast

import jax.numpy as jnp
import pytest
from pydantic import BaseModel, ValidationError
from scripts.dev.visual_debugger.scenarios import vision_moment_scenario
from tests.evaluation.test_evaluation_pov import _runtime_provenance
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client.control import (
    create_session,
    submit_joint_action,
)
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation.replay_io import LoadedReplay
from marl_battlegrounds.evaluation.replay_v4 import build_replay_v4
from marl_battlegrounds.viewer.match_summary import (
    MatchAgentV1,
    MatchHonorsV1,
    MatchHonorV1,
    MatchTotalsV1,
    build_match_summary_v1,
    match_honors_v1,
    match_transition_totals_v1,
)
from marl_battlegrounds.viewer.replay_service import ReplayViewerService


def _session(start: str) -> DebuggerSession:
    return create_session(
        vision_moment_scenario(start),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="manual",
        team_b_controller="manual",
        execution_information_mode="shared_obs",
        manual_slots=tuple(range(10)),
        slot_labels=("Manual",) * 10,
    )


def _charge() -> Action:
    # The Team A Warrior (slot 1) Charges the Team B Mage (target action 6).
    zeros = jnp.zeros((10,), dtype=jnp.int32)
    return Action(
        move=zeros,
        select_target=zeros.at[1].set(6),
        use_ultimate=zeros.at[1].set(1),
    )


def _turn(session: DebuggerSession, action: Action) -> DebuggerSession:
    return submit_joint_action(
        session, action, submission_kind="interactive", report_actor_slots=()
    )


@pytest.fixture(scope="module")
def last_game() -> tuple[DebuggerSession, DebuggerSession]:
    start = _session("last_tick")
    return start, _turn(start, _charge())


def _totals_of(session: DebuggerSession) -> MatchTotalsV1:
    view = session.incoming_evaluation_view
    assert view is not None
    summary = build_match_summary_v1(
        session.evaluation_context,
        session.current_evaluation_frame,
        view.transition.events,
        transition_start_frame=view.start_frame,
    )
    return match_transition_totals_v1(
        session.evaluation_context, summary, view.transition.events
    )


def test_one_charge_takedown_counts_kill_damage_death_and_ultimate() -> None:
    charged = _turn(_session("fall"), _charge())
    totals = _totals_of(charged)
    fallen = [slot for slot, count in enumerate(totals.deaths) if count]
    assert len(fallen) == 1 and fallen[0] >= 5
    assert totals.kill_contributions[1] == 1
    assert totals.ultimate_activations[1] == 1
    # The fallen Mage had 1 life; the Charge's whole blow still counts.
    assert totals.damage_done[1] > 1
    assert sum(totals.kill_contributions) == 1
    assert totals.kill_credit_known
    with pytest.raises(ValueError, match="needs its match summary"):
        match_transition_totals_v1(
            charged.evaluation_context,
            None,
            charged.incoming_evaluation_view.transition.events,  # pyright: ignore[reportOptionalMemberAccess]
        )


def test_honors_pick_tied_leaders_per_view_and_refuse_impossible_records(
    last_game: tuple[DebuggerSession, DebuggerSession],
) -> None:
    context = last_game[0].evaluation_context
    zeros = (0,) * 10
    totals = MatchTotalsV1(
        kill_contributions=(3, 1, 3, 0, 0, 2, 0, 0, 0, 0),
        damage_done=(10.5, 0.0, 0.0, 0.0, 0.0, 10.4, 0.0, 0.0, 0.0, 0.0),
        effective_healing_done=(0.0,) * 10,
        deaths=zeros,
    )
    oracle = match_honors_v1(context, totals, team_id=None)
    active = [row.public_agent_id for row in context.roster if row.configured_active]
    assert [row.public_agent_id for row in oracle.champions] == active
    assert [honor.metric for honor in oracle.honors] == [
        "kill_contributions",
        "damage_done",
    ]
    kills, damage = oracle.honors
    assert (kills.label, kills.value, len(kills.leaders)) == (
        "Kill Contributions",
        3,
        2,
    )
    # 10.5 rounds up to 11 and 10.4 down to 10: one leader.
    assert (damage.label, damage.value, len(damage.leaders)) == ("Damage Done", 11, 1)
    assert [row.label for row in oracle.labels][3] == "Effective Healing Done"
    team_b = match_honors_v1(context, totals, team_id=2)
    assert {row.team_id for row in team_b.champions} == {2}
    assert [(h.metric, h.value) for h in team_b.honors] == [
        ("kill_contributions", 2),
        ("damage_done", 10),
    ]
    unknown = match_honors_v1(
        context, replace(totals, kill_credit_known=False), team_id=None
    )
    assert [honor.metric for honor in unknown.honors] == ["damage_done"]
    assert all(row.kill_contributions is None for row in unknown.champions)
    leader = oracle.honors[0].leaders[0]
    with pytest.raises(ValidationError, match="best value"):
        MatchHonorsV1(
            schema_version=1,
            team_id=None,
            labels=oracle.labels,
            honors=(
                MatchHonorV1(
                    metric="kill_contributions",
                    label="Kill Contributions",
                    value=1,
                    leaders=(leader,),
                ),
            ),
            champions=oracle.champions,
        )
    with pytest.raises(ValidationError, match="only that team"):
        MatchHonorsV1(
            schema_version=1,
            team_id=2,
            labels=oracle.labels,
            honors=(),
            champions=oracle.champions,
        )
    assert isinstance(leader, MatchAgentV1)


def _revalidate_with_honors(
    payload: object, honors: MatchHonorsV1 | None, **changes: object
) -> None:
    model = cast(BaseModel, payload)
    fields = type(model).model_fields
    type(model).model_validate(
        {
            **{name: getattr(model, name) for name in fields},
            "match_honors": honors,
            **changes,
        }
    )


def _presentation(
    session: DebuggerSession, view_mode: str, focus: int | None = None
) -> object:
    service = DebuggerService(
        session if focus is None else replace(session, controlled_global_slot=focus),
        view_mode=view_mode,  # pyright: ignore[reportArgumentType]
        preset="analysis",
        include_stress=False,
    )
    try:
        return service.current_presentation().payload
    finally:
        service.close()


def test_live_and_replay_honors_follow_each_view(
    last_game: tuple[DebuggerSession, DebuggerSession],
) -> None:
    start, ended = last_game
    assert getattr(_presentation(start, "researcher"), "match_honors", 0) is None
    oracle = getattr(_presentation(ended, "researcher"), "match_honors", None)
    assert isinstance(oracle, MatchHonorsV1)
    assert oracle.team_id is None
    assert [honor.metric for honor in oracle.honors] == ["damage_done"]
    warrior = ended.evaluation_context.roster[1].public_agent_id
    assert [row.public_agent_id for row in oracle.honors[0].leaders] == [warrior]
    # Agent POV shows every champion at game over, for either team.
    pov = getattr(_presentation(ended, "pov", focus=1), "match_honors", None)
    assert pov == oracle
    team_b = getattr(_presentation(ended, "pov", focus=6), "match_honors", None)
    assert team_b == oracle
    context = ended.evaluation_context
    provenance = _runtime_provenance().model_copy(
        update={"package_version": context.code_revision.package_version}
    )
    loaded = LoadedReplay(
        build_replay_v4(
            context,
            [start.current_evaluation_frame, ended.current_evaluation_frame],
            [ended.incoming_evaluation_view.transition],  # pyright: ignore[reportOptionalMemberAccess]
            runtime_provenance=provenance,
        )
    )
    replay = ReplayViewerService(loaded)
    assert replay._match_honors(0, team_id=None) is None
    assert replay._match_honors(1, team_id=None) == oracle
    own_team = replay._match_honors(1, team_id=1)
    assert own_team is not None and {row.team_id for row in own_team.champions} == {1}
    viewer = ReplayViewerService(loaded, view_mode="pov", initial_frame_index=1)
    payload = viewer.current_presentation().payload
    assert getattr(payload, "match_honors", None) == oracle
    recipient = str(
        getattr(
            getattr(payload, "source", None), "source_recipient_public_agent_id", ""
        )
    )
    recipient_team = next(
        row.configured_team_id
        for row in context.roster
        if row.public_agent_id == recipient
    )
    other_team = replay._match_honors(1, team_id=2 if recipient_team == 1 else 1)
    with pytest.raises(ValidationError, match="every champion or"):
        _revalidate_with_honors(payload, other_team)
    # Before game over no frame may carry Honors, in either view.
    for mid_game in (
        _presentation(start, "researcher"),
        _presentation(start, "pov", focus=1),
        ReplayViewerService(loaded, view_mode="pov", initial_frame_index=0)
        .current_presentation()
        .payload,
    ):
        summary = getattr(mid_game, "match_summary", None)
        assert summary is not None and summary.outcome == "in_progress"
        with pytest.raises(ValidationError, match="finished game"):
            _revalidate_with_honors(mid_game, oracle)
        # No summary, or a summary of a game without a result, is no proof either.
        diagnostic = summary.model_copy(
            update={
                "task_mode": 0,
                "outcome": "not_applicable",
                "score_threshold": 0,
                "scores": (0, 0),
                "previous_scores": (0, 0),
                "scores_ten_ticks_ago": (0, 0),
            }
        )
        for unfinished in (None, diagnostic):
            with pytest.raises(ValidationError, match="finished game"):
                _revalidate_with_honors(mid_game, oracle, match_summary=unfinished)
