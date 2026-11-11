"""Check how Capture the Flag games reach the clients' scenes and event lines.

Contracts: a Capture the Flag frame's researcher scene holds both flags (state,
position, home, carrier slot, countdown) and the rule settings, and its Oracle
scene names a carrier only by presentation key; a Team Deathmatch scene has no
flags (None). The visual event batch turns all six flag events into their
visual records, in recorded order, with each flag's start and successor
records; Oracle View keys every flag event agent and carrier. No flag event
reaches an Agent POV, even when every agent is seen. The records refuse a
carrier on the flag's own team, a home flag away from its home and flag events
without flag trajectories. The rendering layer's flag-row columns equal Core's.
A NoSharedObs Agent POV scene reads both flags from the recipient's own
objective rows only (decision 139's state bits): it names a carrier it sees by
POV key, leaves an unseen carrier unnamed, has no return ticks (None), and
refuses inactive flag rows, a home row without its owner (your home 1, 0, the
enemy home 0, 1), a named carrier the view does not hold and bits that break
decision 139. Its event summary gets one flag_changed line
per flag whose state changed between its two scenes (a pickup names the
teammate it sees, at successor), and one for a dropped flag grabbed and
dropped again (a new spot or a restarted countdown, decision 169), which a
whole summary accepts, never for a dropped flag that only counts down; never
one of the six Oracle flag events; and refuses flag lines without flag
trajectories. The POV scene test first reads Core's own rows from one real
tick (both flags at home); every other flag state is hand-set on that frame.
Host only, on CPU.
"""

import dataclasses
from typing import cast

import jax
import pytest
from tests.helpers.evaluation_fixtures import (
    current_captured_evaluation_trajectory,
    neutral_action,
)

from marl_battlegrounds.core import types as core_types
from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.evaluation._metrics.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation._recording.capture import (
    _capture_evaluation_transition_unit,  # pyright: ignore[reportPrivateUsage]
    _capture_initial_evaluation_frame,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.evaluation._recording.events import (
    decode_evaluation_events_v1,
)
from marl_battlegrounds.evaluation._recording.recording_context import (
    build_recording_context,
)
from marl_battlegrounds.evaluation._replays.pov import export_actor_pov_replay_v4
from marl_battlegrounds.evaluation._replays.replay_v5 import build_replay_v5
from marl_battlegrounds.evaluation.models import (
    CaptureTheFlagTransitionFactsV1,
    CodeRevisionV2,
    EvaluationEpisodeContextV5,
    EvaluationFrameV4,
    EvaluationTransitionV2,
)
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.rendering import evaluation_wire_features as wire
from marl_battlegrounds.rendering.authorized_pov_scene import (
    build_no_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AgentPovVisualIncomingFlagChangedEventV1,
    AuthorizedBattlefieldSceneV1,
    AuthorizedCaptureTheFlagFlagV1,
    _agent_pov_flag_line_changed,  # pyright: ignore[reportPrivateUsage]
    _agent_pov_visual_event_is_authorized,  # pyright: ignore[reportPrivateUsage]
    _project_replay_incoming_summary_v1,  # pyright: ignore[reportPrivateUsage]
    build_agent_pov_visual_incoming_summary_v1,
    build_oracle_authorized_scene_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    build_evaluation_battlefield_scene_v2,
    build_visual_event_batch_v2,
)
from marl_battlegrounds.rendering.scene import (
    CaptureTheFlagFlagSceneV1,
    CaptureTheFlagFlagState,
    VisualEventBatchV2,
)
from marl_battlegrounds.tasks import make_standard_capture_the_flag_config

_CONFIG = make_standard_capture_the_flag_config(
    map_id=42, team_a_roster=("mage", "warrior"), team_b_roster=("rogue",), max_steps=50
)
_KEYS = {0: "key-a0", 1: "key-a1", 5: "key-b0"}
_RUNTIME = capture_runtime_provenance("0.0.0")


def _context() -> EvaluationEpisodeContextV5:
    context, _ = build_recording_context(
        jax.device_get(_CONFIG),
        run_id="run-1",
        phase="evaluation",
        pass_id="1",
        episode={"episode_id": 7, "map_id": 42},
        policies={"team_a": "Alpha", "team_b": "Beta"},
        details={
            "code_revision": CodeRevisionV2(package_version="0.0.0").model_dump(
                mode="json"
            ),
            "runtime_provenance": _RUNTIME.model_dump(mode="json"),
            "execution_information_mode": "no_shared_obs",
        },
    )
    assert type(context) is EvaluationEpisodeContextV5
    return context


_CONTEXT = _context()


@pytest.fixture(scope="module")
def game() -> tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4]:
    state, observation, mask, _ = reset(_CONFIG, jax.random.PRNGKey(0))
    start = _capture_initial_evaluation_frame(_CONTEXT, state, observation, mask)
    state, observation, reward, done, mask, info = step(
        _CONFIG, state, mask, neutral_action(), jax.random.PRNGKey(1)
    )
    transition, successor = _capture_evaluation_transition_unit(
        _CONTEXT, start, state, observation, mask, info.transition_facts, reward, done
    )
    assert type(start) is EvaluationFrameV4 and type(successor) is EvaluationFrameV4
    assert type(transition) is EvaluationTransitionV2
    return start, transition, successor


def _frame(frame: EvaluationFrameV4, **flags: object) -> EvaluationFrameV4:
    snapshot = frame.snapshot.model_dump()
    for name, value in flags.items():
        snapshot[f"capture_the_flag_{name}"] = value
    return EvaluationFrameV4.model_validate(
        {**frame.model_dump(), "snapshot": snapshot}
    )


def _view(
    start: EvaluationFrameV4,
    transition: EvaluationTransitionV2,
    successor: EvaluationFrameV4,
    outcome: int = 0,
    context: EvaluationEpisodeContextV5 = _CONTEXT,
    **agent_facts: set[int],
) -> EvaluationTransitionViewV1:
    flags: dict[str, object] = {
        name: (False,) * 10
        for name in CaptureTheFlagTransitionFactsV1.model_fields
        if name.endswith("_by_agent")
    }
    flags["flag_was_automatically_returned_by_team"] = (False, False)
    for name, slots in agent_facts.items():
        length = 2 if name.endswith("_by_team") else 10
        flags[name] = tuple(i in slots for i in range(length))
    facts = transition.facts.model_copy(
        update={
            "capture_the_flag_facts": CaptureTheFlagTransitionFactsV1.model_validate(
                {**flags, "outcome": outcome}
            )
        }
    )
    events = decode_evaluation_events_v1(context, start, facts, successor)
    return EvaluationTransitionViewV1(
        context=context,
        start_frame=start,
        transition=transition.model_copy(update={"facts": facts, "events": events}),
        successor_frame=successor,
    )


def _flag_rows_of(batch: VisualEventBatchV2) -> list[tuple[object, ...]]:
    return [
        (
            event.event_type,
            getattr(event, "agent_global_slot", None),
            getattr(event, "flag_team_index", None),
        )
        for event in batch.events
        if event.phase_rank >= 130
    ]


def test_scenes_carry_both_flags_and_name_carriers_by_key(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
) -> None:
    start, transition, successor = game
    carrier_position = tuple(successor.snapshot.agent_positions[0])
    homes = _CONTEXT.resolved_env_config.capture_the_flag_flag_home_positions
    carried = _frame(
        successor,
        flag_carrier_slots=(None, 0),
        flag_positions=(homes[0], carrier_position),
    )
    pickup = _view(start, transition, carried, flag_was_picked_up_by_agent={0})
    tdm = current_captured_evaluation_trajectory()
    tdm_scene = build_evaluation_battlefield_scene_v2(tdm.context, tdm.frames[0])
    assert tdm_scene.capture_the_flag is None
    scene = build_evaluation_battlefield_scene_v2(
        _CONTEXT, carried, transition_view=pickup
    )
    assert scene.capture_the_flag is not None
    flags = scene.capture_the_flag.flags
    assert [(flag.state, flag.carrier_global_slot) for flag in flags] == [
        ("home", None),
        ("carried", 0),
    ]
    assert scene.capture_the_flag.return_steps == (
        _CONTEXT.resolved_env_config.capture_the_flag_dropped_flag_auto_return_steps
    )
    oracle = build_oracle_authorized_scene_v1(
        _CONTEXT, scene, authority_session_id="session-1"
    )
    assert oracle.capture_the_flag is not None
    public_id = next(
        row.public_agent_id for row in scene.agents if row.global_slot == 0
    )
    agent_zero = next(row for row in oracle.agents if row.public_agent_id == public_id)
    carried_flag = oracle.capture_the_flag.flags[1]
    assert (carried_flag.carrier_presentation_key, carried_flag.state) == (
        agent_zero.presentation_key,
        "carried",
    )


def test_all_six_flag_events_reach_oracle_view_and_never_agent_pov(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
) -> None:
    start, transition, successor = game
    homes = _CONTEXT.resolved_env_config.capture_the_flag_flag_home_positions
    return_steps = (
        _CONTEXT.resolved_env_config.capture_the_flag_dropped_flag_auto_return_steps
    )
    body = tuple(successor.snapshot.agent_positions[0])
    # Agent 0 takes Team B's flag and falls with it on one tick (decision 169).
    grab_and_fall = _view(
        start,
        transition,
        _frame(
            successor,
            dropped_flag_countdowns=(0, return_steps),
            flag_positions=(homes[0], body),
        ),
        flag_was_picked_up_by_agent={0},
        flag_was_dropped_by_agent={0},
    )
    # Agent 1 brings Team B's flag home for the winning capture.
    carrier = tuple(start.snapshot.agent_positions[1])
    capture = _view(
        _frame(
            start,
            scores=(2, 0),
            flag_carrier_slots=(None, 1),
            flag_positions=(homes[0], carrier),
        ),
        transition,
        _frame(successor, scores=(3, 0)),
        outcome=1,
        enemy_flag_was_captured_by_agent={1},
    )
    # Agent 1 returns Team A's dropped flag; Team B's runs out of time.
    returns = _view(
        _frame(start, dropped_flag_countdowns=(5, 1)),
        transition,
        successor,
        ally_flag_was_returned_by_agent={1},
        flag_was_automatically_returned_by_team={1},
    )
    batches = [
        build_visual_event_batch_v2(view) for view in (grab_and_fall, capture, returns)
    ]
    assert [_flag_rows_of(batch) for batch in batches] == [
        [("flag_picked_up", 0, 1), ("flag_dropped", 0, 1)],
        [("flag_captured", 1, 1), ("capture_the_flag_completed", None, None)],
        [("flag_returned", 1, 0), ("flag_automatically_returned", None, 1)],
    ]
    first = batches[0].flag_trajectories
    assert first is not None
    assert (first[1].transition_start.state, first[1].successor.state) == (
        "home",
        "dropped",
    )
    oracle = [
        _project_replay_incoming_summary_v1(batch, key_by_internal_slot=_KEYS)
        for batch in batches
    ]
    assert [
        [
            (
                event.event_kind,
                getattr(getattr(event, "agent_anchor", None), "presentation_key", None),
            )
            for event in summary.events
            if event.phase_rank >= 130
        ]
        for summary in oracle
    ] == [
        [("flag_picked_up", "key-a0"), ("flag_dropped", "key-a0")],
        [("flag_captured", "key-a1"), ("capture_the_flag_completed", None)],
        [("flag_returned", "key-a1"), ("flag_automatically_returned", None)],
    ]
    carried_start = oracle[1].flag_trajectories
    assert carried_start is not None
    assert carried_start[1].transition_start.carrier_presentation_key == _KEYS[1]
    # Every agent is seen at both ends; only membership matters to the filter.
    seen = dict.fromkeys(_KEYS)
    for batch in batches:
        trajectories = {row.global_slot: row for row in batch.agent_phase_trajectories}
        assert not any(
            _agent_pov_visual_event_is_authorized(
                event,
                recipient_global_slot=0,
                transition_start_agents_by_slot=seen,  # pyright: ignore[reportArgumentType]
                successor_agents_by_slot=seen,  # pyright: ignore[reportArgumentType]
                trajectory_by_internal_slot=trajectories,
            )
            for event in batch.events
            if event.phase_rank >= 130
        )


def test_flag_records_refuse_impossible_states(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
) -> None:
    home = (1.0, 5.0)
    with pytest.raises(ValueError, match="enemy team"):
        CaptureTheFlagFlagSceneV1(
            team_index=0,
            team_id=1,
            state="carried",
            position=home,
            home_position=home,
            carrier_global_slot=2,
            countdown_steps=0,
        )
    with pytest.raises(ValueError, match="home position"):
        CaptureTheFlagFlagSceneV1(
            team_index=0,
            team_id=1,
            state="home",
            position=(2.0, 5.0),
            home_position=home,
            carrier_global_slot=None,
            countdown_steps=0,
        )
    start, transition, successor = game
    batch = build_visual_event_batch_v2(
        _view(
            _frame(start, dropped_flag_countdowns=(0, 1)),
            transition,
            successor,
            flag_was_automatically_returned_by_team={1},
        )
    )
    with pytest.raises(ValueError, match="need the batch's flag trajectories"):
        dataclasses.replace(batch, flag_trajectories=None)


def test_flag_row_columns_match_core() -> None:
    assert (
        wire.OBJECTIVE_SLOT_ALLY_FLAG_HOME_V1,
        wire.OBJECTIVE_SLOT_ENEMY_FLAG_HOME_V1,
        wire.OBJECTIVE_SLOT_ALLY_FLAG_V1,
        wire.OBJECTIVE_SLOT_ENEMY_FLAG_V1,
        wire.OBJECTIVE_FEATURE_ACTIVE_V1,
        wire.OBJECTIVE_FEATURE_X_V1,
        wire.OBJECTIVE_FEATURE_Y_V1,
        wire.OBJECTIVE_FEATURE_INTERACTION_DISTANCE_V1,
        wire.OBJECTIVE_FEATURE_CONTROLLED_BY_ALLY_TEAM_V1,
        wire.OBJECTIVE_FEATURE_CONTROLLED_BY_ENEMY_TEAM_V1,
        wire.OBJECTIVE_FEATURE_STEPS_UNTIL_DROPPED_FLAG_RETURNS_V1,
        wire.OBJECTIVE_FEATURE_CARRIER_ROW_START_V1,
        wire.OBJECTIVE_FEATURE_CARRIER_ROW_STOP_V1,
    ) == (
        core_types.OBJECTIVE_SLOT_ALLY_FLAG_HOME,
        core_types.OBJECTIVE_SLOT_ENEMY_FLAG_HOME,
        core_types.OBJECTIVE_SLOT_ALLY_FLAG,
        core_types.OBJECTIVE_SLOT_ENEMY_FLAG,
        core_types.OBJECTIVE_FEATURE_IS_ACTIVE,
        core_types.OBJECTIVE_FEATURE_X,
        core_types.OBJECTIVE_FEATURE_Y,
        core_types.OBJECTIVE_FEATURE_INTERACTION_DISTANCE,
        core_types.OBJECTIVE_FEATURE_IS_CONTROLLED_BY_ALLY_TEAM,
        core_types.OBJECTIVE_FEATURE_IS_CONTROLLED_BY_ENEMY_TEAM,
        core_types.OBJECTIVE_FEATURE_STEPS_UNTIL_DROPPED_FLAG_AUTO_RETURNS,
        core_types.OBJECTIVE_FEATURE_UNIT_ROW_0_IS_FLAG_CARRIER,
        core_types.OBJECTIVE_FEATURE_UNIT_ROW_4_IS_FLAG_CARRIER + 1,
    )


def _row(
    point: tuple[float, ...],
    reach: float,
    *,
    ally: float = 0.0,
    enemy: float = 0.0,
    carrier: int | None = None,
) -> tuple[float, ...]:
    one_hot = tuple(1.0 if row == carrier else 0.0 for row in range(5))
    return (1.0, point[0], point[1], reach, ally, enemy, 0.0, *one_hot)


def _with_rows(
    frame: EvaluationFrameV4, rows: list[tuple[float, ...]]
) -> EvaluationFrameV4:
    # Only slot 0's own rows change; the snapshot is left as it is, so a POV
    # scene can only have read the flags from the actor's rows.
    data = frame.model_dump()
    observation = data["base_observation"]
    objective = [list(agent) for agent in observation["objective_features"]]
    objective[0][: len(rows)] = rows
    observation["objective_features"] = tuple(
        tuple(tuple(row) for row in agent) for agent in objective
    )
    return EvaluationFrameV4.model_validate(data)


def _pov_scene(
    start: EvaluationFrameV4,
    transition: EvaluationTransitionV2,
    successor: EvaluationFrameV4,
    frame_index: int,
) -> AuthorizedBattlefieldSceneV1:
    replay = build_replay_v5(
        _CONTEXT, [start, successor], [transition], runtime_provenance=_RUNTIME
    )
    return build_no_shared_obs_authorized_scene_v1(
        export_actor_pov_replay_v4(replay, global_slot=0).content,
        public_catalog=_CONTEXT.static_mechanics_catalog,
        authority_session_id="session-1",
        frame_index=frame_index,
    ).scene


def _flag_rows(
    *, own: tuple[float, ...], enemy: tuple[float, ...]
) -> list[tuple[float, ...]]:
    config = _CONTEXT.resolved_env_config
    homes = config.capture_the_flag_flag_home_positions
    capture = config.capture_the_flag_flag_capture_distance
    return [
        _row(homes[0], capture, ally=1.0),
        _row(homes[1], capture, enemy=1.0),
        own,
        enemy,
    ]


def test_agent_pov_reads_both_flags_from_its_own_rows(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
) -> None:
    start, transition, successor = game
    config = _CONTEXT.resolved_env_config
    homes = config.capture_the_flag_flag_home_positions
    touch = config.capture_the_flag_flag_interaction_distance
    positions = successor.snapshot.agent_positions

    def scene(
        own: tuple[float, ...], enemy: tuple[float, ...]
    ) -> AuthorizedBattlefieldSceneV1:
        rows = _flag_rows(own=own, enemy=enemy)
        return _pov_scene(start, transition, _with_rows(successor, rows), 1)

    # Core's own rows: both flags at home.
    from_core = _pov_scene(start, transition, successor, 1).capture_the_flag
    assert from_core is not None
    assert [flag.state for flag in from_core.flags] == ["home", "home"]
    blank = (0.0,) * len(_row(homes[0], touch))
    with pytest.raises(ValueError, match="no flag rows"):
        _pov_scene(start, transition, _with_rows(successor, [blank] * 4), 1)
    ownerless_home = _flag_rows(
        own=_row(homes[0], touch, ally=1.0), enemy=_row(homes[1], touch, enemy=1.0)
    )
    ownerless_home[0] = _row(homes[0], config.capture_the_flag_flag_capture_distance)
    with pytest.raises(ValueError, match="names its owner"):
        _pov_scene(start, transition, _with_rows(successor, ownerless_home), 1)
    # Its own flag is home; its teammate in ally row 1 carries the enemy flag.
    seen = scene(
        _row(homes[0], touch, ally=1.0), _row(positions[1], touch, ally=1.0, carrier=1)
    )
    teammate = next(
        agent
        for agent in seen.agents
        if agent.public_agent_id == _CONTEXT.roster[1].public_agent_id
    )
    assert seen.capture_the_flag is not None
    assert seen.capture_the_flag.return_steps is None
    assert [
        (flag.team_id, flag.state, flag.carrier_presentation_key)
        for flag in seen.capture_the_flag.flags
    ] == [(1, "home", None), (2, "carried", teammate.presentation_key)]
    # The far enemy carries its own flag away: carried, but unnamed.
    unseen = scene(
        _row(positions[5], touch, enemy=1.0), _row(homes[1], touch, enemy=1.0)
    )
    assert unseen.capture_the_flag is not None
    assert [
        (flag.state, flag.carrier_presentation_key)
        for flag in unseen.capture_the_flag.flags
    ] == [("carried", None), ("home", None)]
    with pytest.raises(ValueError, match="does not hold"):
        scene(
            _row(positions[5], touch, enemy=1.0, carrier=0),
            _row(homes[1], touch, enemy=1.0),
        )
    with pytest.raises(ValueError, match="decision 139"):
        scene(
            _row(homes[0], touch, ally=1.0, enemy=1.0), _row(homes[1], touch, enemy=1.0)
        )


def test_agent_pov_gets_its_own_flag_lines_and_no_oracle_flag_event(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
) -> None:
    start, transition, successor = game
    config = _CONTEXT.resolved_env_config
    homes = config.capture_the_flag_flag_home_positions
    touch = config.capture_the_flag_flag_interaction_distance
    carrier = tuple(successor.snapshot.agent_positions[1])
    at_home = _with_rows(
        start,
        _flag_rows(
            own=_row(homes[0], touch, ally=1.0), enemy=_row(homes[1], touch, enemy=1.0)
        ),
    )
    # Its teammate, agent 1, picks up the enemy flag from its home.
    picked_up = _with_rows(
        _frame(
            successor, flag_carrier_slots=(None, 1), flag_positions=(homes[0], carrier)
        ),
        _flag_rows(
            own=_row(homes[0], touch, ally=1.0),
            enemy=_row(carrier, touch, ally=1.0, carrier=1),
        ),
    )
    view = _view(at_home, transition, picked_up, flag_was_picked_up_by_agent={1})
    batch = build_visual_event_batch_v2(view)
    assert [row[0] for row in _flag_rows_of(batch)] == ["flag_picked_up"]
    assert isinstance(view.transition, EvaluationTransitionV2)
    start_scene, successor_scene = (
        _pov_scene(at_home, view.transition, picked_up, index) for index in (0, 1)
    )
    prefix = f"{batch.episode_id}:actor-pov:{_CONTEXT.roster[0].public_agent_id}"
    summary = build_agent_pov_visual_incoming_summary_v1(
        batch,
        transition_start_scene=start_scene,
        successor_scene=successor_scene,
        recipient_public_agent_id=_CONTEXT.roster[0].public_agent_id,
        incoming_recipient_transition_id=f"{prefix}:transition:0",
        incoming_start_recipient_frame_id=f"{prefix}:frame:0",
        incoming_successor_recipient_frame_id=f"{prefix}:frame:1",
    )
    teammate = next(
        agent
        for agent in successor_scene.agents
        if agent.public_agent_id == _CONTEXT.roster[1].public_agent_id
    )
    lines = [event for event in summary.events if event.event_kind == "flag_changed"]
    assert len(lines) == 1 and not any(
        event.event_kind.startswith("flag_") and event.event_kind != "flag_changed"
        for event in summary.events
    )
    line = lines[0]
    assert (line.flag_team_id, line.state_before, line.state_after) == (
        2,
        "home",
        "carried",
    )
    assert line.agent_anchor is not None
    assert (line.agent_anchor.presentation_key, line.agent_anchor.phase) == (
        teammate.presentation_key,
        "successor",
    )
    with pytest.raises(ValueError, match="need the flag trajectories"):
        dataclasses.replace(summary, flag_trajectories=None)


def test_a_flag_grabbed_and_dropped_again_gets_an_agent_pov_line() -> None:
    def dropped(
        position: tuple[float, float], countdown: int
    ) -> AuthorizedCaptureTheFlagFlagV1:
        return AuthorizedCaptureTheFlagFlagV1(
            team_id=1,
            state="dropped",
            position=position,
            home_position=(0.5, 5.0),
            carrier_presentation_key=None,
            carrier_public_agent_id=None,
            countdown_steps=countdown,
        )

    lying = dropped((7.5, 3.5), 5)
    changed = _agent_pov_flag_line_changed
    assert not changed(lying, dropped((7.5, 3.5), 4))  # it only counts down
    assert changed(lying, dropped((7.75, 3.5), 30))  # grabbed and dropped again
    assert changed(lying, dropped((7.5, 3.5), 30))  # same spot, a fresh countdown

    def line(
        before: CaptureTheFlagFlagState, after: CaptureTheFlagFlagState
    ) -> AgentPovVisualIncomingFlagChangedEventV1:
        return AgentPovVisualIncomingFlagChangedEventV1(
            event_id="e:visual-event:0000",
            ordinal=0,
            phase_rank=130,
            event_kind="flag_changed",
            flag_team_id=1,
            state_before=before,
            state_after=after,
            agent_anchor=None,
        )

    assert line("dropped", "dropped").state_after == "dropped"
    with pytest.raises(ValueError, match="dropped twice"):
        line("home", "home")


def test_a_whole_agent_pov_summary_accepts_a_grab_and_fall_from_the_ground(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
) -> None:
    start, transition, successor = game
    config = _CONTEXT.resolved_env_config
    homes = config.capture_the_flag_flag_home_positions
    touch = config.capture_the_flag_flag_interaction_distance
    full = config.capture_the_flag_dropped_flag_auto_return_steps
    here, there = (7.5, 3.5), (7.75, 3.5)

    def lying(frame: EvaluationFrameV4, spot: tuple[float, float], left: int) -> object:
        # Team B's flag lies dropped at spot with left ticks; Team A's is home.
        dropped = (1.0, *spot, touch, 0.0, 0.0, float(left), 0.0, 0.0, 0.0, 0.0, 0.0)
        return _with_rows(
            _frame(
                frame,
                flag_positions=(homes[0], spot),
                dropped_flag_countdowns=(0, left),
            ),
            _flag_rows(own=_row(homes[0], touch, ally=1.0), enemy=dropped),
        )

    before = cast(EvaluationFrameV4, lying(start, here, 5))
    after = cast(EvaluationFrameV4, lying(successor, there, full))
    view = _view(before, transition, after)
    batch = build_visual_event_batch_v2(view)
    assert isinstance(view.transition, EvaluationTransitionV2)
    start_scene, successor_scene = (
        _pov_scene(before, view.transition, after, index) for index in (0, 1)
    )
    prefix = f"{batch.episode_id}:actor-pov:{_CONTEXT.roster[0].public_agent_id}"
    summary = build_agent_pov_visual_incoming_summary_v1(
        batch,
        transition_start_scene=start_scene,
        successor_scene=successor_scene,
        recipient_public_agent_id=_CONTEXT.roster[0].public_agent_id,
        incoming_recipient_transition_id=f"{prefix}:transition:0",
        incoming_start_recipient_frame_id=f"{prefix}:frame:0",
        incoming_successor_recipient_frame_id=f"{prefix}:frame:1",
    )
    lines = [
        (event.flag_team_id, event.state_before, event.state_after, event.agent_anchor)
        for event in summary.events
        if event.event_kind == "flag_changed"
    ]
    # One line for the flag picked up and dropped again; its grabber stays unnamed.
    assert lines == [(2, "dropped", "dropped", None)]
