"""Check Capture the Flag game records: their shapes, capture, checks and events.

Contracts: the Capture the Flag snapshot holds the shared snapshot fields plus
EnvState's four Capture the Flag fields, the leaf facts are Core's
CaptureTheFlagTransitionFacts field for field, and frame V4, transition V2 and
context V5 hold the same fields as frame V3, transition V1 and context V4, so a
field added to Core fails here. Binding table V5 names config 3, context 5,
snapshot 2, frame 4, facts 2 and transition 2, and context V5 pairs with frame
V4 and transition V2. A real game captured under context V5 records frame V4
and transition V2 with the flags at home and no carrier (None), and every
record reloads from its JSON. The snapshot refuses a carrier on the flag's own
team, a carried flag that counts down, and one away from its carrier. The
decoder turns hand-set flag facts into events in C1.C's order: a steal and a
drop by one agent on one tick, a capture that wins at the threshold, a return
and a timed return. It refuses captures, pickups, drops and timed returns that
disagree with their frames, and an outcome that disagrees with the captures.
The task authority rewards a Capture the Flag win +1 and a loss -1 and names
the end reason. Context rows follow C1.C (column 9 is 4, the captures in 14
and 15, the threshold in 18). A Capture the Flag config rebuilt from its
record, exchanged or not, equals the original. Host only, on CPU.
"""

import jax
import jax.numpy as jnp
import pytest
from tests.helpers.evaluation_fixtures import (
    current_evaluation_context,
    evaluation_env_config,
    neutral_action,
)

from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    CaptureTheFlagTransitionFacts,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.evaluation._recording.capture import (
    _capture_evaluation_transition_unit,  # pyright: ignore[reportPrivateUsage]
    _capture_initial_evaluation_frame,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.evaluation._recording.events import decode_evaluation_events_v1
from marl_battlegrounds.evaluation._recording.validation import (
    derive_and_validate_task_authority,
    validate_context_joined_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.catalog import (
    build_evaluation_episode_context,
    build_resolved_env_config_v3,
    reconstruct_env_config_v1,
)
from marl_battlegrounds.evaluation.models import (
    REQUIRED_SCHEMA_BINDINGS_V5,
    CaptureTheFlagTransitionFactsV1,
    EvaluationEpisodeContextV4,
    EvaluationEpisodeContextV5,
    EvaluationFrameV3,
    EvaluationFrameV4,
    EvaluationTransitionV1,
    EvaluationTransitionV2,
    GlobalAnalysisSnapshotCaptureTheFlagV2,
    GlobalAnalysisSnapshotV1,
    TransitionFactsCaptureTheFlagV2,
    TransitionFactsV1,
    evaluation_frame_type_for_context,
    evaluation_transition_type_for_context,
)

_HOMES = ((1.5, 6.0), (18.5, 6.0))
_RETURN_STEPS = 30
_TEAM_A, _TEAM_B = 0, 1


def _ctf_config() -> EnvConfig:
    return evaluation_env_config(max_steps=100)._replace(
        task_mode=3,
        capture_the_flag_score_threshold=3,
        capture_the_flag_flag_home_positions=jnp.asarray(_HOMES, dtype=jnp.float32),
        capture_the_flag_flag_interaction_distance=0.75,
        capture_the_flag_flag_capture_distance=1.25,
        capture_the_flag_dropped_flag_auto_return_steps=_RETURN_STEPS,
    )


def _ctf_context(config: EnvConfig) -> EvaluationEpisodeContextV5:
    tdm = current_evaluation_context(evaluation_env_config(max_steps=100))
    context = build_evaluation_episode_context(
        config=config,
        public_agent_id_by_global_slot=tuple(row.public_agent_id for row in tdm.roster),
        identity=tdm.identity,
        aggregation_keys=tdm.aggregation_keys,
        expected_horizon=100,
        policy_assignments=tdm.policy_assignments,
        seed_protocol=tdm.seed_protocol,
        capture_profile=tdm.capture_profile,
        execution_information_mode=tdm.execution_information_mode,
        actor_projection=tdm.actor_projection,
        critic_information_regime=tdm.critic_information_regime,
        canonical_reward_mode=tdm.canonical_reward_mode.model_copy(
            update={"identifier": "core-ctf"}
        ),
        shaping_configuration=tdm.shaping_configuration,
        code_revision=tdm.code_revision,
    )
    assert type(context) is EvaluationEpisodeContextV5
    return context


_CONTEXT = _ctf_context(_ctf_config())


@pytest.fixture(scope="module")
def game() -> tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4]:
    state, observation, mask, _ = reset(_ctf_config(), jax.random.PRNGKey(0))
    start = _capture_initial_evaluation_frame(_CONTEXT, state, observation, mask)
    state, observation, reward, done, mask, info = step(
        _ctf_config(), state, mask, neutral_action(), jax.random.PRNGKey(1)
    )
    transition, successor = _capture_evaluation_transition_unit(
        _CONTEXT, start, state, observation, mask, info.transition_facts, reward, done
    )
    assert type(start) is EvaluationFrameV4 and type(successor) is EvaluationFrameV4
    assert type(transition) is EvaluationTransitionV2
    return start, transition, successor


def test_records_mirror_core_and_their_twins() -> None:
    shared = set(GlobalAnalysisSnapshotV1.model_fields) - {"team_deathmatch_scores"}
    ctf_state = {
        name for name in EnvState._fields if name.startswith("capture_the_flag")
    }
    assert (
        set(GlobalAnalysisSnapshotCaptureTheFlagV2.model_fields) == shared | ctf_state
    )
    assert (
        tuple(CaptureTheFlagTransitionFactsV1.model_fields)
        == CaptureTheFlagTransitionFacts._fields
    )
    assert set(TransitionFactsCaptureTheFlagV2.model_fields) == (
        set(TransitionFactsV1.model_fields) - {"team_deathmatch_facts"}
    ) | {"capture_the_flag_facts"}
    for new, old in (
        (EvaluationFrameV4, EvaluationFrameV3),
        (EvaluationTransitionV2, EvaluationTransitionV1),
        (EvaluationEpisodeContextV5, EvaluationEpisodeContextV4),
    ):
        assert set(new.model_fields) == set(old.model_fields)
    assert dict(REQUIRED_SCHEMA_BINDINGS_V5) == {
        "marl_battlegrounds.evaluation.static_mechanics_catalog": 1,
        "marl_battlegrounds.evaluation.resolved_env_config": 3,
        "marl_battlegrounds.evaluation.episode_context": 5,
        "marl_battlegrounds.evaluation.global_analysis_snapshot": 2,
        "marl_battlegrounds.evaluation.frame": 4,
        "marl_battlegrounds.evaluation.transition_facts": 2,
        "marl_battlegrounds.evaluation.event": 1,
        "marl_battlegrounds.evaluation.transition": 2,
    }
    assert evaluation_frame_type_for_context(_CONTEXT) is EvaluationFrameV4
    assert evaluation_transition_type_for_context(_CONTEXT) is EvaluationTransitionV2


def test_a_real_game_records_its_flags_and_reloads(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
) -> None:
    start, transition, successor = game
    snapshot = start.snapshot
    assert snapshot.capture_the_flag_flag_positions == _HOMES
    assert snapshot.capture_the_flag_flag_carrier_slots == (None, None)
    assert snapshot.capture_the_flag_scores == (0, 0)
    for record in (_CONTEXT, start, transition, successor):
        assert type(record).model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"capture_the_flag_flag_carrier_slots": (0, None)}, "enemy team"),
        (
            {
                "capture_the_flag_flag_carrier_slots": (5, None),
                "capture_the_flag_dropped_flag_countdowns": (4, 0),
            },
            "cannot count down",
        ),
        ({"capture_the_flag_flag_carrier_slots": (5, None)}, "carrier's position"),
    ],
)
def test_the_snapshot_refuses_impossible_flag_states(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
    change: dict[str, object],
    message: str,
) -> None:
    payload = {**game[0].snapshot.model_dump(), **change}
    with pytest.raises(ValueError, match=message):
        GlobalAnalysisSnapshotCaptureTheFlagV2.model_validate(payload)


def _frame(frame: EvaluationFrameV4, **flags: object) -> EvaluationFrameV4:
    snapshot = frame.snapshot.model_dump()
    for name, value in flags.items():
        snapshot[f"capture_the_flag_{name}"] = value
    return EvaluationFrameV4.model_validate(
        {**frame.model_dump(), "snapshot": snapshot}
    )


def _facts(
    transition: EvaluationTransitionV2, outcome: int = 0, **agent_facts: set[int]
) -> TransitionFactsCaptureTheFlagV2:
    flags: dict[str, object] = {
        name: (False,) * 10
        for name in CaptureTheFlagTransitionFactsV1.model_fields
        if name.endswith("_by_agent")
    }
    flags["flag_was_automatically_returned_by_team"] = (False, False)
    for name, slots in agent_facts.items():
        length = 2 if name.endswith("_by_team") else 10
        flags[name] = tuple(i in slots for i in range(length))
    return transition.facts.model_copy(
        update={
            "capture_the_flag_facts": CaptureTheFlagTransitionFactsV1.model_validate(
                {**flags, "outcome": outcome}
            )
        }
    )


def _flag_events(
    start: EvaluationFrameV4,
    facts: TransitionFactsCaptureTheFlagV2,
    successor: EvaluationFrameV4,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for event in decode_evaluation_events_v1(_CONTEXT, start, facts, successor):
        fields = event.model_dump()
        if fields["phase_rank"] < 130:
            continue
        rows.append(
            (
                fields["event_type"],
                fields.get("agent_global_slot"),
                fields.get("flag_team_index"),
                *(
                    fields[name]
                    for name in ("picked_up_from", "successor_score", "outcome")
                    if name in fields
                ),
            )
        )
    return rows


def test_flag_facts_decode_into_ordered_events(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
) -> None:
    start, transition, successor = game
    positions = start.snapshot.agent_positions
    # Agent 0 steals Team B's flag from its home and dies on the same tick.
    stolen = _frame(successor, dropped_flag_countdowns=(0, _RETURN_STEPS))
    facts = _facts(
        transition, flag_was_picked_up_by_agent={0}, flag_was_dropped_by_agent={0}
    )
    assert _flag_events(start, facts, stolen) == [
        ("flag_picked_up", 0, _TEAM_B, "home"),
        ("flag_dropped", 0, _TEAM_B),
    ]
    # Agent 1 brings Team B's flag home for the third, winning capture.
    carried = _frame(
        start,
        scores=(2, 0),
        flag_carrier_slots=(None, 1),
        flag_positions=(_HOMES[0], tuple(positions[1])),
    )
    won = _frame(successor, scores=(3, 0))
    facts = _facts(transition, outcome=1, enemy_flag_was_captured_by_agent={1})
    assert _flag_events(carried, facts, won) == [
        ("flag_captured", 1, _TEAM_B, 3),
        ("capture_the_flag_completed", None, None, "team_a_win"),
    ]
    rewards = (1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0)
    assert derive_and_validate_task_authority(
        _CONTEXT, carried, facts, won, rewards, True, False
    ) == ((1.0, -1.0), "capture_the_flag_score_threshold")
    with pytest.raises(ValueError, match="reward must match"):
        derive_and_validate_task_authority(
            _CONTEXT, carried, facts, won, (0.0,) * 10, True, False
        )
    # Agent 2 returns Team A's dropped flag; Team B's runs out of time.
    dropped = _frame(start, dropped_flag_countdowns=(5, 1))
    facts = _facts(
        transition,
        ally_flag_was_returned_by_agent={2},
        flag_was_automatically_returned_by_team={1},
    )
    assert _flag_events(dropped, facts, successor) == [
        ("flag_returned", 2, _TEAM_A),
        ("flag_automatically_returned", None, _TEAM_B),
    ]


@pytest.mark.parametrize(
    ("start_flags", "successor_flags", "agent_facts", "outcome", "message"),
    [
        ({}, {}, {"enemy_flag_was_captured_by_agent": {1}}, 0, "captures must rise"),
        ({}, {}, {"flag_was_picked_up_by_agent": {0}}, 0, "carried after"),
        (
            {},
            {"dropped_flag_countdowns": (0, _RETURN_STEPS - 1)},
            {"flag_was_picked_up_by_agent": {0}, "flag_was_dropped_by_agent": {0}},
            0,
            "fresh countdown",
        ),
        (
            {"dropped_flag_countdowns": (0, 2)},
            {},
            {"flag_was_automatically_returned_by_team": {1}},
            0,
            "countdown 1",
        ),
        ({"scores": (2, 0)}, {"scores": (3, 0)}, {}, 0, "captures must rise"),
        ({}, {}, {}, 3, "outcome fact conflicts"),
    ],
)
def test_facts_that_disagree_with_their_frames_are_refused(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
    start_flags: dict[str, object],
    successor_flags: dict[str, object],
    agent_facts: dict[str, set[int]],
    outcome: int,
    message: str,
) -> None:
    start, transition, successor = game
    with pytest.raises(ValueError, match=message):
        decode_evaluation_events_v1(
            _CONTEXT,
            _frame(start, **start_flags),
            _facts(transition, outcome, **agent_facts),
            _frame(successor, **successor_flags),
        )


def test_context_rows_follow_the_capture_the_flag_contract(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],
) -> None:
    frame = _frame(game[0], scores=(2, 1))
    rows: list[tuple[float, ...]] = []
    for roster_row, features in zip(
        _CONTEXT.roster, frame.base_observation.context_features, strict=True
    ):
        row = list(features)
        if roster_row.configured_active:
            own = 2.0 if roster_row.configured_team_id == 1 else 1.0
            row[9], row[14], row[15], row[18] = 4.0, own, 3.0 - own, 3.0
        rows.append(tuple(row))

    def with_rows(rows: list[tuple[float, ...]]) -> EvaluationFrameV4:
        observation = {
            **frame.base_observation.model_dump(),
            "context_features": tuple(rows),
        }
        return EvaluationFrameV4.model_validate(
            {**frame.model_dump(), "base_observation": observation}
        )

    validate_context_joined_evaluation_frame_v1(_CONTEXT, with_rows(rows))
    rows[0] = (*rows[0][:18], 5.0, *rows[0][19:])
    with pytest.raises(ValueError, match="conflicts with resolved task authority"):
        validate_context_joined_evaluation_frame_v1(_CONTEXT, with_rows(rows))


@pytest.mark.parametrize("exchanged", [False, True])
def test_a_capture_the_flag_config_rebuilds_from_its_record(exchanged: bool) -> None:
    config = _ctf_config()
    if exchanged:
        config = config._replace(
            team_spawn_pad_positions=config.team_spawn_pad_positions[::-1],
            capture_the_flag_flag_home_positions=(
                config.capture_the_flag_flag_home_positions[::-1]
            ),
        )
    rebuilt = reconstruct_env_config_v1(_ctf_context(config))
    assert build_resolved_env_config_v3(rebuilt) == build_resolved_env_config_v3(config)
    assert (
        rebuilt.team_deathmatch_score_threshold,
        rebuilt.team_deathmatch_red_zone_depth,
    ) == (0, 0.0)
