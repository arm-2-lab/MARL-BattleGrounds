"""Check how Capture the Flag games reach the clients' scenes and event lines.

Contracts: a Capture the Flag frame's researcher scene holds both flags (state,
position, home, carrier slot, countdown) and the rule settings, and its Oracle
scene names a carrier only by presentation key; a Team Deathmatch scene has no
flags (None). The visual event batch turns all six flag events into their
visual records, in recorded order, with each flag's start and successor
records; Oracle View keys every flag event agent and carrier. No flag event
reaches an Agent POV, even when every agent is seen. The records refuse a
carrier on the flag's own team, a home flag away from its home and flag events
without flag trajectories. Real games come once Core scores Capture the Flag
(his C1.3 and C1.4); these frames are hand-set, and the scene test uses frame 1
because today's Core leaves frame 0's Capture the Flag context rows at zero.
Host only, on CPU.
"""

import dataclasses

import jax
import pytest
from tests.helpers.evaluation_fixtures import neutral_action

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
from marl_battlegrounds.evaluation.models import (
    CaptureTheFlagTransitionFactsV1,
    CodeRevisionV2,
    EvaluationEpisodeContextV5,
    EvaluationFrameV4,
    EvaluationTransitionV2,
)
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.rendering.authorized_presentation import (
    _agent_pov_visual_event_is_authorized,  # pyright: ignore[reportPrivateUsage]
    _project_replay_incoming_summary_v1,  # pyright: ignore[reportPrivateUsage]
    build_oracle_authorized_scene_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    build_evaluation_battlefield_scene_v2,
    build_visual_event_batch_v2,
)
from marl_battlegrounds.rendering.scene import (
    CaptureTheFlagFlagSceneV1,
    VisualEventBatchV2,
)
from marl_battlegrounds.tasks import make_standard_capture_the_flag_config

_CONFIG = make_standard_capture_the_flag_config(
    map_id=42, team_a_roster=("mage", "warrior"), team_b_roster=("rogue",), max_steps=50
)
_KEYS = {0: "key-a0", 1: "key-a1", 5: "key-b0"}


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
            "runtime_provenance": capture_runtime_provenance("0.0.0").model_dump(
                mode="json"
            ),
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
    **agent_facts: set[int],
) -> EvaluationTransitionViewV1:
    flags: dict[str, object] = {
        name: (False,) * 10
        for name in CaptureTheFlagTransitionFactsV1.model_fields
        if name.endswith("_by_agent")
    }
    flags["flag_was_automatically_returned_by_team"] = (False, False)
    for name, slots in agent_facts.items():
        flags[name] = tuple(i in slots for i in range(len(flags[name])))  # pyright: ignore[reportArgumentType]
    facts = transition.facts.model_copy(
        update={
            "capture_the_flag_facts": CaptureTheFlagTransitionFactsV1.model_validate(
                {**flags, "outcome": outcome}
            )
        }
    )
    events = decode_evaluation_events_v1(_CONTEXT, start, facts, successor)
    return EvaluationTransitionViewV1(
        context=_CONTEXT,
        start_frame=start,
        transition=transition.model_copy(update={"facts": facts, "events": events}),
        successor_frame=successor,
    )


def _flag_rows(batch: VisualEventBatchV2) -> list[tuple[object, ...]]:
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
    assert [_flag_rows(batch) for batch in batches] == [
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
