"""Check the Replay Viewer's Capture the Flag games and match facts.

Contracts: a NoSharedObs Capture the Flag replay V5 opens in the Replay Viewer
in Oracle View and in Agent POV, at frame 0 and later, and both views carry
both flags in their scene; a SharedObs one is refused with a plain message
until its player views exist. build_match_summary_v1 summarizes a Capture
the Flag frame (context V5) as task mode 3, with the captures as scores, the
captures to win as the threshold, the result from the
capture_the_flag_completed event and the configured last tick as the episode
limit. Without the start frame, the score
before the tick comes from the flag_captured event, credited to the carrier's
team (the captured flag's enemy), whether the events are the transition's own
or the visual batch's. The frames are hand-set, as in
tests/rendering/test_capture_the_flag_presentation.py, until Core scores
Capture the Flag (his C1.3 and C1.4). Host only, on CPU.
"""

import jax
import pytest
from tests.helpers.evaluation_fixtures import neutral_action
from tests.rendering.test_capture_the_flag_presentation import (
    _CONFIG,  # pyright: ignore[reportPrivateUsage]
    _CONTEXT,  # pyright: ignore[reportPrivateUsage]
    _RUNTIME,  # pyright: ignore[reportPrivateUsage]
    _frame,  # pyright: ignore[reportPrivateUsage]
    _view,  # pyright: ignore[reportPrivateUsage]
    game,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)

from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.evaluation._recording.capture import (
    _capture_evaluation_transition_unit,  # pyright: ignore[reportPrivateUsage]
    _capture_initial_evaluation_frame,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.evaluation._replays.replay_v5 import build_replay_v5
from marl_battlegrounds.evaluation.models import (
    EvaluationFrameV4,
    EvaluationTransitionV2,
)
from marl_battlegrounds.evaluation.replay_io import LoadedReplay
from marl_battlegrounds.rendering.evaluation_adapter import build_visual_event_batch_v2
from marl_battlegrounds.viewer.match_summary import build_match_summary_v1
from marl_battlegrounds.viewer.presentation_protocol import (
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplayOracleAuthorizedPresentationFrameV1,
)
from marl_battlegrounds.viewer.replay_service import ReplayViewerService


def test_a_winning_capture_summarizes_as_capture_the_flag(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],  # noqa: F811
) -> None:
    start, transition, successor = game
    homes = _CONTEXT.resolved_env_config.capture_the_flag_flag_home_positions
    carrying = _frame(
        start,
        scores=(2, 0),
        flag_carrier_slots=(None, 1),
        flag_positions=(homes[0], tuple(start.snapshot.agent_positions[1])),
    )
    won = _frame(successor, scores=(3, 0))
    view = _view(
        carrying, transition, won, outcome=1, enemy_flag_was_captured_by_agent={1}
    )
    summary = build_match_summary_v1(
        _CONTEXT, won, view.transition.events, transition_start_frame=carrying
    )
    assert (
        summary.task_mode,
        summary.score_threshold,
        summary.scores,
        summary.previous_scores,
        summary.outcome,
        summary.episode_limit,
    ) == (3, 3, (3, 0), (2, 0), "team_a_win", 50)
    for events in (view.transition.events, build_visual_event_batch_v2(view).events):
        assert build_match_summary_v1(_CONTEXT, won, events).previous_scores == (2, 0)


def _with_c1c_rows(frame: EvaluationFrameV4) -> EvaluationFrameV4:
    # Fill each active actor's context columns 6-18 and flag rows 0-3 the way
    # C1.C defines them, with both flags at home, as Core will from C1.4 on.
    config = _CONTEXT.resolved_env_config
    homes = config.capture_the_flag_flag_home_positions
    capture = config.capture_the_flag_flag_capture_distance
    touch = config.capture_the_flag_flag_interaction_distance
    data = frame.model_dump()
    observation = data["base_observation"]
    contexts = [list(row) for row in observation["context_features"]]
    objectives = [list(map(list, agent)) for agent in observation["objective_features"]]
    scores = frame.snapshot.capture_the_flag_scores
    for slot, roster in enumerate(_CONTEXT.roster):
        if not roster.configured_active:
            continue
        team = roster.configured_team_id
        own, enemy = float(scores[team - 1]), float(scores[2 - team])
        threshold = float(config.capture_the_flag_score_threshold)
        # Columns 6-18: is TDM, KOTH, CTF; active objective rows; TDM and KOTH
        # scores; CTF scores (own, enemy); TDM, KOTH and CTF thresholds.
        contexts[slot][6:19] = [
            *(0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0),
            *(own, enemy, 0.0, 0.0, threshold),
        ]
        rows = (
            (homes[team - 1], capture, 0.0, 0.0),
            (homes[2 - team], capture, 0.0, 0.0),
            (homes[team - 1], touch, 1.0, 0.0),
            (homes[2 - team], touch, 0.0, 1.0),
        )
        for row, (point, reach, ally, enemy_bit) in enumerate(rows):
            objectives[slot][row] = [1.0, point[0], point[1], reach, ally, enemy_bit]
            objectives[slot][row] += [0.0] * 6
    observation["context_features"] = tuple(tuple(row) for row in contexts)
    observation["objective_features"] = tuple(
        tuple(tuple(row) for row in agent) for agent in objectives
    )
    return EvaluationFrameV4.model_validate(data)


def test_a_capture_the_flag_replay_opens_in_oracle_view_and_agent_pov() -> None:
    state, observation, mask, _ = reset(_CONFIG, jax.random.PRNGKey(0))
    initial = _capture_initial_evaluation_frame(_CONTEXT, state, observation, mask)
    assert isinstance(initial, EvaluationFrameV4)
    frames = [_with_c1c_rows(initial)]
    transitions: list[EvaluationTransitionV2] = []
    for index in range(3):
        state, observation, reward, done, mask, info = step(
            _CONFIG, state, mask, neutral_action(), jax.random.PRNGKey(index + 1)
        )
        transition, frame = _capture_evaluation_transition_unit(
            _CONTEXT,
            frames[-1],
            state,
            observation,
            mask,
            info.transition_facts,
            reward,
            done,
        )
        assert isinstance(frame, EvaluationFrameV4)
        assert isinstance(transition, EvaluationTransitionV2)
        frames.append(_with_c1c_rows(frame))
        transitions.append(transition)
    replay = build_replay_v5(_CONTEXT, frames, transitions, runtime_provenance=_RUNTIME)
    for frame_index in (0, 2):
        oracle = ReplayViewerService(
            LoadedReplay(replay), initial_frame_index=frame_index
        ).current_presentation()
        pov = ReplayViewerService(
            LoadedReplay(replay),
            initial_frame_index=frame_index,
            view_mode="pov",
            pov_global_slot=0,
        ).current_presentation()
        assert isinstance(oracle.payload, ReplayOracleAuthorizedPresentationFrameV1)
        assert isinstance(pov.payload, ReplayNoSharedObsAuthorizedPresentationFrameV1)
        for scene in (
            oracle.payload.current_endpoint.scene,
            pov.payload.current_endpoint.parts.scene,
        ):
            assert scene.capture_the_flag is not None
            assert [flag.state for flag in scene.capture_the_flag.flags] == [
                "home",
                "home",
            ]
    context = replay.header.context.model_copy(
        update={"execution_information_mode": "shared_obs"}
    )
    shared = replay.model_copy(
        update={"header": replay.header.model_copy(update={"context": context})}
    )
    with pytest.raises(ValueError, match="SharedObs Capture the Flag"):
        ReplayViewerService(LoadedReplay(shared))
