"""Check the Replay Viewer's Capture the Flag match facts.

Contracts: build_match_summary_v1 summarizes a Capture the Flag frame (context
V5) as task mode 3, with the captures as scores, the captures to win as the
threshold, the result from the capture_the_flag_completed event and the
configured last tick as the episode limit. Without the start frame, the score
before the tick comes from the flag_captured event, credited to the carrier's
team (the captured flag's enemy), whether the events are the transition's own
or the visual batch's. The frames are hand-set, as in
tests/rendering/test_capture_the_flag_presentation.py, until Core scores
Capture the Flag (his C1.3 and C1.4). Host only, on CPU.
"""

from tests.rendering.test_capture_the_flag_presentation import (
    _CONTEXT,  # pyright: ignore[reportPrivateUsage]
    _frame,  # pyright: ignore[reportPrivateUsage]
    _view,  # pyright: ignore[reportPrivateUsage]
    game,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)

from marl_battlegrounds.evaluation.models import (
    EvaluationFrameV4,
    EvaluationTransitionV2,
)
from marl_battlegrounds.rendering.evaluation_adapter import build_visual_event_batch_v2
from marl_battlegrounds.viewer.match_summary import build_match_summary_v1


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
