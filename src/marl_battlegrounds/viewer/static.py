"""Display one saved replay frame with optional Matplotlib.

The debugger and replay launchers call these helpers for static mode. They open
one figure and wait for the display window; they do not start an HTTP server.
Without a display they refuse with a message naming the browser Replay Viewer,
instead of returning without showing anything.
Replay rendering reads saved evidence and never advances the simulator.
Matplotlib is imported only when a static display is requested.
"""

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from marl_battlegrounds.rendering import render_scene_geometry

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation._replays.replay import ReplayArtifactV1
    from marl_battlegrounds.evaluation._replays.replay_v2 import ReplayArtifactV2
    from marl_battlegrounds.evaluation._replays.replay_v3 import ReplayArtifactV3
    from marl_battlegrounds.evaluation._replays.replay_v4 import ReplayArtifactV4
    from marl_battlegrounds.evaluation._replays.replay_v5 import ReplayArtifactV5

# Every replay version this view opens (evaluated only by type checkers).
type _StaticReplay = (
    ReplayArtifactV1
    | ReplayArtifactV2
    | ReplayArtifactV3
    | ReplayArtifactV4
    | ReplayArtifactV5
)


class _PyplotLike(Protocol):
    """The small Matplotlib interface required to display or close a static figure."""

    def show(self) -> object:
        """Display open figures using Matplotlib's configured window backend."""
        ...

    def close(self, figure: object) -> object:
        """Close the supplied figure and release its display resources."""
        ...


def _load_pyplot() -> _PyplotLike:
    """Import Matplotlib on demand and check that it can open a window.

    Raises ImportError, naming the optional install, when Matplotlib is
    missing. Raises OSError when the session has no display: Matplotlib then
    falls back to a backend that only writes files (such as "agg"), and
    ``show`` would return at once and show nothing. The message points to the
    browser Replay Viewer instead.
    """
    try:
        pyplot = import_module("matplotlib.pyplot")
    except ImportError as exc:
        msg = (
            "Matplotlib is required for static Visual Debugger and Analyzer "
            "snapshots. "
            "Run pip install 'marl-battlegrounds[viz]'."
        )
        raise ImportError(msg) from exc
    backends = import_module("matplotlib.backends")
    backend = str(import_module("matplotlib").get_backend()).lower()
    file_only = backends.backend_registry.list_builtin(
        backends.BackendFilter.NON_INTERACTIVE
    )
    if backend in file_only:
        msg = (
            f"--static opens a desktop window, and this session has no display "
            f"(Matplotlib is using its {backend!r} backend). Open the browser "
            "Replay Viewer instead: python -m marl_battlegrounds replay FILE. On "
            "a remote machine, add --no-open --port 8765, forward the port with "
            "ssh -L 8765:127.0.0.1:8765 and open the printed 127.0.0.1 link."
        )
        raise OSError(msg)
    return cast(_PyplotLike, pyplot)


def run_static_replay_renderer(
    *,
    replay_path: Path,
    frame_index: int,
    show_ranges: bool,
) -> int:
    """Load a saved replay and display one recorded researcher frame.

    Parameters
    ----------
    replay_path : Path
        Replay file accepted by the shared replay loader.
    frame_index : int
        Zero-based recorded frame index; must be a Python integer in range.
    show_ranges : bool
        Whether the view includes interaction and observation ranges.

    Returns
    -------
    int
        Zero after the display call returns.

    Raises
    ------
    ValueError
        If the replay cannot be loaded or the frame index is invalid.
    ImportError
        If Matplotlib is unavailable.
    OSError
        If the session has no display, so no window can open; the message
        points to the browser Replay Viewer.

    Notes
    -----
    Reads the replay file and opens a window. It does not write files or run steps.
    """
    from marl_battlegrounds.evaluation.replay_io import (
        ReplayLoadError,
        load_replay,
    )

    if type(frame_index) is not int:
        raise ValueError("replay frame index must be a Python integer.")
    try:
        replay = load_replay(replay_path).replay
    except ReplayLoadError as exc:
        raise ValueError(f"Replay could not be loaded: {exc}") from exc
    return run_static_replay_artifact_renderer(
        replay=replay,
        frame_index=frame_index,
        show_ranges=show_ranges,
    )


def run_static_replay_artifact_renderer(
    *,
    replay: _StaticReplay,
    frame_index: int,
    show_ranges: bool,
) -> int:
    """Display one frame from an already validated replay object.

    Parameters
    ----------
    replay : ReplayArtifactV1, ReplayArtifactV2, ReplayArtifactV3, ReplayArtifactV4 or
        ReplayArtifactV5
        Exact supported replay model, with matching frame and transition sequences.
        Replay V4 records the Red Zone depth; this Matplotlib view does not draw
        the Red Zone strips. Replay V5 is any mode's replay, Capture the Flag's
        included; the view draws its bases and flags.
    frame_index : int
        Zero-based frame index, including zero for the reset frame.
    show_ranges : bool
        Whether to include configured range overlays.

    Returns
    -------
    int
        Zero after the Matplotlib display call returns.

    Raises
    ------
    TypeError
        If the replay is not an exact supported model type.
    ValueError
        If the frame index is not a Python integer in range.
    ImportError
        If Matplotlib is unavailable.
    OSError
        If the session has no display, so no window can open; the message
        points to the browser Replay Viewer.

    Notes
    -----
    Builds recorded status-source evidence and the authorized researcher projection.
    It opens a window without advancing simulator state or changing the replay.
    """
    from marl_battlegrounds.evaluation._metrics.metrics import (
        EvaluationTransitionViewV1,
    )
    from marl_battlegrounds.evaluation._replays.replay import ReplayArtifactV1
    from marl_battlegrounds.evaluation._replays.replay_v2 import ReplayArtifactV2
    from marl_battlegrounds.evaluation._replays.replay_v3 import ReplayArtifactV3
    from marl_battlegrounds.evaluation._replays.replay_v4 import ReplayArtifactV4
    from marl_battlegrounds.evaluation._replays.replay_v5 import ReplayArtifactV5
    from marl_battlegrounds.rendering.evaluation_adapter import (
        EvaluationScenePresentationStateV1,
        build_researcher_analyzer_projection_v2,
        build_status_source_evidence_index_v2,
    )

    if type(replay) not in (
        ReplayArtifactV1,
        ReplayArtifactV2,
        ReplayArtifactV3,
        ReplayArtifactV4,
        ReplayArtifactV5,
    ):
        raise TypeError("replay must be an exact supported replay artifact.")
    if type(frame_index) is not int:
        raise ValueError("replay frame index must be a Python integer.")
    if not 0 <= frame_index < len(replay.frames):
        raise ValueError(
            f"replay frame index must be in [0, {len(replay.frames) - 1}]."
        )
    frame = replay.frames[frame_index]
    transition_view = (
        None
        if frame_index == 0
        else EvaluationTransitionViewV1(
            context=replay.header.context,
            start_frame=replay.frames[frame_index - 1],
            transition=replay.transitions[frame_index - 1],
            successor_frame=frame,
        )
    )
    status_index = build_status_source_evidence_index_v2(
        replay.header.context,
        replay.frames,
        replay.transitions,
    )
    projection = build_researcher_analyzer_projection_v2(
        replay.header.context,
        frame,
        transition_view=transition_view,
        presentation=EvaluationScenePresentationStateV1(show_ranges=show_ranges),
        status_source_evidence_state=status_index.state_for_frame(frame_index),
    )
    pyplot = _load_pyplot()
    render_scene_geometry(
        projection.scene,
        event_batch=projection.incoming_events,
    )
    pyplot.show()
    return 0


__all__ = [
    "run_static_replay_artifact_renderer",
    "run_static_replay_renderer",
]
