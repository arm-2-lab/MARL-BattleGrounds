"""Launch the shared Viewer from a replay path or a validated loaded bundle.

Browser launch uses installed package resources and a loopback server. Repository
launchers may supply their editable frontend directory. Code running from a source
checkout, which has no packaged page, uses that same checkout's page. Loading and
validation finish before the server starts, and so does every agent's Agent POV
work (``ReplayViewerService.prepare_agent_pov``), so the page opens with Agent POV
ready. This module does not build simulator scenarios.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING

from marl_battlegrounds.viewer.options import PlaybackOptions, validate_playback_options

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundle
    from marl_battlegrounds.viewer.replay_service import ReplayViewerService
    from marl_battlegrounds.viewer.server import HttpCoordinatorBinding


def launch_replay(
    path: str | Path | None,
    *,
    options: PlaybackOptions,
    loaded_bundle: LoadedReplayBundle | None = None,
    asset_root: Path | None = None,
) -> int:
    """Open a validated replay in the browser or optional static renderer.

    Parameters
    ----------
    path : str, pathlib.Path or None
        Saved replay to load once. Use None when passing ``loaded_bundle``.
    options : PlaybackOptions
        Shared resolved playback settings. Static mode needs an explicit frame.
    loaded_bundle : LoadedReplayBundle or None, optional
        Already validated replay for repository samples. It is not loaded again.
        Supply exactly one of path and this bundle.
    asset_root : pathlib.Path or None, optional
        Developer-only editable frontend directory. None uses package resources,
        keeping any temporary extraction alive for the complete server lifetime;
        when the code runs from a source checkout without packaged resources,
        None uses that checkout's ``web/visual_debugger`` (see
        ``checkout_browser_files``).

    Returns
    -------
    int
        The browser server or static renderer exit status.

    Raises
    ------
    ValueError
        If source selection, options, replay contents or authorization are invalid.
    OSError
        If replay/resources cannot be read or the local server cannot bind.
    ImportError
        If optional static plotting dependencies are unavailable.

    Notes
    -----
    Browser mode may open a browser and blocks until server shutdown. Static mode
    opens a Matplotlib window. Neither mode advances the environment or changes
    replay files. Existing requested numerical analysis remains lazy. Browser
    mode prints one line to standard error and builds every agent's Agent POV work
    (``ReplayViewerService.prepare_agent_pov``) before the server starts and the
    browser opens, the way the BattleClient compiles its first turn first, so
    every switch into Agent POV or between agents is quick from the start.
    That takes about 7 s for a 246-tick five-against-five replay on CPU and
    well under a second for a short scenario. If that work fails its
    validation, the launch prints a second line and goes on: the replay still
    opens, and the failing part builds on demand, where its error shows.
    """
    validate_playback_options(options)
    if (path is None) == (loaded_bundle is None):
        raise ValueError("Supply exactly one replay path or loaded bundle.")
    if options.static:
        from marl_battlegrounds.viewer.static import (
            run_static_replay_artifact_renderer,
            run_static_replay_renderer,
        )

        assert options.frame_index is not None
        if loaded_bundle is not None:
            return run_static_replay_artifact_renderer(
                replay=loaded_bundle.replay,
                frame_index=options.frame_index,
                show_ranges=options.ranges,
            )
        assert path is not None
        return run_static_replay_renderer(
            replay_path=Path(path),
            frame_index=options.frame_index,
            show_ranges=options.ranges,
        )

    bundle = loaded_bundle
    if bundle is None:
        from marl_battlegrounds.evaluation.replay_io import ReplayLoadError, load_replay

        assert path is not None
        try:
            bundle = load_replay(path)
        except ReplayLoadError as exc:
            raise ValueError(f"Replay could not be loaded: {exc}") from exc

    from marl_battlegrounds.viewer.replay_service import ReplayViewerService
    from marl_battlegrounds.viewer.server import serve_browser_debugger

    service = ReplayViewerService(
        bundle,
        initial_frame_index=0 if options.frame_index is None else options.frame_index,
        view_mode=options.view,
        pov_global_slot=options.pov_slot,
        preset="analysis",
        show_ranges=options.ranges,
        verbose=False,
    )
    coordinator = replay_http_binding(service)
    if asset_root is not None:
        resource_context = nullcontext(asset_root)
    else:
        packaged = files("marl_battlegrounds.viewer").joinpath("web")
        checkout = checkout_browser_files(packaged)
        resource_context = (
            nullcontext(checkout) if checkout is not None else as_file(packaged)
        )
    # Build Agent POV before the page opens, so no switch waits.
    print(
        "Replay Viewer: getting Agent POV ready for every agent...",
        file=sys.stderr,
        flush=True,
    )
    if not service.prepare_agent_pov():
        print(
            "Replay Viewer: Agent POV could not be prepared; it builds when opened.",
            file=sys.stderr,
            flush=True,
        )
    with resource_context as root:
        return serve_browser_debugger(
            service,
            asset_root=root,
            port=options.port,
            open_browser=not options.no_open,
            coordinator=coordinator,
        )


def checkout_browser_files(packaged: object) -> Path | None:
    """Find the browser page of the source checkout that is running this code.

    A built package ships the page inside ``marl_battlegrounds/viewer/web``. A
    source checkout (for example after ``pip install -e .``) has no such folder;
    its page lives in ``web/visual_debugger`` at the checkout's top. Public
    launchers call this first, so the BattleClient and the Replay Viewer also
    work from a checkout. The page always comes from the same checkout as the
    running code; no other folder is ever searched.

    Parameters
    ----------
    packaged : importlib.resources.abc.Traversable
        The package's ``web`` resource, ``files("marl_battlegrounds.viewer")``
        joined with ``"web"``. Typed as ``object`` because only a plain
        filesystem ``Path`` can belong to a checkout.

    Returns
    -------
    pathlib.Path or None
        None when ``packaged`` is an existing folder (an installed package) or
        not a filesystem path (for example inside a zip file). Otherwise the
        checkout's ``web/visual_debugger`` folder, but only when the folder three
        levels above ``src/marl_battlegrounds/viewer`` holds ``pyproject.toml``
        and ``web/visual_debugger/index.html``. Returns None in every other case,
        so the caller keeps its existing package lookup and its existing errors.

    Notes
    -----
    Reads only file existence; it opens, creates and changes nothing.
    """
    if not isinstance(packaged, Path) or packaged.is_dir():
        return None
    checkout = packaged.parents[3]
    page = checkout / "web" / "visual_debugger"
    if (checkout / "pyproject.toml").is_file() and (page / "index.html").is_file():
        return page
    return None


def replay_http_binding(service: ReplayViewerService) -> HttpCoordinatorBinding:
    """Bind one Viewer service to the shared replay HTTP routes.

    Parameters
    ----------
    service : ReplayViewerService
        An already validated Viewer service for one loaded replay.

    Returns
    -------
    HttpCoordinatorBinding
        A replay-mode binding whose operations forward to that same service:
        frame, command, timeline, presentation, metric report, metric analysis,
        metric catalog, episode details and the per-audience timeline marks
        (``GET /api/replay/timeline-marks``, from ``service.timeline_marks``).
        Both the standalone Replay Viewer and the BattleClient's Open Replay
        child use this binding, so both serve the timeline marks.

    Notes
    -----
    Building the binding does not open a socket, read assets or change the
    loaded replay. Launchers own those resources.
    """
    from marl_battlegrounds.viewer.replay_protocol import (
        ReplayApiErrorV1,
        ReplayCommandRequestV1,
    )
    from marl_battlegrounds.viewer.server import (
        REPLAY_HTTP_ROUTES,
        HttpCoordinatorBinding,
    )

    return HttpCoordinatorBinding(
        mode="replay",
        initial_show_ranges=service.show_ranges,
        routes=REPLAY_HTTP_ROUTES,
        request_model=ReplayCommandRequestV1,
        error_factory=ReplayApiErrorV1,
        current_frame=service.current_frame,
        apply_command=service.apply_command,
        current_timeline=service.current_timeline,
        current_presentation=service.current_presentation,
        current_metric_report=service.current_metric_report,
        metric_analysis=service.metric_analysis,
        metric_catalog=service.metric_catalog,
        episode_details=service.episode_details,
        timeline_marks=service.timeline_marks,
    )
