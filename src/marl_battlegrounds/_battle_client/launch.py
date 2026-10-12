"""Launch public live combat from prepared installed content and browser assets.

The public call owns one local session until Exit or Ctrl-C. Task setup, method
execution, recording, server transport and drawing retain their existing owners.
Private integrations may explicitly supply public editable assets. The public
call uses package resources; code running from a source checkout, which has no
packaged page, uses that same checkout's page and never searches anywhere else.
The public BattleClient opens in Agent POV; private callers (the browser test
harness, the DevClient) pass ``view_mode="researcher"`` to ``_launch_setup`` to
keep Oracle View. Every launch opens with the reach rings shown (the owner's
1 October 2026 decision for the BattleClient and the DevClient; G hides them).
Every launch compiles the first turn before the page opens, so no Lock In
waits for compilation. A finished game saves its replay in the background, and
a Replay Viewer opened with Open Replay keeps serving after Exit until its own
Exit or Ctrl-C.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack, nullcontext
from importlib.resources import as_file, files
from pathlib import Path
from types import TracebackType
from typing import cast
from uuid import uuid4

from marl_battlegrounds._battle_client.control import (
    create_session,
    warm_compiled_turn,
)
from marl_battlegrounds._battle_client.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import team_controller_action_source
from marl_battlegrounds._battle_client.protocol import ViewMode
from marl_battlegrounds._battle_client.recording import (
    build_debugger_recording_specification_v1,
    recording_policy_execution_included,
)
from marl_battlegrounds._battle_client.recording_coordinator import (
    KeptReplayViewer,
    RecordingDebuggerCoordinator,
)
from marl_battlegrounds._battle_client.replay_recorder import DebuggerReplayRecorder
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import (
    BattleSetup,
    TeamControllers,
    prepare_battle_setup,
)
from marl_battlegrounds._battle_client.system_worker import (
    SystemWorker,
    remove_unplayed_evidence,
)
from marl_battlegrounds.evaluation.evaluation_conditions import OMITTED, Omitted
from marl_battlegrounds.evaluation.replay_io import (
    REPLAY_FILE_SUFFIX_V1,
    preflight_replay_destination,
)
from marl_battlegrounds.evaluation.revision import discover_code_revision_v2
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.tasks import AgentClassName
from marl_battlegrounds.viewer.launch import checkout_browser_files
from marl_battlegrounds.viewer.server import serve_browser_debugger


def _close_worker(worker: SystemWorker) -> None:
    """Drain caller-owned cleanup before the launch releases its resource leases.

    No callback runs under an HTTP or service lock. Retained cleanup failures
    propagate from drain; arbitrary caller-owned close methods are never called.
    """
    worker.close()
    while worker.drain():
        pass


def _launch_setup(
    setup: BattleSetup,
    *,
    port: int,
    open_browser: bool,
    output_dir: Path,
    capture_replay: bool,
    asset_root: Path | None = None,
    view_mode: ViewMode = "pov",
    replay_port: int = 0,
) -> int:
    """Serve prepared content while keeping its resources and recorder alive.

    setup contains validated scientific inputs. port and flags are checked by the
    caller. output_dir is an explicit destination, touched only when capture is
    enabled. replay_port is the Open Replay Viewer's loopback port, checked by
    the caller and passed to RecordingDebuggerCoordinator (0, the default, picks
    a free port at each start; unused when capture is off). asset_root is an
    explicit public frontend for internal source tools; None uses installed
    resources, or the running checkout's own web/visual_debugger when there are
    none (viewer.launch.checkout_browser_files).
    view_mode is the starting view: "pov" (the default, Agent POV, what the
    public BattleClient uses so a player sees only what the viewing team sees)
    or "researcher" (Oracle View, for the browser test harness and the
    DevClient). The player can switch views at any time. The session starts
    with the reach rings shown (show_ranges True); G hides or shows them.
    Before serving, it compiles the first turn once (control.warm_compiled_turn)
    and prints one line saying so, so the first Lock In does not wait for JAX.
    With capture on, a game that ends saves its replay in the background
    (DebuggerService save_in_background), so game over shows at once, and a
    Replay Viewer opened with Open Replay outlives Exit: after the live game,
    its methods and its assets are released, this call keeps serving that
    Viewer (KeptReplayViewer.serve_until_closed) until the Viewer's own Exit
    or Ctrl-C. Once the live game has closed, the call folder of any match
    nobody played (``<replay>.model-calls``, which then holds only its setup
    record) is deleted (system_worker.remove_unplayed_evidence); folders with
    a played turn or a model call stay. Return the existing server exit
    status. Setup, resource and server errors propagate, every created service
    is closed, and a kept
    Viewer is stopped when an error follows its release.
    """
    resolved = resolve_controllers(setup)
    controllers = resolved.controllers
    exit_code = 0
    kept_viewer: KeptReplayViewer | None = None

    def stop_kept_viewer_on_error(
        error_type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        """Stop a released Viewer if the launch then fails; never hide the error."""
        if error_type is not None and kept_viewer is not None:
            kept_viewer.close()
        return False

    with ExitStack() as lifetime:
        # Pushed first, so it runs last: after every later cleanup step.
        lifetime.push(stop_kept_viewer_on_error)
        for method in resolved.methods:
            if method.resource_scope is not None:
                lifetime.enter_context(method.resource_scope(False))
        revision = discover_code_revision_v2()
        session = create_session(
            setup.scenario,
            seed=setup.seed,
            evaluation_launch_specification=build_debugger_evaluation_launch_specification_v1(
                root_seed=setup.seed,
                code_revision=revision,
                capture_profile="evaluation_metric_complete"
                if capture_replay
                else "debug",
            ),
            controlled_global_slot=None,
            # Reach rings start shown; G toggles them (session.show_ranges).
            show_ranges=True,
            verbose_logging=False,
            team_a_controller=controllers[0],
            team_b_controller=controllers[1],
            execution_information_mode="shared_obs",
            systems=resolved.systems,
            borrow_systems=True,
            manual_slots=resolved.manual_slots,
            slot_labels=resolved.slot_labels,
        )
        # Compile the first turn before the page opens, so no Lock In waits.
        print("BattleClient: preparing the first turn (compiling once)...", flush=True)
        warm_compiled_turn(session)
        recorder = None
        if capture_replay:
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = preflight_replay_destination(
                output_dir / f"battle_{uuid4().hex}{REPLAY_FILE_SUFFIX_V1}"
            )
            recorder = DebuggerReplayRecorder(
                specification=build_debugger_recording_specification_v1(
                    action_source_kind=team_controller_action_source(
                        *controllers, manual_slots=resolved.manual_slots
                    ),
                    runtime_provenance=capture_runtime_provenance(
                        revision.package_version,
                        policy_execution_included=recording_policy_execution_included(
                            session.evaluation_context
                        ),
                    ),
                ),
                destination=destination,
                context=session.evaluation_context,
                initial_frame=session.current_evaluation_frame,
                scenario_name=setup.scenario.name,
            )
        worker = None
        if session.systems is not None:
            worker = SystemWorker(caller_thread=True)
            lifetime.callback(_close_worker, worker)
            worker.install(
                session,
                directory=None if recorder is None else recorder.evidence_directory,
            )
        service = DebuggerService(
            session,
            view_mode=view_mode,
            preset="analysis",
            include_stress=False,
            recorder=recorder,
            system_worker=worker,
            save_in_background=True,
        )
        try:
            if asset_root is not None:
                resources = nullcontext(asset_root)
            else:
                packaged = files("marl_battlegrounds.viewer").joinpath("web")
                checkout = checkout_browser_files(packaged)
                resources = (
                    nullcontext(checkout) if checkout is not None else as_file(packaged)
                )
            with resources as root:
                coordinator = (
                    None
                    if recorder is None
                    else RecordingDebuggerCoordinator(
                        service,
                        asset_root=root,
                        keep_viewer_after_exit=True,
                        replay_port=replay_port,
                    )
                )
                try:
                    exit_code = serve_browser_debugger(
                        service,
                        asset_root=root,
                        port=port,
                        open_browser=open_browser,
                        coordinator_router=None
                        if coordinator is None
                        else coordinator.router,
                        caller_drain=service.pump,
                        graceful_close=None
                        if coordinator is None
                        else coordinator.graceful_close,
                    )
                    if coordinator is not None:
                        kept_viewer = coordinator.release_kept_viewer()
                finally:
                    if coordinator is not None:
                        coordinator.close()
        finally:
            # Waits for a background save, so the replay file exists first.
            service.close()
    if recorder is not None:
        # Every worker has closed and any save has finished.
        remove_unplayed_evidence(recorder.evidence_directory)
    if kept_viewer is not None:
        kept_viewer.serve_until_closed()
    return exit_code


def battle_client(
    *,
    map_id: int | Omitted = OMITTED,
    scenario_id: int | Omitted = OMITTED,
    team_a: TeamControllers = "manual",
    team_b: TeamControllers | Omitted = OMITTED,
    team_a_roster: Sequence[AgentClassName] | Omitted = OMITTED,
    team_b_roster: Sequence[AgentClassName] | Omitted = OMITTED,
    seed: int | Omitted = OMITTED,
    max_steps: int | Omitted = OMITTED,
    score_threshold: int | Omitted = OMITTED,
    red_zone_depth: float | Omitted = OMITTED,
    spawn_mode: str | Omitted = OMITTED,
    port: int = 0,
    replay_port: int = 0,
    open_browser: bool = True,
    output_dir: str | Path = "./recordings",
    capture_replay: bool = True,
) -> int:
    """Play or watch a local match in the browser until Exit or Ctrl-C.

    Parameters
    ----------
    map_id, scenario_id : int or Omitted
        Supply exactly one. Packaged maps are 0 through 51; fixed scenarios are
        1 through 8. There is no silently chosen map.
    team_a, team_b : controller or ordered sequence
        Manual, a built-in name, checkpoint folder, portable model file, released
        model name (for example ``"MARL-BattleGrounds/tdm-season-0-qmix@final"``,
        downloaded once into Hugging Face's cache) or its ``hf://`` address,
        trusted module:function factory, or a live Policy/System; a list assigns
        one per roster slot. Ordinary teams default to Manual. A fixed scenario
        accepts only Team A.
    team_a_roster, team_b_roster : sequence of class names or Omitted
        Ordinary teams independently have one to five agents, including repeated
        classes. Defaults use mage, warrior, hunter, rogue, priest in that order.
    seed : int or Omitted
        Ordinary unsigned 32-bit seed, default 0.
    max_steps, score_threshold : int or Omitted
        Ordinary step limit and winning score, default 300 and 20.
    red_zone_depth : float or Omitted
        Ordinary Red Zone depth in map units, default 5.0. SharedObs is used.
    spawn_mode : {'default', 'swapped'} or Omitted
        Where each team starts on an ordinary map; default 'default': on every
        packaged map Team A starts on the left and Team B on the right.
        'swapped' exchanges the two teams' spawn ends, as
        ``evaluate(spawn_mode='swapped')`` does: Team A starts, respawns and
        has its Red Zone on the right. Teams, slots, rosters and controls stay
        the same, so you can stay Team A and play from either end. Restart
        keeps the ends. The saved replay records them as evaluate does
        (``marl_bgs.spawn_locations`` "0" or "1"). 'paired' alternates ends
        across many games, so one live game rejects it.
    port : int, default=0
        Loopback port. Zero asks the operating system for an available port.
    replay_port : int, default=0
        Loopback port of the Replay Viewer that Open Replay starts. Zero asks
        the operating system for an available port each time. Set it when the
        port must be known in advance, for example so Docker can publish it
        (the project's Docker image uses 8767). With a set port, opening a
        different replay first stops the Replay Viewer already on that port;
        if the new one then cannot start, the page says to try Open Replay
        again. A nonzero replay_port must differ from a nonzero port. Unused
        when capture_replay is False.
    open_browser : bool, default=True
        Open the printed local URL. Browser-open failure leaves the URL usable.
    output_dir : str or pathlib.Path, default='./recordings'
        Replay folder, relative to the caller's working directory. Created only
        with capture enabled. Each launch gets a separate replay destination.
    capture_replay : bool, default=True
        Capture from frame zero for later Save or Finish And Save. When a game
        ends, its replay saves by itself in the background: game over shows at
        once, the Recording pill reads "Saving Replay…" and then "Saved".
        Exit or Ctrl-C before the first tick writes no replay. False skips
        recorder and output-file work.

    Returns
    -------
    int
        Server exit status. The call blocks until Exit or Ctrl-C; closing or
        refreshing a tab does not silently destroy the session. If you opened
        the saved replay with Open Replay, Exit closes only the game: the call
        then prints the Replay Viewer's address and keeps serving it until you
        press Exit in the Replay Viewer or Ctrl-C in the terminal.

    Raises
    ------
    TypeError, ValueError
        Invalid setup or controller input. Fixed scenarios reject explicit Team B,
        map, roster, seed, rule or spawn_mode overrides even when equal to the
        fixed value.
    ValueError
        port or replay_port is not an integer from 0 through 65535, or both
        name the same nonzero port. This check runs before any setup.
    OSError
        A required asset/output path cannot be used or the server cannot bind.
    ImportError
        A selected method needs an optional dependency that is not installed.

    Notes
    -----
    Speed: before the browser opens, the call compiles the first turn once and
    prints "preparing the first turn (compiling once)". This takes a few seconds
    (measured with 7b+pool against TDM-Beta: about 4 to 7 s on CPU and about
    12 s on an RTX 5090). After that no Lock In compiles anything: every JAX
    method (built-in names, saved checkpoints and your own JAX Systems) runs
    as one compiled program per pair of methods, with weights, memory, inputs
    and keys as ordinary values, so each later turn takes about 0.02 s plus
    your method's own time. Restart reuses the compiled program. A method that
    runs on the host (an LLM, for example) is never called during this
    preparation; its outside effects and cost happen only on real turns, and
    its JAX opponent is still compiled once.

    The page opens in Agent POV: the arena and the BattleClient panels show only
    what the viewing team sees, and unseen enemies read "Unseen". The viewing
    team is the team of the Manual champion you last focused (inspecting a
    policy champion or an enemy never changes it); in a game with no Manual
    champion it follows the champion you look at. Oracle View, which shows
    everything, is one switch away and is marked during play. Open Replay always
    opens the Replay Viewer in Oracle View. Agent POV is a fairness aid for play,
    not a security boundary: the browser still receives the global researcher
    panels, which developer tools can read.

    Requires installed browser assets, no Node or Docker. Custom methods supply
    their own dependencies. Checkpoint folders use the training extra; portable
    model files and released models do not. A custom checkpoint
    format can be loaded by the researcher's Python code and passed as a System.
    Borrowed methods run on this call's original thread and context. Their
    descriptors and parameters are retained, so keep their external behavior and
    parameters fixed during the call. The client enters declared resource scopes;
    it never copies, serializes or arbitrarily closes caller-owned objects. Restart
    creates fresh episode memory without reloading unchanged methods. A failed
    opaque host callback can leave private state uncertain; inspect/save the last
    accepted game, then make a fresh launch before further play. Cancelling
    System work also fences the launch; the last accepted game remains saveable.
    Human-assisted play is not autonomous zero-shot coordination evaluation.
    """
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer from 0 through 65535.")
    if type(replay_port) is not int or not 0 <= replay_port <= 65535:
        raise ValueError("replay_port must be an integer from 0 through 65535.")
    if replay_port != 0 and replay_port == port:
        raise ValueError("replay_port must differ from port.")
    if type(open_browser) is not bool or type(capture_replay) is not bool:
        raise TypeError("open_browser and capture_replay must be booleans.")
    if not isinstance(cast(object, output_dir), (str, Path)):
        raise TypeError("output_dir must be a path or path string.")
    destination = Path(output_dir).expanduser().resolve()
    if capture_replay:
        for candidate in (destination, *destination.parents):
            if candidate.exists():
                if not candidate.is_dir():
                    raise ValueError("output_dir and its parents must be directories.")
                break
    setup = prepare_battle_setup(
        map_id=map_id,
        scenario_id=scenario_id,
        team_a=team_a,
        team_b=team_b,
        team_a_roster=team_a_roster,
        team_b_roster=team_b_roster,
        seed=seed,
        max_steps=max_steps,
        score_threshold=score_threshold,
        red_zone_depth=red_zone_depth,
        spawn_mode=spawn_mode,
    )
    return _launch_setup(
        setup,
        port=port,
        open_browser=open_browser,
        output_dir=destination,
        capture_replay=capture_replay,
        replay_port=replay_port,
    )
