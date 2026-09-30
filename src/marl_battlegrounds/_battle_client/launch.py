"""Launch public live combat from prepared installed content and browser assets.

The public call owns one local session until Exit or Ctrl-C. Task setup, method
execution, recording, server transport and drawing retain their existing owners.
Private integrations may explicitly supply public editable assets; the public
call uses package resources only and never searches a checkout for missing files.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack, nullcontext
from importlib.resources import as_file, files
from pathlib import Path
from typing import cast
from uuid import uuid4

from marl_battlegrounds._battle_client.control import create_session
from marl_battlegrounds._battle_client.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import team_controller_action_source
from marl_battlegrounds._battle_client.recording import (
    build_debugger_recording_specification_v1,
    recording_policy_execution_included,
)
from marl_battlegrounds._battle_client.recording_coordinator import (
    RecordingDebuggerCoordinator,
)
from marl_battlegrounds._battle_client.replay_recorder import DebuggerReplayRecorder
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import (
    BattleSetup,
    TeamControllers,
    prepare_battle_setup,
)
from marl_battlegrounds._battle_client.system_worker import SystemWorker
from marl_battlegrounds.evaluation.evaluation_conditions import OMITTED, Omitted
from marl_battlegrounds.evaluation.replay_io import (
    REPLAY_FILE_SUFFIX_V1,
    preflight_replay_destination,
)
from marl_battlegrounds.evaluation.revision import discover_code_revision_v2
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.tasks import AgentClassName
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
) -> int:
    """Serve prepared content while keeping its resources and recorder alive.

    setup contains validated scientific inputs. port and flags are checked by the
    caller. output_dir is an explicit destination, touched only when capture is
    enabled. asset_root is an explicit public frontend for internal source tools;
    None uses installed resources. Return the existing server exit status. Setup,
    resource and server errors propagate, and every created service is closed.
    """
    resolved = resolve_controllers(setup)
    controllers = resolved.controllers
    with ExitStack() as lifetime:
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
            show_ranges=False,
            verbose_logging=False,
            team_a_controller=controllers[0],
            team_b_controller=controllers[1],
            execution_information_mode="shared_obs",
            systems=resolved.systems,
            borrow_systems=True,
            manual_slots=resolved.manual_slots,
            slot_labels=resolved.slot_labels,
        )
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
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
            recorder=recorder,
            system_worker=worker,
        )
        try:
            resources = (
                nullcontext(asset_root)
                if asset_root is not None
                else as_file(files("marl_battlegrounds.viewer").joinpath("web"))
            )
            with resources as root:
                coordinator = (
                    None
                    if recorder is None
                    else RecordingDebuggerCoordinator(service, asset_root=root)
                )
                try:
                    return serve_browser_debugger(
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
                finally:
                    if coordinator is not None:
                        coordinator.close()
        finally:
            service.close()


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
    port: int = 0,
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
        Manual, a built-in name, checkpoint folder, trusted module:function
        factory, or a live Policy/System; a list assigns one per roster slot.
        Ordinary teams default to Manual. A fixed scenario accepts only Team A.
    team_a_roster, team_b_roster : sequence of class names or Omitted
        Ordinary teams independently have one to five agents, including repeated
        classes. Defaults use mage, warrior, hunter, rogue, priest in that order.
    seed : int or Omitted
        Ordinary unsigned 32-bit seed, default 0.
    max_steps, score_threshold : int or Omitted
        Ordinary step limit and winning score, default 300 and 20.
    red_zone_depth : float or Omitted
        Ordinary Red Zone depth in map units, default 5.0. SharedObs is used.
    port : int, default=0
        Loopback port. Zero asks the operating system for an available port.
    open_browser : bool, default=True
        Open the printed local URL. Browser-open failure leaves the URL usable.
    output_dir : str or pathlib.Path, default='./recordings'
        Replay folder, relative to the caller's working directory. Created only
        with capture enabled. Each launch gets a separate replay destination.
    capture_replay : bool, default=True
        Capture from frame zero for later Save or Finish And Save. False skips
        recorder and output-file work.

    Returns
    -------
    int
        Server exit status. The call blocks until Exit or Ctrl-C; closing or
        refreshing a tab does not silently destroy the session.

    Raises
    ------
    TypeError, ValueError
        Invalid setup or controller input. Fixed scenarios reject explicit Team B,
        map, roster, seed or rule overrides even when equal to the fixed value.
    OSError
        A required asset/output path cannot be used or the server cannot bind.
    ImportError
        A selected method needs an optional dependency that is not installed.

    Notes
    -----
    Requires installed browser assets, no Node or Docker. Custom methods supply
    their own dependencies. Checkpoints use the training extra. A custom checkpoint
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
    )
    return _launch_setup(
        setup,
        port=port,
        open_browser=open_browser,
        output_dir=destination,
        capture_replay=capture_replay,
    )
