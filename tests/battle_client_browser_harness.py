"""Run public combat for browser tests without private authoring files.

The fixture uses the package's live session, recording and HTTP owners. It
accepts the old browser tests' setup flags while the public Python launch API is
qualified separately. Its explicit editable frontend path is test setup, never
an installed-package fallback. Every route, the controller fixtures included,
opens in Oracle View unless --view pov asks for Agent POV, so browser specs keep
their Oracle starting view although the public BattleClient opens in Agent POV.
"""

from __future__ import annotations

# This source-only harness supplies explicit test assets to the shared launcher.
# pyright: reportPrivateUsage=false
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--record-replay", type=Path)
    parser.add_argument("--controlled-slot", type=int)
    parser.add_argument(
        "--view", choices=("oracle", "researcher", "pov"), default="oracle"
    )
    parser.add_argument(
        "--ranges", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--team-a-controller", default="manual")
    parser.add_argument("--team-b-controller", default="manual")
    parser.add_argument(
        "--execution-information-mode",
        choices=("shared_obs", "no_shared_obs"),
        default="shared_obs",
    )
    parser.add_argument("--preset", default="analysis")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--offer-system", action="append", default=[])
    parser.add_argument(
        "--controller-fixture",
        choices=("mixed", "autonomous", "fixed_manual", "finishing"),
    )
    parser.add_argument("--finish-release", type=Path)
    options = parser.parse_args(argv)
    if options.controller_fixture is not None:
        from tests.visual_debugger_system_browser_harness import make_system

        from marl_battlegrounds._battle_client.launch import _launch_setup
        from marl_battlegrounds._battle_client.setup import prepare_battle_setup

        if options.controller_fixture == "finishing":
            import time

            from marl_battlegrounds._battle_client.system_worker import (
                SystemInstallation,
            )

            if options.finish_release is None or options.record_replay is None:
                parser.error("Finishing fixture needs recording and release paths")
            release = options.finish_release
            original_write = SystemInstallation.write_details

            def write_details(self: SystemInstallation) -> None:
                if self.metadata.get("accepted_turns") == 1:
                    while not release.exists():
                        time.sleep(0.01)
                original_write(self)

            SystemInstallation.write_details = write_details

        method = make_system()
        if options.controller_fixture == "fixed_manual":
            setup = prepare_battle_setup(scenario_id=6, team_a="manual")
        else:
            team_a = (
                ["manual", "manual", method, method, method]
                if options.controller_fixture == "mixed"
                else method
            )
            setup = prepare_battle_setup(
                map_id=0, team_a=team_a, team_b="random", max_steps=8
            )
        return _launch_setup(
            setup,
            port=options.port,
            open_browser=not options.no_open,
            output_dir=options.record_replay.parent
            if options.record_replay is not None
            else Path("unused-browser-recordings"),
            capture_replay=options.record_replay is not None,
            asset_root=_REPOSITORY_ROOT / "web" / "visual_debugger",
            view_mode="pov" if options.view == "pov" else "researcher",
        )

    from scripts.dev.visual_debugger.scenarios import get_scenario

    from marl_battlegrounds._battle_client.control import create_session
    from marl_battlegrounds._battle_client.evaluation_bridge import (
        DebuggerActionSourceKindV1,
        build_debugger_evaluation_launch_specification_v1,
    )
    from marl_battlegrounds._battle_client.recording import (
        build_debugger_recording_specification_v1,
        recording_action_source,
        recording_policy_execution_included,
    )
    from marl_battlegrounds._battle_client.recording_coordinator import (
        RecordingDebuggerCoordinator,
    )
    from marl_battlegrounds._battle_client.replay_recorder import DebuggerReplayRecorder
    from marl_battlegrounds._battle_client.service import DebuggerService
    from marl_battlegrounds.evaluation.replay_io import preflight_replay_destination
    from marl_battlegrounds.evaluation.revision import discover_code_revision_v1
    from marl_battlegrounds.evaluation.runtime_provenance import (
        capture_debugger_runtime_provenance_v1,
    )
    from marl_battlegrounds.viewer.server import serve_browser_debugger

    revision = discover_code_revision_v1(_REPOSITORY_ROOT)
    destination = (
        None
        if options.record_replay is None
        else preflight_replay_destination(options.record_replay)
    )
    session = create_session(
        get_scenario("arena_5v5"),
        seed=options.seed,
        evaluation_launch_specification=build_debugger_evaluation_launch_specification_v1(
            root_seed=options.seed,
            code_revision=revision,
            capture_profile="debug"
            if destination is None
            else "evaluation_metric_complete",
        ),
        controlled_global_slot=options.controlled_slot,
        show_ranges=options.ranges,
        verbose_logging=False,
        team_a_controller=options.team_a_controller,
        team_b_controller=options.team_b_controller,
        execution_information_mode=options.execution_information_mode,
    )
    recorder = None
    if destination is not None:
        recorder = DebuggerReplayRecorder(
            specification=build_debugger_recording_specification_v1(
                action_source_kind=cast(
                    DebuggerActionSourceKindV1,
                    recording_action_source(session.evaluation_context),
                ),
                runtime_provenance=capture_debugger_runtime_provenance_v1(
                    revision,
                    policy_execution_included=recording_policy_execution_included(
                        session.evaluation_context
                    ),
                ),
            ),
            destination=destination,
            context=session.evaluation_context,
            initial_frame=session.current_evaluation_frame,
            scenario_name=session.scenario.name
            if session.evaluation_context.identity.scenario is not None
            else None,
        )
    service = DebuggerService(
        session,
        view_mode="pov" if options.view == "pov" else "researcher",
        preset="analysis",
        include_stress=False,
        recorder=recorder,
        offered_systems=dict(value.split("=", 1) for value in options.offer_system),
    )
    coordinator = (
        None
        if recorder is None
        else RecordingDebuggerCoordinator(
            service, asset_root=_REPOSITORY_ROOT / "web" / "visual_debugger"
        )
    )
    try:
        return serve_browser_debugger(
            service,
            asset_root=_REPOSITORY_ROOT / "web" / "visual_debugger",
            port=options.port,
            open_browser=not options.no_open,
            coordinator_router=None if coordinator is None else coordinator.router,
            graceful_close=None if coordinator is None else coordinator.graceful_close,
        )
    finally:
        if coordinator is not None:
            coordinator.close()
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
