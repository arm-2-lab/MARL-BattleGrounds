"""Check Capture the Flag games in the BattleClient through the private launch route.

prepare_battle_setup(task="ctf") builds a Capture the Flag game on a CTF map
(the scenario is named ctf_map_N) and refuses scenario_id, since the mode has
no scenarios yet. Contracts: a game of the built-in tdm-alpha bot against Random
plays to its time limit through the live service, from the default spawn ends
and with the ends swapped; it records an episode context V5 whose task is
capture_the_flag, names the packaged map with its spawn ends
(marl_bgs.spawn_locations "0" or "1"), shows no Red Zone in its live frames,
reports the time limit as truncation, and saves a replay V5 that loads with one
frame per tick. Flags move through the live service: from a hand-built start
on map 0 with a threshold of 1 capture, the first turn records one return (a
Manual champion touches its own dropped flag) and one capture (a Manual carrier
at its own home), scores 1 to 0 and ends the game as terminated; the saved
replay holds both events. A flag dropped by a dying carrier is not built here
(it needs a kill); Core's rule tests and the replay story cover it. A Manual
player whose champion is dead from the first tick still gets an Agent POV, and
it shows both flags at their homes.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import jax.numpy as jnp
import pytest
from tests.battle_client.test_battle_client_worker import _finish
from tests.helpers.visual_debugger_fixtures import debugger_test_launch_specification
from tests.visual_debugger.test_visual_debugger_systems import _send

from marl_battlegrounds._battle_client.control import create_session
from marl_battlegrounds._battle_client.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
    debugger_action_source_kind_v1,
)
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.protocol import (
    FinishAndReviewCommandV1,
    KeyboardCommandV1,
)
from marl_battlegrounds._battle_client.recording import (
    build_debugger_recording_specification_v1,
    recording_policy_execution_included,
)
from marl_battlegrounds._battle_client.replay_recorder import DebuggerReplayRecorder
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import BattleSetup, prepare_battle_setup
from marl_battlegrounds._battle_client.system_worker import SystemWorker
from marl_battlegrounds.core.types import EnvConfig, EnvState
from marl_battlegrounds.evaluation._replays.replay_v5 import ReplayArtifactV5
from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV5
from marl_battlegrounds.evaluation.replay_io import (
    load_replay,
    preflight_replay_destination,
)
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.viewer.presentation_protocol import (
    PresentationApiErrorV1,
    SharedObsAuthorizedCurrentEndpointV1,
)

_TICKS = 6


def _starting_from(
    setup: BattleSetup, arrange: Callable[[EnvState], EnvState]
) -> BattleSetup:
    build = setup.scenario.build_scenario

    def arranged() -> tuple[EnvConfig, EnvState]:
        config, state = build()
        return config, arrange(state)

    return dataclasses.replace(
        setup, scenario=dataclasses.replace(setup.scenario, build_scenario=arranged)
    )


def _session(setup: BattleSetup, *, recorded: bool = False) -> DebuggerSession:
    resolved = resolve_controllers(setup)
    launch = debugger_test_launch_specification(setup.seed)
    if recorded:
        launch = build_debugger_evaluation_launch_specification_v1(
            root_seed=setup.seed,
            code_revision=launch.code_revision,
            capture_profile="evaluation_metric_complete",
        )
    return create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=launch,
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller=resolved.controllers[0],
        team_b_controller=resolved.controllers[1],
        execution_information_mode="shared_obs",
        systems=resolved.systems,
        borrow_systems=True,
        manual_slots=resolved.manual_slots,
        slot_labels=resolved.slot_labels,
    )


@contextmanager
def _recorded_game(
    setup: BattleSetup, tmp_path: Path
) -> Iterator[tuple[DebuggerService, SystemWorker, DebuggerReplayRecorder]]:
    session = _session(setup, recorded=True)
    context = session.evaluation_context
    recorder = DebuggerReplayRecorder(
        specification=build_debugger_recording_specification_v1(
            action_source_kind=debugger_action_source_kind_v1(
                session.scenario, session.team_a_controller, session.team_b_controller
            ),
            runtime_provenance=capture_runtime_provenance(
                context.code_revision.package_version,
                policy_execution_included=recording_policy_execution_included(context),
            ),
        ),
        destination=preflight_replay_destination(tmp_path / "ctf.marlbg-replay.json"),
        context=context,
        initial_frame=session.current_evaluation_frame,
        scenario_name=setup.scenario.name,
    )
    worker = SystemWorker(caller_thread=True)
    worker.install(session, directory=recorder.evidence_directory)
    service = DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        recorder=recorder,
        system_worker=worker,
    )
    try:
        yield service, worker, recorder
    finally:
        service.close()
        worker.close()
        while worker.drain():
            pass


def _play(service: DebuggerService, worker: SystemWorker, turns: int) -> None:
    for _ in range(turns):
        _send(service, KeyboardCommandV1(key="Enter"))
        _finish(service, worker)


def _save(
    service: DebuggerService, recorder: DebuggerReplayRecorder
) -> ReplayArtifactV5:
    _send(service, FinishAndReviewCommandV1())
    saved = recorder.saved_bundle
    assert saved is not None
    replay = load_replay(saved.replay_path).replay
    assert isinstance(replay, ReplayArtifactV5)
    return replay


def test_ctf_setup_names_its_map_and_refuses_scenarios() -> None:
    setup = prepare_battle_setup(task="ctf", map_id=3)
    assert setup.scenario.name == "ctf_map_3"
    assert prepare_battle_setup(map_id=3).scenario.name == "tdm_map_3"
    with pytest.raises(ValueError, match="no scenarios yet"):
        prepare_battle_setup(task="ctf", scenario_id=1)


@pytest.mark.parametrize("spawn_mode", ["default", "swapped"])
def test_a_bot_game_plays_to_the_time_limit_and_saves_replay_v5(
    tmp_path: Path, spawn_mode: str
) -> None:
    setup = prepare_battle_setup(
        task="ctf",
        map_id=3,
        team_a="tdm-alpha",
        team_b="random",
        max_steps=_TICKS,
        spawn_mode=spawn_mode,
    )
    assert setup.scenario.spawn_banks_exchanged is (spawn_mode == "swapped")
    with _recorded_game(setup, tmp_path) as (service, worker, recorder):
        context = service.session.evaluation_context
        assert isinstance(context, EvaluationEpisodeContextV5)
        assert context.identity.task.identifier == "capture_the_flag"
        keys = {row.name: row.value for row in context.aggregation_keys}
        assert (keys["map_origin"], keys["map_id"]) == ("registered", "3")
        assert keys["marl_bgs.spawn_locations"] == (
            "1" if spawn_mode == "swapped" else "0"
        )
        assert service.current_frame().combat_configuration.red_zone_depth is None
        _play(service, worker, _TICKS)
        frame = service.current_frame()
        assert frame.combat_configuration.red_zone_depth is None
        assert frame.terminal.is_sealed and frame.terminal.reason == "truncated"
        replay = _save(service, recorder)
    assert (len(replay.frames), len(replay.transitions)) == (_TICKS + 1, _TICKS)


def test_a_live_return_and_winning_capture_end_the_game(tmp_path: Path) -> None:
    def flags_in_play(state: EnvState) -> EnvState:
        # Our Mage (slot 0) carries their flag 1.0 from our home, inside the
        # 1.25 capture reach; our flag lies dropped under our slot 1.
        point = jnp.asarray((5.0, 5.0), jnp.float32)
        return state._replace(
            agent_positions=state.agent_positions.at[0]
            .set(jnp.asarray((1.5, 5.0), jnp.float32))
            .at[1]
            .set(point),
            capture_the_flag_flag_positions=state.capture_the_flag_flag_positions.at[0]
            .set(point)
            .at[1]
            .set(jnp.asarray((1.5, 5.0), jnp.float32)),
            capture_the_flag_flag_carrier_slots=(
                state.capture_the_flag_flag_carrier_slots.at[1].set(0)
            ),
            capture_the_flag_dropped_flag_countdowns=(
                state.capture_the_flag_dropped_flag_countdowns.at[0].set(10)
            ),
        )

    setup = _starting_from(
        prepare_battle_setup(task="ctf", map_id=0, team_b="random", score_threshold=1),
        flags_in_play,
    )
    with _recorded_game(setup, tmp_path) as (service, worker, recorder):
        _play(service, worker, 1)
        frame = service.current_frame()
        assert frame.terminal.is_sealed and frame.terminal.reason == "terminated"
        replay = _save(service, recorder)
    (transition,) = replay.transitions
    flag_events = Counter(
        type(event).__name__
        for event in transition.events
        if type(event).__name__.startswith("Flag")
    )
    assert flag_events == {"FlagReturnedEventV1": 1, "FlagCapturedEventV1": 1}
    last = replay.frames[-1].snapshot
    assert last.capture_the_flag_scores == (1, 0)
    assert last.capture_the_flag_flag_carrier_slots == (None, None)


def test_a_dead_manual_players_agent_pov_shows_both_flags() -> None:
    setup = prepare_battle_setup(task="ctf", map_id=0, team_b="random")
    session = _session(
        _starting_from(
            setup,
            lambda state: state._replace(
                alive_mask=state.alive_mask.at[0].set(False),
                current_health=state.current_health.at[0].set(0),
            ),
        )
    )
    assert not bool(session.state.alive_mask[0]) and 0 in session.manual_slots
    service = DebuggerService(
        session, view_mode="pov", preset="analysis", include_stress=False
    )
    try:
        result = service.current_presentation()
        assert result.outcome == "response"
        payload = result.payload
        assert not isinstance(payload, PresentationApiErrorV1)
        endpoint = payload.current_endpoint
        assert isinstance(endpoint, SharedObsAuthorizedCurrentEndpointV1)
        flags = endpoint.parts.scene.capture_the_flag
        assert flags is not None
        homes = setup.scenario.build_scenario()[0].capture_the_flag_flag_home_positions
        for flag, home in zip(flags.flags, homes.tolist(), strict=True):
            assert flag.state == "home"
            assert flag.position == flag.home_position == pytest.approx(tuple(home))
    finally:
        service.close()
