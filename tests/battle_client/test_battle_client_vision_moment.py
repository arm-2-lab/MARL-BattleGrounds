"""Check the Vision Moment: the BattleClient vision page's frozen frame as a real game.

build_vision_moment returns the standard map 47 (Teutoburg Forest) config with
its two spawn banks exchanged, so Team A spawns and has its Red Zone on the
right, and a state that Core accepts as an authored start. The state pins the
vision's frame: tick 91 of 300, Team A 2 and Team B 4 of 20, every champion
alive at the vision's position and life, the vision's Ultimate cooldowns, the
Team A Rogue's 2-tick spawn shield, the Team B Mage's 5 ticks of Burst,
Blessing of Freedom on both Mages, respawn-wave clocks (2, 2) (one below the
recording's (3, 3), so the vision's "Respawn Wave in 3 Ticks" is true: Core
brings the fallen back in clock + 1 ticks), the recorded out-of-combat
countdowns and no action history. Its action masks allow what
the vision page allows: the Team A Warrior only Charge on the Team B Mage
(or no target), the Team B Mage only its Basic on Team A's Mage, Hunter and
Priest (or no target). vision_moment_scenario is an
interactive scene outside the DevClient registry that focuses the Team A
Warrior (slot 1). It declares packaged map 47 with exchanged spawn banks, so
the live client's match record names the map Teutoburg Forest (Map 47) and
records the exchange, for every start. The start is playable: through the real
BattleClient service, with the Warrior Manual and every other slot on a built-in
policy, Lock In advances the game to tick 92.

The near-end starts (VISION_MOMENT_STARTS) are valid Core starts that differ
from the Vision Moment only as declared: "fall" leaves the Team B Mage 1 life
and the Team B Priest's Ultimate 30 ticks from ready, and the staged Charge
makes the Mage fall whatever that Priest does;
"own_return" has the Team A Warrior fallen (dead, 0 life) with Team A's
respawn clock at 0; "last_tick" is tick 299; "trap_break" holds the Team B
Mage in a Hunter's Freezing Trap with 3 ticks left. One Core step with every
champion standing still brings the Warrior back alive ("own_return") and
ends the game at the 300-tick limit ("last_tick"); in "trap_break" the staged
Charge's blow breaks the trap (its counter goes to 0, the Mage lives), while a
still step only counts it down. An unknown start is refused, and
vision_moment_scenario names each start's scene.

The four event starts make one Visual Filters event happen in the next Core
step, whatever the other champions choose: "home_fall" (the Team B Mage, 1
life, inside Team B's Red Zone with the Warrior's Charge as its one legal
target) falls at home for 2 points whatever the Team B Priest does;
"step_aside" (the Team A Hunter fallen 0.45 units from the Warrior, whose
Ultimate cools down) keeps the Warrior standing on the body; "into_sight" (the
Team B Mage with a 1-tick spawn shield) is hidden from Team A now and seen
after one step; "rest" (the Team A Rogue at 60 of 100 life, out of combat)
regains 4 life in one step. Three more stun every other living, unshielded
champion for one tick, so only the named champions can act: in "barrier" the
staged Charge leaves the Team B Mage (18 life) alive with 1 life only because
the Team B Warrior's Guardian's Barrier covers it (with that Warrior out of
reach the same blow fells it), and Team A sees both; in "saved" tdm-beta's own
Priest answers the lethal Charge with Holy Word: Salvation on the Mage, which
lives (it falls without that heal); in "streak" your Warrior, alone on Team A
with the other four fallen, fells the five trapped 1-life Team B champions
with one Basic a turn, the fifth whatever the then free Team B Mage does, and
never falls, so the fifth turn is its 5th takedown without falling.

The vision's "Respawn Wave in 3 Ticks" is true of the Vision Moment: a Team A
champion fallen at tick 91 is still fallen after two still Core steps and back
after the third (tick 94).
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.scenarios import (
    SCENARIOS,
    VISION_MOMENT_STARTS,
    build_vision_moment,
    vision_moment_scenario,
)
from tests.helpers.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client.control import create_session
from marl_battlegrounds._battle_client.evaluation_bridge import (
    build_debugger_evaluation_context_v1,
)
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.protocol import (
    CommandRequestV1,
    FinishSystemCommandV1,
    KeyboardCommandV1,
)
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds._battle_client.system_worker import SystemWorker
from marl_battlegrounds.core.config import (
    validate_env_config,
    validate_scenario_initial_state,
)
from marl_battlegrounds.core.env import (
    _build_global_visibility_mask_and_distances,
    _build_observation_and_action_mask,
    step,
)
from marl_battlegrounds.core.types import (
    STUN_CHANNEL_HUNTER_TRAP,
    TEAM_B_ID,
    Action,
    DoneFlags,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.evaluation._recording.map_identity import recorded_map
from marl_battlegrounds.modes.team_deathmatch.bots.beta import reactive_tdm_beta_policy
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.shared_obs import (
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)
from marl_battlegrounds.tasks import (
    _swap_spawn_banks,
    canonical_tournament_rosters,
    list_tdm_maps,
    make_standard_team_deathmatch_config,
)

_POSITIONS = (
    (11.608, 8.807),
    (15.853, 4.733),
    (12.835, 7.227),
    (19.5, 9.0),
    (10.552, 8.334),
    (11.042, 6.471),
    (8.431, 8.776),
    (9.298, 6.733),
    (9.0, 7.702),
    (8.12, 6.798),
)
_HEALTH = (80.0, 200.0, 100.0, 100.0, 100.0, 77.0, 200.0, 89.8, 100.0, 70.0)


def test_vision_moment_is_a_valid_start_on_map_47_with_exchanged_spawns() -> None:
    config, state = build_vision_moment()
    validate_env_config(config)
    validate_scenario_initial_state(config, state)
    rosters = canonical_tournament_rosters()
    standard = make_standard_team_deathmatch_config(
        map_id=47, team_a_roster=rosters[0], team_b_roster=rosters[1]
    )
    exchanged = _swap_spawn_banks(standard)
    np.testing.assert_array_equal(
        np.asarray(config.team_spawn_pad_positions),
        np.asarray(exchanged.team_spawn_pad_positions),
    )
    assert float(np.asarray(config.team_spawn_pad_positions)[0, 0, 0]) == 19.5
    np.testing.assert_array_equal(
        np.asarray(config.obstacles), np.asarray(standard.obstacles)
    )
    assert int(config.max_steps) == 300
    assert int(config.team_deathmatch_score_threshold) == 20
    assert float(config.team_deathmatch_red_zone_depth) == 5.0


def test_vision_moment_pins_the_vision_frame() -> None:
    _, state = build_vision_moment()
    assert int(state.step_count) == 91
    assert np.asarray(state.team_deathmatch_scores).tolist() == [2, 4]
    assert np.asarray(state.alive_mask).all()
    np.testing.assert_allclose(
        np.asarray(state.agent_positions), np.asarray(_POSITIONS), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(state.current_health), np.asarray(_HEALTH), atol=1e-5
    )
    assert np.asarray(state.ultimate_cooldowns).tolist() == [
        16,
        0,
        1,
        20,
        19,
        30,
        0,
        0,
        0,
        0,
    ]
    assert np.asarray(state.spawn_shield_durations).tolist() == [
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert np.asarray(state.mage_burst_damage_amplification_durations).tolist() == [
        0,
        0,
        0,
        0,
        0,
        5,
        0,
        0,
        0,
        0,
    ]
    assert np.asarray(
        state.priest_blessing_of_freedom_slow_floor_durations
    ).tolist() == [
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    ]
    assert np.asarray(state.team_respawn_wave_countdowns).tolist() == [2, 2]
    assert np.asarray(state.steps_until_out_of_combat).tolist() == [
        0,
        0,
        0,
        0,
        0,
        5,
        0,
        1,
        0,
        5,
    ]
    assert not np.asarray(state.stun_durations).any()
    assert not np.asarray(state.slow_durations).any()
    assert not np.asarray(state.rogue_poison_anti_heal_durations).any()
    assert not bool(state.has_previous_timestep_joint_action)


def test_vision_moment_allows_exactly_the_vision_actions() -> None:
    config, state = build_vision_moment()
    _, mask = _build_observation_and_action_mask(state, config)
    joint = np.asarray(mask.select_target_use_ultimate_joint_mask)
    # Rows are [target category, use Ultimate]; target 6 is the opposing row 0.
    assert np.argwhere(joint[1]).tolist() == [[0, 0], [6, 1]]
    assert np.argwhere(joint[5]).tolist() == [[0, 0], [6, 0], [8, 0], [10, 0]]


def test_vision_moment_scenario_is_interactive_and_outside_the_registry() -> None:
    scenario = vision_moment_scenario()
    assert scenario.name == "vision_moment"
    assert scenario.title == "Teutoburg Forest"
    assert scenario.mode == "interactive"
    assert scenario.frames == ()
    assert scenario.default_controlled_slot == 1
    assert scenario.packaged_map == list_tdm_maps()[47]
    assert scenario.spawn_banks_exchanged is True
    assert scenario.scenario_specification is None
    assert scenario.provenance is None
    assert scenario.build_scenario is build_vision_moment
    assert "vision_moment" not in SCENARIOS


@pytest.mark.parametrize("start", ["vision", "home_fall"])
def test_vision_moment_match_record_names_map_47_with_exchanged_spawns(
    start: str,
) -> None:
    scenario = vision_moment_scenario(start)
    config, _ = scenario.build_scenario()
    context = build_debugger_evaluation_context_v1(
        debugger_test_launch_specification(),
        scenario=scenario,
        config=config,
        run_generation=0,
        action_source_kind="manual",
        team_a_controller="manual",
        team_b_controller="manual",
        execution_information_mode="shared_obs",
    )
    keys = {row.name: row.value for row in context.aggregation_keys}
    assert keys["map_id"] == "47"
    assert keys["map_origin"] == "registered"
    assert keys["marl_bgs.spawn_locations"] == "1"
    shown = recorded_map(context)
    assert shown.display_name == "Teutoburg Forest"
    assert shown.map_id == 47


def test_lock_in_from_the_vision_moment_advances_to_tick_92() -> None:
    policy = "tdm-alpha"
    setup = prepare_battle_setup(
        map_id=47,
        team_a=[policy, "manual", policy, policy, policy],
        team_b="tdm-beta",
    )
    setup = replace(setup, scenario=vision_moment_scenario())
    resolved = resolve_controllers(setup)
    session = create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=debugger_test_launch_specification(setup.seed),
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
    worker = SystemWorker(caller_thread=True)
    worker.install(session, directory=None)
    service = DebuggerService(
        session,
        view_mode="pov",
        preset="analysis",
        include_stress=False,
        system_worker=worker,
    )
    try:
        assert session.manual_slots == (1,)
        assert service.session.controlled_global_slot == 1
        assert service.current_frame().simulator_step_count == 91
        service.apply_command(
            CommandRequestV1(
                client_id="vision-moment-test",
                command_id="lock-in",
                base_revision=service.revision,
                command=KeyboardCommandV1(key="Enter"),
            )
        )
        finished = 0
        while worker.operation is not None:
            while service.drain_system_work():
                pass
            operation = worker.operation
            assert operation is not None
            service.apply_command(
                CommandRequestV1(
                    client_id="vision-moment-test",
                    command_id=f"finish-{finished}",
                    base_revision=service.revision,
                    command=FinishSystemCommandV1(operation_id=operation.identifier),
                )
            )
            finished += 1
            assert not service.faulted
        assert service.current_frame().simulator_step_count == 92
    finally:
        service.close()
        worker.close()
        while worker.drain():
            pass


def _stand_still_step(start: str) -> tuple[EnvState, EnvState, DoneFlags]:
    config, state = build_vision_moment(start)
    _, mask = _build_observation_and_action_mask(state, config)
    zeros = jnp.zeros((10,), dtype=jnp.int32)
    action = Action(move=zeros, select_target=zeros, use_ultimate=zeros)
    successor, _, _, done, _, _ = step(config, state, mask, action, jax.random.key(1))
    return state, successor, done


def test_near_end_starts_are_valid_and_change_only_what_they_declare() -> None:
    assert VISION_MOMENT_STARTS == (
        "vision",
        "fall",
        "own_return",
        "last_tick",
        "trap_break",
        "home_fall",
        "step_aside",
        "into_sight",
        "rest",
        "barrier",
        "saved",
        "streak",
    )
    _, vision = build_vision_moment()
    for start in VISION_MOMENT_STARTS[1:]:
        config, state = build_vision_moment(start)
        validate_scenario_initial_state(config, state)
        changed = {
            name
            for name in state._fields
            if not np.array_equal(
                np.asarray(getattr(state, name)), np.asarray(getattr(vision, name))
            )
        }
        expected = {
            "fall": {"current_health", "ultimate_cooldowns"},
            "own_return": {
                "current_health",
                "alive_mask",
                "team_respawn_wave_countdowns",
            },
            "last_tick": {"step_count"},
            "trap_break": {"stun_durations"},
            "home_fall": {"agent_positions", "current_health", "ultimate_cooldowns"},
            "step_aside": {
                "agent_positions",
                "alive_mask",
                "current_health",
                "ultimate_cooldowns",
            },
            "into_sight": {"spawn_shield_durations"},
            "rest": {"current_health"},
            "barrier": {"agent_positions", "current_health", "stun_durations"},
            "saved": {"current_health", "stun_durations"},
            "streak": {
                "agent_positions",
                "alive_mask",
                "current_health",
                "ultimate_cooldowns",
                "priest_blessing_of_freedom_slow_floor_durations",
                "spawn_shield_durations",
                "steps_until_out_of_combat",
                "team_respawn_wave_countdowns",
                "stun_durations",
            },
        }[start]
        assert changed == expected, start
    _, fall = build_vision_moment("fall")
    assert float(fall.current_health[5]) == 1.0
    assert int(fall.ultimate_cooldowns[9]) == 30
    _, own = build_vision_moment("own_return")
    assert not bool(own.alive_mask[1])
    assert float(own.current_health[1]) == 0.0
    assert np.asarray(own.team_respawn_wave_countdowns).tolist() == [0, 2]
    assert int(build_vision_moment("last_tick")[1].step_count) == 299
    trap = np.asarray(build_vision_moment("trap_break")[1].stun_durations)
    assert int(trap[5, STUN_CHANNEL_HUNTER_TRAP]) == 3
    assert int(np.count_nonzero(trap)) == 1


def test_one_still_step_reaches_each_near_end_state() -> None:
    _, returned, _ = _stand_still_step("own_return")
    assert bool(returned.alive_mask[1])
    assert int(returned.step_count) == 92
    _, ended, done = _stand_still_step("last_tick")
    assert int(ended.step_count) == 300
    assert bool(np.asarray(done.truncated).any() or np.asarray(done.terminated).any())


def test_the_vision_wave_text_is_true_a_fallen_champion_returns_in_three_ticks() -> (
    None
):
    config, state = build_vision_moment()
    state = state._replace(
        alive_mask=state.alive_mask.at[1].set(False),
        current_health=state.current_health.at[1].set(0.0),
    )
    zeros = jnp.zeros((10,), dtype=jnp.int32)
    action = Action(move=zeros, select_target=zeros, use_ultimate=zeros)
    alive_after: list[bool] = []
    for _ in range(3):
        _, mask = _build_observation_and_action_mask(state, config)
        state, *_ = step(config, state, mask, action, jax.random.key(1))
        alive_after.append(bool(state.alive_mask[1]))
    assert alive_after == [False, False, True]
    assert int(state.step_count) == 94


def test_unknown_start_is_refused_and_scenes_are_named_per_start() -> None:
    with pytest.raises(ValueError, match="unknown Vision Moment start"):
        build_vision_moment("sunrise")
    with pytest.raises(ValueError, match="unknown Vision Moment start"):
        vision_moment_scenario("sunrise")
    fall = vision_moment_scenario("fall")
    assert fall.name == "vision_moment_fall"
    assert fall.default_controlled_slot == 1
    assert int(fall.build_scenario()[1].current_health[5]) == 1


def test_the_staged_charge_makes_the_mage_fall_whatever_the_priest_does() -> None:
    config, state = build_vision_moment("fall")
    _, mask = _build_observation_and_action_mask(state, config)
    allowed = np.argwhere(np.asarray(mask.select_target_use_ultimate_joint_mask)[9])
    assert allowed.tolist() == [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
    for target, ultimate in allowed.tolist():
        select = np.zeros(10, dtype=np.int32)
        use = np.zeros(10, dtype=np.int32)
        select[1], use[1] = 6, 1
        select[9], use[9] = target, ultimate
        action = Action(
            move=jnp.zeros((10,), dtype=jnp.int32),
            select_target=jnp.asarray(select),
            use_ultimate=jnp.asarray(use),
        )
        successor, *_ = step(config, state, mask, action, jax.random.key(1))
        assert not bool(successor.alive_mask[5])
        assert np.asarray(successor.team_deathmatch_scores).tolist() == [3, 4]


def test_the_staged_charge_breaks_the_freezing_trap_in_the_trap_break_start() -> None:
    config, state = build_vision_moment("trap_break")
    _, mask = _build_observation_and_action_mask(state, config)
    select = np.zeros(10, dtype=np.int32)
    use = np.zeros(10, dtype=np.int32)
    select[1], use[1] = 6, 1
    action = Action(
        move=jnp.zeros((10,), dtype=jnp.int32),
        select_target=jnp.asarray(select),
        use_ultimate=jnp.asarray(use),
    )
    charged, *_ = step(config, state, mask, action, jax.random.key(1))
    assert bool(charged.alive_mask[5])
    assert float(charged.current_health[5]) < float(state.current_health[5])
    assert int(charged.stun_durations[5, STUN_CHANNEL_HUNTER_TRAP]) == 0
    _, still, _ = _stand_still_step("trap_break")
    assert int(still.stun_durations[5, STUN_CHANNEL_HUNTER_TRAP]) == 2


def test_the_home_fall_start_fells_the_mage_at_home_for_two_points() -> None:
    config, state = build_vision_moment("home_fall")
    assert np.asarray(state.agent_positions[5]).tolist() == [2.0, 4.0]
    assert np.asarray(state.agent_positions[1]).tolist() == [5.0, 3.0]
    _, mask = _build_observation_and_action_mask(state, config)
    joint = np.asarray(mask.select_target_use_ultimate_joint_mask)
    assert np.argwhere(joint[1]).tolist() == [[0, 0], [6, 1]]
    for target, ultimate in np.argwhere(joint[9]).tolist():
        select = np.zeros(10, dtype=np.int32)
        use = np.zeros(10, dtype=np.int32)
        select[1], use[1] = 6, 1
        select[9], use[9] = target, ultimate
        action = Action(
            move=jnp.zeros((10,), dtype=jnp.int32),
            select_target=jnp.asarray(select),
            use_ultimate=jnp.asarray(use),
        )
        successor, *_ = step(config, state, mask, action, jax.random.key(1))
        assert not bool(successor.alive_mask[5])
        assert np.asarray(successor.team_deathmatch_scores).tolist() == [4, 4]


def test_the_step_aside_start_keeps_the_warrior_on_the_fallen_hunter() -> None:
    config, state = build_vision_moment("step_aside")
    _, mask = _build_observation_and_action_mask(state, config)
    assert np.argwhere(
        np.asarray(mask.select_target_use_ultimate_joint_mask)[1]
    ).tolist() == [[0, 0]]
    _, still, _ = _stand_still_step("step_aside")
    assert not bool(still.alive_mask[2])
    assert bool(still.alive_mask[1])
    gap = np.asarray(still.agent_positions[1]) - np.asarray(still.agent_positions[2])
    assert float(np.hypot(*gap)) < 0.6


def test_the_into_sight_start_brings_the_shielded_mage_into_sight() -> None:
    config, state = build_vision_moment("into_sight")
    seen, _ = _build_global_visibility_mask_and_distances(state, config)
    assert not bool(np.asarray(seen)[0:5, 5].any())
    _, still, _ = _stand_still_step("into_sight")
    seen, _ = _build_global_visibility_mask_and_distances(still, config)
    assert bool(np.asarray(seen)[0:5, 5].any())


def test_the_rest_start_lets_the_rogue_regain_life_out_of_combat() -> None:
    _, state = build_vision_moment("rest")
    assert float(state.current_health[3]) == 60.0
    _, still, _ = _stand_still_step("rest")
    assert float(still.current_health[3]) == 64.0


def _stunned_slots(state: EnvState) -> list[int]:
    return np.flatnonzero(np.asarray(state.stun_durations).any(axis=1)).tolist()


def _charge_step(
    config: EnvConfig, state: EnvState, team_b: ActorAction | None = None
) -> EnvState:
    _, mask = _build_observation_and_action_mask(state, config)
    move = np.zeros(10, dtype=np.int32)
    select = np.zeros(10, dtype=np.int32)
    use = np.zeros(10, dtype=np.int32)
    select[1], use[1] = 6, 1
    if team_b is not None:
        move[5:] = np.asarray(team_b.move)
        select[5:] = np.asarray(team_b.select_target)
        use[5:] = np.asarray(team_b.use_ultimate)
    action = Action(
        move=jnp.asarray(move),
        select_target=jnp.asarray(select),
        use_ultimate=jnp.asarray(use),
    )
    successor, *_ = step(config, state, mask, action, jax.random.key(1))
    return successor


def _only_stay(state: EnvState, config: EnvConfig, slot: int) -> bool:
    _, mask = _build_observation_and_action_mask(state, config)
    return np.argwhere(
        np.asarray(mask.select_target_use_ultimate_joint_mask)[slot]
    ).tolist() == [[0, 0]] and np.flatnonzero(
        np.asarray(mask.move_mask)[slot]
    ).tolist() == [0]


def test_the_barrier_start_keeps_the_mage_alive_only_through_the_barrier() -> None:
    config, state = build_vision_moment("barrier")
    assert np.asarray(state.agent_positions[6]).tolist() == pytest.approx(
        list(_POSITIONS[7])
    )
    assert float(state.current_health[5]) == 18.0
    # Every living champion but your Warrior is stunned (the Team A Rogue's
    # spawn shield already keeps it from striking).
    assert _stunned_slots(state) == [0, 2, 4, 5, 6, 7, 8, 9]
    for slot in _stunned_slots(state):
        assert _only_stay(state, config, slot), slot
    gap = np.asarray(state.agent_positions[6]) - np.asarray(state.agent_positions[5])
    assert float(np.hypot(*gap)) < 2.0
    seen, _ = _build_global_visibility_mask_and_distances(state, config)
    assert bool(np.asarray(seen)[0:5, 5].any())
    assert bool(np.asarray(seen)[0:5, 6].any())
    held = _charge_step(config, state)
    assert bool(held.alive_mask[5])
    assert float(held.current_health[5]) == pytest.approx(1.0)
    # The same blow with the barrier's Warrior out of reach fells the Mage.
    away = state._replace(
        agent_positions=state.agent_positions.at[6].set(jnp.asarray(_POSITIONS[6]))
    )
    assert not bool(_charge_step(config, away).alive_mask[5])


def test_the_saved_start_lets_the_tdm_beta_priest_save_the_mage() -> None:
    config, state = build_vision_moment("saved")
    assert float(state.current_health[5]) == 1.0
    assert int(state.ultimate_cooldowns[9]) == 0
    assert _stunned_slots(state) == [0, 2, 4, 5, 6, 7, 8]
    observation, mask = _build_observation_and_action_mask(state, config)
    profile = config.agent_profile
    team_policy = cast(Callable[..., ActorAction], execute_shared_obs_team_policy)
    team_b = team_policy(
        observation,
        mask,
        jax.random.split(jax.random.key(0), 10),
        build_shared_obs_sensor_source_bank(observation),
        build_default_shared_obs_information_availability(
            profile.active_mask, profile.team_ids
        ),
        reactive_tdm_beta_policy,
        TEAM_B_ID,
    )
    # tdm-beta's Priest (Team B slot 4) answers with Holy Word: Salvation on
    # the Mage (its ally row 1); the stunned others can only stay.
    assert int(team_b.use_ultimate[4]) == 1
    assert int(team_b.select_target[4]) == 1
    assert np.asarray(team_b.select_target[:4]).tolist() == [0, 0, 0, 0]
    saved = _charge_step(config, state, team_b)
    assert bool(saved.alive_mask[5])
    assert float(saved.current_health[5]) > 1.0
    assert not bool(_charge_step(config, state).alive_mask[5])


def test_the_streak_start_lets_your_warrior_fell_five_in_five_turns() -> None:
    config, state = build_vision_moment("streak")
    assert (
        np.asarray(state.alive_mask).tolist()
        == [False, True, False, False, False] + [True] * 5
    )
    assert np.asarray(state.team_respawn_wave_countdowns).tolist() == [4, 4]
    assert (
        np.asarray(state.stun_durations)[5:, STUN_CHANNEL_HUNTER_TRAP].tolist()
        == [4] * 5
    )
    seen, _ = _build_global_visibility_mask_and_distances(state, config)
    assert bool(np.asarray(seen)[1, 5:].all())
    jitted = cast(Callable[..., tuple[EnvState, ...]], jax.jit(step))

    def basic(
        current: EnvState, target: int, mage: tuple[int, int, int] = (0, 0, 0)
    ) -> EnvState:
        _, mask = _build_observation_and_action_mask(current, config)
        assert bool(
            np.asarray(mask.select_target_use_ultimate_joint_mask)[1, target + 1, 0]
        )
        move = np.zeros(10, dtype=np.int32)
        select = np.zeros(10, dtype=np.int32)
        use = np.zeros(10, dtype=np.int32)
        select[1] = target + 1
        move[5], select[5], use[5] = mage
        action = Action(
            move=jnp.asarray(move),
            select_target=jnp.asarray(select),
            use_ultimate=jnp.asarray(use),
        )
        successor: EnvState = jitted(config, current, mask, action, jax.random.key(1))[
            0
        ]
        return successor

    for turn, target in enumerate((9, 8, 6, 7)):
        for slot in (0, 2, 3, 4, *(s for s in (5, 6, 7, 8, 9) if s != target)):
            if bool(state.alive_mask[slot]):
                assert _only_stay(state, config, slot), (turn, slot)
        state = basic(state, target)
        assert not bool(state.alive_mask[target]), turn
        assert bool(state.alive_mask[1])
        assert np.asarray(state.team_deathmatch_scores).tolist() == [3 + turn, 4]
    # Turn 5: the Team B Mage is free again; every choice it has still falls.
    _, mask = _build_observation_and_action_mask(state, config)
    moves = np.flatnonzero(np.asarray(mask.move_mask)[5]).tolist()
    combat = np.argwhere(
        np.asarray(mask.select_target_use_ultimate_joint_mask)[5]
    ).tolist()
    assert len(moves) == 9
    for move_choice in moves:
        for target, ultimate in combat:
            last = basic(state, 5, (move_choice, target, ultimate))
            assert not bool(last.alive_mask[5]), (move_choice, target, ultimate)
            assert bool(last.alive_mask[1])
            assert np.asarray(last.team_deathmatch_scores).tolist() == [7, 4]
