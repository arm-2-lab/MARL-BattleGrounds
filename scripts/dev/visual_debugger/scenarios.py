"""Build deterministic live scenarios for the local visual debugger.

The registry joins import-light catalog labels to configuration/state factories
and scripted command sequences. ``get_scenario`` selects a definition;
``list_scenarios`` and ``cycle_scenario_name`` support launcher navigation.
Factories reset through Core with a fixed key, then install authored positions
or lifecycle state for demonstration. These are debugger scenes, not the
packaged TDM benchmark tasks, with one exception outside the registry: the
Vision Moment (``build_vision_moment`` and ``vision_moment_scenario``), a real
map 47 state equal to the BattleClient vision page's frozen frame (its scene
declares packaged map 47 with exchanged spawn banks, so the client names the
map), with eleven other starts built from it (``VISION_MOMENT_STARTS``: four
near-end starts and seven that make one Visual Filters event happen), which
the private parity tools launch directly. Calling a factory creates JAX arrays but
writes no files or replay records.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds._battle_client.model import (
    ActorCommand,
    DebuggerScenario,
    ScenarioFrame,
)
from marl_battlegrounds.core.combat import OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS
from marl_battlegrounds.core.config import (
    CANONICAL_PRODUCT_MOVEMENT_SCALE,
    resolve_agent_profile,
)
from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_STAY,
    MOVE_WEST,
    NEUTRAL_CLASS_ID,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_WARRIOR_CHARGE,
    WARRIOR_CLASS_ID,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.tasks import (
    _swap_spawn_banks,  # pyright: ignore[reportPrivateUsage]
    canonical_tournament_rosters,
    list_tdm_maps,
    make_standard_team_deathmatch_config,
)
from scripts.dev.visual_debugger.scenario_catalog import SCENARIO_CATALOG_BY_NAME
from scripts.dev.visual_debugger.scenario_catalog import (
    iter_scenario_summaries as iter_scenario_summaries,
)

_MAX_STEPS = 300


def _registered_scenario(
    name: str,
    *,
    build_scenario: Callable[[], tuple[EnvConfig, EnvState]],
    frames: tuple[ScenarioFrame, ...],
) -> DebuggerScenario:
    """Bind one live constructor to the catalog's canonical launch metadata."""
    metadata = SCENARIO_CATALOG_BY_NAME[name]
    return DebuggerScenario(
        name=metadata.name,
        title=metadata.title,
        description=metadata.description,
        mode=metadata.mode,
        build_scenario=build_scenario,
        frames=frames,
        default_controlled_slot=metadata.default_controlled_slot,
        audience=metadata.audience,
    )


def _empty_obstacles() -> Array:
    """Return an inactive float32 obstacle table of shape (MAX_OBSTACLE_SLOTS,
    OBSTACLE_FEATURES).
    """
    return jnp.zeros(
        (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
        dtype=jnp.float32,
    )


def _pillar(x: float, y: float, radius: float) -> Array:
    """Build one active float32 pillar row from center and radius in map units.

    The result has shape (OBSTACLE_FEATURES,). Configuration validation owns
    geometric validity; this helper only fills the declared obstacle fields.
    """
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(y)
    obstacle = obstacle.at[OBSTACLE_FEATURE_RADIUS].set(radius)
    return obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _wall(
    x: float,
    y: float,
    width: float,
    height: float,
    theta: float,
) -> Array:
    """Build one active float32 wall row from center, dimensions, and angle.

    Position, width, and height use map units; theta uses radians. The result has
    shape (OBSTACLE_FEATURES,). Configuration validation owns geometric validity.
    """
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_WALL)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(y)
    obstacle = obstacle.at[OBSTACLE_FEATURE_WIDTH].set(width)
    obstacle = obstacle.at[OBSTACLE_FEATURE_HEIGHT].set(height)
    obstacle = obstacle.at[OBSTACLE_FEATURE_THETA].set(theta)
    return obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _spawn_pad_positions(map_width: float, map_height: float) -> Array:
    """Return two unobstructed edge formations for later wave respawns."""
    y_coordinates = jnp.linspace(
        1.5,
        map_height - 1.5,
        MAX_AGENTS_PER_TEAM,
        dtype=jnp.float32,
    )
    team_a = jnp.stack(
        (jnp.full_like(y_coordinates, 1.5), y_coordinates),
        axis=-1,
    )
    team_b = jnp.stack(
        (jnp.full_like(y_coordinates, map_width - 1.5), y_coordinates),
        axis=-1,
    )
    return jnp.stack((team_a, team_b), axis=0)


def _scenario(
    *,
    map_width: float,
    map_height: float,
    team_sizes: tuple[int, int],
    class_ids: tuple[int, ...],
    active_positions: dict[int, tuple[float, float]],
    obstacles: Array | None = None,
) -> tuple[EnvConfig, EnvState]:
    """Build Core reset state, then install the declared debugger positions.

    Parameters
    ----------
    map_width, map_height : float
        Arena dimensions in map units.
    team_sizes : tuple of int
        Active counts for Team A and Team B.
    class_ids : tuple of int
        Exactly MAX_AGENT_SLOTS class IDs in simulator slot order.
    active_positions : dict of int to tuple of float
        Slot-to-(x, y) positions in map units to install after reset.
    obstacles : Array or None, optional
        Fixed float32 obstacle table. None creates an inactive table.

    Returns
    -------
    tuple of EnvConfig and EnvState
        A 300-step neutral-task config (task mode 0, no score threshold and
        Red Zone depth 0.0, so no Red Zone tint or scoring) with the canonical
        movement scale, edge spawn pads, five-step respawn waves, and the
        authored starting state.

    Raises
    ------
    ValueError
        The class count or Core profile/reset configuration is invalid.

    Notes
    -----
    This creates JAX arrays and resets with key zero. The later authored position
    replacement is not itself a second Core reset or a gameplay transition.
    """
    if len(class_ids) != MAX_AGENT_SLOTS:
        msg = f"class_ids must contain {MAX_AGENT_SLOTS} fixed-slot values."
        raise ValueError(msg)

    profile = resolve_agent_profile(
        jnp.asarray(class_ids, dtype=jnp.int32),
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    positions = jnp.zeros(
        (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        dtype=jnp.float32,
    )
    for global_slot, position in active_positions.items():
        positions = positions.at[global_slot].set(
            jnp.asarray(position, dtype=jnp.float32)
        )

    config = EnvConfig(
        task_mode=0,
        team_deathmatch_score_threshold=0,
        team_deathmatch_red_zone_depth=0.0,
        max_steps=_MAX_STEPS,
        map_width=map_width,
        map_height=map_height,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=CANONICAL_PRODUCT_MOVEMENT_SCALE,
        team_spawn_pad_positions=_spawn_pad_positions(map_width, map_height),
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray((5, 5), dtype=jnp.int32),
    )
    initial_state, _, _, _ = reset(config, jax.random.key(0))
    return config, initial_state._replace(agent_positions=positions)


def _arena_5v5_scenario() -> tuple[EnvConfig, EnvState]:
    """Build two five-class teams across a pillar and an angled wall.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    obstacles = _empty_obstacles()
    obstacles = obstacles.at[0].set(_pillar(10.0, 3.0, 0.9))
    obstacles = obstacles.at[1].set(_wall(10.0, 7.8, 3.0, 0.5, 0.45))
    roster = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
    )
    positions = {
        0: (3.0, 1.0),
        1: (3.0, 3.0),
        2: (3.0, 5.0),
        3: (3.0, 7.0),
        4: (3.0, 9.0),
        5: (17.0, 1.0),
        6: (17.0, 3.0),
        7: (17.0, 5.0),
        8: (17.0, 7.0),
        9: (17.0, 9.0),
    }
    return _scenario(
        map_width=20.0,
        map_height=10.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions=positions,
        obstacles=obstacles,
    )


def _basic_support_scenario() -> tuple[EnvConfig, EnvState]:
    """Build a three-versus-three formation for Basic and support actions.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        MAGE_CLASS_ID,
        HUNTER_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(3, 3),
        class_ids=roster,
        active_positions={
            0: (6.0, 3.0),
            1: (6.0, 6.0),
            2: (6.0, 9.0),
            5: (9.0, 3.0),
            6: (9.0, 6.0),
            7: (9.0, 9.0),
        },
    )


def _ultimate_showcase_scenario() -> tuple[EnvConfig, EnvState]:
    """Build two five-class teams placed for Ultimate demonstrations.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
    )
    return _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (4.0, 2.0),
            1: (6.0, 5.0),
            2: (6.0, 8.0),
            3: (9.0, 5.0),
            4: (4.0, 10.0),
            5: (8.0, 6.0),
            6: (9.0, 8.0),
            7: (11.0, 3.0),
            8: (13.0, 8.0),
            9: (14.0, 10.0),
        },
    )


def _aura_crossfire_scenario() -> tuple[EnvConfig, EnvState]:
    """Build opposed three-agent groups for aura and crossfire display.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(3, 3),
        class_ids=roster,
        active_positions={
            0: (6.0, 5.0),
            1: (6.0, 7.0),
            2: (7.5, 6.0),
            5: (12.0, 5.0),
            6: (12.0, 7.0),
            7: (10.5, 6.0),
        },
    )


def _stacked_team_auras_scenario() -> tuple[EnvConfig, EnvState]:
    """Build duplicate Mage and Warrior sources on each five-agent team.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        MAGE_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        MAGE_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
    )
    return _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (6.0, 5.0),
            1: (6.0, 7.0),
            2: (7.5, 4.5),
            3: (7.5, 7.5),
            4: (7.5, 6.0),
            5: (12.0, 5.0),
            6: (12.0, 7.0),
            7: (10.5, 4.5),
            8: (10.5, 7.5),
            9: (10.5, 6.0),
        },
    )


def _status_stack_scenario() -> tuple[EnvConfig, EnvState]:
    """Build a three-versus-two formation for interacting status demonstrations.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        HUNTER_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(3, 2),
        class_ids=roster,
        active_positions={
            0: (5.0, 6.0),
            1: (7.5, 4.4),
            2: (10.0, 5.0),
            5: (10.0, 6.0),
            6: (10.0, 8.0),
        },
    )


def _team_focus_crossfire_scenario() -> tuple[EnvConfig, EnvState]:
    """Build four attackers opposite a Warrior with three Priests.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
        WARRIOR_CLASS_ID,
        PRIEST_CLASS_ID,
        PRIEST_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(4, 4),
        class_ids=roster,
        active_positions={
            0: (7.0, 6.0),
            1: (8.0, 5.0),
            2: (9.0, 3.0),
            3: (9.0, 4.6),
            5: (9.0, 6.0),
            6: (11.5, 6.0),
            7: (10.8, 8.0),
            8: (7.2, 8.0),
        },
    )


def _moving_focus_crossfire_scenario() -> tuple[EnvConfig, EnvState]:
    """Return a radially separated focus-fire fixture for minimum-view review."""
    roster = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
        WARRIOR_CLASS_ID,
        PRIEST_CLASS_ID,
        PRIEST_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=16.0,
        map_height=12.0,
        team_sizes=(4, 4),
        class_ids=roster,
        active_positions={
            0: (6.02, 4.02),
            1: (6.52, 6.0),
            2: (6.02, 7.98),
            3: (9.48, 6.0),
            5: (8.0, 6.0),
            6: (9.98, 4.02),
            7: (9.98, 7.98),
            8: (8.0, 8.8),
        },
    )


def _mirrored_ultimates_scenario() -> tuple[EnvConfig, EnvState]:
    """Build two five-class teams in separated reciprocal Ultimate positions.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
    )
    return _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (3.0, 1.0),
            1: (6.0, 4.0),
            2: (6.6, 8.0),
            3: (7.3, 11.0),
            4: (4.0, 10.5),
            5: (15.0, 1.0),
            6: (10.0, 4.0),
            7: (9.4, 8.0),
            8: (8.7, 11.0),
            9: (12.0, 10.5),
        },
    )


def _moving_basic_crossfire_scenario() -> tuple[EnvConfig, EnvState]:
    """Build two five-class teams positioned for movement and Basic crossfire.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
    )
    return _scenario(
        map_width=14.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (5.0, 4.0),
            1: (5.0, 8.5),
            2: (5.0, 6.0),
            3: (8.7, 8.5),
            4: (3.0, 4.5),
            5: (7.0, 6.0),
            6: (6.3, 8.5),
            7: (7.0, 4.0),
            8: (10.0, 8.5),
            9: (9.0, 5.5),
        },
    )


def _charge_convergence_scenario() -> tuple[EnvConfig, EnvState]:
    """Build two Warriors aimed toward one opposing Warrior.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        WARRIOR_CLASS_ID,
        WARRIOR_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        WARRIOR_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=14.0,
        map_height=12.0,
        team_sizes=(2, 1),
        class_ids=roster,
        active_positions={
            0: (3.0, 4.0),
            1: (3.0, 8.0),
            5: (8.0, 6.0),
        },
    )


def _trap_lifecycle_scenario() -> tuple[EnvConfig, EnvState]:
    """Build five Hunters opposite five Warriors for trap demonstrations.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        HUNTER_CLASS_ID,
        HUNTER_CLASS_ID,
        HUNTER_CLASS_ID,
        HUNTER_CLASS_ID,
        HUNTER_CLASS_ID,
        WARRIOR_CLASS_ID,
        WARRIOR_CLASS_ID,
        WARRIOR_CLASS_ID,
        WARRIOR_CLASS_ID,
        WARRIOR_CLASS_ID,
    )
    return _scenario(
        map_width=12.0,
        map_height=12.0,
        team_sizes=(5, 5),
        class_ids=roster,
        active_positions={
            0: (3.0, 2.0),
            1: (3.0, 4.0),
            2: (3.0, 6.0),
            3: (3.0, 8.0),
            4: (4.5, 3.0),
            5: (6.0, 2.0),
            6: (6.0, 4.0),
            7: (6.0, 6.0),
            8: (6.0, 8.0),
            9: (6.0, 10.0),
        },
    )


def _max_status_stack_scenario() -> tuple[EnvConfig, EnvState]:
    """Build a Mage and Priest opposite four status-producing opponents.

    Returns the fixed configuration and authored initial state; scripted frames
    in the registry choose the later actions.
    """
    roster = (
        MAGE_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    return _scenario(
        map_width=16.0,
        map_height=12.0,
        team_sizes=(2, 4),
        class_ids=roster,
        active_positions={
            0: (8.0, 6.0),
            1: (10.5, 7.0),
            5: (8.0, 1.0),
            6: (10.0, 4.0),
            7: (11.0, 6.0),
            8: (8.0, 7.4),
        },
    )


def _lifecycle_density_scenario() -> tuple[EnvConfig, EnvState]:
    """Start before concurrent recovery and a status-clearing lethal lifecycle."""
    roster = (
        MAGE_CLASS_ID,
        HUNTER_CLASS_ID,
        PRIEST_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    config, state = _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(3, 1),
        class_ids=roster,
        active_positions={
            0: (13.0, 1.5),
            1: (13.0, 3.0),
            2: (3.0, 10.0),
            5: (14.5, 2.25),
        },
    )
    return config, state._replace(
        current_health=(state.current_health.at[2].set(50.0).at[5].set(5.0)),
        ultimate_cooldowns=state.ultimate_cooldowns.at[2].set(1),
        steps_until_out_of_combat=state.steps_until_out_of_combat.at[2].set(0),
        slow_durations=state.slow_durations.at[
            5,
            SLOW_CHANNEL_WARRIOR_CHARGE,
        ].set(3),
        stun_durations=state.stun_durations.at[
            5,
            STUN_CHANNEL_HUNTER_TRAP,
        ].set(3),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[5].set(3)
        ),
        team_respawn_wave_countdowns=jnp.asarray((4, 2), dtype=jnp.int32),
    )


def _death_respawn_cycle_scenario() -> tuple[EnvConfig, EnvState]:
    """Start immediately before a multi-source lethal and later shielded respawn."""
    roster = (
        MAGE_CLASS_ID,
        HUNTER_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        ROGUE_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    config, state = _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(2, 1),
        class_ids=roster,
        active_positions={
            0: (13.0, 1.5),
            1: (13.0, 3.0),
            5: (14.5, 2.25),
        },
    )
    return config, state._replace(
        current_health=state.current_health.at[5].set(5.0),
        slow_durations=state.slow_durations.at[
            5,
            SLOW_CHANNEL_WARRIOR_CHARGE,
        ].set(3),
        stun_durations=state.stun_durations.at[
            5,
            STUN_CHANNEL_HUNTER_TRAP,
        ].set(3),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[5].set(3)
        ),
        team_respawn_wave_countdowns=jnp.asarray((4, 2), dtype=jnp.int32),
    )


def _recovery_refresh_cycle_scenario() -> tuple[EnvConfig, EnvState]:
    """Start before recovery, cooldown-ready, refresh, and reapply trajectories."""
    roster = (
        ROGUE_CLASS_ID,
        ROGUE_CLASS_ID,
        HUNTER_CLASS_ID,
        HUNTER_CLASS_ID,
        MAGE_CLASS_ID,
        HUNTER_CLASS_ID,
        PRIEST_CLASS_ID,
        WARRIOR_CLASS_ID,
        NEUTRAL_CLASS_ID,
        NEUTRAL_CLASS_ID,
    )
    config, state = _scenario(
        map_width=18.0,
        map_height=12.0,
        team_sizes=(5, 3),
        class_ids=roster,
        active_positions={
            0: (8.0, 4.5),
            1: (8.0, 5.5),
            2: (6.7, 7.0),
            3: (7.5, 8.0),
            4: (8.5, 7.0),
            5: (9.2, 5.0),
            6: (13.0, 9.0),
            7: (9.5, 7.5),
        },
    )
    return config, state._replace(
        current_health=state.current_health.at[6].set(50.0),
        ultimate_cooldowns=state.ultimate_cooldowns.at[6].set(1),
        steps_until_out_of_combat=state.steps_until_out_of_combat.at[6].set(0),
    )


_BASIC_SUPPORT_FRAMES = (
    ScenarioFrame(
        "three-basics",
        "Mage and Hunters apply simultaneous Basic effects.",
        (
            ActorCommand(0, MOVE_STAY, 5, 0),
            ActorCommand(1, MOVE_STAY, 6, 0),
            ActorCommand(7, MOVE_STAY, 2, 0),
        ),
    ),
    ScenarioFrame(
        "simultaneous-heal-damage",
        "Priest self-heal exceeds simultaneous Hunter damage by 2 HP.",
        (
            ActorCommand(2, MOVE_STAY, 2, 0),
            ActorCommand(7, MOVE_STAY, 2, 0),
        ),
    ),
)

_ULTIMATE_SHOWCASE_FRAMES = (
    ScenarioFrame(
        "mage-basic",
        "The opposing Mage damages the allied Hunter before the showcase.",
        (ActorCommand(5, MOVE_STAY, 2, 0),),
    ),
    ScenarioFrame(
        "five-ultimates",
        "All five Team A classes activate their Ultimate simultaneously.",
        (
            ActorCommand(0, MOVE_STAY, None, 1),
            ActorCommand(1, MOVE_STAY, 7, 1),
            ActorCommand(2, MOVE_STAY, 6, 1),
            ActorCommand(3, MOVE_STAY, 5, 1),
            ActorCommand(4, MOVE_STAY, 2, 1),
        ),
    ),
    ScenarioFrame(
        "break-trap",
        "Hunter Basic breaks the active Freezing Trap and applies its slow.",
        (ActorCommand(2, MOVE_STAY, 6, 0),),
    ),
    ScenarioFrame(
        "age-ultimate-statuses-one",
        "A neutral transition ages every remaining Ultimate status.",
        (),
    ),
    ScenarioFrame(
        "age-ultimate-statuses-two",
        "A second neutral transition continues the public durations.",
        (),
    ),
    ScenarioFrame(
        "age-ultimate-statuses-three",
        "Shorter Ultimate statuses reach their natural expiry.",
        (),
    ),
    ScenarioFrame(
        "age-ultimate-statuses-four",
        "The final long-duration Ultimate statuses age to zero.",
        (),
    ),
)

_AURA_CROSSFIRE_FRAMES = (
    ScenarioFrame(
        "reciprocal-hunter-basics",
        "Both Hunters fire through simultaneous Sorcerer\u2019s Empowerment and "
        "Guardian\u2019s Barrier auras.",
        (
            ActorCommand(2, MOVE_STAY, 7, 0),
            ActorCommand(7, MOVE_STAY, 2, 0),
        ),
    ),
)

_STACKED_TEAM_AURAS_FRAMES = (
    ScenarioFrame(
        "reciprocal-stacked-aura-basics",
        "Both Hunters fire through two same-team Mage and Warrior emitters.",
        (
            ActorCommand(4, MOVE_STAY, 9, 0),
            ActorCommand(9, MOVE_STAY, 4, 0),
        ),
    ),
)

_STATUS_STACK_FRAMES = (
    ScenarioFrame(
        "stack",
        "Charge, Freezing Trap, Crippling Poison, and Freedom land on one recipient.",
        (
            ActorCommand(0, MOVE_NORTH, 5, 1),
            ActorCommand(1, MOVE_STAY, 5, 1),
            ActorCommand(2, MOVE_STAY, 5, 1),
            ActorCommand(6, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "break-and-refresh",
        "Hunter Basic breaks Freezing Trap while the stunned recipient stays still.",
        (
            ActorCommand(1, MOVE_STAY, 5, 0),
            ActorCommand(5, MOVE_STAY, None, 0),
            ActorCommand(6, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "freedom-speed",
        "The recipient moves with the current Freedom speed floor.",
        (ActorCommand(5, MOVE_EAST, None, 0),),
    ),
    ScenarioFrame(
        "stacked-slow-speed",
        "The recipient moves under the remaining multiplicative slows.",
        (ActorCommand(5, MOVE_EAST, None, 0),),
    ),
)


_TEAM_FOCUS_CROSSFIRE_FRAMES = (
    ScenarioFrame(
        "balanced-warrior-priest-resolution",
        "An opposing Warrior Basic and allied Priest Basic resolve to no net "
        "health change at the health cap.",
        (
            ActorCommand(1, MOVE_STAY, 5, 0),
            ActorCommand(6, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "first-hunter-basic",
        "The Hunter opens repeated fire on the opposing Warrior.",
        (ActorCommand(2, MOVE_STAY, 5, 0),),
    ),
    ScenarioFrame(
        "repeated-hunter-basic",
        "The same Hunter repeats its Basic against the same recipient.",
        (ActorCommand(2, MOVE_STAY, 5, 0),),
    ),
    ScenarioFrame(
        "focus-fire-and-healing",
        "Four attackers focus one Warrior while three Priests heal it.",
        (
            ActorCommand(0, MOVE_STAY, 5, 0),
            ActorCommand(1, MOVE_STAY, 5, 0),
            ActorCommand(2, MOVE_STAY, 5, 0),
            ActorCommand(3, MOVE_STAY, 5, 0),
            ActorCommand(6, MOVE_STAY, 5, 0),
            ActorCommand(7, MOVE_STAY, 5, 0),
            ActorCommand(8, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "poison-and-same-epoch-healing",
        "Crippling Poison lands while three precommitted Priest Basics still heal.",
        (
            ActorCommand(3, MOVE_STAY, 5, 1),
            ActorCommand(6, MOVE_STAY, 5, 0),
            ActorCommand(7, MOVE_STAY, 5, 0),
            ActorCommand(8, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "current-anti-heal",
        "Three Priest Basics heal under the now-current Crippling Poison anti-heal.",
        (
            ActorCommand(6, MOVE_STAY, 5, 0),
            ActorCommand(7, MOVE_STAY, 5, 0),
            ActorCommand(8, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "three-holy-words",
        "All three Priests use Holy Word: Salvation into the allied Warrior's "
        "health cap.",
        (
            ActorCommand(6, MOVE_STAY, 5, 1),
            ActorCommand(7, MOVE_STAY, 5, 1),
            ActorCommand(8, MOVE_STAY, 5, 1),
        ),
    ),
)

_MIRRORED_ULTIMATES_FRAMES = (
    ScenarioFrame(
        "mirrored-bursts",
        "Both Mages activate Burst simultaneously.",
        (
            ActorCommand(0, MOVE_NORTH, None, 1),
            ActorCommand(5, MOVE_NORTH, None, 1),
        ),
    ),
    ScenarioFrame(
        "reciprocal-charges",
        "The opposing Warriors Charge one another.",
        (
            ActorCommand(1, MOVE_STAY, 6, 1),
            ActorCommand(6, MOVE_STAY, 1, 1),
        ),
    ),
    ScenarioFrame(
        "reciprocal-traps",
        "The opposing Hunters use Freezing Trap on one another.",
        (
            ActorCommand(2, MOVE_NORTH, 7, 1),
            ActorCommand(7, MOVE_NORTH, 2, 1),
        ),
    ),
    ScenarioFrame(
        "reciprocal-poisons",
        "The opposing Rogues use Crippling Poison on one another.",
        (
            ActorCommand(3, MOVE_EAST, 8, 1),
            ActorCommand(8, MOVE_EAST, 3, 1),
        ),
    ),
    ScenarioFrame(
        "mirrored-holy-words",
        "Each Priest uses Holy Word: Salvation on its allied damaged Rogue.",
        (
            ActorCommand(4, MOVE_NORTH, 3, 1),
            ActorCommand(9, MOVE_NORTH, 8, 1),
        ),
    ),
)

_MOVING_BASIC_CROSSFIRE_FRAMES = (
    ScenarioFrame(
        "north-east-crossfire",
        "Every class pair moves north or east through reciprocal Basics and healing.",
        (
            ActorCommand(0, MOVE_NORTH, 5, 0),
            ActorCommand(1, MOVE_EAST, 6, 0),
            ActorCommand(2, MOVE_NORTH, 7, 0),
            ActorCommand(3, MOVE_EAST, 8, 0),
            ActorCommand(4, MOVE_NORTH, 0, 0),
            ActorCommand(5, MOVE_NORTH, 0, 0),
            ActorCommand(6, MOVE_EAST, 1, 0),
            ActorCommand(7, MOVE_NORTH, 2, 0),
            ActorCommand(8, MOVE_EAST, 3, 0),
            ActorCommand(9, MOVE_NORTH, 5, 0),
        ),
    ),
    ScenarioFrame(
        "south-west-crossfire",
        "The same pairs reverse south or west and repeat their completed effects.",
        (
            ActorCommand(0, MOVE_SOUTH, 5, 0),
            ActorCommand(1, MOVE_WEST, 6, 0),
            ActorCommand(2, MOVE_SOUTH, 7, 0),
            ActorCommand(3, MOVE_WEST, 8, 0),
            ActorCommand(4, MOVE_SOUTH, 0, 0),
            ActorCommand(5, MOVE_SOUTH, 0, 0),
            ActorCommand(6, MOVE_WEST, 1, 0),
            ActorCommand(7, MOVE_SOUTH, 2, 0),
            ActorCommand(8, MOVE_WEST, 3, 0),
            ActorCommand(9, MOVE_SOUTH, 5, 0),
        ),
    ),
)

_MOVING_FOCUS_CROSSFIRE_FRAMES = (
    ScenarioFrame(
        "balanced-warrior-priest-resolution",
        "An opposing Warrior Basic and allied Priest Basic resolve to no net "
        "health change at the health cap.",
        (
            ActorCommand(1, MOVE_STAY, 5, 0),
            ActorCommand(6, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "eastbound-focus-fire-and-healing",
        "Four attackers, three healers, and their shared recipient move east together.",
        (
            ActorCommand(0, MOVE_EAST, 5, 0),
            ActorCommand(1, MOVE_EAST, 5, 0),
            ActorCommand(2, MOVE_EAST, 5, 0),
            ActorCommand(3, MOVE_EAST, 5, 0),
            ActorCommand(5, MOVE_EAST, None, 0),
            ActorCommand(6, MOVE_EAST, 5, 0),
            ActorCommand(7, MOVE_EAST, 5, 0),
            ActorCommand(8, MOVE_EAST, 5, 0),
        ),
    ),
)

_CHARGE_CONVERGENCE_FRAMES = (
    ScenarioFrame(
        "three-converging-charges",
        "Two Warriors Charge one target while that target Charges one back.",
        (
            ActorCommand(0, MOVE_STAY, 5, 1),
            ActorCommand(1, MOVE_STAY, 5, 1),
            ActorCommand(5, MOVE_STAY, 0, 1),
        ),
    ),
)

_TRAP_LIFECYCLE_FRAMES = (
    ScenarioFrame(
        "four-trap-applications",
        "Four Hunters use Freezing Trap on four adjacent Warriors.",
        (
            ActorCommand(0, MOVE_STAY, 5, 1),
            ActorCommand(1, MOVE_STAY, 6, 1),
            ActorCommand(2, MOVE_STAY, 7, 1),
            ActorCommand(3, MOVE_STAY, 8, 1),
        ),
    ),
    ScenarioFrame(
        "exact-trap-break",
        "A Hunter Basic breaks a Freezing Trap with more than one tick remaining.",
        (ActorCommand(0, MOVE_STAY, 5, 0),),
    ),
    ScenarioFrame(
        "neutral-aging-transition",
        "A canonical neutral transition ages the remaining Freezing Traps.",
        (),
    ),
    ScenarioFrame(
        "trap-reapplication",
        "The fifth Hunter reapplies Freezing Trap to the second target.",
        (ActorCommand(4, MOVE_STAY, 6, 1),),
    ),
    ScenarioFrame(
        "ambiguous-end-and-expiry",
        "Damage accompanies one duration-one ending while another expires.",
        (ActorCommand(2, MOVE_STAY, 7, 0),),
    ),
)

_MAX_STATUS_STACK_FRAMES = (
    ScenarioFrame(
        "maximum-compatible-status-stack",
        "All compatible control, slow, modifier, and Burst channels land together.",
        (
            ActorCommand(0, MOVE_STAY, None, 1),
            ActorCommand(1, MOVE_STAY, 0, 0),
            ActorCommand(5, MOVE_STAY, 0, 1),
            ActorCommand(6, MOVE_STAY, 0, 1),
            ActorCommand(7, MOVE_STAY, 0, 0),
            ActorCommand(8, MOVE_STAY, 0, 1),
        ),
    ),
    ScenarioFrame(
        "salvation-on-damaged-mage",
        "The Priest uses Holy Word: Salvation on the damaged, fully statused Mage.",
        (ActorCommand(1, MOVE_STAY, 0, 1),),
    ),
    ScenarioFrame(
        "age-status-stack-one",
        "A neutral transition ages the remaining status stack.",
        (),
    ),
    ScenarioFrame(
        "age-status-stack-two",
        "A second neutral transition continues the public durations.",
        (),
    ),
    ScenarioFrame(
        "age-status-stack-three",
        "The shorter remaining status tokens naturally expire.",
        (),
    ),
    ScenarioFrame(
        "age-status-stack-four",
        "The longest status tokens naturally expire.",
        (),
    ),
    ScenarioFrame(
        "expired-status-stack",
        "A final neutral transition proves the recipient remains clear.",
        (),
    ),
)

_LIFECYCLE_DENSITY_FRAMES = (
    ScenarioFrame(
        "lethal-recovery-and-readiness",
        "Two sources kill a statused Rogue while a damaged Priest regenerates "
        "and becomes Ultimate-ready.",
        (
            ActorCommand(0, MOVE_STAY, 5, 0),
            ActorCommand(1, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "corpse-awaits-respawn-wave",
        "The cleared corpse remains out until the next Team B wave.",
        (),
    ),
    ScenarioFrame(
        "respawn-with-spawn-shield",
        "The due Team B wave respawns the Rogue with Spawn Shield.",
        (),
    ),
    ScenarioFrame(
        "age-spawn-shield-one",
        "A neutral transition ages the visible Spawn Shield.",
        (),
    ),
    ScenarioFrame(
        "age-spawn-shield-two",
        "A second neutral transition leaves one protected tick.",
        (),
    ),
    ScenarioFrame(
        "expire-spawn-shield",
        "The final neutral transition expires Spawn Shield.",
        (),
    ),
)

_DEATH_RESPAWN_CYCLE_FRAMES = (
    ScenarioFrame(
        "multi-source-lethal",
        "Mage and Hunter contributions kill a statused Rogue.",
        (
            ActorCommand(0, MOVE_STAY, 5, 0),
            ActorCommand(1, MOVE_STAY, 5, 0),
        ),
    ),
    ScenarioFrame(
        "corpse-waits-for-wave",
        "The corpse remains out until its team's next due wave.",
        (),
    ),
    ScenarioFrame(
        "due-wave-respawn",
        "A corpse submission is rejected while the due wave restores it.",
        (ActorCommand(5, MOVE_WEST, 0, 1),),
    ),
    ScenarioFrame(
        "shielded-movement-and-rejection",
        "The protected Rogue may move while combat intent is rejected.",
        (ActorCommand(5, MOVE_WEST, 0, 0),),
    ),
    ScenarioFrame(
        "shield-countdown-two-to-one",
        "The invulnerable shield remains visible with one tick left.",
        (ActorCommand(5, MOVE_STAY, 0, 0),),
    ),
    ScenarioFrame(
        "shield-expiry",
        "Transition-start protection still rejects combat before expiring.",
        (ActorCommand(5, MOVE_STAY, 0, 0),),
    ),
    ScenarioFrame(
        "first-unshielded-interaction",
        "The Rogue's first post-shield Basic is accepted normally.",
        (ActorCommand(5, MOVE_STAY, 0, 0),),
    ),
)

_RECOVERY_REFRESH_CYCLE_FRAMES = (
    ScenarioFrame(
        "application-recovery-and-readiness",
        "Crippling Poison and Freezing Trap apply while a damaged Priest "
        "regenerates and becomes ready.",
        (
            ActorCommand(0, MOVE_STAY, 5, 1),
            ActorCommand(2, MOVE_STAY, 7, 1),
            ActorCommand(6, MOVE_STAY, 6, 1),
        ),
    ),
    ScenarioFrame(
        "refresh-and-break-reapplication",
        "Crippling Poison refreshes while Mage damage breaks and Hunter reapplies "
        "Freezing Trap.",
        (
            ActorCommand(1, MOVE_STAY, 5, 1),
            ActorCommand(3, MOVE_STAY, 7, 1),
            ActorCommand(4, MOVE_STAY, 7, 0),
        ),
    ),
    ScenarioFrame("age-one", "Current effects age without new applications.", ()),
    ScenarioFrame("age-two", "Current effects continue their public duration.", ()),
    ScenarioFrame(
        "age-three", "Freezing Trap reaches its final active decision epoch.", ()
    ),
    ScenarioFrame("trap-expiry", "Freezing Trap ages from one tick to zero.", ()),
    ScenarioFrame(
        "poison-expiry", "Crippling Poison slow ages from one tick to zero.", ()
    ),
)


RESEARCHER_SCENARIOS: dict[str, DebuggerScenario] = {
    "arena_5v5": _registered_scenario(
        "arena_5v5",
        build_scenario=_arena_5v5_scenario,
        frames=(),
    ),
    "basic_support": _registered_scenario(
        "basic_support",
        build_scenario=_basic_support_scenario,
        frames=_BASIC_SUPPORT_FRAMES,
    ),
    "ultimate_showcase": _registered_scenario(
        "ultimate_showcase",
        build_scenario=_ultimate_showcase_scenario,
        frames=_ULTIMATE_SHOWCASE_FRAMES,
    ),
    "aura_crossfire": _registered_scenario(
        "aura_crossfire",
        build_scenario=_aura_crossfire_scenario,
        frames=_AURA_CROSSFIRE_FRAMES,
    ),
    "stacked_team_auras": _registered_scenario(
        "stacked_team_auras",
        build_scenario=_stacked_team_auras_scenario,
        frames=_STACKED_TEAM_AURAS_FRAMES,
    ),
    "status_stack": _registered_scenario(
        "status_stack",
        build_scenario=_status_stack_scenario,
        frames=_STATUS_STACK_FRAMES,
    ),
    "team_focus_crossfire": _registered_scenario(
        "team_focus_crossfire",
        build_scenario=_team_focus_crossfire_scenario,
        frames=_TEAM_FOCUS_CROSSFIRE_FRAMES,
    ),
    "mirrored_ultimates": _registered_scenario(
        "mirrored_ultimates",
        build_scenario=_mirrored_ultimates_scenario,
        frames=_MIRRORED_ULTIMATES_FRAMES,
    ),
    "death_respawn_cycle": _registered_scenario(
        "death_respawn_cycle",
        build_scenario=_death_respawn_cycle_scenario,
        frames=_DEATH_RESPAWN_CYCLE_FRAMES,
    ),
    "recovery_refresh_cycle": _registered_scenario(
        "recovery_refresh_cycle",
        build_scenario=_recovery_refresh_cycle_scenario,
        frames=_RECOVERY_REFRESH_CYCLE_FRAMES,
    ),
}


STRESS_SCENARIOS: dict[str, DebuggerScenario] = {
    "moving_basic_crossfire": _registered_scenario(
        "moving_basic_crossfire",
        build_scenario=_moving_basic_crossfire_scenario,
        frames=_MOVING_BASIC_CROSSFIRE_FRAMES,
    ),
    "moving_focus_crossfire": _registered_scenario(
        "moving_focus_crossfire",
        build_scenario=_moving_focus_crossfire_scenario,
        frames=_MOVING_FOCUS_CROSSFIRE_FRAMES,
    ),
    "charge_convergence": _registered_scenario(
        "charge_convergence",
        build_scenario=_charge_convergence_scenario,
        frames=_CHARGE_CONVERGENCE_FRAMES,
    ),
    "trap_lifecycle": _registered_scenario(
        "trap_lifecycle",
        build_scenario=_trap_lifecycle_scenario,
        frames=_TRAP_LIFECYCLE_FRAMES,
    ),
    "max_status_stack": _registered_scenario(
        "max_status_stack",
        build_scenario=_max_status_stack_scenario,
        frames=_MAX_STATUS_STACK_FRAMES,
    ),
    "lifecycle_density": _registered_scenario(
        "lifecycle_density",
        build_scenario=_lifecycle_density_scenario,
        frames=_LIFECYCLE_DENSITY_FRAMES,
    ),
}


SCENARIOS: dict[str, DebuggerScenario] = {
    **RESEARCHER_SCENARIOS,
    **STRESS_SCENARIOS,
}


def get_scenario(name: str) -> DebuggerScenario:
    """Find one registered live debugger scenario.

    Parameters
    ----------
    name : str
        Exact registry name.

    Returns
    -------
    DebuggerScenario
        Definition with a state factory and optional scripted frames. Lookup
        alone does not call the factory.

    Raises
    ------
    ValueError
        The name is unknown; the error lists available names.
    """
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        choices = ", ".join(SCENARIOS)
        msg = f"unknown scenario {name!r}; choose one of: {choices}."
        raise ValueError(msg) from exc


def list_scenarios(
    *,
    include_stress: bool = False,
) -> tuple[DebuggerScenario, ...]:
    """List scenario definitions in stable launcher order.

    Parameters
    ----------
    include_stress : bool, optional
        Include dense stress demonstrations when True. Defaults to False.

    Returns
    -------
    tuple of DebuggerScenario
        Registered definitions without constructing their JAX states.
    """
    registry = SCENARIOS if include_stress else RESEARCHER_SCENARIOS
    return tuple(registry.values())


def cycle_scenario_name(
    current_name: str,
    direction: int,
    *,
    include_stress: bool = False,
) -> str:
    """Choose the previous or next allowed scenario, wrapping at either end.

    Parameters
    ----------
    current_name : str
        Registered current scenario.
    direction : int
        Minus one for previous or one for next.
    include_stress : bool, optional
        Whether navigation includes stress scenes. Defaults to False. A current
        stress scene also selects the complete registry so navigation remains valid.

    Returns
    -------
    str
        Adjacent registry name in stable launcher order.

    Raises
    ------
    ValueError
        The direction or current name is unknown.
    """
    if direction not in (-1, 1):
        msg = f"scenario direction must be -1 or 1; got {direction}."
        raise ValueError(msg)
    registry = (
        SCENARIOS
        if include_stress or current_name in STRESS_SCENARIOS
        else RESEARCHER_SCENARIOS
    )
    names = tuple(registry)
    try:
        current_index = names.index(current_name)
    except ValueError as exc:
        msg = f"unknown current scenario {current_name!r}."
        raise ValueError(msg) from exc
    return names[(current_index + direction) % len(names)]


# The Vision Moment: the frozen frame of the approved BattleClient vision page
# (map 47 Teutoburg Forest, tick 91 of a recorded game whose spawn ends were
# exchanged, so Team A starts on the right). Values are slot-ordered, Team A
# slots 0-4 then Team B slots 5-9, each team Mage, Warrior, Hunter, Rogue,
# Priest. Positions are world units; durations and cooldowns are ticks.
VISION_MOMENT_MAP_ID = 47
VISION_MOMENT_TICK = 91
VISION_MOMENT_POSITIONS = (
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
VISION_MOMENT_HEALTH = (
    80.0,
    200.0,
    100.0,
    100.0,
    100.0,
    77.0,
    200.0,
    89.8,
    100.0,
    70.0,
)
VISION_MOMENT_ULTIMATE_COOLDOWNS = (16, 0, 1, 20, 19, 30, 0, 0, 0, 0)
VISION_MOMENT_SPAWN_SHIELDS = (0, 0, 0, 2, 0, 0, 0, 0, 0, 0)
VISION_MOMENT_BURST = (0, 0, 0, 0, 0, 5, 0, 0, 0, 0)
VISION_MOMENT_FREEDOM = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0)
VISION_MOMENT_OUT_OF_COMBAT = (0, 0, 0, 0, 0, 5, 0, 1, 0, 5)
VISION_MOMENT_SCORES = (2, 4)
# Respawn-wave clocks: (2, 2), one less than the recorded frame's (3, 3). Core
# brings the fallen back in the transition that starts when a clock reads 0
# (core/env.py, _build_was_respawned_this_transition_by_agent_array), so a
# clock of c means "back in c + 1 ticks". The vision page writes "Respawn Wave
# In 3 Ticks"; the owner's 2 Oct decision sets the clocks so that text is true.
VISION_MOMENT_RESPAWN_WAVES = (2, 2)
# The Vision Moment and its other starts, each a small change away from it so
# the parity tools reach a rare state in one turn instead of a long game:
# "vision" is the vision page's frame; "fall" leaves the Team B Mage (slot 5)
# 1 life and the Team B Priest's Ultimate cooling down, so the staged Charge
# on the Mage makes it fall in the next turn (with the Ultimate ready, the
# tdm-beta Priest saves it: seen 1 Oct 2026); "own_return" has the Team A
# Warrior (slot 1, the Manual champion) fallen with Team A's respawn wave
# due, so it returns in the next turn; "last_tick" is tick 299 of 300, so the
# next turn ends the game; "trap_break" holds the Team B Mage in a Hunter's
# Freezing Trap (3 ticks), so the staged Charge's blow breaks the trap in the
# next turn (Core breaks a Freezing Trap on accepted raw damage; the Mage's
# 77 life survives the blow). The last four make one event of a Visual
# Filters switch happen in the next turn, whatever the policies do:
# "home_fall" (a fall inside the victim's own Red Zone), "step_aside" (your
# Warrior stands on a fallen ally), "into_sight" (an opponent comes into your
# team's sight) and "rest" (a champion regains life out of combat). Three more
# freeze every other champion with a 1-tick stun (Core's longest Charge stun),
# so only the champions they name can act: "barrier" (another champion's
# Guardian's Barrier keeps the Team B Mage alive through the staged Charge:
# Barrier Held, Aura Glints and Last Breath), "saved" (the tdm-beta Priest's
# Holy Word: Salvation keeps the Mage alive through it: Saved Beat) and
# "streak" (five turns in which your Warrior fells the five Team B champions,
# one a turn, its 5th takedown without falling: Champion Streaks).
VISION_MOMENT_STARTS = (
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
# "home_fall": the Team B Mage (slot 5, 1 life) inside Team B's Red Zone
# (x 0 to 5 after the exchanged banks) and the Team A Warrior (slot 1) 3.16
# units east of it, so the staged Charge (reach 5.75) has the Mage as its one
# legal target and the fall is worth 2 points. "step_aside": the fallen Team A
# Hunter's body (slot 2) 0.45 units from the Warrior, closer than the 0.6-unit
# Fallen Step Aside distance (looks/champions.js STEP_ASIDE_DISTANCE).
VISION_MOMENT_HOME_FALL_POSITIONS = ((2.0, 4.0), (5.0, 3.0))
VISION_MOMENT_STEP_ASIDE_BODY = (15.628, 5.123)
# "barrier": the Team B Mage's life. The staged Charge deals 20; the Team B
# Warrior's Guardian's Barrier (x0.85) cuts it to 17, so the Mage keeps 1
# life, and without the barrier the 20 would fell it.
VISION_MOMENT_BARRIER_MAGE_LIFE = 18.0
# "streak": the Team B champions in the order your Warrior fells them (Priest,
# Rogue, Warrior, Hunter, Mage: the healer first, the Rogue before its shorter
# out-of-combat delay ends) and their places, each 1.1 units from the Warrior
# (inside its 1.5-unit Basic reach, at least 1.0 unit from every other body,
# clear of map 47's stones).
VISION_MOMENT_STREAK_ORDER = (9, 8, 6, 7, 5)
VISION_MOMENT_STREAK_POSITIONS = (
    (16.229, 5.767),
    (15.161, 5.588),
    (14.764, 4.580),
    (15.406, 3.728),
    (16.484, 3.832),
)


def _stun_all_but(state: EnvState, free: tuple[int, ...]) -> EnvState:
    """Return the state with every other living champion stunned for 1 tick.

    Parameters
    ----------
    state : EnvState
        A Vision Moment state (not changed; JAX arrays are immutable).
    free : tuple of int
        Global slots (0 to 9) that stay free to act.

    Returns
    -------
    EnvState
        The same state, except each living slot not in ``free`` and with no
        spawn shield gets a Warrior Charge stun of 1 tick
        (``stun_durations[slot, STUN_CHANNEL_WARRIOR_CHARGE]``, Core's
        largest allowed value for that channel), so its only legal action in
        the next tick is to stay with no target, whatever its policy chooses.
        Spawn-shielded slots are left alone: Core refuses a shield and a stun
        on one slot, and a shielded champion cannot strike or be struck.
    """
    stuns = state.stun_durations
    for slot in range(MAX_AGENT_SLOTS):
        if (
            slot not in free
            and bool(state.alive_mask[slot])
            and int(state.spawn_shield_durations[slot]) == 0
        ):
            stuns = stuns.at[slot, STUN_CHANNEL_WARRIOR_CHARGE].set(1)
    return state._replace(stun_durations=stuns)


def build_vision_moment(start: str = "vision") -> tuple[EnvConfig, EnvState]:
    """Build the real map 47 game state that the BattleClient vision page shows.

    The vision page (the owner-approved BattleClient design) freezes one moment
    of a recorded game on map 47, Teutoburg Forest: tick 91 of 300, Team A 2
    and Team B 4 of 20 points. This factory makes that moment a real, playable
    simulator state, so the live BattleClient can open on the same screen and
    Lock In advances it to tick 92 with real controllers.

    Parameters
    ----------
    start : str, default "vision"
        Which start of ``VISION_MOMENT_STARTS``: ``"vision"`` is the vision
        page's frame. Each other start changes one thing so a rare state comes
        in the next turn: ``"fall"`` sets the Team B Mage's life to 1 (it falls
        in the next turn when any blow lands, such as the staged Charge) and
        puts the Team B Priest's Ultimate on a 30-tick cooldown (when ready,
        the tdm-beta Priest's Ultimate saves the Mage);
        ``"own_return"`` makes the Team A Warrior (the Manual champion) fallen
        (dead, 0 life, its body where it stood) with Team A's respawn clock at
        0, so the wave brings it back in the next turn; ``"last_tick"`` sets
        the tick to 299 (the config's ``max_steps`` minus 1), so the next
        turn reaches the 300-tick limit and ends the game; ``"trap_break"``
        puts the Team B Mage (slot 5, 77 of 80 life) in a Hunter's Freezing
        Trap with 3 ticks left (``stun_durations[5, STUN_CHANNEL_HUNTER_TRAP]``),
        so any blow that lands on it in the next turn, such as the staged
        Charge, breaks the trap (Core's trap-break rule) while the Mage lives.
        The last four starts make one event of a Visual Filters switch happen
        in the next turn, whatever the policies choose: ``"home_fall"`` moves
        the Team B Mage (1 life) to (2.0, 4.0), inside Team B's Red Zone, and
        the Team A Warrior to (5.0, 3.0), so the staged Charge on the Mage
        (its one legal target) fells it at home for 2 points (Red Zone Home
        Fall), with the Team B Priest's Ultimate cooling down as in
        ``"fall"``; ``"step_aside"`` makes the Team A Hunter (slot 2) fallen,
        its body at (15.628, 5.123), 0.45 units from the Warrior, and puts the
        Warrior's Ultimate on a 5-tick cooldown, so the Warrior has no legal
        target, stays where it is and stands on the body (Fallen Step Aside);
        ``"into_sight"`` gives the Team B Mage a 1-tick spawn shield, so Team
        A cannot see it now (Core hides shielded opponents) and sees it after
        the next tick (an opponent coming into sight); ``"rest"`` sets the
        Team A Rogue (slot 3, far from every fight and spawn-shielded) to 60 of
        100 life, so it regains 4 life out of combat in the next tick (Rest).
        The last three stun every other living, unshielded champion for 1
        tick (``_stun_all_but``), so the event cannot depend on what those
        champions' policies choose: ``"barrier"`` exchanges the places of the
        Team B Warrior (slot 6) and Hunter (slot 7), so the Warrior stands 1.76
        units from the Team B Mage, inside its 2.0-unit Guardian's Barrier,
        and sets the Mage to 18 life (``VISION_MOMENT_BARRIER_MAGE_LIFE``);
        only your Warrior acts, and its staged Charge (20 damage, cut to 17 by
        the barrier) leaves the Mage 1 life, which it keeps only because of
        another champion's barrier (Barrier Held), with the barrier helping
        an ally of its emitter (Aura Glints) and the Mage losing more than it
        has left (Last Breath); ``"saved"`` sets the Mage to 1 life and leaves
        only your Warrior and the Team B Priest (slot 9, its Ultimate ready)
        free: the staged Charge would fell the Mage, and the tdm-beta Priest
        answers with Holy Word: Salvation on it, so a heal keeps it alive
        through a lethal blow (Saved Beat; the choice is tdm-beta's, which is
        deterministic, and is pinned in the tests); ``"streak"`` is a
        five-turn start for your Warrior's Champion Streak: Team A's other
        four champions are fallen (Team A's respawn clock at 4, the largest
        Core allows, so they are back only in the fifth tick and never act
        before it), and the five Team B champions stand around your Warrior
        at ``VISION_MOMENT_STREAK_POSITIONS`` (in ``VISION_MOMENT_STREAK_ORDER``)
        with 1 life, a 4-tick Freezing Trap each (Core's longest stun), their
        out-of-combat countdowns at their class's full delay (no Rest before
        they fall) and Team B's respawn clock at 4; a Basic on each in that
        order fells one a turn, the fifth whatever the then free Team B Mage
        does, so your Warrior takes part in its 5th takedown without falling
        in the fifth tick. Everything else is as for ``"vision"``.

    Returns
    -------
    tuple of EnvConfig and EnvState
        config is the standard map 47 Team Deathmatch config (canonical
        mirrored rosters, 300 ticks, 20 points, 5.0-unit Red Zones) with the
        two spawn banks exchanged, so Team A spawns, respawns and has its Red
        Zone on the right (x 15 to 20), as in the recorded game. state is a
        Core reset of that config (key 0) with these fields replaced:
        step_count 91; scores (2, 4); all ten champions alive at
        VISION_MOMENT_POSITIONS with VISION_MOMENT_HEALTH; Ultimate
        cooldowns; the Team A Rogue's 2-tick spawn shield (it just respawned);
        the Team B Mage's 5 ticks of Burst; Blessing of Freedom (1 tick) on
        both Mages; respawn-wave clocks (2, 2), so the next wave brings the
        fallen back in 3 ticks, as the vision page says (see Notes); and
        out-of-combat countdowns. This is the ``"vision"`` start; every other
        start then changes only what ``start`` above says.

    Notes
    -----
    The respawn-wave clocks are the one value set apart from the recording,
    which reads (3, 3) at tick 91. Core respawns the fallen in the transition
    that starts when a team's clock reads 0, so a clock of c brings them back
    in c + 1 ticks. The vision page's "Respawn Wave In 3 Ticks" is therefore
    true only for clocks of 2; the owner decided on 2 Oct 2026 that the moment
    must make the vision's text true, so the clocks are (2, 2).

    The event starts exist for the private parity tools: each Visual Filters
    switch must be judged on a turn where its event really happens. Their
    events are proven with Core steps in
    tests/test_battle_client_vision_moment.py (for ``"saved"`` with the
    tdm-beta Priest's own choice, and for ``"streak"`` over its five turns,
    the fifth for every action the Team B Mage may take).

    Out-of-combat countdowns come from the recorded combat of ticks 86 to 90
    (the Team B Mage and Priest 5, the Team B Hunter 1). Slots with no
    recorded combat there start at 0, which means out of combat; they are at
    full health, so this only affects later healing rules. Stun, slow and
    Rogue Poison counters stay at reset's zero, as recorded (only the
    ``"trap_break"``, ``"barrier"``, ``"saved"`` and ``"streak"`` starts set
    stun counters, as said above). Accepted-action
    history stays at reset's neutral values (no previous transition), and
    each policy starts with fresh memory. This is a debugger start, not a
    continuation of the recorded game. This config alone has no map identity:
    a match built from it without a declaration records a custom map. The
    scene from ``vision_moment_scenario`` declares it as packaged map 47 with
    exchanged spawn banks, so the live client checks it against the catalog
    with the banks put back and names it Teutoburg Forest. Calling this
    creates JAX arrays and runs one Core reset; it writes no files.

    Raises
    ------
    ValueError
        The packaged map 47 content or its Core configuration is invalid, or
        ``start`` is not one of ``VISION_MOMENT_STARTS``.
    """
    if start not in VISION_MOMENT_STARTS:
        msg = (
            f"unknown Vision Moment start {start!r}; choose one of "
            f"{', '.join(VISION_MOMENT_STARTS)}."
        )
        raise ValueError(msg)
    team_a_roster, team_b_roster = canonical_tournament_rosters()
    config = _swap_spawn_banks(
        make_standard_team_deathmatch_config(
            map_id=VISION_MOMENT_MAP_ID,
            team_a_roster=team_a_roster,
            team_b_roster=team_b_roster,
        )
    )
    reset_state, _, _, _ = reset(config, jax.random.key(0))
    burst = jnp.asarray(VISION_MOMENT_BURST, dtype=jnp.int32)
    alive: list[bool] = [True] * MAX_AGENT_SLOTS
    health: list[float] = list(VISION_MOMENT_HEALTH)
    waves: list[int] = list(VISION_MOMENT_RESPAWN_WAVES)
    tick = VISION_MOMENT_TICK
    cooldowns: list[int] = list(VISION_MOMENT_ULTIMATE_COOLDOWNS)
    if start == "fall":
        health[5] = 1.0
        cooldowns[9] = 30
    elif start == "own_return":
        # Every per-champion counter of the Warrior is already 0 in the vision
        # frame, as Core requires of a dead row; only life and the clock change.
        alive[1] = False
        health[1] = 0.0
        waves[0] = 0
    elif start == "last_tick":
        tick = int(config.max_steps) - 1
    elif start == "home_fall":
        health[5] = 1.0
        cooldowns[9] = 30
    elif start == "step_aside":
        # A dead row keeps every per-champion counter at 0 (Core), so the
        # Hunter's 1-tick Ultimate cooldown goes to 0.
        alive[2] = False
        health[2] = 0.0
        cooldowns[2] = 0
        cooldowns[1] = 5
    elif start == "rest":
        health[3] = 60.0
    elif start == "barrier":
        health[5] = VISION_MOMENT_BARRIER_MAGE_LIFE
    elif start == "saved":
        health[5] = 1.0
    freedom: list[int] = list(VISION_MOMENT_FREEDOM)
    out_of_combat: list[int] = list(VISION_MOMENT_OUT_OF_COMBAT)
    positions: list[list[float]] = [list(point) for point in VISION_MOMENT_POSITIONS]
    shields: list[int] = list(VISION_MOMENT_SPAWN_SHIELDS)
    if start == "streak":
        # A dead row keeps every per-champion counter at 0 (Core).
        delays = jnp.asarray(OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS)[
            config.agent_profile.class_ids
        ]
        for slot in (0, 2, 3, 4):
            alive[slot] = False
            health[slot] = 0.0
            cooldowns[slot] = 0
            freedom[slot] = 0
            shields[slot] = 0
            out_of_combat[slot] = 0
        for slot, point in zip(
            VISION_MOMENT_STREAK_ORDER, VISION_MOMENT_STREAK_POSITIONS, strict=True
        ):
            positions[slot] = list(point)
            health[slot] = 1.0
            out_of_combat[slot] = int(delays[slot])
        waves = [
            int(period) - 1 for period in config.team_respawn_wave_period_step_count
        ]
    if start == "home_fall":
        positions[5], positions[1] = (
            list(p) for p in VISION_MOMENT_HOME_FALL_POSITIONS
        )
    elif start == "step_aside":
        positions[2] = list(VISION_MOMENT_STEP_ASIDE_BODY)
    elif start == "into_sight":
        shields[5] = 1
    elif start == "barrier":
        positions[6], positions[7] = positions[7], positions[6]
    state = reset_state._replace(
        team_deathmatch_scores=jnp.asarray(VISION_MOMENT_SCORES, dtype=jnp.int32),
        step_count=jnp.asarray(tick, dtype=jnp.int32),
        agent_positions=jnp.asarray(positions, dtype=jnp.float32),
        alive_mask=jnp.asarray(alive, dtype=jnp.bool_),
        current_health=jnp.asarray(health, dtype=jnp.float32),
        ultimate_cooldowns=jnp.asarray(cooldowns, dtype=jnp.int32),
        mage_burst_damage_amplification_durations=burst,
        priest_blessing_of_freedom_slow_floor_durations=jnp.asarray(
            freedom, dtype=jnp.int32
        ),
        team_respawn_wave_countdowns=jnp.asarray(waves, dtype=jnp.int32),
        spawn_shield_durations=jnp.asarray(shields, dtype=jnp.int32),
        steps_until_out_of_combat=jnp.asarray(out_of_combat, dtype=jnp.int32),
    )
    if start == "trap_break":
        state = state._replace(
            stun_durations=state.stun_durations.at[5, STUN_CHANNEL_HUNTER_TRAP].set(3)
        )
    elif start == "barrier":
        state = _stun_all_but(state, (1,))
    elif start == "saved":
        state = _stun_all_but(state, (1, 9))
    elif start == "streak":
        state = state._replace(
            stun_durations=state.stun_durations.at[
                jnp.asarray(VISION_MOMENT_STREAK_ORDER), STUN_CHANNEL_HUNTER_TRAP
            ].set(4)
        )
    return config, state


def vision_moment_scenario(start: str = "vision") -> DebuggerScenario:
    """Return the Vision Moment (or one of its other starts) as a live scenario.

    Parameters
    ----------
    start : str, default "vision"
        One of ``VISION_MOMENT_STARTS``; see ``build_vision_moment``.

    Returns
    -------
    DebuggerScenario
        Named "vision_moment" (or "vision_moment_<start>" for another start),
        titled "Teutoburg Forest", interactive, with build_vision_moment for
        that start as its factory, no scripted frames and the Team A Warrior
        (slot 1, the vision's Manual champion) focused at launch. It
        declares packaged map 47 (the catalog entry, ``packaged_map``) with
        its spawn banks exchanged (``spawn_banks_exchanged`` True), so the
        live client checks the config against the catalog with the banks put
        back and names the map "Teutoburg Forest" (Map 47) in the top bar
        and the match record, which also records the exchange; if that check
        failed, the client would record a custom map. It carries no scenario
        specification or provenance. The factory is not called here.

    Raises
    ------
    ValueError
        ``start`` is not one of ``VISION_MOMENT_STARTS``.

    Notes
    -----
    This scene is not in SCENARIOS, so the DevClient's scenario list and
    cycling do not offer it; the private BattleClient parity tools pass it to
    the live launch directly.
    """
    if start not in VISION_MOMENT_STARTS:
        msg = (
            f"unknown Vision Moment start {start!r}; choose one of "
            f"{', '.join(VISION_MOMENT_STARTS)}."
        )
        raise ValueError(msg)
    description = (
        "Map 47 at tick 91 of 300, the BattleClient vision page's frozen frame."
    )
    if start != "vision":
        description += f" Start {start!r} (see build_vision_moment)."

    def build() -> tuple[EnvConfig, EnvState]:
        """Build this start of the Vision Moment (see build_vision_moment)."""
        return build_vision_moment(start)

    return DebuggerScenario(
        name="vision_moment" if start == "vision" else f"vision_moment_{start}",
        title="Teutoburg Forest",
        description=description,
        mode="interactive",
        build_scenario=build_vision_moment if start == "vision" else build,
        frames=(),
        default_controlled_slot=1,
        packaged_map=list_tdm_maps()[VISION_MOMENT_MAP_ID],
        spawn_banks_exchanged=True,
    )
