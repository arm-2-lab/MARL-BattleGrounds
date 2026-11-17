"""Measure Capture the Flag's own play: who carries, who escorts, how carries end.

One pure helper, capture_the_flag_tick, reads one tick's start-of-tick state into
the masks every Capture the Flag column uses: who is eligible (configured, alive
and without a spawn shield), who carries, who stands near its team's carrier,
and how many teammates stand near each agent. "Near" is within NEAR_DISTANCE map
units and in clear sight (Core's has_clear_line_of_sight, the fog-of-war test).
initialize_capture_the_flag and update_capture_the_flag keep a game's running
totals and a little memory (the open carries, who dropped each flag since it
left home, last tick's Freedom casts) in fixed shapes; capture_the_flag_data
reads them as values keyed by scope and stem, which full metrics pack in the
catalog's order (capture_the_flag_catalog.py). Flag events come from Core's
flag facts, never from a change in the carrier slot, because a carrier killed
on a tick can still capture with its last move. Combat amounts come from the
tables full metrics already build for the tick. Nothing runs on the host per
tick, and nothing is kept per tick.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.geometry import has_clear_line_of_sight
from marl_battlegrounds.core.types import EnvConfig, EnvState, Info
from marl_battlegrounds.evaluation._metrics.episode_metrics import MetricValues
from marl_battlegrounds.evaluation.combat_metrics import CombatQuantities
from marl_battlegrounds.evaluation.metric_catalog import MetricScope

type CaptureTheFlagTotals = dict[str, Array]

NEAR_DISTANCE = 3.0
"""Largest center-to-center distance, in map units, at which an agent is near a
carrier or another agent (a chosen value, the same for every class)."""

MAX_ALLIES_NEAR = 4
"""Most teammates that can be near an agent: a team has five slots."""

DEATH_WINDOWS = (5, 10, 20)
"""Carry lengths, in ticks, for the counts of carries ending in death within them."""

_SLOTS = jnp.arange(10)
_TEAM = _SLOTS // 5
_SAME_TEAM = (_TEAM[:, None] == _TEAM[None, :]) & (_SLOTS[:, None] != _SLOTS[None, :])


def _length(vectors: Array) -> Array:
    """Return the Euclidean length of each vector along the last axis."""
    return jnp.sqrt(jnp.square(vectors).sum(axis=-1))


def _count(mask: Array) -> Array:
    """Return a boolean mask as int32 ones and zeros, for adding to a count."""
    return mask.astype(jnp.int32)


def escort_split(size: int) -> str:
    """Name one escort size as its column names write it.

    Parameters
    ----------
    size : int
        How many of the carrier's teammates were near it, 0 to MAX_ALLIES_NEAR.

    Returns
    -------
    str
        "with_1_ally_near" for one teammate, else "with_{size}_allies_near"
        (for example "with_0_allies_near"), as the approved inventory names them.
    """
    return "with_1_ally_near" if size == 1 else f"with_{size}_allies_near"


class CaptureTheFlagTick(NamedTuple):
    """Hold one tick's Capture the Flag situations, read at the start of the tick.

    Attributes
    ----------
    eligible : Array
        bool (10,): configured, alive and without a spawn shield.
    carrying : Array
        bool (10,): the agent holds the enemy flag.
    team_has_carrier : Array
        bool (10,): a teammate (or the agent) holds the enemy flag.
    ally_carrier : Array
        int32 (10,): the global slot of the agent's team's carrier, or 0 when
        team_has_carrier is False (a safe index, always masked by it).
    distances : Array
        float32 (10, 10): center-to-center distances in map units.
    near : Array
        bool (10, 10): j is eligible, another agent, within NEAR_DISTANCE of i
        and in clear sight of it. Not symmetric in eligibility: row i does not
        check i itself.
    allies_near : Array
        int32 (10,): how many teammates are near the agent, 0 to 4.
    near_the_ally_carrier : Array
        bool (10,): the agent is eligible, not the carrier, and within
        NEAR_DISTANCE of its team's carrier and in clear sight of it.
    closest_ally_distance : Array
        float32 (10,): distance to the closest eligible teammate, or 0 where
        has_eligible_ally is False.
    has_eligible_ally : Array
        bool (10,): at least one other teammate is eligible.
    """

    eligible: Array
    carrying: Array
    team_has_carrier: Array
    ally_carrier: Array
    distances: Array
    near: Array
    allies_near: Array
    near_the_ally_carrier: Array
    closest_ally_distance: Array
    has_eligible_ally: Array


def capture_the_flag_tick(config: EnvConfig, state: EnvState) -> CaptureTheFlagTick:
    """Read one tick's Capture the Flag situations from its start-of-tick state.

    Parameters
    ----------
    config : EnvConfig
        The game's config: its active slots and obstacles.
    state : EnvState
        Core's state at the start of the tick, before the tick's actions.

    Returns
    -------
    CaptureTheFlagTick
        The tick's masks, distances and counts (see its attributes).

    Notes
    -----
    Pure JAX with fixed shapes; an outer vmap adds games. The sight test runs
    once for each ordered pair of agents (100 segments against the 32 obstacle
    rows). A carrier slot of -1 is masked before it is used as an index,
    because JAX reads slot 9 for -1 without an error.
    """
    eligible = (
        config.agent_profile.active_mask
        & state.alive_mask
        & (state.spawn_shield_durations == 0)
    )
    carriers = state.capture_the_flag_flag_carrier_slots
    # Flag t is team t's own flag, so a team's carrier holds the other flag.
    team_carrier = carriers[1 - _TEAM]
    team_has_carrier = team_carrier >= 0
    ally_carrier = jnp.where(team_has_carrier, team_carrier, 0)
    carrying = team_has_carrier & (ally_carrier == _SLOTS)
    positions = state.agent_positions
    distances = _length(positions[:, None, :] - positions[None, :, :])
    sight = jax.vmap(
        jax.vmap(has_clear_line_of_sight, in_axes=(None, 0, None)),
        in_axes=(0, None, None),
    )(positions, positions, config.obstacles)
    near = (
        eligible[None, :]
        & (_SLOTS[:, None] != _SLOTS[None, :])
        & (distances <= NEAR_DISTANCE)
        & sight
    )
    allies_near = (near & _SAME_TEAM).sum(axis=1)
    near_the_ally_carrier = (
        eligible
        & team_has_carrier
        & ~carrying
        & near[_SLOTS, ally_carrier]
        & eligible[ally_carrier]
    )
    ally_options = _SAME_TEAM & eligible[None, :]
    has_eligible_ally = ally_options.any(axis=1)
    closest_ally_distance = jnp.where(
        has_eligible_ally,
        jnp.where(ally_options, distances, jnp.inf).min(axis=1),
        0.0,
    )
    return CaptureTheFlagTick(
        eligible,
        carrying,
        team_has_carrier,
        ally_carrier,
        distances,
        near,
        allies_near,
        near_the_ally_carrier,
        closest_ally_distance,
        has_eligible_ally,
    )


# Running counts (int32) and amounts (float32), by global slot unless noted.
# The "*_sum" amounts pair with "*_observations" counts for means.
_COUNT_SHAPES = {
    "flag_steals": (10,),
    "dropped_enemy_flag_pickups": (10,),
    "flag_captures": (10,),
    "steps_carrying": (10,),
    "deaths_carrying": (10,),
    "carry_duration_steps": (10,),
    "carry_duration_observations": (10,),
    "carry_steps_ending_in_a_capture": (10,),
    "carry_steps_ending_in_death": (10,),
    "carry_speed_observations": (10,),
    "carry_steps_in_capture_range_with_the_ally_flag_away": (10,),
    "captures_by_an_ally_after_it_dropped": (10,),
    "captures_after_an_ally_dropped": (10,),
    "steps_with_allies_near_carrying": (10, MAX_ALLIES_NEAR + 1),
    "flag_captures_with_allies_near": (10, MAX_ALLIES_NEAR + 1),
    "deaths_with_allies_near_carrying": (10, MAX_ALLIES_NEAR + 1),
    "carries_ending_in_death_within": (10, len(DEATH_WINDOWS)),
    "steps_near_the_ally_carrier": (10,),
    "solo_steps_near_the_ally_carrier": (10,),
    "team_steps_with_an_agent_near_the_ally_carrier": (2,),
    "steps_attacking_near_the_ally_carrier": (10,),
    "solo_steps_attacking_near_the_ally_carrier": (10,),
    "team_steps_with_an_agent_attacking_near_the_ally_carrier": (2,),
    "kill_contributions_attacking_near_the_ally_carrier": (10,),
    "ally_carrier_captures_near": (10,),
    "ally_carrier_deaths_near": (10,),
    "deaths_near_the_ally_carrier": (10,),
    "distance_to_the_ally_carrier_observations": (10,),
    "pair_distance_to_the_ally_carrier_observations": (10, 10),
    "closest_ally_observations_carrying": (10,),
    "closest_ally_observations": (10,),
    # Memory: the current carry's ticks so far, by its carrier.
    "current_carry_steps": (10,),
}
_AMOUNT_SHAPES = {
    "carry_distance": (10,),
    "distance_to_the_ally_flag_home_sum_carrying": (10,),
    "distance_to_the_ally_carrier_sum": (10,),
    "pair_distance_to_the_ally_carrier_sum": (10, 10),
    "closest_ally_sum_carrying": (10,),
    "closest_ally_sum": (10,),
    "damage_attacking_near_the_ally_carrier": (10,),
    # Priest healing, by healer and receiver, on ticks the receiver carried.
    "effective_healing_to_carrying": (10, 10),
    "excess_healing_to_carrying": (10, 10),
    # Priest healing to teammates other than itself on ticks its team carried.
    "effective_healing_during_ally_carries": (10,),
    "effective_priest_healing_received_carrying": (10,),
}
_FLAG_SHAPES = {
    # Memory: an open carry, by its carrier, from pickup to capture or drop.
    "carry_open": (10,),
    # Memory: [flag, slot] the slot carried this flag and dropped it by dying
    # since the flag last left its home.
    "dropped_since_home": (2, 10),
    # Memory: [Priest, ally] the Priest cast Freedom on the ally this tick; it
    # acts on the next tick (the Priest rule, rule 10).
    "freedom_pairs": (10, 10),
}


def initialize_capture_the_flag() -> CaptureTheFlagTotals:
    """Create one Capture the Flag game's zero totals and empty memory.

    Returns
    -------
    CaptureTheFlagTotals
        A dict with fixed keys: int32 counts, float32 amounts and bool memory,
        each with the shape declared beside its name. About 2 KB per game.

    Notes
    -----
    A game that starts with a flag already carried (a scenario) counts that
    carry from the first tick, as a carry of unknown start: its ticks count, and
    so does how it ends, but it is not a steal or a pickup, so its duration and
    the death windows leave it out.
    """
    totals = {
        name: jnp.zeros(shape, jnp.int32) for name, shape in _COUNT_SHAPES.items()
    }
    totals.update(
        {name: jnp.zeros(shape, jnp.float32) for name, shape in _AMOUNT_SHAPES.items()}
    )
    totals.update(
        {name: jnp.zeros(shape, bool) for name, shape in _FLAG_SHAPES.items()}
    )
    return totals


def update_capture_the_flag(
    totals: CaptureTheFlagTotals,
    config: EnvConfig,
    state: EnvState,
    info: Info,
    quantities: CombatQuantities,
    freedom_protected: Array,
) -> CaptureTheFlagTotals:
    """Add one tick of a Capture the Flag game to its totals.

    Parameters
    ----------
    totals : CaptureTheFlagTotals
        The totals from initialize_capture_the_flag or the previous tick.
    config : EnvConfig
        The game's config.
    state : EnvState
        Core's state at the start of this tick.
    info : Info
        Core's facts for this tick. A tick with has_transition False leaves the
        totals unchanged.
    quantities : CombatQuantities
        This tick's combat amounts by source and recipient, the ones full
        metrics build (combat_quantities), so they are computed once.
    freedom_protected : Array
        bool (10,): Freedom on the agent beat a slow this tick, as full metrics
        count it. With last tick's Freedom casts it says which Priest freed whom.

    Returns
    -------
    CaptureTheFlagTotals
        New totals with the same keys, shapes and dtypes.

    Notes
    -----
    Situations come from the start of the tick; pickups, drops, captures,
    returns and deaths from Core's facts for the tick. A pickup is a steal when
    the enemy flag was at home at the start of the tick (countdown 0, no
    carrier), and a pickup of the dropped flag otherwise. A carry ends with a
    capture or a drop by its carrier; drops happen only when a carrier dies.

    Attacking Enemies Near the Ally Carrier: an agent other than the carrier
    whose Basic or Ultimate damaged an enemy that was eligible, within
    NEAR_DISTANCE of the ally carrier and in clear sight of it. A Priest joins
    the situation that tick when it effectively healed, or freed (last tick's
    Freedom beat a slow this tick), a teammate who is in it by a hit; one
    level only.
    """
    facts = info.transition_facts
    flags = facts.capture_the_flag_facts
    tick = capture_the_flag_tick(config, state)
    picked = flags.flag_was_picked_up_by_agent
    dropped = flags.flag_was_dropped_by_agent
    captured = flags.enemy_flag_was_captured_by_agent
    died = facts.death_facts.is_newly_dead_by_recipient
    enemy_flag = 1 - _TEAM
    countdowns = state.capture_the_flag_dropped_flag_countdowns
    carriers = state.capture_the_flag_flag_carrier_slots
    enemy_flag_home = (countdowns[enemy_flag] == 0) & (carriers[enemy_flag] < 0)
    own_flag_home = (countdowns[_TEAM] == 0) & (carriers[_TEAM] < 0)
    carrying = tick.carrying
    allies_near = jax.nn.one_hot(tick.allies_near, MAX_ALLIES_NEAR + 1, dtype=jnp.int32)

    # The carry: its ticks count while held at the start of the tick, and it
    # ends on its carrier's capture or drop this tick, this tick included.
    carry_steps = totals["current_carry_steps"] + _count(carrying)
    ended = captured | dropped
    open_carry = totals["carry_open"] | picked
    closed = ended & open_carry
    # A carry held since a scenario's start has no known pickup: it counts in how
    # carries ended, but not in duration or the death windows.
    finished = ended & (open_carry | carrying)
    homes = config.capture_the_flag_flag_home_positions
    to_home = _length(state.agent_positions - homes[_TEAM])
    physical = facts.physical_facts
    moved = _length(physical.charge_phase_displacement_by_agent) + _length(
        physical.ordinary_movement_phase_displacement_by_agent
    )
    running = carrying & ~died

    # Relays: a capture of flag f after a teammate carried f and dropped it,
    # with no trip home in between. Flags go home by a capture, a touch return
    # or a timed return.
    memory = totals["dropped_since_home"]
    flag_of_capture = jnp.zeros((2, 10), bool).at[enemy_flag, _SLOTS].set(captured)
    capture_of_flag = flag_of_capture.any(axis=1)
    relay_by_agent = (memory & capture_of_flag[:, None] & ~flag_of_capture).any(axis=0)
    relay_capture = captured & (memory[enemy_flag] & _SAME_TEAM).any(axis=1)
    returned_home = (
        capture_of_flag
        | jnp.zeros(2, bool).at[_TEAM].max(flags.ally_flag_was_returned_by_agent)
        | flags.flag_was_automatically_returned_by_team
    )
    memory = (memory & ~returned_home[:, None]) | jnp.zeros((2, 10), bool).at[
        enemy_flag, _SLOTS
    ].set(dropped)

    # Attacking Enemies Near the Ally Carrier, by a hit, then the Priest rule.
    enemy = ~_SAME_TEAM & (_SLOTS[:, None] != _SLOTS[None, :])
    enemy_near_carrier = (
        enemy
        & (tick.team_has_carrier & ~carrying)[:, None]
        & tick.near[tick.ally_carrier]
    )
    damage = quantities.damage
    effective = quantities.healing - quantities.excess_healing
    hitting = ((damage > 0) & enemy_near_carrier).any(axis=1)
    combat = facts.combat_transition_facts
    recipient = combat.combat_effect_recipient_global_slot_by_source
    casts = combat.priest_blessing_of_freedom_is_applied_by_source[:, None] & (
        recipient[:, None] == _SLOTS[None, :]
    )
    freed = totals["freedom_pairs"] & freedom_protected[None, :]
    helped = ((effective > 0) | freed) & _SAME_TEAM & hitting[None, :]
    attacking = hitting | helped.any(axis=1)
    team_attacking = jnp.zeros(2, jnp.int32).at[_TEAM].add(_count(attacking))
    contributions = quantities.kill_contributions

    team_carrying = tick.team_has_carrier
    healing_to_carrier = (effective * carrying[None, :]) * _SAME_TEAM
    excess_to_carrier = (quantities.excess_healing * carrying[None, :]) * _SAME_TEAM
    within = carry_steps[:, None] <= jnp.asarray(DEATH_WINDOWS)[None, :]

    near = tick.near_the_ally_carrier
    carrier = tick.ally_carrier
    team_near = jnp.zeros(2, bool).at[_TEAM].max(near)
    team_near_count = jnp.zeros(2, jnp.int32).at[_TEAM].add(_count(near))
    measured = tick.eligible & tick.team_has_carrier & ~carrying
    pair_measured = tick.eligible[:, None] & _SAME_TEAM & carrying[None, :]
    closest = tick.eligible & tick.has_eligible_ally

    new = dict(totals)
    new.update(
        flag_steals=totals["flag_steals"] + _count(picked & enemy_flag_home),
        dropped_enemy_flag_pickups=totals["dropped_enemy_flag_pickups"]
        + _count(picked & ~enemy_flag_home),
        flag_captures=totals["flag_captures"] + _count(captured),
        steps_carrying=totals["steps_carrying"] + _count(carrying),
        deaths_carrying=totals["deaths_carrying"] + _count(carrying & died),
        carry_duration_steps=totals["carry_duration_steps"]
        + jnp.where(closed, carry_steps, 0),
        carry_duration_observations=totals["carry_duration_observations"]
        + _count(closed),
        carry_steps_ending_in_a_capture=totals["carry_steps_ending_in_a_capture"]
        + jnp.where(finished & captured, carry_steps, 0),
        carry_steps_ending_in_death=totals["carry_steps_ending_in_death"]
        + jnp.where(finished & dropped & ~captured, carry_steps, 0),
        current_carry_steps=jnp.where(ended, 0, carry_steps),
        carry_open=open_carry & ~ended,
        carry_distance=totals["carry_distance"] + jnp.where(running, moved, 0.0),
        carry_speed_observations=totals["carry_speed_observations"] + _count(running),
        carry_steps_in_capture_range_with_the_ally_flag_away=totals[
            "carry_steps_in_capture_range_with_the_ally_flag_away"
        ]
        + _count(
            carrying
            & ~own_flag_home
            & (to_home <= config.capture_the_flag_flag_capture_distance)
        ),
        distance_to_the_ally_flag_home_sum_carrying=totals[
            "distance_to_the_ally_flag_home_sum_carrying"
        ]
        + jnp.where(carrying, to_home, 0.0),
        captures_by_an_ally_after_it_dropped=totals[
            "captures_by_an_ally_after_it_dropped"
        ]
        + _count(relay_by_agent),
        captures_after_an_ally_dropped=totals["captures_after_an_ally_dropped"]
        + _count(relay_capture),
        dropped_since_home=memory,
        carries_ending_in_death_within=totals["carries_ending_in_death_within"]
        + _count((closed & dropped & ~captured)[:, None] & within),
        steps_attacking_near_the_ally_carrier=totals[
            "steps_attacking_near_the_ally_carrier"
        ]
        + _count(attacking),
        solo_steps_attacking_near_the_ally_carrier=totals[
            "solo_steps_attacking_near_the_ally_carrier"
        ]
        + _count(attacking & (team_attacking[_TEAM] == 1)),
        team_steps_with_an_agent_attacking_near_the_ally_carrier=totals[
            "team_steps_with_an_agent_attacking_near_the_ally_carrier"
        ]
        + _count(team_attacking > 0),
        damage_attacking_near_the_ally_carrier=totals[
            "damage_attacking_near_the_ally_carrier"
        ]
        + jnp.where(enemy_near_carrier, damage, 0.0).sum(axis=1),
        kill_contributions_attacking_near_the_ally_carrier=totals[
            "kill_contributions_attacking_near_the_ally_carrier"
        ]
        + (contributions & enemy_near_carrier).sum(axis=1, dtype=jnp.int32),
        effective_healing_to_carrying=totals["effective_healing_to_carrying"]
        + healing_to_carrier,
        excess_healing_to_carrying=totals["excess_healing_to_carrying"]
        + excess_to_carrier,
        effective_healing_during_ally_carries=totals[
            "effective_healing_during_ally_carries"
        ]
        + jnp.where(team_carrying, (effective * _SAME_TEAM).sum(axis=1), 0.0),
        effective_priest_healing_received_carrying=totals[
            "effective_priest_healing_received_carrying"
        ]
        + jnp.where(carrying, effective.sum(axis=0), 0.0),
        freedom_pairs=casts,
        steps_with_allies_near_carrying=totals["steps_with_allies_near_carrying"]
        + allies_near * _count(carrying)[:, None],
        flag_captures_with_allies_near=totals["flag_captures_with_allies_near"]
        + allies_near * _count(captured)[:, None],
        deaths_with_allies_near_carrying=totals["deaths_with_allies_near_carrying"]
        + allies_near * _count(carrying & died)[:, None],
        steps_near_the_ally_carrier=totals["steps_near_the_ally_carrier"]
        + _count(near),
        solo_steps_near_the_ally_carrier=totals["solo_steps_near_the_ally_carrier"]
        + _count(near & (team_near_count[_TEAM] == 1)),
        team_steps_with_an_agent_near_the_ally_carrier=totals[
            "team_steps_with_an_agent_near_the_ally_carrier"
        ]
        + _count(team_near),
        ally_carrier_captures_near=totals["ally_carrier_captures_near"]
        + _count(near & captured[carrier]),
        ally_carrier_deaths_near=totals["ally_carrier_deaths_near"]
        + _count(near & died[carrier]),
        deaths_near_the_ally_carrier=totals["deaths_near_the_ally_carrier"]
        + _count(near & died),
        distance_to_the_ally_carrier_sum=totals["distance_to_the_ally_carrier_sum"]
        + jnp.where(measured, tick.distances[_SLOTS, carrier], 0.0),
        distance_to_the_ally_carrier_observations=totals[
            "distance_to_the_ally_carrier_observations"
        ]
        + _count(measured),
        pair_distance_to_the_ally_carrier_sum=totals[
            "pair_distance_to_the_ally_carrier_sum"
        ]
        + jnp.where(pair_measured, tick.distances, 0.0),
        pair_distance_to_the_ally_carrier_observations=totals[
            "pair_distance_to_the_ally_carrier_observations"
        ]
        + _count(pair_measured),
        closest_ally_sum_carrying=totals["closest_ally_sum_carrying"]
        + jnp.where(closest & carrying, tick.closest_ally_distance, 0.0),
        closest_ally_observations_carrying=totals["closest_ally_observations_carrying"]
        + _count(closest & carrying),
        closest_ally_sum=totals["closest_ally_sum"]
        + jnp.where(closest, tick.closest_ally_distance, 0.0),
        closest_ally_observations=totals["closest_ally_observations"] + _count(closest),
    )
    return {
        name: jnp.where(facts.has_transition, new[name], total)
        for name, total in totals.items()
    }


def _team(values: Array) -> Array:
    """Add a per-slot array's leading axis up by team: (10, ...) to (2, ...)."""
    return values.reshape(2, 5, *values.shape[1:]).sum(axis=1)


def _measured(values: Array) -> MetricValues:
    """Pair a measured array with an all-true availability mask."""
    return MetricValues(values.astype(jnp.float32), jnp.ones(values.shape, bool))


def _mean(total: Array, observations: Array) -> MetricValues:
    """Divide a total by its count; a count of 0 makes the mean blank (rule 7)."""
    valid = observations > 0
    return MetricValues(
        (total / jnp.where(valid, observations, 1)).astype(jnp.float32), valid
    )


type CaptureTheFlagData = dict[tuple[MetricScope, str], MetricValues]


def capture_the_flag_data(totals: CaptureTheFlagTotals) -> CaptureTheFlagData:
    """Read a game's Capture the Flag totals as values keyed by scope and stem.

    Parameters
    ----------
    totals : CaptureTheFlagTotals
        The totals after the game's last recorded tick.

    Returns
    -------
    CaptureTheFlagData
        A dict from (scope, stem) to MetricValues: agent arrays (10,) in global
        slot order, team arrays (2,) in Team A, Team B order, and
        source_recipient arrays (10, 10) from agent i to teammate j, blank where
        i is j. Each escort size has its own stem (escort_split). Means and
        shares whose count is 0 are blank. The packer adds the roster's blanks
        (empty slots).

    Notes
    -----
    A carry still open at the end of the game counts in the duration mean with
    its ticks so far, and in neither "ending in" total. A count that only copies
    other columns (a view) is never returned (rule 31): the carry duration and
    carry speed counts, the distance-to-home count, and the solo ticks and team
    sums of Carrying, where a team has at most one carrier.
    """
    data: CaptureTheFlagData = {}

    def both(stem: str, agents: Array, teams: Array | None = None) -> None:
        """Register an agent array and its team total (summed unless given)."""
        data[("agent", stem)] = _measured(agents)
        data[("team", stem)] = _measured(_team(agents) if teams is None else teams)

    def both_means(stem: str, total: Array, observations: Array) -> None:
        """Register an agent mean and its team mean, without their counts."""
        data[("agent", f"{stem}_mean")] = _mean(total, observations)
        data[("team", f"{stem}_mean")] = _mean(_team(total), _team(observations))

    def situated(stem: str, situation: str) -> str:
        """Put a situation's slug last, after two underscores (rule 18)."""
        return f"{stem}__{situation}"

    steals = totals["flag_steals"]
    captures = totals["flag_captures"]
    open_carry = totals["carry_open"]
    both("flag_steals", steals)
    both("dropped_enemy_flag_pickups", totals["dropped_enemy_flag_pickups"])
    data[("team", "flag_pickups")] = _measured(
        _team(steals + totals["dropped_enemy_flag_pickups"])
    )
    # Carrying's tick block (final/01, table 5): a team has at most one carrier,
    # so its solo ticks and team sums are views, two or more is always blank, and
    # the headcount and solo share are 1 whenever there was a carrier.
    carrying = totals["steps_carrying"]
    team_carrying = _team(carrying)
    data[("agent", situated("steps", "carrying"))] = _measured(carrying)
    data[("team", situated("steps_with_an_agent", "carrying"))] = _measured(
        team_carrying
    )
    data[("team", situated("steps_with_two_or_more_agents", "carrying"))] = (
        MetricValues(jnp.zeros(2, jnp.float32), jnp.zeros(2, bool))
    )
    data[("team", situated("headcount_mean", "carrying"))] = _mean(
        team_carrying, team_carrying
    )
    data[("agent", situated("step_participation", "carrying"))] = _mean(
        carrying, jnp.repeat(team_carrying, 5)
    )
    data[("agent", situated("solo_step_fraction", "carrying"))] = _mean(
        carrying, carrying
    )
    both("flag_captures", captures)
    both(situated("deaths", "carrying"), totals["deaths_carrying"])
    both_means(
        "carry_duration",
        totals["carry_duration_steps"]
        + jnp.where(open_carry, totals["current_carry_steps"], 0),
        totals["carry_duration_observations"] + _count(open_carry),
    )
    data[("team", "flag_captures_per_steal")] = _mean(_team(captures), _team(steals))
    both("carry_steps_ending_in_a_capture", totals["carry_steps_ending_in_a_capture"])
    both("carry_steps_ending_in_death", totals["carry_steps_ending_in_death"])
    both("carry_distance", totals["carry_distance"])
    both_means(
        "carry_speed", totals["carry_distance"], totals["carry_speed_observations"]
    )
    both(
        "carry_steps_in_capture_range_with_the_ally_flag_away",
        totals["carry_steps_in_capture_range_with_the_ally_flag_away"],
    )
    closest_carrying = totals["closest_ally_observations_carrying"]
    for name, total, observations in (
        (
            "distance_to_the_ally_flag_home",
            totals["distance_to_the_ally_flag_home_sum_carrying"],
            carrying,
        ),
        (
            "distance_to_the_closest_ally",
            totals["closest_ally_sum_carrying"],
            closest_carrying,
        ),
    ):
        data[("agent", situated(f"{name}_mean", "carrying"))] = _mean(
            total, observations
        )
        data[("team", situated(f"{name}_mean", "carrying"))] = _mean(
            _team(total), _team(observations)
        )
    both(
        situated("distance_to_the_closest_ally_observations", "carrying"),
        closest_carrying,
    )
    data[("agent", "flag_captures_by_an_ally_after_it_dropped_the_flag")] = _measured(
        totals["captures_by_an_ally_after_it_dropped"]
    )
    both(
        "flag_captures_after_an_ally_dropped_the_flag",
        totals["captures_after_an_ally_dropped"],
    )

    both_means(
        "distance_to_the_ally_carrier",
        totals["distance_to_the_ally_carrier_sum"],
        totals["distance_to_the_ally_carrier_observations"],
    )
    both(
        "distance_to_the_ally_carrier_observations",
        totals["distance_to_the_ally_carrier_observations"],
    )
    # Pairs run from agent i to teammate j; an agent paired with itself is blank.
    pair_observations = totals["pair_distance_to_the_ally_carrier_observations"]
    others = ~jnp.eye(10, dtype=bool)
    mean = _mean(totals["pair_distance_to_the_ally_carrier_sum"], pair_observations)
    data[("source_recipient", "distance_to_the_ally_carrier_mean")] = MetricValues(
        mean.values, mean.valid & others
    )
    data[("source_recipient", "distance_to_the_ally_carrier_observations")] = (
        MetricValues(pair_observations.astype(jnp.float32), others)
    )
    both_means(
        "distance_to_the_closest_ally",
        totals["closest_ally_sum"],
        totals["closest_ally_observations"],
    )
    both(
        "distance_to_the_closest_ally_observations", totals["closest_ally_observations"]
    )
    near = "near_the_ally_carrier"
    data[("agent", situated("ally_carrier_captures", near))] = _measured(
        totals["ally_carrier_captures_near"]
    )
    data[("agent", situated("ally_carrier_deaths", near))] = _measured(
        totals["ally_carrier_deaths_near"]
    )
    data[("agent", situated("deaths", near))] = _measured(
        totals["deaths_near_the_ally_carrier"]
    )
    for stem, split in (
        ("steps", totals["steps_with_allies_near_carrying"]),
        ("flag_captures", totals["flag_captures_with_allies_near"]),
        ("deaths", totals["deaths_with_allies_near_carrying"]),
    ):
        for size in range(MAX_ALLIES_NEAR + 1):
            both(situated(f"{stem}_{escort_split(size)}", "carrying"), split[:, size])

    # The tick block (final/01, table 5) for each situation built so far.
    for situation, prefix in (
        (near, "steps_near_the_ally_carrier"),
        (
            "attacking_enemies_near_the_ally_carrier",
            "steps_attacking_near_the_ally_carrier",
        ),
    ):
        steps = totals[prefix]
        solo = totals[f"solo_{prefix}"]
        with_an_agent = totals[
            f"team_steps_with_an_agent_{prefix.removeprefix('steps_')}"
        ]
        data[("agent", situated("steps", situation))] = _measured(steps)
        data[("agent", situated("solo_steps", situation))] = _measured(solo)
        data[("team", situated("steps_with_an_agent", situation))] = _measured(
            with_an_agent
        )
        data[("team", situated("agent_steps", situation))] = _measured(_team(steps))
        data[("team", situated("solo_steps", situation))] = _measured(_team(solo))
        data[("team", situated("steps_with_two_or_more_agents", situation))] = (
            _measured(with_an_agent - _team(solo))
        )
        data[("team", situated("headcount_mean", situation))] = _mean(
            _team(steps), with_an_agent
        )
        data[("agent", situated("step_participation", situation))] = _mean(
            steps, jnp.repeat(with_an_agent, 5)
        )
        data[("agent", situated("solo_step_fraction", situation))] = _mean(solo, steps)
    attacking = "attacking_enemies_near_the_ally_carrier"
    data[("agent", situated("damage_done", attacking))] = _measured(
        totals["damage_attacking_near_the_ally_carrier"]
    )
    data[("agent", situated("kill_contributions", attacking))] = _measured(
        totals["kill_contributions_attacking_near_the_ally_carrier"]
    )

    # Healing the ally carrier (Priest-only columns; the packer blanks the rest).
    to_carrier = totals["effective_healing_to_carrying"]
    excess_to_carrier = totals["excess_healing_to_carrying"]
    healing_stem = situated("effective_healing_done_to_allies_who_are", "carrying")
    excess_stem = situated("excess_healing_to_allies_who_are", "carrying")
    both(healing_stem, to_carrier.sum(axis=1))
    both(excess_stem, excess_to_carrier.sum(axis=1))
    data[("source_recipient", healing_stem)] = MetricValues(to_carrier, others)
    data[("source_recipient", excess_stem)] = MetricValues(excess_to_carrier, others)
    during = totals["effective_healing_during_ally_carries"]
    both("effective_healing_done_to_allies_during_ally_carries", during)
    data[("agent", "ally_carrier_healing_focus_fraction")] = _mean(
        to_carrier.sum(axis=1), during
    )
    data[("team", "ally_carrier_healing_focus_fraction")] = _mean(
        _team(to_carrier.sum(axis=1)), _team(during)
    )
    data[("agent", situated("effective_priest_healing_received", "carrying"))] = (
        _measured(totals["effective_priest_healing_received_carrying"])
    )
    windows = totals["carries_ending_in_death_within"]
    for index, ticks in enumerate(DEATH_WINDOWS):
        both(f"carries_ending_in_death_within_{ticks}_steps", windows[:, index])
    return data
