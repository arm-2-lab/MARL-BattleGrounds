"""Apply the objective modes' rules for one tick, in pure JAX.

update_capture_the_flag_flags is the one owner of how the two flags move
between home, carried and dropped, and of the capture score: carry, return,
pickup, capture, the death drop and the dropped-flag timer, in that order.
Core's step calls it once per tick inside the Capture the Flag mode branch,
after deaths and before the respawn wave. Every other mode keeps its flags and
scores through env's _not_capture_the_flag.

update_king_of_the_hills_hills is the one owner of the hills: who stands on
each, contest, ownership with the empty-hill timer, points and scores, in that
order. Core's step calls it once per tick inside the King of the Hills mode
branch, at the same point as the flag update, and every other mode keeps its
hills and scores through env's _not_king_of_the_hills. It reads the same
end-of-movement positions and start-of-tick state as the flag update. Both
updates count only eligible agents, through one shared rule, and
build_agent_hill_interaction_matrix is the exported, geometry-only test of who
stands on which hill.

Who wins is decided in env's outcome branch from the successor state, not
here. No host work, randomness or I/O; everything runs under jit, vmap and
scan.
"""

from typing import NamedTuple, cast

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MAX_HILL_SLOTS,
    NO_TEAM_ID,
    TASK_MODE_OUTCOME_ONGOING,
    TEAM_A_ID,
    TEAM_B_ID,
    CaptureTheFlagTransitionFacts,
    EnvConfig,
    EnvState,
    KingOfTheHillsTransitionFacts,
)


class CaptureTheFlagFlagUpdate(NamedTuple):
    """Hold the flag state and scores after one tick, and that tick's flag facts.

    Flag arrays use flag order [Team A's flag, Team B's flag], which is also
    team order. Step copies the three flag arrays and the scores into the
    successor EnvState and the facts into TransitionFacts. This immutable record
    computes nothing.
    """

    next_capture_the_flag_flag_positions_by_flag: Array
    """Float32 (2, 2): each flag's [x, y] in map units after this tick."""
    next_capture_the_flag_flag_carrier_slots_by_flag: Array
    """Int32 (2,): each flag's carrier as a global slot (0-9), or -1 for none."""
    next_capture_the_flag_dropped_flag_countdowns_by_flag: Array
    """Int32 (2,): ticks before each dropped flag goes home; 0 when not dropped."""
    next_capture_the_flag_scores_by_team: Array
    """Int32 (2,): each team's captures so far, after this tick, [Team A, Team B]."""
    capture_the_flag_facts: CaptureTheFlagTransitionFacts
    """Who picked up, dropped, returned and captured each flag; timer returns."""


class KingOfTheHillsHillUpdate(NamedTuple):
    """Hold the hill state and scores after one tick, and that tick's hill facts.

    Hill arrays use hill-slot order (slot 0 the left-half hill, 1 the midline
    hill, 2 the right-half hill); team arrays use team order [Team A, Team B].
    Step copies the four next fields into the successor EnvState and the facts
    into TransitionFacts. This immutable record computes nothing.
    """

    next_king_of_the_hills_hill_owners_by_hill: Array
    """Int32 (3,): each hill's owner after this tick: 0 nobody, 1 Team A, 2 Team B."""
    next_king_of_the_hills_empty_hill_ownership_countdowns_by_hill: Array
    """Int32 (3,): empty ticks each hill still stays its owner's; 0 when neutral."""
    next_king_of_the_hills_hill_is_contested_by_hill: Array
    """Bool (3,): True where eligible agents of both teams stand on the hill."""
    next_king_of_the_hills_scores_by_team: Array
    """Int32 (2,): each team's score after this tick's points, [Team A, Team B]."""
    king_of_the_hills_facts: KingOfTheHillsTransitionFacts
    """Who stood on each hill, the owners and this tick's points; outcome ongoing."""


def _build_objective_eligibility_by_agent(state: EnvState, config: EnvConfig) -> Array:
    """Return which agents may act on an objective this tick.

    Every objective rule (flags and hills) counts only eligible agents, and
    each mode's update calls this one function, so the rule has one owner.

    Parameters
    ----------
    state : EnvState
        The state at the start of the tick. Reads alive_mask and
        spawn_shield_durations.
    config : EnvConfig
        The game's validated settings. Reads agent_profile.active_mask.

    Returns
    -------
    Array
        Bool (10,), one entry per agent slot: True when the agent is active,
        alive and unshielded (spawn shield 0) at the start of the tick.
        Unused slots are always False.

    Notes
    -----
    The rule reads the start of the tick, so an agent killed this tick still
    counts this tick. Pure JAX, with no host work; it runs under jit, vmap
    and scan.
    """
    is_alive_and_active_by_agent = jnp.logical_and(
        state.alive_mask, config.agent_profile.active_mask
    )
    is_eligible_to_interact_with_objective_by_agent = jnp.logical_and(
        is_alive_and_active_by_agent, state.spawn_shield_durations == 0
    )

    return is_eligible_to_interact_with_objective_by_agent


def build_agent_hill_interaction_matrix(
    agent_positions: Array, hill_positions: Array, hill_radii: Array
) -> Array:
    """Return which agents stand on which hill, from geometry alone.

    An agent is on a hill when the hill slot is used (radius above 0) and the
    agent's center is within the radius; touching the edge counts. Nothing
    else is checked: no eligibility, line of sight or team. Callers apply
    eligibility and choose which moment's positions to pass (the end of
    movement inside step; the start of the tick for the planned hill
    metrics).

    Parameters
    ----------
    agent_positions : Array
        Float32 (10, 2): each agent's [x, y] in map units.
    hill_positions : Array
        Float32 (3, 2): each hill slot's center [x, y] in map units
        (config.king_of_the_hills_hill_positions). An unused slot is [0, 0].
    hill_radii : Array
        Float32 (3,): each hill slot's radius in map units
        (config.king_of_the_hills_hill_radii); 0 marks an unused slot.

    Returns
    -------
    Array
        Bool (10, 3), agent by hill: True when that agent's center is within
        that used hill's radius. An unused hill's column is all False.

    Notes
    -----
    Distances are Core's own float32 results, so a point exactly on the edge
    on paper can round one float32 step either way, as in Capture the Flag's
    distance tests. Pure JAX, with no host work; it runs under jit, vmap and
    scan.
    """
    distance_by_agent_by_hill = cast(
        Array,
        jnp.linalg.norm(
            agent_positions[:, None, :] - hill_positions[None, :, :],  # (10,3,2)
            axis=-1,
        ),
    )

    return (distance_by_agent_by_hill <= hill_radii) & (hill_radii != 0)


def update_capture_the_flag_flags(
    current_state: EnvState,
    config: EnvConfig,
    next_agent_positions: Array,
    is_newly_dead_by_recipient: Array,
) -> CaptureTheFlagFlagUpdate:
    """Move both flags through one tick's flag rules and record what happened.

    The rules run in a fixed order, each reading the result of the one before:
    carry, return, pickup, capture, drop, then the dropped-flag timer. A flag
    is carried when its carrier slot is 0 or more, dropped when its countdown
    is above 0, and home otherwise.

    Parameters
    ----------
    current_state : EnvState
        The state at the start of the tick. Reads the three flag fields,
        capture_the_flag_scores and, for eligibility, alive_mask and
        spawn_shield_durations.
    config : EnvConfig
        The game's validated settings. Reads agent_profile.active_mask,
        agent_profile.team_ids, capture_the_flag_flag_home_positions,
        capture_the_flag_flag_interaction_distance,
        capture_the_flag_flag_capture_distance and
        capture_the_flag_dropped_flag_auto_return_steps.
    next_agent_positions : Array
        Float32 (10, 2): every agent's [x, y] at the end of this tick's
        movement, before the respawn wave moves anyone.
    is_newly_dead_by_recipient : Array
        Bool (10,): True for each agent killed this tick (death facts).

    Returns
    -------
    CaptureTheFlagFlagUpdate
        The flag state and scores for the successor, and this tick's flag
        facts.

    Notes
    -----
    - Eligible: active, alive and unshielded at the start of the tick. An
      agent killed this tick still acts this tick.
    - Touching: an eligible agent whose end-of-movement center is within the
      interaction distance of the flag (inclusive). Distances, and so exact
      ties, are Core's own float32 results, so a distance exactly at the reach
      on paper can round one float32 step either way.
    - Carry: a carried flag moves to its carrier's position.
    - Return and pickup: on a dropped flag, the nearest eligible toucher of
      either team decides. An owner sends the flag home; an enemy picks it up;
      an exact owner-enemy tie goes to the pickup. A home flag can only be
      picked up. A tie within one team goes to the lowest slot. Each event
      credits exactly one agent.
    - Capture: a carried flag is captured when its carrier (after the pickup)
      is within the capture distance (inclusive) of its own team's home, while
      its own team's flag is home after the return and the pickup. A flag sent
      home by the return this tick counts as home; one picked up by an enemy
      this tick does not. A dropped enemy flag picked up near home can be
      captured on the same tick. The captured flag goes home with no carrier,
      and the carrier's team scores 1. At most one flag is captured per tick,
      because a capture needs the capturer's own flag home and so uncarried.
    - Drop: a carrier killed this tick, including one that picked the flag up
      this tick, drops it where it stands with a full countdown. A carrier
      that captured this tick has no flag left, so it drops nothing (decision
      160).
    - Timer: every other dropped flag counts down by 1 and goes home at 0. A
      flag the timer sends home counts as home only from the next tick.
    - The facts' outcome is left ongoing (0); step sets it from its outcome
      branch.

    Pure JAX for one game, with no host work, randomness or input mutation;
    use it under jit, scan and an outer vmap. Step calls it only in Capture
    the Flag games.
    """
    # Carry: a carried flag moves with its carrier; the others stay put.
    flag_carrier_slot_by_flag = current_state.capture_the_flag_flag_carrier_slots

    flag_carrier_next_agent_position_by_flag = next_agent_positions[
        flag_carrier_slot_by_flag
    ]

    current_flag_position_by_flag = current_state.capture_the_flag_flag_positions

    flag_is_carried_by_flag = ~(flag_carrier_slot_by_flag == -1)

    next_flag_position_by_flag = jnp.where(
        flag_is_carried_by_flag[:, None],
        flag_carrier_next_agent_position_by_flag,
        current_flag_position_by_flag,
    )

    # Eligible: active, alive and unshielded at the start of the tick.
    is_eligible_to_interact_with_a_flag_by_agent = (
        _build_objective_eligibility_by_agent(current_state, config)
    )

    # Touching: eligible, and within reach of the flag at the end of movement.
    distance_by_flag_by_agent = cast(
        Array,
        jnp.linalg.norm(
            next_agent_positions[None, :, :]  # (1, 10, 2)
            - next_flag_position_by_flag[:, None, :],  # (2, 1, 2)
            axis=-1,
        ),
    )

    is_within_interaction_radius_by_flag_by_agent = (
        distance_by_flag_by_agent <= config.capture_the_flag_flag_interaction_distance
    )

    is_eligible_and_touching_by_flag_by_agent = jnp.logical_and(
        is_within_interaction_radius_by_flag_by_agent,
        is_eligible_to_interact_with_a_flag_by_agent[None, :],
    )

    # Split the touchers by side: the flag's owners (allies) and its enemies.
    ally_team_id_by_flag = jnp.array((TEAM_A_ID, TEAM_B_ID))[:, None]
    enemy_team_id_by_flag = jnp.array((TEAM_B_ID, TEAM_A_ID))[:, None]
    team_id_by_agent = config.agent_profile.team_ids[None, :]

    is_ally_by_flag_by_agent = ally_team_id_by_flag == team_id_by_agent
    is_enemy_by_flag_by_agent = enemy_team_id_by_flag == team_id_by_agent

    is_eligible_ally_touching_by_flag_by_agent = jnp.logical_and(
        is_eligible_and_touching_by_flag_by_agent, is_ally_by_flag_by_agent
    )

    is_eligible_enemy_touching_by_flag_by_agent = jnp.logical_and(
        is_eligible_and_touching_by_flag_by_agent, is_enemy_by_flag_by_agent
    )

    # An agent that does not touch gets an infinite distance, so it is never nearest.
    eligible_and_touching_ally_distance_by_flag_by_agent = jnp.where(
        is_eligible_ally_touching_by_flag_by_agent, distance_by_flag_by_agent, jnp.inf
    )

    eligible_and_touching_enemy_distance_by_flag_by_agent = jnp.where(
        is_eligible_enemy_touching_by_flag_by_agent, distance_by_flag_by_agent, jnp.inf
    )

    # The nearest touching ally and enemy of each flag, ties to the lowest slot.
    # argmin returns 0 over a row of infinities, so any() turns "nobody" into -1.
    nearest_eligible_and_touching_ally_distance_by_flag = jnp.min(
        eligible_and_touching_ally_distance_by_flag_by_agent, axis=1
    )
    nearest_eligible_and_touching_enemy_distance_by_flag = jnp.min(
        eligible_and_touching_enemy_distance_by_flag_by_agent, axis=1
    )

    nearest_eligible_and_touching_ally_slot_by_flag = jnp.where(
        jnp.any(is_eligible_ally_touching_by_flag_by_agent, axis=1),
        jnp.argmin(eligible_and_touching_ally_distance_by_flag_by_agent, axis=1),
        -1,
    )

    nearest_eligible_and_touching_enemy_slot_by_flag = jnp.where(
        jnp.any(is_eligible_enemy_touching_by_flag_by_agent, axis=1),
        jnp.argmin(eligible_and_touching_enemy_distance_by_flag_by_agent, axis=1),
        -1,
    )

    # Return: a dropped flag goes home when an owner is strictly the nearest.
    flag_was_dropped_at_tick_start_by_flag = (
        current_state.capture_the_flag_dropped_flag_countdowns > 0
    )

    flag_is_returned_this_tick_by_flag = flag_was_dropped_at_tick_start_by_flag & (
        nearest_eligible_and_touching_ally_distance_by_flag
        < nearest_eligible_and_touching_enemy_distance_by_flag
    )

    # Pickup: an uncarried flag that did not go home goes to its nearest enemy.
    flag_is_picked_up_this_tick_by_flag = (
        ~flag_is_returned_this_tick_by_flag
        & (nearest_eligible_and_touching_enemy_slot_by_flag != -1)
        & ~flag_is_carried_by_flag
    )

    # Positions after the return or the pickup.
    condlist = [
        flag_is_returned_this_tick_by_flag[:, None],
        flag_is_picked_up_this_tick_by_flag[:, None],
    ]
    choicelist = [
        config.capture_the_flag_flag_home_positions,
        next_agent_positions[nearest_eligible_and_touching_enemy_slot_by_flag],
    ]
    next_flag_position_by_flag = jnp.select(
        condlist=condlist, choicelist=choicelist, default=next_flag_position_by_flag
    )

    # Carriers after the pickup, kept under their own name: the capture reads them.
    flag_carrier_slot_after_pickup_by_flag = jnp.where(
        flag_is_picked_up_this_tick_by_flag,
        nearest_eligible_and_touching_enemy_slot_by_flag,
        flag_carrier_slot_by_flag,
    )

    # A pickup or a return clears the countdown; the timer below ages the rest.
    next_dropped_flag_countdown_by_flag = jnp.where(
        flag_is_picked_up_this_tick_by_flag | flag_is_returned_this_tick_by_flag,
        0,
        current_state.capture_the_flag_dropped_flag_countdowns,
    )

    # Home after the return and the pickup: not carried and not dropped.
    flags_are_at_home_by_flag = (flag_carrier_slot_after_pickup_by_flag == -1) & (
        next_dropped_flag_countdown_by_flag == 0
    )

    # Each carrier within capture distance of its own team's home, which is the
    # other flag's home (the reversed rows).
    flag_carrier_is_near_home_base_by_flag = cast(
        Array,
        jnp.linalg.norm(
            next_agent_positions[flag_carrier_slot_after_pickup_by_flag]
            - config.capture_the_flag_flag_home_positions[::-1],
            axis=-1,
        )
        <= config.capture_the_flag_flag_capture_distance,
    )

    # Capture: a carried flag whose carrier is in its own team's zone while its
    # own team's flag (the other row) is home.
    flag_is_capped_this_tick_by_flag = (
        flags_are_at_home_by_flag[::-1]
        & flag_carrier_is_near_home_base_by_flag
        & (flag_carrier_slot_after_pickup_by_flag != -1)
    )

    # A captured flag goes home this tick.
    next_flag_position_by_flag = jnp.where(
        flag_is_capped_this_tick_by_flag[:, None],
        config.capture_the_flag_flag_home_positions,
        next_flag_position_by_flag,
    )

    # Carriers after the capture: a captured flag has none, so its carrier drops
    # nothing below, even if killed this tick (decision 160).
    flag_carrier_slot_after_capture_by_flag = jnp.where(
        flag_is_capped_this_tick_by_flag, -1, flag_carrier_slot_after_pickup_by_flag
    )

    # The capturing team scores 1: a capture of flag f counts for the other team.
    next_capture_the_flag_scores_by_team = jnp.where(
        flag_is_capped_this_tick_by_flag[::-1],
        current_state.capture_the_flag_scores + 1,
        current_state.capture_the_flag_scores,
    )

    # Drop: the carrier after the capture was killed this tick.
    flag_is_dropped_this_tick_by_flag = (
        flag_carrier_slot_after_capture_by_flag != -1
    ) & (is_newly_dead_by_recipient[flag_carrier_slot_after_capture_by_flag])

    # A dropped flag starts a full countdown and loses its carrier.
    next_dropped_flag_countdown_by_flag = jnp.where(
        flag_is_dropped_this_tick_by_flag,
        config.capture_the_flag_dropped_flag_auto_return_steps,
        next_dropped_flag_countdown_by_flag,
    )

    next_flag_carrier_slot_by_flag = jnp.where(
        flag_is_dropped_this_tick_by_flag, -1, flag_carrier_slot_after_capture_by_flag
    )

    # Timer: age every dropped flag except one dropped this tick.
    flag_must_be_aged_by_flag = (
        next_dropped_flag_countdown_by_flag > 0
    ) & ~flag_is_dropped_this_tick_by_flag

    next_dropped_flag_countdown_by_flag = jnp.where(
        flag_must_be_aged_by_flag,
        next_dropped_flag_countdown_by_flag - 1,
        next_dropped_flag_countdown_by_flag,
    )

    # A flag whose countdown reaches 0 goes home on its own.
    flag_is_automatically_returned_this_tick_by_flag = flag_must_be_aged_by_flag & (
        next_dropped_flag_countdown_by_flag == 0
    )

    next_flag_position_by_flag = jnp.where(
        flag_is_automatically_returned_this_tick_by_flag[:, None],
        config.capture_the_flag_flag_home_positions,
        next_flag_position_by_flag,
    )

    # Facts: who did what this tick, one Boolean per agent. Each record writes a
    # per-flag event into that flag's agent slot with .max, not .set: a flag with
    # nobody has slot -1, which wraps to slot 9, and .max never erases a True.
    all_false_by_agent = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.bool_)

    # Who picked up each flag: the nearest eligible enemy touching it.
    flag_is_picked_up_this_tick_by_agent = all_false_by_agent.at[
        nearest_eligible_and_touching_enemy_slot_by_flag
    ].max(flag_is_picked_up_this_tick_by_flag)

    # Who dropped each flag: its carrier after the capture, if killed this tick.
    flag_is_dropped_this_tick_by_agent = all_false_by_agent.at[
        flag_carrier_slot_after_capture_by_flag
    ].max(flag_is_dropped_this_tick_by_flag)

    # Who returned each flag: the nearest eligible owner touching it.
    ally_flag_is_returned_this_tick_by_agent = all_false_by_agent.at[
        nearest_eligible_and_touching_ally_slot_by_flag
    ].max(flag_is_returned_this_tick_by_flag)

    # Who captured each flag: its carrier after the pickup.
    enemy_flag_was_captured_this_tick_by_agent = all_false_by_agent.at[
        flag_carrier_slot_after_pickup_by_flag
    ].max(flag_is_capped_this_tick_by_flag)

    # This tick's flag facts.
    capture_the_flag_facts = CaptureTheFlagTransitionFacts(
        flag_was_picked_up_by_agent=flag_is_picked_up_this_tick_by_agent,
        flag_was_dropped_by_agent=flag_is_dropped_this_tick_by_agent,
        ally_flag_was_returned_by_agent=ally_flag_is_returned_this_tick_by_agent,
        enemy_flag_was_captured_by_agent=enemy_flag_was_captured_this_tick_by_agent,
        flag_was_automatically_returned_by_team=flag_is_automatically_returned_this_tick_by_flag,
        outcome=jnp.asarray(TASK_MODE_OUTCOME_ONGOING, dtype=jnp.int32),
    )

    return CaptureTheFlagFlagUpdate(
        next_capture_the_flag_flag_positions_by_flag=next_flag_position_by_flag,
        next_capture_the_flag_flag_carrier_slots_by_flag=next_flag_carrier_slot_by_flag,
        next_capture_the_flag_dropped_flag_countdowns_by_flag=next_dropped_flag_countdown_by_flag,
        next_capture_the_flag_scores_by_team=next_capture_the_flag_scores_by_team,
        capture_the_flag_facts=capture_the_flag_facts,
    )


def update_king_of_the_hills_hills(
    current_state: EnvState,
    config: EnvConfig,
    next_agent_positions: Array,
) -> KingOfTheHillsHillUpdate:
    """Move the hills through one tick's rules and record what happened.

    The rules run in a fixed order, each reading the one before: who stands
    on each hill, contest, ownership with the empty-hill timer, then points
    and scores.

    Parameters
    ----------
    current_state : EnvState
        The state at the start of the tick. Reads king_of_the_hills_hill_owners,
        king_of_the_hills_empty_hill_ownership_countdowns,
        king_of_the_hills_scores and, for eligibility, alive_mask and
        spawn_shield_durations.
    config : EnvConfig
        The game's validated settings. Reads agent_profile.active_mask,
        agent_profile.team_ids, king_of_the_hills_hill_positions,
        king_of_the_hills_hill_radii,
        king_of_the_hills_empty_hill_ownership_steps,
        king_of_the_hills_hill_points_per_step and
        king_of_the_hills_three_hill_sweep_bonus_points_per_step.
    next_agent_positions : Array
        Float32 (10, 2): every agent's [x, y] at the end of this tick's
        movement, before the respawn wave moves anyone.

    Returns
    -------
    KingOfTheHillsHillUpdate
        The hill state and scores for the successor, and this tick's hill
        facts.

    Notes
    -----
    - Eligible: active, alive and unshielded at the start of the tick. An
      agent killed this tick still counts this tick, so no death facts are
      needed.
    - On a hill: an eligible agent whose end-of-movement center is within a
      used hill's radius (inclusive).
    - Contested: at least one eligible agent of each team on the hill.
    - Ownership: a team alone on a hill takes it at once, whatever its ticks
      left, and its ticks left becomes the timer. With both teams on it the
      owner stays, and an owned hill's ticks left becomes the timer. An empty
      hill keeps its owner while its ticks left is above 0, losing one tick;
      an empty hill at 0 turns neutral. A neutral hill always has 0 ticks
      left, and an unused hill is never owned or contested.
    - Points: each owned, uncontested hill pays its owner the rate, empty
      ticks included while the timer runs. A team holding all three hills
      uncontested also gets the sweep bonus, once per tick. Both teams can
      score on one tick, and each score rises by its points.
    - The facts' outcome is left ongoing (0); step sets it from its outcome
      branch.

    Pure JAX for one game, with no host work, randomness or input mutation;
    use it under jit, scan and an outer vmap. Step calls it only in King of the
    Hills games.
    """
    hill_interaction_matrix_by_agent_by_hill = build_agent_hill_interaction_matrix(
        next_agent_positions,
        config.king_of_the_hills_hill_positions,
        config.king_of_the_hills_hill_radii,
    )

    is_on_hill_by_agent_by_hill = (
        hill_interaction_matrix_by_agent_by_hill
        & _build_objective_eligibility_by_agent(current_state, config)[:, None]
    )  # (10, 3)

    # Which teams are on the hills? (2,3)
    is_teammate_by_team_by_agent = jnp.vstack(
        (
            config.agent_profile.team_ids == TEAM_A_ID,
            config.agent_profile.team_ids == TEAM_B_ID,
        ),
    )  # (2, 10)

    is_on_hill_by_team_by_agent_by_hill = (
        is_teammate_by_team_by_agent[:, :, None]
        & is_on_hill_by_agent_by_hill[None, :, :]
    )  # (2, 10, 3)

    # (2, 3)
    is_on_hill_by_team_by_hill = jnp.any(is_on_hill_by_team_by_agent_by_hill, axis=1)

    # (3,)
    hill_is_contested_by_hill = jnp.all(is_on_hill_by_team_by_hill, axis=0)

    team_a_on_hill_alone_by_hill = is_on_hill_by_team_by_hill[TEAM_A_ID - 1] & (
        ~is_on_hill_by_team_by_hill[TEAM_B_ID - 1]
    )
    team_b_on_hill_alone_by_hill = is_on_hill_by_team_by_hill[TEAM_B_ID - 1] & (
        ~is_on_hill_by_team_by_hill[TEAM_A_ID - 1]
    )
    ownership_count_is_zero_on_empty_hill_by_hill = (
        current_state.king_of_the_hills_empty_hill_ownership_countdowns == 0
    ) & (~is_on_hill_by_team_by_hill.any(axis=0))
    condlist = [
        team_a_on_hill_alone_by_hill,  # Team A alone takes the hill
        team_b_on_hill_alone_by_hill,  # Team B alone takes the hill
        ownership_count_is_zero_on_empty_hill_by_hill,  # Empty at 0 ticks left: neutral
    ]

    choicelist = [TEAM_A_ID, TEAM_B_ID, NO_TEAM_ID]

    king_of_the_hills_hill_owners_by_hill = jnp.select(
        condlist,
        choicelist,
        # Otherwise the owner stays: both teams on it, or empty with ticks
        # left above 0.
        current_state.king_of_the_hills_hill_owners,
    )

    # Ticks left: an empty hill above 0 loses one tick; a hill owned after the
    # update above, with anyone on it, resets to the timer; every other hill
    # gets 0.
    ownership_count_is_greater_than_zero_on_empty_hill_by_hill = (
        current_state.king_of_the_hills_empty_hill_ownership_countdowns > 0
    ) & (~is_on_hill_by_team_by_hill.any(axis=0))

    agents_on_hill_that_is_not_neutral_by_hill = is_on_hill_by_team_by_hill.any(
        axis=0
    ) & (
        king_of_the_hills_hill_owners_by_hill != NO_TEAM_ID
    )  # Reset, someone's on the hill

    condlist = [
        ownership_count_is_greater_than_zero_on_empty_hill_by_hill,  # Decrement
        agents_on_hill_that_is_not_neutral_by_hill,  # Reset
    ]

    choicelist = [
        current_state.king_of_the_hills_empty_hill_ownership_countdowns - 1,
        config.king_of_the_hills_empty_hill_ownership_steps,
    ]

    king_of_the_hills_empty_hill_ownership_countdowns_by_hill = jnp.select(
        condlist, choicelist, 0
    )

    # Points earned this tick by team. A contested hill pays nobody: its owner
    # times False is 0, which is NO_TEAM_ID.
    point_giving_hills_this_tick_by_hill = king_of_the_hills_hill_owners_by_hill * (
        ~hill_is_contested_by_hill
    )

    team_a_point_giving_hill_count = jnp.sum(
        point_giving_hills_this_tick_by_hill == TEAM_A_ID
    )
    team_b_point_giving_hill_count = jnp.sum(
        point_giving_hills_this_tick_by_hill == TEAM_B_ID
    )

    point_giving_hill_count_by_team = jnp.array(
        (team_a_point_giving_hill_count, team_b_point_giving_hill_count)
    )

    points_this_tick_before_sweep_bonus_by_team = (
        point_giving_hill_count_by_team * config.king_of_the_hills_hill_points_per_step
    )

    points_this_tick_by_team = jnp.where(
        point_giving_hill_count_by_team == MAX_HILL_SLOTS,
        points_this_tick_before_sweep_bonus_by_team
        + config.king_of_the_hills_three_hill_sweep_bonus_points_per_step,
        points_this_tick_before_sweep_bonus_by_team,
    )

    king_of_the_hills_scores_by_team = (
        current_state.king_of_the_hills_scores + points_this_tick_by_team
    )

    king_of_the_hills_facts = KingOfTheHillsTransitionFacts(
        is_on_hill_by_agent_by_hill=is_on_hill_by_agent_by_hill,
        hill_owners=king_of_the_hills_hill_owners_by_hill,
        points_scored_by_team=points_this_tick_by_team,
        outcome=jnp.asarray(TASK_MODE_OUTCOME_ONGOING, dtype=jnp.int32),
    )

    return KingOfTheHillsHillUpdate(
        next_king_of_the_hills_hill_owners_by_hill=king_of_the_hills_hill_owners_by_hill,
        next_king_of_the_hills_empty_hill_ownership_countdowns_by_hill=king_of_the_hills_empty_hill_ownership_countdowns_by_hill,
        next_king_of_the_hills_hill_is_contested_by_hill=hill_is_contested_by_hill,
        next_king_of_the_hills_scores_by_team=king_of_the_hills_scores_by_team,
        king_of_the_hills_facts=king_of_the_hills_facts,
    )
