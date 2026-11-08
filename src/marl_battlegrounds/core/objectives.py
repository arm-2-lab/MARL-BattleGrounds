"""Apply Capture the Flag's flag rules for one tick, in pure JAX.

build_flag_state_aggregation_result is the one owner of how the two flags move
between home, carried and dropped: carry, return, pickup, the death drop and the
dropped-flag timer, in that order. Core's step calls it once per tick inside the
Capture the Flag mode branch, after deaths and before the respawn wave. Every
other mode keeps its flags through env's _not_capture_the_flag. There is no
capture rule yet: nobody scores and the outcome stays ongoing. No host work,
randomness or I/O; it runs under jit, vmap and scan.
"""

from typing import NamedTuple, cast

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    TASK_MODE_OUTCOME_ONGOING,
    TEAM_A_ID,
    TEAM_B_ID,
    CaptureTheFlagTransitionFacts,
    EnvConfig,
    EnvState,
)


class CaptureTheFlagFlagStateAggregationResult(NamedTuple):
    """Hold the flag state after one tick and that tick's flag facts.

    Flag arrays use flag order [Team A's flag, Team B's flag], which is also
    team order. Step copies the three arrays into the successor EnvState's flag
    fields and the facts into TransitionFacts. This immutable record computes
    nothing.
    """

    next_capture_the_flag_flag_positions_by_flag: Array
    """Float32 (2, 2): each flag's [x, y] in map units after this tick."""
    next_capture_the_flag_flag_carrier_slots_by_flag: Array
    """Int32 (2,): each flag's carrier as a global slot (0-9), or -1 for none."""
    next_capture_the_flag_dropped_flag_countdowns_by_flag: Array
    """Int32 (2,): ticks before each dropped flag goes home; 0 when not dropped."""
    capture_the_flag_facts: CaptureTheFlagTransitionFacts
    """Who picked up, dropped and returned each flag, and the timer returns."""


def build_flag_state_aggregation_result(
    current_state: EnvState,
    config: EnvConfig,
    next_agent_positions: Array,
    is_newly_dead_by_recipient: Array,
) -> CaptureTheFlagFlagStateAggregationResult:
    """Move both flags through one tick's flag rules and record what happened.

    The rules run in a fixed order, each reading the result of the one before:
    carry, return, pickup, drop, then the dropped-flag timer. A flag is carried
    when its carrier slot is 0 or more, dropped when its countdown is above 0,
    and home otherwise.

    Parameters
    ----------
    current_state : EnvState
        The state at the start of the tick. Reads the three flag fields and,
        for eligibility, alive_mask and spawn_shield_durations.
    config : EnvConfig
        The game's validated settings. Reads agent_profile.active_mask,
        agent_profile.team_ids, capture_the_flag_flag_home_positions,
        capture_the_flag_flag_interaction_distance and
        capture_the_flag_dropped_flag_auto_return_steps.
    next_agent_positions : Array
        Float32 (10, 2): every agent's [x, y] at the end of this tick's
        movement, before the respawn wave moves anyone.
    is_newly_dead_by_recipient : Array
        Bool (10,): True for each agent killed this tick (death facts).

    Returns
    -------
    CaptureTheFlagFlagStateAggregationResult
        The flag state for the successor and this tick's flag facts.

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
    - Drop: a carrier killed this tick, including one that picked the flag up
      this tick, drops it where it stands with a full countdown.
    - Timer: every other dropped flag counts down by 1 and goes home at 0.
    - No capture rule yet: the capture fact is all False and the outcome is
      ongoing (0).

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

    flag_is_carried_by_flag = jnp.logical_not(flag_carrier_slot_by_flag == -1)

    next_flag_position_by_flag = jnp.where(
        flag_is_carried_by_flag[:, None],
        flag_carrier_next_agent_position_by_flag,
        current_flag_position_by_flag,
    )

    # Eligible: active, alive and unshielded at the start of the tick.
    is_alive_and_active_by_agent = jnp.logical_and(
        current_state.alive_mask, config.agent_profile.active_mask
    )
    is_eligible_to_interact_with_a_flag_by_agent = jnp.logical_and(
        is_alive_and_active_by_agent, current_state.spawn_shield_durations == 0
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

    flag_is_returned_this_tick_by_flag = (
        flag_was_dropped_at_tick_start_by_flag
        & (nearest_eligible_and_touching_ally_slot_by_flag != -1)
        & (
            nearest_eligible_and_touching_ally_distance_by_flag
            < nearest_eligible_and_touching_enemy_distance_by_flag
        )
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

    # Carriers after the pickup, kept under their own name for the drop record.
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

    # Drop: the carrier after the pickup was killed this tick.
    flag_is_dropped_this_tick_by_flag = (
        flag_carrier_slot_after_pickup_by_flag != -1
    ) & (is_newly_dead_by_recipient[flag_carrier_slot_after_pickup_by_flag])

    # A dropped flag starts a full countdown and loses its carrier.
    next_dropped_flag_countdown_by_flag = jnp.where(
        flag_is_dropped_this_tick_by_flag,
        config.capture_the_flag_dropped_flag_auto_return_steps,
        next_dropped_flag_countdown_by_flag,
    )

    next_flag_carrier_slot_by_flag = jnp.where(
        flag_is_dropped_this_tick_by_flag, -1, flag_carrier_slot_after_pickup_by_flag
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

    # Who dropped each flag: its carrier after the pickup, if killed this tick.
    flag_is_dropped_this_tick_by_agent = all_false_by_agent.at[
        flag_carrier_slot_after_pickup_by_flag
    ].max(flag_is_dropped_this_tick_by_flag)

    # Who returned each flag: the nearest eligible owner touching it.
    ally_flag_is_returned_this_tick_by_agent = all_false_by_agent.at[
        nearest_eligible_and_touching_ally_slot_by_flag
    ].max(flag_is_returned_this_tick_by_flag)

    # This tick's flag facts.
    capture_the_flag_facts = CaptureTheFlagTransitionFacts(
        flag_was_picked_up_by_agent=flag_is_picked_up_this_tick_by_agent,
        flag_was_dropped_by_agent=flag_is_dropped_this_tick_by_agent,
        ally_flag_was_returned_by_agent=ally_flag_is_returned_this_tick_by_agent,
        enemy_flag_was_captured_by_agent=all_false_by_agent,  # No capture rule yet.
        flag_was_automatically_returned_by_team=flag_is_automatically_returned_this_tick_by_flag,
        outcome=jnp.asarray(TASK_MODE_OUTCOME_ONGOING, dtype=jnp.int32),
    )

    return CaptureTheFlagFlagStateAggregationResult(
        next_capture_the_flag_flag_positions_by_flag=next_flag_position_by_flag,
        next_capture_the_flag_flag_carrier_slots_by_flag=next_flag_carrier_slot_by_flag,
        next_capture_the_flag_dropped_flag_countdowns_by_flag=next_dropped_flag_countdown_by_flag,
        capture_the_flag_facts=capture_the_flag_facts,
    )
