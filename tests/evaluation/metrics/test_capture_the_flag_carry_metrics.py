"""Check Capture the Flag's carry and carrier-support columns against worked values.

Six hand-built games on map 0 (three champions a team, no walls, so every
agent sees every other) run for 4 ticks through evaluate_episodes, and
replay_metrics(metrics="full") measures each saved replay. Nobody moves. Each
agent heals its own team's first agent (slot 0 or 5) when it can, which only a
Priest can do, or else hits the other team's first agent when it is in reach.
Flag 0 is Team A's, flag 1 Team B's; Team A's flag home is (0.5, 5) and Team
B's is (19.5, 5). "Start-state carry" means the game starts with the flag
already held, so that carry has no pickup in the game.

Game 1: Team A's Mage steals at Team B's flag home on tick 1 and carries for 3
ticks with its Warrior 2 units away. Team B's Priest has a start-state carry
and heals Team B's Mage, which is not carrying, for 8 a tick.
Game 2: Team A's Mage has a start-state carry 0.6 from its own flag home, with
its Warrior 2 units away, and captures on tick 1.
Game 3: each team's Mage carries the other's flag, so neither can score. Team
A's carrier stands 0.9 from its flag home (outside the 0.75 capture distance,
inside the old 1.25) with its Warrior 1.5 away, inside Warrior aura, and its
Priest 2.5 away healing it for 8 a tick. Team B's carrier stands 0.5 from its
flag home, alone.
Game 4: Team A's Mage steals on tick 1 at 20 health, and Team B's Mage, 2.8
away, kills it on tick 2. Team A's Warrior, 2.4 from the carrier, hits that
enemy Mage for 8 every tick.
Game 5: Team B's Mage has a start-state carry at 1 health with no ally near and
is killed on tick 1. Team A's Priest picks Team B's flag up off the ground on
tick 1 and carries for 3 ticks with no ally near.
Game 6: Team A's Priest carries Team B's flag at (8, 5) and heals Team A's Mage,
2 units away at 40 health, while that Mage hits Team B's Mage, which stands 2
units from the carrier and hits back. The Mage is attacking enemies near the
carrier every tick; the carrier never is, even through the Priest rule.

Contracts: each game plays as scripted (carriers and deaths at the start of
every tick). Position rows read the start of the tick, so a pickup counts from
the next tick and a death tick still counts as carried. Each test ID names a
game and a family of columns. carry: steals, pickups, carry ticks, captures,
deaths, carry duration, captures per steal, carry ticks ending in a capture or
a death, carry distance and speed, the capture-range count at each game's 0.75
capture distance, distance to the flag home, and carries ending in death
within 5, 10 and 20 ticks. escort: the escort-size splits ("with_1_ally_near"
for one ally), distances to the carrier and the closest ally, and captures and
deaths near the carrier. healing: healing to allies who are carrying and the
carrier's share of a Priest's healing. own_healing: a carrying Priest's own
healing, and Freedom. aura: aura cover of the carrier. carrier_combat: damage
and kill contributions while carrying. situations and carrying_block: the tick
blocks for Near the Ally Carrier, Attacking Enemies Near the Ally Carrier and
Carrying. own_healing, aura and carrier_combat are columns a later packet
builds, so they are expected to fail until it lands. Blanks are NaN: means
with nothing to average, healing by agents that are not Priests, an agent
paired with itself. Views and pairs across teams are not columns. The training
reward's carry_steps counter, run over each game's saved carrier slots with
Core's drop facts marking carrier deaths, counts the same deaths within 5 ticks
of the pickup as the metric does, for every team that starts its game without a
carrier (as training games do). These games hold no death on a pickup or
capture tick, so those two cases rest on the counter's own tests. Runs
on CPU in about 40 s.
"""

# pyright: reportPrivateUsage=false
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest

from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import ActionMask, EnvState
from marl_battlegrounds.evaluation._replays.replay_v5 import ReplayArtifactV5
from marl_battlegrounds.evaluation.analysis import replay_metrics
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.models import TransitionFactsCaptureTheFlagV2
from marl_battlegrounds.evaluation.policy_execution import Policy
from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundle, load_replay
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput
from marl_battlegrounds.tasks import (
    AgentClassName,
    make_standard_capture_the_flag_config,
)
from marl_battlegrounds.training.shaping import _team_carry_steps

_ROSTER: tuple[AgentClassName, ...] = ("mage", "warrior", "priest")
_CONFIG = make_standard_capture_the_flag_config(
    map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER, max_steps=4
)
_FAR = {
    0: (3.0, 1.0),
    1: (3.0, 3.0),
    2: (3.0, 9.0),
    5: (12.0, 1.0),
    6: (12.0, 3.0),
    7: (12.0, 9.0),
}
_BLANK = math.nan
_Point = tuple[float, float]


def _heal_or_hit(
    variables: jax.Array,
    carry: jax.Array,
    actor: ActorInput,
    mask: ActionMask,
    key: jax.Array,
) -> tuple[ActorAction, jax.Array]:
    # Target 1 is the actor's own team's first row (slot 0 or 5), target 6 the
    # other team's first row; column 0 of the joint mask is "Basic, no Ultimate".
    pairs = mask.select_target_use_ultimate_joint_mask
    target = jnp.where(pairs[1, 0], 1, jnp.where(pairs[6, 0], 6, 0))
    stay = jnp.zeros((), jnp.int32)
    return ActorAction(stay, target.astype(jnp.int32), stay), carry


def _start(
    positions: Mapping[int, _Point],
    *,
    flags: Mapping[int, _Point],
    carriers: Mapping[int, int] | None = None,
    countdowns: Mapping[int, int] | None = None,
    health: Mapping[int, int] | None = None,
) -> EnvState:
    state = reset(_CONFIG, jax.random.key(0))[0]
    where = state.agent_positions
    for slot, point in {**_FAR, **positions}.items():
        where = where.at[slot].set(jnp.asarray(point, jnp.float32))
    flag_positions = state.capture_the_flag_flag_positions
    for flag, point in flags.items():
        flag_positions = flag_positions.at[flag].set(jnp.asarray(point, jnp.float32))
    carrier_slots = state.capture_the_flag_flag_carrier_slots
    for flag, slot in (carriers or {}).items():
        carrier_slots = carrier_slots.at[flag].set(slot)
    flag_countdowns = state.capture_the_flag_dropped_flag_countdowns
    for flag, steps in (countdowns or {}).items():
        flag_countdowns = flag_countdowns.at[flag].set(steps)
    current_health = state.current_health
    for slot, points in (health or {}).items():
        current_health = current_health.at[slot].set(points)
    return state._replace(
        agent_positions=where,
        capture_the_flag_flag_positions=flag_positions,
        capture_the_flag_flag_carrier_slots=carrier_slots,
        capture_the_flag_dropped_flag_countdowns=flag_countdowns,
        current_health=current_health,
        spawn_shield_durations=jnp.zeros_like(state.spawn_shield_durations),
    )


_GAMES = {
    1: _start(
        {
            0: (19.0, 5.0),
            1: (17.0, 5.0),
            5: (12.0, 1.0),
            6: (12.0, 9.0),
            7: (12.0, 3.5),
        },
        flags={0: (12.0, 3.5)},
        carriers={0: 7},
        health={5: 20},
    ),
    2: _start({0: (1.1, 5.0), 1: (1.1, 7.0)}, flags={1: (1.1, 5.0)}, carriers={1: 0}),
    3: _start(
        {0: (1.4, 5.0), 1: (1.4, 6.5), 2: (3.9, 5.0), 5: (19.0, 5.0)},
        flags={0: (19.0, 5.0), 1: (1.4, 5.0)},
        carriers={0: 5, 1: 0},
        health={0: 20},
    ),
    4: _start(
        {0: (19.0, 5.0), 1: (18.0, 7.2), 5: (19.0, 7.8)}, flags={}, health={0: 20}
    ),
    5: _start(
        {
            0: (12.5, 5.0),
            1: (10.0, 6.2),
            5: (10.0, 5.0),
            6: (17.0, 1.0),
            7: (17.0, 9.0),
        },
        flags={0: (10.0, 5.0), 1: (3.0, 9.0)},
        carriers={0: 5},
        countdowns={1: 10},
        health={5: 1},
    ),
    6: _start(
        {0: (8.0, 3.0), 2: (8.0, 5.0), 5: (10.0, 5.0)},
        flags={1: (8.0, 5.0)},
        carriers={1: 2},
        health={0: 40},
    ),
}

# At the start of ticks 1 to 4: (Team A's carrier, Team B's carrier, dead slots).
_TIMELINE: dict[int, list[tuple[int | None, int | None, list[int]]]] = {
    1: [(None, 7, []), (0, 7, []), (0, 7, []), (0, 7, [])],
    2: [(0, None, []), (None, None, []), (None, None, []), (None, None, [])],
    3: [(0, 5, []), (0, 5, []), (0, 5, []), (0, 5, [])],
    4: [(None, None, []), (0, None, []), (None, None, [0]), (None, None, [0])],
    5: [(None, 5, []), (2, None, [5]), (2, None, [5]), (2, None, [5])],
    6: [(2, None, []), (2, None, []), (2, None, []), (2, None, [])],
}


def _escort(name: str, stem: str, size: int, count: float) -> dict[str, float]:
    # All five escort-size buckets of one column; the bucket for one ally reads
    # "with_1_ally_near". Every bucket but `size` holds 0.
    return {
        f"{name}_{stem}_with_{n}_{'ally' if n == 1 else 'allies'}_near__carrying": (
            count if n == size else 0.0
        )
        for n in range(5)
    }


def _deaths_within(name: str, count: float) -> dict[str, float]:
    return {
        f"{name}_carries_ending_in_death_within_{n}_steps": count for n in (5, 10, 20)
    }


# Families: "carry" is Table 1; "escort", "healing", "own_healing" and "aura"
# are Table 2; "carrier_combat" is the Carrying row of the combat grid;
# "situations" and "carrying_block" are the situation tick blocks.
_EXPECTED: dict[tuple[int, str], dict[str, float]] = {
    (1, "carry"): {
        "agent_0_flag_steals": 1,
        "team_a_flag_steals": 1,
        "agent_0_dropped_enemy_flag_pickups": 0,
        "team_a_dropped_enemy_flag_pickups": 0,
        # Picked up on tick 1, so held at the start of ticks 2, 3 and 4.
        "agent_0_steps__carrying": 3,
        "agent_1_steps__carrying": 0,
        "team_a_steps_with_an_agent__carrying": 3,
        "agent_7_steps__carrying": 4,
        "team_b_steps_with_an_agent__carrying": 4,
        "agent_0_flag_captures": 0,
        "team_a_flag_captures": 0,
        "agent_0_deaths__carrying": 0,
        "team_a_deaths__carrying": 0,
        # A carry still going at the end counts with its ticks so far: 3 / 1.
        "agent_0_carry_duration_mean": 3.0,
        "team_a_carry_duration_mean": 3.0,
        "agent_1_carry_duration_mean": _BLANK,
        # A start-state carry has no pickup, so there is nothing to divide by.
        "agent_7_carry_duration_mean": _BLANK,
        "team_b_carry_duration_mean": _BLANK,
        "team_a_flag_captures_per_steal": 0.0,
        "team_b_flag_captures_per_steal": _BLANK,
        "agent_0_carry_steps_ending_in_a_capture": 0,
        # Ticks of a carry still going when the game ends are not counted.
        "agent_0_carry_steps_ending_in_death": 0,
        "team_a_carry_steps_ending_in_death": 0,
        "agent_0_carry_distance": 0.0,
        "agent_0_carry_speed_mean": 0.0,
        "team_a_carry_speed_mean": 0.0,
        "agent_1_carry_speed_mean": _BLANK,
        "agent_0_carry_steps_in_capture_range_with_the_ally_flag_away": 0,
        "agent_0_distance_to_the_ally_flag_home_mean__carrying": 18.5,
        "team_a_distance_to_the_ally_flag_home_mean__carrying": 18.5,
        "agent_1_distance_to_the_ally_flag_home_mean__carrying": _BLANK,
        "agent_7_distance_to_the_ally_flag_home_mean__carrying": math.hypot(7.5, 1.5),
        **_deaths_within("agent_0", 0),
        **_deaths_within("team_a", 0),
    },
    (1, "escort"): {
        **_escort("agent_0", "steps", 1, 3),
        **_escort("team_a", "steps", 1, 3),
        **_escort("agent_1", "steps", 0, 0),
        **_escort("agent_7", "steps", 1, 4),
        **_escort("team_b", "steps", 1, 4),
        "agent_1_distance_to_the_ally_carrier_mean": 2.0,
        "agent_1_distance_to_the_ally_carrier_observations": 3,
        "agent_2_distance_to_the_ally_carrier_mean": math.hypot(16.0, 4.0),
        "agent_2_distance_to_the_ally_carrier_observations": 3,
        # The carrier itself is never measured.
        "agent_0_distance_to_the_ally_carrier_mean": _BLANK,
        "agent_0_distance_to_the_ally_carrier_observations": 0,
        "team_a_distance_to_the_ally_carrier_mean": (2.0 + math.hypot(16.0, 4.0)) / 2,
        "team_a_distance_to_the_ally_carrier_observations": 6,
        "agent_1_to_agent_0_distance_to_the_ally_carrier_mean": 2.0,
        "agent_1_to_agent_0_distance_to_the_ally_carrier_observations": 3,
        "agent_0_to_agent_0_distance_to_the_ally_carrier_mean": _BLANK,
        "agent_0_to_agent_1_distance_to_the_ally_carrier_mean": _BLANK,
        "agent_0_to_agent_1_distance_to_the_ally_carrier_observations": 0,
        "agent_5_distance_to_the_ally_carrier_mean": 2.5,
        "agent_5_distance_to_the_ally_carrier_observations": 4,
        "agent_6_distance_to_the_ally_carrier_mean": 5.5,
        "team_b_distance_to_the_ally_carrier_mean": 4.0,
        "team_b_distance_to_the_ally_carrier_observations": 8,
        "agent_5_to_agent_7_distance_to_the_ally_carrier_mean": 2.5,
        "agent_5_to_agent_7_distance_to_the_ally_carrier_observations": 4,
        "agent_0_distance_to_the_closest_ally_mean__carrying": 2.0,
        "agent_0_distance_to_the_closest_ally_observations__carrying": 3,
        "team_a_distance_to_the_closest_ally_mean__carrying": 2.0,
        "team_a_distance_to_the_closest_ally_observations__carrying": 3,
        "agent_7_distance_to_the_closest_ally_mean__carrying": 2.5,
        "agent_7_distance_to_the_closest_ally_observations__carrying": 4,
        "agent_1_ally_carrier_captures__near_the_ally_carrier": 0,
        "agent_1_ally_carrier_deaths__near_the_ally_carrier": 0,
        "agent_1_deaths__near_the_ally_carrier": 0,
    },
    (1, "healing"): {
        # Team B's carrier is the Priest, healing a teammate who is not
        # carrying: none of it is healing to allies who are carrying.
        "agent_7_effective_healing_done_to_allies_who_are__carrying": 0.0,
        "team_b_effective_healing_done_to_allies_who_are__carrying": 0.0,
        "agent_7_to_agent_5_effective_healing_done_to_allies_who_are__carrying": 0.0,
        "agent_5_effective_priest_healing_received__carrying": 0.0,
        "agent_7_effective_healing_done_to_allies_during_ally_carries": 32.0,
        "team_b_effective_healing_done_to_allies_during_ally_carries": 32.0,
        "agent_7_ally_carrier_healing_focus_fraction": 0.0,
        "team_b_ally_carrier_healing_focus_fraction": 0.0,
        "agent_2_effective_healing_done_to_allies_who_are__carrying": 0.0,
        "agent_2_effective_healing_done_to_allies_during_ally_carries": 0.0,
        "agent_2_ally_carrier_healing_focus_fraction": _BLANK,
        "team_a_ally_carrier_healing_focus_fraction": _BLANK,
        "agent_0_effective_healing_done_to_allies_who_are__carrying": _BLANK,
        "agent_1_effective_healing_done_to_allies_during_ally_carries": _BLANK,
    },
    (1, "own_healing"): {
        # The carrying Priest's healing counts under its own Carrying.
        "agent_7_effective_healing_done__carrying": 32.0,
        "team_b_effective_healing_done__carrying": 32.0,
        "team_a_effective_healing_done__carrying": 0.0,
        "agent_0_effective_healing_done__carrying": _BLANK,
        "agent_7_priest_freedom_applications_to_allies_who_are__carrying": 0,
    },
    (1, "situations"): {
        "agent_0_steps__near_the_ally_carrier": 0,
        "agent_1_steps__near_the_ally_carrier": 3,
        "agent_2_steps__near_the_ally_carrier": 0,
        "agent_1_solo_steps__near_the_ally_carrier": 3,
        "team_a_steps_with_an_agent__near_the_ally_carrier": 3,
        "team_a_agent_steps__near_the_ally_carrier": 3,
        "team_a_solo_steps__near_the_ally_carrier": 3,
        "team_a_steps_with_two_or_more_agents__near_the_ally_carrier": 0,
        "team_a_headcount_mean__near_the_ally_carrier": 1.0,
        "agent_1_step_participation__near_the_ally_carrier": 1.0,
        "agent_2_step_participation__near_the_ally_carrier": 0.0,
        "agent_1_solo_step_fraction__near_the_ally_carrier": 1.0,
        "agent_2_solo_step_fraction__near_the_ally_carrier": _BLANK,
        "agent_5_steps__near_the_ally_carrier": 4,
        "agent_6_steps__near_the_ally_carrier": 0,
        "team_b_steps_with_an_agent__near_the_ally_carrier": 4,
        "team_b_headcount_mean__near_the_ally_carrier": 1.0,
        "team_a_steps_with_an_agent__attacking_enemies_near_the_ally_carrier": 0,
        "team_a_headcount_mean__attacking_enemies_near_the_ally_carrier": _BLANK,
        "agent_1_step_participation__attacking_enemies_near_the_ally_carrier": _BLANK,
    },
    (1, "carrying_block"): {
        "agent_0_step_participation__carrying": 1.0,
        "agent_1_step_participation__carrying": 0.0,
        "agent_0_solo_step_fraction__carrying": 1.0,
        "agent_1_solo_step_fraction__carrying": _BLANK,
        "team_a_headcount_mean__carrying": 1.0,
        "team_a_steps_with_two_or_more_agents__carrying": _BLANK,
        "team_b_headcount_mean__carrying": 1.0,
        "team_b_steps_with_two_or_more_agents__carrying": _BLANK,
    },
    (2, "carry"): {
        "agent_0_flag_captures": 1,
        "team_a_flag_captures": 1,
        "team_b_flag_captures": 0,
        "agent_0_flag_steals": 0,
        "team_a_flag_steals": 0,
        "agent_0_steps__carrying": 1,
        "team_a_steps_with_an_agent__carrying": 1,
        "agent_0_carry_steps_ending_in_a_capture": 1,
        "team_a_carry_steps_ending_in_a_capture": 1,
        "agent_0_carry_steps_ending_in_death": 0,
        "agent_0_deaths__carrying": 0,
        "team_a_flag_captures_per_steal": _BLANK,
        "agent_0_carry_duration_mean": _BLANK,
        "team_a_carry_duration_mean": _BLANK,
        "agent_0_carry_distance": 0.0,
        "agent_0_carry_speed_mean": 0.0,
        "agent_0_carry_steps_in_capture_range_with_the_ally_flag_away": 0,
        "agent_0_distance_to_the_ally_flag_home_mean__carrying": 0.6,
        "team_b_distance_to_the_ally_flag_home_mean__carrying": _BLANK,
        "team_b_carry_duration_mean": _BLANK,
        "team_b_carry_speed_mean": _BLANK,
    },
    (2, "escort"): {
        **_escort("agent_0", "flag_captures", 1, 1),
        **_escort("team_a", "flag_captures", 1, 1),
        **_escort("agent_0", "steps", 1, 1),
        **_escort("agent_0", "deaths", 0, 0),
        "agent_1_ally_carrier_captures__near_the_ally_carrier": 1,
        "agent_2_ally_carrier_captures__near_the_ally_carrier": 0,
        "agent_0_ally_carrier_captures__near_the_ally_carrier": 0,
        "agent_1_distance_to_the_ally_carrier_mean": 2.0,
        "agent_1_distance_to_the_ally_carrier_observations": 1,
        "agent_2_distance_to_the_ally_carrier_mean": math.hypot(1.9, 4.0),
        "agent_0_distance_to_the_closest_ally_mean__carrying": 2.0,
        "agent_0_distance_to_the_closest_ally_observations__carrying": 1,
        "team_b_distance_to_the_ally_carrier_mean": _BLANK,
        "team_b_distance_to_the_ally_carrier_observations": 0,
    },
    (2, "situations"): {
        "agent_1_steps__near_the_ally_carrier": 1,
        "team_a_steps_with_an_agent__near_the_ally_carrier": 1,
    },
    (2, "carrying_block"): {
        "team_b_headcount_mean__carrying": _BLANK,
        "agent_5_step_participation__carrying": _BLANK,
    },
    (3, "carry"): {
        "agent_0_steps__carrying": 4,
        "agent_5_steps__carrying": 4,
        "team_a_flag_captures": 0,
        "team_b_flag_captures": 0,
        # 0.9 from its flag home is outside the game's 0.75 capture distance.
        "agent_0_carry_steps_in_capture_range_with_the_ally_flag_away": 0,
        "team_a_carry_steps_in_capture_range_with_the_ally_flag_away": 0,
        "agent_5_carry_steps_in_capture_range_with_the_ally_flag_away": 4,
        "team_b_carry_steps_in_capture_range_with_the_ally_flag_away": 4,
        "agent_0_distance_to_the_ally_flag_home_mean__carrying": 0.9,
        "team_a_distance_to_the_ally_flag_home_mean__carrying": 0.9,
        "agent_5_distance_to_the_ally_flag_home_mean__carrying": 0.5,
    },
    (3, "escort"): {
        **_escort("agent_0", "steps", 2, 4),
        **_escort("team_a", "steps", 2, 4),
        **_escort("agent_5", "steps", 0, 4),
        **_escort("team_b", "steps", 0, 4),
        "agent_1_distance_to_the_ally_carrier_mean": 1.5,
        "agent_1_distance_to_the_ally_carrier_observations": 4,
        "agent_2_distance_to_the_ally_carrier_mean": 2.5,
        "team_a_distance_to_the_ally_carrier_mean": 2.0,
        "team_a_distance_to_the_ally_carrier_observations": 8,
        "agent_0_distance_to_the_closest_ally_mean__carrying": 1.5,
        "agent_0_distance_to_the_closest_ally_observations__carrying": 4,
        "agent_5_distance_to_the_closest_ally_mean__carrying": math.hypot(7.0, 2.0),
    },
    (3, "healing"): {
        # The carrier starts at 20 of 80 health and is missing more than 8
        # before each of the four heals, so every heal of 8 is fully effective.
        "agent_2_effective_healing_done_to_allies_who_are__carrying": 32.0,
        "agent_2_excess_healing_to_allies_who_are__carrying": 0.0,
        "team_a_effective_healing_done_to_allies_who_are__carrying": 32.0,
        "team_a_excess_healing_to_allies_who_are__carrying": 0.0,
        "agent_2_to_agent_0_effective_healing_done_to_allies_who_are__carrying": 32.0,
        "agent_2_to_agent_0_excess_healing_to_allies_who_are__carrying": 0.0,
        "agent_2_to_agent_1_effective_healing_done_to_allies_who_are__carrying": 0.0,
        "agent_2_to_agent_2_effective_healing_done_to_allies_who_are__carrying": _BLANK,
        "agent_1_to_agent_0_effective_healing_done_to_allies_who_are__carrying": _BLANK,
        "agent_0_effective_priest_healing_received__carrying": 32.0,
        "agent_2_effective_healing_done_to_allies_during_ally_carries": 32.0,
        "team_a_effective_healing_done_to_allies_during_ally_carries": 32.0,
        "agent_2_ally_carrier_healing_focus_fraction": 1.0,
        "team_a_ally_carrier_healing_focus_fraction": 1.0,
        "agent_0_effective_healing_done_to_allies_who_are__carrying": _BLANK,
        "agent_1_effective_healing_done_to_allies_who_are__carrying": _BLANK,
    },
    (3, "own_healing"): {
        "agent_2_effective_healing_done__carrying": 0.0,
        # Every Lesser Heal gives Freedom; with no slow, none of it is effective.
        "agent_2_priest_freedom_applications_to_allies_who_are__carrying": 4,
        "agent_2_effective_priest_freedom_applications_to_allies_who_are__carrying": 0,
        "agent_0_freedom_protected_steps__carrying": 0,
    },
    (3, "aura"): {
        "agent_1_warrior_aura_covered_ally_steps_to_allies_who_are__carrying": 4,
        "agent_0_steps_covered_by_an_ally_warrior_aura__carrying": 4,
        "agent_0_steps_eligible_for_an_ally_warrior_aura__carrying": 4,
        "team_a_agent_steps_covered_by_an_ally_warrior_aura__carrying": 4,
        # The carrier is the team's only Mage, and its own aura never counts.
        "agent_0_steps_covered_by_an_ally_mage_aura__carrying": 0,
        "agent_0_steps_eligible_for_an_ally_mage_aura__carrying": 0,
        "agent_0_mage_aura_covered_ally_steps_to_allies_who_are__carrying": 0,
        "team_a_agent_steps_covered_by_an_ally_mage_aura__carrying": 0,
        "agent_1_mage_aura_covered_ally_steps_to_allies_who_are__carrying": _BLANK,
        "agent_2_warrior_aura_covered_ally_steps_to_allies_who_are__carrying": _BLANK,
        # Team B's Warrior is alive but 7.3 away from its carrier.
        "agent_5_steps_covered_by_an_ally_warrior_aura__carrying": 0,
        "agent_5_steps_eligible_for_an_ally_warrior_aura__carrying": 4,
        "team_b_agent_steps_covered_by_an_ally_warrior_aura__carrying": 0,
    },
    (3, "situations"): {
        "agent_1_steps__near_the_ally_carrier": 4,
        "agent_2_steps__near_the_ally_carrier": 4,
        "agent_1_solo_steps__near_the_ally_carrier": 0,
        "agent_2_solo_steps__near_the_ally_carrier": 0,
        "team_a_steps_with_an_agent__near_the_ally_carrier": 4,
        "team_a_agent_steps__near_the_ally_carrier": 8,
        "team_a_solo_steps__near_the_ally_carrier": 0,
        "team_a_steps_with_two_or_more_agents__near_the_ally_carrier": 4,
        "team_a_headcount_mean__near_the_ally_carrier": 2.0,
        "agent_1_step_participation__near_the_ally_carrier": 1.0,
        "agent_1_solo_step_fraction__near_the_ally_carrier": 0.0,
        "team_b_steps_with_an_agent__near_the_ally_carrier": 0,
        "team_b_headcount_mean__near_the_ally_carrier": _BLANK,
        "agent_6_step_participation__near_the_ally_carrier": _BLANK,
        "agent_6_solo_step_fraction__near_the_ally_carrier": _BLANK,
    },
    (4, "carry"): {
        "agent_0_flag_steals": 1,
        "team_a_flag_steals": 1,
        # Held at the start of tick 2, the tick it died.
        "agent_0_steps__carrying": 1,
        "agent_0_deaths__carrying": 1,
        "team_a_deaths__carrying": 1,
        "agent_0_carry_duration_mean": 1.0,
        "team_a_carry_duration_mean": 1.0,
        "team_a_flag_captures_per_steal": 0.0,
        "agent_0_carry_steps_ending_in_death": 1,
        "team_a_carry_steps_ending_in_death": 1,
        "agent_0_carry_steps_ending_in_a_capture": 0,
        "agent_0_carry_distance": 0.0,
        # Its one carry tick is its death tick, which speed leaves out.
        "agent_0_carry_speed_mean": _BLANK,
        "team_a_carry_speed_mean": _BLANK,
        "agent_0_distance_to_the_ally_flag_home_mean__carrying": 18.5,
        # Died 1 tick after its pickup.
        **_deaths_within("agent_0", 1),
        **_deaths_within("team_a", 1),
        **_deaths_within("agent_1", 0),
    },
    (4, "carrier_combat"): {
        # Its Arcane Blast of 13 is raised by its own Mage aura: 13 x 1.15.
        "agent_0_damage_done__carrying": 14.95,
        "agent_0_kill_contributions__carrying": 0,
    },
    (4, "escort"): {
        **_escort("agent_0", "steps", 1, 1),
        **_escort("agent_0", "deaths", 1, 1),
        **_escort("team_a", "deaths", 1, 1),
        "agent_1_ally_carrier_deaths__near_the_ally_carrier": 1,
        "agent_2_ally_carrier_deaths__near_the_ally_carrier": 0,
        "agent_1_ally_carrier_captures__near_the_ally_carrier": 0,
        "agent_1_distance_to_the_ally_carrier_mean": math.hypot(1.0, 2.2),
        "agent_1_distance_to_the_ally_carrier_observations": 1,
        "agent_0_distance_to_the_closest_ally_mean__carrying": math.hypot(1.0, 2.2),
        "agent_0_distance_to_the_closest_ally_observations__carrying": 1,
    },
    (4, "situations"): {
        # Only tick 2 starts with an ally carrier, and the enemy Mage stood
        # 2.8 from it. The carrier's own hits on that Mage never count here.
        "agent_1_steps__attacking_enemies_near_the_ally_carrier": 1,
        "agent_1_solo_steps__attacking_enemies_near_the_ally_carrier": 1,
        "agent_0_steps__attacking_enemies_near_the_ally_carrier": 0,
        "agent_2_steps__attacking_enemies_near_the_ally_carrier": 0,
        "team_a_steps_with_an_agent__attacking_enemies_near_the_ally_carrier": 1,
        "team_a_agent_steps__attacking_enemies_near_the_ally_carrier": 1,
        "team_a_solo_steps__attacking_enemies_near_the_ally_carrier": 1,
        (
            "team_a_steps_with_two_or_more_agents"
            "__attacking_enemies_near_the_ally_carrier"
        ): 0,
        "team_a_headcount_mean__attacking_enemies_near_the_ally_carrier": 1.0,
        "agent_1_step_participation__attacking_enemies_near_the_ally_carrier": 1.0,
        "agent_0_step_participation__attacking_enemies_near_the_ally_carrier": 0.0,
        "agent_1_solo_step_fraction__attacking_enemies_near_the_ally_carrier": 1.0,
        "agent_0_solo_step_fraction__attacking_enemies_near_the_ally_carrier": _BLANK,
        "agent_1_damage_done__attacking_enemies_near_the_ally_carrier": 8.0,
        "agent_0_damage_done__attacking_enemies_near_the_ally_carrier": 0.0,
        "agent_1_kill_contributions__attacking_enemies_near_the_ally_carrier": 0,
        "team_b_steps_with_an_agent__attacking_enemies_near_the_ally_carrier": 0,
        "agent_1_steps__near_the_ally_carrier": 1,
    },
    (5, "carry"): {
        "agent_2_dropped_enemy_flag_pickups": 1,
        "team_a_dropped_enemy_flag_pickups": 1,
        "agent_2_flag_steals": 0,
        "team_a_flag_steals": 0,
        "agent_2_steps__carrying": 3,
        "team_a_steps_with_an_agent__carrying": 3,
        "agent_2_carry_duration_mean": 3.0,
        "team_a_carry_duration_mean": 3.0,
        "team_a_flag_captures_per_steal": _BLANK,
        "agent_2_carry_speed_mean": 0.0,
        "agent_2_distance_to_the_ally_flag_home_mean__carrying": math.hypot(2.5, 4.0),
        **_deaths_within("agent_2", 0),
        # Held at the start of tick 1 and killed on it.
        "agent_5_steps__carrying": 1,
        "team_b_steps_with_an_agent__carrying": 1,
        "agent_5_deaths__carrying": 1,
        "team_b_deaths__carrying": 1,
        "agent_5_carry_steps_ending_in_death": 1,
        "team_b_carry_steps_ending_in_death": 1,
        "agent_5_carry_distance": 0.0,
        "agent_5_carry_speed_mean": _BLANK,
        "team_b_carry_speed_mean": _BLANK,
        "agent_5_carry_duration_mean": _BLANK,
        "agent_5_distance_to_the_ally_flag_home_mean__carrying": 9.5,
    },
    (5, "carrier_combat"): {
        # A carrier killed this tick still lands the hit it chose: 13 x 1.15.
        "agent_5_damage_done__carrying": 14.95,
        "agent_5_kill_contributions__carrying": 0,
    },
    (5, "escort"): {
        **_escort("agent_2", "steps", 0, 3),
        **_escort("team_a", "steps", 0, 3),
        **_escort("agent_5", "steps", 0, 1),
        **_escort("agent_5", "deaths", 0, 1),
        **_escort("team_b", "deaths", 0, 1),
        "agent_2_distance_to_the_closest_ally_mean__carrying": math.hypot(7.0, 2.8),
        "agent_2_distance_to_the_closest_ally_observations__carrying": 3,
        "agent_0_distance_to_the_ally_carrier_mean": math.hypot(9.5, 4.0),
        "agent_1_distance_to_the_ally_carrier_mean": math.hypot(7.0, 2.8),
        "agent_1_distance_to_the_ally_carrier_observations": 3,
        "agent_5_distance_to_the_closest_ally_mean__carrying": math.hypot(7.0, 4.0),
        "team_b_distance_to_the_ally_carrier_mean": math.hypot(7.0, 4.0),
        "team_b_distance_to_the_ally_carrier_observations": 2,
        "agent_6_ally_carrier_deaths__near_the_ally_carrier": 0,
    },
    (5, "own_healing"): {
        "agent_2_effective_healing_done__carrying": 0.0,
        "team_a_effective_healing_done__carrying": 0.0,
    },
    (6, "situations"): {
        "agent_0_steps__attacking_enemies_near_the_ally_carrier": 4,
        "agent_2_steps__attacking_enemies_near_the_ally_carrier": 0,
        "team_a_steps_with_an_agent__attacking_enemies_near_the_ally_carrier": 4,
        "team_a_solo_steps__attacking_enemies_near_the_ally_carrier": 4,
    },
    (5, "situations"): {
        "team_a_steps_with_an_agent__near_the_ally_carrier": 0,
        "team_a_headcount_mean__near_the_ally_carrier": _BLANK,
        "agent_0_step_participation__near_the_ally_carrier": _BLANK,
        "team_b_steps_with_an_agent__near_the_ally_carrier": 0,
        "team_a_steps_with_an_agent__attacking_enemies_near_the_ally_carrier": 0,
    },
    (5, "carrying_block"): {
        "agent_2_step_participation__carrying": 1.0,
    },
}

# Families whose columns a later packet builds (the grid's cells and Freedom).
_LATER_FAMILIES = frozenset({"own_healing", "aura", "carrier_combat"})

# Views (copies read from another column, never shipped) and pairs across teams.
_NOT_COLUMNS = (
    "agent_0_carry_steps",
    "agent_0_solo_steps__carrying",
    "team_a_agent_steps__carrying",
    "team_a_solo_steps__carrying",
    "team_a_headcount_observations__near_the_ally_carrier",
    "agent_0_carry_duration_observations",
    "team_a_carry_duration_observations",
    "agent_0_carry_speed_observations",
    "team_a_carry_speed_observations",
    "agent_0_distance_to_the_ally_flag_home_observations__carrying",
    "team_a_allies_near_the_ally_carrier_mean",
    "agent_0_isolated_deaths__carrying",
    "team_a_isolated_deaths__carrying",
    "agent_0_to_agent_5_distance_to_the_ally_carrier_mean",
    "agent_2_to_agent_5_effective_healing_done_to_allies_who_are__carrying",
)


@pytest.fixture(scope="module")
def replays(tmp_path_factory: pytest.TempPathFactory) -> dict[int, LoadedReplayBundle]:
    out = tmp_path_factory.mktemp("ctf-carry-metrics")
    policy = Policy("heal-or-hit", _heal_or_hit, jnp.int32(0), jnp.int32(0))
    evaluate_episodes(
        policy,
        policy,
        [
            EpisodeSpec(game, _CONFIG, initial_state=start)
            for game, start in _GAMES.items()
        ],
        num_envs=len(_GAMES),
        save_replays=len(_GAMES),
        output_dir=out,
    )
    loaded: dict[int, LoadedReplayBundle] = {}
    for path in out.rglob("*.marlbg-replay.json"):
        bundle = load_replay(path)
        assert isinstance(bundle.replay, ReplayArtifactV5)
        # The episode ID ends with the scheduled game: "...:episode-3".
        loaded[
            int(bundle.replay.header.context.identity.episode_id.rsplit("-", 1)[1])
        ] = bundle
    return loaded


@pytest.fixture(scope="module")
def measured(replays: dict[int, LoadedReplayBundle]) -> dict[int, dict[str, float]]:
    return {
        game: replay_metrics(bundle, metrics="full") for game, bundle in replays.items()
    }


@pytest.mark.parametrize("game", sorted(_TIMELINE))
def test_each_game_plays_as_scripted(
    replays: dict[int, LoadedReplayBundle], game: int
) -> None:
    replay = replays[game].replay
    assert isinstance(replay, ReplayArtifactV5)
    timeline = [
        (
            frame.snapshot.capture_the_flag_flag_carrier_slots[1],
            frame.snapshot.capture_the_flag_flag_carrier_slots[0],
            [slot for slot in _FAR if not frame.snapshot.alive_mask[slot]],
        )
        for frame in replay.frames[:4]
    ]
    assert timeline == _TIMELINE[game]


@pytest.mark.parametrize(
    ("game", "family"),
    [
        pytest.param(
            *key,
            id=f"game{key[0]}-{key[1]}",
            marks=pytest.mark.xfail(
                strict=True, reason="built with the Capture the Flag grid packet"
            )
            if key[1] in _LATER_FAMILIES
            else (),
        )
        for key in _EXPECTED
    ],
)
def test_columns_match_the_hand_worked_values(
    measured: dict[int, dict[str, float]], game: int, family: str
) -> None:
    values = measured[game]
    wrong = [
        f"{name}: expected {want}, got {values.get(name, 'no such column')}"
        for name, want in _EXPECTED[game, family].items()
        if name not in values
        or values[name] != pytest.approx(want, rel=1e-5, abs=1e-5, nan_ok=True)
    ]
    assert wrong == []


def test_views_and_pairs_across_teams_are_not_columns(
    measured: dict[int, dict[str, float]],
) -> None:
    assert [name for name in _NOT_COLUMNS if name in measured[1]] == []


def test_the_training_carry_counter_counts_the_same_quick_deaths(
    replays: dict[int, LoadedReplayBundle], measured: dict[int, dict[str, float]]
) -> None:
    compared = 0
    for game, bundle in replays.items():
        replay = bundle.replay
        assert isinstance(replay, ReplayArtifactV5)
        # Each frame's carrier slot for flags 0 and 1, with None (no carrier) as -1.
        slots = jnp.asarray(
            [
                [-1 if slot is None else slot for slot in carriers]
                for carriers in (
                    frame.snapshot.capture_the_flag_flag_carrier_slots
                    for frame in replay.frames
                )
            ],
            jnp.int32,
        )
        held = jnp.zeros((1, 2), jnp.int32)
        quick = [0, 0]
        for tick, transition in enumerate(replay.transitions):
            steps, held = _team_carry_steps(
                held,
                jnp.asarray([tick == 0]),
                slots[tick][None],
                slots[tick + 1][None],
                jnp.asarray([True]),
            )
            # A carrier's death is its drop; a death on the capture tick drops
            # nothing, and a pickup-tick death drops with carry_steps 0.
            facts = transition.facts
            assert isinstance(facts, TransitionFactsCaptureTheFlagV2)
            dropped = facts.capture_the_flag_facts.flag_was_dropped_by_agent
            for team in (0, 1):
                if any(dropped[team * 5 : team * 5 + 5]) and int(steps[0, team]) <= 5:
                    quick[team] += 1
        for team, name in enumerate("ab"):
            if int(slots[0, 1 - team]) == -1:
                expected = measured[game][
                    f"team_{name}_carries_ending_in_death_within_5_steps"
                ]
                assert quick[team] == expected, (game, name)
                compared += quick[team]
    # Game 4's carrier dies 1 tick after its steal, so one quick death is compared.
    assert compared == 1
