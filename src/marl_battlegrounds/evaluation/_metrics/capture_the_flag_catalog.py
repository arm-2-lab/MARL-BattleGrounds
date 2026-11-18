"""Name and describe Capture the Flag's metric columns, as the audited inventory does.

The approved inventory of 28 October owns every Capture the
Flag column's name, label, description and direction. _INVENTORY_ROWS holds the rows
this build measures, copied word for word (with the decided capture reach, 0.75);
rows the inventory flags as views (copies of other columns) are never shipped.
_WRITTEN_ROWS writes out the rows the inventory gives as one text for
several columns: the whole-game closest ally, each situation's tick block, and the
approved carries ending in death within 5, 10 and 20 ticks.
capture_the_flag_columns expands each pattern to its columns: agent_{i} to the ten
slots, team_{t} to Team A and Team B, agent_{i}_to_agent_{j} to the 50 ordered pairs
of teammates (the 10 pairs of an agent with itself stay blank), and {n} to escort
sizes 0 to 4 ("with_1_ally_near" for one). capture_the_flag_metric_columns is a
Capture the Flag game's full
list: Team Deathmatch's columns without the Red Zone, then these. Saved result
files name that list "marlbg.ctf.scalar", version 1 (metric_catalog's
metric_schema_for_task_mode); a change to the list needs a new version, with version
1's reader kept. Healing given is a Priest's alone, so those columns name the Priest
as their required class (the class rule). This module also owns the sizes the
columns are cut by (MAX_ALLIES_NEAR, DEATH_WINDOWS, escort_split), which the
measuring code in _metrics/capture_the_flag.py imports. It imports no JAX, so the
result readers and writers can name a Capture the Flag file's columns without it.
Capture the Flag stays out of the public exports until version 1.1.0.
"""

from functools import cache
from typing import Literal

from marl_battlegrounds.evaluation.metric_catalog import (
    METRIC_COLUMNS,
    MetricColumn,
    MetricScope,
)

MAX_ALLIES_NEAR = 4
"""Most teammates that can be near an agent: a team has five slots."""

DEATH_WINDOWS = (5, 10, 20)
"""Carry lengths, in ticks, for the counts of carries ending in death within them."""

_PRIEST_CLASS_ID = 5
"""Core's PRIEST_CLASS_ID, written here because this module imports no JAX (a test
pins the two together)."""


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


# (pattern, label, description, direction), from the inventory's Tables 1 and 2.
_INVENTORY_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "agent_{i}_flag_steals",
        "Flag Steals",
        "How many times this agent picked up the enemy flag from the enemy flag home.",
        "Context dependent",
    ),
    (
        "team_{t}_flag_steals",
        "Flag Steals",
        "How many times agents on this team picked up the enemy flag from the "
        "enemy flag home.",
        "Context dependent",
    ),
    (
        "agent_{i}_dropped_enemy_flag_pickups",
        "Pickups of the Dropped Enemy Flag",
        "How many times this agent picked the enemy flag up off the ground, where "
        "it fell when a carrier on this agent's team died.",
        "Context dependent",
    ),
    (
        "team_{t}_dropped_enemy_flag_pickups",
        "Pickups of the Dropped Enemy Flag",
        "How many times agents on this team picked the enemy flag up off the "
        "ground after a carrier on this team died.",
        "Context dependent",
    ),
    (
        "team_{t}_flag_pickups",
        "Flag Pickups",
        "How many times agents on this team picked up the enemy flag in any way: "
        "steals from the enemy flag home plus pickups of the dropped enemy flag.",
        "Context dependent",
    ),
    (
        "agent_{i}_steps__carrying",
        "Ticks Carrying",
        "How many ticks this agent held the enemy flag at the start of the tick.",
        "Context dependent",
    ),
    (
        "team_{t}_steps_with_an_agent__carrying",
        "Ticks With an Agent Carrying",
        "How many ticks an agent of this team held the enemy flag at the start of "
        "the tick.",
        "Context dependent",
    ),
    (
        "agent_{i}_flag_captures",
        "Flag Captures",
        "How many times this agent brought the enemy flag to its own flag home "
        "while its own flag was at home, scoring one point.",
        "Higher is better",
    ),
    (
        "team_{t}_flag_captures",
        "Flag Captures",
        "How many flag captures agents on this team made. Points set when a "
        "scenario starts do not count.",
        "Higher is better",
    ),
    (
        "agent_{i}_deaths__carrying",
        "Deaths While Carrying",
        "How many times this agent died while it held the enemy flag at the start "
        "of the tick. Each such death drops the flag, unless the agent captured on "
        "that tick: a carrier that this tick's combat kills still captures.",
        "Context dependent",
    ),
    (
        "team_{t}_deaths__carrying",
        "Total Deaths While Carrying",
        "How many times agents on this team died while they held the enemy flag at "
        "the start of the tick.",
        "Context dependent",
    ),
    (
        "agent_{i}_carry_duration_mean",
        "Mean Carry Duration in Ticks",
        "The average number of ticks one of this agent's carries lasted, from its "
        "pickup (a steal or a pickup of the dropped flag) until it scored or died. "
        "A carry still going when the game ends counts with its ticks so far. "
        "Blank if it never picked the flag up.",
        "Context dependent",
    ),
    (
        "team_{t}_carry_duration_mean",
        "Mean Carry Duration in Ticks",
        "The average number of ticks one carry by this team lasted, from a pickup "
        "(a steal or a pickup of the dropped flag) until the carrier scored or "
        "died. A carry still going when the game ends counts with its ticks so "
        "far. Blank if the team never picked the flag up.",
        "Context dependent",
    ),
    (
        "team_{t}_flag_captures_per_steal",
        "Flag Captures per Steal",
        "This team's flag captures divided by its steals. Each steal starts a run "
        "(the enemy flag's trip from its home until it is captured or sent home), "
        "so this is the share of runs that scored. Blank if the team never stole "
        "the flag.",
        "Context dependent",
    ),
    (
        "agent_{i}_carry_steps_ending_in_a_capture",
        "Ticks on Carries Ending in a Capture",
        "How many ticks this agent held the enemy flag during carries that ended "
        "with this agent capturing it.",
        "Context dependent",
    ),
    (
        "team_{t}_carry_steps_ending_in_a_capture",
        "Ticks on Carries Ending in a Capture",
        "How many ticks an agent of this team held the enemy flag during carries "
        "that ended in a capture.",
        "Context dependent",
    ),
    (
        "agent_{i}_carry_steps_ending_in_death",
        "Ticks on Carries Ending in Death",
        "How many ticks this agent held the enemy flag during carries that ended "
        "with its death. Ticks of a carry still going when the game ends are not "
        "counted.",
        "Context dependent",
    ),
    (
        "team_{t}_carry_steps_ending_in_death",
        "Ticks on Carries Ending in Death",
        "How many ticks an agent of this team held the enemy flag during carries "
        "that ended with the carrier's death. Ticks of a carry still going when "
        "the game ends are not counted.",
        "Context dependent",
    ),
    (
        "agent_{i}_carry_distance",
        "Carry Distance",
        "How far this agent moved, in map units, on ticks it held the enemy flag "
        "at the start of the tick and did not die. Charge movement counts.",
        "Context dependent",
    ),
    (
        "team_{t}_carry_distance",
        "Carry Distance",
        "How far this team's carriers moved, in map units, on ticks they held the "
        "enemy flag at the start of the tick and did not die.",
        "Context dependent",
    ),
    (
        "agent_{i}_carry_speed_mean",
        "Mean Carry Speed",
        "This agent's average movement per tick, in map units per tick, over the "
        "ticks it held the enemy flag and did not die.",
        "Context dependent",
    ),
    (
        "team_{t}_carry_speed_mean",
        "Mean Carry Speed",
        "The average movement per tick of this team's carriers, in map units per "
        "tick, over the ticks a carrier held the enemy flag and did not die.",
        "Context dependent",
    ),
    (
        "agent_{i}_carry_steps_in_capture_range_with_the_ally_flag_away",
        "Carry Ticks in Capture Range With the Ally Flag Away",
        "How many ticks this agent held the enemy flag within the game's capture "
        "distance (0.75 map units by default) of its own flag home while its own "
        "flag was away, so it could not score.",
        "Context dependent",
    ),
    (
        "team_{t}_carry_steps_in_capture_range_with_the_ally_flag_away",
        "Carry Ticks in Capture Range With the Ally Flag Away",
        "How many ticks this team's carrier held the enemy flag within the game's "
        "capture distance (0.75 map units by default) of its own flag home while "
        "its own flag was away, so it could not score.",
        "Context dependent",
    ),
    (
        "agent_{i}_distance_to_the_ally_flag_home_mean__carrying",
        "Mean Distance to the Ally Flag Home While Carrying",
        "This agent's average straight-line distance, in map units, to its own "
        "flag home (where it scores), over the ticks it held the enemy flag. Blank "
        "if it never carried.",
        "Context dependent",
    ),
    (
        "team_{t}_distance_to_the_ally_flag_home_mean__carrying",
        "Mean Distance to the Ally Flag Home While Carrying",
        "The average straight-line distance, in map units, from this team's "
        "carrier to its own flag home, over all of the team's carry ticks taken "
        "together.",
        "Context dependent",
    ),
    (
        "agent_{i}_flag_captures_by_an_ally_after_it_dropped_the_flag",
        "Captures by an Ally After This Agent Dropped the Flag",
        "How many times a teammate captured the enemy flag after this agent had "
        "carried that same flag and dropped it by dying, without the flag going "
        "back to its home in between.",
        "Context dependent",
    ),
    (
        "agent_{i}_flag_captures_after_an_ally_dropped_the_flag",
        "Captures After an Ally Dropped the Flag",
        "How many times this agent captured the enemy flag after a teammate had "
        "carried that same flag and dropped it by dying, without the flag going "
        "back to its home in between.",
        "Context dependent",
    ),
    (
        "team_{t}_flag_captures_after_an_ally_dropped_the_flag",
        "Captures After an Ally Dropped the Flag",
        "How many of this team's flag captures came after another teammate had "
        "carried that same flag and dropped it by dying, without the flag going "
        "back to its home in between.",
        "Context dependent",
    ),
    (
        "agent_{i}_distance_to_the_ally_carrier_mean",
        "Mean Distance to the Ally Carrier",
        "This agent's average distance, in map units, to its team's carrier, over "
        "ticks its team had a carrier and this agent was alive without a spawn "
        "shield and not the carrier.",
        "Context dependent",
    ),
    (
        "agent_{i}_distance_to_the_ally_carrier_observations",
        "Distance to the Ally Carrier Measurements",
        "How many ticks this agent's distance to its team's carrier was measured: "
        "ticks its team had a carrier and this agent was alive without a spawn "
        "shield and not the carrier.",
        "Context dependent",
    ),
    (
        "agent_{i}_to_agent_{j}_distance_to_the_ally_carrier_mean",
        "Mean Distance to This Ally While It Carried",
        "Agent i's average distance, in map units, to teammate j, over the ticks j "
        "held the enemy flag and agent i was alive without a spawn shield. Blank "
        "when i is j or j never carried.",
        "Context dependent",
    ),
    (
        "agent_{i}_to_agent_{j}_distance_to_the_ally_carrier_observations",
        "Distance to This Ally Measurements While It Carried",
        "How many ticks agent i's distance to teammate j was measured while j held "
        "the enemy flag and agent i was alive without a spawn shield.",
        "Context dependent",
    ),
    (
        "team_{t}_distance_to_the_ally_carrier_mean",
        "Mean Distance to the Ally Carrier",
        "The average distance, in map units, from this team's agents to its "
        "carrier, over every agent's measured ticks taken together (ticks the team "
        "had a carrier and the agent was alive without a spawn shield and not the "
        "carrier).",
        "Context dependent",
    ),
    (
        "team_{t}_distance_to_the_ally_carrier_observations",
        "Distance to the Ally Carrier Measurements",
        "How many distances to its carrier were measured for this team's agents, "
        "one per agent per tick, added together.",
        "Context dependent",
    ),
    (
        "agent_{i}_distance_to_the_closest_ally_mean__carrying",
        "Mean Distance to the Closest Ally While Carrying",
        "This agent's average distance, in map units, to its closest teammate that "
        "was alive without a spawn shield, over the ticks it held the enemy flag "
        "and had at least one such teammate.",
        "Context dependent",
    ),
    (
        "agent_{i}_distance_to_the_closest_ally_observations__carrying",
        "Distance to the Closest Ally Measurements While Carrying",
        "How many ticks this agent's distance to its closest teammate was measured "
        "while it held the enemy flag: its carry ticks with at least one teammate "
        "alive without a spawn shield.",
        "Context dependent",
    ),
    (
        "team_{t}_distance_to_the_closest_ally_mean__carrying",
        "Mean Distance to the Closest Ally While Carrying",
        "The average distance, in map units, from this team's carrier to its "
        "closest teammate that was alive without a spawn shield, over all of the "
        "team's measured carry ticks taken together.",
        "Context dependent",
    ),
    (
        "team_{t}_distance_to_the_closest_ally_observations__carrying",
        "Distance to the Closest Ally Measurements While Carrying",
        "How many carry ticks this team's carrier had its distance to its closest "
        "teammate measured.",
        "Context dependent",
    ),
    (
        "agent_{i}_ally_carrier_captures__near_the_ally_carrier",
        "Ally Carrier Captures While This Agent Was Near It",
        "How many flag captures its team's carrier made while this agent was "
        "within 3 map units of it and in clear sight of it. On the capture tick, "
        "every teammate standing at its own flag home is near the carrier too.",
        "Context dependent",
    ),
    (
        "agent_{i}_ally_carrier_deaths__near_the_ally_carrier",
        "Ally Carrier Deaths While This Agent Was Near It",
        "How many times its team's carrier died while this agent was within 3 map "
        "units of it and in clear sight of it.",
        "Context dependent",
    ),
    (
        "agent_{i}_deaths__near_the_ally_carrier",
        "Deaths While Near the Ally Carrier",
        "How many times this agent died while it was within 3 map units of its "
        "team's carrier and in clear sight of it at the start of the tick.",
        "Context dependent",
    ),
    (
        "agent_{i}_steps_with_{n}_allies_near__carrying",
        "Ticks Carrying by Escort Size: Exactly 2 Allies Near",
        "How many ticks this agent held the enemy flag with exactly 2 (or the "
        "column's number of) teammates alive without a spawn shield within 3 map "
        "units of it and in clear sight of it.",
        "Context dependent",
    ),
    (
        "team_{t}_steps_with_{n}_allies_near__carrying",
        "Ticks Carrying by Escort Size: Exactly 0 Allies Near",
        "How many ticks this team's carrier held the enemy flag with exactly 0 (or "
        "the column's number of) teammates alive without a spawn shield within 3 "
        "map units of it and in clear sight of it.",
        "Context dependent",
    ),
    (
        "agent_{i}_flag_captures_with_{n}_allies_near__carrying",
        "Flag Captures by Escort Size: Exactly 2 Allies Near",
        "How many flag captures this agent made with exactly 2 (or the column's "
        "number of) teammates alive without a spawn shield within 3 map units of "
        "it and in clear sight of it. On the capture tick, every teammate standing "
        "at its own flag home counts as near.",
        "Context dependent",
    ),
    (
        "team_{t}_flag_captures_with_{n}_allies_near__carrying",
        "Flag Captures by Escort Size: Exactly 2 Allies Near",
        "How many flag captures this team made with exactly 2 (or the column's "
        "number of) teammates alive without a spawn shield within 3 map units of "
        "the carrier and in clear sight of it.",
        "Context dependent",
    ),
    (
        "agent_{i}_deaths_with_{n}_allies_near__carrying",
        "Deaths While Carrying by Escort Size: Exactly 0 Allies Near",
        "How many times this agent died while holding the enemy flag with exactly "
        "0 (or the column's number of) teammates alive without a spawn shield "
        "within 3 map units of it and in clear sight of it.",
        "Context dependent",
    ),
    (
        "team_{t}_deaths_with_{n}_allies_near__carrying",
        "Deaths While Carrying by Escort Size: Exactly 0 Allies Near",
        "How many times this team's carrier died with exactly 0 (or the column's "
        "number of) teammates alive without a spawn shield within 3 map units of "
        "it and in clear sight of it.",
        "Context dependent",
    ),
    (
        "agent_{i}_damage_done__attacking_enemies_near_the_ally_carrier",
        "Damage Done While Attacking Enemies Near the Ally Carrier",
        "How much damage this agent dealt to enemies that were within 3 map units "
        "of its team's carrier and in clear sight of it.",
        "Context dependent",
    ),
    (
        "agent_{i}_kill_contributions__attacking_enemies_near_the_ally_carrier",
        "Kill Contributions While Attacking Enemies Near the Ally Carrier",
        "How many kills this agent helped with of enemies that were within 3 map "
        "units of its team's carrier and in clear sight of it.",
        "Context dependent",
    ),
    (
        "agent_{i}_effective_healing_done_to_allies_who_are__carrying",
        "Effective Healing Done to Allies Who Are Carrying",
        "How much useful Priest healing this agent gave to teammates on ticks they "
        "held the enemy flag. Healing it gave itself is not counted. Read it "
        "beside its team's carry ticks. Blank for agents that are not Priests.",
        "Context dependent",
    ),
    (
        "agent_{i}_excess_healing_to_allies_who_are__carrying",
        "Excess Healing to Allies Who Are Carrying",
        "How much of this agent's Priest healing to teammates who held the enemy "
        "flag did not fit below their maximum health. Healing it gave itself is "
        "not counted. Blank for agents that are not Priests.",
        "Context dependent",
    ),
    (
        "team_{t}_effective_healing_done_to_allies_who_are__carrying",
        "Effective Healing Done to Allies Who Are Carrying",
        "How much useful Priest healing, and how much excess healing, this team's "
        "Priests gave to teammates on ticks those teammates held the enemy flag. "
        "Self-healing is not counted.",
        "Context dependent",
    ),
    (
        "team_{t}_excess_healing_to_allies_who_are__carrying",
        "Excess Healing to Allies Who Are Carrying",
        "How much useful Priest healing, and how much excess healing, this team's "
        "Priests gave to teammates on ticks those teammates held the enemy flag. "
        "Self-healing is not counted.",
        "Context dependent",
    ),
    (
        "agent_{i}_effective_healing_done_to_allies_during_ally_carries",
        "Effective Healing Done to Allies During Ally Carries",
        "How much useful Priest healing this agent gave to teammates, itself not "
        "included, on ticks its team had a carrier. Blank for agents that are not "
        "Priests.",
        "Context dependent",
    ),
    (
        "team_{t}_effective_healing_done_to_allies_during_ally_carries",
        "Effective Healing Done to Allies During Ally Carries",
        "How much useful Priest healing this team's Priests gave to teammates "
        "other than themselves on ticks the team had a carrier.",
        "Context dependent",
    ),
    (
        "agent_{i}_ally_carrier_healing_focus_fraction",
        "Share of Healing During Ally Carries That Went to the Carrier",
        "Of the useful Priest healing this agent gave to teammates on ticks its "
        "team had a carrier, the share that went to the carrier. Healing it gave "
        "itself is left out of both parts. Blank when it gave no such healing.",
        "Context dependent",
    ),
    (
        "team_{t}_ally_carrier_healing_focus_fraction",
        "Share of Healing During Ally Carries That Went to the Carrier",
        "Of the useful Priest healing this team gave to teammates on ticks it had "
        "a carrier, the share that went to the carrier. Self-healing is left out "
        "of both parts. Blank when there was no such healing.",
        "Context dependent",
    ),
    (
        "agent_{i}_to_agent_{j}_effective_healing_done_to_allies_who_are__carrying",
        "Effective Healing Done to This Ally While It Carried",
        "How much useful Priest healing agent i gave teammate j on ticks j held "
        "the enemy flag. Blank when i is j or agent i is not a Priest.",
        "Context dependent",
    ),
    (
        "agent_{i}_to_agent_{j}_excess_healing_to_allies_who_are__carrying",
        "Excess Healing to This Ally While It Carried",
        "How much of agent i's Priest healing to teammate j, on ticks j held the "
        "enemy flag, did not fit below j's maximum health. Blank when i is j or "
        "agent i is not a Priest.",
        "Context dependent",
    ),
    (
        "agent_{i}_effective_priest_healing_received__carrying",
        "Effective Priest Healing Received While Carrying",
        "How much useful Priest healing this agent received on ticks it held the "
        "enemy flag.",
        "Context dependent",
    ),
    (
        "agent_{i}_priest_freedom_applications_to_allies_who_are__carrying",
        "Priest Freedom Applications to Allies Who Are Carrying",
        "How many times this agent gave Freedom to a teammate who held the enemy "
        "flag. Blank for agents that are not Priests.",
        "Context dependent",
    ),
    (
        "agent_{i}_effective_priest_freedom_applications_to_allies_who_are__carrying",
        "Effective Priest Freedom Applications to Allies Who Are Carrying",
        "How many times this agent gave Freedom to a teammate who held the enemy flag "
        "and, on the next tick, when the Freedom acted, it raised that teammate's "
        "speed limit against a slow. Blank for agents that are not Priests.",
        "Context dependent",
    ),
    (
        "agent_{i}_freedom_protected_steps__carrying",
        "Ticks Freedom Beat a Slow While Carrying",
        "How many ticks, while it held the enemy flag, Freedom let this agent move "
        "faster despite a slow. When this agent is a Priest, its own Freedom counts.",
        "Context dependent",
    ),
    (
        "agent_{i}_freedom_eligible_steps__carrying",
        "Ticks With Freedom on This Agent While Carrying",
        "How many ticks, while it held the enemy flag, this agent had Freedom on it "
        "and was not stunned, slowed or not. When this agent is a Priest, its own "
        "Freedom counts.",
        "Context dependent",
    ),
    (
        "agent_{i}_freedom_protection_fraction__carrying",
        "Share of Ticks With Freedom That Beat a Slow While Carrying",
        "Of the ticks this agent had Freedom on it while it held the enemy flag, the "
        "share on which Freedom let it move faster despite a slow.",
        "Context dependent",
    ),
    (
        "agent_{i}_rescue_opportunities__carrying",
        "Chances to Save This Agent While Carrying",
        "How many times, while it held the enemy flag, this agent took damage that "
        "would have killed it and its team's Priests had enough healing ready to save "
        "it. When this agent is a Priest, its own healing counts.",
        "Context dependent",
    ),
    (
        "agent_{i}_rescues__carrying",
        "Times Rescued While Carrying",
        "How many times, while it held the enemy flag, Priest healing saved this agent "
        "from damage that would have killed it. When this agent is a Priest, its own "
        "healing counts.",
        "Context dependent",
    ),
    (
        "agent_{i}_rescue_contributions_to_allies_who_are__carrying",
        "Lethal Damage Rescue Contributions to Allies Who Are Carrying",
        "How many times this agent's useful Priest healing helped save a teammate who "
        "held the enemy flag from damage that would have killed it. Blank for agents "
        "that are not Priests.",
        "Context dependent",
    ),
)

_NEAR = "near_the_ally_carrier"

# The inventory gives the whole-game closest-ally row one text for its four columns,
# and the tick block one text for every situation, so these rows are written out for
# their columns here, in the inventory's words.
_CLOSEST_ALLY_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "agent_{i}_distance_to_the_closest_ally_mean",
        "Mean Distance to the Closest Ally",
        "Over the whole game: this agent's average distance, in map units, to the "
        "closest teammate alive without a spawn shield, on ticks it was alive "
        "without a spawn shield with at least one such teammate.",
        "Context dependent",
    ),
    (
        "agent_{i}_distance_to_the_closest_ally_observations",
        "Distance to the Closest Ally Measurements",
        "Over the whole game: how many ticks this agent's distance to its closest "
        "teammate was measured, the ticks it was alive without a spawn shield with "
        "at least one such teammate.",
        "Context dependent",
    ),
    (
        "team_{t}_distance_to_the_closest_ally_mean",
        "Mean Distance to the Closest Ally",
        "Over the whole game: the average distance, in map units, from this team's "
        "agents to their closest teammate alive without a spawn shield, over every "
        "agent's measured ticks taken together.",
        "Context dependent",
    ),
    (
        "team_{t}_distance_to_the_closest_ally_observations",
        "Distance to the Closest Ally Measurements",
        "Over the whole game: how many distances to the closest teammate were "
        "measured for this team's agents, one per agent per tick, added together.",
        "Context dependent",
    ),
)


def _tick_block_rows(
    slug: str, name: str, where: str, definition: str
) -> tuple[tuple[str, str, str, str], ...]:
    """Write a situation's tick block (the inventory's Table 5) for its columns.

    slug is the situation's slug, name its title ("Near the Ally Carrier"),
    where a phrase that finishes "was ..." ("near its team's carrier"), and
    definition the sentence that defines a tick in it.
    """
    rows = (
        (f"agent_{{i}}_steps__{slug}", f"Ticks {name}", definition),
        (
            f"agent_{{i}}_solo_steps__{slug}",
            f"Solo Ticks {name}",
            f"How many ticks this agent was the only agent of its team {where}. Only "
            "teammates are checked; enemies may be there.",
        ),
        (
            f"team_{{t}}_steps_with_an_agent__{slug}",
            f"Ticks With an Agent {name}",
            f"How many ticks at least one of the team's agents was {where}. A tick "
            "counts once, however many agents were.",
        ),
        (
            f"team_{{t}}_agent_steps__{slug}",
            f"Agent-Ticks {name}",
            f"The team's agents' ticks {where}, added together: 3 agents for 1 tick "
            "count 3.",
        ),
        (
            f"team_{{t}}_solo_steps__{slug}",
            f"Solo Ticks {name}",
            f"How many ticks exactly one of the team's agents was {where}: its "
            "agents' solo ticks added together.",
        ),
        (
            f"team_{{t}}_steps_with_two_or_more_agents__{slug}",
            f"Ticks With Two or More Agents {name}",
            f"How many ticks at least 2 of the team's agents were {where} together: "
            "ticks with an agent minus solo ticks.",
        ),
        (
            f"team_{{t}}_headcount_mean__{slug}",
            f"Mean Headcount When Any of Its Agents Was {name}",
            f"On ticks with at least one of the team's agents {where}, the average "
            "number of its agents that were: agent-ticks divided by ticks with an "
            "agent. Blank when none of its agents ever was.",
        ),
        (
            f"agent_{{i}}_step_participation__{slug}",
            f"This Agent's Share of Its Team's Ticks With an Agent {name}",
            f"Of the ticks at least one of the team's agents was {where}, the share "
            "that included this agent. The five agents' shares can add up to more "
            "than 100%. Blank only when no agent of its team, this one included, "
            "ever was.",
        ),
        (
            f"agent_{{i}}_solo_step_fraction__{slug}",
            f"Share of This Agent's Ticks {name} That Were Solo",
            f"This agent's solo ticks divided by its ticks {where}. Blank when it "
            "never was.",
        ),
    )
    return tuple((*row, "Context dependent") for row in rows)


def _death_window_rows() -> tuple[tuple[str, str, str, str], ...]:
    """Write the approved carries-ending-in-death-within rows."""
    rows: list[tuple[str, str, str, str]] = []
    for ticks in DEATH_WINDOWS:
        rows.append(
            (
                f"agent_{{i}}_carries_ending_in_death_within_{ticks}_steps",
                f"Carries Ending in Death Within {ticks} Ticks",
                f"How many of this agent's carries ended with its death within "
                f"{ticks} ticks of the pickup (a steal or a pickup of the dropped "
                "flag) that started them. A carrier killed on the tick it picked "
                "the flag up counts, with a carry of 0 ticks.",
                "Context dependent",
            )
        )
        rows.append(
            (
                f"team_{{t}}_carries_ending_in_death_within_{ticks}_steps",
                f"Carries Ending in Death Within {ticks} Ticks",
                "How many of this team's carries ended with the carrier's death "
                f"within {ticks} ticks of the pickup that started them. A carrier "
                "killed on the tick it picked the flag up counts, with a carry of 0 "
                "ticks.",
                "Context dependent",
            )
        )
    return tuple(rows)


# Carrying's tick block beyond its two stored counts (the inventory's Table 5, the
# Carrying row): a team has at most one carrier, so its solo ticks and team sums
# are views (never shipped) and these four keep their always-known values.
_CARRYING_BLOCK_ROWS = tuple(
    (*row, "Context dependent")
    for row in (
        (
            "team_{t}_steps_with_two_or_more_agents__carrying",
            "Ticks With Two or More Agents Carrying",
            "How many ticks at least 2 of the team's agents held the enemy flag "
            "together. Always blank, because a team has at most one carrier.",
        ),
        (
            "team_{t}_headcount_mean__carrying",
            "Mean Headcount When Any of Its Agents Was Carrying",
            "On ticks one of the team's agents held the enemy flag, the average "
            "number of its agents that did. Always 1, because a team has at most one "
            "carrier. Blank when none of its agents ever carried.",
        ),
        (
            "agent_{i}_step_participation__carrying",
            "This Agent's Share of Its Team's Ticks With an Agent Carrying",
            "Of the ticks one of the team's agents held the enemy flag, the share on "
            "which this agent held it, so it shows who carried. Blank only when no "
            "agent of its team, this one included, ever carried.",
        ),
        (
            "agent_{i}_solo_step_fraction__carrying",
            "Share of This Agent's Ticks Carrying That Were Solo",
            "This agent's solo ticks divided by its ticks holding the enemy flag. "
            "Always 1 when it carried, because a team has at most one carrier. Blank "
            "when it never carried.",
        ),
    )
)

# The inventory gives the team's three rescue rows one text; each gets its part.
_TEAM_RESCUE_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "team_{t}_rescue_opportunities__carrying",
        "Lethal Damage Rescue Opportunities While Carrying",
        "For this team's carriers: how many times damage would have killed the "
        "carrier while its Priests had enough healing ready. A carrying Priest's own "
        "healing counts.",
        "Context dependent",
    ),
    (
        "team_{t}_rescues__carrying",
        "Lethal Damage Rescues While Carrying",
        "For this team's carriers: how many times Priest healing saved the carrier "
        "from damage that would have killed it. A carrying Priest's own healing "
        "counts.",
        "Context dependent",
    ),
    (
        "team_{t}_rescue_rate__carrying",
        "Lethal Damage Rescue Rate While Carrying",
        "For this team's carriers: the rescues divided by the rescue opportunities. "
        "Blank when there was no rescue opportunity.",
        "Context dependent",
    ),
)

_WRITTEN_ROWS = (
    _CLOSEST_ALLY_ROWS
    + _CARRYING_BLOCK_ROWS
    + _tick_block_rows(
        _NEAR,
        "Near the Ally Carrier",
        "near its team's carrier",
        "How many ticks this agent was near its team's carrier: alive without a "
        "spawn shield, not the carrier itself, and within 3 map units of the "
        "carrier and in clear sight of it at the start of the tick.",
    )
    + _tick_block_rows(
        "attacking_enemies_near_the_ally_carrier",
        "Attacking Enemies Near the Ally Carrier",
        "attacking enemies near its team's carrier",
        "How many ticks this agent's Basic or Ultimate damaged an enemy that was "
        "within 3 map units of its team's carrier and in clear sight of it at the "
        "start of the tick. A Priest is in it too on a tick it effectively healed, "
        "or freed with last tick's Freedom, a teammate who was. The carrier itself "
        "never is; its own hits count while carrying.",
    )
    + _death_window_rows()
    + _TEAM_RESCUE_ROWS
)

_DIRECTIONS: dict[str, Literal["higher", "lower", "descriptive"]] = {
    "Context dependent": "descriptive",
    "Higher is better": "higher",
    "Lower is better": "lower",
}
# Columns whose combined value across games weighs each game by another column
# (MetricColumn.summary_weight), by stem, or by (scope, stem) where the scopes
# differ; any other mean's weight is its _observations twin. A weight starting
# "team:" is the agent's team's column. A mean whose count is a view (a
# copy, never shipped) is weighed by the shipped column the view copies, or by None,
# so its games count equally, when the view adds or subtracts columns.
_WEIGHTS: dict[str | tuple[MetricScope, str], str | None] = {
    "flag_captures_per_steal": "flag_steals",
    "ally_carrier_healing_focus_fraction": (
        "effective_healing_done_to_allies_during_ally_carries"
    ),
    "rescue_rate__carrying": "rescue_opportunities__carrying",
    "freedom_protection_fraction__carrying": "freedom_eligible_steps__carrying",
    ("agent", "carry_duration_mean"): None,
    ("team", "carry_duration_mean"): "flag_pickups",
    "carry_speed_mean": None,
    ("agent", "distance_to_the_ally_flag_home_mean__carrying"): "steps__carrying",
    ("team", "distance_to_the_ally_flag_home_mean__carrying"): (
        "steps_with_an_agent__carrying"
    ),
}
# The tick block's means and shares, by the part of the stem before "__".
_TICK_BLOCK_WEIGHTS = {
    "headcount_mean": "steps_with_an_agent",
    "step_participation": "team:steps_with_an_agent",
    "solo_step_fraction": "steps",
}


def _weight(stem: str, scope: MetricScope) -> str | None:
    """Name the stem whose value weighs this stem's games when they combine."""
    measure, _, situation = stem.partition("__")
    if (scope, stem) in _WEIGHTS:
        return _WEIGHTS[scope, stem]
    if stem in _WEIGHTS:
        return _WEIGHTS[stem]
    if measure in _TICK_BLOCK_WEIGHTS:
        return f"{_TICK_BLOCK_WEIGHTS[measure]}__{situation}"
    if "_mean" in stem:
        return stem.replace("_mean", "_observations")
    return None


def _unit(stem: str) -> str:
    """Name a stem's unit in the catalog's words."""
    if stem.endswith(("_fraction", "_participation")) or any(
        part in stem for part in ("_fraction__", "_participation__", "_rate__")
    ):
        return "fraction"
    if stem.endswith("_observations") or "_observations__" in stem:
        return "count"
    if "distance" in stem:
        return "distance"
    if "healing" in stem or "damage" in stem:
        return "health"
    if "_mean" in stem or stem.endswith("_per_steal"):
        return "ratio"
    if stem.startswith(
        (
            "steps",
            "solo_steps",
            "agent_steps",
            "carry_steps",
            "freedom_protected_steps",
            "freedom_eligible_steps",
        )
    ):
        return "agent_steps"
    return "count"


def _family(stem: str) -> str:
    """Name a stem's family: the carry (Table 1) or the escort (Table 2).

    Table 2 also holds the carrier's rescues and Freedom, and the Priests' part in
    them.
    """
    escort = ("ally_carrier", "closest_ally", "allies_near", _NEAR, "rescue", "freedom")
    return "ally_carrier_support" if any(w in stem for w in escort) else "flag_carry"


def _escort(text: str, n: int, *, label: bool) -> str:
    """Write one escort size into a split row's label or description."""
    if label:
        word = "Ally" if n == 1 else "Allies"
        for example in range(MAX_ALLIES_NEAR + 1):
            text = text.replace(
                f"Exactly {example} Allies Near", f"Exactly {n} {word} Near"
            )
        return text
    word = "teammate" if n == 1 else "teammates"
    for example in range(MAX_ALLIES_NEAR + 1):
        text = text.replace(
            f"exactly {example} (or the column's number of) teammates",
            f"exactly {n} {word}",
        )
    return text


_CARRIER_RESCUE_STEMS = (
    "rescue_opportunities__carrying",
    "rescues__carrying",
    "rescue_rate__carrying",
)
"""The team's rescue rows for its carriers, which need a Priest on the team."""


def _column(
    name: str,
    stem: str,
    scope: MetricScope,
    subjects: tuple[int, ...],
    label: str,
    description: str,
    direction: str,
) -> MetricColumn:
    """Build one column's record, with its missing rule and its summary weight."""
    prefix = name[: len(name) - len(stem)]
    weight = _weight(stem, scope)
    # A carrier's rescue chances need a Priest on its team, as in Team Deathmatch:
    # the team's three rescue rows and an agent's chances are blank without one.
    team_priest = (scope == "team" and stem in _CARRIER_RESCUE_STEMS) or (
        scope == "agent" and stem == "rescue_opportunities__carrying"
    )
    if scope == "team":
        who = f"nobody is playing on Team {'AB'[subjects[0] - 1]}"
    elif scope == "agent":
        who = "this agent was not included in this game"
    else:
        who = "either agent was not included in this game"
    if team_priest:
        letter = "AB"[subjects[0] - 1] if scope == "team" else "AB"[subjects[0] // 5]
        who += f"; or Team {letter} has no Priest in the game"
    return MetricColumn(
        name=name,
        label=label,
        family=_family(stem),
        unit=_unit(stem),
        scope=scope,
        subjects=subjects,
        description=description,
        missing_when=f"This measurement was not recorded; or {who}.",
        direction=_DIRECTIONS[direction],
        stem=stem,
        # Healing, rescues and Freedom given are a Priest's alone (the class rule);
        # a team's carrier rescue rows need a Priest on the team.
        required_class_id=_PRIEST_CLASS_ID
        if "healing_done_to_allies" in stem
        or "excess_healing_to_allies" in stem
        or "healing_focus" in stem
        or "rescue_contributions_to_allies" in stem
        or "freedom_applications_to_allies" in stem
        or (scope == "team" and team_priest)
        else None,
        summary_weight=None
        if weight is None
        else f"team_{'ab'[subjects[0] // 5]}_{weight.removeprefix('team:')}"
        if weight.startswith("team:")
        else prefix + weight,
    )


@cache
def capture_the_flag_columns() -> tuple[MetricColumn, ...]:
    """Return Capture the Flag's own columns, in the inventory's order.

    Returns
    -------
    tuple of MetricColumn
        One record per column, from each row of _INVENTORY_ROWS and
        _WRITTEN_ROWS expanded over its slots, teams, pairs and escort sizes.
        Pairs are ordered teammates, i to j, i equal to j included (always
        blank); their scope is source_recipient.
    """
    columns: list[MetricColumn] = []
    for pattern, label, description, direction in _INVENTORY_ROWS + _WRITTEN_ROWS:
        sizes = range(MAX_ALLIES_NEAR + 1) if "{n}" in pattern else (None,)
        for n in sizes:
            text = (
                pattern
                if n is None
                else pattern.replace("with_{n}_allies_near", escort_split(n))
            )
            name_label = label if n is None else _escort(label, n, label=True)
            meaning = description if n is None else _escort(description, n, label=False)
            if text.startswith("agent_{i}_to_agent_{j}_"):
                stem = text.removeprefix("agent_{i}_to_agent_{j}_")
                columns.extend(
                    _column(
                        f"agent_{i}_to_agent_{j}_{stem}",
                        stem,
                        "source_recipient",
                        (i, j),
                        name_label,
                        meaning,
                        direction,
                    )
                    for team in (0, 5)
                    for i in range(team, team + 5)
                    for j in range(team, team + 5)
                )
            elif text.startswith("agent_{i}_"):
                stem = text.removeprefix("agent_{i}_")
                columns.extend(
                    _column(
                        f"agent_{i}_{stem}",
                        stem,
                        "agent",
                        (i,),
                        name_label,
                        meaning,
                        direction,
                    )
                    for i in range(10)
                )
            else:
                stem = text.removeprefix("team_{t}_")
                columns.extend(
                    _column(
                        f"team_{t}_{stem}",
                        stem,
                        "team",
                        (k,),
                        name_label,
                        meaning,
                        direction,
                    )
                    for k, t in ((1, "a"), (2, "b"))
                )
    return tuple(columns)


@cache
def capture_the_flag_metric_columns() -> tuple[MetricColumn, ...]:
    """Return a Capture the Flag game's full column list.

    Returns
    -------
    tuple of MetricColumn
        Team Deathmatch's full columns without the Red Zone's (a Capture the
        Flag game has no Red Zone), in their order, then capture_the_flag_columns.
    """
    return (
        tuple(column for column in METRIC_COLUMNS if column.family != "red_zone")
        + capture_the_flag_columns()
    )
