"""Find the teamwork that separates strong tournament entrants from weak ones.

Run with XLA_FLAGS=--xla_gpu_deterministic_ops=true python -m
marl_battlegrounds.experiments.tournament_analysis tournament_dir=/path/saved-run
field_file=/path/field.json output_dir=/path/out.

A tournament saves priority metrics (results, points, kills and deaths) for
every game, which is enough to rate the entrants but not to see how they play.
This stage shows how to get the rest from the public API:

1. Read the saved tournament with marl_bgs.load_results.
2. Play its schedule again with marl_bgs.run_tournament, recording full
   metrics (all 11,192 measures) for every game through
   full_metrics_episodes, and check that every game repeats exactly.
3. Summarize every entrant over all of its games with one call to
   results.system_metrics, which reads each game from the side the entrant
   played, combines games by each column's catalog rule and gives 95%
   intervals from resampling whole game pairs.
4. Build class tables (who damaged, helped kill and healed whom, whose aura
   reached which ally, who got the Priest's Freedom, and how far apart each
   pair of classes stood) from one by="game" call, where agents 0-4 are
   always the entrant's team and each game's own classes count, so any roster
   works. The same rows check each matchup's point margin in the replayed
   full metrics against the points the original run saved (head_to_head), a
   check across two tables.
5. Rank every team-level metric in the catalog by its rank correlation with
   the saved Elo, with per-game totals taken per 100 ticks because games last
   different times, to find the measures that follow strength most closely,
   and test each one: an exact permutation p-value with Holm's and the
   Westfall-Young max-statistic corrections for testing every metric, whether
   the winner of a decided game had more of it than the loser, and, inside
   each matchup (one opponent, map and spawn end), whether an entrant had
   more of it in its wins than in its losses.

Outputs in output_dir: teamwork.csv (each MEASURES value with its interval),
by_class.csv (the class tables), trends.csv (each MEASURES value's rank
correlation with Elo and its exact p-value), elo_screen.csv (step 5, with its
tests), summary.md
(readable tables), teamwork.png (each teamwork measure against Elo) and
analysis.json (everything, plus the checks, whether the replay declared XLA's
deterministic mode, and the replay's and analysis's times). Give replay_dir to
redo steps 3-5 from an earlier replay without playing again.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import time
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
# isort: on

from marl_battlegrounds.evaluation.metric_catalog import METRIC_COLUMNS_BY_NAME
from marl_battlegrounds.evaluation.runtime_provenance import (
    declared_gpu_deterministic_ops,
    read_declared_xla_flags,
)

type Floats = NDArray[np.float64]
type Columns = Mapping[str, Any]

# Class IDs 1-5, in the simulator's order.
CLASSES = ("Mage", "Warrior", "Hunter", "Rogue", "Priest")

# Each measure: (name, full-metric column or ratio name, plain meaning). Names
# use Team A's columns because system_metrics shows every entrant as Team A.
# Counts, health and ticks are means per game; fractions and the distance
# combine by the catalog's summary weight (total numerator over total
# denominator, or a mean weighted by its observations).
MEASURES: tuple[tuple[str, str, str], ...] = (
    ("margin", "score_difference", "Points minus the opponent's, per game"),
    ("kills", "team_a_kills", "Kills per game"),
    ("deaths", "team_a_deaths", "Deaths per game"),
    (
        "focus_fire_ticks",
        "team_a_focus_fire_steps",
        "Ticks per game when two or more teammates dealt damage",
    ),
    (
        "shared_kill_fraction",
        "team_a_multi_contributor_kill_fraction",
        "Share of kills with two or more helpers on the killing tick",
    ),
    (
        "slowed_kill_fraction",
        "slowed_kill_fraction",
        "Share of kills on enemies a teammate Hunter had already slowed",
    ),
    (
        "ultimate_kill_fraction",
        "team_a_ultimate_kill_fraction",
        "Share of kills helped by an Ultimate used on the killing tick",
    ),
    (
        "ally_distance",
        "team_a_ally_distance_mean",
        "Average distance between living teammates, in map units",
    ),
    (
        "mage_aura_coverage",
        "team_a_mage_aura_coverage",
        "Share of teammate ticks spent inside a Mage aura",
    ),
    (
        "warrior_aura_coverage",
        "team_a_warrior_aura_coverage",
        "Share of teammate ticks spent inside a Warrior aura",
    ),
    (
        "mage_aura_damage",
        "team_a_damage_from_mage_aura",
        "Extra damage per game that Mage auras added to teammates' attacks",
    ),
    (
        "warrior_aura_protection",
        "team_a_damage_prevented_by_warrior_aura",
        "Damage per game that Warrior auras kept off teammates",
    ),
    (
        "useful_healing_fraction",
        "team_a_effective_healing_fraction",
        "Share of Priest healing that was useful, not excess",
    ),
    (
        "freedom_protection",
        "team_a_freedom_protection_fraction",
        "Share of ticks with Freedom on which it held up a slowed ally's speed",
    ),
    (
        "warrior_ultimate_kill_fraction",
        "team_a_warrior_ultimate_kill_fraction",
        "Share of kills a Warrior helped with its Ultimate on the killing tick",
    ),
    (
        "rescue_rate",
        "team_a_rescue_rate",
        "Share of chances to save a dying ally with Priest healing that were taken",
    ),
    (
        "own_red_zone_deaths",
        "team_a_red_zone_deaths",
        "Deaths per game inside the team's own Red Zone, worth 2 points each",
    ),
    ("basic_uses", "team_a_basic_activations", "Basic abilities used per game"),
    (
        "mage_aura_damage_share",
        "mage_aura_damage_share",
        "Share of the team's damage that Mage auras added",
    ),
    (
        "warrior_aura_protection_share",
        "warrior_aura_protection_share",
        "Damage Warrior auras prevented per point of damage the team took",
    ),
    ("kills_per_tick", "kills_per_tick", "Kills per tick"),
    ("ultimates_per_tick", "ultimates_per_tick", "Ultimate abilities used per tick"),
    (
        "slowed_damage_per_tick",
        "slowed_damage_per_tick",
        "Damage per tick to enemies a teammate Hunter had already slowed",
    ),
    (
        "focus_fire_per_tick",
        "focus_fire_per_tick",
        "Share of ticks when two or more teammates dealt damage",
    ),
    (
        "useful_basic_healing_fraction",
        "team_a_basic_effective_healing_fraction",
        "Share of the Priest's Basic healing that was useful, not excess",
    ),
    (
        "ultimate_uses",
        "team_a_ultimate_activations",
        "Ultimate abilities used per game",
    ),
)

# Measures without their own catalog column, each a total divided by a total
# over all of an entrant's games: kills on slowed enemies over all kills, the
# auras' payoff over the damage it belongs to (so longer games do not count
# more), and kills, Ultimates, damage to slowed enemies and ticks with two or
# more attackers per tick.
RATIOS = {
    "slowed_kill_fraction": (
        "team_a_kills_of_hunter_basic_slow_recipient",
        "team_a_kills",
    ),
    "mage_aura_damage_share": ("team_a_damage_from_mage_aura", "team_a_damage_done"),
    "warrior_aura_protection_share": (
        "team_a_damage_prevented_by_warrior_aura",
        "team_a_damage_received",
    ),
    "kills_per_tick": ("team_a_kills", "episode_length"),
    "ultimates_per_tick": ("team_a_ultimate_activations", "episode_length"),
    "slowed_damage_per_tick": (
        "team_a_damage_to_hunter_basic_slow_recipient",
        "episode_length",
    ),
    "focus_fire_per_tick": ("team_a_focus_fire_steps", "episode_length"),
}

# Measures that are game results; the figure shows only the teamwork after them.
RESULTS = ("margin", "kills", "deaths")

# Class tables: (name, plain meaning). Matrices have the entrant's class as
# rows and the enemy class as columns; vectors have one value per own class.
CLASS_TABLES = (
    ("damage_share", "Share of the team's damage, attacker class to enemy class"),
    ("kill_help", "Kill contributions per game, helper class to enemy class"),
    ("healing_share", "Share of the Priest's useful healing each ally class received"),
    ("damage_taken_share", "Share of the damage the team received, by class"),
    (
        "mage_aura_coverage_by_class",
        "Share of each ally class's ticks inside a Mage aura",
    ),
    (
        "warrior_aura_coverage_by_class",
        "Share of each ally class's ticks inside a Warrior aura",
    ),
    ("freedom_share", "Share of the Priest's Freedom uses each ally class received"),
    (
        "class_pair_distance",
        "Average distance between two living classes, in map units",
    ),
)

# Tables with one value per class: (table, the class the values come from;
# "Enemy" when they come from the opponents).
VECTOR_TABLES = (
    ("healing_share", "Priest"),
    ("damage_taken_share", "Enemy"),
    ("freedom_share", "Priest"),
    ("mage_aura_coverage_by_class", "Mage"),
    ("warrior_aura_coverage_by_class", "Warrior"),
)

# The aura tables: (table, class that gives the aura, catalog stem).
AURAS = (
    ("mage_aura_coverage_by_class", "Mage", "mage_aura"),
    ("warrior_aura_coverage_by_class", "Warrior", "warrior_aura"),
)

# Agent columns behind the class tables: the entrant's agents are 0-4 and its
# enemies 5-9 in every system_metrics row.
CLASS_COLUMNS = tuple(
    [
        f"agent_{source}_to_agent_{target}_{stem}"
        for source in range(5)
        for target in range(5, 10)
        for stem in ("damage_done", "kill_contributions")
    ]
    + [
        f"agent_{source}_to_agent_{ally}_effective_healing_done"
        for source in range(5)
        for ally in range(5)
    ]
    + [f"agent_{agent}_damage_received" for agent in range(5)]
    + [
        f"agent_{source}_to_agent_{ally}_{stem}_{part}"
        for _, _, stem in AURAS
        for part in ("covered_steps", "eligible_steps")
        for source in range(5)
        for ally in range(5)
        if source != ally
    ]
    + [
        f"agent_{source}_to_agent_{ally}_priest_freedom_applications"
        for source in range(5)
        for ally in range(5)
    ]
    + [
        f"agent_{first}_and_agent_{second}_ally_distance_{part}"
        for first in range(5)
        for second in range(first + 1, 5)
        for part in ("mean", "observations")
    ]
)

# Step 5's tests: a matchup cell (one entrant, opponent, map and spawn end)
# needs this many wins and as many losses before its wins and losses are
# compared; exact p-values enumerate every ordering of up to EXACT_ENTRANTS
# ratings, and larger fields draw PERMUTATION_DRAWS random orderings from
# PERMUTATION_SEED.
MIN_CELL_RESULTS = 2
EXACT_ENTRANTS = 9
PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 20261007

# Team-level metrics the Elo screen ranks: every Team A team column except the
# game result itself.
SCREEN_COLUMNS = tuple(
    name
    for name, column in METRIC_COLUMNS_BY_NAME.items()
    if column.scope == "team"
    and name.startswith("team_a_")
    and name.removeprefix("team_a_") not in ("win", "draw", "loss", "return", "score")
)


# Units the screen compares per 100 ticks when the catalog sums them per game
# (counts, health and ticks), because games last different times.
PER_TICK_UNITS = frozenset({"count", "health", "agent_steps", "steps", "pair_steps"})


def per_tick(column: str) -> bool:
    """Tell whether the screen compares a column per 100 ticks.

    Parameters
    ----------
    column : str
        A catalog column, such as ``team_a_kills``.

    Returns
    -------
    bool
        True for per-game totals: a unit in ``PER_TICK_UNITS``, no summary
        weight in the catalog (the summary is a mean per game) and no "mean"
        in the name (so averages such as a mean respawn wait keep their
        scale). False for fractions, distances and averages.

    Notes
    -----
    Learned Season 0 teams draw far more often than the scripted bots (20-29%
    of games against 2.5-4.8%), and draws run the full 300 ticks, so a per-game
    total partly measures game length.
    """
    info = METRIC_COLUMNS_BY_NAME[column]
    return (
        info.unit in PER_TICK_UNITS
        and info.summary_weight is None
        and "mean" not in column.removeprefix("team_a_")
    )


def summary_columns() -> list[str]:
    """List the full-metric columns one system_metrics call needs, once each.

    Returns
    -------
    list of str
        The ``MEASURES`` columns that are catalog columns, then
        ``SCREEN_COLUMNS`` and ``episode_length`` (for the screen's per-tick
        values), in that order without repeats. The ``RATIOS``
        measures are not listed: ``ratios=`` reads their input columns itself.
        The class tables read ``CLASS_COLUMNS`` per game instead.
    """
    names = [column for _, column, _ in MEASURES if column not in RATIOS]
    return list(dict.fromkeys([*names, *SCREEN_COLUMNS, "episode_length"]))


def compare_matches(
    saved: Mapping[str, Any], replayed: Mapping[str, Any]
) -> dict[str, Any]:
    """Count replayed games equal to the saved games in every match column.

    Parameters
    ----------
    saved, replayed : mapping of str to numpy.ndarray
        ``matches`` tables of the saved run and of its replay.

    Returns
    -------
    dict
        ``games`` (saved games), ``identical`` (games whose every column other
        than ``run_id`` matches exactly; two missing values count as equal),
        and ``different`` (logical game IDs that differ or were not replayed).
    """

    def rows(table: Mapping[str, Any]) -> dict[int, tuple[str, ...]]:
        """Map each logical game ID to its row's cells, run_id left out."""
        names = [name for name in table if name != "run_id"]
        return {
            int(table["episode_id"][index]): tuple(
                repr(table[name][index]) for name in names
            )
            for index in range(len(table["episode_id"]))
        }

    before, after = rows(saved), rows(replayed)
    different = sorted(game for game in before if before[game] != after.get(game))
    return {
        "games": len(before),
        "identical": len(before) - len(different),
        "different": different,
    }


def team_profiles(
    summary: Columns, names: Mapping[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """Turn a system_metrics summary into each entrant's measure profile.

    Parameters
    ----------
    summary : mapping of str to numpy.ndarray
        ``results.system_metrics("full_metrics", columns=summary_columns(),
        ratios=RATIOS)``: one row per entrant, every value from its own side.
    names : mapping of str to str or None, default=None
        Display name for each recorded ``system_name``; names left out are
        used as recorded.

    Returns
    -------
    dict of str to dict
        Display name to a profile: ``games`` (games with full metrics),
        ``independent_units`` (resampled pairs from matchups with random
        sampling; 0 when no matchup samples, None when sampling is unknown or
        cannot be checked), ``interval_status`` and ``unavailable_reason``
        (why an interval is withheld, or None), and for each ``MEASURES`` name
        a dict with ``value`` and its 95% interval ``low`` and ``high``
        (fractions as 0-1, distances in map units, health and counts per game;
        NaN where nothing counts or the interval is withheld).

    Notes
    -----
    Each profile copies the interval status ``system_metrics`` gives its row.
    A System of the researcher's own has unknown sampling, so its rows keep
    their intervals with the status "Conditional: Sampling not verified".
    Intervals are withheld only for the other reasons ``system_metrics``
    names, such as partial results.
    """
    names = names or {}
    profiles: dict[str, dict[str, Any]] = {}
    for row, recorded in enumerate(summary["system_name"]):
        units = summary["independent_units"][row]
        reason = summary["unavailable_reason"][row]
        profile: dict[str, Any] = {
            "games": int(summary["games"][row]),
            "independent_units": None if units is None else int(units),
            "interval_status": str(summary["interval_status"][row]),
            "unavailable_reason": None if reason is None else str(reason),
        }
        for measure, column, _ in MEASURES:
            profile[measure] = {
                "value": float(summary[column][row]),
                "low": float(summary[f"{column}_ci_low"][row]),
                "high": float(summary[f"{column}_ci_high"][row]),
            }
        profiles[names.get(str(recorded), str(recorded))] = profile
    return profiles


def class_tables(
    games: Columns, names: Mapping[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """Build each entrant's class tables from its games, each from its own side.

    Parameters
    ----------
    games : mapping of str to numpy.ndarray
        ``results.system_metrics("full_metrics", by="game",
        columns=CLASS_COLUMNS)``: one row per entrant per game, with that
        game's ``agent_<k>_class_id`` and ``agent_<k>_active`` from the
        entrant's side.
    names : mapping of str to str or None, default=None
        Display name for each recorded ``system_name``.

    Returns
    -------
    dict of str to dict
        Display name to ``damage_share`` and ``kill_help`` as 5 x 5 lists
        (the entrant's classes as rows, enemy classes as columns, in
        ``CLASSES`` order; kill help per game); ``healing_share``,
        ``damage_taken_share`` and ``freedom_share`` (the Priest's Freedom
        uses by ally class, the Priest itself included) as 5 values;
        ``mage_aura_coverage_by_class`` and ``warrior_aura_coverage_by_class`` as
        5 values, the
        share of each ally class's ticks inside that class's aura (covered
        ticks over ticks the aura could cover, never the aura giver itself);
        and ``class_pair_distance``, a symmetric 5 x 5 list of the mean distance in
        map units between two living teammates of those classes, weighted by
        the ticks both were alive. Shares and coverage are rounded to 4
        places, kill help and distance to 3; an empty share gives zeros, and
        coverage or distance with no ticks to count gives NaN (for example
        a roster without a Warrior, or one Mage and the Mage-Mage cell).

    Notes
    -----
    Agents 0-4 are the entrant's team and 5-9 its enemies in every row. Each
    game's own classes decide the cells, inactive agents are skipped and
    missing values add nothing, so any roster, fixed or not, works. Games with
    the same classes are summed together with NumPy.
    """
    names = names or {}
    systems = np.asarray(games["system_name"], dtype=object)
    classes = np.stack(
        [np.asarray(games[f"agent_{k}_class_id"], dtype=np.int64) for k in range(10)],
        axis=1,
    )
    active = np.stack(
        [np.asarray(games[f"agent_{k}_active"], dtype=bool) for k in range(10)],
        axis=1,
    )
    tables: dict[str, dict[str, Any]] = {}
    for recorded in dict.fromkeys(systems.tolist()):
        mine = np.flatnonzero(systems == recorded)
        damage = np.zeros((5, 5))
        kill_help = np.zeros((5, 5))
        healing = np.zeros(5)
        taken = np.zeros(5)
        freedom = np.zeros(5)
        auras = {table: np.zeros((2, 5)) for table, _, _ in AURAS}
        apart = np.zeros((2, 5, 5))
        rosters: dict[tuple[tuple[int, ...], tuple[bool, ...]], list[int]] = {}
        for row in mine:
            key = (tuple(classes[row].tolist()), tuple(active[row].tolist()))
            rosters.setdefault(key, []).append(int(row))
        for (kinds, alive), rows in rosters.items():

            def total(name: str, rows: list[int] = rows) -> float:
                """Sum one column over these games; missing values add nothing."""
                return float(np.nansum(np.asarray(games[name], np.float64)[rows]))

            def weighted(stem: str, rows: list[int] = rows) -> tuple[float, float]:
                """Sum mean times observations, and observations, over games."""
                mean = np.asarray(games[f"{stem}_mean"], np.float64)[rows]
                count = np.asarray(games[f"{stem}_observations"], np.float64)[rows]
                kept = np.isfinite(mean) & np.isfinite(count)
                return float((mean * count)[kept].sum()), float(count[kept].sum())

            kind = {
                agent: kinds[agent] - 1
                for agent in range(10)
                if alive[agent] and kinds[agent] > 0
            }
            own = [agent for agent in kind if agent < 5]
            enemy = [agent for agent in kind if agent >= 5]
            for source in own:
                taken[kind[source]] += total(f"agent_{source}_damage_received")
                for target in enemy:
                    pair = f"agent_{source}_to_agent_{target}"
                    damage[kind[source], kind[target]] += total(f"{pair}_damage_done")
                    kill_help[kind[source], kind[target]] += total(
                        f"{pair}_kill_contributions"
                    )
                if kind[source] == CLASSES.index("Priest"):
                    for ally in own:
                        healing[kind[ally]] += total(
                            f"agent_{source}_to_agent_{ally}_effective_healing_done"
                        )
                        freedom[kind[ally]] += total(
                            f"agent_{source}_to_agent_{ally}_priest_freedom_applications"
                        )
                for table, giver, stem in AURAS:
                    if kind[source] != CLASSES.index(giver):
                        continue
                    for ally in own:
                        if ally != source:
                            pair = f"agent_{source}_to_agent_{ally}_{stem}"
                            auras[table][0, kind[ally]] += total(
                                f"{pair}_covered_steps"
                            )
                            auras[table][1, kind[ally]] += total(
                                f"{pair}_eligible_steps"
                            )
                for other in own:
                    if other > source:
                        summed, count = weighted(
                            f"agent_{source}_and_agent_{other}_ally_distance"
                        )
                        for row, column in {
                            (kind[source], kind[other]),
                            (kind[other], kind[source]),
                        }:
                            apart[0, row, column] += summed
                            apart[1, row, column] += count
        tables[names.get(str(recorded), str(recorded))] = {
            "damage_share": _share(damage),
            "kill_help": (kill_help / max(len(mine), 1)).round(3).tolist(),
            "healing_share": _share(healing),
            "damage_taken_share": _share(taken),
            "freedom_share": _share(freedom),
            **{
                table: _ratio(covered, eligible, 4)
                for table, (covered, eligible) in auras.items()
            },
            "class_pair_distance": _ratio(apart[0], apart[1], 3),
        }
    return tables


def check_sides(games: Columns, head_to_head: Columns) -> float:
    """Check each matchup's replayed point margin against the saved results.

    Parameters
    ----------
    games : mapping of str to numpy.ndarray
        ``results.system_metrics("full_metrics", by="game", ...)`` of the
        replay with ``score_difference`` among its columns: one row per entrant
        per game, read from the entrant's side, with ``system_name`` and
        ``opponent_name``.
    head_to_head : mapping of str to numpy.ndarray
        ``marl_bgs.load_results(run).head_to_head()`` of the saved tournament,
        with ``system_name``, ``opponent_name`` and ``mean_point_margin``.

    Returns
    -------
    float
        The largest difference, in points per game, between a matchup's mean
        ``score_difference`` over its games and its saved ``mean_point_margin``.

    Raises
    ------
    ValueError
        The two list different matchups, or a difference is above 1e-6
        points: the replay's games differ from the saved ones, or a side was
        mixed up.
    """
    totals: dict[tuple[str, str], list[float]] = {}
    for system, opponent, value in zip(
        games["system_name"],
        games["opponent_name"],
        games["score_difference"],
        strict=True,
    ):
        total = totals.setdefault((str(system), str(opponent)), [0.0, 0.0])
        total[0] += float(value)
        total[1] += 1.0
    saved = {
        (str(system), str(opponent)): float(margin)
        for system, opponent, margin in zip(
            head_to_head["system_name"],
            head_to_head["opponent_name"],
            head_to_head["mean_point_margin"],
            strict=True,
        )
    }
    if totals.keys() != saved.keys():
        raise ValueError("the replay and head_to_head list different matchups")
    largest = max(
        abs(points / count - saved[key]) for key, (points, count) in totals.items()
    )
    if largest > 1e-6:
        raise ValueError(f"margins differ from head_to_head by up to {largest}")
    return largest


def elo_correlations(
    profiles: Mapping[str, dict[str, Any]], elo: Mapping[str, float]
) -> dict[str, float]:
    """Rank correlation of each measure with Elo across the entrants.

    Parameters
    ----------
    profiles : mapping of str to dict
        Display name to a ``team_profiles`` profile.
    elo : mapping of str to float
        Display name to the tournament's Elo rating.

    Returns
    -------
    dict of str to float
        Each ``MEASURES`` name to Spearman's rank correlation between the
        entrants' values and their ratings, from -1 (the measure falls as Elo
        rises) to 1 (it rises with Elo). Tied values share their average rank.
        NaN when fewer than three entrants have a value or all values are equal.

    Notes
    -----
    With a field of nine, a correlation describes this field only; it is not
    a test that the measure causes strength.
    """
    return {
        measure: _rank_correlation(
            {name: profile[measure]["value"] for name, profile in profiles.items()},
            elo,
            list(profiles),
        )
        for measure, _, _ in MEASURES
    }


def screen_measures(
    summary: Columns,
    elo: Mapping[str, float],
    names: Mapping[str, str] | None = None,
    scripted: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Rank every team-level full metric by how closely it follows Elo.

    Parameters
    ----------
    summary : mapping of str to numpy.ndarray
        The ``system_metrics`` summary that ``team_profiles`` also reads; it
        must hold ``SCREEN_COLUMNS`` and ``episode_length``.
    elo : mapping of str to float
        Display name to the saved tournament's Elo rating, for every entrant.
    names : mapping of str to str or None, default=None
        Display name for each recorded ``system_name``.
    scripted : collection of str, default=()
        Display names of scripted entrants; ``correlation_learned`` leaves
        them out, so a measure that only separates scripted from learned
        entrants shows a weak learned correlation.

    Returns
    -------
    list of dict
        One row per ``SCREEN_COLUMNS`` metric: ``measure`` (the name after
        ``team_a_``), ``unit``, ``scale`` ("per 100 ticks" for ``per_tick``
        columns, else "as summarized"), ``description`` (the catalog's
        description for one team), ``correlation_all`` and
        ``correlation_learned`` (rank correlations with Elo, as in
        ``elo_correlations``, over all entrants and over those not in
        ``scripted``) and each entrant's value on that scale under its display
        name. Rows are sorted from the strongest correlation over all
        entrants, rising or falling, down.

    Notes
    -----
    Values are the summary's own, combined by each column's catalog rule:
    counts are means per game, fractions with a saved denominator are total
    numerator over total denominator, and the few without one are means per
    game (``metric_catalog.SUMMARY_WEIGHT_MISSING``). Per-game totals are then
    divided by the entrant's mean game length and multiplied by 100. With about
    145 metrics and a field of nine, some strong correlations appear by chance;
    read a trend together with its learned correlation and meaning.
    """
    names = names or {}
    entrants = [names.get(str(name), str(name)) for name in summary["system_name"]]
    learned = [name for name in entrants if name not in scripted]
    length = np.asarray(summary["episode_length"], np.float64)
    rows: list[dict[str, Any]] = []
    for name in SCREEN_COLUMNS:
        column = METRIC_COLUMNS_BY_NAME[name]
        scale = 100.0 / length if per_tick(name) else np.ones_like(length)
        values = {
            entrant: float(value)
            for entrant, value in zip(
                entrants, np.asarray(summary[name], np.float64) * scale, strict=True
            )
        }
        description = column.description.replace("Team A's", "the team's")
        row: dict[str, Any] = {
            "measure": name.removeprefix("team_a_"),
            "unit": column.unit,
            "scale": "per 100 ticks" if per_tick(name) else "as summarized",
            "description": description.replace("Team A", "the team"),
            "correlation_all": _rank_correlation(values, elo, entrants),
            "correlation_learned": _rank_correlation(values, elo, learned),
        }
        rows.append(row | values)
    return sorted(
        rows, key=lambda row: -np.nan_to_num(abs(row["correlation_all"]), nan=-1.0)
    )


def screen_tests(
    screen: Sequence[Mapping[str, Any]],
    games: Columns,
    elo: Mapping[str, float],
    names: Mapping[str, str] | None = None,
    scripted: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Test each screened metric's trend with Elo, across teams and inside matchups.

    Parameters
    ----------
    screen : sequence of mapping
        ``screen_measures`` rows, each with every entrant's value under its
        display name.
    games : mapping of str to numpy.ndarray
        ``results.system_metrics("full_metrics", by="game", ...)`` with
        ``team_a_win``, ``team_a_loss``, ``episode_length`` and every screened
        ``team_a_<measure>`` among its columns: one row per entrant per game,
        from the entrant's side, with ``system_name``, ``opponent_name``,
        ``map_id``, ``spawn_end`` and the game's ``run_id``, ``phase``,
        ``pass_id`` and ``episode_id``.
    elo : mapping of str to float
        Display name to the saved tournament's Elo rating.
    names : mapping of str to str or None, default=None
        Display name for each recorded ``system_name``.
    scripted : collection of str, default=()
        Display names left out of ``p_value_learned``.

    Returns
    -------
    list of dict
        Copies of the screen rows, in the same order, each with these values:

        - ``p_value_all`` and ``p_value_learned``: the two-sided permutation
          p-value of ``correlation_all`` and ``correlation_learned``, the share
          of orderings of the ratings whose rank correlation with the values is
          at least as far from zero. With nine entrants the smallest possible
          value is 2 / 9! (about 0.0000055); with six, 2 / 720 (about 0.0028).
        - ``holm_p_value_all``: ``p_value_all`` after Holm's correction over
          every screened metric with a value.
        - ``max_t_p_value_all``: the Westfall-Young max-statistic correction
          over the same metrics: the share of orderings of the ratings in which
          the largest rank correlation of any metric is at least as far from
          zero as this one's. It allows for metrics that move together, so it
          is less strict than Holm when they overlap, and still controls the
          chance of any false trend. Only metrics with a value for every
          entrant enter it; the others get NaN.
        - ``winner_higher_share``: over decided games between two different
          entrants, the share in which the winner had more of the measure than
          the loser. Ties and missing values are left out;
          ``decided_games_compared`` counts the rest.
        - ``wins_over_losses``: inside each matchup cell (one entrant against
          one opponent on one map from one spawn end) with at least
          ``MIN_CELL_RESULTS`` wins and as many losses, the chance that one of
          its wins, picked at random, shows more of the measure than one of its
          losses (ties count half); the average over cells, weighted by wins
          times losses. Holding the opponent fixed matters: a team's wins come
          mostly against weak opponents, so a pooled comparison would compare
          opponents, not results. 0.5 means the measure does not change with
          the result inside a matchup; NaN if no cell qualifies.

    Notes
    -----
    Values follow ``screen_measures``: per-game totals are taken per 100 ticks
    (``per_tick``), here game by game. Exact p-values count every ordering of
    up to ``EXACT_ENTRANTS`` ratings; larger fields use ``PERMUTATION_DRAWS``
    random orderings from ``PERMUTATION_SEED``, counted as (hits + 1) / (draws
    + 1), so their p-values are estimates. Ties share their average rank, values
    are compared to 7 significant digits, and a metric whose values agree to
    within one part in a million counts as constant (no correlation). The
    permutation test treats each entrant as one observation, so a field of
    nine supports only strong trends, and near-copies (Season 0's three
    scripted bots) make the field effectively smaller. Matchups between
    methods that never sample (scripted bots, greedy QMIX or PQN-VDN) repeat
    the same few games, which count once per copy in both game checks. A
    measure that is itself part of winning (kills, deaths) passes the game
    checks trivially. Host-only NumPy.
    """
    names = names or {}
    systems = np.asarray(
        [names.get(str(name), str(name)) for name in games["system_name"]], object
    )
    win = np.asarray(games["team_a_win"], np.float64)
    loss = np.asarray(games["team_a_loss"], np.float64)
    length = np.asarray(games["episode_length"], np.float64)
    sides: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for row, key in enumerate(
        zip(
            games["run_id"],
            games["phase"],
            games["pass_id"],
            games["episode_id"],
            strict=True,
        )
    ):
        sides[tuple(str(part) for part in key)].append(row)
    decided = np.asarray(
        [
            (first, second) if win[first] == 1 else (second, first)
            for first, second in (rows for rows in sides.values() if len(rows) == 2)
            if win[first] + win[second] == 1
        ],
        np.int64,
    ).reshape(-1, 2)
    cells: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for row, key in enumerate(
        zip(
            systems,
            games["opponent_name"],
            games["map_id"],
            games["spawn_end"],
            strict=True,
        )
    ):
        cells[tuple(str(part) for part in key)].append(row)
    entrants = [name for name in screen[0] if name in elo] if screen else []
    learned = [name for name in entrants if name not in scripted]
    rows: list[dict[str, Any]] = []
    for source in screen:
        row = dict(source)
        for label, group in (("all", entrants), ("learned", learned)):
            row[f"p_value_{label}"] = _permutation_p(
                [float(row[name]) for name in group], [elo[name] for name in group]
            )
        column = f"team_a_{row['measure']}"
        value = np.asarray(games[column], np.float64)
        if per_tick(column):
            value = value / length * 100.0
        winner, loser = value[decided[:, 0]], value[decided[:, 1]]
        compared = np.isfinite(winner) & np.isfinite(loser) & (winner != loser)
        row["winner_higher_share"] = (
            float(np.mean(winner[compared] > loser[compared]))
            if compared.any()
            else math.nan
        )
        row["decided_games_compared"] = int(compared.sum())
        weighted, weights = 0.0, 0.0
        for members in cells.values():
            indices = np.asarray(members)
            won = value[indices[win[indices] == 1]]
            lost = value[indices[loss[indices] == 1]]
            won, lost = won[np.isfinite(won)], lost[np.isfinite(lost)]
            if len(won) >= MIN_CELL_RESULTS and len(lost) >= MIN_CELL_RESULTS:
                ranks = _ranks(np.concatenate([won, lost]))
                above = float(ranks[: len(won)].sum()) - len(won) * (len(won) + 1) / 2
                weighted += above
                weights += len(won) * len(lost)
        row["wins_over_losses"] = weighted / weights if weights else math.nan
        rows.append(row)
    for row, adjusted in zip(
        rows, _holm([row["p_value_all"] for row in rows]), strict=True
    ):
        row["holm_p_value_all"] = adjusted
    adjusted = _max_t([[float(row[name]) for name in entrants] for row in rows], [
        elo[name] for name in entrants
    ])  # fmt: skip
    for row, value in zip(rows, adjusted, strict=True):
        row["max_t_p_value_all"] = value
    return rows


def write_outputs(
    profiles: Mapping[str, dict[str, Any]],
    elo: Mapping[str, float],
    output_dir: Path,
    screen: Sequence[Mapping[str, Any]] = (),
    scripted: Collection[str] = (),
) -> None:
    """Write teamwork.csv, by_class.csv, trends.csv, elo_screen.csv, summary.md, PNG.

    Parameters
    ----------
    profiles : mapping of str to dict
        Display name to a ``team_profiles`` profile joined with its
        ``class_tables`` entry.
    elo : mapping of str to float
        Display name to the saved tournament's Elo rating; tables list the
        entrants from the highest rating down.
    output_dir : pathlib.Path
        Existing folder; files there with these names are replaced.
    screen : sequence of mapping, default=()
        ``screen_measures`` rows, optionally with ``screen_tests``' columns.
        When given, they are written to elo_screen.csv and the 20 strongest
        trends are added to summary.md, with the tests when present.
    scripted : collection of str, default=()
        Display names drawn as open points in teamwork.png; the others are
        filled.

    Notes
    -----
    trends.csv gives, for each ``MEASURES`` name, its ``elo_correlations``
    value (``correlation``), the exact two-sided permutation p-value of that
    correlation over every ordering of the entrants (``p_value``, as for the
    screen) and how many entrants have a value (``entrants``); NaN as for
    ``elo_correlations``. teamwork.png plots every ``MEASURES`` value after
    the game results against Elo, with its 95% interval, and gives each panel's rank
    correlation in its title. A percentile interval can lie wholly above or
    below its value; that side's bar is then drawn with zero length. It needs
    matplotlib (the viz extra).
    """
    order = sorted(profiles, key=lambda name: -elo[name])
    correlations = elo_correlations(profiles, elo)
    with (output_dir / "teamwork.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = "system elo games independent_units interval_status measure"
        writer.writerow([*header.split(), "value", "low", "high"])
        for name in order:
            profile = profiles[name]
            counts = (
                profile["games"],
                profile["independent_units"],
                profile["interval_status"],
            )
            for measure, _, _ in MEASURES:
                cell = profile[measure]
                bounds = (round(cell[key], 4) for key in ("value", "low", "high"))
                writer.writerow((name, round(elo[name], 1), *counts, measure, *bounds))
    with (output_dir / "by_class.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["system", "table", "source_class", "target_class", "value"])
        for name in order:
            profile = profiles[name]
            for table in ("damage_share", "kill_help", "class_pair_distance"):
                for row, source in enumerate(CLASSES):
                    for column, target in enumerate(CLASSES):
                        cell = profile[table][row][column]
                        writer.writerow([name, table, source, target, cell])
            for table, source in VECTOR_TABLES:
                for column, target in enumerate(CLASSES):
                    writer.writerow(
                        [name, table, source, target, profile[table][column]]
                    )
    if screen:
        with (output_dir / "elo_screen.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(screen[0]))
            writer.writeheader()
            for row in screen:
                # p-values keep 4 significant digits; other values 4 places.
                writer.writerow(
                    {
                        k: (float(f"{v:.4g}") if "p_value" in k else round(v, 4))
                        if isinstance(v, float)
                        else v
                        for k, v in row.items()
                    }
                )

    with (output_dir / "trends.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["measure", "correlation", "p_value", "entrants"])
        for measure, _, _ in MEASURES:
            values = [float(profiles[name][measure]["value"]) for name in order]
            ratings = [elo[name] for name in order]
            p_value = _permutation_p(values, ratings)
            writer.writerow(
                [
                    measure,
                    round(correlations[measure], 4),
                    float(f"{p_value:.4g}"),
                    sum(not math.isnan(value) for value in values),
                ]
            )

    lines = ["# Teamwork of Every Entrant", ""]
    lines.append(
        "All games of each entrant, from its own side; 95% intervals from "
        "resampling whole game pairs within each opponent and map, where "
        "matchups that repeat exactly add no width. A withheld interval shows "
        "nan; teamwork.csv gives each entrant's interval status."
    )
    lines += ["", "| Measure | Meaning | Rank correlation with Elo | "]
    lines[-1] += " | ".join(order) + " |"
    lines.append("| --- | --- | ---: |" + " ---: |" * len(order))
    lines.append(
        "| elo | Saved tournament rating | +1.00 | "
        + " | ".join(f"{elo[name]:,.0f}" for name in order)
        + " |"
    )
    for measure, _, meaning in MEASURES:
        cells = [
            "{value:,.3f} ({low:,.3f}-{high:,.3f})".format(**profiles[name][measure])
            for name in order
        ]
        lines.append(
            f"| {measure} | {meaning} | {correlations[measure]:+.2f} | "
            + " | ".join(cells)
            + " |"
        )
    meanings = dict(CLASS_TABLES)
    for table, _ in VECTOR_TABLES:
        meaning = meanings[table]
        lines += ["", f"**{meaning}** ({table})", ""]
        lines += ["| System | " + " | ".join(CLASSES) + " |", "| --- |" + " ---: |" * 5]
        for name in order:
            values = " | ".join(f"{value:.3f}" for value in profiles[name][table])
            lines.append(f"| {name} | {values} |")
    pairs = [(first, second) for first in range(5) for second in range(first + 1, 5)]
    lines += ["", f"**{meanings['class_pair_distance']}** (class_pair_distance)", ""]
    lines.append(
        "| System | "
        + " | ".join(f"{CLASSES[a]}-{CLASSES[b]}" for a, b in pairs)
        + " |"
    )
    lines.append("| --- |" + " ---: |" * len(pairs))
    for name in order:
        distance = profiles[name]["class_pair_distance"]
        cells = " | ".join(f"{distance[a][b]:.2f}" for a, b in pairs)
        lines.append(f"| {name} | {cells} |")
    if screen:
        tested = "p_value_all" in screen[0]
        lines += ["", "**Strongest trends with Elo among all team metrics**", ""]
        if tested:
            lines.append(
                "Per-game totals are per 100 ticks (Scale). p: exact permutation "
                "p-value over all entrants; Holm and Max-t: after Holm's and the "
                "Westfall-Young max-statistic corrections over the screen; Winner "
                "higher: share of decided games the winner had more; In matchup: "
                "chance a win shows more than a loss against the same opponent, "
                "map and spawn end (0.5: no change)."
            )
            lines.append("")
            lines.append(
                "| Measure | Scale | All | p | Holm | Max-t | Learned | p "
                "| Winner higher | In matchup | Meaning |"
            )
            lines.append("| --- | --- |" + " ---: |" * 8 + " --- |")
        else:
            lines.append("| Measure | Unit | All | Learned | Meaning |")
            lines.append("| --- | --- | ---: | ---: | --- |")
        for row in screen[:20]:
            cells = [
                row["measure"],
                row.get("scale", row["unit"]),
                f"{row['correlation_all']:+.2f}",
            ]
            if tested:
                cells += [
                    f"{row['p_value_all']:.2g}",
                    f"{row['holm_p_value_all']:.2g}",
                    f"{row['max_t_p_value_all']:.2g}",
                    f"{row['correlation_learned']:+.2f}",
                    f"{row['p_value_learned']:.2g}",
                    f"{row['winner_higher_share']:.2f}",
                    f"{row['wins_over_losses']:.2f}",
                ]
            else:
                cells.append(f"{row['correlation_learned']:+.2f}")
            lines.append("| " + " | ".join([*cells, row["description"]]) + " |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    from marl_battlegrounds.training.analysis import (
        _prepare_plot_backend,  # pyright: ignore[reportPrivateUsage]
    )

    _prepare_plot_backend()
    import matplotlib.pyplot as plt

    shown = [measure for measure in MEASURES if measure[0] not in RESULTS]
    columns = 5
    rows = -(-len(shown) // columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 2.7 * rows), constrained_layout=True
    )
    for axis, (measure, _, _) in zip(axes.flat, shown, strict=False):
        for name in order:
            cell = profiles[name][measure]
            axis.errorbar(
                elo[name],
                cell["value"],
                # A percentile interval can miss its own value; draw no bar then.
                yerr=[
                    [max(cell["value"] - cell["low"], 0.0)],
                    [max(cell["high"] - cell["value"], 0.0)],
                ],
                fmt="o",
                color="0.2",
                markerfacecolor="white" if name in scripted else "0.2",
                markersize=4.5,
                capsize=2,
            )
        axis.set_title(f"{measure} ({correlations[measure]:+.2f})", fontsize=9)
        axis.set_xlabel("Elo", fontsize=8)
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(shown) :]:
        axis.set_visible(False)
    figure.savefig(output_dir / "teamwork.png", dpi=150)
    plt.close(figure)


def _share(values: Floats) -> list[Any]:
    """Each value's share of the total, rounded to 4 places; zeros if empty."""
    return (values / max(float(values.sum()), 1e-9)).round(4).tolist()


def _ratio(top: Floats, bottom: Floats, places: int) -> list[Any]:
    """Cell-by-cell top over bottom, rounded; NaN where bottom is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bottom > 0, top / np.where(bottom > 0, bottom, 1.0), np.nan)
    return ratio.round(places).tolist()


def _rank_correlation(
    values: Mapping[str, float], elo: Mapping[str, float], names: Sequence[str]
) -> float:
    """Spearman correlation of values with Elo over the named entrants.

    Entrants whose value or rating is NaN are left out, such as every entrant
    outside a complete tournament. NaN when fewer than three remain or the
    values or ratings are all equal (``_constant``). Tied values share their
    average rank.
    """
    kept = [
        name for name in names if not np.isnan(values[name]) and not np.isnan(elo[name])
    ]
    measure = np.array([values[name] for name in kept])
    ratings = np.array([elo[name] for name in kept])
    if len(kept) < 3 or _constant(measure) or _constant(ratings):
        return float("nan")
    return float(np.corrcoef(_ranks(_rounded(measure)), _ranks(ratings))[0, 1])


def _constant(values: Floats) -> bool:
    """True when the values are equal to within one part in a million."""
    return bool(np.ptp(values) <= 1e-6 * max(1.0, float(np.max(np.abs(values)))))


def _rounded(values: Floats) -> Floats:
    """Round to 7 significant digits, so float noise does not break ties."""
    return np.asarray([float(f"{value:.7g}") for value in values], np.float64)


def _ranks(values: Floats) -> Floats:
    """Ranks from 1 upward; tied values share their average rank (no NaN)."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    return (upper - (counts - 1) / 2)[inverse].astype(np.float64)


@cache
def _orderings(count: int) -> NDArray[np.int64]:
    """Every ordering of count items, or seeded random ones for large counts."""
    if count <= EXACT_ENTRANTS:
        return np.asarray(list(itertools.permutations(range(count))), np.int64)
    generator = np.random.default_rng(PERMUTATION_SEED)
    return generator.permuted(np.tile(np.arange(count), (PERMUTATION_DRAWS, 1)), axis=1)


def _permutation_p(values: Sequence[float], ratings: Sequence[float]) -> float:
    """Two-sided permutation p-value of the rank correlation of values with ratings.

    The share of orderings of the ratings (all of them up to EXACT_ENTRANTS
    entrants; above that, PERMUTATION_DRAWS random ones counted as (hits + 1)
    / (draws + 1)) whose correlation is at least as far from zero as the
    observed one. Pairs with a NaN are left out; NaN when fewer than three
    remain or either side is constant, as for _rank_correlation.
    """
    kept = [
        (value, rating)
        for value, rating in zip(values, ratings, strict=True)
        if not (math.isnan(value) or math.isnan(rating))
    ]
    if len(kept) < 3:
        return math.nan
    raw = np.asarray([value for value, _ in kept])
    if _constant(raw):
        return math.nan
    x = _ranks(_rounded(raw))
    y = _ranks(np.asarray([rating for _, rating in kept]))
    x, y = x - x.mean(), y - y.mean()
    scale = math.sqrt(float((x**2).sum() * (y**2).sum()))
    if scale == 0:
        return math.nan
    observed = abs(float(np.dot(x, y))) / scale
    orderings = _orderings(len(kept))
    shuffled: Floats = np.abs(y[orderings] @ x) / scale
    hits = int(np.count_nonzero(shuffled >= observed - 1e-12))
    if len(kept) <= EXACT_ENTRANTS:
        return hits / len(orderings)
    return (hits + 1) / (len(orderings) + 1)


def _max_t(values: Sequence[Sequence[float]], ratings: Sequence[float]) -> list[float]:
    """Westfall-Young max-statistic adjusted p-values for rank correlations.

    values holds one list per metric, each with a value per entrant in the
    order of ratings. Metrics with a NaN or constant values get NaN and do not
    enter the maximum. Each kept metric's adjusted p-value is the share of
    orderings of the ratings (as _permutation_p) in which the largest absolute
    rank correlation over all kept metrics reaches its own.
    """
    keep = [
        index
        for index, row in enumerate(values)
        if not np.isnan(row).any() and not _constant(np.asarray(row, np.float64))
    ]
    adjusted = [math.nan] * len(values)
    if not keep or len(ratings) < 3:
        return adjusted
    x = np.stack([_ranks(_rounded(np.asarray(values[i], np.float64))) for i in keep])
    x = x - x.mean(axis=1, keepdims=True)
    x = x / np.sqrt((x**2).sum(axis=1, keepdims=True))
    y = _ranks(np.asarray(ratings, np.float64))
    y = (y - y.mean()) / math.sqrt(float(((y - y.mean()) ** 2).sum()))
    observed = np.abs(x @ y)
    orderings = _orderings(len(ratings))
    largest = np.concatenate(
        [
            np.abs(y[orderings[start : start + 20_000]] @ x.T).max(axis=1)
            for start in range(0, len(orderings), 20_000)
        ]
    )
    exact = len(ratings) <= EXACT_ENTRANTS
    for index, value in zip(keep, observed, strict=True):
        hits = int(np.count_nonzero(largest >= value - 1e-12))
        adjusted[index] = (
            hits / len(largest) if exact else (hits + 1) / (len(largest) + 1)
        )
    return adjusted


def _holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment over the finite p-values; NaN stays NaN."""
    finite = sorted(
        (value, index) for index, value in enumerate(p_values) if not math.isnan(value)
    )
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    for position, (value, index) in enumerate(finite):
        running = max(running, min(1.0, (len(finite) - position) * value))
        adjusted[index] = running
    return adjusted


@hydra.main(
    version_base="1.3", config_path="configs", config_name="tournament_analysis"
)
def main(config: DictConfig) -> dict[str, Any]:
    """Replay a tournament with full metrics, check it, and rank its teamwork.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``tournament_dir`` (the exact saved tournament run); ``field_file`` (the
        tournament declaration it was played from, given to
        ``marl_bgs.run_tournament`` as ``config``); ``names`` (display name
        for each entrant, keyed by the name the saved games record; entrants
        left out keep their recorded name); ``scripted`` (display names of
        scripted entrants, or null); ``num_envs`` (parallel games of the
        replay; use the saved run's value); ``replay_dir`` (an earlier replay
        to reuse, or null to play one); and the required ``output_dir``.
        Relative paths use the folder the command was started from.

    Returns
    -------
    dict
        ``identity`` (the ``compare_matches`` result), ``side_check`` (the
        ``check_sides`` result), ``replay_dir``, ``replay_seconds``
        (wall-clock seconds of the ``run_tournament`` call, which loads the
        entrants, compiles, plays and saves; None when ``replay_dir`` was
        given), ``execution_settings`` (the replay's
        ``results.execution_settings``: whether its games declared XLA's
        deterministic mode; None inside for replays saved before that record
        existed), ``analysis_seconds`` (wall-clock seconds of the two
        ``system_metrics`` calls), ``elo`` (display name to the saved rating),
        ``elo_correlations`` and ``profiles`` (display name to its
        ``team_profiles`` profile joined with its ``class_tables`` entry). The
        same is written to ``analysis.json``.

    Raises
    ------
    ValueError
        A replay is requested on a GPU without XLA's deterministic mode, the
        replay saved no run folder, an entrant has no saved rating, or
        ``check_sides`` finds a mismatch.

    Notes
    -----
    The replay runs the whole schedule with full metrics for every game,
    because each game's random draws depend on its place in the schedule.
    For Season 0 (7,200 games, 128 in parallel, one RTX 5090) it took about
    14 minutes and wrote about 51 kB of full metrics per game. With
    deterministic mode on the same GPU model and software, every game repeats
    exactly; ``identity`` shows whether it did. The analysis itself uses NumPy
    on the host and no JAX.
    """
    import marl_battlegrounds as marl_bgs

    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    names: dict[str, str] = dict(settings["names"] or {})
    scripted: list[str] = list(settings["scripted"] or [])

    def path(value: str) -> Path:
        """Resolve a setting's path from the folder the command started in."""
        return Path(to_absolute_path(str(Path(value).expanduser())))

    output_dir = path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read the saved tournament's games and ratings.
    saved = marl_bgs.load_results(path(settings["tournament_dir"]))
    matches = saved.table("matches")
    ratings = saved.table("tournament_results")
    elo = {
        names.get(str(policy), str(policy)): float(rating)
        for policy, rating in zip(ratings["policy"], ratings["elo"], strict=True)
    }

    # 2. Play the same schedule again, recording full metrics for every game.
    replay_seconds = None
    if settings["replay_dir"] is None:
        import jax

        if jax.default_backend() != "cpu" and not declared_gpu_deterministic_ops(
            read_declared_xla_flags()
        ):
            raise ValueError(
                "set XLA_FLAGS=--xla_gpu_deterministic_ops=true before Python starts"
            )
        started = time.perf_counter()
        replay = marl_bgs.run_tournament(
            config=path(settings["field_file"]),
            num_envs=settings["num_envs"],
            metrics="priority",
            full_metrics_episodes=sorted(int(game) for game in matches["episode_id"]),
            output_dir=output_dir / "replay",
        )
        replay_seconds = round(time.perf_counter() - started, 1)
        if replay.run_dir is None:
            raise ValueError("the replay saved no run folder")
        replay_dir = Path(replay.run_dir)
    else:
        replay_dir = path(settings["replay_dir"])
    replayed = marl_bgs.load_results(replay_dir)
    identity = compare_matches(matches, replayed.table("matches"))

    # 3. Summarize every entrant from its own side, and check the margins.
    started = time.perf_counter()
    summary = replayed.system_metrics(
        "full_metrics", columns=summary_columns(), ratios=RATIOS
    )
    games = replayed.system_metrics(
        "full_metrics",
        by="game",
        columns=list(
            dict.fromkeys(
                [
                    "score_difference",
                    "team_a_win",
                    "team_a_loss",
                    "episode_length",
                    *SCREEN_COLUMNS,
                    *CLASS_COLUMNS,
                ]
            )
        ),
    )
    analysis_seconds = round(time.perf_counter() - started, 1)
    side_check = check_sides(games, saved.head_to_head())

    # 4. Join each entrant's measures with its class tables.
    profiles = team_profiles(summary, names)
    for name, tables in class_tables(games, names).items():
        profiles[name].update(tables)
    unrated = sorted(set(profiles) - set(elo))
    if unrated:
        raise ValueError(f"no saved rating for {unrated}")

    # 5. Rank every team-level metric by its correlation with Elo, test each
    # trend, then save.
    screen = screen_tests(
        screen_measures(summary, elo, names, scripted), games, elo, names, scripted
    )
    write_outputs(profiles, elo, output_dir, screen, scripted)
    settings_record = replayed.execution_settings
    record = {
        "identity": identity,
        "side_check": side_check,
        "replay_dir": str(replay_dir),
        "replay_seconds": replay_seconds,
        "execution_settings": settings_record,
        "analysis_seconds": analysis_seconds,
        "elo": elo,
        "elo_correlations": elo_correlations(profiles, elo),
        "profiles": profiles,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{identity['identical']} of {identity['games']} games identical")
    print(f"Margins match head_to_head within {side_check:.1e} points")
    deterministic = settings_record["declared_gpu_deterministic_ops"]
    print(f"Deterministic mode declared for the replay: {deterministic}")
    if replay_seconds is not None:
        print(f"Replay with full metrics: {replay_seconds:,.1f} seconds")
    print(f"Analysis: {analysis_seconds:,.1f} seconds")
    print(output_dir / "summary.md")
    return record


if __name__ == "__main__":
    prepare_hydra()
    main()
