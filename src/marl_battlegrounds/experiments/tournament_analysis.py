"""Find the teamwork that separates strong tournament entrants from weak ones.

Run with XLA_FLAGS=--xla_gpu_deterministic_ops=true python -m
marl_battlegrounds.experiments.tournament_analysis tournament_dir=/path/saved-run
field=/path/field.json output_dir=/path/out.

A tournament saves priority metrics (results, points, kills and deaths) for
every game, which is enough to rate the entrants but not to see how they play.
This stage shows how to get the rest from the public API:

1. Read the saved tournament with marl_bgs.load_results.
2. Play its schedule again with marl_bgs.run_tournament, recording full
   metrics (all 11,192 measures) for every game through
   full_metrics_episodes, and check that every game repeats exactly.
3. Check that each entrant's point margin, read from the side it played in
   each game, equals the saved head_to_head result, so Team A and Team B
   columns are never mixed up.
4. Summarize each entrant's teamwork over all of its games (MEASURES and
   CLASS_TABLES), with 95% intervals from resampling its game pairs.
5. Rank every team-level metric in the catalog by its rank correlation with
   Elo, to find the measures that follow strength most closely.

Outputs in output_dir: teamwork.csv (each MEASURES value with its interval),
by_class.csv (who damaged, helped kill and healed whom, class by class),
elo_screen.csv (step 5), summary.md (readable tables), teamwork.png (each
measure against Elo) and analysis.json (everything, plus the checks and the
replay's time). Give replay_dir to redo steps 3-5 from an earlier replay
without playing again.
"""

from __future__ import annotations

import csv
import json
import os
import time
import warnings
from collections.abc import Collection, Mapping, Sequence
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

type Floats = NDArray[np.float64]
type Games = Mapping[int, tuple[str, str | None]]

# Class IDs 1-5, in the simulator's order.
CLASSES = ("Mage", "Warrior", "Hunter", "Rogue", "Priest")

# Each measure: (name, how games combine, full-metric stems after "team_a_" or
# "team_b_", plain meaning). "per_game" adds the stem over games and divides by
# the games played; "difference" does the same for the entrant's stem minus
# its opponent's; "ratio" divides the first stem's total by the second's;
# "weighted" averages the first stem with the second as its weight.
MEASURES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("margin", "difference", ("score",), "Points minus the opponent's, per game"),
    ("kills", "per_game", ("kills",), "Kills per game"),
    ("deaths", "per_game", ("deaths",), "Deaths per game"),
    (
        "focus_fire_ticks",
        "per_game",
        ("focus_fire_steps",),
        "Ticks per game when two or more teammates dealt damage",
    ),
    (
        "shared_kill_fraction",
        "ratio",
        ("multi_contributor_kills", "kills"),
        "Share of kills with two or more helpers on the killing tick",
    ),
    (
        "slowed_kill_fraction",
        "ratio",
        ("kills_of_hunter_basic_slow_recipient", "kills"),
        "Share of kills on enemies a teammate Hunter had already slowed",
    ),
    (
        "ultimate_kill_fraction",
        "ratio",
        ("ultimate_kills", "kills"),
        "Share of kills helped by an Ultimate used on the killing tick",
    ),
    (
        "ally_distance",
        "weighted",
        ("ally_distance_mean", "ally_distance_observations"),
        "Average distance between living teammates, in map units",
    ),
    (
        "mage_aura_coverage",
        "ratio",
        ("mage_aura_covered_steps", "mage_aura_eligible_steps"),
        "Share of teammate ticks spent inside a Mage aura",
    ),
    (
        "warrior_aura_coverage",
        "ratio",
        ("warrior_aura_covered_steps", "warrior_aura_eligible_steps"),
        "Share of teammate ticks spent inside a Warrior aura",
    ),
    (
        "mage_aura_damage",
        "per_game",
        ("damage_from_mage_aura",),
        "Extra damage per game that Mage auras added to teammates' attacks",
    ),
    (
        "warrior_aura_protection",
        "per_game",
        ("damage_prevented_by_warrior_aura",),
        "Damage per game that Warrior auras kept off teammates",
    ),
    (
        "useful_healing_fraction",
        "ratio",
        ("effective_healing_done", "healing_done"),
        "Share of Priest healing that was useful, not excess",
    ),
    (
        "rescue_rate",
        "ratio",
        ("rescues", "rescue_opportunities"),
        "Share of chances to save a dying ally with Priest healing that were taken",
    ),
    (
        "own_red_zone_deaths",
        "per_game",
        ("red_zone_deaths",),
        "Deaths per game inside the team's own Red Zone, worth 2 points each",
    ),
    ("basic_uses", "per_game", ("basic_activations",), "Basic abilities used per game"),
    (
        "ultimate_uses",
        "per_game",
        ("ultimate_activations",),
        "Ultimate abilities used per game",
    ),
)

# Measures that are game results; the figure shows only the teamwork after them.
RESULTS = ("margin", "kills", "deaths")

# Class tables: (name, plain meaning). Matrices have the entrant's class as
# rows and the enemy class as columns; vectors have one value per own class.
CLASS_TABLES = (
    ("damage_share", "Share of the team's damage, attacker class to enemy class"),
    ("kill_help", "Kill contributions per game, helper class to enemy class"),
    ("healing_share", "Share of the Priest's useful healing each ally class received"),
    ("damage_taken_share", "Share of the damage the team received, by class"),
)

# Resamples for the 95% intervals; the generator seed keeps them repeatable.
RESAMPLES = 2000
INTERVAL_SEED = 0


def entrant_games(
    matches: Mapping[str, Any], name: str, repeating: Collection[str] = ()
) -> dict[int, tuple[str, str | None]]:
    """Find every game an entrant played, on either team.

    Parameters
    ----------
    matches : mapping of str to numpy.ndarray
        The ``matches`` table of a saved tournament, as returned by
        ``marl_bgs.load_results(run).table("matches")``.
    name : str
        The entrant exactly as the ``team_a_policy`` and ``team_b_policy``
        columns record it.
    repeating : collection of str, default=()
        Opponents whose games against this entrant repeat exactly, as the
        tournament's ``matchup_results`` mark with ``determinism`` equal to
        ``"Deterministic"``: both sides choose actions without randomness.

    Returns
    -------
    dict of int to (str, str or None)
        Logical game ID (``episode_id``) to the side, ``"a"`` or ``"b"``, the
        entrant played and the game's ``block_id``, the pair of games that
        share a seed and swap spawn ends. The pair is None for games against
        a ``repeating`` opponent.

    Raises
    ------
    ValueError
        The entrant played no game, or played itself.
    """
    games: dict[int, tuple[str, str | None]] = {}
    for episode, team_a, team_b, block in zip(
        matches["episode_id"],
        matches["team_a_policy"],
        matches["team_b_policy"],
        matches["block_id"],
        strict=True,
    ):
        if team_a == name and team_b == name:
            raise ValueError(f"{name!r} played itself")
        if name in (team_a, team_b):
            opponent = team_b if team_a == name else team_a
            pair = None if opponent in repeating else str(block)
            games[int(episode)] = ("a" if team_a == name else "b", pair)
    if not games:
        raise ValueError(f"{name!r} played no game")
    return games


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


def team_profile(full: Mapping[str, Any], games: Games) -> dict[str, Any]:
    """Summarize one entrant's teamwork from the full metrics of its games.

    Parameters
    ----------
    full : mapping of str to numpy.ndarray
        ``full_metrics`` rows, from
        ``marl_bgs.load_results(run).table("full_metrics")``. Rows of games
        not in ``games`` are ignored.
    games : mapping of int to (str, str or None)
        ``entrant_games`` result: logical game ID to the entrant's side and
        the game's pair ID, or None for a game that repeats exactly.

    Returns
    -------
    dict
        ``games`` (games found), ``pairs`` (resampled pairs among them) and
        ``fixed_games`` (games that repeat exactly); for each ``MEASURES``
        name a dict with ``value`` and its 95% interval ``low`` and ``high``
        (fractions as 0-1, distances in map units, health and counts per
        game); and the ``CLASS_TABLES``: ``damage_share`` and ``kill_help``
        as 5 x 5 lists (the entrant's classes as rows, enemy classes as
        columns, in ``CLASSES`` order), ``healing_share`` and
        ``damage_taken_share`` as 5 values. A value whose denominator is zero
        in every game is NaN.

    Raises
    ------
    ValueError
        No row of ``full`` belongs to a game in ``games``.

    Notes
    -----
    Every value is read from the side the entrant played in that game. Agents
    0-4 are Team A and 5-9 Team B; each agent's class comes from its
    ``agent_<i>_class_id`` column and inactive agents are skipped, so any
    roster works. Intervals resample whole pairs with replacement, as the
    tournament's own Elo intervals do, ``RESAMPLES`` times with seed
    ``INTERVAL_SEED``. Games that repeat exactly count in every value but are
    held fixed in each resample, as the tournament gives such pairs no
    sampling interval; copies of one game would otherwise narrow the interval.
    """
    rows = [
        (index, *games[int(game)])
        for index, game in enumerate(full["episode_id"])
        if int(game) in games
    ]
    if not rows:
        raise ValueError("no full metrics for the requested games")
    pair_ids = sorted({pair for _, _, pair in rows if pair is not None})
    pair_of = {pair: number for number, pair in enumerate(pair_ids)}
    # Fixed games share one extra slot, after the pairs, that is never drawn.
    pairs = len(pair_ids)
    slot = np.array([pairs if pair is None else pair_of[pair] for _, _, pair in rows])

    def column(stem: str, opponent: bool = False) -> Floats:
        # The entrant's own side, or its opponent's; missing values add nothing.
        flip = {"a": "b", "b": "a"}
        return np.nan_to_num(
            np.array(
                [
                    float(full[f"team_{flip[side] if opponent else side}_{stem}"][row])
                    for row, side, _ in rows
                ]
            )
        )

    # Per measure, the per-game numerator and denominator whose totals give it.
    ones = np.ones(len(rows))
    parts: list[tuple[Floats, Floats]] = []
    for _, combine, stems, _ in MEASURES:
        if combine == "per_game":
            parts.append((column(stems[0]), ones))
        elif combine == "difference":
            parts.append((column(stems[0]) - column(stems[0], opponent=True), ones))
        elif combine == "ratio":
            parts.append((column(stems[0]), column(stems[1])))
        else:
            weight = column(stems[1])
            parts.append((column(stems[0]) * weight, weight))
    numerators = np.zeros((pairs + 1, len(MEASURES)))
    denominators = np.zeros((pairs + 1, len(MEASURES)))
    np.add.at(numerators, slot, np.stack([n for n, _ in parts], axis=1))
    np.add.at(denominators, slot, np.stack([d for _, d in parts], axis=1))
    value = _ratio(numerators.sum(axis=0), denominators.sum(axis=0))
    draws = np.random.default_rng(INTERVAL_SEED).integers(
        0, max(pairs, 1), size=(RESAMPLES, pairs)
    )
    samples = _ratio(
        numerators[pairs] + numerators[draws].sum(axis=1),
        denominators[pairs] + denominators[draws].sum(axis=1),
    )
    with warnings.catch_warnings():
        # A measure with no denominator in any game stays NaN, without a warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    profile: dict[str, Any] = {
        "games": len(rows),
        "pairs": pairs,
        "fixed_games": int((slot == pairs).sum()),
    }
    for number, (name, _, _, _) in enumerate(MEASURES):
        profile[name] = {
            "value": float(value[number]),
            "low": float(low[number]),
            "high": float(high[number]),
        }

    damage = np.zeros((5, 5))
    kill_help = np.zeros((5, 5))
    healing = np.zeros(5)
    taken = np.zeros(5)
    for row, side, _ in rows:
        kind = {
            agent: int(full[f"agent_{agent}_class_id"][row]) - 1
            for agent in range(10)
            if bool(full[f"agent_{agent}_active"][row])
        }
        own = [agent for agent in kind if (agent < 5) == (side == "a")]
        enemy = [agent for agent in kind if agent not in own]

        def read(name: str, row: int = row) -> float:
            # Missing values (NaN) add nothing.
            return float(np.nan_to_num(float(full[name][row])))

        for source in own:
            taken[kind[source]] += read(f"agent_{source}_damage_received")
            for target in enemy:
                pair = f"agent_{source}_to_agent_{target}"
                damage[kind[source], kind[target]] += read(f"{pair}_damage_done")
                kill_help[kind[source], kind[target]] += read(
                    f"{pair}_kill_contributions"
                )
            if kind[source] == CLASSES.index("Priest"):
                for ally in own:
                    healing[kind[ally]] += read(
                        f"agent_{source}_to_agent_{ally}_effective_healing_done"
                    )
    profile["damage_share"] = _share(damage)
    profile["kill_help"] = (kill_help / len(rows)).round(3).tolist()
    profile["healing_share"] = _share(healing)
    profile["damage_taken_share"] = _share(taken)
    return profile


def check_sides(
    profiles: Mapping[str, dict[str, Any]],
    head_to_head: Mapping[str, Any],
    names: Mapping[str, str],
) -> float:
    """Check each entrant's margin against the saved head-to-head results.

    Parameters
    ----------
    profiles : mapping of str to dict
        Display name to ``team_profile`` result.
    head_to_head : mapping of str to numpy.ndarray
        ``marl_bgs.load_results(run).head_to_head()`` of the saved tournament:
        one row per entrant and opponent with ``system_name``, ``games`` and
        ``mean_point_margin``.
    names : mapping of str to str
        Display name for each recorded entrant name; names left out are used
        as recorded.

    Returns
    -------
    float
        The largest difference, in points per game, between an entrant's
        ``margin`` and its game-weighted mean point margin in
        ``head_to_head``.

    Raises
    ------
    ValueError
        A difference is above 1e-6 points: a measure was read from the wrong
        team's columns, or the replay's games differ from the saved ones.
    """
    totals: dict[str, list[float]] = {}
    for system, games, margin in zip(
        head_to_head["system_name"],
        head_to_head["games"],
        head_to_head["mean_point_margin"],
        strict=True,
    ):
        total = totals.setdefault(names.get(str(system), str(system)), [0.0, 0.0])
        total[0] += float(games) * float(margin)
        total[1] += float(games)
    largest = max(
        abs(profiles[name]["margin"]["value"] - points / games)
        for name, (points, games) in totals.items()
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
        Display name to ``team_profile`` result.
    elo : mapping of str to float
        Display name to the saved tournament's Elo rating.

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
        for measure, _, _, _ in MEASURES
    }


def screen_measures(
    full: Mapping[str, Any],
    games: Mapping[str, Games],
    elo: Mapping[str, float],
    scripted: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Rank every team-level full metric by how closely it follows Elo.

    Parameters
    ----------
    full : mapping of str to numpy.ndarray
        ``full_metrics`` rows of the tournament's games.
    games : mapping of str to mapping
        Display name to that entrant's ``entrant_games`` result.
    elo : mapping of str to float
        Display name to the saved tournament's Elo rating.
    scripted : collection of str, default=()
        Display names of scripted entrants; ``correlation_learned`` leaves
        them out, so a measure that only separates scripted from learned
        entrants shows a weak learned correlation.

    Returns
    -------
    list of dict
        One row per team-level metric in the catalog except the game result
        itself (win, draw, loss, return and score): ``measure`` (the name
        after ``team_a_``), ``unit``, ``description`` (the catalog's
        description for one team), ``correlation_all`` and
        ``correlation_learned`` (rank correlations with Elo, as in
        ``elo_correlations``, over all entrants and over those not in
        ``scripted``) and each entrant's value under its display name. Rows
        are sorted from the strongest correlation over all entrants, rising
        or falling, down.

    Notes
    -----
    Each value is a plain average over the entrant's games: fractions over
    the games where they exist, everything else over all games. The curated
    ``MEASURES`` instead divide totals, so the two can differ slightly. With
    about 145 metrics and a field of nine, some strong correlations appear by
    chance; read a trend together with its learned correlation and meaning.
    """
    from marl_battlegrounds.evaluation.metric_catalog import METRIC_COLUMNS_BY_NAME

    index = {int(game): row for row, game in enumerate(full["episode_id"])}
    learned = [name for name in games if name not in scripted]
    rows: list[dict[str, Any]] = []
    for name, column in METRIC_COLUMNS_BY_NAME.items():
        stem = name.removeprefix("team_a_")
        if column.scope != "team" or not name.startswith("team_a_"):
            continue
        if stem in ("win", "draw", "loss", "return", "score"):
            continue
        sides = {"a": np.asarray(full[name], dtype=float)}
        sides["b"] = np.asarray(full[f"team_b_{stem}"], dtype=float)
        values: dict[str, float] = {}
        for entrant, played in games.items():
            picked = np.array(
                [sides[side][index[game]] for game, (side, _) in played.items()]
            )
            if column.unit == "fraction":
                present = picked[~np.isnan(picked)]
                values[entrant] = float(present.mean()) if len(present) else np.nan
            else:
                values[entrant] = float(np.nan_to_num(picked).mean())
        description = column.description.replace("Team A's", "the team's")
        row: dict[str, Any] = {
            "measure": stem,
            "unit": column.unit,
            "description": description.replace("Team A", "the team"),
            "correlation_all": _rank_correlation(values, elo, list(games)),
            "correlation_learned": _rank_correlation(values, elo, learned),
        }
        rows.append(row | values)
    return sorted(
        rows, key=lambda row: -np.nan_to_num(abs(row["correlation_all"]), nan=-1.0)
    )


def write_outputs(
    profiles: Mapping[str, dict[str, Any]],
    elo: Mapping[str, float],
    output_dir: Path,
    screen: Sequence[Mapping[str, Any]] = (),
    scripted: Collection[str] = (),
) -> None:
    """Write teamwork.csv, by_class.csv, elo_screen.csv, summary.md and the PNG.

    Parameters
    ----------
    profiles : mapping of str to dict
        Display name to ``team_profile`` result.
    elo : mapping of str to float
        Display name to the saved tournament's Elo rating; tables list the
        entrants from the highest rating down.
    output_dir : pathlib.Path
        Existing folder; files there with these names are replaced.
    screen : sequence of mapping, default=()
        ``screen_measures`` rows. When given, they are written to
        elo_screen.csv and the 20 strongest trends are added to summary.md.
    scripted : collection of str, default=()
        Display names drawn as open points in teamwork.png; the others are
        filled.

    Notes
    -----
    teamwork.png plots every ``MEASURES`` value after the game results
    against Elo, with its 95% interval, and gives each panel's rank
    correlation in its title. It needs matplotlib (the viz extra).
    """
    order = sorted(profiles, key=lambda name: -elo[name])
    correlations = elo_correlations(profiles, elo)
    with (output_dir / "teamwork.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = "system elo games pairs fixed_games measure value low high"
        writer.writerow(header.split())
        for name in order:
            profile = profiles[name]
            counts = (profile["games"], profile["pairs"], profile["fixed_games"])
            for measure, _, _, _ in MEASURES:
                cell = profile[measure]
                bounds = (round(cell[key], 4) for key in ("value", "low", "high"))
                writer.writerow((name, round(elo[name], 1), *counts, measure, *bounds))
    with (output_dir / "by_class.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["system", "table", "source_class", "target_class", "value"])
        for name in order:
            profile = profiles[name]
            for table in ("damage_share", "kill_help"):
                for row, source in enumerate(CLASSES):
                    for column, target in enumerate(CLASSES):
                        cell = profile[table][row][column]
                        writer.writerow([name, table, source, target, cell])
            for table, source in (
                ("healing_share", "Priest"),
                ("damage_taken_share", "Enemy"),
            ):
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
                writer.writerow(
                    {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in row.items()
                    }
                )

    lines = ["# Teamwork Of Every Entrant", ""]
    lines.append(
        "All games of each entrant; 95% intervals from resampling game pairs, "
        "with games that repeat exactly held fixed."
    )
    lines += ["", "| Measure | Meaning | Rank correlation with Elo | "]
    lines[-1] += " | ".join(order) + " |"
    lines.append("| --- | --- | ---: |" + " ---: |" * len(order))
    lines.append(
        "| elo | Saved tournament rating | +1.00 | "
        + " | ".join(f"{elo[name]:,.0f}" for name in order)
        + " |"
    )
    for measure, _, _, meaning in MEASURES:
        cells = [
            "{value:,.3f} ({low:,.3f}-{high:,.3f})".format(**profiles[name][measure])
            for name in order
        ]
        lines.append(
            f"| {measure} | {meaning} | {correlations[measure]:+.2f} | "
            + " | ".join(cells)
            + " |"
        )
    for table, meaning in CLASS_TABLES[2:]:
        lines += ["", f"**{meaning}** ({table})", ""]
        lines += ["| System | " + " | ".join(CLASSES) + " |", "| --- |" + " ---: |" * 5]
        for name in order:
            values = " | ".join(f"{value:.3f}" for value in profiles[name][table])
            lines.append(f"| {name} | {values} |")
    if screen:
        lines += ["", "**Strongest trends with Elo among all team metrics**", ""]
        lines.append("| Measure | Unit | All | Learned | Meaning |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for row in screen[:20]:
            lines.append(
                f"| {row['measure']} | {row['unit']} | "
                f"{row['correlation_all']:+.2f} | {row['correlation_learned']:+.2f} | "
                f"{row['description']} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [measure for measure in MEASURES if measure[0] not in RESULTS]
    columns = 5
    rows = -(-len(shown) // columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 2.7 * rows), constrained_layout=True
    )
    for axis, (measure, _, _, _) in zip(axes.flat, shown, strict=False):
        for name in order:
            cell = profiles[name][measure]
            axis.errorbar(
                elo[name],
                cell["value"],
                yerr=[[cell["value"] - cell["low"]], [cell["high"] - cell["value"]]],
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


def _ratio(top: Floats, bottom: Floats) -> Floats:
    """Divide totals elementwise; NaN where the denominator is not positive."""
    safe = np.where(bottom > 0, bottom, 1.0)
    return np.where(bottom > 0, top / safe, np.nan)


def _share(values: Floats) -> list[Any]:
    """Each value's share of the total, rounded to 4 places; zeros if empty."""
    return (values / max(float(values.sum()), 1e-9)).round(4).tolist()


def _rank_correlation(
    values: Mapping[str, float], elo: Mapping[str, float], names: Sequence[str]
) -> float:
    """Spearman correlation of values with Elo over the named entrants.

    Entrants whose value is NaN are left out. NaN when fewer than three remain
    or the values or ratings are all equal. Tied values share their average
    rank.
    """
    kept = [name for name in names if not np.isnan(values[name])]
    measure = np.array([values[name] for name in kept])
    ratings = np.array([elo[name] for name in kept])
    if len(kept) < 3 or np.ptp(measure) == 0 or np.ptp(ratings) == 0:
        return float("nan")
    return float(np.corrcoef(_ranks(measure), _ranks(ratings))[0, 1])


def _ranks(values: Floats) -> Floats:
    """Ranks from 1 upward; tied values share their average rank."""
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


@hydra.main(
    version_base="1.3", config_path="configs", config_name="tournament_analysis"
)
def main(config: DictConfig) -> dict[str, Any]:
    """Replay a tournament with full metrics, check it, and rank its teamwork.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``tournament_dir`` (the exact saved tournament run); ``field`` (the
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
        given), ``elo`` (display name to saved rating), ``elo_correlations``
        and ``profiles`` (display name to ``team_profile`` result). The same
        is written to ``analysis.json``.

    Raises
    ------
    ValueError
        A replay is requested on a GPU without XLA's deterministic mode, the
        replay saved no run folder, or ``check_sides`` finds mixed-up sides.

    Notes
    -----
    The replay runs the whole schedule with full metrics for every game,
    because each game's random draws depend on its place in the schedule.
    For Season 0 (7,200 games, 128 in parallel, one RTX 5090) it took about
    14 minutes and wrote about 51 kB of full metrics per game. With
    deterministic mode on the same GPU model and software, every game repeats
    exactly; ``identity`` shows whether it did.
    """
    import marl_battlegrounds as marl_bgs

    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    names: dict[str, str] = dict(settings["names"] or {})
    scripted: list[str] = list(settings["scripted"] or [])

    def path(value: str) -> Path:
        return Path(to_absolute_path(str(Path(value).expanduser())))

    output_dir = path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read the saved tournament: its games, ratings and pairings.
    saved = marl_bgs.load_results(path(settings["tournament_dir"]))
    matches = saved.table("matches")
    ratings = saved.table("tournament_results")
    matchups = saved.table("matchup_results")
    elo = {
        names.get(str(policy), str(policy)): float(rating)
        for policy, rating in zip(ratings["policy"], ratings["elo"], strict=True)
    }

    # 2. Play the same schedule again, recording full metrics for every game.
    replay_seconds = None
    if settings["replay_dir"] is None:
        import jax

        flags = os.environ.get("XLA_FLAGS", "")
        if (
            jax.default_backend() != "cpu"
            and "--xla_gpu_deterministic_ops=true" not in flags
        ):
            raise ValueError(
                "set XLA_FLAGS=--xla_gpu_deterministic_ops=true before Python starts"
            )
        started = time.perf_counter()
        replay = marl_bgs.run_tournament(
            config=path(settings["field"]),
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
    full = replayed.table("full_metrics")

    # 3. Summarize each entrant over all of its games, from the side it played,
    # and check the margins against the saved head-to-head results.
    repeating = {
        (str(policy), str(opponent))
        for policy, opponent, kind in zip(
            matchups["policy"],
            matchups["opponent"],
            matchups["determinism"],
            strict=True,
        )
        if kind == "Deterministic"
    }
    games = {
        names.get(str(policy), str(policy)): entrant_games(
            matches,
            str(policy),
            {other for one, other in repeating if one == str(policy)},
        )
        for policy in ratings["policy"]
    }
    profiles = {name: team_profile(full, played) for name, played in games.items()}
    side_check = check_sides(profiles, saved.head_to_head(), names)

    # 4. Rank every team-level metric by its correlation with Elo, then save.
    screen = screen_measures(full, games, elo, scripted)
    write_outputs(profiles, elo, output_dir, screen, scripted)
    record = {
        "identity": identity,
        "side_check": side_check,
        "replay_dir": str(replay_dir),
        "replay_seconds": replay_seconds,
        "elo": elo,
        "elo_correlations": elo_correlations(profiles, elo),
        "profiles": profiles,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{identity['identical']} of {identity['games']} games identical")
    print(f"Margins match head_to_head within {side_check:.1e} points")
    if replay_seconds is not None:
        print(f"Replay with full metrics: {replay_seconds:,.1f} seconds")
    print(output_dir / "summary.md")
    return record


if __name__ == "__main__":
    prepare_hydra()
    main()
