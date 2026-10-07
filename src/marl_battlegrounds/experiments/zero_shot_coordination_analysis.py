"""Compare every zero-shot coordination team with the host team alone, game by game.

Use python -m marl_battlegrounds.experiments.zero_shot_coordination_analysis
zsc_dir=/path/run output_dir=/path/out, where zsc_dir is the output_dir of the
zero_shot_coordination stage.

Every team played games with the same starts as the host team alone: game N of
every team has the same opponent, map, spawn end and seed. Matching games by
opponent and game ID pairs these starts; the games themselves differ, because
a different agent plays. 95% intervals come from resampling whole seed pairs
(the two games that share a seed and swap spawn ends) within each opponent.
The stage reads each saved run with marl_bgs.load_results and
results.system_metrics(by="game"), from the team's own side (Team A), and
measures, per game:

- the team: point margin (``score_difference``), wins, draws and losses;
- the newcomer, the agent that joined the team: kills it helped make, deaths,
  damage done, useful healing done (the Priest's job), damage received, and
  useful healing received from the team's Priest;
- the newcomer's aura (Mage and Warrior only): the share of its teammates'
  ticks spent inside it;
- distance: how far the newcomer stays from its four teammates, and how far
  the four kept agents stay from each other, in map units weighted by the
  ticks both agents were alive. In Season 0, stronger teams stand closer.

For the host team alone, "the newcomer" is the host's own agent of the same
class, so every difference compares a stranger with the agent it replaced. A
newcomer that plays random legal actions (the "Random" System) marks the
floor: ``slot_value_kept`` places each newcomer between the host's own agent
(1) and that floor (0), with an interval from the same draws. ``fit_drivers``
then asks which measure follows the cost of a stranger within each class.

Outputs in output_dir: teams.csv (one row per team and class: each measure,
its change from the host alone, their intervals and the slot value kept),
fit_drivers.csv (step above), map_ends.csv (each team's games against each
opponent on each map and spawn end), summary.md and zero_shot_coordination.png
(the margin change of every newcomer, one row per class). The analysis uses
NumPy on the host and no JAX.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections.abc import Mapping, Sequence
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

from marl_battlegrounds.experiments.zero_shot_coordination import (
    CLASSES,
    folder_name,
    team_plan,
)

type Columns = Mapping[str, NDArray[Any]]
type Floats = NDArray[np.float64]

# Measures per game, in table order: name and meaning.
MEASURES = (
    ("margin", "Point margin per game"),
    ("newcomer_kill_contributions", "Newcomer: kills it helped make, per game"),
    ("newcomer_deaths", "Newcomer: deaths per game"),
    ("newcomer_damage", "Newcomer: damage done per game"),
    ("newcomer_healing", "Newcomer: useful healing done per game"),
    ("newcomer_damage_received", "Newcomer: damage received per game"),
    ("newcomer_healing_received", "Newcomer: useful healing from the Priest"),
    ("newcomer_aura_coverage", "Share of teammates' ticks inside its aura"),
    ("newcomer_distance", "Newcomer's mean distance to its teammates (map units)"),
    ("kept_distance", "Mean distance among the four kept agents (map units)"),
)

# Each aura class and its catalog stem.
AURAS = {"Mage": "mage_aura", "Warrior": "warrior_aura"}

# The newcomer whose games mark the floor of slot_value_kept.
FLOOR = "Random"


def read_columns() -> list[str]:
    """List the full-metric columns the analysis reads from every run.

    Returns
    -------
    list of str
        ``score_difference``, Team A's wins, draws and losses; each Team A
        agent's kill contributions, deaths, damage done, damage received and
        useful healing done; the useful healing the Priest (slot 4) gave each
        teammate; the Mage's (slot 0) and Warrior's (slot 1) covered and
        eligible aura ticks for each other teammate; and the mean ally
        distance and its tick count for every Team A pair. Each agent's
        class, the map and the spawn end come with every ``system_metrics``
        row.
    """
    names = ["score_difference", "team_a_win", "team_a_draw", "team_a_loss"]
    for slot in range(5):
        names += [
            f"agent_{slot}_{stem}"
            for stem in (
                "kill_contributions",
                "deaths",
                "damage_done",
                "damage_received",
                "effective_healing_done",
            )
        ]
        names.append(f"agent_4_to_agent_{slot}_effective_healing_done")
    for kind, stem in AURAS.items():
        giver = CLASSES.index(kind)
        for ally in range(5):
            if ally != giver:
                names += [
                    f"agent_{giver}_to_agent_{ally}_{stem}_{part}"
                    for part in ("covered_steps", "eligible_steps")
                ]
    for first in range(5):
        for second in range(first + 1, 5):
            pair = f"agent_{first}_and_agent_{second}_ally_distance"
            names += [f"{pair}_mean", f"{pair}_observations"]
    return names


def game_measures(games: Columns, slot: int) -> dict[str, Floats]:
    """Compute every ``MEASURES`` value for each game, around one slot.

    Parameters
    ----------
    games : mapping of str to numpy.ndarray
        Per-game rows from Team A's side, with every ``read_columns`` column;
        each array has one entry per game.
    slot : int
        The newcomer's Team A slot, 0 to 4, in the canonical roster (0 Mage,
        1 Warrior, 4 Priest).

    Returns
    -------
    dict of str to numpy.ndarray
        Each ``MEASURES`` name to float64 values, one per game. A missing count
        reads as zero. NaN: both distances in a game where the agents were
        never alive together; ``newcomer_aura_coverage`` for a class without
        an aura, or a game with no tick it could cover; and
        ``newcomer_healing_received`` for the Priest itself.
    """
    others = [other for other in range(5) if other != slot]

    def column(name: str) -> Floats:
        """One column as float64, missing values as zero."""
        return np.nan_to_num(np.asarray(games[name], np.float64))

    def distance(pairs: Sequence[tuple[int, int]]) -> Floats:
        """Tick-weighted mean distance over agent pairs, NaN with no ticks."""
        names = [
            f"agent_{min(a, b)}_and_agent_{max(a, b)}_ally_distance" for a, b in pairs
        ]
        means = np.stack([column(f"{name}_mean") for name in names])
        weights = np.stack([column(f"{name}_observations") for name in names])
        ticks = weights.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ticks > 0, (means * weights).sum(axis=0) / ticks, np.nan)

    count = len(np.asarray(games["score_difference"]))
    coverage = np.full(count, np.nan)
    kind = CLASSES[slot]
    if kind in AURAS:
        covered = sum(
            column(f"agent_{slot}_to_agent_{ally}_{AURAS[kind]}_covered_steps")
            for ally in others
        )
        eligible = sum(
            column(f"agent_{slot}_to_agent_{ally}_{AURAS[kind]}_eligible_steps")
            for ally in others
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            coverage = np.where(eligible > 0, covered / eligible, np.nan)  # pyright: ignore[reportOperatorIssue] - sums of arrays are arrays.
    received = (
        column(f"agent_4_to_agent_{slot}_effective_healing_done")
        if slot != CLASSES.index("Priest")
        else np.full(count, np.nan)
    )
    return {
        "margin": column("score_difference"),
        "newcomer_kill_contributions": column(f"agent_{slot}_kill_contributions"),
        "newcomer_deaths": column(f"agent_{slot}_deaths"),
        "newcomer_damage": column(f"agent_{slot}_damage_done"),
        "newcomer_healing": column(f"agent_{slot}_effective_healing_done"),
        "newcomer_damage_received": column(f"agent_{slot}_damage_received"),
        "newcomer_healing_received": received,
        "newcomer_aura_coverage": coverage,
        "newcomer_distance": distance([(slot, other) for other in others]),
        "kept_distance": distance(list(itertools.combinations(others, 2))),
    }


def _units(units: NDArray[Any], strata: NDArray[Any] | None) -> tuple[Any, list[Any]]:
    """Sorted unit labels, each game's unit index and the units of each stratum."""
    labels, index = np.unique(units, return_inverse=True)
    if strata is None:
        return (labels, index), [np.arange(labels.size)]
    unit_stratum = np.empty(labels.size, dtype=object)
    unit_stratum[index] = np.asarray(strata, dtype=object)
    groups = [
        np.flatnonzero(unit_stratum == stratum)
        for stratum in dict.fromkeys(unit_stratum.tolist())
    ]
    return (labels, index), groups


def _draws(groups: Sequence[NDArray[Any]], draws: int, seed: int) -> NDArray[np.int64]:
    """Resampled unit indices, (draws, units), whole units within each stratum."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            group[rng.integers(0, group.size, size=(draws, group.size))]
            for group in groups
        ],
        axis=1,
    )


def paired_interval(
    values: Floats,
    units: NDArray[Any],
    *,
    draws: int,
    seed: int,
    strata: NDArray[Any] | None = None,
) -> tuple[float, float, float]:
    """Return a mean and its 95% interval from resampling whole units.

    Parameters
    ----------
    values : numpy.ndarray
        One value per game (a measure, or a team's value minus the host's in
        the same game). NaN values are left out.
    units : numpy.ndarray
        One unit label per game, the same length as ``values``; the games of a
        unit are resampled together.
    draws : int
        Resampling draws, at least 2.
    seed : int
        Seed of a private NumPy generator, so the interval repeats exactly.
        The same seed, units and strata give the same draws, so intervals of
        different teams share them.
    strata : numpy.ndarray or None, default=None
        One stratum label per game (such as its opponent); units are drawn
        within each stratum, keeping each stratum's number of units. None
        draws from all units together. A unit must lie in one stratum.

    Returns
    -------
    (float, float, float)
        The mean over games, then the 2.5th and 97.5th percentiles of the
        resampled means. All NaN when no value is finite.
    """
    keep = np.isfinite(values)
    if not keep.any():
        return (math.nan, math.nan, math.nan)
    (labels, index), groups = _units(
        units[keep], None if strata is None else np.asarray(strata)[keep]
    )
    totals = np.bincount(index, weights=values[keep], minlength=labels.size)
    counts = np.bincount(index, minlength=labels.size).astype(np.float64)
    picks = _draws(groups, draws, seed)
    means = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(values[keep].mean()), float(low), float(high))


def read_team(team_dir: Path, opponents: Sequence[str]) -> dict[str, NDArray[Any]]:
    """Read one team's games against every opponent, from its own side.

    Parameters
    ----------
    team_dir : pathlib.Path
        The team's folder from the zero_shot_coordination stage, holding
        ``runs.json``.
    opponents : sequence of str
        Opponent names, in order; each must be in ``runs.json``.

    Returns
    -------
    dict of str to numpy.ndarray
        Every ``read_columns`` column, plus ``agent_<slot>_class_id`` for
        slots 0-4, ``opponent``, ``episode_id``, ``seed_id``, ``map_id`` and
        ``spawn_end`` ("team_a", the left end, "team_b", or None when not
        saved), concatenated over opponents in order, one entry per game.

    Raises
    ------
    KeyError
        An opponent has no recorded run.
    ValueError, OSError
        A run cannot be read.
    """
    import marl_battlegrounds as marl_bgs

    runs = json.loads((team_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    parts: list[dict[str, NDArray[Any]]] = []
    for opponent in opponents:
        games = marl_bgs.load_results(runs[opponent]).system_metrics(
            "full_metrics", by="game", columns=read_columns()
        )
        # Keep the team's own side (a System that met itself is listed once).
        mine = np.asarray(games["played_as"]) == "team_a"
        names = [*read_columns(), *(f"agent_{slot}_class_id" for slot in range(5))]
        part = {name: np.asarray(games[name])[mine] for name in names}
        for name in ("episode_id", "seed_id", "map_id"):
            part[name] = np.asarray(games[name])[mine].astype(np.int64)
        part["spawn_end"] = np.asarray(games["spawn_end"], dtype=object)[mine]
        part["opponent"] = np.full(int(mine.sum()), opponent, dtype=object)
        parts.append(part)
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def matched(team: Columns, host: Columns) -> tuple[NDArray[np.int64], NDArray[Any]]:
    """Match each of a team's games with the host's game of the same start.

    Parameters
    ----------
    team, host : mapping of str to numpy.ndarray
        ``read_team`` results.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        For each team game, the index of the host's game with the same
        opponent and game ID; and each team game's resampling unit,
        "opponent|seed" (the seed pair it belongs to).

    Raises
    ------
    ValueError
        A game has no match, or a matched pair differs in seed, map or spawn
        end.
    """
    keys = list(zip(host["opponent"], host["episode_id"], strict=True))
    position = {key: index for index, key in enumerate(keys)}
    try:
        order = np.array(
            [
                position[key]
                for key in zip(team["opponent"], team["episode_id"], strict=True)
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(f"game {error} has no match in the host's games") from None
    for name in ("seed_id", "map_id", "spawn_end"):
        if np.any(np.asarray(team[name]) != np.asarray(host[name])[order]):
            raise ValueError(f"matched games differ in {name}")
    units = np.array(
        [f"{o}|{s}" for o, s in zip(team["opponent"], team["seed_id"], strict=True)]
    )
    return order, units


def compare(
    team: Columns,
    host: Columns,
    newcomer_class: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Compare one team with the host team alone, game by game.

    Parameters
    ----------
    team, host : mapping of str to numpy.ndarray
        ``read_team`` results for the team and for the host alone (see
        ``matched``).
    newcomer_class : str
        The class from ``CLASSES`` whose agent is the newcomer; for the host
        alone it names the host's own agent of that class.
    draws, seed : int
        Resampling draws and seed for ``paired_interval``; measure k uses
        seed + k, so every team's margin intervals share their draws.

    Returns
    -------
    dict
        ``games``, ``wins``, ``draws`` and ``losses``; then, for each
        ``MEASURES`` name, the team's value with ``_low`` and ``_high`` and
        its change from the host alone (``_change``, ``_change_low``,
        ``_change_high``), all means per game. Intervals resample seed pairs
        within each opponent.

    Raises
    ------
    ValueError
        As ``matched``, or the slot's recorded class is not ``newcomer_class``.
    """
    slot = CLASSES.index(newcomer_class)
    for games in (team, host):
        if np.any(np.asarray(games[f"agent_{slot}_class_id"]) != slot + 1):
            raise ValueError(f"slot {slot} is not the {newcomer_class} in every game")
    order, units = matched(team, host)
    strata = np.asarray(team["opponent"], dtype=object)
    mine, theirs = game_measures(team, slot), game_measures(host, slot)
    row: dict[str, Any] = {
        "games": int(units.size),
        "wins": int(np.nansum(np.asarray(team["team_a_win"], np.float64))),
        "draws": int(np.nansum(np.asarray(team["team_a_draw"], np.float64))),
        "losses": int(np.nansum(np.asarray(team["team_a_loss"], np.float64))),
    }
    for index, (name, _) in enumerate(MEASURES):
        value = paired_interval(
            mine[name], units, draws=draws, seed=seed + index, strata=strata
        )
        change = paired_interval(
            mine[name] - theirs[name][order],
            units,
            draws=draws,
            seed=seed + index,
            strata=strata,
        )
        row |= dict(zip((name, f"{name}_low", f"{name}_high"), value, strict=True))
        row |= dict(
            zip(
                (f"{name}_change", f"{name}_change_low", f"{name}_change_high"),
                change,
                strict=True,
            )
        )
    return row


def kept_interval(
    change: Floats,
    floor: Floats,
    units: NDArray[Any],
    strata: NDArray[Any],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    """Slot value kept and its 95% interval, from draws shared with the floor.

    Parameters
    ----------
    change : numpy.ndarray
        The team's margin change from the host, one value per game, in the
        host's game order.
    floor : numpy.ndarray
        The ``FLOOR`` team's margin change in the same class, in the same
        order, so both arrays line up game by game.
    units, strata : numpy.ndarray
        Each game's unit and stratum (its opponent), as for ``paired_interval``.
    draws, seed : int
        The same draws and seed ``compare`` used for the margin, so each draw
        resamples the same seed pairs for the team and for the floor.

    Returns
    -------
    (float, float, float)
        1 minus the team's mean change divided by the floor's, then the 2.5th
        and 97.5th percentiles of that ratio over the draws. All NaN when the
        floor's mean change is zero.
    """
    (labels, index), groups = _units(units, strata)
    picks = _draws(groups, draws, seed)
    counts = np.bincount(index, minlength=labels.size).astype(np.float64)
    totals = np.bincount(index, weights=change, minlength=labels.size)
    floor_totals = np.bincount(index, weights=floor, minlength=labels.size)
    if floor_totals.sum() == 0:
        return (math.nan, math.nan, math.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        kept = 1.0 - totals[picks].sum(axis=1) / floor_totals[picks].sum(axis=1)
    low, high = np.nanpercentile(kept, [2.5, 97.5])
    point = 1.0 - (totals.sum() / counts.sum()) / (floor_totals.sum() / counts.sum())
    return (float(point), float(low), float(high))


def slot_value_kept(rows: Sequence[Mapping[str, Any]], host: str) -> list[float]:
    """Place each newcomer between the host's own agent and the floor.

    Parameters
    ----------
    rows : sequence of mapping
        ``compare`` rows with ``newcomer``, ``newcomer_class`` and
        ``margin_change``, including the ``FLOOR`` newcomer of each class.
    host : str
        The host's name; its own rows keep 1.

    Returns
    -------
    list of float
        For each row, in order: 1 minus its margin change divided by the
        floor's margin change in the same class. 1 means the newcomer plays
        its slot as well as the host's own agent did, 0 as badly as random
        legal actions; values can leave that range. NaN when the class has no
        floor row or the floor did not change the margin.
    """
    floor = {
        row["newcomer_class"]: float(row["margin_change"])
        for row in rows
        if row["newcomer"] == FLOOR
    }
    kept: list[float] = []
    for row in rows:
        loss = floor.get(row["newcomer_class"], math.nan)
        if row["newcomer"] == host:
            kept.append(1.0)
        elif math.isfinite(loss) and loss != 0:
            kept.append(1.0 - float(row["margin_change"]) / loss)
        else:
            kept.append(math.nan)
    return kept


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rank correlation; tied values share their average rank."""

    def ranks(values: Sequence[float]) -> Floats:
        """Ranks from 1 upward, ties sharing their average rank."""
        _, inverse, counts = np.unique(
            np.asarray(values), return_inverse=True, return_counts=True
        )
        upper = np.cumsum(counts)
        return (upper - (counts - 1) / 2)[inverse].astype(np.float64)

    a, b = ranks(x), ranks(y)
    a, b = a - a.mean(), b - b.mean()
    scale = math.sqrt(float((a**2).sum() * (b**2).sum()))
    return float(np.dot(a, b)) / scale if scale > 0 else math.nan


def fit_drivers(rows: Sequence[Mapping[str, Any]], host: str) -> list[dict[str, Any]]:
    """Ask which measure's change follows the cost of a stranger within a class.

    Parameters
    ----------
    rows : sequence of mapping
        ``teams.csv`` rows. The host's rows and the ``FLOOR`` rows are left
        out, so each class compares the learned strangers only.
    host : str
        The host's name.

    Returns
    -------
    list of dict
        One row per ``MEASURES`` name after the margin: ``measure``, the rank
        correlation between its change and the margin change within each
        class (``rho_<Class>``), their mean (``mean_rho``) and ``p_value``,
        the exact two-sided permutation p-value of that mean: the share of all
        ways of shuffling the strangers within every class whose mean is at
        least as far from zero. Classes with fewer than three strangers or a
        constant measure are left out of the mean; NaN when none is left.

    Notes
    -----
    Each class's correlations over all orderings of its strangers are
    combined exactly by convolution, so no shuffle is drawn at random. With
    five strangers per class, one class alone cannot go below p = 1/60. A
    measure that follows the cost is a sign of how strangers fail, not a
    cause.
    """
    strangers = [r for r in rows if r["newcomer"] not in (host, FLOOR)]
    classes = list(dict.fromkeys(str(r["newcomer_class"]) for r in strangers))
    result: list[dict[str, Any]] = []
    for name, _ in MEASURES[1:]:
        record: dict[str, Any] = {"measure": name}
        distributions: list[dict[float, int]] = []
        observed: list[float] = []
        for kind in classes:
            group = [r for r in strangers if r["newcomer_class"] == kind]
            x = [float(r[f"{name}_change"]) for r in group]
            y = [float(r["margin_change"]) for r in group]
            rho = _spearman(x, y) if len(group) >= 3 else math.nan
            record[f"rho_{kind}"] = rho
            if not math.isfinite(rho):
                continue
            observed.append(rho)
            counts: dict[float, int] = {}
            for order in itertools.permutations(range(len(y))):
                value = round(_spearman(x, [y[i] for i in order]), 9)
                counts[value] = counts.get(value, 0) + 1
            distributions.append(counts)
        if not observed:
            record |= {"mean_rho": math.nan, "p_value": math.nan}
            result.append(record)
            continue
        total: dict[float, int] = {0.0: 1}
        for counts in distributions:
            combined: dict[float, int] = {}
            for left, ways in total.items():
                for right, more in counts.items():
                    key = round(left + right, 9)
                    combined[key] = combined.get(key, 0) + ways * more
            total = combined
        target = abs(sum(observed)) - 1e-9
        extreme = sum(ways for value, ways in total.items() if abs(value) >= target)
        record["mean_rho"] = sum(observed) / len(observed)
        record["p_value"] = extreme / sum(total.values())
        result.append(record)
    return result


def map_end_rows(
    team: Columns, name: str, newcomer: str, newcomer_class: str
) -> list[dict[str, Any]]:
    """Summarize one team's games by opponent, map and spawn end.

    Parameters
    ----------
    team : mapping of str to numpy.ndarray
        A ``read_team`` result.
    name, newcomer, newcomer_class : str
        The team's name, its newcomer (the host's name for the host alone) and
        the newcomer's class (empty for the host alone).

    Returns
    -------
    list of dict
        One row per (opponent, map, spawn end) in first-seen order: the three
        names, ``opponent``, ``map_id``, ``side`` ("left" for spawn end
        "team_a", "right" for "team_b", "unknown" when not saved), ``games``,
        the mean ``margin`` and ``wins``, ``draws`` and ``losses``.
    """
    groups: dict[tuple[str, int, str], list[int]] = {}
    for index, key in enumerate(
        zip(team["opponent"], team["map_id"], team["spawn_end"], strict=True)
    ):
        opponent, map_id, end = key
        groups.setdefault((str(opponent), int(map_id), str(end)), []).append(index)
    margin = np.asarray(team["score_difference"], np.float64)
    flags = (("wins", "win"), ("draws", "draw"), ("losses", "loss"))
    sides = {"team_a": "left", "team_b": "right"}
    rows: list[dict[str, Any]] = []
    for (opponent, map_id, end), indices in groups.items():
        counts = {
            result: int(
                np.nansum(np.asarray(team[f"team_a_{flag}"], np.float64)[indices])
            )
            for result, flag in flags
        }
        rows.append(
            {
                "team": name,
                "newcomer": newcomer,
                "newcomer_class": newcomer_class,
                "opponent": opponent,
                "map_id": map_id,
                "side": sides.get(end, "unknown"),
                "games": len(indices),
                "margin": float(np.mean(margin[indices])),
            }
            | counts
        )
    return rows


def write_figure(rows: Sequence[Mapping[str, Any]], host: str, path: Path) -> None:
    """Draw each newcomer's margin change from the host alone, one row per class.

    Parameters
    ----------
    rows : sequence of mapping
        ``teams.csv`` rows of the teams with a newcomer.
    host : str
        The host's name, for the axis label.
    path : pathlib.Path
        PNG file to write.

    Notes
    -----
    Each newcomer has its own marker, named in the legend, with its 95%
    interval; the ``FLOOR`` newcomer is a grey cross. Zero, the dashed line,
    is the host's own agent. Needs matplotlib (the viz extra).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(dict.fromkeys(str(row["newcomer_class"]) for row in rows))
    newcomers = list(dict.fromkeys(str(row["newcomer"]) for row in rows))
    markers = dict(zip(newcomers, "osD^vp<>h", strict=False))
    figure, axis = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    offsets = np.linspace(-0.3, 0.3, len(newcomers))
    for position, newcomer in enumerate(newcomers):
        floor = newcomer == FLOOR
        mine = [row for row in rows if row["newcomer"] == newcomer]
        changes = np.array([float(row["margin_change"]) for row in mine])
        axis.errorbar(
            changes,
            [
                classes.index(str(row["newcomer_class"])) + offsets[position]
                for row in mine
            ],
            xerr=[
                changes - np.array([float(row["margin_change_low"]) for row in mine]),
                np.array([float(row["margin_change_high"]) for row in mine]) - changes,
            ],
            fmt="x" if floor else markers[newcomer],
            color="0.55" if floor else "0.15",
            markersize=4,
            capsize=1.5,
            linestyle="none",
            label=newcomer,
        )
    axis.axvline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    axis.set_yticks(range(len(classes)), classes)
    axis.invert_yaxis()
    axis.set_xlabel(f"Point margin change per game from {host} alone")
    axis.legend(fontsize=7, loc="lower left", frameon=False)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_summary(
    rows: Sequence[Mapping[str, Any]],
    drivers: Sequence[Mapping[str, Any]],
    host: str,
    output_dir: Path,
) -> None:
    """Write summary.md: each newcomer's changes and slot value kept, by class.

    Parameters
    ----------
    rows : sequence of mapping
        ``teams.csv`` rows.
    drivers : sequence of mapping
        ``fit_drivers`` rows.
    host : str
        The host's name; its rows give the host's own values.
    output_dir : pathlib.Path
        Existing folder; summary.md there is replaced.
    """
    lines = [
        "# Zero-Shot Coordination",
        "",
        f"Each team is {host} with one agent replaced by a newcomer of the same "
        "class. Changes are per game, against the games of the host alone with "
        "the same starts, with 95% intervals; slot value kept is 1 for the host's "
        f"own agent and 0 for a {FLOOR} newcomer.",
        "",
        "| Class | Newcomer | Margin | Change | Slot value kept | Distance change "
        "| Kept agents' distance change | W-D-L |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:

        def cell(name: str, row: Mapping[str, Any] = row) -> str:
            """A change with its interval."""
            return (
                f"{row[f'{name}_change']:+.2f} ({row[f'{name}_change_low']:+.2f} "
                f"to {row[f'{name}_change_high']:+.2f})"
            )

        kept = (
            f"{row['slot_value_kept']:.2f} ({row['slot_value_kept_low']:.2f} to "
            f"{row['slot_value_kept_high']:.2f})"
        )
        lines.append(
            f"| {row['newcomer_class']} | {row['newcomer']} | {row['margin']:+.2f} | "
            f"{cell('margin')} | {kept} | {cell('newcomer_distance')} | "
            f"{cell('kept_distance')} | {row['wins']}-{row['draws']}-{row['losses']} |"
        )
    classes = [key.removeprefix("rho_") for key in drivers[0] if key.startswith("rho_")]
    lines += [
        "",
        "**What follows the cost of a stranger within each class** (rank "
        "correlation between a measure's change and the margin change over the "
        "learned strangers; p: exact permutation test of the mean)",
        "",
        "| Measure | " + " | ".join(classes) + " | Mean | p |",
        "| --- |" + " ---: |" * (len(classes) + 2),
    ]
    for row in drivers:
        cells = " | ".join(f"{row[f'rho_{kind}']:+.2f}" for kind in classes)
        lines.append(
            f"| {row['measure']} | {cells} | {row['mean_rho']:+.2f} | "
            f"{row['p_value']:.2g} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="zero_shot_coordination_analysis",
)
def main(config: DictConfig) -> list[dict[str, Any]]:
    """Compare every team with the host alone and write the tables and figure.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``zsc_dir`` (the zero_shot_coordination stage's output_dir),
        ``bootstrap_draws``, ``bootstrap_seed`` and the required
        ``output_dir``. Relative paths use the folder the command was
        started from.

    Returns
    -------
    list of dict
        The ``teams.csv`` rows: ``team``, ``newcomer`` and ``newcomer_class``,
        then the ``compare`` values and ``slot_value_kept`` with
        ``slot_value_kept_low`` and ``_high`` (``kept_interval``; the host's
        rows are 1 with no width). The host alone has one row per class, each
        comparing it with itself (changes of zero).

    Raises
    ------
    KeyError, ValueError, OSError
        A team or opponent run is missing or unreadable, or games do not match.

    Notes
    -----
    Host-only NumPy and CSV reading, no JAX. Writes only to output_dir.
    """
    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.

    def path(value: str) -> Path:
        """Resolve a setting's path from the folder the command started in."""
        return Path(to_absolute_path(str(Path(value).expanduser())))

    zsc_dir, output_dir = path(settings["zsc_dir"]), path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    played = json.loads((zsc_dir / "settings.json").read_text(encoding="utf-8"))
    host_name = str(played["host"])
    opponents = [str(name) for name in played["opponents"]]
    teams = team_plan(host_name, played["newcomers"], played["classes"])
    draws, seed = int(settings["bootstrap_draws"]), int(settings["bootstrap_seed"])
    host = read_team(zsc_dir / "teams" / folder_name(host_name), opponents)
    rows: list[dict[str, Any]] = []
    # Every team's margin change per game, in the host's game order, so the
    # changes of two teams line up game by game for kept_interval.
    _, host_units = matched(host, host)
    host_strata = np.asarray(host["opponent"], dtype=object)
    host_margin = np.asarray(host["score_difference"], np.float64)
    changes: dict[tuple[str, str], Floats] = {}
    map_ends = map_end_rows(host, host_name, host_name, "")
    for entry in teams:
        name = str(entry["name"])
        if entry["newcomer"] is None:
            games, classes = host, [str(kind) for kind in played["classes"]]
        else:
            games = read_team(zsc_dir / "teams" / folder_name(name), opponents)
            classes = [str(entry["newcomer_class"])]
            map_ends += map_end_rows(games, name, str(entry["newcomer"]), classes[0])
        order, _ = matched(games, host)
        change = np.full(host_margin.size, np.nan)
        change[order] = (
            np.asarray(games["score_difference"], np.float64) - host_margin[order]
        )
        for agent_class in classes:
            newcomer = str(entry["newcomer"] or host_name)
            changes[(newcomer, agent_class)] = change
            rows.append(
                {"team": name, "newcomer": newcomer, "newcomer_class": agent_class}
                | compare(games, host, agent_class, draws=draws, seed=seed)
            )
        print(f"{name}: {rows[-1]['margin']:+.2f} points per game")
    for row, kept in zip(rows, slot_value_kept(rows, host_name), strict=True):
        key = (str(row["newcomer"]), str(row["newcomer_class"]))
        floor = changes.get((FLOOR, key[1]))
        low = high = kept
        if row["newcomer"] != host_name and floor is not None:
            _, low, high = kept_interval(
                changes[key], floor, host_units, host_strata, draws=draws, seed=seed
            )
        row |= {
            "slot_value_kept": kept,
            "slot_value_kept_low": low,
            "slot_value_kept_high": high,
        }
    drivers = fit_drivers(rows, host_name)
    for name, table in (
        ("teams.csv", rows),
        ("fit_drivers.csv", drivers),
        ("map_ends.csv", map_ends),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(
                {
                    key: round(value, 4)
                    if isinstance(value, float) and key != "p_value"
                    else float(f"{value:.4g}")
                    if isinstance(value, float)
                    else value
                    for key, value in row.items()
                }
                for row in table
            )
    write_summary(rows, drivers, host_name, output_dir)
    write_figure(
        [row for row in rows if row["newcomer"] != host_name],
        host_name,
        output_dir / "zero_shot_coordination.png",
    )
    print(output_dir / "summary.md")
    return rows


if __name__ == "__main__":
    prepare_hydra()
    main()
