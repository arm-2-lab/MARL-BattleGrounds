"""Compare every zero-shot coordination team with the host team alone, game by game.

Use python -m marl_battlegrounds.experiments.zero_shot_coordination_analysis
zsc_dir=/path/run output_dir=/path/out, where zsc_dir is the output_dir of the
zero_shot_coordination stage.

Every team played the same games as the host team alone: the same opponents,
maps, spawn ends and seeds. So each game of a team is compared with the same
game of the host team alone, matched by opponent and episode ID, and 95%
intervals come from resampling whole seed pairs (the two games that share a
seed and swap spawn ends) within each opponent. The stage reads each saved run
with marl_bgs.load_results and results.system_metrics(by="game"), from the
team's own side (Team A), and measures, per game:

- the team: point margin (``score_difference``), wins, draws and losses;
- the newcomer, the agent that joined the team: its kill contributions (kills
  it helped make), deaths and damage done;
- distance: how far the newcomer stays from its four teammates, the mean of
  the four pair distances in map units, weighted by the ticks both were alive.
  In Season 0, stronger teams stand closer together.

For the host team alone, "the newcomer" is the host's own agent of the same
class, so every difference compares a stranger with the agent it replaced. A
newcomer that plays random legal actions (the "Random" System) marks the
floor: ``slot_value_kept`` places each newcomer between the host's own agent
(1) and that floor (0).

Outputs in output_dir: teams.csv (one row per team and class: each measure,
its change from the host alone, their intervals and the slot value kept),
map_ends.csv (each team's games against each opponent on each map and spawn
end), summary.md and zero_shot_coordination.png (the margin change of every
newcomer, one row per class). The analysis uses NumPy on the host and no JAX.
"""

from __future__ import annotations

import csv
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

# Measures per game, in table order: name and meaning.
MEASURES = (
    ("margin", "Point margin per game"),
    ("newcomer_kill_contributions", "Newcomer: kills it helped make, per game"),
    ("newcomer_deaths", "Newcomer: deaths per game"),
    ("newcomer_damage", "Newcomer: damage done per game"),
    ("newcomer_distance", "Newcomer's mean distance to its teammates (map units)"),
)

# The newcomer whose games mark the floor of slot_value_kept.
FLOOR = "Random"


def read_columns() -> list[str]:
    """List the full-metric columns the analysis reads from every run.

    Returns
    -------
    list of str
        ``score_difference``, Team A's wins, draws and losses, each Team A
        agent's kill contributions, deaths and damage done, and the mean ally
        distance and its tick count for every Team A pair. Each agent's class,
        the map and the spawn end come with every ``system_metrics`` row.
    """
    names = ["score_difference", "team_a_win", "team_a_draw", "team_a_loss"]
    for slot in range(5):
        names += [
            f"agent_{slot}_kill_contributions",
            f"agent_{slot}_deaths",
            f"agent_{slot}_damage_done",
        ]
    for first in range(5):
        for second in range(first + 1, 5):
            pair = f"agent_{first}_and_agent_{second}_ally_distance"
            names += [f"{pair}_mean", f"{pair}_observations"]
    return names


def game_measures(games: Columns, slot: int) -> dict[str, NDArray[np.float64]]:
    """Compute every ``MEASURES`` value for each game, around one slot.

    Parameters
    ----------
    games : mapping of str to numpy.ndarray
        Per-game rows from Team A's side, with every ``read_columns`` column;
        each array has one entry per game.
    slot : int
        The newcomer's Team A slot, 0 to 4.

    Returns
    -------
    dict of str to numpy.ndarray
        Each ``MEASURES`` name to float64 values, one per game. The distance is
        NaN in a game where the newcomer and its teammates were never alive
        together; a missing count reads as zero.
    """
    others = [other for other in range(5) if other != slot]

    def column(name: str) -> NDArray[np.float64]:
        """One column as float64, missing values as zero."""
        return np.nan_to_num(np.asarray(games[name], np.float64))

    def pair(other: int, part: str) -> NDArray[np.float64]:
        """One pair-distance column between the slot and another agent."""
        first, second = sorted((slot, other))
        name = f"agent_{first}_and_agent_{second}_ally_distance_{part}"
        return np.nan_to_num(np.asarray(games[name], np.float64))

    weights = np.stack([pair(other, "observations") for other in others])
    means = np.stack([pair(other, "mean") for other in others])
    ticks = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = np.where(ticks > 0, (means * weights).sum(axis=0) / ticks, np.nan)
    return {
        "margin": column("score_difference"),
        "newcomer_kill_contributions": column(f"agent_{slot}_kill_contributions"),
        "newcomer_deaths": column(f"agent_{slot}_deaths"),
        "newcomer_damage": column(f"agent_{slot}_damage_done"),
        "newcomer_distance": distance,
    }


def paired_interval(
    values: NDArray[np.float64],
    units: NDArray[Any],
    *,
    draws: int,
    seed: int,
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

    Returns
    -------
    (float, float, float)
        The mean over games, then the 2.5th and 97.5th percentiles of the
        resampled means. All NaN when no value is finite.
    """
    keep = np.isfinite(values)
    values, units = values[keep], units[keep]
    if values.size == 0:
        return (math.nan, math.nan, math.nan)
    labels, index = np.unique(units, return_inverse=True)
    totals = np.bincount(index, weights=values, minlength=labels.size)
    counts = np.bincount(index, minlength=labels.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, labels.size, size=(draws, labels.size))
    means = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(values.mean()), float(low), float(high))


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
        ``spawn_end`` ("team_a", the left end, or "team_b"), concatenated over
        opponents in order, one entry per game.

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
        # A team that also meets itself has both sides' rows; keep Team A's.
        mine = np.asarray(games["played_as"]) == "team_a"
        names = [*read_columns(), *(f"agent_{slot}_class_id" for slot in range(5))]
        part = {name: np.asarray(games[name])[mine] for name in names}
        for name in ("episode_id", "seed_id", "map_id"):
            part[name] = np.asarray(games[name])[mine].astype(np.int64)
        part["spawn_end"] = np.asarray(games["spawn_end"], dtype=object)[mine]
        part["opponent"] = np.full(int(mine.sum()), opponent, dtype=object)
        parts.append(part)
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


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
        ``read_team`` results for the team and for the host alone. Each game
        of ``team`` must have the same game (opponent and episode ID) in
        ``host``.
    newcomer_class : str
        The class from ``CLASSES`` whose agent is the newcomer; for the host
        alone it names the host's own agent of that class.
    draws, seed : int
        Resampling draws and seed for ``paired_interval``.

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
        A game has no match in ``host``, or the slot's recorded class is not
        ``newcomer_class``.
    """
    slot = CLASSES.index(newcomer_class)
    for games in (team, host):
        if np.any(np.asarray(games[f"agent_{slot}_class_id"]) != slot + 1):
            raise ValueError(f"slot {slot} is not the {newcomer_class} in every game")
    keys = list(zip(host["opponent"], host["episode_id"], strict=True))
    position = {key: index for index, key in enumerate(keys)}
    try:
        order = np.array(
            [
                position[key]
                for key in zip(team["opponent"], team["episode_id"], strict=True)
            ]
        )
    except KeyError as error:
        raise ValueError(f"game {error} has no match in the host's games") from None
    units = np.array(
        [f"{o}|{s}" for o, s in zip(team["opponent"], team["seed_id"], strict=True)]
    )
    mine, theirs = game_measures(team, slot), game_measures(host, slot)
    row: dict[str, Any] = {
        "games": int(units.size),
        "wins": int(np.nansum(np.asarray(team["team_a_win"], np.float64))),
        "draws": int(np.nansum(np.asarray(team["team_a_draw"], np.float64))),
        "losses": int(np.nansum(np.asarray(team["team_a_loss"], np.float64))),
    }
    for index, (name, _) in enumerate(MEASURES):
        value = paired_interval(mine[name], units, draws=draws, seed=seed + index)
        change = paired_interval(
            mine[name] - theirs[name][order], units, draws=draws, seed=seed + index
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
        "team_a", else "right"), ``games``, the mean ``margin`` and
        ``wins``, ``draws`` and ``losses``.
    """
    groups: dict[tuple[str, int, str], list[int]] = {}
    for index, key in enumerate(
        zip(team["opponent"], team["map_id"], team["spawn_end"], strict=True)
    ):
        opponent, map_id, end = key
        groups.setdefault((str(opponent), int(map_id), str(end)), []).append(index)
    margin = np.asarray(team["score_difference"], np.float64)
    flags = (("wins", "win"), ("draws", "draw"), ("losses", "loss"))
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
                "side": "left" if end == "team_a" else "right",
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
    Each newcomer is a dot with its 95% interval; the ``FLOOR`` newcomer is a
    cross. Zero, the dashed line, is the host's own agent. Needs matplotlib
    (the viz extra).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(dict.fromkeys(str(row["newcomer_class"]) for row in rows))
    newcomers = list(dict.fromkeys(str(row["newcomer"]) for row in rows))
    figure, axis = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    offsets = np.linspace(-0.3, 0.3, len(newcomers))
    for row in rows:
        y = classes.index(str(row["newcomer_class"]))
        y += float(offsets[newcomers.index(str(row["newcomer"]))])
        change = float(row["margin_change"])
        floor = row["newcomer"] == FLOOR
        axis.errorbar(
            change,
            y,
            xerr=[
                [change - float(row["margin_change_low"])],
                [float(row["margin_change_high"]) - change],
            ],
            fmt="x" if floor else "o",
            color="0.55" if floor else "0.15",
            markersize=4,
            capsize=1.5,
        )
    axis.axvline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    axis.set_yticks(range(len(classes)), classes)
    axis.invert_yaxis()
    axis.set_xlabel(f"Point margin change per game from {host} alone")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_summary(
    rows: Sequence[Mapping[str, Any]], host: str, output_dir: Path
) -> None:
    """Write summary.md: each newcomer's changes and slot value kept, by class.

    Parameters
    ----------
    rows : sequence of mapping
        ``teams.csv`` rows.
    host : str
        The host's name; its rows give the host's own values.
    output_dir : pathlib.Path
        Existing folder; summary.md there is replaced.
    """
    lines = [
        "# Zero-Shot Coordination",
        "",
        f"Each team is {host} with one agent replaced by a newcomer of the same "
        "class. Changes are per game, against the same games of the host alone, "
        "with 95% intervals; slot value kept is 1 for the host's own agent and 0 "
        f"for a {FLOOR} newcomer.",
        "",
        "| Class | Newcomer | Margin | Change | Slot value kept | Distance change "
        "| Newcomer kill help change | W-D-L |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:

        def cell(name: str, row: Mapping[str, Any] = row) -> str:
            """A change with its interval."""
            return (
                f"{row[f'{name}_change']:+.2f} ({row[f'{name}_change_low']:+.2f} "
                f"to {row[f'{name}_change_high']:+.2f})"
            )

        lines.append(
            f"| {row['newcomer_class']} | {row['newcomer']} | {row['margin']:+.2f} | "
            f"{cell('margin')} | {row['slot_value_kept']:.2f} | "
            f"{cell('newcomer_distance')} | {cell('newcomer_kill_contributions')} | "
            f"{row['wins']}-{row['draws']}-{row['losses']} |"
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
        then the ``compare`` values and ``slot_value_kept``. The host alone has
        one row per class, each comparing it with itself (changes of zero).

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
    map_ends = map_end_rows(host, host_name, host_name, "")
    for entry in teams:
        name = str(entry["name"])
        if entry["newcomer"] is None:
            games, classes = host, [str(kind) for kind in played["classes"]]
        else:
            games = read_team(zsc_dir / "teams" / folder_name(name), opponents)
            classes = [str(entry["newcomer_class"])]
            map_ends += map_end_rows(games, name, str(entry["newcomer"]), classes[0])
        for agent_class in classes:
            rows.append(
                {
                    "team": name,
                    "newcomer": entry["newcomer"] or host_name,
                    "newcomer_class": agent_class,
                }
                | compare(games, host, agent_class, draws=draws, seed=seed)
            )
        print(f"{name}: {rows[-1]['margin']:+.2f} points per game")
    for row, kept in zip(rows, slot_value_kept(rows, host_name), strict=True):
        row["slot_value_kept"] = kept
    for name, table in (("teams.csv", rows), ("map_ends.csv", map_ends)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(
                {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in row.items()
                }
                for row in table
            )
    write_summary(rows, host_name, output_dir)
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
