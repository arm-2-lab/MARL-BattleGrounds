"""Compare a person's recorded games with the Systems' own games.

Run with python -m marl_battlegrounds.experiments.human_games_analysis
series_dir=/path/series field_dir=/path/field-run output_dir=/path/out.

series_dir is a folder written by the human_games stage: one folder per game
with its game.json, full_metrics.csv and replay. field_dir is a saved run with
full metrics for every game of the field the Systems come from, such as the
Season 0 tournament's full-metrics replay (tournament_analysis). This stage
shows how to read both with the public API and put them side by side:

1. Read each recorded game's full metrics and its replay
   (marl_battlegrounds.evaluation.load_replay).
2. Read the reference games from the field with one
   results.system_metrics(by="game") call, each row from the System's side.
3. Against games (the person controls all of Team A): screen every team-level
   metric of the person's team against the opponent's own wins on the same map,
   and keep the differences that point the same way in every game. Counts,
   health and ticks are compared per 100 ticks, because games end at different
   times; fractions and distances are compared as they are.
4. Class tables for the person's team, the opponent in the same games and the
   opponent's wins: who died, who took damage, who received the Priest's
   Freedom, how much of each ally class's time was inside the Mage's and the
   Warrior's auras, and which enemy classes each side killed.
5. Ultimate timing from the replays: for each class and side, how long after
   its Ultimate was ready the agent used it, and how many enemies its team
   killed in the next window_ticks ticks, beside the team's kills in an
   average window of that length.
6. Alongside games (the person controls one class beside a partner System):
   compare each game with the partner's own games against the same opponent on
   the same map and spawn end, as a result (point margin, kills, deaths) and
   as a screen of every team-level metric and of the person's own agent.

Outputs in output_dir: against_screen.csv (step 3), classes.csv (step 4),
ultimates.csv (step 5), alongside.csv and alongside_screen.csv (step 6),
summary.md (readable tables) and analysis.json (settings and inputs). With
three games per kind, every result is a description of these games, found by
screening many metrics, not a test: a pattern must be checked in more games
before it is a claim.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
# isort: on

from marl_battlegrounds.evaluation.metric_catalog import METRIC_COLUMNS_BY_NAME

# One game: metric values as floats; names (System, spawn end) as text.
type Row = dict[str, Any]

# Canonical roster order: slot k plays CLASSES[k] on each team.
CLASSES = ("Mage", "Warrior", "Hunter", "Rogue", "Priest")

# Units compared per 100 ticks; other units (fractions, distances, agents per
# wave) are compared as they are.
RATE_UNITS = frozenset({"count", "health", "agent_steps", "steps", "pair_steps"})

# Every team-level metric except the game result itself.
TEAM_COLUMNS = tuple(
    name
    for name, column in METRIC_COLUMNS_BY_NAME.items()
    if column.scope == "team"
    and name.startswith("team_a_")
    and name.removeprefix("team_a_") not in ("win", "draw", "loss", "return", "score")
)

# Agent columns behind the class tables, for agents 0-4 (one team).
CLASS_COLUMNS = tuple(
    [
        f"agent_{agent}_{stem}"
        for agent in range(10)
        for stem in ("deaths", "damage_received", "damage_done")
    ]
    + [
        f"agent_4_to_agent_{ally}_{stem}"
        for ally in range(5)
        for stem in ("priest_freedom_applications", "effective_healing_done")
    ]
    + [
        f"agent_{first}_and_agent_{second}_ally_distance_{part}"
        for first in range(5)
        for second in range(first + 1, 5)
        for part in ("mean", "observations")
    ]
    + [
        f"agent_{giver}_to_agent_{ally}_{aura}_{part}"
        for giver, aura in ((0, "mage_aura"), (1, "warrior_aura"))
        for ally in range(5)
        if ally != giver
        for part in ("covered_steps", "eligible_steps")
    ]
)


def agent_columns(slot: int) -> list[str]:
    """List one agent's own full-metric columns (no pair columns).

    Parameters
    ----------
    slot : int
        Agent slot 0-9.

    Returns
    -------
    list of str
        Every catalog column ``agent_<slot>_*`` that is not an agent-pair
        column (``agent_<i>_to_agent_<j>_*`` or ``agent_<i>_and_agent_<j>_*``),
        in catalog order.
    """
    prefix = f"agent_{slot}_"
    return [
        name
        for name in METRIC_COLUMNS_BY_NAME
        if name.startswith(prefix)
        and "_to_agent_" not in name
        and "_and_agent_" not in name
    ]


# Agent indices in a column name, for reading a game from Team B's side.
_AGENT = re.compile(r"agent_(\d+)")


def mirrored(row: Mapping[str, float]) -> Row:
    """Read a game's full metrics from Team B's side.

    Parameters
    ----------
    row : mapping of str to float
        One game's full metrics as Team A recorded them: ``team_a_*`` and
        ``team_b_*`` columns and agents 0-9, Team A's agents first.

    Returns
    -------
    dict of str to float
        The same values with ``team_a`` and ``team_b`` exchanged and each agent
        index k moved to (k + 5) mod 10, so agents 0-4 are Team B's. Columns
        of neither team, such as ``episode_length``, are kept as they are.
    """

    def name(column: str) -> str:
        """Rename one column to Team B's side."""
        swapped = column.replace("team_a_", "\0").replace("team_b_", "team_a_")
        swapped = swapped.replace("\0", "team_b_")
        return _AGENT.sub(lambda m: f"agent_{(int(m.group(1)) + 5) % 10}", swapped)

    return {name(column): value for column, value in row.items()}


def per_tick_value(column: str, value: float, ticks: float) -> float:
    """Put one metric on the scale used to compare games of different lengths.

    Parameters
    ----------
    column : str
        A full-metric column of the catalog, such as ``team_a_kills``.
    value : float
        Its value in one game; NaN stays NaN.
    ticks : float
        The game's length in ticks (``episode_length``), above zero.

    Returns
    -------
    float
        value per 100 ticks for counts, health and ticks (``RATE_UNITS``);
        value itself for fractions, distances and other units.
    """
    if METRIC_COLUMNS_BY_NAME[column].unit in RATE_UNITS:
        return value / ticks * 100.0
    return value


def screen_games(
    games: Sequence[Row],
    references: Sequence[Sequence[Row]],
    columns: Sequence[str] = TEAM_COLUMNS,
) -> list[dict[str, Any]]:
    """Screen metrics of the person's team in each game against reference games.

    Parameters
    ----------
    games : sequence of dict
        The person's games: full metrics from Team A's side, with
        ``episode_length``.
    references : sequence of sequence of dict
        For each game, in the same order, the reference games to compare it
        with, each from the reference System's side, with ``episode_length``:
        the opponent's own wins on the same map for an against game, or the
        partner's own games against the same opponent on the same map and
        spawn end for an alongside game.
    columns : sequence of str, default=TEAM_COLUMNS
        Full-metric columns of Team A (``team_a_*`` or an agent's own
        ``agent_<k>_*`` column) to screen.

    Returns
    -------
    list of dict
        One row per column: ``measure`` (the column, without ``team_a_``),
        ``unit``, ``scale`` ("per 100 ticks" or "as is"), and for each game g
        (1, 2, ...) ``person_g`` (the person's team), ``other_side_g`` (the
        same column for the other team in that game; NaN for agent columns),
        ``reference_g`` (the mean over that game's references) and ``z_g``
        (the person's value minus that mean, divided by the references'
        standard deviation; NaN when they do not vary). ``same_direction`` is
        True when every game's z is finite and has the same sign;
        ``smallest_z`` is then the z nearest zero, with its sign, and 0
        otherwise. Rows are sorted by the size of ``smallest_z``, largest
        first.

    Notes
    -----
    A z of +2 means the person's team was two standard deviations above the
    references' mean. With few games and many metrics, some differences appear
    by chance; ``same_direction`` across every game is the stage's filter, not
    a test.
    """
    rows: list[dict[str, Any]] = []
    for column in columns:
        unit = METRIC_COLUMNS_BY_NAME[column].unit
        row: dict[str, Any] = {
            "measure": column.removeprefix("team_a_"),
            "unit": unit,
            "scale": "per 100 ticks" if unit in RATE_UNITS else "as is",
        }
        scores: list[float] = []
        for number, (game, reference) in enumerate(
            zip(games, references, strict=True), 1
        ):
            ticks = game["episode_length"]
            person = per_tick_value(column, game.get(column, math.nan), ticks)
            other = (
                per_tick_value(
                    column,
                    game.get(column.replace("team_a_", "team_b_", 1), math.nan),
                    ticks,
                )
                if column.startswith("team_a_")
                else math.nan
            )
            values = np.asarray(
                [
                    per_tick_value(
                        column, past.get(column, math.nan), past["episode_length"]
                    )
                    for past in reference
                ],
                np.float64,
            )
            values = values[np.isfinite(values)]
            mean = float(values.mean()) if len(values) else math.nan
            spread = float(values.std()) if len(values) else math.nan
            score = (person - mean) / spread if spread > 0 else math.nan
            row |= {
                f"person_{number}": person,
                f"other_side_{number}": other,
                f"reference_{number}": mean,
                f"z_{number}": score,
            }
            scores.append(score)
        finite = all(math.isfinite(score) for score in scores)
        same = finite and (all(s > 0 for s in scores) or all(s < 0 for s in scores))
        row["same_direction"] = same
        row["smallest_z"] = min(scores, key=abs) if same and scores else 0.0
        rows.append(row)
    return sorted(rows, key=lambda row: -abs(row["smallest_z"]))


def class_profile(
    games: Sequence[Row], focus: int | None = None
) -> dict[str, list[float]]:
    """Summarize who did what, class by class, for one team over its games.

    Parameters
    ----------
    games : sequence of dict
        Full metrics of games played with the canonical roster, each from the
        team's own side (agents 0-4 the team, 5-9 its enemies), with
        ``episode_length``. Missing values count as zero.
    focus : int or None, default=None
        A slot (0-4) whose distance to each teammate is added, such as the
        person's slot in alongside games; None adds nothing.

    Returns
    -------
    dict of str to list of float
        Five values each, in ``CLASSES`` order: ``deaths`` and
        ``damage_received`` (per 100 ticks over all the games), ``freedom_share``
        (the share of the Priest's Freedom uses each class received, itself
        included), ``mage_aura_coverage`` and ``warrior_aura_coverage`` (the
        share of each ally class's ticks inside that aura that it could have
        been inside; NaN for the aura giver and when nothing counts) and
        ``kills_by_victim`` (the share of the team's kills that hit each enemy
        class), ``healing_share`` (the share of the Priest's useful healing
        each class received) and, with ``focus``, ``distance_to_focus`` (the
        mean distance in map units between each class and the focus slot
        while both lived, weighted by those ticks; NaN for the focus itself).
        Shares are 0 when there is nothing to share.
    """

    def total(column: str) -> float:
        """Sum one column over the games; missing values add nothing."""
        values = [game.get(column, math.nan) for game in games]
        return float(np.nansum(np.asarray(values, np.float64)))

    ticks = max(total("episode_length"), 1e-9)
    freedom = np.asarray(
        [
            total(f"agent_4_to_agent_{ally}_priest_freedom_applications")
            for ally in range(5)
        ]
    )
    victims = np.asarray([total(f"agent_{enemy}_deaths") for enemy in range(5, 10)])
    healing = np.asarray(
        [total(f"agent_4_to_agent_{ally}_effective_healing_done") for ally in range(5)]
    )
    profile: dict[str, list[float]] = {
        "deaths": [total(f"agent_{k}_deaths") / ticks * 100 for k in range(5)],
        "damage_received": [
            total(f"agent_{k}_damage_received") / ticks * 100 for k in range(5)
        ],
        "freedom_share": (freedom / max(float(freedom.sum()), 1e-9)).tolist(),
        "kills_by_victim": (victims / max(float(victims.sum()), 1e-9)).tolist(),
        "healing_share": (healing / max(float(healing.sum()), 1e-9)).tolist(),
    }
    if focus is not None:
        apart: list[float] = []
        for other in range(5):
            pair = f"agent_{min(focus, other)}_and_agent_{max(focus, other)}"
            together = [
                (
                    game.get(f"{pair}_ally_distance_mean", math.nan),
                    game.get(f"{pair}_ally_distance_observations", math.nan),
                )
                for game in games
            ]
            together = [
                (mean, count)
                for mean, count in together
                if math.isfinite(mean) and math.isfinite(count)
            ]
            ticks_together = sum(count for _, count in together)
            apart.append(
                sum(mean * count for mean, count in together) / ticks_together
                if other != focus and ticks_together > 0
                else math.nan
            )
        profile["distance_to_focus"] = apart
    for giver, aura in ((0, "mage_aura"), (1, "warrior_aura")):
        cover: list[float] = []
        for ally in range(5):
            eligible = total(f"agent_{giver}_to_agent_{ally}_{aura}_eligible_steps")
            covered = total(f"agent_{giver}_to_agent_{ally}_{aura}_covered_steps")
            cover.append(
                covered / eligible if ally != giver and eligible > 0 else math.nan
            )
        profile[f"{aura}_coverage"] = cover
    return profile


def ultimate_timing(
    events: Sequence[Sequence[Any]], person_slots: Sequence[int], window_ticks: int
) -> list[dict[str, Any]]:
    """Measure when each agent used its Ultimate and what followed.

    Parameters
    ----------
    events : sequence of sequence
        One replay's ``transitions[i].events``, tick by tick
        (``load_replay(path).replay.transitions``). Events are read by their
        ``event_type``: ``cooldown_ready`` (``agent_global_slot``),
        ``ability_activated`` with ``ability_component`` "ultimate"
        (``source_global_slot``) and ``agent_died``
        (``recipient_global_slot``).
    person_slots : sequence of int
        Global slots (0-9) the person controlled.
    window_ticks : int
        Ticks after a use in which the user's team's kills are counted, at
        least 1.

    Returns
    -------
    list of dict
        One row per Ultimate use: ``slot``, ``class``, ``team`` ("A" or "B"),
        ``person`` (True for the person's slots), ``tick``, ``wait`` (ticks
        since the Ultimate was last ready; the first use of a game waits from
        tick 0, when every Ultimate starts ready), ``kills_after`` (enemies
        the user's team killed in the next window_ticks ticks) and
        ``kills_per_window`` (the team's kills in the whole game per
        window_ticks ticks: what an average window holds).
    """
    deaths = [
        (tick, event.recipient_global_slot)
        for tick, tick_events in enumerate(events)
        for event in tick_events
        if event.event_type == "agent_died"
    ]
    length = max(len(events), 1)
    kills = {
        team: sum(1 for _, slot in deaths if (slot >= 5) == (team == "A"))
        for team in ("A", "B")
    }
    ready = dict.fromkeys(range(10), 0)
    rows: list[dict[str, Any]] = []
    for tick, tick_events in enumerate(events):
        for event in tick_events:
            if event.event_type == "cooldown_ready":
                ready[event.agent_global_slot] = tick
            elif (
                event.event_type == "ability_activated"
                and event.ability_component == "ultimate"
            ):
                slot = event.source_global_slot
                team = "A" if slot < 5 else "B"
                rows.append(
                    {
                        "slot": slot,
                        "class": CLASSES[slot % 5],
                        "team": team,
                        "person": slot in person_slots,
                        "tick": tick,
                        "wait": tick - ready[slot],
                        "kills_after": sum(
                            1
                            for death, victim in deaths
                            if tick < death <= tick + window_ticks
                            and (victim >= 5) == (team == "A")
                        ),
                        "kills_per_window": kills[team] / length * window_ticks,
                    }
                )
    return rows


def compare_alongside(game: Row, reference: Sequence[Row]) -> dict[str, Any]:
    """Compare one alongside game with the partner's own games.

    Parameters
    ----------
    game : dict
        The alongside game's full metrics from Team A's side (the person and
        the partner), with ``episode_length``.
    reference : sequence of dict
        The partner's own games against the same opponent on the same map and
        spawn end, each from the partner's side (``score_difference``,
        ``team_a_kills``, ``team_a_deaths`` and ``team_a_win``, ``_draw`` and
        ``_loss``).

    Returns
    -------
    dict
        ``games`` (reference games), the person's team's ``margin``, ``kills``
        and ``deaths`` with the reference ``*_mean``, ``*_low`` and ``*_high``
        (the smallest and largest reference values), the reference ``wins``,
        ``draws`` and ``losses``, and ``margin_rank`` (the share of reference
        games with a smaller margin, ties counting half).
    """
    record: dict[str, Any] = {"games": len(reference)}
    for name, column in (
        ("margin", "score_difference"),
        ("kills", "team_a_kills"),
        ("deaths", "team_a_deaths"),
    ):
        values = np.asarray([row[column] for row in reference], np.float64)
        record[name] = game[column]
        record[f"{name}_mean"] = float(values.mean()) if len(values) else math.nan
        record[f"{name}_low"] = float(values.min()) if len(values) else math.nan
        record[f"{name}_high"] = float(values.max()) if len(values) else math.nan
    margins = np.asarray([row["score_difference"] for row in reference], np.float64)
    margin = float(game["score_difference"])
    record["margin_rank"] = (
        float(np.mean(margins < margin))
        + 0.5 * float(np.count_nonzero(margins == margin) / len(margins))
        if len(margins)
        else math.nan
    )
    for result, name in (("win", "wins"), ("draw", "draws"), ("loss", "losses")):
        record[name] = int(sum(row[f"team_a_{result}"] for row in reference))
    return record


def _read_game(folder: Path) -> tuple[dict[str, Any], Row, Path]:
    """Read one recorded game: game.json, its full metrics row and its replay."""
    details = json.loads((folder / "game.json").read_text(encoding="utf-8"))
    with (folder / "full_metrics.csv").open(newline="", encoding="utf-8") as file:
        raw = next(csv.DictReader(file))
    row: Row = {}
    for column, text in raw.items():
        try:
            row[column] = float(text)
        except ValueError:
            continue
    return details, row, folder / details["replay"]


def _rows(table: Mapping[str, Any]) -> list[Row]:
    """Turn a column table into one dict per row: numbers as float, else text.

    Cells are converted one by one, because a column of mixed values comes
    back as an object array; None becomes NaN only in number columns' cells.
    """
    count = len(next(iter(table.values()))) if table else 0

    def cell(value: object) -> object:
        """A number as float; a bool, text or None as it is."""
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, (bool, np.bool_)
        ):
            return float(cast(float, value))
        return value

    return [
        {name: cell(values[index]) for name, values in table.items()}
        for index in range(count)
    ]


def write_outputs(record: Mapping[str, Any], output_dir: Path) -> None:
    """Write the stage's CSV files, summary.md and analysis.json.

    Parameters
    ----------
    record : mapping
        ``main``'s record: ``against_screen``, ``alongside_screen``,
        ``classes``, ``ultimates`` and ``alongside`` rows, ``games`` and
        ``settings``.
    output_dir : pathlib.Path
        Existing folder; files with these names are replaced.
    """

    def table(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Write one list of dicts as a CSV file, rounding floats to 4 places."""
        if not rows:
            return
        with (output_dir / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: round(value, 4) if isinstance(value, float) else value
                        for key, value in row.items()
                    }
                )

    table("against_screen.csv", record["against_screen"])
    table("alongside_screen.csv", record["alongside_screen"])
    table("classes.csv", record["classes"])
    table("ultimates.csv", record["ultimates"])
    table("alongside.csv", record["alongside"])
    lines = ["# The Person's Games Beside The Systems' Own", ""]
    lines.append(
        "Found by screening many metrics in a few games: a description of these "
        "games, not a test."
    )
    for key, title, reference in (
        ("against_screen", "Against games", "the opponent's wins on that map"),
        (
            "alongside_screen",
            "Alongside games",
            "the partner's own games on that map and spawn end",
        ),
    ):
        screen = [row for row in record[key] if row["same_direction"]]
        if not screen:
            continue
        count = sum(1 for name in screen[0] if name.startswith("z_"))
        lines += [
            "",
            f"**{title}: differences that point the same way in every game** "
            f"(z: the person's team minus the mean of {reference}, in standard "
            "deviations of those games)",
            "",
            "| Measure | Scale | "
            + " | ".join(
                f"Person {g} | Reference {g} | z {g}" for g in range(1, count + 1)
            )
            + " |",
            "| --- | --- |" + " ---: | ---: | ---: |" * count,
        ]
        for row in screen[:25]:
            cells = " | ".join(
                f"{row[f'person_{g}']:.3g} | {row[f'reference_{g}']:.3g} | "
                f"{row[f'z_{g}']:+.1f}"
                for g in range(1, count + 1)
            )
            lines.append(f"| {row['measure']} | {row['scale']} | {cells} |")
    if record["classes"]:
        lines += ["", "**Class by class** (Mage, Warrior, Hunter, Rogue, Priest)", ""]
        lines.append("| Team | Table | " + " | ".join(CLASSES) + " |")
        lines.append("| --- | --- |" + " ---: |" * 5)
        for row in record["classes"]:
            cells = " | ".join(f"{row[name]:.3g}" for name in CLASSES)
            lines.append(f"| {row['team']} | {row['table']} | {cells} |")
    if record["ultimates"]:
        lines += [
            "",
            "**Ultimates** (kills by the user's team in the next window, beside an "
            "average window)",
            "",
            "| Games | Who | Class | Uses | Median wait after ready "
            "| Kills after, per use | Kills in an average window |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in ultimate_summary(record["ultimates"]):
            lines.append(
                f"| {row['kind']} | {row['who']} | {row['class']} | {row['uses']} | "
                f"{row['median_wait']:.1f} | {row['kills_after_per_use']:.2f} | "
                f"{row['kills_per_window']:.2f} |"
            )
    if record["alongside"]:
        lines += [
            "",
            "**Alongside games beside the partner's own games** (same opponent, map "
            "and spawn end)",
            "",
            "| Game | Map | Team A starts | Margin "
            "| Partner alone: mean (lowest to highest) "
            "| Share of partner games with a smaller margin | Partner W-D-L |",
            "| ---: | ---: | --- | ---: | ---: | ---: | :---: |",
        ]
        for row in record["alongside"]:
            lines.append(
                f"| {row['number']} | {row['map_id']} | {row['side']} | "
                f"{row['margin']:+.0f} | {row['margin_mean']:+.1f} "
                f"({row['margin_low']:+.0f} to {row['margin_high']:+.0f}) | "
                f"{row['margin_rank']:.2f} | "
                f"{row['wins']}-{row['draws']}-{row['losses']} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "analysis.json").write_text(
        json.dumps({"settings": record["settings"], "games": record["games"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def ultimate_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pool ``ultimate_timing`` rows by who controlled the agent and its class.

    Parameters
    ----------
    rows : sequence of mapping
        ``ultimate_timing`` rows with ``kind`` (the game's kind) and ``who``
        (the person, or the System's display name) added.

    Returns
    -------
    list of dict
        One row per (kind, who, class) in first-seen order: ``uses``,
        ``median_wait`` (ticks), ``kills_after_per_use`` and
        ``kills_per_window`` (the mean of the games' average windows over the
        uses, so each use is compared with its own game).
    """
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["kind"], row["who"], row["class"]), []).append(row)
    return [
        {
            "kind": game_kind,
            "who": who,
            "class": kind,
            "uses": len(group),
            "median_wait": float(np.median([row["wait"] for row in group])),
            "kills_after_per_use": float(
                np.mean([row["kills_after"] for row in group])
            ),
            "kills_per_window": float(
                np.mean([row["kills_per_window"] for row in group])
            ),
        }
        for (game_kind, who, kind), group in groups.items()
    ]


@hydra.main(
    version_base="1.3", config_path="configs", config_name="human_games_analysis"
)
def main(config: DictConfig) -> dict[str, Any]:
    """Read a human games series and the field's games, and compare them.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``series_dir`` (a human_games output folder), ``field_dir`` (a saved
        run with full metrics for every game of the Systems' field), ``names``
        (display name for each System as the field run records it; the series
        uses display names), ``window_ticks`` (ticks after an Ultimate in which
        kills count) and the required ``output_dir``. Relative paths use the
        folder the command was started from.

    Returns
    -------
    dict
        ``against_screen``, ``alongside_screen``, ``classes``, ``ultimates``
        and ``alongside`` rows (see ``screen_games``, ``class_profile``,
        ``ultimate_timing`` and
        ``compare_alongside``), ``games`` (each recorded game's number, kind,
        map, side, opponent, partner and result) and ``settings``.

    Raises
    ------
    ValueError
        The series has no recorded game, or a System the series names is not
        in ``names``.

    Notes
    -----
    Host-only: NumPy and the replay loader, no JAX and no policy calls. Games
    whose folder has no game.json (not yet played) are skipped. Writes only to
    output_dir.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation import load_replay

    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.

    def path(value: str) -> Path:
        """Resolve a setting's path from the folder the command started in."""
        return Path(to_absolute_path(str(Path(value).expanduser())))

    output_dir = path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    recorded = {display: name for name, display in settings["names"].items()}
    window = int(settings["window_ticks"])

    # 1. The recorded games.
    series = [
        _read_game(folder)
        for folder in sorted(path(settings["series_dir"]).glob("game-*"))
        if (folder / "game.json").exists()
    ]
    if not series:
        raise ValueError("the series has no recorded game")
    wanted = {
        details[key]
        for details, _, _ in series
        for key in ("opponent", "partner")
        if details.get(key)
    }
    unknown = sorted(wanted - set(recorded))
    if unknown:
        raise ValueError(f"names gives no recorded name for {unknown}")

    def person_slots(details: Mapping[str, Any]) -> tuple[int, ...]:
        """The Team A slots the person controlled in one recorded game."""
        if details["kind"] == "against":
            return tuple(range(5))
        return (CLASSES.index(str(details["player_class"]).capitalize()),)

    alongside_slots = sorted(
        {
            slot
            for details, _, _ in series
            if details["kind"] == "alongside"
            for slot in person_slots(details)
        }
    )

    # 2. The field's games of those Systems, each from its own side.
    field = marl_bgs.load_results(path(settings["field_dir"]))
    columns = [
        "score_difference",
        "episode_length",
        "team_a_win",
        "team_a_draw",
        "team_a_loss",
        *TEAM_COLUMNS,
        *CLASS_COLUMNS,
        *(column for slot in alongside_slots for column in agent_columns(slot)),
    ]
    table = field.system_metrics(
        "full_metrics",
        by="game",
        columns=list(dict.fromkeys(columns)),
        systems=sorted(recorded[name] for name in wanted),
    )
    reference = _rows(table)
    for row in reference:
        row["system"] = settings["names"].get(row["system_name"], row["system_name"])
        row["opponent"] = settings["names"].get(
            row["opponent_name"], row["opponent_name"]
        )

    # 3-5. Against games.
    against = [game for game in series if game[0]["kind"] == "against"]
    against_screen: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    ultimates: list[dict[str, Any]] = []
    if against:
        opponent = against[0][0]["opponent"]
        wins = [
            row
            for row in reference
            if row["system"] == opponent and row["team_a_win"] == 1
        ]
        maps = [int(details["map_id"]) for details, _, _ in against]
        against_screen = screen_games(
            [row for _, row, _ in against],
            [[row for row in wins if row["map_id"] == map_id] for map_id in maps],
        )
        teams = {
            "Person": [row for _, row, _ in against],
            f"{opponent} in the same games": [mirrored(row) for _, row, _ in against],
            f"{opponent}, its wins on the same maps": [
                row for row in wins if row["map_id"] in maps
            ],
        }
        for team, games in teams.items():
            for name, values in class_profile(games).items():
                classes.append(
                    {"team": team, "table": name}
                    | dict(zip(CLASSES, values, strict=True))
                )
    for details, _, replay in series:
        events = [
            transition.events for transition in load_replay(replay).replay.transitions
        ]
        for row in ultimate_timing(events, person_slots(details), window):
            who = (
                "Person"
                if row["person"]
                else details["partner"]
                if row["team"] == "A"
                else details["opponent"]
            )
            ultimates.append(
                {"number": details["number"], "kind": details["kind"], "who": who} | row
            )

    # 6. Alongside games.
    alongside: list[dict[str, Any]] = []
    alongside_games: list[Row] = []
    alongside_references: list[list[Row]] = []
    alongside_slot = 0
    for details, row, _ in series:
        if details["kind"] != "alongside":
            continue
        partner_games = [
            game
            for game in reference
            if game["system"] == details["partner"]
            and game["opponent"] == details["opponent"]
            and game["map_id"] == details["map_id"]
            # Team A's original spawn bank is the left one.
            and game["spawn_end"]
            == ("team_a" if details["side"] == "left" else "team_b")
        ]
        alongside.append(
            {
                key: details[key]
                for key in ("number", "map_id", "side", "partner", "opponent")
            }
            | compare_alongside(row, partner_games)
        )
        alongside_games.append(row)
        alongside_references.append(partner_games)
        alongside_slot = person_slots(details)[0]
    if alongside_games:
        partner = alongside[0]["partner"]
        teams = {
            f"Person beside {partner}": alongside_games,
            f"{partner} alone, same maps and spawn ends": [
                game for games in alongside_references for game in games
            ],
        }
        for team, games in teams.items():
            for name, values in class_profile(games, focus=alongside_slot).items():
                classes.append(
                    {"team": team, "table": name}
                    | dict(zip(CLASSES, values, strict=True))
                )
    alongside_screen = (
        screen_games(
            alongside_games,
            alongside_references,
            [
                *TEAM_COLUMNS,
                *(column for slot in alongside_slots for column in agent_columns(slot)),
            ],
        )
        if alongside_games
        else []
    )

    record: dict[str, Any] = {
        "against_screen": against_screen,
        "alongside_screen": alongside_screen,
        "classes": classes,
        "ultimates": ultimates,
        "alongside": alongside,
        "games": [
            {
                key: details.get(key)
                for key in (
                    "number",
                    "kind",
                    "map_id",
                    "side",
                    "opponent",
                    "partner",
                    "player_class",
                    "outcome",
                    "team_a_score",
                    "team_b_score",
                )
            }
            for details, _, _ in series
        ],
        "settings": settings,
    }
    write_outputs(record, output_dir)
    print(output_dir / "summary.md")
    return record


if __name__ == "__main__":
    prepare_hydra()
    main()
