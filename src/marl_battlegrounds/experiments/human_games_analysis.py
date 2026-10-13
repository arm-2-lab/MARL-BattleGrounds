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
   and again against only its wins by lead_margin points or more (a team as
   far ahead as the person was), and keep the differences that point the same
   way in every game. Counts, health and ticks are compared per 100 ticks,
   because games end at different times; fractions and distances are compared
   as they are. Each measure also gets the person's lowest and highest game,
   the middle half of all the reference games and how many of them reach the
   person's level.
4. Class tables for the person's team, the opponent in the same games and the
   opponent's wins: who died, who took damage, who received the Priest's
   Freedom, how much of each ally class's time was inside the Mage's and the
   Warrior's auras, and which enemy classes each side killed.
5. Ultimate timing from the replays: for each class and side, how long after
   its Ultimate was ready the agent used it, how many enemies its team killed
   in the next window_ticks ticks, beside the team's kills in an average
   window of that length, and how often it came within SYNC_TICKS ticks of a
   teammate's Ultimate (for example, the Mage's Burst with the Warrior's
   Charge), beside how often that happens by luck: each agent's Ultimate times
   slid at random within its game, chance_shuffles times. With the replay's
   action mask, each use also says whether it came on the first tick the
   agent could use it and how many teammates could have joined.
7. Head to head in the against games, choice by choice against what the game
   allowed: Ultimates used the first tick they could be, Bursts pressed with an
   enemy in reach and the Mage's damage during Burst against its own normal
   rate, the Priest's Basic heals to hurt teammates and its answer to a
   teammate in danger, and each team's attack targets against an even pick.
6. Alongside games (the person controls one class beside a partner System):
   compare each game with the partner's own games against the same opponent on
   the same map and spawn end, as a result (point margin, kills, deaths) and
   as a screen of every team-level metric and of the person's own agent.

Outputs in output_dir: against_screen.csv and against_screen_lead.csv (step 3),
classes.csv (step 4), ultimates.csv and ultimates_together.csv (step 5),
alongside.csv and alongside_screen.csv (step 6), head_to_head.csv (step 7),
field_record.csv (the opposing System's own record in the field on the
person's maps: games, wins, draws, losses and its largest loss), summary.md
(readable tables) and analysis.json (settings and inputs). With
three games per kind, every result is a description of these games, found by
screening many metrics, not a test: a pattern must be checked in more games
before it is a claim.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

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

# A head-to-head measure: chosen game records and a team ("A" or "B") to a value.
type GameMeasure = Callable[[Sequence[Mapping[str, Any]], str], float]

# Two Ultimates count as used together when they are this many ticks apart or
# fewer.
SYNC_TICKS = 4

# Ratios screened beside the team columns, per game: name to (numerator,
# denominator), NaN when the denominator is zero. Damage per Burst; the damage
# the Warrior's aura prevented per point of damage taken (as in
# tournament_analysis); and the Priest's healing to teammates slowed by the
# enemy Hunter or the enemy Warrior's Charge, per tick a teammate spent slowed.
DERIVED = {
    "team_a_burst_damage_per_use": (
        "team_a_burst_damage",
        "team_a_mage_burst_applications",
    ),
    "team_a_warrior_aura_protection_share": (
        "team_a_damage_prevented_by_warrior_aura",
        "team_a_damage_received",
    ),
    "team_a_hunter_slowed_ally_healing_per_tick": (
        "team_a_healing_to_hunter_basic_slow_recipient",
        "team_a_hunter_basic_slow_active_steps",
    ),
    "team_a_charge_slowed_ally_healing_per_tick": (
        "team_a_healing_to_warrior_charge_slow_recipient",
        "team_a_warrior_charge_slow_active_steps",
    ),
}

# Canonical roster order: slot k plays CLASSES[k] on each team.
CLASSES = ("Mage", "Warrior", "Hunter", "Rogue", "Priest")

# A teammate counts as in danger at or below this share of its maximum health,
# and the Priest has the use tick and this many ticks after it to heal it.
DANGER_FRACTION = 0.35
DANGER_TICKS = 2


class GameArrays(NamedTuple):
    """One replay's per-tick state, as NumPy arrays, for the choice measures.

    Row t is the state at the start of tick t, when the agents chose that
    tick's actions; there is one more row than ticks. Slots are global (0-4
    Team A, 5-9 Team B) and follow ``CLASSES``.

    Attributes
    ----------
    alive : numpy.ndarray
        Bool (ticks + 1, 10): the agent is alive.
    health : numpy.ndarray
        Float (ticks + 1, 10): current health.
    burst_left : numpy.ndarray
        Int (ticks + 1, 10): ticks of the Mage's Burst boost still to run; 0
        when no Burst is active, and always 0 for other classes.
    choices : numpy.ndarray
        Bool (ticks + 1, 10, 11, 2): the game's action mask for exact target
        and Ultimate pairs, ``[actor, target, Ultimate]``. Target 0 is no
        target, 1-5 the actor's own team's rows 0-4 (itself included) and 6-10
        the other team's rows 0-4; Ultimate 1 uses it.
    max_health : numpy.ndarray
        Float (10,): each slot's maximum health.
    """

    alive: np.ndarray
    health: np.ndarray
    burst_left: np.ndarray
    choices: np.ndarray
    max_health: np.ndarray


def game_arrays(frames: Sequence[Any], max_health: Sequence[float]) -> GameArrays:
    """Stack one replay's frames into ``GameArrays``.

    Parameters
    ----------
    frames : sequence
        ``load_replay(path).replay.frames``: each with ``snapshot``
        (``alive_mask``, ``current_health``,
        ``mage_burst_damage_amplification_durations``) and ``action_mask``
        (``select_target_use_ultimate_joint_mask``).
    max_health : sequence of float
        Ten values, each slot's maximum health, in slot order.

    Returns
    -------
    GameArrays
        The frames' state, one row per frame.
    """
    return GameArrays(
        alive=np.asarray([f.snapshot.alive_mask for f in frames], bool),
        health=np.asarray([f.snapshot.current_health for f in frames], np.float64),
        burst_left=np.asarray(
            [f.snapshot.mage_burst_damage_amplification_durations for f in frames],
            np.int64,
        ),
        choices=np.asarray(
            [f.action_mask.select_target_use_ultimate_joint_mask for f in frames], bool
        ),
        max_health=np.asarray(max_health, np.float64),
    )


def _rows_of(team: str) -> tuple[list[int], list[int]]:
    """A team's own slots and the other team's slots, each in roster order."""
    own = list(range(5)) if team == "A" else list(range(5, 10))
    return own, [slot for slot in range(10) if slot not in own]


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

# Agent columns behind the class tables: each agent's own columns for both
# teams (0-9), and the Priest's and the aura givers' columns toward teammates.
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
        A full-metric column of the catalog, such as ``team_a_kills``, or a
        ``DERIVED`` ratio (kept as it is).
    value : float
        Its value in one game; NaN stays NaN.
    ticks : float
        The game's length in ticks (``episode_length``), above zero.

    Returns
    -------
    float
        value per 100 ticks for per-game totals: a unit in ``RATE_UNITS``,
        no summary weight in the catalog and no "mean" in the name; value
        itself for fractions, distances and averages (such as a mean respawn
        wait).
    """
    return value / ticks * 100.0 if per_game_total(column) else value


def per_game_total(column: str) -> bool:
    """Tell whether a column is a per-game total, compared per 100 ticks.

    Parameters
    ----------
    column : str
        A catalog column or a ``DERIVED`` ratio.

    Returns
    -------
    bool
        True for a catalog column with a unit in ``RATE_UNITS``, no summary
        weight and no "mean" in its name; False for fractions, distances,
        averages and every ``DERIVED`` ratio.
    """
    info = METRIC_COLUMNS_BY_NAME.get(column)
    return (
        info is not None
        and info.unit in RATE_UNITS
        and info.summary_weight is None
        and "mean" not in column.removeprefix("team_a_")
    )


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
        ``agent_<k>_*`` column), or a ``DERIVED`` ratio, to screen. A derived
        ratio is computed per game and stored in each game's dict.

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
        otherwise. Pooled over all games: ``person_low`` and ``person_high``
        (the person's lowest and highest game), ``reference_games`` (the
        distinct reference games with a value, each counted once even when
        several of the person's games share it), their ``reference_q25``,
        ``reference_median`` and ``reference_q75`` (25th, 50th and 75th
        percentiles: the middle half of the references lies between q25 and
        q75), ``reference_at_person_level`` (how many of those reference games
        were at least as far in the person's direction as the person's least
        extreme game; NaN unless ``same_direction``), and for per-game totals
        ``person_sum`` and ``reference_sum`` (the raw totals, not per tick,
        such as 14 rescues of 15 chances; NaN for other measures). Values are
        on the compared scale (per 100 ticks or as is); NaN where nothing has
        a value. Rows are sorted by the size of ``smallest_z``, largest first.

    Notes
    -----
    A z of +2 means the person's team was two standard deviations above the
    references' mean. With few games and many metrics, some differences appear
    by chance; ``same_direction`` across every game is the stage's filter, not
    a test.
    """
    pooled_references = list(
        {id(past): past for reference in references for past in reference}.values()
    )
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column in DERIVED:
            top, bottom = DERIVED[column]
            for game in [*games, *(past for group in references for past in group)]:
                below = game.get(bottom, math.nan)
                game[column] = game.get(top, math.nan) / below if below else math.nan
            other = column.replace("team_a_", "team_b_", 1)
            top_b, bottom_b = (
                name.replace("team_a_", "team_b_", 1) for name in DERIVED[column]
            )
            for game in games:
                below = game.get(bottom_b, math.nan)
                game[other] = game.get(top_b, math.nan) / below if below else math.nan
        unit = METRIC_COLUMNS_BY_NAME[column].unit if column not in DERIVED else "ratio"
        row: dict[str, Any] = {
            "measure": column.removeprefix("team_a_"),
            "unit": unit,
            "scale": "per 100 ticks" if per_game_total(column) else "as is",
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
        row |= _pooled(column, games, pooled_references, scores if same else [])
        rows.append(row)
    return sorted(rows, key=lambda row: -abs(row["smallest_z"]))


def _pooled(
    column: str,
    games: Sequence[Row],
    references: Sequence[Row],
    scores: Sequence[float],
) -> dict[str, float]:
    """Summarize one screened measure over all of the person's games at once.

    Parameters
    ----------
    column : str
        The screened column or ``DERIVED`` ratio. A derived value must already
        be stored in every game, as ``screen_games`` does before calling this.
    games : sequence of dict
        The person's games, each with ``episode_length``.
    references : sequence of dict
        The distinct reference games of all the person's games, each with
        ``episode_length``.
    scores : sequence of float
        The person's z in each game when they all share one sign, otherwise
        empty. Its sign sets the direction for ``reference_at_person_level``.

    Returns
    -------
    dict of str to float
        ``person_low``, ``person_high``, ``reference_games``,
        ``reference_q25``, ``reference_median``, ``reference_q75``,
        ``reference_at_person_level``, ``person_sum`` and ``reference_sum``,
        as ``screen_games`` describes them. Missing values are left out; a
        summary of nothing is NaN, and a count of nothing is 0.
    """

    def scaled(rows: Sequence[Row]) -> np.ndarray:
        """The measure in each game on the compared scale, missing ones dropped."""
        values = np.asarray(
            [
                per_tick_value(column, row.get(column, math.nan), row["episode_length"])
                for row in rows
            ],
            np.float64,
        )
        return values[np.isfinite(values)]

    def raw_sum(rows: Sequence[Row]) -> float:
        """The raw total of a per-game total over the games; NaN otherwise."""
        values = np.asarray([row.get(column, math.nan) for row in rows], np.float64)
        values = values[np.isfinite(values)]
        return (
            float(values.sum()) if per_game_total(column) and len(values) else math.nan
        )

    person = scaled(games)
    reference = scaled(references)
    q25, median, q75 = (
        np.quantile(reference, [0.25, 0.5, 0.75]).tolist()
        if len(reference)
        else [math.nan] * 3
    )
    level = math.nan
    if scores and len(person):
        level = float(
            np.count_nonzero(
                reference >= person.min()
                if scores[0] > 0
                else reference <= person.max()
            )
        )
    return {
        "person_low": float(person.min()) if len(person) else math.nan,
        "person_high": float(person.max()) if len(person) else math.nan,
        "reference_games": float(len(reference)),
        "reference_q25": q25,
        "reference_median": median,
        "reference_q75": q75,
        "reference_at_person_level": level,
        "person_sum": raw_sum(games),
        "reference_sum": raw_sum(references),
    }


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
    events: Sequence[Sequence[Any]],
    person_slots: Sequence[int],
    window_ticks: int,
    arrays: GameArrays | None = None,
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
    arrays : GameArrays or None, default=None
        The same replay's per-tick state (``game_arrays``). With it, each row
        also says what the game allowed; None leaves those columns NaN.

    Returns
    -------
    list of dict
        One row per Ultimate use: ``slot``, ``class``, ``team`` ("A" or "B"),
        ``person`` (True for the person's slots), ``tick``, ``wait`` (ticks
        from the first tick the Ultimate could be used again, the tick after
        its ``cooldown_ready`` event, to the use; the first use of a game waits
        from tick 0, when every Ultimate starts ready; time dead counts),
        ``usable_wait`` (of those ticks, the ones on which the game's action
        mask let the agent use its Ultimate: alive, not stunned, with a target
        in reach), ``first_usable`` (True when ``usable_wait`` is 0: used on
        the first tick it could be), ``mates_usable`` (how many teammates could
        have used their own Ultimate on the use tick), ``enemy_attackable``
        (True when the user could have attacked an enemy with its Basic on the
        use tick: for the Mage, Burst pressed with an enemy in range),
        ``kills_after`` (enemies the user's team killed in the next
        window_ticks ticks, not counting the use's own tick),
        ``kills_per_window`` (the team's kills in the whole game per
        window_ticks ticks: what an average window holds) and
        ``teammate_gap`` (ticks to the nearest Ultimate by another agent of
        the same team, before or after; NaN if none) with ``warrior_gap``
        (the same, to the team's Warrior's Ultimate, Charge; NaN for the
        Warrior itself or with none).
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
                # The event fires on the tick the cooldown runs out; the
                # Ultimate can be used from the next tick on.
                ready[event.agent_global_slot] = tick + 1
            elif (
                event.event_type == "ability_activated"
                and event.ability_component == "ultimate"
            ):
                slot = event.source_global_slot
                team = "A" if slot < 5 else "B"
                row: dict[str, Any] = {
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
                row |= _allowed(arrays, slot, ready[slot], tick)
                rows.append(row)
    for row in rows:
        others = [
            other["tick"]
            for other in rows
            if other["team"] == row["team"] and other["slot"] != row["slot"]
        ]
        warrior = [
            other["tick"]
            for other in rows
            if other["team"] == row["team"]
            and other["class"] == "Warrior"
            and other["slot"] != row["slot"]
        ]
        row["teammate_gap"] = (
            float(min(abs(tick - row["tick"]) for tick in others))
            if others
            else math.nan
        )
        row["warrior_gap"] = (
            float(min(abs(tick - row["tick"]) for tick in warrior))
            if warrior
            else math.nan
        )
    return rows


def _allowed(
    arrays: GameArrays | None, slot: int, ready: int, tick: int
) -> dict[str, Any]:
    """What the game allowed around one Ultimate use, from the action mask.

    Parameters
    ----------
    arrays : GameArrays or None
        The replay's per-tick state; None gives NaN for every value.
    slot : int
        The user's global slot.
    ready : int
        The first tick the Ultimate could be used again (0 for a game's first
        use).
    tick : int
        The use tick.

    Returns
    -------
    dict
        ``usable_wait``, ``first_usable``, ``mates_usable`` and
        ``enemy_attackable``, as ``ultimate_timing`` describes them.
    """
    if arrays is None:
        return dict.fromkeys(
            ("usable_wait", "first_usable", "mates_usable", "enemy_attackable"),
            math.nan,
        )
    usable: np.ndarray = np.any(arrays.choices[:, :, :, 1], axis=2)  # (ticks + 1, 10)
    own, _ = _rows_of("A" if slot < 5 else "B")
    wait = int(np.count_nonzero(usable[ready:tick, slot]))
    return {
        "usable_wait": wait,
        "first_usable": wait == 0,
        "mates_usable": int(
            sum(bool(usable[tick, mate]) for mate in own if mate != slot)
        ),
        "enemy_attackable": bool(arrays.choices[tick, slot, 6:11, 0].any()),
    }


def _damage_by_tick(events: Sequence[Sequence[Any]]) -> dict[int, np.ndarray]:
    """Damage each slot dealt on each tick: slot to a (ticks,) float array.

    Reads ``source_damage_output`` events; damage dealt is the source's
    modified output times the recipient's modifier, including damage beyond
    the recipient's remaining health.
    """
    dealt = {slot: np.zeros(len(events)) for slot in range(10)}
    for tick, tick_events in enumerate(events):
        for event in tick_events:
            if event.event_type == "source_damage_output":
                dealt[event.source_global_slot][tick] += (
                    event.source_modified_damage_output
                    * event.recipient_damage_modifier
                )
    return dealt


def burst_value(
    events: Sequence[Sequence[Any]], arrays: GameArrays
) -> dict[str, dict[str, float]]:
    """Measure what each team's Mage got from its Bursts in one game.

    A Burst's window is its use tick and the following ticks on which the
    boost was active (``arrays.burst_left`` above zero at the start of the
    tick). The use tick counts because pressing Burst spends that tick's
    action, so the window shows what pressing it bought.

    Parameters
    ----------
    events : sequence of sequence
        The replay's events tick by tick: ``ability_activated`` (Ultimate uses
        by the Mages, slots 0 and 5) and ``source_damage_output``.
    arrays : GameArrays
        The same replay's per-tick state.

    Returns
    -------
    dict
        "A" and "B" to: ``bursts``, ``window_ticks`` and ``window_damage``
        (over all Burst windows), ``other_ticks`` and ``other_damage`` (the
        Mage's living ticks outside every window), and ``opening_damage``
        (damage in the game's first Burst window; NaN with no Burst).
    """
    dealt = _damage_by_tick(events)
    ticks = len(events)
    result: dict[str, dict[str, float]] = {}
    for team, mage in (("A", 0), ("B", 5)):
        uses = [
            tick
            for tick, tick_events in enumerate(events)
            for event in tick_events
            if event.event_type == "ability_activated"
            and event.ability_component == "ultimate"
            and event.source_global_slot == mage
        ]
        inside = np.zeros(ticks, bool)
        windows: list[range] = []
        for use in uses:
            end = use + 1
            while end < ticks and arrays.burst_left[end, mage] > 0:
                end += 1
            windows.append(range(use, end))
            inside[use:end] = True
        living = arrays.alive[:ticks, mage] & ~inside
        result[team] = {
            "bursts": float(len(uses)),
            "window_ticks": float(np.count_nonzero(inside)),
            "window_damage": float(dealt[mage][inside].sum()),
            "other_ticks": float(np.count_nonzero(living)),
            "other_damage": float(dealt[mage][living].sum()),
            "opening_damage": (
                float(dealt[mage][windows[0].start : windows[0].stop].sum())
                if windows
                else math.nan
            ),
        }
    return result


def heal_choices(
    events: Sequence[Sequence[Any]], arrays: GameArrays, heal_amount: float
) -> dict[str, dict[str, float]]:
    """Count how often each team's Priest sent its Basic heal to a hurt teammate.

    Parameters
    ----------
    events : sequence of sequence
        The replay's events tick by tick; ``ability_activated`` with
        ``ability_component`` "basic" from the Priests (slots 4 and 9).
    arrays : GameArrays
        The same replay's per-tick state.
    heal_amount : float
        The Basic heal's size. A teammate is hurt when it is missing at least
        this much health, so the whole heal would count.

    Returns
    -------
    dict
        "A" and "B" to: ``heals`` (Basic heals on the Priest's own team),
        ``with_hurt_in_reach`` (of those, heals made while the action mask let
        the Priest heal at least one hurt, living teammate, itself included)
        and ``to_hurt`` (of those, heals that went to a hurt teammate).
    """
    result: dict[str, dict[str, float]] = {}
    for team in ("A", "B"):
        own, _ = _rows_of(team)
        priest = own[4]
        counts: dict[str, float] = dict.fromkeys(
            ("heals", "with_hurt_in_reach", "to_hurt"), 0.0
        )
        for tick, tick_events in enumerate(events):
            for event in tick_events:
                if not (
                    event.event_type == "ability_activated"
                    and event.ability_component == "basic"
                    and event.source_global_slot == priest
                    and event.recipient_global_slot in own
                ):
                    continue
                counts["heals"] += 1
                missing = arrays.max_health[own] - arrays.health[tick, own]
                hurt = (missing >= heal_amount) & arrays.alive[tick, own]
                if (hurt & arrays.choices[tick, priest, 1:6, 0]).any():
                    counts["with_hurt_in_reach"] += 1
                    counts["to_hurt"] += float(
                        hurt[own.index(event.recipient_global_slot)]
                    )
        result[team] = counts
    return result


def danger_responses(
    events: Sequence[Sequence[Any]], arrays: GameArrays
) -> dict[str, dict[str, float]]:
    """Count how often each team's Priest healed a teammate who fell into danger.

    A teammate falls into danger on tick t when it starts the tick alive at
    ``DANGER_FRACTION`` of its maximum health or less, having started the tick
    before above it. It counts only while its own Priest is alive at t, and is
    answered when that Priest heals it (Basic or Ultimate) on tick t or the
    ``DANGER_TICKS`` ticks after.

    Parameters
    ----------
    events : sequence of sequence
        The replay's events tick by tick; ``ability_activated`` from the
        Priests (slots 4 and 9).
    arrays : GameArrays
        The same replay's per-tick state.

    Returns
    -------
    dict
        "A" and "B" to ``falls`` (non-Priest teammates falling into danger),
        ``in_reach`` (of those, falls on whose tick the action mask let the
        Priest heal that teammate: the same chance for both teams to answer)
        and ``answered`` (of the falls, healed in time).
    """
    ticks = len(events)
    heals: dict[int, set[int]] = {}
    for tick, tick_events in enumerate(events):
        for event in tick_events:
            if event.event_type == "ability_activated" and event.source_global_slot in (
                4,
                9,
            ):
                heals.setdefault(tick, set()).add(event.recipient_global_slot)
    result: dict[str, dict[str, float]] = {}
    for team in ("A", "B"):
        own, _ = _rows_of(team)
        priest = own[4]
        counts = {"falls": 0.0, "in_reach": 0.0, "answered": 0.0}
        for mate in own[:4]:
            line = DANGER_FRACTION * arrays.max_health[mate]
            for tick in range(1, ticks):
                fell = (
                    arrays.alive[tick, mate]
                    and arrays.health[tick, mate] <= line
                    and arrays.health[tick - 1, mate] > line
                    and arrays.alive[tick, priest]
                )
                if not fell:
                    continue
                counts["falls"] += 1
                counts["in_reach"] += float(
                    arrays.choices[tick, priest, own.index(mate) + 1, 0]
                )
                counts["answered"] += float(
                    any(
                        mate in heals.get(later, set())
                        for later in range(tick, tick + DANGER_TICKS + 1)
                    )
                )
        result[team] = counts
    return result


def target_choices(
    events: Sequence[Sequence[Any]], arrays: GameArrays
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare each team's Basic attack targets with an even pick, class by class.

    Only attacks made while the action mask offered two or more enemies count,
    so there was a choice. An even pick hits each offered enemy with chance 1
    over the number offered.

    Parameters
    ----------
    events : sequence of sequence
        The replay's events tick by tick; ``ability_activated`` with
        ``ability_component`` "basic" on an agent of the other team.
    arrays : GameArrays
        The same replay's per-tick state.

    Returns
    -------
    dict
        "A" and "B" (the attacking team) to each enemy class in ``CLASSES``
        to ``offered`` (attacks with that enemy among the choices), ``hit``
        (of those, attacks on it) and ``even_pick`` (the hits an even pick
        would expect over those attacks).
    """
    result = {
        team: {name: {"offered": 0.0, "hit": 0.0, "even_pick": 0.0} for name in CLASSES}
        for team in ("A", "B")
    }
    for tick, tick_events in enumerate(events):
        for event in tick_events:
            if not (
                event.event_type == "ability_activated"
                and event.ability_component == "basic"
            ):
                continue
            source = event.source_global_slot
            team = "A" if source < 5 else "B"
            _, other = _rows_of(team)
            if event.recipient_global_slot not in other:
                continue
            offered = arrays.choices[tick, source, 6:11, 0]
            count = int(np.count_nonzero(offered))
            if count < 2:
                continue
            for row in np.flatnonzero(offered):
                counts = result[team][CLASSES[row]]
                counts["offered"] += 1
                counts["even_pick"] += 1 / count
                counts["hit"] += float(other[row] == event.recipient_global_slot)
    return result


def head_to_head(
    games: Sequence[Mapping[str, Any]], luck: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Put the person's team and the opponent side by side, measure by measure.

    Parameters
    ----------
    games : sequence of mapping
        One record per against game, in order: ``ultimates`` (that game's
        ``ultimate_timing`` rows, with ``arrays``), ``burst`` (``burst_value``),
        ``heals`` (``heal_choices``), ``danger`` (``danger_responses``) and
        ``targets`` (``target_choices``). Team A is the person's team.
    luck : mapping of str to float
        "A" and "B" to the median share of Ultimates together under luck
        (``ultimates_together``'s ``chance_median``), shown beside that row.

    Returns
    -------
    list of dict
        One row per measure: ``measure`` (plain words), ``person`` and
        ``opponent`` (over all the games), ``person_chance`` and
        ``opponent_chance`` (what luck or an even pick gives, where that
        applies; NaN otherwise), ``person_g`` and ``opponent_g`` for each game
        g (1, 2, ...), and ``same_direction`` (True when the person's value is
        above the opponent's in every game, or below it in every game). Shares
        are fractions; NaN where nothing counts.
    """

    def uses(chosen: Sequence[Mapping[str, Any]], team: str) -> list[Mapping[str, Any]]:
        """The team's Ultimate rows in the chosen games."""
        return [
            row for game in chosen for row in game["ultimates"] if row["team"] == team
        ]

    def share(values: Sequence[float]) -> float:
        """The mean of 0/1 values; NaN when there are none."""
        return float(np.mean(values)) if len(values) else math.nan

    def total(
        chosen: Sequence[Mapping[str, Any]], key: str, team: str, name: str
    ) -> float:
        """One count added up over the chosen games."""
        return float(sum(game[key][team][name] for game in chosen))

    def ratio(top: float, bottom: float) -> float:
        """top / bottom; NaN when bottom is zero."""
        return top / bottom if bottom else math.nan

    def target(name: str, field: str) -> GameMeasure:
        """Share of attacks on one enemy class, or an even pick's share."""
        return lambda chosen, team: ratio(
            sum(game["targets"][team][name][field] for game in chosen),
            sum(game["targets"][team][name]["offered"] for game in chosen),
        )

    measures: list[tuple[str, GameMeasure, Callable[[str], float] | str | None]] = [
        (
            "Ultimates used on the first tick the agent could use them",
            lambda chosen, team: share(
                [row["first_usable"] for row in uses(chosen, team)]
            ),
            None,
        ),
        (
            "Ticks the Warrior waited with a Charge it could use (median)",
            lambda chosen, team: float(
                np.median(
                    [
                        r["usable_wait"]
                        for r in uses(chosen, team)
                        if r["class"] == "Warrior"
                    ]
                    or [math.nan]
                )
            ),
            None,
        ),
        (
            "Damage dealt by the opening Burst (mean over games)",
            lambda chosen, team: float(
                np.mean([game["burst"][team]["opening_damage"] for game in chosen])
            ),
            None,
        ),
        (
            "Bursts pressed with an enemy the Mage could attack",
            lambda chosen, team: share(
                [
                    r["enemy_attackable"]
                    for r in uses(chosen, team)
                    if r["class"] == "Mage"
                ]
            ),
            None,
        ),
        (
            "Mage damage per tick during Burst over its own normal rate",
            lambda chosen, team: ratio(
                ratio(
                    total(chosen, "burst", team, "window_damage"),
                    total(chosen, "burst", team, "window_ticks"),
                ),
                ratio(
                    total(chosen, "burst", team, "other_damage"),
                    total(chosen, "burst", team, "other_ticks"),
                ),
            ),
            None,
        ),
        (
            f"Ultimates within {SYNC_TICKS} ticks of a teammate's",
            lambda chosen, team: share(
                [row["teammate_gap"] <= SYNC_TICKS for row in uses(chosen, team)]
            ),
            lambda team: luck.get(team, math.nan),
        ),
        (
            f"Mage's Bursts within {SYNC_TICKS} ticks of the Warrior's Charge",
            lambda chosen, team: share(
                [
                    r["warrior_gap"] <= SYNC_TICKS
                    for r in uses(chosen, team)
                    if r["class"] == "Mage"
                ]
            ),
            None,
        ),
        (
            "Teammates who could also use their Ultimate at that moment (mean)",
            lambda chosen, team: float(
                np.mean(
                    [row["mates_usable"] for row in uses(chosen, team)] or [math.nan]
                )
            ),
            None,
        ),
        (
            "Basic heals that went to a hurt teammate, when one was in reach",
            lambda chosen, team: ratio(
                total(chosen, "heals", team, "to_hurt"),
                total(chosen, "heals", team, "with_hurt_in_reach"),
            ),
            None,
        ),
        (
            f"Teammates at {DANGER_FRACTION:.0%} health or less healed within "
            f"{DANGER_TICKS} ticks",
            lambda chosen, team: ratio(
                total(chosen, "danger", team, "answered"),
                total(chosen, "danger", team, "falls"),
            ),
            None,
        ),
        (
            "Teammates in danger whom the Priest could heal at once",
            lambda chosen, team: ratio(
                total(chosen, "danger", team, "in_reach"),
                total(chosen, "danger", team, "falls"),
            ),
            None,
        ),
        (
            "Attacks on the enemy Priest when it was in reach",
            target("Priest", "hit"),
            "Priest",
        ),
        (
            "Attacks on the enemy Rogue when it was in reach",
            target("Rogue", "hit"),
            "Rogue",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, value, chance in measures:
        row: dict[str, Any] = {
            "measure": name,
            "person": value(games, "A"),
            "opponent": value(games, "B"),
        }
        if isinstance(chance, str):
            row["person_chance"] = target(chance, "even_pick")(games, "A")
            row["opponent_chance"] = target(chance, "even_pick")(games, "B")
        elif chance is not None:
            row["person_chance"] = chance("A")
            row["opponent_chance"] = chance("B")
        else:
            row["person_chance"] = row["opponent_chance"] = math.nan
        signs: list[float] = []
        for number, game in enumerate(games, 1):
            person, opponent = value([game], "A"), value([game], "B")
            row[f"person_{number}"] = person
            row[f"opponent_{number}"] = opponent
            signs.append(float(np.sign(person - opponent)))
        row["same_direction"] = bool(
            signs
            and all(math.isfinite(sign) for sign in signs)
            and abs(sum(signs)) == len(signs)
        )
        rows.append(row)
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
    """Read one recorded game: game.json, its full metrics row and its replay.

    Parameters
    ----------
    folder : pathlib.Path
        A game folder written by the human_games stage.

    Returns
    -------
    (dict, dict, pathlib.Path)
        game.json as a dict; the one full_metrics.csv row with every number
        as float (text cells such as the replay name are left out); and the
        path of the counted replay named in game.json.

    Raises
    ------
    OSError, json.JSONDecodeError, StopIteration
        A file is missing, unreadable or empty.
    """
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

    Parameters
    ----------
    table : mapping of str to numpy.ndarray
        Columns of equal length, such as a ``system_metrics(by="game")``
        result.

    Returns
    -------
    list of dict
        One dict per row. Cells are converted one by one, because a column of
        mixed values comes back as an object array: numbers become float;
        booleans, text and None stay as they are. Empty for an empty table.
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


def field_record(
    reference: Sequence[Row], system: str, maps: Sequence[int]
) -> dict[str, Any]:
    """Sum up a System's own games in the field on the person's maps.

    This says how unusual the person's results are: how often the System
    lost on these maps, and by how much at worst.

    Parameters
    ----------
    reference : sequence of dict
        The field's games, one row per System per game from that System's own
        side, with ``system`` (display name), ``map_id``, ``score_difference``
        and ``team_a_win``, ``team_a_draw`` and ``team_a_loss``.
    system : str
        The display name of the System the person played against.
    maps : sequence of int
        The maps of the person's games; each counts once.

    Returns
    -------
    dict
        ``system``, ``maps`` (the sorted map numbers joined by spaces),
        ``games``, ``wins``, ``draws``, ``losses`` and ``largest_loss``: the
        most points any opponent beat the System by in one of these games,
        0 when it lost none.
    """
    kept = sorted(set(int(map_id) for map_id in maps))
    rows = [
        row
        for row in reference
        if row["system"] == system and int(row["map_id"]) in kept
    ]
    margins = [float(row["score_difference"]) for row in rows]
    return {
        "system": system,
        "maps": " ".join(str(map_id) for map_id in kept),
        "games": len(rows),
        "wins": int(sum(row["team_a_win"] == 1 for row in rows)),
        "draws": int(sum(row["team_a_draw"] == 1 for row in rows)),
        "losses": int(sum(row["team_a_loss"] == 1 for row in rows)),
        "largest_loss": max([0.0, *(-margin for margin in margins)]),
    }


def write_outputs(record: Mapping[str, Any], output_dir: Path) -> None:
    """Write the stage's CSV files, summary.md and analysis.json.

    Parameters
    ----------
    record : mapping
        ``main``'s record: ``against_screen``, ``lead_screen``,
        ``alongside_screen``, ``classes``, ``ultimates``, ``together``,
        ``head_to_head`` and ``alongside`` rows, ``games`` and ``settings``,
        and optionally ``field_record`` rows (written to field_record.csv).
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
    table("against_screen_lead.csv", record["lead_screen"])
    table("alongside_screen.csv", record["alongside_screen"])
    table("classes.csv", record["classes"])
    table("ultimates.csv", record["ultimates"])
    table("ultimates_together.csv", record["together"])
    table("head_to_head.csv", record["head_to_head"])
    table("alongside.csv", record["alongside"])
    table("field_record.csv", record.get("field_record", []))
    lines = ["# The Person's Games Beside the Systems' Own", ""]
    lines.append(
        "Found by screening many metrics in a few games: a description of these "
        "games, not a test."
    )
    for key, title, reference in (
        ("against_screen", "Against games", "the opponent's wins on that map"),
        (
            "lead_screen",
            "Against games, lead-matched",
            "the opponent's wins by lead_margin or more on that map",
        ),
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
            "| Kills after, per use | Kills in an average window "
            f"| With a teammate's Ultimate (within {SYNC_TICKS} ticks) "
            "| With the Warrior's Ultimate |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in ultimate_summary(record["ultimates"]):
            lines.append(
                f"| {row['kind']} | {row['who']} | {row['class']} | {row['uses']} | "
                f"{row['median_wait']:.1f} | {row['kills_after_per_use']:.2f} | "
                f"{row['kills_per_window']:.2f} | {row['with_teammate_share']:.2f} | "
                f"{row['with_warrior_share']:.2f} |"
            )
    if record["together"]:
        lines += [
            "",
            f"**Ultimates used together** (within {SYNC_TICKS} ticks of a "
            "teammate's, beside luck: each agent's Ultimate times slid at random "
            "within its game)",
            "",
            "| Games | Team | Ultimates | Together | Luck: median (95% range) "
            "| Slides at least as high |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in record["together"]:
            lines.append(
                f"| {row['kind']} | {row['who']} | {row['ultimates']} | "
                f"{row['together']} ({row['together_share']:.0%}) | "
                f"{row['chance_median']:.0%} ({row['chance_low']:.0%} to "
                f"{row['chance_high']:.0%}) | {row['chance_at_least']:.3f} |"
            )
    if record["head_to_head"]:
        count = sum(
            1
            for name in record["head_to_head"][0]
            if name.startswith("person_") and name[7:].isdigit()
        )
        lines += [
            "",
            "**Head to head in the against games** (the person's team and the opponent "
            "in the same games; shares as fractions; chance is what luck or an even "
            "pick gives)",
            "",
            "| Measure | Person | Opponent | Chance (person, opponent) | "
            + " | ".join(f"Game {g}" for g in range(1, count + 1))
            + " | Same way every game |",
            "| --- | ---: | ---: | ---: |" + " ---: |" * count + " :---: |",
        ]
        for row in record["head_to_head"]:
            games_text = " | ".join(
                f"{row[f'person_{g}']:.3g} vs {row[f'opponent_{g}']:.3g}"
                for g in range(1, count + 1)
            )
            lines.append(
                f"| {row['measure']} | {row['person']:.3g} | {row['opponent']:.3g} | "
                f"{row['person_chance']:.3g}, {row['opponent_chance']:.3g} | "
                f"{games_text} | {'yes' if row['same_direction'] else 'no'} |"
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
        ``median_wait`` (ticks), ``kills_after_per_use``,
        ``kills_per_window`` (the mean of the games' average windows over the
        uses, so each use is compared with its own game),
        ``with_teammate_share`` (the share of uses within ``SYNC_TICKS`` ticks
        of a teammate's Ultimate) and ``with_warrior_share`` (the same for the
        team's Warrior's Ultimate; NaN for the Warrior).
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
            "with_teammate_share": float(
                np.mean([row["teammate_gap"] <= SYNC_TICKS for row in group])
            ),
            "with_warrior_share": (
                math.nan
                if kind == "Warrior"
                else float(np.mean([row["warrior_gap"] <= SYNC_TICKS for row in group]))
            ),
        }
        for (game_kind, who, kind), group in groups.items()
    ]


def _together(
    games: Mapping[int, tuple[np.ndarray, np.ndarray]],
    lengths: Mapping[int, int],
    rng: np.random.Generator | None,
) -> int:
    """Count one team's Ultimates used within SYNC_TICKS of a teammate's.

    Parameters
    ----------
    games : mapping of int to (numpy.ndarray, numpy.ndarray)
        Each game's number to its Ultimates: the user's slot and the tick,
        two int arrays of the same length.
    lengths : mapping of int to int
        Each game's length in ticks.
    rng : numpy.random.Generator or None
        None counts the games as played. A generator first slides each
        agent's ticks in each game by one random number of ticks in
        [0, length), wrapping past the end to the start, and draws from it.

    Returns
    -------
    int
        Ultimates with another agent's Ultimate at most ``SYNC_TICKS`` ticks
        away in the same game.
    """
    count = 0
    for number, (slots, ticks) in games.items():
        if rng is not None:
            agents, index = np.unique(slots, return_inverse=True)
            offsets = rng.integers(lengths[number], size=len(agents))
            ticks = (ticks + offsets[index]) % lengths[number]
        near = np.abs(ticks[:, None] - ticks[None, :]) <= SYNC_TICKS
        others = slots[:, None] != slots[None, :]
        count += int(np.count_nonzero((near & others).any(axis=1)))
    return count


def ultimates_together(
    rows: Sequence[Mapping[str, Any]],
    lengths: Mapping[int, int],
    shuffles: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Test whether each team used its Ultimates together more often than luck.

    An Ultimate counts as together when another agent of the same team used
    its Ultimate at most ``SYNC_TICKS`` ticks before or after it in the same
    game. Luck is measured by sliding each agent's Ultimate times in a game by
    one random number of ticks, wrapping past the end of the game to its
    start, and counting again. A slide keeps how often and how evenly each
    agent used its Ultimate, but makes the agents' timings independent of each
    other.

    Parameters
    ----------
    rows : sequence of mapping
        ``ultimate_timing`` rows with ``number`` (the game), ``kind`` and
        ``who`` added, as ``main`` builds them.
    lengths : mapping of int to int
        Each game's number to its length in ticks, at least 1.
    shuffles : int
        How many random slides to count, at least 1.
    seed : int
        Seed of the slides (``numpy.random.default_rng``). The same rows,
        lengths, shuffles and seed give the same result.

    Returns
    -------
    list of dict
        One row per team in each kind of game, in first-seen order: ``kind``,
        ``team`` ("A" or "B"), ``who`` (the team's controllers in first-seen
        order joined by " and ", such as "Person and Recurrent IPPO"),
        ``games``, ``ultimates``, ``together`` (a count), ``together_share``,
        ``chance_median``, ``chance_low`` and ``chance_high`` (the median, 2.5th
        and 97.5th percentiles of the share over the slides) and
        ``chance_at_least`` (the share of slides with at least
        ``together_share``: a one-sided p-value, where 0 means no slide
        reached it).
    """
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["kind"]), str(row["team"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (kind, team), group in groups.items():
        games = {
            number: (
                np.asarray(
                    [int(row["slot"]) for row in group if row["number"] == number]
                ),
                np.asarray(
                    [int(row["tick"]) for row in group if row["number"] == number]
                ),
            )
            for number in dict.fromkeys(int(row["number"]) for row in group)
        }
        together = _together(games, lengths, None)
        chance = np.asarray(
            [_together(games, lengths, rng) for _ in range(shuffles)], np.float64
        ) / len(group)
        share = together / len(group)
        result.append(
            {
                "kind": kind,
                "team": team,
                "who": " and ".join(dict.fromkeys(str(row["who"]) for row in group)),
                "games": len(games),
                "ultimates": len(group),
                "together": together,
                "together_share": share,
                "chance_median": float(np.median(chance)),
                "chance_low": float(np.quantile(chance, 0.025)),
                "chance_high": float(np.quantile(chance, 0.975)),
                "chance_at_least": float(np.mean(chance >= share)),
            }
        )
    return result


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
        kills count), ``lead_margin`` (the smallest point margin of the
        opponent's wins in the lead-matched reference, so the reference is as
        far ahead as the person was), ``chance_shuffles`` and ``chance_seed``
        (how many random slides measure how often Ultimates come together by
        luck, and their seed; see ``ultimates_together``) and the required
        ``output_dir``.
        Relative paths use the folder the command was started from.

    Returns
    -------
    dict
        ``against_screen`` and ``lead_screen`` (against all of the opponent's
        wins and against its wins by ``lead_margin`` or more),
        ``alongside_screen``, ``classes``, ``ultimates``, ``together``,
        ``head_to_head`` and ``alongside`` rows (see ``screen_games``,
        ``class_profile``, ``ultimate_timing``, ``ultimates_together``,
        ``head_to_head`` and ``compare_alongside``), ``field_record`` (one
        ``field_record`` row for the System the against games played, empty
        without against games), ``games`` (each recorded game's number, kind,
        map, side, opponent, partner and result) and ``settings``.

    Raises
    ------
    ValueError
        The series has no recorded game, or a System the series names is not
        in ``names``.

    Notes
    -----
    Host-only: NumPy and the replay loader, no JAX computation and no policy
    calls; class health and the Basic heal's size come from the simulator's
    constants. Games whose folder has no game.json (not yet played) are
    skipped; a replay without frames gives no head-to-head rows. The only
    randomness is the luck slides, drawn from ``chance_seed``. Writes only to
    output_dir.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.core.combat import (
        BASIC_HEALING_BY_CLASS,
        MAX_HEALTH_BY_CLASS,
    )
    from marl_battlegrounds.evaluation import load_replay

    # Class ids 1-5 are Mage to Priest, so slot k's class id is k % 5 + 1.
    max_health = [float(MAX_HEALTH_BY_CLASS[slot % 5 + 1]) for slot in range(10)]
    heal_amount = float(BASIC_HEALING_BY_CLASS[CLASSES.index("Priest") + 1])

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
    lead_screen: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    ultimates: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    if against:
        opponent = against[0][0]["opponent"]
        wins = [
            row
            for row in reference
            if row["system"] == opponent and row["team_a_win"] == 1
        ]
        maps = [int(details["map_id"]) for details, _, _ in against]
        records.append(field_record(reference, opponent, maps))
        lead = int(settings["lead_margin"])
        lopsided = [row for row in wins if row["score_difference"] >= lead]
        columns = [*TEAM_COLUMNS, *DERIVED]
        against_screen = screen_games(
            [row for _, row, _ in against],
            [[row for row in wins if row["map_id"] == map_id] for map_id in maps],
            columns,
        )
        lead_screen = screen_games(
            [row for _, row, _ in against],
            [[row for row in lopsided if row["map_id"] == map_id] for map_id in maps],
            columns,
        )
        teams = {
            "Person": [row for _, row, _ in against],
            f"{opponent} in the same games": [mirrored(row) for _, row, _ in against],
            f"{opponent}, its wins on the same maps": [
                row for row in wins if row["map_id"] in maps
            ],
            f"{opponent}, its wins by {lead} or more on the same maps": [
                row for row in lopsided if row["map_id"] in maps
            ],
        }
        for team, games in teams.items():
            for name, values in class_profile(games).items():
                classes.append(
                    {"team": team, "table": name}
                    | dict(zip(CLASSES, values, strict=True))
                )
    lengths: dict[int, int] = {}
    choices: list[dict[str, Any]] = []
    for details, _, replay in series:
        loaded = load_replay(replay).replay
        events = [transition.events for transition in loaded.transitions]
        arrays = game_arrays(loaded.frames, max_health) if loaded.frames else None
        lengths[int(details["number"])] = max(len(events), 1)
        timing = ultimate_timing(events, person_slots(details), window, arrays)
        if details["kind"] == "against" and arrays is not None:
            choices.append(
                {
                    "ultimates": timing,
                    "burst": burst_value(events, arrays),
                    "heals": heal_choices(events, arrays, heal_amount),
                    "danger": danger_responses(events, arrays),
                    "targets": target_choices(events, arrays),
                }
            )
        for row in timing:
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

    together = ultimates_together(
        ultimates,
        lengths,
        int(settings["chance_shuffles"]),
        int(settings["chance_seed"]),
    )
    luck = {
        row["team"]: row["chance_median"]
        for row in together
        if row["kind"] == "against"
    }
    versus = head_to_head(choices, luck) if choices else []

    record: dict[str, Any] = {
        "against_screen": against_screen,
        "lead_screen": lead_screen,
        "alongside_screen": alongside_screen,
        "classes": classes,
        "ultimates": ultimates,
        "together": together,
        "head_to_head": versus,
        "alongside": alongside,
        "field_record": records,
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
