"""Explain how a language-model team played, from its saved games and calls.

Use python -m marl_battlegrounds.experiments.llm_analysis field_dir=/path/games
scenarios_dir=/path/scenarios output_dir=/path/analysis after the llm_challenge
stage has played with metrics=full. field_dir is that stage's output_dir (it
holds runs.json); scenarios_dir is its scenarios folder (it holds
scenario_runs.json); either may be null. The stage reads every saved game through
marl_bgs.load_results and every saved model call through llm.read_calls. It
writes behavior.csv (the model's team and its opponents, per game, in the same
games), answers.csv (how often the model's answers were legal, what they chose,
their tokens and how long they took) and analysis.md. It plays no games, calls
no model and needs no GPU.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
# isort: on

from marl_battlegrounds.llm.actions import MOVE_NAMES

# Per-game team measures, as (name in the output, full_metrics column suffix).
# Each is read as team_a_<suffix> for the model and team_b_<suffix> for its
# opponent. own_red_zone_deaths counts deaths inside the team's own Red Zone.
MEASURES = (
    ("basic_uses", "basic_activations"),
    ("damage_done", "damage_done"),
    ("focus_fire_ticks", "focus_fire_steps"),
    ("shared_kills", "multi_contributor_kills"),
    ("ally_distance", "ally_distance_mean"),
    ("deaths", "deaths"),
    ("own_red_zone_deaths", "red_zone_deaths"),
)
_SIDES = (("model", "a"), ("opponent", "b"))


def team_values(full: Mapping[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Read each team's measures for every game of one saved full_metrics table.

    Parameters
    ----------
    full : mapping
        A run's ``full_metrics`` table, as ``marl_bgs.load_results(run).table``
        returns it: one array per column, one row per game.

    Returns
    -------
    dict
        ``{"model": {...}, "opponent": {...}}``: for Team A (the model's team)
        and Team B, each ``MEASURES`` name maps to one float per game. A game
        without a value, such as ``ally_distance`` when no two allies were
        alive together, holds NaN.

    Raises
    ------
    KeyError
        A measure's column is missing, for example in a run played with
        ``metrics="priority"``.
    """
    import numpy as np

    return {
        side: {
            name: [
                float(value)
                for value in np.asarray(full[f"team_{team}_{column}"], np.float64)
            ]
            for name, column in MEASURES
        }
        for side, team in _SIDES
    }


def answer_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count what Team A's model answered, from saved model-call rows.

    Parameters
    ----------
    rows : iterable of mapping
        Rows from ``llm.read_calls``. Only Team A rows (``team == 0``) whose
        decision was played (``outcome == "played"``) count.

    Returns
    -------
    dict
        ``calls`` (model requests: answers plus fallbacks), ``legal_percent``
        (answers that were legal actions, as a share of calls), ``failures``
        (count per error class name, such as ``"ReplyFormatError"``),
        ``forced_turns`` (turns with one legal choice, which need no call),
        ``no_combat_percent`` (legal answers with no target and no Ultimate),
        ``move_percent`` (share of legal answers per move name, in the
        System's own frame: with the default left frame, ``east`` is toward
        the enemy's side) and ``median_call_seconds`` (request time of a call
        while the other agents' calls also run). Percentages and the median are
        None when there is nothing to count.
    """
    calls = forced = 0
    failures: Counter[str] = Counter()
    moves: Counter[str] = Counter()
    no_combat = 0
    seconds: list[float] = []
    for row in rows:
        if row.get("team") != 0 or row.get("outcome") != "played":
            continue
        if row.get("source") == "forced":
            forced += 1
            continue
        calls += 1
        if row.get("failure"):
            failures[str(row["failure"])] += 1
        if row.get("seconds") is not None:
            seconds.append(float(row["seconds"]))
        if row.get("source") == "model":
            move, target, ultimate = (int(value) for value in row["action"])
            moves[MOVE_NAMES[move]] += 1
            no_combat += (target, ultimate) == (0, 0)
    legal = sum(moves.values())
    return {
        "calls": calls,
        "legal_percent": 100 * legal / calls if calls else None,
        "failures": dict(failures),
        "forced_turns": forced,
        "no_combat_percent": 100 * no_combat / legal if legal else None,
        "move_percent": {
            name: 100 * moves[name] / legal if legal else None for name in MOVE_NAMES
        },
        "median_call_seconds": statistics.median(seconds) if seconds else None,
    }


def _mean(values: Sequence[float]) -> float | None:
    """Average the values that exist; None when every value is NaN or missing."""
    present = [value for value in values if value == value]
    return round(sum(present) / len(present), 2) if present else None


def _answers_row(
    experiment: str, records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Join call counts from the challenge records with one answer summary."""
    calls = sum(int(record["model_calls"]) for record in records)

    def per_call(key: str) -> float | None:
        return round(sum(int(r[key]) for r in records) / calls, 1) if calls else None

    def rounded(value: float | None) -> float | None:
        return None if value is None else round(value, 2)

    return {
        "experiment": experiment,
        "games": sum(int(record["games"]) for record in records),
        "calls": summary["calls"],
        "legal_percent": rounded(summary["legal_percent"]),
        "failures": "; ".join(
            f"{k}: {v}" for k, v in sorted(summary["failures"].items())
        ),
        "forced_turns": summary["forced_turns"],
        "input_tokens_per_call": per_call("input_tokens"),
        "output_tokens_per_call": per_call("output_tokens"),
        "median_call_seconds": rounded(summary["median_call_seconds"]),
        "wall_seconds": round(sum(float(r["seconds"]) for r in records), 1),
        "no_combat_percent": rounded(summary["no_combat_percent"]),
    } | {
        f"move_{name}_percent": rounded(value)
        for name, value in summary["move_percent"].items()
    }


def read_field(field_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize the games against a field: behavior per opponent and answers.

    Parameters
    ----------
    field_dir : pathlib.Path
        The llm_challenge stage's ``output_dir``, holding ``runs.json``. Each
        pair's saved ``run_dir`` is read as recorded there.

    Returns
    -------
    behavior : list of dict
        One row per opponent, in ``runs.json`` order, then ``"All"``:
        ``opponent``, ``games`` and, for every ``MEASURES`` name, the per-game
        mean for the model's team (``model_<name>``) and its opponent
        (``opponent_<name>``), over the games where it exists.
    answers : dict
        The ``answers.csv`` row for these games (experiment ``"field"``), with
        ``seconds_per_tick``: the pairs' wall time over the length of each
        pair's longer game, because a pair's games play side by side.

    Raises
    ------
    FileNotFoundError
        ``runs.json`` is missing.
    KeyError
        A run has no full metrics.
    """
    import numpy as np

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import llm

    pairs = json.loads((field_dir / "runs.json").read_text(encoding="utf-8"))["pairs"]
    values: dict[str, dict[str, dict[str, list[float]]]] = {}
    rows: list[dict[str, Any]] = []
    ticks = 0
    for pair in pairs:
        saved = marl_bgs.load_results(pair["run_dir"])
        full = saved.table("full_metrics")
        if not full:
            raise KeyError(
                f"{pair['run_dir']} has no full metrics; play with metrics=full"
            )
        mine = values.setdefault(
            pair["opponent"], {side: defaultdict(list) for side, _ in _SIDES}
        )
        for side, numbers in team_values(full).items():
            for name, per_game in numbers.items():
                mine[side][name] += per_game
        ticks += int(np.max(np.asarray(saved.table("episodes")["episode_length"])))
        rows += llm.read_calls(pair["run_dir"])
    groups = list(values.items())
    groups.append(
        (
            "All",
            {
                side: {
                    name: [v for group in values.values() for v in group[side][name]]
                    for name, _ in MEASURES
                }
                for side, _ in _SIDES
            },
        )
    )
    behavior = [
        {"opponent": opponent, "games": len(group["model"]["deaths"])}
        | {
            f"{side}_{name}": _mean(group[side][name])
            for name, _ in MEASURES
            for side, _ in _SIDES
        }
        for opponent, group in groups
    ]
    answers = _answers_row("field", pairs, answer_summary(rows))
    answers["seconds_per_tick"] = (
        round(answers["wall_seconds"] / ticks, 2) if ticks else None
    )
    return behavior, answers


def read_scenarios(scenarios_dir: Path) -> dict[str, Any]:
    """Summarize the model's answers in the endgame scenarios.

    Parameters
    ----------
    scenarios_dir : pathlib.Path
        The llm_challenge stage's scenarios folder, holding
        ``scenario_runs.json``.

    Returns
    -------
    dict
        The ``answers.csv`` row for these games (experiment ``"scenarios"``).
        ``seconds_per_tick`` is None: many games share the request slots, so
        their ticks do not take turns one by one.

    Raises
    ------
    FileNotFoundError
        ``scenario_runs.json`` is missing.
    """
    from marl_battlegrounds import llm

    path = scenarios_dir / "scenario_runs.json"
    records = json.loads(path.read_text(encoding="utf-8"))["scenarios"]
    rows = [row for record in records for row in llm.read_calls(record["run_dir"])]
    return _answers_row("scenarios", records, answer_summary(rows)) | {
        "seconds_per_tick": None
    }


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write dict rows to a CSV file with the first row's keys as header.

    Parameters
    ----------
    rows : sequence of mapping
        Nonempty rows that share one set of keys. None values are written as
        empty cells.
    path : pathlib.Path
        File to create or replace.
    """
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    behavior: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Write analysis.md: the model's team beside its opponents, and its answers.

    Parameters
    ----------
    behavior : sequence of mapping
        Rows from ``read_field``; only the ``"All"`` row is summarized. May be
        empty.
    answers : sequence of mapping
        ``answers.csv`` rows.
    path : pathlib.Path
        Markdown file to create or replace.
    """
    lines = ["# LLM Team Analysis", ""]
    overall = [row for row in behavior if row["opponent"] == "All"]
    if overall:
        row = overall[0]
        lines += [
            f"Per game, over {row['games']} games against the field "
            "(model's team | its opponents):",
            "",
            "| Measure | Model | Opponents |",
            "| --- | ---: | ---: |",
        ] + [
            f"| {name.replace('_', ' ').capitalize()} | {row[f'model_{name}']} "
            f"| {row[f'opponent_{name}']} |"
            for name, _ in MEASURES
        ]
        lines.append("")
    for row in answers:
        moves = ", ".join(
            f"{name} {row[f'move_{name}_percent']}%"
            for name in MOVE_NAMES
            if row[f"move_{name}_percent"]
        )
        lines.append(
            f"- {row['experiment'].capitalize()}: {row['calls']:,} calls, "
            f"{row['legal_percent']}% legal ({row['failures'] or 'no failures'}); "
            f"{row['input_tokens_per_call']} input and "
            f"{row['output_tokens_per_call']} output tokens per call; median "
            f"call {row['median_call_seconds']} s; "
            + (
                f"{row['seconds_per_tick']} s per tick; "
                if row["seconds_per_tick"]
                else ""
            )
            + f"no combat in {row['no_combat_percent']}% of answers; moves: {moves}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra.main(version_base="1.3", config_path="configs", config_name="llm_analysis")
def main(config: DictConfig) -> dict[str, int]:
    """Read the llm_challenge stage's saved runs and write the analysis files.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``field_dir`` (the llm_challenge ``output_dir``, or null),
        ``scenarios_dir`` (its scenarios folder, or null) and ``output_dir``.
        Relative paths use the folder the command was started from.

    Returns
    -------
    dict of str to int
        Row counts of behavior.csv (0 without field_dir) and answers.csv.

    Raises
    ------
    ValueError
        Both field_dir and scenarios_dir are null.
    FileNotFoundError
        A given folder lacks its runs.json or scenario_runs.json.
    KeyError
        A field run has no full metrics.
    """
    settings = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))

    def folder(key: str) -> Path | None:
        value = settings.get(key)
        return (
            None
            if value is None
            else Path(to_absolute_path(str(Path(value).expanduser())))
        )

    field_dir, scenarios_dir = folder("field_dir"), folder("scenarios_dir")
    output = folder("output_dir")
    if output is None:
        raise ValueError("output_dir is required")
    if field_dir is None and scenarios_dir is None:
        raise ValueError("Give field_dir, scenarios_dir or both")
    behavior: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    if field_dir is not None:
        behavior, field_answers = read_field(field_dir)
        answers.append(field_answers)
    if scenarios_dir is not None:
        answers.append(read_scenarios(scenarios_dir))
    output.mkdir(parents=True, exist_ok=True)
    if behavior:
        write_csv(behavior, output / "behavior.csv")
    write_csv(answers, output / "answers.csv")
    write_summary(behavior, answers, output / "analysis.md")
    print(output / "analysis.md", flush=True)
    return {"behavior": len(behavior), "answers": len(answers)}


if __name__ == "__main__":
    prepare_hydra()
    main()
