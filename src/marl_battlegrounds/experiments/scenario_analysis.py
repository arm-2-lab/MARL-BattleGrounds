"""Explain saved scenario results with full metrics and each scenario's start.

Use python -m marl_battlegrounds.experiments.scenario_analysis
scenarios_dir=/path/run output_dir=/path/analysis after the scenarios stage has
played with metrics=full. The stage reads each system's runs.json, every game's
episodes and full_metrics tables through marl_bgs.load_results, and each
scenario's authored start through marl_bgs.load_tdm_scenario. It writes
agents.csv (which agents died and which used their Ultimate, by outcome),
games.csv (how many ticks games lasted, by outcome), reference.csv (each
system's wins against a reference system's, with Fisher's exact test) and
analysis.md. It plays no games and needs no GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
# isort: on

from marl_battlegrounds.experiments.scenarios import count_outcomes

# Values of the saved episodes table's outcome column, from Team A's side.
_OUTCOME_NAMES = {1: "win", 3: "draw", 2: "loss"}
_TEAM_SIZE = 5


def scenario_start(scenario_id: int) -> list[dict[str, Any]]:
    """Describe every agent alive at a scenario's authored start.

    Parameters
    ----------
    scenario_id : int
        Scenario ID from 1 through 8.

    Returns
    -------
    list of dict
        One record per living agent, in global slot order (Team A slots 0-4,
        then Team B slots 5-9): ``slot``, ``team`` (``"A"`` or ``"B"``),
        ``agent_class`` (for example ``"Priest"``), ``start_health`` (health
        points) and ``start_ultimate_cooldown`` (decisions until the Ultimate is
        ready; 0 means ready on the first tick).

    Raises
    ------
    ValueError
        Raised by ``marl_bgs.load_tdm_scenario`` for an unknown scenario ID.
    """
    import numpy as np

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.metric_catalog import ULTIMATE_CLASS_NAMES

    scenario = marl_bgs.load_tdm_scenario(scenario_id)
    state = scenario.initial_state
    alive = np.asarray(state.alive_mask)
    health = np.asarray(state.current_health)
    cooldown = np.asarray(state.ultimate_cooldowns)
    return [
        {
            "slot": slot,
            "team": "A" if slot < _TEAM_SIZE else "B",
            "agent_class": ULTIMATE_CLASS_NAMES[class_id - 1].title(),
            "start_health": round(float(health[slot]), 1),
            "start_ultimate_cooldown": int(cooldown[slot]),
        }
        for slot, class_id in enumerate(scenario.info.class_ids)
        if alive[slot]
    ]


def read_games(
    system: str, scenario_id: int, run_dir: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize one saved scenario run by outcome, agent and game length.

    Parameters
    ----------
    system : str
        Display name written into every row.
    scenario_id : int
        Scenario the run played; it supplies the starting agents.
    run_dir : str or pathlib.Path
        Exact run folder written by ``marl_bgs.evaluate_scenarios`` with
        ``metrics="full"``.

    Returns
    -------
    agents : list of dict
        One row per outcome that occurred and per agent alive at the start:
        ``system``, ``scenario``, ``outcome`` (``"win"``, ``"draw"`` or
        ``"loss"``, from Team A's side), the ``scenario_start`` fields except
        ``slot``, ``games`` with that outcome, ``died`` (games in which the
        agent died at least once) and ``used_ultimate`` (games in which it used
        its Ultimate at least once).
    games : list of dict
        One row per outcome and game length: ``system``, ``scenario``,
        ``outcome``, ``ticks`` (decisions played) and ``games``.

    Raises
    ------
    KeyError
        The run has no full metrics, for example because it was played with
        ``metrics="priority"``.
    ValueError, OSError
        Raised by ``marl_bgs.load_results`` for an invalid or unreadable run.

    Notes
    -----
    Rows of the episodes and full_metrics tables are matched by ``episode_id``.
    """
    import numpy as np

    import marl_battlegrounds as marl_bgs

    saved = marl_bgs.load_results(run_dir)
    episodes = saved.table("episodes")
    full = saved.table("full_metrics")
    if not full:
        raise KeyError(f"{run_dir} has no full metrics; play with metrics=full")
    order = {int(e): i for i, e in enumerate(np.asarray(full["episode_id"]))}
    rows = [order[int(e)] for e in np.asarray(episodes["episode_id"])]
    outcome = np.asarray(episodes["outcome"], dtype=np.int64)
    ticks = np.asarray(episodes["episode_length"], dtype=np.int64)
    start = scenario_start(scenario_id)
    agents: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    for code, name in _OUTCOME_NAMES.items():
        chosen = outcome == code
        if not chosen.any():
            continue
        picked = np.asarray(rows)[chosen]
        base = {"system": system, "scenario": scenario_id, "outcome": name}
        for agent in start:
            slot = agent["slot"]
            deaths = np.asarray(full[f"agent_{slot}_deaths"], dtype=np.float64)[picked]
            ultimates = np.asarray(
                full[f"agent_{slot}_ultimate_activations"], dtype=np.float64
            )[picked]
            agents.append(
                base
                | {key: value for key, value in agent.items() if key != "slot"}
                | {
                    "games": int(chosen.sum()),
                    "died": int(np.sum(deaths > 0)),
                    "used_ultimate": int(np.sum(ultimates > 0)),
                }
            )
        lengths: Counter[int] = Counter(int(value) for value in ticks[chosen])
        for length, count in sorted(lengths.items()):
            games.append(base | {"ticks": length, "games": count})
    return agents, games


def fisher_exact(wins_a: int, games_a: int, wins_b: int, games_b: int) -> float:
    """Return the two-sided Fisher's exact test p-value for two win counts.

    Parameters
    ----------
    wins_a, games_a : int
        Wins and games of the first system; ``0 <= wins_a <= games_a``.
    wins_b, games_b : int
        Wins and games of the second system; ``0 <= wins_b <= games_b``.

    Returns
    -------
    float
        Probability, with both systems' games and the total win count fixed, of
        a split at least as unlikely as the observed one. 1.0 means no evidence
        of a difference.

    Raises
    ------
    ValueError
        A count is negative or exceeds its game count.

    Examples
    --------
    >>> round(fisher_exact(8, 256, 2, 256), 3)
    0.106
    """
    if not (0 <= wins_a <= games_a and 0 <= wins_b <= games_b):
        raise ValueError("wins must be between 0 and the number of games")
    total_wins, total = wins_a + wins_b, games_a + games_b

    def chance(k: int) -> float:
        return (
            math.comb(games_a, k)
            * math.comb(games_b, total_wins - k)
            / math.comb(total, total_wins)
        )

    observed = chance(wins_a)
    low, high = max(0, total_wins - games_b), min(games_a, total_wins)
    # The relative tolerance keeps tables as likely as the observed one.
    return min(
        1.0,
        sum(
            p
            for p in (chance(k) for k in range(low, high + 1))
            if p <= observed * (1 + 1e-7)
        ),
    )


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write dict rows to a CSV file with the first row's keys as header.

    Parameters
    ----------
    rows : sequence of dict
        Nonempty rows that share one set of keys.
    path : pathlib.Path
        File to create or replace.
    """
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _share(part: int, whole: int) -> str:
    """Format a count as "part of whole (percent)" for the summary."""
    return f"{part:,} of {whole:,} ({100 * part / whole:.1f}%)"


def write_summary(
    agents: Sequence[dict[str, Any]],
    games: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    reference_name: str,
    path: Path,
) -> None:
    """Write analysis.md: per scenario, how the field's games ended and why.

    Parameters
    ----------
    agents, games : sequence of dict
        Rows from ``read_games`` for every system and scenario.
    reference : sequence of dict
        Rows of reference.csv.
    reference_name : str
        Display name of the reference system.
    path : pathlib.Path
        Markdown file to create or replace.

    Notes
    -----
    Counts pool every system. For each scenario the summary gives the losses,
    how many lost games ended within two ticks, and how often each Team A agent
    died in lost games, most often first, with its starting health.
    """
    lines = [
        "# Scenario Analysis",
        "",
        "Counts pool every system; percentages are of the games counted.",
        "",
    ]
    for scenario in sorted({row["scenario"] for row in games}):
        rows = [row for row in games if row["scenario"] == scenario]
        played = sum(row["games"] for row in rows)
        lost = [row for row in rows if row["outcome"] == "loss"]
        losses = sum(row["games"] for row in lost)
        line = f"- Scenario {scenario}: {_share(losses, played)} games lost"
        if losses:
            fast = sum(row["games"] for row in lost if row["ticks"] <= 2)
            died: Counter[tuple[str, float]] = Counter()
            for row in agents:
                if (row["scenario"], row["outcome"], row["team"]) == (
                    scenario,
                    "loss",
                    "A",
                ):
                    died[(row["agent_class"], row["start_health"])] += row["died"]
            victims = ", ".join(
                f"{name} ({health:g} HP at the start) {_share(count, losses)}"
                for (name, health), count in died.most_common()
            )
            line += (
                f"; {_share(fast, losses)} of those ended within two ticks. "
                f"Team A deaths in lost games: {victims}"
            )
        lines.append(line + ".")
    lines += ["", f"## Wins Against {reference_name}", ""]
    lines += [
        f"- {row['system']}, Scenario {row['scenario']}: {row['wins']} vs. "
        f"{row['reference_wins']} wins in {row['games']} games, p = {row['p_value']}"
        for row in reference
        if row["wins"] or row["reference_wins"]
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra.main(version_base="1.3", config_path="configs", config_name="scenario_analysis")
def main(config: DictConfig) -> dict[str, int]:
    """Read every system's saved scenario runs and write the analysis files.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``scenarios_dir`` (the scenarios stage's ``output_dir``),
        ``output_dir`` and ``reference`` (the display name of the system every
        other system's wins are compared with; default ``"Random"``). Relative
        paths use the folder the command was started from. Systems keep the
        order of the scenarios stage's results.csv; without it, folder order.

    Returns
    -------
    dict of str to int
        Row counts of agents.csv, games.csv and reference.csv.

    Raises
    ------
    ValueError
        The scenarios folder holds no runs.json, or no system has the
        reference name.
    KeyError
        A run has no full metrics.
    """
    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    source = Path(to_absolute_path(str(Path(settings["scenarios_dir"]).expanduser())))
    output = Path(to_absolute_path(str(Path(settings["output_dir"]).expanduser())))
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(source.glob("*/runs.json"))
    ]
    # Keep the scenarios stage's system order (results.csv) when it is there.
    table = source / "results.csv"
    if table.is_file():
        with table.open(encoding="utf-8") as file:
            order = list(dict.fromkeys(row["system"] for row in csv.DictReader(file)))
        records.sort(
            key=lambda r: (
                order.index(r["system"]) if r["system"] in order else len(order)
            )
        )
    if not records:
        raise ValueError(f"{source} has no system folders with runs.json")
    reference = [r for r in records if r["system"] == settings["reference"]]
    if not reference:
        raise ValueError(f"no system is named {settings['reference']!r}")
    output.mkdir(parents=True, exist_ok=True)
    agents: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    for record in records:
        for scenario, run_dir in record["scenario_runs"].items():
            new_agents, new_games = read_games(record["system"], int(scenario), run_dir)
            agents += new_agents
            games += new_games
    compared: list[dict[str, Any]] = []
    for record in records:
        if record is reference[0]:
            continue
        for scenario, run_dir in record["scenario_runs"].items():
            mine = count_outcomes(run_dir)
            theirs = count_outcomes(reference[0]["scenario_runs"][scenario])
            p = fisher_exact(
                mine["wins"], mine["games"], theirs["wins"], theirs["games"]
            )
            compared.append(
                {
                    "system": record["system"],
                    "scenario": int(scenario),
                    "games": mine["games"],
                    "wins": mine["wins"],
                    "reference_wins": theirs["wins"],
                    "p_value": round(p, 4),
                }
            )
    write_csv(agents, output / "agents.csv")
    write_csv(games, output / "games.csv")
    write_csv(compared, output / "reference.csv")
    write_summary(
        agents, games, compared, settings["reference"], output / "analysis.md"
    )
    print(output / "analysis.md")
    return {"agents": len(agents), "games": len(games), "reference": len(compared)}


if __name__ == "__main__":
    prepare_hydra()
    main()
