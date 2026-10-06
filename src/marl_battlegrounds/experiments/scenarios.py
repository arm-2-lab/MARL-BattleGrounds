"""Play a field of Systems as Team A in the endgame scenarios and tabulate the results.

Use python -m marl_battlegrounds.experiments.scenarios output_dir=/path/run.
By default the field is the built-in systems TDM-Beta, TDM-Gamma, TDM-Alpha and
Random. Add +field=season_0 models_dir=/path/models to play the Season 0 field
(the six released baselines and the three scripted systems) and Random as a
reference. Override systems=
to play any other methods. Each system meets every scenario's own scripted
opponent through marl_bgs.evaluate_scenarios; the table is read back from the
saved games with marl_bgs.load_results. This stage trains nothing.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
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

# Values of the saved episodes table's outcome column, from Team A's side.
_TEAM_A_WIN, _TEAM_B_WIN, _DRAW = 1, 2, 3


def folder_name(system_name: str) -> str:
    """Turn a display name into the system's output folder name.

    Parameters
    ----------
    system_name : str
        Display name, for example ``"Recurrent IPPO"``.

    Returns
    -------
    str
        Lowercase letters, digits and single hyphens, for example
        ``"recurrent-ippo"``.
    """
    return re.sub(r"[^a-z0-9]+", "-", system_name.lower()).strip("-")


def play_system(
    name: str,
    method: str,
    *,
    scenario_ids: Sequence[int],
    num_episodes: int,
    num_envs: int,
    seed: int,
    metrics: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Play one Team A system in every listed scenario and record its runs.

    Parameters
    ----------
    name : str
        Display name. It names the folder ``output_dir / folder_name(name)``.
    method : str
        Any reference ``marl_bgs.load_method`` accepts: a built-in name such as
        ``"tdm-alpha"``, a saved actor folder or a trusted ``module:function``.
    scenario_ids : sequence of int
        Distinct scenario IDs from 1 through 8, played in this order.
    num_episodes : int
        Games per scenario. Each starts from the authored position with fresh
        method memory; only random draws differ.
    num_envs : int
        Games played side by side. GPU runs use 32, 64, 128, 512 or 1024.
    seed : int
        Root of the evaluator's random draws.
    metrics : {'priority', 'full', 'none'}
        Per-game metrics the evaluator saves with each game.
    output_dir : pathlib.Path
        Parent folder of every system's results.

    Returns
    -------
    dict
        The record also written to ``runs.json`` in the system's folder:
        ``system``, ``method``, ``scenario_runs`` (scenario ID as text to the
        exact run folder), the settings, and ``wall_seconds`` (loading,
        compilation and play). Each run folder's ``run_details.json`` names the
        device that played.

    Raises
    ------
    ValueError, TypeError, OSError
        Raised by the evaluator for invalid settings, an unreadable method or a
        file error. Scenarios finished before the error stay saved.

    Notes
    -----
    Loads the method and plays on the default JAX device. A repeat call adds
    new run folders and points ``runs.json`` at them; earlier folders stay.
    """
    import marl_battlegrounds as marl_bgs

    system_dir = output_dir / folder_name(name)
    started = time.perf_counter()
    results = marl_bgs.evaluate_scenarios(
        method,
        scenario_ids=list(scenario_ids),
        num_episodes=num_episodes,
        seed=seed,
        num_envs=num_envs,
        metrics=metrics,  # pyright: ignore[reportArgumentType] - the evaluator checks it.
        output_dir=system_dir,
    )
    record: dict[str, Any] = {
        "system": name,
        "method": method,
        "scenario_runs": {
            str(identifier): str(result.run_dir)
            for identifier, result in zip(scenario_ids, results, strict=True)
        },
        "num_episodes": num_episodes,
        "num_envs": num_envs,
        "seed": seed,
        "metrics": metrics,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (system_dir / "runs.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{name}: {len(results)} scenarios in {record['wall_seconds']} s")
    return record


def count_outcomes(run_dir: str | Path) -> dict[str, int]:
    """Count Team A's wins, draws and losses in one saved scenario run.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Exact run folder written by ``marl_bgs.evaluate_scenarios``.

    Returns
    -------
    dict of str to int
        ``games``, ``wins``, ``draws`` and ``losses`` from Team A's side; the
        last three add up to ``games``.

    Raises
    ------
    ValueError, OSError
        Raised by ``marl_bgs.load_results`` for an invalid or unreadable run.

    Notes
    -----
    Reads the episodes table's ``outcome`` column, which is always Team A's
    result. ``head_to_head`` groups by system identity instead, so it merges
    both sides when Team A and Team B are the same system.
    """
    import numpy as np

    import marl_battlegrounds as marl_bgs

    outcome = np.asarray(marl_bgs.load_results(run_dir).table("episodes")["outcome"])
    return {
        "games": int(outcome.size),
        "wins": int(np.sum(outcome == _TEAM_A_WIN)),
        "draws": int(np.sum(outcome == _DRAW)),
        "losses": int(np.sum(outcome == _TEAM_B_WIN)),
    }


def write_tables(
    records: Sequence[dict[str, Any]],
    scenario_ids: Sequence[int],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Write ``results.csv`` and ``results.md`` from the systems' saved runs.

    Parameters
    ----------
    records : sequence of dict
        ``play_system`` records in table order.
    scenario_ids : sequence of int
        Scenario columns, in table order. Each record must have a run for each.
    output_dir : pathlib.Path
        Folder for the two tables. Existing tables are replaced.

    Returns
    -------
    list of dict
        One row per system and scenario: ``system``, ``scenario``, ``games``,
        ``wins``, ``draws`` and ``losses``.

    Raises
    ------
    KeyError
        A record has no run for a requested scenario.
    ValueError, OSError
        A saved run cannot be read.
    """
    rows = [
        {"system": record["system"], "scenario": identifier}
        | count_outcomes(record["scenario_runs"][str(identifier)])
        for record in records
        for identifier in scenario_ids
    ]
    fields = ["system", "scenario", "games", "wins", "draws", "losses"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cells = {(row["system"], row["scenario"]): row for row in rows}
    lines = [
        "| Team A | " + " | ".join(f"S{i}" for i in scenario_ids) + " |",
        "| --- |" + " :---: |" * len(scenario_ids),
    ]
    for record in records:
        lines.append(
            f"| {record['system']} | "
            + " | ".join(
                "{wins} · {draws} · {losses}".format(**cells[(record["system"], i)])
                for i in scenario_ids
            )
            + " |"
        )
    games = sorted({row["games"] for row in rows})
    note = (
        f"Wins · draws · losses for Team A out of {games[0]} games per cell."
        if len(games) == 1
        else "Wins · draws · losses for Team A; game counts differ (results.csv)."
    )
    (output_dir / "results.md").write_text(
        note + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    return rows


@hydra.main(version_base="1.3", config_path="configs", config_name="scenarios")
def main(config: DictConfig) -> list[dict[str, Any]]:
    """Play every system in ``config.systems``, then write the tables.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``systems`` (a list of ``name`` and ``method``), ``scenario_ids``,
        ``num_episodes``, ``num_envs``, ``seed``, ``metrics`` and the required
        ``output_dir``; ``models_dir`` fills ``${models_dir}`` in method paths.
        Relative paths use the folder the command was started from.

    Returns
    -------
    list of dict
        The table rows returned by ``write_tables``.

    Raises
    ------
    ValueError
        Two systems share a folder name or no system is listed.
    """
    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    systems = [
        (str(entry["name"]), str(entry["method"])) for entry in settings["systems"]
    ]
    if not systems:
        raise ValueError("systems must list at least one system")
    if len({folder_name(name) for name, _ in systems}) != len(systems):
        raise ValueError("system names must give distinct folder names")
    output_dir = Path(to_absolute_path(str(Path(settings["output_dir"]).expanduser())))
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name, method in systems:
        try:
            records.append(
                play_system(
                    name,
                    method,
                    scenario_ids=settings["scenario_ids"],
                    num_episodes=settings["num_episodes"],
                    num_envs=settings["num_envs"],
                    seed=settings["seed"],
                    metrics=settings["metrics"],
                    output_dir=output_dir,
                )
            )
        except Exception:
            print(f"Scenario output: {output_dir / folder_name(name)}", file=sys.stderr)
            raise
    rows = write_tables(records, settings["scenario_ids"], output_dir)
    print(output_dir / "results.md")
    return rows


if __name__ == "__main__":
    prepare_hydra()
    main()
