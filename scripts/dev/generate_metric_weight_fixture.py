"""Write the small real full-metrics fixture that checks every summary weight.

Run ``JAX_PLATFORMS=cpu python -m scripts.dev.generate_metric_weight_fixture``
from the repository root. It plays four tiny software games on the CPU through
the public API and writes ``tests/fixtures/metric_weights/full_metrics.csv``
(identity and full-metric columns, one row per game) and ``fixture.json`` (the
metric schema version and how each game was made). It changes no other file and
takes about one minute.

The games are chosen so that wrong summary-weight links show up:

- TDM-Alpha against TDM-Beta on test map 47, one game from each spawn end, with
  two Hunters and no Rogue on Team A and a four-agent team without a Hunter on
  Team B. Two Hunters can trap one enemy on the same tick, which separates
  Trap periods from Trap applications, and the missing classes and empty slot
  check that "not applicable" stays blank.
- Endgame scenarios 1 and 8, played by TDM-Alpha against each scenario's own
  pressure. They start with dead agents, an active Mage Burst (scenario 1) and
  an agent already stunned (scenario 8), so effects already present at the
  start must be counted the same way by a fraction and its weight.

These games prove nothing about learning. Rebuild the fixture whenever the
metric schema version changes; tests/test_metric_mirror.py checks the version.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation.metric_catalog import METRIC_SCHEMA_VERSION

OUTPUT = Path("tests/fixtures/metric_weights")


def _rows(table: Mapping[str, NDArray[np.generic]]) -> list[dict[str, str]]:
    """Turn one full_metrics table into text rows; NaN becomes an empty cell."""
    rows: list[dict[str, str]] = []
    for index in range(len(next(iter(table.values())))):
        row: dict[str, str] = {}
        for name, values in table.items():
            value: object = values[index]
            if isinstance(value, (float, np.floating)):
                number = float(cast(float, value))
                row[name] = "" if math.isnan(number) else repr(number)
            else:
                row[name] = "" if value is None else str(value)
        rows.append(row)
    return rows


def main() -> None:
    """Play the four games and write the fixture files."""
    games: list[dict[str, str]] = []
    rows: list[dict[str, str]] = []
    field = marl_bgs.evaluate(
        "tdm-alpha",
        "tdm-beta",
        num_episodes=2,
        maps=[47],
        system_roster=["hunter", "hunter", "priest", "warrior", "mage"],
        opponent_roster=["mage", "warrior", "rogue", "priest"],
        metrics="full",
        seed=20261007,
        num_envs=2,
    )
    rows += _rows(field.table("full_metrics"))
    games += [
        {
            "how": "evaluate tdm-alpha vs tdm-beta, map 47, seed 20261007",
            "team_a_roster": "hunter hunter priest warrior mage",
            "team_b_roster": "mage warrior rogue priest",
        }
    ] * 2
    for result, scenario in zip(
        marl_bgs.evaluate_scenarios(
            "tdm-alpha", (1, 8), num_episodes=1, metrics="full", seed=20261007
        ),
        (1, 8),
        strict=True,
    ):
        rows += _rows(result.table("full_metrics"))
        games.append({"how": f"evaluate_scenarios tdm-alpha, scenario {scenario}"})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = list(rows[0])
    with (OUTPUT / "full_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    record = {"metric_schema_version": METRIC_SCHEMA_VERSION, "games": games}
    (OUTPUT / "fixture.json").write_text(json.dumps(record, indent=1) + "\n")
    print(OUTPUT / "full_metrics.csv", len(rows), "games")


if __name__ == "__main__":
    main()
