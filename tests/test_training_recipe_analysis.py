"""Check mean-recipe selection from saved independent training vectors.

Synthetic evidence separates recipe means from lucky runs and keeps missing
cells, seeds and lineage visible. It establishes report behavior, not learning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.test_training_uncertainty import _grouping, _reduce, _run

from marl_battlegrounds.training import analysis

# pyright: reportPrivateUsage=false


def _recipe(
    tmp_path: Path, margins: tuple[float, ...] = (9, -9, 2, 2)
) -> dict[str, Any]:
    entries: list[tuple[Path, str, str]] = []
    for index, margin in enumerate(margins):
        root = _run(tmp_path / f"run{index}", (0.5,) * 4)
        path = root / "selection.json"
        selected = json.loads(path.read_text())
        for cell in selected["cells"]:
            cell.update(mean_team_a_score=10 + margin, mean_team_b_score=10)
        path.write_text(json.dumps(selected))
        entries.append((root, "Lucky" if index < 2 else "Steady", f"unit{index}"))
    return {
        **_grouping(entries),
        "schema_version": 2,
        "measure": "point_margin",
        "recipe_selection": [
            {
                "name": "Mean",
                "groups": ["Lucky", "Steady"],
                "direction": "maximize",
                "tie_order": ["Lucky", "Steady"],
            }
        ],
    }


def test_mean_recipe_beats_one_lucky_run_and_public_output_is_reusable(
    tmp_path: Path,
) -> None:
    grouping = _recipe(tmp_path)
    report = analysis.analyze(
        [row["run_dir"] for row in grouping["runs"]],
        grouping=grouping,
        output_dir=tmp_path / "report",
    )
    variation = report["training_variation"]
    choice = variation["recipe_selection"]["decisions"][0]
    assert choice["winner"] == "Steady"
    assert [row["mean"] for row in choice["candidates"]] == [0, 2]
    assert variation["runs"][0]["cells"][0]["game_score"] == 0.5
    assert variation["runs"][0]["cells"][0]["point_margin"] == 9
    saved = json.loads(Path(report["artifacts"]["recipe_selection"]).read_text())
    assert saved == variation["recipe_selection"]
    assert _reduce(variation["declaration"]) == variation


@pytest.mark.parametrize(
    "direction,winner", [("maximize", "Steady"), ("minimize", "Lucky")]
)
def test_declared_direction_and_exact_tie_order(
    tmp_path: Path, direction: str, winner: str
) -> None:
    grouping = _recipe(tmp_path)
    grouping["recipe_selection"][0]["direction"] = direction
    assert _reduce(grouping)["recipe_selection"]["decisions"][0]["winner"] == winner
    for entry in grouping["runs"]:
        path = Path(entry["run_dir"]) / "selection.json"
        selected = json.loads(path.read_text())
        for cell in selected["cells"]:
            cell["mean_team_a_score"] = 10
        path.write_text(json.dumps(selected))
    grouping["recipe_selection"][0]["tie_order"] = ["Steady", "Lucky"]
    assert _reduce(grouping)["recipe_selection"]["decisions"][0]["winner"] == "Steady"


def test_weights_zero_cell_and_missing_candidate_remain_required(
    tmp_path: Path,
) -> None:
    grouping = _recipe(tmp_path)
    grouping["cell_weights"] = [
        {"map_id": map_id, "opponent": opponent, "weight": weight}
        for (map_id, opponent), weight in zip(
            [(42, "Alpha"), (42, "Beta"), (43, "Alpha"), (43, "Beta")],
            [3, 1, 0, 0],
            strict=True,
        )
    ]
    complete = _reduce(grouping)
    assert complete["recipe_selection"]["normalized_cell_weights"] == [0.75, 0.25, 0, 0]
    path = Path(grouping["runs"][0]["run_dir"]) / "selection.json"
    selected = json.loads(path.read_text())
    selected["cells"].pop()
    path.write_text(json.dumps(selected))
    incomplete = _reduce(grouping)["recipe_selection"]["decisions"][0]
    assert incomplete["status"] == "incomplete" and incomplete["winner"] is None
    assert incomplete["candidates"][0]["available_mean"] == -9
    assert incomplete["candidates"][1]["mean"] == 2


@pytest.mark.parametrize("change", ["seed", "lineage", "nonfinite", "absent"])
def test_unknown_or_missing_evidence_cannot_select(tmp_path: Path, change: str) -> None:
    grouping = _recipe(tmp_path)
    root = Path(grouping["runs"][0]["run_dir"])
    if change in ("seed", "lineage"):
        path = root / "run_details.json"
        details = json.loads(path.read_text())
        if change == "seed":
            details["config"].pop("seed")
        else:
            details["continuation"] = {"parent_run_id": "unknown"}
        path.write_text(json.dumps(details))
    else:
        path = root / "selection.json"
        selected = json.loads(path.read_text())
        if change == "nonfinite":
            selected["cells"][0]["mean_team_a_score"] = float("nan")
        else:
            selected["cells"][0].pop("mean_team_a_score")
        path.write_text(json.dumps(selected))
    report = _reduce(grouping)
    assert report["recipe_selection"]["decisions"][0]["winner"] is None
    json.dumps(report, allow_nan=False)


def test_one_unit_is_software_choice_without_training_interval(tmp_path: Path) -> None:
    grouping = _recipe(tmp_path)
    grouping["runs"] = [grouping["runs"][0], grouping["runs"][2]]
    report = _reduce(grouping)
    assert report["recipe_selection"]["decisions"][0]["winner"] == "Lucky"
    assert all(row["ci_low"] is None for row in report["groups"])


@pytest.mark.parametrize(
    "change",
    [
        {"measure": "kills"},
        {"cell_weights": []},
        {"cell_weights": [{"map_id": 42, "opponent": "Alpha", "weight": -1}]},
        {
            "recipe_selection": [
                {
                    "name": "Bad",
                    "groups": ["Lucky"],
                    "direction": "maximize",
                    "tie_order": [],
                }
            ]
        },
    ],
)
def test_bad_declarations_fail_without_outputs(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    grouping = {**_recipe(tmp_path), **change}
    with pytest.raises(ValueError):
        analysis.analyze(
            [row["run_dir"] for row in grouping["runs"]],
            grouping=grouping,
            output_dir=tmp_path / "bad",
        )
    assert not (tmp_path / "bad").exists()
