"""Run the complete synthetic prepare/analyze/load workflow through public entries.

The fixture is invented saved evidence. These CPU host checks cover first-call
recipe choice/index publication, portable paths, clear collisions and no hidden
experiment launch. They also check three analyze outputs. Recipe learning
curves: known per-seed point margins give the expected CSV rows and recipe
means, every seed gets its own line, each method gets its own panel, only the
grouping's maps, opponents and panel count, and a report with no validation
results skips the curves and says so. The base pick rule: a challenger replaces
its base only with a mean gain above min_margin_gain (exactly equal fails) and
at least min_units_won paired units won; partial or invalid rules raise. The
gradient-norm panel: None, missing keys and non-finite counts never fail.
They establish software behavior, never learning results.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pytest
from omegaconf import OmegaConf

from marl_battlegrounds import training
from marl_battlegrounds.experiments.prepare import prepare_analysis_software_check
from marl_battlegrounds.training import analysis

# pyright: reportPrivateUsage=false

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Record = dict[str, Any]
CELLS = ((42, "Alpha"), (42, "Beta"), (43, "Alpha"), (43, "Beta"))


@pytest.mark.parametrize("empty", [False, True])
def test_public_recipe_commands_and_checked_relocation(
    tmp_path: Path, empty: bool
) -> None:
    root = tmp_path / "example"
    grouping = prepare_analysis_software_check(root)
    if empty:
        run = root / "calibration/study/cases/lucky--seed-101/run"
        shutil.rmtree(run)
        run.mkdir()
        path = root / "calibration/study/study.json"
        state = json.loads(path.read_text())
        state["cases"]["lucky--seed-101"].update(status="pending", attempts=[])
        path.write_text(json.dumps(state))
    output = root / "reports"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "marl_battlegrounds.experiments.analyze",
            f"study_dirs=[{root / 'calibration/study'}]",
            f"grouping={grouping}",
            f"output_dir={output}",
        ],
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    recipe = json.loads(Path(report["reports"][0]["recipe_selection"]).read_text())
    assert recipe["decisions"][0]["winner"] == (None if empty else "steady")
    assert [
        candidate["mean"] for candidate in recipe["decisions"][0]["candidates"]
    ] == ([None, 2] if empty else [0, 2])
    paths = training.load_study_data(output / "analysis-data")
    index = json.loads(paths["data_index.json"].read_text())
    assert len(index["tables"]) == 13
    assert index["tables"]["selection/recipes.csv"]["rows"] == 2
    with paths["run_index.csv"].open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["recipe"] for row in rows] == ["lucky", "lucky", "steady", "steady"]
    if empty:
        assert rows[0]["status"] == "pending" and rows[0]["attempt_id"] == ""
    destination = tmp_path / "moved"
    root.rename(destination)
    loaded = training.load_study_data(destination / "reports/analysis-data")
    assert loaded["selection/recipes.csv"].exists()


@pytest.mark.parametrize("damage", ["nonempty", "invalid_details"])
def test_grouping_does_not_hide_damaged_admitted_runs(
    tmp_path: Path, damage: str
) -> None:
    root = tmp_path / "example"
    path = prepare_analysis_software_check(root)
    grouping = cast(
        dict[str, Any], OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    )
    assert isinstance(grouping, dict)
    for row in grouping["runs"]:
        row["run_dir"] = str((path.parent / row["run_dir"]).resolve())
    damaged = Path(grouping["runs"][0]["run_dir"]) / "run_details.json"
    if damage == "nonempty":
        damaged.unlink()
    else:
        damaged.write_text("[]")
    with pytest.raises(ValueError, match=r"run_details\.json"):
        training.analyze(
            [row["run_dir"] for row in grouping["runs"]],
            grouping=grouping,
            output_dir=root / "out",
        )


def _cells(margin: float, *, extra: bool = False) -> list[Record]:
    cells: list[Record] = [
        {
            "map_id": map_id,
            "opponent": opponent,
            "score": 0.5,
            "mean_team_a_score": 10 + margin,
            "mean_team_b_score": 10,
        }
        for map_id, opponent in CELLS
    ]
    if extra:
        cells.append({**cells[0], "map_id": 45, "mean_team_a_score": 99})
    return cells


def _result(purpose: str, steps: int, margin: float, **fields: object) -> Record:
    return {
        "checkpoint_id": f"{purpose}-{steps}",
        "actor_digest": f"digest-{steps}",
        "panel_digest": "frozen-panel",
        "purpose": purpose,
        "complete": True,
        "red_zone_depth": 5.0,
        "env_steps": steps,
        "score": 0.5,
        "cells": _cells(margin, extra=True),
        **fields,
    }


def _seed_run(
    path: Path,
    *,
    method: str,
    seed: int,
    selected: float,
    curve: tuple[float, ...] = (),
) -> Path:
    path.mkdir(parents=True)
    results = [
        _result("initialization" if index == 0 else "routine", 100 * index, margin)
        for index, margin in enumerate(curve)
    ]
    if curve:
        results.append(_result("routine", 100, 50.0, panel_digest="other-panel"))
    selection = {
        **_result("confirmation", 100 * max(len(curve) - 1, 1), selected),
        "cells": _cells(selected),
    }
    for name, value in (
        (
            "run_details.json",
            {
                "run_id": path.name,
                "config": {"seed": seed, "method": method, "red_zone_depth": 5.0},
            },
        ),
        ("status.json", {"status": "complete"}),
        ("selection.json", selection),
        ("validation_results.json", results),
    ):
        (path / name).write_text(json.dumps(value))
    return path


def _grouping(runs: list[tuple[Path, str, str]], **fields: object) -> Record:
    return {
        "schema_version": 2,
        "measure": "point_margin",
        "maps": [42, 43],
        "opponents": ["Alpha", "Beta"],
        "panel_digest": "frozen-panel",
        "red_zone_depth": 5.0,
        "runs": [
            {"run_dir": str(path), "group": group, "unit": unit}
            for path, group, unit in runs
        ],
        **fields,
    }


def test_recipe_curves_draw_every_seed_and_known_recipe_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curves = {
        "steady": ("mappo", ((0, 1, 2), (0, 3, 4), (0, 2, 6))),
        "lucky": ("mappo", ((0, -1, 9), (0, -3, -3), (0, -2, 0))),
        "other": ("ippo", ((1, 2, 3), (-1, 0, 1), (0, 1, 2))),
    }
    runs: list[tuple[Path, str, str]] = []
    for group, (method, seeds) in curves.items():
        for index, curve in enumerate(seeds):
            path = _seed_run(
                tmp_path / f"{group}-{index}",
                method=method,
                seed=101 + index,
                selected=curve[-1],
                curve=curve,
            )
            runs.append((path, group, f"seed-{index}"))
    figures: list[Any] = []
    monkeypatch.setattr(plt, "close", figures.append)
    report = training.analyze(
        None, grouping=_grouping(runs), output_dir=tmp_path / "report"
    )
    artifacts = report["artifacts"]
    with Path(artifacts["recipe_curves_csv"]).open() as stream:
        rows = list(csv.DictReader(stream))
    assert Path(artifacts["recipe_curves_png"]).stat().st_size > 0
    assert Path(artifacts["recipe_curves_svg"]).stat().st_size > 0
    # The map 45 cell (margin 89) and the other-panel result (margin 50) are
    # outside the declared field, so they never change a margin.
    assert [{**row, "point_margin": float(row["point_margin"])} for row in rows] == [
        {
            "group": group,
            "unit": f"seed-{index}",
            "run_dir": str(tmp_path / f"{group}-{index}"),
            "method": method,
            "env_steps": str(100 * step),
            "point_margin": float(margin),
            "cells_used": "4",
        }
        for group, (method, seeds) in curves.items()
        for index, curve in enumerate(seeds)
        for step, margin in enumerate(curve)
    ]
    figure = next(
        item
        for item in figures
        if item._suptitle is not None and "Recipe" in item._suptitle.get_text()
    )
    axes = [axis for axis in figure.axes if axis.get_visible()]
    assert [axis.get_title() for axis in axes] == ["Method: mappo", "Method: ippo"]
    for axis, groups in zip(axes, (("steady", "lucky"), ("other",)), strict=True):
        thin = [line for line in axis.lines if line.get_linewidth() == 0.9]
        thick = {
            line.get_label(): list(line.get_ydata())
            for line in axis.lines
            if line.get_linewidth() == 2.8
        }
        assert len(thin) == 3 * len(groups)
        assert thick == {
            f"{group}: Mean of 3 Seeds": [
                sum(step) / 3 for step in zip(*curves[group][1], strict=True)
            ]
            for group in groups
        }
    summary = Path(artifacts["summary"]).read_text()
    assert "## Recipe Learning Curves" in summary and "recipe_curves.csv" in summary
    assert report["recipe_curves"].startswith("Recipe learning curves: ")


def test_recipe_curves_skip_without_validation_results(tmp_path: Path) -> None:
    runs = [
        (
            _seed_run(
                tmp_path / f"run-{index}", method="mappo", seed=index, selected=1
            ),
            "only",
            f"seed-{index}",
        )
        for index in range(2)
    ]
    output = tmp_path / "report"
    output.mkdir()
    (output / "recipe_curves.png").write_text("stale")
    report = training.analyze(None, grouping=_grouping(runs), output_dir=output)
    assert not any(key.startswith("recipe_curves") for key in report["artifacts"])
    assert not (output / "recipe_curves.png").exists()
    assert "skipped" in report["recipe_curves"]
    assert report["recipe_curves"] in Path(report["artifacts"]["summary"]).read_text()


def test_base_rule_needs_gain_above_threshold_and_enough_units_won(
    tmp_path: Path,
) -> None:
    margins = {
        "base": (0, 0, 0),
        "exact": (2, 2, 2),
        "passes": (4, 4, -1),
        "one-unit": (7.5, -0.5, -0.5),
    }
    runs = [
        (
            _seed_run(
                tmp_path / f"{group}-{index}",
                method="qmix",
                seed=101 + index,
                selected=margin,
            ),
            group,
            f"seed-{index}",
        )
        for group, values in margins.items()
        for index, margin in enumerate(values)
    ]
    rule = {"base": "base", "min_margin_gain": 2, "min_units_won": 2}
    grouping = _grouping(
        runs,
        recipe_selection=[
            {
                "name": "settings",
                "groups": list(margins),
                "direction": "maximize",
                "tie_order": list(margins),
                **rule,
            },
            {
                "name": "exact-only",
                "groups": ["base", "exact"],
                "direction": "maximize",
                "tie_order": ["exact", "base"],
                **rule,
            },
        ],
    )
    report = training.analyze(None, grouping=grouping, output_dir=tmp_path / "out")
    settings, exact = report["training_variation"]["recipe_selection"]["decisions"]
    found = {
        row["group"]: (row["mean_gain"], row["units_won"], row["passed"])
        for row in settings["challengers"]
    }
    assert found == {
        "exact": (2.0, 3, False),
        "passes": (pytest.approx(7 / 3), 2, True),
        "one-unit": (pytest.approx(6.5 / 3), 1, False),
    }
    assert (settings["winner"], settings["passed_challengers"]) == (
        "passes",
        ["passes"],
    )
    assert all(row["paired_units"] == 3 for row in settings["challengers"])
    assert (exact["winner"], exact["passed_challengers"]) == ("base", [])
    assert settings["min_margin_gain"] == 2.0
    saved = json.loads(Path(report["artifacts"]["recipe_selection"]).read_text())
    assert saved == report["training_variation"]["recipe_selection"]


@pytest.mark.parametrize(
    "change",
    [
        {"base": "base"},
        {"base": "missing", "min_margin_gain": 2, "min_units_won": 2},
        {"base": "base", "min_margin_gain": -1, "min_units_won": 2},
        {"base": "base", "min_margin_gain": 2, "min_units_won": True},
    ],
)
def test_invalid_base_rules_raise(tmp_path: Path, change: Record) -> None:
    runs = [
        (_seed_run(tmp_path / name, method="mappo", seed=1, selected=0), name, "s")
        for name in ("base", "other")
    ]
    choice = {
        "name": "pick",
        "groups": ["base", "other"],
        "direction": "maximize",
        "tie_order": ["base", "other"],
        **change,
    }
    with pytest.raises(ValueError, match=r"base|min_"):
        analysis._training_grouping(_grouping(runs, recipe_selection=[choice]), None)


def test_gradient_norm_panel_handles_none_missing_keys_and_nonfinite(
    tmp_path: Path,
) -> None:
    new = [
        {
            "env_steps": 10,
            "grad_norm_mean": 1.5,
            "grad_norm_max": 3.0,
            "grad_norm_nonfinite": 0,
        },
        {
            "env_steps": 20,
            "grad_norm_mean": None,
            "grad_norm_max": None,
            "grad_norm_nonfinite": 2,
        },
        {
            "env_steps": 30,
            "grad_norm_mean": 2.0,
            "grad_norm_max": float("inf"),
            "grad_norm_nonfinite": 0,
        },
    ]
    old = [{"env_steps": 10, "loss": 1.0}, {"env_steps": 20, "loss": 0.5}]
    figure, (both, legacy, failed) = plt.subplots(1, 3)
    analysis._grad_norm_panel(both, [("New", new), ("Old", old)])
    analysis._grad_norm_panel(legacy, [("Old", old)])
    analysis._grad_norm_panel(
        failed,
        [
            (
                "Failed",
                [
                    {
                        "env_steps": 5,
                        "grad_norm_mean": None,
                        "grad_norm_max": None,
                        "grad_norm_nonfinite": 4,
                    }
                ],
            )
        ],
    )
    lines = {line.get_label(): line for line in both.lines}
    assert set(lines) == {"New: Mean", "New: Max", "Update With a Non-Finite Norm"}
    np.testing.assert_array_equal(lines["New: Mean"].get_ydata(), [1.5, np.nan, 2.0])
    np.testing.assert_array_equal(lines["New: Max"].get_ydata(), [3.0, np.nan, np.nan])
    assert list(lines["Update With a Non-Finite Norm"].get_xdata()) == [20, 20]
    assert both.get_yscale() == "log"
    assert not legacy.lines and legacy.get_yscale() == "linear"
    assert [text.get_text() for text in legacy.texts] == [
        "No Gradient Norms Saved in These Runs"
    ]
    assert len(failed.lines) == 1 and failed.get_yscale() == "linear"
    plt.close(figure)
    paths = {
        **analysis._qmix_figure([("New", new), ("Old", old)], tmp_path, plt),
        **analysis._pqn_figure([("Old", old)], tmp_path, plt),
    }
    assert all(Path(path).stat().st_size > 0 for path in paths.values())
