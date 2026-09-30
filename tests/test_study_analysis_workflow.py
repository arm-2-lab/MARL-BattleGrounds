"""Run the complete synthetic prepare/analyze/load workflow through public entries.

The fixture is invented saved evidence. These CPU host checks cover first-call
recipe choice/index publication, portable paths, clear collisions and no hidden
experiment launch. They establish software behavior, never learning results.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from marl_battlegrounds import training
from marl_battlegrounds.experiments.prepare import prepare_analysis_software_check


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
