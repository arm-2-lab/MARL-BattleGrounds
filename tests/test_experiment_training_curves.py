"""Check the training_curves experiment stage.

Contracts: the packaged settings with +training_runs=season_0 compose to the six
Season 0 baselines in rank order, each with its 40M run and then its 250M
continuation under runs_dir; the stage joins each lineage's routine validation
checks in training order, numbers the runs, sums the check's games and copies
margin, kills (kills_for) and deaths (kills_against); it rejects runs listed out
of order or overlapping, and a folder that is not exactly one training run; it
writes curves.csv, summary.md (first, best and last check) and curves.png. A
fake training.list_runs replaces saved run folders.
"""

import csv
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from marl_battlegrounds import training
from marl_battlegrounds.experiments import training_curves


def check(steps: int, margin: float, kills: float, deaths: float) -> dict[str, Any]:
    return {
        "env_steps": steps,
        "opponents": [{"games": 32}, {"games": 32}],
        "margin": margin,
        "kills_for": kills,
        "kills_against": deaths,
    }


RUNS = {
    "first": [check(0, -20.0, 0.5, 11.0), check(40, -8.0, 9.0, 15.0)],
    "continued": [check(50, 1.0, 16.0, 14.0), check(60, 3.0, 18.0, 13.0)],
    "overlap": [check(40, 0.0, 10.0, 10.0)],
}


@pytest.fixture
def fake_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    def list_runs(root: str, *, progress: bool = False) -> tuple[dict[str, Any], ...]:
        assert progress
        name = Path(root).name
        if name == "missing":
            return ()
        return ({"run_dir": root, "progress": {"checks": RUNS[name]}},)

    monkeypatch.setattr(training, "list_runs", list_runs)


def test_season_0_runs_compose() -> None:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        config = compose(
            config_name="training_curves",
            overrides=["+training_runs=season_0", "runs_dir=/r", "output_dir=out"],
        )
    names = [lineage.name for lineage in config.lineages]
    assert names == [
        "Recurrent IPPO",
        "PQN-VDN",
        "Recurrent MAPPO",
        "QMIX",
        "Feedforward IPPO",
        "Feedforward MAPPO",
    ]
    for lineage in config.lineages:
        first, continued = lineage.runs
        assert first.startswith("/r/") and first.endswith("/run")
        assert continued.split("/")[2] in {"250m", "ff-mappo-250m"}
        assert first.split("/cases/")[1] == continued.split("/cases/")[1]


def test_stage_joins_runs_and_writes_files(fake_runs: None, tmp_path: Path) -> None:
    config = OmegaConf.create(
        {
            "lineages": [{"name": "Mine", "runs": ["a/first", "a/continued"]}],
            "output_dir": str(tmp_path),
        }
    )
    assert training_curves.main(config) == {"Mine": 4}  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.

    with (tmp_path / "curves.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [(row["run"], row["env_steps"]) for row in rows] == [
        ("1", "0"),
        ("1", "40"),
        ("2", "50"),
        ("2", "60"),
    ]
    assert rows[2] == {
        "system": "Mine",
        "run": "2",
        "env_steps": "50",
        "games": "64",
        "margin": "1.0",
        "kills": "16.0",
        "deaths": "14.0",
    }
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| Mine | 4 | -20.0 / 0.5 / 11.0 | +3.0 / 18.0 / 13.0 (60) |" in summary
    assert (tmp_path / "curves.png").stat().st_size > 0


def test_stage_rejects_bad_run_lists(fake_runs: None) -> None:
    with pytest.raises(ValueError, match="training order"):
        training_curves.read_lineage("Mine", ["first", "overlap"])
    with pytest.raises(ValueError, match="training order"):
        training_curves.read_lineage("Mine", ["continued", "first"])
    with pytest.raises(ValueError, match="exactly one"):
        training_curves.read_lineage("Mine", ["missing"])
