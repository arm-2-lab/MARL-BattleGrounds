"""Check the training_curves experiment stage.

Contracts: the packaged settings with +training_runs=season_0 compose to the six
Season 0 baselines in rank order, each named by its released repo, and with
+training_runs=season_0_local to the same baselines, each with its 40M run and
then its 250M continuation under runs_dir; the stage joins each lineage's
routine validation checks in training order, numbers the runs, sums the check's
games and copies margin, kills (kills_for) and deaths (kills_against) and the
share of games won; given opponents, every value covers only those opponents,
each weighted equally, and a missing name is rejected; it rejects runs listed
out of order or overlapping, and a folder that is not exactly one training run;
it writes curves.csv, crossings.csv, summary.md (first, best and last check)
and curves.png; first_crossings gives each system's first check with a margin
above zero and its first that won at least half its games, None when none did.
Plotting keeps a backend the caller already uses: when pyplot is loaded, or
inside a Jupyter kernel (its backend named in MPLBACKEND before pyplot loads),
the package's plot helpers leave the backend alone; otherwise they choose the
file-only Agg backend (each case in a fresh Python).
A fake training.list_runs replaces saved run folders.
"""

import csv
import os
import subprocess
import sys
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
        "opponents": [
            {
                "name": "bot",
                "games": 32,
                "wins": 8,
                "margin": margin + 2.0,
                "kills_for": kills + 1.0,
                "kills_against": deaths - 1.0,
            },
            {
                "name": "model",
                "games": 32,
                "wins": 0,
                "margin": margin - 2.0,
                "kills_for": kills - 1.0,
                "kills_against": deaths + 1.0,
            },
        ],
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
        released = compose(
            config_name="training_curves",
            overrides=["+training_runs=season_0", "output_dir=out"],
        )
        config = compose(
            config_name="training_curves",
            overrides=[
                "+training_runs=season_0_local",
                "runs_dir=/r",
                "output_dir=out",
            ],
        )
    assert [lineage.repo for lineage in released.lineages] == [
        f"MARL-BattleGrounds/tdm-season-0-{name}"
        for name in (
            "recurrent-ippo",
            "pqn-vdn",
            "recurrent-mappo",
            "qmix",
            "feedforward-ippo",
            "feedforward-mappo",
        )
    ]
    names = [lineage.name for lineage in config.lineages]
    assert [lineage.name for lineage in released.lineages] == names
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
        "win_share": "0.125",
    }
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| Mine | 4 | -20.0 / 0.5 / 11.0 | +3.0 / 18.0 / 13.0 (60) |" in summary
    assert (tmp_path / "curves.png").stat().st_size > 0
    with (tmp_path / "crossings.csv").open(encoding="utf-8") as file:
        assert next(csv.DictReader(file))["margin_above_zero_steps"] == "50"


def test_first_crossings_find_the_first_winning_check() -> None:
    rows = [
        {"system": "A", "env_steps": step, "margin": margin, "win_share": share}
        for step, margin, share in ((0, -3.0, 0.0), (10, 0.0, 0.5), (20, 2.0, 0.4))
    ]
    rows.append({"system": "B", "env_steps": 0, "margin": -1.0, "win_share": 0.1})
    assert training_curves.first_crossings(rows) == [
        {"system": "A", "margin_above_zero_steps": 20, "half_won_steps": 10},
        {"system": "B", "margin_above_zero_steps": None, "half_won_steps": None},
    ]


def test_stage_rejects_bad_run_lists(fake_runs: None) -> None:
    with pytest.raises(ValueError, match="training order"):
        training_curves.read_lineage("Mine", ["first", "overlap"])
    with pytest.raises(ValueError, match="training order"):
        training_curves.read_lineage("Mine", ["continued", "first"])
    with pytest.raises(ValueError, match="exactly one"):
        training_curves.read_lineage("Mine", ["missing"])


def test_stage_keeps_only_the_named_opponents(fake_runs: None) -> None:
    rows = training_curves.read_lineage("Mine", ["first"], opponents=["bot"])
    assert rows[1] == {
        "system": "Mine",
        "run": 1,
        "env_steps": 40,
        "games": 32,
        "margin": -6.0,
        "kills": 10.0,
        "deaths": 14.0,
        "win_share": 0.25,
    }
    both = training_curves.read_lineage("Mine", ["first"], opponents=["bot", "model"])
    assert (both[1]["margin"], both[1]["games"]) == (-8.0, 64)
    with pytest.raises(ValueError, match="no opponent 'other'"):
        training_curves.read_lineage("Mine", ["first"], opponents=["other"])


KERNEL = "import sys, types; sys.modules['ipykernel'] = types.ModuleType('k'); "


@pytest.mark.parametrize(
    ("preload", "environment", "backend"),
    [
        ("import matplotlib.pyplot; matplotlib.use('svg'); ", None, "svg"),
        (KERNEL, "svg", "svg"),
        ("", None, "agg"),
    ],
)
def test_plotting_keeps_a_backend_the_caller_already_uses(
    preload: str, environment: str | None, backend: str
) -> None:
    code = (
        f"{preload}import matplotlib; "
        "from marl_battlegrounds.training.analysis import _prepare_plot_backend; "
        "_prepare_plot_backend(); import matplotlib.pyplot; "
        "print(matplotlib.get_backend().lower())"
    )
    env = {key: value for key, value in os.environ.items() if key != "MPLBACKEND"}
    if environment is not None:
        env["MPLBACKEND"] = environment
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert done.stdout.strip() == backend
