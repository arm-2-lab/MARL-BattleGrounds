"""Check the llm_challenge experiment stage.

Contracts: the packaged settings name the pinned Qwen3.5-9B recipe, test maps
47-51, two games per map, full metrics, an 8-hour limit and no scenarios, with
the scenarios stage's 256 games, 128 side by side and seed 20261006 when
scenarios are asked for, and with +field=season_0 they play the nine field
members because Random is skipped; the stage plays map by map, every opponent
once per map, sums each pair's outcomes, points, kills, deaths, ticks and Team
A model costs into runs.json, skips pairs already in runs.json when run again,
starts no pair once the time limit is used up, and writes one results row per
opponent; with no maps it plays each listed scenario once through
evaluate_scenarios, saves it in scenarios/scenario_runs.json, skips saved
scenarios when run again, and writes the scenarios stage's results table and
costs.csv; it refuses an odd games_per_map and settings with neither maps nor
scenarios. Fakes replace evaluate, evaluate_scenarios, load_method,
make_system, call_summary and the saved-outcome reader, so no model server,
saved opponent or game is needed.
"""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

import marl_battlegrounds
from marl_battlegrounds import llm
from marl_battlegrounds.experiments import llm_challenge, scenarios


class FakeResult:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def table(self, name: str) -> dict[str, Any]:
        assert name == "priority_metrics"
        return {
            "episode_length": np.array([100, 60]),
            "team_a_win": np.array([0, 0]),
            "team_a_draw": np.array([1, 0]),
            "team_a_loss": np.array([0, 1]),
            "team_a_score": np.array([3, 1]),
            "team_b_score": np.array([3, 20]),
            "team_a_kills": np.array([3, 1]),
            "team_b_kills": np.array([3, 20]),
            "team_a_deaths": np.array([3, 20]),
        }


@pytest.fixture
def played(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []

    def evaluate(system: object, opponent: str, **options: object) -> FakeResult:
        assert system == "llm-system"
        assert (options["num_episodes"], options["spawn_mode"]) == (2, "paired")
        calls.append((cast(list[int], options["maps"])[0], opponent))
        return FakeResult(Path(cast(str, options["output_dir"])) / "run")

    costs = {"model_calls": 800, "input_tokens": 9, "output_tokens": 3}
    costs |= {"fallback_actions": 2, "reply_failures": 2}

    def make_system(*args: object, **kwargs: object) -> str:
        return "llm-system"

    def call_summary(result: FakeResult) -> dict[str, dict[str, dict[str, int]]]:
        return {"completed_games": {"team_a": costs}}

    def evaluate_scenarios(
        system: object, *, scenario_ids: list[int], **options: object
    ) -> tuple[FakeResult]:
        assert system == "llm-system"
        assert (options["num_episodes"], options["seed"]) == (256, 20261006)
        calls.append((scenario_ids[0], "scenario"))
        folder = Path(cast(str, options["output_dir"])) / f"scenario-{scenario_ids[0]}"
        return (FakeResult(folder),)

    def count_outcomes(run_dir: object) -> dict[str, int]:
        return {"games": 256, "wins": 1, "draws": 0, "losses": 255}

    monkeypatch.setattr(marl_battlegrounds, "evaluate", evaluate)
    monkeypatch.setattr(marl_battlegrounds, "evaluate_scenarios", evaluate_scenarios)
    monkeypatch.setattr(scenarios, "count_outcomes", count_outcomes)
    monkeypatch.setattr(marl_battlegrounds, "load_method", str)
    monkeypatch.setattr(llm, "make_system", make_system)
    monkeypatch.setattr(llm, "call_summary", call_summary)
    return calls


def settings(output_dir: Path, **changes: object) -> DictConfig:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        config = compose(config_name="llm_challenge", overrides=["output_dir=x"])
    values = cast(dict[str, Any], OmegaConf.to_container(config))
    values["output_dir"] = str(output_dir)
    values["systems"] = [
        {"name": "TDM-Alpha", "method": "tdm-alpha"},
        {"name": "Random", "method": "random"},
        {"name": "TDM-Beta", "method": "tdm-beta"},
    ]
    values.pop("hydra", None)
    return OmegaConf.create(values | {"maps": [47, 48]} | changes)


def test_settings_compose() -> None:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        config = compose(config_name="llm_challenge", overrides=["output_dir=x"])
        field = compose(
            config_name="llm_challenge",
            overrides=["+field=season_0", "models_dir=/m", "output_dir=x"],
        )
    assert (config.model, config.model_revision) == (
        "qwen3.5-9b-fp8",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    )
    assert list(config.maps) == [47, 48, 49, 50, 51]
    assert (config.games_per_map, config.metrics, config.time_limit_hours) == (
        2,
        "full",
        8.0,
    )
    names = [entry.name for entry in field.systems if entry.name not in field.skip]
    assert len(names) == 9 and names[0] == "Recurrent IPPO"
    assert (list(config.scenario_ids), config.scenario_games) == ([], 256)
    assert (config.scenario_envs, config.scenario_seed) == (128, 20261006)


def test_stage_plays_map_by_map_and_resumes(
    played: list[tuple[int, str]], tmp_path: Path
) -> None:
    rows = llm_challenge.main(settings(tmp_path))["field"]  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.
    assert played == [
        (47, "tdm-alpha"),
        (47, "tdm-beta"),
        (48, "tdm-alpha"),
        (48, "tdm-beta"),
    ]
    pairs = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))["pairs"]
    assert pairs[0]["run_dir"].endswith("map-47/tdm-alpha/run")
    assert {key: pairs[0][key] for key in ("wins", "draws", "losses", "ticks")} == {
        "wins": 0,
        "draws": 1,
        "losses": 1,
        "ticks": 160,
    }
    assert rows[0] | {"seconds_per_game": 0} == {
        "opponent": "TDM-Alpha",
        "games": 4,
        "wins": 0,
        "draws": 2,
        "losses": 2,
        "margin": -9.5,
        "kills": 2.0,
        "deaths": 11.5,
        "ticks": 80.0,
        "fallback_percent": 0.25,
        "seconds_per_game": 0,
    }
    assert "| TDM-Beta | 4 | 0 · 2 · 2 | -9.50 |" in (
        tmp_path / "results.md"
    ).read_text(encoding="utf-8")

    llm_challenge.main(settings(tmp_path))  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.
    assert len(played) == 4


def test_time_limit_and_bad_settings(
    played: list[tuple[int, str]], tmp_path: Path
) -> None:
    assert llm_challenge.main(settings(tmp_path, time_limit_hours=0.0)) == {  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.
        "field": [],
        "scenarios": [],
    }
    assert played == []
    with pytest.raises(ValueError, match="even"):
        llm_challenge.main(settings(tmp_path, games_per_map=3))  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.
    with pytest.raises(ValueError, match="maps, scenario_ids or both"):
        llm_challenge.main(settings(tmp_path, maps=[]))  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.


def test_stage_plays_scenarios_and_resumes(
    played: list[tuple[int, str]], tmp_path: Path
) -> None:
    config = settings(tmp_path, maps=[], scenario_ids=[1, 3])
    records = llm_challenge.main(config)["scenarios"]  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.
    assert played == [(1, "scenario"), (3, "scenario")]
    assert [record["scenario"] for record in records] == [1, 3]
    assert records[0]["run_dir"].endswith("scenarios/scenario-1")
    assert (records[0]["wins"], records[0]["fallback_actions"]) == (1, 2)
    folder = tmp_path / "scenarios"
    assert "| qwen3.5-9b-fp8 | 1 · 0 · 255 | 1 · 0 · 255 |" in (
        folder / "results.md"
    ).read_text(encoding="utf-8")
    assert (
        (folder / "costs.csv")
        .read_text(encoding="utf-8")
        .startswith(
            "scenario,games,model_calls,input_tokens,output_tokens,fallback_actions"
        )
    )
    assert not (tmp_path / "runs.json").exists()

    llm_challenge.main(config)  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.
    assert len(played) == 2
