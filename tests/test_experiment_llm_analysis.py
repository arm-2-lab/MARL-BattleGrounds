"""Check the llm_analysis experiment stage.

Contracts: the packaged settings leave field_dir and scenarios_dir null;
team_values reads Team A as the model's team and Team B as its opponent for
every MEASURES column, one value per game; answer_summary counts only Team A
rows whose decision was played, separates forced turns from calls, gives the
legal share of calls, the no-combat share and per-move shares of legal answers
and the median call time, and returns None shares when nothing counts; the
stage refuses settings with neither folder and, given only scenarios,
writes answers.csv and analysis.md but no behavior.csv. Hand-made tables and
call rows replace saved runs.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

from marl_battlegrounds import llm
from marl_battlegrounds.experiments import llm_analysis


def call(
    source: str, action: list[int] | None = None, **extra: object
) -> dict[str, Any]:
    return {"team": 0, "outcome": "played", "source": source, "action": action} | extra


def test_settings_compose() -> None:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        config = compose(config_name="llm_analysis", overrides=["output_dir=o"])
    assert (config.field_dir, config.scenarios_dir) == (None, None)


def test_team_values_reads_both_sides() -> None:
    full: dict[str, Any] = {}
    for _, column in llm_analysis.MEASURES:
        full[f"team_a_{column}"] = np.array([1.0, 3.0])
        full[f"team_b_{column}"] = np.array([10.0, np.nan])
    values = llm_analysis.team_values(full)
    assert values["model"]["basic_uses"] == [1.0, 3.0]
    assert values["opponent"]["own_red_zone_deaths"][0] == 10.0
    assert np.isnan(values["opponent"]["ally_distance"][1])
    with pytest.raises(KeyError):
        llm_analysis.team_values({"team_a_deaths": np.zeros(1)})


def test_answer_summary_counts_played_team_a_answers() -> None:
    rows = [
        call("model", [3, 0, 0], seconds=1.0),
        call("model", [1, 8, 0], seconds=3.0),
        call("fallback", [0, 0, 0], seconds=2.0, failure="ReplyFormatError"),
        call("forced", [0, 0, 0]),
        call("model", [3, 0, 0], outcome="abandoned"),
        call("model", [3, 0, 0]) | {"team": 1},
    ]
    summary = llm_analysis.answer_summary(rows)
    assert (summary["calls"], summary["forced_turns"]) == (3, 1)
    assert summary["legal_percent"] == pytest.approx(200 / 3)
    assert summary["failures"] == {"ReplyFormatError": 1}
    assert summary["no_combat_percent"] == 50.0
    assert summary["move_percent"]["east"] == 50.0
    assert summary["move_percent"]["stay"] == 0.0
    assert summary["median_call_seconds"] == 2.0
    empty = llm_analysis.answer_summary([])
    assert empty["legal_percent"] is None and empty["median_call_seconds"] is None


def test_stage_refuses_no_input_and_reads_scenarios_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def settings(**values: object) -> DictConfig:
        return OmegaConf.create({"field_dir": None, "scenarios_dir": None} | values)

    def read_calls(run_dir: object) -> Iterator[dict[str, Any]]:
        return iter(rows)

    with pytest.raises(ValueError, match="field_dir, scenarios_dir or both"):
        llm_analysis.main(settings(output_dir=str(tmp_path / "out")))  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.

    record = {"run_dir": "r", "games": 2, "model_calls": 2, "seconds": 4.0}
    record |= {"input_tokens": 30, "output_tokens": 6}
    (tmp_path / "scenario_runs.json").write_text(json.dumps({"scenarios": [record]}))
    rows = [call("model", [3, 0, 0], seconds=1.0), call("model", [1, 6, 1])]
    monkeypatch.setattr(llm, "read_calls", read_calls)
    counts: dict[str, int] = llm_analysis.main(  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.
        settings(scenarios_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    )
    assert counts == {"behavior": 0, "answers": 1}
    answers = (tmp_path / "out" / "answers.csv").read_text(encoding="utf-8")
    assert answers.startswith("experiment,games,calls,legal_percent")
    assert "scenarios,2,2,100.0,,0,15.0,3.0,1.0,4.0,50.0" in answers
    assert not (tmp_path / "out" / "behavior.csv").exists()
    assert "Scenarios: 2 calls, 100.0% legal" in (
        tmp_path / "out" / "analysis.md"
    ).read_text(encoding="utf-8")
