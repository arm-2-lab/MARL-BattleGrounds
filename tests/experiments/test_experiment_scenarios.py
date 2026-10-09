"""Check the scenarios experiment stage and shared authored-start preparation.

Contracts: the evaluator builds one authored scenario start per batch for lanes
that share the same start and configuration, keeps ordinary resets per lane, and
stacks one row per lane; the packaged scenarios settings compose to the four
built-in systems by default and to the nine-system Season 0 field plus Random
with +field=season_0, filling models_dir into the six actor paths; the stage counts
wins, draws and losses from Team A's outcome column and writes results.csv and
results.md in system order. The scenario_analysis stage's Fisher's exact test
matches known p-values and rejects impossible counts; it matches full-metrics
rows to games by episode ID, counts each starting agent's deaths and Ultimate
use per outcome with the scenario's starting health and cooldown, counts game
lengths per outcome, and summarizes losses within two ticks. Fakes replace
games, so no JAX program is compiled.
"""

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.experiments import scenario_analysis, scenarios

# The package also exports a function named evaluate, so load the module itself.
evaluate = importlib.import_module("marl_battlegrounds.evaluation.evaluate")


def composed(*overrides: str) -> DictConfig:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        return compose(config_name="scenarios", overrides=list(overrides))


def test_shared_authored_start_is_built_once_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object]] = []

    def start(initial_state: object, config: object) -> tuple[Any, ...]:
        calls.append((initial_state, config))
        return jnp.arange(3.0), jnp.ones(2), jnp.zeros(4, bool), None

    def reset(config: object, key: object) -> tuple[Any, ...]:
        return jnp.full(3, 9.0), jnp.ones(2), jnp.zeros(4, bool), None

    monkeypatch.setattr(evaluate.core, "initialize_scenario_state", start)
    monkeypatch.setattr(evaluate.core, "reset", reset)
    authored, config = object(), object()
    specs = [SimpleNamespace(initial_state=authored, env_config=config)] * 3
    specs.append(SimpleNamespace(initial_state=None, env_config=config))
    snapshot = evaluate._initial_snapshots(specs, jnp.zeros((4, 2), jnp.uint32))  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    assert calls == [(authored, config)]
    assert snapshot is not None
    state = np.asarray(snapshot[0])
    assert state.shape == (4, 3)
    np.testing.assert_array_equal(state[:3], np.tile(np.arange(3.0), (3, 1)))
    np.testing.assert_array_equal(state[3], np.full(3, 9.0))


def test_default_and_season_0_fields_compose() -> None:
    default = composed("output_dir=out")
    assert [entry.method for entry in default.systems] == [
        "tdm-beta",
        "tdm-gamma",
        "tdm-alpha",
        "random",
    ]
    assert list(default.scenario_ids) == list(range(1, 9))

    season = composed("+field=season_0", "models_dir=/models", "output_dir=out")
    names = [entry.name for entry in season.systems]
    assert names[0] == "Recurrent IPPO" and names[-4:] == [
        "TDM-Beta",
        "TDM-Gamma",
        "TDM-Alpha",
        "Random",
    ]
    assert len(names) == 10
    assert all(str(entry.method).startswith("/models/") for entry in season.systems[:6])


def test_tables_count_team_a_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {"a": np.array([1, 3, 2, 2]), "b": np.array([1, 1, 1, 3])}

    def load_results(run_dir: str) -> SimpleNamespace:
        episodes = {"outcome": outcomes[str(run_dir)]}

        def table(name: str) -> dict[str, np.ndarray]:
            return episodes

        return SimpleNamespace(table=table)

    monkeypatch.setattr(marl_bgs, "load_results", load_results)
    records = [
        {"system": "First", "scenario_runs": {"1": "a"}},
        {"system": "Second", "scenario_runs": {"1": "b"}},
    ]
    rows = scenarios.write_tables(records, [1], tmp_path)

    assert [(r["system"], r["wins"], r["draws"], r["losses"]) for r in rows] == [
        ("First", 1, 1, 2),
        ("Second", 3, 1, 0),
    ]
    table = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "out of 4 games per cell" in table
    assert "| First | 1 · 1 · 2 |" in table and "| Second | 3 · 1 · 0 |" in table
    assert (tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()[1] == (
        "First,1,4,1,1,2"
    )
    assert scenarios.folder_name("Recurrent IPPO") == "recurrent-ippo"


def test_fisher_exact_matches_known_values() -> None:
    assert round(scenario_analysis.fisher_exact(8, 256, 2, 256), 3) == 0.106
    assert round(scenario_analysis.fisher_exact(7, 256, 2, 256), 3) == 0.176
    assert scenario_analysis.fisher_exact(1, 256, 2, 256) == 1.0
    assert scenario_analysis.fisher_exact(0, 256, 0, 256) == 1.0
    with pytest.raises(ValueError, match="between 0"):
        scenario_analysis.fisher_exact(3, 2, 0, 2)


def test_analysis_counts_deaths_ultimates_and_lengths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Scenario 5 starts with Team A Mage, Warrior and Priest (slots 0, 1, 4).
    episodes = {
        "episode_id": np.array([1, 2, 3]),
        "outcome": np.array([2, 2, 3]),
        "episode_length": np.array([2, 2, 5]),
    }
    full: dict[str, np.ndarray] = {"episode_id": np.array([3, 1, 2])}
    for slot in range(10):
        full[f"agent_{slot}_deaths"] = np.zeros(3)
        full[f"agent_{slot}_ultimate_activations"] = np.zeros(3)
    full["agent_4_deaths"] = np.array([0, 1, 1])  # Priest dies in games 1 and 2.
    full["agent_0_ultimate_activations"] = np.array([1, 1, 0])  # Burst in 3 and 1.

    def load_results(run_dir: str) -> SimpleNamespace:
        tables = {"episodes": episodes, "full_metrics": full}

        def table(name: str) -> dict[str, np.ndarray]:
            return tables[name]

        return SimpleNamespace(table=table)

    monkeypatch.setattr(marl_bgs, "load_results", load_results)
    agents, games = scenario_analysis.read_games("First", 5, "run")

    priest = [r for r in agents if r["team"] == "A" and r["agent_class"] == "Priest"]
    assert [(r["outcome"], r["games"], r["died"]) for r in priest] == [
        ("draw", 1, 0),
        ("loss", 2, 2),
    ]
    assert (
        priest[1]["start_health"] == 1.0 and priest[1]["start_ultimate_cooldown"] == 2
    )
    mage = [r for r in agents if r["team"] == "A" and r["agent_class"] == "Mage"]
    assert [(r["outcome"], r["used_ultimate"]) for r in mage] == [
        ("draw", 1),
        ("loss", 1),
    ]
    assert [(r["outcome"], r["ticks"], r["games"]) for r in games] == [
        ("draw", 5, 1),
        ("loss", 2, 2),
    ]
    scenario_analysis.write_summary(agents, games, [], "Random", tmp_path / "a.md")
    summary = (tmp_path / "a.md").read_text(encoding="utf-8")
    assert "2 of 2 (100.0%) of those ended within two ticks" in summary
    assert "Priest (1 HP at the start) 2 of 2 (100.0%)" in summary
