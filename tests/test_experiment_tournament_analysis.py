"""Check the tournament_analysis experiment stage.

Contracts: the packaged settings give display names for Season 0's nine
entrants as the saved tournament records them and mark the three scripted bots,
with 128 parallel games and no earlier replay; compare_matches counts games
equal in every column but run_id, with missing values equal to missing values;
summary_columns lists every MEASURES catalog column and the 145 screened team
metrics once each, leaving the ratio measure to ratios= and the class-table
agent columns to the per-game call; team_profiles turns a system_metrics
summary into profiles under display names, with each measure's value and
interval, NaN kept, and keeps an entrant whose sampling is unknown with its
values, no resampled units and its interval status and reason; class_tables
builds the damage, kill-help, healing and damage-taken tables from per-game
rows, agents 0-4 as the entrant's team, each game's own classes, skipping
inactive agents and missing values; check_sides passes when every matchup's
mean score_difference over its per-game rows equals its saved head-to-head
margin and refuses a mismatch or a different set of matchups; elo_correlations
gives each measure's rank correlation with Elo, with tied values sharing their
average rank, and NaN when every value is equal or fewer than three rated
entrants have one; screen_measures takes each team metric's summary value per
entrant and ranks the metrics by their correlation with the given Elo, over all
entrants and without the scripted ones; write_outputs lists entrants from the
highest Elo down and writes teamwork.csv (with each entrant's interval status),
by_class.csv, elo_screen.csv, summary.md (with the correlations) and
teamwork.png; main reads the saved Elo, joins profiles and class tables,
records the replay's execution settings in analysis.json and refuses an entrant
without a saved rating. Small hand-made tables and stand-in results replace
saved runs; one small saved tournament runs the stage's own system_metrics
calls end to end, and one saved run with an unknown-sampling entrant runs
team_profiles.
"""

# pyright: reportPrivateUsage=false
import csv
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

import marl_battlegrounds
from marl_battlegrounds.evaluation.results import load_results
from marl_battlegrounds.experiments import tournament_analysis as stage


def test_settings_compose() -> None:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        config = compose(
            config_name="tournament_analysis",
            overrides=["tournament_dir=t", "field=f", "output_dir=o"],
        )
    assert len(config.names) == 9
    assert config.names["IPPO at step 250,675,200 (run)"] == "Recurrent IPPO"
    assert list(config.scripted) == ["TDM-Beta", "TDM-Gamma", "TDM-Alpha"]
    assert (config.num_envs, config.replay_dir) == (128, None)


def test_compare_matches() -> None:
    matches = {
        "run_id": np.array(["r1", "r1", "r1"]),
        "episode_id": np.array([1, 2, 3]),
        "team_a_score": np.array([20.0, np.nan, 5.0]),
    }
    replayed = dict(matches) | {
        "run_id": np.array(["r2", "r2", "r2"]),
        "team_a_score": np.array([20.0, np.nan, 6.0]),
    }
    assert stage.compare_matches(matches, replayed) == {
        "games": 3,
        "identical": 2,
        "different": [3],
    }


def test_summary_columns_cover_every_measure_once() -> None:
    columns = stage.summary_columns()
    assert len(columns) == len(set(columns))
    for _, column, _ in stage.MEASURES:
        assert (column in columns) == (column not in stage.RATIOS)
    assert len(stage.CLASS_COLUMNS) == 25 + 25 + 25 + 5
    assert not set(stage.CLASS_COLUMNS) & set(columns)
    assert len(stage.SCREEN_COLUMNS) == 145
    assert "team_a_win" not in columns and "team_a_score" not in columns


def summary(rows: int = 2) -> defaultdict[str, Any]:
    # Entrant "x-run" and entrant "y"; every column not set here is zero.
    table: defaultdict[str, Any] = defaultdict(lambda: np.zeros(rows))
    table["system_name"] = np.array(["x-run", "y"], dtype=object)
    table["games"] = np.array([8, 4])
    table["independent_units"] = np.array([4, None], dtype=object)
    table["interval_status"] = np.array(["Available", "Withheld"], dtype=object)
    table["unavailable_reason"] = np.array(
        [None, "Sampling mechanism unknown"], dtype=object
    )
    table["team_a_kills"] = np.array([7.0, 3.0])
    table["team_a_kills_ci_low"] = np.array([6.5, np.nan])
    table["team_a_kills_ci_high"] = np.array([7.5, np.nan])
    table["team_a_rescue_rate"] = np.array([np.nan, 0.5])
    return table


def test_team_profiles_read_measures_and_withheld_intervals() -> None:
    profiles = stage.team_profiles(summary(), {"x-run": "X"})
    assert list(profiles) == ["X", "y"]
    first, second = profiles["X"], profiles["y"]
    assert (first["games"], first["independent_units"]) == (8, 4)
    assert (first["interval_status"], first["unavailable_reason"]) == (
        "Available",
        None,
    )
    assert first["kills"] == {"value": 7.0, "low": 6.5, "high": 7.5}
    assert math.isnan(first["rescue_rate"]["value"])
    assert second["independent_units"] is None
    assert second["unavailable_reason"] == "Sampling mechanism unknown"
    assert second["kills"]["value"] == 3.0 and math.isnan(second["kills"]["low"])
    assert second["rescue_rate"]["value"] == 0.5


def test_team_profiles_keep_an_entrant_with_unknown_sampling(
    tmp_path: Path,
) -> None:
    from tests.test_system_metrics import _run

    path = _run(
        tmp_path,
        [
            ("s1", "s2", "stochastic"),
            ("s1", "s3", "unknown"),
            ("s3", "s2", "stochastic"),
        ],
    )
    table = load_results(path).system_metrics(
        "full_metrics", columns=stage.summary_columns(), ratios=stage.RATIOS
    )
    profiles = stage.team_profiles(table)
    assert profiles["One"]["independent_units"] is None
    assert "unknown" in str(profiles["One"]["unavailable_reason"])
    assert math.isfinite(profiles["One"]["kills"]["value"])
    assert profiles["Two"]["independent_units"] is not None


def games(rows: int = 3) -> defaultdict[str, Any]:
    # Entrant "x-run" plays two games, the second with its agent 3 inactive;
    # entrant "y" plays one game with Mage and Warrior swapped in both teams.
    table: defaultdict[str, Any] = defaultdict(lambda: np.zeros(rows))
    table["system_name"] = np.array(["x-run", "x-run", "y"], dtype=object)
    table["opponent_name"] = np.array(["y", "y", "x-run"], dtype=object)
    table["score_difference"] = np.array([3.0, -1.0, -1.0], dtype=np.float32)
    for agent in range(10):
        table[f"agent_{agent}_class_id"] = np.full(rows, agent % 5 + 1)
        table[f"agent_{agent}_active"] = np.ones(rows, dtype=int)
    table["agent_3_active"] = np.array([1, 0, 1])
    for agent in (0, 1, 5, 6):
        table[f"agent_{agent}_class_id"] = np.array(
            [agent % 5 + 1, agent % 5 + 1, 2 - agent % 5]
        )
    table["agent_0_to_agent_9_damage_done"] = np.array([30.0, 0.0, 40.0])
    table["agent_1_to_agent_5_damage_done"] = np.array([0.0, 10.0, 0.0])
    table["agent_4_to_agent_5_kill_contributions"] = np.array([1.0, 1.0, 0.0])
    table["agent_4_to_agent_1_effective_healing_done"] = np.array([75.0, np.nan, 0.0])
    table["agent_4_to_agent_0_effective_healing_done"] = np.array([25.0, 0.0, 0.0])
    table["agent_1_damage_received"] = np.array([30.0, 0.0, 0.0])
    table["agent_3_damage_received"] = np.array([10.0, 99.0, 0.0])
    return table


def test_class_tables_read_each_games_own_classes() -> None:
    tables = stage.class_tables(games(), {"x-run": "X"})
    mage, warrior, rogue, priest = 0, 1, 3, 4
    first = tables["X"]
    assert first["damage_share"][mage][priest] == 0.75
    assert first["damage_share"][warrior][mage] == 0.25
    assert first["kill_help"][priest][mage] == 1.0
    assert first["healing_share"] == [0.25, 0.75, 0.0, 0.0, 0.0]
    # The second game's 99 is skipped: its agent 3 was inactive.
    assert first["damage_taken_share"][warrior] == 0.75
    assert first["damage_taken_share"][rogue] == 0.25
    # In y's game agent 0 is a Warrior: its damage counts for that class.
    assert tables["y"]["damage_share"][warrior][priest] == 1.0


def test_check_sides_matches_head_to_head_margins() -> None:
    head_to_head = {
        "system_name": np.array(["x-run", "y"]),
        "opponent_name": np.array(["y", "x-run"]),
        "mean_point_margin": np.array([1.0, -1.0]),
    }
    assert stage.check_sides(games(), head_to_head) == 0.0
    swapped = games()
    swapped["score_difference"] = -swapped["score_difference"]
    with pytest.raises(ValueError, match=r"head_to_head by up to 2\.0"):
        stage.check_sides(swapped, head_to_head)
    other = {key: value[:1] for key, value in head_to_head.items()}
    with pytest.raises(ValueError, match="different matchups"):
        stage.check_sides(games(), other)


def test_elo_correlations_rank_measures_against_elo() -> None:
    def profile(margin: float, kills: float) -> dict[str, Any]:
        cells = {name: {"value": 1.0} for name, *_ in stage.MEASURES}
        return cells | {"margin": {"value": margin}, "kills": {"value": kills}}

    profiles = {"a": profile(1, 5), "b": profile(2, 5), "c": profile(3, 9)}
    elo = {"a": 1000.0, "b": 1100.0, "c": 1200.0}
    correlations = stage.elo_correlations(profiles, elo)
    assert correlations["margin"] == pytest.approx(1.0)
    # Tied kills share rank 1.5: corr([1.5, 1.5, 3], [1, 2, 3]).
    assert correlations["kills"] == pytest.approx(math.sqrt(3) / 2)
    assert math.isnan(correlations["deaths"])
    reversed_elo = {"a": 1200.0, "b": 1100.0, "c": 1000.0}
    assert stage.elo_correlations(profiles, reversed_elo)["margin"] == pytest.approx(
        -1.0
    )
    unrated = elo | {"c": math.nan}
    assert math.isnan(stage.elo_correlations(profiles, unrated)["margin"])


def test_screen_measures_ranks_team_metrics_by_elo() -> None:
    table: defaultdict[str, Any] = defaultdict(lambda: np.zeros(3))
    table["system_name"] = np.array(["x", "y", "z"], dtype=object)
    table["team_a_kills"] = np.array([6.0, 4.0, 2.0])
    table["team_a_multi_contributor_kill_fraction"] = np.array([0.7, np.nan, 0.5])
    elo = {"X": 1300.0, "y": 1200.0, "z": 1100.0}
    rows = stage.screen_measures(table, elo, {"x": "X"}, scripted=["z"])
    by_name = {row["measure"]: row for row in rows}
    assert len(rows) == len(stage.SCREEN_COLUMNS)
    assert rows[0]["measure"] == "kills"
    assert (rows[0]["X"], rows[0]["y"], rows[0]["z"]) == (6.0, 4.0, 2.0)
    assert rows[0]["correlation_all"] == pytest.approx(1.0)
    # Two learned entrants are too few for a correlation.
    assert math.isnan(rows[0]["correlation_learned"])
    shared = by_name["multi_contributor_kill_fraction"]
    assert shared["X"] == pytest.approx(0.7) and math.isnan(shared["y"])
    assert "Team A" not in shared["description"]
    assert "win" not in by_name and "score" not in by_name


def test_write_outputs_lists_entrants_by_elo(tmp_path: Path) -> None:
    profiles = stage.team_profiles(summary(), {"x-run": "High", "y": "Low"})
    for name, tables in stage.class_tables(
        games(), {"x-run": "High", "y": "Low"}
    ).items():
        profiles[name].update(tables)
    screen: list[dict[str, Any]] = [
        {"measure": "kills", "unit": "count", "description": "Kills"}
        | {"correlation_all": 1.0, "correlation_learned": float("nan")}
    ]
    stage.write_outputs(
        profiles, {"Low": 1100.0, "High": 1300.0}, tmp_path, screen, ["Low"]
    )
    with (tmp_path / "teamwork.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 * len(stage.MEASURES)
    assert [rows[0]["system"], rows[-1]["system"]] == ["High", "Low"]
    assert (rows[0]["games"], rows[0]["independent_units"]) == ("8", "4")
    assert (rows[0]["interval_status"], rows[-1]["interval_status"]) == (
        "Available",
        "Withheld",
    )
    assert rows[1]["measure"] == "kills" and float(rows[1]["value"]) == 7.0
    with (tmp_path / "by_class.csv").open(encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 2 * (25 + 25 + 5 + 5)
    with (tmp_path / "elo_screen.csv").open(encoding="utf-8") as file:
        assert next(csv.DictReader(file))["correlation_all"] == "1.0"
    summary_text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert summary_text.index("| High | Low |") < summary_text.index("| margin |")
    # Two entrants are too few for a rank correlation.
    assert "| margin | Points minus the opponent's, per game | +nan |" in summary_text
    assert "| kills | count | +1.00 | +nan | Kills |" in summary_text
    assert (tmp_path / "teamwork.png").stat().st_size > 0


def test_stage_calls_work_on_a_saved_tournament(tmp_path: Path) -> None:
    from tests.test_system_metrics import _canonical

    path, _, _ = _canonical(tmp_path)
    results = load_results(path)
    table = results.system_metrics(
        "full_metrics", columns=stage.summary_columns(), ratios=stage.RATIOS
    )
    profiles = stage.team_profiles(table)
    assert len(profiles) == 3
    assert all(profile["games"] == 8 for profile in profiles.values())
    per_game = results.system_metrics(
        "full_metrics", by="game", columns=["score_difference", *stage.CLASS_COLUMNS]
    )
    assert set(stage.class_tables(per_game)) == set(profiles)
    assert stage.check_sides(per_game, results.head_to_head()) == 0.0
    elo = dict.fromkeys(profiles, 1200.0)
    assert len(stage.screen_measures(table, elo)) == len(stage.SCREEN_COLUMNS)


class _Results:
    # Stands in for load_results: a saved tournament and its replay.
    def __init__(self, rated: tuple[str, ...]) -> None:
        self.rated = rated
        self.execution_settings = {
            "declared_xla_flags_by_attempt": [None],
            "declared_gpu_deterministic_ops": None,
        }

    def table(self, name: str) -> dict[str, Any]:
        if name == "matches":
            return {"run_id": np.array(["r"]), "episode_id": np.array([1])}
        assert name == "tournament_results"
        return {
            "policy": np.array(self.rated),
            "elo": np.array([1300.0, 1100.0][: len(self.rated)]),
        }

    def head_to_head(self) -> dict[str, Any]:
        return {
            "system_name": np.array(["x-run", "y"]),
            "opponent_name": np.array(["y", "x-run"]),
            "mean_point_margin": np.array([1.0, -1.0]),
        }

    def system_metrics(self, table: str, **options: object) -> dict[str, Any]:
        if options.get("by") == "game":
            return games()
        return summary()


def _loader(rated: tuple[str, ...]) -> Callable[[object], _Results]:
    def load(path: object) -> _Results:
        del path
        return _Results(rated)

    return load


def test_main_joins_saved_ratings_profiles_and_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = OmegaConf.create(
        {
            "tournament_dir": str(tmp_path / "saved"),
            "field": str(tmp_path / "field.json"),
            "names": {"x-run": "X"},
            "scripted": ["y"],
            "num_envs": 128,
            "replay_dir": str(tmp_path / "replay"),
            "output_dir": str(tmp_path / "out"),
        }
    )
    monkeypatch.setattr(marl_battlegrounds, "load_results", _loader(("x-run", "y")))
    record = stage.main(config)
    assert record["elo"] == {"X": 1300.0, "y": 1100.0}
    assert record["identity"]["identical"] == 1 and record["side_check"] == 0.0
    assert record["execution_settings"]["declared_gpu_deterministic_ops"] is None
    assert record["profiles"]["X"]["healing_share"] == [0.25, 0.75, 0.0, 0.0, 0.0]
    saved = json.loads((tmp_path / "out" / "analysis.json").read_text("utf-8"))
    assert saved["profiles"]["y"]["unavailable_reason"] == "Sampling mechanism unknown"
    monkeypatch.setattr(marl_battlegrounds, "load_results", _loader(("x-run",)))
    with pytest.raises(ValueError, match=r"no saved rating for \['y'\]"):
        stage.main(config)
