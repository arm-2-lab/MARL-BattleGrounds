"""Check the tournament_analysis experiment stage.

Contracts: the packaged settings give display names for Season 0's nine
entrants as the saved tournament records them and mark the three scripted bots,
with 128 parallel games and no earlier replay; compare_matches counts games
equal in every column but run_id, with missing values equal to missing values;
summary_columns lists every MEASURES catalog column and the 145 screened team
metrics once each, leaving the ratio measures to ratios= and the class-table
agent columns to the per-game call; team_profiles turns a system_metrics
summary into profiles under display names, with each measure's value and
interval, NaN kept, and keeps an entrant whose sampling is unknown with its
values and intervals, no resampled units, the status "Conditional: Sampling not
verified" and no reason; class_tables builds the damage, kill-help, healing and
damage-taken tables, the share of each ally class's ticks inside a Mage or
Warrior aura (covered over eligible, NaN with nothing to count), the Priest's
Freedom share by ally class and the class-pair distance weighted by observed
ticks from per-game rows, agents 0-4 as the entrant's team, each game's own
classes, skipping inactive agents and missing values; per_tick marks per-game
totals (not fractions, distances or averages) and summary_columns asks for
episode_length; screen_measures puts per-game totals per 100 ticks and treats a
rate constant up to float noise as having no trend; screen_tests adds each
screened metric's exact two-sided permutation p-value over all and over learned
entrants, Holm's correction and the Westfall-Young max-statistic correction
over the screen, the share of decided games the winner had more of it (ties and
missing left out) and, inside each matchup cell of one opponent, map and spawn
end, the chance a win shows more than a loss, NaN or 0.5 where nothing
separates; _permutation_p counts every ordering up to nine entrants and draws
seeded random orderings above that, counted with one added hit; _max_t gives
the max-statistic p-values; _holm adjusts finite p-values and keeps NaN;
check_sides passes when every matchup's mean score_difference over its per-game
rows equals its saved head-to-head margin and refuses a mismatch or a different
set of matchups; elo_correlations gives each measure's rank correlation with
Elo, with tied values sharing their average rank, and NaN when every value is
equal or fewer than three rated entrants have one; screen_measures takes each
team metric's summary value per entrant and ranks the metrics by their
correlation with the given Elo, over all entrants and without the scripted
ones; write_outputs lists entrants from the highest Elo down and writes
teamwork.csv (with each entrant's interval status), by_class.csv, trends.csv
(each measure's correlation with Elo, its exact p-value and its entrant count),
elo_screen.csv, summary.md (with the correlations) and teamwork.png; main reads
the saved Elo, joins profiles and class tables, records the replay's execution
settings in analysis.json and refuses an entrant without a saved rating. Small
hand-made tables and stand-in results replace saved runs; one small saved
tournament runs the stage's own system_metrics calls end to end, and one saved
run with an unknown-sampling entrant runs team_profiles.
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
            overrides=["tournament_dir=t", "field_file=f", "output_dir=o"],
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
    # Damage, kill help, healing, damage taken; then two auras' covered and
    # eligible ticks, Freedom uses, and each class pair's distance and count.
    assert len(stage.CLASS_COLUMNS) == 25 + 25 + 25 + 5 + 2 * 2 * 20 + 25 + 10 * 2
    assert not set(stage.CLASS_COLUMNS) & set(columns)
    assert len(stage.SCREEN_COLUMNS) == 145
    assert "team_a_win" not in columns and "team_a_score" not in columns


def test_per_tick_covers_per_game_totals_only() -> None:
    assert stage.per_tick("team_a_kills")
    assert stage.per_tick("team_a_damage_from_mage_aura")
    assert not stage.per_tick("team_a_multi_contributor_kill_fraction")
    assert not stage.per_tick("team_a_ally_distance_mean")
    assert not stage.per_tick("team_a_mean_observed_respawn_wait_steps")
    assert "episode_length" in stage.summary_columns()


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
    table["episode_length"] = np.full(rows, 200.0)
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
    from tests.evaluation.metrics.test_system_metrics import _run

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
    assert profiles["One"]["interval_status"] == "Conditional: Sampling not verified"
    assert profiles["One"]["unavailable_reason"] is None
    assert math.isfinite(profiles["One"]["kills"]["value"])
    assert math.isfinite(profiles["One"]["kills"]["low"])
    assert math.isfinite(profiles["One"]["kills"]["high"])
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
    table["episode_length"] = np.full(rows, 200.0)
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


def test_class_tables_read_auras_freedom_and_class_distances() -> None:
    table = games()
    # x-run's games: Mage agent 0 covers Warrior agent 1 for 30 of 40 ticks,
    # then 10 of 10; the Priest gives Freedom to itself once and the Mage 3
    # times; Mage and Priest stand 2 apart for 10 ticks, then 5 apart for 30.
    table["agent_0_to_agent_1_mage_aura_covered_steps"] = np.array([30.0, 10.0, 0])
    table["agent_0_to_agent_1_mage_aura_eligible_steps"] = np.array([40.0, 10.0, 0])
    table["agent_4_to_agent_4_priest_freedom_applications"] = np.array([1.0, 0, 0])
    table["agent_4_to_agent_0_priest_freedom_applications"] = np.array([2.0, 1.0, 0])
    table["agent_0_and_agent_4_ally_distance_mean"] = np.array([2.0, 5.0, np.nan])
    table["agent_0_and_agent_4_ally_distance_observations"] = np.array([10.0, 30, 0])
    tables = stage.class_tables(table, {"x-run": "X"})
    mage, warrior, priest = 0, 1, 4
    first = tables["X"]
    assert first["mage_aura_coverage_by_class"][warrior] == 0.8
    assert math.isnan(first["mage_aura_coverage_by_class"][mage])
    assert all(math.isnan(value) for value in first["warrior_aura_coverage_by_class"])
    assert first["freedom_share"] == [0.75, 0.0, 0.0, 0.0, 0.25]
    distance = first["class_pair_distance"]
    assert distance[mage][priest] == distance[priest][mage] == 4.25
    assert math.isnan(distance[mage][warrior])
    # In y's game agent 0 is a Warrior: its aura columns would count as a
    # Warrior's, and its distance to the Priest is missing.
    assert math.isnan(tables["y"]["class_pair_distance"][warrior][priest])


def test_screen_tests_give_exact_p_values_and_game_checks() -> None:
    def row(measure: str, values: list[float]) -> dict[str, Any]:
        named = dict(zip("ABCD", values, strict=True))
        return {"measure": measure, "unit": "count", "description": measure} | named

    screen = [
        row("kills", [4.0, 3.0, 2.0, 1.0]),
        row("multi_contributor_kill_fraction", [0.5, 0.5, 0.5, 0.5]),
    ]
    elo = {"A": 1400.0, "B": 1300.0, "C": 1200.0, "D": 1100.0}
    # Fifty games of a-run against B: a-run wins the first 25, B the rest.
    count = 50
    table: dict[str, Any] = {
        "system_name": np.array(["a-run", "B"] * count, dtype=object),
        "opponent_name": np.array(["B", "a-run"] * count, dtype=object),
        "map_id": np.full(2 * count, 47),
        "spawn_end": np.array(["team_a", "team_b"] * count, dtype=object),
        "episode_length": np.full(2 * count, 100.0),
        "run_id": np.array(["r"] * 2 * count),
        "phase": np.array(["p"] * 2 * count),
        "pass_id": np.array(["0"] * 2 * count),
        "episode_id": np.repeat(np.arange(count), 2),
    }
    first_wins = np.arange(count) < 25
    table["team_a_win"] = np.stack([first_wins, ~first_wins], axis=1).ravel() * 1.0
    table["team_a_loss"] = 1.0 - table["team_a_win"]
    table["team_a_kills"] = np.where(table["team_a_win"] == 1, 5.0, 1.0)
    table["team_a_multi_contributor_kill_fraction"] = np.full(2 * count, 0.5)
    rows = stage.screen_tests(screen, table, elo, {"a-run": "A"}, scripted=["D"])
    kills, shared = rows
    # A perfect order of four: 2 of the 24 orderings are as extreme.
    assert kills["p_value_all"] == pytest.approx(2 / 24)
    assert kills["p_value_learned"] == pytest.approx(2 / 6)
    assert kills["holm_p_value_all"] == pytest.approx(2 / 24)
    # The only metric with a value sets the maximum itself.
    assert kills["max_t_p_value_all"] == pytest.approx(2 / 24)
    assert kills["winner_higher_share"] == 1.0
    assert kills["decided_games_compared"] == count
    assert kills["wins_over_losses"] == 1.0
    assert math.isnan(shared["p_value_all"]) and math.isnan(shared["holm_p_value_all"])
    assert math.isnan(shared["max_t_p_value_all"])
    assert math.isnan(shared["winner_higher_share"])
    assert shared["decided_games_compared"] == 0
    assert shared["wins_over_losses"] == 0.5
    assert screen[0].keys() < kills.keys()
    # Inside one matchup cell, a's wins come only against B: holding the
    # opponent fixed, a measure higher in every loss gives 0.
    flipped = dict(table)
    flipped["team_a_kills"] = np.where(table["team_a_win"] == 1, 1.0, 5.0)
    (reversed_kills, _) = stage.screen_tests(screen, flipped, elo, {"a-run": "A"})
    assert reversed_kills["wins_over_losses"] == 0.0


def test_permutation_p_values_and_holm() -> None:
    assert stage._permutation_p([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(
        2 / 6
    )
    assert math.isnan(stage._permutation_p([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(stage._permutation_p([1.0, math.nan, 2.0], [1.0, 2.0, 3.0]))
    ten = [float(value) for value in range(stage.EXACT_ENTRANTS + 1)]
    # Ten entrants use random orderings, counted with one added hit and draw.
    assert stage._orderings(len(ten)).shape == (stage.PERMUTATION_DRAWS, len(ten))
    assert 0 < stage._permutation_p(ten, ten) < 1e-3
    # The maximum over two metrics reaches 1 when either one is in perfect
    # order: 2 orderings each, 4 of the 24.
    pair = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 3.0]]
    adjusted = stage._max_t(pair, [1, 2, 3, 4])
    assert adjusted[0] == pytest.approx(4 / 24) and adjusted[0] <= adjusted[1]
    adjusted = stage._holm([0.01, 0.04, 0.03, math.nan])
    assert adjusted[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert math.isnan(adjusted[3])


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
    # Float noise on a truly constant rate must not become a trend.
    table["team_a_action_acceptance_rate"] = 1.0 - np.array([1e-9, 3e-9, 2e-9])
    table["episode_length"] = np.array([200.0, 100.0, 400.0])
    elo = {"X": 1300.0, "y": 1200.0, "z": 1100.0}
    rows = stage.screen_measures(table, elo, {"x": "X"}, scripted=["z"])
    by_name = {row["measure"]: row for row in rows}
    assert len(rows) == len(stage.SCREEN_COLUMNS)
    assert rows[0]["measure"] == "kills" and rows[0]["scale"] == "per 100 ticks"
    # Kills are per-game totals: 6 in 200 ticks is 3 per 100 ticks.
    assert (rows[0]["X"], rows[0]["y"], rows[0]["z"]) == (3.0, 4.0, 0.5)
    assert math.isnan(by_name["action_acceptance_rate"]["correlation_all"])
    assert by_name["multi_contributor_kill_fraction"]["scale"] == "as summarized"
    assert rows[0]["correlation_all"] == pytest.approx(0.5)
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
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 * (3 * 25 + 5 * len(stage.VECTOR_TABLES))
    with (tmp_path / "elo_screen.csv").open(encoding="utf-8") as file:
        assert next(csv.DictReader(file))["correlation_all"] == "1.0"
    with (tmp_path / "trends.csv").open(encoding="utf-8") as file:
        trends = {row["measure"]: row for row in csv.DictReader(file)}
    assert set(trends) == {measure for measure, _, _ in stage.MEASURES}
    # Two entrants are too few for a correlation or its p-value.
    assert (trends["margin"]["correlation"], trends["margin"]["p_value"]) == (
        "nan",
        "nan",
    )
    assert trends["margin"]["entrants"] == "2"
    summary_text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert summary_text.index("| High | Low |") < summary_text.index("| margin |")
    # Two entrants are too few for a rank correlation.
    assert "| margin | Points minus the opponent's, per game | +nan |" in summary_text
    assert "| kills | count | +1.00 | +nan | Kills |" in summary_text
    assert (tmp_path / "teamwork.png").stat().st_size > 0


def test_stage_calls_work_on_a_saved_tournament(tmp_path: Path) -> None:
    from tests.evaluation.metrics.test_system_metrics import _canonical

    path, _, _ = _canonical(tmp_path)
    results = load_results(path)
    table = results.system_metrics(
        "full_metrics", columns=stage.summary_columns(), ratios=stage.RATIOS
    )
    profiles = stage.team_profiles(table)
    assert len(profiles) == 3
    assert all(profile["games"] == 8 for profile in profiles.values())
    columns = ["score_difference", "team_a_win", "team_a_loss", "episode_length"]
    per_game = results.system_metrics(
        "full_metrics",
        by="game",
        columns=[*columns, *stage.SCREEN_COLUMNS, *stage.CLASS_COLUMNS],
    )
    assert set(stage.class_tables(per_game)) == set(profiles)
    assert stage.check_sides(per_game, results.head_to_head()) == 0.0
    elo = dict(zip(profiles, (1300.0, 1200.0, 1100.0), strict=True))
    screen = stage.screen_measures(table, elo)
    assert len(screen) == len(stage.SCREEN_COLUMNS)
    tested = stage.screen_tests(screen, per_game, elo)
    assert all("max_t_p_value_all" in row for row in tested)
    assert all(0 <= row["decided_games_compared"] <= 12 for row in tested)


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
            "field_file": str(tmp_path / "field.json"),
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
