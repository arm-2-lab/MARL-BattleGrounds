"""Check the tournament_analysis experiment stage.

Contracts: the packaged settings give display names for Season 0's nine
entrants as the saved tournament records them and mark the three scripted
bots, with 128 parallel games and no earlier replay; entrant_games finds every
game an entrant played, its side and its pair (none against an opponent whose
games repeat exactly), and refuses entrants that never played or played
themselves; compare_matches counts games equal in every column but run_id,
with missing values equal to missing values; team_profile reads Team A or
Team B columns by the side played, combines games as per-game averages,
totals' ratios, weighted means and point differences, holds exactly repeating
games fixed in its intervals, gives NaN when nothing counts, takes classes
from each game's class columns, skips inactive agents and builds the damage,
kill-help, healing and damage-taken class tables; check_sides compares each
margin with the game-weighted head-to-head margin under display names and
refuses a mismatch; elo_correlations gives each measure's rank correlation
with Elo, with tied values sharing their average rank, and NaN when every
value is equal or fewer than three entrants have one; screen_measures
averages every team-level catalog metric except the game result per entrant,
fractions only over games where they exist, and ranks the metrics by their
correlation with Elo, over all entrants and without the scripted ones;
write_outputs lists entrants from the highest Elo down and writes
teamwork.csv, by_class.csv, elo_screen.csv, summary.md (with the
correlations) and teamwork.png. Small hand-made tables replace saved runs.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from marl_battlegrounds.experiments import tournament_analysis


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


def test_entrant_games_and_compare_matches() -> None:
    matches = {
        "run_id": np.array(["r1", "r1", "r1"]),
        "episode_id": np.array([1, 2, 3]),
        "team_a_policy": np.array(["x", "y", "y"]),
        "team_b_policy": np.array(["y", "x", "z"]),
        "block_id": np.array(["p1", "p1", "p2"]),
        "team_a_score": np.array([20.0, np.nan, 5.0]),
    }
    assert tournament_analysis.entrant_games(matches, "x") == {
        1: ("a", "p1"),
        2: ("b", "p1"),
    }
    assert tournament_analysis.entrant_games(matches, "y", ["z"])[3] == ("a", None)
    with pytest.raises(ValueError, match="no game"):
        tournament_analysis.entrant_games(matches, "w")
    mirror = matches | {"team_b_policy": np.array(["x", "x", "z"])}
    with pytest.raises(ValueError, match="itself"):
        tournament_analysis.entrant_games(mirror, "x")

    replayed = dict(matches) | {
        "run_id": np.array(["r2", "r2", "r2"]),
        "team_a_score": np.array([20.0, np.nan, 6.0]),
    }
    assert tournament_analysis.compare_matches(matches, replayed) == {
        "games": 3,
        "identical": 2,
        "different": [3],
    }


def full_metrics() -> defaultdict[str, Any]:
    # Games 7 and 8 are one pair; the entrant is Team A in 7 and Team B in 8.
    # Game 9 is not the entrant's and holds large values that must be ignored.
    full: defaultdict[str, Any] = defaultdict(lambda: np.zeros(3))
    full["episode_id"] = np.array([7, 8, 9])
    for agent in range(10):
        full[f"agent_{agent}_class_id"] = np.full(3, agent % 5 + 1)
        full[f"agent_{agent}_active"] = np.ones(3)
    full["agent_3_active"] = np.array([0.0, 1.0, 1.0])
    full["agent_3_damage_received"] = np.array([1000.0, 0.0, 0.0])
    values = {
        "score": ([20.0, 3.0, 0.0], [5.0, 20.0, 0.0]),
        "kills": ([10.0, 0.0, 99.0], [0.0, 4.0, 99.0]),
        "multi_contributor_kills": ([5.0, 0.0, 99.0], [0.0, 1.0, 0.0]),
        "focus_fire_steps": ([6.0, 0.0, 0.0], [0.0, 2.0, 50.0]),
        "ally_distance_mean": ([2.0, 0.0, 0.0], [0.0, 4.0, np.nan]),
        "ally_distance_observations": ([10.0, 0.0, 0.0], [0.0, 30.0, 50.0]),
    }
    for stem, (team_a, team_b) in values.items():
        full[f"team_a_{stem}"] = np.array(team_a)
        full[f"team_b_{stem}"] = np.array(team_b)
    full["agent_0_to_agent_9_damage_done"] = np.array([30.0, 0.0, 500.0])
    full["agent_6_to_agent_0_damage_done"] = np.array([0.0, 10.0, 0.0])
    full["agent_4_to_agent_5_kill_contributions"] = np.array([2.0, 0.0, 0.0])
    full["agent_4_to_agent_1_effective_healing_done"] = np.array([75.0, 0.0, 0.0])
    full["agent_9_to_agent_6_effective_healing_done"] = np.array([0.0, 25.0, 0.0])
    full["agent_9_to_agent_5_effective_healing_done"] = np.array([0.0, 100.0, 0.0])
    full["agent_1_damage_received"] = np.array([30.0, 0.0, 0.0])
    full["agent_8_damage_received"] = np.array([0.0, 10.0, 0.0])
    return full


def test_team_profile_reads_the_side_played() -> None:
    full = full_metrics()
    profile = tournament_analysis.team_profile(full, {7: ("a", "p1"), 8: ("b", "p1")})
    assert (profile["games"], profile["pairs"], profile["fixed_games"]) == (2, 1, 0)
    assert profile["margin"]["value"] == 16.0
    assert profile["kills"]["value"] == 7.0
    assert profile["shared_kill_fraction"]["value"] == pytest.approx(6 / 14)
    # Weighted by observations: (2 x 10 + 4 x 30) / 40.
    assert profile["ally_distance"]["value"] == pytest.approx(3.5)
    assert profile["focus_fire_ticks"]["value"] == 4.0
    # One pair: every resample is the same, so the interval is the value.
    assert profile["kills"]["low"] == profile["kills"]["high"] == 7.0
    assert math.isnan(profile["rescue_rate"]["value"])

    mage, warrior, rogue, priest = 0, 1, 3, 4
    assert profile["damage_share"][mage][priest] == 0.75
    assert profile["damage_share"][warrior][mage] == 0.25
    assert profile["kill_help"][priest][mage] == 1.0
    assert profile["healing_share"] == [0.5, 0.5, 0.0, 0.0, 0.0]
    assert profile["damage_taken_share"][warrior] == 0.75
    assert profile["damage_taken_share"][rogue] == 0.25
    # Games that repeat exactly count but are never resampled.
    fixed = tournament_analysis.team_profile(full, {7: ("a", None), 8: ("b", None)})
    assert (fixed["pairs"], fixed["fixed_games"]) == (0, 2)
    assert fixed["kills"] == {"value": 7.0, "low": 7.0, "high": 7.0}
    with pytest.raises(ValueError, match="no full metrics"):
        tournament_analysis.team_profile(full, {1: ("a", "p9")})


def test_check_sides_matches_head_to_head_margins() -> None:
    profiles = {"High": {"margin": {"value": 2.0}}, "low": {"margin": {"value": -2.0}}}
    head_to_head = {
        "system_name": np.array(["high-run", "high-run", "low"]),
        "games": np.array([100, 300, 400]),
        "mean_point_margin": np.array([5.0, 1.0, -2.0]),
    }
    names = {"high-run": "High"}
    assert tournament_analysis.check_sides(profiles, head_to_head, names) == 0.0
    swapped = profiles | {"High": {"margin": {"value": -2.0}}}
    with pytest.raises(ValueError, match="head_to_head"):
        tournament_analysis.check_sides(swapped, head_to_head, names)


def test_elo_correlations_rank_measures_against_elo() -> None:
    def profile(margin: float, kills: float) -> dict[str, Any]:
        cells = {name: {"value": 1.0} for name, *_ in tournament_analysis.MEASURES}
        return cells | {"margin": {"value": margin}, "kills": {"value": kills}}

    profiles = {"a": profile(1, 5), "b": profile(2, 5), "c": profile(3, 9)}
    elo = {"a": 1000.0, "b": 1100.0, "c": 1200.0}
    correlations = tournament_analysis.elo_correlations(profiles, elo)
    assert correlations["margin"] == pytest.approx(1.0)
    # Tied kills share rank 1.5: corr([1.5, 1.5, 3], [1, 2, 3]).
    assert correlations["kills"] == pytest.approx(math.sqrt(3) / 2)
    assert math.isnan(correlations["deaths"])
    reversed_elo = {"a": 1200.0, "b": 1100.0, "c": 1000.0}
    assert tournament_analysis.elo_correlations(profiles, reversed_elo)[
        "margin"
    ] == pytest.approx(-1.0)


def test_screen_measures_ranks_team_metrics_by_elo() -> None:
    # Three entrants each play two of three games, once on each team.
    full: defaultdict[str, Any] = defaultdict(lambda: np.zeros(3))
    full["episode_id"] = np.array([1, 2, 3])
    full["team_a_kills"] = np.array([6.0, 4.0, 2.0])
    full["team_b_kills"] = np.array([4.0, 2.0, 6.0])
    full["team_a_multi_contributor_kill_fraction"] = np.array([0.5, np.nan, 0.1])
    full["team_b_multi_contributor_kill_fraction"] = np.array([np.nan, 0.2, 0.9])
    games = {
        "x": {1: ("a", "p1"), 3: ("b", "p3")},
        "y": {1: ("b", "p1"), 2: ("a", "p2")},
        "z": {2: ("b", "p2"), 3: ("a", "p3")},
    }
    elo = {"x": 1300.0, "y": 1200.0, "z": 1100.0}
    rows = tournament_analysis.screen_measures(full, games, elo, scripted=["z"])
    by_name = {row["measure"]: row for row in rows}
    assert rows[0]["measure"] == "kills"
    assert (rows[0]["x"], rows[0]["y"], rows[0]["z"]) == (6.0, 4.0, 2.0)
    assert rows[0]["correlation_all"] == pytest.approx(1.0)
    # Two learned entrants are too few for a correlation.
    assert math.isnan(rows[0]["correlation_learned"])
    shared = by_name["multi_contributor_kill_fraction"]
    assert shared["x"] == pytest.approx(0.7) and math.isnan(shared["y"])
    assert "Team A" not in shared["description"]
    assert "win" not in by_name and "score" not in by_name


def test_write_outputs_lists_entrants_by_elo(tmp_path: Path) -> None:
    full = full_metrics()
    first = tournament_analysis.team_profile(full, {7: ("a", "p1"), 8: ("b", "p1")})
    second = tournament_analysis.team_profile(full, {7: ("b", "p1"), 8: ("a", "p1")})
    profiles = {"Low": second, "High": first}
    screen: list[dict[str, Any]] = [
        {"measure": "kills", "unit": "count", "description": "Kills"}
        | {"correlation_all": 1.0, "correlation_learned": float("nan")}
    ]
    tournament_analysis.write_outputs(
        profiles, {"Low": 1100.0, "High": 1300.0}, tmp_path, screen, ["Low"]
    )
    with (tmp_path / "teamwork.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 * len(tournament_analysis.MEASURES)
    assert [rows[0]["system"], rows[-1]["system"]] == ["High", "Low"]
    assert rows[0]["measure"] == "margin" and float(rows[0]["value"]) == 16.0
    with (tmp_path / "by_class.csv").open(encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 2 * (25 + 25 + 5 + 5)
    with (tmp_path / "elo_screen.csv").open(encoding="utf-8") as file:
        assert next(csv.DictReader(file))["correlation_all"] == "1.0"
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert summary.index("| High | Low |") < summary.index("| margin |")
    # Two entrants are too few for a rank correlation.
    assert "| margin | Points minus the opponent's, per game | +nan |" in summary
    assert "| kills | count | +1.00 | +nan | Kills |" in summary
    assert (tmp_path / "teamwork.png").stat().st_size > 0
