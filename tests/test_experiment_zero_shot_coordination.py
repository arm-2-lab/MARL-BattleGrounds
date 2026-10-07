"""Check the zero_shot_coordination stage and its analysis on small stand-ins.

Contracts: the packaged settings compose to recurrent IPPO hosting the other
five learned Season 0 methods and Random, in all five classes, against the nine
members on maps 47 to 51 with 200 full-metric games from seed 20261007, and the
analysis to 2,000 draws from seed 20261007; folder_name keeps lowercase
letters, digits and single hyphens; team_plan lists the host alone, then every
newcomer in every class, and refuses no newcomer, repeats, the host as a
newcomer and an unknown class; play_team plays each opponent once, records its
run and time in runs.json after each one and skips opponents already recorded.
The analysis: game_measures reads margin, the newcomer's kill help, deaths and
damage and its tick-weighted distance to teammates (NaN with no shared ticks);
game_measures also gives the newcomer's aura coverage of its teammates (NaN
without an aura or ticks), the Priest's useful healing to it (NaN for the
Priest) and the four kept agents' tick-weighted distance; paired_interval
resamples whole units with a fixed seed, within strata when given, and gives
zero width when every unit (or stratum) has the same mean; compare matches
games by opponent and episode ID, refuses a missing game, a matched game on
another map and a slot of another class and reports each measure's change;
slot_value_kept gives 1 to the host, 0 to the Random floor and the share in
between, NaN without a floor, and kept_interval gives the same value with an
interval from shared draws (0 for the floor itself); fit_drivers gives each
measure's rank correlation with the margin change within each class and an
exact two-sided permutation p-value of the mean, NaN for a constant measure;
map_end_rows groups games by opponent, map and spawn end, an unsaved end
labeled "unknown", with each class's mean distance to its teammates; play_team
also records each opponent's seconds. Stand-ins replace the evaluator and saved
runs, so no game is played.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_module

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.experiments import zero_shot_coordination as stage
from marl_battlegrounds.experiments import zero_shot_coordination_analysis as analysis


def test_settings_compose() -> None:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        run = compose(
            config_name="zero_shot_coordination",
            overrides=["output_dir=o", "models_dir=m"],
        )
        study = compose(
            config_name="zero_shot_coordination_analysis",
            overrides=["zsc_dir=z", "output_dir=o"],
        )
    assert run.host == "Recurrent IPPO"
    assert list(run.newcomers)[-1] == "Random" and len(run.newcomers) == 6
    assert list(run.classes) == list(stage.CLASSES) and len(run.opponents) == 9
    assert (list(run.maps), run.num_episodes, run.seed) == (
        [47, 48, 49, 50, 51],
        200,
        20261007,
    )
    assert (study.bootstrap_draws, study.bootstrap_seed) == (2000, 20261007)


def test_folder_names_and_team_plan() -> None:
    assert stage.folder_name("Recurrent IPPO with a QMIX Priest") == (
        "recurrent-ippo-with-a-qmix-priest"
    )
    plan = stage.team_plan("Host", ["A", "B"], ["Mage", "Priest"])
    assert plan[0] == {"name": "Host", "newcomer": None, "newcomer_class": None}
    assert [team["name"] for team in plan[1:]] == [
        "Host with a A Mage",
        "Host with a A Priest",
        "Host with a B Mage",
        "Host with a B Priest",
    ]
    for newcomers, classes, message in (
        ([], ["Mage"], "at least one"),
        (["A", "A"], ["Mage"], "must not repeat"),
        (["Host"], ["Mage"], "cannot also be"),
        (["A"], ["Bard"], "unknown classes"),
    ):
        with pytest.raises(ValueError, match=message):
            stage.team_plan("Host", newcomers, classes)


def test_play_team_records_each_opponent_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    played: list[str] = []

    def evaluate(team: object, opponent: str, **options: object) -> SimpleNamespace:
        played.append(opponent)
        return SimpleNamespace(run_dir=Path(str(options["output_dir"])) / "run")

    monkeypatch.setattr(marl_bgs, "evaluate", evaluate)
    settings: dict[str, Any] = {
        "maps": [47],
        "num_episodes": 2,
        "seed": 1,
        "num_envs": 2,
        "metrics": "none",
    }
    team: Any = "team"
    opponents: dict[str, Any] = {"First One": "a", "Second": "b"}
    record = stage.play_team(team, opponents, team_dir=tmp_path, **settings)
    assert played == ["a", "b"]
    assert record["runs"]["First One"] == str(tmp_path / "first-one" / "run")
    assert json.loads((tmp_path / "runs.json").read_text())["runs"] == record["runs"]
    assert set(record["seconds"]) == {"First One", "Second"}
    more: dict[str, Any] = opponents | {"Third": "c"}
    stage.play_team(team, more, team_dir=tmp_path, **settings)
    assert played == ["a", "b", "c"]


def _games(margins: list[float], slot_distance: list[float]) -> dict[str, Any]:
    count = len(margins)
    table: dict[str, Any] = {name: np.zeros(count) for name in analysis.read_columns()}
    table["score_difference"] = np.asarray(margins)
    table["agent_0_kill_contributions"] = np.arange(count, dtype=float)
    for other in range(1, 5):
        table[f"agent_0_and_agent_{other}_ally_distance_mean"] = np.asarray(
            slot_distance
        )
        table[f"agent_0_and_agent_{other}_ally_distance_observations"] = np.full(
            count, 10.0
        )
    for slot in range(5):
        table[f"agent_{slot}_class_id"] = np.full(count, slot + 1)
    table["opponent"] = np.asarray(["X"] * count, dtype=object)
    table["episode_id"] = np.arange(count)
    table["seed_id"] = np.arange(count) // 2
    table["map_id"] = np.full(count, 47)
    table["spawn_end"] = np.asarray(["team_a", "team_b"] * (count // 2), dtype=object)
    table["team_a_win"] = (np.asarray(margins) > 0).astype(float)
    table["team_a_loss"] = (np.asarray(margins) < 0).astype(float)
    table["team_a_draw"] = (np.asarray(margins) == 0).astype(float)
    return table


def test_game_measures_and_paired_interval() -> None:
    games = _games([3.0, -1.0], [2.0, 4.0])
    games["agent_0_and_agent_1_ally_distance_observations"] = np.array([10.0, 0.0])
    games["agent_0_to_agent_2_mage_aura_covered_steps"] = np.array([3.0, 0.0])
    games["agent_0_to_agent_2_mage_aura_eligible_steps"] = np.array([4.0, 0.0])
    games["agent_4_to_agent_0_effective_healing_done"] = np.array([5.0, 6.0])
    for first, second in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)):
        stem = f"agent_{first}_and_agent_{second}_ally_distance"
        games[f"{stem}_mean"] = np.array([1.0, 3.0])
        games[f"{stem}_observations"] = np.array([5.0, 5.0])
    measures = analysis.game_measures(games, 0)
    assert measures["newcomer_aura_coverage"][0] == 0.75
    assert np.isnan(measures["newcomer_aura_coverage"][1])
    assert measures["newcomer_healing_received"].tolist() == [5.0, 6.0]
    assert measures["kept_distance"].tolist() == [1.0, 3.0]
    priest = analysis.game_measures(games, 4)
    assert np.isnan(priest["newcomer_healing_received"]).all()
    assert np.isnan(priest["newcomer_aura_coverage"]).all()
    assert measures["margin"].tolist() == [3.0, -1.0]
    assert measures["newcomer_kill_contributions"].tolist() == [0.0, 1.0]
    assert measures["newcomer_distance"].tolist() == [2.0, 4.0]
    games["agent_0_and_agent_1_ally_distance_observations"] = np.zeros(2)
    for other in range(2, 5):
        games[f"agent_0_and_agent_{other}_ally_distance_observations"] = np.zeros(2)
    assert np.isnan(analysis.game_measures(games, 0)["newcomer_distance"]).all()
    values = np.array([1.0, 3.0, 2.0, 2.0])
    units = np.array(["a", "a", "b", "b"])
    mean, low, high = analysis.paired_interval(values, units, draws=50, seed=3)
    assert (mean, low, high) == (2.0, 2.0, 2.0)
    # Within strata, each opponent keeps its one unit, so nothing varies.
    spread = np.array([0.0, 0.0, 4.0, 4.0])
    strata = np.array(["X", "X", "Y", "Y"])
    assert analysis.paired_interval(spread, units, draws=50, seed=3, strata=strata) == (
        2.0,
        2.0,
        2.0,
    )
    pooled = analysis.paired_interval(spread, units, draws=50, seed=3)
    assert pooled[1] < 2.0 < pooled[2]
    again = analysis.paired_interval(
        np.array([0.0, 4.0, 1.0, 1.0]), units, draws=50, seed=3
    )
    assert again == analysis.paired_interval(
        np.array([0.0, 4.0, 1.0, 1.0]), units, draws=50, seed=3
    )
    assert all(math.isnan(x) for x in analysis.paired_interval(
        np.array([np.nan]), np.array(["a"]), draws=5, seed=1
    ))  # fmt: skip


def test_compare_and_slot_value_kept() -> None:
    host = _games([4.0, 2.0, 6.0, 4.0], [2.0, 2.0, 2.0, 2.0])
    team = _games([1.0, -1.0, 3.0, 1.0], [3.0, 3.0, 3.0, 3.0])
    row = analysis.compare(team, host, "Mage", draws=20, seed=1)
    assert row["games"] == 4 and (row["wins"], row["losses"]) == (3, 1)
    assert row["margin_change"] == -3.0
    assert row["margin_change_low"] == row["margin_change_high"] == -3.0
    assert row["newcomer_distance_change"] == 1.0
    with pytest.raises(ValueError, match="is not the Warrior"):
        bad = dict(team) | {"agent_1_class_id": np.full(4, 9)}
        analysis.compare(bad, host, "Warrior", draws=20, seed=1)
    with pytest.raises(ValueError, match="differ in map_id"):
        analysis.compare(
            team | {"map_id": np.full(4, 48)}, host, "Mage", draws=20, seed=1
        )
    with pytest.raises(ValueError, match="no match"):
        analysis.compare(
            team | {"episode_id": np.arange(4) + 9}, host, "Mage", draws=20, seed=1
        )
    rows = [
        {"newcomer": "Host", "newcomer_class": "Mage", "margin_change": 0.0},
        {"newcomer": "A", "newcomer_class": "Mage", "margin_change": -2.0},
        {"newcomer": "Random", "newcomer_class": "Mage", "margin_change": -8.0},
        {"newcomer": "A", "newcomer_class": "Priest", "margin_change": -1.0},
    ]
    kept = analysis.slot_value_kept(rows, "Host")
    assert kept[:3] == [1.0, 0.75, 0.0] and math.isnan(kept[3])
    units = np.array(["a", "a", "b", "b"])
    strata = np.array(["X"] * 4)
    floor = np.array([-8.0, -8.0, -4.0, -4.0])
    point, low, high = analysis.kept_interval(
        floor / 4, floor, units, strata, draws=50, seed=2
    )
    assert point == low == high == 0.75
    assert analysis.kept_interval(floor, floor, units, strata, draws=5, seed=2) == (
        0.0,
        0.0,
        0.0,
    )


def test_fit_drivers_rank_measures_within_each_class_with_exact_p() -> None:
    rows: list[dict[str, Any]] = []
    for kind in ("Mage", "Priest"):
        for rank, newcomer in enumerate(["A", "B", "C", "D", "E"]):
            row: dict[str, Any] = {"newcomer": newcomer, "newcomer_class": kind}
            for name, _ in analysis.MEASURES:
                row[f"{name}_change"] = 0.0
            row["margin_change"] = -float(rank)
            row["newcomer_distance_change"] = float(rank)
            row["newcomer_deaths_change"] = float(rank % 2)
            rows.append(row)
    rows.append({"newcomer": "Random", "newcomer_class": "Mage", "margin_change": -9.0})
    rows.append({"newcomer": "Host", "newcomer_class": "Mage", "margin_change": 0.0})
    drivers = {row["measure"]: row for row in analysis.fit_drivers(rows, "Host")}
    distance = drivers["newcomer_distance"]
    assert (
        distance["rho_Mage"] == distance["rho_Priest"] == distance["mean_rho"] == -1.0
    )
    # Both classes reversed exactly: 1 of 120 orderings each, two-sided.
    assert distance["p_value"] == pytest.approx(2 / 120**2)
    assert math.isnan(drivers["newcomer_damage"]["mean_rho"])


def test_map_end_rows_group_by_opponent_map_and_spawn_end() -> None:
    games = _games([4.0, 2.0, 6.0, -1.0], [2.0] * 4)
    games["spawn_end"] = np.asarray(["team_a", "team_b", "team_a", None], dtype=object)
    left, right, unknown = analysis.map_end_rows(games, "Team", "A", "Mage")
    assert (unknown["side"], unknown["games"]) == ("unknown", 1)
    assert (left["map_id"], left["side"], left["games"]) == (47, "left", 2)
    assert left["margin"] == 5.0 and right["margin"] == 2.0
    assert (right["wins"], right["losses"], right["side"]) == (1, 0, "right")
    assert left["team"] == "Team" and left["newcomer_class"] == "Mage"
    assert left["mage_distance"] == left["warrior_distance"] == 2.0
