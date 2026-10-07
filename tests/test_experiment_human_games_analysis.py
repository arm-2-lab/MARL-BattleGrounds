"""Check the human_games_analysis stage on small hand-made games.

Contracts: the packaged settings compose with the nine Season 0 display names
keyed as the saved tournament records them and a 10-tick window;
agent_columns lists an agent's own columns and no pair columns; mirrored reads
a game from Team B's side by exchanging team_a and team_b and moving each
agent index by five; per_tick_value puts per-game totals (counts, health,
ticks) per 100 ticks and keeps fractions, averages and derived ratios as they
are (per_game_total); screen_games compares each game with
its own references (mean, z in their standard deviations), marks a metric
whose z has one sign in every game and sorts by the z nearest zero, giving
the other team's value for team columns and NaN for agent columns;
class_profile gives deaths and damage received per 100 ticks, the shares of
the Priest's Freedom and useful healing, aura coverage per ally class (NaN
for the giver and with nothing to count), the share of kills by enemy class
and, with a focus slot, the tick-weighted distance to it; ultimate_timing
waits from tick 0 or the last cooldown_ready, counts the user's team's kills
in the window after each use, gives the team's kills in an average window and
the ticks to the nearest teammate's and Warrior's Ultimate; ultimate_summary
pools uses by kind, controller and class, with the shares used together;
screen_games also screens the derived Burst damage per use;
compare_alongside places a game among the partner's own games; main reads a
series and a field run through the public loaders, screens against all of
the opponent's wins and its wins by lead_margin or more, refuses an empty series
and a System without a recorded name, and writes its files. Stand-ins replace
saved runs and replays, so no game is played.
"""

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

import marl_battlegrounds as marl_bgs
import marl_battlegrounds.evaluation as evaluation
from marl_battlegrounds.experiments import human_games_analysis as stage


def test_settings_compose() -> None:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        config = compose(
            config_name="human_games_analysis",
            overrides=["series_dir=s", "field_dir=f", "output_dir=o"],
        )
    assert len(config.names) == 9
    assert config.names["IPPO at step 250,675,200 (run)"] == "Recurrent IPPO"
    assert (config.window_ticks, config.lead_margin) == (10, 12)


def test_agent_columns_and_mirrored_rows() -> None:
    columns = stage.agent_columns(0)
    assert "agent_0_kills" in columns or "agent_0_deaths" in columns
    assert all(name.startswith("agent_0_") for name in columns)
    assert not any("_to_agent_" in name or "_and_agent_" in name for name in columns)
    row = {
        "team_a_kills": 3.0,
        "team_b_kills": 5.0,
        "agent_1_deaths": 1.0,
        "agent_6_deaths": 2.0,
        "agent_4_to_agent_9_damage_done": 7.0,
        "episode_length": 100.0,
    }
    flipped = stage.mirrored(row)
    assert flipped["team_a_kills"] == 5.0 and flipped["team_b_kills"] == 3.0
    assert flipped["agent_1_deaths"] == 2.0 and flipped["agent_6_deaths"] == 1.0
    assert flipped["agent_9_to_agent_4_damage_done"] == 7.0
    assert flipped["episode_length"] == 100.0


def test_per_tick_values() -> None:
    assert stage.per_tick_value("team_a_kills", 5.0, 200.0) == 2.5
    fraction = "team_a_multi_contributor_kill_fraction"
    assert stage.per_tick_value(fraction, 0.4, 200.0) == 0.4
    # An average stays an average, and a derived ratio keeps its scale.
    wait = "team_a_mean_observed_respawn_wait_steps"
    assert stage.per_tick_value(wait, 6.0, 200.0) == 6.0
    assert not stage.per_game_total("team_a_burst_damage_per_use")
    assert stage.per_tick_value("team_a_burst_damage_per_use", 80.0, 200.0) == 80.0


def test_screen_games_compares_each_game_with_its_own_references() -> None:
    games = [
        {"episode_length": 100.0, "team_a_kills": 9.0, "team_b_kills": 2.0},
        {"episode_length": 200.0, "team_a_kills": 10.0, "team_b_kills": 4.0},
    ]
    references = [
        [{"episode_length": 100.0, "team_a_kills": value} for value in (4.0, 6.0)],
        [{"episode_length": 100.0, "team_a_kills": value} for value in (6.0, 8.0)],
    ]
    rows = stage.screen_games(
        games, references, ["team_a_kills", "agent_0_damage_done"]
    )
    kills = rows[0]
    assert kills["measure"] == "kills" and kills["scale"] == "per 100 ticks"
    assert (kills["person_1"], kills["reference_1"], kills["z_1"]) == (9.0, 5.0, 4.0)
    # Game 2 is 200 ticks long: 10 kills are 5 per 100 ticks, below 7.
    assert (kills["person_2"], kills["z_2"]) == (5.0, -2.0)
    assert kills["other_side_1"] == 2.0 and kills["other_side_2"] == 2.0
    assert not kills["same_direction"] and kills["smallest_z"] == 0.0
    agent = rows[1]
    assert math.isnan(agent["other_side_1"]) and math.isnan(agent["z_1"])
    same = stage.screen_games(games[:1], references[:1], ["team_a_kills"])
    assert same[0]["same_direction"] and same[0]["smallest_z"] == 4.0
    burst = [
        {"episode_length": 100.0, "team_a_burst_damage": 90.0}
        | {"team_a_mage_burst_applications": 3.0, "team_b_burst_damage": 20.0}
        | {"team_b_mage_burst_applications": 0.0}
    ]
    past = [
        {"episode_length": 100.0, "team_a_burst_damage": damage}
        | {"team_a_mage_burst_applications": 2.0}
        for damage in (40.0, 60.0)
    ]
    (per_use,) = stage.screen_games(burst, [past], ["team_a_burst_damage_per_use"])
    assert (per_use["person_1"], per_use["reference_1"]) == (30.0, 25.0)
    assert math.isnan(per_use["other_side_1"]) and per_use["scale"] == "as is"


def test_class_profile_reads_classes_shares_and_focus_distance() -> None:
    game: dict[str, Any] = {"episode_length": 200.0}
    game |= {"agent_3_deaths": 2.0, "agent_1_damage_received": 50.0}
    game |= {"agent_7_deaths": 3.0, "agent_9_deaths": 1.0}
    game |= {
        "agent_4_to_agent_0_priest_freedom_applications": 3.0,
        "agent_4_to_agent_4_priest_freedom_applications": 1.0,
        "agent_4_to_agent_1_effective_healing_done": 30.0,
        "agent_4_to_agent_0_effective_healing_done": 10.0,
        "agent_0_to_agent_4_mage_aura_covered_steps": 20.0,
        "agent_0_to_agent_4_mage_aura_eligible_steps": 80.0,
        "agent_0_and_agent_2_ally_distance_mean": 2.0,
        "agent_0_and_agent_2_ally_distance_observations": 10.0,
    }
    other = {
        "episode_length": 200.0,
        "agent_0_and_agent_2_ally_distance_mean": 5.0,
        "agent_0_and_agent_2_ally_distance_observations": 30.0,
    }
    profile = stage.class_profile([game, other], focus=0)
    mage, warrior, hunter, rogue, priest = range(5)
    assert profile["deaths"][rogue] == 0.5
    assert profile["damage_received"][warrior] == 12.5
    assert profile["kills_by_victim"][hunter] == 0.75
    assert profile["freedom_share"] == [0.75, 0.0, 0.0, 0.0, 0.25]
    assert profile["healing_share"] == [0.25, 0.75, 0.0, 0.0, 0.0]
    assert profile["mage_aura_coverage"][priest] == 0.25
    assert math.isnan(profile["mage_aura_coverage"][mage])
    assert math.isnan(profile["mage_aura_coverage"][warrior])
    assert profile["distance_to_focus"][hunter] == 4.25
    assert math.isnan(profile["distance_to_focus"][mage])
    assert "distance_to_focus" not in stage.class_profile([game])


def _event(kind: str, **fields: object) -> SimpleNamespace:
    return SimpleNamespace(event_type=kind, **fields)


def test_ultimate_timing_and_summary() -> None:
    events: list[list[SimpleNamespace]] = [[] for _ in range(30)]
    events[2].append(
        _event("ability_activated", ability_component="ultimate", source_global_slot=0)
    )
    events[4].append(_event("agent_died", recipient_global_slot=7))
    events[4].append(
        _event("ability_activated", ability_component="ultimate", source_global_slot=1)
    )
    events[6].append(_event("agent_died", recipient_global_slot=8))
    events[20].append(_event("cooldown_ready", agent_global_slot=0))
    events[25].append(
        _event("ability_activated", ability_component="ultimate", source_global_slot=0)
    )
    events[26].append(
        _event("ability_activated", ability_component="basic", source_global_slot=5)
    )
    events[27].append(
        _event("ability_activated", ability_component="ultimate", source_global_slot=5)
    )
    rows = stage.ultimate_timing(events, (0,), 10)
    first, warrior, second, enemy = rows
    # The Mage's first Ultimate came 2 ticks before the Warrior's.
    assert (first["teammate_gap"], first["warrior_gap"]) == (2.0, 2.0)
    assert math.isnan(warrior["warrior_gap"]) and warrior["teammate_gap"] == 2.0
    assert math.isnan(enemy["teammate_gap"])
    assert (first["wait"], first["kills_after"], first["person"]) == (2, 2, True)
    assert (second["wait"], second["kills_after"]) == (5, 0)
    assert first["kills_per_window"] == pytest.approx(2 / 30 * 10)
    assert (enemy["team"], enemy["class"], enemy["person"]) == ("B", "Mage", False)
    assert enemy["kills_per_window"] == 0.0
    labeled = [row | {"kind": "against", "who": "Person"} for row in (first, second)]
    (pooled,) = stage.ultimate_summary(labeled)
    assert (pooled["uses"], pooled["median_wait"]) == (2, 3.5)
    assert pooled["kills_after_per_use"] == 1.0
    # Only the first Mage use was within SYNC_TICKS of the Warrior's.
    assert pooled["with_warrior_share"] == pooled["with_teammate_share"] == 0.5


def test_compare_alongside_places_a_game_among_the_partners_games() -> None:
    def game(margin: float, result: str) -> dict[str, float]:
        flags = {
            f"team_a_{name}": float(name == result) for name in ("win", "draw", "loss")
        }
        return {
            "score_difference": margin,
            "team_a_kills": 10.0,
            "team_a_deaths": 5.0,
        } | flags

    reference = [game(-2.0, "loss"), game(0.0, "draw"), game(4.0, "win")]
    record = stage.compare_alongside(game(0.0, "draw"), reference)
    assert record["games"] == 3 and record["margin_rank"] == 0.5
    assert (record["margin_low"], record["margin_high"]) == (-2.0, 4.0)
    assert (record["wins"], record["draws"], record["losses"]) == (1, 1, 1)


class _Field:
    # Stands in for load_results: the field's games, read from each side.
    def system_metrics(self, table: str, **options: object) -> dict[str, Any]:
        assert table == "full_metrics" and options["by"] == "game"
        rows = 4
        values: dict[str, Any] = {
            column: np.zeros(rows) for column in cast(list[str], options["columns"])
        }
        values["system_name"] = np.array(["ippo-run"] * rows, dtype=object)
        values["opponent_name"] = np.array(["pqn-run"] * rows, dtype=object)
        values["spawn_end"] = np.array(["team_a", "team_b"] * 2, dtype=object)
        values["map_id"] = np.array([47, 47, 48, 48], dtype=object)
        values["team_a_win"] = np.array([1, 1, 0, 1], dtype=object)
        values["episode_length"] = np.full(rows, 100.0)
        values["score_difference"] = np.array([5.0, 7.0, -1.0, 3.0])
        values["team_a_kills"] = np.array([8.0, 9.0, 4.0, 6.0])
        return values


def _series(folder: Path, games: list[dict[str, Any]]) -> None:
    for details in games:
        game = folder / f"game-{details['number']:02d}"
        game.mkdir(parents=True)
        (game / "game.json").write_text(json.dumps(details | {"replay": "r.json"}))
        with (game / "full_metrics.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            header = ["episode_length", "score_difference", "team_a_kills"]
            writer.writerow([*header, "team_a_deaths"])
            writer.writerow([50, 4, 6, 2])


def test_main_reads_the_series_and_the_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = {"partner": None, "player_class": None, "outcome": "win"}
    _series(
        tmp_path / "series",
        [
            base
            | {
                "number": 1,
                "kind": "against",
                "map_id": 47,
                "side": "left",
                "opponent": "Rec",
            },
            base
            | {
                "number": 2,
                "kind": "alongside",
                "map_id": 47,
                "side": "left",
                "opponent": "PQN",
            }
            | {"partner": "Rec", "player_class": "mage"},
        ],
    )
    (tmp_path / "series" / "game-03").mkdir()  # not played yet: skipped
    replay = SimpleNamespace(replay=SimpleNamespace(transitions=[]))

    def load_results(path: object) -> _Field:
        return _Field()

    def load_replay(path: object) -> SimpleNamespace:
        return replay

    monkeypatch.setattr(marl_bgs, "load_results", load_results)
    monkeypatch.setattr(evaluation, "load_replay", load_replay)
    config = OmegaConf.create(
        {
            "series_dir": str(tmp_path / "series"),
            "field_dir": str(tmp_path / "field"),
            "names": {"ippo-run": "Rec", "pqn-run": "PQN"},
            "window_ticks": 10,
            "lead_margin": 6,
            "output_dir": str(tmp_path / "out"),
        }
    )
    record = stage.main(config)
    assert [game["number"] for game in record["games"]] == [1, 2]
    kills = next(row for row in record["against_screen"] if row["measure"] == "kills")
    # 6 kills in 50 ticks against the opponent's two wins on map 47 (8, 9).
    assert kills["person_1"] == 12.0 and kills["reference_1"] == 8.5
    # Lead-matched: only the opponent's win by 7 on map 47 is at least 6.
    lead = next(row for row in record["lead_screen"] if row["measure"] == "kills")
    assert lead["reference_1"] == 9.0
    (alongside,) = record["alongside"]
    assert alongside["games"] == 1 and alongside["margin_mean"] == 5.0
    for name in (
        "against_screen.csv",
        "against_screen_lead.csv",
        "alongside.csv",
        "summary.md",
        "analysis.json",
    ):
        assert (tmp_path / "out" / name).exists()
    config.names = {"ippo-run": "Rec"}
    with pytest.raises(ValueError, match="no recorded name for"):
        stage.main(config)
    empty = tmp_path / "empty"
    empty.mkdir()
    config.series_dir = str(empty)
    with pytest.raises(ValueError, match="no recorded game"):
        stage.main(config)
