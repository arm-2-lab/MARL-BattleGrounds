"""Check the human_games_analysis stage on small hand-made games.

Contracts: the packaged settings compose with the nine Season 0 display names
keyed as the saved tournament records them, a 10-tick window and 2,000
luck slides from seed 20261016;
agent_columns lists an agent's own columns and no pair columns; mirrored reads
a game from Team B's side by exchanging team_a and team_b and moving each
agent index by five; per_tick_value puts per-game totals (counts, health,
ticks) per 100 ticks and keeps fractions, averages and derived ratios as they
are (per_game_total); screen_games compares each game with
its own references (mean, z in their standard deviations), marks a metric
whose z has one sign in every game and sorts by the z nearest zero, giving
the other team's value for team columns and NaN for agent columns, and
pools each measure over all games: the person's lowest and highest game,
the distinct references' quartiles, how many references reach the person's
level in the person's direction and raw totals for per-game totals only;
class_profile gives deaths and damage received per 100 ticks, the shares of
the Priest's Freedom and useful healing, aura coverage per ally class (NaN
for the giver and with nothing to count), the share of kills by enemy class
and, with a focus slot, the tick-weighted distance to it; ultimate_timing
waits from tick 0 or the tick after the last cooldown_ready (the first tick
the Ultimate can be used again), counts the user's team's kills
in the window after each use, gives the team's kills in an average window and
the ticks to the nearest teammate's and Warrior's Ultimate, and with
game_arrays reads the action mask: ticks the Ultimate could be used before
it was, whether it came on the first such tick, teammates who could have used
theirs and whether an enemy could be attacked; ultimate_summary
pools uses by kind, controller and class, with the shares used together;
screen_games also screens the derived Burst damage per use and the Priest's
healing to slowed teammates per slowed tick; ultimates_together counts each
team's Ultimates within SYNC_TICKS of a teammate's beside random slides of
each agent's times, repeats exactly with the same seed, and gives a lone
user nothing together and a p-value of 1;
burst_value reads each Burst's window (use tick and boosted ticks) and the
Mage's damage outside every window; heal_choices counts Basic heals made with
a hurt teammate in reach and those that went to one; danger_responses counts
teammates falling to the danger line, those the Priest could reach at once
and those it healed in time;
target_choices compares attack targets with an even pick among two or more
offered enemies; head_to_head puts both teams side by side, game by game,
with luck and even-pick shares and the same-direction flag;
compare_alongside places a game among the partner's own games; field_record
counts the opposing System's own games, wins, draws and losses on the person's
maps (each once) and its largest loss, 0 without a loss; main reads a
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
    assert (config.chance_shuffles, config.chance_seed) == (2000, 20261016)


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
    healing = "team_a_hunter_slowed_ally_healing_per_tick"
    slowed = [
        {"episode_length": 100.0, "team_a_hunter_basic_slow_active_steps": 40.0}
        | {"team_a_healing_to_hunter_basic_slow_recipient": 300.0}
    ]
    past = [
        {"episode_length": 100.0, "team_a_hunter_basic_slow_active_steps": 20.0}
        | {"team_a_healing_to_hunter_basic_slow_recipient": 100.0}
    ]
    (support,) = stage.screen_games(slowed, [past], [healing])
    assert (support["person_1"], support["reference_1"]) == (7.5, 5.0)
    assert math.isnan(support["person_sum"]) and support["scale"] == "as is"


def test_screen_games_pools_every_game_and_reference() -> None:
    games = [
        {"episode_length": 100.0, "team_a_kills": 9.0},
        {"episode_length": 200.0, "team_a_kills": 14.0},
    ]
    shared = [
        {"episode_length": 100.0, "team_a_kills": value}
        for value in (2.0, 4.0, 6.0, 8.0)
    ]
    # Both games share one reference list: its four games count once.
    (kills,) = stage.screen_games(games, [shared, shared], ["team_a_kills"])
    assert kills["same_direction"]
    # 14 kills in 200 ticks are 7 per 100 ticks; the raw total stays 23.
    assert kills["person_low"] == pytest.approx(7.0) and kills["person_high"] == 9.0
    assert kills["reference_games"] == 4.0
    quartiles = ("reference_q25", "reference_median", "reference_q75")
    assert tuple(kills[name] for name in quartiles) == (3.5, 5.0, 6.5)
    # Only the reference game with 8 reaches the person's lowest game, 7.
    assert kills["reference_at_person_level"] == 1.0
    assert (kills["person_sum"], kills["reference_sum"]) == (23.0, 20.0)
    below = [{"episode_length": 100.0, "team_a_kills": 1.0}]
    (fewer,) = stage.screen_games(below, [shared], ["team_a_kills"])
    assert fewer["reference_at_person_level"] == 0.0
    mixed = [{"episode_length": 100.0, "team_a_kills": 1.0}, games[0]]
    (split,) = stage.screen_games(mixed, [shared, shared], ["team_a_kills"])
    assert math.isnan(split["reference_at_person_level"])


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
    # Ready at tick 20 means usable from tick 21, so the use at 25 waited 4.
    assert (second["wait"], second["kills_after"]) == (4, 0)
    assert math.isnan(second["usable_wait"])
    assert first["kills_per_window"] == pytest.approx(2 / 30 * 10)
    assert (enemy["team"], enemy["class"], enemy["person"]) == ("B", "Mage", False)
    assert enemy["kills_per_window"] == 0.0
    labeled = [row | {"kind": "against", "who": "Person"} for row in (first, second)]
    (pooled,) = stage.ultimate_summary(labeled)
    assert (pooled["uses"], pooled["median_wait"]) == (2, 3.0)
    assert pooled["kills_after_per_use"] == 1.0
    # Only the first Mage use was within SYNC_TICKS of the Warrior's.
    assert pooled["with_warrior_share"] == pooled["with_teammate_share"] == 0.5


def test_ultimates_together_beside_luck() -> None:
    rows = [
        {"number": 1, "kind": "against", "who": "Person", "team": "A"}
        | {"slot": slot, "tick": tick}
        for tick in (10, 47, 120, 151)
        for slot in (0, 1)
    ]
    rows += [
        {"number": 1, "kind": "against", "who": "Rec", "team": "B"}
        | {"slot": 5, "tick": tick}
        for tick in (20, 70)
    ]
    first = stage.ultimates_together(rows, {1: 200}, 200, 3)
    person, alone = first
    assert (person["games"], person["ultimates"], person["together"]) == (1, 8, 8)
    assert person["who"] == "Person" and person["together_share"] == 1.0
    # Slid apart at random, the two agents' four uses rarely all meet again.
    assert person["chance_median"] < 0.5 and person["chance_at_least"] < 0.1
    assert (alone["together"], alone["chance_high"]) == (0, 0.0)
    assert alone["chance_at_least"] == 1.0
    assert stage.ultimates_together(rows, {1: 200}, 200, 3) == first


def _arrays(ticks: int) -> stage.GameArrays:
    return stage.GameArrays(
        alive=np.ones((ticks + 1, 10), bool),
        health=np.tile(np.array([80.0, 200, 100, 100, 100] * 2), (ticks + 1, 1)),
        burst_left=np.zeros((ticks + 1, 10), np.int64),
        choices=np.zeros((ticks + 1, 10, 11, 2), bool),
        max_health=np.array([80.0, 200, 100, 100, 100] * 2),
    )


def _use(
    slot: int, part: str = "ultimate", target: int | None = None
) -> SimpleNamespace:
    return _event(
        "ability_activated",
        ability_component=part,
        source_global_slot=slot,
        recipient_global_slot=target,
    )


def test_game_arrays_and_what_the_game_allowed() -> None:
    joint = np.zeros((10, 11, 2), bool)
    joint[0, 7, 0] = True
    frame = SimpleNamespace(
        snapshot=SimpleNamespace(
            alive_mask=(True,) * 10,
            current_health=(80.0,) * 10,
            mage_burst_damage_amplification_durations=(5,) + (0,) * 9,
        ),
        action_mask=SimpleNamespace(select_target_use_ultimate_joint_mask=joint),
    )
    built = stage.game_arrays([frame, frame], [80.0] * 10)
    assert built.choices.shape == (2, 10, 11, 2) and built.burst_left[0, 0] == 5
    arrays = _arrays(6)
    # The Warrior (slot 1) could use Charge on ticks 1 and 3, and uses it on 4;
    # the Mage could too on tick 4, and could attack an enemy then.
    arrays.choices[[1, 3, 4], 1, 7, 1] = True
    arrays.choices[4, 0, 0, 1] = True
    arrays.choices[4, 1, 6, 0] = True
    events: list[list[Any]] = [[] for _ in range(6)]
    events[4].append(_use(1, target=6))
    (charge,) = stage.ultimate_timing(events, (1,), 3, arrays)
    assert (charge["wait"], charge["usable_wait"]) == (4, 2)
    assert not charge["first_usable"] and charge["mates_usable"] == 1
    assert charge["enemy_attackable"]


def test_burst_value_reads_windows_against_the_normal_rate() -> None:
    arrays = _arrays(10)
    arrays.burst_left[3:6, 0] = (3, 2, 1)  # boosted on ticks 3-5 after a use on 2
    events: list[list[Any]] = [[] for _ in range(10)]
    events[2].append(_use(0))
    for tick, amount in ((0, 4.0), (3, 12.0), (5, 12.0), (8, 4.0)):
        events[tick].append(
            _event(
                "source_damage_output",
                source_global_slot=0,
                source_modified_damage_output=amount,
                recipient_damage_modifier=1.0,
            )
        )
    value = stage.burst_value(events, arrays)
    mage = value["A"]
    assert (mage["bursts"], mage["window_ticks"], mage["window_damage"]) == (1, 4, 24)
    assert (mage["other_ticks"], mage["other_damage"]) == (6, 8)
    assert mage["opening_damage"] == 24.0
    assert value["B"]["bursts"] == 0 and math.isnan(value["B"]["opening_damage"])


def test_priest_heal_choices_and_answers_to_danger() -> None:
    arrays = _arrays(6)
    arrays.health[1:, 3] = 30.0  # the Rogue falls to 30 of 100 on tick 1
    arrays.health[:, 1] = 196.0  # the Warrior misses only 4: not hurt
    arrays.choices[:, 4, 1:6, 0] = True  # the Priest can reach all five
    events: list[list[Any]] = [[] for _ in range(6)]
    events[0].append(_use(4, "basic", 1))  # nobody hurt yet: not counted
    events[2].append(_use(4, "basic", 1))  # Rogue hurt and in reach: a miss
    events[3].append(_use(4, "basic", 3))  # to the hurt Rogue
    heals = stage.heal_choices(events, arrays, 8.0)["A"]
    assert (heals["heals"], heals["with_hurt_in_reach"], heals["to_hurt"]) == (3, 2, 1)
    danger = stage.danger_responses(events, arrays)["A"]
    # The Rogue fell on tick 1; the Priest healed it on tick 3, two ticks later.
    assert (danger["falls"], danger["in_reach"], danger["answered"]) == (1, 1, 1)
    arrays.alive[1, 4] = False  # with the Priest dead, the fall does not count
    assert stage.danger_responses(events, arrays)["A"]["falls"] == 0


def test_target_choices_against_an_even_pick() -> None:
    arrays = _arrays(3)
    arrays.choices[:, 0, [6, 10], 0] = True  # Team A's Mage can hit Mage or Priest
    arrays.choices[:, 5, 6, 0] = True  # Team B's Mage has only one choice
    events: list[list[Any]] = [[] for _ in range(3)]
    events[0].append(_use(0, "basic", 9))
    events[1].append(_use(0, "basic", 9))
    events[2].append(_use(5, "basic", 0))
    choices = stage.target_choices(events, arrays)
    priest = choices["A"]["Priest"]
    assert (priest["offered"], priest["hit"], priest["even_pick"]) == (2, 2, 1)
    assert choices["A"]["Mage"]["hit"] == 0 and choices["B"]["Mage"]["offered"] == 0


def test_head_to_head_sets_both_teams_side_by_side() -> None:
    def game(first: bool, opening: float) -> dict[str, Any]:
        rows = [
            {"team": team, "class": "Mage", "first_usable": team == "B" or first}
            | {"usable_wait": 0, "enemy_attackable": team == "A", "mates_usable": 1}
            | {"teammate_gap": 1.0 if team == "A" else 9.0, "warrior_gap": 1.0}
            for team in ("A", "B")
        ]
        burst = {
            team: {"opening_damage": opening if team == "A" else 0.0}
            | {"window_damage": 30.0, "window_ticks": 6.0}
            | {"other_damage": 10.0, "other_ticks": 10.0}
            for team in ("A", "B")
        }
        counts = {"heals": 2.0, "with_hurt_in_reach": 2.0, "to_hurt": 1.0}
        danger = {"falls": 2.0, "in_reach": 2.0, "answered": 1.0}
        target = {
            name: {"offered": 2.0, "hit": 1.0, "even_pick": 1.0}
            for name in stage.CLASSES
        }
        return {
            "ultimates": rows,
            "burst": burst,
            "heals": {"A": counts, "B": counts},
            "danger": {"A": danger, "B": danger},
            "targets": {"A": target, "B": target},
        }

    rows = {
        row["measure"]: row
        for row in stage.head_to_head(
            [game(False, 80.0), game(False, 90.0)], {"A": 0.6}
        )
    }
    first = rows["Ultimates used on the first tick the agent could use them"]
    assert (first["person"], first["opponent"], first["same_direction"]) == (
        0.0,
        1.0,
        True,
    )
    opening = rows["Damage dealt by the opening Burst (mean over games)"]
    assert (opening["person"], opening["person_1"], opening["opponent_2"]) == (
        85.0,
        80.0,
        0.0,
    )
    together = rows["Ultimates within 4 ticks of a teammate's"]
    assert together["person_chance"] == 0.6 and math.isnan(together["opponent_chance"])
    priest = rows["Attacks on the enemy Priest when it was in reach"]
    assert (priest["person"], priest["person_chance"]) == (0.5, 0.5)
    # Equal values in a game are not a direction.
    assert not priest["same_direction"]


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
    replay = SimpleNamespace(replay=SimpleNamespace(transitions=[], frames=[]))

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
            "chance_shuffles": 10,
            "chance_seed": 0,
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
    # The stand-in replays have no Ultimates and no frames: nothing is counted
    # together and there is no head-to-head row.
    assert record["together"] == [] and record["head_to_head"] == []
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


def test_field_record_sums_the_systems_games_on_the_persons_maps() -> None:
    def row(system: str, map_id: int, margin: float) -> dict[str, Any]:
        return {
            "system": system,
            "map_id": float(map_id),
            "score_difference": margin,
            "team_a_win": float(margin > 0),
            "team_a_draw": float(margin == 0),
            "team_a_loss": float(margin < 0),
        }

    reference = [
        row("Host", 47, 5.0),
        row("Host", 48, -14.0),
        row("Host", 48, 0.0),
        row("Host", 50, -20.0),
        row("Other", 47, -30.0),
    ]
    assert stage.field_record(reference, "Host", [48, 47, 48]) == {
        "system": "Host",
        "maps": "47 48",
        "games": 3,
        "wins": 1,
        "draws": 1,
        "losses": 1,
        "largest_loss": 14.0,
    }
    assert stage.field_record(reference[:1], "Host", [47])["largest_loss"] == 0.0
