"""Check that replay_metrics measures saved Capture the Flag games in their own columns.

The four hand-built games of the Capture the Flag evaluation-replay test (map 0,
three champions a team; Team A stands still, Team B is the tdm-alpha bot) are
saved by evaluate_episodes with full metrics and measured with
replay_metrics(metrics="full"). Contracts: evaluate saves the run under the
"marlbg.ctf.scalar" version 1 schema (12,124 names, pinned by digest), load_results'
system_metrics reads its Capture the Flag columns from each System's side, and
its full_metrics.csv holds the same
value as replay_metrics in every column of every game (a blank cell is NaN
there), so live and replay measurement are one calculation. A Capture the Flag
game returns exactly its own column list, Team
Deathmatch's full columns without the Red Zone and then the Capture the Flag
columns, and its score columns count captures. Each game's carry reads as the
scene was built: a steal carried two ticks to the end (game 1), a carry from the
scenario's start that captures (game 2), one that dies (game 3), and a ground
pickup (game 4). The checks Baselines' counter runs hold in every game:
captures equal the score, pickups are steals plus ground pickups, the escort
split adds up to the carry ticks, and the ticks of carries that ended add up to
at most the carry ticks. Means with no measurement and an agent paired with
itself are blank. The catalog gives every Capture the Flag column a value, a
name unique in the whole saved list, and a summary weight that names a real
column; it marks healing, rescues and Freedom given, and a team's carrier
rescue rows, as needing a Priest, with Core's Priest class ID, and each escort
column's label names its own escort size.
"""

# pyright: reportPrivateUsage=false

import csv
import json
import math
from hashlib import sha256
from pathlib import Path

import jax.numpy as jnp
import pytest
from tests.evaluation.replays.test_capture_the_flag_evaluation_replays import (
    _CONFIG,
    _GAMES,
    _still,
)

from marl_battlegrounds.core.types import PRIEST_CLASS_ID
from marl_battlegrounds.evaluation._metrics.capture_the_flag import (
    capture_the_flag_data,
    initialize_capture_the_flag,
)
from marl_battlegrounds.evaluation._metrics.capture_the_flag_catalog import (
    MAX_ALLIES_NEAR,
    capture_the_flag_columns,
    capture_the_flag_metric_columns,
    escort_split,
)
from marl_battlegrounds.evaluation.analysis import replay_metrics
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.policy_execution import Policy
from marl_battlegrounds.evaluation.replay_io import load_replay
from marl_battlegrounds.evaluation.results import load_results


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("ctf-metrics")
    evaluate_episodes(
        Policy("still", _still, jnp.int32(0), jnp.int32(0)),
        "tdm-alpha",
        [
            EpisodeSpec(game, _CONFIG, initial_state=start)
            for game, (start, _) in _GAMES.items()
        ],
        num_envs=2,
        metrics="full",
        save_replays=len(_GAMES),
        output_dir=out,
    )
    return out


@pytest.fixture(scope="module")
def games(saved_run: Path) -> dict[int, dict[str, float]]:
    measured: dict[int, dict[str, float]] = {}
    for path in saved_run.rglob("*.marlbg-replay.json"):
        loaded = load_replay(path)
        episode = loaded.replay.header.context.identity.episode_id
        measured[int(episode.rsplit("-", 1)[1])] = replay_metrics(
            Path(path), metrics="full"
        )
    return measured


def test_a_capture_the_flag_game_has_its_own_columns(
    games: dict[int, dict[str, float]],
) -> None:
    names = [column.name for column in capture_the_flag_metric_columns()]
    assert sorted(games) == sorted(_GAMES)
    for values in games.values():
        assert list(values) == names
        assert not any("red_zone" in name for name in values)
    assert games[2]["team_a_score"] == games[2]["team_a_flag_captures"] == 1.0


def test_evaluate_saves_the_same_values_as_replay_metrics(
    saved_run: Path, games: dict[int, dict[str, float]]
) -> None:
    (details,) = saved_run.rglob("run_details.json")
    manifest = json.loads(details.read_text())
    assert (manifest["metric_schema_id"], manifest["metric_schema_version"]) == (
        "marlbg.ctf.scalar",
        1,
    )
    with (details.parent / "full_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [column.name for column in capture_the_flag_metric_columns()]
    # This pins version 1's list: changing it needs "marlbg.ctf.scalar" version 2,
    # with version 1's list kept for its readers.
    assert len(names) == 12_124
    assert sha256("\n".join(names).encode()).hexdigest() == (
        "46594855411d420af915c2224df33bd1cc93d474c80e297af89ba0b189dd6424"
    )
    assert sorted(int(row["episode_id"]) for row in rows) == sorted(_GAMES)
    for row in rows:
        assert list(row)[-len(names) :] == names
        replayed = games[int(row["episode_id"])]
        for name in names:
            # A blank cell is a value replay_metrics gives as NaN.
            cell = row[name]
            if cell == "":
                assert math.isnan(replayed[name]), name
            else:
                assert float(cell) == replayed[name], name


def test_load_results_reads_the_run_from_each_system_side(saved_run: Path) -> None:
    (details,) = saved_run.rglob("run_details.json")
    results = load_results(details.parent)
    table = results.table("full_metrics")
    carried = {"agent_0_steps__carrying", "agent_5_steps__carrying"}
    games = results.system_metrics("full_metrics", by="game", columns=sorted(carried))
    # Team B's System reads its own carry from Team B's columns, mirrored.
    by_side = {
        "still": "agent_0_steps__carrying",
        "tdm-alpha": "agent_5_steps__carrying",
    }
    for system, name in by_side.items():
        mine = games["system_name"] == system
        assert sorted(games["agent_0_steps__carrying"][mine]) == sorted(table[name])


def test_each_carry_reads_as_its_scene_was_built(
    games: dict[int, dict[str, float]],
) -> None:
    steal = games[1]
    assert steal["agent_0_flag_steals"] == steal["team_a_flag_pickups"] == 1.0
    assert steal["agent_0_steps__carrying"] == 2.0
    # The carry is still open at the end: it counts with its ticks so far.
    assert steal["agent_0_carry_duration_mean"] == 2.0
    assert steal["agent_0_steps_with_0_allies_near__carrying"] == 2.0
    assert steal["agent_0_distance_to_the_ally_flag_home_mean__carrying"] == 18.5
    capture = games[2]
    assert capture["agent_0_carry_steps_ending_in_a_capture"] == 1.0
    assert capture["agent_0_carry_steps_in_capture_range_with_the_ally_flag_away"] == 1
    assert math.isnan(capture["agent_0_carry_duration_mean"])
    death = games[3]
    assert death["agent_0_deaths__carrying"] == 1.0
    assert death["agent_0_carry_steps_ending_in_death"] == 1.0
    assert death["agent_0_deaths_with_0_allies_near__carrying"] == 1.0
    assert games[4]["agent_1_dropped_enemy_flag_pickups"] == 1.0


def test_the_counters_checks_hold_in_every_game(
    games: dict[int, dict[str, float]],
) -> None:
    for values in games.values():
        for team in "ab":
            prefix = f"team_{team}_"
            totals = {
                name.removeprefix(prefix): value
                for name, value in values.items()
                if name.startswith(prefix)
            }
            assert totals["flag_captures"] == totals["score"]
            assert totals["flag_pickups"] == (
                totals["flag_steals"] + totals["dropped_enemy_flag_pickups"]
            )
            carrying = totals["steps_with_an_agent__carrying"]
            escorts = sum(
                totals[f"steps_{escort_split(size)}__carrying"]
                for size in range(MAX_ALLIES_NEAR + 1)
            )
            assert escorts == carrying
            ended = (
                totals["carry_steps_ending_in_a_capture"]
                + totals["carry_steps_ending_in_death"]
            )
            assert ended <= carrying
        assert math.isnan(
            values["agent_0_to_agent_0_distance_to_the_ally_carrier_mean"]
        )
        assert math.isnan(values["team_b_carry_duration_mean"])


def test_the_catalog_and_the_values_cover_each_other() -> None:
    columns = capture_the_flag_columns()
    names = {column.name for column in columns}
    # Every name appears once in the whole saved list, Team Deathmatch's shared
    # columns included (so a later priority column such as team_a_flag_pickups
    # cannot appear twice).
    saved = [column.name for column in capture_the_flag_metric_columns()]
    assert len(set(saved)) == len(saved)
    data = capture_the_flag_data(initialize_capture_the_flag())
    assert {(column.scope, column.stem) for column in columns} == set(data)
    given_words = (
        "healing_done_to_allies",
        "healing_to_allies",
        "healing_focus",
        "rescue_contributions_to_allies",
        "freedom_applications_to_allies",
    )
    for column in columns:
        assert column.summary_weight is None or column.summary_weight in names
        # The catalog writes Core's Priest class ID itself; this pins the two. A
        # team's carrier rescue rows need a Priest on the team.
        assert (column.required_class_id == PRIEST_CLASS_ID) is (
            any(word in column.stem for word in given_words)
            or (column.scope == "team" and column.stem.startswith("rescue"))
        )
        for size in range(MAX_ALLIES_NEAR + 1):
            if escort_split(size) in column.stem:
                word = "Ally" if size == 1 else "Allies"
                assert f"Exactly {size} {word} Near" in column.label, column.name
