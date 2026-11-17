"""Check that replay_metrics measures saved Capture the Flag games in their own columns.

The four hand-built games of the Capture the Flag evaluation-replay test (map 0,
three champions a team; Team A stands still, Team B is the tdm-alpha bot) are
saved by evaluate_episodes and measured with replay_metrics(metrics="full").
Contracts: a Capture the Flag game returns exactly its own column list, Team
Deathmatch's full columns without the Red Zone and then the Capture the Flag
columns, and its score columns count captures. Each game's carry reads as the
scene was built: a steal carried two ticks to the end (game 1), a carry from the
scenario's start that captures (game 2), one that dies (game 3), and a ground
pickup (game 4). The checks Baselines' counter runs hold in every game:
captures equal the score, pickups are steals plus ground pickups, the escort
split adds up to the carry ticks, and the ticks of carries that ended add up to
at most the carry ticks. Means with no measurement and an agent paired with
itself are blank. The catalog gives every Capture the Flag column a value, a
unique name, and a summary weight that names a real column, and marks healing
given as Priest-only.
"""

# pyright: reportPrivateUsage=false

import math
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
    MAX_ALLIES_NEAR,
    capture_the_flag_data,
    escort_split,
    initialize_capture_the_flag,
)
from marl_battlegrounds.evaluation._metrics.capture_the_flag_catalog import (
    capture_the_flag_columns,
    capture_the_flag_metric_columns,
)
from marl_battlegrounds.evaluation.analysis import replay_metrics
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.policy_execution import Policy
from marl_battlegrounds.evaluation.replay_io import load_replay


@pytest.fixture(scope="module")
def games(tmp_path_factory: pytest.TempPathFactory) -> dict[int, dict[str, float]]:
    out = tmp_path_factory.mktemp("ctf-metrics")
    evaluate_episodes(
        Policy("still", _still, jnp.int32(0), jnp.int32(0)),
        "tdm-alpha",
        [
            EpisodeSpec(game, _CONFIG, initial_state=start)
            for game, (start, _) in _GAMES.items()
        ],
        num_envs=2,
        save_replays=len(_GAMES),
        output_dir=out,
    )
    measured: dict[int, dict[str, float]] = {}
    for path in out.rglob("*.marlbg-replay.json"):
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
    assert len(names) == len(columns)
    data = capture_the_flag_data(initialize_capture_the_flag())
    assert {(column.scope, column.stem) for column in columns} == set(data)
    for column in columns:
        assert column.summary_weight is None or column.summary_weight in names
        given = "healing_done_to_allies" in column.stem or "healing_to_allies" in (
            column.stem
        )
        assert (column.required_class_id == PRIEST_CLASS_ID) is (
            given or "healing_focus" in column.stem
        )
