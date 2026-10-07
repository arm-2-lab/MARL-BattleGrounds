"""Check that marl_bgs.replay_metrics measures a saved game exactly like evaluate.

Two real games (TDM-Alpha against TDM-Beta on map 47, one from each spawn end,
40 ticks) are played on the CPU with full metrics and saved replays. For each
saved replay, replay_metrics(metrics='full') gives every one of the full
catalog's columns, in catalog order, and each value equals the same game's row
of evaluate's saved full_metrics table, with NaN in exactly the same places;
the default, 'priority', gives the priority columns, equal to the saved
priority_metrics table. A path and the loaded replay give the same values. Any
other metrics value fails. The package exports replay_metrics without loading
JAX until it is first used. The human_games stage reads these real replays:
check_replay accepts their map, spawn end, seed and completion and refuses them
only because no person played, and record_game saves their measured result.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation import load_replay
from marl_battlegrounds.evaluation.metric_catalog import (
    METRIC_COLUMNS,
    PRIORITY_METRIC_COLUMNS,
)
from marl_battlegrounds.experiments import human_games


@pytest.fixture(scope="module")
def saved_games(tmp_path_factory: pytest.TempPathFactory) -> marl_bgs.EvaluationResult:
    return marl_bgs.evaluate(
        "tdm-alpha",
        "tdm-beta",
        num_episodes=2,
        maps=[47],
        max_steps=40,
        metrics="full",
        save_replays=2,
        num_envs=2,
        output_dir=tmp_path_factory.mktemp("games"),
    )


def _run_dir(result: marl_bgs.EvaluationResult) -> Path:
    assert result.run_dir is not None
    return Path(result.run_dir)


def _replays(result: marl_bgs.EvaluationResult) -> list[Path]:
    return sorted(_run_dir(result).rglob("*.marlbg-replay.json"))


def _row(table: dict[str, np.ndarray], replay: Path) -> int:
    # A replay's episode ID ends with the run table's episode number.
    episode = load_replay(replay).replay.header.context.identity.episode_id
    rows = np.flatnonzero(table["episode_id"] == int(episode.rsplit("-", 1)[-1]))
    assert rows.size == 1
    return int(rows[0])


@pytest.mark.parametrize(
    ("metrics", "table_name", "columns"),
    [
        ("full", "full_metrics", METRIC_COLUMNS),
        ("priority", "priority_metrics", PRIORITY_METRIC_COLUMNS),
    ],
)
def test_replay_metrics_equal_the_live_tables(
    saved_games: marl_bgs.EvaluationResult,
    metrics: str,
    table_name: str,
    columns: tuple[object, ...],
) -> None:
    table = marl_bgs.load_results(_run_dir(saved_games)).table(table_name)
    replays = _replays(saved_games)
    assert len(replays) == 2
    for replay in replays:
        values = marl_bgs.replay_metrics(replay, metrics=metrics)  # pyright: ignore[reportArgumentType]
        assert list(values) == [column.name for column in columns]  # pyright: ignore[reportAttributeAccessIssue]
        row = _row(table, replay)
        live = np.array([float(table[name][row]) for name in values])
        np.testing.assert_array_equal(np.array(list(values.values())), live)
    loaded = marl_bgs.replay_metrics(load_replay(replays[0]))
    np.testing.assert_array_equal(
        list(loaded.values()), list(marl_bgs.replay_metrics(replays[0]).values())
    )
    assert len(loaded) == len(PRIORITY_METRIC_COLUMNS)


def test_unknown_metrics_value_fails(saved_games: marl_bgs.EvaluationResult) -> None:
    with pytest.raises(ValueError, match="'priority' or 'full'"):
        marl_bgs.replay_metrics(_replays(saved_games)[0], metrics="none")  # pyright: ignore[reportArgumentType]


def test_package_export_is_lazy() -> None:
    code = (
        "import sys, marl_battlegrounds as m; "
        "assert 'replay_metrics' in m.__all__; assert 'jax' not in sys.modules; "
        "m.replay_metrics; assert 'jax' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_human_games_read_real_replays(
    saved_games: marl_bgs.EvaluationResult, tmp_path: Path
) -> None:
    # A private copy, so the game's files never land in the shared run folder.
    replay = tmp_path / _replays(saved_games)[0].name
    shutil.copyfile(_replays(saved_games)[0], replay)
    loaded = load_replay(replay)
    context = loaded.replay.header.context
    keys = {row.name: row.value for row in context.aggregation_keys}
    game = human_games.Game(
        1, "against", 1, 47, keys["marl_bgs.spawn_locations"] == "1"
    )
    seed = context.seed_protocol.root_seed
    assert seed is not None
    with pytest.raises(ValueError) as caught:
        human_games.check_replay(game, loaded, seed=seed, player_class="priest")
    # Only the missing person fails: map, end, seed and completion all match.
    assert str(caught.value) == (
        "Game 1's replay does not have the person in slots (0, 1, 2, 3, 4)."
    )
    settings = {
        "opponent": {"name": "TDM-Beta", "method": "tdm-beta"},
        "partner": {"name": "TDM-Beta", "method": "tdm-beta"},
        "player_class": "priest",
        "seed": seed,
    }
    record = human_games.record_game(game, [(replay, loaded)], settings=settings)
    full = marl_bgs.replay_metrics(loaded, metrics="full")
    assert record["outcome"] in ("win", "draw", "loss")
    assert record["team_a_kills"] == int(full["team_a_kills"])
    assert record["controllers"][:5] == ["tdm-alpha"] * 5
    assert record["classes"][:5] == ["Mage", "Warrior", "Hunter", "Rogue", "Priest"]
    assert (replay.parent / "game.json").is_file()
