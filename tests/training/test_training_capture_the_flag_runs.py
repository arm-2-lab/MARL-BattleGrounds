"""Check that Capture the Flag runs train, record their mode and resume.

Tiny CPU runs of 4 games play Capture the Flag's own maps. Recurrent MAPPO,
QMIX and PQN-VDN each stop after an early checkpoint and resume to the end.
Every checkpoint records content binding version 4 with task "ctf", winning
score 3 and Red Zone depth 0.0, actor input schema 2 (one number for the one
layout every mode shares) and, for the methods that keep one, training-state
schema 3 (973 features). Training continues across modes both ways for all
three methods: each resumed Capture the Flag run's final actor starts a Team
Deathmatch run, whose checkpoints record task "tdm" and training-state schema
2, and that run's final actor starts a new Capture the Flag run, which records
task "ctf" and training-state schema 3 again; every run records actor input
schema 2. For MAPPO the second start goes through a portable
model file, so the portable-file route is covered too. The resumed runs are
made once per method and shared by both tests. A MAPPO run's Random checks
play Capture the Flag (task "ctf", depth 0.0, winning score 3 in the saved
pass) and pass the resume checks, and its saved status records the mode and
its validation winning score.
"""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.baselines.config import PPOConfig, PQNConfig, QMIXConfig
from marl_battlegrounds.portable_models import export_model
from marl_battlegrounds.training import runner

_SETTINGS: dict[str, tuple[dict[str, Any], int]] = {
    "mappo": ({"ppo": PPOConfig(rollout_length=2, epochs=1)}, 64),
    "qmix": (
        {
            "qmix": QMIXConfig(
                rollout_length=8,
                buffer_size=64,
                min_buffer_size=32,
                sample_sequence_length=20,
                sample_batch_size=4,
                epochs=1,
                update_period=2,
            )
        },
        192,
    ),
    "pqn_vdn": (
        {
            "pqn": PQNConfig(
                rollout_length=4, memory_window=4, epochs=1, num_minibatches=2
            )
        },
        64,
    ),
}


def _config(method: str, task: str = "ctf") -> runner.TrainConfig:
    settings, steps = _SETTINGS[method]
    return runner.TrainConfig(
        method=method,  # pyright: ignore[reportArgumentType]
        task=task,  # pyright: ignore[reportArgumentType]
        num_envs=4,
        total_env_steps=steps,
        keep_past=0,
        checkpoint_interval_updates=1,
        validation_fractions=(1.0,),
        metrics="none",
        verbose=False,
        **settings,
    )


def _stop_at(update: int) -> Callable[[dict[str, Any]], None]:
    def stop(row: dict[str, Any]) -> None:
        if row["update_index"] == update:
            raise RuntimeError("stopped for the test")

    return stop


def _details(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads((path / "checkpoint_details.json").read_text())
        for path in sorted((root / "checkpoints").iterdir())
    ]


def _modes(root: Path) -> set[tuple[str, int, int]]:
    # A Team Deathmatch binding names no task, as before other modes existed;
    # PQN-VDN keeps no training state, so count it as its run's mode would.
    return {
        (
            task := d["collection"]["content_binding"].get("task", "tdm"),
            d["schemas"]["actor_input"],
            d["schemas"].get("training_state", 2 if task == "tdm" else 3),
        )
        for d in _details(root)
    }


@pytest.fixture(scope="module")
def resumed_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str], tuple[Path, runner.TrainResult]]:
    runs: dict[str, tuple[Path, runner.TrainResult]] = {}

    def run(method: str) -> tuple[Path, runner.TrainResult]:
        if method not in runs:
            root = tmp_path_factory.mktemp(method) / "run"
            with pytest.raises(RuntimeError, match="on_update failed"):
                runner.train(_config(method), output_dir=root, on_update=_stop_at(2))
            runs[method] = (root, runner.train(resume_from=root))
        return runs[method]

    return run


@pytest.mark.parametrize("method", ["mappo", "qmix", "pqn_vdn"])
def test_capture_the_flag_runs_record_their_mode_and_resume(
    method: str, resumed_run: Callable[[str], tuple[Path, runner.TrainResult]]
) -> None:
    root, result = resumed_run(method)
    assert result.completed_env_steps == _SETTINGS[method][1]
    saved = _details(root)
    assert len(saved) >= 2
    for details in saved:
        binding = details["collection"]["content_binding"]
        assert (binding["schema_version"], binding["task"]) == (4, "ctf")
        assert (binding["score_thresholds"], binding["red_zone_depth"]) == ([3], 0.0)
        assert details["schemas"]["actor_input"] == 2
        assert details["schemas"].get("training_state", 3) == 3


@pytest.mark.parametrize("method", ["mappo", "qmix", "pqn_vdn"])
def test_an_actor_continues_training_in_the_other_mode_both_ways(
    method: str,
    resumed_run: Callable[[str], tuple[Path, runner.TrainResult]],
    tmp_path: Path,
) -> None:
    _, capture_the_flag = resumed_run(method)
    back = replace(
        _config(method, "tdm"), initial_actor=str(capture_the_flag.final_actor)
    )
    team_deathmatch = runner.train(back, output_dir=tmp_path / "tdm")
    assert _modes(tmp_path / "tdm") == {("tdm", 2, 2)}
    source = str(team_deathmatch.final_actor)
    if method == "mappo":
        source = str(export_model(source, tmp_path / "tdm-actor.zip"))
    forward = replace(_config(method), initial_actor=source)
    runner.train(forward, output_dir=tmp_path / "ctf")
    assert _modes(tmp_path / "ctf") == {("ctf", 2, 3)}


def test_capture_the_flag_random_checks_play_the_mode_and_resume(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    config = replace(_config("mappo"), random_diagnostic_seed_pairs=1)
    with pytest.raises(RuntimeError, match="on_update failed"):
        runner.train(config, output_dir=root, on_update=_stop_at(2))
    runner.train(resume_from=root)
    records = json.loads((root / "random_diagnostics.json").read_text())
    assert [row["env_steps"] for row in records] == [0, 64]
    for row in records:
        assert (row["task"], row["red_zone_depth"], row["games"]) == ("ctf", 0.0, 10)
        details = json.loads(
            (Path(row["pass_paths"][0]) / "run_details.json").read_text()
        )
        (entry,) = details["passes"].values()
        options = entry["details"]["evaluation_contract"]["options"]
        assert (options["task"], options["score_threshold"]) == ("ctf", 3)
    status = json.loads((root / "status.json").read_text())
    assert (status["task"], status["validation_score_threshold"]) == ("ctf", 3)
