"""Check that Capture the Flag runs train, record their mode and resume.

Tiny CPU runs of 4 games play Capture the Flag's own maps. Recurrent MAPPO,
QMIX and PQN-VDN each stop after an early checkpoint and resume to the end.
Every checkpoint records content binding version 4 with task "ctf", winning
score 3 and Red Zone depth 0.0, actor input schema 3 and, for the methods
that keep one, training-state schema 3 (973 features). A Team Deathmatch MAPPO
actor, saved at actor input schema 2 (the same layout), starts a Capture the
Flag run, which then records schema 3.
"""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.baselines.config import PPOConfig, PQNConfig, QMIXConfig
from marl_battlegrounds.training import checkpoints, runner

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


@pytest.mark.parametrize("method", ["mappo", "qmix", "pqn_vdn"])
def test_capture_the_flag_runs_record_their_mode_and_resume(
    method: str, tmp_path: Path
) -> None:
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="on_update failed"):
        runner.train(_config(method), output_dir=root, on_update=_stop_at(2))
    result = runner.train(resume_from=root)
    assert result.completed_env_steps == _SETTINGS[method][1]
    saved = _details(root)
    assert len(saved) >= 2
    for details in saved:
        binding = details["collection"]["content_binding"]
        assert (binding["schema_version"], binding["task"]) == (4, "ctf")
        assert (binding["score_thresholds"], binding["red_zone_depth"]) == ([3], 0.0)
        assert details["schemas"]["actor_input"] == 3
        assert details["schemas"].get("training_state", 3) == 3


def test_a_team_deathmatch_actor_starts_a_capture_the_flag_run(tmp_path: Path) -> None:
    team_deathmatch = runner.train(_config("mappo", "tdm"), output_dir=tmp_path / "tdm")
    actor = checkpoints.read_checkpoint_description(Path(team_deathmatch.final_actor))
    assert actor["schemas"]["actor_input"] == 2
    flags = replace(_config("mappo"), initial_actor=str(team_deathmatch.final_actor))
    runner.train(flags, output_dir=tmp_path / "ctf")
    assert {
        details["schemas"]["actor_input"] for details in _details(tmp_path / "ctf")
    } == {3}
