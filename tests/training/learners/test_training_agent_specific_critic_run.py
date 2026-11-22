"""Check a short public training run with the agent-specific MAPPO critic.

The run saves PPOConfig.agent_specific_state in its settings, and resumed from
its step-8 checkpoint (where the new critic weight and its optimizer moments
are already trained) it ends with the same actor weights as without a resume.
This test establishes no learning or speed claim.
"""

from pathlib import Path

import jax
import numpy as np

from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.training import TrainConfig, checkpoints, load_system, train


def test_a_short_run_saves_the_choice_and_resumes_exactly(tmp_path: Path) -> None:
    config = TrainConfig(
        method="mappo",
        num_envs=4,
        total_env_steps=16,
        keep_past=0,
        ppo=PPOConfig(rollout_length=2, epochs=1, agent_specific_state=True),
        checkpoint_interval_updates=1,
        metrics="none",
        verbose=False,
    )
    first = train(config, output_dir=tmp_path / "run")
    details = checkpoints.read_checkpoint_details(first.final_checkpoint)
    assert details["metadata"]["config"]["ppo"]["agent_specific_state"] is True
    weights = jax.tree.leaves(load_system(first.final_actor).variables)
    middle = sorted((first.run_dir / "checkpoints").iterdir())[1]
    assert "_step_000000000008_" in middle.name
    resumed = train(config, resume_from=middle)
    assert resumed.completed_env_steps == first.completed_env_steps
    again = jax.tree.leaves(load_system(resumed.final_actor).variables)
    assert len(again) == len(weights)
    for before, after in zip(weights, again, strict=True):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
