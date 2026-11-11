"""Check that a Capture the Flag run's validation passes read back and carry over.

A tiny CPU recurrent MAPPO run plays Capture the Flag and validates once, at
the end, against a Random panel. Its saved passes read back through
read_run_evidence, which checks that every exchanged game reverses each spawn
exchange field its record holds: the pads and, in Capture the Flag, the flag
homes. A continuation of that run (extend_training) inherits the parent's
validated candidates, so the parent's final actor is still a selection
candidate for the child. These runs make no learning or speed claim.
"""

import json
from pathlib import Path

from marl_battlegrounds.baselines.config import PPOConfig
from marl_battlegrounds.training import checkpoints, extend_training, train, validation
from marl_battlegrounds.training._selection_evidence import (
    read_inherited_candidates,
    read_run_evidence,
)
from marl_battlegrounds.training.runner import TrainConfig


def test_capture_the_flag_passes_read_back_and_carry_over(tmp_path: Path) -> None:
    panel = validation.create_panel(opponents=["random"], output_dir=tmp_path / "panel")
    parent = train(
        TrainConfig(
            method="mappo",
            task="ctf",
            num_envs=4,
            total_env_steps=64,
            keep_past=0,
            seed=811,
            ppo=PPOConfig(rollout_length=2, epochs=1),
            checkpoint_interval_updates=1,
            validation_panel=str(panel.path),
            validation_fractions=(1.0,),
            routine_seed_pairs=1,
            confirmation_seed_pairs=1,
            metrics="none",
            verbose=False,
        ),
        output_dir=tmp_path / "parent",
    )
    parent_id = checkpoints.read_checkpoint_description(parent.final_actor)["metadata"][
        "checkpoint_id"
    ]
    evidence = read_run_evidence(parent.run_dir)
    assert evidence["status"] == "complete"
    assert parent_id in evidence["actors"]

    pointer = json.loads((parent.run_dir / "latest_checkpoint.json").read_text())
    child = extend_training(
        parent.run_dir / pointer["relative_path"],
        additional_env_steps=8,
        output_dir=tmp_path / "child",
    )
    details = json.loads((child.run_dir / "run_details.json").read_text())
    inherited = read_inherited_candidates(
        details["continuation"]["inherited_candidates"],
        declaration=details["validation_declaration"],
    )
    assert parent_id in inherited["actors"]
