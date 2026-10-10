"""Continue a full learner checkpoint into a new child run.

Run from the checkout with the training and hydra extras installed::

    python examples/training/training_continuation.py checkpoint=RUN/checkpoints/ID \
        output_dir=CHILD additional_env_steps=1048576

The matching CLI is python -m marl_battlegrounds extend-training
RUN/checkpoints/ID --output-dir CHILD --additional-env-steps 1048576.
An optional changes=FILE reads future rules from JSON;
it never changes the parent's files. Scientific qualification is separate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training


def complete_example(output_dir: str | Path) -> dict[str, object]:
    """Train, extend, resume, evaluate and warm-start through the public API.

    ``output_dir`` must be new or empty. This small example uses four CPU lanes
    and 16 transitions to prove the workflow, not learned skill. Set the device
    before calling; GPU examples must use the project's allowed batch sizes.
    Every child has its own folder. Return the parent, child, warm-start result
    and evaluation, including their saved paths and full resumable checkpoints.
    """
    from dataclasses import replace

    from marl_battlegrounds.baselines.config import PPOConfig

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Example output_dir must be new or empty")
    config = training.TrainConfig(
        keep_past=0,
        method="ff_ippo",
        num_envs=4,
        total_env_steps=16,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        metrics="none",
        verbose=False,
    )
    parent = training.train(config, output_dir=root / "parent")
    child = training.extend_training(
        parent.final_checkpoint,
        additional_env_steps=8,
        output_dir=root / "child",
        changes={"seed": 43, "ppo": {"entropy_coefficient": 0.02}},
    )
    resumed = training.train(resume_from=child.final_checkpoint)
    actor = training.load_system(resumed.final_actor)
    results = marl_bgs.evaluate(
        actor, "tdm-alpha", num_episodes=2, maps=[0], seed=7, num_envs=2
    )
    warm = training.train(
        replace(config, initial_actor=str(child.final_actor)),
        output_dir=root / "warm_start",
    )
    return {
        "parent": parent,
        "child": resumed,
        "warm_start": warm,
        "evaluation": results,
    }


def run(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    additional_env_steps: int,
    changes: dict[str, Any] | str | Path | None = None,
) -> None:
    """Extend a parent checkpoint into a new child, leaving the parent unchanged.

    checkpoint is a complete learner checkpoint. output_dir is a new child
    directory. additional_env_steps is the exact positive transition budget.
    changes is an optional future-rule mapping or JSON file; a panel path in
    that file is relative to the file. Inline paths use the current directory.
    The API checks supported changes and writes recovery state; print its actual
    paths. Exact resume and actor warm-start remain in complete_example above.
    """
    if isinstance(changes, (str, Path)):
        path = Path(changes).resolve()
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Continuation changes must be a JSON object")
        changes = cast(dict[str, Any], value)
        validation = changes.get("validation")
        if isinstance(validation, dict):
            validation = cast(dict[str, Any], validation)
            panel = validation.get("panel")
            if panel is not None and not isinstance(panel, str):
                raise ValueError("Continuation validation panel must be a path")
            if panel is not None and not Path(panel).is_absolute():
                validation["panel"] = str((path.parent / panel).resolve())
    result = training.extend_training(
        checkpoint,
        additional_env_steps=additional_env_steps,
        output_dir=output_dir,
        changes=changes,
    )
    print(f"Child Run: {result.run_dir}")
    print(f"Selected Actor: {result.selected_actor}")
    print(f"Final Actor: {result.final_actor}")
    print(f"Final Checkpoint: {result.final_checkpoint}")


def main() -> int:
    """Compose packaged YAML, call the Python lesson and return its exit status."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("training_continuation", run)


if __name__ == "__main__":
    raise SystemExit(main())
