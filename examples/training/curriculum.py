"""Train an explicit map and roster curriculum, then load and evaluate its actor.

Install training and hydra, then run a small CPU software check::

    python examples/training/curriculum.py output_dir=artifacts/curriculum

The Python run function builds real CurriculumStage values through the public
API. Requested stage lengths and observed episode exposure stay separate.
Default four-game, 32-transition training establishes workflow behavior only.
Use --help or --cfg job to inspect settings; import run to keep using Python.
"""

from __future__ import annotations

import json
from pathlib import Path


def run(
    *, output_dir: str | Path, num_envs: int = 4, total_env_steps: int = 32
) -> None:
    """Train the declared roster/map curriculum, then load and evaluate its actor.

    Parameters
    ----------
    output_dir : str | Path
        Exact new training-run folder. Analysis and final evaluation live below that
        run.
    num_envs : int, default=4
        Number of parallel games. This changes batching, not the declared game or
        transition budget.
    total_env_steps : int, default=32
        Total real training transitions across all games. Must form whole learner
        rounds for the configured batch.

    Returns
    -------
    None
        Print results and saved locations. No scientific learning claim is made.

    Notes
    -----
    API errors propagate without an automatic retry. Python callers use current
    working-directory paths; Hydra preserves its original launch directory.
    """
    output_dir = Path(output_dir)

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.config import PPOConfig

    config = training.TrainConfig(
        num_envs=num_envs,
        total_env_steps=total_env_steps,
        seed=42,
        keep_past=0,
        curriculum=[
            {
                "share": 0.5,
                "maps": [0],
                "rosters": {"system": ["mage", "mage"], "opponent": ["warrior"]},
                "score_threshold": 5,
            },
            {
                "share": 0.5,
                "maps": [1, 2, 3],
                "rosters": {
                    "system": ["mage", "mage", "priest"],
                    "opponent": ["warrior", "hunter"],
                },
                "score_threshold": 20,
            },
        ],
        ppo=PPOConfig(rollout_length=4, epochs=1),
    )
    result = training.train(config, output_dir=output_dir)
    actor_path = result.selected_actor or result.final_actor
    actor = training.load_system(actor_path)
    evaluation_dir = result.run_dir / "curriculum_evaluation"
    evaluation_batch = min(num_envs, 32)
    evaluation = marl_bgs.evaluate(
        actor,
        "random",
        system_roster=["mage", "mage", "priest"],
        opponent_roster=["warrior", "hunter"],
        maps=[42],
        num_episodes=evaluation_batch,
        num_envs=evaluation_batch,
        score_threshold=20,
        max_steps=300,
        seed=43,
        phase="validation",
        output_dir=evaluation_dir,
    )
    exposure = json.loads((result.run_dir / "exposure.json").read_text())
    print(f"Actor: {actor_path}")
    print(f"Actual Transitions by Stage: {exposure['steps_by_episode_stage']}")
    print(f"Actual Game Starts by Stage: {exposure['starts_by_episode_stage']}")
    print(f"Evaluation: {evaluation.run_dir}")


def main() -> int:
    """Compose packaged YAML and call run; use --help or --cfg job to inspect."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("curriculum", run)


if __name__ == "__main__":
    raise SystemExit(main())
