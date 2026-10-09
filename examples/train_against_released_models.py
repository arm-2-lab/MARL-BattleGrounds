"""Train a small PPO run against a ladder of released Season 0 saves.

Install training and hydra, then use JAX_PLATFORMS=cpu::

    python examples/train_against_released_models.py output_dir=artifacts/ladder

Three saves of the released QMIX model (at 25M and 100M training steps, and
its last save) join self-play as training opponents. Each name downloads one
portable model file once (a few MB) into the standard Hugging Face cache; after
that the run works offline with HF_HUB_OFFLINE=1. The saved run records every
opponent as Hugging Face's exact file address (``hf://...@<commit>/...``), so
resuming always loads the same files. Four games and 32 transitions check the
wiring, not learning. Use --help or --cfg job; import run for Python.
"""

from __future__ import annotations

from pathlib import Path

LADDER = {
    "qmix_25m": "MARL-BattleGrounds/tdm-season-0-qmix@25M",
    "qmix_100m": "MARL-BattleGrounds/tdm-season-0-qmix@100M",
    "qmix_final": "MARL-BattleGrounds/tdm-season-0-qmix@final",
}


def run(output_dir: str | Path, *, num_envs: int = 4, rollout_length: int = 8) -> Path:
    """Train feedforward IPPO briefly against self-play and three released saves.

    Parameters
    ----------
    output_dir : str or Path
        A new folder for the run, relative to the current working directory.
    num_envs : int, default=4
        Parallel games. Use an even number; GPU runs use 32 or more.
    rollout_length : int, default=8
        Decisions per update. The run makes one update.

    Returns
    -------
    Path
        The saved run's final actor folder.

    Raises
    ------
    ValueError
        The settings are invalid or a released name cannot be resolved.
    OSError
        The download or the output folder fails.

    Notes
    -----
    Downloads use huggingface_hub's cache (``HF_HOME``); nothing else is
    written outside ``output_dir``. Opponent weights are 25% each for self-play
    and the three saves.
    """
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.config import PPOConfig

    config = training.TrainConfig(
        method="ff_ippo",
        num_envs=num_envs,
        total_env_steps=num_envs * rollout_length,
        seed=0,
        opponents={"self": 0.25, **dict.fromkeys(LADDER, 0.25)},
        keep_past=0,
        metrics="none",
        verbose=False,
        ppo=PPOConfig(rollout_length=rollout_length, epochs=1, minibatches=1, groups=1),
    )
    result = training.train(config, output_dir=output_dir, opponents=LADDER)
    print(f"Saved run: {result.run_dir}")
    print(f"Final actor: {result.final_actor}")
    return Path(result.final_actor)


def main() -> int:
    """Compose packaged YAML, call the Python lesson and return its exit status."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("train_against_released_models", run)


if __name__ == "__main__":
    raise SystemExit(main())
