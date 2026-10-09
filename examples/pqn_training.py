"""Train, load and inspect a small recurrent PQN-VDN run through the public API.

Install training and hydra, then use JAX_PLATFORMS=cpu::

    python examples/pqn_training.py output_dir=artifacts/pqn-example
    python examples/pqn_training.py output_dir=artifacts/pqn-shaped shaping=true

Four games and 176 real transitions cover the small lesson's warmup and updates.
No past copies are kept (keep_past=0), since a run this short has no room for them.
These budgets establish no learning claim. curriculum=true and shaping=true
apply only to the tiny fresh recipe; config and resume_from own their settings.
config accepts the existing JSON file or a plain mapping. evaluate=true adds
separate evaluation and tournament calls after loading the saved actor. Reports
print their locations. Use --help or --cfg job; import run for plain Python.
"""

from __future__ import annotations

from pathlib import Path


def run(
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    config: str | Path | dict[str, object] | None = None,
    curriculum: bool = False,
    shaping: bool = False,
    evaluate: bool = False,
) -> None:
    """Train a small recurrent PQN-VDN example and inspect its saved actor.

    Parameters
    ----------
    output_dir : str | Path | None, default=None
        Chosen output folder, relative to the current working directory. The lesson
        prints the actual saved locations.
    resume_from : str | Path | None, default=None
        Saved run or checkpoint to continue. Omit output_dir on exact resume. None
        starts new work.
    config : str | Path | dict[str, object] | None, default=None
        Training settings as a JSON path or a plain mapping. None uses this lesson's
        small software-check settings for a fresh run; resumed work inherits its
        saved settings.
    curriculum : bool, default=False
        Enable the built-in curriculum for the tiny fresh run. Cannot accompany
        config or resume_from.
    shaping : bool, default=False
        Enable the built-in reward shaping for the tiny fresh run. Cannot accompany
        config or resume_from.
    evaluate : bool, default=False
        Whether to play the lesson's separate small evaluation after loading the
        saved actor. False skips those games; saved training analysis still runs.

    Returns
    -------
    None
        Print results and saved locations. No scientific learning claim is made.

    Notes
    -----
    API errors propagate without an automatic retry. Python callers use current
    working-directory paths; Hydra preserves its original launch directory.
    """
    for flag, value in (
        ("curriculum", curriculum),
        ("shaping", shaping),
        ("evaluate", evaluate),
    ):
        if type(value) is not bool:
            raise TypeError(f"{flag} must be true or false")
    output_dir = None if output_dir is None else Path(output_dir)
    resume_from = None if resume_from is None else Path(resume_from)
    if (output_dir is None) == (resume_from is None):
        raise ValueError("Choose exactly one output_dir or resume_from")
    source_config = config
    if (curriculum or shaping) and (
        source_config is not None or resume_from is not None
    ):
        raise ValueError(
            "curriculum and shaping cannot accompany config or resume_from"
        )

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.config import PQNConfig
    from marl_battlegrounds.training.config import config_from_dict, read_config

    parsed_config = None
    if source_config is not None:
        parsed_config = (
            config_from_dict(source_config)
            if isinstance(source_config, dict)
            else read_config(source_config)
        )
    elif resume_from is None:
        parsed_config = training.TrainConfig(
            method="pqn_vdn",
            seed=42,
            keep_past=0,
            num_envs=4,
            total_env_steps=176,
            curriculum=curriculum,
            shaping=shaping,
            pqn=PQNConfig(
                rollout_length=4,
                memory_window=2,
                epochs=1,
                num_minibatches=2,
            ),
        )
    result = training.train(
        parsed_config,
        output_dir=output_dir,
        resume_from=resume_from,
    )
    actor = training.load_system(result.selected_actor or result.final_actor)
    if evaluate:
        marl_bgs.evaluate(
            actor,
            "tdm-alpha",
            num_episodes=2,
            maps=[42],
            seed=43,
            num_envs=2,
            phase="validation",
            output_dir=result.run_dir / "example_evaluation",
        )
        marl_bgs.run_tournament(
            (actor, "tdm-alpha"),
            maps=[47],
            episodes_per_pair=2,
            seed=44,
            num_envs=2,
            output_dir=result.run_dir / "example_tournament",
        )
    reports = training.analyze([result.run_dir], output_dir=result.run_dir / "analysis")
    print(f"Run: {result.run_dir}")
    print(f"Frozen Actor: {result.selected_actor or result.final_actor}")
    print(f"Optimizer Steps: {result.completed_updates}")
    print(f"Summary: {reports['artifacts']['summary']}")


def main() -> int:
    """Compose packaged YAML and call run; use --help or --cfg job to inspect."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("pqn_training", run)


if __name__ == "__main__":
    raise SystemExit(main())
