"""Train, load and inspect a small PPO run through the public API.

Install the training and hydra extras, then run with JAX_PLATFORMS=cpu::

    python examples/mappo_training.py output_dir=artifacts/mappo-example
    python examples/mappo_training.py method=ippo output_dir=artifacts/ippo-example
    python examples/mappo_training.py resume_from=CHECKPOINT evaluate=true

Defaults are recurrent MAPPO, four games and 32 real transitions. They check
software wiring, not learning. config accepts the existing JSON settings or an
inline mapping. The selected actor, or final actor when none is selected, is
loaded; saved-run reports go under analysis/. Optional evaluation plays two
games against Random. Training and evaluation keep their declared Red Zone
settings; this lesson does not change the simulator or claim competence.
Use --help or --cfg job for YAML settings. Import run for plain Python use.
"""

from __future__ import annotations

from pathlib import Path


def run(
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    config: str | Path | dict[str, object] | None = None,
    method: str | None = None,
    evaluate: bool = False,
) -> None:
    """Train a small PPO example, load its chosen actor and print saved reports.

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
    method : str | None, default=None
        One of mappo, ippo, ff_mappo or ff_ippo for a new tiny run. None selects
        mappo. Do not combine this shortcut with config or resume_from.
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
    if type(evaluate) is not bool:
        raise TypeError("evaluate must be true or false")
    output_dir = None if output_dir is None else Path(output_dir)
    resume_from = None if resume_from is None else Path(resume_from)
    if (output_dir is None) == (resume_from is None):
        raise ValueError("Choose exactly one output_dir or resume_from")
    source_config = config
    if method is not None and method not in ("mappo", "ippo", "ff_mappo", "ff_ippo"):
        raise ValueError("method must name one of the four PPO methods")
    if method is not None and (source_config is not None or resume_from is not None):
        raise ValueError("method cannot accompany config or resume_from")

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.config import PPOConfig
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
            method=method or "mappo",
            seed=42,
            num_envs=4,
            total_env_steps=32,
            ppo=PPOConfig(rollout_length=4, epochs=1),
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
            "random",
            num_episodes=2,
            maps=[42],
            seed=43,
            num_envs=2,
            phase="validation",
            output_dir=result.run_dir / "example_evaluation",
        )
    reports = training.analyze([result.run_dir], output_dir=result.run_dir / "analysis")
    print(f"Run: {result.run_dir}")
    print(f"Frozen Actor: {result.selected_actor or result.final_actor}")
    print(f"Summary: {reports['artifacts']['summary']}")


def main() -> int:
    """Compose packaged YAML and call run; use --help or --cfg job to inspect."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("mappo_training", run)


if __name__ == "__main__":
    raise SystemExit(main())
