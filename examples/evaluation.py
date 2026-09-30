"""Evaluate fixed methods, run repeated validation and inspect a small tournament.

Install the hydra extra alongside the evaluation dependencies, then run::

    python examples/evaluation.py workflow=evaluate output_dir=runs/evaluation

The default compares ALPHA with BETA for 32 paired games. metrics=full and
save_replays=2 opt into richer offline inspection. workflow=validation uses one
explicit shared writer for two labelled passes. workflow=tournament compares
the same entrants. All paths use the existing public owners; printed locations
identify the saved run. Use --help or --cfg job; import run for plain Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


def run(
    *,
    workflow: str = "evaluate",
    episodes: int | None = None,
    num_envs: int = 32,
    metrics: Literal["none", "priority", "full"] = "priority",
    save_replays: int = 0,
    red_zone_depth: float = 5.0,
    output_dir: str | Path | None = None,
) -> None:
    """Run an evaluation, repeated validation or a small tournament.

    Parameters
    ----------
    workflow : str, default='evaluate'
        evaluate compares ALPHA with BETA; validation writes two labelled passes;
        tournament compares the two entrants.
    episodes : int | None, default=None
        Total games, or games per entrant pair for a tournament. None selects 32 for
        evaluation/validation and 100 for tournaments.
    num_envs : int, default=32
        Number of parallel games. This changes batching, not the declared game or
        transition budget.
    metrics : Literal['none', 'priority', 'full'], default='priority'
        none, priority or full. Priority keeps compact game measurements. Full
        collects the larger behavior tables.
    save_replays : int, default=0
        Number of first scheduled game IDs to record for Viewer inspection, from
        zero through the game budget. Zero skips replay recording.
    red_zone_depth : float or int, default=5.0
        Depth of the evaluation Red Zone in world units. This is a saved evaluation
        condition.
    output_dir : str | Path | None, default=None
        None keeps evaluation/tournament results in memory. Otherwise these
        workflows create a new run below this parent. Validation requires this exact
        shared-writer folder.

    Returns
    -------
    None
        Print results and saved locations. No scientific learning claim is made.

    Notes
    -----
    API errors propagate without an automatic retry. Python callers use current
    working-directory paths; Hydra preserves its original launch directory.
    """
    from marl_battlegrounds.training.config import validate_red_zone_depth

    if type(red_zone_depth) is int:
        try:
            red_zone_depth = float(red_zone_depth)
        except OverflowError as error:
            raise ValueError("red_zone_depth must be finite") from error
    red_zone_depth = validate_red_zone_depth(red_zone_depth)
    output_dir = None if output_dir is None else Path(output_dir)
    if workflow not in ("evaluate", "validation", "tournament"):
        raise ValueError("workflow must be evaluate, validation or tournament")
    if episodes is None:
        episodes = 100 if workflow == "tournament" else 32
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.tasks import list_tdm_maps

    if not 0 <= save_replays <= episodes:
        raise ValueError("save_replays must be between zero and episodes")
    replays = range(1, save_replays + 1)

    if workflow == "tournament":
        result = marl_bgs.run_tournament(
            ["tdm-alpha", "tdm-beta"],
            episodes_per_pair=episodes,
            seed=42,
            num_envs=num_envs,
            metrics=metrics,
            replay_episodes=replays,
            output_dir=output_dir,
            red_zone_depth=red_zone_depth,
        )
        for row in result.tournament_results:
            print(row)
        print("Run Directory:", result.run_dir or "Results Were Not Saved")
        return

    if workflow == "validation":
        if output_dir is None:
            raise ValueError("this shared-writer example requires output_dir")
        validation_maps = [
            item for item in list_tdm_maps() if item.split == "validation"
        ]
        with marl_bgs.RunWriter(output_dir) as writer:
            for training_episode in (1_000, 5_000):
                # A real trainer supplies its current frozen checkpoint here.
                # Pass identity separates repeated episode IDs in the same CSV.
                marl_bgs.evaluate(
                    "tdm-alpha",
                    "tdm-beta",
                    maps=validation_maps,
                    num_episodes=episodes,
                    seed=training_episode,
                    num_envs=num_envs,
                    metrics=metrics,
                    replay_episodes=replays,
                    writer=writer,
                    red_zone_depth=red_zone_depth,
                    phase="validation",
                    pass_id=f"after-{training_episode}-training-episodes",
                )
            writer.flush()
            print("Files:", writer.paths)
        return

    # For example, red_zone_depth 6.0 gives evaluate(..., red_zone_depth=6.0).
    result = marl_bgs.evaluate(
        "tdm-alpha",
        "tdm-beta",
        num_episodes=episodes,
        seed=42,
        num_envs=num_envs,
        metrics=metrics,
        replay_episodes=replays,
        output_dir=output_dir,
        red_zone_depth=red_zone_depth,
    )
    print("Completed episodes:", result.completed_episode_ids)
    print("Run Directory:", result.run_dir or "Results Were Not Saved")


def main() -> int:
    """Compose packaged YAML and call run; use --help or --cfg job to inspect."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("evaluation", run)


if __name__ == "__main__":
    raise SystemExit(main())
