"""Show paired and fixed-spawn games, authored starts and saved-result analysis.

Install the hydra extra, then run::

    python examples/evaluate/evaluation_results.py workflow=paired
    python examples/evaluate/evaluation_results.py workflow=read run_dir=SAVED_RUN

The short default games are software checks. Other workflow values demonstrate
validation/selection, exact authored starts, saved recovery and tournaments.
read starts no games. Use --help or --cfg job for settings. Python callers can
use run or the individual lesson functions directly.
"""

from __future__ import annotations

from pathlib import Path


def validate(num_envs: int, max_steps: int, output_dir: Path | None = None) -> None:
    """Select one frozen illustrative checkpoint using held-out validation games.

    num_envs is a positive worker count; max_steps sets the short demonstration's
    horizon. Each checkpoint gets a distinct pass ID, identical evaluation seed
    and fresh memory. output_dir=None creates no files. An explicit directory
    saves the illustrative numerical checkpoints, then reloads the selected one
    for held-out evaluation. Print the selected checkpoint and Team A game score.
    No learner update is performed; these short games prove no learning.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import Array

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.policy_execution import Policy, PolicyTree
    from marl_battlegrounds.policies.input import ActorInput
    from marl_battlegrounds.policies.random_valid import random_policy
    from marl_battlegrounds.types import ActionMask, ActorAction

    def choose(
        variables: PolicyTree,
        memory: PolicyTree,
        actor: ActorInput,
        mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        """Sample combat legally and optionally prefer the legal idle movement.

        variables is a scalar float32 preference, memory is unchanged and actor
        and mask belong to this actor alone. Return one int32 ActorAction and
        memory. The illustrative threshold is fixed for each frozen checkpoint.
        """
        action = random_policy(actor.observation, mask, key)
        move = jnp.where((variables > 0.5) & mask.move_mask[0], 0, action.move)
        return action._replace(move=move.astype(jnp.int32)), memory

    training_key = jax.random.key(17)
    key_before = np.asarray(jax.random.key_data(training_key)).copy()
    checkpoints = {
        "sample-moves": np.asarray(0.0, np.float32),
        "prefer-idle": np.asarray(1.0, np.float32),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, parameter in checkpoints.items():
            np.save(output_dir / f"{name}.npy", parameter, allow_pickle=False)
    scores: dict[str, float] = {}
    for name, parameter in checkpoints.items():
        snapshot = marl_bgs.shared_policy(Policy(name, choose, variables=parameter))
        result = marl_bgs.evaluate(
            snapshot,
            "random",
            num_episodes=2,
            phase="validation",
            pass_id=name,
            seed=42,
            max_steps=max_steps,
            num_envs=num_envs,
        )
        scores[name] = float(
            np.asarray(
                result.table("episodes")["system_game_score"], dtype=np.float64
            ).mean()
        )
    np.testing.assert_array_equal(jax.random.key_data(training_key), key_before)
    selected = max(scores, key=lambda name: scores[name])
    print(
        "Selected Illustrative Checkpoint:", selected, "Game Score:", scores[selected]
    )
    if output_dir is not None:
        checkpoint_path = output_dir / f"{selected}.npy"
        restored_parameter = np.load(checkpoint_path, allow_pickle=False)
        np.testing.assert_array_equal(restored_parameter, checkpoints[selected])
        restored_system = marl_bgs.shared_policy(
            Policy(selected, choose, variables=restored_parameter)
        )
        tested = marl_bgs.evaluate(
            restored_system,
            "random",
            num_episodes=2,
            phase="evaluation",
            seed=42,
            max_steps=max_steps,
            num_envs=num_envs,
        )
        print("Loaded Checkpoint:", checkpoint_path)
        print("Held-Out Outcomes:", tested.table("episodes")["system_game_score"])
        np.testing.assert_array_equal(jax.random.key_data(training_key), key_before)


def exact(num_envs: int) -> None:
    """Run one authored episode, then a separately declared custom comparison.

    num_envs controls workers only. Both examples keep a packaged scenario's
    starting state and explicitly shorten its horizon to one remaining step.
    No counterpart is required for the first schedule. Neither creates files.
    Use scenario_evaluation.py for complete, unchanged packaged scenarios.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.tasks import load_tdm_scenario

    scenario = load_tdm_scenario(1)
    env_config = scenario.config._replace(
        max_steps=int(scenario.initial_state.step_count) + 1
    )
    initial = scenario.initial_state
    single = marl_bgs.EpisodeSpec(7, env_config, seed_id=3, initial_state=initial)
    result = marl_bgs.evaluate_episodes("random", "random", [single], num_envs=num_envs)
    print("Single Authored Episode:", result.table("episodes")["episode_id"])
    comparison = [
        marl_bgs.EpisodeSpec(
            i,
            env_config,
            seed_id=3,
            initial_state=initial,
            paired_comparison_key="two-exact-conditions",
        )
        for i in (8, 9)
    ]
    custom = marl_bgs.evaluate_episodes(
        "random", "random", comparison, num_envs=num_envs
    )
    print("Custom Comparison:", custom.metadata["spawn_balance"])


def saved(output_dir: Path, num_envs: int, max_steps: int) -> None:
    """Save none-mode outcomes, selected full/replay data and resume without games.

    output_dir is the new-run parent. Worker count and horizon control the small
    first call. The second call inherits its saved scientific conditions. An
    explicit conflicting seed is rejected. Print actual paths and bounded rows.
    """
    import marl_battlegrounds as marl_bgs

    method = marl_bgs.shared_policy(marl_bgs.policy("random"))
    result = marl_bgs.evaluate(
        method,
        "random",
        num_episodes=2,
        maps=[12],
        seed=42,
        num_envs=num_envs,
        max_steps=max_steps,
        metrics="none",
        full_metrics_episodes=[1],
        save_replays=1,
        output_dir=output_dir,
    )
    resumed = marl_bgs.evaluate(
        method,
        "random",
        num_episodes=2,
        resume_from=result.run_dir,
        num_envs=num_envs,
        chunk_size=1,
    )
    assert resumed.episodes == ()
    print("Run Directory:", resumed.run_dir)
    print("Replay Paths:", resumed.replay_paths)
    if resumed.replay_paths:
        import shlex
        import sys

        print(
            "Open Replay:",
            shlex.join(
                [
                    sys.executable,
                    "-m",
                    "marl_battlegrounds",
                    "replay",
                    str(resumed.replay_paths[0]),
                    "--no-open",
                ]
            ),
        )
    for rows in resumed.iter_table("episodes", rows=1):
        print("Saved Outcome:", rows)
    try:
        marl_bgs.evaluate(
            method, "random", num_episodes=2, resume_from=result.run_dir, seed=0
        )
    except ValueError as error:
        print("Expected Explicit Conflict:", error)
    else:
        raise AssertionError("conflicting seed was accepted")


def run(
    *,
    workflow: str = "paired",
    num_envs: int = 32,
    max_steps: int = 2,
    output_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> None:
    """Demonstrate saved results, exact starts, selection and result reading.

    Parameters
    ----------
    workflow : str, default='paired'
        paired, fixed, validation, authored, save, tournament or read. Each uses the
        existing result and evaluation APIs. tournament and read also print
        system_metrics: each System's metrics seen from its own side.
    num_envs : int, default=32
        Number of parallel games. This changes batching, not the declared game or
        transition budget.
    max_steps : int, default=2
        Horizon for generated example games. Authored starts keep their explicit
        state and budget. read starts no games.
    output_dir : str | Path | None, default=None
        Optional result parent. save requires it; validation also saves its
        illustrative checkpoints there. Authored starts do not use it.
    run_dir : str | Path | None, default=None
        Exact saved result folder required by read. Other workflows do not use it.

    Returns
    -------
    None
        Print results and saved locations. No scientific learning claim is made.

    Notes
    -----
    API errors propagate without an automatic retry. Python callers use current
    working-directory paths; Hydra preserves its original launch directory.
    """
    output_dir = None if output_dir is None else Path(output_dir)
    run_dir = None if run_dir is None else Path(run_dir)
    if workflow not in (
        "paired",
        "fixed",
        "validation",
        "authored",
        "save",
        "tournament",
        "read",
    ):
        raise ValueError("Unknown saved-results workflow")
    if workflow == "read":
        if run_dir is None:
            raise ValueError("read needs run_dir naming an existing run")
        from marl_battlegrounds import load_results

        result = load_results(run_dir)
        print("Status:", result.status)
        for rows in result.iter_table("episodes", rows=128):
            print(rows)
        # Every System's margin and kills from its own side; {} when disabled.
        print(
            "Each System As Team A:",
            result.system_metrics(
                "priority_metrics", columns=["score_difference", "team_a_kills"]
            ),
        )
        return
    if workflow == "validation":
        validate(num_envs, max_steps, output_dir)
        return
    if workflow == "authored":
        exact(num_envs)
        return
    if workflow == "save":
        if output_dir is None:
            raise ValueError("save needs output_dir")
        saved(output_dir, num_envs, max_steps)
        return
    import marl_battlegrounds as marl_bgs

    if workflow == "tournament":
        # Four games per pair of Systems: two spawn pairs, enough for intervals.
        result = marl_bgs.run_tournament(
            ["random", "tdm-alpha"],
            maps=[12],
            episodes_per_pair=4,
            num_envs=num_envs,
            max_steps=max_steps,
            output_dir=output_dir,
        )
        print("Rankings:", result.table("tournament_rankings"))
        print("Headlines:", result.table("tournament_headline_metrics"))
        # Each System seen as Team A, whichever side it played. These 2-tick
        # games are a software check: most values and interval widths are zero.
        field = result.system_metrics(
            "priority_metrics", columns=["score_difference", "team_a_kills"]
        )
        print("Each System As Team A:", field)
        return
    fixed = workflow == "fixed"
    result = marl_bgs.evaluate(
        marl_bgs.shared_policy(marl_bgs.policy("random")),
        "random",
        num_episodes=1 if fixed else 2,
        maps=[12],
        spawn_mode="swapped" if fixed else "paired",
        num_envs=num_envs,
        max_steps=max_steps,
        metrics="priority",
        output_dir=output_dir,
    )
    print("Status:", result.status)
    print("Outcomes:", result.table("episodes"))
    print("Spawn Coverage:", result.metadata["spawn_balance"])


def main() -> int:
    """Compose packaged YAML and call run; use --help or --cfg job to inspect."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("evaluation_results", run)


if __name__ == "__main__":
    raise SystemExit(main())
