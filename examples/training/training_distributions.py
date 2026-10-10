"""Collect sampled training episodes through the public collection API.

Install the hydra extra; add training only for mappo=true. Run
python examples/training/training_distributions.py steps=320. Defaults use 32 games,
Random on both teams, three active agents per team and all 42 training maps.
Each game has the ordinary 300-transition horizon, so 320 rounds reach resets.
No learner updates occur. output_dir enables ordinary saved priority records;
leave it null to keep game recording disabled. The developer reference in
scripts/dev/training_distribution_reference.py keeps the synthetic tape and
recording-rewind checks. For direct recording recovery, see
examples/your_method/recorded_rollout.py.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

type Tree = Any

if TYPE_CHECKING:
    from jax import Array

    import marl_battlegrounds as marl_bgs


def random_actor() -> marl_bgs.System:
    """Build a stateless native System by batching the public Random policy.

    No actions are chosen here. Each call gives every actor only its permitted
    input and mask. Independent keys cover five slots in each game; variables
    and memory are empty. This native form fits the training collection owner,
    whose evolving actor does not accept a Policy adapter marker.
    """
    import jax

    import marl_battlegrounds as marl_bgs

    policy = marl_bgs.policy("random")
    choose = jax.vmap(
        jax.vmap(policy.apply, in_axes=(None, None, 0, 0, 0)),
        in_axes=(None, None, 0, 0, 0),
    )

    def split_key(key: Array) -> Array:
        """Give the five actor slots independent keys within one game."""
        return jax.random.split(key, 5)

    def act(
        variables: Tree, memory: Tree, inputs: marl_bgs.SystemInput, keys: Array
    ) -> marl_bgs.SystemOutput:
        """Sample each actor separately from (B,5) inputs; retain empty memory."""
        actor_keys = jax.vmap(split_key)(keys)
        actions, _ = choose(
            variables, memory, inputs.actors, inputs.action_mask, actor_keys
        )
        return marl_bgs.SystemOutput(actions, memory)

    return marl_bgs.System("Random Collection Actor", act)


def run(
    *,
    steps: int = 320,
    output_dir: str | Path | None = None,
    mappo: bool = False,
    num_envs: int = 32,
    seed: int = 42,
) -> dict[str, object]:
    """Collect complete blocks and return actual public exposure counts.

    steps is a positive number of rounds per game; transitions equal steps times
    num_envs, a positive even batch. seed separates model and collection keys.
    mappo chooses an explicitly untrained recurrent MAPPO actor; Team B stays
    Random. output_dir optionally holds a new recording, not a learner checkpoint.
    No files are made when it is None. Errors propagate and writers always close.
    Return the small training_summary mapping, actor label, zero updates and the
    actual run_dir when saved. No trajectories are copied for the printed report.
    """
    if type(mappo) is not bool:
        raise TypeError("mappo must be true or false")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    import jax

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training

    actor = random_actor()
    if mappo:
        from marl_battlegrounds.baselines import ppo

        variables = ppo.initialize_ppo(
            jax.random.fold_in(jax.random.key(seed), 1)
        ).actor_params
        actor = ppo.make_recurrent_mappo_system(
            variables, name="Untrained Recurrent MAPPO"
        )
    schedule = training.make_training_schedule(
        total_env_steps=steps * num_envs,
        num_envs=num_envs,
        curriculum=[{"share": 1.0, "maps": list(range(42)), "team_size": 3}],
    )
    collection, carry = training.init_training_collection(
        actor,
        actor.variables,
        schedule=schedule,
        seed=seed,
        keep_past=0,
        recording=output_dir is not None,
        opponent_population={"Random": "random"},
        opponent_selection={"Random": 1.0},
    )
    destination = (
        marl_bgs.RunWriter(
            output_dir,
            phase="training",
            policies={"team_a": collection.actor, "team_b": collection.opponent},
            details={
                "training_collection": {
                    "seed": seed,
                    "actor_status": "Untrained; no updates",
                    "schedule": dict(schedule.rounding_report),
                    "content_binding": collection.binding.model_dump(mode="json"),
                }
            },
        )
        if output_dir is not None
        else nullcontext()
    )
    with destination as writer:
        for _ in range((steps + 127) // 128):
            carry, _ = training.collect_training_rollout(
                collection, carry, length=128, writer=writer
            )
        summary = training.training_summary(collection, carry)
        summary["actor"] = "Untrained MAPPO" if mappo else "Random"
        summary["actor_updates"] = 0
        if writer is not None:
            summary["run_dir"] = str(writer.run_dir)
    return summary


def main() -> int:
    """Compose settings and print collection counts, without a learning claim."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("training_distributions", run)


if __name__ == "__main__":
    raise SystemExit(main())
