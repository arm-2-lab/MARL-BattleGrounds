"""Train a partial team beside frozen partners, then load and evaluate it.

Install training and hydra, then use JAX_PLATFORMS=cpu::

    python examples/training/partner_training.py output_dir=artifacts/partner-demo

The default learner owns physical slots 0 and 2. Frozen partners own the other
active slots; actor inputs and recurrent memory retain their information limits.
All six built-in learners use the same public training route. The repeated-class
roster and short default budget are software checks, not learned coordination.
Python run accepts live named partners. Hydra accepts ordinary reference mappings.
Use --help or --cfg job to inspect all settings and choose a new output folder.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jax import Array

    from marl_battlegrounds import Policy, System, SystemInput
    from marl_battlegrounds.tasks import AgentClassName
    from marl_battlegrounds.types import ActorAction


type Tree = Any


def idle(
    variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Tree]:
    """Return legal idle actions for (B,5) physical slots and keep empty memory.

    inputs contains permitted actor rows and masks. No observation or teammate
    private row is read. variables and keys are unused; memory is (). The team
    helper keeps only this member's controlled actions. Pure fixed-shape JAX.
    """
    import jax.numpy as jnp

    from marl_battlegrounds.types import ActorAction

    del variables, keys
    zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    return ActorAction(zero, zero, zero), memory


def load_idle() -> System:
    """Return a fresh, fixed custom System with no weights, memory or host service."""
    import marl_battlegrounds as marl_bgs

    return marl_bgs.System("Custom Idle Partner", idle)


def run(
    output_dir: Path,
    *,
    method: str = "ff_ippo",
    parameter_sharing: str = "all",
    partners: Mapping[str, System | Policy | str] | None = None,
    learner_slots: tuple[int, ...] = (0, 2),
    num_envs: int = 2,
    updates: int = 1,
    seed: int = 42,
    evaluation_max_steps: int = 2,
) -> dict[str, Any]:
    """Train a repeated-class roster, select on deployed teams, and evaluate them.

    output_dir must be new. method is mappo, ippo, ff_mappo, ff_ippo, qmix or
    pqn_vdn; parameter_sharing is all, class or none. partners maps distinct
    display aliases to ordinary methods; None chooses ALPHA and the custom idle
    System above. A repeating game-start order visits that declared pool.
    learner_slots names distinct physical slots 0..4 and must leave partner
    slots. This example fixes Team A to Mage, Warrior, Mage, Priest, Hunter;
    Team B is Warrior, Mage, Warrior, Hunter, Priest. Physical slots and
    information rights do not move. Class mode shares weights between the two
    Mages; none mode keeps their weights separate. Recurrent memory is separate
    for each actor in both modes.

    num_envs is the positive even training/evaluation batch (CPU default 2;
    choose an allowed batch such as 32 on GPU). updates is a positive count of
    tiny learning blocks, with two collected rounds per block. PQN first collects
    its required three initial rounds and uses a constant rate for this tiny run.
    Each block has one optimizer step. These small settings are a demonstration,
    not a recommended research recipe.
    seed controls training. Before training, declare a Random validation panel
    with roots 919/920. The runner validates every fixed partner on its normal
    development maps and 300-step horizon, then selects by native point margin.
    Final evaluation uses a separate shared seed 921, map 47 and both spawn
    ends, with the same explicit rosters. evaluation_max_steps is its game horizon,
    default 2 for a software check; use 300 for normal-length games.

    Return and save partner_results.json with actual steps, complete checkpoint,
    selected actor, selection record and each composed team's evaluator folder
    and exact IDs. An actor export holds the learner alone: team() restores the
    intended deployed
    team. Declared training membership does not by itself prove exposure to each
    member; saved training assignments provide that evidence. This function
    writes files and uses the public runner/evaluator, with no custom train loop.
    """
    output_dir = Path(output_dir)
    learner_slots = tuple(learner_slots)
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.config import PPOConfig, PQNConfig, QMIXConfig
    from marl_battlegrounds.baselines.methods import validate_training_method
    from marl_battlegrounds.training.validation import create_panel

    chosen = validate_training_method(method)
    if type(updates) is not int or updates < 1:
        raise ValueError("updates must be a positive integer")
    members = (
        dict(partners)
        if partners is not None
        else {"scripted": "tdm-alpha", "custom": load_idle()}
    )
    if not members:
        raise ValueError("Supply at least one frozen partner")
    rosters: dict[str, list[AgentClassName]] = {
        "system": ["mage", "warrior", "mage", "priest", "hunter"],
        "opponent": ["warrior", "mage", "warrior", "hunter", "priest"],
    }
    options: dict[str, Any] = {}
    if chosen == "qmix":
        options["qmix"] = QMIXConfig(
            rollout_length=2,
            buffer_size=4,
            min_buffer_size=2,
            sample_sequence_length=2,
            sample_batch_size=num_envs,
            epochs=1,
            parameter_sharing=parameter_sharing,
        )
    elif chosen == "pqn_vdn":
        options["pqn"] = PQNConfig(
            rollout_length=2,
            memory_window=1,
            epochs=1,
            num_minibatches=1,
            lr_linear_decay=False,
            parameter_sharing=parameter_sharing,
        )
    else:
        options["ppo"] = PPOConfig(
            rollout_length=2,
            epochs=1,
            minibatches=1,
            groups=1,
            parameter_sharing=parameter_sharing,
        )
    rounds = 2 * updates + (3 if chosen == "pqn_vdn" else 0)
    config = training.TrainConfig(
        method=chosen,
        num_envs=num_envs,
        total_env_steps=rounds * num_envs,
        seed=seed,
        curriculum=[{"share": 1.0, "maps": [0], "rosters": rosters}],
        learner_slots=learner_slots,
        partners=tuple(members),
        opponents={"self": 1.0},
        keep_past=0,
        validation_panel=str(output_dir / "panel" / "panel.json"),
        validation_fractions=(1.0,),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
        metrics="none",
        verbose=False,
        **options,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    create_panel(
        opponents=["random"],
        output_dir=output_dir / "panel",
        roots={"routine": 919, "confirmation": 920},
    )
    result = training.train(
        config, output_dir=output_dir / "training", partners=members
    )
    if result.selected_actor is None:
        raise RuntimeError("The trained actor has no completed validation selection")
    actor = training.load_system(result.selected_actor)
    other_slots = [slot for slot in range(5) if slot not in learner_slots]
    evaluations = []
    for index, (name, partner) in enumerate(members.items()):
        deployed = marl_bgs.team(actor, partner, slots=[learner_slots, other_slots])
        label = (
            re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:20].rstrip("_")
            or "partner"
        )
        games = marl_bgs.evaluate(
            deployed,
            "tdm-alpha",
            maps=[47],
            system_roster=rosters["system"],
            opponent_roster=rosters["opponent"],
            num_episodes=2,
            num_envs=num_envs,
            spawn_mode="paired",
            seed=921,
            max_steps=evaluation_max_steps,
            metrics="none",
            output_dir=output_dir / f"partner_{index:02d}_{label}",
        )
        evaluations.append(
            {
                "training_partner": name,
                "run_dir": str(games.run_dir),
                "spawn_balance": games.metadata["spawn_balance"],
            }
        )
    report = {
        "method": chosen,
        "parameter_sharing": parameter_sharing,
        "learner_slots": list(learner_slots),
        "rosters": rosters,
        "actor": str(result.selected_actor),
        "selection": str(result.run_dir / "selection.json"),
        "full_checkpoint": str(result.final_checkpoint),
        "completed_env_steps": result.completed_env_steps,
        "completed_updates": result.completed_updates,
        "evaluation_max_steps": evaluation_max_steps,
        "evaluations": evaluations,
    }
    (output_dir / "partner_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> int:
    """Compose packaged YAML, call the Python lesson and return its exit status."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("partner_training", run)


if __name__ == "__main__":
    raise SystemExit(main())
