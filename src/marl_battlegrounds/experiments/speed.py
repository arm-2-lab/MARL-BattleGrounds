"""Measure how fast the simulator and one learner run on this machine.

Use XLA_PYTHON_CLIENT_PREALLOCATE=false python -m
marl_battlegrounds.experiments.speed output_dir=/path/speed on the GPU you want
to time. Part 1 times the simulator alone, as JaxMARL times its environments:
for each batch size, canonical 5v5 games on one map play steps ticks of random
legal actions (Environment.sample_actions, step and reset_done inside one
jax.jit and lax.scan), with compilation timed separately. Part 2 trains a PPO
baseline in pure self-play with marl_battlegrounds.training.train and times its
warm learner updates from the run's own training_updates.jsonl. The stage writes
simulator.csv, training.json, speed.md and the training run folder (training/).
Set train_blocks=0 to skip part 2. Without XLA_PYTHON_CLIENT_PREALLOCATE=false,
JAX reserves most of the card at start, so memory readings show the reservation,
not real use.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
# isort: on

# A canonical 5v5 game step moves 10 agents.
_AGENTS = 10


def simulator_speed(
    num_envs: int, *, map_id: int, steps: int, repeats: int, seed: int
) -> dict[str, Any]:
    """Time random legal play of canonical 5v5 games, compilation excluded.

    Parameters
    ----------
    num_envs : int
        Parallel games. On a GPU the project times 32, 64, 128, 512 and 1024.
    map_id : int
        Map every game plays, for example 47 (a test map with 32 obstacles).
    steps : int
        Ticks per timed call. Finished games restart inside the loop, so every
        tick plays ``num_envs`` games.
    repeats : int
        Warm timed calls after the first call; the median is reported.
    seed : int
        Root of the random keys for the reset and the actions.

    Returns
    -------
    dict
        ``num_envs``, ``map_id``, ``steps``, ``first_call_seconds`` (compile
        plus one run), ``warm_median_seconds``, ``warm_spread_percent`` (the
        warm calls' range as a share of their median),
        ``game_steps_per_second`` (ticks times games, per second of the
        median warm call) and ``agent_steps_per_second`` (10 agents per game).

    Notes
    -----
    Each call is one compiled ``jax.lax.scan`` over ``steps`` ticks: sample
    random legal actions, step, then reset finished games. Every timing waits
    for the device with ``jax.block_until_ready``. Runs on JAX's default
    device; pin a GPU with ``CUDA_VISIBLE_DEVICES``.
    """
    import jax

    import marl_battlegrounds as marl_bgs

    env = marl_bgs.make("tdm", map_id=map_id, num_envs=num_envs)
    state: Any = env.reset(jax.random.key(seed))[1]

    def tick(carry: tuple[Any, Any], unused: None) -> tuple[tuple[Any, Any], None]:
        state, key = carry
        key, act_key, step_key, reset_key = jax.random.split(key, 4)
        actions = env.sample_actions(act_key, state)
        state = env.step(step_key, state, actions)[1]
        state = env.reset_done(reset_key, state)[1]
        return (state, key), None

    def play(state: Any, key: Any) -> Any:  # noqa: ANN401 - a pytree of arrays.
        (state, _), _ = jax.lax.scan(tick, (state, key), None, length=steps)
        return state

    compiled: Any = jax.jit(play)
    start = time.perf_counter()
    state = cast(Any, jax.block_until_ready(compiled(state, jax.random.key(seed + 1))))
    first = time.perf_counter() - start
    times: list[float] = []
    for repeat in range(repeats):
        key = jax.random.key(seed + 2 + repeat)
        start = time.perf_counter()
        state = cast(Any, jax.block_until_ready(compiled(state, key)))
        times.append(time.perf_counter() - start)
    median = statistics.median(times)
    return {
        "num_envs": num_envs,
        "map_id": map_id,
        "steps": steps,
        "first_call_seconds": round(first, 2),
        "warm_median_seconds": round(median, 4),
        "warm_spread_percent": round(100 * (max(times) - min(times)) / median, 2),
        "game_steps_per_second": round(num_envs * steps / median),
        "agent_steps_per_second": round(_AGENTS * num_envs * steps / median),
    }


def training_speed(
    run_dir: Path,
    *,
    method: str,
    num_envs: int,
    rollout_length: int,
    blocks: int,
    warmup_blocks: int,
) -> dict[str, Any]:
    """Train a PPO baseline in pure self-play and time its warm updates.

    Parameters
    ----------
    run_dir : pathlib.Path
        New, empty folder for the training run.
    method : str
        One of the four PPO baselines: ``"mappo"`` (recurrent MAPPO),
        ``"ippo"``, ``"ff_mappo"`` or ``"ff_ippo"``.
    num_envs : int
        Parallel games, for example 1024.
    rollout_length : int
        Ticks collected per game before each learner update, for example 32.
    blocks : int
        Learner updates to run; each adds ``num_envs * rollout_length`` game
        steps.
    warmup_blocks : int
        First updates left out of the warm median (the first one compiles).
        Must be smaller than ``blocks``.

    Returns
    -------
    dict
        ``method``, ``num_envs``, ``rollout_length``, ``blocks``,
        ``first_block_seconds``, ``warm_median_seconds`` and
        ``warm_min_max_seconds`` (collection plus update time per block, from
        the run's training_updates.jsonl), ``game_steps_per_second`` and
        ``agent_steps_per_second`` at the warm median, and from the run's
        status.json ``peak_jax_bytes`` (JAX memory in use at its peak),
        ``jax_pool_bytes`` (memory the allocator reserved) and
        ``peak_ram_bytes`` (the process's peak RAM, all work included).

    Raises
    ------
    ValueError
        ``warmup_blocks`` is not smaller than ``blocks``, or the trainer
        rejects the settings.

    Notes
    -----
    Pure self-play: every game is against the current weights, with no saved
    copies, no fixed opponents and no recovery saves during the timed blocks.
    The only validation runs after the last block, outside the timed blocks.
    Other settings are the method's defaults. Training writes ``run_dir``.
    """
    if not 0 <= warmup_blocks < blocks:
        raise ValueError("warmup_blocks must be at least 0 and below blocks")
    from marl_battlegrounds.baselines.config import PPOConfig
    from marl_battlegrounds.training import TrainConfig, train

    config = TrainConfig(
        method=cast(Any, method),  # TrainConfig checks the method name.
        num_envs=num_envs,
        total_env_steps=num_envs * rollout_length * blocks,
        ppo=PPOConfig(rollout_length=rollout_length),
        opponents={"self": 1.0},
        keep_past=0,
        validation_fractions=(1.0,),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
        checkpoint_interval_updates=10**8,
        verbose=False,
    )
    train(config, output_dir=run_dir)
    lines = (run_dir / "training_updates.jsonl").read_text(encoding="utf-8")
    seconds = [
        float(json.loads(line)["collection_update_seconds"])
        for line in lines.splitlines()
        if line.strip()
    ]
    warm = statistics.median(seconds[warmup_blocks:])
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    allocator: dict[str, Any] = status["memory"]["jax_allocator"] or {}
    return {
        "method": method,
        "num_envs": num_envs,
        "rollout_length": rollout_length,
        "blocks": len(seconds),
        "first_block_seconds": round(seconds[0], 2),
        "warm_median_seconds": round(warm, 4),
        "warm_min_max_seconds": [
            round(min(seconds[warmup_blocks:]), 4),
            round(max(seconds[warmup_blocks:]), 4),
        ],
        "game_steps_per_second": round(num_envs * rollout_length / warm),
        "agent_steps_per_second": round(_AGENTS * num_envs * rollout_length / warm),
        "peak_jax_bytes": allocator.get("peak_bytes_in_use"),
        "jax_pool_bytes": allocator.get("peak_pool_bytes"),
        "peak_ram_bytes": status["memory"]["process_peak_ram_bytes"],
    }


def write_summary(
    device: str,
    simulator: Sequence[dict[str, Any]],
    training: dict[str, Any] | None,
    path: Path,
) -> None:
    """Write speed.md: the device, the simulator table and the training line.

    Parameters
    ----------
    device : str
        Device name, for example ``"NVIDIA GeForce RTX 5090"``.
    simulator : sequence of dict
        Rows from ``simulator_speed``, in batch-size order.
    training : dict or None
        Result of ``training_speed``; None when part 2 was skipped.
    path : pathlib.Path
        Markdown file to create or replace.
    """

    def gigabytes(value: object) -> str:
        return "unknown" if value is None else f"{int(str(value)) / 1e9:.1f} GB"

    lines = [
        "# Speed",
        "",
        f"Device: {device}. Compilation is excluded from every rate.",
        "",
        "| Parallel Games | Game Steps per Second | Agent Steps per Second "
        "| First Call (Compile + Run) | Warm Spread |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {row['num_envs']:,} | {row['game_steps_per_second']:,} "
        f"| {row['agent_steps_per_second']:,} | {row['first_call_seconds']} s "
        f"| {row['warm_spread_percent']}% |"
        for row in simulator
    ]
    if training is not None:
        lines += [
            "",
            f"Training {training['method']} in pure self-play on "
            f"{training['num_envs']:,} games with {training['rollout_length']}-tick "
            f"rollouts: {training['game_steps_per_second']:,} game steps per second "
            f"at the median warm update ({training['warm_median_seconds']} s); the "
            "first update, with compilation, took "
            f"{training['first_block_seconds']} s. "
            f"Peak JAX memory in use {gigabytes(training['peak_jax_bytes'])} "
            f"(allocator pool {gigabytes(training['jax_pool_bytes'])}); peak RAM "
            f"{gigabytes(training['peak_ram_bytes'])}.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra.main(version_base="1.3", config_path="configs", config_name="speed")
def main(config: DictConfig) -> dict[str, Any]:
    """Run both timing parts and write the result files.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``map_id`` (default 47), ``batch_sizes`` (default 32, 64, 128, 512 and
        1024), ``steps`` (ticks per timed call, default 500), ``repeats`` (warm
        calls, default 5), ``seed`` (default 0), ``train_method`` (default
        ``"mappo"``), ``train_num_envs`` (default 1024),
        ``train_rollout_length`` (default 32), ``train_blocks`` (default 40;
        0 skips training), ``warmup_blocks`` (default 3) and ``output_dir``,
        which must not already hold a ``training`` folder. Relative paths use
        the folder the command was started from.

    Returns
    -------
    dict
        ``device``, ``simulator`` (the rows) and ``training`` (or None).

    Raises
    ------
    ValueError
        ``batch_sizes`` is empty.
    FileExistsError
        ``output_dir/training`` already exists and training is requested.
    """
    import jax

    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    output = Path(to_absolute_path(str(Path(settings["output_dir"]).expanduser())))
    if not settings["batch_sizes"]:
        raise ValueError("give at least one batch size")
    run_dir = output / "training"
    if settings["train_blocks"] and run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; choose a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    device = str(cast(Any, jax.devices()[0]).device_kind)
    simulator: list[dict[str, Any]] = []
    for num_envs in settings["batch_sizes"]:
        row = simulator_speed(
            int(num_envs),
            map_id=settings["map_id"],
            steps=settings["steps"],
            repeats=settings["repeats"],
            seed=settings["seed"],
        )
        print(json.dumps(row), flush=True)
        simulator.append(row)
    with (output / "simulator.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(simulator[0]))
        writer.writeheader()
        writer.writerows(simulator)
    training = None
    if settings["train_blocks"]:
        training = training_speed(
            run_dir,
            method=settings["train_method"],
            num_envs=settings["train_num_envs"],
            rollout_length=settings["train_rollout_length"],
            blocks=settings["train_blocks"],
            warmup_blocks=settings["warmup_blocks"],
        )
        print(json.dumps(training), flush=True)
    (output / "training.json").write_text(
        json.dumps(training, indent=1) + "\n", encoding="utf-8"
    )
    write_summary(device, simulator, training, output / "speed.md")
    print(output / "speed.md")
    return {"device": device, "simulator": simulator, "training": training}


if __name__ == "__main__":
    prepare_hydra()
    main()
