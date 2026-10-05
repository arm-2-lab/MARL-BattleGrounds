"""Write six untrained actor folders for the Battle Client examples.

Install MARL-BattleGrounds with its training extra, then run this copied script:
``JAX_PLATFORMS=cpu python prepare_battle_client.py --output /new/actor-folder``.
The script uses public APIs and works outside the checkout. It initializes real
random network weights, exports one actor per native method, and writes
preparation.json with paths, identities and setup costs. It never collects
experience or updates weights. These fixtures prove loading, not playing skill.
The existing initializers also allocate training-only arrays during setup; only
actor arrays are exported. No complete learner checkpoint is claimed.
"""

import argparse
import json
import resource
from pathlib import Path
from time import perf_counter
from typing import Any

import jax

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines.ppo import PPOConfig, initialize_ppo
from marl_battlegrounds.baselines.pqn import PQNConfig, initialize_pqn
from marl_battlegrounds.baselines.qmix import QMIXConfig, initialize_qmix
from marl_battlegrounds.training import prepare_training_content
from marl_battlegrounds.training.checkpoints import artifact_identity, export_system

METHODS = ("mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn")


def prepare(output: Path, seed: int = 17) -> dict[str, Any]:
    """Initialize and export all six methods without any training.

    Parameters
    ----------
    output : Path
        New parent directory. Existing paths are refused before setup. Each
        method gets a child folder with its method name. preparation.json keeps
        the exact paths, saved identities and measured wall-clock seconds.
    seed : int, default=17
        Nonnegative root seed. Method number n uses seed+n, so each method has
        independently initialized weights. All use shared parameters, input
        scale 1.0 and the left spawn frame.

    Returns
    -------
    dict[str, Any]
        The same finite JSON report written to preparation.json. Costs include
        network initialization and export, not training or inference speed.

    Raises
    ------
    ValueError
        seed is not a nonnegative integer.
    FileExistsError
        output already exists. Choose a new output path.
    OSError
        Output cannot be written. A failed run may leave a partial new folder.

    Notes
    -----
    Host-only orchestration uses the JAX device selected by JAX_PLATFORMS.
    Use CPU for this small fixture task. The content binding names the actual
    packaged training content; capture_origin records initialization at update
    zero without inventing an originating checkpoint. No network calls occur.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    content = prepare_training_content()
    rows: list[dict[str, Any]] = []
    for index, method in enumerate(METHODS):
        method_seed = seed + index
        method_started = perf_counter()
        key = jax.random.key(method_seed)
        if method == "qmix":
            actor = initialize_qmix(key, QMIXConfig(spawn_frame="left")).online_q
        elif method == "pqn_vdn":
            actor = initialize_pqn(
                key, pqn=PQNConfig(spawn_frame="left"), planned_learning_blocks=1
            ).network
        else:
            actor = initialize_ppo(
                key, PPOConfig(spawn_frame="left"), method=method
            ).actor_params
        jax.block_until_ready(actor)
        initialized_seconds = perf_counter() - method_started
        metadata: dict[str, object] = {
            "run_id": f"untrained-battle-client-{method}-seed-{method_seed}",
            "seed": method_seed,
            "env_steps": 0,
            "optimizer_steps": 0,
            "purpose": "Untrained integration fixture; no playing skill claimed",
            "capture_origin": {
                "capture_id": index,
                "update_index": 0,
                "content_binding": content.binding.model_dump(mode="json"),
                "epsilon": 0.0 if method in ("qmix", "pqn_vdn") else None,
            },
        }
        path = export_system(
            actor,
            output / method,
            metadata=metadata,
            input_scale=1.0,
            spawn_frame="left",
            method=method,
            parameter_sharing="all",
        )
        row = {
            "method": method,
            "path": str(path),
            "seed": method_seed,
            "env_steps": 0,
            "optimizer_steps": 0,
            "parameter_sharing": "all",
            "identity": artifact_identity(path),
            "initialization_seconds": initialized_seconds,
            "initialization_and_export_seconds": perf_counter() - method_started,
            "bytes": sum(
                item.stat().st_size for item in path.rglob("*") if item.is_file()
            ),
        }
        rows.append(row)
        print(f"{method}: {path} (0 training steps)", flush=True)
        del actor
    report = {
        "purpose": "Untrained integration fixtures; no learning or skill evidence",
        "package_file": marl_bgs.__file__,
        "jax_platform": jax.default_backend(),
        "actors": rows,
        "total_seconds": perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Read output and seed options, then write the six untrained actor folders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    prepare(args.output, args.seed)


if __name__ == "__main__":
    main()
