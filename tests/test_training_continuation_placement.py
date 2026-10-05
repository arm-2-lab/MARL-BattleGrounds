"""Check the actual child boundary's placement for all six training methods.

Real small initial learners pass through _prepare_extension on CPU. Historical
candidate file lookup is replaced because this test has no saved run. No game,
optimizer update, checkpoint restore or GPU run is performed. Numerical values,
dtypes, weak types and resident buffers survive the final placement. A cheap
compiled state transition then reuses one signature across three calls.
"""

# pyright: reportPrivateUsage=false
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.baselines.pqn import PQNConfig
from marl_battlegrounds.baselines.qmix import QMIXConfig
from marl_battlegrounds.training import _selection_evidence, checkpoints, runner
from marl_battlegrounds.training.config import config_to_dict
from marl_battlegrounds.training.curriculum import make_training_schedule


def _facts(state: Any) -> list[tuple[Any, ...]]:  # noqa: ANN401
    result: list[tuple[Any, ...]] = []
    for value in jax.tree.leaves(state):
        data = (
            jax.random.key_data(value)
            if jnp.issubdtype(value.dtype, jax.dtypes.prng_key)
            else value
        )
        result.append(
            (
                value.shape,
                str(value.dtype),
                jax.typeof(value).weak_type,
                hashlib.sha256(np.asarray(data).tobytes()).hexdigest(),
            )
        )
    return result


@pytest.mark.parametrize(
    "method", ["mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"]
)
def test_real_child_boundary_keeps_one_placement_signature(
    method: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schedule = make_training_schedule(total_env_steps=128, num_envs=4, curriculum=False)
    common: dict[str, Any] = {
        "schedule": schedule,
        "seed": 719,
        "keep_past": 0,
        "history_capture_capacity": 0,
        "metrics": "none",
    }
    config_options: dict[str, Any] = {}
    if method == "qmix":
        from marl_battlegrounds.training.qmix_learner import init_qmix_learner

        settings = QMIXConfig(
            rollout_length=2,
            epochs=1,
            buffer_size=4,
            min_buffer_size=2,
            sample_batch_size=1,
            sample_sequence_length=2,
        )
        collection, initial = init_qmix_learner(**common, qmix=settings)
        config_options["qmix"] = settings
    elif method == "pqn_vdn":
        from marl_battlegrounds.training.pqn_learner import init_pqn_learner

        settings = PQNConfig(
            rollout_length=2, memory_window=1, epochs=1, num_minibatches=2
        )
        collection, initial = init_pqn_learner(**common, pqn=settings)
        config_options["pqn"] = settings
    else:
        from marl_battlegrounds.training.learner import init_learner

        settings = PPOConfig(rollout_length=2, epochs=1)
        collection, initial = init_learner(**common, ppo=settings, method=method)
        config_options["ppo"] = settings
    config = runner.TrainConfig(
        method=cast(Any, method),
        num_envs=4,
        total_env_steps=128,
        curriculum=False,
        keep_past=0,
        metrics="none",
        verbose=False,
        **config_options,
    )
    device = cast(Any, jax.devices("cpu")[0])
    initial = jax.device_put(initial, device)
    source = {
        "schema_version": 2,
        "package_version": "test",
        "source_tree_digest": "a" * 64,
    }
    details = {
        "checkpoint_id": "b" * 64,
        "counters": checkpoints._counters(initial, 4, method),
        "metadata": {
            "run_id": "parent",
            "source": source,
            "config": config_to_dict(config),
            "host_state": {},
        },
    }
    restored = checkpoints.RestoredCheckpoint(initial, details, tmp_path / "parent")

    def no_candidates(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        return []

    monkeypatch.setattr(
        _selection_evidence, "freeze_inherited_candidates", no_candidates
    )
    original_put = cast(Any, jax.device_put)
    placed: list[dict[str, Any]] = []

    def checked_put(
        value: Any,  # noqa: ANN401
        destination: Any = None,  # noqa: ANN401
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        before = _facts(value) if isinstance(value, type(initial)) else None
        result = original_put(value, destination, *args, **kwargs)
        if before is not None:
            assert _facts(result) == before
            assert destination == initial.carry.root_key.sharding
            assert all(
                leaf.committed and leaf.devices() == {device}
                for leaf in jax.tree.leaves(result)
            )
            reused = 0
            for old, new in zip(
                jax.tree.leaves(value), jax.tree.leaves(result), strict=True
            ):
                if (
                    old.committed
                    and old.size
                    and not jnp.issubdtype(old.dtype, jax.dtypes.prng_key)
                ):
                    assert old.unsafe_buffer_pointer() == new.unsafe_buffer_pointer()
                    reused += 1
            placed.append(
                {
                    "values_preserved": True,
                    "resident_buffers_reused": reused,
                    "fresh_leaf_count": sum(
                        not leaf.committed for leaf in jax.tree.leaves(value)
                    ),
                }
            )
        return result

    monkeypatch.setattr(jax, "device_put", checked_put)
    for branch_seed in (False, True):
        changes: dict[str, Any] = {"seed": 720} if branch_seed else {}
        if method == "pqn_vdn":
            changes["learning_rate"] = {"kind": "constant", "q_lr": 0.00025}
        child: Any = runner._prepare_extension(
            config,
            collection,
            restored,
            None,
            {"changes": changes, "additional_env_steps": 128},
            source,
        )[2]
        assert placed[-1]["fresh_leaf_count"] > 0
        assert all(
            leaf.committed and leaf.devices() == {device}
            for leaf in jax.tree.leaves(child)
        )

        @jax.jit
        def signature_step(state: Any) -> Any:  # noqa: ANN401
            history = state.carry.history._replace(
                capture_interval_rounds=state.carry.history.capture_interval_rounds
                + jnp.int32(1)
            )
            return state._replace(carry=state.carry._replace(history=history))

        compiled = cast(Any, signature_step)
        caches: list[int] = []
        for _ in range(3):
            child = compiled(child)
            jax.block_until_ready(child)
            caches.append(cast(int, compiled._cache_size()))
        assert caches == [1, 1, 1]
        print(
            json.dumps(
                {
                    "method": method,
                    "branch_seed": branch_seed,
                    "jit_cache_sizes": caches,
                    **placed[-1],
                },
                sort_keys=True,
            ),
            flush=True,
        )
