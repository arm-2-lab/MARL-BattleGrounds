"""Check saved-actor placement for all six built-in learner initializers.

Real small CPU learners start fresh, then start again from explicitly placed
actor weights. These checks cover the returned numerical state before any game
or optimizer update. Values, dtypes, weak types and resident buffers stay intact.
Ordinary fresh state keeps its existing uncommitted placement. A cheap JIT
call proves that the former mixed state needed two signatures and the repaired
state needs one. This is not a learner speed or learning-quality benchmark.
"""

# pyright: reportPrivateUsage=false
import json
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from tests.training.continuation.test_training_continuation_placement import _facts

from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.baselines.pqn import PQNConfig
from marl_battlegrounds.baselines.qmix import QMIXConfig
from marl_battlegrounds.training import _compilation
from marl_battlegrounds.training.curriculum import make_training_schedule

type Tree = Any


def _signatures(initial: Tree) -> list[int]:
    @jax.jit
    def step(state: Tree) -> Tree:
        return state._replace(completed_updates=state.completed_updates + jnp.int32(1))

    state: Tree = initial
    compiled = cast(Any, step)
    result: list[int] = []
    for _ in range(3):
        state = compiled(state)
        jax.block_until_ready(state)
        result.append(int(compiled._cache_size()))
    return result


@pytest.mark.parametrize(
    "method", ["mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"]
)
def test_saved_actor_has_one_placement_across_the_learner(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    common: dict[str, Any] = {
        "schedule": make_training_schedule(
            total_env_steps=128, num_envs=4, curriculum=False
        ),
        "seed": 719,
        "keep_past": 0,
        "history_capture_capacity": 0,
        "metrics": "none",
    }
    if method == "qmix":
        from marl_battlegrounds.training import qmix_learner as module

        initialize = module.init_qmix_learner
        common["qmix"] = QMIXConfig(
            rollout_length=2,
            epochs=1,
            buffer_size=4,
            min_buffer_size=2,
            sample_batch_size=1,
            sample_sequence_length=2,
        )
    elif method == "pqn_vdn":
        from marl_battlegrounds.training import pqn_learner as module

        initialize = module.init_pqn_learner
        common["pqn"] = PQNConfig(
            rollout_length=2, memory_window=1, epochs=1, num_minibatches=2
        )
    else:
        from marl_battlegrounds.training import learner as module

        initialize = module.init_learner
        common.update(ppo=PPOConfig(rollout_length=2, epochs=1), method=method)
    seen: list[tuple[Tree, Tree]] = []

    def capture(state: Tree, *, sharding: Tree) -> Tree:
        before = _facts(state)
        result = _compilation._place_training_state(state, sharding=sharding)
        assert _facts(result) == before
        for old, new in zip(
            jax.tree.leaves(state), jax.tree.leaves(result), strict=True
        ):
            if (
                getattr(old, "committed", False)
                and old.size
                and not jnp.issubdtype(old.dtype, jax.dtypes.prng_key)
            ):
                assert old.unsafe_buffer_pointer() == new.unsafe_buffer_pointer()
        seen.append((state, result))
        return result

    monkeypatch.setattr(module, "_place_training_state", capture)
    _, fresh = cast(Any, initialize)(**common)
    assert seen[-1][0] is fresh
    assert not any(getattr(leaf, "committed", False) for leaf in jax.tree.leaves(fresh))
    variables = fresh.carry.history.current_variables
    actor = (
        variables.params
        if method == "qmix"
        else variables.network
        if method == "pqn_vdn"
        else variables
    )
    loaded = jax.device_put(actor, fresh.carry.root_key.sharding)
    _, warm = cast(Any, initialize)(**common, initial_actor=loaded)
    before, after = seen[-1]
    flags = [getattr(leaf, "committed", False) for leaf in jax.tree.leaves(before)]
    assert any(flags) and not all(flags)
    assert after is warm
    assert all(getattr(leaf, "committed", False) for leaf in jax.tree.leaves(warm))
    old_counts, fixed_counts = _signatures(before), _signatures(warm)
    assert old_counts == [1, 2, 2]
    assert fixed_counts == [1, 1, 1]
    print(
        json.dumps(
            {
                "method": method,
                "old_cache_sizes": old_counts,
                "fixed_cache_sizes": fixed_counts,
                "fresh_state_unchanged": True,
                "values_preserved": True,
                "resident_buffers_reused": True,
            }
        ),
        flush=True,
    )
