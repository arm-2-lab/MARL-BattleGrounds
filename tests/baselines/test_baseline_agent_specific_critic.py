"""Check the agent-specific MAPPO critic (PPOConfig.agent_specific_state).

With the choice off, every centralized critic keeps its parameter tree and
values exactly; with it on, the critic gains only one bias-free "own_view"
weight on its first layer. IPPO methods refuse the choice, saved settings leave
out its default, and an agent-specific critic refuses to run without its rows'
own views, so it can never silently act as a shared-state critic. Each critic
row reads only its own actor's view, and one update with the choice on trains
the new weight.
"""

import dataclasses
import json
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
from jax import Array
from tests.baselines.test_baseline_ppo import (
    numerical_batch,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)

from marl_battlegrounds.baselines.config import (
    PPOConfig,
    check_ppo_method_settings,
    saved_settings,
    with_later_settings,
)
from marl_battlegrounds.baselines.ppo import (
    FeedForwardValueNet,
    PPOMinibatch,
    PPOTrainState,
    RecurrentValueNet,
    critic_values,
    initialize_ppo,
    update_minibatch,
)

type Tree = Any


def _leaves(tree: Tree) -> dict[str, np.ndarray]:
    pairs = cast(list[tuple[Any, Any]], jax.tree_util.tree_flatten_with_path(tree)[0])
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in pairs}


@pytest.mark.parametrize("method", ("mappo", "ff_mappo"))
def test_the_choice_adds_only_the_own_view_weight(method: str) -> None:
    key = jax.random.key(3)
    off = _leaves(initialize_ppo(key, PPOConfig(), method=method).critic_params)
    on = _leaves(
        initialize_ppo(
            key, PPOConfig(agent_specific_state=True), method=method
        ).critic_params
    )
    added = sorted(set(on) - set(off))
    assert len(added) == 1 and added[0].endswith("['own_view']['kernel']")
    assert on[added[0]].shape == (5165, 128)
    assert set(off) <= set(on)
    for name, value in off.items():
        np.testing.assert_array_equal(on[name], value)
    assert not any("own_view" in name for name in off)


@pytest.mark.parametrize("method", ("ippo", "ff_ippo"))
def test_ippo_methods_refuse_the_choice(method: str) -> None:
    config = PPOConfig(agent_specific_state=True)
    with pytest.raises(ValueError, match="local critic"):
        check_ppo_method_settings(method, config)
    with pytest.raises(ValueError, match="local critic"):
        initialize_ppo(jax.random.key(0), config, method=method)


def test_the_choice_must_be_a_bool() -> None:
    with pytest.raises(ValueError, match="agent_specific_state"):
        PPOConfig(agent_specific_state=cast(bool, 1))


def test_saved_settings_leave_the_default_out() -> None:
    assert "agent_specific_state" not in saved_settings(PPOConfig())
    assert saved_settings(PPOConfig(agent_specific_state=True))["agent_specific_state"]
    full = json.loads(json.dumps(dataclasses.asdict(PPOConfig())))
    assert with_later_settings(PPOConfig, saved_settings(PPOConfig())) == full


def test_each_recurrent_row_reads_only_its_own_view() -> None:
    net = RecurrentValueNet()
    steps, games = 3, 2
    physical = jax.random.normal(jax.random.key(1), (steps, games, 920))
    own = jax.random.normal(jax.random.key(2), (steps, games, 5, 5165))
    flags = jnp.zeros((steps, games, 5), jnp.bool_)
    valid = jnp.ones_like(flags)
    carry = jnp.zeros((games, 5, 128))
    params = net.init(jax.random.key(4), carry, physical, flags, valid, own)

    def apply(view: Array, own_view: Array) -> Array:
        return cast(
            tuple[Array, Array], net.apply(params, carry, view, flags, valid, own_view)
        )[1]

    values = cast(Any, jax.jit(apply))

    base = values(physical, own)
    for row in range(5):
        changed = np.asarray(
            jnp.abs(values(physical, own.at[:, :, row].add(1.0)) - base).max(
                axis=(0, 1)
            )
            > 0
        )
        assert changed.tolist() == [slot == row for slot in range(5)]


def test_each_feedforward_row_reads_only_its_own_view() -> None:
    net = FeedForwardValueNet()
    physical = jax.random.normal(jax.random.key(5), (4, 920))
    own = jax.random.normal(jax.random.key(6), (4, 5, 5165))
    params = net.init(jax.random.key(7), physical, own)
    base = cast(Array, net.apply(params, physical, own))
    assert base.shape == (4, 5)
    for row in range(5):
        moved = cast(Array, net.apply(params, physical, own.at[:, row].add(1.0)))
        changed = np.asarray(jnp.abs(moved - base).max(axis=0) > 0)
        assert changed.tolist() == [slot == row for slot in range(5)]


def test_an_agent_specific_critic_refuses_to_run_without_own_views() -> None:
    params = initialize_ppo(
        jax.random.key(9), PPOConfig(agent_specific_state=True), method="mappo"
    ).critic_params
    with pytest.raises(ValueError, match="own_view"):
        critic_values(
            params,
            jnp.zeros((2, 5, 128)),
            jnp.zeros((3, 2, 920)),
            jnp.zeros((3, 2), jnp.bool_),
            jnp.ones((3, 2), jnp.bool_),
        )


def test_an_update_trains_the_new_weight(
    numerical_batch: tuple[PPOTrainState, PPOMinibatch, PPOConfig],  # noqa: F811
) -> None:
    state, batch, config = numerical_batch
    on = dataclasses.replace(config, agent_specific_state=True)
    critic = RecurrentValueNet().init(
        jax.random.key(8),
        batch.critic_memory[0],
        batch.critic_features[0, :1, :, 0, :],
        batch.episode_start[0, :1],
        batch.valid[0, :1],
        batch.actor_features[0, :1],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(on.max_grad_norm),
        optax.adam(on.critic_lr, eps=on.adam_epsilon),
    )
    specific = state._replace(
        critic_params=critic, critic_opt_state=optimizer.init(critic)
    )
    compact = batch._replace(critic_features=batch.critic_features[..., 0, :])
    after, _ = update_minibatch(specific, compact, on)
    weight = "['params']['pre_torso']['own_view']['kernel']"
    assert not np.array_equal(
        _leaves(after.critic_params)[weight], _leaves(critic)[weight]
    )
