"""Check the current-action oracle's block in the actor and Q networks.

A network initialized with a block gains only one bias-free "handed" weight, on
its post-memory layer (the feedforward actor: its second torso layer); every
other weight starts exactly as it does without the block. The weight starts at
zero, so at initialization any block gives exactly the outputs of a zero block,
while its gradient is not zero, so it learns. A block of zeros gives exactly the
logits or values and memory of the same network without that weight. Once the
weight is not zero, a nonzero block changes the outputs and never the memory. A
network with the
weight refuses to run without a block, and one without it refuses a block, in
the shared and the grouped paths. The grouped path with five equal groups
matches the shared path.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

import flax.errors
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.baselines.ppo import (
    FeedForwardActor,
    RecurrentActor,
    TarMACActor,
)
from marl_battlegrounds.baselines.qmix import RecurrentQNetwork

T, E, F, K = 3, 2, 37, 220
FEATURES = jax.random.normal(jax.random.key(1), (T, E, 5, F))
RESETS = jnp.zeros((T, E, 5), bool).at[0].set(True)
VALID = jnp.ones((T, E, 5), bool)
SENDERS = jnp.ones((T, E, 5, 5), bool)
GROUPS = jnp.broadcast_to(jnp.arange(5, dtype=jnp.int32), (T, E, 5))
BLOCK = jax.random.uniform(jax.random.key(2), (T, E, 5, K))

type Tree = Any
Init = Callable[[Array | None], Tree]
Apply = Callable[[Tree, Array | None, Array | None], tuple[Array | None, Array]]


def _recurrent(module: nn.Module, width: int) -> tuple[Init, Apply]:
    carry = jnp.zeros((E, 5, width))

    def init(handed: Array | None) -> Tree:
        return module.init(
            jax.random.key(0), carry, FEATURES, RESETS, VALID, handed=handed
        )

    def apply(
        variables: Tree, handed: Array | None, groups: Array | None
    ) -> tuple[Array | None, Array]:
        return cast(
            tuple[Array, Array],
            module.apply(
                variables,
                carry,
                FEATURES,
                RESETS,
                VALID,
                actor_group=groups,
                handed=handed,
            ),
        )

    return init, apply


def _feedforward() -> tuple[Init, Apply]:
    module = FeedForwardActor()

    def init(handed: Array | None) -> Tree:
        return module.init(jax.random.key(0), FEATURES, handed=handed)

    def apply(
        variables: Tree, handed: Array | None, groups: Array | None
    ) -> tuple[Array | None, Array]:
        return None, cast(
            Array, module.apply(variables, FEATURES, actor_group=groups, handed=handed)
        )

    return init, apply


def _tarmac() -> tuple[Init, Apply]:
    module = TarMACActor(rounds=2)
    carry = jnp.zeros((E, 5, 128))

    def init(handed: Array | None) -> Tree:
        return module.init(
            jax.random.key(0), carry, FEATURES, RESETS, VALID, SENDERS, handed
        )

    def apply(
        variables: Tree, handed: Array | None, groups: Array | None
    ) -> tuple[Array | None, Array]:
        assert groups is None
        return cast(
            tuple[Array, Array],
            module.apply(variables, carry, FEATURES, RESETS, VALID, SENDERS, handed),
        )

    return init, apply


NETWORKS = {
    "recurrent_actor": (lambda: _recurrent(RecurrentActor(), 128), "post_torso"),
    "feedforward_actor": (_feedforward, "torso"),
    "tarmac_actor": (_tarmac, "post_torso"),
    "qmix_q_network": (lambda: _recurrent(RecurrentQNetwork(), 256), "post_torso"),
}
GROUPED = ("recurrent_actor", "feedforward_actor", "qmix_q_network")


def _paths(tree: Tree, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if not isinstance(tree, Mapping):
        return {prefix}
    items = cast(Mapping[str, Tree], tree)
    return {
        path for key, value in items.items() for path in _paths(value, (*prefix, key))
    }


def _without_handed(variables: Tree, owner: str) -> Tree:
    params = dict(variables["params"])
    params[owner] = {k: v for k, v in params[owner].items() if k != "handed"}
    return {"params": params}


def _trained(variables: Tree, owner: str) -> Tree:
    params = dict(variables["params"])
    kernel = params[owner]["handed"]["kernel"]
    weight = jax.random.normal(jax.random.key(3), kernel.shape) / K**0.5
    params[owner] = {**params[owner], "handed": {"kernel": weight}}
    return {"params": params}


def _stack_five(leaf: Array) -> Array:
    return jnp.stack([leaf] * 5)


def _five_groups(variables: Tree) -> Tree:
    return jax.tree.map(_stack_five, variables)


@pytest.mark.parametrize("name", NETWORKS)
def test_the_block_adds_only_one_bias_free_weight(name: str) -> None:
    build, owner = NETWORKS[name]
    init, _ = build()
    plain, oracle = init(None), init(BLOCK)
    added = _paths(oracle) - _paths(plain)
    assert added == {("params", owner, "handed", "kernel")}
    assert oracle["params"][owner]["handed"]["kernel"].shape[0] == K
    stripped = _without_handed(oracle, owner)
    assert jax.tree.structure(stripped) == jax.tree.structure(plain)
    for leaf, other in zip(
        jax.tree.leaves(plain), jax.tree.leaves(stripped), strict=True
    ):
        np.testing.assert_array_equal(leaf, other)


@pytest.mark.parametrize("name", NETWORKS)
def test_a_zero_block_adds_exactly_nothing(name: str) -> None:
    build, owner = NETWORKS[name]
    init, apply = build()
    oracle = init(BLOCK)
    zero_memory, zero_out = apply(oracle, jnp.zeros_like(BLOCK), None)
    memory, out = apply(_without_handed(oracle, owner), None, None)
    np.testing.assert_array_equal(zero_out, out)
    if memory is not None:
        np.testing.assert_array_equal(zero_memory, memory)


@pytest.mark.parametrize("name", NETWORKS)
def test_the_weight_starts_at_zero_and_still_learns(name: str) -> None:
    build, owner = NETWORKS[name]
    init, apply = build()
    oracle = init(BLOCK)
    np.testing.assert_array_equal(oracle["params"][owner]["handed"]["kernel"], 0.0)
    memory, out = apply(oracle, BLOCK, None)
    zero_memory, zero_out = apply(oracle, jnp.zeros_like(BLOCK), None)
    np.testing.assert_array_equal(out, zero_out)
    if memory is not None:
        np.testing.assert_array_equal(memory, zero_memory)

    def loss(kernel: Array) -> Array:
        params = dict(oracle["params"])
        params[owner] = {**params[owner], "handed": {"kernel": kernel}}
        return jnp.sum(apply({"params": params}, BLOCK, None)[1] ** 2)

    kernel = cast(Array, oracle["params"][owner]["handed"]["kernel"])
    gradient = cast(Array, jax.grad(loss)(kernel))
    assert np.any(np.asarray(gradient) != 0.0)


@pytest.mark.parametrize("name", NETWORKS)
def test_a_block_changes_outputs_never_memory(name: str) -> None:
    build, owner = NETWORKS[name]
    init, apply = build()
    oracle = _trained(init(BLOCK), owner)
    memory, out = apply(oracle, BLOCK, None)
    zero_memory, zero_out = apply(oracle, jnp.zeros_like(BLOCK), None)
    assert not np.allclose(out, zero_out)
    if memory is not None:
        np.testing.assert_array_equal(memory, zero_memory)


@pytest.mark.parametrize("name", NETWORKS)
def test_the_block_and_its_weight_come_together(name: str) -> None:
    build, owner = NETWORKS[name]
    init, apply = build()
    oracle = init(BLOCK)
    with pytest.raises(ValueError, match="handed actions"):
        apply(oracle, None, None)
    with pytest.raises(flax.errors.ScopeParamNotFoundError):
        apply(_without_handed(oracle, owner), BLOCK, None)
    if name in GROUPED:
        with pytest.raises(ValueError, match="handed actions"):
            apply(_five_groups(oracle), None, GROUPS)
        with pytest.raises(ValueError, match="no weight for handed actions"):
            apply(_five_groups(_without_handed(oracle, owner)), BLOCK, GROUPS)


@pytest.mark.parametrize("name", GROUPED)
def test_five_equal_groups_match_the_shared_network(name: str) -> None:
    build, owner = NETWORKS[name]
    init, apply = build()
    oracle = _trained(init(BLOCK), owner)
    memory, out = apply(oracle, BLOCK, None)
    group_memory, group_out = apply(_five_groups(oracle), BLOCK, GROUPS)
    np.testing.assert_allclose(group_out, out, atol=3e-6, rtol=3e-5)
    if memory is not None and group_memory is not None:
        np.testing.assert_allclose(group_memory, memory, atol=3e-6, rtol=3e-5)
