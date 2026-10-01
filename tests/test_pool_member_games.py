"""Check that pool members which act game by game run only on their own games.

A pool gives each game to one member. JAX Policy adapters and System apply
hooks declared lane-independent run through the compact route: their games are
gathered in fixed-size groups, applied and written back. On CPU the compact
route must give exactly the full-batch route's owned actions, policy IDs and
next memory, with several groups, a single group and no games at all. Learner
outputs keep their full-batch shapes. A pooled Policy is traced once, without
per-slot copies. Only Policies and declared, reset-free, leading-B members are
row-safe, and the built-in PPO, QMIX and PQN actor factories declare their hooks.
"""

# pyright: reportPrivateUsage=false
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines.ppo import make_ppo_system
from marl_battlegrounds.baselines.pqn import make_pqn_system
from marl_battlegrounds.baselines.qmix import make_qmix_system
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    SystemInput,
    SystemOutput,
    _adapter_template,
    _compact_jax_apply,
    _composed_jax_apply,
    _declare_lane_independent,
    _execution,
    _initial_memory,
    _row_safe,
    _slot_keys,
    shared_policy,
    system_inputs,
)
from marl_battlegrounds.policies.actor import ActorAction

type Tree = Any


def _inputs(size: int) -> SystemInput:
    env = marl_bgs.make("tdm", map_id=0, num_envs=size, metrics="none")
    obs, state = env.reset(jax.random.key(17))
    return system_inputs(obs, state, team=1)


def _policy_member(traces: list[int] | None = None) -> System:
    def counting(
        variables: Array, carry: Array, actor: Tree, mask: Tree, key: Array
    ) -> tuple[ActorAction, Array]:
        if traces is not None:
            traces.append(1)
        draw = jax.random.randint(key, (), 0, 7)
        move = ((carry + variables + draw) % 7).astype(jnp.int32)
        return ActorAction(move, jnp.int32(0), draw.astype(jnp.int32)), carry + 1

    return shared_policy(
        Policy("Counting", counting, jnp.int32(3), initial_carry=jnp.int32(0))
    )


def _lane_member(reset: bool = False) -> System:
    def lane_apply(
        variables: Array, memory: Array, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        def draw(key: Array) -> Array:
            return jax.random.randint(key, (5,), 0, 9)

        bits = jax.vmap(draw)(keys)
        move = ((memory + variables + bits) % 9).astype(jnp.int32)
        zero = jnp.zeros_like(move)
        return SystemOutput(
            ActorAction(move, zero, zero),
            memory + inputs.active_mask.astype(jnp.int32),
            learning_outputs={"value": move.astype(jnp.float32)},
        )

    def lane_init(variables: Array, inputs: SystemInput, keys: Array) -> Array:
        return jnp.zeros(inputs.active_mask.shape, jnp.int32)

    def lane_reset(old: Array, fresh: Array, mask: Array) -> Array:
        return jnp.where(mask[:, None], fresh, old)

    return System(
        "Lane",
        _declare_lane_independent(lane_apply),
        jnp.int32(2),
        init=lane_init,
        reset_memory=lane_reset if reset else None,
    )


def _both_routes(
    member: System, size: int, owned_games: np.ndarray, *, keep: bool
) -> tuple[SystemOutput, SystemOutput, SystemInput]:
    inputs = _inputs(size)
    execution = _execution(member)
    template = _adapter_template(member)
    keys = jax.random.split(jax.random.key(5), size)
    owned = inputs.active_mask & jnp.asarray(owned_games)[:, None]
    child = inputs._replace(
        controlled_mask=owned, valid=inputs.valid & jnp.any(owned, axis=1)
    )
    memory = _initial_memory(execution, member.variables, template, inputs, keys)

    def varied(leaf: Array) -> Array:
        return leaf + jnp.arange(leaf.size, dtype=leaf.dtype).reshape(leaf.shape)

    memory = jax.tree.map(varied, memory)
    arguments = (
        execution,
        member.variables,
        template,
        memory,
        child,
        keys,
        _slot_keys(keys),
    )
    full = cast(
        SystemOutput, _composed_jax_apply(*arguments, keep_learning_outputs=keep)
    )
    compact = cast(
        SystemOutput, _compact_jax_apply(*arguments, keep_learning_outputs=keep)
    )
    return full, compact, child


@pytest.mark.parametrize("member", ("policy", "lane"))
@pytest.mark.parametrize(
    "owned_games",
    (
        [True, False, True, True, False, True, True, False],
        [False, False, False, True, False, False, False, False],
        [False] * 8,
    ),
    ids=("several-groups", "one-group", "no-games"),
)
def test_compact_route_matches_full_batch_route(
    member: str, owned_games: list[bool]
) -> None:
    method = _policy_member() if member == "policy" else _lane_member()
    full, compact, child = _both_routes(
        method, 8, np.asarray(owned_games), keep=member == "lane"
    )
    retained = np.asarray(cast(Array, child.controlled_mask) & child.valid[:, None])
    for full_head, compact_head in zip(full.actions, compact.actions, strict=True):
        np.testing.assert_array_equal(
            np.asarray(compact_head)[retained], np.asarray(full_head)[retained]
        )
    np.testing.assert_array_equal(
        np.where(retained, np.asarray(compact.policy_ids), -9),
        np.where(retained, np.asarray(full.policy_ids), -9),
    )
    for full_leaf, compact_leaf in zip(
        jax.tree.leaves(full.next_memory),
        jax.tree.leaves(compact.next_memory),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(compact_leaf), np.asarray(full_leaf))
    if member == "lane":
        games = np.asarray(child.valid)
        np.testing.assert_array_equal(
            np.asarray(compact.learning_outputs["value"])[games],
            np.asarray(full.learning_outputs["value"])[games],
        )
        assert not np.any(np.asarray(compact.learning_outputs["value"])[~games])
    else:
        assert compact.learning_outputs == full.learning_outputs == ()


def test_pooled_policy_is_traced_once_without_slot_copies() -> None:
    traces: list[int] = []
    member = _policy_member(traces)
    inputs = _inputs(4)
    execution = _execution(member)
    template = _adapter_template(member)
    keys = jax.random.split(jax.random.key(8), 4)
    memory = _initial_memory(execution, member.variables, template, inputs, keys)

    def trace() -> SystemOutput:
        return cast(
            SystemOutput,
            _compact_jax_apply(
                execution,
                member.variables,
                template,
                memory,
                inputs,
                keys,
                _slot_keys(keys),
                keep_learning_outputs=False,
            ),
        )

    jax.make_jaxpr(trace)()
    assert len(traces) == 1


def test_only_policies_and_declared_leading_batch_members_are_row_safe() -> None:
    inputs = _inputs(4)
    keys = jax.random.split(jax.random.key(9), 4)

    def row_safe(method: System, memory: Tree | None = None) -> bool:
        execution = _execution(method)
        if memory is None:
            memory = _initial_memory(
                execution, method.variables, _adapter_template(method), inputs, keys
            )
        return _row_safe(execution, memory, 4)

    def plain(
        variables: Array, memory: Array, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
        return SystemOutput(ActorAction(zero, zero, zero), memory)

    assert row_safe(_policy_member())
    assert row_safe(_lane_member())
    assert not row_safe(_lane_member(reset=True))
    assert not row_safe(_lane_member(), memory=(jnp.zeros((4, 5)), jnp.int32(0)))
    assert not row_safe(System("Plain", plain))
    host = Policy("Host", _policy_member()._policies[0].apply, execution="host")
    assert not row_safe(shared_policy(host), memory=())


def test_built_in_actor_factories_declare_lane_independent_hooks() -> None:
    systems = (
        make_ppo_system({}, method="mappo"),
        make_ppo_system({}, method="ff_mappo"),
        make_ppo_system({}, method="mappo", parameter_sharing="class"),
        make_qmix_system({}),
        make_pqn_system({}),  # pyright: ignore[reportArgumentType]
    )
    for system in systems:
        assert getattr(system.apply, "_marl_lane_independent", False)
