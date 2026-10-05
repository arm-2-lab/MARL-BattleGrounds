"""Check that pool members which act game by game run only on their own games.

A pool gives each game to one member. When the pool controls whole teams and
learner outputs are dropped, JAX Policy adapters and System apply hooks
declared lane-independent take the compact route: their games are gathered
into groups of ceil(B / 16) (padding rows are never written back), run and
written back, and no games means no call. On CPU the compact route must give
exactly the full-batch route's owned actions, policy IDs and next memory with
one partly filled group, one full group, several groups, every game and no
games. A real pool takes the compact route and compiles a Policy member once;
the same pool inside a team, or with learner outputs kept, keeps the per-slot
route. Only Policies and declared,
reset-free, leading-B members are row-safe; a wrapper that copies the marker
is not. The built-in PPO, QMIX and PQN actor factories declare their hooks.
"""

# pyright: reportPrivateUsage=false
import functools
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
    _jax_apply,
    _row_safe,
    _slot_keys,
    policy,
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
    member: System, size: int, owned_games: np.ndarray
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
        SystemOutput, _composed_jax_apply(*arguments, keep_learning_outputs=False)
    )
    compact = cast(SystemOutput, _compact_jax_apply(*arguments))
    return full, compact, child


def _games(size: int, owned: tuple[int, ...]) -> np.ndarray:
    games = np.zeros(size, np.bool_)
    games[list(owned)] = True
    return games


@pytest.mark.parametrize("member", ("policy", "lane"))
@pytest.mark.parametrize(
    "size,owned",
    (
        (48, (3, 17)),
        (48, (0, 20, 47)),
        (48, (1, 2, 3, 4, 5, 40)),
        (8, tuple(range(8))),
        (8, ()),
    ),
    ids=(
        "one-group-with-padding",
        "one-full-group",
        "several-groups",
        "every-game",
        "no-games",
    ),
)
def test_compact_route_matches_full_batch_route(
    member: str, size: int, owned: tuple[int, ...]
) -> None:
    method = _policy_member() if member == "policy" else _lane_member()
    full, compact, child = _both_routes(method, size, _games(size, owned))
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
    assert compact.learning_outputs == full.learning_outputs == ()


def _policy_traces(*, inside_team: bool, keep: bool) -> int:
    traces: list[int] = []
    member = _policy_member(traces)
    mixture = marl_bgs.pool({member: 1.0, shared_policy(policy("random")): 1.0})
    system = (
        marl_bgs.team(mixture, "random", slots=[[0, 1, 2], [3, 4]])
        if inside_team
        else mixture
    )
    inputs = _inputs(48)
    keys = jax.random.split(jax.random.key(12), 48)
    execution = _execution(system)
    template = _adapter_template(system)
    memory = _initial_memory(execution, system.variables, template, inputs, keys)

    def trace() -> SystemOutput:
        return _jax_apply(
            execution,
            system.variables,
            template,
            memory,
            inputs,
            keys,
            keys,
            jnp.zeros(48, jnp.bool_),
            keep_learning_outputs=keep,
        )

    jax.make_jaxpr(trace)()
    return len(traces)


def test_whole_team_pool_with_dropped_outputs_takes_the_compact_route() -> None:
    # One whole-team copy in the group loop; the per-slot route traces a
    # whole-team copy and five column copies.
    assert _policy_traces(inside_team=False, keep=False) == 1
    assert _policy_traces(inside_team=True, keep=False) >= 6
    assert _policy_traces(inside_team=False, keep=True) >= 6


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


def test_wrappers_that_copy_the_marker_keep_the_full_batch() -> None:
    base = make_ppo_system({}, method="mappo")
    memory = jnp.zeros((4, 5, 128), jnp.float32)

    @functools.wraps(base.apply)
    def wrapped(
        variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        return base.apply(variables, memory, inputs, keys)

    assert getattr(wrapped, "_marl_lane_independent", None) is base.apply
    assert _row_safe(_execution(base), memory, 4)
    for apply in (wrapped, jax.jit(base.apply)):
        method = System("Wrapped", apply, init=base.init)
        assert not _row_safe(_execution(method), memory, 4)


def test_built_in_actor_factories_declare_lane_independent_hooks() -> None:
    systems = (
        make_ppo_system({}, method="mappo"),
        make_ppo_system({}, method="ff_mappo"),
        make_ppo_system({}, method="mappo", parameter_sharing="class"),
        make_qmix_system({}),
        make_pqn_system({}),  # pyright: ignore[reportArgumentType]
    )
    for system in systems:
        assert getattr(system.apply, "_marl_lane_independent", None) is system.apply
