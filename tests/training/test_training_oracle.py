"""Check the current-action oracle's handed rows against Core's own builder.

With one-hot probabilities (actual actions) and every actor chosen, the handed
rows equal Core's rows of last tick's actions exactly, with or without
visibility masks, so every observer's target remap is Core's. Rows of actors not
yet chosen, and each observer's own row, are zero. Rows from probabilities that
are not one-hot are the probability-weighted mix of the one-hot rows. The action
split follows the 198-way action codec, and the network's flat block
(baselines/inputs.py) keeps its documented order and width. Each tick's order
ranks a fixed opponent's slots first and is otherwise a permutation that
changes with the key; the chosen-before matrix never hands an observer its own,
an absent or a later slot's row, and the teammates-only arm drops the other
team. Choosing in turn hands each slot exactly the slots chosen before it, in
games against a fixed opponent and against a copy, with probabilities or with
sampled actions, and keeps each slot's result from its own turn. A fixed
opponent's route is its own marked step for a built-in method; a team of the
scripted bots hands its unchanged actions as certain one-hot rows, the same
whatever key it gets; any other fixed opponent is refused. Played through the
shared runner's pair call against a pool of a saved team and a bot team, an
oracle whose handed weight starts at zero gives exactly the runner's actions,
learning values and memory, tick after tick, in both arms; and each slot is
handed all five enemies (which choose first) and exactly the teammates chosen
before it, each once.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.helpers.evaluation_fixtures import evaluation_env_config

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines.actions import decode_actions, encode_actions
from marl_battlegrounds.baselines.inputs import (
    _HANDED_WIDTH,  # pyright: ignore[reportPrivateUsage]
    _encode_handed_rows,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.baselines.ppo import (
    DEFAULT_PPO_CONFIG,
    _ActorStep,  # pyright: ignore[reportPrivateUsage]
    initialize_ppo,
    make_ppo_system,
    oracle_step,
)
from marl_battlegrounds.core.env import (
    _build_ally_enemy_one_hot_action_tensors,  # pyright: ignore[reportPrivateUsage]
    _build_visibility_masked_previous_timestep_action_observation,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.core.types import EnvState, PreviousTimestepActionObservation
from marl_battlegrounds.environment import EnvironmentState
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    apply_systems,
    init_systems,
    policy,
    pool,
    shared_policy,
    system_inputs,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.training._oracle import (
    action_part_probabilities,
    choose_in_order,
    chosen_before,
    fixed_member_step,
    handed_rows,
    oracle_apply_systems,
    oracle_member,
    turn_order,
)

Parts = tuple[Array, Array, Array]
type Tree = Any
ALLY_OF, ENEMY_OF = _build_ally_enemy_one_hot_action_tensors(jnp.eye(10), 10)
NOT_SELF = ~jnp.eye(10, dtype=bool)


@pytest.fixture(scope="module")
def state() -> EnvState:
    env = marl_bgs.make("tdm", map_id=0, balance_spawn_locations=False)
    return env.reset(jax.random.key(0))[1].core_state


def _actions(seed: int) -> Parts:
    keys = jax.random.split(jax.random.key(seed), 3)
    return (
        jax.random.randint(keys[0], (10,), 0, 9),
        jax.random.randint(keys[1], (10,), 0, 11),
        jax.random.randint(keys[2], (10,), 0, 2),
    )


def _one_hot(actions: Parts) -> Parts:
    return (
        jax.nn.one_hot(actions[0], 9),
        jax.nn.one_hot(actions[1], 11),
        jax.nn.one_hot(actions[2], 2),
    )


def _core(
    state: EnvState, actions: Parts, visible: tuple[Array, Array]
) -> PreviousTimestepActionObservation:
    history = state._replace(
        previous_timestep_move_actions=actions[0],
        previous_timestep_select_target_actions=actions[1],
        previous_timestep_use_ultimate_actions=actions[2],
        has_previous_timestep_joint_action=jnp.asarray(True),
    )
    return _build_visibility_masked_previous_timestep_action_observation(
        history, *visible
    )


def _per_row(per_actor: Array) -> tuple[Array, Array]:
    """Spread an (observer, actor) array over Core's ally and enemy relation rows."""
    return (
        jnp.einsum("ora,oa->or", ALLY_OF, per_actor),
        jnp.einsum("ora,oa->or", ENEMY_OF, per_actor),
    )


@pytest.mark.parametrize("masked", [False, True])
def test_actual_actions_give_cores_rows(state: EnvState, masked: bool) -> None:
    actions = _actions(3)
    keys = jax.random.split(jax.random.key(7))
    visible = (
        (
            jax.random.bernoulli(keys[0], 0.6, (10, 5)),
            jax.random.bernoulli(keys[1], 0.6, (10, 5)),
        )
        if masked
        else (jnp.ones((10, 5), bool), jnp.ones((10, 5), bool))
    )
    rows = handed_rows(
        state,
        *_one_hot(actions),
        chosen=jnp.ones((10, 10), bool),
        visible=visible if masked else None,
    )
    for field, expected in zip(rows, _core(state, actions, visible), strict=True):
        np.testing.assert_array_equal(field, expected)


def test_rows_of_unchosen_actors_and_the_observer_are_zero(state: EnvState) -> None:
    actions = _actions(4)
    chosen = jax.random.bernoulli(jax.random.key(5), 0.5, (10, 10)) & NOT_SELF
    rows = handed_rows(state, *_one_hot(actions), chosen=chosen)
    full = _core(state, actions, (jnp.ones((10, 5), bool), jnp.ones((10, 5), bool)))
    kept = _per_row(chosen.astype(jnp.float32))
    for index, (field, expected) in enumerate(zip(rows, full, strict=True)):
        np.testing.assert_array_equal(field, expected * kept[index % 2][..., None])
    own = np.asarray(rows.ally_previous_timestep_move_actions_one_hot).sum(-1)
    assert (own[np.arange(10), np.arange(10) % 5] == 0).all()


def test_mixed_probabilities_mix_the_one_hot_rows(state: EnvState) -> None:
    first, second = _one_hot(_actions(8)), _one_hot(_actions(9))
    weight = jax.random.uniform(jax.random.key(10), (10,))
    mixed = tuple(
        weight[:, None] * a + (1 - weight[:, None]) * b
        for a, b in zip(first, second, strict=True)
    )
    rows = handed_rows(state, *mixed, chosen=NOT_SELF)
    left = handed_rows(state, *first, chosen=NOT_SELF)
    right = handed_rows(state, *second, chosen=NOT_SELF)
    row_weight = _per_row(jnp.broadcast_to(weight, (10, 10)))
    for index, (field, a, b) in enumerate(zip(rows, left, right, strict=True)):
        w = row_weight[index % 2][..., None]
        np.testing.assert_allclose(field, w * a + (1 - w) * b, atol=1e-6)


def test_the_action_split_follows_the_codec() -> None:
    indices = jnp.arange(198, dtype=jnp.int32)
    move, target, ultimate = action_part_probabilities(jax.nn.one_hot(indices, 198))
    decoded = decode_actions(indices)
    np.testing.assert_array_equal(move.argmax(-1), decoded.move)
    np.testing.assert_array_equal(target.argmax(-1), decoded.select_target)
    np.testing.assert_array_equal(ultimate.argmax(-1), decoded.use_ultimate)
    distribution = jax.nn.softmax(jax.random.normal(jax.random.key(1), (4, 198)))
    for part in action_part_probabilities(distribution):
        np.testing.assert_allclose(part.sum(-1), 1.0, atol=1e-6)


def test_the_block_keeps_its_order_and_width(state: EnvState) -> None:
    rows = handed_rows(state, *_one_hot(_actions(11)), chosen=NOT_SELF)
    block = _encode_handed_rows(rows)
    assert block.shape == (10, _HANDED_WIDTH)
    order = (
        rows.ally_previous_timestep_move_actions_one_hot,
        rows.ally_previous_timestep_select_target_actions_one_hot,
        rows.ally_previous_timestep_use_ultimate_actions_one_hot,
        rows.enemy_previous_timestep_move_actions_one_hot,
        rows.enemy_previous_timestep_select_target_actions_one_hot,
        rows.enemy_previous_timestep_use_ultimate_actions_one_hot,
    )
    start = 0
    for field in order:
        width = field.shape[-2] * field.shape[-1]
        np.testing.assert_array_equal(
            block[:, start : start + width], field.reshape(10, -1)
        )
        start += width
    assert start == _HANDED_WIDTH


def test_fixed_opponents_choose_first_and_orders_vary() -> None:
    fixed = jnp.arange(10) >= 5
    orders = {
        tuple(np.asarray(turn_order(jax.random.key(k), fixed))) for k in range(20)
    }
    for ranks in orders:
        assert sorted(ranks) == list(range(10))
        assert set(ranks[5:]) == set(range(5))
    assert len(orders) > 1
    open_orders = [
        np.asarray(turn_order(jax.random.key(k), jnp.zeros(10, bool)))
        for k in range(40)
    ]
    assert len({int(np.argmin(r)) for r in open_orders}) > 2


@pytest.mark.parametrize("teammates_only", [False, True])
def test_chosen_before_hands_only_earlier_present_slots(teammates_only: bool) -> None:
    ranks = turn_order(jax.random.key(3), jnp.zeros(10, bool))
    present = jnp.ones(10, bool).at[2].set(False)
    chosen = np.asarray(chosen_before(ranks, present, teammates_only=teammates_only))
    r = np.asarray(ranks)
    assert not chosen.diagonal().any()
    assert not chosen[:, 2].any()
    for o in range(10):
        for a in range(10):
            same_team = o // 5 == a // 5
            expected = r[a] < r[o] and a != 2 and (same_team or not teammates_only)
            assert chosen[o, a] == expected


def _counting_step(rows: PreviousTimestepActionObservation) -> _ActorStep:
    """Choose index = how many other agents' rows this slot was handed."""
    count = rows.ally_previous_timestep_move_actions_one_hot.sum(
        (-2, -1)
    ) + rows.enemy_previous_timestep_move_actions_one_hot.sum((-2, -1))
    index = jnp.round(count).astype(jnp.int32)
    return _ActorStep(
        index,
        jnp.zeros(index.shape),
        jnp.zeros(index.shape),
        jax.nn.one_hot(index, 198),
    )


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("teammates_only", [False, True])
def test_choosing_in_turn_hands_each_slot_the_earlier_slots(
    teammates_only: bool, sampled: bool
) -> None:
    env = marl_bgs.make("tdm", map_id=0, num_envs=4)
    states = env.reset(jax.random.key(0))[1].core_state
    fixed = jnp.array([True, True, False, False])[:, None] & (jnp.arange(10) >= 5)
    keys = jax.random.split(jax.random.key(4), 4)
    ranks = jax.vmap(turn_order)(keys, fixed)
    present = jnp.ones((4, 10), bool).at[1, 3].set(False)
    choices = choose_in_order(
        states,
        ranks,
        present,
        lambda rows_a, rows_b: (_counting_step(rows_a), _counting_step(rows_b)),
        teammates_only=teammates_only,
        hand_sampled_actions=sampled,
    )
    expected = jax.vmap(
        functools.partial(chosen_before, teammates_only=teammates_only)
    )(ranks, present).sum(-1)
    np.testing.assert_array_equal(choices.indices, expected)
    np.testing.assert_array_equal(choices.probabilities.argmax(-1), expected)
    fixed_ranks = np.asarray(ranks)[:2, 5:]
    assert set(fixed_ranks.ravel()) == set(range(5))


@pytest.fixture(scope="module")
def opening() -> tuple[Observations, EnvironmentState]:
    env = marl_bgs.make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(5, 5), max_steps=12),
        num_envs=4,
        metrics="none",
    )
    return env.reset(jax.random.key(10))


def _lane_keys(seed: int) -> Array:
    return jax.random.split(jax.random.key(seed), 4)


@pytest.mark.parametrize("name", ["tdm-alpha", "tdm-beta", "tdm-gamma"])
def test_bots_hand_their_actions_as_certain_rows(
    opening: tuple[Observations, EnvironmentState], name: str
) -> None:
    observations, game = opening
    bots = shared_policy(policy(name))
    memory = init_systems(bots, bots, observations, game, jax.random.key(5))
    inputs = system_inputs(observations, game, team=1)
    step = fixed_member_step(bots)
    first = step(bots.variables, memory.team_b, inputs, _lane_keys(1))
    second = step(bots.variables, memory.team_b, inputs, _lane_keys(2))
    action, _, _ = apply_systems(
        bots, bots, memory, observations, game, jax.random.key(6)
    )
    team_b_heads = ActorAction(*(head[:, 5:] for head in action))
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.indices, encode_actions(team_b_heads))
    np.testing.assert_array_equal(
        first.probabilities, jax.nn.one_hot(first.indices, 198, dtype=jnp.float32)
    )
    np.testing.assert_array_equal(first.log_prob, 0.0)


def test_built_in_methods_use_their_own_marked_step() -> None:
    params = initialize_ppo(
        jax.random.key(91), DEFAULT_PPO_CONFIG, method="mappo"
    ).actor_params
    system = make_ppo_system(params)
    assert fixed_member_step(system) is oracle_step(system.apply)


def test_unrouted_fixed_opponents_are_refused() -> None:
    def plain(
        variables: object, memory: object, inputs: SystemInput, keys: Array
    ) -> tuple[object, object]:
        raise AssertionError("never called")

    for member in (shared_policy(policy("random")), System("plain", plain)):
        with pytest.raises(ValueError, match="no route"):
            fixed_member_step(member)


def _with_handed_weight(params: Tree) -> Tree:
    torso = {
        **params["params"]["post_torso"],
        "handed": {"kernel": jnp.zeros((_HANDED_WIDTH, 128), jnp.float32)},
    }
    return {"params": {**params["params"], "post_torso": torso}}


@pytest.mark.parametrize("teammates_only", [False, True])
def test_a_zero_started_oracle_plays_exactly_as_the_shared_runner(
    teammates_only: bool,
) -> None:
    env = marl_bgs.make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(5, 5), max_steps=12),
        num_envs=4,
        metrics="none",
    )
    observations, game = env.reset(jax.random.key(11))
    learner = initialize_ppo(jax.random.key(1), DEFAULT_PPO_CONFIG, method="mappo")
    saved = initialize_ppo(jax.random.key(2), DEFAULT_PPO_CONFIG, method="mappo")
    bots = shared_policy(policy("tdm-alpha"))
    plain_a = make_ppo_system(learner.actor_params)
    plain_b = pool({make_ppo_system(saved.actor_params): 1.0, bots: 1.0})
    oracle_a = oracle_member(
        make_ppo_system(_with_handed_weight(learner.actor_params)), reads_rows=True
    )
    oracle_b = pool(
        {
            oracle_member(make_ppo_system(saved.actor_params), reads_rows=False): 1.0,
            oracle_member(bots, reads_rows=False): 1.0,
        }
    )
    plain = init_systems(plain_a, plain_b, observations, game, jax.random.key(3))
    oracle = init_systems(oracle_a, oracle_b, observations, game, jax.random.key(3))
    fixed = jnp.broadcast_to(jnp.arange(10) >= 5, (4, 10))
    for tick in range(3):
        roots = _lane_keys(20 + tick)
        actions, plain, learning = apply_systems(
            plain_a, plain_b, plain, observations, game, roots
        )
        step = oracle_apply_systems(
            oracle_a,
            oracle_b,
            oracle,
            observations,
            game,
            roots,
            variables_a=oracle_a.variables,
            variables_b=oracle_b.variables,
            ranks=jax.vmap(turn_order)(_lane_keys(40 + tick), fixed),
            teammates_only=teammates_only,
            hand_sampled_actions=False,
        )
        oracle = step.memory
        for left, right in zip(step.actions, actions, strict=True):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(
            jax.tree.leaves(step.learning_outputs[0]),
            jax.tree.leaves(learning[0]),
            strict=True,
        ):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(oracle.team_a.inner, plain.team_a)
        observations, game = env.step(_lane_keys(60 + tick), game, actions)[:2]


def test_each_slot_is_handed_every_earlier_choice_once() -> None:
    env = marl_bgs.make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(5, 5), max_steps=12),
        num_envs=4,
        metrics="none",
    )
    observations, game = env.reset(jax.random.key(12))
    learner = initialize_ppo(jax.random.key(1), DEFAULT_PPO_CONFIG, method="mappo")
    saved = initialize_ppo(jax.random.key(2), DEFAULT_PPO_CONFIG, method="mappo")
    team_a = oracle_member(
        make_ppo_system(_with_handed_weight(learner.actor_params)), reads_rows=True
    )
    team_b = pool(
        {
            oracle_member(make_ppo_system(saved.actor_params), reads_rows=False): 1.0,
            oracle_member(shared_policy(policy("tdm-beta")), reads_rows=False): 1.0,
        }
    )
    memory = init_systems(team_a, team_b, observations, game, jax.random.key(3))
    fixed = jnp.broadcast_to(jnp.arange(10) >= 5, (4, 10))
    ranks = jax.vmap(turn_order)(_lane_keys(5), fixed)
    step = oracle_apply_systems(
        team_a,
        team_b,
        memory,
        observations,
        game,
        _lane_keys(6),
        variables_a=team_a.variables,
        variables_b=team_b.variables,
        ranks=ranks,
        teammates_only=False,
        hand_sampled_actions=False,
    )
    earlier = jax.vmap(functools.partial(chosen_before, teammates_only=False))(
        ranks, game.core_state.alive_mask
    )
    allies = step.handed.ally_previous_timestep_move_actions_one_hot.sum((-2, -1))
    enemies = step.handed.enemy_previous_timestep_move_actions_one_hot.sum((-2, -1))
    np.testing.assert_allclose(allies + enemies, earlier.sum(-1), atol=1e-5)
    np.testing.assert_allclose(enemies[:, :5], 5.0, atol=1e-5)
