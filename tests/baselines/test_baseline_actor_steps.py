"""Check that each actor step returns the distribution its action was drawn from.

The private steps behind the PPO, QMIX and PQN-VDN System hooks are the
current-action oracle's route to an actor's action probabilities. Each returns,
beside its index in world directions, the masked distribution in world
directions. For the PPO actors (shared recurrent, feedforward, per-class and
TarMAC) the probability of the returned index equals the exp of the step's own
log-probability exactly, for both teams, so views the spawn frame mirrors are
covered; illegal actions get probability zero and each distribution sums to one.
QMIX and PQN-VDN hand their epsilon-greedy distribution: one-hot on the greedy
action at epsilon zero and uniform over the legal actions at epsilon one, with
the chosen index always possible. A System hook's code is part of every saved
System's recorded identity, so each step is a separate copy of its hook's body:
with nothing handed, every step gives exactly its hook's actions, next memory and
learning outputs, for both teams. Every System the PPO, QMIX and PQN-VDN
factories build carries its own step through the oracle marker (the
historical schema-1 and grouped PQN-VDN hooks carry none, so the oracle refuses
them), and the marked step reproduces the System's own call.
"""

import functools
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.helpers.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.baselines import pqn, qmix
from marl_battlegrounds.baselines.actions import categorical_action_mask, decode_actions
from marl_battlegrounds.baselines.ppo import (
    _PARAMETER_SHARING,  # pyright: ignore[reportPrivateUsage]
    DEFAULT_PPO_CONFIG,
    PPOLearningOutputs,
    _actor_step,  # pyright: ignore[reportPrivateUsage]
    _ActorStep,  # pyright: ignore[reportPrivateUsage]
    _apply_actor,  # pyright: ignore[reportPrivateUsage]
    _apply_feedforward_actor,  # pyright: ignore[reportPrivateUsage]
    _apply_tarmac_actor,  # pyright: ignore[reportPrivateUsage]
    _feedforward_step,  # pyright: ignore[reportPrivateUsage]
    _grouped_ppo_actor_apply,  # pyright: ignore[reportPrivateUsage]
    _grouped_step,  # pyright: ignore[reportPrivateUsage]
    _tarmac_step,  # pyright: ignore[reportPrivateUsage]
    initialize_ppo,
    make_ppo_system,
    oracle_step,
)
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
    system_inputs,
)

type Tree = Any
LEFT = 1
GAMES = 4


@pytest.fixture(scope="module")
def teams() -> tuple[SystemInput, SystemInput]:
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(5, 5), max_steps=12),
        num_envs=GAMES,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(10))
    return (
        system_inputs(observations, state, team=0),
        system_inputs(observations, state, team=1),
    )


@functools.cache
def _ppo_params(method: str, **settings: str | int) -> Tree:
    config = replace(DEFAULT_PPO_CONFIG, **settings)
    return initialize_ppo(jax.random.key(91), config, method=method).actor_params


def _memory() -> Array:
    return jnp.zeros((GAMES, 5, 128), jnp.float32)


def _keys(seed: int) -> Array:
    return jax.random.split(jax.random.key(seed), GAMES)


PPO_STEPS: dict[str, Callable[[SystemInput], _ActorStep]] = {
    "recurrent": lambda inputs: _actor_step(
        _ppo_params("mappo"),
        _memory(),
        inputs,
        _keys(1),
        input_scale=1.0,
        spawn_frame_index=LEFT,
    ),
    "feedforward": lambda inputs: _feedforward_step(
        _ppo_params("ff_ippo"),
        inputs,
        _keys(2),
        input_scale=1.0,
        spawn_frame_index=LEFT,
    ),
    "per_class": lambda inputs: _grouped_step(
        _ppo_params("happo", parameter_sharing="class"),
        _memory(),
        inputs,
        _keys(3),
        input_scale=1.0,
        spawn_frame_index=LEFT,
        parameter_sharing_index=_PARAMETER_SHARING.index("class"),
        recurrent_actor=True,
    ),
    "tarmac": lambda inputs: _tarmac_step(
        _ppo_params("tarmac", message_rounds=2),
        _memory(),
        inputs,
        _keys(4),
        input_scale=1.0,
        spawn_frame_index=LEFT,
        messages_index=0,
    ),
}


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("name", PPO_STEPS)
def test_a_ppo_steps_distribution_gives_its_own_log_probability(
    teams: tuple[SystemInput, SystemInput], name: str, team: int
) -> None:
    step = PPO_STEPS[name](teams[team])
    chosen = jnp.take_along_axis(step.probabilities, step.indices[..., None], -1)
    np.testing.assert_array_equal(chosen[..., 0], jnp.exp(step.log_prob))
    np.testing.assert_allclose(step.probabilities.sum(-1), 1.0, atol=1e-5)


def test_a_mirrored_view_returns_world_directions(
    teams: tuple[SystemInput, SystemInput],
) -> None:
    inputs = teams[1]
    step = PPO_STEPS["recurrent"](inputs)
    legal = np.asarray(categorical_action_mask(inputs.action_mask))
    assert (np.asarray(step.probabilities)[~legal] == 0).all()


@functools.cache
def _q_variables(epsilon: float) -> qmix.QMIXActorVariables:
    state = qmix.initialize_qmix(jax.random.key(5))
    return qmix.QMIXActorVariables(state.online_q, jnp.asarray(epsilon, jnp.float32))


@functools.cache
def _pqn_variables(epsilon: float) -> pqn.PQNActorVariables:
    state = pqn.initialize_pqn(jax.random.key(6), planned_learning_blocks=10)
    return pqn.PQNActorVariables(state.network, jnp.asarray(epsilon, jnp.float32))


def _q_step(inputs: SystemInput, epsilon: float) -> _ActorStep:
    return qmix._q_step(  # pyright: ignore[reportPrivateUsage]
        _q_variables(epsilon),
        jnp.zeros((GAMES, 5, 256), jnp.float32),
        inputs,
        _keys(7),
        input_scale=1.0,
        spawn_frame_index=LEFT,
    )


def _pqn_step(inputs: SystemInput, epsilon: float) -> _ActorStep:
    return pqn._pqn_step(  # pyright: ignore[reportPrivateUsage]
        _pqn_variables(epsilon),
        jnp.zeros((GAMES, 5, 512), jnp.float32),
        inputs,
        _keys(8),
        input_scale=1.0,
        spawn_frame_index=LEFT,
        normalization_index=0,
    )


@pytest.mark.parametrize("step", [_q_step, _pqn_step], ids=["qmix", "pqn_vdn"])
@pytest.mark.parametrize("team", [0, 1])
def test_value_methods_hand_their_epsilon_greedy_distribution(
    teams: tuple[SystemInput, SystemInput],
    step: Callable[[SystemInput, float], _ActorStep],
    team: int,
) -> None:
    inputs = teams[team]
    legal = np.asarray(categorical_action_mask(inputs.action_mask))
    greedy = step(inputs, 0.0)
    chosen = np.take_along_axis(
        np.asarray(greedy.probabilities), np.asarray(greedy.indices)[..., None], -1
    )
    np.testing.assert_array_equal(chosen, 1.0)
    uniform = step(inputs, 1.0)
    count = legal.sum(-1)
    world_legal = np.asarray(uniform.probabilities) > 0
    np.testing.assert_array_equal(world_legal.sum(-1), count)
    np.testing.assert_allclose(
        np.asarray(uniform.probabilities).max(-1), 1.0 / count, rtol=1e-6
    )
    for result in (greedy, uniform):
        np.testing.assert_array_equal(np.isfinite(result.log_prob), True)


def _hook_pairs(
    inputs: SystemInput,
) -> dict[str, tuple[_ActorStep, SystemOutput, Tree]]:
    """Run each step and its frozen hook on the same arguments."""
    recurrent = _ppo_params("mappo")
    feedforward = _ppo_params("ff_ippo")
    grouped = _ppo_params("happo", parameter_sharing="class")
    tarmac = _ppo_params("tarmac", message_rounds=2)
    q_variables, pqn_variables = _q_variables(0.3), _pqn_variables(0.3)
    q_memory = jnp.zeros((GAMES, 5, 256), jnp.float32)
    pqn_memory = jnp.zeros((GAMES, 5, 512), jnp.float32)
    hook = _grouped_ppo_actor_apply(1.0, "left", "class", True)
    return {
        "recurrent": (
            PPO_STEPS["recurrent"](inputs),
            _apply_actor(
                recurrent,
                _memory(),
                inputs,
                _keys(1),
                input_scale=1.0,
                spawn_frame_index=LEFT,
            ),
            None,
        ),
        "feedforward": (
            PPO_STEPS["feedforward"](inputs),
            _apply_feedforward_actor(
                feedforward,
                (),
                inputs,
                _keys(2),
                input_scale=1.0,
                spawn_frame_index=LEFT,
            ),
            None,
        ),
        "per_class": (
            PPO_STEPS["per_class"](inputs),
            hook(grouped, _memory(), inputs, _keys(3)),
            None,
        ),
        "tarmac": (
            PPO_STEPS["tarmac"](inputs),
            _apply_tarmac_actor(
                tarmac,
                _memory(),
                inputs,
                _keys(4),
                messages_index=0,
                input_scale=1.0,
                spawn_frame_index=LEFT,
            ),
            "senders",
        ),
        "qmix": (
            _q_step(inputs, 0.3),
            qmix._apply_q_actor(  # pyright: ignore[reportPrivateUsage]
                q_variables,
                q_memory,
                inputs,
                _keys(7),
                input_scale=1.0,
                spawn_frame_index=LEFT,
            ),
            "values",
        ),
        "pqn_vdn": (
            _pqn_step(inputs, 0.3),
            pqn._apply_pqn_actor(  # pyright: ignore[reportPrivateUsage]
                pqn_variables,
                pqn_memory,
                inputs,
                _keys(8),
                input_scale=1.0,
                spawn_frame_index=LEFT,
            ),
            "values",
        ),
    }


@pytest.mark.parametrize("team", [0, 1])
def test_every_step_with_nothing_handed_is_its_frozen_hook(
    teams: tuple[SystemInput, SystemInput], team: int
) -> None:
    for name, (step, output, kind) in _hook_pairs(teams[team]).items():
        for left, right in zip(
            decode_actions(step.indices), output.actions, strict=True
        ):
            np.testing.assert_array_equal(left, right, err_msg=name)
        for left, right in zip(
            jax.tree.leaves(step.memory),
            jax.tree.leaves(output.next_memory),
            strict=True,
        ):
            np.testing.assert_array_equal(left, right, err_msg=name)
        if kind is None:
            expected = PPOLearningOutputs(step.indices, step.log_prob)
            for left, right in zip(expected, output.learning_outputs, strict=True):
                np.testing.assert_array_equal(left, right, err_msg=name)
        elif kind == "senders":
            outputs = output.learning_outputs
            np.testing.assert_array_equal(step.indices, outputs.action_indices)
            np.testing.assert_array_equal(step.log_prob, outputs.log_prob)


def _marked_systems() -> dict[str, tuple[System, Tree]]:
    q_state = qmix.initialize_qmix(jax.random.key(5))
    pqn_state = pqn.initialize_pqn(jax.random.key(6), planned_learning_blocks=10)
    ln_state = pqn.initialize_pqn(
        jax.random.key(6),
        pqn=replace(pqn.DEFAULT_PQN_CONFIG, normalization="layer_norm"),
        planned_learning_blocks=10,
    )
    recurrent = _memory()
    return {
        "mappo_world": (
            make_ppo_system(_ppo_params("mappo"), spawn_frame="world"),
            recurrent,
        ),
        "mappo_left": (make_ppo_system(_ppo_params("mappo")), recurrent),
        "happo_class": (
            make_ppo_system(
                _ppo_params("happo", parameter_sharing="class"),
                method="happo",
                parameter_sharing="class",
            ),
            recurrent,
        ),
        "tarmac": (
            make_ppo_system(_ppo_params("tarmac", message_rounds=2), method="tarmac"),
            recurrent,
        ),
        "ff_ippo": (make_ppo_system(_ppo_params("ff_ippo"), method="ff_ippo"), ()),
        "qmix": (
            qmix.make_qmix_system(q_state.online_q, epsilon=0.3),
            jnp.zeros((GAMES, 5, 256), jnp.float32),
        ),
        "qmix_class": (
            qmix.make_qmix_system(
                qmix.initialize_qmix(
                    jax.random.key(5),
                    replace(qmix.DEFAULT_QMIX_CONFIG, parameter_sharing="class"),
                ).online_q,
                epsilon=0.3,
                parameter_sharing="class",
            ),
            jnp.zeros((GAMES, 5, 256), jnp.float32),
        ),
        "pqn_vdn": (
            pqn.make_pqn_system(pqn_state.network, epsilon=0.3),
            jnp.zeros((GAMES, 5, 512), jnp.float32),
        ),
        "pqn_vdn_layer_norm": (
            pqn.make_pqn_system(
                ln_state.network, epsilon=0.3, normalization="layer_norm"
            ),
            jnp.zeros((GAMES, 5, 512), jnp.float32),
        ),
    }


def test_every_marked_system_reaches_its_own_step(
    teams: tuple[SystemInput, SystemInput],
) -> None:
    inputs = teams[1]
    for name, (system, memory) in _marked_systems().items():
        step = oracle_step(system.apply)
        assert step is not None, name
        result = step(system.variables, memory, inputs, _keys(9))
        output = cast(
            SystemOutput, system.apply(system.variables, memory, inputs, _keys(9))
        )
        for left, right in zip(
            decode_actions(result.indices), output.actions, strict=True
        ):
            np.testing.assert_array_equal(left, right, err_msg=name)
        for left, right in zip(
            jax.tree.leaves(result.memory),
            jax.tree.leaves(output.next_memory),
            strict=True,
        ):
            np.testing.assert_array_equal(left, right, err_msg=name)


def test_historical_and_unbuilt_hooks_carry_no_step() -> None:
    schema_1 = make_ppo_system(_ppo_params("mappo"), actor_input_schema=1)
    assert oracle_step(schema_1.apply) is None
    grouped_pqn = pqn.make_pqn_system(
        pqn.initialize_pqn(
            jax.random.key(6),
            pqn=replace(pqn.DEFAULT_PQN_CONFIG, parameter_sharing="class"),
            planned_learning_blocks=10,
        ).network,
        parameter_sharing="class",
    )
    assert oracle_step(grouped_pqn.apply) is None
    assert oracle_step(lambda: None) is None
