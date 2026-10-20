"""Check HAPPO's sequential update against independent references on the CPU.

Contracts: _happo_factor multiplies the factor by exp(new - old) over the
updated group's sample rows only, with every other row counting 1, matching a
NumPy reference; the actor loss weights each row's clipped surrogate by
actor_weight after standardizing advantages, and entropy not at all, matching
a NumPy recomputation, while no weight leaves the ordinary objective. On a
small real three-agent rollout (Mage, Warrior, Priest), one "happo" update
leaves the critic exactly where a recurrent MAPPO update with class sharing
puts it, keeps the absent Hunter and Rogue groups' actor parameters and
optimizer states bit for bit, moves every present group, and updates the
first group in its random order as a MAPPO update restricted to that group's
slots does; "happo" with parameter_sharing "all" is refused. No learning run.
"""

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.helpers.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.baselines import ppo
from marl_battlegrounds.baselines.actions import (
    action_log_prob,
    categorical_action_mask,
    decode_actions,
)
from marl_battlegrounds.baselines.inputs import encode_training_state
from marl_battlegrounds.environment import make

LENGTH, GAMES = 4, 4
CONFIG = ppo.PPOConfig(
    rollout_length=LENGTH,
    epochs=1,
    minibatches=1,
    groups=1,
    parameter_sharing="class",
    value_normalization=False,
)


def test_happo_factor_matches_numpy() -> None:
    generator = np.random.default_rng(5)
    factor = generator.uniform(0.5, 1.5, (3, 2)).astype(np.float32)
    new = generator.normal(-2, 0.3, (3, 2, 5)).astype(np.float32)
    old = generator.normal(-2, 0.3, (3, 2, 5)).astype(np.float32)
    updated = generator.random((3, 2, 5)) < 0.4
    expected = factor * np.prod(np.where(updated, np.exp(new - old), 1.0), axis=-1)
    actual = ppo._happo_factor(
        jnp.asarray(factor), jnp.asarray(new), jnp.asarray(old), jnp.asarray(updated)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    none = ppo._happo_factor(
        jnp.asarray(factor), jnp.asarray(new), jnp.asarray(old), jnp.zeros_like(updated)
    )
    np.testing.assert_array_equal(none, factor)


def _rollout(team_sizes: tuple[int, int]) -> tuple[ppo.PPOBatch, ppo.PPOTrainState]:
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=team_sizes),
        num_envs=GAMES,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(3))
    rows: dict[str, list[Any]] = {
        key: [] for key in ("obs", "physical", "mask", "start", "action", "active")
    }
    for timestep in range(LENGTH):
        inputs = env.policy_inputs(observations, state)
        legal = np.asarray(categorical_action_mask(inputs.action_mask))
        choices = np.empty((GAMES, 5), np.int32)
        for game, slot in np.ndindex(GAMES, 5):
            options = np.flatnonzero(legal[game, slot])
            choices[game, slot] = options[
                (timestep * 5 + game * 3 + slot) % len(options)
            ]
        rows["obs"].append(observations)
        rows["physical"].append(encode_training_state(state.core_state, state.config))
        rows["mask"].append(inputs.action_mask)
        rows["start"].append(inputs.episode_start)
        rows["action"].append(jnp.asarray(choices))
        rows["active"].append(inputs.active_mask)
        opponent = env.policy_inputs(observations, state, team=1)
        zero = decode_actions(jnp.zeros(opponent.active_mask.shape, jnp.int32))
        observations, state, *_ = env.step(
            jax.random.key(10 + timestep),
            state,
            env.join_actions(decode_actions(jnp.asarray(choices)), zero),
        )
    stack = functools.partial(jax.tree.map, lambda *values: jnp.stack(values))
    active = jnp.stack(rows["active"])
    shape = (LENGTH, GAMES, 5)
    generator = np.random.default_rng(7)
    learner = ppo.initialize_ppo(jax.random.key(4), CONFIG, method="happo")
    batch = ppo.PPOBatch(
        stack(*rows["obs"]),
        jnp.stack(rows["physical"]),
        stack(*rows["mask"]),
        jnp.stack(rows["action"]),
        jnp.asarray(generator.normal(-5.0, 0.2, shape), jnp.float32),
        jnp.asarray(generator.normal(0, 0.3, shape), jnp.float32),
        jnp.asarray(generator.normal(0, 0.5, shape), jnp.float32),
        jnp.zeros((LENGTH, GAMES), bool),
        jnp.stack(rows["start"]),
        jnp.ones((LENGTH, GAMES), bool),
        active,
        active,
        jnp.asarray(generator.normal(0, 0.1, (GAMES, 5)), jnp.float32),
        jnp.zeros((GAMES, 5, 128), jnp.float32),
        jnp.zeros((GAMES, 5, 128), jnp.float32),
    )
    return batch, learner


def _group(tree: object, group: int) -> list[Array]:
    return [leaf[group] for leaf in jax.tree.leaves(tree)]


def test_one_happo_update_against_mappo_references() -> None:
    batch, learner = _rollout((3, 2))
    key = jax.random.key(21)
    happo, metrics = ppo.update_ppo(learner, batch, key, CONFIG, method="happo")
    mappo, _ = ppo.update_ppo(learner, batch, key, CONFIG, method="mappo")
    for actual, expected in zip(
        jax.tree.leaves(happo.critic_params),
        jax.tree.leaves(mappo.critic_params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    # Team A holds a Mage (0), a Warrior (1) and a Priest (4).
    for group in (2, 3):
        for before, after in zip(
            _group((learner.actor_params, learner.actor_opt_state), group),
            _group((happo.actor_params, happo.actor_opt_state), group),
            strict=True,
        ):
            np.testing.assert_array_equal(before, after)
    for group in (0, 1, 4):
        assert any(
            not np.array_equal(before, after)
            for before, after in zip(
                _group(learner.actor_params, group),
                _group(happo.actor_params, group),
                strict=True,
            )
        )
    assert int(np.sum(metrics.actor_samples)) == int(np.sum(batch.active))
    order = list(np.asarray(jax.random.permutation(jax.random.fold_in(key, 1), 5)))
    first = next(group for group in order if group in (0, 1, 4))
    # Slots 0-2 hold the Mage, Warrior and Priest (class groups 0, 1, 4).
    slot_groups = np.asarray([0, 1, 4, -1, -1])
    alone = batch._replace(
        learner_active=jnp.broadcast_to(
            jnp.asarray(slot_groups == first), batch.actions.shape
        )
    )
    restricted, _ = ppo.update_ppo(learner, alone, key, CONFIG, method="mappo")
    for actual, expected in zip(
        _group(happo.actor_params, first),
        _group(restricted.actor_params, first),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_weighted_surrogate_matches_numpy() -> None:
    batch, learner = _rollout((3, 2))
    key = jax.random.key(31)
    weight = jax.random.uniform(key, batch.actions.shape, minval=0.5, maxval=1.5)
    captured: list[ppo.PPOMinibatch] = []
    original = ppo.update_minibatch

    def capture(
        state: ppo.PPOTrainState,
        minibatch: ppo.PPOMinibatch,
        config: ppo.PPOConfig = ppo.DEFAULT_PPO_CONFIG,
        *,
        method: str = "mappo",
    ) -> tuple[ppo.PPOTrainState, ppo.PPOMetrics]:
        captured.append(minibatch)
        return original(state, minibatch, config, method=method)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ppo, "update_minibatch", capture)
        with jax.disable_jit():
            ppo.update_ppo(learner, batch, key, CONFIG, method="mappo")
    group: ppo.PPOMinibatch = jax.tree.map(lambda value: value[0], captured[0])  # pyright: ignore[reportUnknownArgumentType]
    weighted = group._replace(actor_weight=weight)
    loss, (entropy, _, _) = ppo._actor_loss_with_metrics(
        learner.actor_params, weighted, CONFIG
    )
    plain, _ = ppo._actor_loss_with_metrics(learner.actor_params, group, CONFIG)
    logits = ppo._actor_logits(learner.actor_params, group, CONFIG, method="mappo")
    logp = np.asarray(action_log_prob(logits, group.action_mask, group.actions))
    samples = np.asarray(group.actor_samples)
    advantages = np.asarray(group.advantages)
    mean = advantages[samples].mean()
    standardized = (advantages - mean) / (advantages[samples].std() + 1e-8)
    ratio = np.exp(logp - np.asarray(group.old_log_prob))
    clipped = np.clip(ratio, 0.8, 1.2)
    surrogate = np.minimum(ratio * standardized, clipped * standardized)
    expected = -(np.asarray(weight) * surrogate)[samples].mean()
    np.testing.assert_allclose(
        float(loss) + CONFIG.entropy_coefficient * float(entropy), expected, rtol=1e-4
    )
    np.testing.assert_allclose(
        float(plain) + CONFIG.entropy_coefficient * float(entropy),
        -surrogate[samples].mean(),
        rtol=1e-4,
    )


def test_happo_refuses_shared_actors() -> None:
    batch, learner = _rollout((3, 2))
    config = ppo.PPOConfig(
        rollout_length=LENGTH,
        epochs=1,
        minibatches=1,
        groups=1,
        value_normalization=False,
    )
    with pytest.raises(ValueError, match="one actor group at a time"):
        ppo.update_ppo(learner, batch, jax.random.key(1), config, method="happo")
