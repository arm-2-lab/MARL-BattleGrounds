"""Check HAPPO's sequential update against independent references on the CPU.

Contracts: _happo_factor multiplies the factor by exp(new - old) where the
updated group had a sample, with every other step and game counting 1,
matching a NumPy reference; with two gradient groups and two minibatches,
under class and slot sharing, every group pass receives as its weight exactly
the product of the earlier groups' new-to-old ratios on their own rows, each
game once, recomputed from the parameters each earlier pass left; the actor
loss weights each row's clipped surrogate by actor_weight after standardizing
advantages, and entropy not at all, matching a NumPy recomputation, while no
weight leaves the ordinary objective. On a
small real three-agent rollout (Mage, Warrior, Priest), one "happo" update
leaves the critic exactly where a recurrent MAPPO update with class sharing
puts it, keeps the absent Hunter and Rogue groups' actor parameters and
optimizer states bit for bit, moves every present group, and updates the
first group in its random order as a MAPPO update restricted to that group's
slots does; "happo" with parameter_sharing "all" is refused. No learning run.
"""

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import contextlib
import dataclasses
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
    log_ratio = generator.normal(0, 0.3, (3, 2)).astype(np.float32)
    updated = generator.random((3, 2)) < 0.4
    expected = factor * np.where(updated, np.exp(log_ratio), 1.0)
    actual = ppo._happo_factor(
        jnp.asarray(factor), jnp.asarray(log_ratio), jnp.asarray(updated)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    none = ppo._happo_factor(
        jnp.asarray(factor), jnp.asarray(log_ratio), jnp.zeros_like(updated)
    )
    np.testing.assert_array_equal(none, factor)


def _learner(config: ppo.PPOConfig = CONFIG) -> ppo.PPOTrainState:
    return ppo.initialize_ppo(jax.random.key(4), config, method="happo")


@functools.cache
def _rollout() -> ppo.PPOBatch:
    # Team A holds a Mage, a Warrior and a Priest; Team B two actors.
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(3, 2)),
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
    return ppo.PPOBatch(
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


def _group(tree: object, group: int) -> list[Array]:
    return [leaf[group] for leaf in jax.tree.leaves(tree)]


class _CapturedError(Exception):
    pass


def _captured_update(
    batch: ppo.PPOBatch,
    learner: ppo.PPOTrainState,
    key: Array,
    config: ppo.PPOConfig,
    *,
    method: str,
    calls: int | None = None,
) -> tuple[list[tuple[ppo.PPOTrainState, ppo.PPOMinibatch]], ppo.PPOTrainState | None]:
    captured: list[tuple[ppo.PPOTrainState, ppo.PPOMinibatch]] = []
    original = ppo.update_minibatch

    def capture(
        state: ppo.PPOTrainState,
        minibatch: ppo.PPOMinibatch,
        config: ppo.PPOConfig = ppo.DEFAULT_PPO_CONFIG,
        *,
        method: str = "mappo",
    ) -> tuple[ppo.PPOTrainState, ppo.PPOMetrics]:
        captured.append((state, minibatch))
        if len(captured) == calls:
            raise _CapturedError
        return original(state, minibatch, config, method=method)

    final = None
    with pytest.MonkeyPatch.context() as patch, jax.disable_jit():
        patch.setattr(ppo, "update_minibatch", capture)
        with contextlib.suppress(_CapturedError):
            final, _ = ppo.update_ppo(learner, batch, key, config, method=method)
    return captured, final


@pytest.mark.parametrize("sharing", ["class", "none"])
def test_each_group_pass_is_weighted_by_the_earlier_groups_ratios(
    sharing: str,
) -> None:
    config = dataclasses.replace(
        CONFIG, minibatches=2, groups=2, parameter_sharing=sharing
    )
    batch = _rollout()
    calls, final = _captured_update(
        batch, _learner(config), jax.random.key(21), config, method="happo"
    )
    assert final is not None
    per_pass = config.epochs * config.minibatches
    assert len(calls) == 6 * per_pass
    old = np.asarray(batch.old_log_prob)
    factor = np.ones(old.shape[:2])
    for group_pass in range(1, 6):
        after = (
            calls[(group_pass + 1) * per_pass][0].actor_params
            if group_pass < 5
            else final.actor_params
        )
        ratio = np.ones_like(factor)
        seen: list[int] = []
        for _, minibatch in calls[group_pass * per_pass : (group_pass + 1) * per_pass]:
            logits = jax.lax.map(
                functools.partial(
                    ppo._actor_logits, after, config=config, method="happo"
                ),
                minibatch,
            )
            logp = np.asarray(
                action_log_prob(logits, minibatch.action_mask, minibatch.actions)
            )
            rows = np.asarray(minibatch.old_log_prob)
            for gradient_group, row in np.ndindex(rows.shape[0], rows.shape[2]):
                column = rows[gradient_group, :, row, 0]
                (game,) = {
                    game
                    for game, slot in np.ndindex(*old.shape[1:])
                    if np.array_equal(old[:, game, slot], column)
                }
                seen.append(game)
                np.testing.assert_allclose(
                    np.asarray(minibatch.actor_weight)[gradient_group, :, row, 0],
                    factor[:, game],
                    rtol=1e-5,
                )
                sampled = np.asarray(minibatch.actor_samples)[gradient_group, :, row, 0]
                ratio[:, game] = np.where(
                    sampled, np.exp(logp[gradient_group, :, row, 0] - column), 1.0
                )
        assert sorted(seen) == list(range(GAMES))
        factor = factor * ratio
    assert np.ptp(factor) > 0


def test_one_happo_update_against_mappo_references() -> None:
    batch, learner = _rollout(), _learner()
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
    order_key = jax.random.fold_in(key, ppo._HAPPO_ORDER_TAG)
    order = list(np.asarray(jax.random.permutation(order_key, 5)))
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
    batch, learner = _rollout(), _learner()
    key = jax.random.key(31)
    weight = jax.random.uniform(key, batch.actions.shape, minval=0.5, maxval=1.5)
    [(_, captured)], _ = _captured_update(
        batch, learner, key, CONFIG, method="mappo", calls=1
    )
    group: ppo.PPOMinibatch = jax.tree.map(lambda value: value[0], captured)  # pyright: ignore[reportUnknownArgumentType]
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
    batch, learner = _rollout(), _learner()
    config = ppo.PPOConfig(
        rollout_length=LENGTH,
        epochs=1,
        minibatches=1,
        groups=1,
        value_normalization=False,
    )
    with pytest.raises(ValueError, match="one actor group at a time"):
        ppo.update_ppo(learner, batch, jax.random.key(1), config, method="happo")
