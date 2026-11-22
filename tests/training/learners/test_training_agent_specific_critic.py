"""Check that the agent-specific MAPPO critic reads each row's own view in training.

With PPOConfig.agent_specific_state, the batch's stored critic values and
bootstrap values for every centralized-critic method (mappo, ff_mappo, happo,
tarmac) equal one direct critic run whose row i reads the physical view and
actor i's own rebuilt permitted view, so no row reads a broadcast, shifted or
other actor's view. The update's first critic pass reads the same inputs: its
value error equals the stored values' error. Dead agents' rows follow the same
rule, because the critic's loss counts them. These tests establish no learning
or speed claim.
"""

from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.helpers.training_learner_helpers import scanner
from tests.training.learners.test_training_learner import (
    _combat_context,  # pyright: ignore[reportPrivateUsage]
)

from marl_battlegrounds.baselines.config import is_recurrent_method
from marl_battlegrounds.baselines.inputs import (
    encode_actor_inputs,
    spawn_frame_flag,
    team_obstacle_partners,
)
from marl_battlegrounds.baselines.ppo import (
    FeedForwardValueNet,
    PPOConfig,
    RecurrentValueNet,
    calculate_gae,
    update_ppo,
)
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.policies.input import (
    Observations,
    build_team_actor_input,
    mirror_team_view,
)
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    make_training_schedule,
    prepare_training_content,
)
from marl_battlegrounds.training.collection import TrainingRollout
from marl_battlegrounds.training.learner import (
    LearnerState,
    _ppo_state,  # pyright: ignore[reportPrivateUsage]
    build_ppo_batch,
    init_learner,
)

type Tree = Any


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


def _views(observations: Observations, mask: ActionMask, ppo: PPOConfig) -> Array:
    def one(data: tuple[Observations, ActionMask]) -> Array:
        compact, row_mask = data
        actors = jax.vmap(build_team_actor_input, in_axes=(0, None))(compact, 0)
        actors, _ = mirror_team_view(
            actors,
            row_mask,
            spawn_frame_flag(actors, ppo.spawn_frame),
            obstacle_partners=team_obstacle_partners(actors),
        )
        return encode_actor_inputs(actors)

    return cast(Array, jax.lax.map(one, (observations, mask)))


def _time_axis(value: Array) -> Array:
    return value[None]


def _expected(
    state: LearnerState, rollout: TrainingRollout, ppo: PPOConfig, method: str
) -> tuple[Array, Array]:
    rows = rollout.transitions
    physical = cast(Array, rows.training_state)
    own = _views(rows.observations, rows.action_mask, ppo)
    lanes = (*rows.valid.shape, 5)
    valid = jnp.broadcast_to(rows.valid[..., None], lanes)
    final_physical = cast(Array, rollout.final_training_state)[None]
    final_own = _views(
        jax.tree.map(_time_axis, rollout.final_observations),
        jax.tree.map(_time_axis, rollout.final_action_mask),
        ppo,
    )
    continues = ~rollout.final_ended
    final_valid = jnp.broadcast_to(continues[None, :, None], (1, *lanes[1:]))
    if is_recurrent_method(method):
        reset = (rows.valid[0] & rows.episode_start[0])[:, None, None]
        memory = jnp.where(reset, 0.0, state.critic_memory)
        net = RecurrentValueNet(input_scale=ppo.input_scale)
        memory, old = cast(
            tuple[Array, Array],
            net.apply(
                state.critic_params,
                memory,
                physical,
                jnp.broadcast_to(rows.episode_start[..., None], lanes),
                valid,
                own,
            ),
        )
        _, final = cast(
            tuple[Array, Array],
            net.apply(
                state.critic_params,
                memory,
                final_physical,
                jnp.zeros(final_valid.shape, jnp.bool_),
                final_valid,
                final_own,
            ),
        )
    else:
        net = FeedForwardValueNet(input_scale=ppo.input_scale)
        old = cast(Array, net.apply(state.critic_params, physical, own))
        final = cast(Array, net.apply(state.critic_params, final_physical, final_own))
    owned = rollout.final_active
    if rollout.final_learner_active is not None:
        owned = owned & rollout.final_learner_active
    return (
        jnp.where(valid, old, 0.0),
        jnp.where(continues[:, None] & owned, final[0], 0.0),
    )


def _check(
    state: LearnerState, rollout: TrainingRollout, ppo: PPOConfig, method: str
) -> None:
    batch, _ = build_ppo_batch(state, rollout, ppo=ppo, method=method)
    old, final = _expected(state, rollout, ppo, method)
    np.testing.assert_allclose(batch.old_values, old, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(batch.final_values, final, rtol=2e-5, atol=2e-5)
    # Rows differ: the critic no longer gives all five actors one value.
    assert not np.allclose(old[..., 0], old[..., 1])
    # With one epoch, minibatch and group, the update's first critic pass sees
    # the whole batch, so its error equals the stored values' error only if the
    # update feeds every row the same view as the stored values.
    owned = batch.active
    if batch.learner_active is not None:
        owned = owned & batch.learner_active
    samples = batch.valid[..., None] & owned
    _, targets = calculate_gae(
        batch.rewards,
        batch.old_values,
        batch.ended,
        batch.final_values,
        valid=samples,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
    )
    _, metrics = update_ppo(
        _ppo_state(state), batch, jax.random.key(0), ppo, method=method
    )
    error = jnp.where(samples, jnp.square(batch.old_values - targets), 0.0)
    expected = jnp.sqrt(jnp.sum(error) / jnp.maximum(jnp.sum(samples), 1))
    np.testing.assert_allclose(
        float(metrics.value_rmse[0, 0]), float(expected), rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize("method", ("mappo", "ff_mappo", "happo", "tarmac"))
def test_batch_values_give_each_row_its_own_view(
    prepared: PreparedTrainingContent, method: str
) -> None:
    ppo = PPOConfig(
        rollout_length=3,
        epochs=1,
        minibatches=1,
        groups=1,
        input_scale=0.01,
        value_normalization=False,
        agent_specific_state=True,
        parameter_sharing="class" if method == "happo" else "all",
    )
    collection, state = init_learner(
        schedule=make_training_schedule(total_env_steps=12, num_envs=4),
        seed=19112201,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
        method=method,
    )
    _, rollout = scanner(collection, ppo.rollout_length)(state.carry)
    _check(state, rollout, ppo, method)


def test_dead_rows_read_their_own_view_too(prepared: PreparedTrainingContent) -> None:
    original, authored, original_ppo = _combat_context(prepared, "dead")
    ppo = replace(
        original_ppo,
        input_scale=0.01,
        value_normalization=False,
        agent_specific_state=True,
    )
    collection, state = init_learner(
        schedule=original.schedule,
        seed=75,
        ppo=ppo,
        prepared=prepared,
        metrics="none",
    )
    _, rollout = scanner(collection, ppo.rollout_length)(authored.carry)
    assert not bool(jnp.any(rollout.transitions.alive))
    assert bool(jnp.any(rollout.transitions.valid))
    batch, _ = build_ppo_batch(state, rollout, ppo=ppo)
    old, final = _expected(state, rollout, ppo, "mappo")
    np.testing.assert_allclose(batch.old_values, old, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(batch.final_values, final, rtol=2e-5, atol=2e-5)
