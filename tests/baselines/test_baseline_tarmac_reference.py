"""Check TarMAC's one-round teammate messages against references on the CPU.

Contracts: one TarMAC tick (keys, values and queries from each actor's stored
memory, the masked softmax over valid senders with the receiver's own message
included, the 32-float message joined to the GRU's input) matches an
independent NumPy computation under "attention", "mean" (equal weights over
the same senders) and "blocked" (no sender), and a receiver with no valid
sender reads an all-zero message; a reset row starts its GRU from zero and an
invalid row keeps its memory and emits zero; the gradient stays finite on a first tick
where no sender is valid. A message sent on one tick is read on the next and
never within it (rule 6, "Each Agent Decides Alone"): a teammate's view at
tick t never changes another actor's tick-t logits and does change its tick
t+1 logits, never under "blocked". On a real game, played by a TarMAC team
on slots 0 and 2 beside a Random partner on slot 1 (team(...)), with empty
slots 3 and 4: every sender is masked on the first tick; inactive slots, the
partner's slot and a teammate that really died are masked, living teammates
are heard, and a dead receiver still hears them; changing a masked sender's
memory never changes a receiver's next memory, while changing a heard
sender's memory does. On that game, every stored mask equals the simulator's
own roster, ownership, alive and game-start facts, and every reset row (ticks
0 and 100) is empty. Replay
matches play, with an action head scaled 100 times so messages matter as in
trained weights: on a block that starts mid-game at tick 52 from the stored
team memory, with every masked-sender kind, heard senders on its first tick
and a reset inside, the update's own minibatch reproduces each stored log
probability within 1e-5, while a transposed, game-shifted or tick-shifted
mask does not,
and the update's policy change is zero. A "blocked" control trained with
learner_slots beside a Random partner and a past copy keeps its name, its
setting and its "blocked" hook through export and load, is refused as a
warm start or a restore with "attention" settings, plays as a pool member
from its folder, is refused as a portable model file, and extends with its
setting kept. Other methods refuse a messages choice.
"""

# pyright: reportPrivateUsage=false
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax import Array
from tests.helpers.evaluation_fixtures import evaluation_env_config

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines import ppo
from marl_battlegrounds.baselines.actions import (
    action_log_prob,
    categorical_action_mask,
    decode_actions,
)
from marl_battlegrounds.baselines.config import TARMAC_MESSAGES, saved_settings
from marl_battlegrounds.baselines.inputs import (
    encode_training_state,
)
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import SystemOutput
from marl_battlegrounds.training import TrainResult

GAMES, TICKS, START = 32, 104, 52
CONFIG = ppo.PPOConfig(
    rollout_length=TICKS - START,
    epochs=1,
    minibatches=1,
    groups=1,
    value_normalization=False,
)
OWNED = np.asarray([True, False, True, False, False])


@functools.cache
def _learner() -> ppo.PPOTrainState:
    return ppo.initialize_ppo(jax.random.key(4), CONFIG, method="tarmac")


@functools.cache
def _sharp() -> ppo.PPOTrainState:
    # A 100x action head, so messages move log probabilities as trained weights
    # do and a wrong replay mask shows; untrained heads hide it (gain 0.01).
    learner = _learner()
    params = learner.actor_params["params"]
    head = params["action_head"]["Dense_0"]
    sharp = {
        **params,
        "action_head": {"Dense_0": {**head, "kernel": head["kernel"] * 100}},
    }
    return learner._replace(actor_params={"params": sharp})


def _cell() -> ppo.Tree:
    return _learner().actor_params["params"]["TarMACRNN_0"]


@pytest.mark.parametrize("messages", TARMAC_MESSAGES)
def test_one_tick_matches_a_numpy_reference(messages: str) -> None:
    generator = np.random.default_rng(1)
    carry = generator.normal(size=(3, 5, 128)).astype(np.float32)
    embedding = generator.normal(size=(3, 5, 128)).astype(np.float32)
    senders = generator.random((3, 5, 5)) < 0.6
    senders[0, 2] = False
    flags, valid = np.zeros((3, 5), bool), np.ones((3, 5), bool)
    flags[1, 3], valid[2, 4] = True, False
    memory, output = cast(
        tuple[Array, Array],
        ppo.TarMACRNN(messages=messages).apply(
            {"params": _cell()},
            jnp.asarray(carry),
            (embedding[None], flags[None], valid[None], senders[None]),
        ),
    )
    weights = jax.tree.map(np.asarray, _cell())

    def dense(name: str) -> np.ndarray:
        return carry @ weights[name]["kernel"] + weights[name]["bias"]

    heard = senders & (messages != "blocked")
    scores = (
        np.einsum("eif,ejf->eij", dense("query"), dense("key")) / 4.0
        if messages == "attention"
        else np.zeros((3, 5, 5), np.float32)
    )
    top = np.where(heard, scores, -np.inf).max(-1, keepdims=True)
    exp = np.where(heard, np.exp(scores - np.where(np.isfinite(top), top, 0.0)), 0.0)
    total = exp.sum(-1, keepdims=True)
    alpha = np.divide(exp, total, out=np.zeros_like(exp), where=total > 0)
    message = np.einsum("eij,ejf->eif", alpha, dense("value"))
    assert not message[0, 2].any()
    before = np.where((flags & valid)[..., None], 0.0, carry)
    updated, _ = nn.GRUCell(features=128).apply(
        {"params": _cell()["GRUCell_1"]},
        jnp.asarray(before),
        jnp.asarray(np.concatenate([embedding, message], -1)),
    )
    updated = np.asarray(updated)
    np.testing.assert_allclose(
        memory, np.where(valid[..., None], updated, carry), atol=2e-5
    )
    np.testing.assert_allclose(
        output[0], np.where(valid[..., None], updated, 0.0), atol=2e-5
    )


def test_gradient_is_finite_on_a_first_tick_with_no_valid_sender() -> None:
    generator = np.random.default_rng(2)
    features = jnp.asarray(generator.normal(0, 0.1, (2, 2, 5, ppo.ACTOR_FEATURE_SIZE)))
    starts = jnp.zeros((2, 2, 5), bool).at[0].set(True)
    senders = jnp.ones((2, 2, 5, 5), bool).at[0].set(False)

    def loss(params: ppo.Tree) -> Array:
        memory, logits = cast(
            tuple[Array, Array],
            ppo.TarMACActor().apply(
                params,
                jnp.asarray(generator.normal(size=(2, 5, 128)), jnp.float32),
                features,
                starts,
                jnp.ones((2, 2, 5), bool),
                senders,
            ),
        )
        return jnp.sum(logits) + jnp.sum(memory)

    gradient = cast(ppo.Tree, jax.grad(loss)(_learner().actor_params))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradient))


@pytest.mark.parametrize("messages", ("attention", "blocked"))
def test_a_message_arrives_one_tick_late(messages: str) -> None:
    generator = np.random.default_rng(3)
    features = generator.normal(0, 0.1, (2, 2, 5, ppo.ACTOR_FEATURE_SIZE))
    carry = jnp.asarray(generator.normal(size=(2, 5, 128)), jnp.float32)

    def logits(view: np.ndarray) -> np.ndarray:
        _, result = ppo.TarMACActor(messages=messages).apply(
            _learner().actor_params,
            carry,
            jnp.asarray(view, jnp.float32),
            jnp.zeros((2, 2, 5), bool),
            jnp.ones((2, 2, 5), bool),
            jnp.ones((2, 2, 5, 5), bool),
        )
        return np.asarray(result)

    changed = features.copy()
    changed[0, :, 3] += 1.0
    base, moved = logits(features), logits(changed)
    others = [0, 1, 2, 4]
    np.testing.assert_array_equal(base[0][:, others], moved[0][:, others])
    if messages == "blocked":
        np.testing.assert_array_equal(base[1][:, others], moved[1][:, others])
    else:
        assert np.abs(base[1][:, others] - moved[1][:, others]).min() > 0


@functools.cache
def _played() -> dict[str, Any]:
    # TarMAC on slots 0 and 2, a Random partner on slot 1, slots 3 and 4 empty,
    # against a team choosing random legal moves; games restart after 100 ticks.
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(3, 3), max_steps=100),
        num_envs=GAMES,
        metrics="none",
    )
    tarmac = ppo.make_ppo_system(_sharp().actor_params, method="tarmac")
    composed = marl_bgs.team(tarmac, "random", slots=[[0, 2], [1, 3, 4]])
    observations, state = env.reset(jax.random.key(3))
    inputs = env.policy_inputs(observations, state)
    assert composed.init is not None
    memory = composed.init(
        composed.variables, inputs, jax.random.split(jax.random.key(5), GAMES)
    )
    apply = cast(Callable[..., SystemOutput], jax.jit(composed.apply))
    step = cast(Callable[..., Any], jax.jit(env.step))
    played: dict[str, Any] = {"first": inputs}
    rows: dict[str, list[Any]] = {
        name: []
        for name in ("obs", "physical", "mask", "start", "valid", "active", "alive")
    }
    outputs: list[Any] = []
    for tick in range(TICKS):
        inputs = env.policy_inputs(observations, state)
        keys = jax.random.split(jax.random.key(1000 + tick), GAMES)
        if tick == START:
            played["start_memory"] = memory.members[0]
        result = apply(composed.variables, memory, inputs, keys)
        alive = np.asarray(
            inputs.actors.observation.spawn_lifecycle.alive_mask_by_agent_by_team[
                :, :, 0, :
            ]
        )
        dead = (
            np.asarray(inputs.active_mask)
            & ~np.diagonal(alive, axis1=1, axis2=2)
            & ~np.asarray(inputs.episode_start)[:, None]
            & OWNED
        )
        if "dying" not in played and dead.any():
            game, slot = (int(value) for value in np.argwhere(dead)[0])
            played["dying"] = (inputs, memory, keys, game, slot)
        rows["obs"].append(observations)
        rows["physical"].append(encode_training_state(state.core_state, state.config))
        rows["mask"].append(inputs.action_mask)
        rows["start"].append(inputs.episode_start)
        rows["valid"].append(inputs.valid)
        rows["active"].append(inputs.active_mask)
        rows["alive"].append(state.core_state.alive_mask[:, :5])
        outputs.append(result.learning_outputs[0])
        opponent = env.policy_inputs(observations, state, team=1)
        legal = categorical_action_mask(opponent.action_mask)
        noise = jax.random.gumbel(jax.random.key(7000 + tick), legal.shape)
        choice = jnp.argmax(jnp.where(legal, noise, -jnp.inf), -1)
        observations, state, *_ = step(
            jax.random.key(10 + tick),
            state,
            env.join_actions(result.actions, decode_actions(choice)),
        )
        observations, state = env.reset_done(jax.random.key(9000 + tick), state)
        memory = result.next_memory
    played["composed"], played["apply"] = composed, apply

    def stack(values: list[ppo.Tree]) -> ppo.Tree:
        return jax.tree.map(lambda *rows: jnp.stack(rows[START:]), *values)

    learned = jax.tree.map(lambda *rows: jnp.stack(rows), *outputs)
    played["senders"] = np.asarray(learned.sender_mask)
    played["rows"] = {
        name: np.asarray(jnp.stack(rows[name]))
        for name in ("start", "valid", "active", "alive")
    }
    active = stack(rows["active"])
    shape = (TICKS - START, GAMES, 5)
    generator = np.random.default_rng(7)
    played["batch"] = ppo.PPOBatch(
        stack(rows["obs"]),
        stack(rows["physical"]),
        stack(rows["mask"]),
        learned.action_indices[START:],
        learned.log_prob[START:],
        jnp.asarray(generator.normal(0, 0.3, shape), jnp.float32),
        jnp.asarray(generator.normal(0, 0.5, shape), jnp.float32),
        jnp.zeros((TICKS - START, GAMES), bool),
        stack(rows["start"]),
        stack(rows["valid"]),
        active,
        stack(rows["alive"]),
        jnp.zeros((GAMES, 5), jnp.float32),
        played["start_memory"],
        jnp.zeros((GAMES, 5, 128), jnp.float32),
        learner_active=active & OWNED,
        sender_mask=learned.sender_mask[START:],
    )
    return played


def test_real_play_masks_every_sender_kind_and_dead_receivers_still_hear() -> None:
    played = _played()
    assert not np.asarray(ppo.tarmac_sender_mask(played["first"])).any()
    inputs, _, _, game, slot = played["dying"]
    mask = np.asarray(ppo.tarmac_sender_mask(inputs))
    flags = inputs.actors.observation.spawn_lifecycle.alive_mask_by_agent_by_team
    alive = np.asarray(flags)[game, 2 - slot, 0]
    living = [other for other in (0, 1, 2) if alive[other]]
    assert alive[2 - slot] and not alive[slot]
    assert not mask[:, :, 3:].any()
    assert not mask[game, :, slot].any()
    assert mask[game][np.ix_([0, 1, 2], living)].all()
    assert mask[game, slot, living].all()
    owned = inputs._replace(controlled_mask=jnp.asarray(inputs.active_mask & OWNED))
    assert not np.asarray(ppo.tarmac_sender_mask(owned))[:, :, 1].any()
    # Every stored mask equals the simulator's own facts, not the flags the
    # code reads: a sender is heard exactly when it is on the roster, owned by
    # TarMAC, alive in the simulator and the game is not starting.
    stored, rows = played["senders"], played["rows"]
    expected = (rows["active"] & OWNED & rows["alive"] & ~rows["start"][..., None])[
        :, :, None, :
    ]
    receivers = (rows["valid"][..., None] & rows["active"] & OWNED)[..., None]
    np.testing.assert_array_equal(
        np.where(receivers, stored, False),
        np.where(receivers, np.broadcast_to(expected, stored.shape), False),
    )
    assert rows["start"][0].all() and rows["start"][100].all()
    assert not stored[rows["start"]].any()


def test_masked_senders_never_reach_a_receiver_through_the_team() -> None:
    played = _played()
    inputs, memory, keys, game, slot = played["dying"]
    composed, apply = played["composed"], played["apply"]
    heard = 2 - slot
    receivers = [0, 2]

    def next_memory(sender: int | None) -> np.ndarray:
        members = list(memory.members)
        if sender is not None:
            members[0] = members[0].at[game, sender].add(1.0)
        changed = memory._replace(members=tuple(members))
        result = apply(composed.variables, changed, inputs, keys)
        return np.asarray(result.next_memory.members[0][game])

    base = next_memory(None)
    for sender in (1, 3, 4, slot):
        moved = next_memory(sender)
        others = [receiver for receiver in receivers if receiver != sender]
        np.testing.assert_array_equal(moved[others], base[others])
    moved = next_memory(heard)
    assert not np.array_equal(moved[slot], base[slot])


class _CapturedError(Exception):
    pass


def test_replay_matches_play_on_a_block_with_every_masked_kind() -> None:
    batch = cast(ppo.PPOBatch, _played()["batch"])
    valid = np.asarray(batch.valid)[..., None]
    active = np.asarray(batch.active)
    owned = valid & active & OWNED
    counts = {
        "inactive slot": int((valid & ~active).sum()),
        "partner slot": int((valid & active & ~OWNED).sum()),
        "dead teammate": int((owned & ~np.asarray(batch.alive)).sum()),
        "reset inside": int(np.asarray(batch.episode_start)[1:].sum()),
        "heard at the first tick": int(np.asarray(batch.sender_mask)[0].sum()),
    }
    assert all(count > 0 for count in counts.values()), counts
    assert bool(jnp.any(batch.actor_memory != 0))
    captured: list[ppo.PPOMinibatch] = []

    def capture(
        state: ppo.PPOTrainState,
        minibatch: ppo.PPOMinibatch,
        config: ppo.PPOConfig = ppo.DEFAULT_PPO_CONFIG,
        *,
        method: str = "mappo",
    ) -> tuple[ppo.PPOTrainState, ppo.PPOMetrics]:
        captured.append(minibatch)
        raise _CapturedError

    with pytest.MonkeyPatch.context() as patch, jax.disable_jit():
        patch.setattr(ppo, "update_minibatch", capture)
        with pytest.raises(_CapturedError):
            ppo.update_ppo(_sharp(), batch, jax.random.key(6), CONFIG, method="tarmac")

    def first(value: Array) -> Array:
        return value[0]

    group = jax.tree.map(first, captured[0])

    def gap(senders: Array) -> float:
        logits = ppo._actor_logits(
            _sharp().actor_params,
            group._replace(sender_mask=senders),
            CONFIG,
            method="tarmac",
        )
        replayed = action_log_prob(logits, group.action_mask, group.actions)
        return float(
            jnp.max(
                jnp.where(
                    group.actor_samples, jnp.abs(replayed - group.old_log_prob), 0
                )
            )
        )

    assert gap(group.sender_mask) < 1e-5, counts
    assert gap(jnp.swapaxes(group.sender_mask, -1, -2)) > 1e-3
    assert gap(jnp.roll(group.sender_mask, 1, axis=1)) > 1e-3
    assert gap(jnp.roll(group.sender_mask, 1, axis=0)) > 1e-3
    _, metrics = ppo.update_ppo(
        _sharp(), batch, jax.random.key(6), CONFIG, method="tarmac"
    )
    assert abs(float(metrics.approx_kl[0, 0])) < 1e-6
    assert float(metrics.policy_clip_fraction[0, 0]) == 0.0


def test_other_methods_refuse_a_messages_choice() -> None:
    with pytest.raises(ValueError, match="Only TarMAC has a messages setting"):
        ppo.initialize_ppo(
            jax.random.key(0), ppo.PPOConfig(messages="mean"), method="mappo"
        )
    with pytest.raises(ValueError, match="parameter_sharing 'all'"):
        ppo.make_ppo_system(
            _learner().actor_params, method="tarmac", parameter_sharing="class"
        )
    batch = cast(ppo.PPOBatch, _played()["batch"])
    with pytest.raises(ValueError, match="stored sender masks"):
        ppo.update_ppo(
            _learner(),
            batch._replace(sender_mask=None),
            jax.random.key(6),
            CONFIG,
            method="tarmac",
        )
    assert "messages" not in saved_settings(ppo.PPOConfig())
    assert saved_settings(ppo.PPOConfig(messages="mean"))["messages"] == "mean"


@pytest.fixture(scope="module")
def blocked_run(tmp_path_factory: pytest.TempPathFactory) -> TrainResult:
    from marl_battlegrounds import training

    config = training.TrainConfig(
        keep_past=1,
        history_capture_env_steps=(16,),
        learner_slots=(0, 2),
        partners={"random": 1.0},
        method="tarmac",
        seed=19047101,
        num_envs=4,
        total_env_steps=32,
        ppo=ppo.PPOConfig(
            rollout_length=4, groups=2, minibatches=2, epochs=1, messages="blocked"
        ),
        checkpoint_interval_updates=1,
        validation_fractions=(1.0,),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
        recording=False,
        verbose=False,
    )
    return training.train(
        config, output_dir=tmp_path_factory.mktemp("tarmac") / "blocked"
    )


def test_a_blocked_control_keeps_its_setting_from_save_to_load(
    blocked_run: TrainResult, tmp_path: Path
) -> None:
    from marl_battlegrounds import training
    from marl_battlegrounds.training import checkpoints

    run = blocked_run
    details = checkpoints.read_checkpoint_details(run.final_actor)
    assert details["messages"] == "blocked"
    loaded = training.load_system(run.final_actor)
    assert loaded.name.startswith("TarMAC (trained without messages) at step")
    assert loaded.apply is ppo._tarmac_actor_apply(1.0, "left", "blocked")
    assert checkpoints.artifact_identity(run.final_actor)["actor_digest"] == (
        loaded.checkpoint
    )
    with pytest.raises(ValueError, match="Initial actor messages differ"):
        checkpoints.load_initial_actor(
            run.final_actor, method="tarmac", input_scale=1.0, spawn_frame="left"
        )
    checkpoints.load_initial_actor(
        run.final_actor,
        method="tarmac",
        input_scale=1.0,
        spawn_frame="left",
        messages="blocked",
    )
    learner = checkpoints.read_checkpoint_details(run.final_checkpoint)
    expected = {
        key: learner["metadata"][key]
        for key in ("config", "source", "dependencies", "execution")
    }
    with pytest.raises(ValueError, match="Checkpoint messages differs"):
        checkpoints.restore_checkpoint(
            run.final_checkpoint,
            cast(Any, None),
            cast(Any, None),
            expected_metadata=expected,
            ppo=ppo.PPOConfig(rollout_length=4, groups=2, minibatches=2, epochs=1),
            method="tarmac",
        )
    result = marl_bgs.evaluate(
        marl_bgs.pool({str(run.final_actor): 1.0, "random": 1.0}),
        "random",
        num_episodes=2,
        num_envs=4,
        maps=[42],
        seed=19047101,
        metrics="priority",
        output_dir=tmp_path / "pool",
    )
    assert len(result.completed_episode_ids) == 2
    with pytest.raises(ValueError, match="TarMAC actors exchange messages"):
        marl_bgs.export_model(run.final_actor, tmp_path / "tarmac.zip")
    assert not (tmp_path / "tarmac.zip").exists()
    extended = training.extend_training(
        run.final_checkpoint,
        additional_env_steps=8,
        output_dir=tmp_path / "extended",
        partners=None,
    )
    child = checkpoints.read_checkpoint_details(extended.final_actor)
    assert child["messages"] == "blocked"
    assert child["metadata"]["env_steps"] == 40
