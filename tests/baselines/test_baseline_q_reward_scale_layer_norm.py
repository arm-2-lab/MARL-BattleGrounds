"""Check the Q learners' reward scale and PQN-VDN's LayerNorm choice.

Contracts checked here, all on CPU. ``reward_scale`` on QMIXConfig and
PQNConfig accepts only finite positive real numbers and PQNConfig's
``normalization`` only "batch_norm" or "layer_norm". One update with
``reward_scale`` 10 equals, bit for bit, the same update at scale 1 on a batch
whose rewards were multiplied by 10, for both QMIX and PQN-VDN, and differs
from the unscaled update. ``saved_settings`` leaves these later settings out
while they hold the value older runs used, so config_to_dict keeps historical
bytes, and the checkpoint settings checks accept a block saved before the
settings existed or one stating their old values, while a changed value is
written, read back and rejected against the old block. A LayerNorm network
keeps the same 4,596,512 parameters with only the input layer's 10,330
statistics, and its grouped rows equal the shared network in inference and,
with one group, in training mode. A LayerNorm actor exports with an explicit
"normalization" key (a BatchNorm export has none), refuses a wrong layout or a
non-PQN method, loads back to the LayerNorm hook with a distinct identity,
plays the same greedy actions and memory as the direct System on real 5v5
inputs, and is recognized as deterministic at epsilon 0. BatchNorm Systems keep
their original hook. The LayerNorm network matches JaxMARL's own ``layer_norm``
network in ``test_baseline_pqn_reference``. These checks prove software
contracts, not learning benefit or GPU cost.
"""

# pyright: reportPrivateUsage=false
import dataclasses
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.helpers.evaluation_fixtures import evaluation_env_config

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines import pqn, qmix
from marl_battlegrounds.baselines.config import PPOConfig, saved_settings
from marl_battlegrounds.evaluation._recording.sampling_evidence import (
    method_sampling_fact,
)
from marl_battlegrounds.tasks import balanced_spawn_configs
from marl_battlegrounds.training import checkpoints
from marl_battlegrounds.training.config import (
    TrainConfig,
    config_from_dict,
    config_to_dict,
)

type Tree = Any

_Q_SMALL = qmix.QMIXConfig(
    sample_batch_size=3, sample_sequence_length=5, min_buffer_size=5
)
_P_SMALL = pqn.PQNConfig(rollout_length=4, memory_window=2, epochs=1, num_minibatches=1)
_PROVENANCE: dict[str, object] = {
    "run_id": "layer-norm-export-test",
    "seed": 19100101,
    "env_steps": 0,
    "checkpoint_id": "0" * 64,
    "optimizer_steps": 0,
}


def _target(value: Array) -> Array:
    return value * 0.9


def _five_groups(value: Array) -> Array:
    return jnp.stack([value] * 5)


def _equal(actual: Tree, expected: Tree) -> None:
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _qmix_case() -> tuple[qmix.QMIXTrainState, qmix.QMIXBatch]:
    keys = jax.random.split(jax.random.key(19100001), 6)
    shape = (3, 5)
    mask = (jax.random.uniform(keys[1], (*shape, 5, 198)) > 0.4).at[..., 0].set(True)
    batch = qmix.QMIXBatch(
        jax.random.normal(keys[0], (*shape, 5, 7)),
        mask,
        jax.random.categorical(keys[2], jnp.where(mask, 0.0, -jnp.inf)).astype(
            jnp.int32
        ),
        jax.random.normal(keys[3], shape) * 0.3,
        jnp.zeros(shape, bool).at[1, 2].set(True),
        jnp.zeros(shape, bool).at[1, 1].set(True),
        jnp.ones(shape, bool),
        jnp.ones((*shape, 5), bool),
        jax.random.normal(keys[4], (*shape, 11)),
    )
    q = qmix.RecurrentQNetwork().init(
        keys[5],
        jnp.zeros((1, 5, qmix.QMIX_HIDDEN_SIZE)),
        jnp.zeros((1, 1, 5, 7)),
        jnp.zeros((1, 1, 5), bool),
        jnp.ones((1, 1, 5), bool),
    )
    mixer = qmix.QMixingNetwork().init(
        keys[5], jnp.zeros((1, 1, 5)), jnp.zeros((1, 1, 11))
    )
    target = jax.tree.map(_target, q)
    state = qmix.QMIXTrainState(
        q,
        target,
        mixer,
        mixer,
        qmix._optimizer(_Q_SMALL).init((q, mixer)),
        jnp.int32(3),
    )
    return state, batch


def _pqn_batch() -> pqn.PQNBatch:
    keys = jax.random.split(jax.random.key(19100002), 4)
    shape = (4, 2)
    mask = (jax.random.uniform(keys[1], (*shape, 5, 198)) > 0.4).at[..., 0].set(True)
    return pqn.PQNBatch(
        jax.random.normal(keys[0], (*shape, 5, 5165)),
        mask,
        jax.random.categorical(keys[2], jnp.where(mask, 0.0, -jnp.inf)).astype(
            jnp.int32
        ),
        jax.random.normal(keys[3], shape) * 0.3,
        jnp.zeros(shape, bool).at[0].set(True),
        jnp.zeros(shape, bool),
        jnp.ones(shape, bool),
        jnp.ones((*shape, 5), bool),
        jnp.zeros((2, 5, 512), jnp.float32),
    )


@pytest.mark.parametrize(
    ("record", "values"),
    (
        (qmix.QMIXConfig, {"reward_scale": 0.0}),
        (qmix.QMIXConfig, {"reward_scale": float("nan")}),
        (pqn.PQNConfig, {"reward_scale": -1.0}),
        (pqn.PQNConfig, {"reward_scale": True}),
        (pqn.PQNConfig, {"normalization": "group_norm"}),
        (pqn.PQNConfig, {"normalization": 1}),
    ),
)
def test_new_settings_reject_invalid_values(
    record: type[Any], values: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match=r"reward_scale|normalization"):
        record(**values)


def test_qmix_reward_scale_multiplies_the_rewards_one_update_learns_from() -> None:
    state, batch = _qmix_case()
    scaled = dataclasses.replace(_Q_SMALL, reward_scale=10.0)
    first = qmix.update_qmix(state, batch, config=scaled)
    second = qmix.update_qmix(
        state, batch._replace(rewards=batch.rewards * 10.0), config=_Q_SMALL
    )
    _equal(first, second)
    plain = qmix.update_qmix(state, batch, config=_Q_SMALL)
    assert float(plain[1].loss) != float(first[1].loss)


def test_pqn_reward_scale_multiplies_the_rewards_one_update_learns_from() -> None:
    state = pqn.initialize_pqn(
        jax.random.key(19100003), pqn=_P_SMALL, planned_learning_blocks=2
    )
    batch = _pqn_batch()
    scaled = dataclasses.replace(_P_SMALL, reward_scale=10.0)
    first = pqn.update_pqn(state, batch, pqn=scaled, planned_learning_blocks=2)
    second = pqn.update_pqn(
        state,
        batch._replace(rewards=batch.rewards * 10.0),
        pqn=_P_SMALL,
        planned_learning_blocks=2,
    )
    _equal(first, second)
    plain = pqn.update_pqn(state, batch, pqn=_P_SMALL, planned_learning_blocks=2)
    assert float(plain[1].loss) != float(first[1].loss)


def test_saved_settings_keep_old_bytes_and_old_blocks_still_match() -> None:
    ppo = PPOConfig()
    assert saved_settings(ppo) == dataclasses.asdict(ppo)
    old_qmix, old_pqn = (
        saved_settings(qmix.QMIXConfig()),
        saved_settings(pqn.PQNConfig()),
    )
    assert "reward_scale" not in old_qmix
    assert not {"reward_scale", "normalization"} & old_pqn.keys()
    changed = pqn.PQNConfig(reward_scale=10.0, normalization="layer_norm")
    assert saved_settings(changed)["reward_scale"] == 10.0
    assert saved_settings(changed)["normalization"] == "layer_norm"
    for settings in (pqn.PQNConfig(), changed):
        config = TrainConfig(
            method="pqn_vdn", num_envs=16, total_env_steps=16 * 140, pqn=settings
        )
        saved = config_to_dict(config)
        assert saved["pqn"] == saved_settings(settings)
        assert config_from_dict(saved) == config
    # A block saved before the settings existed, or one stating their old
    # values, matches those old values only.
    explicit = {**old_pqn, "reward_scale": 1.0, "normalization": "batch_norm"}
    checkpoints._check_pqn_settings(
        {"method": "pqn_vdn", "pqn": explicit}, pqn.PQNConfig()
    )
    checkpoints._check_pqn_settings(
        {"method": "pqn_vdn", "pqn": old_pqn}, pqn.PQNConfig()
    )
    checkpoints._check_qmix_settings(
        {"method": "qmix", "qmix": old_qmix}, qmix.QMIXConfig()
    )
    with pytest.raises(ValueError, match="PQN settings differ"):
        checkpoints._check_pqn_settings({"method": "pqn_vdn", "pqn": old_pqn}, changed)
    with pytest.raises(ValueError, match="QMIX settings differ"):
        checkpoints._check_qmix_settings(
            {"method": "qmix", "qmix": old_qmix}, qmix.QMIXConfig(reward_scale=2.0)
        )


def _forward(
    network: pqn.PQNInferenceVariables,
    features: Array,
    groups: Array | None,
    *,
    train: bool,
) -> tuple[Array, Array]:
    rows, games = features.shape[:2]
    arguments = (
        jnp.zeros((games, 5, 512), jnp.float32),
        features,
        jnp.zeros((rows, games), bool).at[0].set(True),
        jnp.ones((rows, games), bool),
        jnp.ones((rows, games, 5), bool),
    )
    module = pqn.PQNNetwork(normalization="layer_norm")
    variables = pqn._flax_variables(network)
    if train:
        output, _ = module.apply(
            variables,
            *arguments,
            train=True,
            actor_group=groups,
            mutable=["batch_stats"],
        )
        return cast(tuple[Array, Array], output)
    return cast(
        tuple[Array, Array],
        module.apply(variables, *arguments, train=False, actor_group=groups),
    )


def test_layer_norm_layout_and_grouped_rows_match_the_shared_network() -> None:
    template = pqn.pqn_actor_template(normalization="layer_norm")
    counts = [
        sum(
            int(np.prod(leaf.shape))
            for leaf in cast(list[Array], jax.tree.leaves(tree))
        )
        for tree in (template.params, template.batch_stats)
    ]
    assert counts == [4_596_512, 10_330]
    assert set(template.params) >= {"LayerNorm_0", "LayerNorm_1", "BatchNorm_0"}
    assert set(template.batch_stats) == {"BatchNorm_0"}
    shared = pqn._network_variables(
        jax.random.key(19100004), 1.0, normalization="layer_norm"
    )
    grouped = cast(pqn.PQNInferenceVariables, jax.tree.map(_five_groups, shared))
    features = jax.random.normal(jax.random.key(19100005), (3, 2, 5, 5165))
    mixed = jax.random.randint(jax.random.key(19100006), (3, 2, 5), 0, 5)
    for groups, train in ((mixed, False), (jnp.zeros((3, 2, 5), jnp.int32), True)):
        expected = _forward(shared, features, None, train=train)
        actual = _forward(grouped, features, groups, train=train)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_allclose(left, right, atol=2e-5, rtol=2e-5)


def _policy_inputs() -> Any:  # noqa: ANN401
    config = evaluation_env_config(team_sizes=(5, 5), max_steps=8)
    env = marl_bgs.make(
        "tdm",
        env_config=balanced_spawn_configs(config, num_envs=2),
        num_envs=2,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(19100007))
    return env.policy_inputs(observations, state)


def test_layer_norm_actor_exports_loads_and_keeps_batch_norm_hooks(
    tmp_path: Path,
) -> None:
    layer = pqn.initialize_pqn(
        jax.random.key(19100008),
        pqn=pqn.PQNConfig(normalization="layer_norm"),
        planned_learning_blocks=1,
    ).network
    batch = pqn.initialize_pqn(
        jax.random.key(19100008), planned_learning_blocks=1
    ).network
    with pytest.raises(ValueError, match="model schema"):
        checkpoints.export_system(
            layer,
            tmp_path / "wrong",
            metadata=_PROVENANCE,
            spawn_frame="left",
            method="pqn_vdn",
        )
    with pytest.raises(ValueError, match="normalization choice"):
        checkpoints.export_system(
            layer,
            tmp_path / "qmix",
            metadata=_PROVENANCE,
            spawn_frame="left",
            method="qmix",
            normalization="layer_norm",
        )
    exported = checkpoints.export_system(
        layer,
        tmp_path / "layer",
        metadata=_PROVENANCE,
        spawn_frame="left",
        method="pqn_vdn",
        normalization="layer_norm",
    )
    plain = checkpoints.export_system(
        batch,
        tmp_path / "batch",
        metadata=_PROVENANCE,
        spawn_frame="left",
        method="pqn_vdn",
    )
    layer_details = checkpoints.read_checkpoint_details(exported)
    batch_details = checkpoints.read_checkpoint_details(plain)
    assert layer_details["normalization"] == "layer_norm"
    assert "normalization" not in batch_details
    loaded = checkpoints.load_system(exported)
    assert loaded.apply is pqn._layer_norm_pqn_actor_apply(1.0, "left")
    assert checkpoints.load_system(plain).apply is pqn._pqn_actor_apply(1.0, "left")
    unbound = dict(layer_details, normalization="batch_norm")
    assert checkpoints._inference_digest(
        layer_details
    ) != checkpoints._inference_digest(unbound)
    direct = pqn.make_pqn_system(layer, normalization="layer_norm")
    assert direct.apply is loaded.apply
    assert method_sampling_fact(loaded)["determinism"] == "deterministic"
    with pytest.raises(ValueError, match="current actor input schema"):
        pqn.make_pqn_system(layer, normalization="layer_norm", actor_input_schema=1)
    inputs = _policy_inputs()
    memory = jnp.zeros((*inputs.active_mask.shape, pqn.PQN_HIDDEN_SIZE), jnp.float32)
    keys = jax.random.split(jax.random.key(19100009), 2)
    first = loaded.apply(loaded.variables, memory, inputs, keys)
    second = direct.apply(
        direct.variables, memory, inputs, jax.random.split(jax.random.key(19100010), 2)
    )
    _equal(first.actions, second.actions)
    _equal(first.next_memory, second.next_memory)
    # The LayerNorm network really ran: memory moved away from zero.
    assert float(jnp.abs(first.next_memory).sum()) > 0.0
