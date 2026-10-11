"""Check portable model files: export, safe reading, the loader System and jit/vmap.

Contracts checked with a tiny untrained per-agent Policy on CPU:
- a file loaded with load_portable_model plays the same actions as the Policy
  it was made from, through apply_systems, and inside pool and team;
- the loaded System works under jit, under one and two levels of vmap, with
  batched weights (the lax.map fallback) and under lax.cond inside vmap;
- a Policy that calls a loaded file's program exports again and gives the
  same actions (the JAX-upgrade refresh route);
- exporting twice gives identical bytes, with no file paths inside;
- a shared file names no folder on the exporting computer: a saved actor
  description's local opponent folder becomes "a local folder (not shared)"
  (a released name is kept), the old ID moves to metadata.derived_from, the
  new ID follows the usual rule, the run's own ID is unchanged, and a
  description with nothing local comes back byte for byte; a Policy named by
  a home folder with a GPU id as its checkpoint label exports with labels and
  passes the shared local-path rule;
- tampered, padded and non-format files are refused or recognized;
- host Policies are refused;
- the loader's apply and init hooks keep their pinned code digests, because
  saved runs record them (checked on Python 3.14, which made the pins: the
  digest is of Python bytecode, which differs between versions).
The six real Season 0 models are checked against their originals by the
release proof in artifacts, not here.
"""

# pyright: reportPrivateUsage=false
import hashlib
import json
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import portable_models
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.environment import Environment
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    system_inputs,
)
from marl_battlegrounds.evaluation.recording_identity import (
    _callable_evidence,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput

APPLY_CODE_DIGEST = "52b2fc87f534adfb1fa4bb3850db3baea0b58f07be9e7991674c8940462b9e8f"
INIT_CODE_DIGEST = "f4924e58bac5a9b34c2132c7e6103440f77202c582b853e93d04708476c6b742"


def _apply(
    variables: PolicyTree,
    carry: jax.Array,
    actor: ActorInput,
    mask: ActionMask,
    key: jax.Array,
) -> tuple[ActorAction, jax.Array]:
    del key
    score = jnp.where(mask.move_mask, variables["move"] + carry[0], -jnp.inf)
    pair = jnp.argmax(mask.select_target_use_ultimate_joint_mask.reshape(-1))
    action = ActorAction(
        jnp.argmax(score).astype(jnp.int32),
        (pair // 2).astype(jnp.int32),
        (pair % 2).astype(jnp.int32),
    )
    after = carry * 0.5 + variables["memory"] * actor.observation.self_features[:4]
    return action, after


def _policy(scale: float = 1.0) -> marl_bgs.Policy:
    return marl_bgs.Policy(
        "tiny",
        _apply,
        variables={
            "move": jnp.linspace(0.0, scale, 9, dtype=jnp.float32),
            "memory": jnp.full((4,), 0.1 * scale, jnp.float32),
        },
        initial_carry=jnp.zeros((4,), jnp.float32),
    )


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    folder = tmp_path_factory.mktemp("portable")
    first = portable_models.export_model(_policy(), folder / "tiny.zip")
    second = portable_models.export_model(_policy(2.0), folder / "tiny-2.zip")
    return folder, first, second


@pytest.fixture(scope="module")
def game() -> tuple[Any, Any, Any]:
    env = marl_bgs.make("tdm", map_id=0, num_envs=2, metrics="none")
    observations, state = env.reset(jax.random.key(0))
    return env, observations, state


def _rollout(env: Environment, system: System, steps: int = 6) -> np.ndarray:
    alpha = marl_bgs.shared_policy(marl_bgs.policy("tdm-alpha"))
    observations, state = env.reset(jax.random.key(1))
    memory = marl_bgs.init_systems(
        system, alpha, observations, state, jax.random.key(2)
    )
    key = jax.random.key(3)
    played: list[np.ndarray] = []
    for _ in range(steps):
        key, act, step = jax.random.split(key, 3)
        actions, memory, _ = marl_bgs.apply_systems(
            system, alpha, memory, observations, state, act
        )
        played.append(np.asarray(actions.move))
        observations, state, _, _, _ = env.step(step, state, actions)
    return np.stack(played)


def _stack(*trees: PolicyTree) -> PolicyTree:
    return jax.tree.map(_stack_leaves, *trees)


def _stack_leaves(*leaves: jax.Array) -> jax.Array:
    return jnp.stack(leaves)


def _row(tree: PolicyTree, *index: int) -> PolicyTree:
    def take(leaf: jax.Array) -> jax.Array:
        return leaf[index]

    return jax.tree.map(take, tree)


def _same(first: PolicyTree, second: PolicyTree) -> bool:
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True)
    )


def test_loaded_file_plays_like_its_policy(
    made: tuple[Path, Path, Path], game: tuple[Any, Any, Any]
) -> None:
    env, _, _ = game
    loaded = portable_models.load_portable_model(made[1])
    assert np.array_equal(
        _rollout(env, loaded), _rollout(env, marl_bgs.shared_policy(_policy()))
    )
    second = portable_models.load_portable_model(made[2])
    assert second.apply is loaded.apply
    mixed = marl_bgs.pool({loaded: 0.5, second: 0.5})
    split = marl_bgs.team(loaded, "tdm-alpha", slots=[[0, 1, 2], [3, 4]])
    assert _rollout(env, mixed).shape == _rollout(env, split).shape == (6, 2, 10)


def test_loaded_system_works_under_jit_and_vmap(
    made: tuple[Path, Path, Path], game: tuple[Any, Any, Any]
) -> None:
    _, observations, state = game
    system = portable_models.load_portable_model(made[1])
    inputs = system_inputs(observations, state, team=0)
    keys = jax.random.split(jax.random.key(4), 2)
    assert system.init is not None
    memory = system.init(system.variables, inputs, keys) + 0.25
    eager = system.apply(system.variables, memory, inputs, keys)
    assert _same(jax.jit(system.apply)(system.variables, memory, inputs, keys), eager)

    def twice(tree: PolicyTree) -> PolicyTree:
        return _stack(tree, tree)

    once = jax.vmap(system.apply, in_axes=(None, 0, 0, 0))(
        system.variables, twice(memory), twice(inputs), twice(keys)
    )
    assert _same(_row(once, 1), eager)
    nested = jax.vmap(
        jax.vmap(system.apply, in_axes=(None, 0, 0, 0)), in_axes=(None, 0, 0, 0)
    )(system.variables, twice(twice(memory)), twice(twice(inputs)), twice(twice(keys)))
    assert _same(_row(nested, 1, 0), eager)
    other = portable_models.load_portable_model(made[2])
    weights = _stack(system.variables, other.variables)
    batched = jax.vmap(system.apply, in_axes=(0, None, None, None))(
        weights, memory, inputs, keys
    )
    assert _same(
        _row(batched, 1),
        other.apply(other.variables, memory, inputs, keys),
    )

    def chosen(
        flag: jax.Array, carry: jax.Array, view: PolicyTree, key: jax.Array
    ) -> jax.Array:
        def play() -> jax.Array:
            return system.apply(system.variables, carry, view, key).actions.move

        def rest() -> jax.Array:
            return jnp.zeros_like(view.active_mask, jnp.int32)

        return cast(jax.Array, jax.lax.cond(flag, play, rest))

    moves = jax.vmap(chosen)(
        jnp.array([True, False]), twice(memory), twice(inputs), twice(keys)
    )
    assert np.array_equal(moves[0], eager.actions.move) and not moves[1].any()


def test_a_policy_calling_a_loaded_file_exports_again(
    made: tuple[Path, Path, Path], game: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    _, observations, state = game
    original = portable_models.load_portable_model(made[1])
    exported = jax.export.deserialize(
        bytearray(zipfile.ZipFile(made[1]).read("decide.jaxexport"))
    )
    call = cast(Callable[..., PolicyTree], exported.call)
    program = portable_models._vmappable(call)  # pyright: ignore[reportPrivateUsage]

    def refreshed(
        variables: PolicyTree,
        carry: jax.Array,
        actor: ActorInput,
        mask: ActionMask,
        key: jax.Array,
    ) -> tuple[ActorAction, jax.Array]:
        def rows(tree: PolicyTree) -> tuple[jax.Array, ...]:
            return tuple(leaf[None] for leaf in jax.tree.leaves(tree))

        move, target, ultimate, after, _ = program(
            tuple(variables),
            (carry[None],),
            rows(actor),
            rows(mask),
            jax.random.key_data(key)[None],
        )
        return ActorAction(move[0], target[0], ultimate[0]), after[0][0]

    wrapper = marl_bgs.Policy(
        "tiny (refreshed)",
        refreshed,
        variables=original.variables.weights,
        initial_carry=jnp.zeros((4,), jnp.float32),
    )
    again = portable_models.load_portable_model(
        portable_models.export_model(wrapper, tmp_path / "again.zip")
    )
    inputs = system_inputs(observations, state, team=0)
    keys = jax.random.split(jax.random.key(5), 2)
    assert original.init is not None
    memory = original.init(original.variables, inputs, keys)
    assert _same(
        again.apply(again.variables, memory, inputs, keys).actions,
        original.apply(original.variables, memory, inputs, keys).actions,
    )


def test_files_are_reproducible_checked_and_recognized(
    made: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    folder, first, _ = made
    again = portable_models.export_model(_policy(), tmp_path / "again.zip")
    assert again.read_bytes() == first.read_bytes()
    assert b"/home/" not in first.read_bytes() and b"/tmp/" not in first.read_bytes()
    record = portable_models.read_portable_model(first)
    assert record.model["format_version"] == 1 and record.model["method"] == "policy"
    assert portable_models.is_portable_model_file(first)
    assert not portable_models.is_portable_model_file(folder)
    with zipfile.ZipFile(first) as source:
        parts = {name: source.read(name) for name in source.namelist()}
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w") as target:
        for name, data in parts.items():
            if name == "weights.npz":
                data = data[:100] + bytes([data[100] ^ 1]) + data[101:]
            target.writestr(name, data)
    with pytest.raises(ValueError, match="does not match its SHA-256"):
        portable_models.read_portable_model(tampered)
    padded = tmp_path / "padded.zip"
    with zipfile.ZipFile(padded, "w") as target:
        for name, data in {**parts, "../extra.txt": b"x"}.items():
            target.writestr(name, data)
    with pytest.raises(ValueError, match="expected the parts"):
        portable_models.read_portable_model(padded)
    with pytest.raises(ValueError, match="built-in actor"):
        portable_models.portable_actor_variables(first)
    assert json.loads(parts["model.json"])["parts"]["weights.npz"]


def test_shared_files_name_no_folder_on_the_exporting_computer(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.released_models import _local_path_findings
    from marl_battlegrounds.training._checkpoint_metadata import _json_bytes

    def signed(body: dict[str, Any]) -> bytes:
        identity = hashlib.sha256(_json_bytes(body)).hexdigest()
        return _json_bytes({**body, "checkpoint_id": identity})

    members = [
        {"name": "old", "reference": "/home/sam/runs/old"},
        {"name": "alpha", "reference": "tdm-alpha"},
    ]
    body = {"kind": "actor", "metadata": {"checkpoint_id": "run-id"}}
    raw = signed(
        {**body, "metadata": {"checkpoint_id": "run-id", "opponent_members": members}}
    )
    shared = json.loads(portable_models._shareable_description(raw))
    metadata = shared["metadata"]
    assert [m["reference"] for m in metadata["opponent_members"]] == [
        "a local folder (not shared)",
        "tdm-alpha",
    ]
    assert metadata["derived_from"] == json.loads(raw)["checkpoint_id"]
    assert metadata["checkpoint_id"] == "run-id"
    assert signed({k: v for k, v in shared.items() if k != "checkpoint_id"}) == (
        portable_models._shareable_description(raw)
    )
    clean = signed(body)
    assert portable_models._shareable_description(clean) is clean

    gpu = "GPU-6b11a0c4-14e8-6782-8932-1df56d599796"
    policy = marl_bgs.Policy(
        "/home/sam/runs/tiny",
        _apply,
        variables=_policy().variables,
        initial_carry=_policy().initial_carry,
        checkpoint=gpu,
    )
    path = portable_models.export_model(policy, tmp_path / "named.zip")
    assert _local_path_findings(path.name, path.read_bytes()) == []
    record = portable_models.read_portable_model(path)
    assert record.actor_details["name"] == record.model["display_name"]
    assert record.actor_details["name"] == "a local folder (not shared)"
    assert record.model["checkpoint_id"] == record.actor_details["checkpoint"]
    assert record.model["checkpoint_id"] == "GPU-<id>"


def test_host_policies_are_refused(tmp_path: Path) -> None:
    host = marl_bgs.Policy("host", _apply, execution="host")
    with pytest.raises(ValueError, match="only JAX Policies"):
        portable_models.export_model(host, tmp_path / "host.zip")
    assert not Path(tmp_path / "host.zip").exists()


@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="Saved code digests reproduce only on the Python that made them",
)
def test_loader_hooks_keep_their_recorded_code_digest(
    made: tuple[Path, Path, Path],
) -> None:
    system = portable_models.load_portable_model(made[1])
    apply, init = _callable_evidence(system.apply), _callable_evidence(system.init)
    assert apply is not None and apply["code_digest"] == APPLY_CODE_DIGEST
    assert init is not None and init["code_digest"] == INIT_CODE_DIGEST
