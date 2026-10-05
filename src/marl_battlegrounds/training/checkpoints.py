"""Save complete PPO, QMIX and PQN-VDN learner boundaries and load frozen actors.

The runner owns its run lock, immutable settings and training logs. This module
owns synchronous Orbax payloads, file integrity, publication and checked numerical
restore. Restore reads everything before the optional recording recovery helper
can rewind M8 files. Actor exports need neither training content nor a learner.
All methods share the file layout; the saved model schema names the method.
A QMIX learner checkpoint's actor payload holds only the Q-network parameters;
its epsilon, targets, mixer, optimizer and replay are in the state payload.
A PQN-VDN actor payload is Flax's ``{"params", "batch_stats"}`` dictionary,
built and read by name (Orbax stores NamedTuples as dictionaries); its
epsilon, optimizer state and kept rows are in the state payload. QMIX and
PQN-VDN exports and loaded Systems are greedy (epsilon 0, first legal
maximum).

Artifacts saved before Red Zone use actor input schema 1 (5,164 features).
load_system still loads them, through the schema-1 System hooks, so they play
exactly as before and keep their identities. They cannot be exported again,
and their learner checkpoints cannot resume training in this version.
"""

from __future__ import annotations

# Shared private metadata owners retain their historical aliases here.
# pyright: reportPrivateUsage=false
import json
import math
import os
import shutil
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache, partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp  # pyright: ignore[reportMissingTypeStubs]

from marl_battlegrounds.baselines.config import (
    DEFAULT_PPO_CONFIG,
    DEFAULT_PQN_CONFIG,
    DEFAULT_QMIX_CONFIG,
    PPOConfig,
    PQNConfig,
    QMIXConfig,
    _pqn_normalization,  # pyright: ignore[reportPrivateUsage]
    with_later_settings,
)
from marl_battlegrounds.baselines.inputs import (
    ACTOR_FEATURE_SIZE,
    ACTOR_FEATURE_SIZES,
    ACTOR_INPUT_SCHEMA_VERSION,
)
from marl_battlegrounds.baselines.methods import (
    is_ppo_method,
    validate_training_method,
)
from marl_battlegrounds.baselines.ppo import (
    _parameter_sharing,  # pyright: ignore[reportPrivateUsage]
    initialize_ppo,
    make_ppo_system,
    validate_ppo_method,
)
from marl_battlegrounds.baselines.pqn import (
    PQNInferenceVariables,
    _nonnegative_variances,  # pyright: ignore[reportPrivateUsage]
    make_pqn_system,
    pqn_actor_template,
)
from marl_battlegrounds.baselines.qmix import (
    QMIX_TIE_RULE,
    make_qmix_system,
    qmix_actor_template,
)
from marl_battlegrounds.evaluation.recording_identity import tree_digest
from marl_battlegrounds.training._checkpoint_metadata import (
    _ACTOR_DESCRIPTION as _ACTOR_DESCRIPTION,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _ACTOR_INPUT_1_SCHEMAS as _ACTOR_INPUT_1_SCHEMAS,
)
from marl_battlegrounds.training._checkpoint_metadata import _CONTEXT as _CONTEXT
from marl_battlegrounds.training._checkpoint_metadata import (
    _DESCRIPTION as _DESCRIPTION,
)
from marl_battlegrounds.training._checkpoint_metadata import _MODELS as _MODELS
from marl_battlegrounds.training._checkpoint_metadata import (
    _PQN_COUNTERS as _PQN_COUNTERS,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _PQN_SCHEMAS as _PQN_SCHEMAS,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _QMIX_COUNTERS as _QMIX_COUNTERS,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _QMIX_SCHEMAS as _QMIX_SCHEMAS,
)
from marl_battlegrounds.training._checkpoint_metadata import _SCHEMAS as _SCHEMAS
from marl_battlegrounds.training._checkpoint_metadata import (
    _actor_input_scale as _actor_input_scale,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _actor_normalization as _actor_normalization,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _actor_parameter_sharing as _actor_parameter_sharing,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _actor_spawn_frame as _actor_spawn_frame,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _artifact_identity as _artifact_identity,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _check_actor_origin as _check_actor_origin,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _check_context as _check_context,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _config_input_scale as _config_input_scale,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _config_method as _config_method,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _config_spawn_frame as _config_spawn_frame,
)
from marl_battlegrounds.training._checkpoint_metadata import _digest as _digest
from marl_battlegrounds.training._checkpoint_metadata import _directory as _directory
from marl_battlegrounds.training._checkpoint_metadata import (
    _inference_digest as _inference_digest,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _input_scale as _input_scale,
)
from marl_battlegrounds.training._checkpoint_metadata import _json_bytes as _json_bytes
from marl_battlegrounds.training._checkpoint_metadata import _object as _object
from marl_battlegrounds.training._checkpoint_metadata import _read_json as _read_json
from marl_battlegrounds.training._checkpoint_metadata import _relative as _relative
from marl_battlegrounds.training._checkpoint_metadata import (
    _require_current_schemas as _require_current_schemas,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _schema_method as _schema_method,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _settings_block as _settings_block,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _spawn_frame as _spawn_frame,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    artifact_directory as artifact_directory,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    artifact_name as artifact_name,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    checkpoint_schemas as checkpoint_schemas,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    read_checkpoint_description as read_checkpoint_description,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    saved_training_config as saved_training_config,
)
from marl_battlegrounds.training._compilation import execution_identity

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.policy_execution import System
    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.training._continuation_schedules import LearnerContinuation
    from marl_battlegrounds.training.collection import TrainingCollection
    from marl_battlegrounds.training.learner import LearnerState
    from marl_battlegrounds.training.pqn_learner import PQNLearnerState
    from marl_battlegrounds.training.qmix_learner import QMIXLearnerState

type Tree = Any
_COPY_BYTES = 1024 * 1024
# The exact schema dictionaries artifacts carried before Red Zone (actor input
# schema 1, and training state schema 1 where present), frozen. load_system
# still reads them; export and resume refuse them.


@dataclass(frozen=True)
class RestoredCheckpoint:
    """Hold a fully checked learner before any recording or log rewind.

    state is the complete restored numerical learner: a PPO LearnerState for
    the four PPO methods, a QMIXLearnerState for ``method="qmix"`` or a
    PQNLearnerState for ``method="pqn_vdn"``, as the saved model schema says.
    It is typed ``Any`` so existing PPO code keeps reading its fields without
    casts; check the method (or use isinstance) when a caller handles more
    than one. details is checked JSON metadata; path is the
    immutable checkpoint directory. This object owns no run lock or writer.
    Keep the runner's lock while installing its recovery.
    """

    state: Any
    details: dict[str, Any]
    path: Path


def _check_qmix_settings(config: object, settings: QMIXConfig) -> None:
    """Require a saved QMIX config's settings to equal the given settings.

    config is the saved JSON learner config; settings is the QMIXConfig the
    caller will save or continue with. Compares every field (for example
    gamma, q_lr, tau, the target rule, T, C, S, M, the exploration clock and
    reward_scale) after the same JSON round trip that config_to_dict applies.
    ``with_later_settings`` first fills a setting the block omits because it
    was saved before that setting existed, so an old block matches
    reward_scale 1.0. Raises ValueError on any difference. Reads no arrays and
    writes nothing.
    """
    expected = json.loads(json.dumps(asdict(settings)))
    if with_later_settings(QMIXConfig, _settings_block(config)) != expected:
        raise ValueError("Checkpoint config QMIX settings differ from the given ones")


def _check_pqn_settings(config: object, settings: PQNConfig) -> None:
    """Require a saved PQN-VDN config's settings to equal the given settings.

    config is the saved JSON learner config; settings is the PQNConfig the
    caller will save or continue with. Compares every field (T, H, epochs,
    minibatches, rates, schedules, gamma, lambda, input_scale, spawn_frame,
    reward_scale and normalization) after the same JSON round trip that
    config_to_dict applies, so a missing or partial "pqn" block fails.
    ``with_later_settings`` first fills only the settings added later, so a
    block saved before reward_scale and normalization existed matches 1.0
    and "batch_norm". Raises ValueError. Reads no arrays and writes nothing.
    """
    expected = json.loads(json.dumps(asdict(settings)))
    if with_later_settings(PQNConfig, _settings_block(config)) != expected:
        raise ValueError("Checkpoint config PQN settings differ from the given ones")


def _config_value_normalization(config: object) -> bool:
    """Read the saved critic setting; absence means historical raw-value training.

    config must be a JSON learner-config object. A present setting must be a
    Python bool; PPOConfig owns its validation. No saved input is changed.
    """
    ppo = _object(_object(config, "Learner config").get("ppo", {}), "PPO config")
    return PPOConfig(
        value_normalization=cast(bool, ppo.get("value_normalization", False))
    ).value_normalization


def _config_red_zone_depth(config: object) -> float | None:  # pyright: ignore[reportUnusedFunction]
    """Read the Red Zone depth a saved training config declares.

    Parameters
    ----------
    config : object
        A saved JSON training config (metadata.config of a checkpoint).

    Returns
    -------
    float or None
        The saved red_zone_depth in map units, or None when the config has no
        such key because it was saved before the Red Zone rule (one point per
        death). None tells validation helpers to expect the older record
        layouts, which carry no depth.

    Raises
    ------
    ValueError
        config is not a JSON object, or its red_zone_depth is not a float.
        The value's range was checked by TrainConfig when the run started.

    Notes
    -----
    Host-only; reads no file and changes nothing. Use this where the saved
    layout matters (validation task and Random record checks). A config that
    must be rebuilt as a TrainConfig uses saved_training_config, which reads a
    missing depth as 0.0 instead.
    """
    value = _object(config, "Learner config").get("red_zone_depth")
    if value is None:
        return None
    if type(value) is not float:
        raise ValueError("Saved red_zone_depth must be a float")
    return value


def _sync_dir(path: Path) -> None:
    """Persist local directory entries; propagate durability failures."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    """Replace JSON durably and remove only this call's unpublished temporary.

    Keep the checkpoint owner's exact canonical bytes. path.parent must exist.
    After replacement, a later directory-sync failure leaves the complete new
    file visible; never roll it back. Serialization and filesystem errors
    propagate. This call does not remove or replace any other temporary file.
    """
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    owned = False
    try:
        with temporary.open("xb") as stream:
            owned = True
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


@contextmanager
def _pending_write(
    directory: Path, *, kind: str, attempt_id: object = None
) -> Generator[Path]:
    """Own one unpublished payload tree until its caller renames or rejects it.

    directory is an existing checked local folder; kind is learner or actor.
    attempt_id is optional provenance, outside the immutable artifact. Record
    this process's PID, start and boot identity in a sibling receipt before any
    payload write. The receipt never enters the payload inventory or digest.
    The caller closes all array writers before leaving this context. Remove
    only our temporary tree and receipt on exit; an already renamed artifact
    remains untouched. Errors propagate. If cleanup fails, name the remaining
    path and preserve the original exception as context. No lock is acquired:
    the runner/library caller retains its existing single-writer contract.
    """
    from marl_battlegrounds.training._run_io import process_identity

    temporary = directory / f".pending-{uuid4().hex}"
    receipt = directory / f"{temporary.name}.owner.json"
    owns_tree = owns_receipt = False
    try:
        temporary.mkdir()
        owns_tree = True
        with receipt.open("xb"):
            owns_receipt = True
        _atomic_json(
            receipt,
            {
                "schema_version": 1,
                "temporary_name": temporary.name,
                "kind": kind,
                "attempt_id": attempt_id,
                "writer": process_identity(),
            },
        )
        yield temporary
    finally:
        try:
            if owns_tree and (temporary.exists() or temporary.is_symlink()):
                shutil.rmtree(temporary)
            if owns_receipt:
                receipt.unlink(missing_ok=True)
            if owns_tree or owns_receipt:
                _sync_dir(directory)
        except OSError as error:
            raise OSError(
                f"Could not finish cleanup for unpublished write at {temporary}: "
                f"{error}"
            ) from error


def _pending_writer_status(writer: Mapping[str, Any]) -> str:
    """Check a saved process identity without signalling or changing a process.

    The caller has checked the receipt fields. A different Linux boot or start
    identity proves the original writer is gone even if its PID was reused.
    A missing /proc read alone proves nothing; signal zero checks existence
    only. Return dead, alive or unknown. Permission/read failures preserve data.
    """
    from marl_battlegrounds.training._run_io import process_identity

    current = process_identity(writer["pid"])
    if current["start_ticks"] is not None and current["boot_id"] is not None:
        return "alive" if current == writer else "dead"
    try:
        os.kill(writer["pid"], 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError, OSError:
        pass
    return "unknown"


def _cleanup_pending_writes(directory: Path) -> Generator[dict[str, str]]:  # pyright: ignore[reportUnusedFunction]
    """Remove recognized abandoned writes under an already-held owner lock.

    directory is a run's checkpoints, actors or past_copies folder, or a library
    folder held by its existing owner. Missing folders need no work. Check direct
    children only. Receipt fields, sibling names and original writer death must
    all be known before deletion. Never follow links or remove published models.
    Live/unknown writers, old unowned temporaries and malformed receipts remain
    with a plain-English reason in the returned records. The caller saves these
    records through its existing event/report writer before requesting the next
    item. This keeps earlier removals recorded if a later cleanup fails.
    Cleanup errors name their
    path and propagate; no broad glob deletion or new lock is introduced.
    """
    if not directory.exists():
        return
    _directory(directory)
    names = {
        entry.name.removesuffix(".owner.json")
        for entry in directory.iterdir()
        if entry.name.startswith(".pending-") or entry.name.endswith(".tmp")
    }
    for name in sorted(names):
        path = directory / name
        receipt = directory / f"{name}.owner.json"
        reason = "No recognized ownership receipt; preserved for inspection"
        token = name.removeprefix(".pending-")
        known_name = (
            name.startswith(".pending-")
            and len(token) == 32
            and all(character in "0123456789abcdef" for character in token)
        )
        status, original = "preserved", "unknown"
        if known_name and not path.is_symlink() and not receipt.is_symlink():
            try:
                value = _object(
                    json.loads(receipt.read_text()), "Pending write receipt"
                )
                writer = _object(value.get("writer"), "Pending writer")
                if (
                    type(value["schema_version"]) is not int
                    or value["schema_version"] != 1
                    or value["temporary_name"] != name
                    or value["kind"] not in ("learner", "actor")
                    or set(writer) != {"pid", "start_ticks", "boot_id"}
                    or type(writer["pid"]) is not int
                    or not 0 < writer["pid"] <= 2**31 - 1
                    or not isinstance(writer["start_ticks"], str)
                    or not writer["start_ticks"].isdigit()
                    or not isinstance(writer["boot_id"], str)
                    or str(UUID(writer["boot_id"])) != writer["boot_id"]
                ):
                    raise ValueError("Incomplete write ownership")
                if path.exists() and not path.is_dir():
                    raise ValueError("Pending payload is not a directory")
                original = _pending_writer_status(writer)
                reason = f"Original writer is {original}; temporary data preserved"
            except (ValueError, TypeError, KeyError, OSError) as error:
                reason = f"Ownership could not be checked; preserved: {error}"
        if original == "dead":
            try:
                if path.exists():
                    shutil.rmtree(path)
                receipt.unlink()
                _sync_dir(directory)
            except OSError as error:
                raise OSError(
                    f"Could not clean abandoned write at {path}: {error}"
                ) from error
            status, reason = "removed", "Original writer is gone"
        yield {"path": str(path), "status": status, "reason": reason}


def _file_hash(path: Path) -> str:
    """Hash a regular file using bounded host memory."""
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(_COPY_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, object]:
    """Describe every payload file while rejecting links and special files."""
    result: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint payload must not contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Checkpoint payload must contain regular files")
        result[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": _file_hash(path),
        }
    return result


def _sync_tree(root: Path) -> None:
    """Persist every published file and directory before exposing its name."""
    paths = sorted(root.rglob("*"))
    for path in paths:
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in reversed(paths):
        if path.is_dir():
            _sync_dir(path)
    _sync_dir(root)


def _layout(tree: Tree) -> list[dict[str, object]]:
    """Describe logical array paths without serializing Python types or trees."""
    rows: list[dict[str, object]] = []
    flattened = cast(
        list[tuple[tuple[Any, ...], Any]],
        jax.tree_util.tree_flatten_with_path(tree)[0],
    )
    for path, value in flattened:
        keys = []
        for key in path:
            if isinstance(key, jax.tree_util.GetAttrKey):
                keys.append({"field": key.name})
            elif isinstance(key, jax.tree_util.SequenceKey):
                keys.append({"index": key.idx})
            elif isinstance(
                key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)
            ):
                keys.append({"key": key.key})
            else:
                raise TypeError("Unsupported numerical checkpoint field path")
        rows.append(
            {"path": keys, "shape": list(value.shape), "dtype": str(value.dtype)}
        )
    return rows


def _is_actor(row: dict[str, object], method: str = "mappo") -> bool:
    """Identify the learner's actor payload rows by their exact record path.

    PPO saves the whole current actor tree. QMIX saves only its Q-network
    parameters (current_variables.params) and PQN-VDN only its network
    (current_variables.network: parameters and statistics); the epsilon leaf
    stays in the state payload. method defaults to MAPPO.
    """
    prefix = [{"field": "carry"}, {"field": "history"}, {"field": "current_variables"}]
    path = cast(list[object], row["path"])
    if method == "qmix":
        return path[:4] == [*prefix, {"field": "params"}]
    if method == "pqn_vdn":
        return path[:4] == [*prefix, {"field": "network"}]
    return path[:3] == prefix


def _pqn_actor_item(network: PQNInferenceVariables) -> dict[str, Tree]:
    """Return PQN-VDN's on-disk actor item, Flax's variables dictionary.

    network is the in-memory PQNInferenceVariables (arrays or shape-only
    leaves). The item is ``{"params": ..., "batch_stats": ...}``, built by
    name, because Orbax stores a NamedTuple's metadata as a dictionary. Its
    leaves come in sorted-key order (statistics first), unlike the
    NamedTuple; :func:`_pqn_network` reverses this by name.
    """
    return {"params": network.params, "batch_stats": network.batch_stats}


def _pqn_network(item: Mapping[str, Tree]) -> PQNInferenceVariables:
    """Rebuild PQNInferenceVariables from an actor item by name.

    item must hold exactly "params" and "batch_stats". The result's leaves
    are in NamedTuple order (parameters first), the order of the learner's
    full-state rows, so restore can reinsert them without swapping the two.
    Raises ValueError for other keys.
    """
    if set(item) != {"params", "batch_stats"}:
        raise ValueError("PQN-VDN actor item needs exactly params and batch_stats")
    return PQNInferenceVariables(params=item["params"], batch_stats=item["batch_stats"])


def _actor_payload(state: Tree, method: str = "mappo") -> Tree:
    """Return the tree saved in actor/.

    PPO's actor tree, QMIX's Q-network parameters, or PQN-VDN's actor item
    (``_pqn_actor_item`` of the current network).
    """
    current = state.carry.history.current_variables
    if method == "pqn_vdn":
        return _pqn_actor_item(current.network)
    return current.params if method == "qmix" else current


def _state_payload(state: Tree, method: str = "mappo") -> dict[str, Any]:
    """Flatten state excluding the actor payload and zero-element leaves.

    state is a PPO, QMIX or PQN-VDN learner, or a shape-only template of one;
    method selects the actor rows (see _is_actor). Orbax rejects zero-element
    arrays.
    Their exact path, shape and dtype remain in the checked layout; restore
    rebuilds their empty storage from the template.
    """
    return {
        f"leaf_{index:05d}": leaf
        for index, (leaf, row) in enumerate(
            zip(jax.tree.leaves(state), _layout(state), strict=True)
        )
        if not _is_actor(row, method) and leaf.size
    }


def _counters(state: Tree, games: int, method: str) -> dict[str, int]:
    """Return a learner's saved counters as exact Python integers.

    PPO records updates and env_steps. QMIX and PQN-VDN add completed_blocks
    and learning_blocks; their "updates" means optimizer steps.
    """
    counters = {
        "updates": int(state.completed_updates),
        "env_steps": int(state.carry.progress.rounds) * games,
    }
    if method in ("qmix", "pqn_vdn"):
        counters["completed_blocks"] = int(state.completed_blocks)
        counters["learning_blocks"] = int(state.learning_blocks)
    return counters


def _check_qmix_counters(
    counters: object,
    qmix: QMIXConfig,
    games: int,
    *,
    continuation: LearnerContinuation | None = None,
) -> None:
    """Check saved QMIX counters from metadata alone, before arrays are read.

    counters must hold exactly updates, env_steps, completed_blocks and
    learning_blocks as nonnegative plain integers, env_steps a whole number of
    rounds, updates equal to learning_blocks * epochs, and a reachable
    readiness-aware block sequence. After a reward change, that sequence starts
    at the saved reset offset; cumulative counters remain unchanged. Raises
    ValueError. continuation is the checked child declaration, or None.
    """
    from marl_battlegrounds.training.qmix_learner import (
        _block_counts_possible,  # pyright: ignore[reportPrivateUsage]
    )

    value = _object(counters, "Checkpoint counters")
    if set(value) != _QMIX_COUNTERS or any(
        type(item) is not int or item < 0 for item in value.values()
    ):
        raise ValueError("QMIX checkpoint counters are malformed")
    if value["env_steps"] % games or value["updates"] != (
        value["learning_blocks"] * qmix.epochs
    ):
        raise ValueError("QMIX checkpoint counters are impossible")
    reset = None if continuation is None else continuation.reward_reset
    if not _block_counts_possible(
        value["env_steps"] // games - (0 if reset is None else reset.rounds),
        value["completed_blocks"] - (0 if reset is None else reset.blocks),
        value["learning_blocks"] - (0 if reset is None else reset.learning_blocks),
        rollout_length=qmix.rollout_length,
        minimum=qmix.min_buffer_size,
    ):
        raise ValueError("QMIX checkpoint counters are impossible")


def _check_pqn_counters(
    counters: object,
    pqn: PQNConfig,
    games: int,
    total_rounds: int,
    *,
    continuation: LearnerContinuation | None = None,
) -> None:
    """Check saved PQN-VDN counters from metadata alone, before arrays are read.

    Parameters
    ----------
    counters : object
        The description's counters: exactly updates, env_steps,
        completed_blocks and learning_blocks as nonnegative plain integers.
    pqn : PQNConfig
        Settings giving T, H, epochs and minibatches.
    games : int
        Number of games B; env_steps must be a whole number of rounds r.
    total_rounds : int
        The run's declared total rounds R.

    Raises
    ------
    ValueError
        A counter is malformed, or r is not a reachable collection boundary
        (``validation._collection_boundary`` with W initial rounds), or the
        blocks are not its ordinal, or learning_blocks is not
        ``max(0, ordinal - ceil(W / T))``, or updates is not
        ``learning_blocks * epochs * num_minibatches``.

    Notes
    -----
    Host-only arithmetic; totals may exceed int32 and stay exact.
    """
    from marl_battlegrounds.training.validation import (
        _collection_boundary,  # pyright: ignore[reportPrivateUsage]
    )

    value = _object(counters, "Checkpoint counters")
    if set(value) != _PQN_COUNTERS or any(
        type(item) is not int or item < 0 for item in value.values()
    ):
        raise ValueError("PQN-VDN checkpoint counters are malformed")
    rounds, remainder = divmod(value["env_steps"], games)
    if remainder or rounds > total_rounds:
        raise ValueError("PQN-VDN checkpoint counters are impossible")
    if continuation is not None:
        from marl_battlegrounds.training._continuation_schedules import (
            continuation_boundary,
        )

        reachable, ordinal, learning = continuation_boundary(
            rounds,
            total_rounds=total_rounds,
            continuation=continuation,
            rollout_length=pqn.rollout_length,
            initial_rounds=pqn.initial_rounds,
        )
        if (
            reachable != rounds
            or value["completed_blocks"] != ordinal
            or value["learning_blocks"] != learning
            or value["updates"] != learning * pqn.epochs * pqn.num_minibatches
        ):
            raise ValueError("PQN-VDN continuation checkpoint counters are impossible")
        return
    reachable, ordinal = _collection_boundary(
        rounds,
        total_env_steps=total_rounds,
        num_envs=1,
        rollout_length=pqn.rollout_length,
        initial_rounds=pqn.initial_rounds,
    )
    learning = max(0, ordinal - -(-pqn.initial_rounds // pqn.rollout_length))
    if (
        reachable != rounds
        or value["completed_blocks"] != ordinal
        or value["learning_blocks"] != learning
        or value["updates"] != learning * pqn.epochs * pqn.num_minibatches
    ):
        raise ValueError("PQN-VDN checkpoint counters are impossible")


def _check_method_settings(
    method: str, qmix: QMIXConfig | None, pqn: PQNConfig | None
) -> None:
    """Refuse settings objects that belong to another method.

    qmix may be given only for "qmix" and pqn only for "pqn_vdn". Raises
    ValueError naming the method family ("PPO checkpoints take no QMIX
    settings" keeps its historical words). Reads and writes nothing.
    """
    family = "PPO" if is_ppo_method(method) else method.upper().replace("_", "-")
    if method != "qmix" and qmix is not None:
        raise ValueError(f"{family} checkpoints take no QMIX settings")
    if method != "pqn_vdn" and pqn is not None:
        raise ValueError(f"{family} checkpoints take no PQN settings")


def _targets(tree: Tree, device: object | None) -> Tree:
    """Build explicit array restore targets on the caller's selected device."""
    selected = cast(Any, jax.devices()[0] if device is None else device)
    sharding = jax.sharding.SingleDeviceSharding(selected)

    def target(value: Tree) -> jax.ShapeDtypeStruct:
        """Place one abstract array on the chosen device, preserving typed keys."""
        return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding)

    return jax.tree.map(target, tree)


def _save_arrays(path: Path, tree: Tree) -> None:
    """Synchronously save numerical arrays using the installed Orbax handler."""
    with cast(Any, ocp.Checkpointer(ocp.StandardCheckpointHandler())) as checkpointer:
        checkpointer.save(path, args=ocp.args.StandardSave(tree))


def _restore_arrays(path: Path, template: Tree, device: object | None) -> Tree:
    """Restore strictly into code-owned shapes and explicit current placement."""
    with cast(Any, ocp.Checkpointer(ocp.StandardCheckpointHandler())) as checkpointer:
        saved = checkpointer.metadata(path).item_metadata.tree

        def physical(value: Tree) -> Tree:
            """Compare Orbax's physical key storage before any dtype conversion."""
            if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
                return jax.eval_shape(jax.random.key_data, value)
            return value

        if _layout(saved) != _layout(jax.tree.map(physical, template)):
            raise ValueError("Stored payload paths, shapes or dtypes differ")
        return checkpointer.restore(
            path, args=ocp.args.StandardRestore(_targets(template, device), strict=True)
        )


def _collection_details(collection: TrainingCollection) -> dict[str, Any]:
    """Record reconstructible static settings and verified content identities.

    A named pinned opponent's JSON record is added under "pinned_opponent" only
    when one is pinned, and a variables hook's identity (QMIX exploration) under
    "actor_variables_at_step" only when one is set, so descriptions of other
    runs keep their bytes.
    """
    details = _object(
        {
            "content_binding": collection.binding.model_dump(mode="json"),
            "schedule": dict(collection.schedule.rounding_report),
            "score_threshold_curriculum": (
                collection.schedule.score_threshold_curriculum
            ),
            "root_bits": list(collection.root_bits),
            "key_schema": collection.key_schema,
            "reward_settings": list(collection.reward_settings),
            "shaping": collection.shaping,
            "shaping_mode": collection.shaping_mode,
            "pinned_opponent_share": collection.pinned_opponent_share,
            "collect_training_state": collection.collect_training_state,
            "metrics": collection.metrics,
            "recording": collection.recording,
            "opponent_rows": collection.carry_spec.progress.opponent_steps.shape[0],
            "history_capture_capacity": (
                collection.carry_spec.progress.opponent_steps.shape[0]
                - 2
                - len(collection.opponent_names)
            ),
        },
        "Collection details",
    )
    if collection.reward_identity is not None:
        details["reward_identity"] = collection.reward_identity
    if collection.opponent_names:
        details["opponent_members"] = list(collection.opponent_records)
    if collection.learner_slots is not None:
        details["learner_slots"] = list(collection.learner_slots)
        details["partner_members"] = list(collection.partner_records)
    if collection.pinned_opponent is not None:
        details["pinned_opponent"] = collection.pinned_opponent
    if collection.actor_variables_at_step is not None:
        details["actor_variables_at_step"] = _object(
            collection.actor_variables_at_step.identity, "Variables hook identity"
        )
    return details


def _checked_member_binding(saved: object, current: object) -> dict[str, Any] | None:
    """Check a frozen member while allowing a proved artifact to move.

    Both records must retain exact names, registration, variable digest and
    execution rules. A matching artifact-description identity permits a new
    reload path and keeps already checked historical exposure, even if the old
    sibling learner is no longer nearby. Missing old input receipts permit only
    the original reference and evidence; they do not authorize relocation.
    Factory and live-method identity fields still compare exactly. Return a
    fresh current record or None. Invalid or changed identities raise ValueError.
    """
    if saved is None or current is None:
        if saved != current:
            raise ValueError("Checkpoint member binding differs")
        return None
    previous = _object(saved, "Saved member binding")
    chosen = _object(current, "Current member binding")
    checked = dict(chosen)
    before = previous.get("input_identity")
    after = chosen.get("input_identity")
    same_artifact = (
        isinstance(before, dict)
        and cast(dict[str, Any], before).get("kind") == "artifact"
        and before == after
    )
    if same_artifact:
        # The receipt identifies the exact saved description; the remaining
        # registration/variable comparison below proves inference is unchanged.
        checked["reference"] = previous.get("reference")
        checked["evidence"] = previous.get("evidence")
        chosen["evidence"] = previous.get("evidence")
    if "input_identity" not in previous:
        checked.pop("input_identity", None)
    if previous != checked:
        raise ValueError("Checkpoint member binding differs")
    return chosen


def _check_collection_compatibility(
    details: dict[str, Any], collection: TrainingCollection
) -> TrainingCollection:
    """Check bound members and collection rules without arrays, hashes or writes.

    details is an already checked learner description. Preserve historical
    defaults and compare all collection settings. Only identical saved artifacts
    may change their reload path; checked saved exposure stays attached to those
    members. Return a collection carrying the current paths and preserved facts.
    Reuse this preflight before output setup and inside numerical restore so both
    routes enforce the same rule. No member is loaded or called here.
    """
    from dataclasses import replace

    saved = _object(details["collection"], "Saved collection settings")
    saved.setdefault("shaping_mode", "potential")
    saved.setdefault("score_threshold_curriculum", False)
    saved.setdefault("pinned_opponent_share", 0.0)
    saved.setdefault("history_capture_capacity", saved.get("opponent_rows", 22) - 2)
    try:
        pinned = _checked_member_binding(
            saved.get("pinned_opponent"), collection.pinned_opponent
        )
    except ValueError as error:
        raise ValueError("Checkpoint pinned opponent differs") from error
    current = _collection_details(collection)
    replacements: dict[str, Any] = {"pinned_opponent": pinned}
    for key, field in (
        ("opponent_members", "opponent_records"),
        ("partner_members", "partner_records"),
    ):
        previous = saved.get(key, [])
        present = current.get(key, [])
        if not isinstance(previous, list) or len(cast(list[object], previous)) != len(
            present
        ):
            raise ValueError("Checkpoint content or collection settings differ")
        replacements[field] = tuple(
            _checked_member_binding(old, new)
            for old, new in zip(cast(list[object], previous), present, strict=True)
        )
        if key in saved:
            current[key] = previous
    saved.pop("pinned_opponent", None)
    current.pop("pinned_opponent", None)
    if saved != current:
        raise ValueError("Checkpoint content or collection settings differ")
    return replace(collection, **replacements)


def _check_recording(value: object, root: Path) -> dict[str, Any]:
    """Validate original recording constructor arguments before any rewind."""
    record = _object(value, "Recording settings")
    required = {
        "relative_path",
        "phase",
        "pass_id",
        "policies",
        "checkpoint_id",
        "details",
    }
    if set(record) != required or record["phase"] != "training":
        raise ValueError("Recording settings need the original dedicated training pass")
    _relative(root, record["relative_path"])
    if not isinstance(record["pass_id"], str) or not record["pass_id"]:
        raise ValueError("Recording pass ID must be nonempty")
    if record["checkpoint_id"] is not None and not isinstance(
        record["checkpoint_id"], str
    ):
        raise ValueError("Recording checkpoint label must be text or None")
    _object(record["policies"], "Original training registrations")
    _object(record["details"], "Original training pass details")
    return record


def _check_recording_registration(record: dict[str, Any], writer_path: Path) -> None:
    """Match constructor facts to the writer's already-durable registration.

    This reads the public run manifest before asking the writer to make a token.
    M8 still owns all token/table/replay checks and every recording mutation.
    """
    details = _read_json(writer_path / "run_details.json")
    key = json.dumps((record["phase"], record["pass_id"]), separators=(",", ":"))
    passes = _object(details.get("passes"), "Recorded passes")
    if set(passes) != {key}:
        raise ValueError("Checkpoint recording needs its original dedicated pass")
    saved = _object(passes[key], "Original training pass")
    for name in ("phase", "pass_id", "policies", "checkpoint_id", "details"):
        if saved.get(name) != record[name]:
            raise ValueError("Checkpoint recording registration differs")


def save_checkpoint(
    run_dir: str | Path,
    collection: TrainingCollection,
    state: LearnerState | QMIXLearnerState | PQNLearnerState,
    *,
    metadata: dict[str, object],
    writer: RunWriter | None = None,
    ppo: PPOConfig = DEFAULT_PPO_CONFIG,
    method: str = "mappo",
    qmix: QMIXConfig | None = None,
    pqn: PQNConfig | None = None,
) -> Path:
    """Publish a complete update boundary, then replace its latest pointer.

    Parameters
    ----------
    run_dir : str or Path
        Existing run directory. The caller must hold its exclusive run lock.
    collection, state
        Matching stable collection descriptor and complete accepted learner
        (LearnerState for PPO, QMIXLearnerState for QMIX, PQNLearnerState for
        PQN-VDN). Initialization, QMIX warmup and PQN-VDN initial-collection
        boundaries are allowed; no actor decision occurs.
        Saving reuses the descriptor's installed-content check from setup or
        restore. It still checks the actual carried source bank and learner.
    metadata : dict
        Finite JSON with run_id, attempt_id, parent_checkpoint, config, source,
        dependencies and execution. Execution must match the current shared
        compilation/runtime owner. The runner supplies verified scientific facts.
        Optional host_state and log_cursors are stored unchanged. With recording,
        recording holds the original writer constructor facts documented below.
    writer : RunWriter or None, default None
        Required exactly when collection.recording is true. Its token is saved
        with the learner. Recording metadata contains relative_path, phase,
        pass_id, policies, checkpoint_id and details from original registration.
    ppo : PPOConfig, default DEFAULT_PPO_CONFIG
        Existing numerical learner settings used by boundary validation. Its
        input scale, spawn frame and value-normalization setting must equal the
        saved config's, and the state must carry matching statistics, else the
        save is refused before any file changes. The default's frame is "left".
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Static learner method, matching metadata.config.method and the complete
        numerical state. Missing historical config method means MAPPO. Conflicts
        fail before obtaining a writer token or writing any payload.
    qmix : QMIXConfig or None, default=None
        QMIX settings; None means DEFAULT_QMIX_CONFIG. Used only when method
        is "qmix": every field must equal the saved config's "qmix" block,
        and validate_qmix_learner checks the state with it. It must be None
        for other methods, whose ppo argument is ignored for QMIX.
    pqn : PQNConfig or None, default=None
        PQN-VDN settings; None means DEFAULT_PQN_CONFIG. Used only when
        method is "pqn_vdn": every field must equal the saved config's "pqn"
        block, and validate_pqn_learner checks the state with it. It must be
        None for other methods.

    Returns
    -------
    Path
        Immutable content-identified checkpoint directory. All payload files and
        its latest_checkpoint.json pointer are durable on the local filesystem.
        A QMIX checkpoint's actor/ holds only the Q-network parameters; a
        PQN-VDN checkpoint's holds its network's params and batch_stats. Both
        add completed_blocks and learning_blocks to the counters.

    Raises
    ------
    ValueError
        State, context, writer or recorded boundary is incompatible.
    OSError
        Saving or publication fails. Earlier checkpoints remain intact. Caught
        failures remove this call's unfinished payload and ownership receipt.
        A renamed complete artifact survives later pointer or sync failures.
        The old pointer stays unchanged until its atomic replacement; a failure
        after replacement may leave the new complete checkpoint as latest.
        Retrying can publish another complete artifact for the same logical
        boundary: Orbax writes fresh payload filenames and timestamps. Existing
        published artifacts and their identities remain unchanged.

    Notes
    -----
    After an abrupt process exit, the runner checks temporary-write receipts
    under its existing run lock, after validating recovery inputs. It removes
    only unfinished writes whose original process is known to have exited.
    Unrecognized data and live or unknown writers are preserved, with reasons
    saved in run_events.jsonl. Published artifacts are never removed by cleanup.
    """
    method = validate_training_method(method)
    root = _directory(Path(run_dir))
    context = _check_context(metadata, require_execution=True)
    if _config_method(context["config"]) != method:
        raise ValueError("Checkpoint config method differs from the selected method")
    _check_method_settings(method, qmix, pqn)
    if method == "qmix":
        from marl_battlegrounds.training.qmix_learner import validate_qmix_learner

        settings = DEFAULT_QMIX_CONFIG if qmix is None else qmix
        _check_qmix_settings(context["config"], settings)
        validate_qmix_learner(
            collection,
            cast("QMIXLearnerState", state),
            qmix=settings,
            recheck_installed_content=False,
        )
    elif method == "pqn_vdn":
        from marl_battlegrounds.training.pqn_learner import validate_pqn_learner

        chosen = DEFAULT_PQN_CONFIG if pqn is None else pqn
        _check_pqn_settings(context["config"], chosen)
        validate_pqn_learner(
            collection,
            cast("PQNLearnerState", state),
            pqn=chosen,
            recheck_installed_content=False,
        )
    else:
        from marl_battlegrounds.training.learner import validate_learner

        if _config_input_scale(context["config"]) != ppo.input_scale:
            raise ValueError("Checkpoint config input_scale differs from PPO settings")
        if _config_spawn_frame(context["config"]) != ppo.spawn_frame:
            raise ValueError("Checkpoint config spawn_frame differs from PPO settings")
        if _config_value_normalization(context["config"]) != ppo.value_normalization:
            raise ValueError(
                "Checkpoint config value_normalization differs from PPO settings"
            )
        validate_learner(
            collection,
            cast("LearnerState", state),
            ppo=ppo,
            method=method,
            recheck_installed_content=False,
        )
    if collection.recording != (writer is not None):
        raise ValueError("Checkpoint writer must match collection recording")
    token = None
    if writer is not None:
        recording = _check_recording(context.get("recording"), root)
        if (
            _relative(root, recording["relative_path"]).resolve()
            != writer.run_dir.resolve()
        ):
            raise ValueError("Recording path differs from the open writer")
        _check_recording_registration(recording, writer.run_dir)
        token = writer.checkpoint_recording()
    elif context.get("recording") is not None:
        raise ValueError("Unrecorded checkpoint must not claim a recording writer")
    directory = root / "checkpoints"
    directory.mkdir(exist_ok=True)
    _directory(directory)
    with _pending_write(
        directory, kind="learner", attempt_id=context.get("attempt_id")
    ) as temporary:
        actor = _actor_payload(state, method)
        _save_arrays(temporary / "actor", actor)
        _save_arrays(temporary / "state", _state_payload(state, method))
        details: dict[str, Any] = {
            "schema_version": 1,
            "kind": "learner",
            "schemas": checkpoint_schemas(method),
            "metadata": context,
            "collection": _collection_details(collection),
            "layout": _layout(state),
            "actor_layout": _layout(actor),
            "actor_digest": tree_digest(actor),
            "recording_token": token,
            "counters": _counters(state, collection.schedule.num_envs, method),
            "files": _inventory(temporary),
        }
        identifier = sha256(_json_bytes(details)).hexdigest()
        details["checkpoint_id"] = identifier
        _atomic_json(temporary / _DESCRIPTION, details)
        _sync_tree(temporary)
        existing = artifact_directory(root, "checkpoints", identifier)
        destination = (
            existing
            if existing.exists()
            else directory
            / artifact_name(
                method, "checkpoint", details["counters"]["env_steps"], identifier
            )
        )
        if destination.exists():
            if read_checkpoint_details(destination) != details:
                raise ValueError("Checkpoint identity already contains different data")
            shutil.rmtree(temporary)
        else:
            temporary.rename(destination)
        _sync_dir(directory)
        _atomic_json(
            root / "latest_checkpoint.json",
            {
                "schema_version": 1,
                "checkpoint_id": identifier,
                "relative_path": str(destination.relative_to(root)),
            },
        )
        for ancestor in (root, *root.parents):
            _sync_dir(ancestor)
        return destination


def validation_directory(
    run_dir: str | Path, purpose: str, identifier: str, *, actor: Path | None = None
) -> Path:
    """Locate a boundary's validation output, preserving historical paths.

    run_dir owns the validation output. purpose is a single Python identifier
    such as routine, confirmation or random. identifier is the full originating
    checkpoint ID. Optional actor supplies an inherited actor outside this run;
    otherwise look in its actors directory. Read only the small description to
    name new output by method and actual step. Existing historical purpose-ID
    paths stay in use. Reject competing directories; the validation owner checks
    their task identity before reuse. No files or numerical arrays change.
    """
    if not purpose.isidentifier():
        raise ValueError("Validation purpose must be a single identifier")
    _digest(identifier, "Validation checkpoint identity")
    root = Path(run_dir) / "validation"
    legacy = root / f"{purpose}-{identifier}"
    matches = list(root.glob(f"{purpose}_*_{identifier}"))
    if legacy.exists() or legacy.is_symlink():
        matches.append(legacy)
    if len(matches) > 1:
        raise ValueError("Multiple validation directories claim this saved boundary")
    if matches:
        return matches[0]
    actor = (
        artifact_directory(run_dir, "actors", identifier) if actor is None else actor
    )
    stem = f"saved_actor_{identifier}"
    if actor.exists():
        details = read_checkpoint_description(actor)
        if (
            details["kind"] != "actor"
            or details["metadata"].get("checkpoint_id") != identifier
        ):
            raise ValueError("Validation actor differs from its checkpoint identity")
        stem = artifact_name(
            _schema_method(details["schemas"]),
            "actor",
            details["metadata"]["env_steps"],
            identifier,
        )
    return root / f"{purpose}_{stem}"


def read_checkpoint_details(path: str | Path) -> dict[str, Any]:
    """Read and verify a complete learner or actor artifact without mutation.

    path names an immutable directory. Check every payload file's bytes and the
    description through read_checkpoint_description. A pruned learner is not
    available. No learner is reconstructed and no recording is changed here.
    """
    root = _directory(Path(path))
    details = read_checkpoint_description(root)
    name = _DESCRIPTION if details["kind"] == "learner" else _ACTOR_DESCRIPTION
    actual = {k: v for k, v in _inventory(root).items() if k != name}
    if actual != details["files"]:
        raise ValueError("Checkpoint payload files are missing, changed or unexpected")
    return details


def restore_checkpoint(
    path: str | Path,
    collection: TrainingCollection,
    template: LearnerState | QMIXLearnerState | PQNLearnerState,
    *,
    expected_metadata: Mapping[str, object],
    ppo: PPOConfig = DEFAULT_PPO_CONFIG,
    method: str = "mappo",
    device: object | None = None,
    qmix: QMIXConfig | None = None,
    pqn: PQNConfig | None = None,
    _source_compatibility: Mapping[str, object] | None = None,
) -> RestoredCheckpoint:
    """Fully validate and restore numerical state without changing any files.

    Parameters
    ----------
    path : str or Path
        Explicit complete learner checkpoint; never a guessed latest directory.
    collection, template
        Fresh code-built descriptor/state from the saved seed and settings. All
        numerical template values are replaced; its static structure is trusted
        only after comparison with the saved content and settings. The template
        may be shape-only (``jax.ShapeDtypeStruct`` leaves), so a QMIX or
        PQN-VDN resume need not hold a second learner while restoring.
        Missing historical shaping_mode means potential; score_delta must be
        saved explicitly and cannot resume a potential checkpoint.
        Missing historical score_threshold_curriculum means False. New threshold
        banks and schedules must match exactly; actor-only exports remain usable
        without reconstructing a training schedule. Missing historical
        pinned_opponent_share means zero in the saved config, in the expected
        config and in the saved collection settings, so an older run compares
        equal to today's default; a positive share must be saved explicitly.
        Missing historical ppo.spawn_frame means "world" on both sides; a
        checkpoint saved in another frame must be resumed in that frame.
        Missing historical ppo.value_normalization means False on both sides.
        Enabled checkpoints carry three scalar statistics; disabled checkpoints
        retain their historical array paths and payload keys.
        Missing validation_opponents and slot_diagnostic_actor mean None.
        Missing historical pinned_opponent means None in both configs and the
        saved collection settings. A pinned opponent must match its saved record
        exactly (reference, registration, variables digest, evidence, memory
        rule), else restore is refused before arrays are read. After arrays are
        read, and before the slower learner validation, a JAX pinned System's
        weights must match their recorded digest, and a pinned host method with
        memory refuses the resume when any of its games is unfinished, because
        that memory is never saved.
    expected_metadata : mapping
        Required config, source, dependencies and execution discovered for
        this execution. Each must equal its saved value. Additional supplied
        keys are equality assertions; attempt/parent IDs usually are omitted.
    ppo : PPOConfig, default DEFAULT_PPO_CONFIG
        Matched numerical update settings for restored-boundary validation. Its
        input scale, spawn frame and value-normalization setting must equal the
        checkpoint's, else restore
        is refused before arrays are read. The default's frame is "left", so a
        checkpoint trained in "world" needs the matching config passed here.
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Static requested method. It must agree with saved/expected config and
        the model schema before any numerical arrays are restored. Missing
        historical config method means MAPPO, never a guess from array shapes.
    qmix : QMIXConfig or None, default=None
        QMIX settings (None means DEFAULT_QMIX_CONFIG), used only for "qmix".
        Scale and frame must equal the checkpoint's, every field must equal
        the saved config's "qmix" block, and the saved counters must be
        possible for these settings (``_check_qmix_counters``), all before any
        array is read. It must be None for other methods.
    pqn : PQNConfig or None, default=None
        PQN-VDN settings (None means DEFAULT_PQN_CONFIG), used only for
        "pqn_vdn", with the same metadata-only checks: scale, frame, every
        field of the saved "pqn" block, and counters that describe a reachable
        boundary of the collection's total (``_check_pqn_counters``). The
        actor item is reinserted into the full state by name (parameters,
        then statistics). It must be None for other methods.
    device : jax.Device or None, default None
        Explicit target device; None uses this process's first selected device.
        Every restored array, including empty IDs, is committed to this device.
        Saved device ordinals are ignored. The target backend and device kind
        must match the checked execution identity. Use frozen actor loading for
        independent inference on another backend.

    _source_compatibility : mapping or None, default=None
        Private checked continuation transition. None keeps exact source
        equality. The separate continuation route supplies a record from
        continuation_source_compatibility; this function verifies it again.
        Dependencies, execution, scientific config and payload checks stay exact.

    Returns
    -------
    RestoredCheckpoint
        Checked numerical state and metadata, before writer/log recovery.

    Raises
    ------
    ValueError
        Any payload, context, content, schema, array or boundary check fails.
        Faults that need only metadata (method, config, schema, layout
        including replay, counters) are raised before any array is read.
        A learner saved before Red Zone (actor input schema 1) is refused by
        _require_current_schemas right after the checkpoint kind check.
        No recording files or trainer logs have been changed.
    """
    method = validate_training_method(method)
    root = _directory(Path(path))
    details = read_checkpoint_details(root)
    if details["kind"] != "learner":
        raise ValueError("Actor exports cannot resume a learner")
    _require_current_schemas(details)
    if _schema_method(details["schemas"]) != method:
        raise ValueError("Checkpoint model differs from the requested method")
    expected = _object(dict(expected_metadata), "Expected execution metadata")
    if not {"config", "source", "dependencies", "execution"} <= expected.keys():
        raise ValueError(
            "Restore requires current config, source, dependencies and execution"
        )
    if _config_method(expected["config"]) != method:
        raise ValueError("Expected config method differs from the requested method")
    if expected["execution"] != execution_identity():
        raise ValueError("Expected execution identity differs from the current runtime")
    saved_metadata = dict(details["metadata"])
    # Configs saved before pinned_opponent_share existed compare at its default.
    for record in (saved_metadata, expected):
        config_record = record.get("config")
        if isinstance(config_record, dict):
            normalized = dict(cast(dict[str, Any], config_record))
            normalized.setdefault("method", "mappo")
            normalized.setdefault("retain_all_actors", False)
            normalized.setdefault("pinned_opponent_share", 0.0)
            normalized.setdefault("pinned_opponent", None)
            normalized.setdefault("initial_actor", None)
            normalized.setdefault("validation_opponents", None)
            normalized.setdefault("slot_diagnostic_actor", None)
            ppo_record = normalized.get("ppo", {})
            if is_ppo_method(normalized["method"]) and isinstance(ppo_record, dict):
                # Configs saved before spawn_frame existed compare at "world".
                nested = dict(cast(dict[str, Any], ppo_record))
                nested.setdefault("input_scale", 1.0)
                nested.setdefault("spawn_frame", "world")
                nested.setdefault("value_normalization", False)
                normalized["ppo"] = nested
            record["config"] = normalized
    if _source_compatibility is not None:
        checked_transition = continuation_source_compatibility(
            saved_metadata["source"], expected["source"]
        )
        if dict(_source_compatibility) != checked_transition:
            raise ValueError("Continuation source compatibility evidence differs")
    for key, value in expected.items():
        if saved_metadata.get(key) != value and not (
            key == "source" and _source_compatibility is not None
        ):
            raise ValueError("Checkpoint execution metadata differs")
    _check_method_settings(method, qmix, pqn)
    settings = DEFAULT_QMIX_CONFIG if qmix is None else qmix
    chosen = DEFAULT_PQN_CONFIG if pqn is None else pqn
    if method == "qmix":
        if _actor_input_scale(details) != settings.input_scale:
            raise ValueError("Checkpoint input_scale differs from QMIX settings")
        if _actor_spawn_frame(details) != settings.spawn_frame:
            raise ValueError("Checkpoint spawn_frame differs from QMIX settings")
        _check_qmix_settings(details["metadata"]["config"], settings)
        from marl_battlegrounds.training._continuation_schedules import (
            schedule_continuation,
        )

        _check_qmix_counters(
            details["counters"],
            settings,
            collection.schedule.num_envs,
            continuation=schedule_continuation(collection.schedule),
        )
    elif method == "pqn_vdn":
        if _actor_input_scale(details) != chosen.input_scale:
            raise ValueError("Checkpoint input_scale differs from PQN settings")
        if _actor_spawn_frame(details) != chosen.spawn_frame:
            raise ValueError("Checkpoint spawn_frame differs from PQN settings")
        _check_pqn_settings(details["metadata"]["config"], chosen)
        from marl_battlegrounds.training._continuation_schedules import (
            schedule_continuation,
        )

        _check_pqn_counters(
            details["counters"],
            chosen,
            collection.schedule.num_envs,
            collection.schedule.total_env_steps // collection.schedule.num_envs,
            continuation=schedule_continuation(collection.schedule),
        )
    else:
        if _actor_input_scale(details) != ppo.input_scale:
            raise ValueError("Checkpoint input_scale differs from PPO settings")
        if _actor_spawn_frame(details) != ppo.spawn_frame:
            raise ValueError("Checkpoint spawn_frame differs from PPO settings")
        if (
            _config_value_normalization(details["metadata"]["config"])
            != ppo.value_normalization
        ):
            raise ValueError("Checkpoint value_normalization differs from PPO settings")
    collection = _check_collection_compatibility(details, collection)
    recorded = details["metadata"].get("recording")
    if collection.recording != (recorded is not None):
        raise ValueError("Checkpoint recording declaration differs")
    if collection.recording:
        record = _check_recording(recorded, root.parent.parent)
        _check_recording_registration(
            record, _relative(root.parent.parent, record["relative_path"])
        )
    if details["layout"] != _layout(template) or details["actor_layout"] != _layout(
        _actor_payload(template, method)
    ):
        raise ValueError("Checkpoint array paths, shapes or dtypes differ")
    selected_device = cast(Any, jax.devices()[0] if device is None else device)
    runtime = expected["execution"]["runtime"]
    if (
        selected_device.platform != runtime["backend"]
        or selected_device.device_kind != runtime["device"]
    ):
        raise ValueError("Restore device differs from the checked execution identity")
    actor = _restore_arrays(
        root / "actor", _actor_payload(template, method), selected_device
    )
    if tree_digest(actor) != details["actor_digest"]:
        raise ValueError("Restored actor digest differs")
    payload = _restore_arrays(
        root / "state", _state_payload(template, method), selected_device
    )
    # A PQN-VDN item's leaves are in sorted-key order (statistics first); the
    # full state holds the network in NamedTuple order, so rebuild it by name.
    actor_leaves = iter(
        jax.tree.leaves(_pqn_network(actor) if method == "pqn_vdn" else actor)
    )
    leaves: list[Any] = []
    for index, (row, initial) in enumerate(
        zip(details["layout"], jax.tree.leaves(template), strict=True)
    ):
        if _is_actor(row, method):
            leaves.append(next(actor_leaves))
        elif initial.size == 0:
            leaves.append(
                jax.device_put(np.empty(initial.shape, initial.dtype), selected_device)
            )
        else:
            leaves.append(payload[f"leaf_{index:05d}"])
    state = cast(
        "LearnerState | QMIXLearnerState | PQNLearnerState",
        jax.tree.unflatten(cast(Any, jax.tree.structure(template)), leaves),
    )
    if _layout(state) != details["layout"]:
        raise ValueError("Restored array schema differs")
    # The pinned checks read a few restored arrays; they run before the full
    # learner validation, which re-checks the installed content and is slow.
    _check_restored_pinned_opponent(collection, state)
    if method == "qmix":
        from marl_battlegrounds.training.qmix_learner import validate_qmix_learner

        validate_qmix_learner(
            collection, cast("QMIXLearnerState", state), qmix=settings
        )
    elif method == "pqn_vdn":
        from marl_battlegrounds.training.pqn_learner import validate_pqn_learner

        validate_pqn_learner(collection, cast("PQNLearnerState", state), pqn=chosen)
    else:
        from marl_battlegrounds.training.learner import validate_learner

        validate_learner(
            collection, cast("LearnerState", state), ppo=ppo, method=method
        )
    counters = _counters(state, collection.schedule.num_envs, method)
    if counters != details["counters"]:
        raise ValueError("Checkpoint counters disagree with numerical state")
    return RestoredCheckpoint(state, details, root)


def _check_restored_pinned_opponent(
    collection: TrainingCollection,
    state: LearnerState | QMIXLearnerState | PQNLearnerState,
) -> None:
    """Check frozen weights and refuse unrestorable live host-game memory.

    Numerical opponent and partner weights are restored by the normal array
    loader. Accessible host parameter arrays use that same saved tree; opaque
    clients must be rebound. No current host memory is serialized, so a stateful
    host member cannot resume an unfinished game. Check this before changing any
    recording or log. Partner A and mirrored Partner B have separate memory;
    an external opponent replaces all of Team B and uses no mirrored partner.
    """
    from marl_battlegrounds.evaluation.policy_execution import (
        _CompositionVariables,  # pyright: ignore[reportPrivateUsage]
    )

    carry = state.carry
    record = collection.pinned_opponent
    if record is not None:
        pinned = cast(tuple[Any, ...], carry.pinned_opponent)
        if collection.opponent_records and collection.host_opponent is None:
            variables = cast(_CompositionVariables, pinned[0])
            _check_member_weights(collection.opponent_records, variables.members)
        elif (
            record.get("variables_digest") is not None
            and tree_digest(pinned[0]) != record["variables_digest"]
        ):
            raise ValueError("Restored pinned opponent weights differ from its record")
    if collection.partner_records and collection.host_partners is None:
        variables = cast(_CompositionVariables, carry.partner_values[0])
        _check_member_weights(collection.partner_records, variables.members)

    unfinished = ~np.asarray(carry.state.done.done)
    external = np.asarray(carry.history.lane_snapshot) == -2
    host = collection.host_opponent
    if host is not None and host.stateful:
        _require_finished_host_games(unfinished & external, "opponent")
    partners = collection.host_partners
    if partners is not None:
        assert collection.learner_slots is not None
        owned = np.asarray([slot not in collection.learner_slots for slot in range(5)])
        for side, holder in enumerate(partners):
            active = np.asarray(carry.state.config.agent_profile.active_mask)[
                :, side * 5 : (side + 1) * 5
            ]
            playing = unfinished & np.any(active & owned, axis=1)
            if side == 1 and record is not None:
                playing &= ~external
            if holder.stateful:
                _require_finished_host_games(playing, f"Team {'AB'[side]} partner")
    saved_hosts = iter(carry.frozen_host_values)
    if host is not None:
        host.restore_values(next(saved_hosts))
    if partners is not None:
        values = next(saved_hosts)
        for holder in partners:
            holder.restore_values(values)


def _check_member_weights(
    records: tuple[dict[str, Any], ...], values: tuple[Any, ...]
) -> None:
    """Compare each frozen JAX member with its saved declaration, in alias order."""
    if len(records) != len(values):
        raise ValueError("Restored population size differs")
    for member, weights in zip(records, values, strict=True):
        if tree_digest(weights) != member["variables_digest"]:
            raise ValueError(f"Restored member {member['name']!r} weights differ")


def _require_finished_host_games(unfinished: np.ndarray, name: str) -> None:
    """Refuse loss of active host memory before checkpoint recovery writes files."""
    if bool(unfinished.any()):
        raise ValueError(
            f"The {name} host method's memory is not saved, and "
            f"{int(unfinished.sum())} of its games are unfinished at this checkpoint; "
            "ordinary resume cannot continue them without that memory. Resume from "
            "a checkpoint where none of its games is unfinished, or start a new run."
        )


def resume_recording(
    restored: RestoredCheckpoint, run_dir: str | Path
) -> RunWriter | None:
    """Mark checked recovery, then open M8 with its original token/registrations.

    The caller holds the run lock and must pass restore_checkpoint's result.
    Returns an open writer when recording was enabled, otherwise None. The
    runner then restores its logs and calls finish_checkpoint_recovery. Failure
    leaves the same checkpoint/token marker for an explicit retry; this function
    never clears it on an error or selects a different recovery checkpoint.
    """
    from marl_battlegrounds.evaluation.run_writer import RunWriter

    root = _directory(Path(run_dir))
    if (
        restored.path.parent.name != "checkpoints"
        or restored.path.parent.parent != root
    ):
        raise ValueError("Restored checkpoint belongs to a different run directory")
    token = restored.details["recording_token"]
    record = restored.details["metadata"].get("recording")
    if (record is None) != (token is None):
        raise ValueError("Recording settings and checkpoint token disagree")
    checked = None if record is None else _check_recording(record, root)
    marker = {
        "schema_version": 1,
        "checkpoint_id": restored.details["checkpoint_id"],
        "recording_token": token,
    }
    marker_path = root / "checkpoint_recovery.json"
    if marker_path.exists() and _read_json(marker_path) != marker:
        raise ValueError("Interrupted recovery requires its original checkpoint")
    _atomic_json(marker_path, marker)
    if checked is None:
        return None
    return RunWriter(
        resume_from=_relative(root, checked["relative_path"]),
        recording_checkpoint=token,
        phase=checked["phase"],
        pass_id=checked["pass_id"],
        policies=checked["policies"],
        checkpoint_id=checked["checkpoint_id"],
        details=checked["details"],
    )


def finish_checkpoint_recovery(
    restored: RestoredCheckpoint, run_dir: str | Path
) -> None:
    """Clear the matching marker after writer and trainer-log recovery succeed.

    Caller holds the run lock. A missing/different marker is an error, preventing
    accidental completion of another recovery transaction. This writes no data
    besides removing the marker and syncing its directory.
    """
    root = _directory(Path(run_dir))
    marker = root / "checkpoint_recovery.json"
    expected = {
        "schema_version": 1,
        "checkpoint_id": restored.details["checkpoint_id"],
        "recording_token": restored.details["recording_token"],
    }
    if _read_json(marker) != expected:
        raise ValueError("Recovery marker differs from the restored checkpoint")
    marker.unlink()
    _sync_dir(root)


@lru_cache(maxsize=16, typed=True)
def _actor_template(
    method: str = "mappo",
    actor_input_schema: int = ACTOR_INPUT_SCHEMA_VERSION,
    parameter_sharing: str = "all",
    normalization: str = "batch_norm",
) -> Tree:
    """Cache actor shapes/dtypes for one checked method and actor input schema.

    Parameters
    ----------
    method : str, default="mappo"
        Training method: "mappo", "ippo", "ff_mappo", "ff_ippo", "qmix" or
        "pqn_vdn".
    actor_input_schema : int, default=ACTOR_INPUT_SCHEMA_VERSION
        2 for the current 5,165-feature actors, or 1 for actors saved before
        Red Zone (5,164 features).
    parameter_sharing : {"all", "class", "none"}, default="all"
        Actor weight ownership; grouped modes add a leading axis of five.
    normalization : {"batch_norm", "layer_norm"}, default="batch_norm"
        PQN-VDN hidden-layer normalization, which changes its parameter and
        statistic layout. Other methods accept only "batch_norm".

    Returns
    -------
    PyTree
        Abstract ``jax.ShapeDtypeStruct`` actor leaves; no weights, mixer or
        optimizer arrays are allocated. The method-aware initializer is traced
        with eval_shape. For QMIX the tree is the Q-network parameters only
        (qmix_actor_template); for PQN-VDN it is the on-disk actor item,
        Flax's ``{"params", "batch_stats"}`` dictionary built from
        pqn_actor_template. The two schemas differ only in the input width,
        so the schema-1 tree is the current one with every 5,165-long input
        axis (the first Dense kernel's rows and, for PQN-VDN, the input
        BatchNorm leaves) set to 5,164; paths, order, dtypes and every other
        shape are the same. Tests prove it equals the recorded pre-Red-Zone
        layouts.

    Raises
    ------
    ValueError
        The method is unknown, actor_input_schema is not the int 1 or 2, the
        sharing or normalization value is unknown, or a non-PQN method names
        "layer_norm". Raised before tracing.

    Notes
    -----
    Callers must not change this shared cached template; restore creates its
    own placement targets. The cache is typed, so True never reuses the
    template cached for 1 and is refused like any other non-int schema.
    """
    method = validate_training_method(method)
    sharing = _parameter_sharing(parameter_sharing)
    if (
        type(actor_input_schema) is not int
        or actor_input_schema not in ACTOR_FEATURE_SIZES
    ):
        raise ValueError("actor_input_schema must be 1 or 2.")
    if _pqn_normalization(normalization) != "batch_norm" and method != "pqn_vdn":
        raise ValueError("Only PQN-VDN models have a normalization choice")
    if method == "qmix":
        template = qmix_actor_template(QMIXConfig(parameter_sharing=sharing))
    elif method == "pqn_vdn":
        template = _pqn_actor_item(
            pqn_actor_template(parameter_sharing=sharing, normalization=normalization)
        )
    else:
        template = jax.eval_shape(
            partial(
                initialize_ppo,
                config=PPOConfig(parameter_sharing=sharing),
                method=validate_ppo_method(method),
            ),
            jax.random.key(0),
        ).actor_params
    width = ACTOR_FEATURE_SIZES[actor_input_schema]
    if width == ACTOR_FEATURE_SIZE:
        return template

    def resize(leaf: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        """Give one abstract leaf the schema's input width on its input axes."""
        shape = tuple(
            width if size == ACTOR_FEATURE_SIZE else size for size in leaf.shape
        )
        return jax.ShapeDtypeStruct(shape, leaf.dtype)

    return jax.tree.map(resize, template)


def _check_actor_provenance(metadata: object, method: str) -> dict[str, Any]:
    """Validate saved actor counts and its single checkpoint or capture origin.

    Reuse this admission check for numerical export and anonymous byte copies.
    Extra finite metadata remains allowed here; anonymous export classifies it
    separately. This reads no payloads and performs no numerical work.
    """
    context = _object(metadata, "Actor provenance")
    if not {"run_id", "seed", "env_steps"} <= context.keys():
        raise ValueError("Actor provenance needs run_id, seed and env_steps")
    if not isinstance(context["run_id"], str) or not context["run_id"]:
        raise ValueError("Actor run ID must be nonempty")
    counts = (
        ("seed", "env_steps", "optimizer_steps")
        if method in ("qmix", "pqn_vdn")
        else (
            "seed",
            "env_steps",
        )
    )
    for name in counts:
        if type(context.get(name)) is not int or context[name] < 0:
            raise ValueError(f"Actor {name} must be a nonnegative integer")
    if ("checkpoint_id" in context) == ("capture_origin" in context):
        raise ValueError(
            "Actor provenance needs exactly one checkpoint_id or capture_origin"
        )
    if "checkpoint_id" in context:
        _digest(context["checkpoint_id"], "Origin checkpoint")
    else:
        origin = _object(context["capture_origin"], "Capture origin")
        if set(origin) != {"capture_id", "update_index", "content_binding", "epsilon"}:
            raise ValueError("Capture origin fields differ")
        if any(
            type(origin[name]) is not int or origin[name] < 0
            for name in ("capture_id", "update_index")
        ):
            raise ValueError("Capture identity and update must be nonnegative integers")
        from marl_battlegrounds.training._content import TrainingContentBinding

        TrainingContentBinding.model_validate_json(
            _json_bytes(origin["content_binding"])
        )
        epsilon = origin["epsilon"]
        if is_ppo_method(method):
            if epsilon is not None:
                raise ValueError("A PPO capture has no epsilon")
        elif (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon)
            or not 0 <= epsilon <= 1
        ):
            raise ValueError("A Q capture needs its sampled epsilon within [0, 1]")
    return context


def export_system(
    actor_variables: Tree,
    destination: str | Path,
    *,
    metadata: dict[str, object],
    input_scale: float = 1.0,
    spawn_frame: str,
    method: str = "mappo",
    parameter_sharing: str = "all",
    normalization: str = "batch_norm",
) -> Path:
    """Publish a standalone immutable actor artifact (sampled PPO, greedy Q-values).

    Parameters
    ----------
    actor_variables : PyTree
        Finite variables matching the selected method's actor schema. For QMIX
        this is the Q-network parameters only, never epsilon or a mixer. For
        PQN-VDN it is the network's PQNInferenceVariables (parameters and
        BatchNorm statistics), never epsilon, optimizer state or kept rows.
    destination : str or Path
        Exact output directory with an existing parent. Matching exports are
        reused; different weights, scale or provenance at this path are rejected.
    metadata : dict
        Finite JSON provenance with run_id, nonnegative integer seed and
        env_steps, and either the originating checkpoint_id or capture_origin.
        A capture origin records capture_id, update_index, the content binding
        and sampled exploration (epsilon, or None for PPO). It does not claim a
        complete learner checkpoint exists. QMIX and PQN-VDN also need a
        nonnegative integer optimizer_steps. Extra facts are retained.
    input_scale : float, default=1.0
        Finite positive factor applied before the actor's first Dense layer.
        Use the originating learner config's value. It is part of
        inference identity; changing it does not rewrite weights or the raw
        input schema.
    spawn_frame : str
        Required. Spawn frame the weights were trained in: "world" or "left".
        Use the originating learner config's value. There is no default, because
        an export writes a durable identity: a guessed frame would silently
        label world weights as left, or the reverse. "left" is written into
        the description and the inference identity; "world" writes no key, so
        world exports keep their historical description bytes. PQN-VDN
        exports always write the frame.
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Method that trained these weights. Its model schema is the export's
        sole method authority. Equal actor shapes/bytes do not make different
        methods interchangeable; an existing different-method path is rejected.
        QMIX and PQN-VDN exports also record epsilon 0.0 and tie_rule
        "first_legal_maximum": loading one gives the greedy System.
    parameter_sharing : {"all", "class", "none"}, default="all"
        Actor weight ownership used in training. Only grouped modes write a
        "parameter_sharing" key.
    normalization : {"batch_norm", "layer_norm"}, default="batch_norm"
        PQN-VDN hidden-layer normalization used in training
        (``PQNConfig.normalization``). Only "layer_norm" writes a
        "normalization" key, so BatchNorm exports keep their historical
        description bytes. Other methods accept only "batch_norm".

    Returns
    -------
    Path
        Absolute immutable export directory, after durable publication.

    Raises
    ------
    TypeError
        PQN-VDN variables are not a PQNInferenceVariables record.
    ValueError
        Variables, scale, frame or provenance are invalid (for PQN-VDN this
        includes a negative BatchNorm running variance), or the destination
        conflicts. Weights shaped for historical actor input schema 1 (5,164
        features, saved before Red Zone) are refused with a message saying to
        keep the original artifact, which load_system reads directly.
    OSError
        A required directory or payload cannot be read or written.

    Notes
    -----
    Host-only. No map preparation, critic, optimizer or game state is required.
    A PQN-VDN export saves the same actor item, with the same actor_digest, as
    the learner checkpoint it came from. Every TypeError or ValueError above
    is raised before any file or directory is written. Exports always record
    the current schemas. Caught write failures remove only this call's
    unfinished tree and ownership receipt. A renamed complete actor remains
    after a later directory-sync failure; retrying with identical inputs checks
    and reuses it. Abrupt process exits may leave unpublished data for the
    owning run or model collection to check under its existing lock.
    """
    method = validate_training_method(method)
    sharing = _parameter_sharing(parameter_sharing)
    normalization = _pqn_normalization(normalization)
    if normalization != "batch_norm" and method != "pqn_vdn":
        raise ValueError("Only PQN-VDN models have a normalization choice")
    if method == "pqn_vdn":
        if not isinstance(cast(object, actor_variables), PQNInferenceVariables):
            raise TypeError("A PQN-VDN export takes PQNInferenceVariables")
        actor_variables = _pqn_actor_item(actor_variables)
    input_scale = _input_scale(input_scale)
    spawn_frame = _spawn_frame(spawn_frame)
    context = _check_actor_provenance(metadata, method)
    layout = _layout(actor_variables)
    if normalization == "batch_norm" and layout == _layout(
        _actor_template(method, 1, sharing)
    ):
        raise ValueError(
            "These weights use historical actor input schema 1 (5,164 features). "
            "Export writes current actors only; keep the original artifact, "
            "which load_system reads directly."
        )
    if layout != _layout(
        _actor_template(method, parameter_sharing=sharing, normalization=normalization)
    ):
        raise ValueError("Actor variables differ from the selected model schema")
    if not all(
        bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(actor_variables)
    ):
        raise ValueError("Actor variables must be finite")
    if method == "pqn_vdn" and not _nonnegative_variances(
        actor_variables["batch_stats"]
    ):
        raise ValueError("A PQN-VDN running variance is negative")
    target = Path(destination).absolute()
    parent = _directory(target.parent)
    digest = tree_digest(actor_variables)
    if target.exists():
        saved = read_checkpoint_details(target)
        if (
            saved["kind"] == "actor"
            and saved["schemas"] == checkpoint_schemas(method)
            and saved["actor_digest"] == digest
            and _actor_input_scale(saved) == input_scale
            and _actor_spawn_frame(saved) == spawn_frame
            and _actor_parameter_sharing(saved) == sharing
            and _actor_normalization(saved) == normalization
            and saved["metadata"] == context
        ):
            return target
        raise ValueError("Actor destination already contains a different artifact")
    with _pending_write(
        parent, kind="actor", attempt_id=context.get("attempt_id")
    ) as temporary:
        _save_arrays(temporary / "actor", actor_variables)
        details: dict[str, Any] = {
            "schema_version": 1,
            "kind": "actor",
            "schemas": checkpoint_schemas(method),
            "metadata": context,
            "actor_layout": _layout(actor_variables),
            "actor_digest": digest,
            "input_scale": input_scale,
            "files": _inventory(temporary),
        }
        if sharing != "all":
            details["parameter_sharing"] = sharing
        if normalization != "batch_norm":
            details["normalization"] = normalization
        if method in ("qmix", "pqn_vdn"):
            details["epsilon"] = 0.0
            details["tie_rule"] = QMIX_TIE_RULE
        if spawn_frame != "world" or method == "pqn_vdn":
            details["spawn_frame"] = spawn_frame
        details["checkpoint_id"] = sha256(_json_bytes(details)).hexdigest()
        _atomic_json(temporary / _ACTOR_DESCRIPTION, details)
        _sync_tree(temporary)
        temporary.rename(target)
        _sync_dir(parent)
        return target


def export_actor(
    checkpoint: str | Path, *, output_dir: str | Path, anonymous: bool = False
) -> Path:
    """Copy one saved model into an immutable, standalone actor folder.

    Parameters
    ----------
    checkpoint : str or Path
        Existing actor or complete learner checkpoint. Verify its description
        and every payload hash first. A pruned learner cannot supply weights.
    output_dir : str or Path
        Exact destination, relative to the working directory when needed.
        Create missing parents after checking the input. Reuse matching immutable
        bytes; reject a different artifact, symlink or non-directory there.
    anonymous : bool, default=False
        True prepares a separately identified copy with unchanged model bytes.
        Classify saved provenance and scan retained text before any output write.
        Capture run labels and audit paths may be replaced by stable aliases.
        Checkpoint origins retain their original opaque run ID and checked
        description. Unsafe or unclassified required metadata is refused.
        Ordinary trainer checkpoints often contain private host paths; this
        narrow route does not silently strip those records or approve release.

    Returns
    -------
    Path
        Absolute standalone actor folder. Ordinary actor copies keep every byte
        and identity. Learner extraction retains its origin and inference identity.
        Anonymous current-schema copies receive a new ID and a metadata scan
        record; derived_from names the input artifact. Historical schema-1 actors
        are copied unchanged only if their required metadata already passes.

    Raises
    ------
    ValueError, OSError
        Input hashes or metadata differ, source bytes change during copying,
        destination conflicts, or a file cannot be read or written.

    Notes
    -----
    No arrays, optimizer, replay buffer or learner are restored. Full learner
    verification still reads its payload from disk. Only actor bytes are copied,
    including PQN running statistics. Q exports play greedily. Schema, sharing,
    scale, frame and known exposure provenance remain unchanged. Published
    artifacts survive later sync failures; unpublished writes clean themselves.
    The caller must serialize writers to the same destination.
    """
    if type(anonymous) is not bool:
        raise TypeError("anonymous must be a boolean")
    source = _directory(Path(checkpoint).expanduser()).resolve()
    details = read_checkpoint_details(source)
    target = Path(output_dir).expanduser().absolute()
    from marl_battlegrounds.evaluation.revision import (
        _check_output_path,  # pyright: ignore[reportPrivateUsage]
    )

    _check_output_path(target)
    if source in target.resolve().parents:
        raise ValueError("Actor output cannot be inside its immutable source")
    exported = details["kind"] == "actor"
    actor: dict[str, Any]
    if exported:
        actor = details
    else:
        method = _schema_method(details["schemas"])
        metadata = details["metadata"]
        context = {
            "run_id": metadata["run_id"],
            "seed": metadata["config"]["seed"],
            "env_steps": details["counters"]["env_steps"],
            "checkpoint_id": details["checkpoint_id"],
            "source": metadata.get("source"),
            "dependencies": metadata.get("dependencies"),
            "attempt_id": metadata.get("attempt_id"),
            "initial_actor": metadata.get("initial_actor"),
            "opponent_members": metadata.get("used_opponent_members", []),
            "partner_members": metadata.get("used_partner_members", []),
            "pinned_opponent": (
                None
                if details["collection"].get("opponent_members")
                else details["collection"].get("pinned_opponent")
            ),
        }
        if method in ("qmix", "pqn_vdn"):
            context["optimizer_steps"] = details["counters"]["updates"]
        actor = {
            "schema_version": details["schema_version"],
            "kind": "actor",
            "schemas": details["schemas"],
            "metadata": context,
            "actor_layout": details["actor_layout"],
            "actor_digest": details["actor_digest"],
            "input_scale": _actor_input_scale(details),
            "files": {
                key: value
                for key, value in details["files"].items()
                if key.startswith("actor/")
            },
        }
        sharing = _actor_parameter_sharing(details)
        frame = _actor_spawn_frame(details)
        if sharing != "all":
            actor["parameter_sharing"] = sharing
        if _actor_normalization(details) != "batch_norm":
            actor["normalization"] = _actor_normalization(details)
        if method in ("qmix", "pqn_vdn"):
            actor.update(epsilon=0.0, tie_rule=QMIX_TIE_RULE)
        if frame != "world" or method == "pqn_vdn":
            actor["spawn_frame"] = frame
        actor["checkpoint_id"] = sha256(_json_bytes(actor)).hexdigest()
        _check_actor_origin(actor, details)
    extra: dict[str, bytes] = {}
    if anonymous:
        from marl_battlegrounds.training._anonymous import prepare_anonymous_actor

        actor, extra = prepare_anonymous_actor(source, actor, details)
    if target.exists() or target.is_symlink():
        if read_checkpoint_details(target) == actor:
            return target.resolve()
        raise ValueError(f"Actor destination contains a different artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    parent = _directory(target.parent)
    with _pending_write(
        parent, kind="actor", attempt_id=actor["metadata"].get("attempt_id")
    ) as temporary:
        if exported:
            shutil.copytree(source, temporary, dirs_exist_ok=True)
        else:
            shutil.copytree(source / "actor", temporary / "actor")
        for name, raw in extra.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        if not exported or actor != details:
            _atomic_json(temporary / _ACTOR_DESCRIPTION, actor)
        # Recheck copied bytes and descriptions without restoring arrays or
        # repeating full source-payload verification. Rewriting derivative
        # metadata must not hide a source edit made during copying.
        if read_checkpoint_description(source) != details:
            raise ValueError("Source description changed during actor copying")
        if anonymous and exported and extra:
            from marl_battlegrounds.training._checkpoint_metadata import (
                _actor_origin_directory,
            )

            origin = _actor_origin_directory(source, details)
            if (origin / _DESCRIPTION).read_bytes() != next(iter(extra.values())):
                raise ValueError("Source origin changed during actor copying")
        if read_checkpoint_details(temporary) != actor:
            raise ValueError("Copied actor changed before publication")
        _sync_tree(temporary)
        temporary.rename(target)
        _sync_dir(parent)
    return target.resolve()


def artifact_identity(path: str | Path) -> dict[str, Any]:
    """Read verified actor identity/provenance without restoring numerical state.

    Parameters
    ----------
    path : str or Path
        A standalone actor export or a complete learner checkpoint directory.
        A pruned learner, whose payload files retention removed, is refused.

    Returns
    -------
    dict[str, Any]
        Direct actor_digest, weight_digest, input_scale, spawn_frame,
        checkpoint_id, schemas, run_id, seed and env_steps fields plus the
        original metadata. actor_digest binds model, weights, inference scale
        and spawn frame. MAPPO at scale 1.0 in the world frame keeps its
        historical weight digest. weight_digest always identifies only the
        saved variables. schemas identifies the method and the actor input
        schema (1 for artifacts saved before Red Zone, whose identities keep
        their original bytes). seed and env_steps come from an export's
        metadata or a learner's config and counters; seed may be None. PPO
        results carry no separate method field. QMIX and PQN-VDN results add
        "method" ("qmix" or "pqn_vdn") and the integer "optimizer_steps" (the
        export's recorded count, or a learner's saved "updates" counter),
        which validation summaries and selection use.

    Raises
    ------
    ValueError
        read_checkpoint_details rejects the artifact: its description is
        invalid (see read_checkpoint_description), or a payload file is
        missing, changed or unexpected.
    OSError
        A saved file cannot be read.

    Notes
    -----
    Neither identity proves competence. All file hashes are checked; this
    operation may read a full learner payload from disk. Missing historical
    scale means 1.0 and a missing frame means "world". Host-only: no array is
    restored and no file is changed.
    """
    return _artifact_identity(read_checkpoint_details(path))


def load_system(checkpoint: str | Path) -> System:
    """Load an exact saved actor: a sampled PPO System or a greedy Q-value System.

    Parameters
    ----------
    checkpoint : str or Path
        Standalone actor export or complete learner checkpoint directory.

    Returns
    -------
    System
        Frozen numerical actor variables with their saved input scale and verified
        inference digest as the checkpoint label. The validated model schema
        chooses the actor architecture. Each evaluator creates fresh recurrent
        memory or empty feedforward memory. A PPO
        actor samples legal masked actions; a QMIX or PQN-VDN actor plays
        greedily (epsilon 0, first legal maximum), PQN-VDN with its saved
        BatchNorm statistics. No critic, mixer, target, replay, optimizer,
        kept row or training-only input array is restored. Integrity checks still
        read every saved payload, including learner state. No map preparation or
        full run directory is needed.

    Raises
    ------
    ValueError
        Artifact files, schemas, shapes, dtypes, digest or finite arrays differ,
        or a PQN-VDN actor has a negative BatchNorm running variance.
    OSError
        Required artifact bytes cannot be read.

    Notes
    -----
    Host-only; reads files and restores actor arrays onto the process's selected
    device. It needs the training extra but does not run an actor or learner.
    A frozen actor identity makes no claim about learned competence.
    Historical artifacts without an input scale keep their original scale 1.0
    and weight-digest identity. Learner checkpoints use their config's ppo,
    qmix or pqn input_scale;
    actor exports use their explicit saved scale. The saved spawn frame is
    restored the same way and is never inferred from a model's results; a
    missing frame means "world". A PQN-VDN model is rebuilt with its saved
    hidden normalization (an export's "normalization" key or its learner
    config's ``pqn.normalization``); a missing value means "batch_norm".
    Neither route rewrites weights. The saved actor input schema picks the
    template and the System route: an artifact
    saved before Red Zone (schema 1, 5,164 features) is restored with the
    schema-1 template and built with actor_input_schema=1, so it plays
    through the schema-1 hooks exactly as before. Its checkpoint label (the
    inference digest) keeps its bytes; its registration ID is new because
    the hook is new.
    """
    root = _directory(Path(checkpoint))
    details = read_checkpoint_details(root)
    method = _schema_method(details["schemas"])
    schema = cast(int, details["schemas"]["actor_input"])
    sharing = _actor_parameter_sharing(details)
    normalization = _actor_normalization(details)
    template = _actor_template(method, schema, sharing, normalization)
    if details["actor_layout"] != _layout(template):
        raise ValueError("Actor artifact schema differs from the installed model")
    actor = _restore_arrays(root / "actor", template, None)
    if tree_digest(actor) != details["actor_digest"] or not all(
        bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(actor)
    ):
        raise ValueError("Restored actor is nonfinite or its digest differs")
    if method == "pqn_vdn" and not _nonnegative_variances(actor["batch_stats"]):
        raise ValueError("A PQN-VDN running variance is negative")
    display_name = details.get("metadata", {}).get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        display_name = None
    if method == "qmix":
        return make_qmix_system(
            actor,
            name=display_name or "QMIX",
            epsilon=0.0,
            checkpoint=_inference_digest(details),
            input_scale=_actor_input_scale(details),
            spawn_frame=_actor_spawn_frame(details),
            actor_input_schema=schema,
            parameter_sharing=sharing,
        )
    if method == "pqn_vdn":
        return make_pqn_system(
            _pqn_network(actor),
            name=display_name or "PQN-VDN",
            epsilon=0.0,
            checkpoint=_inference_digest(details),
            input_scale=_actor_input_scale(details),
            spawn_frame=_actor_spawn_frame(details),
            actor_input_schema=schema,
            parameter_sharing=sharing,
            normalization=normalization,
        )
    return make_ppo_system(
        actor,
        method=method,
        name=display_name,
        checkpoint=_inference_digest(details),
        input_scale=_actor_input_scale(details),
        spawn_frame=_actor_spawn_frame(details),
        actor_input_schema=schema,
        parameter_sharing=sharing,
    )


def initial_actor_variables(initial: Tree, template: Tree) -> Tree:
    """Check copied actor arrays against a fresh learner's actor tree.

    ``initial`` is an already loaded numerical actor, including PQN running
    statistics. ``template`` is the fresh actor with the requested layout.
    Return the immutable input tree without copying device arrays. Shape,
    dtype, structure or nonfinite values raise ValueError. Optimizer, critic,
    mixer, replay and random state never enter this operation.
    """
    import jax
    import jax.numpy as jnp

    if _layout(initial) != _layout(template) or jax.tree.structure(
        initial
    ) != jax.tree.structure(template):
        raise ValueError("Initial actor structure, shapes or dtypes differ")
    if not all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(initial)):
        raise ValueError("Initial actor weights must be finite")
    return initial


def load_initial_actor(
    path: str | Path,
    *,
    method: str,
    input_scale: float,
    spawn_frame: str,
    parameter_sharing: str = "all",
    normalization: str = "batch_norm",
) -> tuple[System, dict[str, Any]]:
    """Load compatible built-in weights before a new training run opens files.

    ``path`` names an actor export or full learner checkpoint. Only its actor
    is imported. The method, current actor schema, input scale, spawn frame,
    parameter sharing and (for PQN-VDN) hidden normalization must match;
    load_system checks layout, dtypes, payload hashes and values.
    Return that System and its verified description for provenance. This is a
    warm start with fresh learner state, not a continuation. Trainer setup reuses
    the same loaded artifact across roles; ordinary calls load independently.
    Generic Systems remain supported through opponents and researcher-owned loops.
    """
    details = read_checkpoint_description(path)
    if _schema_method(details["schemas"]) != method:
        raise ValueError("Initial actor method differs from the new learner")
    if details["schemas"] != checkpoint_schemas(method):
        raise ValueError("Initial actor schema differs from the installed learner")
    if _actor_input_scale(details) != input_scale:
        raise ValueError("Initial actor input_scale differs from the new learner")
    if _actor_spawn_frame(details) != spawn_frame:
        raise ValueError("Initial actor spawn_frame differs from the new learner")
    if _actor_parameter_sharing(details) != _parameter_sharing(parameter_sharing):
        raise ValueError("Initial actor parameter_sharing differs from the new learner")
    if _actor_normalization(details) != _pqn_normalization(normalization):
        raise ValueError("Initial actor normalization differs from the new learner")
    from marl_battlegrounds._method_loading import load_method

    # Training setup may already hold this same checked actor as a pool member.
    return cast("System", load_method(str(Path(path).expanduser().absolute()))), details


def checkpoint_dependencies(method: str = "mappo") -> dict[str, str]:
    """Read installed numerical dependency versions for the runner's identity.

    Parameters
    ----------
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Training method. QMIX also records Flashbax; PPO and PQN-VDN record
        exactly the historical set, so PPO identities keep their bytes.

    Returns
    -------
    dict[str, str]
        Python, JAX, JAXlib, NumPy, Flax, Optax and Orbax versions, plus
        Flashbax for QMIX only.

    Raises
    ------
    PackageNotFoundError
        An optional training package is missing.
    ValueError
        method is unknown.

    Notes
    -----
    Reads package metadata only; installs nothing and initializes no device.
    """
    import platform

    names = ["jax", "jaxlib", "numpy", "flax", "optax", "orbax-checkpoint"]
    if validate_training_method(method) == "qmix":
        names.append("flashbax")
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in names},
    }


def runtime_identity(method: str = "mappo") -> dict[str, object]:
    """Read the current imported-source identity and installed dependencies.

    method selects the dependency set (see checkpoint_dependencies); the
    default keeps the PPO record. The runner calls this at setup under a
    stable source snapshot, then reuses the result as its source/dependencies
    equality assertion for resume. Git or installed-package discovery reads
    files and may run read-only Git commands; no package is installed, device
    initialized or artifact written.
    """
    from marl_battlegrounds.evaluation.revision import discover_code_revision_v2

    return {
        "source": discover_code_revision_v2().model_dump(mode="json"),
        "dependencies": checkpoint_dependencies(method),
    }


def continuation_source_compatibility(
    parent_source: object, current_source: object
) -> dict[str, object]:
    """Record a source change without treating a version string as incompatibility.

    Both inputs are actual nonempty source identities. The complete checkpoint
    restore still checks schema, array layout, input and game contracts, content,
    dependencies and execution settings. This record alone proves neither
    numerical equivalence nor learning quality. Old validation evidence keeps
    its original source identity. No file is written or old identity replaced.
    """
    parent = _object(parent_source, "Parent source")
    current = _object(current_source, "Current source")
    if not parent or not current:
        raise ValueError("Continuation needs complete source identities")
    return {
        "parent_source": parent,
        "child_source": current,
        "qualification": "Exact source"
        if parent == current
        else "Checkpoint contract checks",
    }
