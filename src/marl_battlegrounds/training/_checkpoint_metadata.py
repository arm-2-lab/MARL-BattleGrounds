"""Read saved actor and learner descriptions without numerical imports.

This is the shared metadata owner for checkpoint loading, launch previews and
recovery. It checks descriptions, identities, schema versions and safe paths;
payload verification and numerical restore stay in checkpoints. Existing public
checkpoint names re-export these functions. No read changes saved files.
"""

from __future__ import annotations

# Shared metadata helpers are called by collection and recovery.
# pyright: reportUnusedFunction=false
import json
import math
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.baselines.config import (
    ACTION_SCHEMA_VERSION,
    ACTOR_INPUT_SCHEMA_VERSION,
    PQN_KEY_SCHEMA_VERSION,
    PQN_NORMALIZATION_SCHEMA_VERSION,
    PQN_RECENT_WINDOW_SCHEMA_VERSION,
    PQN_TIE_RULE,
    QMIX_KEY_SCHEMA_VERSION,
    QMIX_REPLAY_SCHEMA_VERSION,
    QMIX_TIE_RULE,
    TRAINING_STATE_SCHEMA_VERSION,
    PPOConfig,
    _parameter_sharing,
    _pqn_normalization,
    validate_ppo_method,
)
from marl_battlegrounds.baselines.methods import (
    is_ppo_method,
    method_settings_field,
    validate_training_method,
)
from marl_battlegrounds.training._compilation import execution_identity
from marl_battlegrounds.training.config import TRAINING_KEY_SCHEMA_VERSION

# pyright: reportPrivateUsage=false

_DESCRIPTION = "checkpoint_details.json"

_ACTOR_DESCRIPTION = "actor_details.json"

_SCHEMAS = {
    "checkpoint": 1,
    "model": "recurrent_mappo_128",
    "actor_input": ACTOR_INPUT_SCHEMA_VERSION,
    "training_state": TRAINING_STATE_SCHEMA_VERSION,
    "action": ACTION_SCHEMA_VERSION,
    "collection_keys": TRAINING_KEY_SCHEMA_VERSION,
    "learner_keys": 1,
}

_MODELS = {
    "mappo": "recurrent_mappo_128",
    "ippo": "recurrent_ippo_128",
    "ff_mappo": "feedforward_mappo_128x128",
    "ff_ippo": "feedforward_ippo_128x128",
    "qmix": "recurrent_qmix_256",
    "pqn_vdn": "recurrent_pqn_vdn_512",
}

_QMIX_SCHEMAS = {
    "checkpoint": 1,
    "model": _MODELS["qmix"],
    "actor_input": ACTOR_INPUT_SCHEMA_VERSION,
    "training_state": TRAINING_STATE_SCHEMA_VERSION,
    "action": ACTION_SCHEMA_VERSION,
    "collection_keys": TRAINING_KEY_SCHEMA_VERSION,
    "qmix_keys": QMIX_KEY_SCHEMA_VERSION,
    "replay": QMIX_REPLAY_SCHEMA_VERSION,
}

_QMIX_COUNTERS = {"updates", "env_steps", "completed_blocks", "learning_blocks"}

_PQN_SCHEMAS = {
    "checkpoint": 1,
    "model": _MODELS["pqn_vdn"],
    "actor_input": ACTOR_INPUT_SCHEMA_VERSION,
    "action": ACTION_SCHEMA_VERSION,
    "collection_keys": TRAINING_KEY_SCHEMA_VERSION,
    "pqn_keys": PQN_KEY_SCHEMA_VERSION,
    "recent_window": PQN_RECENT_WINDOW_SCHEMA_VERSION,
    "normalization": PQN_NORMALIZATION_SCHEMA_VERSION,
}

_PQN_COUNTERS = _QMIX_COUNTERS

_ACTOR_INPUT_1_SCHEMAS: dict[str, dict[str, int | str]] = {
    **{
        method: {
            "checkpoint": 1,
            "model": _MODELS[method],
            "actor_input": 1,
            "training_state": 1,
            "action": 1,
            "collection_keys": 1,
            "learner_keys": 1,
        }
        for method in ("mappo", "ippo", "ff_mappo", "ff_ippo")
    },
    "qmix": {
        "checkpoint": 1,
        "model": _MODELS["qmix"],
        "actor_input": 1,
        "training_state": 1,
        "action": 1,
        "collection_keys": 1,
        "qmix_keys": 1,
        "replay": 1,
    },
    "pqn_vdn": {
        "checkpoint": 1,
        "model": _MODELS["pqn_vdn"],
        "actor_input": 1,
        "action": 1,
        "collection_keys": 1,
        "pqn_keys": 1,
        "recent_window": 1,
        "normalization": 1,
    },
}

_CONTEXT = {
    "run_id",
    "attempt_id",
    "parent_checkpoint",
    "config",
    "source",
    "dependencies",
}


def _json_bytes(value: object) -> bytes:
    """Encode finite JSON deterministically, including one final newline."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _object(value: object, name: str) -> dict[str, Any]:
    """Require a JSON object with string keys and copy its finite contents."""
    if not isinstance(value, dict) or any(
        not isinstance(k, str) for k in cast(dict[object, object], value)
    ):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], json.loads(_json_bytes(cast(object, value))))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a finite JSON object, rejecting duplicate keys and symbolic links."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular file is missing: {path.name}")

    return _parse_json(path.read_bytes(), path.name)


def _parse_json(value: str | bytes, name: str) -> dict[str, Any]:
    """Decode a finite JSON object with no repeated keys, including nested objects.

    value contains JSON text or UTF-8 bytes; name identifies its source in errors.
    No files are read or written. Malformed or ambiguous records raise ValueError.
    """

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        """Reject ambiguous repeated object keys."""
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        """Reject NaN and Infinity accepted by the default JSON reader."""
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return _object(
        json.loads(value, object_pairs_hook=pairs, parse_constant=constant),
        name,
    )


def _digest(value: object, name: str) -> str:
    """Require one complete lowercase SHA256 identity."""
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA256 identity")
    return value


def _input_scale(value: object) -> float:
    """Validate one finite positive network scale through the PPO owner."""
    return float(PPOConfig(input_scale=cast(float, value)).input_scale)


def _spawn_frame(value: object) -> str:
    """Validate one spawn frame name ("world" or "left") via the PPO owner."""
    return PPOConfig(spawn_frame=cast(str, value)).spawn_frame


def _settings_block(config: object) -> dict[str, Any]:
    """Return a saved config's method settings: its "ppo", "qmix" or "pqn" object.

    config must be a JSON learner config; ``methods.method_settings_field``
    names the block. A missing block is an empty object, so historical PPO
    configs keep their defaults. Raises ValueError.
    """
    value = _object(config, "Learner config")
    name = method_settings_field(_config_method(value))
    return _object(value.get(name, {}), f"{name.upper()} config")


def _config_spawn_frame(config: object) -> str:
    """Read a learner's spawn frame; historical configs without it use "world".

    PPO configs keep it in "ppo", QMIX configs in "qmix", PQN-VDN configs in
    "pqn" (always present there).
    """
    return _spawn_frame(_settings_block(config).get("spawn_frame", "world"))


def _config_method(config: object) -> str:
    """Read a saved learner method; only historical absence means MAPPO.

    config must be a JSON object. Reject unknown method values through the
    shared method check (PPO names, "qmix" and "pqn_vdn"). No saved object is
    changed and no device work runs. Array shapes never decide which method produced a
    checkpoint.
    """
    value = _object(config, "Learner config")
    return validate_training_method(value.get("method", "mappo"))


def _schema_method(schemas: object) -> str:
    """Return the method named by one complete supported schema dictionary.

    schemas must exactly match either checkpoint_schemas for a known model
    (the current dictionary) or that model's frozen pre-Red-Zone dictionary in
    _ACTOR_INPUT_1_SCHEMAS (actor input 1, and training state 1 where the
    method has it). A mixed dictionary, such as actor input 2 with training
    state 1, matches neither and is rejected. Reject unknown models, altered
    versions/types, missing fields and extra fields with ValueError. Each
    method's own dictionary sets the types, so Boolean or float versions (such
    as a forged ``"replay": true``) do not stand in for integers. This
    host-only check reads no arrays and changes no input. Use
    _require_current_schemas where only the current dictionary is allowed.
    """
    value = _object(schemas, "Checkpoint schemas")
    for method, model in _MODELS.items():
        for expected in (checkpoint_schemas(method), _ACTOR_INPUT_1_SCHEMAS[method]):
            if (
                value.get("model") == model
                and value == expected
                and all(
                    type(value[key]) is type(item) for key, item in expected.items()
                )
            ):
                return method
    raise ValueError("Checkpoint model or schema differs from this implementation")


def _require_current_schemas(details: dict[str, Any]) -> None:  # pyright: ignore[reportUnusedFunction]
    """Refuse to resume training from a checkpoint saved before Red Zone.

    Parameters
    ----------
    details : dict
        A checkpoint description from read_checkpoint_details, whose
        "schemas" entry is a supported current or historical dictionary.

    Raises
    ------
    ValueError
        The saved schemas are the frozen pre-Red-Zone dictionary (actor input
        schema 1). The message says that this version cannot resume the run
        and that the original source environment that created it (for example
        its launch package) can; load_system still reads its actor. Also
        raised, by _schema_method, for an unsupported dictionary.

    Notes
    -----
    This is the one owner of the historical-resume refusal. restore_checkpoint
    calls it right after the checkpoint kind check, before any array is read.
    read_checkpoint_description never calls it, so launch, ancestry and
    analysis can still read old descriptions. Host-only; reads no files.
    """
    schemas = _object(details["schemas"], "Checkpoint schemas")
    if schemas != checkpoint_schemas(_schema_method(schemas)):
        raise ValueError(
            "This checkpoint was saved before Red Zone scoring (historical "
            "actor input schema 1, 5,164 features). This version cannot resume "
            "its training. Resuming it needs the original source environment "
            "that created it, for example its launch package; load_system "
            "still reads its actor."
        )


def saved_training_config(details: dict[str, Any]) -> dict[str, Any]:
    """Return a learner checkpoint's saved config with its historical meaning fixed.

    Parameters
    ----------
    details : dict
        A checkpoint description from read_checkpoint_details with
        metadata.config, the JSON training config saved with the run.

    Returns
    -------
    dict
        A new top-level dictionary. A QMIX or PQN-VDN config keeps its
        "qmix" or "pqn" block exactly as saved and gains no "ppo" block. For
        PPO it also returns a new ppo dictionary. Missing method becomes "mappo";
        missing ppo.input_scale becomes 1.0. ppo.spawn_frame is filled with
        "world" when the saved config has no ppo block or no such key, because
        a checkpoint saved before the setting existed trained in raw world
        coordinates whatever the current PPOConfig default is. Missing
        ppo.value_normalization is filled with False because historical critics
        predicted raw reward units. Missing validation_opponents and
        slot_diagnostic_actor become None, keeping the historical panel route.
        Missing red_zone_depth becomes 0.0, because a run saved before the
        Red Zone rule scored one point per death whatever the current
        TrainConfig default (5.0) is. Every other
        saved value is returned as saved. Missing retain_all_actors is False;
        older runs did not promise every actor. Other absent keys keep the current
        TrainConfig and PPOConfig defaults through config_from_dict.

    Raises
    ------
    ValueError, TypeError
        metadata.config or its ppo block is not a JSON object.

    Notes
    -----
    Host-only; reads no file and changes neither details nor the saved config.
    The runner decodes a resumed run's settings through this function, so an
    explicit resume config that names a different frame is rejected by the
    runner's exact-settings comparison rather than silently adopted.
    """
    config = _object(details["metadata"]["config"], "Saved training config")
    if is_ppo_method(_config_method(config)):
        ppo = dict(_object(config.get("ppo", {}), "Saved ppo settings"))
        ppo.setdefault("input_scale", 1.0)
        ppo.setdefault("spawn_frame", "world")
        ppo.setdefault("value_normalization", False)
        config["ppo"] = ppo
        config.setdefault("method", "mappo")
    config.setdefault("validation_opponents", None)
    config.setdefault("slot_diagnostic_actor", None)
    config.setdefault("red_zone_depth", 0.0)
    config.setdefault("retain_all_actors", False)
    return config


def _actor_spawn_frame(details: dict[str, Any]) -> str:
    """Read the saved spawn frame; historical descriptions without it use "world"."""
    if details["kind"] == "actor":
        return _spawn_frame(details.get("spawn_frame", "world"))
    return _config_spawn_frame(details["metadata"]["config"])


def _config_input_scale(config: object) -> float:
    """Read a learner's scale; historical configs without it use 1.0.

    PPO configs keep it in "ppo", QMIX configs in "qmix", PQN-VDN configs in
    "pqn".
    """
    return _input_scale(_settings_block(config).get("input_scale", 1.0))


def _actor_input_scale(details: dict[str, Any]) -> float:
    """Read the saved inference scale without changing historical descriptions."""
    if details["kind"] == "actor":
        return _input_scale(details.get("input_scale", 1.0))
    return _config_input_scale(details["metadata"]["config"])


def _actor_parameter_sharing(details: dict[str, Any]) -> str:
    """Read saved actor weight ownership, with all-shared historical default."""
    value = (
        details.get("parameter_sharing", "all")
        if details["kind"] == "actor"
        else _settings_block(details["metadata"]["config"]).get(
            "parameter_sharing", "all"
        )
    )
    if not isinstance(value, str):
        raise ValueError("Saved parameter_sharing must be all, class or none")
    return _parameter_sharing(value)


def _actor_normalization(details: dict[str, Any]) -> str:
    """Read a saved PQN-VDN hidden-layer normalization; absence is "batch_norm".

    details is a checked actor or learner description. Actor exports carry an
    explicit "normalization" key only for "layer_norm"; learner checkpoints
    read their config's "pqn" block. Every saved model without the key (all
    PPO and QMIX models and PQN-VDN models from before the option) used the
    original layout, "batch_norm". Raises ValueError for an unknown value.
    """
    value = (
        details.get("normalization", "batch_norm")
        if details["kind"] == "actor"
        else _settings_block(details["metadata"]["config"]).get(
            "normalization", "batch_norm"
        )
    )
    return _pqn_normalization(value)


def _directory(path: Path) -> Path:
    """Reject symbolic links along an existing directory's complete path."""
    absolute = path.absolute()
    for ancestor in (*reversed(absolute.parents), absolute):
        if ancestor.is_symlink():
            raise ValueError("Checkpoint directories must not use symbolic links")
    if not absolute.is_dir():
        raise ValueError("Checkpoint directory is missing")
    return absolute


def _relative(root: Path, value: object) -> Path:
    """Resolve a relative child without traversal or symbolic links."""
    if not isinstance(value, str):
        raise ValueError("Saved file path must be a string")
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in (".", "..") for p in relative.parts)
    ):
        raise ValueError("Saved file path must stay below its owner")
    child = root
    for part in relative.parts:
        child /= part
        if child.is_symlink():
            raise ValueError("Saved paths must not use symbolic links")
    return child


def _check_context(
    metadata: object, *, require_execution: bool = False
) -> dict[str, Any]:
    """Check scientific metadata and, for new saves, the current execution policy.

    require_execution defaults to False so historical descriptions remain
    readable. True requires the current compilation/runtime identity. Missing
    old execution metadata never receives the current policy implicitly.
    """
    value = _object(metadata, "Checkpoint metadata")
    if not value.keys() >= _CONTEXT:
        raise ValueError(f"Checkpoint metadata needs {sorted(_CONTEXT)}")
    for key in ("run_id", "attempt_id"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if value["parent_checkpoint"] is not None:
        _digest(value["parent_checkpoint"], "Parent checkpoint")
    for key in ("config", "source", "dependencies"):
        if not _object(value[key], key):
            raise ValueError(f"{key} must not be empty")
    if "selection_rule" in value and value["selection_rule"] != "point_margin":
        raise ValueError("Unknown recorded checkpoint selection rule")
    if "execution" in value:
        _object(value["execution"], "Execution identity")
    if require_execution and value.get("execution") != execution_identity():
        raise ValueError("Checkpoint needs the current execution identity")
    return value


def read_checkpoint_description(path: str | Path) -> dict[str, Any]:
    """Verify one immutable description, including a pruned ancestor's record.

    Parameters
    ----------
    path : str or Path
        An actor or learner directory. It must exist, and no part of its path
        may be a symbolic link. The learner description
        (checkpoint_details.json) is read when present, otherwise the actor
        description (actor_details.json).

    Returns
    -------
    dict[str, Any]
        The verified description exactly as saved: schema_version (always the
        integer 1), checkpoint_id, kind ("learner" or "actor"), schemas,
        metadata, actor_layout, actor_digest and the files inventory, plus the
        kind's and method's own fields (for example a learner's collection,
        layout, recording_token and counters). Payload files are not read or
        checked.

    Raises
    ------
    ValueError
        The directory is missing or uses a symbolic link; the description is
        missing, a symbolic link, not a finite JSON object or has a repeated
        key; its hash differs from checkpoint_id; its schema, kind or field
        set is not a supported one; a QMIX or PQN-VDN rule below is broken; a
        learner's saved config names a different method than its model
        schema; or an inventory path, provenance, layout, digest, input scale
        or spawn frame is invalid.

    Notes
    -----
    This checks the description's own hash, schema, metadata and safe inventory
    paths without reading payload files. This supports ancestry after retention
    removes old numerical payloads. It does not make that ancestor restorable or
    prove its payload still exists. Use read_checkpoint_details before loading
    or restoring numerical data.

    A QMIX actor export must carry input_scale, epsilon exactly 0.0 (not -0.0),
    tie_rule "first_legal_maximum" and a nonnegative integer
    metadata.optimizer_steps; spawn_frame is present when not "world". A
    PQN-VDN export carries the same fields and always its spawn_frame. QMIX and
    PQN-VDN learners must carry exactly their four counters. Both the current
    schema dictionaries and the exact pre-Red-Zone ones (actor input schema 1)
    are accepted, so old descriptions stay readable for launch, ancestry and
    analysis. Host-only; no file is changed.
    """
    root = _directory(Path(path))
    name = _DESCRIPTION if (root / _DESCRIPTION).exists() else _ACTOR_DESCRIPTION
    details = _read_json(root / name)
    identifier = _digest(details.get("checkpoint_id"), "Checkpoint ID")
    unsigned = {k: v for k, v in details.items() if k != "checkpoint_id"}
    if sha256(_json_bytes(unsigned)).hexdigest() != identifier:
        raise ValueError("Checkpoint description digest differs")
    if type(details.get("schema_version")) is not int or details["schema_version"] != 1:
        raise ValueError("Checkpoint model or schema differs from this implementation")
    method = _schema_method(details.get("schemas"))
    if details.get("kind") not in ("learner", "actor"):
        raise ValueError("Unsupported checkpoint kind")
    required = {
        "schema_version",
        "kind",
        "schemas",
        "metadata",
        "actor_layout",
        "actor_digest",
        "files",
        "checkpoint_id",
    }
    if details["kind"] == "learner":
        required |= {"collection", "layout", "recording_token", "counters"}
    elif method == "qmix":
        required |= {"input_scale", "epsilon", "tie_rule"}
        required |= {"spawn_frame"} if "spawn_frame" in details else set()
    elif method == "pqn_vdn":
        required |= {"input_scale", "spawn_frame", "epsilon", "tie_rule"}
    else:
        # Actor exports carry these optional inference settings only when set.
        required |= {key for key in ("input_scale", "spawn_frame") if key in details}
    if details["kind"] == "actor" and "parameter_sharing" in details:
        required.add("parameter_sharing")
    if details["kind"] == "actor" and "normalization" in details:
        required.add("normalization")
    if set(details) != required:
        raise ValueError("Checkpoint description fields differ from its schema")
    _actor_parameter_sharing(details)
    if _actor_normalization(details) != "batch_norm" and method != "pqn_vdn":
        raise ValueError("Only PQN-VDN models have a normalization choice")
    if method in ("qmix", "pqn_vdn") and details["kind"] == "actor":
        steps = _object(details["metadata"], "Actor provenance").get("optimizer_steps")
        if (
            type(details["epsilon"]) is not float
            or details["epsilon"] != 0.0
            or math.copysign(1.0, details["epsilon"]) < 0
            or details["tie_rule"] != QMIX_TIE_RULE
            or type(steps) is not int
            or steps < 0
        ):
            raise ValueError(
                f"{'QMIX' if method == 'qmix' else 'PQN-VDN'} export must be "
                "greedy with its optimizer count"
            )
    if method in ("qmix", "pqn_vdn") and details["kind"] == "learner":
        counters = _object(details["counters"], "Checkpoint counters")
        if set(counters) != _QMIX_COUNTERS:
            raise ValueError(
                f"{'QMIX' if method == 'qmix' else 'PQN-VDN'} checkpoint "
                "counters are malformed"
            )
    _object(details["metadata"], "Checkpoint provenance")
    if not isinstance(details["actor_layout"], list):
        raise ValueError("Checkpoint actor layout must be an array schema")
    _digest(details.get("actor_digest"), "Actor digest")
    files = _object(details.get("files"), "Checkpoint file inventory")
    for relative in files:
        _relative(root, relative)
    if details["kind"] == "learner":
        _check_context(details.get("metadata"))
        if _config_method(details["metadata"]["config"]) != method:
            raise ValueError("Checkpoint config method differs from its model schema")
    _actor_input_scale(details)
    _actor_spawn_frame(details)
    return details


def checkpoint_schemas(method: str = "mappo") -> dict[str, int | str]:
    """Return the installed checkpoint and selected model format identifiers.

    Parameters
    ----------
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Static method. Unknown values raise ValueError through the shared
        method check. MAPPO keeps its historical keys, including learner_keys.

    Returns
    -------
    dict[str, int or str]
        A fresh JSON-ready copy of the checkpoint version, model name, actor
        input (2), training-state (2) and action versions, and collection/learner
        key versions. QMIX replaces learner_keys with qmix_keys and adds its
        replay row version. PQN-VDN has no training-state entry (it keeps no
        physical state) and adds pqn_keys, recent_window and normalization
        versions. These are the same identifiers checked during restore; new
        saves and exports write them. Artifacts saved before Red Zone carry
        the frozen dictionaries with actor input 1 (and training state 1),
        which load_system reads but export and resume refuse. Changing the
        returned dictionary does not change the shared authority.

    Notes
    -----
    Reads only fixed metadata; no file access or device initialization occurs.
    """
    if validate_training_method(method) == "qmix":
        return dict(_QMIX_SCHEMAS)
    if method == "pqn_vdn":
        return dict(_PQN_SCHEMAS)
    return {**_SCHEMAS, "model": _MODELS[validate_ppo_method(method)]}


def _latest_checkpoint(root: Path) -> tuple[Path, dict[str, Any]] | None:
    """Read a run's sole latest pointer, preserving path and identity checks.

    root is an absolute run directory. No pointer means None; malformed or linked
    pointers raise ValueError. Return its safe checkpoint path and small pointer
    record without reading payloads. The caller verifies the description before
    recovery; study uses this only to choose its next ordinary trainer request.
    """
    if root.resolve() != root:
        raise ValueError("Run directory follows a link or escapes its owner")
    source = root / "latest_checkpoint.json"
    if not source.exists() and not source.is_symlink():
        return None
    pointer = _read_json(root / "latest_checkpoint.json")
    if (
        set(pointer) != {"schema_version", "checkpoint_id", "relative_path"}
        or type(pointer["schema_version"]) is not int
        or pointer["schema_version"] != 1
    ):
        raise ValueError("Run has an invalid latest checkpoint pointer")
    identifier = _digest(pointer["checkpoint_id"], "Latest checkpoint ID")
    relative = pointer["relative_path"]
    selected = _relative(root, relative)
    if Path(relative).parent != Path("checkpoints") or not (
        selected.name == identifier or selected.name.endswith(f"_{identifier}")
    ):
        raise ValueError("Latest checkpoint must be inside this run's checkpoints")
    return selected, pointer


def read_resume_checkpoint(
    path: str | Path, *, base_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Resolve a run's saved latest pointer or check an explicit checkpoint.

    path is an existing run or full learner checkpoint folder. Run folders use
    only latest_checkpoint.json, never a glob or timestamp. Check pointer schema,
    safe relative path and matching learner identity, then return the absolute
    checkpoint path and its checked description. Named and historical hash-only
    folders are accepted. Payloads remain unchecked until ordinary train restore.
    Relative paths use base_dir, or the working directory when omitted. Check
    links before normalization, so aliases cannot hide a linked saved path.
    Missing, malformed, stale, actor-only or linked inputs raise ValueError before
    any output changes. This reads only small records and starts no backend.
    """
    if not isinstance(cast(object, path), (str, Path)) or not str(path).strip():
        raise ValueError("resume_from must be a nonempty path")
    requested = Path(path).expanduser()
    root = _directory((Path.cwd() if base_dir is None else base_dir) / requested)
    if (root / _DESCRIPTION).exists() or (root / _ACTOR_DESCRIPTION).exists():
        selected = root
        pointer = None
    else:
        latest = _latest_checkpoint(root.resolve())
        if latest is None:
            raise ValueError(f"Run has no latest_checkpoint.json: {root}")
        selected, pointer = latest
    details = read_checkpoint_description(selected)
    if details["kind"] != "learner":
        raise ValueError("Resume needs a full learner checkpoint, not an actor export")
    if pointer is not None and details["checkpoint_id"] != pointer["checkpoint_id"]:
        raise ValueError("Latest checkpoint identity differs from its saved pointer")
    return selected.resolve(), details


def _reward_component_keys(identity: object) -> tuple[str, ...]:
    """Read the optional saved reward component schema without numerical imports.

    Old/array-only identities have neither schema nor keys and return ().
    Named parts require schema 1 and sorted, unique, nonempty string names.
    Reject partial or malformed declarations before building restore templates.
    """
    if identity is None:
        return ()
    if not isinstance(identity, dict):
        raise ValueError("Reward identity must be a mapping")
    identity = cast(dict[str, Any], identity)
    if not {"component_schema", "component_keys"} & identity.keys():
        return ()
    raw_keys: object = identity.get("component_keys")
    if not isinstance(raw_keys, list):
        raise ValueError("Saved reward component keys must be a list")
    keys = cast(list[object], raw_keys)
    if (
        type(identity.get("component_schema")) is not int
        or identity["component_schema"] != 1
        or not keys
        or any(not isinstance(key, str) or not key.strip() for key in keys)
        or keys != sorted(set(cast(list[str], keys)))
    ):
        raise ValueError("Saved reward component schema or keys are invalid")
    return tuple(cast(list[str], keys))


def artifact_name(method: str, kind: str, env_steps: int, identifier: str) -> str:
    """Name one saved boundary by method, purpose, steps and exact identity.

    method is a built-in training method, kind is checkpoint or actor,
    env_steps is its actual nonnegative transition count, and identifier is its
    full lowercase SHA-256. Return a lowercase snake_case basename. Keeping the
    full suffix avoids short-name collisions; publication still verifies bytes
    before reusing an existing immutable directory. This function writes nothing.
    """
    validate_training_method(method)
    if kind not in ("checkpoint", "actor"):
        raise ValueError("Saved artifact kind must be checkpoint or actor")
    if type(env_steps) is not int or env_steps < 0:
        raise ValueError("Saved environment steps must be a nonnegative integer")
    _digest(identifier, "Saved artifact identity")
    return f"{method}_{kind}_step_{env_steps:012d}_{identifier}"


def artifact_directory(run_dir: str | Path, kind: str, identifier: str) -> Path:
    """Find a run's saved checkpoint or actor without equating its name and ID.

    kind is checkpoints or actors. identifier is the full checkpoint identity;
    an actor stores that origin in its metadata. Accept historical hash-only
    basenames and readable basenames ending in the full identity. Verify the
    description and reject links, mismatched identities or multiple candidates.
    If no artifact exists, return the historical path so the normal reader gives
    its missing-file error. No files change and numerical payloads are not read.
    """
    if kind not in ("checkpoints", "actors"):
        raise ValueError("Saved artifact directory must be checkpoints or actors")
    _digest(identifier, "Saved artifact identity")
    directory = Path(run_dir) / kind
    legacy = directory / identifier
    candidates = ([legacy] if legacy.exists() or legacy.is_symlink() else []) + list(
        directory.glob(f"*_{identifier}")
    )
    if len(candidates) > 1:
        raise ValueError("Multiple saved artifacts claim the same checkpoint identity")
    if not candidates:
        return legacy
    path = candidates[0]
    details = read_checkpoint_description(path)
    saved = (
        details["checkpoint_id"]
        if kind == "checkpoints"
        else details["metadata"].get("checkpoint_id")
    )
    if (
        details["kind"] != ("learner" if kind == "checkpoints" else "actor")
        or saved != identifier
    ):
        raise ValueError("Saved artifact name and full identity differ")
    return path


def _artifact_identity(details: dict[str, Any]) -> dict[str, Any]:
    """Build inference/provenance facts from an already checked description.

    This performs no I/O. The caller owns payload verification when required;
    description-only readers must not claim numerical bytes were checked.
    """
    metadata = details["metadata"]
    actor = details["kind"] == "actor"
    identity: dict[str, Any] = {
        "actor_digest": _inference_digest(details),
        "weight_digest": details["actor_digest"],
        "input_scale": _actor_input_scale(details),
        "spawn_frame": _actor_spawn_frame(details),
        "checkpoint_id": details["checkpoint_id"],
        "schemas": details["schemas"],
        "metadata": metadata,
        "run_id": metadata["run_id"],
        "seed": metadata.get("seed") if actor else metadata["config"].get("seed"),
        "env_steps": metadata["env_steps"]
        if actor
        else details["counters"]["env_steps"],
    }
    method = _schema_method(details["schemas"])
    if method in ("qmix", "pqn_vdn"):
        identity["method"] = method
        identity["optimizer_steps"] = (
            metadata["optimizer_steps"] if actor else details["counters"]["updates"]
        )
    return identity


def _check_actor_origin(actor: dict[str, Any], learner: dict[str, Any]) -> None:
    """Bind a standalone actor description to its original learner boundary.

    Both descriptions must already pass their content-ID checks. Compare method,
    inference identity, sharing and original run/seed/step/optimizer facts. This
    small-record check reads no files or arrays and proves no validation score.
    Raise ValueError for any mismatch, including a claimed but incorrect origin.
    """
    actual, expected = _artifact_identity(actor), _artifact_identity(learner)
    keys = (
        "actor_digest",
        "weight_digest",
        "schemas",
        "run_id",
        "seed",
        "env_steps",
        "input_scale",
        "spawn_frame",
        "optimizer_steps",
    )
    if (
        actor["kind"] != "actor"
        or learner["kind"] != "learner"
        or actor["metadata"].get("checkpoint_id") != learner["checkpoint_id"]
        or any(actual.get(key) != expected.get(key) for key in keys)
        or _actor_parameter_sharing(actor) != _actor_parameter_sharing(learner)
        or _actor_normalization(actor) != _actor_normalization(learner)
    ):
        raise ValueError("Saved actor differs from its originating checkpoint")


def _inference_digest(details: dict[str, Any]) -> str:
    """Bind weights, scale and spawn frame; keep old identities where unchanged.

    For MAPPO at scale 1.0 in the world frame the identity is the raw weight digest. A
    non-default scale alone uses the version-1 envelope exactly as before, so
    every existing scaled identity is unchanged. A "left" frame adds the frame
    to a version-2 envelope, so equal weights played in different frames are
    different Systems. Other PPO methods always use the version-1 ppo_inference
    envelope with model, actor_digest, input_scale and spawn_frame. QMIX uses a
    version-1 qmix_inference envelope that also binds the greedy rule (epsilon
    0.0 and the first-legal-maximum tie rule): learner checkpoints supply these
    constants and exports their saved, checked values, so both routes give one
    identity for the same Q-network. PQN-VDN uses a version-1
    pqn_vdn_inference envelope with the same greedy fields; its actor_digest
    covers the parameters and the BatchNorm statistics, so a statistics-only
    change is a different System. A PQN-VDN "layer_norm" model also binds
    "normalization": "layer_norm" in its envelope; "batch_norm" adds no key,
    so every earlier identity is unchanged. A synthetic historical MAPPO
    identity may omit schemas; saved artifacts cannot omit it.
    """
    scale = _actor_input_scale(details)
    frame = _actor_spawn_frame(details)
    method = _schema_method(details.get("schemas", _SCHEMAS))
    sharing = _actor_parameter_sharing(details)
    normalization = (
        {}
        if _actor_normalization(details) == "batch_norm"
        else {"normalization": _actor_normalization(details)}
    )
    if sharing != "all":
        return sha256(
            _json_bytes(
                {
                    "kind": "grouped_actor_inference",
                    "version": 1,
                    "model": _MODELS[method],
                    "actor_digest": details["actor_digest"],
                    "input_scale": scale,
                    "spawn_frame": frame,
                    "parameter_sharing": sharing,
                    **(
                        {}
                        if is_ppo_method(method)
                        else {"epsilon": 0.0, "tie_rule": QMIX_TIE_RULE}
                    ),
                    **normalization,
                }
            )
        ).hexdigest()
    if method == "pqn_vdn":
        exported = details["kind"] == "actor"
        return sha256(
            _json_bytes(
                {
                    "kind": "pqn_vdn_inference",
                    "version": 1,
                    "model": _MODELS[method],
                    "actor_digest": details["actor_digest"],
                    "input_scale": scale,
                    "spawn_frame": frame,
                    "epsilon": details["epsilon"] if exported else 0.0,
                    "tie_rule": details["tie_rule"] if exported else PQN_TIE_RULE,
                    **normalization,
                }
            )
        ).hexdigest()
    if method == "qmix":
        exported = details["kind"] == "actor"
        return sha256(
            _json_bytes(
                {
                    "kind": "qmix_inference",
                    "version": 1,
                    "model": _MODELS[method],
                    "actor_digest": details["actor_digest"],
                    "input_scale": scale,
                    "spawn_frame": frame,
                    "epsilon": details["epsilon"] if exported else 0.0,
                    "tie_rule": details["tie_rule"] if exported else QMIX_TIE_RULE,
                }
            )
        ).hexdigest()
    if method != "mappo":
        return sha256(
            _json_bytes(
                {
                    "kind": "ppo_inference",
                    "version": 1,
                    "model": _MODELS[method],
                    "actor_digest": details["actor_digest"],
                    "input_scale": scale,
                    "spawn_frame": frame,
                }
            )
        ).hexdigest()
    if scale == 1.0 and frame == "world":
        return details["actor_digest"]
    envelope: dict[str, Any] = {
        "kind": "recurrent_mappo_inference",
        "version": 1,
        "actor_digest": details["actor_digest"],
        "input_scale": scale,
    }
    if frame != "world":
        envelope["version"] = 2
        envelope["spawn_frame"] = frame
    return sha256(_json_bytes(envelope)).hexdigest()


def _actor_origin_directory(export: Path, actor: dict[str, Any]) -> Path:
    """Find a checked embedded origin, otherwise the historical sibling origin.

    An embedded origin is a description only, not a recoverable learner. Its
    bytes must appear in the actor inventory. A broken embedded origin never
    falls back to a neighboring checkpoint. No model or payload tree is loaded.
    """
    identifier = _digest(actor["metadata"].get("checkpoint_id"), "Origin checkpoint")
    name = f"origin/{identifier}/{_DESCRIPTION}"
    embedded = any(key.startswith("origin/") for key in actor["files"])
    required = actor["metadata"].get("anonymous_copy") is not None
    if embedded or required:
        if name not in actor["files"]:
            raise ValueError("Embedded actor origin is missing from its inventory")
        path = _relative(export, name)
        raw = path.read_bytes()
        saved = actor["files"][name]
        if saved != {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}:
            raise ValueError("Embedded actor origin bytes differ from its inventory")
        return path.parent
    return artifact_directory(export.resolve().parent.parent, "checkpoints", identifier)
