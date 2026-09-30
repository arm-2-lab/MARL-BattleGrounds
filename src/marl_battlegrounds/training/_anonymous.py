"""Prepare a narrowly checked anonymous actor copy for the checkpoint publisher.

The publisher owns bytes, identities and atomic writes. This helper classifies
retained metadata, removes only named audit labels, and scans retained text.
It never restores arrays, changes exposure facts or approves public release.
Unclassified metadata fails before the publisher creates output directories.
"""

from __future__ import annotations

import copy
import getpass
import json
import re
import socket
from collections.abc import Set
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.evaluation.models import (
    CodeRevisionV1,
    CodeRevisionV2,
    ContentAddressedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.training._checkpoint_metadata import (
    _actor_origin_directory,
    _check_actor_origin,
    _digest,
    _json_bytes,
    _object,
    _parse_json,
    _read_json,
    _schema_method,
    read_checkpoint_description,
    saved_training_config,
)

# This is part of the private checkpoint owner, not a general privacy framework.
# pyright: reportPrivateUsage=false
Record = dict[str, Any]
_ACTOR_KEYS = set(
    [
        "run_id",
        "seed",
        "env_steps",
        "optimizer_steps",
        "checkpoint_id",
        "capture_origin",
        "display_name",
        "source",
        "dependencies",
        "attempt_id",
        "initial_actor",
        "pinned_opponent",
        "opponent_members",
        "partner_members",
        "derived_from",
        "anonymous_copy",
    ]
)
_CALLABLE_KEYS = set(
    [
        "callable_name",
        "code_digest",
        "code_scope",
        "closure_content",
        "defaults_digest",
        "external_state",
    ]
)
_COMPONENT_KEYS = {"name", "checkpoint", "code_digest"}


def _keys(value: object, allowed: Set[str], required: Set[str], path: str) -> Record:
    """Require classified fields and report the first missing or unknown field."""
    if not isinstance(value, dict):
        raise ValueError(f"Anonymous export needs an object at {path}")
    row = cast(Record, value)
    if missing := required - row.keys():
        raise ValueError(f"Anonymous export needs {path}.{sorted(missing)[0]}")
    if extra := row.keys() - allowed:
        raise ValueError(
            f"Anonymous export has unclassified field {path}.{sorted(extra)[0]}"
        )
    return row


def _scalars(row: Record, structured: Set[str], path: str) -> None:
    """Reject hidden records or lists in fields classified as scalar facts."""
    for name, value in row.items():
        if name not in structured and isinstance(value, (dict, list, tuple)):
            raise ValueError(f"Anonymous export needs a scalar at {path}.{name}")


def _numeric_tree(value: object, path: str) -> None:
    """Admit numbers and numeric lists, never hidden records or text labels."""
    if isinstance(value, list):
        for index, item in enumerate(cast(list[object], value)):
            _numeric_tree(item, f"{path}[{index}]")
    elif type(value) not in (int, float):
        raise ValueError(f"Anonymous export needs numeric values at {path}")


def _scan(value: object, path: str, names: frozenset[str] | None = None) -> None:
    """Reject local names, absolute paths, URLs and emails in retained text/keys.

    This bounded check does not prove anonymity against outside knowledge. It
    reports a field path, never silently edits scientific metadata or licenses.
    """
    if names is None:
        names = frozenset(
            name
            for name in (getpass.getuser().lower(), socket.gethostname().lower())
            if len(name) >= 4
        )
    if isinstance(value, dict):
        for key, item in cast(Record, value).items():
            _scan(key, f"{path}.<key>", names)
            _scan(item, f"{path}.{key}", names)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(cast(list[object], value)):
            _scan(item, f"{path}[{index}]", names)
    elif isinstance(value, str):
        words = set(re.findall(r"[a-z0-9_-]+", value.lower()))
        if (
            re.search(r"(?<![\w/])(?:/|~[/\\]|[A-Za-z]:[/\\]|\\\\)", value)
            or re.search(
                r"[a-z][a-z0-9+.-]*://|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value, re.I
            )
            or words & names
        ):
            raise ValueError(f"Anonymous export retains identifying text at {path}")


def _components(value: object, path: str) -> None:
    """Classify the small component records produced by built-in adapters."""
    if not isinstance(value, list):
        raise ValueError(f"Anonymous export needs a component list at {path}")
    for index, item in enumerate(cast(list[object], value)):
        row = _keys(item, _COMPONENT_KEYS, {"name"}, f"{path}[{index}]")
        _scalars(row, set(), f"{path}[{index}]")


def _callable(value: object, path: str) -> None:
    """Keep a saved callable receipt unchanged without importing its code."""
    if value is not None:
        row = _keys(value, _CALLABLE_KEYS, _CALLABLE_KEYS, path)
        _scalars(row, set(), path)


def _registration(value: object, path: str) -> None:
    """Classify saved built-in Policy/System registration fields without defaults."""
    row = _object(value, path)
    common = set(
        [
            "kind",
            "name",
            "checkpoint",
            "execution",
            "components",
            "variables_digest",
            "parameter_evidence",
            "parameter_status",
            "variables_frozen",
            "hooks",
        ]
    )
    if row.get("kind") == "system":
        fields = common | {"adapter_kind", "adapter_policies"}
        _keys(row, fields | {"resource_scope"}, fields, path)
        hooks = {"apply", "init", "reset_memory"}
        for index, policy in enumerate(row["adapter_policies"]):
            label = f"{path}.adapter_policies[{index}]"
            keys = set(
                [
                    "name",
                    "checkpoint",
                    "execution",
                    "variables_frozen",
                    "callable_name",
                    "controller_identity",
                    "variables_digest",
                    "initial_carry_digest",
                    "apply",
                ]
            )
            entry = _keys(policy, keys, keys, label)
            _scalars(entry, {"apply", "controller_identity"}, label)
            _callable(entry["apply"], label + ".apply")
            if entry["controller_identity"] is not None:
                ContentAddressedIdentityV1.model_validate_json(
                    _json_bytes(entry["controller_identity"])
                )
        _callable(row.get("resource_scope"), path + ".resource_scope")
    elif row.get("kind") == "policy":
        fields = common | {
            "callable_name",
            "controller_identity",
            "initial_carry_digest",
        }
        _keys(row, fields, fields, path)
        hooks = {"apply"}
        if row["controller_identity"] is not None:
            ContentAddressedIdentityV1.model_validate_json(
                _json_bytes(row["controller_identity"])
            )
    else:
        raise ValueError(
            f"Anonymous export has unclassified registration kind at {path}.kind"
        )
    _scalars(
        row,
        {
            "components",
            "hooks",
            "adapter_policies",
            "controller_identity",
            "resource_scope",
        },
        path,
    )
    _components(row["components"], path + ".components")
    if row["parameter_evidence"] not in ("registration_snapshot", "unknown"):
        raise ValueError(
            f"Anonymous export has unclassified parameter evidence at {path}"
        )
    if (
        row["parameter_status"] not in ("frozen", "evolving", "unknown")
        or type(row["variables_frozen"]) is not bool
    ):
        raise ValueError(f"Anonymous export has invalid parameter status at {path}")
    for name, hook in _keys(row["hooks"], hooks, hooks, path + ".hooks").items():
        _callable(hook, path + ".hooks." + name)


def _evidence(value: object, path: str) -> None:
    """Retain complete known, none or unknown exposure with its existing strength."""
    fields = {"source", "exposure", "controllers", "familiar_scenarios"}
    row = _keys(value, fields | {"declared_components"}, fields, path)
    _scalars(row, {"controllers", "familiar_scenarios", "declared_components"}, path)
    if row["source"] not in {
        "installed",
        "verified export",
        "verified checkpoint",
        "verified capture",
        "declared export",
        "declared checkpoint",
        "researcher method",
    } or row["exposure"] not in {"none", "known", "unknown"}:
        raise ValueError(f"Anonymous export has invalid exposure at {path}")
    if not isinstance(row["controllers"], list) or not all(
        isinstance(item, str) for item in cast(list[object], row["controllers"])
    ):
        raise ValueError(
            f"Anonymous export needs controller names at {path}.controllers"
        )
    if not isinstance(row["familiar_scenarios"], list) or not all(
        type(item) is int for item in cast(list[object], row["familiar_scenarios"])
    ):
        raise ValueError(
            f"Anonymous export needs scenario IDs at {path}.familiar_scenarios"
        )
    if "declared_components" in row:
        _components(row["declared_components"], path + ".declared_components")


def _inherited(value: object, kind: str, path: str) -> None:
    """Check one initial actor, named member or legacy pin without weakening it."""
    if value is None:
        return
    if kind == "initial":
        fields = {"path", "actor_digest", "checkpoint_id", "evidence"}
        row = _keys(value, fields, fields, path)
        _digest(row["actor_digest"], path + ".actor_digest")
        _digest(row["checkpoint_id"], path + ".checkpoint_id")
    else:
        fields = set(
            [
                "name",
                "reference",
                "execution",
                "registration_id",
                "registration",
                "variables_digest",
                "evidence",
            ]
        )
        fields.add("memory_rule" if kind == "pin" else "system_name")
        row = _keys(value, fields | {"input_identity"}, fields, path)
        _registration(row["registration"], path + ".registration")
        if canonical_digest_sha256(row["registration"]) != row["registration_id"]:
            raise ValueError(f"Anonymous export registration hash differs at {path}")
        identity = row.get("input_identity")
        if identity is not None:
            identity = _object(identity, path + ".input_identity")
            _scalars(
                identity,
                {"controller", "callable", "kwargs", "wrapped_callables"},
                path + ".input_identity",
            )
            which = identity.get("kind")
            if which == "artifact":
                _keys(
                    identity,
                    {"kind", "checkpoint_id"},
                    {"kind", "checkpoint_id"},
                    path + ".input_identity",
                )
                _digest(
                    identity["checkpoint_id"], path + ".input_identity.checkpoint_id"
                )
            elif which == "builtin":
                fields = {"kind", "name", "controller", "callable"}
                _keys(identity, fields, fields, path + ".input_identity")
                _callable(identity["callable"], path + ".input_identity.callable")
                if identity["controller"] is not None:
                    ContentAddressedIdentityV1.model_validate_json(
                        _json_bytes(identity["controller"])
                    )
            elif which == "factory":
                fields = {
                    "kind",
                    "reference",
                    "callable",
                    "defining_file_sha256",
                    "kwargs",
                    "external_state",
                }
                _keys(
                    identity,
                    fields | {"wrapped_callables"},
                    fields,
                    path + ".input_identity",
                )
                if identity["kwargs"] or identity["external_state"] != "unknown":
                    raise ValueError(
                        f"Anonymous export cannot classify "
                        f"{path}.input_identity.kwargs/external_state"
                    )
                _callable(identity["callable"], path + ".input_identity.callable")
                for wrapped in identity.get("wrapped_callables", []):
                    keys = {"callable", "defining_file_sha256"}
                    _keys(
                        wrapped, keys, keys, path + ".input_identity.wrapped_callables"
                    )
                    _scalars(wrapped, {"callable"}, path + ".wrapped_callables")
                    if wrapped["defining_file_sha256"] is not None:
                        _digest(
                            wrapped["defining_file_sha256"],
                            path + ".defining_file_sha256",
                        )
                    _callable(
                        wrapped["callable"],
                        path + ".input_identity.wrapped_callables.callable",
                    )
            else:
                raise ValueError(
                    f"Anonymous export has unclassified input identity at {path}"
                )
    _scalars(row, {"registration", "evidence", "input_identity"}, path)
    _evidence(row["evidence"], path + ".evidence")


def _members(metadata: Record, *, learner: bool, path: str) -> None:
    """Apply one inherited-record check to capture, actor and learner records."""
    _inherited(metadata.get("initial_actor"), "initial", path + ".initial_actor")
    _inherited(metadata.get("pinned_opponent"), "pin", path + ".pinned_opponent")
    for key in ("opponent_members", "partner_members"):
        name = "used_" + key if learner else key
        rows = metadata.get(name, [])
        if not isinstance(rows, list):
            raise ValueError(f"Anonymous export needs a list at {path}.{name}")
        for index, item in enumerate(cast(list[object], rows)):
            _inherited(item, "member", f"{path}.{name}[{index}]")


def _source(metadata: Record, path: str) -> None:
    """Validate original source/dependency facts without substituting today's code."""
    source = metadata.get("source")
    if source is not None:
        source = _object(source, path + ".source")
        owner = CodeRevisionV2 if source.get("schema_version") == 2 else CodeRevisionV1
        owner.model_validate_json(_json_bytes(source))
    dependencies = metadata.get("dependencies")
    if dependencies is not None:
        allowed = set(
            [
                "python",
                "jax",
                "jaxlib",
                "numpy",
                "flax",
                "optax",
                "orbax-checkpoint",
                "flashbax",
            ]
        )
        row = _keys(dependencies, allowed, set(), path + ".dependencies")
        if not all(isinstance(value, str) and value for value in row.values()):
            raise ValueError(
                f"Anonymous export needs version strings at {path}.dependencies"
            )


def _binding(value: object, *, project_paths: bool) -> Record:
    """Validate the existing binding; optionally replace only named audit paths."""
    from marl_battlegrounds.training._content import TrainingContentBinding

    original = TrainingContentBinding.model_validate_json(_json_bytes(value))
    result = _object(value, "content_binding")
    if project_paths:
        for row in [*result["maps"], *result["protected_scenarios"]]:
            for key in ("source", "approved_source"):
                entry = row["info"].get(key)
                if (
                    entry is not None
                    and entry.get("source_path") is not None
                    and (
                        Path(entry["source_path"]).is_absolute()
                        or "://" in entry["source_path"]
                        or entry["source_path"].startswith("~")
                    )
                ):
                    entry["source_path"] = (
                        "content/" + sha256(entry["source_path"].encode()).hexdigest()
                    )
        result["canonical_digest"] = canonical_digest_sha256(
            result, exclude={"canonical_digest"}
        )
        checked = TrainingContentBinding.model_validate_json(_json_bytes(result))
        if checked.scientific_projection() != original.scientific_projection():
            raise ValueError("Anonymous capture changed its scientific content")
    return result


def _array_shape(value: object, name: str) -> None:
    """Require a saved array shape: a list of nonnegative integer dimensions."""
    if not isinstance(value, list) or any(
        type(dimension) is not int or dimension < 0
        for dimension in cast(list[object], value)
    ):
        raise ValueError(f"Anonymous {name} must be a list of nonnegative integers")


def _storage_description(details: Record) -> None:
    """Classify the retained array layouts and inventory without restoring arrays."""
    for name in ("actor_layout", "layout"):
        if name not in details:
            continue
        rows = details[name]
        if not isinstance(rows, list):
            raise ValueError(f"Anonymous {name} must be a list")
        for row in cast(list[object], rows):
            fields = {"dtype", "path", "shape"}
            row = _keys(row, fields, fields, name)
            if not isinstance(row["dtype"], str):
                raise ValueError(f"Anonymous {name}.dtype must be text")
            _array_shape(row["shape"], name + ".shape")
            if not isinstance(row["path"], list):
                raise ValueError(f"Anonymous {name}.path must be a list")
            for key in cast(list[object], row["path"]):
                key = _keys(key, {"field", "key", "index"}, set(), name + ".path")
                if len(key) != 1 or type(next(iter(key.values()))) not in (str, int):
                    raise ValueError(f"Anonymous {name}.path needs simple array keys")
    inventory = _object(details["files"], "files")
    for name, entry in inventory.items():
        entry = _keys(entry, {"bytes", "sha256"}, {"bytes", "sha256"}, "files." + name)
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise ValueError("Anonymous file size must be a nonnegative integer")
        _digest(entry["sha256"], "File digest")


def _orbax_metadata(value: object, name: str) -> None:
    """Admit the pinned Orbax array metadata shapes; refuse custom extensions.

    This is a copy-admission check, not a storage reader. Unknown formats remain
    usable through ordinary exports but need classification before anonymous use.
    Opaque array bytes are never decoded here.
    """
    if name == "actor/_CHECKPOINT_METADATA":
        fields = {
            "item_handlers",
            "metrics",
            "performance_metrics",
            "init_timestamp_nsecs",
            "commit_timestamp_nsecs",
            "custom_metadata",
        }
        row = _keys(value, fields, fields, name)
        for key in ("metrics", "performance_metrics", "custom_metadata"):
            if row[key] not in (None, {}):
                raise ValueError(f"Anonymous export cannot classify {name}.{key}")
        if not isinstance(row["item_handlers"], str):
            raise ValueError(f"Anonymous {name}.item_handlers must be text")
        for key in ("init_timestamp_nsecs", "commit_timestamp_nsecs"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(
                    f"Anonymous {name}.{key} must be a nonnegative integer"
                )
    elif name == "actor/_METADATA":
        fields = {
            "tree_metadata",
            "use_ocdbt",
            "use_zarr3",
            "store_array_data_equal_to_fill_value",
            "custom_metadata",
        }
        row = _keys(value, fields, fields, name)
        if row["custom_metadata"] is not None:
            raise ValueError("Anonymous actor metadata must have no custom_metadata")
        for key in fields - {"tree_metadata", "custom_metadata"}:
            if type(row[key]) is not bool:
                raise ValueError(f"Anonymous {name}.{key} must be Boolean")
        tree = _object(row["tree_metadata"], name + ".tree_metadata")
        for item in tree.values():
            fields = {"key_metadata", "value_metadata"}
            item = _keys(item, fields, fields, name + ".tree_metadata")
            if not isinstance(item["key_metadata"], list):
                raise ValueError("Anonymous key_metadata must be a list")
            for key in cast(list[object], item["key_metadata"]):
                fields = {"key", "key_type"}
                key = _keys(key, fields, fields, name + ".key_metadata")
                if not isinstance(key["key"], str) or (
                    type(key["key_type"]) is not int or key["key_type"] not in (1, 2)
                ):
                    raise ValueError(
                        f"Anonymous {name}.key_metadata needs a text key "
                        "and type 1 or 2"
                    )
            fields = {"value_type", "skip_deserialize", "write_shape"}
            leaf = _keys(
                item["value_metadata"], fields, fields, name + ".value_metadata"
            )
            if (
                leaf["value_type"] != "jax.Array"
                or type(leaf["skip_deserialize"]) is not bool
            ):
                raise ValueError(
                    f"Anonymous {name}.value_metadata needs a JAX array "
                    "and Boolean skip_deserialize"
                )
            _array_shape(leaf["write_shape"], name + ".write_shape")
    elif name.startswith("actor/array_metadatas/"):
        row = _keys(value, {"array_metadatas"}, {"array_metadatas"}, name)
        if not isinstance(row["array_metadatas"], list):
            raise ValueError("Anonymous array_metadatas must be a list")
        for item in cast(list[object], row["array_metadatas"]):
            item = _keys(item, {"array_metadata"}, {"array_metadata"}, name)
            fields = {"param_name", "write_shape", "chunk_shape", "ext_metadata"}
            leaf = _keys(item["array_metadata"], fields, fields, name)
            if leaf["ext_metadata"] is not None or not isinstance(
                leaf["param_name"], str
            ):
                raise ValueError(
                    "Anonymous array metadata needs a name and no extension"
                )
            for key in ("write_shape", "chunk_shape"):
                _array_shape(leaf[key], name + "." + key)
    else:
        import base64

        for encoded_key, encoded_value in _object(value, name).items():
            try:
                key = base64.b64decode(encoded_key, validate=True).decode()
                leaf = _parse_json(encoded_value, name)
            except (ValueError, TypeError, UnicodeError) as error:
                raise ValueError(
                    "Anonymous sharding metadata cannot be decoded"
                ) from error
            _scan(key, name + ".key")
            fields = {"sharding_type", "device_str"}
            leaf = _keys(leaf, fields, fields, name)
            if (
                leaf["sharding_type"] != "SingleDeviceSharding"
                or not isinstance(leaf["device_str"], str)
                or re.fullmatch(r"(?:cpu|cuda|gpu|tpu):[0-9]+", leaf["device_str"])
                is None
            ):
                raise ValueError(
                    "Anonymous exports support checked single-device sharding"
                )
    _scan(value, name)


def _learner(details: Record) -> None:
    """Admit an unchanged description closure or refuse unclassified bookkeeping.

    Nonempty host/recovery/callback records need a later explicit classification;
    they are not dropped. This narrow route can export direct checkpoint API
    saves; ordinary later trainer checkpoints often contain private actor paths.
    """
    from marl_battlegrounds.training.config import config_from_dict

    _storage_description(details)
    if details["recording_token"] is not None:
        raise ValueError(
            "Anonymous origin cannot retain an unclassified recording token"
        )
    fields = {"updates", "env_steps"}
    if _schema_method(details["schemas"]) in ("qmix", "pqn_vdn"):
        fields |= {"completed_blocks", "learning_blocks"}
    counters = _keys(details["counters"], fields, fields, "origin.counters")
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError("Anonymous counters must be nonnegative integers")

    base = set(
        [
            "run_id",
            "attempt_id",
            "parent_checkpoint",
            "config",
            "source",
            "dependencies",
        ]
    )
    optional = set(
        [
            "execution",
            "host_state",
            "selection_rule",
            "initial_actor",
            "callback_inputs",
            "recording",
            "validation_deployment",
            "validation_declaration",
            "validation_partner_references",
            "continuation",
            "study_attempt",
            "log_cursors",
            "used_opponent_members",
            "used_partner_members",
            "opponent_selection",
            "partner_selection",
        ]
    )
    row = _keys(details["metadata"], base | optional, base, "origin.metadata")
    _scalars(row, {"config", "source", "dependencies"} | optional, "origin.metadata")
    config_from_dict(saved_training_config(details))
    _source(row, "origin.metadata")
    _members(row, learner=True, path="origin.metadata")
    for name in optional - {
        "execution",
        "selection_rule",
        "initial_actor",
        "used_opponent_members",
        "used_partner_members",
    }:
        if row.get(name):
            raise ValueError(f"Anonymous export cannot classify origin.metadata.{name}")
    if row.get("execution") is not None:
        execution = _keys(
            row["execution"],
            {
                "schema_version",
                "compiler_options",
                "runtime",
                "jax",
                "declared_xla_flags",
            },
            {
                "schema_version",
                "compiler_options",
                "runtime",
                "jax",
                "declared_xla_flags",
            },
            "origin.metadata.execution",
        )
        _scalars(
            execution,
            {"compiler_options", "runtime", "jax"},
            "origin.metadata.execution",
        )
        if (
            type(execution["schema_version"]) is not int
            or execution["schema_version"] != 1
        ):
            raise ValueError("Anonymous execution needs schema_version=1")
        if not isinstance(execution["declared_xla_flags"], str):
            raise ValueError("Anonymous execution declared_xla_flags must be text")
        for name, fields in {
            "compiler_options": {"xla_gpu_autotune_level"},
            "runtime": {"backend", "device", "runtime_version", "precision"},
            "jax": set(
                [
                    "enable_x64",
                    "disable_jit",
                    "default_matmul_precision",
                    "default_prng_impl",
                    "threefry_partitionable",
                    "threefry_gpu_kernel_lowering",
                    "enable_pgle",
                ]
            ),
        }.items():
            values = _keys(
                execution[name], fields, fields, "origin.metadata.execution." + name
            )
            _scalars(values, set(), "origin.metadata.execution." + name)
    fields = set(
        [
            "content_binding",
            "schedule",
            "score_threshold_curriculum",
            "root_bits",
            "key_schema",
            "reward_settings",
            "shaping",
            "shaping_mode",
            "pinned_opponent_share",
            "collect_training_state",
            "metrics",
            "recording",
            "opponent_rows",
            "history_capture_capacity",
            "reward_identity",
            "opponent_members",
            "learner_slots",
            "partner_members",
            "pinned_opponent",
            "actor_variables_at_step",
        ]
    )
    collection = _keys(
        details["collection"], fields, {"content_binding"}, "origin.collection"
    )
    _scalars(
        collection,
        {
            "content_binding",
            "schedule",
            "root_bits",
            "reward_settings",
            "learner_slots",
            "opponent_members",
            "partner_members",
            "pinned_opponent",
            "actor_variables_at_step",
        },
        "origin.collection",
    )
    for name in ("root_bits", "reward_settings", "learner_slots"):
        if name in collection:
            _numeric_tree(collection[name], "origin.collection." + name)
    for name in (
        "score_threshold_curriculum",
        "shaping",
        "collect_training_state",
        "recording",
    ):
        if name in collection and type(collection[name]) is not bool:
            raise ValueError(f"Anonymous origin.collection.{name} must be Boolean")
    for name in ("key_schema", "opponent_rows", "history_capture_capacity"):
        if name in collection and type(collection[name]) is not int:
            raise ValueError(f"Anonymous origin.collection.{name} must be an integer")
    if type(collection.get("pinned_opponent_share", 0.0)) not in (int, float):
        raise ValueError("Anonymous pinned_opponent_share must be numeric")
    if collection.get("metrics", "none") not in ("none", "priority", "full"):
        raise ValueError("Anonymous origin.collection.metrics is unknown")
    if collection.get("shaping_mode", "potential") not in ("potential", "score_delta"):
        raise ValueError("Anonymous origin.collection.shaping_mode is unknown")
    _binding(collection["content_binding"], project_paths=False)
    _members(collection, learner=False, path="origin.collection")
    if collection.get("reward_identity"):
        raise ValueError(
            "Anonymous export cannot classify origin.collection.reward_identity"
        )
    hook = collection.get("actor_variables_at_step")
    if hook is not None:
        fields = {"kind", "version", "eps_min", "eps_decay", "num_envs"}
        hook = _keys(hook, fields, fields, "origin.collection.actor_variables_at_step")
        _scalars(hook, set(), "origin.collection.actor_variables_at_step")
        if hook["kind"] != "qmix_epsilon" or hook["version"] != 1:
            raise ValueError(
                "Anonymous export cannot classify "
                "origin.collection.actor_variables_at_step"
            )
    # schedule consists of the existing rounding owner's numeric report and one
    # fixed rule string. Unknown nested fields must not carry private metadata.
    schedule = _object(collection.get("schedule", {}), "origin.collection.schedule")
    fields = set(
        [
            "total_env_steps",
            "num_envs",
            "stage_shares",
            "ideal_round_counts",
            "assigned_round_counts",
            "cumulative_round_ends",
            "rounding_rule",
            "stage_maps",
            "team_sizes",
            "roster_class_ids",
            "score_thresholds",
            "history_thresholds",
        ]
    )
    _keys(schedule, fields, set(), "origin.collection.schedule")
    for name, value in schedule.items():
        if name == "rounding_rule":
            if not isinstance(value, str):
                raise ValueError("Anonymous schedule rounding_rule must be text")
        else:
            _numeric_tree(value, "origin.collection.schedule." + name)
    _scan(schedule, "origin.collection.schedule")
    _scan(details, "origin")


def prepare_anonymous_actor(
    source: Path, actor: Record, original: Record
) -> tuple[Record, dict[str, bytes]]:
    """Return a checked actor description and added origin bytes without writing.

    source is an already payload-verified actor or learner. actor is the normal
    export description; original is its input description. Current copies get a
    new identity and a scan record. Historical schema-1 actors may only pass as
    unchanged copies. Required unsafe/unclassified evidence raises ValueError.
    """
    from marl_battlegrounds.training import checkpoints
    from marl_battlegrounds.training._content import (
        TrainingContentBinding,
        _export_origin,
    )

    result = copy.deepcopy(actor)
    _storage_description(result)
    context = checkpoints._check_actor_provenance(
        result["metadata"], _schema_method(result["schemas"])
    )
    _keys(context, _ACTOR_KEYS, {"run_id", "seed", "env_steps"}, "metadata")
    _scalars(
        context,
        {
            "source",
            "dependencies",
            "capture_origin",
            "initial_actor",
            "pinned_opponent",
            "opponent_members",
            "partner_members",
            "anonymous_copy",
        },
        "metadata",
    )
    if "derived_from" in context and "anonymous_copy" not in context:
        raise ValueError("Anonymous export has derived_from without its scan record")
    _source(context, "metadata")
    _members(context, learner=False, path="metadata")
    extra: dict[str, bytes] = {}
    capture = context.get("capture_origin")
    historical = result["schemas"]["actor_input"] == 1
    opaque = (
        re.fullmatch(r"(?:[0-9a-f]{32}|run-[0-9a-f]{64})", context["run_id"])
        is not None
    )
    if historical and original["kind"] != "actor":
        raise ValueError(
            "Historical schema-1 anonymous export only copies existing actors unchanged"
        )
    if (
        historical
        and capture is None
        and not any(name.startswith("origin/") for name in original["files"])
    ):
        raise ValueError(
            "Historical actor needs an existing embedded origin; "
            "copying alone would lose its source evidence"
        )
    if capture is not None:
        binding = TrainingContentBinding.model_validate_json(
            _json_bytes(capture["content_binding"])
        )
        if not historical:
            if not opaque:
                context["run_id"] = (
                    "run-" + sha256(context["run_id"].encode()).hexdigest()
                )
            capture["content_binding"] = _binding(
                capture["content_binding"], project_paths=True
            )
    else:
        if not opaque:
            raise ValueError(
                "Anonymous export cannot replace metadata.run_id without "
                "breaking its original checkpoint link"
            )
        origin = (
            source
            if original["kind"] == "learner"
            else _actor_origin_directory(source, original)
        )
        learner = read_checkpoint_description(origin)
        _check_actor_origin(actor, learner)
        _learner(learner)
        binding = TrainingContentBinding.model_validate_json(
            _json_bytes(learner["collection"]["content_binding"])
        )
        if not historical:
            raw = (origin / "checkpoint_details.json").read_bytes()
            if json.loads(raw) != learner:
                raise ValueError(
                    "Origin description changed during anonymous preparation"
                )
            name = f"origin/{learner['checkpoint_id']}/checkpoint_details.json"
            extra[name] = raw
            result["files"][name] = {
                "bytes": len(raw),
                "sha256": sha256(raw).hexdigest(),
            }
    status, _ = _export_origin(binding, source)
    if not status.startswith("verified"):
        raise ValueError(
            "Anonymous export needs the complete verified source description; "
            "declared history is not upgraded"
        )
    if not historical:
        if "display_name" in context:
            context["display_name"] = (
                "Anonymous " + _schema_method(result["schemas"]) + " actor"
            )
        if (
            context.get("attempt_id") is not None
            and context.get("anonymous_copy") is None
        ):
            context["attempt_id"] = (
                "attempt-" + sha256(context["attempt_id"].encode()).hexdigest()
            )
    scanned: list[str] = []
    for name in result["files"]:
        _scan(name, "files.<name>")
        if not name.startswith("actor/"):
            if (
                capture is not None
                or name != f"origin/{context['checkpoint_id']}/checkpoint_details.json"
            ):
                raise ValueError(f"Anonymous export has unclassified payload {name}")
            continue
        if re.fullmatch(
            r"actor/(?:ocdbt\.process_[0-9]+/)?(?:d/[0-9a-f]{32,64}|manifest\.ocdbt)",
            name,
        ):
            continue  # Opaque Orbax array storage; copy bytes without decoding.
        if not re.fullmatch(
            r"actor/(?:_METADATA|_CHECKPOINT_METADATA|_sharding|array_metadatas/process_[0-9]+)",
            name,
        ):
            raise ValueError(f"Anonymous export has unclassified payload {name}")
        try:
            value = _read_json(source / name)
        except (ValueError, UnicodeError) as error:
            raise ValueError(
                f"Anonymous export cannot scan payload metadata {name}"
            ) from error
        _orbax_metadata(value, name)
        scanned.append(name)
    existing = context.get("anonymous_copy")
    marker = {
        "schema_version": 1,
        "text_files_checked": sorted(scanned),
        "scan": "Known metadata fields and local names, paths, URLs and emails",
    }
    if existing is not None and existing != marker:
        raise ValueError("Anonymous export has an unknown scan record")
    if existing is not None:
        _digest(context.get("derived_from"), "Original anonymous source")
    result["metadata"] = context
    _scan(result, "actor")
    if historical:
        if context != actor["metadata"]:
            raise ValueError(
                "Historical schema-1 anonymous exports must remain unchanged"
            )
        return result, {}
    if existing is None:
        context["derived_from"] = original["checkpoint_id"]
    else:
        _digest(context.get("derived_from"), "Original anonymous source")
    context["anonymous_copy"] = marker
    result["metadata"] = context
    result.pop("checkpoint_id", None)
    result["checkpoint_id"] = sha256(_json_bytes(result)).hexdigest()
    return result, extra
