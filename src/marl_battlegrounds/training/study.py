"""Run declared training cases through the ordinary trainer and launcher.

This module owns finite study declarations, ordered cases and saved reports.
The trainer still owns learning, validation, selection and checkpoint recovery.
Without declared resources, one worker inherits the caller's environment.
Explicit resources admit bounded workers through the same case owner. A
declared GPU missing at start is paused while the other devices keep working.
Import, status, the progress view (study_status with progress=True) and stop
use no numerical backend and query no GPU. Workers preserve Python import paths.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from functools import wraps
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.training import _launch  # pyright: ignore[reportPrivateUsage]
from marl_battlegrounds.training._continuation_schedules import CONTINUATION_CHANGE_KEYS
from marl_battlegrounds.training._run_io import (
    atomic_json,
    process_identity,
    study_attempt_scope,
    unused_run_folder,
    utc_now,
)

type Record = dict[str, Any]
type Declaration = str | Path | Mapping[str, Any]
_DECLARATION = "declaration.json"


def _read(path: Path) -> Record:
    """Read a regular JSON object through the shared launch file owner."""
    return _launch._read(path)


def _optional(path: Path) -> Record:
    """Read an existing object, returning an empty mapping only when absent."""
    return _read(path) if path.exists() else {}


def _digest(value: object) -> str:
    """Hash finite sorted JSON while preserving every list's declared order."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _object(value: object, name: str) -> Record:
    """Copy a finite JSON object without retaining caller-owned containers."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    try:
        return cast(
            Record,
            json.loads(
                json.dumps(dict(cast(Mapping[str, Any], value)), allow_nan=False)
            ),
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must contain only finite JSON values") from error


def _keys(value: Record, allowed: set[str], name: str) -> None:
    """Reject unknown fields before starting any declared work."""
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")


def _duration(value: object, name: str) -> float | None:
    """Check a positive number of wall-clock seconds, or no limit."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be positive seconds or None")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite positive seconds")
    return float(value)


def _path(value: object, base: Path, name: str) -> str:
    """Resolve one declared nonempty path against the declaration's directory."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _reference(value: object, base: Path) -> object:
    """Resolve explicit folders; keep registered names and factory strings.

    A bare module:function is always a factory. Spell a colon-containing folder
    as ./model:version or an absolute path to remove that ambiguity. Other bare
    existing directories retain their historical declaration-relative meaning.
    No factory is imported or invoked by this path-only step.
    """
    if not isinstance(value, str):
        return value
    from marl_battlegrounds._method_loading import _reference_kind

    return (
        _path(value, base, "System reference")
        if _reference_kind(value, base_dir=base) == "artifact"
        else value
    )


def _config_paths(
    value: Record, base: Path, *, runtime: Mapping[str, Any] | None = None
) -> Record:
    """Copy settings and resolve declared input paths against base.

    This shared host step opens no payload or factory. Policy names and factory
    references retain their meaning. Unknown settings remain for the config
    parser to reject. base is the original launch directory or declaration parent.
    runtime aliases keep their exact names even beside a folder with that name.
    """
    config = dict(value)
    for name in ("validation_panel", "random_initialization_result"):
        if config.get(name) is not None:
            config[name] = _path(config[name], base, name)
    for name in ("initial_actor", "pinned_opponent", "slot_diagnostic_actor"):
        if config.get(name) is not None:
            config[name] = _reference(config[name], base)
    if isinstance(config.get("validation_opponents"), (list, tuple)):
        config["validation_opponents"] = [
            _reference(value, base) for value in config["validation_opponents"]
        ]
    for name in ("opponents", "partners", "validation_partners"):
        declared = config.get(name)
        aliases: Mapping[str, Any] = (runtime or {}).get(name) or {}

        def resolve_member(
            item: object, aliases: Mapping[str, Any] = aliases
        ) -> object:
            """Keep bound aliases; resolve only unbound member references."""
            return (
                item
                if isinstance(item, str)
                and (item in ("self", "past") or item in aliases)
                else _reference(item, base)
            )

        if isinstance(declared, dict):
            declared = cast(Record, declared)
            if name == "validation_partners":
                config[name] = {
                    key: _reference(item, base) for key, item in declared.items()
                }
            else:
                resolved: Record = {}
                for key, item in declared.items():
                    normalized = str(resolve_member(key))
                    if normalized in resolved:
                        raise ValueError(
                            f"{name} contains different names for the same resolved "
                            f"member: {normalized}. Use distinct runtime aliases "
                            "to keep separate members."
                        )
                    resolved[normalized] = item
                config[name] = resolved
        elif isinstance(declared, (list, tuple)):
            config[name] = [
                resolve_member(item) for item in cast(Sequence[Any], declared)
            ]
    return config


def _runtime_paths(value: object, base: Path) -> Record:
    """Check reloadable trainer bindings and resolve their input locations.

    Accept the trainer's four population/validation keyword names. Mapping keys
    are aliases and stay unchanged; values and ordered validation opponents are
    nonempty method references. on_update and matchmaking accept a trusted
    factory/kwargs descriptor or None. Syntax is checked without importing or
    calling the factory. No opaque Python object is serialized here.
    """
    runtime = _object(value, "Runtime bindings")
    _keys(
        runtime,
        {
            "opponents",
            "partners",
            "validation_partners",
            "validation_opponents",
            "on_update",
            "matchmaking",
        },
        "runtime",
    )
    for name, bindings in runtime.items():
        if bindings is None:
            continue
        if name in ("on_update", "matchmaking"):
            declaration = _object(bindings, f"runtime.{name}")
            _keys(declaration, {"factory", "kwargs"}, f"runtime.{name}")
            reference = declaration.get("factory")
            if (
                not isinstance(reference, str)
                or reference.count(":") != 1
                or not all(
                    part.isidentifier()
                    for part in reference.replace(":", ".").split(".")
                )
                or "." in reference.split(":")[1]
            ):
                raise ValueError(f"runtime.{name}.factory must use module:function")
            runtime[name] = {
                "factory": reference,
                "kwargs": _object(
                    declaration.get("kwargs", {}), f"runtime.{name}.kwargs"
                ),
            }
        elif name == "validation_opponents":
            if not isinstance(bindings, list):
                raise TypeError(
                    "runtime.validation_opponents must be a list of references"
                )
            references = cast(list[Any], bindings)
            if any(not isinstance(ref, str) or not ref.strip() for ref in references):
                raise TypeError(f"runtime.{name} needs nonempty method references")
            runtime[name] = [_reference(ref, base) for ref in references]
        else:
            if not isinstance(bindings, dict):
                raise TypeError(f"runtime.{name} must map nonempty names to references")
            named = cast(Record, bindings)
            if any(not alias.strip() for alias in named):
                raise TypeError(f"runtime.{name} must map nonempty names to references")
            if any(
                not isinstance(ref, str) or not ref.strip() for ref in named.values()
            ):
                raise TypeError(f"runtime.{name} needs nonempty method references")
            runtime[name] = {
                alias: _reference(ref, base) for alias, ref in named.items()
            }
    return runtime


def _callback_inputs(runtime: Mapping[str, Any]) -> Record:
    """Bind callback factory code and explicit kwargs without running its body.

    runtime is already checked by _runtime_paths. This imports trusted factory
    modules only during execution setup, after resource choices. Preview must
    not call it. Unknown remote or closure state remains explicitly unknown.
    """
    from marl_battlegrounds._method_loading import _reference_identity

    return {
        name: {
            **_reference_identity(declared["factory"]),
            "kwargs": _object(declared["kwargs"], f"runtime.{name}.kwargs"),
        }
        for name in ("on_update", "matchmaking")
        if (declared := runtime.get(name)) is not None
    }


def _runtime_callbacks(
    runtime: Record, stack: ExitStack, *, identities: Record | None = None
) -> Record:
    """Materialize trusted callback declarations once in the caller's scope.

    Copy runtime, preserving all System references and aliases. Each callback
    factory receives its finite explicit kwargs once. A context manager is
    entered with stack and must yield a callable; bare callables own no cleanup.
    The returned wrapper carries the checked factory receipt for trainer recovery.
    Errors propagate; the caller's ExitStack closes earlier entered resources.
    Opaque callback state is not saved or automatically restored. identities,
    when supplied by a worker, are its already checked _callback_inputs receipts;
    this avoids hashing the same defining files twice during worker setup.
    """
    from marl_battlegrounds._method_loading import installed_callable

    result = dict(runtime)
    identities = _callback_inputs(runtime) if identities is None else identities
    for name in ("on_update", "matchmaking"):
        declaration = runtime.get(name)
        if declaration is None:
            continue
        callback = installed_callable(declaration["factory"])(
            **_object(declaration["kwargs"], f"runtime.{name}.kwargs")
        )
        if isinstance(callback, AbstractContextManager):
            callback = stack.enter_context(
                cast(AbstractContextManager[object], callback)
            )
        if not callable(callback):
            raise TypeError(f"runtime.{name} factory must return or yield a callable")
        result[name] = _bound_callback(callback, identities[name])
    return result


def _bound_callback(
    callback: Callable[..., Any], identity: Record
) -> Callable[..., Any]:
    """Forward callback calls unchanged and attach its finite setup receipt.

    The trainer reads this private attribute before recovery writes. It is an
    identity record, not callback state or a promise about remote services.
    """

    @wraps(callback)
    def invoke(*args: object, **kwargs: object) -> object:
        """Return the researcher's callback result and preserve its exceptions."""
        return callback(*args, **kwargs)

    setattr(invoke, "_marl_callback_identity", identity)  # noqa: B010
    return invoke


def _config(
    value: object, base: Path, seed: int, *, runtime: Mapping[str, Any] | None = None
) -> Record:
    """Resolve reference paths and let the trainer check all method settings."""

    from marl_battlegrounds.training.config import config_from_dict, config_to_dict

    config = _object(value, "Case config")
    if "seed" in config:
        raise ValueError("Use the case seeds list; config.seed is not allowed")
    if "method" not in config or "total_env_steps" not in config:
        raise ValueError("Each config must declare method and total_env_steps")
    config = _config_paths(config, base, runtime=runtime)
    result = config_to_dict(config_from_dict({**config, "seed": seed}))
    result.pop("seed")
    return result


def _checkpoint_identity(path: str) -> Record:
    """Verify all parent payloads and pin its complete learner description."""
    import jax

    with jax.default_device(jax.devices("cpu")[0]):
        from marl_battlegrounds.training.checkpoints import read_checkpoint_details

        details = read_checkpoint_details(path)
        if details["kind"] != "learner":
            raise ValueError("A study continuation requires a full learner checkpoint")
        return {
            "checkpoint_id": details["checkpoint_id"],
            "description_sha256": _digest(details),
            "method": details["metadata"]["config"]["method"],
        }


def _identities(methods: Sequence[str]) -> Record:
    """Read source and method dependencies through the checkpoint owner."""
    from marl_battlegrounds.training.checkpoints import runtime_identity

    return {method: runtime_identity(method) for method in sorted(set(methods))}


def _file_inputs(case: Record) -> Record:
    """Bind every declared file and method before setup or a worker attempt.

    Panel/result files keep exact byte hashes. The existing panel owner resolves
    member references, including portable and historical paths. Method inputs
    use checked artifact descriptions or installed/factory code evidence; factory
    bodies and numerical model loads do not run. The trainer later checks full
    payloads and frozen registrations. Aliases and zero-share members remain in
    the declaration. Each distinct input is inspected once per call. Unknown
    external service state stays unknown rather than being called reproducible.
    """
    from marl_battlegrounds._method_loading import _reference_identity

    result: Record = {}

    def method(value: object) -> None:
        """Bind one reference once; direct live objects stay outside declarations."""
        if value is None:
            return
        if not isinstance(value, str) or not value.strip():
            raise TypeError("Study method inputs must be nonempty references")
        if value in ("self", "past") or value in result:
            return
        result[value] = _reference_identity(value)

    def file(
        value: object, *, panel: bool = False, bindings: Sequence[Any] | None = None
    ) -> None:
        """Bind a file and, for panels, all effective ordered member inputs."""
        if value is None:
            return
        path = Path(str(value))
        if panel and path.is_dir():
            path /= "panel.json"
        key = str(path)
        if key not in result:
            result[key] = {
                "kind": "file",
                "sha256": _launch._hash(path),
                "bytes": path.stat().st_size,
            }
        if panel:
            from marl_battlegrounds.training.validation import (
                _json,
                _panel_input_references,
            )

            for reference in _panel_input_references(path, _json(path), bindings):
                if reference is None:
                    raise ValueError(
                        "Live-only panel members need explicit runtime bindings"
                    )
                method(reference)

    config = case.get("config", {})
    runtime = case.get("runtime", {})
    validation_opponents = runtime.get("validation_opponents")
    if validation_opponents is None:
        validation_opponents = config.get("validation_opponents")
    file(config.get("validation_panel"), panel=True, bindings=validation_opponents)
    file(config.get("random_initialization_result"))
    for name in ("initial_actor", "pinned_opponent", "slot_diagnostic_actor"):
        method(config.get(name))
    for name in ("opponents", "partners"):
        bindings: Record = runtime.get(name) or {}
        for member in config.get(name) or ():
            method(bindings.get(member, member))
        for reference in bindings.values():
            method(reference)
    partners: Record = {
        **(config.get("validation_partners") or {}),
        **(runtime.get("validation_partners") or {}),
    }
    for reference in partners.values():
        method(reference)
    for reference in validation_opponents or ():
        method(reference)
    changes = case.get("changes", {})
    for name in ("opponents", "partners"):
        bindings = runtime.get(name) or {}
        for member in changes.get(name) or ():
            method(bindings.get(member, member))
    for reward in (config.get("reward"), changes.get("reward")):
        if reward is not None:
            method(reward)
    callbacks = _callback_inputs(runtime)
    if callbacks:
        result["callback_inputs"] = callbacks
    validation = changes.get("validation", {})
    file(validation.get("panel"), panel=True, bindings=validation.get("bindings"))
    for reference in validation.get("bindings") or ():
        method(reference)
    return result


def _effective_job_identity(
    case: Record, *, seed: int, artifact_cache: dict[str, Record] | None = None
) -> str:
    """Compare effective training inputs, preserving roles, order and shares.

    case is a normalized fresh case. Its external_inputs, when available, replace
    artifact/file locations with their existing owner identities. Preview lacks
    those identities; artifact_cache optionally reads existing actor descriptions
    once per input, without payloads. Other references stay literal in preview;
    it cannot import arbitrary user modules. Callback kwargs remain exact. This is
    only Hydra's duplicate-job check, not recipe grouping or a new scientific identity.
    """
    config = _object(case["config"], "Case config")
    config.pop("seed", None)
    runtime = _object(case.get("runtime", {}), "Case runtime")
    inputs = case.get("external_inputs", {})

    def reference(value: object) -> object:
        """Replace one declared input reference with its already checked identity."""
        if not isinstance(value, str):
            return value
        if value in inputs:
            return inputs[value]
        if (
            artifact_cache is not None
            and Path(value).is_absolute()
            and Path(value).is_dir()
        ):
            if value not in artifact_cache:
                from marl_battlegrounds.training._checkpoint_metadata import (
                    read_checkpoint_description,
                )

                details = read_checkpoint_description(value)
                artifact_cache[value] = {
                    "kind": "artifact",
                    "checkpoint_id": details["checkpoint_id"],
                }
            return artifact_cache[value]
        return value

    for name in (
        "initial_actor",
        "pinned_opponent",
        "slot_diagnostic_actor",
        "reward",
    ):
        if name in config:
            config[name] = reference(config[name])
    for name in ("validation_panel", "random_initialization_result"):
        value = config.get(name)
        if value is not None:
            path = Path(value)
            if name == "validation_panel" and path.is_dir():
                path /= "panel.json"
            config[name] = inputs.get(str(path), str(path))
    for name in ("opponents", "partners"):
        bindings: Record = runtime.pop(name, {}) or {}
        if config.get(name) is not None:
            selection = config[name]
            config[name] = (
                {
                    "shares": [
                        [reference(bindings.get(member, member)), share]
                        for member, share in cast(
                            Mapping[str, float], selection
                        ).items()
                    ]
                }
                if isinstance(selection, Mapping)
                else {
                    "sequence": [
                        reference(bindings.get(member, member)) for member in selection
                    ]
                }
            )
        if bindings:
            runtime[name] = [
                [
                    member if member not in inputs else reference(member),
                    reference(value),
                ]
                for member, value in bindings.items()
            ]
    for owner in (config, runtime):
        if owner.get("validation_opponents") is not None:
            owner["validation_opponents"] = [
                reference(value) for value in owner["validation_opponents"]
            ]
        if owner.get("validation_partners") is not None:
            owner["validation_partners"] = [
                [slot, reference(value)]
                for slot, value in owner["validation_partners"].items()
            ]
    for name, identity in inputs.get("callback_inputs", {}).items():
        runtime[name] = identity
    return _digest({"config": config, "runtime": runtime, "seed": seed})


def _normalize(config: Declaration, *, check_inputs: bool = True) -> Record:
    """Resolve ordered cases and defaults, then optionally freeze external facts.

    check_inputs=False previews settings without loading methods or checking
    payloads and source identity. Both routes use the same field validation,
    path rules and ordered case expansion. Execution checks all inputs.
    """
    if isinstance(config, (str, Path)):
        source = Path(config).expanduser().resolve()
        declared, base = _launch.read_declaration(source), source.parent
    else:
        declared, base = _object(config, "Study declaration"), Path.cwd()
    _keys(
        declared,
        {
            "schema_version",
            "name",
            "duration_seconds",
            "failure_policy",
            "cases",
            "finalist_of",
            "resources",
            "launch",
        },
        "study",
    )
    if (
        type(declared.get("schema_version", 1)) is not int
        or declared.get("schema_version", 1) != 1
    ):
        raise ValueError("Unsupported study schema_version")
    if not isinstance(declared.get("name"), str) or not declared["name"].strip():
        raise ValueError("Study name must be nonempty")
    failure = declared.get("failure_policy", "stop")
    if failure not in ("stop", "continue"):
        raise ValueError("failure_policy must be stop or continue")
    cases = declared.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Study cases must be a nonempty list")
    normalized: Record = {
        "schema_version": 1,
        "name": declared["name"],
        "duration_seconds": _duration(
            declared.get("duration_seconds"), "Study duration_seconds"
        ),
        "failure_policy": failure,
        "cases": [],
    }
    if declared.get("resources") is not None:
        normalized["resources"] = _launch.resource_settings(
            declared["resources"], base_dir=base
        )
    linked_root: Path | None = None
    if declared.get("finalist_of") is not None:
        linked_root = Path(_path(declared["finalist_of"], base, "finalist_of"))
        discovery = _read(linked_root / _DECLARATION)
        normalized["finalist_of"] = {
            "study_dir": str(linked_root),
            "declaration_sha256": _digest(discovery),
        }
    if declared.get("launch") is not None:
        launch = _object(declared["launch"], "Launch evidence")
        _keys(
            launch,
            {"kind", "directory", "overrides", "sweeper_params", "jobs"},
            "launch evidence",
        )
        if launch.get("kind") != "hydra" or not isinstance(launch.get("jobs"), list):
            raise ValueError("Study launch evidence must name a finite Hydra batch")
        launch["directory"] = _path(launch.get("directory"), base, "Launch directory")
        launch["sweeper_params"] = _object(
            launch.get("sweeper_params", {}), "Sweeper params"
        )
        rows = [_object(row, "Hydra job mapping") for row in launch["jobs"]]
        for row in [launch, *rows]:
            values = row.get("overrides")
            if not isinstance(values, list) or any(
                not isinstance(value, str) for value in cast(list[object], values)
            ):
                raise ValueError("Hydra overrides must be lists of strings")
        for row in rows:
            _keys(row, {"id", "overrides", "run_dir"}, "Hydra job mapping")
            row["run_dir"] = _path(row.get("run_dir"), base, "Hydra case run_dir")
        launch["jobs"] = rows
        normalized["launch"] = launch
    ids: set[str] = set()
    for raw in cast(list[Any], cases):
        case = _object(raw, "Study case")
        identifier = case.get("id")
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identifier) is None
        ):
            raise ValueError(
                "Case id must use 1 to 80 letters, digits, underscores or hyphens"
            )
        if identifier in ids:
            raise ValueError(f"Duplicate study case id: {identifier}")
        ids.add(identifier)
        common: Record = {
            "id": identifier,
            "duration_seconds": _duration(
                case.get("duration_seconds"), f"Case {identifier} duration_seconds"
            ),
        }
        if "runtime" in case:
            common["runtime"] = _runtime_paths(case["runtime"], base)
        if "config" in case:
            _keys(
                case,
                {"id", "config", "seeds", "duration_seconds", "runtime"},
                "training case",
            )
            seeds = case.get("seeds")
            if not isinstance(seeds, list) or not seeds:
                raise ValueError("Case seeds must be distinct integers in [0, 2**32)")
            seeds = cast(list[Any], seeds)
            if any(
                type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds
            ) or len(set(seeds)) != len(seeds):
                raise ValueError("Case seeds must be distinct integers in [0, 2**32)")
            checked = (
                _config(case["config"], base, seeds[0], runtime=common["runtime"])
                if common.get("runtime")
                else _config(case["config"], base, seeds[0])
            )
            normalized["cases"].append({**common, "config": checked, "seeds": seeds})
        else:
            _keys(
                case,
                {
                    "id",
                    "checkpoint",
                    "additional_env_steps",
                    "changes",
                    "runtime",
                    "duration_seconds",
                },
                "continuation case",
            )
            checkpoint = _path(case.get("checkpoint"), base, "checkpoint")
            steps = case.get("additional_env_steps")
            if type(steps) is not int or steps <= 0:
                raise ValueError("additional_env_steps must be a positive integer")
            changes = _object(case.get("changes", {}), "Continuation changes")
            _keys(
                changes,
                set(CONTINUATION_CHANGE_KEYS),
                "continuation changes",
            )
            changes = _config_paths(changes, base, runtime=common.get("runtime"))
            if isinstance(changes.get("validation"), dict):
                validation = changes["validation"]
                if validation.get("panel") is not None:
                    validation["panel"] = _path(
                        validation["panel"], base, "validation panel"
                    )
                if isinstance(validation.get("bindings"), list):
                    validation["bindings"] = [
                        _reference(value, base) for value in validation["bindings"]
                    ]
            if linked_root is not None:
                parents = {
                    str((linked_root / "cases" / row["id"] / "run").resolve())
                    for row in _jobs(_read(linked_root / _DECLARATION))
                }
                if str(Path(checkpoint).parent.parent) not in parents:
                    raise ValueError(
                        "Finalist continuation parent must belong to the linked study"
                    )
            normalized["cases"].append(
                {
                    **common,
                    "checkpoint": checkpoint,
                    **(
                        {"parent_identity": _checkpoint_identity(checkpoint)}
                        if check_inputs
                        else {}
                    ),
                    "additional_env_steps": steps,
                    "changes": changes,
                }
            )
    if check_inputs:
        for case in normalized["cases"]:
            case["external_inputs"] = _file_inputs(case)
    jobs = _jobs(normalized)
    if len({row["id"] for row in jobs}) != len(jobs):
        raise ValueError("Expanded study case IDs conflict")
    if "launch" in normalized:
        seen_jobs: set[str] = set()
        for job in jobs:
            if job["kind"] != "train":
                raise ValueError(
                    "Hydra sweeps start fresh training cases; resume the saved study"
                )
            key = _effective_job_identity(job, seed=job["config"]["seed"])
            if key in seen_jobs:
                raise ValueError(f"Duplicate effective Hydra job: {job['id']}")
            seen_jobs.add(key)
        if [row.get("id") for row in normalized["launch"]["jobs"]] != [
            job["id"] for job in jobs
        ]:
            raise ValueError("Hydra job mapping differs from declared study order")
    if check_inputs:
        normalized["runtime_identities"] = _identities(
            [
                row["config"]["method"]
                if row["kind"] == "train"
                else row["parent_identity"]["method"]
                for row in jobs
            ]
        )
    return normalized


def _jobs(declared: Record) -> list[Record]:
    """Expand each declared seed once, keeping case order and seed order."""
    rows: list[Record] = []
    for case in declared["cases"]:
        if "config" in case:
            for seed in case["seeds"]:
                rows.append(
                    {
                        "id": f"{case['id']}--seed-{seed}",
                        "kind": "train",
                        "config": {**case["config"], "seed": seed},
                        "duration_seconds": case["duration_seconds"],
                        "external_inputs": case.get("external_inputs", {}),
                        **({"runtime": case["runtime"]} if "runtime" in case else {}),
                    }
                )
        else:
            rows.append({**case, "kind": "extend"})
    return rows


def _freeze(config: Declaration) -> Record:
    """Validate one declaration on CPU without starting the caller's backend.

    One short subprocess owns trainer imports and any parent-artifact reads.
    It inherits Python paths and changes device selection only in that child.
    Temporary request/results are removed on success or failure; no study
    folder exists yet. Invalid declarations raise ValueError with the child error.
    """
    value = (
        str(Path(config).expanduser().resolve())
        if isinstance(config, (str, Path))
        else _object(config, "Study declaration")
    )
    with tempfile.TemporaryDirectory(prefix="marl-study-declaration-") as temporary:
        directory = Path(temporary)
        atomic_json(directory / "request.json", {"config": value})
        completed = subprocess.run(
            _command("_normalize", str(directory)),
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        outcome = _optional(directory / "outcome.json")
        if completed.returncode or "declaration" not in outcome:
            raise ValueError(
                outcome.get("error")
                or completed.stderr.strip()
                or "Study declaration validation failed"
            )
        return cast(Record, outcome["declaration"])


def _prepare(
    config: Declaration | None,
    output_dir: str | Path | None,
    resume_from: str | Path | None,
) -> tuple[Path, Record, bool]:
    """Check destinations and saved declarations before writes or case setup.

    New and resumed roots must pass the shared source-path guard. Reject unsafe
    locations before freezing source identity, acquiring locks or making folders.
    """
    from marl_battlegrounds.evaluation.revision import (
        _check_output_path,  # pyright: ignore[reportPrivateUsage]
    )

    if resume_from is not None:
        if output_dir is not None:
            raise ValueError("Use output_dir for a new study or resume_from, not both")
        root = Path(resume_from).expanduser().resolve()
        _check_output_path(root)
        declared = _read(root / _DECLARATION)
        if config is not None and _freeze(config) != declared:
            raise ValueError(
                "Supplied study declaration differs from the saved declaration"
            )
        with _launch.launch_lock(root):
            started = any(
                (root / name).exists()
                for name in (
                    "launch_time.json",
                    "study.json",
                    "process.json",
                    "report.json",
                )
            )
            for name in ("cases", "logs"):
                folder = root / name
                if folder.is_symlink():
                    raise ValueError(
                        f"Study {name} directory must not be a symbolic link"
                    )
                if started and not folder.is_dir():
                    raise ValueError(f"Started study is missing its {name} directory")
                folder.mkdir(exist_ok=True)
        return root, declared, True
    if config is None or output_dir is None:
        raise ValueError("A new study requires config and an exact output_dir")
    root = Path(output_dir).expanduser().resolve()
    _check_output_path(root)
    declared = _freeze(config)
    if "launch" in declared and any(
        Path(row["run_dir"]) != root / "cases" / row["id"] / "run"
        for row in declared["launch"]["jobs"]
    ):
        raise ValueError("Hydra job paths differ from the exact study destination")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Study output_dir must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    with _launch.launch_lock(root):
        if any(path.name != ".launch.lock" for path in root.iterdir()):
            raise ValueError("Study output_dir must be new or empty")
        atomic_json(root / _DECLARATION, declared)
        (root / "cases").mkdir()
        (root / "logs").mkdir()
    return root, declared, False


def _case_directory(root: Path, identifier: str) -> Path:
    """Keep every case below its declared root without following nested links."""
    directory = root / "cases" / identifier
    if directory.resolve() != directory:
        raise ValueError("Study case directory follows a link or escapes its owner")
    return directory


def _process_paths(root: Path) -> list[Path]:
    """List only the controller and declared case process records."""
    declared = _read(root / _DECLARATION)
    return [
        root / "process.json",
        root / "analysis_job/process.json",
        *(_case_directory(root, row["id"]) / "process.json" for row in _jobs(declared)),
    ]


def _clock(root: Path, declared: Record, *, create: bool) -> Record:
    """Read or create the one fixed study deadline through the launch owner."""
    duration = declared["duration_seconds"]
    if not (root / "launch_time.json").exists() and (
        any(
            (root / name).exists()
            for name in ("study.json", "process.json", "report.json")
        )
        or any((root / "cases").iterdir())
    ):
        raise ValueError("Started study is missing its original launch clock")
    return _launch.launch_clock(
        root / "launch_time.json",
        durations={} if duration is None else {"deadline_at": duration},
        binding={"declaration_sha256": _digest(declared)},
        create=create,
    )


def _command(*arguments: str) -> list[str]:
    """Use this interpreter and installed package without changing import paths."""
    return [sys.executable, "-m", "marl_battlegrounds.training.study", *arguments]


def _latest(run: Path) -> Path | None:
    """Resolve only the saved latest pointer; ordinary train verifies its bytes."""
    from marl_battlegrounds.training._checkpoint_metadata import _latest_checkpoint

    result = _latest_checkpoint(run)
    return None if result is None else result[0]


def _report(root: Path, declared: Record, state: Record) -> Record:
    """Save all declared outcomes, including missing, failed and unfinished work."""
    rows = [
        state["cases"].get(
            job["id"], {"id": job["id"], "status": "pending", "attempts": []}
        )
        for job in _jobs(declared)
    ]
    result = {
        "schema_version": 1,
        "study_dir": str(root),
        "declaration_sha256": _digest(declared),
        "status": state["status"],
        "cases": rows,
        "paused_devices": state.get("paused_devices", {}),
        "analysis": state.get("analysis", {}),
        "updated_at": utc_now(),
        "qualification": "Execution records only; no claim of learned competence",
    }
    atomic_json(root / "report.json", result)
    return result


def _finished_supervisor(directory: Path) -> None:
    """Release this controller's finished supervisor role, retaining child facts.

    supervise_command runs inside this still-live controller. Once it returns,
    that role is finished even though Python remains alive. Clear only our exact
    identity from an exited record. Saved child/group cleanup remains intact so
    surviving workers still prevent another launch.
    """
    record = _optional(directory / "process.json")
    if record.get("state") == "exited" and record.get("process") == process_identity():
        record["supervisor_identity"] = record["process"]
        record["process"] = None
        atomic_json(directory / "process.json", record)


def _stopped(directory: Path) -> bool:
    """Read saved stop intent, distinguishing requests from failed worker exits.

    The supervisor records user/controller stop requests separately from child
    exit codes. A child killed by the operating system is a failed attempt and
    follows failure_policy; it does not stop unrelated cases under continue.
    Deadline cleanup remains a time limit rather than a user stop.
    """
    process = _optional(directory / "process.json")
    return process.get("stop_reason") != "deadline" and bool(
        process.get("stop_signal") or process.get("stop_reason") == "signal"
    )


def _attempt_resources(directory: Path, attempt: Record) -> None:
    """Attach only resource facts belonging to this saved attempt request.

    Call before replacing a recovery request or finishing its worker. Missing or
    stale facts are not invented; the next attempt keeps a separate record.
    """
    request = _optional(directory / "request.json")
    resources = _optional(directory / "resources.json")
    if request.get("attempt") == attempt["number"] and resources.get(
        "request_sha256"
    ) == _digest(request):
        attempt["resource_facts"] = resources


def _case_attempt(
    root: Path,
    declared: Record,
    state: Record,
    clock: Record,
    job: Record,
    *,
    resume: bool,
) -> tuple[Path, Record, float] | None:
    """Prepare one ordered case attempt using its original clocks and recovery.

    The controller holds the study lock. Return the case folder, mutable attempt
    record and absolute cleanup-start time, or None for completed/refused cases.
    Update study status when admission must stop. Write the exact request before
    any process starts; preserve failed/unfinished attempts and saved deadlines.
    """
    row = state["cases"].setdefault(
        job["id"],
        {
            "id": job["id"],
            "kind": job["kind"],
            "status": "pending",
            "attempts": [],
        },
    )
    if row["status"] == "complete":
        return None
    reserve = _launch.cleanup_reserve_seconds()
    if clock.get("deadline_at", math.inf) - reserve <= time.time():
        state["status"] = "time_limit"
        return None
    directory = _case_directory(root, job["id"])
    directory.mkdir(exist_ok=True)
    run = directory / "run"
    if run.resolve() != run:
        raise ValueError("Study run directory follows a link or escapes its case")
    case_clock = _launch.launch_clock(
        directory / "launch_time.json",
        durations={}
        if job["duration_seconds"] is None
        else {"deadline_at": job["duration_seconds"]},
        binding={
            "declaration_sha256": _digest(declared),
            "case_id": job["id"],
        },
        create=not (
            row["attempts"] or run.exists() or (directory / "process.json").exists()
        ),
    )
    deadline = min(
        clock.get("deadline_at", math.inf),
        case_clock.get("deadline_at", math.inf),
    )
    stop_at = deadline - reserve
    if stop_at <= time.time():
        row["status"] = "time_limit"
        if declared["failure_policy"] == "stop":
            state["status"] = "incomplete"
            return None
        return None
    previous = bool(row["attempts"]) or run.exists()
    try:
        checkpoint = _latest(run) if previous else None
        restart_empty = resume and checkpoint is None and unused_run_folder(run)
    except (ValueError, OSError) as error:
        row.update(status="failed", error=str(error))
        if declared["failure_policy"] == "stop":
            state["status"] = "incomplete"
            return None
        return None
    if previous and (not resume or (checkpoint is None and not restart_empty)):
        row.update(
            status="failed",
            error=(
                f"No complete checkpoint is available in {run}; "
                "the case is not restarted. Preserve any partial work. "
                "Restore a complete checkpoint to resume, or declare "
                "a new study in another output folder."
            ),
        )
        if declared["failure_policy"] == "stop":
            state["status"] = "incomplete"
            return None
        return None
    if row["attempts"] and "finished_at" not in row["attempts"][-1]:
        _attempt_resources(directory, row["attempts"][-1])
        row["attempts"][-1].update(
            interrupted=True,
            recovered_at=utc_now(),
            error="Previous controller ended before recording an outcome",
        )
    attempt: Record = {
        "number": len(row["attempts"]) + 1,
        "study_attempt_id": uuid.uuid4().hex,
        "started_at": utc_now(),
        "resume_from": None if checkpoint is None else str(checkpoint),
    }
    row["attempts"].append(attempt)
    row.pop("error", None)
    row.update(status="running", run_dir=str(run))
    atomic_json(
        directory / "request.json",
        {
            "job": job,
            "resume_from": attempt["resume_from"],
            "attempt": attempt["number"],
            "study_attempt_id": attempt["study_attempt_id"],
        },
    )
    atomic_json(root / "study.json", state)
    return directory, attempt, stop_at


def _finish_case(
    root: Path,
    declared: Record,
    state: Record,
    job: Record,
    directory: Path,
    attempt: Record,
    code: int,
) -> bool:
    """Record a finished attempt and say whether further admission must stop.

    Completion needs this request's worker outcome, complete run status and a
    successful supervisor exit. Preserve all attempts and cleanup failures.
    A user signal or incomplete worker cleanup stops the study regardless of
    failure_policy. This controller is the only writer of study.json/report.json.
    """
    row = state["cases"][job["id"]]
    run = directory / "run"
    _finished_supervisor(directory)
    attempt.update(exit_code=code, finished_at=utc_now())
    outcome = _optional(directory / "outcome.json")
    if outcome.get("request_sha256") != _digest(_read(directory / "request.json")):
        outcome = {
            "status": "failed",
            "error": "No outcome for this exact attempt",
        }
    saved_status = _optional(run / "status.json")
    binding: Record = saved_status.get("study_attempt") or {}
    exact_binding = (
        attempt.get("study_attempt_id") is not None
        and binding.get("study_attempt_id") == attempt["study_attempt_id"]
        and binding.get("request_sha256") == attempt.get("request_sha256")
        and binding.get("number") == attempt["number"]
        and binding.get("case_id") == job["id"]
    )
    if exact_binding:
        attempt["run_id"] = saved_status.get("run_id")
        attempt["run_attempt_id"] = saved_status.get("attempt_id")
        attempt["run_status"] = saved_status
    elif attempt.get("study_attempt_id") is not None:
        # A failure before runner admission must not borrow the old attempt's
        # completed status, counters or costs. Its raw files remain unchanged.
        saved_status = {}
    row["run_status"] = saved_status
    _attempt_resources(directory, attempt)
    row["status"] = (
        "complete"
        if code == 0
        and saved_status.get("status") == "complete"
        and outcome.get("status") == "complete"
        else "time_limit"
        if code == 124
        else "failed"
    )
    attempt["status"] = row["status"]
    if outcome.get("error"):
        attempt.setdefault("error", outcome["error"])
    atomic_json(root / "study.json", state)
    _report(root, declared, state)
    if _launch.live_process_records([directory / "process.json"]):
        row["error"] = "Worker cleanup is incomplete; no further case can start"
        state["status"] = "stopped" if _stopped(directory) else "incomplete"
        return True
    if _stopped(directory):
        state["status"] = "stopped"
        return True
    if row["status"] != "complete" and declared["failure_policy"] == "stop":
        state["status"] = "incomplete"
        return True
    return False


def _reconcile_completed_attempts(root: Path, declared: Record, state: Record) -> None:
    """Finish durable successful outcomes missed by an interrupted controller.

    The study lock and live-owner check already hold. Reuse the same completion
    writer only when the prior request, zero supervisor exit, worker outcome and
    complete run status agree. No new attempt, seed, deadline or worker is made.
    Missing/failed outcomes continue through ordinary explicit recovery.
    """
    for job in _jobs(declared):
        row = state["cases"].get(job["id"], {})
        if row.get("status") == "complete" or not row.get("attempts"):
            continue
        attempt = row["attempts"][-1]
        directory = _case_directory(root, job["id"])
        request = _optional(directory / "request.json")
        outcome = _optional(directory / "outcome.json")
        process = _optional(directory / "process.json")
        if (
            request.get("job") == job
            and request.get("attempt") == attempt["number"]
            and process.get("state") == "exited"
            and process.get("exit_code") == 0
            and not _stopped(directory)
            and outcome.get("status") == "complete"
            and outcome.get("request_sha256") == _digest(request)
            and _optional(directory / "run/status.json").get("status") == "complete"
        ):
            attempt["recovered_at"] = utc_now()
            _finish_case(root, declared, state, job, directory, attempt, 0)


def _resource_slots(declared: Record) -> list[Record | None]:
    """Expand checked devices in declared order into fixed worker slots.

    Each record retains its device settings and cache path. Repeated slots on
    one device share no learner state. No resources means one inherited slot.
    """
    resources = declared.get("resources")
    if resources is None:
        return [None]
    return [
        {"device": entry, "compilation_cache": resources.get("compilation_cache")}
        for entry in resources["devices"]
        for _ in range(entry["max_workers"])
    ]


def _slot_gpu(resources: Record | None) -> str | None:
    """Return the GPU UUID of one expanded worker slot, or None for CPU/inherited."""
    if resources is None:
        return None
    device = cast(str, resources["device"]["device"])
    return None if device == "cpu" else device


def _pause_failed_gpu(
    state: Record, resources: Record | None, job: Record, attempt: Record
) -> None:
    """Stop admitting cases to a GPU whose case failed before any new checkpoint.

    Parameters
    ----------
    state : dict
        The controller's live study.json state. A pause is written to
        ``state["paused_devices"][uuid]`` with the case ID, attempt number,
        error and time; the attempt also gets ``device_paused`` True.
    resources : dict or None
        The finished case's expanded worker slot. CPU and inherited slots are
        never paused.
    job : dict
        The finished case's declared job.
    attempt : dict
        The finished attempt record after ``_finish_case``. Its
        ``checkpoint_before`` (the run's latest checkpoint at admission) and
        exactly bound ``run_status`` show whether this attempt saved a new
        checkpoint.

    Notes
    -----
    A failed GPU case that saved no checkpoint of its own most often means the
    device itself cannot start or keep work: a lost external card fails every
    later case within seconds and would otherwise take the whole queue. The
    pause keeps the remaining cases for healthy GPUs. Completed, stopped,
    timed-out and interrupted attempts never pause. A case that fails after
    saving a checkpoint does not pause its GPU; if the device is lost, the
    next case fails early and pauses it then. The pause lasts for this
    controller only: start or resume clears it (and pauses only GPUs missing
    at that start), and the failed case is retried by the ordinary resume
    rules. The record's reason is "failed before its first checkpoint".
    Host-only; no process or device is touched.
    """
    device = _slot_gpu(resources)
    if (
        device is None
        or device in state.get("paused_devices", {})
        or state["cases"][job["id"]]["status"] != "failed"
        or attempt.get("stop_requested")
        or attempt.get("interrupted")
    ):
        return
    status = cast(Record, attempt.get("run_status") or {})
    saved = status.get("latest_checkpoint")
    if saved is not None and saved != attempt.get("checkpoint_before"):
        return
    attempt["device_paused"] = True
    state.setdefault("paused_devices", {})[device] = {
        "reason": "failed before its first checkpoint",
        "case_id": job["id"],
        "attempt": attempt["number"],
        "error": attempt.get("error"),
        "paused_at": utc_now(),
    }


def _missing_gpus(declared: Record) -> list[str]:
    """Return declared GPUs that are missing now, refusing only if none is left.

    Parameters
    ----------
    declared : dict
        The saved study declaration. Without resources, or with CPU entries
        only, no device query runs and the result is empty.

    Returns
    -------
    list of str
        Declared GPU UUIDs that nvidia-smi does not list, in declared order.
        The caller pauses them (reason "missing at start") and keeps working
        on the remaining devices.

    Raises
    ------
    ValueError
        Every declared device is a GPU and every one is missing, so no case
        could start. A forbidden compilation cache path also raises here.
    OSError, subprocess.SubprocessError
        nvidia-smi is missing, fails or times out (see
        _launch.missing_resource_devices).

    Notes
    -----
    Host-only: reads the GPU list, starts no backend and changes no file.
    """
    resources = declared.get("resources")
    missing = _launch.missing_resource_devices(resources)
    if (
        resources is not None
        and missing
        and all(row["device"] in missing for row in resources["devices"])
    ):
        raise ValueError(
            f"Declared GPU devices are unavailable: {sorted(missing)}; "
            "no declared device is left to run cases"
        )
    return missing


def _cases_complete(declared: Record, state: Record) -> bool:
    """Check whether every declared case has a durable successful completion."""
    return all(
        state.get("cases", {}).get(job["id"], {}).get("status") == "complete"
        for job in _jobs(declared)
    )


def _run_cases(
    root: Path,
    declared: Record,
    state: Record,
    clock: Record,
    *,
    resume: bool,
    missing: Sequence[str] = (),
) -> None:
    """Admit ordered cases through existing supervisors and finish every attempt.

    One inherited slot uses the historical inline supervisor. Explicit slots use
    one detached supervisor process per active case, with the same preparation
    and completion owners. The controller polls small process records, never
    model data. A failure under stop policy stops admission and requests bounded
    shutdown of active siblings. Signals do the same. Never retry automatically.
    A GPU whose case fails before saving a checkpoint is paused for the rest of
    this controller (see _pause_failed_gpu); the other slots keep admitting
    cases, and when every slot is paused admission stops with status
    "incomplete". Each start or resume clears earlier pauses, then pauses
    every GPU UUID in missing (from _missing_gpus) with reason "missing at
    start", so its slots get no case while the other devices keep working.
    """
    slots = _resource_slots(declared)
    state["paused_devices"] = {
        device: {
            "reason": "missing at start",
            "case_id": None,
            "attempt": None,
            "error": "nvidia-smi does not list this declared GPU",
            "paused_at": utc_now(),
        }
        for device in missing
    }
    jobs = iter(_jobs(declared))
    active: dict[int, tuple[Record, Path, Record]] = {}
    exhausted = False
    stop = False
    signal_received: int | None = None
    previous: dict[int, Any] = {}
    stop_until: float | None = None

    def request_stop(number: int, _frame: object) -> None:
        """Defer stopping owned supervisors to the main controller loop."""
        nonlocal signal_received
        signal_received = number

    if declared.get("resources") is not None:
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[number] = signal.signal(number, request_stop)
    try:
        while active or (not exhausted and not stop):
            if signal_received is not None:
                state["status"] = "stopped"
                stop = True
            if (
                clock.get("deadline_at", math.inf) - _launch.cleanup_reserve_seconds()
                <= time.time()
            ):
                if state["status"] != "stopped":
                    state["status"] = "time_limit"
                stop = True
            for slot, (job, directory, attempt) in list(active.items()):
                process = _optional(directory / "process.json")
                if _launch.live_process_records([directory / "process.json"]):
                    continue
                code = process.get("exit_code", 1)
                if type(code) is not int:
                    code = 1
                if attempt.get("stop_requested"):
                    attempt["interrupted"] = True
                old_status = state["status"]
                stop = (
                    _finish_case(root, declared, state, job, directory, attempt, code)
                    or stop
                )
                _pause_failed_gpu(state, slots[slot], job, attempt)
                atomic_json(root / "study.json", state)
                if (
                    attempt.get("stop_requested")
                    and state["cases"][job["id"]]["status"] != "complete"
                ):
                    state["cases"][job["id"]]["status"] = (
                        "time_limit" if old_status == "time_limit" else "interrupted"
                    )
                    state["status"] = old_status
                del active[slot]
            if stop:
                if stop_until is None:
                    stop_until = time.monotonic() + _launch.cleanup_reserve_seconds()
                elif active and time.monotonic() >= stop_until:
                    raise RuntimeError(
                        "Worker cleanup did not finish within its bound; "
                        "saved live owners block restart"
                    )
                for _job, directory, attempt in active.values():
                    if not attempt.get("stop_requested"):
                        attempt["stop_requested"] = utc_now()
                        _launch.request_stop([directory / "process.json"])
                atomic_json(root / "study.json", state)
            else:
                for slot, resources in enumerate(slots):
                    if (
                        slot in active
                        or exhausted
                        or _slot_gpu(resources) in state["paused_devices"]
                    ):
                        continue
                    try:
                        job = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    prepared = _case_attempt(
                        root, declared, state, clock, job, resume=resume
                    )
                    if prepared is None:
                        stop = state["status"] != "running"
                        if stop:
                            break
                        continue
                    directory, attempt, stop_at = prepared
                    if resources is not None:
                        attempt["checkpoint_before"] = _optional(
                            directory / "run/status.json"
                        ).get("latest_checkpoint")
                    try:
                        if resources is None:
                            attempt["request_sha256"] = _digest(
                                _read(directory / "request.json")
                            )
                            atomic_json(root / "study.json", state)
                            code = _launch.supervise_command(
                                directory,
                                _command("_worker", str(root), job["id"]),
                                deadline_at=None if math.isinf(stop_at) else stop_at,
                            )
                        else:
                            request = _read(directory / "request.json")
                            request.update(
                                resources=resources,
                                resource_slot=slot,
                                stop_at=None if math.isinf(stop_at) else stop_at,
                            )
                            atomic_json(directory / "request.json", request)
                            attempt["request_sha256"] = _digest(request)
                            attempt["resources"] = resources
                            attempt["resource_slot"] = slot
                            atomic_json(root / "study.json", state)
                            with _launch.launch_lock(directory):
                                _launch.spawn_detached(
                                    directory,
                                    _command("_supervise", str(root), job["id"]),
                                    log_path=directory / "process.log",
                                    mode="study-case",
                                    env=_launch.resource_environment(
                                        resources["device"],
                                        compilation_cache=resources[
                                            "compilation_cache"
                                        ],
                                    ),
                                )
                            active[slot] = (job, directory, attempt)
                            continue
                    except Exception as error:
                        code = 1
                        attempt["error"] = f"{type(error).__name__}: {error}"
                    stop = _finish_case(
                        root, declared, state, job, directory, attempt, code
                    )
                    _pause_failed_gpu(state, resources, job, attempt)
                    if stop:
                        break
                if (
                    not active
                    and not exhausted
                    and not stop
                    and all(
                        _slot_gpu(resources) in state["paused_devices"]
                        for resources in slots
                    )
                ):
                    # Every GPU is paused: leave the rest pending for a resume.
                    state["status"] = "incomplete"
                    stop = True
                    atomic_json(root / "study.json", state)
            if active:
                time.sleep(0.05)
        if state["status"] == "running":
            state["status"] = (
                "complete" if _cases_complete(declared, state) else "incomplete"
            )
    finally:
        if active:
            paths = [directory / "process.json" for _, directory, _ in active.values()]
            _launch.request_stop(paths)
            until = (
                stop_until
                if stop_until is not None
                else time.monotonic() + _launch.cleanup_reserve_seconds()
            )
            while _launch.live_process_records(paths) and time.monotonic() < until:
                time.sleep(0.05)
            for job, directory, attempt in active.values():
                attempt["interrupted"] = True
                process = _optional(directory / "process.json")
                code = process.get("exit_code", 1)
                _finish_case(
                    root,
                    declared,
                    state,
                    job,
                    directory,
                    attempt,
                    code if type(code) is int else 1,
                )
        for number, handler in previous.items():
            signal.signal(number, handler)


def _execute(root: Path, *, resume: bool, handoff: bool = False) -> Record:
    """Run unfinished cases once while holding the study's launch lock."""
    with _launch.launch_lock(root, blocking=handoff):
        declared = _read(root / _DECLARATION)
        if "finalist_of" in declared:
            link = declared["finalist_of"]
            if (
                _digest(_read(Path(link["study_dir"]) / _DECLARATION))
                != link["declaration_sha256"]
            ):
                raise ValueError(
                    "Linked discovery declaration differs from its frozen identity"
                )
        live = _launch.live_process_records(_process_paths(root))
        own = process_identity()
        if any(
            path != root / "process.json" or _read(path).get("process") != own
            for path in live
        ):
            raise RuntimeError("This study still has an active owned process")
        clock = _clock(
            root, declared, create=not resume or not (root / "study.json").exists()
        )
        state = _optional(root / "study.json") or {"schema_version": 1, "cases": {}}
        state.update(status="running", **clock)
        process: Record = {
            "schema_version": 1,
            "state": "running",
            "process": own,
            "started_at": utc_now(),
        }
        atomic_json(root / "process.json", process)
        atomic_json(root / "study.json", state)
        try:
            if resume:
                _reconcile_completed_attempts(root, declared, state)
            if _cases_complete(declared, state):
                state["status"] = "complete"
            else:
                missing = _missing_gpus(declared)
                _run_cases(root, declared, state, clock, resume=resume, missing=missing)
            available = [
                root / "cases" / job["id"] / "run"
                for job in _jobs(declared)
                if (root / "cases" / job["id"] / "run/run_details.json").exists()
            ]
            if available and state["status"] != "stopped":
                analysis_job = root / "analysis_job"
                analysis_job.mkdir(exist_ok=True)
                code = 1
                try:
                    code = _launch.supervise_command(
                        analysis_job,
                        _command("_analyze", str(root)),
                        env={**os.environ, "JAX_PLATFORMS": "cpu"},
                    )
                    outcome = _optional(analysis_job / "outcome.json")
                    state["analysis"] = (
                        outcome
                        if code == 0 and outcome.get("status") == "complete"
                        else {
                            "status": "failed",
                            "exit_code": code,
                            "error": outcome.get(
                                "error", "Analysis did not finish this attempt"
                            ),
                        }
                    )
                except Exception as error:
                    state["analysis"] = {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                finally:
                    _finished_supervisor(analysis_job)
                if _stopped(analysis_job):
                    state["status"] = "stopped"
                    state["analysis"]["status"] = "stopped"
            atomic_json(root / "study.json", state)
            return _report(root, declared, state)
        except Exception as error:
            state.update(status="incomplete", error=f"{type(error).__name__}: {error}")
            atomic_json(root / "study.json", state)
            _report(root, declared, state)
            raise
        finally:
            process.update(
                state="exited",
                finished_at=utc_now(),
                controller_identity=own,
                process=None,
            )
            atomic_json(root / "process.json", process)


def preview_study(
    config: Declaration | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> Record:
    """Return ordered study cases and destinations without starting work.

    config is the same JSON/YAML file or mapping accepted by run_study. Fresh
    previews require config and output_dir. Resume accepts only resume_from and
    reads its saved declaration; it does not promise that recovery will succeed.
    Paths and settings use the study owner's normal rules. Return preview=True,
    the normalized declaration, ordered cases with run_dir, resource settings,
    and the checks still required at execution. Create no files, load no model,
    call no factory and start no numerical backend. Missing or malformed settings
    raise ValueError or TypeError. YAML needs the optional hydra extra.
    """
    if resume_from is not None:
        if config is not None or output_dir is not None:
            raise ValueError("Resume preview inherits settings; omit config/output_dir")
        root = Path(resume_from).expanduser().resolve()
        declared = _read(root / _DECLARATION)
    else:
        if config is None or output_dir is None:
            raise ValueError("A new study preview needs config and output_dir")
        root = Path(output_dir).expanduser().resolve()
        declared = _normalize(config, check_inputs=False)
    jobs = _jobs(declared)
    return {
        "preview": True,
        "output_dir": str(root),
        "declaration": declared,
        "cases": [
            {**job, "run_dir": str(root / "cases" / job["id"] / "run")} for job in jobs
        ],
        "pending_checks": [
            "Input payloads, method/source identities and available devices",
            "Output admission, original deadlines and recovery state",
        ],
    }


def run_study(
    config: Declaration | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> Record:
    """Run a declared study in order and return its saved JSON report.

    Parameters
    ----------
    config : path, mapping or None, default=None
        A schema-1 JSON/YAML declaration or mapping with name and cases.
        YAML needs the optional hydra extra. Each fresh case has id,
        config and distinct seeds; config names method and total_env_steps.
        A continuation instead has checkpoint, additional_env_steps and optional
        changes. Defaults are frozen before work. failure_policy defaults to
        "stop"; "continue" leaves failed cases visible and moves to the next.
        Under either policy, a GPU whose case fails before saving a checkpoint
        gets no new case for the rest of this command (recorded in the report's
        paused_devices); other GPUs continue, and a resume clears the pause.
        A declared GPU that nvidia-smi does not list when the command starts
        is paused the same way, with reason "missing at start", and the other
        devices run the cases; each start or resume checks again. The command
        refuses only when every declared device is a missing GPU.
        Optional duration_seconds on the study and cases limits wall-clock
        work from their first launches, including downtime. finalist_of links
        a separate declaration to an existing discovery study. File-relative
        references resolve beside the declaration. A mapping uses the current
        directory. Source and method dependency identities are frozen automatically;
        workers refuse a changed install or file input. Panel/result bytes and
        actor/checkpoint descriptions are pinned automatically. Factory references
        do not freeze remote service state or external provider behavior.
        Each case may supply runtime with named opponents/partners, validation
        bindings, and on_update/matchmaking factory+kwargs declarations. Factories
        run once per worker attempt; context managers yield a callable and close
        on success or failure. Code and explicit kwargs are checked before work;
        hidden callback state is not restored. None is accepted only on resume.
    output_dir : path or None, default=None
        Exact new or empty study folder. Required for a new study and mutually
        exclusive with resume_from. No generated parent directory is added.
        In an editable Git checkout, use an ignored folder or one outside the
        checkout so result files do not change the saved source identity.
    resume_from : path or None, default=None
        Existing study folder. Reuses its declaration and original deadlines.
        A supplied config must resolve to the same declaration. Only explicit
        resume may recover an unfinished case from its latest saved full
        checkpoint. If no checkpoint exists and the case's run folder is missing,
        empty or holds only the trainer's ``.training.lock``, explicit resume may
        start that same case again. Any other run without a checkpoint is
        refused. Attempts, seeds and deadlines are kept;
        a restart never adds time or removes partial work.

    Returns
    -------
    dict
        The finite JSON report also saved as report.json, with every declared
        case and all attempts. Successful cases survive another case's failure
        or an analysis error. Ordinary training owns each run's saved selection;
        this driver makes no selection across runs and no learning claim.

    Raises
    ------
    ValueError, TypeError
        Declaration, output, saved clock or checkpoint settings are invalid,
        or no declared device is present (every declared GPU is missing).
    RuntimeError
        A live study or worker already owns this output.
    OSError
        Study files or subprocesses cannot be read or written.

    Notes
    -----
    Uses supervised subprocesses in the caller's environment. Without resources,
    one worker runs at a time. resources.devices entries name cpu or a full GPU
    UUID, max_workers (default one), and optional per-worker memory_fraction.
    Multiple workers on one GPU require fractions with total <=1. The allocator
    fraction is not a total-VRAM limit. Optional resources.compilation_cache is
    a directory. Admission follows case and device order; failures never retry
    automatically. Status and import need no numerical backend. Config validation and
    analysis use CPU subprocesses without starting the caller's numerical
    backend. Numerical workers stop early enough to reserve the launcher's
    bounded cleanup time inside each saved deadline. A case with no remaining
    execution time starts no worker. Report rebuilding may finish later.
    Completed files remain available.
    """
    root, _, resume = _prepare(config, output_dir, resume_from)
    return _execute(root, resume=resume)


def start_study(
    config: Declaration | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> Record:
    """Launch run_study detached and return its saved controller identity.

    config, output_dir and resume_from have the same meaning and checks as
    run_study. The first launch fixes deadlines before the child starts; resume
    keeps them. stdin is closed and output appends to logs/study.log. The child
    inherits Python paths, device visibility and runtime settings. A pipe gate
    prevents work before its process identity is saved. Duplicate live owners
    raise RuntimeError. When unfinished cases remain and every declared device
    is a missing GPU, ValueError is raised before any process starts; if only
    some GPUs are missing, the child pauses them and runs on the others.
    Declaration, file and process errors propagate.
    """
    root, declared, resume = _prepare(config, output_dir, resume_from)
    with _launch.launch_lock(root):
        if _launch.live_process_records(_process_paths(root)):
            raise RuntimeError("This study still has an active owned process")
        state = _optional(root / "study.json") or {"schema_version": 1, "cases": {}}
        if resume:
            _reconcile_completed_attempts(root, declared, state)
        if not _cases_complete(declared, state):
            _missing_gpus(declared)
        _clock(root, declared, create=not resume or not (root / "study.json").exists())
        command = _command("_control", str(root), *(["--resume"] if resume else []))
        return _launch.spawn_detached(
            root,
            command,
            log_path=root / "logs/study.log",
            mode="resume" if resume else "start",
        )


_PROGRESS_LEGEND = (
    "Table 1 comes from the routine validation checks saved in each run's",
    "validation_results.json, with our learned team as Team A, plus status.json:",
    "  Seed: the case's training seed. State: the case state; a running case",
    "    also shows the run's phase.",
    "  Done: training steps finished out of the run's total.",
    "  Steps/s: recent training speed (transitions per second). Spent: run time",
    "    so far, without gaps between attempts. Left: remaining steps divided by",
    "    Steps/s (running cases only); it leaves out later validation and saving.",
    "  GPU: the last six characters of the case's GPU UUID (resources.json);",
    "    cpu for a CPU worker.",
    "  Check: training step of the latest finished routine check.",
    "  vs NAME: wins-draws-losses against that validation opponent, all maps,",
    "    and the average game score (win 1, draw 0.5, loss 0).",
    "  Points: our average points and theirs per game (a Red Zone kill is",
    "    worth 2). Margin: ours minus theirs, equal weight for every map and",
    "    opponent; the main measure.",
    "  K/D: our kills and our deaths per game.",
    "  RZ K: our Red Zone kills per game (our points minus our kills).",
    "  Home D: our deaths inside our own Red Zone per game (their points minus",
    "    their kills) and their share of all our deaths. A high share is the",
    "    turtling sign.",
    "  Best: the routine check with the highest point margin so far, and that",
    "    margin.",
    "Table 2 comes from the run's own training games (all training opponents),",
    "Team A, from collected_reward_components and opponent_results in",
    "training_updates.jsonl. A window is 25 saved update rows (25 collection",
    "rounds, about 25 learner updates). Per game unless stated:",
    "  Games: training games that ended in the window. Len: ticks per game.",
    "  Pts: our points and theirs per game.",
    "  K/D: kills are not saved for training games, so K is always '-'; D is",
    "    our deaths per game. Kill credits per kill cannot be worked out either,",
    "    because several teammates can share one kill.",
    "  Home D: our deaths inside our own Red Zone, and their share of our deaths.",
    "  Kill Cr: kill credits (one for each credited teammate per enemy death).",
    "  RZ Cr: the extra credits for enemies killed in their own Red Zone.",
    "  Dmg: capped damage our agents dealt (HP). Heal: the Priest's effective",
    "    healing (HP). Range: ticks x living Priest/ally pairs within heal range.",
    "  Own zone: share of our agent ticks spent inside our own Red Zone.",
    "  Shaping: combat/support reward each learner agent got per game.",
    "  Counts are exact for the built-in combat_support_reward and",
    "  combat_support_team_reward; another reward shows '-'.",
    "Table 3 sums the whole run so far: for each training opponent group",
    "(current weights, past copies, the pinned opponent, each named model Mn and",
    "the scripted bots), our wins-draws-losses in finished training games and,",
    "in brackets, the group's share of our training ticks in those games.",
    "'-' means the saved files do not hold that value yet (or at all).",
)


def _text_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a plain left-aligned text table with a dashed rule under the headers.

    Parameters
    ----------
    headers : sequence of str
        Column titles.
    rows : sequence of sequences of str
        Cells in header order. Every row must have one cell per header.

    Returns
    -------
    str
        Lines joined by newlines; each column is as wide as its widest cell.
    """
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *rows, strict=True)
    ]
    lines = [
        "  ".join(
            str(value).ljust(width) for value, width in zip(row, widths, strict=True)
        )
        for row in (headers, ["-" * width for width in widths], *rows)
    ]
    return "\n".join(line.rstrip() for line in lines)


def _duration_text(seconds: object) -> str:
    """Format seconds as "45s", "12m" or "3h 05m"; "-" when not a number."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return "-"
    if not math.isfinite(seconds) or seconds < 0:
        return "-"
    minutes = int(seconds // 60)
    if minutes == 0:
        return f"{int(seconds)}s"
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h {minutes % 60:02d}m"


def _tail_lines(path: Path, count: int) -> list[str]:
    """Return the last nonblank lines of a text log, reading at most 16 KiB.

    Parameters
    ----------
    path : Path
        A log file such as cases/<id>/process.log. A missing file gives [].
    count : int
        How many lines to keep, at most.

    Returns
    -------
    list of str
        Lines without newline characters, oldest first. A first line cut in
        the middle by the 16 KiB read is dropped.
    """
    size = 16_384
    try:
        with path.open("rb") as stream:
            end = stream.seek(0, 2)
            stream.seek(max(0, end - size))
            lines = stream.read().decode(errors="replace").splitlines()
    except FileNotFoundError:
        return []
    if end > size:
        lines = lines[1:]
    return [line for line in lines if line.strip()][-count:]


def _live_processes(root: Path) -> list[Record]:
    """Describe every live process this study owns, with the command to stop it.

    Parameters
    ----------
    root : Path
        The study folder.

    Returns
    -------
    list of dict
        One record per live process record (controller, case or analysis):
        ``owner`` (plain words), ``record`` (process.json path),
        ``supervisor_pid`` and ``trainer_pid`` (int when that exact process
        is still alive, else None), ``worker_pids`` (live members of the
        trainer's private process group), ``leftover`` (True when workers
        remain with no live supervisor or trainer) and ``stop_command`` (the
        exact shell command a person can run).

    Notes
    -----
    Uses the launcher's liveness checks (_launch.live_process_records and
    _alive), which read files and /proc only. Nothing is signaled here, and a
    leaderless worker group is never stopped automatically: its PIDs could
    have been reused, so the person checks them with ps before running kill.
    """
    result: list[Record] = []
    for path in _launch.live_process_records(_process_paths(root)):
        record = _read(path)
        identities: dict[str, int | None] = {}
        for name in ("process", "trainer"):
            identity = record.get(name)
            identities[name] = (
                int(cast(Record, identity)["pid"])
                if isinstance(identity, dict)
                and _launch._alive({name: cast(Record, identity)})
                else None
            )
        group = record.get("trainer_group")
        workers = (
            sorted(_launch._group_members(int(cast(Record, group)["pid"])))
            if isinstance(group, dict)
            and _launch._alive({"trainer_group": cast(Record, group)})
            else []
        )
        owner = (
            "study controller"
            if path == root / "process.json"
            else "analysis"
            if path.parent.name == "analysis_job"
            else f"case {path.parent.name}"
        )
        pids = sorted(
            {pid for pid in (identities["trainer"], *workers) if pid is not None}
        )
        listed = " ".join(map(str, pids))
        result.append(
            {
                "owner": owner,
                "record": str(path),
                "supervisor_pid": identities["process"],
                "trainer_pid": identities["trainer"],
                "worker_pids": workers,
                "leftover": identities["process"] is None
                and identities["trainer"] is None
                and bool(workers),
                "stop_command": (
                    "python -m marl_battlegrounds study stop " + shlex.quote(str(root))
                )
                if identities["process"] is not None
                else f"ps -o pid,args -p {','.join(map(str, pids))} && "
                f"kill -TERM {listed}",
            }
        )
    return result


def _seed_text(job: Record) -> str:
    """Return a case's training seed, or its continuation branch seed, or "-"."""
    seed = job.get("config", {}).get("seed", job.get("changes", {}).get("seed"))
    return "-" if seed is None else str(seed)


def _case_gpu(directory: Path, row: Record) -> str:
    """Return the last six characters of a case's GPU UUID, "cpu" or "-".

    Reads the worker's resources.json first, then the study's latest attempt
    record. Inherited (undeclared) resources and unstarted cases give "-".
    """
    requested = _optional(directory / "resources.json").get("requested")
    if not isinstance(requested, dict):
        attempts = cast(list[Record], row.get("attempts") or [{}])
        requested = cast(object, attempts[-1].get("resources"))
    device = (
        cast(Record, requested).get("device", {}).get("device")
        if isinstance(requested, dict)
        else None
    )
    if not isinstance(device, str):
        return "-"
    return device if device == "cpu" else device[-6:]


def _per_game(value: object, games: int, digits: int, *, sign: bool = False) -> str:
    """Format a window total per game; "-" without games or a known total."""
    if not games or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{value / games:{'+' if sign else ''}.{digits}f}"


def _window_cells(window: Record) -> list[str]:
    """Turn one training window record into the Table 2 cells after Window.

    Parameters
    ----------
    window : dict
        One entry of a run's progress ``windows`` (analysis._update_progress).

    Returns
    -------
    list of str
        Games, Len, Pts, K/D, Home D, Kill Cr, RZ Cr, Dmg, Heal, Range,
        Own zone and Shaping, per game where the legend says so; "-" where
        the window has no games or the value is unknown.
    """
    games = int(window.get("games") or 0)
    counts = cast(Record, window.get("counts") or {})
    deaths = counts.get("own_team_deaths")
    home = counts.get("own_red_zone_deaths")
    zone = counts.get("own_zone_ticks")
    agents = int(window.get("agent_transitions") or 0)
    return [
        str(games),
        _per_game(window.get("length_sum"), games, 0),
        f"{_per_game(window.get('points_for'), games, 1)}-"
        f"{_per_game(window.get('points_against'), games, 1)}"
        if games
        else "-",
        f"-/{_per_game(deaths, games, 1)}" if games and deaths is not None else "-",
        f"{_per_game(home, games, 2)} ({100 * home / deaths:.0f}%)"
        if games and home is not None and deaths
        else _per_game(home, games, 2),
        _per_game(counts.get("enemy_death_credit"), games, 1),
        _per_game(counts.get("enemy_red_zone_death_credit"), games, 2),
        _per_game(counts.get("enemy_damage_hp"), games, 0),
        _per_game(counts.get("priest_effective_healing_hp"), games, 0),
        _per_game(counts.get("priest_heal_range_pairs"), games, 0),
        f"{100 * zone / agents:.1f}%" if zone is not None and agents else "-",
        _per_game(window.get("shaping"), games, 2, sign=True),
    ]


def _check_cells(check: Record | None, opponents: Sequence[str]) -> list[str]:
    """Turn one routine check into Table 1 cells from Check to Home D.

    Parameters
    ----------
    check : dict or None
        One entry of a run's progress ``checks`` (analysis._progress_checks),
        or None before the first finished check.
    opponents : sequence of str
        Validation opponent names in column order.

    Returns
    -------
    list of str
        Check, one "W-D-L (score)" cell per opponent, Points, Margin, K/D,
        RZ K and Home D; "-" for values the check cannot give.
    """
    if check is None:
        return ["-"] * (len(opponents) + 6)
    by_name = {entry["name"]: entry for entry in check["opponents"]}
    cells = [f"{int(check['env_steps'] or 0) / 1e6:.1f}M"]
    for name in opponents:
        entry = by_name.get(name)
        cells.append(
            "-"
            if not entry or not entry["games"]
            else f"{entry['wins']}-{entry['draws']}-{entry['losses']} "
            f"({entry['score']:.2f})"
        )
    points_for, points_against = check["points_for"], check["points_against"]
    kills_for, kills_against = check["kills_for"], check["kills_against"]
    home = check["home_deaths"]
    cells += [
        "-"
        if points_for is None or points_against is None
        else f"{points_for:.1f}-{points_against:.1f}",
        "-" if check["margin"] is None else f"{check['margin']:+.1f}",
        "-"
        if kills_for is None or kills_against is None
        else f"{kills_for:.1f}/{kills_against:.1f}",
        "-" if check["red_zone_kills"] is None else f"{check['red_zone_kills']:.2f}",
        "-"
        if home is None or kills_against is None
        else f"{home:.1f} ({100 * home / kills_against:.0f}%)"
        if kills_against
        else f"{home:.1f}",
    ]
    return cells


def _best_text(checks: Sequence[Record]) -> str:
    """Format the routine check with the highest point margin, earliest on ties."""
    scored = [check for check in checks if check["margin"] is not None]
    if not scored:
        return "-"
    best = max(scored, key=lambda check: (check["margin"], -int(check["env_steps"])))
    return f"{int(best['env_steps']) / 1e6:.1f}M: {best['margin']:+.1f}"


def _group_cells(groups: Sequence[Record], columns: Sequence[str]) -> list[str]:
    """Format Table 3 cells "W-D-L (share%)" for one run in column order.

    The share is the group's part of our training ticks in finished games,
    or of finished games when any length is unknown. "-" marks a group with
    no finished game.
    """
    by_group = {group["group"]: group for group in groups}
    lengths = [group["length_sum"] for group in groups]
    use_ticks = all(length is not None for length in lengths)
    total = sum(
        (group["length_sum"] if use_ticks else group["games"]) for group in groups
    )
    cells: list[str] = []
    for column in columns:
        group = by_group.get(column)
        if group is None or not group["games"]:
            cells.append("-")
            continue
        part = group["length_sum"] if use_ticks else group["games"]
        share = f" ({100 * part / total:.0f}%)" if total else ""
        cells.append(f"{group['wins']}-{group['draws']}-{group['losses']}{share}")
    return cells


def _progress_text(
    root: Path,
    declared: Record,
    state: Record,
    runs: Sequence[Mapping[str, Any]],
    live: Sequence[Record],
    *,
    summary: bool,
    shaping: bool,
) -> str:
    """Build the study progress view: header, three tables, legend and logs.

    Parameters
    ----------
    root : Path
        The study folder.
    declared, state : dict
        The saved declaration.json and study.json (empty before a start).
    runs : sequence of mappings
        analysis.list_runs(root, progress=True) rows.
    live : sequence of dict
        _live_processes(root) records.
    summary : bool
        Add the point margin at every routine check, one row per case.
    shaping : bool
        Show every training window in Table 2, not only the first and latest.

    Returns
    -------
    str
        The complete text view, lines joined by newlines.
    """
    from datetime import datetime

    jobs = _jobs(declared)
    cases = cast(Record, state.get("cases") or {})
    by_run = {str(row["run_dir"]): row for row in runs}
    live_cases = {
        str(Path(entry["record"]).parent.name)
        for entry in live
        if entry["owner"].startswith("case ")
    }
    counts: dict[str, int] = {}
    for job in jobs:
        status = cases.get(job["id"], {}).get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    clock = datetime.now().astimezone().strftime("%-I:%M:%S %p %Z")
    lines = [
        f"Study {declared.get('name') or root.name} | {clock} | "
        f"Status: {state.get('status', 'not started')}",
        "Cases: " + ", ".join(f"{n} {name}" for name, n in sorted(counts.items())),
    ]
    for device, reason in cast(Record, state.get("paused_devices") or {}).items():
        why = reason.get("reason") or "failed before its first checkpoint"
        after = f" after case {reason['case_id']}" if reason.get("case_id") else ""
        lines.append(f"Paused GPU: {device} ({why}{after})")
    opponents: list[str] = []
    named: list[str] = []
    table_one: list[list[str]] = []
    curves: list[tuple[str, list[Record]]] = []
    windows_rows: list[list[str]] = []
    group_rows: list[tuple[str, list[Record]]] = []
    for job in jobs:
        run = by_run.get(str(root / "cases" / job["id"] / "run"), {})
        progress = cast(Record, run.get("progress") or {})
        for check in progress.get("checks", []):
            for entry in check["opponents"]:
                if entry["name"] not in opponents:
                    opponents.append(entry["name"])
        for group in progress.get("opponent_groups", []):
            if group["group"] not in (
                "current",
                "past",
                "pinned",
                "bots",
                *named,
            ):
                named.append(group["group"])
    for job in jobs:
        row = cast(Record, cases.get(job["id"]) or {})
        run = by_run.get(str(root / "cases" / job["id"] / "run"), {})
        progress = cast(Record, run.get("progress") or {})
        checks = cast(list[Record], progress.get("checks", []))
        status = row.get("status", "pending")
        running = status == "running"
        steps, total = run.get("env_steps"), progress.get("total_env_steps")
        speed = progress.get("steps_per_second")
        table_one.append(
            [
                job["id"],
                _seed_text(job),
                f"running: {progress['phase']}"
                if running and progress.get("phase")
                else status,
                f"{100 * steps / total:.1f}%"
                if isinstance(steps, int) and isinstance(total, int) and total
                else "-",
                f"{speed:,.0f}" if isinstance(speed, (int, float)) else "-",
                _case_gpu(root / "cases" / job["id"], row),
                _duration_text(progress.get("elapsed_seconds")),
                _duration_text(progress.get("seconds_left")) if running else "-",
                *_check_cells(checks[-1] if checks else None, opponents),
                _best_text(checks),
            ]
        )
        curves.append((job["id"], checks))
        windows = cast(list[Record], progress.get("windows", []))
        shown = list(enumerate(windows))
        if not shaping and len(windows) > 2:
            shown = [shown[0], shown[-1]]
        for index, window in shown:
            at = f"to {int(window['env_steps'] or 0) / 1e6:.1f}M"
            label = (
                f"{index + 1}, {at}"
                if shaping
                else f"first, {at}"
                if index == 0
                else f"latest, {at}"
            )
            windows_rows.append([job["id"], label, *_window_cells(window)])
        if not windows:
            windows_rows.append([job["id"], "-", *(["-"] * 12)])
        group_rows.append((job["id"], progress.get("opponent_groups", [])))
    lines += [
        "",
        "Table 1. Validation, latest finished routine check (all validation maps):",
        _text_table(
            [
                "Case",
                "Seed",
                "State",
                "Done",
                "Steps/s",
                "GPU",
                "Spent",
                "Left",
                "Check",
                *(f"vs {name}" for name in opponents),
                "Points",
                "Margin",
                "K/D",
                "RZ K",
                "Home D",
                "Best",
            ],
            table_one,
        ),
    ]
    if summary:
        steps_seen = sorted(
            {int(check["env_steps"]) for _, checks in curves for check in checks}
        )
        margin_rows: list[list[str]] = []
        for case, checks in curves:
            by_step = {int(check["env_steps"]): check["margin"] for check in checks}
            margin_rows.append(
                [
                    case,
                    *(
                        "-" if by_step.get(step) is None else f"{by_step[step]:+.1f}"
                        for step in steps_seen
                    ),
                ]
            )
        lines += [
            "",
            "Point margin at every routine check (learning curve):",
            _text_table(
                ["Case", *(f"{step / 1e6:.1f}M" for step in steps_seen)], margin_rows
            ),
        ]
    lines += [
        "",
        "Table 2. Training games (all opponents), Team A, per game unless stated"
        + (", every window:" if shaping else ", first and latest window:"),
        _text_table(
            [
                "Case",
                "Window",
                "Games",
                "Len",
                "Pts",
                "K/D",
                "Home D",
                "Kill Cr",
                "RZ Cr",
                "Dmg",
                "Heal",
                "Range",
                "Own zone",
                "Shaping",
            ],
            windows_rows,
        ),
    ]
    columns = ["current", "past"]
    if any(group["group"] == "pinned" for _, groups in group_rows for group in groups):
        columns.append("pinned")
    columns += [*named, "bots"]
    titles = {"current": "Current", "past": "Past copies", "pinned": "Pinned"}
    titles |= {name: f"M{index + 1}" for index, name in enumerate(named)}
    titles["bots"] = "Bots"
    lines += [
        "",
        "Table 3. Training results by opponent group, whole run so far: "
        "W-D-L (share of training ticks):",
        _text_table(
            ["Case", *(titles[column] for column in columns)],
            [[case, *_group_cells(groups, columns)] for case, groups in group_rows],
        ),
    ]
    lines += [f"  M{index + 1} = {name}" for index, name in enumerate(named)]
    for case, groups in group_rows:
        start = max((int(group["start_env_steps"]) for group in groups), default=0)
        if start:
            lines.append(f"  {case}: counts start at step {start:,} (continued run)")
    lines += ["", "What the columns mean:", *(f"  {line}" for line in _PROGRESS_LEGEND)]
    if live:
        lines += ["", "Live processes (nothing here is stopped automatically):"]
        for entry in live:
            pids = ", ".join(
                f"{name} {entry[key]}"
                for name, key in (
                    ("supervisor", "supervisor_pid"),
                    ("trainer", "trainer_pid"),
                )
                if entry[key] is not None
            )
            workers = " ".join(map(str, entry["worker_pids"]))
            lines.append(
                f"  {entry['owner']}: {pids or 'no supervisor or trainer'}"
                + (f"; workers {workers}" if workers else "")
                + ("; LEFTOVER workers with no supervisor" if entry["leftover"] else "")
            )
            lines.append(f"    Stop: {entry['stop_command']}")
    for job in jobs:
        if job["id"] not in live_cases:
            continue
        tail = _tail_lines(root / "cases" / job["id"] / "process.log", 5)
        lines += ["", f"== {job['id']} (live log) ==", *tail]
    return "\n".join(lines)


def study_status(
    output_dir: str | Path,
    *,
    progress: bool = False,
    summary: bool = False,
    shaping: bool = False,
) -> Record:
    """Read study state and exact live owners without JAX or file changes.

    Parameters
    ----------
    output_dir : str or Path
        An existing study folder.
    progress : bool, default=False
        Also build the progress view (the ``progress``, ``runs`` and
        ``live_processes`` keys). summary or shaping turn it on too.
    summary : bool, default=False
        Add the point margin at every routine check, one row per case, as a
        learning-curve table in the view.
    shaping : bool, default=False
        Show every training window in the view's Table 2, from start to
        finish, instead of only the first and latest.

    Returns
    -------
    dict
        Always: ``study_dir``, ``declaration_sha256``, ``clock``, ``study``
        (study.json), ``report`` (report.json) and ``owned_live_records``
        (live process-record paths). Missing results stay empty. With the
        view also: ``runs`` (analysis.list_runs(output_dir, progress=True)
        rows), ``live_processes`` (each live process with its PIDs and the
        exact stop command; leaderless workers are marked ``leftover``) and
        ``progress`` (the text view: one row per case with speed, time left,
        validation results, point margin and best check; the first and latest
        training windows with every combat/support reward part per game;
        training results by opponent group; a legend; and the last five lines
        of each live case's process.log).

    Raises
    ------
    ValueError, OSError
        The declaration is missing or malformed, or a saved file is malformed.

    Notes
    -----
    Reads files and /proc only. It never imports JAX, never queries a GPU
    (no nvidia-smi), never signals a process and never recovers, restarts or
    otherwise changes a run. Call it again to refresh; unchanged update logs
    are not parsed again in the same process.
    """
    root = Path(output_dir).expanduser().resolve()
    declared = _read(root / _DECLARATION)
    result: Record = {
        "study_dir": str(root),
        "declaration_sha256": _digest(declared),
        "clock": _optional(root / "launch_time.json"),
        "study": _optional(root / "study.json"),
        "report": _optional(root / "report.json"),
        "owned_live_records": [
            str(path) for path in _launch.live_process_records(_process_paths(root))
        ],
    }
    if progress or summary or shaping:
        from marl_battlegrounds.training.analysis import list_runs

        runs = list(list_runs(root, progress=True))
        live = _live_processes(root)
        result.update(
            runs=runs,
            live_processes=live,
            progress=_progress_text(
                root,
                declared,
                result["study"],
                runs,
                live,
                summary=summary,
                shaping=shaping,
            ),
        )
    return result


def stop_study(output_dir: str | Path) -> Record:
    """Request a stop from exact live study owners and return current status.

    output_dir names an existing study. Shared launcher checks guard PID reuse
    and signal only matching process owners. This is a request, not proof that
    cleanup has finished. Status stays backend-free and files are not removed.
    Read, permission and signalling errors propagate.
    """
    root = Path(output_dir).expanduser().resolve()
    signalled = _launch.request_stop(_process_paths(root))
    return {**study_status(root), "stop_requested_pids": signalled}


def _worker_resources(root: Path, request: Record) -> Record:
    """Apply this request's frozen slot before any numerical import or factory.

    Check that the slot and settings belong to the saved study. Explicit resource
    application refuses an already imported JAX backend. Return requested and
    applied environment facts; these are startup settings, not measured memory.
    No numerical module is imported. Legacy requests inherit their environment.
    """
    declared = _read(root / _DECLARATION)
    resources = request.get("resources")
    if declared.get("resources") is not None:
        slot = request.get("resource_slot")
        slots = _resource_slots(declared)
        if (
            type(slot) is not int
            or not 0 <= slot < len(slots)
            or resources != slots[slot]
        ):
            raise ValueError("Study worker resource slot differs from its declaration")
        resources = cast(Record, resources)
        if "jax" in sys.modules:
            raise RuntimeError(
                "Apply study resources in a fresh worker before importing JAX"
            )
        environment = _launch.resource_environment(
            resources["device"], compilation_cache=resources["compilation_cache"]
        )
        os.environ.update(environment)
    elif resources is not None or request.get("resource_slot") is not None:
        raise ValueError("Study worker has undeclared resources")
    return {
        "requested": resources,
        "request_sha256": _digest(request),
        "environment": _launch.resource_facts(os.environ),
    }


def _supervise_case(root: Path, identifier: str) -> int:
    """Own one declared worker process and preserve its bounded cleanup record.

    Called only in a separate supervisor process admitted by the study owner.
    Apply the checked environment before launching the ordinary worker, then use
    supervise_command for signals, deadlines, process-group cleanup and exit
    facts. Clear only this completed supervisor role; leftover workers remain
    visible and block further admission/recovery.
    """
    directory = _case_directory(root, identifier)
    request = _read(directory / "request.json")
    try:
        _worker_resources(root, request)
        return _launch.supervise_command(
            directory,
            _command("_worker", str(root), identifier),
            deadline_at=request.get("stop_at"),
        )
    finally:
        _finished_supervisor(directory)


def _worker(root: Path, identifier: str) -> int:
    """Run one saved case through ordinary train or explicit continuation."""
    directory = _case_directory(root, identifier)
    request = _read(directory / "request.json")
    job = request["job"]
    try:
        resources = _worker_resources(root, request)
        atomic_json(directory / "resources.json", resources)
        from marl_battlegrounds.training.config import config_from_dict
        from marl_battlegrounds.training.runner import extend_training, train

        if (directory / "run").resolve() != directory / "run":
            raise ValueError("Study run directory follows a link or escapes its case")
        declared = _read(root / _DECLARATION)
        declared_jobs = {row["id"]: row for row in _jobs(declared)}
        if job != declared_jobs.get(identifier):
            raise ValueError("Study worker request differs from its frozen declaration")
        method = (
            job["config"]["method"]
            if job["kind"] == "train"
            else job["parent_identity"]["method"]
        )
        actual_identity = _identities([method])[method]
        if actual_identity != declared["runtime_identities"][method]:
            raise ValueError(
                "Study worker source or dependencies differ from the "
                "frozen declaration. In an editable checkout, use an ignored "
                "output folder such as artifacts/ or a folder outside the checkout."
            )
        current_inputs = _file_inputs(job)
        if current_inputs != job.get("external_inputs", {}):
            raise ValueError(
                "Study external file input differs from its frozen identity"
            )
        if request["resume_from"] is not None and Path(
            request["resume_from"]
        ) != _latest(directory / "run"):
            raise ValueError(
                "Study worker resume differs from its saved latest checkpoint"
            )
        with ExitStack() as stack:
            runtime = _runtime_callbacks(
                job.get("runtime", {}),
                stack,
                identities=current_inputs.get("callback_inputs", {}),
            )
            stack.enter_context(
                study_attempt_scope(
                    directory / "run",
                    {
                        "study_attempt_id": request.get("study_attempt_id"),
                        "case_id": identifier,
                        "number": request["attempt"],
                        "declaration_sha256": _digest(declared),
                        "request_sha256": _digest(request),
                    },
                )
            )
            if request["resume_from"] is not None:
                if job["kind"] == "train":
                    result = train(
                        config_from_dict(job["config"]),
                        resume_from=request["resume_from"],
                        **runtime,
                    )
                else:
                    from marl_battlegrounds.training.checkpoints import (
                        read_checkpoint_description,
                    )

                    child = read_checkpoint_description(request["resume_from"])
                    context = child["metadata"].get("continuation", {})
                    changes = _object(job["changes"], "Continuation changes")
                    if isinstance(changes.get("validation"), dict):
                        future_bindings = changes["validation"].pop("bindings", None)
                        if "panel" in changes["validation"]:
                            runtime["validation_opponents"] = future_bindings
                    expected = {
                        "parent_checkpoint": job["checkpoint"],
                        "parent_checkpoint_id": job["parent_identity"]["checkpoint_id"],
                        "additional_env_steps": job["additional_env_steps"],
                        "changes": changes,
                    }
                    if any(
                        context.get(key) != value for key, value in expected.items()
                    ):
                        raise ValueError(
                            "Study continuation child differs from "
                            "its declared parent or extension"
                        )
                    result = train(resume_from=request["resume_from"], **runtime)
            elif job["kind"] == "train":
                result = train(
                    config_from_dict(job["config"]),
                    output_dir=directory / "run",
                    **runtime,
                )
            else:
                if _checkpoint_identity(job["checkpoint"]) != job["parent_identity"]:
                    raise ValueError(
                        "Study continuation parent differs from its frozen identity"
                    )
                result = extend_training(
                    job["checkpoint"],
                    additional_env_steps=job["additional_env_steps"],
                    output_dir=directory / "run",
                    changes=job["changes"],
                    **runtime,
                )
        atomic_json(
            directory / "outcome.json",
            {
                "status": result.status,
                "request_sha256": _digest(request),
                "runtime_identity": actual_identity,
                "run_dir": str(result.run_dir),
                "completed_env_steps": result.completed_env_steps,
                "completed_updates": result.completed_updates,
                "final_actor": str(result.final_actor),
                "selected_actor": None
                if result.selected_actor is None
                else str(result.selected_actor),
            },
        )
        return 0
    except Exception as error:
        atomic_json(
            directory / "outcome.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "request_sha256": _digest(request),
            },
        )
        return 1


def _analyze(root: Path) -> int:
    """Write ordinary saved-run analysis, preserving failures as report facts."""
    from marl_battlegrounds.training.analysis import analyze

    try:
        runs = [
            root / "cases" / job["id"] / "run"
            for job in _jobs(_read(root / _DECLARATION))
            if (root / "cases" / job["id"] / "run/run_details.json").exists()
        ]
        report = analyze(runs, output_dir=root / "analysis")
        atomic_json(
            root / "analysis_job/outcome.json", {"status": "complete", "report": report}
        )
        return 0
    except Exception as error:
        atomic_json(
            root / "analysis_job/outcome.json",
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run private subprocess roles; public commands use the package CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("_control", "_worker", "_supervise", "_analyze", "_normalize"),
    )
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("case_id", nargs="?")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "_normalize":
        try:
            declared = _normalize(_read(args.study_dir / "request.json")["config"])
            atomic_json(args.study_dir / "outcome.json", {"declaration": declared})
            return 0
        except Exception as error:
            atomic_json(
                args.study_dir / "outcome.json",
                {"error": f"{type(error).__name__}: {error}"},
            )
            return 1
    if args.action == "_control":
        result = _execute(args.study_dir, resume=args.resume, handoff=True)
        return 0 if result["status"] == "complete" else 1
    if args.action in ("_worker", "_supervise"):
        if args.case_id is None:
            parser.error(f"{args.action} requires a case ID")
        owner = _worker if args.action == "_worker" else _supervise_case
        return owner(args.study_dir, args.case_id)
    return _analyze(args.study_dir)


if __name__ == "__main__":
    raise SystemExit(main())
