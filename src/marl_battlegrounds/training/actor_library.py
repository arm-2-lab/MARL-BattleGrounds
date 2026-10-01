"""Collect saved actors into a portable folder with evidence-backed labels.

collect_actors reads trainer records and immutable models. Checkpoints own byte
verification and extraction; this module owns only inventory and catalog files.
It starts no learner or evaluation and creates no database. Relative actor paths
keep a collected library loadable after moving its root folder.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.training._checkpoint_metadata import (
    _relative,  # pyright: ignore[reportPrivateUsage]
    read_checkpoint_description,
    read_resume_checkpoint,
)
from marl_battlegrounds.training._run_io import append_jsonl, read_run_events, run_lock

Record = dict[str, Any]


def _read(path: Path) -> Record:
    """Read a JSON object, rejecting missing, malformed or non-object records."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(Record, value)


def _local_path(root: Path, kind: str, saved: str) -> Path:
    """Prefer the matching local model name after relocation; else keep its path."""
    local = root / kind / Path(saved).name
    return local if local.exists() or local.is_symlink() else Path(saved)


def _actor_facts(details: Record) -> Record:
    """Read inference facts through the checkpoint owner's historical rules."""
    from marl_battlegrounds.training import checkpoints

    return {
        "schemas": details["schemas"],
        "weight_digest": details["actor_digest"],
        "inference_id": checkpoints._inference_digest(details),  # pyright: ignore[reportPrivateUsage]
        "input_scale": checkpoints._actor_input_scale(details),  # pyright: ignore[reportPrivateUsage]
        "spawn_frame": checkpoints._actor_spawn_frame(details),  # pyright: ignore[reportPrivateUsage]
        "parameter_sharing": checkpoints._actor_parameter_sharing(details),  # pyright: ignore[reportPrivateUsage]
        # Only a PQN-VDN LayerNorm actor adds this fact; older facts stay equal.
        **(
            {}
            if checkpoints._actor_normalization(details) == "batch_norm"  # pyright: ignore[reportPrivateUsage]
            else {"normalization": checkpoints._actor_normalization(details)}  # pyright: ignore[reportPrivateUsage]
        ),
    }


def _require_recorded_prune(
    path: Path, identifier: str, pruned: set[str | None]
) -> None:
    """Reject a missing learner actor unless its payload removal was completed.

    path is the exact learner folder; identifier is its checked checkpoint ID.
    pruned contains IDs from that run's saved prune events. A missing actor with
    no event or remaining recovery state is a partial/corrupt input, not an
    unavailable historical model. Raise before creating collection outputs.
    """
    if identifier not in pruned or (path / "state").exists():
        raise ValueError(f"Missing actor without a complete recorded prune: {path}")


def _run_inventory(root: Path, *, strict: bool) -> tuple[Record, list[Record]]:
    """Describe declared, published and pruned models in one saved training run.

    Read small records and immediate model folders. Keep distinct captures and
    checkpoints even when inference bytes match. Verify model payloads later
    through export_actor. strict rejects missing promised exports; a recorded
    historical prune stays unavailable. Corrupt or partial models always fail.
    Published checkpoints outside active ancestry are labelled as orphans, never
    adopted as recovery or selected models. Return source evidence and candidates.
    """
    from marl_battlegrounds.training._run_io import (
        _check_past_copies,  # pyright: ignore[reportPrivateUsage]
        checkpoint_ancestry,
        saved_result_actors,
        selected_checkpoint_description,
    )

    info = _read(root / "run_details.json")
    config = info.get("config")
    if not isinstance(config, dict) or not isinstance(info.get("run_id"), str):
        raise ValueError(f"Not a training run: {root}")
    status = _read(root / "status.json") if (root / "status.json").exists() else {}
    if status and status.get("run_id") != info["run_id"]:
        raise ValueError(f"Run and status identities differ: {root}")
    latest, saved = read_resume_checkpoint(root)
    if saved["metadata"]["run_id"] != info["run_id"]:
        raise ValueError(f"Run and checkpoint identities differ: {root}")
    if saved["metadata"]["config"] != config:
        raise ValueError(f"Run and checkpoint settings differ: {root}")
    owners = _check_past_copies(
        root, saved, require_retained=strict, verify_payloads=False
    )
    host = saved["metadata"]["host_state"]
    ancestry = checkpoint_ancestry(root, saved)
    events = read_run_events(root / "run_events.jsonl")
    pruned = {
        row.get("checkpoint_id")
        for row in events
        if row.get("event") == "checkpoint_payload_pruned"
    }
    exports: dict[str, Path] = {}
    history: dict[str, Path] = {}
    for kind in ("actors", "past_copies"):
        directory = root / kind
        if directory.is_symlink():
            raise ValueError(f"Model folder follows a link: {directory}")
        for path in sorted(directory.iterdir()) if directory.exists() else []:
            if path.name.startswith("."):
                continue
            actor = read_checkpoint_description(path)
            if actor["kind"] != "actor":
                raise ValueError(f"Expected a standalone actor: {path}")
            meta = actor["metadata"]
            if (
                meta["run_id"] not in owners
                or (kind == "actors" and meta["run_id"] != info["run_id"])
                or meta["seed"] != owners[meta["run_id"]]["metadata"]["config"]["seed"]
                or actor["schemas"] != saved["schemas"]
            ):
                raise ValueError(
                    f"Published actor belongs to another run or method: {path}"
                )
            key = meta.get("checkpoint_id")
            if key is not None:
                if key in exports:
                    raise ValueError(
                        f"Multiple exports name the same checkpoint: {path}"
                    )
                exports[key] = path
            else:
                capture = str(meta["capture_origin"]["capture_id"])
                if capture in history:
                    raise ValueError(f"Multiple exports name the same capture: {path}")
                history[capture] = path
    for key, path in host.get("actors", {}).items():
        expected = _local_path(root, "actors", path)
        if key in exports and exports[key] != expected:
            raise ValueError(
                f"Saved actor path differs from published export: {expected}"
            )
        if not expected.exists():
            if strict and config.get("retain_all_actors", False):
                raise ValueError(f"Required retained actor is missing: {expected}")
        else:
            exports[key] = expected
    selection, final_path, selected_path = saved_result_actors(
        root, saved, status, ancestry=ancestry, allow_missing=not strict
    )
    complete = (
        status.get("status") == "complete"
        and not (root / "checkpoint_recovery.json").exists()
    )
    if complete and (
        status.get("env_steps") != saved["counters"]["env_steps"]
        or status.get("completed_updates") != saved["counters"]["updates"]
    ):
        raise ValueError(f"Complete run counters differ from its checkpoint: {root}")
    selected = None if selection is None or not complete else selection["checkpoint_id"]
    candidates: list[Record] = []
    seen: set[str] = set()
    for path in sorted((root / "checkpoints").iterdir()):
        if path.name.startswith("."):
            continue
        details = read_checkpoint_description(path)
        key = details["checkpoint_id"]
        if (
            details["kind"] != "learner"
            or details["metadata"]["run_id"] != info["run_id"]
        ):
            raise ValueError(f"Checkpoint belongs to another run: {path}")
        if key in seen:
            raise ValueError(f"Duplicate checkpoint identity: {path}")
        seen.add(key)
        roles = []
        if details["metadata"].get("parent_checkpoint") is None:
            roles.append("Initial")
        if key == selected:
            roles.append("Selected")
        if complete and path == latest:
            roles.append("Final")
        if key not in ancestry:
            roles.append("Published orphan")
        source = exports.get(key, path)
        if (
            complete
            and path == latest
            and final_path is not None
            and final_path.exists()
        ):
            source = final_path
        if key == selected and selected_path is not None and selected_path.exists():
            source = selected_path
        available = source != path or (path / "actor").exists()
        if source != path:
            exported = read_checkpoint_description(source)
            from marl_battlegrounds.training import checkpoints

            checkpoints._check_actor_origin(exported, details)  # pyright: ignore[reportPrivateUsage]
        if not available:
            _require_recorded_prune(path, key, pruned)
            if strict and details["metadata"]["config"].get("retain_all_actors", False):
                raise ValueError(f"Required retained actor is missing for {path}")
        elif (
            source == path
            and strict
            and config.get("retain_all_actors", False)
            and key in ancestry
            and key != saved["checkpoint_id"]
        ):
            raise ValueError(
                "Required actor export is missing; collect its matching checkpoint "
                f"with strict=False to preserve a separate library copy: {path}"
            )
        candidates.append(
            {
                "entry_id": sha256(
                    f"{info['run_id']}:checkpoint:{key}".encode()
                ).hexdigest(),
                "source": str(source),
                "source_checkpoint_id": key,
                "source_attempt_id": details["metadata"].get("attempt_id"),
                "env_steps": details["counters"]["env_steps"],
                "roles": roles or ["Checkpoint"],
                "availability": "Available"
                if available
                else "Unavailable: recorded prune",
                "selection_evidence": selection if key == selected else None,
                "selection_rule": info.get("selection_rule")
                if key == selected
                else None,
                "expected_actor_facts": _actor_facts(details),
                "expected_source_id": read_checkpoint_description(source)[
                    "checkpoint_id"
                ]
                if available
                else None,
            }
        )
    if selected is not None and selected not in seen:
        path = cast(Path, selected_path)
        boundary = selected_checkpoint_description(root, saved, selected)
        from marl_battlegrounds.training import checkpoints

        if not path.exists():
            path = checkpoints.artifact_directory(
                path.parent.parent, "checkpoints", selected
            )
            if path.exists():
                source = read_checkpoint_description(path)
                if source["checkpoint_id"] != boundary["checkpoint_id"]:
                    raise ValueError("Inherited selected source differs")
        available = path.exists() and (
            (path / "actor_details.json").is_file() or (path / "actor").is_dir()
        )
        if not available and path.exists():
            parent_events = read_run_events(path.parent.parent / "run_events.jsonl")
            _require_recorded_prune(
                path,
                selected,
                {
                    row.get("checkpoint_id")
                    for row in parent_events
                    if row.get("event") == "checkpoint_payload_pruned"
                },
            )
        actor = read_checkpoint_description(path) if available else boundary
        candidates.append(
            {
                "entry_id": sha256(
                    f"{info['run_id']}:inherited-selected:{selected}".encode()
                ).hexdigest(),
                "source": str(path),
                "source_checkpoint_id": selected,
                "source_attempt_id": actor["metadata"].get("attempt_id"),
                "env_steps": boundary["counters"]["env_steps"],
                "roles": ["Selected", "Inherited"],
                "availability": "Available"
                if available
                else "Unavailable: missing selected export",
                "selection_evidence": selection,
                "selection_rule": info.get("selection_rule"),
                "expected_actor_facts": _actor_facts(actor),
                "expected_source_id": actor["checkpoint_id"],
            }
        )
    if set(exports) - seen:
        raise ValueError(
            f"Actor export has no originating checkpoint description: {root}"
        )
    for capture, entry in host.get("past_copies", {}).items():
        path = _local_path(root, "past_copies", entry["path"])
        if capture in history and history[capture] != path:
            raise ValueError(f"History path differs from its saved catalog: {path}")
        history[capture] = path
    for capture, path in sorted(history.items(), key=lambda item: int(item[0])):
        entry = host.get("past_copies", {}).get(capture)
        available = path.exists()
        if available:
            description = read_checkpoint_description(path)
            metadata = description["metadata"]
            origin = metadata.get("capture_origin", {})
            if origin.get("capture_id") != int(capture) or (
                entry is not None
                and (
                    metadata["env_steps"] != entry["env_steps"]
                    or origin.get("epsilon") != entry["sampled_epsilon"]
                    or entry.get("artifact_id", description["checkpoint_id"])
                    != description["checkpoint_id"]
                )
            ):
                raise ValueError(
                    f"History actor differs from its saved capture: {path}"
                )
        else:
            description = None
            start = host.get("retention_start_capture_id", 0)
            if (
                strict
                and config.get("retain_all_actors", False)
                and int(capture) >= start
            ):
                raise ValueError(f"Required retained history actor is missing: {path}")
        candidates.append(
            {
                "entry_id": sha256(
                    f"{info['run_id']}:history:{capture}".encode()
                ).hexdigest(),
                "source": str(path),
                "source_checkpoint_id": None,
                "source_attempt_id": None,
                "env_steps": description["metadata"]["env_steps"]
                if description is not None
                else cast(Record, entry)["env_steps"],
                "roles": ["History"]
                if entry is not None
                else ["Unaccepted history capture"],
                "availability": "Available"
                if available
                else "Unavailable: historical export missing",
                "selection_evidence": None,
                "selection_rule": None,
                "expected_actor_facts": None
                if description is None
                else _actor_facts(description),
                "expected_source_id": None
                if description is None
                else description["checkpoint_id"],
            }
        )
    return {"run": info, "status": status, "selection": selection}, candidates


def _copy_origin_description(actor: Record, source: Path, destination: Path) -> None:
    """Preserve available actor-to-learner evidence beside a collected model.

    actor is already checked. source is its original actor or learner folder;
    destination is the locked library root. Capture actors need no extra file.
    Copy only a checked matching learner description, never recovery payloads.
    Missing historical closure remains missing. Existing unequal descriptions
    raise; identical ones keep their bytes. The catalog is published afterward.
    """
    from marl_battlegrounds.training import checkpoints

    identifier = actor["metadata"].get("checkpoint_id")
    if identifier is None:
        return
    origin = checkpoints.artifact_directory(
        source.resolve().parent.parent, "checkpoints", identifier
    )
    if not origin.exists():
        return
    description = read_checkpoint_description(origin)
    checkpoints._check_actor_origin(actor, description)  # pyright: ignore[reportPrivateUsage]
    root = destination / "checkpoints"
    if root.is_symlink():
        raise ValueError(f"Collection provenance follows a link: {root}")
    target = root / identifier
    if target.is_symlink():
        raise ValueError(f"Collection provenance follows a link: {target}")
    if target.exists():
        if read_checkpoint_description(target) != description:
            raise ValueError(f"Collected origin description conflicts: {target}")
        return
    root.mkdir(exist_ok=True)
    with checkpoints._pending_write(root, kind="learner") as temporary:  # pyright: ignore[reportPrivateUsage]
        checkpoints._atomic_json(  # pyright: ignore[reportPrivateUsage]
            temporary / "checkpoint_details.json", description
        )
        if read_checkpoint_description(temporary) != description:
            raise ValueError("Copied origin description changed before publication")
        temporary.rename(target)
        checkpoints._sync_dir(root)  # pyright: ignore[reportPrivateUsage]


def collect_actors(
    run_dirs: Sequence[str | Path], *, output_dir: str | Path, strict: bool = True
) -> Path:
    """Collect saved actors and their roles into a portable local model folder.

    Parameters
    ----------
    run_dirs : sequence of str or Path
        Training roots to add or refresh. Each must have a latest saved learner
        description. Include failed runs when they contain useful saved models.
    output_dir : str or Path
        Model-library root. Models go to models/, run evidence to runs/,
        and checked origin descriptions to checkpoints/. The
        authoritative catalog is catalog.json. Paths inside it are root-relative.
    strict : bool, default=True
        Reject missing exports promised by complete retention. False permits
        extraction from a matching surviving learner. Recorded historical pruning
        remains unavailable; corruption is never silently labelled pruning.

    Returns
    -------
    Path
        Absolute catalog.json. Its index path names a matching readable CSV.
        Each checkpoint/capture remains a separate row, including duplicate
        inference identities. Selected labels retain their rule and saved evidence.

    Raises
    ------
    ValueError, OSError, RuntimeError
        Inputs or existing catalog conflict, promised actors are absent, files
        cannot be read/written, or another collector holds the output lock.

    Notes
    -----
    No training, evaluation or numerical restore. Verification hashes files and
    can read full learner payloads. Existing matching models are reused without
    copying. Publish the content-named CSV before replacing the catalog commit
    point. An interrupted call can leave verified models/evidence but never
    points the old catalog at a partly written new index. Retry merges those files.
    Source runs and immutable actors are never edited. Anonymous distribution is
    a separate operation; this catalog preserves original provenance.
    """
    from marl_battlegrounds.evaluation.revision import (
        _check_output_path,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training import checkpoints
    from marl_battlegrounds.training.analysis import (  # pyright: ignore[reportPrivateUsage]
        _atomic_text,  # pyright: ignore[reportPrivateUsage]
        _csv,  # pyright: ignore[reportPrivateUsage]
    )

    if isinstance(run_dirs, (str, Path)) or not run_dirs:
        raise ValueError("run_dirs must be a nonempty sequence of training folders")
    if type(strict) is not bool:
        raise TypeError("strict must be a boolean")
    roots = tuple(dict.fromkeys(Path(path).expanduser().resolve() for path in run_dirs))
    inputs = [(root, *_run_inventory(root, strict=strict)) for root in roots]
    destination = Path(output_dir).expanduser().absolute()
    resolved = destination.resolve()
    _check_output_path(destination)
    sources = {
        *roots,
        *(
            Path(candidate["source"]).resolve()
            for _, _, candidates in inputs
            for candidate in candidates
        ),
    }
    if destination.is_symlink() or any(
        resolved == source or resolved in source.parents or source in resolved.parents
        for source in sources
    ):
        raise ValueError("Collection output must be separate from its source runs")
    destination.mkdir(parents=True, exist_ok=True)
    with run_lock(destination):
        for name in (
            "models",
            "runs",
            "checkpoints",
            "catalog.json",
            "collection_events.jsonl",
        ):
            if (destination / name).is_symlink():
                raise ValueError(
                    f"Collection output follows a link: {destination / name}"
                )
        models = destination / "models"
        models.mkdir(exist_ok=True)
        for parent in (models, destination / "checkpoints"):
            if parent.exists():
                for event in checkpoints._cleanup_pending_writes(parent):  # pyright: ignore[reportPrivateUsage]
                    append_jsonl(
                        destination / "collection_events.jsonl", event, durable=True
                    )
        catalog_path = destination / "catalog.json"
        catalog: Record = (
            _read(catalog_path)
            if catalog_path.exists()
            else {"schema_version": 1, "actors": []}
        )
        if catalog.get("schema_version") != 1 or not isinstance(
            catalog.get("actors"), list
        ):
            raise ValueError(f"Unsupported model catalog: {catalog_path}")
        if "index" in catalog:
            index = _relative(destination, catalog["index"])
            digest = sha256(index.read_bytes()).hexdigest()
            if index.name != f"actors-{digest}.csv":
                raise ValueError("Saved catalog CSV identity differs")
        if any(
            not isinstance(row, dict) or not isinstance(row.get("entry_id"), str)
            for row in catalog["actors"]
        ):
            raise ValueError("Saved model catalog has an invalid actor row")
        merged: dict[str, Record] = {row["entry_id"]: row for row in catalog["actors"]}
        if len(merged) != len(catalog["actors"]):
            raise ValueError("Saved model catalog repeats entry identities")
        for root, evidence, candidates in inputs:
            run = evidence["run"]
            evidence_bytes = json.dumps(
                evidence, sort_keys=True, allow_nan=False
            ).encode()
            evidence_name = Path("runs") / f"{sha256(evidence_bytes).hexdigest()}.json"
            (destination / "runs").mkdir(exist_ok=True)
            evidence_path = destination / evidence_name
            if evidence_path.exists():
                if _read(evidence_path) != evidence:
                    raise ValueError(
                        f"Saved collection evidence conflicts: {evidence_path}"
                    )
            else:
                checkpoints._atomic_json(evidence_path, evidence)  # pyright: ignore[reportPrivateUsage]
            for candidate in candidates:
                expected_facts = candidate.pop("expected_actor_facts")
                expected_source_id = candidate.pop("expected_source_id")
                entry: Record = {
                    **candidate,
                    "run_id": run["run_id"],
                    "source_run": str(root),
                    "run_status": evidence["status"].get("status", "incomplete"),
                    "run_evidence": evidence_name.as_posix(),
                    "method": run["config"].get("method", "mappo"),
                    "seed": run["config"]["seed"],
                    "actor_path": None,
                    "actor_export_id": None,
                    "inference_id": None,
                    "weight_digest": None,
                }
                if candidate["availability"] == "Available":
                    name = checkpoints.artifact_name(
                        entry["method"], "actor", entry["env_steps"], entry["entry_id"]
                    )
                    previous = merged.get(entry["entry_id"])
                    actor_path = (
                        _relative(destination, previous["actor_path"])
                        if previous is not None and previous.get("actor_path")
                        else models / name
                    )
                    if actor_path.exists():
                        source = checkpoints.read_checkpoint_details(
                            candidate["source"]
                        )
                        actor = checkpoints.read_checkpoint_details(actor_path)
                        if (
                            source["checkpoint_id"] != expected_source_id
                            or (
                                previous is not None
                                and previous.get("actor_path")
                                and actor["checkpoint_id"]
                                != previous["actor_export_id"]
                            )
                            or _actor_facts(source) != expected_facts
                        ):
                            raise ValueError(
                                "Collected source or saved actor identity differs"
                            )
                        if source["kind"] == "learner":
                            checkpoints._check_actor_origin(actor, source)  # pyright: ignore[reportPrivateUsage]
                        elif any(
                            actor["metadata"].get(key) != source["metadata"].get(key)
                            for key in (
                                "run_id",
                                "seed",
                                "env_steps",
                                "checkpoint_id",
                                "capture_origin",
                            )
                        ):
                            raise ValueError(
                                "Collected actor origin differs from its source"
                            )
                    else:
                        actor_path = checkpoints.export_actor(
                            candidate["source"], output_dir=models / name
                        )
                        actor = read_checkpoint_description(actor_path)
                        origin = actor["metadata"].get("checkpoint_id")
                        if expected_source_id not in (origin, actor["checkpoint_id"]):
                            raise ValueError(
                                "Collected source identity changed during export"
                            )
                    _copy_origin_description(
                        actor, Path(candidate["source"]), destination
                    )
                    entry["source_artifact_id"] = expected_source_id
                    metadata = actor["metadata"]
                    entry.update(
                        actor_path=actor_path.relative_to(destination).as_posix(),
                        actor_export_id=actor["checkpoint_id"],
                        inference_id=checkpoints._inference_digest(actor),  # pyright: ignore[reportPrivateUsage]
                        weight_digest=actor["actor_digest"],
                        input_scale=checkpoints._actor_input_scale(actor),  # pyright: ignore[reportPrivateUsage]
                        spawn_frame=checkpoints._actor_spawn_frame(actor),  # pyright: ignore[reportPrivateUsage]
                        parameter_sharing=checkpoints._actor_parameter_sharing(actor),  # pyright: ignore[reportPrivateUsage]
                        schemas=actor["schemas"],
                        source_actor_run_id=metadata["run_id"],
                        source_actor_seed=metadata["seed"],
                        optimizer_steps=metadata.get("optimizer_steps"),
                        action_behavior="Greedy"
                        if entry["method"] in ("qmix", "pqn_vdn")
                        else "Sampled",
                        captured_epsilon=metadata.get("capture_origin", {}).get(
                            "epsilon"
                        ),
                    )
                    if (
                        expected_facts is not None
                        and _actor_facts(actor) != expected_facts
                    ):
                        raise ValueError(
                            f"Collected actor inference facts differ: {actor_path}"
                        )
                entry["source_availability"] = candidate["availability"]
                previous = merged.get(entry["entry_id"])
                if (
                    entry["actor_path"] is None
                    and previous is not None
                    and previous.get("actor_path")
                ):
                    kept = _relative(destination, previous["actor_path"])
                    identity = checkpoints.artifact_identity(kept)
                    if identity["checkpoint_id"] != previous["actor_export_id"]:
                        raise ValueError(
                            f"Saved collected actor identity differs: {kept}"
                        )
                    for key in (
                        "actor_path",
                        "actor_export_id",
                        "inference_id",
                        "weight_digest",
                        "input_scale",
                        "spawn_frame",
                        "parameter_sharing",
                        "schemas",
                        "source_artifact_id",
                        "source_actor_run_id",
                        "source_actor_seed",
                        "optimizer_steps",
                        "action_behavior",
                        "captured_epsilon",
                    ):
                        entry[key] = previous.get(key)
                    entry["availability"] = "Available"
                merged[entry["entry_id"]] = entry
        rows = [merged[key] for key in sorted(merged)]
        index_rows = [
            {
                key: json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            for row in rows
        ]
        text = _csv(index_rows)
        index_name = f"actors-{sha256(text.encode()).hexdigest()}.csv"
        index_path = destination / index_name
        if index_path.exists():
            if index_path.read_text() != text:
                raise ValueError(f"Saved model index conflicts: {index_path}")
        else:
            _atomic_text(index_path, text)
        updated = {"schema_version": 1, "index": index_name, "actors": rows}
        if updated != catalog:
            checkpoints._sync_dir(destination)  # pyright: ignore[reportPrivateUsage]
            checkpoints._atomic_json(catalog_path, updated)  # pyright: ignore[reportPrivateUsage]
        return catalog_path.resolve()
