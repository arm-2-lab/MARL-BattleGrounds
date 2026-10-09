"""Check portable actor extraction, collection and read-only run discovery.

Real PPO/QMIX/PQN payloads use the existing shared learner fixture. Extraction
must not restore arrays or optimizers; loading afterward checks the resulting
inference identity and Q behavior. Empty-payload host fixtures separately check
signed descriptions, catalog roles and inherited references. Their synthetic
step counts are not executed transitions. These checks make no learning claim.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.training.test_training_checkpoint_writes import saved_family as saved_family

from marl_battlegrounds.training import checkpoints, collect_actors, list_runs
from marl_battlegrounds.training._run_io import append_jsonl


def test_real_actor_export_collect_relocate_and_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_family: tuple[Any, Any, dict[str, Any], dict[str, Any]],
) -> None:
    collection, state, metadata, settings = saved_family
    method = settings["method"]
    root = tmp_path / "source run"
    root.mkdir()
    checkpoint = checkpoints.save_checkpoint(
        root, collection, state, metadata=metadata, **settings
    )
    details = checkpoints.read_checkpoint_description(checkpoint)
    info = {
        "run_id": metadata["run_id"],
        "config": metadata["config"],
        "selection_rule": None,
    }
    (root / "run_details.json").write_text(json.dumps(info))
    (root / "status.json").write_text(
        json.dumps({"run_id": info["run_id"], "status": "failed", "env_steps": 0})
    )
    actor = root / "actors" / "initial actor"

    def forbidden(*_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
        raise AssertionError("Actor export restored numerical arrays")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_restore_arrays", forbidden)
        assert checkpoints.export_actor(checkpoint, output_dir=actor) == actor
        assert checkpoints.export_actor(checkpoint, output_dir=actor) == actor
    original = checkpoints.artifact_identity(actor)
    from marl_battlegrounds.training._content import (
        TrainingContentBinding,
        pinned_opponent_evidence,
    )

    binding = TrainingContentBinding.model_validate_json(
        json.dumps(details["collection"]["content_binding"])
    )
    original_exposure = pinned_opponent_evidence(
        binding, checkpoints.load_system(actor), export=actor
    )
    assert original_exposure["source"] == "verified export"
    assert original["weight_digest"] == details["actor_digest"]
    assert (
        original["actor_digest"]
        == checkpoints.artifact_identity(checkpoint)["actor_digest"]
    )
    assert not (actor / "state").exists()
    system = checkpoints.load_system(actor)
    assert system.checkpoint == original["actor_digest"]
    if method != "mappo":
        assert float(system.variables.epsilon) == 0.0
    library = tmp_path / "model library"
    catalog_path = collect_actors([root], output_dir=library)
    catalog = json.loads(catalog_path.read_text())
    row = catalog["actors"][0]
    assert len(catalog["actors"]) == 1
    assert row["roles"] == ["Initial"] and row["run_status"] == "failed"
    assert row["selection_evidence"] is None
    assert (
        checkpoints.artifact_identity(library / row["actor_path"])["checkpoint_id"]
        == original["checkpoint_id"]
    )
    before = {
        p.relative_to(library): p.stat().st_mtime_ns
        for p in library.rglob("*")
        if p.is_file() and p.name != ".training.lock"
    }
    with monkeypatch.context() as patch:
        patch.setattr(checkpoints.shutil, "copytree", forbidden)
        assert collect_actors([root], output_dir=library) == catalog_path
    assert before == {
        p.relative_to(library): p.stat().st_mtime_ns
        for p in library.rglob("*")
        if p.is_file() and p.name != ".training.lock"
    }
    # A subsequent historical prune cannot delete an already collected actor.
    shutil.rmtree(actor)
    for name in ("actor", "state"):
        shutil.rmtree(checkpoint / name)
    append_jsonl(
        root / "run_events.jsonl",
        {
            "event": "checkpoint_payload_pruned",
            "checkpoint_id": details["checkpoint_id"],
        },
        durable=True,
    )
    collect_actors([root], output_dir=library)
    saved = json.loads(catalog_path.read_text())["actors"][0]
    assert saved["availability"] == "Available"
    assert saved["source_availability"] == "Unavailable: recorded prune"
    moved = tmp_path / "relocated"
    library.rename(moved)
    loaded = checkpoints.load_system(moved / saved["actor_path"])
    assert loaded.checkpoint == original["actor_digest"]
    assert (
        pinned_opponent_evidence(binding, loaded, export=moved / saved["actor_path"])
        == original_exposure
    )
    assert (
        checkpoints.read_checkpoint_description(
            moved / "checkpoints" / details["checkpoint_id"]
        )
        == details
    )
    rows = list_runs(tmp_path)
    assert any(
        row["run_dir"] == str(root) and row["status"] == "failed" for row in rows
    )


def _host_run(root: Path) -> tuple[dict[str, Any], Path]:
    from tests.training.test_training_progress import _actor, _description, _saved_host

    from marl_battlegrounds.training._run_io import atomic_json

    root.mkdir()
    details = _description(root, _saved_host(steps=0, pending=None))
    actor = _actor(root, details)
    atomic_json(
        root / "run_details.json",
        {"run_id": "one-run", "config": details["metadata"]["config"]},
    )
    atomic_json(
        root / "latest_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_id": details["checkpoint_id"],
            "relative_path": f"checkpoints/{details['checkpoint_id']}",
        },
    )
    atomic_json(
        root / "status.json",
        {
            "run_id": "one-run",
            "status": "failed",
            "env_steps": 0,
            "completed_updates": 0,
        },
    )
    atomic_json(root / "selection.json", None)
    return details, actor


def test_collection_publication_lock_and_duplicate_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.training.test_training_progress import _actor, _description, _saved_host

    from marl_battlegrounds.training._run_io import run_lock

    root = tmp_path / "run"
    first, _ = _host_run(root)
    library = tmp_path / "models"
    catalog = collect_actors([root], output_dir=library)
    before = catalog.read_bytes()
    # A complete published orphan stays separate despite identical actor weights.
    orphan = _description(root, _saved_host(steps=8, pending=None), parent=first)
    _actor(root, orphan)
    atomic = checkpoints._atomic_json

    def interrupt(path: Path, value: object) -> None:
        if path == catalog:
            raise OSError("Injected catalog interruption")
        atomic(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_atomic_json", interrupt)
        with pytest.raises(OSError, match="catalog interruption"):
            collect_actors([root], output_dir=library)
    assert catalog.read_bytes() == before
    old = json.loads(before)
    assert (library / old["index"]).is_file()
    collect_actors([root], output_dir=library)
    rows = json.loads(catalog.read_text())["actors"]
    assert len(rows) == 2 and len({row["inference_id"] for row in rows}) == 1
    assert any(row["roles"] == ["Published orphan"] for row in rows)
    with run_lock(library), pytest.raises(RuntimeError, match="active writer"):
        collect_actors([root], output_dir=library)
    # Catalog and index must remain a linked pair after an interrupted refresh.
    latest = json.loads(catalog.read_text())
    (library / latest["index"]).write_text("changed")
    with pytest.raises(ValueError, match="CSV identity"):
        collect_actors([root], output_dir=library)


def test_run_loading_discovery_and_relocation(tmp_path: Path) -> None:
    from marl_battlegrounds import _cli
    from marl_battlegrounds.training import load_run
    from marl_battlegrounds.training._run_io import atomic_json

    root = tmp_path / "completed run"
    details, actor = _host_run(root)
    with pytest.raises(ValueError, match="not complete"):
        load_run(root)
    (root / "report.csv").write_text("value\n1\n")
    status = {
        "run_id": "one-run",
        "status": "complete",
        "env_steps": 0,
        "completed_updates": 0,
        "final_actor": str(actor),
        "selected_actor": None,
        "evidence_paths": ["report.csv"],
    }
    atomic_json(root / "status.json", status)
    result = load_run(root)
    assert result.selected_actor is None and result.final_actor == actor
    assert result.evidence_paths == (root / "report.csv",)
    moved = tmp_path / "moved"
    root.rename(moved)
    assert load_run(moved).final_actor == moved / "actors" / actor.name
    assert _cli.main(["runs", str(tmp_path)]) == 0
    assert _cli.main(["runs", str(tmp_path), "--json"]) == 0
    atomic_json(moved / "checkpoint_recovery.json", {})
    assert list_runs(tmp_path)[0]["status"] == "recovering"
    with pytest.raises(ValueError, match="not complete"):
        load_run(moved)
    (moved / "checkpoint_recovery.json").unlink()
    (moved / "status.json").write_text("{")
    assert list_runs(tmp_path)[0]["status"] == "unreadable"
    assert details["checkpoint_id"]


def test_collection_rejects_source_overlap_and_preserves_failed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.training.test_training_progress import _actor, _description, _saved_host

    root = tmp_path / "run"
    details, actor = _host_run(root)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    for destination in (root, root / "library", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            collect_actors([root], output_dir=destination)
    with pytest.raises(ValueError, match="immutable source"):
        checkpoints.export_actor(actor, output_dir=actor / "copy")
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    copied = tmp_path / "copied"
    checkpoints.export_actor(actor, output_dir=copied)
    other = _description(root, _saved_host(steps=8, pending=None), parent=details)
    other_actor = _actor(root, other)
    with pytest.raises(ValueError, match="different artifact"):
        checkpoints.export_actor(other_actor, output_dir=copied)
    copytree = checkpoints.shutil.copytree

    def interrupt(source: Path, destination: Path, *, dirs_exist_ok: bool) -> None:
        copytree(source, destination, dirs_exist_ok=dirs_exist_ok)
        raise OSError("Injected copy interruption")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints.shutil, "copytree", interrupt)
        with pytest.raises(OSError, match="copy interruption"):
            checkpoints.export_actor(actor, output_dir=tmp_path / "interrupted")
    assert not (tmp_path / "interrupted").exists()
    assert not list(tmp_path.glob(".pending-*"))
    assert (
        checkpoints.read_checkpoint_description(copied)["metadata"]["checkpoint_id"]
        == details["checkpoint_id"]
    )


def test_collection_cleans_proven_dead_writes_under_existing_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _host_run(root)
    library = tmp_path / "library"
    collect_actors([root], output_dir=library)
    program = """
import os, sys
from pathlib import Path
from marl_battlegrounds.training.checkpoints import _pending_write
with _pending_write(Path(sys.argv[1]), kind="actor", attempt_id="crashed") as path:
    (path / "partial").write_bytes(b"unfinished")
    os._exit(23)
"""
    process = subprocess.run(
        [sys.executable, "-B", "-c", program, str(library / "models")], check=False
    )
    assert process.returncode == 23
    assert len(list((library / "models").glob(".pending-*"))) == 2
    collect_actors([root], output_dir=library)
    assert not list((library / "models").glob(".pending-*"))
    events = [
        json.loads(line)
        for line in (library / "collection_events.jsonl").read_text().splitlines()
    ]
    assert any(row["status"] == "removed" for row in events)


def test_complete_roles_and_never_started_study_are_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from marl_battlegrounds import _cli
    from marl_battlegrounds.training import load_run
    from marl_battlegrounds.training._run_io import atomic_json

    root = tmp_path / "run"
    details, actor = _host_run(root)
    (root / "report.csv").write_text("value\n1\n")
    selection = {
        "checkpoint_id": details["checkpoint_id"],
        "rule": "Saved fixture rule",
    }
    atomic_json(root / "selection.json", selection)
    status = {
        "run_id": "one-run",
        "status": "complete",
        "env_steps": 0,
        "completed_updates": 0,
        "final_actor": str(actor),
        "selected_actor": str(actor),
        "evidence_paths": ["report.csv"],
    }
    atomic_json(root / "status.json", status)
    result = load_run(root)
    assert result.final_actor == result.selected_actor
    catalog = collect_actors([root], output_dir=tmp_path / "library")
    row = json.loads(catalog.read_text())["actors"][0]
    assert row["roles"] == ["Initial", "Selected", "Final"]
    assert row["selection_evidence"] == selection
    status["env_steps"] = 1
    atomic_json(root / "status.json", status)
    with pytest.raises(ValueError, match="counters differ"):
        collect_actors([root], output_dir=tmp_path / "library")
    study = tmp_path / "study"
    study.mkdir()
    atomic_json(
        study / "declaration.json",
        {
            "cases": [
                {
                    "id": "recipe",
                    "config": {"method": "qmix"},
                    "seeds": [12, 13],
                    "duration_seconds": 30,
                }
            ]
        },
    )
    rows = list_runs(study)
    assert [row["case_id"] for row in rows] == ["recipe--seed-12", "recipe--seed-13"]
    assert all(row["status"] == "pending" and row["reason"] for row in rows)
    atomic_json(
        study / "study.json",
        {
            "status": "stopped",
            "cases": {
                "recipe--seed-12": {"id": "recipe--seed-12", "status": "interrupted"}
            },
        },
    )
    assert _cli.main(["study", "status", str(study)]) == 0
    output = capsys.readouterr().out
    assert "Case" in output and "recipe--seed-12" in output and "interrupted" in output


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop("checkpoint_id", None)
    result["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(result)
    ).hexdigest()
    return result


def _revise_learner(root: Path, details: dict[str, Any]) -> dict[str, Any]:
    from marl_battlegrounds.training._run_io import atomic_json

    old = root / "checkpoints" / details["checkpoint_id"]
    revised = _signed(details)
    new = root / "checkpoints" / revised["checkpoint_id"]
    if old != new:
        old.rename(new)
    atomic_json(new / "checkpoint_details.json", revised)
    atomic_json(
        root / "latest_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_id": revised["checkpoint_id"],
            "relative_path": f"checkpoints/{revised['checkpoint_id']}",
        },
    )
    atomic_json(
        root / "run_details.json",
        {
            "run_id": revised["metadata"]["run_id"],
            "config": revised["metadata"]["config"],
        },
    )
    atomic_json(
        root / "status.json",
        {
            "run_id": revised["metadata"]["run_id"],
            "status": "failed",
            "env_steps": revised["counters"]["env_steps"],
            "completed_updates": revised["counters"]["updates"],
        },
    )
    return revised


def _history_run(
    root: Path, history: Path, *, owner: str = "one-run", retain: bool = False
) -> dict[str, Any]:
    from tests.training.test_training_progress import _actor, _description, _saved_host

    from marl_battlegrounds.training._run_io import atomic_json

    root.mkdir()
    actor = _signed(
        {
            "schema_version": 1,
            "kind": "actor",
            "schemas": checkpoints._SCHEMAS,
            "metadata": {
                "run_id": owner,
                "seed": 42,
                "env_steps": 8,
                "capture_origin": {
                    "capture_id": 0,
                    "update_index": 0,
                    "content_binding": {},
                    "epsilon": None,
                },
            },
            "actor_layout": [],
            "actor_digest": "b" * 64,
            "files": {},
        }
    )
    history.mkdir(parents=True)
    atomic_json(history / "actor_details.json", actor)
    host = _saved_host(steps=16, pending=None)
    host.update(
        retention_start_capture_id=0,
        past_copies={
            "0": {
                "env_steps": 8,
                "path": str(history),
                "sampled_epsilon": None,
                "artifact_id": actor["checkpoint_id"],
            }
        },
    )
    details = _description(root, host)
    details["collection"] = {"history_capture_capacity": 2, "content_binding": {}}
    details["metadata"]["config"]["retain_all_actors"] = retain
    details = _revise_learner(root, details)
    _actor(root, details)
    return details


@pytest.mark.parametrize("strict", [True, False])
def test_relaxed_collection_never_accepts_foreign_history(
    tmp_path: Path, strict: bool
) -> None:
    root = tmp_path / "run"
    _history_run(root, tmp_path / "foreign", owner="foreign")
    with pytest.raises(ValueError, match="different run"):
        collect_actors([root], output_dir=tmp_path / "library", strict=strict)
    assert not (tmp_path / "library").exists()


def test_external_history_is_preserved_and_verified_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, history = tmp_path / "run", tmp_path / "external"
    _history_run(root, history, retain=True)
    before = {p: p.read_bytes() for p in history.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="separate"):
        collect_actors([root], output_dir=history / "library")
    assert before == {p: p.read_bytes() for p in history.rglob("*") if p.is_file()}
    inventory = checkpoints._inventory
    scanned: list[Path] = []

    def counted(path: Path) -> dict[str, Any]:
        scanned.append(path)
        return inventory(path)

    monkeypatch.setattr(checkpoints, "_inventory", counted)
    catalog = collect_actors([root], output_dir=tmp_path / "library")
    assert scanned.count(history) == 1
    assert len(scanned) == 4  # Two sources and their two published copies.
    rows = json.loads(catalog.read_text())["actors"]
    assert any(
        row["roles"] == ["History"] and row["captured_epsilon"] is None for row in rows
    )
    shutil.rmtree(history)
    with pytest.raises(ValueError, match="Required retained history"):
        collect_actors([root], output_dir=tmp_path / "missing-strict")
    relaxed = collect_actors(
        [root], output_dir=tmp_path / "missing-relaxed", strict=False
    )
    row = next(
        row
        for row in json.loads(relaxed.read_text())["actors"]
        if row["roles"] == ["History"]
    )
    assert row["actor_path"] is None and row["availability"].startswith("Unavailable")


@pytest.mark.parametrize(
    "missing_export", ["present", "recoverable", "partial", "pruned"]
)
def test_relocated_inherited_history_and_selected_actor(
    tmp_path: Path, missing_export: str
) -> None:
    from tests.training.test_training_progress import _actor, _description

    from marl_battlegrounds.training import load_run
    from marl_battlegrounds.training._run_io import append_jsonl, atomic_json

    parent_root = tmp_path / "parent"
    history = parent_root / "past_copies" / "capture"
    parent = _history_run(parent_root, history)
    parent_path = parent_root / "checkpoints" / parent["checkpoint_id"]
    child = tmp_path / "child"
    child.mkdir()
    details = _description(child, copy.deepcopy(parent["metadata"]["host_state"]))
    details["collection"] = parent["collection"]
    details["metadata"]["run_id"] = "child"
    details["metadata"]["continuation"] = {
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_id": parent["checkpoint_id"],
        "inherited_candidates": [
            {
                "parent_checkpoint": str(parent_path),
                "parent_checkpoint_id": parent["checkpoint_id"],
                "run_id": "one-run",
                "tasks": [{"checkpoint_id": parent["checkpoint_id"]}],
            }
        ],
    }
    details = _revise_learner(child, details)
    actor = _actor(child, details)
    description = json.loads((actor / "actor_details.json").read_text())
    description["metadata"]["run_id"] = "child"
    atomic_json(actor / "actor_details.json", _signed(description))
    shutil.copytree(history, child / "past_copies" / "capture")
    inherited_actor = parent_root / "actors" / parent["checkpoint_id"]
    atomic_json(child / "selection.json", {"checkpoint_id": parent["checkpoint_id"]})
    (child / "report.csv").write_text("value\n1\n")
    atomic_json(
        child / "status.json",
        {
            "run_id": "child",
            "status": "complete",
            "env_steps": 16,
            "completed_updates": 1,
            "final_actor": str(actor),
            "selected_actor": str(inherited_actor),
            "evidence_paths": ["report.csv"],
        },
    )
    assert load_run(child).selected_actor == inherited_actor
    if missing_export != "present":
        shutil.rmtree(inherited_actor)
        if missing_export == "recoverable":
            (parent_path / "actor").mkdir()
        elif missing_export == "partial":
            (parent_path / "state").mkdir()
        else:
            append_jsonl(
                parent_root / "run_events.jsonl",
                {
                    "event": "checkpoint_payload_pruned",
                    "checkpoint_id": parent["checkpoint_id"],
                },
            )
        with pytest.raises(ValueError, match="missing"):
            load_run(child)
    if missing_export == "partial":
        with pytest.raises(ValueError, match="complete recorded prune"):
            collect_actors([child], output_dir=tmp_path / "library", strict=False)
        assert not (tmp_path / "library").exists()
        return
    catalog = collect_actors(
        [child], output_dir=tmp_path / "library", strict=missing_export == "present"
    )
    rows = json.loads(catalog.read_text())["actors"]
    assert any(
        row["roles"] == ["History"] and row["source_actor_run_id"] == "one-run"
        for row in rows
    )
    selected = next(row for row in rows if row["roles"] == ["Selected", "Inherited"])
    assert selected["availability"].startswith("Unavailable") == (
        missing_export == "pruned"
    )


def test_recollection_preserves_extracted_actor_after_normal_export(
    tmp_path: Path,
) -> None:
    from tests.training.test_training_progress import _actor

    root = tmp_path / "run"
    details, original = _host_run(root)
    shutil.rmtree(original)
    (root / "checkpoints" / details["checkpoint_id"] / "actor").mkdir()
    catalog = collect_actors([root], output_dir=tmp_path / "library")
    before = json.loads(catalog.read_text())["actors"][0]
    normal = _actor(root, details)
    normal_id = checkpoints.artifact_identity(normal)["checkpoint_id"]
    assert normal_id != before["actor_export_id"]
    collect_actors([root], output_dir=tmp_path / "library")
    after = json.loads(catalog.read_text())["actors"][0]
    assert after["actor_export_id"] == before["actor_export_id"]
    assert after["source_artifact_id"] == normal_id
    changed = json.loads((normal / "actor_details.json").read_text())
    changed["actor_digest"] = "c" * 64
    (normal / "actor_details.json").write_text(json.dumps(_signed(changed)))
    with pytest.raises(ValueError, match="originating checkpoint"):
        collect_actors([root], output_dir=tmp_path / "library")


@pytest.mark.parametrize("fault", ["wrong_digest", "orphan_selection"])
def test_saved_result_rejects_conflicting_actor_or_selection(
    tmp_path: Path, fault: str
) -> None:
    from tests.training.test_training_progress import _actor, _description, _saved_host

    from marl_battlegrounds.training import load_run
    from marl_battlegrounds.training._run_io import atomic_json

    root = tmp_path / "run"
    details, actor = _host_run(root)
    selected = None
    if fault == "wrong_digest":
        changed = json.loads((actor / "actor_details.json").read_text())
        changed["actor_digest"] = "c" * 64
        atomic_json(actor / "actor_details.json", _signed(changed))
    else:
        orphan = _description(root, _saved_host(steps=8, pending=None), parent=details)
        selected = str(_actor(root, orphan))
        atomic_json(root / "selection.json", {"checkpoint_id": orphan["checkpoint_id"]})
    (root / "report.csv").write_text("value\n1\n")
    atomic_json(
        root / "status.json",
        {
            "run_id": "one-run",
            "status": "complete",
            "env_steps": 0,
            "completed_updates": 0,
            "final_actor": str(actor),
            "selected_actor": selected,
            "evidence_paths": ["report.csv"],
        },
    )
    for operation in (
        lambda: load_run(root),
        lambda: collect_actors([root], output_dir=tmp_path / "library"),
    ):
        with pytest.raises(ValueError, match=r"originating checkpoint|outside active"):
            operation()


def test_malformed_study_status_has_readable_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from marl_battlegrounds import _cli
    from marl_battlegrounds.training._run_io import atomic_json

    atomic_json(tmp_path / "declaration.json", {"cases": []})
    atomic_json(tmp_path / "study.json", [])
    rows = list_runs(tmp_path)
    assert rows[0]["status"] == "unreadable" and "objects" in str(rows[0]["reason"])
    assert _cli.main(["runs", str(tmp_path)]) == 0
    assert "objects" in capsys.readouterr().out


@pytest.mark.parametrize(
    "fault",
    [
        "missing_selected",
        "missing_selection",
        "wrong_selected",
        "missing_final",
        "wrong_final",
    ],
)
def test_complete_result_actor_bindings_agree_between_readers(
    tmp_path: Path, fault: str
) -> None:
    from marl_battlegrounds.training import load_run
    from marl_battlegrounds.training._run_io import atomic_json

    root = tmp_path / "run"
    details, actor = _host_run(root)
    selection = {"checkpoint_id": details["checkpoint_id"]}
    status: dict[str, Any] = {
        "run_id": "one-run",
        "status": "complete",
        "env_steps": 0,
        "completed_updates": 0,
        "final_actor": str(actor),
        "selected_actor": str(actor),
        "evidence_paths": ["report.csv"],
    }
    if fault == "missing_selected":
        status["selected_actor"] = None
    elif fault == "missing_selection":
        selection = None
    elif fault == "wrong_selected":
        status["selected_actor"] = str(tmp_path / "absent-selected")
    elif fault == "missing_final":
        status["final_actor"] = None
    else:
        status["final_actor"] = str(tmp_path / "absent-final")
    (root / "report.csv").write_text("value\n1\n")
    atomic_json(root / "selection.json", selection)
    atomic_json(root / "status.json", status)
    for operation in (
        lambda: load_run(root),
        lambda: collect_actors([root], output_dir=tmp_path / "library"),
        lambda: collect_actors([root], output_dir=tmp_path / "library", strict=False),
    ):
        with pytest.raises((ValueError, OSError)):
            operation()
    assert not (tmp_path / "library").exists()


@pytest.mark.parametrize("strict", [True, False])
def test_missing_inherited_catalog_row_is_never_a_missing_payload(
    tmp_path: Path, strict: bool
) -> None:
    from tests.training.test_training_progress import _actor, _description

    parent_root = tmp_path / "parent"
    parent = _history_run(parent_root, tmp_path / "history", retain=True)
    child = tmp_path / "child"
    child.mkdir()
    host = copy.deepcopy(parent["metadata"]["host_state"])
    host["past_copies"] = {}
    details = _description(child, host)
    details["collection"] = parent["collection"]
    details["metadata"]["run_id"] = "child"
    details["metadata"]["continuation"] = {
        "parent_checkpoint": str(parent_root / "checkpoints" / parent["checkpoint_id"]),
        "parent_checkpoint_id": parent["checkpoint_id"],
    }
    details = _revise_learner(child, details)
    _actor(child, details)
    with pytest.raises(
        ValueError, match="inherited history actor is missing from the catalog"
    ):
        collect_actors([child], output_dir=tmp_path / "library", strict=strict)
    assert not (tmp_path / "library").exists()


def test_interrupted_collection_reuses_published_extraction_and_keeps_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.training.test_training_progress import _actor

    from marl_battlegrounds.training._run_io import append_jsonl

    root, library = tmp_path / "run", tmp_path / "library"
    details, original = _host_run(root)
    shutil.rmtree(original)
    payload = root / "checkpoints" / details["checkpoint_id"] / "actor"
    payload.mkdir()
    write = checkpoints._atomic_json

    def interrupt(path: Path, value: object) -> None:
        if path.name == "catalog.json":
            raise OSError("Injected catalog interruption")
        write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_atomic_json", interrupt)
        with pytest.raises(OSError, match="catalog interruption"):
            collect_actors([root], output_dir=library)
    assert not (library / "catalog.json").exists()
    kept = next((library / "models").iterdir())
    original_id = checkpoints.artifact_identity(kept)["checkpoint_id"]
    normal = _actor(root, details)
    normal_id = checkpoints.artifact_identity(normal)["checkpoint_id"]
    assert original_id != normal_id
    catalog = collect_actors([root], output_dir=library)
    row = json.loads(catalog.read_text())["actors"][0]
    assert row["actor_export_id"] == original_id
    assert row["source_artifact_id"] == normal_id
    shutil.rmtree(normal)
    shutil.rmtree(payload)
    append_jsonl(
        root / "run_events.jsonl",
        {
            "event": "checkpoint_payload_pruned",
            "checkpoint_id": details["checkpoint_id"],
        },
    )
    collect_actors([root], output_dir=library)
    after = json.loads(catalog.read_text())["actors"][0]
    assert after["source_artifact_id"] == normal_id
    assert after["actor_export_id"] == original_id
    assert after["availability"] == "Available"
    assert after["source_availability"].startswith("Unavailable")


@pytest.mark.parametrize("selected", [False, True])
def test_relaxed_complete_collection_extracts_missing_admitted_export(
    tmp_path: Path, selected: bool
) -> None:
    from marl_battlegrounds.training import load_run
    from marl_battlegrounds.training._run_io import atomic_json

    root = tmp_path / "run"
    details, actor = _host_run(root)
    (root / "report.csv").write_text("value\n1\n")
    atomic_json(
        root / "selection.json",
        {"checkpoint_id": details["checkpoint_id"]} if selected else None,
    )
    atomic_json(
        root / "status.json",
        {
            "run_id": "one-run",
            "status": "complete",
            "env_steps": 0,
            "completed_updates": 0,
            "final_actor": str(actor),
            "selected_actor": str(actor) if selected else None,
            "evidence_paths": ["report.csv"],
        },
    )
    shutil.rmtree(actor)
    (root / "checkpoints" / details["checkpoint_id"] / "actor").mkdir()
    with pytest.raises(ValueError, match="missing"):
        load_run(root)
    with pytest.raises(ValueError, match="missing"):
        collect_actors([root], output_dir=tmp_path / "strict")
    catalog = collect_actors([root], output_dir=tmp_path / "relaxed", strict=False)
    row = json.loads(catalog.read_text())["actors"][0]
    assert row["source_checkpoint_id"] == details["checkpoint_id"]
    assert row["availability"] == "Available" and "Final" in row["roles"]
    assert ("Selected" in row["roles"]) == selected
    assert not actor.exists()  # Salvage writes only the separate library.
