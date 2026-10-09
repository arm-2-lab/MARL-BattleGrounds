"""Check owned temporary writes and conservative recovery after interrupted saves.

Shared JSON/tree failure injection protects old pointers and published artifacts.
A real child-process crash leaves an ownership receipt that recovery can check;
live, unknown and malformed ownership never authorizes deleting another write.
These host/filesystem checks make no learning or GPU performance claim.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from marl_battlegrounds.training import _run_io, checkpoints

type Tree = Any

# These tests deliberately exercise the private persistence owner.
# pyright: reportPrivateUsage=false


@pytest.mark.parametrize(
    "phase", ("serialization", "file_sync", "replace", "directory_sync")
)
def test_json_failure_cleans_own_file_and_preserves_complete_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    target = tmp_path / "pointer.json"
    checkpoints._atomic_json(target, {"old": True})
    previous = target.read_bytes()
    foreign = tmp_path / ".someone-else.tmp"
    foreign.write_bytes(b"keep")

    def fail(*args: object) -> None:
        raise OSError("Injected JSON publication failure")

    value: object = {"new": True}
    if phase == "serialization":
        value = {"invalid": float("nan")}
    elif phase == "file_sync":
        monkeypatch.setattr(checkpoints.os, "fsync", fail)
    elif phase == "replace":
        monkeypatch.setattr(checkpoints.os, "replace", fail)
    else:
        monkeypatch.setattr(checkpoints, "_sync_dir", fail)
    with pytest.raises((ValueError, OSError)):
        checkpoints._atomic_json(target, value)
    assert target.read_bytes() == (
        checkpoints._json_bytes(value) if phase == "directory_sync" else previous
    )
    assert set(tmp_path.iterdir()) == {target, foreign}
    assert foreign.read_bytes() == b"keep"


@pytest.mark.parametrize("published", (False, True))
def test_pending_context_cleans_only_unpublished_tree(
    tmp_path: Path, published: bool
) -> None:
    final = tmp_path / "actor"
    with (
        pytest.raises(OSError, match="Injected"),
        checkpoints._pending_write(
            tmp_path, kind="actor", attempt_id="test"
        ) as temporary,
    ):
        (temporary / "payload").write_bytes(b"complete bytes")
        inventory = checkpoints._inventory(temporary)
        assert list(inventory) == ["payload"]
        assert len(list(tmp_path.glob("*.owner.json"))) == 1
        if published:
            temporary.rename(final)
        raise OSError("Injected failure")
    assert list(tmp_path.iterdir()) == ([final] if published else [])
    if published:
        assert (final / "payload").read_bytes() == b"complete bytes"


def _abandoned(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    temporary = root / f".pending-{uuid4().hex}"
    temporary.mkdir()
    (temporary / "payload").write_bytes(b"partial")
    receipt = root / f"{temporary.name}.owner.json"
    record = {
        "schema_version": 1,
        "temporary_name": temporary.name,
        "kind": "learner",
        "attempt_id": "failed",
        "writer": _run_io.process_identity(),
    }
    checkpoints._atomic_json(receipt, record)
    return temporary, receipt, record


@pytest.mark.parametrize(
    "condition", ("live", "unknown", "reused_pid", "previous_boot", "gone")
)
def test_abandoned_write_needs_proven_original_writer_death(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
) -> None:
    temporary, receipt, record = _abandoned(tmp_path)
    writer = record["writer"]
    current = dict(writer)
    if condition == "reused_pid":
        current["start_ticks"] = str(int(writer["start_ticks"]) + 1)
    elif condition == "previous_boot":
        current["boot_id"] = str(uuid4())
    elif condition in ("unknown", "gone"):
        current.update(start_ticks=None, boot_id=None)

    def identity(pid: int | None = None) -> dict[str, Any]:
        assert pid == writer["pid"]
        return current

    def existence(pid: int, signal: int) -> None:
        assert pid == writer["pid"] and signal == 0
        if condition == "gone":
            raise ProcessLookupError("Original process exited")
        raise PermissionError("Cannot establish process identity")

    monkeypatch.setattr(_run_io, "process_identity", identity)
    monkeypatch.setattr(checkpoints.os, "kill", existence)
    with _run_io.run_lock(tmp_path):
        records = list(checkpoints._cleanup_pending_writes(tmp_path))
    dead = condition in ("reused_pid", "previous_boot", "gone")
    assert temporary.exists() != dead and receipt.exists() != dead
    assert records[0]["status"] == ("removed" if dead else "preserved")
    assert records[0]["reason"]


@pytest.mark.parametrize(
    "fault",
    ("missing", "malformed", "outside", "receipt_link", "payload_link", "unknown_name"),
)
def test_unknown_pending_data_and_published_artifacts_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    temporary, receipt, record = _abandoned(tmp_path)
    published = tmp_path / "published-actor"
    published.mkdir()
    (published / "payload").write_bytes(b"published")
    if fault == "missing":
        receipt.unlink()
    elif fault == "malformed":
        receipt.write_text('{"partial":')
    elif fault == "outside":
        record["temporary_name"] = "../published-actor"
        checkpoints._atomic_json(receipt, record)
    elif fault == "receipt_link":
        receipt.rename(tmp_path / "foreign.json")
        receipt.symlink_to(tmp_path / "foreign.json")
    elif fault == "payload_link":
        (temporary / "payload").unlink()
        temporary.rmdir()
        temporary.symlink_to(published, target_is_directory=True)
    else:
        temporary.rename(tmp_path / ".pending-unrecognized")
        receipt.unlink()
        temporary = tmp_path / ".pending-unrecognized"

    def dead(writer: object) -> str:
        return "dead"

    monkeypatch.setattr(checkpoints, "_pending_writer_status", dead)
    before = (published / "payload").read_bytes()
    with _run_io.run_lock(tmp_path):
        records = list(checkpoints._cleanup_pending_writes(tmp_path))
    assert records and all(row["status"] == "preserved" for row in records)
    assert temporary.exists()
    assert (published / "payload").read_bytes() == before


def test_actual_crashed_writer_is_cleaned_without_touching_run_events(
    tmp_path: Path,
) -> None:
    events = tmp_path / "run_events.jsonl"
    events.write_bytes(b'{"event":"uncommitted_exposure"}\n')
    script = """
import os, sys
from pathlib import Path
from marl_battlegrounds.training.checkpoints import _pending_write
with _pending_write(Path(sys.argv[1]), kind="actor", attempt_id="crashed") as path:
    (path / "partial").write_bytes(b"unfinished")
    os._exit(19)
"""
    child = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path)],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 19, child.stderr
    assert len(list(tmp_path.glob(".pending-*"))) == 2
    with _run_io.run_lock(tmp_path):
        records = list(checkpoints._cleanup_pending_writes(tmp_path))
    assert records[0]["status"] == "removed"
    assert not list(tmp_path.glob(".pending-*"))
    assert events.read_bytes() == b'{"event":"uncommitted_exposure"}\n'


def test_cleanup_failure_reports_remaining_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary, receipt, _ = _abandoned(tmp_path)

    def dead(writer: object) -> str:
        return "dead"

    def fail(path: Path) -> None:
        raise PermissionError("Cannot remove payload")

    monkeypatch.setattr(checkpoints, "_pending_writer_status", dead)
    monkeypatch.setattr(checkpoints.shutil, "rmtree", fail)
    with _run_io.run_lock(tmp_path), pytest.raises(OSError, match=str(temporary)):
        list(checkpoints._cleanup_pending_writes(tmp_path))
    assert temporary.is_dir() and receipt.is_file()


@pytest.fixture(scope="module", params=("mappo", "qmix", "pqn_vdn"))
def saved_family(
    request: pytest.FixtureRequest,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    method = str(request.param)
    if method == "mappo":
        from tests.training.test_training_checkpoints import PPO, _metadata

        from marl_battlegrounds.training import (
            make_training_schedule,
            prepare_training_content,
        )
        from marl_battlegrounds.training.learner import init_learner

        collection, state = init_learner(
            schedule=make_training_schedule(total_env_steps=20, num_envs=4),
            seed=42,
            ppo=PPO,
            prepared=prepare_training_content(),
            metrics="none",
            recording=False,
        )
        return collection, state, _metadata(), {"method": method, "ppo": PPO}
    if method == "qmix":
        from tests.training.test_training_qmix_checkpoints import (
            _CONFIG,
            _learner,
            _metadata,
        )

        collection, state = _learner()
        return collection, state, _metadata(), {"method": method, "qmix": _CONFIG}
    from tests.training.test_training_pqn_checkpoints import (
        _CONFIG,
        _learner,
        _metadata,
    )

    collection, state = _learner()
    return collection, state, _metadata(), {"method": method, "pqn": _CONFIG}


def _fail_publication(monkeypatch: pytest.MonkeyPatch, phase: str, root: Path) -> None:
    arrays, atomic = checkpoints._save_arrays, checkpoints._atomic_json
    sync, tree, rename = checkpoints._sync_dir, checkpoints._sync_tree, Path.rename
    renamed = False
    pointer_replaced = False

    def save_arrays(path: Path, value: Tree) -> None:
        if phase == path.name + "_array":
            path.mkdir()
            (path / "partial").write_bytes(b"unfinished")
            raise OSError("Injected array failure")
        arrays(path, value)

    def write_json(path: Path, value: object) -> None:
        nonlocal pointer_replaced
        if phase == "description" and path.name in (
            "checkpoint_details.json",
            "actor_details.json",
        ):
            raise OSError("Injected description failure")
        if phase == "pointer" and path.name == "latest_checkpoint.json":
            raise OSError("Injected pointer failure")
        atomic(path, value)
        if path.name == "latest_checkpoint.json":
            pointer_replaced = True

    def sync_tree(path: Path) -> None:
        if phase == "tree_sync":
            raise OSError("Injected tree sync failure")
        tree(path)

    def publish(path: Path, destination: str | Path) -> Path:
        nonlocal renamed
        if path.name.startswith(".pending-"):
            if phase == "rename":
                raise OSError("Injected rename failure")
            result = rename(path, destination)
            renamed = True
            return result
        return rename(path, destination)

    def sync_dir(path: Path) -> None:
        nonlocal renamed, pointer_replaced
        # Fail once so the finally cleanup can itself finish and be checked.
        if phase == "published_sync" and renamed:
            renamed = False
            raise OSError("Injected published directory sync failure")
        if phase == "pointer_sync" and pointer_replaced and path == root:
            pointer_replaced = False
            raise OSError("Injected pointer directory sync failure")
        sync(path)

    monkeypatch.setattr(checkpoints, "_save_arrays", save_arrays)
    monkeypatch.setattr(checkpoints, "_atomic_json", write_json)
    monkeypatch.setattr(checkpoints, "_sync_tree", sync_tree)
    monkeypatch.setattr(checkpoints, "_sync_dir", sync_dir)
    monkeypatch.setattr(Path, "rename", publish)


def test_real_learner_publication_failures_preserve_recovery_and_retry(
    tmp_path: Path,
    saved_family: tuple[Any, Any, dict[str, Any], dict[str, Any]],
) -> None:
    collection, state, metadata, settings = saved_family
    baseline = checkpoints.save_checkpoint(
        tmp_path, collection, state, metadata=metadata, **settings
    )
    original = checkpoints.read_checkpoint_details(baseline)
    for phase in (
        "actor_array",
        "state_array",
        "description",
        "tree_sync",
        "rename",
        "published_sync",
        "pointer",
        "pointer_sync",
    ):
        pointer = tmp_path / "latest_checkpoint.json"
        previous = pointer.read_bytes()
        before = set((tmp_path / "checkpoints").iterdir())
        candidate: dict[str, object] = {**metadata, "attempt_id": phase}
        with pytest.MonkeyPatch.context() as patch:
            _fail_publication(patch, phase, tmp_path)
            with pytest.raises(OSError, match="Injected"):
                checkpoints.save_checkpoint(
                    tmp_path, collection, state, metadata=candidate, **settings
                )
        assert not list(tmp_path.rglob(".pending-*"))
        assert not list(tmp_path.rglob("*.tmp"))
        published = set((tmp_path / "checkpoints").iterdir()) - before
        after_rename = phase in ("published_sync", "pointer", "pointer_sync")
        assert len(published) == int(after_rename)
        assert (pointer.read_bytes() == previous) != (phase == "pointer_sync")
        if after_rename:
            checkpoints.read_checkpoint_details(next(iter(published)))
        retried = checkpoints.save_checkpoint(
            tmp_path, collection, state, metadata=candidate, **settings
        )
        # Orbax uses fresh physical filenames and timestamps for each save.
        # Retry must recover the same logical boundary, without deleting the
        # earlier published artifact or changing its immutable identity.
        checked = checkpoints.read_checkpoint_details(retried)
        assert checked["metadata"] == candidate
        assert checked["actor_digest"] == original["actor_digest"]
        assert checked["counters"] == original["counters"]
        if after_rename:
            assert next(iter(published)).is_dir()
        assert tmp_path / json.loads(pointer.read_text())["relative_path"] == retried
    assert checkpoints.read_checkpoint_details(baseline) == original


def test_real_actor_export_failures_preserve_published_payload_and_identity(
    tmp_path: Path,
    saved_family: tuple[Any, Any, dict[str, Any], dict[str, Any]],
) -> None:
    _, state, metadata, settings = saved_family
    method = settings["method"]
    config = settings[
        "ppo" if method == "mappo" else "qmix" if method == "qmix" else "pqn"
    ]
    variables = state.carry.history.current_variables
    actor = (
        variables
        if method == "mappo"
        else variables.params
        if method == "qmix"
        else variables.network
    )
    provenance = {
        "run_id": metadata["run_id"],
        "seed": metadata["config"]["seed"],
        "env_steps": 0,
        "checkpoint_id": "0" * 64,
    }
    if method != "mappo":
        provenance["optimizer_steps"] = 0
    for phase in (
        "actor_array",
        "description",
        "tree_sync",
        "rename",
        "published_sync",
    ):
        destination = tmp_path / phase
        kwargs: dict[str, Any] = dict(
            metadata=provenance,
            input_scale=config.input_scale,
            spawn_frame=config.spawn_frame,
            method=method,
        )
        with pytest.MonkeyPatch.context() as patch:
            _fail_publication(patch, phase, tmp_path)
            with pytest.raises(OSError, match="Injected"):
                checkpoints.export_system(actor, destination, **kwargs)
        assert not list(tmp_path.glob(".pending-*"))
        assert destination.exists() == (phase == "published_sync")
        if destination.exists():
            before = checkpoints.read_checkpoint_details(destination)
        else:
            before = None
        result = checkpoints.export_system(actor, destination, **kwargs)
        assert result == destination
        saved = checkpoints.read_checkpoint_details(destination)
        if before is not None:
            assert saved == before
        loaded = checkpoints.load_system(destination)
        assert loaded is not None


@pytest.mark.parametrize("kind", ("json", "tree", "receipt"))
def test_name_collision_never_claims_or_deletes_foreign_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    token = uuid4()
    monkeypatch.setattr(checkpoints, "uuid4", lambda: token)
    if kind == "json":
        target = tmp_path / "pointer.json"
        foreign = tmp_path / f".pointer.json.{token.hex}.tmp"
        foreign.write_bytes(b"Foreign file")
        with pytest.raises(FileExistsError):
            checkpoints._atomic_json(target, {"value": 1})
        assert foreign.read_bytes() == b"Foreign file"
        assert not target.exists()
    else:
        temporary = tmp_path / f".pending-{token.hex}"
        receipt = tmp_path / f"{temporary.name}.owner.json"
        receipt.write_bytes(b"Foreign receipt")
        if kind == "tree":
            temporary.mkdir()
            (temporary / "keep").write_bytes(b"Foreign payload")
        with (
            pytest.raises(FileExistsError),
            checkpoints._pending_write(tmp_path, kind="actor"),
        ):
            pytest.fail("Collision must not grant ownership")
        assert receipt.read_bytes() == b"Foreign receipt"
        assert temporary.exists() == (kind == "tree")
        if kind == "tree":
            assert (temporary / "keep").read_bytes() == b"Foreign payload"


def test_cleanup_records_first_removal_before_second_removal_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = sorted((_abandoned(tmp_path)[0], _abandoned(tmp_path)[0]))
    remove = checkpoints.shutil.rmtree

    def fail_second(path: Path) -> None:
        if path == second:
            raise PermissionError("Second removal refused")
        remove(path)

    def dead(writer: object) -> str:
        return "dead"

    monkeypatch.setattr(checkpoints, "_pending_writer_status", dead)
    monkeypatch.setattr(checkpoints.shutil, "rmtree", fail_second)
    log = tmp_path / "run_events.jsonl"
    with _run_io.run_lock(tmp_path), pytest.raises(OSError, match=str(second)):
        for row in checkpoints._cleanup_pending_writes(tmp_path):
            _run_io.append_jsonl(
                log, {"event": "pending_write_cleanup", **row}, durable=True
            )
    saved = _run_io.read_run_events(log)
    assert len(saved) == 1
    assert saved[0]["path"] == str(first) and saved[0]["status"] == "removed"
    assert not first.exists() and second.exists()
