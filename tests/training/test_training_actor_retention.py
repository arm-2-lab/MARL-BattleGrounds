"""Check opt-in actor retention without keeping every full recovery payload.

Public setup creates real PPO, QMIX and PQN learners. Repeated stable saves
exercise publication, export, pruning and independent actor loading, without
learning updates. Export failure leaves the current full checkpoint usable;
public resume completes that export even when no validation task is pending.
CPU byte/time records describe storage work, not learning or GPU throughput.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import json
import time
from pathlib import Path
from typing import Any, Literal

import pytest
from tests.training_continuation_helpers import fixed_source

from marl_battlegrounds.baselines.config import PPOConfig, PQNConfig, QMIXConfig
from marl_battlegrounds.training import _run_io, checkpoints, runner


@pytest.mark.parametrize("method", ("mappo", "qmix", "pqn_vdn"))
def test_all_saved_actors_survive_pruning_and_export_failure_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: Literal["mappo", "qmix", "pqn_vdn"],
) -> None:
    fixed_source(monkeypatch, method)
    settings: dict[str, Any] = {"ppo": PPOConfig(rollout_length=2, epochs=1)}
    if method == "qmix":
        settings = {
            "qmix": QMIXConfig(
                rollout_length=2,
                buffer_size=4,
                min_buffer_size=2,
                sample_sequence_length=2,
                sample_batch_size=2,
                epochs=1,
                update_period=1,
            )
        }
    elif method == "pqn_vdn":
        settings = {
            "pqn": PQNConfig(
                rollout_length=2, memory_window=1, epochs=1, num_minibatches=2
            )
        }
    config = runner.TrainConfig(
        method=method,
        num_envs=4,
        total_env_steps=32,
        keep_past=0,
        metrics="none",
        verbose=False,
        retain_all_actors=True,
        **settings,
    )
    owners: list[runner._Run] = []
    paths: list[Path] = []
    samples: list[dict[str, Any]] = []
    exporting = checkpoints.export_system
    pending_bytes: dict[str, list[int]] = {"checkpoints": [], "actors": []}
    rename = Path.rename

    def measure_pending(path: Path, target: str | Path) -> Path:
        if path.name.startswith(".pending-") and path.parent.name in pending_bytes:
            pending_bytes[path.parent.name].append(
                sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            )
            if path.parent.name == "actors":
                assert not (path / "state").exists()
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", measure_pending)

    class SetupReachedError(Exception):
        pass

    def fail_export(*args: object, **kwargs: object) -> Path:
        raise OSError("Injected actor export failure")

    def exercise_storage(owner: runner._Run) -> Any:  # noqa: ANN401
        owners.append(owner)
        if owner.checkpoint is not None:
            assert owner.host["pending"] is None
            owner.finish_pending()
            assert owner.checkpoint_id in owner.host["actors"]
            raise SetupReachedError
        for _ in range(4):
            started = time.perf_counter()
            paths.append(owner.save())
            samples.append({"save_and_export_seconds": time.perf_counter() - started})
        assert len(owner.host["actors"]) == 4
        assert sum((path / "state").exists() for path in paths) == 2
        for path in paths:
            description = checkpoints.read_checkpoint_description(path)
            exported = Path(owner.host["actors"][description["checkpoint_id"]])
            system = checkpoints.load_system(exported)
            identity = checkpoints.artifact_identity(exported)
            assert identity["weight_digest"] == description["actor_digest"]
            if method != "mappo":
                assert float(system.variables.epsilon) == 0.0
            assert (path / "checkpoint_details.json").is_file()
        monkeypatch.setattr(checkpoints, "export_system", fail_export)
        owner.save()

    monkeypatch.setattr(runner._Run, "execute", exercise_storage)
    root = tmp_path / "run"
    with pytest.raises(OSError, match="Injected actor export"):
        runner.train(config, output_dir=root)
    latest = (
        root
        / json.loads((root / "latest_checkpoint.json").read_text())["relative_path"]
    )
    assert latest not in paths and (latest / "state").is_dir()
    monkeypatch.setattr(checkpoints, "export_system", exporting)
    with pytest.raises(SetupReachedError):
        runner.train(resume_from=latest)
    assert len(owners[-1].host["actors"]) == 5
    assert sum((path / "state").exists() for path in [*paths, latest]) == 2
    actors = [Path(value) for value in owners[-1].host["actors"].values()]
    for actor in actors:
        checkpoints.load_system(actor)
    # Public recovery refuses a missing earlier export before logs are changed.
    first_actor = actors[0]
    hidden = first_actor.with_name(first_actor.name + ".saved")
    first_actor.rename(hidden)
    events = (root / "run_events.jsonl").read_bytes()
    updates = (root / "training_updates.jsonl").read_bytes()
    try:
        with pytest.raises((ValueError, FileNotFoundError), match=r"actor|checkpoint"):
            runner.train(resume_from=latest)
        assert (root / "run_events.jsonl").read_bytes() == events
        assert (root / "training_updates.jsonl").read_bytes() == updates
    finally:
        hidden.rename(first_actor)
    details = copy.deepcopy(checkpoints.read_checkpoint_description(latest))
    details["metadata"]["host_state"]["actors"].pop(
        checkpoints.read_checkpoint_description(paths[0])["checkpoint_id"]
    )
    with pytest.raises(ValueError, match="missing from the catalog"):
        _run_io.validate_host_state(root, details, panel=None, history_capture_count=0)
    record = {
        "method": method,
        "scope": (
            "CPU initialized learner, zero transitions; four save/export/prune "
            "calls plus one failed export and public recovery"
        ),
        "samples": samples,
        "largest_pending_checkpoint_bytes": max(pending_bytes["checkpoints"]),
        "largest_pending_actor_bytes": max(pending_bytes["actors"]),
        "pending_measurement": (
            "Complete temporary tree at publication; no state subtree in actor exports"
        ),
        "actor_bytes": sum(
            p.stat().st_size
            for actor in actors
            for p in actor.rglob("*")
            if p.is_file()
        ),
        "recovery_bytes": sum(
            p.stat().st_size for p in (root / "checkpoints").rglob("*") if p.is_file()
        ),
    }
    (tmp_path / "retention-cost.json").write_text(json.dumps(record, indent=2))


@pytest.mark.parametrize(
    "method", ("mappo", "ff_mappo", "ippo", "ff_ippo", "qmix", "pqn_vdn")
)
def test_retention_config_round_trip_and_continuation_contract(method: str) -> None:
    from marl_battlegrounds.training._checkpoint_metadata import saved_training_config
    from marl_battlegrounds.training._continuation_schedules import (
        CONTINUATION_CHANGE_KEYS,
    )
    from marl_battlegrounds.training.config import config_from_dict, config_to_dict

    config = config_from_dict({"method": method, "retain_all_actors": True})
    assert config_from_dict(config_to_dict(config)) == config
    assert "retain_all_actors" in CONTINUATION_CHANGE_KEYS
    old = config_to_dict(config)
    old.pop("retain_all_actors")
    assert (
        saved_training_config({"metadata": {"config": old}})["retain_all_actors"]
        is False
    )
    for invalid in (1, "true", None):
        with pytest.raises((TypeError, ValueError), match="retain_all_actors"):
            config_from_dict({"method": method, "retain_all_actors": invalid})


def test_history_retention_begins_at_declared_capture_and_preserves_parent_promise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.training.test_training_progress import _description, _saved_host

    parent = _description(tmp_path, _saved_host(steps=16, pending=None))
    parent["collection"]["history_capture_capacity"] = 4
    parent["metadata"]["config"]["retain_all_actors"] = True
    host = parent["metadata"]["host_state"]
    host.update(retention_start_capture_id=1, past_copies={})
    for capture in range(2):
        host["past_copies"][str(capture)] = {
            "env_steps": 8 * (capture + 1),
            "path": str(tmp_path / f"missing-{capture}"),
            "sampled_epsilon": None,
        }
    with pytest.raises(ValueError, match="missing-1"):
        _run_io._check_past_copies(tmp_path, parent, capture_count=2)
    # Enabling future retention does not promise older unavailable captures.
    missing = copy.deepcopy(parent)
    missing["metadata"]["host_state"].pop("past_copies")
    with pytest.raises(ValueError, match="missing from the catalog"):
        _run_io._check_past_copies(tmp_path, missing, capture_count=2)
    host["retention_start_capture_id"] = 2
    _run_io._check_past_copies(tmp_path, parent, capture_count=2)
    for invalid in (-1, True, 3, None):
        host["retention_start_capture_id"] = invalid
        with pytest.raises(ValueError, match="starting capture"):
            _run_io._check_past_copies(tmp_path, parent, capture_count=2)
    host["retention_start_capture_id"] = 1
    child = copy.deepcopy(parent)
    child["metadata"]["run_id"] = "child-run"
    child["metadata"]["config"]["retain_all_actors"] = False
    child["metadata"]["continuation"] = {
        "parent_checkpoint": str(tmp_path / "parent"),
        "parent_checkpoint_id": parent["checkpoint_id"],
    }

    def parent_description(_path: object) -> dict[str, Any]:
        return parent

    monkeypatch.setattr(checkpoints, "read_checkpoint_description", parent_description)
    with pytest.raises(ValueError, match="missing-1"):
        _run_io._check_past_copies(tmp_path, child, capture_count=2)
    child["metadata"]["host_state"].pop("past_copies")
    with pytest.raises(ValueError, match=r"inherited history actor.*catalog"):
        _run_io._check_past_copies(tmp_path, child, capture_count=2)


def test_public_child_enables_future_retention_and_resumes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_source(monkeypatch, "mappo")
    owners: list[runner._Run] = []
    exports: list[Path] = []
    exporting = checkpoints.export_system

    class SetupReachedError(Exception):
        pass

    def counted_export(*args: Any, **kwargs: Any) -> Path:  # noqa: ANN401
        result = exporting(*args, **kwargs)
        exports.append(result)
        return result

    def stop_after_save(owner: runner._Run) -> Any:  # noqa: ANN401
        owners.append(owner)
        if owner.checkpoint is None:
            owner.save()
        owner.finish_pending()
        if owner.config.retain_all_actors and owner.metadata.get("continuation"):
            owner.save()
            owner.finish_pending()
        assert len(owner.host["recovery_checkpoints"]) <= 2
        raise SetupReachedError

    monkeypatch.setattr(checkpoints, "export_system", counted_export)
    monkeypatch.setattr(runner._Run, "execute", stop_after_save)
    config = runner.TrainConfig(
        num_envs=4,
        total_env_steps=32,
        keep_past=0,
        metrics="none",
        verbose=False,
        ppo=PPOConfig(rollout_length=2, epochs=1),
    )
    with pytest.raises(SetupReachedError):
        runner.train(config, output_dir=tmp_path / "parent")
    assert not exports
    parent = owners[-1].checkpoint
    assert parent is not None
    parent_description = checkpoints.read_checkpoint_description(parent)
    assert parent_description["metadata"]["config"]["retain_all_actors"] is False
    with pytest.raises(SetupReachedError):
        runner.extend_training(
            parent,
            additional_env_steps=8,
            output_dir=tmp_path / "child",
            changes={"retain_all_actors": True},
        )
    assert len(exports) == 2
    child = owners[-1].checkpoint
    assert child is not None
    assert owners[-1].host["retention_start_capture_id"] == 0
    assert checkpoints.read_checkpoint_description(parent) == parent_description
    for _ in range(3):
        previous_exports = len(exports)
        with pytest.raises(SetupReachedError):
            runner.train(resume_from=child)
        assert owners[-1].config.retain_all_actors is True
        assert owners[-1].checkpoint_id in owners[-1].host["actors"]
        # Reuse the current export, then save exactly one new boundary.
        assert len(exports) == previous_exports + 2
        assert exports[previous_exports] == exports[previous_exports - 1]
        child = owners[-1].checkpoint
        assert child is not None
        assert len(list(child.parent.glob("*/state"))) == 2


def test_inherited_capture_keeps_known_identity_before_payload_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.training.test_training_progress import _description, _saved_host

    parent = _description(tmp_path, _saved_host(steps=16, pending=None))
    parent["collection"]["history_capture_capacity"] = 2
    parent["metadata"]["config"]["retain_all_actors"] = True
    parent["metadata"]["host_state"].update(
        retention_start_capture_id=0,
        past_copies={
            "0": {
                "env_steps": 8,
                "path": str(tmp_path / "old"),
                "sampled_epsilon": None,
                "artifact_id": "a" * 64,
            }
        },
    )
    child = copy.deepcopy(parent)
    child["metadata"]["run_id"] = "child"
    child["metadata"]["config"]["retain_all_actors"] = False
    child["metadata"]["continuation"] = {
        "parent_checkpoint": str(tmp_path / "parent"),
        "parent_checkpoint_id": parent["checkpoint_id"],
    }
    moved = tmp_path / "relocated"
    moved.mkdir()
    entry = child["metadata"]["host_state"]["past_copies"]["0"]
    entry["path"] = str(moved)

    def parent_description(_path: object) -> dict[str, Any]:
        return parent

    class PayloadCheckReachedError(Exception):
        pass

    def payload_check(_path: object) -> dict[str, Any]:
        raise PayloadCheckReachedError

    monkeypatch.setattr(checkpoints, "read_checkpoint_description", parent_description)
    monkeypatch.setattr(checkpoints, "artifact_identity", payload_check)
    for changed in ("b" * 64, None):
        if changed is None:
            entry.pop("artifact_id")
        else:
            entry["artifact_id"] = changed
        with pytest.raises(ValueError, match="inherited capture identity"):
            _run_io._check_past_copies(tmp_path, child, capture_count=1)
    entry["artifact_id"] = "a" * 64
    with pytest.raises(PayloadCheckReachedError):
        _run_io._check_past_copies(tmp_path, child, capture_count=1)
    for field, changed in (("env_steps", 16), ("sampled_epsilon", 0.5)):
        before = entry[field]
        entry[field] = changed
        with pytest.raises(ValueError, match="inherited capture identity"):
            _run_io._check_past_copies(tmp_path, child, capture_count=1)
        entry[field] = before
    # Unknown legacy IDs may become checked, but known IDs may not disappear.
    parent["metadata"]["host_state"]["past_copies"]["0"].pop("artifact_id")
    with pytest.raises(PayloadCheckReachedError):
        _run_io._check_past_copies(tmp_path, child, capture_count=1)
