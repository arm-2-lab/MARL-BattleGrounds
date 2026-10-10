"""Check saved history exports and one full learner inventory per recovery.

Real actor files from PPO, QMIX and PQN carry their existing content identities.
Tests reject replacements, missing catalog rows and changed provenance; old
missing exports stay unavailable. Relocated and parent-owned captures remain
usable. Public train/extend setup restores real learners but stops before an
update, isolating recovery scan counts from later saves or evaluation work.
These CPU software checks make no learning or GPU-performance claim.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from tests.training.test_training_checkpoint_writes import saved_family as saved_family

from marl_battlegrounds.training import _run_io, checkpoints


def test_real_history_identity_relocation_and_missing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_family: tuple[Any, Any, dict[str, Any], dict[str, Any]],
) -> None:
    collection, state, metadata, settings = saved_family
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
    epsilon = None if method == "mappo" else float(variables.epsilon)
    destination = tmp_path / "past_copies" / f"{method}_step_8_capture_000000"
    destination.parent.mkdir()
    provenance: dict[str, Any] = {
        "run_id": metadata["run_id"],
        "seed": metadata["config"]["seed"],
        "env_steps": 8,
        "capture_origin": {
            "capture_id": 0,
            "update_index": 0,
            "epsilon": epsilon,
            "content_binding": collection.binding.model_dump(mode="json"),
        },
    }
    if method != "mappo":
        provenance["optimizer_steps"] = 0
    kwargs: dict[str, Any] = dict(
        metadata=provenance,
        method=method,
        input_scale=config.input_scale,
        spawn_frame=config.spawn_frame,
    )
    checkpoints.export_system(actor, destination, **kwargs)
    actor_id = checkpoints.read_checkpoint_description(destination)["checkpoint_id"]
    details: dict[str, Any] = {
        "schemas": checkpoints.checkpoint_schemas(method),
        "counters": {"updates": 0, "learning_blocks": 0, "env_steps": 8},
        "collection": {
            "history_capture_capacity": 2,
            "content_binding": collection.binding.model_dump(mode="json"),
        },
        "metadata": {
            **metadata,
            "host_state": {
                "env_steps": 8,
                "past_copies": {
                    "0": {
                        "env_steps": 8,
                        "path": str(destination),
                        "sampled_epsilon": epsilon,
                        "artifact_id": actor_id,
                    }
                },
            },
        },
    }
    _run_io._check_past_copies(tmp_path, details, capture_count=1)
    inventory = checkpoints._inventory
    measured: list[dict[str, Any]] = []

    def counted_history(path: Path) -> dict[str, object]:
        result = inventory(path)
        if path == destination:
            measured.append(
                {
                    "payload_bytes": sum(
                        int(cast(dict[str, Any], row)["bytes"])
                        for row in result.values()
                    )
                }
            )
        return result

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_inventory", counted_history)
        for _ in range(5):
            start = time.perf_counter()
            _run_io._check_past_copies(tmp_path, details, capture_count=1)
            measured[-1]["seconds"] = time.perf_counter() - start
    assert len(measured) == 5
    (tmp_path / "history-read-cost.json").write_text(
        json.dumps(
            {
                "method": method,
                "scope": (
                    "One present history actor, warm local files, full host "
                    "verification, one inventory per call; linear in available "
                    "catalog payload bytes, no updates or GPU work"
                ),
                "samples": measured,
            },
            indent=2,
        )
    )
    if method != "mappo":
        identity = checkpoints.artifact_identity(destination)
        identity["metadata"]["capture_origin"]["update_index"] = 2
        unequal = copy.deepcopy(details)
        unequal["counters"].update(updates=8, learning_blocks=1)

        def impossible_index(path: str | Path) -> dict[str, Any]:
            assert Path(path) == destination
            return identity

        with monkeypatch.context() as patch:
            patch.setattr(checkpoints, "artifact_identity", impossible_index)
            with pytest.raises(ValueError, match="provenance differs"):
                _run_io._check_past_copies(tmp_path, unequal, capture_count=1)
    # Parent-owned captures keep the parent's seed when a child branches it.
    parent_path = tmp_path / "parent-description"
    parent: dict[str, Any] = {**copy.deepcopy(details), "checkpoint_id": "a" * 64}
    child = copy.deepcopy(details)
    child["metadata"].update(
        run_id="child",
        continuation={
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_id": "a" * 64,
        },
    )
    child["metadata"]["config"]["seed"] += 1
    read_description = checkpoints.read_checkpoint_description

    def parent_description(path: str | Path) -> dict[str, Any]:
        return parent if Path(path) == parent_path else read_description(path)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "read_checkpoint_description", parent_description)
        _run_io._check_past_copies(tmp_path, child, capture_count=1)
        parent["counters"]["env_steps"] = 4
        with pytest.raises(ValueError, match="provenance differs"):
            _run_io._check_past_copies(tmp_path, child, capture_count=1)
    replaced = copy.deepcopy(details)
    replaced["metadata"]["host_state"]["past_copies"]["0"]["artifact_id"] = "f" * 64
    with pytest.raises(ValueError, match=r"identity.*differs"):
        _run_io._check_past_copies(tmp_path, replaced, capture_count=1)
    dropped = copy.deepcopy(details)
    dropped["metadata"]["host_state"]["past_copies"] = {}
    with pytest.raises(ValueError, match="capture IDs"):
        _run_io._check_past_copies(tmp_path, dropped, capture_count=1)
    changed = copy.deepcopy(details)
    changed["metadata"]["host_state"]["past_copies"]["0"]["env_steps"] = 4
    with pytest.raises(ValueError, match="provenance differs"):
        _run_io._check_past_copies(tmp_path, changed, capture_count=1)
    # A whole local folder can move while its original receipt stays unchanged.
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.move(str(destination.parent), moved / "past_copies")
    _run_io._check_past_copies(moved, details, capture_count=1)
    relocated = moved / "past_copies" / destination.name
    payload = relocated / next(
        iter(checkpoints.read_checkpoint_description(relocated)["files"])
    )
    previous = payload.read_bytes()
    payload.write_bytes(previous + b"damage")
    with pytest.raises(ValueError, match=str(relocated)):
        _run_io._check_past_copies(moved, details, capture_count=1)
    payload.write_bytes(previous)
    # A well-formed replacement with another origin still cannot become this copy.
    other = moved / "other"
    replacement_kwargs: dict[str, Any] = {
        **kwargs,
        "metadata": {**provenance, "seed": provenance["seed"] + 1},
    }
    checkpoints.export_system(actor, other, **replacement_kwargs)
    original = moved / "original"
    relocated.rename(original)
    other.rename(relocated)
    with pytest.raises(ValueError, match="provenance differs"):
        _run_io._check_past_copies(moved, details, capture_count=1)
    shutil.rmtree(relocated)
    original.rename(relocated)
    # Legacy records prove present provenance but never invent missing bytes.
    legacy = copy.deepcopy(details)
    del legacy["metadata"]["host_state"]["past_copies"]["0"]["artifact_id"]
    _run_io._check_past_copies(moved, legacy, capture_count=1)
    shutil.rmtree(relocated)
    _run_io._check_past_copies(moved, legacy, capture_count=1)
    _run_io._check_past_copies(moved, details, capture_count=1)
    del legacy["metadata"]["host_state"]["past_copies"]
    _run_io._check_past_copies(moved, legacy, capture_count=1)


@pytest.mark.parametrize("method", ("mappo", "qmix", "pqn_vdn"))
def test_public_resume_and_continuation_verify_learner_inventory_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: Literal["mappo", "qmix", "pqn_vdn"],
) -> None:
    from marl_battlegrounds.baselines.config import PPOConfig, PQNConfig, QMIXConfig
    from marl_battlegrounds.training import runner

    kwargs: dict[str, Any] = {"ppo": PPOConfig(rollout_length=2, epochs=1)}
    if method == "qmix":
        kwargs = {
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
        kwargs = {
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
        **kwargs,
    )
    from tests.helpers.training_continuation_helpers import fixed_source

    fixed_source(monkeypatch, method)
    reached: list[Any] = []

    class SetupReachedError(Exception):
        pass

    def stop_after_setup(owner: runner._Run) -> Any:  # noqa: ANN401
        reached.append(owner)
        if owner.checkpoint is None:
            owner.save()
        raise SetupReachedError

    monkeypatch.setattr(runner._Run, "execute", stop_after_setup)
    root = tmp_path / "parent"
    with pytest.raises(SetupReachedError):
        runner.train(config, output_dir=root)
    target = reached[-1].checkpoint
    assert target is not None
    inventory = checkpoints._inventory
    scans: list[dict[str, Any]] = []

    def counted(path: Path) -> dict[str, object]:
        start = time.perf_counter()
        value = inventory(path)
        if path.resolve() == target.resolve():
            scans.append(
                {
                    "seconds": time.perf_counter() - start,
                    "bytes": sum(
                        int(cast(dict[str, Any], row)["bytes"])
                        for row in value.values()
                    ),
                }
            )
        return value

    monkeypatch.setattr(checkpoints, "_inventory", counted)
    with pytest.raises(SetupReachedError):
        runner.train(resume_from=target)
    assert len(scans) == 1
    scans.clear()
    with pytest.raises(SetupReachedError):
        runner.extend_training(
            target, additional_env_steps=8, output_dir=tmp_path / "child"
        )
    assert len(scans) == 1
    (tmp_path / "inventory-cost.json").write_text(
        json.dumps(
            {
                "method": method,
                "scope": "CPU public recovery setup, no learner update",
                "scan": scans[0],
            },
            indent=2,
        )
    )


def test_relocated_capture_retry_keeps_checked_original_provenance() -> None:
    from marl_battlegrounds.training.runner import _checked_export_provenance

    member: dict[str, Any] = {
        "name": "member",
        "reference": "/old/actor",
        "registration": {"actor": "one"},
        "registration_id": "r",
        "variables_digest": "weights",
        "input_identity": {"kind": "artifact", "checkpoint_id": "a" * 64},
        "evidence": {"source": "verified export", "exposure": "known"},
    }
    saved: dict[str, Any] = {
        "initial_actor": None,
        "capture_origin": {"capture_id": 2},
        "opponent_members": [member],
        "partner_members": [member],
        "display_name": "Original capture",
    }
    requested = copy.deepcopy(saved)
    requested["display_name"] = "Moved capture"
    for key in ("opponent_members", "partner_members"):
        requested[key][0]["reference"] = "/new/actor"
    before = copy.deepcopy(saved)
    assert _checked_export_provenance(saved, requested) == before
    assert saved == before
    requested["capture_origin"]["capture_id"] = 3
    with pytest.raises(ValueError, match="provenance differs"):
        _checked_export_provenance(saved, requested)


def test_unknown_legacy_history_prefix_remains_unknown_after_new_saves(
    tmp_path: Path,
) -> None:
    from tests.training.test_training_progress import _description, _saved_host

    details = _description(tmp_path, _saved_host(steps=16, pending=None))
    details["collection"]["history_capture_capacity"] = 4
    host = details["metadata"]["host_state"]
    host.pop("past_copies", None)
    _run_io._check_past_copies(tmp_path, details, capture_count=2)
    # A resumed old learner begins a catalog at its next capture, not at zero.
    host.update(past_copies={}, past_copies_start_capture_id=2)
    _run_io._check_past_copies(tmp_path, details, capture_count=2)
    host["past_copies"]["2"] = {
        "env_steps": 16,
        "path": str(tmp_path / "unavailable"),
        "sampled_epsilon": None,
    }
    _run_io._check_past_copies(tmp_path, details, capture_count=3)
    host["past_copies"].clear()
    with pytest.raises(ValueError, match="capture IDs"):
        _run_io._check_past_copies(tmp_path, details, capture_count=3)
    for invalid in (-1, True, 4):
        host["past_copies_start_capture_id"] = invalid
        with pytest.raises(ValueError, match="catalog start"):
            _run_io._check_past_copies(tmp_path, details, capture_count=3)
