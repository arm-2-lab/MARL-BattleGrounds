"""Check anonymous actor bytes, provenance, relocation and fail-before-write rules.

Use real PPO/QMIX/PQN payloads from the shared save fixture. The safe original
metadata is built through current schema owners; it is not a claim that every
historical run is anonymous. No training update or scientific result is produced.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import json
import shutil
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest
from tests.training.test_training_checkpoint_writes import saved_family as saved_family

from marl_battlegrounds.training import checkpoints
from marl_battlegrounds.training._anonymous import _binding
from marl_battlegrounds.training._content import (
    TrainingContentBinding,
    pinned_opponent_evidence,
)


def _safe_checkpoint(
    tmp_path: Path, family: tuple[Any, Any, dict[str, Any], dict[str, Any]]
) -> tuple[Path, TrainingContentBinding]:
    collection, state, metadata, settings = family
    safe = _binding(collection.binding.model_dump(mode="json"), project_paths=True)
    binding = TrainingContentBinding.model_validate_json(json.dumps(safe))
    collection = replace(collection, binding=binding)
    metadata = copy.deepcopy(metadata)
    metadata.update(checkpoints.runtime_identity(settings["method"]))
    metadata.update(run_id="a" * 32, attempt_id="b" * 32, host_state={})
    root = tmp_path / "source"
    root.mkdir()
    return checkpoints.save_checkpoint(
        root, collection, state, metadata=metadata, **settings
    ), binding


def _signed(details: dict[str, Any]) -> dict[str, Any]:
    details = copy.deepcopy(details)
    details.pop("checkpoint_id", None)
    details["checkpoint_id"] = sha256(checkpoints._json_bytes(details)).hexdigest()
    return details


def _actor_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in (root / "actor").rglob("*")
        if p.is_file()
    }


def test_anonymous_checkpoint_actor_and_capture_preserve_real_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_family: tuple[Any, Any, dict[str, Any], dict[str, Any]],
) -> None:
    checkpoint, binding = _safe_checkpoint(tmp_path, saved_family)
    original = checkpoints.read_checkpoint_details(checkpoint)
    actor = checkpoints.export_actor(
        checkpoint, output_dir=tmp_path / "source/actors/original"
    )
    original_bytes = _actor_bytes(actor)
    before = {
        p: p.read_bytes() for p in (tmp_path / "source").rglob("*") if p.is_file()
    }
    system = checkpoints.load_system(actor)
    evidence = pinned_opponent_evidence(binding, system, export=actor)
    assert evidence["source"] == "verified export"
    assert evidence["exposure"] == "none"

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Anonymous copy restored arrays")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "_restore_arrays", forbidden)
        anonymous = checkpoints.export_actor(
            actor, output_dir=tmp_path / "review/actor", anonymous=True
        )
        from_checkpoint = checkpoints.export_actor(
            checkpoint, output_dir=tmp_path / "review/from-checkpoint", anonymous=True
        )
        assert (
            checkpoints.export_actor(actor, output_dir=anonymous, anonymous=True)
            == anonymous
        )
        repeated = checkpoints.export_actor(
            anonymous, output_dir=tmp_path / "review/repeated", anonymous=True
        )
    for path in (anonymous, from_checkpoint, repeated):
        details = checkpoints.read_checkpoint_details(path)
        assert _actor_bytes(path) == original_bytes
        assert details["actor_digest"] == original["actor_digest"]
        assert details["schemas"] == original["schemas"]
        assert details["metadata"]["run_id"] == "a" * 32
        assert details["metadata"]["checkpoint_id"] == original["checkpoint_id"]
        assert details["metadata"]["anonymous_copy"]["text_files_checked"]
        assert pinned_opponent_evidence(binding, system, export=path) == evidence
    assert checkpoints.read_checkpoint_details(
        anonymous
    ) == checkpoints.read_checkpoint_details(repeated)
    assert before == {p: p.read_bytes() for p in before}
    moved = tmp_path / "relocated"
    anonymous.rename(moved)
    assert (
        pinned_opponent_evidence(binding, checkpoints.load_system(moved), export=moved)
        == evidence
    )
    assert (
        checkpoints.artifact_identity(moved)["actor_digest"]
        == checkpoints.artifact_identity(actor)["actor_digest"]
    )

    capture = tmp_path / "capture"
    shutil.copytree(actor, capture)
    details = checkpoints.read_checkpoint_details(capture)
    metadata = details["metadata"]
    metadata.pop("checkpoint_id")
    metadata["run_id"] = "Human experiment label"
    metadata["capture_origin"] = {
        "capture_id": 7,
        "update_index": 0,
        "content_binding": saved_family[0].binding.model_dump(mode="json"),
        "epsilon": None if saved_family[3]["method"] == "mappo" else 0.25,
    }
    (capture / "actor_details.json").write_bytes(
        checkpoints._json_bytes(_signed(details))
    )
    for field in ("dtype", "path"):
        altered = copy.deepcopy(details)
        if field == "dtype":
            altered["actor_layout"][0]["dtype"] = "author@example.com"
        else:
            altered["actor_layout"][0]["path"][0]["key"] = "/home/private-author/model"
        (capture / "actor_details.json").write_bytes(
            checkpoints._json_bytes(_signed(altered))
        )
        with pytest.raises(ValueError, match="actor_layout"):
            checkpoints.export_actor(
                capture, output_dir=tmp_path / "layout-failure/actor", anonymous=True
            )
        assert not (tmp_path / "layout-failure").exists()
    (capture / "actor_details.json").write_bytes(
        checkpoints._json_bytes(_signed(details))
    )
    storage = capture / "actor/_CHECKPOINT_METADATA"
    raw_storage = storage.read_bytes()
    variants = [
        b'{"custom_metadata":{"private_note":"Ada Example"},' + raw_storage.lstrip()[1:]
    ]
    for key in ("init_timestamp_nsecs", "commit_timestamp_nsecs"):
        altered_storage = json.loads(raw_storage)
        altered_storage[key] = "Ada Example"
        variants.append(checkpoints._json_bytes(altered_storage))
    for raw in variants:
        storage.write_bytes(raw)
        altered = copy.deepcopy(details)
        altered["files"]["actor/_CHECKPOINT_METADATA"] = {
            "bytes": len(raw),
            "sha256": sha256(raw).hexdigest(),
        }
        (capture / "actor_details.json").write_bytes(
            checkpoints._json_bytes(_signed(altered))
        )
        with monkeypatch.context() as patch:
            patch.setattr(checkpoints, "_restore_arrays", forbidden)
            with pytest.raises(ValueError):
                checkpoints.export_actor(
                    capture,
                    output_dir=tmp_path / "storage-failure/actor",
                    anonymous=True,
                )
        assert not (tmp_path / "storage-failure").exists()
        assert storage.read_bytes() == raw
    storage.write_bytes(raw_storage)
    (capture / "actor_details.json").write_bytes(
        checkpoints._json_bytes(_signed(details))
    )
    old_capture_evidence = pinned_opponent_evidence(binding, system, export=capture)
    anonymous_capture = checkpoints.export_actor(
        capture, output_dir=tmp_path / "review/capture", anonymous=True
    )
    changed = checkpoints.read_checkpoint_details(anonymous_capture)
    assert changed["metadata"]["run_id"].startswith("run-")
    assert changed["metadata"]["capture_origin"]["capture_id"] == 7
    assert "checkpoint_id" not in changed["metadata"]
    assert _actor_bytes(anonymous_capture) == original_bytes
    assert (
        pinned_opponent_evidence(binding, system, export=anonymous_capture)
        == old_capture_evidence
    )
    copied = checkpoints.export_actor(
        anonymous_capture, output_dir=tmp_path / "review/capture-again", anonymous=True
    )
    assert checkpoints.read_checkpoint_details(copied) == changed

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
    )

    shared = marl_bgs.shared_policy(marl_bgs.policy("tdm-beta"))
    opaque_system = marl_bgs.System(
        name="External research system",
        init=shared.init,
        apply=shared.apply,
        reset_memory=shared.reset_memory,
        variables=shared.variables,
    )
    for label, partner in (("known", shared), ("unknown", opaque_system)):
        exposure = pinned_opponent_evidence(binding, partner)
        assert exposure["exposure"] == label
        identifier, registration = normalize_system_registration(
            partner, phase="evaluation", frozen=True
        )
        member = {
            "name": "Partner",
            "reference": None,
            "system_name": partner.name,
            "registration_id": identifier,
            "registration": registration,
            "execution": partner.execution,
            "variables_digest": registration["variables_digest"],
            "evidence": exposure,
        }
        inherited = copy.deepcopy(details)
        inherited["metadata"]["partner_members"] = [member]
        (capture / "actor_details.json").write_bytes(
            checkpoints._json_bytes(_signed(inherited))
        )
        before_exposure = pinned_opponent_evidence(binding, system, export=capture)
        exported = checkpoints.export_actor(
            capture, output_dir=tmp_path / "review" / label, anonymous=True
        )
        relocated = tmp_path / (label + "-moved")
        exported.rename(relocated)
        assert (
            pinned_opponent_evidence(binding, system, export=relocated)
            == before_exposure
        )
        assert checkpoints.read_checkpoint_details(relocated)["metadata"][
            "partner_members"
        ] == [member]
        assert _actor_bytes(relocated) == original_bytes


def test_anonymous_admission_refuses_unknown_private_and_broken_origins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_family: tuple[Any, Any, dict[str, Any], dict[str, Any]],
) -> None:
    checkpoint, _ = _safe_checkpoint(tmp_path, saved_family)
    original_checkpoint = checkpoints.read_checkpoint_details(checkpoint)
    for name in ("metrics", "root_bits", "shaping", "recording", "reward_settings"):
        changed = copy.deepcopy(original_checkpoint)
        changed["collection"][name] = {"private_note": "Ada Example"}
        (checkpoint / "checkpoint_details.json").write_bytes(
            checkpoints._json_bytes(_signed(changed))
        )
        with pytest.raises(ValueError, match=r"scalar|numeric"):
            checkpoints.export_actor(
                checkpoint, output_dir=tmp_path / "unclassified/actor", anonymous=True
            )
        assert not (tmp_path / "unclassified").exists()
    from marl_battlegrounds.training._compilation import execution_identity

    for name in ("schema_version", "declared_xla_flags"):
        changed = copy.deepcopy(original_checkpoint)
        changed["metadata"]["execution"] = execution_identity()
        changed["metadata"]["execution"][name] = {"private_note": "Ada Example"}
        (checkpoint / "checkpoint_details.json").write_bytes(
            checkpoints._json_bytes(_signed(changed))
        )
        with pytest.raises(ValueError, match="scalar"):
            checkpoints.export_actor(
                checkpoint, output_dir=tmp_path / "unclassified/actor", anonymous=True
            )
        assert not (tmp_path / "unclassified").exists()
    for section, key, value in (
        ("counters", "updates", {"private_note": "Ada Example"}),
        ("counters", "extra", "Ada Example"),
        ("collection", "shaping", "Ada Example"),
    ):
        changed = copy.deepcopy(original_checkpoint)
        changed[section][key] = value
        (checkpoint / "checkpoint_details.json").write_bytes(
            checkpoints._json_bytes(_signed(changed))
        )
        with pytest.raises(ValueError):
            checkpoints.export_actor(
                checkpoint, output_dir=tmp_path / "unclassified/actor", anonymous=True
            )
        assert not (tmp_path / "unclassified").exists()
    (checkpoint / "checkpoint_details.json").write_bytes(
        checkpoints._json_bytes(original_checkpoint)
    )
    actor = checkpoints.export_actor(
        checkpoint, output_dir=tmp_path / "source/actors/original"
    )
    original = checkpoints.read_checkpoint_details(actor)
    for key, value in (
        ("unknown_note", "Private"),
        ("run_id", "Named Research Run"),
        ("source", {"scope": "Unclassified source"}),
    ):
        changed = copy.deepcopy(original)
        changed["metadata"][key] = value
        (actor / "actor_details.json").write_bytes(
            checkpoints._json_bytes(_signed(changed))
        )
        with pytest.raises(ValueError):
            checkpoints.export_actor(
                actor, output_dir=tmp_path / "new-parent/actor", anonymous=True
            )
        assert not (tmp_path / "new-parent").exists()
    (actor / "actor_details.json").write_bytes(checkpoints._json_bytes(original))
    for name in (
        "actor/private-note.json",
        "origin/private-note.txt",
        "actor/foreign/d/abcdef",
    ):
        entry = actor / name
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text('{"note": "Ada Example"}')
        changed = copy.deepcopy(original)
        changed["files"][name] = {
            "bytes": entry.stat().st_size,
            "sha256": sha256(entry.read_bytes()).hexdigest(),
        }
        (actor / "actor_details.json").write_bytes(
            checkpoints._json_bytes(_signed(changed))
        )
        with pytest.raises(ValueError):
            checkpoints.export_actor(
                actor, output_dir=tmp_path / "new-parent/actor", anonymous=True
            )
        assert not (tmp_path / "new-parent").exists()
        entry.unlink()
        (actor / "actor_details.json").write_bytes(checkpoints._json_bytes(original))

    copytree = checkpoints.shutil.copytree

    def change_description(
        source: Path,
        destination: Path,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        result = copytree(source, destination, *args, **kwargs)
        if Path(source) == actor:
            changed = copy.deepcopy(original)
            changed["metadata"]["run_id"] = "c" * 32
            (actor / "actor_details.json").write_bytes(
                checkpoints._json_bytes(_signed(changed))
            )
        return str(result)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints.shutil, "copytree", change_description)
        with pytest.raises(ValueError, match="Source description changed"):
            checkpoints.export_actor(
                actor, output_dir=tmp_path / "copy-failure/actor", anonymous=True
            )
    assert not (tmp_path / "copy-failure/actor").exists()
    assert not list((tmp_path / "copy-failure").glob(".pending-*"))
    (actor / "actor_details.json").write_bytes(checkpoints._json_bytes(original))

    write = Path.write_bytes

    def fail_origin(path: Path, value: bytes) -> int:
        if path.name == "checkpoint_details.json" and any(
            part.startswith(".pending-") for part in path.parts
        ):
            raise OSError("Injected origin write failure")
        return write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", fail_origin)
        with pytest.raises(OSError, match="origin write"):
            checkpoints.export_actor(
                actor, output_dir=tmp_path / "origin-failure/actor", anonymous=True
            )
    assert not list((tmp_path / "origin-failure").iterdir())

    anonymous = checkpoints.export_actor(
        actor, output_dir=tmp_path / "review/actor", anonymous=True
    )
    origin = (
        anonymous
        / "origin"
        / original["metadata"]["checkpoint_id"]
        / "checkpoint_details.json"
    )
    origin.write_text("Damaged")
    with pytest.raises(ValueError):
        checkpoints.export_actor(
            anonymous, output_dir=tmp_path / "new-parent/actor", anonymous=True
        )
    assert not (tmp_path / "new-parent").exists()


@pytest.mark.parametrize(
    "value",
    [
        r"\\Ada-Laptop\users\Ada\study",
        "Saved at (/home/ada/study)",
        "https://example.com/ada",
        "ada@example.com",
    ],
)
def test_anonymous_scan_rejects_identifying_text(value: str) -> None:
    from marl_battlegrounds.training._anonymous import _scan

    with pytest.raises(ValueError, match=r"metadata\.name"):
        _scan(value, "metadata.name", frozenset())


@pytest.mark.parametrize("helper", ["callable", "component"])
def test_known_fields_cannot_hide_unclassified_nested_metadata(helper: str) -> None:
    from marl_battlegrounds.training._anonymous import _callable, _components

    if helper == "component":
        with pytest.raises(ValueError, match="scalar"):
            _components([{"name": {"private_note": "Ada Example"}}], "components")
    else:
        value: dict[str, Any] = {
            name: None
            for name in (
                "callable_name",
                "code_digest",
                "code_scope",
                "closure_content",
                "defaults_digest",
                "external_state",
            )
        }
        value["callable_name"] = {"private_note": "Ada Example"}
        with pytest.raises(ValueError, match="scalar"):
            _callable(value, "hook")


def test_anonymous_storage_descriptions_reject_unclassified_records() -> None:
    from marl_battlegrounds.training._anonymous import _storage_description

    layout = [{"dtype": "float32", "path": [{"key": "params"}], "shape": [2]}]
    good: dict[str, Any] = {
        "actor_layout": layout,
        "files": {"actor/data": {"bytes": 4, "sha256": "a" * 64}},
    }
    _storage_description(good)
    for location in ("layout", "path", "file"):
        value = copy.deepcopy(good)
        if location == "layout":
            value["actor_layout"][0]["private_note"] = "Ada Example"
        elif location == "path":
            value["actor_layout"][0]["path"][0]["key"] = {"private_note": "Ada Example"}
        else:
            value["files"]["actor/data"]["private_note"] = "Ada Example"
        with pytest.raises(ValueError):
            _storage_description(value)


def test_anonymous_orbax_metadata_refuses_extensions_and_encoded_records() -> None:
    import base64

    from marl_battlegrounds.training._anonymous import _orbax_metadata

    tree: dict[str, Any] = {
        "tree_metadata": {},
        "use_ocdbt": True,
        "use_zarr3": False,
        "store_array_data_equal_to_fill_value": True,
        "custom_metadata": None,
    }
    arrays: dict[str, Any] = {
        "array_metadatas": [
            {
                "array_metadata": {
                    "param_name": "params",
                    "write_shape": [2],
                    "chunk_shape": [2],
                    "ext_metadata": None,
                }
            }
        ]
    }
    key = base64.b64encode(b"params").decode()
    sharding: dict[str, Any] = {
        key: json.dumps(
            {"sharding_type": "SingleDeviceSharding", "device_str": "cpu:0"}
        )
    }
    for name, value in (
        ("actor/_METADATA", tree),
        ("actor/array_metadatas/process_0", arrays),
        ("actor/_sharding", sharding),
    ):
        _orbax_metadata(value, name)
        changed = copy.deepcopy(value)
        if name.endswith("_METADATA"):
            changed["custom_metadata"] = {"private_note": "Ada Example"}
        elif name.endswith("process_0"):
            changed["array_metadatas"][0]["array_metadata"]["ext_metadata"] = {
                "private_note": "Ada Example"
            }
        else:
            inner = json.loads(changed[key])
            inner["private_note"] = "Ada Example"
            changed[key] = json.dumps(inner)
        with pytest.raises(ValueError):
            _orbax_metadata(changed, name)


def test_anonymous_storage_scalar_types_and_nested_duplicate_json() -> None:
    import base64

    from marl_battlegrounds.training._anonymous import _orbax_metadata

    leaf: dict[str, Any] = {
        "key_metadata": [{"key": "params", "key_type": 2}],
        "value_metadata": {
            "value_type": "jax.Array",
            "skip_deserialize": False,
            "write_shape": [2],
        },
    }
    tree: dict[str, Any] = {
        "tree_metadata": {"params": leaf},
        "use_ocdbt": True,
        "use_zarr3": False,
        "store_array_data_equal_to_fill_value": True,
        "custom_metadata": None,
    }
    _orbax_metadata(tree, "actor/_METADATA")
    for part, field in (
        ("key_metadata", "key_type"),
        ("value_metadata", "skip_deserialize"),
        ("value_metadata", "value_type"),
    ):
        changed = copy.deepcopy(tree)
        target = changed["tree_metadata"]["params"][part]
        if isinstance(target, list):
            target = cast(list[dict[str, Any]], target)[0]
        target[field] = "Ada Example"
        with pytest.raises(ValueError):
            _orbax_metadata(changed, "actor/_METADATA")
    encoded = base64.b64encode(b"params").decode()
    duplicate = (
        '{"device_str":"Ada Example",'
        '"sharding_type":"SingleDeviceSharding","device_str":"cpu:0"}'
    )
    with pytest.raises(ValueError, match="cannot be decoded"):
        _orbax_metadata({encoded: duplicate}, "actor/_sharding")
