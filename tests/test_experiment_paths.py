"""Check exact output paths and sparse recovery without changing saved settings.

Small signed descriptions exercise all six learner configs without model arrays.
Cover omission, null, nested overrides, safe latest pointers, continuation rules,
relocation of path inputs, and rejection before source-changing Hydra writes.
Numerical restore remains the checkpoint owner's separate responsibility.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_module

from marl_battlegrounds.baselines.methods import TRAINING_METHODS, method_settings_field
from marl_battlegrounds.experiments.hydra import run_training
from marl_battlegrounds.training import _checkpoint_metadata as metadata
from marl_battlegrounds.training.config import TrainConfig, config_to_dict

# These tests inspect the shared host reader without loading model libraries.
# pyright: reportPrivateUsage=false
ROOT = Path(__file__).resolve().parents[1]


def saved_run(root: Path, method: str = "mappo") -> tuple[Path, dict[str, Any]]:
    config = config_to_dict(TrainConfig(method=method, seed=137))  # type: ignore[arg-type]
    counters = {"env_steps": 0, "updates": 0}
    if method in ("qmix", "pqn_vdn"):
        counters.update(completed_blocks=0, learning_blocks=0)
    description: dict[str, Any] = {
        "schema_version": 1,
        "kind": "learner",
        "schemas": metadata.checkpoint_schemas(method),
        "metadata": {
            "run_id": "saved-run",
            "attempt_id": "saved-attempt",
            "parent_checkpoint": None,
            "config": config,
            "source": {"test": "source"},
            "dependencies": {"test": "version"},
        },
        "actor_layout": [],
        "actor_digest": "a" * 64,
        "files": {},
        "collection": {},
        "layout": [],
        "recording_token": None,
        "counters": counters,
    }
    identifier = hashlib.sha256(metadata._json_bytes(description)).hexdigest()
    description["checkpoint_id"] = identifier
    checkpoint = (
        root / "checkpoints" / f"{method}_checkpoint_step_000000000000_{identifier}"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint_details.json").write_bytes(
        metadata._json_bytes(description)
    )
    (root / "latest_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_id": identifier,
                "relative_path": str(checkpoint.relative_to(root)),
            }
        )
    )
    return checkpoint, description


@pytest.mark.parametrize("method", TRAINING_METHODS)
def test_sparse_resume_inherits_saved_values_without_fresh_defaults(
    tmp_path: Path, method: str
) -> None:
    run = tmp_path / "saved run"
    checkpoint, description = saved_run(run, method)
    before = (checkpoint / "checkpoint_details.json").read_bytes()
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        cfg = compose(
            config_name="resume", overrides=[f'resume_from="{run}"', "preview=true"]
        )
    assert "method" not in cfg and "seed" not in cfg and "recipe" not in cfg
    report = run_training(cfg, base_dir=tmp_path, launch_dir=tmp_path / "launch")
    assert isinstance(report, dict)
    assert report["config"] == description["metadata"]["config"]
    assert report["resume_from"] == str(checkpoint)
    assert report["output_dir"] == str(run)
    assert not (tmp_path / "launch").exists()
    assert (checkpoint / "checkpoint_details.json").read_bytes() == before


@pytest.mark.parametrize("method", TRAINING_METHODS)
def test_resume_merges_only_learner_leaves_and_compares_effective_values(
    tmp_path: Path, method: str
) -> None:
    run = tmp_path / "run"
    _, description = saved_run(run, method)
    field = method_settings_field(method)
    gamma = description["metadata"]["config"][field]["gamma"]
    report = run_training(
        {"resume_from": str(run), "preview": True, field: {"gamma": gamma}},
        base_dir=tmp_path,
        launch_dir=tmp_path / "launch",
    )
    assert isinstance(report, dict)
    assert report["config"][field] == description["metadata"]["config"][field]
    for change in (
        {field: {"gamma": 0.1}},
        {"seed": 42},
        {"reward": "custom:reward"},
        {"opponents": {"self": 1}},
    ):
        with pytest.raises(ValueError, match="differs"):
            run_training(
                {"resume_from": str(run), "preview": True, **change},
                base_dir=tmp_path,
                launch_dir=tmp_path / "launch",
            )
    assert not (tmp_path / "launch").exists()


def test_resume_does_not_merge_population_assertions(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoint, description = saved_run(run)
    description["metadata"]["config"]["opponents"] = {"self": 0.6, "past": 0.4}
    unsigned = {k: v for k, v in description.items() if k != "checkpoint_id"}
    description["checkpoint_id"] = hashlib.sha256(
        metadata._json_bytes(unsigned)
    ).hexdigest()
    # Explicit checkpoint input does not infer its identity from the folder name.
    (checkpoint / "checkpoint_details.json").write_bytes(
        metadata._json_bytes(description)
    )
    for value in ({"self": 0.6}, None):
        with pytest.raises(ValueError, match="differs"):
            run_training(
                {"resume_from": str(checkpoint), "preview": True, "opponents": value},
                base_dir=tmp_path,
                launch_dir=tmp_path / "launch",
            )


def test_resume_omits_config_and_output_kwargs_when_nothing_was_asserted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marl_battlegrounds.training as training
    from marl_battlegrounds.training.runner import TrainResult

    run = tmp_path / "run"
    checkpoint, _ = saved_run(run)
    calls: list[tuple[object, dict[str, Any]]] = []

    def train(config: object = None, **kwargs: object) -> TrainResult:
        calls.append((config, kwargs))
        return TrainResult(run, run / "actor", None, 0, 0, "complete", (), checkpoint)

    monkeypatch.setattr(training, "train", train)
    run_training(
        {"resume_from": str(run)}, base_dir=tmp_path, launch_dir=tmp_path / "launch"
    )
    assert calls == [(None, {"resume_from": checkpoint})]
    with pytest.raises(ValueError, match="omit output_dir"):
        run_training(
            {"resume_from": str(run), "output_dir": str(run)},
            base_dir=tmp_path,
            launch_dir=tmp_path / "other",
        )
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize(
    "fault", ["schema", "identity", "escape", "symlink", "missing", "duplicate"]
)
def test_latest_pointer_failures_do_not_pick_a_different_checkpoint(
    tmp_path: Path, fault: str
) -> None:
    run = tmp_path / "run"
    checkpoint, _ = saved_run(run)
    pointer_path = run / "latest_checkpoint.json"
    pointer = json.loads(pointer_path.read_text())
    if fault == "schema":
        pointer["schema_version"] = True
    elif fault == "identity":
        pointer["checkpoint_id"] = "b" * 64
    elif fault == "escape":
        pointer["relative_path"] = "../other"
    elif fault == "missing":
        pointer["relative_path"] = "checkpoints/missing_" + pointer["checkpoint_id"]
    elif fault == "symlink":
        original = checkpoint.with_name("original")
        checkpoint.rename(original)
        checkpoint.symlink_to(original, target_is_directory=True)
    if fault == "duplicate":
        pointer_path.write_text('{"schema_version":1,"schema_version":1}')
    else:
        pointer_path.write_text(json.dumps(pointer))
    before = pointer_path.read_bytes()
    with pytest.raises(ValueError):
        metadata.read_resume_checkpoint(run)
    assert pointer_path.read_bytes() == before


def test_resume_preview_stays_backend_free_from_a_different_directory(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run with spaces"
    saved_run(run, "pqn_vdn")
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    program = r"""import sys,runpy
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax','jaxlib','numpy','flax','optax','orbax'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
sys.argv=['train','--config-name','resume','preview=true',sys.argv[1]]
runpy.run_module('marl_battlegrounds.experiments.train',run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, 'resume_from="../run with spaces"'],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["config"]["seed"] == 137
    assert Path(report["launch_dir"]).parent == tmp_path / "run with spaces.launches"
    assert not list(cwd.iterdir())
    assert not list(run.glob(".hydra*"))


@pytest.mark.parametrize("control", ["output_dir", "hydra.run.dir"])
def test_package_output_is_rejected_before_hydra_writes(control: str) -> None:
    destination = ROOT / "src/marl_battlegrounds/unsafe-experiment"
    assert not destination.exists()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "marl_battlegrounds.experiments.train",
            "preview=true",
            f"{control}={destination}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "source package" in result.stdout + result.stderr
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".launches").exists()


def test_checkout_outputs_must_stay_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation import revision

    root = tmp_path / "checkout"
    package = root / "src/marl_battlegrounds"
    package.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("artifacts/\n")
    monkeypatch.setattr(revision, "__file__", str(package / "evaluation/revision.py"))
    revision._check_output_path(root / "artifacts/demo/run")
    revision._check_output_path(tmp_path / "outside")
    with pytest.raises(ValueError, match="source identity"):
        revision._check_output_path(root / "experiment/run")
    before = revision._untracked_content_digest(root)
    records = root / "artifacts/demo/.hydra"
    records.mkdir(parents=True)
    (records / "config.yaml").write_text("seed: 137\n")
    assert revision._untracked_content_digest(root) == before
    assert not (root / "experiment").exists()


def test_resume_cannot_hide_a_link_by_normalizing_first(tmp_path: Path) -> None:
    run = tmp_path / "run"
    saved_run(run)
    link = tmp_path / "linked run"
    link.symlink_to(run, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        run_training(
            {"resume_from": str(link), "preview": True},
            base_dir=tmp_path,
            launch_dir=tmp_path / "launch",
        )
    assert not (tmp_path / "launch").exists()


def test_short_continuation_resume_keeps_its_existing_allowance(tmp_path: Path) -> None:
    from copy import deepcopy

    from marl_battlegrounds.training.config import config_from_dict

    parent_path, parent = saved_run(tmp_path / "parent", "qmix")
    child = deepcopy(parent)
    child.pop("checkpoint_id")
    original = parent["metadata"]["config"]["total_env_steps"]
    child["metadata"]["run_id"] = "child-run"
    child["metadata"]["config"]["total_env_steps"] = 32
    child["metadata"]["validation_declaration"] = {"panel_path": None}
    child["metadata"]["continuation"] = {
        "schema_version": 1,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_id": parent["checkpoint_id"],
        "parent_run_id": parent["metadata"]["run_id"],
        "parent_source": parent["metadata"]["source"],
        "child_source": child["metadata"]["source"],
        "start_env_steps": 0,
        "additional_env_steps": 32,
        "resulting_total_env_steps": 32,
        "changes": {},
        "root_run_id": parent["metadata"]["run_id"],
        "segment": {"root_schedule": {"total_env_steps": original}},
    }
    child["checkpoint_id"] = hashlib.sha256(metadata._json_bytes(child)).hexdigest()
    path = tmp_path / "child/checkpoints" / child["checkpoint_id"]
    path.mkdir(parents=True)
    (path / "checkpoint_details.json").write_bytes(metadata._json_bytes(child))
    with pytest.raises(ValueError):
        config_from_dict(child["metadata"]["config"])
    for asserted in (
        {},
        {"qmix": {"gamma": child["metadata"]["config"]["qmix"]["gamma"]}},
    ):
        report = run_training(
            {"resume_from": str(path), "preview": True, **asserted},
            base_dir=tmp_path,
            launch_dir=tmp_path / "launch",
        )
        assert isinstance(report, dict)
        assert report["total_env_steps"] == 32
    assert not (tmp_path / "launch").exists()


def test_launch_records_never_overwrite_an_existing_directory(tmp_path: Path) -> None:
    launch = tmp_path / "launch"
    launch.mkdir()
    marker = launch / "original.txt"
    marker.write_text("Keep the first declaration")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "marl_battlegrounds.experiments.train",
            "preview=true",
            f"hydra.run.dir={launch}",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must be new or empty" in result.stdout + result.stderr
    assert list(launch.iterdir()) == [marker]
    assert marker.read_text() == "Keep the first declaration"


@pytest.mark.parametrize("preview", [False, True])
def test_null_resume_never_dispatches_fresh_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preview: bool
) -> None:
    import marl_battlegrounds.training as training

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid resume dispatched training")

    monkeypatch.setattr(training, "train", unexpected)
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        cfg = compose(
            config_name="resume",
            overrides=["resume_from=null", f"preview={str(preview).lower()}"],
        )
    with pytest.raises(ValueError, match="resume_from must name"):
        run_training(cfg, base_dir=tmp_path, launch_dir=tmp_path / "launch")
    assert not (tmp_path / "launch").exists()


@pytest.mark.parametrize("override", ["resume_from=null", "~resume_from"])
def test_resume_path_cannot_be_cleared_before_hydra_writes(
    tmp_path: Path, override: str
) -> None:
    launch = tmp_path / "launch"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "marl_battlegrounds.experiments.train",
            "--config-name",
            "resume",
            *(
                ["resume_from=unused", override]
                if override.startswith("~")
                else [override]
            ),
            "preview=true",
            f"hydra.run.dir={launch}",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "resume_from must name" in result.stdout + result.stderr
    assert not launch.exists()


def test_direct_launch_cannot_replace_previous_effective_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marl_battlegrounds.training as training

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Occupied launch dispatched training")

    monkeypatch.setattr(training, "train", unexpected)
    launch = tmp_path / "launch"
    records = launch / ".hydra"
    records.mkdir(parents=True)
    config_file = records / "training_config.json"
    config_file.write_text('{"seed":137}\n')
    before = config_file.read_bytes()
    with pytest.raises(ValueError, match="must be new or empty"):
        run_training({"seed": 138}, base_dir=tmp_path, launch_dir=launch)
    assert config_file.read_bytes() == before
    assert sorted(p.relative_to(launch).as_posix() for p in launch.rglob("*")) == [
        ".hydra",
        ".hydra/training_config.json",
    ]


@pytest.mark.parametrize("resume", [False, True])
def test_study_destination_is_checked_before_identity_reads_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    from marl_battlegrounds.evaluation import revision
    from marl_battlegrounds.training import study

    root = tmp_path / "unsafe-study"
    checks: list[Path] = []

    def reject(path: Path) -> None:
        checks.append(path)
        raise ValueError("Unsafe source identity destination")

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Unsafe destination reached declaration setup")

    monkeypatch.setattr(revision, "_check_output_path", reject)
    monkeypatch.setattr(study, "_freeze", unexpected)
    monkeypatch.setattr(study, "_read", unexpected)
    with pytest.raises(ValueError, match="Unsafe source identity"):
        study._prepare(
            None if resume else {}, None if resume else root, root if resume else None
        )
    assert checks == [root]
    assert not root.exists()
