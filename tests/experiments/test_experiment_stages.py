"""Check authored YAML paths and thin study/selection/report stage contracts.

These host and mocked-worker checks prove command wiring, sparse recovery,
ordered preview, and exact confirmation handoff. Existing numerical validation
and C22 prove played games; these checks make no learning or speed claim.
"""

import copy
import json
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from marl_battlegrounds import training
from marl_battlegrounds.experiments import select, study
from marl_battlegrounds.training._launch import read_declaration
from marl_battlegrounds.training.selection import (
    _declaration,  # pyright: ignore[reportPrivateUsage]
)


def test_yaml_and_json_study_preview_share_path_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration = {
        "name": "Software Check",
        "cases": [
            {
                "id": method,
                "config": {
                    "method": method,
                    "total_env_steps": 20480,
                    "num_envs": 32,
                    "pinned_opponent_share": 0.2,
                    "pinned_opponent": "./actors/opponent",
                },
                "seeds": [710, 711],
            }
            for method in ("mappo", "ff_mappo", "ippo", "ff_ippo", "qmix", "pqn_vdn")
        ],
        "resources": {"devices": [{"device": "cpu"}]},
    }
    source = tmp_path / "declarations"
    source.mkdir()
    json_path, yaml_path = source / "study.json", source / "study.yaml"
    json_path.write_text(json.dumps(declaration))
    OmegaConf.save(OmegaConf.create(declaration), yaml_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = tmp_path / "study"
    before = sorted(tmp_path.rglob("*"))
    yaml_preview = training.preview_study(yaml_path, output_dir=output)
    assert yaml_preview == training.preview_study(json_path, output_dir=output)
    assert len(yaml_preview["cases"]) == 12
    assert [row["config"]["seed"] for row in yaml_preview["cases"]] == [710, 711] * 6
    for row in yaml_preview["cases"]:
        assert row["config"]["pinned_opponent"] == str(source / "actors/opponent")
        assert row["run_dir"] == str(output / "cases" / row["id"] / "run")
    assert sorted(tmp_path.rglob("*")) == before
    assert yaml_preview["pending_checks"]


def test_preview_is_host_only_in_fresh_process(tmp_path: Path) -> None:
    source = tmp_path / "study.yaml"
    source.write_text(
        "name: Host Check\ncases:\n- id: short\n  config:\n"
        "    method: ff_ippo\n    total_env_steps: 20480\n  seeds: [710]\n"
    )
    script = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'flax', 'optax', 'flashbax', 'torch'}:
            raise AssertionError('Numerical import: ' + fullname)
sys.meta_path.insert(0, Guard())
from marl_battlegrounds import training
result = training.preview_study(sys.argv[1], output_dir=sys.argv[2])
assert result['preview'] and len(result['cases']) == 1
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(source), str(tmp_path / "out")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("text", ["field: ???", "[]", "field: .nan"])
def test_bad_yaml_fails_before_output(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises((ValueError, TypeError)):
        read_declaration(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.yaml"]


def test_selection_yaml_keeps_file_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "selection.yaml"
    value: dict[str, Any] = {
        "name": "Separate Choice",
        "previous_decision": "decisions/selection_decision.json",
        "confirmation_results": ["validation/validation_summary.json"],
    }
    OmegaConf.save(OmegaConf.create(value), path)
    monkeypatch.chdir(tmp_path.parent)
    parsed = _declaration(path)
    assert parsed["previous_decision"] == str(tmp_path / value["previous_decision"])
    assert parsed["confirmation_results"] == [
        str(tmp_path / value["confirmation_results"][0])
    ]


@pytest.mark.parametrize("resume", [False, True])
def test_study_stage_calls_owner_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    calls: list[dict[str, Any]] = []

    def execute(config: object, **kwargs: object) -> dict[str, Any]:
        calls.append({"config": config, **kwargs})
        return {"study_dir": str(tmp_path / "study"), "status": "complete"}

    monkeypatch.setattr(training, "run_study", execute)
    value = {
        "declaration": None if resume else str(tmp_path / "study.yaml"),
        "output_dir": None if resume else str(tmp_path / "study"),
        "resume_from": str(tmp_path / "study") if resume else None,
        "preview": False,
    }
    result = study.main.__wrapped__(OmegaConf.create(value))
    assert result["status"] == "complete"
    assert calls == [
        {
            "config": value["declaration"],
            **{k: value[k] for k in ("output_dir", "resume_from")},
        }
    ]


def test_selection_stage_preserves_expected_task_and_links_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "decision"
    task = {"task_id": "original-task", "partners": [{"name": "Priest"}]}
    arguments = {
        "checkpoint": "actor",
        "panel": "panel",
        "output_dir": str(output / "confirmation/task"),
        "expected_task": task,
        "partners": {"Priest": "tdm-alpha"},
    }
    declared = {
        "schema_version": 1,
        "name": "Software Choice",
        "confirmation_results": ["earlier-summary.json"],
    }
    pending: dict[str, Any] = {
        "status": "needs_confirmation",
        "declaration": declared,
        "runs": [
            {
                "run_dir": str(tmp_path / "run"),
                "needs_confirmation": [
                    {
                        "ready_to_run": True,
                        "arguments": arguments,
                    }
                ],
            }
        ],
    }
    selections: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []

    def choose(run_dirs: object, **kwargs: object) -> dict[str, Any]:
        selections.append({"run_dirs": run_dirs, **kwargs})
        return pending if len(selections) == 1 else {"status": "complete", "runs": []}

    def validate(**kwargs: object) -> dict[str, Any]:
        validations.append(kwargs)
        return {"complete": True}

    monkeypatch.setattr(training, "reselect_checkpoint", choose)
    monkeypatch.setattr(training, "validate_checkpoint", validate)
    result = select.main.__wrapped__(
        OmegaConf.create(
            {
                "run_dirs": [str(tmp_path / "run")],
                "declaration": str(tmp_path / "selection.yaml"),
                "output_dir": str(output),
                "resume_from": None,
                "confirm": True,
            }
        )
    )
    assert result["status"] == "complete"
    assert validations == [arguments]
    assert selections[1]["declaration"] == {
        **declared,
        "previous_decision": str(output / "selection_decision.json"),
        "confirmation_results": [
            "earlier-summary.json",
            str(output / "confirmation/task/validation_summary.json"),
        ],
    }
    assert selections[1]["output_dir"] == output / "confirmed"


@pytest.mark.parametrize(
    "condition", ["wrong_parent", "completed_missing_evidence", "unfinished_child"]
)
def test_selection_resume_rejects_stale_or_incomplete_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    from marl_battlegrounds.training import selection

    root = tmp_path / "selection"
    root.mkdir()
    pending: dict[str, Any] = {
        "schema_version": 1,
        "status": "needs_confirmation",
        "declaration": {},
        "runs": [],
        "run_dirs": [],
    }
    if condition == "completed_missing_evidence":
        pending.update(status="complete", run_dirs=[str(tmp_path / "missing")])
    pending["decision_id"] = selection._digest(pending)  # pyright: ignore[reportPrivateUsage]
    (root / "selection_decision.json").write_text(json.dumps(pending))
    if condition == "unfinished_child":
        (root / "confirmed").mkdir()
        (root / "confirmed/foreign.txt").write_text("Keep")
    elif condition == "wrong_parent":
        (root / "confirmed").mkdir()
        (root / "confirmed/selection_decision.json").write_text("Saved")
        original = selection.read_selection_decision

        def read(path: Path, *, verify_evidence: bool = False) -> dict[str, Any]:
            if path.parent.name == "confirmed":
                assert verify_evidence
                return {"status": "complete", "previous_decision_id": "wrong"}
            return original(path, verify_evidence=verify_evidence)

        monkeypatch.setattr(selection, "read_selection_decision", read)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, OSError)):
        select.main.__wrapped__(
            OmegaConf.create(
                {
                    "run_dirs": [],
                    "declaration": None,
                    "output_dir": None,
                    "resume_from": str(root),
                    "confirm": True,
                }
            )
        )
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("problem", ["pool", "member", "path", "outside"])
def test_prepare_rejects_bad_pool_before_output(tmp_path: Path, problem: str) -> None:
    from marl_battlegrounds.experiments.prepare import prepare_software_check

    pool = json.loads(
        files("marl_battlegrounds.experiments")
        .joinpath("fixtures/combat_support_opponents.json")
        .read_text()
    )
    pool = copy.deepcopy(pool)
    if problem == "pool":
        pool = []
    elif problem == "member":
        pool["members"][0] = None
    elif problem == "outside":
        pool["members"][0]["path"] = "../private-actor"
    (tmp_path / "pool.json").write_text(json.dumps(pool))
    with pytest.raises(ValueError):
        prepare_software_check(tmp_path, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_analysis_missing_protocol_values_fail_before_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.experiments.analyze import run_analysis

    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("cases: ???\n")
    called: list[object] = []

    def record(*args: object, **kwargs: object) -> None:
        called.append(args)

    monkeypatch.setattr(training, "analyze", record)
    with pytest.raises(ValueError, match=r"missing|Missing"):
        run_analysis(
            OmegaConf.create(
                {
                    "study_dirs": [],
                    "output_dir": str(tmp_path / "out"),
                    "grouping": str(tmp_path / "grouping.yaml"),
                    "selection": None,
                    "evaluation_dirs": [],
                    "analysis_data_dir": None,
                    "protocol": str(protocol),
                }
            )
        )
    assert not called
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("location", ["bundle", "actor", "protocol_parent", "link"])
def test_preparation_rejects_input_overlap_without_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    from marl_battlegrounds.experiments.prepare import prepare_software_check
    from marl_battlegrounds.training import checkpoints

    bundle = tmp_path / "inputs"
    actor = bundle / "actors/model"
    actor.mkdir(parents=True)
    (actor / "actor_details.json").write_text("Preserve these bytes")
    (bundle / "pool.json").write_text("Not read before path admission")
    output = bundle if location == "bundle" else actor
    if location == "protocol_parent":
        bundle.rename(tmp_path / "protocol")
        bundle = tmp_path / "protocol"
        output = tmp_path
    elif location == "link":
        output = tmp_path / "alias"
        output.symlink_to(bundle, target_is_directory=True)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Path preflight loaded an actor")

    monkeypatch.setattr(checkpoints, "artifact_identity", forbidden)
    before = {p: p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match=r"outside|symbolic"):
        prepare_software_check(bundle, output)
    assert before == {p: p.read_bytes() for p in bundle.rglob("*") if p.is_file()}


def test_preparation_hydra_callback_protects_inputs_before_records(
    tmp_path: Path,
) -> None:
    from hydra.types import RunMode

    from marl_battlegrounds.experiments.hydra import (
        _prepare_launch,  # pyright: ignore[reportPrivateUsage]
    )

    bundle = tmp_path / "inputs"
    bundle.mkdir()
    output = tmp_path / "workflow"
    for launch in (bundle / ".launches", tmp_path / "safe-launch"):
        protected_output = output if launch.parent == bundle else bundle / "actor"
        with pytest.raises(ValueError, match="outside"):
            _prepare_launch(
                mode=RunMode.RUN,
                output_dir=protected_output,
                resume_from=None,
                launch_dir=str(launch),
                records_subdir=".hydra",
                opponent_bundle=str(bundle),
            )
    assert list(tmp_path.iterdir()) == [bundle]


def test_selection_paths_expand_home_before_yaml_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    declarations = tmp_path / "declarations"
    declarations.mkdir()
    path = declarations / "select.yaml"
    path.write_text(
        "name: Home Paths\nprevious_decision: ~/decisions/prior.json\n"
        "confirmation_results: [~/checks/validation_summary.json]\n"
    )
    parsed = _declaration("~/declarations/select.yaml")
    assert parsed["previous_decision"] == str(tmp_path / "decisions/prior.json")
    assert parsed["confirmation_results"] == [
        str(tmp_path / "checks/validation_summary.json")
    ]


def test_grouping_expands_home_paths_before_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training.analysis import (
        _training_grouping,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    source = tmp_path / "declarations/grouping.yaml"
    source.parent.mkdir()
    source.write_text(
        "schema_version: 1\nmaps: [42]\nopponents: [Opponent]\npanel_digest: saved\n"
        "red_zone_depth: 5.0\nruns:\n- run_dir: ~/runs/one\n"
        "  group: Method\n  unit: seed-710\n"
    )
    expected = str(tmp_path / "runs/one")
    for inputs in (None, ["~/runs/one"]):
        result = _training_grouping("~/declarations/grouping.yaml", inputs)
        assert result["runs"][0]["run_dir"] == expected


def test_stage_wrappers_expand_home_before_hydra_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.experiments import analyze, prepare

    monkeypatch.setenv("HOME", str(tmp_path))
    calls: list[dict[str, Any]] = []

    def prepare_pool(bundle: str, output: str) -> Path:
        calls.append({"bundle": bundle, "output": output})
        return Path(output) / "protocol/study.yaml"

    def run_study(declaration: object, **kwargs: object) -> dict[str, Any]:
        calls.append({"declaration": declaration, **kwargs})
        return {"study_dir": str(tmp_path / "study")}

    def select_actor(runs: object, **kwargs: object) -> dict[str, Any]:
        calls.append({"runs": runs, **kwargs})
        return {"status": "complete"}

    def report(runs: object, **kwargs: object) -> dict[str, Any]:
        calls.append({"runs": runs, **kwargs})
        return {"artifacts": {}}

    def export(roots: object, **kwargs: object) -> dict[str, Path]:
        calls.append({"roots": roots, **kwargs})
        return {"data_index.json": tmp_path / "data/index.json"}

    monkeypatch.setattr(prepare, "prepare_software_check", prepare_pool)
    prepare.main.__wrapped__(
        OmegaConf.create(
            {
                "protocol": "software_check",
                "opponent_bundle": "~/pool",
                "output_dir": "~/out",
            }
        )
    )
    assert calls.pop() == {
        "bundle": str(tmp_path / "pool"),
        "output": str(tmp_path / "out"),
    }
    monkeypatch.setattr(training, "run_study", run_study)
    study.main.__wrapped__(
        OmegaConf.create(
            {
                "declaration": "~/study.yaml",
                "output_dir": "~/study",
                "resume_from": None,
                "preview": False,
            }
        )
    )
    assert calls.pop() == {
        "declaration": str(tmp_path / "study.yaml"),
        "output_dir": str(tmp_path / "study"),
        "resume_from": None,
    }
    monkeypatch.setattr(training, "reselect_checkpoint", select_actor)
    select.main.__wrapped__(
        OmegaConf.create(
            {
                "run_dirs": ["~/run"],
                "declaration": "~/select.yaml",
                "output_dir": "~/choice",
                "resume_from": None,
                "confirm": False,
            }
        )
    )
    selected = calls.pop()
    assert selected == {
        "runs": [str(tmp_path / "run")],
        "declaration": str(tmp_path / "select.yaml"),
        "output_dir": tmp_path / "choice",
    }
    monkeypatch.setattr(training, "analyze", report)
    monkeypatch.setattr(training, "export_study_data", export)

    def no_runs(path: Path) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(training, "list_runs", no_runs)
    (tmp_path / "protocol.yaml").write_text("schema_version: 1\n")
    analyze.run_analysis(
        OmegaConf.create(
            {
                "study_dirs": ["~/study"],
                "output_dir": "~/reports",
                "analysis_data_dir": "~/data",
                "grouping": "~/grouping.yaml",
                "protocol": "~/protocol.yaml",
                "selection": "~/choice.json",
                "evaluation_dirs": ["~/evaluation"],
            }
        )
    )
    assert calls[0]["grouping"] == str(tmp_path / "grouping.yaml")
    assert calls[0]["selection"] == str(tmp_path / "choice.json")
    assert calls[0]["output_dir"] == tmp_path / "reports"
    assert calls[1]["roots"] == [tmp_path / "study"]
    assert calls[1]["output_dir"] == tmp_path / "data"
    assert calls[1]["protocol"] == str(tmp_path / "protocol.yaml")
    assert calls[1]["evaluation_dirs"] == [str(tmp_path / "evaluation")]
