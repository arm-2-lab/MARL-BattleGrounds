"""Check declared study bookkeeping without running a scientific campaign.

Real trainer configs cover six methods and eighteen plain/treatment settings.
Small fake workers cover fixed case order, ordinary trainer dispatch, visible
failures, immutable clocks, explicit recovery, linked continuation identities,
backend-free status, duplicate owners and detached launch parity. They prove
host orchestration, not learner correctness, learning or device throughput.
Runtime bindings cover every input role, retain aliases and zero-weight members,
and reach fresh/resumed workers. Panel members and factory code are pinned too.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from marl_battlegrounds.training import study
from marl_battlegrounds.training._run_io import atomic_json

_REAL_SUPERVISE = study._launch.supervise_command
_SAVED_ID = "a" * 64
_CHILD_ID = "b" * 64


def _declaration() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "Small declared study",
        "cases": [
            {
                "id": "small",
                "config": {"method": "ff_ippo", "total_env_steps": 32},
                "seeds": [9, 4],
            }
        ],
    }


@pytest.fixture
def host_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def config(
        value: object, base: Path, seed: int, *, runtime: object = None
    ) -> dict[str, Any]:
        return copy.deepcopy(cast(dict[str, Any], value))

    def identities(methods: Sequence[str]) -> dict[str, Any]:
        return {method: {"source": "fixture"} for method in methods}

    monkeypatch.setattr(study, "_config", config)
    monkeypatch.setattr(study, "_freeze", study._normalize)
    monkeypatch.setattr(study, "_identities", identities)


def _fake_worker(
    monkeypatch: pytest.MonkeyPatch,
    codes: list[int],
    *,
    checkpoints: bool = False,
    analysis: bool = False,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def supervise(directory: Path, command: list[str], **kwargs: object) -> int:
        if "_analyze" in command:
            atomic_json(
                directory / "outcome.json",
                {"status": "failed", "error": "Optional plotting unavailable"},
            )
            return 1
        request = study._read(directory / "request.json")
        calls.append({**request, "deadline_at": kwargs.get("deadline_at")})
        code = codes.pop(0)
        run = directory / "run"
        run.mkdir(exist_ok=True)
        atomic_json(
            run / "status.json",
            {
                "status": "complete" if code == 0 else "interrupted",
                "env_steps": 32 if code == 0 else 16,
                "run_id": "fixture-run",
                "attempt_id": f"runner-{request['attempt']}",
                "study_attempt": {
                    "study_attempt_id": request["study_attempt_id"],
                    "request_sha256": study._digest(request),
                    "number": request["attempt"],
                    "case_id": request["job"]["id"],
                },
            },
        )
        if checkpoints:
            (run / "checkpoints" / _SAVED_ID).mkdir(parents=True, exist_ok=True)
            atomic_json(
                run / "latest_checkpoint.json",
                {
                    "schema_version": 1,
                    "checkpoint_id": _SAVED_ID,
                    "relative_path": f"checkpoints/{_SAVED_ID}",
                },
            )
        if analysis:
            atomic_json(run / "run_details.json", {"fake": True})
        atomic_json(
            directory / "outcome.json",
            {
                "status": "complete" if code == 0 else "failed",
                "request_sha256": study._digest(request),
            },
        )
        return code

    monkeypatch.setattr(study._launch, "supervise_command", supervise)
    return calls


def _method_config(method: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "method": method,
        "num_envs": 4,
        "total_env_steps": 256,
        "metrics": "none",
    }
    if method == "qmix":
        config["qmix"] = {
            "rollout_length": 8,
            "buffer_size": 64,
            "min_buffer_size": 32,
            "sample_sequence_length": 20,
            "sample_batch_size": 4,
            "epochs": 1,
            "update_period": 2,
        }
    elif method == "pqn_vdn":
        config["pqn"] = {
            "rollout_length": 4,
            "memory_window": 4,
            "epochs": 1,
            "num_minibatches": 2,
        }
    else:
        config["ppo"] = {
            "rollout_length": 4,
            "groups": 2,
            "minibatches": 2,
            "epochs": 1,
        }
    return config


def test_all_eighteen_settings_use_existing_config_owner_and_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    declared: dict[str, Any] = {"name": "All engineering settings", "cases": []}
    expected: list[tuple[str, bool, bool]] = []
    for method in ("mappo", "ippo", "qmix", "pqn_vdn", "ff_mappo", "ff_ippo"):
        treatments = (
            ((False, False),)
            if method.startswith("ff_")
            else ((False, False), (True, False), (False, True), (True, True))
        )
        for index, (curriculum, shaping) in enumerate(treatments):
            config = {
                **_method_config(method),
                "curriculum": curriculum,
                "shaping": shaping,
            }
            declared["cases"].append(
                {"id": f"{method}-{index}", "config": config, "seeds": [21]}
            )
            expected.append((method, curriculum, shaping))
    frozen = study._normalize(declared)

    def freeze(config: study.Declaration) -> dict[str, Any]:
        return frozen

    monkeypatch.setattr(study, "_freeze", freeze)
    calls = _fake_worker(monkeypatch, [0] * 18)
    report = study.run_study(declared, output_dir=tmp_path / "study")
    assert report["status"] == "complete"
    assert len(report["cases"]) == 18
    assert [
        (
            row["job"]["config"]["method"],
            row["job"]["config"]["curriculum"],
            row["job"]["config"]["shaping"],
        )
        for row in calls
    ] == expected
    assert all(row["job"]["config"]["seed"] == 21 for row in calls)


def test_fixed_seed_order_and_defaults_and_no_input_mutation(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    declared = _declaration()
    before = copy.deepcopy(declared)
    calls = _fake_worker(monkeypatch, [0, 0])
    root = tmp_path / "exact"
    result = study.run_study(declared, output_dir=root)
    assert declared == before
    assert [row["job"]["config"]["seed"] for row in calls] == [9, 4]
    assert study._read(root / "declaration.json")["failure_policy"] == "stop"
    assert result == study._read(root / "report.json")
    assert not study.study_status(root)["owned_live_records"]
    original = (root / "declaration.json").read_bytes()
    resumed = study.run_study(resume_from=root)
    assert resumed["status"] == "complete" and len(calls) == 2
    assert (root / "declaration.json").read_bytes() == original


@pytest.mark.parametrize(
    "change",
    [
        {"seeds": [1, 1]},
        {"seeds": [True]},
        {"seeds": []},
        {"id": "../bad"},
        {"duration_seconds": float("nan")},
        {"wat": 1},
    ],
)
def test_invalid_declaration_writes_no_study(
    host_only: None, tmp_path: Path, change: dict[str, Any]
) -> None:
    declared = _declaration()
    declared["cases"][0].update(change)
    with pytest.raises((ValueError, TypeError)):
        study.run_study(declared, output_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_config_seed_and_bad_method_are_checked_by_trainer(tmp_path: Path) -> None:
    for patch in ({"seed": 9}, {"method": "unknown"}, {"total_env_steps": 3}):
        declared = _declaration()
        declared["cases"][0]["config"] = {**_method_config("ff_ippo"), **patch}
        with pytest.raises((ValueError, TypeError)):
            study._normalize(declared)


def test_declaration_paths_resolve_beside_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    def inputs(case: dict[str, Any]) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(study, "_file_inputs", inputs)
    captured: list[dict[str, Any]] = []

    def config_read(value: dict[str, Any]) -> dict[str, Any]:
        captured.append(value)
        return value

    def config_save(value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    monkeypatch.setattr(
        "marl_battlegrounds.training.config.config_from_dict", config_read
    )
    monkeypatch.setattr(
        "marl_battlegrounds.training.config.config_to_dict", config_save
    )
    declared = _declaration()
    declared["cases"][0]["config"].update(
        validation_panel="panel.json",
        pinned_opponent="actors/policy",
        validation_opponents=["tdm-alpha", "example:factory", "./another"],
    )
    declaration = tmp_path / "declaration.json"
    atomic_json(declaration, declared)
    frozen = study._normalize(declaration)
    config = frozen["cases"][0]["config"]
    assert config["validation_panel"] == str(tmp_path / "panel.json")
    assert config["pinned_opponent"] == str(tmp_path / "actors/policy")
    assert config["validation_opponents"] == [
        "tdm-alpha",
        "example:factory",
        str(tmp_path / "another"),
    ]
    assert captured[0]["seed"] == 9


def test_expired_resume_keeps_both_original_clocks(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    now = [100.0]
    monkeypatch.setattr(study.time, "time", lambda: now[0])
    declared = _declaration()
    declared.update(duration_seconds=100)
    declared["cases"][0]["duration_seconds"] = 40
    calls = _fake_worker(monkeypatch, [124], checkpoints=True)
    root = tmp_path / "study"
    report = study.run_study(declared, output_dir=root)
    assert report["status"] == "incomplete"
    assert calls[0]["deadline_at"] == 140 - study._launch.cleanup_reserve_seconds()
    clock = (root / "launch_time.json").read_bytes()
    case_clock = (root / "cases/small--seed-9/launch_time.json").read_bytes()
    now[0] = 201.0
    resumed = study.run_study(resume_from=root)
    assert resumed["status"] == "time_limit"
    assert len(calls) == 1
    assert (root / "launch_time.json").read_bytes() == clock
    assert (root / "cases/small--seed-9/launch_time.json").read_bytes() == case_clock
    assert resumed["cases"][1]["status"] == "pending"


def test_explicit_resume_uses_saved_checkpoint_without_replacement_seed(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_worker(monkeypatch, [1, 0, 0], checkpoints=True)
    root = tmp_path / "study"
    first = study.run_study(_declaration(), output_dir=root)
    assert first["status"] == "incomplete"
    result = study.run_study(resume_from=root)
    assert result["status"] == "complete"
    assert [row["job"]["config"]["seed"] for row in calls] == [9, 9, 4]
    assert calls[1]["resume_from"] == str(
        root / f"cases/small--seed-9/run/checkpoints/{_SAVED_ID}"
    )
    assert [row["exit_code"] for row in result["cases"][0]["attempts"]] == [1, 0]


@pytest.mark.parametrize("run_folder", ["missing", "empty"])
def test_explicit_resume_restarts_only_an_unwritten_case_with_original_deadlines(
    host_only: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_folder: str,
) -> None:
    now = [100.0]
    monkeypatch.setattr(study.time, "time", lambda: now[0])
    declared = _declaration()
    declared["duration_seconds"] = 200
    declared["cases"][0]["duration_seconds"] = 100
    completed = _fake_worker(monkeypatch, [0, 0])
    finish = cast(Callable[..., int], study._launch.supervise_command)
    failed: list[dict[str, Any]] = []

    def before_training(directory: Path, command: list[str], **kwargs: object) -> int:
        if not failed:
            failed.append(study._read(directory / "request.json"))
            if run_folder == "empty":
                (directory / "run").mkdir()
            return 1
        return finish(directory, command, **kwargs)

    monkeypatch.setattr(study._launch, "supervise_command", before_training)
    root = tmp_path / "study"
    first = study.run_study(declared, output_dir=root)
    assert first["status"] == "incomplete"
    original_attempt = first["cases"][0]["attempts"][0]
    clock = (root / "launch_time.json").read_bytes()
    case_clock = (root / "cases/small--seed-9/launch_time.json").read_bytes()
    frozen = (root / "declaration.json").read_bytes()
    now[0] = 110.0
    report = study.run_study(resume_from=root)
    assert report["status"] == "complete"
    attempts = report["cases"][0]["attempts"]
    assert attempts[0] == original_attempt
    assert [row["number"] for row in attempts] == [1, 2]
    assert [row["exit_code"] for row in attempts] == [1, 0]
    assert completed[0]["resume_from"] is None
    assert [row["job"]["config"]["seed"] for row in [*failed, *completed]] == [9, 9, 4]
    assert completed[0]["job"] == failed[0]["job"]
    assert completed[0]["deadline_at"] == 200 - study._launch.cleanup_reserve_seconds()
    assert (root / "launch_time.json").read_bytes() == clock
    assert (root / "cases/small--seed-9/launch_time.json").read_bytes() == case_clock
    assert (root / "declaration.json").read_bytes() == frozen


def test_failure_without_checkpoint_stays_visible_and_is_not_retried(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    declared = _declaration()
    declared["failure_policy"] = "continue"
    calls = _fake_worker(monkeypatch, [1, 0])
    root = tmp_path / "study"
    first = study.run_study(declared, output_dir=root)
    assert [row["status"] for row in first["cases"]] == ["failed", "complete"]
    run = root / "cases/small--seed-9/run"
    partial = (run / "status.json").read_bytes()
    attempts = copy.deepcopy(first["cases"][0]["attempts"])
    second = study.run_study(resume_from=root)
    assert second["status"] == "incomplete" and len(calls) == 2
    error = second["cases"][0]["error"]
    assert "not restarted" in error and str(run) in error
    assert "Restore a complete checkpoint" in error
    assert "new study in another output folder" in error
    assert second["cases"][0]["attempts"] == attempts
    assert (run / "status.json").read_bytes() == partial


def test_analysis_failure_preserves_successful_cases(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_worker(monkeypatch, [0, 0], analysis=True)
    report = study.run_study(_declaration(), output_dir=tmp_path / "study")
    assert report["status"] == "complete"
    assert all(row["status"] == "complete" for row in report["cases"])
    assert report["analysis"]["status"] == "failed"


def test_changed_declaration_or_clock_refused(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_worker(monkeypatch, [0, 0])
    root = tmp_path / "study"
    study.run_study(_declaration(), output_dir=root)
    changed = _declaration()
    changed["cases"][0]["seeds"] = [10]
    with pytest.raises(ValueError, match="differs"):
        study.run_study(changed, resume_from=root)
    clock = study._read(root / "launch_time.json")
    clock["declaration_sha256"] = "changed"
    atomic_json(root / "launch_time.json", clock)
    with pytest.raises(ValueError, match="clock"):
        study.run_study(resume_from=root)


def test_linked_finalist_pins_parent_and_leaves_discovery_unchanged(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_worker(monkeypatch, [0, 0])
    discovery = tmp_path / "discovery"
    study.run_study(_declaration(), output_dir=discovery)
    original = (discovery / "declaration.json").read_bytes()
    parent = discovery / f"cases/small--seed-9/run/checkpoints/{_SAVED_ID}"

    def parent_identity(path: str) -> dict[str, Any]:
        return {
            "checkpoint_id": _SAVED_ID,
            "description_sha256": "parent",
            "method": "ff_ippo",
        }

    monkeypatch.setattr(study, "_checkpoint_identity", parent_identity)
    finalist: dict[str, Any] = {
        "name": "Finalists",
        "finalist_of": str(discovery),
        "cases": [
            {
                "id": "longer",
                "checkpoint": str(parent),
                "additional_env_steps": 32,
                "changes": {"learning_rate": {"keep_final": True}},
            }
        ],
    }
    frozen = study._normalize(finalist)
    assert frozen["cases"][0]["parent_identity"]["description_sha256"] == "parent"
    assert frozen["finalist_of"]["declaration_sha256"] == study._digest(
        study._read(discovery / "declaration.json")
    )
    assert (discovery / "declaration.json").read_bytes() == original
    finalist["cases"][0]["checkpoint"] = str(
        tmp_path / f"elsewhere/checkpoints/{_SAVED_ID}"
    )
    with pytest.raises(ValueError, match="linked study"):
        study._normalize(finalist)


def test_worker_calls_ordinary_train_and_checks_frozen_request(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import marl_battlegrounds.training.runner as runner

    root, frozen, _ = study._prepare(_declaration(), tmp_path / "study", None)
    job = study._jobs(frozen)[0]
    directory = root / "cases" / job["id"]
    directory.mkdir()
    request: dict[str, Any] = {"job": job, "resume_from": None, "attempt": 1}
    atomic_json(directory / "request.json", request)
    called: list[Any] = []

    def config_read(value: dict[str, Any]) -> dict[str, Any]:
        return value

    def train(config: object = None, **kwargs: object) -> SimpleNamespace:
        called.append((config, kwargs))
        return SimpleNamespace(
            status="complete",
            run_dir=directory / "run",
            completed_env_steps=32,
            completed_updates=1,
            final_actor=directory / "actor",
            selected_actor=None,
        )

    monkeypatch.setattr(
        "marl_battlegrounds.training.config.config_from_dict", config_read
    )
    monkeypatch.setattr(runner, "train", train)
    assert study._worker(root, job["id"]) == 0
    assert called[0][0] == job["config"]
    assert called[0][1] == {"output_dir": directory / "run"}
    assert study._read(directory / "outcome.json")["request_sha256"] == study._digest(
        request
    )
    request["job"]["config"]["seed"] = 100
    atomic_json(directory / "request.json", request)
    assert study._worker(root, job["id"]) == 1 and len(called) == 1


def test_start_uses_shared_launch_with_same_saved_declaration(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Any] = []

    def spawn(root: Path, command: list[str], **kwargs: object) -> dict[str, Any]:
        calls.append((root, command, kwargs))
        return {"state": "starting"}

    monkeypatch.setattr(study._launch, "spawn_detached", spawn)
    root = tmp_path / "study"
    assert study.start_study(_declaration(), output_dir=root) == {"state": "starting"}
    assert calls[0][1] == study._command("_control", str(root))
    assert calls[0][2] == {"log_path": root / "logs/study.log", "mode": "start"}
    before = (root / "launch_time.json").read_bytes()
    study.start_study(resume_from=root)
    assert "--resume" in calls[1][1]
    assert (root / "launch_time.json").read_bytes() == before


def test_duplicate_foreground_lock_refused_immediately(
    host_only: None, tmp_path: Path
) -> None:
    root, _, _ = study._prepare(_declaration(), tmp_path / "study", None)
    with study._launch.launch_lock(root), pytest.raises(BlockingIOError):
        study.run_study(resume_from=root)


def test_status_and_stop_import_no_numerical_backend(
    host_only: None, tmp_path: Path
) -> None:
    root, _, _ = study._prepare(_declaration(), tmp_path / "study", None)
    code = (
        "from marl_battlegrounds.training.study import study_status,stop_study; "
        "import sys; study_status(sys.argv[1]); stop_study(sys.argv[1]); "
        "assert not any(x in sys.modules for x in "
        "('jax','numpy','flax','marl_battlegrounds.training.runner'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        env={**os.environ, "JAX_PLATFORMS": "unavailable"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cpu_declaration_subprocess_does_not_initialize_caller_backend(
    tmp_path: Path,
) -> None:
    declared = _declaration()
    declared["cases"][0]["config"] = _method_config("ff_ippo")
    path = tmp_path / "declare.json"
    atomic_json(path, declared)
    code = (
        "from marl_battlegrounds.training.study import _freeze; import sys; "
        "result=_freeze(sys.argv[1]); "
        "assert result['cases'][0]['config']['method']=='ff_ippo'; "
        "assert 'jax' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        env={**os.environ, "JAX_PLATFORMS": "unavailable"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_prepared_but_never_started_study_gets_first_clock_on_resume(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _, _ = study._prepare(_declaration(), tmp_path / "study", None)
    assert not (root / "launch_time.json").exists()
    calls = _fake_worker(monkeypatch, [0, 0])
    report = study.run_study(resume_from=root)
    assert report["status"] == "complete" and len(calls) == 2


@pytest.mark.parametrize(
    "lost", ["launch_time.json", "cases/small--seed-9/launch_time.json"]
)
def test_started_study_cannot_replace_a_lost_clock(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lost: str
) -> None:
    calls = _fake_worker(monkeypatch, [1], checkpoints=True)
    root = tmp_path / "study"
    study.run_study(_declaration(), output_dir=root)
    (root / lost).unlink()
    with pytest.raises(ValueError, match=r"clock|missing"):
        study.run_study(resume_from=root)
    assert len(calls) == 1
    assert not (root / lost).exists()


def test_stale_outcome_cannot_finish_a_new_attempt(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_worker(monkeypatch, [1], checkpoints=True)
    root = tmp_path / "study"
    study.run_study(_declaration(), output_dir=root)
    directory = root / "cases/small--seed-9"
    old_request = study._read(directory / "request.json")
    atomic_json(
        directory / "outcome.json",
        {"status": "complete", "request_sha256": study._digest(old_request)},
    )
    atomic_json(directory / "run/status.json", {"status": "complete"})

    def missing_outcome(directory: Path, command: list[str], **kwargs: object) -> int:
        return 0

    monkeypatch.setattr(study._launch, "supervise_command", missing_outcome)
    report = study.run_study(resume_from=root)
    assert report["status"] == "incomplete"
    assert report["cases"][0]["status"] == "failed"
    assert "exact attempt" in report["cases"][0]["attempts"][-1]["error"]


def test_worker_refuses_different_installed_source_before_training(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, frozen, _ = study._prepare(_declaration(), tmp_path / "study", None)
    job = study._jobs(frozen)[0]
    directory = root / "cases" / job["id"]
    directory.mkdir()
    atomic_json(
        directory / "request.json", {"job": job, "resume_from": None, "attempt": 1}
    )

    def changed(methods: Sequence[str]) -> dict[str, Any]:
        return {method: {"source": "changed"} for method in methods}

    monkeypatch.setattr(study, "_identities", changed)
    assert study._worker(root, job["id"]) == 1
    assert not (directory / "run").exists()
    assert "source or dependencies" in study._read(directory / "outcome.json")["error"]


def test_finished_real_supervisor_does_not_keep_foreground_python_live(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    assert study._launch.supervise_command(job, [sys.executable, "-c", "pass"]) == 0
    before = study._read(job / "process.json")
    assert study._launch._alive(before)
    study._finished_supervisor(job)
    saved = study._read(job / "process.json")
    assert saved["process"] is None
    assert saved["supervisor_identity"] == before["process"]
    assert saved["trainer"] == before["trainer"]
    assert saved["trainer_group"] == before["trainer_group"]
    assert not study._launch._alive(saved)


def test_finished_supervisor_keeps_unfinished_worker_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    own = study.process_identity()
    record = {
        "state": "exited",
        "process": own,
        "trainer_group": own,
        "cleanup": {"remaining_pids": [own["pid"]]},
    }
    atomic_json(job / "process.json", record)

    def members(group: int) -> list[int]:
        return [int(own["pid"])]

    monkeypatch.setattr(study._launch, "_group_members", members)
    study._finished_supervisor(job)
    assert study._launch._alive(study._read(job / "process.json"))


def test_continuation_worker_calls_existing_extension_and_pins_parent(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import marl_battlegrounds.training.runner as runner

    identity = {
        "checkpoint_id": _SAVED_ID,
        "description_sha256": "fixed",
        "method": "ff_ippo",
    }

    def parent_identity(path: str) -> dict[str, str]:
        return identity

    monkeypatch.setattr(study, "_checkpoint_identity", parent_identity)
    declared = {
        "name": "Extension",
        "cases": [
            {
                "id": "child",
                "checkpoint": str(tmp_path / f"parent/checkpoints/{_SAVED_ID}"),
                "additional_env_steps": 32,
            }
        ],
    }
    root, frozen, _ = study._prepare(declared, tmp_path / "study", None)
    job = study._jobs(frozen)[0]
    directory = root / "cases/child"
    directory.mkdir()
    atomic_json(
        directory / "request.json", {"job": job, "resume_from": None, "attempt": 1}
    )
    called: list[object] = []

    def extend(checkpoint: str, **kwargs: object) -> SimpleNamespace:
        called.append((checkpoint, kwargs))
        return SimpleNamespace(
            status="complete",
            run_dir=directory / "run",
            completed_env_steps=64,
            completed_updates=2,
            final_actor=directory / "actor",
            selected_actor=None,
        )

    monkeypatch.setattr(runner, "extend_training", extend)
    assert study._worker(root, "child") == 0
    assert called == [
        (
            job["checkpoint"],
            {
                "additional_env_steps": 32,
                "output_dir": directory / "run",
                "changes": {},
            },
        )
    ]
    identity["description_sha256"] = "changed"
    assert study._worker(root, "child") == 1
    assert len(called) == 1
    assert "parent differs" in study._read(directory / "outcome.json")["error"]


def test_finalist_relative_parent_link_matches_absolute_checkpoint(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_worker(monkeypatch, [0, 0])
    discovery = tmp_path / "discovery"
    study.run_study(_declaration(), output_dir=discovery)
    folder = tmp_path / "finalist"
    folder.mkdir()

    def parent_identity(path: str) -> dict[str, str]:
        return {
            "checkpoint_id": _SAVED_ID,
            "description_sha256": "fixed",
            "method": "ff_ippo",
        }

    monkeypatch.setattr(study, "_checkpoint_identity", parent_identity)
    declaration = {
        "name": "Extension",
        "finalist_of": "../discovery",
        "cases": [
            {
                "id": "child",
                "checkpoint": str(
                    discovery / f"cases/small--seed-9/run/checkpoints/{_SAVED_ID}"
                ),
                "additional_env_steps": 32,
            }
        ],
    }
    atomic_json(folder / "declaration.json", declaration)
    frozen = study._normalize(folder / "declaration.json")
    assert frozen["finalist_of"]["study_dir"] == str(discovery)


def test_real_detached_expired_study_starts_no_case_worker(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    declared = _declaration()
    declared["duration_seconds"] = 0.001
    monkeypatch.setenv("JAX_PLATFORMS", "unavailable")
    root = tmp_path / "study"
    process = study.start_study(declared, output_dir=root)
    try:
        while True:
            state = study.study_status(root)
            if not state["owned_live_records"]:
                # Read once more after exit, in case the child just published.
                state = study.study_status(root)
                assert state["report"], (root / "logs/study.log").read_text()
                break
            time.sleep(0.02)
    finally:
        if study._launch._alive({"process": process["process"]}):
            study.stop_study(root)
        with suppress(ChildProcessError):
            os.waitpid(process["process"]["pid"], 0)
    assert process["state"] == "starting"
    assert state["report"]["status"] == "time_limit"
    assert all(row["status"] == "pending" for row in state["report"]["cases"])
    assert not list((root / "cases").iterdir())


def test_leftover_worker_blocks_next_case_even_with_continue_policy(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    declared = _declaration()
    declared["failure_policy"] = "continue"
    calls = _fake_worker(monkeypatch, [1])
    original = study._launch.live_process_records

    def live(paths: Sequence[Path]) -> list[Path]:
        if len(paths) == 1 and calls:
            return list(paths)
        return original(paths)

    monkeypatch.setattr(study._launch, "live_process_records", live)
    report = study.run_study(declared, output_dir=tmp_path / "study")
    assert len(calls) == 1
    assert report["status"] == "incomplete"
    assert "cleanup is incomplete" in report["cases"][0]["error"]
    assert report["cases"][1]["status"] == "pending"


def test_analysis_nonzero_exit_cannot_reuse_prior_success(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_worker(monkeypatch, [0, 0], analysis=True)
    root = tmp_path / "study"
    study.run_study(_declaration(), output_dir=root)
    atomic_json(
        root / "analysis_job/outcome.json", {"status": "complete", "report": "stale"}
    )

    def failure(directory: Path, command: list[str], **kwargs: object) -> int:
        return 1

    monkeypatch.setattr(study._launch, "supervise_command", failure)
    report = study.run_study(resume_from=root)
    assert report["status"] == "complete"
    assert report["analysis"]["status"] == "failed"
    assert report["analysis"]["exit_code"] == 1


def test_completed_analysis_releases_foreground_supervisor_role(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    declared = _declaration()
    root, _, _ = study._prepare(declared, tmp_path / "study", None)
    calls = _fake_worker(monkeypatch, [0, 0], analysis=True)
    fake = cast(Callable[..., int], study._launch.supervise_command)
    real = _REAL_SUPERVISE

    def supervise(directory: Path, command: list[str], **kwargs: object) -> int:
        if "_analyze" not in command:
            return fake(directory, command, **kwargs)
        program = (
            "import json,pathlib,sys; "
            "p=pathlib.Path(sys.argv[1]); "
            "(p/'outcome.json').write_text(json.dumps({'status':'complete'}))"
        )
        return real(directory, [sys.executable, "-c", program, str(directory)])

    monkeypatch.setattr(study._launch, "supervise_command", supervise)
    report = study.run_study(resume_from=root)
    assert report["status"] == report["analysis"]["status"] == "complete"
    assert len(calls) == 2
    process = study._read(root / "analysis_job/process.json")
    assert process["process"] is None and process["supervisor_identity"]
    assert not study.study_status(root)["owned_live_records"]


@pytest.mark.parametrize("handled_exit", [None, 0, 7])
def test_public_stop_never_starts_next_seed_under_continue_policy(
    host_only: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    handled_exit: int | None,
) -> None:
    root = tmp_path / "study"
    marker = tmp_path / "second-seed-started"
    declaration = _declaration()
    declaration["failure_policy"] = "continue"

    def command(*arguments: str) -> list[str]:
        if arguments[2].endswith("seed-9"):
            handler = (
                ""
                if handled_exit is None
                else (
                    "signal.signal(signal.SIGTERM, "
                    f"lambda *_: sys.exit({handled_exit})); "
                )
            )
            program = (
                "import signal,sys,time; "
                "from marl_battlegrounds.training.study import stop_study; "
                + handler
                + "stop_study(sys.argv[1]); time.sleep(30)"
            )
            return [sys.executable, "-c", program, str(root)]
        return [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
            str(marker),
        ]

    monkeypatch.setattr(study, "_command", command)
    result = study.run_study(declaration, output_dir=root)
    assert result["status"] == "stopped"
    assert len(result["cases"][0]["attempts"]) == 1
    assert result["cases"][1]["status"] == "pending"
    assert not marker.exists()
    process = study._read(root / "cases/small--seed-9/process.json")
    assert process["stop_reason"] == "signal" and process["stop_signal"] == 15
    assert not study.study_status(root)["owned_live_records"]


@pytest.mark.parametrize("handled_exit", [0, 7])
def test_public_stop_during_deadline_cleanup_never_starts_next_seed(
    host_only: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    handled_exit: int,
) -> None:
    root = tmp_path / "study"
    marker = tmp_path / "second-seed-started"
    declaration = _declaration()
    declaration["failure_policy"] = "continue"
    declaration["cases"][0]["duration_seconds"] = (
        study._launch.cleanup_reserve_seconds() + 2.0
    )

    def command(*arguments: str) -> list[str]:
        if arguments[2].endswith("seed-9"):
            program = (
                "import signal,sys,time\n"
                "from marl_battlegrounds.training.study import stop_study\n"
                "def stop(*_):\n"
                "    stop_study(sys.argv[1])\n"
                f"    sys.exit({handled_exit})\n"
                "signal.signal(signal.SIGTERM,stop)\n"
                "time.sleep(30)\n"
            )
            return [sys.executable, "-c", program, str(root)]
        return [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
            str(marker),
        ]

    monkeypatch.setattr(study, "_command", command)
    result = study.run_study(declaration, output_dir=root)
    assert result["status"] == "stopped"
    assert result["cases"][1]["status"] == "pending"
    assert not marker.exists()
    process = study._read(root / "cases/small--seed-9/process.json")
    assert process["stop_reason"] == "signal" and process["stop_signal"] == 15
    assert process["finished_at"] > process["started_at"]
    assert process["cleanup"]["trainer_exit_code"] == handled_exit
    assert process["exit_code"] != 124
    assert "deadline_at" in process
    assert not study.study_status(root)["owned_live_records"]


def test_public_stop_during_analysis_stays_visible(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_worker(monkeypatch, [0, 0], analysis=True)
    fake = cast(Callable[..., int], study._launch.supervise_command)
    root = tmp_path / "study"

    def supervise(directory: Path, command: list[str], **kwargs: object) -> int:
        if "_analyze" not in command:
            return fake(directory, command, **kwargs)
        program = (
            "import sys,time; "
            "from marl_battlegrounds.training.study import stop_study; "
            "stop_study(sys.argv[1]); time.sleep(30)"
        )
        return _REAL_SUPERVISE(directory, [sys.executable, "-c", program, str(root)])

    monkeypatch.setattr(study._launch, "supervise_command", supervise)
    result = study.run_study(_declaration(), output_dir=root)
    assert result["status"] == result["analysis"]["status"] == "stopped"
    assert all(row["status"] == "complete" for row in result["cases"])
    assert not study.study_status(root)["owned_live_records"]


@pytest.mark.parametrize("scope", ["study", "case"])
def test_cleanup_reserve_leaves_insufficient_time_to_start_worker(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scope: str
) -> None:
    declaration = _declaration()
    reserve = study._launch.cleanup_reserve_seconds()
    target = declaration if scope == "study" else declaration["cases"][0]
    target["duration_seconds"] = reserve - 1
    calls = _fake_worker(monkeypatch, [])
    root = tmp_path / "study"
    result = study.run_study(declaration, output_dir=root)
    assert not calls
    assert result["status"] != "complete"
    original_clock = (root / "launch_time.json").read_bytes()
    resumed = study.run_study(resume_from=root)
    assert not calls and resumed["status"] != "complete"
    assert (root / "launch_time.json").read_bytes() == original_clock


@pytest.mark.parametrize("panel_directory", [False, True])
def test_different_valid_panel_refused_before_pending_seed_or_resume(
    host_only: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    panel_directory: bool,
) -> None:
    from marl_battlegrounds.training import runner, validation

    panel = validation.create_panel(
        opponents=("tdm-alpha",), output_dir=tmp_path / "panel"
    )
    declaration = _declaration()
    declaration["cases"][0]["config"]["validation_panel"] = str(
        panel.path.parent if panel_directory else panel.path
    )
    root, frozen, _ = study._prepare(declaration, tmp_path / "study", None)
    jobs = study._jobs(frozen)
    calls: list[dict[str, Any]] = []

    def config_read(value: dict[str, Any]) -> dict[str, Any]:
        return value

    def train(config: object = None, **kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return SimpleNamespace(
            status="complete",
            run_dir=tmp_path / "fake-run",
            completed_env_steps=32,
            completed_updates=1,
            final_actor=tmp_path / "actor",
            selected_actor=None,
        )

    monkeypatch.setattr(
        "marl_battlegrounds.training.config.config_from_dict", config_read
    )
    monkeypatch.setattr(runner, "train", train)
    for job in jobs:
        directory = root / "cases" / job["id"]
        directory.mkdir()
        atomic_json(
            directory / "request.json", {"job": job, "resume_from": None, "attempt": 1}
        )
    assert study._worker(root, jobs[0]["id"]) == 0
    original = panel.path.read_bytes()
    changed = study._read(panel.path)
    changed["roots"]["confirmation"] += 1
    changed["panel_digest"] = validation._panel_digest(changed)
    atomic_json(panel.path, changed)
    assert validation.load_panel(panel.path).digest != panel.digest
    assert study._worker(root, jobs[1]["id"]) == 1
    second = root / "cases" / jobs[1]["id"]
    run = second / "run"
    (run / f"checkpoints/{_SAVED_ID}").mkdir(parents=True)
    atomic_json(
        run / "latest_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_id": _SAVED_ID,
            "relative_path": f"checkpoints/{_SAVED_ID}",
        },
    )
    atomic_json(
        second / "request.json",
        {
            "job": jobs[1],
            "resume_from": str(run / f"checkpoints/{_SAVED_ID}"),
            "attempt": 2,
        },
    )
    assert study._worker(root, jobs[1]["id"]) == 1
    assert len(calls) == 1
    panel.path.write_bytes(original)
    assert study._worker(root, jobs[1]["id"]) == 0
    assert calls[-1] == {"resume_from": str(run / f"checkpoints/{_SAVED_ID}")}


def test_external_actor_and_result_inputs_use_existing_description_owner(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from marl_battlegrounds import _method_loading
    from marl_battlegrounds.training import _checkpoint_metadata as checkpoints

    def factory() -> object:
        pytest.fail("Declaration binding must not call the factory")

    def installed(reference: str) -> Callable[[], object]:
        return factory

    monkeypatch.setattr(_method_loading, "installed_callable", installed)

    actor = tmp_path / "actor"
    actor.mkdir()
    random_result = tmp_path / "initial.json"
    atomic_json(random_result, {"result": "first"})
    description: dict[str, Any] = {
        "checkpoint_id": "first",
        "files": {"weights": "digest"},
    }
    reads: list[str] = []

    def read_description(path: str) -> dict[str, Any]:
        reads.append(str(path))
        return dict(description)

    monkeypatch.setattr(checkpoints, "read_checkpoint_description", read_description)
    case = {
        "config": {
            "validation_opponents": [str(actor), "module:factory", "tdm-alpha"],
            "pinned_opponent": str(actor),
            "slot_diagnostic_actor": str(actor),
            "random_initialization_result": str(random_result),
        }
    }
    first = study._file_inputs(case)
    assert reads == [str(actor)]
    assert set(first) == {str(actor), str(random_result), "module:factory", "tdm-alpha"}
    assert first["module:factory"]["external_state"] == "unknown"
    description["checkpoint_id"] = "second"
    second = study._file_inputs(case)
    assert first[str(actor)] != second[str(actor)]
    assert first[str(random_result)] == second[str(random_result)]
    atomic_json(random_result, {"result": "second"})
    assert study._file_inputs(case)[str(random_result)] != first[str(random_result)]


def test_continuation_replacement_panel_is_bound_before_extension(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from marl_battlegrounds.training import runner, validation

    panel = tmp_path / "future-panel.json"
    content: dict[str, Any] = {
        "schema_version": 2,
        "members": [{"reference": "random"}],
    }
    content["panel_digest"] = validation._panel_digest(content)
    atomic_json(panel, content)

    def parent_identity(path: str) -> dict[str, str]:
        return {
            "checkpoint_id": _SAVED_ID,
            "description_sha256": "fixed",
            "method": "ff_ippo",
        }

    def forbidden(checkpoint: str, **kwargs: object) -> None:
        pytest.fail("A changed panel reached the extension owner")

    monkeypatch.setattr(study, "_checkpoint_identity", parent_identity)
    monkeypatch.setattr(runner, "extend_training", forbidden)
    declaration = {
        "name": "Extension",
        "cases": [
            {
                "id": "child",
                "checkpoint": str(tmp_path / f"parent/checkpoints/{_SAVED_ID}"),
                "additional_env_steps": 32,
                "changes": {"validation": {"panel": str(panel)}},
            }
        ],
    }
    root, frozen, _ = study._prepare(declaration, tmp_path / "study", None)
    job = study._jobs(frozen)[0]
    directory = root / "cases/child"
    directory.mkdir()
    atomic_json(
        directory / "request.json", {"job": job, "resume_from": None, "attempt": 1}
    )
    content["members"] = [{"reference": "tdm-alpha"}]
    content["panel_digest"] = validation._panel_digest(content)
    atomic_json(panel, content)
    assert study._worker(root, "child") == 1
    assert "external file input" in study._read(directory / "outcome.json")["error"]


@pytest.mark.parametrize("detached", [False, True])
def test_resume_after_declaration_publication_restores_only_unstarted_folders(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, detached: bool
) -> None:
    root = tmp_path / "study"
    root.mkdir()
    atomic_json(root / "declaration.json", study._normalize(_declaration()))
    calls = _fake_worker(monkeypatch, [0, 0])

    def spawn(root: Path, command: list[str], **kwargs: object) -> dict[str, Any]:
        return {"state": "starting"}

    monkeypatch.setattr(study._launch, "spawn_detached", spawn)
    result = (
        study.start_study(resume_from=root)
        if detached
        else study.run_study(resume_from=root)
    )
    assert result["state" if detached else "status"] == (
        "starting" if detached else "complete"
    )
    assert (root / "cases").is_dir() and (root / "logs").is_dir()
    assert len(calls) == (0 if detached else 2)


def test_worker_passes_frozen_config_to_ordinary_resume_guard(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from marl_battlegrounds.training import runner

    root, frozen, _ = study._prepare(_declaration(), tmp_path / "study", None)
    job = study._jobs(frozen)[0]
    directory = root / "cases" / job["id"]
    run = directory / "run"
    checkpoint = run / f"checkpoints/{_SAVED_ID}"
    checkpoint.mkdir(parents=True)
    atomic_json(
        run / "latest_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_id": _SAVED_ID,
            "relative_path": f"checkpoints/{_SAVED_ID}",
        },
    )
    atomic_json(
        directory / "request.json",
        {"job": job, "resume_from": str(checkpoint), "attempt": 2},
    )
    supplied: list[object] = []

    def config_read(value: dict[str, Any]) -> dict[str, Any]:
        return value

    def train(config: object = None, **kwargs: object) -> None:
        supplied.append(config)
        raise ValueError("Saved config differs from the supplied frozen seed")

    monkeypatch.setattr(
        "marl_battlegrounds.training.config.config_from_dict", config_read
    )
    monkeypatch.setattr(runner, "train", train)
    assert study._worker(root, job["id"]) == 1
    assert supplied == [job["config"]]


@pytest.mark.parametrize(
    "field", ["parent_checkpoint_id", "additional_env_steps", "changes"]
)
def test_continuation_resume_refuses_another_declared_child(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    from marl_battlegrounds.training import checkpoints, runner

    def parent_identity(path: str) -> dict[str, str]:
        return {
            "checkpoint_id": "parent",
            "description_sha256": "fixed",
            "method": "ff_ippo",
        }

    monkeypatch.setattr(study, "_checkpoint_identity", parent_identity)
    declared = {
        "name": "Extension",
        "cases": [
            {
                "id": "child",
                "checkpoint": str(tmp_path / "parent/checkpoints/parent"),
                "additional_env_steps": 32,
            }
        ],
    }
    root, frozen, _ = study._prepare(declared, tmp_path / "study", None)
    job = study._jobs(frozen)[0]
    directory = root / "cases/child"
    run = directory / "run"
    checkpoint = run / f"checkpoints/{_CHILD_ID}"
    checkpoint.mkdir(parents=True)
    atomic_json(
        run / "latest_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_id": _CHILD_ID,
            "relative_path": f"checkpoints/{_CHILD_ID}",
        },
    )
    atomic_json(
        directory / "request.json",
        {"job": job, "resume_from": str(checkpoint), "attempt": 2},
    )
    context: dict[str, Any] = {
        "parent_checkpoint": job["checkpoint"],
        "parent_checkpoint_id": "parent",
        "additional_env_steps": 32,
        "changes": {},
    }
    context[field] = "different"

    def read_description(path: str) -> dict[str, Any]:
        return {"metadata": {"continuation": context}}

    def train(**kwargs: object) -> None:
        pytest.fail("An unrelated child reached ordinary resume")

    monkeypatch.setattr(checkpoints, "read_checkpoint_description", read_description)
    monkeypatch.setattr(runner, "train", train)
    assert study._worker(root, "child") == 1
    assert (
        "declared parent or extension"
        in study._read(directory / "outcome.json")["error"]
    )


def test_symlinked_case_never_receives_a_request_or_changes_external_files(
    host_only: None, tmp_path: Path
) -> None:
    root, frozen, _ = study._prepare(_declaration(), tmp_path / "study", None)
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_text("Keep")
    identifier = study._jobs(frozen)[0]["id"]
    (root / "cases" / identifier).symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="link"):
        study.run_study(resume_from=root)
    assert {path.name for path in external.iterdir()} == {"sentinel"}
    assert (external / "sentinel").read_text() == "Keep"


def test_worker_refuses_linked_run_before_ordinary_resume(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from marl_battlegrounds.training import runner

    root, frozen, _ = study._prepare(_declaration(), tmp_path / "study", None)
    job = study._jobs(frozen)[0]
    directory = root / "cases" / job["id"]
    directory.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (directory / "run").symlink_to(external, target_is_directory=True)
    atomic_json(
        directory / "request.json",
        {
            "job": job,
            "resume_from": str(directory / f"run/checkpoints/{_SAVED_ID}"),
            "attempt": 2,
        },
    )

    def train(**kwargs: object) -> None:
        pytest.fail("A linked external run reached ordinary resume")

    monkeypatch.setattr(runner, "train", train)
    assert study._worker(root, job["id"]) == 1
    assert "link" in study._read(directory / "outcome.json")["error"]
    assert not list(external.iterdir())


def test_latest_checkpoint_cannot_follow_an_external_directory_link(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (run / f"checkpoints/{_SAVED_ID}").symlink_to(external, target_is_directory=True)
    atomic_json(
        run / "latest_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_id": _SAVED_ID,
            "relative_path": f"checkpoints/{_SAVED_ID}",
        },
    )
    with pytest.raises(ValueError, match="link"):
        study._latest(run)


def test_explicit_colon_folder_is_bound_and_factory_and_builtin_stay_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from marl_battlegrounds.training import _checkpoint_metadata as checkpoints

    actor = tmp_path / "model:version"
    actor.mkdir()
    (tmp_path / "random").mkdir()
    called: list[str] = []

    def description(path: str) -> dict[str, Any]:
        called.append(str(path))
        return {"checkpoint_id": "fixed", "files": {}}

    monkeypatch.setattr(checkpoints, "read_checkpoint_description", description)
    assert study._reference("./model:version", tmp_path) == str(actor)
    assert study._reference("model:version", tmp_path) == "model:version"
    assert study._reference(str(actor), tmp_path) == str(actor)
    assert study._reference("module:factory", tmp_path) == "module:factory"
    assert study._reference("random", tmp_path) == "random"
    result = study._file_inputs({"config": {"validation_opponents": [str(actor)]}})
    assert set(result) == {str(actor)} and called == [str(actor)]


def test_continuation_binding_folders_resolve_beside_declaration(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from marl_battlegrounds import _method_loading
    from marl_battlegrounds.training import _checkpoint_metadata as checkpoints

    def factory() -> object:
        pytest.fail("Declaration binding must not call the factory")

    def installed(reference: str) -> Callable[[], object]:
        return factory

    monkeypatch.setattr(_method_loading, "installed_callable", installed)

    actor = tmp_path / "model:version"
    actor.mkdir()
    from marl_battlegrounds.training import validation

    panel: dict[str, Any] = {
        "schema_version": 2,
        "members": [{"reference": "random"}, {"reference": "tdm-alpha"}],
    }
    panel["panel_digest"] = validation._panel_digest(panel)
    atomic_json(tmp_path / "panel.json", panel)

    def parent_identity(path: str) -> dict[str, str]:
        return {
            "checkpoint_id": "parent",
            "description_sha256": "fixed",
            "method": "ff_ippo",
        }

    def description(path: str) -> dict[str, Any]:
        assert str(path) == str(actor)
        return {"checkpoint_id": "actor", "files": {}}

    monkeypatch.setattr(study, "_checkpoint_identity", parent_identity)
    monkeypatch.setattr(checkpoints, "read_checkpoint_description", description)
    declared = {
        "name": "Extension",
        "cases": [
            {
                "id": "child",
                "checkpoint": f"parent/checkpoints/{_SAVED_ID}",
                "additional_env_steps": 32,
                "changes": {
                    "validation": {
                        "panel": "panel.json",
                        "bindings": ["./model:version", "example:factory"],
                    }
                },
            }
        ],
    }
    path = tmp_path / "declaration.json"
    atomic_json(path, declared)
    frozen = study._normalize(path)
    child = frozen["cases"][0]
    assert child["changes"]["validation"]["bindings"] == [str(actor), "example:factory"]
    assert set(child["external_inputs"]) == {
        str(actor),
        str(tmp_path / "panel.json"),
        "example:factory",
    }


def test_runtime_bindings_preserve_aliases_and_all_input_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds import _method_loading

    (tmp_path / "friend").mkdir()
    runtime = study._runtime_paths(
        {
            "opponents": {"friend": "./actor one", "unused": "tdm-gamma"},
            "partners": {"helper": "./actor two"},
            "validation_partners": {"fresh": "client:make"},
            "validation_opponents": ["./validator", "tdm-beta"],
        },
        tmp_path,
    )
    config = study._config_paths(
        {
            "initial_actor": "./initial",
            "opponents": {"self": 0.6, "past": 0.2, "friend": 0.2, "unused": 0.0},
            "partners": ["helper", "helper"],
            "validation_partners": {"extra": "./extra"},
        },
        tmp_path,
        runtime=runtime,
    )
    assert config["opponents"] == {
        "self": 0.6,
        "past": 0.2,
        "friend": 0.2,
        "unused": 0.0,
    }
    assert config["partners"] == ["helper", "helper"]
    assert runtime["opponents"]["friend"] == str(tmp_path / "actor one")
    called: list[str] = []

    def identity(reference: str) -> dict[str, str]:
        called.append(reference)
        return {"checked": reference}

    monkeypatch.setattr(_method_loading, "_reference_identity", identity)
    inputs = study._file_inputs({"config": config, "runtime": runtime})
    expected = {
        str(tmp_path / name)
        for name in ("initial", "actor one", "actor two", "validator", "extra")
    }
    expected |= {"client:make", "tdm-beta", "tdm-gamma"}
    assert set(inputs) == expected and len(called) == len(expected)


@pytest.mark.parametrize(
    "key", ["opponents", "partners", "validation_partners", "validation_opponents"]
)
def test_runtime_rejects_opaque_or_malformed_bindings(key: str, tmp_path: Path) -> None:
    for value in ("random", 3, [object()]):
        with pytest.raises((ValueError, TypeError)):
            study._runtime_paths({key: value}, tmp_path)


@pytest.mark.parametrize("resume", [False, True])
def test_worker_forwards_runtime_bindings_on_fresh_and_resumed_training(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resume: bool
) -> None:
    from marl_battlegrounds.training import runner

    declared = _declaration()
    runtime = {
        "opponents": {"bot": "random"},
        "validation_partners": {"partner": "tdm-alpha"},
    }
    declared["cases"][0]["runtime"] = runtime

    # This test isolates worker dispatch; ordinary config and method bindings
    # have separate real-parser proofs.
    def config(
        value: object, base: Path, seed: int, **kwargs: object
    ) -> dict[str, Any]:
        return copy.deepcopy(cast(dict[str, Any], value))

    monkeypatch.setattr(study, "_config", config)
    root, frozen, _ = study._prepare(declared, tmp_path / "study", None)
    job = study._jobs(frozen)[0]
    directory = root / "cases" / job["id"]
    directory.mkdir()
    checkpoint = directory / f"run/checkpoints/{_SAVED_ID}"
    if resume:
        checkpoint.mkdir(parents=True)
        atomic_json(
            directory / "run/latest_checkpoint.json",
            {
                "schema_version": 1,
                "checkpoint_id": _SAVED_ID,
                "relative_path": f"checkpoints/{_SAVED_ID}",
            },
        )
    atomic_json(
        directory / "request.json",
        {"job": job, "resume_from": str(checkpoint) if resume else None, "attempt": 1},
    )
    calls: list[dict[str, Any]] = []

    def train(config: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            status="complete",
            run_dir=directory / "run",
            completed_env_steps=32,
            completed_updates=1,
            final_actor=directory / "actor",
            selected_actor=None,
        )

    monkeypatch.setattr(runner, "train", train)
    assert study._worker(root, job["id"]) == 0
    assert calls == [
        {
            **runtime,
            **(
                {"resume_from": str(checkpoint)}
                if resume
                else {"output_dir": directory / "run"}
            ),
        }
    ]


@pytest.mark.parametrize("field", ["opponents", "partners"])
@pytest.mark.parametrize("weighted", [False, True])
def test_population_control_names_survive_same_named_folders(
    tmp_path: Path, field: str, weighted: bool
) -> None:
    from marl_battlegrounds.experiments.hydra import resolve_training_config

    (tmp_path / "self").mkdir()
    (tmp_path / "past").mkdir()
    names = (
        ["self", "past", "tdm-alpha"] if field == "opponents" else ["self", "tdm-alpha"]
    )
    declaration = dict.fromkeys(names, 1.0) if weighted else names
    supplied = {field: declaration}
    assert study._config_paths(supplied, tmp_path) == supplied
    if field == "partners":
        with pytest.raises(ValueError, match="Partner names cannot be self or past"):
            resolve_training_config(supplied, base_dir=tmp_path)
        return
    actual = resolve_training_config(supplied, base_dir=tmp_path)
    assert getattr(actual, field) == (declaration if weighted else tuple(declaration))
    normalized = study._config(
        {"method": "ff_ippo", "total_env_steps": 32_768, **supplied}, tmp_path, 1
    )
    assert normalized[field] == declaration


@pytest.mark.parametrize("field", ["opponents", "partners"])
def test_population_path_alias_collision_does_not_drop_a_share(
    tmp_path: Path, field: str
) -> None:
    from marl_battlegrounds.experiments.hydra import resolve_training_config

    actor = tmp_path / "actor"
    actor.mkdir()
    supplied = {field: {"./actor": 0.1, str(actor): 0.2, "self": 0.7}}
    before = copy.deepcopy(supplied)
    with pytest.raises(ValueError, match="same resolved member"):
        study._config_paths(supplied, tmp_path)
    with pytest.raises(ValueError, match="same resolved member"):
        resolve_training_config(supplied, base_dir=tmp_path)
    assert supplied == before
    # Lists deliberately preserve repeated entries and their sampling shares.
    assert study._config_paths({field: ["./actor", str(actor)]}, tmp_path)[field] == [
        str(actor),
        str(actor),
    ]


@pytest.mark.parametrize(
    "name",
    [
        "validation",
        "learning_rate",
        "exploration",
        "history_capture_env_steps",
        "past_capture_interval",
        "keep_past",
        "retain_all_actors",
        "opponents",
        "reward",
        "partners",
        "shaping",
        "shaping_coefficient",
        "shaping_mode",
        "seed",
        "curriculum",
        "ppo",
        "qmix",
        "pqn",
    ],
)
def test_study_admits_every_shared_continuation_change(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from marl_battlegrounds.training._continuation_schedules import (
        CONTINUATION_CHANGE_KEYS,
    )

    values: dict[str, Any] = {
        "validation": {},
        "learning_rate": 0.0003,
        "exploration": 0.05,
        "history_capture_env_steps": [64],
        "past_capture_interval": 64,
        "keep_past": 2,
        "retain_all_actors": True,
        "opponents": {"self": 1.0},
        "reward": None,
        "partners": None,
        "shaping": False,
        "shaping_coefficient": 0.25,
        "shaping_mode": "potential",
        "seed": 17,
        "curriculum": False,
        "ppo": {"gamma": 0.99},
        "qmix": {"gamma": 0.99},
        "pqn": {"gamma": 0.99},
    }
    assert set(values) == CONTINUATION_CHANGE_KEYS

    def identity(path: str) -> dict[str, str]:
        return {
            "checkpoint_id": "parent",
            "description_sha256": "fixed",
            "method": "ff_ippo",
        }

    monkeypatch.setattr(study, "_checkpoint_identity", identity)
    frozen = study._normalize(
        {
            "name": "Continuation adapter",
            "cases": [
                {
                    "id": "child",
                    "checkpoint": str(tmp_path / "parent"),
                    "additional_env_steps": 32,
                    "changes": {name: values[name]},
                }
            ],
        }
    )
    assert frozen["cases"][0]["changes"] == {name: values[name]}
    # Method-specific validity still belongs to extend_training; the study must
    # not silently filter any of the direct API's declared change names.


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("binding", ["explicit", "omitted", "null", "unchanged_panel"])
def test_child_worker_keeps_future_live_panel_bindings_separate_from_lineage(
    host_only: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume: bool,
    binding: str,
) -> None:
    from marl_battlegrounds.training import checkpoints, runner, validation

    panel = tmp_path / "future.json"
    content: dict[str, Any] = {
        "schema_version": 2,
        "members": [{"reference": None if binding == "explicit" else "random"}],
    }
    content["panel_digest"] = validation._panel_digest(content)
    atomic_json(panel, content)
    parent = {
        "checkpoint_id": "parent",
        "description_sha256": "fixed",
        "method": "ff_ippo",
    }

    def identity(path: str) -> dict[str, str]:
        return parent

    monkeypatch.setattr(study, "_checkpoint_identity", identity)
    changes: dict[str, Any] = {"validation": {"panel": str(panel)}}
    if binding == "explicit":
        changes["validation"]["bindings"] = ["random"]
    elif binding == "null":
        changes["validation"]["bindings"] = None
    elif binding == "unchanged_panel":
        changes = {"validation": {"every_env_steps": 32}}
    declaration = {
        "name": "Live child panel",
        "cases": [
            {
                "id": "child",
                "checkpoint": str(tmp_path / "parent/checkpoints/parent"),
                "additional_env_steps": 32,
                "changes": changes,
                "runtime": {"validation_opponents": ["tdm-alpha"]},
            }
        ],
    }
    root, frozen, _ = study._prepare(declaration, tmp_path / "study", None)
    job = study._jobs(frozen)[0]
    before = copy.deepcopy(job)
    directory = root / "cases/child"
    directory.mkdir()
    checkpoint = directory / "run" / "checkpoints" / _CHILD_ID
    if resume:
        checkpoint.mkdir(parents=True)
        atomic_json(
            directory / "run/latest_checkpoint.json",
            {
                "schema_version": 1,
                "checkpoint_id": _CHILD_ID,
                "relative_path": f"checkpoints/{_CHILD_ID}",
            },
        )
        context = {
            "parent_checkpoint": job["checkpoint"],
            "parent_checkpoint_id": "parent",
            "additional_env_steps": 32,
            "changes": {
                "validation": {
                    key: value
                    for key, value in changes["validation"].items()
                    if key != "bindings"
                }
            },
        }

        def description(path: str) -> dict[str, Any]:
            return {"metadata": {"continuation": context}}

        monkeypatch.setattr(checkpoints, "read_checkpoint_description", description)
    atomic_json(
        directory / "request.json",
        {
            "job": job,
            "resume_from": str(checkpoint) if resume else None,
            "attempt": 2 if resume else 1,
        },
    )
    received: list[dict[str, Any]] = []

    def run(*args: object, **kwargs: object) -> None:
        received.append(kwargs)
        raise RuntimeError("Reached ordinary trainer")

    monkeypatch.setattr(runner, "train", run)
    monkeypatch.setattr(runner, "extend_training", run)
    assert study._worker(root, "child") == 1
    assert len(received) == 1
    expected_binding = ["tdm-alpha"]
    if resume and binding != "unchanged_panel":
        expected_binding = ["random"] if binding == "explicit" else None
    assert received[0]["validation_opponents"] == expected_binding
    if not resume:
        assert received[0]["changes"] == changes
    assert job == before
    assert (
        "Reached ordinary trainer" in study._read(directory / "outcome.json")["error"]
    )


@pytest.mark.parametrize("decorated", [False, True])
def test_changed_callback_factory_is_rejected_before_call_or_training(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decorated: bool
) -> None:
    from marl_battlegrounds.training import runner

    module = tmp_path / "study_callback_fixture.py"
    prefix = (
        "from contextlib import contextmanager\n@contextmanager\n" if decorated else ""
    )
    suffix = "    yield print\n" if decorated else ""
    module.write_text(
        prefix
        + "def make():\n    raise AssertionError('Factory must not run')\n"
        + suffix
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        declaration = _declaration()
        declaration["cases"][0]["runtime"] = {
            "on_update": {"factory": "study_callback_fixture:make", "kwargs": {}},
        }
        root, frozen, _ = study._prepare(declaration, tmp_path / "study", None)
        job = study._jobs(frozen)[0]
        directory = root / "cases" / job["id"]
        directory.mkdir()
        atomic_json(
            directory / "request.json", {"job": job, "resume_from": None, "attempt": 1}
        )
        module.write_text(
            prefix
            + "def make():\n    raise AssertionError('Changed provider must not run')\n"
            + suffix
        )

        sys.modules.pop("study_callback_fixture", None)

        def forbidden(*args: object, **kwargs: object) -> None:
            pytest.fail("Changed callback reached the trainer")

        monkeypatch.setattr(runner, "train", forbidden)
        assert study._worker(root, job["id"]) == 1
        assert "external file input" in study._read(directory / "outcome.json")["error"]
        assert not (directory / "run").exists()
    finally:
        sys.modules.pop("study_callback_fixture", None)


def test_preflight_failure_cannot_borrow_prior_run_attempt_costs(
    host_only: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_worker(monkeypatch, [1], checkpoints=True)
    root = tmp_path / "study"
    study.run_study(_declaration(), output_dir=root)
    directory = root / "cases/small--seed-9"
    previous = study._read(directory / "run/status.json")
    previous.update(status="complete", training_seconds=123, elapsed_seconds=456)
    atomic_json(directory / "run/status.json", previous)
    before = (directory / "run/status.json").read_bytes()

    def fail_before_runner(
        directory: Path, command: list[str], **kwargs: object
    ) -> int:
        request = study._read(directory / "request.json")
        atomic_json(
            directory / "outcome.json",
            {
                "status": "failed",
                "request_sha256": study._digest(request),
                "error": "Factory failed before runner admission",
            },
        )
        return 1

    monkeypatch.setattr(study._launch, "supervise_command", fail_before_runner)
    report = study.run_study(resume_from=root)
    case = report["cases"][0]
    assert case["status"] == "failed" and case["run_status"] == {}
    first, second = case["attempts"]
    assert first["run_attempt_id"] == "runner-1"
    assert first["study_attempt_id"] != second["study_attempt_id"]
    assert second["request_sha256"] != first["request_sha256"]
    assert "run_attempt_id" not in second and "run_status" not in second
    assert (directory / "run/status.json").read_bytes() == before
    assert len(calls) == 1


def test_study_attempt_scope_copies_restores_and_excludes_nested_runs(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.training._run_io import (
        study_attempt_binding,
        study_attempt_scope,
    )

    first, second = tmp_path / "first", tmp_path / "second"
    binding = {"study_attempt_id": "first"}
    assert study_attempt_binding(first) is None
    with study_attempt_scope(first, binding):
        binding["study_attempt_id"] = "mutated"
        assert study_attempt_binding(first) == {"study_attempt_id": "first"}
        assert study_attempt_binding(second) is None
        with (
            pytest.raises(RuntimeError),
            study_attempt_scope(second, {"study_attempt_id": "second"}),
        ):
            assert study_attempt_binding(first) is None
            raise RuntimeError("Interrupted inner call")
        assert study_attempt_binding(first) == {"study_attempt_id": "first"}
    assert study_attempt_binding(first) is None
