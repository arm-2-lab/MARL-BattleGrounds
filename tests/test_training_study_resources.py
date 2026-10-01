"""Check declared resource slots through the ordinary study and launch owners.

Host validation rejects invalid capacity and environment settings. Real short
subprocesses check concurrent admission, failure, bounded stop and explicit
recovery without models or GPU allocation. Their timings measure orchestration,
not learner throughput. Fresh-process probes check settings before JAX import.
GPU settings keep CUDA first and required, permit CPU checkpoint checks and
turn off JAX's memory preallocation unless the caller set that flag. A declared
compilation cache also keeps programs that compile in under a second. With two
declared GPUs (probes only pretend to be GPU workers; no GPU is touched), a GPU
whose case fails before any checkpoint is paused and recorded while the other
GPU finishes every remaining case; when every GPU is paused the study stops
with the rest pending instead of failing them; each resume clears the pauses.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from tests.test_training_study import host_only as host_only

from marl_battlegrounds.training import _launch, study

_GPU = "GPU-6b11a0c4-14e8-6782-8932-1df56d599796"
_OTHER_GPU = "GPU-8dcabaa4-3802-4240-98e3-6ddebbc6239d"


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"devices": []},
        {"devices": [{"device": "cuda:0"}]},
        {"devices": [{"device": "cpu", "max_workers": True}]},
        {"devices": [{"device": "cpu", "memory_fraction": 0.5}]},
        {"devices": [{"device": _GPU, "max_workers": 2}]},
        {"devices": [{"device": _GPU, "max_workers": 2, "memory_fraction": 0.6}]},
        {"devices": [{"device": _GPU, "memory_fraction": float("nan")}]},
        {"devices": [{"device": "cpu"}, {"device": "cpu"}]},
        {"devices": [{"device": "cpu"}], "other": 1},
    ],
)
def test_resource_declaration_rejects_invalid_capacity(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(ValueError):
        _launch.resource_settings(value, base_dir=tmp_path)


@pytest.mark.parametrize("inherited_platforms", ["cpu", "cuda", "cuda,cpu"])
def test_resource_environment_preserves_researcher_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inherited_platforms: str
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", inherited_platforms)
    monkeypatch.setenv("PYTHONPATH", "/research/methods")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setenv("JAX_ENABLE_X64", "true")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    declared = {
        "devices": [{"device": _GPU, "max_workers": 2, "memory_fraction": 0.4}],
        "compilation_cache": "cache",
    }
    before = copy.deepcopy(declared)
    normalized = _launch.resource_settings(declared, base_dir=tmp_path)
    assert normalized is not None and declared == before
    env = _launch.resource_environment(
        normalized["devices"][0], compilation_cache=normalized["compilation_cache"]
    )
    assert env["PYTHONPATH"] == "/research/methods"
    assert env["JAX_ENABLE_X64"] == "true"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["CUDA_VISIBLE_DEVICES"] == _GPU
    assert env["JAX_PLATFORMS"].split(",") == ["cuda", "cpu"]
    assert os.environ["JAX_PLATFORMS"] == inherited_platforms
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.4"
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "cache")
    assert env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"
    assert not (tmp_path / "cache").exists()
    assert _launch.resource_environment(None) == dict(os.environ)
    cpu = _launch.resource_environment({"device": "cpu", "max_workers": 1})
    assert cpu["JAX_PLATFORMS"] == "cpu" and cpu["CUDA_VISIBLE_DEVICES"] == ""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    with pytest.raises(ValueError, match="ignores memory_fraction"):
        _launch.resource_environment(normalized["devices"][0])


def _probe_program(path: Path) -> None:
    path.write_text("""import json, os, resource, signal, sys, time
from pathlib import Path
from marl_battlegrounds.training import study
from marl_battlegrounds.training._run_io import atomic_json
role, root = sys.argv[1], Path(sys.argv[2])
if role == "control":
    study._command = lambda *args: [sys.executable, __file__, "supervise", *args[1:]]
    study._execute(root, resume="--resume" in sys.argv, handoff=True)
    raise SystemExit(0)
identifier = sys.argv[3]
if role == "supervise":
    study._command = lambda *args: [sys.executable, __file__, "work", *args[1:]]
    raise SystemExit(study._supervise_case(root, identifier))
directory = root / "cases" / identifier
request = study._read(directory / "request.json")
assert "jax" not in sys.modules
resources = study._worker_resources(root, request)
assert "jax" not in sys.modules
gpu = os.environ.get("RESOURCE_PROBE_GPU") == "1"
assert os.environ["JAX_PLATFORMS"] == ("cuda,cpu" if gpu else "cpu")
assert (os.environ["CUDA_VISIBLE_DEVICES"] != "") == gpu
import researcher_module
assert researcher_module.VALUE == 23
atomic_json(directory / "resources.json", resources)
atomic_json(directory / "probe-start.json", {"at": time.time()})
seed = request["job"]["config"]["seed"]
device = os.environ["CUDA_VISIBLE_DEVICES"]
failing = os.environ.get("RESOURCE_PROBE_FAIL")
fail = failing is not None and failing in (str(seed), device, "all")
if os.environ.get("RESOURCE_PROBE_KILL") == str(seed):
    os.kill(os.getpid(), signal.SIGKILL)
time.sleep(0.15 if fail else float(os.environ.get("RESOURCE_PROBE_DELAY", "0.6")))
if fail:
    atomic_json(directory / "outcome.json", {
        "status": "failed", "error": "Injected worker failure",
        "request_sha256": study._digest(request),
    })
    raise SystemExit(1)
run = directory / "run"
run.mkdir(exist_ok=True)
atomic_json(run / "status.json", {
    "status": "complete", "run_id": "fixture-run",
    "attempt_id": "runner-" + str(request["attempt"]),
    "study_attempt": {
        "study_attempt_id": request["study_attempt_id"],
        "request_sha256": study._digest(request), "number": request["attempt"],
        "case_id": request["job"]["id"],
    },
})
atomic_json(directory / "outcome.json", {
    "status": "complete", "request_sha256": study._digest(request),
})
atomic_json(directory / "probe-end.json", {
    "at": time.time(),
    "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
})
""")


def _resource_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_policy: str
) -> dict[str, Any]:
    script = tmp_path / "worker.py"
    _probe_program(script)
    (tmp_path / "researcher_module.py").write_text("VALUE = 23\n")
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    command = study._command

    def worker_command(*arguments: str) -> list[str]:
        roles = {"_supervise": "supervise", "_worker": "work", "_control": "control"}
        if arguments[0] in roles:
            return [sys.executable, str(script), roles[arguments[0]], *arguments[1:]]
        return command(*arguments)

    monkeypatch.setattr(study, "_command", worker_command)
    return {
        "name": "Resource software check",
        "failure_policy": failure_policy,
        "resources": {"devices": [{"device": "cpu", "max_workers": 2}]},
        "cases": [
            {
                "id": "recipe",
                "config": {"method": "ff_ippo", "total_env_steps": 32},
                "seeds": [1, 2, 3, 4],
            }
        ],
    }


def test_concurrent_slots_environment_and_completed_resume(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declared = _resource_study(tmp_path, monkeypatch, "stop")
    root = tmp_path / "study"
    start = time.perf_counter()
    report = study.run_study(declared, output_dir=root)
    assert report["status"] == "complete"
    assert [row["id"] for row in report["cases"]] == [
        f"recipe--seed-{seed}" for seed in (1, 2, 3, 4)
    ]
    intervals: list[tuple[float, int]] = []
    for row in report["cases"]:
        folder = root / "cases" / row["id"]
        began, ended = (
            json.loads((folder / f"probe-{kind}.json").read_text())["at"]
            for kind in ("start", "end")
        )
        intervals.extend(((began, 1), (ended, -1)))
        facts = row["attempts"][0]["resource_facts"]
        assert facts["environment"]["JAX_PLATFORMS"] == "cpu"
        assert facts["requested"]["device"]["device"] == "cpu"
    active = maximum = 0
    for _, change in sorted(intervals):
        active += change
        maximum = max(maximum, active)
    assert maximum == 2 and active == 0
    assert not study._launch.live_process_records(study._process_paths(root))
    resumed = study.run_study(resume_from=root)
    assert all(len(row["attempts"]) == 1 for row in resumed["cases"])
    (tmp_path / "orchestration-cost.json").write_text(
        json.dumps(
            {
                "seconds": time.perf_counter() - start,
                "workers": 4,
                "concurrency": maximum,
                "worker_sleep_seconds": 0.6,
            }
        )
    )


@pytest.mark.parametrize("failure_policy", ["stop", "continue"])
def test_failed_worker_keeps_all_declared_cases(
    host_only: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_policy: str,
) -> None:
    declared = _resource_study(tmp_path, monkeypatch, failure_policy)
    monkeypatch.setenv("RESOURCE_PROBE_FAIL", "1")
    root = tmp_path / "study"
    report = study.run_study(declared, output_dir=root)
    assert len(report["cases"]) == 4 and report["status"] == "incomplete"
    assert report["cases"][0]["status"] == "failed"
    if failure_policy == "stop":
        assert report["cases"][1]["status"] == "interrupted"
        assert all(
            row["status"] == "pending" and not row["attempts"]
            for row in report["cases"][2:]
        )
    else:
        assert all(row["status"] == "complete" for row in report["cases"][1:])
    assert not study._launch.live_process_records(study._process_paths(root))
    monkeypatch.delenv("RESOURCE_PROBE_FAIL")
    repaired = study.run_study(resume_from=root)
    assert repaired["status"] == "complete"
    assert len(repaired["cases"][0]["attempts"]) == 2


def _no_device_query(resources: object) -> None:
    del resources


def test_gpu_workers_default_to_no_memory_preallocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    gpu = {"device": _GPU, "max_workers": 1}
    assert _launch.resource_environment(gpu)["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    cpu = _launch.resource_environment({"device": "cpu", "max_workers": 1})
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in cpu
    # An explicit caller choice wins over the default.
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    assert _launch.resource_environment(gpu)["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"


def test_failed_gpu_is_paused_while_the_other_gpu_finishes_and_resume_clears_it(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declared = _resource_study(tmp_path, monkeypatch, "continue")
    declared["resources"] = {"devices": [{"device": _GPU}, {"device": _OTHER_GPU}]}
    monkeypatch.setattr(study._launch, "check_resource_devices", _no_device_query)
    monkeypatch.setenv("RESOURCE_PROBE_GPU", "1")
    root = tmp_path / "study"
    # Both GPUs fail before a checkpoint: admission stops, nothing else fails.
    monkeypatch.setenv("RESOURCE_PROBE_FAIL", "all")
    report = study.run_study(declared, output_dir=root)
    assert report["status"] == "incomplete"
    assert set(report["paused_devices"]) == {_GPU, _OTHER_GPU}
    assert [row["status"] for row in report["cases"]] == [
        "failed",
        "failed",
        "pending",
        "pending",
    ]
    assert all(row["attempts"][0]["device_paused"] for row in report["cases"][:2])
    # Resume clears the pauses; only the first GPU fails now.
    monkeypatch.setenv("RESOURCE_PROBE_FAIL", _GPU)
    resumed = study.run_study(resume_from=root)
    paused = resumed["paused_devices"]
    assert list(paused) == [_GPU]
    assert paused[_GPU]["case_id"] == "recipe--seed-1"
    assert paused[_GPU]["error"] == "Injected worker failure"
    rows = resumed["cases"]
    assert [row["status"] for row in rows] == ["failed"] + ["complete"] * 3
    assert all(
        row["attempts"][-1]["resources"]["device"]["device"] == _OTHER_GPU
        for row in rows[1:]
    )
    monkeypatch.delenv("RESOURCE_PROBE_FAIL")
    repaired = study.run_study(resume_from=root)
    assert repaired["status"] == "complete" and repaired["paused_devices"] == {}
    assert len(repaired["cases"][0]["attempts"]) == 3
    assert not study._launch.live_process_records(study._process_paths(root))


def test_declared_gpu_availability_is_checked_without_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def query(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ]
        return subprocess.CompletedProcess(command, 0, stdout="GPU-other\n")

    monkeypatch.setattr(_launch.subprocess, "run", query)
    resources = _launch.resource_settings(
        {"devices": [{"device": _GPU}]}, base_dir=tmp_path
    )
    with pytest.raises(ValueError, match="unavailable"):
        _launch.check_resource_devices(resources)
    _launch.check_resource_devices({"devices": [{"device": "cpu", "max_workers": 1}]})


def test_concurrent_stop_and_deadline_keep_pending_cases(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declared = _resource_study(tmp_path, monkeypatch, "continue")
    monkeypatch.setenv("RESOURCE_PROBE_DELAY", "30")
    root = tmp_path / "stopped-study"
    failures: list[str] = []

    def stop_when_running() -> None:
        try:
            while not list(root.glob("cases/*/probe-start.json")):
                time.sleep(0.01)
            assert list(root.glob("cases/*/probe-start.json"))
            study.stop_study(root)
        except Exception as error:
            failures.append(str(error))

    stopper = threading.Thread(target=stop_when_running)
    stopper.start()
    report = study.run_study(declared, output_dir=root)
    stopper.join()
    assert not failures and not stopper.is_alive()
    assert report["status"] == "stopped"
    assert all(row["status"] == "pending" for row in report["cases"][2:])
    assert not _launch.live_process_records(study._process_paths(root))
    declared["duration_seconds"] = _launch.cleanup_reserve_seconds() + 1
    timed = study.run_study(declared, output_dir=tmp_path / "timed-study")
    assert timed["status"] == "time_limit"
    assert any(row["status"] == "time_limit" for row in timed["cases"])
    assert not _launch.live_process_records(
        study._process_paths(tmp_path / "timed-study")
    )


def test_single_run_resource_preview_and_late_backend_refusal(tmp_path: Path) -> None:
    program = """
import json, sys
from pathlib import Path
from marl_battlegrounds.experiments.hydra import run_training
base = Path(sys.argv[1])
settings = {"method": "ff_ippo", "total_env_steps": 32, "num_envs": 4,
            "resources": {"devices": [{"device": "cpu"}]}, "preview": True}
report = run_training(settings, base_dir=base, launch_dir=base / "launch")
assert "jax" not in sys.modules
assert report["resource_environment"]["JAX_PLATFORMS"] == "cpu"
assert not (base / "launch").exists()
settings["resources"]["devices"][0]["max_workers"] = 2
try:
    run_training(settings, base_dir=base, launch_dir=base / "launch")
except ValueError as error:
    assert "single run" in str(error)
else:
    raise AssertionError("Multiple workers accepted for a single run")
settings["resources"]["devices"][0]["max_workers"] = 1
settings["preview"] = False
sys.modules["jax"] = object()
try:
    run_training(settings, base_dir=base, launch_dir=base / "launch")
except RuntimeError as error:
    assert "fresh launcher" in str(error)
else:
    raise AssertionError("Late resource application was accepted")
assert not (base / "launch").exists()
"""
    process = subprocess.run(
        [sys.executable, "-B", "-c", program, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_real_detached_resource_study_and_live_duplicate_refusal(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declared = _resource_study(tmp_path, monkeypatch, "continue")
    root = tmp_path / "detached"
    process = study.start_study(declared, output_dir=root)
    try:
        with pytest.raises(RuntimeError, match="active owned process"):
            study.start_study(resume_from=root)
        while True:
            status = study.study_status(root)
            if not status["owned_live_records"]:
                break
            time.sleep(0.03)
        report = study.study_status(root)["report"]
        assert report["status"] == "complete", (root / "logs/study.log").read_text()
        assert all(
            row["attempts"][0]["resource_facts"]["environment"]["JAX_PLATFORMS"]
            == "cpu"
            for row in report["cases"]
        )
    finally:
        if study.study_status(root)["owned_live_records"]:
            study.stop_study(root)
        with suppress(ChildProcessError):
            os.waitpid(process["process"]["pid"], 0)


def test_startup_failure_and_lost_controller_keep_attempt_facts(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training._run_io import atomic_json

    declared = _resource_study(tmp_path, monkeypatch, "continue")
    spawn = _launch.spawn_detached

    def unavailable(*args: object, **kwargs: object) -> dict[str, Any]:
        raise OSError("Injected process startup failure")

    root = tmp_path / "failed-start"
    with monkeypatch.context() as patch:
        patch.setattr(_launch, "spawn_detached", unavailable)
        report = study.run_study(declared, output_dir=root)
    assert report["status"] == "incomplete"
    assert all(
        row["status"] == "failed" and len(row["attempts"]) == 1
        for row in report["cases"]
    )
    assert all(
        "Injected process startup failure" in row["attempts"][0]["error"]
        for row in report["cases"]
    )
    assert _launch.spawn_detached is spawn
    # Recover the saved boundary left by a controller that vanished mid-attempt.
    state = study._read(root / "study.json")
    row = state["cases"]["recipe--seed-1"]
    row["attempts"][-1].pop("finished_at")
    folder = root / "cases" / row["id"]
    request = study._read(folder / "request.json")
    facts = {
        "request_sha256": study._digest(request),
        "environment": {"JAX_PLATFORMS": "cpu"},
    }
    atomic_json(folder / "resources.json", facts)
    atomic_json(root / "study.json", state)
    repaired = study.run_study(resume_from=root)
    assert repaired["status"] == "complete"
    old, new = repaired["cases"][0]["attempts"]
    assert old["interrupted"] and old["resource_facts"] == facts
    assert new["number"] == 2 and new["resource_facts"]["requested"]


def test_resource_process_cost_against_inherited_single_worker(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    declared = _resource_study(tmp_path, monkeypatch, "stop")
    declared["cases"][0]["seeds"] = [1, 2]
    rows = []
    for name, count in (("inherited", None), ("one", 1), ("two", 2)):
        config = copy.deepcopy(declared)
        if count is None:
            config.pop("resources")
        else:
            config["resources"]["devices"][0]["max_workers"] = count
        root = tmp_path / name
        began = time.perf_counter()
        report = study.run_study(config, output_dir=root)
        elapsed = time.perf_counter() - began
        assert report["status"] == "complete"
        ends = [
            study._read(root / "cases" / row["id"] / "probe-end.json")
            for row in report["cases"]
        ]
        rows.append(
            {
                "mode": name,
                "wall_seconds": elapsed,
                "worker_peak_ram_bytes": max(end["peak_ram_bytes"] for end in ends),
                "output_bytes": sum(
                    path.stat().st_size for path in root.rglob("*") if path.is_file()
                ),
                "cases": 2,
                "sleep_seconds_per_worker": 0.6,
            }
        )
    (tmp_path / "process-costs.json").write_text(json.dumps(rows, indent=2))


@pytest.mark.parametrize("failure_policy", ["stop", "continue"])
def test_killed_worker_is_failure_not_a_user_stop(
    host_only: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_policy: str,
) -> None:
    declared = _resource_study(tmp_path, monkeypatch, failure_policy)
    monkeypatch.setenv("RESOURCE_PROBE_KILL", "1")
    root = tmp_path / "killed"
    report = study.run_study(declared, output_dir=root)
    assert report["status"] == "incomplete"
    assert report["cases"][0]["status"] == "failed"
    process = study._read(root / "cases/recipe--seed-1/process.json")
    assert process["exit_code"] == -9 and "stop_reason" not in process
    if failure_policy == "continue":
        assert all(row["status"] == "complete" for row in report["cases"][1:])
    else:
        assert report["cases"][1]["status"] == "interrupted"
        assert all(row["status"] == "pending" for row in report["cases"][2:])
    assert not _launch.live_process_records(study._process_paths(root))


@pytest.mark.parametrize("controller_lost", [False, True])
@pytest.mark.parametrize("detached", [False, True])
def test_completed_worker_is_reconciled_after_controller_loss(
    host_only: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controller_lost: bool,
    detached: bool,
) -> None:
    from marl_battlegrounds.training._run_io import atomic_json

    declared = _resource_study(tmp_path, monkeypatch, "stop")
    declared["duration_seconds"] = 120
    root = tmp_path / "study"
    report = study.run_study(declared, output_dir=root)
    assert report["status"] == "complete"
    state = study._read(root / "study.json")
    first = state["cases"][report["cases"][0]["id"]]
    if controller_lost:
        first["status"] = "running"
        first["attempts"][0].pop("finished_at")
        first["attempts"][0].pop("resource_facts")
        state["status"] = "running"
        atomic_json(root / "study.json", state)
    before = {
        path: path.read_bytes() for path in (root / "cases").rglob("probe-start.json")
    }
    expired = time.time() + 1000
    monkeypatch.setattr(study.time, "time", lambda: expired)

    def no_device_check(resources: object) -> None:
        raise AssertionError("A completed study needs no device")

    monkeypatch.setattr(_launch, "check_resource_devices", no_device_check)
    if detached:
        process = study.start_study(resume_from=root)
        try:
            while study.study_status(root)["owned_live_records"]:
                time.sleep(0.02)
            resumed = study.study_status(root)["report"]
        finally:
            if study.study_status(root)["owned_live_records"]:
                study.stop_study(root)
            with suppress(ChildProcessError):
                os.waitpid(process["process"]["pid"], 0)
    else:
        resumed = study.run_study(resume_from=root)
    assert resumed["status"] == "complete"
    assert all(len(row["attempts"]) == 1 for row in resumed["cases"])
    attempt = resumed["cases"][0]["attempts"][0]
    assert attempt["resource_facts"]
    if controller_lost:
        assert attempt["recovered_at"]
    assert not attempt.get("interrupted")
    assert all(path.read_bytes() == content for path, content in before.items())


def test_missing_device_fails_before_detached_spawn_and_records_controller_error(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declared = _resource_study(tmp_path, monkeypatch, "stop")
    declared["resources"] = {"devices": [{"device": _GPU}]}

    def unavailable(resources: object) -> None:
        raise ValueError("Declared GPU devices are unavailable")

    monkeypatch.setattr(_launch, "check_resource_devices", unavailable)
    detached = tmp_path / "detached"
    with pytest.raises(ValueError, match="unavailable"):
        study.start_study(declared, output_dir=detached)
    assert not (detached / "process.json").exists()
    foreground = tmp_path / "foreground"
    with pytest.raises(ValueError, match="unavailable"):
        study.run_study(declared, output_dir=foreground)
    status = study.study_status(foreground)
    assert status["report"]["status"] == "incomplete"
    assert "unavailable" in status["study"]["error"]
    assert not status["owned_live_records"]
    assert all(row["status"] == "pending" for row in status["report"]["cases"])
