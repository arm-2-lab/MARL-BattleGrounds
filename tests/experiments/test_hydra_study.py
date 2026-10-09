"""Check finite Hydra sweeps through the existing study owner.

Host previews exercise all composed jobs, order, paths, seeds, overrides and
rejection before dispatch. Controlled public subprocesses check one study call
and truthful Hydra failure mapping. These checks do not establish learning.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.experiments import hydra as adapter
from marl_battlegrounds.training import study


def _jobs() -> list[dict[str, Any]]:
    return [
        {"method": method, "seed": seed, "total_env_steps": 20480, "preview": True}
        for method in ("ff_ippo", "qmix")
        for seed in (1, 2)
    ]


def _sweep(jobs: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    return adapter._study_sweep(
        jobs,
        overrides=[[f"seed={row['seed']}"] for row in jobs],
        base_dir=root,
        launch_dir=root / "launch",
        sweep_overrides=["seed=1,2"],
        sweeper_params={},
    )


def test_preview_preserves_jobs_paths_and_uses_no_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Preview started execution")

    monkeypatch.setattr(study, "run_study", forbidden)
    monkeypatch.setattr(study, "_freeze", forbidden)
    report = _sweep(_jobs(), tmp_path)
    assert report["case_count"] == 4 and report["total_env_steps"] == 81920
    assert [row["id"] for row in report["cases"]] == [
        "ff_ippo-job-0000--seed-1",
        "ff_ippo-job-0001--seed-2",
        "qmix-job-0002--seed-1",
        "qmix-job-0003--seed-2",
    ]
    assert all("seed" not in row["config"] for row in report["declaration"]["cases"])
    assert all(row["status"] == "preview" for row in report["cases"])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_dir", "other"),
        ("name", "other"),
        ("preview", False),
        ("resources", {"devices": [{"device": "cpu"}]}),
        ("study_options", {"failure_policy": "continue"}),
    ],
)
def test_sweep_refuses_different_shared_controls(
    tmp_path: Path, field: str, value: object
) -> None:
    jobs = _jobs()
    jobs[-1][field] = value
    with pytest.raises(ValueError, match="All sweep jobs must share"):
        _sweep(jobs, tmp_path)
    assert not list(tmp_path.iterdir())


def test_duplicate_jobs_and_copied_artifact_references_are_rejected(
    tmp_path: Path,
) -> None:
    import shutil

    from tests.training.test_training_actor_library import _host_run

    jobs = _jobs()
    jobs.append(copy.deepcopy(jobs[0]))
    with pytest.raises(ValueError, match="Duplicate effective"):
        _sweep(jobs, tmp_path)
    _, actor = _host_run(tmp_path / "source")
    copied = tmp_path / "copied"
    shutil.copytree(actor, copied)
    jobs = [dict(_jobs()[0], initial_actor=str(path)) for path in (actor, copied)]
    with pytest.raises(ValueError, match="Duplicate effective"):
        _sweep(jobs, tmp_path)
    assert not (tmp_path / "launch").exists()


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["hydra.sweeper.max_batch_size=2"], "max_batch_size=null"),
        (["hydra/launcher=basic"], "require the study launcher"),
        (["hydra.job.chdir=true"], "chdir/env_set/env_copy"),
        (["+hydra.job.env_set.JAX_PLATFORMS=cpu"], "chdir/env_set/env_copy"),
        (["+qmix.gamma=0.9"], "qmix"),
        (["seed=1,1"], "Duplicate effective"),
    ],
)
def test_public_sweep_rejects_unsupported_inputs_before_workers(
    tmp_path: Path, arguments: list[str], expected: str
) -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "marl_battlegrounds.experiments.train",
            "-m",
            "method=ff_ippo",
            "seed=1,2",
            "preview=true",
            f"hydra.sweep.dir={tmp_path / 'launch'}",
            *arguments,
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode != 0
    assert expected in process.stdout + process.stderr
    assert not (tmp_path / "launch/study").exists()


def test_public_sweep_preview_context_and_method_switching(tmp_path: Path) -> None:
    script = tmp_path / "preview.py"
    script.write_text("""import sys
from marl_battlegrounds.experiments.train import main
main()
assert "jax" not in sys.modules
""")
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "-m",
            "method=ff_ippo,qmix",
            "seed=1,2",
            "preview=true",
            f"output_dir={tmp_path / 'exact'}",
            f"hydra.sweep.dir={tmp_path / 'launch'}",
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout)
    assert report["case_count"] == 4 and report["output_dir"] == str(tmp_path / "exact")
    assert not (tmp_path / "exact").exists()
    assert (tmp_path / "launch/multirun.yaml").exists()


@pytest.mark.parametrize("failure_policy", ["stop", "continue"])
def test_public_sweep_uses_one_study_and_keeps_report_before_hydra_error(
    tmp_path: Path, failure_policy: str
) -> None:
    script = tmp_path / "dispatch.py"
    script.write_text("""import json
from pathlib import Path
from marl_battlegrounds.experiments.train import main
from marl_battlegrounds.training import study
calls = []
def run(config, *, output_dir):
    calls.append(config)
    assert len(calls) == 1
    root = Path(output_dir)
    root.mkdir()
    jobs = [{"id": row["id"] + "--seed-" + str(row["seeds"][0]),
             "status": "failed" if i == 0 else (
                 "pending" if config["failure_policy"] == "stop" else "complete")}
            for i, row in enumerate(config["cases"])]
    result = {"status": "incomplete", "cases": jobs}
    (root / "report.json").write_text(json.dumps(result))
    (root / "declared.json").write_text(json.dumps(config))
    return result
study.run_study = run
main()
""")
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "-m",
            "method=ff_ippo,qmix",
            "seed=1,2",
            f"+study_options.failure_policy={failure_policy}",
            f"output_dir={tmp_path / 'exact'}",
            f"hydra.sweep.dir={tmp_path / 'launch'}",
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode != 0
    assert "saved report:" in process.stderr
    result = json.loads((tmp_path / "exact/report.json").read_text())
    declared = json.loads((tmp_path / "exact/declared.json").read_text())
    assert len(result["cases"]) == 4 and len(declared["launch"]["jobs"]) == 4
    assert (tmp_path / "launch/.hydra/study_result.json").exists()
    assert [row["seeds"][0] for row in declared["cases"]] == [1, 2, 1, 2]
    assert all(
        Path(row["run_dir"]).is_relative_to(tmp_path / "exact")
        for row in declared["launch"]["jobs"]
    )


@pytest.mark.parametrize("mode", ["empty", "partial", "repeated"])
def test_plugin_rejects_partial_repeated_or_empty_batch(mode: str) -> None:
    from hydra_plugins.marl_battlegrounds_study import StudyLauncher

    launcher = StudyLauncher()
    launcher.used = mode == "repeated"
    batch = [] if mode == "empty" else [["seed=1"]]
    with pytest.raises(ValueError, match="one complete nonempty"):
        launcher.launch(batch, 1 if mode == "partial" else 0)


def test_preview_resolves_each_hydra_job_context(tmp_path: Path) -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "marl_battlegrounds.experiments.train",
            "-m",
            "method=ff_ippo,qmix",
            "seed=${hydra:job.num}",
            "preview=true",
            f"hydra.sweep.dir={tmp_path / 'launch'}",
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout)
    assert [row["seeds"] for row in report["declaration"]["cases"]] == [[0], [1]]


def test_method_specific_grid_and_output_overlap_refusal(tmp_path: Path) -> None:
    output = tmp_path / "overlap"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "marl_battlegrounds.experiments.train",
            "-m",
            "method=qmix",
            "qmix.q_lr=0.0001,0.0003",
            "seed=1,2",
            "preview=true",
            f"output_dir={output}",
            f"hydra.sweep.dir={output}",
        ],
        text=True,
        capture_output=True,
    )
    assert (
        process.returncode != 0
        and "outside the experiment destination" in process.stderr
    )
    assert not output.exists()
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "marl_battlegrounds.experiments.train",
            "-m",
            "method=qmix",
            "qmix.q_lr=0.0001,0.0003",
            "seed=1,2",
            "preview=true",
            f"hydra.sweep.dir={tmp_path / 'launch'}",
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout)
    assert report["case_count"] == 4
    assert [
        case["config"]["qmix"]["q_lr"] for case in report["declaration"]["cases"]
    ] == [0.0001, 0.0001, 0.0003, 0.0003]


def test_effective_identity_keeps_roles_shares_and_callback_arguments() -> None:
    identity = {"kind": "artifact", "checkpoint_id": "a" * 64}
    first: dict[str, Any] = {
        "config": {"method": "ff_ippo", "opponents": {"/one": 0.2, "self": 0.8}},
        "external_inputs": {"/one": identity},
    }
    second: dict[str, Any] = {
        "config": {"method": "ff_ippo", "opponents": {"/two": 0.2, "self": 0.8}},
        "external_inputs": {"/two": identity},
    }
    assert study._effective_job_identity(
        first, seed=1
    ) == study._effective_job_identity(second, seed=1)
    second["config"]["opponents"] = {"/two": 0.4, "self": 0.6}
    assert study._effective_job_identity(
        first, seed=1
    ) != study._effective_job_identity(second, seed=1)
    assert study._effective_job_identity(
        first, seed=1
    ) != study._effective_job_identity(first, seed=2)
    first["runtime"] = {
        "on_update": {"factory": "custom:callback", "kwargs": {"path": "/one"}}
    }
    second = copy.deepcopy(first)
    second["runtime"]["on_update"]["kwargs"]["path"] = "/two"
    assert study._effective_job_identity(
        first, seed=1
    ) != study._effective_job_identity(second, seed=1)


@pytest.mark.parametrize(
    "setting",
    [
        "opponents=[self,past,tdm-alpha]",
        "partners=[tdm-alpha]",
    ],
)
def test_public_sweep_preserves_ordered_populations(
    tmp_path: Path, setting: str
) -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "marl_battlegrounds.experiments.train",
            "-m",
            "method=ff_ippo",
            "seed=1,2",
            "preview=true",
            setting,
            "learner_slots=[0]",
            f"hydra.sweep.dir={tmp_path / 'launch'}",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout)
    key = setting.split("=")[0]
    assert isinstance(report["declaration"]["cases"][0]["config"][key], list)
    sequence = {"config": {key: ["random", "tdm-alpha"]}}
    reverse = {"config": {key: ["tdm-alpha", "random"]}}
    shares = {"config": {key: {"random": 1.0, "tdm-alpha": 1.0}}}
    assert (
        len(
            {
                study._effective_job_identity(row, seed=1)
                for row in (sequence, reverse, shares)
            }
        )
        == 3
    )


def test_sweep_panel_directory_and_file_share_one_identity(tmp_path: Path) -> None:
    from marl_battlegrounds.training.validation import create_panel

    panel = create_panel(opponents=["random"], output_dir=tmp_path / "panel")
    job = dict(_jobs()[0], validation_panel=str(panel.path.parent))
    assert _sweep([job], tmp_path)["case_count"] == 1
    with pytest.raises(ValueError, match="Duplicate effective"):
        _sweep([job, dict(job, validation_panel=str(panel.path))], tmp_path)
    left: dict[str, Any] = {
        "config": {"validation_panel": str(panel.path.parent)},
        "external_inputs": {str(panel.path): {"kind": "file", "sha256": "bound"}},
    }
    right = copy.deepcopy(left)
    right["config"]["validation_panel"] = str(panel.path)
    assert study._effective_job_identity(left, seed=1) == study._effective_job_identity(
        right, seed=1
    )


@pytest.mark.parametrize("preview", [True, False])
def test_single_run_rejects_study_options_before_preview_or_training(
    tmp_path: Path, preview: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds import training

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unsupported single-run options reached training")

    monkeypatch.setattr(training, "train", forbidden)
    with pytest.raises(ValueError, match="study_options requires"):
        adapter.run_training(
            {
                "method": "ff_ippo",
                "preview": preview,
                "study_options": {"failure_policy": "continue"},
            },
            base_dir=tmp_path,
            launch_dir=tmp_path / "launch",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("resume", [None, "/saved/run"])
def test_sweep_refuses_explicit_resume_even_when_null(
    tmp_path: Path, resume: str | None
) -> None:
    with pytest.raises(ValueError, match="Resume the saved study"):
        _sweep([dict(_jobs()[0], resume_from=resume)], tmp_path)
    assert not list(tmp_path.iterdir())
