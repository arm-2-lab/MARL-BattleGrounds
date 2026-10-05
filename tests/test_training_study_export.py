"""Check saved-study tables, exact attempt joins and last-published indexes.

Host-only fake workers and real saved scalar-table readers cover failed and
never-started jobs, explicit factor rows, bounded table iteration, interruption
and refresh. No model, training, evaluation or provider runs in these tests.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from tests.test_point_margin_summary import _saved
from tests.test_training_study import _declaration, _fake_worker, host_only

from marl_battlegrounds import training
from marl_battlegrounds.training import analysis, study

# pyright: reportPrivateUsage=false
# Reuse the exact study producer fixture; it never runs a numerical worker.
__all__ = ["host_only"]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _fake_worker(monkeypatch, [0, 1])
    declaration = _declaration()
    declaration["cases"][0]["seeds"] = [9, 4, 7]
    root = tmp_path / "study"
    study.run_study(declaration, output_dir=root)
    return root


def test_export_all_attempts_and_never_started_cases_with_fixed_empty_headers(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _study(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    output = tmp_path / "report"
    paths = training.export_study_data([root], output_dir=output)
    assert all(isinstance(path, Path) and path.is_absolute() for path in paths.values())
    rows = _rows(paths["run_index.csv"])
    assert [row["case_id"] for row in rows] == [
        "small--seed-9",
        "small--seed-4",
        "small--seed-7",
    ]
    assert rows[0]["run_attempt_id"] == rows[1]["run_attempt_id"] == "runner-1"
    assert rows[2]["attempt_id"] == rows[2]["actual_env_steps"] == ""
    assert rows[2]["status"] == "pending"
    index = analysis._read_data_index(output)
    assert len(index["tables"]) == 13
    assert index["tables"]["hyperparameters/configurations.jsonl"]["rows"] == 3
    assert _rows(paths["behavior/metrics.csv"]) == []
    assert paths["behavior/metrics.csv"].read_text().startswith("evaluation_id,")
    assert all(path.read_bytes() == data for path, data in before.items())
    original = {name: path.read_bytes() for name, path in paths.items()}
    again = training.export_study_data([root], output_dir=output)
    assert all(path.read_bytes() == original[name] for name, path in again.items())


def test_evaluation_games_and_explicit_factors_are_in_first_index(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _study(tmp_path, monkeypatch)
    evaluation_parent = tmp_path / "eval"
    evaluation_parent.mkdir()
    evaluation = _saved(evaluation_parent, [(0, 1, None), (0, 3, None)])
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "cases": [
            {
                "study_dir": str(root),
                "case_id": "small--seed-9",
                "stage": "Software check",
                "recipe": "Tiny",
            }
        ],
        "factors": [
            {
                "study_dir": str(root),
                "case_id": "small--seed-9",
                "factor": "learning_rate",
                "level": 0.0003,
                "coding": 0,
                "scale": "log",
                "role": "fixed",
            }
        ],
    }
    paths = training.export_study_data(
        [root],
        output_dir=tmp_path / "reports",
        evaluation_dirs=[evaluation],
        protocol=protocol,
    )
    index = analysis._read_data_index(tmp_path / "reports")
    assert index["tables"]["evaluation/games.csv"]["rows"] == 4
    assert {row["system_id"] for row in _rows(paths["evaluation/games.csv"])} == {"a"}
    assert _rows(paths["hyperparameters/design.csv"])[0]["role"] == "fixed"
    assert _rows(paths["run_index.csv"])[0]["stage"] == "Software check"
    assert any("No saved Viewer" in reason for reason in index["unavailable"])


def test_interrupted_publication_is_refused_until_refreshed(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _study(tmp_path, monkeypatch)
    output = tmp_path / "reports"
    training.export_study_data([root], output_dir=output)
    state = study._read(root / "study.json")
    state["cases"]["small--seed-4"]["attempts"][0]["error"] = "New observed failure"
    study.atomic_json(root / "study.json", state)
    original = analysis.os.replace

    def interrupt(source: str | Path, target: str | Path) -> None:
        if Path(str(target)) == output / "costs/attempt_costs.csv":
            raise OSError("Injected publication interruption")
        original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(analysis.os, "replace", interrupt)
        with pytest.raises(OSError, match="Injected"):
            training.export_study_data([root], output_dir=output)
    with pytest.raises(ValueError, match="inconsistent"):
        analysis._read_data_index(output)
    training.export_study_data([root], output_dir=output)
    assert analysis._read_data_index(output)["schema_version"] == 1


def test_bad_protocol_and_raw_destination_fail_before_replacing_outputs(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _study(tmp_path, monkeypatch)
    output = tmp_path / "reports"
    output.mkdir()
    marker = output / "run_index.csv"
    marker.write_text("Keep old derived rows")
    with pytest.raises(ValueError):
        training.export_study_data(
            [root],
            output_dir=output,
            protocol={"schema_version": 1, "cases": [{"invented": True}]},
        )
    assert marker.read_text() == "Keep old derived rows"
    with pytest.raises(ValueError, match="raw run"):
        training.export_study_data([root], output_dir=root / "cases/small--seed-9/run")


def test_checked_public_paths_work_after_moving_derived_folder(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _study(tmp_path, monkeypatch)
    output = tmp_path / "reports"
    training.export_study_data([root], output_dir=output)
    target = tmp_path / "moved"
    output.rename(target)
    paths = training.load_study_data(target)
    index = json.loads(paths["data_index.json"].read_text())
    assert index["tables"]["run_index.csv"]["rows"] == 3
    assert _rows(paths["run_index.csv"])[0]["case_id"] == "small--seed-9"
    paths["run_index.csv"].write_text("Changed derived data")
    with pytest.raises(ValueError, match="inconsistent"):
        training.load_study_data(target)


def test_attempt_clocks_are_differences_and_old_status_is_not_borrowed() -> None:
    binding = {"study_attempt_id": "study-second", "request_sha256": "request-two"}
    context: dict[str, Any] = {
        "events": [
            {
                "event": "attempt_complete",
                "attempt_id": "old",
                "study_attempt": {"study_attempt_id": "first", "request_sha256": "one"},
                "training_seconds": 99,
            },
            {
                "event": "attempt_started",
                "attempt_id": "new",
                "study_attempt": binding,
                "training_start_seconds": 40,
                "elapsed_start_seconds": 60,
                "setup_seconds": 2,
            },
            {
                "event": "validation_complete",
                "attempt_id": "new",
                "study_attempt": binding,
                "score": 0.5,
            },
            {
                "event": "validation_complete",
                "attempt_id": "new",
                "study_attempt": binding,
                "seconds": 3,
            },
            {
                "event": "attempt_failed",
                "attempt_id": "new",
                "study_attempt": binding,
                "training_seconds": 45,
                "elapsed_seconds": 70,
                "wall_seconds": 120,
            },
        ]
    }
    row = analysis._attempt_export_costs(context, {**binding, "run_attempt_id": "new"})
    assert row["training_seconds"] == 45 and row["elapsed_seconds"] == 70
    assert row["attempt_training_seconds"] == 5 and row["attempt_active_seconds"] == 10
    assert row["validation_seconds"] == 3 and row["setup_seconds"] == 2
    absent = analysis._attempt_export_costs(
        context, {"study_attempt_id": "third", "request_sha256": "three"}
    )
    assert absent["training_seconds"] is absent["attempt_active_seconds"] is None


def test_update_export_checks_owner_and_preserves_suffixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training import _checkpoint_metadata as metadata
    from marl_battlegrounds.training import _run_io

    root = tmp_path / "run"
    root.mkdir()
    (root / "run_details.json").write_text('{"run_id":"owner"}')
    (root / "latest_checkpoint.json").write_text("{}")
    log = root / "training_updates.jsonl"
    log.write_text('{"update_index":1,"env_steps":4,"attempt_id":"accepted"}\n')
    cursor = _run_io.log_cursor(log)
    saved: dict[str, Any] = {
        "metadata": {"run_id": "owner", "log_cursors": {log.name: cursor}}
    }

    def read(_root: Path) -> tuple[Path, dict[str, Any]]:
        return root, saved

    monkeypatch.setattr(metadata, "read_resume_checkpoint", read)
    with log.open("a") as stream:
        stream.write('{"update_index":2,"env_steps":8}\n{"torn"')
    archive = _run_io.preserve_log_suffix(log, cursor, root / "attempts/old")
    assert archive is not None
    (archive.parent / "run_events.jsonl").write_text('{"event":"Not an update"}\n')
    rows = list(analysis._export_update_rows(root))
    assert [row["acceptance"] for row in rows] == [
        "Accepted",
        "Uncommitted",
        "Uncommitted",
        "Superseded",
        "Superseded",
    ]
    assert analysis._read_rows(log) == [
        {"update_index": 1, "env_steps": 4, "attempt_id": "accepted"}
    ]
    (root / "checkpoint_recovery.json").write_text("{}")
    assert {row["acceptance"] for row in analysis._export_update_rows(root)} == {
        "Recovery pending",
        "Superseded",
    }
    saved["metadata"]["run_id"] = "foreign"
    with pytest.raises(ValueError, match="another run"):
        list(analysis._export_update_rows(root))


@pytest.mark.parametrize("remove", [False, True])
def test_changed_or_disappeared_source_preserves_old_export(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remove: bool
) -> None:
    root = _study(tmp_path, monkeypatch)
    output = tmp_path / "reports"
    paths = training.export_study_data([root], output_dir=output)
    before = {name: path.read_bytes() for name, path in paths.items()}
    source = root / "study.json"
    original = analysis._export_csv
    calls = 0

    def mutate(
        path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]
    ) -> int:
        nonlocal calls
        result = original(path, columns, rows)
        calls += 1
        if calls == 12:
            if remove:
                source.unlink()
            else:
                source.write_text(source.read_text() + "\n")
        return result

    monkeypatch.setattr(analysis, "_export_csv", mutate)
    with pytest.raises(ValueError, match=r"source (changed|disappeared)"):
        training.export_study_data([root], output_dir=output)
    assert all(path.read_bytes() == before[name] for name, path in paths.items())


def test_mixed_metric_coverage_keeps_priority_only_episodes_and_conditions(
    tmp_path: Path,
) -> None:
    import numpy as np

    from marl_battlegrounds.evaluation.metric_catalog import METRIC_COLUMNS_BY_NAME

    metric = next(iter(METRIC_COLUMNS_BY_NAME))
    entry: dict[str, Any] = {
        "phase": "evaluation",
        "pass_id": "one",
        "system_ids": {"team_a": "actor", "team_b": "other"},
        "episodes": {"1": {"config_id": "config"}, "2": {"config_id": "config"}},
        "result_state": {"status": "complete"},
    }

    class SavedView:
        metadata: ClassVar[dict[str, Any]] = {
            "run_id": "saved",
            "passes": {"one": entry},
            "tables": {
                name: {"availability": "available"}
                for name in ("full_metrics", "priority_metrics")
            },
            "configurations": {
                "config": {
                    "agent_profile": {"active_mask": [True, False], "class_ids": [1, 2]}
                }
            },
            "systems": {"actor": {"name": "Own system"}, "other": {"name": "Opponent"}},
        }

        def iter_table(self, table: str, *, rows: int) -> Iterable[dict[str, Any]]:
            identifiers = [1] if table == "full_metrics" else [1, 2]
            yield {
                "run_id": np.array(["saved"] * len(identifiers)),
                "phase": np.array(["evaluation"] * len(identifiers)),
                "pass_id": np.array(["one"] * len(identifiers)),
                "episode_id": np.array(identifiers),
                metric: np.array(identifiers),
            }

    view = SavedView()
    rows = list(
        analysis._evaluation_export_rows(
            "behavior/metrics.csv", [("evaluation_1", tmp_path, view)], tmp_path, []
        )
    )
    assert [(row["episode_id"], row["table"]) for row in rows] == [
        (1, "full_metrics"),
        (2, "priority_metrics"),
    ]
    conditions = list(
        analysis._evaluation_export_rows(
            "evaluation/conditions.csv",
            [("evaluation_1", tmp_path, view)],
            tmp_path,
            [],
        )
    )
    assert conditions[0]["configurations_json"] == view.metadata["configurations"]
    assert conditions[0]["systems_json"] == view.metadata["systems"]


@pytest.mark.parametrize(
    "mismatch", [None, "number", "case_id", "declaration_sha256", "request_sha256"]
)
def test_live_status_costs_require_the_same_case_and_request(
    mismatch: str | None,
) -> None:
    binding: dict[str, Any] = {
        "study_attempt_id": "active-study-attempt",
        "request_sha256": "request",
        "case_id": "one",
        "number": 2,
        "declaration_sha256": "declaration",
    }
    status: dict[str, Any] = {
        "attempt_id": "active-runner",
        "study_attempt": dict(binding),
        "training_seconds": 45,
        "elapsed_seconds": 70,
        "wall_seconds": 120,
    }
    if mismatch is not None:
        status["study_attempt"][mismatch] = "Different"
    context: dict[str, Any] = {
        "events": [],
        "status": status,
        "job": {"id": "one"},
        "declaration_sha256": "declaration",
    }
    row = analysis._attempt_export_costs(context, binding)
    assert row["training_seconds"] == (45 if mismatch is None else None)
    assert row["run_attempt_id"] == ("active-runner" if mismatch is None else None)


def test_legacy_completed_case_keeps_its_status_without_borrowing_costs(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _study(tmp_path, monkeypatch)
    state = study._read(root / "study.json")
    attempt = state["cases"]["small--seed-9"]["attempts"][-1]
    attempt.pop("study_attempt_id")
    attempt.pop("run_status")
    attempt.pop("run_attempt_id")
    attempt["status"] = "running"
    study.atomic_json(root / "study.json", state)
    paths = training.export_study_data([root], output_dir=tmp_path / "reports")
    for name in ("run_index.csv", "costs/attempt_costs.csv"):
        row = _rows(paths[name])[0]
        assert row["status"] == "complete"
        assert row.get("training_seconds", "") == ""


def test_active_attempt_progress_and_updates_use_the_same_checked_binding(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _study(tmp_path, monkeypatch)
    state = study._read(root / "study.json")
    case = state["cases"]["small--seed-9"]
    attempt = case["attempts"][0]
    run = root / "cases/small--seed-9/run"
    status = study._read(run / "status.json")
    status.update(status="running", env_steps=24, training_seconds=4.0)
    status["study_attempt"]["declaration_sha256"] = state["declaration_sha256"]
    for key in ("run_status", "run_attempt_id", "run_id", "exit_code", "ended_at"):
        attempt.pop(key, None)
    attempt["status"] = case["status"] = state["status"] = "running"
    study.atomic_json(root / "study.json", state)
    study.atomic_json(run / "status.json", status)
    (run / "training_updates.jsonl").write_text(
        json.dumps({"attempt_id": status["attempt_id"], "env_steps": 24}) + "\n"
    )
    paths = training.export_study_data([root], output_dir=tmp_path / "out")
    index = _rows(paths["run_index.csv"])[0]
    updates = _rows(paths["learning/training_updates.csv"])
    assert index["actual_env_steps"] == "24"
    assert index["run_attempt_id"] == status["attempt_id"]
    assert updates[0]["attempt_id"] == attempt["study_attempt_id"]
    assert updates[0]["run_attempt_id"] == status["attempt_id"]


def _completed_export_run(root: Path) -> tuple[dict[str, Any], Path]:
    from tests.test_training_actor_library import _host_run

    details, actor = _host_run(root)
    study.atomic_json(
        root / "selection.json", {"checkpoint_id": details["checkpoint_id"]}
    )
    study.atomic_json(
        root / "status.json",
        {
            "run_id": details["metadata"]["run_id"],
            "status": "complete",
            "env_steps": 0,
            "completed_updates": 0,
            "final_actor": str(actor),
            "selected_actor": str(actor),
        },
    )
    return details, actor


@pytest.mark.parametrize(
    "field", ["final_actor", "selected_actor", "selection_presence"]
)
def test_export_rejects_foreign_actor_or_changed_selection_before_replacing_tables(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    from tests.test_training_actor_library import _revise_learner, _signed
    from tests.test_training_progress import _actor

    root = _study(tmp_path, monkeypatch)
    run = root / "cases/small--seed-9/run"
    shutil.rmtree(run)
    _completed_export_run(run)
    other = tmp_path / "foreign"
    foreign, old_actor = _completed_export_run(other)
    foreign["metadata"]["run_id"] = "another-run"
    foreign = _revise_learner(other, foreign)
    shutil.rmtree(old_actor)
    actor = _actor(other, foreign)
    description = study._read(actor / "actor_details.json")
    description["metadata"]["run_id"] = "another-run"
    study.atomic_json(actor / "actor_details.json", _signed(description))
    output = tmp_path / "output"
    paths = training.export_study_data([root], output_dir=output)
    before = {name: path.read_bytes() for name, path in paths.items()}
    status = study._read(run / "status.json")
    status["selected_actor" if field == "selection_presence" else field] = (
        None if field == "selection_presence" else str(actor)
    )
    study.atomic_json(run / "status.json", status)
    with pytest.raises(
        ValueError, match=r"originating checkpoint|selection and status"
    ):
        training.export_study_data([root], output_dir=output)
    assert all(path.read_bytes() == before[name] for name, path in paths.items())


def test_export_admits_relocated_and_inherited_actors_with_description_receipts(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_training_actor_library import _revise_learner, _signed
    from tests.test_training_progress import _actor

    root = _study(tmp_path, monkeypatch)
    run = root / "cases/small--seed-9/run"
    shutil.rmtree(run)
    child, old_actor = _completed_export_run(run)
    parent_root = tmp_path / "parent"
    parent, inherited_actor = _completed_export_run(parent_root)
    parent_path = parent_root / "checkpoints" / parent["checkpoint_id"]
    child["metadata"]["run_id"] = "child"
    child["metadata"]["continuation"] = {
        "inherited_candidates": [
            {
                "parent_checkpoint": str(parent_path),
                "parent_checkpoint_id": parent["checkpoint_id"],
                "run_id": "one-run",
                "tasks": [{"checkpoint_id": parent["checkpoint_id"]}],
            }
        ]
    }
    child = _revise_learner(run, child)
    shutil.rmtree(old_actor)
    final = _actor(run, child)
    actor_details = study._read(final / "actor_details.json")
    actor_details["metadata"]["run_id"] = "child"
    study.atomic_json(final / "actor_details.json", _signed(actor_details))
    study.atomic_json(
        run / "status.json",
        {
            "run_id": "child",
            "status": "complete",
            "env_steps": 0,
            "completed_updates": 0,
            "final_actor": str(tmp_path / "old/actors" / final.name),
            "selected_actor": str(inherited_actor),
        },
    )
    study.atomic_json(
        run / "selection.json", {"checkpoint_id": parent["checkpoint_id"]}
    )
    paths = training.export_study_data([root], output_dir=tmp_path / "out")
    index = json.loads(paths["data_index.json"].read_text())
    source_paths = {
        (
            paths["data_index.json"].parent
            / index["source_roots"][row["root"]]
            / row["path"]
        ).resolve()
        for row in index["sources"]
    }
    assert parent_path / "checkpoint_details.json" in source_paths
    assert inherited_actor / "actor_details.json" in source_paths
    selected = _rows(paths["selection/checkpoints.csv"])
    assert selected[0]["checkpoint_id"] == parent["checkpoint_id"]
    assert selected[1]["checkpoint_id"] == child["checkpoint_id"]
    assert _rows(paths["run_index.csv"])[0]["final_actor_path"].endswith(final.name)


@pytest.mark.parametrize("admitted", [False, True])
@pytest.mark.parametrize("seed", [0, 707])
def test_export_keeps_branch_seed_separate_from_parent_seed(
    host_only: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admitted: bool,
    seed: int,
) -> None:
    from tests.test_training_actor_library import _host_run

    root = _study(tmp_path, monkeypatch)
    parent_root = tmp_path / "parent"
    parent, _ = _host_run(parent_root)
    declaration = study._read(root / "declaration.json")
    declaration["cases"] = [
        {
            "id": "child",
            "kind": "extend",
            "duration_seconds": 10,
            "checkpoint": str(parent_root / "checkpoints" / parent["checkpoint_id"]),
            "parent_identity": {
                "checkpoint_id": parent["checkpoint_id"],
                "description_sha256": study._digest(parent),
                "method": "mappo",
            },
            "additional_env_steps": 4,
            "changes": {"seed": seed},
        }
    ]
    study.atomic_json(root / "declaration.json", declaration)
    study.atomic_json(
        root / "study.json",
        {
            "schema_version": 1,
            "status": "pending",
            "cases": {},
            "declaration_sha256": study._digest(declaration),
        },
    )
    if admitted:
        run = root / "cases/child/run"
        run.mkdir(parents=True)
        study.atomic_json(
            run / "run_details.json",
            {
                "run_id": "child",
                "config": parent["metadata"]["config"],
                "continuation": {"changes": {"seed": seed}},
            },
        )
    paths = training.export_study_data([root], output_dir=tmp_path / "out")
    row = _rows(paths["run_index.csv"])[0]
    assert row["seed"] == "42" and row["branch_seed"] == str(seed)
    assert row["actual_env_steps"] == ""


def test_never_started_cases_keep_declared_runtime_identities(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def identities(methods: Sequence[str]) -> dict[str, Any]:
        return {
            method: {"source": "fixture", "dependencies": {"python": "test"}}
            for method in methods
        }

    monkeypatch.setattr(study, "_identities", identities)
    root = _study(tmp_path, monkeypatch)
    declaration = study._read(root / "declaration.json")
    expected = declaration["runtime_identities"][
        declaration["cases"][0]["config"]["method"]
    ]
    paths = training.export_study_data([root], output_dir=tmp_path / "out")
    row = _rows(paths["run_index.csv"])[-1]
    assert row["source_id"] == expected["source"]
    assert json.loads(row["dependency_id"]) == expected["dependencies"]
    assert row["runtime_identity_scope"] == "Declared worker"


def test_final_and_selected_rows_use_the_same_inference_identity(
    host_only: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_training_actor_library import _revise_learner, _signed
    from tests.test_training_progress import _actor

    from marl_battlegrounds.training._checkpoint_metadata import _artifact_identity

    root = _study(tmp_path, monkeypatch)
    run = root / "cases/small--seed-9/run"
    shutil.rmtree(run)
    learner, old_actor = _completed_export_run(run)
    learner["metadata"]["config"]["ppo"]["input_scale"] = 2.0
    learner = _revise_learner(run, learner)
    shutil.rmtree(old_actor)
    actor = _actor(run, learner)
    description = study._read(actor / "actor_details.json")
    description["input_scale"] = 2.0
    description = _signed(description)
    study.atomic_json(actor / "actor_details.json", description)
    inference = _artifact_identity(description)["actor_digest"]
    assert inference != description["actor_digest"]
    study.atomic_json(
        run / "selection.json",
        {
            "checkpoint_id": learner["checkpoint_id"],
            "actor_digest": inference,
        },
    )
    study.atomic_json(
        run / "status.json",
        {
            "run_id": "one-run",
            "status": "complete",
            "env_steps": 0,
            "completed_updates": 0,
            "final_actor": str(actor),
            "selected_actor": str(actor),
        },
    )
    paths = training.export_study_data([root], output_dir=tmp_path / "out")
    rows = _rows(paths["selection/checkpoints.csv"])
    assert [row["actor_digest"] for row in rows] == [inference, inference]
    assert (
        json.loads(rows[1]["records_json"])["actor_digest"]
        == description["actor_digest"]
    )
