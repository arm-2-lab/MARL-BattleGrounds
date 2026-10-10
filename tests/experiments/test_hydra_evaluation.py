"""Exercise all six evaluation launchers through the shared public API.

Use real untrained actor exports, saved result/replay reads and exact recovery.
Host checks cover sparse settings, identities, path safety, import costs and
failure order. Tiny games prove software behavior, not learned competence.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import inspect
import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import jax
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines import ppo, pqn, qmix
from marl_battlegrounds.evaluation.replay_io import load_replay
from marl_battlegrounds.evaluation.results import EvaluationResult
from marl_battlegrounds.experiments import hydra as launcher
from marl_battlegrounds.training.checkpoints import export_system

METHODS = {
    "mappo": "recurrent_mappo",
    "ff_mappo": "feedforward_mappo",
    "ippo": "recurrent_ippo",
    "ff_ippo": "feedforward_ippo",
    "qmix": "qmix",
    "pqn_vdn": "pqn_vdn",
}
REPO = Path(__file__).resolve().parents[2]
type Tree = Any


@pytest.fixture(scope="module", params=tuple(METHODS))
def actor(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, Path]:
    method = str(request.param)
    key = jax.random.key(5)
    if method == "qmix":
        weights = qmix.initialize_qmix(key).online_q
    elif method == "pqn_vdn":
        weights = pqn.initialize_pqn(key, planned_learning_blocks=1).network
    else:
        weights = ppo.initialize_ppo(key, method=method).actor_params
    root = tmp_path_factory.mktemp("evaluation-actors")
    path = export_system(
        weights,
        root / "misleading-name",
        method=method,
        spawn_frame="left",
        metadata={
            "run_id": "evaluation-software-check",
            "seed": 5,
            "env_steps": 0,
            "checkpoint_id": "b" * 64,
            **({"optimizer_steps": 0} if method in {"qmix", "pqn_vdn"} else {}),
        },
    )
    return method, path


def _settings(path: Path, **changes: object) -> dict[str, Any]:
    return {
        "checkpoint": str(path),
        "opponent": "random",
        "evaluation": {
            "num_episodes": 2,
            "maps": [47],
            "spawn_mode": "paired",
            "seed": 41,
            "num_envs": 2,
            "keep_batch_size": True,
            "max_steps": 2,
            "system_roster": ["mage", "priest"],
            "opponent_roster": ["rogue"],
            "metrics": "priority",
            "replay_episodes": [1],
            "chunk_size": 1,
            **changes,
        },
    }


def _run(
    settings: dict[str, Any], method: str, root: Path
) -> EvaluationResult | dict[str, Any]:
    return launcher.run_evaluation(
        settings, expected_method=method, base_dir=root, launch_dir=root / "launch"
    )


def test_every_serializable_public_evaluation_keyword_has_one_route() -> None:
    signature = inspect.signature(marl_bgs.evaluate)
    keywords = {
        name
        for name, item in signature.parameters.items()
        if item.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert launcher._EVALUATION_FIELDS | {"writer"} == keywords


@pytest.mark.parametrize("method", tuple(METHODS))
@pytest.mark.parametrize("arguments", [["--help"], ["--cfg", "job"]])
def test_six_entry_help_and_settings_need_no_numerical_imports(
    method: str, arguments: list[str], tmp_path: Path
) -> None:
    script = REPO / "examples" / "evaluate" / f"evaluate_{METHODS[method]}.py"
    program = f"""
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'jax', 'jaxlib', 'flax', 'optax', 'numpy'}}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
runpy.run_path({str(script)!r}, run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, *arguments],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checkpoint" in result.stdout
    assert not list(tmp_path.iterdir())


def test_six_methods_evaluate_and_reload_the_same_results_as_python(
    actor: tuple[str, Path], tmp_path: Path
) -> None:
    method, path = actor
    settings = _settings(
        path,
        output_dir=str(tmp_path / "results"),
        metrics="full"
        if method == "qmix"
        else "none"
        if method == "ff_ippo"
        else "priority",
    )
    result = _run(settings, method, tmp_path)
    assert isinstance(result, EvaluationResult)
    assert result.run_dir is not None
    assert result.status == "complete" and result.completed_episode_ids == (1, 2)
    saved = marl_bgs.load_results(result.run_dir)
    assert saved.status == "complete"
    assert len(saved.table("episodes")["episode_id"]) == 2
    expected_settings = {
        k: v for k, v in settings["evaluation"].items() if k != "output_dir"
    }
    expected = marl_bgs.evaluate(
        marl_bgs.load_method(str(path)), "random", **expected_settings
    )
    assert result.episodes == expected.episodes
    assert len(result.replay_paths) == len(expected.replays) == 1
    replay = load_replay(result.replay_paths[0]).replay
    for field in ("frames", "transitions"):
        actual_rows = [
            json.loads(
                item.model_dump_json().replace(result.metadata["run_id"], "<run>")
            )
            for item in getattr(replay, field)
        ]
        expected_rows = [
            json.loads(
                item.model_dump_json().replace(expected.metadata["run_id"], "<run>")
            )
            for item in getattr(expected.replays[0], field)
        ]
        assert actual_rows == expected_rows
    receipt = json.loads(
        (tmp_path / "launch/.hydra/evaluation_result.json").read_text()
    )
    assert receipt["run_dir"] == str(result.run_dir)
    assert result.run_dir.parent == tmp_path / "results"


def test_wrong_method_fails_before_load_or_any_output(
    actor: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, path = actor

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Model work started")

    monkeypatch.setattr(marl_bgs, "load_method", forbidden)
    monkeypatch.setattr(marl_bgs, "evaluate", forbidden)
    wrong = next(item for item in METHODS if item != method)
    with pytest.raises(ValueError, match="checkpoint is"):
        _run(_settings(path), wrong, tmp_path)
    assert not list(tmp_path.iterdir())


def test_preview_reads_metadata_but_never_loads_or_calls_factory(
    actor: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, path = actor

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Numerical work started")

    monkeypatch.setattr(marl_bgs, "load_method", forbidden)
    settings = _settings(path)
    settings.update(preview=True, opponent="unavailable_factory:make")
    result = _run(settings, method, tmp_path)
    assert isinstance(result, dict)
    assert result["preview"] and result["observed_method"] == method
    assert "Payload verification" in result["pending_checks"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("destination", ["launch", "output", "cache", "receipts"])
def test_paths_reject_writes_inside_input_actors_before_output(
    actor: tuple[str, Path], tmp_path: Path, destination: str
) -> None:
    method, path = actor
    settings = _settings(path)
    launch = tmp_path / "launch"
    if destination == "launch":
        launch = path / "new-output"
    elif destination == "output":
        settings["evaluation"]["output_dir"] = str(path / "new-output")
    elif destination == "cache":
        settings["resources"] = {
            "devices": [{"device": "cpu", "max_workers": 1}],
            "compilation_cache": str(path / "new-output"),
        }
    else:
        launch.mkdir()
        (launch / ".hydra").symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError, match=r"input artifacts|symbolic links"):
        launcher.run_evaluation(
            settings, expected_method=method, base_dir=tmp_path, launch_dir=launch
        )
    assert not (path / "new-output").exists()
    assert not (path / "evaluation_request.json").exists()


def test_loaded_actor_must_match_the_preread_identity(
    actor: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    method, path = actor

    def changed(reference: str) -> SimpleNamespace:
        return SimpleNamespace(checkpoint="changed")

    monkeypatch.setattr(marl_bgs, "load_method", changed)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Games started with the wrong actor")

    monkeypatch.setattr(marl_bgs, "evaluate", forbidden)
    with pytest.raises(ValueError, match="identity changed"):
        _run(_settings(path), method, tmp_path)


def test_sparse_resume_count_belongs_to_the_selected_pass(
    actor: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    method, path = actor
    settings = _settings(path)
    settings["evaluation"] = {"resume_from": str(tmp_path / "saved")}

    def missing(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            metadata={"num_episodes": 999, "passes": {"one": {"details": {}}}}
        )

    monkeypatch.setattr(marl_bgs, "load_results", missing)
    with pytest.raises(ValueError, match="selected pass"):
        _run(settings, method, tmp_path)
    assert not (tmp_path / "launch").exists()


def test_hydra_rejects_actor_child_before_creating_launch_records(
    actor: tuple[str, Path], tmp_path: Path
) -> None:
    method, path = actor
    target = path / "forbidden-launch"
    script = REPO / "examples" / "evaluate" / f"evaluate_{METHODS[method]}.py"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            f"checkpoint={path}",
            "preview=true",
            f"hydra.run.dir={target}",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "outside input artifacts" in result.stderr
    assert not target.exists()


def test_shared_reference_loads_and_verifies_payload_once(
    actor: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training import checkpoints

    method, path = actor
    checks: list[Path] = []
    original = checkpoints.read_checkpoint_details

    def counted(source: str | Path) -> dict[str, Any]:
        checks.append(Path(source))
        return original(source)

    monkeypatch.setattr(checkpoints, "read_checkpoint_details", counted)

    def receive(left: object, right: object, **kwargs: object) -> EvaluationResult:
        assert left is right
        raise RuntimeError("Stop before games")

    monkeypatch.setattr(marl_bgs, "evaluate", receive)
    settings = _settings(path)
    settings["opponent"] = str(path / ".." / path.name)
    with pytest.raises(RuntimeError, match="Stop before games"):
        _run(settings, method, tmp_path)
    assert checks == [path]


def test_sparse_resume_keeps_original_inputs_and_only_plays_missing_games(
    actor: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    import numpy as np

    from marl_battlegrounds.evaluation._recording.recording_context import (
        capture_recording_provenance,
    )

    method, path = actor
    if method != "ff_ippo":
        pytest.skip("One shared recovery owner; six families load in the paired proof")
    runner = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance()

    def same_source(**kwargs: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(runner, "capture_recording_provenance", same_source)
    original = runner._jax_system_chunk

    def interrupted(*args: Tree, **kwargs: Tree) -> Tree:
        if np.max(np.asarray(args[5].state.episode_id)) >= 3:
            raise RuntimeError("Interrupted after durable games")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_jax_system_chunk", interrupted)
    fresh = _settings(
        path,
        output_dir=str(tmp_path / "results"),
        num_episodes=4,
        max_steps=1,
        red_zone_depth=0,
        phase="software",
        pass_id="saved",
    )
    with pytest.raises(
        RuntimeError, match="Interrupted after durable games"
    ) as failure:
        _run(fresh, method, tmp_path)
    run = next((tmp_path / "results").iterdir())
    assert str(run) in str(getattr(failure.value, "__notes__", []))
    saved = marl_bgs.load_results(run, phase="software", pass_id="saved")
    assert len(saved.table("episodes")["episode_id"]) == 2
    before = {
        p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in run.rglob("*")
        if p.is_file()
    }
    resume: dict[str, Any] = {
        "checkpoint": str(path),
        "opponent": "random",
        "evaluation": {
            "resume_from": str(run),
            "num_envs": 2,
            "phase": "software",
            "pass_id": "saved",
        },
    }
    changed = {**resume, "evaluation": {**resume["evaluation"], "seed": 999}}
    with pytest.raises(ValueError, match="seed"):
        _run(changed, method, tmp_path / "rejected")
    assert before == {
        p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in run.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(runner, "_jax_system_chunk", original)
    result = _run(resume, method, tmp_path / "resumed")
    assert isinstance(result, EvaluationResult) and result.status == "complete"
    assert [row.episode_id for row in result.episodes] == [3, 4]
    assert result.completed_episode_ids == (1, 2, 3, 4)
    complete = marl_bgs.load_results(run, phase="software", pass_id="saved")
    assert tuple(complete.table("episodes")["episode_id"]) == (1, 2, 3, 4)
    assert len(saved.table("episodes")["episode_id"]) == 2
    request = json.loads(
        (tmp_path / "resumed/launch/.hydra/evaluation_request.json").read_text()
    )
    assert request["effective_evaluation"]["num_episodes"] == 4
    assert "seed" not in request["effective_evaluation"]
    assert "red_zone_depth" not in request["effective_evaluation"]


def test_linked_cache_fails_on_the_direct_adapter_route(
    actor: tuple[str, Path], tmp_path: Path
) -> None:
    method, path = actor
    cache = tmp_path / "linked-cache"
    cache.symlink_to(path, target_is_directory=True)
    settings = _settings(path)
    settings["resources"] = {
        "devices": [{"device": "cpu", "max_workers": 1}],
        "compilation_cache": str(cache),
    }
    with pytest.raises(ValueError, match="symbolic links"):
        _run(settings, method, tmp_path)
    assert not (tmp_path / "launch").exists()


def test_failed_launcher_receipt_still_names_the_saved_evaluation(
    actor: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from marl_battlegrounds.training import _run_io

    method, path = actor
    saved = tmp_path / "saved-results"

    def evaluated(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(run_dir=saved, status="complete", metadata={})

    original = _run_io.atomic_json

    def receipt(destination: Path, value: dict[str, Any]) -> None:
        if destination.name == "evaluation_result.json":
            raise OSError("Receipt disk failure")
        original(destination, value)

    monkeypatch.setattr(marl_bgs, "evaluate", evaluated)
    monkeypatch.setattr(_run_io, "atomic_json", receipt)
    with pytest.raises(OSError, match="Receipt disk failure") as failure:
        _run(_settings(path), method, tmp_path)
    assert str(saved) in str(getattr(failure.value, "__notes__", []))


def test_changed_weights_reuse_compilation_and_adapter_cost_is_visible(
    actor: tuple[str, Path], tmp_path: Path
) -> None:
    import statistics
    import time

    method, path = actor
    if method != "ff_mappo":
        pytest.skip("One same-schema cost probe; all methods have functional proof")
    runner = import_module("marl_battlegrounds.evaluation.evaluate")
    settings = _settings(path, max_steps=1, metrics="none", replay_episodes=[])
    _run(settings, method, tmp_path / "warmup")
    compiled = runner._jax_system_chunk._cache_size()
    second = export_system(
        ppo.initialize_ppo(jax.random.key(6), method=method).actor_params,
        tmp_path / "different-weights",
        method=method,
        spawn_frame="left",
        metadata={
            "run_id": "second",
            "seed": 6,
            "env_steps": 0,
            "checkpoint_id": "c" * 64,
        },
    )
    other = {**settings, "checkpoint": str(second)}
    _run(other, method, tmp_path / "second")
    assert runner._jax_system_chunk._cache_size() == compiled
    elapsed: dict[str, list[float]] = {"direct_seconds": [], "adapter_seconds": []}
    for index in range(3):
        start = time.perf_counter()
        direct = marl_bgs.evaluate(
            marl_bgs.load_method(str(path)), "random", **settings["evaluation"]
        )
        elapsed["direct_seconds"].append(time.perf_counter() - start)
        start = time.perf_counter()
        actual = _run(settings, method, tmp_path / f"measured-{index}")
        elapsed["adapter_seconds"].append(time.perf_counter() - start)
        assert isinstance(actual, EvaluationResult)
        assert actual.episodes == direct.episodes
        assert actual.run_dir is None and actual.replays == ()
    assert runner._jax_system_chunk._cache_size() == compiled
    (tmp_path / "costs.json").write_text(
        json.dumps(
            {
                **elapsed,
                "medians": {
                    key: statistics.median(value) for key, value in elapsed.items()
                },
                "scope": (
                    "Warm CPU, two paired one-step games, B2; "
                    "load and complete API call"
                ),
                "compile_cache_entries": compiled,
                "launch_bytes": sum(
                    p.stat().st_size
                    for folder in tmp_path.glob("measured-*")
                    for p in folder.rglob("*")
                    if p.is_file()
                ),
                "limits": "No GPU, peak RAM or transfer measurement; no learning claim",
            },
            indent=2,
        )
        + "\n"
    )


def test_all_settings_forward_without_filling_omitted_fields(
    actor: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, path = actor
    forwarded: dict[str, object] = {}

    def receive(*args: object, **kwargs: object) -> EvaluationResult:
        forwarded.update(kwargs)
        raise RuntimeError("Checked forwarding")

    monkeypatch.setattr(marl_bgs, "evaluate", receive)
    values: dict[str, Any] = {
        "num_episodes": 4,
        "maps": [47, 48],
        "spawn_mode": "paired",
        "system_roster": ["mage", "mage"],
        "opponent_roster": None,
        "seed": 88,
        "num_envs": 2,
        "keep_batch_size": False,
        "metrics": "none",
        "full_metrics_episodes": [1],
        "replay_episodes": [2],
        "save_replays": 0,
        "output_dir": "outputs",
        "score_threshold": 7,
        "max_steps": 3,
        "red_zone_depth": 0,
        "phase": "custom",
        "pass_id": "fixed",
        "chunk_size": 2,
    }
    with pytest.raises(RuntimeError, match="Checked forwarding"):
        _run(
            {"checkpoint": str(path), "opponent": "random", "evaluation": values},
            method,
            tmp_path,
        )
    assert forwarded == {
        **values,
        "output_dir": str(tmp_path / "outputs"),
        "red_zone_depth": 0.0,
    }
    assert type(forwarded["red_zone_depth"]) is float
    assert "resume_from" not in forwarded


@pytest.mark.parametrize("fails", [False, True])
def test_factory_execution_uses_original_cwd_even_when_hydra_changes_it(
    actor: tuple[str, Path], tmp_path: Path, fails: bool
) -> None:
    method, path = actor
    if method != "ff_mappo":
        pytest.skip("One common launcher owns cwd across all six entries")
    (tmp_path / "factory-value.txt").write_text("original directory")
    (tmp_path / "opponent_factory.py").write_text("""from pathlib import Path
import marl_battlegrounds as marl_bgs

def make():
    assert Path('factory-value.txt').read_text() == 'original directory'
    print('FACTORY ORIGINAL CWD')
    return marl_bgs.policy('random')
""")
    program = f"""
from pathlib import Path
from types import SimpleNamespace
import marl_battlegrounds as marl_bgs
from marl_battlegrounds.experiments.hydra import evaluation_main
original = Path.cwd()
def evaluated(*args, **kwargs):
    assert Path.cwd() == original
    if {fails!r}:
        raise RuntimeError('Declared failure after factory')
    return SimpleNamespace(run_dir=None, status='complete', metadata={{}})
marl_bgs.evaluate = evaluated
try:
    status = evaluation_main({method!r})
except SystemExit:
    assert Path.cwd() == original
    print('CWD RESTORED AFTER FAILURE')
    raise
assert Path.cwd() == original
print('CWD RESTORED AFTER SUCCESS')
raise SystemExit(status)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            program,
            f"checkpoint={path}",
            "opponent=opponent_factory:make",
            "hydra.job.chdir=true",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO / "src"))),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == (1 if fails else 0), result.stdout + result.stderr
    assert "FACTORY ORIGINAL CWD" in result.stdout
    assert "CWD RESTORED AFTER " + ("FAILURE" if fails else "SUCCESS") in result.stdout
