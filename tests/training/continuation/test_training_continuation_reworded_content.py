"""Check that a run keeps resuming after a display-only change to its content.

A short recurrent IPPO run with a past copy stops after its first checkpoint
and its one history capture. The installed content then differs from the saved
binding only in one protected scenario's packaging hash, which rewording that
scenario's notes changes, while its scientific content stays the same. The run
resumes, stops again after a later checkpoint and resumes to the end. Every
checkpoint and the history actor keep the first binding, so the second
resume's history check, which compares the actor with its run's latest
checkpoint, sees one binding. Before that, a scientifically different binding
(another Red Zone depth) that reaches the same check is refused, and nothing
is written.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests.helpers.training_continuation_helpers import fixed_source

import marl_battlegrounds.training._content as _content
from marl_battlegrounds._task_settings import TaskName
from marl_battlegrounds.baselines.config import PPOConfig
from marl_battlegrounds.modes.team_deathmatch._assets import TDMScenarioInfo
from marl_battlegrounds.training import checkpoints, runner


def _config() -> runner.TrainConfig:
    return runner.TrainConfig(
        method="ippo",
        num_envs=4,
        total_env_steps=64,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        opponents={"self": 0.5, "past": 0.5},
        keep_past=2,
        history_capture_env_steps=(16,),
        checkpoint_interval_updates=2,
        metrics="none",
        verbose=False,
    )


def _stop_at(update: int) -> Callable[[dict[str, Any]], None]:
    def stop(row: dict[str, Any]) -> None:
        if row["update_index"] == update:
            raise RuntimeError("stopped for the test")

    return stop


def _scenario_reworded(monkeypatch: pytest.MonkeyPatch) -> None:
    scenarios = _content.scenarios_for_task("tdm")
    original = scenarios[0]
    reworded = original.model_copy(update={"resource_sha256": "0" * 64})
    read = _content.scenario_content

    def listed(task: TaskName) -> tuple[TDMScenarioInfo, ...]:
        return (reworded, *scenarios[1:]) if task == "tdm" else ()

    monkeypatch.setattr(_content, "scenarios_for_task", listed)

    def read_reworded(info: TDMScenarioInfo) -> object:
        return read(original if info == reworded else info)

    monkeypatch.setattr(_content, "scenario_content", read_reworded)


def _binding(path: Path) -> object:
    # Older checkpoints keep only their description once retention prunes them.
    details = json.loads((path / "checkpoint_details.json").read_text())
    return details["collection"]["content_binding"]


def _other_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare = _content.prepare_training_content

    def other_depth(
        *,
        expected: object = None,
        score_thresholds: tuple[int, ...] | None = None,
        red_zone_depth: float | None = None,
        task: TaskName | None = None,
    ) -> _content.PreparedTrainingContent:
        # Skip the early saved-content check so the collection check decides.
        del expected, red_zone_depth
        return prepare(score_thresholds=score_thresholds, red_zone_depth=0.0, task=task)

    monkeypatch.setattr(_content, "prepare_training_content", other_depth)


def test_reworded_content_resumes_on_one_binding_and_other_science_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_source(monkeypatch, "ippo")
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="on_update failed"):
        runner.train(_config(), output_dir=root, on_update=_stop_at(3))
    first = _binding(next((root / "checkpoints").iterdir()))
    before = sorted(str(path) for path in root.rglob("*"))
    with monkeypatch.context() as scientific:
        _other_depth(scientific)
        with pytest.raises(ValueError, match="Checkpoint content or collection"):
            runner.train(resume_from=root)
    assert sorted(str(path) for path in root.rglob("*")) == before
    _scenario_reworded(monkeypatch)
    with pytest.raises(RuntimeError, match="on_update failed"):
        runner.train(resume_from=root, on_update=_stop_at(7))
    result = runner.train(resume_from=root)
    assert result.completed_env_steps == 64
    saved = sorted((root / "checkpoints").iterdir())
    assert len(saved) >= 3
    assert all(_binding(path) == first for path in saved)
    copies = sorted((root / "past_copies").iterdir())
    assert len(copies) == 1
    for copy in copies:
        details = checkpoints.read_checkpoint_description(copy)
        assert details["metadata"]["capture_origin"]["content_binding"] == first
