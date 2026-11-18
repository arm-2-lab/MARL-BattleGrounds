"""Check that saved evaluation passes resume only as the task they played.

Contracts: a complete saved Capture the Flag pass resumes with task="ctf", or
with no task (an omitted task inherits the saved pass), and plays no more
games; resuming it with task="tdm" is refused. A Team Deathmatch pass saved
without a task resumes with task="tdm" and is refused with task="ctf". A new
Team Deathmatch pass run with task="tdm" writes run_details.json byte for byte
as one run without a task, apart from its run ID and creation time. A run keeps
one metric schema: a Team Deathmatch run, given as a writer or by resume_from,
refuses Capture the Flag full metrics before any game is played and without
adding the pass, and a writer resumed with task="ctf" keeps the run's schema. A
Capture the Flag pass saved before "marlbg.ctf.scalar" existed (its run and
pass say "marlbg.tdm.scalar" 15) still resumes, under that schema. Games are
tiny (two random games of 8 ticks on CPU).
"""

import json
from pathlib import Path
from typing import Any

import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation.run_writer import RunWriter

_RULES: dict[str, Any] = {"num_episodes": 2, "num_envs": 2, "max_steps": 8}


def _evaluate(
    output: dict[str, Path | RunWriter],
    **options: Any,  # noqa: ANN401 - Pass evaluate's keywords through.
) -> marl_bgs.EvaluationResult:
    return marl_bgs.evaluate("random", "random", **_RULES, **output, **options)


def _saved(tmp_path: Path, **options: Any) -> Path:  # noqa: ANN401 - Pass evaluate's keywords through.
    _evaluate({"output_dir": tmp_path}, **options)
    (run_dir,) = tmp_path.iterdir()
    return run_dir


def test_a_saved_capture_the_flag_pass_resumes_only_as_capture_the_flag(
    tmp_path: Path,
) -> None:
    run_dir = _saved(tmp_path, maps=[42], task="ctf")
    for same in ({"task": "ctf"}, {}):
        resumed = _evaluate({"resume_from": run_dir}, maps=[42], **same)
        assert resumed.episodes == ()
        assert resumed.completed_episode_ids == (1, 2)
    with pytest.raises(ValueError, match="differ"):
        _evaluate({"resume_from": run_dir}, maps=[42], task="tdm")
    # Relabel the run as the code before "marlbg.ctf.scalar" saved it.
    path = run_dir / "run_details.json"
    details = json.loads(path.read_text())
    old = {"metric_schema_id": "marlbg.tdm.scalar", "metric_schema_version": 15}
    details.update(old)
    for entry in details["passes"].values():
        entry["details"].update(old)
    path.write_text(json.dumps(details))
    resumed = _evaluate({"resume_from": run_dir}, maps=[42], task="ctf")
    assert resumed.completed_episode_ids == (1, 2)
    assert resumed.metadata["metric_schema_id"] == "marlbg.tdm.scalar"


def test_a_saved_team_deathmatch_pass_resumes_only_as_team_deathmatch(
    tmp_path: Path,
) -> None:
    run_dir = _saved(tmp_path, maps=[47])
    resumed = _evaluate({"resume_from": run_dir}, maps=[47], task="tdm")
    assert resumed.episodes == ()
    with pytest.raises(ValueError, match="differ"):
        _evaluate({"resume_from": run_dir}, maps=[47], task="ctf")


def test_an_explicit_team_deathmatch_task_saves_the_same_run_details(
    tmp_path: Path,
) -> None:
    texts = []
    for name, options in (("omitted", {}), ("explicit", {"task": "tdm"})):
        run_dir = _saved(tmp_path / name, maps=[47], **options)
        text = (run_dir / "run_details.json").read_text()
        details = json.loads(text)
        texts.append(
            text.replace(details["run_id"], "RUN").replace(
                details["created_at"], "TIME"
            )
        )
    assert texts[0] == texts[1]


def test_a_run_keeps_one_metric_schema_for_its_full_metrics(tmp_path: Path) -> None:
    with RunWriter(tmp_path / "tdm") as writer:
        assert writer.metric_schema == ("marlbg.tdm.scalar", 15)
        # A second pass: one pass cannot change its task, but a run can add one.
        with pytest.raises(ValueError, match=r"start a new run for these games$"):
            _evaluate(
                {"writer": writer}, maps=[42], task="ctf", metrics="full", pass_id="2"
            )
        assert writer.completed_episode_ids == frozenset()
    run_dir = _saved(tmp_path / "saved", maps=[47])
    before = (run_dir / "run_details.json").read_bytes()
    with pytest.raises(ValueError, match=r"start a new run for these games$"):
        _evaluate(
            {"resume_from": run_dir}, maps=[42], task="ctf", metrics="full", pass_id="2"
        )
    # Refused before the writer opened, so the run gained no pass.
    assert (run_dir / "run_details.json").read_bytes() == before
    with RunWriter(resume_from=run_dir, pass_id="2", task="ctf") as resumed:
        assert resumed.metric_schema == ("marlbg.tdm.scalar", 15)
