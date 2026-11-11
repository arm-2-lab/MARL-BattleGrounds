"""Check that saved evaluation passes resume only as the task they played.

Contracts: a complete saved Capture the Flag pass resumes with task="ctf", or
with no task (an omitted task inherits the saved pass), and plays no more
games; resuming it with task="tdm" is refused. A Team Deathmatch pass saved
without a task resumes with task="tdm" and is refused with task="ctf". A new
Team Deathmatch pass run with task="tdm" writes run_details.json byte for byte
as one run without a task, apart from its run ID and creation time. Games are
tiny (two random games of 8 ticks on CPU).
"""

import json
from pathlib import Path
from typing import Any

import pytest

import marl_battlegrounds as marl_bgs

_RULES: dict[str, Any] = {"num_episodes": 2, "num_envs": 2, "max_steps": 8}


def _evaluate(output: dict[str, Path], **options: Any) -> marl_bgs.EvaluationResult:  # noqa: ANN401 - Pass evaluate's keywords through.
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
