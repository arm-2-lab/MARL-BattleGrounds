"""Check that evaluation plays and records Capture the Flag games.

Contracts: evaluate(task="ctf") builds every map-ID game in Capture the Flag
and records options["task"] "ctf", the mode's winning score 3, Red Zone depth
0.0 and map evidence that names Capture the Flag's own maps; evaluate_episodes
plays the mode its exact configs hold and records it the same way. A Team
Deathmatch pass records no task key, so an explicit task="tdm" gives the same
options as an omitted one. Before any game or file: a schedule mixing modes, a
neutral or King of the Hills config, an exact config of another mode (the error
names the task to pass), an unknown task ("koth" is told King of the Hills
arrives in v1.2.0), a Capture the Flag depth other than
exactly 0.0, an exact source whose winning score conflicts with
score_threshold, and replays in a Capture the Flag pass are refused. These run
in memory on CPU; saved Capture the Flag passes need captures and outcomes in
Core first.
"""

from pathlib import Path
from typing import Any

import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.tasks import (
    AgentClassName,
    make_standard_capture_the_flag_config,
    make_standard_team_deathmatch_config,
)

_A: tuple[AgentClassName, ...] = ("mage", "warrior")
_B: tuple[AgentClassName, ...] = ("rogue",)
_RULES: dict[str, Any] = {"num_envs": 2, "max_steps": 8}
_CTF = make_standard_capture_the_flag_config(
    map_id=42, team_a_roster=_A, team_b_roster=_B, max_steps=8
)


def _evaluate(**options: Any) -> marl_bgs.EvaluationResult:  # noqa: ANN401 - Pass evaluate's keywords through.
    return marl_bgs.evaluate("random", "random", **{**_RULES, **options})


def test_evaluate_plays_and_records_capture_the_flag() -> None:
    result = _evaluate(
        num_episodes=4, maps=[42, 43], task="ctf", system_roster=_A, opponent_roster=_B
    )
    options = result.metadata["evaluation_contract"]["options"]
    assert (options["task"], options["score_threshold"], options["red_zone_depth"]) == (
        "ctf",
        3,
        0.0,
    )
    modes = {row["task_mode"] for row in result.metadata["configurations"].values()}
    assert modes == {3}
    evidence = [row["map_metadata"] for row in result.metadata["schedule"].values()]
    assert [(rows[0]["value"], rows[2]["value"]) for rows in evidence] == [
        ("42", "registered"),
        ("42", "registered"),
        ("43", "registered"),
        ("43", "registered"),
    ]
    assert evidence[0][1]["value"] == "ctf_map_id_42_cheshire_cat_validation"
    assert len(result.episodes) == 4


def test_exact_schedules_play_the_mode_their_configs_hold() -> None:
    result = evaluate_episodes(
        "random", "random", [EpisodeSpec(1, _CTF), EpisodeSpec(2, _CTF)], num_envs=2
    )
    assert result.metadata["evaluation_contract"]["options"]["task"] == "ctf"
    tdm = make_standard_team_deathmatch_config(
        map_id=42, team_a_roster=_A, team_b_roster=_B
    )
    with pytest.raises(ValueError, match=r"mixes task_mode 1 and 3$"):
        evaluate_episodes(
            "random", "random", [EpisodeSpec(1, tdm), EpisodeSpec(2, _CTF)]
        )
    for mode in (0, 2):
        other = _CTF._replace(task_mode=mode)
        with pytest.raises(ValueError, match=rf"'ctf' configs, not task_mode {mode}$"):
            evaluate_episodes("random", "random", [EpisodeSpec(1, other)])


def test_a_team_deathmatch_pass_records_no_task() -> None:
    omitted = _evaluate(num_episodes=2, maps=[47])
    explicit = _evaluate(num_episodes=2, maps=[47], task="tdm")
    options = omitted.metadata["evaluation_contract"]["options"]
    assert "task" not in options
    assert explicit.metadata["evaluation_contract"]["options"] == options


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"maps": [_CTF]}, r"which task 'tdm' does not play; pass task='ctf'$"),
        (
            {"task": "koth"},
            r"^task must be one of 'tdm', 'ctf', not 'koth' "
            r"\(King of the Hills arrives in v1\.2\.0\)$",
        ),
        ({"task": "ctf", "red_zone_depth": 5.0}, r"ctf takes only 0.0$"),
        ({"task": "ctf", "red_zone_depth": -0.0}, r"ctf takes only 0.0$"),
        (
            {
                "task": "ctf",
                "maps": [_CTF._replace(capture_the_flag_score_threshold=5)],
                "score_threshold": 4,
            },
            r"^score_threshold conflicts with the exact source configuration$",
        ),
        ({"task": "ctf", "save_replays": 1}, r"^Capture the Flag replays arrive"),
    ],
)
def test_bad_capture_the_flag_requests_fail_before_any_file(
    options: dict[str, Any], message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        _evaluate(**{"num_episodes": 2, "maps": [42], **options}, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
