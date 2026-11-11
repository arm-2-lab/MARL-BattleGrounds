"""Check that a small PSRO run plays Capture the Flag in every game it plays.

A tiny recurrent IPPO oracle with task "ctf" and one seeded member (Random)
trains exactly one round. Every game the run plays goes through evaluate with
task "ctf" and Red Zone depth 0.0, the oracle's own rules: the payoff pair
between Random and the new member, the member's validation on a Random panel
and the candidate's confirmation, three game batches in all. The run makes no
learning or speed claim.
"""

from pathlib import Path
from typing import Any

import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training
from marl_battlegrounds.baselines.config import PPOConfig


def test_a_capture_the_flag_psro_run_plays_its_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules: list[tuple[str, float]] = []
    real_evaluate = marl_bgs.evaluate

    def recording_evaluate(*args: Any, **kwargs: Any) -> object:  # noqa: ANN401
        rules.append((kwargs["task"], kwargs["red_zone_depth"]))
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(marl_bgs, "evaluate", recording_evaluate)
    oracle = training.TrainConfig(
        seed=710,
        method="ippo",
        task="ctf",
        num_envs=8,
        # One 256-step round plus the six games PSRO reserves at 300 steps
        # each, so a second round never fits.
        total_env_steps=256 + 6 * 300,
        ppo=PPOConfig(rollout_length=4, epochs=1, groups=2, minibatches=2),
        validation_maps=(42,),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
    )
    result = training.train_psro(
        oracle,
        output_dir=tmp_path / "run",
        seed_members={"random": "random"},
        panel={"random": "random"},
        round_env_steps=256,
        payoff_games=2,
        payoff_maps=(0,),
        eval_num_envs=2,
        shortlist=0,
    )
    assert [entry["round"] for entry in result["rounds"]] == [1]
    assert rules == [("ctf", 0.0)] * 3
