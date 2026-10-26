"""Check a whole small PSRO run on the CPU: rounds, records, resume and entrant.

Contracts: train_psro with scripted seeded members (Random and TDM-Alpha) and
a tiny recurrent IPPO oracle plays the seeded pair once, trains round 1 with
train and later rounds with extend_training into rounds/round-NN, with the
earlier members as weighted opponents; records one payoff row per member
pair with its seed and game steps; keeps every game step (training, payoff
and validation) inside total_env_steps; validates each round's newest member
on the panel; takes as each round's candidate the Nash mixture of PSRO's own
members that the members' routine point margins prefer, and scores it, with
no games, as the Nash weighted mean of the members' own routine results; and
writes entrant.json naming the
confirmed round, its members with positive weight and the shortlist, which
always holds the last round. A run stopped inside round 2's training, after
its first update, continues through the trainer's own resume, and a run
stopped during a round's validation continues without retraining a round or
replaying a recorded pair; a call on a finished run plays no game and returns
the same entrant; a call with other settings and a seeded name starting with
"psro-" are refused. load_psro_entrant returns a System that plays a game, and
refuses a folder with no entrant.json.
"""

import json
from pathlib import Path
from typing import Any

import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training
from marl_battlegrounds.baselines.config import PPOConfig
from marl_battlegrounds.training import psro

ORACLE = training.TrainConfig(
    seed=710,
    method="ippo",
    num_envs=32,
    total_env_steps=12288,
    ppo=PPOConfig(rollout_length=4, epochs=1, groups=2, minibatches=2),
    validation_maps=(42,),
    routine_seed_pairs=1,
    confirmation_seed_pairs=1,
)
ARGUMENTS: dict[str, Any] = {
    "seed_members": {"random": "random", "tdm-alpha": "tdm-alpha"},
    "panel": {"tdm-beta": "tdm-beta"},
    "round_env_steps": 2048,
    "payoff_games": 2,
    "payoff_maps": (0, 1),
    "eval_num_envs": 2,
}


def test_small_run_records_resumes_and_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    real_extend, real_validate = training.extend_training, psro.validate

    def stop_in_training(*args: Any, **kwargs: Any) -> object:  # noqa: ANN401
        def stop(row: object) -> None:
            raise RuntimeError("stopped in training")

        return real_extend(*args, on_update=stop, **kwargs)

    def stop_in_validation(*args: object, **kwargs: object) -> dict[str, Any]:
        raise RuntimeError("stopped in validation")

    def statuses() -> list[str]:
        rounds = json.loads((run / "rounds.json").read_text(encoding="utf-8"))
        return [entry["status"] for entry in rounds]

    monkeypatch.setattr(training, "extend_training", stop_in_training)
    with pytest.raises(RuntimeError, match="on_update failed"):
        training.train_psro(ORACLE, output_dir=run, **ARGUMENTS)
    assert statuses() == ["complete", "training"]
    monkeypatch.setattr(training, "extend_training", real_extend)
    monkeypatch.setattr(psro, "validate", stop_in_validation)
    with pytest.raises(RuntimeError, match="stopped in validation"):
        training.train_psro(ORACLE, output_dir=run, **ARGUMENTS)
    assert statuses() == ["complete", "trained"]
    monkeypatch.setattr(psro, "validate", real_validate)
    result = training.train_psro(ORACLE, output_dir=run, **ARGUMENTS)

    rounds = result["rounds"]
    assert len(rounds) >= 2 and all(entry["status"] == "complete" for entry in rounds)
    assert sorted(path.name for path in (run / "rounds").iterdir()) == [
        entry["run_dir"].split("/")[-1] for entry in rounds
    ]
    assert rounds[0]["opponent_weights"].keys() == {"random", "tdm-alpha"}
    assert "psro-01" in rounds[1]["opponent_weights"]
    names = ["random", "tdm-alpha", *(entry["member"] for entry in rounds)]
    rows = psro._read_payoff(run / "payoff.csv")  # pyright: ignore[reportPrivateUsage]
    pairs = [frozenset((row["member"], row["opponent"])) for row in rows]
    assert len(pairs) == len(set(pairs)) == len(names) * (len(names) - 1) // 2
    assert [row["seed"] for row in rows] == [20261021 + k for k in range(len(rows))]
    assert all(row["wins"] + row["draws"] + row["losses"] == 2 for row in rows)
    members = {entry["member"]: entry["validation"]["member"] for entry in rounds}
    played = [
        psro.PairResult(r["member"], r["opponent"], r["wins"], r["draws"], r["losses"])
        for r in rows
    ]
    for entry in rounds:
        own = [e["member"] for e in rounds[: entry["round"]]]
        table = psro.score_table(
            own, [r for r in played if {r.member, r.opponent} <= set(own)]
        )
        margins = [members[name]["point_margin"] for name in own]
        assert list(entry["candidate_weights"].values()) == pytest.approx(
            psro.nash_mixture(table, preference=margins).tolist()
        )
        assert entry["validation"]["member"]["games"] == 2
        candidate = entry["validation"]["candidate"]
        assert (candidate["games"], candidate["game_steps"]) == (0, 0)
        assert candidate["point_margin"] == pytest.approx(
            sum(
                weight * members[name]["point_margin"]
                for name, weight in entry["candidate_weights"].items()
            )
        )
    entrant = result["entrant"]
    assert entrant["ledger_env_steps"] <= entrant["budget_env_steps"] == 12288
    assert entrant["round"] in entrant["shortlist"]
    assert rounds[-1]["round"] in entrant["shortlist"]
    assert set(entrant["members"]) <= {entry["member"] for entry in rounds}
    assert json.loads((run / "entrant.json").read_text(encoding="utf-8")) == entrant

    def no_games(*args: object, **kwargs: object) -> None:
        raise AssertionError("a finished run played a game")

    monkeypatch.setattr(psro, "play_pair", no_games)
    monkeypatch.setattr(psro, "validate", no_games)
    again = training.train_psro(ORACLE, output_dir=run, **ARGUMENTS)
    assert again["entrant"] == entrant
    other_seed: dict[str, Any] = {**ARGUMENTS, "payoff_seed": 7}
    with pytest.raises(ValueError, match="other settings"):
        training.train_psro(ORACLE, output_dir=run, **other_seed)
    own_name: dict[str, Any] = {**ARGUMENTS, "seed_members": {"psro-01": "random"}}
    with pytest.raises(ValueError, match="psro-"):
        training.train_psro(ORACLE, output_dir=tmp_path / "other", **own_name)

    system = training.load_psro_entrant(run)
    played = marl_bgs.evaluate(
        system, "tdm-beta", num_episodes=2, maps=[42], num_envs=2, metrics="none"
    )
    assert len(played.episodes) == 2
    with pytest.raises(ValueError, match=r"no entrant\.json"):
        training.load_psro_entrant(tmp_path)
