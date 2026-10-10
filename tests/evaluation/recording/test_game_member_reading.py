"""Check how saved results show which member played each slot of each game.

Contracts: load_results reads policy_assignments with typed columns, the
recorded System's name and a member label that stays unique within its System
("name #<policy_id>" when two components share a name); metadata counts each
member System's recorded and unrecorded games; system_metrics(by="game") adds
agent_<k>_member for every slot from the System's own side, member for the
System's distinct labels joined in slot order (a pool of teams names the drawn
team) and opponent_member for the other side, also in self-play; games that
already have per-decision rows get no second summary (reading would refuse the
overlap); a run saved without summary rows, as before this record existed,
shows members only for its replayed games and counts the rest as not recorded;
and in memory the table refuses with the reason while the member columns stay
present and empty. One small saved self-play run of a pool of two teams is
shared by the checks.
"""

# pyright: reportPrivateUsage=false
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
)
from marl_battlegrounds.evaluation.results import SavedResults, load_results
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.policies.actor import ActorAction

LABELS = {"same #0", "b", "same #2", "c"}
DRAWS = {"same #0 / b", "same #2 / c"}


def _act(
    variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, PolicyTree]:
    del variables, keys
    still = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    return ActorAction(still, still, still), memory


def _still(name: str) -> System:
    # A tiny System that stands still; small programs keep compiling cheap.
    return System(name, _act)


def _pool() -> System:
    slots = [[0, 1], [2, 3, 4]]
    return marl_bgs.pool(
        {
            marl_bgs.team(_still("same"), _still("b"), slots=slots): 1,
            marl_bgs.team(_still("same"), _still("c"), slots=slots): 1,
        }
    )


def _saved(path: Path) -> SavedResults:
    method = _pool()
    result = marl_bgs.evaluate(
        method,
        method,
        maps=[0],
        num_episodes=6,
        num_envs=3,
        max_steps=3,
        metrics="priority",
        replay_episodes=(2, 4, 6),
        output_dir=path,
    )
    assert result.paths is not None
    return load_results(result.paths["run_details"].parent)


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> SavedResults:
    return _saved(tmp_path_factory.mktemp("members"))


def test_table_has_types_names_and_unique_labels(saved: SavedResults) -> None:
    description = saved.metadata["tables"]["policy_assignments"]
    assert description["availability"] == "available"
    assert description["columns"][-2:] == ("system_name", "member")
    assert description["column_types"]["team"][0] == "text"
    table = saved.table("policy_assignments")
    assert table["policy_id"].dtype == np.int64
    assert set(table["member"].tolist()) <= LABELS
    assert set(table["system_name"].tolist()) == {
        "Pool: Team: same / b / Team: same / c"
    }
    (coverage,) = description["member_coverage"].values()
    assert coverage == {
        "system_name": "Pool: Team: same / b / Team: same / c",
        "recorded": 6,
        "not_recorded": 0,
    }


def test_game_rows_name_each_sides_members(saved: SavedResults) -> None:
    games = saved.system_metrics(
        "priority_metrics", by="game", columns=["team_a_kills"]
    )
    # Self-play: one row per game, read as Team A.
    assert len(games["system_id"]) == 6
    assert set(games["member"].tolist()) <= DRAWS
    assert set(games["opponent_member"].tolist()) <= DRAWS
    for row in range(6):
        own = [games[f"agent_{slot}_member"][row] for slot in range(5)]
        assert " / ".join(dict.fromkeys(own)) == games["member"][row]
        enemy = [games[f"agent_{slot}_member"][row] for slot in range(5, 10)]
        assert " / ".join(dict.fromkeys(enemy)) == games["opponent_member"][row]
        # The two slots of the first team member share its label.
        assert games["agent_0_member"][row] == games["agent_1_member"][row]


def test_runs_saved_before_this_record_show_replayed_games_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def skip(self: RunWriter, members: object) -> None:
        del self, members

    monkeypatch.setattr(RunWriter, "_stage_members", skip)
    saved = _saved(tmp_path)
    description = saved.metadata["tables"]["policy_assignments"]
    (coverage,) = description["member_coverage"].values()
    assert (coverage["recorded"], coverage["not_recorded"]) == (3, 3)
    games = saved.system_metrics(
        "priority_metrics", by="game", columns=["team_a_kills"]
    )
    named = {
        int(episode): member
        for episode, member in zip(games["episode_id"], games["member"], strict=True)
    }
    assert {episode for episode, member in named.items() if member} == {2, 4, 6}


def test_in_memory_results_refuse_with_the_reason() -> None:
    method = _pool()
    result = marl_bgs.evaluate(
        method, method, maps=[0], num_episodes=2, num_envs=2, max_steps=1
    )
    with pytest.raises(ValueError, match="recorded in saved runs; pass output_dir"):
        result.table("policy_assignments")
    games = result.system_metrics(
        "priority_metrics", by="game", columns=["team_a_kills"]
    )
    assert set(games["member"].tolist()) == {None}
    assert set(games["agent_9_member"].tolist()) == {None}
