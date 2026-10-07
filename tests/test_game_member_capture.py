"""Check that saved evaluations record which member played each slot of each game.

Contracts: a System with two or more components gets one policy_assignments row
per active slot for every saved game without per-decision rows; those rows
equal the merged per-decision rows of the same games played with replays, up to
the first change of member, after which one row with policy ID -1 runs to the
end of the game; games with per-decision rows keep only those rows; games of
different lengths, refills inside a chunk, an authored start and inactive
slots all record correctly, and the pool's draws differ between games so a
mixed-up game would show; with a one-game buffer, every game's member rows
flush together with its outcome, never earlier or later; a host-execution
System records the same rows as its JAX twin; and a matchup without members
keeps today's carry and completion leaves, so it compiles the same program.
"""

# pyright: reportPrivateUsage=false
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core import env as core
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.evaluation.results import load_results
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

# The module, not the evaluate function the package exports under its name.
evaluation = import_module("marl_battlegrounds.evaluation.evaluate")
SWITCH = 3
LENGTHS = (5, 11, 3, 8, 2, 7, 1, 4)
REPLAYED = tuple(range(2, len(LENGTHS) + 1, 2))


def _init(variables: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.valid.shape[0], jnp.int32)


def _apply(
    variables: PolicyTree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    # Reports member 0 for its first SWITCH decisions, then member 1.
    del variables, keys
    still = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    ids = jnp.where(memory[:, None] >= SWITCH, 1, 0)
    ids = jnp.where(inputs.active_mask, ids, -1).astype(jnp.int32)
    following = jnp.where(inputs.valid, memory + 1, memory)
    return SystemOutput(ActorAction(still, still, still), following, policy_ids=ids)


def _host_init(variables: PolicyTree, inputs: SystemInput, keys: Array) -> list[int]:
    del variables, keys
    return [0] * int(np.shape(inputs.valid)[0])


def _host_apply(
    variables: PolicyTree, memory: list[int], inputs: SystemInput, keys: Array
) -> SystemOutput:
    del variables, keys
    active = np.asarray(inputs.active_mask)
    still = np.zeros(active.shape, np.int32)
    ids = np.where(np.asarray(memory)[:, None] >= SWITCH, 1, 0)
    ids = np.where(active, ids, -1).astype(np.int32)
    valid = np.asarray(inputs.valid)
    following = [count + int(alive) for count, alive in zip(memory, valid, strict=True)]
    # A host System returns NumPy arrays where JAX ones are declared.
    return SystemOutput(
        ActorAction(*cast(tuple[Array, ...], (still,) * 3)),
        following,
        policy_ids=cast(Array, ids),
    )


def _act(
    variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, PolicyTree]:
    del variables, keys
    still = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    return ActorAction(still, still, still), memory


def _still(name: str) -> System:
    # A tiny System that stands still; small programs keep compiling cheap.
    return System(name, _act)


def _switcher(execution: str = "jax") -> System:
    host = execution == "host"
    return System(
        "Switcher",
        _host_apply if host else _apply,
        init=_host_init if host else _init,
        execution="host" if host else "jax",
        components=({"name": "early"}, {"name": "late"}),
    )


def _specs() -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    for episode, steps in enumerate(LENGTHS, start=1):
        config = make_standard_team_deathmatch_config(
            map_id=12,
            team_a_roster=("priest", "mage", "rogue"),
            team_b_roster=("priest", "warrior"),
            max_steps=steps,
        )
        if episode == 4:
            start, *_ = core.reset(config, jax.random.key(0))
            start = start._replace(step_count=jnp.int32(3))
            specs.append(EpisodeSpec(episode, config, initial_state=start))
        else:
            specs.append(EpisodeSpec(episode, config))
    return specs


def _rows(
    path: Path, replays: Sequence[int], execution: str = "jax"
) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    result = marl_bgs.evaluate_episodes(
        marl_bgs.pool({_still("a"): 1, _still("b"): 1}),
        _switcher(execution),
        _specs(),
        num_envs=3,
        metrics="none",
        output_dir=path,
        replay_episodes=replays,
        chunk_size=4,
    )
    assert result.paths is not None
    table = load_results(result.paths["run_details"].parent).table("policy_assignments")
    rows: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for episode, slot, choice, start, stop in zip(
        table["episode_id"],
        table["global_slot"],
        table["policy_id"],
        table["decision_start"],
        table["decision_stop"],
        strict=True,
    ):
        rows.setdefault((int(episode), int(slot)), []).append(
            (int(start), int(stop), int(choice))
        )
    return {key: sorted(value) for key, value in rows.items()}


def _summary(intervals: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    # What one summary row per slot keeps of a per-decision history.
    _, stop, first = intervals[0]
    length = intervals[-1][1]
    if stop == length:
        return [(0, length, first)]
    return [(0, stop, first), (stop, length, -1)]


type Rows = dict[tuple[int, int], list[tuple[int, int, int]]]
type Flushes = list[tuple[set[int], set[int]]]


@pytest.fixture(scope="module")
def some(tmp_path_factory: pytest.TempPathFactory) -> tuple[Rows, Flushes]:
    # Even games replayed; a one-game buffer makes every completion flush, and
    # each flush notes its pending games and its buffered member-row games.
    flushes: Flushes = []
    setup = RunWriter.__init__
    flush = RunWriter.flush

    def small(self: RunWriter, *args: object, **kwargs: object) -> None:
        setup(
            self, *cast(tuple[Any, ...], args), **cast(Any, kwargs | {"buffer_size": 1})
        )

    def watched(self: RunWriter) -> None:
        pending = {episode for _, episode in self._pending}
        rows = {int(cast(int, row[3])) for row in self._rows["policy_assignments.csv"]}
        flushes.append((pending, rows))
        flush(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RunWriter, "__init__", small)
        patch.setattr(RunWriter, "flush", watched)
        rows = _rows(tmp_path_factory.mktemp("some"), REPLAYED)
    return rows, flushes


def test_summary_rows_match_replayed_decisions(
    tmp_path: Path, some: tuple[Rows, Flushes]
) -> None:
    every = _rows(tmp_path / "every", replays=range(1, len(LENGTHS) + 1))
    summarized, _ = some
    lengths = {episode: steps for episode, steps in enumerate(LENGTHS, start=1)}
    lengths[4] -= 3  # The authored start began at tick 3.
    # Team A is the pool (slots 0-2 active); Team B the switcher (slots 5-6).
    active = (0, 1, 2, 5, 6)
    for episode, length in lengths.items():
        for slot in active:
            traced = every[(episode, slot)]
            assert traced[0][0] == 0 and traced[-1][1] == length
            if episode % 2 == 0 or slot < 5:
                # Replayed games and the pool's unchanging side: same rows.
                assert summarized[(episode, slot)] == traced
            else:
                assert summarized[(episode, slot)] == _summary(traced)
        for slot in set(range(10)) - set(active):
            assert (episode, slot) not in summarized
    # The pool's draws differ between games, so a mixed-up game would show.
    assert {summarized[(episode, 0)][0][2] for episode in lengths} == {0, 1}
    # The pool's member side never changes; the switcher's changes at decision 3.
    assert summarized[(1, 5)] == [(0, 3, 0), (3, 5, -1)]
    assert summarized[(7, 6)] == [(0, 1, 0)]
    assert every[(2, 5)] == [(0, 3, 0), (3, 11, 1)]


def test_member_rows_flush_with_their_games(some: tuple[Rows, Flushes]) -> None:
    _, flushes = some
    assert len(flushes) >= len(LENGTHS)
    for pending, rows in flushes:
        summarized = {episode for episode in rows if episode % 2}
        assert summarized == {episode for episode in pending if episode % 2}


def test_host_member_records_match_the_jax_twin(
    tmp_path: Path, some: tuple[Rows, Flushes]
) -> None:
    assert _rows(tmp_path / "host", REPLAYED, execution="host") == some[0]


def test_member_free_carry_keeps_todays_leaves() -> None:
    completed = evaluation._Completed(*(jnp.zeros(3) for _ in range(10)))
    parts = (jnp.zeros(2), jnp.zeros(2), jnp.zeros(3), completed)
    carry = evaluation._SystemCarry(*cast(tuple[Any, ...], parts), None)
    assert completed.members is None
    assert jax.tree.leaves(carry) == jax.tree.leaves(parts)
