"""Check the King of the Hills rules in Core.

The exported on-hill test (build_agent_hill_interaction_matrix) is geometry
only: it returns agent by hill (10, 3), True for a center exactly on a used
hill's edge, and never True for an agent standing exactly on an unused slot's
position.

Every other start is hand-built and goes through initialize_scenario_state, so
Core's own validators accept it, and every tick goes through core.step. Map 0
has no obstacles; the hills stand at (5, 5), (10, 5) and (15, 5) with radius
1.25. Every test but the random walk runs once for each team: that team's
slots and positions are mirrored (slot s becomes s + 5, x becomes 20 - x, and
hill slot h becomes 2 - h). The hill rules read no spawn pads, so these tests
keep map 0's spawn ends; the observation tests exchange them.

Presence: an agent whose center is exactly the radius from a hill's center is
on it, and one 1.26 from a hill's center is not; a spawn-shielded agent never
takes (while its unshielded teammate takes another hill on the same tick),
contests or refreshes a hill, and its presence fact is False; an agent this
tick's combat kills still counts on its hill that tick, with presence True.
Ownership at the default timer 0: a team alone takes a neutral hill at once and
scores that tick, and the tick its last holder steps off turns the hill neutral
with no points. Above timer 0 (2 ticks): an empty owned hill keeps scoring while
its ticks left run down 2, 1, 0, then turns neutral; an enemy alone takes an
owned empty hill at once whatever its ticks left, and its ticks left becomes the
timer. Contests: both teams on a hill
keep its owner, set an owned hill's ticks left to the timer, leave a neutral
hill's at 0, and score for nobody. Points: both teams score on one tick from
different hills; one team alone on all three hills scores the rate on each plus
the sweep bonus (rate 2, bonus 5: 11), and a contest on one of its three owned
hills ends the sweep (4); one-hill and two-hill maps never sweep, and their
unused hill slots are never owned or contested. Over 60 ticks of random moves,
under a threshold no team reaches so the game never ends, each hill's owner and
ticks left follow the ownership rules from the presence fact and the previous
tick (a team alone takes the hill and resets the timer; both teams keep the
owner, resetting an owned hill's timer; an empty hill loses one tick, or turns
neutral at 0), every tick's points fact equals the rate times each team's
uncontested owned hills plus the bonus while sweeping, the scores rise by
exactly those points, the owners fact equals the successor state's owners,
ticks left is never negative and is 0 on every neutral hill, and the walk
passes through contested hills and owned empty hills. Outcomes: the points that
reach the threshold win, before the last tick (terminated) and on it (terminated and
truncated), with +1 for the winners' configured slots and -1 for the losers';
when both teams pass the threshold on one tick the higher score wins and equal
scores draw; the time limit draws even with a lead, with zero rewards. The Team
Deathmatch and Capture the Flag outcomes stay 0 throughout.
"""

from collections.abc import Callable
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.axis_mappings import global_slot_to_target_action
from marl_battlegrounds.core.objectives import build_agent_hill_interaction_matrix
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_STAY,
    MOVE_WEST,
    TASK_MODE_OUTCOME_DRAW,
    TASK_MODE_OUTCOME_ONGOING,
    TASK_MODE_OUTCOME_TEAM_A_WIN,
    TASK_MODE_OUTCOME_TEAM_B_WIN,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Reward,
)
from marl_battlegrounds.tasks import make_standard_king_of_the_hills_config

_ROSTER = ("mage", "warrior", "priest")
# Slots 0-2 are Team A's Mage, Warrior and Priest; 5-7 are Team B's. Every far
# spot is more than 1.25 from every hill.
_FAR = {
    0: (2.0, 1.0),
    1: (2.0, 3.0),
    2: (2.0, 9.0),
    5: (18.0, 1.0),
    6: (18.0, 3.0),
    7: (18.0, 9.0),
}
_MAP_WIDTH = 20.0
_SIDES = pytest.mark.parametrize("us", [0, 1], ids=["team-a", "team-b"])
_STEP = cast(
    Callable[..., tuple[EnvState, object, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(core.step),
)

Arrange = Callable[[EnvState], EnvState]


_config = partial(
    make_standard_king_of_the_hills_config,
    map_id=0,
    team_a_roster=_ROSTER,
    team_b_roster=_ROSTER,
)
_CONFIG = _config()


def _start(
    positions: dict[int, tuple[float, float]],
    arrange: Arrange = lambda state: state,
    config: EnvConfig = _CONFIG,
) -> tuple[EnvState, ActionMask]:
    state, _, _, _ = core.reset(config, jax.random.key(0))
    where = state.agent_positions
    for slot, point in {**_FAR, **positions}.items():
        where = where.at[slot].set(jnp.asarray(point, jnp.float32))
    state = state._replace(agent_positions=where)
    state, _, mask, _ = core.initialize_scenario_state(arrange(state), config)
    return state, mask


def _act(
    attacks: dict[int, int] | None = None, moves: dict[int, int] | None = None
) -> Action:
    target = jnp.zeros((MAX_AGENT_SLOTS,), jnp.int32)
    for source, recipient in (attacks or {}).items():
        target = target.at[source].set(global_slot_to_target_action(source, recipient))
    move = jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, jnp.int32)
    for slot, direction in (moves or {}).items():
        move = move.at[slot].set(direction)
    return Action(
        move=move,
        select_target=target,
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), jnp.int32),
    )


def _play(
    start: tuple[EnvState, ActionMask],
    actions: list[Action],
    config: EnvConfig = _CONFIG,
) -> list[tuple[EnvState, Info]]:
    state, mask = start
    ticks: list[tuple[EnvState, Info]] = []
    for tick, action in enumerate(actions, 1):
        state, _, _, _, mask, info = _STEP(
            config, state, mask, action, jax.random.key(tick)
        )
        ticks.append((state, info))
    return ticks


# Team-relative seats: team `us` (0 for Team A, 1 for Team B) is written as
# Team A on the left; Team B's copy is the mirror image.
def _slot(us: int, slot: int) -> int:
    return slot + 5 * us


def _enemy(us: int, slot: int) -> int:
    return slot + 5 * (1 - us)


def _point(us: int, x: float, y: float) -> tuple[float, float]:
    return (x if us == 0 else _MAP_WIDTH - x, y)


def _away(us: int) -> int:
    return MOVE_WEST if us == 0 else MOVE_EAST


def _teams[T](us: int, ours: T, theirs: T) -> list[T]:
    return [ours, theirs] if us == 0 else [theirs, ours]


def _by_hill[T](us: int, values: list[T]) -> list[T]:
    return values if us == 0 else values[::-1]


# Owners by hill, team-relative: "us", "them" or "" for nobody.
def _owners(us: int, *names: str) -> list[int]:
    code = {"": 0, "us": us + 1, "them": 2 - us}
    return _by_hill(us, [code[name] for name in names])


def _owned(us: int, *names: str, ticks_left: tuple[int, int, int]) -> Arrange:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            king_of_the_hills_hill_owners=jnp.asarray(_owners(us, *names), jnp.int32),
            king_of_the_hills_empty_hill_ownership_countdowns=jnp.asarray(
                _by_hill(us, list(ticks_left)), jnp.int32
            ),
        )

    return arrange


def _shielded(slot: int, then: Arrange = lambda state: state) -> Arrange:
    def arrange(state: EnvState) -> EnvState:
        state = then(state)
        return state._replace(
            spawn_shield_durations=state.spawn_shield_durations.at[slot].set(1)
        )

    return arrange


def _scored(us: int, ours: int, theirs: int, step_count: int) -> Arrange:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            step_count=jnp.asarray(step_count, state.step_count.dtype),
            king_of_the_hills_scores=jnp.asarray(_teams(us, ours, theirs), jnp.int32),
        )

    return arrange


def _hills(state: EnvState) -> tuple[list[int], list[bool], list[int], list[int]]:
    return (
        state.king_of_the_hills_hill_owners.tolist(),
        state.king_of_the_hills_hill_is_contested.tolist(),
        state.king_of_the_hills_empty_hill_ownership_countdowns.tolist(),
        state.king_of_the_hills_scores.tolist(),
    )


def _points(info: Info) -> list[int]:
    return info.transition_facts.king_of_the_hills_facts.points_scored_by_team.tolist()


def _on(info: Info) -> Array:
    return info.transition_facts.king_of_the_hills_facts.is_on_hill_by_agent_by_hill


def test_the_on_hill_test_is_geometry_only_and_skips_unused_slots() -> None:
    hills = jnp.asarray([[0.0, 0.0], [10.0, 5.0], [0.0, 0.0]], jnp.float32)
    radii = jnp.asarray([0.0, 1.25, 0.0], jnp.float32)
    # Slot 1 touches hill 1's edge and slot 2 is 1.26 from its center; every
    # other agent stands exactly on an unused slot's position, (0, 0).
    agents = (
        jnp.zeros((MAX_AGENT_SLOTS, 2), jnp.float32)
        .at[1]
        .set(jnp.asarray([8.75, 5.0]))
        .at[2]
        .set(jnp.asarray([10.0, 6.26]))
    )
    on = build_agent_hill_interaction_matrix(agents, hills, radii)
    assert on.shape == (MAX_AGENT_SLOTS, 3)
    assert on.tolist() == [[False, slot == 1, False] for slot in range(MAX_AGENT_SLOTS)]


@_SIDES
def test_tangency_counts_and_a_hair_farther_does_not(us: int) -> None:
    edge, beyond = _slot(us, 0), _slot(us, 1)
    # 1.26 from hill 0's center, and on no other hill.
    start = _start({edge: _point(us, 8.75, 5.0), beyond: _point(us, 6.26, 5.0)})
    ((state, info),) = _play(start, [_act()])
    hill = 1
    assert bool(_on(info)[edge, hill])
    assert not bool(_on(info)[beyond].any())
    assert _hills(state) == (
        _owners(us, "", "us", ""),
        [False] * 3,
        [0, 0, 0],
        _teams(us, 1, 0),
    )
    assert _points(info) == _teams(us, 1, 0)


@_SIDES
def test_a_shielded_agent_never_takes_a_neutral_hill(us: int) -> None:
    ours, teammate = _slot(us, 0), _slot(us, 1)
    start = _start(
        {ours: _point(us, 10.0, 5.0), teammate: _point(us, 5.0, 5.0)}, _shielded(ours)
    )
    ((state, info),) = _play(start, [_act()])
    assert not bool(_on(info)[ours].any())
    assert _hills(state) == (
        _owners(us, "us", "", ""),
        [False] * 3,
        [0, 0, 0],
        _teams(us, 1, 0),
    )
    assert _points(info) == _teams(us, 1, 0)


@_SIDES
def test_a_shielded_owner_does_not_refresh_its_hill(us: int) -> None:
    config = _config(empty_hill_ownership_steps=2)
    ours = _slot(us, 0)
    start = _start(
        {ours: _point(us, 10.0, 5.0)},
        _shielded(ours, _owned(us, "", "us", "", ticks_left=(0, 1, 0))),
        config,
    )
    ((state, info),) = _play(start, [_act()], config)
    assert _hills(state)[:3] == (_owners(us, "", "us", ""), [False] * 3, [0, 0, 0])
    assert _points(info) == _teams(us, 1, 0)


@_SIDES
def test_a_shielded_enemy_does_not_contest(us: int) -> None:
    ours, theirs = _slot(us, 0), _enemy(us, 0)
    start = _start(
        {ours: _point(us, 9.4, 5.0), theirs: _point(us, 10.6, 5.0)},
        _shielded(theirs),
    )
    ((state, info),) = _play(start, [_act()])
    assert not bool(_on(info)[theirs].any())
    assert _hills(state)[:2] == (_owners(us, "", "us", ""), [False] * 3)
    assert _points(info) == _teams(us, 1, 0)


@_SIDES
def test_an_agent_killed_this_tick_still_counts_on_its_hill(us: int) -> None:
    warrior, mage = _slot(us, 1), _enemy(us, 0)

    def weak(state: EnvState) -> EnvState:
        return state._replace(current_health=state.current_health.at[warrior].set(1))

    start = _start({warrior: _point(us, 10.0, 5.0), mage: _point(us, 12.5, 5.0)}, weak)
    ((state, info),) = _play(start, [_act(attacks={mage: warrior})])
    dead = info.transition_facts.death_facts.is_newly_dead_by_recipient
    assert [slot for slot, value in enumerate(dead.tolist()) if value] == [warrior]
    assert bool(_on(info)[warrior, 1])
    assert _hills(state)[0] == _owners(us, "", "us", "")
    assert _points(info) == _teams(us, 1, 0)


@_SIDES
def test_at_timer_0_the_tick_the_holder_leaves_the_hill_turns_neutral(us: int) -> None:
    ours = _slot(us, 0)
    start = _start({ours: _point(us, 8.8, 5.0)})
    (first, first_info), (second, second_info) = _play(
        start, [_act(), _act(moves={ours: _away(us)})]
    )
    assert _hills(first) == (
        _owners(us, "", "us", ""),
        [False] * 3,
        [0, 0, 0],
        _teams(us, 1, 0),
    )
    assert not bool(_on(second_info)[ours].any())
    assert _hills(second) == ([0, 0, 0], [False] * 3, [0, 0, 0], _teams(us, 1, 0))
    assert _points(first_info) == _teams(us, 1, 0)
    assert _points(second_info) == [0, 0]


@_SIDES
def test_above_timer_0_an_empty_owned_hill_scores_until_its_ticks_run_out(
    us: int,
) -> None:
    config = _config(empty_hill_ownership_steps=2)
    ours = _slot(us, 0)
    start = _start({ours: _point(us, 8.8, 5.0)}, config=config)
    # One step away (1 map unit) leaves hill 1 and stays clear of hill 0.
    ticks = _play(start, [_act(), _act(moves={ours: _away(us)})] + [_act()] * 3, config)
    assert [_hills(state)[0] for state, _ in ticks] == [
        _owners(us, "", "us", "")
    ] * 3 + [[0, 0, 0]] * 2
    assert [_hills(state)[2] for state, _ in ticks] == [
        _by_hill(us, [0, ticks_left, 0]) for ticks_left in (2, 1, 0, 0, 0)
    ]
    assert [_points(info) for _, info in ticks] == [_teams(us, 1, 0)] * 3 + [[0, 0]] * 2


@_SIDES
def test_an_enemy_alone_takes_an_owned_empty_hill_at_once(us: int) -> None:
    config = _config(empty_hill_ownership_steps=2)
    ours = _slot(us, 0)
    start = _start(
        {ours: _point(us, 10.0, 5.0)},
        _owned(us, "", "them", "", ticks_left=(0, 1, 0)),
        config,
    )
    ((state, info),) = _play(start, [_act()], config)
    assert _hills(state)[:3] == (
        _owners(us, "", "us", ""),
        [False] * 3,
        _by_hill(us, [0, 2, 0]),
    )
    assert _points(info) == _teams(us, 1, 0)


@_SIDES
@pytest.mark.parametrize("owner", ["us", ""], ids=["owned", "neutral"])
def test_a_contested_hill_keeps_its_owner_and_scores_for_nobody(
    us: int, owner: str
) -> None:
    config = _config(empty_hill_ownership_steps=2)
    ours, theirs = _slot(us, 0), _enemy(us, 0)
    start = _start(
        {ours: _point(us, 9.4, 5.0), theirs: _point(us, 10.6, 5.0)},
        _owned(us, "", owner, "", ticks_left=(0, 1 if owner else 0, 0)),
        config,
    )
    ((state, info),) = _play(start, [_act()], config)
    assert _hills(state)[:3] == (
        _owners(us, "", owner, ""),
        [False, True, False],
        _by_hill(us, [0, 2 if owner else 0, 0]),
    )
    assert _points(info) == [0, 0]


@_SIDES
def test_both_teams_score_on_one_tick_from_different_hills(us: int) -> None:
    start = _start(
        {_slot(us, 0): _point(us, 5.0, 5.0), _enemy(us, 0): _point(us, 15.0, 5.0)}
    )
    ((state, info),) = _play(start, [_act()])
    assert _hills(state)[0] == _owners(us, "us", "", "them")
    assert _points(info) == [1, 1]


@_SIDES
def test_a_sweep_pays_the_bonus_and_a_contest_on_one_hill_ends_it(us: int) -> None:
    config = _config(hill_points_per_step=2, three_hill_sweep_bonus_points_per_step=5)
    holders = {
        _slot(us, h): _point(us, x, 5.0) for h, x in enumerate((5.0, 10.0, 15.0))
    }
    ((_, sweep),) = _play(_start(holders, config=config), [_act()], config)
    assert _points(sweep) == _teams(us, 11, 0)
    # All three already ours; an enemy contests the center one.
    contested = {**holders, _enemy(us, 0): _point(us, 10.6, 5.9)}
    owned = _owned(us, "us", "us", "us", ticks_left=(0, 0, 0))
    ((state, info),) = _play(_start(contested, owned, config), [_act()], config)
    assert _hills(state)[:2] == (_owners(us, "us", "us", "us"), [False, True, False])
    assert _points(info) == _teams(us, 4, 0)


@_SIDES
@pytest.mark.parametrize(
    "centers",
    [(None, (10.0, 5.0), None), ((5.0, 5.0), None, (15.0, 5.0))],
    ids=["one-hill", "two-hill"],
)
def test_one_and_two_hill_maps_never_sweep(
    us: int, centers: tuple[tuple[float, float] | None, ...]
) -> None:
    config = _config(
        hill_centers=centers,
        hill_points_per_step=2,
        three_hill_sweep_bonus_points_per_step=5,
    )
    used = [center for center in centers if center is not None]
    holders = {
        _slot(us, index): _point(us, *center) for index, center in enumerate(used)
    }
    ((state, info),) = _play(_start(holders, config=config), [_act()], config)
    owners = _owners(us, *("" if center is None else "us" for center in centers))
    assert _hills(state)[:2] == (owners, [False] * 3)
    assert _points(info) == _teams(us, 2 * len(used), 0)


def test_points_scores_and_owners_agree_on_every_random_tick() -> None:
    timer, rate, bonus = 2, 2, 5
    config = _config(
        empty_hill_ownership_steps=timer,
        hill_points_per_step=rate,
        three_hill_sweep_bonus_points_per_step=bonus,
        score_threshold=10_000,
    )
    near = {
        0: (4.5, 5.0),
        1: (9.4, 5.0),
        2: (14.0, 5.0),
        5: (15.5, 5.0),
        6: (10.6, 5.0),
        7: (6.0, 5.0),
    }
    state, mask = _start(near, config=config)
    moves = (MOVE_STAY, MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST)
    contested_seen = owned_empty_seen = 0
    for tick in range(60):
        key = jax.random.fold_in(jax.random.key(7), tick)
        picks = jax.random.randint(key, (MAX_AGENT_SLOTS,), 0, len(moves))
        action = _act(moves={slot: moves[int(picks[slot])] for slot in near})
        before = state.king_of_the_hills_scores.tolist()
        owners_before = state.king_of_the_hills_hill_owners.tolist()
        ticks_before = state.king_of_the_hills_empty_hill_ownership_countdowns.tolist()
        state, _, _, done, mask, info = _STEP(config, state, mask, action, key)
        assert not (bool(done.terminated) or bool(done.truncated)), tick
        facts = info.transition_facts.king_of_the_hills_facts
        owners = state.king_of_the_hills_hill_owners.tolist()
        contested = state.king_of_the_hills_hill_is_contested.tolist()
        ticks_left = state.king_of_the_hills_empty_hill_ownership_countdowns.tolist()
        presence = facts.is_on_hill_by_agent_by_hill
        on_by_team = presence.reshape(2, 5, 3).any(axis=1).tolist()
        for hill in range(3):
            a, b = bool(on_by_team[0][hill]), bool(on_by_team[1][hill])
            owner, left = owners_before[hill], ticks_before[hill]
            if a != b:  # one team alone takes it
                owner, left = (1 if a else 2), timer
            elif a:  # both teams on it: the owner stays
                left = timer if owner else 0
            elif left > 0:  # empty and still owned
                left -= 1
            else:  # empty at 0: neutral
                owner = 0
            assert (owners[hill], ticks_left[hill]) == (owner, left), (tick, hill)
        expected = [0, 0]
        for team in (1, 2):
            held = [o == team and not c for o, c in zip(owners, contested, strict=True)]
            expected[team - 1] = rate * sum(held) + (bonus if all(held) else 0)
        assert facts.points_scored_by_team.tolist() == expected, tick
        assert [b + p for b, p in zip(before, expected, strict=True)] == (
            state.king_of_the_hills_scores.tolist()
        ), tick
        assert facts.hill_owners.tolist() == owners, tick
        assert all(t >= 0 for t in ticks_left), tick
        assert all(t == 0 for o, t in zip(owners, ticks_left, strict=True) if o == 0)
        contested_seen += sum(contested)
        empty = ~np.asarray(facts.is_on_hill_by_agent_by_hill).any(axis=0)
        owned_empty_seen += int(np.sum((np.asarray(owners) != 0) & empty))
    # The walk must reach the cases the invariants are about.
    assert contested_seen > 0 and owned_empty_seen > 0


@_SIDES
@pytest.mark.parametrize(
    "last_tick", [False, True], ids=["before-the-limit", "on-the-last-tick"]
)
def test_the_points_that_reach_the_threshold_win(us: int, last_tick: bool) -> None:
    threshold = int(_CONFIG.king_of_the_hills_score_threshold)
    step_count = _CONFIG.max_steps - 1 if last_tick else 5
    ours = _slot(us, 0)
    state, mask = _start(
        {ours: _point(us, 10.0, 5.0)}, _scored(us, threshold - 1, 0, step_count)
    )
    next_state, _, reward, done, _, info = _STEP(
        _CONFIG, state, mask, _act(), jax.random.key(1)
    )
    facts = info.transition_facts
    assert next_state.king_of_the_hills_scores.tolist() == _teams(us, threshold, 0)
    assert int(facts.king_of_the_hills_facts.outcome) == (
        TASK_MODE_OUTCOME_TEAM_A_WIN if us == 0 else TASK_MODE_OUTCOME_TEAM_B_WIN
    )
    assert int(facts.team_deathmatch_facts.outcome) == TASK_MODE_OUTCOME_ONGOING
    assert int(facts.capture_the_flag_facts.outcome) == TASK_MODE_OUTCOME_ONGOING
    assert (bool(done.terminated), bool(done.truncated)) == (True, last_tick)
    expected = [r for team in _teams(us, 1.0, -1.0) for r in (team,) * 3 + (0.0,) * 2]
    assert reward.rewards.tolist() == expected


@_SIDES
@pytest.mark.parametrize("ours_hills", [2, 1], ids=["higher-wins", "equal-draws"])
def test_both_teams_over_the_threshold_on_one_tick(us: int, ours_hills: int) -> None:
    threshold = int(_CONFIG.king_of_the_hills_score_threshold)
    positions = {_enemy(us, 0): _point(us, 15.0, 5.0)}
    for index, x in enumerate((5.0, 10.0)[:ours_hills]):
        positions[_slot(us, index)] = _point(us, x, 5.0)
    state, mask = _start(positions, _scored(us, threshold - 1, threshold - 1, 5))
    _, _, reward, done, _, info = _STEP(_CONFIG, state, mask, _act(), jax.random.key(1))
    outcome = int(info.transition_facts.king_of_the_hills_facts.outcome)
    if ours_hills == 2:
        assert outcome == (
            TASK_MODE_OUTCOME_TEAM_A_WIN if us == 0 else TASK_MODE_OUTCOME_TEAM_B_WIN
        )
    else:
        assert outcome == TASK_MODE_OUTCOME_DRAW
        assert reward.rewards.tolist() == [0.0] * MAX_AGENT_SLOTS
    assert bool(done.terminated)


@_SIDES
def test_the_time_limit_draws_even_with_a_lead(us: int) -> None:
    state, mask = _start({}, _scored(us, 10, 0, _CONFIG.max_steps - 1))
    _, _, reward, done, _, info = _STEP(_CONFIG, state, mask, _act(), jax.random.key(1))
    facts = info.transition_facts
    assert int(facts.king_of_the_hills_facts.outcome) == TASK_MODE_OUTCOME_DRAW
    assert int(facts.team_deathmatch_facts.outcome) == TASK_MODE_OUTCOME_ONGOING
    assert int(facts.capture_the_flag_facts.outcome) == TASK_MODE_OUTCOME_ONGOING
    assert (bool(done.terminated), bool(done.truncated)) == (False, True)
    assert reward.rewards.tolist() == [0.0] * MAX_AGENT_SLOTS
