"""Check the Capture the Flag flag life cycle, captures and outcome in Core.

Every start is hand-built and goes through initialize_scenario_state, so Core's
own validators accept it, and every tick goes through core.step. Map 0 has no
obstacles; Team A's flag home is (0.5, 5) and Team B's is (19.5, 5).

Contracts: a spawn-shielded agent never picks up or returns a flag, even on
its last shielded tick (counter 1), and acts on the next tick. On a dropped
flag the nearest eligible toucher decides: a nearer owner returns it, a nearer
enemy picks it up, and an exact owner-enemy tie goes to the pickup; a flag sent
home stays home that tick even with an enemy at its home. One owner is
credited per return: the nearer of two, the lower slot on an exact tie, and
nobody when an enemy wins the flag. A dying carrier's flag stays where it fell
on the drop tick, with a teammate and a defender touching it, and the teammate
picks it up on the next tick. An agent killed this tick still returns a flag,
or picks one up and drops it at its body with both facts true, no carrier and
a fresh countdown. A carrier killed during its own Warrior Charge drops the
flag where the Charge left its body. The dropped countdown ages once a tick and
sends the flag home, with the timer fact, after 1, 2 or 5 ticks on the ground.
A same-tick respawn wave does not move a dropped flag from its carrier's body.

Captures: a carrier scores when it ends the tick within the capture distance
of its own flag's home while its own flag is home. The captured flag goes home
and the team's count rises by 1. A carrier killed this tick still captures if
its last move reaches the home, and it drops nothing. A return and a capture
on one tick both count, and an enemy flag dropped at the doorstep can be picked
up and captured on one tick. There is no capture while your own flag is
carried, lies dropped, is picked up at its home on the same tick, or is sent
home by its timer that tick: the timer runs after captures, so the capture
comes on the next tick. The capture that reaches the threshold wins, and so
terminates, before the time limit or on the last tick, which also truncates;
the winners get +1 and the losers -1. The time limit draws even with a capture
lead, with zero rewards. The Team Deathmatch outcome stays 0 throughout.
Every capture test runs once for each team: that team's slots and positions
are mirrored (slot s becomes s + 5 and x becomes 20 - x), and positions agree
within 1e-5.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from tests.helpers.capture_the_flag_fixtures import capture_the_flag_config

from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.axis_mappings import global_slot_to_target_action
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MOVE_EAST,
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
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

_ROSTER = ("mage", "warrior", "priest")
_CONFIG = capture_the_flag_config(
    make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER
    )
)
# Slots 0-2 are Team A's Mage, Warrior and Priest; 5-7 are Team B's.
_FAR = {
    0: (3.0, 1.0),
    1: (3.0, 3.0),
    2: (3.0, 9.0),
    5: (17.0, 1.0),
    6: (17.0, 3.0),
    7: (17.0, 9.0),
}
_TEAM_A_HOME = [0.5, 5.0]
_MAP_WIDTH = 20.0
_SIDES = pytest.mark.parametrize("us", [0, 1], ids=["team-a", "team-b"])
# jax.jit hides core.step's result types; the cast names the ones used here.
_STEP = cast(
    Callable[..., tuple[EnvState, object, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(core.step),
)

Arrange = Callable[[EnvState], EnvState]
# Extra positions, an arrangement of your own flag, and who picks it up.
OwnFlag = tuple[dict[int, tuple[float, float]], Arrange, list[int]]


def _start(
    positions: dict[int, tuple[float, float]],
    arrange: Arrange = lambda state: state,
    config: EnvConfig = _CONFIG,
) -> tuple[EnvState, ActionMask]:
    state, _, _, _ = core.reset(config, jax.random.key(0))
    where = state.agent_positions
    for slot, point in {**_FAR, **positions}.items():
        where = where.at[slot].set(jnp.asarray(point, jnp.float32))
    state = state._replace(
        agent_positions=where,
        spawn_shield_durations=jnp.zeros(MAX_AGENT_SLOTS, jnp.int32),
    )
    state, _, mask, _ = core.initialize_scenario_state(arrange(state), config)
    return state, mask


def _act(
    attacks: dict[int, int] | None = None,
    ultimates: tuple[int, ...] = (),
    moves: dict[int, int] | None = None,
) -> Action:
    target = jnp.zeros((MAX_AGENT_SLOTS,), jnp.int32)
    for source, recipient in (attacks or {}).items():
        target = target.at[source].set(global_slot_to_target_action(source, recipient))
    move = jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, jnp.int32)
    for slot, direction in (moves or {}).items():
        move = move.at[slot].set(direction)
    ultimate = (
        jnp.zeros((MAX_AGENT_SLOTS,), jnp.int32)
        .at[jnp.asarray(ultimates, jnp.int32)]
        .set(1)
    )
    return Action(move=move, select_target=target, use_ultimate=ultimate)


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


def _slots(mask: Array) -> list[int]:
    return [slot for slot, value in enumerate(mask.tolist()) if value]


def _events(info: Info) -> dict[str, list[int]]:
    facts = info.transition_facts.capture_the_flag_facts
    return {
        "picked up": _slots(facts.flag_was_picked_up_by_agent),
        "dropped": _slots(facts.flag_was_dropped_by_agent),
        "returned": _slots(facts.ally_flag_was_returned_by_agent),
        "captured": _slots(facts.enemy_flag_was_captured_by_agent),
        "timer": _slots(facts.flag_was_automatically_returned_by_team),
    }


def _none(**events: list[int]) -> dict[str, list[int]]:
    return {
        "picked up": [],
        "dropped": [],
        "returned": [],
        "captured": [],
        "timer": [],
        **events,
    }


def _flag(state: EnvState, flag: int) -> tuple[list[float], int, int]:
    return (
        state.capture_the_flag_flag_positions[flag].tolist(),
        int(state.capture_the_flag_flag_carrier_slots[flag]),
        int(state.capture_the_flag_dropped_flag_countdowns[flag]),
    )


def _dropped_at(
    point: tuple[float, float], countdown: int = 10, flag: int = 0
) -> Arrange:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            capture_the_flag_flag_positions=state.capture_the_flag_flag_positions.at[
                flag
            ].set(jnp.asarray(point, jnp.float32)),
            capture_the_flag_dropped_flag_countdowns=(
                state.capture_the_flag_dropped_flag_countdowns.at[flag].set(countdown)
            ),
        )

    return arrange


def _carried_by(slot: int, point: tuple[float, float], flag: int = 0) -> Arrange:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            current_health=state.current_health.at[slot].set(1),
            capture_the_flag_flag_positions=state.capture_the_flag_flag_positions.at[
                flag
            ].set(jnp.asarray(point, jnp.float32)),
            capture_the_flag_flag_carrier_slots=(
                state.capture_the_flag_flag_carrier_slots.at[flag].set(slot)
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


def _all(*arranges: Arrange) -> Arrange:
    def arrange(state: EnvState) -> EnvState:
        for step in arranges:
            state = step(state)
        return state

    return arrange


def _scored(us: int, ours: int, step_count: int) -> Arrange:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            step_count=jnp.asarray(step_count, state.step_count.dtype),
            capture_the_flag_scores=jnp.asarray(_teams(us, ours, 0), jnp.int32),
        )

    return arrange


# Team-relative seats: team `us` (0 for Team A, 1 for Team B) is written as
# Team A on the left; Team B's copy is the mirror image.
def _slot(us: int, slot: int) -> int:
    return slot + 5 * us


def _point(us: int, x: float, y: float) -> tuple[float, float]:
    return (x if us == 0 else _MAP_WIDTH - x, y)


def _teams[T](us: int, ours: T, theirs: T) -> list[T]:
    return [ours, theirs] if us == 0 else [theirs, ours]


def _at(us: int, x: float, y: float) -> object:
    return pytest.approx(list(_point(us, x, y)), abs=1e-5)


@pytest.mark.parametrize(
    ("positions", "arrange", "second_tick"),
    [
        ({5: (0.9, 5.0)}, _shielded(5), _none(**{"picked up": [5]})),
        ({1: (8.45, 5.0)}, _shielded(1, _dropped_at((8.0, 5.0))), _none(returned=[1])),
    ],
    ids=["pickup-at-home", "return-of-a-dropped-flag"],
)
def test_a_shielded_agent_never_touches_a_flag(
    positions: dict[int, tuple[float, float]],
    arrange: Arrange,
    second_tick: dict[str, list[int]],
) -> None:
    start = _start(positions, arrange)
    (first, first_info), (_, second_info) = _play(start, [_act(), _act()])
    assert _events(first_info) == _none()
    assert _flag(first, 0)[1] == -1
    assert _events(second_info) == second_tick


@pytest.mark.parametrize(
    ("owner", "enemy", "events", "flag"),
    [
        ((8.45, 5.0), (7.4, 5.0), _none(returned=[1]), (_TEAM_A_HOME, -1, 0)),
        ((8.6, 5.0), (7.55, 5.0), _none(**{"picked up": [6]}), ([7.55, 5.0], 6, 0)),
        # 9/16 offsets are exact in float32, so the tie is exact on any device.
        (
            (8.5625, 5.0),
            (7.4375, 5.0),
            _none(**{"picked up": [6]}),
            ([7.4375, 5.0], 6, 0),
        ),
    ],
    ids=["owner-nearer", "enemy-nearer", "exact-tie-goes-to-pickup"],
)
def test_the_nearest_eligible_toucher_wins_a_dropped_flag(
    owner: tuple[float, float],
    enemy: tuple[float, float],
    events: dict[str, list[int]],
    flag: tuple[list[float], int, int],
) -> None:
    ((state, info),) = _play(
        _start({1: owner, 6: enemy}, _dropped_at((8.0, 5.0))), [_act()]
    )
    assert _events(info) == events
    position, carrier, countdown = _flag(state, 0)
    assert (position, carrier, countdown) == (
        [float(jnp.float32(x)) for x in flag[0]],
        flag[1],
        flag[2],
    )


def test_a_flag_sent_home_stays_home_that_tick() -> None:
    start = _start({1: (8.45, 5.0), 5: (0.9, 5.0)}, _dropped_at((8.0, 5.0)))
    (first, first_info), (_, second_info) = _play(start, [_act(), _act()])
    assert _events(first_info) == _none(returned=[1])
    assert _flag(first, 0) == (_TEAM_A_HOME, -1, 0)
    assert _events(second_info) == _none(**{"picked up": [5]})


@pytest.mark.parametrize(
    ("positions", "events"),
    [
        ({0: (8.6, 5.0), 1: (7.5, 5.0)}, _none(returned=[1])),
        ({0: (8.5625, 5.0), 1: (7.4375, 5.0)}, _none(returned=[0])),
        ({1: (8.6, 5.0), 6: (7.55, 5.0)}, _none(**{"picked up": [6]})),
    ],
    ids=["nearer-owner", "exact-tie-goes-to-lowest-slot", "enemy-wins-nobody-credited"],
)
def test_one_owner_is_credited_per_return(
    positions: dict[int, tuple[float, float]], events: dict[str, list[int]]
) -> None:
    ((_, info),) = _play(_start(positions, _dropped_at((8.0, 5.0))), [_act()])
    assert _events(info) == events
    returned = (
        info.transition_facts.capture_the_flag_facts.ally_flag_was_returned_by_agent
    )
    assert int(returned[:5].sum()) <= 1 and int(returned[5:].sum()) <= 1


def test_a_dying_carrier_s_flag_stays_where_it_fell_until_the_next_tick() -> None:
    # A longer reach lets a teammate and a defender touch the carrier's body.
    config = _CONFIG._replace(capture_the_flag_flag_interaction_distance=1.25)
    positions = {5: (8.0, 5.0), 0: (5.5, 5.0), 1: (8.0, 6.2), 6: (9.1, 5.0)}
    start = _start(positions, _carried_by(5, (8.0, 5.0)), config)
    (first, first_info), (second, second_info) = _play(
        start, [_act(attacks={0: 5}), _act()], config
    )
    assert _events(first_info) == _none(dropped=[5])
    assert _flag(first, 0) == ([8.0, 5.0], -1, 30)
    assert _events(second_info) == _none(**{"picked up": [6]})
    assert _flag(second, 0) == ([9.100000381469727, 5.0], 6, 0)


def test_an_agent_killed_this_tick_still_returns_a_flag() -> None:
    def weak_owner(state: EnvState) -> EnvState:
        return _dropped_at((8.0, 5.0))(state)._replace(
            current_health=state.current_health.at[1].set(1)
        )

    start = _start({1: (8.45, 5.0), 5: (11.0, 5.0)}, weak_owner)
    ((state, info),) = _play(start, [_act(attacks={5: 1})])
    assert _slots(info.transition_facts.death_facts.is_newly_dead_by_recipient) == [1]
    assert _events(info) == _none(returned=[1])
    assert _flag(state, 0) == (_TEAM_A_HOME, -1, 0)


def test_an_agent_killed_this_tick_picks_up_and_drops_at_its_body() -> None:
    def weak_enemy(state: EnvState) -> EnvState:
        return _dropped_at((8.0, 5.0))(state)._replace(
            current_health=state.current_health.at[6].set(1)
        )

    start = _start({6: (8.3, 5.0), 0: (9.35, 5.0)}, weak_enemy)
    ((state, info),) = _play(start, [_act(attacks={0: 6})])
    assert _slots(info.transition_facts.death_facts.is_newly_dead_by_recipient) == [6]
    assert _events(info) == _none(**{"picked up": [6], "dropped": [6]})
    assert _flag(state, 0) == (state.agent_positions[6].tolist(), -1, 30)


def test_a_carrier_killed_in_its_own_charge_drops_the_flag_at_the_charge_end() -> None:
    start = _start({6: (11.0, 6.0), 0: (8.5, 6.0)}, _carried_by(6, (11.0, 6.0)))
    ((state, info),) = _play(start, [_act(attacks={6: 0, 0: 6}, ultimates=(6,))])
    charge = info.transition_facts.physical_facts.charge_phase_displacement_by_agent
    assert charge[6].tolist() == [-1.5, 0.0]
    assert _slots(info.transition_facts.death_facts.is_newly_dead_by_recipient) == [6]
    assert _events(info) == _none(dropped=[6])
    assert _flag(state, 0) == ([9.5, 6.0], -1, 30)
    assert state.agent_positions[6].tolist() == [9.5, 6.0]


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_the_dropped_countdown_sends_the_flag_home_on_time(steps: int) -> None:
    config = _CONFIG._replace(capture_the_flag_dropped_flag_auto_return_steps=steps)
    start = _start({5: (8.0, 5.0), 0: (5.5, 5.0)}, _carried_by(5, (8.0, 5.0)), config)
    ticks = _play(start, [_act(attacks={0: 5})] + [_act()] * steps, config)
    countdowns = [_flag(state, 0)[2] for state, _ in ticks]
    assert countdowns == list(range(steps, -1, -1))
    assert [_events(info)["timer"] for _, info in ticks] == [[]] * steps + [[0]]
    assert [_flag(state, 0)[0] for state, _ in ticks] == [[8.0, 5.0]] * steps + [
        _TEAM_A_HOME
    ]


def test_a_same_tick_respawn_wave_leaves_the_flag_at_the_body() -> None:
    def wave_due(state: EnvState) -> EnvState:
        state = _carried_by(5, (8.0, 5.0))(state)
        return state._replace(
            alive_mask=state.alive_mask.at[7].set(False),
            current_health=state.current_health.at[7].set(0),
            team_respawn_wave_countdowns=state.team_respawn_wave_countdowns.at[1].set(
                0
            ),
        )

    start = _start({5: (8.0, 5.0), 0: (5.5, 5.0)}, wave_due)
    ((state, info),) = _play(start, [_act(attacks={0: 5})])
    assert bool(state.alive_mask[7]) and not bool(state.alive_mask[5])
    assert _events(info) == _none(dropped=[5])
    assert _flag(state, 0) == ([8.0, 5.0], -1, 30)
    assert state.agent_positions[7].tolist() != [8.0, 5.0]


@_SIDES
def test_a_carrier_killed_this_tick_still_captures_with_its_last_move(us: int) -> None:
    carrier, mage = _slot(us, 1), _slot(1 - us, 0)
    # The carrier starts 2.0 from home, outside the 1.25 capture distance, and
    # its one-unit move ends 1.0 from home.
    start = _start(
        {carrier: _point(us, 2.5, 5.0), mage: _point(us, 3.75, 5.0)},
        _carried_by(carrier, _point(us, 2.5, 5.0), flag=1 - us),
    )
    homeward = MOVE_WEST if us == 0 else MOVE_EAST
    ((state, info),) = _play(
        start, [_act(attacks={mage: carrier}, moves={carrier: homeward})]
    )
    assert state.agent_positions[carrier].tolist() == _at(us, 1.5, 5.0)
    assert _slots(info.transition_facts.death_facts.is_newly_dead_by_recipient) == [
        carrier
    ]
    assert _events(info) == _none(captured=[carrier])
    assert state.capture_the_flag_scores.tolist() == _teams(us, 1, 0)
    assert _flag(state, 1 - us) == (_at(us, 19.5, 5.0), -1, 0)


@_SIDES
def test_a_return_and_a_capture_on_one_tick_both_count(us: int) -> None:
    carrier, priest = _slot(us, 1), _slot(us, 2)
    start = _start(
        {carrier: _point(us, 1.0, 5.0), priest: _point(us, 3.4, 5.0)},
        _all(
            _carried_by(carrier, _point(us, 1.0, 5.0), flag=1 - us),
            _dropped_at(_point(us, 3.0, 5.0), flag=us),
        ),
    )
    ((state, info),) = _play(start, [_act()])
    assert _events(info) == _none(returned=[priest], captured=[carrier])
    assert state.capture_the_flag_scores.tolist() == _teams(us, 1, 0)
    assert _flag(state, us) == (_at(us, 0.5, 5.0), -1, 0)
    assert _flag(state, 1 - us) == (_at(us, 19.5, 5.0), -1, 0)
    assert int(info.transition_facts.capture_the_flag_facts.outcome) == (
        TASK_MODE_OUTCOME_ONGOING
    )


def _own_flag_carried_by_an_enemy(us: int) -> OwnFlag:
    enemy = _slot(1 - us, 0)
    return (
        {enemy: _point(us, 10.0, 5.0)},
        _carried_by(enemy, _point(us, 10.0, 5.0), flag=us),
        [],
    )


def _own_flag_dropped(us: int) -> OwnFlag:
    return {}, _dropped_at(_point(us, 10.0, 5.0), flag=us), []


def _own_flag_picked_up_at_home(us: int) -> OwnFlag:
    enemy = _slot(1 - us, 0)
    return {enemy: _point(us, 0.5, 4.4)}, lambda state: state, [enemy]


@_SIDES
@pytest.mark.parametrize(
    "own_flag",
    [_own_flag_carried_by_an_enemy, _own_flag_dropped, _own_flag_picked_up_at_home],
    ids=["carried-by-an-enemy", "dropped", "picked-up-at-home-this-tick"],
)
def test_no_capture_while_your_own_flag_is_away(
    us: int,
    own_flag: Callable[[int], OwnFlag],
) -> None:
    carrier = _slot(us, 1)
    positions, arrange, picked_up = own_flag(us)
    # 1.1 from home, inside the capture distance.
    start = _start(
        {carrier: _point(us, 0.5, 6.1), **positions},
        _all(_carried_by(carrier, _point(us, 0.5, 6.1), flag=1 - us), arrange),
    )
    ((state, info),) = _play(start, [_act()])
    assert _events(info) == _none(**{"picked up": picked_up})
    assert state.capture_the_flag_scores.tolist() == [0, 0]
    assert _flag(state, 1 - us) == (_at(us, 0.5, 6.1), carrier, 0)


@_SIDES
def test_a_flag_its_timer_sends_home_counts_from_the_next_tick(us: int) -> None:
    carrier = _slot(us, 1)
    start = _start(
        {carrier: _point(us, 1.0, 5.0)},
        _all(
            _carried_by(carrier, _point(us, 1.0, 5.0), flag=1 - us),
            _dropped_at(_point(us, 10.0, 5.0), countdown=1, flag=us),
        ),
    )
    (first, first_info), (second, second_info) = _play(start, [_act(), _act()])
    assert _events(first_info) == _none(timer=[us])
    assert _flag(first, us) == (_at(us, 0.5, 5.0), -1, 0)
    assert first.capture_the_flag_scores.tolist() == [0, 0]
    assert _events(second_info) == _none(captured=[carrier])
    assert second.capture_the_flag_scores.tolist() == _teams(us, 1, 0)


@_SIDES
def test_an_enemy_flag_dropped_at_your_doorstep_is_taken_and_captured_on_one_tick(
    us: int,
) -> None:
    # The Mage is 0.6 from the flag (reach 0.75) and about 1.17 from home (1.25).
    mage = _slot(us, 0)
    start = _start(
        {mage: _point(us, 1.5, 5.6)}, _dropped_at(_point(us, 1.5, 5.0), flag=1 - us)
    )
    ((state, info),) = _play(start, [_act()])
    assert _events(info) == _none(**{"picked up": [mage], "captured": [mage]})
    assert state.capture_the_flag_scores.tolist() == _teams(us, 1, 0)
    assert _flag(state, 1 - us) == (_at(us, 19.5, 5.0), -1, 0)


@_SIDES
@pytest.mark.parametrize(
    "last_tick", [False, True], ids=["before-the-time-limit", "on-the-last-tick"]
)
def test_the_capture_that_reaches_the_threshold_wins(us: int, last_tick: bool) -> None:
    carrier = _slot(us, 1)
    step_count = _CONFIG.max_steps - 1 if last_tick else 5
    state, mask = _start(
        {carrier: _point(us, 1.0, 5.0)},
        _all(
            _carried_by(carrier, _point(us, 1.0, 5.0), flag=1 - us),
            _scored(us, _CONFIG.capture_the_flag_score_threshold - 1, step_count),
        ),
    )
    next_state, _, reward, done, _, info = _STEP(
        _CONFIG, state, mask, _act(), jax.random.key(1)
    )
    facts = info.transition_facts
    assert next_state.capture_the_flag_scores.tolist() == _teams(us, 3, 0)
    assert int(facts.capture_the_flag_facts.outcome) == (
        TASK_MODE_OUTCOME_TEAM_A_WIN if us == 0 else TASK_MODE_OUTCOME_TEAM_B_WIN
    )
    assert int(facts.team_deathmatch_facts.outcome) == TASK_MODE_OUTCOME_ONGOING
    assert (bool(done.terminated), bool(done.truncated)) == (True, last_tick)
    # Three configured slots per team, then two unused ones.
    expected = [r for team in _teams(us, 1.0, -1.0) for r in (team,) * 3 + (0.0,) * 2]
    assert reward.rewards.tolist() == expected


@_SIDES
def test_the_time_limit_draws_even_with_a_capture_lead(us: int) -> None:
    state, mask = _start({}, _scored(us, 2, _CONFIG.max_steps - 1))
    _, _, reward, done, _, info = _STEP(_CONFIG, state, mask, _act(), jax.random.key(1))
    facts = info.transition_facts
    assert int(facts.capture_the_flag_facts.outcome) == TASK_MODE_OUTCOME_DRAW
    assert int(facts.team_deathmatch_facts.outcome) == TASK_MODE_OUTCOME_ONGOING
    assert (bool(done.terminated), bool(done.truncated)) == (False, True)
    assert reward.rewards.tolist() == [0.0] * MAX_AGENT_SLOTS
