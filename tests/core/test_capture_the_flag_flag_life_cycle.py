"""Check the Capture the Flag flag life cycle in Core, from hand-built starts.

Every start goes through initialize_scenario_state, so Core's own validators
accept it, and every tick goes through core.step. Map 0 has no obstacles;
Team A's flag home is (0.5, 5) and Team B's is (19.5, 5).

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
    MOVE_STAY,
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
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
# jax.jit hides core.step's result types; the cast names the three used here.
_STEP = cast(
    Callable[..., tuple[EnvState, object, object, object, ActionMask, Info]],
    jax.jit(core.step),
)

Arrange = Callable[[EnvState], EnvState]


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
    attacks: dict[int, int] | None = None, ultimates: tuple[int, ...] = ()
) -> Action:
    target = jnp.zeros((MAX_AGENT_SLOTS,), jnp.int32)
    for source, recipient in (attacks or {}).items():
        target = target.at[source].set(global_slot_to_target_action(source, recipient))
    ultimate = (
        jnp.zeros((MAX_AGENT_SLOTS,), jnp.int32)
        .at[jnp.asarray(ultimates, jnp.int32)]
        .set(1)
    )
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, jnp.int32),
        select_target=target,
        use_ultimate=ultimate,
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


def _slots(mask: Array) -> list[int]:
    return [slot for slot, value in enumerate(mask.tolist()) if value]


def _events(info: Info) -> dict[str, list[int]]:
    facts = info.transition_facts.capture_the_flag_facts
    return {
        "picked up": _slots(facts.flag_was_picked_up_by_agent),
        "dropped": _slots(facts.flag_was_dropped_by_agent),
        "returned": _slots(facts.ally_flag_was_returned_by_agent),
        "timer": _slots(facts.flag_was_automatically_returned_by_team),
    }


def _none(**events: list[int]) -> dict[str, list[int]]:
    return {"picked up": [], "dropped": [], "returned": [], "timer": [], **events}


def _team_a_flag(state: EnvState) -> tuple[list[float], int, int]:
    return (
        state.capture_the_flag_flag_positions[0].tolist(),
        int(state.capture_the_flag_flag_carrier_slots[0]),
        int(state.capture_the_flag_dropped_flag_countdowns[0]),
    )


def _dropped_at(point: tuple[float, float], countdown: int = 10) -> Arrange:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            capture_the_flag_flag_positions=state.capture_the_flag_flag_positions.at[
                0
            ].set(jnp.asarray(point, jnp.float32)),
            capture_the_flag_dropped_flag_countdowns=(
                state.capture_the_flag_dropped_flag_countdowns.at[0].set(countdown)
            ),
        )

    return arrange


def _carried_by(slot: int, point: tuple[float, float]) -> Arrange:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            current_health=state.current_health.at[slot].set(1),
            capture_the_flag_flag_positions=state.capture_the_flag_flag_positions.at[
                0
            ].set(jnp.asarray(point, jnp.float32)),
            capture_the_flag_flag_carrier_slots=(
                state.capture_the_flag_flag_carrier_slots.at[0].set(slot)
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
    assert _team_a_flag(first)[1] == -1
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
    position, carrier, countdown = _team_a_flag(state)
    assert (position, carrier, countdown) == (
        [float(jnp.float32(x)) for x in flag[0]],
        flag[1],
        flag[2],
    )


def test_a_flag_sent_home_stays_home_that_tick() -> None:
    start = _start({1: (8.45, 5.0), 5: (0.9, 5.0)}, _dropped_at((8.0, 5.0)))
    (first, first_info), (_, second_info) = _play(start, [_act(), _act()])
    assert _events(first_info) == _none(returned=[1])
    assert _team_a_flag(first) == (_TEAM_A_HOME, -1, 0)
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
    assert _team_a_flag(first) == ([8.0, 5.0], -1, 30)
    assert _events(second_info) == _none(**{"picked up": [6]})
    assert _team_a_flag(second) == ([9.100000381469727, 5.0], 6, 0)


def test_an_agent_killed_this_tick_still_returns_a_flag() -> None:
    def weak_owner(state: EnvState) -> EnvState:
        return _dropped_at((8.0, 5.0))(state)._replace(
            current_health=state.current_health.at[1].set(1)
        )

    start = _start({1: (8.45, 5.0), 5: (11.0, 5.0)}, weak_owner)
    ((state, info),) = _play(start, [_act(attacks={5: 1})])
    assert _slots(info.transition_facts.death_facts.is_newly_dead_by_recipient) == [1]
    assert _events(info) == _none(returned=[1])
    assert _team_a_flag(state) == (_TEAM_A_HOME, -1, 0)


def test_an_agent_killed_this_tick_picks_up_and_drops_at_its_body() -> None:
    def weak_enemy(state: EnvState) -> EnvState:
        return _dropped_at((8.0, 5.0))(state)._replace(
            current_health=state.current_health.at[6].set(1)
        )

    start = _start({6: (8.3, 5.0), 0: (9.35, 5.0)}, weak_enemy)
    ((state, info),) = _play(start, [_act(attacks={0: 6})])
    assert _slots(info.transition_facts.death_facts.is_newly_dead_by_recipient) == [6]
    assert _events(info) == _none(**{"picked up": [6], "dropped": [6]})
    assert _team_a_flag(state) == (state.agent_positions[6].tolist(), -1, 30)


def test_a_carrier_killed_in_its_own_charge_drops_the_flag_at_the_charge_end() -> None:
    start = _start({6: (11.0, 6.0), 0: (8.5, 6.0)}, _carried_by(6, (11.0, 6.0)))
    ((state, info),) = _play(start, [_act(attacks={6: 0, 0: 6}, ultimates=(6,))])
    charge = info.transition_facts.physical_facts.charge_phase_displacement_by_agent
    assert charge[6].tolist() == [-1.5, 0.0]
    assert _slots(info.transition_facts.death_facts.is_newly_dead_by_recipient) == [6]
    assert _events(info) == _none(dropped=[6])
    assert _team_a_flag(state) == ([9.5, 6.0], -1, 30)
    assert state.agent_positions[6].tolist() == [9.5, 6.0]


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_the_dropped_countdown_sends_the_flag_home_on_time(steps: int) -> None:
    config = _CONFIG._replace(capture_the_flag_dropped_flag_auto_return_steps=steps)
    start = _start({5: (8.0, 5.0), 0: (5.5, 5.0)}, _carried_by(5, (8.0, 5.0)), config)
    ticks = _play(start, [_act(attacks={0: 5})] + [_act()] * steps, config)
    countdowns = [_team_a_flag(state)[2] for state, _ in ticks]
    assert countdowns == list(range(steps, -1, -1))
    assert [_events(info)["timer"] for _, info in ticks] == [[]] * steps + [[0]]
    assert [_team_a_flag(state)[0] for state, _ in ticks] == [[8.0, 5.0]] * steps + [
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
    assert _team_a_flag(state) == ([8.0, 5.0], -1, 30)
    assert state.agent_positions[7].tolist() != [8.0, 5.0]
