"""Check Capture the Flag's objective rows and context columns in Core observations.

Every start is hand-built on map 0 (no obstacles) with three agents a team and
goes through initialize_scenario_state; each check runs for Team A and for its
mirror image as Team B. Contracts: an observer's flag rows (0-3) pass the
recorded-row decoder (decode_capture_the_flag_objective_rows_v1): the home
rows name their owner (yours 1, 0, the enemy's 0, 1) and each flag's state
follows decision 139, at home (both flags at the start, from both teams),
carried by an enemy or dropped with its countdown, and the enemy flag the same
way seen from the other side. The carrier one-hot names the carrier's roster
row only when the observer sees it or is it (the observer's own
self_ally_index); a hidden carrier, either team's, gives an empty one-hot
while the flag's position stays public, and a flag nobody carries names no
row. A dead configured observer keeps its rows with an empty one-hot. Context
columns: 9 holds 4, 14 and 15 the captures (own, enemy, from both teams) and
18 the threshold. With a 3 against 2 roster every unused observer's objective
rows and context columns are zero, and in Team Deathmatch the objective rows
and columns 9, 14, 15 and 18 stay zero.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    CONTEXT_FEATURE_ACTIVE_OBJECTIVE_COUNT,
    CONTEXT_FEATURE_CTF_ALLY_SCORE,
    CONTEXT_FEATURE_CTF_ENEMY_SCORE,
    CONTEXT_FEATURE_CTF_SCORE_THRESHOLD,
    MAX_AGENT_SLOTS,
    EnvConfig,
    EnvState,
    Observation,
)
from marl_battlegrounds.rendering.evaluation_wire_features import (
    CaptureTheFlagObjectiveRowsV1,
    decode_capture_the_flag_objective_rows_v1,
)
from marl_battlegrounds.tasks import (
    AgentClassName,
    make_standard_capture_the_flag_config,
    make_standard_team_deathmatch_config,
)

_Arrange = Callable[[EnvState], EnvState]
_ROSTER: tuple[AgentClassName, ...] = ("mage", "warrior", "priest")
_CONFIG = make_standard_capture_the_flag_config(
    map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER
)
_FAR = {
    0: (3.0, 1.0),
    1: (3.0, 3.0),
    2: (3.0, 9.0),
    5: (17.0, 1.0),
    6: (17.0, 3.0),
    7: (17.0, 9.0),
}
_SIDES = pytest.mark.parametrize("us", [0, 1], ids=["team-a", "team-b"])


def _slot(us: int, slot: int) -> int:
    return slot + 5 * us


def _point(us: int, x: float, y: float) -> tuple[float, float]:
    return (x if us == 0 else 20.0 - x, y)


def _observe(
    positions: dict[int, tuple[float, float]],
    arrange: _Arrange | None = None,
    config: EnvConfig = _CONFIG,
) -> tuple[EnvState, Observation]:
    state, _, _, _ = core.reset(config, jax.random.key(0))
    where = state.agent_positions
    for slot, point in {**_FAR, **positions}.items():
        if slot < MAX_AGENT_SLOTS and bool(config.agent_profile.active_mask[slot]):
            where = where.at[slot].set(jnp.asarray(point, jnp.float32))
    state = state._replace(
        agent_positions=where,
        spawn_shield_durations=jnp.zeros(MAX_AGENT_SLOTS, jnp.int32),
    )
    if arrange is not None:
        state = arrange(state)
    state, observation, _, _ = core.initialize_scenario_state(state, config)
    return state, observation


def _rows(observation: Observation, observer: int) -> CaptureTheFlagObjectiveRowsV1:
    rows = np.asarray(observation.objective_features[observer]).tolist()
    return decode_capture_the_flag_objective_rows_v1(
        tuple(tuple(float(value) for value in row) for row in rows)
    )


def _carry(slot: int, point: tuple[float, float], flag: int) -> _Arrange:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            capture_the_flag_flag_positions=state.capture_the_flag_flag_positions.at[
                flag
            ].set(jnp.asarray(point, jnp.float32)),
            capture_the_flag_flag_carrier_slots=(
                state.capture_the_flag_flag_carrier_slots.at[flag].set(slot)
            ),
        )

    return arrange


def _drop(point: tuple[float, float], flag: int, countdown: int = 10) -> _Arrange:
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


def _sees(observation: Observation, observer: int, slot: int) -> bool:
    own_team = observer // 5
    if slot // 5 == own_team:
        return bool(observation.ally_visibility_mask[observer, slot - 5 * own_team])
    return bool(observation.enemy_visibility_mask[observer, slot - 5 * (1 - own_team)])


@_SIDES
def test_flag_rows_name_every_flag_state_from_both_sides(us: int) -> None:
    ours, theirs = _slot(us, 0), _slot(1 - us, 0)
    carrier = _slot(1 - us, 1)
    home = _point(us, 0.5, 5.0)
    # Our flag carried by their Warrior, in sight of our Mage and of their own
    # Mage; their flag dropped. We have 2 captures, they have 1.
    scores = jnp.asarray((2, 1) if us == 0 else (1, 2), jnp.int32)
    _, observation = _observe(
        {
            ours: _point(us, 6.0, 5.0),
            carrier: _point(us, 7.0, 5.0),
            theirs: _point(us, 8.5, 6.5),
        },
        lambda state: _drop(_point(us, 12.0, 5.0), 1 - us)(
            _carry(carrier, _point(us, 7.0, 5.0), us)(state)
        )._replace(capture_the_flag_scores=scores),
    )
    assert _sees(observation, ours, carrier) and _sees(observation, theirs, carrier)
    mine = _rows(observation, ours)
    assert mine.ally_home == pytest.approx(home)
    assert (mine.ally_flag.carried, mine.ally_flag.carrier_row) == (True, 1)
    assert (mine.enemy_flag.carried, mine.enemy_flag.countdown_steps) == (False, 10)
    assert mine.enemy_flag.position == pytest.approx(_point(us, 12.0, 5.0))
    yours = _rows(observation, theirs)
    assert yours.enemy_flag.carried and yours.enemy_flag.carrier_row == 1
    assert yours.ally_flag.countdown_steps == 10
    for observer, captures in ((ours, (2.0, 1.0)), (theirs, (1.0, 2.0))):
        context = np.asarray(observation.context_features[observer])
        assert context[CONTEXT_FEATURE_ACTIVE_OBJECTIVE_COUNT] == 4.0
        assert (
            context[CONTEXT_FEATURE_CTF_ALLY_SCORE],
            context[CONTEXT_FEATURE_CTF_ENEMY_SCORE],
        ) == captures
        assert context[CONTEXT_FEATURE_CTF_SCORE_THRESHOLD] == 3.0


@_SIDES
def test_a_hidden_carrier_is_never_named_but_its_flag_stays_public(us: int) -> None:
    ours = _slot(us, 0)
    enemy_carrier, ally_carrier = _slot(1 - us, 1), _slot(us, 2)
    # Both carriers far from our Mage, beyond its sight.
    _, observation = _observe(
        {
            ours: _point(us, 1.0, 1.0),
            enemy_carrier: _point(us, 18.0, 9.0),
            ally_carrier: _point(us, 18.0, 1.0),
        },
        lambda state: _carry(enemy_carrier, _point(us, 18.0, 9.0), us)(
            _carry(ally_carrier, _point(us, 18.0, 1.0), 1 - us)(state)
        ),
    )
    assert not _sees(observation, ours, enemy_carrier)
    assert not _sees(observation, ours, ally_carrier)
    mine = _rows(observation, ours)
    assert mine.ally_flag.carrier_row is None
    assert mine.enemy_flag.carrier_row is None
    assert mine.ally_flag.position == pytest.approx(_point(us, 18.0, 9.0))
    # The carrier itself is named at its own row.
    self_rows = _rows(observation, ally_carrier)
    assert self_rows.enemy_flag.carrier_row == int(
        observation.self_ally_index[ally_carrier]
    )


@_SIDES
def test_a_dead_observer_keeps_its_rows_without_naming_a_carrier(us: int) -> None:
    ours, carrier = _slot(us, 0), _slot(1 - us, 1)

    def dead_observer(state: EnvState) -> EnvState:
        state = _carry(carrier, _point(us, 7.0, 5.0), us)(state)
        return state._replace(
            alive_mask=state.alive_mask.at[ours].set(False),
            current_health=state.current_health.at[ours].set(0),
        )

    _, observation = _observe(
        {ours: _point(us, 6.0, 5.0), carrier: _point(us, 7.0, 5.0)}, dead_observer
    )
    mine = _rows(observation, ours)
    assert mine.ally_flag.carried and mine.ally_flag.carrier_row is None


def test_unused_observers_are_zero_and_team_deathmatch_has_no_flag_rows() -> None:
    config = make_standard_capture_the_flag_config(
        map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER[:2]
    )
    _, observation = _observe({}, config=config)
    # At the start both flags are home, read the same way from both teams.
    homes = np.asarray(config.capture_the_flag_flag_home_positions).tolist()
    for observer, (own, enemy) in ((0, (0, 1)), (5, (1, 0))):
        rows = _rows(observation, observer)
        assert rows.ally_home == pytest.approx(tuple(homes[own]))
        assert rows.enemy_home == pytest.approx(tuple(homes[enemy]))
        for flag, home in (
            (rows.ally_flag, homes[own]),
            (rows.enemy_flag, homes[enemy]),
        ):
            assert (flag.carried, flag.countdown_steps, flag.carrier_row) == (
                False,
                0,
                None,
            )
            assert flag.position == pytest.approx(tuple(home))
    unused = ~np.asarray(config.agent_profile.active_mask)
    assert not np.asarray(observation.objective_features)[unused].any()
    assert not np.asarray(observation.context_features)[unused].any()
    tdm = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER
    )
    _, observation = _observe({}, config=tdm)
    assert not np.asarray(observation.objective_features).any()
    columns = [
        CONTEXT_FEATURE_ACTIVE_OBJECTIVE_COUNT,
        CONTEXT_FEATURE_CTF_ALLY_SCORE,
        CONTEXT_FEATURE_CTF_ENEMY_SCORE,
        CONTEXT_FEATURE_CTF_SCORE_THRESHOLD,
    ]
    assert not np.asarray(observation.context_features)[:, columns].any()
