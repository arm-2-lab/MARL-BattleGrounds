"""Check King of the Hills' hill rows and context columns in Core observations.

Every start is hand-built on map 0 (no obstacles) and goes through
initialize_scenario_state. Contracts: hill rows are by side, never by x: for an
observer whose team's spawn bank is on the left, hill slot h fills row 4 + h;
on the right it fills row 6 - h, for both teams at both spawn ends (map 0's own
ends, and the two banks exchanged). Each used hill's row
holds 1 (active), x, y, the radius, owned by the ally team, owned by the enemy
team (the stored owner, also while empty or contested), contested (the stored
value), the rate, the sweep bonus (0 unless all three hill slots are used) and
ticks left, then two zeros; an unused hill's row is all zero. The owner reaches
an agent only as ally or enemy, so the same hill reads ally to one team and
enemy to the other. Rows 0-3 stay zero in King of the Hills. Context columns: 9
holds the number of used hill slots, 12 and 13 the scores (own, enemy) and 17
the threshold. A one-hill center map fills row 5 alone, with bonus 0 and
context 9 equal to 1. A dead configured observer keeps its hill rows and
context columns, since hills are public; with a 3 against 2 roster every unused
observer's objective rows and context columns are zero. In Team Deathmatch and
Capture the Flag, even with hill settings and hill state present (given
straight to the observation builder, past the host checks that refuse them),
rows 4-6 and context columns 12, 13 and 17 stay zero, and column 9 holds the
mode's own count (0 and 4).
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    CONTEXT_FEATURE_ACTIVE_OBJECTIVE_COUNT,
    CONTEXT_FEATURE_KOTH_ALLY_SCORE,
    CONTEXT_FEATURE_KOTH_ENEMY_SCORE,
    CONTEXT_FEATURE_KOTH_SCORE_THRESHOLD,
    OBJECTIVE_SLOT_FAR_SIDE_HILL,
    OBJECTIVE_SLOT_SPAWN_SIDE_HILL,
    EnvConfig,
    EnvState,
    Observation,
)
from marl_battlegrounds.tasks import (
    AgentClassName,
    _swap_spawn_banks,  # pyright: ignore[reportPrivateUsage]
    make_standard_capture_the_flag_config,
    make_standard_king_of_the_hills_config,
    make_standard_team_deathmatch_config,
)

_Arrange = Callable[[EnvState], EnvState]
_ROSTER: tuple[AgentClassName, ...] = ("mage", "warrior", "priest")
_HILL_ROWS = slice(OBJECTIVE_SLOT_SPAWN_SIDE_HILL, OBJECTIVE_SLOT_FAR_SIDE_HILL + 1)
_KOTH_CONTEXT = [
    CONTEXT_FEATURE_KOTH_ALLY_SCORE,
    CONTEXT_FEATURE_KOTH_ENEMY_SCORE,
    CONTEXT_FEATURE_KOTH_SCORE_THRESHOLD,
]


_config = partial(
    make_standard_king_of_the_hills_config,
    map_id=0,
    team_a_roster=_ROSTER,
    team_b_roster=_ROSTER,
)


def _observe(
    arrange: _Arrange | None = None, config: EnvConfig | None = None
) -> Observation:
    config = _config() if config is None else config
    state, _, _, _ = core.reset(config, jax.random.key(0))
    if arrange is not None:
        state = arrange(state)
    _, observation, _, _ = core.initialize_scenario_state(state, config)
    return observation


def _hill_state(
    owners: list[int], contested: list[bool], ticks_left: list[int], scores: list[int]
) -> _Arrange:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            king_of_the_hills_hill_owners=jnp.asarray(owners, jnp.int32),
            king_of_the_hills_hill_is_contested=jnp.asarray(contested, jnp.bool_),
            king_of_the_hills_empty_hill_ownership_countdowns=jnp.asarray(
                ticks_left, jnp.int32
            ),
            king_of_the_hills_scores=jnp.asarray(scores, jnp.int32),
        )

    return arrange


def _rows(observation: Observation, observer: int) -> list[list[float]]:
    return np.asarray(observation.objective_features[observer]).tolist()


def _hill(
    x: float, ally: int, enemy: int, contested: int, ticks: int, bonus: float
) -> list[float]:
    return [1.0, x, 5.0, 1.25, ally, enemy, contested, 1.0, bonus, ticks, 0.0, 0.0]


@pytest.mark.parametrize("swapped", [False, True], ids=["team-a-left", "team-a-right"])
def test_three_hills_fill_rows_4_to_6_by_side_with_owners_as_ally_or_enemy(
    swapped: bool,
) -> None:
    config = _config(empty_hill_ownership_steps=2)
    # Hill 0 is Team A's with 1 tick left, hill 1 is Team B's and contested,
    # hill 2 is neutral.
    observation = _observe(
        _hill_state([1, 2, 0], [False, True, False], [1, 2, 0], [7, 4]),
        _swap_spawn_banks(config) if swapped else config,
    )
    # Hill slots 0, 1 and 2 from the left, as each team reads them.
    team_a_view = [
        _hill(5.0, 1, 0, 0, 1, 3.0),
        _hill(10.0, 0, 1, 1, 2, 3.0),
        _hill(15.0, 0, 0, 0, 0, 3.0),
    ]
    team_b_view = [
        _hill(5.0, 0, 1, 0, 1, 3.0),
        _hill(10.0, 1, 0, 1, 2, 3.0),
        _hill(15.0, 0, 0, 0, 0, 3.0),
    ]
    assert _rows(observation, 0)[_HILL_ROWS] == (
        team_a_view[::-1] if swapped else team_a_view
    )
    assert _rows(observation, 5)[_HILL_ROWS] == (
        team_b_view if swapped else team_b_view[::-1]
    )
    for observer, scores in ((0, [7.0, 4.0]), (5, [4.0, 7.0])):
        assert not np.asarray(_rows(observation, observer)[:4]).any()
        context = np.asarray(observation.context_features[observer])
        assert context[CONTEXT_FEATURE_ACTIVE_OBJECTIVE_COUNT] == 3.0
        assert context[_KOTH_CONTEXT].tolist() == [*scores, 100.0]


def test_a_one_hill_center_map_fills_row_5_alone_without_a_bonus() -> None:
    config = _config(hill_centers=(None, (10.0, 5.0), None))
    observation = _observe(config=config)
    for observer in (0, 5):
        rows = _rows(observation, observer)
        assert rows[_HILL_ROWS] == [
            [0.0] * 12,
            _hill(10.0, 0, 0, 0, 0, 0.0),
            [0.0] * 12,
        ]
        context = np.asarray(observation.context_features[observer])
        assert context[CONTEXT_FEATURE_ACTIVE_OBJECTIVE_COUNT] == 1.0
        assert context[CONTEXT_FEATURE_KOTH_SCORE_THRESHOLD] == float(
            config.king_of_the_hills_score_threshold
        )


def test_a_dead_observer_keeps_its_hill_rows_and_unused_observers_are_zero() -> None:
    config = make_standard_king_of_the_hills_config(
        map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER[:2]
    )

    def dead(state: EnvState) -> EnvState:
        state = _hill_state([1, 0, 0], [False] * 3, [0, 0, 0], [3, 0])(state)
        return state._replace(
            alive_mask=state.alive_mask.at[1].set(False),
            current_health=state.current_health.at[1].set(0),
        )

    observation = _observe(dead, config)
    assert np.asarray(_rows(observation, 0)).any()
    assert _rows(observation, 1) == _rows(observation, 0)
    context = np.asarray(observation.context_features)
    np.testing.assert_array_equal(context[1, _KOTH_CONTEXT], context[0, _KOTH_CONTEXT])
    unused = ~np.asarray(config.agent_profile.active_mask)
    assert not np.asarray(observation.objective_features)[unused].any()
    assert not np.asarray(observation.context_features)[unused].any()


def test_other_modes_write_no_hill_rows_even_with_hill_data_present() -> None:
    koth = _config()
    names = (
        "king_of_the_hills_hill_positions",
        "king_of_the_hills_hill_radii",
        "king_of_the_hills_hill_points_per_step",
        "king_of_the_hills_three_hill_sweep_bonus_points_per_step",
        "king_of_the_hills_empty_hill_ownership_steps",
        "king_of_the_hills_score_threshold",
    )
    hills = {name: getattr(koth, name) for name in names}
    for factory, objective_count in (
        (make_standard_team_deathmatch_config, 0.0),
        (make_standard_capture_the_flag_config, 4.0),
    ):
        config = factory(map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER)
        state, _, _, _ = core.reset(config, jax.random.key(0))
        state = _hill_state([1, 2, 0], [False, True, False], [1, 2, 0], [7, 4])(state)
        # Past the host checks, which refuse hill data outside King of the Hills.
        observation, _ = core._build_observation_and_action_mask(  # pyright: ignore[reportPrivateUsage]
            state, config._replace(**hills)
        )
        assert not np.asarray(observation.objective_features)[:, _HILL_ROWS].any()
        context = np.asarray(observation.context_features)
        assert not context[:, _KOTH_CONTEXT].any()
        active = np.asarray(config.agent_profile.active_mask)
        assert (
            context[active, CONTEXT_FEATURE_ACTIVE_OBJECTIVE_COUNT] == objective_count
        ).all()
