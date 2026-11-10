"""Check that the left spawn frame reflects objective rows like unit rows.

Contracts: mirror_team_view sets x to map width minus x on every active
objective row (a row whose active column is above 0), in each flagged actor's
own rows and in its shared sources' rows, and leaves every other column, every
inactive row and every unflagged actor unchanged; rows keep their order, so a
carrier one-hot still names the same unit row. Reflecting twice returns the
input within float rounding, and the compiled call equals the eager one. In
Team Deathmatch, whose objective rows are all zero, the objective arrays come
back unchanged. The objective values are hand-placed on a Capture the Flag
reset, so the check does not wait for Core to fill the rows.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import (
    OBJECTIVE_FEATURE_IS_ACTIVE,
    OBJECTIVE_FEATURE_UNIT_ROW_2_IS_FLAG_CARRIER,
    OBJECTIVE_FEATURE_X,
    OBJECTIVE_FEATURE_Y,
    ActionMask,
)
from marl_battlegrounds.policies.input import (
    ActorInput,
    build_observations,
    build_team_actor_input,
    mirror_team_view,
)
from marl_battlegrounds.tasks import (
    make_standard_capture_the_flag_config,
    make_standard_team_deathmatch_config,
)

_ROSTER = ("mage", "warrior", "priest")
_WIDTH = 20.0


def _team_b_input(task: str) -> tuple[ActorInput, ActionMask]:
    factory = (
        make_standard_capture_the_flag_config
        if task == "ctf"
        else make_standard_team_deathmatch_config
    )
    config = factory(map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER)
    state, observation, mask, _ = reset(config, jax.random.key(0))
    del state
    actors = build_team_actor_input(build_observations(observation, config), team=1)
    return actors, ActionMask(*(field[5:10] for field in mask))


# Rows 0-3 become active objective rows at x 1.5, 2.5, 3.5 and 4.5.
def _placed(rows: Array) -> Array:
    for row in range(4):
        rows = rows.at[..., row, OBJECTIVE_FEATURE_IS_ACTIVE].set(1.0)
        rows = rows.at[..., row, OBJECTIVE_FEATURE_X].set(1.5 + row)
        rows = rows.at[..., row, OBJECTIVE_FEATURE_Y].set(5.0)
    return rows.at[..., 3, OBJECTIVE_FEATURE_UNIT_ROW_2_IS_FLAG_CARRIER].set(1.0)


def test_active_objective_rows_are_reflected_in_own_and_shared_rows() -> None:
    actors, mask = _team_b_input("ctf")
    observation = actors.observation._replace(
        objective_features=_placed(actors.observation.objective_features)
    )
    bank = actors.source_bank._replace(
        objective_features_by_source=_placed(
            actors.source_bank.objective_features_by_source
        )
    )
    actors = ActorInput(observation, bank, actors.source_availability)
    flag = jnp.asarray([True, True, False, False, False])
    mirrored, _ = mirror_team_view(actors, mask, flag)
    for before, after in (
        (observation.objective_features, mirrored.observation.objective_features),
        (
            bank.objective_features_by_source,
            mirrored.source_bank.objective_features_by_source,
        ),
    ):
        before, after = np.asarray(before), np.asarray(after)
        expected = before.copy()
        expected[:2, ..., :4, OBJECTIVE_FEATURE_X] = (
            _WIDTH - before[:2, ..., :4, OBJECTIVE_FEATURE_X]
        )
        np.testing.assert_allclose(after, expected, atol=1e-6, rtol=0)
        assert (
            after[..., 3, OBJECTIVE_FEATURE_UNIT_ROW_2_IS_FLAG_CARRIER] == 1.0
        ).all()
    again, _ = mirror_team_view(mirrored, mask, flag)
    np.testing.assert_allclose(
        np.asarray(again.observation.objective_features),
        np.asarray(observation.objective_features),
        atol=2e-6,
        rtol=0,
    )
    compiled, _ = cast(
        tuple[ActorInput, ActionMask], jax.jit(mirror_team_view)(actors, mask, flag)
    )
    np.testing.assert_array_equal(
        np.asarray(compiled.source_bank.objective_features_by_source),
        np.asarray(mirrored.source_bank.objective_features_by_source),
    )


def test_team_deathmatch_objective_rows_come_back_unchanged() -> None:
    actors, mask = _team_b_input("tdm")
    mirrored, _ = mirror_team_view(actors, mask, jnp.ones(5, jnp.bool_))
    assert not np.asarray(actors.observation.objective_features).any()
    np.testing.assert_array_equal(
        np.asarray(mirrored.observation.objective_features),
        np.asarray(actors.observation.objective_features),
    )
    np.testing.assert_array_equal(
        np.asarray(mirrored.source_bank.objective_features_by_source),
        np.asarray(actors.source_bank.objective_features_by_source),
    )
