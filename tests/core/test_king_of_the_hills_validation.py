"""Check King of the Hills configs and snapshots at the host boundary.

Contracts: make_standard_king_of_the_hills_config builds three hills of radius
1.25 at x = 5, 10 and 15 on a Team Deathmatch map, with 1 point per hill, a
sweep bonus of 3 and an empty-hill timer of 0, and Core accepts it. The points
to win are 100 with three hills and 50 with one. One step gains at most
3 * points + bonus (maximum_king_of_the_hills_score_increment), so the highest
reachable score is threshold + that - 1, and Core accepts a threshold up to
where that score is 2**24. Core refuses each bad setting: points per step below
1, a negative bonus, a threshold below 1 or one whose highest score passes
2**24, a negative timer, an unused slot that is not exactly zero (-0.0
included), a negative or subnormal radius, a center off the map or not finite,
a hill outside its slot's part of the world (slot 1 one float32 step off the
midline included), no hills, hills that overlap or touch, a spawn pad on a
hill, both spawn banks on one half, and an empty team. A snapshot accepts
points up to the highest score and owned, contested and counting-down hills; it
refuses points above it, an owner other than 0, 1 or 2, an owned or contested
unused hill, and a countdown above the timer or on a hill nobody owns. An
authored start must stay below the threshold.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.core.config import (
    maximum_king_of_the_hills_score_increment,
    resolve_agent_profile,
    validate_env_config,
    validate_env_state,
    validate_scenario_initial_state,
)
from marl_battlegrounds.core.types import TASK_MODE_KOTH, EnvConfig, EnvState
from marl_battlegrounds.tasks import (
    load_tdm_scenario,
    make_standard_king_of_the_hills_config,
)

_KOTH = make_standard_king_of_the_hills_config(
    map_id=0, team_a_roster=("mage",), team_b_roster=("priest",)
)
_ONE_HILL = _KOTH._replace(
    king_of_the_hills_hill_positions=jnp.asarray(
        ((0.0, 0.0), (10.0, 5.0), (0.0, 0.0)), jnp.float32
    ),
    king_of_the_hills_hill_radii=jnp.asarray((0.0, 1.25, 0.0), jnp.float32),
)
_CENTERS = _KOTH.king_of_the_hills_hill_positions
_PADS = _KOTH.team_spawn_pad_positions
_JUST_RIGHT_OF_MIDLINE = float(np.nextafter(np.float32(10.0), np.float32(11.0)))


def _radii(*radii: float) -> Array:
    return jnp.asarray(radii, jnp.float32)


def test_the_test_builder_makes_the_default_game() -> None:
    assert validate_env_config(_KOTH) is None
    assert _CENTERS.tolist() == [[5.0, 5.0], [10.0, 5.0], [15.0, 5.0]]
    assert _KOTH.king_of_the_hills_hill_radii.tolist() == [1.25] * 3
    assert (
        _KOTH.task_mode,
        _KOTH.king_of_the_hills_score_threshold,
        _KOTH.king_of_the_hills_hill_points_per_step,
        _KOTH.king_of_the_hills_three_hill_sweep_bonus_points_per_step,
        _KOTH.king_of_the_hills_empty_hill_ownership_steps,
    ) == (TASK_MODE_KOTH, 100, 1, 3, 0)
    one_hill = make_standard_king_of_the_hills_config(
        map_id=0,
        team_a_roster=("mage",),
        team_b_roster=("priest",),
        hill_centers=(None, (10.0, 5.0), None),
    )
    assert one_hill.king_of_the_hills_score_threshold == 50
    assert one_hill.king_of_the_hills_hill_positions.tolist() == [
        [0.0, 0.0],
        [10.0, 5.0],
        [0.0, 0.0],
    ]
    assert one_hill.king_of_the_hills_hill_radii.tolist() == [0.0, 1.25, 0.0]


def test_the_threshold_may_reach_the_last_exact_float32_score() -> None:
    assert maximum_king_of_the_hills_score_increment(1, 3) == 6
    assert maximum_king_of_the_hills_score_increment(2, 5) == 11
    largest = _KOTH._replace(king_of_the_hills_score_threshold=2**24 - 5)
    assert validate_env_config(largest) is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            _KOTH._replace(king_of_the_hills_hill_points_per_step=0),
            "points_per_step must be at least 1",
        ),
        (
            _KOTH._replace(king_of_the_hills_three_hill_sweep_bonus_points_per_step=-1),
            "bonus_points_per_step must be at least 0",
        ),
        (
            _KOTH._replace(king_of_the_hills_score_threshold=0),
            "threshold must be at least 1",
        ),
        (
            _KOTH._replace(king_of_the_hills_score_threshold=2**24 - 4),
            "at most 16777216, so every score stays exact in float32, not 16777217",
        ),
        (
            _KOTH._replace(king_of_the_hills_empty_hill_ownership_steps=-1),
            r"ownership_steps must be in \[0, ",
        ),
        (
            _ONE_HILL._replace(king_of_the_hills_hill_positions=_CENTERS),
            "Hill slot 0 has radius 0, so it is unused",
        ),
        (
            _ONE_HILL._replace(king_of_the_hills_hill_radii=_radii(-0.0, 1.25, 0.0)),
            "Hill slot 0 has radius 0, so it is unused",
        ),
        (
            _KOTH._replace(king_of_the_hills_hill_radii=_radii(-1.0, 1.25, 1.25)),
            r"hill_radii\[0\] must be 0.0 for an unused slot, or finite",
        ),
        (
            _KOTH._replace(king_of_the_hills_hill_radii=_radii(1e-40, 1.25, 1.25)),
            r"hill_radii\[0\] must be 0.0 for an unused slot, or finite",
        ),
        (
            _KOTH._replace(
                king_of_the_hills_hill_positions=_CENTERS.at[0, 0].set(jnp.nan)
            ),
            r"hill_positions\[0\] must lie on the map",
        ),
        (
            _KOTH._replace(
                king_of_the_hills_hill_positions=_CENTERS.at[0, 1].set(-1.0)
            ),
            r"hill_positions\[0\] must lie on the map",
        ),
        (
            _KOTH._replace(king_of_the_hills_hill_positions=_CENTERS[::-1]),
            r"hill_positions\[0\] must have x left of the midline",
        ),
        (
            _KOTH._replace(
                king_of_the_hills_hill_positions=_CENTERS.at[1, 0].set(
                    _JUST_RIGHT_OF_MIDLINE
                )
            ),
            r"hill_positions\[1\] must have x exactly on the midline",
        ),
        (
            _KOTH._replace(
                king_of_the_hills_hill_positions=jnp.zeros((3, 2), jnp.float32),
                king_of_the_hills_hill_radii=jnp.zeros(3, jnp.float32),
            ),
            "needs at least one hill",
        ),
        (
            _KOTH._replace(king_of_the_hills_hill_radii=_radii(2.5, 2.5, 1.0)),
            r"Hills 0 and 1 must be farther apart than their two radii \(5.0\)",
        ),
        (
            _KOTH._replace(
                king_of_the_hills_hill_positions=_CENTERS.at[0].set(_PADS[0, 2])
            ),
            r"team_spawn_pad_positions\[0\]\[2\] lies on hill 0",
        ),
        (
            _KOTH._replace(team_spawn_pad_positions=_PADS.at[1, :, 0].add(-17.0)),
            "two spawn banks on opposite halves",
        ),
        (
            _KOTH._replace(
                agent_profile=resolve_agent_profile(
                    jnp.asarray((1, 0, 0, 0, 0, 0, 0, 0, 0, 0), jnp.int32),
                    jnp.asarray((1, 0), jnp.int32),
                )
            ),
            "at least one configured active member on each team",
        ),
    ],
    ids=[
        "no-hill-points",
        "negative-bonus",
        "zero-threshold",
        "inexact-highest-score",
        "negative-timer",
        "unused-slot-with-center",
        "unused-slot-negative-zero",
        "negative-radius",
        "subnormal-radius",
        "nan-center",
        "center-off-map",
        "hills-in-wrong-slots",
        "midline-one-step-right",
        "no-hills",
        "touching-hills",
        "pad-on-hill",
        "banks-on-one-half",
        "empty-team",
    ],
)
def test_king_of_the_hills_refuses_each_bad_setting(
    config: EnvConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_env_config(config)


_SCENARIO = load_tdm_scenario(1)
_START_CONFIG = _SCENARIO.config._replace(
    task_mode=TASK_MODE_KOTH,
    team_deathmatch_score_threshold=0,
    team_deathmatch_red_zone_depth=0.0,
    **{
        name: getattr(_KOTH, name)
        for name in EnvConfig._fields
        if name.startswith("king_of_the_hills_")
    },
)._replace(king_of_the_hills_empty_hill_ownership_steps=2)
_START = _SCENARIO.initial_state._replace(
    team_deathmatch_scores=jnp.zeros(2, jnp.int32)
)


def _hills(
    owners: tuple[int, int, int] = (0, 0, 0),
    countdowns: tuple[int, int, int] = (0, 0, 0),
    contested: tuple[bool, bool, bool] = (False, False, False),
    scores: tuple[int, int] = (0, 0),
) -> EnvState:
    return _START._replace(
        king_of_the_hills_hill_owners=jnp.asarray(owners, jnp.int32),
        king_of_the_hills_empty_hill_ownership_countdowns=jnp.asarray(
            countdowns, jnp.int32
        ),
        king_of_the_hills_hill_is_contested=jnp.asarray(contested),
        king_of_the_hills_scores=jnp.asarray(scores, jnp.int32),
    )


def test_king_of_the_hills_accepts_owned_contested_and_counting_down_hills() -> None:
    assert validate_scenario_initial_state(_START_CONFIG, _START) is None
    state = _hills(
        owners=(1, 2, 0), countdowns=(2, 1, 0), contested=(True, False, True)
    )
    assert validate_env_state(_START_CONFIG, state) is None
    assert validate_env_state(_START_CONFIG, _hills(scores=(105, 0))) is None


@pytest.mark.parametrize(
    ("config", "state", "message"),
    [
        (
            _START_CONFIG,
            _hills(scores=(0, 106)),
            r"king_of_the_hills_scores must be in \[0, 105\]",
        ),
        (_START_CONFIG, _hills(owners=(3, 0, 0)), r"hill_owners\[0\] must be 0"),
        (
            _START_CONFIG._replace(
                king_of_the_hills_hill_positions=_ONE_HILL.king_of_the_hills_hill_positions,
                king_of_the_hills_hill_radii=_ONE_HILL.king_of_the_hills_hill_radii,
            ),
            _hills(owners=(0, 0, 2)),
            "Hill slot 2 is unused, so it must be neither contested nor owned",
        ),
        (
            _START_CONFIG._replace(
                king_of_the_hills_hill_positions=_ONE_HILL.king_of_the_hills_hill_positions,
                king_of_the_hills_hill_radii=_ONE_HILL.king_of_the_hills_hill_radii,
            ),
            _hills(contested=(True, False, False)),
            "Hill slot 0 is unused, so it must be neither contested nor owned",
        ),
        (
            _START_CONFIG,
            _hills(owners=(1, 0, 0), countdowns=(3, 0, 0)),
            r"countdowns\[0\] must be in \[0, 2\] on an owned hill",
        ),
        (
            _START_CONFIG,
            _hills(countdowns=(0, 1, 0)),
            r"countdowns\[1\] .* not 1 with owner 0",
        ),
    ],
    ids=[
        "points-above-highest-score",
        "unknown-owner",
        "owned-unused-hill",
        "contested-unused-hill",
        "countdown-above-timer",
        "countdown-on-neutral-hill",
    ],
)
def test_king_of_the_hills_refuses_each_bad_hill_or_score(
    config: EnvConfig, state: EnvState, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_env_state(config, state)


def test_an_authored_king_of_the_hills_start_stays_below_the_threshold() -> None:
    with pytest.raises(ValueError, match="King of the Hills scenario scores"):
        validate_scenario_initial_state(_START_CONFIG, _hills(scores=(100, 0)))
