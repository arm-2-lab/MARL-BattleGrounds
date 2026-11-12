"""Check that the baseline encoder reads Capture the Flag's flags alike from both ends.

A two-lane game on map 0 with balanced spawns puts Team A on the left bank in
lane 0 and on the right bank in lane 1, the exact mirror image, flag homes
included. Decision 139's three flag states are authored into both lanes as
mirror twins: both flags at home (the reset), both carried in sight of the
other team (ours by an enemy, theirs by one of us), and both dropped with
different return countdowns. For each state, seen by Team A and by Team B,
the encoded actor features (encode_actor_inputs, read at ACTOR_FEATURE_OFFSETS)
hold the decision's control columns on the flag rows in both lanes: at home
ours 1, 0 and theirs 0, 1; carried ours 0, 1 and theirs 1, 0; dropped 0, 0
with each flag's countdown in column 6. A carried flag names its carrier's
unit row for someone in each lane. In the world frame the far lane's flag and
flag-home rows differ from the near lane's only in x, which is reflected (map
width minus x). In the left frame (spawn_frame_flag, then mirror_team_view)
the near lane is unchanged and the far lane's whole encoded vector equals the
near lane's, the teammates' flag rows in the source bank included, so a
baseline sees the same game from either end.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.baselines.inputs import (
    ACTOR_FEATURE_OFFSETS,
    encode_actor_inputs,
    spawn_frame_flag,
)
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    MAX_AGENTS_PER_TEAM,
    OBJECTIVE_FEATURE_IS_CONTROLLED_BY_ALLY_TEAM,
    OBJECTIVE_FEATURE_STEPS_UNTIL_DROPPED_FLAG_AUTO_RETURNS,
    OBJECTIVE_FEATURE_UNIT_ROW_0_IS_FLAG_CARRIER,
    OBJECTIVE_FEATURE_X,
    OBJECTIVE_FEATURES,
    OBJECTIVE_SLOT_ALLY_FLAG,
    OBJECTIVE_SLOT_ENEMY_FLAG,
    EnvConfig,
)
from marl_battlegrounds.environment import EnvironmentState, make
from marl_battlegrounds.evaluation.policy_execution import system_inputs
from marl_battlegrounds.policies.input import Observations, mirror_team_view
from marl_battlegrounds.tasks import (
    AgentClassName,
    _swap_spawn_banks,  # pyright: ignore[reportPrivateUsage]
    balanced_spawn_configs,
    make_standard_capture_the_flag_config,
)

_WIDTH = 20.0
_FLAG_ROWS = 4  # Both flag homes, then both flags.
_CONTROL = slice(
    OBJECTIVE_FEATURE_IS_CONTROLLED_BY_ALLY_TEAM,
    OBJECTIVE_FEATURE_IS_CONTROLLED_BY_ALLY_TEAM + 2,
)
_CARRIER = slice(
    OBJECTIVE_FEATURE_UNIT_ROW_0_IS_FLAG_CARRIER,
    OBJECTIVE_FEATURE_UNIT_ROW_0_IS_FLAG_CARRIER + MAX_AGENTS_PER_TEAM,
)
_COUNTDOWN = OBJECTIVE_FEATURE_STEPS_UNTIL_DROPPED_FLAG_AUTO_RETURNS
# Lane 0's authored flags, Team A's first: positions, carrier slots (-1 for
# none) and return countdowns. Each carrier stands at its flag, near the other
# team's bank. None keeps the reset, both flags at home.
_AUTHORED = {
    "home": None,
    "carried": (((3.0, 3.0), (17.0, 7.0)), (6, 1), (0, 0)),
    "dropped": (((6.0, 2.0), (13.0, 8.0)), (-1, -1), (10, 4)),
}
# Decision 139's columns 4 and 5 on our flag's row, then on theirs.
_CONTROL_CODES = {
    "home": ((1, 0), (0, 1)),
    "carried": ((0, 1), (1, 0)),
    "dropped": ((0, 0), (0, 0)),
}


_ROSTER: tuple[AgentClassName, ...] = ("mage", "warrior", "hunter", "rogue", "priest")
type _Reset = tuple[EnvironmentState, tuple[EnvConfig, EnvConfig]]


@pytest.fixture(scope="module")
def reset() -> _Reset:
    config = make_standard_capture_the_flag_config(
        map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER
    )
    batch = balanced_spawn_configs(config, num_envs=2)
    _, state = make("ctf", env_config=batch, num_envs=2).reset(jax.random.key(11))
    # Each lane's host config, which authoring needs: lane 1 exchanges the
    # spawn ends, and with them the flag homes, as balanced_spawn_configs does.
    return state, (config, _swap_spawn_banks(config))


def _lane[T](value: T, lane: int) -> T:
    def take(leaf: Array) -> Array:
        return leaf[lane]

    return jax.tree.map(take, value)


def _authored(reset: _Reset, flag_state: str) -> EnvironmentState:
    state, configs = reset
    lanes = []
    for lane in range(2):
        game = _lane(state.core_state, lane)
        game = game._replace(
            spawn_shield_durations=jnp.zeros_like(game.spawn_shield_durations)
        )
        if (authored := _AUTHORED[flag_state]) is not None:
            points, carriers, countdowns = authored
            flags = jnp.asarray(
                [(x if lane == 0 else _WIDTH - x, y) for x, y in points], jnp.float32
            )
            where = game.agent_positions
            for slot, point in zip(carriers, flags, strict=True):
                if slot >= 0:
                    where = where.at[slot].set(point)
            game = game._replace(
                agent_positions=where,
                capture_the_flag_flag_positions=flags,
                capture_the_flag_flag_carrier_slots=jnp.asarray(carriers, jnp.int32),
                capture_the_flag_dropped_flag_countdowns=jnp.asarray(
                    countdowns, jnp.int32
                ),
            )
        lanes.append(core.initialize_scenario_state(game, configs[lane])[:3])
    game, observation, mask = jax.tree.map(lambda *leaves: jnp.stack(leaves), *lanes)
    return state._replace(core_state=game, observation=observation, action_mask=mask)


@pytest.mark.parametrize("flag_state", list(_AUTHORED))
@pytest.mark.parametrize("team", [0, 1], ids=["team-a", "team-b"])
def test_flag_rows_encode_alike_from_both_spawn_ends(
    reset: _Reset, flag_state: str, team: int
) -> None:
    state = _authored(reset, flag_state)
    inputs = system_inputs(
        Observations(state.observation, state.source_availability), state, team=team
    )
    world = np.asarray(encode_actor_inputs(inputs.actors))
    rows = world[..., ACTOR_FEATURE_OFFSETS["observation.objective_features"]]
    rows = rows.reshape(*rows.shape[:-1], -1, OBJECTIVE_FEATURES)[..., :_FLAG_ROWS, :]
    ours, theirs = _CONTROL_CODES[flag_state]
    for slot, codes in (
        (OBJECTIVE_SLOT_ALLY_FLAG, ours),
        (OBJECTIVE_SLOT_ENEMY_FLAG, theirs),
    ):
        assert (rows[..., slot, _CONTROL] == codes).all(), (flag_state, slot)
    countdowns = (0, 0)
    if flag_state == "dropped":
        countdowns = (10, 4) if team == 0 else (4, 10)
    assert (rows[..., OBJECTIVE_SLOT_ALLY_FLAG:, _COUNTDOWN] == countdowns).all()
    if flag_state == "carried":
        assert rows[..., OBJECTIVE_SLOT_ALLY_FLAG:, _CARRIER].any(axis=(1, 3)).all()
    near, far = team, 1 - team
    np.testing.assert_allclose(
        rows[far, ..., OBJECTIVE_FEATURE_X],
        _WIDTH - rows[near, ..., OBJECTIVE_FEATURE_X],
    )
    other = np.arange(OBJECTIVE_FEATURES) != OBJECTIVE_FEATURE_X
    np.testing.assert_array_equal(rows[far][..., other], rows[near][..., other])
    flag = spawn_frame_flag(inputs.actors, "left")
    actors, _ = mirror_team_view(inputs.actors, inputs.action_mask, flag)
    left = np.asarray(encode_actor_inputs(actors))
    np.testing.assert_array_equal(left[near], world[near])
    np.testing.assert_allclose(left[far], world[near], atol=1e-5, rtol=0)
    bank = ACTOR_FEATURE_OFFSETS["source_bank.objective_features_by_source"]
    assert world[..., bank].any(axis=-1).all()
