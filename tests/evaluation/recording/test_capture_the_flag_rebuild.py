"""Check that saved Capture the Flag records rebuild into Core's own state and facts.

A short game on the shared test arena (three champions against two, flag homes
at (1.5, 6) and (18.5, 6)) starts with Team B's flag carried by Team A's slot 0
and Team A's flag dropped under Team A's slot 1. So the first tick records a
return by touch, and a carrier stays in play. Each frame and transition is
captured under context V5, then rebuilt with reconstruct_env_state_v1 and
reconstruct_transition_facts_v1, on the device and on the host. Contracts:
every rebuilt leaf equals Core's own leaf in value, shape and dtype. That covers
the flag fields, a missing carrier as -1, Team Deathmatch's zero scores and
outcome, and King of the Hills' neutral fields. Host only, on CPU.
"""

# pyright: reportPrivateUsage=false

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.evaluation.recording.test_capture_the_flag_records import (
    _CONTEXT,
    _ctf_config,
)
from tests.helpers.evaluation_fixtures import neutral_action

from marl_battlegrounds.core.env import initialize_scenario_state, reset, step
from marl_battlegrounds.evaluation._recording.capture import (
    capture_evaluation_transition_unit,
    capture_initial_evaluation_frame,
    reconstruct_env_state_v1,
    reconstruct_transition_facts_v1,
)


def _same(core: object, rebuilt: object) -> None:
    assert jax.tree.structure(rebuilt) == jax.tree.structure(core)
    leaves = cast(
        list[tuple[object, object]],
        jax.tree_util.tree_leaves_with_path(core),
    )
    for (path, expected), actual in zip(leaves, jax.tree.leaves(rebuilt), strict=True):
        where = str(path)
        expected, actual = np.asarray(expected), np.asarray(actual)
        assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape), where
        np.testing.assert_array_equal(actual, expected, err_msg=where)


@pytest.mark.parametrize("host", [False, True])
def test_capture_the_flag_records_rebuild_into_cores_own_state_and_facts(
    host: bool,
) -> None:
    config = _ctf_config()
    state = reset(config, jax.random.PRNGKey(0))[0]
    dropped = jnp.asarray((6.0, 6.0), jnp.float32)
    state = state._replace(
        agent_positions=state.agent_positions.at[1].set(dropped),
        capture_the_flag_flag_positions=state.capture_the_flag_flag_positions.at[0]
        .set(dropped)
        .at[1]
        .set(state.agent_positions[0]),
        capture_the_flag_flag_carrier_slots=(
            state.capture_the_flag_flag_carrier_slots.at[1].set(0)
        ),
        capture_the_flag_dropped_flag_countdowns=(
            state.capture_the_flag_dropped_flag_countdowns.at[0].set(5)
        ),
        spawn_shield_durations=jnp.zeros_like(state.spawn_shield_durations),
    )
    state, observation, mask, _ = initialize_scenario_state(state, config)
    frame = capture_initial_evaluation_frame(_CONTEXT, state, observation, mask)
    returned = False
    for tick in range(3):
        _same(state, reconstruct_env_state_v1(frame, host=host))
        state, observation, reward, done, mask, info = step(
            config, state, mask, neutral_action(), jax.random.PRNGKey(tick + 1)
        )
        transition, frame = capture_evaluation_transition_unit(
            _CONTEXT,
            frame,
            state,
            observation,
            mask,
            info.transition_facts,
            reward,
            done,
        )
        _same(
            info.transition_facts,
            reconstruct_transition_facts_v1(transition.facts, host=host),
        )
        flags = info.transition_facts.capture_the_flag_facts
        returned |= bool(flags.ally_flag_was_returned_by_agent.any())
    _same(state, reconstruct_env_state_v1(frame, host=host))
    assert returned and int(state.capture_the_flag_flag_carrier_slots[1]) == 0
    assert int(state.capture_the_flag_flag_carrier_slots[0]) == -1
