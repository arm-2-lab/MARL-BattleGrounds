"""Check the Capture the Flag rules in Core.

Contracts so far: a Capture the Flag game starts with both flags at their
homes, no carrier, no dropped countdown, no captures and every King of the
Hills field neutral, and its first record's Capture the Flag and King of the
Hills facts are Core's neutral records. Played for 20 ticks with random valid
actions on the environment path, every game's state passes Core's state checks
after each tick and the King of the Hills fields stay neutral; Core's steps
keep the King of the Hills facts neutral. A reset and step compiled once serve
a new map, new flag homes and new setting values without compiling again. The
flag life cycle itself is checked in test_capture_the_flag_flag_life_cycle.py.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from tests.helpers.capture_the_flag_fixtures import capture_the_flag_config

from marl_battlegrounds import make
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.config import (
    neutral_objective_fields,
    validate_env_config,
    validate_env_state,
)
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MOVE_STAY,
    Action,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

_ROSTER = ("mage", "warrior", "priest")
_CONFIG = capture_the_flag_config(
    make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER
    )
)
_HILL_FIELDS = {
    name: value
    for name, value in neutral_objective_fields(EnvState, host=True).items()
    if name.startswith("king_of_the_hills_")
}


def _assert_trees_equal(actual: object, expected: object) -> None:
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)


def _assert_objective_facts_neutral(info: core.Info) -> None:
    facts = info.transition_facts
    _assert_trees_equal(
        facts.capture_the_flag_facts, core.build_reset_capture_the_flag_facts()
    )
    _assert_trees_equal(
        facts.king_of_the_hills_facts, core.build_reset_king_of_the_hills_facts()
    )


def test_a_game_starts_with_both_flags_home_and_neutral_hills() -> None:
    state, _, _, info = core.reset(_CONFIG, jax.random.key(0))
    np.testing.assert_array_equal(
        state.capture_the_flag_flag_positions,
        _CONFIG.capture_the_flag_flag_home_positions,
    )
    assert state.capture_the_flag_flag_carrier_slots.tolist() == [-1, -1]
    assert state.capture_the_flag_dropped_flag_countdowns.tolist() == [0, 0]
    assert state.capture_the_flag_scores.tolist() == [0, 0]
    for name, neutral in _HILL_FIELDS.items():
        np.testing.assert_array_equal(getattr(state, name), neutral)
    _assert_objective_facts_neutral(info)


def test_core_steps_keep_the_king_of_the_hills_facts_neutral() -> None:
    state, _, mask, _ = core.reset(_CONFIG, jax.random.key(0))
    stay = Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), jnp.int32),
    )
    for tick in range(3):
        state, _, _, _, mask, info = core.step(
            _CONFIG, state, mask, stay, jax.random.key(tick)
        )
        _assert_trees_equal(
            info.transition_facts.king_of_the_hills_facts,
            core.build_reset_king_of_the_hills_facts(),
        )


def test_twenty_random_ticks_keep_every_state_valid_and_the_hills_neutral() -> None:
    env = make("ctf", env_config=_CONFIG, num_envs=2, metrics="none")
    _, state = env.reset(jax.random.key(0))
    for tick in range(20):
        key = jax.random.fold_in(jax.random.key(1), tick)
        _, state, *_ = env.step(key, state, env.sample_actions(key, state))
        for game in range(2):
            game_state = EnvState(*(field[game] for field in state.core_state))
            validate_env_state(_CONFIG, game_state)
        for name, neutral in _HILL_FIELDS.items():
            assert np.all(np.asarray(getattr(state.core_state, name)) == neutral)


def test_a_new_map_home_or_value_reuses_the_compiled_game() -> None:
    env = make("ctf", env_config=_CONFIG, num_envs=2, metrics="none")
    other = capture_the_flag_config(
        make_standard_team_deathmatch_config(
            map_id=29, team_a_roster=_ROSTER, team_b_roster=_ROSTER
        )
    )._replace(
        capture_the_flag_score_threshold=5,
        capture_the_flag_flag_home_positions=jnp.asarray(
            ((1.0, 4.0), (19.0, 6.0)), jnp.float32
        ),
        capture_the_flag_flag_interaction_distance=1.5,
        capture_the_flag_flag_capture_distance=1.0,
        capture_the_flag_dropped_flag_auto_return_steps=12,
    )
    validate_env_config(other)
    traces: list[int] = []

    @jax.jit
    def play(key: jax.Array, config: EnvConfig) -> jax.Array:
        traces.append(1)
        _, state = env.reset(key, config)
        _, state, *_ = env.step(key, state, env.sample_actions(key, state))
        return state.core_state.capture_the_flag_flag_positions

    for config in (_CONFIG, other):
        positions = cast(jax.Array, play(jax.random.key(0), config))
        np.testing.assert_array_equal(
            positions[0], config.capture_the_flag_flag_home_positions
        )
    assert len(traces) == 1
