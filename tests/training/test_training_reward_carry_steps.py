"""Check the carry_steps counter a training reward can declare.

The counting rule: a team's value is the ticks its current carrier has held the
enemy flag at the start of a tick, this tick included, so a pickup on tick p and
a death on tick p + k read k on the death tick; it is 0 without a carrier,
restarts after a capture, a drop or a change of carrier and at every game's
first tick (a game that starts with a carrier counts from 1), and padding leaves
it unchanged. The reward contract: a required carry_steps that is missing
fails, a wrong shape or dtype fails, and the reward's identity records the
contract only when the reward declares it. Collection: a reward that declares
carry_steps gets an int32 (B, 2) counter in the carry, sees it advance tick by
tick and restart at 0 when its game ends and a new one starts; a reward without
it keeps an empty field, so its carry and checkpoint layout are unchanged; a
changed reward that starts using it gets a zero counter, and one that stops
drops it. Runs on CPU with the collection tests' synthetic two-tick Team
Deathmatch games, where Core passes a placed carrier through unchanged.
"""

# pyright: reportPrivateUsage=false
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.training.test_training_collection import _actor, _scan, _synthetic

import marl_battlegrounds.training.collection as collection_module
from marl_battlegrounds.core.types import EnvState
from marl_battlegrounds.environment import TrainingFacts
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    init_training_collection,
    make_training_schedule,
    prepare_training_content,
)
from marl_battlegrounds.training.shaping import (
    CARRY_STEPS_CONTRACT,
    RewardFunction,
    _prepare_reward,
    _reward_output,
    _team_carry_steps,
    resolve_reward,
)

# (first tick of a game, carriers before, carriers after, real transition,
# expected carry_steps), one lane, by flag: Team A carries flag 1, Team B flag 0.
_TICKS = (
    (True, (-1, -1), (-1, 0), True, (0, 0)),  # Team A's agent 0 picks up flag 1.
    (False, (-1, 0), (7, 0), True, (1, 0)),  # Team B's agent 7 picks up flag 0.
    (False, (7, 0), (7, 0), False, (2, 1)),  # Padding: read, but the counter stays.
    (False, (7, 0), (7, 0), True, (2, 1)),
    (False, (7, 0), (7, -1), True, (3, 2)),  # Team A's carrier dies 3 ticks on.
    (False, (7, -1), (7, 1), True, (0, 3)),  # Team A picks the flag up again.
    (False, (7, 1), (-1, 1), True, (1, 4)),  # Team B captures.
    (False, (-1, 1), (-1, 1), True, (2, 0)),
    (True, (-1, 4), (-1, 4), True, (1, 0)),  # A new game starts with a carrier.
    # Picked up and killed on one tick: the slots read -1 before and after, so 0;
    # a reward sees that death through Core's drop fact.
    (False, (-1, -1), (-1, -1), True, (0, 0)),
)


def _steps_reward(
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    progress: Array,
    *,
    carry_steps: Array,
) -> Array:
    del before, facts, after, progress
    return jnp.zeros(10, jnp.float32).at[:2].set(carry_steps.astype(jnp.float32))


def _plain_reward(
    before: EnvState, facts: TrainingFacts, after: EnvState, progress: Array
) -> Array:
    del before, facts, after, progress
    return jnp.zeros(10, jnp.float32)


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


def test_the_counter_follows_each_carry_from_its_pickup() -> None:
    held = jnp.zeros((1, 2), jnp.int32)
    for starting, before, after, real, expected in _TICKS:
        steps, held = _team_carry_steps(
            held,
            jnp.asarray([starting]),
            jnp.asarray([before], jnp.int32),
            jnp.asarray([after], jnp.int32),
            jnp.asarray([real]),
        )
        np.testing.assert_array_equal(steps, [expected])


def test_carry_steps_is_an_opt_in_keyword_with_a_checked_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds import _method_loading

    rewards = {"tests:steps": _steps_reward, "tests:plain": _plain_reward}

    def installed(reference: str) -> RewardFunction:
        return cast(RewardFunction, rewards[reference])

    monkeypatch.setattr(_method_loading, "installed_callable", installed)
    declared = resolve_reward("tests:steps")[1]
    plain = resolve_reward("tests:plain")[1]
    assert declared is not None and plain is not None
    assert declared["carry_steps_contract"] == CARRY_STEPS_CONTRACT
    assert "carry_steps_contract" not in plain

    prepared = _prepare_reward(_steps_reward)
    state = cast(EnvState, None)
    required = len(TrainingFacts._fields) - len(TrainingFacts._field_defaults)
    facts = TrainingFacts(*([jnp.bool_(True)] + [jnp.zeros(())] * (required - 1)))
    with pytest.raises(ValueError, match="carry_steps counter"):
        prepared(state, facts, state, jnp.int32(0))
    for steps, error in (
        (jnp.zeros(3, jnp.int32), ValueError),
        (jnp.zeros(2, jnp.float32), TypeError),
    ):
        with pytest.raises(error, match="carry_steps must have"):
            _reward_output(
                prepared, state, facts, state, jnp.int32(0), carry_steps=steps
            )


def test_collection_keeps_the_counter_only_for_a_reward_that_declares_it(
    prepared: PreparedTrainingContent,
) -> None:
    actor = _actor()
    options: dict[str, Any] = dict(
        schedule=make_training_schedule(total_env_steps=8, num_envs=2),
        prepared=prepared,
        metrics="none",
        keep_past=0,
        history_capture_capacity=0,
    )
    plain = init_training_collection(
        actor,
        actor.variables,
        reward=_plain_reward,
        reward_identity={"reference": "tests:plain"},
        **options,
    )
    assert plain[1].carry_steps == ()
    assert plain[0].reward_identity is not None
    assert "carry_steps_contract" not in plain[0].reward_identity
    settings: dict[str, Any] = dict(
        shaping=False,
        shaping_mode="potential",
        discount=float(plain[1].discount),
        coefficient=float(plain[1].coefficient),
    )
    started = collection_module.change_training_reward(
        *plain,
        reward=_steps_reward,
        reward_identity={"reference": "tests:steps"},
        **settings,
    )
    np.testing.assert_array_equal(started[1].carry_steps, np.zeros((2, 2)))
    stopped = collection_module.change_training_reward(
        *started,
        reward=_plain_reward,
        reward_identity={"reference": "tests:plain"},
        **settings,
    )
    assert stopped[1].carry_steps == ()

    collection, carry = _synthetic(
        init_training_collection(
            actor,
            actor.variables,
            reward=_steps_reward,
            reward_identity={"reference": "tests:steps"},
            **options,
        )
    )
    assert collection.reward_identity is not None
    assert collection.reward_identity["carry_steps_contract"] == CARRY_STEPS_CONTRACT
    assert isinstance(carry.carry_steps, Array)
    assert carry.carry_steps.dtype == jnp.int32
    np.testing.assert_array_equal(carry.carry_steps, np.zeros((2, 2)))
    # Place a Team A carrier in lane 0 and a Team B carrier in lane 1; a Team
    # Deathmatch step leaves the slots as they are until the two-tick game ends.
    core = carry.state.core_state._replace(
        capture_the_flag_flag_carrier_slots=jnp.asarray([[-1, 0], [5, -1]], jnp.int32)
    )
    carry = carry._replace(state=carry.state._replace(core_state=core))
    end, rollout = _scan((collection, carry), length=4)
    received = rollout.transitions.custom_rewards
    assert received is not None
    # Ticks 1 and 2 count the placed carries; the new games have no carrier.
    np.testing.assert_array_equal(
        received[:, :, :2],
        [[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
    )
    assert np.asarray(rollout.transitions.ended)[1].all()
    np.testing.assert_array_equal(end.carry_steps, np.zeros((2, 2)))
    assert jax.tree.structure(end) == jax.tree.structure(carry)
