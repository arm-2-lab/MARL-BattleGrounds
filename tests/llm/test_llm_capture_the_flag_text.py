"""Check the actor text of a Capture the Flag game.

Contracts: a Capture the Flag view names Capture the Flag's scoring in its
rules instead of Team Deathmatch's. When both flag rows are active, a Flags
line says in words whether each flag is home, carried (naming the seen carrier
by roster name, an enemy for your flag and an ally for theirs, or saying the
carrier is not seen) or dropped with its ticks until it goes home, read from
columns 4 to 11 (decision 139); each teammate source with active flag rows gets
its own flags line. The exact objective rows are still printed. Flag rows are
hand-placed on a Capture the Flag reset, so the check does not wait for Core
to fill them.
"""

import jax
import numpy as np
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import (
    OBJECTIVE_FEATURE_IS_ACTIVE,
    OBJECTIVE_FEATURE_IS_CONTROLLED_BY_ALLY_TEAM,
    OBJECTIVE_FEATURE_IS_CONTROLLED_BY_ENEMY_TEAM,
    OBJECTIVE_FEATURE_STEPS_UNTIL_DROPPED_FLAG_AUTO_RETURNS,
    OBJECTIVE_FEATURE_UNIT_ROW_0_IS_FLAG_CARRIER,
    ActionMask,
)
from marl_battlegrounds.llm.text import format_actor_view
from marl_battlegrounds.policies.input import ActorInput
from marl_battlegrounds.tasks import make_standard_capture_the_flag_config

_ROSTER = ("mage", "warrior", "priest")


def _actor() -> tuple[ActorInput, ActionMask]:
    config = make_standard_capture_the_flag_config(
        map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER
    )
    env = marl_bgs.make("ctf", env_config=config, metrics="none")
    observations, state = env.reset(jax.random.key(3))
    inputs = env.policy_inputs(observations, state)

    def first(value: Array) -> Array:
        return value[0][0]

    return jax.tree.map(first, inputs.actors), jax.tree.map(first, inputs.action_mask)


def _flags(
    rows: np.ndarray, own: tuple[int, int, int, int], their: tuple[int, int, int, int]
) -> np.ndarray:
    # own and their: (ally bit, enemy bit, carrier row or -1, ticks).
    rows = rows.copy()
    rows[..., :4, OBJECTIVE_FEATURE_IS_ACTIVE] = 1.0
    for row, (ally, enemy, carrier, ticks) in ((2, own), (3, their)):
        rows[..., row, OBJECTIVE_FEATURE_IS_CONTROLLED_BY_ALLY_TEAM] = ally
        rows[..., row, OBJECTIVE_FEATURE_IS_CONTROLLED_BY_ENEMY_TEAM] = enemy
        rows[..., row, OBJECTIVE_FEATURE_STEPS_UNTIL_DROPPED_FLAG_AUTO_RETURNS] = ticks
        if carrier >= 0:
            rows[..., row, OBJECTIVE_FEATURE_UNIT_ROW_0_IS_FLAG_CARRIER + carrier] = 1.0
    return rows


def _text(own: tuple[int, int, int, int], their: tuple[int, int, int, int]) -> str:
    actor, mask = _actor()
    observation = actor.observation._replace(
        objective_features=_flags(
            np.asarray(actor.observation.objective_features), own, their
        )
    )
    bank = actor.source_bank._replace(
        objective_features_by_source=_flags(
            np.asarray(actor.source_bank.objective_features_by_source), own, their
        )
    )
    return format_actor_view(
        ActorInput(observation, bank, actor.source_availability), mask
    )


def test_the_rules_name_capture_the_flag_scoring_and_the_flags_read_as_words() -> None:
    text = _text(own=(0, 1, 1, 0), their=(0, 0, -1, 12))
    assert "Scoring: In Capture the Flag" in text
    assert "In Team Deathmatch" not in text
    assert (
        "Flags: your flag carried by enemy_1; their flag dropped, home in 12 ticks"
        in text
    )
    assert "Objectives: 0:" in text
    lines = text.splitlines()
    sources = [
        line
        for line in lines
        if line.endswith(
            " flags: your flag carried by enemy_1; their flag dropped, home in 12 ticks"
        )
    ]
    assert len(sources) == 5


def test_home_flags_and_an_unseen_carrier() -> None:
    text = _text(own=(1, 0, -1, 0), their=(1, 0, -1, 0))
    assert "Flags: your flag home; their flag carried, carrier not seen" in text
    text = _text(own=(1, 0, -1, 0), their=(1, 0, 2, 0))
    assert "Flags: your flag home; their flag carried by ally_2" in text
