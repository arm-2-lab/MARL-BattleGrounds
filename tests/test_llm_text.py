"""Prove faithful actor text, exact numbers, source privacy and frame consistency.

All inputs come from development map 0 or explicitly synthetic field changes.
Tests retain every permitted field, distinguish hidden/absent history from zero,
and use the existing delivery and reflection helpers without a model/tokenizer.
Text version 2 contracts: every unit is named by its roster slot (never a
per-view ID), the actor is marked "(you)" in its Teams, unit, sight, source
and previous-action lines, equal rows of different units keep separate lines,
a view that reports a unit differently gets a "_via_" line and the main line is
the first visible row, names stay fixed when another unit disappears, the
Teams line marks dead, unused and unknown classes, the combat note names no
real choice, a self_ally_index outside 0-4 is refused, and the text reads back
exactly to the self row, own sight tables and masks, source rows, visibility,
permissions and objectives, and the six previous-action tables, including
synthetic edge cases.
"""

# pyright: reportPrivateUsage=false

import json
import re
from collections.abc import Iterator
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_X,
    ActionMask,
)
from marl_battlegrounds.environment import Environment, EnvironmentState
from marl_battlegrounds.llm import format_actor_view
from marl_battlegrounds.llm.actions import MOVE_NAMES, TARGET_NAMES
from marl_battlegrounds.llm.text import _UNIT_FIELDS, _number
from marl_battlegrounds.policies.input import (
    ActorInput,
    Observations,
    build_team_actor_input,
    mirror_team_view,
    team_on_right,
)
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


def _row[T](tree: T, index: int) -> T:
    def take(value: Array) -> Array:
        return value[index]

    return jax.tree.map(take, tree)


@pytest.fixture(scope="module")
def game() -> tuple[Environment, Observations, EnvironmentState]:
    config = make_standard_team_deathmatch_config(
        map_id=0,
        team_a_roster=("warrior", "mage", "hunter", "rogue", "priest"),
        team_b_roster=("warrior", "mage", "hunter", "rogue", "priest"),
        max_steps=20,
    )
    env = marl_bgs.make("tdm", env_config=config, metrics="none")
    observations, state = env.reset(jax.random.key(711))
    return env, observations, state


@pytest.fixture(scope="module")
def actor(
    game: tuple[Environment, Observations, EnvironmentState],
) -> tuple[ActorInput, ActionMask]:
    env, observations, state = game
    inputs = env.policy_inputs(observations, state)
    return _row(_row(inputs.actors, 0), 0), _row(_row(inputs.action_mask, 0), 0)


def test_float32_numbers_round_trip_bits_including_signed_zero() -> None:
    rng = np.random.default_rng(173)
    values = rng.integers(0, 2**32, 512, dtype=np.uint32).view(np.float32)
    edge = np.asarray(
        [
            0.0,
            -0.0,
            np.finfo(np.float32).max,
            np.finfo(np.float32).smallest_subnormal,
            9.292893,
        ],
        np.float32,
    )
    for value in np.concatenate((values[np.isfinite(values)], edge)):
        assert np.float32(_number(value)).tobytes() == value.tobytes()
    assert _number(np.float32(9.292893)) == "9.292893"
    assert _number(np.int32(2147483647)) == "2147483647"
    for value in (np.float32(np.nan), np.float32(np.inf), np.float64(1.5)):
        with pytest.raises(ValueError):
            _number(value)


def test_all_input_leaves_remain_detectable_without_mutation(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    host = jax.device_get(source)
    baseline = format_actor_view(host, masks, frame="world")
    leaves = jax.tree.leaves(host)
    before = [np.asarray(leaf).copy() for leaf in leaves]
    for index, leaf in enumerate(leaves):
        changed = np.asarray(leaf).copy()
        flat = changed.reshape(-1)
        if changed.dtype == np.bool_:
            flat[0] = not flat[0]
        elif changed.dtype.kind == "i":
            flat[0] += 1
        else:
            flat[0] = np.float32(123.125)
        inputs = list(leaves)
        inputs[index] = changed
        replacements = iter(inputs)

        def replace_leaf(
            value: Array, replacements: Iterator[Array] = replacements
        ) -> Array:
            del value
            return next(replacements)

        variant = jax.tree.map(replace_leaf, host)
        assert format_actor_view(variant, masks, frame="world") != baseline, index
    for original, actual in zip(before, jax.tree.leaves(host), strict=True):
        np.testing.assert_array_equal(original, actual)


def test_inactive_padding_and_reserved_nonzero_values_are_not_silently_lost(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    obs = source.observation
    changed = source._replace(
        observation=obs._replace(
            objective_features=obs.objective_features.at[7, 11].set(0.125),
            map_obstacle_features=obs.map_obstacle_features.at[31, 6].set(-0.0),
        )
    )
    text = format_actor_view(changed, masks)
    assert "Objectives: 7:" in text and "0.125]" in text
    assert "Obstacle 31:" in text and "theta=-0" in text


def test_absent_and_neutral_previous_actions_are_different(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    before = format_actor_view(source, masks)
    assert "ally_0 (you): Not seen" in before
    old = source.observation.previous_timestep_actions
    observed = old._replace(
        ally_previous_timestep_move_actions_one_hot=old[0].at[0, 0].set(1),
        ally_previous_timestep_select_target_actions_one_hot=old[2].at[0, 0].set(1),
        ally_previous_timestep_use_ultimate_actions_one_hot=old[4].at[0, 0].set(1),
    )
    after = format_actor_view(
        source._replace(
            observation=source.observation._replace(previous_timestep_actions=observed)
        ),
        masks,
    )
    assert "ally_0 (you): move=stay target=target_none ultimate=No" in after
    assert before != after


def test_previous_enemy_targets_keep_observer_names_when_target_is_hidden(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    old = source.observation.previous_timestep_actions
    observed = old._replace(
        enemy_previous_timestep_move_actions_one_hot=old[1].at[0, 0].set(1),
        enemy_previous_timestep_select_target_actions_one_hot=old[3].at[0, 8].set(1),
        enemy_previous_timestep_use_ultimate_actions_one_hot=old[5].at[0, 1].set(1),
    )
    obs = source.observation._replace(
        previous_timestep_actions=observed,
        enemy_visibility_mask=source.observation.enemy_visibility_mask.at[2].set(False),
    )
    text = format_actor_view(source._replace(observation=obs), masks)
    assert "enemy_0: move=stay target=enemy_2 ultimate=Yes" in text
    assert "enemy_2 Hidden" in text


def test_source_permissions_and_visibility_are_separate(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor

    def clear(value: Array) -> Array:
        return jnp.zeros_like(value)

    zeros = jax.tree.map(clear, source.source_bank)
    no_sources = source._replace(
        source_bank=zeros, source_availability=jnp.zeros(5, bool)
    )
    permission = no_sources._replace(
        source_availability=jnp.asarray([False, True, False, False, False])
    )
    a, b = format_actor_view(no_sources, masks), format_actor_view(permission, masks)
    assert "Source ally_1: Unavailable; No visible rows" in a
    assert "Source ally_1: Allowed; No visible rows" in b
    visible_zero = permission._replace(
        source_bank=zeros._replace(
            unit_visibility_by_source_and_candidate=zeros.unit_visibility_by_source_and_candidate.at[
                1, 7
            ].set(True)
        )
    )
    assert "Source ally_1: Allowed; sees enemy_2 (Zero)" in format_actor_view(
        visible_zero, masks
    )


def test_forbidden_source_material_cannot_change_delivered_text(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = game
    restricted = observations._replace(
        source_availability=jnp.zeros_like(observations.source_availability)
    )
    original = env.policy_inputs(restricted, state)

    def poison(value: Array) -> Array:
        selector = (jnp.arange(10) != 0).reshape((10,) + (1,) * (value.ndim - 1))
        return jnp.where(selector, jnp.full_like(value, 73), value)

    poisoned = restricted._replace(
        observation=jax.tree.map(poison, restricted.observation)
    )
    changed = env.policy_inputs(poisoned, state)
    masks = _row(_row(original.action_mask, 0), 0)
    assert format_actor_view(
        _row(_row(original.actors, 0), 0), masks
    ) == format_actor_view(_row(_row(changed.actors, 0), 0), masks)


def test_team_labels_on_same_physical_side_do_not_change_text(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = game
    permutation = jnp.concatenate((jnp.arange(5, 10), jnp.arange(5)))

    def reorder(value: Array) -> Array:
        return value[permutation]

    swapped = Observations(
        jax.tree.map(reorder, observations.observation),
        observations.source_availability[permutation][:, permutation],
    )
    own = _row(build_team_actor_input(observations, 0), 0)
    relabelled = _row(build_team_actor_input(swapped, 1), 0)
    masks = _row(_row(env.policy_inputs(observations, state).action_mask, 0), 0)
    assert format_actor_view(own, masks, frame="world") == format_actor_view(
        relabelled, masks, frame="world"
    )


def test_frame_label_does_not_transform_coordinates(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    world = format_actor_view(source, masks, frame="world")
    left = format_actor_view(source, masks, frame="left")
    assert world.splitlines()[1:] == left.splitlines()[1:]
    with pytest.raises(ValueError, match="Frame"):
        format_actor_view(source, masks, frame=cast(Literal["left", "world"], "other"))


def test_existing_mirror_handles_asymmetric_map_before_formatting(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    obstacle = (
        jnp.zeros((32, 8), jnp.float32)
        .at[0]
        .set(jnp.asarray([2, 4, 3, 0, 2, 1, 0.25, 1], jnp.float32))
    )
    source = source._replace(
        observation=source.observation._replace(map_obstacle_features=obstacle)
    )
    mirrored, mask = mirror_team_view(source, masks, jnp.asarray(True))
    text = format_actor_view(mirrored, mask)
    assert "Obstacle 0:" in text and "theta=-0.25" in text
    width = source.observation.context_features[2]
    assert f"x={_number(np.float32(width - 4))}" in text
    assert float(mirrored.observation.self_features[AGENT_FEATURE_X]) != float(
        source.observation.self_features[AGENT_FEATURE_X]
    )
    assert bool(team_on_right(source)) == bool(
        team_on_right(
            source._replace(
                observation=source.observation._replace(
                    self_features=source.observation.self_features.at[0].set(100)
                )
            )
        )
    )


def test_rejects_batched_or_nonfinite_inputs(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor

    def batch(value: Array) -> Array:
        return value[None]

    with pytest.raises(ValueError, match="shape"):
        format_actor_view(jax.tree.map(batch, source), masks)
    bad = source._replace(
        observation=source.observation._replace(
            self_features=source.observation.self_features.at[0].set(jnp.nan)
        )
    )
    with pytest.raises(ValueError, match="finite"):
        format_actor_view(bad, masks)
    for index in (-1, 5):
        outside = source._replace(
            observation=source.observation._replace(
                self_ally_index=jnp.asarray(index, jnp.int32)
            )
        )
        with pytest.raises(ValueError, match="self_ally_index"):
            format_actor_view(outside, masks)


def test_every_unit_context_and_obstacle_column_is_preserved(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = jax.device_get(actor)
    baseline = format_actor_view(source, masks)
    for field, width in (
        ("self_features", 58),
        ("context_features", 20),
        ("map_obstacle_features", 8),
    ):
        for column in range(width):
            values = np.asarray(getattr(source.observation, field)).copy()
            if values.ndim == 2:
                values[0, column] = np.float32(123.125)
            else:
                values[column] = np.float32(123.125)
            changed = source._replace(
                observation=source.observation._replace(**{field: cast(Array, values)})
            )
            text = format_actor_view(changed, masks)
            assert text != baseline and "123.125" in text, (field, column)


def test_each_unit_has_its_own_line_and_differing_views_are_named(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    own = source.observation.self_features
    observation = source.observation._replace(
        ally_unit_features=jnp.broadcast_to(own, (5, 58)),
        enemy_unit_features=jnp.zeros((5, 58), jnp.float32),
        ally_visibility_mask=jnp.ones(5, bool),
        enemy_visibility_mask=jnp.zeros(5, bool),
    )
    rows = jnp.zeros((5, 10, 58), jnp.float32).at[:, 0].set(own)
    bank = source.source_bank._replace(
        unit_features_by_source_and_candidate=rows,
        unit_visibility_by_source_and_candidate=jnp.zeros((5, 10), bool)
        .at[:, 0]
        .set(True),
    )
    view = source._replace(
        observation=observation, source_bank=bank, source_availability=jnp.ones(5, bool)
    )
    text = format_actor_view(view, masks)
    lines = text.splitlines()
    fields = next(line for line in lines if line.startswith("ally_0 (you): "))[14:]
    for index in range(1, 5):
        # Equal rows of different units are never merged into one line.
        assert f"ally_{index}: {fields}" in lines
    assert "_via_" not in text
    assert "Source ally_0 (you): Allowed; sees ally_0 (you)" in lines
    for index in range(1, 5):
        assert f"Source ally_{index}: Allowed; sees ally_0 (you)" in lines
    different = view._replace(
        source_bank=bank._replace(
            unit_features_by_source_and_candidate=rows.at[2, 0, 0].set(
                jnp.nextafter(own[0], jnp.inf)
            )
        )
    )
    changed = format_actor_view(different, masks).splitlines()
    assert sum(line.startswith("ally_0_via_ally_2: ") for line in changed) == 1
    assert "Source ally_2: Allowed; sees ally_0 (you) (ally_0_via_ally_2)" in changed
    assert any(
        line.startswith("A line named like enemy_2_via_ally_3") for line in changed
    )
    _assert_reads_back(different, masks, "differing teammate row")


def test_every_slot_is_named_and_marked_as_you(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = game
    inputs = env.policy_inputs(observations, state)
    classes = ("Warrior", "Mage", "Hunter", "Rogue", "Priest")
    for slot in range(5):
        text = format_actor_view(
            _row(_row(inputs.actors, 0), slot), _row(_row(inputs.action_mask, 0), slot)
        )
        lines = text.splitlines()
        assert f"You are ally_{slot}, the {classes[slot]}." in lines
        teams = next(line for line in lines if line.startswith("Teams: "))
        assert teams.count("(you)") == 1
        assert f"ally_{slot} {classes[slot]} (you)" in teams
        assert any(line.startswith(f"ally_{slot} (you): x=") for line in lines)
        assert f"ally_{slot} (you) Visible" in text
        assert (
            f"Source ally_{slot} (you): Unavailable; your own sight is listed above"
            in lines
        )
        assert f"ally_{slot} (you): Not seen" in lines
        assert re.search(r"\bunit_\d", text) is None


def test_names_stay_the_same_when_another_unit_disappears(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = jax.device_get(actor)
    obs, bank = source.observation, source.source_bank
    gone = source._replace(
        observation=obs._replace(
            ally_unit_features=_with(obs.ally_unit_features, 1, 0),
            ally_visibility_mask=_with(obs.ally_visibility_mask, 1, False),
        ),
        source_bank=bank._replace(
            unit_features_by_source_and_candidate=_with(
                bank.unit_features_by_source_and_candidate, (slice(None), 1), 0
            ),
            unit_visibility_by_source_and_candidate=_with(
                bank.unit_visibility_by_source_and_candidate, (slice(None), 1), False
            ),
        ),
    )
    views = [
        format_actor_view(view, masks, include_static=False, reply_instruction=None)
        for view in (source, gone)
    ]

    def line(text: str, name: str) -> str:
        return next(row for row in text.splitlines() if row.startswith(f"{name}: x="))

    # v1 renumbered its per-view unit_N IDs here; roster names do not move.
    assert line(views[0], "ally_2") == line(views[1], "ally_2")
    assert "ally_1 Hidden" in views[1]
    assert not any(row.startswith("ally_1: x=") for row in views[1].splitlines())


def test_teams_line_marks_dead_unused_and_unknown_classes(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = jax.device_get(actor)
    life = source.observation.spawn_lifecycle
    changed = life._replace(
        alive_mask_by_agent_by_team=_with(
            life.alive_mask_by_agent_by_team, (0, 2), False
        ),
        active_mask_by_agent_by_team=_with(
            life.active_mask_by_agent_by_team, (1, 4), False
        ),
        class_ids_by_agent_by_team=_with(life.class_ids_by_agent_by_team, (0, 0), 9),
    )
    text = format_actor_view(
        source._replace(
            observation=source.observation._replace(spawn_lifecycle=changed)
        ),
        masks,
    )
    assert "You are ally_0." in text.splitlines()
    assert "Teams: ally_0 class_id 9 (you), ally_1 Mage, ally_2 Hunter (dead)," in text
    assert "enemy_4 unused" in text


def test_combat_note_names_no_real_choice(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor

    def note(pairs: list[tuple[int, int]]) -> str:
        joint = np.zeros((11, 2), bool)
        for target, ultimate in pairs:
            joint[target, ultimate] = True
        changed = masks._replace(
            select_target_use_ultimate_joint_mask=jnp.asarray(joint),
            select_target_mask=jnp.asarray(joint.any(axis=1)),
            use_ultimate_mask=jnp.asarray(joint.any(axis=0)),
        )
        text = format_actor_view(source, changed)
        return next(line for line in text.splitlines() if line.startswith("Combat: "))

    notes = {
        note(pairs) for pairs in ([(0, 0)], [(0, 0), (1, 0), (8, 1)], [(0, 0), (8, 0)])
    }
    # One fixed pattern for every menu: it cannot anchor a target or a masked pair.
    assert len(notes) == 1
    (line,) = notes
    assert "enemy_k_basic would use your Basic on enemy_k" in line
    assert "ally_k_ultimate your Ultimate on ally_k" in line
    assert re.search(r"\b(?:ally|enemy)_\d_(?:basic|ultimate)\b", line) is None


def test_text_reads_back_to_every_input_array_it_names(
    actor: tuple[ActorInput, ActionMask],
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    source, masks = jax.device_get(actor)
    env, observations, state = game
    inputs = jax.device_get(env.policy_inputs(observations, state))
    obs, bank = source.observation, source.source_bank
    own = np.asarray(obs.self_features)
    shifted = own.copy()
    shifted[0] += np.float32(0.25)
    other = own.copy()
    other[1] += np.float32(0.5)
    previous = list(obs.previous_timestep_actions)
    previous[0] = _with(previous[0], (2, 3), 0.5)
    previous[0] = _with(previous[0], (2, 0), -0.0)
    previous[5] = _with(previous[5], (1, 1), 1)
    cases: dict[str, ActorInput] = {
        "real slot 4": _row(_row(inputs.actors, 0), 4),
        "own table row of self differs": source._replace(
            observation=obs._replace(
                ally_unit_features=_with(obs.ally_unit_features, 0, shifted)
            )
        ),
        "visible all-zero rows": source._replace(
            observation=obs._replace(
                enemy_unit_features=_with(obs.enemy_unit_features, 3, 0),
                enemy_visibility_mask=_with(obs.enemy_visibility_mask, 3, True),
            ),
            source_bank=bank._replace(
                unit_features_by_source_and_candidate=_with(
                    bank.unit_features_by_source_and_candidate, (1, 8), 0
                ),
                unit_visibility_by_source_and_candidate=_with(
                    bank.unit_visibility_by_source_and_candidate, (1, 8), True
                ),
            ),
        ),
        "hidden row with data": source._replace(
            observation=obs._replace(
                enemy_unit_features=_with(obs.enemy_unit_features, 1, shifted),
                enemy_visibility_mask=_with(obs.enemy_visibility_mask, 1, False),
            )
        ),
        "hidden own row, visible teammate row": source._replace(
            observation=obs._replace(
                ally_unit_features=_with(obs.ally_unit_features, 1, shifted),
                ally_visibility_mask=_with(obs.ally_visibility_mask, 1, False),
            ),
            source_bank=bank._replace(
                unit_features_by_source_and_candidate=_with(
                    bank.unit_features_by_source_and_candidate, (2, 1), other
                ),
                unit_visibility_by_source_and_candidate=_with(
                    bank.unit_visibility_by_source_and_candidate, (2, 1), True
                ),
            ),
        ),
        "own source allowed with data": source._replace(
            source_availability=_with(source.source_availability, 0, True),
            source_bank=bank._replace(
                unit_features_by_source_and_candidate=_with(
                    bank.unit_features_by_source_and_candidate, (0, 5), shifted
                ),
                unit_visibility_by_source_and_candidate=_with(
                    bank.unit_visibility_by_source_and_candidate, (0, 5), True
                ),
                objective_features_by_source=_with(
                    bank.objective_features_by_source, (0, 2, 4), 0.125
                ),
            ),
        ),
        "two units with equal rows": source._replace(
            observation=obs._replace(
                ally_unit_features=_with(
                    _with(obs.ally_unit_features, 1, shifted), 2, shifted
                ),
                ally_visibility_mask=_with(
                    _with(obs.ally_visibility_mask, 1, True), 2, True
                ),
            )
        ),
        "own source unavailable with a hidden row": source._replace(
            source_availability=_with(source.source_availability, 0, False),
            source_bank=bank._replace(
                unit_features_by_source_and_candidate=_with(
                    bank.unit_features_by_source_and_candidate, (0, 6), shifted
                ),
                unit_visibility_by_source_and_candidate=_with(
                    bank.unit_visibility_by_source_and_candidate, (0, 6), False
                ),
            ),
        ),
        "teammate hidden rows, differing and equal": source._replace(
            source_bank=bank._replace(
                unit_features_by_source_and_candidate=_with(
                    _with(bank.unit_features_by_source_and_candidate, (3, 1), other),
                    (1, 1),
                    np.asarray(obs.ally_unit_features)[1],
                ),
                unit_visibility_by_source_and_candidate=_with(
                    _with(bank.unit_visibility_by_source_and_candidate, (3, 1), False),
                    (1, 1),
                    False,
                ),
            ),
        ),
        "own source unavailable with a visible zero row": source._replace(
            source_availability=_with(source.source_availability, 0, False),
            source_bank=bank._replace(
                unit_features_by_source_and_candidate=_with(
                    bank.unit_features_by_source_and_candidate, (0, 8), 0
                ),
                unit_visibility_by_source_and_candidate=_with(
                    bank.unit_visibility_by_source_and_candidate, (0, 8), True
                ),
            ),
        ),
        "self row all zero": source._replace(
            observation=obs._replace(self_features=np.zeros_like(own))
        ),
        "unusual previous actions": source._replace(
            observation=obs._replace(
                previous_timestep_actions=type(obs.previous_timestep_actions)(*previous)
            )
        ),
    }
    _assert_reads_back(source, masks, "real slot 0")
    slot_masks = _row(_row(inputs.action_mask, 0), 4)
    for name, view in cases.items():
        _assert_reads_back(view, slot_masks if name == "real slot 4" else masks, name)
    # The main line is the first visible nonzero row: here the teammate's view.
    text = format_actor_view(cases["hidden own row, visible teammate row"], masks)
    fields = next(row for row in text.splitlines() if row.startswith("ally_1: "))
    assert fields.split()[1] == f"x={_number(np.float32(other[0]))}"
    assert "ally_1 Hidden, has data ally_1_via_your_sight" in text
    lines = format_actor_view(cases["own source unavailable with a hidden row"], masks)
    assert "Source ally_0 (you): Unavailable; Hidden, has data" in lines
    hidden = format_actor_view(
        cases["teammate hidden rows, differing and equal"], masks
    )
    assert "Hidden, has data ally_1_via_ally_3" in hidden


def _with(array: object, index: object, value: object) -> Array:
    changed = np.array(array, copy=True)
    changed[index] = value  # pyright: ignore[reportArgumentType, reportCallIssue]
    return cast(Array, changed)


_LABEL = r"(?:ally|enemy)_\d"
_UNIT_LINE = re.compile(rf"^({_LABEL}(?:_via_\w+)?)(?: \(you\))?: (.*)$")
_OWN_ENTRY = re.compile(
    rf"^({_LABEL})(?: \(you\))? (Visible|Hidden)(?: \((\w+)\))?(?:, has data (\w+))?$"
)
_SEEN_ENTRY = re.compile(rf"^({_LABEL})(?: \(you\))?(?: \((\w+)\))?$")
_SOURCE_LINE = re.compile(r"^Source ally_(\d)(?: \(you\))?( objectives)?: (.*)$")
_PREVIOUS_LINE = re.compile(rf"^({_LABEL})(?: \(you\))?: (.*)$")
_CHOICES = re.compile(r"^move=(.*) target=(.*) ultimate=(.*)$")


def _unit_index(label: str) -> int:
    relation, slot = label.split("_")[:2]
    return int(slot) + (5 if relation == "enemy" else 0)


def _floats(text: str) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.asarray(
        json.loads(text, parse_float=np.float32, parse_int=np.float32), np.float32
    )


def _read_back(text: str) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
    lines = text.splitlines()
    you = int(next(line for line in lines if line.startswith("You are ally_"))[13])
    rows: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    start = next(i for i, line in enumerate(lines) if line.startswith("Units ("))
    for line in lines[start + 1 :]:
        match = _UNIT_LINE.match(line)
        if match is None:
            break
        row = np.zeros(58, np.float32)
        for pair in match.group(2).split():
            name, value = pair.split("=")
            row[_UNIT_FIELDS.index(name)] = np.float32(value)
        rows[match.group(1)] = row
    own_rows = np.zeros((10, 58), np.float32)
    own_seen = np.zeros(10, bool)
    sight = next(line for line in lines if line.startswith("Your own sight: "))
    for item in sight[len("Your own sight: ") :].split("; "):
        match = _OWN_ENTRY.match(item)
        assert match is not None, item
        unit = _unit_index(match.group(1))
        own_seen[unit] = match.group(2) == "Visible"
        if match.group(4):
            own_rows[unit] = rows[match.group(4)]
        elif own_seen[unit] and match.group(3) != "Zero":
            own_rows[unit] = rows[match.group(3) or match.group(1)]
    source_rows = np.zeros((5, 10, 58), np.float32)
    source_seen = np.zeros((5, 10), bool)
    allowed = np.zeros(5, bool)
    objectives = np.zeros((5, 8, 12), np.float32)
    for line in lines:
        match = _SOURCE_LINE.match(line)
        if match is None:
            continue
        source, body = int(match.group(1)), match.group(3)
        if match.group(2):
            for part in body.split("; ") if body != "Zero" else []:
                index, values = part.split(":", 1)
                objectives[source, int(index)] = _floats(values)
            continue
        parts = body.split("; ")
        allowed[source] = parts[0] == "Allowed"
        for part in parts[1:]:
            if part.startswith("sees "):
                for item in part[5:].split(", "):
                    seen = _SEEN_ENTRY.match(item)
                    assert seen is not None, item
                    unit = _unit_index(seen.group(1))
                    source_seen[source, unit] = True
                    if seen.group(2) != "Zero":
                        source_rows[source, unit] = rows[seen.group(2) or seen.group(1)]
            elif part.startswith("Hidden, has data "):
                for label in part[len("Hidden, has data ") :].split(", "):
                    source_rows[source, _unit_index(label)] = rows[label]
    names = (MOVE_NAMES, TARGET_NAMES, ("No", "Yes"))
    previous = [np.zeros((5, size), np.float32) for size in (9, 9, 11, 11, 2, 2)]
    start = lines.index(
        "Previous accepted actions (targets already use your roster names):"
    )
    for line in lines[start + 1 : start + 11]:
        match = _PREVIOUS_LINE.match(line)
        assert match is not None, line
        unit = _unit_index(match.group(1))
        if match.group(2) == "Not seen":
            continue
        choices = _CHOICES.match(match.group(2))
        assert choices is not None, line
        for kind, value in enumerate(choices.groups()):
            table = previous[2 * kind + unit // 5]
            if value.startswith("Values"):
                table[unit % 5] = _floats(value[len("Values") :])
            elif value != "Not seen":
                table[unit % 5, names[kind].index(value)] = 1
    self_row = (
        np.zeros(58, np.float32)
        if "Your own row is all zero." in lines
        else rows[f"ally_{you}"]
    )
    return {
        "self_ally_index": np.asarray(you, np.int32),
        "self_features": self_row,
        "own_rows": own_rows,
        "own_seen": own_seen,
        "source_rows": source_rows,
        "source_seen": source_seen,
        "source_availability": allowed,
        "source_objectives": objectives,
        **{f"previous_{index}": table for index, table in enumerate(previous)},
    }


def _assert_reads_back(view: ActorInput, masks: ActionMask, case: str) -> None:
    host = jax.device_get(view)
    obs, bank = host.observation, host.source_bank
    back = _read_back(format_actor_view(host, masks))
    expected = {
        "self_ally_index": obs.self_ally_index,
        "self_features": obs.self_features,
        "own_rows": np.concatenate((obs.ally_unit_features, obs.enemy_unit_features)),
        "own_seen": np.concatenate(
            (obs.ally_visibility_mask, obs.enemy_visibility_mask)
        ),
        "source_rows": bank.unit_features_by_source_and_candidate,
        "source_seen": bank.unit_visibility_by_source_and_candidate,
        "source_availability": host.source_availability,
        "source_objectives": bank.objective_features_by_source,
        **{
            f"previous_{index}": table
            for index, table in enumerate(obs.previous_timestep_actions)
        },
    }
    for key, value in expected.items():
        array = np.asarray(value)
        assert array.dtype == back[key].dtype, (case, key)
        assert array.tobytes() == back[key].tobytes(), (case, key)
