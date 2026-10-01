"""Check the public combat/support recipes against independent raw-fact arithmetic.

Scenes use public reset/step and ordinary actions. The oracle uses NumPy and the
historical fixed coefficients, not the new combat helper. It checks effective
healing, overkill caps, one-hop Priest credit, deaths and successor range/zone
ownership. Every scene also checks the team form: each configured Team A slot
gets the three shared parts once plus the Team A total of the five own-agent
parts, and the QMIX/PQN-VDN team average of that form keeps each event's
per-agent price (+0.05 kill credit, -0.05 teammate death, own-zone ticks per
player). Both recipes' saved identities bind the shared combat helper.
These authored transitions prove software meaning, not learning.
"""

# Tests inspect the compact numerical contract without exposing it to actors.
# pyright: reportPrivateUsage=false
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config, neutral_action

from marl_battlegrounds.baselines.qmix import team_task_reward
from marl_battlegrounds.core.axis_mappings import global_slot_to_target_action
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import initialize_scenario_state
from marl_battlegrounds.core.types import Action, EnvConfig, EnvState
from marl_battlegrounds.environment import (
    Environment,
    EnvironmentState,
    TrainingFacts,
    make,
)
from marl_battlegrounds.training import (
    RewardOutput,
    combat_support_reward,
    combat_support_team_reward,
)
from marl_battlegrounds.training.shaping import reward_adjustments


def _start(
    *,
    missing_health: float = 40,
    depth: float = 0.0,
    arrange: Callable[[EnvState], EnvState] | None = None,
    team_sizes: tuple[int, int] = (5, 5),
    width: float = 20.0,
    mirrored: bool = False,
) -> tuple[Environment, EnvironmentState]:
    config = evaluation_env_config(
        team_sizes=team_sizes,
        task_mode=1,
        team_deathmatch_score_threshold=20,
        max_steps=4,
        team_deathmatch_red_zone_depth=depth,
    )
    config = config._replace(map_width=width)
    if mirrored:
        config = config._replace(
            team_spawn_pad_positions=config.team_spawn_pad_positions[::-1]
        )
    classes = config.agent_profile.class_ids.at[3].set(5)
    config = config._replace(
        agent_profile=resolve_agent_profile(classes, jnp.asarray(team_sizes, jnp.int32))
    )
    env = make("tdm", metrics="none", training_facts=True)
    _, initial = env.reset(jax.random.key(0), config)
    positions = jnp.asarray(
        [(6.0, 1.5 + 2 * i) for i in range(5)] + [(9.0, 1.5 + 2 * i) for i in range(5)],
        jnp.float32,
    )
    positions = positions.at[2].set((6.0, 2.5)).at[3].set((7.0, 2.5))
    core = initial.core_state._replace(
        agent_positions=jnp.where(
            config.agent_profile.active_mask[:, None], positions, 0
        ),
        spawn_shield_durations=jnp.zeros(10, jnp.int32),
        current_health=initial.core_state.current_health.at[0]
        .set(80 - missing_health)
        .at[5]
        .set(1),
    )
    if arrange is not None:
        core = arrange(core)
    ready = initialize_scenario_state(core, config)
    _, before = env.reset(jax.random.key(1), config, initial=ready[:3])
    return env, before


def _actions(*choices: tuple[int, int]) -> Action:
    action = neutral_action()
    for source, recipient in choices:
        action = action._replace(
            select_target=action.select_target.at[source].set(
                global_slot_to_target_action(source, recipient)
            )
        )
    return action


def _oracle(
    config: EnvConfig, before: EnvState, facts: TrainingFacts, after: EnvState
) -> dict[str, np.ndarray[Any, Any]]:
    cfg, old, raw, new = cast(
        tuple[Any, Any, Any, Any], jax.device_get((config, before, facts, after))
    )
    routing = np.zeros((10, 10), np.float32)
    for source in range(10):
        if raw.combat_effect_has_recipient_by_source[source]:
            routing[
                source, int(raw.combat_effect_recipient_global_slot_by_source[source])
            ] = 1
    damage = (
        routing
        * (
            raw.source_modified_damage_output_by_source
            * raw.recipient_damage_modifier_by_source
        )[:, None]
    )
    healing = (
        routing
        * (
            raw.source_modified_healing_output_by_source
            * raw.recipient_healing_modifier_by_source
        )[:, None]
    )
    excess = np.clip(
        old.current_health
        + (
            raw.total_effective_healing_by_recipient
            - raw.total_effective_damage_by_recipient
        )
        - raw.health_after_combat_resolution_by_recipient,
        0,
        raw.total_effective_healing_by_recipient,
    )
    effective = np.maximum(
        healing
        - healing
        * (
            excess
            / np.where(
                raw.total_effective_healing_by_recipient > 0,
                raw.total_effective_healing_by_recipient,
                1,
            )
        )[None, :],
        0,
    )
    direct = routing.astype(bool) & raw.contributed_to_new_death_by_source[:, None]
    useful = (
        (healing > 0)
        & (raw.total_effective_healing_by_recipient > excess)[None, :]
        & (cfg.agent_profile.class_ids == 5)[:, None]
    )
    credit = direct.copy()
    for priest in range(10):
        for supported in range(10):
            if useful[priest, supported]:
                credit[priest] |= direct[supported]
    active = cfg.agent_profile.active_mask
    dead = old.alive_mask & ~new.alive_mask & active
    depth = float(cfg.team_deathmatch_red_zone_depth)
    right = np.repeat(
        cfg.team_spawn_pad_positions[:, :, 0].mean(1) > float(cfg.map_width) / 2, 5
    )
    old_zone = cast(
        np.ndarray[Any, Any],
        np.where(
            right,
            old.agent_positions[:, 0] >= float(cfg.map_width) - depth,
            old.agent_positions[:, 0] <= depth,
        )
        & (depth > 0),
    )
    zone = cast(
        np.ndarray[Any, Any],
        (
            np.where(
                right,
                new.agent_positions[:, 0] >= float(cfg.map_width) - depth,
                new.agent_positions[:, 0] <= depth,
            )
            & (depth > 0)
            & new.alive_mask
            & active
        ),
    )
    credited = credit[:5, 5:] & dead[None, 5:]
    limit = old.current_health[5:] + effective[:, 5:].sum(0)
    cap = np.minimum(1, limit / np.maximum(damage[:, 5:].sum(0), 1e-6))
    pairs = 0
    for priest in range(5):
        for ally in range(5):
            if (
                priest != ally
                and cfg.agent_profile.class_ids[priest] == 5
                and active[priest]
                and active[ally]
                and new.alive_mask[priest]
                and new.alive_mask[ally]
                and not zone[priest]
                and not zone[ally]
            ):
                pairs += int(
                    np.linalg.norm(
                        new.agent_positions[priest] - new.agent_positions[ally]
                    )
                    <= cfg.agent_profile.basic_interaction_radii[priest]
                )
    parts: dict[str, Any] = {
        "enemy_death_credit": 0.05 * credited.sum(1),
        "own_team_deaths": np.full(5, -0.05 * dead[:5].sum()),
        "enemy_red_zone_death_credit": 0.10 * (credited & old_zone[None, 5:]).sum(1),
        "own_red_zone_deaths": np.full(5, -0.10 * (dead[:5] & old_zone[:5]).sum()),
        "enemy_damage_hp": 0.001 * (damage[:5, 5:] * cap[None, :]).sum(1),
        "priest_effective_healing_hp": 0.001
        * effective[:5, :5].sum(1)
        * (cfg.agent_profile.class_ids[:5] == 5)
        * active[:5],
        "priest_heal_range_pairs": np.full(5, 0.001 * pairs),
        "own_zone_ticks": -0.002 * zone[:5],
    }
    return {
        name: np.concatenate(
            (np.where(active[:5] & raw.has_transition, value, 0), np.zeros(5))
        ).astype(np.float32)
        for name, value in parts.items()
    }


def _check(
    env: Environment, before: EnvironmentState, action: Action
) -> tuple[EnvironmentState, TrainingFacts, RewardOutput]:
    _, after, _, _, info = env.step(jax.random.key(2), before, action)
    assert info.training_facts is not None
    facts = info.training_facts
    actual = combat_support_reward(
        before.core_state, facts, after.core_state, jnp.int32(0), config=before.config
    )
    expected = _oracle(before.config, before.core_state, facts, after.core_state)
    assert actual.components.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(
            actual.components[name], expected[name], rtol=1e-5, atol=1e-7
        )
    np.testing.assert_allclose(
        actual.adjustments,
        np.stack(tuple(expected.values())).sum(0),
        rtol=1e-5,
        atol=1e-7,
    )
    team = combat_support_team_reward(
        before.core_state, facts, after.core_state, jnp.int32(0), config=before.config
    )
    recipients = np.asarray(before.config.agent_profile.active_mask[:5]) & bool(
        facts.has_transition
    )
    shared = ("own_team_deaths", "own_red_zone_deaths", "priest_heal_range_pairs")
    assert team.components.keys() == expected.keys()
    for name, value in expected.items():
        wanted = value.copy()
        if name not in shared:
            wanted[:5] = np.where(recipients, value[:5].sum(), 0)
        np.testing.assert_allclose(team.components[name], wanted, rtol=1e-5, atol=1e-7)
    per_slot = sum(expected[name][:5] for name in shared) + np.where(
        recipients,
        sum(expected[name][:5].sum() for name in expected if name not in shared),
        0,
    )
    np.testing.assert_allclose(team.adjustments[:5], per_slot, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(team.adjustments[5:], 0)
    assert info.priority is None
    return after, facts, actual


@pytest.mark.parametrize("missing_health", [0.0, 0.5, 16.0, 40.0])
def test_public_step_matches_independent_overkill_and_two_priest_oracle(
    missing_health: float,
) -> None:
    env, before = _start(missing_health=missing_health)
    after, facts, actual = _check(env, before, _actions((0, 5), (2, 0), (3, 0)))
    assert not after.core_state.alive_mask[5]
    assert float(actual.components["enemy_damage_hp"][0]) == pytest.approx(0.001)
    assert bool(actual.components["enemy_death_credit"][2]) == (missing_health > 0)
    assert float(actual.components["priest_effective_healing_hp"][2]) == pytest.approx(
        0.001 * min(missing_health, 16) / 2
    )
    compiled = cast(
        Callable[..., Array], jax.jit(reward_adjustments, static_argnums=0)
    )(
        combat_support_reward,
        before.core_state,
        facts,
        after.core_state,
        jnp.int32(0),
        config=before.config,
    )
    np.testing.assert_allclose(compiled, actual.adjustments, atol=1e-7)


@pytest.mark.parametrize(
    "mode", ["enemy_heal", "dying_priest", "nonrecursive", "zone", "inactive"]
)
def test_public_step_covers_credit_cap_zone_and_inactive_edges(mode: str) -> None:
    def arrange(state: EnvState) -> EnvState:
        if mode == "enemy_heal":
            return state._replace(
                agent_positions=state.agent_positions.at[9].set((10.0, 2.5))
            )
        if mode == "dying_priest":
            return state._replace(
                agent_positions=state.agent_positions.at[3]
                .set((7.0, 5.5))
                .at[6]
                .set((7.0, 2.5)),
                current_health=state.current_health.at[2].set(1),
            )
        if mode == "nonrecursive":
            return state._replace(current_health=state.current_health.at[2].set(50))
        return state

    env, before = _start(
        arrange=arrange,
        depth=7.0 if mode == "zone" else 0.0,
        team_sizes=(3, 2) if mode == "inactive" else (5, 5),
    )
    choices = [(0, 5), (2, 0)]
    if mode == "enemy_heal":
        choices.append((9, 5))
    elif mode == "dying_priest":
        choices.append((6, 2))
    elif mode == "nonrecursive":
        choices.append((3, 2))
    after, facts, actual = _check(env, before, _actions(*choices))
    if mode == "enemy_heal":
        assert float(facts.total_effective_healing_by_recipient[5]) > 0
        assert float(actual.components["enemy_damage_hp"][0]) > 0.001
    if mode == "dying_priest":
        assert not after.core_state.alive_mask[2]
        assert actual.components["enemy_death_credit"][2] > 0
    if mode == "nonrecursive":
        assert actual.components["enemy_death_credit"][3] == 0
    if mode == "inactive":
        np.testing.assert_array_equal(actual.adjustments[3:5], 0)


@pytest.mark.parametrize("inside_before", [False, True])
def test_red_zone_credit_uses_pre_action_position(inside_before: bool) -> None:
    from marl_battlegrounds.core.types import MOVE_EAST, MOVE_WEST

    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[0]
            .set((10.0, 1.5))
            .at[5]
            .set((13.0 if inside_before else 12.5, 1.5))
        )

    env, before = _start(depth=7.0, arrange=arrange)
    action = _actions((0, 5))._replace(
        move=neutral_action().move.at[5].set(MOVE_WEST if inside_before else MOVE_EAST)
    )
    after, _, actual = _check(env, before, action)
    assert not after.core_state.alive_mask[5]
    assert bool(after.core_state.agent_positions[5, 0] >= 13) != inside_before
    assert float(actual.components["enemy_red_zone_death_credit"][0]) == pytest.approx(
        0.1 if inside_before else 0
    )


def test_terminal_autoreset_reward_uses_producing_successor_and_padding() -> None:
    from marl_battlegrounds.autoreset import AutoReset

    def arrange(state: EnvState) -> EnvState:
        return state._replace(team_deathmatch_scores=jnp.asarray([19, 0], jnp.int32))

    env, before = _start(arrange=arrange)
    action = _actions((0, 5), (2, 0))
    manual, facts, expected = _check(env, before, action)
    assert manual.done.done
    automatic = AutoReset(env, include_training_state=True).step(
        jax.random.key(2), before, action
    )
    final = automatic[4].final
    assert final is not None and final.training_state is not None
    actual = reward_adjustments(
        combat_support_reward,
        before.core_state,
        facts,
        final.training_state,
        jnp.int32(0),
        config=before.config,
    )
    np.testing.assert_array_equal(actual, expected.adjustments)
    assert not automatic[1].done.done
    _, padded, _, _, info = env.step(jax.random.key(3), manual, neutral_action())
    assert info.training_facts is not None and not info.training_facts.has_transition
    zero = reward_adjustments(
        combat_support_reward,
        manual.core_state,
        info.training_facts,
        padded.core_state,
        jnp.int32(1),
        config=manual.config,
    )
    np.testing.assert_array_equal(zero, 0)


@pytest.mark.parametrize("width", [20.0, 24.0])
@pytest.mark.parametrize("mirrored", [False, True])
def test_custom_map_width_and_spawn_sides_keep_recipient_ownership(
    width: float, mirrored: bool
) -> None:
    env, before = _start(depth=7.0, width=width, mirrored=mirrored)
    _, _, actual = _check(env, before, _actions((0, 5), (2, 0), (3, 0)))
    assert float(actual.components["own_zone_ticks"][0]) == pytest.approx(
        0 if mirrored else -0.002
    )
    np.testing.assert_array_equal(actual.adjustments[5:], 0)


@pytest.mark.parametrize("respawn", [False, True])
def test_deaths_and_due_respawns_keep_historical_before_after_meaning(
    respawn: bool,
) -> None:
    def arrange(state: EnvState) -> EnvState:
        if respawn:
            return state._replace(
                alive_mask=state.alive_mask.at[0].set(False),
                current_health=state.current_health.at[0].set(0),
                team_respawn_wave_countdowns=state.team_respawn_wave_countdowns.at[
                    0
                ].set(0),
            )
        return state._replace(current_health=state.current_health.at[0].set(1))

    env, before = _start(depth=7.0, arrange=arrange)
    after, _, actual = _check(
        env, before, neutral_action() if respawn else _actions((0, 5), (5, 0))
    )
    if respawn:
        assert after.core_state.alive_mask[0] and not before.core_state.alive_mask[0]
        np.testing.assert_array_equal(actual.components["own_team_deaths"], 0)
    else:
        assert not after.core_state.alive_mask[0] and not after.core_state.alive_mask[5]
        np.testing.assert_allclose(actual.components["own_team_deaths"][:5], -0.05)
        np.testing.assert_allclose(actual.components["own_red_zone_deaths"][:5], -0.1)
        assert actual.components["enemy_death_credit"][0] > 0
        assert actual.components["own_zone_ticks"][0] == 0


def test_team_average_keeps_each_event_at_its_per_agent_price() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(current_health=state.current_health.at[0].set(1))

    env, before = _start(depth=7.0, arrange=arrange)
    after, facts, actual = _check(env, before, _actions((0, 5), (5, 0)))
    assert not after.core_state.alive_mask[0] and not after.core_state.alive_mask[5]
    team = combat_support_team_reward(
        before.core_state, facts, after.core_state, jnp.int32(0), config=before.config
    )
    active = before.config.agent_profile.active_mask[:5]
    parts = {name: np.asarray(value[:5]) for name, value in actual.components.items()}
    shared = (
        parts["own_team_deaths"][0]
        + parts["own_red_zone_deaths"][0]
        + parts["priest_heal_range_pairs"][0]
    )
    own = sum(
        parts[name].sum()
        for name in (
            "enemy_death_credit",
            "enemy_red_zone_death_credit",
            "enemy_damage_hp",
            "priest_effective_healing_hp",
            "own_zone_ticks",
        )
    )
    assert parts["own_team_deaths"][0] == pytest.approx(-0.05)
    assert parts["enemy_death_credit"][0] == pytest.approx(0.05)
    assert parts["own_zone_ticks"].sum() < 0
    averaged = float(team_task_reward(team.adjustments[:5], active))
    assert averaged == pytest.approx(shared + own, rel=1e-5)
    assert float(team.components["enemy_death_credit"][1]) == pytest.approx(
        parts["enemy_death_credit"].sum()
    )
    diluted = float(team_task_reward(actual.adjustments[:5], active))
    assert diluted == pytest.approx(shared + own / 5, rel=1e-5)


@pytest.mark.parametrize(
    "name", ["combat_support_reward", "combat_support_team_reward"]
)
def test_both_recipe_identities_bind_the_shared_combat_helper(name: str) -> None:
    from marl_battlegrounds.training.shaping import resolve_reward

    reward, identity = resolve_reward(
        f"marl_battlegrounds.training.reward_recipes:{name}"
    )
    assert reward is not None and identity is not None
    assert identity["reference"].endswith(name)
    assert isinstance(identity["combat_source_digest"], str)
    assert identity["call_contract"] == "producing_config_keyword_v1"
