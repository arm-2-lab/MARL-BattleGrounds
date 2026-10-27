"""Provide reusable training feedback above the unchanged simulator.

combat_support_reward adds an explicit combat/support objective to native task
reward, one amount per agent, for learners that train each agent on its own
reward (the PPO family). combat_support_team_reward gives the same terms as one
shared team amount, for learners that train on one team reward (QMIX, PQN-VDN).
Both use compact producing facts, never full metrics or replay capture.

Both recipes are frozen at version 1, the definition every run of them has
used. They compute damage, healing and kill credit with their own copy of the
Season 0 rule (``_observed_combat``), so later changes to the evaluation
metrics never change these rewards, and each declares the reward identity its
runs saved (``reward_identity``), so an edit to this file that keeps the
rewards still resumes those runs. tests/training/test_training_reward_recipes.py
pins the rewards against independent arithmetic; a change to any reward value
needs a new recipe name.
"""

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.env import red_zone_death_mask
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    PRIEST_CLASS_ID,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.environment import TrainingFacts
from marl_battlegrounds.training.shaping import RewardOutput

_KILL = 0.05
_DEATH = 0.05
_RED_ZONE_KILL = 0.10
_RED_ZONE_DEATH = 0.10
_DAMAGE = 0.001
_PRIEST_HEAL = 0.001
_HEAL_RANGE = 0.001
_ZONE_TICK = 0.002
# Parts combat_support_reward already gives every recipient in equal amounts.
# The other five parts belong to the agent who earned them.
_SHARED_PARTS = frozenset(
    ("own_team_deaths", "own_red_zone_deaths", "priest_heal_range_pairs")
)


def _observed_combat(
    config: EnvConfig, state: EnvState, facts: TrainingFacts
) -> tuple[Array, Array, Array, Array]:
    """Return one transition's damage, healing, excess healing and kill credit.

    A deliberate copy of the evaluation metrics' observed-combat rule as it
    stood for Season 0 (``evaluation.combat_metrics``, 30 September 2026). The
    metrics may change that rule (for example to credit Blessing of Freedom);
    this copy stays fixed so the recipes' rewards never move with them.

    Parameters
    ----------
    config : EnvConfig
        The producing game's scalar configuration.
    state : EnvState
        The pre-action state; only its current health is read.
    facts : TrainingFacts
        The same transition's compact facts, with routing validity and
        post-combat health present.

    Returns
    -------
    tuple of Array
        Damage, healing and excess healing, float32 (10, 10), then kill
        credit, bool (10, 10). Axes are global source slot, then recipient.
        Kill credit marks direct damage to a newly dead recipient, plus each
        Priest who usefully healed one of those direct attackers (one hop,
        not recursive).

    Raises
    ------
    ValueError
        The facts lack routing validity or post-combat health (at setup or
        tracing).
    """
    if (
        facts.combat_effect_has_recipient_by_source is None
        or facts.health_after_combat_resolution_by_recipient is None
    ):
        raise ValueError(
            "Observed combat needs routing validity and post-combat health facts"
        )
    routing = (
        jax.nn.one_hot(
            facts.combat_effect_recipient_global_slot_by_source,
            MAX_AGENT_SLOTS,
            dtype=jnp.float32,
        )
        * facts.combat_effect_has_recipient_by_source[:, None]
    )
    damage = (
        routing
        * (
            facts.source_modified_damage_output_by_source
            * facts.recipient_damage_modifier_by_source
        )[:, None]
    )
    healing = (
        routing
        * (
            facts.source_modified_healing_output_by_source
            * facts.recipient_healing_modifier_by_source
        )[:, None]
    )
    total_healing = facts.total_effective_healing_by_recipient
    uncapped = state.current_health + (
        total_healing - facts.total_effective_damage_by_recipient
    )
    excess = jnp.clip(
        uncapped - facts.health_after_combat_resolution_by_recipient,
        min=0,
        max=total_healing,
    )
    # Compare recipient totals, so rounding cannot leave a false useful heal.
    useful = (
        (healing > 0)
        & (total_healing > excess)[None, :]
        & (config.agent_profile.class_ids == PRIEST_CLASS_ID)[:, None]
    )
    direct = (
        routing.astype(jnp.bool_) & facts.contributed_to_new_death_by_source[:, None]
    )
    kill_credit = direct | jnp.any(useful[:, :, None] & direct[None, :, :], axis=1)
    excess_healing = (
        healing * (excess / jnp.where(total_healing > 0, total_healing, 1))[None, :]
    )
    return damage, healing, excess_healing, kill_credit


def combat_support_reward(
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    progress: Array,
    *,
    config: EnvConfig,
) -> RewardOutput:
    """Add combat, Priest support and own-zone feedback for Team A learners.

    Parameters
    ----------
    before, after : EnvState
        Scalar states around the same producing action. Use the terminal state
        before AutoReset, never its replacement game. These privileged states
        belong only to training feedback and must not enter actor inputs.
    facts : TrainingFacts
        Compact facts from make(training_facts=True), including valid routing
        and post-combat health. Missing required facts raise ValueError.
    progress : Array
        Completed-round counter accepted for the shared callback contract.
        This fixed recipe does not change with training progress.
    config : EnvConfig
        Keyword-only configuration of the producing game, from before.config.
        Its ordinary values remain dynamic under jit/vmap.

    Returns
    -------
    RewardOutput
        Float32 (10,) adjustments and eight named parts. Team B and inactive
        slots are zero. Each configured Team A recipient gets -0.05 per own
        death, -0.10 extra per own Red Zone death, and +0.001 per ordered
        living Priest/ally pair within Basic range and outside their own zone.
        Credited actors get +0.05 per enemy death and +0.10 extra for Red Zone
        deaths. Actors get +0.001 per capped enemy damage HP; Priests also get
        +0.001 per effective own-team healing HP. Living actors lose 0.002 per
        tick inside their own zone. Collection excludes partner-controlled
        slots when assigning feedback and counting component totals.

    Notes
    -----
    Death means alive before and dead after; same-step death/respawn therefore
    follows that exact rule. Death-zone checks use pre-action positions; zone
    occupancy and heal range use successor positions. Damage caps include
    same-step effective healing. Kill credit includes useful one-hop Priest
    support and can name several actors. Range counts ordered pairs, excludes
    self and includes the radius boundary. This incentive can reward standing
    within healing range even without a heal; it is part of this declared rule.
    Native rewards remain separate. Disable built-in shaping when using this
    recipe alone. Pure JAX code computes observed combat once, with no extra
    simulator step, rescue calculation, host callback or metric/replay output.
    QMIX and PQN-VDN average the Team A rewards into one team reward, which
    would divide every own-agent part by the number of players; give those
    learners combat_support_team_reward instead.

    See Also
    --------
    combat_support_team_reward : The same terms as one shared team amount.
    """
    del progress
    active = config.agent_profile.active_mask
    dead = before.alive_mask & ~after.alive_mask & active
    red_zone_dead = red_zone_death_mask(config, before.agent_positions, dead)
    in_zone = red_zone_death_mask(
        config, after.agent_positions, after.alive_mask & active
    )
    damage, healing, excess, kill_credit = _observed_combat(config, before, facts)
    credit = kill_credit[:5, 5:] & dead[None, 5:]
    red_zone_credit = credit & red_zone_dead[None, 5:]
    effective = jnp.maximum(healing - excess, 0.0)
    health_limit = before.current_health[5:] + effective[:, 5:].sum(0)
    received = damage[:, 5:].sum(0)
    cap = jnp.minimum(1.0, health_limit / jnp.maximum(received, 1e-6))
    caused_damage = (damage[:5, 5:] * cap[None, :]).sum(-1)
    priests = (config.agent_profile.class_ids[:5] == PRIEST_CLASS_ID) & active[:5]
    priest_healing = effective[:5, :5].sum(-1) * priests
    alive = after.alive_mask[:5] & active[:5]
    positions = after.agent_positions[:5]
    distance = cast(
        Array, jnp.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    )
    counted = alive & ~in_zone[:5]
    near = (
        (distance <= config.agent_profile.basic_interaction_radii[:5, None])
        & (priests & counted)[:, None]
        & counted[None, :]
        & ~jnp.eye(5, dtype=jnp.bool_)
    )
    parts = {
        "enemy_death_credit": _KILL * credit.sum(-1).astype(jnp.float32),
        "own_team_deaths": jnp.full((5,), -_DEATH * dead[:5].sum(), jnp.float32),
        "enemy_red_zone_death_credit": _RED_ZONE_KILL
        * red_zone_credit.sum(-1).astype(jnp.float32),
        "own_red_zone_deaths": jnp.full(
            (5,), -_RED_ZONE_DEATH * red_zone_dead[:5].sum(), jnp.float32
        ),
        "enemy_damage_hp": _DAMAGE * caused_damage,
        "priest_effective_healing_hp": _PRIEST_HEAL * priest_healing,
        "priest_heal_range_pairs": jnp.full(
            (5,), _HEAL_RANGE * near.sum().astype(jnp.float32), jnp.float32
        ),
        "own_zone_ticks": -_ZONE_TICK * in_zone[:5].astype(jnp.float32),
    }
    team = parts["own_team_deaths"] + parts["own_red_zone_deaths"]
    team += parts["priest_heal_range_pairs"]
    individual = parts["own_zone_ticks"] + parts["enemy_death_credit"]
    individual += parts["enemy_red_zone_death_credit"]
    individual += parts["enemy_damage_hp"]
    individual += parts["priest_effective_healing_hp"]
    recipients = active[:5] & facts.has_transition

    def slots(value: Array) -> Array:
        """Keep configured Team A recipients and pad Team B with zeros."""
        return jnp.concatenate(
            (jnp.where(recipients, value, jnp.float32(0)), jnp.zeros(5, jnp.float32))
        )

    return RewardOutput(
        slots(team + individual), {k: slots(v) for k, v in parts.items()}
    )


def combat_support_team_reward(
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    progress: Array,
    *,
    config: EnvConfig,
) -> RewardOutput:
    """Give Team A the combat/support terms as one shared team amount.

    Use this form for learners that train on one team reward, such as QMIX and
    PQN-VDN. Each event pays the team what combat_support_reward pays the agent
    it happens to. Example, 5v5: our Mage's kill credit pays the team +0.05 (the
    Mage's own amount), and a teammate's death costs the team -0.05 (every
    agent's amount), so kills and deaths keep the same balance as in PPO.

    Parameters
    ----------
    before, facts, after, progress, config
        The same inputs, meanings and requirements as combat_support_reward.
        config is required and keyword-only. progress is not used.

    Returns
    -------
    RewardOutput
        Float32 (10,) adjustments and the same eight named parts. Every
        configured Team A recipient gets one equal amount: the three shared
        parts once (own_team_deaths, own_red_zone_deaths and
        priest_heal_range_pairs, which every agent already gets in equal
        amounts), plus the Team A total of each of the five own-agent parts
        (enemy_death_credit, enemy_red_zone_death_credit, enemy_damage_hp,
        priest_effective_healing_hp and own_zone_ticks). Each named part holds
        that per-recipient amount, so the parts still sum to adjustments. Team
        B, unconfigured slots and padding rows (has_transition False) are zero.

    Notes
    -----
    QMIX and PQN-VDN average the Team A slot rewards into one team reward
    (baselines.qmix.team_task_reward). With combat_support_reward that average
    divides each own-agent part by the number of configured players: in 5v5 a
    kill credit pays the team only +0.01 while a teammate's death still costs
    -0.05. With this form every slot holds the same amount, so the average is
    exactly the shared parts plus the Team A total of the own-agent parts. The
    native win reward, the same for every slot, still counts once.

    One difference from the PPO family remains by design: a team reward also
    pays every agent's value for a teammate's own events, such as the Mage's
    damage. Team totals include every configured Team A slot, including slots a
    fixed partner controls; learners still learn only from their own slots.

    Pure JAX: calls combat_support_reward once and adds one five-slot sum per
    own-agent part, with the same jit and vmap use. The coefficients and term
    definitions are combat_support_reward's.

    See Also
    --------
    combat_support_reward : The per-agent form for the PPO family.
    """
    output = combat_support_reward(before, facts, after, progress, config=config)
    recipients = jnp.concatenate(
        (
            config.agent_profile.active_mask[:5] & facts.has_transition,
            jnp.zeros(5, jnp.bool_),
        )
    )
    parts = {
        name: value
        if name in _SHARED_PARTS
        else jnp.where(recipients, value[:5].sum(), jnp.float32(0))
        for name, value in output.components.items()
    }
    return RewardOutput(jnp.stack(tuple(parts.values())).sum(0), parts)


# Version 1 of both recipes: the reward identity every run of them saved
# (shaping.resolve_reward adds the reference). Its digests describe the files
# as they stood when those runs began; the rewards are pinned by tests, not by
# these digests, so an edit that keeps the rewards still resumes the runs.
_SAVED_IDENTITY_V1: dict[str, dict[str, object]] = {
    name: {
        "callable": {
            "callable_name": f"marl_battlegrounds.training.reward_recipes.{name}",
            "code_digest": code_digest,
            "code_scope": "python_function_bytecode",
            "closure_content": "none",
            "defaults_digest": (
                "4dbd94257a409786a8b0f5a10c8e3e0fc8453c7c81f70c61a81a0bf3bdf9f746"
            ),
            "external_state": "unknown",
        },
        "source_digest": (
            "73df677933b043d9eb2d601a65719e73bd26030066f0086ba8c3f595d251bdd6"
        ),
        "source_scope": "defining_module",
        "configuration_scope": "source_and_defaults_only",
        "external_state": "unknown",
        "progress_clock": "completed_training_rounds_before_step",
        "call_contract": "producing_config_keyword_v1",
        "combat_source_digest": (
            "afb0c609d63efad4b7c1c61fb6831ff91efee20951d0834fd33f074adb5e39ed"
        ),
    }
    for name, code_digest in (
        (
            "combat_support_reward",
            "1dfa534d83faff7767cd9047f47e76202b964baebcaf605c51ad7e3815f11d25",
        ),
        (
            "combat_support_team_reward",
            "ed9ff0d11578cdd569554709238a91e48faa281e4b7025b4147aaa9a74ce7a7c",
        ),
    )
}
# setattr, because the type checker refuses attribute assignment on functions.
for _recipe in (combat_support_reward, combat_support_team_reward):
    setattr(_recipe, "reward_identity", _SAVED_IDENTITY_V1[_recipe.__name__])  # noqa: B010
