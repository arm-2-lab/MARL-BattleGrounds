"""Provide reusable training feedback above the unchanged simulator.

combat_support_reward adds an explicit combat/support objective to native task
reward, one amount per agent, for learners that train each agent on its own
reward (the PPO family). combat_support_team_reward gives the same terms as one
shared team amount, for learners that train on one team reward (QMIX, PQN-VDN).
Both use compact producing facts, never full metrics or replay capture. Their
coefficients are fixed here so the saved source identity binds their values.
"""

# The observed-combat helper is shared with metrics; Core is only read.
# pyright: reportPrivateUsage=false
from typing import cast

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.env import red_zone_death_mask
from marl_battlegrounds.core.types import PRIEST_CLASS_ID, EnvConfig, EnvState
from marl_battlegrounds.environment import TrainingFacts
from marl_battlegrounds.evaluation.combat_metrics import _observed_combat_quantities
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
    _, damage, healing, excess, combat = _observed_combat_quantities(
        config, before, facts, facts.contributed_to_new_death_by_source
    )
    credit = combat.kill_contributions[:5, 5:] & dead[None, 5:]
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
