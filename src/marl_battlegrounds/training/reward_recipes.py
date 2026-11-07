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

The Capture the Flag recipes pay Team A for flag progress as potentials (a
capture lead, each flag's state, the carrier's own share) plus zero-sum
capture and return credits: ctf_objective_reward and its team form
ctf_objective_team_reward. ctf_objective_route_reward and
ctf_objective_route_team_reward add the carriers' walks home around obstacles,
read from a bank of route fields built once per process. They need the
learner's discount and the Capture the Flag facts. All four are at version
1 and declare that version's reward identity, so an edit to this file that
keeps their rewards still resumes their runs.
tests/training/test_training_capture_the_flag_rewards.py pins them against
independent arithmetic; a change to any of their reward values needs a new
version of the declaration or a new recipe name.
"""

import functools
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds._task_settings import DEFAULT_CTF_FLAG_HOME_POSITIONS
from marl_battlegrounds.core.env import red_zone_death_mask
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    PRIEST_CLASS_ID,
    TASK_MODE_OUTCOME_ONGOING,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.environment import TrainingFacts
from marl_battlegrounds.policies.route_field import (
    FINE_CELL_SIZE,
    blocked_cells,
    route_distance_at,
    route_distances,
)
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


# Capture the Flag recipe values (version 1). Potentials are worth this much
# while their state lasts; credits are paid once per event.
_CAPTURE_MARGIN = 0.10  # Per capture of Team A's lead.
_FLAG_CARRIED = 0.05  # A flag held by its enemy.
_FLAG_DROPPED = 0.02  # A flag lying where its carrier fell.
_CARRIER = 0.02  # The carrier's own share, in the per-agent form only.
_CAPTURE_CREDIT = 0.02
_RETURN_CREDIT = 0.02
_ROUTE = 0.03  # A carrier that has walked the whole way home.
# Parts every Capture the Flag recipient gets in equal amounts. The other
# three parts belong to the agents who earned them.
_CTF_SHARED_PARTS = frozenset(("objective_potential", "route_potential"))


def _route_bank_maps() -> list[EnvConfig]:
    """Return one configuration per Capture the Flag map the route bank covers.

    It reads Capture the Flag's own catalog (tasks.maps_for_task("ctf")), so a
    map that changes there changes the bank. It is the one place that names
    the bank's maps.
    """
    from marl_battlegrounds import tasks

    return [
        tasks.make_standard_capture_the_flag_config(
            map_id=info.map_id, team_a_roster=("mage",), team_b_roster=("mage",)
        )
        for info in tasks.maps_for_task("ctf")
    ]


@functools.cache
def _route_bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the walking distances to each flag home on every bank map, once.

    Returns
    -------
    tuple of numpy.ndarray
        Read-only arrays. Obstacles, float32 (M, MAX_OBSTACLES, 8), one table
        per map, in _route_bank_maps' order. Fields, float32
        (M, 2, rows, columns): the walk in map units from each fine-grid cell
        to each default flag home (DEFAULT_CTF_FLAG_HOME_POSITIONS, Team A's
        first). Home walks, float32 (M, 2): the walk to each default home from
        the other one.

    Raises
    ------
    ValueError
        Two maps share an obstacle table (a lookup would be ambiguous), the
        maps differ in size, or a home cannot be reached from the other.

    Notes
    -----
    One compiled call builds every map's fields, about 5 seconds on a CPU
    with the map configs, once per process (8.3 MB for 52 maps). A caller
    inside a trace must call it under jax.ensure_compile_time_eval, so the
    arrays stay concrete and enter the compiled program as constants.
    """
    configs = _route_bank_maps()
    sizes = {(float(config.map_width), float(config.map_height)) for config in configs}
    if len(sizes) != 1:
        raise ValueError("Route bank maps must all have the same size")
    (width, height), homes = (
        sizes.pop(),
        np.asarray(DEFAULT_CTF_FLAG_HOME_POSITIONS),
    )
    obstacles = np.stack(
        [np.asarray(config.obstacles, np.float32) for config in configs]
    )
    if len({table.tobytes() for table in obstacles}) != len(obstacles):
        raise ValueError("Two route bank maps share an obstacle table")

    def walks_home(table: Array) -> tuple[Array, Array]:
        """Return one map's fields toward both homes and each home-to-home walk."""
        blocked = blocked_cells(
            table, map_width=width, map_height=height, cell_size=FINE_CELL_SIZE
        )
        fields = route_distances(blocked, jnp.asarray(homes), cell_size=FINE_CELL_SIZE)
        read = functools.partial(route_distance_at, cell_size=FINE_CELL_SIZE)
        return fields, jax.vmap(read)(fields, jnp.asarray(homes[::-1]))

    build = jax.jit(jax.vmap(walks_home))
    fields, home_walks = cast(tuple[Array, Array], build(obstacles))
    bank = (
        obstacles,
        np.asarray(fields, np.float32),
        np.asarray(home_walks, np.float32),
    )
    if not np.isfinite(bank[2]).all():
        raise ValueError("A route bank map has a flag home the other cannot reach")
    for array in bank:
        array.flags.writeable = False
    return bank


def _route_potential(config: EnvConfig, before: EnvState, after: EnvState) -> Array:
    """Return the route potential before and after one transition.

    Parameters
    ----------
    config : EnvConfig
        The producing game's scalar configuration. Its obstacle table picks
        the bank map, compared exactly in float32; its flag homes pick each
        team's field, so a spawn exchange needs nothing extra.
    before, after : EnvState
        Scalar states around the same producing action.

    Returns
    -------
    Array
        Float32 (2,), [before, after]: _ROUTE times Team A's carrier progress
        minus Team B's. A carrier's progress is clip(1 - its walk to its own
        home / the walk between the two homes, 0, 1), and 0 when nobody
        carries. On a map or with homes outside the bank, both walks are
        straight lines instead.
    """
    with jax.ensure_compile_time_eval():
        bank_obstacles, bank_fields, bank_home_walks = _route_bank()
    homes = config.capture_the_flag_flag_home_positions
    default_homes = jnp.asarray(DEFAULT_CTF_FLAG_HOME_POSITIONS, jnp.float32)
    same_map = jnp.all(jnp.asarray(bank_obstacles) == config.obstacles, axis=(1, 2))
    same_home = jnp.all(homes[:, None] == default_homes[None], axis=-1)
    known = jnp.any(same_map) & jnp.all(jnp.any(same_home, axis=-1))
    game_map, home = jnp.argmax(same_map), jnp.argmax(same_home, axis=-1)
    fields = jnp.asarray(bank_fields)[game_map, home]
    straight = cast(Array, jnp.linalg.norm(homes[0] - homes[1]))
    home_walk = jnp.where(known, jnp.asarray(bank_home_walks)[game_map, home], straight)
    read = jax.vmap(functools.partial(route_distance_at, cell_size=FINE_CELL_SIZE))

    def potential(state: EnvState) -> Array:
        """Score one state; Team A's carrier stands where Team B's flag is."""
        positions = state.capture_the_flag_flag_positions[::-1]
        carried = state.capture_the_flag_flag_carrier_slots[::-1] >= 0
        straight = cast(Array, jnp.linalg.norm(positions - homes, axis=-1))
        walk = jnp.where(known, read(fields, positions), straight)
        progress = jnp.where(carried, jnp.clip(1 - walk / home_walk, 0, 1), 0)
        return _ROUTE * (progress[0] - progress[1])

    return jnp.stack((potential(before), potential(after)))


def _ctf_reward(
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    *,
    config: EnvConfig,
    discount: Array,
    route: bool,
) -> RewardOutput:
    """Compute the per-agent form of the Capture the Flag recipes.

    The public recipes call this; their docstrings define every part. route
    adds the route potential (recipe 2). Raises ValueError, at setup or
    tracing, when facts lack the Capture the Flag record.
    """
    ctf = facts.capture_the_flag_facts
    if ctf is None:
        raise ValueError(
            "Capture the Flag rewards need the Capture the Flag facts: make a "
            '"ctf" environment with training_facts=True'
        )
    # A finished game's next potential is 0, so potentials pay nothing net.
    keep = jnp.where(ctf.outcome == TASK_MODE_OUTCOME_ONGOING, discount, 0.0)

    def objective(state: EnvState) -> Array:
        """Score Team A's capture lead plus the enemy flag's value minus ours."""
        scores = state.capture_the_flag_scores.astype(jnp.float32)
        value = jnp.where(
            state.capture_the_flag_flag_carrier_slots >= 0,
            _FLAG_CARRIED,
            jnp.where(
                state.capture_the_flag_dropped_flag_countdowns > 0, _FLAG_DROPPED, 0
            ),
        )
        return _CAPTURE_MARGIN * (scores[0] - scores[1]) + value[1] - value[0]

    def carrier(state: EnvState) -> Array:
        """Mark the Team A agent that carries Team B's flag."""
        slot = state.capture_the_flag_flag_carrier_slots[1]
        return _CARRIER * (jnp.arange(5) == slot).astype(jnp.float32)

    active = config.agent_profile.active_mask[:5]
    players = jnp.maximum(active.sum(), 1).astype(jnp.float32)
    captured = ctf.enemy_flag_was_captured_by_agent.astype(jnp.float32)
    returned = ctf.ally_flag_was_returned_by_agent.astype(jnp.float32)
    parts = {
        "objective_potential": jnp.full(
            (5,), keep * objective(after) - objective(before), jnp.float32
        ),
        "carrier_potential": keep * carrier(after) - carrier(before),
        "capture_credit": _CAPTURE_CREDIT
        * (captured[:5] - jnp.max(captured[5:]) / players),
        "return_credit": _RETURN_CREDIT
        * (returned[:5] - jnp.max(returned[5:]) / players),
    }
    if route:
        potentials = _route_potential(config, before, after)
        parts["route_potential"] = jnp.full(
            (5,), keep * potentials[1] - potentials[0], jnp.float32
        )
    recipients = active & facts.has_transition

    def slots(value: Array) -> Array:
        """Keep configured Team A recipients and pad Team B with zeros."""
        return jnp.concatenate(
            (jnp.where(recipients, value, jnp.float32(0)), jnp.zeros(5, jnp.float32))
        )

    components = {name: slots(value) for name, value in parts.items()}
    return RewardOutput(jnp.stack(tuple(components.values())).sum(0), components)


def _ctf_team_form(
    output: RewardOutput, config: EnvConfig, facts: TrainingFacts
) -> RewardOutput:
    """Turn a per-agent Capture the Flag output into its team form.

    Every configured Team A recipient gets the shared parts once plus the Team
    A total of each own-agent part; Team B and padding stay zero.
    """
    recipients = jnp.concatenate(
        (
            config.agent_profile.active_mask[:5] & facts.has_transition,
            jnp.zeros(5, jnp.bool_),
        )
    )
    parts = {
        name: value
        if name in _CTF_SHARED_PARTS
        else jnp.where(recipients, value[:5].sum(), jnp.float32(0))
        for name, value in output.components.items()
    }
    return RewardOutput(jnp.stack(tuple(parts.values())).sum(0), parts)


def ctf_objective_reward(
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    progress: Array,
    *,
    config: EnvConfig,
    discount: Array,
) -> RewardOutput:
    """Reward Team A learners for flag progress, captures and returns.

    Use this form for learners that train each agent on its own reward (the
    PPO family). Example: a Team A agent picks up Team B's flag at its home.
    Every Team A agent gets 0.99 x 0.05 now (Team B's flag is carried), and the
    carrier also gets 0.99 x 0.02 (its own share).

    Parameters
    ----------
    before, after : EnvState
        Scalar states around the same producing action. Use the terminal state
        before AutoReset, never its replacement game. These privileged states
        belong only to training feedback and must not enter actor inputs.
    facts : TrainingFacts
        Compact facts from a Capture the Flag environment made with
        training_facts=True. Missing Capture the Flag facts raise ValueError.
    progress : Array
        Completed-round counter accepted for the shared callback contract.
        This fixed recipe does not change with training progress.
    config : EnvConfig
        Keyword-only configuration of the producing game, from before.config.
        Its ordinary values remain dynamic under jit/vmap.
    discount : Array
        Keyword-only float32 scalar: the learner's discount per transition,
        which collection passes (0.99 for the PPO defaults).

    Returns
    -------
    RewardOutput
        Float32 (10,) adjustments and four named parts. Team B, unconfigured
        slots and padding rows (has_transition False) are zero.
        objective_potential: every configured Team A recipient gets
        discount x phi(after) - phi(before), where phi is 0.10 x (Team A's
        captures - Team B's) + v(Team B's flag) - v(Team A's flag), and v is 0
        at home, 0.02 when dropped and 0.05 when carried.
        carrier_potential: the same difference for phi_i, 0.02 while agent i
        carries Team B's flag, else 0.
        capture_credit: +0.02 to the Team A agent who captures; a Team B
        capture costs each Team A recipient 0.02 / the number of configured
        Team A agents.
        return_credit: +0.02 to the Team A agent who returns their flag by
        touch (the nearest one touching it); a Team B return costs each Team A
        recipient 0.02 / the number of configured Team A agents.

    Raises
    ------
    ValueError
        facts has no Capture the Flag record (at setup or tracing).

    Notes
    -----
    Every potential reads the flag state after the tick, never a drop inferred
    from a death, so a capture on the tick its carrier dies counts as a capture
    (objectives resolve before combat). Once a game ends (its Capture the Flag
    outcome is no longer ongoing), phi(after) is 0, so over a whole game the
    potentials sum to 0 undiscounted and keep the best strategies unchanged
    (Ng, Harada and Russell, 1999); a capture lead that ends in a timeout draw
    is worth nothing. The capture and return credits are zero-sum between the
    teams, so self-play cannot farm them. Pure JAX with no host callback or
    metric output. Native rewards remain separate; disable built-in shaping
    when using this recipe. QMIX and PQN-VDN average the Team A rewards into
    one team reward; give them ctf_objective_team_reward instead.

    See Also
    --------
    ctf_objective_team_reward : The team form.
    ctf_objective_route_reward : This recipe plus the route
        potential.
    """
    del progress
    return _ctf_reward(
        before, facts, after, config=config, discount=discount, route=False
    )


def ctf_objective_team_reward(
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    progress: Array,
    *,
    config: EnvConfig,
    discount: Array,
) -> RewardOutput:
    """Give Team A ctf_objective_reward as one shared team amount.

    Use this form for learners that train on one team reward, such as QMIX and
    PQN-VDN. Each event pays the team what the per-agent form pays the agent it
    happens to. Example: our carrier captures. The team gets the capturer's
    +0.02 credit once, as the capturer does in the per-agent form.

    Parameters
    ----------
    before, facts, after, progress, config, discount
        The same inputs, meanings and requirements as
        ctf_objective_reward. progress is not used.

    Returns
    -------
    RewardOutput
        Float32 (10,) adjustments and the same four named parts. Every
        configured Team A recipient gets one equal amount: objective_potential
        once, plus the Team A total of carrier_potential, capture_credit and
        return_credit. Team B, unconfigured slots and padding rows are zero.

    Raises
    ------
    ValueError
        facts has no Capture the Flag record (at setup or tracing).

    Notes
    -----
    QMIX and PQN-VDN average the Team A slot rewards into one team reward;
    with this form every slot holds the same amount, so the average is exactly
    that amount. Team totals include slots a fixed partner controls.

    See Also
    --------
    ctf_objective_reward : The per-agent form for the PPO family.
    """
    del progress
    output = _ctf_reward(
        before, facts, after, config=config, discount=discount, route=False
    )
    return _ctf_team_form(output, config, facts)


def ctf_objective_route_reward(
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    progress: Array,
    *,
    config: EnvConfig,
    discount: Array,
) -> RewardOutput:
    """Add the carrier's walk home to ctf_objective_reward.

    Use this form for learners that train each agent on its own reward (the
    PPO family). Example: our carrier walks from halfway home to three
    quarters of the way home. Every Team A agent's route potential rises from
    0.015 to 0.0225.

    Parameters
    ----------
    before, facts, after, progress, config, discount
        The same inputs, meanings and requirements as
        ctf_objective_reward. progress is not used.

    Returns
    -------
    RewardOutput
        Float32 (10,) adjustments: the four parts of
        ctf_objective_reward plus route_potential. Every
        configured Team A recipient gets discount x phi(after) - phi(before),
        where phi is 0.03 x (Team A's carrier progress - Team B's). A
        carrier's progress is clip(1 - its walk to its own flag home / the
        walk between the two homes, 0, 1), and 0 when nobody carries.

    Raises
    ------
    ValueError
        facts has no Capture the Flag record (at setup or tracing).

    Notes
    -----
    Walks go around obstacles on the fine route field
    (policies.route_field). The first trace builds the fields for every
    Capture the Flag map once per process (about 5 seconds on a CPU, 8.3 MB),
    and each game finds its map by its exact obstacle table and each team's
    field by its exact flag home. On a researcher's own map, or with other
    flag homes, both walks are straight lines. The potential ends with the
    game, like the others.

    See Also
    --------
    ctf_objective_route_team_reward : The team form.
    """
    del progress
    return _ctf_reward(
        before, facts, after, config=config, discount=discount, route=True
    )


def ctf_objective_route_team_reward(
    before: EnvState,
    facts: TrainingFacts,
    after: EnvState,
    progress: Array,
    *,
    config: EnvConfig,
    discount: Array,
) -> RewardOutput:
    """Give Team A ctf_objective_route_reward as one team amount.

    Use this form for learners that train on one team reward, such as QMIX and
    PQN-VDN.

    Parameters
    ----------
    before, facts, after, progress, config, discount
        The same inputs, meanings and requirements as
        ctf_objective_reward. progress is not used.

    Returns
    -------
    RewardOutput
        Float32 (10,) adjustments and the same five named parts. Every
        configured Team A recipient gets one equal amount: objective_potential
        and route_potential once, plus the Team A total of carrier_potential,
        capture_credit and return_credit. Team B, unconfigured slots and
        padding rows are zero.

    Raises
    ------
    ValueError
        facts has no Capture the Flag record (at setup or tracing).

    See Also
    --------
    ctf_objective_route_reward : The per-agent form.
    ctf_objective_team_reward : The team form without the route.
    """
    del progress
    output = _ctf_reward(
        before, facts, after, config=config, discount=discount, route=True
    )
    return _ctf_team_form(output, config, facts)


# Version 1 of the four Capture the Flag recipes: the reward identity their runs
# save (shaping.resolve_reward adds the reference). Its digests describe this
# file as it stood when version 1 was frozen on 7 November 2026; the rewards are
# pinned by tests, not by these digests, so an edit that keeps the rewards still
# resumes the runs, and an edit that changes a reward needs a new version.
_CTF_SAVED_IDENTITY_V1: dict[str, dict[str, object]] = {
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
            "c067d3570f614ddfc7f58b4c614cf00bcdea36efa26e512fa735218b7e57e6a2"
        ),
        "source_scope": "defining_module",
        "configuration_scope": "source_and_defaults_only",
        "external_state": "unknown",
        "progress_clock": "completed_training_rounds_before_step",
        "call_contract": "producing_config_keyword_v1",
        "discount_contract": "learner_discount_keyword_v1",
    }
    for name, code_digest in (
        (
            "ctf_objective_reward",
            "4cc755e118794c54dff22519563346715bc1a268e0f3eb40abcb14f0b38d3dde",
        ),
        (
            "ctf_objective_team_reward",
            "d5a8493c7c9a31f183bee598e98f461147982568265d100dac237121034ecfcf",
        ),
        (
            "ctf_objective_route_reward",
            "4c65a1a5d2c72e21d7e7afab8371768dd9a6c65014da8725a36bf5f945d714b8",
        ),
        (
            "ctf_objective_route_team_reward",
            "9d1cd6415e3a7744a6ab365d3118bd0f0fa473d59ccdf65899c3b4b676200b3e",
        ),
    )
}
for _recipe in (
    ctf_objective_reward,
    ctf_objective_team_reward,
    ctf_objective_route_reward,
    ctf_objective_route_team_reward,
):
    setattr(_recipe, "reward_identity", _CTF_SAVED_IDENTITY_V1[_recipe.__name__])  # noqa: B010
