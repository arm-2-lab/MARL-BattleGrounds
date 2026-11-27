"""Run the current-action oracle for an upper-bound training run.

The oracle hands each agent, as it chooses, the action probabilities of every
agent that has already chosen on the same tick. This private module owns that
block, the order in which agents choose, and the route to each fixed
opponent's probabilities. It is used by the current-action oracle's training
and evaluation steps, never by the public System boundary, and its results are
always labeled as an oracle's.

The block has the layout of the observation's block of last tick's actions
(``PreviousTimestepActionObservation``): for each observer, five ally rows and five
enemy rows, each holding a move (9), target (11) and Ultimate (2) part. Core owns
how an actor's action appears in another agent's rows: which row it fills, and
how an opposing observer's target categories are remapped. This module never
copies that rule. It calls Core's own builder once per category of each part and
mixes the results by the probabilities, which is exact because the builder is
linear in each actor's one-hot action. The network's flat block is encoded from
these rows by ``baselines.inputs._encode_handed_rows``.

Main entry points: ``handed_rows`` (the rows, from ``action_part_probabilities``),
``turn_order`` and ``chosen_before`` (who chooses when, and who is handed whom),
``choose_in_order`` (one tick of ten turns), ``fixed_member_step`` (a fixed
opponent's route, refused when it has none), ``oracle_member`` (a team member
the oracle can hand rows to) and ``oracle_apply_systems`` (the oracle's form of
``apply_systems`` for one training decision).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.baselines.actions import (
    NUM_ACTIONS,
    decode_actions,
    encode_actions,
)
from marl_battlegrounds.baselines.ppo import (
    _ActorStep,  # pyright: ignore[reportPrivateUsage]
    oracle_step,
)
from marl_battlegrounds.core.env import (
    _build_ally_enemy_one_hot_action_tensors,  # pyright: ignore[reportPrivateUsage]
    _build_visibility_masked_previous_timestep_action_observation,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    Action,
    EnvState,
    PreviousTimestepActionObservation,
)
from marl_battlegrounds.environment import EnvironmentState
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
    SystemState,
    _action_keys,  # pyright: ignore[reportPrivateUsage]
    _adapter_actions,  # pyright: ignore[reportPrivateUsage]
    _apply_system_pair,  # pyright: ignore[reportPrivateUsage]
    _declare_lane_independent,  # pyright: ignore[reportPrivateUsage]
    _execution,  # pyright: ignore[reportPrivateUsage]
    _initialization_keys,  # pyright: ignore[reportPrivateUsage]
    _reset_for_decision,  # pyright: ignore[reportPrivateUsage]
    controller_identity,
    system_inputs,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations


def action_part_probabilities(probabilities: Array) -> tuple[Array, Array, Array]:
    """Split 198-way action probabilities into the chances of each action part.

    Parameters
    ----------
    probabilities : Array
        Float32 (...,198) probabilities over the categorical action indices of
        ``baselines.actions`` (index = (move x 11 + target) x 2 + Ultimate), in the
        convention of the actions an actor submits: moves in the world's frame,
        targets relative to the actor's own team. Rows of an absent actor may be
        zero.

    Returns
    -------
    tuple[Array, Array, Array]
        Float32 (...,9) move, (...,11) target and (...,2) Ultimate probabilities,
        each the joint distribution summed over the other two parts. Which move
        goes with which target is not kept.

    Notes
    -----
    Pure JAX; works inside jit and vmap. No input is changed.
    """
    joint = probabilities.reshape(
        (*probabilities.shape[:-1], NUM_MOVE_ACTIONS, NUM_TARGET_ACTIONS, -1)
    )
    return joint.sum((-2, -1)), joint.sum((-3, -1)), joint.sum((-3, -2))


def handed_rows(
    state: EnvState,
    move: Array,
    target: Array,
    ultimate: Array,
    chosen: Array,
    visible: tuple[Array, Array] | None = None,
) -> PreviousTimestepActionObservation:
    """Place the actions chosen earlier this tick into every observer's rows.

    Parameters
    ----------
    state : EnvState
        One game's unbatched state at the current decision. Only its shapes and
        dtypes are used; its own history fields are replaced, never read.
    move, target, ultimate : Array
        Float32 (10,9), (10,11) and (10,2) per-part probabilities of every slot's
        action this tick (see ``action_part_probabilities``), in global slot
        order, moves in the world's frame. A one-hot row hands the actual action.
    chosen : Array
        Bool (10,10), [observer, actor]: True when that actor chose before that
        observer on this tick and is alive. An observer's own entry must be False.
    visible : tuple[Array, Array] or None, default=None
        Optional bool (10,5) ally and (10,5) enemy visibility masks, in the
        observation's [observer, relation row] layout, for an "in view only"
        score. None hands every row, whether in view or not.

    Returns
    -------
    PreviousTimestepActionObservation
        Float32 rows with axes [observer, relation row, category], as Core lays
        out last tick's actions, in the world's frame (each actor's own
        spawn-frame step is applied later, as for the observation's block). A
        row is zero when its actor has not chosen before that observer, is
        hidden by ``visible``, or is the observer itself.

    Notes
    -----
    Exact for any probabilities: the result is the probability-weighted sum of
    Core's rows for each category. Pure JAX; works inside jit and vmap (add a
    leading game axis to every argument).
    """
    all_rows = jnp.ones((MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM), bool)
    ally_visible, enemy_visible = (
        visible if visible is not None else (all_rows, all_rows)
    )
    history = state._replace(has_previous_timestep_joint_action=jnp.asarray(True))

    def core_rows(part: str, category: Array) -> PreviousTimestepActionObservation:
        """Return Core's rows when every slot chose this category for this part."""
        heads = {
            "previous_timestep_move_actions": jnp.zeros(MAX_AGENT_SLOTS, jnp.int32),
            "previous_timestep_select_target_actions": jnp.zeros(
                MAX_AGENT_SLOTS, jnp.int32
            ),
            "previous_timestep_use_ultimate_actions": jnp.zeros(
                MAX_AGENT_SLOTS, jnp.int32
            ),
        }
        heads[part] = jnp.full(MAX_AGENT_SLOTS, category, jnp.int32)
        return _build_visibility_masked_previous_timestep_action_observation(
            history._replace(**heads), ally_visible, enemy_visible
        )

    # Core's own row arrangement: rows[o, r, a] is 1 when actor a fills relation
    # row r for observer o, so a weight per row needs no copied index rule.
    ally_of, enemy_of = _build_ally_enemy_one_hot_action_tensors(
        jnp.eye(MAX_AGENT_SLOTS, dtype=jnp.float32), MAX_AGENT_SLOTS
    )
    chosen_rows = chosen.astype(jnp.float32)

    def mixed(part: str, probabilities: Array, size: int) -> tuple[Array, Array]:
        """Mix one part's per-category rows by each row's actor's probability."""

        def for_category(category: Array) -> PreviousTimestepActionObservation:
            """Return Core's rows for one category of this part."""
            return core_rows(part, category)

        by_category = jax.vmap(for_category)(jnp.arange(size))
        index = {
            "previous_timestep_move_actions": (0, 1),
            "previous_timestep_select_target_actions": (2, 3),
            "previous_timestep_use_ultimate_actions": (4, 5),
        }[part]
        rows: list[Array] = []
        for layout, field in zip((ally_of, enemy_of), index, strict=True):
            weights = jnp.einsum("ora,oa,ac->orc", layout, chosen_rows, probabilities)
            rows.append(jnp.einsum("orc,corx->orx", weights, by_category[field]))
        return rows[0], rows[1]

    ally_move, enemy_move = mixed(
        "previous_timestep_move_actions", move, NUM_MOVE_ACTIONS
    )
    ally_target, enemy_target = mixed(
        "previous_timestep_select_target_actions", target, NUM_TARGET_ACTIONS
    )
    ally_ultimate, enemy_ultimate = mixed(
        "previous_timestep_use_ultimate_actions", ultimate, NUM_ULTIMATE_ACTIONS
    )
    return PreviousTimestepActionObservation(
        ally_move, enemy_move, ally_target, enemy_target, ally_ultimate, enemy_ultimate
    )


ORACLE_ORDER_TAG = 0x4F524452
"""Fold-in tag ("ORDR") that derives a tick's order key from its action key.

The order never reuses an action key, so with the oracle switched off every
actor's keys, and so its actions, are exactly today's.
"""


def turn_order(key: Array, fixed_first: Array) -> Array:
    """Draw one tick's order over the ten slots, as each slot's rank 0..9.

    Parameters
    ----------
    key : Array
        One typed Threefry key for this game and tick (derive it with
        ``jax.random.fold_in(action_key, ORACLE_ORDER_TAG)``).
    fixed_first : Array
        Bool (10,): True for the slots of a fixed opponent (a team that cannot
        read a handed block, such as a Season 0 team or a scripted bot). Those
        slots always take ranks 0..4. Against a copy of the learner (all False),
        all ten slots are ranked in one uniform random order.

    Returns
    -------
    Array
        Int32 (10,) ranks: 0 chooses first. Ties are impossible.

    Notes
    -----
    Pure JAX; vmap it over games. A fixed opponent's slots are ordered among
    themselves too, which changes nothing, since they read no block.
    """
    priority = jax.random.uniform(key, (MAX_AGENT_SLOTS,)) - fixed_first.astype(
        jnp.float32
    )
    return jnp.argsort(jnp.argsort(priority)).astype(jnp.int32)


def chosen_before(ranks: Array, present: Array, *, teammates_only: bool) -> Array:
    """Mark, for each observer, the slots that chose before it this tick.

    Parameters
    ----------
    ranks : Array
        Int32 (10,) ranks from ``turn_order``.
    present : Array
        Bool (10,): True for configured, living slots. An absent or dead slot
        chooses nothing, so no one is handed its row.
    teammates_only : bool
        Static. True keeps only each observer's own teammates (the
        teammates-only arm); False keeps both teams.

    Returns
    -------
    Array
        Bool (10,10), [observer, actor], the ``chosen`` argument of
        ``handed_rows``. An observer's own entry is always False, since its rank
        is never below itself.
    """
    earlier = ranks[None, :] < ranks[:, None]
    chosen = earlier & present[None, :]
    if teammates_only:
        team = jnp.arange(MAX_AGENT_SLOTS) // MAX_AGENTS_PER_TEAM
        chosen = chosen & (team[:, None] == team[None, :])
    return chosen


class OracleChoices(NamedTuple):
    """One tick of choices made in the oracle's order, for both teams.

    Attributes
    ----------
    indices : Array
        Int32 (B,10) categorical indices in world directions, one per slot,
        each taken from the call at that slot's own turn.
    probabilities : Array
        Float32 (B,10,198) each slot's distribution at its own turn, in world
        directions.
    handed : PreviousTimestepActionObservation
        The rows each slot was handed at its own turn, leading axes (B,10). The
        learner stores Team A's rows and recomputes its update with them.
    learning_outputs : tuple[Tree, Tree]
        Each team's learning values, every leaf (B,5,...) taken per slot from
        that slot's own turn (for the PPO family, its index and
        log-probability).
    memory_a, memory_b : Tree
        Each team step's next memory. A block never enters memory, so every
        turn's call gives the same memory, and the last one is kept.
    """

    indices: Array
    probabilities: Array
    handed: PreviousTimestepActionObservation
    learning_outputs: tuple[Tree, Tree]
    memory_a: object
    memory_b: object


type Tree = object
"""Any JAX tree of arrays (memory or learning values)."""

PairStep = Callable[
    [PreviousTimestepActionObservation, PreviousTimestepActionObservation],
    tuple[_ActorStep, _ActorStep],
]
"""Both teams' steps for one turn, with weights, memory, inputs and keys bound.

It receives the rows handed to Team A's and Team B's five slots (leading axes
(B,5) each) and returns one _ActorStep per team. A fixed opponent ignores its
rows. One call computes both teams, as the shared System runner does.
"""


def fixed_member_step(member: System) -> Callable[..., _ActorStep]:
    """Return the oracle step of a fixed opponent, or refuse one with no route.

    A fixed opponent (a frozen saved team or a scripted bot) chooses first and
    cannot read a handed block. The oracle still needs the distribution it acts
    from, to hand to the agents that choose after it.

    Parameters
    ----------
    member : System
        One fixed member of Team B's training or evaluation pool.

    Returns
    -------
    callable
        ``step(variables, memory, inputs, keys, handed=None)`` returning an
        _ActorStep: a built-in method's own marked step (``ppo.oracle_step``),
        or, for a team of the scripted bots ALPHA, BETA or GAMMA, a step that
        runs them exactly as the shared runner runs a Policy adapter and
        returns their actions as one-hot rows.
        The bots ignore their keys, so their choice is certain and its
        log-probability is zero.

    Raises
    ------
    ValueError
        member is neither a marked built-in method nor a team of the three
        scripted bots. The oracle never falls back to zero rows.
    """
    step = oracle_step(member.apply)
    if step is not None:
        return step
    if member._policies and all(  # pyright: ignore[reportPrivateUsage]
        controller_identity(policy) is not None
        for policy in member._policies  # pyright: ignore[reportPrivateUsage]
    ):
        execution = _execution(member)

        def bot_step(
            variables: object,
            memory: object,
            inputs: SystemInput,
            keys: Array,
            handed: PreviousTimestepActionObservation | None = None,
        ) -> _ActorStep:
            """Run the bots as the shared runner does; return certain rows."""
            del handed
            output = _adapter_actions(execution, variables, memory, inputs, keys)
            indices = encode_actions(output.actions)
            return _ActorStep(
                indices,
                output.next_memory,
                jnp.zeros(indices.shape, jnp.float32),
                jax.nn.one_hot(indices, NUM_ACTIONS, dtype=jnp.float32),
                output.learning_outputs,
            )

        return bot_step
    raise ValueError(
        f"The current-action oracle has no route to {member.name!r}'s action "
        "probabilities. It supports the built-in methods and the scripted bots "
        "tdm-alpha, tdm-beta and tdm-gamma as fixed opponents."
    )


def choose_in_order(
    states: EnvState,
    ranks: Array,
    present: Array,
    pair: PairStep,
    *,
    teammates_only: bool,
    hand_sampled_actions: bool,
) -> OracleChoices:
    """Let the ten slots choose in turn, each handed everything chosen before it.

    Parameters
    ----------
    states : EnvState
        The B games' core states at this decision (leading axis B); only their
        shapes reach Core's row builder.
    ranks : Array
        Int32 (B,10) turn ranks from ``turn_order``.
    present : Array
        Bool (B,10) configured, living slots.
    pair : PairStep
        Slots 0..4 and 5..9, called once per turn with the rows each team's
        slots would be handed then; under jit the work that does not read the
        rows (encoder and memory) is the same in every call.
    teammates_only : bool
        Static. True hands only each slot's own teammates (the teammates-only
        arm).
    hand_sampled_actions : bool
        Static. False (the main oracle) hands probabilities; True hands each
        chosen slot's sampled action as a one-hot row (the sampled-action
        arm).

    Returns
    -------
    OracleChoices
        Each slot's index, distribution, handed rows and learning values from
        its own turn, and both teams' next memories.

    Notes
    -----
    Ten turns, always: a game against a fixed opponent spends its first five on
    the opponent, whose rows it reads once they are known. Pure JAX; works
    inside jit.
    """
    games = ranks.shape[0]
    chosen = jax.vmap(functools.partial(chosen_before, teammates_only=teammates_only))(
        ranks, present
    )
    known = jnp.zeros((games, MAX_AGENT_SLOTS, NUM_ACTIONS), jnp.float32)
    indices = jnp.zeros((games, MAX_AGENT_SLOTS), jnp.int32)
    probabilities = jnp.zeros((games, MAX_AGENT_SLOTS, NUM_ACTIONS), jnp.float32)
    handed: PreviousTimestepActionObservation | None = None
    learning: tuple[Tree, Tree] | None = None
    memory_a: object = None
    memory_b: object = None
    for turn in range(MAX_AGENT_SLOTS):
        move, target, ultimate = action_part_probabilities(known)
        rows = jax.vmap(handed_rows)(states, move, target, ultimate, chosen)
        step_a, step_b = pair(_team_rows(rows, 0), _team_rows(rows, 1))
        memory_a, memory_b = step_a.memory, step_b.memory
        now = ranks == turn
        step_index = jnp.concatenate((step_a.indices, step_b.indices), axis=1)
        step_probabilities = jnp.concatenate(
            (step_a.probabilities, step_b.probabilities), axis=1
        )
        indices = jnp.where(now, step_index, indices)
        probabilities = jnp.where(now[..., None], step_probabilities, probabilities)
        handed = rows if handed is None else _keep_turn_rows(now, rows, handed)
        turn_learning = (step_a.learning_outputs, step_b.learning_outputs)
        learning = (
            turn_learning
            if learning is None
            else (
                _keep_turn_tree(
                    now[:, :MAX_AGENTS_PER_TEAM], turn_learning[0], learning[0]
                ),
                _keep_turn_tree(
                    now[:, MAX_AGENTS_PER_TEAM:], turn_learning[1], learning[1]
                ),
            )
        )
        offered = (
            jax.nn.one_hot(step_index, NUM_ACTIONS, dtype=jnp.float32)
            if hand_sampled_actions
            else step_probabilities
        )
        known = jnp.where(now[..., None], offered, known)
    assert handed is not None and learning is not None
    return OracleChoices(indices, probabilities, handed, learning, memory_a, memory_b)


def _team_rows(
    rows: PreviousTimestepActionObservation, team: int
) -> PreviousTimestepActionObservation:
    """Take one team's five observer rows (axis 1) from all ten slots' rows."""
    start = team * MAX_AGENTS_PER_TEAM
    return PreviousTimestepActionObservation(
        *(field[:, start : start + MAX_AGENTS_PER_TEAM] for field in rows)
    )


def _keep_turn_rows(
    now: Array,
    rows: PreviousTimestepActionObservation,
    kept: PreviousTimestepActionObservation,
) -> PreviousTimestepActionObservation:
    """Keep each slot's rows from the turn where now (B,10) marks it."""
    mask = now[:, :, None, None]
    return PreviousTimestepActionObservation(
        *(jnp.where(mask, new, old) for new, old in zip(rows, kept, strict=True))
    )


def _keep_turn_tree(now: Array, new: Tree, kept: Tree) -> Tree:
    """Keep each slot's leaves from the turn where now (B,5) marks it.

    Every leaf of new and kept must have leading axes (B,5), as every built-in
    hook's learning values do; anything else is refused rather than broadcast.
    """

    def keep(fresh: Array, old: Array) -> Array:
        """Select one leaf per slot."""
        if fresh.shape[:2] != now.shape:
            raise ValueError(
                "The oracle keeps learning values per slot; each leaf needs "
                f"leading axes {now.shape}, got {fresh.shape}"
            )
        mask = now.reshape(now.shape + (1,) * (fresh.ndim - 2))
        return jnp.where(mask, fresh, old)

    return jax.tree.map(keep, new, kept)


class OracleMemory(NamedTuple):
    """One wrapped member's memory in an oracle run (see ``oracle_member``).

    Every leaf has the games axis first, so pools, teams and the copy wrapper
    carry it per game like any other memory.

    Attributes
    ----------
    inner : Tree
        The member's own memory, exactly as its hook keeps it.
    rows : PreviousTimestepActionObservation
        Float32 rows with leading axes (B,5): what the member's five slots are
        handed on the current turn. The oracle writes them before each call.
    probabilities : Array
        Float32 (B,5,198) distribution, in world directions, of each slot the
        member acted for on the current turn (valid games, owned slots), and
        zero elsewhere. The oracle zeroes it before each call.
    """

    inner: Tree
    rows: PreviousTimestepActionObservation
    probabilities: Array


def _zero_rows(games: int) -> PreviousTimestepActionObservation:
    """Return float32 rows handing nothing to five slots of each game."""
    allies, enemies = MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM
    shape = (games, MAX_AGENTS_PER_TEAM)
    return PreviousTimestepActionObservation(
        jnp.zeros((*shape, allies, NUM_MOVE_ACTIONS), jnp.float32),
        jnp.zeros((*shape, enemies, NUM_MOVE_ACTIONS), jnp.float32),
        jnp.zeros((*shape, allies, NUM_TARGET_ACTIONS), jnp.float32),
        jnp.zeros((*shape, enemies, NUM_TARGET_ACTIONS), jnp.float32),
        jnp.zeros((*shape, allies, NUM_ULTIMATE_ACTIONS), jnp.float32),
        jnp.zeros((*shape, enemies, NUM_ULTIMATE_ACTIONS), jnp.float32),
    )


def oracle_member(member: System, *, reads_rows: bool) -> System:
    """Wrap one team member so the oracle can hand it rows and read its distribution.

    Parameters
    ----------
    member : System
        A built-in method's System or a team of the scripted bots (see
        ``fixed_member_step``): the learner, or one of its fixed opponents.
    reads_rows : bool
        Keyword-only. True for the learner and its copies, whose networks carry
        the handed weight and are called with the rows. False for a fixed
        opponent (a frozen saved team or the bots), which chooses first, cannot
        read a block and is called with none.

    Returns
    -------
    System
        A JAX System with the member's own variables and components, whose
        memory is an ``OracleMemory``. Its apply calls the member's step with the
        memory's rows and returns exactly the member's actions, next memory and
        learning values; it writes the distribution into the memory's
        ``probabilities`` for valid games and owned slots only. Its init adds
        zero rows and probabilities to the member's own init; it has no custom
        reset hook, so the shared runner resets every leaf of a new game, and
        it can be the base of ``make_opponent_system``. It is lane-independent
        whenever the member is, as every built-in method and the bots are.

    Raises
    ------
    ValueError
        member has no route (``fixed_member_step``).
    """
    step = fixed_member_step(member)
    inner_init = member.init

    def init(variables: Tree, inputs: SystemInput, keys: Array) -> OracleMemory:
        """Start the member's memory, handing nothing yet."""
        games = inputs.valid.shape[0]
        return OracleMemory(
            inner_init(variables, inputs, keys) if inner_init is not None else (),
            _zero_rows(games),
            jnp.zeros((games, MAX_AGENTS_PER_TEAM, NUM_ACTIONS), jnp.float32),
        )

    def apply(
        variables: Tree, memory: OracleMemory, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        """Act as the member does, handed the memory's rows."""
        result = step(
            variables,
            memory.inner,
            inputs,
            keys,
            handed=memory.rows if reads_rows else None,
        )
        owned = inputs.valid[:, None] & (
            inputs.active_mask
            if inputs.controlled_mask is None
            else inputs.controlled_mask
        )
        return SystemOutput(
            decode_actions(result.indices),
            OracleMemory(
                result.memory,
                memory.rows,
                jnp.where(owned[..., None], result.probabilities, memory.probabilities),
            ),
            learning_outputs=result.learning_outputs,
        )

    return System(
        f"Oracle: {member.name}",
        _declare_lane_independent(apply),
        variables=member.variables,
        init=init,
        components=member.components,
        checkpoint=member.checkpoint,
    )


def _is_oracle_memory(node: object) -> bool:
    """Tell an OracleMemory node apart from the tree around it."""
    return isinstance(node, OracleMemory)


def _hand(memory: Tree, rows: PreviousTimestepActionObservation) -> Tree:
    """Write rows into every OracleMemory of one team and zero its distributions."""

    def hand(node: object) -> object:
        """Hand one wrapped member this turn's rows."""
        if not isinstance(node, OracleMemory):
            return node
        return node._replace(
            rows=rows, probabilities=jnp.zeros_like(node.probabilities)
        )

    return jax.tree.map(hand, memory, is_leaf=_is_oracle_memory)


def _acted_probabilities(memory: Tree) -> Array:
    """Sum one team's distributions over its wrapped members.

    Before each turn every member's distribution is zeroed, and a member writes
    only the games and slots it acted for, so the sum is the distribution of
    the member that acted in each game and slot.
    """
    nodes = [
        node
        for node in jax.tree.leaves(memory, is_leaf=_is_oracle_memory)
        if isinstance(node, OracleMemory)
    ]
    if not nodes:
        raise ValueError("An oracle team needs its members wrapped with oracle_member")
    return functools.reduce(jnp.add, (node.probabilities for node in nodes))


class OracleStep(NamedTuple):
    """One decision of an oracle run, in the shape the training loop needs.

    Attributes
    ----------
    actions : Action
        The joint actions submitted: each slot's choice from its own turn.
    memory : SystemState
        Both teams' next memories and reset bindings.
    learning_outputs : tuple[Tree, Tree]
        Each team's learning values, per slot from its own turn.
    handed : PreviousTimestepActionObservation
        Each slot's handed rows from its own turn, leading axes (B,10).
    """

    actions: Action
    memory: SystemState
    learning_outputs: tuple[Tree, Tree]
    handed: PreviousTimestepActionObservation


def oracle_apply_systems(
    actor: System,
    opponent: System,
    memory: SystemState,
    observations: Observations,
    state: EnvironmentState,
    action_roots: Array,
    *,
    variables_a: Tree,
    variables_b: Tree,
    ranks: Array,
    teammates_only: bool,
    hand_sampled_actions: bool,
) -> OracleStep:
    """Choose both teams' actions in the oracle's order, once per training decision.

    The oracle's form of ``apply_systems`` for native JAX training batches. Both
    Systems are built from ``oracle_member`` wrappers. It first applies the
    shared runner's own reset of new games' memory, with the runner's keys, so
    that the rows written before each turn are not wiped; then it runs ten
    turns through the runner's own pair call, each with the same memory, keys
    and variables, and keeps each slot's result from its own turn.

    Parameters
    ----------
    actor, opponent : System
        Team A's and Team B's Systems, built from wrapped members.
    memory : SystemState
        The latest memories, as ``apply_systems`` takes them.
    observations, state
        The current native decision (leading axis B).
    action_roots : Array
        The B lane action keys the training loop gives ``apply_systems``.
    variables_a, variables_b : Tree
        Each team's dynamic variables, as for ``apply_systems``.
    ranks : Array
        Int32 (B,10) turn ranks from ``turn_order``.
    teammates_only, hand_sampled_actions : bool
        Static; see ``choose_in_order``.

    Returns
    -------
    OracleStep
        The actions to submit, the next memory, the learning values and the
        handed rows.

    Notes
    -----
    With every row zero and a zero-started handed weight, each slot's call
    computes exactly what ``apply_systems`` computes (proved by the switch's
    off-equivalence test). Pure JAX; works inside jit and scan.
    """
    execution_a, execution_b = _execution(actor), _execution(opponent)
    ids, generations = state.episode_id, state.reset_generation
    reset = (generations != memory.reset_generation) & ~state.done.done
    fresh: list[Tree] = []
    for team, (execution, old, variables) in enumerate(
        (
            (execution_a, memory.team_a, variables_a),
            (execution_b, memory.team_b, variables_b),
        )
    ):
        keys = _action_keys(action_roots, ids, team)
        init_keys = (
            _initialization_keys(memory.init_key, ids, team)
            if execution.init is not None
            else keys
        )
        fresh.append(
            _reset_for_decision(
                execution,
                variables,
                memory.adapter_templates[team],
                old,
                system_inputs(observations, state, team=team),
                init_keys,
                reset,
            )
        )
    ready = memory._replace(
        team_a=fresh[0], team_b=fresh[1], reset_generation=generations
    )
    calls: list[tuple[Action, SystemState]] = []

    def pair(
        rows_a: PreviousTimestepActionObservation,
        rows_b: PreviousTimestepActionObservation,
    ) -> tuple[_ActorStep, _ActorStep]:
        """Run one turn of both teams through the shared runner's pair call."""
        actions, updated, learning = _apply_system_pair(
            execution_a,
            execution_b,
            variables_a,
            variables_b,
            ready._replace(
                team_a=_hand(ready.team_a, rows_a), team_b=_hand(ready.team_b, rows_b)
            ),
            observations,
            state,
            action_roots,
        )
        calls.append((actions, updated))
        indices = encode_actions(
            ActorAction(actions.move, actions.select_target, actions.use_ultimate)
        )
        steps = tuple(
            _ActorStep(
                indices[
                    :, team * MAX_AGENTS_PER_TEAM : (team + 1) * MAX_AGENTS_PER_TEAM
                ],
                team_memory,
                jnp.zeros((indices.shape[0], MAX_AGENTS_PER_TEAM), jnp.float32),
                _acted_probabilities(team_memory),
                team_learning,
            )
            for team, (team_memory, team_learning) in enumerate(
                ((updated.team_a, learning[0]), (updated.team_b, learning[1]))
            )
        )
        return steps[0], steps[1]

    choices = choose_in_order(
        state.core_state,
        ranks,
        state.core_state.alive_mask,
        pair,
        teammates_only=teammates_only,
        hand_sampled_actions=hand_sampled_actions,
    )
    chosen = decode_actions(choices.indices)
    return OracleStep(
        Action(chosen.move, chosen.select_target, chosen.use_ultimate),
        calls[-1][1],
        choices.learning_outputs,
        choices.handed,
    )
