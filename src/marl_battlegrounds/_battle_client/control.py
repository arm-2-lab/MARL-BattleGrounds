"""Apply debugger drafts and scripted commands through the shared simulator boundary.

The input layer calls these helpers to build immutable ``DebuggerSession`` values.
Draft/selection helpers change host controls only. ``agent_pov_recipient_slot``
is the one owner of whose eyes Agent POV shows (the last Manual champion you
focused, or in Policies Only the champion you look at). Submission builds both teams'
actions before one Core step, then captures its matching state, observation, mask
and transition record. Reset helpers create a new recorded episode. Random keys
remain explicit in the session; this module does not write replay files or serve
HTTP requests.

Every Submit's Core step runs compiled, for manual, scripted and policy
Submits alike. SharedObs policy teams (ALPHA, BETA, GAMMA and Random) also run
the policy executor compiled. This module wraps the real ``apply_policies`` and
``step`` in ``jax.jit`` once, at import. JAX reuses a compiled program only
while the controller pair and every array shape and dtype stay the same.
Run uncompiled, one Submit took seconds, because every array operation ran on
its own. NoSharedObs Random still runs uncompiled through
``execute_no_shared_obs_team_policy``; it is fast once JAX has warmed up. A
replaced executor or step (for example a test spy) is called as given,
uncompiled.

Loaded Systems (saved checkpoints, the built-in names and your own JAX
methods) run through the same compiled pair program that evaluation uses, when
both teams' methods run in JAX. Uncompiled, JAX rebuilt and recompiled the
whole model on every turn (about 3 to 4 s per warm turn on CPU with 7b+pool).
Weights, memory, inputs and keys stay dynamic, so later turns and Restart reuse
the program. A pair with a host method (for example an LLM) runs uncompiled, as
evaluation does; its JAX opponent stays compiled inside that shared route.

``warm_compiled_turn`` runs one Submit and throws it away, so the launchers and
a DevClient setup change compile before anyone presses Lock In.

Every compiled call passes its inputs through ``_committed`` first. JAX keys
a compiled program on whether each input array is committed to a device; the
first frame's arrays are not, later frames' arrays are, so without it every
program compiled a second time on the second turn.

Compile cost: the first Submit in a DevClient session (the first in this
Python process) compiles for several seconds, about 6 s on CPU. The first
Submit after switching to a controller pair not yet used in this process
compiles the executor again, about 3 s. After that a Submit takes about
0.03 s. With 7b+pool against TDM-Beta (1 Oct 2026): the first Lock In about
4 to 7 s on CPU and 12.5 s on the RTX 5090, every later one 0.02 s, where the
uncompiled route took 3 to 5 s on CPU and 5 to 7 s on the GPU per turn. These
CPU and GPU times are examples, not guarantees. The old
built-in command route holds the service lock during execution. Queued System
callbacks run outside HTTP, router and service locks; adoption briefly locks
the accepted state. The public launcher drains its queue on the caller thread.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds._battle_client.evaluation_bridge import (
    DebuggerCaptureProfileV1,
    DebuggerEvaluationLaunchSpecificationV1,
    build_debugger_evaluation_context_v1,
    build_debugger_evaluation_launch_specification_v1,
    debugger_action_source_kind_v1,
)
from marl_battlegrounds._battle_client.live_presentation import (
    advance_live_match_history_v1,
)
from marl_battlegrounds._battle_client.model import (
    SCENARIO_OPPONENT_ERROR,
    DebuggerScenario,
    DebuggerSession,
    Lane,
    LaneAvailability,
    PendingAction,
    ScenarioFrame,
    SubmissionKind,
    TeamBController,
    TeamController,
    is_system_controller,
    is_team_controller,
    normalize_slot_ownership,
)
from marl_battlegrounds.core import env as core_env
from marl_battlegrounds.core.config import validate_product_env_config
from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MOVE_STAY,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    TEAM_A_ID,
    TEAM_B_ID,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.environment import (
    EnvironmentState,
    _checked_increment,  # pyright: ignore[reportPrivateUsage] - shared counter owner
    make,
)
from marl_battlegrounds.evaluation import policy_execution
from marl_battlegrounds.evaluation._metrics.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation._recording.capture import (
    capture_evaluation_transition_unit_v3,
    capture_initial_evaluation_frame_v3,
)
from marl_battlegrounds.evaluation.models import (
    ActionMaskV1,
    EvaluationEpisodeContextV4,
    EvaluationFrameV3,
    ExecutionInformationMode,
)
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTree,
    System,
    SystemState,
    apply_policies,
    init_systems,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
    team_controller_names,
)
from marl_battlegrounds.evaluation.system_evaluation import freeze_evaluation_method
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.input import ActorInput, Observations
from marl_battlegrounds.policies.no_shared_obs import (
    execute_no_shared_obs_team_policy,
)
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.policies.shared_obs import (
    build_default_shared_obs_information_availability,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    advance_status_source_evidence_v2,
    initialize_status_source_evidence_v2,
)
from marl_battlegrounds.viewer.champion_streaks import ChampionTakedownsV1
from marl_battlegrounds.viewer.match_moments import MatchMomentHistoryV1

# Compiled forms of the real policy executor and Core step. Controller callables
# are static (the registered adapters are stable objects), so each Team A/Team B
# controller pair gets its own program. Observations, masks, keys, states and
# manual actions stay dynamic: new frames, maps and seeds reuse a program only
# while every array shape and dtype stays the same; any change compiles again.
# These functions keep their compile cache for the whole process, so a test that
# patches code inside the compiled path must replace control.apply_policies or
# control.step; patching deeper internals would not reach a cached program.
_COMPILED_APPLY_POLICIES = cast(
    Callable[..., tuple[Action, PolicyTree, PolicyTree]],
    jax.jit(policy_execution.apply_policies, static_argnums=(0, 1)),
)
_COMPILED_STEP = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(core_env.step),
)
# The shared System pair authority, compiled the way evaluation compiles it:
# the two call-structure descriptors are static, everything numerical is dynamic.
_COMPILED_SYSTEM_PAIR = cast(
    Callable[..., tuple[Action, SystemState, tuple[PolicyTree, PolicyTree]]],
    jax.jit(
        policy_execution._apply_system_pair,  # pyright: ignore[reportPrivateUsage]
        static_argnums=(0, 1),
        static_argnames=("keep_learning_outputs",),
    ),
)


def _committed[T](tree: T) -> T:
    """Commit every JAX array in tree to its device so a program compiles once.

    A compiled program's cache key includes whether each input array is
    committed to a device. Arrays made outside a compiled call (the first frame,
    fresh memory) are uncommitted; a compiled call's outputs are committed when
    any input was (weights loaded from a checkpoint are). Without this step
    every compiled program here compiled twice: once on the first turn and
    again on the second. Only the real compiled programs' inputs pass through
    it; a replaced executor or step (a test spy) receives its inputs unchanged.

    Parameters
    ----------
    tree : T
        A pytree of compiled-call inputs that this client or a JAX method owns.
        Never pass a host method's variables or memory: a host method receives
        its own objects exactly as given. Uncommitted JAX arrays (including
        typed keys) are committed on the device they are already on. Committed
        arrays, NumPy arrays, Python scalars, None and other leaves pass through
        as the same objects, so weak types and static values keep their meaning.

    Returns
    -------
    T
        ``tree`` itself when nothing needed committing (the usual case after the
        first turn), otherwise the same structure with committed arrays. Values,
        dtypes and shapes never change; no array is copied to another device.
    """
    if not any(
        isinstance(leaf, jax.Array) and not leaf.committed
        for leaf in jax.tree.leaves(tree)
    ):
        return tree

    def commit(leaf: object) -> object:
        """Commit one uncommitted JAX array where it is; keep anything else."""
        if isinstance(leaf, jax.Array) and not leaf.committed:
            return jax.device_put(leaf, leaf.sharding)
        return leaf

    return cast(T, jax.tree.map(commit, tree))


type DebuggerTransitionFailureStageV1 = Literal[
    "action_build",
    "simulation",
    "capture",
    "validation",
]
type DebuggerTransitionFailureCodeV1 = Literal[
    "interactive_action_build_failed",
    "policy_action_build_failed",
    "scripted_action_build_failed",
    "invalid_submitted_action",
    "simulator_step_failed",
    "transition_capture_failed",
    "transition_packaging_failed",
]


class DebuggerTransitionFailureV1(RuntimeError):  # noqa: N818 - frozen protocol name
    """Stable submission-stage failure without leaking raw exception detail."""

    __slots__ = ("stable_code", "stage")

    stage: DebuggerTransitionFailureStageV1
    stable_code: DebuggerTransitionFailureCodeV1

    def __init__(
        self,
        stage: DebuggerTransitionFailureStageV1,
        stable_code: DebuggerTransitionFailureCodeV1,
    ) -> None:
        """Retain a stable failure stage/code without publishing raw exception
        details.
        """
        if stage not in ("action_build", "simulation", "capture", "validation"):
            raise ValueError("unknown debugger transition failure stage")
        if stable_code not in (
            "interactive_action_build_failed",
            "policy_action_build_failed",
            "scripted_action_build_failed",
            "invalid_submitted_action",
            "simulator_step_failed",
            "transition_capture_failed",
            "transition_packaging_failed",
        ):
            raise ValueError("unknown debugger transition failure code")
        self.stage = stage
        self.stable_code = stable_code
        super().__init__(f"debugger transition failed during {stage}")


def _transition_failure(
    stage: DebuggerTransitionFailureStageV1,
    stable_code: DebuggerTransitionFailureCodeV1,
    error: Exception,
) -> DebuggerTransitionFailureV1:
    """Convert a caught failure into the public stage/code pair; omit the raw
    message.
    """
    del error
    return DebuggerTransitionFailureV1(stage, stable_code)


def make_neutral_joint_action() -> Action:
    """Create a fixed-size joint action with no movement, target or Ultimate use.

    Returns
    -------
    Action
        Three int32 JAX arrays shaped [10], using Stay and zero combat categories.
        This initializes a request; current masks still determine submitted-action
        acceptance when the request reaches Core.
    """
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _validate_active_context_slot(
    context: EvaluationEpisodeContextV4,
    global_slot: int,
    *,
    name: str,
) -> None:
    """Require a global slot in range whose recorded roster row is active."""
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        raise ValueError(f"{name} must be in [0, {MAX_AGENT_SLOTS}).")
    if not context.roster[global_slot].configured_active:
        raise ValueError(f"{name} g{global_slot} is inactive.")


def _active_context_slots(session: DebuggerSession) -> tuple[int, ...]:
    """Return active slots from the already-host canonical roster."""
    return tuple(
        row.global_slot
        for row in session.evaluation_context.roster
        if row.configured_active
    )


def _target_action_from_context(
    context: EvaluationEpisodeContextV4,
    actor_global_slot: int,
    target_global_slot: int | None,
) -> int:
    """Resolve one target category through the serialized catalog authority."""
    if target_global_slot is None:
        return 0
    catalog = context.static_mechanics_catalog
    mapping = catalog.global_recipient_slot_by_actor_and_target_action[
        actor_global_slot
    ]
    try:
        return mapping.index(target_global_slot)
    except ValueError as error:
        raise ValueError("target is absent from the serialized action axis") from error


def lane_availability(
    action_mask: ActionMask | ActionMaskV1,
    actor_global_slot: int,
    target_action: int,
    armed_lane: Lane | None,
) -> LaneAvailability:
    """Read exact lane availability from the authoritative joint mask.

    Parameters
    ----------
    action_mask : ActionMask | ActionMaskV1
        Current simulator mask or its recorded ActionMaskV1 representation.
    actor_global_slot : int
        Global actor index from zero through nine.
    target_action : int
        Target category on the actor-relative fixed target axis.
    armed_lane : Lane | None
        0 selects Basic, 1 selects Ultimate, and None means neither is armed.

    Returns
    -------
    LaneAvailability
        Both exact lane flags and whether the selected target/lane pair is
        legal.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
    if not 0 <= actor_global_slot < MAX_AGENT_SLOTS:
        msg = f"actor_global_slot must be in [0, {MAX_AGENT_SLOTS})."
        raise ValueError(msg)
    if not 0 <= target_action < NUM_TARGET_ACTIONS:
        msg = f"target_action must be in [0, {NUM_TARGET_ACTIONS})."
        raise ValueError(msg)
    if armed_lane not in (None, 0, 1):
        msg = f"armed_lane must be None, 0, or 1; got {armed_lane}."
        raise ValueError(msg)
    if type(action_mask) is ActionMaskV1:
        lane_values = action_mask.select_target_use_ultimate_joint_mask[
            actor_global_slot
        ][target_action]
    else:
        core_mask = cast(ActionMask, action_mask)
        lane_values = core_mask.select_target_use_ultimate_joint_mask[
            actor_global_slot,
            target_action,
        ]
    lane_0_available = bool(lane_values[0])
    lane_1_available = bool(lane_values[1])
    armed_pair_legal = (
        False
        if armed_lane is None
        else lane_0_available
        if armed_lane == 0
        else lane_1_available
    )
    return LaneAvailability(
        target_action=target_action,
        lane_0_available=lane_0_available,
        lane_1_available=lane_1_available,
        armed_lane=armed_lane,
        armed_pair_legal=armed_pair_legal,
    )


def _default_pending_actions(
    context: EvaluationEpisodeContextV4,
    action_mask: ActionMask | ActionMaskV1,
) -> tuple[PendingAction, ...]:
    """Build one exact fixed-slot draft tuple for a fresh decision epoch."""
    pending_actions: list[PendingAction] = []
    for actor_slot in range(MAX_AGENT_SLOTS):
        if not context.roster[actor_slot].configured_active:
            pending_actions.append(PendingAction(armed_lane=None, arm_origin=None))
            continue
        basic_available = lane_availability(
            action_mask,
            actor_slot,
            0,
            0,
        ).lane_0_available
        pending_actions.append(
            PendingAction(
                armed_lane=0 if basic_available else None,
                arm_origin="automatic" if basic_available else None,
            )
        )
    return tuple(pending_actions)


def _replace_controlled_pending_action(
    session: DebuggerSession,
    pending_action: PendingAction,
) -> DebuggerSession:
    """Replace only the controlled actor's row in the immutable draft tuple."""
    if (
        session.scenario.mode != "scripted"
        and session.controlled_global_slot not in session.manual_slots
    ):
        return session
    pending_actions = list(session.pending_actions)
    pending_actions[session.controlled_global_slot] = pending_action
    return replace(session, pending_actions=tuple(pending_actions))


def build_interactive_joint_action(
    context: EvaluationEpisodeContextV4,
    pending_actions: tuple[PendingAction, ...],
    *,
    actor_global_slots: tuple[int, ...],
) -> Action:
    """Build one authorized joint request without pre-filtering it by the mask.

    Parameters
    ----------
    context : EvaluationEpisodeContextV4
        Recorded episode roster and target-action mapping.
    pending_actions : tuple[PendingAction, ...]
        Exactly ten fixed-slot pending rows from the current decision.
    actor_global_slots : tuple[int, ...]
        Nonempty tuple of distinct active actors authorized for this
        submission.

    Returns
    -------
    Action
        Fixed-slot joint request. Actions are not prefiltered by masks; Core
        records acceptance.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
    if len(pending_actions) != MAX_AGENT_SLOTS:
        msg = (
            f"pending_actions must contain {MAX_AGENT_SLOTS} fixed-slot rows; "
            f"got {len(pending_actions)}."
        )
        raise ValueError(msg)
    if not actor_global_slots:
        raise ValueError("actor_global_slots must contain at least one active actor.")
    if len(actor_global_slots) != len(set(actor_global_slots)):
        raise ValueError("actor_global_slots must not contain duplicates.")

    action = make_neutral_joint_action()
    move = action.move
    target = action.select_target
    ultimate = action.use_ultimate
    for actor_slot in actor_global_slots:
        _validate_active_context_slot(
            context,
            actor_slot,
            name="submission actor",
        )
        pending_action = pending_actions[actor_slot]
        move = move.at[actor_slot].set(pending_action.move_action)
        if pending_action.armed_lane is None:
            continue
        if pending_action.selected_global_target_slot is not None:
            _validate_active_context_slot(
                context,
                pending_action.selected_global_target_slot,
                name="pending target",
            )
        target_action = _target_action_from_context(
            context,
            actor_slot,
            pending_action.selected_global_target_slot,
        )
        target = target.at[actor_slot].set(target_action)
        ultimate = ultimate.at[actor_slot].set(pending_action.armed_lane)
    return Action(move=move, select_target=target, use_ultimate=ultimate)


def _fixed_key_coordinates(session: DebuggerSession) -> tuple[Array, Array]:
    """Return the existing evaluator's one-lane seed and local-step coordinates.

    session belongs to a fixed scenario and its key is the unchanged evaluator
    root. The episode seed comes from its recorded schedule, never run_generation.
    Both returned int arrays have shape [1]; no random draw happens here.
    """
    return (
        jnp.asarray(
            [_required_seed(session.evaluation_context.seed_protocol.episode_seed)],
            dtype=jnp.uint32,
        ),
        jnp.asarray([session.current_evaluation_frame.frame_index], dtype=jnp.int32),
    )


def _policy_keys(session: DebuggerSession) -> Array:
    """Derive role-correct actor keys without consuming the environment stream."""
    if session.scenario.scenario_specification is not None:
        from marl_battlegrounds.evaluation.evaluate import (
            _actor_keys,  # pyright: ignore[reportPrivateUsage]
        )

        seed_ids, local_steps = _fixed_key_coordinates(session)
        return _actor_keys(session.key, seed_ids, local_steps)[0]
    context = session.evaluation_context
    seeds = context.seed_protocol
    frame_index = session.current_evaluation_frame.frame_index

    def keys_for_seed(seed: object, *, role: str) -> Array:
        """Derive this decision's actor keys from the recorded role-specific seed."""
        if type(seed) is not int:
            raise ValueError(f"{role} actors require a policy seed")
        decision_key = jax.random.fold_in(jax.random.key(seed), frame_index)
        return jax.random.split(decision_key, num=MAX_AGENT_SLOTS)

    focal_slot = session.scenario.default_controlled_slot
    focal_team_id = context.roster[focal_slot].configured_team_id
    focal_keys = keys_for_seed(seeds.focal_policy_seed, role="focal")
    actor_keys = focal_keys
    cooperative_keys: Array | None = None
    adversarial_keys: Array | None = None
    for row in context.roster:
        if not row.configured_active or row.global_slot == focal_slot:
            continue
        if row.configured_team_id == focal_team_id:
            if cooperative_keys is None:
                cooperative_keys = keys_for_seed(
                    seeds.cooperative_partner_seed,
                    role="cooperative",
                )
            role_keys = cooperative_keys
        else:
            if adversarial_keys is None:
                adversarial_keys = keys_for_seed(
                    seeds.adversarial_opponent_seed,
                    role="adversarial",
                )
            role_keys = adversarial_keys
        actor_keys = actor_keys.at[row.global_slot].set(role_keys[row.global_slot])
    return actor_keys


def _required_seed(seed: int | None) -> int:
    """Require a recorded seed value before deriving the next random-key stream."""
    if seed is None:
        raise ValueError("Live debugger execution requires a recorded seed")
    return seed


def _manual_policy(
    variables: PolicyTree,
    carry: PolicyTree,
    actor: ActorInput,
    mask: ActionMask,
    key: Array,
) -> tuple[ActorAction, PolicyTree]:
    """Return a neutral scalar row without reading or receiving human drafts.

    Inputs follow the existing Policy contract. Carry is returned unchanged;
    no randomness is consumed. The turn owner merges the human's owned rows
    after both methods have chosen from the same permitted decision inputs.
    """
    del variables, actor, mask, key
    zero = jnp.asarray(0, dtype=jnp.int32)
    return ActorAction(zero, zero, zero), carry


def _configured_policy(
    session: DebuggerSession, team_identity: int, controller: TeamBController
) -> Policy:
    """Adapt a built-in choice; Manual has no access to staged human actions."""
    del session, team_identity
    if controller == "manual":
        return Policy("manual", _manual_policy)
    return policy(
        {
            "reactive_tdm": "tdm-alpha",
            "random_valid": "random",
            "scenario_5": "tdm-beta",
            "tdm_gamma": "tdm-gamma",
        }[controller]
    )


def _prepare_session_systems(
    controllers: tuple[TeamController, TeamController],
    supplied: tuple[System | None, System | None] | None,
    *,
    borrow_systems: bool = False,
) -> tuple[System, System] | None:
    """Freeze declared methods once and adapt built-ins only for a System pair.

    supplied follows Team A/B order. Each system: alias needs its actual System.
    Built-ins use their existing Policy implementations. Manual stays neutral;
    the turn owner later merges only its configured human-owned rows. This
    creates no client scope and makes no model request. borrow_systems=True keeps
    supplied descriptors and parameter leaves unchanged; the caller owns their
    lifetime and must keep their external behavior fixed throughout the launch.
    """
    if type(borrow_systems) is not bool:
        raise TypeError("borrow_systems must be a Python bool")
    if not all(is_team_controller(value) for value in controllers):
        raise ValueError("Choose a built-in controller or a declared System alias")
    if not any(is_system_controller(value) for value in controllers):
        if supplied is not None and any(value is not None for value in supplied):
            raise ValueError("Supplied Systems need a declared System controller")
        return None
    if supplied is None:
        raise ValueError("A declared System controller needs its loaded method")
    methods: list[System] = []
    prepared: dict[int, System] = {}
    for controller, method in zip(controllers, supplied, strict=True):
        if is_system_controller(controller):
            if not isinstance(method, System):
                raise ValueError("A declared System controller needs a System")
            if id(method) not in prepared:
                prepared[id(method)] = (
                    method
                    if borrow_systems
                    else cast(System, freeze_evaluation_method(method))
                )
            methods.append(prepared[id(method)])
        elif controller == "manual":
            methods.append(shared_policy(Policy("manual", _manual_policy)))
        else:
            methods.append(
                shared_policy(
                    policy(
                        {
                            "reactive_tdm": "tdm-alpha",
                            "random_valid": "random",
                            "scenario_5": "tdm-beta",
                            "tdm_gamma": "tdm-gamma",
                        }[controller]
                    )
                )
            )
    return methods[0], methods[1]


def _system_slot_names(
    controllers: tuple[TeamController, TeamController],
    systems: tuple[System, System] | None,
    config: EnvConfig,
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """Name each champion of a System-alias team from its loaded System.

    A team chosen by alias (``system:counter``) would otherwise be labeled,
    and recorded in replays, by that alias. This helper gives
    ``normalize_slot_ownership`` the real names instead.

    Parameters
    ----------
    controllers : tuple of two TeamController
        Team A and Team B controller kinds or System aliases.
    systems : tuple of two System, or None
        The session's prepared Team A/B methods (``_prepare_session_systems``),
        or None when neither team uses a System alias.
    config : EnvConfig
        The match configuration; only its ten class IDs are read.

    Returns
    -------
    tuple of two (tuple of five str or None)
        In Team A/B order: a System-alias team's five names in team-local slot
        order, from ``team_controller_names``, so a ``team(...)`` names each
        champion by its member. None for a built-in or Manual team, and for
        both teams when ``systems`` is None.

    Notes
    -----
    Host-only setup work, run once per new session or changed setup. It reads
    names and a team's small slot table; it never calls a method.
    """
    if systems is None:
        return None, None
    class_ids = [int(value) for value in jax.device_get(config.agent_profile.class_ids)]
    first, second = (
        team_controller_names(system, class_ids[team * 5 : team * 5 + 5])
        if is_system_controller(controller)
        else None
        for team, (controller, system) in enumerate(
            zip(controllers, systems, strict=True)
        )
    )
    return first, second


def _session_system_registrations(
    methods: tuple[System, System] | None,
) -> tuple[tuple[str, dict[str, object]], tuple[str, dict[str, object]]] | None:
    """Register the two frozen setup methods once, without reading live memory."""
    if methods is None:
        return None
    first = normalize_system_registration(methods[0], phase="evaluation", frozen=True)
    second = (
        first
        if methods[0] is methods[1]
        else normalize_system_registration(methods[1], phase="evaluation", frozen=True)
    )
    return first, second


def _initial_system_state(
    systems: tuple[System, System] | None,
    config: EnvConfig,
    state: EnvState,
    observation: Observation,
    action_mask: ActionMask,
    availability: Array | None,
    context: EvaluationEpisodeContextV4,
    registrations: tuple[tuple[str, dict[str, object]], tuple[str, dict[str, object]]]
    | None,
    *,
    numeric_episode_id: int = 1,
) -> tuple[EnvironmentState | None, SystemState | None, tuple[str, str] | None]:
    """Pair the existing raw snapshot with fresh public System memory, once.

    No transition occurs. numeric_episode_id defaults to 1 for old diagnostics;
    fixed scenarios use the first case ID from the existing qualification plan.
    Their reset and initialization keys use the evaluator's streams 0 and 3.
    Other inputs are the paired raw view, recorded context and frozen registrations.
    The wrapper retains the exact raw objects and permitted source matrix.
    The caller owns any resource scopes needed by custom initializers.
    """
    if systems is None:
        return None, None, None
    if availability is None:
        raise ValueError("Systems require SharedObs source permissions")
    env = make("tdm", metrics="none", balance_spawn_locations=False)
    if context.seed_protocol.seed_protocol.identifier == "episode-fold-in-v1":
        from marl_battlegrounds.evaluation.evaluate import episode_keys

        root = jax.random.key(_required_seed(context.seed_protocol.root_seed))
        seed_ids = jnp.asarray(
            [_required_seed(context.seed_protocol.episode_seed)], dtype=jnp.uint32
        )
        local_steps = jnp.zeros(1, dtype=jnp.int32)
        reset_key = episode_keys(root, seed_ids, local_steps, 0)[0]
        init_key = episode_keys(root, seed_ids, local_steps, 3)
    else:
        reset_key = init_key = jax.random.key(
            _required_seed(context.seed_protocol.focal_policy_seed)
        )
    _, wrapped = env.reset(
        reset_key,
        env_config=config,
        initial=(state, observation, action_mask),
        episode_id=numeric_episode_id,
    )
    wrapped = wrapped._replace(
        config=config,
        core_state=state,
        observation=observation,
        action_mask=action_mask,
        source_availability=availability,
    )
    memory = init_systems(*systems, env.get_observations(wrapped), wrapped, init_key)
    assert registrations is not None
    identifiers = (registrations[0][0], registrations[1][0])
    return wrapped, memory, (identifiers[0], identifiers[1])


def _build_system_joint_action(session: DebuggerSession) -> tuple[Action, SystemState]:
    """Choose one complete joint action and stage memory without changing the game.

    No method receives Manual drafts. The shared System pair authority (the one
    behind public apply_systems) filters each actor's information and keeps the
    existing whole-System batch. When both methods run in JAX it runs as the
    compiled pair program (see the module description); otherwise uncompiled.
    Memory is adopted only after submit_joint_action packages its successor.

    Parameters
    ----------
    session : DebuggerSession
        Session with loaded systems, their wrapped environment state and memory.
        Fixed scenarios use the evaluator's exact scheduled keys; other games
        fold the focal policy seed with the current frame index.

    Returns
    -------
    tuple[Action, SystemState]
        Both teams' joint action and the proposed successor memory. The game
        itself does not change.

    Raises
    ------
    ValueError
        The session has no systems, environment state or memory, or a method
        returns an invalid action or memory layout (from the pair authority).
    """
    methods, wrapped, memory = (
        session.systems,
        session.environment_state,
        session.system_memory,
    )
    if methods is None or wrapped is None or memory is None:
        raise ValueError("System state is missing")
    execution_a = policy_execution._execution(methods[0])  # pyright: ignore[reportPrivateUsage]
    execution_b = policy_execution._execution(methods[1])  # pyright: ignore[reportPrivateUsage]
    host_a, host_b = execution_a.execution == "host", execution_b.execution == "host"
    apply_pair = (
        policy_execution._apply_system_pair  # pyright: ignore[reportPrivateUsage]
        if host_a or host_b
        else _COMPILED_SYSTEM_PAIR
    )
    # Commit what this client and the JAX methods own. A host method receives
    # its own variables and memory exactly as given (the same objects).
    observations, wrapped = _committed(
        (Observations(wrapped.observation, wrapped.source_availability), wrapped)
    )
    variables_a = methods[0].variables if host_a else _committed(methods[0].variables)
    variables_b = methods[1].variables if host_b else _committed(methods[1].variables)
    memory = memory._replace(
        team_a=memory.team_a if host_a else _committed(memory.team_a),
        team_b=memory.team_b if host_b else _committed(memory.team_b),
        episode_id=_committed(memory.episode_id),
        reset_generation=_committed(memory.reset_generation),
        init_key=_committed(memory.init_key),
        policy_trace=_committed(memory.policy_trace),
        adapter_templates=_committed(memory.adapter_templates),
    )
    if session.scenario.scenario_specification is not None:
        from marl_battlegrounds.evaluation.evaluate import (
            _actor_keys,  # pyright: ignore[reportPrivateUsage]
            episode_keys,
        )

        seed_ids, local_steps = _fixed_key_coordinates(session)
        action_roots, actor_keys = _committed(
            (
                episode_keys(session.key, seed_ids, local_steps, 2),
                _actor_keys(session.key, seed_ids, local_steps),
            )
        )
        action, proposed, _ = apply_pair(
            execution_a,
            execution_b,
            variables_a,
            variables_b,
            memory,
            observations,
            wrapped,
            action_roots,
            actor_keys=actor_keys,
            keep_learning_outputs=False,
        )
        return action, proposed
    key = jax.random.fold_in(
        jax.random.key(
            _required_seed(session.evaluation_context.seed_protocol.focal_policy_seed)
        ),
        session.current_evaluation_frame.frame_index,
    )
    # The same arguments public apply_systems passes, including its default
    # keep_learning_outputs=True, so the actions match the uncompiled route.
    action, proposed, _ = apply_pair(
        execution_a,
        execution_b,
        variables_a,
        variables_b,
        memory,
        observations,
        wrapped,
        _committed(key),
    )
    return action, proposed


def _build_configured_joint_action(session: DebuggerSession) -> Action:
    """Resolve both teams before the single existing simulator step.

    SharedObs uses the public five-argument application authority, compiled
    with one program per controller pair, reused only while array shapes and
    dtypes stay the same (see the module description); a replaced executor
    runs uncompiled. Legacy NoSharedObs Random keeps its original
    three-argument, no-source-bank ABI and runs uncompiled.
    """
    controllers = (session.team_a_controller, session.team_b_controller)
    if controllers == ("manual", "manual"):
        raise ValueError("configured policy assembly requires one policy team")
    mode = session.evaluation_context.execution_information_mode
    availability = _captured_information_availability(session)
    keys = _policy_keys(session)
    if mode == "shared_obs":
        if availability is None:
            raise ValueError("SharedObs requires captured information availability")
        first = _configured_policy(session, TEAM_A_ID, controllers[0])
        second = _configured_policy(session, TEAM_B_ID, controllers[1])
        compiled = apply_policies is policy_execution.apply_policies
        executor = _COMPILED_APPLY_POLICIES if compiled else apply_policies
        inputs = (
            first.variables,
            second.variables,
            first.initial_carry,
            second.initial_carry,
            Observations(session.observation, availability),
            session.action_mask,
            keys,
        )
        action, _, _ = executor(
            first.apply,
            second.apply,
            *(_committed(inputs) if compiled else inputs),
        )
        return action
    if mode != "no_shared_obs" or availability is not None:
        raise ValueError("unsupported execution information mode")
    actions: list[ActorAction] = []
    for team_identity, controller in zip(
        (TEAM_A_ID, TEAM_B_ID), controllers, strict=True
    ):
        if controller == "manual":
            actions.append(
                ActorAction(
                    *(jnp.zeros(MAX_AGENTS_PER_TEAM, jnp.int32) for _ in range(3))
                )
            )
        elif controller == "random_valid":
            actions.append(
                cast(
                    ActorAction,
                    execute_no_shared_obs_team_policy(
                        session.observation,
                        session.action_mask,
                        keys,
                        policy=random_policy,
                        team_identity=team_identity,
                    ),
                )
            )
        else:
            raise ValueError("unsupported NoSharedObs team controller")
    return build_joint_action_from_actor_actions(actions[0], actions[1])


def build_scripted_joint_action(
    context: EvaluationEpisodeContextV4,
    frame: ScenarioFrame,
) -> Action:
    """Build a potentially multi-actor scripted request from neutral defaults.

    Parameters
    ----------
    context : EvaluationEpisodeContextV4
        Recorded episode roster and target-action mapping.
    frame : ScenarioFrame
        One script frame containing distinct actor commands; omitted actors
        remain neutral.

    Returns
    -------
    Action
        Fixed-slot joint request with neutral heads for actors omitted from the
        script frame.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
    action = make_neutral_joint_action()
    move = action.move
    target = action.select_target
    ultimate = action.use_ultimate
    seen_slots: set[int] = set()
    for command in frame.commands:
        if command.actor_global_slot in seen_slots:
            msg = (
                f"frame {frame.label!r} contains duplicate command for "
                f"g{command.actor_global_slot}."
            )
            raise ValueError(msg)
        seen_slots.add(command.actor_global_slot)
        _validate_active_context_slot(
            context,
            command.actor_global_slot,
            name="command actor",
        )
        if command.target_global_slot is not None:
            _validate_active_context_slot(
                context,
                command.target_global_slot,
                name="command target",
            )
        move = move.at[command.actor_global_slot].set(command.move_action)
        target = target.at[command.actor_global_slot].set(
            _target_action_from_context(
                context,
                command.actor_global_slot,
                command.target_global_slot,
            )
        )
        ultimate = ultimate.at[command.actor_global_slot].set(command.use_ultimate)
    return Action(move=move, select_target=target, use_ultimate=ultimate)


def _fresh_snapshot(
    scenario: DebuggerScenario,
    seed: int,
) -> tuple[float, EnvConfig, EnvState, Observation, ActionMask]:
    """Validate and expose an authored scenario without replacing its state."""
    authored_config, authored_state = scenario.build_scenario()
    validate_product_env_config(authored_config)
    scenario_default_movement_scale = authored_config.ordinary_movement_distance_scale
    config = authored_config
    del seed
    state, observation, action_mask, _info = initialize_scenario_state(
        authored_state,
        config,
    )
    return (
        scenario_default_movement_scale,
        config,
        state,
        observation,
        action_mask,
    )


def _initial_information_availability(
    config: EnvConfig,
    execution_information_mode: ExecutionInformationMode,
) -> Array | None:
    """Build the episode-wide SharedObs topology exactly once per fresh epoch."""
    if execution_information_mode == "shared_obs":
        return build_default_shared_obs_information_availability(
            config.agent_profile.active_mask,
            config.agent_profile.team_ids,
        )
    if execution_information_mode == "no_shared_obs":
        return None
    raise ValueError("execution_information_mode must be shared_obs or no_shared_obs")


def _captured_information_availability(session: DebuggerSession) -> Array | None:
    """Return the exact matrix recorded on the current policy-input frame."""
    captured = session.current_evaluation_frame.shared_obs_information_availability_by_recipient_and_sensor_source  # noqa: E501
    if session.evaluation_context.execution_information_mode == "shared_obs":
        if captured is None:
            raise ValueError("SharedObs execution requires captured availability.")
        return jnp.asarray(captured, dtype=jnp.bool_)
    if captured is not None:
        raise ValueError("NoSharedObs execution must omit SharedObs availability.")
    return None


def _debugger_expected_horizon(
    scenario: DebuggerScenario,
    config: EnvConfig,
    state: EnvState,
) -> int:
    """Use the script length or remaining interactive simulator transitions."""
    if scenario.scenario_specification is not None:
        horizon = scenario.scenario_specification.horizon
        if horizon != config.max_steps - int(state.step_count):
            raise ValueError("Fixed scenario horizon must match its exact start")
        return horizon
    return (
        len(scenario.frames)
        if scenario.mode == "scripted"
        else config.max_steps - int(state.step_count)
    )


class CombatConfigurationRejectedError(ValueError):
    """An expected selection error that leaves the current session usable."""


def _validate_reactive_controller_selection(
    scenario: DebuggerScenario,
    *,
    team_a_controller: TeamController,
    team_b_controller: TeamBController,
    execution_information_mode: ExecutionInformationMode,
) -> None:
    """Check execution boundaries, independently of scenario content."""
    if not any(
        controller in ("reactive_tdm", "scenario_5", "tdm_gamma")
        or is_system_controller(controller)
        for controller in (team_a_controller, team_b_controller)
    ):
        return
    if scenario.mode != "interactive":
        raise CombatConfigurationRejectedError(
            "Reactive controllers require an interactive session."
        )
    if execution_information_mode != "shared_obs":
        raise CombatConfigurationRejectedError(
            "Reactive controllers require SharedObs."
        )


def _validate_fixed_setup(
    scenario: DebuggerScenario,
    seed: int,
    team_b_controller: TeamBController,
    information_mode: ExecutionInformationMode,
) -> int:
    """Check fixed ownership before setup callbacks; return the numeric episode ID.

    Ordinary scenarios keep ID 1. Fixed scenarios use their first existing seed
    case, SharedObs and declared ALPHA/BETA opponent. No content or method is loaded.
    """
    identifier = scenario.packaged_scenario_id
    if identifier is None:
        return 1
    from marl_battlegrounds.modes.team_deathmatch.scenarios import (
        TDM_BETA_SCENARIO_IDS,
        _pressure_identity,  # pyright: ignore[reportPrivateUsage]
        build_tdm_qualification_seed_schedule,
    )

    specification = scenario.scenario_specification
    assert specification is not None
    expected = "scenario_5" if identifier in TDM_BETA_SCENARIO_IDS else "reactive_tdm"
    if team_b_controller != expected:
        raise CombatConfigurationRejectedError(SCENARIO_OPPONENT_ERROR)
    if seed != 0 or information_mode != "shared_obs":
        raise CombatConfigurationRejectedError(
            "Fixed scenarios keep their seed, SharedObs and Team B opponent"
        )
    if specification.seed_schedule.realized_seed_protocols[
        0
    ] != build_tdm_qualification_seed_schedule().realized_seed_protocols[
        0
    ] or specification.pressure_protocol != _pressure_identity(identifier):
        raise CombatConfigurationRejectedError(
            "Fixed scenario seed and pressure metadata must match "
            "the installed contract"
        )
    return scenario.numeric_episode_id


def _validate_fixed_initial_frame(
    scenario: DebuggerScenario, frame: EvaluationFrameV3
) -> None:
    """Check the exact authored start before initializing any supplied method."""
    specification = scenario.scenario_specification
    if specification is not None:
        from marl_battlegrounds.evaluation.scenario import (
            resolved_initial_state_digest_sha256,
        )

        if (
            resolved_initial_state_digest_sha256(
                frame.simulator_step_count, frame.snapshot
            )
            != specification.resolved_initial_state_digest_sha256
        ):
            raise ValueError(
                "Fixed scenario initial state differs from its specification"
            )


def _default_focus_slot(
    scenario: DebuggerScenario,
    manual_slots: tuple[int, ...] | None,
) -> int:
    """Return the champion a new or restarted match starts focused on.

    scenario.default_controlled_slot, unless the match has Manual slots and
    that slot is not one of them: then the lowest Manual slot, so a player who
    plays one champion starts giving orders to it rather than looking at a
    policy's champion (Watch Only). manual_slots are global slots 0-9 (None or
    empty means no Manual slot). Pure. The policy seed's focal slot stays
    scenario.default_controlled_slot.
    """
    default_slot = scenario.default_controlled_slot
    if manual_slots and default_slot not in manual_slots:
        return min(manual_slots)
    return default_slot


def create_session(
    scenario: DebuggerScenario,
    *,
    seed: int,
    evaluation_launch_specification: DebuggerEvaluationLaunchSpecificationV1,
    controlled_global_slot: int | None,
    show_ranges: bool,
    verbose_logging: bool,
    team_a_controller: TeamController = "manual",
    team_b_controller: TeamBController = "manual",
    execution_information_mode: ExecutionInformationMode = "no_shared_obs",
    systems: tuple[System | None, System | None] | None = None,
    borrow_systems: bool = False,
    manual_slots: tuple[int, ...] | None = None,
    slot_labels: tuple[str | None, ...] | None = None,
    run_generation: int = 0,
) -> DebuggerSession:
    """Create one deterministic immutable debugger session.

    Defaults keep low-level diagnostics on their former manual, NoSharedObs
    contract while live launchers pass the researcher-facing values explicitly.
    Fixed packaged scenarios require root seed 0, SharedObs and their declared
    Team B controller. Their exact config/start is checked before method init.

    Parameters
    ----------
    scenario : DebuggerScenario
        Factory and metadata for the authored initial state or scripted
        demonstration.
    seed : int
        Root seed, which must equal the launch specification root seed.
    evaluation_launch_specification : DebuggerEvaluationLaunchSpecificationV1
        Recorded identity, seed and capture choices for the new episode.
    controlled_global_slot : int | None
        Requested zero-based active actor slot (the focus), or None for the
        default: the scenario's default slot, or the lowest Manual slot when
        the match has Manual slots and the scenario's default is a policy's
        (see _default_focus_slot). An unavailable request falls back to that
        default.
    show_ranges : bool
        Whether the initial view displays configured range overlays.
    verbose_logging : bool
        Whether the host session requests verbose diagnostic logging.
    team_a_controller : TeamController
        Team A controller; default manual. Reactive choices require SharedObs.
    team_b_controller : TeamBController
        Team B controller; default manual. Both teams accept ALPHA
        (``reactive_tdm``), BETA (``scenario_5``) and GAMMA (``tdm_gamma``),
        which require SharedObs, as well as Manual and Random.
    execution_information_mode : ExecutionInformationMode
        no_shared_obs by default for direct diagnostics; shared_obs enables the
        composed team input contract. Declared Systems require shared_obs.
    systems : tuple of System or None, optional
        Actual Team A/B methods for system: aliases. Built-in teams use None.
        Methods are frozen at setup by default; fresh memory belongs to this match.
        The caller owns resource scopes, including any custom initializer work.
    borrow_systems : bool, optional
        False copies method setup as before. True retains the exact supplied
        descriptors and parameter leaves, including across restart. The caller
        must keep their parameters and external behavior fixed during the launch.
    manual_slots : tuple[int, ...] or None
        Human-owned physical rows, including configured dead actors. None derives
        whole Manual teams. Explicit rows may belong to a composed System team.
    slot_labels : tuple[str | None, ...] or None
        Ten physical controller labels. None derives them: inactive rows use
        None, Manual rows use Manual, a built-in team uses its public name
        (``tdm-beta``) and a System-alias team uses its loaded System's name
        for each champion (a ``team(...)`` names each champion by its member),
        never the alias. Labels never enter policy input.
        The episode context records them as each slot's and team's controller
        name, so a saved replay shows the same names.
    run_generation : int
        Nonnegative host match generation, default 0. Trusted setup replacement
        passes the previous generation plus one; it never changes scenario RNG.

    Returns
    -------
    DebuggerSession
        Initial matching session with frame-zero evidence, default pending rows
        and explicit random key.

    Raises
    ------
    ValueError
        If the scenario, seed, active selection or controller/input combination is
        invalid.
    """
    if type(run_generation) is not int or run_generation < 0:
        raise ValueError("run_generation must be a nonnegative Python int")
    if scenario.scenario_specification is not None:
        if systems is not None and systems[1] is not None:
            raise ValueError(SCENARIO_OPPONENT_ERROR)
        if manual_slots is not None and any(
            type(slot) is int and slot >= MAX_AGENTS_PER_TEAM for slot in manual_slots
        ):
            raise ValueError(SCENARIO_OPPONENT_ERROR)
    numeric_episode_id = _validate_fixed_setup(
        scenario, seed, team_b_controller, execution_information_mode
    )
    prepared_systems = _prepare_session_systems(
        (team_a_controller, team_b_controller), systems, borrow_systems=borrow_systems
    )
    if seed != evaluation_launch_specification.root_seed:
        raise ValueError("seed must equal the debugger evaluation launch root seed.")
    (
        scenario_default_movement_scale,
        config,
        state,
        observation,
        action_mask,
    ) = _fresh_snapshot(scenario, seed)
    _validate_reactive_controller_selection(
        scenario,
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        execution_information_mode=execution_information_mode,
    )
    manual_slots, slot_labels = normalize_slot_ownership(
        active_slots=tuple(
            slot
            for slot, active in enumerate(
                jax.device_get(config.agent_profile.active_mask)
            )
            if bool(active)
        ),
        controllers=(team_a_controller, team_b_controller),
        manual_slots=manual_slots,
        slot_labels=slot_labels,
        scripted=scenario.mode == "scripted",
        system_slot_names=None
        if slot_labels is not None
        else _system_slot_names(
            (team_a_controller, team_b_controller), prepared_systems, config
        ),
    )
    registrations = _session_system_registrations(prepared_systems)
    evaluation_context = build_debugger_evaluation_context_v1(
        evaluation_launch_specification,
        scenario=scenario,
        config=config,
        run_generation=run_generation,
        action_source_kind=debugger_action_source_kind_v1(
            scenario,
            team_a_controller,
            team_b_controller,
            manual_slots=manual_slots,
        ),
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        execution_information_mode=execution_information_mode,
        expected_horizon=_debugger_expected_horizon(scenario, config, state),
        system_registrations=registrations,
        manual_slots=manual_slots,
        slot_labels=slot_labels,
    )
    next_key = jax.random.key(
        _required_seed(
            evaluation_context.seed_protocol.root_seed
            if scenario.scenario_specification is not None
            else evaluation_context.seed_protocol.environment_seed
        )
    )
    information_availability = _initial_information_availability(
        config,
        execution_information_mode,
    )
    initial_frame = capture_initial_evaluation_frame_v3(
        evaluation_context,
        state,
        observation,
        action_mask,
        information_availability,
    )
    _validate_fixed_initial_frame(scenario, initial_frame)
    status_source_evidence_state = initialize_status_source_evidence_v2(
        evaluation_context,
        initial_frame,
    )
    default_slot = _default_focus_slot(scenario, manual_slots)
    requested_slot = (
        default_slot if controlled_global_slot is None else controlled_global_slot
    )
    if not (
        0 <= requested_slot < MAX_AGENT_SLOTS
        and evaluation_context.roster[requested_slot].configured_active
    ):
        requested_slot = default_slot
    _validate_active_context_slot(
        evaluation_context,
        requested_slot,
        name="controlled_global_slot",
    )

    wrapped, memory, system_ids = _initial_system_state(
        prepared_systems,
        config,
        state,
        observation,
        action_mask,
        information_availability,
        evaluation_context,
        registrations,
        numeric_episode_id=numeric_episode_id,
    )
    session = DebuggerSession(
        borrow_systems=borrow_systems,
        manual_slots=manual_slots,
        slot_labels=slot_labels,
        systems=prepared_systems,
        system_registrations=registrations,
        numeric_episode_id=numeric_episode_id,
        environment_state=wrapped,
        system_memory=memory,
        system_ids=system_ids,
        scenario=scenario,
        seed=seed,
        run_generation=run_generation,
        scenario_default_movement_scale=scenario_default_movement_scale,
        config=config,
        key=next_key,
        state=state,
        observation=observation,
        action_mask=action_mask,
        raw_continuation_identity=None,
        evaluation_context=evaluation_context,
        current_evaluation_frame=initial_frame,
        incoming_evaluation_view=None,
        status_source_evidence_state=status_source_evidence_state,
        last_submission_kind=None,
        last_report_actor_slots=(),
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        controlled_global_slot=requested_slot,
        pending_actions=_default_pending_actions(
            evaluation_context,
            initial_frame.action_mask,
        ),
        next_script_frame_index=0,
        show_ranges=show_ranges,
        verbose_logging=False,
    )
    return session


def select_clicked_target(
    session: DebuggerSession,
    target_global_slot: int,
) -> DebuggerSession:
    """Select an active target and auto-arm Basic only when exact lane zero is legal.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and pending action rows.
    target_global_slot : int
        Active global target slot selected in the authorized researcher view.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
    _validate_active_context_slot(
        session.evaluation_context,
        target_global_slot,
        name="clicked target",
    )
    target_action = _target_action_from_context(
        session.evaluation_context,
        session.controlled_global_slot,
        target_global_slot,
    )
    availability = lane_availability(
        session.current_evaluation_frame.action_mask,
        session.controlled_global_slot,
        target_action,
        0,
    )
    pending = PendingAction(
        move_action=session.pending_action.move_action,
        selected_global_target_slot=target_global_slot,
        armed_lane=0 if availability.lane_0_available else None,
        arm_origin="automatic" if availability.lane_0_available else None,
    )
    return _replace_controlled_pending_action(session, pending)


def clear_pending_target(session: DebuggerSession) -> DebuggerSession:
    """Clear target selection while preserving an explicit Mage Burst arm.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and controlled actor draft.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.
    """
    class_id = session.evaluation_context.roster[
        session.controlled_global_slot
    ].class_id
    keep_mage_ultimate = (
        class_id == MAGE_CLASS_ID
        and session.pending_action.armed_lane == 1
        and session.pending_action.arm_origin == "explicit"
    )
    pending = PendingAction(
        move_action=session.pending_action.move_action,
        selected_global_target_slot=None,
        armed_lane=1 if keep_mage_ultimate else 0,
        arm_origin="explicit" if keep_mage_ultimate else "automatic",
    )
    return _replace_controlled_pending_action(session, pending)


def arm_basic(session: DebuggerSession) -> DebuggerSession:
    """Explicitly arm lane zero even when the current pair is unavailable.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and controlled actor draft.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.
    """
    return _replace_controlled_pending_action(
        session,
        replace(
            session.pending_action,
            armed_lane=0,
            arm_origin="explicit",
        ),
    )


def arm_ultimate(session: DebuggerSession) -> DebuggerSession:
    """Explicitly arm lane one; Mage Burst always uses target-none.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and controlled actor draft.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.
    """
    class_id = session.evaluation_context.roster[
        session.controlled_global_slot
    ].class_id
    return _replace_controlled_pending_action(
        session,
        replace(
            session.pending_action,
            selected_global_target_slot=(
                None
                if class_id == MAGE_CLASS_ID
                else session.pending_action.selected_global_target_slot
            ),
            armed_lane=1,
            arm_origin="explicit",
        ),
    )


def select_no_combat(session: DebuggerSession) -> DebuggerSession:
    """Stage no-combat intent while preserving movement and target context.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and controlled actor draft.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.
    """
    return _replace_controlled_pending_action(
        session,
        replace(
            session.pending_action,
            armed_lane=None,
            arm_origin=None,
        ),
    )


def set_pending_movement(
    session: DebuggerSession,
    move_action: int,
) -> DebuggerSession:
    """Set one in-domain pending movement category without inspecting legality.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and controlled actor draft.
    move_action : int
        Movement category in the fixed move-action domain; current legality is
        left to submission.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
    if not 0 <= move_action < NUM_MOVE_ACTIONS:
        msg = f"move_action must be in [0, {NUM_MOVE_ACTIONS}); got {move_action}."
        raise ValueError(msg)
    return _replace_controlled_pending_action(
        session,
        replace(session.pending_action, move_action=move_action),
    )


def cycle_controlled_actor(
    session: DebuggerSession,
    direction: int,
) -> DebuggerSession:
    """Cycle active fixed slots through direct controlled-actor selection.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session with its active roster and selection.
    direction : int
        1 selects the next active slot; -1 selects the previous active slot.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
    if direction not in (-1, 1):
        msg = f"direction must be -1 or 1; got {direction}."
        raise ValueError(msg)
    active_slots = _active_context_slots(session)
    current_index = active_slots.index(session.controlled_global_slot)
    controlled_slot = active_slots[(current_index + direction) % len(active_slots)]
    return select_controlled_actor(session, controlled_slot)


def select_controlled_actor(
    session: DebuggerSession,
    global_slot: int,
) -> DebuggerSession:
    """Select an active actor without changing any staged draft or simulator epoch.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and staged joint draft.
    global_slot : int
        Active zero-based global actor slot to control.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
    _validate_active_context_slot(
        session.evaluation_context,
        global_slot,
        name="controlled actor",
    )
    return replace(session, controlled_global_slot=global_slot)


def agent_pov_recipient_slot(
    session: DebuggerSession,
    previous_recipient: int | None,
) -> int:
    """Choose whose eyes Agent POV shows: the viewing champion (the recipient).

    The recipient is separate from the focused (controlled) champion. Rules:

    - Mixed control (at least one Manual slot): the recipient is the last
      Manual champion that was focused, so the view always belongs to a
      champion you order. Focusing a policy champion, by Tab, a click or a
      roster pick, never moves it; neither does an enemy you only look at.
      With Manual champions on both teams, the view follows the one you
      focused last.
    - Policies Only (no Manual slot): the recipient is the focused champion,
      so the view follows the champion you look at.

    Parameters
    ----------
    session : DebuggerSession
        Session after the latest edit. Its ``controlled_global_slot`` is the
        focused champion and its ``manual_slots`` say who is Manual.
    previous_recipient : int or None
        The recipient before this edit, or None at launch. It is kept only
        while it is still a configured-active Manual slot and the focus rests
        on a policy champion.

    Returns
    -------
    int
        Global slot 0-9 of the recipient. At launch with a policy focus it is
        the lowest Manual slot.

    Notes
    -----
    Pure host rule; no simulator, file or random work. Calling it again with
    its own result returns the same slot, so the live service can recompute it
    for any candidate session without drift. It never reads what anybody can
    see; the Tab sight rule lives in the input layer.
    """
    focus = session.controlled_global_slot
    manual = session.manual_slots
    if not manual or focus in manual:
        return focus
    if (
        previous_recipient is not None
        and previous_recipient in manual
        and session.evaluation_context.roster[previous_recipient].configured_active
    ):
        return previous_recipient
    return min(manual)


def _validate_joint_action(action: Action) -> None:
    """Require all three submitted heads to be int32 arrays with shape [10]."""
    for name, head in zip(Action._fields, action, strict=True):
        if head.shape != (MAX_AGENT_SLOTS,):
            msg = (
                f"action.{name} must have shape ({MAX_AGENT_SLOTS},); got {head.shape}."
            )
            raise ValueError(msg)
        if head.dtype != jnp.int32:
            msg = f"action.{name} must have dtype int32; got {head.dtype}."
            raise ValueError(msg)


def _terminal_reason(session: DebuggerSession) -> str | None:
    """Return the sealed endpoint reason in priority order, or None while stepping is
    allowed.
    """
    if session.terminated:
        return "terminated"
    if session.truncated:
        return "truncated"
    if session.reached_declared_horizon:
        return "at its declared horizon"
    return None


def _post_submit_pending(
    session: DebuggerSession,
    action_mask: ActionMaskV1,
) -> tuple[PendingAction, ...]:
    """Keep target choices while resetting movement and re-arming only a legal Basic
    pair.
    """
    pending_actions: list[PendingAction] = []
    for actor_slot in range(MAX_AGENT_SLOTS):
        if not session.evaluation_context.roster[actor_slot].configured_active:
            pending_actions.append(PendingAction(armed_lane=None, arm_origin=None))
            continue
        target_slot = session.pending_actions[actor_slot].selected_global_target_slot
        target_action = _target_action_from_context(
            session.evaluation_context,
            actor_slot,
            target_slot,
        )
        availability = lane_availability(
            action_mask,
            actor_slot,
            target_action,
            0,
        )
        pending_actions.append(
            PendingAction(
                move_action=MOVE_STAY,
                selected_global_target_slot=target_slot,
                armed_lane=0 if availability.lane_0_available else None,
                arm_origin="automatic" if availability.lane_0_available else None,
            )
        )
    return tuple(pending_actions)


def submit_joint_action(
    session: DebuggerSession,
    submitted_action: Action,
    *,
    submission_kind: SubmissionKind,
    report_actor_slots: tuple[int, ...],
    system_memory: SystemState | None = None,
) -> DebuggerSession:
    """Split once, step once, diagnose once, and advance all paired epoch fields.

    Parameters
    ----------
    session : DebuggerSession
        Current paired state, observation, action mask, random key and
        evaluation frame.
    submitted_action : Action
        Three int32 action heads shaped [10], chosen before this transition.
    submission_kind : SubmissionKind
        interactive or scripted, retained with the resulting transition.
    report_actor_slots : tuple[int, ...]
        Distinct active global actor slots whose submitted/accepted rows should
        be reported.
    system_memory : SystemState | None
        Proposed memory from this exact joint choice. None retains the current
        memory. It is adopted only with the completely packaged successor.

    Returns
    -------
    DebuggerSession
        Matching successor session and captured incoming transition, or the
        unchanged session when already sealed. Its recent_scores gains the
        old frame's public team scores (DebuggerSession.next_recent_scores,
        the Momentum switch's 10-tick history), its champion_takedowns adds
        this tick to every view's Champion Streaks counts and its
        match_moments adds it to every view's Herald Cards history (runs,
        rescues, the earlier lead and the fresh start), both through
        live_presentation.advance_live_match_history_v1 (Agent POV scene work
        only on a tick with a fall or a rescue).

    Raises
    ------
    DebuggerTransitionFailureV1
        If action construction, simulation, capture or successor packaging fails.
        The stable stage/code identifies the boundary without exposing raw details.

    Notes
    -----
        Numerical work runs through Core and transition capture; no file is written.
        The input session is immutable, so a failed call cannot install a partial
        successor. Higher-level recording code decides how to handle a failure.

        Every Submit's Core step runs compiled here, for manual, scripted and
        policy Submits alike, unless a caller has replaced ``control.step``.
        Policy actions arrive already built: ``submit_interactive`` runs
        SharedObs policy teams through the compiled policy executor, and
        NoSharedObs Random uncompiled through
        ``execute_no_shared_obs_team_policy`` (fast once JAX has warmed up).
        The first Submit in a DevClient session (the first in this Python
        process) compiles for several seconds, about 6 s on CPU, counting the
        executor's compile when a policy plays. The first Submit after
        switching to a controller pair not yet used in this process compiles
        the executor again, about 3 s. After that a Submit takes about 0.03 s
        while array shapes and dtypes stay the same. These historical CPU times
        are examples, not guarantees. Old built-in dispatch holds the service
        lock; queued System calls run outside HTTP, router and service locks.
        Publishing the accepted result briefly locks the service state.
    """
    terminal_reason = _terminal_reason(session)
    if terminal_reason is not None:
        return session
    try:
        _validate_joint_action(submitted_action)
        if len(report_actor_slots) != len(set(report_actor_slots)):
            raise ValueError("report_actor_slots must not contain duplicates.")
        for actor_slot in report_actor_slots:
            _validate_active_context_slot(
                session.evaluation_context,
                actor_slot,
                name="report actor",
            )
    except Exception as error:
        raise _transition_failure(
            "action_build",
            "invalid_submitted_action",
            error,
        ) from error

    try:
        if session.scenario.scenario_specification is not None:
            from marl_battlegrounds.evaluation.evaluate import episode_keys

            seed_ids, local_steps = _fixed_key_coordinates(session)
            next_key = session.key
            step_key = episode_keys(session.key, seed_ids, local_steps, 1)[0]
        else:
            next_key, step_key = jax.random.split(session.key)
        compiled = step is core_env.step
        stepper = _COMPILED_STEP if compiled else step
        step_inputs = (
            session.config,
            session.state,
            session.action_mask,
            submitted_action,
            step_key,
        )
        (
            next_state,
            next_observation,
            reward,
            done_flags,
            next_action_mask,
            info,
        ) = stepper(*(_committed(step_inputs) if compiled else step_inputs))
    except Exception as error:
        raise _transition_failure(
            "simulation",
            "simulator_step_failed",
            error,
        ) from error
    try:
        transition, successor_frame = capture_evaluation_transition_unit_v3(
            session.evaluation_context,
            session.current_evaluation_frame,
            next_state,
            next_observation,
            next_action_mask,
            info.transition_facts,
            reward,
            done_flags,
            successor_shared_obs_information_availability_by_recipient_and_sensor_source=(
                _captured_information_availability(session)
            ),
        )
    except Exception as error:
        raise _transition_failure(
            "capture",
            "transition_capture_failed",
            error,
        ) from error
    try:
        coherent_view = EvaluationTransitionViewV1(
            context=session.evaluation_context,
            start_frame=session.current_evaluation_frame,
            transition=transition,
            successor_frame=successor_frame,
        )
        status_source_evidence_state = advance_status_source_evidence_v2(
            session.status_source_evidence_state,
            coherent_view,
        )
        champion_takedowns, match_moments = advance_live_match_history_v1(
            session.champion_takedowns, session.match_moments, coherent_view
        )
        wrapped = session.environment_state
        if wrapped is not None:
            count, exhausted = _checked_increment(
                wrapped.cumulative_transition_count, 1
            )
            wrapped = wrapped._replace(
                core_state=next_state,
                observation=next_observation,
                action_mask=next_action_mask,
                done=done_flags,
                episode_start=jnp.asarray(False),
                cumulative_transition_count=count,
                lifecycle_error=wrapped.lifecycle_error | exhausted,
            )
        return replace(
            session,
            environment_state=wrapped,
            system_memory=session.system_memory
            if system_memory is None
            else system_memory,
            key=next_key,
            state=next_state,
            observation=next_observation,
            action_mask=next_action_mask,
            evaluation_context=coherent_view.context,
            current_evaluation_frame=coherent_view.successor_frame,
            incoming_evaluation_view=coherent_view,
            recent_scores=session.next_recent_scores(),
            champion_takedowns=champion_takedowns,
            match_moments=match_moments,
            status_source_evidence_state=status_source_evidence_state,
            last_submission_kind=submission_kind,
            last_report_actor_slots=tuple(sorted(report_actor_slots)),
            raw_continuation_identity=None,
            pending_actions=_post_submit_pending(
                session,
                coherent_view.successor_frame.action_mask,
            ),
        )
    except Exception as error:
        raise _transition_failure(
            "validation",
            "transition_packaging_failed",
            error,
        ) from error


def submit_interactive(
    session: DebuggerSession,
    *,
    actor_global_slots: tuple[int, ...] | None = None,
) -> DebuggerSession:
    """Submit one authorized collection of same-epoch pending actor rows.

    Parameters
    ----------
    session : DebuggerSession
        Current session carrying both team controller choices and staged manual
        rows.
    actor_global_slots : tuple[int, ...] | None
        Optional configured Manual submission subset. None uses every Manual row,
        including dead actors. Methods always choose both teams first; only the
        selected human-owned rows replace their outputs before the single step.
        An explicit subset must be nonempty. Other configured Manual rows stay
        neutral, regardless of a method's proposal for those rows.

    Returns
    -------
    DebuggerSession
        Successor session after one shared submission, or the unchanged sealed
        endpoint.

    Raises
    ------
    DebuggerTransitionFailureV1
        If action construction, simulation, capture or successor packaging fails.
        The stable stage/code identifies the boundary without exposing raw details.

    Notes
    -----
        Numerical work runs through Core and transition capture; no file is written.
        The input session is immutable, so a failed call cannot install a partial
        successor. Higher-level recording code decides how to handle a failure.

        Every Submit's Core step runs compiled, manual Submits included. With
        SharedObs, the policy teams (ALPHA, BETA, GAMMA and Random) also run
        through the compiled policy executor. NoSharedObs Random still runs
        uncompiled through ``execute_no_shared_obs_team_policy``; it is fast
        once JAX has warmed up. The first Submit in a DevClient session (the
        first in this Python process) compiles for several seconds, about 6 s
        on CPU. The first Submit after switching to a controller pair not yet
        used in this process compiles the executor again, about 3 s. After
        that a Submit takes about 0.03 s. These historical CPU times are examples,
        not guarantees. Old built-in dispatch holds the service lock. Queued
        System callbacks run outside HTTP, router and service locks, on their
        execution owner. Publishing the result briefly locks accepted state.
    """
    if session.episode_sealed:
        return session
    submission_slots = (
        session.manual_slots if actor_global_slots is None else actor_global_slots
    )
    try:
        if actor_global_slots == ():
            raise ValueError("An explicit Manual submission subset must be nonempty")
        if type(submission_slots) is not tuple or any(
            type(slot) is not int or slot not in session.manual_slots
            for slot in submission_slots
        ):
            raise ValueError("Only configured Manual rows can submit human drafts")
        manual_action = (
            build_interactive_joint_action(
                session.evaluation_context,
                session.pending_actions,
                actor_global_slots=submission_slots,
            )
            if submission_slots
            else None
        )
    except Exception as error:
        raise _transition_failure(
            "action_build", "interactive_action_build_failed", error
        ) from error
    policy_mode = any(
        controller != "manual"
        for controller in (session.team_a_controller, session.team_b_controller)
    )
    proposed = None
    if policy_mode:
        try:
            if session.systems is None:
                action = _build_configured_joint_action(session)
            else:
                action, proposed = _build_system_joint_action(session)
            if manual_action is not None:
                rows = jnp.asarray(session.manual_slots, dtype=jnp.int32)
                action = Action(
                    *(
                        head.at[rows].set(manual_head[rows])
                        for head, manual_head in zip(action, manual_action, strict=True)
                    )
                )
            if proposed is not None and session.manual_slots:
                trace = proposed.policy_trace
                proposed = proposed._replace(
                    policy_trace=trace._replace(
                        policy_ids=trace.policy_ids.at[
                            :, jnp.asarray(session.manual_slots, dtype=jnp.int32)
                        ].set(-1)
                    )
                )
        except Exception as error:
            raise _transition_failure(
                "action_build", "policy_action_build_failed", error
            ) from error
    else:
        if manual_action is None:
            raise _transition_failure(
                "action_build",
                "interactive_action_build_failed",
                ValueError("Manual submission needs at least one configured row"),
            )
        action = manual_action
    report_slots = _active_context_slots(session) if policy_mode else submission_slots
    if proposed is not None:
        return submit_joint_action(
            session,
            action,
            submission_kind="interactive",
            report_actor_slots=report_slots,
            system_memory=proposed,
        )
    return submit_joint_action(
        session,
        action,
        submission_kind="interactive",
        report_actor_slots=report_slots,
    )


def warm_compiled_turn(session: DebuggerSession) -> bool:
    """Compile a turn's programs now, so the player's first Lock In is immediate.

    Runs one Submit on ``session`` exactly as Lock In would and throws the
    result away. This compiles the policy and Core step programs (about 4 to
    7 s on CPU and 12 s on the RTX 5090 for 7b+pool) before the page opens or
    while new controllers load, instead of on the first Lock In. The game,
    the session and every method's memory are unchanged: sessions are
    immutable and JAX methods are pure.

    Parameters
    ----------
    session : DebuggerSession
        A freshly created or replaced session, before anyone has played on it.

    Returns
    -------
    bool
        True when the turn was compiled. False when it was skipped: a method
        runs on the host (an LLM, for example), whose outside effects and cost
        must happen only on a real turn; or the turn fails, which the first
        real Lock In then reports through the client's normal failure path.

    Notes
    -----
    Later turns reuse the compiled programs (see the module description and
    ``_committed``), so after this no Lock In compiles anything. No file,
    recording or network I/O happens here.
    """
    if any(
        policy_execution._execution(method).execution == "host"  # pyright: ignore[reportPrivateUsage]
        for method in session.systems or ()
    ):
        return False
    try:
        submit_interactive(session)
    except DebuggerTransitionFailureV1:
        return False
    return True


def submit_next_script_frame(
    session: DebuggerSession,
) -> DebuggerSession:
    """Submit the next registered multi-actor frame through the shared boundary.

    Parameters
    ----------
    session : DebuggerSession
        Current scripted session and its next-frame cursor.

    Returns
    -------
    DebuggerSession
        Session with one captured scripted transition and advanced script
        cursor, or the original endpoint if no step is possible.

    Raises
    ------
    DebuggerTransitionFailureV1
        If action construction, simulation, capture or successor packaging fails.
        The stable stage/code identifies the boundary without exposing raw details.

    Notes
    -----
        Numerical work runs through Core and transition capture; no file is written.
        The input session is immutable, so a failed call cannot install a partial
        successor. Higher-level recording code decides how to handle a failure.

        The scripted Core step runs compiled through ``submit_joint_action``;
        its Notes give the first-Submit compile cost.
    """
    scenario = session.scenario
    if session.next_script_frame_index >= len(scenario.frames):
        return session
    frame = scenario.frames[session.next_script_frame_index]
    try:
        action = build_scripted_joint_action(session.evaluation_context, frame)
        report_slots = tuple(
            sorted(command.actor_global_slot for command in frame.commands)
        )
    except Exception as error:
        raise _transition_failure(
            "action_build",
            "scripted_action_build_failed",
            error,
        ) from error
    submitted = submit_joint_action(
        session,
        action,
        submission_kind="scripted",
        report_actor_slots=report_slots,
    )
    if (
        submitted.current_evaluation_frame.frame_index
        == session.current_evaluation_frame.frame_index
    ):
        return submitted
    return replace(
        submitted,
        next_script_frame_index=session.next_script_frame_index + 1,
    )


def reset_session(
    session: DebuggerSession,
) -> DebuggerSession:
    """Recreate the deterministic initial epoch at the product scale.

    Parameters
    ----------
    session : DebuggerSession
        Session to recreate from its loaded scenario and recorded root seed.

    Returns
    -------
    DebuggerSession
        Fresh deterministic initial session with a new run generation and
        preserved control choice when valid.

    Raises
    ------
    ValueError
        If the scenario, seed, active selection or controller/input combination is
        invalid.
    """
    scenario = session.scenario
    return _restart_session(
        session,
        scenario,
        preserve_controlled_slot=True,
    )


def _restart_session(
    session: DebuggerSession,
    scenario: DebuggerScenario,
    *,
    preserve_controlled_slot: bool,
    team_a_controller: TeamController | None = None,
    team_b_controller: TeamBController | None = None,
    execution_information_mode: ExecutionInformationMode | None = None,
    systems: tuple[System | None, System | None] | None = None,
    manual_slots: tuple[int, ...] | None = None,
    slot_labels: tuple[str | None, ...] | None = None,
) -> DebuggerSession:
    """Build one coherent fresh epoch without entering the simulator step seam.

    Fixed scenarios accept a plain restart only. Any setup override is rejected
    before reading a start or initializing memory, even when values are equal.
    The old accepted session and its memory remain unchanged on failure. A plain
    restart retains slot ownership. Explicit setup changes validate new tuples
    before initialization; omitted tuples derive whole-team legacy ownership.
    Derived labels name a System-alias team's champions from its loaded System,
    never by the alias. The new episode context records the resulting slot
    labels as controller names. Both follow create_session. The fresh epoch
    starts with empty recent_scores, zero champion_takedowns and an empty
    match_moments history, so Momentum, Champion Streaks and the Herald Cards
    show nothing of the old game.
    """
    if session.scenario.scenario_specification is not None and (
        scenario is not session.scenario
        or team_a_controller is not None
        or team_b_controller is not None
        or execution_information_mode is not None
        or systems is not None
        or manual_slots is not None
        or slot_labels is not None
    ):
        raise CombatConfigurationRejectedError(
            "Fixed scenarios can restart without setup overrides"
        )
    numeric_episode_id = _validate_fixed_setup(
        scenario,
        session.seed,
        session.team_b_controller if team_b_controller is None else team_b_controller,
        session.evaluation_context.execution_information_mode
        if execution_information_mode is None
        else execution_information_mode,
    )
    next_team_a_controller = (
        session.team_a_controller if team_a_controller is None else team_a_controller
    )
    next_team_b_controller = (
        session.team_b_controller if team_b_controller is None else team_b_controller
    )
    next_information_mode = (
        session.evaluation_context.execution_information_mode
        if execution_information_mode is None
        else execution_information_mode
    )
    selected = (next_team_a_controller, next_team_b_controller)
    provided = systems
    previous = (session.team_a_controller, session.team_b_controller)
    if provided is None and any(is_system_controller(value) for value in selected):
        if any(
            is_system_controller(value) and value != previous[index]
            for index, value in enumerate(selected)
        ):
            raise ValueError("A new System alias needs its loaded method")
        provided = session.systems
    prepared_systems = (
        session.systems
        if selected == previous and provided is session.systems
        else _prepare_session_systems(
            selected, provided, borrow_systems=session.borrow_systems
        )
    )
    (
        scenario_default_movement_scale,
        config,
        state,
        observation,
        action_mask,
    ) = _fresh_snapshot(scenario, session.seed)
    _validate_reactive_controller_selection(
        scenario,
        team_a_controller=next_team_a_controller,
        team_b_controller=next_team_b_controller,
        execution_information_mode=next_information_mode,
    )
    same_setup = (
        scenario is session.scenario
        and selected == previous
        and systems is None
        and execution_information_mode is None
    )
    manual_slots, slot_labels = normalize_slot_ownership(
        active_slots=tuple(
            slot
            for slot, active in enumerate(
                jax.device_get(config.agent_profile.active_mask)
            )
            if bool(active)
        ),
        controllers=selected,
        manual_slots=session.manual_slots
        if same_setup and manual_slots is None
        else manual_slots,
        slot_labels=session.slot_labels
        if same_setup and slot_labels is None
        else slot_labels,
        scripted=scenario.mode == "scripted",
        system_slot_names=None
        if same_setup or slot_labels is not None
        else _system_slot_names(selected, prepared_systems, config),
    )
    run_generation = session.run_generation + 1
    launch_specification = build_debugger_evaluation_launch_specification_v1(
        root_seed=_required_seed(session.evaluation_context.seed_protocol.root_seed),
        code_revision=session.evaluation_context.code_revision,
        capture_profile=cast(
            DebuggerCaptureProfileV1,
            session.evaluation_context.capture_profile,
        ),
    )
    registrations = (
        session.system_registrations
        if prepared_systems is session.systems
        else _session_system_registrations(prepared_systems)
    )
    evaluation_context = build_debugger_evaluation_context_v1(
        launch_specification,
        scenario=scenario,
        config=config,
        run_generation=run_generation,
        action_source_kind=debugger_action_source_kind_v1(
            scenario,
            next_team_a_controller,
            next_team_b_controller,
            manual_slots=manual_slots,
        ),
        team_a_controller=next_team_a_controller,
        team_b_controller=next_team_b_controller,
        execution_information_mode=next_information_mode,
        expected_horizon=_debugger_expected_horizon(scenario, config, state),
        system_registrations=registrations,
        manual_slots=manual_slots,
        slot_labels=slot_labels,
    )
    next_key = jax.random.key(
        _required_seed(
            evaluation_context.seed_protocol.root_seed
            if scenario.scenario_specification is not None
            else evaluation_context.seed_protocol.environment_seed
        )
    )
    information_availability = _initial_information_availability(
        config,
        next_information_mode,
    )
    initial_frame = capture_initial_evaluation_frame_v3(
        evaluation_context,
        state,
        observation,
        action_mask,
        information_availability,
    )
    _validate_fixed_initial_frame(scenario, initial_frame)
    status_source_evidence_state = initialize_status_source_evidence_v2(
        evaluation_context,
        initial_frame,
    )
    default_slot = _default_focus_slot(scenario, manual_slots)
    controlled_slot = (
        session.controlled_global_slot if preserve_controlled_slot else default_slot
    )
    if not (
        0 <= controlled_slot < MAX_AGENT_SLOTS
        and evaluation_context.roster[controlled_slot].configured_active
    ):
        controlled_slot = default_slot
    _validate_active_context_slot(
        evaluation_context,
        controlled_slot,
        name="controlled_global_slot",
    )
    wrapped, memory, system_ids = _initial_system_state(
        prepared_systems,
        config,
        state,
        observation,
        action_mask,
        information_availability,
        evaluation_context,
        registrations,
        numeric_episode_id=numeric_episode_id,
    )
    restarted = replace(
        session,
        manual_slots=manual_slots,
        slot_labels=slot_labels,
        systems=prepared_systems,
        system_registrations=registrations,
        numeric_episode_id=numeric_episode_id,
        environment_state=wrapped,
        system_memory=memory,
        system_ids=system_ids,
        scenario=scenario,
        run_generation=run_generation,
        scenario_default_movement_scale=scenario_default_movement_scale,
        config=config,
        key=next_key,
        state=state,
        observation=observation,
        action_mask=action_mask,
        raw_continuation_identity=None,
        evaluation_context=evaluation_context,
        current_evaluation_frame=initial_frame,
        incoming_evaluation_view=None,
        status_source_evidence_state=status_source_evidence_state,
        last_submission_kind=None,
        last_report_actor_slots=(),
        team_a_controller=next_team_a_controller,
        team_b_controller=next_team_b_controller,
        controlled_global_slot=controlled_slot,
        pending_actions=_default_pending_actions(
            evaluation_context,
            initial_frame.action_mask,
        ),
        next_script_frame_index=0,
        # A restarted game has no earlier frames, so Momentum starts empty
        # and nobody has a takedown yet.
        recent_scores=(),
        champion_takedowns=ChampionTakedownsV1(),
        match_moments=MatchMomentHistoryV1(),
    )
    return restarted


def replace_loaded_setup(
    session: DebuggerSession,
    *,
    scenario: DebuggerScenario,
    seed: int,
    team_a_controller: TeamController,
    team_b_controller: TeamController,
    systems: tuple[System | None, System | None] | None,
    manual_slots: tuple[int, ...],
    slot_labels: tuple[str | None, ...],
    execution_information_mode: ExecutionInformationMode = "shared_obs",
) -> DebuggerSession:
    """Start one trusted, already resolved setup through the normal session owner.

    This private launcher boundary accepts the chosen scenario, root seed, Team
    A/B labels, loaded methods and physical ownership tuples. It does not load
    methods or enter their resource scopes. Fixed scenarios still require their
    exact root, opponent and SharedObs mode; supplied Team B methods or Manual
    rows are rejected before initialization. Browser setup must reject forbidden
    fixed fields before loading anything. Existing reset/configuration commands
    keep their stricter guards.

    Reuse the old recording revision, capture profile, ownership mode and display
    preferences. Advance the host generation once and allocate fresh method
    memory through create_session. Preserve the old accepted session on failure.
    execution_information_mode defaults to shared_obs for loaded methods.
    """
    launch = build_debugger_evaluation_launch_specification_v1(
        root_seed=seed,
        code_revision=session.evaluation_context.code_revision,
        capture_profile=cast(
            DebuggerCaptureProfileV1, session.evaluation_context.capture_profile
        ),
    )
    return create_session(
        scenario,
        seed=seed,
        evaluation_launch_specification=launch,
        controlled_global_slot=session.controlled_global_slot,
        show_ranges=session.show_ranges,
        verbose_logging=session.verbose_logging,
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        execution_information_mode=execution_information_mode,
        systems=systems,
        borrow_systems=session.borrow_systems,
        manual_slots=manual_slots,
        slot_labels=slot_labels,
        run_generation=session.run_generation + 1,
    )


def set_combat_configuration(
    session: DebuggerSession,
    *,
    team_a_controller: TeamController,
    team_b_controller: TeamBController,
    execution_information_mode: ExecutionInformationMode,
    systems: tuple[System | None, System | None] | None = None,
    manual_slots: tuple[int, ...] | None = None,
    slot_labels: tuple[str | None, ...] | None = None,
) -> DebuggerSession:
    """Replace the episode only when its controller or information mode changes.

    Parameters
    ----------
    session : DebuggerSession
        Current session, preserved if all requested settings already match.
    team_a_controller : TeamController
        Supported Team A controller kind.
    team_b_controller : TeamBController
        Supported Team B controller kind; both teams accept the same kinds.
    execution_information_mode : ExecutionInformationMode
        shared_obs or no_shared_obs; must satisfy the selected controllers'
        input needs. Declared Systems require shared_obs.
    systems : tuple of System or None, optional
        Loaded Team A/B methods for a deliberate replacement. Omission retains
        current declarations on restart. Supplying methods creates fresh memory
        even when controller labels are unchanged. Caller owns resource scopes.
    manual_slots, slot_labels : tuple or None
        New physical Manual ownership and ten controller labels. Omission derives
        whole-team defaults on a changed setup (a System-alias team's labels are
        its loaded System's names, as in create_session); an unchanged request
        is a no-op.

    Returns
    -------
    DebuggerSession
        New episode when settings change, otherwise the original session.

    Raises
    ------
    ValueError
        If the scenario, seed, active selection or controller/input combination is
        invalid.
    """
    if session.scenario.scenario_specification is not None:
        raise CombatConfigurationRejectedError(
            "Fixed scenarios do not accept whole-match setup changes"
        )
    if not is_team_controller(team_a_controller):
        raise CombatConfigurationRejectedError(
            "team_a_controller must be manual, reactive_tdm, random_valid, "
            "scenario_5, or tdm_gamma"
        )
    if not is_team_controller(team_b_controller):
        raise CombatConfigurationRejectedError(
            "team_b_controller must be manual, reactive_tdm, random_valid, "
            "scenario_5, or tdm_gamma"
        )
    if execution_information_mode not in ("shared_obs", "no_shared_obs"):
        raise CombatConfigurationRejectedError(
            "execution_information_mode must be shared_obs or no_shared_obs"
        )
    if (
        session.team_a_controller == team_a_controller
        and session.team_b_controller == team_b_controller
        and session.evaluation_context.execution_information_mode
        == execution_information_mode
        and systems is None
        and manual_slots is None
        and slot_labels is None
    ):
        return session
    return _restart_session(
        session,
        session.scenario,
        preserve_controlled_slot=True,
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        execution_information_mode=execution_information_mode,
        systems=systems,
        manual_slots=manual_slots,
        slot_labels=slot_labels,
    )


def switch_scenario(
    session: DebuggerSession,
    scenario: DebuggerScenario,
    *,
    systems: tuple[System | None, System | None] | None = None,
) -> DebuggerSession:
    """Start another scenario at the canonical product movement scale.

    Parameters
    ----------
    session : DebuggerSession
        Current host session whose presentation settings carry into the
        replacement.
    scenario : DebuggerScenario
        New validated scenario factory and metadata.
    systems : tuple of System or None, optional
        Fresh loaded Team A/B methods, when a declared replacement owns setup.
        Omission preserves the existing caller-owned methods.

    Returns
    -------
    DebuggerSession
        Fresh episode for the selected scenario at the product movement
        setting.

    Raises
    ------
    ValueError
        If the scenario, seed, active selection or controller/input combination is
        invalid.
    """
    if (
        session.scenario.scenario_specification is not None
        or scenario.scenario_specification is not None
    ):
        raise CombatConfigurationRejectedError(
            "Fixed scenarios start through their own launch setup"
        )
    return _restart_session(
        session,
        scenario,
        preserve_controlled_slot=False,
        systems=systems,
    )
