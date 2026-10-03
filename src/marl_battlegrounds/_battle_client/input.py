"""Turn validated browser input into host edits or one explicit simulator action.

``dispatch_command`` handles keyboard, pointer, roster and view commands for the
live service. Pointer selection uses the currently authorized scene. Draft edits
leave simulator state and random keys unchanged; submissions and resets call the
control layer. Results explicitly identify transitions and episode restarts so
recording can follow the correct endpoint. No HTTP response or replay file is
written here.

Keys: W A S D Q E Z C step and X stays (arrow keys never move; in the
BattleClient they do nothing), 0 1 2 pick No Combat, Basic or Ultimate, Space
and Enter lock in (Policies Only: play one turn), N is one tick (in interactive
play the same as Space; in scripted playback the next registered frame), R
restarts, G toggles ranges, Tab and Shift+Tab move the focus, Escape clears
the target. V, B and F belong to the browser and are never handled here.

Agent POV rules (the service passes the recipient as ``pov_global_slot``):
clicks hit only bodies the recipient's team sees; Tab skips champions that
team cannot see, except Manual champions and teammates; a roster pick of an
unseen enemy policy champion is refused without naming where it is; and every
order (step, 0 1 2, Escape, target picks) goes to the recipient, the champion
whose view is shown, moving the focus back to it.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import hypot, isfinite
from typing import Literal

from marl_battlegrounds._battle_client.control import (
    CombatConfigurationRejectedError,
    DebuggerTransitionFailureV1,
    arm_basic,
    arm_ultimate,
    clear_pending_target,
    lane_availability,
    reset_session,
    select_clicked_target,
    select_controlled_actor,
    select_no_combat,
    set_combat_configuration,
    set_pending_movement,
    submit_interactive,
    submit_next_script_frame,
)
from marl_battlegrounds._battle_client.model import (
    DebuggerSession,
    RawContinuationIdentity,
)
from marl_battlegrounds._battle_client.protocol import (
    ActorPovTargetActionCommandV1,
    BattlefieldPointerCommandV1,
    DebuggerCommandV1,
    ExitCommandV1,
    KeyboardCommandV1,
    Preset,
    ResetCommandV1,
    RosterSelectionCommandV1,
    ScenarioSwitchCommandV1,
    SetCombatConfigurationCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
    ViewMode,
)
from marl_battlegrounds.core.types import (
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_NORTHEAST,
    MOVE_NORTHWEST,
    MOVE_SOUTH,
    MOVE_SOUTHEAST,
    MOVE_SOUTHWEST,
    MOVE_STAY,
    MOVE_WEST,
)
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.rendering.authorized_pov_scene import (
    build_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    build_researcher_analyzer_projection_v2,
    build_shared_obs_authority_source_material_projection_v1,
)
from marl_battlegrounds.rendering.pov_scene import (
    build_actor_pov_analyzer_projection_v1,
)
from marl_battlegrounds.rendering.scene import AgentSceneV1, AgentSceneV2
from marl_battlegrounds.rendering.vocabulary import (
    class_token_from_id,
    team_token_from_id,
)
from marl_battlegrounds.viewer.no_shared_visual import (
    build_live_no_shared_obs_visual_current_slice_v1,
)

_MOVEMENT_KEYS = {
    "w": MOVE_NORTH,
    "s": MOVE_SOUTH,
    "d": MOVE_EAST,
    "a": MOVE_WEST,
    "q": MOVE_NORTHWEST,
    "e": MOVE_NORTHEAST,
    "z": MOVE_SOUTHWEST,
    "c": MOVE_SOUTHEAST,
    "x": MOVE_STAY,
}
"""Step keys. Arrow keys are deliberately absent: they never move a champion."""

type RecordingRestartIntentV1 = Literal["reset", "combat_configuration"]


def recording_restart_intent_v1(
    session: DebuggerSession,
    command: DebuggerCommandV1,
    *,
    view_mode: ViewMode,
    include_stress: bool,
) -> RecordingRestartIntentV1 | None:
    """Classify a command that would replace the current recording episode.

    Parameters
    ----------
    session : DebuggerSession
        Current controller choices and recorded execution mode.
    command : DebuggerCommandV1
        Parsed command to classify before constructing a replacement session.
    view_mode : ViewMode
        Retained dispatch-context argument; it does not change restart classification.
    include_stress : bool
        Retained dispatch-context argument; it does not change restart classification.

    Returns
    -------
    {"reset", "combat_configuration"} or None
        Reset intent, a changed combat configuration, or None for other/no-change
        commands. Modified keyboard shortcuts that are not reset requests stay None.

    Notes
    -----
    This only describes intent. It does not construct a session or discard a recording.
    """
    del view_mode, include_stress
    if isinstance(command, SetCombatConfigurationCommandV1):
        if (
            command.team_a_controller == session.team_a_controller
            and command.team_b_controller == session.team_b_controller
            and command.execution_information_mode
            == session.evaluation_context.execution_information_mode
        ):
            return None
        return "combat_configuration"
    if isinstance(command, KeyboardCommandV1):
        if command.ctrl_key or command.alt_key or command.meta_key:
            return None
        key = normalize_key(command.key, shift_key=command.shift_key)
        if key == "r":
            return "reset"
        return None
    if isinstance(command, ResetCommandV1):
        return "reset"
    return None


_SUBMISSION_KEYS = frozenset(("space", "enter", "n"))

_SCRIPTED_INSPECTION_NOTICE = (
    "Scripted playback is inspection-only; press N without modifiers to advance "
    "the registered frame."
)


def _scripted_inspection_command_is_allowed(command: DebuggerCommandV1) -> bool:
    """Keep scripted playback to its closed advance and presentation surface."""
    if isinstance(command, KeyboardCommandV1):
        if (
            command.shift_key
            or command.ctrl_key
            or command.alt_key
            or command.meta_key
            or command.repeat
        ):
            return False
        return normalize_key(command.key, shift_key=False) in ("n", "g")
    return isinstance(command, (SetViewCommandV1, ExitCommandV1))


@dataclass(frozen=True, slots=True)
class InputDispatchResult:
    """One authoritative input outcome for the browser debugger service."""

    session: DebuggerSession
    view_mode: ViewMode
    preset: Preset
    handled: bool
    changed: bool
    transition_applied: EvaluationTransitionViewV1 | None = None
    episode_restarted: bool = False
    raw_continuation_identity: RawContinuationIdentity | None = None
    notice: str | None = None
    shutdown_requested: bool = False

    def __post_init__(self) -> None:
        """Require transition/restart markers to match the returned session and exact
        raw inputs.

        A result cannot both step and restart. UI-only edits must not claim a new raw
        simulator continuation.
        """
        transition = self.transition_applied
        if transition is not None:
            if type(transition) is not EvaluationTransitionViewV1:
                raise TypeError("transition_applied must be an exact coherent V1 view.")
            if self.episode_restarted:
                raise ValueError(
                    "one input result cannot both apply a transition and restart."
                )
            if not self.changed or self.session.incoming_evaluation_view != transition:
                raise ValueError(
                    "transition_applied must identify the candidate session view."
                )
        if type(self.episode_restarted) is not bool:
            raise TypeError("episode_restarted must be a Python bool.")
        if self.episode_restarted and (
            not self.changed
            or self.session.current_evaluation_frame.frame_index != 0
            or self.session.incoming_evaluation_view is not None
        ):
            raise ValueError(
                "restarted results must expose a fresh frame-zero episode."
            )
        has_scientific_marker = transition is not None or self.episode_restarted
        if has_scientific_marker:
            if (
                type(self.raw_continuation_identity) is not RawContinuationIdentity
                or self.raw_continuation_identity
                is not self.session.raw_continuation_identity
            ):
                raise ValueError(
                    "transition and restart results must bind the candidate raw "
                    "continuation identity."
                )
        elif self.raw_continuation_identity is not None:
            raise ValueError(
                "UI-only results must not carry a raw continuation identity."
            )


def normalize_key(
    key: str | None,
    *,
    shift_key: bool | None = None,
) -> str | None:
    """Normalize browser key aliases without applying an action.

    Parameters
    ----------
    key : str or None
        Browser key text, including supported historical spellings of Tab,
        Space, Enter and Escape.
    shift_key : bool or None, optional
        Explicit Shift state. None leaves modifier interpretation to the key text.

    Returns
    -------
    str or None
        Lowercase normalized key. None represents a missing key or the disabled
        Shift+R shortcut. Unknown names, including arrow keys, remain lowercase
        for dispatch to ignore.
    """
    if key is None:
        return None
    normalized = key.lower()
    if normalized == "shift+r" or (normalized == "r" and shift_key is True):
        return None
    if normalized in (
        "shift+tab",
        "backtab",
        "iso_left_tab",
        "shift+iso_left_tab",
    ):
        return "shift+tab"
    if normalized in (" ", "spacebar"):
        return "space"
    if normalized == "return":
        return "enter"
    if normalized == "esc":
        return "escape"
    if normalized == "tab" and shift_key:
        return "shift+tab"
    return normalized


def _hit_test_rows(
    rows: Iterable[tuple[int, tuple[float, float], float, bool]],
    x: float,
    y: float,
) -> int | None:
    """Choose the hit with smallest distance divided by body radius, breaking ties by
    slot.

    Ignore inactive rows, nonpositive radii and nonfinite pointer coordinates.
    Callers must supply only rows authorized for the current view.
    """
    if not isfinite(x) or not isfinite(y):
        return None
    candidates: list[tuple[float, int]] = []
    for global_slot, position, radius, active in rows:
        if not active or radius <= 0:
            continue
        normalized_distance = hypot(x - position[0], y - position[1]) / radius
        if normalized_distance <= 1.0:
            candidates.append((normalized_distance, global_slot))
    if not candidates:
        return None
    return min(candidates)[1]


def hit_test_scene_agents(
    agents: Iterable[AgentSceneV1 | AgentSceneV2],
    x: float,
    y: float,
) -> int | None:
    """Find an authorized scene agent whose body contains the pointer.

    Parameters
    ----------
    agents : iterable of AgentSceneV1 or AgentSceneV2
        Agents already authorized in the selected serialized scene.
    x : float
        Pointer x coordinate in map units.
    y : float
        Pointer y coordinate in map units.

    Returns
    -------
    int or None
        Global slot of the closest relative body hit, or None when nothing is hit.
        Overlaps choose the lowest distance/body-radius ratio, then the lowest slot.
        Nonfinite coordinates, inactive rows and nonpositive radii cannot be hit.

    Notes
    -----
    This performs no hidden-world lookup; the caller owns the authorized input list.
    """
    return _hit_test_rows(
        (
            (
                agent.global_slot,
                agent.position,
                agent.radius,
                agent.active if isinstance(agent, AgentSceneV1) else True,
            )
            for agent in agents
        ),
        x,
        y,
    )


def _result(
    session: DebuggerSession,
    *,
    view_mode: ViewMode,
    preset: Preset,
    handled: bool,
    changed: bool,
    transition_applied: EvaluationTransitionViewV1 | None = None,
    episode_restarted: bool = False,
    notice: str | None = None,
    shutdown_requested: bool = False,
) -> InputDispatchResult:
    """Package one dispatch result and bind raw inputs only for a step or restart."""
    return InputDispatchResult(
        session=session,
        view_mode=view_mode,
        preset=preset,
        handled=handled,
        changed=changed,
        transition_applied=transition_applied,
        episode_restarted=episode_restarted,
        raw_continuation_identity=(
            session.raw_continuation_identity
            if transition_applied is not None or episode_restarted
            else None
        ),
        notice=notice,
        shutdown_requested=shutdown_requested,
    )


def _applied_transition_result(
    session: DebuggerSession,
    *,
    view_mode: ViewMode,
    preset: Preset,
) -> InputDispatchResult:
    """Package one captured transition behind the typed validation boundary."""
    try:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
            transition_applied=session.incoming_evaluation_view,
        )
    except DebuggerTransitionFailureV1:
        raise
    except Exception as error:
        raise DebuggerTransitionFailureV1(
            "validation",
            "transition_packaging_failed",
        ) from error


def _pending_edit_result(
    before: DebuggerSession,
    after: DebuggerSession,
    *,
    view_mode: ViewMode,
    preset: Preset,
) -> InputDispatchResult:
    """Mark whether the pending draft changed without claiming a simulator
    transition.
    """
    changed = (
        before.controlled_global_slot != after.controlled_global_slot
        or before.pending_actions != after.pending_actions
    )
    return _result(
        after if changed else before,
        view_mode=view_mode,
        preset=preset,
        handled=True,
        changed=changed,
    )


def _is_terminal(session: DebuggerSession) -> bool:
    """Report whether the current session is terminated, truncated or at its declared
    horizon.
    """
    return session.episode_sealed


def _terminal_notice(session: DebuggerSession) -> str:
    """Explain why the current sealed endpoint cannot accept another simulator step."""
    reason = (
        "terminated"
        if session.terminated
        else "truncated"
        if session.truncated
        else "at its declared horizon"
    )
    return f"Episode is {reason}; reset or switch scenario to continue."


def _target_action(
    session: DebuggerSession,
    actor_global_slot: int,
    target_global_slot: int | None,
) -> int:
    """Resolve a global target through the recorded actor-relative target mapping."""
    if target_global_slot is None:
        return 0
    catalog = session.evaluation_context.static_mechanics_catalog
    mapping = catalog.global_recipient_slot_by_actor_and_target_action[
        actor_global_slot
    ]
    try:
        return mapping.index(target_global_slot)
    except ValueError as error:
        raise ValueError("target is absent from the serialized action axis") from error


def _authorized_pointer_rows(
    session: DebuggerSession,
    *,
    view_mode: ViewMode,
    recipient_global_slot: int | None = None,
) -> tuple[tuple[int, tuple[float, float], float, bool], ...]:
    """Build hit-test rows only from the selected researcher or actor-authorized
    scene.

    In Agent POV the rows come from recipient_global_slot's authorized scene
    (None means the focused slot), so only bodies that recipient's team sees
    can be hit. Each row is (global slot, position in map units, body radius,
    active). Builds host projections only; no simulator or file work.
    """
    recipient = (
        session.controlled_global_slot
        if recipient_global_slot is None
        else recipient_global_slot
    )
    if view_mode == "researcher":
        projection = build_researcher_analyzer_projection_v2(
            session.evaluation_context,
            session.current_evaluation_frame,
            transition_view=session.incoming_evaluation_view,
            presentation=EvaluationScenePresentationStateV1(
                controlled_global_slot=session.controlled_global_slot,
                selected_global_slot=session.controlled_global_slot,
                show_ranges=session.show_ranges,
            ),
            status_source_evidence_state=session.status_source_evidence_state,
        )
        return tuple(
            (agent.global_slot, agent.position, agent.radius, True)
            for agent in projection.scene.agents
        )
    if session.evaluation_context.execution_information_mode == "shared_obs":
        context = session.evaluation_context
        frame = session.current_evaluation_frame
        sources = tuple(
            build_shared_obs_authority_source_material_projection_v1(
                context,
                frame,
                selected_global_slot=row.global_slot,
            )
            for row in context.roster
            if row.configured_active
        )
        recipient_source = next(
            source
            for source in sources
            if source.base_sensor_scene.self_actor.global_slot == recipient
        )
        authorized = build_shared_obs_authorized_scene_v1(
            recipient_source,
            all_active_nonrecipient_source_material=tuple(
                source for source in sources if source is not recipient_source
            ),
            public_catalog=context.static_mechanics_catalog,
            authority_session_id="pointer-hit-test",
        )
        slot_by_public_id = {
            row.public_agent_id: row.global_slot for row in context.roster
        }
        return tuple(
            (
                slot_by_public_id[agent.public_agent_id],
                agent.position,
                agent.radius,
                True,
            )
            for agent in authorized.scene.agents
        )
    slice_ = build_live_no_shared_obs_visual_current_slice_v1(
        session.evaluation_context,
        session.current_evaluation_frame,
        global_slot=recipient,
        incoming_transition_view=session.incoming_evaluation_view,
    )
    projection = build_actor_pov_analyzer_projection_v1(slice_)
    slot_by_public_id = {
        row.public_agent_id: row.global_slot
        for row in session.evaluation_context.roster
    }
    self_actor = projection.scene.self_actor
    return (
        (
            self_actor.global_slot,
            self_actor.position,
            self_actor.radius,
            True,
        ),
        *(
            (
                slot_by_public_id[body.public_agent_id],
                body.position,
                body.radius,
                True,
            )
            for body in projection.scene.visible_bodies
        ),
    )


def _manual_edit_rejected(
    session: DebuggerSession, *, view_mode: ViewMode, preset: Preset
) -> InputDispatchResult:
    """Keep policy-owned focus inspectable while rejecting a human draft edit."""
    return _result(
        session,
        view_mode=view_mode,
        preset=preset,
        handled=True,
        changed=False,
        notice=(
            "This actor is controlled by a policy. "
            "Select a Manual actor to edit actions."
        ),
    )


def _agent_pov_focusable_slots(
    session: DebuggerSession,
    recipient_global_slot: int,
) -> frozenset[int]:
    """Return the slots Agent POV may focus: yours, your team, or seen.

    A configured-active slot may take the focus when it is Manual, on the
    recipient's team, or a body in the recipient's authorized scene (seen by
    that team now). Any other champion is unseen, and focusing it could reveal
    nothing it may show, so Tab skips it and a roster pick is refused.
    """
    roster = session.evaluation_context.roster
    viewing_team = roster[recipient_global_slot].configured_team_id
    seen = {
        row[0]
        for row in _authorized_pointer_rows(
            session,
            view_mode="pov",
            recipient_global_slot=recipient_global_slot,
        )
    }
    return frozenset(
        row.global_slot
        for row in roster
        if row.configured_active
        and (
            row.global_slot in session.manual_slots
            or row.configured_team_id == viewing_team
            or row.global_slot in seen
        )
    )


def _out_of_sight_notice(session: DebuggerSession, global_slot: int) -> str:
    """Explain a refused Agent POV focus without naming where the champion is."""
    row = session.evaluation_context.roster[global_slot]
    label = f"{class_token_from_id(row.class_id).label} · ID {global_slot}"
    return f"{label} is out of sight. Agent POV hides where it is and how it fares."


def _focus_notice(
    session: DebuggerSession, global_slot: int, *, view_mode: ViewMode
) -> str | None:
    """Explain an Agent POV focus that leaves the view where it is.

    Only in Agent POV with Manual champions in the game, when the focus lands
    on a policy champion: the view stays with your Manual champion, so the
    notice names the champion looked at (class, ID and team, all public) and
    says the view is kept. None in every other case, including Oracle View.
    """
    if (
        view_mode != "pov"
        or session.scenario.mode == "scripted"
        or not session.manual_slots
        or global_slot in session.manual_slots
    ):
        return None
    row = session.evaluation_context.roster[global_slot]
    label = f"{class_token_from_id(row.class_id).label} · ID {global_slot}"
    team = team_token_from_id(row.configured_team_id).label
    return (
        f"Looking at {label} of {team}, played by a policy. "
        "Agent POV keeps your champion's view."
    )


def _dispatch_keyboard(
    session: DebuggerSession,
    command: KeyboardCommandV1,
    *,
    view_mode: ViewMode,
    preset: Preset,
    include_stress: bool,
    pov_global_slot: int | None = None,
) -> InputDispatchResult:
    """Apply supported unmodified keys while preserving draft, terminal and playback
    boundaries.

    pov_global_slot is the Agent POV recipient; in Agent POV, Tab and
    Shift+Tab visit only the slots from _agent_pov_focusable_slots. Arrow keys
    and V are unhandled. N is one tick: in scripted playback it submits the
    next registered frame; in interactive play it takes Space's own path
    (submit_interactive), so it locks in the Manual orders, or plays one turn
    in Policies Only, and is refused with Space's words at game over.
    """
    if command.ctrl_key or command.alt_key or command.meta_key:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=False,
            changed=False,
        )
    key = normalize_key(command.key, shift_key=command.shift_key)
    if key is None:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=False,
            changed=False,
        )
    if command.repeat and key in _SUBMISSION_KEYS:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice="Repeated submission input ignored.",
        )

    if key in ("tab", "shift+tab"):
        direction = -1 if key == "shift+tab" else 1
        active_slots = tuple(
            row.global_slot
            for row in session.evaluation_context.roster
            if row.configured_active
        )
        if view_mode == "pov":
            recipient = (
                session.controlled_global_slot
                if pov_global_slot is None
                else pov_global_slot
            )
            focusable = _agent_pov_focusable_slots(session, recipient)
        else:
            focusable = frozenset(active_slots)
        current_index = active_slots.index(session.controlled_global_slot)
        controlled_slot = next(
            (
                slot
                for step in range(1, len(active_slots) + 1)
                if (
                    slot := active_slots[
                        (current_index + direction * step) % len(active_slots)
                    ]
                )
                in focusable
            ),
            session.controlled_global_slot,
        )
        edited = select_controlled_actor(session, controlled_slot)
        result = _pending_edit_result(
            session,
            edited,
            view_mode=view_mode,
            preset=preset,
        )
        notice = (
            _focus_notice(session, controlled_slot, view_mode=view_mode)
            if result.changed
            else None
        )
        return result if notice is None else replace(result, notice=notice)

    if (
        (key in _MOVEMENT_KEYS or key in ("escape", "0", "1", "2"))
        and session.scenario.mode != "scripted"
        and session.controlled_global_slot not in session.manual_slots
    ):
        return _manual_edit_rejected(session, view_mode=view_mode, preset=preset)

    if key == "escape":
        return _pending_edit_result(
            session,
            clear_pending_target(session),
            view_mode=view_mode,
            preset=preset,
        )

    scenario = session.scenario
    terminal = _is_terminal(session)
    if scenario.mode == "scripted" and (
        key in _MOVEMENT_KEYS or key in ("0", "1", "2", "space", "enter")
    ):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=(
                "Scripted playback is inspection-only; press N to advance "
                "the registered frame."
            ),
        )
    if key in _MOVEMENT_KEYS:
        if terminal:
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=_terminal_notice(session),
            )
        move_action = _MOVEMENT_KEYS[key]
        if not bool(
            session.current_evaluation_frame.action_mask.move_mask[
                session.controlled_global_slot
            ][move_action]
        ):
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=(
                    f"Movement action {move_action} is unavailable in the "
                    "current action mask; pending action unchanged."
                ),
            )
        return _pending_edit_result(
            session,
            set_pending_movement(session, move_action),
            view_mode=view_mode,
            preset=preset,
        )
    if key in ("0", "1", "2"):
        if terminal:
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=_terminal_notice(session),
            )
        if key == "0":
            return _pending_edit_result(
                session,
                select_no_combat(session),
                view_mode=view_mode,
                preset=preset,
            )
        edited = arm_basic(session) if key == "1" else arm_ultimate(session)
        edited_pending = edited.pending_action
        target_action = _target_action(
            edited,
            edited.controlled_global_slot,
            edited_pending.selected_global_target_slot,
        )
        if key == "1" and target_action == 0:
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=(
                    "Basic requires a selected target; target-none lane zero is "
                    "the canonical no-combat tuple. Pending action unchanged."
                ),
            )
        availability = lane_availability(
            session.current_evaluation_frame.action_mask,
            session.controlled_global_slot,
            target_action,
            edited_pending.armed_lane,
        )
        if not availability.armed_pair_legal:
            ability = "Basic" if key == "1" else "Ultimate"
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=(
                    f"{ability} is unavailable for the selected target in the "
                    "current action mask; pending action unchanged."
                ),
            )
        return _pending_edit_result(
            session,
            edited,
            view_mode=view_mode,
            preset=preset,
        )

    if key == "n" and scenario.mode == "scripted":
        edited = submit_next_script_frame(session)
        changed = edited is not session
        if changed:
            return _applied_transition_result(
                edited,
                view_mode=view_mode,
                preset=preset,
            )
        notice = (
            _terminal_notice(session)
            if terminal
            else "Script is already complete."
            if edited is session
            else None
        )
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=changed,
            notice=notice,
        )
    if key in ("space", "enter", "n"):
        # N is one tick: in interactive play it is Space, through the same
        # path (Manual: lock in the orders; Policies Only: play one turn).
        edited = submit_interactive(session, actor_global_slots=None)
        transition_applied = edited is not session
        if transition_applied:
            return _applied_transition_result(
                edited,
                view_mode=view_mode,
                preset=preset,
            )
        changed = edited is not session
        notice = (
            _terminal_notice(session)
            if terminal
            else "Script is already complete."
            if edited is session
            else None
        )
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=changed,
            notice=notice,
        )
    if key == "r":
        return _result(
            reset_session(session),
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
            episode_restarted=True,
        )
    if key == "g":
        return _result(
            replace(session, show_ranges=not session.show_ranges),
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
        )
    if key in ("[", "]"):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=("Scenario navigation moved to the read-only Replay Viewer."),
        )
    return _result(
        session,
        view_mode=view_mode,
        preset=preset,
        handled=False,
        changed=False,
    )


def _dispatch_pointer(
    session: DebuggerSession,
    command: BattlefieldPointerCommandV1,
    *,
    view_mode: ViewMode,
    preset: Preset,
    pov_global_slot: int | None = None,
) -> InputDispatchResult:
    """Resolve an authorized pointer hit and apply the requested control or target
    edit.

    In Agent POV the hit test uses pov_global_slot's scene (None: the focus),
    so a click can focus only a body that team sees.
    """
    if command.ctrl_key or command.alt_key or command.meta_key:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=False,
            changed=False,
        )
    if (
        (command.button == "secondary" or command.shift_key)
        and session.scenario.mode != "scripted"
        and session.controlled_global_slot not in session.manual_slots
    ):
        return _manual_edit_rejected(session, view_mode=view_mode, preset=preset)
    if command.button == "secondary":
        return _pending_edit_result(
            session,
            clear_pending_target(session),
            view_mode=view_mode,
            preset=preset,
        )
    target = _hit_test_rows(
        _authorized_pointer_rows(
            session,
            view_mode=view_mode,
            recipient_global_slot=pov_global_slot if view_mode == "pov" else None,
        ),
        command.world_x,
        command.world_y,
    )
    if target is None:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
        )
    edited = (
        select_clicked_target(session, target)
        if command.shift_key
        else select_controlled_actor(session, target)
    )
    result = _pending_edit_result(
        session,
        edited,
        view_mode=view_mode,
        preset=preset,
    )
    notice = (
        _focus_notice(session, target, view_mode=view_mode)
        if result.changed and not command.shift_key
        else None
    )
    return result if notice is None else replace(result, notice=notice)


def _dispatch_roster_selection(
    session: DebuggerSession,
    command: RosterSelectionCommandV1,
    *,
    view_mode: ViewMode,
    preset: Preset,
    pov_global_slot: int | None = None,
) -> InputDispatchResult:
    """Apply researcher roster selection and reject its use as an actor-view
    disclosure.

    In Agent POV a focus pick (role 'control') of a slot outside
    _agent_pov_focusable_slots for pov_global_slot is refused with a notice
    that names only the champion's class and ID.
    """
    authorized_slots = {
        row.global_slot
        for row in session.evaluation_context.roster
        if row.configured_active
    }
    if command.global_slot not in authorized_slots:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=f"Agent g{command.global_slot} is unavailable in this view.",
        )
    if (
        command.role == "control"
        and view_mode == "pov"
        and command.global_slot
        not in _agent_pov_focusable_slots(
            session,
            session.controlled_global_slot
            if pov_global_slot is None
            else pov_global_slot,
        )
    ):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=_out_of_sight_notice(session, command.global_slot),
        )
    if (
        command.role != "control"
        and session.scenario.mode != "scripted"
        and session.controlled_global_slot not in session.manual_slots
    ):
        return _manual_edit_rejected(session, view_mode=view_mode, preset=preset)
    edited = (
        select_controlled_actor(session, command.global_slot)
        if command.role == "control"
        else select_clicked_target(session, command.global_slot)
    )
    result = _pending_edit_result(
        session,
        edited,
        view_mode=view_mode,
        preset=preset,
    )
    notice = (
        _focus_notice(session, command.global_slot, view_mode=view_mode)
        if result.changed and command.role == "control"
        else None
    )
    return result if notice is None else replace(result, notice=notice)


def _dispatch_actor_pov_target_action(
    session: DebuggerSession,
    command: ActorPovTargetActionCommandV1,
    *,
    view_mode: ViewMode,
    preset: Preset,
) -> InputDispatchResult:
    """Resolve one recipient-relative POV target action on the trusted host."""
    if view_mode != "pov":
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice="Actor-relative target selection is available only in POV.",
        )
    if (
        session.scenario.mode != "scripted"
        and session.controlled_global_slot not in session.manual_slots
    ):
        return _manual_edit_rejected(session, view_mode=view_mode, preset=preset)
    if command.target_action == 0:
        return _pending_edit_result(
            session,
            clear_pending_target(session),
            view_mode=view_mode,
            preset=preset,
        )

    mapping = session.evaluation_context.static_mechanics_catalog
    recipients = mapping.global_recipient_slot_by_actor_and_target_action[
        session.controlled_global_slot
    ]
    target_global_slot = recipients[command.target_action]
    authorized_slots = {
        row.global_slot
        for row in session.evaluation_context.roster
        if row.configured_active
    }
    if target_global_slot not in authorized_slots:
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=(
                f"Target action {command.target_action} is unavailable in the "
                "current authorized POV."
            ),
        )
    return _pending_edit_result(
        session,
        select_clicked_target(session, target_global_slot),
        view_mode=view_mode,
        preset=preset,
    )


def _is_order_command(command: DebuggerCommandV1) -> bool:
    """Return True for input that edits a champion's draft (an order)."""
    if isinstance(command, KeyboardCommandV1):
        if command.ctrl_key or command.alt_key or command.meta_key:
            return False
        key = normalize_key(command.key, shift_key=command.shift_key)
        return key in _MOVEMENT_KEYS or key in ("escape", "0", "1", "2")
    if isinstance(command, BattlefieldPointerCommandV1):
        return command.button == "secondary" or command.shift_key
    if isinstance(command, RosterSelectionCommandV1):
        return command.role != "control"
    return isinstance(command, ActorPovTargetActionCommandV1)


def dispatch_command(
    session: DebuggerSession,
    command: DebuggerCommandV1,
    *,
    view_mode: ViewMode,
    preset: Preset,
    include_stress: bool,
    pov_global_slot: int | None = None,
) -> InputDispatchResult:
    """Apply one parsed live command through the correct host or control helper.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable simulator endpoint and pending joint draft.
    command : DebuggerCommandV1
        Strict command model already parsed by the HTTP boundary.
    view_mode : ViewMode
        Current researcher or actor view, used to preserve input authorization.
    preset : Preset
        Current presentation choice, copied or updated by the command.
    include_stress : bool
        Whether the surrounding launcher permits stress demonstrations.
    pov_global_slot : int or None, optional
        The Agent POV recipient from control.agent_pov_recipient_slot, a
        configured-active slot 0-9. Used only in Agent POV: clicks hit the
        bodies its team sees, Tab skips champions that team cannot see, and
        orders go to it (an order while the focus rests on another champion
        moves the focus back to the recipient first). None (the default) means
        the focused slot, the behaviour before the recipient existed.

    Returns
    -------
    InputDispatchResult
        Candidate session and presentation settings, handled/changed flags, notice,
        shutdown request, and explicit transition or restart markers where applicable.

    Raises
    ------
    DebuggerTransitionFailureV1
        If a delegated simulator submission or its capture/packaging fails.

    Notes
    -----
    Scripted sessions accept only their supported playback and presentation commands.
    Input handling delegates RNG and simulator semantics to the control layer; it does
    not write files or publish the candidate endpoint itself.
    """
    scripted_inspection = session.scenario.mode == "scripted"
    if scripted_inspection and not _scripted_inspection_command_is_allowed(command):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=_SCRIPTED_INSPECTION_NOTICE,
        )
    if (
        view_mode == "pov"
        and pov_global_slot is not None
        and pov_global_slot != session.controlled_global_slot
        and _is_order_command(command)
    ):
        # Agent POV shows the recipient, so its orders go to the recipient.
        refocused = select_controlled_actor(session, pov_global_slot)
        result = dispatch_command(
            refocused,
            command,
            view_mode=view_mode,
            preset=preset,
            include_stress=include_stress,
            pov_global_slot=pov_global_slot,
        )
        if result.changed:
            return result
        return replace(result, session=refocused, changed=True)
    if isinstance(command, KeyboardCommandV1):
        return _dispatch_keyboard(
            session,
            command,
            view_mode=view_mode,
            preset=preset,
            include_stress=include_stress,
            pov_global_slot=pov_global_slot,
        )
    if isinstance(command, BattlefieldPointerCommandV1):
        return _dispatch_pointer(
            session,
            command,
            view_mode=view_mode,
            preset=preset,
            pov_global_slot=pov_global_slot,
        )
    if isinstance(command, RosterSelectionCommandV1):
        return _dispatch_roster_selection(
            session,
            command,
            view_mode=view_mode,
            preset=preset,
            pov_global_slot=pov_global_slot,
        )
    if isinstance(command, ActorPovTargetActionCommandV1):
        return _dispatch_actor_pov_target_action(
            session,
            command,
            view_mode=view_mode,
            preset=preset,
        )
    if isinstance(command, ScenarioSwitchCommandV1):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice=("Scenario switching moved to the read-only Replay Viewer."),
        )
    if isinstance(command, ResetCommandV1):
        edited = reset_session(session)
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=True,
            episode_restarted=True,
        )
    if isinstance(command, SetCombatConfigurationCommandV1):
        try:
            edited = set_combat_configuration(
                session,
                team_a_controller=command.team_a_controller,
                team_b_controller=command.team_b_controller,
                execution_information_mode=command.execution_information_mode,
            )
        except CombatConfigurationRejectedError as error:
            return _result(
                session,
                view_mode=view_mode,
                preset=preset,
                handled=True,
                changed=False,
                notice=str(error),
            )
        changed = edited is not session
        return _result(
            edited,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=changed,
            episode_restarted=changed,
        )
    if isinstance(command, SetViewCommandV1):
        edited = session
        changed = command.view_mode != view_mode
        return _result(
            edited,
            view_mode=command.view_mode,
            preset=preset,
            handled=True,
            changed=changed,
        )
    if isinstance(command, SetPresetCommandV1):
        return _result(
            session,
            view_mode=view_mode,
            preset=command.preset,
            handled=True,
            changed=command.preset != preset,
        )
    if isinstance(command, ExitCommandV1):
        return _result(
            session,
            view_mode=view_mode,
            preset=preset,
            handled=True,
            changed=False,
            notice="Debugger shutdown requested.",
            shutdown_requested=True,
        )
    return _result(
        session,
        view_mode=view_mode,
        preset=preset,
        handled=True,
        changed=False,
        notice="Replay recording is not enabled for this debugger session.",
    )
