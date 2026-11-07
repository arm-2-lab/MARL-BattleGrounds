"""Validate the joins between captured episode, frame, and transition records.

These host-only checks compare declared wire types, stable identifiers, decision
epochs, inactive padding, information sharing, task completion, and canonical
events. They read Core-authored facts; they do not replay simulator physics.
Historical frame versions remain paired with their declared context version.
Capture the Flag records (context V5, frame V4, transition V2) get the same
checks with their own task authority and context columns, written to the C1.C
contract.
"""

from typing import cast

from pydantic import ValidationError

from marl_battlegrounds.evaluation._recording.events import (
    _derive_capture_the_flag_authority,  # pyright: ignore[reportPrivateUsage]
    _derive_team_deathmatch_authority_v1,  # pyright: ignore[reportPrivateUsage]
    decode_evaluation_events_v1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContext,
    EvaluationEpisodeContextV4,
    EvaluationEpisodeContextV5,
    EvaluationFrame,
    EvaluationFrameV4,
    EvaluationModel,
    EvaluationTransition,
    GlobalAnalysisSnapshotCaptureTheFlagV2,
    TransitionFactsCaptureTheFlagV2,
    TransitionFactsV1,
    evaluation_context_type,
    evaluation_frame_type_for_context,
    evaluation_transition_type_for_context,
    float32_value,
    recorded_score_threshold,
    recorded_team_scores,
)

# These are frozen V1 wire-column coordinates, duplicated intentionally so
# replay validation remains host-only and never imports JAX-backed core types.
_CONTEXT_FEATURE_CURRENT_TIMESTEP_V1 = 0
_CONTEXT_FEATURE_EPISODE_HORIZON_V1 = 1
_CONTEXT_FEATURE_IS_TDM_V1 = 6
_CONTEXT_FEATURE_TDM_SCORE_THRESHOLD_V1 = 16
# Frame V3 (context V4) appends the Team Deathmatch Red Zone depth.
_CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH_V2 = 19


def _is_canonical_neutral(value: object) -> bool:
    """Recognize nested tuple padding made only of None, false, or numeric zero.

    Lists, dictionaries, strings, and other objects are not canonical padding.
    Boolean and numeric types are checked exactly so coercible objects do not pass.
    """
    if value is None:
        return True
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return all(_is_canonical_neutral(item) for item in items)
    if type(value) is bool:
        return not value
    if type(value) in (int, float):
        return value == 0
    return False


def _require_inactive_slot_neutral(
    value: object,
    *,
    global_slot: int,
    field_name: str,
) -> None:
    """Raise ValueError when a named inactive-slot value is not canonical padding.

    The slot index and field name are used only to make the error actionable.
    """
    if not _is_canonical_neutral(value):
        raise ValueError(f"inactive slot {global_slot} must be neutral in {field_name}")


def _validate_inactive_frame_padding(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
) -> None:
    """Check zero padding in every dynamic snapshot row for inactive roster slots.

    This checks recorded snapshot values only. It neither reconstructs actor inputs
    nor changes the frame. A nonneutral row raises ValueError.
    """
    snapshot = frame.snapshot
    slot_fields = (
        "alive_mask",
        "agent_positions",
        "current_health",
        "ultimate_cooldowns",
        "slow_durations",
        "stun_durations",
        "rogue_poison_anti_heal_durations",
        "mage_burst_damage_amplification_durations",
        "priest_blessing_of_freedom_slow_floor_durations",
        "spawn_shield_durations",
        "steps_until_out_of_combat",
        "previous_timestep_move_actions",
        "previous_timestep_select_target_actions",
        "previous_timestep_use_ultimate_actions",
    )
    for global_slot, roster_row in enumerate(context.roster):
        if roster_row.configured_active:
            continue
        for field_name in slot_fields:
            _require_inactive_slot_neutral(
                getattr(snapshot, field_name)[global_slot],
                global_slot=global_slot,
                field_name=f"frame.snapshot.{field_name}",
            )
        if isinstance(snapshot, GlobalAnalysisSnapshotCaptureTheFlagV2) and (
            global_slot in snapshot.capture_the_flag_flag_carrier_slots
        ):
            raise ValueError(f"inactive slot {global_slot} cannot carry a flag")


def _validate_inactive_fact_padding(
    context: EvaluationEpisodeContext,
    transition: EvaluationTransition,
) -> None:
    """Check inactive rows and source/recipient routes in Core-authored facts.

    Accepted actions, combat, death, shield, regeneration, movement, status, respawn,
    aura and Capture the Flag agent facts must be neutral for inactive slots.
    Combat routes cannot name an inactive recipient. Submitted intent is
    deliberately retained without this neutrality rule. Violations raise
    ValueError; records are not changed.
    """
    facts = transition.facts
    acceptance = facts.action_acceptance_facts
    combat = facts.combat_transition_facts
    source_aligned_combat_fields = (
        "basic_effect_is_activated_by_source",
        "ultimate_effect_is_activated_by_source",
        "combat_effect_has_recipient_by_source",
        "combat_effect_recipient_global_slot_by_source",
        "raw_damage_output_by_source",
        "source_modified_damage_output_by_source",
        "recipient_damage_modifier_by_source",
        "raw_healing_output_by_source",
        "source_modified_healing_output_by_source",
        "recipient_healing_modifier_by_source",
        "slow_is_applied_by_source_and_channel",
        "stun_is_applied_by_source_and_channel",
        "rogue_poison_anti_heal_is_applied_by_source",
        "mage_burst_damage_amplification_is_applied_by_source",
        "priest_blessing_of_freedom_is_applied_by_source",
    )
    recipient_aligned_combat_fields = (
        "total_effective_damage_by_recipient",
        "total_effective_healing_by_recipient",
        "health_after_combat_resolution_by_recipient",
    )
    per_agent_fact_fields = (
        (
            facts.death_facts,
            (
                "is_newly_dead_by_recipient",
                "contributed_to_new_death_by_source",
                "attributed_death_damage_by_source",
            ),
            "death_facts",
        ),
        (
            facts.spawn_shield_facts,
            (
                "was_active_at_transition_start_by_agent",
                "expired_at_transition_end_by_agent",
            ),
            "spawn_shield_facts",
        ),
        (
            facts.regeneration_facts,
            (
                "combat_countdown_was_reset_by_agent",
                "actual_health_regenerated_this_step_by_agent",
            ),
            "regeneration_facts",
        ),
        (
            facts.physical_facts,
            (
                "charge_phase_displacement_by_agent",
                "ordinary_movement_phase_displacement_by_agent",
            ),
            "physical_facts",
        ),
        (
            facts.status_lifecycle_facts,
            (
                "aged_to_zero_by_recipient_and_status_channel",
                "refreshed_or_extended_by_recipient_and_status_channel",
                "broken_by_damage_by_recipient_and_status_channel",
                "cleared_by_new_death_by_recipient_and_status_channel",
            ),
            "status_lifecycle_facts",
        ),
    )
    if isinstance(facts, TransitionFactsCaptureTheFlagV2):
        flag_facts = facts.capture_the_flag_facts
        per_agent_fact_fields += (
            (
                flag_facts,
                (
                    "flag_was_picked_up_by_agent",
                    "flag_was_dropped_by_agent",
                    "ally_flag_was_returned_by_agent",
                    "enemy_flag_was_captured_by_agent",
                ),
                "capture_the_flag_facts",
            ),
        )

    for recipient_global_slot in combat.combat_effect_recipient_global_slot_by_source:
        if (
            recipient_global_slot is not None
            and not context.roster[recipient_global_slot].configured_active
        ):
            raise ValueError(
                "combat recipient routes must not resolve to an inactive slot"
            )

    for global_slot, roster_row in enumerate(context.roster):
        if roster_row.configured_active:
            continue
        accepted = acceptance.accepted_joint_action
        for field_name in ("move", "select_target", "use_ultimate"):
            _require_inactive_slot_neutral(
                getattr(accepted, field_name)[global_slot],
                global_slot=global_slot,
                field_name=f"action_acceptance_facts.accepted_joint_action.{field_name}",
            )
        for field_name in source_aligned_combat_fields:
            _require_inactive_slot_neutral(
                getattr(combat, field_name)[global_slot],
                global_slot=global_slot,
                field_name=f"combat_transition_facts.{field_name}",
            )
        for field_name in recipient_aligned_combat_fields:
            _require_inactive_slot_neutral(
                getattr(combat, field_name)[global_slot],
                global_slot=global_slot,
                field_name=f"combat_transition_facts.{field_name}",
            )
        for model, field_names, subtree_name in per_agent_fact_fields:
            for field_name in field_names:
                _require_inactive_slot_neutral(
                    getattr(model, field_name)[global_slot],
                    global_slot=global_slot,
                    field_name=f"{subtree_name}.{field_name}",
                )
        _require_inactive_slot_neutral(
            facts.respawn_facts.was_respawned_this_transition_by_agent[global_slot],
            global_slot=global_slot,
            field_name="respawn_facts.was_respawned_this_transition_by_agent",
        )
        for field_name in (
            "is_covered_by_mage_damage_aura_by_emitter_and_beneficiary",
            "is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary",
        ):
            coverage = getattr(facts.aura_facts, field_name)
            _require_inactive_slot_neutral(
                coverage[global_slot],
                global_slot=global_slot,
                field_name=f"aura_facts.{field_name} emitter row",
            )
            _require_inactive_slot_neutral(
                tuple(row[global_slot] for row in coverage),
                global_slot=global_slot,
                field_name=f"aura_facts.{field_name} beneficiary column",
            )


def _matches_declared_model_tree(candidate: object, canonical: object) -> bool:
    """Compare a validated model tree with an input tree using exact Python types.

    Reject model subclasses, extra/private model state, changed fields, and primitive
    coercions. Recursively compare tuples and dictionaries without relying on
    Pydantic model equality. Return a boolean and leave both trees unchanged.
    """
    if isinstance(canonical, EvaluationModel):
        if type(candidate) is not type(canonical):
            return False
        if getattr(candidate, "__pydantic_private__", None):
            return False
        if getattr(candidate, "__pydantic_extra__", None):
            return False
        if candidate.__dict__.keys() != canonical.__dict__.keys():
            return False
        return all(
            _matches_declared_model_tree(
                getattr(candidate, field_name),
                getattr(canonical, field_name),
            )
            for field_name in type(canonical).model_fields
        )
    if isinstance(canonical, tuple):
        if type(candidate) is not tuple:
            return False
        candidate_items = cast(tuple[object, ...], candidate)
        canonical_items = cast(tuple[object, ...], canonical)
        return len(candidate_items) == len(canonical_items) and all(
            _matches_declared_model_tree(candidate_item, canonical_item)
            for candidate_item, canonical_item in zip(
                candidate_items,
                canonical_items,
                strict=True,
            )
        )
    if isinstance(canonical, dict):
        if type(candidate) is not dict:
            return False
        candidate_items = cast(dict[object, object], candidate)
        canonical_items = cast(dict[object, object], canonical)
        if candidate_items.keys() != canonical_items.keys():
            return False
        return all(
            _matches_declared_model_tree(candidate_items[key], canonical_value)
            for key, canonical_value in canonical_items.items()
        )
    return type(candidate) is type(canonical) and candidate == canonical


def validate_declared_model_tree(
    model: EvaluationModel,
    *,
    record_name: str,
    expected_type: type[EvaluationModel],
) -> EvaluationModel:
    """Revalidate one record without accepting hidden types or coercions.

    Parameters
    ----------
    model : EvaluationModel
        Existing evaluation model to check, including all nested values.
    record_name : str
        Human-readable record label used in errors.
    expected_type : type[EvaluationModel]
        Exact permitted root model class; subclasses are rejected.

    Returns
    -------
    EvaluationModel
        Newly validated model reconstructed from the input's Python-mode field dump.

    Raises
    ------
    ValueError
        The root type differs, structural validation fails, or nested
        types, private/extra state, or values change during revalidation.

    Notes
    -----
    This host boundary also rejects invalid objects made with unchecked Pydantic
    construction. It does not mutate the input or accept merely equal subclasses.
    """
    if type(model) is not expected_type:
        raise ValueError(
            f"{record_name} must use exact declared root type {expected_type.__name__}"
        )
    try:
        reconstructed = expected_type.model_validate(
            model.model_dump(mode="python", warnings=False)
        )
    except ValidationError as error:
        raise ValueError(f"{record_name} fails structural revalidation") from error
    if type(reconstructed) is not expected_type or not _matches_declared_model_tree(
        model,
        reconstructed,
    ):
        raise ValueError(
            f"{record_name} contains an undeclared nested model type or changes "
            "under structural revalidation"
        )
    return reconstructed


def _validate_frame_information_regime(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
) -> None:
    """Require the context's frame version and allowed SharedObs source cells.

    Context V4 requires frame V3, context V3 frame V2 and older contexts frame V1
    (evaluation_frame_type_for_context). NoSharedObs omits availability. SharedObs
    supplies it and forbids self, inactive, and cross-team cells. Other same-team
    cells may be false. Mismatches raise ValueError.
    """
    expected_frame_type = evaluation_frame_type_for_context(context)
    if type(frame) is not expected_frame_type:
        raise ValueError("frame version does not match the episode context")
    availability = (
        frame.shared_obs_information_availability_by_recipient_and_sensor_source
    )
    if context.execution_information_mode == "no_shared_obs":
        if availability is not None:
            raise ValueError("no_shared_obs frames must omit SharedObs availability")
        return
    if availability is None:
        raise ValueError("shared_obs frames require SharedObs availability")

    for recipient, recipient_row in enumerate(context.roster):
        for sensor_source, source_row in enumerate(context.roster):
            is_forbidden = (
                recipient == sensor_source
                or not recipient_row.configured_active
                or not source_row.configured_active
                or recipient_row.configured_team_id != source_row.configured_team_id
            )
            if is_forbidden and availability[recipient][sensor_source]:
                raise ValueError(
                    "SharedObs availability must be false on diagonal, cross-team, "
                    "and inactive recipient/source cells"
                )


def _validate_task_context_projection(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
) -> None:
    """Check policy-visible task fields against recorded snapshot/configuration facts.

    For each active slot, verify tick, horizon, mode, team-relative scores, threshold,
    and unused task padding; for context V4 also check that column 19 equals the
    float32 value of the recorded Red Zone depth. A Capture the Flag row (context
    V5) follows C1.C: column 8 is 1, column 9 the four objective rows, columns 14
    and 15 the observer's own and the enemy's captures, column 18 the capture
    threshold, and every other task column, the depth included, 0. Inactive
    context rows must be neutral. Raise ValueError on disagreement; these frozen
    wire coordinates keep this reader free of JAX.
    """
    config = context.resolved_env_config
    scores = recorded_team_scores(frame.snapshot)
    threshold = float(recorded_score_threshold(config))
    is_team_deathmatch = config.task_mode == 1
    is_capture_the_flag = isinstance(context, EvaluationEpisodeContextV5)
    for global_slot, (roster_row, context_row) in enumerate(
        zip(context.roster, frame.base_observation.context_features, strict=True)
    ):
        if not roster_row.configured_active:
            _require_inactive_slot_neutral(
                context_row,
                global_slot=global_slot,
                field_name="base_observation.context_features",
            )
            continue

        if context_row[_CONTEXT_FEATURE_CURRENT_TIMESTEP_V1] != float(
            frame.simulator_step_count
        ):
            raise ValueError("policy context timestep must match the snapshot epoch")
        if context_row[_CONTEXT_FEATURE_EPISODE_HORIZON_V1] != float(
            config.maximum_episode_steps
        ):
            raise ValueError("policy context horizon must match resolved configuration")

        own = float(scores[roster_row.configured_team_id - 1])
        enemy = float(scores[2 - roster_row.configured_team_id])
        # Columns 6-18: is TDM, KOTH, CTF; active objective rows; TDM, KOTH and
        # CTF scores (own, enemy); TDM, KOTH and CTF thresholds.
        expected_task_slice = (
            (0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, own, enemy, 0.0, 0.0, threshold)
            if is_capture_the_flag
            else (
                1.0 if is_team_deathmatch else 0.0,
                0.0,
                0.0,
                0.0,
                own,
                enemy,
                0.0,
                0.0,
                0.0,
                0.0,
                threshold,
                0.0,
                0.0,
            )
        )
        actual_task_slice = context_row[
            _CONTEXT_FEATURE_IS_TDM_V1 : _CONTEXT_FEATURE_TDM_SCORE_THRESHOLD_V1 + 3
        ]
        if actual_task_slice != expected_task_slice:
            raise ValueError(
                "policy context task mode, scores, or threshold conflicts with "
                "resolved task authority"
            )
        if type(context) is EvaluationEpisodeContextV4 and context_row[
            _CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH_V2
        ] != float32_value(context.resolved_env_config.team_deathmatch_red_zone_depth):
            raise ValueError(
                "policy context Red Zone depth must match resolved configuration"
            )
        if is_capture_the_flag and context_row[_CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH_V2]:
            raise ValueError("Capture the Flag context rows keep the Red Zone depth 0")


def derive_and_validate_task_authority(
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrame,
    facts: TransitionFactsV1 | TransitionFactsCaptureTheFlagV2,
    successor_frame: EvaluationFrame,
    canonical_reward_by_agent: tuple[float, ...],
    terminated: bool,
    truncated: bool,
) -> tuple[tuple[float, float] | None, str | None]:
    """Check rewards and completion flags against the mode's joined task facts.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact supported context; a V5 context is a Capture the Flag game, the
        others Team Deathmatch or neutral games.
    start_frame, successor_frame : EvaluationFrame
        The adjacent frames of the context's frame version.
    facts : TransitionFactsV1 | TransitionFactsCaptureTheFlagV2
        Real transition facts of the context's version.
    canonical_reward_by_agent : tuple[float, ...]
        Ten recorded rewards in global-slot order.
    terminated, truncated : bool
        The recorded done flags.

    Returns
    -------
    tuple[tuple[float, float] | None, str | None]
        Per-team reward and owning end reason for Team Deathmatch and Capture
        the Flag, or (None, None) for task-neutral transitions.

    Raises
    ------
    ValueError
        Flags, rewards, scores or the outcome disagree with the recorded facts.

    Notes
    -----
    Both scored modes reward a win +1 and a loss -1 (a draw and an ongoing
    game 0); active slots get their configured team's reward and inactive
    slots zero. terminated equals threshold completion and truncated horizon
    completion; both may hold. End reasons are the mode's prefix plus
    _score_threshold, _horizon or _score_threshold_at_horizon. Recorded Core
    facts only; nothing is rerun. The Capture the Flag rule is checked
    against Core's own rewards at C1.3's acceptance.
    """
    if isinstance(context, EvaluationEpisodeContextV5):
        authority = _derive_capture_the_flag_authority(
            context,
            cast(EvaluationFrameV4, start_frame),
            cast(TransitionFactsCaptureTheFlagV2, facts),
            cast(EvaluationFrameV4, successor_frame),
        )
        mode_name, short_name, end_prefix = (
            "Capture the Flag",
            "CTF",
            "capture_the_flag",
        )
    else:
        authority = _derive_team_deathmatch_authority_v1(
            context,
            start_frame,
            cast(TransitionFactsV1, facts),
            successor_frame,
        )
        mode_name, short_name, end_prefix = (
            "Team Deathmatch",
            "TDM",
            "team_deathmatch",
        )
    if context.resolved_env_config.task_mode == 0:
        if any(reward != 0.0 for reward in canonical_reward_by_agent):
            raise ValueError("task-neutral canonical reward must remain zero")
        if terminated:
            raise ValueError("task-neutral transitions cannot terminate")
        if truncated != authority.horizon_reached:
            raise ValueError(
                "task-neutral truncation must equal resolved horizon completion"
            )
        return None, None

    if terminated != authority.threshold_reached:
        raise ValueError(f"{mode_name} terminated must equal threshold completion")
    if truncated != authority.horizon_reached:
        raise ValueError(f"{mode_name} truncated must equal horizon completion")

    reward_by_team_by_outcome = {
        0: (0.0, 0.0),
        1: (1.0, -1.0),
        2: (-1.0, 1.0),
        3: (0.0, 0.0),
    }
    canonical_reward_by_team = reward_by_team_by_outcome[authority.outcome]
    expected_reward_by_agent = tuple(
        (
            canonical_reward_by_team[roster_row.configured_team_id - 1]
            if roster_row.configured_active
            else 0.0
        )
        for roster_row in context.roster
    )
    if canonical_reward_by_agent != expected_reward_by_agent:
        raise ValueError(
            f"per-agent canonical {short_name} reward must match configured team "
            "membership"
        )

    owning_task_end_reason = (
        None
        if authority.completion_basis is None
        else f"{end_prefix}_{authority.completion_basis}"
    )
    return canonical_reward_by_team, owning_task_end_reason


def _validate_context_joined_frame(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
    *,
    record_name: str,
) -> EvaluationFrame:
    """Return a revalidated frame after checking episode, task, and information joins.

    Require its context-owned frame version, canonical ID, inactive padding, and
    policy-visible task fields. Neutral frames keep TDM scores zero. Raise ValueError
    with the supplied record label on a mismatch; do not mutate the original frame.
    """
    frame = cast(
        EvaluationFrame,
        validate_declared_model_tree(
            frame,
            record_name=record_name,
            expected_type=evaluation_frame_type_for_context(context),
        ),
    )
    episode_id = context.identity.episode_id
    if frame.episode_id != episode_id:
        raise ValueError(f"{record_name} must join to the context episode")
    expected_frame_id = f"{episode_id}:frame:{frame.frame_index}"
    if frame.frame_id != expected_frame_id:
        raise ValueError(f"{record_name} ID is not canonical")
    if context.resolved_env_config.task_mode == 0 and recorded_team_scores(
        frame.snapshot
    ) != (0, 0):
        raise ValueError(
            f"{record_name} must keep Team Deathmatch scores zero outside TDM"
        )
    _validate_frame_information_regime(context, frame)
    _validate_inactive_frame_padding(context, frame)
    _validate_task_context_projection(context, frame)
    return frame


def validate_context_joined_evaluation_frame_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
) -> None:
    """Check one captured frame against its episode context.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact supported episode context with resolved configuration and roster.
    frame : EvaluationFrame
        Matching frame type from evaluation_frame_type_for_context: V4 for
        context V5, V3 for context V4, V2 for context V3, otherwise V1.

    Returns
    -------
    None
        None.

    Raises
    ------
    ValueError
        Structural types, IDs, padding, task fields, or information
        availability disagree with the context.

    Notes
    -----
    Any valid frame index is allowed. This host-only validator does not require
    frame zero, a preceding transition, or simulator execution.
    """
    validate_declared_model_tree(
        context,
        record_name="context",
        expected_type=evaluation_context_type(context),
    )
    _validate_context_joined_frame(
        context,
        frame,
        record_name="frame",
    )


def validate_initial_evaluation_frame_v1(
    context: EvaluationEpisodeContext,
    initial_frame: EvaluationFrame,
) -> None:
    """Check frame zero and the episode's remaining capture horizon.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact supported episode context with declared transition count.
    initial_frame : EvaluationFrame
        Matching captured frame at artifact index zero. Its simulator
        tick may be nonzero, as in scenario or resumed-state capture.

    Returns
    -------
    None
        None.

    Raises
    ------
    ValueError
        The context/frame join is invalid, the artifact index is not zero,
        or a Team Deathmatch or Capture the Flag game starts terminal or
        declares a horizon inconsistent with the remaining simulator ticks.

    Notes
    -----
    Artifact indices and simulator ticks are different coordinates. For TDM and
    Capture the Flag, initial simulator tick plus expected_horizon must equal
    maximum_episode_steps, and the frame must precede both score-threshold and
    horizon completion.
    """
    validate_declared_model_tree(
        context,
        record_name="context",
        expected_type=evaluation_context_type(context),
    )
    _validate_context_joined_frame(
        context,
        initial_frame,
        record_name="initial frame",
    )
    if initial_frame.frame_index != 0:
        raise ValueError("initial frame index must be zero")
    expected_frame_id = f"{context.identity.episode_id}:frame:0"
    if initial_frame.frame_id != expected_frame_id:
        raise ValueError("initial frame ID must identify artifact frame zero")
    config = context.resolved_env_config
    mode_name = {1: "Team Deathmatch", 3: "Capture the Flag"}.get(config.task_mode)
    if mode_name is not None:
        threshold = recorded_score_threshold(config)
        if initial_frame.simulator_step_count >= config.maximum_episode_steps or any(
            score >= threshold for score in recorded_team_scores(initial_frame.snapshot)
        ):
            raise ValueError(f"initial {mode_name} frame must be preterminal")
        if (
            initial_frame.simulator_step_count + context.expected_horizon
            != config.maximum_episode_steps
        ):
            raise ValueError(
                f"{mode_name} expected_horizon must equal the remaining "
                "simulator transitions"
            )


def validate_evaluation_transition_unit_v1(
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrame,
    transition: EvaluationTransition,
    successor_frame: EvaluationFrame,
) -> None:
    """Check one complete start/action/result unit from a captured episode.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact supported episode context owning configuration and roster.
    start_frame : EvaluationFrame
        Captured decision frame before the transition.
    transition : EvaluationTransition
        Exact transition of the context's version (V2 for context V5, V1
        otherwise) with submitted/accepted action facts, rewards, completion
        flags, and decoded events.
    successor_frame : EvaluationFrame
        Context-compatible frame exactly one artifact index and one
        simulator tick after start_frame.

    Returns
    -------
    None
        None.

    Raises
    ------
    ValueError
        Any type, episode/ID join, adjacency, inactive fact, task result,
        reward, end reason, or canonical event sequence is inconsistent.

    Notes
    -----
    Initialization facts cannot serve as a transition. Validation reconstructs
    declared models and decodes events from recorded facts on the host. It checks
    record consistency, not whether a fresh simulation would reproduce the facts.
    """
    _canonicalize_evaluation_transition_unit_v1(
        context, start_frame, transition, successor_frame
    )


def _canonicalize_evaluation_transition_unit_v1(
    context: EvaluationEpisodeContext,
    start_frame: EvaluationFrame,
    transition: EvaluationTransition,
    successor_frame: EvaluationFrame,
) -> tuple[
    EvaluationEpisodeContext,
    EvaluationFrame,
    EvaluationTransition,
    EvaluationFrame,
]:
    """Validate a joined transition and return detached context/frame/transition copies.

    The returned order matches the four arguments. Check exact model types, canonical
    IDs, consecutive artifact indices and simulator ticks, real transition facts,
    inactive padding, task-owned rewards/completion, and the entire decoded event
    sequence. Raise ValueError on a mismatch. Internal consumers reuse these copies
    to avoid validating and reconstructing the same unit twice.
    """
    context = cast(
        EvaluationEpisodeContext,
        validate_declared_model_tree(
            context,
            record_name="context",
            expected_type=evaluation_context_type(context),
        ),
    )
    transition = cast(
        EvaluationTransition,
        validate_declared_model_tree(
            transition,
            record_name="transition",
            expected_type=evaluation_transition_type_for_context(context),
        ),
    )
    start_frame = _validate_context_joined_frame(
        context,
        start_frame,
        record_name="start frame",
    )
    successor_frame = _validate_context_joined_frame(
        context,
        successor_frame,
        record_name="successor frame",
    )

    episode_id = context.identity.episode_id
    if transition.episode_id != episode_id:
        raise ValueError("all transition-unit records must join to the context episode")

    expected_start_frame_id = f"{episode_id}:frame:{start_frame.frame_index}"
    expected_successor_index = start_frame.frame_index + 1
    expected_successor_frame_id = f"{episode_id}:frame:{expected_successor_index}"
    expected_transition_id = f"{episode_id}:transition:{start_frame.frame_index}"
    if start_frame.frame_id != expected_start_frame_id:
        raise ValueError("start frame ID is not canonical")
    if successor_frame.frame_index != expected_successor_index:
        raise ValueError("successor frame index must be start frame index plus one")
    if successor_frame.frame_id != expected_successor_frame_id:
        raise ValueError("successor frame ID is not canonical")
    if transition.transition_index != start_frame.frame_index:
        raise ValueError("transition index must equal its start frame index")
    if transition.transition_id != expected_transition_id:
        raise ValueError("transition ID is not canonical")
    if transition.start_frame_id != expected_start_frame_id:
        raise ValueError("transition start-frame reference is not canonical")
    if transition.successor_frame_id != expected_successor_frame_id:
        raise ValueError("transition successor-frame reference is not canonical")

    if not transition.facts.has_transition:
        raise ValueError("initialization facts cannot construct a transition")
    if transition.facts.transition_start_step_count != start_frame.simulator_step_count:
        raise ValueError("facts must name the start frame's simulator step")
    if successor_frame.simulator_step_count != start_frame.simulator_step_count + 1:
        raise ValueError("successor simulator step must be start step plus one")

    _validate_inactive_fact_padding(context, transition)

    expected_team_reward, expected_end_reason = derive_and_validate_task_authority(
        context,
        start_frame,
        transition.facts,
        successor_frame,
        transition.canonical_reward_by_agent,
        transition.terminated,
        transition.truncated,
    )
    if transition.canonical_reward_by_team != expected_team_reward:
        raise ValueError(
            "canonical_reward_by_team must equal task-derived reward authority"
        )
    if transition.owning_task_end_reason != expected_end_reason:
        raise ValueError(
            "owning_task_end_reason must equal task-derived completion authority"
        )

    expected_events = decode_evaluation_events_v1(
        context,
        start_frame,
        transition.facts,
        successor_frame,
    )
    if transition.events != expected_events:
        raise ValueError("transition events must exactly equal canonical fact decoding")
    return context, start_frame, transition, successor_frame


__all__ = [
    "derive_and_validate_task_authority",
    "validate_context_joined_evaluation_frame_v1",
    "validate_evaluation_transition_unit_v1",
    "validate_initial_evaluation_frame_v1",
]


# Frame and transition admission follow the context's versions.
validate_initial_evaluation_frame = validate_initial_evaluation_frame_v1
validate_context_joined_evaluation_frame = validate_context_joined_evaluation_frame_v1
validate_evaluation_transition_unit = validate_evaluation_transition_unit_v1
