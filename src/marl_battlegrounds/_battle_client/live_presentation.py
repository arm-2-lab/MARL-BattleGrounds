"""Package committed live endpoints for the authorized browser presentation API.

The live service calls the Oracle, NoSharedObs, or SharedObs presentation builder
for its selected view. Agent battlefields contain only authorized facts; separate
researcher controls may carry nonspatial match data. Builders return immutable
presentation models. They do not step the simulator, save history, read replay
files, or perform HTTP I/O.

Agent POV scenes also carry the team's merged sight (``team_sight``) for the
Night Veil: the floor the team's living sensors see, with stones blocking
sight as in Core, and the shapes the veil passes through while the incoming
turn plays. rendering.team_sight computes it from the POV scenes' own sensors
and stones (the Replay Viewer uses the same owner). The browser only paints
it.

``advance_live_match_history_v1`` advances the session's Champion Streaks
counts (every view's takedowns without falling) and its Herald Cards history
(every view's unanswered runs and rescues, the earlier lead and the fresh
start; viewer.match_moments) by one transition, building each recipient's
Agent POV visual events at most once; control.submit_joint_action calls it.
``advance_live_champion_takedowns_v1`` advances the streak counts alone. Each
builder's ``champion_streaks`` and ``match_moments`` arguments carry the shown
view's own rows into its frame. Each builder's
``match_honors`` argument carries the shown view's game-over Honors and Match
Summary rows (viewer.match_summary.match_honors_v1, totalled by the live
service) into the completion frame.
"""

from __future__ import annotations

from dataclasses import replace

from marl_battlegrounds._battle_client.model import PendingAction
from marl_battlegrounds._battle_client.protocol import (
    ActorPovHudFrameV1,
    ActorPovLiveDebuggerFrameV2,
    ActorPovPendingActionCardV1,
    ActorPovTargetReferenceV1,
    PendingActionCardV1,
    ResearcherHudFrameV2,
    ResearcherLiveDebuggerFrameV2,
    SharedObsAgentPovLiveDebuggerFrameV2,
    TargetReferenceV1,
)
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    AgentDiedEventV1,
    EvaluationEpisodeContext,
    EvaluationFrame,
    EvaluationFrameV1,
    EvaluationFrameV2,
    EvaluationFrameV3,
    StaticMechanicsCatalogV1,
    evaluation_context_type,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovAdjacentTransitionSlice,
    ActorPovAdjacentTransitionSliceV1,
    ActorPovAdjacentTransitionSliceV2,
    ActorPovAdjacentTransitionSliceV3,
    ActorPovCurrentSlice,
    ActorPovCurrentSliceV1,
    ActorPovCurrentSliceV2,
    ActorPovCurrentSliceV3,
)
from marl_battlegrounds.rendering.authorized_incoming import (
    build_live_no_shared_obs_incoming_summary_v1,
    build_shared_obs_incoming_summary_v1,
)
from marl_battlegrounds.rendering.authorized_inspection import (
    AuthorizedAxisOnlyTargetActionV1,
    AuthorizedNoTargetActionV1,
    AuthorizedVisibleTargetActionV1,
    DraftArmedLaneV1,
    LiveDraftInspectionPresentationV1,
    build_live_no_shared_obs_draft_inspection_v1,
    build_live_oracle_draft_inspection_v1,
    build_live_shared_obs_draft_inspection_v1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    SharedObsAuthorizedScenePartsV1,
    build_no_shared_obs_authorized_scene_v1,
    build_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AcceptedActionTupleV1,
    AgentPovVisualIncomingSummaryV1,
    AuthorizedBattlefieldSceneV1,
    SubmittedActionTupleV1,
    build_agent_pov_visual_incoming_summary_v1,
    build_replay_oracle_presentation_parts_v1,
    oracle_presentation_key_v1,
    visual_batch_home_fall_slots_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    SharedObsSourceMaterialProjection,
    build_shared_obs_authority_source_material_projection_v1,
    build_visual_event_batch_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovAnalyzerProjectionV1,
    build_actor_pov_analyzer_projection_v1,
)
from marl_battlegrounds.rendering.scene import (
    ResearcherAnalyzerProjectionV2,
    VisualEventBatchV2,
)
from marl_battlegrounds.rendering.team_sight import with_team_sight_v1
from marl_battlegrounds.viewer.champion_streaks import (
    NO_TAKEDOWNS,
    ChampionStreakV1,
    ChampionTakedownsV1,
    agent_pov_tick_takedowns_v1,
    oracle_tick_takedowns_v1,
)
from marl_battlegrounds.viewer.match_moments import (
    MatchMomentFactsV1,
    MatchMomentHistoryV1,
    RescueFact,
    agent_pov_tick_rescues_v1,
    oracle_tick_rescues_v1,
)
from marl_battlegrounds.viewer.match_summary import (
    MatchHonorsV1,
    build_match_summary_v1,
)
from marl_battlegrounds.viewer.presentation_protocol import (
    AgentPovActionAxisV1,
    LatestTransitionActionRowV1,
    LiveEditableDraftInspectionV1,
    LiveNoSharedObsAuthorizedPresentationFrameV1,
    LiveNoSharedObsInspectionEnvelopeV1,
    LiveNoSharedObsPresentationSourceIdentityV1,
    LiveNoSharedObsTechnicalFrameV1,
    LiveOracleAuthorizedPresentationFrameV1,
    LiveOracleInspectionEnvelopeV1,
    LiveOraclePresentationSourceIdentityV1,
    LiveOracleTechnicalFrameV1,
    LivePendingActionTupleV1,
    LivePendingJointActionRowV1,
    LivePendingJointActionV1,
    LivePolicyInspectionV1,
    LiveResearcherDraftInspectionV1,
    LiveResearcherEditableDraftInspectionV1,
    LiveResearcherSpaceV1,
    LiveScriptedPlaybackInspectionV1,
    LiveSharedObsAuthorizedPresentationFrameV1,
    LiveSharedObsInspectionEnvelopeV1,
    LiveSharedObsPresentationSourceIdentityV1,
    LiveSharedObsTechnicalFrameV1,
    NoSharedObsLatestTransitionV1,
    NoSharedObsPresentationAuthorityV1,
    OracleLatestTransitionV1,
    OraclePresentationAuthorityV1,
    ReplayResearcherRosterAgentV1,
    SharedObsPresentationAuthorityV1,
    build_no_shared_obs_authorized_current_endpoint_v1,
    build_oracle_authorized_current_endpoint_v1,
    build_shared_obs_authorized_current_endpoint_v1,
    build_shared_obs_latest_transition_v1,
)


def _require_live_header(
    raw_frame: ResearcherLiveDebuggerFrameV2 | ActorPovLiveDebuggerFrameV2,
    *,
    researcher: bool,
) -> None:
    """Require the exact live root, schema, canonical frame ID, and nonnegative
    epoch.
    """
    expected_type = (
        ResearcherLiveDebuggerFrameV2 if researcher else ActorPovLiveDebuggerFrameV2
    )
    expected_kind = (
        "researcher_live_debugger" if researcher else "actor_pov_live_debugger"
    )
    expected_mode = "researcher" if researcher else "pov"
    if type(raw_frame) is not expected_type:
        raise TypeError(f"raw_frame must be the exact {expected_type.__name__} root.")
    if (
        type(raw_frame.schema_version) is not int
        or raw_frame.schema_version != 2
        or type(raw_frame.frame_kind) is not str
        or raw_frame.frame_kind != expected_kind
        or type(raw_frame.view_mode) is not str
        or raw_frame.view_mode != expected_mode
    ):
        raise ValueError("raw live frame does not retain its exact wire identity.")
    for name in (
        "session_id",
        "episode_id",
        "frame_id",
    ):
        value = getattr(raw_frame, name)
        if type(value) is not str or not value:
            raise TypeError(f"raw live {name} must be a nonempty exact string.")
    for name in (
        "run_generation",
        "revision",
        "frame_index",
        "simulator_step_count",
    ):
        value = getattr(raw_frame, name)
        if type(value) is not int or value < 0:
            raise TypeError(f"raw live {name} must be a non-negative exact int.")
    if raw_frame.frame_id != f"{raw_frame.episode_id}:frame:{raw_frame.frame_index}":
        raise ValueError("raw live frame ID is not canonical.")


def _require_oracle_used_containers(raw_frame: ResearcherLiveDebuggerFrameV2) -> None:
    """Check the exact Oracle projection and pending-action model types before use."""
    if type(raw_frame.projection) is not ResearcherAnalyzerProjectionV2:
        raise TypeError("raw Oracle projection must use its exact V2 root.")
    if type(raw_frame.hud) is not ResearcherHudFrameV2:
        raise TypeError("raw Oracle HUD must use its exact V2 root.")
    pending = raw_frame.hud.pending_action
    if pending is None:
        return
    if type(pending) is not PendingActionCardV1:
        raise TypeError("raw Oracle pending action must use its exact V1 root.")
    if type(pending.target) is not TargetReferenceV1:
        raise TypeError("raw Oracle pending target must use its exact V1 root.")


def _require_no_shared_used_containers(
    raw_frame: ActorPovLiveDebuggerFrameV2,
) -> None:
    """Check the exact NoSharedObs projection and pending-action model types before
    use.
    """
    if type(raw_frame.projection) is not ActorPovAnalyzerProjectionV1:
        raise TypeError("raw POV projection must use its exact V1 root.")
    if type(raw_frame.hud) is not ActorPovHudFrameV1:
        raise TypeError("raw POV HUD must use its exact V1 root.")
    pending = raw_frame.hud.pending_action
    if pending is None:
        return
    if type(pending) is not ActorPovPendingActionCardV1:
        raise TypeError("raw POV pending action must use its exact V1 root.")
    if type(pending.target) is not ActorPovTargetReferenceV1:
        raise TypeError("raw POV pending target must use its exact V1 root.")


def _require_shared_live_header(
    raw_frame: SharedObsAgentPovLiveDebuggerFrameV2,
) -> None:
    """Require the exact SharedObs live root, canonical identity, and nonnegative
    epoch.
    """
    if type(raw_frame) is not SharedObsAgentPovLiveDebuggerFrameV2:
        raise TypeError(
            "raw_frame must be the exact SharedObsAgentPovLiveDebuggerFrameV2 root."
        )
    if (
        raw_frame.schema_version != 2
        or raw_frame.frame_kind != "shared_obs_agent_pov_live_debugger"
        or raw_frame.view_mode != "pov"
        or raw_frame.verbose is not False
    ):
        raise ValueError("raw SharedObs frame does not retain its exact wire identity.")
    for name in ("session_id", "episode_id", "frame_id"):
        value = getattr(raw_frame, name)
        if type(value) is not str or not value:
            raise TypeError(f"raw SharedObs {name} must be a nonempty exact string.")
    for name in (
        "run_generation",
        "revision",
        "frame_index",
        "simulator_step_count",
    ):
        value = getattr(raw_frame, name)
        if type(value) is not int or value < 0:
            raise TypeError(f"raw SharedObs {name} must be a nonnegative exact int.")
    if raw_frame.frame_id != f"{raw_frame.episode_id}:frame:{raw_frame.frame_index}":
        raise ValueError("raw SharedObs global frame ID is not canonical.")


def _canonical_live_view(
    context: EvaluationEpisodeContext,
    current_frame: EvaluationFrame,
    incoming_transition_view: EvaluationTransitionViewV1 | None,
) -> EvaluationTransitionViewV1 | None:
    """Revalidate the incoming transition and require it to enter the current endpoint.

    Frame zero has no incoming transition. Every later frame must have one whose
    context and successor match the selected live frame.
    """
    evaluation_context_type(context)
    if (
        type(current_frame) is not EvaluationFrameV1
        and type(current_frame) is not EvaluationFrameV2
        and type(current_frame) is not EvaluationFrameV3
    ):
        raise TypeError("current_frame must be the exact EvaluationFrameV1 root.")
    if current_frame.frame_index == 0:
        if incoming_transition_view is not None:
            raise ValueError("live frame zero cannot carry incoming T_(n-1).")
        return None
    if type(incoming_transition_view) is not EvaluationTransitionViewV1:
        raise TypeError("nonzero live frames require an exact incoming view.")
    view = EvaluationTransitionViewV1(
        context=incoming_transition_view.context,
        start_frame=incoming_transition_view.start_frame,
        transition=incoming_transition_view.transition,
        successor_frame=incoming_transition_view.successor_frame,
    )
    if view.context != context or view.successor_frame != current_frame:
        raise ValueError("incoming live view must enter the selected current frame.")
    return view


def _shared_obs_source_materials(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
    *,
    recipient_global_slot: int,
) -> tuple[
    SharedObsSourceMaterialProjection,
    tuple[SharedObsSourceMaterialProjection, ...],
]:
    """Build active-actor source projections and identify the selected recipient.

    These compact sources feed the shared observation authority; they do not grant
    a recipient permission to read another actor's private data directly.
    """
    sources = tuple(
        build_shared_obs_authority_source_material_projection_v1(
            context,
            frame,
            selected_global_slot=roster.global_slot,
        )
        for roster in context.roster
        if roster.configured_active
    )
    recipient = next(
        (
            source
            for source in sources
            if source.base_sensor_scene.self_actor.global_slot == recipient_global_slot
        ),
        None,
    )
    if recipient is None:
        raise ValueError("SharedObs recipient is absent from active source material.")
    nonrecipient = tuple(source for source in sources if source is not recipient)
    return recipient, nonrecipient


def advance_live_champion_takedowns_v1(
    previous: ChampionTakedownsV1,
    view: EvaluationTransitionViewV1,
) -> ChampionTakedownsV1:
    """Advance every view's Champion Streaks counts by one live transition.

    Parameters
    ----------
    previous : ChampionTakedownsV1
        The counts after the transition's start frame (the session's
        ``champion_takedowns``); frame zero holds ``ChampionTakedownsV1()``.
    view : EvaluationTransitionViewV1
        The coherent live transition just captured (context, start frame,
        transition, successor frame).

    Returns
    -------
    ChampionTakedownsV1
        The counts after the successor frame, for Oracle View and for the
        Agent POV of every configured-active recipient (see
        viewer.champion_streaks for the rule). ``previous`` is unchanged.

    Raises
    ------
    ValueError
        A recorded event or authorized scene fails its existing validation.

    Notes
    -----
    A transition without a recorded fall changes no count, so it returns
    at once (``ChampionTakedownsV1.quiet``) with no scene work. On a tick
    with a fall, Oracle View reads the recorded events and every
    configured-active recipient's Agent POV is built exactly as the live
    Agent POV presentation builds it (start and successor scenes, their
    death-only corpse scenes, then build_agent_pov_visual_incoming_summary_v1),
    so each view counts only what it shows. SharedObs builds each slot's
    source material once per frame for all recipients. Host-only: no JAX,
    file or simulator work. Measured on CPU (2 October 2026): about 8 ms per
    recipient on a tick with a fall.
    """
    events = view.transition.events
    if not any(isinstance(event, AgentDiedEventV1) for event in events):
        return previous.quiet()
    context = view.context
    agent_pov = tuple(
        NO_TAKEDOWNS
        if summary is None
        else agent_pov_tick_takedowns_v1(context, summary)
        for summary in _live_agent_pov_visual_events_by_recipient(view)
    )
    return previous.advance(oracle_tick_takedowns_v1(context, events), agent_pov)


def advance_live_match_history_v1(
    takedowns: ChampionTakedownsV1,
    moments: MatchMomentHistoryV1,
    view: EvaluationTransitionViewV1,
) -> tuple[ChampionTakedownsV1, MatchMomentHistoryV1]:
    """Advance every view's streak counts and Herald Cards history by one transition.

    Parameters
    ----------
    takedowns : ChampionTakedownsV1
        The Champion Streaks counts after the transition's start frame (the
        session's ``champion_takedowns``); frame zero holds
        ``ChampionTakedownsV1()``.
    moments : MatchMomentHistoryV1
        The Herald Cards history after the start frame (the session's
        ``match_moments``); frame zero holds ``MatchMomentHistoryV1()``.
    view : EvaluationTransitionViewV1
        The coherent live transition just captured (context, start frame,
        transition, successor frame).

    Returns
    -------
    takedowns : ChampionTakedownsV1
        The streak counts after the successor frame; equal to
        ``advance_live_champion_takedowns_v1(takedowns, view)``.
    moments : MatchMomentHistoryV1
        The history after the successor frame, for Oracle View and the Agent
        POV of every configured-active recipient (see viewer.match_moments
        for the rules): the fresh start (set from the start frame of the
        first transition), the largest lead at the start of any tick so far,
        each view's unanswered runs and this tick's rescues. Both inputs are
        unchanged.

    Raises
    ------
    ValueError
        A recorded event or authorized scene fails its existing validation.

    Notes
    -----
    Each recipient's Agent POV visual events are built once, and only on a
    tick with a recorded fall or an Oracle rescue (most ticks have
    neither, so they cost no scene work); streaks, runs and rescues all read
    the same build. A tick with a rescue but no fall leaves the streak
    counts unchanged (``ChampionTakedownsV1.quiet``). Host-only: no JAX,
    file or simulator work. The scene work is the same as the streak count
    alone (about 8 ms per recipient on CPU, measured 2 October 2026), now
    also run on the rare ticks with a rescue and no fall.
    """
    events = view.transition.events
    context = view.context
    start = view.start_frame
    fell = any(isinstance(event, AgentDiedEventV1) for event in events)
    oracle_rescues = oracle_tick_rescues_v1(context, events)
    visual = (
        _live_agent_pov_visual_events_by_recipient(view)
        if fell or oracle_rescues
        else (None,) * len(context.roster)
    )
    if fell:
        oracle_tick = oracle_tick_takedowns_v1(context, events)
        pov_ticks = tuple(
            NO_TAKEDOWNS
            if summary is None
            else agent_pov_tick_takedowns_v1(context, summary)
            for summary in visual
        )
        takedowns = takedowns.advance(oracle_tick, pov_ticks)
    else:
        oracle_tick = NO_TAKEDOWNS
        pov_ticks = (NO_TAKEDOWNS,) * len(visual)
        takedowns = takedowns.quiet()
    pov_rescues: tuple[tuple[RescueFact, ...], ...] = tuple(
        ()
        if summary is None
        else agent_pov_tick_rescues_v1(context, oracle_rescues, events, summary)
        for summary in visual
    )
    scores = view.start_frame.snapshot.team_deathmatch_scores
    moments = moments.advance(
        context,
        start_frame_index=start.frame_index,
        start_simulator_step_count=start.simulator_step_count,
        start_scores=(int(scores[0]), int(scores[1])),
        oracle_falls=oracle_tick.falls,
        oracle_rescues=oracle_rescues,
        agent_pov_falls=tuple(tick.falls for tick in pov_ticks),
        agent_pov_rescues=pov_rescues,
    )
    return takedowns, moments


def _live_agent_pov_visual_events_by_recipient(
    view: EvaluationTransitionViewV1,
) -> tuple[AgentPovVisualIncomingSummaryV1 | None, ...]:
    """Build every configured-active recipient's Agent POV visual events.

    view is the coherent live transition. Return ten entries in global slot
    order: the recipient's AgentPovVisualIncomingSummaryV1, built from the
    same scenes the live Agent POV presentation uses (NoSharedObs: the
    recipient's own current slice and carrier; SharedObs: the team's sensor
    union, each slot's source material built once per frame), or None for an
    inactive slot. Corpse scenes use the public class records of the scenes
    built here, which together cover every active class. Raises ValueError
    when a scene or event fails validation. Host-only; no JAX or file work.
    """
    from marl_battlegrounds.viewer.local_oracle_corpse_overlay import (
        build_local_oracle_corpse_overlay_v1,
        compose_local_oracle_corpse_scene_v1,
    )
    from marl_battlegrounds.viewer.no_shared_visual import (
        build_live_no_shared_obs_visual_adjacent_slice_v1,
        build_live_no_shared_obs_visual_current_slice_v1,
    )

    context = view.context
    catalog = context.static_mechanics_catalog
    authority = "champion-streaks"
    roster = context.roster
    active = tuple(row.global_slot for row in roster if row.configured_active)
    frames = (view.start_frame, view.successor_frame)
    # (start scene, start sensors, successor scene, successor sensors, IDs).
    built: dict[
        int,
        tuple[
            AuthorizedBattlefieldSceneV1,
            tuple[str, ...],
            AuthorizedBattlefieldSceneV1,
            tuple[str, ...],
            tuple[str, str, str],
        ],
    ] = {}
    if context.execution_information_mode == "shared_obs":
        sources = tuple(
            {
                slot: build_shared_obs_authority_source_material_projection_v1(
                    context, frame, selected_global_slot=slot
                )
                for slot in active
            }
            for frame in frames
        )
        for recipient in active:
            ends = []
            for frame, by_slot in zip(frames, sources, strict=True):
                parts = build_shared_obs_authorized_scene_v1(
                    by_slot[recipient],
                    all_active_nonrecipient_source_material=tuple(
                        by_slot[slot] for slot in active if slot != recipient
                    ),
                    public_catalog=catalog,
                    authority_session_id=authority,
                )
                living = {
                    row.public_agent_id
                    for row in roster
                    if row.configured_active
                    and frame.snapshot.alive_mask[row.global_slot]
                }
                sensors = tuple(
                    source.source_public_agent_id
                    for source in parts.authorized_sensor_sources
                    if source.source_public_agent_id in living
                )
                ends.append((parts.scene, sensors, parts.source_recipient_frame_id))
            prefix = (
                f"{context.identity.episode_id}:shared-obs-visual-union:"
                f"{roster[recipient].public_agent_id}"
            )
            built[recipient] = (
                ends[0][0],
                ends[0][1],
                ends[1][0],
                ends[1][1],
                (
                    f"{prefix}:transition:{view.start_frame.frame_index}",
                    ends[0][2],
                    ends[1][2],
                ),
            )
    else:
        for recipient in active:
            current_slice = build_live_no_shared_obs_visual_current_slice_v1(
                context,
                view.successor_frame,
                global_slot=recipient,
                incoming_transition_view=view,
            )
            carrier = build_live_no_shared_obs_visual_adjacent_slice_v1(
                view, global_slot=recipient
            )
            successor = build_no_shared_obs_authorized_scene_v1(
                current_slice,
                public_catalog=catalog,
                authority_session_id=authority,
            )
            start = build_no_shared_obs_authorized_scene_v1(
                carrier,
                public_catalog=catalog,
                authority_session_id=authority,
                frame_index=carrier.start_frame.frame_index,
            )
            public_id = roster[recipient].public_agent_id
            built[recipient] = (
                start.scene,
                (public_id,) if view.start_frame.snapshot.alive_mask[recipient] else (),
                successor.scene,
                (public_id,)
                if view.successor_frame.snapshot.alive_mask[recipient]
                else (),
                (
                    carrier.transition.pov_transition_id,
                    carrier.transition.start_pov_frame_id,
                    carrier.transition.successor_pov_frame_id,
                ),
            )
    class_mechanics = tuple(
        {
            row.class_id: row
            for scenes in built.values()
            for scene in (scenes[0], scenes[2])
            for row in scene.class_mechanics
        }.values()
    )
    batch = build_visual_event_batch_v2(view)
    home_falls = visual_batch_home_fall_slots_v1(context, batch)
    summaries: list[AgentPovVisualIncomingSummaryV1 | None] = [None] * len(roster)
    for recipient, (
        start_scene,
        start_sensors,
        end_scene,
        end_sensors,
        (transition_id, start_id, end_id),
    ) in built.items():
        public_id = roster[recipient].public_agent_id
        corpse_scenes = tuple(
            compose_local_oracle_corpse_scene_v1(
                scene,
                build_local_oracle_corpse_overlay_v1(
                    context,
                    frame,
                    scene,
                    authority_session_id=authority,
                    source_authority_epoch=0,
                    recipient_public_agent_id=public_id,
                    living_sensor_public_agent_ids=sensors,
                ),
                researcher_class_mechanics=class_mechanics,
            )
            for frame, scene, sensors in (
                (view.start_frame, start_scene, start_sensors),
                (view.successor_frame, end_scene, end_sensors),
            )
        )
        summaries[recipient] = build_agent_pov_visual_incoming_summary_v1(
            batch,
            transition_start_scene=start_scene,
            successor_scene=end_scene,
            transition_start_corpse_choreography_scene=corpse_scenes[0],
            successor_corpse_choreography_scene=corpse_scenes[1],
            recipient_public_agent_id=public_id,
            incoming_recipient_transition_id=transition_id,
            incoming_start_recipient_frame_id=start_id,
            incoming_successor_recipient_frame_id=end_id,
            home_fall_global_slots=home_falls,
        )
    return tuple(summaries)


def _draft_lane_v1(
    *,
    target_action: int,
    raw_armed_lane: int | None,
) -> DraftArmedLaneV1:
    """Translate the raw UI lane into the presentation's named draft state."""
    if type(target_action) is not int or target_action < 0:
        raise TypeError("draft target action must be a non-negative exact int.")
    if raw_armed_lane is not None and (
        type(raw_armed_lane) is not int or raw_armed_lane not in (0, 1)
    ):
        raise TypeError("raw draft lane must be exact 0, 1, or None.")
    if raw_armed_lane is None or (raw_armed_lane == 0 and target_action == 0):
        return "none"
    return "basic" if raw_armed_lane == 0 else "ultimate"


def _policy_inspection_v1(
    draft: LiveDraftInspectionPresentationV1,
) -> LivePolicyInspectionV1:
    """Keep existing authorized mask facts and discard the temporary action draft.

    The shared draft builders establish scene and mask authority without running
    a method. Their neutral action is only a construction detail and is never
    sent or shown as a policy's next action.
    """
    return LivePolicyInspectionV1(
        inspection_kind="policy_read_only",
        submission_scope="joint_turn",
        current_simulator_step_count=draft.current_simulator_step_count,
        actor_presentation_key=draft.actor_presentation_key,
        actor_public_agent_id=draft.actor_public_agent_id,
        actor_anchor=draft.actor_anchor,
        decision_mask=draft.decision_mask,
    )


def _pending_joint_action_v1(
    context: EvaluationEpisodeContext,
    raw_frame: ResearcherLiveDebuggerFrameV2,
) -> LivePendingJointActionV1 | None:
    """Project configured Manual drafts without slots or geometry.

    Dead configured rows remain. Policy rows are absent because their actions
    do not exist until submit. An autonomous match returns an empty row tuple.
    """
    if raw_frame.hud.pending_submission_scope != "joint_turn":
        return None
    rows: list[LivePendingJointActionRowV1] = []
    for pending in raw_frame.hud.pending_actions:
        roster = context.roster[pending.actor_global_slot]
        if not roster.configured_active:
            raise ValueError("Pending Joint Action cannot include an inactive actor.")
        target_action = pending.target_action
        if type(target_action) is not int:
            raise ValueError("researcher pending target action must be public.")
        armed_lane = _draft_lane_v1(
            target_action=target_action,
            raw_armed_lane=pending.armed_lane,
        )
        if armed_lane == "none":
            submitted_target_action = 0
            submitted_ultimate_action = 0
        else:
            submitted_target_action = target_action
            submitted_ultimate_action = pending.armed_lane
            if type(submitted_ultimate_action) is not int:
                raise ValueError("an armed pending action requires an exact lane.")
        rows.append(
            LivePendingJointActionRowV1(
                actor_presentation_key=oracle_presentation_key_v1(
                    authority_session_id=raw_frame.session_id,
                    public_agent_id=roster.public_agent_id,
                ),
                actor_public_agent_id=roster.public_agent_id,
                pending_action=LivePendingActionTupleV1(
                    move_action=pending.move_action,
                    target_action=submitted_target_action,
                    use_ultimate_action=submitted_ultimate_action,
                ),
            )
        )
    return LivePendingJointActionV1(
        current_simulator_step_count=raw_frame.simulator_step_count,
        action_rows=tuple(rows),
    )


def _oracle_latest_transition_v1(
    context: EvaluationEpisodeContext,
    incoming_view: EvaluationTransitionViewV1 | None,
    *,
    authority_session_id: str,
) -> OracleLatestTransitionV1 | None:
    """Package recorded submitted and accepted joint actions for the incoming
    transition.

    Return None at reset. Actor rows use fixed roster identity and catalog target
    mappings; no action acceptance is recomputed here.
    """
    if incoming_view is None:
        return None
    transition = incoming_view.transition
    acceptance = transition.facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    accepted = acceptance.accepted_joint_action
    catalog = context.static_mechanics_catalog
    rows: list[LatestTransitionActionRowV1] = []
    for actor_slot, roster in enumerate(context.roster):
        if not roster.configured_active:
            continue
        target_slots = catalog.global_recipient_slot_by_actor_and_target_action[
            actor_slot
        ]
        rows.append(
            LatestTransitionActionRowV1(
                actor_presentation_key=oracle_presentation_key_v1(
                    authority_session_id=authority_session_id,
                    public_agent_id=roster.public_agent_id,
                ),
                actor_public_agent_id=roster.public_agent_id,
                target_action_recipient_public_agent_id_by_id=tuple(
                    None
                    if target_slot is None
                    else context.roster[target_slot].public_agent_id
                    for target_slot in target_slots
                ),
                submitted_action=SubmittedActionTupleV1(
                    move_action=submitted.move[actor_slot],
                    target_action=submitted.select_target[actor_slot],
                    use_ultimate_action=submitted.use_ultimate[actor_slot],
                ),
                accepted_action=AcceptedActionTupleV1(
                    move_action=accepted.move[actor_slot],
                    target_action=accepted.select_target[actor_slot],
                    use_ultimate_action=accepted.use_ultimate[actor_slot],
                ),
            )
        )
    facts = transition.facts
    return OracleLatestTransitionV1(
        transition_kind="oracle_incoming_submitted_accepted",
        episode_id=transition.episode_id,
        incoming_transition_index=transition.transition_index,
        incoming_transition_id=transition.transition_id,
        incoming_start_frame_id=transition.start_frame_id,
        incoming_successor_frame_id=transition.successor_frame_id,
        incoming_start_simulator_step_count=facts.transition_start_step_count,
        incoming_successor_simulator_step_count=(
            incoming_view.successor_frame.simulator_step_count
        ),
        action_rows=tuple(rows),
    )


def _no_shared_latest_transition_v1(
    carrier: ActorPovAdjacentTransitionSlice | None,
    *,
    endpoint_action_axis: AgentPovActionAxisV1,
) -> NoSharedObsLatestTransitionV1 | None:
    """Describe the selected actor's submitted and accepted incoming action only.

    The authorized carrier supplies the transition. Target references use the
    endpoint action axis, and reset endpoints return None.
    """
    if carrier is None:
        return None
    action_axis = endpoint_action_axis
    transition = carrier.transition
    start = carrier.start_frame
    successor = carrier.successor_frame
    return NoSharedObsLatestTransitionV1(
        transition_kind="no_shared_obs_incoming_submitted_accepted",
        episode_id=transition.episode_id,
        incoming_transition_index=transition.transition_index,
        incoming_transition_id=transition.pov_transition_id,
        incoming_start_frame_id=transition.start_pov_frame_id,
        incoming_successor_frame_id=transition.successor_pov_frame_id,
        incoming_start_simulator_step_count=start.simulator_step_count,
        incoming_successor_simulator_step_count=successor.simulator_step_count,
        action_rows=(
            LatestTransitionActionRowV1(
                actor_presentation_key=action_axis.owner_presentation_key,
                actor_public_agent_id=action_axis.owner_public_agent_id,
                target_action_recipient_public_agent_id_by_id=(
                    action_axis.target_public_agent_id_by_action
                ),
                submitted_action=SubmittedActionTupleV1(
                    move_action=transition.submitted_action.move,
                    target_action=transition.submitted_action.select_target,
                    use_ultimate_action=transition.submitted_action.use_ultimate,
                ),
                accepted_action=AcceptedActionTupleV1(
                    move_action=transition.accepted_action.move,
                    target_action=transition.accepted_action.select_target,
                    use_ultimate_action=transition.accepted_action.use_ultimate,
                ),
            ),
        ),
        recipient_public_agent_id=action_axis.owner_public_agent_id,
        recipient_presentation_key=action_axis.owner_presentation_key,
    )


def build_live_oracle_authorized_presentation_v1(
    context: EvaluationEpisodeContext,
    current_frame: EvaluationFrame,
    incoming_transition_view: EvaluationTransitionViewV1 | None,
    raw_frame: ResearcherLiveDebuggerFrameV2,
    *,
    scores_ten_ticks_ago: tuple[int, int] | None = None,
    champion_streaks: tuple[ChampionStreakV1, ...] | None = None,
    match_honors: MatchHonorsV1 | None = None,
    match_moments: MatchMomentFactsV1 | None = None,
) -> LiveOracleAuthorizedPresentationFrameV1:
    """Build a global live display with incoming history and the current action
    draft.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact supported episode context declaring roster, rules, and information mode.
    current_frame : EvaluationFrame
        Exact current captured frame, matching the context and raw response identity.
    incoming_transition_view : EvaluationTransitionViewV1 | None
        Checked adjacent start/transition/successor records, or None at frame zero.
    raw_frame : ResearcherLiveDebuggerFrameV2
        Exact mode-specific service response for the committed source revision.
    scores_ten_ticks_ago : tuple of (int, int) or None, optional, keyword-only
        Team A and Team B scores 10 ticks before current_frame, from the live
        session's short history (DebuggerSession.scores_ten_ticks_ago). It
        becomes match_summary.scores_ten_ticks_ago for the scorebug's Momentum
        switch. None (the default) leaves that field None (no recent points).
    champion_streaks : tuple of ChampionStreakV1 or None, optional, keyword-only
        The champions that reached 5, 10, 15 or 20 takedowns (every 5th)
        without falling on this tick in this view, from the live session's counts
        (DebuggerSession.champion_takedowns.streaks, viewer.champion_streaks):
        Oracle View's row, counted from every recorded fall. Copied into the
        frame's ``champion_streaks`` field, which checks that each named
        champion is one this view saw at the start of the tick. None (the
        default) means not counted; browsers then show no streak.
    match_honors : MatchHonorsV1 or None, optional, keyword-only
        This view's game-over Honors and Match Summary rows
        (viewer.match_summary.match_honors_v1), from the live service's
        totals of every transition of the game; give it only on the
        completion frame. Every view's
        honors list every champion: the game is over, so Agent POV has
        nothing left to hide. Copied into the frame's ``match_honors``
        field, which refuses Honors unless the frame's own match summary
        shows a finished game. None (the default) means no Honors.
    match_moments : MatchMomentFactsV1 or None, optional, keyword-only
        This view's Herald Cards history facts (viewer.match_moments), from
        the live session's history (DebuggerSession.match_moments.facts):
        whether the match began fresh, the largest earlier lead, the
        unanswered runs reached on this tick and the rescues. An Agent POV
        view's facts name only what its recipient's team saw. Copied into
        the frame's ``match_moments`` field. None (the default) means not
        counted; browsers then show no First Blood, Comeback, run or
        Rescue card.

    Returns
    -------
    LiveOracleAuthorizedPresentationFrameV1
        Checked immutable presentation for the supplied source frame.

    Raises
    ------
    TypeError
        A required source or nested model has an unsupported exact type.
    ValueError
        Source identity, epoch, action meaning, or display authorization is
        inconsistent, or Honors are given on a frame whose match summary
        does not show a finished game.

    Notes
    -----
    Validate source identity and rebuild incoming visual facts before packaging
    them. Scripted playback exposes no editable pending joint action.
    This performs host projection and model validation, with no simulator step
    or file write.
    """
    _require_live_header(
        raw_frame,
        researcher=True,
    )
    _require_oracle_used_containers(raw_frame)
    incoming_view = _canonical_live_view(
        context,
        current_frame,
        incoming_transition_view,
    )
    if (
        raw_frame.episode_id != context.identity.episode_id
        or raw_frame.episode_id != current_frame.episode_id
        or raw_frame.frame_index != current_frame.frame_index
        or raw_frame.frame_id != current_frame.frame_id
        or raw_frame.simulator_step_count != current_frame.simulator_step_count
    ):
        raise ValueError("raw Oracle frame does not join the current live authority.")
    expected_events = (
        None if incoming_view is None else build_visual_event_batch_v2(incoming_view)
    )
    if raw_frame.projection.incoming_events != expected_events:
        raise ValueError("raw Oracle incoming events diverge from T_(n-1).")
    parts = build_replay_oracle_presentation_parts_v1(
        context,
        raw_frame.projection.scene,
        expected_events,
        authority_session_id=raw_frame.session_id,
        final_frame_index=current_frame.frame_index,
        selected_internal_slot=None,
        outgoing_transition=None,
    )
    controlled_slot = raw_frame.hud.controlled_global_slot
    endpoint = build_oracle_authorized_current_endpoint_v1(
        context=context,
        source_scene=raw_frame.projection.scene,
        authority_session_id=raw_frame.session_id,
        selected_internal_slot=controlled_slot,
    )
    if endpoint.scene != parts.current_scene:
        raise ValueError("Oracle endpoint scene diverged from the incoming packager.")
    submission_scope = raw_frame.hud.pending_submission_scope
    if submission_scope == "joint_turn" and raw_frame.hud.pending_action is None:
        input_inspection = _policy_inspection_v1(
            build_live_oracle_draft_inspection_v1(
                context,
                current_frame,
                endpoint.scene,
                controlled_internal_slot=controlled_slot,
                draft_move_action=0,
                draft_target_internal_slot=None,
                draft_armed_lane="none",
            )
        )
    elif submission_scope == "joint_turn":
        pending = raw_frame.hud.pending_action
        assert pending is not None
        pending_target_action = (
            0 if pending.target.global_slot is None else pending.target_action
        )
        if pending_target_action is None:
            raise ValueError("controlled Oracle draft target action is unavailable.")
        input_inspection = LiveEditableDraftInspectionV1(
            inspection_kind="editable_live_draft",
            submission_scope="joint_turn",
            draft=build_live_oracle_draft_inspection_v1(
                context,
                current_frame,
                endpoint.scene,
                controlled_internal_slot=controlled_slot,
                draft_move_action=pending.move_action,
                draft_target_internal_slot=pending.target.global_slot,
                draft_armed_lane=_draft_lane_v1(
                    target_action=pending_target_action,
                    raw_armed_lane=pending.armed_lane,
                ),
            ),
        )
    elif submission_scope == "scripted_playback":
        input_inspection = LiveScriptedPlaybackInspectionV1(
            inspection_kind="scripted_playback_inspection",
            submission_scope="scripted_playback",
            editable_draft_available=False,
            advance_semantics="registered_script_frame",
        )
    else:
        raise ValueError("Oracle live frame has an invalid submission scope.")
    latest_transition = _oracle_latest_transition_v1(
        context,
        incoming_view,
        authority_session_id=raw_frame.session_id,
    )
    source = LiveOraclePresentationSourceIdentityV1(
        source_kind="live_oracle_frame",
        source_session_id=raw_frame.session_id,
        source_run_generation=raw_frame.run_generation,
        source_revision=raw_frame.revision,
        source_authority_epoch=raw_frame.revision,
        episode_id=raw_frame.episode_id,
        source_frame_index=raw_frame.frame_index,
        source_frame_id=raw_frame.frame_id,
        source_simulator_step_count=raw_frame.simulator_step_count,
        source_submission_scope=submission_scope,
        source_authorized_endpoint_digest_sha256=(
            endpoint.authorized_endpoint_digest_sha256
        ),
    )
    return LiveOracleAuthorizedPresentationFrameV1(
        run_state=raw_frame.run_state,
        fresh_launch_required=raw_frame.fresh_launch_required,
        match_summary=build_match_summary_v1(
            context,
            current_frame,
            () if incoming_view is None else incoming_view.transition.events,
            transition_start_frame=(
                None if incoming_view is None else incoming_view.start_frame
            ),
            scores_ten_ticks_ago=scores_ten_ticks_ago,
        ),
        champion_streaks=champion_streaks,
        match_honors=match_honors,
        match_moments=match_moments,
        schema_version=1,
        presentation_kind="live_oracle",
        product_kind="combat_debugger",
        source=source,
        authority=OraclePresentationAuthorityV1(
            authority_kind="oracle",
            projection_basis="global_evaluation_projection",
        ),
        analysis_mode="analysis",
        current_endpoint=endpoint,
        latest_events=parts.incoming_summary,
        latest_transition=latest_transition,
        pending_joint_action=_pending_joint_action_v1(context, raw_frame),
        technical_frame=LiveOracleTechnicalFrameV1(
            technical_kind="live_oracle_technical_frame",
            episode_id=source.episode_id,
            evaluation_frame_index=source.source_frame_index,
            simulator_step_count=source.source_simulator_step_count,
            incoming_transition_id=(
                None
                if latest_transition is None
                else latest_transition.incoming_transition_id
            ),
        ),
        live_inspection=LiveOracleInspectionEnvelopeV1(
            envelope_kind="live_oracle_source_bound_inspection",
            source_session_id=source.source_session_id,
            source_run_generation=source.source_run_generation,
            source_revision=source.source_revision,
            source_authority_epoch=source.source_authority_epoch,
            episode_id=source.episode_id,
            source_frame_index=source.source_frame_index,
            source_frame_id=source.source_frame_id,
            source_simulator_step_count=source.source_simulator_step_count,
            inspection=input_inspection,
        ),
    )


def build_live_researcher_space_v1(
    oracle: LiveOracleAuthorizedPresentationFrameV1,
) -> LiveResearcherSpaceV1:
    """Extract global researcher panels from a validated live Oracle display.

    Parameters
    ----------
    oracle : LiveOracleAuthorizedPresentationFrameV1
        Exact validated global live presentation from the same service revision.

    Returns
    -------
    LiveResearcherSpaceV1
        Checked immutable presentation for the supplied source frame.

    Raises
    ------
    TypeError
        A required source or nested model has an unsupported exact type.
    ValueError
        Source identity, epoch, action meaning, or display authorization is
        inconsistent.

    Notes
    -----
    Target rows keep action meaning while removing battlefield anchors. This
    prevents the panels from becoming a second global scene in an actor view.
    This performs host projection and model validation, with no simulator step
    or file write.
    """
    if type(oracle) is not LiveOracleAuthorizedPresentationFrameV1:
        raise TypeError("oracle must be an exact live Oracle presentation.")
    endpoint = oracle.current_endpoint
    action_axis = endpoint.action_axis
    if action_axis is None:
        raise ValueError("live researcher controls require a selected Oracle actor.")
    source = oracle.source
    directory_by_id = {
        row.public_agent_id: row for row in endpoint.identity_directory.identities
    }
    inspection = oracle.live_inspection.inspection
    if isinstance(inspection, (LiveEditableDraftInspectionV1, LivePolicyInspectionV1)):
        source_draft = (
            inspection
            if isinstance(inspection, LivePolicyInspectionV1)
            else inspection.draft
        )
        no_target = source_draft.decision_mask.target_actions[0]
        if type(no_target) is not AuthorizedNoTargetActionV1:
            raise ValueError("Oracle draft target zero changed its exact variant.")
        positive_targets: list[AuthorizedAxisOnlyTargetActionV1] = []
        for row in source_draft.decision_mask.target_actions[1:]:
            if (
                type(row) is AuthorizedVisibleTargetActionV1
                or type(row) is AuthorizedAxisOnlyTargetActionV1
            ):
                target_public_agent_id = row.target_public_agent_id
            else:
                raise ValueError("Oracle draft positive target changed its variant.")
            positive_targets.append(
                AuthorizedAxisOnlyTargetActionV1(
                    target_kind="axis_only_authorized_agent",
                    target_action=row.target_action,
                    display_name=row.display_name,
                    target_public_agent_id=target_public_agent_id,
                )
            )
        geometry_free_targets = (no_target, *positive_targets)
        geometry_free_mask = replace(
            source_draft.decision_mask,
            target_actions=geometry_free_targets,
        )
        if type(inspection) is LivePolicyInspectionV1:
            pending_inspection = LivePolicyInspectionV1(
                inspection_kind="policy_read_only",
                submission_scope="joint_turn",
                current_simulator_step_count=inspection.current_simulator_step_count,
                actor_presentation_key=inspection.actor_presentation_key,
                actor_public_agent_id=inspection.actor_public_agent_id,
                actor_anchor=None,
                decision_mask=geometry_free_mask,
            )
        else:
            assert isinstance(source_draft, LiveDraftInspectionPresentationV1)
            pending_inspection = LiveResearcherEditableDraftInspectionV1(
                inspection_kind="editable_live_draft",
                submission_scope="joint_turn",
                draft=LiveResearcherDraftInspectionV1(
                    schema_version=source_draft.schema_version,
                    inspection_kind="live_draft_action",
                    current_simulator_step_count=(
                        source_draft.current_simulator_step_count
                    ),
                    actor_presentation_key=source_draft.actor_presentation_key,
                    actor_public_agent_id=source_draft.actor_public_agent_id,
                    decision_mask=geometry_free_mask,
                    draft_action=source_draft.draft_action,
                    draft_target=geometry_free_targets[
                        source_draft.draft_action.target_action
                    ],
                    draft_legality=source_draft.draft_legality,
                ),
            )
    elif type(inspection) is LiveScriptedPlaybackInspectionV1:
        pending_inspection = inspection
    else:  # pragma: no cover - the Oracle root owns this exact union.
        raise AssertionError("validated Oracle inspection variant disappeared.")

    return LiveResearcherSpaceV1(
        researcher_space_kind="global_live_researcher_space",
        source_session_id=source.source_session_id,
        source_run_generation=source.source_run_generation,
        source_revision=source.source_revision,
        source_authority_epoch=source.source_authority_epoch,
        episode_id=source.episode_id,
        frame_index=source.source_frame_index,
        simulator_step_count=source.source_simulator_step_count,
        selected_public_agent_id=action_axis.owner_public_agent_id,
        identity_directory=endpoint.identity_directory,
        roster_agents=tuple(
            ReplayResearcherRosterAgentV1(
                presentation_key=agent.presentation_key,
                public_agent_id=agent.public_agent_id,
                team_id=agent.team_id,
                team_local_slot=directory_by_id[agent.public_agent_id].team_local_slot,
                class_id=agent.class_id,
                class_name=agent.class_name,
                life_state=agent.life_state,
                current_health=agent.current_health,
                maximum_health=agent.maximum_health,
                effective_movement_speed=agent.effective_movement_speed,
                ultimate_cooldown_remaining=agent.ultimate_cooldown_remaining,
                spawn_shield_remaining=agent.spawn_shield_remaining,
                steps_until_out_of_combat=agent.steps_until_out_of_combat,
                out_of_combat_delay_steps=agent.out_of_combat_delay_steps,
                statuses=agent.statuses,
                aura_modifiers=agent.aura_modifiers,
            )
            for agent in endpoint.scene.agents
        ),
        class_mechanics=endpoint.scene.class_mechanics,
        latest_transition=oracle.latest_transition,
        pending_joint_action=oracle.pending_joint_action,
        technical_frame=oracle.technical_frame,
        pending_inspection=pending_inspection,
    )


def build_live_no_shared_obs_authorized_presentation_v1(
    current_slice: ActorPovCurrentSlice,
    incoming_carrier: ActorPovAdjacentTransitionSlice | None,
    raw_frame: ActorPovLiveDebuggerFrameV2,
    *,
    global_context: EvaluationEpisodeContext,
    current_global_frame: EvaluationFrame,
    previous_global_frame: EvaluationFrame | None,
    public_catalog: StaticMechanicsCatalogV1,
    incoming_visual_events: VisualEventBatchV2 | None,
    researcher_space: LiveResearcherSpaceV1,
    scores_ten_ticks_ago: tuple[int, int] | None = None,
    champion_streaks: tuple[ChampionStreakV1, ...] | None = None,
    match_honors: MatchHonorsV1 | None = None,
    match_moments: MatchMomentFactsV1 | None = None,
) -> LiveNoSharedObsAuthorizedPresentationFrameV1:
    """Build a live own-observation display and separately scoped researcher panels.

    Parameters
    ----------
    current_slice : ActorPovCurrentSlice
        Exact current actor slice with its own observation, masks, and recorded axis.
    incoming_carrier : ActorPovAdjacentTransitionSlice | None
        Exact adjacent actor transition slice entering current_slice, or None at frame
        zero.
    raw_frame : ActorPovLiveDebuggerFrameV2
        Exact mode-specific service response for the committed source revision.
    global_context : EvaluationEpisodeContext
        Trusted global episode context used only by the authorized host projection.
    current_global_frame : EvaluationFrame
        Trusted global frame at the selected cursor, used to validate local display
        facts.
    previous_global_frame : EvaluationFrame | None
        Trusted immediately preceding frame, or None at frame zero.
    public_catalog : StaticMechanicsCatalogV1
        Exact recorded public class-mechanics catalog for these source frames.
    incoming_visual_events : VisualEventBatchV2 | None
        Visual events for the immediately preceding transition, or None at frame zero.
    researcher_space : LiveResearcherSpaceV1
        Separately authorized global panels for this same frame, without a global
        battlefield scene.
    scores_ten_ticks_ago : tuple of (int, int) or None, optional
        Team A and Team B scores 10 ticks before current_global_frame, from the
        live session's short history (DebuggerSession.scores_ten_ticks_ago).
        It becomes match_summary.scores_ten_ticks_ago for the scorebug's
        Momentum switch; scores are public, so Agent POV gets the same value
        as Oracle View. None (the default) leaves that field None.
    champion_streaks : tuple of ChampionStreakV1 or None, optional
        The champions that reached 5, 10, 15 or 20 takedowns (every 5th)
        without falling on this tick in this view, from the live session's counts
        (DebuggerSession.champion_takedowns.streaks, viewer.champion_streaks):
        this recipient's Agent POV row, counted only from what it saw. Copied
        into the frame's ``champion_streaks`` field, which checks that each
        named champion is one this view saw at the start of the tick. None (the
        default) means not counted; browsers then show no streak.
    match_honors : MatchHonorsV1 or None, optional, keyword-only
        This view's game-over Honors and Match Summary rows
        (viewer.match_summary.match_honors_v1), from the live service's
        totals of every transition of the game; give it only on the
        completion frame. Every view's
        honors list every champion: the game is over, so Agent POV has
        nothing left to hide. Copied into the frame's ``match_honors``
        field, which refuses Honors unless the frame's own match summary
        shows a finished game. None (the default) means no Honors.
    match_moments : MatchMomentFactsV1 or None, optional, keyword-only
        This view's Herald Cards history facts (viewer.match_moments), from
        the live session's history (DebuggerSession.match_moments.facts):
        whether the match began fresh, the largest earlier lead, the
        unanswered runs reached on this tick and the rescues. An Agent POV
        view's facts name only what its recipient's team saw. Copied into
        the frame's ``match_moments`` field. None (the default) means not
        counted; browsers then show no First Blood, Comeback, run or
        Rescue card.

    Returns
    -------
    LiveNoSharedObsAuthorizedPresentationFrameV1
        Checked immutable presentation for the supplied source frame.

    Raises
    ------
    TypeError
        A required source or nested model has an unsupported exact type.
    ValueError
        Source identity, epoch, action meaning, or display authorization is
        inconsistent, or Honors are given on a frame whose match summary
        does not show a finished game.

    Notes
    -----
    Recheck the actor slice and incoming carrier. Trusted global state is used
    only for authorized host display work, including paint-only corpse facts.
    This performs host projection and model validation, with no simulator step
    or file write.
    """
    from marl_battlegrounds.viewer.local_oracle_corpse_overlay import (
        build_local_oracle_corpse_overlay_v1,
        compose_local_oracle_corpse_scene_v1,
        validate_local_oracle_corpse_overlay_against_source_v1,
    )

    _require_live_header(
        raw_frame,
        researcher=False,
    )
    _require_no_shared_used_containers(raw_frame)
    if (
        type(current_slice) is not ActorPovCurrentSliceV1
        and type(current_slice) is not ActorPovCurrentSliceV2
        and type(current_slice) is not ActorPovCurrentSliceV3
    ):
        raise TypeError("current_slice must be the exact ActorPovCurrentSliceV1 root.")
    if type(public_catalog) is not StaticMechanicsCatalogV1:
        raise TypeError("public_catalog must be the exact V1 catalog root.")
    current_slice = type(current_slice).model_validate(
        current_slice.model_dump(mode="python")
    )
    expected_projection = build_actor_pov_analyzer_projection_v1(current_slice)
    if raw_frame.projection != expected_projection:
        raise ValueError("raw POV projection does not join the current slice.")
    if (
        raw_frame.episode_id != current_slice.episode_id
        or raw_frame.frame_index != current_slice.frame.frame_index
        or raw_frame.frame_id != current_slice.frame.source_frame_id
        or raw_frame.simulator_step_count != current_slice.frame.simulator_step_count
        or raw_frame.hud.controlled_public_agent_id != current_slice.public_agent_id
        or raw_frame.incoming_pov_transition_id
        != expected_projection.incoming_transition_id
    ):
        raise ValueError("raw POV frame does not join the current live recipient.")
    if current_slice.frame.frame_index == 0:
        if incoming_carrier is not None:
            raise ValueError("live NoSharedObs frame zero cannot carry a carrier.")
        carrier = None
    else:
        if (
            type(incoming_carrier) is not ActorPovAdjacentTransitionSliceV1
            and type(incoming_carrier) is not ActorPovAdjacentTransitionSliceV2
            and type(incoming_carrier) is not ActorPovAdjacentTransitionSliceV3
        ):
            raise TypeError("nonzero live NoSharedObs frames require an exact carrier.")
        carrier = type(incoming_carrier).model_validate(
            incoming_carrier.model_dump(mode="python")
        )
        if (
            carrier.episode_id != current_slice.episode_id
            or carrier.public_agent_id != current_slice.public_agent_id
            or carrier.selected_global_slot != current_slice.selected_global_slot
            or carrier.selected_team_local_slot
            != current_slice.selected_team_local_slot
            or carrier.configured_team_id != current_slice.configured_team_id
            or carrier.class_id != current_slice.class_id
            or carrier.axis_mapping != current_slice.axis_mapping
            or carrier.successor_frame != current_slice.frame
            or carrier.transition != current_slice.incoming_transition
        ):
            raise ValueError(
                "live NoSharedObs carrier does not enter the current slice."
            )
    parts = build_no_shared_obs_authorized_scene_v1(
        current_slice,
        public_catalog=public_catalog,
        authority_session_id=raw_frame.session_id,
    )
    # The recipient's scene at the start of the incoming turn (None on frame
    # zero): the Night Veil's turn shapes and the visual events both read it.
    previous = (
        None
        if carrier is None
        else build_no_shared_obs_authorized_scene_v1(
            carrier,
            public_catalog=public_catalog,
            authority_session_id=raw_frame.session_id,
            frame_index=carrier.start_frame.frame_index,
        )
    )
    # The Night Veil's merged sight: under NoSharedObs the recipient alone.
    parts = replace(
        parts,
        scene=with_team_sight_v1(
            parts.scene,
            sensor_public_agent_ids=(parts.recipient_public_agent_id,),
            own_public_agent_id=parts.recipient_public_agent_id,
            turn_start_scene=None if previous is None else previous.scene,
            turn_start_sensor_public_agent_ids=(parts.recipient_public_agent_id,),
        ),
    )
    endpoint = build_no_shared_obs_authorized_current_endpoint_v1(
        parts=parts,
        axis_mapping=current_slice.axis_mapping,
    )
    current_sensor_ids = (
        (current_slice.public_agent_id,)
        if current_global_frame.snapshot.alive_mask[current_slice.selected_global_slot]
        else ()
    )
    corpse_overlay = build_local_oracle_corpse_overlay_v1(
        global_context,
        current_global_frame,
        parts.scene,
        authority_session_id=raw_frame.session_id,
        source_authority_epoch=raw_frame.revision,
        recipient_public_agent_id=current_slice.public_agent_id,
        living_sensor_public_agent_ids=current_sensor_ids,
    )
    validate_local_oracle_corpse_overlay_against_source_v1(
        corpse_overlay,
        global_context,
        current_global_frame,
        parts.scene,
        authority_session_id=raw_frame.session_id,
        source_authority_epoch=raw_frame.revision,
        recipient_public_agent_id=current_slice.public_agent_id,
        living_sensor_public_agent_ids=current_sensor_ids,
    )
    if carrier is None:
        if incoming_visual_events is not None:
            raise ValueError("live NoSharedObs frame zero cannot carry visual events.")
        visual_events = None
    else:
        if type(incoming_visual_events) is not VisualEventBatchV2:
            raise TypeError(
                "non-initial live NoSharedObs frames require exact visual events."
            )
        if previous is None:  # pragma: no cover - built above with the carrier.
            raise AssertionError("live NoSharedObs start scene disappeared")
        if (
            type(previous_global_frame) is not EvaluationFrameV1
            and type(previous_global_frame) is not EvaluationFrameV2
            and type(previous_global_frame) is not EvaluationFrameV3
        ):
            raise TypeError(
                "non-initial live NoSharedObs frames require the exact prior "
                "global frame."
            )
        previous_sensor_ids = (
            (current_slice.public_agent_id,)
            if previous_global_frame.snapshot.alive_mask[
                current_slice.selected_global_slot
            ]
            else ()
        )
        previous_overlay = build_local_oracle_corpse_overlay_v1(
            global_context,
            previous_global_frame,
            previous.scene,
            authority_session_id=raw_frame.session_id,
            source_authority_epoch=raw_frame.revision,
            recipient_public_agent_id=current_slice.public_agent_id,
            living_sensor_public_agent_ids=previous_sensor_ids,
        )
        visual_events = build_agent_pov_visual_incoming_summary_v1(
            incoming_visual_events,
            transition_start_scene=previous.scene,
            successor_scene=parts.scene,
            transition_start_corpse_choreography_scene=(
                compose_local_oracle_corpse_scene_v1(
                    previous.scene,
                    previous_overlay,
                    researcher_class_mechanics=researcher_space.class_mechanics,
                )
            ),
            successor_corpse_choreography_scene=(
                compose_local_oracle_corpse_scene_v1(
                    parts.scene,
                    corpse_overlay,
                    researcher_class_mechanics=researcher_space.class_mechanics,
                )
            ),
            recipient_public_agent_id=parts.recipient_public_agent_id,
            incoming_recipient_transition_id=carrier.transition.pov_transition_id,
            incoming_start_recipient_frame_id=carrier.transition.start_pov_frame_id,
            incoming_successor_recipient_frame_id=(
                carrier.transition.successor_pov_frame_id
            ),
            home_fall_global_slots=visual_batch_home_fall_slots_v1(
                global_context, incoming_visual_events
            ),
        )
    latest_events = (
        None
        if carrier is None
        else build_live_no_shared_obs_incoming_summary_v1(
            carrier,
            public_catalog=public_catalog,
            authority_session_id=raw_frame.session_id,
        )
    )
    latest_transition = _no_shared_latest_transition_v1(
        carrier,
        endpoint_action_axis=endpoint.action_axis,
    )
    submission_scope = raw_frame.hud.pending_submission_scope
    if submission_scope == "joint_turn" and raw_frame.hud.pending_action is None:
        input_inspection = _policy_inspection_v1(
            build_live_no_shared_obs_draft_inspection_v1(
                current_slice,
                parts,
                draft_move_action=0,
                draft_target_action=0,
                draft_armed_lane="none",
            )
        )
    elif submission_scope == "joint_turn":
        pending = raw_frame.hud.pending_action
        assert pending is not None
        pending_target_action = pending.target.target_action
        input_inspection = LiveEditableDraftInspectionV1(
            inspection_kind="editable_live_draft",
            submission_scope="joint_turn",
            draft=build_live_no_shared_obs_draft_inspection_v1(
                current_slice,
                parts,
                draft_move_action=pending.move_action,
                draft_target_action=pending_target_action,
                draft_armed_lane=_draft_lane_v1(
                    target_action=pending_target_action,
                    raw_armed_lane=pending.armed_lane,
                ),
            ),
        )
    elif submission_scope == "scripted_playback":
        input_inspection = LiveScriptedPlaybackInspectionV1(
            inspection_kind="scripted_playback_inspection",
            submission_scope="scripted_playback",
            editable_draft_available=False,
            advance_semantics="registered_script_frame",
        )
    else:
        raise ValueError("NoSharedObs live frame has an invalid submission scope.")
    source = LiveNoSharedObsPresentationSourceIdentityV1(
        source_kind="live_no_shared_obs_frame",
        source_session_id=raw_frame.session_id,
        source_run_generation=raw_frame.run_generation,
        source_revision=raw_frame.revision,
        source_authority_epoch=raw_frame.revision,
        episode_id=current_slice.episode_id,
        source_frame_index=current_slice.frame.frame_index,
        source_recipient_public_agent_id=current_slice.public_agent_id,
        source_recipient_frame_id=parts.source_recipient_frame_id,
        source_simulator_step_count=current_slice.frame.simulator_step_count,
        source_submission_scope=submission_scope,
        source_authorized_endpoint_digest_sha256=(
            endpoint.authorized_endpoint_digest_sha256
        ),
    )
    return LiveNoSharedObsAuthorizedPresentationFrameV1(
        run_state=raw_frame.run_state,
        fresh_launch_required=raw_frame.fresh_launch_required,
        match_summary=build_match_summary_v1(
            global_context,
            current_global_frame,
            () if incoming_visual_events is None else incoming_visual_events.events,
            transition_start_frame=previous_global_frame,
            scores_ten_ticks_ago=scores_ten_ticks_ago,
        ),
        champion_streaks=champion_streaks,
        match_honors=match_honors,
        match_moments=match_moments,
        schema_version=1,
        presentation_kind="live_no_shared_obs_agent_pov",
        product_kind="combat_debugger",
        source=source,
        authority=NoSharedObsPresentationAuthorityV1(
            authority_kind="agent_pov",
            observation_mode="no_shared_obs",
            recipient_public_agent_id=parts.recipient_public_agent_id,
            recipient_presentation_key=parts.recipient_presentation_key,
            projection_basis="recipient_own_recorded_observation",
            exact_actor_input_export_available=True,
        ),
        analysis_mode="analysis",
        current_endpoint=endpoint,
        local_oracle_corpse_overlay=corpse_overlay,
        latest_events=latest_events,
        visual_events=visual_events,
        latest_transition=latest_transition,
        technical_frame=LiveNoSharedObsTechnicalFrameV1(
            technical_kind="live_no_shared_obs_technical_frame",
            episode_id=source.episode_id,
            recipient_frame_index=source.source_frame_index,
            simulator_step_count=source.source_simulator_step_count,
            incoming_recipient_transition_id=(
                None
                if latest_transition is None
                else latest_transition.incoming_transition_id
            ),
        ),
        live_inspection=LiveNoSharedObsInspectionEnvelopeV1(
            envelope_kind="live_no_shared_obs_source_bound_inspection",
            source_session_id=source.source_session_id,
            source_run_generation=source.source_run_generation,
            source_revision=source.source_revision,
            source_authority_epoch=source.source_authority_epoch,
            episode_id=source.episode_id,
            source_frame_index=source.source_frame_index,
            source_recipient_public_agent_id=(source.source_recipient_public_agent_id),
            source_recipient_frame_id=source.source_recipient_frame_id,
            source_simulator_step_count=source.source_simulator_step_count,
            inspection=input_inspection,
        ),
        researcher_space=researcher_space,
    )


def build_live_shared_obs_authorized_presentation_v1(
    context: EvaluationEpisodeContext,
    current_frame: EvaluationFrame,
    incoming_transition_view: EvaluationTransitionViewV1 | None,
    raw_frame: SharedObsAgentPovLiveDebuggerFrameV2,
    *,
    authorized_recipient_global_slot: int,
    pending_action: PendingAction,
    researcher_space: LiveResearcherSpaceV1,
    scores_ten_ticks_ago: tuple[int, int] | None = None,
    champion_streaks: tuple[ChampionStreakV1, ...] | None = None,
    match_honors: MatchHonorsV1 | None = None,
    match_moments: MatchMomentFactsV1 | None = None,
) -> LiveSharedObsAuthorizedPresentationFrameV1:
    """Build a live SharedObs scene and the recipient's current action inspection.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Exact supported episode context declaring roster, rules, and information mode.
    current_frame : EvaluationFrame
        Exact current captured frame, matching the context and raw response identity.
    incoming_transition_view : EvaluationTransitionViewV1 | None
        Checked adjacent start/transition/successor records, or None at frame zero.
    raw_frame : SharedObsAgentPovLiveDebuggerFrameV2
        Exact mode-specific service response for the committed source revision.
    authorized_recipient_global_slot : int
        Configured-active recipient slot supplied by the trusted service, from 0 through
        9.
    pending_action : PendingAction
        Stored recipient draft. Only Manual inspection uses its action values;
        policy inspection shows the current mask without a future action.
    researcher_space : LiveResearcherSpaceV1
        Separately authorized global panels for this same frame, without a global
        battlefield scene.
    scores_ten_ticks_ago : tuple of (int, int) or None, optional
        Team A and Team B scores 10 ticks before current_frame, from the live
        session's short history (DebuggerSession.scores_ten_ticks_ago). It
        becomes match_summary.scores_ten_ticks_ago for the scorebug's Momentum
        switch; scores are public, so Agent POV gets the same value as Oracle
        View. None (the default) leaves that field None.
    champion_streaks : tuple of ChampionStreakV1 or None, optional
        The champions that reached 5, 10, 15 or 20 takedowns (every 5th)
        without falling on this tick in this view, from the live session's counts
        (DebuggerSession.champion_takedowns.streaks, viewer.champion_streaks):
        this recipient's Agent POV row, counted only from what its team saw.
        Copied into the frame's ``champion_streaks`` field, which checks that
        each named champion is one this view saw at the start of the tick. None
        (the default) means not counted; browsers then show no streak.
    match_honors : MatchHonorsV1 or None, optional, keyword-only
        This view's game-over Honors and Match Summary rows
        (viewer.match_summary.match_honors_v1), from the live service's
        totals of every transition of the game; give it only on the
        completion frame. Every view's
        honors list every champion: the game is over, so Agent POV has
        nothing left to hide. Copied into the frame's ``match_honors``
        field, which refuses Honors unless the frame's own match summary
        shows a finished game. None (the default) means no Honors.
    match_moments : MatchMomentFactsV1 or None, optional, keyword-only
        This view's Herald Cards history facts (viewer.match_moments), from
        the live session's history (DebuggerSession.match_moments.facts):
        whether the match began fresh, the largest earlier lead, the
        unanswered runs reached on this tick and the rescues. An Agent POV
        view's facts name only what its recipient's team saw. Copied into
        the frame's ``match_moments`` field. None (the default) means not
        counted; browsers then show no First Blood, Comeback, run or
        Rescue card.

    Returns
    -------
    LiveSharedObsAuthorizedPresentationFrameV1
        Checked immutable presentation for the supplied source frame.

    Raises
    ------
    TypeError
        A required source or nested model has an unsupported exact type.
    ValueError
        Source identity, epoch, action meaning, or display authorization is
        inconsistent, or Honors are given on a frame whose match summary
        does not show a finished game.

    Notes
    -----
    Derive current and preceding sensor source material from the trusted live
    frames. Display authorization does not replace the policy information contract.
    This performs host projection and model validation, with no simulator step
    or file write.
    """
    from marl_battlegrounds.viewer.local_oracle_corpse_overlay import (
        build_local_oracle_corpse_overlay_v1,
        compose_local_oracle_corpse_scene_v1,
        validate_local_oracle_corpse_overlay_against_source_v1,
    )

    _require_shared_live_header(raw_frame)
    view = _canonical_live_view(context, current_frame, incoming_transition_view)
    if context.execution_information_mode != "shared_obs":
        raise ValueError("live SharedObs presentation requires SharedObs context.")
    if type(authorized_recipient_global_slot) is not int or not (
        0 <= authorized_recipient_global_slot < len(context.roster)
    ):
        raise ValueError("SharedObs recipient slot must be an exact roster index.")
    recipient_roster = context.roster[authorized_recipient_global_slot]
    if not recipient_roster.configured_active:
        raise ValueError("SharedObs recipient must be configured active.")
    if type(pending_action) is not PendingAction:
        raise TypeError("pending_action must be the exact PendingAction root.")

    current_recipient, current_nonrecipient = _shared_obs_source_materials(
        context,
        current_frame,
        recipient_global_slot=authorized_recipient_global_slot,
    )
    current = build_shared_obs_authorized_scene_v1(
        current_recipient,
        all_active_nonrecipient_source_material=current_nonrecipient,
        public_catalog=context.static_mechanics_catalog,
        authority_session_id=raw_frame.session_id,
    )
    # The recipient's SharedObs scene at the start of the incoming turn (None
    # on frame zero): the Night Veil's turn shapes and the visual events both
    # read it.
    previous: SharedObsAuthorizedScenePartsV1 | None = None
    if view is not None:
        previous_recipient, previous_nonrecipient = _shared_obs_source_materials(
            context,
            view.start_frame,
            recipient_global_slot=authorized_recipient_global_slot,
        )
        previous = build_shared_obs_authorized_scene_v1(
            previous_recipient,
            all_active_nonrecipient_source_material=previous_nonrecipient,
            public_catalog=context.static_mechanics_catalog,
            authority_session_id=raw_frame.session_id,
        )
    # The Night Veil's merged team sight, from the team's own sensors.
    current = replace(
        current,
        scene=with_team_sight_v1(
            current.scene,
            sensor_public_agent_ids=tuple(
                source.source_public_agent_id
                for source in current.authorized_sensor_sources
            ),
            own_public_agent_id=current.recipient_public_agent_id,
            turn_start_scene=None if previous is None else previous.scene,
            turn_start_sensor_public_agent_ids=(
                ()
                if previous is None
                else tuple(
                    source.source_public_agent_id
                    for source in previous.authorized_sensor_sources
                )
            ),
        ),
    )
    endpoint = build_shared_obs_authorized_current_endpoint_v1(
        parts=current,
        axis_mapping=current_recipient.axis_mapping,
    )
    if (
        raw_frame.episode_id != current.source_episode_id
        or raw_frame.frame_index != current.source_frame_index
        or raw_frame.frame_id != current_frame.frame_id
        or raw_frame.simulator_step_count != current.source_simulator_step_count
        or raw_frame.recipient_public_agent_id != recipient_roster.public_agent_id
        or raw_frame.recipient_public_agent_id != current.recipient_public_agent_id
        or raw_frame.recipient_frame_id != current.source_recipient_frame_id
    ):
        raise ValueError("raw SharedObs identity does not join authorized s_n.")

    current_living_ids = {
        roster.public_agent_id
        for roster in context.roster
        if roster.configured_active
        and current_frame.snapshot.alive_mask[roster.global_slot]
    }
    current_sensor_ids = tuple(
        source.source_public_agent_id
        for source in current.authorized_sensor_sources
        if source.source_public_agent_id in current_living_ids
    )
    corpse_overlay = build_local_oracle_corpse_overlay_v1(
        context,
        current_frame,
        current.scene,
        authority_session_id=raw_frame.session_id,
        source_authority_epoch=raw_frame.revision,
        recipient_public_agent_id=current.recipient_public_agent_id,
        living_sensor_public_agent_ids=current_sensor_ids,
    )
    validate_local_oracle_corpse_overlay_against_source_v1(
        corpse_overlay,
        context,
        current_frame,
        current.scene,
        authority_session_id=raw_frame.session_id,
        source_authority_epoch=raw_frame.revision,
        recipient_public_agent_id=current.recipient_public_agent_id,
        living_sensor_public_agent_ids=current_sensor_ids,
    )

    visual_events = None
    if view is not None:
        if previous is None:  # pragma: no cover - built above with the view.
            raise AssertionError("live SharedObs start scene disappeared")
        previous_living_ids = {
            roster.public_agent_id
            for roster in context.roster
            if roster.configured_active
            and view.start_frame.snapshot.alive_mask[roster.global_slot]
        }
        previous_sensor_ids = tuple(
            source.source_public_agent_id
            for source in previous.authorized_sensor_sources
            if source.source_public_agent_id in previous_living_ids
        )
        previous_overlay = build_local_oracle_corpse_overlay_v1(
            context,
            view.start_frame,
            previous.scene,
            authority_session_id=raw_frame.session_id,
            source_authority_epoch=raw_frame.revision,
            recipient_public_agent_id=current.recipient_public_agent_id,
            living_sensor_public_agent_ids=previous_sensor_ids,
        )
        incoming_recipient_transition_id = raw_frame.incoming_recipient_transition_id
        if incoming_recipient_transition_id is None:
            raise ValueError("non-initial SharedObs frame lacks its incoming identity.")
        incoming_batch = build_visual_event_batch_v2(view)
        visual_events = build_agent_pov_visual_incoming_summary_v1(
            incoming_batch,
            transition_start_scene=previous.scene,
            successor_scene=current.scene,
            transition_start_corpse_choreography_scene=(
                compose_local_oracle_corpse_scene_v1(
                    previous.scene,
                    previous_overlay,
                    researcher_class_mechanics=researcher_space.class_mechanics,
                )
            ),
            successor_corpse_choreography_scene=(
                compose_local_oracle_corpse_scene_v1(
                    current.scene,
                    corpse_overlay,
                    researcher_class_mechanics=researcher_space.class_mechanics,
                )
            ),
            recipient_public_agent_id=current.recipient_public_agent_id,
            incoming_recipient_transition_id=incoming_recipient_transition_id,
            incoming_start_recipient_frame_id=previous.source_recipient_frame_id,
            incoming_successor_recipient_frame_id=current.source_recipient_frame_id,
            home_fall_global_slots=visual_batch_home_fall_slots_v1(
                context, incoming_batch
            ),
        )

    latest_events = build_shared_obs_incoming_summary_v1(previous, current)
    latest_transition = build_shared_obs_latest_transition_v1(
        None if view is None else view.transition,
        successor=current,
        action_axis=endpoint.action_axis,
        authorized_recipient_global_slot=authorized_recipient_global_slot,
    )
    if raw_frame.incoming_recipient_transition_id != (
        None if latest_transition is None else latest_transition.incoming_transition_id
    ):
        raise ValueError("raw SharedObs incoming identity does not join Latest.")

    if pending_action.selected_global_target_slot is None:
        pending_target_action = 0
    else:
        catalog = context.static_mechanics_catalog
        target_axis = catalog.global_recipient_slot_by_actor_and_target_action[
            authorized_recipient_global_slot
        ]
        try:
            pending_target_action = target_axis.index(
                pending_action.selected_global_target_slot
            )
        except ValueError as error:
            raise ValueError(
                "pending target is absent from the SharedObs target axis."
            ) from error

    submission_scope = raw_frame.pending_submission_scope
    if (
        submission_scope == "joint_turn"
        and researcher_space.pending_inspection.inspection_kind == "policy_read_only"
    ):
        input_inspection = _policy_inspection_v1(
            build_live_shared_obs_draft_inspection_v1(
                current,
                current_recipient,
                authorized_recipient_global_slot=authorized_recipient_global_slot,
                draft_move_action=0,
                draft_target_action=0,
                draft_armed_lane="none",
            )
        )
    elif submission_scope == "joint_turn":
        input_inspection = LiveEditableDraftInspectionV1(
            inspection_kind="editable_live_draft",
            submission_scope="joint_turn",
            draft=build_live_shared_obs_draft_inspection_v1(
                current,
                current_recipient,
                authorized_recipient_global_slot=authorized_recipient_global_slot,
                draft_move_action=pending_action.move_action,
                draft_target_action=pending_target_action,
                draft_armed_lane=_draft_lane_v1(
                    target_action=pending_target_action,
                    raw_armed_lane=pending_action.armed_lane,
                ),
            ),
        )
    elif submission_scope == "scripted_playback":
        input_inspection = LiveScriptedPlaybackInspectionV1(
            inspection_kind="scripted_playback_inspection",
            submission_scope="scripted_playback",
            editable_draft_available=False,
            advance_semantics="registered_script_frame",
        )
    else:  # pragma: no cover - exact raw model narrows this branch.
        raise ValueError("SharedObs live frame has an invalid submission scope.")

    source = LiveSharedObsPresentationSourceIdentityV1(
        source_kind="live_shared_obs_visual_union_frame",
        source_session_id=raw_frame.session_id,
        source_run_generation=raw_frame.run_generation,
        source_revision=raw_frame.revision,
        source_authority_epoch=raw_frame.revision,
        episode_id=current.source_episode_id,
        source_frame_index=current.source_frame_index,
        source_recipient_public_agent_id=current.recipient_public_agent_id,
        source_recipient_frame_id=current.source_recipient_frame_id,
        source_simulator_step_count=current.source_simulator_step_count,
        source_submission_scope=submission_scope,
        source_authorized_endpoint_digest_sha256=(
            endpoint.authorized_endpoint_digest_sha256
        ),
    )
    return LiveSharedObsAuthorizedPresentationFrameV1(
        run_state=raw_frame.run_state,
        fresh_launch_required=raw_frame.fresh_launch_required,
        match_summary=build_match_summary_v1(
            context,
            current_frame,
            () if view is None else view.transition.events,
            transition_start_frame=None if view is None else view.start_frame,
            scores_ten_ticks_ago=scores_ten_ticks_ago,
        ),
        champion_streaks=champion_streaks,
        match_honors=match_honors,
        match_moments=match_moments,
        schema_version=1,
        presentation_kind="live_shared_obs_agent_pov",
        product_kind="combat_debugger",
        source=source,
        authority=SharedObsPresentationAuthorityV1(
            authority_kind="agent_pov",
            observation_mode="shared_obs_visual_union",
            recipient_public_agent_id=current.recipient_public_agent_id,
            recipient_presentation_key=current.recipient_presentation_key,
            projection_basis="authorized_same_epoch_sensor_source_visual_union",
            exact_actor_input_export_available=False,
        ),
        analysis_mode="analysis",
        current_endpoint=endpoint,
        local_oracle_corpse_overlay=corpse_overlay,
        latest_events=latest_events,
        visual_events=visual_events,
        latest_transition=latest_transition,
        technical_frame=LiveSharedObsTechnicalFrameV1(
            technical_kind="live_shared_obs_technical_frame",
            episode_id=source.episode_id,
            recipient_frame_index=source.source_frame_index,
            simulator_step_count=source.source_simulator_step_count,
            incoming_recipient_transition_id=(
                None
                if latest_transition is None
                else latest_transition.incoming_transition_id
            ),
        ),
        live_inspection=LiveSharedObsInspectionEnvelopeV1(
            envelope_kind="live_shared_obs_source_bound_inspection",
            source_session_id=source.source_session_id,
            source_run_generation=source.source_run_generation,
            source_revision=source.source_revision,
            source_authority_epoch=source.source_authority_epoch,
            episode_id=source.episode_id,
            source_frame_index=source.source_frame_index,
            source_recipient_public_agent_id=source.source_recipient_public_agent_id,
            source_recipient_frame_id=source.source_recipient_frame_id,
            source_simulator_step_count=source.source_simulator_step_count,
            inspection=input_inspection,
        ),
        researcher_space=researcher_space,
    )


__all__ = [
    "build_live_no_shared_obs_authorized_presentation_v1",
    "build_live_oracle_authorized_presentation_v1",
    "build_live_researcher_space_v1",
    "build_live_shared_obs_authorized_presentation_v1",
]
