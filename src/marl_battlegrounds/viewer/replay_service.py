"""Serve a loaded replay without changing its captured trajectory.

``ReplayViewerService`` owns the current cursor and display choices. It
serializes commands with a lock, rejects stale revisions, and caches recent
command identities. Researcher analysis and actor-specific projections are
built lazily and reused. The HTTP layer owns authentication and file responses;
this service returns checked models or encoded bytes and runs no simulator.

It also owns the per-audience timeline marks of the Saga Chronicle
(``ReplayViewerService.timeline_marks``, served at
``/api/replay/timeline-marks``): falls, Ultimates and returns per tick plus
fight chapters, built by ``build_replay_fight_chapters_v1``. Oracle View marks
come from the recorded events. Agent POV marks come only from the Agent POV
visual events of the current recipient, so a fight the team never saw is
never marked.

It also keeps each view's Champion Streaks table (``_champion_streaks``,
counted by ``viewer.champion_streaks``), grown frame by frame once per
replay and view and shared with the timeline marks' Agent POV pass, so every
presentation's ``champion_streaks`` field is exact after seeks, view switches
and reloads. In the same way it keeps each view's Herald Cards history
(``_match_moments``, counted by ``viewer.match_moments``: the fresh start,
the largest earlier lead, unanswered runs and rescues), which reuses the
streak pass's Agent POV takedowns, for every presentation's
``match_moments`` field. Every view's match summary also gets the recorded
scores 10 frames back for the Momentum switch.

On the completion frame (the last frame of a finished Team Deathmatch
replay) every presentation also carries that view's ``match_honors``
(``_match_honors``): the whole replay's per-champion totals, added up once
per replay from each transition's events and kill credit
(``viewer.match_summary``), then picked for Oracle View (every champion) or
Agent POV (the recipient's own team).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import pairwise
from secrets import token_urlsafe
from threading import RLock
from typing import TYPE_CHECKING, Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V2,
)
from marl_battlegrounds.evaluation.events import team_deathmatch_home_fall_slots_v1
from marl_battlegrounds.evaluation.metric_catalog import METRIC_SCHEMA_VERSION
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV1,
    AssignedPolicySlotV2,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovReplayContent,
    export_actor_pov_replay_v1,
    export_actor_pov_replay_v2,
    export_actor_pov_replay_v3,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayArtifactReferenceV1,
    ReplayArtifactV1,
)
from marl_battlegrounds.evaluation.replay_io import (
    LoadedReplay,
    LoadedReplayBundle,
    LoadedReplayBundleV1,
    canonical_metric_report_artifact_json_bytes_v1,
)
from marl_battlegrounds.evaluation.replay_v2 import (
    ReplayArtifactReferenceV2,
    ReplayArtifactV2,
    replay_reference_v2,
)
from marl_battlegrounds.evaluation.replay_v3 import (
    ReplayArtifactReferenceV3,
    ReplayArtifactV3,
    replay_reference_v3,
)
from marl_battlegrounds.evaluation.replay_v4 import (
    ReplayArtifactReferenceV4,
    ReplayArtifactV4,
    replay_reference_v4,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    build_no_shared_obs_authorized_scene_v1,
    build_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AgentPovVisualIncomingSummaryV1,
    AuthorizedBattlefieldSceneV1,
    AuthorizedClassMechanics,
    build_agent_pov_visual_incoming_summary_v1,
    visual_batch_home_fall_slots_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    SharedObsSourceMaterialProjection,
    build_researcher_analyzer_projection_v2,
    build_shared_obs_authority_source_material_projection_v1,
    build_status_source_evidence_index_v2,
    build_visual_event_batch_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovProjectionIndexV1,
    build_actor_pov_analyzer_projection_v1,
    build_actor_pov_projection_index_v1,
)
from marl_battlegrounds.viewer.champion_streaks import (
    DAMAGING_ULTIMATE_CLASS_IDS,
    NO_TAKEDOWNS,
    ZERO_COUNTS,
    ChampionStreakV1,
    TickTakedownsV1,
    advance_takedown_counts_v1,
    agent_pov_tick_takedowns_v1,
    champion_streaks_v1,
    oracle_tick_takedowns_v1,
)
from marl_battlegrounds.viewer.match_moments import (
    NO_RUNS,
    MatchMomentFactsV1,
    RescueFact,
    advance_team_runs_v1,
    agent_pov_tick_rescues_v1,
    is_fresh_start_v1,
    match_moment_facts_v1,
    oracle_tick_rescues_v1,
    score_lead_v1,
)
from marl_battlegrounds.viewer.match_summary import (
    MOMENTUM_TICKS,
    MatchHonorsV1,
    MatchTotalsV1,
    build_match_summary_v1,
    match_honors_v1,
    match_transition_totals_v1,
)
from marl_battlegrounds.viewer.no_shared_visual import (
    build_replay_no_shared_obs_visual_content_v1,
)
from marl_battlegrounds.viewer.presentation import (
    build_replay_no_shared_obs_authorized_presentation_v1,
    build_replay_oracle_authorized_presentation_v1,
    build_replay_researcher_space_v1,
    build_replay_shared_obs_authorized_presentation_v1,
)
from marl_battlegrounds.viewer.presentation_protocol import (
    PresentationResourceResultV1,
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplayResearcherSpaceV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
)
from marl_battlegrounds.viewer.replay_protocol import (
    ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1,
    ActorPovProcessingDisclosureV1,
    ActorPovReplayCompletionBadgeV1,
    ActorPovReplayTimelineRowV1,
    ActorPovReplayTimelineV1,
    ActorPovReplayViewerFrameV1,
    ReplayAbsoluteSeekCommandV1,
    ReplayApiErrorV1,
    ReplayArtifactFactsV1,
    ReplayArtifactSummaryV1,
    ReplayCommandRequestV1,
    ReplayCommandResponseV1,
    ReplayCompletionBadgeV1,
    ReplayCursorV1,
    ReplayExitCommandV1,
    ReplayFirstFrameCommandV1,
    ReplayLastFrameCommandV1,
    ReplayNextFrameCommandV1,
    ReplayPresetV1,
    ReplayPreviousFrameCommandV1,
    ReplayProcessingBadgeV1,
    ReplaySelectAgentCommandV1,
    ReplaySetPovActorCommandV1,
    ReplaySetPresetCommandV1,
    ReplaySetRangesCommandV1,
    ReplaySetVerbosityCommandV1,
    ReplaySetViewCommandV1,
    ReplayTimelineEndpointKindV1,
    ReplayTimelineV1,
    ReplayViewerFrameV1,
    ReplayViewModeV1,
    ResearcherReplayTimelineRowV1,
    ResearcherReplayTimelineV1,
    ResearcherReplayViewerFrameV1,
    SharedObsAgentPovReplayArtifactSummaryV1,
    SharedObsAgentPovReplayTimelineRowV1,
    SharedObsAgentPovReplayTimelineV1,
    SharedObsAgentPovReplayViewerFrameV1,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.analysis import ReplayAnalysis

_COMMAND_RECORD_LIMIT = 256
_METRIC_REPORT_SUFFIX = ".marlbg-metrics.json"
FIGHT_MERGE_QUIET_TICKS = 2
"""A fight keeps growing while its next event comes after at most this many
quiet ticks (ticks with no opening or extending event)."""

type ReplayServiceOutcomeV1 = Literal[
    "response",
    "invalid_cursor",
    "audience_unavailable",
    "stale_revision",
    "command_id_conflict",
    "server_shutting_down",
    "service_faulted",
]

type ReplayMetricReportOutcomeV1 = Literal[
    "available",
    "missing",
    "forbidden",
]


@dataclass(frozen=True, slots=True)
class ReplayServiceCommandResultV1:
    """One transport-neutral replay command result."""

    outcome: ReplayServiceOutcomeV1
    payload: ReplayCommandResponseV1 | ReplayApiErrorV1
    shutdown_requested: bool = False


@dataclass(frozen=True, slots=True)
class ReplayMetricReportResultV1:
    """One transport-neutral canonical metric-report retrieval result."""

    outcome: ReplayMetricReportOutcomeV1
    payload: bytes | None
    filename: str | None

    def __post_init__(self) -> None:
        """Require attachment bytes and a name only when the metric report is
        available.
        """
        if self.outcome not in ("available", "missing", "forbidden"):
            raise ValueError("unknown replay metric-report outcome")
        if self.outcome == "available":
            if type(self.payload) is not bytes:
                raise TypeError("available metric report requires immutable bytes")
            if type(self.filename) is not str or not self.filename:
                raise TypeError("available metric report requires a filename")
            return
        if self.payload is not None or self.filename is not None:
            raise ValueError(
                "unavailable metric report cannot carry bytes or a filename"
            )


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    """Retain a request fingerprint and shutdown result for duplicate detection."""

    fingerprint: str
    shutdown_requested: bool


type _PovScenes = tuple[AuthorizedBattlefieldSceneV1, AuthorizedBattlefieldSceneV1, str]
"""One Agent POV recipient's (scene, corpse scene, recipient frame ID) at a frame."""


@dataclass(slots=True)
class _ChampionStreakTable:
    """One view's Champion Streaks count, grown frame by frame from frame zero.

    counts holds the ten takedown counts (global slot order) after the last
    counted frame; reached[i] holds the (global slot, streak) pairs reached on
    frame i, so len(reached) - 1 is the last counted frame. Mutable: only
    ReplayViewerService._champion_streaks changes it, under its lock.
    """

    counts: tuple[int, ...] = ZERO_COUNTS
    reached: list[tuple[tuple[int, int], ...]] = field(default_factory=lambda: [()])


@dataclass(slots=True)
class _MatchMomentTable:
    """One view's Herald Cards runs and rescues, grown frame by frame from frame zero.

    runs holds (Team A, Team B) takedowns in a row without a loss after the
    last counted frame; reached[i] holds the (team ID, run) pairs reached on
    frame i and rescues[i] the (global slot, by_ultimate) rescues this view
    saw on frame i (viewer.match_moments), so len(reached) - 1 is the last
    counted frame. Mutable: only ReplayViewerService._match_moments changes
    it, under the streak lock.
    """

    runs: tuple[int, int] = NO_RUNS
    reached: list[tuple[tuple[int, int], ...]] = field(default_factory=lambda: [()])
    rescues: list[tuple[RescueFact, ...]] = field(default_factory=lambda: [()])


@dataclass(frozen=True, slots=True)
class _PovCacheEntry:
    """Keep one actor's projected capture, timeline, index, and exact-export
    capability.
    """

    content: ActorPovReplayContent
    completion: ActorPovReplayCompletionBadgeV1
    projection_index: ActorPovProjectionIndexV1
    timeline: ActorPovReplayTimelineV1
    exact_actor_input_export_available: bool


def _safe_metric_report_filename(episode_id: str) -> str:
    """Return the bounded attachment basename derived from one episode ID."""
    if type(episode_id) is not str:
        raise TypeError("episode_id must be a string")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", episode_id).strip("._-")
    stem = stem[:96].strip("._-")
    stem = stem.replace(_METRIC_REPORT_SUFFIX, "-").strip("._-") or "replay"
    return f"{stem}{_METRIC_REPORT_SUFFIX}"


def _replay_reference(
    bundle: LoadedReplayBundle,
) -> (
    ReplayArtifactReferenceV1
    | ReplayArtifactReferenceV2
    | ReplayArtifactReferenceV3
    | ReplayArtifactReferenceV4
):
    """Build the version-appropriate immutable replay identity and byte-length facts.

    Replay V1-V4 each get their own reference version. Raise TypeError for any
    other replay type instead of labelling it as V1.
    """
    replay = bundle.replay
    if type(replay) is ReplayArtifactV4:
        return replay_reference_v4(replay)
    if type(replay) is ReplayArtifactV3:
        return replay_reference_v3(replay)
    if type(replay) is ReplayArtifactV2:
        return replay_reference_v2(replay)
    if type(replay) is not ReplayArtifactV1:
        raise TypeError("replay reference requires an exact supported replay")
    return ReplayArtifactReferenceV1(
        artifact_id=replay.artifact_id,
        episode_id=replay.header.context.identity.episode_id,
        context_digest_sha256=replay.header.context_digest_sha256,
        trajectory_content_digest_sha256=replay.trajectory_content_digest_sha256,
        canonical_digest_sha256=replay.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(replay)),
    )


def _completion_badge(bundle: LoadedReplayBundle) -> ReplayCompletionBadgeV1:
    """Copy recorded completion and failure facts into the researcher badge."""
    completion = bundle.replay.completion
    return ReplayCompletionBadgeV1(
        episode_id=completion.episode_id,
        completion_state=completion.completion_state,
        expected_transition_count=completion.expected_transition_count,
        validated_transition_count=completion.validated_transition_count,
        last_valid_frame_index=completion.last_valid_frame_index,
        last_valid_frame_id=completion.last_valid_frame_id,
        terminated=completion.terminated,
        truncated=completion.truncated,
        completion_bases=completion.completion_bases,
        end_or_failure_reason=completion.end_or_failure_reason,
        failure_origin=completion.failure_origin,
    )


def _processing_badge(bundle: LoadedReplayBundle) -> ReplayProcessingBadgeV1:
    """Report legacy metric processing; newer capture-only formats report not
    requested.

    Replay V2-V4 record no metric processing; only replay V1 carries it.
    """
    if type(bundle.replay) in (ReplayArtifactV2, ReplayArtifactV3, ReplayArtifactV4):
        return ReplayProcessingBadgeV1(
            status="not_requested", processed_transition_count=0
        )
    processing = cast(ReplayArtifactV1, bundle.replay).processing_status
    failure = processing.failure
    return ReplayProcessingBadgeV1(
        status=processing.status,
        processed_transition_count=processing.processed_transition_count,
        failure_stage=None if failure is None else failure.stage,
        failure_code=None if failure is None else failure.code,
        attempted_transition_index=(
            None if failure is None else failure.attempted_transition_index
        ),
    )


def _pov_completion_badge(
    content: ActorPovReplayContent,
) -> ActorPovReplayCompletionBadgeV1:
    """Copy only the actor-export completion facts into a POV badge."""
    completion = content.completion
    return ActorPovReplayCompletionBadgeV1(
        episode_id=content.episode_id,
        completion_state=completion.completion_state,
        expected_transition_count=completion.expected_transition_count,
        captured_transition_count=completion.captured_transition_count,
        terminated=completion.terminated,
        truncated=completion.truncated,
        completion_bases=completion.completion_bases,
        public_end_or_failure_reason=completion.public_end_or_failure_reason,
    )


def _shared_completion_badge(
    bundle: LoadedReplayBundle,
) -> ActorPovReplayCompletionBadgeV1:
    """Project physical SharedObs completion without researcher failure detail."""
    completion = bundle.replay.completion
    return ActorPovReplayCompletionBadgeV1(
        episode_id=completion.episode_id,
        completion_state=completion.completion_state,
        expected_transition_count=completion.expected_transition_count,
        captured_transition_count=completion.validated_transition_count,
        terminated=completion.terminated,
        truncated=completion.truncated,
        completion_bases=completion.completion_bases,
        public_end_or_failure_reason=(
            None if completion.completion_state == "complete" else "captured_prefix"
        ),
    )


def _endpoint_kind(
    completion: ReplayCompletionBadgeV1 | ActorPovReplayCompletionBadgeV1,
) -> ReplayTimelineEndpointKindV1:
    """Map complete endpoint bases or an incomplete prefix to the timeline end
    marker.
    """
    if completion.completion_state != "complete":
        return "captured_prefix"
    if completion.completion_bases == ("task_terminal", "declared_horizon"):
        return "task_terminal_and_declared_horizon"
    if completion.completion_bases == ("task_terminal",):
        return "task_terminal"
    if completion.completion_bases == ("declared_horizon",):
        return "declared_horizon"
    raise ValueError("complete replay has an unsupported endpoint basis")


class _TimelineMarkModel(BaseModel):
    """Strict frozen base for timeline-mark records; unknown fields are refused."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReplayTimelineAgentMarkV1(_TimelineMarkModel):
    """One champion named by a timeline mark.

    Attributes
    ----------
    public_agent_id : str
        The champion's public ID, as in every presentation.
    team_id : {1, 2}
        Team A (1) or Team B (2).
    class_id : {1, 2, 3, 4, 5}
        Mage, Warrior, Hunter, Rogue or Priest.
    """

    public_agent_id: Annotated[str, Field(min_length=1)]
    team_id: Literal[1, 2]
    class_id: Literal[1, 2, 3, 4, 5]


class ReplayTimelineFallMarkV1(ReplayTimelineAgentMarkV1):
    """A champion that fell on this tick.

    Attributes
    ----------
    home_fall : bool
        True when the fall started inside the champion's own Red Zone and so
        gave the other team 2 points.
    """

    home_fall: bool


class ReplayTimelineUltimateMarkV1(ReplayTimelineAgentMarkV1):
    """A champion that used its Ultimate on this tick (the caster).

    Attributes
    ----------
    damaging : bool
        True for Charge, Freezing Trap and Crippling Poison (Warrior, Hunter,
        Rogue), which can open a fight. False for Burst and Holy Word:
        Salvation, which only extend one.
    """

    damaging: bool


class ReplayTimelineTickMarksV1(_TimelineMarkModel):
    """Everything one audience may mark on one tick of the Chronicle.

    Attributes
    ----------
    frame_index : int
        The frame this tick's transition enters (1 or more). Frame zero has
        no incoming tick and never appears.
    falls : tuple of ReplayTimelineFallMarkV1
        Champions that fell, in global slot order.
    ultimates : tuple of ReplayTimelineUltimateMarkV1
        Ultimate casters, in event order.
    returns : tuple of ReplayTimelineAgentMarkV1
        Champions that returned (respawned), in global slot order.
    """

    frame_index: Annotated[int, Field(ge=1)]
    falls: tuple[ReplayTimelineFallMarkV1, ...]
    ultimates: tuple[ReplayTimelineUltimateMarkV1, ...]
    returns: tuple[ReplayTimelineAgentMarkV1, ...]


class ReplayTimelineFightV1(_TimelineMarkModel):
    """One fight chapter: a viewing aid, never a metric.

    Attributes
    ----------
    fight_number : int
        1 for the first fight, counting up in time.
    first_frame_index, last_frame_index : int
        Frames entered by the fight's first and last ticks, both included.
    ultimate_count : int
        Ultimates marked on the ticks from first to last frame.
    fallen_count : int
        Falls marked on the ticks from first to last frame.
    """

    fight_number: Annotated[int, Field(ge=1)]
    first_frame_index: Annotated[int, Field(ge=1)]
    last_frame_index: Annotated[int, Field(ge=1)]
    ultimate_count: Annotated[int, Field(ge=0)]
    fallen_count: Annotated[int, Field(ge=0)]


class ReplayTimelineMarksV1(_TimelineMarkModel):
    """The Saga Chronicle marks of one replay for one audience.

    Attributes
    ----------
    schema_version : 1
        Wire version of this payload.
    marks_kind : 'replay_timeline_marks'
        Fixed payload discriminator.
    audience : {'oracle', 'agent_pov'}
        Oracle View (every recorded event) or Agent POV (only what the
        recipient's team saw).
    recipient_public_agent_id : str or None
        The Agent POV recipient; None in Oracle View.
    final_frame_index : int
        Last captured frame index (the number of captured transitions).
    ticks : tuple of ReplayTimelineTickMarksV1
        Only ticks with at least one mark, in time order.
    fights : tuple of ReplayTimelineFightV1
        Fight chapters from build_replay_fight_chapters_v1, in time order.
    """

    schema_version: Literal[1]
    marks_kind: Literal["replay_timeline_marks"]
    audience: Literal["oracle", "agent_pov"]
    recipient_public_agent_id: Annotated[str, Field(min_length=1)] | None
    final_frame_index: Annotated[int, Field(ge=0)]
    ticks: tuple[ReplayTimelineTickMarksV1, ...]
    fights: tuple[ReplayTimelineFightV1, ...]


def build_replay_fight_chapters_v1(
    ticks: tuple[ReplayTimelineTickMarksV1, ...],
) -> tuple[ReplayTimelineFightV1, ...]:
    """Group marked ticks into fight chapters with the Saga Chronicle rule.

    A fall or a damaging Ultimate (Charge, Freezing Trap, Crippling Poison)
    opens a fight. A tick with only Burst or Holy Word: Salvation extends an
    open fight and never opens one. A fight grows while its next opening or
    extending tick comes after at most FIGHT_MERGE_QUIET_TICKS (2) quiet
    ticks; otherwise a new fight starts. Returns alone never touch a fight.

    Parameters
    ----------
    ticks : tuple of ReplayTimelineTickMarksV1
        One audience's marked ticks, in increasing frame_index order. Pass
        Agent POV ticks to get the chapters that audience saw.

    Returns
    -------
    tuple of ReplayTimelineFightV1
        Fights numbered from 1 in time order; empty when nothing opens one.
        Counts cover every marked tick from the first to the last frame.

    Raises
    ------
    ValueError
        The ticks are not in strictly increasing frame order.

    Examples
    --------
    Falls at frames 3 and 6 (2 quiet ticks between) make one fight; a fall
    at frame 10 after 3 quiet ticks starts a second fight.
    """
    frames = tuple(tick.frame_index for tick in ticks)
    if any(later <= earlier for earlier, later in pairwise(frames)):
        raise ValueError("timeline ticks must be in strictly increasing frame order")
    spans: list[list[int]] = []
    for tick in ticks:
        opening = bool(tick.falls) or any(row.damaging for row in tick.ultimates)
        extending = not opening and bool(tick.ultimates)
        near = bool(spans) and (
            tick.frame_index - spans[-1][1] - 1 <= FIGHT_MERGE_QUIET_TICKS
        )
        if opening:
            if near:
                spans[-1][1] = tick.frame_index
            else:
                spans.append([tick.frame_index, tick.frame_index])
        elif extending and near:
            spans[-1][1] = tick.frame_index
    return tuple(
        ReplayTimelineFightV1(
            fight_number=number,
            first_frame_index=first,
            last_frame_index=last,
            ultimate_count=sum(
                len(tick.ultimates)
                for tick in ticks
                if first <= tick.frame_index <= last
            ),
            fallen_count=sum(
                len(tick.falls) for tick in ticks if first <= tick.frame_index <= last
            ),
        )
        for number, (first, last) in enumerate(spans, start=1)
    )


class ReplayViewerService:
    """Serialize immutable cursor and viewer-owned presentation around one replay.

    Initial reference, selection, and lane values describe the viewer handoff;
    they are not historical fields recovered independently at each replay cursor.
    """

    def __init__(
        self,
        bundle: LoadedReplayBundle,
        *,
        initial_frame_index: int = 0,
        view_mode: ReplayViewModeV1 = "researcher",
        reference_global_slot: int | None = None,
        selected_global_slot: int | None = None,
        armed_lane: Literal[0, 1] | None = None,
        pov_global_slot: int | None = None,
        preset: ReplayPresetV1 | Literal["technical", "debug"] = "analysis",
        show_ranges: bool = False,
        verbose: bool = False,
        viewer_session_id: str | None = None,
    ) -> None:
        """Open a viewer state around a validated in-memory replay.

        Parameters
        ----------
        bundle : LoadedReplayBundle
            Exact LoadedReplay or LoadedReplayBundleV1 root from the public loader.
        initial_frame_index : int, optional
            Captured frame index, from zero through the last retained frame. Defaults
            to zero; the final frame index equals the number of captured transitions.
        view_mode : {'researcher', 'pov'}, optional
            Initial display authority. Defaults to researcher.
        reference_global_slot : int or None, optional
            Configured-active researcher reference. None uses the first focal slot,
            if one exists. This is viewer state, not a recorded historical selection.
        selected_global_slot : int or None, optional
            Initial active inspection target. Requires a reference when supplied.
            Defaults to None.
        armed_lane : {0, 1} or None, optional
            Viewer inspection lane. Requires a selected target. Defaults to None.
        pov_global_slot : int or None, optional
            Active actor whose POV is displayed. None uses the first focal slot.
            POV mode requires a resolved actor.
        preset : ReplayPresetV1 or {'technical', 'debug'}, optional
            Accepted compatibility preset. All accepted values use analysis mode.
            Defaults to analysis.
        show_ranges : bool, optional
            Initial range-overlay preference. Defaults to False.
        verbose : bool, optional
            Accepted compatibility flag; current viewer verbosity remains False.
        viewer_session_id : str or None, optional
            Nonblank viewer identity. None creates a random identity.

        Raises
        ------
        TypeError
            The loaded root or boolean flags have unsupported exact types.
        ValueError
            Status, cursor, view, slot selection, lane, or viewer identity is invalid.

        Notes
        -----
        Construction builds the researcher timeline and status-evidence index. Actor
        projections are cached as needed. It performs no replay write or simulation.
        Initial viewer choices are not reconstructed from each historical frame.
        """
        if type(bundle) not in (LoadedReplayBundleV1, LoadedReplay):
            raise TypeError("bundle must be an exact loaded replay root")
        if bundle.status not in (
            "complete",
            "metric_report_missing",
            "not_recorded",
        ) or (bundle.metric_report_artifact is None) != (bundle.status != "complete"):
            raise ValueError(
                "loaded replay status must match metric-report sidecar availability"
            )
        if type(initial_frame_index) is not int or not (
            0 <= initial_frame_index < len(bundle.replay.frames)
        ):
            raise ValueError("initial frame index is outside the captured replay")
        if view_mode not in ("researcher", "pov"):
            raise ValueError("unknown replay view mode")
        if preset not in ("presentation", "analysis", "technical", "debug"):
            raise ValueError("unknown replay preset")
        if type(show_ranges) is not bool or type(verbose) is not bool:
            raise TypeError("replay presentation flags must be booleans")
        if viewer_session_id is not None and (
            type(viewer_session_id) is not str or not viewer_session_id.strip()
        ):
            raise ValueError("viewer_session_id must be a nonempty string when set")

        self._bundle = bundle
        self._metric_analysis: ReplayAnalysis | None = None
        self._metric_analysis_lock = RLock()
        self._metric_catalog_bytes: bytes | None = None
        self._timeline_marks_lock = RLock()
        self._timeline_marks_bytes: dict[tuple[str, int | None], bytes] = {}
        self._replay = bundle.replay
        self._context = bundle.replay.header.context
        # SharedObs source rows by (frame index, slot); see _shared_authority_sources.
        self._source_materials: dict[
            tuple[int, int], SharedObsSourceMaterialProjection
        ] = {}
        # Champion Streaks: one table per view (None is Oracle View, an int an
        # Agent POV recipient), and each recipient's seen takedowns by
        # (recipient, frame index); see _champion_streaks.
        self._champion_streaks_lock = RLock()
        self._champion_streak_tables: dict[int | None, _ChampionStreakTable] = {}
        self._pov_tick_takedowns: dict[tuple[int, int], TickTakedownsV1] = {}
        # Herald Cards history: one table per view, grown under the same lock
        # and reusing _pov_tick_takedowns, plus the public largest lead before
        # each frame (built once); see _match_moments.
        self._match_moment_tables: dict[int | None, _MatchMomentTable] = {}
        self._largest_leads: tuple[int, ...] | None = None
        # Honors: the whole replay's per-champion totals, added up once on
        # the first completion-frame request; see _match_honors.
        self._honors_lock = RLock()
        self._honors_totals: MatchTotalsV1 | None = None
        self._active_slots: tuple[int, ...] = tuple(
            row.global_slot for row in self._context.roster if row.configured_active
        )
        focal_slots: tuple[int, ...] = tuple(
            row.global_slot
            for row in self._context.policy_assignments
            if isinstance(row, (AssignedPolicySlotV1, AssignedPolicySlotV2))
            and row.evaluation_role == "focal"
        )
        default_slot: int | None = min(focal_slots) if focal_slots else None
        researcher_slot: int | None = (
            default_slot if reference_global_slot is None else reference_global_slot
        )
        inspection_slot: int | None = selected_global_slot
        actor_slot: int | None = (
            default_slot if pov_global_slot is None else pov_global_slot
        )
        self._require_active_or_none(
            researcher_slot,
            name="reference_global_slot",
        )
        self._require_active_or_none(inspection_slot, name="selected_global_slot")
        self._require_active_or_none(actor_slot, name="pov_global_slot")
        if armed_lane is not None and (
            type(armed_lane) is not int or armed_lane not in (0, 1)
        ):
            raise ValueError("armed_lane must be the Python int zero or one, or None")
        if inspection_slot is not None and researcher_slot is None:
            raise ValueError("selected_global_slot requires a researcher reference")
        if armed_lane is not None and inspection_slot is None:
            raise ValueError("armed_lane requires a selected_global_slot")
        if view_mode == "pov" and actor_slot is None:
            raise ValueError("POV replay view requires a configured-active actor")

        self._artifact_summary = ReplayArtifactSummaryV1(
            replay_reference=_replay_reference(bundle),
            expected_transition_count=self._replay.header.expected_transition_count,
            recorded_transition_count=len(self._replay.transitions),
            recorded_frame_count=len(self._replay.frames),
            metric_report_availability=(
                "not_recorded"
                if type(bundle.replay)
                in (ReplayArtifactV2, ReplayArtifactV3, ReplayArtifactV4)
                else "available"
                if bundle.metric_report_artifact is not None
                else "missing"
            ),
        )
        self._pov_artifact_summary = ReplayArtifactSummaryV1(
            replay_reference=self._artifact_summary.replay_reference,
            expected_transition_count=self._artifact_summary.expected_transition_count,
            recorded_transition_count=self._artifact_summary.recorded_transition_count,
            recorded_frame_count=self._artifact_summary.recorded_frame_count,
            metric_report_availability=ACTOR_POV_METRIC_REPORT_AVAILABILITY_V1,
        )
        self._completion = _completion_badge(bundle)
        self._shared_completion = _shared_completion_badge(bundle)
        self._processing = _processing_badge(bundle)
        self._artifact_facts = ReplayArtifactFactsV1(
            artifact_summary=self._artifact_summary,
            completion=self._completion,
            processing=self._processing,
        )
        self._status_index = build_status_source_evidence_index_v2(
            self._context,
            self._replay.frames,
            self._replay.transitions,
        )
        self._researcher_timeline = self._build_researcher_timeline()
        self._pov_cache: dict[int, _PovCacheEntry] = {}
        self._shared_timeline_cache: dict[int, SharedObsAgentPovReplayTimelineV1] = {}

        self._viewer_session_id = (
            token_urlsafe(24) if viewer_session_id is None else viewer_session_id
        )
        self._revision = 0
        self._frame_index = initial_frame_index
        self._cursor_generation = 0
        self._choreography_generation = 0
        self._view_mode: ReplayViewModeV1 = view_mode
        self._reference_global_slot: int | None = researcher_slot
        self._inspection_global_slot: int | None = inspection_slot
        self._armed_lane = armed_lane
        self._pov_global_slot: int | None = actor_slot
        self._preset: ReplayPresetV1 = "analysis"
        self._show_ranges = show_ranges
        self._verbose = False
        self._shutting_down = False
        self._faulted = False
        self._lock = RLock()
        self._command_records: OrderedDict[tuple[str, str], _CommandRecord] = (
            OrderedDict()
        )
        self._frame = self._build_frame(
            revision=self._revision,
            frame_index=self._frame_index,
            cursor_generation=self._cursor_generation,
            choreography_generation=self._choreography_generation,
            view_mode=self._view_mode,
            inspection_global_slot=self._inspection_global_slot,
            armed_lane=self._armed_lane,
            pov_global_slot=self._pov_global_slot,
            preset=self._preset,
            show_ranges=self._show_ranges,
            verbose=self._verbose,
            pov_cache=self._pov_cache,
            shared_timeline_cache=self._shared_timeline_cache,
        )

    @property
    def replay_id(self) -> str:
        """Return the saved replay identity, unchanged by cursor or POV choices.

        This host-only value lets a launcher reuse an existing Viewer. It reads
        the identity already checked at load time without rebuilding a frame.
        """
        return self._artifact_summary.replay_reference.artifact_id

    @property
    def show_ranges(self) -> bool:
        """Return the viewer preference without exposing an actor observation."""
        with self._lock:
            return self._show_ranges

    @property
    def revision(self) -> int:
        """Return the committed viewer revision under the service lock."""
        with self._lock:
            return self._revision

    @property
    def command_cache_size(self) -> int:
        """Return how many recent command identities are cached, at most 256."""
        with self._lock:
            return len(self._command_records)

    @property
    def pov_index_build_count(self) -> int:
        """Return the number of cached actor projection indexes."""
        with self._lock:
            return len(self._pov_cache)

    @property
    def shared_timeline_build_count(self) -> int:
        """Return the number of cached SharedObs recipient timelines."""
        with self._lock:
            return len(self._shared_timeline_cache)

    @property
    def shutting_down(self) -> bool:
        """Return whether an accepted exit command fenced new commands."""
        with self._lock:
            return self._shutting_down

    @property
    def faulted(self) -> bool:
        """Return whether an internal failure fenced new commands."""
        with self._lock:
            return self._faulted

    def current_frame(self) -> ReplayViewerFrameV1:
        """Read the current immutable viewer frame under the service lock.

        Returns
        -------
        ReplayViewerFrameV1
            The already built frame for the committed cursor and display choices.
        """
        with self._lock:
            return self._frame

    def current_metric_report(self) -> ReplayMetricReportResultV1:
        """Read the recorded metric attachment independently of battlefield view.

        Returns
        -------
        ReplayMetricReportResultV1
            Canonical JSON bytes and a safe basename when a metric artifact exists;
            otherwise a missing result with no bytes or filename.

        Notes
        -----
        This serializes the recorded artifact under the service lock. It does not
        compute a new analysis or write a file. Actor battlefield view does not hide
        researcher analysis resources from the researcher using the viewer.
        """
        with self._lock:
            artifact = self._bundle.metric_report_artifact
            if artifact is None:
                return ReplayMetricReportResultV1(
                    outcome="missing",
                    payload=None,
                    filename=None,
                )
            return ReplayMetricReportResultV1(
                outcome="available",
                payload=canonical_metric_report_artifact_json_bytes_v1(artifact),
                filename=_safe_metric_report_filename(
                    self._context.identity.episode_id
                ),
            )

    def episode_details(self) -> tuple[bytes, str]:
        """Encode recorded identity and configuration without trajectory arrays.

        Returns
        -------
        tuple of bytes and str
            Canonical JSON content and the filename episode-details.json. Content
            includes context, completion, runtime, wrappers, and captured count.

        Notes
        -----
        This creates an in-memory attachment; the HTTP caller handles delivery.
        """
        replay = self._replay
        return canonical_json_bytes(
            {
                "schema_id": "marlbg.replay.episode_details",
                "schema_version": 1,
                "source_replay": self._artifact_summary.replay_reference,
                "context": self._context,
                "completion": replay.completion,
                "runtime_provenance": replay.header.runtime_provenance,
                "wrapper_stack": replay.header.wrapper_stack,
                "captured_transition_count": len(replay.transitions),
            }
        ), "episode-details.json"

    def metric_analysis(
        self, frame_index: int, scope: str, format_: str
    ) -> tuple[bytes, str | None]:
        """Encode researcher metrics at a cursor or at the final captured frame.

        Parameters
        ----------
        frame_index : int
            Valid captured frame index, checked even for final scope.
        scope : {'cursor', 'final'}
            Whether to summarize the selected frame or all captured transitions.
        format_ : {'json', 'csv'}
            Output encoding. JSON includes applicable statistics; CSV keeps its
            complete fixed column schema.

        Returns
        -------
        tuple of bytes and str or None
            Encoded content and an attachment filename for CSV; JSON returns None
            as the filename.

        Raises
        ------
        ValueError
            Scope or format is unsupported.
        IndexError
            The requested frame is outside the captured replay.

        Notes
        -----
        The first request computes and caches full analysis. Subsequent requests
        reuse it. This creates no file and changes no replay or cursor state.
        """
        if scope not in ("cursor", "final") or format_ not in ("json", "csv"):
            raise ValueError("unsupported metric scope or format")
        self._bundle.frame_at(frame_index)
        analysis = self._analyzed_metrics()
        if format_ == "json":
            summary = analysis.summary(frame_index, scope=scope)
            statistics = cast(list[dict[str, object]], summary["statistics"])
            # The CSV keeps its fixed schema. Transport only rows this roster can
            # display; undefined ratios and reachable zeros remain available.
            summary["statistics"] = [row for row in statistics if row["applicable"]]
            return canonical_json_bytes(summary), None
        selected = analysis.frame_count - 1 if scope == "final" else frame_index
        episode = _safe_metric_report_filename(
            self._context.identity.episode_id
        ).removesuffix(_METRIC_REPORT_SUFFIX)
        return (
            analysis.csv(frame_index, scope=scope).encode("utf-8"),
            f"tdm-metrics__episode-{episode}__schema-{METRIC_SCHEMA_VERSION}"
            f"__{scope}__frame-{selected}.csv",
        )

    def _analyzed_metrics(self) -> ReplayAnalysis:
        """Share one immutable analysis across values, downloads and search."""
        with self._metric_analysis_lock:
            if self._metric_analysis is None:
                from marl_battlegrounds.evaluation.analysis import analyze_replay

                self._metric_analysis = analyze_replay(self._bundle, full=True)
            return self._metric_analysis

    def metric_catalog(self) -> bytes:
        """Encode and cache measurement descriptions for this replay.

        Returns
        -------
        bytes
            Canonical JSON catalog shared across later requests.

        Notes
        -----
        The first call may build full replay analysis; later calls reuse both the
        analysis and encoded catalog under a separate analysis lock.
        """
        with self._metric_analysis_lock:
            if self._metric_catalog_bytes is None:
                self._metric_catalog_bytes = canonical_json_bytes(
                    self._analyzed_metrics().catalog()
                )
            return self._metric_catalog_bytes

    def timeline_marks(self) -> bytes:
        """Return the Saga Chronicle marks for the current audience as JSON bytes.

        Returns
        -------
        bytes
            Canonical JSON of one ReplayTimelineMarksV1. In Oracle View it holds
            every recorded fall, Ultimate and return. In Agent POV it holds only
            the falls, Ultimates and returns found in the current recipient's
            Agent POV visual events (the same filter the arena uses), and the
            fights are built from those marks alone.

        Raises
        ------
        RuntimeError
            Agent POV view has no recipient.
        ValueError
            A recorded event or scene fails its existing validation.

        Notes
        -----
        The audience is read under the service lock; the build then runs under
        a separate marks lock, so cursor commands are not blocked. Each audience
        (Oracle, or each Agent POV recipient) is built once and cached. An
        Agent POV build checks only ticks whose recorded events hold a fall, an
        Ultimate or a return, and builds each needed recipient scene once.
        Measured on a 283-frame SharedObs match on CPU (1 Oct 2026): about
        11 seconds for the first Agent POV build of one recipient and about
        2 milliseconds for Oracle marks; later calls return the cached bytes.
        No file or simulator work.
        """
        with self._lock:
            view_mode = self._view_mode
            recipient = self._pov_global_slot
            if view_mode != "researcher" and recipient is None:
                raise RuntimeError("POV replay view has no selected actor")
            key: tuple[str, int | None] = (
                ("oracle", None)
                if view_mode == "researcher"
                else ("agent_pov", recipient)
            )
            pov_entry = (
                self._pov_entry(recipient, cache=self._pov_cache)
                if key[0] == "agent_pov"
                and recipient is not None
                and self._context.execution_information_mode == "no_shared_obs"
                else None
            )
            class_mechanics = (
                self._researcher_space_for_frame(
                    frame_index=0, selected_global_slot=recipient
                ).class_mechanics
                if key[0] == "agent_pov" and recipient is not None
                else ()
            )
        with self._timeline_marks_lock:
            cached = self._timeline_marks_bytes.get(key)
            if cached is not None:
                return cached
            if key[1] is None:
                ticks = self._oracle_timeline_ticks()
            else:
                ticks = self._agent_pov_timeline_ticks(
                    key[1],
                    pov_entry=pov_entry,
                    class_mechanics=class_mechanics,
                )
            marks = ReplayTimelineMarksV1(
                schema_version=1,
                marks_kind="replay_timeline_marks",
                audience="oracle" if key[1] is None else "agent_pov",
                recipient_public_agent_id=(
                    None
                    if key[1] is None
                    else self._context.roster[key[1]].public_agent_id
                ),
                final_frame_index=len(self._replay.transitions),
                ticks=ticks,
                fights=build_replay_fight_chapters_v1(ticks),
            )
            payload = canonical_json_bytes(marks)
            self._timeline_marks_bytes[key] = payload
            return payload

    def _timeline_agent(self, global_slot: int) -> ReplayTimelineAgentMarkV1:
        """Name one configured-active roster slot for a timeline mark."""
        row = self._context.roster[global_slot]
        return ReplayTimelineAgentMarkV1(
            public_agent_id=row.public_agent_id,
            team_id=cast("Literal[1, 2]", row.configured_team_id),
            class_id=cast("Literal[1, 2, 3, 4, 5]", row.class_id),
        )

    def _timeline_tick(
        self,
        frame_index: int,
        *,
        fall_slots: tuple[int, ...],
        home_fall_slots: frozenset[int],
        ultimate_slots: tuple[int, ...],
        return_slots: tuple[int, ...],
    ) -> ReplayTimelineTickMarksV1 | None:
        """Build one tick's marks from global slots, or None when nothing happened."""
        if not (fall_slots or ultimate_slots or return_slots):
            return None
        falls = tuple(
            ReplayTimelineFallMarkV1(
                **self._timeline_agent(slot).model_dump(),
                home_fall=slot in home_fall_slots,
            )
            for slot in sorted(set(fall_slots))
        )
        ultimates = tuple(
            ReplayTimelineUltimateMarkV1(
                **self._timeline_agent(slot).model_dump(),
                damaging=self._context.roster[slot].class_id
                in DAMAGING_ULTIMATE_CLASS_IDS,
            )
            for slot in ultimate_slots
        )
        returns = tuple(
            self._timeline_agent(slot) for slot in sorted(set(return_slots))
        )
        return ReplayTimelineTickMarksV1(
            frame_index=frame_index,
            falls=falls,
            ultimates=ultimates,
            returns=returns,
        )

    def _recorded_mark_slots(
        self, frame_index: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Read recorded fall, Ultimate-caster and return slots entering a frame."""
        events = self._replay.transitions[frame_index - 1].events
        falls = tuple(
            event.recipient_global_slot
            for event in events
            if event.event_type == "agent_died"
        )
        ultimates = tuple(
            event.source_global_slot
            for event in events
            if event.event_type == "ability_activated"
            and event.ability_component == "ultimate"
        )
        returns = tuple(
            event.agent_global_slot
            for event in events
            if event.event_type == "agent_respawned"
        )
        return falls, ultimates, returns

    def _oracle_timeline_ticks(self) -> tuple[ReplayTimelineTickMarksV1, ...]:
        """Build every marked tick from the recorded evaluation events."""
        ticks: list[ReplayTimelineTickMarksV1] = []
        for frame_index in range(1, len(self._replay.frames)):
            falls, ultimates, returns = self._recorded_mark_slots(frame_index)
            home_falls: frozenset[int] = (
                team_deathmatch_home_fall_slots_v1(
                    self._context,
                    self._replay.frames[frame_index - 1].snapshot.agent_positions,
                    tuple(slot in falls for slot in range(len(self._context.roster))),
                )
                if falls
                else frozenset[int]()
            )
            tick = self._timeline_tick(
                frame_index,
                fall_slots=falls,
                home_fall_slots=home_falls,
                ultimate_slots=ultimates,
                return_slots=returns,
            )
            if tick is not None:
                ticks.append(tick)
        return tuple(ticks)

    def _agent_pov_timeline_ticks(
        self,
        recipient_global_slot: int,
        *,
        pov_entry: _PovCacheEntry | None,
        class_mechanics: tuple[AuthorizedClassMechanics, ...],
    ) -> tuple[ReplayTimelineTickMarksV1, ...]:
        """Build marked ticks from one recipient's Agent POV visual events.

        Only ticks whose recorded events hold a fall, an Ultimate or a return
        are checked; others cannot produce a mark. For each, the recipient's
        Agent POV visual events (_agent_pov_tick_visual_events, the same filter
        the arena uses) give the marks: only its agent_died, Ultimate
        ability_activated and agent_respawned rows. On a tick with a recorded
        fall it also stores the recipient's seen takedowns in
        _pov_tick_takedowns, so a later Champion Streaks count reuses them.
        """
        context = self._context
        slot_by_public_id = {
            row.public_agent_id: row.global_slot for row in context.roster
        }
        scenes: dict[int, _PovScenes] = {}
        ticks: list[ReplayTimelineTickMarksV1] = []
        for frame_index in range(1, len(self._replay.frames)):
            recorded_falls, recorded_ultimates, recorded_returns = (
                self._recorded_mark_slots(frame_index)
            )
            if not (recorded_falls or recorded_ultimates or recorded_returns):
                continue
            visual = self._agent_pov_tick_visual_events(
                recipient_global_slot,
                frame_index,
                pov_entry=pov_entry,
                class_mechanics=class_mechanics,
                scenes=scenes,
            )
            if recorded_falls:
                self._pov_tick_takedowns[(recipient_global_slot, frame_index)] = (
                    agent_pov_tick_takedowns_v1(context, visual)
                )
            falls: list[int] = []
            home_falls: set[int] = set()
            ultimates: list[int] = []
            returns: list[int] = []
            for event in visual.events:
                if event.event_kind == "agent_died":
                    slot = slot_by_public_id[event.recipient_anchor.public_agent_id]
                    falls.append(slot)
                    if event.home_fall:
                        home_falls.add(slot)
                elif (
                    event.event_kind == "ability_activated"
                    and event.ability_component == "ultimate"
                ):
                    ultimates.append(
                        slot_by_public_id[event.source_anchor.public_agent_id]
                    )
                elif event.event_kind == "agent_respawned":
                    returns.append(
                        slot_by_public_id[event.agent_anchor.public_agent_id]
                    )
            tick = self._timeline_tick(
                frame_index,
                fall_slots=tuple(falls),
                home_fall_slots=frozenset(home_falls),
                ultimate_slots=tuple(ultimates),
                return_slots=tuple(returns),
            )
            if tick is not None:
                ticks.append(tick)
        return tuple(ticks)

    def _agent_pov_tick_visual_events(
        self,
        recipient_global_slot: int,
        frame_index: int,
        *,
        pov_entry: _PovCacheEntry | None,
        class_mechanics: tuple[AuthorizedClassMechanics, ...],
        scenes: dict[int, _PovScenes],
    ) -> AgentPovVisualIncomingSummaryV1:
        """Build one recipient's Agent POV visual events for one recorded tick.

        Parameters
        ----------
        recipient_global_slot : int
            The Agent POV recipient, a configured-active global slot 0-9.
        frame_index : int
            The frame the tick enters, 1 to the last captured frame.
        pov_entry : _PovCacheEntry or None
            The recipient's cached NoSharedObs projection, or None for a
            SharedObs replay (its scenes come from _shared_authority_sources).
        class_mechanics : tuple of AuthorizedClassMechanics
            Public class records covering every class, for the corpse scenes.
        scenes : dict
            The caller's per-pass cache of (scene, corpse scene, recipient frame
            ID) by frame index; this call adds the two frames it builds.

        Returns
        -------
        AgentPovVisualIncomingSummaryV1
            build_agent_pov_visual_incoming_summary_v1 over the recipient's
            start and successor scenes and their death-only corpse scenes, the
            same inputs the Agent POV presentation uses, so marks and Champion
            Streaks see exactly what the arena shows.

        Raises
        ------
        ValueError
            A recorded event or scene fails its existing validation.

        Notes
        -----
        Host-only scene and event checks; no lock, file or simulator work. The
        authority label only names the build; it changes no event decision.
        """
        context = self._context
        recipient_public_id = context.roster[recipient_global_slot].public_agent_id
        view = cast(EvaluationTransitionViewV1, self._transition_view(frame_index))
        batch = build_visual_event_batch_v2(view)
        start_scene, start_corpses, start_id = self._agent_pov_scene_at(
            recipient_global_slot,
            frame_index - 1,
            pov_entry=pov_entry,
            class_mechanics=class_mechanics,
            scenes=scenes,
        )
        end_scene, end_corpses, end_id = self._agent_pov_scene_at(
            recipient_global_slot,
            frame_index,
            pov_entry=pov_entry,
            class_mechanics=class_mechanics,
            scenes=scenes,
        )
        transition_id = (
            pov_entry.content.transitions[frame_index - 1].pov_transition_id
            if pov_entry is not None
            else (
                f"{context.identity.episode_id}:shared-obs-visual-union:"
                f"{recipient_public_id}:transition:{frame_index - 1}"
            )
        )
        return build_agent_pov_visual_incoming_summary_v1(
            batch,
            transition_start_scene=start_scene,
            successor_scene=end_scene,
            transition_start_corpse_choreography_scene=start_corpses,
            successor_corpse_choreography_scene=end_corpses,
            recipient_public_agent_id=recipient_public_id,
            incoming_recipient_transition_id=transition_id,
            incoming_start_recipient_frame_id=start_id,
            incoming_successor_recipient_frame_id=end_id,
            home_fall_global_slots=visual_batch_home_fall_slots_v1(context, batch),
        )

    def _agent_pov_scene_at(
        self,
        recipient_global_slot: int,
        frame_index: int,
        *,
        pov_entry: _PovCacheEntry | None,
        class_mechanics: tuple[AuthorizedClassMechanics, ...],
        scenes: dict[int, _PovScenes],
    ) -> _PovScenes:
        """Return one recipient's (scene, corpse scene, frame ID) at one frame.

        The arguments are those of _agent_pov_tick_visual_events; frame_index
        may be 0. A frame already in scenes is returned from it; otherwise it
        is built once and stored there. The corpse scene adds only the corpses
        the recipient's living sensors may show (local Oracle corpse overlay).
        Host-only; no lock, file or simulator work.
        """
        from marl_battlegrounds.viewer.local_oracle_corpse_overlay import (
            build_local_oracle_corpse_overlay_v1,
            compose_local_oracle_corpse_scene_v1,
        )

        cached = scenes.get(frame_index)
        if cached is not None:
            return cached
        context = self._context
        roster = context.roster
        recipient_public_id = roster[recipient_global_slot].public_agent_id
        authority = f"{self._viewer_session_id}:timeline-marks"
        frame = self._replay.frames[frame_index]
        if pov_entry is not None:
            parts = build_no_shared_obs_authorized_scene_v1(
                pov_entry.projection_index,
                public_catalog=context.static_mechanics_catalog,
                authority_session_id=authority,
                frame_index=frame_index,
            )
            sensors: tuple[str, ...] = (
                (recipient_public_id,)
                if frame.snapshot.alive_mask[recipient_global_slot]
                else ()
            )
            scene = parts.scene
            frame_id = parts.source_recipient_frame_id
        else:
            recipient_source, other_sources = self._shared_authority_sources(
                frame_index=frame_index,
                recipient_global_slot=recipient_global_slot,
            )
            shared = build_shared_obs_authorized_scene_v1(
                recipient_source,
                all_active_nonrecipient_source_material=other_sources,
                public_catalog=context.static_mechanics_catalog,
                authority_session_id=authority,
            )
            living = {
                row.public_agent_id
                for row in roster
                if row.configured_active and frame.snapshot.alive_mask[row.global_slot]
            }
            sensors = tuple(
                row.source_public_agent_id
                for row in shared.authorized_sensor_sources
                if row.source_public_agent_id in living
            )
            scene = shared.scene
            frame_id = shared.source_recipient_frame_id
        overlay = build_local_oracle_corpse_overlay_v1(
            context,
            frame,
            scene,
            authority_session_id=authority,
            source_authority_epoch=0,
            recipient_public_agent_id=recipient_public_id,
            living_sensor_public_agent_ids=sensors,
        )
        built = (
            scene,
            compose_local_oracle_corpse_scene_v1(
                scene,
                overlay,
                researcher_class_mechanics=class_mechanics,
            ),
            frame_id,
        )
        scenes[frame_index] = built
        return built

    def _champion_streaks(
        self,
        frame_index: int,
        *,
        recipient_global_slot: int | None,
        pov_entry: _PovCacheEntry | None = None,
    ) -> tuple[ChampionStreakV1, ...]:
        """Return one view's Champion Streaks fact for one captured frame.

        Parameters
        ----------
        frame_index : int
            Captured frame index, 0 to the last frame.
        recipient_global_slot : int or None, keyword-only
            None for Oracle View; otherwise the Agent POV recipient's global
            slot. Each view has its own table, so a view switch never mixes
            counts.
        pov_entry : _PovCacheEntry or None, optional, keyword-only
            The recipient's cached NoSharedObs projection (required for a
            NoSharedObs Agent POV view); None for Oracle View and SharedObs.

        Returns
        -------
        tuple of ChampionStreakV1
            The champions that reached 5, 10, 15 or 20 takedowns (every 5th)
            without falling on this frame's tick in this view
            (viewer.champion_streaks). Empty on frame zero and on most frames.

        Raises
        ------
        ValueError
            A recorded event or scene fails its existing validation.

        Notes
        -----
        Each view's table grows frame by frame from frame zero and is kept for
        the service's life, so a seek, a view switch or a page reload reads the
        same exact value, and each tick is counted once per view. Only ticks
        with a recorded fall need work: Oracle View reads the recorded events;
        Agent POV builds the recipient's visual events (or reuses those the
        timeline marks stored). Plain Python on small integers, no JAX. Runs
        under _champion_streaks_lock; it never takes the service lock, so the
        caller may hold it.
        """
        context = self._context
        with self._champion_streaks_lock:
            table = self._champion_streak_tables.setdefault(
                recipient_global_slot, _ChampionStreakTable()
            )
            scenes: dict[int, _PovScenes] = {}
            class_mechanics: tuple[AuthorizedClassMechanics, ...] | None = None
            while len(table.reached) <= frame_index:
                tick_index = len(table.reached)
                events = self._replay.transitions[tick_index - 1].events
                if not any(event.event_type == "agent_died" for event in events):
                    tick = NO_TAKEDOWNS
                elif recipient_global_slot is None:
                    tick = oracle_tick_takedowns_v1(context, events)
                else:
                    key = (recipient_global_slot, tick_index)
                    cached = self._pov_tick_takedowns.get(key)
                    if cached is None:
                        if class_mechanics is None:
                            class_mechanics = self._researcher_space_for_frame(
                                frame_index=0,
                                selected_global_slot=recipient_global_slot,
                            ).class_mechanics
                        cached = agent_pov_tick_takedowns_v1(
                            context,
                            self._agent_pov_tick_visual_events(
                                recipient_global_slot,
                                tick_index,
                                pov_entry=pov_entry,
                                class_mechanics=class_mechanics,
                                scenes=scenes,
                            ),
                        )
                        self._pov_tick_takedowns[key] = cached
                    tick = cached
                table.counts, reached = advance_takedown_counts_v1(table.counts, tick)
                table.reached.append(reached)
            return champion_streaks_v1(context, table.reached[frame_index])

    def _match_moments(
        self,
        frame_index: int,
        *,
        recipient_global_slot: int | None,
        pov_entry: _PovCacheEntry | None = None,
    ) -> MatchMomentFactsV1:
        """Return one view's Herald Cards history facts for one captured frame.

        Parameters
        ----------
        frame_index : int
            Captured frame index, 0 to the last frame.
        recipient_global_slot : int or None, keyword-only
            None for Oracle View; otherwise the Agent POV recipient's global
            slot. Each view has its own table, so a view switch never mixes
            runs or rescues.
        pov_entry : _PovCacheEntry or None, optional, keyword-only
            The recipient's cached NoSharedObs projection (required for a
            NoSharedObs Agent POV view); None for Oracle View and SharedObs.

        Returns
        -------
        MatchMomentFactsV1
            The frame's ``match_moments`` field (viewer.match_moments):
            fresh_start (frame zero was tick 0 with 0-0; False on frame zero
            itself), the largest lead at the start of this tick or earlier
            (public recorded scores, the same in every view), and this view's
            runs reached and rescues seen on this frame's tick.

        Raises
        ------
        ValueError
            A recorded event or scene fails its existing validation.

        Notes
        -----
        Each view's table grows frame by frame from frame zero and is kept
        for the service's life, so a seek, a view switch or a page reload
        reads the same exact value, and each tick is counted once per view.
        Agent POV needs the recipient's visual events only on a tick with a
        recorded fall (reusing the takedowns the streak count or timeline
        marks stored in _pov_tick_takedowns) or an Oracle rescue. Plain
        Python on small integers, no JAX. Runs under _champion_streaks_lock;
        it never takes the service lock, so the caller may hold it.
        """
        context = self._context
        frames = self._replay.frames
        with self._champion_streaks_lock:
            if self._largest_leads is None:
                leads = [0]
                for frame in frames[:-1]:
                    leads.append(
                        max(
                            leads[-1],
                            score_lead_v1(frame.snapshot.team_deathmatch_scores),
                        )
                    )
                self._largest_leads = tuple(leads)
            table = self._match_moment_tables.setdefault(
                recipient_global_slot, _MatchMomentTable()
            )
            scenes: dict[int, _PovScenes] = {}
            class_mechanics: tuple[AuthorizedClassMechanics, ...] | None = None
            while len(table.reached) <= frame_index:
                tick_index = len(table.reached)
                events = self._replay.transitions[tick_index - 1].events
                fell = any(event.event_type == "agent_died" for event in events)
                oracle_rescues = oracle_tick_rescues_v1(context, events)
                if recipient_global_slot is None:
                    falls = (
                        oracle_tick_takedowns_v1(context, events).falls
                        if fell
                        else frozenset[int]()
                    )
                    rescues = oracle_rescues
                else:
                    key = (recipient_global_slot, tick_index)
                    cached = self._pov_tick_takedowns.get(key) if fell else NO_TAKEDOWNS
                    visual = None
                    if cached is None or oracle_rescues:
                        if class_mechanics is None:
                            class_mechanics = self._researcher_space_for_frame(
                                frame_index=0,
                                selected_global_slot=recipient_global_slot,
                            ).class_mechanics
                        visual = self._agent_pov_tick_visual_events(
                            recipient_global_slot,
                            tick_index,
                            pov_entry=pov_entry,
                            class_mechanics=class_mechanics,
                            scenes=scenes,
                        )
                    if cached is None and visual is not None:
                        cached = agent_pov_tick_takedowns_v1(context, visual)
                        self._pov_tick_takedowns[key] = cached
                    falls = NO_TAKEDOWNS.falls if cached is None else cached.falls
                    rescues = (
                        agent_pov_tick_rescues_v1(
                            context, oracle_rescues, events, visual
                        )
                        if visual is not None
                        else ()
                    )
                table.runs, reached = advance_team_runs_v1(table.runs, context, falls)
                table.reached.append(reached)
                table.rescues.append(rescues)
            first = frames[0]
            return match_moment_facts_v1(
                context,
                fresh_start=frame_index >= 1
                and is_fresh_start_v1(
                    first.simulator_step_count, first.snapshot.team_deathmatch_scores
                ),
                largest_lead_before=self._largest_leads[frame_index],
                reached_runs=table.reached[frame_index],
                rescues=table.rescues[frame_index],
            )

    def current_presentation(self) -> PresentationResourceResultV1:
        """Build a checked display resource from the locked viewer snapshot.

        Returns
        -------
        PresentationResourceResultV1
            Researcher, NoSharedObs, or SharedObs presentation for the current cursor.

        Raises
        ------
        RuntimeError
            The committed frame, selected actor, or source authority is inconsistent.
        ValueError
            An epoch, model, or authorized display contract fails validation.

        Notes
        -----
        This may populate actor projection caches. It does not advance a simulator,
        change the cursor, or expand an actor's allowed information. Every view's
        match summary carries the recorded scores 10 frames back
        (_scores_ten_ticks_ago, the Momentum switch). The frame's
        champion_streaks field is this view's own count (_champion_streaks)
        and its match_moments field this view's own Herald Cards history
        (_match_moments): Oracle View for the researcher view, the
        recipient's Agent POV otherwise; the first request at a late frame
        counts every earlier frame of that view once.
        """
        with self._lock:
            if self._view_mode != "researcher":
                if self._context.execution_information_mode == "no_shared_obs":
                    raw_frame = self._frame
                    if type(raw_frame) is not ActorPovReplayViewerFrameV1:
                        raise RuntimeError(
                            "NoSharedObs replay view does not hold its committed "
                            "recipient frame"
                        )
                    self._validate_committed_presentation_snapshot(
                        raw_frame,
                        oracle=False,
                    )
                    if self._pov_global_slot is None:
                        raise RuntimeError(
                            "NoSharedObs replay view has no fixed POV recipient"
                        )
                    entry = self._pov_cache.get(self._pov_global_slot)
                    if entry is None:
                        raise RuntimeError(
                            "committed NoSharedObs replay frame has no POV cache entry"
                        )
                    if (
                        entry.projection_index.content.selected_global_slot
                        != self._pov_global_slot
                    ):
                        raise RuntimeError(
                            "committed NoSharedObs cache entry does not join the "
                            "fixed POV recipient"
                        )
                    frame = build_replay_no_shared_obs_authorized_presentation_v1(
                        entry.projection_index,
                        raw_frame,
                        global_context=self._context,
                        current_global_frame=self._replay.frames[
                            raw_frame.cursor.frame_index
                        ],
                        previous_global_frame=(
                            None
                            if raw_frame.cursor.frame_index == 0
                            else self._replay.frames[raw_frame.cursor.frame_index - 1]
                        ),
                        public_catalog=self._context.static_mechanics_catalog,
                        source_authority_epoch=raw_frame.revision,
                        incoming_visual_events=(
                            None
                            if raw_frame.cursor.frame_index == 0
                            else build_visual_event_batch_v2(
                                cast(
                                    EvaluationTransitionViewV1,
                                    self._transition_view(raw_frame.cursor.frame_index),
                                )
                            )
                        ),
                        researcher_space=self._researcher_space_for_frame(
                            frame_index=raw_frame.cursor.frame_index,
                            selected_global_slot=self._pov_global_slot,
                        ),
                        exact_actor_input_export_available=(
                            entry.exact_actor_input_export_available
                        ),
                        scores_ten_ticks_ago=self._scores_ten_ticks_ago(
                            raw_frame.cursor.frame_index
                        ),
                        champion_streaks=self._champion_streaks(
                            raw_frame.cursor.frame_index,
                            recipient_global_slot=self._pov_global_slot,
                            pov_entry=entry,
                        ),
                        match_honors=self._match_honors(
                            raw_frame.cursor.frame_index,
                            team_id=self._team_of(self._pov_global_slot),
                        ),
                        match_moments=self._match_moments(
                            raw_frame.cursor.frame_index,
                            recipient_global_slot=self._pov_global_slot,
                            pov_entry=entry,
                        ),
                    )
                    return PresentationResourceResultV1(
                        outcome="response",
                        payload=frame,
                    )
                raw_frame = self._frame
                if type(raw_frame) is not SharedObsAgentPovReplayViewerFrameV1:
                    raise RuntimeError(
                        "SharedObs replay view does not hold its committed "
                        "private recipient frame"
                    )
                self._validate_committed_presentation_snapshot(
                    raw_frame,
                    oracle=False,
                )
                if self._pov_global_slot is None:
                    raise RuntimeError(
                        "SharedObs replay view has no fixed POV recipient"
                    )
                frame_index = raw_frame.cursor.frame_index
                current_recipient, current_nonrecipient = (
                    self._shared_authority_sources(
                        frame_index=frame_index,
                        recipient_global_slot=self._pov_global_slot,
                    )
                )
                if frame_index == 0:
                    previous_recipient = None
                    previous_nonrecipient: tuple[
                        SharedObsSourceMaterialProjection, ...
                    ] = ()
                    incoming_transition = None
                else:
                    previous_recipient, previous_nonrecipient = (
                        self._shared_authority_sources(
                            frame_index=frame_index - 1,
                            recipient_global_slot=self._pov_global_slot,
                        )
                    )
                    incoming_transition = self._replay.transitions[frame_index - 1]
                outgoing_transition = (
                    None
                    if frame_index == len(self._replay.transitions)
                    else self._replay.transitions[frame_index]
                )
                frame = build_replay_shared_obs_authorized_presentation_v1(
                    raw_frame,
                    global_context=self._context,
                    current_global_frame=self._replay.frames[frame_index],
                    previous_global_frame=(
                        None
                        if frame_index == 0
                        else self._replay.frames[frame_index - 1]
                    ),
                    public_catalog=self._context.static_mechanics_catalog,
                    source_authority_epoch=raw_frame.revision,
                    authorized_recipient_global_slot=self._pov_global_slot,
                    current_recipient_source_material=current_recipient,
                    current_active_nonrecipient_source_material=(current_nonrecipient),
                    previous_recipient_source_material=previous_recipient,
                    previous_active_nonrecipient_source_material=(
                        previous_nonrecipient
                    ),
                    incoming_visual_events=(
                        None
                        if frame_index == 0
                        else build_visual_event_batch_v2(
                            cast(
                                EvaluationTransitionViewV1,
                                self._transition_view(frame_index),
                            )
                        )
                    ),
                    incoming_transition=incoming_transition,
                    outgoing_transition=outgoing_transition,
                    researcher_space=self._researcher_space_for_frame(
                        frame_index=frame_index,
                        selected_global_slot=self._pov_global_slot,
                    ),
                    scores_ten_ticks_ago=self._scores_ten_ticks_ago(frame_index),
                    champion_streaks=self._champion_streaks(
                        frame_index, recipient_global_slot=self._pov_global_slot
                    ),
                    match_honors=self._match_honors(
                        frame_index, team_id=self._team_of(self._pov_global_slot)
                    ),
                    match_moments=self._match_moments(
                        frame_index, recipient_global_slot=self._pov_global_slot
                    ),
                )
                return PresentationResourceResultV1(outcome="response", payload=frame)

            raw_frame = self._frame
            if type(raw_frame) is not ResearcherReplayViewerFrameV1:
                raise RuntimeError(
                    "Oracle replay view does not hold its committed researcher frame"
                )
            self._validate_committed_presentation_snapshot(raw_frame, oracle=True)
            source_frame_index = raw_frame.cursor.frame_index
            current_frame = self._replay.frames[source_frame_index]
            incoming_transition = (
                None
                if source_frame_index == 0
                else self._replay.transitions[source_frame_index - 1]
            )
            outgoing_transition = (
                None
                if source_frame_index == len(self._replay.transitions)
                else self._replay.transitions[source_frame_index]
            )
            frame = build_replay_oracle_authorized_presentation_v1(
                self._context,
                current_frame,
                raw_frame,
                source_authority_epoch=raw_frame.revision,
                selected_internal_slot=self._inspection_global_slot,
                incoming_transition=incoming_transition,
                outgoing_transition=outgoing_transition,
                previous_frame=(
                    None
                    if source_frame_index == 0
                    else self._replay.frames[source_frame_index - 1]
                ),
                scores_ten_ticks_ago=self._scores_ten_ticks_ago(source_frame_index),
                champion_streaks=self._champion_streaks(
                    source_frame_index, recipient_global_slot=None
                ),
                match_honors=self._match_honors(source_frame_index, team_id=None),
                match_moments=self._match_moments(
                    source_frame_index, recipient_global_slot=None
                ),
            )
            return PresentationResourceResultV1(
                outcome="response",
                payload=frame,
            )

    def _team_of(self, global_slot: int) -> Literal[1, 2]:
        """Return the recorded team (1 or 2) of a global slot 0-9. Pure."""
        return cast(
            "Literal[1, 2]", self._context.roster[global_slot].configured_team_id
        )

    def _match_honors(
        self, frame_index: int, *, team_id: Literal[1, 2] | None
    ) -> MatchHonorsV1 | None:
        """Return one view's game-over Honors for a captured frame, or None.

        Parameters
        ----------
        frame_index : int
            Captured frame index, 0 to the last frame.
        team_id : {1, 2} or None, keyword-only
            None for Oracle View (every champion); the Agent POV recipient's
            team otherwise (only that team's own champions).

        Returns
        -------
        MatchHonorsV1 or None
            Honors only on the last frame of a Team Deathmatch replay whose
            last transition records the game's completion
            (viewer.match_summary.match_honors_v1 over the whole replay);
            None on every other frame and for an unfinished replay.

        Notes
        -----
        The first call on the completion frame adds up every transition's
        totals once (match_transition_totals_v1; a match summary with kill
        credit is built only for transitions with a fall) and keeps them for
        the service's life under _honors_lock, which never takes the service
        lock. Reads the immutable replay only; no file or simulator work.
        """
        transitions = self._replay.transitions
        if (
            self._context.resolved_env_config.task_mode != 1
            or not transitions
            or frame_index != len(transitions)
            or not any(
                getattr(event, "event_type", None) == "team_deathmatch_completed"
                for event in transitions[-1].events
            )
        ):
            return None
        with self._honors_lock:
            if self._honors_totals is None:
                totals = MatchTotalsV1()
                frames = self._replay.frames
                for index, transition in enumerate(transitions):
                    events = transition.events
                    summary = (
                        build_match_summary_v1(
                            self._context,
                            frames[index + 1],
                            events,
                            transition_start_frame=frames[index],
                        )
                        if any(event.event_type == "agent_died" for event in events)
                        else None
                    )
                    totals = totals.plus(
                        match_transition_totals_v1(self._context, summary, events)
                    )
                self._honors_totals = totals
            totals = self._honors_totals
        return match_honors_v1(self._context, totals, team_id=team_id)

    def _scores_ten_ticks_ago(self, frame_index: int) -> tuple[int, ...]:
        """Return the recorded scores ``MOMENTUM_TICKS`` (10) frames back.

        frame_index is a captured frame index (0 to the last frame). The result
        is (Team A, Team B) from the recorded frame ``max(0, frame_index -
        MOMENTUM_TICKS)``, so a younger game reads frame zero and frame zero
        reads its own scores. Every view's match summary gets it for the
        scorebug's Momentum switch; scores are public match facts. Reads the
        immutable replay only; no lock, file or simulator work.
        """
        earlier = self._replay.frames[max(0, frame_index - MOMENTUM_TICKS)]
        return tuple(earlier.snapshot.team_deathmatch_scores)

    def _validate_committed_presentation_snapshot(
        self,
        raw_frame: ReplayViewerFrameV1,
        *,
        oracle: bool,
    ) -> None:
        """Require the raw envelope to equal this locked service snapshot."""
        cursor = raw_frame.cursor
        if type(cursor) is not ReplayCursorV1:
            raise RuntimeError("committed replay frame has an invalid cursor root")
        if (
            type(raw_frame.viewer_session_id) is not str
            or raw_frame.viewer_session_id != self._viewer_session_id
            or type(raw_frame.revision) is not int
            or raw_frame.revision != self._revision
            or type(cursor.frame_index) is not int
            or cursor.frame_index != self._frame_index
            or type(cursor.final_frame_index) is not int
            or cursor.final_frame_index != len(self._replay.transitions)
        ):
            raise RuntimeError("committed replay frame does not join service state")
        if oracle and (
            type(cursor.cursor_generation) is not int
            or cursor.cursor_generation != self._cursor_generation
            or type(cursor.choreography_generation) is not int
            or cursor.choreography_generation != self._choreography_generation
        ):
            raise RuntimeError("committed Oracle cursor generations are stale")
        if not oracle:
            if type(raw_frame) not in (
                ActorPovReplayViewerFrameV1,
                SharedObsAgentPovReplayViewerFrameV1,
            ):
                raise RuntimeError(
                    "committed Agent replay artifact facts use the wrong root"
                )
            agent_raw_frame = cast(
                ActorPovReplayViewerFrameV1 | SharedObsAgentPovReplayViewerFrameV1,
                raw_frame,
            )
            if (
                type(agent_raw_frame.artifact_facts) is not ReplayArtifactFactsV1
                or agent_raw_frame.artifact_facts != self._artifact_facts
            ):
                raise RuntimeError(
                    "committed Agent replay artifact facts do not join service state"
                )
        if not oracle and self._context.execution_information_mode == "shared_obs":
            if type(raw_frame) is not SharedObsAgentPovReplayViewerFrameV1:
                raise RuntimeError(
                    "committed SharedObs frame has the wrong product root"
                )
            if self._pov_global_slot is None:
                raise RuntimeError("committed SharedObs frame has no fixed recipient")
            timeline = self._shared_timeline_cache.get(self._pov_global_slot)
            if timeline is None:
                raise RuntimeError(
                    "committed SharedObs frame has no private timeline cache entry"
                )
            roster = self._context.roster[self._pov_global_slot]
            episode_id = self._context.identity.episode_id
            prefix = f"{episode_id}:shared-obs-visual-union:{roster.public_agent_id}"
            expected_incoming_id = (
                None
                if self._frame_index == 0
                else f"{prefix}:transition:{self._frame_index - 1}"
            )
            source_frame = self._replay.frames[self._frame_index]
            if (
                type(raw_frame.artifact_summary)
                is not SharedObsAgentPovReplayArtifactSummaryV1
                or raw_frame.artifact_summary != timeline.artifact_summary
                or raw_frame.timeline_id != timeline.timeline_id
                or raw_frame.public_agent_id != roster.public_agent_id
                or raw_frame.recipient_frame_id != f"{prefix}:frame:{self._frame_index}"
                or raw_frame.simulator_step_count != source_frame.simulator_step_count
                or raw_frame.incoming_recipient_transition_id != expected_incoming_id
                or type(raw_frame.completion) is not ActorPovReplayCompletionBadgeV1
                or raw_frame.completion != self._shared_completion
                or raw_frame.artifact_facts != self._artifact_facts
            ):
                raise RuntimeError(
                    "committed SharedObs transport identity does not join service "
                    "authority"
                )
        if oracle:
            if type(raw_frame) is not ResearcherReplayViewerFrameV1:
                raise RuntimeError("committed Oracle frame has the wrong product root")
            recorded_scale = (
                self._context.resolved_env_config.ordinary_movement_distance_scale
            )
            source_frame = self._replay.frames[self._frame_index]
            incoming = self._transition_view(self._frame_index)
            expected_projection = build_researcher_analyzer_projection_v2(
                self._context,
                source_frame,
                transition_view=incoming,
                presentation=EvaluationScenePresentationStateV1(
                    controlled_global_slot=self._reference_global_slot,
                    selected_global_slot=self._inspection_global_slot,
                    armed_lane=self._armed_lane,
                    show_ranges=self._show_ranges,
                ),
                status_source_evidence_state=self._status_index.state_for_frame(
                    self._frame_index
                ),
            )
            if (
                type(raw_frame.artifact_summary) is not ReplayArtifactSummaryV1
                or raw_frame.artifact_summary is not self._artifact_summary
                or type(raw_frame.timeline_id) is not str
                or raw_frame.timeline_id != self._researcher_timeline.timeline_id
                or type(raw_frame.recorded_ordinary_movement_distance_scale)
                is not float
                or raw_frame.recorded_ordinary_movement_distance_scale != recorded_scale
                or type(raw_frame.show_ranges) is not bool
                or raw_frame.show_ranges != self._show_ranges
                or type(raw_frame.projection) is not type(expected_projection)
                or raw_frame.projection != expected_projection
            ):
                raise RuntimeError(
                    "committed Oracle provenance does not join service authority"
                )

    def current_timeline(self) -> ReplayTimelineV1:
        """Read the timeline allowed for the current view and actor.

        Returns
        -------
        ReplayTimelineV1
            Researcher timeline or the selected actor's cached timeline. SharedObs
            timelines use recipient-local identities and completion facts.

        Raises
        ------
        RuntimeError
            POV mode has no selected actor.

        Notes
        -----
        The service lock covers any first-time actor timeline construction.
        """
        with self._lock:
            if self._view_mode == "researcher":
                return self._researcher_timeline
            if self._pov_global_slot is None:
                raise RuntimeError("POV replay view has no selected actor")
            if self._context.execution_information_mode == "no_shared_obs":
                return self._pov_entry(
                    self._pov_global_slot,
                    cache=self._pov_cache,
                ).timeline
            return self._shared_timeline(
                self._pov_global_slot,
                cache=self._shared_timeline_cache,
            )

    def apply_command(
        self,
        request: ReplayCommandRequestV1,
    ) -> ReplayServiceCommandResultV1:
        """Apply a checked viewer command without changing recorded replay data.

        Parameters
        ----------
        request : ReplayCommandRequestV1
            Exact validated root with client ID, command ID, base revision, and a
            supported cursor or display command.

        Returns
        -------
        ReplayServiceCommandResultV1
            New frame, duplicate response, or typed rejection. Exit requests are
            returned for the HTTP host to act on after writing the response.

        Raises
        ------
        TypeError
            The request is not the exact supported model.

        Notes
        -----
        View switches move the selection like this. Entering Agent POV with an
        Oracle selection makes that champion the POV recipient. Leaving Agent
        POV clears the Oracle selection and any armed lane, so the Oracle frame
        has no selected champion, exactly like a fresh Oracle render; the last
        POV recipient is kept only as the recipient for the next POV entry.

        The lock covers validation and installation. The latest 256 command IDs are
        remembered; exact repeats while cached do not apply twice. Stale revisions
        and reused IDs with different contents are rejected. Projection caches may
        be built, but no simulator step or replay write occurs. Unexpected internal
        failures can fence further commands and propagate.
        """
        if type(request) is not ReplayCommandRequestV1:
            raise TypeError("request must be the exact ReplayCommandRequestV1 root")
        command_key = (request.client_id, request.command_id)
        fingerprint = request.model_dump_json()
        with self._lock:
            previous = self._command_records.get(command_key)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    return self._error_result(
                        "command_id_conflict",
                        "command_id_conflict",
                        "This client reused a command_id for a different request.",
                    )
                result = ReplayServiceCommandResultV1(
                    outcome="response",
                    payload=ReplayCommandResponseV1(
                        result="duplicate",
                        frame=self._frame,
                        notice="Command already processed; current frame returned.",
                        animate_incoming=False,
                    ),
                    shutdown_requested=previous.shutdown_requested,
                )
                records = self._command_records.copy()
                records.move_to_end(command_key)
                self._command_records = records
                return result
            if self._faulted:
                return self._record_error(
                    command_key,
                    fingerprint,
                    "service_faulted",
                    "internal_error",
                    "The replay viewer entered a safe fault state; restart it.",
                )
            if self._shutting_down:
                return self._record_error(
                    command_key,
                    fingerprint,
                    "server_shutting_down",
                    "server_shutting_down",
                    "The replay viewer is shutting down; the command was not applied.",
                )
            if request.base_revision != self._revision:
                return self._record_error(
                    command_key,
                    fingerprint,
                    "stale_revision",
                    "stale_revision",
                    "The replay cursor advanced; the latest frame is attached.",
                )

            command = request.command
            frame_index = self._frame_index
            cursor_generation = self._cursor_generation
            choreography_generation = self._choreography_generation
            view_mode = self._view_mode
            inspection_global_slot = self._inspection_global_slot
            armed_lane = self._armed_lane
            pov_global_slot = self._pov_global_slot
            preset = self._preset
            show_ranges = self._show_ranges
            verbose = self._verbose
            pov_cache = self._pov_cache.copy()
            shared_timeline_cache = self._shared_timeline_cache.copy()
            changed = False
            animate_incoming = False
            shutdown_requested = False
            notice: str | None = None

            if type(command) is ReplayAbsoluteSeekCommandV1:
                if (
                    command.frame_index
                    > self._artifact_summary.recorded_transition_count
                ):
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "invalid_cursor",
                        "invalid_cursor",
                        "The requested frame is outside the captured replay prefix.",
                    )
                frame_index = command.frame_index
                cursor_generation += 1
                changed = True
            elif type(command) is ReplayFirstFrameCommandV1:
                frame_index = 0
                cursor_generation += 1
                changed = True
            elif type(command) is ReplayLastFrameCommandV1:
                frame_index = self._artifact_summary.recorded_transition_count
                cursor_generation += 1
                changed = True
            elif type(command) is ReplayPreviousFrameCommandV1:
                if frame_index == 0:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "invalid_cursor",
                        "invalid_cursor",
                        "The replay cursor cannot move before frame zero.",
                    )
                frame_index -= 1
                cursor_generation += 1
                changed = True
            elif type(command) is ReplayNextFrameCommandV1:
                if frame_index == self._artifact_summary.recorded_transition_count:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "invalid_cursor",
                        "invalid_cursor",
                        "The replay cursor cannot move past the captured prefix.",
                    )
                frame_index += 1
                cursor_generation += 1
                choreography_generation += 1
                changed = True
                animate_incoming = True
            elif type(command) is ReplaySelectAgentCommandV1:
                if view_mode != "researcher":
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "Researcher selection is unavailable in POV replay.",
                    )
                requested_selection: int | None = command.selected_global_slot
                try:
                    self._require_active_or_none(
                        requested_selection,
                        name="selected_global_slot",
                    )
                except ValueError:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "The requested researcher selection is unavailable.",
                    )
                selection_changed = requested_selection != inspection_global_slot
                pov_changed = (
                    requested_selection is not None
                    and requested_selection != pov_global_slot
                )
                if selection_changed or pov_changed:
                    inspection_global_slot = requested_selection
                    if requested_selection is not None:
                        pov_global_slot = requested_selection
                    armed_lane = None
                    changed = True
            elif type(command) is ReplaySetViewCommandV1:
                if command.view_mode == "pov" and pov_global_slot is None:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "No configured-active actor is available for POV replay.",
                    )
                if command.view_mode != view_mode:
                    if command.view_mode == "pov":
                        if (
                            inspection_global_slot is not None
                            and inspection_global_slot != pov_global_slot
                        ):
                            pov_global_slot = inspection_global_slot
                    else:
                        # Leaving Agent POV clears the Oracle selection (the
                        # Reference chip and brackets), so Oracle -> POV ->
                        # Oracle draws exactly what a fresh Oracle render draws.
                        # The POV actor is kept only as the next POV recipient.
                        inspection_global_slot = None
                        armed_lane = None
                    view_mode = command.view_mode
                    changed = True
            elif type(command) is ReplaySetPovActorCommandV1:
                requested_slot: int | None = command.global_slot
                try:
                    if command.presentation_key is not None:
                        if view_mode != "pov":
                            raise ValueError(
                                "presentation-key actor switching requires POV"
                            )
                        requested_slot = self._visible_pov_actor_slot(
                            command.presentation_key
                        )
                    self._require_active_or_none(
                        requested_slot,
                        name="pov_global_slot",
                    )
                    if requested_slot is None:  # pragma: no cover - model invariant.
                        raise ValueError("POV actor reference is absent")
                except ValueError:
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "The requested POV actor is unavailable.",
                    )
                if (
                    requested_slot != pov_global_slot
                    or requested_slot != inspection_global_slot
                ):
                    pov_global_slot = requested_slot
                    inspection_global_slot = requested_slot
                    armed_lane = None
                    changed = True
            elif type(command) is ReplaySetPresetCommandV1:
                # V1 compatibility command. All legacy preset requests
                # canonicalize to the single product Analysis presentation.
                preset = "analysis"
            elif type(command) is ReplaySetRangesCommandV1:
                if view_mode != "researcher":
                    return self._record_error(
                        command_key,
                        fingerprint,
                        "audience_unavailable",
                        "audience_unavailable",
                        "Researcher ranges are unavailable in POV replay.",
                    )
                if command.show_ranges != show_ranges:
                    show_ranges = command.show_ranges
                    changed = True
            elif type(command) is ReplaySetVerbosityCommandV1:
                # Retained as a V1 compatibility command. Verbose presentation
                # is no longer a product mode, so both values are a fixed no-op.
                verbose = False
            elif type(command) is ReplayExitCommandV1:
                shutdown_requested = True
            else:  # pragma: no cover - exact discriminated union is exhaustive.
                raise TypeError("unsupported replay command root")

            candidate_revision = self._revision + int(changed)
            try:
                candidate_frame = (
                    self._build_frame(
                        revision=candidate_revision,
                        frame_index=frame_index,
                        cursor_generation=cursor_generation,
                        choreography_generation=choreography_generation,
                        view_mode=view_mode,
                        inspection_global_slot=inspection_global_slot,
                        armed_lane=armed_lane,
                        pov_global_slot=pov_global_slot,
                        preset=preset,
                        show_ranges=show_ranges,
                        verbose=verbose,
                        pov_cache=pov_cache,
                        shared_timeline_cache=shared_timeline_cache,
                    )
                    if changed
                    else self._frame
                )
                result_kind: Literal["applied", "no_op", "shutdown_scheduled"] = (
                    "shutdown_scheduled"
                    if shutdown_requested
                    else "applied"
                    if changed
                    else "no_op"
                )
                response = ReplayCommandResponseV1(
                    result=result_kind,
                    frame=candidate_frame,
                    notice=notice,
                    animate_incoming=animate_incoming,
                )
                records = self._command_records.copy()
                self._remember_in(records, command_key, fingerprint, shutdown_requested)
                result = ReplayServiceCommandResultV1(
                    outcome="response",
                    payload=response,
                    shutdown_requested=shutdown_requested,
                )
            except Exception:
                records = self._command_records.copy()
                self._remember_in(records, command_key, fingerprint, False)
                self._faulted = True
                self._command_records = records
                raise

            self._revision = candidate_revision
            self._frame_index = frame_index
            self._cursor_generation = cursor_generation
            self._choreography_generation = choreography_generation
            self._view_mode = view_mode
            self._inspection_global_slot = inspection_global_slot
            self._armed_lane = armed_lane
            self._pov_global_slot = pov_global_slot
            self._preset = preset
            self._show_ranges = show_ranges
            self._verbose = verbose
            self._pov_cache = pov_cache
            self._shared_timeline_cache = shared_timeline_cache
            if shutdown_requested:
                self._shutting_down = True
            self._frame = candidate_frame
            self._command_records = records
            return result

    def _build_frame(
        self,
        *,
        revision: int,
        frame_index: int,
        cursor_generation: int,
        choreography_generation: int,
        view_mode: ReplayViewModeV1,
        inspection_global_slot: int | None,
        armed_lane: Literal[0, 1] | None,
        pov_global_slot: int | None,
        preset: ReplayPresetV1,
        show_ranges: bool,
        verbose: bool,
        pov_cache: dict[int, _PovCacheEntry],
        shared_timeline_cache: dict[int, SharedObsAgentPovReplayTimelineV1],
    ) -> ReplayViewerFrameV1:
        """Build and validate one proposed cursor/view frame using caller-owned
        caches.
        """
        cursor = ReplayCursorV1(
            frame_index=frame_index,
            final_frame_index=self._artifact_summary.recorded_transition_count,
            cursor_generation=cursor_generation,
            choreography_generation=choreography_generation,
        )
        frame = self._replay.frames[frame_index]
        if view_mode == "researcher":
            incoming = self._transition_view(frame_index)
            projection = build_researcher_analyzer_projection_v2(
                self._context,
                frame,
                transition_view=incoming,
                presentation=EvaluationScenePresentationStateV1(
                    controlled_global_slot=self._reference_global_slot,
                    selected_global_slot=inspection_global_slot,
                    armed_lane=armed_lane,
                    show_ranges=show_ranges,
                ),
                status_source_evidence_state=self._status_index.state_for_frame(
                    frame_index
                ),
            )
            return ResearcherReplayViewerFrameV1(
                viewer_session_id=self._viewer_session_id,
                revision=revision,
                artifact_summary=self._artifact_summary,
                timeline_id=self._researcher_timeline.timeline_id,
                cursor=cursor,
                preset=preset,
                verbose=False,
                frame_id=frame.frame_id,
                simulator_step_count=frame.simulator_step_count,
                incoming_transition_index=(
                    None if incoming is None else incoming.transition.transition_index
                ),
                incoming_transition_id=(
                    None if incoming is None else incoming.transition.transition_id
                ),
                completion=self._completion,
                processing=self._processing,
                show_ranges=show_ranges,
                recorded_ordinary_movement_distance_scale=(
                    self._context.resolved_env_config.ordinary_movement_distance_scale
                ),
                projection=projection,
            )
        if pov_global_slot is None:
            raise ValueError("POV replay view requires a configured-active actor")
        roster = self._context.roster[pov_global_slot]
        if self._context.execution_information_mode == "no_shared_obs":
            incoming = self._transition_view(frame_index)
            entry = self._pov_entry(pov_global_slot, cache=pov_cache)
            projection = build_actor_pov_analyzer_projection_v1(
                entry.projection_index,
                frame_index=frame_index,
            )
            pov_frame = entry.content.frames[frame_index]
            return ActorPovReplayViewerFrameV1(
                viewer_session_id=self._viewer_session_id,
                revision=revision,
                artifact_summary=self._pov_artifact_summary,
                timeline_id=entry.timeline.timeline_id,
                cursor=cursor,
                preset=preset,
                verbose=False,
                pov_global_slot=pov_global_slot,
                public_agent_id=roster.public_agent_id,
                pov_frame_id=pov_frame.pov_frame_id,
                simulator_step_count=pov_frame.simulator_step_count,
                incoming_pov_transition_id=projection.incoming_transition_id,
                completion=entry.completion,
                processing_disclosure=ActorPovProcessingDisclosureV1(),
                artifact_facts=self._artifact_facts,
                projection=projection,
            )
        timeline = self._shared_timeline(
            pov_global_slot,
            cache=shared_timeline_cache,
        )
        prefix = (
            f"{self._context.identity.episode_id}:shared-obs-visual-union:"
            f"{roster.public_agent_id}"
        )
        return SharedObsAgentPovReplayViewerFrameV1(
            schema_version=1,
            frame_kind="shared_obs_agent_pov_replay_viewer",
            viewer_session_id=self._viewer_session_id,
            revision=revision,
            artifact_summary=timeline.artifact_summary,
            timeline_id=timeline.timeline_id,
            cursor=cursor,
            preset="analysis",
            verbose=False,
            view_mode="pov",
            public_agent_id=roster.public_agent_id,
            recipient_frame_id=f"{prefix}:frame:{frame_index}",
            simulator_step_count=frame.simulator_step_count,
            incoming_recipient_transition_id=(
                None if frame_index == 0 else f"{prefix}:transition:{frame_index - 1}"
            ),
            completion=self._shared_completion,
            artifact_facts=self._artifact_facts,
        )

    def _build_researcher_timeline(self) -> ResearcherReplayTimelineV1:
        """Build ordered frame and incoming-event rows with one final endpoint
        marker.
        """
        endpoint = _endpoint_kind(self._completion)
        final = len(self._replay.frames) - 1
        rows = tuple(
            ResearcherReplayTimelineRowV1(
                frame_index=index,
                frame_id=frame.frame_id,
                simulator_step_count=frame.simulator_step_count,
                incoming_transition_id=(
                    None
                    if index == 0
                    else self._replay.transitions[index - 1].transition_id
                ),
                incoming_event_count=(
                    0 if index == 0 else len(self._replay.transitions[index - 1].events)
                ),
                endpoint_kind=endpoint if index == final else "none",
            )
            for index, frame in enumerate(self._replay.frames)
        )
        return ResearcherReplayTimelineV1(
            timeline_id=f"{self._replay.artifact_id}:timeline:researcher",
            artifact_summary=self._artifact_summary,
            final_frame_index=final,
            completion=self._completion,
            rows=rows,
        )

    def _pov_entry(
        self,
        global_slot: int,
        *,
        cache: dict[int, _PovCacheEntry],
    ) -> _PovCacheEntry:
        """Get or build one actor's immutable visual projection and timeline.

        Populate the supplied cache only after construction. Preserve whether the
        replay version supports exact actor-input export or only a visual projection:
        replay V4 exports POV V3, replay V3 exports POV V2, historical projection-V1
        replays export POV V1, and replay V2 (or projection V2) is visual only.
        """
        cached = cache.get(global_slot)
        if cached is not None:
            return cached
        if type(self._replay) is ReplayArtifactV4:
            content = export_actor_pov_replay_v3(
                self._replay, global_slot=global_slot
            ).content
            exact_actor_input_export_available = True
        elif type(self._replay) is ReplayArtifactV3:
            content = export_actor_pov_replay_v2(
                self._replay, global_slot=global_slot
            ).content
            exact_actor_input_export_available = True
        elif (
            type(self._replay) is ReplayArtifactV2
            or self._context.actor_projection == NO_SHARED_OBS_ACTOR_PROJECTION_V2
        ):
            content = build_replay_no_shared_obs_visual_content_v1(
                self._replay,
                global_slot=global_slot,
            )
            exact_actor_input_export_available = False
        else:
            artifact = export_actor_pov_replay_v1(
                cast(ReplayArtifactV1, self._replay),
                global_slot=global_slot,
            )
            content = artifact.content
            exact_actor_input_export_available = True
        projection_index = build_actor_pov_projection_index_v1(content)
        completion = _pov_completion_badge(content)
        endpoint = _endpoint_kind(completion)
        final = len(content.frames) - 1
        rows = tuple(
            ActorPovReplayTimelineRowV1(
                frame_index=index,
                pov_frame_id=frame.pov_frame_id,
                simulator_step_count=frame.simulator_step_count,
                incoming_pov_transition_id=(
                    None
                    if index == 0
                    else content.transitions[index - 1].pov_transition_id
                ),
                incoming_cue_count=(
                    0 if index == 0 else len(content.transitions[index - 1].cues)
                ),
                endpoint_kind=endpoint if index == final else "none",
            )
            for index, frame in enumerate(content.frames)
        )
        timeline = ActorPovReplayTimelineV1(
            timeline_id=(
                f"{self._replay.artifact_id}:timeline:actor-pov:"
                f"{content.public_agent_id}"
            ),
            artifact_summary=self._pov_artifact_summary,
            final_frame_index=final,
            pov_global_slot=global_slot,
            public_agent_id=content.public_agent_id,
            completion=completion,
            rows=rows,
        )
        entry = _PovCacheEntry(
            content=content,
            completion=completion,
            projection_index=projection_index,
            timeline=timeline,
            exact_actor_input_export_available=(exact_actor_input_export_available),
        )
        cache[global_slot] = entry
        return entry

    def _shared_authority_sources(
        self,
        *,
        frame_index: int,
        recipient_global_slot: int,
    ) -> tuple[
        SharedObsSourceMaterialProjection,
        tuple[SharedObsSourceMaterialProjection, ...],
    ]:
        """Return one same-epoch fixed-recipient authority source set.

        Each (frame, slot) source is built once per replay and then shared by
        every viewer and request: the replay's frames never change, and a
        source row depends only on the context, the frame and its own slot,
        never on who is watching. Building them again for each new Agent POV
        agent made its timeline marks take seconds. A race between threads can
        only build an equal row twice; no lock is needed.
        """
        if self._context.execution_information_mode != "shared_obs":
            raise RuntimeError("Shared authority sources require a SharedObs replay")
        if type(frame_index) is not int or not (
            0 <= frame_index < len(self._replay.frames)
        ):
            raise RuntimeError("Shared authority source frame is outside the replay")
        if (
            type(recipient_global_slot) is not int
            or recipient_global_slot not in self._active_slots
        ):
            raise RuntimeError("Shared authority recipient is not configured active")
        frame = self._replay.frames[frame_index]

        def build(global_slot: int) -> SharedObsSourceMaterialProjection:
            """Project one active slot's source material at the already selected
            frame, once per replay.
            """
            key = (frame_index, global_slot)
            source = self._source_materials.get(key)
            if source is None:
                source = build_shared_obs_authority_source_material_projection_v1(
                    self._context,
                    frame,
                    selected_global_slot=global_slot,
                )
                self._source_materials[key] = source
            return source

        recipient = build(recipient_global_slot)
        contributors = tuple(
            build(global_slot)
            for global_slot in self._active_slots
            if global_slot != recipient_global_slot
        )
        return recipient, contributors

    def _shared_timeline(
        self,
        global_slot: int,
        *,
        cache: dict[int, SharedObsAgentPovReplayTimelineV1],
    ) -> SharedObsAgentPovReplayTimelineV1:
        """Get or cache a recipient-local SharedObs timeline with no global event
        counts.
        """
        cached = cache.get(global_slot)
        if cached is not None:
            return cached
        roster = self._context.roster[global_slot]
        summary = self._shared_artifact_summary(roster.public_agent_id)
        endpoint = _endpoint_kind(self._shared_completion)
        final = len(self._replay.frames) - 1
        prefix = (
            f"{self._context.identity.episode_id}:shared-obs-visual-union:"
            f"{roster.public_agent_id}"
        )
        rows = tuple(
            SharedObsAgentPovReplayTimelineRowV1(
                frame_index=index,
                recipient_frame_id=f"{prefix}:frame:{index}",
                simulator_step_count=frame.simulator_step_count,
                incoming_recipient_transition_id=(
                    None if index == 0 else f"{prefix}:transition:{index - 1}"
                ),
                endpoint_kind=endpoint if index == final else "none",
            )
            for index, frame in enumerate(self._replay.frames)
        )
        timeline = SharedObsAgentPovReplayTimelineV1(
            schema_version=1,
            timeline_kind="shared_obs_agent_pov",
            timeline_id=f"{prefix}:timeline",
            artifact_summary=summary,
            final_frame_index=final,
            completion=self._shared_completion,
            rows=rows,
        )
        cache[global_slot] = timeline
        return timeline

    def _shared_artifact_summary(
        self,
        public_agent_id: str,
    ) -> SharedObsAgentPovReplayArtifactSummaryV1:
        """Build recipient-local replay identity and capture counts without global
        hashes.
        """
        episode_id = self._context.identity.episode_id
        return SharedObsAgentPovReplayArtifactSummaryV1(
            schema_version=1,
            recipient_replay_id=(
                f"{episode_id}:shared-obs-visual-union:{public_agent_id}:replay"
            ),
            episode_id=episode_id,
            public_agent_id=public_agent_id,
            expected_transition_count=self._replay.header.expected_transition_count,
            captured_transition_count=len(self._replay.transitions),
            captured_frame_count=len(self._replay.frames),
        )

    def _transition_view(self, frame_index: int) -> EvaluationTransitionViewV1 | None:
        """Join the selected frame to its incoming transition; frame zero returns
        None.
        """
        if frame_index == 0:
            return None
        return EvaluationTransitionViewV1(
            context=self._context,
            start_frame=self._replay.frames[frame_index - 1],
            transition=self._replay.transitions[frame_index - 1],
            successor_frame=self._replay.frames[frame_index],
        )

    def _researcher_space_for_frame(
        self,
        *,
        frame_index: int,
        selected_global_slot: int,
    ) -> ReplayResearcherSpaceV1:
        """Build geometry-free global researcher facts for one selected frame and
        agent.
        """
        incoming_view = self._transition_view(frame_index)
        projection = build_researcher_analyzer_projection_v2(
            self._context,
            self._replay.frames[frame_index],
            transition_view=incoming_view,
            presentation=EvaluationScenePresentationStateV1(
                controlled_global_slot=self._reference_global_slot,
                selected_global_slot=selected_global_slot,
                armed_lane=None,
                show_ranges=self._show_ranges,
            ),
            status_source_evidence_state=self._status_index.state_for_frame(
                frame_index
            ),
        )
        return build_replay_researcher_space_v1(
            self._context,
            projection.scene,
            authority_session_id=self._viewer_session_id,
            final_frame_index=self._artifact_summary.recorded_transition_count,
            selected_global_slot=selected_global_slot,
            incoming_transition=(
                None if incoming_view is None else incoming_view.transition
            ),
            outgoing_transition=(
                None
                if frame_index == len(self._replay.transitions)
                else self._replay.transitions[frame_index]
            ),
        )

    def _require_active_or_none(self, value: int | None, *, name: str) -> None:
        """Allow None or an exact integer naming a configured-active roster slot."""
        if value is None:
            return
        if type(value) is not int or value not in self._active_slots:
            raise ValueError(f"{name} must name a configured-active replay actor")

    def _visible_pov_actor_slot(self, presentation_key: str) -> int:
        """Resolve one current POV-scene key without exposing slot topology."""
        result = self.current_presentation()
        presentation = result.payload
        if result.outcome != "response" or type(presentation) not in (
            ReplayNoSharedObsAuthorizedPresentationFrameV1,
            ReplaySharedObsAuthorizedPresentationFrameV1,
        ):
            raise ValueError("current POV presentation is unavailable")
        agent_presentation = cast(
            ReplayNoSharedObsAuthorizedPresentationFrameV1
            | ReplaySharedObsAuthorizedPresentationFrameV1,
            presentation,
        )
        matching_agents = tuple(
            agent
            for agent in agent_presentation.current_endpoint.parts.scene.agents
            if agent.presentation_key == presentation_key
        )
        if len(matching_agents) != 1:
            raise ValueError("POV actor key is not visible in current authority")
        public_agent_id = matching_agents[0].public_agent_id
        matching_slots = tuple(
            row.global_slot
            for row in self._context.roster
            if row.configured_active and row.public_agent_id == public_agent_id
        )
        if len(matching_slots) != 1 or matching_slots[0] not in self._active_slots:
            raise ValueError("POV actor key does not join one active roster row")
        return matching_slots[0]

    def _error_result(
        self,
        outcome: ReplayServiceOutcomeV1,
        error_code: Literal[
            "invalid_cursor",
            "audience_unavailable",
            "stale_revision",
            "command_id_conflict",
            "server_shutting_down",
            "internal_error",
        ],
        message: str,
    ) -> ReplayServiceCommandResultV1:
        """Build a typed rejection carrying the latest committed viewer frame."""
        return ReplayServiceCommandResultV1(
            outcome=outcome,
            payload=ReplayApiErrorV1(
                error_code=error_code,
                message=message,
                latest_frame=self._frame,
            ),
        )

    def _record_error(
        self,
        key: tuple[str, str],
        fingerprint: str,
        outcome: ReplayServiceOutcomeV1,
        error_code: Literal[
            "invalid_cursor",
            "audience_unavailable",
            "stale_revision",
            "command_id_conflict",
            "server_shutting_down",
            "internal_error",
        ],
        message: str,
    ) -> ReplayServiceCommandResultV1:
        """Build a rejection and retain its request identity in the bounded cache."""
        result = self._error_result(outcome, error_code, message)
        records = self._command_records.copy()
        self._remember_in(
            records,
            key,
            fingerprint,
            False,
        )
        self._command_records = records
        return result

    @staticmethod
    def _remember_in(
        records: OrderedDict[tuple[str, str], _CommandRecord],
        key: tuple[str, str],
        fingerprint: str,
        shutdown_requested: bool,
    ) -> None:
        """Add a request record and evict oldest entries beyond the 256-record limit."""
        records[key] = _CommandRecord(
            fingerprint=fingerprint,
            shutdown_requested=shutdown_requested,
        )
        records.move_to_end(key)
        while len(records) > _COMMAND_RECORD_LIMIT:
            records.popitem(last=False)


__all__ = [
    "FIGHT_MERGE_QUIET_TICKS",
    "PresentationResourceResultV1",
    "ReplayServiceCommandResultV1",
    "ReplayServiceOutcomeV1",
    "ReplayTimelineAgentMarkV1",
    "ReplayTimelineFallMarkV1",
    "ReplayTimelineFightV1",
    "ReplayTimelineMarksV1",
    "ReplayTimelineTickMarksV1",
    "ReplayTimelineUltimateMarkV1",
    "ReplayViewerService",
    "build_replay_fight_chapters_v1",
]
