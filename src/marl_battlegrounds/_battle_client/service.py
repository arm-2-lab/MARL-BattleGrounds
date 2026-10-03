"""Own the live browser session and serialize its commands.

``DebuggerService`` keeps one committed session and response frame under a
reentrant lock. Candidate frames and authorized presentations are checked
before installation. A bounded command cache prevents repeated execution
while an ID remains cached. Optional recording captures accepted transitions,
saves through the recorder, and hands verified output to the replay service.
The HTTP server owns transport and authentication. Callers may supply a scenario
catalog; otherwise each frame lists only the current scenario.

Agent POV recipient: the service keeps whose eyes Agent POV shows separately
from the focused champion (``control.agent_pov_recipient_slot``): the last
Manual champion you focused, or in Policies Only the champion you look at. It
is recomputed for every candidate session and committed with it, so an Agent
POV frame always describes the recipient while Oracle View describes the focus.
Open Replay always opens the child Replay Viewer in Oracle View.

Honors: the presentation builder keeps each committed game's per-transition
totals (``viewer.match_summary.match_transition_totals_v1``, from the Oracle
match summary it builds anyway) by frame index, and on the completion frame
adds them up into each view's ``match_honors`` (Oracle View every champion,
Agent POV the recipient's own team). A game whose history has a gap (a frame
never built) gets no Honors rather than wrong ones.

Background save (``save_in_background=True``, what the BattleClient uses): when
the game ends, the service installs the game-over frame at once with recording
lifecycle "sealed" (the page says "Saving Replay…") and saves the replay on its
own thread, outside the service lock. When the save ends, that thread installs
one new revision whose recording says "saved" (or "persistence_failed"). Any
command that needs the recording (Exit, Restart, Open Replay, Retry Save, Save
As, Ctrl-C closeout, ``close``) first waits for the save; view, preset, focus
and the G, V, P and ? keys do not wait. Without the flag every save runs inside
the command that ends the game, as before.

Exit, Ctrl-C and Restart before the first tick write no replay file: a game
with no played tick has nothing to watch.
"""

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from secrets import token_urlsafe
from threading import Event, RLock, Thread
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from marl_battlegrounds._battle_client.control import (
    DebuggerTransitionFailureStageV1,
    DebuggerTransitionFailureV1,
    agent_pov_recipient_slot,
    reset_session,
    set_combat_configuration,
    switch_scenario,
)
from marl_battlegrounds._battle_client.frame import (
    LiveDebuggerFrame,
    build_debugger_frame,
)
from marl_battlegrounds._battle_client.input import (
    InputDispatchResult,
    _agent_pov_focusable_slots,  # pyright: ignore[reportPrivateUsage]
    dispatch_command,
    normalize_key,
    recording_restart_intent_v1,
)
from marl_battlegrounds._battle_client.live_presentation import (
    build_live_no_shared_obs_authorized_presentation_v1,
    build_live_oracle_authorized_presentation_v1,
    build_live_researcher_space_v1,
    build_live_shared_obs_authorized_presentation_v1,
)
from marl_battlegrounds._battle_client.methods import LoadedControllers
from marl_battlegrounds._battle_client.model import (
    SCENARIO_OPPONENT_ERROR,
    DebuggerScenario,
    DebuggerSession,
    is_system_controller,
)
from marl_battlegrounds._battle_client.protocol import (
    ActorPovLiveDebuggerFrameV2,
    ApiErrorV2,
    CancelSystemCommandV1,
    CommandRequestV1,
    CommandResponseV2,
    ConfirmDiscardAndReplaceCommandV1,
    DebuggerCommandV1,
    ExitCommandV1,
    FinishAndReviewCommandV1,
    FinishSystemCommandV1,
    KeyboardCommandV1,
    PauseCommandV1,
    Preset,
    RecordingLifecycleV1,
    RecordingPersistenceErrorCodeV1,
    RecordingStatusV1,
    ResearcherLiveDebuggerFrameV2,
    ResetCommandV1,
    RetrySaveCommandV1,
    ReviewReplayCommandV1,
    RosterSelectionCommandV1,
    RunCommandV1,
    SaveAndReplaceCommandV1,
    SaveAsCommandV1,
    ScenarioSwitchCommandV1,
    SetCombatConfigurationCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
    SharedObsAgentPovLiveDebuggerFrameV2,
    SystemChoiceV1,
    SystemOperationV1,
    ViewMode,
)
from marl_battlegrounds._battle_client.recording import (
    DebuggerRecordingCloseCauseV1,
)
from marl_battlegrounds._battle_client.replay_recorder import DebuggerReplayRecorder
from marl_battlegrounds._battle_client.system_worker import (
    PendingSystemOperation,
    SystemInstallation,
    SystemWorker,
)
from marl_battlegrounds.evaluation.replay_io import LoadedReplay
from marl_battlegrounds.rendering.evaluation_adapter import (
    build_visual_event_batch_v2,
)
from marl_battlegrounds.viewer.match_summary import (
    MatchHonorsV1,
    MatchSummaryV1,
    MatchTotalsV1,
    match_honors_v1,
    match_transition_totals_v1,
)
from marl_battlegrounds.viewer.no_shared_visual import (
    build_live_no_shared_obs_visual_adjacent_slice_v1,
    build_live_no_shared_obs_visual_current_slice_v1,
)
from marl_battlegrounds.viewer.presentation_protocol import (
    AuthorizedPresentationFrameV1,
    PresentationResourceResultV1,
)
from marl_battlegrounds.viewer.replay_service import ReplayViewerService

# How many recent presentations a service keeps for reuse: the commit build, the
# page's request and the live snapshot share one; a view switch adds another.
_REMEMBERED_PRESENTATIONS = 4


class LiveSnapshotV1(BaseModel):
    """Keep one authorized live display and its raw frame at the same revision.

    Both fields are existing validated response models. The service constructs
    them while holding one snapshot lock; no method is called by this read.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    frame: LiveDebuggerFrame
    presentation: AuthorizedPresentationFrameV1


_COMMAND_RECORD_LIMIT = 256
_CLOSED_RECORDING_PRESENTATION_KEYS = frozenset(("g", "v", "p", "?"))

type ServiceOutcome = Literal[
    "response",
    "stale_revision",
    "command_id_conflict",
    "server_shutting_down",
    "service_faulted",
]


@dataclass(frozen=True, slots=True)
class ServiceCommandResult:
    """One transport-neutral command outcome and its validated payload.

    queued_operation_id is host-only proof that this request queued a new setup.
    It stays None for rejections, duplicates and unrelated pending work.
    """

    outcome: ServiceOutcome
    payload: CommandResponseV2 | ApiErrorV2
    shutdown_requested: bool = False
    replay_handoff: ReplayViewerService | None = None
    queued_operation_id: str | None = None


@dataclass(frozen=True, slots=True)
class RecordingCloseResult:
    """Host-only result of one keyboard-interrupt recording closeout."""

    saved: bool
    message: str

    def __post_init__(self) -> None:
        """Require an exact boolean saved flag and a nonempty host message."""
        if type(self.saved) is not bool:
            raise TypeError("recording close saved flag must be a Python bool.")
        if type(self.message) is not str or not self.message:
            raise ValueError("recording close message must be nonempty.")


@dataclass(frozen=True, slots=True)
class _LoadedSetup:
    """Keep one trusted host replacement attached only to its current request.

    identity is canonical validated request text, not a callable/object repr.
    expected_session fences validation against a replaced match. load/build run
    only in the worker; on_installed publishes private choices after adoption.
    """

    identity: str
    expected_session: DebuggerSession
    load: Callable[[], LoadedControllers]
    build: Callable[[LoadedControllers], DebuggerSession]
    on_installed: Callable[[], None] | None


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    """Remember one request fingerprint and whether it requested shutdown."""

    fingerprint: str
    shutdown_requested: bool


@dataclass(frozen=True, slots=True)
class _RecordingResponseCandidate:
    """Hold a validated frame, response, and cache update before state is committed."""

    revision: int
    frame: LiveDebuggerFrame
    response: CommandResponseV2
    command_records: OrderedDict[tuple[str, str], _CommandRecord]
    shutdown_requested: bool


@dataclass(frozen=True, slots=True)
class _EndpointResponseCandidates:
    """Hold checked success and save-failure responses for a sealed episode."""

    saved: _RecordingResponseCandidate
    failures: dict[
        RecordingPersistenceErrorCodeV1,
        _RecordingResponseCandidate,
    ]


@dataclass(frozen=True, slots=True)
class _FailureCloseoutCandidates:
    """Hold checked responses for saving the last accepted prefix after a failure."""

    saved: _RecordingResponseCandidate
    failures: dict[
        RecordingPersistenceErrorCodeV1,
        _RecordingResponseCandidate,
    ]


@dataclass(slots=True)
class _BackgroundSave:
    """One game-over replay save that runs on its own thread.

    Attributes
    ----------
    recorder : DebuggerReplayRecorder
        The sealed recorder being saved. While the save runs, only its thread
        touches it; the service reads ``status`` instead.
    close_cause : {'endpoint', 'truncation'}
        Why capture ended, passed to ``finalize_and_save``.
    status : RecordingStatusV1
        The frozen "sealed" status every frame shows while the save runs.
    done : threading.Event
        Set by the save thread when ``finalize_and_save`` returns or raises,
        before that thread asks for the service lock.
    error : BaseException or None
        An unexpected exception from ``finalize_and_save``; None after an
        ordinary saved or failed result (the recorder holds that result).
    thread : threading.Thread or None
        The save thread, set once it is created.
    """

    recorder: DebuggerReplayRecorder
    close_cause: Literal["endpoint", "truncation"]
    status: RecordingStatusV1
    done: Event = field(default_factory=Event)
    error: BaseException | None = None
    thread: Thread | None = None


class DebuggerService:
    """Serialize commands around one immutable debugger session."""

    def __init__(
        self,
        session: DebuggerSession,
        *,
        view_mode: ViewMode,
        preset: Preset | Literal["technical", "debug"],
        include_stress: bool,
        session_id: str | None = None,
        recorder: DebuggerReplayRecorder | None = None,
        offered_systems: Mapping[str, str] | None = None,
        scenario_catalog: Sequence[DebuggerScenario] | None = None,
        system_worker: SystemWorker | None = None,
        save_in_background: bool = False,
    ) -> None:
        """Build the live service around an existing immutable session.

        Parameters
        ----------
        session : DebuggerSession
            Current live episode and matching captured frame.
        view_mode : ViewMode
            Starting view: "researcher" (Oracle View) or "pov" (Agent POV). Agent
            POV frames describe the recipient chosen by
            control.agent_pov_recipient_slot, not necessarily the focus.
        preset : Preset or {'technical', 'debug'}
            Accepted compatibility preset. All accepted values start in analysis mode.
        include_stress : bool
            Whether registered stress scenarios may be selected.
        session_id : str or None, optional
            Session identity for protocol frames. Missing or empty values create a
            random identity. Defaults to None.
        recorder : DebuggerReplayRecorder or None, optional
            Exact recorder owning this open episode prefix. None requires the debug
            capture profile; recording requires metric-complete capture.

        scenario_catalog : sequence of DebuggerScenario or None, optional
            Ordered scenario choices retained as a tuple. None shows only the
            current scenario, including after replacement. Stress choices are
            shown only when include_stress=True. Factories are not called.
        system_worker : SystemWorker or None, optional
            Worker already owning this exact session. None preserves lazy private
            setup. The server pump drains caller work on its creating thread and
            adopts worker results outside HTTP/router locks.
        offered_systems : mapping of str to str, optional
            Launcher names mapped to trusted module:function factories. Only a
            selected name loads. Browser input cannot supply a factory or URL.
        save_in_background : bool, default=False
            True saves the replay of a game that ends (score limit or tick
            limit) on a separate thread: the command that ends the game returns
            at once with recording lifecycle "sealed", and a later revision
            shows "saved" or "persistence_failed". Use it for interactive
            launchers (the BattleClient). False keeps the save inside that
            command, so a scripted caller sees "saved" as soon as the command
            returns. Finish And Save, Exit, Retry Save, Save As, failure
            closeouts and Ctrl-C always save inside their own call. Ignored
            without a recorder.

        Raises
        ------
        TypeError
            The recorder has an unsupported exact type, or save_in_background is
            not a bool.
        ValueError
            Scenario access, preset, capture profile, or recorder identity is invalid.

        Notes
        -----
        Construction builds the initial host frame and lock. It does not advance the
        simulator or publish a replay file. The first Agent POV recipient is the
        focus when it is Manual or nobody is Manual, else the lowest Manual slot.
        A background save thread is not a daemon: ``close`` waits for it, so the
        replay is written before a launcher returns.
        """
        if type(save_in_background) is not bool:
            raise TypeError("save_in_background must be a bool.")
        if session.scenario.audience == "stress" and not include_stress:
            msg = (
                f"stress scenario {session.scenario_name!r} requires "
                "include_stress=True."
            )
            raise ValueError(msg)
        if preset not in ("presentation", "analysis", "technical", "debug"):
            raise ValueError("unknown debugger preset")
        self._session: DebuggerSession = session
        self._view_mode: ViewMode = view_mode
        self._preset: Preset = "analysis"
        self._include_stress = include_stress
        self._scenario_catalog = (
            None if scenario_catalog is None else tuple(scenario_catalog)
        )
        self._session_id = session_id or token_urlsafe(24)
        self._revision = 0
        self._running = False
        self._pump_client_id = token_urlsafe(24)
        self._pump_sequence = 0
        self._shutting_down = False
        self._faulted = False
        self._lock = RLock()
        # Newest presentations with their exact inputs (_remembered_presentation).
        self._presentations: list[
            tuple[tuple[object, ...], PresentationResourceResultV1]
        ] = []
        # Honors history (_match_honors): the game it belongs to (episode ID,
        # run generation) and each built frame's transition totals by index.
        self._honors_game: tuple[str, int] | None = None
        self._honors_totals: dict[int, MatchTotalsV1] = {}
        self._command_records: OrderedDict[
            tuple[str, str],
            _CommandRecord,
        ] = OrderedDict()
        self._recorder: DebuggerReplayRecorder | None = self._validated_recorder(
            self._session,
            recorder,
        )
        self._replay_handoff_cache: tuple[LoadedReplay, ReplayViewerService] | None = (
            None
        )
        self._save_in_background = save_in_background
        # The running game-over save, or None (see _start_background_save).
        self._background_save: _BackgroundSave | None = None
        self._offered_systems = dict(offered_systems or {})
        if any(
            not is_system_controller(f"system:{name}") or not reference.strip()
            for name, reference in self._offered_systems.items()
        ):
            raise ValueError(
                "Offered Systems need a bounded name and factory reference"
            )
        self._system_choices = tuple(
            SystemChoiceV1(
                id=f"system:{name}",
                label=name.replace("_", " ").replace("-", " ").title(),
            )
            for name in self._offered_systems
        )
        if system_worker is not None and (
            system_worker.active is None or system_worker.active.session is not session
        ):
            raise ValueError("The supplied worker must own this exact session")
        self._system_worker: SystemWorker | None = system_worker
        self._system_evidence: SystemInstallation | None = (
            None if system_worker is None else system_worker.active
        )
        self._system_confirmed_discard = False
        self._system_failed = False
        self._pov_global_slot: int = agent_pov_recipient_slot(session, None)
        self._frame: LiveDebuggerFrame = self._build_frame()

    @staticmethod
    def _validated_recorder(
        session: DebuggerSession,
        recorder: DebuggerReplayRecorder | None,
    ) -> DebuggerReplayRecorder | None:
        """Require the recorder to own this exact open capture, or require debug
        mode.
        """
        if recorder is None:
            if session.evaluation_context.capture_profile != "debug":
                raise ValueError(
                    "unrecorded live debugger sessions require the debug profile."
                )
            return None
        raw_recorder = cast(object, recorder)
        if type(raw_recorder) is not DebuggerReplayRecorder:
            raise TypeError("recorder must be an exact supported debugger recorder.")
        if session.evaluation_context.capture_profile != "evaluation_metric_complete":
            raise ValueError("recording sessions require metric-complete capture.")
        if (
            recorder.lifecycle != "recording"
            or recorder.context != session.evaluation_context
            or recorder.current_frame != session.current_evaluation_frame
            or recorder.validated_transition_count
            != session.current_evaluation_frame.frame_index
        ):
            raise ValueError(
                "recorder must own the current open debugger episode prefix."
            )
        return recorder

    @property
    def session(self) -> DebuggerSession:
        """Return the current immutable session for diagnostics and tests."""
        with self._lock:
            return self._session

    def load_scenario(
        self,
        scenario: DebuggerScenario,
        *,
        on_installed: Callable[[], None] | None = None,
    ) -> DebuggerSession | PendingSystemOperation:
        """Replace an unrecorded live episode after validating its full response.

        Parameters
        ----------
        scenario : DebuggerScenario
            Exact compiled scenario to load into a reset episode.
        on_installed : callable or None, optional
            Trusted host callback called once after successful adoption. None does
            no extra work. Cancellation and failed candidates never call it.

        Returns
        -------
        DebuggerSession or PendingSystemOperation
            Newly installed session, or a declared System setup awaiting explicit
            completion. Pending work leaves the old game intact. Response and
            presentation checks precede replacement. Synchronous loads clear old
            command IDs; deferred loads retain the normal bounded command cache.

        Raises
        ------
        TypeError
            The scenario has the wrong exact type.
        RuntimeError
            The service is stopping, faulted, or owns a replay recording.
        ValueError
            Scenario setup or its candidate response fails validation.

        Notes
        -----
        The operation holds the service lock and performs reset work. It does not
        write an authored scenario file.
        """
        if type(scenario) is not DebuggerScenario:
            raise TypeError("scenario must be the exact DebuggerScenario type.")
        with self._lock:
            if (
                self._session.scenario.scenario_specification is not None
                or scenario.scenario_specification is not None
            ):
                raise ValueError(
                    "Scenarios are fixed. "
                    "Start a new BattleClient to change the source."
                )
            if self._shutting_down:
                raise RuntimeError("the debugger is shutting down")
            if self._faulted:
                raise RuntimeError("the debugger service is faulted")
            if self._recorder is not None:
                raise RuntimeError(
                    "authored scenarios cannot replace a replay-recording session"
                )

            worker = self._system_worker
            if worker is not None and worker.caller_thread:
                raise ValueError(
                    "Launch assignments are fixed. Start a new BattleClient."
                )
            if worker is not None:
                if worker.operation is not None:
                    worker.cancel(worker.operation)
                    raise RuntimeError(
                        "System work is being cancelled; retry after cleanup"
                    )
                base = self._session
                controllers = (base.team_a_controller, base.team_b_controller)
                if any(is_system_controller(value) for value in controllers):
                    operation = worker.prepare(
                        base,
                        self._offered_systems,
                        controllers,
                        lambda pair: switch_scenario(base, scenario, systems=pair),
                        directory=None,
                    )
                    operation.on_installed = on_installed
                    self._system_failed = False
                    self._system_confirmed_discard = False
                    self._revision += 1
                    self._frame = self._build_frame()
                    return operation
            candidate_session = switch_scenario(self._session, scenario)
            candidate_revision = self._revision + 1
            candidate_frame = self._build_frame(
                session=candidate_session,
                revision=candidate_revision,
            )
            self._build_authorized_presentation(
                session=candidate_session,
                raw_frame=candidate_frame,
                view_mode=self._view_mode,
            )

            if candidate_session.run_generation != self._session.run_generation:
                self._running = False
            self._pov_global_slot = self._recipient_for(candidate_session)
            self._session = candidate_session
            self._revision = candidate_revision
            self._frame = candidate_frame
            self._command_records.clear()
            if on_installed is not None:
                on_installed()
            return candidate_session

    @property
    def revision(self) -> int:
        """Return the current service/frame revision."""
        with self._lock:
            return self._revision

    @property
    def command_cache_size(self) -> int:
        """Return the bounded idempotency-record count."""
        with self._lock:
            return len(self._command_records)

    @property
    def evaluation_validated_transition_count(self) -> int:
        """Return the count from the committed episode frame."""
        with self._lock:
            if self._recorder is not None:
                return self._recorder.validated_transition_count
            return self._session.current_evaluation_frame.frame_index

    @property
    def evaluation_observer_lifecycle_state(self) -> str:
        """Return whether the committed episode is open or finished."""
        with self._lock:
            if self._recorder is not None:
                return self._recorder.observer_lifecycle_state
            incoming = self._session.incoming_evaluation_view
            if self._session.reached_declared_horizon or (
                incoming is not None
                and (incoming.transition.terminated or incoming.transition.truncated)
            ):
                return "sealed"
            return "open"

    @property
    def recording_status(self) -> RecordingStatusV1 | None:
        """Return the path-free recorder status without exposing its mutator.

        While a background save runs this is its frozen "sealed" status, the
        same status every frame shows (see _current_recording_status).
        """
        with self._lock:
            return self._current_recording_status()

    def close_recording_for_keyboard_interrupt(self) -> RecordingCloseResult:
        """Try to save active capture before the hosting process exits.

        Returns
        -------
        RecordingCloseResult
            Whether capture is safe to leave and a host-facing status message.
            No active capture, an already saved replay and a recording with no
            played tick all count as success.

        Notes
        -----
        The service lock covers response preparation and saving. A running
        background save is waited for first and then counts as the save. A
        recording that is still open with zero captured ticks (Ctrl-C right
        after start) writes no file: an empty replay holds nothing to watch, as
        Restart already treats it. A failed initial save is retried; a failed
        retry tries a deterministic recovery sibling. The revision and displayed
        recording status reflect the final result. A verified save queues its
        exact episode's worker evidence before cleanup. Unexpected invariant or
        response-validation failures still propagate.
        """
        with self._lock:
            self._await_background_save()
            recorder = self._recorder
            if recorder is None or recorder.lifecycle == "discarded":
                return RecordingCloseResult(
                    saved=True,
                    message="No replay recording was active.",
                )
            if (
                recorder.lifecycle == "recording"
                and recorder.validated_transition_count == 0
            ):
                return RecordingCloseResult(
                    saved=True,
                    message="No tick was played, so no replay was saved.",
                )
            if recorder.lifecycle in ("saved", "reviewing"):
                saved = recorder.saved_bundle
                if saved is None:
                    raise AssertionError("saved recording is missing its artifact.")
                self._acknowledge_saved_system_prefix()
                return RecordingCloseResult(
                    saved=True,
                    message=(
                        f"Replay recording was already saved at {saved.replay_path}."
                    ),
                )

            close_cause: DebuggerRecordingCloseCauseV1 | None = (
                "keyboard_interrupt"
                if recorder.lifecycle == "recording"
                else recorder.close_cause
            )
            if close_cause is None:
                raise AssertionError("recording closeout is missing its close cause.")
            completion_reason = recorder.status.completion_reason
            candidate_revision = self._revision + 1
            captured_transition_count = recorder.validated_transition_count
            saved_frame = self._build_frame(
                revision=candidate_revision,
                recording_status=recorder.preview_status_v1(
                    captured_transition_count=captured_transition_count,
                    lifecycle="saved",
                    close_cause=close_cause,
                    completion_reason=completion_reason,
                ),
            )
            failure_frames = {
                error_code: self._build_frame(
                    revision=candidate_revision,
                    recording_status=recorder.preview_status_v1(
                        captured_transition_count=captured_transition_count,
                        lifecycle="persistence_failed",
                        close_cause=close_cause,
                        completion_reason=completion_reason,
                        persistence_error_code=error_code,
                    ),
                )
                for error_code in cast(
                    tuple[RecordingPersistenceErrorCodeV1, ...],
                    (
                        "target_unavailable",
                        "publication_failed",
                        "verification_failed",
                    ),
                )
            }
            self._build_authorized_presentation(
                session=self._session,
                raw_frame=saved_frame,
                view_mode=self._view_mode,
            )
            for failure_frame in failure_frames.values():
                self._build_authorized_presentation(
                    session=self._session,
                    raw_frame=failure_frame,
                    view_mode=self._view_mode,
                )

            if recorder.lifecycle == "persistence_failed":
                outcome = recorder.retry_save()
            else:
                outcome = recorder.finalize_and_save(close_cause)
                if outcome == "persistence_failed":
                    outcome = recorder.retry_save()
            if outcome == "persistence_failed":
                outcome = recorder.save_recovery_copy()

            self._revision = candidate_revision
            if outcome == "saved":
                saved = recorder.saved_bundle
                if saved is None:
                    raise AssertionError("successful closeout lacks a saved artifact.")
                self._frame = saved_frame
                self._acknowledge_saved_system_prefix()
                return RecordingCloseResult(
                    saved=True,
                    message=f"Replay recording saved at {saved.replay_path}.",
                )

            error_code = recorder.persistence_error_code
            if error_code is None:
                raise AssertionError("failed closeout lacks a persistence error code.")
            self._frame = failure_frames[error_code]
            return RecordingCloseResult(
                saved=False,
                message=(
                    "Replay recording could not be saved; no recovery copy was written."
                ),
            )

    def _validated_transition_count(self) -> int:
        """Read the committed recorder count, or the unrecorded current frame index."""
        if self._recorder is not None:
            return self._recorder.validated_transition_count
        return self._session.current_evaluation_frame.frame_index

    @property
    def shutting_down(self) -> bool:
        """Return whether an accepted Exit has fenced new commands."""
        with self._lock:
            return self._shutting_down

    @property
    def faulted(self) -> bool:
        """Return whether an internal failure fenced all later commands."""
        with self._lock:
            return self._faulted

    def session_snapshot(self) -> tuple[DebuggerSession, int]:
        """Read the immutable session and its transport revision under one lock.

        Private setup uses this pair to describe or validate a source while the
        autonomous pump can advance. It does not run methods or perform I/O.
        """
        with self._lock:
            return self._session, self._revision

    def current_snapshot(self) -> LiveSnapshotV1:
        """Read raw and authorized live responses from one accepted snapshot.

        This is the atomic alternative to two HTTP reads while Run is advancing.
        It reuses existing audience validation and returns no private host paths.
        The read may build display data; it never calls a method or changes play.
        """
        with self._lock:
            presentation = self.current_presentation()
            return LiveSnapshotV1(
                frame=self._frame,
                presentation=cast(AuthorizedPresentationFrameV1, presentation.payload),
            )

    def current_frame(self) -> LiveDebuggerFrame:
        """Read the current immutable transport frame under the service lock.

        Returns
        -------
        LiveDebuggerFrame
            The already built frame for the committed revision; no simulator work
            or new recording I/O is performed.
        """
        with self._lock:
            return self._frame

    def current_presentation(self) -> PresentationResourceResultV1:
        """Build the authorized display resource for the committed live snapshot.

        Returns
        -------
        PresentationResourceResultV1
            Checked presentation for the current researcher or actor view.

        Raises
        ------
        RuntimeError
            The retained transport frame differs from the service-owned snapshot.
        ValueError
            An authority, epoch, or display contract fails validation.

        Notes
        -----
        This holds the service lock while deriving host display data. It does not
        advance the simulator or expand an actor's information rights.
        """
        with self._lock:
            raw_frame = self._frame
            # Worker completion can change between polls. Compare scientific
            # state with the status that was actually published in this frame.
            expected_frame = self._build_frame().model_copy(
                update={
                    "system_operation": raw_frame.system_operation,
                    "run_state": raw_frame.run_state,
                    "fresh_launch_required": raw_frame.fresh_launch_required,
                }
            )
            if (
                type(raw_frame) is not type(expected_frame)
                or raw_frame != expected_frame
            ):
                raise RuntimeError(
                    "committed live frame diverged from service-owned state."
                )
            return self._build_authorized_presentation(
                session=self._session,
                raw_frame=raw_frame,
                view_mode=self._view_mode,
            )

    def _recipient_for(self, session: DebuggerSession) -> int:
        """Return the Agent POV recipient for a candidate or committed session.

        Applies control.agent_pov_recipient_slot with the committed recipient as
        the previous one. Pure: building frames and committing a candidate both
        call it with the same inputs, so they agree. No I/O or simulator work.
        """
        return agent_pov_recipient_slot(session, self._pov_global_slot)

    @property
    def pov_global_slot(self) -> int:
        """Return the committed Agent POV recipient, a global slot 0-9.

        This is whose eyes Agent POV shows (see control.agent_pov_recipient_slot);
        it may differ from session.controlled_global_slot, the focus. Read under
        the service lock; it changes only when a session is committed.
        """
        with self._lock:
            return self._pov_global_slot

    def _build_authorized_presentation(
        self,
        *,
        session: DebuggerSession,
        raw_frame: LiveDebuggerFrame,
        view_mode: ViewMode,
    ) -> PresentationResourceResultV1:
        """Build and strictly validate one parameterized live presentation.

        session is the committed or candidate session, raw_frame its matching
        transport frame and view_mode the audience. Oracle View describes the
        focus. Agent POV describes the recipient from _recipient_for: its sight,
        draft and the researcher space's selected agent are all the recipient's,
        so a focused policy champion never changes what Agent POV shows. Every
        audience's match summary gets the session's scores_ten_ticks_ago (the
        Momentum switch's public score history). The frame's champion_streaks
        is this audience's own row of session.champion_takedowns, and its
        match_moments is the same audience's row of session.match_moments
        (the Herald Cards history): Oracle View's, or the recipient's Agent
        POV row (never another view's). On the
        completion frame its match_honors is this audience's Honors
        (_match_honors: Oracle View every champion, Agent POV the recipient's
        own team); the Oracle frame is then built a second time with them,
        once per game. Raises RuntimeError when raw_frame does not match the
        audience and ValueError when a presentation contract fails. No
        simulator step or file write.

        A commit builds the candidate's presentation to prove it can be shown;
        the page's request and the live snapshot then ask for the same one. A
        build is therefore remembered (_remembered_presentation) and reused when
        every input is the same: the same session and raw frame objects (both
        frozen), the same audience and, in Agent POV, the same recipient and an
        equal researcher-space frame, rebuilt here because it carries live
        worker status. Any difference builds afresh, so a reused result always
        equals a fresh build.
        """
        context = session.evaluation_context
        current = session.current_evaluation_frame
        incoming = session.incoming_evaluation_view
        if view_mode == "researcher":
            if type(raw_frame) is not ResearcherLiveDebuggerFrameV2:
                raise RuntimeError(
                    "researcher live service lacks its committed Oracle frame."
                )
            key: tuple[object, ...] = ("researcher", session, raw_frame)
            remembered = self._remembered_presentation(key)
            if remembered is not None:
                return remembered
            presentation = build_live_oracle_authorized_presentation_v1(
                context,
                current,
                incoming,
                raw_frame,
                scores_ten_ticks_ago=session.scores_ten_ticks_ago,
                champion_streaks=session.champion_takedowns.streaks(context, None),
                match_moments=session.match_moments.facts(context, None),
            )
            honors = self._match_honors(session, presentation.match_summary, None)
            if honors is not None:
                presentation = build_live_oracle_authorized_presentation_v1(
                    context,
                    current,
                    incoming,
                    raw_frame,
                    scores_ten_ticks_ago=session.scores_ten_ticks_ago,
                    champion_streaks=session.champion_takedowns.streaks(context, None),
                    match_honors=honors,
                    match_moments=session.match_moments.facts(context, None),
                )
        else:
            if type(raw_frame) not in (
                ActorPovLiveDebuggerFrameV2,
                CancelSystemCommandV1,
                FinishSystemCommandV1,
                SetCombatConfigurationCommandV1,
                SystemChoiceV1,
                SystemOperationV1,
                SharedObsAgentPovLiveDebuggerFrameV2,
            ):
                raise RuntimeError(
                    "POV live service lacks its committed recipient frame."
                )
            source_session = session
            recipient = self._recipient_for(session)
            if recipient != session.controlled_global_slot:
                # Agent POV panels and joins describe the recipient, never the focus.
                session = replace(session, controlled_global_slot=recipient)
            oracle_raw_frame = self._build_frame(
                session=session,
                revision=raw_frame.revision,
                view_mode="researcher",
                preset=raw_frame.preset,
                recording_status=raw_frame.recording,
            )
            if type(oracle_raw_frame) is not ResearcherLiveDebuggerFrameV2:
                raise RuntimeError(
                    "POV live service cannot build its researcher-space epoch."
                )
            key = ("pov", source_session, raw_frame, recipient, oracle_raw_frame)
            remembered = self._remembered_presentation(key)
            if remembered is not None:
                return remembered
            oracle_presentation = build_live_oracle_authorized_presentation_v1(
                context,
                current,
                incoming,
                oracle_raw_frame,
                scores_ten_ticks_ago=session.scores_ten_ticks_ago,
            )
            researcher_space = build_live_researcher_space_v1(oracle_presentation)
            honors = self._match_honors(
                session,
                oracle_presentation.match_summary,
                cast(
                    "Literal[1, 2]",
                    context.roster[recipient].configured_team_id,
                ),
            )
            if type(raw_frame) is SharedObsAgentPovLiveDebuggerFrameV2:
                presentation = build_live_shared_obs_authorized_presentation_v1(
                    context,
                    current,
                    incoming,
                    raw_frame,
                    authorized_recipient_global_slot=recipient,
                    pending_action=session.pending_actions[recipient],
                    researcher_space=researcher_space,
                    scores_ten_ticks_ago=session.scores_ten_ticks_ago,
                    champion_streaks=session.champion_takedowns.streaks(
                        context, recipient
                    ),
                    match_honors=honors,
                    match_moments=session.match_moments.facts(context, recipient),
                )
            else:
                if type(raw_frame) is not ActorPovLiveDebuggerFrameV2:
                    raise AssertionError("POV raw-frame narrowing failed.")
                current_slice = build_live_no_shared_obs_visual_current_slice_v1(
                    context,
                    current,
                    global_slot=recipient,
                    incoming_transition_view=incoming,
                )
                carrier = (
                    None
                    if incoming is None
                    else build_live_no_shared_obs_visual_adjacent_slice_v1(
                        incoming,
                        global_slot=recipient,
                    )
                )
                presentation = build_live_no_shared_obs_authorized_presentation_v1(
                    current_slice,
                    carrier,
                    raw_frame,
                    global_context=context,
                    current_global_frame=current,
                    previous_global_frame=(
                        None if incoming is None else incoming.start_frame
                    ),
                    public_catalog=context.static_mechanics_catalog,
                    incoming_visual_events=(
                        None
                        if incoming is None
                        else build_visual_event_batch_v2(incoming)
                    ),
                    researcher_space=researcher_space,
                    scores_ten_ticks_ago=session.scores_ten_ticks_ago,
                    champion_streaks=session.champion_takedowns.streaks(
                        context, recipient
                    ),
                    match_honors=honors,
                    match_moments=session.match_moments.facts(context, recipient),
                )
        result = PresentationResourceResultV1(
            outcome="response",
            payload=presentation,
        )
        with self._lock:
            self._presentations.append((key, result))
            del self._presentations[:-_REMEMBERED_PRESENTATIONS]
        return result

    def _match_honors(
        self,
        session: DebuggerSession,
        summary: MatchSummaryV1 | None,
        team_id: Literal[1, 2] | None,
    ) -> MatchHonorsV1 | None:
        """Record this frame's transition totals and, at game end, give Honors.

        Parameters
        ----------
        session : DebuggerSession
            The committed or candidate session being shown.
        summary : MatchSummaryV1 or None
            The Oracle match summary of the session's current frame (built with
            its transition start frame), or None (no summary: no Honors).
        team_id : {1, 2} or None
            None for Oracle View; the Agent POV recipient's team otherwise.

        Returns
        -------
        MatchHonorsV1 or None
            On the completion frame of a Team Deathmatch game whose frames 1
            to the last were all built, this view's Honors
            (viewer.match_summary.match_honors_v1 over every frame's totals);
            otherwise None.

        Notes
        -----
        Side effect: under the service lock, stores the totals of the frame's
        incoming transition by frame index for its game (episode ID and run
        generation; a new game or restart clears the store). A later build of
        the same frame replaces its entry, so a rejected candidate never
        counts twice. Host work on the transition's own events only; no JAX
        (the summary already holds the kill credit), file or simulator work.
        """
        if summary is None or summary.task_mode != 1:
            return None
        current = session.current_evaluation_frame
        incoming = session.incoming_evaluation_view
        game = (current.episode_id, session.run_generation)
        with self._lock:
            if self._honors_game != game:
                self._honors_game = game
                self._honors_totals = {}
            if incoming is not None and current.frame_index > 0:
                self._honors_totals[current.frame_index] = match_transition_totals_v1(
                    session.evaluation_context, summary, incoming.transition.events
                )
            if summary.outcome in ("in_progress", "not_applicable"):
                return None
            totals = MatchTotalsV1()
            for index in range(1, current.frame_index + 1):
                step = self._honors_totals.get(index)
                if step is None:
                    return None
                totals = totals.plus(step)
        return match_honors_v1(session.evaluation_context, totals, team_id=team_id)

    def _remembered_presentation(
        self, key: tuple[object, ...]
    ) -> PresentationResourceResultV1 | None:
        """Return a presentation already built from exactly these inputs, if any.

        key is (audience, session, raw frame) for Oracle View and (audience,
        source session, raw frame, recipient, researcher-space frame) for Agent
        POV. Sessions and raw frames match only as the same objects; the
        recipient and researcher-space frame match by value. The newest
        _REMEMBERED_PRESENTATIONS builds are kept, holding their inputs so no
        object identity is reused while remembered. Thread safe; no I/O.
        """
        with self._lock:
            for remembered_key, result in reversed(self._presentations):
                if len(remembered_key) == len(key) and all(
                    (left is right) if index in (1, 2) else (left == right)
                    for index, (left, right) in enumerate(
                        zip(remembered_key, key, strict=True)
                    )
                ):
                    return result
        return None

    def _recording_no_op(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        notice: str,
    ) -> ServiceCommandResult:
        """Remember an unavailable recording command and return the unchanged frame."""
        record = _CommandRecord(
            fingerprint=fingerprint,
            shutdown_requested=False,
        )
        self._remember_command(command_key, record)
        return ServiceCommandResult(
            outcome="response",
            payload=CommandResponseV2(
                result="no_op",
                frame=self._frame,
                notice=notice,
            ),
        )

    def _build_replay_handoff(self) -> ReplayViewerService:
        """Reuse one healthy Viewer for the exact verified bundle, or build it once.

        A new Viewer starts at frame zero in Oracle View, whatever the live view,
        with the focused champion as its reference and selection and the Agent
        POV recipient as the champion its own Agent POV switch would show.
        Reopening a healthy Viewer preserves its cursor and avoids hashing the
        replay again. Recorder replacement drops this cache; the coordinator owns
        its server.
        """
        recorder = self._recorder
        if recorder is None or recorder.verified_loaded_bundle is None:
            raise RuntimeError("replay handoff requires a verified recording.")
        cached = self._replay_handoff_cache
        if (
            cached is not None
            and cached[0] is recorder.verified_loaded_bundle
            and not cached[1].shutting_down
            and not cached[1].faulted
        ):
            return cached[1]
        reference_global_slot = self._session.controlled_global_slot
        pending = self._session.pending_actions[reference_global_slot]
        selected_global_slot = (
            reference_global_slot
            if pending.selected_global_target_slot is None
            else pending.selected_global_target_slot
        )
        armed_lane: Literal[0, 1] | None = pending.armed_lane
        viewer = ReplayViewerService(
            recorder.verified_loaded_bundle,
            initial_frame_index=0,
            view_mode="researcher",
            reference_global_slot=reference_global_slot,
            selected_global_slot=selected_global_slot,
            armed_lane=armed_lane,
            pov_global_slot=self._pov_global_slot,
            preset=self._preset,
            show_ranges=self._session.show_ranges,
            verbose=False,
        )
        self._replay_handoff_cache = (recorder.verified_loaded_bundle, viewer)
        return viewer

    def _prepare_recording_response(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        recording_status: RecordingStatusV1,
        result: Literal["applied", "no_op", "shutdown_scheduled"],
        notice: str,
        changed: bool,
        shutdown_requested: bool = False,
    ) -> _RecordingResponseCandidate:
        """Validate a proposed recording response and copied command cache before
        mutation.
        """
        candidate_revision = self._revision + int(changed)
        candidate_frame = self._build_frame(
            revision=candidate_revision,
            recording_status=recording_status,
        )
        self._build_authorized_presentation(
            session=self._session,
            raw_frame=candidate_frame,
            view_mode=self._view_mode,
        )
        response = CommandResponseV2(
            result=result,
            frame=candidate_frame,
            notice=notice,
        )
        record = _CommandRecord(
            fingerprint=fingerprint,
            shutdown_requested=shutdown_requested,
        )
        records = self._command_records.copy()
        self._remember_command_in(records, command_key, record)
        return _RecordingResponseCandidate(
            revision=candidate_revision,
            frame=candidate_frame,
            response=response,
            command_records=records,
            shutdown_requested=shutdown_requested,
        )

    def _commit_recording_response(
        self,
        candidate: _RecordingResponseCandidate,
        *,
        replay_handoff: ReplayViewerService | None = None,
    ) -> ServiceCommandResult:
        """Install a prepared response and optional shutdown flag; return any replay
        handoff.
        """
        self._acknowledge_saved_system_prefix()
        self._revision = candidate.revision
        self._frame = candidate.frame
        self._command_records = candidate.command_records
        if candidate.shutdown_requested:
            self._shutting_down = True
        return ServiceCommandResult(
            outcome="response",
            payload=candidate.response,
            shutdown_requested=candidate.shutdown_requested,
            replay_handoff=replay_handoff,
        )

    def _preview_recording_status(
        self,
        *,
        lifecycle: RecordingLifecycleV1,
        close_cause: DebuggerRecordingCloseCauseV1 | None = None,
        completion_reason: str | None = None,
        persistence_error_code: RecordingPersistenceErrorCodeV1 | None = None,
    ) -> RecordingStatusV1:
        """Build proposed lifecycle facts using the current accepted capture count."""
        recorder = self._recorder
        if recorder is None:
            raise AssertionError("recording status preview requires a recorder.")
        return recorder.preview_status_v1(
            captured_transition_count=recorder.validated_transition_count,
            lifecycle=lifecycle,
            close_cause=close_cause,
            completion_reason=completion_reason,
            persistence_error_code=persistence_error_code,
        )

    def _apply_recording_lifecycle_command(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        command: object,
    ) -> ServiceCommandResult | None:
        """Handle save, review, retry, and exit commands for an attached recorder.

        Return None when ordinary command dispatch should continue. Build response
        candidates before recorder mutations and retain a saved fallback if review
        construction fails.

        Exit while the recording is still open saves the game so far, then
        schedules shutdown. Exit before the first tick (the recording is open
        with zero captured ticks) writes no replay file: it schedules shutdown
        at once with the notice "No tick was played, so no replay was saved."
        and the unchanged recording status, the same rule as Ctrl-C
        (close_recording_for_keyboard_interrupt) and Restart.
        """
        recorder = self._recorder
        if recorder is None:
            return None
        if isinstance(
            command, (ConfirmDiscardAndReplaceCommandV1, SaveAndReplaceCommandV1)
        ):
            return None
        if isinstance(command, FinishAndReviewCommandV1):
            if recorder.lifecycle != "recording":
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice="Finish And Save is unavailable in the current lifecycle.",
                )
            success, failures = self._prepare_publication_responses(
                command_key=command_key,
                fingerprint=fingerprint,
                close_cause="finish_and_review",
                success_notice=(
                    "Replay saved. Open Replay to inspect it or Restart to play again."
                ),
                failure_notice="Replay save failed; choose Retry Save or Save As.",
            )
            outcome = recorder.finalize_and_save("finish_and_review")
            return self._commit_publication_outcome(
                outcome=outcome,
                success=success,
                failures=failures,
            )
        if isinstance(command, ReviewReplayCommandV1):
            if recorder.lifecycle != "saved":
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice="Replay review is unavailable until saving succeeds.",
                )
            reviewing = self._prepare_recording_response(
                command_key=command_key,
                fingerprint=fingerprint,
                recording_status=recorder.status,
                result="applied",
                notice="The saved replay is open in a separate Replay Viewer.",
                changed=False,
            )
            saved_fallback = self._prepare_recording_response(
                command_key=command_key,
                fingerprint=fingerprint,
                recording_status=recorder.status,
                result="no_op",
                notice=(
                    "The replay remains saved, but review could not open; retry "
                    "Open Replay."
                ),
                changed=False,
            )
            try:
                handoff = self._build_replay_handoff()
            except Exception:
                return self._commit_recording_response(saved_fallback)
            return self._commit_recording_response(
                reviewing,
                replay_handoff=handoff,
            )
        if isinstance(command, RetrySaveCommandV1):
            if recorder.lifecycle != "persistence_failed":
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice="Retry is unavailable because no save failure is pending.",
                )
            success, failures = self._prepare_publication_responses(
                command_key=command_key,
                fingerprint=fingerprint,
                close_cause=recorder.close_cause,
                completion_reason=recorder.status.completion_reason,
                success_notice=(
                    "Replay saved. Open Replay to inspect it or Restart to play again."
                ),
                failure_notice="Replay save still failed; try Save As.",
            )
            return self._commit_publication_outcome(
                outcome=recorder.retry_save(),
                success=success,
                failures=failures,
            )
        if isinstance(command, SaveAsCommandV1):
            if recorder.lifecycle != "persistence_failed":
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice="Save As is unavailable because no save failure is pending.",
                )
            success, failures = self._prepare_publication_responses(
                command_key=command_key,
                fingerprint=fingerprint,
                close_cause=recorder.close_cause,
                completion_reason=recorder.status.completion_reason,
                success_notice=(
                    "Replay saved under the new name. Open Replay to inspect it."
                ),
                failure_notice="Replay could not be saved under the requested name.",
            )
            return self._commit_publication_outcome(
                outcome=recorder.save_as(command.file_name),
                success=success,
                failures=failures,
            )
        if isinstance(command, ExitCommandV1):
            if (
                recorder.lifecycle == "recording"
                and recorder.validated_transition_count == 0
            ):
                # No tick was played: an empty replay holds nothing to watch,
                # so Exit closes without a file, as Ctrl-C and Restart do.
                closing = self._prepare_recording_response(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    recording_status=recorder.status,
                    result="shutdown_scheduled",
                    notice="No tick was played, so no replay was saved.",
                    changed=False,
                    shutdown_requested=True,
                )
                return self._commit_recording_response(closing)
            if recorder.lifecycle == "recording":
                success, failures = self._prepare_publication_responses(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    close_cause="user_exit",
                    success_notice="Replay saved; debugger shutdown requested.",
                    failure_notice=(
                        "Replay save failed; the debugger remains open for recovery."
                    ),
                    success_result="shutdown_scheduled",
                    shutdown_requested=True,
                )
                outcome = recorder.finalize_and_save("user_exit")
            elif recorder.lifecycle == "persistence_failed":
                success, failures = self._prepare_publication_responses(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    close_cause=recorder.close_cause,
                    completion_reason=recorder.status.completion_reason,
                    success_notice="Replay saved; debugger shutdown requested.",
                    failure_notice=(
                        "Replay save failed; the debugger remains open for recovery."
                    ),
                    success_result="shutdown_scheduled",
                    shutdown_requested=True,
                )
                outcome = recorder.retry_save()
            elif recorder.lifecycle in ("saved", "reviewing"):
                shutdown = self._prepare_recording_response(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    recording_status=recorder.status,
                    result="shutdown_scheduled",
                    notice="Replay saved; debugger shutdown requested.",
                    changed=False,
                    shutdown_requested=True,
                )
                return self._commit_recording_response(shutdown)
            else:
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice=(
                        "Exit is unavailable while recording closeout is incomplete."
                    ),
                )
            if outcome == "persistence_failed":
                return self._commit_recording_response(
                    self._failure_response_for_current_status(failures)
                )
            return self._commit_recording_response(success)
        return None

    @staticmethod
    def _closed_recording_command_is_allowed(command: object) -> bool:
        """Allow only lifecycle and presentation authority after capture closes."""
        if isinstance(
            command,
            (
                SetViewCommandV1,
                SetPresetCommandV1,
                ResetCommandV1,
                SetCombatConfigurationCommandV1,
                ConfirmDiscardAndReplaceCommandV1,
                SaveAndReplaceCommandV1,
            ),
        ):
            return True
        if isinstance(command, RosterSelectionCommandV1):
            return command.role == "control"
        if not isinstance(command, KeyboardCommandV1):
            return False
        return normalize_key(
            command.key, shift_key=command.shift_key
        ) in _CLOSED_RECORDING_PRESENTATION_KEYS | {"r"}

    def _prepare_publication_responses(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        close_cause: DebuggerRecordingCloseCauseV1 | None,
        completion_reason: str | None = None,
        success_notice: str,
        failure_notice: str,
        success_result: Literal["applied", "shutdown_scheduled"] = "applied",
        shutdown_requested: bool = False,
    ) -> tuple[
        _RecordingResponseCandidate,
        dict[RecordingPersistenceErrorCodeV1, _RecordingResponseCandidate],
    ]:
        """Validate the saved response and each expected save error before I/O."""
        success = self._prepare_recording_response(
            command_key=command_key,
            fingerprint=fingerprint,
            recording_status=self._preview_recording_status(
                lifecycle="saved",
                close_cause=close_cause,
                completion_reason=completion_reason,
            ),
            result=success_result,
            notice=success_notice,
            changed=True,
            shutdown_requested=shutdown_requested,
        )
        error_codes: tuple[RecordingPersistenceErrorCodeV1, ...] = (
            "target_unavailable",
            "publication_failed",
            "verification_failed",
        )
        failures: dict[
            RecordingPersistenceErrorCodeV1,
            _RecordingResponseCandidate,
        ] = {
            error_code: self._prepare_recording_response(
                command_key=command_key,
                fingerprint=fingerprint,
                recording_status=self._preview_recording_status(
                    lifecycle="persistence_failed",
                    close_cause=close_cause,
                    completion_reason=completion_reason,
                    persistence_error_code=error_code,
                ),
                result="applied",
                notice=failure_notice,
                changed=True,
            )
            for error_code in error_codes
        }
        return success, failures

    def _failure_response_for_current_status(
        self,
        candidates: dict[
            RecordingPersistenceErrorCodeV1,
            _RecordingResponseCandidate,
        ],
    ) -> _RecordingResponseCandidate:
        """Select the prebuilt response matching the recorder's public save error."""
        recorder = self._recorder
        if recorder is None or recorder.persistence_error_code is None:
            raise AssertionError(
                "persistence failure did not expose a canonical error code."
            )
        return candidates[recorder.persistence_error_code]

    def _commit_publication_outcome(
        self,
        *,
        outcome: Literal["saved", "persistence_failed"],
        success: _RecordingResponseCandidate,
        failures: dict[
            RecordingPersistenceErrorCodeV1,
            _RecordingResponseCandidate,
        ],
    ) -> ServiceCommandResult:
        """Install the save result while keeping the live match available."""
        if outcome == "persistence_failed":
            return self._commit_recording_response(
                self._failure_response_for_current_status(failures)
            )
        return self._commit_recording_response(success)

    def _finalize_endpoint_after_transition(
        self,
        *,
        candidates: _EndpointResponseCandidates,
    ) -> ServiceCommandResult:
        """Save a committed sealed capture and install its prevalidated endpoint
        response.
        """
        recorder = self._recorder
        if recorder is None or recorder.lifecycle != "sealed":
            raise AssertionError("endpoint finalization requires a sealed recorder.")
        close_cause = recorder.close_cause
        if close_cause not in ("endpoint", "truncation"):
            raise AssertionError("sealed recorder has an invalid close cause.")
        outcome = recorder.finalize_and_save(close_cause)
        if outcome == "persistence_failed":
            return self._install_endpoint_response(
                self._failure_response_for_current_status(candidates.failures)
            )
        return self._install_endpoint_response(candidates.saved)

    def _install_endpoint_response(
        self,
        candidate: _RecordingResponseCandidate,
    ) -> ServiceCommandResult:
        """Install endpoint save status only at the already committed transition
        revision.
        """
        self._acknowledge_saved_system_prefix()
        if candidate.revision != self._revision:
            raise AssertionError("endpoint response revision drifted after commit.")
        self._frame = candidate.frame
        self._command_records = candidate.command_records
        return ServiceCommandResult(
            outcome="response",
            payload=candidate.response,
        )

    def _prepare_endpoint_responses(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        session: DebuggerSession,
        revision: int,
        view_mode: ViewMode,
        preset: Preset,
        transition_close_status: RecordingStatusV1,
    ) -> _EndpointResponseCandidates:
        """Validate endpoint success and save-failure frames before accepting the
        transition.
        """
        if transition_close_status.lifecycle != "sealed":
            raise ValueError("endpoint response candidates require sealed status.")
        completion_state = transition_close_status.completion_state
        close_cause: DebuggerRecordingCloseCauseV1 = (
            "endpoint" if completion_state == "complete" else "truncation"
        )
        completion_reason = transition_close_status.completion_reason

        def prepare(
            *,
            status: RecordingStatusV1,
            notice: str,
        ) -> _RecordingResponseCandidate:
            """Build one endpoint response from proposed recording status and public
            notice.
            """
            frame = self._build_frame(
                session=session,
                revision=revision,
                view_mode=view_mode,
                preset=preset,
                recording_status=status,
            )
            self._build_authorized_presentation(
                session=session,
                raw_frame=frame,
                view_mode=view_mode,
            )
            response = CommandResponseV2(
                result="applied",
                frame=frame,
                notice=notice,
            )
            records = self._command_records.copy()
            self._remember_command_in(
                records,
                command_key,
                _CommandRecord(
                    fingerprint=fingerprint,
                    shutdown_requested=False,
                ),
            )
            return _RecordingResponseCandidate(
                revision=revision,
                frame=frame,
                response=response,
                command_records=records,
                shutdown_requested=False,
            )

        recorder = self._recorder
        if recorder is None:
            raise AssertionError("endpoint response preview requires a recorder.")
        saved = prepare(
            status=recorder.preview_status_v1(
                captured_transition_count=transition_close_status.captured_transition_count,
                lifecycle="saved",
                close_cause=close_cause,
                completion_reason=completion_reason,
            ),
            notice="The episode ended and its replay was saved.",
        )
        error_codes: tuple[RecordingPersistenceErrorCodeV1, ...] = (
            "target_unavailable",
            "publication_failed",
            "verification_failed",
        )
        failures: dict[
            RecordingPersistenceErrorCodeV1,
            _RecordingResponseCandidate,
        ] = {
            error_code: prepare(
                status=recorder.preview_status_v1(
                    captured_transition_count=(
                        transition_close_status.captured_transition_count
                    ),
                    lifecycle="persistence_failed",
                    close_cause=close_cause,
                    completion_reason=completion_reason,
                    persistence_error_code=error_code,
                ),
                notice=(
                    "The episode ended, but replay saving failed; choose Retry "
                    "Save or Save As."
                ),
            )
            for error_code in error_codes
        }
        return _EndpointResponseCandidates(saved=saved, failures=failures)

    def _prepare_failure_closeout_responses(
        self,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        close_cause: Literal[
            "simulation_failure",
            "policy_failure",
            "capture_failure",
            "validation_failure",
            "processing_failure",
        ],
        session: DebuggerSession | None = None,
        captured_transition_count: int | None = None,
    ) -> _FailureCloseoutCandidates:
        """Validate save outcomes for the accepted capture prefix before failure
        closeout.
        """
        recorder = self._recorder
        if recorder is None:
            raise AssertionError("failure closeout requires a recorder.")
        count = (
            recorder.validated_transition_count
            if captured_transition_count is None
            else captured_transition_count
        )
        resolved_session = self._session if session is None else session
        candidate_revision = self._revision + 1
        completion_reason = (
            "evaluation_processing_failure"
            if close_cause == "processing_failure"
            else close_cause
        )
        validated_presentation_fields: dict[str, object] | None = None

        def prepare(
            *,
            lifecycle: Literal["saved", "persistence_failed"],
            notice: str,
            persistence_error_code: RecordingPersistenceErrorCodeV1 | None = None,
        ) -> _RecordingResponseCandidate:
            """Build and check one save outcome, reusing unchanged presentation
            validation only within this closeout group.
            """
            nonlocal validated_presentation_fields
            status = recorder.preview_status_v1(
                captured_transition_count=count,
                lifecycle=lifecycle,
                close_cause=close_cause,
                completion_reason=completion_reason,
                persistence_error_code=persistence_error_code,
            )
            frame = self._build_frame(
                session=resolved_session,
                revision=candidate_revision,
                recording_status=status,
            )
            # Recording status is validated by the raw frame and is not part of
            # the authorized presentation. All other fields, including Run state,
            # must match before reusing this group's presentation validation.
            presentation_fields = frame.model_dump(exclude={"recording"})
            if presentation_fields != validated_presentation_fields:
                self._build_authorized_presentation(
                    session=resolved_session,
                    raw_frame=frame,
                    view_mode=self._view_mode,
                )
                validated_presentation_fields = presentation_fields
            response = CommandResponseV2(
                result="applied",
                frame=frame,
                notice=notice,
            )
            records = self._command_records.copy()
            self._remember_command_in(
                records,
                command_key,
                _CommandRecord(
                    fingerprint=fingerprint,
                    shutdown_requested=False,
                ),
            )
            return _RecordingResponseCandidate(
                revision=candidate_revision,
                frame=frame,
                response=response,
                command_records=records,
                shutdown_requested=False,
            )

        saved = prepare(
            lifecycle="saved",
            notice=(
                "The transition failed; the last validated replay prefix was saved."
            ),
        )
        error_codes: tuple[RecordingPersistenceErrorCodeV1, ...] = (
            "target_unavailable",
            "publication_failed",
            "verification_failed",
        )
        failures: dict[
            RecordingPersistenceErrorCodeV1,
            _RecordingResponseCandidate,
        ] = {
            error_code: prepare(
                lifecycle="persistence_failed",
                persistence_error_code=error_code,
                notice=(
                    "The transition failed and replay saving also failed; choose "
                    "Retry Save or Save As."
                ),
            )
            for error_code in error_codes
        }
        return _FailureCloseoutCandidates(saved=saved, failures=failures)

    def _close_failed_recording(
        self,
        *,
        close_cause: Literal[
            "simulation_failure",
            "policy_failure",
            "capture_failure",
            "validation_failure",
            "processing_failure",
        ],
        candidates: _FailureCloseoutCandidates,
    ) -> ServiceCommandResult:
        """Save the accepted prefix, install its result, and fence unexpected
        closeout failures.
        """
        recorder = self._recorder
        if recorder is None:
            raise AssertionError("failure closeout requires a recorder.")
        try:
            outcome = recorder.finalize_and_save(close_cause)
        except Exception:
            self._faulted = True
            self._command_records = candidates.saved.command_records
            raise
        if outcome == "persistence_failed":
            return self._commit_recording_response(
                self._failure_response_for_current_status(candidates.failures)
            )
        return self._commit_recording_response(candidates.saved)

    def apply_loaded_setup(
        self,
        request: CommandRequestV1,
        *,
        setup_identity: str,
        expected_session: DebuggerSession,
        load: Callable[[], LoadedControllers],
        build: Callable[[LoadedControllers], DebuggerSession],
        on_installed: Callable[[], None] | None = None,
    ) -> ServiceCommandResult:
        """Queue a validated private setup through normal request and save guards.

        Parameters
        ----------
        request : CommandRequestV1
            Host-created Reset, Save-And-Replace Reset, or confirmed-discard Reset,
            with the browser's
            client ID, unique command ID and current revision.
        setup_identity : str
            Canonical text of the validated private source request. It joins the
            duplicate fingerprint; changed setup under the same ID is rejected.
        expected_session : DebuggerSession
            Accepted snapshot used for validation. A replaced snapshot is stale.
        load : callable
            Trusted zero-argument resolver returning LoadedControllers. Browser
            strings must already be built-ins, checkpoint Paths or declared
            aliases. It runs only on the existing worker, outside HTTP locks.
        build : callable
            Initialize a fresh session from those exact loaded assignments, using
            the existing control owner. Fixed scenario restrictions still apply.
        on_installed : callable or None, optional
            Publish private setup choices after adoption. None adds no callback.

        Returns
        -------
        ServiceCommandResult
            Pending work or the existing stale/duplicate/recording rejection.
            This call never stores a loader registry or runs a method under lock.
        """
        effective = (
            request.command.replacement
            if isinstance(
                request.command,
                (ConfirmDiscardAndReplaceCommandV1, SaveAndReplaceCommandV1),
            )
            else request.command
        )
        if not isinstance(effective, ResetCommandV1):
            raise ValueError("A loaded setup must be an explicit new-match request")
        if not isinstance(cast(object, setup_identity), str) or not setup_identity:
            raise ValueError("A loaded setup needs its validated request identity")
        return self.apply_command(
            request,
            _loaded_setup=_LoadedSetup(
                setup_identity, expected_session, load, build, on_installed
            ),
        )

    def apply_command(
        self, request: CommandRequestV1, *, _loaded_setup: _LoadedSetup | None = None
    ) -> ServiceCommandResult:
        """Apply one revision-matched command under the service lock.

        Parameters
        ----------
        request : CommandRequestV1
            Validated request containing client ID, command ID, base revision, and
            one supported live or recording command.
        _loaded_setup : _LoadedSetup or None, optional
            Internal call-scoped host setup from apply_loaded_setup. HTTP bindings
            never supply this argument. None keeps ordinary command behavior.

        Returns
        -------
        ServiceCommandResult
            Response or typed rejection with the current frame. A saved replay may
            include a prepared replay-service handoff. Shutdown is only requested
            in the result; the HTTP host owns stopping its request loop.

        Notes
        -----
        The latest 256 request IDs are retained. An exact repeat while cached returns
        the current frame without repeating the command; changed content under that
        ID is rejected. Stale revisions, shutdown, and fault state reject new work.
        Step and reset commands may execute JAX work; recording commands may write
        files. Candidate response validation precedes installation. Unexpected
        internal failures can fence further commands and propagate. With
        save_in_background, the step that ends the game returns before its replay
        is saved (recording lifecycle "sealed"); a later command that reads or
        replaces the recording (_command_needs_saved_recording) first waits for
        that save and then continues from its result, and a Discard chosen during
        the save restarts without discarding, because the replay was saved.
        """
        command_key = (request.client_id, request.command_id)
        fingerprint = request.model_dump_json()
        if _loaded_setup is not None:
            fingerprint += "\0" + _loaded_setup.identity
        with self._lock:
            previous = self._command_records.get(command_key)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    return ServiceCommandResult(
                        outcome="command_id_conflict",
                        payload=ApiErrorV2(
                            error_code="command_id_conflict",
                            message=(
                                "This client reused a command_id for a "
                                "different request."
                            ),
                            latest_frame=self._frame,
                        ),
                    )
                self._command_records.move_to_end(command_key)
                return ServiceCommandResult(
                    outcome="response",
                    payload=CommandResponseV2(
                        result="duplicate",
                        frame=self._frame,
                        notice="Command already processed; current frame returned.",
                    ),
                    shutdown_requested=previous.shutdown_requested,
                )

            if self._faulted:
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                return ServiceCommandResult(
                    outcome="service_faulted",
                    payload=ApiErrorV2(
                        error_code="internal_error",
                        message=(
                            "The debugger entered a safe fault state; restart it "
                            "before sending another command."
                        ),
                        latest_frame=self._frame,
                    ),
                )

            if self._shutting_down:
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                return ServiceCommandResult(
                    outcome="server_shutting_down",
                    payload=ApiErrorV2(
                        error_code="server_shutting_down",
                        message=(
                            "The debugger is shutting down; this command was not "
                            "applied."
                        ),
                        latest_frame=self._frame,
                    ),
                )

            pause = (
                request.command if isinstance(request.command, PauseCommandV1) else None
            )
            matching_pause = (
                pause is not None
                and pause.session_id == self._session_id
                and pause.run_generation == self._session.run_generation
                and request.base_revision <= self._revision
            )
            if (
                (request.base_revision != self._revision and not matching_pause)
                or (pause is not None and not matching_pause)
            ) or (
                _loaded_setup is not None
                and _loaded_setup.expected_session is not self._session
            ):
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                return ServiceCommandResult(
                    outcome="stale_revision",
                    payload=ApiErrorV2(
                        error_code="stale_revision",
                        message=(
                            "The debugger advanced after this client frame; "
                            "the latest frame is attached."
                        ),
                        latest_frame=self._frame,
                    ),
                )

            command = request.command
            if isinstance(command, (RunCommandV1, PauseCommandV1)):
                return self._apply_run_control(command, command_key, fingerprint)
            if isinstance(
                command,
                (CancelSystemCommandV1, FinishAndReviewCommandV1, ExitCommandV1),
            ):
                self._running = False
            if isinstance(command, (FinishSystemCommandV1, CancelSystemCommandV1)):
                return self._finish_system_command(command, command_key, fingerprint)
            # A command that reads or replaces the recording waits for a running
            # game-over save, then goes on from the "saved" (or failed) state.
            # It passed the revision check above, so the save's own revision
            # does not make it stale.
            waited_for_save = False
            if self._background_save is not None and (
                self._command_needs_saved_recording(command)
            ):
                self._await_background_save()
                waited_for_save = True
                if self._faulted:
                    self._remember_command(
                        command_key,
                        _CommandRecord(
                            fingerprint=fingerprint,
                            shutdown_requested=False,
                        ),
                    )
                    return ServiceCommandResult(
                        outcome="service_faulted",
                        payload=ApiErrorV2(
                            error_code="internal_error",
                            message=(
                                "The replay save failed unexpectedly; restart the "
                                "BattleClient before sending another command."
                            ),
                            latest_frame=self._frame,
                        ),
                    )
            effective = (
                command.replacement
                if isinstance(
                    command,
                    (ConfirmDiscardAndReplaceCommandV1, SaveAndReplaceCommandV1),
                )
                else command
            )
            if self._session.scenario.scenario_specification is not None and isinstance(
                effective, (SetCombatConfigurationCommandV1, ScenarioSwitchCommandV1)
            ):
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice=(
                        SCENARIO_OPPONENT_ERROR
                        if isinstance(effective, SetCombatConfigurationCommandV1)
                        else "Scenarios are fixed. "
                        "Start a new BattleClient to change the source."
                    ),
                )
            if (
                self._system_worker is not None
                and self._system_worker.caller_thread
                and isinstance(
                    effective,
                    (SetCombatConfigurationCommandV1, ScenarioSwitchCommandV1),
                )
            ):
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice="Launch assignments are fixed. Start a new BattleClient.",
                )
            if isinstance(effective, SetCombatConfigurationCommandV1):
                selected = (effective.team_a_controller, effective.team_b_controller)
                if any(
                    is_system_controller(value)
                    and value.removeprefix("system:") not in self._offered_systems
                    for value in selected
                ):
                    return self._recording_no_op(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        notice="Choose a System declared when DevClient started.",
                    )
            worker = self._system_worker
            operation = None if worker is None else worker.operation
            inspect_failed_actor = (
                self._requires_fresh_launch()
                and self._frame.system_operation is not None
                and self._frame.system_operation.state == "failed"
                and isinstance(command, RosterSelectionCommandV1)
                and command.role == "control"
            )
            if (
                operation is not None
                and not inspect_failed_actor
                and not isinstance(command, (SetViewCommandV1, SetPresetCommandV1))
            ):
                if isinstance(command, (ExitCommandV1, FinishAndReviewCommandV1)):
                    assert worker is not None
                    worker.cancel(operation)
                else:
                    return self._system_response(
                        command_key,
                        fingerprint,
                        "A System is still working. "
                        "Cancel it before changing the match or drafts.",
                    )
            confirmed_discard = False
            saved_replacement = False
            if (
                _loaded_setup is not None
                and self._recorder is None
                and isinstance(command, ConfirmDiscardAndReplaceCommandV1)
            ):
                return self._recording_no_op(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    notice="There is no recorded prefix to discard.",
                )
            if self._recorder is not None:
                lifecycle_result = self._apply_recording_lifecycle_command(
                    command_key=command_key,
                    fingerprint=fingerprint,
                    command=command,
                )
                if lifecycle_result is not None:
                    return lifecycle_result
                if (
                    self._recorder.lifecycle != "recording"
                    and not self._closed_recording_command_is_allowed(command)
                ):
                    return self._recording_no_op(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        notice=(
                            "Scientific controls are fenced because this recording "
                            "is no longer capturing transitions."
                        ),
                    )
                save_requested = isinstance(command, SaveAndReplaceCommandV1)
                confirmed_discard = isinstance(
                    command, ConfirmDiscardAndReplaceCommandV1
                ) and not (
                    # Discard chosen while the game-over save ran: the save won,
                    # so the replay stays saved and the restart simply goes on.
                    waited_for_save
                    and self._recorder.lifecycle in ("saved", "reviewing")
                )
                if isinstance(
                    command,
                    (ConfirmDiscardAndReplaceCommandV1, SaveAndReplaceCommandV1),
                ):
                    command = command.replacement
                restart_intent = recording_restart_intent_v1(
                    self._session,
                    command,
                    view_mode=self._view_mode,
                    include_stress=self._include_stress,
                )
                if (confirmed_discard or save_requested) and (
                    restart_intent is None
                    or (
                        confirmed_discard
                        and not self._recorder.status.discard_available
                    )
                ):
                    return self._recording_no_op(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        notice=(
                            "This confirmation does not describe "
                            "an available replacement."
                        ),
                    )
                if save_requested and restart_intent is not None:
                    self._running = False
                    recorder = self._recorder
                    if recorder.lifecycle not in ("saved", "reviewing"):
                        cause = recorder.close_cause or "finish_and_review"
                        success, failures = self._prepare_publication_responses(
                            command_key=command_key,
                            fingerprint=fingerprint,
                            close_cause=cause,
                            completion_reason=recorder.status.completion_reason,
                            success_notice=("Replay saved. Preparing the next match."),
                            failure_notice=(
                                "Replay save failed. The match is retained; "
                                "choose Retry Save or Save As."
                            ),
                        )
                        outcome = (
                            recorder.retry_save()
                            if recorder.lifecycle == "persistence_failed"
                            else recorder.finalize_and_save(cause)
                        )
                        saved_result = self._commit_publication_outcome(
                            outcome=outcome,
                            success=success,
                            failures=failures,
                        )
                        if outcome != "saved":
                            return saved_result
                saved_replacement = (
                    restart_intent is not None
                    and self._recorder.lifecycle in ("saved", "reviewing")
                )
                if (
                    not confirmed_discard
                    and restart_intent is not None
                    and self._recorder.status.restart_fenced
                ):
                    return self._recording_no_op(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        notice=(
                            "This recording has unsaved progress. Choose Save, "
                            "Discard or Cancel before replacing the match."
                        ),
                    )

            asynchronous = self._start_system_command(
                command,
                command_key,
                fingerprint,
                confirmed_discard,
                loaded_setup=_loaded_setup,
            )
            if asynchronous is not None:
                return asynchronous
            try:
                dispatched = dispatch_command(
                    self._session,
                    command,
                    view_mode=self._view_mode,
                    preset=self._preset,
                    include_stress=self._include_stress,
                    pov_global_slot=self._pov_global_slot,
                )
            except DebuggerTransitionFailureV1 as error:
                recorder = self._recorder
                if recorder is None:
                    self._faulted = True
                    self._remember_command(
                        command_key,
                        _CommandRecord(
                            fingerprint=fingerprint,
                            shutdown_requested=False,
                        ),
                    )
                    raise
                return self._close_transition_failure(error, command_key, fingerprint)
            except Exception:
                if saved_replacement:
                    logging.getLogger(__name__).exception(
                        "Saved match replacement failed"
                    )
                    return self._recording_no_op(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        notice=(
                            "Replay saved, but the next match could not start. "
                            "The accepted match is unchanged."
                        ),
                    )
                self._faulted = True
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                raise

            return self._accept_dispatch_result(
                dispatched,
                command_key=command_key,
                fingerprint=fingerprint,
                confirmed_discard=confirmed_discard,
            )

    def _close_transition_failure(
        self,
        error: DebuggerTransitionFailureV1,
        command_key: tuple[str, str],
        fingerprint: str,
    ) -> ServiceCommandResult:
        """Use the recorder's existing failure cause and accepted-prefix closeout."""
        failure_causes: dict[
            DebuggerTransitionFailureStageV1,
            Literal[
                "simulation_failure",
                "policy_failure",
                "capture_failure",
                "validation_failure",
            ],
        ] = {
            "action_build": "policy_failure",
            "simulation": "simulation_failure",
            "capture": "capture_failure",
            "validation": "validation_failure",
        }
        close_cause: Literal[
            "simulation_failure",
            "policy_failure",
            "capture_failure",
            "validation_failure",
        ] = failure_causes[error.stage]
        candidates = self._prepare_failure_closeout_responses(
            command_key=command_key,
            fingerprint=fingerprint,
            close_cause=close_cause,
        )
        return self._close_failed_recording(
            close_cause=close_cause,
            candidates=candidates,
        )

    @property
    def system_evidence_directory(self) -> Path | None:
        """Return the current match's call folder for host tools, never the browser."""
        installation = self._system_evidence
        return None if installation is None else installation.directory

    def _requires_fresh_launch(self) -> bool:
        """Keep the uncertain-method fence visible after pending cleanup finishes."""
        worker = self._system_worker
        return self._system_failed or (worker is not None and worker.failed)

    def _run_state_for(self, frame: LiveDebuggerFrame) -> Literal["paused", "running"]:
        """Report Run only for this still-open, healthy accepted episode."""
        worker = self._system_worker
        operation_status = self._system_operation_status()
        return (
            "running"
            if self._running
            and frame.run_generation == self._session.run_generation
            and not frame.terminal.is_sealed
            and (frame.recording is None or frame.recording.lifecycle == "recording")
            and not frame.combat_configuration.manual_slots
            and not self._faulted
            and not self._shutting_down
            and not self._system_failed
            and worker is not None
            and not worker.failed
            and (operation_status is None or operation_status.state != "failed")
            else "paused"
        )

    def _apply_run_control(
        self,
        command: RunCommandV1 | PauseCommandV1,
        key: tuple[str, str],
        fingerprint: str,
    ) -> ServiceCommandResult:
        """Change Run intent after request identity and revision checks.

        Pause does not cancel the current turn. Run starts no callback here; the
        server pump submits it after this request releases its locks.
        """
        before = self._frame.run_state
        notice = "Paused. A turn already in progress may finish."
        if isinstance(command, PauseCommandV1):
            self._running = False
        else:
            worker = self._system_worker
            operation = None if worker is None else worker.operation
            operation_status = self._system_operation_status()
            if self._session.scenario.mode != "interactive":
                notice = "Run is available only for live interactive matches."
            elif self._session.manual_slots:
                notice = "Manual slots wait for Submit, including dead Manual agents."
            elif self._frame.terminal.is_sealed or (
                self._recorder is not None and self._recorder.lifecycle != "recording"
            ):
                notice = "This match has ended. Restart before running again."
            elif (
                self._system_failed
                or (worker is not None and worker.failed)
                or (operation_status is not None and operation_status.state == "failed")
            ):
                notice = "Method work failed. Start a fresh client launch."
            elif worker is None or worker.active is None or worker.active.closed:
                notice = "This match has no active method execution owner."
            elif operation is not None and operation.kind == "setup":
                notice = "Wait for the new match to finish loading."
            else:
                self._running = True
                notice = "Running. Pause stops the next turn."
        response = self._system_response(key, fingerprint, notice)
        if before != self._frame.run_state and isinstance(
            response.payload, CommandResponseV2
        ):
            return replace(
                response,
                payload=response.payload.model_copy(update={"result": "applied"}),
            )
        return response

    def pump(self) -> bool:
        """Advance queued live work without depending on a browser request.

        Call this on the original launch thread, outside HTTP/router locks. It
        drains at most one caller callback outside the service lock, adopts only
        ready results under that lock, then queues at most one autonomous turn.
        Owned private callbacks keep their existing worker Context. Return True
        when work advanced; False means the server may wait briefly. Paused games
        still finish already submitted work. No idle tick rebuilds a frame.
        """
        progressed = self.drain_system_work()
        with self._lock:
            worker = self._system_worker
            operation = None if worker is None else worker.operation
            if operation is not None:
                future = (
                    operation.acceptance_future
                    if operation.adopted
                    else operation.cleanup_future
                    if operation.cancelled
                    else operation.future
                )
                if (
                    future is not None
                    and future.done()
                    and not (
                        (operation.adopted or operation.cancelled)
                        and self._frame.system_operation is not None
                        and self._frame.system_operation.state == "failed"
                    )
                ):
                    self._pump_sequence += 1
                    request = CommandRequestV1(
                        client_id=self._pump_client_id,
                        command_id=str(self._pump_sequence),
                        base_revision=self._revision,
                        command=FinishSystemCommandV1(
                            operation_id=operation.identifier
                        ),
                    )
                    self.apply_command(request)
                    progressed = True
            if self._running and self._run_state_for(self._frame) != "running":
                self._running = False
                self._refresh_system_status(advance_revision=True)
                progressed = True
            if (
                self._running
                and worker is not None
                and worker.operation is None
                and worker.active is not None
                and not worker.active.closed
            ):
                self._pump_sequence += 1
                self.apply_command(
                    CommandRequestV1(
                        client_id=self._pump_client_id,
                        command_id=str(self._pump_sequence),
                        base_revision=self._revision,
                        command=KeyboardCommandV1(key="Enter"),
                    )
                )
                progressed = True
        return progressed

    def drain_system_work(self) -> bool:
        """Run one caller-thread callback without holding a service or router lock.

        The launch loop calls this on the original caller thread. False means no
        work is queued. HTTP handlers only enqueue work and never call this hook.
        """
        worker = self._system_worker
        return False if worker is None else worker.drain()

    def close(self) -> None:
        """Fence pending results and queue owned cleanup without stopping a server.

        Arbitrary Python factories can delay thread exit. The worker retains
        cleanup futures so host tools can inspect failures; supplied Systems are
        caller-owned and are not closed here. A running background save is
        waited for first, so its replay file and its saved-replay evidence are
        written before the worker closes.
        """
        with self._lock:
            self._await_background_save()
            self._running = False
            self._refresh_system_status(advance_revision=True)
            worker = self._system_worker
            if worker is not None:
                worker.close()
        if worker is not None and worker.caller_thread:
            while worker.drain():
                pass

    def _refresh_system_status(self, *, advance_revision: bool) -> None:
        """Update worker status while reusing the unchanged accepted display.

        Status-only commands increment the transport revision. Adoption already
        installed its new revision and complete frame. Researcher and NoSharedObs
        HUDs also display that revision, so their existing fact must stay in sync.
        Scientific, recording, view and draft changes still use _build_frame.
        """
        status = self._system_operation_status()
        run_state = self._run_state_for(self._frame)
        fresh_launch_required = self._requires_fresh_launch()
        if advance_revision and (
            self._frame.system_operation != status
            or self._frame.run_state != run_state
            or self._frame.fresh_launch_required != fresh_launch_required
        ):
            self._revision += 1
        if (
            self._frame.system_operation == status
            and self._frame.run_state == run_state
            and self._frame.fresh_launch_required == fresh_launch_required
            and self._frame.revision == self._revision
        ):
            return
        updates: dict[str, object] = {
            "revision": self._revision,
            "system_operation": status,
            "run_state": run_state,
            "fresh_launch_required": fresh_launch_required,
        }
        if self._frame.revision != self._revision and isinstance(
            self._frame, (ResearcherLiveDebuggerFrameV2, ActorPovLiveDebuggerFrameV2)
        ):
            updates["hud"] = self._frame.hud.model_copy(
                update={
                    "diagnostics": tuple(
                        fact.model_copy(update={"value": str(self._revision)})
                        if fact.fact_id == "revision"
                        else fact
                        for fact in self._frame.hud.diagnostics
                    )
                }
            )
        self._frame = self._frame.model_copy(update=updates)

    def _system_response(
        self, key: tuple[str, str], fingerprint: str, notice: str
    ) -> ServiceCommandResult:
        """Publish current operation status without changing the accepted game."""
        self._refresh_system_status(advance_revision=True)
        return self._recording_no_op(
            command_key=key, fingerprint=fingerprint, notice=notice
        )

    def _start_system_command(
        self,
        command: object,
        key: tuple[str, str],
        fingerprint: str,
        confirmed_discard: bool,
        *,
        loaded_setup: _LoadedSetup | None = None,
    ) -> ServiceCommandResult | None:
        """Submit selected setup or one joint turn; leave ordinary dispatch unchanged.

        A joint turn starts from an unmodified, non-repeated Space, Enter or N
        in interactive play (N is one tick, the same submission as Space).
        Called after recording replacement checks. The captured session and draft
        belong to this operation until an explicit completion command accepts it.
        No model call, factory or initialization runs under the service lock.
        """
        session = self._session
        restart = recording_restart_intent_v1(
            session,
            cast("DebuggerCommandV1", command),
            view_mode=self._view_mode,
            include_stress=self._include_stress,
        )
        if restart is not None:
            self._running = False
        controllers = (session.team_a_controller, session.team_b_controller)
        mode = session.evaluation_context.execution_information_mode
        if isinstance(command, SetCombatConfigurationCommandV1):
            controllers = (command.team_a_controller, command.team_b_controller)
            mode = command.execution_information_mode
        selected = any(is_system_controller(value) for value in controllers)
        worker = self._system_worker
        if (
            worker is not None
            and (worker.failed or self._system_failed)
            and (loaded_setup is not None or restart is not None)
        ):
            return self._system_response(
                key,
                fingerprint,
                "System work failed. Start a fresh client launch; "
                "the accepted replay can still be saved.",
            )
        if loaded_setup is not None:
            if worker is not None and worker.caller_thread:
                return self._system_response(
                    key,
                    fingerprint,
                    "Launch assignments are fixed. Start a new BattleClient.",
                )
            if worker is None:
                worker = self._system_worker = SystemWorker()
            self._system_confirmed_discard = confirmed_discard
            self._system_failed = False
            directory = (
                None
                if self._recorder is None
                else self._recorder.evidence_directory
                / f"generation-{session.run_generation + 1}"
            )
            operation = worker.prepare_loaded(
                session, loaded_setup.load, loaded_setup.build, directory=directory
            )
            operation.on_installed = loaded_setup.on_installed
            operation.prepared_replacement = True
            operation.replacement_recording_identity = self._recorder
            return replace(
                self._system_response(
                    key, fingerprint, "Loading the chosen controllers."
                ),
                queued_operation_id=operation.identifier,
            )
        if (
            worker is not None
            and restart is not None
            and (
                worker.caller_thread
                or (
                    restart == "reset"
                    and (
                        worker.active is not None
                        or worker.failed
                        or self._system_failed
                    )
                )
            )
        ):
            if isinstance(
                command, (SetCombatConfigurationCommandV1, ScenarioSwitchCommandV1)
            ):
                return self._system_response(
                    key,
                    fingerprint,
                    "Launch assignments are fixed. Start a new BattleClient.",
                )
            if worker.failed or self._system_failed:
                return self._system_response(
                    key,
                    fingerprint,
                    (
                        "System work failed. Start a fresh Python launch; "
                        if worker.caller_thread
                        else "System work failed. Start a fresh DevClient launch; "
                    )
                    + "the accepted replay can still be saved.",
                )
            if worker.operation is not None:
                return self._system_response(
                    key, fingerprint, "Wait for System cleanup before restarting."
                )
            self._system_confirmed_discard = confirmed_discard
            directory = (
                None
                if self._recorder is None
                else self._recorder.evidence_directory
                / f"generation-{session.run_generation + 1}"
            )
            operation = worker.restart(
                session, lambda: reset_session(session), directory=directory
            )
            operation.replacement_recording_identity = self._recorder
            return self._system_response(key, fingerprint, "Restarting the match.")
        if restart is not None and (
            selected or (worker is not None and worker.active is not None)
        ):
            if worker is None:
                worker = self._system_worker = SystemWorker()
            if worker.operation is not None:
                return self._system_response(
                    key,
                    fingerprint,
                    "Wait for System cleanup before loading a replacement.",
                )
            self._system_failed = False
            self._system_confirmed_discard = confirmed_discard
            directory = None
            if self._recorder is not None:
                directory = (
                    self._recorder.evidence_directory
                    / f"generation-{session.run_generation + 1}"
                )
            operation = worker.prepare(
                session,
                self._offered_systems,
                controllers,
                lambda pair: set_combat_configuration(
                    session,
                    team_a_controller=controllers[0],
                    team_b_controller=controllers[1],
                    execution_information_mode=mode,
                    systems=pair,
                ),
                directory=directory,
            )
            operation.replacement_recording_identity = self._recorder
            return self._system_response(
                key, fingerprint, "Loading the selected Systems."
            )
        submission = (
            isinstance(command, KeyboardCommandV1)
            and not (
                command.ctrl_key
                or command.alt_key
                or command.meta_key
                or command.repeat
            )
            # N is one tick: in interactive play it submits the turn as Space
            # and Enter do (scripted N stays the next registered frame).
            and normalize_key(command.key, shift_key=command.shift_key)
            in ("space", "enter", "n")
            and session.scenario.mode == "interactive"
        )
        if (
            submission
            and worker is not None
            and (
                selected
                or (
                    worker.active is not None
                    and any(value != "manual" for value in controllers)
                )
            )
        ):
            if (
                worker.active is None
                or worker.active.closed
                or self._system_failed
                or worker.failed
            ):
                return self._system_response(
                    key,
                    fingerprint,
                    (
                        "Start a fresh Python launch before playing again. "
                        if worker.caller_thread
                        else "Start a fresh DevClient launch before playing again. "
                    )
                    + "The accepted replay can still be saved.",
                )
            view, preset = self._view_mode, self._preset
            recipient = self._pov_global_slot
            worker.decide(
                session,
                lambda: dispatch_command(
                    session,
                    cast("DebuggerCommandV1", command),
                    view_mode=view,
                    preset=preset,
                    include_stress=self._include_stress,
                    pov_global_slot=recipient,
                ),
            )
            return self._system_response(
                key, fingerprint, "The Systems are choosing this turn."
            )
        return None

    def _finish_system_command(
        self,
        command: FinishSystemCommandV1 | CancelSystemCommandV1,
        key: tuple[str, str],
        fingerprint: str,
    ) -> ServiceCommandResult:
        """Poll, cancel or accept one operation without waiting on unfinished work."""
        worker = self._system_worker
        operation = None if worker is None else worker.operation
        if (
            worker is None
            or operation is None
            or operation.identifier != command.operation_id
        ):
            return self._system_response(
                key, fingerprint, "This System operation is no longer pending."
            )
        if isinstance(command, CancelSystemCommandV1) and not operation.adopted:
            worker.cancel(operation)
        try:
            if operation.adopted:
                complete = worker.finish_accept(operation)
                return self._system_response(
                    key,
                    fingerprint,
                    "System turn accepted."
                    if complete
                    else "Finishing System records and cleanup.",
                )
            if operation.cancelled:
                complete = worker.finish_cancel(operation)
                return self._system_response(
                    key,
                    fingerprint,
                    "System work failed. "
                    "Start a fresh client launch before playing again."
                    if complete and self._system_failed
                    else "System work cancelled. The accepted game is unchanged."
                    if complete
                    else "Waiting for cancelled System work to finish cleanup.",
                )
            if not operation.future.done():
                return self._system_response(
                    key, fingerprint, "The System is still working."
                )
            prepared = operation.future.result()
            base, current = operation.base, self._session
            if (
                base.raw_continuation_identity is not current.raw_continuation_identity
                or (
                    operation.kind == "setup"
                    and operation.replacement_recording_identity is not self._recorder
                )
                or base.pending_actions != current.pending_actions
                or base.evaluation_context is not current.evaluation_context
                or base.run_generation != current.run_generation
                or base.current_evaluation_frame is not current.current_evaluation_frame
            ):
                worker.cancel(operation)
                return self._system_response(
                    key,
                    fingerprint,
                    "The match changed; this System result was abandoned.",
                )
            if isinstance(prepared, SystemInstallation):
                assert prepared.session is not None
                candidate = prepared.session
                dispatched = InputDispatchResult(
                    candidate,
                    self._view_mode,
                    self._preset,
                    True,
                    True,
                    episode_restarted=True,
                    raw_continuation_identity=candidate.raw_continuation_identity,
                )
            else:
                dispatched = prepared
            candidate = replace(
                dispatched.session,
                controlled_global_slot=current.controlled_global_slot
                if dispatched.session.scenario is current.scenario
                else dispatched.session.controlled_global_slot,
                show_ranges=current.show_ranges,
                verbose_logging=current.verbose_logging,
            )
            dispatched = replace(
                dispatched,
                session=candidate,
                view_mode=self._view_mode,
                preset=self._preset,
            )
            result = self._accept_dispatch_result(
                dispatched,
                command_key=key,
                fingerprint=fingerprint,
                confirmed_discard=self._system_confirmed_discard
                if operation.kind == "setup"
                else False,
                system_operation=operation,
            )
            if not operation.adopted:
                worker.cancel(operation)
            self._refresh_system_status(advance_revision=False)
            if isinstance(result.payload, CommandResponseV2):
                result = replace(
                    result,
                    payload=result.payload.model_copy(update={"frame": self._frame}),
                )
            return result
        except DebuggerTransitionFailureV1 as error:
            self._running = False
            logging.getLogger(__name__).exception("DevClient System decision failed")
            worker.cancel(operation)
            self._system_failed = True
            if self._recorder is not None:
                return self._close_transition_failure(error, key, fingerprint)
            return self._system_response(
                key,
                fingerprint,
                "System decision failed. "
                "Start a fresh client launch before playing again.",
            )
        except Exception:
            self._running = False
            logging.getLogger(__name__).exception("DevClient System operation failed")
            if not operation.adopted:
                worker.cancel(operation)
            self._system_failed = operation.kind == "decision" or operation.adopted
            return self._system_response(
                key,
                fingerprint,
                "System work failed. The accepted game and replay remain available; "
                "check the host error log.",
            )

    def _acknowledge_saved_system_prefix(self) -> None:
        """Queue the recorder's verified save for its exact call-evidence owner."""
        recorder, worker, installation = (
            self._recorder,
            self._system_worker,
            self._system_evidence,
        )
        if recorder is None or worker is None or installation is None:
            return
        saved = recorder.saved_bundle
        if saved is not None:
            assert recorder.verified_loaded_bundle is not None
            worker.record_saved_prefix(
                installation=installation,
                replay_id=recorder.verified_loaded_bundle.replay.artifact_id,
                path=saved.replay_path,
                transitions=recorder.validated_transition_count,
            )

    def _adopt_system_operation(
        self, operation: PendingSystemOperation | None, session: DebuggerSession
    ) -> None:
        """Acknowledge each real adoption before later recording work can fail."""
        if operation is None:
            return
        worker = self._system_worker
        assert worker is not None
        self._system_evidence = (
            operation.installation if session.systems is not None else None
        )
        worker.accept(operation, session)
        if operation.on_installed is not None:
            operation.on_installed()

    def _accept_dispatch_result(
        self,
        dispatched: InputDispatchResult,
        *,
        command_key: tuple[str, str],
        fingerprint: str,
        confirmed_discard: bool = False,
        system_operation: PendingSystemOperation | None = None,
    ) -> ServiceCommandResult:
        """Validate and adopt one prepared result under the existing service lock.

        Both ordinary dispatch and later worker completion use this one path.
        command_key/fingerprint identify the request to remember. confirmed_discard
        permits the already approved replacement of an accepted recording prefix.
        Validate frames before installation; recorder acceptance remains the owner
        of saved progress, including processing errors after append succeeds.
        A transition that seals the recording (game over) saves here and returns
        the saved or failed response; with save_in_background it instead commits
        the game-over frame (lifecycle "sealed"), starts _start_background_save
        and returns at once.
        """
        candidate_session = dispatched.session
        candidate_view_mode = dispatched.view_mode
        candidate_preset = dispatched.preset
        candidate_revision = self._revision + int(dispatched.changed)
        previous_scientific_state = (
            self._session.scenario_name,
            self._session.seed,
            self._session.run_generation,
            self._session.scenario_default_movement_scale,
            self._session.evaluation_context,
            self._session.current_evaluation_frame,
            self._session.incoming_evaluation_view,
            self._session.status_source_evidence_state,
            self._session.last_submission_kind,
            self._session.last_report_actor_slots,
            self._session.next_script_frame_index,
        )
        candidate_scientific_state = (
            candidate_session.scenario_name,
            candidate_session.seed,
            candidate_session.run_generation,
            candidate_session.scenario_default_movement_scale,
            candidate_session.evaluation_context,
            candidate_session.current_evaluation_frame,
            candidate_session.incoming_evaluation_view,
            candidate_session.status_source_evidence_state,
            candidate_session.last_submission_kind,
            candidate_session.last_report_actor_slots,
            candidate_session.next_script_frame_index,
        )
        previous_continuation = (
            self._session.config,
            self._session.key,
            self._session.state,
            self._session.observation,
            self._session.action_mask,
            self._session.raw_continuation_identity,
        )
        candidate_continuation = (
            candidate_session.config,
            candidate_session.key,
            candidate_session.state,
            candidate_session.observation,
            candidate_session.action_mask,
            candidate_session.raw_continuation_identity,
        )
        previous_system_parts = (
            self._session.systems,
            self._session.system_ids,
            self._session.environment_state,
            self._session.system_memory,
        )
        candidate_system_parts = (
            candidate_session.systems,
            candidate_session.system_ids,
            candidate_session.environment_state,
            candidate_session.system_memory,
        )
        if dispatched.transition_applied is not None:
            if (
                candidate_session.systems is not self._session.systems
                or candidate_session.system_ids is not self._session.system_ids
            ):
                self._faulted = True
                raise RuntimeError("A transition cannot replace installed Systems")
            if (
                self._session.environment_state is not None
                and candidate_session.environment_state
                is self._session.environment_state
            ):
                self._faulted = True
                raise RuntimeError("A System transition needs its successor wrapper")
            transition_view = dispatched.transition_applied
            scenario = self._session.scenario
            expected_submission_kind: Literal["interactive", "scripted"] = (
                "scripted" if scenario.mode == "scripted" else "interactive"
            )
            expected_script_cursor = (
                self._session.next_script_frame_index + 1
                if scenario.mode == "scripted"
                else self._session.next_script_frame_index
            )
            expected_report_slots = (
                tuple(
                    sorted(
                        command.actor_global_slot
                        for command in scenario.frames[
                            self._session.next_script_frame_index
                        ].commands
                    )
                )
                if scenario.mode == "scripted"
                else tuple(
                    row.global_slot
                    for row in self._session.evaluation_context.roster
                    if row.configured_active
                )
            )
            if (
                transition_view.context != self._session.evaluation_context
                or transition_view.start_frame != self._session.current_evaluation_frame
                or transition_view.successor_frame
                != candidate_session.current_evaluation_frame
                or candidate_session.incoming_evaluation_view != transition_view
                or candidate_session.run_generation != self._session.run_generation
                or candidate_session.scenario_name != self._session.scenario_name
                or candidate_session.seed != self._session.seed
                or candidate_session.scenario_default_movement_scale
                != self._session.scenario_default_movement_scale
                or candidate_session.last_submission_kind != expected_submission_kind
                or candidate_session.last_report_actor_slots != expected_report_slots
                or candidate_session.next_script_frame_index != expected_script_cursor
                or dispatched.raw_continuation_identity
                is not candidate_session.raw_continuation_identity
                or candidate_session.config is not self._session.config
                or (
                    candidate_session.key is not self._session.key
                    if scenario.scenario_specification is not None
                    else candidate_session.key is self._session.key
                )
                or any(
                    candidate is previous
                    for candidate, previous in zip(
                        candidate_continuation[2:],
                        previous_continuation[2:],
                        strict=True,
                    )
                )
            ):
                self._faulted = True
                raise RuntimeError(
                    "transition marker does not describe exactly one session advance"
                )
        elif dispatched.episode_restarted:
            prepared = (
                None
                if system_operation is None
                else system_operation.installation.session
            )
            if (
                candidate_session.run_generation != self._session.run_generation + 1
                or candidate_session.current_evaluation_frame.frame_index != 0
                or candidate_session.incoming_evaluation_view is not None
                or candidate_session.evaluation_context.identity.episode_id
                == self._session.evaluation_context.identity.episode_id
                or (
                    candidate_session.seed != self._session.seed
                    and not (
                        system_operation is not None
                        and system_operation.prepared_replacement
                        and system_operation.kind == "setup"
                        and prepared is not None
                        and candidate_session.raw_continuation_identity
                        is prepared.raw_continuation_identity
                        and candidate_session.scenario is prepared.scenario
                        and candidate_session.seed == prepared.seed
                    )
                )
                or candidate_session.last_submission_kind is not None
                or candidate_session.last_report_actor_slots
                or candidate_session.next_script_frame_index != 0
                or dispatched.raw_continuation_identity
                is not candidate_session.raw_continuation_identity
                or any(
                    candidate is previous
                    for candidate, previous in zip(
                        candidate_continuation[1:],
                        previous_continuation[1:],
                        strict=True,
                    )
                )
            ):
                self._faulted = True
                raise RuntimeError(
                    "restart marker does not describe a fresh episode generation"
                )
        elif (
            any(
                candidate is not previous
                for candidate, previous in zip(
                    candidate_system_parts, previous_system_parts, strict=True
                )
            )
            or candidate_scientific_state != previous_scientific_state
            or any(
                candidate is not previous
                for candidate, previous in zip(
                    candidate_continuation,
                    previous_continuation,
                    strict=True,
                )
            )
        ):
            self._faulted = True
            raise RuntimeError(
                "scientific session state changed without transition or restart marker"
            )
        candidate_recorder = self._recorder
        endpoint_candidates: _EndpointResponseCandidates | None = None
        validation_failure_candidates: _FailureCloseoutCandidates | None = None
        processing_failure_candidates: _FailureCloseoutCandidates | None = None
        candidate_recording_status = self._current_recording_status()
        candidate_frame = self._frame
        if dispatched.changed:
            try:
                if self._recorder is not None:
                    if dispatched.transition_applied is not None:
                        candidate_recording_status = (
                            self._recorder.preview_status_after_append_v1(
                                dispatched.transition_applied.transition
                            )
                        )
                        validation_failure_candidates = (
                            self._prepare_failure_closeout_responses(
                                command_key=command_key,
                                fingerprint=fingerprint,
                                close_cause="validation_failure",
                            )
                        )
                        processing_failure_candidates = (
                            self._prepare_failure_closeout_responses(
                                command_key=command_key,
                                fingerprint=fingerprint,
                                close_cause="processing_failure",
                                session=candidate_session,
                                captured_transition_count=(
                                    self._recorder.validated_transition_count + 1
                                ),
                            )
                        )
                    elif dispatched.episode_restarted:
                        candidate_recorder = self._recorder.replacement_for(
                            candidate_session.evaluation_context,
                            candidate_session.current_evaluation_frame,
                        )
                        candidate_recording_status = candidate_recorder.status
                candidate_frame = self._build_frame(
                    session=candidate_session,
                    revision=candidate_revision,
                    view_mode=candidate_view_mode,
                    preset=candidate_preset,
                    recording_status=candidate_recording_status,
                )
                self._build_authorized_presentation(
                    session=candidate_session,
                    raw_frame=candidate_frame,
                    view_mode=candidate_view_mode,
                )
                if (
                    self._recorder is not None
                    and dispatched.transition_applied is not None
                    and candidate_recording_status is not None
                    and candidate_recording_status.lifecycle == "sealed"
                    # A background save shows its result in a later revision,
                    # so it needs no prebuilt saved or failed responses.
                    and not self._save_in_background
                ):
                    endpoint_candidates = self._prepare_endpoint_responses(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        session=candidate_session,
                        revision=candidate_revision,
                        view_mode=candidate_view_mode,
                        preset=candidate_preset,
                        transition_close_status=candidate_recording_status,
                    )
            except Exception:
                if (
                    dispatched.episode_restarted
                    and self._recorder is not None
                    and self._recorder.lifecycle in ("saved", "reviewing")
                ):
                    logging.getLogger(__name__).exception(
                        "Saved recording replacement failed"
                    )
                    return self._recording_no_op(
                        command_key=command_key,
                        fingerprint=fingerprint,
                        notice=(
                            "Replay saved, but the next match could not start. "
                            "The accepted match is unchanged."
                        ),
                    )
                self._faulted = True
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                raise

        result_kind: Literal[
            "applied",
            "no_op",
            "shutdown_scheduled",
        ] = (
            "shutdown_scheduled"
            if dispatched.shutdown_requested
            else "applied"
            if dispatched.changed
            else "no_op"
        )
        try:
            candidate_response = CommandResponseV2(
                result=result_kind,
                frame=candidate_frame,
                notice=dispatched.notice,
            )
            candidate_record = _CommandRecord(
                fingerprint=fingerprint,
                shutdown_requested=dispatched.shutdown_requested,
            )
            candidate_result = ServiceCommandResult(
                outcome="response",
                payload=candidate_response,
                shutdown_requested=dispatched.shutdown_requested,
            )
            candidate_command_records = self._command_records.copy()
            self._remember_command_in(
                candidate_command_records,
                command_key,
                candidate_record,
            )
        except Exception:
            self._faulted = True
            self._remember_command(
                command_key,
                _CommandRecord(
                    fingerprint=fingerprint,
                    shutdown_requested=False,
                ),
            )
            raise

        if dispatched.transition_applied is not None:
            transition_view = dispatched.transition_applied
            if (
                candidate_session.current_evaluation_frame
                != transition_view.successor_frame
                or candidate_session.evaluation_context != transition_view.context
                or transition_view.context != self._session.evaluation_context
                or transition_view.start_frame != self._session.current_evaluation_frame
                or self._validated_transition_count()
                != transition_view.transition.transition_index
            ):
                self._faulted = True
                self._remember_command(
                    command_key,
                    _CommandRecord(
                        fingerprint=fingerprint,
                        shutdown_requested=False,
                    ),
                )
                raise RuntimeError(
                    "candidate transition and committed observer epoch diverged"
                )
            try:
                if self._recorder is not None:
                    self._recorder.append(
                        transition_view.transition,
                        candidate_session.current_evaluation_frame,
                    )
            except Exception as error:
                recorder = self._recorder
                if recorder is None:
                    self._faulted = True
                    self._remember_command(
                        command_key,
                        _CommandRecord(
                            fingerprint=fingerprint,
                            shutdown_requested=False,
                        ),
                    )
                    raise
                previous_count = transition_view.transition.transition_index
                if recorder.validated_transition_count == previous_count:
                    if validation_failure_candidates is None:
                        raise AssertionError(
                            "validation failure response was not prebuilt."
                        ) from error
                    return self._close_failed_recording(
                        close_cause="validation_failure",
                        candidates=validation_failure_candidates,
                    )
                if recorder.validated_transition_count != previous_count + 1:
                    self._faulted = True
                    raise RuntimeError(
                        "recording append failed with incoherent validated progress"
                    ) from error
                if candidate_session.run_generation != self._session.run_generation:
                    self._running = False
                self._pov_global_slot = self._recipient_for(candidate_session)
                self._session = candidate_session
                self._adopt_system_operation(system_operation, candidate_session)
                self._view_mode = candidate_view_mode
                self._preset = candidate_preset
                self._revision = candidate_revision
                self._frame = candidate_frame
                self._command_records = candidate_command_records
                if recorder.lifecycle == "sealed":
                    if self._save_in_background:
                        self._start_background_save(candidate_frame.recording)
                        return candidate_result
                    if endpoint_candidates is None:
                        raise AssertionError(
                            "sealed processing failure lacks endpoint candidates."
                        ) from error
                    return self._finalize_endpoint_after_transition(
                        candidates=endpoint_candidates,
                    )
                if processing_failure_candidates is None:
                    raise AssertionError(
                        "processing failure response was not prebuilt."
                    ) from error
                return self._close_failed_recording(
                    close_cause="processing_failure",
                    candidates=processing_failure_candidates,
                )

        if dispatched.episode_restarted:
            self._acknowledge_saved_system_prefix()
        if (
            confirmed_discard
            and dispatched.episode_restarted
            and self._recorder is not None
        ):
            self._recorder.discard()

        if candidate_session.run_generation != self._session.run_generation:
            self._running = False
        self._pov_global_slot = self._recipient_for(candidate_session)
        self._session = candidate_session
        self._adopt_system_operation(system_operation, candidate_session)
        self._view_mode = candidate_view_mode
        self._preset = candidate_preset
        if dispatched.shutdown_requested:
            self._shutting_down = True
        if dispatched.changed:
            self._revision = candidate_revision
            self._frame = candidate_frame
        if self._recorder is not candidate_recorder:
            self._replay_handoff_cache = None
        self._recorder = candidate_recorder
        self._command_records = candidate_command_records
        if endpoint_candidates is not None:
            return self._finalize_endpoint_after_transition(
                candidates=endpoint_candidates,
            )
        if (
            self._save_in_background
            and dispatched.transition_applied is not None
            and self._recorder is not None
            and self._recorder.lifecycle == "sealed"
        ):
            # Game over is already installed; the save runs on its own thread.
            self._start_background_save(candidate_frame.recording)
        return candidate_result

    def _command_needs_saved_recording(self, command: object) -> bool:
        """Say whether a command must wait for a running background save.

        command is the request's command, possibly a replacement wrapper.
        Return True for the recording commands (Finish And Save, Open Replay,
        Retry Save, Save As, Exit), the replacement wrappers (Save And Replace,
        Confirm Discard And Replace), scenario switches and every command that
        would restart the match (input.recording_restart_intent_v1, for example
        R or a changed combat setup). Views, presets, focus, Run, Pause, System
        polls and the other keys never change the recorder, so they return
        False and stay responsive while the replay saves. Pure; no I/O.
        """
        if isinstance(
            command,
            (
                FinishAndReviewCommandV1,
                ReviewReplayCommandV1,
                RetrySaveCommandV1,
                SaveAsCommandV1,
                ExitCommandV1,
                ConfirmDiscardAndReplaceCommandV1,
                SaveAndReplaceCommandV1,
                ScenarioSwitchCommandV1,
            ),
        ):
            return True
        return (
            recording_restart_intent_v1(
                self._session,
                cast("DebuggerCommandV1", command),
                view_mode=self._view_mode,
                include_stress=self._include_stress,
            )
            is not None
        )

    def _current_recording_status(self) -> RecordingStatusV1 | None:
        """Return the recording status that frames show now.

        The caller holds the service lock. Return None without a recorder.
        While a background save runs, return its frozen "sealed" status and
        never read the recorder, which the save thread is changing.
        """
        if self._recorder is None:
            return None
        pending = self._background_save
        if pending is not None:
            return pending.status
        return self._recorder.status

    def _start_background_save(self, status: RecordingStatusV1 | None) -> None:
        """Save the sealed recorder on its own thread; game over is already shown.

        status is the "sealed" recording status of the frame just committed;
        every frame shows it until the save ends. The caller holds the service
        lock and has committed the game-over transition. The new thread
        ("marl-replay-save", not a daemon) runs only
        recorder.finalize_and_save, without the service lock, then installs the
        result through _adopt_background_save. If the thread cannot start, the
        save runs here instead, so it is never lost.

        Raises
        ------
        AssertionError
            No sealed recorder, a save already running, a missing status or an
            invalid close cause (all internal invariants).
        """
        recorder = self._recorder
        if (
            recorder is None
            or recorder.lifecycle != "sealed"
            or status is None
            or status.lifecycle != "sealed"
            or self._background_save is not None
        ):
            raise AssertionError("a background save needs one sealed recorder.")
        close_cause = recorder.close_cause
        if close_cause not in ("endpoint", "truncation"):
            raise AssertionError("sealed recorder has an invalid close cause.")
        job = _BackgroundSave(recorder=recorder, close_cause=close_cause, status=status)
        self._background_save = job
        job.thread = Thread(
            target=self._run_background_save,
            args=(job,),
            name="marl-replay-save",
        )
        try:
            job.thread.start()
        except RuntimeError:
            logging.getLogger(__name__).exception(
                "Replay save thread did not start; saving now instead"
            )
            self._run_background_save(job)

    def _run_background_save(self, job: _BackgroundSave) -> None:
        """Run one save, mark it done, then install it under the service lock.

        job is the save that _start_background_save created. An ordinary saved
        or failed result stays on the recorder; any other exception is kept in
        job.error for _adopt_background_save to log. job.done is set before the
        service lock is requested, so a command that holds the lock and waits
        for the save (_await_background_save) cannot deadlock with this thread.
        Writes the replay file (and its reload check) through the recorder.
        """
        try:
            job.recorder.finalize_and_save(job.close_cause)
        except Exception as error:  # Logged by _adopt_background_save.
            job.error = error
        finally:
            job.done.set()
            with self._lock:
                self._adopt_background_save(job)

    def _adopt_background_save(self, job: _BackgroundSave) -> None:
        """Install a finished background save once; later calls do nothing.

        The caller holds the service lock, and job.done is set. Build and check
        the frame with the recorder's real status ("saved" or
        "persistence_failed"), then commit it as one new revision, so the
        page's next read shows Saved, and queue the saved-replay evidence for
        the System worker. An unexpected save exception, a recorder in any other
        state, or a frame that cannot be built marks the service faulted and is
        logged; the game-over frame then stays installed.

        Raises
        ------
        AssertionError
            The recorder was replaced while its save ran (an internal invariant:
            every replacement waits for the save first).
        """
        if self._background_save is not job:
            return
        self._background_save = None
        recorder = job.recorder
        if self._recorder is not recorder:
            raise AssertionError("the recorder changed during its background save.")
        log = logging.getLogger(__name__)
        if job.error is not None or recorder.lifecycle not in (
            "saved",
            "persistence_failed",
        ):
            self._faulted = True
            log.error("The background replay save failed", exc_info=job.error)
            return
        revision = self._revision + 1
        try:
            frame = self._build_frame(revision=revision)
            self._build_authorized_presentation(
                session=self._session,
                raw_frame=frame,
                view_mode=self._view_mode,
            )
        except Exception:
            self._faulted = True
            log.exception("The saved replay's status could not be shown")
            return
        self._revision = revision
        self._frame = frame
        self._acknowledge_saved_system_prefix()

    def _await_background_save(self) -> None:
        """Wait for a running background save and install its result.

        The caller holds the service lock. The save thread sets job.done before
        it asks for this lock, so waiting here cannot deadlock; this call then
        installs the result itself and the thread's own install does nothing.
        Does nothing when no save runs. Blocks for the rest of the save (a
        300-tick game, 23.7 MB, took about 13 s on a busy CPU, measured 3 Oct
        2026).
        """
        job = self._background_save
        if job is None:
            return
        job.done.wait()
        self._adopt_background_save(job)

    def _system_operation_status(self) -> SystemOperationV1 | None:
        """Read bounded worker state without polling or changing the game."""
        worker = self._system_worker
        operation = None if worker is None else worker.operation
        if operation is None:
            return None
        finishing = (
            operation.acceptance_future
            if operation.adopted
            else operation.cleanup_future
        )
        cleanup_failed = (
            finishing is not None
            and finishing.done()
            and finishing.exception() is not None
        )
        return SystemOperationV1(
            operation_id=operation.identifier,
            state="failed"
            if cleanup_failed
            else "finishing"
            if operation.adopted
            else "cancelling"
            if operation.cancelled
            else "loading"
            if operation.kind == "setup"
            else "thinking",
        )

    def _build_frame(
        self,
        *,
        session: DebuggerSession | None = None,
        revision: int | None = None,
        view_mode: ViewMode | None = None,
        preset: Preset | None = None,
        recording_status: RecordingStatusV1 | None = None,
    ) -> LiveDebuggerFrame:
        """Build a candidate frame, using committed values wherever overrides are
        None.

        Agent POV frames are built for _recipient_for(session), the candidate's
        recipient; researcher frames for the focus. An Agent POV frame whose
        focus is not the recipient (you look at a policy's champion, or at an
        enemy your team sees now) also carries focused_public_agent_id, the
        focused champion's public ID (see _pov_focused_public_agent_id), so the
        page can show Watch Only. Otherwise that field is absent. With no
        recording_status override, the frame shows _current_recording_status
        (the frozen "sealed" status while a background save runs).
        """
        resolved_recording_status = recording_status
        if resolved_recording_status is None:
            resolved_recording_status = self._current_recording_status()
        resolved_session = self._session if session is None else session
        frame = build_debugger_frame(
            resolved_session,
            session_id=self._session_id,
            revision=self._revision if revision is None else revision,
            view_mode=self._view_mode if view_mode is None else view_mode,
            preset=self._preset if preset is None else preset,
            include_stress=self._include_stress,
            recording_status=resolved_recording_status,
            scenario_catalog=self._scenario_catalog,
            pov_global_slot=self._recipient_for(resolved_session),
        )
        status = self._system_operation_status()
        return frame.model_copy(
            update={
                "system_choices": self._system_choices,
                "system_operation": status,
                "run_state": self._run_state_for(frame),
                "fresh_launch_required": self._requires_fresh_launch(),
                "focused_public_agent_id": (
                    self._pov_focused_public_agent_id(resolved_session)
                    if frame.view_mode == "pov"
                    else None
                ),
            }
        )

    def _pov_focused_public_agent_id(self, session: DebuggerSession) -> str | None:
        """Return the Agent POV focus's public ID when it is not the recipient.

        session is the candidate or committed session. Agent POV shows the
        recipient's eyes (_recipient_for); session.controlled_global_slot is the
        focus, the champion the player looks at. When the two differ and the
        focus is one Agent POV may focus now (_agent_pov_focusable_slots: a
        Manual champion, one on the recipient's team, or one the recipient's
        team sees now), return the focus's public ID, for example "agent_2".
        Return None when the focus is the recipient or may not be focused now,
        so an unseen champion is never named. The page reads it only to show
        Watch Only and to refuse edits; policies never see it.
        Pure apart from the authorized-scene read; no simulator step or I/O.
        """
        focus = session.controlled_global_slot
        recipient = self._recipient_for(session)
        if focus == recipient:
            return None
        if focus not in _agent_pov_focusable_slots(session, recipient):
            return None
        return session.evaluation_context.roster[focus].public_agent_id

    def _remember_command(
        self,
        key: tuple[str, str],
        record: _CommandRecord,
    ) -> None:
        """Insert a request record into the current bounded recent-command cache."""
        self._remember_command_in(self._command_records, key, record)

    @staticmethod
    def _remember_command_in(
        records: OrderedDict[tuple[str, str], _CommandRecord],
        key: tuple[str, str],
        record: _CommandRecord,
    ) -> None:
        """Prepare a bounded cache mutation before scientific state commits."""
        records[key] = record
        records.move_to_end(key)
        while len(records) > _COMMAND_RECORD_LIMIT:
            records.popitem(last=False)
