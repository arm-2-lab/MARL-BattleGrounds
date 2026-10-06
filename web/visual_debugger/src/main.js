/**
 * @file Start the BattleClient (also the DevClient combat screen) or the Replay
 * Viewer page and connect its controls. The launcher supplies product identity
 * and a capability token. This module binds the required HTML elements, owns
 * the page state, installs page-lifetime listeners, and loads a matched
 * transport frame and authorized presentation. It paints only installed
 * authority, serializes commands, and manages local focus, filters, replay
 * playback, and downloads.
 *
 * Three page parts live in their own modules. Each gets this module's elements,
 * state and helpers once at startup and never assigns page state: shell.js (the
 * header, session toolbar and Help key lists), tactics-turn.js (the live command
 * deck) and filter-controls.js (the Visual Filters panel with its presets).
 *
 * It mounts the V5 chrome parts once at startup: the HUD (looks/hud.js; party
 * frames in the BattleClient and DevClient, the Arena Frames along the bottom
 * of the stage in the Replay Viewer), the camera switch (looks/camera.js), Frame
 * Explanation (F), the Palette choice, and in the Replay Viewer the five-tab
 * rail (ledger.js), the Rune Scrubber dock (rune-scrubber.js) and Tab agent stepping
 * (replay-recipient-rotation.js). It hands the choreographer a page surface that
 * adds the HUD and panel beat-hold roots, and draws the Replay Viewer's Champion
 * Frame after each tick settles. After each tick it redraws the match moment
 * cards (looks/heralds.js) in the herald layer: the Lineup card on tick 0, the
 * Verdict card on the final tick (the Replay Viewer's last tick, or game over
 * in the BattleClient and DevClient), else the Match Point or Ticks Left state
 * chip. Routine loads and frame updates make no top-bar notice
 * (clearNotice); warnings, errors and asked-for results do (setNotice).
 *
 * Run in Policies Only is paced by this page (toggleLiveRun): it sends Play
 * One Turn, waits for the next tick and its choreography, then sends the
 * next, so every tick glides to its end; the server's run_state stays
 * "paused". Those turns leave the message line alone, so "Running." stays
 * there until Pause. While Run is active a Space, Enter or N from the player
 * is ignored (pageRunOwnsLockIn), as on the vision page. N is one tick: the
 * battlefield forwards it and the page treats it as a Lock In key, like Space
 * (isLockInKeyName): as Manual it locks in, in Policies Only Python plays one
 * turn, and in scripted inspection it advances the next frame (the Lock In
 * button sends N there too). Order edits made while an earlier order is out
 * are queued in order (retainFencedCommand) and sent after it, still as
 * battlefield input.
 * A view switch (Agent POV <-> Oracle View) shows the tick already played at
 * its end instead of replaying it (settleLiveAudienceSwitch), so orders stay
 * open right after the switch.
 *
 * This module also owns: the display-switch state and the preset in use (every
 * client starts at Default Configuration); the camera on screen and its one
 * entry point, setCameraFrame (Whole Map until the camera look sets another
 * frame); the page's own local key jobs of the one command set (Help on ? and
 * Ranges on G in the Replay Viewer; V, B and F belong to the camera and Frame
 * Explanation owners); and turning a one-tick replay move into one replay_step
 * glide request for the choreographer.
 *
 * In the BattleClient and DevClient it also owns the live controls' behavior:
 * How To Play over the arena (#htp: ?, the How To Play button, Close and Full
 * Guide); the deck's message line (#msg) in the vision page's words for the
 * player's own commands (ownOrderFacts, laneRefusal, acceptedCommandSentence,
 * hostNoticeSentence; the words come from statusSentence in controls.js);
 * Tab, Shift+Tab, ‹ Prev and Next ›, which give the orders only to your
 * champions that can take orders (ownChampionTabStep);
 * the Keys chip and the arena's keys-on frame (a top-bar button press or a
 * view switch keeps keys on, and a click on the arena never pins a card);
 * R and G while orders are closed
 * (game over, after a save); the Recording pill and popover
 * (recordingWords); the top bar's save button (saveButtonState: Finish And
 * Save, Saving Replay…, then Open Replay once saved, or Retry Save), Finish
 * And Save in the popover and the Stopped overlay; the arena's on-screen V and
 * F keys (arena-buttons.js); Back To Your champion (Watch Only); the Restart dialog's
 * Save focus and Cancel sentence; Open Replay's new tab, which shows the dark
 * "Opening Replay" page (opening-replay-page.js) until the Replay Viewer's
 * page arrives; and the closed page after Exit. Agent POV: the renderer and the
 * choreographer receive the whole installed frame, and each must read only the
 * audience's own scene and events for anything drawn on or over the arena; the
 * camera frame passed to setCameraFrame must come from that same scene.
 *
 * Open the launcher URL; this module is not a standalone Node program.
 */

import {
  acquireCapabilityToken,
  acquireClientId,
  DebuggerApiError,
  extractJoinedFrame,
  extractNotice,
  extractReplayUrl,
  getCurrentFrameAndPresentation,
  getCurrentLiveSnapshot,
  getCurrentPresentation,
  getReplayEpisodeDetails,
  getReplayMetricCatalog,
  getReplayMetrics,
  getReplayTimeline,
  getReplayTimelineMarks,
  postCommand,
  postReplayCommand,
} from "./api.js";
import { wireArenaButtons } from "./arena-buttons.js";
import {
  authorizedOracleCommandSlotForPresentationKey,
  authorizedOracleCommandSlotForPublicAgentId,
  authorizedPresentationAgentDisplayId,
  authorizedPresentationAudience,
  authorizedPresentationFrameExplanationInput,
  authorizedPresentationHasResearcherSpace,
  authorizedPresentationIdentityRows,
  authorizedPresentationIncomingRows,
  authorizedPresentationInspectionState,
  authorizedPresentationPreferenceKey,
  authorizedPresentationResearcherInspectionState,
  authorizedPresentationResearcherSceneView,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
  sameAuthorizedPresentationPreferenceKey,
} from "./authorized-presentation-adapter.js";
import {
  isJoinedTransportAndAuthorizedPresentationV1,
  isPresentationJoinRace,
  joinReplayTransportAndTimelineV1,
  validateReplayTransportContinuityV1,
} from "./authorized-presentation-normalizer.js";
import { CombatChoreographer, ConsumedTransitionLedger } from "./choreography.js";
import { SvgChoreographyPainter } from "./choreography-painter.js";
import { beatView, isSubmissionCommand } from "./choreography-plan.js";
import { classLoreV1, classNamesV1 } from "./class-documentation.js";
import { registerLocalCommandHandler } from "./command-set.js";
import {
  collapsedControllerLine,
  controllerLineParts,
  rosterTeam,
} from "./controller-names.js";
import {
  audienceSwitchKeepsTick,
  bindBattlefieldControls,
  commandResponseSchedulesShutdown,
  keyboardCommand,
  nextOrderSlot,
  presentationRequiresSubmissionSettle,
  recordingCommandDecision,
  recordingSaveAsCommand,
  statusSentence,
} from "./controls.js";
import { explainAgent, explainTechnicalFact } from "./explanations.js";
import {
  createFilterControls,
  installVisualFilterControls,
  VISUAL_FILTER_CONTROL_HELP,
} from "./filter-controls.js";
import { setIconStyleDefaults } from "./icons.js";
import { createViewportTransform } from "./layout.js";
import { mountLedgerRail } from "./ledger.js";
import { createCameraSwitch, killCamKeyframes } from "./looks/camera.js";
import { championFrameView, renderChampionFrame } from "./looks/champion-frame.js";
import {
  createFrameExplanation,
  tooltip as frameExplanationTooltip,
} from "./looks/frame-explanation.js";
import * as heralds from "./looks/heralds.js";
import { createHud } from "./looks/hud.js";
import * as requiems from "./looks/requiems.js";
import {
  controllerRosterFromFrame,
  hudSourcesFromFrame,
  matchSummaryView,
  renderMatchSummary,
} from "./match-summary.js";
import {
  buildMetricSearchIndex,
  renderMetricDefinition,
  renderMetricNavigation,
  renderMetricRows,
  renderMetricSearchResults,
  searchMeasurements,
} from "./metrics-panel.js";
import { writeOpeningReplayPage } from "./opening-replay-page.js";
import {
  authorizedInspectorView,
  DebuggerPanels,
  describeRecordedAction,
  disclosurePanelInitiallyOpen,
  fullRecordLabel,
  renderSemanticInspector,
} from "./panels.js";
import {
  pendingPresentationSurfaceView,
  resolveInstalledPresentationAuthorityV1,
} from "./presentation-authority-view.js";
import { PresentationInstallCoordinator } from "./presentation-install.js";
import {
  bindReplayTimelineControls,
  REPLAY_TRANSPORT_STATES,
  ReplayPlaybackController,
  renderReplayTimelineControls,
  replayCommandRequest,
  replayTimelineSimulatorStep,
  validateReplayCommandOutcome,
} from "./replay-controls.js";
import { captureReplayBattlefieldPngV1 } from "./replay-export.js";
import {
  bindReplayAgentStepping,
  isReplayAgentRecipientRotation,
} from "./replay-recipient-rotation.js";
import { createRuneScrubber, scrubberBeatsFromSchedule } from "./rune-scrubber.js";
import { BattlefieldRenderer } from "./scene.js";
import {
  applyProductHeader,
  championCardFacts,
  createShell,
  renderCommandHelpLists,
} from "./shell.js";
import {
  requiresSharedObs as isReactiveController,
  isSystemController,
  isTeamController,
} from "./system-controls.js";
import {
  classGuideCardHtml,
  closedSentence,
  commandPreparesDeferredSubmit,
  commandRole,
  createTacticsTurn,
  isDeferredDraftKey,
  isFreshDeferredDraft,
  plainHostNotice,
  recordingWords,
  saveButtonState,
  saveFailedReason,
} from "./tactics-turn.js";
import {
  createChampionCardDescriptor,
  createSemanticDescriptor,
  createTooltipController,
  registerSymbolOwner,
  registerTooltipOwner,
} from "./tooltip.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualFilterEnabled,
  reduceVisualFilterState,
} from "./visual-filters.js";

/**
 * Find a required page element by its HTML ID. Return the element, or throw
 * Error when the page template is missing it; startup must not continue with
 * a partly connected interface.
 *
 * @param {string} id
 * @returns {any}
 */
function requiredElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Browser shell is missing required element #${id}.`);
  }
  return element;
}

const elements = {
  appTitle: requiredElement("app-title"),
  connectionStatus: requiredElement("connection-status"),
  audienceBadge: requiredElement("audience-badge"),
  terminalBadge: requiredElement("terminal-badge"),
  matchScoreboard: requiredElement("match-scoreboard"),
  matchTask: requiredElement("match-task"),
  matchTeamA: requiredElement("match-team-a"),
  matchTeamB: requiredElement("match-team-b"),
  recordingBadge: requiredElement("recording-badge"),
  viewSelect: requiredElement("view-select"),
  stepValue: requiredElement("step-value"),
  transitionValue: requiredElement("transition-value"),
  recordingPanel: requiredElement("recording-panel"),
  recordingLifecycle: requiredElement("recording-lifecycle"),
  recordingProgress: requiredElement("recording-progress"),
  recordingCompletion: requiredElement("recording-completion"),
  recordingPersistenceFact: requiredElement("recording-persistence-fact"),
  recordingPersistenceError: requiredElement("recording-persistence-error"),
  recordingStatusNote: requiredElement("recording-status-note"),
  recordingMetricsProgress: requiredElement("recording-metrics-progress"),
  recordingMetricsHelp: requiredElement("recording-metrics-help"),
  recordingFinishButton: requiredElement("recording-finish-button"),
  recordingReviewButton: requiredElement("recording-review-button"),
  recordingReplayLink: requiredElement("recording-replay-link"),
  recordingRetryButton: requiredElement("recording-retry-button"),
  recordingSaveAsControl: requiredElement("recording-save-as-control"),
  recordingSaveAsInput: requiredElement("recording-save-as-input"),
  recordingSaveAsButton: requiredElement("recording-save-as-button"),
  recordingDiscardDialog: requiredElement("recording-discard-dialog"),
  recordingDiscardIntent: requiredElement("recording-discard-intent"),
  recordingDiscardCancelButton: requiredElement("recording-discard-cancel-button"),
  recordingDiscardConfirmButton: requiredElement("recording-discard-confirm-button"),
  recordingSaveReplaceButton: requiredElement("recording-save-replace-button"),
  replayTimeline: requiredElement("replay-timeline"),
  replayArtifactReference: requiredElement("replay-artifact-reference"),
  replayCompletionBadge: requiredElement("replay-completion-badge"),
  replayProcessingBadge: requiredElement("replay-processing-badge"),
  replayEndReason: requiredElement("replay-end-reason"),
  replayFirstButton: requiredElement("replay-first-button"),
  replayBackTenButton: requiredElement("replay-back-ten-button"),
  replayPreviousButton: requiredElement("replay-previous-button"),
  replayPlayPauseButton: requiredElement("replay-play-pause-button"),
  replayNextButton: requiredElement("replay-next-button"),
  replayForwardTenButton: requiredElement("replay-forward-ten-button"),
  replayLastButton: requiredElement("replay-last-button"),
  replayFrameSlider: requiredElement("replay-frame-slider"),
  replayFramePosition: requiredElement("replay-frame-position"),
  replayPlaybackRate: requiredElement("replay-playback-rate"),
  replayTransportStatus: requiredElement("replay-transport-status"),
  replayArtifactActions: requiredElement("replay-artifact-actions"),
  replayExportPngButton: requiredElement("replay-export-png-button"),
  replayDownloadMetricsButton: requiredElement("replay-download-metrics-button"),
  replayEpisodeDetailsButton: requiredElement("replay-episode-details-button"),
  replayMetricsPreparation: requiredElement("replay-metrics-preparation"),
  replayMetricsPreparationText: requiredElement("replay-metrics-preparation-text"),
  metricPanel: requiredElement("evaluation-metrics"),
  metricScope: requiredElement("metric-scope"),
  metricSelection: requiredElement("metric-selection"),
  metricView: requiredElement("metric-view"),
  metricViewField: requiredElement("metric-view-field"),
  metricSearchArea: requiredElement("metric-search-area"),
  metricSearchContent: requiredElement("metric-search-content"),
  metricSearch: requiredElement("metric-search"),
  metricSearchStatus: requiredElement("metric-search-status"),
  metricSearchResults: requiredElement("metric-search-results"),
  metricSearchMore: requiredElement("metric-search-more"),
  metricSearchDefinition: requiredElement("metric-search-definition"),
  metricStatus: requiredElement("metric-status"),
  metricProgress: requiredElement("metric-progress"),
  metricRows: requiredElement("metric-rows"),
  metricDescription: requiredElement("metric-description"),
  replayRangesButton: requiredElement("replay-ranges-button"),
  replayClearReferenceButton: requiredElement("replay-clear-reference-button"),
  reconnectButton: requiredElement("reconnect-button"),
  helpButton: requiredElement("help-button"),
  // How To Play over the arena (BattleClient and DevClient); the Replay
  // Viewer keeps the Help dialog (#help-dialog).
  howToPlay: requiredElement("htp"),
  howToPlayCloseButton: requiredElement("help-close-button"),
  howToPlayFullButton: requiredElement("htp-full"),
  railFullGuideButton: requiredElement("full-guide"),
  recordingPillText: requiredElement("rec-pill-text"),
  recordingPopover: requiredElement("rec-pop"),
  recordingPopState: requiredElement("pop-state"),
  recordingPopReason: requiredElement("pop-reason"),
  recordingPopFinishButton: requiredElement("pop-finish"),
  stoppedSaveButton: requiredElement("stopped-save"),
  watchOnlyBackButton: requiredElement("wo-back"),
  statusMessage: requiredElement("msg"),
  keysChip: requiredElement("keys-chip"),
  keysText: requiredElement("keys-text"),
  arena: requiredElement("arena"),
  exitButton: requiredElement("exit-button"),
  resetButton: requiredElement("reset-button"),
  liveRangesButton: requiredElement("live-ranges-button"),
  notice: requiredElement("notice"),
  workspace: requiredElement("workspace"),
  scenarioDescription: requiredElement("scenario-description"),
  battlefieldShell: requiredElement("battlefield-shell"),
  battlefield: requiredElement("battlefield"),
  battlefieldEmpty: requiredElement("battlefield-empty"),
  commandDeck: document.querySelector(".command-deck"),
  commandControlledActor: requiredElement("command-controlled-actor"),
  liveRunButton: requiredElement("live-run-button"),
  liveRunStatus: requiredElement("live-run-status"),
  liveControlHelp: requiredElement("live-control-help"),
  liveMatchRules: requiredElement("live-match-rules"),
  roster: requiredElement("roster"),
  rosterCount: requiredElement("roster-count"),
  agentDetails: requiredElement("agent-details"),
  visualFilters: requiredElement("visual-filters"),
  visualFilterOptions: requiredElement("visual-filter-options"),
  visualFilterCount: requiredElement("visual-filter-count"),
  visualFilterPresets: requiredElement("visual-filter-presets"),
  visualFilterPresetStatus: requiredElement("visual-filter-preset-status"),
  resetVisualFiltersButton: requiredElement("reset-visual-filters-button"),
  enableAllVisualFiltersButton: requiredElement("enable-all-visual-filters-button"),
  disableAllVisualFiltersButton: requiredElement("disable-all-visual-filters-button"),
  defaultVisualFiltersButton: requiredElement("default-visual-filters-button"),
  partyFrames: requiredElement("party-frames"),
  teamRosters: requiredElement("team-rosters"),
  fallenFeed: requiredElement("fallen-feed"),
  championFrame: requiredElement("champion-frame"),
  chronicle: requiredElement("chronicle"),
  replayCameraSwitch: requiredElement("replay-camera-switch"),
  cameraTips: requiredElement("camera-tips"),
  liveCameraSwitch: requiredElement("live-camera-switch"),
  visualFiltersButton: requiredElement("visual-filters-button"),
  arenaButtons: requiredElement("arena-buttons"),
  arenaEveryoneInViewButton: /** @type {HTMLButtonElement} */ (
    requiredElement("arena-everyone-in-view-button")
  ),
  arenaBattleCamButton: /** @type {HTMLButtonElement} */ (
    requiredElement("arena-battle-cam-button")
  ),
  arenaClearSelectionButton: /** @type {HTMLButtonElement} */ (
    requiredElement("arena-clear-selection-button")
  ),
  arenaFrameExplanationButton: /** @type {HTMLButtonElement} */ (
    requiredElement("arena-frame-explanation-button")
  ),
  arenaMatchSummaryButton: /** @type {HTMLButtonElement} */ (
    requiredElement("arena-match-summary-button")
  ),
  replayRailFrame: requiredElement("replay-rail-frame"),
  replayRailActions: requiredElement("replay-rail-actions"),
  thisTurn: requiredElement("this-turn"),
  visualFiltersBody: requiredElement("visual-filters-body"),
  selectionCard: requiredElement("selection-card"),
  selectionHeading: requiredElement("selection-heading"),
  pendingHeading: requiredElement("pending-heading"),
  pendingCount: requiredElement("pending-count"),
  pendingScope: requiredElement("pending-scope"),
  pendingCard: requiredElement("pending-card"),
  stayButton: requiredElement("stay-button"),
  commandTargetSelect: requiredElement("command-target-select"),
  policyTargetMasks: requiredElement("policy-target-masks"),
  policyTargetMaskRows: requiredElement("policy-target-mask-rows"),
  noCombatButton: requiredElement("no-combat-button"),
  basicButton: requiredElement("basic-button"),
  ultimateButton: requiredElement("ultimate-button"),
  submitTurnButton: requiredElement("submit-turn-button"),
  commandCommitTitle: requiredElement("command-commit-title"),
  commandCommitSummary: requiredElement("command-commit-summary"),
  acceptedCard: requiredElement("accepted-card"),
  acceptedAnnouncement: requiredElement("accepted-announcement"),
  diagnosticsCard: requiredElement("diagnostics-card"),
  visualTooltip: requiredElement("visual-tooltip"),
  visualTooltipTitle: requiredElement("visual-tooltip-title"),
  visualTooltipDetails: requiredElement("visual-tooltip-details"),
  helpDialog: requiredElement("help-dialog"),
  helpHeading: requiredElement("help-heading"),
  battlefieldInstructions: requiredElement("battlefield-instructions"),
  liveOnly: /** @type {NodeListOf<HTMLElement>} */ (
    document.querySelectorAll("[data-live-only]")
  ),
  liveHelpModes: /** @type {NodeListOf<HTMLElement>} */ (
    document.querySelectorAll("[data-live-help-mode]")
  ),
  replayOnly: /** @type {NodeListOf<HTMLElement>} */ (
    document.querySelectorAll("[data-replay-only]")
  ),
};

/**
 * Page-lifetime presentation state. It is replaced atomically, never stored,
 * and deliberately survives transport, authority, audience, and episode
 * changes until the document itself reloads. Every client starts with the
 * same shipped switch state (DEFAULT_VISUAL_FILTER_STATE) and Default
 * Configuration as the preset in use. visualFilterPresetId is the preset the
 * viewer last picked ("default", "researcher", "manuscript" or
 * "minimal"); hand changes are counted against it.
 */
let visualFilterState = DEFAULT_VISUAL_FILTER_STATE;
/** @type {string} */
let visualFilterPresetId = "default";
let agentLocalRangesVisible = false;

/**
 * The public ID of the last Manual champion the player gave orders to, or null
 * before any. Back To Your <Class> (Watch Only) returns there. Only the
 * player's own champions are stored, so it never names a hidden champion.
 *
 * @type {string | null}
 */
let lastManualPublicId = null;
let agentLocalRangesInitialized = false;
let confirmedResearcherRangesVisible = false;

/**
 * One camera frame other than Whole Map: camera "everyone_in_view" or
 * "battle_cam", and the world rectangle it fits on screen, in world units with
 * y counting up (it may reach past the map edge for a margin).
 *
 * @typedef {Readonly<{
 *   camera: "everyone_in_view" | "battle_cam",
 *   minX: number,
 *   minY: number,
 *   maxX: number,
 *   maxY: number,
 * }>} PageCameraFrame
 */

/**
 * The camera on screen: null for Whole Map (the default in every client), or
 * one PageCameraFrame. setCameraFrame is the only writer. Page-lifetime memory
 * only.
 *
 * @type {PageCameraFrame | null}
 */
let cameraFrame = null;

/**
 * The page's own SVG group of match moment cards (the Lineup card, the Verdict
 * card or the state chip) in the renderer's herald layer, or null when none is
 * drawn. Only renderMatchMomentCards replaces it.
 *
 * @type {SVGGElement | null}
 */
let matchMomentCardsGroup = null;

/**
 * The Verdict card's working Match Summary and Show Arena buttons, laid over
 * the arena in the battlefield shell (heralds.buildVerdictActions), or null.
 * Only renderMatchMomentCards replaces it.
 *
 * @type {HTMLElement | null}
 */
let verdictActions = null;

/**
 * True once the player pressed the Verdict card's Show Arena: the Verdict
 * stays tucked into its band plate for the rest of this page session.
 */
let verdictTucked = false;

/**
 * The open Match Summary sheet (heralds.buildMatchSummarySheet) below the
 * arena, or null: in the BattleClient and DevClient it takes the deck's place
 * (the deck is inert at game over); in the Replay Viewer it covers the
 * timeline dock under the arena. Opened by the Verdict card's Match
 * Summary button or, while the Verdict is folded into its band, by the
 * arena's Match Summary key (which also closes it); closed by its Close
 * button, a view switch or when the Verdict leaves.
 *
 * @type {HTMLElement | null}
 */
let matchSummarySheet = null;

/**
 * What the Fallen panel remembers between paints (requiems.fallenFeedHistory):
 * the falls of the last 8 ticks of the audience shown last, or null. Used
 * only when a frame carries no recent_falls (an older producer): both
 * clients' servers now send that history with every frame.
 * installJoinedAuthority sets it to null whenever the installed view kind or
 * Agent POV recipient changes, even if no plan of the other view was painted
 * in between, so a quick Oracle -> POV -> Oracle round trip equals a fresh
 * page.
 *
 * @type {import("./looks/requiems.js").FallenFeedMemory | null}
 */
let fallenFeedMemory = null;

/**
 * @type {{
 *   token: string | null,
 *   clientId: string,
 *   authority: Readonly<Record<string, any>> | null,
 *   frame: Record<string, any> | null,
 *   presentation: Readonly<Record<string, any>> | null,
 *   timeline: Record<string, any> | null,
 *   busy: boolean,
 *   quietBusy: boolean,
 *   offline: boolean,
 *   resyncRequired: boolean,
 *   shuttingDown: boolean,
 *   exitRecording: Record<string, any> | null,
 *   notice: string | null,
 *   noticeLevel: string,
 *   pageRun: boolean,
 *   pageRunLoop: boolean,
 * }}
 */
const state = {
  token: acquireCapabilityToken(),
  clientId: acquireClientId(),
  authority: null,
  frame: null,
  presentation: null,
  timeline: null,
  busy: false,
  // True while the request in flight is autoplay's own next-frame fetch
  // (sendReplayCommand); the connection pill then stays "Connected".
  quietBusy: false,
  offline: false,
  resyncRequired: false,
  shuttingDown: false,
  // The recording status that came with the accepted Exit answer, or null
  // (no Exit yet, or the match records nothing). It is newer than the
  // installed frame's when that answer's frame could not be installed (the
  // read raced the server's shutdown), so the Recording pill, popover and
  // save button paint it once shuttingDown is true. Set only by dispatchCommand's
  // Exit paths; never cleared, because a page that accepted Exit stops
  // reading.
  exitRecording: null,
  notice: null,
  noticeLevel: "info",
  // True while the page's own Run plays turn after turn (toggleLiveRun).
  pageRun: false,
  // True while runLiveTurns is going: from Run until its last turn has landed
  // and played to its end, so also for that last tick after Pause has cleared
  // pageRun. toggleLiveRun reads it so a quick Pause then Run never starts a
  // second loop. Its Lock In keys are ignored then (pageRunOwnsLockIn), so the
  // deck keeps Play One Turn and Lock In disabled while it is true
  // (tactics-turn.js renderCommandAvailability). Only runLiveTurns writes it.
  pageRunLoop: false,
};

/**
 * The most order keys (steps, 0, 1, 2) one live request keeps while it is
 * out; a further key is dropped. Four covers a step, an action, a target
 * and a change of mind pressed during one round trip.
 */
const MAX_DEFERRED_DRAFTS = 4;

/**
 * One live request may retain up to MAX_DEFERRED_DRAFTS fresh draft intents
 * and focus changes (retainFocusChange keeps at most one), oldest first, and
 * one following Lock In key (Enter, Space or N) while a coherent response is
 * installed; they are sent in that order afterwards. This is page-local
 * latency state, not simulator, action, or replay state.
 *
 * keepsTarget is true when the request cannot change the staged target (a
 * step, 0, 1 or 2 key), so a 1 or 2 pressed meanwhile can be refused at once;
 * lateRefusal keeps that refusal so it is shown again after the request's
 * own sentence (the keys were pressed in that order).
 *
 * @type {{
 *   allowsDeferredSubmit: boolean,
 *   deferredDrafts: Array<Readonly<Record<string, unknown>>>,
 *   deferredSubmit: Readonly<Record<string, unknown>> | null,
 *   releaseFocusAfterSettlement: boolean,
 *   recordingPreparation: "metrics" | "possible" | null,
 *   keepsTarget: boolean,
 *   lateRefusal: string | null,
 * } | null}
 */
let activeLiveCommandTransaction = null;

/**
 * One page-local read-only artifact transaction. It is an async ownership
 * fence, not replay transport state, and is invalidated synchronously whenever
 * installed presentation authority is cleared or the camera frame changes.
 * cameraFrame is the camera on screen when it began (null for Whole Map).
 *
 * @type {Readonly<{
 *   kind: "export_png" | "download_metrics" | "download_episode_details",
 *   authority: Readonly<Record<string, any>>,
 *   transport: Readonly<Record<string, any>>,
 *   presentation: Readonly<Record<string, any>>,
 *   playbackGeneration: number,
 *   localInspectedPresentationKey: string | null,
 *   showRanges: boolean,
 *   visualFilters: typeof visualFilterState,
 *   cameraFrame: PageCameraFrame | null,
 * }> | null}
 */
let replayArtifactActionTransaction = null;

const SCIENTIFIC_DISCLOSURE_BODY_IDS = Object.freeze({
  "command-deck": "command-deck-body",
  "roster-details": "roster-details-body",
  "agent-details": "agent-details-body",
  "pending-turn-details": "pending-turn-details-body",
  "latest-transition-details": "latest-transition-details-body",
  "technical-frame-details": "technical-frame-details-body",
});

const scientificDisclosures = Object.freeze(
  Object.entries(SCIENTIFIC_DISCLOSURE_BODY_IDS).map(([panelId, bodyId]) => {
    const panel = requiredElement(panelId);
    const body = requiredElement(bodyId);
    if (!(panel instanceof HTMLDetailsElement) || !(body instanceof HTMLElement)) {
      throw new TypeError(`Scientific disclosure ${panelId} has invalid markup.`);
    }
    return Object.freeze({ panelId, panel, body });
  }),
);

/**
 * The only retained scientific preference record. This is deliberately one
 * active record rather than a cache keyed by authority: unrelated A -> B
 * crossings replace A, so a later B -> A boundary receives fresh defaults.
 * One certified Agent recipient rotation retains only the current
 * user-interface preferences; local inspection is still rebuilt from the new
 * authority.
 *
 * @type {{
 *   authorityKey: Readonly<{tuple: readonly unknown[], serialized: string}>,
 *   disclosures: Record<string, {open: boolean, scrollTop: number}>,
 *   agentDetailsAutoOpenAllowed: boolean,
 *   primaryFocus: null | {
 *     surface: "roster" | "battlefield",
 *     presentationKey: string,
 *     publicAgentId: string,
 *   },
 *   localInspection: {owned: false} | {owned: true, presentationKey: string | null},
 * } | null}
 */
let activePresentationPreference = null;

/**
 * One Replay Agent activation is staged until its certified successor is
 * installed. This keeps the previous Agent Details content stable while the
 * retained battlefield is busy and carries no geometry or opaque authority
 * data into the successor.
 *
 * @type {null | Readonly<{
 *   sourcePreferenceGeneration: number,
 *   recipientPublicAgentId: string,
 *   autoOpenAgentDetails: boolean,
 * }>}
 */
let pendingReplayRecipientActivation = null;

let presentationPreferenceGeneration = 0;
let presentationPreferenceNeedsContentRender = false;
let consumedReplayRestartGeneration = -1;
// The last replay intent generation whose one-tick step glide was handed to the
// choreographer; each step glides once (installedChoreographyControl).
let consumedReplayStepGeneration = -1;
let suppressPlaybackStateRender = false;
/** @type {number | null} */
let pendingPresentationPreferenceRestoreFrame = null;
/** @type {Element | null} */
let pendingPrimaryFocusRestoreAnchor = null;
/** @type {string | null} */
let workspaceMinimumHeightBeforeAuthorityInstall = null;

/** @type {WeakMap<HTMLDetailsElement, Readonly<{open: boolean}>>} */
const expectedDisclosureToggles = new WeakMap();

const REPLAY_SAVE_MESSAGE = "Saving replay. Longer recordings can take longer.";
const METRIC_PREPARATION_MESSAGE = "Preparing Metrics · Longer Games Take Longer.";

const SCRIPTED_INSPECTION_RECORDING_COMMANDS = new Set([
  "finish_and_review",
  "review_replay",
  "retry_save",
  "save_as",
  "confirm_discard_and_replace",
  "exit",
]);

/**
 * Mark a frame that belongs to a different product than this page.
 * This Error subclass lets request handlers clear displayed authority and require
 * a reconnect instead of showing a live frame in the Replay Viewer or vice versa.
 */
class ProductIdentityMismatchError extends Error {}

/** @type {Readonly<{schema_version: 1, product_kind: "combat_debugger" | "replay_viewer"}> | null} */
let productIdentity = null;
/** @type {string | null} */
let startupProductIdentityError = null;

/**
 * Validate and install the launcher product identity before accepting frames.
 * Accept schema version 1, a known product and an
 * optional boolean initial range choice.
 * Throw TypeError for invalid input. Update page labels and product state, and
 * apply the initial range choice only on its first installation. The display
 * switches need no product default: every client starts at
 * DEFAULT_VISUAL_FILTER_STATE (Target Selection Visuals is on everywhere).
 * Side effects: applyProductHeader (shell.js) writes the tab title, #app-title
 * and #help-heading, and `<html data-product-kind>` gets the product. The
 * private DevClient retitles the page after this runs.
 *
 * @param {unknown} rawIdentity
 * @returns {Readonly<{schema_version: 1, product_kind: "combat_debugger" | "replay_viewer"}>}
 */
function applyBootstrap(rawIdentity) {
  if (!isRecord(rawIdentity)) {
    throw new TypeError("Product bootstrap must be an object.");
  }
  const hasInitialRanges = Object.hasOwn(rawIdentity, "initial_show_ranges");
  if (hasInitialRanges && typeof rawIdentity.initial_show_ranges !== "boolean") {
    throw new TypeError("Initial range visibility must be a Boolean.");
  }
  const keys = Object.keys(rawIdentity)
    .filter((key) => key !== "initial_show_ranges")
    .sort();
  if (keys.length !== 2 || keys[0] !== "product_kind" || keys[1] !== "schema_version") {
    throw new TypeError("Product bootstrap has an invalid shape.");
  }
  if (rawIdentity.schema_version !== 1) {
    throw new TypeError("Product bootstrap schema version is unsupported.");
  }
  if (
    rawIdentity.product_kind !== "combat_debugger" &&
    rawIdentity.product_kind !== "replay_viewer"
  ) {
    throw new TypeError("Product bootstrap kind is unsupported.");
  }
  if (!agentLocalRangesInitialized && hasInitialRanges) {
    agentLocalRangesVisible = rawIdentity.initial_show_ranges;
    agentLocalRangesInitialized = true;
  }
  productIdentity = Object.freeze({
    schema_version: 1,
    product_kind: rawIdentity.product_kind,
  });
  applyProductHeader(elements, productIdentity.product_kind);
  document.documentElement.dataset.productKind = productIdentity.product_kind;
  return productIdentity;
}

/**
 * Check that a frame agrees with the product installed from the launcher.
 * Throw ProductIdentityMismatchError when identity is absent or live/replay modes
 * differ. A frame cannot choose which product this page is allowed to become.
 *
 * @param {Record<string, any>} frame
 */
function assertFrameMatchesProductIdentity(frame) {
  if (productIdentity === null) {
    throw new ProductIdentityMismatchError(
      "No validated product identity is installed.",
    );
  }
  const frameIsReplay = frame.viewer_mode === "replay";
  const identityIsReplay = productIdentity.product_kind === "replay_viewer";
  if (frameIsReplay !== identityIsReplay) {
    throw new ProductIdentityMismatchError(
      "The authoritative frame does not match this route's product identity.",
    );
  }
}

try {
  applyBootstrap(Reflect.get(globalThis, "__MARL_DEBUGGER_BOOTSTRAP__"));
} catch (error) {
  startupProductIdentityError =
    error instanceof Error ? error.message : "Product bootstrap is invalid.";
}

// Page parts with their own modules. They are created here, after the page
// state exists and before the choreographer or the replay controller can ask
// for a repaint. Each part captures the elements, the state object and the
// helper functions below; function declarations are hoisted, so passing them
// now is safe. The deck needs renderConnection from the shell and the shell
// needs renderCommandAvailability from the deck, so the deck gets a small
// wrapper that looks renderConnection up when it runs.
const {
  policyControllerBlocksActionEdit,
  renderCommandAvailability,
  renderSystemOperation,
  bindCommandDeck,
  showStatusMessage,
  forgetRefusalEchoes,
} = createTacticsTurn({
  elements,
  state,
  isReplayMode,
  isTerminal,
  isRecord,
  asArray,
  agentIdentity,
  installedAuthorityIsCoherent,
  liveScriptedInspectionOnly,
  recordingScientificControlsFenced,
  modeAvailability,
  authorizedResearcherAgentForPresentationKey,
  clearPresentationTooltipOwner,
  dispatchCommand,
  dispatchCommandFromDraftControl,
  scheduleLivePoll,
  setNotice,
  toggleLiveRun,
  renderConnection: () => renderConnection(),
  classGuideEnabled: () => isVisualFilterEnabled(visualFilterState, "class_guide"),
  visualFilters: () => visualFilterState,
});

const {
  renderConnection,
  applyBattlefieldBoundaryCopy,
  renderSessionToolbar,
  refreshHoverCard,
} = createShell({
  elements,
  state,
  getProductIdentity: () => productIdentity,
  getActiveLiveCommandTransaction: () => activeLiveCommandTransaction,
  isReplayMode,
  isTerminal,
  isRecord,
  asArray,
  installedPresentationAuthority,
  installedPresentationRangesVisible,
  liveScriptedInspectionOnly,
  liveBattlefieldCommandsInteractive,
  scriptedBattlefieldLocallyActionable,
  liveAgentCloseoutInspectionActionable,
  installedLiveScientificInspectionAvailable,
  recordingScientificControlsFenced,
  recordingRestartControlsBlocked,
  applyAuthorizedInspectorChrome,
  renderInstalledMatch,
  renderPendingPresentationChrome,
  renderRecordingControls,
  renderReplayMetadata,
  renderCommandAvailability,
  registerAuthorityAwareUtilityHelp,
  getVisualFilterState: () => visualFilterState,
  activateReplayPovAgent: (command) => dispatchPanelCommand(command),
});

// The Visual Filters panel lists only the switches of this client, so it is
// built once the product identity is known (none when startup was refused).
// The Help dialog's key lists come from the one command set. Panel Tokens
// changes only the docked Roster, Agent and Actions panels: the DevClient
// shows them in its Developer drawer (unhidden before this module loads), the
// BattleClient has none, so its drawer leaves that switch out.
const liveDockedPanelsShown =
  document.getElementById("developer-drawer")?.hidden === false;
const visualFilterSwitches = installVisualFilterControls(
  elements,
  installedProductKind(),
  {
    omit:
      installedProductKind() === "combat_debugger" && !liveDockedPanelsShown
        ? ["panel_tokens"]
        : [],
  },
);
renderCommandHelpLists(document);

const { renderVisualFilterControls, bindFilterControls } = createFilterControls({
  elements,
  state,
  switches: visualFilterSwitches,
  installedPresentationRangesVisible,
  applyVisualFilterAction,
  setActiveRangesVisible,
});

const CONTROL_HELP = Object.freeze([
  [
    "#metric-scope",
    "Evaluation Metric Scope",
    "Up to Current Tick counts only the ticks up to this one. Entire Episode counts the whole saved game.",
  ],
  [
    "#metric-selection",
    "Evaluation Topic",
    "Choose a topic, then its Totals or By Recipient view when there is one. Metrics are for researchers; the agents never see them.",
  ],
  [
    "#replay-episode-details-button",
    "Episode Details",
    "Download the recorded episode identity, policies, configuration, completion and runtime details as JSON.",
  ],
  [
    "#replay-timeline",
    "Replay Timeline",
    "Move through the saved game. The replay never changes.",
    "composite",
  ],
  [
    "#view-select",
    "Audience View",
    "Switch between Oracle View (everything) and Agent POV (only what one team saw).",
  ],
  ["#reconnect-button", "Reconnect", "Load the newest game state again."],
  ["#help-button", "How To Play", "Open the guide to the keys and controls."],
  [
    "#visual-filters-button",
    "Visual Filters",
    "Choose which marks and effects the battlefield shows. It changes only your screen, never the game.",
  ],
  [
    "#arena-everyone-in-view-button",
    "Everyone In View",
    "Move the camera so every champion on the field is in view. Press it again for the whole map. You can also press V.",
  ],
  [
    "#arena-battle-cam-button",
    "Battle Cam",
    "Follow the fights: the camera frames each fight, leans in on each fall and slows it down while the replay plays. Press it again for the whole map. You can also press B.",
  ],
  [
    "#arena-clear-selection-button",
    "Clear Selection",
    "Clear the selected champion, so no champion is picked. You can also press Esc.",
  ],
  [
    "#arena-frame-explanation-button",
    "Frame Explanation",
    "Number what happened on this tick, on the battlefield and in a list. You can also press F.",
  ],
  [
    "#arena-match-summary-button",
    "Match Summary",
    "Open or close the game's Match Summary: each champion's kills, deaths, damage, healing, Red Zone deaths and Ultimates, for both teams.",
  ],
  [
    "#help-close-button",
    "Close How To Play",
    "Close the guide to the keys and controls.",
  ],
  [
    "#full-guide",
    "Full Guide",
    "Open How To Play over the arena: every key and control, in plain words. You can also press ?.",
  ],
  [
    '#guide-level button[data-level="guide"]',
    "Guide",
    "The most help: Do This Now, a dashed outline around the control to use next, and the How To Play card. Your choice is remembered.",
  ],
  [
    '#guide-level button[data-level="hints"]',
    "Hints",
    "Some help: Do This Now only, with no outline and no How To Play card. Your choice is remembered.",
  ],
  [
    '#guide-level button[data-level="off"]',
    "Off",
    "No help on this card: Do This Now, the outline and the How To Play card are hidden. Your choice is remembered.",
  ],
  [
    "#recording-badge",
    "Replay Recording",
    "Open the replay menu: how much of this game the replay has kept, where it is saved, Finish And Save, Open Replay, and Retry Save or Save As if a save fails.",
  ],
  [
    "#pop-finish",
    "Finish And Save",
    "Save this game and keep it on screen to look over. Restart begins another match.",
  ],
  ["#exit-button", "Exit", "Close the client and stop its server safely."],
  [
    "#recording-finish-button",
    "Finish And Save",
    "Save this game and keep it on screen to look over. Restart begins another match.",
  ],
  [
    "#recording-review-button",
    "Open Replay",
    "Open the saved game in a separate Replay Viewer tab.",
  ],
  ["#recording-retry-button", "Retry Save", "Try saving the same replay again."],
  [
    "#recording-save-as-input",
    "Save As File Name",
    "Type a file name only, no folder. An existing file is never replaced.",
  ],
  [
    "#recording-save-as-button",
    "Save As",
    "Save the same replay under the name you typed.",
  ],
  ["#recording-discard-cancel-button", "Cancel", "Keep this game and do not restart."],
  [
    "#recording-discard-confirm-button",
    "Discard",
    "Throw away the unsaved game and start the new one.",
  ],
  [
    "#recording-save-replace-button",
    "Save",
    "Save this game, then start the selected next match.",
  ],
  [
    "#recording-replay-link",
    "Open Saved Replay In New Tab",
    "Open the separate local Replay Viewer.",
  ],
  ["#replay-first-button", "Start Replay Tick", "Go to tick 0."],
  ["#replay-back-ten-button", "Back Ten Ticks", "Go back 10 ticks."],
  ["#replay-previous-button", "Previous Replay Frame", "Go back one tick."],
  ["#replay-play-pause-button", "Replay Playback", "Play or pause the replay."],
  ["#replay-next-button", "Next Replay Frame", "Go forward one tick."],
  ["#replay-forward-ten-button", "Forward Ten Ticks", "Go forward 10 ticks."],
  ["#replay-last-button", "End Replay Tick", "Go to the last tick."],
  [
    "#replay-frame-slider",
    "Replay Tick",
    "Drag to pick a tick; the replay jumps there when you let go.",
  ],
  [
    "#replay-playback-rate",
    "Replay Playback Speed",
    "Change how fast the replay plays. The saved game does not change.",
  ],
  ["#command-target-select", "Selected Target", "Pick a target for your champion."],
  [
    "#submit-turn-button",
    "Lock In Turn",
    "Send your orders; policies then choose and everyone acts at once. In a scripted scene it plays the next scripted tick. Looking at a policy's champion does not give you control of it.",
  ],
  [
    "#live-run-button",
    "Run Or Pause",
    "Run advances matches with no Manual slots. Pause stops the next turn; a turn already choosing actions may finish.",
  ],
  [
    "#reset-button",
    "Restart",
    "Start a new match. If this game is not saved, you choose Save, Discard or Cancel first.",
  ],
  ...VISUAL_FILTER_CONTROL_HELP,
  [
    ".diagnostics > summary",
    "Technical Frame",
    "See the raw data this frame was drawn from, for developers.",
  ],
  ["[data-key='Escape']", "Clear Target", "Clear the target and turn keys off."],
]);

const battlefieldRenderer = new BattlefieldRenderer({
  battlefield: elements.battlefield,
  empty: elements.battlefieldEmpty,
});

const reducedMotionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
/** @type {Storage | null} */
let presentationStorage = null;
try {
  presentationStorage = window.sessionStorage;
} catch {
  // Storage is an optional presentation convenience, never an authority input.
}

const choreographer = new CombatChoreographer({
  painter: new SvgChoreographyPainter(),
  ledger: new ConsumedTransitionLedger({ storage: presentationStorage }),
  motionMode: reducedMotionPreference.matches ? "reduced" : "normal",
  /**
   * Reflect a new choreography snapshot in page state and command controls.
   * The presentation argument is the choreographer's current snapshot. This
   * synchronous callback updates display/availability without issuing a command.
   */
  onStateChange: (presentation) => {
    exposePresentationState(presentation);
    renderCommandAvailability();
  },
});

const replayTimelineElements = {
  root: elements.replayTimeline,
  keyboardTarget: document,
  // Tab on the field while a step loads is held (replay-controls.js).
  fieldRoot: elements.battlefieldShell,
  /**
   * Return whether a matched frame and presentation are currently installed.
   * This lets keyboard handling distinguish an installed view from a loading gap;
   * it does not decide whether another command may start.
   */
  keyboardAuthorityInstalled: () => installedPresentationAuthority() !== null,
  /**
   * Return whether installed presentation authority exists and the page is idle.
   * Replay keyboard commands remain fenced while a page request is busy.
   */
  keyboardEnabled: () => installedPresentationAuthority() !== null && !state.busy,
  /**
   * Clear the current replay inspection selection through its shared handler.
   * The handler may send the supported selection command; playback position is
   * not changed by this callback itself.
   */
  clearSelection: () => clearReplaySelection(),
  firstButton: elements.replayFirstButton,
  backTenButton: elements.replayBackTenButton,
  previousButton: elements.replayPreviousButton,
  playPauseButton: elements.replayPlayPauseButton,
  nextButton: elements.replayNextButton,
  forwardTenButton: elements.replayForwardTenButton,
  lastButton: elements.replayLastButton,
  slider: elements.replayFrameSlider,
  position: elements.replayFramePosition,
  rateSelect: elements.replayPlaybackRate,
  status: elements.replayTransportStatus,
  /**
   * Return the recorded simulator tick at frameIndex, or null if unavailable.
   * The numeric frame index is a captured-frame position, not a simulator tick.
   * Require installed authority before consulting the recorded timeline.
   */
  tickForFrameIndex: (/** @type {number} */ frameIndex) =>
    installedPresentationAuthority() === null
      ? null
      : replayTimelineSimulatorStep(state.timeline, frameIndex),
};

/**
 * Return the playback snapshot used to draw timeline controls.
 * While a page request is busy, return a frozen copy marked ADVANCING unless the
 * controller is OFFLINE; otherwise return the original snapshot. This does not
 * change playback state or start a request.
 *
 * @param {ReturnType<ReplayPlaybackController["snapshot"]>} playback
 */
function replayTimelineRenderState(playback) {
  if (!state.busy || playback.transportState === REPLAY_TRANSPORT_STATES.OFFLINE) {
    return playback;
  }
  return Object.freeze({
    ...playback,
    transportState: REPLAY_TRANSPORT_STATES.ADVANCING,
  });
}

/**
 * The Replay Viewer's Rune Scrubber (the transport dock under the arena: the
 * Event Track and one button row), or null in the BattleClient and DevClient.
 * Created after the playback controller it drives; its update runs on every
 * playback state change.
 *
 * @type {ReturnType<typeof createRuneScrubber> | null}
 */
let runeScrubber = null;

const replayPlayback = new ReplayPlaybackController({
  // 1x everywhere: with the beat clock a quiet replay tick already lasts 1.7 s.
  playbackRate: 1,
  request: sendReplayTransportCommand,
  /**
   * Return a promise that resolves without a value when choreography settles.
   * Playback uses this barrier before requesting the next recorded frame.
   */
  waitForPresentation: () => choreographer.whenSettled(),
  /**
   * Return the choreographer's current normal/reduced motion preference.
   * Read a snapshot without starting an animation or changing playback.
   */
  getMotionMode: () => choreographer.snapshot().motionMode,
  /**
   * Apply the supplied playback snapshot to timeline and artifact controls.
   * Invalidate an export when its playback generation changed, synchronize animation
   * rate, and render the page only when the installed view is idle and rendering
   * is not suppressed. This callback does not send a replay request.
   */
  onStateChange: (playback) => {
    if (
      replayArtifactActionTransaction !== null &&
      playback.generation !== replayArtifactActionTransaction.playbackGeneration
    ) {
      invalidateReplayArtifactAction();
    }
    choreographer.setPlaybackRate(playback.playbackRate);
    updateRuneScrubber(playback);
    const timeline =
      installedPresentationAuthority() === null
        ? pendingPresentationSurfaceView(playback).replay.timeline
        : playback;
    renderReplayTimelineControls(
      replayTimelineElements,
      replayTimelineRenderState(timeline),
    );
    renderReplayArtifactActions(installedPresentationAuthority());
    if (
      installedPresentationAuthority() !== null &&
      !state.busy &&
      !suppressPlaybackStateRender &&
      playback.transportState !== "ADVANCING"
    ) {
      render();
    }
  },
  /**
   * Show a playback failure unless an error notice is already visible: one
   * plain sentence, "Could not move the replay. Try again, or press Reconnect."
   * The raw error is not shown on the page; it goes to the browser console in
   * one line (logRawError). Update connection presentation without replacing
   * an existing error notice.
   *
   * @param {unknown} error The failure the controller caught.
   */
  onError: (error) => {
    logRawError("Replay move failed", error);
    if (!state.notice || state.noticeLevel !== "error") {
      setNotice("Could not move the replay. Try again, or press Reconnect.", "error");
      renderConnection();
    }
  },
});

const panels = new DebuggerPanels({
  actionsPanel: isReplayMode() ? elements.replayRailActions : null,
  roster: elements.roster,
  rosterCount: elements.rosterCount,
  selectionCard: elements.selectionCard,
  pendingHeading: elements.pendingHeading,
  pendingCount: elements.pendingCount,
  pendingScope: elements.pendingScope,
  pendingCard: elements.pendingCard,
  acceptedCard: elements.acceptedCard,
  acceptedAnnouncement: elements.acceptedAnnouncement,
  diagnosticsCard: elements.diagnosticsCard,
  onCommand: dispatchPanelCommand,
});

const tooltipController = createTooltipController({
  root: document.body,
  tooltip: elements.visualTooltip,
  title: elements.visualTooltipTitle,
  details: elements.visualTooltipDetails,
  // The Champion Card's box (owner, 2 October 2026): the Replay Viewer's
  // field (its #arena adds no box there), live play's #arena. The card stays
  // inside it; only a Hold Shift card that would shrink below 85% there grows
  // past it.
  championCardBox: () => (isReplayMode() ? elements.battlefieldShell : elements.arena),
});
tooltipController.setSymbolTipsEnabled(visualFilterState.symbol_explanations !== false);

// V5 chrome parts, mounted once the product is known. Each reads only the
// shown audience's data for anything drawn on or over the arena.
const replayProductPage = isReplayMode();

/** The HUD: party frames (and the score strip's Tab stop) for every client. */
const hud = createHud({
  frames: replayProductPage ? elements.teamRosters : elements.partyFrames,
  extraRoots: [elements.matchScoreboard],
  onSelect: (publicAgentId) => selectAgentFromHud(publicAgentId),
  onRefuse: (sentence) => setNotice(sentence),
  // Live play: a scorebug portrait's screen-reader copy of the Champion Card
  // comes from the visible hover card's own facts, so both read the same.
  cardFacts: replayProductPage
    ? null
    : (subject) => {
        const installed = installedPresentationAuthority();
        return championCardFacts({
          presentation: installed?.presentation ?? null,
          combatConfiguration: installed?.transport?.combat_configuration,
          subject,
        });
      },
});

/** The Replay Viewer's six-tab rail (Roster, Agent, Metrics, Actions, Visuals, Frame). */
const ledgerRail = replayProductPage ? mountLedgerRail(document) : null;

/**
 * The Replay Viewer's Class Guide: the BattleClient rail's Class Guide card
 * (tactics-turn.js classGuideCardHtml, the one owner of its words) for the
 * selected champion, under the Champion Frame in the Agent tab. Null outside
 * the Replay Viewer.
 */
const replayClassGuide = (() => {
  if (!replayProductPage) return null;
  const card = document.createElement("section");
  card.id = "replay-class-card";
  card.className = "cg replay-class-card";
  card.setAttribute("aria-label", "Class Guide");
  card.hidden = true;
  elements.championFrame.after(card);
  return card;
})();

/**
 * The Replay Viewer's Full Class Details (owner, 4 Oct 2026, Micro List M22): a
 * closed disclosure under the Class Guide holding the facts only the old long
 * Agent card showed: the order name, then the old card's own class part
 * (panels.js renderSemanticInspector: Class Overview, the Tactical Guide's
 * role, strength and weakness, every class mechanic such as body radius and the
 * out-of-combat delay and regeneration, the passive and its rules, and the rule
 * notes). renderReplayClassDetails fills it. Null outside the Replay Viewer.
 */
const replayClassDetails = (() => {
  if (replayClassGuide === null) return null;
  const details = document.createElement("details");
  details.id = "replay-class-details";
  details.className = "replay-class-details";
  details.hidden = true;
  const summary = document.createElement("summary");
  summary.className = "replay-class-details__summary";
  summary.textContent = "Full Class Details";
  const body = document.createElement("div");
  body.className = "replay-class-details__body";
  details.append(summary, body);
  replayClassGuide.after(details);
  return Object.freeze({ details, body });
})();

/** The champion and class whose facts #replay-class-details holds, or null. */
let replayClassDetailsKey = /** @type {string | null} */ (null);

// The Replay Viewer keeps its Visual Filters in the rail's Visuals tab, so
// its top-bar Visual Filters button opens that tab instead of the live
// drawer (which stays empty there).
if (ledgerRail !== null) {
  const filtersButton = elements.visualFiltersButton;
  filtersButton.removeAttribute("popovertarget");
  filtersButton.removeAttribute("aria-haspopup");
  filtersButton.setAttribute("aria-controls", "replay-rail-visuals");
  filtersButton.addEventListener("click", () => ledgerRail.selectTab("visuals"));
}

/** The camera switch: Replay Viewer transport slot, or the field corner live. */
const cameraSwitch =
  productIdentity === null
    ? null
    : createCameraSwitch({
        slot: replayProductPage
          ? elements.replayCameraSwitch
          : elements.liveCameraSwitch,
        product: replayProductPage ? "replay_viewer" : "combat_debugger",
        getFilters: () => visualFilterState,
        reducedMotion: () => choreographer.snapshot().motionMode !== "normal",
        applyFrame: (frame, { ease }) => applyCameraFrameWithEase(frame, ease),
        // The camera's buttons are the arena's V and B keys (owner, 4 Oct
        // 2026: no camera buttons in the bottom row; no lit key is Whole Map).
        // Its tips sit in the scorebug's empty score half, covering nothing
        // (owner, 3 Oct 2026, Micro List M7).
        ...(replayProductPage
          ? {
              switchCameras: /** @type {const} */ ([]),
              tipHost: elements.cameraTips,
            }
          : {}),
      });

/** Where the Frame Explanation list goes: the Frame tab, or under Your Turn live. */
const frameExplanationList = document.createElement("section");
frameExplanationList.id = "frame-explanation-list";
frameExplanationList.className = "frame-explanation-list";
frameExplanationList.setAttribute("aria-label", "Frame Explanation");
frameExplanationList.hidden = true;
if (replayProductPage) {
  elements.replayRailFrame.prepend(frameExplanationList);
} else {
  elements.thisTurn.after(frameExplanationList);
}

/** Frame Explanation (F): numbered marks and a list for the shown tick. */
const frameExplanation =
  productIdentity === null
    ? null
    : createFrameExplanation({
        product: replayProductPage ? "replay_viewer" : "combat_debugger",
        getTickInput: () =>
          authorizedPresentationFrameExplanationInput(
            installedPresentationAuthority()?.presentation ?? null,
          ),
        getTransform: () => battlefieldRenderer.transform,
        getLayer: () => battlefieldRenderer.layers.explain,
        listHost: frameExplanationList,
        // Every client's one Frame Explanation button is the arena's corner
        // key (owner, 3 Oct 2026; live play since 4 Oct 2026).
        button: elements.arenaFrameExplanationButton,
        isPaused: () => !replayPlayback.snapshot().playing,
        pause: () => replayPlayback.pause("frame_explanation"),
        canExplain: () => installedPresentationAuthority() !== null && !state.busy,
        onOpenChange: (open) => {
          if (open && ledgerRail !== null) ledgerRail.selectTab("frame");
        },
        registerSymbolTip: (node, record) => {
          const descriptor = frameExplanationTooltip(record);
          if (descriptor) {
            registerSymbolOwner(node, descriptor, `frame-explanation:${record.n}`);
          }
        },
      });

// The arena's on-screen keys, stacked at its top right (owner, 4 Oct 2026):
// the only Everyone In View (V), Battle Cam (B), Frame Explanation (F) and
// Clear Selection (Esc) buttons. V, B and Esc press their key through the one
// command set; Frame Explanation is createFrameExplanation's own button
// (above). The Replay Viewer first; live play (the BattleClient and the
// DevClient) wires V and F below.
if (replayProductPage && productIdentity !== null) {
  wireArenaButtons({
    group: elements.arenaButtons,
    everyoneInView: elements.arenaEveryoneInViewButton,
    battleCam: elements.arenaBattleCamButton,
    clearSelection: elements.arenaClearSelectionButton,
    product: "replay_viewer",
    cameraSlot: elements.replayCameraSwitch,
  });
}

// Live play: the same two keys (owner, 4 Oct 2026). A click on Everyone In
// View passes the same {interactive} fact the arena's V key passes
// (controls.js bindBattlefieldControls), and its lit state follows the live
// camera slot.
if (!replayProductPage && productIdentity !== null) {
  wireArenaButtons({
    group: elements.arenaButtons,
    everyoneInView: elements.arenaEveryoneInViewButton,
    product: "combat_debugger",
    cameraSlot: elements.liveCameraSwitch,
    context: () => ({ interactive: liveBattlefieldCommandsInteractive() }),
  });
}

if (replayProductPage) {
  runeScrubber = createRuneScrubber(elements.chronicle, {
    controller: replayPlayback,
    loadMarks: () => getReplayTimelineMarks(state.token),
    tickForFrameIndex: replayTimelineElements.tickForFrameIndex,
  });
  bindReplayAgentStepping({
    getPresentation: () => state.presentation,
    getCurrentPublicAgentId: () => activatedAgentPublicId ?? null,
    isFieldTarget: (target) =>
      target instanceof Node && elements.battlefieldShell.contains(target),
    onCommand: (command) => {
      void dispatchPanelCommand(command);
    },
  });
}

/** @type {string | null} */
let activatedAgentPublicId = null;

let lastBattlefieldSizeKey = "";
/** @type {number | null} */
let pendingResizeFrame = null;
/** @type {{session_id: string, run_generation: number, revision: number,
 * label: string, perform: (choice: "save" | "discard") => Promise<unknown>} | null} */
let pendingRecordingReplacement = null;
/** @type {string | null} */
let openedReplayEpisodeId = null;

/**
 * Remove the element attributes that expose a presentation-owned tooltip.
 * Also remove its accessible description. Old weak-map entries may still exist,
 * but the missing owner attribute prevents them from being selected.
 *
 * @param {Element} element
 */
function clearPresentationTooltipOwner(element) {
  element.removeAttribute("data-tooltip-owner");
  element.removeAttribute("data-tooltip-kind");
  element.removeAttribute("data-tooltip-text");
  element.removeAttribute("data-tooltip-tone");
  element.removeAttribute("data-tooltip-accent");
  element.removeAttribute("aria-description");
}

/**
 * Clear scientific labels and disable controls while no presentation is installed.
 * Reset replay, recording, selection, and tooltip surfaces, and close any pending
 * recording-discard dialog. This paints an unavailable state without guessing
 * facts from an unjoined transport frame.
 */
function renderPendingPresentationChrome() {
  const pending = pendingPresentationSurfaceView(replayPlayback.snapshot());
  clearPresentationTooltipOwner(elements.replayArtifactReference);
  clearPresentationTooltipOwner(elements.replayCompletionBadge);
  clearPresentationTooltipOwner(elements.replayProcessingBadge);
  elements.replayArtifactReference.removeAttribute("title");
  elements.replayArtifactReference.textContent = pending.replay.artifactReference;
  elements.replayCompletionBadge.textContent = pending.replay.completion;
  elements.replayProcessingBadge.textContent = pending.replay.processing;
  elements.replayEndReason.textContent = pending.replay.endReason;
  elements.replayRangesButton.setAttribute("aria-pressed", "false");
  elements.replayRangesButton.disabled = true;
  elements.replayClearReferenceButton.disabled = true;
  renderReplayTimelineControls(replayTimelineElements, pending.replay.timeline);
  renderReplayArtifactActions(null);

  elements.terminalBadge.hidden = pending.terminal.hidden;
  elements.terminalBadge.textContent = pending.terminal.text;
  elements.viewSelect.value = pending.viewMode;
  elements.viewSelect.disabled = true;
  elements.scenarioDescription.textContent = pending.scenarioDescription;

  elements.recordingPanel.toggleAttribute("hidden", pending.recording.hidden);
  elements.recordingBadge.toggleAttribute("hidden", pending.recording.hidden);
  elements.recordingPillText.textContent = pending.recording.badgeText;
  delete elements.recordingBadge.dataset.lifecycle;
  delete document.documentElement.dataset.recordingLifecycle;
  elements.recordingLifecycle.textContent = pending.recording.lifecycle;
  elements.recordingProgress.textContent = pending.recording.progress;
  elements.recordingCompletion.textContent = pending.recording.completion;
  elements.recordingPersistenceFact.toggleAttribute("hidden", true);
  elements.recordingPersistenceError.textContent = pending.recording.persistence;
  elements.recordingStatusNote.textContent = pending.recording.status;
  paintSaveFailedReason(null);
  for (const control of [
    elements.recordingFinishButton,
    elements.recordingReviewButton,
    elements.recordingRetryButton,
  ]) {
    control.toggleAttribute("hidden", true);
    control.disabled = true;
  }
  elements.recordingSaveAsControl.toggleAttribute("hidden", true);
  elements.recordingSaveAsInput.disabled = true;
  elements.recordingSaveAsButton.disabled = true;
  elements.recordingDiscardIntent.textContent = pending.recording.status;
  elements.recordingDiscardConfirmButton.disabled = true;
  elements.recordingSaveReplaceButton.disabled = true;
  elements.recordingReplayLink.hidden = true;
  if (elements.recordingDiscardDialog.open) {
    elements.recordingDiscardDialog.close();
  }
  pendingRecordingReplacement = null;
}

/**
 * Remove the installed scientific presentation for the supplied reason.
 * Save permitted display preferences, hold the workspace height during replacement,
 * clear scenes and panels, and invalidate pending display/download work. Raw
 * transport may remain for diagnostics and request accounting; it cannot authorize
 * what the page displays.
 *
 * @param {string} reason
 */
function clearPresentationAuthority(reason) {
  renderInstalledMatch(null);
  renderMatchMomentCards(null);
  renderFallenFeed(null);
  elements.metricProgress.hidden = true;
  elements.metricRows.setAttribute("aria-busy", "false");
  elements.recordingMetricsProgress.hidden = true;
  elements.recordingMetricsHelp.hidden = true;
  invalidateReplayArtifactAction();
  holdWorkspaceHeightDuringAuthorityInstall();
  savePresentationPreferenceBeforeClear();
  enterPendingPresentationPreferenceState();
  state.authority = null;
  state.presentation = null;
  tooltipController.hide();
  choreographer.clear(reason);
  battlefieldRenderer.render(null, {
    offline: true,
    visualFilterState,
  });
  panels.render(null, {
    busy: true,
    shuttingDown: state.shuttingDown,
    resyncRequired: state.resyncRequired,
    offline: true,
  });
  elements.pendingCard.removeAttribute("data-submission-scope");
  elements.pendingCard.removeAttribute("data-inspection-state");
  elements.pendingCard.removeAttribute("data-pending-count");
  elements.pendingHeading.textContent = "Inspection unavailable";
  elements.pendingCount.textContent = "0 actors";
  elements.pendingScope.textContent = "Waiting for authorized action details.";
  elements.stepValue.textContent = "—";
  elements.transitionValue.textContent = "—";
  elements.audienceBadge.textContent = "View Unavailable";
  elements.audienceBadge.dataset.audience = "unavailable";
  document.documentElement.dataset.audience = "unavailable";
  renderPendingPresentationChrome();
  elements.liveRangesButton.disabled = true;
  elements.resetButton.disabled = true;
  elements.exitButton.disabled = true;
  for (const control of [
    elements.recordingFinishButton,
    elements.recordingReviewButton,
    elements.recordingRetryButton,
    elements.recordingSaveAsInput,
    elements.recordingSaveAsButton,
    elements.recordingDiscardConfirmButton,
    elements.recordingSaveReplaceButton,
  ]) {
    control.disabled = true;
  }
  if (elements.recordingDiscardDialog.open) {
    elements.recordingDiscardDialog.close();
  }
  pendingRecordingReplacement = null;
  elements.commandControlledActor.textContent = "Actor · unavailable";
  elements.commandControlledActor.removeAttribute("aria-label");
  delete elements.commandControlledActor.dataset.controlledSlot;
  const emptyTarget = document.createElement("option");
  emptyTarget.value = "";
  emptyTarget.textContent = "No authorized targets";
  elements.commandTargetSelect.replaceChildren(emptyTarget);
  elements.commandTargetSelect.disabled = true;
  if (elements.commandDeck) {
    for (const button of elements.commandDeck.querySelectorAll("button")) {
      /** @type {HTMLButtonElement} */ (button).disabled = true;
      if (button.hasAttribute("data-authoritative-available")) {
        clearPresentationTooltipOwner(button);
        button.removeAttribute("data-authoritative-available");
        button.removeAttribute("data-selected");
        button.setAttribute("aria-disabled", "true");
        button.setAttribute("aria-pressed", "false");
      }
    }
  }
  elements.agentDetails.removeAttribute("data-tone");
  elements.agentDetails.removeAttribute("data-accent");
  elements.selectionHeading.textContent = AUTHORIZED_INSPECTOR_TITLE;
  elements.battlefield.removeAttribute("aria-activedescendant");
  document.documentElement.dataset.presentationAuthority = "pending";
}

/**
 * Prepare the page for an authority request with the supplied reason and policy.
 * With retain_last_authorized and an installed pair, keep its display preferences
 * and mark the retained surface pending; the hover tip is hidden, except during
 * autoplay's own next-frame fetch (state.quietBusy), when it stays up until the
 * next render refreshes it. Otherwise clear presentation authority.
 * This does not send the request or install its result.
 *
 * @param {string} reason
 * @param {"retain_last_authorized" | "clear"} pendingPolicy
 */
function beginPresentationAuthorityAttempt(reason, pendingPolicy) {
  if (
    pendingPolicy === "retain_last_authorized" &&
    installedPresentationAuthority() !== null
  ) {
    retainPresentationPreferenceAcrossBusyRender();
    // Autoplay's own next-frame fetch (state.quietBusy) keeps a hover tip up:
    // hiding it here made it blink off for ~80 ms before every played tick.
    // The next render's refresh redraws it for the new tick.
    if (!state.quietBusy) tooltipController.hide();
    document.documentElement.dataset.presentationAuthority = "retained";
    return;
  }
  clearPresentationAuthority(reason);
}

/**
 * Keep the current workspace height while its scientific content is replaced.
 * Save the previous inline minimum height once, then use the measured positive
 * height in CSS pixels. Missing or nonpositive measurements leave the style alone.
 */
function holdWorkspaceHeightDuringAuthorityInstall() {
  if (workspaceMinimumHeightBeforeAuthorityInstall !== null) {
    return;
  }
  const height = elements.workspace.getBoundingClientRect().height;
  if (!Number.isFinite(height) || height <= 0) {
    return;
  }
  workspaceMinimumHeightBeforeAuthorityInstall = elements.workspace.style.minHeight;
  elements.workspace.style.minHeight = `${Math.ceil(height)}px`;
}

/**
 * Restore the workspace minimum-height style saved before authority replacement.
 * Do nothing when no height is held; clear the saved value after restoration.
 */
function releaseWorkspaceHeightAfterAuthorityInstall() {
  if (workspaceMinimumHeightBeforeAuthorityInstall === null) {
    return;
  }
  const previous = workspaceMinimumHeightBeforeAuthorityInstall;
  workspaceMinimumHeightBeforeAuthorityInstall = null;
  if (previous.length === 0) {
    elements.workspace.style.removeProperty("min-height");
  } else {
    elements.workspace.style.minHeight = previous;
  }
}

/**
 * Name who a presentation is shown to: its presentation kind (Oracle or an
 * Agent POV kind, live or replay) and, in Agent POV, the recipient's public
 * ID. Two frames of the same view and recipient give the same key; a view
 * switch or a new POV agent gives a different one. Returns "" for anything
 * that is not an object (no presentation installed yet). Pure.
 *
 * @param {unknown} presentation The installed presentation, or null.
 * @returns {string}
 */
function viewAudienceKey(presentation) {
  if (!isRecord(presentation)) return "";
  return `${String(presentation.presentation_kind)}:${String(
    presentation.authority?.recipient_public_agent_id ?? "",
  )}`;
}

/**
 * Install a validated transport/presentation pair and its replay timeline.
 * Require the trusted join marker, matching product, and a usable preference key;
 * throw on an invalid pair before treating it as authority. Reconcile local
 * preferences, publish confirmed combat settings, and request a content repaint.
 * A change of view kind or Agent POV recipient (viewAudienceKey) also clears
 * the Fallen panel's memory (fallenFeedMemory), so the panel after any view
 * change equals a freshly loaded page.
 *
 * @param {Readonly<Record<string, any>>} joined
 */
function installJoinedAuthority(joined) {
  if (
    !isJoinedTransportAndAuthorizedPresentationV1(joined) ||
    !isRecord(joined.transport) ||
    !isAuthorizedPresentationFrame(joined.presentation) ||
    (joined.transport.viewer_mode === "replay" && !isRecord(joined.timeline))
  ) {
    throw new TypeError("Joined browser authority is incomplete or unbranded.");
  }
  assertFrameMatchesProductIdentity(joined.transport);
  const preferenceKey = authorizedPresentationPreferenceKey(joined.presentation);
  if (preferenceKey === null) {
    throw new TypeError("Authorized presentation has no certified preference key.");
  }
  installActivePresentationPreference(joined.presentation, preferenceKey, {
    previousAuthority: state.authority,
    nextAuthority: joined,
  });
  if (
    authorizedPresentationAudience(joined.presentation) === "researcher" &&
    typeof joined.transport.show_ranges === "boolean"
  ) {
    confirmedResearcherRangesVisible = joined.transport.show_ranges;
  }
  if (
    !agentLocalRangesInitialized &&
    typeof joined.transport.show_ranges === "boolean"
  ) {
    agentLocalRangesVisible = joined.transport.show_ranges;
    agentLocalRangesInitialized = true;
  }
  if (viewAudienceKey(state.presentation) !== viewAudienceKey(joined.presentation)) {
    fallenFeedMemory = null;
  }
  state.authority = joined;
  state.frame = joined.transport;
  state.presentation = joined.presentation;
  state.timeline = isRecord(joined.timeline) ? joined.timeline : null;
  publishInstalledCombatConfiguration(joined.transport);
  presentationPreferenceNeedsContentRender = true;
  document.documentElement.dataset.presentationAuthority = "installed";
}

/**
 * Prepare a trusted joined pair for installation without installing it.
 * Check product identity and, when a previous replay authority is supplied, replay
 * continuity. Replay pairs also fetch and join their timeline; live pairs return
 * directly. Previous authority defaults to null and continuity to stale_resync.
 * Validation and request failures reject the returned promise.
 *
 * @param {unknown} joined
 * @param {{
 *   previousAuthority?: Readonly<Record<string, any>> | null,
 *   continuityResult?: unknown,
 * }} options
 */
async function prepareJoinedAuthority(
  joined,
  { previousAuthority = null, continuityResult = "stale_resync" } = {},
) {
  if (!isJoinedTransportAndAuthorizedPresentationV1(joined)) {
    throw new TypeError(
      "Browser authority pair is not certified by the join boundary.",
    );
  }
  const certified = /** @type {Readonly<Record<string, any>>} */ (joined);
  const frame = certified.transport;
  assertFrameMatchesProductIdentity(frame);
  if (
    frame.viewer_mode === "replay" &&
    previousAuthority?.transport?.viewer_mode === "replay"
  ) {
    validateReplayTransportContinuityV1(previousAuthority, certified, continuityResult);
  }
  if (frame.viewer_mode !== "replay") {
    return certified;
  }
  const rawTimeline = await getReplayTimeline(state.token);
  return joinReplayTransportAndTimelineV1(certified, rawTimeline);
}

const presentationInstallation = new PresentationInstallCoordinator({
  onAttemptBegin: beginPresentationAuthorityAttempt,
  install: installJoinedAuthority,
  isJoinRace: isPresentationJoinRace,
});

const AUTHORIZED_INSPECTOR_TITLE = "Comprehensive Agent Class Details";

/**
 * Set the inspector heading and accent for the activated authorized agent.
 * Reset them when no agent is activated. Read identity through the authorized
 * inspector view rather than a stale DOM row.
 */
function applyAuthorizedInspectorChrome() {
  if (activatedAgentPublicId === null) {
    elements.selectionHeading.textContent = AUTHORIZED_INSPECTOR_TITLE;
    elements.agentDetails.removeAttribute("data-tone");
    elements.agentDetails.removeAttribute("data-accent");
    return;
  }
  const inspector = authorizedInspectorView(
    state.presentation,
    installedLocalInspectedPresentationKey(state.presentation),
    activatedAgentPublicId,
  );
  elements.selectionHeading.textContent =
    inspector?.title ?? AUTHORIZED_INSPECTOR_TITLE;
  elements.agentDetails.removeAttribute("data-tone");
  if (typeof inspector?.owner_class_accent === "string") {
    elements.agentDetails.dataset.accent = inspector.owner_class_accent;
  } else {
    elements.agentDetails.removeAttribute("data-accent");
  }
}

/**
 * Attach the static control-help registry to matching page elements.
 * Entries without a kind use control; missing matching elements need no handler.
 */
function registerControlHelp() {
  for (const [selector, title, summary, kind = "control"] of CONTROL_HELP) {
    for (const control of document.querySelectorAll(selector)) {
      registerControlHelpOwner(control, selector, title, summary, kind);
    }
  }
}

/**
 * Attach a non-inspectable help tooltip to one control.
 * Use the supplied selector as its stable ID and the supplied title and summary
 * as prose; kind defaults to control. Battlefield help follows the pointer, while
 * other controls anchor the tooltip to their element.
 *
 * @param {Element} control
 * @param {string} selector
 * @param {string} title
 * @param {string} summary
 * @param {string} [kind]
 */
function registerControlHelpOwner(control, selector, title, summary, kind = "control") {
  registerTooltipOwner(
    control,
    createSemanticDescriptor({
      kind,
      id: `control:${selector}:${title}`,
      title,
      tone: "information",
      accent: "none",
      summary,
      rows: [],
      sections: [],
      metadata: { compact: true, full: false },
      anchor: selector === "#battlefield" ? "pointer" : "element",
    }),
    { inspectable: false },
  );
}

/**
 * Refresh utility help for the currently installed product and audience.
 * Explain range, selection, and Tab behavior using the same authority and
 * availability checks as their controls, including page-local Agent POV choices.
 * With Manual champions, Next and Previous Actor say they move the orders to
 * your next champion that can take orders (ownChampionTabStep). In Policies
 * Only they say they look at the next champion: in Agent POV one the team can
 * see (Python skips unseen champions there).
 */
function registerAuthorityAwareUtilityHelp() {
  const presentation = state.presentation;
  const installed =
    isAuthorizedPresentationFrame(presentation) && installedAuthorityIsCoherent();
  const audience = installed ? authorizedPresentationAudience(presentation) : null;
  const agentPov = audience === "agent_pov";
  const researcher = audience === "researcher";
  registerControlHelpOwner(
    elements.liveRangesButton,
    "#live-ranges-button",
    "Ranges",
    agentPov
      ? "Show or hide the reach rings your team may see. Nothing is sent to the game."
      : researcher
        ? "Show or hide the reach rings."
        : "Reach rings are not ready until the game's first frame has loaded.",
  );
  registerControlHelpOwner(
    elements.replayRangesButton,
    "#replay-ranges-button",
    "Replay Ranges",
    agentPov
      ? "Show or hide the reach rings of the champion you look at. The replay and the view do not change."
      : researcher
        ? "Show or hide the reach rings in this replay."
        : "Reach rings are not ready until the replay's first frame has loaded.",
  );
  registerControlHelpOwner(
    elements.replayClearReferenceButton,
    "#replay-clear-reference-button",
    "Clear Selection",
    agentPov
      ? "Stop looking at the selected champion. The replay and the view do not change."
      : researcher
        ? "Stop looking at the selected champion."
        : "Nothing is selected until the replay's first frame has loaded.",
  );
  // With Manual champions, Tab and these buttons move the orders only between
  // your champions that can take orders (ownChampionTabStep).
  const ownChampionsOnly =
    installed &&
    !isReplayMode() &&
    asArray(state.frame?.combat_configuration?.manual_slots).length > 0;
  for (const [selector, title, oracleSummary, ownSummary] of [
    [
      "[data-key='Tab']:not([data-shift])",
      "Next Actor",
      "Move Oracle View control to the next active actor.",
      "Give the orders to your next champion that can take orders. A policy's champion is skipped; click it to look at it. Staged orders are kept.",
    ],
    [
      "[data-key='Tab'][data-shift='true']",
      "Previous Actor",
      "Move Oracle View control to the previous active actor.",
      "Give the orders to your previous champion that can take orders. A policy's champion is skipped; click it to look at it. Staged orders are kept.",
    ],
  ]) {
    for (const control of document.querySelectorAll(selector)) {
      const oracleActorCyclingAvailable =
        researcher && control instanceof HTMLButtonElement && !control.disabled;
      const agentActorCyclingAvailable =
        agentPov && control instanceof HTMLButtonElement && !control.disabled;
      registerControlHelpOwner(
        control,
        selector,
        title,
        ownChampionsOnly && (oracleActorCyclingAvailable || agentActorCyclingAvailable)
          ? ownSummary
          : agentActorCyclingAvailable
            ? `${oracleSummary.replace("Oracle View", "Agent POV").replace("active actor", "champion your team can see")} Staged drafts are preserved.`
            : agentPov
              ? "Actor cycling is unavailable in the current Agent POV state. Tab and Shift+Tab retain native browser focus navigation."
              : oracleActorCyclingAvailable
                ? oracleSummary
                : researcher
                  ? "Actor cycling is unavailable in the current Oracle View state. Tab and Shift+Tab retain native browser focus navigation."
                  : "Actor cycling is unavailable until one coherent authorized live frame is installed.",
      );
    }
  }
}

/**
 * Return whether a value is a non-null object that is not an array.
 * This is a shape guard only; it does not validate a protocol or certify authority.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Cancel the queued animation-frame preference restore, if present.
 * Clear its handle so a later render can schedule a fresh restore.
 */
function cancelPendingPresentationPreferenceRestore() {
  if (pendingPresentationPreferenceRestoreFrame === null) {
    return;
  }
  window.cancelAnimationFrame(pendingPresentationPreferenceRestoreFrame);
  pendingPresentationPreferenceRestoreFrame = null;
}

/**
 * Set a details panel open or closed without treating that change as user intent.
 * Record the expected final toggle state before assigning open, because browsers
 * queue and may combine toggle events. Do nothing when the state already matches.
 *
 * @param {HTMLDetailsElement} panel
 * @param {boolean} open
 */
function setProgrammaticDisclosureOpen(panel, open) {
  if (panel.open === open) {
    return;
  }
  expectedDisclosureToggles.set(panel, Object.freeze({ open }));
  panel.open = open;
}

/**
 * Enable or disable the scientific details panels and their keyboard summaries.
 * When unavailable, close them unless preserveOpen is true; that option defaults
 * to false. Use inert and accessibility attributes to block interaction while
 * presentation authority is missing or a request is busy.
 *
 * @param {boolean} available
 * @param {{preserveOpen?: boolean}} [options]
 */
function setScientificDisclosureAvailability(available, { preserveOpen = false } = {}) {
  for (const { panel, body } of scientificDisclosures) {
    const summary = panel.querySelector(":scope > summary");
    if (!available) {
      if (!preserveOpen) {
        setProgrammaticDisclosureOpen(panel, false);
      }
      panel.setAttribute("inert", "");
      body.setAttribute("inert", "");
      if (summary instanceof HTMLElement) {
        summary.setAttribute("aria-disabled", "true");
        summary.setAttribute("tabindex", "-1");
      }
      continue;
    }
    panel.removeAttribute("inert");
    body.removeAttribute("inert");
    if (summary instanceof HTMLElement) {
      summary.removeAttribute("aria-disabled");
      summary.removeAttribute("tabindex");
    }
  }
}

/**
 * Return the active local preference only for this exact installed presentation.
 * Return null for an untrusted presentation, an incoherent installed pair, a
 * different object, or a mismatched authority key.
 *
 * @param {unknown} presentation
 * @returns {typeof activePresentationPreference}
 */
function installedActivePresentationPreference(presentation) {
  if (
    activePresentationPreference === null ||
    !isAuthorizedPresentationFrame(presentation) ||
    !installedAuthorityIsCoherent() ||
    state.presentation !== presentation
  ) {
    return null;
  }
  const installedKey = authorizedPresentationPreferenceKey(presentation);
  return installedKey !== null &&
    sameAuthorizedPresentationPreferenceKey(
      activePresentationPreference.authorityKey,
      installedKey,
    )
    ? activePresentationPreference
    : null;
}

/**
 * Create fresh local display preferences for a presentation and authority key.
 * Initialize disclosure openness and zero scroll offsets, allow automatic inspector
 * opening, and start without keyboard focus. Agent POV and scripted live scenes
 * own local inspection; select the recipient only if its key exists in this scene.
 *
 * @param {Readonly<Record<string, any>>} presentation
 * @param {Readonly<{tuple: readonly unknown[], serialized: string}>} authorityKey
 * @returns {NonNullable<typeof activePresentationPreference>}
 */
function defaultPresentationPreference(presentation, authorityKey) {
  /** @type {Record<string, {open: boolean, scrollTop: number}>} */
  const disclosures = {};
  const replay = presentation.product_kind === "replay_viewer";
  for (const { panelId } of scientificDisclosures) {
    disclosures[panelId] = {
      open: disclosurePanelInitiallyOpen(panelId, replay),
      scrollTop: 0,
    };
  }
  const audience = authorizedPresentationAudience(presentation);
  const ownsLocalInspection =
    audience === "agent_pov" ||
    authorizedPresentationInspectionState(presentation).state_kind === "live_scripted";
  const recipientKey =
    audience === "agent_pov" ? presentation.authority.recipient_presentation_key : null;
  const inspectedPresentationKey =
    typeof recipientKey === "string" &&
    authorizedAgentForPresentationKey(presentation, recipientKey) !== null
      ? recipientKey
      : null;
  return {
    authorityKey,
    disclosures,
    agentDetailsAutoOpenAllowed: true,
    primaryFocus: null,
    localInspection: ownsLocalInspection
      ? { owned: true, presentationKey: inspectedPresentationKey }
      : { owned: false },
  };
}

/**
 * Recheck retained selection and focus against a replacement authorized scene.
 * If a selected key disappeared, clear selection, close Agent Details, and
 * invalidate downloads. Rebind focus only through the new scene; do not silently
 * replace a missing selection with the recipient.
 *
 * @param {NonNullable<typeof activePresentationPreference>} preference
 * @param {Readonly<Record<string, any>>} presentation
 */
function reconcilePresentationPreference(preference, presentation) {
  if (
    preference.localInspection.owned &&
    preference.localInspection.presentationKey !== null &&
    authorizedAgentForPresentationKey(
      presentation,
      preference.localInspection.presentationKey,
    ) === null
  ) {
    invalidateReplayArtifactAction();
    preference.localInspection.presentationKey = null;
    preference.disclosures["agent-details"].open = false;
  }
  if (preference.primaryFocus !== null) {
    preference.primaryFocus = rebindAuthorizedPrimaryFocus(
      presentation,
      preference.primaryFocus,
    );
  }
}

/**
 * Resolve saved keyboard focus in the new authorized roster or battlefield.
 * Match the opaque presentation key; optional public-identity fallback defaults
 * to false and is used only for an approved recipient rotation. Return the new
 * focus record or null. Roster lookup uses the researcher directory, not fogged SVG.
 *
 * @param {unknown} presentation
 * @param {Readonly<{
 *   surface: "roster" | "battlefield",
 *   presentationKey: string,
 *   publicAgentId: string,
 * }>} focus
 * @param {Readonly<{allowPublicIdentityFallback?: boolean}>} [options]
 * @returns {null | {surface: "roster" | "battlefield", presentationKey: string, publicAgentId: string}}
 */
function rebindAuthorizedPrimaryFocus(
  presentation,
  focus,
  { allowPublicIdentityFallback = false } = {},
) {
  const scene =
    focus.surface === "roster"
      ? authorizedPresentationResearcherSceneView(presentation)
      : authorizedPresentationSceneView(presentation);
  const authorizedAgent = asArray(scene?.agents).find(
    (candidate) =>
      isRecord(candidate) &&
      (candidate.presentation_key === focus.presentationKey ||
        (allowPublicIdentityFallback &&
          candidate.public_agent_id === focus.publicAgentId)),
  );
  if (!isRecord(authorizedAgent)) {
    return null;
  }
  return {
    surface: focus.surface,
    presentationKey: String(authorizedAgent.presentation_key),
    publicAgentId: String(authorizedAgent.public_agent_id),
  };
}

/**
 * Return whether two preference keys describe a recipient change in one live
 * Agent POV episode. Require matching product, session, episode, and information
 * mode, with changed recipient and view identities. Other boundaries get fresh
 * preferences instead of borrowing old opaque keys.
 *
 * @param {Readonly<{tuple: readonly unknown[]}>} previous
 * @param {Readonly<{tuple: readonly unknown[]}>} next
 */
function isLiveAgentRecipientRotation(previous, next) {
  const left = previous.tuple;
  const right = next.tuple;
  const sameLiveAgentMode =
    right[3] === left[3] &&
    right[5] === left[5] &&
    ((left[3] === "live_no_shared_obs_agent_pov" && left[5] === "no_shared_obs") ||
      (left[3] === "live_shared_obs_agent_pov" &&
        left[5] === "shared_obs_visual_union"));
  return (
    left[0] === "combat_debugger" &&
    right[0] === left[0] &&
    right[1] === left[1] &&
    right[2] === left[2] &&
    sameLiveAgentMode &&
    left[4] === "agent_pov" &&
    right[4] === left[4] &&
    left[6] === null &&
    right[6] === null &&
    typeof left[7] === "string" &&
    typeof right[7] === "string" &&
    right[7] !== left[7] &&
    right[8] !== left[8]
  );
}

/**
 * Return whether an allowed live recipient rotation selects the saved focused agent.
 * Use the focus public ID only after the surrounding authority keys establish
 * that this is the same live Agent POV episode.
 *
 * @param {Readonly<{tuple: readonly unknown[]}>} previous
 * @param {Readonly<{tuple: readonly unknown[]}>} next
 * @param {Readonly<{publicAgentId: string}>} focus
 */
function isLiveAgentRecipientFocusRotation(previous, next, focus) {
  return (
    isLiveAgentRecipientRotation(previous, next) &&
    next.tuple[7] === focus.publicAgentId
  );
}

/**
 * Install or reconcile display preferences for a new presentation authority.
 * Cancel older restores and advance the preference generation. New authority keys
 * get fresh defaults, with narrowly allowed live/replay recipient rotations retaining
 * disclosures and rebinding focus. Optional previous and next authorities default
 * to null; no preference may authorize scientific facts.
 *
 * @param {Readonly<Record<string, any>>} presentation
 * @param {Readonly<{tuple: readonly unknown[], serialized: string}>} authorityKey
 * @param {{
 *   previousAuthority?: Readonly<Record<string, any>> | null,
 *   nextAuthority?: Readonly<Record<string, any>> | null,
 * }} [options]
 */
function installActivePresentationPreference(
  presentation,
  authorityKey,
  { previousAuthority = null, nextAuthority = null } = {},
) {
  presentationPreferenceGeneration += 1;
  cancelPendingPresentationPreferenceRestore();
  if (
    activePresentationPreference === null ||
    !sameAuthorizedPresentationPreferenceKey(
      activePresentationPreference.authorityKey,
      authorityKey,
    )
  ) {
    const previousPreference = activePresentationPreference;
    const previousAuthorityKey = previousPreference?.authorityKey ?? null;
    const previousPrimaryFocus = previousPreference?.primaryFocus ?? null;
    const retainLiveAgentPreferences =
      previousPreference !== null &&
      isLiveAgentRecipientRotation(previousPreference.authorityKey, authorityKey);
    const retainReplayAgentPreferences =
      previousPreference !== null &&
      isReplayAgentRecipientRotation(previousAuthority, nextAuthority);
    activePresentationPreference = defaultPresentationPreference(
      presentation,
      authorityKey,
    );
    if (
      (retainLiveAgentPreferences || retainReplayAgentPreferences) &&
      previousPreference !== null
    ) {
      activePresentationPreference.disclosures = Object.fromEntries(
        Object.entries(previousPreference.disclosures).map(([panelId, saved]) => [
          panelId,
          { ...saved },
        ]),
      );
      activePresentationPreference.agentDetailsAutoOpenAllowed =
        previousPreference.agentDetailsAutoOpenAllowed;
    }
    if (
      retainReplayAgentPreferences &&
      pendingReplayRecipientActivation !== null &&
      pendingReplayRecipientActivation.sourcePreferenceGeneration ===
        presentationPreferenceGeneration - 1 &&
      pendingReplayRecipientActivation.recipientPublicAgentId === authorityKey.tuple[7]
    ) {
      activatedAgentPublicId = pendingReplayRecipientActivation.recipientPublicAgentId;
      if (
        pendingReplayRecipientActivation.autoOpenAgentDetails &&
        activePresentationPreference.agentDetailsAutoOpenAllowed
      ) {
        activePresentationPreference.disclosures["agent-details"].open = true;
      }
      pendingReplayRecipientActivation = null;
    }
    if (
      previousPrimaryFocus !== null &&
      previousAuthorityKey !== null &&
      isLiveAgentRecipientFocusRotation(
        previousAuthorityKey,
        authorityKey,
        previousPrimaryFocus,
      )
    ) {
      activePresentationPreference.primaryFocus = rebindAuthorizedPrimaryFocus(
        presentation,
        previousPrimaryFocus,
        { allowPublicIdentityFallback: true },
      );
    }
  } else {
    reconcilePresentationPreference(activePresentationPreference, presentation);
  }
}

/**
 * Read focused roster or battlefield agent identity through the supplied presentation.
 * Return its surface, opaque key, and public ID, or null when focus is elsewhere
 * or the identity is absent. DOM order and slot numbers do not establish identity.
 *
 * @param {Readonly<Record<string, any>>} presentation
 * @returns {null | {
 *   surface: "roster" | "battlefield",
 *   presentationKey: string,
 *   publicAgentId: string,
 * }}
 */
function focusedAuthorizedPrimaryAction(presentation) {
  const active = document.activeElement;
  if (!(active instanceof Element)) {
    return null;
  }
  const rosterAction = active.closest(
    "#roster .roster-primary-action[data-presentation-key]",
  );
  const battlefieldAction = active.closest(
    "#battlefield .agent[data-presentation-key]",
  );
  const action = rosterAction ?? battlefieldAction;
  const presentationKey = action?.getAttribute("data-presentation-key");
  const authorizedAgent =
    rosterAction !== null
      ? authorizedResearcherAgentForPresentationKey(presentation, presentationKey)
      : authorizedAgentForPresentationKey(presentation, presentationKey);
  if (typeof presentationKey !== "string" || authorizedAgent === null) {
    return null;
  }
  return {
    surface: rosterAction !== null ? "roster" : "battlefield",
    presentationKey,
    publicAgentId: String(authorizedAgent.public_agent_id),
  };
}

/**
 * Save current disclosure openness, scroll positions, and authorized focus.
 * A user closing Agent Details disables later automatic opening; expected
 * programmatic toggles do not. Mutate the supplied active preference in place.
 *
 * @param {NonNullable<typeof activePresentationPreference>} preference
 * @param {Readonly<Record<string, any>>} presentation
 */
function capturePresentationPreference(preference, presentation) {
  for (const { panelId, panel, body } of scientificDisclosures) {
    const saved = preference.disclosures[panelId];
    const expected = expectedDisclosureToggles.get(panel);
    const userClosedBeforeToggle =
      saved?.open === true && !panel.open && expected?.open !== false;
    if (panelId === "agent-details" && userClosedBeforeToggle) {
      preference.agentDetailsAutoOpenAllowed = false;
    }
    preference.disclosures[panelId] = {
      open: panel.open,
      scrollTop:
        panel.open || userClosedBeforeToggle ? body.scrollTop : (saved?.scrollTop ?? 0),
    };
  }
  preference.primaryFocus = focusedAuthorizedPrimaryAction(presentation);
}

/**
 * Save preferences before removing the current authorized content.
 * If focus belongs to that content, move it to Help and retain a restore anchor
 * only for a recognized primary agent action. Do nothing without matching authority.
 * Focus on the battlefield element itself stays there: the clear empties the
 * battlefield but keeps the element, so keys stay on across a view switch
 * (Agent POV <-> Oracle View), as the owner asked on 3 October 2026. Only
 * focus on something inside it (a body, a mark) or in a disclosure panel,
 * which the clear removes, moves to Help.
 */
function savePresentationPreferenceBeforeClear() {
  const presentation = state.presentation;
  const preference = installedActivePresentationPreference(presentation);
  if (preference === null || !isAuthorizedPresentationFrame(presentation)) {
    return;
  }
  capturePresentationPreference(preference, presentation);
  const active = document.activeElement;
  const presentationOwnedFocus =
    active instanceof Element &&
    active !== elements.battlefield &&
    (elements.battlefield.contains(active) ||
      scientificDisclosures.some(({ panel }) => panel.contains(active)));
  if (!presentationOwnedFocus) {
    pendingPrimaryFocusRestoreAnchor = null;
    return;
  }
  elements.helpButton.focus({ preventScroll: true });
  pendingPrimaryFocusRestoreAnchor =
    preference.primaryFocus === null ? null : elements.helpButton;
}

/**
 * Invalidate older preference restores and disable scientific disclosures.
 * Advance the generation and clear the content-render flag so stale callbacks
 * cannot reopen panels while authority is pending.
 */
function enterPendingPresentationPreferenceState() {
  presentationPreferenceGeneration += 1;
  presentationPreferenceNeedsContentRender = false;
  cancelPendingPresentationPreferenceRestore();
  setScientificDisclosureAvailability(false);
}

/**
 * Save current display choices before a normal content repaint.
 * Skip capture while new content or an earlier restore is pending; those states
 * would overwrite saved user choices with temporary DOM state.
 */
function capturePresentationPreferenceBeforeRender() {
  if (
    presentationPreferenceNeedsContentRender ||
    pendingPresentationPreferenceRestoreFrame !== null
  ) {
    return;
  }
  const presentation = state.presentation;
  const preference = installedActivePresentationPreference(presentation);
  if (preference === null || !isAuthorizedPresentationFrame(presentation)) {
    return;
  }
  capturePresentationPreference(preference, presentation);
  pendingPrimaryFocusRestoreAnchor =
    preference.primaryFocus === null ? null : document.activeElement;
}

/**
 * Save focus and disclosure choices before a busy repaint of the same authority.
 * Move a recognized primary focus to Help temporarily, and mark content for a
 * restore. Later restoration keeps focus only if the user has not moved elsewhere.
 */
function retainPresentationPreferenceAcrossBusyRender() {
  const presentation = state.presentation;
  const preference = installedActivePresentationPreference(presentation);
  if (preference === null || !isAuthorizedPresentationFrame(presentation)) {
    return;
  }
  capturePresentationPreference(preference, presentation);
  if (preference.primaryFocus === null) {
    pendingPrimaryFocusRestoreAnchor = null;
  } else {
    elements.helpButton.focus({ preventScroll: true });
    pendingPrimaryFocusRestoreAnchor = elements.helpButton;
  }
  presentationPreferenceNeedsContentRender = true;
}

/**
 * Queue scroll and primary-focus restoration for the next animation frame.
 * Cancel the older callback and require the same preference generation and
 * authority key. Restore focus only to a current enabled authorized element while
 * the saved focus anchor still owns focus; release the held workspace height.
 *
 * @param {NonNullable<typeof activePresentationPreference>} preference
 */
function schedulePresentationPreferenceRestore(preference) {
  cancelPendingPresentationPreferenceRestore();
  const generation = presentationPreferenceGeneration;
  const authorityKey = preference.authorityKey;
  pendingPresentationPreferenceRestoreFrame = window.requestAnimationFrame(() => {
    pendingPresentationPreferenceRestoreFrame = null;
    const installed = installedActivePresentationPreference(state.presentation);
    if (
      generation !== presentationPreferenceGeneration ||
      installed === null ||
      !sameAuthorizedPresentationPreferenceKey(installed.authorityKey, authorityKey)
    ) {
      return;
    }
    try {
      for (const { panelId, panel, body } of scientificDisclosures) {
        const saved = installed.disclosures[panelId];
        if (panel.open && saved?.open === true) {
          body.scrollTop = saved.scrollTop;
        }
      }
      const focus = installed.primaryFocus;
      const anchor = pendingPrimaryFocusRestoreAnchor;
      pendingPrimaryFocusRestoreAnchor = null;
      if (focus === null || anchor === null) {
        return;
      }
      const reboundFocus = rebindAuthorizedPrimaryFocus(state.presentation, focus);
      const active = document.activeElement;
      const focusWasNotMovedByUser =
        active === anchor || (!anchor.isConnected && active === document.body);
      if (!focusWasNotMovedByUser || reboundFocus === null) {
        return;
      }
      installed.primaryFocus = reboundFocus;
      const root =
        reboundFocus.surface === "roster" ? elements.roster : elements.battlefield;
      const selector =
        reboundFocus.surface === "roster"
          ? `.roster-primary-action[data-presentation-key="${CSS.escape(reboundFocus.presentationKey)}"]`
          : `.agent[data-presentation-key="${CSS.escape(reboundFocus.presentationKey)}"]`;
      const target = root.querySelector(selector);
      if (
        !(target instanceof HTMLElement || target instanceof SVGElement) ||
        target.getAttribute("aria-disabled") === "true" ||
        target.getAttribute("tabindex") === "-1" ||
        (target instanceof HTMLButtonElement && target.disabled)
      ) {
        return;
      }
      target.focus({ preventScroll: true });
    } finally {
      releaseWorkspaceHeightAfterAuthorityInstall();
    }
  });
}

/**
 * Restore disclosure choices after painting a coherent non-busy presentation.
 * Disable panels when authority is missing; retain their openness while busy.
 * Otherwise reopen saved panels and schedule guarded scroll/focus restoration.
 */
function restorePresentationPreferenceAfterRender() {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null) {
    setScientificDisclosureAvailability(false);
    if (!state.busy) {
      releaseWorkspaceHeightAfterAuthorityInstall();
    }
    return;
  }
  if (state.busy) {
    setScientificDisclosureAvailability(false, { preserveOpen: true });
    return;
  }
  setScientificDisclosureAvailability(true);
  for (const { panelId, panel } of scientificDisclosures) {
    setProgrammaticDisclosureOpen(
      panel,
      preference.disclosures[panelId]?.open === true,
    );
  }
  presentationPreferenceNeedsContentRender = false;
  schedulePresentationPreferenceRestore(preference);
}

/**
 * Open Agent Details only when the active preference permits automatic opening.
 * Return true after updating the saved and DOM states, or false when authority
 * is absent or the user previously disabled automatic opening by closing it.
 */
function openAgentDetails() {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null || !preference.agentDetailsAutoOpenAllowed) {
    return false;
  }
  preference.disclosures["agent-details"].open = true;
  setProgrammaticDisclosureOpen(elements.agentDetails, true);
  return true;
}

/**
 * Remember inspector activation until the requested replay recipient is installed.
 * For the already installed recipient, activate and open details locally. Without
 * active preferences clear pending activation; otherwise save its source generation
 * and whether a successful handoff should open the panel.
 *
 * @param {string} publicAgentId
 */
function stageReplayRecipientActivation(publicAgentId) {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null) {
    pendingReplayRecipientActivation = null;
    return;
  }
  if (state.presentation?.authority?.recipient_public_agent_id === publicAgentId) {
    pendingReplayRecipientActivation = null;
    activatedAgentPublicId = publicAgentId;
    openAgentDetails();
    return;
  }
  pendingReplayRecipientActivation = Object.freeze({
    sourcePreferenceGeneration: presentationPreferenceGeneration,
    recipientPublicAgentId: publicAgentId,
    autoOpenAgentDetails:
      preference.agentDetailsAutoOpenAllowed &&
      preference.disclosures["agent-details"].open !== true,
  });
}

/**
 * Close Agent Details in both saved preferences and the DOM when authority exists.
 * This programmatic close preserves permission for a later automatic opening.
 */
function closeAgentDetailsWithoutLatching() {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null) {
    return;
  }
  preference.disclosures["agent-details"].open = false;
  setProgrammaticDisclosureOpen(elements.agentDetails, false);
}

/**
 * Return whether the installed launcher product is the Replay Viewer.
 * Use product identity, not frame metadata, to select the page mode.
 */
function isReplayMode() {
  return productIdentity?.product_kind === "replay_viewer";
}

/**
 * Return the installed launcher product: "combat_debugger" (the BattleClient
 * and the DevClient combat screen) or "replay_viewer", or null when startup
 * refused the bootstrap. Reads the product identity only; pure.
 *
 * @returns {"combat_debugger" | "replay_viewer" | null}
 */
function installedProductKind() {
  return productIdentity?.product_kind ?? null;
}

/**
 * Return the display label for a controller value.
 * Known reactive and random controllers have explicit names; other values use
 * Manual. This label helper does not validate the requested configuration.
 *
 * @param {unknown} controller
 */
function combatControllerLabel(controller) {
  if (isSystemController(controller)) return String(controller).slice(7);
  if (controller === "reactive_tdm") {
    return "Reactive TDM ALPHA";
  }
  if (controller === "random_valid") {
    return "Random";
  }
  if (controller === "scenario_5") {
    return "Reactive TDM BETA";
  }
  if (controller === "tdm_gamma") {
    return "Reactive TDM GAMMA";
  }
  return "Manual";
}

/**
 * Read and validate the three public combat settings from a frame-like object.
 * Require known controllers and information mode. Both teams accept the same
 * built-in controllers and host-declared Systems. Reactive controllers
 * (reactive_tdm, scenario_5 and tdm_gamma) and Systems require SharedObs. Return a frozen copy of
 * those settings, or null for malformed or unsupported input.
 *
 * @param {unknown} frame
 */
function combatConfigurationFromFrame(frame) {
  if (!isRecord(frame)) {
    return null;
  }
  const candidate = frame.combat_configuration;
  if (
    !isRecord(candidate) ||
    !isTeamController(candidate.team_a_controller) ||
    !isTeamController(candidate.team_b_controller) ||
    (candidate.execution_information_mode !== "shared_obs" &&
      candidate.execution_information_mode !== "no_shared_obs") ||
    ((isReactiveController(candidate.team_a_controller) ||
      isReactiveController(candidate.team_b_controller)) &&
      candidate.execution_information_mode !== "shared_obs")
  ) {
    return null;
  }
  return Object.freeze({
    team_a_controller: candidate.team_a_controller,
    team_b_controller: candidate.team_b_controller,
    execution_information_mode: candidate.execution_information_mode,
  });
}

/**
 * Publish confirmed combat settings to BattleClient through a document event.
 * Only the Combat Debugger sends this event, and only for a valid configuration.
 * No event is sent for replay or missing settings.
 *
 * @param {unknown} frame
 */
function publishInstalledCombatConfiguration(frame) {
  if (productIdentity?.product_kind !== "combat_debugger") {
    return;
  }
  const configuration = combatConfigurationFromFrame(frame);
  if (configuration === null) {
    return;
  }
  document.dispatchEvent(
    new CustomEvent("marl-combat-configuration-installed", {
      detail: {
        ...configuration,
        task_mode: matchSummaryView(state.presentation)?.taskMode ?? null,
        system_choices: isRecord(frame) ? (frame.system_choices ?? []) : [],
        manual_slots: isRecord(frame)
          ? (frame.combat_configuration?.manual_slots ?? [])
          : [],
        slot_labels: isRecord(frame)
          ? (frame.combat_configuration?.slot_labels ?? [])
          : [],
        scenario_name: isRecord(frame) ? (frame.scenario?.name ?? null) : null,
        revision: isRecord(frame) ? frame.revision : null,
        session_id: isRecord(frame) ? frame.session_id : null,
        run_generation: isRecord(frame) ? frame.run_generation : null,
      },
    }),
  );
}

/**
 * Build a frozen combat-configuration command from a requested settings object.
 * Return null when either requested or installed settings are invalid, or all
 * three settings already match. This prepares a command without sending it.
 *
 * @param {unknown} value
 */
function requestedCombatConfigurationCommand(value) {
  if (!isRecord(value)) {
    return null;
  }
  const requested = combatConfigurationFromFrame({ combat_configuration: value });
  const installed = combatConfigurationFromFrame(state.frame);
  if (
    requested === null ||
    installed === null ||
    (requested.team_a_controller === installed.team_a_controller &&
      requested.team_b_controller === installed.team_b_controller &&
      requested.execution_information_mode === installed.execution_information_mode)
  ) {
    return null;
  }
  return Object.freeze({
    command_type: "set_combat_configuration",
    team_a_controller: requested.team_a_controller,
    team_b_controller: requested.team_b_controller,
    execution_information_mode: requested.execution_information_mode,
  });
}

/**
 * Apply a local display-switch action and return whether anything changed.
 *
 * action is one reduceVisualFilterState action: {type: "set", filterId,
 * enabled}, {type: "enable_all"}, {type: "disable_all"}, {type: "apply_preset",
 * presetId} or {type: "restore_defaults"} (Reset To Default Configuration).
 * apply_preset makes presetId the preset in use and restore_defaults makes it
 * "default"; the other actions keep the preset in use, so their changes count
 * as hand changes. When the switch state changes: invalidate downloads, replace
 * the whole snapshot, set the page-wide icon styles from Lore Emblems and Status
 * Icons From Marks (setIconStyleDefaults), pause replay and repaint the page. When only the preset
 * in use changes (the switches already matched it): repaint the filter panel
 * only. Neither touches the View, Ranges, layout or motion.
 *
 * @param {unknown} action
 * @returns {boolean} Whether the switch state or the preset in use changed.
 * @throws {TypeError | RangeError} From reduceVisualFilterState for a malformed
 *   action, an unknown switch or an unknown preset; nothing changes then.
 */
function applyVisualFilterAction(action) {
  const next = reduceVisualFilterState(visualFilterState, action);
  const record = isRecord(action) ? action : {};
  const nextPresetId =
    record.type === "apply_preset"
      ? String(record.presetId)
      : record.type === "restore_defaults"
        ? "default"
        : visualFilterPresetId;
  const presetChanged = nextPresetId !== visualFilterPresetId;
  visualFilterPresetId = nextPresetId;
  if (next === visualFilterState) {
    if (presetChanged) {
      renderVisualFilterControls(visualFilterState, visualFilterPresetId);
    }
    return presetChanged;
  }
  invalidateReplayArtifactAction();
  visualFilterState = next;
  tooltipController.setSymbolTipsEnabled(next.symbol_explanations !== false);
  // Lore Emblems and Status Icons From Marks pick the icon style everywhere
  // (bodies, deck, scorebug, rail, cards); the repaint below redraws them.
  setIconStyleDefaults({
    classEmblems: next.lore_emblems === false ? "outline" : "sigil",
    statusIcons: next.status_marks_icons === false ? "outline" : "sigil",
  });
  if (isReplayMode()) {
    replayPlayback.pause("visual_filter_changed");
  }
  render();
  return true;
}

/**
 * Set the camera on screen. This is the one camera entry point; the camera
 * look (Whole Map, Everyone In View, Battle Cam) calls it.
 *
 * frame is null for Whole Map, or {camera, minX, minY, maxX, maxY}: camera is
 * "whole_map", "everyone_in_view" or "battle_cam", and the rectangle is the
 * world area to fit on screen, in world units with y counting up (it may reach
 * past the map edge for a margin). "whole_map" is stored as null whatever its
 * rectangle. The frame is fixed until the next call: the caller computes a new
 * one per tick. The caller must build it only from the current audience's own
 * scene (in Agent POV, only bodies the team sees).
 *
 * When the frame changes: invalidate a pending PNG export (the picture moved)
 * and repaint the page, so the renderer, the choreography surface and the next
 * export all use the new frame. There is no ease here; the camera look owns any
 * temporary ease (choreographer.easeCamera). No request is sent.
 *
 * @param {unknown} frame The new camera frame, or null for Whole Map.
 * @returns {boolean} Whether the camera frame changed.
 * @throws {TypeError} For a frame that is not null or an exact record with a
 *   known camera and finite numbers.
 * @throws {RangeError} For a rectangle without positive width and height.
 */
function setCameraFrame(frame) {
  /** @type {PageCameraFrame | null} */
  let next = null;
  if (frame !== null) {
    if (!isRecord(frame)) {
      throw new TypeError("A camera frame must be null or a record.");
    }
    const keys = Object.keys(frame).sort().join(",");
    if (keys !== "camera,maxX,maxY,minX,minY") {
      throw new TypeError("A camera frame has exactly camera, minX, minY, maxX, maxY.");
    }
    const { camera, minX, minY, maxX, maxY } = frame;
    if (
      camera !== "whole_map" &&
      camera !== "everyone_in_view" &&
      camera !== "battle_cam"
    ) {
      throw new TypeError(`Unknown camera ${String(camera)}.`);
    }
    for (const value of [minX, minY, maxX, maxY]) {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new TypeError("Camera frame bounds must be finite numbers.");
      }
    }
    if (!(maxX > minX) || !(maxY > minY)) {
      throw new RangeError("A camera frame needs a positive width and height.");
    }
    next =
      camera === "whole_map" ? null : Object.freeze({ camera, minX, minY, maxX, maxY });
  }
  const same =
    next === cameraFrame ||
    (next !== null &&
      cameraFrame !== null &&
      next.camera === cameraFrame.camera &&
      next.minX === cameraFrame.minX &&
      next.minY === cameraFrame.minY &&
      next.maxX === cameraFrame.maxX &&
      next.maxY === cameraFrame.maxY);
  if (same) {
    return false;
  }
  invalidateReplayArtifactAction();
  cameraFrame = next;
  render();
  return true;
}

/**
 * True while render() runs. A camera change asked for during a render (the
 * camera switch's update) waits for that render to finish, so the page never
 * paints inside its own paint.
 */
let renderInProgress = false;

/**
 * Apply a camera frame chosen by the camera switch, with or without an ease.
 *
 * frame is null for Whole Map or a camera frame for setCameraFrame; ease true
 * eases the picture over 520 ms (choreographer.easeCamera) after the repaint,
 * false cuts (reduced motion, a view or recipient change, paused). During a
 * render the change is queued to a microtask. Side effects: setCameraFrame (a
 * repaint) and one camera ease animation. Errors from a bad frame propagate.
 * Agent POV: the switch builds the frame only from the audience's own scene.
 *
 * @param {unknown} frame
 * @param {boolean} ease
 */
function applyCameraFrameWithEase(frame, ease) {
  if (renderInProgress) {
    queueMicrotask(() => applyCameraFrameWithEase(frame, ease));
    return;
  }
  const before = battlefieldRenderer.transform;
  if (!setCameraFrame(frame)) return;
  choreographer.easeCamera(
    battlefieldRenderer.cameraGroup,
    before,
    battlefieldRenderer.transform,
    { instant: !ease },
  );
}

/** The audience and recipient of the last camera update ("" before the first). */
let lastCameraAudienceKey = "";

/**
 * Tell the camera switch about the tick just presented, and lean for Kill Cam.
 *
 * presentationFrame is the installed presentation; renderPolicy is the request
 * the choreographer got. The switch reads the choreography plan's camera and
 * beats (built from this audience's own events), the audience's arena scene
 * and the world size; the reason is "view" or "recipient" when the audience
 * changed, "step" for a one-tick replay step, "play" for a played tick and
 * "seek" otherwise. A Kill Cam record on a played tick starts the lean through
 * choreographer.leanCamera. Never reads researcher_space or the match summary.
 * Side effects: camera switch DOM, possibly a queued camera change and one lean
 * animation.
 *
 * @param {unknown} presentationFrame
 * @param {string} renderPolicy
 */
function updateCameraForTick(presentationFrame, renderPolicy) {
  const transform = battlefieldRenderer.transform;
  if (
    cameraSwitch === null ||
    transform === null ||
    !isAuthorizedPresentationFrame(presentationFrame)
  ) {
    return;
  }
  const audience = String(authorizedPresentationAudience(presentationFrame));
  const audienceKey = `${audience}:${String(
    presentationFrame.authority?.recipient_public_agent_id ?? "",
  )}`;
  const reason =
    lastCameraAudienceKey !== "" && audienceKey !== lastCameraAudienceKey
      ? lastCameraAudienceKey.startsWith(`${audience}:`)
        ? "recipient"
        : "view"
      : renderPolicy === "replay_step"
        ? "step"
        : renderPolicy === "replay_animated" || renderPolicy === "live_once"
          ? "play"
          : "seek";
  lastCameraAudienceKey = audienceKey;
  const plan = choreographer.plan;
  const playback = replayPlayback.snapshot();
  const records = cameraSwitch.update({
    camera: plan?.camera ?? null,
    beats: plan?.beats ?? null,
    scene: authorizedPresentationSceneView(presentationFrame),
    worldWidth: transform.worldWidth,
    worldHeight: transform.worldHeight,
    transform,
    reason,
    playing: isReplayMode() ? playback.playing : renderPolicy === "live_once",
    chosenRate: playback.playbackRate,
  });
  const kill = records.find((record) => record.kind === "kill-cam");
  const tickMs = Number(plan?.beats?.tickMs);
  if (kill === undefined || reason !== "play" || !(tickMs > 0)) return;
  /**
   * Build the viewport transform of one camera frame at today's viewport size.
   *
   * @param {unknown} frame
   */
  const frameTransform = (frame) =>
    createViewportTransform({
      worldWidth: transform.worldWidth,
      worldHeight: transform.worldHeight,
      viewportWidth: transform.viewportBounds.width,
      viewportHeight: transform.viewportBounds.height,
      padding: 24,
      cameraFrame: /** @type {any} */ (frame),
    });
  choreographer.leanCamera(
    battlefieldRenderer.cameraGroup,
    killCamKeyframes(
      /** @type {any} */ (kill),
      transform,
      frameTransform(kill.frame),
      frameTransform(kill.creepFrame),
      tickMs,
    ),
  );
}

/**
 * Update the Rune Scrubber from one playback snapshot (Replay Viewer only).
 *
 * playback is the ReplayPlaybackController snapshot. The audience is Oracle View
 * or Agent POV with the current recipient; the replay key is the installed
 * replay's canonical digest. Does nothing in the BattleClient or DevClient.
 * Side effects: Rune Scrubber DOM and, on an audience or replay change, one
 * timeline-marks GET (made by the Rune Scrubber).
 *
 * @param {ReturnType<ReplayPlaybackController["snapshot"]>} playback
 */
function updateRuneScrubber(playback) {
  if (runeScrubber === null) return;
  const installed = installedPresentationAuthority();
  const presentation = installed?.presentation ?? null;
  const transport = installed?.transport ?? null;
  const summary =
    transport?.artifact_facts?.artifact_summary ?? transport?.artifact_summary;
  runeScrubber.update({
    playback,
    audience:
      authorizedPresentationAudience(presentation) === "agent_pov"
        ? {
            kind: "agent_pov",
            recipientPublicAgentId:
              presentation?.authority?.recipient_public_agent_id ?? null,
          }
        : { kind: "oracle" },
    filters: visualFilterState,
    replayKey: summary?.replay_reference?.canonical_digest_sha256 ?? null,
    visible: true,
  });
}

/**
 * Select a champion from its HUD frame, as a click on its Roster row does.
 *
 * Replay Viewer (any frame, seen or unseen): send exactly the command that
 * champion's Roster row sends (panels.pickAgent), so in Agent POV a click
 * moves the view to that champion, enemy or teammate, and in Oracle View it
 * selects it; nothing happens while the roster is disabled (busy).
 *
 * Live play: publicAgentId names a champion drawn in the shown audience's
 * arena scene, or your own Manual champion that this view cannot see, for
 * example yours on the other team in One On Each Team. A drawn champion:
 * find its presentation key in that scene and route activate_authorized_agent
 * through dispatchPanelCommand. Your unseen Manual champion: send
 * roster_selection (role "control") for its command slot, as Back To Your
 * Champion does; the host then shows its view. Any other champion not in the
 * scene does nothing. Side effects: the same as a Roster click.
 *
 * @param {string} publicAgentId
 */
function selectAgentFromHud(publicAgentId) {
  if (isReplayMode()) {
    panels.pickAgent(publicAgentId);
    return;
  }
  const scene = installedAuthorizedPresentationSceneView(state.presentation);
  const agent = asArray(scene?.agents).find(
    (candidate) => isRecord(candidate) && candidate.public_agent_id === publicAgentId,
  );
  if (!isReplayMode() && !isRecord(agent)) {
    const row = authorizedPresentationIdentityRows(state.presentation).find(
      (identity) => identity.public_agent_id === publicAgentId,
    );
    const slot = row?.command_global_slot;
    const manual = asArray(state.frame?.combat_configuration?.manual_slots);
    if (Number.isInteger(slot) && manual.includes(slot)) {
      void dispatchCommand(
        { command_type: "roster_selection", role: "control", global_slot: slot },
        { source: "control" },
      );
    }
    return;
  }
  if (!isRecord(agent) || typeof agent.presentation_key !== "string") return;
  void dispatchPanelCommand({
    command_type: "activate_authorized_agent",
    presentation_key: agent.presentation_key,
  });
}

/**
 * Draw the Replay Viewer's Champion Frame for the selected champion.
 *
 * presentation is the installed presentation. Only in the Replay Viewer and only
 * while the selected_agent_frame switch is on; otherwise the slot is hidden. The
 * frame reads the researcher scene for docked facts (a docked researcher panel,
 * labelled Full Record in Agent POV) and the incoming rows of the shown
 * audience. It is drawn after the choreographer settles, so a life or fall never
 * shows before its beat. With no authorized presentation (authority cleared or
 * pending) the slot is hidden and emptied at once, so no champion of an old
 * scene stays in the page. Side effects: DOM inside #champion-frame.
 *
 * @param {unknown} presentation
 */
function renderChampionFrameForPage(presentation) {
  const slot = elements.championFrame;
  if (
    !isReplayMode() ||
    visualFilterState.selected_agent_frame === false ||
    !isAuthorizedPresentationFrame(presentation)
  ) {
    slot.hidden = true;
    if (!isAuthorizedPresentationFrame(presentation)) slot.replaceChildren();
    renderReplayClassGuide(presentation, [], null);
    return;
  }
  const pov = authorizedPresentationAudience(presentation) === "agent_pov";
  const upcoming = authorizedPresentationHasResearcherSpace(presentation)
    ? presentation.researcher_space?.upcoming_transition
    : presentation.upcoming_transition;
  const selectedKey =
    installedAuthorizedPresentationSceneView(presentation)?.selection
      ?.inspection_owner_presentation_key;
  const researcherAgents = asArray(
    authorizedPresentationResearcherSceneView(presentation)?.agents,
  );
  const selected = researcherAgents.find(
    (agent) => isRecord(agent) && agent.presentation_key === selectedKey,
  );
  renderChampionFrame(
    slot,
    championFrameView({
      selectedPublicAgentId: isRecord(selected)
        ? String(selected.public_agent_id)
        : (activatedAgentPublicId ?? null),
      agents: researcherAgents,
      chip: fullRecordLabel(presentation),
      audience: pov ? "agent_pov" : "researcher",
      incomingRows: authorizedPresentationIncomingRows(presentation),
      hasIncomingTick: isRecord(
        pov ? presentation.visual_events : presentation.latest_events,
      ),
      povSceneAgentIds: pov
        ? asArray(authorizedPresentationSceneView(presentation)?.agents).map(
            (agent) => agent.public_agent_id,
          )
        : null,
      replay: presentation.viewer_mode === "replay",
      upcomingActionRows:
        isRecord(upcoming) && Array.isArray(upcoming.action_rows)
          ? upcoming.action_rows
          : null,
      describeAction: describeRecordedAction,
    }),
  );
  slot.hidden = false;
  renderReplayClassGuide(presentation, researcherAgents, selected);
}

/**
 * Fill the Replay Viewer's Class Guide (#replay-class-card, under the
 * Champion Frame in the Agent tab) for the selected champion: its class's
 * guide with the numbers from the recorded class_mechanics and its Ultimate
 * readiness ("Ultimate · Ready In 3 Ticks"), in the BattleClient's words
 * (classGuideCardHtml). With nobody selected, or the Class Guide switch
 * (class_guide) off, the card hides. It also fills or hides the Full Class
 * Details under it (renderReplayClassDetails). The Agent tab is a docked researcher
 * panel, so in Agent POV it reads the full record like the Champion Frame;
 * nothing here draws over the arena.
 *
 * @param {unknown} presentation The installed presentation.
 * @param {ReadonlyArray<Record<string, any>>} researcherAgents The full-record
 *   scene agents.
 * @param {unknown} selected The selected agent record, if any.
 * @returns {void} Side effects: the card's DOM and hidden flag, and the Full
 *   Class Details section's.
 */
function renderReplayClassGuide(presentation, researcherAgents, selected) {
  const card = replayClassGuide;
  if (card === null) return;
  const chosenId = isRecord(selected)
    ? String(selected.public_agent_id)
    : (activatedAgentPublicId ?? null);
  const agent = researcherAgents.find(
    (row) => isRecord(row) && String(row.public_agent_id) === chosenId,
  );
  if (!isRecord(agent) || !isVisualFilterEnabled(visualFilterState, "class_guide")) {
    card.hidden = true;
    card.replaceChildren();
    renderReplayClassDetails(presentation, null);
    return;
  }
  const mechanics =
    asArray(authorizedPresentationResearcherSceneView(presentation)?.class_mechanics)
      .filter(isRecord)
      .find((row) => row.class_id === agent.class_id) ?? null;
  const cooldown = Number.isInteger(agent.ultimate_cooldown)
    ? Math.max(0, Number(agent.ultimate_cooldown))
    : null;
  // classGuideCardHtml escapes every recorded word it writes.
  card.innerHTML = classGuideCardHtml(document, agent.class_id, mechanics, {
    cooldown,
  });
  card.hidden = false;
  renderReplayClassDetails(presentation, agent);
}

/**
 * Fill or hide the Replay Viewer's Full Class Details (#replay-class-details)
 * for the champion the Class Guide shows.
 *
 * agent is that champion's full-record scene agent, or null to hide the
 * section (nobody selected, or the Class Guide switch off). The facts are the
 * class's, read through the same authorized inspector view as the old Agent
 * card (authorizedInspectorView with this champion's public ID), so the
 * section holds exactly what that card held: "Order" from classLoreV1, then
 * renderSemanticInspector of the inspector's owner_descriptor. A champion
 * whose class mechanics are missing hides the section. The body is rebuilt
 * only when the champion or its class changes, so an open section stays open
 * and keeps its scroll while the replay plays. It is a docked researcher panel
 * (Full Record in Agent POV, like the rest of the Agent tab); nothing here
 * draws over the arena.
 *
 * @param {unknown} presentation The installed presentation.
 * @param {Record<string, any> | null} agent The selected full-record agent.
 * @returns {void} Side effects: the section's DOM, hidden flag and
 *   replayClassDetailsKey.
 */
function renderReplayClassDetails(presentation, agent) {
  if (replayClassDetails === null) return;
  const { details, body } = replayClassDetails;
  const descriptor = isRecord(agent)
    ? (authorizedInspectorView(
        presentation,
        installedLocalInspectedPresentationKey(presentation),
        String(agent.public_agent_id),
      )?.owner_descriptor ?? null)
    : null;
  if (!isRecord(agent) || descriptor === null) {
    details.hidden = true;
    body.replaceChildren();
    replayClassDetailsKey = null;
    return;
  }
  const key = `${agent.public_agent_id}:${agent.class_id}`;
  if (key !== replayClassDetailsKey) {
    const lore = classLoreV1(agent.class_id);
    const parts = [];
    if (lore !== null) {
      const order = document.createElement("p");
      order.className = "replay-class-details__order";
      const label = document.createElement("span");
      label.textContent = "Order";
      const value = document.createElement("strong");
      value.textContent = lore.order;
      order.append(label, value);
      parts.push(order);
    }
    const classPart = document.createElement("div");
    classPart.className = "replay-class-details__class";
    renderSemanticInspector(classPart, descriptor);
    parts.push(classPart);
    body.replaceChildren(...parts);
    replayClassDetailsKey = key;
  }
  details.hidden = false;
}

/**
 * Return the choreography surface for this page: the renderer's surface plus
 * the HUD and panel roots whose beat-held nodes the choreographer also times
 * (the old score strip, the scorebug #scorebug in every client, so its score
 * and portraits change at the Fall and Return beats and not when the frame
 * installs, the party frames or the Replay Viewer's Arena Frames, the
 * Roster, the selection card, the Fallen panel and the Replay Viewer's
 * Actions tab), and the
 * beforeBeatHolds hook that redraws the Fallen panel from each plan before the
 * holds are collected. Null before a scene exists. Pure apart from returning
 * that hook.
 */
function pageChoreographySurface() {
  const surface = battlefieldRenderer.choreographySurface();
  if (surface === null) return null;
  return Object.freeze({
    ...surface,
    beatHoldRoots: Object.freeze([
      elements.matchScoreboard,
      ...[document.getElementById("scorebug")].filter((root) => root !== null),
      isReplayMode() ? elements.teamRosters : elements.partyFrames,
      elements.roster,
      elements.selectionCard,
      elements.fallenFeed,
      ...(isReplayMode() ? [elements.replayRailActions] : []),
    ]),
    beforeBeatHolds: renderFallenFeed,
  });
}

/**
 * Redraw the Fallen panel (#fallen-feed, the top of the side rail) for one
 * choreography plan.
 *
 * plan is the plan about to be shown (the choreographer passes it through the
 * surface's beforeBeatHolds hook), or null to clear the panel. This tick's
 * falls come from the plan. The earlier ticks' falls (the last 8 ticks) come
 * from the installed frame's recent_falls host fact when that frame is the
 * plan's tick (the server owns the history: the Replay Viewer service counts
 * it from the recording, the live service keeps it on its session; so a seek,
 * a view switch or a reload shows the same rows as playing there), with
 * identities from the frame's identity rows; otherwise (older producers)
 * requiems.fallenFeedHistory carries them from tick to tick (the same
 * audience; a seek, view or recipient change starts again). The panel shows only while
 * the Fallen Feed switch (death_announcer) is on and the plan has a tick.
 * Shown IDs come from authorizedPresentationAgentDisplayId on the installed
 * presentation.
 *
 * Side effects: updates fallenFeedMemory, replaces the slot's children with
 * requiems.buildFallenFeed (which registers row tooltips) and sets its hidden
 * flag. This tick's rows carry the Fall beat hold, so the choreographer fades
 * them in at that champion's Fall beat. No timers or storage.
 *
 * Agent POV: the plan's fallen_feed_row rows and the frame's recent_falls hold
 * only the falls the team saw and only the attackers whose blow it saw land;
 * identity rows add only names, classes and teams, so the panel never names
 * an unseen champion. Oracle -> POV -> Oracle gives a fresh panel.
 *
 * @param {Record<string, any> | null} plan
 * @returns {void}
 */
function renderFallenFeed(plan) {
  const slot = elements.fallenFeed;
  const installed = installedPresentationAuthority()?.presentation ?? null;
  const recentFalls =
    installed !== null &&
    Array.isArray(installed.recent_falls) &&
    installed.source?.source_simulator_step_count === plan?.simulatorStep
      ? installed.recent_falls
      : null;
  const identities = new Map(
    recentFalls === null
      ? []
      : authorizedPresentationIdentityRows(installed).map((row) => [
          row.public_agent_id,
          row.agent,
        ]),
  );
  const history = requiems.fallenFeedHistory(
    fallenFeedMemory,
    plan,
    recentFalls === null
      ? null
      : { rows: recentFalls, identityOf: (id) => identities.get(id) ?? null },
  );
  fallenFeedMemory = history.memory;
  if (
    history.tick === null ||
    !isVisualFilterEnabled(visualFilterState, "death_announcer")
  ) {
    slot.replaceChildren();
    slot.hidden = true;
    return;
  }
  const presentation = installedPresentationAuthority()?.presentation ?? null;
  const records = requiems.fallenFeedRecords({
    tick: history.tick,
    rows: history.rows,
    displayIdOf: (publicAgentId) =>
      authorizedPresentationAgentDisplayId(presentation, publicAgentId),
  });
  slot.replaceChildren(requiems.buildFallenFeed(document, records));
  slot.hidden = false;
}

/**
 * Return the world rectangle the renderer fits on screen, or null for Whole
 * Map. Reads the camera state only.
 *
 * @returns {Readonly<{minX: number, minY: number, maxX: number, maxY: number}> | null}
 */
function cameraRenderFrame() {
  return cameraFrame === null
    ? null
    : Object.freeze({
        minX: cameraFrame.minX,
        minY: cameraFrame.minY,
        maxX: cameraFrame.maxX,
        maxY: cameraFrame.maxY,
      });
}

/**
 * Request the boolean range visibility for the current authorized audience.
 * Agent POV changes page-local state; Oracle View sends its existing server command.
 * Return whether a local change or request started. Return false while unavailable,
 * busy, disconnected, or already at the requested value.
 *
 * @param {boolean} visible
 * @returns {boolean} Whether a local change or one service request was started.
 */
function setActiveRangesVisible(visible) {
  const presentation = state.presentation;
  if (
    typeof visible !== "boolean" ||
    !isAuthorizedPresentationFrame(presentation) ||
    state.busy ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline ||
    installedPresentationRangesVisible(presentation) === visible
  ) {
    return false;
  }
  const audience = authorizedPresentationAudience(presentation);
  if (audience === "agent_pov") {
    if (!setAgentLocalRangesVisible(visible)) {
      return false;
    }
    render();
    return true;
  }
  if (audience !== "researcher") {
    return false;
  }
  if (isReplayMode()) {
    void dispatchReplayCommand({
      command_type: "set_ranges",
      show_ranges: visible,
    });
  } else {
    void dispatchCommand(keyboardCommand("g"));
  }
  return true;
}

/**
 * Choose animation policy from the matched installed transport and playback state.
 * Live uses live_once; replay uses the current cursor intent or replay_static.
 * Return a frozen policy carrying the supplied filters. consumeAnimatedRestart
 * defaults to false; when true, consume each requested animated restart once,
 * and turn a one-tick step into the choreographer's "replay_step" request once
 * per intent generation.
 *
 * replay_step: when the current replay intent is the paused still
 * (replay_static) and carries stepGlide "forward" or "backward" (the playback
 * controller sets it only when a one-tick move landed exactly one frame away),
 * the first consuming call for that intent generation returns "replay_step"
 * with stepDirection set to it, so the choreographer plays a +1's tick or
 * rewinds a -1's. Every later call returns replay_static (the settled still). The renderer and
 * the PNG export still see replay_static: render() maps replay_step to
 * replay_static for the battlefield, and the export reads the controller's
 * intent, never this request.
 *
 * @param {unknown} presentation
 * @param {typeof visualFilterState} visualFilters
 * @param {{consumeAnimatedRestart?: boolean}} [options]
 * @returns {Readonly<{
 *   renderPolicy: "live_once" | "replay_animated" | "replay_static" | "replay_step",
 *   visualFilters: typeof visualFilterState,
 *   restartAnimated?: true,
 *   stepDirection?: "forward" | "backward",
 * }>}
 */
function installedChoreographyControl(
  presentation,
  visualFilters,
  { consumeAnimatedRestart = false } = {},
) {
  const authority = state.authority;
  const coherentInstalledPair =
    isAuthorizedPresentationFrame(presentation) &&
    isJoinedTransportAndAuthorizedPresentationV1(authority) &&
    authority.presentation === presentation &&
    authority.transport === state.frame;
  const replayInstalled =
    coherentInstalledPair &&
    isReplayMode() &&
    authority.transport.viewer_mode === "replay";
  if (!replayInstalled) {
    return Object.freeze({
      renderPolicy: "live_once",
      visualFilters,
    });
  }

  const playback = replayPlayback.snapshot();
  const intent = playback.presentationIntent;
  const currentIntent =
    isRecord(intent) &&
    replayCursorsMatch(playback.cursor, authority.transport.cursor) &&
    playback.transportState !== "OFFLINE";
  const renderPolicy = currentIntent ? intent.renderPolicy : "replay_static";
  const restartAnimated =
    consumeAnimatedRestart &&
    renderPolicy === "replay_animated" &&
    intent?.restartAnimated === true &&
    intent.generation > consumedReplayRestartGeneration;
  if (restartAnimated) {
    consumedReplayRestartGeneration = intent.generation;
  }
  const stepGlide =
    consumeAnimatedRestart &&
    currentIntent &&
    renderPolicy === "replay_static" &&
    (intent.stepGlide === "forward" || intent.stepGlide === "backward") &&
    intent.generation > consumedReplayStepGeneration;
  if (stepGlide) {
    consumedReplayStepGeneration = intent.generation;
  }
  return Object.freeze({
    renderPolicy: stepGlide ? "replay_step" : renderPolicy,
    visualFilters,
    ...(restartAnimated ? { restartAnimated: true } : {}),
    ...(stepGlide ? { stepDirection: intent.stepGlide } : {}),
  });
}

/**
 * Compare the five cursor version, index, and generation fields of two records.
 * Return false for non-records. This equality helper assumes separate protocol
 * validation; matching missing fields alone does not certify a cursor.
 *
 * @param {unknown} left @param {unknown} right
 */
function replayCursorsMatch(left, right) {
  if (!isRecord(left) || !isRecord(right)) {
    return false;
  }
  return (
    left.schema_version === right.schema_version &&
    left.frame_index === right.frame_index &&
    left.final_frame_index === right.final_frame_index &&
    left.cursor_generation === right.cursor_generation &&
    left.choreography_generation === right.choreography_generation
  );
}

/**
 * Resolve the installed transport and presentation into one coherent authority.
 * Return the trusted pair or null when its objects or join marker do not agree.
 */
function installedPresentationAuthority() {
  return resolveInstalledPresentationAuthorityV1(
    state.authority,
    state.frame,
    state.presentation,
  );
}

/**
 * Return whether the currently installed transport and presentation form a
 * trusted matching pair. This is the common gate for scientific UI actions.
 */
function installedAuthorityIsCoherent() {
  return installedPresentationAuthority() !== null;
}

/**
 * Find an opaque agent key in the supplied trusted presentation scene.
 * Return the matching authorized agent or null for invalid input or no match.
 * Do not infer identity from DOM order, raw slots, or an earlier presentation.
 *
 * @param {unknown} presentation
 * @param {unknown} presentationKey
 * @returns {Readonly<Record<string, any>> | null}
 */
function authorizedAgentForPresentationKey(presentation, presentationKey) {
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    typeof presentationKey !== "string"
  ) {
    return null;
  }
  return (
    asArray(authorizedPresentationSceneView(presentation)?.agents).find(
      (candidate) =>
        isRecord(candidate) && candidate.presentation_key === presentationKey,
    ) ?? null
  );
}

/**
 * Find an opaque agent key in a trusted presentation researcher directory.
 * Return its agent or null. This lookup serves non-battlefield roster controls;
 * SVG painting and hit testing must use the audience-specific scene instead.
 *
 * @param {unknown} presentation
 * @param {unknown} presentationKey
 * @returns {Readonly<Record<string, any>> | null}
 */
function authorizedResearcherAgentForPresentationKey(presentation, presentationKey) {
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    typeof presentationKey !== "string"
  ) {
    return null;
  }
  return (
    asArray(authorizedPresentationResearcherSceneView(presentation)?.agents).find(
      (candidate) =>
        isRecord(candidate) && candidate.presentation_key === presentationKey,
    ) ?? null
  );
}

/**
 * Return a frozen local selection/range view for matching installed authority.
 * Return null when the presentation is untrusted or does not own local inspection.
 * The returned choices affect display only and cannot add scientific facts.
 *
 * @param {unknown} presentation
 * @returns {Readonly<{
 *   inspectedPresentationKey: string | null,
 *   rangesVisible: boolean,
 * }> | null}
 */
function installedLocalPresentationPreference(presentation) {
  if (!isAuthorizedPresentationFrame(presentation)) {
    return null;
  }
  const preference = installedActivePresentationPreference(presentation);
  return preference?.localInspection.owned
    ? Object.freeze({
        inspectedPresentationKey: preference.localInspection.presentationKey,
        rangesVisible: agentLocalRangesVisible,
      })
    : null;
}

/**
 * Return the active local selection key, explicit null, or undefined.
 * Null means locally cleared selection; undefined leaves the server-confirmed
 * Oracle selection unchanged.
 *
 * @param {unknown} presentation
 * @returns {string | null | undefined}
 */
function installedLocalInspectedPresentationKey(presentation) {
  return installedLocalPresentationPreference(presentation)?.inspectedPresentationKey;
}

/**
 * Build the supplied presentation scene with its current allowed local selection.
 * Pass undefined when this authority does not own local inspection so the accepted
 * Oracle selection remains in force.
 *
 * @param {unknown} presentation
 */
function installedAuthorizedPresentationSceneView(presentation) {
  return authorizedPresentationSceneView(
    presentation,
    installedLocalInspectedPresentationKey(presentation),
  );
}

/**
 * Set the active local selection to an authorized opaque key or explicit null.
 * Return false without mutation when local inspection is unavailable or the key
 * is absent. Otherwise invalidate downloads, store the selection, and return true;
 * callers decide when to repaint.
 *
 * @param {string | null} presentationKey
 * @returns {boolean}
 */
function setLocalInspectedPresentationKey(presentationKey) {
  const presentation = state.presentation;
  const preference = installedActivePresentationPreference(presentation);
  if (
    preference === null ||
    !preference.localInspection.owned ||
    (presentationKey !== null &&
      authorizedAgentForPresentationKey(presentation, presentationKey) === null)
  ) {
    return false;
  }
  invalidateReplayArtifactAction();
  preference.localInspection.presentationKey = presentationKey;
  return true;
}

/**
 * Set local range visibility for an authority that owns local inspection.
 * Require a boolean value different from the current choice. Return false for
 * an unavailable or unchanged request; otherwise invalidate downloads and return
 * true after updating page state. This helper does not repaint.
 *
 * @param {boolean} visible @returns {boolean}
 */
function setAgentLocalRangesVisible(visible) {
  const preference = installedActivePresentationPreference(state.presentation);
  if (
    preference === null ||
    !preference.localInspection.owned ||
    typeof visible !== "boolean" ||
    agentLocalRangesVisible === visible
  ) {
    return false;
  }
  invalidateReplayArtifactAction();
  agentLocalRangesVisible = visible;
  return true;
}

/**
 * Toggle the current local range choice through its normal availability gate.
 * Return whether that helper changed the value; this does not send a server command.
 *
 * @returns {boolean}
 */
function toggleAgentLocalRanges() {
  return setAgentLocalRangesVisible(!agentLocalRangesVisible);
}

/**
 * Return range visibility for this exact installed authorized presentation.
 * Oracle uses the server-confirmed choice; other audiences use allowed local
 * preferences. Invalid or different presentation objects return false.
 *
 * @param {unknown} presentation
 */
function installedPresentationRangesVisible(presentation) {
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    state.presentation !== presentation
  ) {
    return false;
  }
  const localPreference = installedLocalPresentationPreference(presentation);
  return authorizedPresentationAudience(presentation) === "researcher"
    ? confirmedResearcherRangesVisible
    : localPreference?.rangesVisible === true;
}

/**
 * Compute replay download availability from a complete UI state snapshot.
 * Require settled static replay, matching authority/cursor, a connected visible
 * page, and no pending work. PNG additionally requires a ready battlefield; metric
 * downloads remain researcher tools even when the visible scene is Agent POV.
 *
 * @param {{
 *   replayProduct: boolean,
 *   coherentAuthority: boolean,
 *   audience: string | null,
 *   transportState: string,
 *   connected: boolean,
 *   hidden: boolean,
 *   playing: boolean,
 *   requestPending: boolean,
 *   presentationPending: boolean,
 *   renderPolicy: string | null,
 *   cursorMatches: boolean,
 *   operationallyBlocked: boolean,
 *   actionPending: boolean,
 *   battlefieldReady: boolean,
 * }} input
 */
function replayArtifactActionCapabilities(input) {
  const settled =
    input.replayProduct &&
    input.coherentAuthority &&
    (input.audience === "researcher" || input.audience === "agent_pov") &&
    input.transportState === "SETTLED" &&
    input.connected &&
    !input.hidden &&
    !input.playing &&
    !input.requestPending &&
    !input.presentationPending &&
    input.renderPolicy === "replay_static" &&
    input.cursorMatches &&
    !input.operationallyBlocked &&
    !input.actionPending;
  return Object.freeze({
    exportPng: settled && input.battlefieldReady,
    downloadMetrics: settled,
  });
}

/**
 * Return whether the replay SVG has a settled static render and positive integer
 * CSS-pixel dimensions. This is a download readiness check, not a schema validator.
 */
function replayBattlefieldReady() {
  return (
    elements.battlefield.id === "battlefield" &&
    elements.battlefield.dataset.renderPolicy === "replay_static" &&
    elements.battlefieldShell.getAttribute("aria-busy") === "false" &&
    Number.isInteger(elements.battlefield.clientWidth) &&
    elements.battlefield.clientWidth > 0 &&
    Number.isInteger(elements.battlefield.clientHeight) &&
    elements.battlefield.clientHeight > 0
  );
}

/**
 * Collect playback, page, cursor, and battlefield facts for artifact controls.
 * Use the supplied installed pair or null. ignoreActionPending defaults to false;
 * an existing transaction uses true to check that its other prerequisites remain
 * valid without blocking itself.
 *
 * @param {ReturnType<typeof installedPresentationAuthority>} installed
 * @param {{ignoreActionPending?: boolean}} [options]
 */
function currentReplayArtifactActionCapabilities(
  installed,
  { ignoreActionPending = false } = {},
) {
  const playback = replayPlayback.snapshot();
  return replayArtifactActionCapabilities({
    replayProduct: isReplayMode(),
    coherentAuthority: installed !== null,
    audience:
      installed === null
        ? null
        : (authorizedPresentationAudience(installed.presentation) ?? null),
    transportState: playback.transportState,
    connected: playback.connected,
    hidden: playback.hidden,
    playing: playback.playing,
    requestPending: playback.requestPending,
    presentationPending: playback.presentationPending,
    renderPolicy: playback.presentationIntent?.renderPolicy ?? null,
    cursorMatches:
      installed !== null &&
      replayCursorsMatch(playback.cursor, installed.transport.cursor),
    operationallyBlocked:
      state.busy || state.shuttingDown || state.resyncRequired || state.offline,
    actionPending: !ignoreActionPending && replayArtifactActionTransaction !== null,
    battlefieldReady: replayBattlefieldReady(),
  });
}

/**
 * Paint PNG, metric, and episode-details download buttons from current capabilities.
 * Mark the matching transaction busy and show metric preparation only while the
 * metric download is pending with installed authority.
 *
 * @param {ReturnType<typeof installedPresentationAuthority>} installed
 */
function renderReplayArtifactActions(installed) {
  const capabilities = currentReplayArtifactActionCapabilities(installed);
  const pending = replayArtifactActionTransaction;
  elements.replayExportPngButton.disabled = !capabilities.exportPng;
  elements.replayDownloadMetricsButton.disabled = !capabilities.downloadMetrics;
  elements.replayEpisodeDetailsButton.disabled = !capabilities.downloadMetrics;
  elements.replayExportPngButton.setAttribute(
    "aria-busy",
    String(pending?.kind === "export_png"),
  );
  elements.replayDownloadMetricsButton.setAttribute(
    "aria-busy",
    String(pending?.kind === "download_metrics"),
  );
  elements.replayEpisodeDetailsButton.setAttribute(
    "aria-busy",
    String(pending?.kind === "download_episode_details"),
  );
  elements.replayMetricsPreparation.hidden =
    installed === null || pending?.kind !== "download_metrics";
  elements.replayMetricsPreparationText.textContent = METRIC_PREPARATION_MESSAGE;
}

/**
 * Start one allowed PNG, metric, or episode-details download transaction.
 * Return null if another action is pending or the current surface is unavailable.
 * Otherwise freeze the exact authority, playback generation, local selection,
 * ranges, filters and camera frame, store the transaction, and repaint its
 * controls.
 *
 * @param {"export_png" | "download_metrics" | "download_episode_details"} kind
 */
function beginReplayArtifactAction(kind) {
  if (replayArtifactActionTransaction !== null) {
    return null;
  }
  const installed = installedPresentationAuthority();
  const capabilities = currentReplayArtifactActionCapabilities(installed);
  if (
    installed === null ||
    (kind === "export_png" ? !capabilities.exportPng : !capabilities.downloadMetrics)
  ) {
    return null;
  }
  const authority = state.authority;
  if (
    !isJoinedTransportAndAuthorizedPresentationV1(authority) ||
    authority.transport !== installed.transport ||
    authority.presentation !== installed.presentation
  ) {
    return null;
  }
  const localInspectedPresentationKey =
    installedLocalInspectedPresentationKey(installed.presentation) ?? null;
  const transaction = Object.freeze({
    kind,
    authority,
    transport: installed.transport,
    presentation: installed.presentation,
    playbackGeneration: replayPlayback.snapshot().generation,
    localInspectedPresentationKey,
    showRanges: installedPresentationRangesVisible(installed.presentation),
    visualFilters: visualFilterState,
    cameraFrame,
  });
  replayArtifactActionTransaction = transaction;
  renderReplayArtifactActions(installed);
  return transaction;
}

/**
 * Return whether a captured download transaction still owns the current UI state.
 * Require the same authority objects, playback generation, and download capability.
 * PNG also requires unchanged filters, range choice, local selection and camera
 * frame.
 *
 * @param {NonNullable<typeof replayArtifactActionTransaction>} transaction
 */
function replayArtifactActionIsCurrent(transaction) {
  if (replayArtifactActionTransaction !== transaction) {
    return false;
  }
  const installed = installedPresentationAuthority();
  if (
    installed === null ||
    state.authority !== transaction.authority ||
    installed.transport !== transaction.transport ||
    installed.presentation !== transaction.presentation
  ) {
    return false;
  }
  const capabilities = currentReplayArtifactActionCapabilities(installed, {
    ignoreActionPending: true,
  });
  if (
    replayPlayback.snapshot().generation !== transaction.playbackGeneration ||
    (transaction.kind === "export_png"
      ? !capabilities.exportPng
      : !capabilities.downloadMetrics)
  ) {
    return false;
  }
  if (transaction.kind === "export_png") {
    return (
      visualFilterState === transaction.visualFilters &&
      (installedLocalInspectedPresentationKey(installed.presentation) ?? null) ===
        transaction.localInspectedPresentationKey &&
      installedPresentationRangesVisible(installed.presentation) ===
        transaction.showRanges &&
      cameraFrame === transaction.cameraFrame
    );
  }
  return true;
}

/**
 * Clear a completed download only if it is still the active transaction.
 * Refresh controls from current authority; an older completion cannot clear a
 * newer transaction.
 *
 * @param {NonNullable<typeof replayArtifactActionTransaction>} transaction
 */
function finishReplayArtifactAction(transaction) {
  if (replayArtifactActionTransaction !== transaction) {
    return;
  }
  replayArtifactActionTransaction = null;
  renderReplayArtifactActions(installedPresentationAuthority());
}

/**
 * Discard the active download identity and hide metric-preparation indicators.
 * This makes later asynchronous results stale; it does not abort their underlying
 * network request or image work.
 */
function invalidateReplayArtifactAction() {
  replayArtifactActionTransaction = null;
  elements.replayMetricsPreparation.hidden = true;
  elements.replayDownloadMetricsButton.setAttribute("aria-busy", "false");
  elements.replayEpisodeDetailsButton.setAttribute("aria-busy", "false");
}

/**
 * Trigger a browser download for a Blob and a safe basename.
 * Require 1–240 filename characters using letters, digits, dots, underscores,
 * and hyphens, beginning with a letter or digit; throw TypeError otherwise.
 * Create an object URL, click a temporary anchor, and revoke the URL in finally.
 *
 * @param {Blob} blob @param {string} filename
 */
function downloadReplayArtifact(blob, filename) {
  if (
    !(blob instanceof Blob) ||
    typeof filename !== "string" ||
    filename.length < 1 ||
    filename.length > 240 ||
    !/^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(filename)
  ) {
    throw new TypeError("Replay download artifact is invalid.");
  }
  const objectUrl = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = filename;
    anchor.rel = "noopener";
    anchor.click();
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

/**
 * Convert a download failure into a frozen user notice and severity.
 * Use specific messages for missing reports and denied authorization; preserve
 * an Error message for other failures. This does not display the notice.
 *
 * @param {unknown} error
 */
function replayMetricDownloadError(error) {
  if (error instanceof DebuggerApiError && error.status === 404) {
    return Object.freeze({
      message: "This replay has no metrics to download.",
      level: "warning",
    });
  }
  if (error instanceof DebuggerApiError && error.status === 403) {
    return Object.freeze({
      message:
        "This replay session cannot download metrics. Open the link printed in the terminal again.",
      level: "error",
    });
  }
  return Object.freeze({
    message: "Could not download the file. Try again.",
    level: "error",
  });
}

/**
 * Capture and download the current settled replay battlefield as a PNG.
 * Return without work when unavailable. First wait until the choreographer has
 * settled (a one-tick step glide or a camera ease moves computed transforms for
 * up to about half a second, and the export copies computed styles). The
 * capture records the camera on screen (the transaction's cameraFrame; Whole
 * Map when null) in provenance version 3. Check transaction identity after
 * capture and validate its artifact type; ignore stale results, display current
 * failures, and always release this transaction. No simulator command is sent.
 */
async function exportReplayBattlefieldPng() {
  const transaction = beginReplayArtifactAction("export_png");
  if (transaction === null) {
    return;
  }
  try {
    await choreographer.whenSettled();
    const playback = replayPlayback.snapshot();
    const artifact = await captureReplayBattlefieldPngV1({
      battlefield: elements.battlefield,
      installedAuthority: transaction.authority,
      /**
       * Return whether this PNG transaction still matches the installed view.
       * The exporter calls this during asynchronous work to reject a changed frame,
       * selection, filter configuration or playback generation.
       */
      isCurrent: () => replayArtifactActionIsCurrent(transaction),
      transportState: playback.transportState,
      renderPolicy: playback.presentationIntent?.renderPolicy ?? null,
      showRanges: transaction.showRanges,
      localInspectedPresentationKey: transaction.localInspectedPresentationKey,
      visualFilters: transaction.visualFilters,
      cameraFrame: transaction.cameraFrame,
    });
    if (!replayArtifactActionIsCurrent(transaction)) {
      return;
    }
    if (artifact.schemaVersion !== 1 || artifact.blob.type !== "image/png") {
      throw new TypeError("Replay PNG builder returned an invalid artifact.");
    }
    downloadReplayArtifact(artifact.blob, artifact.filename);
    const exportedFrame = playback.cursor?.frame_index ?? 0;
    const exportedTick = replayTimelineElements.tickForFrameIndex(exportedFrame);
    setNotice(
      `Exported the PNG of tick ${Number.isSafeInteger(exportedTick) ? exportedTick : exportedFrame}.`,
      "success",
    );
    renderConnection();
  } catch (error) {
    if (!replayArtifactActionIsCurrent(transaction)) {
      return;
    }
    logRawError("PNG export failed", error);
    setNotice("Could not export the PNG. Try again.", "error");
    renderConnection();
  } finally {
    finishReplayArtifactAction(transaction);
  }
}

/**
 * Download metrics CSV, or episode-details JSON when details is true.
 * details defaults to false. Capture one settled replay transaction, request its
 * current metric scope/frame when needed, and discard stale replies. Display
 * current request failures and always release the transaction.
 *
 * @param {boolean} [details]
 */
async function downloadReplayMetricReport(details = false) {
  const transaction = beginReplayArtifactAction(
    details ? "download_episode_details" : "download_metrics",
  );
  if (transaction === null) {
    return;
  }
  try {
    const context = replayMetricContext();
    if (!details && context === null) return;
    const report =
      !details && context !== null
        ? await getReplayMetrics(state.token, context.frameIndex, context.scope, "csv")
        : await getReplayEpisodeDetails(state.token);
    if (
      !replayArtifactActionIsCurrent(transaction) ||
      (!details && replayMetricContext()?.key !== context?.key)
    ) {
      return;
    }
    downloadReplayArtifact(
      new Blob([report.bytes], {
        type: details ? "application/json; charset=utf-8" : "text/csv; charset=utf-8",
      }),
      report.filename,
    );
    setNotice(
      details ? "Downloaded the episode details." : "Downloaded the metrics CSV.",
      "success",
    );
    renderConnection();
  } catch (error) {
    if (!replayArtifactActionIsCurrent(transaction)) {
      return;
    }
    const failure = replayMetricDownloadError(error);
    logRawError("Download failed", error);
    setNotice(failure.message, failure.level);
    renderConnection();
  } finally {
    finishReplayArtifactAction(transaction);
  }
}

/** @type {{key: string, summary: Record<string, any>} | null} */
let replayMetricSummary = null;
/** @type {string | null} */
let replayMetricRequest = null;
/** @type {{key: string, message: string} | null} */
let replayMetricFailure = null;
let replayMetricRenderKey = "";
let replayMetricLoadedKey = "";
let replayMetricGeneration = 0;
/** @type {Record<string, any> | null} */
let replayMetricCatalog = null;
/** @type {ReturnType<typeof buildMetricSearchIndex>} */
let replayMetricSearchIndex = buildMetricSearchIndex([]);
let replayMetricCatalogPending = false;
let replayMetricCatalogFailure = "";
let replayMetricSearchLimit = 20;
/** @type {HTMLButtonElement | null} */
let replayMetricSearchActive = null;
/** @type {string | null} */
let replayMetricFocus = null;
/** @type {number | null} */
let replayMetricHighlightTimer = null;

/**
 * Cancel the metric-row highlight timer and remove all current highlight classes.
 * This does not change metric selection or request state.
 */
function clearReplayMetricHighlight() {
  if (replayMetricHighlightTimer !== null) {
    window.clearTimeout(replayMetricHighlightTimer);
    replayMetricHighlightTimer = null;
  }
  for (const row of elements.metricRows.querySelectorAll(".metric-search-match")) {
    row.classList.remove("metric-search-match");
  }
}

/**
 * Set the active search-result button for keyboard navigation, or clear it with null.
 * Update aria-selected on old/new buttons and the search input active-descendant.
 *
 * @param {HTMLButtonElement | null} button
 */
function setActiveMetricSearchResult(button) {
  replayMetricSearchActive?.setAttribute("aria-selected", "false");
  replayMetricSearchActive = button;
  if (button) {
    button.setAttribute("aria-selected", "true");
    elements.metricSearch.setAttribute("aria-activedescendant", button.id);
  } else {
    elements.metricSearch.removeAttribute("aria-activedescendant");
  }
}

/**
 * Show or hide search suggestions, requiring a nonempty trimmed search value.
 * Update aria-expanded and clear active keyboard selection whenever the list closes.
 *
 * @param {boolean} open
 */
function setReplayMetricSearchOpen(open) {
  elements.metricSearchContent.hidden = !open || !elements.metricSearch.value.trim();
  elements.metricSearch.setAttribute(
    "aria-expanded",
    String(!elements.metricSearchContent.hidden),
  );
  if (elements.metricSearchContent.hidden) setActiveMetricSearchResult(null);
}

/**
 * Render catalog search matches up to the current result limit.
 * Clear the active result, update status and the More control, and wire each row
 * to measurement selection. Every team, agent and recipient row gets its crest
 * or class emblem from the catalog's agent list. Do nothing until a catalog is
 * available.
 */
function renderReplayMetricSearch() {
  if (replayMetricCatalog === null) return;
  setActiveMetricSearchResult(null);
  const query = elements.metricSearch.value.trim();
  const { matches, message } = searchMeasurements(replayMetricSearchIndex, query);
  elements.metricSearchResults.hidden = !query;
  elements.metricSearchMore.hidden = matches.length <= replayMetricSearchLimit;
  elements.metricSearchStatus.textContent = query
    ? (message ??
      `${Math.min(matches.length, replayMetricSearchLimit)} of ${matches.length} measurements found.`)
    : "Search includes every numerical CSV column, including those that do not apply to this roster.";
  // The catalog's agent list gives every team, agent and recipient row its
  // crest or class emblem beside the words (owner, 4 Oct 2026).
  renderMetricSearchResults(
    elements.metricSearchResults,
    matches,
    replayMetricCatalog.topics,
    replayMetricSearchLimit,
    selectReplayMeasurement,
    { agents: replayMetricCatalog.agents },
  );
}

/**
 * Handle selection of a catalog measurement row.
 * For an inapplicable row, show its definition. Otherwise select its topic/view,
 * remember the row to focus, and refresh metrics. Clear old search focus/highlights;
 * do nothing without a loaded catalog.
 *
 * @param {Record<string, any>} row
 */
function selectReplayMeasurement(row) {
  if (replayMetricCatalog === null) return;
  setActiveMetricSearchResult(null);
  clearReplayMetricHighlight();
  replayMetricFocus = null;
  if (!row.applicable) {
    renderMetricDefinition(
      elements.metricSearchDefinition,
      row,
      replayMetricCatalog.topics,
    );
    return;
  }
  elements.metricSearchDefinition.hidden = true;
  elements.metricSelection.value = row.primary_topic;
  renderMetricNavigation(
    elements.metricSelection,
    elements.metricView,
    elements.metricViewField,
    replayMetricCatalog.topics,
  );
  elements.metricView.value = row.primary_view;
  replayMetricFocus = row.name;
  renderReplayMetrics();
}

/**
 * Focus and scroll to the requested rendered metric row when it exists.
 * Highlight it for 3 seconds, then clear the highlight. Keep a missing row pending
 * so a later metric render can complete the focus request.
 */
function focusReplayMeasurement() {
  if (replayMetricFocus === null) return;
  const row = elements.metricRows.querySelector(
    `[data-metric="${CSS.escape(replayMetricFocus)}"]`,
  );
  if (row) {
    clearReplayMetricHighlight();
    row.classList.add("metric-search-match");
    replayMetricHighlightTimer = window.setTimeout(() => {
      row.classList.remove("metric-search-match");
      replayMetricHighlightTimer = null;
    }, 3000);
    row.scrollIntoView({ block: "nearest" });
    row.querySelector(".metric-measure").focus({ preventScroll: true });
    replayMetricFocus = null;
  }
}

/**
 * Return the installed replay digest, frame, scope, and request/cache keys.
 * Use final scope only when selected, otherwise cursor scope; final data keys omit
 * the changing cursor. Return null without a replay pair, string digest, or safe
 * integer frame index.
 */
function replayMetricContext() {
  const installed = installedPresentationAuthority();
  if (!isReplayMode() || installed === null) return null;
  const frame = installed.transport;
  const summary = frame.artifact_facts?.artifact_summary ?? frame.artifact_summary;
  const digest = summary?.replay_reference?.canonical_digest_sha256;
  const frameIndex = frame.cursor?.frame_index;
  const scope = elements.metricScope.value === "final" ? "final" : "cursor";
  if (typeof digest !== "string" || !Number.isSafeInteger(frameIndex)) return null;
  return {
    digest,
    frameIndex,
    scope: /** @type {"cursor" | "final"} */ (scope),
    catalogKey: `${installed.presentation.source.source_session_id}:${digest}`,
    key: `${installed.presentation.source.source_session_id}:${digest}:${scope}:${scope === "final" ? "final" : frameIndex}`,
  };
}

/**
 * Render metrics and start the catalog or summary GETs that the current scope needs.
 * Reset cached UI state when the replay changes, reuse current results, and label
 * loading or failed requests. Check generation and replay digest before accepting
 * async replies so an old artifact cannot populate the new page. The table shows
 * team and agent icons from the catalog's agent list once the catalog has
 * arrived (the catalog's arrival draws the table again).
 */
function renderReplayMetrics() {
  const context = replayMetricContext();
  const loadedKey = context?.catalogKey ?? "";
  if (loadedKey !== replayMetricLoadedKey) {
    clearReplayMetricHighlight();
    replayMetricLoadedKey = loadedKey;
    replayMetricGeneration += 1;
    replayMetricSummary = null;
    replayMetricRequest = null;
    replayMetricFailure = null;
    replayMetricRenderKey = "";
    replayMetricCatalog = null;
    replayMetricSearchIndex = buildMetricSearchIndex([]);
    replayMetricCatalogPending = false;
    replayMetricCatalogFailure = "";
    replayMetricFocus = null;
    replayMetricSearchLimit = 20;
    elements.metricSearch.value = "";
    elements.metricSearch.disabled = true;
    setReplayMetricSearchOpen(false);
    elements.metricSearchResults.replaceChildren();
    elements.metricSearchResults.hidden = true;
    elements.metricSearchMore.hidden = true;
    elements.metricSearchDefinition.hidden = true;
  }
  elements.metricProgress.hidden = true;
  elements.metricRows.setAttribute("aria-busy", "false");
  elements.metricPanel.hidden =
    context === null ||
    installedPresentationAuthority()?.presentation.match_summary?.task_mode !== 1;
  if (context === null || elements.metricPanel.hidden || !elements.metricPanel.open)
    return;
  const generation = replayMetricGeneration;
  if (
    replayMetricCatalog === null &&
    !replayMetricCatalogPending &&
    !replayMetricCatalogFailure
  ) {
    replayMetricCatalogPending = true;
    elements.metricSearchStatus.textContent = "Preparing the measurement catalog.";
    void getReplayMetricCatalog(state.token)
      .then((catalog) => {
        if (generation !== replayMetricGeneration) return;
        if (catalog.source_replay_digest !== context.digest) {
          throw new TypeError("Measurement catalog belongs to another replay.");
        }
        replayMetricCatalog = catalog;
        replayMetricSearchIndex = buildMetricSearchIndex(
          catalog.measurements,
          catalog.topics,
          catalog.agents,
          catalog.class_names,
        );
        elements.metricSearch.disabled = false;
        renderMetricNavigation(
          elements.metricSelection,
          elements.metricView,
          elements.metricViewField,
          catalog.topics,
        );
        renderReplayMetricSearch();
      })
      .catch((error) => {
        if (generation !== replayMetricGeneration) return;
        replayMetricCatalogFailure =
          error instanceof Error ? error.message : "Catalog request failed.";
        elements.metricSearchStatus.textContent = `Search unavailable: ${replayMetricCatalogFailure}`;
      })
      .finally(() => {
        if (generation !== replayMetricGeneration) return;
        replayMetricCatalogPending = false;
        renderReplayMetrics();
      });
  }
  if (replayMetricSummary?.key === context.key) {
    const summary = replayMetricSummary.summary;
    renderMetricNavigation(
      elements.metricSelection,
      elements.metricView,
      elements.metricViewField,
      summary.topics,
    );
    // The table draws team and agent icons once the catalog (and its agent
    // list) has arrived, so its arrival draws the table again.
    const agents = replayMetricCatalog?.agents;
    const renderKey = `${context.key}:${elements.metricSelection.value}:${elements.metricView.value}:${agents ? "icons" : "words"}`;
    if (renderKey !== replayMetricRenderKey) {
      renderMetricRows(
        elements.metricRows,
        elements.metricSelection.value,
        elements.metricView.value,
        summary,
        elements.metricDescription,
        agents ? { agents } : {},
      );
      replayMetricRenderKey = renderKey;
    }
    // The Scope switch already names the scope; the line gives the tick only.
    elements.metricStatus.textContent =
      summary.scope === "final"
        ? `Whole Game · Tick ${summary.simulator_step_count}`
        : `Up To Tick ${summary.simulator_step_count}`;
    focusReplayMeasurement();
    return;
  }
  elements.metricRows.replaceChildren();
  replayMetricRenderKey = "";
  if (replayMetricFailure?.key === context.key) {
    elements.metricStatus.textContent = replayMetricFailure.message;
    return;
  }
  elements.metricProgress.hidden = false;
  elements.metricRows.setAttribute("aria-busy", "true");
  elements.metricStatus.textContent = `${METRIC_PREPARATION_MESSAGE} Playback And POV Still Work.`;
  if (replayMetricRequest !== null) return;
  replayMetricRequest = context.key;
  void getReplayMetrics(state.token, context.frameIndex, context.scope)
    .then((summary) => {
      if (generation !== replayMetricGeneration) return;
      if (summary.source_replay_digest !== context.digest) {
        throw new TypeError("Metric analysis belongs to another replay.");
      }
      replayMetricSummary = { key: context.key, summary };
      replayMetricFailure = null;
    })
    .catch((error) => {
      if (generation !== replayMetricGeneration) return;
      replayMetricFailure = {
        key: context.key,
        message:
          error instanceof Error
            ? `Metrics unavailable: ${error.message}`
            : "Metrics unavailable.",
      };
    })
    .finally(() => {
      if (generation !== replayMetricGeneration) return;
      replayMetricRequest = null;
      renderReplayMetrics();
      refreshLedgerRail(installedPresentationAuthority()?.presentation ?? null);
    });
}

/**
 * Refresh the Replay Viewer rail (ledger.js) for the installed presentation:
 * its Full Record chips, Scope tick and the Match Summary at the head of the
 * Metrics tab.
 *
 * presentation is the installed presentation, or null. The Match Summary gets
 * the last metric summary only when it belongs to the current metric context
 * (same replay, scope and frame), else null (scores only, while it loads). The
 * Match Summary switch (ledger_summary) off hides the summary. Does nothing
 * outside the Replay Viewer. Side effects: the rail's DOM (see
 * ledger.mountLedgerRail's update). The rail is a docked researcher panel; it
 * draws nothing on or over the arena.
 *
 * @param {unknown} presentation
 * @returns {void}
 */
function refreshLedgerRail(presentation) {
  if (ledgerRail === null) return;
  const metricKey = replayMetricContext()?.key ?? null;
  ledgerRail.update({
    presentation,
    metricSummary:
      replayMetricSummary !== null && replayMetricSummary.key === metricKey
        ? replayMetricSummary.summary
        : null,
    summaryEnabled: isVisualFilterEnabled(visualFilterState, "ledger_summary"),
  });
}

/**
 * Return whether the installed trusted live presentation is scripted inspection.
 * Require a supported live transport kind and coherent pair; raw frame metadata
 * or an unjoined presentation cannot enable these controls.
 */
function liveScriptedInspectionOnly() {
  return (
    installedAuthorityIsCoherent() &&
    (state.frame?.frame_kind === "researcher_live_debugger" ||
      state.frame?.frame_kind === "actor_pov_live_debugger" ||
      state.frame?.frame_kind === "shared_obs_agent_pov_live_debugger") &&
    authorizedPresentationInspectionState(state.presentation).state_kind ===
      "live_scripted"
  );
}

/**
 * Return whether a command is allowed by the scripted-live UI gate.
 * Allow view changes, recording commands accepted by their lifecycle guard, and
 * fresh unmodified N/G keys. This browser check does not replace server validation.
 *
 * @param {Record<string, unknown>} command
 */
function allowedDuringLiveScriptedInspection(command) {
  if (command.command_type === "set_view") {
    return true;
  }
  if (SCRIPTED_INSPECTION_RECORDING_COMMANDS.has(String(command.command_type))) {
    return recordingCommandDecision(state.frame ?? {}, command).action === "allow";
  }
  if (
    command.command_type !== "keyboard" ||
    (command.key !== "n" && command.key !== "g")
  ) {
    return false;
  }
  return (
    command.shift_key === false &&
    command.ctrl_key === false &&
    command.alt_key === false &&
    command.meta_key === false &&
    command.repeat === false
  );
}

/**
 * Return recording metadata from the installed coherent transport, or null.
 * A stale or unjoined raw frame cannot control recording UI availability.
 */
function recordingStatus() {
  const frame = installedPresentationAuthority()?.transport;
  return isRecord(frame?.recording) ? frame.recording : null;
}

/**
 * Return whether installed recording metadata has left the recording lifecycle.
 * No recording metadata means this particular closeout gate is inactive.
 */
function recordingScientificControlsFenced() {
  const recording = recordingStatus();
  return recording !== null && recording.lifecycle !== "recording";
}

/**
 * Return whether recording metadata blocks restart without an available discard.
 * Require both restart_fenced and no explicit discard_available permission.
 */
function recordingRestartControlsBlocked() {
  const recording = recordingStatus();
  return (
    recording !== null &&
    recording.restart_fenced === true &&
    recording.discard_available !== true
  );
}

/**
 * Return whether the live battlefield may submit manual commands now.
 * Require a coherent nonterminal editable presentation and an idle connected page,
 * with no scripted-inspection or recording-closeout gate.
 */
function liveBattlefieldCommandsInteractive() {
  return (
    !isReplayMode() &&
    isAuthorizedPresentationFrame(state.presentation) &&
    installedAuthorityIsCoherent() &&
    !liveScriptedInspectionOnly() &&
    !isTerminal(state.frame) &&
    !state.busy &&
    !state.shuttingDown &&
    !state.resyncRequired &&
    !state.offline &&
    (!state.frame?.system_operation ||
      (state.frame?.fresh_launch_required &&
        state.frame.system_operation.state === "failed")) &&
    !recordingScientificControlsFenced()
  );
}

/**
 * Return whether the installed scripted-live battlefield may accept local inspection.
 * Require a nonterminal frame, no recording-closeout gate, and an idle connected
 * page. This does not permit manual simulator action editing.
 */
function scriptedBattlefieldLocallyActionable() {
  return (
    liveScriptedInspectionOnly() &&
    !isTerminal(state.frame) &&
    !state.busy &&
    !state.shuttingDown &&
    !state.resyncRequired &&
    !state.offline &&
    !recordingScientificControlsFenced()
  );
}

/**
 * Return whether local Agent POV inspection remains available during recording closeout.
 * Require a coherent live nonterminal Agent presentation and an idle connected page.
 * Simulator and pending-action controls remain governed by their separate gates.
 */
function liveAgentCloseoutInspectionActionable() {
  return (
    !isReplayMode() &&
    authorizedPresentationAudience(state.presentation) === "agent_pov" &&
    isAuthorizedPresentationFrame(state.presentation) &&
    installedAuthorityIsCoherent() &&
    recordingScientificControlsFenced() &&
    !isTerminal(state.frame) &&
    !state.busy &&
    !state.shuttingDown &&
    !state.resyncRequired &&
    !state.offline
  );
}

/**
 * Return whether any coherent authorized live presentation is installed.
 * This broad display check does not imply commands are idle, legal, or editable.
 */
function installedLiveScientificInspectionAvailable() {
  return (
    !isReplayMode() &&
    isAuthorizedPresentationFrame(state.presentation) &&
    installedAuthorityIsCoherent()
  );
}

/**
 * Paint replay identity, completion, audience, reference, range, and timeline controls.
 * Use the supplied coherent pair and the authorized artifact branch for its audience.
 * Do nothing outside replay or without a pair; availability follows current busy,
 * connection, selection, and playback state.
 *
 * @param {ReturnType<typeof installedPresentationAuthority>} installed
 */
function renderReplayMetadata(installed) {
  if (!isReplayMode()) {
    return;
  }
  if (installed === null) {
    return;
  }
  const frame = installed.transport;
  const presentation = installed.presentation;
  const artifactFacts =
    authorizedPresentationAudience(presentation) === "researcher"
      ? {
          artifact_summary: frame.artifact_summary,
          completion: frame.completion,
          processing: frame.processing,
        }
      : frame.artifact_facts;
  const summary = isRecord(artifactFacts?.artifact_summary)
    ? artifactFacts.artifact_summary
    : {};
  const reference = isRecord(summary.replay_reference) ? summary.replay_reference : {};
  const completion = isRecord(artifactFacts?.completion)
    ? artifactFacts.completion
    : {};
  elements.replayArtifactReference.textContent = String(
    reference.artifact_id ?? "Unavailable",
  );
  elements.replayArtifactReference.removeAttribute("title");
  registerTooltipOwner(
    elements.replayArtifactReference,
    createSemanticDescriptor({
      kind: "control",
      id: "replay-artifact-reference",
      title: "Replay Artifact",
      tone: "information",
      accent: "none",
      summary: "Canonical identity for the loaded immutable replay artifact.",
      rows: [
        {
          label: "Artifact",
          value: String(reference.artifact_id ?? "Unavailable"),
          metadata: { compact: true, full: true },
        },
        {
          label: "Canonical Digest",
          value: String(reference.canonical_digest_sha256 ?? "Unavailable"),
          metadata: { compact: false, full: true },
        },
      ],
      sections: [],
      metadata: { compact: true, full: true },
      anchor: "element",
    }),
  );
  elements.replayCompletionBadge.textContent = humanize(
    completion.completion_state ?? "unavailable",
  );
  elements.replayProcessingBadge.textContent = "Authorized replay";
  registerTooltipOwner(
    elements.replayCompletionBadge,
    explainTechnicalFact("completion"),
    { inspectable: false },
  );
  registerTooltipOwner(
    elements.replayProcessingBadge,
    explainTechnicalFact("processing"),
    { inspectable: false },
  );
  elements.replayEndReason.textContent = String(
    completion.public_end_or_failure_reason ??
      completion.end_or_failure_reason ??
      (asArray(completion.completion_bases).length > 0
        ? asArray(completion.completion_bases).map(humanize).join(" + ")
        : "Captured prefix"),
  );
  elements.replayRangesButton.setAttribute(
    "aria-pressed",
    String(installedPresentationRangesVisible(presentation)),
  );
  elements.replayRangesButton.disabled =
    state.busy || state.shuttingDown || state.resyncRequired || state.offline;
  const scene = installedAuthorizedPresentationSceneView(presentation);
  const selectedOwnerKey = isRecord(scene?.selection)
    ? scene.selection.inspection_owner_presentation_key
    : null;
  elements.replayClearReferenceButton.disabled =
    state.busy ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline ||
    typeof selectedOwnerKey !== "string";
  renderReplayTimelineControls(
    replayTimelineElements,
    replayTimelineRenderState(replayPlayback.snapshot()),
  );
}

/** @type {Readonly<Record<string, string>>} */
const recordingPersistenceLabels = Object.freeze({
  target_unavailable: "Destination unavailable",
  publication_failed: "Publication failed",
  verification_failed: "Publication verification failed",
});

/**
 * Paint live play's save button in the top bar (#recording-finish-button)
 * from saveButtonState (tactics-turn.js, the one owner of its words): its
 * label, its action in data-action ("finish", "open" or "retry"; none for
 * "Saving Replay…"), disabled unless the state is usable and nothing blocks
 * input (blocked), and its hover help. A null state hides the button (the
 * match records nothing, the page is the Replay Viewer, or Exit was
 * accepted). The click is handled by saveButtonClick.
 *
 * @param {ReturnType<typeof saveButtonState>} saveButton The state to show,
 *   or null to hide the button.
 * @param {boolean} [blocked] True while the page or the recording blocks
 *   input (busy, offline, closed, or a recording rule). Default false.
 * @returns {void}
 */
function paintSaveButton(saveButton, blocked = false) {
  const button = elements.recordingFinishButton;
  button.hidden = saveButton === null;
  if (saveButton === null) {
    button.disabled = true;
    delete button.dataset.action;
    return;
  }
  if (button.textContent !== saveButton.label) button.textContent = saveButton.label;
  if (saveButton.action === null) delete button.dataset.action;
  else button.dataset.action = saveButton.action;
  button.disabled = blocked || !saveButton.usable;
  if (button.dataset.helpLabel !== saveButton.label) {
    button.dataset.helpLabel = saveButton.label;
    registerControlHelpOwner(
      button,
      "#recording-finish-button",
      saveButton.label,
      saveButton.help,
    );
  }
}

/**
 * Write the Recording popover's reason line (#pop-reason) for one recording
 * status: after a failed save it says why (saveFailedReason, tactics-turn.js);
 * otherwise it is emptied and hidden. Side effects: that line's text and
 * `hidden` only.
 *
 * @param {Readonly<Record<string, any>> | null} recording The recording
 *   status, or null when the match records nothing.
 * @returns {void}
 */
function paintSaveFailedReason(recording) {
  const reason = saveFailedReason(recording);
  elements.recordingPopReason.textContent = reason;
  elements.recordingPopReason.hidden = reason === "";
}

/**
 * Paint live recording lifecycle, persistence facts, actions, and progress hints.
 * Use the supplied installed pair and recording metadata; hide replay/unavailable
 * recording surfaces and close obsolete discard dialogs. Progress text describes
 * current request state, not a fabricated percentage. Writes the Recording
 * pill (#rec-pill-text) and the popover's state line (#pop-state) through
 * recordingWords (tactics-turn.js, the one owner of these words), and after a
 * failed save the reason line under it (#pop-reason, paintSaveFailedReason), with the
 * `saved` class once saved, and the top bar's save button through
 * saveButtonState and paintSaveButton (Finish And Save, Saving Replay…, Open
 * Replay or Retry Save; owner, 4 Oct 2026: it turns into Open Replay instead
 * of disappearing). Finish And Save in the popover and the Stopped overlay
 * and Retry Save show only when they work; the popover's Open Replay stays
 * visible and is disabled until a save exists; after an accepted Exit, the
 * popover's Finish And Save and Retry Save and the top bar's save button
 * are hidden. While the server
 * saves a finished game in the background (lifecycle "sealed"), the words
 * read "Saving Replay…" and the popover note says Open Replay works once it
 * says Saved. After an accepted Exit, the Exit answer's own recording status
 * (state.exitRecording) is painted instead of the installed one, in its
 * closed form (a game closed before its first tick reads "No Replay").
 *
 * @param {ReturnType<typeof installedPresentationAuthority>} installed
 */
function renderRecordingControls(installed) {
  elements.recordingMetricsProgress.hidden = true;
  elements.recordingMetricsHelp.hidden = true;
  if (installed === null) {
    // An Exit answer whose frame raced the shutdown may leave no installed
    // pair: its own recording words still replace "Saving Replay…".
    if (state.shuttingDown && state.exitRecording !== null) {
      const words = recordingWords(state.exitRecording, 0, true);
      paintSaveButton(null);
      elements.recordingPillText.textContent = words.pill;
      elements.recordingPopState.textContent = words.popover;
      paintSaveFailedReason(state.exitRecording);
      elements.recordingBadge.classList.toggle(
        "saved",
        ["saved", "reviewing"].includes(String(state.exitRecording.lifecycle)),
      );
    }
    return;
  }
  const recording =
    state.shuttingDown && state.exitRecording !== null
      ? state.exitRecording
      : recordingStatus();
  elements.recordingReplayLink.hidden =
    openedReplayEpisodeId !== installed.transport.episode_id ||
    !elements.recordingReplayLink.hasAttribute("href");
  const unavailable = isReplayMode() || recording === null;
  elements.recordingPanel.toggleAttribute("hidden", unavailable);
  elements.recordingBadge.toggleAttribute("hidden", unavailable);
  if (unavailable) {
    paintSaveButton(null);
    paintSaveFailedReason(null);
    delete document.documentElement.dataset.recordingLifecycle;
    if (elements.recordingDiscardDialog.open) {
      elements.recordingDiscardDialog.close();
    }
    pendingRecordingReplacement = null;
    return;
  }

  document.documentElement.dataset.recordingLifecycle = recording.lifecycle;
  elements.recordingBadge.dataset.lifecycle = recording.lifecycle;
  const words = recordingWords(
    recording,
    integer(installed.transport.simulator_step_count, 0),
    state.shuttingDown,
  );
  elements.recordingPillText.textContent = words.pill;
  elements.recordingPopState.textContent = words.popover;
  paintSaveFailedReason(recording);
  const saved = recording.lifecycle === "saved" || recording.lifecycle === "reviewing";
  elements.recordingBadge.classList.toggle("saved", saved);
  elements.recordingLifecycle.textContent = humanize(recording.lifecycle);
  elements.recordingProgress.textContent = `${recording.captured_transition_count} / ${recording.expected_transition_count} transitions`;
  elements.recordingCompletion.textContent =
    recording.completion_state === null
      ? "Capture in progress"
      : recording.completion_reason === null
        ? humanize(recording.completion_state)
        : `${humanize(recording.completion_state)} · ${humanize(recording.completion_reason)}`;

  const persistenceLabel =
    recordingPersistenceLabels[recording.persistence_error_code] ?? null;
  elements.recordingPersistenceFact.toggleAttribute(
    "hidden",
    persistenceLabel === null,
  );
  elements.recordingPersistenceError.textContent =
    persistenceLabel ?? "No persistence error";

  const interactionDisabled =
    state.busy ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline ||
    recordingCommandDecision(installed.transport, {
      command_type: "finish_and_review",
    }).action === "block";
  const actionAvailability = [
    [elements.recordingPopFinishButton, recording.finish_available === true],
    [elements.stoppedSaveButton, recording.finish_available === true],
    [elements.recordingReviewButton, recording.review_available === true],
    [elements.recordingRetryButton, recording.retry_available === true],
  ];
  for (const [element, available] of actionAvailability) {
    // Open Replay stays in the Recording popover, disabled until a save
    // exists, as on the vision page; the others show only when they work,
    // so none shows once Exit was accepted.
    if (element !== elements.recordingReviewButton) {
      element.toggleAttribute("hidden", !available || state.shuttingDown);
    } else {
      element.hidden = false;
    }
    element.disabled = interactionDisabled || !available;
  }
  const saveAsAvailable = recording.save_as_available === true;
  elements.recordingSaveAsControl.toggleAttribute("hidden", !saveAsAvailable);
  elements.recordingSaveAsInput.disabled = interactionDisabled || !saveAsAvailable;
  elements.recordingSaveAsButton.disabled = interactionDisabled || !saveAsAvailable;

  elements.recordingStatusNote.textContent =
    recording.lifecycle === "recording" && recording.discard_available === true
      ? "Capture is active. Restart offers Save, Discard or Cancel."
      : recording.lifecycle === "recording"
        ? "Capture starts at frame zero. Finish And Save ends this recording."
        : recording.lifecycle === "persistence_failed"
          ? "Your recording is kept in memory. Retry Save or choose a new file name with Save As."
          : recording.lifecycle === "saved"
            ? "Replay saved. Inspect this game, open it in a separate Viewer, or Restart to play again."
            : recording.lifecycle === "reviewing"
              ? "Replay saved. Open Replay uses a separate Viewer."
              : recording.lifecycle === "sealed" ||
                  recording.lifecycle === "finalized_unsaved"
                ? "The game is over. Saving the replay now; Open Replay works once it says Saved."
                : "This recording has ended. Save or discard it before starting another match.";
  const preparation =
    state.busy && !state.offline && !state.resyncRequired
      ? activeLiveCommandTransaction?.recordingPreparation
      : null;
  if (preparation) {
    elements.recordingMetricsProgress.hidden = false;
    elements.recordingMetricsHelp.hidden = false;
    elements.recordingStatusNote.textContent =
      preparation === "metrics"
        ? REPLAY_SAVE_MESSAGE
        : "Saving the recorded game. Longer replays can take longer. Metrics can be computed in Replay Viewer.";
  }
  const saveButton = saveButtonState(recording, {
    saving: preparation === "metrics",
    reopen:
      openedReplayEpisodeId === installed.transport.episode_id &&
      elements.recordingReplayLink.hasAttribute("href"),
    closed: state.shuttingDown,
  });
  paintSaveButton(saveButton, interactionDisabled);

  if (recording.discard_available !== true && elements.recordingDiscardDialog.open) {
    elements.recordingDiscardDialog.close();
    pendingRecordingReplacement = null;
  }
}

/**
 * Return an array unchanged, or a fresh empty array for any other input.
 * This does not validate array elements.
 *
 * @param {unknown} value
 * @returns {any[]}
 */
function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * Return an integer numeric input unchanged, otherwise fallback, which defaults to 0.
 * Numeric strings are not coerced into accepted integers.
 *
 * @param {unknown} value
 * @param {number} fallback
 * @returns {number}
 */
function integer(value, fallback = 0) {
  return Number.isInteger(value) ? Number(value) : fallback;
}

/**
 * Build a readable agent label using the current authorized display ID when available.
 * A nonempty public ID falls back to itself; invalid IDs return Agent unavailable.
 *
 * @param {unknown} publicAgentId
 * @returns {string}
 */
function agentIdentity(publicAgentId) {
  const displayId = authorizedPresentationAgentDisplayId(
    state.presentation,
    publicAgentId,
  );
  return typeof publicAgentId === "string" && publicAgentId.length > 0
    ? `Agent ID ${displayId ?? publicAgentId}`
    : "Agent unavailable";
}

/**
 * Read whether transport is at its terminal or final-frame boundary.
 * Replay compares current and final frame indices; live accepts the supported
 * boolean/object terminal forms and legacy terminated/truncated flags. This is
 * a compatibility reader, not a validator of an arbitrary frame.
 *
 * @param {unknown} frame
 */
function isTerminal(frame) {
  const record = isRecord(frame) ? frame : {};
  if (record.viewer_mode === "replay" && isRecord(record.cursor)) {
    return record.cursor.frame_index === record.cursor.final_frame_index;
  }
  if (typeof record.terminal === "boolean") {
    return record.terminal;
  }
  if (isRecord(record.terminal)) {
    return Boolean(
      record.terminal.is_sealed ||
        record.terminal.terminated ||
        record.terminal.truncated ||
        record.terminal.reached_declared_horizon,
    );
  }
  return Boolean(record.terminated || record.truncated);
}

/**
 * Return whether a KeyboardEvent key value names a Lock In key: Enter (any
 * case), Space (" ") or N (any case). N is one tick: in interactive play
 * Python treats it as Space (Manual: Lock In; Policies Only: play one turn),
 * so the page queues, ignores and refuses it exactly as it does Space. Pure.
 *
 * @param {unknown} key The command's key value; a non-string returns false.
 * @returns {boolean} True for "Enter", " " or "n" in any letter case.
 */
function isLockInKeyName(key) {
  if (typeof key !== "string") {
    return false;
  }
  const lower = key.toLowerCase();
  return lower === "enter" || key === " " || lower === "n";
}

/**
 * Return whether a command is a fresh Lock In key (Enter, Space or N, see
 * isLockInKeyName) with every modifier and repeat false. Only a keyboard
 * command with one of those string keys qualifies. Space and N are Lock In
 * too (Policies Only: play one turn), so one pressed while an order is still
 * being applied is kept, not lost.
 *
 * @param {Record<string, unknown>} command
 */
function isFreshUnmodifiedEnter(command) {
  return (
    command.command_type === "keyboard" &&
    isLockInKeyName(command.key) &&
    command.shift_key === false &&
    command.ctrl_key === false &&
    command.alt_key === false &&
    command.meta_key === false &&
    command.repeat === false
  );
}

/**
 * Keep a focus change (a portrait, roster or rail click that picks which
 * champion the deck shows: roster_selection with role "control") made while
 * one live request is out, so the click is not lost. It joins that request's
 * queue (deferredDrafts) and is sent after it, like a queued order key; a
 * later focus change replaces an earlier queued one (the last click wins).
 * Returns false, and keeps nothing, in the Replay Viewer, for any other
 * command, when no live request owns the page, when the page must resync or
 * is offline, when a Lock In is already queued (it must stay last) or when the
 * queue is full (MAX_DEFERRED_DRAFTS); the caller then refuses as before.
 * Changes only the active request's queue; sends nothing. Agent POV: the
 * queued command is the same one the click would have sent, and the host
 * still decides which champions the POV team may focus.
 *
 * @param {Readonly<Record<string, unknown>>} command A command about to be sent.
 * @returns {boolean} True when the focus change was queued.
 */
function retainFocusChange(command) {
  const transaction = activeLiveCommandTransaction;
  if (
    isReplayMode() ||
    transaction === null ||
    state.resyncRequired ||
    state.offline ||
    command.command_type !== "roster_selection" ||
    command.role !== "control" ||
    transaction.deferredSubmit !== null
  ) {
    return false;
  }
  const kept = transaction.deferredDrafts.filter(
    (queued) =>
      !(queued.command_type === "roster_selection" && queued.role === "control"),
  );
  if (kept.length >= MAX_DEFERRED_DRAFTS) {
    return false;
  }
  kept.push(Object.freeze({ ...command }));
  transaction.deferredDrafts.splice(0, transaction.deferredDrafts.length, ...kept);
  return true;
}

/**
 * Return whether a non-null command is the Escape keyboard command.
 * This recognizes key identity only; modifier and repeat checks belong elsewhere.
 *
 * @param {Readonly<Record<string, unknown>> | null} command
 */
function isEscapeKeyboardCommand(command) {
  return (
    command !== null &&
    command.command_type === "keyboard" &&
    typeof command.key === "string" &&
    command.key.toLowerCase() === "escape"
  );
}

/**
 * Consume eligible live input while one request owns the battlefield.
 * Retain up to MAX_DEFERRED_DRAFTS order edits (fresh draft keys, a Shift +
 * click target pick or a right click that clears the target) and fresh
 * unmodified G presses (Ranges; in Agent POV G then toggles the page's own
 * reach rings, in Oracle View it goes to Python), oldest first, and a
 * following Lock In key (Enter, Space or N: isLockInKeyName) for ordered
 * dispatch after a coherent successor.
 * Escape also saves focus-release intent. Return whether the input was
 * consumed; replay and other unavailable states keep native behavior.
 *
 * @param {Record<string, unknown>} command
 */
function retainFencedCommand(command) {
  const transaction = activeLiveCommandTransaction;
  if (
    isReplayMode() ||
    !state.busy ||
    transaction === null ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline
  ) {
    return false;
  }
  if (isEscapeKeyboardCommand(command)) {
    transaction.releaseFocusAfterSettlement = true;
  }
  // A refused 1 or 2 needs no server: when the order being applied cannot
  // change the target, refuse it now in the vision's words.
  if (transaction.keepsTarget) {
    const refusal = laneRefusal(
      command,
      ownOrderFacts(state.presentation, state.frame),
    );
    if (refusal !== null) {
      setNotice(refusal.sentence, "warning");
      shakeSlot(
        refusal.lane === "basic" ? elements.basicButton : elements.ultimateButton,
      );
      transaction.lateRefusal = refusal.sentence;
      return true;
    }
  }
  const targetEdit =
    command.command_type === "battlefield_pointer" &&
    (command.button === "secondary" || command.shift_key === true);
  // G only changes what the human sees. Pressed while a request is out (for
  // example right after the click that turns keys on), it waits its turn
  // instead of being dropped without a word.
  const rangesKey =
    command.command_type === "keyboard" &&
    typeof command.key === "string" &&
    command.key.toLowerCase() === "g" &&
    command.shift_key === false &&
    command.ctrl_key === false &&
    command.alt_key === false &&
    command.meta_key === false &&
    command.repeat === false;
  if (isDeferredDraftKey(command) || targetEdit || rangesKey) {
    if (
      transaction.deferredDrafts.length < MAX_DEFERRED_DRAFTS &&
      transaction.deferredSubmit === null &&
      (targetEdit || rangesKey || isFreshDeferredDraft(command))
    ) {
      // A queued pick changes the target, so a later 1 or 2 is judged after it.
      if (targetEdit) transaction.keepsTarget = false;
      transaction.deferredDrafts.push(Object.freeze({ ...command }));
      setNotice(
        "Input queued; waiting for the current battlefield update first…",
        "info",
      );
      renderConnection();
    }
    return true;
  }
  if (command.command_type !== "keyboard" || !isLockInKeyName(command.key)) {
    return false;
  }
  if (
    transaction.deferredSubmit === null &&
    (transaction.allowsDeferredSubmit || transaction.deferredDrafts.length > 0) &&
    isFreshUnmodifiedEnter(command)
  ) {
    transaction.deferredSubmit = Object.freeze({ ...command });
    setNotice(
      "Submit queued; waiting for the staged action to be installed first…",
      "info",
    );
    renderConnection();
  }
  return true;
}

/**
 * Return the browser advisory availability and notice for a command/frame pair.
 * Reject draft and submit roles at a terminal boundary; other commands pass this
 * particular gate. Python repeats the real authorization before changing state.
 *
 * @param {Record<string, unknown>} command
 * @param {Record<string, any> | null} frame
 */
function modeAvailability(command, frame) {
  const role = commandRole(command);
  if (
    frame?.fresh_launch_required &&
    (role !== null ||
      [
        "reset",
        "run",
        "switch_scenario",
        "set_combat_configuration",
        "battlefield_pointer",
      ].includes(String(command.command_type)) ||
      (command.command_type === "keyboard" &&
        ["r", "escape"].includes(String(command.key).toLowerCase())) ||
      (command.command_type === "roster_selection" && command.role === "target"))
  )
    return {
      allowed: false,
      notice:
        "Start a fresh Python launch before playing again. The accepted game remains available for inspection and saving.",
    };
  if (!role) {
    return { allowed: true, notice: null };
  }
  if (isTerminal(frame)) {
    return {
      allowed: false,
      notice: "The episode is terminal; reset to continue.",
    };
  }
  return { allowed: true, notice: null };
}

/**
 * Read the installed raw transport revision, falling back to 0 if it is not an integer.
 * Callers use it as the command base revision; it does not establish authority.
 */
function currentRevision() {
  return integer(state.frame?.revision, 0);
}

/**
 * Store a notice message and level for the next connection render.
 * The level defaults to info; this helper does not repaint or send a request.
 * In the BattleClient and DevClient the message line's message (#msg, through
 * paintStatusLine) also shows it ("warning" and "error" keep their level and
 * look refused, anything else shows as info). In the Replay Viewer the Rune
 * Scrubber shows it under the readout for 4 s (runeScrubber.showNotice; the
 * top-bar #notice pill stays hidden there), after the same plain-words
 * translation of host notices the live message line uses (plainHostNotice); a
 * warning or error whose words that translation changed also goes to the
 * browser console in one line (logRawError), so the exact host text is kept.
 * Side effects: page state and that message line or notice line.
 *
 * @param {string} message
 * @param {string} level
 */
function setNotice(message, level = "info") {
  state.notice = message;
  state.noticeLevel = level;
  if (!isReplayMode() && typeof message === "string") {
    // The live deck's message line repeats the latest notice in plain words.
    paintStatusLine(message, level === "error" || level === "warning" ? level : "info");
  } else if (runeScrubber !== null && typeof message === "string") {
    // Host words (a server notice, a stale or repeated command) read in plain
    // words, through the BattleClient's own translation (plainHostNotice).
    const plain = plainHostNotice(message);
    if (plain !== message && (level === "error" || level === "warning")) {
      logRawError("Host notice", message);
    }
    runeScrubber.showNotice(plain, level);
  }
}

/**
 * Write a raw error to the browser console as one line, for developers
 * (owner, 4 Oct 2026, Micro List M15). The Replay Viewer page shows only plain
 * words; this keeps the exact text a bug report needs.
 *
 * @param {string} context What failed, in a few words ("PNG export failed").
 * @param {unknown} error Any thrown value or host text: an Error gives its name
 *   and message, anything else its string form. Runs of whitespace (line
 *   breaks included) become one space.
 * @returns {void} Side effect: one console.error call, "MARL-BGs <context>:
 *   <text>".
 */
function logRawError(context, error) {
  const text =
    error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  console.error(`MARL-BGs ${context}: ${text.replace(/\s+/gu, " ").trim()}`);
}

/**
 * Write one message into the deck's message line (#msg, the deck's header), as
 * the vision page does: the text, `data-level` ("info", "warning" or "error")
 * and the `refuse` class for a warning or error (a refusal or a problem). Also
 * hands the same message to the deck (showStatusMessage), which keeps it for
 * its own repaint. BattleClient and DevClient only; the caller checks that.
 *
 * @param {string} text The message; "" empties the line.
 * @param {"info" | "warning" | "error"} level How the line is styled.
 * @returns {void}
 */
function paintStatusLine(text, level) {
  elements.statusMessage.textContent = text;
  elements.statusMessage.dataset.level = level;
  elements.statusMessage.classList.toggle("refuse", level !== "info");
  showStatusMessage(text, level);
}

/**
 * Clear the top-bar notice after routine work: a load, a frame update or a
 * request still waiting for its answer. Routine work makes no top-bar notice;
 * the top bar keeps only the connection status and the Recording pill. Warnings,
 * errors and results the user asked for still go through setNotice.
 *
 * In the BattleClient and DevClient the message line's message (#msg, through
 * paintStatusLine) shows statusText at info level (an empty string clears
 * that line). The Replay Viewer has no deck, so statusText is not shown there.
 * This helper does not repaint or send a request. Side effects: page state and
 * that message line.
 *
 * @param {string} [statusText] Plain words for the deck message line. Defaults
 *   to "" (an empty line).
 * @returns {void}
 */
function clearNotice(statusText = "") {
  state.notice = null;
  state.noticeLevel = "info";
  if (!isReplayMode()) {
    paintStatusLine(statusText, "info");
  }
}

/**
 * @typedef {Readonly<{
 *   mode: "one" | "each" | "watch",
 *   audience: string | null,
 *   viewers: string,
 *   actorId: string | null,
 *   actorName: string | null,
 *   actorClass: string | null,
 *   actorManual: boolean,
 *   actorController: string,
 *   editable: boolean,
 *   basicName: string,
 *   ultimateName: string,
 *   isMage: boolean,
 *   cooldown: number,
 *   move: number | null,
 *   lane: string | null,
 *   targetAction: number,
 *   targetName: string | null,
 *   pairs: ReadonlyArray<ReadonlyArray<boolean>>,
 *   unseenEnemies: number,
 *   showRanges: boolean,
 * }>} OwnOrderFacts
 */

/**
 * Read the facts the message line needs about the champion in focus and its
 * staged order, from one installed live presentation and its transport frame.
 *
 * Agent POV: everything comes from the audience's own arena scene (champion
 * names, classes, cooldown) and the audience's own live inspection (the
 * focused champion's draft and decision mask), never from researcher_space or
 * match_summary. The controller labels and Manual slots come from the match
 * setup (combat_configuration), which every audience already sees in the
 * scorebug. A champion the scene does not draw is never named; its name is
 * null.
 *
 * @param {unknown} presentation The installed authorized presentation.
 * @param {Record<string, any> | null} frame The installed transport frame.
 * @returns {OwnOrderFacts | null} The facts, or null outside live play or
 *   without an authorized presentation.
 */
function ownOrderFacts(presentation, frame) {
  if (isReplayMode() || !isAuthorizedPresentationFrame(presentation) || !frame) {
    return null;
  }
  const config = isRecord(frame.combat_configuration) ? frame.combat_configuration : {};
  const labels = asArray(config.slot_labels).map((label) => String(label));
  const manualSlots = asArray(config.manual_slots).filter((slot) =>
    Number.isInteger(slot),
  );
  const perTeam = labels.length > 0 ? labels.length / 2 : 5;
  const manualTeams = new Set(manualSlots.map((slot) => (slot < perTeam ? 1 : 2)));
  const mode =
    manualSlots.length === 0 ? "watch" : manualTeams.size > 1 ? "each" : "one";
  const scene = authorizedPresentationSceneView(presentation);
  const agents = asArray(scene?.agents).filter(isRecord);
  /** @param {unknown} id */
  const agentFor = (id) =>
    agents.find((agent) => String(agent.public_agent_id) === String(id)) ?? null;
  /** @param {Record<string, any> | null} agent */
  const nameOf = (agent) => {
    if (agent === null) return null;
    const id = String(agent.public_agent_id);
    const shown = authorizedPresentationAgentDisplayId(presentation, id) ?? id;
    return `${String(agent.class_name ?? "Champion")} (ID ${shown})`;
  };
  const inspectionState = authorizedPresentationInspectionState(presentation);
  const inspection = inspectionState.inspection;
  const actorId =
    isRecord(inspection) && inspection.actor_public_agent_id !== undefined
      ? String(inspection.actor_public_agent_id)
      : null;
  const actor = actorId === null ? null : agentFor(actorId);
  const actorSlot = actorId === null ? Number.NaN : Number(actorId);
  const editable = inspectionState.state_kind === "live_editable";
  const draft =
    editable && isRecord(inspection?.draft_action) ? inspection.draft_action : null;
  const mask = isRecord(inspection?.decision_mask) ? inspection.decision_mask : null;
  const targetAction = draft !== null ? integer(draft.target_action, 0) : 0;
  const targetRow =
    mask === null
      ? null
      : (asArray(mask.target_actions).find(
          (row) => isRecord(row) && Number(row.target_action) === targetAction,
        ) ?? null);
  const names = classNamesV1(actor?.class_id);
  const audience = authorizedPresentationAudience(presentation);
  const ownTeam = Number(actor?.team_id ?? (manualSlots[0] < perTeam ? 1 : 2));
  const enemyShown = agents.filter((agent) => Number(agent.team_id) !== ownTeam).length;
  return Object.freeze({
    mode,
    audience,
    viewers: mode === "watch" ? `Team ${ownTeam === 2 ? "B" : "A"}` : "your team",
    actorId,
    actorName: nameOf(actor),
    actorClass: actor === null ? null : String(actor.class_name ?? "Champion"),
    actorManual: manualSlots.includes(actorSlot),
    actorController: manualSlots.includes(actorSlot)
      ? "You"
      : (labels[actorSlot] ?? "a policy"),
    editable,
    basicName: names?.basicName ?? "Basic",
    ultimateName: names?.ultimateName ?? "Ultimate",
    isMage: names?.className === "Mage",
    cooldown: integer(actor?.ultimate_cooldown_remaining, 0),
    move: draft === null ? null : integer(draft.move_action, 0),
    lane: draft === null ? null : String(draft.armed_lane ?? "none"),
    targetAction,
    targetName:
      targetAction > 0 && isRecord(targetRow)
        ? nameOf(agentFor(targetRow.target_public_agent_id))
        : null,
    pairs: asArray(mask?.target_use_ultimate_joint_mask).map((pair) =>
      asArray(pair).map((value) => value === true),
    ),
    unseenEnemies: audience === "agent_pov" ? Math.max(0, perTeam - enemyShown) : 0,
    showRanges:
      audience === "agent_pov" ? agentLocalRangesVisible : frame.show_ranges === true,
  });
}

/**
 * Decide, before sending, whether the page refuses a 1 or 2 press (or a
 * Basic or Ultimate slot click) for the champion in focus, in the vision's
 * words. Mirrors Python's own check of the staged order against the decision
 * mask, so a refusal never blocks an order Python would accept.
 *
 * @param {Record<string, unknown>} command The command about to be sent.
 * @param {OwnOrderFacts | null} facts The facts before sending.
 * @returns {{sentence: string, lane: "basic" | "ultimate"} | null} The
 *   refusal, or null when the page lets the command go to Python.
 */
function laneRefusal(command, facts) {
  if (
    facts === null ||
    !facts.editable ||
    command.command_type !== "keyboard" ||
    (command.key !== "1" && command.key !== "2")
  ) {
    return null;
  }
  const pair = facts.pairs[facts.targetAction] ?? [];
  if (command.key === "1") {
    if (facts.targetAction === 0) {
      return {
        sentence: statusSentence({ kind: "needs_target", ability: facts.basicName }),
        lane: "basic",
      };
    }
    return pair[0] === true
      ? null
      : {
          sentence: statusSentence({
            kind: "not_allowed",
            ability: facts.basicName,
            onTarget: true,
            readyIn: 0,
          }),
          lane: "basic",
        };
  }
  if (facts.isMage) {
    return (facts.pairs[0] ?? [])[1] === true
      ? null
      : {
          sentence: statusSentence({
            kind: "not_allowed",
            ability: facts.ultimateName,
            onTarget: false,
            readyIn: facts.cooldown,
          }),
          lane: "ultimate",
        };
  }
  if (facts.targetAction === 0) {
    return {
      sentence: statusSentence({ kind: "needs_target", ability: facts.ultimateName }),
      lane: "ultimate",
    };
  }
  return pair[1] === true
    ? null
    : {
        sentence: statusSentence({
          kind: "not_allowed",
          ability: facts.ultimateName,
          onTarget: true,
          readyIn: facts.cooldown,
        }),
        lane: "ultimate",
      };
}

/**
 * Say three host notices in the vision page's words; every other notice is
 * shown as the host wrote it. Python's "The Systems are choosing this turn."
 * (its answer to Lock In and Play One Turn) keeps the page's own Lock In
 * sentence: "Waiting for all players to submit their actions…"
 * (WAITING_FOR_PLAYERS, the same in Policies Only). Python's "This actor is controlled by a policy.
 * Select a Manual actor to edit actions." becomes "This champion is played by
 * a policy. You can look, not command. Pick one of yours to give orders."
 * ("Nobody is yours this game. You can look, not command." in Policies Only),
 * and its Agent POV focus notice "Looking at Hunter · ID 2 of Team A, played
 * by a policy. ..." becomes "Looking at the Hunter (ID 2), played by
 * <controller>." The controller is the slot's label from the match setup
 * (combat_configuration.slot_labels), which every audience sees; the class and
 * ID come from the host's own sentence, which names only a champion the
 * audience may focus.
 *
 * @param {string | null} notice The host's notice, or null.
 * @param {OwnOrderFacts | null} facts Facts after the command.
 * @returns {string | null} The vision sentence, or null to keep the notice.
 */
function hostNoticeSentence(notice, facts) {
  if (typeof notice !== "string") return null;
  if (notice === "The Systems are choosing this turn.") {
    return statusSentence({ kind: "lock_in", watch: facts?.mode === "watch" });
  }
  if (notice.startsWith("This actor is controlled by a policy.")) {
    return statusSentence({ kind: "policy_champion", watch: facts?.mode === "watch" });
  }
  const look = /^Looking at (\w+) · ID (\d+) of Team [AB], played by a policy\./u.exec(
    notice,
  );
  if (look !== null) {
    const labels = asArray(state.frame?.combat_configuration?.slot_labels);
    return statusSentence({
      kind: "look",
      target: `${look[1]} (ID ${look[2]})`,
      controller: String(labels[Number(look[2])] ?? "a policy"),
    });
  }
  return null;
}

/**
 * Restart the shake animation on a refused action slot, as the vision does.
 * Side effects: toggles the `shake` class on the slot button only.
 *
 * @param {HTMLElement} button The Basic or Ultimate slot button.
 * @returns {void}
 */
function shakeSlot(button) {
  button.classList.remove("shake");
  void button.offsetWidth;
  button.classList.add("shake");
}

/**
 * Return the status sentence for one accepted own command, comparing the
 * facts before and after it. Returns null when no vision sentence applies
 * (the caller then keeps Python's notice or the routine empty line).
 *
 * @param {Record<string, unknown>} command The command Python accepted.
 * @param {OwnOrderFacts | null} before Facts before sending.
 * @param {OwnOrderFacts | null} after Facts after the successor installed.
 * @param {"field" | "control" | "page"} source "field" for a key or click on
 *   the battlefield (Esc then also turns keys off), "control" for a deck or
 *   rail button, "page" for anything else.
 * @returns {string | null}
 */
function acceptedCommandSentence(command, before, after, source) {
  if (after === null) return null;
  const type = command.command_type;
  const key = typeof command.key === "string" ? command.key.toLowerCase() : "";
  if (type === "set_view") {
    return statusSentence({
      kind: "view",
      view: after.audience === "agent_pov" ? "pov" : "oracle",
      unseen: after.unseenEnemies,
      viewers: after.viewers,
      watch: after.mode === "watch",
    });
  }
  if (isSubmissionCommand(command)) {
    return statusSentence({ kind: "lock_in", watch: after.mode === "watch" });
  }
  if (type === "keyboard" && key === "g") {
    return statusSentence({ kind: "ranges", shown: after.showRanges });
  }
  if (type === "run") return statusSentence({ kind: "running" });
  if (type === "pause") {
    // A pause between turns takes effect at once; during a turn it waits.
    return statusSentence({
      kind:
        state.frame?.run_state === "paused" && !state.pageRun ? "paused" : "pausing",
    });
  }
  if (type === "cancel_system") return statusSentence({ kind: "work_cancelled" });
  if (
    type === "roster_selection" &&
    command.role === "control" &&
    asArray(state.frame?.combat_configuration?.manual_slots).includes(
      command.global_slot,
    )
  ) {
    // Back to (or on to) one of your own champions; Agent POV follows it.
    return statusSentence({
      kind: "focus_own",
      className: after.actorClass ?? "champion",
    });
  }
  if (before !== null && before.actorId !== after.actorId && after.actorId !== null) {
    return after.actorManual
      ? statusSentence({ kind: "focus_own", className: after.actorClass ?? "champion" })
      : after.actorName === null
        ? null
        : statusSentence({
            kind: "look",
            target: after.actorName,
            controller: after.actorController,
          });
  }
  if (!after.editable) return null;
  if (type === "keyboard" && "wasdqezcx".includes(key) && key.length === 1) {
    return statusSentence({ kind: "step", move: after.move ?? 0 });
  }
  if (type === "keyboard" && key === "0") {
    return statusSentence({ kind: "no_combat" });
  }
  if (type === "keyboard" && (key === "1" || key === "2")) {
    if (key === "2" && after.isMage) return statusSentence({ kind: "burst_armed" });
    return statusSentence({
      kind: "armed",
      ability: key === "1" ? after.basicName : after.ultimateName,
      target: after.targetName,
    });
  }
  if (
    (type === "keyboard" && key === "escape") ||
    (type === "battlefield_pointer" && command.button === "secondary")
  ) {
    return statusSentence({
      kind: "cleared",
      keysOff: type === "keyboard" && source === "field",
    });
  }
  if (
    before !== null &&
    after.targetAction > 0 &&
    after.targetAction !== before.targetAction &&
    after.targetName !== null
  ) {
    const pair = after.pairs[after.targetAction] ?? [];
    return statusSentence({
      kind: "target",
      target: after.targetName,
      basic: after.basicName,
      ultimate: after.ultimateName,
      auto: after.lane === "basic",
      ultimateAllowed: !after.isMage && pair[1] === true,
    });
  }
  return null;
}

/**
 * Clear frames and presentation after a product-identity mismatch.
 * Mark the page offline and in need of resynchronization, disconnect replay playback,
 * and store the error notice so no incompatible scientific surface remains active.
 *
 * @param {ProductIdentityMismatchError} error
 */
function failClosedProductIdentity(error) {
  state.frame = null;
  state.timeline = null;
  clearPresentationAuthority("product_identity_mismatch");
  state.offline = true;
  state.resyncRequired = true;
  replayPlayback.setConnected(false);
  setNotice(error.message, "error");
}

/**
 * Convert a value to a display string with underscores replaced by spaces
 * and word starts capitalized. This is a label formatter, not protocol normalization.
 *
 * @param {unknown} value
 */
function humanize(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

/**
 * Render match title, team summaries, and task selection from a presentation,
 * then the HUD's party frames.
 * Delegate to the shared match-summary renderer with this page DOM targets and
 * the current switch state, beatView("end") result (scores wait for the Fall
 * beat through beat holds; the scorebug is one of the choreographer's
 * beatHoldRoots) and the scene's public wave clock (wave pips), then
 * hud.render with hudSourcesFromFrame (the HUD's
 * one frame reader; Agent POV: unseen champions read "Unseen"). Both run before
 * the choreographer plans the tick, so its beat holds find their nodes.
 *
 * @param {unknown} presentation
 */
function renderInstalledMatch(presentation) {
  const beat = beatView(presentation, "end", { visualFilters: visualFilterState });
  const hudSources = hudSourcesFromFrame(presentation);
  renderMatchSummary(
    {
      root: elements.matchScoreboard,
      task: elements.matchTask,
      teams: [elements.matchTeamA, elements.matchTeamB],
    },
    presentation,
    {
      filters: visualFilterState,
      beat,
      waves: hudSources?.waves ?? null,
    },
  );
  // The HUD is drawn before the choreographer plans the tick, so its beat holds
  // find the "before" nodes. Falls name no killer here (the plan is not built
  // yet); Oracle View falls back to the match summary's credited deaths.
  hud.render(hudSources, {
    filters: visualFilterState,
    beat,
    falls: [],
  });
}

/**
 * Work out the match moment cards for one installed tick (looks/heralds.js
 * records with part "lineup" or "verdict"). Pure: no DOM, storage or timers.
 *
 * - Lineup card: on tick 0 in every product. Its medallions come only from the
 *   audience's authorized arena scene (in Agent POV only the champions the team
 *   sees), and its rules are the public First To and draw tick, plus the Red
 *   Zone rule when the scene's map has a Red Zone.
 * - Verdict card: on the final tick, in the Replay Viewer and at game over in
 *   the BattleClient and DevClient (the herald look gives nothing for another
 *   product). It carries the public scores, the
 *   winner and why the match ended ("threshold" for a win, "tie" when both
 *   teams reached First To with equal scores, else "horizon"), each team's
 *   controller line from the one controller model (controllerRosterFromFrame:
 *   the full line, and, only for a team with two or more policies, the short
 *   "You + 4 Policies" line for when it does not fit; a single policy keeps
 *   its own name), the team whose home end is on the left (its recorded spawn side) as
 *   the left column, and the Honors of the frame's own `match_honors` host
 *   fact (heralds.verdictHonours; Agent POV lists every champion too, because
 *   Honors come only once the game is over and nothing is left to hide then,
 *   viewer/match_summary.py). options.tucked draws its herald plate (Show
 *   Arena).
 * Any other tick gives no card here: the countdown chip and the "Match Point ·
 * Needs 1" chip were removed (owner, 4 Oct 2026); moment cards, 30 Ticks Left
 * included, come from the choreography plan's herald rows.
 *
 * Agent POV: nothing here reads a champion the POV team cannot see. Champions
 * come only from authorizedPresentationSceneView; the match facts are the
 * public scoreboard facts both audiences already see beside the arena (map
 * name, scores, First To, draw tick, tick and result, through matchSummaryView).
 * Scores are the tick's end scores; the scorebug shows the same numbers once
 * the tick's Fall beat lands.
 *
 * @param {unknown} presentation The installed authorized presentation, or
 *   anything else (which gives no cards).
 * @param {import("./looks/look-support.js").Rectangle} bounds
 *   The arena's visible rectangle in CSS pixels (the choreography surface's
 *   viewportBounds), used to place the cards.
 * @param {unknown} _plan This tick's choreography plan (choreographer.plan), or
 *   null; not read since the state chip was removed (kept so callers stay
 *   unchanged).
 * @param {Readonly<Record<string, boolean>>} filters The page's switch state;
 *   match_moments off draws nothing and moment_full_cards off draws the Lineup
 *   as a one-line band and the Verdict as its herald plate.
 * @param {{
 *   combatConfiguration?: unknown,
 *   tucked?: boolean,
 *   chrome?: ReadonlyArray<{left: number, top: number, right: number, bottom: number}>,
 * }} [options] Live play's transport combat_configuration (its manual slots
 *   and slot labels name the controllers; null in the Replay Viewer), whether
 *   the player tucked the Verdict with Show Arena, and the page chrome over
 *   the arena in the bounds' units (the surface's chromeRects, such as the
 *   key stack), which the Verdict's plate keeps clear of. Default {}.
 * @returns {ReadonlyArray<import("./looks/look-support.js").LookRecord>} The
 *   records to draw, in order; empty when no card applies.
 */
function matchMomentCardRecords(presentation, bounds, _plan, filters, options = {}) {
  if (!isAuthorizedPresentationFrame(presentation)) return [];
  const audience = authorizedPresentationAudience(presentation);
  const scene = authorizedPresentationSceneView(presentation);
  const summary = matchSummaryView(presentation);
  if (audience === null || scene === null || summary === null) return [];
  const tick = summary.tick?.current ?? null;
  const pointsToWin = Number.isInteger(summary.threshold) ? summary.threshold : null;
  const tickLimit = summary.tick?.limit ?? null;
  if (tick === 0) {
    return heralds.records({
      part: "lineup",
      scene,
      bounds,
      filters,
      mapTitle: summary.map ?? "",
      mapLabel: summary.taskMode === 1 ? "Team Deathmatch" : "Combat Diagnostic",
      pointsToWin,
      tickLimit,
      redZoneDoubles:
        summary.taskMode === 1 &&
        isRecord(scene.map) &&
        scene.map.red_zone !== null &&
        scene.map.red_zone !== undefined,
    });
  }
  const teams = /** @type {ReadonlyArray<Record<string, any>>} */ (summary.teams);
  const scoreA = teams[0]?.scoreValue;
  const scoreB = teams[1]?.scoreValue;
  if (pointsToWin === null || !Number.isInteger(scoreA) || !Number.isInteger(scoreB)) {
    return [];
  }
  const winnerIndex = teams.findIndex((team) => team.result === "VICTORY");
  const finished = teams.some((team) => team.result !== null);
  if (finished) {
    const winnerTeamId = winnerIndex === 0 ? 1 : winnerIndex === 1 ? 2 : null;
    const roster = controllerRosterFromFrame(
      presentation,
      options.combatConfiguration ?? null,
    );
    // A team's line names its controllers. The short line ("4 Policies")
    // stands in only when the team has two or more policies; a single policy
    // keeps its own name, shrunk or cut by the card (owner, 4 Oct 2026: never
    // "1 Policy").
    /** @param {1 | 2} teamId */
    const lineOf = (teamId) => {
      const team = rosterTeam(roster, teamId);
      if (team === null || team.controllers.length === 0) {
        return { name: String(teams[teamId - 1]?.name ?? "") };
      }
      const policies = team.controllers.filter(
        (controller) => !controller.you && !controller.manual,
      ).length;
      const name = controllerLineParts(team).join(" + ");
      return policies >= 2 ? { name, short: collapsedControllerLine(team) } : { name };
    };
    const left = teams.find((team) => team.side === "left");
    return heralds.records({
      part: "verdict",
      product: installedProductKind(),
      audience,
      bounds,
      filters,
      leftTeamId: left?.teamId === 2 ? 2 : 1,
      tucked: options.tucked === true,
      chrome: options.chrome ?? [],
      verdict: {
        scores: { 1: scoreA, 2: scoreB },
        winnerTeamId,
        tick,
        pointsToWin,
        basis:
          winnerTeamId !== null
            ? "threshold"
            : scoreA >= pointsToWin && scoreB >= pointsToWin
              ? "tie"
              : "horizon",
        teams: { 1: lineOf(1), 2: lineOf(2) },
        honours: heralds.verdictHonours(
          isRecord(presentation) ? presentation.match_honors : null,
        ),
      },
    });
  }
  return [];
}

/**
 * Redraw the match moment cards (Lineup card or Verdict card) for
 * the installed tick in the renderer's herald layer, above every body. Each call
 * first removes the page's previous card group, so every render (and Oracle
 * View to Agent POV to Oracle View) starts fresh. With no presentation, no scene
 * on screen or no card to draw, only the removal happens.
 *
 * Side effects: replaces one `<g data-match-moment-cards>` child of the herald
 * layer (export-safe SVG from heralds.build, pointer-events none so clicks reach
 * the arena) and updates matchMomentCardsGroup. When the Verdict card (or its
 * tucked band plate) is drawn, every painter herald card in the herald layer
 * (a plate: `[data-herald-moments]`) gets `visibility="hidden"`, so it never
 * shares the top slot with the Verdict; on any other tick that attribute is
 * removed again. The full Verdict also gets its two working
 * buttons (heralds.buildVerdictActions) in the battlefield shell, replaced on
 * every call (verdictActions): Show Arena tucks it (verdictTucked) and Match
 * Summary opens the sheet (openMatchSummarySheet). A tick without a Verdict
 * closes an open sheet. No timers or storage; the card holds its end state
 * with no entry animation. Agent POV: see matchMomentCardRecords.
 *
 * @param {unknown} presentation The installed authorized presentation, or null
 *   to clear.
 * @returns {void}
 */
function renderMatchMomentCards(presentation) {
  matchMomentCardsGroup?.remove();
  matchMomentCardsGroup = null;
  verdictActions?.remove();
  verdictActions = null;
  if (presentation === null || presentation === undefined) {
    elements.arenaMatchSummaryButton.hidden = true;
    closeMatchSummarySheet();
    return;
  }
  const surface = battlefieldRenderer.choreographySurface();
  if (surface === null) return;
  const records = matchMomentCardRecords(
    presentation,
    surface.viewportBounds,
    choreographer.plan,
    visualFilterState,
    {
      combatConfiguration:
        installedPresentationAuthority()?.transport?.combat_configuration ?? null,
      tucked: verdictTucked,
      chrome: surface.chromeRects,
    },
  );
  // The Verdict card owns the top slot on the final tick: a herald card the
  // painter drew this tick (a plate; each names its moments in
  // data-herald-moments) is hidden, never stacked.
  const verdictShown = records.some(
    (row) => row.kind === "herald-verdict" || row.kind === "herald-verdict-plate",
  );
  // The arena's Match Summary key opens the sheet again while the Verdict is
  // folded into its plate (Show Arena, or Full Cards off), which has no
  // buttons of its own (owner, 4 Oct 2026).
  const verdictFolded = records.some((row) => row.kind === "herald-verdict-plate");
  elements.arenaMatchSummaryButton.hidden = !verdictFolded;
  if (!verdictShown && !verdictFolded) closeMatchSummarySheet();
  const painterCards = [
    ...surface.heraldLayer.querySelectorAll("[data-herald-moments]"),
  ];
  for (const card of painterCards) {
    if (verdictShown) card.setAttribute("visibility", "hidden");
    else card.removeAttribute("visibility");
  }
  if (records.length === 0) return;
  const doc = surface.ownerDocument;
  const group = /** @type {SVGGElement} */ (
    doc.createElementNS("http://www.w3.org/2000/svg", "g")
  );
  group.setAttribute(
    "data-match-moment-cards",
    records.map((row) => row.kind).join(" "),
  );
  group.setAttribute("pointer-events", "none");
  group.append(...heralds.build(records, doc));
  surface.heraldLayer.append(group);
  matchMomentCardsGroup = group;
  const full = records.find((row) => row.kind === "herald-verdict");
  if (full !== undefined) {
    verdictActions = heralds.buildVerdictActions(document, full, onVerdictAction);
    elements.battlefieldShell.append(verdictActions);
  }
}

/**
 * Do what a Verdict card button says. "show_arena" tucks the Verdict into its
 * band plate for the rest of this page session (verdictTucked) and redraws
 * the cards; "summary" opens the Match Summary sheet. Side effects: page
 * state and DOM only.
 *
 * @param {"summary" | "show_arena"} id
 * @returns {void}
 */
function onVerdictAction(id) {
  if (id === "show_arena") {
    verdictTucked = true;
    renderMatchMomentCards(installedPresentationAuthority()?.presentation ?? null);
    return;
  }
  openMatchSummarySheet();
}

// The arena's Match Summary key: opens the sheet, or closes the open one.
elements.arenaMatchSummaryButton.addEventListener("click", () => {
  if (matchSummarySheet !== null) closeMatchSummarySheet();
  else openMatchSummarySheet();
});

/**
 * Open the Match Summary sheet of the installed frame below the arena (never
 * over it; see matchSummarySheet). Keyboard focus moves to its Close button,
 * unless the arena holds keys (a mouse press with keys on keeps them on). The sheet reads only the frame's own `match_honors` host fact (every
 * champion in both views: the game is over); shown IDs come from
 * authorizedPresentationAgentDisplayId, and the team whose home end is on the
 * left is listed first, as on the Verdict card. Replaces an open sheet. Side
 * effects: DOM and focus only.
 *
 * @returns {void}
 */
function openMatchSummarySheet() {
  closeMatchSummarySheet();
  const presentation = installedPresentationAuthority()?.presentation ?? null;
  const left = matchSummaryView(presentation)?.teams.find(
    (/** @type {Record<string, any>} */ team) => team.side === "left",
  );
  const sheet = heralds.buildMatchSummarySheet(
    document,
    isRecord(presentation) ? presentation.match_honors : null,
    {
      displayIdOf: (publicAgentId) =>
        authorizedPresentationAgentDisplayId(presentation, publicAgentId),
      leftTeamId: left?.teamId === 2 ? 2 : 1,
      onClose: closeMatchSummarySheet,
    },
  );
  const home = isReplayMode()
    ? document.getElementById("replay-timeline")
    : elements.commandDeck;
  if (home instanceof HTMLElement) home.append(sheet);
  else elements.fallenFeed.before(sheet);
  matchSummarySheet = sheet;
  elements.arenaMatchSummaryButton.setAttribute("aria-pressed", "true");
  // A mouse press with keys on keeps them on (keepKeysOnForTopBarPress);
  // otherwise (keyboard use) focus moves to the sheet's Close.
  if (document.activeElement !== elements.battlefield) {
    sheet.querySelector("button")?.focus();
  }
}

/**
 * Close the Match Summary sheet when one is open. When keyboard focus was
 * inside the sheet (its Close was reached with the keyboard), focus goes back
 * to the arena in live play, so keys are on again for R, ? and G; in the
 * Replay Viewer it is left to the browser. Side effects: DOM and focus only.
 *
 * @returns {void}
 */
function closeMatchSummarySheet() {
  const hadFocus = matchSummarySheet?.contains(document.activeElement) === true;
  matchSummarySheet?.remove();
  matchSummarySheet = null;
  elements.arenaMatchSummaryButton.setAttribute("aria-pressed", "false");
  if (hadFocus && !isReplayMode()) elements.battlefield.focus({ preventScroll: true });
}

/**
 * Publish choreography status as root DOM data attributes for diagnostics.
 * Expose motion, pause, render policy, and submission blocking from the supplied
 * snapshot without adding motion-editing controls to the product.
 *
 * @param {ReturnType<CombatChoreographer["snapshot"]>} presentation
 */
function exposePresentationState(presentation) {
  document.documentElement.dataset.motionMode = presentation.motionMode;
  document.documentElement.dataset.motionPaused = String(presentation.paused);
  if (presentation.renderPolicy === null) {
    document.documentElement.removeAttribute("data-render-policy");
  } else {
    document.documentElement.dataset.renderPolicy = presentation.renderPolicy;
  }
  document.documentElement.dataset.submissionBlocked = String(
    presentation.submissionBlocked,
  );
}

/**
 * Label the selected Oracle replay agent as a reference rather than a live target.
 * Resolve the selected key from the installed authorized scene, update its SVG
 * accessible label, and attach its non-inspectable Champion Card
 * (agentHoverDescriptor). Other views do nothing.
 */
function applyReplayReferenceSemantics() {
  if (!isReplayMode()) {
    return;
  }
  const presentation = state.presentation;
  if (authorizedPresentationAudience(presentation) !== "researcher") {
    return;
  }
  const scene = installedAuthorizedPresentationSceneView(presentation);
  const selectedKey = isRecord(scene?.selection)
    ? scene.selection.inspection_owner_presentation_key
    : null;
  if (typeof selectedKey !== "string") {
    return;
  }
  const reference = elements.battlefield.querySelector(
    `.agent[data-presentation-key="${CSS.escape(selectedKey)}"]`,
  );
  if (!(reference instanceof Element)) {
    return;
  }
  const agent = asArray(scene?.agents).find(
    (candidate) => isRecord(candidate) && candidate.presentation_key === selectedKey,
  );
  if (!isRecord(agent) || typeof agent.public_agent_id !== "string") {
    return;
  }
  const publicAgentId = agent.public_agent_id;
  const ariaLabel = reference.getAttribute("aria-label") ?? `Agent ID ${publicAgentId}`;
  reference.setAttribute(
    "aria-label",
    ariaLabel.replace(/selected target/giu, "Reference"),
  );
  registerTooltipOwner(reference, agentHoverDescriptor(agent, "researcher"), {
    inspectable: false,
  });
}

/**
 * Return the hover card descriptor of one arena body.
 *
 * In the Replay Viewer it is the Champion Card (tooltip.js
 * createChampionCardDescriptor of shell.js championCardFacts, read from the
 * installed presentation's own scene), the same card the HUD frames and live
 * play show, in Oracle View and Agent POV alike. In live play it stays the
 * older agent card (explanations.js explainAgent): its look is hidden there,
 * because #hcard shows the Champion Card, and it keeps the body's accessible
 * description. Reads `state.presentation`; pure otherwise.
 *
 * @param {Record<string, any>} agent The body's row in the installed scene.
 * @param {string} audience "researcher" or "agent_pov".
 * @returns {ReturnType<typeof explainAgent>}
 */
function agentHoverDescriptor(agent, audience) {
  if (isReplayMode()) {
    const facts = championCardFacts({
      presentation: state.presentation,
      combatConfiguration: null,
      subject: {
        presentationKey:
          typeof agent.presentation_key === "string" ? agent.presentation_key : null,
        slot: null,
        publicAgentId: null,
        classId: null,
      },
    });
    if (facts !== null) {
      return createChampionCardDescriptor(
        facts,
        `champion:body:${String(agent.public_agent_id)}`,
      );
    }
  }
  return explainAgent(agent, { audience });
}

/**
 * Resolve an opaque scene key to the single action allowed in the current view.
 * Return a frozen local-inspection, replay-selection, replay-POV, or live-control
 * effect, or null when unavailable. Require coherent current authority; DOM role
 * attributes and raw slot values cannot choose the effect.
 *
 * @param {unknown} presentationKey
 * @returns {Readonly<Record<string, any>> | null}
 */
function authorizedAgentActivation(presentationKey) {
  const presentation = state.presentation;
  if (
    typeof presentationKey !== "string" ||
    !isAuthorizedPresentationFrame(presentation) ||
    !installedAuthorityIsCoherent() ||
    state.busy ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline
  ) {
    return null;
  }
  const agent = authorizedAgentForPresentationKey(presentation, presentationKey);
  if (agent === null) {
    return null;
  }
  const audience = authorizedPresentationAudience(presentation);
  const battlefieldInspectionState =
    authorizedPresentationInspectionState(presentation);
  const researcherInspectionState =
    authorizedPresentationResearcherInspectionState(presentation);
  const replayPovSwitch = isReplayMode() && audience === "agent_pov";
  if (audience === "agent_pov" && agent.life_state === "corpse") {
    return Object.freeze({
      effect: "local_inspection",
      presentationKey,
      agent,
      audience,
    });
  }
  if (replayPovSwitch) {
    return Object.freeze({
      effect: "replay_pov_switch",
      presentationKey,
      agent,
      audience,
    });
  }
  if (
    audience === "agent_pov" &&
    (battlefieldInspectionState.state_kind === "live_scripted" ||
      recordingScientificControlsFenced())
  ) {
    return Object.freeze({
      effect: "local_inspection",
      presentationKey,
      agent,
      audience,
    });
  }
  if (audience === "agent_pov") {
    const commandSlot = authorizedOracleCommandSlotForPublicAgentId(
      presentation,
      agent.public_agent_id,
    );
    return ["live_editable", "live_policy"].includes(
      researcherInspectionState.state_kind,
    ) && Number.isInteger(commandSlot)
      ? Object.freeze({
          effect: "live_control",
          presentationKey,
          commandSlot,
          agent,
          audience,
        })
      : null;
  }
  if (battlefieldInspectionState.state_kind === "live_scripted") {
    return Object.freeze({
      effect: "local_inspection",
      presentationKey,
      agent,
      audience,
    });
  }
  if (audience !== "researcher") {
    return null;
  }
  const commandSlot = authorizedOracleCommandSlotForPresentationKey(
    presentation,
    presentationKey,
  );
  if (!Number.isInteger(commandSlot)) {
    return null;
  }
  if (isReplayMode()) {
    return Object.freeze({
      effect: "replay_select",
      presentationKey,
      commandSlot,
      agent,
      audience,
    });
  }
  return ["live_editable", "live_policy"].includes(researcherInspectionState.state_kind)
    ? Object.freeze({
        effect: "live_control",
        presentationKey,
        commandSlot,
        agent,
        audience,
      })
    : null;
}

/**
 * Resolve an event target to an authorized SVG agent activation.
 * Reject non-elements, non-SVG agent owners, and nested tooltip controls. Return
 * the effect plus its element/key, or null; do not perform the activation yet.
 *
 * @param {unknown} target
 * @returns {(NonNullable<ReturnType<typeof authorizedAgentActivation>> & {presentationKey: string, element: SVGElement}) | null}
 */
function authorizedAgentActivationFromTarget(target) {
  if (!(target instanceof Element)) {
    return null;
  }
  const element = target.closest(".agent[data-presentation-key]");
  if (!(element instanceof SVGElement)) {
    return null;
  }
  const nestedTooltipOwner = target.closest("[data-tooltip-owner]");
  if (nestedTooltipOwner !== null && nestedTooltipOwner !== element) {
    return null;
  }
  const activation = authorizedAgentActivation(element.dataset.presentationKey);
  return activation === null
    ? null
    : Object.freeze({
        ...activation,
        presentationKey: String(activation.presentationKey),
        element,
      });
}

/**
 * Assign each painted agent its current accessible action, focus state, and tooltip
 * (agentHoverDescriptor: the Champion Card in the Replay Viewer). Use only the
 * installed authorized scene and activation gate. Unavailable agents remain
 * images without keyboard activation; active agents become labeled buttons.
 */
function installAuthorizedAgentActivation() {
  const presentation = state.presentation;
  const scene = installedAuthorizedPresentationSceneView(presentation);
  const agents = asArray(scene?.agents);
  const audience = authorizedPresentationAudience(presentation);
  for (const element of elements.battlefield.querySelectorAll(
    ".agent[data-presentation-key]",
  )) {
    const presentationKey = element.dataset.presentationKey;
    const agent = agents.find(
      (candidate) =>
        isRecord(candidate) && candidate.presentation_key === presentationKey,
    );
    if (!isRecord(agent)) {
      element.setAttribute("role", "img");
      element.setAttribute("tabindex", "-1");
      continue;
    }
    const activation = authorizedAgentActivation(presentationKey);
    if (activation === null) {
      element.setAttribute("role", "img");
      element.setAttribute("tabindex", "-1");
    } else {
      element.setAttribute("tabindex", "0");
      element.setAttribute("role", "button");
      const identity = agentIdentity(agent.public_agent_id);
      element.setAttribute(
        "aria-label",
        activation.effect === "live_control"
          ? `${identity}. Click to give it orders or look at it.`
          : activation.effect === "replay_pov_switch"
            ? `${identity}. Click to watch the replay from its team's Agent POV.`
            : `${identity}. Click to look at it.`,
      );
    }
    const hover = agentHoverDescriptor(agent, audience ?? "agent_pov");
    registerTooltipOwner(element, hover, { inspectable: false });
  }
}

/**
 * Apply the currently authorized effect for one opaque agent key.
 * Return false when unavailable, otherwise update local inspection or start the
 * proper live/replay request. pointerOriginated defaults to false; pointer control
 * activation hands focus to the battlefield, while keyboard roster activation
 * keeps its native focus route. Server-owned scientific state waits for its reply.
 *
 * @param {string} presentationKey
 * @param {Readonly<{pointerOriginated?: boolean}>} context
 * @returns {boolean}
 */
function activateAuthorizedAgent(presentationKey, { pointerOriginated = false } = {}) {
  const activation = authorizedAgentActivation(presentationKey);
  if (activation === null) {
    return false;
  }
  if (activation.effect === "local_inspection") {
    const localSelectionInstalled = setLocalInspectedPresentationKey(presentationKey);
    if (!localSelectionInstalled && activation.agent.life_state !== "corpse") {
      return false;
    }
    activatedAgentPublicId = String(activation.agent.public_agent_id);
    render();
    openAgentDetails();
    return true;
  }
  if (activation.effect === "replay_pov_switch") {
    stageReplayRecipientActivation(String(activation.agent.public_agent_id));
    void dispatchReplayCommand({
      command_type: "set_pov_actor",
      presentation_key: activation.presentationKey,
    });
  } else if (activation.effect === "replay_select") {
    activatedAgentPublicId = String(activation.agent.public_agent_id);
    openAgentDetails();
    void dispatchReplayCommand({
      command_type: "select_agent",
      selected_global_slot: activation.commandSlot,
    });
  } else if (activation.effect === "live_control") {
    activatedAgentPublicId = String(activation.agent.public_agent_id);
    openAgentDetails();
    const controlCommand = {
      command_type: "roster_selection",
      role: "control",
      global_slot: activation.commandSlot,
    };
    if (pointerOriginated) {
      dispatchCommandFromDraftControl(controlCommand);
    } else {
      void dispatchCommand(controlCommand);
    }
  }
  return true;
}

/**
 * Pause replay for a presentation error while suppressing recursive state renders.
 * Always restore the render-callback flag, including when pause throws.
 *
 * @param {string} reason
 */
function pauseReplayAfterPresentationFailure(reason) {
  suppressPlaybackStateRender = true;
  try {
    replayPlayback.pause(reason);
  } finally {
    suppressPlaybackStateRender = false;
  }
}

/** @type {number | null} */
let livePollTimer = null;
let livePollInFlight = false;
/**
 * Until this performance.now() time (ms), reads come at the fast 80 ms
 * cadence: hurryLivePoll sets it for a short while after the page's own turn
 * command, so the policies' answer shows soon after the server has it.
 */
let livePollFastUntil = 0;

/** How long (ms) after the page's own command the fast cadence lasts. */
const LIVE_POLL_FAST_WINDOW_MS = 3000;

/**
 * Schedule one conditional read. The server owns every simulator turn. The
 * read comes after 1000 ms when nothing is happening, 250 ms while the server
 * runs or policy work is out, and 80 ms while the page's own Run waits for
 * its turn (so the next tick follows the last glide closely) or for 3 s after
 * the page's own command left work out (hurryLivePoll). Does nothing in the
 * Replay Viewer, offline, during a resync or after Exit.
 */
function scheduleLivePoll() {
  if (
    isReplayMode() ||
    !state.frame ||
    state.offline ||
    state.resyncRequired ||
    state.shuttingDown
  ) {
    if (livePollTimer !== null) window.clearTimeout(livePollTimer);
    livePollTimer = null;
    return;
  }
  if (livePollTimer !== null || livePollInFlight) return;
  livePollTimer = window.setTimeout(
    () => {
      void pollLiveMatch();
    },
    state.pageRun || performance.now() < livePollFastUntil
      ? 80
      : state.frame.run_state === "running" || state.frame.system_operation
        ? 250
        : 1000,
  );
}

/**
 * Read the match again soon after a command left work out on the server.
 *
 * A command can be answered while the policies still choose (the frame shows
 * `system_operation`) or while the server runs. The idle read scheduled
 * before the command would then wait up to 1000 ms, so "Policies Choosing"
 * stayed on screen after the policies had already answered (measured 2 Oct
 * 2026: up to 0.9 s after Play One Turn in Policies Only). This drops that
 * idle timer and reads at the fast 80 ms cadence for the next 3 s
 * (LIVE_POLL_FAST_WINDOW_MS; then scheduleLivePoll's 250 ms busy cadence),
 * so the new tick shows within about 0.1 s of the server's answer. Does
 * nothing in the Replay Viewer or when nothing is out on the server; while a
 * read is out it only opens the fast window (that read reschedules itself
 * when it ends).
 *
 * @returns {void} Changes only the live poll timer and its fast window.
 */
function hurryLivePoll() {
  if (isReplayMode()) return;
  if (!state.frame?.system_operation && state.frame?.run_state !== "running") return;
  livePollFastUntil = performance.now() + LIVE_POLL_FAST_WINDOW_MS;
  if (livePollInFlight) return;
  if (livePollTimer !== null) window.clearTimeout(livePollTimer);
  livePollTimer = null;
  scheduleLivePoll();
}

/**
 * Start or stop the page's own Run in Policies Only (the deck's Run and Pause
 * button calls it; see tactics-turn.js).
 *
 * Run is paced by the page, not by the server: the page sends Play One Turn
 * (the Enter key command, exactly what the Play One Turn button sends) once
 * the current tick's bodies have landed, lets the policies choose while that
 * tick's last beats play, holds the next tick back until the current one has
 * settled (choreographer.whenSettled) and then shows it at once (see
 * runLiveTurns). So every tick plays to its end and flows straight into the
 * next; none is skipped or cut short, as could happen when the server ran on
 * its own and the page showed only the newest tick at each poll, and no
 * paused-looking still sits between ticks. The server's run_state stays
 * "paused" throughout.
 *
 * Pause clears state.pageRun at once (the button reads Run again), but the
 * tick on screen and a turn already out (Run sends the next turn once the
 * bodies of the tick on screen have landed) still land and play to their end;
 * only then does the loop stop and the message line say "Paused after the
 * current turn.", as on the vision page. Run pressed again before that turn has landed keeps the same loop
 * going; it never starts a second one, so no turn is sent twice. The loop
 * also stops at game over, at a fresh-launch stop, when the page goes
 * offline, needs a resync or shuts down, when a Manual slot appears, or when
 * a turn does not advance the tick. Side effects: page state (pageRun),
 * repaints and live commands through dispatchCommand. Live play only; does
 * nothing in the Replay Viewer. Returns undefined.
 */
function toggleLiveRun() {
  if (isReplayMode()) return;
  if (state.pageRun) {
    state.pageRun = false;
    render();
    return;
  }
  state.pageRun = true;
  render();
  if (!state.pageRunLoop) void runLiveTurns();
}

/**
 * Return whether the page's own Run owns Lock In right now, so a Space,
 * Enter or N from the player must be ignored: true while runLiveTurns is
 * active (from Run until its last turn has landed and settled, so also right
 * after Pause) and the command is a keyboard command whose key is a Lock In
 * key (isLockInKeyName: Enter or N in any case, or Space " "), with any
 * modifiers. The vision page ignores Lock In while Run plays; without this
 * check, a Space or N pressed during a glide (All Act) would cut the glide
 * short and send a second turn. Pure; reads state.pageRunLoop and the
 * command (and calls isLockInKeyName). Live play only (the loop never runs
 * in the Replay Viewer).
 *
 * @param {Readonly<Record<string, unknown>>} command A command about to be sent.
 * @returns {boolean}
 */
function pageRunOwnsLockIn(command) {
  return (
    state.pageRunLoop &&
    command.command_type === "keyboard" &&
    isLockInKeyName(command.key)
  );
}

/**
 * Return whether the page's own Run may send the next Play One Turn now: a
 * live, non-terminal Policies Only frame with no stop, no connection trouble
 * and no policy work already out. Pure.
 *
 * @param {{ignorePause?: boolean}} [options] ignorePause (default false) skips
 *   the state.pageRun test. The loop uses it while it waits for a turn that is
 *   already out: Pause stops after the current turn, so that wait goes on,
 *   but game over, a stop or connection trouble still end it.
 * @returns {boolean}
 */
function liveRunMayContinue({ ignorePause = false } = {}) {
  const frame = state.frame;
  return (
    (ignorePause || state.pageRun) &&
    !isReplayMode() &&
    frame !== null &&
    !isTerminal(frame) &&
    !frame.fresh_launch_required &&
    Array.isArray(frame.combat_configuration?.manual_slots) &&
    frame.combat_configuration.manual_slots.length === 0 &&
    !state.offline &&
    !state.resyncRequired &&
    !state.shuttingDown
  );
}

/**
 * How long the page's own Run waits, after sending Play One Turn, for the
 * live poll to show the turn's policy work or its next tick before deciding
 * the turn did not start (milliseconds). Above the 250 ms poll interval.
 */
const RUN_TURN_GRACE_MS = 1200;

/**
 * Wait about one animation frame (or 40 ms when the page is hidden). Resolves
 * without a value; no other effect.
 *
 * @returns {Promise<void>}
 */
function nextRunBeat() {
  return new Promise((resolve) => {
    if (document.hidden) window.setTimeout(resolve, 40);
    else window.requestAnimationFrame(() => resolve());
  });
}

/**
 * True while the page's own Run holds the next tick back: from the moment Run
 * sends the next Play One Turn (while the current tick still plays) until the
 * current tick's choreography settles. A live poll that reads a new snapshot
 * while it is true waits for that settle before installing it
 * (pollLiveMatch), so the next tick can never cut the current one short and
 * starts in the same moment the current one ends. runLiveTurns clears it and
 * polls at once when the tick settles (unless a poll is already waiting).
 * Only runLiveTurns writes it.
 */
let liveRunHoldsNextTick = false;

/**
 * Wait, one animation frame at a time, until the current tick's bodies have
 * landed (the choreographer releases submission, about 450 ms into a tick in
 * normal motion; at once in reduced or off motion) or until Run may no longer
 * continue. Resolves without a value; no other effect.
 *
 * @returns {Promise<void>}
 */
async function whenRunTickReleased() {
  while (liveRunMayContinue() && choreographer.snapshot().submissionBlocked === true) {
    await nextRunBeat();
  }
}

/**
 * The loop behind the page's own Run (see toggleLiveRun). One turn at a time,
 * overlapped so the ticks flow into each other: wait until no request or
 * policy work is out and the current tick's bodies have landed
 * (whenRunTickReleased), send Play One Turn (quietly: the message line keeps
 * "Running."; see dispatchCommand) without cutting the current tick short
 * (keepPresentation), so the policies choose while its last beats play; hold
 * the next tick back (liveRunHoldsNextTick) until the current tick's
 * choreography settles, then poll at once, wait until the next tick is
 * installed (this wait goes on after Pause, because Pause stops after the turn
 * already out) and repeat. So every tick plays to its end and the next one
 * starts straight away, with no still, paused-looking frame between them.
 * Ends with state.pageRun false, state.pageRunLoop false,
 * liveRunHoldsNextTick false and, when the player paused, the message line
 * "Paused after the current turn.". Never throws; a failed command reports
 * through dispatchCommand and stops the loop.
 *
 * @returns {Promise<void>}
 */
async function runLiveTurns() {
  state.pageRunLoop = true;
  try {
    while (liveRunMayContinue()) {
      if (state.busy || state.frame?.system_operation) {
        await nextRunBeat();
        continue;
      }
      await whenRunTickReleased();
      if (!liveRunMayContinue()) {
        // Paused (or the game ended) while this tick played: let it finish.
        await choreographer.whenSettled();
        break;
      }
      if (state.busy || state.frame?.system_operation) continue;
      const tick = Number(state.frame?.simulator_step_count);
      const episode = state.frame?.episode_id;
      liveRunHoldsNextTick = true;
      try {
        await dispatchCommand(keyboardCommand("Enter"), {
          source: "control",
          quiet: true,
          keepPresentation: true,
        });
        // The current tick plays to its end while the policies choose.
        await choreographer.whenSettled();
      } finally {
        liveRunHoldsNextTick = false;
      }
      if (livePollTimer !== null) {
        window.clearTimeout(livePollTimer);
        livePollTimer = null;
      }
      if (!livePollInFlight) void pollLiveMatch();
      // The policies choose on the server; the live poll installs the result.
      // A short grace covers a frame that has not yet shown the policy work.
      const heldUntil = performance.now();
      while (
        liveRunMayContinue({ ignorePause: true }) &&
        state.frame?.episode_id === episode &&
        Number(state.frame?.simulator_step_count) === tick &&
        (state.busy ||
          state.frame?.system_operation ||
          performance.now() - heldUntil < RUN_TURN_GRACE_MS)
      ) {
        await nextRunBeat();
      }
      if (
        state.frame?.episode_id !== episode ||
        Number(state.frame?.simulator_step_count) === tick
      ) {
        // The turn did not advance (refused, cancelled or a new game).
        break;
      }
      if (!state.pageRun) {
        // Paused: the turn already out has landed; let it play to its end.
        await choreographer.whenSettled();
      }
    }
  } finally {
    state.pageRunLoop = false;
    liveRunHoldsNextTick = false;
    const paused = !state.pageRun;
    state.pageRun = false;
    if (paused && !isReplayMode()) clearNotice(statusSentence({ kind: "paused" }));
    render();
  }
}

/**
 * Read a changed live snapshot without blocking a Pause click during network I/O.
 * Discard reads overtaken by a command or another installed frame. The existing
 * installation owner checks identity and disclosure before painting anything.
 * While the page's own Run holds the next tick (liveRunHoldsNextTick), a new
 * snapshot waits for the tick on screen to settle before it is installed.
 */
async function pollLiveMatch() {
  livePollTimer = null;
  if (
    livePollInFlight ||
    state.busy ||
    isReplayMode() ||
    !state.frame ||
    state.offline ||
    state.resyncRequired ||
    state.shuttingDown
  ) {
    scheduleLivePoll();
    return;
  }
  livePollInFlight = true;
  const previousFrame = state.frame;
  const previousAuthority = state.authority;
  try {
    const joined = await getCurrentLiveSnapshot(
      state.token,
      `"${previousFrame.session_id}:${previousFrame.revision}"`,
    );
    if (joined !== null && liveRunHoldsNextTick) {
      // Run overlaps the policies' work with the tick on screen: the answer
      // is read early but shown only when that tick has played to its end,
      // in the same moment, so no settled still is painted between ticks.
      await choreographer.whenSettled();
    }
    if (
      joined === null ||
      state.busy ||
      state.shuttingDown ||
      state.frame !== previousFrame
    )
      return;
    const outcome = await presentationInstallation.installFromGet({
      reason: "live_poll",
      pendingPolicy: "retain_last_authorized",
      getJoined: async () =>
        prepareJoinedAuthority(joined, {
          previousAuthority,
          continuityResult: "stale_resync",
        }),
    });
    if (outcome.status !== "superseded") {
      // Pause asked to stop after the current turn; say when it has stopped.
      if (
        previousFrame.run_state === "running" &&
        state.frame?.run_state === "paused"
      ) {
        clearNotice(statusSentence({ kind: "paused" }));
      }
      render();
    }
  } catch (error) {
    if (state.busy || state.frame !== previousFrame || state.shuttingDown) return;
    if (error instanceof ProductIdentityMismatchError) failClosedProductIdentity(error);
    state.offline =
      error instanceof DebuggerApiError && [0, 401, 403].includes(error.status);
    state.resyncRequired = true;
    setNotice(
      error instanceof Error
        ? "Could not read the match. Press Reconnect."
        : "Could not read the match. Press Reconnect.",
      "error",
    );
    render();
  } finally {
    livePollInFlight = false;
    scheduleLivePoll();
  }
}

/**
 * The audience and tick of the last live choreography plan presented, kept
 * across a cleared presentation (a view switch clears the installed
 * presentation before the new view's frame arrives), or null before the
 * first live plan. Only settleLiveAudienceSwitch reads and writes it.
 *
 * @type {{audienceKey: unknown, simulatorStep: unknown} | null}
 */
let lastLivePlanShown = null;

/**
 * Show a live tick at its end, without playing it again, when only the
 * audience changed: Agent POV <-> Oracle View, or a new Agent POV team.
 *
 * Why: the choreographer keys a played tick by its transition id, and Agent
 * POV and Oracle View carry different ids for the same tick. Without this, a
 * view switch after a tick played would glide that tick again under the new
 * view, and during that replay the deck counts the tick as All Act, so a step
 * key or 0 1 2 pressed right after the switch was dropped without a word.
 * A view change only changes what the human sees (owner's rule), so it must
 * not replay motion or close orders.
 *
 * Call it right after choreographer.presentFrame with the plan it installed.
 * It compares that plan with the last live plan presented (lastLivePlanShown)
 * through audienceSwitchKeepsTick (controls.js: same session, same tick,
 * different audience) and then calls choreographer.skip(): every glide and
 * beat hold shows its END, submission gating is released and the deck
 * repaints through onStateChange. Live play only; the Replay Viewer is left
 * alone. A new tick, the first tick shown, a restart or a same-audience
 * repaint plays as before. The plan is already built for the new audience, so
 * nothing hidden from an Agent POV team is drawn. A switch made while the tick
 * is still gliding also jumps to END. Side effects: may finish the
 * choreographer's running animations; updates lastLivePlanShown. Returns
 * undefined.
 *
 * @param {unknown} nextPlan The choreographer's plan after presentFrame, or null.
 * @returns {void}
 */
function settleLiveAudienceSwitch(nextPlan) {
  if (isReplayMode() || !isRecord(nextPlan)) return;
  const shown = lastLivePlanShown;
  lastLivePlanShown = {
    audienceKey: nextPlan.audienceKey,
    simulatorStep: nextPlan.simulatorStep,
  };
  if (audienceSwitchKeepsTick(shown, lastLivePlanShown)) choreographer.skip();
}

/**
 * Paint the complete page from one coherent installed transport/presentation pair.
 * Share a single visual-filter snapshot across scene and choreography, refresh
 * panels, metrics, controls, and tooltips (the tooltip controller and the live
 * hover card, refreshHoverCard, both last), and preserve allowed focus/disclosure
 * choices. Presentation failures pause replay and clear choreography with a notice.
 *
 * The durable renderer always draws the END of the tick at the current camera
 * frame (cameraRenderFrame; Whole Map when null) and records the installed
 * policy: a one-tick replay_step request reaches the choreographer (a +1
 * plays its tick as Play does, then settles into the paused still; a -1
 * rewinds it, from copies of the arena the choreographer takes just before
 * this render, captureStepSnapshot) while the battlefield itself records
 * replay_static. Beat timing (the Fall and Return
 * beats, Longer Tick) comes from the choreography plan, which reads only this
 * audience's own events and the switch snapshot passed here.
 *
 * After the choreographer presents the tick: a live tick that is only being
 * re-shown to a new audience (a view switch) is settled at its END without
 * replaying (settleLiveAudienceSwitch); the camera switch gets the tick
 * (updateCameraForTick, which may lean for Kill Cam), the Rune Scrubber dims
 * its marks until their beats, the rail's Full Record chips and the Rune Scrubber
 * update, Frame Explanation refreshes, and (Replay Viewer) the Champion Frame is
 * drawn once the choreographer settles. While this runs, renderInProgress is
 * true, so a camera change asked for during the paint waits for it to finish.
 */
function render() {
  renderInProgress = true;
  try {
    renderPage();
  } finally {
    renderInProgress = false;
  }
}

/**
 * Paint the page once (the body of render; see render for the contract).
 */
function renderPage() {
  renderSystemOperation();
  capturePresentationPreferenceBeforeRender();
  const installed = installedPresentationAuthority();
  const presentationFrame = installed?.presentation ?? null;
  const transportFrame = installed?.transport ?? null;
  const visualFilterSnapshot = visualFilterState;
  const choreographyControl = installedChoreographyControl(
    presentationFrame,
    visualFilterSnapshot,
    { consumeAnimatedRestart: true },
  );
  renderVisualFilterControls(visualFilterSnapshot, visualFilterPresetId);
  renderConnection();
  renderSessionToolbar(installed);
  // A -1 rewinds from the arena as it is now: copy it before it is redrawn.
  if (
    choreographyControl.renderPolicy === "replay_step" &&
    choreographyControl.stepDirection === "backward"
  ) {
    choreographer.captureStepSnapshot(battlefieldRenderer.choreographySurface());
  }
  battlefieldRenderer.render(presentationFrame, {
    offline: state.offline,
    showRanges: installedPresentationRangesVisible(presentationFrame),
    localInspectedPresentationKey:
      installedLocalInspectedPresentationKey(presentationFrame),
    visualFilterState: visualFilterSnapshot,
    renderPolicy:
      choreographyControl.renderPolicy === "replay_step"
        ? "replay_static"
        : choreographyControl.renderPolicy,
    cameraFrame: cameraRenderFrame(),
    // Agent POV Watch Only: the champion looked at, when it is not yours.
    focusedPublicAgentId: isReplayMode()
      ? null
      : (transportFrame?.focused_public_agent_id ?? null),
    // Live only: the Manual slots, so the hotbar dims a Command Ring round
    // every other Manual champion this audience's scene draws (One On Each
    // Team). Agent POV still hides another team's staged order: the studs
    // need a staged row, and Agent POV frames carry none.
    manualSlots: isReplayMode()
      ? null
      : Array.isArray(transportFrame?.combat_configuration?.manual_slots)
        ? transportFrame.combat_configuration.manual_slots.filter(
            (/** @type {unknown} */ slot) => Number.isInteger(slot),
          )
        : null,
  });
  applyBattlefieldBoundaryCopy();
  installAuthorizedAgentActivation();
  applyReplayReferenceSemantics();
  try {
    choreographer.presentFrame(
      presentationFrame,
      pageChoreographySurface(),
      choreographyControl,
    );
    settleLiveAudienceSwitch(choreographer.plan);
    updateCameraForTick(presentationFrame, choreographyControl.renderPolicy);
    const plan = choreographer.plan;
    if (
      runeScrubber !== null &&
      isRecord(plan) &&
      isRecord(installed?.transport?.cursor)
    ) {
      runeScrubber.presentTick(
        Number(installed.transport.cursor.frame_index),
        scrubberBeatsFromSchedule(plan.beats),
        replayPlayback.snapshot().playbackRate,
      );
    }
  } catch (error) {
    pauseReplayAfterPresentationFailure("presentation_error");
    choreographer.clear("presentation_error");
    setNotice(
      error instanceof Error
        ? "Could not draw this tick. The panels still show its facts."
        : "Could not draw this tick. The panels still show its facts.",
      "error",
    );
    renderConnection();
  }
  renderMatchMomentCards(presentationFrame);
  if (choreographer.plan === null) renderFallenFeed(null);
  renderReplayArtifactActions(installed);
  lastBattlefieldSizeKey = battlefieldSizeKey();
  renderReplayMetrics();
  refreshLedgerRail(presentationFrame);
  updateRuneScrubber(replayPlayback.snapshot());
  frameExplanation?.refresh();
  if (isReplayMode()) {
    const shown = presentationFrame;
    if (!isAuthorizedPresentationFrame(shown)) {
      // No authority to wait for: empty the Champion Frame now.
      renderChampionFrameForPage(shown);
    } else {
      void choreographer.whenSettled().then(() => {
        if (installedPresentationAuthority()?.presentation === shown) {
          renderChampionFrameForPage(shown);
        }
      });
    }
  }
  panels.render(presentationFrame, {
    visualFilterState: visualFilterSnapshot,
    busy: state.busy,
    shuttingDown: state.shuttingDown,
    resyncRequired: state.resyncRequired,
    offline: state.offline,
    activationDisabled:
      !isReplayMode() &&
      Boolean(transportFrame?.system_operation) &&
      !(
        transportFrame?.fresh_launch_required &&
        transportFrame.system_operation.state === "failed"
      ),
    localInspectedPresentationKey:
      installedLocalInspectedPresentationKey(presentationFrame),
    activatedAgentPublicId,
    combatConfiguration: transportFrame?.combat_configuration ?? null,
  });
  tooltipController.refresh();
  // Live play's Champion Card (#hcard) follows the new frame too.
  refreshHoverCard();
  restorePresentationPreferenceAfterRender();
}

/**
 * Return the rounded battlefield width and height as a CSS-pixel size key.
 * Resize handling uses this key to skip unchanged dimensions.
 */
function battlefieldSizeKey() {
  return `${Math.round(elements.battlefield.clientWidth)}x${Math.round(elements.battlefield.clientHeight)}`;
}

/**
 * Coalesce resize notifications into one animation-frame repaint.
 * Skip unchanged rounded dimensions, repaint the scene at the current camera
 * frame, and reproject existing choreography without advancing simulation (a
 * step glide never restarts here). Refresh artifact controls/tooltips (and the
 * live hover card) and preserve local preferences; presentation failures pause
 * replay.
 */
function scheduleBattlefieldResize() {
  if (pendingResizeFrame !== null) {
    return;
  }
  pendingResizeFrame = window.requestAnimationFrame(() => {
    pendingResizeFrame = null;
    const sizeKey = battlefieldSizeKey();
    if (sizeKey === lastBattlefieldSizeKey) {
      return;
    }
    lastBattlefieldSizeKey = sizeKey;
    capturePresentationPreferenceBeforeRender();
    const presentationFrame = state.presentation;
    const visualFilterSnapshot = visualFilterState;
    const choreographyControl = installedChoreographyControl(
      presentationFrame,
      visualFilterSnapshot,
    );
    battlefieldRenderer.render(presentationFrame, {
      offline: state.offline,
      showRanges: installedPresentationRangesVisible(presentationFrame),
      localInspectedPresentationKey:
        installedLocalInspectedPresentationKey(presentationFrame),
      visualFilterState: visualFilterSnapshot,
      renderPolicy:
        choreographyControl.renderPolicy === "replay_step"
          ? "replay_static"
          : choreographyControl.renderPolicy,
      cameraFrame: cameraRenderFrame(),
    });
    applyBattlefieldBoundaryCopy();
    installAuthorizedAgentActivation();
    applyReplayReferenceSemantics();
    try {
      choreographer.reproject(
        presentationFrame,
        pageChoreographySurface(),
        choreographyControl,
      );
      frameExplanation?.refresh();
    } catch (error) {
      pauseReplayAfterPresentationFailure("resize_projection_error");
      choreographer.clear("resize_projection_error");
      setNotice(
        error instanceof Error
          ? "Could not redraw the arena at this size."
          : "Could not redraw the arena at this size.",
        "error",
      );
      renderConnection();
    }
    renderMatchMomentCards(presentationFrame);
    renderReplayArtifactActions(installedPresentationAuthority());
    tooltipController.refresh();
    refreshHoverCard();
    restorePresentationPreferenceAfterRender();
  });
}

/**
 * Wrap a live command with schema version, client ID, new UUID, and current revision.
 * Return the request without sending it. Each call creates a fresh command identity.
 *
 * @param {Record<string, unknown>} command
 */
function commandRequest(command) {
  return {
    schema_version: 1,
    client_id: state.clientId,
    command_id: window.crypto.randomUUID(),
    base_revision: currentRevision(),
    command,
  };
}

/**
 * Describe the episode replacement that a recording-discard dialog would permit.
 * Recognize reset, scenario switch, and combat configuration; use a general
 * replacement label for other commands. This does not perform the replacement.
 *
 * @param {Readonly<Record<string, unknown>>} replacement
 */
function recordingReplacementLabel(replacement) {
  if (replacement.command_type === "reset") {
    return "Reset the current episode";
  }
  if (replacement.command_type === "scenario_switch") {
    return `Switch to scenario ${String(replacement.scenario_name)}`;
  }
  if (replacement.command_type === "set_combat_configuration") {
    const teamAController = combatControllerLabel(replacement.team_a_controller);
    const teamBController = combatControllerLabel(replacement.team_b_controller);
    const information =
      replacement.execution_information_mode === "shared_obs"
        ? "SharedObs"
        : "NoSharedObs";
    return `Restart with Team A ${teamAController}, Team B ${teamBController}, and ${information}`;
  }
  return "Replace the current episode";
}

/**
 * Check the exact match and revision captured when replacement was requested.
 * Confirmations cannot silently follow a newer frame, another tab or a restart.
 *
 * @param {{session_id: string, run_generation: number, revision: number}} request
 */
function recordingReplacementIsCurrent(request) {
  return (
    installedAuthorityIsCoherent() &&
    !state.busy &&
    !state.shuttingDown &&
    !state.resyncRequired &&
    !state.offline &&
    !state.frame?.fresh_launch_required &&
    request.session_id === state.frame?.session_id &&
    request.run_generation === state.frame?.run_generation &&
    request.revision === state.frame?.revision &&
    recordingStatus()?.discard_available === true
  );
}

/**
 * Show Save, Discard and Cancel for one exact, already selected replacement.
 * perform sends one revision-guarded request; Cancel sends no request (the
 * dialog's close then says "Restart cancelled. The game goes on."). Save gets
 * keyboard focus, as on the vision page.
 *
 * @param {NonNullable<typeof pendingRecordingReplacement>} request
 * @returns {boolean} Whether this request still owns the displayed match.
 */
function showRecordingReplacementDialog(request) {
  if (!recordingReplacementIsCurrent(request)) return false;
  pendingRecordingReplacement = Object.freeze({ ...request });
  elements.recordingDiscardConfirmButton.disabled = false;
  elements.recordingSaveReplaceButton.disabled = false;
  elements.recordingDiscardIntent.textContent = request.label;
  if (!elements.recordingDiscardDialog.open)
    elements.recordingDiscardDialog.showModal();
  // Save is the first and suggested choice, as on the vision page.
  elements.recordingSaveReplaceButton.focus({ preventScroll: true });
  return true;
}

/**
 * Capture a live replacement and its current revision for the shared dialog.
 * No save or replacement happens until an explicit confirmation.
 *
 * @param {Readonly<Record<string, unknown>>} replacement
 */
function requestRecordingDiscardConfirmation(replacement) {
  showRecordingReplacementDialog({
    session_id: state.frame?.session_id,
    run_generation: state.frame?.run_generation,
    revision: state.frame?.revision,
    label: recordingReplacementLabel(replacement),
    perform: (choice) =>
      dispatchCommand({
        command_type:
          choice === "save" ? "save_and_replace" : "confirm_discard_and_replace",
        replacement,
      }),
  });
}

/**
 * Confirm one replacement only while its original match and revision still exist.
 * Errors leave the accepted match available and are shown through the normal notice.
 *
 * @param {"save" | "discard"} choice
 */
async function confirmRecordingReplacement(choice) {
  const request = pendingRecordingReplacement;
  pendingRecordingReplacement = null;
  elements.recordingDiscardDialog.close();
  if (request === null || !recordingReplacementIsCurrent(request)) {
    setNotice(
      "The match changed. Choose the replacement again; nothing was replaced.",
      "warning",
    );
    renderConnection();
    return;
  }
  try {
    await request.perform(choice);
  } catch (error) {
    setNotice(
      error instanceof Error ? error.message : "The match could not be replaced.",
      "error",
    );
    renderConnection();
  }
}

/**
 * Send one replay command and install its matching authorized successor.
 * Reject unavailable, busy, or disconnected use. Never retry the POST: a stale
 * response or join race may install a fresh GET pair and return handled_resync.
 * Update connection, shutdown, and notice state; rethrow failures after fencing
 * authority as needed. deferFinalRender defaults to false for ordinary commands.
 * With deferFinalRender true (the playback controller's play and step commands)
 * the page is not repainted while the request is in flight, only the connection
 * and busy state, and the final repaint is left to the controller; so the old
 * frame is never redrawn between two played ticks. Autoplay's own next_frame
 * request also sets state.quietBusy while it runs, so the connection pill stays
 * "Connected" between played ticks.
 *
 * @param {Readonly<Record<string, any>>} command
 * @param {{deferFinalRender?: boolean}} [options]
 */
async function sendReplayCommand(command, { deferFinalRender = false } = {}) {
  if (!isReplayMode() || !state.frame || !state.presentation) {
    throw new DebuggerApiError("Replay controls require an installed replay frame.");
  }
  if (state.busy || state.shuttingDown) {
    throw new DebuggerApiError("A replay request is already in flight.");
  }
  if (state.resyncRequired || state.offline) {
    throw new DebuggerApiError("Reconnect before sending another replay command.");
  }
  /**
   * Resolve the replay-command completion barrier after its finally cleanup.
   * Start as a no-op, then replace it with the Promise resolver before the request;
   * view-switch handlers await this barrier before sending another command.
   */
  let finishCommand = () => {};
  replayCommandCompletion = new Promise((resolve) => {
    finishCommand = resolve;
  });
  invalidateReplayArtifactAction();
  state.busy = true;
  // Autoplay's own fetch of the next frame keeps the pill on "Connected".
  state.quietBusy = deferFinalRender && command.command_type === "next_frame";
  clearNotice();
  const previousAuthority = state.authority;
  const previousFrame = state.frame;
  const previousCursor = previousFrame.cursor;
  const request = replayCommandRequest({
    clientId: state.clientId,
    commandId: window.crypto.randomUUID(),
    baseRevision: currentRevision(),
    command,
  });
  /** @type {{current: {kind: "success" | "stale", payload: any} | null}} */
  const commandOutcome = { current: null };
  try {
    const installPromise = presentationInstallation.installFromCommand({
      reason:
        command.command_type === "set_view"
          ? "replay_audience_change"
          : "replay_command",
      pendingPolicy: "retain_last_authorized",
      /**
       * Post the captured replay request and remember its response.
       * Resolve to a success outcome, or a stale outcome for HTTP 409. Other API and
       * transport failures reject; the request is not retried inside this callback.
       */
      sendCommand: async () => {
        try {
          const payload = await postReplayCommand(state.token, request);
          commandOutcome.current = { kind: "success", payload };
          return commandOutcome.current;
        } catch (error) {
          if (error instanceof DebuggerApiError && error.status === 409) {
            commandOutcome.current = { kind: "stale", payload: error.payload };
            return commandOutcome.current;
          }
          throw error;
        }
      },
      /**
       * Join outcome.payload with the current presentation and prepare installation.
       * The outcome comes from this transaction's send callback. Require replay
       * transport and validate successful command continuity. Missing joins, wrong
       * products, invalid outcomes and network failures reject the promise.
       */
      joinCommandResult: async (outcome) => {
        const joined = await extractJoinedFrame(
          outcome.payload,
          await getCurrentPresentation(state.token),
        );
        if (joined === null) {
          throw new TypeError("Replay response has no joinable transport candidate.");
        }
        if (joined.transport.viewer_mode !== "replay") {
          throw new DebuggerApiError(
            "Replay response did not contain replay transport authority.",
          );
        }
        if (outcome.kind === "success") {
          validateReplayCommandOutcome(
            request.command,
            outcome.payload,
            previousCursor,
          );
        }
        return prepareJoinedAuthority(joined, {
          previousAuthority,
          continuityResult:
            outcome.kind === "success" ? outcome.payload.result : "stale_resync",
        });
      },
      /**
       * Fetch a fresh matched frame/presentation for replay resynchronization.
       * Prepare it against previousAuthority with stale_resync continuity. The
       * promise rejects on fetch, join or preparation failure; no command is repeated.
       */
      getJoined: async () =>
        prepareJoinedAuthority(await getCurrentAuthority(), {
          previousAuthority,
          continuityResult: "stale_resync",
        }),
    });
    if (deferFinalRender) {
      // Playback and stepping (the playback controller's commands): do not
      // repaint the old frame while the next one is fetched. That repaint drew
      // the settled still and dropped the tick's held END cues just before the
      // next tick started: the "paused" flash between ticks. Only the busy
      // state is shown; the controller repaints when the answer lands.
      renderConnection();
    } else {
      render();
    }
    const installOutcome = await installPromise;
    if (installOutcome.status === "superseded") {
      throw new DebuggerApiError(
        "Replay response was superseded by a newer authority request.",
      );
    }
    state.offline = false;
    state.resyncRequired = false;
    replayPlayback.setConnected(true);
    const payload = commandOutcome.current?.payload;
    const stale = commandOutcome.current?.kind === "stale";
    const notice = extractNotice(payload);
    // A plain frame update is routine and makes no notice.
    const routineReplayUpdate =
      !stale &&
      !installOutcome.resynchronized &&
      notice === null &&
      payload?.result !== "duplicate" &&
      payload?.result !== "no_op";
    if (routineReplayUpdate) {
      clearNotice();
    } else {
      setNotice(
        stale
          ? payload?.error_code === "command_id_conflict"
            ? "Two presses crossed, so the page now shows the latest tick. Your press was not repeated."
            : "This tab was behind, so it now shows the latest tick. Your press was not repeated."
          : installOutcome.resynchronized
            ? "Done. The page reloaded the tick to stay in step."
            : (notice ??
              (payload?.result === "duplicate"
                ? "That press was already done."
                : "Already at that tick.")),
        stale ||
          installOutcome.resynchronized ||
          payload?.result === "duplicate" ||
          payload?.result === "no_op"
          ? "warning"
          : "success",
      );
    }
    if (commandResponseSchedulesShutdown(request.command, payload)) {
      state.shuttingDown = true;
      setNotice("The Replay Viewer is closing.", "info");
    }
    return stale || installOutcome.resynchronized
      ? Object.freeze({ handled_resync: true, frame: state.frame })
      : payload;
  } catch (error) {
    if (error instanceof ProductIdentityMismatchError) {
      failClosedProductIdentity(error);
      throw error;
    }
    const status = error instanceof DebuggerApiError ? error.status : 0;
    if (isPresentationJoinRace(error)) {
      clearPresentationAuthority("replay_presentation_identity_mismatch");
    } else if (status === 401 || status === 403) {
      clearPresentationAuthority("replay_authorization_failure");
    }
    const payload = commandOutcome.current?.payload;
    if (commandResponseSchedulesShutdown(request.command, payload)) {
      state.shuttingDown = true;
      setNotice("The Replay Viewer is closing.", "info");
    } else {
      state.offline =
        error instanceof DebuggerApiError &&
        (status === 0 || status === 401 || status === 403);
      state.resyncRequired = true;
      replayPlayback.setConnected(false);
      setNotice(
        status === 401 || status === 403
          ? "This link no longer works. Open the link printed in the terminal again."
          : error instanceof Error
            ? "Could not reach the Replay Viewer. Press Reconnect."
            : "Replay command failed. Reconnect before sending another command.",
        "error",
      );
    }
    throw error;
  } finally {
    state.busy = false;
    state.quietBusy = false;
    if (!deferFinalRender || state.resyncRequired || state.shuttingDown) {
      render();
    }
    finishCommand();
  }
}

/**
 * Send a replay command with its final render deferred to the playback controller.
 * Return the request promise so playback can validate its successor intent before
 * the newly installed frame receives a non-pending render. Failures propagate.
 *
 * @param {Readonly<Record<string, any>>} command
 */
function sendReplayTransportCommand(command) {
  return sendReplayCommand(command, { deferFinalRender: true });
}

/** @type {Promise<void>} */
let replayCommandCompletion = Promise.resolve();

/**
 * Run a user replay command through audience gates and playback coordination.
 * Pause first; view/recipient changes wait for the current request and may resume
 * previous playback if their generation is still current. Return the payload or
 * null for blocked, superseded, or handled failed requests; clear pending recipient
 * activation after settlement.
 *
 * @param {Readonly<Record<string, any>>} command
 */
async function dispatchReplayCommand(command) {
  if (
    authorizedPresentationAudience(state.presentation) !== "researcher" &&
    (command.command_type === "select_agent" || command.command_type === "set_ranges")
  ) {
    setNotice("Ranges and reference picks work only in Oracle View.", "warning");
    renderConnection();
    return null;
  }
  const stagedRecipientActivation =
    command.command_type === "set_pov_actor" ? pendingReplayRecipientActivation : null;
  const changingView = ["set_view", "set_pov_actor"].includes(command.command_type);
  const wasPlaying = replayPlayback.snapshot().playing;
  invalidateReplayArtifactAction();
  const suspended = replayPlayback.pause("user_command");
  try {
    if (changingView) {
      await replayCommandCompletion;
      if (replayPlayback.snapshot().generation !== suspended.generation) {
        return null;
      }
    }
    const payload = await sendReplayCommand(command);
    const frame = state.frame;
    const resume =
      changingView &&
      wasPlaying &&
      replayPlayback.snapshot().generation === suspended.generation;
    if (frame) {
      replayPlayback.installCursor(frame.cursor);
      if (resume && !replayPlayback.snapshot().atEnd) {
        replayPlayback.play({ restartCurrent: false });
      }
    }
    if (
      command.command_type === "set_pov_actor" &&
      !state.resyncRequired &&
      !state.offline &&
      authorizedPresentationAudience(state.presentation) === "agent_pov"
    ) {
      elements.battlefield.focus({ preventScroll: true });
    }
    return payload;
  } catch {
    return null;
  } finally {
    if (pendingReplayRecipientActivation === stagedRecipientActivation) {
      pendingReplayRecipientActivation = null;
    }
  }
}

/**
 * Route a panel action to local activation, live command, or replay command.
 * Validate public identity/slot pairs against authorized identity rows. Pointer
 * activation defaults to false and controls focus handoff. Return a promise for
 * the routed result, or resolved null when the action is local or unavailable.
 *
 * @param {Record<string, unknown>} command
 * @param {Readonly<{pointerOriginated?: boolean}>} context
 */
function dispatchPanelCommand(command, { pointerOriginated = false } = {}) {
  if (
    command.command_type === "activate_authorized_agent" &&
    typeof command.presentation_key === "string"
  ) {
    activateAuthorizedAgent(command.presentation_key, { pointerOriginated });
    return Promise.resolve(null);
  }
  if (
    command.command_type === "activate_replay_pov_agent" &&
    Number.isInteger(command.global_slot) &&
    Number(command.global_slot) >= 0 &&
    Number(command.global_slot) < 10 &&
    typeof command.public_agent_id === "string" &&
    authorizedPresentationIdentityRows(state.presentation).some(
      (identity) =>
        identity.public_agent_id === command.public_agent_id &&
        identity.command_global_slot === command.global_slot,
    ) &&
    isReplayMode()
  ) {
    stageReplayRecipientActivation(command.public_agent_id);
    return dispatchReplayCommand({
      command_type: "set_pov_actor",
      global_slot: command.global_slot,
    });
  }
  if (
    command.command_type === "activate_live_pov_agent" &&
    Number.isInteger(command.global_slot) &&
    Number(command.global_slot) >= 0 &&
    Number(command.global_slot) < 10 &&
    typeof command.public_agent_id === "string" &&
    authorizedPresentationIdentityRows(state.presentation).some(
      (identity) =>
        identity.public_agent_id === command.public_agent_id &&
        identity.command_global_slot === command.global_slot,
    ) &&
    !isReplayMode() &&
    authorizedPresentationAudience(state.presentation) === "agent_pov"
  ) {
    activatedAgentPublicId = command.public_agent_id;
    openAgentDetails();
    const controlCommand = {
      command_type: "roster_selection",
      role: "control",
      global_slot: command.global_slot,
    };
    if (pointerOriginated) {
      dispatchCommandFromDraftControl(controlCommand);
      return Promise.resolve(null);
    }
    return dispatchCommand(controlCommand);
  }
  if (
    command.command_type === "roster_selection" &&
    Number.isInteger(command.global_slot)
  ) {
    openAgentDetails();
    if (!isReplayMode()) {
      return dispatchCommand(command);
    }
    if (command.role === "target") {
      return dispatchReplayCommand({
        command_type: "select_agent",
        selected_global_slot: command.global_slot,
      });
    }
  }
  if (!isReplayMode()) {
    return dispatchCommand(command);
  }
  setNotice("That action works only in live play.", "warning");
  renderConnection();
  return Promise.resolve(null);
}

/**
 * The commands the page has already sent a second time after a "stale"
 * answer (resendsAfterStale), so one is never sent a third time.
 *
 * @type {WeakSet<Record<string, unknown>>}
 */
const resentAfterStale = new WeakSet();

/**
 * Return whether a live command keeps its meaning on a newer frame, so the
 * page sends it once more after a "stale" answer: Exit, and Save And Replace
 * (a restart pressed while the game-over save ran, or Save in the Restart
 * dialog). The game-over save installs a new revision when it ends, so a
 * command sent just before that moment arrives one revision late; every
 * other command still stops at a stale answer, because its meaning depends
 * on the frame the player saw. Pure.
 *
 * @param {Readonly<Record<string, unknown>>} command The live command sent.
 * @returns {boolean}
 */
function resendsAfterStale(command) {
  return command.command_type === "exit" || command.command_type === "save_and_replace";
}

/**
 * Mark an accepted Exit (its answer scheduled the shutdown): the page stops
 * reading (state.shuttingDown), keeps the answer's own recording status
 * (state.exitRecording; null when the match records nothing) for the
 * Recording pill, popover and save button, says the closed sentence for that save
 * state (closedSentence in tactics-turn.js: saved, no tick played, or
 * nothing recorded) and shows keys off, since no key works any more.
 * Side effects: page state, the notice and the Keys chip.
 *
 * @param {unknown} payload The Exit command's answer (result
 *   "shutdown_scheduled"), whose `frame.recording` is read when present.
 * @returns {void}
 */
function acceptExit(payload) {
  state.shuttingDown = true;
  const answerFrame = isRecord(payload) ? payload.frame : null;
  state.exitRecording =
    isRecord(answerFrame) && isRecord(answerFrame.recording)
      ? answerFrame.recording
      : null;
  setNotice(closedSentence(state.exitRecording), "info");
  paintKeysState();
}

/**
 * Serialize one live command through authority, policy, terminal, and recording gates.
 * Optional deferredSubmit defaults to null and may retain one fresh Enter after
 * draft preparation. Send the POST once, install a coherent successor, and only
 * then dispatch retained input. Show handled failures, require resync when needed,
 * and open requested replays separately; normal completion resolves without a value.
 *
 * Message line: the page writes the vision's sentence for the player's own
 * commands (statusSentence in controls.js). A 1 or 2 that the decision mask
 * refuses is refused here, before sending ("Charge needs a target first. Your
 * order is unchanged.", or "... is not allowed on this target right now."),
 * and its slot shakes; an edit of a policy's champion or after a stop gets the
 * vision's refusal; Lock In (and Play One Turn in Policies Only) shows
 * "Waiting for all players to submit their actions…" while the turn is out,
 * so a slow method never looks like a frozen game; an accepted edit,
 * view change, focus change or G shows what changed. When the command's own
 * answer was overtaken (the policies finished the turn first, so one fresh
 * read was installed instead), Lock In still says its sentence and any other
 * command says "Your command was applied once. The game had moved on, so the
 * page now shows the latest state." Python's "The Systems are choosing this
 * turn." is said as the Lock In sentence (hostNoticeSentence). Other host
 * notices still show as before. Agent POV: the facts come from ownOrderFacts.
 *
 * Tab and Shift+Tab (also sent by ‹ Prev and Next ›): with Manual champions,
 * the key is first turned into a focus change (roster_selection, role
 * "control") to your next champion that can take orders, worked out from the
 * newest frame when the command is about to go out (ownChampionTabStep), and
 * that command is sent instead with the same options. When nothing needs
 * sending, the message line says "Giving orders to your <Class>." (the only
 * one that can take orders) or "All your champions are fallen, so nobody can
 * take orders until they return.", and inputs queued behind the Tab still go
 * out in order. In Policies Only the key goes to Python unchanged.
 *
 * A restart (R) pressed while the game-over save runs ("Saving Replay…")
 * opens no Save, Discard or Cancel dialog: recordingCommandDecision turns it
 * into Save And Replace, which is sent instead with the same options, and the
 * server starts the new match once the save has ended. Meanwhile the status
 * line says "The replay is still saving. The new match starts once it is
 * saved."
 *
 * Exit: pressed while the game-over save runs, the message line first says
 * "The replay is still saving. The match closes once it is saved." (the
 * server waits for the save). An accepted Exit (acceptExit) says the closed
 * sentence for the replay's save state from the answer's own recording
 * status ("The game so far is saved", "No tick was played, so no replay was
 * saved", or nothing about a replay when the match records nothing) and
 * turns keys off. Exit
 * and Save And Replace that get a "stale" answer (the game-over save ended
 * just before they arrived) are sent once more from the fresh frame
 * (resendsAfterStale), with the same source and quiet options, and the
 * stale answer itself writes no notice; any other stale command is not sent
 * again.
 *
 * @param {Record<string, unknown>} command
 * @param {{
 *   deferredDrafts?: ReadonlyArray<Readonly<Record<string, unknown>>>,
 *   deferredSubmit?: Readonly<Record<string, unknown>> | null,
 *   source?: "field" | "control" | "page",
 *   quiet?: boolean,
 *   keepPresentation?: boolean,
 * }} options source (default "page") says where the input came from: "field"
 *   for the focused battlefield (Esc then also says keys are off), "control"
 *   for a deck or rail button. quiet (default false) leaves the message line
 *   alone while the command is out and after it is accepted; the page's own
 *   Run sets it so "Running." stays on the line from turn to turn, as on the
 *   vision page. Refusals, stale or duplicate answers and failures still
 *   write their sentence. While the page's own Run is active
 *   (pageRunOwnsLockIn), a Space, Enter or N keyboard command that is not quiet
 *   (a key, the Lock In button or a queued Lock In) is ignored without a
 *   word and nothing is sent, as on the vision page; only Run's quiet turns
 *   go out. A focus change (roster_selection, role "control") made while
 *   an earlier request is out is queued behind it (retainFocusChange) instead
 *   of being refused. deferredDrafts (default none) are order keys and focus
 *   changes made while an earlier request was out, oldest first: they are
 *   sent one after another after this command, each through its own fence
 *   and with source "field", and a retained Lock In (deferredSubmit) goes
 *   last. keepPresentation (default
 *   false) lets a Lock In go out while the current tick still plays: the
 *   tick is not cut to its end first, as a player's Lock In does. Only the
 *   page's own Run sets it (Policies Only, no Manual drafts), so the policies
 *   choose the next turn while this tick's last beats play; runLiveTurns then
 *   holds the next tick until this one settles (liveRunHoldsNextTick).
 *   When the answer leaves policy work out (or the server running), the next
 *   live reads come every 80 ms for 3 s (hurryLivePoll), so the
 *   new tick shows soon after the policies answer.
 */
async function dispatchCommand(
  command,
  {
    deferredDrafts = [],
    deferredSubmit = null,
    source = "page",
    quiet = false,
    keepPresentation = false,
  } = {},
) {
  if (isReplayMode()) {
    setNotice("Live play commands do not work in a replay.", "warning");
    renderConnection();
    return;
  }
  if (!quiet && pageRunOwnsLockIn(command)) {
    // Only the page's own Run sends turns while it plays, as on the vision
    // page: this Space or Enter is ignored, so no turn is sent twice and no
    // glide is cut short.
    return;
  }
  if (state.busy && !state.shuttingDown && retainFocusChange(command)) {
    renderConnection();
    return;
  }
  if (state.busy || state.shuttingDown) {
    setNotice("A command is already in flight; no second command was sent.", "warning");
    renderConnection();
    return;
  }
  if (state.resyncRequired || state.offline) {
    setNotice(
      "Reconnect to install the latest authoritative frame before sending another command.",
      "warning",
    );
    renderConnection();
    return;
  }
  if (!state.frame || !state.presentation) {
    setNotice(
      "No coherent transport/presentation pair is available. Reconnect before sending commands.",
      "error",
    );
    renderConnection();
    return;
  }
  // Tab, Shift+Tab, ‹ Prev and Next › visit only your champions that can take
  // orders (owner's ruling, 3 October 2026): worked out here, at send time,
  // from the newest frame, so a Tab queued behind an earlier order moves on
  // from where that order left the focus.
  const tabStep = ownChampionTabStep(command);
  if (tabStep.kind === "send") {
    return dispatchCommand(tabStep.command, {
      deferredDrafts,
      deferredSubmit,
      source,
      quiet,
      keepPresentation,
    });
  }
  if (tabStep.kind === "stay" || tabStep.kind === "none") {
    clearNotice(
      tabStep.kind === "stay"
        ? statusSentence({ kind: "focus_own", className: tabStep.className })
        : statusSentence({ kind: "nobody_ready" }),
    );
    renderConnection();
    // Nothing was sent; inputs queued behind this Tab still go out in order.
    if (deferredDrafts.length > 0) {
      return dispatchCommand(deferredDrafts[0], {
        deferredDrafts: deferredDrafts.slice(1),
        deferredSubmit,
        source: "field",
      });
    }
    if (deferredSubmit !== null) {
      return dispatchCommand(deferredSubmit, { source: "field" });
    }
    return;
  }
  if (
    command.command_type === "keyboard" &&
    typeof command.key === "string" &&
    authorizedPresentationAudience(state.presentation) === "agent_pov"
  ) {
    if (command.key.toLowerCase() === "g") {
      if (
        command.shift_key === false &&
        command.ctrl_key === false &&
        command.alt_key === false &&
        command.meta_key === false &&
        command.repeat === false &&
        toggleAgentLocalRanges()
      ) {
        clearNotice(statusSentence({ kind: "ranges", shown: agentLocalRangesVisible }));
        render();
      }
      return;
    }
  }
  const beforeFacts = ownOrderFacts(state.presentation, state.frame);
  if (policyControllerBlocksActionEdit(command)) {
    const escapeOnly =
      command.command_type === "keyboard" &&
      String(command.key).toLowerCase() === "escape";
    setNotice(
      escapeOnly
        ? statusSentence({ kind: "keys_off" })
        : statusSentence({
            kind: "policy_champion",
            watch: beforeFacts?.mode === "watch",
          }),
      escapeOnly ? "info" : "warning",
    );
    renderConnection();
    renderCommandAvailability();
    return;
  }
  if (liveScriptedInspectionOnly() && !allowedDuringLiveScriptedInspection(command)) {
    setNotice(
      "Scripted live view is inspection-only. Use Advance scripted frame for the next authorized step.",
      "warning",
    );
    renderConnection();
    renderCommandAvailability();
    return;
  }
  const mode = modeAvailability(command, state.frame);
  if (!mode.allowed) {
    setNotice(
      state.frame?.fresh_launch_required && commandRole(command) !== null
        ? statusSentence({ kind: "stopped" })
        : (mode.notice ?? "That command is unavailable in the current mode."),
      "warning",
    );
    renderConnection();
    renderCommandAvailability();
    return;
  }
  const refusal = laneRefusal(command, beforeFacts);
  if (refusal !== null) {
    setNotice(refusal.sentence, "warning");
    shakeSlot(
      refusal.lane === "basic" ? elements.basicButton : elements.ultimateButton,
    );
    renderConnection();
    return;
  }
  const recordingDecision = recordingCommandDecision(state.frame, command);
  if (recordingDecision.action === "block") {
    setNotice(
      recordingDecision.notice ??
        "That command is fenced by the current recording lifecycle.",
      "warning",
    );
    renderConnection();
    renderSessionToolbar();
    return;
  }
  if (recordingDecision.action === "confirm") {
    if (recordingDecision.replacement) {
      requestRecordingDiscardConfirmation(recordingDecision.replacement);
    }
    return;
  }
  if (
    command.command_type === "exit" &&
    ["sealed", "finalized_unsaved"].includes(String(recordingStatus()?.lifecycle))
  ) {
    // Exit waits for the game-over save: the line says why nothing happens yet.
    clearNotice("The replay is still saving. The match closes once it is saved.");
  }
  if (
    recordingDecision.action === "allow" &&
    recordingDecision.command !== undefined &&
    recordingDecision.command !== command
  ) {
    // A restart pressed while the game-over save runs goes out as Save And
    // Replace (no dialog); it passes the same fences as any other command.
    // The line says why nothing happens until the save ends.
    clearNotice("The replay is still saving. The new match starts once it is saved.");
    return dispatchCommand(recordingDecision.command, {
      deferredDrafts,
      deferredSubmit,
      source,
      quiet,
      keepPresentation,
    });
  }
  if (
    isSubmissionCommand(command) &&
    !keepPresentation &&
    presentationRequiresSubmissionSettle(choreographer.snapshot())
  ) {
    // Submit owns this synchronous edge: settle the current (even paused)
    // explanation, then send the same current draft through the normal fence.
    choreographer.skip();
  }

  const recordingLifecycle = recordingStatus()?.lifecycle;
  /** @type {{
   *   allowsDeferredSubmit: boolean,
   *   deferredDrafts: Array<Readonly<Record<string, unknown>>>,
   *   deferredSubmit: Readonly<Record<string, unknown>> | null,
   *   releaseFocusAfterSettlement: boolean,
   *   recordingPreparation: "metrics" | "possible" | null,
   *   keepsTarget: boolean,
   *   lateRefusal: string | null,
   * }} */
  const liveCommandTransaction = {
    lateRefusal: null,
    keepsTarget:
      command.command_type === "keyboard" &&
      typeof command.key === "string" &&
      command.key.length === 1 &&
      "wasdqezcx012".includes(command.key.toLowerCase()),
    allowsDeferredSubmit: commandPreparesDeferredSubmit(command),
    deferredDrafts: [...deferredDrafts],
    deferredSubmit:
      commandPreparesDeferredSubmit(command) &&
      deferredSubmit !== null &&
      isFreshUnmodifiedEnter(deferredSubmit)
        ? deferredSubmit
        : null,
    releaseFocusAfterSettlement: false,
    recordingPreparation:
      ["recording", "sealed"].includes(recordingLifecycle) &&
      ["finish_and_review", "save_and_replace", "exit"].includes(
        String(command.command_type),
      )
        ? "metrics"
        : null,
  };
  activeLiveCommandTransaction = liveCommandTransaction;
  state.busy = true;
  state.offline = false;
  const recordingHintTimer =
    recordingLifecycle === "recording" && isSubmissionCommand(command)
      ? window.setTimeout(() => {
          if (activeLiveCommandTransaction === liveCommandTransaction && state.busy) {
            liveCommandTransaction.recordingPreparation = "possible";
            renderRecordingControls(installedPresentationAuthority());
          }
        }, 2000)
      : null;
  // A click on the arena that only turns keys on keeps the current message,
  // and so does a turn sent by the page's own Run (quiet), which keeps
  // "Running." on the line as the vision does.
  const quietPointer =
    quiet ||
    (command.command_type === "battlefield_pointer" && command.button === "primary");
  if (!quietPointer) {
    clearNotice(
      isSubmissionCommand(command)
        ? statusSentence({ kind: "lock_in", watch: beforeFacts?.mode === "watch" })
        : "Waiting for the game to answer…",
    );
  }
  // A new command is out: an older sentence its answer repeats is no longer a
  // late echo of an input made before the deck's last refusal.
  forgetRefusalEchoes();
  /** @type {Array<Readonly<Record<string, unknown>>>} */
  let deferredDraftsToDispatch = [];
  /** @type {Readonly<Record<string, unknown>> | null} */
  let deferredSubmitToDispatch = null;
  /** @type {Record<string, unknown> | null} */
  let commandToResend = null;
  const previousAuthority = state.authority;
  const request = commandRequest(command);
  /** @type {{current: {kind: "success" | "stale", payload: any} | null}} */
  const commandOutcome = { current: null };
  const replayWindow =
    command.command_type === "review_replay"
      ? window.open("about:blank", "_blank")
      : null;
  if (replayWindow !== null) {
    replayWindow.opener = null;
    // The Replay Viewer holds its page until Agent POV is ready, so the tab
    // shows a dark "Opening Replay" page in this client's colors meanwhile.
    const tokens = getComputedStyle(
      document.querySelector(".bc") ?? document.documentElement,
    );
    writeOpeningReplayPage(replayWindow, {
      background: tokens.getPropertyValue("--ink"),
      text: tokens.getPropertyValue("--text"),
      muted: tokens.getPropertyValue("--muted"),
      font: getComputedStyle(document.body).fontFamily,
    });
  }
  let replayOpened = false;
  try {
    const installPromise = presentationInstallation.installFromCommand({
      reason:
        command.command_type === "set_view" ? "live_audience_change" : "live_command",
      pendingPolicy:
        command.command_type === "set_view" ? "clear" : "retain_last_authorized",
      /**
       * Post the captured live request and remember its response.
       * Resolve to success or, for HTTP 409, a stale outcome. Other API/transport
       * errors reject. The transaction's join/recovery callbacks own resynchronization.
       */
      sendCommand: async () => {
        try {
          const payload = await postCommand(state.token, request);
          commandOutcome.current = { kind: "success", payload };
          return commandOutcome.current;
        } catch (error) {
          if (error instanceof DebuggerApiError && error.status === 409) {
            commandOutcome.current = { kind: "stale", payload: error.payload };
            return commandOutcome.current;
          }
          throw error;
        }
      },
      /**
       * Prepare the supplied live command outcome for authority installation.
       * Fetch the presentation and require a valid join. Saving and opening a
       * separate Viewer leave this live product unchanged.
       */
      joinCommandResult: async (outcome) => {
        const joined = await extractJoinedFrame(
          outcome.payload,
          await getCurrentPresentation(state.token),
        );
        if (joined === null) {
          throw new TypeError("Live response has no joinable transport candidate.");
        }
        return prepareJoinedAuthority(joined, {
          previousAuthority,
          continuityResult: "stale_resync",
        });
      },
      /**
       * Fetch and prepare current authority after a stale live command.
       * Return a promise for a matched frame/presentation prepared against the previous
       * view. Fetch/join/preparation failures reject; this does not resubmit the action.
       */
      getJoined: async () =>
        prepareJoinedAuthority(await getCurrentAuthority(), {
          previousAuthority,
          continuityResult: "stale_resync",
        }),
    });
    render();
    const installOutcome = await installPromise;
    if (installOutcome.status === "superseded") {
      return;
    }
    state.offline = false;
    state.resyncRequired = false;
    const payload = commandOutcome.current?.payload;
    const stale = commandOutcome.current?.kind === "stale";
    const notice = extractNotice(payload);
    // A plain frame update (after Lock In, say) is routine and makes no notice.
    const routineFrameUpdate =
      !stale &&
      !installOutcome.resynchronized &&
      notice === null &&
      payload?.result !== "duplicate";
    const afterFacts = ownOrderFacts(state.presentation, state.frame);
    if (afterFacts?.actorManual) lastManualPublicId = afterFacts.actorId;
    // A host refusal or focus notice is said in the vision's words; an applied
    // command (or a clear with nothing to clear) says what it did.
    const clearsTarget =
      (command.command_type === "keyboard" &&
        String(command.key).toLowerCase() === "escape") ||
      (command.command_type === "battlefield_pointer" &&
        command.button === "secondary");
    // A step or No Combat that was already staged changes nothing, but the
    // vision still says what is staged ("Step: Stay.").
    const restatesOrder =
      (command.command_type === "keyboard" &&
        typeof command.key === "string" &&
        command.key.length === 1 &&
        "wasdqezcx0".includes(command.key.toLowerCase())) ||
      (command.command_type === "roster_selection" &&
        command.role === "control" &&
        asArray(state.frame?.combat_configuration?.manual_slots).includes(
          command.global_slot,
        ));
    // A resynchronized answer is normal when the policies finish the turn
    // before the page has read the command's own result (common in Agent
    // POV): the command was applied once and the latest state is shown, so
    // Lock In still says its own sentence.
    const ownSentence = stale
      ? null
      : installOutcome.resynchronized
        ? isSubmissionCommand(command)
          ? statusSentence({ kind: "lock_in", watch: beforeFacts?.mode === "watch" })
          : null
        : (hostNoticeSentence(notice, afterFacts) ??
          (payload?.result === "applied" ||
          (payload?.result === "no_op" && (clearsTarget || restatesOrder))
            ? acceptedCommandSentence(command, beforeFacts, afterFacts, source)
            : null));
    // Exit, and a restart pressed while the game-over save ran (Save And
    // Replace), keep their meaning on a newer frame: when the save ended just
    // before the command arrived, the server answers "stale", so the page
    // sends the same command once more from the fresh frame, and says
    // nothing about the stale answer.
    if (stale && resendsAfterStale(command) && !resentAfterStale.has(command)) {
      commandToResend = command;
    }
    if (liveCommandTransaction.lateRefusal !== null) {
      setNotice(liveCommandTransaction.lateRefusal, "warning");
    } else if (commandToResend !== null) {
      // The same command goes out again at once; its answer says the rest.
    } else if (quiet && !stale && payload?.result !== "duplicate") {
      // The page's own Run: an accepted turn leaves the line as it is.
    } else if (ownSentence !== null) {
      clearNotice(ownSentence);
    } else if (routineFrameUpdate) {
      if (!quietPointer) clearNotice();
    } else {
      setNotice(
        stale
          ? payload?.error_code === "command_id_conflict"
            ? "The service rejected a command-ID conflict. Its coherent latest pair was installed; the command was not retried."
            : "This tab was stale. Its coherent latest pair was installed; the command was not retried."
          : installOutcome.resynchronized
            ? "Your command was applied once. The game had moved on, so the page now shows the latest state."
            : (notice ?? "Duplicate command recognized; it was not applied again."),
        stale || installOutcome.resynchronized || payload?.result === "duplicate"
          ? "warning"
          : "success",
      );
    }
    if (!stale && command.command_type === "review_replay") {
      const replayUrl = extractReplayUrl(payload);
      const replayFrame = payload?.frame;
      const replayMatchesInstalledEpisode =
        replayFrame !== undefined &&
        state.frame !== null &&
        replayFrame.session_id === state.frame.session_id &&
        replayFrame.run_generation === state.frame.run_generation &&
        replayFrame.episode_id === state.frame.episode_id;
      if (replayUrl !== null && !replayMatchesInstalledEpisode) {
        setNotice(
          "The match changed while opening the replay. The earlier replay was saved; its link was not attached to this match.",
          "warning",
        );
      } else if (replayUrl !== null) {
        elements.recordingReplayLink.href = replayUrl;
        openedReplayEpisodeId = replayFrame.episode_id;
        if (replayWindow !== null && !replayWindow.closed) {
          replayWindow.location.replace(replayUrl);
          replayOpened = true;
        } else {
          setNotice(
            "Replay ready. Use Open Saved Replay In New Tab below if your browser blocked the new tab.",
            "info",
          );
        }
      }
    }
    if (commandResponseSchedulesShutdown(command, payload)) {
      acceptExit(payload);
    }
    if (
      (liveCommandTransaction.deferredDrafts.length > 0 ||
        liveCommandTransaction.deferredSubmit !== null) &&
      !stale &&
      !installOutcome.resynchronized &&
      (payload?.result === "applied" || payload?.result === "no_op") &&
      !state.shuttingDown &&
      !state.resyncRequired &&
      !state.offline &&
      isAuthorizedPresentationFrame(state.presentation) &&
      installedAuthorityIsCoherent() &&
      !isTerminal(state.frame)
    ) {
      deferredDraftsToDispatch = [...liveCommandTransaction.deferredDrafts];
      deferredSubmitToDispatch = liveCommandTransaction.deferredSubmit;
    }
  } catch (error) {
    if (error instanceof ProductIdentityMismatchError) {
      failClosedProductIdentity(error);
    } else {
      const status = error instanceof DebuggerApiError ? error.status : 0;
      const payload = commandOutcome.current?.payload;
      if (isPresentationJoinRace(error)) {
        clearPresentationAuthority("live_presentation_identity_mismatch");
      } else if (status === 401 || status === 403) {
        clearPresentationAuthority("live_authorization_failure");
      }
      if (commandResponseSchedulesShutdown(command, payload)) {
        // The answer's own recording status is newer than the installed
        // frame's (Exit waited for the save): the pill, popover and save button show
        // it, so they never stay on "Saving Replay…" after a save.
        acceptExit(payload);
        return;
      }
      state.offline =
        error instanceof DebuggerApiError &&
        (status === 0 || status === 401 || status === 403);
      state.resyncRequired = true;
      setNotice(
        status === 401 || status === 403
          ? "Debugger capability is invalid. Reopen the exact URL printed by the Python launcher."
          : error instanceof Error
            ? `${error.message} Reconnect before sending another command.`
            : "Debugger command failed. Reconnect before sending another command.",
        "error",
      );
    }
  } finally {
    if (recordingHintTimer !== null) window.clearTimeout(recordingHintTimer);
    state.busy = false;
    if (activeLiveCommandTransaction === liveCommandTransaction) {
      activeLiveCommandTransaction = null;
    }
    if (replayWindow !== null && !replayOpened) replayWindow.close();
    // A queued order goes out at once, so the page keeps its in-flight look
    // instead of painting one idle frame (Lock In enabled, hover card gone)
    // between the two requests; the last request in the chain repaints.
    if (deferredDraftsToDispatch.length === 0 && deferredSubmitToDispatch === null) {
      render();
    }
    // The policies' answer is read at the busy cadence, not after the idle
    // second scheduled before this command.
    hurryLivePoll();
    if (liveCommandTransaction.releaseFocusAfterSettlement) {
      releaseBattlefieldFocus();
    }
  }
  if (commandToResend !== null) {
    const again = { ...commandToResend };
    resentAfterStale.add(again);
    await dispatchCommand(again, { source, quiet });
    return;
  }
  // Queued inputs (battlefield keys and clicks from retainFencedCommand, focus
  // changes from retainFocusChange) keep the "field" source: a queued Esc
  // still says keys are off.
  if (deferredDraftsToDispatch.length > 0) {
    await dispatchCommand(deferredDraftsToDispatch[0], {
      deferredDrafts: deferredDraftsToDispatch.slice(1),
      deferredSubmit: deferredSubmitToDispatch,
      source: "field",
    });
  } else if (deferredSubmitToDispatch !== null) {
    await dispatchCommand(deferredSubmitToDispatch, { source: "field" });
  } else {
    return;
  }
  // A queued command the page refused before sending did not repaint.
  if (!state.busy) render();
}

/** Fetch one coherent live snapshot, or the existing paired replay resources. */
function getCurrentAuthority() {
  return isReplayMode()
    ? getCurrentFrameAndPresentation(state.token)
    : getCurrentLiveSnapshot(state.token);
}

/**
 * Load and install the current matched transport, presentation, and replay timeline.
 * Skip busy or shutting-down pages, preserve permitted old display while loading, and show
 * handled failures with resync required. The page is also offline, and the
 * pill then reads "Disconnected" (shell.js connectionPill puts it before
 * "Resync Required"), when the failure is not an answer from the server (no
 * answer at all, or a frame the page could not accept) or the server refused
 * the link (401 or 403). A successful plain load clears the
 * top-bar notice (clearNotice); a load that had to discard a mixed pair says so.
 * No load moves keyboard focus (replay keys listen on the whole document), so
 * no help card opens at rest. This never retries a simulator command.
 */
async function loadCurrentFrame() {
  if (productIdentity === null) {
    state.offline = true;
    state.resyncRequired = true;
    setNotice(
      `Startup blocked: ${startupProductIdentityError ?? "Product bootstrap is unavailable."} Reopen the exact URL printed by the Python launcher.`,
      "error",
    );
    render();
    return;
  }
  if (state.busy || state.shuttingDown) {
    return;
  }
  invalidateReplayArtifactAction();
  state.busy = true;
  if (isReplayMode()) {
    replayPlayback.pause("reconnect");
  }
  clearNotice("Loading the current match…");
  const previousFrame = state.frame;
  const previousAuthority = state.authority;
  const retainLastAuthorized = installedPresentationAuthority() !== null;
  try {
    const installPromise = presentationInstallation.installFromGet({
      reason: previousFrame === null ? "initial_authority" : "reconnect_authority",
      pendingPolicy: retainLastAuthorized ? "retain_last_authorized" : "clear",
      /**
       * Fetch current matched authority for the initial load or reconnect.
       * Prepare it with stale_resync continuity relative to the retained previous view.
       * The promise rejects if fetching, matching or preparation fails.
       */
      getJoined: async () =>
        prepareJoinedAuthority(await getCurrentAuthority(), {
          previousAuthority,
          continuityResult: "stale_resync",
        }),
    });
    render();
    const outcome = await installPromise;
    if (outcome.status === "superseded") {
      return;
    }
    const frame = outcome.joined.transport;
    const firstReplayLoad =
      frame.viewer_mode === "replay" && previousFrame?.viewer_mode !== "replay";
    if (firstReplayLoad) {
      choreographer.clear("replay_handoff");
    }
    if (frame.viewer_mode === "replay") {
      replayPlayback.installCursor(frame.cursor);
      replayPlayback.setConnected(true);
    }
    state.offline = false;
    state.resyncRequired = false;
    // A plain load is routine: the connection status already says Online.
    if (outcome.resynchronized) {
      setNotice(
        "A mixed authority GET was discarded; one fresh pair was installed.",
        "success",
      );
    } else {
      clearNotice();
      // The first live load remembers the Manual champion in focus. The
      // first sentence itself has one owner: the deck's startGreeting
      // (tactics-turn.js), which counts the Manual champions on each team.
      const facts =
        previousFrame === null ? ownOrderFacts(state.presentation, state.frame) : null;
      if (facts?.actorManual) lastManualPublicId = facts.actorId;
    }
  } catch (error) {
    if (error instanceof ProductIdentityMismatchError) {
      failClosedProductIdentity(error);
    }
    const status = error instanceof DebuggerApiError ? error.status : 0;
    if (error instanceof TypeError) {
      clearPresentationAuthority("reconnect_invalid_presentation_identity");
    } else if (isPresentationJoinRace(error)) {
      clearPresentationAuthority("reconnect_presentation_identity_mismatch");
    } else if (status === 401 || status === 403) {
      clearPresentationAuthority("reconnect_authorization_failure");
    }
    state.offline = status === 0 || status === 401 || status === 403;
    state.resyncRequired = true;
    if (isReplayMode()) {
      replayPlayback.setConnected(false);
    }
    setNotice(
      status === 401 || status === 403
        ? "This link no longer works. Open the link printed in the terminal again."
        : error instanceof Error
          ? "Could not load the match. Press Reconnect."
          : "Could not load debugger frame.",
      "error",
    );
  } finally {
    state.busy = false;
    render();
  }
  // The first replay load moves no focus: replay keys listen on the whole
  // document, and a moved focus would open the timeline's help card at rest.
}

/**
 * Send a draft-control command after handing keyboard ownership to the battlefield.
 * restoreFocusTo defaults to null. When supplied, restore that connected enabled
 * control after completion only if the battlefield still owns focus; do not steal
 * focus the user moved elsewhere. The request runs asynchronously with no return value.
 *
 * @param {Record<string, unknown>} command
 * @param {{
 *   restoreFocusTo?: HTMLElement | null,
 * }} options
 */
function dispatchCommandFromDraftControl(command, { restoreFocusTo = null } = {}) {
  elements.battlefield.focus({ preventScroll: true });
  const completion = dispatchCommand(command, { source: "control" });
  if (restoreFocusTo === null) {
    void completion;
    return;
  }
  void completion.finally(() => {
    if (
      document.activeElement === elements.battlefield &&
      restoreFocusTo.isConnected &&
      !restoreFocusTo.matches(":disabled") &&
      restoreFocusTo.getAttribute("aria-disabled") !== "true"
    ) {
      restoreFocusTo.focus({ preventScroll: true });
    }
  });
}

/**
 * Work out what Tab or Shift+Tab does in live play with Manual champions
 * (owner's ruling, 3 October 2026): it gives the orders to your next (or
 * previous) champion that can take orders, in slot order, and never to a
 * policy's champion. The deck's ‹ Prev and Next › send the same key, so they
 * follow the same rule. "Can take orders" means a Manual slot of the match
 * setup (combat_configuration.manual_slots; with one Manual champion on each
 * team, both) that the shown audience's scene does not draw as fallen
 * (life_state other than "alive"). Your champion on the other team that this
 * Agent POV cannot see counts as ready: the page has no fact about it, and
 * after the switch its own view shows how it fares.
 *
 * The current champion is the one looked at (the transport frame's
 * focused_public_agent_id, Agent POV Watch Only) or else the one the
 * audience's inspection is about. Public IDs map to command slots through
 * the identity rows, as a portrait click does. Nothing hidden from the POV
 * team is read: only the match setup, the audience's own scene and
 * inspection. The choice is plumbing for which champion you command; it is
 * never a policy input.
 *
 * @param {Readonly<Record<string, unknown>>} command The command about to be
 *   sent.
 * @returns {{kind: "pass"}
 *   | {kind: "send", command: Readonly<Record<string, unknown>>}
 *   | {kind: "stay", className: string}
 *   | {kind: "none"}} "pass" for any other command, or where Tab keeps
 *   Python's own walk: the Replay Viewer, scripted inspection, and Policies
 *   Only (no Manual slot; Tab then looks at every champion in sight).
 *   "send": a roster_selection (role "control") for the next slot. "stay":
 *   the champion in focus is the only one that can take orders (className
 *   names it). "none": every Manual champion is fallen. Sends nothing itself.
 */
function ownChampionTabStep(command) {
  if (
    command.command_type !== "keyboard" ||
    command.key !== "Tab" ||
    command.ctrl_key === true ||
    command.alt_key === true ||
    command.meta_key === true ||
    isReplayMode() ||
    liveScriptedInspectionOnly() ||
    !isAuthorizedPresentationFrame(state.presentation) ||
    !isRecord(state.frame)
  ) {
    return { kind: "pass" };
  }
  const manualSlots = asArray(state.frame.combat_configuration?.manual_slots).filter(
    (slot) => Number.isInteger(slot),
  );
  if (manualSlots.length === 0) return { kind: "pass" };
  const rows = authorizedPresentationIdentityRows(state.presentation);
  const agents = asArray(
    authorizedPresentationSceneView(state.presentation)?.agents,
  ).filter(isRecord);
  /** @param {unknown} slot @returns {Record<string, any> | null} */
  const sceneAgentOf = (slot) => {
    const row = rows.find((identity) => identity.command_global_slot === slot);
    return row === undefined
      ? null
      : (agents.find((agent) => agent.public_agent_id === row.public_agent_id) ?? null);
  };
  const fallenSlots = manualSlots.filter((slot) => {
    const agent = sceneAgentOf(slot);
    return agent !== null && agent.life_state !== "alive";
  });
  const inspection = authorizedPresentationInspectionState(
    state.presentation,
  ).inspection;
  const focusId =
    typeof state.frame.focused_public_agent_id === "string"
      ? state.frame.focused_public_agent_id
      : isRecord(inspection) && typeof inspection.actor_public_agent_id === "string"
        ? inspection.actor_public_agent_id
        : null;
  const focusSlot =
    focusId === null
      ? null
      : (rows.find((identity) => identity.public_agent_id === focusId)
          ?.command_global_slot ?? null);
  const currentSlot = Number.isInteger(focusSlot) ? Number(focusSlot) : null;
  const next = nextOrderSlot({
    manualSlots,
    currentSlot,
    fallenSlots,
    direction: command.shift_key === true ? -1 : 1,
  });
  if (next === null) return { kind: "none" };
  if (next === currentSlot) {
    return {
      kind: "stay",
      className: String(sceneAgentOf(next)?.class_name ?? "champion"),
    };
  }
  return {
    kind: "send",
    command: Object.freeze({
      command_type: "roster_selection",
      role: "control",
      global_slot: next,
    }),
  };
}

/**
 * Return whether a battlefield key still goes to Python while orders are
 * closed: at game over, after Finish And Save, or in scripted inspection. Only
 * R, G and, in scripted inspection only, N pass: R (Restart; it asks Save,
 * Discard or Cancel first when play is unsaved, and
 * restarts at once when the game is saved) and G (ranges; in Agent POV G stays
 * page-local) pass. Requires an idle, connected page with coherent authority
 * and no policy work running; plain keys only (no Ctrl, Alt or Meta; Shift+R
 * is not Restart). The usual gates in dispatchCommand still apply. N (one
 * tick) is a Lock In key, so like Space it does not pass while orders are
 * closed, except in scripted inspection (liveScriptedInspectionOnly), where
 * plain N is the next scripted frame, as the deck's button sends it.
 *
 * @param {Readonly<Record<string, unknown>>} command A keyboard command.
 * @returns {boolean}
 */
function liveKeyAllowedWhileClosed(command) {
  const key = typeof command.key === "string" ? command.key.toLowerCase() : "";
  return (
    !isReplayMode() &&
    command.command_type === "keyboard" &&
    (key === "g" ||
      (key === "r" && command.shift_key !== true) ||
      (key === "n" && command.shift_key !== true && liveScriptedInspectionOnly())) &&
    command.ctrl_key !== true &&
    command.alt_key !== true &&
    command.meta_key !== true &&
    isAuthorizedPresentationFrame(state.presentation) &&
    installedAuthorityIsCoherent() &&
    !state.frame?.system_operation &&
    !state.busy &&
    !state.shuttingDown &&
    !state.resyncRequired &&
    !state.offline
  );
}

/**
 * Move focus to the first enabled command button the deck shows, or Help when
 * none shows (for example while a request is out and the deck's buttons are
 * disabled). A hidden button, such as Watch Only's Back while that overlay is
 * closed, cannot take focus, so it is skipped: otherwise focus would stay on
 * the arena and keys would stay on after Esc said they are off. Use
 * preventScroll so releasing battlefield keyboard control does not move the
 * page.
 */
function releaseBattlefieldFocus() {
  const firstCommand =
    /** @type {HTMLButtonElement[]} */ ([
      ...(elements.commandDeck?.querySelectorAll("button:not([disabled])") ?? []),
    ]).find((button) => button.getClientRects().length > 0) ?? null;
  const focusTarget = firstCommand ?? elements.helpButton;
  focusTarget.focus({ preventScroll: true });
}

bindBattlefieldControls({
  battlefield: elements.battlefield,
  /**
   * Convert the supplied SVG-local point into battlefield world coordinates.
   * Delegate the coordinate transform to the installed renderer. Return its point
   * or null when its current view cannot provide a valid transform.
   */
  toWorldPoint: (point) => battlefieldRenderer.toWorldPoint(point),
  /**
   * Dispatch the supplied normalized live battlefield command.
   * Return the shared dispatch promise. That handler owns action authorization,
   * request fencing, server calls and error notices.
   */
  onCommand: (command) => dispatchCommand(command, { source: "field" }),
  /**
   * Open or close How To Play (? on the battlefield) without a server request.
   */
  onHelp: () => toggleHowToPlay(),
  isInteractive: liveBattlefieldCommandsInteractive,
  onFencedCommand: retainFencedCommand,
  allowsWhileClosed: liveKeyAllowedWhileClosed,
  /**
   * Return whether live control owns Space while a command is fenced.
   * Replay mode leaves its Space handling to the replay keyboard controller.
   */
  ownsFencedSpaceDefault: () => !isReplayMode(),
  /**
   * Return whether a Tab that is not kept while orders are closed stays in
   * the arena (live play). The Replay Viewer returns false: its own Tab steps
   * agents (bindReplayAgentStepping).
   */
  ownsFencedTabDefault: () => !isReplayMode(),
  /**
   * Return whether the focused live battlefield runs local keys (V and F) for
   * the BattleClient and DevClient. The Replay Viewer returns false: its one
   * keyboard dispatcher (bindReplayTimelineControls) owns V, B and F there, so
   * no job runs twice.
   */
  ownsLocalCommands: () => !isReplayMode(),
  onReleaseFocus: releaseBattlefieldFocus,
});

/**
 * Show whether keys are on: they are on while the battlefield has keyboard
 * focus (a click on the arena gives it; Esc and a click elsewhere take it,
 * but a click on a top-bar button or a view switch keeps it).
 * Writes the message line's Keys chip ("Keys On" or "Keys Off · Click The
 * Arena", class `off`) and the `keys-on` class on #arena (its silver frame),
 * as on the vision page. BattleClient and DevClient only; in the Replay Viewer
 * the chip is not shown and #arena keeps no class.
 *
 * When keys turn on, a message line that still says they are off is made true
 * again: "Keys are off until you click the arena again." empties, and
 * "Target cleared. Keys are off until you click the arena again." becomes
 * "Target cleared.". Any other message stays.
 *
 * Once Exit was accepted (state.shuttingDown) no key works any more, so the
 * chip reads "Keys Off" (with no invitation to click the arena) and the
 * arena loses its keys-on frame, wherever focus is.
 *
 * @returns {void}
 */
function paintKeysState() {
  const on =
    !isReplayMode() &&
    !state.shuttingDown &&
    document.activeElement === elements.battlefield;
  elements.keysChip.classList.toggle("off", !on);
  elements.keysText.textContent = on
    ? "Keys On"
    : state.shuttingDown
      ? "Keys Off"
      : "Keys Off · Click The Arena";
  elements.arena.classList.toggle("keys-on", on);
  if (!on) return;
  const shown = elements.statusMessage.textContent ?? "";
  if (shown === statusSentence({ kind: "keys_off" })) {
    clearNotice();
  } else if (shown === statusSentence({ kind: "cleared", keysOff: true })) {
    clearNotice(statusSentence({ kind: "cleared", keysOff: false }));
  }
}

elements.battlefield.addEventListener("focus", paintKeysState);
elements.battlefield.addEventListener("blur", paintKeysState);

/**
 * The top-bar button whose mouse press last kept keys on, or null. Read and
 * cleared by giveKeysBackAfterTopBarDialog.
 *
 * @type {Element | null}
 */
let topBarPressKeptKeys = null;

/**
 * The buttons whose mouse press keeps keys on (keepKeysOnForTopBarPress):
 * every top-bar button, the arena's two on-screen keys (Everyone In View and
 * Frame Explanation), the Verdict card's two buttons and the Match Summary
 * sheet's Close.
 */
const KEYS_KEPT_BUTTONS =
  ".app-header button, .arena-buttons button, .herald-verdict-actions__button, .match-summary-sheet__close";

/**
 * Keep keys on when a top-bar button is pressed with the mouse (owner, 3
 * October 2026): Agent POV, Oracle View, the Recording pill, How To Play,
 * Visual Filters, Finish And Save, Restart and Exit, the arena's Everyone In
 * View and Frame Explanation keys (4 Oct 2026), plus the game-over Verdict
 * card's
 * Match Summary and Show Arena buttons and the Match Summary sheet's Close
 * (KEYS_KEPT_BUTTONS). While the battlefield has keyboard focus, the mouse
 * press on one of these buttons does not move focus to that button (its
 * default is prevented); the click still
 * does the button's job, and the button is remembered (topBarPressKeptKeys)
 * so a dialog it opens gives the keys back (giveKeysBackAfterTopBarDialog).
 * A press with keys off forgets it. BattleClient and DevClient only; the
 * Replay Viewer is left alone. Keyboard users still reach every button with
 * the browser's own Tab order outside the battlefield.
 *
 * @param {MouseEvent} event A mousedown, seen in the capture phase.
 * @returns {void}
 */
function keepKeysOnForTopBarPress(event) {
  if (isReplayMode()) return;
  const target = event.target instanceof Element ? event.target : null;
  const button = target?.closest(KEYS_KEPT_BUTTONS) ?? null;
  if (button === null) return;
  if (document.activeElement !== elements.battlefield) {
    topBarPressKeptKeys = null;
    return;
  }
  event.preventDefault();
  topBarPressKeptKeys = button;
}

/**
 * Give keys back after a dialog opened from the top bar closes (owner, 3
 * October 2026). A modal dialog such as the Restart dialog takes focus while it is
 * open, and on close it hands focus to its own top-bar button. When that
 * button's press had kept keys on (topBarPressKeptKeys), focus goes back to the
 * battlefield right after the dialog's own close work, so keys are on again;
 * if the player moved focus somewhere else meanwhile, it stays there. Runs for
 * a `close` event seen in the capture phase; it clears the remembered button.
 * BattleClient and DevClient only.
 *
 * @param {Event} event A dialog's close event.
 * @returns {void}
 */
function giveKeysBackAfterTopBarDialog(event) {
  const button = topBarPressKeptKeys;
  if (
    isReplayMode() ||
    button === null ||
    !(event.target instanceof HTMLDialogElement)
  ) {
    return;
  }
  topBarPressKeptKeys = null;
  window.setTimeout(() => {
    const active = document.activeElement;
    if (active === button || active === null || active === document.body) {
      elements.battlefield.focus({ preventScroll: true });
    }
  }, 0);
}

document.addEventListener("mousedown", keepKeysOnForTopBarPress, true);
document.addEventListener("close", giveKeysBackAfterTopBarDialog, true);

/**
 * Make a plain click on the arena only turn keys on, never pin a card
 * (owner, 3 October 2026). The tooltip controller pins a symbol's card (a
 * stone, a waystone, a mark) when it is clicked; in the BattleClient and
 * DevClient a click on the arena is the click that gives keys, so a pin on a
 * mark inside the battlefield is dropped right after the click, and the card
 * then shows only while the pointer rests on its mark (hover), as before the
 * click. Pins outside the battlefield and every Replay Viewer pin are kept.
 * Uses only the controller's public pinnedOwner, hide and refresh.
 *
 * @returns {void}
 */
function unpinArenaCardAfterClick() {
  if (isReplayMode()) return;
  const pinned = tooltipController.pinnedOwner();
  if (pinned === null || !elements.battlefield.contains(pinned)) return;
  tooltipController.hide();
  tooltipController.refresh();
}

elements.battlefield.addEventListener("click", unpinArenaCardAfterClick);

elements.battlefield.addEventListener(
  "pointerdown",
  (/** @type {PointerEvent} */ event) => {
    if (event.target instanceof Element) {
      const tooltipOwner = event.target.closest("[data-tooltip-owner]");
      const agent = event.target.closest(".agent[data-presentation-key]");
      if (tooltipOwner !== null && tooltipOwner !== agent) {
        // A right click on a mark still clears the target (the battlefield
        // controls send the secondary pointer). Any other click on a mark
        // keeps its tooltip, sends nothing, and still turns keys on: the
        // arena, not the mark, takes keyboard focus, as on the vision page.
        if (event.button === 2) return;
        if (!isReplayMode()) {
          event.preventDefault();
          elements.battlefield.focus({ preventScroll: true });
        }
        event.stopImmediatePropagation();
        return;
      }
    }
    if (event.button !== 0) {
      return;
    }
    const activation = authorizedAgentActivationFromTarget(event.target);
    const inspectionOnlyCorpse =
      activation?.effect === "local_inspection" &&
      activation.agent.life_state === "corpse";
    if (
      !inspectionOnlyCorpse &&
      (event.shiftKey || event.ctrlKey || event.metaKey || event.altKey)
    ) {
      return;
    }
    if (activation === null) {
      sayOwnChampionClicked(event.target);
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
    elements.battlefield.focus({ preventScroll: true });
    activateAuthorizedAgent(activation.presentationKey);
  },
  true,
);

/**
 * Say "Giving orders to your <Class>." when a plain click lands on your own
 * champion that already has the orders (no focus change, so no command says
 * it). Reads the clicked body's presentation key, the audience's own scene
 * (your champion is always in it) and the Manual slots of the match setup.
 * Does nothing for any other body or outside live play.
 *
 * @param {EventTarget | null} target The pointerdown target.
 * @returns {void}
 */
function sayOwnChampionClicked(target) {
  if (isReplayMode() || !(target instanceof Element)) return;
  const key = target
    .closest(".agent[data-presentation-key]")
    ?.getAttribute("data-presentation-key");
  if (typeof key !== "string") return;
  const scene = authorizedPresentationSceneView(state.presentation);
  const agent = asArray(scene?.agents).find(
    (candidate) => isRecord(candidate) && candidate.presentation_key === key,
  );
  const manual = asArray(state.frame?.combat_configuration?.manual_slots);
  if (isRecord(agent) && manual.includes(Number(agent.public_agent_id))) {
    clearNotice(
      statusSentence({ kind: "focus_own", className: String(agent.class_name) }),
    );
  }
}

elements.battlefield.addEventListener(
  "keydown",
  (/** @type {KeyboardEvent} */ event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    if (event.target instanceof Element) {
      const tooltipOwner = event.target.closest("[data-tooltip-owner]");
      const agent = event.target.closest(".agent[data-presentation-key]");
      if (tooltipOwner !== null && tooltipOwner !== agent) {
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }
    }
    const activation = authorizedAgentActivationFromTarget(event.target);
    if (activation !== null) {
      event.preventDefault();
      event.stopImmediatePropagation();
      activateAuthorizedAgent(activation.presentationKey);
    }
  },
  true,
);

elements.viewSelect.addEventListener("change", () => {
  const command = {
    command_type: "set_view",
    view_mode: elements.viewSelect.value,
  };
  if (isReplayMode()) {
    void dispatchReplayCommand(command);
  } else {
    void dispatchCommand(command);
  }
});

bindCommandDeck();

/**
 * Finish And Save from the top bar, the Recording popover or the Stopped
 * overlay: close the popover, then send one finish_and_review. Python ends the
 * game and writes the replay file; the pill then reads Saved and the top
 * bar's button turns into Open Replay.
 */
function finishAndSave() {
  if (elements.recordingPopover.matches(":popover-open")) {
    elements.recordingPopover.hidePopover();
  }
  void dispatchCommand({ command_type: "finish_and_review" });
}

for (const button of [elements.recordingPopFinishButton, elements.stoppedSaveButton]) {
  button.addEventListener("click", finishAndSave);
}

/**
 * Answer a click on the top bar's save button by its data-action (set by
 * paintSaveButton): "finish" saves (finishAndSave); "open" opens the saved
 * replay, through review_replay the first time (the server starts the Replay
 * Viewer; the new tab shows the "Opening Replay" page meanwhile) and after
 * that by opening the same Replay Viewer link again in a new tab; "retry"
 * sends retry_save. "Saving Replay…" has no action and is disabled.
 */
function saveButtonClick() {
  const button = elements.recordingFinishButton;
  if (button.disabled) return;
  const action = button.dataset.action;
  if (action === "finish") {
    finishAndSave();
  } else if (action === "open") {
    if (recordingStatus()?.review_available === true) {
      void dispatchCommand({ command_type: "review_replay" });
    } else if (elements.recordingReplayLink.hasAttribute("href")) {
      window.open(elements.recordingReplayLink.href, "_blank", "noopener");
    }
  } else if (action === "retry") {
    void dispatchCommand({ command_type: "retry_save" });
  }
}

elements.recordingFinishButton.addEventListener("click", saveButtonClick);

// Watch Only: Back To Your <Class> gives the orders back to your champion (the
// last Manual champion you ordered, else the first Manual slot). The own
// champion is always in the audience's scene, so selectAgentFromHud finds it.
elements.watchOnlyBackButton.addEventListener("click", () => {
  const facts = ownOrderFacts(state.presentation, state.frame);
  const manualSlots = asArray(state.frame?.combat_configuration?.manual_slots);
  const own =
    lastManualPublicId ?? (manualSlots.length > 0 ? String(manualSlots[0]) : null);
  if (facts !== null && own !== null) selectAgentFromHud(own);
});

elements.recordingReviewButton.addEventListener("click", () => {
  void dispatchCommand({ command_type: "review_replay" });
});

elements.recordingRetryButton.addEventListener("click", () => {
  void dispatchCommand({ command_type: "retry_save" });
});

/**
 * Validate the Save As field and send its recording command when valid.
 * On an invalid basename, show the filename rule and focus the input. The host
 * performs persistence; this handler does not write a browser-local replay file.
 */
function dispatchRecordingSaveAs() {
  const command = recordingSaveAsCommand(elements.recordingSaveAsInput.value);
  if (!command) {
    setNotice(
      "Save As requires a basename ending in .marlbg-replay.json; paths, spaces, and hidden filenames are not accepted.",
      "warning",
    );
    renderConnection();
    elements.recordingSaveAsInput.focus({ preventScroll: true });
    return;
  }
  void dispatchCommand(command);
}

elements.recordingSaveAsButton.addEventListener("click", dispatchRecordingSaveAs);
elements.recordingSaveAsInput.addEventListener(
  "keydown",
  (/** @type {KeyboardEvent} */ event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    dispatchRecordingSaveAs();
  },
);

elements.recordingDiscardDialog.addEventListener("close", () => {
  // Cancel (or Esc) closes the dialog with no request: say so, as the vision does.
  if (pendingRecordingReplacement !== null && !isReplayMode()) {
    clearNotice(statusSentence({ kind: "restart_cancelled" }));
  }
  pendingRecordingReplacement = null;
});

elements.recordingDiscardDialog.addEventListener("cancel", () => {
  if (pendingRecordingReplacement !== null && !isReplayMode()) {
    clearNotice(statusSentence({ kind: "restart_cancelled" }));
  }
  pendingRecordingReplacement = null;
});

elements.recordingDiscardConfirmButton.addEventListener("click", () => {
  void confirmRecordingReplacement("discard");
});
elements.recordingSaveReplaceButton.addEventListener("click", () => {
  void confirmRecordingReplacement("save");
});

elements.exitButton.addEventListener("click", () => {
  invalidateReplayArtifactAction();
  if (isReplayMode()) {
    void dispatchReplayCommand({ command_type: "exit" });
  } else {
    void dispatchCommand({ command_type: "exit" });
  }
});

elements.reconnectButton.addEventListener("click", () => {
  invalidateReplayArtifactAction();
  void loadCurrentFrame();
});

if (!isReplayMode()) {
  // Private setup supplies an explicit callback; the public product never imports
  // private code or tries to discover private sources.
  document.addEventListener("marl-client-recording-replacement", (event) => {
    const detail = event instanceof CustomEvent ? event.detail : null;
    if (
      !isRecord(detail) ||
      typeof detail.session_id !== "string" ||
      !Number.isSafeInteger(detail.run_generation) ||
      !Number.isSafeInteger(detail.revision) ||
      typeof detail.label !== "string" ||
      typeof detail.perform !== "function"
    )
      return;
    if (
      showRecordingReplacementDialog({
        session_id: detail.session_id,
        run_generation: detail.run_generation,
        revision: detail.revision,
        label: detail.label,
        perform: detail.perform,
      })
    )
      event.preventDefault();
  });
  document.addEventListener("marl-combat-configuration", (event) => {
    const command = requestedCombatConfigurationCommand(
      event instanceof CustomEvent ? event.detail : null,
    );
    if (command !== null) {
      void dispatchCommand(command);
    } else {
      publishInstalledCombatConfiguration(state.frame);
    }
  });
  // A rejected load has no new frame. Show its error without changing the
  // current match; a later successful load uses the ordinary reconnect notice.
  document.addEventListener("marl-client-load-failed", (event) => {
    const message = event instanceof CustomEvent ? event.detail?.message : null;
    if (typeof message === "string" && message.length > 0) {
      setNotice(message, "error");
      renderConnection();
    }
  });
  document.addEventListener("marl-client-session-replaced", () => {
    void loadCurrentFrame();
  });
}

/**
 * Open or close How To Play. In the BattleClient and DevClient it is the
 * overlay over the arena (#htp), as on the vision page; in the Replay Viewer it
 * is the Help dialog. Works at any time, also at game over and while a command
 * is out: it is page-local and sends nothing.
 *
 * @param {boolean} [open] true opens, false closes; leave it out to toggle.
 * @returns {void}
 */
function toggleHowToPlay(open) {
  if (isReplayMode()) {
    if (open === false) elements.helpDialog.close();
    else if (!elements.helpDialog.open) elements.helpDialog.showModal();
    return;
  }
  elements.howToPlay.hidden = !(open ?? elements.howToPlay.hidden);
}

elements.helpButton.addEventListener("click", () => toggleHowToPlay());
elements.howToPlayCloseButton.addEventListener("click", () => toggleHowToPlay(false));
elements.railFullGuideButton.addEventListener("click", () => toggleHowToPlay());
// Read The Full Guide: the full key guide is the Help dialog with every key.
elements.howToPlayFullButton.addEventListener("click", () => {
  toggleHowToPlay(false);
  elements.helpDialog.showModal();
});

// Local jobs of the one command set that this page owns. The Replay Viewer's
// single keyboard dispatcher (bindReplayTimelineControls) reaches them for ?
// and G; the live battlefield opens Help on ? itself and forwards G to Python,
// so these two run only in the Replay Viewer. Camera keys (V, B) and Frame
// Explanation (F) are registered by their own owners.
registerLocalCommandHandler("help", () => {
  if (elements.helpDialog.open) {
    return false;
  }
  elements.helpDialog.showModal();
  return true;
});
registerLocalCommandHandler("ranges", () => toggleReplayRanges());

bindReplayTimelineControls(replayTimelineElements, replayPlayback);

bindFilterControls();

elements.replayExportPngButton.addEventListener("click", () => {
  void exportReplayBattlefieldPng();
});

elements.replayDownloadMetricsButton.addEventListener("click", () => {
  void downloadReplayMetricReport();
});
elements.replayEpisodeDetailsButton.addEventListener("click", () => {
  void downloadReplayMetricReport(true);
});
for (const control of [
  elements.metricPanel,
  elements.metricScope,
  elements.metricSelection,
  elements.metricView,
]) {
  control.addEventListener(
    control === elements.metricPanel ? "toggle" : "change",
    () => {
      replayMetricFailure = null;
      replayMetricCatalogFailure = "";
      replayMetricFocus = null;
      elements.metricSearchDefinition.hidden = true;
      if (
        control === elements.metricScope &&
        replayArtifactActionTransaction?.kind === "download_metrics"
      ) {
        invalidateReplayArtifactAction();
        renderReplayArtifactActions(installedPresentationAuthority());
      }
      renderReplayMetrics();
    },
  );
}

elements.metricSearch.addEventListener("input", () => {
  replayMetricSearchLimit = 20;
  elements.metricSearchDefinition.hidden = true;
  renderReplayMetricSearch();
  setReplayMetricSearchOpen(true);
});
for (const event of ["focus", "click"]) {
  elements.metricSearch.addEventListener(event, () => setReplayMetricSearchOpen(true));
}
elements.metricSearchArea.addEventListener(
  "focusout",
  (/** @type {FocusEvent} */ event) => {
    // Focus moving to a holder of the search (the Metrics tab panel, which a
    // press on the search's own text focuses) keeps the list open.
    const next = event.relatedTarget;
    if (
      next !== null &&
      !elements.metricSearchArea.contains(/** @type {Node} */ (next)) &&
      !(next instanceof Node && next.contains(elements.metricSearchArea))
    ) {
      setReplayMetricSearchOpen(false);
    }
  },
);
document.addEventListener(
  "pointerdown",
  (event) => {
    if (!elements.metricSearchArea.contains(event.target)) {
      setReplayMetricSearchOpen(false);
    }
  },
  true,
);
elements.metricSearchArea.addEventListener(
  "keydown",
  (/** @type {KeyboardEvent} */ event) => {
    if (event.isComposing) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      elements.metricSearch.focus({ preventScroll: true });
      setReplayMetricSearchOpen(false);
      return;
    }
    if (
      event.target !== elements.metricSearch ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey
    ) {
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      const options = [...elements.metricSearchResults.querySelectorAll("button")];
      if (!options.length) return;
      event.preventDefault();
      event.stopPropagation();
      setReplayMetricSearchOpen(true);
      const index = options.indexOf(replayMetricSearchActive);
      const down = event.key === "ArrowDown";
      const next =
        index < 0
          ? down
            ? 0
            : options.length - 1
          : Math.max(0, Math.min(options.length - 1, index + (down ? 1 : -1)));
      const button = options[next];
      setActiveMetricSearchResult(button);
      button.scrollIntoView({ block: "nearest" });
    } else if (event.key === "Enter" && replayMetricSearchActive) {
      event.preventDefault();
      event.stopPropagation();
      replayMetricSearchActive.click();
    }
  },
);
elements.metricSearchMore.addEventListener("click", () => {
  replayMetricSearchLimit += 20;
  renderReplayMetricSearch();
});

/**
 * Toggle Ranges in the Replay Viewer: the Ranges button and the G key.
 *
 * Agent POV flips the page-local range choice and repaints (no request, no
 * Agent POV switch). Oracle View sends one set_ranges replay command with the
 * opposite of the installed frame's choice. Does nothing outside the Replay
 * Viewer, while the Ranges button is disabled, or without an Oracle View or
 * Agent POV presentation.
 *
 * @returns {boolean} Whether a local change or one request started.
 */
function toggleReplayRanges() {
  if (!isReplayMode() || elements.replayRangesButton.disabled) {
    return false;
  }
  if (authorizedPresentationAudience(state.presentation) === "agent_pov") {
    if (toggleAgentLocalRanges()) {
      render();
      return true;
    }
    return false;
  }
  if (authorizedPresentationAudience(state.presentation) !== "researcher") {
    return false;
  }
  const frame = state.frame;
  if (!frame) {
    return false;
  }
  void dispatchReplayCommand({
    command_type: "set_ranges",
    show_ranges: frame.show_ranges !== true,
  });
  return true;
}

elements.replayRangesButton.addEventListener("click", () => {
  toggleReplayRanges();
});

/**
 * Clear the current replay reference through its permitted authority path.
 * Agent POV clears local inspection and closes details without disabling future
 * automatic opening; Oracle sends select_agent with null. Do nothing outside
 * replay or while the Clear control is disabled.
 */
function clearReplaySelection() {
  if (!isReplayMode() || elements.replayClearReferenceButton.disabled) {
    return;
  }
  if (authorizedPresentationAudience(state.presentation) === "agent_pov") {
    if (setLocalInspectedPresentationKey(null)) {
      closeAgentDetailsWithoutLatching();
      render();
      elements.battlefield.removeAttribute("aria-activedescendant");
    }
    return;
  }
  if (authorizedPresentationAudience(state.presentation) !== "researcher") {
    return;
  }
  void dispatchReplayCommand({
    command_type: "select_agent",
    selected_global_slot: null,
  });
}

elements.replayClearReferenceButton.addEventListener("click", clearReplaySelection);

elements.liveRangesButton.addEventListener("click", () => {
  if (isReplayMode() || elements.liveRangesButton.disabled) {
    return;
  }
  if (authorizedPresentationAudience(state.presentation) === "agent_pov") {
    if (toggleAgentLocalRanges()) {
      render();
    }
    return;
  }
  if (authorizedPresentationAudience(state.presentation) !== "researcher") {
    return;
  }
  void dispatchCommand(keyboardCommand("g"));
});

document.addEventListener("visibilitychange", () => {
  replayPlayback.setHidden(document.hidden);
});

const battlefieldResizeObserver = new ResizeObserver(scheduleBattlefieldResize);
battlefieldResizeObserver.observe(elements.battlefieldShell);
replayPlayback.setHidden(document.hidden);

for (const { panelId, panel, body } of scientificDisclosures) {
  /**
   * Block this disclosure summary click or keyboard activation while authority is pending.
   * Ignore events outside the summary. With a matching active preference, clear the
   * expected programmatic-toggle marker and allow the user action; otherwise prevent
   * default behavior and stop later listeners from opening unavailable content.
   *
   * @param {Event} event
   */
  const blockPendingSummaryActivation = (event) => {
    const summary = panel.querySelector(":scope > summary");
    if (
      !(event.target instanceof Element) ||
      !(summary instanceof Element) ||
      (event.target !== summary && !summary.contains(event.target))
    ) {
      return;
    }
    if (installedActivePresentationPreference(state.presentation) !== null) {
      expectedDisclosureToggles.delete(panel);
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
  };
  panel.addEventListener("click", blockPendingSummaryActivation, true);
  panel.addEventListener(
    "keydown",
    (event) => {
      if (event.key === "Enter" || event.key === " ") {
        blockPendingSummaryActivation(event);
      }
    },
    true,
  );
  panel.addEventListener("toggle", () => {
    const expected = expectedDisclosureToggles.get(panel);
    if (expected?.open === panel.open) {
      expectedDisclosureToggles.delete(panel);
      return;
    }
    expectedDisclosureToggles.delete(panel);
    const preference = installedActivePresentationPreference(state.presentation);
    if (preference === null) {
      setProgrammaticDisclosureOpen(panel, false);
      return;
    }
    const saved = preference.disclosures[panelId];
    preference.disclosures[panelId] = {
      open: panel.open,
      scrollTop: panel.open ? (saved?.scrollTop ?? 0) : body.scrollTop,
    };
    if (panel.open) {
      body.scrollTop = preference.disclosures[panelId].scrollTop;
      return;
    }
    if (panelId === "agent-details") {
      preference.agentDetailsAutoOpenAllowed = false;
    }
    const active = document.activeElement;
    const summary = panel.querySelector(":scope > summary");
    if (
      active instanceof Element &&
      active !== summary &&
      panel.contains(active) &&
      summary instanceof HTMLElement
    ) {
      summary.focus({ preventScroll: true });
    }
  });
}

registerControlHelp();
if (productIdentity === null) {
  state.offline = true;
  state.resyncRequired = true;
  setNotice(
    `Startup blocked: ${startupProductIdentityError ?? "Product bootstrap is unavailable."} Reopen the exact URL printed by the Python launcher.`,
    "error",
  );
  render();
} else {
  render();
  void loadCurrentFrame();
}
