/**
 * @file Control presentation time for already-authorized combat frames.
 *
 * What it owns: CombatChoreographer installs painter-owned SVG, starts and stops
 * every Web Animation of a tick, and suppresses repeated live transitions with a
 * bounded optional storage ledger. Replay can animate a tick, show the settled
 * paused still, or glide one tick (replay_step). A played replay tick keeps its
 * cues, as drawn, after it settles; when the next played tick starts they fade
 * out over 160 ms (normal motion), so continuous play never shows a cue-less
 * still between ticks or a one-frame swap. A seek, a step or a view change
 * clears them at once, so the Settled Still is exact. In every played tick
 * (live or replay) a transient cue whose own fade has ended stays hidden: its
 * still picture, which the looks draw for the Settled Still, does not come
 * back for the rest of the tick (endsHidden), so each blow shows once.
 *
 * The one rule of the beat clock: the DOM always holds the END state. The
 * durable renderer draws every body, life bar and mark where it ends the tick.
 * Body glides and beat holds are Web Animations overrides of the earlier look,
 * created here through the animation factory with ids "mbg:<epoch>:..." and counted
 * in the plan's bounds. Each uses fill "backwards" and never a forward fill, so
 * finishing or cancelling any animation shows END. A new tick, Lock In (skip), a
 * seek, a view change or a recipient change therefore ends every beat at END; a
 * new server revision of the tick on screen for the same audience (live) keeps it
 * playing. No timer and no CSS animation drives a beat-timed element.
 *
 * Durable nodes it may override (contracts for the look modules and the HUD):
 * - Body glide wrappers: g.agent-glide elements (data-glide-key) holding a body
 *   (g.agent with data-presentation-key). A glide is a FLIP translate added on top
 *   of the wrapper's own transform.
 * - Beat-held nodes: any element under the battlefield SVG (or surface.beatHoldRoots)
 *   with data-beat-hold ("impact", "settle", "fall", "return", "set-down" or
 *   "score"), data-beat-key (the champion's presentation key; not needed for
 *   "impact", "settle" or "score") and data-beat-side ("before": shows the value
 *   before the beat and is drawn at its END opacity, normally 0; "after": the END
 *   value). data-beat-fade gives an optional cross-fade in ms (default 0, a cut).
 *   data-beat-pulse="p" on an "after" node drawn at opacity 0 makes a flash instead:
 *   0 to p and back to 0 over the fade (normal motion only). A key with no beat in
 *   this plan (for example a death the Agent POV team did not see) gets no hold.
 * - Shiver nodes: elements with data-beat-shiver (pixels) and
 *   data-beat-shiver-at="impact" shake sideways once at Impact plus data-beat-delay
 *   over data-beat-fade ms (normal motion, playback below 1.5x).
 * - The look-owned overrides of a played tick: the Start Spot, afterimage and trail
 *   marks (looks/movement.js), the Agent POV sight veil and Spotted Glints
 *   (looks/fog.js) and the Death Dimming path (`path.death-dim` with
 *   data-dim-start and data-dim-end, looks/atmosphere.js).
 * - The camera wrapper, through easeCamera and leanCamera (Kill Cam).
 *
 * This module never chooses actions, advances the simulator or calls the debugger
 * command API. It reads only the plan (built from the audience's own events) and
 * the durable DOM, so in Agent POV nothing it times can depend on an unseen
 * champion.
 */
import {
  BEAT_CLOCK,
  beatRealTime,
  buildChoreographyPlan,
  PLAN_RESOURCE_CAPS,
} from "./choreography-plan.js";
import { dimAnimation } from "./looks/atmosphere.js";
import { sightAnimationSpecs } from "./looks/fog.js";
import {
  animationSpecs as movementAnimationSpecs,
  dimShiftSpecs,
} from "./looks/movement.js";

const DEFAULT_LEDGER_KEY = "marl-battlegrounds.visual-debugger.consumed-transitions.v1";
const DEFAULT_LEDGER_LIMIT = 256;
// The most animations one tick may run at once: the plan's own cap (one owner).
const MAX_ACTIVE_ANIMATIONS = PLAN_RESOURCE_CAPS.animations;
const REPLAY_TERMINAL_HOLD_MS = 600;
const SUPPORTED_PLAYBACK_RATES = Object.freeze([
  0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,
]);
// The one shared curve every body glide uses.
const GLIDE_EASING = "cubic-bezier(0.45, 0, 0.55, 1)";
// A camera change eases over this many ms (reduced motion and Agent POV changes cut).
const CAMERA_EASE_MS = 520;
// How long the previous played tick's kept cues take to fade out when the next
// played tick starts (normal motion), in ms at 1x.
const CUE_FADE_OUT_MS = 160;
// A carried scrub offset larger than this many world units is dropped, so a body is
// never dragged across the field.
const STEP_CARRY_LIMIT = 3;
const BEAT_HOLD_KINDS = new Set([
  "impact",
  "settle",
  "fall",
  "return",
  "set-down",
  "score",
]);
// A champion's shiver on a hit plays only below this playback rate.
const SHIVER_RATE_LIMIT = 1.5;

/**
 * @typedef {"normal" | "reduced" | "off"} MotionMode
 * @typedef {"live_once" | "replay_animated" | "replay_static"} RenderPolicy
 * @typedef {RenderPolicy | "replay_step"} RenderRequest
 *   What a caller may ask presentFrame for. "replay_step" is the one-tick step
 *   intent: install the paused still (replay_static), glide each body 240 ms from
 *   where the previous frame showed it to its new END spot, fade the tick's cues in
 *   from 240 ms to 320 ms, then settle into the paused still.
 * @typedef {{
 *   element: Element,
 *   keyframes: Keyframe[] | PropertyIndexedKeyframes,
 *   options: KeyframeAnimationOptions,
 *   id: string,
 * }} AnimationSpec
 * @typedef {{
 *   id: string,
 *   currentTime: CSSNumberish | null,
 *   playbackRate: number,
 *   finished: Promise<unknown>,
 *   pause: () => void,
 *   play: () => void,
 *   finish: () => void,
 *   cancel: () => void,
 *   updatePlaybackRate?: (rate: number) => void,
 * }} AnimationHandle
 * @typedef {{
 *   create: (spec: AnimationSpec) => AnimationHandle,
 *   createClock: (
 *     element: Element,
 *     duration: number,
 *     id: string,
 *   ) => AnimationHandle,
 * }} AnimationFactory
 * @typedef {{
 *   layer: SVGElement,
 *   ownerDocument: Document,
 *   viewportKey: string,
 *   worldToScreen: (
 *     point: {x: number, y: number} | readonly [number, number],
 *   ) => {x: number, y: number},
 *   worldLengthToScreen: (length: number) => number,
 *   battlefield?: Element | null,
 *   bodyWrapper?: (presentationKey: string) => Element | null,
 *   beatHoldRoots?: ReadonlyArray<Element>,
 *   beforeBeatHolds?: (plan: Record<string, any>) => void,
 * }} ChoreographySurface
 *   The durable renderer's surface. layer receives the painter's nodes.
 *   battlefield (optional; default layer's SVG) is searched for glide wrappers and
 *   beat-held nodes. bodyWrapper (optional) returns a body's glide wrapper by its
 *   presentation key; without it wrappers are found by their g.agent child's
 *   data-presentation-key. beatHoldRoots (optional) lists extra roots, such as a
 *   HUD outside the SVG, searched for beat-held nodes. beforeBeatHolds
 *   (optional) is called once with the plan on every install (present, resize
 *   or reproject), before the painter draws and before beat-held nodes are
 *   collected, so page chrome built from the plan (main.js's Fallen panel) can
 *   join the beat holds. It must draw only this audience's plan rows; an error
 *   it throws ends the install like a painter error.
 * @typedef {{
 *   root: Element,
 *   routeRoot?: Element,
 *   connectorRoot?: Element,
 *   animationSpecs?: readonly AnimationSpec[],
 *   nodeCount?: number,
 *   persistentNodeCount?: number,
 *   heldNodeCount?: number,
 *   heldPersistentNodeCount?: number,
 * }} ChoreographyInstallation
 *   What the painter returns. root, routeRoot and connectorRoot (the last two
 *   optional) are the painter's groups that carry the plan's
 *   data-authorization-key. nodeCount and persistentNodeCount (checked
 *   against the plan's bounds) leave out held rows carried from the previous
 *   tick; heldNodeCount and heldPersistentNodeCount (optional, information
 *   only) count those held rows.
 * @typedef {{
 *   getItem: (key: string) => string | null,
 *   setItem: (key: string, value: string) => void,
 * }} PresentationStorage
 */

/**
 * Remember a bounded list of presented live transition epochs.
 *
 * Entries pair an epoch key with its disclosed-content fingerprint. This mutable
 * presentation cache suppresses repeated animation; it grants no information rights.
 * Optional storage failures fall back to the current in-memory entries.
 */
export class ConsumedTransitionLedger {
  /**
   * Load the optional presentation cache and keep its newest bounded entries.
   *
   * options defaults to an empty object. storage defaults to null (memory only),
   * storageKey to the module's versioned ledger key, and limit to 256. A nonpositive
   * or noninteger limit throws RangeError. Storage/JSON errors are ignored; invalid
   * stored rows are removed. The supplied storage object remains caller-owned.
   *
   * @param {{
   *   storage?: PresentationStorage | null,
   *   storageKey?: string,
   *   limit?: number,
   * }} [options]
   */
  constructor(options = {}) {
    this.storage = options.storage ?? null;
    this.storageKey = options.storageKey ?? DEFAULT_LEDGER_KEY;
    this.limit = positiveInteger(options.limit ?? DEFAULT_LEDGER_LIMIT, "limit");
    /** @type {Array<{epochKey: string, fingerprint: string}>} */
    this.entries = this.#load();
  }

  /**
   * Return whether epochKey has an entry, regardless of its fingerprint.
   *
   * This exact string lookup neither updates recency nor reads storage again.
   *
   * @param {string} epochKey
   */
  has(epochKey) {
    return this.entries.some((entry) => entry.epochKey === epochKey);
  }

  /**
   * Return the stored fingerprint for epochKey, or null if it is absent.
   *
   * An empty stored string remains an empty string. This lookup does not mutate the
   * ledger or reread optional storage.
   *
   * @param {string} epochKey
   * @returns {string | null}
   */
  fingerprintFor(epochKey) {
    return (
      this.entries.find((entry) => entry.epochKey === epochKey)?.fingerprint ?? null
    );
  }

  /**
   * Remember epochKey with fingerprint as the newest consumed transition.
   *
   * Replace any older entry for the same key, retain at most limit entries, then
   * attempt persistence. Inputs are caller-supplied strings; this method adds no
   * validation. Storage failure is ignored. Returns undefined.
   *
   * @param {string} epochKey
   * @param {string} fingerprint
   */
  record(epochKey, fingerprint) {
    const withoutEpoch = this.entries.filter((entry) => entry.epochKey !== epochKey);
    withoutEpoch.push({ epochKey, fingerprint });
    this.entries = withoutEpoch.slice(-this.limit);
    this.#save();
  }

  /**
   * Read and sanitize optional storage once during construction.
   *
   * Return a new array of the newest limit records having string epochKey and
   * fingerprint fields. Missing storage, invalid JSON/root or storage errors yield
   * an empty array. Stored identities are not treated as authorization.
   */
  #load() {
    if (!this.storage) {
      return [];
    }
    try {
      const raw = this.storage.getItem(this.storageKey);
      const parsed = raw === null ? [] : JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed
        .filter(
          (entry) =>
            entry &&
            typeof entry === "object" &&
            typeof entry.epochKey === "string" &&
            typeof entry.fingerprint === "string",
        )
        .slice(-this.limit)
        .map((entry) => ({
          epochKey: entry.epochKey,
          fingerprint: entry.fingerprint,
        }));
    } catch {
      return [];
    }
  }

  /**
   * Attempt to persist the current entries under this ledger's storage key.
   *
   * Return undefined; absent storage does nothing and write/serialization failures
   * are ignored because animation suppression must not control authority.
   */
  #save() {
    if (!this.storage) {
      return;
    }
    try {
      this.storage.setItem(this.storageKey, JSON.stringify(this.entries));
    } catch {
      // Presentation persistence is optional; authority is unaffected.
    }
  }
}

/**
 * Adapt the browser Web Animations API to the choreographer's clock interface.
 *
 * Methods create live browser animations. Tests may supply another adapter with
 * the same methods; the controller owns cancellation and settlement.
 */
export class BrowserAnimationFactory {
  /**
   * Start one browser animation from the supplied spec and assign its ID.
   *
   * spec provides the target Element, keyframes, timing options and id. Return the
   * browser Animation; Element.animate errors propagate. The element is not removed.
   *
   * @param {AnimationSpec} spec
   */
  create(spec) {
    const animation = spec.element.animate(spec.keyframes, spec.options);
    animation.id = spec.id;
    return animation;
  }

  /**
   * Create a timer animation whose two opacity values are both one.
   *
   * element hosts the clock, duration is browser animation time in milliseconds,
   * and id names it for inspection. Return the live Animation with fill both.
   * Browser animation errors propagate; the caller owns its lifetime.
   *
   * @param {Element} element
   * @param {number} duration
   * @param {string} id
   */
  createClock(element, duration, id) {
    return this.create({
      element,
      keyframes: [{ opacity: 1 }, { opacity: 1 }],
      options: { duration, fill: "both" },
      id,
    });
  }
}

/**
 * Own one authorized combat explanation and its presentation clocks.
 *
 * The controller mutates only its painter installation under the supplied surface
 * and the temporary Web Animations overrides it puts on durable nodes (body glides,
 * beat holds, the step cue fade and the camera ease). It tracks pause, rate and
 * settlement and reports immutable state snapshots. New authority clears old nodes
 * and cancels every override (END shows) before installing replacements. It never
 * submits an action. Call dispose when the owner no longer needs the controller.
 */
export class CombatChoreographer {
  /**
   * Create an idle controller with a required painter and optional adapters.
   *
   * options.painter supplies install, clear, settle and reproject. planBuilder defaults
   * to buildChoreographyPlan, animationFactory to BrowserAnimationFactory, ledger to
   * a memory-only ConsumedTransitionLedger, and onStateChange to a no-op. planBuilder
   * always receives an installed policy: a replay_step request is planned as
   * replay_static. motionMode defaults to normal and playbackRate to 1. Missing
   * painter throws TypeError; unknown modes or unsupported rates throw RangeError. No
   * frame is installed yet.
   *
   * @param {{
   *   painter: {
   *     install: (
   *       plan: Record<string, any>,
   *       surface: ChoreographySurface,
   *       options: {
   *         motionMode: MotionMode,
   *         renderPolicy: RenderPolicy,
   *         settled: boolean,
   *         persistentOnly: boolean,
   *         retainTransientOnSettle?: boolean,
   *       },
   *     ) => ChoreographyInstallation,
   *     clear: (installation: any, reason: string) => void,
   *     releaseForFade?: (installation: any) => Element[],
   *     settle: (installation: any) => void,
   *     reproject: (
   *       installation: any,
   *       plan: Record<string, any>,
   *       surface: ChoreographySurface,
   *     ) => void,
   *   },
   *   planBuilder?: (
   *     frame: unknown,
   *     surface: ChoreographySurface | null,
   *     visualFilters?: Record<string, boolean>,
   *     renderPolicy?: RenderPolicy,
   *   ) => Record<string, any> | null,
   *   animationFactory?: AnimationFactory,
   *   ledger?: ConsumedTransitionLedger,
   *   onStateChange?: (state: ReturnType<CombatChoreographer["snapshot"]>) => void,
   *   motionMode?: MotionMode,
   *   playbackRate?: number,
   * }} options
   */
  constructor(options) {
    if (!options || typeof options !== "object" || !options.painter) {
      throw new TypeError("CombatChoreographer requires a painter.");
    }
    this.painter = options.painter;
    this.planBuilder = options.planBuilder ?? buildChoreographyPlan;
    this.animationFactory = options.animationFactory ?? new BrowserAnimationFactory();
    this.ledger = options.ledger ?? new ConsumedTransitionLedger();
    this.onStateChange = options.onStateChange ?? (() => {});
    this.motionMode = normalizeMotionMode(options.motionMode ?? "normal");
    this.playbackRate = normalizePlaybackRate(options.playbackRate ?? 1);
    this.paused = false;
    this.submissionBlocked = false;
    this.logicalTime = 0;
    this.generation = 0;
    /** @type {Record<string, any> | null} */
    this.plan = null;
    /** @type {RenderPolicy | null} */
    this.renderPolicy = null;
    /** @type {ChoreographySurface | null} */
    this.surface = null;
    /** @type {ChoreographyInstallation | null} */
    this.installation = null;
    /** @type {AnimationHandle[]} */
    this.animations = [];
    /** @type {AnimationHandle | null} */
    this.gateClock = null;
    /** @type {AnimationHandle | null} */
    this.cleanupClock = null;
    /**
     * Running step glides, kept so a quick next step can start from where each
     * body is drawn (a smooth scrub) instead of jumping back to its last END spot.
     * @type {Array<{presentationKey: string, points: ReadonlyArray<readonly [number, number]>, offsets: ReadonlyArray<number>, animation: AnimationHandle}>}
     */
    this.stepGlides = [];
    /**
     * The previous played tick's kept cues, fading out at the next played tick's
     * start (#fadeOutCues): each root and its fade. Removed when the fade ends,
     * and at once by any clear.
     * @type {Array<{root: Element, animation: AnimationHandle}>}
     */
    this.fadingCues = [];
    /** @type {AnimationHandle | null} */
    this.cameraEase = null;
    /**
     * The running Kill Cam lean of this tick (also held in this.animations), or null.
     * @type {AnimationHandle | null}
     */
    this.cameraLean = null;
    this.cameraGeneration = 0;
    /** @type {Set<(value?: unknown) => void>} */
    this.settleWaiters = new Set();
  }

  /**
   * Return a frozen view of the controller's current presentation state.
   *
   * The result includes active installation, animation count (painter animations,
   * overrides, clocks and a running camera ease), nullable plan identity, render
   * policy (the installed one: a replay_step request reports replay_static), logical
   * milliseconds, motion mode, pause/rate and submission gate. It reports the last
   * captured logical time; it does not sample the clock itself.
   */
  snapshot() {
    return Object.freeze({
      active: this.installation !== null,
      animationCount: this.#allAnimations().length + (this.cameraEase ? 1 : 0),
      epochKey: this.plan?.epochKey ?? null,
      authorizationKey: this.plan?.authorizationKey ?? null,
      fingerprint: this.plan?.fingerprint ?? null,
      paintKey: this.plan?.paintKey ?? null,
      renderPolicy: this.renderPolicy,
      logicalTime: this.logicalTime,
      motionMode: this.motionMode,
      paused: this.paused,
      playbackRate: this.playbackRate,
      submissionBlocked: this.submissionBlocked,
    });
  }

  /**
   * Return a promise resolved when this controller next reports settlement.
   *
   * Resolve immediately when there are no owned animations and submission is not
   * blocked. Otherwise register a waiter until publish observes that state, including
   * a clear or skip. Replay autoplay uses this boundary to avoid overlapping requests.
   * The promise carries no frame data and never advances the simulator.
   */
  whenSettled() {
    if (this.#isSettled()) {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      this.settleWaiters.add(resolve);
    });
  }

  /**
   * Install or reconcile an already-authorized frame after durable rendering.
   *
   * frame is passed to the plan builder; surface supplies the transient SVG layer and
   * projection, or null to clear. presentationControl defaults to live_once with
   * builder-default visualFilters. renderPolicy accepts live_once, replay_animated,
   * replay_static or replay_step (see RenderRequest). restartAnimated=false normally
   * preserves settled replay state; true permits an explicit same-plan
   * static-to-animated restart.
   *
   * Return the frozen controller snapshot. Identical plans preserve time; so does a
   * new live server revision of the same tick for the same audience (live_once,
   * same epochKey, audienceKey, fingerprint and paintKey; only the authorization
   * key's authority epoch moved, as when policy work starts or ends): the running
   * tick keeps playing and the installed groups take the new authorization key.
   * A viewport change reprojects. Filter changes rebuild live plans without replaying consumed
   * motion and make replay explanations static. Changed authority/disclosure clears
   * old nodes first. Absent scene/plan clears the installation. Live epochs use the
   * ledger; replay does not consume it.
   *
   * Beat clock: a new tick cancels every running override, so the previous tick's
   * beats end at END. A played tick (live_once or replay_animated) glides bodies
   * START to END and holds beat-timed nodes until their beat (see #install). A
   * replay_step request on a new frame installs the paused still and glides each body
   * that appears in both the previous and the new plan with the same audienceKey from
   * where the previous frame showed it (its END spot, or where a still-running step
   * glide has drawn it) to its new END spot. When the shown frame has no plan (a
   * replay's first frame: no tick leads into it), a replay_step glides from the new
   * tick's START spots instead, which are where that frame showed each body
   * (stepSourceBefore); a replay_step onto such a frame (a -1 onto the first frame)
   * clears, then glides each body back from the shown tick's END to its START spot
   * (#startPlanlessStepGlide). A body unseen in either Agent POV scene,
   * a death, a return or a change of audience or recipient snaps. A replay_step
   * request for the frame already shown is the paused still. Invalid policy, plans,
   * painter bounds or browser animations may throw; a policy-induced identity change
   * throws Error.
   *
   * @param {unknown} frame
   * @param {ChoreographySurface | null} surface
   * @param {{
   *   renderPolicy?: RenderRequest,
   *   visualFilters?: Record<string, boolean>,
   *   restartAnimated?: boolean,
   * }} [presentationControl]
   */
  presentFrame(frame, surface, presentationControl = {}) {
    const request = normalizeRenderRequest(
      presentationControl.renderPolicy ?? "live_once",
    );
    const stepRequested = request === "replay_step";
    const requestedPolicy = stepRequested ? "replay_static" : request;
    const initialPolicy = requestedPolicy;
    let nextPlan = surface
      ? this.planBuilder(
          frame,
          surface,
          presentationControl.visualFilters,
          initialPolicy,
        )
      : null;
    if (!nextPlan || !surface) {
      // A -1 onto a replay's first frame (it has no plan): the bodies glide back
      // from the shown tick's END to its START spots, where that frame shows them.
      const backFrom =
        stepRequested && surface && this.plan !== null ? this.plan : null;
      const carry = backFrom !== null && surface ? this.#stepCarry(surface) : new Map();
      this.clear("absent_scene_or_event_batch");
      if (backFrom !== null && surface) {
        this.#startPlanlessStepGlide(backFrom, surface, carry);
      }
      return this.snapshot();
    }

    const currentPlan = this.plan;
    const sameEpoch = currentPlan?.epochKey === nextPlan.epochKey;
    const sameAuthorization =
      currentPlan?.authorizationKey === nextPlan.authorizationKey;
    const sameFingerprint = currentPlan?.fingerprint === nextPlan.fingerprint;
    const samePaint = currentPlan?.paintKey === nextPlan.paintKey;
    const replayPaintChange =
      sameEpoch &&
      sameAuthorization &&
      sameFingerprint &&
      !samePaint &&
      isReplayPolicy(initialPolicy);
    const explicitSamePlanReplayRestart =
      presentationControl.restartAnimated === true &&
      sameEpoch &&
      sameAuthorization &&
      sameFingerprint &&
      samePaint &&
      this.renderPolicy === "replay_static" &&
      requestedPolicy === "replay_animated";
    const stickyStatic =
      sameEpoch &&
      this.renderPolicy === "replay_static" &&
      requestedPolicy === "replay_animated" &&
      !explicitSamePlanReplayRestart;
    const nextPolicy =
      replayPaintChange || stickyStatic ? "replay_static" : initialPolicy;
    if (nextPolicy !== initialPolicy) {
      const policyPlan = this.planBuilder(
        frame,
        surface,
        presentationControl.visualFilters,
        nextPolicy,
      );
      if (!samePlanIdentity(nextPlan, policyPlan)) {
        throw new Error("render policy changed the authorized plan identity.");
      }
      nextPlan = policyPlan;
    }
    const samePolicy = this.renderPolicy === nextPolicy;
    // A new live server revision of the same tick for the same audience: only
    // the authority epoch (the server's frame revision) moved, for example when
    // policy work starts or ends. Same transition, same audience, same events
    // and same paint mean the same plan, so the running glide keeps its time.
    // Clearing here cut every glide a revision landed in, the arena jumped to
    // END and Run showed a paused-looking still between ticks.
    const sameAudienceRevision =
      sameEpoch &&
      !sameAuthorization &&
      nextPolicy === "live_once" &&
      typeof currentPlan?.audienceKey === "string" &&
      currentPlan.audienceKey === nextPlan.audienceKey;

    if (
      sameEpoch &&
      (sameAuthorization || sameAudienceRevision) &&
      sameFingerprint &&
      samePaint &&
      samePolicy
    ) {
      if (this.surface?.viewportKey !== surface.viewportKey && this.installation) {
        this.painter.reproject(this.installation, nextPlan, surface);
      }
      if (sameAudienceRevision && this.installation) {
        for (const node of [
          this.installation.root,
          this.installation.routeRoot,
          this.installation.connectorRoot,
        ]) {
          node?.setAttribute?.("data-authorization-key", nextPlan.authorizationKey);
        }
      }
      this.plan = nextPlan;
      this.surface = surface;
      this.#publish();
      return this.snapshot();
    }

    if (sameEpoch && sameAuthorization && sameFingerprint && !samePaint) {
      this.#clearOwned("visual_filters_changed");
      this.plan = nextPlan;
      this.surface = surface;
      this.#installForPolicy(nextPlan, surface, nextPolicy, {
        liveSafeRebuild: true,
      });
      this.#publish();
      return this.snapshot();
    }

    if (sameEpoch && sameAuthorization && sameFingerprint && !samePolicy) {
      this.#clearOwned("render_policy_changed");
      this.plan = nextPlan;
      this.surface = surface;
      this.#installForPolicy(nextPlan, surface, nextPolicy);
      this.#publish();
      return this.snapshot();
    }

    if (sameEpoch) {
      // Clear first: the previous authorization may contain privileged nodes.
      this.#clearOwned("authorization_or_disclosure_changed");
      this.plan = nextPlan;
      this.surface = surface;
      this.#installForPolicy(nextPlan, surface, nextPolicy, {
        liveSafeRebuild: true,
      });
      if (nextPolicy === "live_once") {
        this.ledger.record(nextPlan.epochKey, nextPlan.fingerprint);
      }
      this.#publish();
      return this.snapshot();
    }

    const consumedFingerprint =
      nextPolicy === "live_once" ? this.ledger.fingerprintFor(nextPlan.epochKey) : null;
    const alreadyConsumed = consumedFingerprint !== null;
    const previousPlan = this.plan;
    // Read where running step glides draw each body before they are cancelled.
    const carry = stepRequested ? this.#stepCarry(surface) : new Map();
    // Continuous play: the settled previous tick kept its cues as drawn; they
    // fade out over the new tick's first moments instead of vanishing in one
    // frame. A seek, a step or any other change clears them at once.
    const fadeFrom =
      nextPolicy === "replay_animated" &&
      this.renderPolicy === "replay_animated" &&
      this.motionMode === "normal" &&
      this.installation !== null &&
      this.cleanupClock === null &&
      typeof this.painter.releaseForFade === "function"
        ? this.installation
        : null;
    if (fadeFrom !== null) this.installation = null;
    this.#clearOwned("new_transition");
    this.plan = nextPlan;
    this.surface = surface;
    // A replay's first frame has no plan (no tick leads into it), so a +1 from
    // it glides from the new tick's START spots: where that frame showed them.
    const stepFrom =
      stepRequested && nextPolicy === "replay_static"
        ? (previousPlan ?? stepSourceBefore(nextPlan))
        : null;
    this.#installForPolicy(nextPlan, surface, nextPolicy, {
      liveSafeRebuild: alreadyConsumed,
      step: stepFrom !== null ? { previousPlan: stepFrom, carry } : null,
    });
    if (fadeFrom !== null) this.#fadeOutCues(fadeFrom, nextPlan.epochKey);
    if (nextPolicy === "live_once" && consumedFingerprint !== nextPlan.fingerprint) {
      this.ledger.record(nextPlan.epochKey, nextPlan.fingerprint);
    }
    this.#publish();
    return this.snapshot();
  }

  /**
   * Update installed geometry after a durable resize without replaying time.
   *
   * frame and surface have the presentFrame contract. presentationControl defaults
   * to live_once and builder-default filters; a replay_step request reprojects as the
   * paused still (replay_static). Return a frozen snapshot. If the plan, authority,
   * paint choice or installation no longer matches, delegate to presentFrame. A
   * settled replay remains static. Running overrides keep their time; the durable
   * renderer has already moved every node to its new END spot. Plan/painter errors
   * propagate.
   *
   * @param {unknown} frame
   * @param {ChoreographySurface | null} surface
   * @param {{
   *   renderPolicy?: RenderRequest,
   *   visualFilters?: Record<string, boolean>,
   * }} [presentationControl]
   */
  reproject(frame, surface, presentationControl = {}) {
    const request = normalizeRenderRequest(
      presentationControl.renderPolicy ?? "live_once",
    );
    const requestedPolicy = request === "replay_step" ? "replay_static" : request;
    if (request === "replay_step") {
      presentationControl = { ...presentationControl, renderPolicy: requestedPolicy };
    }
    if (!surface || !this.installation || !this.plan) {
      return this.presentFrame(frame, surface, presentationControl);
    }
    let nextPlan = this.planBuilder(
      frame,
      surface,
      presentationControl.visualFilters,
      requestedPolicy,
    );
    if (
      !nextPlan ||
      nextPlan.epochKey !== this.plan.epochKey ||
      nextPlan.authorizationKey !== this.plan.authorizationKey ||
      nextPlan.fingerprint !== this.plan.fingerprint ||
      nextPlan.paintKey !== this.plan.paintKey
    ) {
      return this.presentFrame(frame, surface, presentationControl);
    }
    const nextPolicy =
      this.renderPolicy === "replay_static" && requestedPolicy === "replay_animated"
        ? "replay_static"
        : requestedPolicy;
    if (nextPolicy !== this.renderPolicy) {
      return this.presentFrame(frame, surface, {
        ...presentationControl,
        renderPolicy: nextPolicy,
      });
    }
    if (nextPolicy !== requestedPolicy) {
      nextPlan = this.planBuilder(
        frame,
        surface,
        presentationControl.visualFilters,
        nextPolicy,
      );
      if (!samePlanIdentity(this.plan, nextPlan)) {
        return this.presentFrame(frame, surface, presentationControl);
      }
    }
    this.painter.reproject(this.installation, nextPlan, surface);
    this.plan = nextPlan;
    this.surface = surface;
    this.#publish();
    return this.snapshot();
  }

  /**
   * Toggle all owned animation clocks and return the resulting snapshot.
   *
   * Motion off or no animations is a no-op. Otherwise pause/play every handle, capture
   * logical time and publish. This pauses presentation only; it sends no game command.
   */
  togglePaused() {
    if (this.motionMode === "off" || this.#allAnimations().length === 0) {
      return this.snapshot();
    }
    this.paused = !this.paused;
    for (const animation of this.#allAnimations()) {
      if (this.paused) {
        animation.pause();
      } else {
        animation.play();
      }
    }
    this.#captureLogicalTime();
    this.#publish();
    return this.snapshot();
  }

  /**
   * Apply a supported presentation speed while retaining current clock progress.
   *
   * rate must be 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75 or 2; otherwise throw RangeError.
   * An unchanged rate is a no-op. Update every owned animation and publish, then return
   * a snapshot. Simulation timing and recorded transition values are unchanged.
   *
   * @param {number} rate
   */
  setPlaybackRate(rate) {
    const nextRate = normalizePlaybackRate(rate);
    if (nextRate === this.playbackRate) {
      return this.snapshot();
    }
    this.#captureLogicalTime();
    this.playbackRate = nextRate;
    for (const animation of this.#allAnimations()) {
      applyPlaybackRate(animation, nextRate);
    }
    this.#publish();
    return this.snapshot();
  }

  /**
   * Set normal, reduced or off motion and reconcile the active explanation.
   *
   * mode outside that set throws RangeError. An unchanged mode is a no-op. Off can
   * reinstall an active played explanation statically; a running step glide, and
   * reduced motion, settle through skip (END shows). Returning to normal does not
   * replay an already settled transition. Return a snapshot; painter/browser failures
   * propagate.
   *
   * @param {MotionMode} mode
   */
  setMotionMode(mode) {
    const nextMode = normalizeMotionMode(mode);
    if (nextMode === this.motionMode) {
      return this.snapshot();
    }

    const plan = this.plan;
    const surface = this.surface;
    const hasActiveExplanation =
      this.installation !== null &&
      plan !== null &&
      surface !== null &&
      this.#allAnimations().length > 0;
    this.motionMode = nextMode;

    if (
      this.motionMode === "off" &&
      hasActiveExplanation &&
      this.renderPolicy !== "replay_static"
    ) {
      this.paused = false;
      this.#clearOwned("motion_disabled_static_reinstall");
      this.#install(plan, surface, {
        renderPolicy: this.renderPolicy ?? "live_once",
        settled: false,
        persistentOnly: false,
      });
      this.#publish();
    } else if (this.motionMode !== "normal") {
      this.paused = false;
      this.skip();
    } else {
      this.#publish();
    }
    return this.snapshot();
  }

  /**
   * Settle the current explanation immediately and release submission gating.
   *
   * Finish owned handles where possible (every glide and beat hold shows END), finish
   * a running camera ease, settle the painter, cancel remaining clocks and publish the
   * final logical time. A step glide settles back into the paused still. With no
   * installation, simply release the gate. Lock In calls this, so a Lock In always
   * ends the beats at END. Return a snapshot; this does not request or skip a
   * simulator transition.
   */
  skip() {
    if (this.cameraEase) {
      safeFinish(this.cameraEase);
      this.cameraEase = null;
    }
    const installation = this.installation;
    if (!installation) {
      this.submissionBlocked = false;
      this.#publish();
      return this.snapshot();
    }
    for (const animation of this.#allAnimations()) {
      safeFinish(animation);
    }
    this.logicalTime = Number(this.plan?.phases?.total ?? this.logicalTime);
    this.painter.settle(installation);
    this.#cancelAnimations();
    this.submissionBlocked = false;
    if (this.renderPolicy === "replay_static") {
      this.paused = true;
    }
    this.#publish();
    return this.snapshot();
  }

  /**
   * Remove this controller's installation and clear its active frame identity.
   *
   * reason defaults to explicit_clear and is passed to the painter. Cancel clocks and
   * overrides (END shows), reset logical time, release the gate and publish,
   * resolving settlement waiters. The dispose reason also cancels a running camera
   * ease. Return a snapshot. The consumed-transition ledger is retained.
   *
   * @param {string} reason
   */
  clear(reason = "explicit_clear") {
    if (reason === "dispose") {
      this.#cancelCameraEase();
    }
    this.#clearOwned(reason);
    this.plan = null;
    this.renderPolicy = null;
    this.surface = null;
    this.logicalTime = 0;
    this.#publish();
    return this.snapshot();
  }

  /**
   * Release owned SVG and clocks using the dispose clear reason.
   *
   * Return the cleared snapshot and resolve settlement waiters. The ledger survives;
   * this method does not destroy caller-owned storage or prevent later reuse.
   */
  dispose() {
    return this.clear("dispose");
  }

  /**
   * Choose settled/static or animated installation for a validated plan.
   *
   * plan and surface are ready for the painter; renderPolicy selects live/replay.
   * options.liveSafeRebuild defaults to false and suppresses repeated live transients
   * when true. replay_static keeps settled explanatory transients and marks paused;
   * replay_animated keeps them too once its cleanup clock settles the tick, so a
   * played tick ends as drawn and its cues leave only when the next tick starts
   * (they fade out then when it is a played tick, see #fadeOutCues; live_once is
   * unchanged: its transients leave at settle);
   * options.step (default null) adds the replay_step glide and cue fade on top of
   * that still: {previousPlan, carry} where previousPlan is the plan shown before
   * this frame and carry maps a presentation key to the world spot where a running
   * step glide drew that body. Mutate controller installation state; return
   * undefined. Installation errors propagate.
   *
   * @param {Record<string, any>} plan
   * @param {ChoreographySurface} surface
   * @param {RenderPolicy} renderPolicy
   * @param {{
   *   liveSafeRebuild?: boolean,
   *   step?: {previousPlan: Record<string, any>, carry: Map<string, readonly [number, number]>} | null,
   * }} [options]
   */
  #installForPolicy(plan, surface, renderPolicy, options = {}) {
    if (renderPolicy === "replay_static") {
      this.paused = true;
      this.#install(plan, surface, {
        renderPolicy,
        settled: true,
        persistentOnly: false,
        retainTransientOnSettle: true,
        step: options.step ?? null,
      });
      return;
    }
    this.paused = false;
    const liveSafeRebuild =
      renderPolicy === "live_once" && options.liveSafeRebuild === true;
    this.#install(plan, surface, {
      renderPolicy,
      settled: liveSafeRebuild,
      persistentOnly: liveSafeRebuild,
      // A played replay tick ends like its Settled Still: when its cleanup
      // clock settles it, the tick's cues stay until the next tick replaces
      // the installation. Removing them there showed a cue-less still for the
      // ~80 ms before the next tick (the "paused" flash). Live play unchanged.
      ...(renderPolicy === "replay_animated" ? { retainTransientOnSettle: true } : {}),
    });
  }

  /**
   * Install bounded painter nodes and create the required animation clocks.
   *
   * plan supplies phases and beats in milliseconds and resource bounds; surface owns
   * projection, the transient layer and the durable nodes. options gives
   * renderPolicy, settled and persistentOnly; retainTransientOnSettle is forwarded
   * only when true; step (replay_static only) adds the one-tick step glide. The
   * painter also gets the page clock (performance.now()) and the playback rate
   * for its card queue. Validate
   * node, persistent-node and animation counts; excess painter bounds clear the new
   * installation and throw RangeError.
   *
   * Overrides: a played tick (not settled, motion normal or reduced) adds body
   * glides (normal motion with Body Glide on: START to END in 0-260 ms, a Charge
   * rush first) and beat holds for every beat-held node whose beat is in the plan
   * (reduced motion scales the hold times by plan.beats.reducedScale). They are
   * counted against plan.bounds.overrideAnimations and the animation limit
   * (PLAN_RESOURCE_CAPS.animations, 1024); over
   * the limit the extra holds are dropped (never the tick). Settled or
   * persistent-only output has no overrides and no clocks, except the step glide.
   *
   * Clocks: animated replay lasts its whole tick: the later of the event windows and
   * the beats (plan.beats.explanationEnd) plus the 600 ms end hold, so a tick with
   * Longer Tick lasts 1700 ms (3200 ms with a death); a plan without beats keeps the
   * old total plus 600 ms. Normal motion gates submission until the 450 ms release
   * only when the painter animates events; a glide never gates it. Creation failures
   * cancel newly created handles, clear the installation and rethrow. Completion
   * handlers use generation checks so stale clocks cannot settle a later authority.
   * A painter animation that hides a transient cue for good (endsHidden) sets that
   * node's opacity attribute to 0 when it finishes, under the same generation and
   * installation check, so the cue's still picture never shows again this tick.
   * Return undefined.
   *
   * @param {Record<string, any>} plan
   * @param {ChoreographySurface} surface
   * @param {{
   *   renderPolicy: RenderPolicy,
   *   settled: boolean,
   *   persistentOnly: boolean,
   *   retainTransientOnSettle?: boolean,
   *   step?: {previousPlan: Record<string, any>, carry: Map<string, readonly [number, number]>} | null,
   * }} options
   */
  #install(plan, surface, options) {
    const persistentOnly = Boolean(options.persistentOnly);
    const settled = Boolean(options.settled);
    /** @type {{
     *   motionMode: MotionMode,
     *   renderPolicy: RenderPolicy,
     *   settled: boolean,
     *   persistentOnly: boolean,
     *   retainTransientOnSettle?: boolean,
     *   clock?: number,
     *   playbackRate: number,
     * }} */
    const painterOptions = {
      motionMode: this.motionMode,
      renderPolicy: options.renderPolicy,
      settled,
      persistentOnly,
      // The painter's card queue times cards with the page clock and the rate.
      clock: globalThis.performance?.now?.(),
      playbackRate: this.playbackRate,
    };
    if (options.retainTransientOnSettle === true) {
      painterOptions.retainTransientOnSettle = true;
    }
    // Page chrome built from the plan (the Fallen panel) is drawn first, so its
    // beat-held rows are found below like every other root's.
    if (typeof surface.beforeBeatHolds === "function") {
      surface.beforeBeatHolds(plan);
    }
    const installation = this.painter.install(plan, surface, painterOptions);
    const nodeCount = nonNegativeInteger(installation.nodeCount ?? 0, "nodeCount");
    const animationSpecs = Array.isArray(installation.animationSpecs)
      ? installation.animationSpecs
      : [];
    if (nodeCount > Number(plan.bounds?.nodes ?? 512)) {
      this.painter.clear(installation, "node_bound_exceeded");
      throw new RangeError("choreography painter exceeded the planned node bound.");
    }
    const persistentNodeCount = nonNegativeInteger(
      installation.persistentNodeCount ?? 0,
      "persistentNodeCount",
    );
    if (persistentNodeCount > Number(plan.bounds?.persistentNodes ?? 64)) {
      this.painter.clear(installation, "persistent_node_bound_exceeded");
      throw new RangeError("choreography painter exceeded the persistent node bound.");
    }
    const staticOff =
      this.motionMode === "off" && !settled && !persistentOnly && nodeCount > 1;
    const eventlessReplayAnimated =
      options.renderPolicy === "replay_animated" &&
      animationSpecs.length === 0 &&
      !settled &&
      !persistentOnly;
    const overrides =
      settled || persistentOnly || this.motionMode === "off"
        ? []
        : [
            ...playedGlideSpecs(plan, surface, this.motionMode),
            ...beatHoldSpecs(plan, surface, this.motionMode),
            ...playedLookSpecs(
              plan,
              surface,
              this.motionMode,
              this.playbackRate,
              options.renderPolicy,
            ),
          ];
    const clockBudget =
      settled || persistentOnly
        ? 0
        : eventlessReplayAnimated || staticOff
          ? 1
          : animationSpecs.length === 0
            ? overrides.length > 0
              ? 1
              : 0
            : this.motionMode === "normal"
              ? 2
              : 1;
    const activeAnimationCount = animationSpecs.length + clockBudget;
    const plannedAnimationLimit = nonNegativeInteger(
      plan.bounds?.animations ?? MAX_ACTIVE_ANIMATIONS,
      "plan.bounds.animations",
    );
    if (
      activeAnimationCount > MAX_ACTIVE_ANIMATIONS ||
      activeAnimationCount > plannedAnimationLimit
    ) {
      this.painter.clear(installation, "animation_bound_exceeded");
      throw new RangeError("choreography painter exceeded the animation bound.");
    }
    const keptOverrides = boundedOverrides(
      overrides,
      plan,
      MAX_ACTIVE_ANIMATIONS - activeAnimationCount,
    );
    this.installation = installation;
    this.renderPolicy = options.renderPolicy;
    this.logicalTime =
      options.renderPolicy === "replay_static"
        ? 0
        : settled
          ? Number(plan.phases?.total ?? 0)
          : 0;
    if (settled || persistentOnly) {
      this.painter.settle(installation);
      this.submissionBlocked = false;
      if (options.step) {
        this.#startStepGlide(plan, surface, installation, options.step);
      }
      return;
    }
    if (
      animationSpecs.length === 0 &&
      keptOverrides.length === 0 &&
      !staticOff &&
      !eventlessReplayAnimated
    ) {
      this.logicalTime = Number(plan.phases?.total ?? 0);
      this.painter.settle(installation);
      this.submissionBlocked = false;
      return;
    }

    const reduced = this.motionMode === "reduced";
    const eventEnd = reduced
      ? Number(plan.phases?.reducedTotal ?? 0)
      : Number(plan.phases?.total ?? 0);
    const beatEnd = Number.isFinite(plan.beats?.explanationEnd)
      ? Number(plan.beats.explanationEnd) *
        (reduced ? Number(plan.beats.reducedScale ?? 1) : 1)
      : 0;
    const overrideEnd = keptOverrides.reduce(
      (latest, spec) =>
        Math.max(
          latest,
          Number(spec.options.delay ?? 0) + Number(spec.options.duration ?? 0),
        ),
      0,
    );
    const duration =
      options.renderPolicy === "replay_animated"
        ? Math.max(eventEnd, beatEnd, overrideEnd) + REPLAY_TERMINAL_HOLD_MS
        : Math.max(eventEnd, overrideEnd);
    const gated =
      this.motionMode === "normal" &&
      !eventlessReplayAnimated &&
      animationSpecs.length > 0;
    /** @type {AnimationHandle[]} */
    const createdAnimations = [];
    /** @type {AnimationHandle | null} */
    let cleanupClock = null;
    /** @type {AnimationHandle | null} */
    let gateClock = null;
    const generation = this.generation;
    try {
      for (const [index, spec] of [...animationSpecs, ...keptOverrides].entries()) {
        const animation = this.animationFactory.create(spec);
        ignoreCancellation(animation.finished);
        if (index < animationSpecs.length && endsHidden(spec)) {
          // A cue that fades out stays out: the still under it is the Settled
          // Still's picture, and showing it again for the rest of the tick
          // made every played tick blink as if paused (see endsHidden).
          ignoreCancellation(
            animation.finished.then(() => {
              if (
                generation === this.generation &&
                this.installation === installation
              ) {
                spec.element.setAttribute("opacity", "0");
              }
            }),
          );
        }
        createdAnimations.push(animation);
      }
      cleanupClock = this.animationFactory.createClock(
        installation.root,
        duration,
        `mbg:${plan.epochKey}:cleanup`,
      );
      ignoreCancellation(cleanupClock.finished);
      if (gated) {
        gateClock = this.animationFactory.createClock(
          installation.root,
          Number(plan.phases?.submissionRelease ?? 0),
          `mbg:${plan.epochKey}:gate`,
        );
        ignoreCancellation(gateClock.finished);
      }
    } catch (error) {
      for (const animation of [
        ...createdAnimations,
        ...(cleanupClock ? [cleanupClock] : []),
        ...(gateClock ? [gateClock] : []),
      ]) {
        safeCancel(animation);
      }
      this.painter.clear(installation, "animation_creation_failed");
      this.installation = null;
      throw error;
    }
    this.animations = createdAnimations;
    this.cleanupClock = cleanupClock;
    this.gateClock = gateClock;
    this.submissionBlocked = gated;
    for (const animation of this.#allAnimations()) {
      applyPlaybackRate(animation, this.playbackRate);
      if (this.paused) {
        animation.pause();
      }
    }
    if (this.gateClock) {
      ignoreCancellation(
        this.gateClock.finished.then(() => {
          if (generation !== this.generation) {
            return;
          }
          this.submissionBlocked = false;
          this.#captureLogicalTime();
          this.#publish();
        }),
      );
    }
    if (this.cleanupClock) {
      ignoreCancellation(
        this.cleanupClock.finished.then(() => {
          if (generation !== this.generation || !this.installation) {
            return;
          }
          this.logicalTime = Number(this.plan?.phases?.total ?? this.logicalTime);
          this.painter.settle(this.installation);
          this.#cancelAnimations();
          this.submissionBlocked = false;
          this.#publish();
        }),
      );
    }
  }

  /**
   * Start the replay_step glide on top of an already settled paused still.
   *
   * plan is the new frame's plan, surface its surface, installation the settled
   * painter installation and step {previousPlan, carry}. In normal motion with Step
   * Glide on (plan.beats.stepGlide, default on) and the same audienceKey, each body
   * present in both plans glides 240 ms from where the previous frame showed it (a
   * running step glide's drawn spot when carry has one, otherwise its previous END
   * spot) to its new END spot; a Charge rush takes the first 100 ms forwards or the
   * first 140 ms backwards. The painter's cue roots fade in from 240 ms to 320 ms,
   * the movement marks fade in over the step glide and, on a +1 step in Agent POV,
   * the sight veil cross-fades. A one-step clock then marks the still paused again. Reduced motion, motion off, a
   * different audience or a move of more than one tick adds nothing. Creation errors
   * cancel what was created and propagate. Mutates this.animations,
   * this.cleanupClock, this.stepGlides and this.paused.
   *
   * @param {Record<string, any>} plan
   * @param {ChoreographySurface} surface
   * @param {ChoreographyInstallation} installation
   * @param {{previousPlan: Record<string, any>, carry: Map<string, readonly [number, number]>}} step
   */
  #startStepGlide(plan, surface, installation, step) {
    const glides = stepGlideSpecs(plan, surface, this.motionMode, step);
    if (glides === null) return;
    const fades = stepCueFadeSpecs(plan, installation);
    const root = battlefieldRoot(surface);
    const forward =
      Number(plan.simulatorStep) - Number(step.previousPlan?.simulatorStep) === 1;
    const lookSpecs = root
      ? [
          ...movementAnimationSpecs(root, {
            epochKey: String(plan.epochKey),
            mode: "step",
            motionMode: this.motionMode,
            beats: plan.beats,
            stepMs: BEAT_CLOCK.stepGlide,
          }),
          ...sightAnimationSpecs(root, plan, {
            motionMode: this.motionMode,
            renderPolicy: "replay_step",
            step: forward ? "forward" : "backward",
          }),
        ]
      : [];
    const specs = boundedOverrides(
      [...glides.map(({ spec }) => spec), ...fades, ...lookSpecs],
      plan,
      MAX_ACTIVE_ANIMATIONS - 1,
    );
    if (specs.length === 0) return;
    /** @type {AnimationHandle[]} */
    const created = [];
    try {
      for (const spec of specs) {
        const animation = this.animationFactory.create(spec);
        ignoreCancellation(animation.finished);
        created.push(animation);
      }
      const clock = this.animationFactory.createClock(
        installation.root,
        BEAT_CLOCK.stepCueFadeEnd,
        `mbg:${plan.epochKey}:step`,
      );
      ignoreCancellation(clock.finished);
      this.cleanupClock = clock;
    } catch (error) {
      for (const animation of created) safeCancel(animation);
      throw error;
    }
    this.animations = created;
    this.stepGlides = glides
      .map(({ spec, presentationKey, points, offsets }) => {
        const animation = created[specs.indexOf(spec)];
        return animation ? { presentationKey, points, offsets, animation } : null;
      })
      .filter((glide) => glide !== null);
    this.paused = false;
    for (const animation of this.#allAnimations()) {
      applyPlaybackRate(animation, this.playbackRate);
    }
    const generation = this.generation;
    const clock = this.cleanupClock;
    ignoreCancellation(
      /** @type {AnimationHandle} */ (clock).finished.then(() => {
        if (generation !== this.generation) return;
        this.#cancelAnimations();
        this.paused = true;
        this.#publish();
      }),
    );
  }

  /**
   * Glide the bodies back onto a frame that has no plan (a -1 onto a replay's
   * first frame).
   *
   * previousPlan is the plan of the tick shown before the step, surface the new
   * frame's surface (the durable renderer already drew that frame's END) and
   * carry where running step glides drew each body. The landing frame shows
   * each body where previousPlan's tick started, so the landing is
   * stepSourceBefore(previousPlan) and stepGlideSpecs runs backward: each body
   * with a known START spot glides 240 ms from where it was drawn (its END, or
   * a running glide's spot) to that START spot, a Charge rush back along its
   * path; deaths and returns snap, as on every step. No painter installation
   * exists for the landing frame, so there are no cue fades, movement marks or
   * sight cross-fade. A one-step clock on the battlefield root then cancels the
   * glides and marks the still paused. Reduced motion, motion off, Step Glide
   * off or no battlefield root adds nothing. Creation errors cancel what was
   * created and propagate. Mutates this.animations, this.cleanupClock,
   * this.stepGlides and this.paused.
   *
   * @param {Record<string, any>} previousPlan
   * @param {ChoreographySurface} surface
   * @param {Map<string, readonly [number, number]>} carry
   */
  #startPlanlessStepGlide(previousPlan, surface, carry) {
    const landing = stepSourceBefore(previousPlan);
    const glides = stepGlideSpecs(landing, surface, this.motionMode, {
      previousPlan,
      carry,
    });
    const root = battlefieldRoot(surface);
    if (glides === null || glides.length === 0 || !root) return;
    const specs = boundedOverrides(
      glides.map(({ spec }) => spec),
      previousPlan,
      MAX_ACTIVE_ANIMATIONS - 1,
    );
    /** @type {AnimationHandle[]} */
    const created = [];
    /** @type {AnimationHandle} */
    let clock;
    try {
      for (const spec of specs) {
        const animation = this.animationFactory.create(spec);
        ignoreCancellation(animation.finished);
        created.push(animation);
      }
      clock = this.animationFactory.createClock(
        root,
        BEAT_CLOCK.stepGlide,
        `mbg:${previousPlan.epochKey}:step-back`,
      );
      ignoreCancellation(clock.finished);
    } catch (error) {
      for (const animation of created) safeCancel(animation);
      throw error;
    }
    this.animations = created;
    this.cleanupClock = clock;
    this.stepGlides = glides
      .map(({ spec, presentationKey, points, offsets }) => {
        const animation = created[specs.indexOf(spec)];
        return animation ? { presentationKey, points, offsets, animation } : null;
      })
      .filter((glide) => glide !== null);
    this.paused = false;
    for (const animation of this.#allAnimations()) {
      applyPlaybackRate(animation, this.playbackRate);
    }
    const generation = this.generation;
    ignoreCancellation(
      clock.finished.then(() => {
        if (generation !== this.generation) return;
        this.#cancelAnimations();
        this.paused = true;
        this.#publish();
      }),
    );
  }

  /**
   * Read where each running step glide draws its body right now, in world units.
   *
   * surface is the new frame's surface (unused for the math; world spots do not
   * depend on the camera). Return a Map from presentation key to the drawn world
   * spot for glides whose animation reports a finite currentTime inside the glide.
   * Pure read of animation state; called just before those glides are cancelled.
   *
   * @param {ChoreographySurface} surface
   * @returns {Map<string, readonly [number, number]>}
   */
  #stepCarry(surface) {
    void surface;
    /** @type {Map<string, readonly [number, number]>} */
    const carry = new Map();
    for (const glide of this.stepGlides) {
      const time = glide.animation.currentTime;
      if (typeof time !== "number" || !Number.isFinite(time)) continue;
      if (time <= 0 || time >= BEAT_CLOCK.stepGlide) continue;
      carry.set(
        glide.presentationKey,
        pointAlongGlide(glide.points, glide.offsets, time / BEAT_CLOCK.stepGlide),
      );
    }
    return carry;
  }

  /**
   * Ease the camera from its previous frame to a new one (the camera entry point).
   *
   * wrapper is the camera wrapper group that holds every arena layer; the durable
   * renderer has already re-rendered at the new frame (END). fromTransform and
   * toTransform are the viewport transforms before and after the change (each with
   * worldToScreen). options.instant (default false) cuts instead: use it for an Agent
   * POV or view change, which never eases. Reduced motion and motion off always cut.
   * Otherwise one 520 ms Web Animation (id "mbg:camera:<n>", ease-in-out, fill
   * "backwards", no forward fill) maps the new picture back onto the old one and eases
   * to none, so finishing or cancelling it shows the exact new frame. A running ease is
   * cancelled first. Playback rate does not change it (it is not tick time); skip and
   * dispose finish or cancel it. Return the frozen snapshot. Transforms that are not
   * uniform scale and translate (or a null wrapper) cut. Errors from Element.animate
   * propagate.
   *
   * @param {Element | null} wrapper
   * @param {{worldToScreen: (point: readonly [number, number]) => {x: number, y: number}} | null} fromTransform
   * @param {{worldToScreen: (point: readonly [number, number]) => {x: number, y: number}} | null} toTransform
   * @param {{instant?: boolean}} [options]
   */
  easeCamera(wrapper, fromTransform, toTransform, options = {}) {
    this.#cancelCameraEase();
    this.#cancelCameraLean();
    const matrix =
      wrapper && fromTransform && toTransform
        ? cameraEaseMatrix(fromTransform, toTransform)
        : null;
    if (
      matrix === null ||
      options.instant === true ||
      this.motionMode !== "normal" ||
      (matrix.k === 1 && Math.abs(matrix.x) < 0.01 && Math.abs(matrix.y) < 0.01)
    ) {
      this.#publish();
      return this.snapshot();
    }
    this.cameraGeneration += 1;
    const generation = this.cameraGeneration;
    const animation = this.animationFactory.create({
      element: /** @type {Element} */ (wrapper),
      keyframes: [
        {
          transform: `matrix(${matrix.k}, 0, 0, ${matrix.k}, ${matrix.x}, ${matrix.y})`,
        },
        { transform: "matrix(1, 0, 0, 1, 0, 0)" },
      ],
      options: { duration: CAMERA_EASE_MS, easing: "ease-in-out", fill: "backwards" },
      id: `mbg:camera:${generation}`,
    });
    ignoreCancellation(animation.finished);
    this.cameraEase = animation;
    ignoreCancellation(
      animation.finished.then(() => {
        if (generation !== this.cameraGeneration || this.cameraEase !== animation)
          return;
        this.cameraEase = null;
        this.#publish();
      }),
    );
    this.#publish();
    return this.snapshot();
  }

  /**
   * Lean the camera toward this tick's seen fall (Kill Cam) for one played tick.
   *
   * wrapper is the battlefield camera wrapper (g.battlefield-camera); lean is the
   * {keyframes, options} pair from looks/camera.js killCamKeyframes (options carry
   * the tick length, linear easing and fill "backwards", so finishing or cancelling
   * shows the resting frame). Only in normal motion with a played tick installed
   * (not paused, not a still): the lean is created through the animation factory
   * (id "mbg:camera-lean:<n>"), counted with this tick's animations, scaled by the
   * playback rate and paused with the tick, so a seek, skip, view change, recipient
   * change, a camera ease or dispose cancels it (END: the resting frame). A running
   * lean is cancelled first. Return true when a lean started, otherwise false.
   * Errors from Element.animate propagate. Agent POV: the caller builds the lean only
   * from this audience's own plan (killCamLean), never from a hidden fall.
   *
   * @param {Element | null} wrapper
   * @param {{keyframes: Keyframe[], options: KeyframeAnimationOptions} | null} lean
   * @returns {boolean}
   */
  leanCamera(wrapper, lean) {
    this.#cancelCameraLean();
    if (
      wrapper === null ||
      lean === null ||
      this.motionMode !== "normal" ||
      this.paused ||
      this.installation === null ||
      this.renderPolicy === "replay_static"
    ) {
      return false;
    }
    this.cameraGeneration += 1;
    const animation = this.animationFactory.create({
      element: wrapper,
      keyframes: lean.keyframes,
      options: { ...lean.options },
      id: `mbg:camera-lean:${this.cameraGeneration}`,
    });
    ignoreCancellation(animation.finished);
    applyPlaybackRate(animation, this.playbackRate);
    this.animations = [...this.animations, animation];
    this.cameraLean = animation;
    return true;
  }

  /**
   * Cancel a running Kill Cam lean (the resting frame shows). Return undefined.
   */
  #cancelCameraLean() {
    if (this.cameraLean) {
      safeCancel(this.cameraLean);
      this.animations = this.animations.filter((handle) => handle !== this.cameraLean);
      this.cameraLean = null;
    }
  }

  /**
   * Cancel a running camera ease (the new frame shows at once). Return undefined.
   */
  #cancelCameraEase() {
    if (this.cameraEase) {
      safeCancel(this.cameraEase);
      this.cameraEase = null;
    }
  }

  /**
   * Cancel this installation before an authority or presentation replacement.
   *
   * reason is passed to painter.clear. Increment generation to invalidate old promise
   * handlers, capture logical time, cancel handles and release the gate. Keep the plan,
   * surface and render policy for the caller to replace. Return undefined.
   *
   * @param {string} reason
   */
  #clearOwned(reason) {
    this.generation += 1;
    this.#captureLogicalTime();
    this.#cancelAnimations();
    this.#clearFadingCues();
    if (this.installation) {
      this.painter.clear(this.installation, reason);
    }
    this.installation = null;
    this.submissionBlocked = false;
  }

  /**
   * Fade the previous played tick's kept cues out at the start of the next
   * played tick, then remove them.
   *
   * installation is the previous tick's settled installation, already detached
   * from this controller. The painter removes its herald root and persistent
   * nodes at once (the new tick draws those again) and hands back the roots that
   * hold the kept cues. Each root's END opacity is 0 (the DOM holds END, as
   * every beat does) and one Web Animation takes it from 1 to 0 over
   * CUE_FADE_OUT_MS at the playback rate, with fill "backwards"; when it ends
   * the root is removed. These fades are not part of the tick's settlement, so
   * they never delay autoplay. Return undefined.
   *
   * @param {Record<string, any>} installation
   * @param {string} epochKey The new tick's epoch, for the animation ids.
   */
  #fadeOutCues(installation, epochKey) {
    const roots = this.painter.releaseForFade?.(installation) ?? [];
    roots.forEach((root, index) => {
      root.setAttribute("opacity", "0");
      const animation = this.animationFactory.create({
        element: root,
        keyframes: [{ opacity: 1 }, { opacity: 0 }],
        options: { duration: CUE_FADE_OUT_MS, fill: "backwards" },
        id: `mbg:${epochKey}:cue-fade-out:${index}`,
      });
      applyPlaybackRate(animation, this.playbackRate);
      const entry = { root, animation };
      this.fadingCues.push(entry);
      ignoreCancellation(
        animation.finished.then(() => {
          root.remove();
          this.fadingCues = this.fadingCues.filter((other) => other !== entry);
        }),
      );
    });
  }

  /**
   * Cancel every cue fade-out and remove its roots at once (a seek, a step, a
   * view change or a new tick: the Settled Still never shows a fading cue).
   * Return undefined.
   */
  #clearFadingCues() {
    for (const { root, animation } of this.fadingCues) {
      safeCancel(animation);
      root.remove();
    }
    this.fadingCues = [];
  }

  /**
   * Capture finite cleanup-clock time, falling back to gate or saved time.
   *
   * Clamp a numeric clock to zero through the authored total when that total is finite.
   * Non-numeric clock values leave saved time unchanged. Mutate logicalTime only.
   */
  #captureLogicalTime() {
    const candidate =
      this.cleanupClock?.currentTime ?? this.gateClock?.currentTime ?? this.logicalTime;
    if (typeof candidate === "number" && Number.isFinite(candidate)) {
      const authoredTotal = Number(this.plan?.phases?.total ?? candidate);
      this.logicalTime = Number.isFinite(authoredTotal)
        ? Math.min(Math.max(candidate, 0), Math.max(authoredTotal, 0))
        : candidate;
    }
  }

  /**
   * Return a new array of owned visual handles followed by gate and cleanup clocks.
   *
   * The visual handles include the painter's event animations and this controller's
   * overrides (glides, beat holds, step cue fades). The camera ease is separate.
   * Absent clocks are omitted. Handles remain the controller's mutable browser objects;
   * this helper does not advance, cancel or copy them.
   */
  #allAnimations() {
    return [
      ...this.animations,
      ...(this.gateClock ? [this.gateClock] : []),
      ...(this.cleanupClock ? [this.cleanupClock] : []),
    ];
  }

  /**
   * Best-effort cancel every owned handle and clear all handle references.
   *
   * Cancelling an override shows END on its node. Cancellation errors are ignored.
   * Painter nodes, plan identity, logical time and submission gating are left to the
   * caller. Return undefined.
   */
  #cancelAnimations() {
    for (const animation of this.#allAnimations()) {
      safeCancel(animation);
    }
    this.animations = [];
    this.stepGlides = [];
    this.gateClock = null;
    this.cleanupClock = null;
    this.cameraLean = null;
  }

  /**
   * Notify the owner with a snapshot and release waiters when settled.
   *
   * Call onStateChange synchronously, then resolve and clear all queued settlement
   * waiters if no animations or submission gate remain. Callback errors propagate.
   */
  #publish() {
    const snapshot = this.snapshot();
    this.onStateChange(snapshot);
    if (this.#isSettled()) {
      const waiters = [...this.settleWaiters];
      this.settleWaiters.clear();
      for (const resolve of waiters) {
        resolve();
      }
    }
  }

  /**
   * Return true exactly when there are no owned animations, no camera ease and no
   * submission gate.
   *
   * A persistent or static installation may still be visible; settlement concerns its
   * presentation clock, not removal of durable explanatory nodes.
   */
  #isSettled() {
    return (
      this.#allAnimations().length === 0 &&
      this.cameraEase === null &&
      !this.submissionBlocked
    );
  }
}

/**
 * Return whether value is a non-null object other than an array.
 *
 * This broad shape guard does not validate a plan or its authority.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return a recognized live_once, replay_animated or replay_static value.
 *
 * Any other value throws RangeError; no coercion or fallback is applied.
 *
 * @param {unknown} value
 * @returns {RenderPolicy}
 */
function normalizeRenderPolicy(value) {
  if (
    value === "live_once" ||
    value === "replay_animated" ||
    value === "replay_static"
  ) {
    return value;
  }
  throw new RangeError(`unknown render policy ${String(value)}.`);
}

/**
 * Return a recognized render request: a render policy or "replay_step".
 *
 * value "replay_step" is returned as is; anything else must be a render policy (see
 * normalizeRenderPolicy), otherwise RangeError is thrown. No coercion.
 *
 * @param {unknown} value
 * @returns {RenderRequest}
 */
function normalizeRenderRequest(value) {
  return value === "replay_step" ? "replay_step" : normalizeRenderPolicy(value);
}

/**
 * Return the element whose subtree holds the durable nodes a plan may override.
 *
 * surface.battlefield wins; otherwise the SVG that owns surface.layer. Return null
 * when there is no DOM (unit-test surfaces), so no override is made. Pure.
 *
 * @param {ChoreographySurface | null | undefined} surface
 * @returns {Element | null}
 */
function battlefieldRoot(surface) {
  if (!surface) return null;
  if (
    surface.battlefield &&
    typeof surface.battlefield.querySelectorAll === "function"
  ) {
    return surface.battlefield;
  }
  const layer = /** @type {any} */ (surface.layer);
  const owner = layer?.ownerSVGElement ?? layer?.closest?.("svg") ?? null;
  return owner && typeof owner.querySelectorAll === "function" ? owner : null;
}

/**
 * Return a lookup from presentation key to a body's glide wrapper.
 *
 * surface.bodyWrapper is used when provided. Otherwise scan root once for glide
 * wrappers (elements with data-glide-key) and key each by its own
 * data-presentation-key or by its g.agent child's data-presentation-key. Return a
 * function giving the wrapper or null. Reads the DOM only.
 *
 * @param {ChoreographySurface} surface
 * @param {Element} root
 * @returns {(presentationKey: string) => Element | null}
 */
function bodyWrapperLookup(surface, root) {
  if (typeof surface.bodyWrapper === "function") {
    const lookup = surface.bodyWrapper;
    return (presentationKey) => lookup(presentationKey) ?? null;
  }
  /** @type {Map<string, Element>} */
  const wrappers = new Map();
  for (const wrapper of root.querySelectorAll("[data-glide-key]")) {
    const key =
      wrapper.getAttribute("data-presentation-key") ??
      wrapper
        .querySelector(".agent[data-presentation-key]")
        ?.getAttribute("data-presentation-key") ??
      null;
    if (key) wrappers.set(key, wrapper);
  }
  return (presentationKey) => wrappers.get(presentationKey) ?? null;
}

/**
 * Describe one FLIP glide of a body wrapper as a Web Animations spec.
 *
 * element is the wrapper; points are world spots in order (the last is END, where
 * the DOM already draws the body) at keyframe offsets between 0 and 1; surface
 * projects world to screen. duration is in ms; id names the animation. Each
 * keyframe translates the wrapper by its spot minus END in CSS pixels (rounded to a
 * thousandth of a pixel), added on top
 * of the wrapper's own transform (composite "add"), easing on the shared glide
 * curve; the last keyframe is no translation. fill "backwards", no forward fill, so
 * finishing or cancelling shows END. Return null for a non-finite projection. Pure.
 *
 * @param {Element} element
 * @param {ReadonlyArray<readonly [number, number]>} points
 * @param {ReadonlyArray<number>} offsets
 * @param {ChoreographySurface} surface
 * @param {number} duration
 * @param {string} id
 * @returns {AnimationSpec | null}
 */
function glideSpec(element, points, offsets, surface, duration, id) {
  const end = surface.worldToScreen(points[points.length - 1]);
  /** @type {Keyframe[]} */
  const keyframes = [];
  for (const [index, world] of points.entries()) {
    const screen = surface.worldToScreen(world);
    const dx = screen.x - end.x;
    const dy = screen.y - end.y;
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) return null;
    const last = index === points.length - 1;
    keyframes.push({
      offset: offsets[index],
      transform: last
        ? "translate(0px, 0px)"
        : `translate(${roundPixel(dx)}px, ${roundPixel(dy)}px)`,
      ...(last ? {} : { easing: GLIDE_EASING }),
    });
  }
  return {
    element,
    keyframes,
    options: { delay: 0, duration, fill: "backwards", composite: "add" },
    id,
  };
}

/**
 * Round a pixel offset to a thousandth of a pixel (no visible change) so glide
 * transforms read cleanly. Pure.
 *
 * @param {number} value
 */
function roundPixel(value) {
  const rounded = Math.round(value * 1000) / 1000;
  return Object.is(rounded, -0) ? 0 : rounded;
}

/**
 * Return whether two world spots are the same within a millionth of a unit.
 *
 * @param {readonly [number, number]} left
 * @param {readonly [number, number]} right
 */
function sameSpot(left, right) {
  return Math.hypot(left[0] - right[0], left[1] - right[1]) <= 1e-6;
}

/**
 * Build the body glides of a played tick.
 *
 * plan carries beats (glide: Body Glide on) and bodies (planBodies records);
 * surface supplies projection and the durable DOM; motionMode is the controller's.
 * Only normal motion with Body Glide on glides. Each body whose record says glide
 * moves START to END over 0-260 ms; a body that charged reaches its post-Charge spot
 * at 110 ms first. A body whose path cuts a stone (body.dimShift, from the plan)
 * takes the Movement look's Dim Shift (move plus fade) instead of a plain glide.
 * Bodies with no wrapper in the DOM are skipped. Return a new array
 * of specs (ids mbg:<epoch>:glide:<public ID>). Reads the DOM only.
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographySurface} surface
 * @param {MotionMode} motionMode
 * @returns {AnimationSpec[]}
 */
function playedGlideSpecs(plan, surface, motionMode) {
  if (motionMode !== "normal" || plan.beats?.glide !== true) return [];
  const root = battlefieldRoot(surface);
  if (!root) return [];
  const wrapperFor = bodyWrapperLookup(surface, root);
  /** @type {AnimationSpec[]} */
  const specs = [];
  for (const body of Array.isArray(plan.bodies) ? plan.bodies : []) {
    if (body?.glide !== true || !body.start) continue;
    const element = wrapperFor(body.presentationKey);
    if (!element) continue;
    const charged = body.charged === true && body.postCharge;
    const points = charged
      ? [body.start, body.postCharge, body.end]
      : [body.start, body.end];
    const offsets = charged
      ? [0, BEAT_CLOCK.chargeRushEnd / BEAT_CLOCK.glideEnd, 1]
      : [0, 1];
    const id = `mbg:${plan.epochKey}:glide:${body.publicAgentId}`;
    if (body.dimShift === true) {
      // A path through stone dims out and back in instead of sliding over it.
      specs.push(
        ...dimShiftSpecs(
          element,
          points,
          offsets,
          surface.worldToScreen,
          BEAT_CLOCK.glideEnd,
          id,
        ),
      );
      continue;
    }
    const spec = glideSpec(element, points, offsets, surface, BEAT_CLOCK.glideEnd, id);
    if (spec) specs.push(spec);
  }
  return specs;
}

/**
 * Return the real time (ms, normal motion) of one beat-held node's beat.
 *
 * beats is plan.beats; kind is "impact" (blow time 260 ms on the tick's clock, for
 * every key), "settle" (the Settle window start, or the end of the dramatic hold on
 * a tick with a Fall or Return beat), "fall" (the champion's Fall beat), "return"
 * (its Return beat start), "set-down" (when it is set down on its pad) or "score"
 * (the Fall beat start, on every tick); key is the champion's presentation key.
 * Return null when this plan has no such beat for that champion (for example a
 * death the Agent POV team did not see), so the node shows END at once. Pure.
 *
 * @param {Record<string, any>} beats
 * @param {string} kind
 * @param {string | null} key
 * @returns {number | null}
 */
function beatTimeFor(beats, kind, key) {
  if (kind === "score") {
    return Number.isFinite(beats.scoreAt) ? Number(beats.scoreAt) : null;
  }
  if (kind === "impact") {
    return beatRealTime(
      BEAT_CLOCK.impactBlow,
      beats.glide === true,
      Number.isFinite(beats.lead) ? Number(beats.lead) : 0,
    );
  }
  if (kind === "settle") {
    // On a tick with a dramatic hold, settled marks wait for the hold to end.
    if (beats.fall || beats.returnBeat) {
      return Number.isFinite(beats.explanationEnd)
        ? Number(beats.explanationEnd)
        : null;
    }
    const settle = beats.windows?.settle;
    return Array.isArray(settle) && Number.isFinite(settle[0])
      ? Number(settle[0])
      : null;
  }
  if (key === null) return null;
  if (kind === "fall") {
    const entry = (Array.isArray(beats.falls) ? beats.falls : []).find(
      (fall) => fall?.presentationKey === key,
    );
    return Number.isFinite(entry?.at) ? Number(entry.at) : null;
  }
  const entry = (Array.isArray(beats.returns) ? beats.returns : []).find(
    (returned) => returned?.presentationKey === key,
  );
  const value = kind === "set-down" ? entry?.setDownAt : entry?.at;
  return Number.isFinite(value) ? Number(value) : null;
}

/**
 * Build the beat holds of a played tick (the dramatic hold).
 *
 * plan carries beats; surface supplies the battlefield and optional
 * beatHoldRoots; motionMode is normal or reduced (reduced multiplies every beat
 * time by beats.reducedScale; fades keep their length). Each node with
 * data-beat-hold, data-beat-key and data-beat-side whose beat is in the plan gets
 * one opacity override: a "before" node goes from data-beat-from (default 1) to its
 * END opacity (its opacity attribute, default 0); an "after" node from
 * data-beat-from (default 0) to its END opacity (default 1). The change starts at
 * the beat and lasts data-beat-fade ms (default a 1 ms cut); fill "backwards"
 * holds the earlier look before it, and nothing holds after it. Return a new array
 * of specs (ids mbg:<epoch>:hold:<kind>:<n>). A node with data-beat-pulse="p"
 * instead flashes 0 -> p (at 37.5% of the fade) -> 0 from its beat, in normal
 * motion only. Reads the DOM only.
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographySurface} surface
 * @param {MotionMode} motionMode
 * @returns {AnimationSpec[]}
 */
function beatHoldSpecs(plan, surface, motionMode) {
  const beats = plan.beats;
  if (motionMode === "off" || !beats || typeof beats !== "object") return [];
  const scale = motionMode === "reduced" ? Number(beats.reducedScale ?? 1) : 1;
  const roots = [
    battlefieldRoot(surface),
    ...(Array.isArray(surface.beatHoldRoots) ? surface.beatHoldRoots : []),
  ].filter(
    (root, index, all) =>
      root !== null &&
      typeof root?.querySelectorAll === "function" &&
      all.indexOf(root) === index,
  );
  /** @type {AnimationSpec[]} */
  const specs = [];
  for (const root of roots) {
    for (const element of /** @type {Element} */ (root).querySelectorAll(
      "[data-beat-hold]",
    )) {
      const kind = element.getAttribute("data-beat-hold") ?? "";
      const side = element.getAttribute("data-beat-side");
      if (!BEAT_HOLD_KINDS.has(kind) || (side !== "before" && side !== "after")) {
        continue;
      }
      const at = beatTimeFor(beats, kind, element.getAttribute("data-beat-key"));
      if (at === null) continue;
      const fade = Math.max(1, Number(element.getAttribute("data-beat-fade")) || 0);
      const pulse = element.getAttribute("data-beat-pulse");
      if (pulse !== null) {
        // A flash: 0 up to the peak and back to 0. Reduced motion skips it.
        if (motionMode !== "normal" || !Number.isFinite(Number(pulse))) continue;
        specs.push({
          element,
          keyframes: [
            { opacity: 0 },
            { opacity: Number(pulse), offset: 0.375 },
            { opacity: 0 },
          ],
          options: { delay: at * scale, duration: fade, fill: "backwards" },
          id: `mbg:${plan.epochKey}:hold:${kind}:${specs.length}`,
        });
        continue;
      }
      const endAttribute = element.getAttribute("opacity");
      const end =
        endAttribute !== null && Number.isFinite(Number(endAttribute))
          ? Number(endAttribute)
          : side === "before"
            ? 0
            : 1;
      const fromAttribute = element.getAttribute("data-beat-from");
      const from =
        fromAttribute !== null && Number.isFinite(Number(fromAttribute))
          ? Number(fromAttribute)
          : side === "before"
            ? 1
            : 0;
      specs.push({
        element,
        keyframes: [{ opacity: from }, { opacity: end }],
        options: { delay: at * scale, duration: fade, fill: "backwards" },
        id: `mbg:${plan.epochKey}:hold:${kind}:${specs.length}`,
      });
    }
  }
  return specs;
}

/**
 * Build the hit shivers of a played tick.
 *
 * plan carries beats; surface supplies the battlefield; motionMode and
 * playbackRate are the controller's. Only normal motion below 1.5x playback
 * shivers. Each element with data-beat-shiver (a distance in CSS px) and
 * data-beat-shiver-at="impact" gets one sideways shake: translateX 0, a, -a, a, 0
 * at offsets 0, 0.2, 0.5, 0.8 and 1, starting at Impact plus data-beat-delay ms and
 * lasting data-beat-fade ms (default 120), with no fill, so it never changes the
 * END look. Return a new array of specs (ids mbg:<epoch>:shiver:<n>). Reads the
 * DOM only.
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographySurface} surface
 * @param {MotionMode} motionMode
 * @param {number} playbackRate
 * @returns {AnimationSpec[]}
 */
function shiverSpecs(plan, surface, motionMode, playbackRate) {
  const beats = plan.beats;
  if (motionMode !== "normal" || playbackRate >= SHIVER_RATE_LIMIT || !beats) return [];
  const root = battlefieldRoot(surface);
  if (!root) return [];
  const impact = beatTimeFor(beats, "impact", null);
  if (impact === null) return [];
  /** @type {AnimationSpec[]} */
  const specs = [];
  for (const element of root.querySelectorAll("[data-beat-shiver]")) {
    if (element.getAttribute("data-beat-shiver-at") !== "impact") continue;
    const distance = Number(element.getAttribute("data-beat-shiver"));
    if (!Number.isFinite(distance) || distance === 0) continue;
    const delay = Number(element.getAttribute("data-beat-delay")) || 0;
    const duration = Math.max(1, Number(element.getAttribute("data-beat-fade")) || 120);
    specs.push({
      element,
      keyframes: [
        { transform: "translateX(0px)", offset: 0 },
        { transform: `translateX(${distance}px)`, offset: 0.2 },
        { transform: `translateX(${-distance}px)`, offset: 0.5 },
        { transform: `translateX(${distance}px)`, offset: 0.8 },
        { transform: "translateX(0px)", offset: 1 },
      ],
      options: { delay: impact + delay, duration, composite: "add" },
      id: `mbg:${plan.epochKey}:shiver:${specs.length}`,
    });
  }
  return specs;
}

/**
 * Build the Death Dimming override of a played tick.
 *
 * plan names the epoch; surface supplies the battlefield; motionMode is the
 * controller's. In normal motion each `path.death-dim` the renderer drew (with
 * data-dim-start and data-dim-end, this audience's Fall beat in ms) gets the
 * Atmosphere look's dimAnimation. Return a new array of specs. Reads the DOM only.
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographySurface} surface
 * @param {MotionMode} motionMode
 * @returns {AnimationSpec[]}
 */
function deathDimSpecs(plan, surface, motionMode) {
  if (motionMode !== "normal") return [];
  const root = battlefieldRoot(surface);
  if (!root) return [];
  /** @type {AnimationSpec[]} */
  const specs = [];
  for (const element of root.querySelectorAll("path.death-dim")) {
    const start = Number(element.getAttribute("data-dim-start"));
    const end = Number(element.getAttribute("data-dim-end"));
    if (!Number.isFinite(start) || !(end > start)) continue;
    const spec = dimAnimation(element, { kind: "death-dim", start, end }, plan);
    specs.push({
      element: spec.element,
      keyframes: spec.keyframes,
      options: { ...spec.options },
      id: spec.id,
    });
  }
  return specs;
}

/**
 * Build the look-owned overrides of a played tick: movement marks, the Agent POV
 * sight veil, hit shivers and Death Dimming.
 *
 * plan, surface and renderPolicy describe the tick; motionMode and playbackRate are
 * the controller's. Return a new array of specs in that order. Reads the DOM only.
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographySurface} surface
 * @param {MotionMode} motionMode
 * @param {number} playbackRate
 * @param {RenderPolicy} renderPolicy
 * @returns {AnimationSpec[]}
 */
function playedLookSpecs(plan, surface, motionMode, playbackRate, renderPolicy) {
  const root = battlefieldRoot(surface);
  if (!root || motionMode === "off") return [];
  return [
    ...movementAnimationSpecs(root, {
      epochKey: String(plan.epochKey),
      mode: "played",
      motionMode,
      beats: plan.beats,
    }),
    ...sightAnimationSpecs(root, plan, { motionMode, renderPolicy }),
    ...shiverSpecs(plan, surface, motionMode, playbackRate),
    ...deathDimSpecs(plan, surface, motionMode),
  ];
}

/**
 * Build the body glides of a one-tick step (replay_step).
 *
 * plan is the new frame's plan; surface its surface; motionMode the controller's;
 * step {previousPlan, carry}. Return null when the step does not glide: motion is
 * not normal, Step Glide is off (plan.beats.stepGlide false), the audience changed
 * (audienceKey differs: another view or recipient) or the frames are not exactly
 * one tick apart. Otherwise return a new array (possibly empty) of
 * {spec, presentationKey, points, offsets}: one 240 ms glide per body present in
 * both plans, alive or fallen at both ends, that did not just return to a pad
 * (forwards) or leave one (backwards), from carry's drawn spot (when within 3 world
 * units of its previous END spot) or its previous END spot, to its new END spot.
 * Forwards, a Charge passes its post-Charge spot at 100 ms; backwards, the left
 * tick's Charge is reversed with its post-Charge spot at 140 ms. A body whose path
 * cuts a stone (dimShift on the new body forwards, on the earlier body backwards)
 * gets the Movement look's Dim Shift pair instead (two entries with the same key).
 * In Agent POV the plans hold only seen bodies, so a body unseen in either frame
 * never glides.
 * Reads the DOM only.
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographySurface} surface
 * @param {MotionMode} motionMode
 * @param {{previousPlan: Record<string, any>, carry: Map<string, readonly [number, number]>}} step
 * @returns {Array<{spec: AnimationSpec, presentationKey: string, points: ReadonlyArray<readonly [number, number]>, offsets: ReadonlyArray<number>}> | null}
 */
function stepGlideSpecs(plan, surface, motionMode, step) {
  const previous = step.previousPlan;
  if (motionMode !== "normal" || plan.beats?.stepGlide === false) return null;
  if (
    typeof plan.audienceKey !== "string" ||
    previous?.audienceKey !== plan.audienceKey
  ) {
    return null;
  }
  const direction = Number(plan.simulatorStep) - Number(previous.simulatorStep);
  if (direction !== 1 && direction !== -1) return null;
  const root = battlefieldRoot(surface);
  if (!root) return [];
  const wrapperFor = bodyWrapperLookup(surface, root);
  const before = new Map(
    (Array.isArray(previous.bodies) ? previous.bodies : []).map((body) => [
      body.presentationKey,
      body,
    ]),
  );
  /** @type {Array<{spec: AnimationSpec, presentationKey: string, points: ReadonlyArray<readonly [number, number]>, offsets: ReadonlyArray<number>}>} */
  const glides = [];
  for (const body of Array.isArray(plan.bodies) ? plan.bodies : []) {
    const earlier = before.get(body.presentationKey);
    if (!earlier || earlier.alive !== body.alive) continue;
    if ((direction === 1 && body.returned) || (direction === -1 && earlier.returned)) {
      continue;
    }
    const carried = step.carry.get(body.presentationKey);
    const from =
      carried &&
      Math.hypot(carried[0] - earlier.end[0], carried[1] - earlier.end[1]) <=
        STEP_CARRY_LIMIT
        ? carried
        : earlier.end;
    const forwardCharge = direction === 1 && body.charged === true && body.postCharge;
    const backwardCharge =
      direction === -1 && earlier.charged === true && earlier.postCharge;
    /** @type {ReadonlyArray<readonly [number, number]>} */
    const points = forwardCharge
      ? [from, body.postCharge, body.end]
      : backwardCharge
        ? [from, earlier.postCharge, body.end]
        : [from, body.end];
    /** @type {ReadonlyArray<number>} */
    const offsets = forwardCharge
      ? [0, BEAT_CLOCK.stepChargeRushForward / BEAT_CLOCK.stepGlide, 1]
      : backwardCharge
        ? [0, BEAT_CLOCK.stepChargeRushBackward / BEAT_CLOCK.stepGlide, 1]
        : [0, 1];
    if (points.every((spot) => sameSpot(spot, body.end))) continue;
    const element = wrapperFor(body.presentationKey);
    if (!element) continue;
    const id = `mbg:${plan.epochKey}:step-glide:${body.publicAgentId}`;
    if (direction === 1 ? body.dimShift === true : earlier.dimShift === true) {
      for (const spec of dimShiftSpecs(
        element,
        points,
        offsets,
        surface.worldToScreen,
        BEAT_CLOCK.stepGlide,
        id,
      )) {
        glides.push({ spec, presentationKey: body.presentationKey, points, offsets });
      }
      continue;
    }
    const spec = glideSpec(element, points, offsets, surface, BEAT_CLOCK.stepGlide, id);
    if (spec) {
      glides.push({ spec, presentationKey: body.presentationKey, points, offsets });
    }
  }
  return glides;
}

/**
 * Build the cue fade of a one-tick step: the painter's roots fade in late.
 *
 * plan names the epoch; installation holds the painter roots (root, connectorRoot,
 * routeRoot). Each present root gets opacity 0 until 240 ms, then fades to its END
 * opacity by 320 ms (fill "backwards", no forward fill), so a stroke never stretches
 * between moving bodies. Return a new array of specs (ids mbg:<epoch>:step-cue:<n>).
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographyInstallation} installation
 * @returns {AnimationSpec[]}
 */
function stepCueFadeSpecs(plan, installation) {
  const record = /** @type {Record<string, any>} */ (installation);
  return [record.routeRoot, record.connectorRoot, record.root]
    .filter((element) => element && typeof element === "object")
    .map((element, index) => ({
      element,
      keyframes: [{ opacity: 0 }, { opacity: 1 }],
      options: {
        delay: BEAT_CLOCK.stepGlide,
        duration: BEAT_CLOCK.stepCueFadeEnd - BEAT_CLOCK.stepGlide,
        fill: "backwards",
      },
      id: `mbg:${plan.epochKey}:step-cue:${index}`,
    }));
}

/**
 * Keep overrides inside the plan's override budget and the free animation room.
 *
 * specs are glides first, then holds; plan.bounds.overrideAnimations is the plan's
 * allowance (a plan without one uses room); room is how many animations the
 * animation limit (PLAN_RESOURCE_CAPS.animations) still allows. Return the first specs that fit: extra beat holds are dropped,
 * never the tick. Pure.
 *
 * @param {ReadonlyArray<AnimationSpec>} specs
 * @param {Record<string, any>} plan
 * @param {number} room
 * @returns {AnimationSpec[]}
 */
function boundedOverrides(specs, plan, room) {
  const allowance = Number.isInteger(plan.bounds?.overrideAnimations)
    ? Number(plan.bounds.overrideAnimations)
    : room;
  return specs.slice(0, Math.max(0, Math.min(allowance, room)));
}

/**
 * Evaluate the shared glide curve, cubic-bezier(0.45, 0, 0.55, 1), at progress x.
 *
 * x is clamped to 0..1. Return the eased progress, found by bisection on the curve's
 * x polynomial (accurate to about 1e-6). Pure.
 *
 * @param {number} x
 */
function glideCurve(x) {
  const target = Math.min(1, Math.max(0, x));
  /**
   * Evaluate one coordinate of the curve's cubic Bezier at curve parameter t (0..1),
   * with control values a and b and end points 0 and 1. Pure.
   *
   * @param {number} t @param {number} a @param {number} b
   */
  const cubic = (t, a, b) =>
    3 * (1 - t) * (1 - t) * t * a + 3 * (1 - t) * t * t * b + t * t * t;
  let low = 0;
  let high = 1;
  for (let step = 0; step < 40; step += 1) {
    const middle = (low + high) / 2;
    if (cubic(middle, 0.45, 0.55) < target) low = middle;
    else high = middle;
  }
  return cubic((low + high) / 2, 0, 1);
}

/**
 * Return where a glide draws its body at progress p (0..1), in world units.
 *
 * points and offsets are the glide's world spots and keyframe offsets; each segment
 * eases on the shared glide curve, as the browser plays it. Pure.
 *
 * @param {ReadonlyArray<readonly [number, number]>} points
 * @param {ReadonlyArray<number>} offsets
 * @param {number} progress
 * @returns {readonly [number, number]}
 */
function pointAlongGlide(points, offsets, progress) {
  const p = Math.min(1, Math.max(0, progress));
  for (let index = 1; index < points.length; index += 1) {
    if (p <= offsets[index] || index === points.length - 1) {
      const span = offsets[index] - offsets[index - 1];
      const local = glideCurve(span > 0 ? (p - offsets[index - 1]) / span : 1);
      const [x0, y0] = points[index - 1];
      const [x1, y1] = points[index];
      return Object.freeze([x0 + (x1 - x0) * local, y0 + (y1 - y0) * local]);
    }
  }
  return points[points.length - 1];
}

/**
 * Find the transform that puts the new camera's picture where the old one was.
 *
 * fromTransform and toTransform project world to screen before and after a camera
 * change. Both must be a uniform scale with the same axis flip plus a translation.
 * Return {k, x, y}: screen_old = k * screen_new + (x, y), or null when either
 * transform is not of that form or is degenerate. Pure.
 *
 * @param {{worldToScreen: (point: readonly [number, number]) => {x: number, y: number}}} fromTransform
 * @param {{worldToScreen: (point: readonly [number, number]) => {x: number, y: number}}} toTransform
 * @returns {{k: number, x: number, y: number} | null}
 */
function cameraEaseMatrix(fromTransform, toTransform) {
  const a0 = fromTransform.worldToScreen([0, 0]);
  const a1 = fromTransform.worldToScreen([1, 0]);
  const a2 = fromTransform.worldToScreen([0, 1]);
  const b0 = toTransform.worldToScreen([0, 0]);
  const b1 = toTransform.worldToScreen([1, 0]);
  const b2 = toTransform.worldToScreen([0, 1]);
  const fromScale = a1.x - a0.x;
  const toScale = b1.x - b0.x;
  const values = [a0.x, a0.y, fromScale, toScale, b0.x, b0.y];
  if (
    !values.every(Number.isFinite) ||
    Math.abs(toScale) < 1e-9 ||
    Math.abs(a2.y - a0.y + fromScale) > 1e-6 ||
    Math.abs(b2.y - b0.y + toScale) > 1e-6 ||
    Math.abs(a1.y - a0.y) > 1e-6 ||
    Math.abs(b1.y - b0.y) > 1e-6
  ) {
    return null;
  }
  const k = fromScale / toScale;
  return { k, x: a0.x - k * b0.x, y: a0.y - k * b0.y };
}

/**
 * Return whether policy selects either replay presentation mode.
 *
 * The caller supplies a validated render policy; live_once returns false.
 *
 * @param {RenderPolicy} policy
 */
function isReplayPolicy(policy) {
  return policy === "replay_animated" || policy === "replay_static";
}

/**
 * Compare the four stable identity fields without changing either plan.
 *
 * first is a plan; second may be unknown. Return false unless second is a record
 * with equal epochKey, authorizationKey, fingerprint and paintKey. This is an
 * identity comparison, not full plan or information-rights validation.
 *
 * @param {Record<string, any>} first
 * @param {unknown} second
 * @returns {second is Record<string, any>}
 */
function samePlanIdentity(first, second) {
  return (
    isRecord(second) &&
    second.epochKey === first.epochKey &&
    second.authorizationKey === first.authorizationKey &&
    second.fingerprint === first.fingerprint &&
    second.paintKey === first.paintKey
  );
}

/**
 * Stand in for the plan of a shown frame that had none, for a +1 step glide.
 *
 * A replay's first frame has no tick leading into it, so the choreographer
 * keeps no plan while it shows (planBuilder returns null and presentFrame
 * clears), and the first +1 from it had nothing to glide from: the bodies
 * snapped while every later +1 glided. plan is the plan of the frame the step
 * lands on, one tick later; each of its bodies' START spots (body.start) is
 * where the first frame showed that body. Return a frozen plan-like record
 * for stepGlideSpecs: plan's audienceKey, epochKey and beats, simulatorStep
 * one less (so the step reads as forward), and one body per plan body with a
 * known START spot: the same presentationKey, publicAgentId and teamId, end
 * = start, alive true when it stood at the tick's start (alive now or fell
 * this tick, and not a return), no Charge, not returned. A body with no
 * START spot is left out, so it snaps; in Agent POV that is every body that
 * came into sight this tick, so nothing glides out of the dark. Pure.
 *
 * @param {Record<string, any>} plan
 * @returns {Readonly<Record<string, any>>}
 */
function stepSourceBefore(plan) {
  const bodies = (Array.isArray(plan.bodies) ? plan.bodies : [])
    .filter((body) => Array.isArray(body?.start))
    .map((body) =>
      Object.freeze({
        presentationKey: body.presentationKey,
        publicAgentId: body.publicAgentId,
        teamId: body.teamId ?? null,
        alive: body.returned !== true && (body.alive === true || body.died === true),
        start: null,
        postCharge: null,
        end: body.start,
        charged: false,
        died: false,
        returned: false,
        glide: false,
        dimShift: false,
      }),
    );
  return Object.freeze({
    audienceKey: plan.audienceKey,
    epochKey: plan.epochKey,
    beats: plan.beats,
    simulatorStep: Number(plan.simulatorStep) - 1,
    bodies: Object.freeze(bodies),
  });
}

/**
 * Report whether a painter animation hides a transient cue for good.
 *
 * spec is one AnimationSpec from the painter's installation (a look's own
 * motion, never a durable-node override). True when its last keyframe sets
 * opacity 0 (at most 0.01), it has no forward fill (fill "backwards" or
 * none; a forward fill already keeps the last frame) and its element is not
 * inside a persistent event group (data-persistent="true"), whose still is
 * the tick's END picture and must return. The looks draw a transient cue's
 * still picture in the DOM for the Settled Still and fade it out on top, so
 * without this rule the still came back when the fade ended and stayed for
 * the rest of the tick (the end hold in the Replay Viewer, the death beats
 * live): every Basic and Ultimate showed twice. An element without closest
 * (a test double) counts as not persistent. Pure.
 *
 * @param {AnimationSpec} spec
 * @returns {boolean}
 */
function endsHidden(spec) {
  const frames = Array.isArray(spec.keyframes) ? spec.keyframes : [];
  const last = /** @type {Record<string, unknown> | undefined} */ (frames.at(-1));
  if (last?.opacity === undefined || !(Number(last.opacity) <= 0.01)) return false;
  const fill = spec.options?.fill;
  if (fill === "forwards" || fill === "both") return false;
  const element = /** @type {any} */ (spec.element);
  if (typeof element?.setAttribute !== "function") return false;
  const persistent =
    typeof element.closest === "function"
      ? element.closest('[data-persistent="true"]')
      : null;
  return persistent === null;
}

/**
 * Attach a no-op rejection handler to a presentation promise.
 *
 * promise is normally animation.finished. All rejections are swallowed, not only
 * cancellation; return undefined. This avoids unhandled cleanup rejections.
 *
 * @param {Promise<unknown>} promise
 */
function ignoreCancellation(promise) {
  promise.catch(() => {});
}

/**
 * Best-effort cancel animation and ignore cancellation errors.
 *
 * Return undefined; this helper does not remove the animation from owner arrays.
 *
 * @param {AnimationHandle} animation
 */
function safeCancel(animation) {
  try {
    animation.cancel();
  } catch {
    // Cancellation is best-effort presentation cleanup.
  }
}

/**
 * Best-effort finish animation and ignore finish errors.
 *
 * Idle or already cancelled browser handles may reject finish; return undefined.
 *
 * @param {AnimationHandle} animation
 */
function safeFinish(animation) {
  try {
    animation.finish();
  } catch {
    // An idle/cancelled animation needs no further presentation work.
  }
}

/**
 * Return normal, reduced or off unchanged, otherwise throw RangeError.
 *
 * value is checked exactly; there is no string coercion or default here.
 *
 * @param {unknown} value
 * @returns {MotionMode}
 */
function normalizeMotionMode(value) {
  if (value === "normal" || value === "reduced" || value === "off") {
    return value;
  }
  throw new RangeError(`unknown motion mode ${String(value)}.`);
}

/**
 * Return a finite supported quarter-step rate from 0.25 through 2.
 *
 * value must be a number in the fixed rate list; otherwise throw RangeError.
 *
 * @param {unknown} value
 */
function normalizePlaybackRate(value) {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    !SUPPORTED_PLAYBACK_RATES.includes(value)
  ) {
    throw new RangeError(
      "playback rate must be one of 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, or 2.00.",
    );
  }
  return value;
}

/**
 * Update animation to rate using its browser rate method when available.
 *
 * rate is already validated. Fall back to assigning playbackRate for compatible
 * adapters. Return undefined; adapter errors propagate.
 *
 * @param {AnimationHandle} animation @param {number} rate
 */
function applyPlaybackRate(animation, rate) {
  if (typeof animation.updatePlaybackRate === "function") {
    animation.updatePlaybackRate(rate);
  } else {
    animation.playbackRate = rate;
  }
}

/**
 * Return value as a number when it is an integer greater than zero.
 *
 * name labels RangeError otherwise. This checks Number.isInteger, not safe-integer
 * precision, and does not accept numeric strings.
 *
 * @param {unknown} value
 * @param {string} name
 */
function positiveInteger(value, name) {
  if (!Number.isInteger(value) || Number(value) <= 0) {
    throw new RangeError(`${name} must be a positive integer.`);
  }
  return Number(value);
}

/**
 * Return value as a number when it is an integer at least zero.
 *
 * name labels RangeError otherwise. No numeric strings or safe-integer guarantee
 * are accepted/implied by this check.
 *
 * @param {unknown} value
 * @param {string} name
 */
function nonNegativeInteger(value, name) {
  if (!Number.isInteger(value) || Number(value) < 0) {
    throw new RangeError(`${name} must be a non-negative integer.`);
  }
  return Number(value);
}
