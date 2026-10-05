/**
 * @file Control presentation time for already-authorized combat frames.
 *
 * What it owns: CombatChoreographer installs painter-owned SVG, starts and stops
 * every Web Animation of a tick, and suppresses repeated live transitions with a
 * bounded optional storage ledger. Replay can animate a tick, show the settled
 * paused still, or step one tick (replay_step): a +1 plays its tick as Play
 * does and then settles into the paused still; a -1 rewinds the tick (fallen
 * champions rise where they fell, every body glides back, then the earlier
 * frame's cues fade in), so nothing shows ahead of a champion. A played replay tick keeps its
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
 * - Glide riders: elements with data-glide-follow (a champion's presentation
 *   key: its selection marks and reach rings, the live Command Ring, Order
 *   Studs and Reach Marks) take that body's glide too, in a played tick and a
 *   rewind, so they never wait at its landing spot; one that a repaint
 *   replaced mid-glide joins the glide at its moment (#reattachRiders).
 * - Landing waits: elements with data-landing-wait (presentation keys: the
 *   what-comes-next marks, the Next Action Thread, All Staged Orders, Next Step
 *   Ghost, the live Order Thread and Step Ghost) fade in once the bodies have
 *   landed when one of those champions glides.
 * - Rewind copies: a -1 puts inert copies (data-rewind-copy) of a rising
 *   champion's corpse, an unreturning champion on its pad and the old Agent
 *   POV veil back in the page to fade them out; they are removed when the
 *   rewind ends or anything replaces it.
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
 *   intent: a +1 plays the new tick as replay_animated and then settles into the
 *   paused still (replay_static); a -1 installs the paused still and rewinds the
 *   tick on top of it (a rise, a 240 ms glide back, then the cues fade in over
 *   80 ms), then settles.
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
 *   ready?: Promise<unknown>,
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
 *   eventNodes?: Map<string, {group: Element, event: Record<string, any>}>,
 *   carry?: {heldRows?: ReadonlyArray<Record<string, any>>},
 * }} ChoreographyInstallation
 *   What the painter returns. root, routeRoot and connectorRoot (the last two
 *   optional) are the painter's groups that carry the plan's
 *   data-authorization-key. nodeCount and persistentNodeCount (checked
 *   against the plan's bounds) leave out held rows carried from the previous
 *   tick; heldNodeCount and heldPersistentNodeCount (optional, information
 *   only) count those held rows. eventNodes (optional) maps each painted
 *   event ID to its group and plan row, and carry.heldRows (optional) lists
 *   the rows held from the tick before; the step cue fade reads both to fade
 *   in this tick's Herald Cards and leave a held card alone.
 * @typedef {{
 *   getItem: (key: string) => string | null,
 *   setItem: (key: string, value: string) => void,
 * }} PresentationStorage
 * @typedef {{
 *   bodies: Map<string, {copy: Element, parent: Element | null, dead: boolean}>,
 *   veil: Array<{copy: Element, parent: Element | null}>,
 * }} StepSnapshot
 *   Inert copies of the arena before a -1 step's frame was drawn
 *   (takeStepSnapshot): each champion's glide wrapper by presentation key,
 *   with the group it sat in and whether it was fallen, and each Agent POV
 *   sight group (g.fog-sight) with the fog layer that holds it.
 * @typedef {{
 *   previousPlan: Record<string, any>,
 *   carry: Map<string, readonly [number, number]>,
 *   snapshot?: StepSnapshot | null,
 * }} StepRequest
 *   A -1 rewind: previousPlan is the plan shown before the step, carry where
 *   running step glides drew each body, snapshot the copies taken before the
 *   new frame was drawn (none for a planless step).
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
   * Create a timer animation that changes nothing on screen.
   *
   * element hosts the clock, duration is browser animation time in milliseconds,
   * and id names it for inspection. The clock has no keyframes, so it never
   * paints: an earlier clock with opacity 1 sat on the painter's event root and,
   * created after the step cue fade, overrode it, so a step showed its landings
   * and numbers before the bodies landed. Return the live Animation with fill
   * both. Browser animation errors propagate; the caller owns its lifetime.
   *
   * @param {Element} element
   * @param {number} duration
   * @param {string} id
   */
  createClock(element, duration, id) {
    return this.create({
      element,
      keyframes: [],
      options: { duration, fill: "both" },
      id,
    });
  }
}

/**
 * Own one authorized combat explanation and its presentation clocks.
 *
 * The controller mutates only its painter installation under the supplied surface,
 * the temporary Web Animations overrides it puts on durable nodes (body glides,
 * beat holds, the step cue fade and the camera ease) and, during a -1 rewind, the
 * inert copies it puts back in the page and removes again. It tracks pause, rate and
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
   * replay_static, and a +1 again as replay_animated. motionMode defaults to normal and playbackRate to 1. Missing
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
     * delay is how long the glide waits (a rise first), in ms.
     * @type {Array<{presentationKey: string, points: ReadonlyArray<readonly [number, number]>, offsets: ReadonlyArray<number>, delay: number, animation: AnimationHandle}>}
     */
    this.stepGlides = [];
    /**
     * Copies of the arena taken just before a -1 step's frame was drawn
     * (captureStepSnapshot), or null: the champions and the visible Agent POV
     * veil as the page showed them. The next presentFrame takes it (and drops
     * it); a rewind puts some copies back in the page to fade them out.
     * @type {StepSnapshot | null}
     */
    this.stepSnapshot = null;
    /**
     * The copies a running rewind put in the page; removed with its animations
     * (#cancelAnimations), so none outlives the step.
     * @type {Element[]}
     */
    this.rewindCopies = [];
    /**
     * The +1 step whose tick is playing now, or null: the frame and switch
     * snapshot it was asked for, so the paused still can be built when the
     * tick ends (#settleStepPlay). Any clear or new install ends it.
     * @type {{frame: unknown, visualFilters: Record<string, boolean> | undefined} | null}
     */
    this.stepPlay = null;
    /**
     * Running body glides of this tick (played or step), each with its
     * champion's presentation key and spec, so a mark that rides that champion
     * and was replaced by a repaint mid-glide joins the glide at the same
     * moment (#reattachRiders).
     * @type {Array<{key: string, spec: AnimationSpec, animation: AnimationHandle}>}
     */
    this.glideSources = [];
    /**
     * The riding marks that already have their glide this tick.
     * @type {WeakSet<Element>}
     */
    this.ridingElements = new WeakSet();
    /**
     * This tick's landing waits (#install, #startStepGlide): which champions
     * glide, when they land and how long a mark takes to fade in, in ms; null
     * when none. A what-comes-next mark a repaint replaced mid-tick gets its
     * wait back from it (#reattachRiders).
     * @type {{gliding: ReadonlySet<string>, at: number, fade: number, epochKey: string} | null}
     */
    this.waitPlan = null;
    /**
     * This rewind's rise (#startStepGlide): the champions whose life changes
     * and how long the rise takes, in ms; null when none. A riding mark of such
     * a champion that a repaint replaced mid-rise fades in with it again
     * (#reattachRiders).
     * @type {{keys: ReadonlySet<string>, riseMs: number, epochKey: string} | null}
     */
    this.risePlan = null;
    /**
     * The marks that already have their landing wait or rise fade this tick.
     * @type {WeakSet<Element>}
     */
    this.fadingElements = new WeakSet();
    /**
     * The animations #reattachRiders added, with their mark, so the ones whose
     * mark a later repaint replaced again are cancelled.
     * @type {Array<{animation: AnimationHandle, element: Element}>}
     */
    this.rejoins = [];
    /** How many animations #reattachRiders has made (unique id suffixes). */
    this.rejoinCount = 0;
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
   * policy (the installed one: a -1 reports replay_static, a +1 replay_animated
   * while its tick plays and replay_static once it settles), logical
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
   * Copy the arena as it is now, just before a -1 step's frame is drawn.
   *
   * surface is the current choreography surface (its battlefield SVG), or
   * null. main.js calls this right before the durable renderer draws the
   * frame a -1 step lands on, because that render changes the champions in
   * place and replaces the veil. The copies (takeStepSnapshot: each
   * champion's glide wrapper with its life state, and the Agent POV sight
   * group, all made inert) wait in this.stepSnapshot; the next
   * presentFrame takes them, and only a rewind for the same audience puts
   * any back in the page (to fade out a corpse that rises, a champion that
   * leaves its pad, or the old veil). So in Agent POV a copy only ever shows
   * what that audience was already shown. Return undefined; reads the DOM.
   *
   * @param {ChoreographySurface | null} surface
   */
  captureStepSnapshot(surface) {
    const root = battlefieldRoot(surface);
    this.stepSnapshot = root ? takeStepSnapshot(root) : null;
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
   * START to END and holds beat-timed nodes until their beat (see #install).
   *
   * A replay_step request (a one-tick move) depends on its direction:
   * - +1 (one tick forward for the same audience, or from a frame with no plan,
   *   motion not off, Step Glide on): the new tick plays exactly as Play plays it
   *   (replay_animated, with no end hold), then settles into the paused still
   *   (#settleStepPlay). Later still requests for the same frame (repaints)
   *   let it play on; Play pressed meanwhile lets it finish as a played tick
   *   (still without the end hold) that does not settle into the still. A
   *   further step jumps the playing tick to its END and starts the next.
   *   presentationControl.stepDirection "backward" (main.js passes the
   *   playback intent's direction) never plays: a -1 from a frame with no
   *   plan shows the still.
   * - -1 (one tick back for the same audience): the paused still is installed
   *   and the tick is rewound on top of it (#startStepGlide): fallen
   *   champions rise first, then bodies glide back, then the cues fade in. It
   *   uses the copies captureStepSnapshot took before the frame was drawn.
   *   A -1 onto a replay's first frame (no plan) clears, then glides each body
   *   back from the shown tick's END to its START spot (#startPlanlessStepGlide).
   * A body unseen in either Agent POV scene or a change of audience or
   * recipient snaps. A replay_step request for the frame already shown is the
   * paused still. Invalid policy, plans, painter bounds or browser animations
   * may throw; a policy-induced identity change throws Error.
   *
   * @param {unknown} frame
   * @param {ChoreographySurface | null} surface
   * @param {{
   *   renderPolicy?: RenderRequest,
   *   visualFilters?: Record<string, boolean>,
   *   restartAnimated?: boolean,
   *   stepDirection?: "forward" | "backward",
   * }} [presentationControl]
   */
  presentFrame(frame, surface, presentationControl = {}) {
    const request = normalizeRenderRequest(
      presentationControl.renderPolicy ?? "live_once",
    );
    const stepRequested = request === "replay_step";
    const stepSnapshot = stepRequested ? this.stepSnapshot : null;
    this.stepSnapshot = null;
    let requestedPolicy = stepRequested ? "replay_static" : request;
    let nextPlan = surface
      ? this.planBuilder(
          frame,
          surface,
          presentationControl.visualFilters,
          requestedPolicy,
        )
      : null;
    // A +1 step plays its tick as Play does, then settles into the paused
    // still (#settleStepPlay); a -1 step rewinds (the step glide).
    const stepPlays =
      stepRequested &&
      presentationControl.stepDirection !== "backward" &&
      nextPlan !== null &&
      surface !== null &&
      this.motionMode !== "off" &&
      nextPlan.beats?.stepGlide !== false &&
      oneTickForward(this.plan, nextPlan);
    if (stepPlays && surface) {
      requestedPolicy = "replay_animated";
      nextPlan = this.planBuilder(
        frame,
        surface,
        presentationControl.visualFilters,
        requestedPolicy,
      );
    }
    const initialPolicy = requestedPolicy;
    if (!nextPlan || !surface) {
      // A -1 onto a replay's first frame (it has no plan): the bodies glide back
      // from the shown tick's END to its START spots, where that frame shows them.
      const backFrom =
        stepRequested && surface && this.plan !== null ? this.plan : null;
      const carry = backFrom !== null && surface ? this.#stepCarry(surface) : new Map();
      this.clear("absent_scene_or_event_batch");
      if (backFrom !== null && surface) {
        this.#startPlanlessStepGlide(backFrom, surface, carry);
        this.#publish();
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
    // While a +1 step plays its tick, the page asks again for the still it
    // will settle into (any repaint); the tick keeps playing. Play pressed
    // meanwhile makes it an ordinary played tick (no settle into the still).
    const stepPlayRunning =
      this.stepPlay !== null &&
      !stepRequested &&
      this.renderPolicy === "replay_animated" &&
      (nextPolicy === "replay_static" || nextPolicy === "replay_animated");
    const samePolicy = this.renderPolicy === nextPolicy || stepPlayRunning;
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
      if (stepPlayRunning && request === "replay_animated") this.stepPlay = null;
      if (this.renderPolicy !== nextPolicy && stepPlayRunning) {
        // Keep the played plan the tick is running on.
        this.surface = surface;
        this.#reattachRiders(surface);
        this.#publish();
        return this.snapshot();
      }
      this.plan = nextPlan;
      this.surface = surface;
      this.#reattachRiders(surface);
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
    // A -1 rewinds from the plan shown before; a +1 plays its tick instead.
    const stepFrom =
      stepRequested && nextPolicy === "replay_static" ? previousPlan : null;
    this.#installForPolicy(nextPlan, surface, nextPolicy, {
      liveSafeRebuild: alreadyConsumed,
      step:
        stepFrom !== null
          ? { previousPlan: stepFrom, carry, snapshot: stepSnapshot }
          : null,
      stepPlay: stepPlays,
    });
    if (stepPlays && this.cleanupClock !== null) {
      this.stepPlay = { frame, visualFilters: presentationControl.visualFilters };
    }
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
    if (this.surface?.viewportKey !== surface.viewportKey) {
      // A rewind's copies keep the old projection: drop them on a resize.
      for (const copy of this.rewindCopies) copy.remove();
      this.rewindCopies = [];
    }
    this.plan = nextPlan;
    this.surface = surface;
    this.#reattachRiders(surface);
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
   * options.step (default null) adds the -1 rewind on top of that still: a
   * StepRequest (the plan shown before, where running step glides drew each
   * body, and the copies taken before the frame was drawn). options.stepPlay (default false) marks a played
   * tick that a +1 step started: it has no 600 ms end hold, because it settles
   * into the paused still as soon as its last beat ends (#settleStepPlay).
   * Mutate controller installation state; return undefined. Installation errors
   * propagate.
   *
   * @param {Record<string, any>} plan
   * @param {ChoreographySurface} surface
   * @param {RenderPolicy} renderPolicy
   * @param {{
   *   liveSafeRebuild?: boolean,
   *   step?: StepRequest | null,
   *   stepPlay?: boolean,
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
      endHold: options.stepPlay !== true,
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
   * the limit the extra holds are dropped (never the tick). The marks that ride
   * a gliding body, the landing waits (from 260 ms over BEAT_CLOCK.landingFade),
   * the movement marks and the Agent POV veil's beats come after them, bounded
   * by the animation limit only, so a busy tick's holds never crowd them out
   * (they keep anything from showing ahead of a champion). Settled or
   * persistent-only output has no overrides and no clocks, except a -1 rewind.
   *
   * Clocks: animated replay lasts its whole tick: the later of the event windows and
   * the beats (plan.beats.explanationEnd) plus the 600 ms end hold, so a tick with
   * Longer Tick lasts 1700 ms (3200 ms with a death); a plan without beats keeps the
   * old total plus 600 ms. options.endHold false (a tick a +1 step plays) leaves
   * the end hold out. Normal motion gates submission until the 450 ms release
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
   *   step?: StepRequest | null,
   *   endHold?: boolean,
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
    const still = settled || persistentOnly || this.motionMode === "off";
    /** @type {ReturnType<typeof playedGlideSpecs>} */
    const played = still
      ? { bodies: [], riders: [], keys: new Set(), sources: [] }
      : playedGlideSpecs(plan, surface, this.motionMode);
    /** @type {ReturnType<typeof playedLookSpecs>} */
    const looks = still
      ? { motion: [], accents: [] }
      : playedLookSpecs(
          plan,
          surface,
          this.motionMode,
          this.playbackRate,
          options.renderPolicy,
        );
    const playedRoot = battlefieldRoot(surface);
    // What keeps a mark from showing early (the riders, the landing waits, the
    // movement marks and the Agent POV veil's beats) is kept apart from the
    // plan's override allowance and bounded by the animation limit only: a
    // busy tick holds many durable nodes, and those holds must never crowd
    // these out. They also depend on switches the allowance does not count.
    const waitSpecs = playedRoot
      ? landingWaitSpecs(
          String(plan.epochKey),
          playedRoot,
          played.keys,
          BEAT_CLOCK.glideEnd,
          BEAT_CLOCK.landingFade,
        )
      : [];
    const extraOverrides = [...played.riders, ...looks.motion];
    const overrides = still
      ? []
      : [
          ...played.bodies,
          ...beatHoldSpecs(plan, surface, this.motionMode),
          ...looks.accents,
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
    const plannedOverrides = boundedOverrides(
      overrides,
      plan,
      MAX_ACTIVE_ANIMATIONS - activeAnimationCount,
    );
    const room = Math.max(
      0,
      MAX_ACTIVE_ANIMATIONS - activeAnimationCount - plannedOverrides.length,
    );
    const keptWaits = waitSpecs.slice(0, room);
    // The waits are created before the beat holds, so a hold on the same mark
    // (the hotbar's set-down on a return tick) comes later and wins.
    const keptOverrides = [
      ...keptWaits,
      ...plannedOverrides,
      ...extraOverrides.slice(0, room - keptWaits.length),
    ];
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
        ? Math.max(eventEnd, beatEnd, overrideEnd) +
          (options.endHold === false ? 0 : REPLAY_TERMINAL_HOLD_MS)
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
    this.glideSources = played.sources
      .map(({ key, spec }) => ({
        key,
        spec,
        animation:
          createdAnimations[animationSpecs.length + keptOverrides.indexOf(spec)],
      }))
      .filter(({ spec, animation }) => keptOverrides.includes(spec) && animation);
    this.ridingElements = new WeakSet(
      played.riders
        .filter((spec) => keptOverrides.includes(spec))
        .map((spec) => spec.element),
    );
    this.waitPlan =
      played.keys.size > 0
        ? {
            gliding: played.keys,
            at: BEAT_CLOCK.glideEnd,
            fade: BEAT_CLOCK.landingFade,
            epochKey: String(plan.epochKey),
          }
        : null;
    this.fadingElements = new WeakSet(keptWaits.map((spec) => spec.element));
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
          if (this.stepPlay !== null) {
            try {
              this.#settleStepPlay();
            } catch {
              // The still could not be built: clear (it publishes and frees
              // every settlement waiter) rather than leave the tick hanging.
              this.clear("step_settle_failed");
              return;
            }
          }
          this.#publish();
        }),
      );
    }
  }

  /**
   * Rewind one tick (a -1 step) on top of the already settled paused still.
   *
   * plan is the new (earlier) frame's plan, surface its surface, installation
   * the settled painter installation and step the StepRequest. It runs only in
   * normal motion with Step Glide on (plan.beats.stepGlide, default on), for
   * the same audienceKey and exactly one tick back (rewindAllowed); a +1 step
   * plays its tick instead (presentFrame), and anything else adds nothing.
   *
   * The rewind is the tick played backwards, in three parts:
   * - Rise (only when a champion's life changes, BEAT_CLOCK.stepRise, 200 ms):
   *   a champion that fell in the tick being undone rises where it fell (the
   *   copy of its corpse fades out while its living body fades in there), and
   *   a champion that came back to its pad in that tick returns to its corpse
   *   (the copy of it on the pad fades out while the corpse fades in).
   * - Glide (240 ms): every body present in both plans and alive in the new
   *   one glides from where the previous frame showed it (a running step
   *   glide's drawn spot when carry has one, its corpse for a rising
   *   champion, otherwise its previous END spot) back to its new END spot; a
   *   Charge rush runs back along its path in the first 140 ms. Its selection
   *   marks and reach rings ride it (data-glide-follow). In Agent POV the copy
   *   of the old veil fades out while the new one fades in over the rise and
   *   the glide, so the light moves back with the champions.
   * - Landing (80 ms): the tick's cues (stepCueFadeSpecs), the movement marks
   *   and the what-comes-next marks of gliding champions (data-landing-wait)
   *   fade in.
   * Overrides are bounded by the animation limit, bodies first
   * (boundedStepOverrides). A one-step clock (it paints nothing) then cancels
   * them, removes the copies and marks the still paused again. Every spec
   * uses fill "backwards", so cancelling shows the new frame's END. Creation
   * errors cancel what was created and propagate. Mutates this.animations,
   * this.cleanupClock, this.stepGlides, this.rewindCopies, this.paused and the
   * page (copies inserted).
   *
   * @param {Record<string, any>} plan
   * @param {ChoreographySurface} surface
   * @param {ChoreographyInstallation} installation
   * @param {StepRequest} step
   */
  #startStepGlide(plan, surface, installation, step) {
    if (!rewindAllowed(plan, step.previousPlan, this.motionMode)) return;
    const root = battlefieldRoot(surface);
    const changes = rewindLifeChanges(step.previousPlan, plan);
    const riseMs = changes.size > 0 ? BEAT_CLOCK.stepRise : 0;
    const landed = riseMs + BEAT_CLOCK.stepGlide;
    const fadeMs = BEAT_CLOCK.stepCueFadeEnd - BEAT_CLOCK.stepGlide;
    const glides = stepGlideSpecs(plan, surface, this.motionMode, step, riseMs) ?? [];
    const copies =
      root && step.snapshot ? placeRewindCopies(step.snapshot, changes) : null;
    this.rewindCopies = copies
      ? [...copies.veil, ...copies.bodies.map(({ copy }) => copy)]
      : [];
    const life = root
      ? rewindLifeSpecs(
          String(plan.epochKey),
          root,
          surface,
          changes,
          copies?.bodies ?? [],
          riseMs,
        )
      : [];
    const fades = stepCueFadeSpecs(plan, installation, landed);
    const waits = root
      ? landingWaitSpecs(
          String(plan.epochKey),
          root,
          new Set(glides.map(({ presentationKey }) => presentationKey)),
          landed,
          fadeMs,
        )
      : [];
    const lookSpecs = root
      ? [
          ...sightAnimationSpecs(root, plan, {
            motionMode: this.motionMode,
            renderPolicy: "replay_step",
            outgoing: copies?.veil ?? [],
            duration: landed,
          }),
          ...movementAnimationSpecs(root, {
            epochKey: String(plan.epochKey),
            mode: "step",
            motionMode: this.motionMode,
            beats: plan.beats,
            stepMs: landed,
            stepCueFadeMs: fadeMs,
          }),
        ]
      : [];
    // Bodies first, then the riders and the rise fades (created after the
    // glides, so a rise's opacity wins over a Dim Shift's held one), then what
    // keeps a mark from showing early: the cue fades and the waits; the look
    // marks last.
    const specs = boundedStepOverrides([
      ...glides.filter(({ follower }) => !follower).map(({ spec }) => spec),
      ...glides.filter(({ follower }) => follower).map(({ spec }) => spec),
      ...life,
      ...fades,
      ...waits,
      ...lookSpecs,
    ]);
    if (specs.length === 0) {
      for (const copy of this.rewindCopies) copy.remove();
      this.rewindCopies = [];
      return;
    }
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
        riseMs + BEAT_CLOCK.stepCueFadeEnd,
        `mbg:${plan.epochKey}:step`,
      );
      ignoreCancellation(clock.finished);
      this.cleanupClock = clock;
    } catch (error) {
      for (const animation of created) safeCancel(animation);
      for (const copy of this.rewindCopies) copy.remove();
      this.rewindCopies = [];
      throw error;
    }
    this.animations = created;
    this.#trackStepGlides(glides, specs, created);
    const gliding = new Set(glides.map(({ presentationKey }) => presentationKey));
    this.waitPlan =
      gliding.size > 0
        ? { gliding, at: landed, fade: fadeMs, epochKey: String(plan.epochKey) }
        : null;
    this.risePlan =
      riseMs > 0
        ? { keys: new Set(changes.keys()), riseMs, epochKey: String(plan.epochKey) }
        : null;
    this.fadingElements = new WeakSet(
      [...waits, ...life]
        .filter((spec) => specs.includes(spec))
        .map((spec) => spec.element),
    );
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
   * path; a body that came back to its pad snaps, and one that fell in the
   * first tick glides back from its corpse (no rise: the first frame has no
   * plan to rise from). Selection marks and reach
   * rings ride their champion's glide and its what-comes-next marks wait for
   * the landing (240-320 ms), as on every step. No painter installation exists
   * for the landing frame, so there are no cue fades, movement marks or sight
   * cross-fade. A one-step clock on the battlefield root (240 ms, or 320 ms when
   * a mark waits; it paints nothing) then cancels the glides and marks the
   * still paused. Reduced motion, motion off, Step Glide
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
    const waits = landingWaitSpecs(
      `${previousPlan.epochKey}:back`,
      root,
      new Set(glides.map(({ presentationKey }) => presentationKey)),
      BEAT_CLOCK.stepGlide,
      BEAT_CLOCK.stepCueFadeEnd - BEAT_CLOCK.stepGlide,
    );
    const specs = boundedStepOverrides([
      ...glides.filter(({ follower }) => !follower).map(({ spec }) => spec),
      ...waits,
      ...glides.filter(({ follower }) => follower).map(({ spec }) => spec),
    ]);
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
        waits.length > 0 ? BEAT_CLOCK.stepCueFadeEnd : BEAT_CLOCK.stepGlide,
        `mbg:${previousPlan.epochKey}:step-back`,
      );
      ignoreCancellation(clock.finished);
    } catch (error) {
      for (const animation of created) safeCancel(animation);
      throw error;
    }
    this.animations = created;
    this.cleanupClock = clock;
    this.#trackStepGlides(glides, specs, created);
    const gliding = new Set(glides.map(({ presentationKey }) => presentationKey));
    this.waitPlan =
      gliding.size > 0
        ? {
            gliding,
            at: BEAT_CLOCK.stepGlide,
            fade: BEAT_CLOCK.stepCueFadeEnd - BEAT_CLOCK.stepGlide,
            epochKey: `${previousPlan.epochKey}:back`,
          }
        : null;
    this.fadingElements = new WeakSet(
      waits.filter((spec) => specs.includes(spec)).map((spec) => spec.element),
    );
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
   * spot for glides whose animation reports a finite currentTime before the
   * glide ends: a glide still waiting for a rise (time before its delay) gives
   * its first spot. Pure read of animation state; called just before those
   * glides are cancelled.
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
      const progress = (time - glide.delay) / BEAT_CLOCK.stepGlide;
      if (progress >= 1 || time <= 0) continue;
      carry.set(
        glide.presentationKey,
        pointAlongGlide(glide.points, glide.offsets, Math.max(0, progress)),
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
    this.stepPlay = null;
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
    this.glideSources = [];
    this.ridingElements = new WeakSet();
    this.waitPlan = null;
    this.risePlan = null;
    this.fadingElements = new WeakSet();
    this.rejoins = [];
    for (const copy of this.rewindCopies) copy.remove();
    this.rewindCopies = [];
    this.gateClock = null;
    this.cleanupClock = null;
    this.cameraLean = null;
  }

  /**
   * Turn a +1 step's finished tick into the paused still.
   *
   * Called when the played tick a +1 step started has ended (its last beat,
   * no end hold). The tick's own frame is planned again as replay_static and
   * installed as the settled paused still, exactly as a seek to that frame
   * shows it: every cue's still picture, a Herald Card at once. So a +1 plays
   * the tick as Play does and then pauses on the same picture the step showed
   * before. Without a surface it does nothing more; when the plan builder
   * gives no plan it clears the installation and sets plan to null (the
   * previous renderPolicy stays). Mutates the installation, plan,
   * renderPolicy and paused; errors from the plan builder or painter
   * propagate (the caller clears then).
   */
  #settleStepPlay() {
    const step = this.stepPlay;
    const surface = this.surface;
    this.stepPlay = null;
    if (step === null || surface === null) return;
    const plan = this.planBuilder(
      step.frame,
      surface,
      step.visualFilters,
      "replay_static",
    );
    this.#clearOwned("step_played");
    if (!plan) {
      this.plan = null;
      return;
    }
    this.plan = plan;
    this.surface = surface;
    this.#installForPolicy(plan, surface, "replay_static");
  }

  /**
   * Remember a step's running glides for the next step and for repaints.
   *
   * glides are stepGlideSpecs' entries, specs the specs that were started (in
   * order) and created their animations, index for index. Body entries (not
   * follower) become this.stepGlides (where a quick next step starts from,
   * #stepCarry) and this.glideSources (#reattachRiders); follower entries'
   * elements join ridingElements. Entries whose spec was not started are
   * skipped. Mutates those three fields only.
   *
   * @param {ReadonlyArray<{spec: AnimationSpec, presentationKey: string, points: ReadonlyArray<readonly [number, number]>, offsets: ReadonlyArray<number>, delay?: number, follower?: boolean}>} glides
   * @param {ReadonlyArray<AnimationSpec>} specs
   * @param {ReadonlyArray<AnimationHandle>} created
   */
  #trackStepGlides(glides, specs, created) {
    /** @type {typeof this.stepGlides} */
    const bodies = [];
    /** @type {typeof this.glideSources} */
    const sources = [];
    /** @type {Element[]} */
    const riding = [];
    for (const { spec, presentationKey, points, offsets, delay, follower } of glides) {
      const animation = created[specs.indexOf(spec)];
      if (!animation) continue;
      if (follower) {
        riding.push(spec.element);
        continue;
      }
      bodies.push({ presentationKey, points, offsets, delay: delay ?? 0, animation });
      sources.push({ key: presentationKey, spec, animation });
    }
    this.stepGlides = bodies;
    this.glideSources = sources;
    this.ridingElements = new WeakSet(riding);
  }

  /**
   * Give the marks a repaint replaced mid-tick their timing back.
   *
   * surface is the current surface. A durable repaint during a running tick (a
   * new server revision, a deck click, a camera move or a resize) keeps nodes
   * that did not change, but replaces changed ones and always replaces a node
   * with a hover card, such as a reach ring; the new node has no animation and
   * would show at once. So, on the new nodes only (not yet in ridingElements or
   * fadingElements):
   * - a mark with a gliding champion's data-glide-follow gets that body's
   *   glide (glideSources), at the body glide's current time;
   * - a mark with data-landing-wait naming a gliding champion gets its wait
   *   (waitPlan), at the tick clock's current time;
   * - a rising champion's riding mark (risePlan, a -1 rewind) fades in with it,
   *   at the tick clock's current time.
   * A glide, wait or rise that has already ended adds nothing. Each added
   * animation gets the playback rate, a unique id (suffix ":rejoin:<n>"), is
   * paused when the tick is, and is synced again once it and its clock are
   * ready (a rate change can still be pending). Added animations whose mark a
   * later repaint replaced again are cancelled. Mutates this.animations,
   * this.rejoins and the two sets; animation errors propagate.
   *
   * @param {ChoreographySurface} surface
   */
  #reattachRiders(surface) {
    const root = battlefieldRoot(surface);
    if (!root) return;
    const generation = this.generation;
    // Drop earlier rejoins whose mark has left the page.
    const stale = this.rejoins.filter(({ element }) => element.isConnected === false);
    if (stale.length > 0) {
      for (const { animation } of stale) safeCancel(animation);
      const gone = new Set(stale.map(({ animation }) => animation));
      this.animations = this.animations.filter((animation) => !gone.has(animation));
      this.rejoins = this.rejoins.filter(({ animation }) => !gone.has(animation));
    }
    /**
     * Start one added animation at clock's current time.
     * @param {Element} element @param {Keyframe[] | PropertyIndexedKeyframes} keyframes
     * @param {KeyframeAnimationOptions} options @param {string} id
     * @param {AnimationHandle} clock
     */
    const join = (element, keyframes, options, id, clock) => {
      const animation = this.animationFactory.create({
        element,
        keyframes,
        options,
        id,
      });
      ignoreCancellation(animation.finished);
      applyPlaybackRate(animation, this.playbackRate);
      animation.currentTime = clock.currentTime;
      if (this.paused) animation.pause();
      if (animation.ready && clock.ready) {
        ignoreCancellation(
          Promise.all([animation.ready, clock.ready]).then(() => {
            if (generation === this.generation)
              animation.currentTime = clock.currentTime;
          }),
        );
      }
      this.animations.push(animation);
      this.rejoins.push({ animation, element });
      this.rejoinCount += 1;
    };
    const riders = glideRiders(root);
    /** @type {Element[]} */
    const riding = [];
    for (const source of this.glideSources) {
      const time = source.animation.currentTime;
      const end =
        Number(source.spec.options.delay ?? 0) +
        Number(source.spec.options.duration ?? 0);
      if (typeof time !== "number" || !Number.isFinite(time) || time >= end) continue;
      for (const element of riders.get(source.key) ?? []) {
        if (this.ridingElements.has(element)) continue;
        join(
          element,
          source.spec.keyframes,
          source.spec.options,
          `${source.spec.id}:rejoin:${this.rejoinCount}`,
          source.animation,
        );
        riding.push(element);
      }
    }
    for (const element of riding) this.ridingElements.add(element);
    const clock = this.cleanupClock;
    const now = clock?.currentTime;
    if (!clock || typeof now !== "number" || !Number.isFinite(now)) return;
    /** @param {Element} element */
    const endOpacity = (element) => {
      const value = Number(element.getAttribute("opacity"));
      return element.hasAttribute("opacity") && Number.isFinite(value) ? value : 1;
    };
    const wait = this.waitPlan;
    if (wait !== null && now < wait.at + wait.fade) {
      for (const element of root.querySelectorAll("[data-landing-wait]")) {
        if (this.fadingElements.has(element)) continue;
        const keys = String(element.getAttribute("data-landing-wait") ?? "").split(" ");
        if (!keys.some((key) => wait.gliding.has(key))) continue;
        join(
          element,
          [{ opacity: 0 }, { opacity: endOpacity(element) }],
          { delay: wait.at, duration: wait.fade, fill: "backwards" },
          `mbg:${wait.epochKey}:landing-wait:rejoin:${this.rejoinCount}`,
          clock,
        );
        this.fadingElements.add(element);
      }
    }
    const rise = this.risePlan;
    if (rise !== null && now < rise.riseMs) {
      for (const key of rise.keys) {
        for (const element of riders.get(key) ?? []) {
          if (this.fadingElements.has(element)) continue;
          join(
            element,
            [{ opacity: 0 }, { opacity: endOpacity(element) }],
            { delay: 0, duration: rise.riseMs, fill: "backwards" },
            `mbg:${rise.epochKey}:rise:rejoin:${this.rejoinCount}`,
            clock,
          );
          this.fadingElements.add(element);
        }
      }
    }
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
 * Build the body glides of a played tick, and the glides of the marks that ride
 * them.
 *
 * plan carries beats (glide: Body Glide on) and bodies (planBodies records);
 * surface supplies projection and the durable DOM; motionMode is the controller's.
 * Only normal motion with Body Glide on glides. Each body whose record says glide
 * moves START to END over 0-260 ms; a body that charged reaches its post-Charge spot
 * at 110 ms first. A body whose path cuts a stone (body.dimShift, from the plan)
 * takes the Movement look's Dim Shift (move plus fade) instead of a plain glide.
 * Bodies with no wrapper in the DOM are skipped. Every element under the
 * battlefield whose data-glide-follow is a gliding body's presentation key (its
 * selection marks and reach rings) rides the same glide (id suffix
 * ":follow:<n>"), so it never waits at the landing spot. Return
 * {bodies, riders, keys, sources}: the body specs (ids
 * mbg:<epoch>:glide:<public ID>), the rider specs, the presentation keys of
 * the bodies that glide (for landingWaitSpecs) and each body spec with its key
 * (for #reattachRiders). Reads the DOM only.
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographySurface} surface
 * @param {MotionMode} motionMode
 * @returns {{bodies: AnimationSpec[], riders: AnimationSpec[], keys: Set<string>, sources: Array<{key: string, spec: AnimationSpec}>}}
 */
function playedGlideSpecs(plan, surface, motionMode) {
  /** @type {AnimationSpec[]} */
  const bodies = [];
  /** @type {AnimationSpec[]} */
  const riders = [];
  /** @type {Set<string>} */
  const keys = new Set();
  /** @type {Array<{key: string, spec: AnimationSpec}>} */
  const sources = [];
  const empty = { bodies, riders, keys, sources };
  if (motionMode !== "normal" || plan.beats?.glide !== true) return empty;
  const root = battlefieldRoot(surface);
  if (!root) return empty;
  const wrapperFor = bodyWrapperLookup(surface, root);
  const riderLookup = glideRiders(root);
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
    keys.add(body.presentationKey);
    for (const [index, target] of [
      element,
      ...(riderLookup.get(body.presentationKey) ?? []),
    ].entries()) {
      const targetId = index > 0 ? `${id}:follow:${index - 1}` : id;
      // A path through stone dims out and back in instead of sliding over it.
      const specs =
        body.dimShift === true
          ? dimShiftSpecs(
              target,
              points,
              offsets,
              surface.worldToScreen,
              BEAT_CLOCK.glideEnd,
              targetId,
            )
          : [
              glideSpec(
                target,
                points,
                offsets,
                surface,
                BEAT_CLOCK.glideEnd,
                targetId,
              ),
            ];
      for (const spec of specs) {
        if (!spec) continue;
        (index > 0 ? riders : bodies).push(spec);
        if (index === 0) sources.push({ key: body.presentationKey, spec });
      }
    }
  }
  return empty;
}

/**
 * Map each champion's presentation key to the marks that ride its glide.
 *
 * root is the battlefield SVG. Every element with a nonempty data-glide-follow
 * (looks/champion-frame.js: the selection moonlight, brackets and halo, the
 * reach rings) is listed under that key, in document order. Return a new Map.
 * Reads the DOM only.
 *
 * @param {Element} root
 * @returns {Map<string, Element[]>}
 */
function glideRiders(root) {
  /** @type {Map<string, Element[]>} */
  const riders = new Map();
  for (const element of root.querySelectorAll("[data-glide-follow]")) {
    const key = element.getAttribute("data-glide-follow");
    if (!key) continue;
    riders.set(key, [...(riders.get(key) ?? []), element]);
  }
  return riders;
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
 * Build the look-owned overrides of a played tick.
 *
 * plan, surface and renderPolicy describe the tick; motionMode and playbackRate are
 * the controller's. Return {motion, accents}: motion holds the movement marks
 * (Start Spots, afterimages, the Focus Trail's dots) and the Agent POV sight
 * veil's beats, which keep a mark or the light from showing ahead of a
 * champion, so #install bounds them by the animation limit only; accents holds
 * the hit shivers and Death Dimming, which share the plan's override allowance.
 * Both are new arrays ([] without a battlefield or with motion off). Reads the
 * DOM only.
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographySurface} surface
 * @param {MotionMode} motionMode
 * @param {number} playbackRate
 * @param {RenderPolicy} renderPolicy
 * @returns {{motion: AnimationSpec[], accents: AnimationSpec[]}}
 */
function playedLookSpecs(plan, surface, motionMode, playbackRate, renderPolicy) {
  const root = battlefieldRoot(surface);
  if (!root || motionMode === "off") return { motion: [], accents: [] };
  return {
    motion: [
      ...movementAnimationSpecs(root, {
        epochKey: String(plan.epochKey),
        mode: "played",
        motionMode,
        beats: plan.beats,
      }),
      ...sightAnimationSpecs(root, plan, { motionMode, renderPolicy }),
    ],
    accents: [
      ...shiverSpecs(plan, surface, motionMode, playbackRate),
      ...deathDimSpecs(plan, surface, motionMode),
    ],
  };
}

/**
 * Build the body glides of a -1 rewind (replay_step).
 *
 * plan is the new (earlier) frame's plan; surface its surface; motionMode the
 * controller's; step the StepRequest; riseMs (default 0) how long the glides
 * wait while fallen champions rise first. Return null when the step does not
 * rewind (rewindAllowed: motion not normal, Step Glide off, another audience,
 * or not exactly one tick back). Otherwise return a new array (possibly empty)
 * of {spec, presentationKey, points, offsets, delay, follower?}: one 240 ms
 * glide, starting at riseMs, per body that is in both plans, alive in the new
 * (earlier) one and did not come back to its pad in the tick being undone
 * (earlier.returned). A fallen body in the new frame never glides, and one
 * that came back to its pad does not either (rewindLifeSpecs cross-fades it
 * to its corpse); a body that fell in the undone tick glides from its corpse
 * after the rise. A body glides from carry's drawn spot (when within 3 world units of
 * its previous END spot), else its previous END spot (a rising champion's
 * corpse), to its new END spot; the left tick's Charge is reversed with its
 * post-Charge spot at 140 ms; a body whose left path cut a stone
 * (earlier.dimShift) gets the Movement look's Dim Shift pair instead (two
 * entries with the same key). The glide holds its first spot until it starts
 * (fill "backwards"). Every element under the battlefield whose
 * data-glide-follow is that body's presentation key rides the same glide
 * (id suffix ":follow:<n>", follower: true). In Agent POV the plans hold only
 * seen bodies, so a body unseen in either frame never glides. Reads the DOM
 * only.
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographySurface} surface
 * @param {MotionMode} motionMode
 * @param {StepRequest} step
 * @param {number} [riseMs]
 * @returns {Array<{spec: AnimationSpec, presentationKey: string, points: ReadonlyArray<readonly [number, number]>, offsets: ReadonlyArray<number>, delay: number, follower?: boolean}> | null}
 */
function stepGlideSpecs(plan, surface, motionMode, step, riseMs = 0) {
  const previous = step.previousPlan;
  if (!rewindAllowed(plan, previous, motionMode)) return null;
  const root = battlefieldRoot(surface);
  if (!root) return [];
  const wrapperFor = bodyWrapperLookup(surface, root);
  const riders = glideRiders(root);
  const before = new Map(
    (Array.isArray(previous.bodies) ? previous.bodies : []).map((body) => [
      body.presentationKey,
      body,
    ]),
  );
  /** @type {Array<{spec: AnimationSpec, presentationKey: string, points: ReadonlyArray<readonly [number, number]>, offsets: ReadonlyArray<number>, delay: number, follower?: boolean}>} */
  const glides = [];
  for (const body of Array.isArray(plan.bodies) ? plan.bodies : []) {
    const earlier = before.get(body.presentationKey);
    if (!earlier || !body.alive || earlier.returned) continue;
    const carried = earlier.alive ? step.carry.get(body.presentationKey) : undefined;
    const from =
      carried &&
      Math.hypot(carried[0] - earlier.end[0], carried[1] - earlier.end[1]) <=
        STEP_CARRY_LIMIT
        ? carried
        : earlier.end;
    const charge = earlier.charged === true && earlier.postCharge;
    /** @type {ReadonlyArray<readonly [number, number]>} */
    const points = charge ? [from, earlier.postCharge, body.end] : [from, body.end];
    /** @type {ReadonlyArray<number>} */
    const offsets = charge
      ? [0, BEAT_CLOCK.stepChargeRushBackward / BEAT_CLOCK.stepGlide, 1]
      : [0, 1];
    if (points.every((spot) => sameSpot(spot, body.end))) continue;
    const element = wrapperFor(body.presentationKey);
    if (!element) continue;
    const id = `mbg:${plan.epochKey}:step-glide:${body.publicAgentId}`;
    for (const [index, target] of [
      element,
      ...(riders.get(body.presentationKey) ?? []),
    ].entries()) {
      const follower = index > 0;
      const targetId = follower ? `${id}:follow:${index - 1}` : id;
      const specs =
        earlier.dimShift === true
          ? dimShiftSpecs(
              target,
              points,
              offsets,
              surface.worldToScreen,
              BEAT_CLOCK.stepGlide,
              targetId,
            )
          : [
              glideSpec(
                target,
                points,
                offsets,
                surface,
                BEAT_CLOCK.stepGlide,
                targetId,
              ),
            ];
      for (const spec of specs) {
        if (!spec) continue;
        glides.push({
          spec: delayedSpec(spec, riseMs),
          presentationKey: body.presentationKey,
          points,
          offsets,
          delay: riseMs,
          ...(follower ? { follower: true } : {}),
        });
      }
    }
  }
  return glides;
}

/**
 * Return whether a step from previous to plan is a -1 rewind the choreographer
 * animates.
 *
 * plan is the new (earlier) frame's plan and previous the plan shown before
 * the step (or null); motionMode is the controller's. True only in normal
 * motion, with Step Glide on (plan.beats.stepGlide not false), when both plans
 * have the same string audienceKey and plan.simulatorStep is exactly one less
 * than previous.simulatorStep. Pure.
 *
 * @param {Record<string, any>} plan
 * @param {Record<string, any> | null | undefined} previous
 * @param {MotionMode} motionMode
 */
function rewindAllowed(plan, previous, motionMode) {
  return (
    motionMode === "normal" &&
    plan.beats?.stepGlide !== false &&
    typeof plan.audienceKey === "string" &&
    previous?.audienceKey === plan.audienceKey &&
    Number(plan.simulatorStep) - Number(previous?.simulatorStep) === -1
  );
}

/**
 * Return the champions whose life changes when a tick is rewound.
 *
 * previous is the plan shown before a -1 step, plan the earlier frame's plan.
 * A champion in both plans that is fallen in previous and alive in plan
 * "rises" (it fell in the tick being undone); one alive in previous and fallen
 * in plan "unreturns" (it came back to its pad in that tick). Return a new Map
 * from presentation key to "rise" or "unreturn". Pure.
 *
 * @param {Record<string, any>} previous
 * @param {Record<string, any>} plan
 * @returns {Map<string, "rise" | "unreturn">}
 */
function rewindLifeChanges(previous, plan) {
  const before = new Map(
    (Array.isArray(previous?.bodies) ? previous.bodies : []).map((body) => [
      body.presentationKey,
      body,
    ]),
  );
  /** @type {Map<string, "rise" | "unreturn">} */
  const changes = new Map();
  for (const body of Array.isArray(plan.bodies) ? plan.bodies : []) {
    const earlier = before.get(body.presentationKey);
    if (!earlier || earlier.alive === body.alive) continue;
    changes.set(body.presentationKey, body.alive ? "rise" : "unreturn");
  }
  return changes;
}

/** Attributes a rewind copy drops, so no code or selector mistakes it for the real node. */
const REWIND_COPY_DROPPED_ATTRIBUTES = Object.freeze([
  "id",
  "tabindex",
  "role",
  "aria-label",
  "aria-description",
  "aria-describedby",
  "data-tooltip-owner",
  "data-glide-key",
  "data-glide-follow",
  "data-landing-wait",
  "data-presentation-key",
  "data-beat-hold",
  "data-beat-key",
  "data-beat-side",
  "data-beat-from",
  "data-beat-fade",
  "data-beat-pulse",
  "data-beat-shiver",
  "data-beat-shiver-at",
  "data-beat-delay",
  "data-fog-state",
  "data-fog-arrive-kind",
  "data-fog-arrive-key",
  "data-fog-leave-kind",
  "data-fog-leave-key",
]);

/**
 * Make an inert copy of an arena node for a rewind.
 *
 * element is a champion's glide wrapper or a veil path. Return a deep clone
 * without ids, focus, hover cards, roles, labels, glide, beat or veil-state
 * attributes (REWIND_COPY_DROPPED_ATTRIBUTES), marked aria-hidden,
 * data-rewind-copy and pointer-events none; a wrapper's agent-glide class
 * becomes rewind-copy. Classes and other data attributes stay, so it looks
 * the same. The clone is detached; its look is the END the page drew.
 *
 * @param {Element} element
 * @returns {Element}
 */
function rewindCopy(element) {
  const copy = /** @type {Element} */ (element.cloneNode(true));
  for (const node of [copy, ...copy.querySelectorAll("*")]) {
    for (const name of REWIND_COPY_DROPPED_ATTRIBUTES) node.removeAttribute(name);
  }
  copy.setAttribute("aria-hidden", "true");
  copy.setAttribute("data-rewind-copy", "");
  copy.setAttribute("pointer-events", "none");
  if (copy.classList?.contains("agent-glide")) {
    copy.classList.replace("agent-glide", "rewind-copy");
  }
  return copy;
}

/**
 * Copy the champions and the visible veil before a -1 step's frame is drawn.
 *
 * root is the battlefield SVG. Each `g.agent-glide` wrapper with a
 * data-glide-key (its last JSON entry is the presentation key) is copied
 * (rewindCopy) with its parent group and whether that group is the fallen
 * one (data-body-state "dead"); each Agent POV sight group (`g.fog-sight`:
 * the veil, lift and edge paths, only the END one visible in a still) is
 * copied with the fog layer that holds it, because the durable render
 * replaces the whole group while the layer stays. Return a StepSnapshot.
 * Reads the DOM only.
 *
 * @param {Element} root
 * @returns {StepSnapshot}
 */
function takeStepSnapshot(root) {
  /** @type {StepSnapshot["bodies"]} */
  const bodies = new Map();
  for (const wrapper of root.querySelectorAll("g.agent-glide[data-glide-key]")) {
    let key = null;
    try {
      const parsed = JSON.parse(String(wrapper.getAttribute("data-glide-key")));
      key = Array.isArray(parsed) ? parsed.at(-1) : null;
    } catch {
      key = null;
    }
    if (typeof key !== "string" || key === "") continue;
    const parent = wrapper.parentElement;
    bodies.set(key, {
      copy: rewindCopy(wrapper),
      parent,
      dead: parent?.getAttribute("data-body-state") === "dead",
    });
  }
  /** @type {StepSnapshot["veil"]} */
  const veil = [];
  for (const group of root.querySelectorAll("g.fog-sight")) {
    // The redraw replaces the whole sight group, so the copy is anchored to
    // the layer that holds it, which stays.
    const layer = group.parentElement;
    if (!layer) continue;
    veil.push({ copy: rewindCopy(group), parent: layer });
  }
  return { bodies, veil };
}

/**
 * Put the copies a rewind fades out back into the page.
 *
 * snapshot is the StepSnapshot taken before the new frame was drawn; changes
 * the rewindLifeChanges map. Each sight group copy goes back into its fog
 * layer (which the render keeps) just before the new frame's sight group, so
 * it paints under it; a corpse copy (a rising
 * champion, fallen in the snapshot) and a pad copy (an unreturning champion,
 * alive in the snapshot) go back into their group. Every copy gets opacity 0
 * as its END, so it is gone once its fade ends even before it is removed.
 * Return {veil, bodies} (bodies: {key, copy}); copies whose group left the
 * page are skipped. Mutates the page.
 *
 * @param {StepSnapshot} snapshot
 * @param {Map<string, "rise" | "unreturn">} changes
 * @returns {{veil: Element[], bodies: Array<{key: string, copy: Element}>}}
 */
function placeRewindCopies(snapshot, changes) {
  /** @type {Element[]} */
  const veil = [];
  for (const { copy, parent } of snapshot.veil) {
    if (!parent?.isConnected) continue;
    copy.setAttribute("opacity", "0");
    parent.insertBefore(copy, parent.querySelector(":scope > g.fog-sight"));
    veil.push(copy);
  }
  /** @type {Array<{key: string, copy: Element}>} */
  const bodies = [];
  for (const [key, change] of changes) {
    const shot = snapshot.bodies.get(key);
    if (!shot?.parent?.isConnected || shot.dead !== (change === "rise")) continue;
    shot.copy.setAttribute("opacity", "0");
    shot.parent.append(shot.copy);
    bodies.push({ key, copy: shot.copy });
  }
  return { veil, bodies };
}

/**
 * Build the rise part of a -1 rewind: champions whose life changes.
 *
 * epochKey names the ids (mbg:<epochKey>:rise:<n>); root is the battlefield
 * SVG; surface finds body wrappers; changes is rewindLifeChanges; copies the
 * placed body copies; riseMs the rise length (0 adds nothing). Each copy fades
 * from 1 to 0 over riseMs. Each changed champion's own wrapper in the new
 * frame (its living body for a rise, its corpse for an unreturn) and every
 * mark riding it fade from 0 to their END opacity (opacity attribute, default
 * 1) over riseMs, so a fallen champion rises where it fell before gliding
 * back, and a champion leaves its pad for its corpse. Fill "backwards". Return
 * a new array of specs. Reads the DOM only.
 *
 * @param {string} epochKey
 * @param {Element} root
 * @param {ChoreographySurface} surface
 * @param {Map<string, "rise" | "unreturn">} changes
 * @param {ReadonlyArray<{key: string, copy: Element}>} copies
 * @param {number} riseMs
 * @returns {AnimationSpec[]}
 */
function rewindLifeSpecs(epochKey, root, surface, changes, copies, riseMs) {
  if (!(riseMs > 0) || changes.size === 0) return [];
  const wrapperFor = bodyWrapperLookup(surface, root);
  const riders = glideRiders(root);
  /** @type {AnimationSpec[]} */
  const specs = [];
  const fade = (
    /** @type {Element} */ element,
    /** @type {number} */ from,
    /** @type {number} */ to,
  ) =>
    specs.push({
      element,
      keyframes: [{ opacity: from }, { opacity: to }],
      options: { delay: 0, duration: riseMs, fill: "backwards" },
      id: `mbg:${epochKey}:rise:${specs.length}`,
    });
  for (const { copy } of copies) fade(copy, 1, 0);
  for (const key of changes.keys()) {
    const wrapper = wrapperFor(key);
    for (const element of wrapper ? [wrapper, ...(riders.get(key) ?? [])] : []) {
      const value = Number(element.getAttribute("opacity"));
      fade(
        element,
        0,
        element.hasAttribute("opacity") && Number.isFinite(value) ? value : 1,
      );
    }
  }
  return specs;
}

/**
 * Return a copy of spec whose animation starts delay ms later.
 *
 * spec is any AnimationSpec; delay is in ms (0 returns spec itself). The
 * copy's options.delay is the old delay (default 0) plus delay; with fill
 * "backwards" it holds its first keyframe until then. Pure.
 *
 * @param {AnimationSpec} spec
 * @param {number} delay
 * @returns {AnimationSpec}
 */
function delayedSpec(spec, delay) {
  if (!(delay > 0)) return spec;
  return {
    ...spec,
    options: { ...spec.options, delay: Number(spec.options.delay ?? 0) + delay },
  };
}

/**
 * Build the landing waits: what-comes-next marks of a gliding champion fade in
 * once the bodies have landed.
 *
 * epochKey names the tick for the ids (mbg:<epochKey>:landing-wait:<n>); root
 * is the battlefield SVG; gliding holds the presentation keys of the champions
 * that glide; at is when the bodies have landed and fade how long the marks
 * take to appear, in ms (a played tick: 260 and 80; a one-tick step: 240 and
 * 80, later when a rewind first raises a fallen champion). Each element under
 * root with data-landing-wait (space-separated presentation keys, see
 * looks/champion-frame.js: a next-action line and glyph, an All Staged Orders
 * line, a Next Step Ghost) that names a gliding champion gets opacity 0 until
 * at, then fades to its END opacity (its opacity attribute, default 1), fill
 * "backwards". A mark whose champions all stay put is left alone, so it does
 * not blink. Return a new array of specs. Reads the DOM only.
 *
 * @param {string} epochKey
 * @param {Element} root
 * @param {ReadonlySet<string>} gliding
 * @param {number} at
 * @param {number} fade
 * @returns {AnimationSpec[]}
 */
function landingWaitSpecs(epochKey, root, gliding, at, fade) {
  if (gliding.size === 0) return [];
  /** @type {AnimationSpec[]} */
  const specs = [];
  for (const element of root.querySelectorAll("[data-landing-wait]")) {
    const keys = String(element.getAttribute("data-landing-wait") ?? "").split(" ");
    if (!keys.some((key) => gliding.has(key))) continue;
    const value = Number(element.getAttribute("opacity"));
    const end = element.hasAttribute("opacity") && Number.isFinite(value) ? value : 1;
    specs.push({
      element,
      keyframes: [{ opacity: 0 }, { opacity: end }],
      options: { delay: at, duration: fade, fill: "backwards" },
      id: `mbg:${epochKey}:landing-wait:${specs.length}`,
    });
  }
  return specs;
}

/**
 * Keep a one-tick step's overrides inside the animation limit.
 *
 * specs are the step's overrides in priority order (a -1 rewind: bodies,
 * riders, rise fades, cue fades, waits, look marks; onto a first frame:
 * bodies, waits, riders). A step runs no beat holds, and its riders and waits
 * depend on switches (Ranges, All Staged Orders, Next Step Ghost) that the
 * plan's played-tick override allowance does not count, so the step is bounded
 * by the animation limit (PLAN_RESOURCE_CAPS.animations, less its one clock)
 * instead; the worst case (ten champions, every switch on) is about 140. Return
 * the first specs that fit. Pure.
 *
 * @param {ReadonlyArray<AnimationSpec>} specs
 * @returns {AnimationSpec[]}
 */
function boundedStepOverrides(specs) {
  return specs.slice(0, MAX_ACTIVE_ANIMATIONS - 1);
}

/**
 * Build the cue fade of a -1 rewind: the tick's cues fade in once the bodies
 * have landed.
 *
 * plan names the epoch; installation holds the painter roots (root, connectorRoot,
 * routeRoot), its eventNodes and its carry; at is when the bodies have landed,
 * in ms (default 240; later when fallen champions rise first). Each present
 * root gets opacity 0 until at, then fades to 1 over 80 ms, so a stroke never
 * stretches between moving bodies and a landing, a number or a mark never
 * shows where a body is still going. Each Herald Card of this tick (an eventNodes entry of kind herald_card
 * that is not one of carry.heldRows) does the same, to its opacity attribute
 * (default 1); a card held from the tick before stays as it was. Every spec uses
 * fill "backwards" and no forward fill. Return a new array of specs (ids
 * mbg:<epoch>:step-cue:<n>). Reads the installation only.
 *
 * @param {Record<string, any>} plan
 * @param {ChoreographyInstallation} installation
 * @param {number} [at]
 * @returns {AnimationSpec[]}
 */
function stepCueFadeSpecs(plan, installation, at = BEAT_CLOCK.stepGlide) {
  const record = /** @type {Record<string, any>} */ (installation);
  const heldIds = new Set(
    (Array.isArray(record.carry?.heldRows) ? record.carry.heldRows : []).map(
      (/** @type {Record<string, any>} */ row) => row?.eventId,
    ),
  );
  /** @type {Array<{element: Element, end: number}>} */
  const targets = [record.routeRoot, record.connectorRoot, record.root]
    .filter((element) => element && typeof element === "object")
    .map((element) => ({ element, end: 1 }));
  if (record.eventNodes instanceof Map) {
    for (const [eventId, entry] of record.eventNodes) {
      if (entry?.event?.kind !== "herald_card" || heldIds.has(eventId)) continue;
      const element = entry.group;
      if (!element || typeof element !== "object") continue;
      const value = Number(element.getAttribute?.("opacity"));
      const end =
        element.hasAttribute?.("opacity") && Number.isFinite(value) ? value : 1;
      targets.push({ element, end });
    }
  }
  return targets.map(({ element, end }, index) => ({
    element,
    keyframes: [{ opacity: 0 }, { opacity: end }],
    options: {
      delay: at,
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
 * Return whether a step from the shown plan to the next plan is one tick
 * forward for the same audience.
 *
 * previous is the plan shown now, or null when the shown frame has none (a
 * replay's first frame: any one-tick step from it is forward); next is the
 * plan the step lands on. True when previous is null, or when both have the
 * same string audienceKey and next.simulatorStep is previous.simulatorStep + 1.
 * Pure.
 *
 * @param {Record<string, any> | null} previous
 * @param {Record<string, any>} next
 */
function oneTickForward(previous, next) {
  if (previous === null) return true;
  return (
    typeof next.audienceKey === "string" &&
    previous.audienceKey === next.audienceKey &&
    Number(next.simulatorStep) - Number(previous.simulatorStep) === 1
  );
}

/**
 * Stand in for the plan of a shown frame that has none, for a -1 onto it.
 *
 * A replay's first frame has no tick leading into it, so the choreographer
 * keeps no plan while it shows (planBuilder returns null and presentFrame
 * clears). plan is the plan of the frame shown before a -1 onto it, one tick
 * later; each of its bodies' START spots (body.start) is where the first
 * frame shows that body. Return a frozen plan-like record for stepGlideSpecs
 * (the landing of #startPlanlessStepGlide): plan's audienceKey, epochKey and
 * beats, simulatorStep one less (so the step reads as one tick back), and one
 * body per plan body with a
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
