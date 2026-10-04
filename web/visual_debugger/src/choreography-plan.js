/**
 * @file Turn authorized incoming facts into bounded combat presentation plans and
 * own the beat clock.
 *
 * What it owns:
 * - buildChoreographyPlan: reads one normalized presentation frame, applies local
 *   paint filters, joins disclosed phase anchors, allocates display space and gives
 *   every event a time window in milliseconds on the beat clock.
 * - The beat clock (BEAT_CLOCK, beatSchedule): bodies glide first (0-260 ms), blows
 *   start at 280 ms, then Settle, then one Fall beat for the deaths (900 ms of blow
 *   time, plus 60 ms for each earlier fall, ordered by public ID), one Return beat,
 *   and a 600 ms end hold. Longer Tick (switch scholar_long_tick, on by default)
 *   makes a tick 1700 ms (3200 ms with a death). Reduced motion keeps a short copy.
 * - beatView: the one owner of values the HUD and panels hold back until a beat
 *   (life before a fall, score before the Fall beat, returns before the Return beat).
 * - The derived event families: Home Fall glow, herald cards, Fallen feed rows, Last Seen
 *   rings and Focus Tally, each with its Oracle View and Agent POV admission rule.
 *   A herald card row carries the tick's moment facts (heraldTickFacts: the
 *   public scoreboard, the frame's own match_moments host fact, and this
 *   audience's Ultimate combos and wiped teams); looks/heralds.js names the
 *   moments.
 * - plan.bodies (glide records for the choreographer, with dimShift for a path
 *   that cuts a stone) and plan.camera (the camera's input: body spots and fights).
 * - The look inputs on each row, all in CSS px and all from this audience's own
 *   anchors: an activation carries sourceRadius, targetRadius, unitPx (also
 *   pixelsPerUnit), sourceEnd and targetEnd (null when its body was not seen at
 *   END) and, for a Charge in Oracle View, chargeLanding and chargeTravel (world
 *   units); a Basic's paint parts add glow, close_range and kindled and an
 *   Ultimate's add softGlow (these refine a drawn activation, they never keep a
 *   switched-off one drawn). sourceKindled says the source's Mage Burst was
 *   active when the blow landed (kindledSourceKeys: this audience's own status
 *   rows, END scene and, in Agent POV, its team's start-of-tick observation);
 *   outputAmount is null (no incoming row carries it yet). A fall or return pulse
 *   and a drawn Spawn Shield expiry (the Ward Lift) carry anchorRadius, and a
 *   fall its homeFall flag; Death Announcer members carry
 *   points and homeFall; a Focus Tally row carries anchorRadius. A health change
 *   carries maximumHealth and recipientRadius (0 when not drawn) and, in Oracle
 *   View only, the recorded totalDamage, totalHealing and damageSources (each
 *   attacker's recorded effective damage on it; Agent POV rows carry null and
 *   none). Its paint parts add savedBeat, lastBreath and battleTextDetail, and a
 *   rest heal's add battleTextDetail (these refine a drawn number, they never keep
 *   a switched-off one drawn). Health numbers and rest heals pop at Impact (blow
 *   260 ms).
 *
 * Who calls it: CombatChoreographer (choreography.js) through presentFrame, the
 * PNG export and the unit tests. Look modules read the records it returns.
 *
 * Boundary: it never reads outgoing inspection, submits actions or reconstructs
 * simulator events. Every spatial fact and every beat time comes only from the
 * audience's own events: Oracle View uses latest_events, Agent POV uses
 * visual_events. In Agent POV it never reads match_summary.deaths or the global
 * researcher_space for anything drawn on or over the arena, so a death the team did
 * not see changes no record and no time. The herald cards also read the public
 * scoreboard (scores before and after the tick, First To, the tick and the draw
 * tick, the same in every view) and the frame's own per-view host facts. Researcher references can still join
 * display identity or Ultimate help for bodies the team does see.
 */
import { canonicalAgentIdentity } from "./agent-identity.js";
import {
  authorizedPresentationAgentDisplayId,
  authorizedPresentationAudience,
  authorizedPresentationIncomingRows,
  authorizedPresentationMatchBeatFacts,
  authorizedPresentationResearcherSceneView,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import { layoutCrossPhaseOccupancy } from "./layout.js";
// Footprints and budgets have one owner each: the look module that draws the part.
import {
  BUDGET as BASIC_BUDGET,
  FOOTPRINT as BASIC_FOOTPRINT,
} from "./looks/basics.js";
import { BUDGET as FOCUS_TALLY_BUDGET } from "./looks/focus-tally.js";
import { BUDGET as FOG_BUDGET } from "./looks/fog.js";
import {
  BUDGET as HEALTH_TEXT_BUDGET,
  FOOTPRINT as HEALTH_TEXT_FOOTPRINT,
  lifeNumberCueBox,
} from "./looks/health-text.js";
import {
  cardMoments,
  BUDGET as HERALD_BUDGET,
  momentEntersAtSettle,
  rowBudget,
} from "./looks/heralds.js";
import { cutsStone } from "./looks/movement.js";
import { BUDGET as RED_ZONE_BUDGET } from "./looks/red-zone.js";
import {
  BUDGET as REQUIEM_BUDGET,
  FOOTPRINT as REQUIEM_FOOTPRINT,
} from "./looks/requiems.js";
import { CALL_BUDGET as WAYSTONE_CALL_BUDGET } from "./looks/spawns.js";
import {
  BUDGET as STATUS_BUDGET,
  FOOTPRINT as STATUS_FOOTPRINT,
} from "./looks/statuses.js";
import { ACTIVATION_FOOTPRINTS } from "./looks/strike-routes.js";
import {
  BUDGET as ULTIMATE_BUDGET,
  FOOTPRINT as ULTIMATE_FOOTPRINT,
} from "./looks/ultimates.js";
import {
  matchSummaryView,
  matchTeamSide,
  publicRosterIdentities,
} from "./match-summary.js";
import { createRouteGeometry, routeMarkerPose } from "./routes.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualPaintPartEnabled,
  visualFilterPaintKey,
} from "./visual-filters.js";
import {
  activationImpactSemantic,
  classTokenFromId,
  resolveVisualToken,
  ultimateTokenFromClassId,
} from "./vocabulary.js";

export const CHOREOGRAPHY_PHASES = Object.freeze({
  activationStart: 0,
  travelStart: 80,
  impactStart: 360,
  outcomeStart: 420,
  submissionRelease: 450,
  settleStart: 760,
  v2AbilityStart: 0,
  v2HealthResolutionStart: 420,
  v2CountdownAndRegenStart: 900,
  v2CooldownStart: 1220,
  v2ChargeStart: 1540,
  v2MovementStart: 2080,
  v2DeathStart: 2080,
  v2StatusStart: 2560,
  v2ShieldStart: 3040,
  v2RespawnWaveStart: 3520,
  v2RespawnStart: 4000,
  // Recipient POV deltas expose only adjacent observations, not privileged
  // transition phases. Spatial POV cues therefore share one explicitly
  // non-causal successor-observation phase.
  povSuccessorObservationStart: 0,
  total: 900,
  reducedTotal: 220,
});

/**
 * The beat clock, in milliseconds of tick time at playback rate 1.
 *
 * "Blow time" (also called part time) is the time every look is authored in: blows
 * travel from 80 ms, land at 360 ms and the choreography ends at 900 ms. When Body
 * Glide is on, the real clock bends blow time around the glide: real 0-80 ms is blow
 * time 0-60 ms, real 80-280 ms is blow time 60-80 ms (the clock waits while bodies
 * land at 260 ms), real 280 ms to 900 ms plus the lead is blow time 80-900 ms, and
 * after that real time is blow time plus the lead. The lead is 200 ms with Longer
 * Tick on (so blows keep their own speed) and 0 ms off (blows play 1.32 times
 * faster). With Body Glide off, real time equals blow time. See beatRealTime.
 *
 * Fields:
 * - glideEnd: every body has landed (real ms). chargeRushEnd: a charging Warrior's
 *   rush from START to its post-Charge spot ends (real ms).
 * - blowStart: the first blow may start (real ms) when bodies glide first.
 * - impactBlow: blow time of the Impact moment, when life changes and numbers
 *   pop; the "impact" beat hold (choreography.js) and beatView's hits use it.
 * - longTickLead: extra real ms per tick while Longer Tick is on.
 * - blowEnd: blow time when the choreography ends (Settle ends).
 * - fallBeat, returnBeat, endHold: beat lengths in ms. fallStagger: extra ms per
 *   earlier fall in the Fall beat.
 * - returnSetDown: ms after the Return beat starts when a champion is set down on its
 *   pad while Resurrection Effects is on. returnWithoutBeat: blow time of a return
 *   when Respawn Wave is off (no Return beat).
 * - reducedChoreography: real ms the blows take in reduced motion.
 * - stepGlide, stepChargeRushForward, stepChargeRushBackward, stepCueFadeEnd: the
 *   one-tick step glide (replay_step): bodies glide 240 ms, a Charge rush takes the
 *   first 100 ms forwards or the first 140 ms backwards, and the tick's cues fade in
 *   from 240 ms to 320 ms.
 */
export const BEAT_CLOCK = Object.freeze({
  glideEnd: 260,
  chargeRushEnd: 110,
  blowStart: 280,
  impactBlow: 260,
  longTickLead: 200,
  blowEnd: 900,
  fallBeat: 1500,
  fallStagger: 60,
  returnBeat: 900,
  returnSetDown: 300,
  returnWithoutBeat: 880,
  endHold: 600,
  reducedChoreography: 220,
  stepGlide: 240,
  stepChargeRushForward: 100,
  stepChargeRushBackward: 140,
  stepCueFadeEnd: 320,
});

/**
 * The paint-part tags of the derived event families this plan builder adds, and the
 * visual filter that owns each one.
 *
 * Each look module that draws one of these families registers exactly this tag in
 * its PAINT_PARTS export, so visual-filters.js can resolve it. Until a tag is
 * registered the family is kept in the plan as data only (not drawn): an
 * unregistered paint part is never silently drawn.
 */
export const DERIVED_EVENT_PAINT_PARTS = Object.freeze({
  home_fall: Object.freeze({
    part: Object.freeze({ surface: "transient", kind: "home_fall" }),
    filter: "red_zone_home_fall",
  }),
  herald_card: Object.freeze({
    part: Object.freeze({ surface: "transient", kind: "herald_card" }),
    filter: "match_moments",
  }),
  fallen_feed_row: Object.freeze({
    part: Object.freeze({ surface: "transient", kind: "fallen_feed_row" }),
    filter: "death_announcer",
  }),
  last_seen: Object.freeze({
    part: Object.freeze({ surface: "transient", kind: "last_seen" }),
    filter: "sight_change_marks",
  }),
  focus_tally: Object.freeze({
    part: Object.freeze({ surface: "transient", kind: "focus_tally" }),
    filter: "focus_fire",
  }),
});

/**
 * Read one boolean switch from a visual filter state, with a default.
 *
 * state is the page's filter state (any object); id is a switch id. Return the
 * switch's boolean when the state carries one, otherwise fallback. This tolerant
 * reader lets switches that another module has not registered yet keep their
 * documented default instead of throwing. It never changes the state.
 *
 * @param {unknown} state
 * @param {string} id
 * @param {boolean} fallback
 */
function filterSwitch(state, id, fallback) {
  const value = /** @type {Record<string, unknown> | null} */ (
    typeof state === "object" && state !== null ? state : null
  )?.[id];
  return typeof value === "boolean" ? value : fallback;
}

/**
 * Convert blow time (the time looks are authored in) to real tick time.
 *
 * part is blow time in ms (any finite number; negative values pass through).
 * glide says whether bodies glide first (Body Glide on, normal motion); lead is the
 * Longer Tick lead in ms (0 or 200). Return real ms at playback rate 1. With glide
 * off the two clocks are equal. The map is piecewise linear through the knots
 * (real, blow) = (0, 0), (80, 60), (280, 80), (900 + lead, 900) and adds the lead
 * after 900 ms of blow time. Pure; no side effects.
 *
 * @param {number} part
 * @param {boolean} glide
 * @param {number} lead
 */
export function beatRealTime(part, glide, lead) {
  if (!glide || part <= 0) return part;
  const knots = [
    [0, 0],
    [80, 60],
    [BEAT_CLOCK.blowStart, 80],
    [BEAT_CLOCK.blowEnd + lead, BEAT_CLOCK.blowEnd],
  ];
  const last = knots[knots.length - 1];
  if (part >= last[1]) return part - last[1] + last[0];
  for (let index = 1; index < knots.length; index += 1) {
    const [realStart, partStart] = knots[index - 1];
    const [realEnd, partEnd] = knots[index];
    if (part <= partEnd) {
      return (
        realStart + ((part - partStart) * (realEnd - realStart)) / (partEnd - partStart)
      );
    }
  }
  return part;
}

/**
 * Compare two public agent IDs in the natural order used for beat staggering.
 *
 * left and right are public agent ID strings such as "agent-slot-5". Digits compare
 * as numbers ("agent-slot-2" before "agent-slot-10"). Return a negative number, zero
 * or a positive number for Array.prototype.sort. Pure.
 *
 * @param {string} left
 * @param {string} right
 */
function comparePublicAgentIds(left, right) {
  return left.localeCompare(right, "en", { numeric: true });
}

/**
 * @typedef {Readonly<{
 *   presentationKey: string,
 *   publicAgentId: string,
 * }>} BeatSubject
 * @typedef {Readonly<{
 *   presentationKey: string,
 *   publicAgentId: string,
 *   order: number,
 *   at: number,
 * }>} FallBeatEntry
 * @typedef {Readonly<{
 *   presentationKey: string,
 *   publicAgentId: string,
 *   at: number,
 *   setDownAt: number,
 * }>} ReturnBeatEntry
 * @typedef {Readonly<{
 *   glide: boolean,
 *   stepGlide: boolean,
 *   longTick: boolean,
 *   lead: number,
 *   blowSpeed: number,
 *   windows: Readonly<{
 *     stride: readonly [number, number] | null,
 *     glide: readonly [number, number] | null,
 *     chargeRush: readonly [number, number] | null,
 *     strike: readonly [number, number],
 *     impact: readonly [number, number],
 *     settle: readonly [number, number],
 *   }>,
 *   choreographyEnd: number,
 *   fall: readonly [number, number] | null,
 *   falls: ReadonlyArray<FallBeatEntry>,
 *   returnBeat: readonly [number, number] | null,
 *   returns: ReadonlyArray<ReturnBeatEntry>,
 *   scoreAt: number,
 *   explanationEnd: number,
 *   endHold: readonly [number, number],
 *   tickMs: number,
 *   reducedScale: number,
 *   reducedTickMs: number,
 * }>} BeatSchedule
 */

/**
 * Return a frozen [start, end] pair of milliseconds.
 *
 * start and end are real ms; no ordering check is made. Pure.
 *
 * @param {number} start
 * @param {number} end
 * @returns {readonly [number, number]}
 */
function msPair(start, end) {
  return /** @type {readonly [number, number]} */ (Object.freeze([start, end]));
}

/**
 * Build the beat schedule for one tick from the audience's own falls and returns.
 *
 * options.falls lists the champions whose agent_died event this audience received
 * (Oracle View: latest_events; Agent POV: visual_events). options.returns lists the
 * champions whose agent_respawned event it received. A death or return the audience
 * did not see must not be passed, so it changes no time. options.visualFilters is
 * the page's filter state (default: every switch at its default): stride_glide
 * (Body Glide, default on), scholar_long_tick (Longer Tick, default on),
 * death_effects (owns the Fall beat, default on), respawn_wave (owns the Return
 * beat, default on) and resurrection_effects (300 ms set-down, default on).
 *
 * Return a frozen schedule in real ms at playback rate 1 for normal motion:
 * - glide (Body Glide on: played ticks glide bodies first), stepGlide (Step Glide,
 *   switch step_glide, default on: a one-tick step glides), longTick and lead (200
 *   ms with Longer Tick, else 0), blowSpeed (1, or 1.32 when blows are squeezed).
 * - windows: stride and glide (null when Body Glide is off), chargeRush, strike,
 *   impact and settle. choreographyEnd is when Settle ends (900 + lead).
 * - fall: the Fall beat [start, end] (1500 ms) or null when no seen death or Death
 *   Effects is off. falls: one entry per seen death, ordered by public ID, each at
 *   the beat start plus 60 ms per earlier fall (at choreographyEnd without a beat).
 * - returnBeat: [start, end] (900 ms) or null; returns: each seen return with the
 *   beat start (at) and its set-down time.
 * - scoreAt: when a score change shows. It is the Fall beat start, choreographyEnd,
 *   on every tick, so in Agent POV it never depends on which deaths were seen.
 * - explanationEnd, endHold (600 ms) and tickMs (the tick's full length: 1700 ms on
 *   an ordinary tick with Longer Tick, 3200 ms with a death).
 * - reducedScale: multiply any time except the end hold by it for reduced motion
 *   (blows end at 220 ms). reducedTickMs is the reduced tick length.
 * Pure; never reads a frame.
 *
 * @param {{
 *   falls?: ReadonlyArray<BeatSubject>,
 *   returns?: ReadonlyArray<BeatSubject>,
 *   visualFilters?: unknown,
 * }} [options]
 * @returns {BeatSchedule}
 */
export function beatSchedule(options = {}) {
  const visualFilters = options.visualFilters ?? null;
  const glide = filterSwitch(visualFilters, "stride_glide", true);
  const longTick = glide && filterSwitch(visualFilters, "scholar_long_tick", true);
  const lead = longTick ? BEAT_CLOCK.longTickLead : 0;
  /**
   * Convert blow time part (ms) to real ms for this schedule's glide and lead. Pure.
   *
   * @param {number} part
   */
  const real = (part) => beatRealTime(part, glide, lead);
  const choreographyEnd = real(BEAT_CLOCK.blowEnd);
  const seenFalls = [...(options.falls ?? [])].sort((left, right) =>
    comparePublicAgentIds(left.publicAgentId, right.publicAgentId),
  );
  const seenReturns = [...(options.returns ?? [])].sort((left, right) =>
    comparePublicAgentIds(left.publicAgentId, right.publicAgentId),
  );
  const fall =
    seenFalls.length > 0 && filterSwitch(visualFilters, "death_effects", true)
      ? msPair(choreographyEnd, choreographyEnd + BEAT_CLOCK.fallBeat)
      : null;
  const falls = Object.freeze(
    seenFalls.map((subject, order) =>
      Object.freeze({
        presentationKey: subject.presentationKey,
        publicAgentId: subject.publicAgentId,
        order,
        at: fall ? fall[0] + BEAT_CLOCK.fallStagger * order : choreographyEnd,
      }),
    ),
  );
  const returnStart = fall ? fall[1] : choreographyEnd;
  const returnBeat =
    seenReturns.length > 0 && filterSwitch(visualFilters, "respawn_wave", true)
      ? msPair(returnStart, returnStart + BEAT_CLOCK.returnBeat)
      : null;
  const setDown =
    returnBeat && filterSwitch(visualFilters, "resurrection_effects", true)
      ? BEAT_CLOCK.returnSetDown
      : 0;
  const returns = Object.freeze(
    seenReturns.map((subject) =>
      Object.freeze({
        presentationKey: subject.presentationKey,
        publicAgentId: subject.publicAgentId,
        at: returnBeat ? returnBeat[0] : real(BEAT_CLOCK.returnWithoutBeat),
        setDownAt: returnBeat
          ? returnBeat[0] + setDown
          : real(BEAT_CLOCK.returnWithoutBeat),
      }),
    ),
  );
  const explanationEnd = returnBeat ? returnBeat[1] : fall ? fall[1] : choreographyEnd;
  const reducedScale = BEAT_CLOCK.reducedChoreography / choreographyEnd;
  return Object.freeze({
    glide,
    stepGlide: filterSwitch(visualFilters, "step_glide", true),
    longTick,
    lead,
    blowSpeed:
      (BEAT_CLOCK.blowEnd - 80) /
      (glide ? choreographyEnd - BEAT_CLOCK.blowStart : BEAT_CLOCK.blowEnd - 80),
    windows: Object.freeze({
      stride: glide ? msPair(0, BEAT_CLOCK.blowStart) : null,
      glide: glide ? msPair(0, BEAT_CLOCK.glideEnd) : null,
      chargeRush: glide ? msPair(0, BEAT_CLOCK.chargeRushEnd) : null,
      strike: msPair(real(80), real(240)),
      impact: msPair(real(240), real(420)),
      settle: msPair(real(420), choreographyEnd),
    }),
    choreographyEnd,
    fall,
    falls,
    returnBeat,
    returns,
    scoreAt: choreographyEnd,
    explanationEnd,
    endHold: msPair(explanationEnd, explanationEnd + BEAT_CLOCK.endHold),
    tickMs: explanationEnd + BEAT_CLOCK.endHold,
    reducedScale,
    reducedTickMs: explanationEnd * reducedScale + BEAT_CLOCK.endHold,
  });
}

/**
 * Return the Fall beat time of one champion, or null when it has no seen fall.
 *
 * schedule is a beatSchedule result; presentationKey is the champion's key in the
 * audience's own scene. Pure.
 *
 * @param {BeatSchedule} schedule
 * @param {string | null | undefined} presentationKey
 */
function fallTimeFor(schedule, presentationKey) {
  return (
    schedule.falls.find((entry) => entry.presentationKey === presentationKey)?.at ??
    null
  );
}

const STATUS_EVENT_TYPES = new Set([
  "status_aged_to_zero",
  "status_broken_by_damage",
  "status_applied",
  "status_refreshed_or_extended",
  "status_cleared_by_new_death",
]);
const SPAWN_SHIELD_STATUS_TOKEN = Object.freeze({
  tokenId: "spawn_shield",
  label: "Spawn Shield",
  shortLabel: "Shield",
  accessibleName: "Spawn Shield",
  glyphKey: "status-spawn-shield",
  cssKey: "spawn-shield",
  fallback: "S",
});
/**
 * @typedef {"ability" | "health" | "recovery" | "cooldown" | "charge" | "movement" | "death" | "status" | "shield" | "respawn_wave" | "respawn"} ChoreographyFamily
 */
/**
 * The beat window each choreography family plays in.
 *
 * "blows": the strike/impact window (blow time 80-760 ms with Body Glide, 0-760 ms
 * without). "settle": the Settle window (blow time 420-900 ms); every outcome shares
 * it instead of queueing one family after another. "fall": the Fall beat of the
 * event's own victim. "return": the Return beat. "none": no time (feed only).
 *
 * @type {Readonly<Record<ChoreographyFamily, "blows" | "settle" | "fall" | "return" | "none">>}
 */
const FAMILY_BEAT_WINDOW = Object.freeze({
  ability: "blows",
  health: "settle",
  recovery: "settle",
  cooldown: "settle",
  charge: "settle",
  movement: "none",
  death: "fall",
  status: "settle",
  shield: "settle",
  respawn_wave: "return",
  respawn: "return",
});
// A death cue without a Fall beat (Death Effects off) keeps today's readable dwell.
const READABLE_OUTCOME_DWELL_MS =
  CHOREOGRAPHY_PHASES.total - CHOREOGRAPHY_PHASES.outcomeStart;
// A return without a Return beat (Respawn Wave off) keeps today's 620 ms dwell.
const RETURN_WITHOUT_BEAT_DWELL_MS = 620;
// Death Announcer cards keep their full 1500 ms reading time.
const DEATH_ANNOUNCEMENT_DWELL_MS = 1500;

/**
 * Shared allocator/painter geometry in presentation pixels, gathered from the look
 * modules that own it (basics.js and ultimates.js for activation marks and the
 * Charge label, strike-routes.js for route padding). Kept as one export so allocator callers and tests read one record;
 * change a number in its look module, never here. Stroke-inclusive radii and flare
 * extents fit inside their footprint; route padding excludes transparent hit
 * targets and underpainting.
 */
export const CHOREOGRAPHY_PAINT_FOOTPRINTS = Object.freeze({
  activation: Object.freeze({
    ...BASIC_FOOTPRINT.activation,
    ...ULTIMATE_FOOTPRINT.activation,
  }),
  chargeOwnership: ULTIMATE_FOOTPRINT.chargeOwnership,
  route: ACTIVATION_FOOTPRINTS.route,
});

// Cue rectangles of the outcome looks (health-text.js and statuses.js own them).
const NET_EFFECT_CUE_RECTANGLE = centeredCueRectangle(
  HEALTH_TEXT_FOOTPRINT.netEffect.width,
  HEALTH_TEXT_FOOTPRINT.netEffect.height,
);
const NET_TEXT_CUE_RECTANGLE = centeredCueRectangle(
  HEALTH_TEXT_FOOTPRINT.netText.width,
  HEALTH_TEXT_FOOTPRINT.netText.height,
);
const UNCHANGED_NET_TEXT_CUE_RECTANGLE = centeredCueRectangle(
  HEALTH_TEXT_FOOTPRINT.unchangedNetText.width,
  HEALTH_TEXT_FOOTPRINT.unchangedNetText.height,
);
/**
 * Body radius in CSS pixels used for a life number's ring when the recipient's
 * projected radius is unknown (a 0.5-unit body on a typical arena).
 */
const RECIPIENT_CUE_FALLBACK_RADIUS_PX = 14;
const REGENERATION_EFFECT_CUE_RECTANGLE = centeredCueRectangle(
  HEALTH_TEXT_FOOTPRINT.regenerationEffect.width,
  HEALTH_TEXT_FOOTPRINT.regenerationEffect.height,
);
const REGENERATION_TEXT_CUE_RECTANGLE = HEALTH_TEXT_FOOTPRINT.regenerationText;

/**
 * @typedef {{
 *   worldToScreen: (point: readonly [number, number] | {x: number, y: number}) =>
 *     {x: number, y: number},
 *   worldLengthToScreen: (length: number) => number,
 *   viewportBounds?: {
 *     left: number,
 *     top: number,
 *     right: number,
 *     bottom: number,
 *     width: number,
 *     height: number,
 *   },
 *   protectedRects?: ReadonlyArray<Record<string, any>>,
 * }} ProjectionSurface
 * @typedef {Record<string, any> & {
 *   events: ReadonlyArray<Record<string, any>>,
 *   derivedEvents: ReadonlyArray<Record<string, any>>,
 *   bodies: ReadonlyArray<PlanBody>,
 *   beats: BeatSchedule,
 *   camera: Readonly<{
 *     audience: "researcher" | "agent_pov" | null,
 *     bodies: ReadonlyArray<Readonly<Record<string, any>>>,
 *     fights: ReadonlyArray<Readonly<Record<string, any>>>,
 *   }>,
 *   bounds: Readonly<{
 *     nodes: number,
 *     animations: number,
 *     persistentNodes: number,
 *     overrideAnimations: number,
 *   }>,
 * }} ChoreographyPlan
 *   One tick's plan (see buildChoreographyPlan for every field).
 */

/**
 * Return whether value is a non-null object other than an array.
 *
 * This broad shape guard does not validate a wire record or grant authority.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return value unchanged when it is a non-array object, otherwise null.
 *
 * No fields are validated or copied; callers check the parts they need.
 *
 * @param {unknown} value
 * @returns {Record<string, any> | null}
 */
function record(value) {
  return isRecord(value) ? value : null;
}

/**
 * Return value unchanged when it is an array, otherwise a new empty array.
 *
 * Elements are not validated or copied. This is a tolerant shape reader.
 *
 * @param {unknown} value
 * @returns {any[]}
 */
function array(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * Return a numeric integer unchanged, otherwise null.
 *
 * No positivity, range or safe-integer precision bound is imposed.
 *
 * @param {unknown} value
 * @returns {number | null}
 */
function integer(value) {
  return Number.isInteger(value) ? Number(value) : null;
}

/**
 * Return value unchanged when it is a finite number, otherwise null.
 *
 * Numeric strings, NaN and infinity are not converted into valid values.
 *
 * @param {unknown} value
 * @returns {number | null}
 */
function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Return trimmed nonblank string value, otherwise null.
 *
 * This checks text presence only, not a particular namespace or authority.
 *
 * @param {unknown} value
 * @returns {string | null}
 */
function identifier(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/**
 * Copy the narrow public identity needed by a semantic explanation.
 *
 * value must provide a nonblank presentation key/public ID, class ID 1..5 and team
 * ID 1 or 2. Return a frozen record with those fields and display_agent_id (or null),
 * or null if the required identity is invalid. Geometry/scientific fields are not
 * copied; display_agent_id is carried as supplied rather than validated here.
 *
 * @param {unknown} value
 */
function authorizedIdentitySnapshot(value) {
  const agent = record(value);
  const presentationKey = identifier(agent?.presentation_key);
  const publicAgentId = identifier(agent?.public_agent_id);
  const classId = integer(agent?.class_id);
  const teamId = integer(agent?.team_id);
  if (
    presentationKey === null ||
    publicAgentId === null ||
    classId === null ||
    classId < 1 ||
    classId > 5 ||
    (teamId !== 1 && teamId !== 2)
  ) {
    return null;
  }
  return Object.freeze({
    presentation_key: presentationKey,
    public_agent_id: publicAgentId,
    display_agent_id: agent?.display_agent_id ?? null,
    class_id: classId,
    team_id: teamId,
  });
}

/**
 * Compare exact public values while allowing one float32 representation step.
 *
 * left/right match through Object.is, or when both finite numbers and one equals
 * the float32 rounding of the other. Return a Boolean; no general tolerance is used.
 *
 * @param {unknown} left @param {unknown} right
 */
function sameRecordedPublicValue(left, right) {
  if (Object.is(left, right)) return true;
  return (
    typeof left === "number" &&
    Number.isFinite(left) &&
    typeof right === "number" &&
    Number.isFinite(right) &&
    (left === Math.fround(right) || Math.fround(left) === right)
  );
}

/**
 * Compare all shared durable-status facts before joining display attribution.
 *
 * rawLocal and rawResearcher must be records. Return true when the eleven status
 * identity, duration, mechanic and break fields match under sameRecordedPublicValue.
 * Direct sources are intentionally checked separately by the attribution join.
 * This comparison does not validate either complete record on its own.
 *
 * @param {unknown} rawLocal
 * @param {unknown} rawResearcher
 */
function samePublicStatusFacts(rawLocal, rawResearcher) {
  const local = record(rawLocal);
  const researcher = record(rawResearcher);
  if (!local || !researcher) return false;
  return [
    "status_channel",
    "status_id",
    "family",
    "configured_duration_steps",
    "remaining_duration",
    "source_class_id",
    "source_class_name",
    "source_action_component",
    "magnitude_kind",
    "magnitude",
    "breaks_on_positive_damage",
  ].every((field) => sameRecordedPublicValue(local[field], researcher[field]));
}

/**
 * Compare public ID, class and team after validating both identity snapshots.
 *
 * left/right may use different presentation keys or display IDs. Return false for
 * invalid identities; matching public facts alone do not authorize shared geometry.
 *
 * @param {unknown} left @param {unknown} right
 */
function samePublicAgentIdentity(left, right) {
  const leftIdentity = authorizedIdentitySnapshot(left);
  const rightIdentity = authorizedIdentitySnapshot(right);
  return (
    leftIdentity !== null &&
    rightIdentity !== null &&
    leftIdentity.public_agent_id === rightIdentity.public_agent_id &&
    leftIdentity.class_id === rightIdentity.class_id &&
    leftIdentity.team_id === rightIdentity.team_id
  );
}

/**
 * Use joined researcher display identity with the local authorized body key.
 *
 * rawResearcherAgent and rawLocalAgent must agree on public ID, class and team.
 * Return a frozen identity carrying the local presentation key, or null on mismatch.
 * Researcher keys and positions never enter cue geometry through this helper.
 *
 * @param {unknown} rawResearcherAgent
 * @param {unknown} rawLocalAgent
 */
function researcherIdentityForLocalAgent(rawResearcherAgent, rawLocalAgent) {
  if (!samePublicAgentIdentity(rawResearcherAgent, rawLocalAgent)) return null;
  const researcherIdentity = authorizedIdentitySnapshot(rawResearcherAgent);
  const localIdentity = authorizedIdentitySnapshot(rawLocalAgent);
  if (researcherIdentity === null || localIdentity === null) return null;
  return Object.freeze({
    ...researcherIdentity,
    presentation_key: localIdentity.presentation_key,
  });
}

/**
 * Add display attribution only to status applications already locally admitted.
 *
 * applicationSources contains the validated local application identities. group
 * names the recipient; statusChannel/statusId select exactly one current local
 * status. sceneByKey and researcherAgentByPublicId supply the two identity spaces.
 * Require equal recipient/status facts and exactly one matching direct source before
 * adding each joined sourceIdentity; otherwise that field is null. Return a frozen
 * array of frozen copies, including empty input. No source is added, no event is
 * admitted, and local keys/anchors remain unchanged.
 *
 * @param {ReadonlyArray<Readonly<Record<string, any>>>} applicationSources
 * @param {AuthorizedStatusGroup} group
 * @param {number} statusChannel
 * @param {string} statusId
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @param {Map<string, Record<string, any>>} researcherAgentByPublicId
 */
function researcherApplicationSources(
  applicationSources,
  group,
  statusChannel,
  statusId,
  sceneByKey,
  researcherAgentByPublicId,
) {
  /**
   * Return copies of admitted application sources with sourceIdentity set to null.
   *
   * The enclosing join failed. Preserve each local event/key/public ID and input order
   * without adding researcher attribution or changing the source array.
   */
  const withoutResearcherIdentity = () =>
    Object.freeze(
      applicationSources.map((source) =>
        Object.freeze({ ...source, sourceIdentity: null }),
      ),
    );
  if (applicationSources.length === 0) return Object.freeze([]);

  const localRecipient = sceneByKey.get(group.recipientPresentationKey);
  const researcherRecipient = researcherAgentByPublicId.get(
    group.recipientPublicAgentId,
  );
  if (
    !localRecipient ||
    !researcherRecipient ||
    !samePublicAgentIdentity(localRecipient, researcherRecipient)
  ) {
    return withoutResearcherIdentity();
  }
  const localStatuses = array(localRecipient.statuses).filter(
    (status) =>
      integer(record(status)?.status_channel) === statusChannel &&
      identifier(record(status)?.status_id) === statusId,
  );
  if (localStatuses.length !== 1) return withoutResearcherIdentity();
  const researcherStatuses = array(researcherRecipient.statuses).filter((status) =>
    samePublicStatusFacts(localStatuses[0], status),
  );
  if (researcherStatuses.length !== 1) return withoutResearcherIdentity();
  const researcherStatus = researcherStatuses[0];
  const directSources = array(record(researcherStatus)?.direct_sources);

  return Object.freeze(
    applicationSources.map((source) => {
      const localSource = sceneByKey.get(source.sourcePresentationKey);
      const researcherSource = researcherAgentByPublicId.get(
        source.sourcePublicAgentId,
      );
      const matchingDirectSources = directSources.filter(
        (directSource) =>
          identifier(record(directSource)?.source_public_agent_id) ===
          source.sourcePublicAgentId,
      );
      const directSource = record(matchingDirectSources[0]);
      const researcherIdentity = researcherIdentityForLocalAgent(
        researcherSource,
        localSource,
      );
      const sourceIdentity =
        matchingDirectSources.length === 1 &&
        directSource &&
        identifier(directSource.source_presentation_key) ===
          identifier(researcherSource?.presentation_key) &&
        researcherIdentity !== null &&
        researcherIdentity.class_id ===
          integer(record(researcherStatus)?.source_class_id)
          ? researcherIdentity
          : null;
      return Object.freeze({ ...source, sourceIdentity });
    }),
  );
}

/**
 * Read a serialized scientific identity as trimmed nonblank text.
 *
 * value returns a string or null via identifier. Canonical namespace/epoch joins
 * are checked by the surrounding plan, not by this text helper.
 *
 * @param {unknown} value
 * @returns {string | null}
 */
function scientificIdentity(value) {
  return identifier(value);
}

/**
 * Copy an exact two-number coordinate array into a frozen tuple.
 *
 * value must be an array of length two with finite numeric elements. Return null
 * for other shapes/types; these coordinates do not establish authorization alone.
 *
 * @param {unknown} value
 * @returns {readonly [number, number] | null}
 */
function point(value) {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    !Number.isFinite(value[0]) ||
    !Number.isFinite(value[1])
  ) {
    return null;
  }
  return Object.freeze([Number(value[0]), Number(value[1])]);
}

/**
 * Project an authorized world tuple onto a supplied screen surface.
 *
 * world or surface absent returns null. Otherwise call worldToScreen and return a
 * frozen finite x/y point, or null for nonfinite output. Projection errors propagate;
 * no hidden position is looked up and inputs are not changed.
 *
 * @param {readonly [number, number] | null} world
 * @param {ProjectionSurface | null} surface
 * @returns {{x: number, y: number} | null}
 */
function project(world, surface) {
  if (!world || !surface) {
    return null;
  }
  const projected = surface.worldToScreen(world);
  if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y)) {
    return null;
  }
  return Object.freeze({ x: Number(projected.x), y: Number(projected.y) });
}

/**
 * Reserve pixels only for enabled pieces of an outcome explanation.
 *
 * event supplies its kind ("net_health" or "regeneration"), outcome and optional
 * paintParts. Missing paintParts uses the legacy all-on footprint. Combine the
 * relevant health or regeneration effect/text rectangles and return frozen
 * centered width/height. Disabled siblings contribute no area, so they cannot
 * displace the visible cue. A drawn health number also adds the box its V5
 * add-ons need (looks/health-text.js lifeNumberCueBox: Battle Text Detail's size
 * and underline, Saved Beat's word), which is null while they are off. Status
 * lifecycle events reserve no box at all (layoutCrossPhaseEvents skips them).
 *
 * @param {Record<string, any>} event
 */
function paintedOutcomeCueDimensions(event) {
  const parts = event.paintParts;
  /**
   * Return whether a named paint part contributes to this event footprint.
   *
   * part is the local piece name. Missing paintParts retains legacy all-on behavior;
   * otherwise only an exact true value enables the piece.
   *
   * @param {string} part
   */
  const enabled = (part) => !parts || parts[part] === true;
  /** @type {Readonly<Record<string, number>>[]} */
  const rectangles = [];
  if (event.kind === "net_health") {
    if (enabled("effect")) {
      rectangles.push(NET_EFFECT_CUE_RECTANGLE);
    }
    if (enabled("battleText") || enabled("recipientText")) {
      rectangles.push(
        event.outcome === "unchanged"
          ? UNCHANGED_NET_TEXT_CUE_RECTANGLE
          : NET_TEXT_CUE_RECTANGLE,
      );
      const addOns = lifeNumberCueBox(event);
      if (addOns !== null) {
        rectangles.push(addOns);
      }
    }
  } else if (event.kind === "regeneration") {
    if (enabled("effect")) {
      rectangles.push(REGENERATION_EFFECT_CUE_RECTANGLE);
    }
    if (enabled("battleText")) {
      rectangles.push(REGENERATION_TEXT_CUE_RECTANGLE);
    }
  }
  return unionCueRectangles(rectangles);
}

/**
 * Return frozen left/top/right/bottom offsets centered on zero.
 *
 * width and height are caller-owned pixel dimensions; no validation is performed.
 *
 * @param {number} width @param {number} height
 */
function centeredCueRectangle(width, height) {
  return Object.freeze({
    left: -width / 2,
    top: -height / 2,
    right: width / 2,
    bottom: height / 2,
  });
}

/**
 * Return the symmetric pixel footprint covering all centered cue rectangles.
 *
 * rectangles contains validated local offsets. Return frozen width/height based on
 * the largest absolute extents; empty input gives zero by zero. Inputs are unchanged.
 *
 * @param {ReadonlyArray<Readonly<Record<string, number>>>} rectangles
 */
function unionCueRectangles(rectangles) {
  if (rectangles.length === 0) {
    return Object.freeze({ width: 0, height: 0 });
  }
  const horizontalExtent = Math.max(
    ...rectangles.flatMap(({ left, right }) => [Math.abs(left), Math.abs(right)]),
  );
  const verticalExtent = Math.max(
    ...rectangles.flatMap(({ top, bottom }) => [Math.abs(top), Math.abs(bottom)]),
  );
  return Object.freeze({
    width: horizontalExtent * 2,
    height: verticalExtent * 2,
  });
}

/**
 * Create an unambiguous display-fragment key without changing event identity.
 *
 * eventId is stringified and role names the fragment. Return JSON text for the tuple
 * [event, eventId, role]; this is a layout key, not a scientific event ID.
 *
 * @param {unknown} eventId
 * @param {string} role
 */
function crossPhaseLayoutKey(eventId, role) {
  return JSON.stringify(["event", String(eventId), role]);
}

/**
 * Return a frozen finite rectangle with computed width/height, or null.
 *
 * value must contain ordered left/top/right/bottom; zero width/height is allowed.
 * The original object is unchanged and no viewport containment is checked.
 *
 * @param {unknown} value
 * @returns {{left: number, top: number, right: number, bottom: number, width: number, height: number} | null}
 */
function normalizedRectangle(value) {
  const bounds = record(value);
  if (!bounds) {
    return null;
  }
  const left = finiteNumber(bounds.left);
  const top = finiteNumber(bounds.top);
  const right = finiteNumber(bounds.right);
  const bottom = finiteNumber(bounds.bottom);
  if (
    left === null ||
    top === null ||
    right === null ||
    bottom === null ||
    right < left ||
    bottom < top
  ) {
    return null;
  }
  return Object.freeze({
    left,
    top,
    right,
    bottom,
    width: right - left,
    height: bottom - top,
  });
}

/**
 * Read explicit protected drawing regions from the projection surface.
 *
 * surface may be null or omit protectedRects, producing a frozen empty array.
 * Each region supplies bounds directly or under bounds; normalize it and retain its
 * layoutKey, falling back to a deterministic legacy index key. Invalid regions throw
 * TypeError. Return frozen region copies; this assigns no body ownership by position.
 *
 * @param {ProjectionSurface | null} surface
 */
function crossPhaseProtectedRects(surface) {
  return Object.freeze(
    array(surface?.protectedRects).map((candidate, index) => {
      const region = record(candidate);
      const bounds = normalizedRectangle(region?.bounds ?? region);
      if (!region || !bounds) {
        throw new TypeError(`protectedRects[${index}] must contain valid bounds.`);
      }
      return Object.freeze({
        layoutKey:
          identifier(region.layoutKey) ?? JSON.stringify(["legacy-protected", index]),
        bounds,
      });
    }),
  );
}

/**
 * Find protected body regions explicitly owned by one presentation key.
 *
 * surface supplies protectedRects; ownerPresentationKey absent returns an empty
 * array. Return matching nonblank layout keys in source order. No proximity or
 * containment inference is used, and this helper does not reject duplicates.
 *
 * @param {ProjectionSurface | null} surface
 * @param {string | null | undefined} ownerPresentationKey
 */
function protectedBodyKeys(surface, ownerPresentationKey) {
  if (!ownerPresentationKey) {
    return [];
  }
  return array(surface?.protectedRects).flatMap((candidate) => {
    const region = record(candidate);
    const key = identifier(region?.layoutKey);
    return region?.protectedKind === "body" &&
      region.ownerPresentationKey === ownerPresentationKey &&
      key !== null
      ? [key]
      : [];
  });
}

/**
 * Return the single protected body key for a disclosed owner, or null.
 *
 * surface and ownerPresentationKey use protectedBodyKeys. More than one matching
 * region throws RangeError because choosing a nearby body would invent ownership.
 *
 * @param {ProjectionSurface | null} surface
 * @param {string | null | undefined} ownerPresentationKey
 */
function protectedBodyKey(surface, ownerPresentationKey) {
  const keys = protectedBodyKeys(surface, ownerPresentationKey);
  if (keys.length > 1) {
    throw new RangeError(
      `presentation owner ${ownerPresentationKey} has multiple protected bodies.`,
    );
  }
  return keys[0] ?? null;
}

/**
 * Return the allocator footprint for one surviving activation glyph.
 *
 * event supplies tokenId, component, optional target and paintParts. Source-local
 * Mage Burst uses its larger footprint; semantic-only target impacts use the
 * compact marks of the owning look (ultimates.js: Trap 16, Charge 18, others 24;
 * basics.js: 24). Otherwise use the registered ability footprint with the existing
 * fallback. All returned dimensions are pixels and no event admission changes.
 *
 * @param {Record<string, any>} event
 */
function activationCueDimensions(event) {
  if (!event.target) {
    return event.tokenId === "mage_burst"
      ? CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.mage_burst
      : CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.local;
  }
  if (event.paintParts?.ability !== true) {
    if (event.component === "ultimate") {
      const sizes = ULTIMATE_FOOTPRINT.semanticOnlyImpact;
      return event.tokenId === "hunter_trap"
        ? sizes.hunter_trap
        : event.tokenId === "warrior_charge"
          ? sizes.warrior_charge
          : sizes.default;
    }
    return BASIC_FOOTPRINT.semanticOnlyImpact;
  }
  const tokenId = /** @type {keyof typeof CHOREOGRAPHY_PAINT_FOOTPRINTS.activation} */ (
    event.tokenId
  );
  return (
    CHOREOGRAPHY_PAINT_FOOTPRINTS.activation[tokenId] ??
    CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.rogue_poison
  );
}

/**
 * Return the fixed pixel footprint for a semantic pulse.
 *
 * event.cueSemantic selects the death/respawn ring (requiems.js FOOTPRINT) or the
 * ordinary pulse size (statuses.js FOOTPRINT). The team respawn wave has no
 * arena box: it is never spatial.
 * These are drawing extents, not simulator radii.
 *
 * @param {Record<string, any>} event
 */
function semanticPulseCueDimensions(event) {
  if (event.cueSemantic === "agent_died" || event.cueSemantic === "agent_respawned") {
    return REQUIEM_FOOTPRINT.lifecycleRing;
  }
  return STATUS_FOOTPRINT.pulseCue;
}

/**
 * Project the disclosed body radius named by presentationKey.
 *
 * sceneByKey contains authorized current bodies; surface converts world lengths
 * to pixels. Return zero when the body/radius/surface is absent or projection is
 * nonfinite. Inputs are unchanged; missing bodies are not searched elsewhere.
 *
 * @param {ProjectionSurface | null} surface
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @param {string | null | undefined} presentationKey
 */
function projectedAgentRadius(surface, sceneByKey, presentationKey) {
  const radius = finiteNumber(sceneByKey.get(presentationKey ?? "")?.radius);
  if (radius === null || !surface) {
    return 0;
  }
  const projected = surface.worldLengthToScreen(radius);
  return Number.isFinite(projected) ? Number(projected) : 0;
}

const CHARGE_DIRECTION_MARKER_CANDIDATES = Object.freeze([
  Object.freeze([1 / 5, 1 / 6]),
  Object.freeze([2 / 5, 1 / 3]),
  Object.freeze([3 / 5, 2 / 3]),
  Object.freeze([4 / 5, 5 / 6]),
]);

/**
 * Choose readable Charge direction markers on a presentation route.
 *
 * route is validated geometry. surface supplies viewport/protected regions;
 * protectedBodyKeysForCharge names only the two owned endpoint bodies, and
 * ownershipBounds is the optional Agent-ID label rectangle. Return a frozen array
 * of up to four progress fractions, each using its preferred fifth or paired fallback.
 * Omit candidates colliding with those protected areas or the inset viewport; no
 * viewport yields an empty array. These compact underlay markers do not reserve
 * foreground space. Invalid geometry/regions may propagate errors.
 *
 * @param {Record<string, any>} route
 * @param {ProjectionSurface | null} surface
 * @param {ReadonlyArray<string>} protectedBodyKeysForCharge
 * @param {unknown} ownershipBounds
 */
function chargeDirectionMarkerProgresses(
  route,
  surface,
  protectedBodyKeysForCharge,
  ownershipBounds,
) {
  const viewport = normalizedRectangle(surface?.viewportBounds);
  if (viewport === null) {
    return Object.freeze([]);
  }
  const bodyKeys = new Set(protectedBodyKeysForCharge);
  const protectedBounds = crossPhaseProtectedRects(surface)
    .filter(({ layoutKey }) => bodyKeys.has(layoutKey))
    .map(({ bounds }) => bounds);
  const ownership = normalizedRectangle(ownershipBounds);
  if (ownership !== null) {
    protectedBounds.push(ownership);
  }
  const padding = CHOREOGRAPHY_PAINT_FOOTPRINTS.route.activationCompactMarkerPadding;
  /**
   * Check one route progress against the local viewport and protected rectangles.
   *
   * progress is a candidate fraction. Return true only when its compact marker padding
   * clears the endpoint bodies, ownership label and viewport edge. Read enclosing
   * geometry without mutation; routeMarkerPose errors propagate.
   *
   * @param {number} progress
   */
  const isSafe = (progress) => {
    const marker = routeMarkerPose(/** @type {any} */ (route), progress);
    if (
      marker.x < viewport.left + padding ||
      marker.x > viewport.right - padding ||
      marker.y < viewport.top + padding ||
      marker.y > viewport.bottom - padding
    ) {
      return false;
    }
    return protectedBounds.every(
      (bounds) =>
        marker.x < bounds.left - padding ||
        marker.x > bounds.right + padding ||
        marker.y < bounds.top - padding ||
        marker.y > bounds.bottom + padding,
    );
  };
  return Object.freeze(
    CHARGE_DIRECTION_MARKER_CANDIDATES.flatMap((candidates) => {
      const progress = candidates.find(isSafe);
      return progress === undefined ? [] : [progress];
    }),
  );
}

/**
 * Allocate display space for enabled fragments without rewriting event anchors.
 *
 * events are filtered plan rows; surface supplies projection/viewport/protected
 * regions; sceneByKey supplies disclosed current radii. Return frozen copied rows
 * with allocated cue/route geometry, or null when required viewport geometry is absent.
 * Keep row order and identities. Activation routes remain direct underlays; direct
 * impact and life-state rings stay at their authorized body anchors. Foreground
 * life numbers and pulse callouts use the shared allocator; a life number's ring
 * radius is the recipient's projected body radius (14 px when unknown). Status
 * lifecycle rows and non-spatial rows (the death announcement and team respawn
 * wave, which are page chrome) make no requests. Filtered siblings make
 * no requests. Layout/region errors propagate; a missing internal binding throws
 * Error. The input scene, event rows and DOM are not modified.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 * @param {ProjectionSurface | null} surface
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @returns {ReadonlyArray<Record<string, any>> | null}
 */
function layoutCrossPhaseEvents(events, surface, sceneByKey) {
  /** @type {Record<string, any>[]} */
  const requests = [];
  /** @type {Map<string, Record<string, any>>} */
  const bindings = new Map();
  /** @type {Record<string, any>[]} */
  const patches = events.map(() => ({}));
  /** @type {Map<number, string>} */
  const activationUnderlayPairByEvent = new Map();
  /** @type {Map<number, ReadonlyArray<string>>} */
  const chargeProtectedBodyKeysByEvent = new Map();

  /**
   * Append one local allocation request and remember where its result belongs.
   *
   * eventIndex identifies the source row, role names its fragment, roleOrder breaks
   * local ties, and request carries allocator fields. binding defaults to an empty
   * object and adds result-join metadata. Mutate only the enclosing requests/bindings;
   * return undefined. The scientific event ID is retained on the original row.
   *
   * @param {number} eventIndex
   * @param {string} role
   * @param {number} roleOrder
   * @param {Record<string, any>} request
   * @param {Record<string, any>} [binding]
   */
  const add = (eventIndex, role, roleOrder, request, binding = {}) => {
    const layoutKey = crossPhaseLayoutKey(events[eventIndex].eventId, role);
    requests.push(
      Object.freeze({
        layoutKey,
        priority: request.kind === "route" ? 1 : 0,
        stableOrder: eventIndex * 4 + roleOrder,
        ...request,
      }),
    );
    bindings.set(layoutKey, Object.freeze({ eventIndex, role, ...binding }));
  };

  for (const [eventIndex, event] of events.entries()) {
    if (!event.spatial) continue;
    if (event.kind === "activation") {
      if (!event.target && event.source && event.paintParts?.ability === true) {
        const dimensions = activationCueDimensions(event);
        add(eventIndex, "source", 0, {
          kind: "perimeter_callout",
          anchor: event.source,
          recipientKey: event.sourcePresentationKey ?? event.eventId,
          allowProtectedKeys: protectedBodyKeys(surface, event.sourcePresentationKey),
          ...dimensions,
        });
      }
      if (
        event.paintParts?.ability === true &&
        event.source &&
        event.target &&
        event.targetPresentationKey
      ) {
        const sourceRadius = projectedAgentRadius(
          surface,
          sceneByKey,
          event.sourcePresentationKey,
        );
        const targetRadius = projectedAgentRadius(
          surface,
          sceneByKey,
          event.targetPresentationKey,
        );
        // Scene body regions describe current successor endpoints. Clip each
        // authorized endpoint independently: an Agent event can retain a
        // start-only source while its still-visible recipient uses the same
        // successor body boundary as Oracle. Historical Charge endpoints
        // remain transition-start anchors and never claim current bodies.
        const sourceSuccessorAnchored = event.sourceEndpointPhase === "successor";
        const targetSuccessorAnchored = event.targetEndpointPhase === "successor";
        const sourceProtectedKey = sourceSuccessorAnchored
          ? protectedBodyKey(surface, event.sourcePresentationKey)
          : null;
        const targetProtectedKey = targetSuccessorAnchored
          ? protectedBodyKey(surface, event.targetPresentationKey)
          : null;
        const pairKey = JSON.stringify([
          event.sourcePresentationKey,
          event.targetPresentationKey,
        ]);
        if (!surface?.viewportBounds) {
          return null;
        }
        // Accepted Basic/Ultimate trajectories are deliberately coincident
        // underlays. Their presence communicates the activation; their impact
        // meets this direct route endpoint while status and realized
        // displacement cues stay in the foreground allocator above.
        const routeLayoutKey = crossPhaseLayoutKey(event.eventId, "route");
        const route = Object.freeze({
          layoutKey: routeLayoutKey,
          priority: 1,
          stableOrder: eventIndex * 4 + 2,
          lane: 0,
          bridgeGaps: Object.freeze([]),
          ...createRouteGeometry(
            {
              eventId: event.eventId,
              source: event.source,
              target: event.target,
              sourceRadius: sourceProtectedKey === null ? 0 : sourceRadius,
              targetRadius: targetProtectedKey === null ? 0 : targetRadius,
              sourceEndpointGap: sourceProtectedKey === null ? 0 : 3,
              targetEndpointGap: targetProtectedKey === null ? 0 : 3,
            },
            {
              viewportBounds: surface.viewportBounds,
            },
          ),
        });
        Object.assign(patches[eventIndex], {
          route,
          routeLayoutKey,
          routeLane: 0,
          routeBridgeGaps: route.bridgeGaps,
        });
        activationUnderlayPairByEvent.set(eventIndex, pairKey);
        if (event.tokenId === "warrior_charge") {
          chargeProtectedBodyKeysByEvent.set(
            eventIndex,
            Object.freeze(
              [
                protectedBodyKey(surface, event.sourcePresentationKey),
                protectedBodyKey(surface, event.targetPresentationKey),
              ].filter((key) => key !== null),
            ),
          );
          const ownershipAnchor = Object.freeze({
            x: (event.source.x + event.target.x) / 2,
            y: (event.source.y + event.target.y) / 2,
          });
          patches[eventIndex].ownershipAnchor = ownershipAnchor;
          add(eventIndex, "ownership", 1, {
            kind: "perimeter_callout",
            anchor: ownershipAnchor,
            recipientKey: pairKey,
            width: CHOREOGRAPHY_PAINT_FOOTPRINTS.chargeOwnership.width,
            height: CHOREOGRAPHY_PAINT_FOOTPRINTS.chargeOwnership.height,
          });
        }
      }
      continue;
    }
    if (event.kind === "status_lifecycle") {
      // Status starts and ends draw only on-body marks and feed rows, so they
      // reserve no arena box that could push life numbers away.
      continue;
    }
    if (event.kind === "net_health" || event.kind === "regeneration") {
      const dimensions = paintedOutcomeCueDimensions(event);
      const recipientPresentationKey =
        event.kind === "regeneration"
          ? event.agentPresentationKey
          : event.recipientPresentationKey;
      // The life number sits just outside the body's Team Ring (store rule:
      // 12 o'clock first, 5 px above the ring), so the ring radius is the
      // projected body radius, not a fixed 60 px.
      const bodyRadius = projectedAgentRadius(
        surface,
        sceneByKey,
        recipientPresentationKey,
      );
      add(eventIndex, "cue", 0, {
        kind: "recipient_cue",
        anchor: event.recipient,
        anchorRadius: bodyRadius > 0 ? bodyRadius : RECIPIENT_CUE_FALLBACK_RADIUS_PX,
        recipientKey: recipientPresentationKey ?? event.eventId,
        allowProtectedKeys: protectedBodyKeys(surface, recipientPresentationKey),
        ...dimensions,
      });
      continue;
    }
    if (event.kind === "semantic_pulse") {
      if (
        event.cueSemantic === "agent_died" ||
        event.cueSemantic === "agent_respawned"
      ) {
        // Lifecycle rings valence their exact authorized body anchor. They are
        // not callouts and therefore never enter collision allocation.
        continue;
      }
      add(eventIndex, "cue", 0, {
        kind: "perimeter_callout",
        anchor: event.anchor,
        recipientKey:
          event.agentPresentationKey ??
          (event.teamId ? `team:${event.teamId}` : event.eventId),
        allowProtectedKeys: protectedBodyKeys(surface, event.agentPresentationKey),
        ...semanticPulseCueDimensions(event),
      });
    }
  }

  const activationUnderlayMultiplicity = new Map();
  for (const pairKey of activationUnderlayPairByEvent.values()) {
    activationUnderlayMultiplicity.set(
      pairKey,
      (activationUnderlayMultiplicity.get(pairKey) ?? 0) + 1,
    );
  }
  for (const [eventIndex, pairKey] of activationUnderlayPairByEvent) {
    patches[eventIndex].routeMultiplicity =
      activationUnderlayMultiplicity.get(pairKey) ?? 1;
  }

  if (requests.length === 0) {
    return Object.freeze(
      events.map((event, index) => Object.freeze({ ...event, ...patches[index] })),
    );
  }
  if (!surface?.viewportBounds) {
    return null;
  }
  const layout = layoutCrossPhaseOccupancy(
    /** @type {any} */ ({
      viewport: surface.viewportBounds,
      protectedRects: crossPhaseProtectedRects(surface),
      requests,
    }),
  );
  const routeMultiplicity = new Map();
  for (const binding of bindings.values()) {
    if (binding.role === "route") {
      routeMultiplicity.set(
        binding.pairKey,
        (routeMultiplicity.get(binding.pairKey) ?? 0) + 1,
      );
    }
  }
  const placements = /** @type {ReadonlyArray<Record<string, any>>} */ (
    layout.placements
  );
  for (const placement of placements) {
    const binding = bindings.get(placement.layoutKey);
    if (!binding) {
      throw new Error(`missing cross-phase binding ${placement.layoutKey}.`);
    }
    const patch = patches[binding.eventIndex];
    if (binding.role === "route") {
      Object.assign(patch, {
        route: placement,
        routeLayoutKey: placement.layoutKey,
        routeLane: placement.lane,
        routeBridgeGaps: placement.bridgeGaps,
        routeMultiplicity: routeMultiplicity.get(binding.pairKey) ?? 1,
      });
      continue;
    }
    const cuePatch = {
      [`${binding.role}Cue`]: placement.center,
      [`${binding.role}Bounds`]: placement.bounds,
      [`${binding.role}Leader`]: placement.leader,
      [`${binding.role}Disposition`]: placement.disposition,
      [`${binding.role}CueCollisionFree`]: placement.collisionFree,
      [`${binding.role}LayoutKey`]: placement.layoutKey,
    };
    if (binding.role === "cue") {
      Object.assign(patch, {
        cue: placement.center,
        cueBounds: placement.bounds,
        cueLeader: placement.leader,
        cueDisposition: placement.disposition,
        cueCollisionFree: placement.collisionFree,
        cueLayoutKey: placement.layoutKey,
        spatialDisposition: "rendered",
      });
    } else if (binding.role === "start" || binding.role === "end") {
      Object.assign(patch, {
        [`${binding.role}Cue`]: placement.center,
        [`${binding.role}CueBounds`]: placement.bounds,
        [`${binding.role}CueLeader`]: placement.leader,
        [`${binding.role}CueDisposition`]: placement.disposition,
        [`${binding.role}CueCollisionFree`]: placement.collisionFree,
        [`${binding.role}CueLayoutKey`]: placement.layoutKey,
      });
    } else if (binding.role === "ownership") {
      Object.assign(patch, cuePatch, {
        ownershipSpatialDisposition: "rendered",
      });
    } else {
      Object.assign(patch, cuePatch);
    }
  }
  for (const [
    eventIndex,
    protectedBodyKeysForCharge,
  ] of chargeProtectedBodyKeysByEvent) {
    const patch = patches[eventIndex];
    if (!patch.route) {
      continue;
    }
    patch.route = Object.freeze({
      ...patch.route,
      markerVariant: "compact",
      markerProgresses: chargeDirectionMarkerProgresses(
        patch.route,
        surface,
        protectedBodyKeysForCharge,
        patch.ownershipBounds,
      ),
    });
  }
  return Object.freeze(
    events.map((event, index) => Object.freeze({ ...event, ...patches[index] })),
  );
}

/**
 * Return an eight-digit unsigned hexadecimal hash of string value.
 *
 * This deterministic noncryptographic hash detects presentation content changes.
 * It is neither authentication nor a substitute for authorization checks.
 *
 * @param {string} value
 */
function hashText(value) {
  let hash = 2166136261;
  for (const character of value) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/**
 * Recognize submission keys in the existing browser command envelope.
 *
 * command must be a record with command_type keyboard and a string key. Return
 * true for space, enter or n after trimming/lowercasing (literal space is handled
 * separately). Other values return false. This does not decide simulator legality,
 * acceptance or whether a particular UI is currently allowed to submit.
 *
 * @param {unknown} command
 */
export function isSubmissionCommand(command) {
  const candidate = record(command);
  if (candidate?.command_type !== "keyboard") {
    return false;
  }
  if (typeof candidate.key !== "string") {
    return false;
  }
  const normalized =
    candidate.key === " " ? "space" : candidate.key.trim().toLowerCase();
  return normalized === "space" || normalized === "enter" || normalized === "n";
}

/**
 * Map an authored plan row to its explanation-time family, or null.
 *
 * event kind/semantic selects ability, health, recovery, cooldown, movement, death,
 * status, shield, wave or respawn. Feed-only/unknown kinds receive no family here;
 * no scientific event ordering or simulator behavior is computed.
 *
 * @param {Record<string, any>} event @returns {ChoreographyFamily | null}
 */
function choreographyFamily(event) {
  if (event.kind === "activation") return "ability";
  if (event.kind === "net_health") return "health";
  if (event.kind === "regeneration") return "recovery";
  if (event.kind === "movement_displacement") return "movement";
  if (event.kind === "status_lifecycle") {
    if (event.eventType === "agent_left_combat") return "recovery";
    return event.cueSemantic === "spawn_shield_expired" ? "shield" : "status";
  }
  if (event.kind !== "semantic_pulse") return null;
  if (
    event.cueSemantic === "cooldown_started" ||
    event.cueSemantic === "cooldown_ready"
  ) {
    return "cooldown";
  }
  if (event.cueSemantic === "agent_died") return "death";
  if (event.cueSemantic === "death_announcement") return "death";
  if (event.cueSemantic === "respawn_wave_occurred") return "respawn_wave";
  if (event.cueSemantic === "agent_respawned") return "respawn";
  return null;
}

/**
 * Resolve independently switchable paint pieces for one admitted event.
 *
 * event supplies the registered kind/component/semantic; visualFilters is the local
 * filter state. Return frozen piece booleans, or null for unregistered rows. Cooldown
 * start is always visually suppressed because accepted Ultimate activation already
 * shows it; its recorded truth remains available. Reapply/refresh status semantics
 * share the applied filter. A health change also carries the V5 add-on parts
 * registered by looks/health-text.js (savedBeat, lastBreath and
 * battleTextDetail; a rest heal only battleTextDetail), which refine a drawn
 * number and never keep a switched-off one drawn (paintPartsEnabled). No row
 * identity or information authority is changed.
 *
 * @param {Record<string, any>} event
 * @param {Record<string, boolean>} visualFilters
 * @returns {Readonly<Record<string, boolean>> | null}
 */
function authorizedPaintParts(event, visualFilters) {
  /**
   * Resolve one registered visual tag against this call's filter state.
   *
   * tag identifies surface/kind/component/part. Return the shared filter authority
   * result; this local reader neither changes filters nor adds event admission.
   *
   * @param {Record<string, string>} tag
   */
  const enabled = (tag) => isVisualPaintPartEnabled(visualFilters, tag);
  if (event.kind === "activation") {
    const ability = enabled({
      surface: "transient",
      kind: "activation",
      component: event.component,
      part: "ability",
    });
    const semantic =
      event.impactSemantic === "damage" || event.impactSemantic === "healing"
        ? enabled({
            surface: "transient",
            kind: "activation",
            component: event.component,
            part: "semantic",
          })
        : false;
    if (event.component === "basic") {
      /**
       * Resolve one optional Basic sub-part (Soft Glow, Close-Range Strokes,
       * Arcane Blast During Burst) registered by looks/basics.js.
       *
       * @param {string} part
       */
      const basicPart = (part) =>
        enabled({ surface: "transient", kind: "activation", component: "basic", part });
      return Object.freeze({
        ability,
        semantic,
        glow: basicPart("glow"),
        close_range: basicPart("close_range"),
        kindled: basicPart("kindled"),
      });
    }
    return Object.freeze({
      ability,
      semantic,
      softGlow: filterSwitch(visualFilters, "soft_glow", true),
    });
  }
  if (event.kind === "net_health") {
    const effect =
      event.outcome === "damage" || event.outcome === "healing"
        ? enabled({
            surface: "transient",
            kind: "net_health",
            outcome: event.outcome,
            part: "effect",
          })
        : false;
    /**
     * Resolve one health-change part registered by looks/health-text.js.
     *
     * @param {string} part
     */
    const healthPart = (part) =>
      enabled({
        surface: "transient",
        kind: "net_health",
        outcome: event.outcome,
        part,
      });
    return Object.freeze({
      effect,
      battleText: healthPart("battle_text"),
      recipientText: healthPart("recipient_text"),
      savedBeat: healthPart("saved_beat"),
      lastBreath: healthPart("last_breath"),
      battleTextDetail: healthPart("battle_text_detail"),
    });
  }
  if (event.kind === "regeneration") {
    return Object.freeze({
      effect: enabled({
        surface: "transient",
        kind: "regeneration",
        part: "effect",
      }),
      battleText: enabled({
        surface: "transient",
        kind: "regeneration",
        part: "battle_text",
      }),
      battleTextDetail: enabled({
        surface: "transient",
        kind: "regeneration",
        part: "battle_text_detail",
      }),
    });
  }
  if (event.kind === "status_lifecycle") {
    if (event.cueSemantic === "spawn_shield_expired") {
      return Object.freeze({
        effect: enabled({
          surface: "transient",
          kind: "spawn_shield_expiry",
        }),
      });
    }
    const visibleLifecycle = [
      "refreshed",
      "reapplied",
      "trap_broken_and_reapplied",
    ].includes(event.lifecycle)
      ? "applied"
      : event.lifecycle;
    return Object.freeze({
      effect: enabled({
        surface: "transient",
        kind: "status_lifecycle",
        lifecycle: visibleLifecycle,
        part: "effect",
      }),
    });
  }
  if (event.kind !== "semantic_pulse") {
    return null;
  }
  // Cooldown truth remains in the authorized event and durable badge. The
  // start pulse duplicates the accepted Ultimate activation, so it is always
  // presentation-suppressed while cooldown-ready retains its distinct cue.
  if (event.cueSemantic === "cooldown_started") {
    return Object.freeze({ effect: false });
  }
  /** @type {Record<string, string> | null} */
  const tag =
    event.cueSemantic === "cooldown_ready"
      ? { surface: "transient", kind: "cooldown", semantic: "ready" }
      : event.cueSemantic === "agent_died"
        ? { surface: "transient", kind: "death_effect" }
        : event.cueSemantic === "death_announcement"
          ? { surface: "transient", kind: "death_announcement" }
          : event.cueSemantic === "respawn_wave_occurred"
            ? { surface: "transient", kind: "respawn_wave" }
            : event.cueSemantic === "agent_respawned"
              ? { surface: "transient", kind: "resurrection_effect" }
              : null;
  return tag === null ? null : Object.freeze({ effect: enabled(tag) });
}

/**
 * Return paint parts and whether any piece is enabled, or null.
 *
 * event and visualFilters use authorizedPaintParts. This decision precedes spatial
 * projection and allocation, so disabled pieces need no geometry.
 *
 * @param {Record<string, any>} event
 * @param {Record<string, boolean>} visualFilters
 */
function authorizedPaintDecision(event, visualFilters) {
  const paintParts = authorizedPaintParts(event, visualFilters);
  return paintParts === null
    ? null
    : Object.freeze({
        paintParts,
        enabled: paintPartsEnabled(paintParts),
      });
}

/**
 * Return whether any main piece of an event's paint parts is on.
 *
 * paintParts is a resolved record of booleans. An activation's optional
 * sub-parts (a Basic's glow, close_range and kindled; an Ultimate's softGlow)
 * only refine its drawing, so they never keep an event drawn on their own:
 * only ability and semantic count for an activation. A health number's add-ons
 * (savedBeat, lastBreath, battleTextDetail) refine it the same way. Every other
 * key counts. Pure.
 *
 * @param {Readonly<Record<string, unknown>>} paintParts
 */
function paintPartsEnabled(paintParts) {
  return Object.entries(paintParts).some(
    ([key, part]) => part === true && !ACTIVATION_SUB_PARTS.has(key),
  );
}

/** Optional sub-parts that refine a drawn activation or health number (see paintPartsEnabled). */
const ACTIVATION_SUB_PARTS = new Set([
  "glow",
  "close_range",
  "kindled",
  "softGlow",
  "savedBeat",
  "lastBreath",
  "battleTextDetail",
]);

/**
 * Mark fully disabled events non-spatial while retaining inspection identity.
 *
 * events remain in input order; visualFilters resolves rows without existing paintParts.
 * Return a new array, reusing unregistered rows and freezing changed row copies.
 * Any exact true main piece (see paintPartsEnabled) keeps spatial eligibility; no
 * filter can make a previously non-spatial row spatial. Inputs and scientific identities are unchanged.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 * @param {Record<string, boolean>} visualFilters
 */
function applyAuthorizedVisualFilters(events, visualFilters) {
  return events.map((event) => {
    const paintParts = event.paintParts ?? authorizedPaintParts(event, visualFilters);
    if (paintParts === null) {
      return event;
    }
    const enabled = paintPartsEnabled(paintParts);
    return Object.freeze({
      ...event,
      paintParts,
      presentationSuppressed: !enabled,
      spatial: Boolean(event.spatial && enabled),
    });
  });
}

/**
 * Give every plan row its window on the beat clock.
 *
 * events supplies already admitted and filtered rows. schedule is the tick's
 * beatSchedule (built from the audience's own falls and returns). Return frozen
 * phases and scheduled event copies in the same order; times are real ms at
 * playback rate 1 in normal motion, not simulator ticks.
 *
 * Windows (see FAMILY_BEAT_WINDOW): Basic and Ultimate activations play in the
 * blows window, blow time 80-760 ms with impact at 360 ms (0-760 ms without Body
 * Glide), so nothing strikes while bodies still glide. Every outcome shares the
 * Settle window (blow time 420-900 ms) instead of queueing family after family. A
 * death cue starts at its own victim's Fall beat time and ends with the Fall beat
 * (480 ms later without a beat). A Death Announcer card starts at the first fall and
 * keeps 1500 ms. A status cleared by a death waits for that victim's fall. A
 * respawn wave or a return plays in the Return beat (or a 620 ms window at blow time
 * 880 ms without one). Ordinary movement has no window. Derived rows (Home Fall,
 * heralds, Fallen feed, Last Seen, Focus Tally) are not scheduled here: they live in
 * plan.derivedEvents with the times set when they were built.
 *
 * phases keeps the painter's field names: activationStart, travelStart,
 * impactStart, outcomeStart, settleStart and the v2...Start fields point at the
 * windows above; submissionRelease stays 450 ms (the live release never waits for a
 * glide); total is the latest window end of a drawn row (0 when nothing is drawn);
 * reducedTotal is total times schedule.reducedScale, so the painter's linear
 * reduced-motion scaling gives the short schedule. These windows explain recorded
 * facts; they never recompute them.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 * @param {BeatSchedule} schedule
 */
function scheduleChoreography(events, schedule) {
  /**
   * Convert blow time part (ms) to real ms for schedule's glide and lead. Pure.
   *
   * @param {number} part
   */
  const real = (part) => beatRealTime(part, schedule.glide, schedule.lead);
  const abilityWindow = Object.freeze({
    start: schedule.glide ? real(80) : 0,
    impact: real(CHOREOGRAPHY_PHASES.impactStart),
    end: real(CHOREOGRAPHY_PHASES.settleStart),
  });
  const settle = schedule.windows.settle;
  const firstFallAt = schedule.falls[0]?.at ?? schedule.choreographyEnd;
  const returnWindow = schedule.returnBeat ?? [
    real(BEAT_CLOCK.returnWithoutBeat),
    real(BEAT_CLOCK.returnWithoutBeat) + RETURN_WITHOUT_BEAT_DWELL_MS,
  ];
  /**
   * Return the end of a death window that starts at at.
   *
   * at is real ms. With a Fall beat the window ends with the beat; without one it
   * keeps the readable 480 ms dwell. Pure.
   *
   * @param {number} at
   */
  const deathEnd = (at) =>
    schedule.fall ? Math.max(schedule.fall[1], at) : at + READABLE_OUTCOME_DWELL_MS;
  /**
   * Return the beat window for one row, or null for rows without a family.
   *
   * event is one plan row. Read only its kind, cue semantic, lifecycle and
   * presentation keys; return {start, end, impact?} in real ms.
   *
   * @param {Record<string, any>} event
   * @returns {{start: number, end: number, impact?: number} | null}
   */
  const windowFor = (event) => {
    const family = choreographyFamily(event);
    if (family === null) return null;
    const beat = FAMILY_BEAT_WINDOW[family];
    if (beat === "blows") return abilityWindow;
    // Life numbers pop at Impact (blow 260 ms), the moment life changes.
    if (event.kind === "net_health" || event.kind === "regeneration") {
      return { start: real(260), end: settle[1] };
    }
    if (beat === "none") return { start: settle[0], end: settle[0] };
    if (beat === "settle") {
      if (event.lifecycle === "cleared_by_death") {
        const at = fallTimeFor(schedule, event.recipientPresentationKey);
        if (at !== null) return { start: at, end: deathEnd(at) };
      }
      return { start: settle[0], end: settle[1] };
    }
    if (beat === "fall") {
      if (event.cueSemantic === "death_announcement") {
        return { start: firstFallAt, end: firstFallAt + DEATH_ANNOUNCEMENT_DWELL_MS };
      }
      const at =
        fallTimeFor(schedule, event.agentPresentationKey) ?? schedule.choreographyEnd;
      return { start: at, end: deathEnd(at) };
    }
    return { start: returnWindow[0], end: returnWindow[1] };
  };
  /** @type {Record<string, any>[]} */
  const scheduledEvents = events.map((event) => {
    const window = windowFor(event);
    if (window === null) return event;
    return /** @type {Record<string, any>} */ (
      Object.freeze({
        ...event,
        phaseStart: window.start,
        phaseImpact:
          window.impact === undefined
            ? event.phaseImpact
            : Math.min(window.impact, window.end),
        phaseEnd: window.end,
      })
    );
  });
  const total = scheduledEvents.reduce(
    (latest, event) =>
      event.spatial && Number.isFinite(event.phaseEnd)
        ? Math.max(latest, Number(event.phaseEnd))
        : latest,
    0,
  );
  const phases = Object.freeze({
    ...CHOREOGRAPHY_PHASES,
    activationStart: abilityWindow.start,
    travelStart: real(CHOREOGRAPHY_PHASES.travelStart),
    impactStart: abilityWindow.impact,
    outcomeStart: settle[0],
    submissionRelease: CHOREOGRAPHY_PHASES.submissionRelease,
    settleStart: abilityWindow.end,
    v2AbilityStart: abilityWindow.start,
    v2HealthResolutionStart: settle[0],
    v2CountdownAndRegenStart: settle[0],
    v2CooldownStart: settle[0],
    v2ChargeStart: settle[0],
    v2MovementStart: settle[0],
    v2DeathStart: firstFallAt,
    v2StatusStart: settle[0],
    v2ShieldStart: settle[0],
    v2RespawnWaveStart: returnWindow[0],
    v2RespawnStart: returnWindow[0],
    povSuccessorObservationStart: settle[0],
    total,
    reducedTotal: total * schedule.reducedScale,
  });
  return Object.freeze({ phases, events: Object.freeze(scheduledEvents) });
}

/**
 * Read a finite position tuple directly from a serialized anchor.
 *
 * rawAnchor may be unknown; return a frozen coordinate pair or null. Identity and
 * phase agreement are checked separately before this point can drive a cue.
 *
 * @param {unknown} rawAnchor
 */
function authorizedWorldPoint(rawAnchor) {
  const anchor = record(rawAnchor);
  return anchor ? point(anchor.position) : null;
}

/**
 * Project the position already supplied by an authorized anchor.
 *
 * rawAnchor supplies a finite tuple and surface supplies projection. Return frozen
 * screen coordinates or null; this helper never substitutes another scene position.
 *
 * @param {unknown} rawAnchor @param {ProjectionSurface | null} surface
 */
function authorizedAnchor(rawAnchor, surface) {
  return project(authorizedWorldPoint(rawAnchor), surface);
}

/**
 * Validate a successor-phase team-wave anchor without creating geometry.
 *
 * rawAnchor must give team_index 0/1 and matching team_id 1/2. teamSide maps a team
 * ID to its home side for this audience (audienceTeamSide: the match summary in
 * Oracle View, the authorized spawn pads in Agent POV). Return frozen team identity,
 * side and label, or null. No body location is inferred from team identity.
 *
 * @param {unknown} rawAnchor
 * @param {(teamId: number) => "left" | "right"} teamSide
 */
function authorizedTeamWaveIdentity(rawAnchor, teamSide) {
  const teamAnchor = record(rawAnchor);
  const teamIndex = integer(teamAnchor?.team_index);
  const teamId = integer(teamAnchor?.team_id);
  if (
    teamAnchor?.phase !== "successor" ||
    (teamIndex !== 0 && teamIndex !== 1) ||
    teamId !== teamIndex + 1
  ) {
    return null;
  }
  return Object.freeze({
    teamIndex,
    teamId,
    teamSide: teamSide(teamId),
    label: `EVENT: Team ${teamIndex === 0 ? "A" : "B"} Respawn`,
  });
}

/**
 * Return true when both values are exact finite coordinate pairs that match.
 *
 * left/right are checked with point. No rounding tolerance or scene-position
 * substitution is allowed in this phase-anchor coherence comparison.
 *
 * @param {unknown} left
 * @param {unknown} right
 */
function sameAuthorizedPoint(left, right) {
  const leftPoint = point(left);
  const rightPoint = point(right);
  return (
    leftPoint !== null &&
    rightPoint !== null &&
    leftPoint[0] === rightPoint[0] &&
    leftPoint[1] === rightPoint[1]
  );
}

/**
 * Validate serialized phase anchors against their disclosed body identities.
 *
 * latest contains an Oracle or fog-filtered Agent visual inventory. sceneByKey holds
 * current authorized bodies; actorInputSuccessorKeys is the exact Agent input-body
 * set (required for Agent), and deathOverlaySuccessorKeys adds disclosed death bodies.
 * Oracle trajectories keep start/post-Charge/successor anchors for every body. Agent
 * trajectories omit post-Charge and may disclose only one adjacent endpoint; a
 * start-only body must not appear in the current scene. Every successor must join
 * the matching current public ID/class/position and exact allowed key set.
 * Return a new key-to-existing-trajectory Map or null on inconsistency. The scene is
 * a coherence check; serialized trajectory anchors remain the geometric authority.
 *
 * @param {Record<string, any>} latest
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @param {Set<string> | null} actorInputSuccessorKeys
 * @param {Set<string>} deathOverlaySuccessorKeys
 * @returns {Map<string, Record<string, any>> | null}
 */
function authorizedTrajectoryMap(
  latest,
  sceneByKey,
  actorInputSuccessorKeys,
  deathOverlaySuccessorKeys,
) {
  if (!Array.isArray(latest.agent_phase_trajectories)) {
    return null;
  }
  const trajectories = latest.agent_phase_trajectories;
  const agentVisual = latest.summary_kind === "agent_pov_fog_filtered_visual_events";
  if (!agentVisual && latest.summary_kind !== "replay_incoming_inventory") {
    return null;
  }
  if (!agentVisual && trajectories.length !== sceneByKey.size) {
    return null;
  }
  const expectedAgentSuccessorKeys =
    agentVisual && actorInputSuccessorKeys instanceof Set
      ? new Set([...actorInputSuccessorKeys, ...deathOverlaySuccessorKeys])
      : null;
  if (agentVisual && expectedAgentSuccessorKeys === null) {
    return null;
  }
  const sceneByPublicId = new Map(
    [...sceneByKey.values()].map((agent) => [identifier(agent.public_agent_id), agent]),
  );
  /** @type {Map<string, Record<string, any>>} */
  const byKey = new Map();
  const publicIds = new Set();
  const successorKeys = new Set();
  for (const candidate of trajectories) {
    const trajectory = record(candidate);
    const key = identifier(trajectory?.agent_presentation_key);
    const publicId = identifier(trajectory?.agent_public_agent_id);
    const agentClassId = integer(trajectory?.agent_class_id);
    if (
      !trajectory ||
      key === null ||
      publicId === null ||
      byKey.has(key) ||
      publicIds.has(publicId)
    ) {
      return null;
    }
    const sceneAgent = sceneByKey.get(key);
    if (agentVisual && Object.hasOwn(trajectory, "post_charge")) {
      return null;
    }
    if (
      agentVisual &&
      (agentClassId === null || agentClassId < 1 || agentClassId > 5)
    ) {
      return null;
    }
    const phases = agentVisual
      ? ["transition_start", "successor"]
      : ["transition_start", "post_charge", "successor"];
    for (const phase of phases) {
      const anchor = record(trajectory[phase]);
      if (agentVisual && trajectory[phase] == null) {
        continue;
      }
      if (
        !anchor ||
        anchor.phase !== phase ||
        identifier(anchor.presentation_key) !== key ||
        identifier(anchor.public_agent_id) !== publicId ||
        point(anchor.position) === null
      ) {
        return null;
      }
    }
    if (agentVisual) {
      if (trajectory.transition_start == null && trajectory.successor == null) {
        return null;
      }
      const publicSceneAgent = sceneByPublicId.get(publicId);
      if (trajectory.successor == null) {
        if (sceneAgent || publicSceneAgent) {
          return null;
        }
      } else {
        if (
          !sceneAgent ||
          sceneAgent !== publicSceneAgent ||
          identifier(sceneAgent.public_agent_id) !== publicId ||
          integer(sceneAgent.class_id) !== agentClassId ||
          !sameAuthorizedPoint(trajectory.successor.position, sceneAgent.position)
        ) {
          return null;
        }
        successorKeys.add(key);
      }
    } else if (
      !sceneAgent ||
      identifier(sceneAgent.public_agent_id) !== publicId ||
      !sameAuthorizedPoint(trajectory.successor.position, sceneAgent.position)
    ) {
      return null;
    }
    byKey.set(key, trajectory);
    publicIds.add(publicId);
  }
  if (
    agentVisual &&
    (successorKeys.size !== expectedAgentSuccessorKeys?.size ||
      [...successorKeys].some((key) => !expectedAgentSuccessorKeys.has(key)))
  ) {
    return null;
  }
  return byKey;
}

/**
 * Join an event anchor to the exact serialized trajectory phase.
 *
 * trajectories is already validated; rawAnchor may be unknown; phase is
 * transition_start, post_charge or successor. Require matching key, public ID,
 * phase and position. Return the existing trajectory or null. A key match alone
 * cannot authorize a point or substitute a different decision epoch.
 *
 * @param {Map<string, Record<string, any>>} trajectories
 * @param {unknown} rawAnchor
 * @param {"transition_start" | "post_charge" | "successor"} phase
 * @returns {Record<string, any> | null}
 */
function trajectoryForAuthorizedAnchor(trajectories, rawAnchor, phase) {
  const anchor = record(rawAnchor);
  const key = identifier(anchor?.presentation_key);
  const publicId = identifier(anchor?.public_agent_id);
  const trajectory = key === null ? null : (trajectories.get(key) ?? null);
  const trajectoryAnchor = record(trajectory?.[phase]);
  if (
    !anchor ||
    key === null ||
    publicId === null ||
    anchor.phase !== phase ||
    !trajectory ||
    identifier(trajectory.agent_public_agent_id) !== publicId ||
    !trajectoryAnchor ||
    !sameAuthorizedPoint(anchor.position, trajectoryAnchor.position)
  ) {
    return null;
  }
  return trajectory;
}

/**
 * @typedef {{
 *   row: Readonly<Record<string, any>>,
 *   event: Readonly<Record<string, any>>,
 *   applicationSource: Readonly<Record<string, any>> | null,
 * }} AuthorizedStatusAtom
 * @typedef {{
 *   atoms: AuthorizedStatusAtom[],
 *   firstIndex: number,
 *   recipientAnchor: Readonly<Record<string, any>>,
 *   recipientIdentity: Readonly<Record<string, any>>,
 *   recipientPresentationKey: string,
 *   recipientPublicAgentId: string,
 * }} AuthorizedStatusGroup
 */

/**
 * Choose one visible status lifecycle from a validated group of atomics.
 *
 * group.atoms retains canonical recorded events. Return frozen applications,
 * lifecycleId and primary, or null for no recognized atomic. Death-clear wins;
 * apply/refresh share the applied cue; otherwise break precedes expiry. Atomic IDs
 * remain available separately, so the simplified display does not erase event facts.
 *
 * @param {AuthorizedStatusGroup} group
 */
function authorizedStatusSelection(group) {
  /**
   * Find the first atom of kind in the enclosing validated status group.
   *
   * Return the existing atom or null; preserve canonical input order without mutation.
   *
   * @param {string} kind
   */
  const first = (kind) => group.atoms.find(({ row }) => row.kind === kind) ?? null;
  const cleared = first("status_cleared_by_new_death");
  const refreshed = first("status_refreshed_or_extended");
  const broken = first("status_broken_by_damage");
  const aged = first("status_aged_to_zero");
  const applications = group.atoms.filter(({ row }) => row.kind === "status_applied");
  const applied = applications.at(0) ?? null;
  // The canonical atomics retain apply, refresh, expiry, and break identity.
  // The shared interactive renderer deliberately presents every transition
  // that leaves the status active with the one ordinary application cue.
  const lifecycleId = cleared
    ? "cleared_by_death"
    : applied || refreshed
      ? "applied"
      : broken
        ? "trap_broken"
        : "expired";
  const primary =
    cleared ??
    (applied && (broken || aged) ? applied : null) ??
    broken ??
    aged ??
    applied ??
    refreshed;
  return primary === null
    ? null
    : Object.freeze({
        applications: Object.freeze(applications),
        lifecycleId,
        primary,
      });
}

/**
 * Compose one visible status row while retaining every atomic event identity.
 *
 * rows is the ordered normalized inventory and transitionId names its transition.
 * sceneByKey supplies local bodies; researcherAgentByPublicId may provide display
 * attribution only when useResearcherStatusAttribution is true and the exact local
 * status/source join succeeds. trajectories supplies validated phase anchors,
 * surface projects enabled cues, and visualFilters decides their paint pieces.
 *
 * Return a Map from every atomic ID to its shared frozen composition; no status
 * rows gives an empty Map. Invalid/duplicate event identities, missing anchor joins,
 * contradictory source facts or missing enabled projection return null. Filtered
 * groups retain metadata but get no lane or geometry. Inputs and normalized Latest
 * Events remain unchanged; source attribution never admits an event or hidden point.
 *
 * @param {ReadonlyArray<Readonly<{
 *   id: string,
 *   kind: string,
 *   vocabulary: "event" | "recipient_cue" | "observation_delta",
 *   payload: Readonly<Record<string, any>>,
 * }>>} rows
 * @param {string} transitionId
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @param {Map<string, Record<string, any>>} researcherAgentByPublicId
 * @param {boolean} useResearcherStatusAttribution
 * @param {Map<string, Record<string, any>> | null} trajectories
 * @param {ProjectionSurface | null} surface
 * @param {Record<string, boolean>} visualFilters
 * @returns {Map<string, Readonly<Record<string, any>>> | null}
 */
function authorizedStatusCompositions(
  rows,
  transitionId,
  sceneByKey,
  researcherAgentByPublicId,
  useResearcherStatusAttribution,
  trajectories,
  surface,
  visualFilters,
) {
  /** @type {Map<string, AuthorizedStatusGroup>} */
  const groups = new Map();
  const seenEventIds = new Set();
  let statusRowCount = 0;
  for (const [index, row] of rows.entries()) {
    if (!STATUS_EVENT_TYPES.has(row.kind)) {
      continue;
    }
    statusRowCount += 1;
    const event = record(row.payload);
    const eventId = identifier(event?.event_id);
    const eventKind = identifier(event?.event_kind);
    const recipientAnchor = record(event?.recipient_anchor);
    const recipientKey = identifier(recipientAnchor?.presentation_key);
    const statusChannel = integer(event?.status_channel);
    const statusId = identifier(event?.status_id);
    if (
      trajectories === null ||
      row.vocabulary !== "event" ||
      !event ||
      eventId !== row.id ||
      eventKind !== row.kind ||
      seenEventIds.has(row.id) ||
      recipientKey === null ||
      statusChannel === null ||
      statusId === null
    ) {
      return null;
    }
    const recipientTrajectory = trajectoryForAuthorizedAnchor(
      trajectories,
      recipientAnchor,
      "successor",
    );
    const recipientAgent = sceneByKey.get(recipientKey);
    const recipientIdentity = authorizedIdentitySnapshot(recipientAgent);
    const recipientWorld = authorizedWorldPoint(recipientTrajectory?.successor);
    if (
      recipientTrajectory === null ||
      !recipientAgent ||
      identifier(recipientAgent.public_agent_id) !==
        identifier(recipientTrajectory.agent_public_agent_id) ||
      recipientWorld === null ||
      recipientIdentity === null
    ) {
      return null;
    }
    /** @type {Readonly<Record<string, any>> | null} */
    let applicationSource = null;
    if (row.kind === "status_applied") {
      const sourceAnchor = record(event.source_anchor);
      const sourceKey = identifier(sourceAnchor?.presentation_key);
      const sourceTrajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        sourceAnchor,
        "successor",
      );
      const sourceAgent = sourceKey === null ? null : sceneByKey.get(sourceKey);
      if (
        sourceKey === null ||
        sourceTrajectory === null ||
        !sourceAgent ||
        identifier(sourceAgent.public_agent_id) !==
          identifier(sourceTrajectory.agent_public_agent_id)
      ) {
        return null;
      }
      applicationSource = Object.freeze({
        eventId: row.id,
        sourcePresentationKey: sourceKey,
        sourcePublicAgentId: sourceTrajectory.agent_public_agent_id,
        sourceIdentity: authorizedIdentitySnapshot(sourceAgent),
      });
      if (applicationSource.sourceIdentity === null) {
        return null;
      }
    }
    seenEventIds.add(row.id);
    const key = JSON.stringify([transitionId, recipientKey, statusChannel, statusId]);
    /** @type {AuthorizedStatusGroup} */
    const group = groups.get(key) ?? {
      atoms: [],
      firstIndex: index,
      recipientAnchor: recipientTrajectory.successor,
      recipientIdentity,
      recipientPresentationKey: recipientKey,
      recipientPublicAgentId: recipientTrajectory.agent_public_agent_id,
    };
    if (
      group.recipientPresentationKey !== recipientKey ||
      group.recipientPublicAgentId !== recipientTrajectory.agent_public_agent_id ||
      !sameAuthorizedPoint(
        group.recipientAnchor.position,
        recipientTrajectory.successor.position,
      )
    ) {
      return null;
    }
    group.atoms.push(
      Object.freeze({
        row,
        event,
        applicationSource,
      }),
    );
    groups.set(key, group);
  }
  if (statusRowCount === 0) {
    return new Map();
  }

  /** @type {Map<string, Readonly<Record<string, any>>>} */
  const presentationByKey = new Map();
  for (const [key, group] of groups) {
    const selection = authorizedStatusSelection(group);
    if (selection === null) {
      return null;
    }
    const paintParts = authorizedPaintParts(
      {
        kind: "status_lifecycle",
        lifecycle: selection.lifecycleId,
      },
      visualFilters,
    );
    if (paintParts === null) {
      return null;
    }
    presentationByKey.set(
      key,
      Object.freeze({
        ...selection,
        paintParts,
        enabled: Object.values(paintParts).some((part) => part === true),
      }),
    );
  }

  /** @type {Map<string, {lane: number, laneCount: number}>} */
  const lanes = new Map();
  /** @type {Map<string, Array<[string, any]>>} */
  const groupsByRecipient = new Map();
  for (const [key, group] of groups) {
    if (presentationByKey.get(key)?.enabled !== true) {
      continue;
    }
    const recipientGroups = groupsByRecipient.get(group.recipientPresentationKey) ?? [];
    recipientGroups.push([key, group]);
    groupsByRecipient.set(group.recipientPresentationKey, recipientGroups);
  }
  for (const recipientGroups of groupsByRecipient.values()) {
    recipientGroups.sort((left, right) => left[1].firstIndex - right[1].firstIndex);
    recipientGroups.forEach(([key], lane) => {
      lanes.set(key, { lane, laneCount: recipientGroups.length });
    });
  }

  /** @type {Map<string, Readonly<Record<string, any>>>} */
  const compositions = new Map();
  for (const [key, group] of groups) {
    const presentation = presentationByKey.get(key);
    if (!presentation) {
      return null;
    }
    const applications = /** @type {ReadonlyArray<AuthorizedStatusAtom>} */ (
      presentation.applications
    );
    const lifecycleId = /** @type {string} */ (presentation.lifecycleId);
    const primary = /** @type {AuthorizedStatusAtom} */ (presentation.primary);
    const paintParts = /** @type {Readonly<Record<string, boolean>>} */ (
      presentation.paintParts
    );
    const enabled = presentation.enabled === true;
    const atomicEventIds = Object.freeze(group.atoms.map(({ row }) => row.id));
    const applicationEventIds = Object.freeze(applications.map(({ row }) => row.id));
    let applicationSources = Object.freeze(
      applications.map(({ applicationSource }) => applicationSource),
    );
    if (applicationSources.some((source) => source === null)) {
      return null;
    }
    if (useResearcherStatusAttribution && lifecycleId === "applied") {
      const statusChannel = integer(primary.event.status_channel);
      const statusId = identifier(primary.event.status_id);
      if (statusChannel === null || statusId === null) return null;
      applicationSources = researcherApplicationSources(
        /** @type {ReadonlyArray<Readonly<Record<string, any>>>} */ (
          applicationSources
        ),
        group,
        statusChannel,
        statusId,
        sceneByKey,
        researcherAgentByPublicId,
      );
    }
    const lane = lanes.get(key) ?? { lane: 0, laneCount: 0 };
    const status = resolveVisualToken("status", primary.event.status_id, primary.event);
    const lifecycle = resolveVisualToken("lifecycle", lifecycleId, primary.event);
    const directApplicationSource = primary.applicationSource;
    const recipient = enabled ? authorizedAnchor(group.recipientAnchor, surface) : null;
    if (enabled && recipient === null) {
      return null;
    }
    const composition = Object.freeze({
      eventId: primary.row.id,
      eventType: primary.row.kind,
      transitionId,
      authorityVocabulary: primary.row.vocabulary,
      kind: "status_lifecycle",
      tokenId: status.tokenId,
      token: status,
      lifecycle: lifecycle.tokenId,
      lifecycleToken: lifecycle,
      recipientPresentationKey: group.recipientPresentationKey,
      recipientPublicAgentId: group.recipientPublicAgentId,
      recipientIdentity: group.recipientIdentity,
      recipient,
      sourcePresentationKey: directApplicationSource?.sourcePresentationKey ?? null,
      sourcePublicAgentId: directApplicationSource?.sourcePublicAgentId ?? null,
      applicationSources,
      durationBefore: null,
      durationAfter: null,
      lane: enabled ? lane.lane : null,
      laneCount: enabled ? lane.laneCount : 0,
      statusLayoutOrder: group.firstIndex,
      atomicEventIds,
      applicationEventIds,
      paintParts,
      presentationSuppressed: !enabled,
      spatial: enabled,
      phaseStart: CHOREOGRAPHY_PHASES.v2StatusStart,
      phaseEnd: CHOREOGRAPHY_PHASES.v2ShieldStart,
    });
    for (const atom of group.atoms) {
      compositions.set(atom.row.id, composition);
    }
  }
  return compositions;
}

/**
 * List the champions named by one kind of beat event in the audience's own rows.
 *
 * rows are this audience's incoming rows (authorizedPresentationIncomingRows: Oracle
 * latest_events, Agent POV visual_events plus the recipient's own cues). kind is
 * "agent_died" (read recipient_anchor, or agent_anchor) or "agent_respawned" (read
 * agent_anchor). Return frozen {presentationKey, publicAgentId} entries, one per
 * champion, in row order; rows without a usable anchor are skipped here (the main
 * loop rejects them later). Pure; reads no researcher data.
 *
 * @param {ReadonlyArray<Readonly<{kind: string, vocabulary: string, payload: any}>>} rows
 * @param {"agent_died" | "agent_respawned"} kind
 * @returns {ReadonlyArray<BeatSubject>}
 */
function beatSubjects(rows, kind) {
  /** @type {BeatSubject[]} */
  const subjects = [];
  const seen = new Set();
  for (const row of rows) {
    if (row.kind !== kind || row.vocabulary !== "event") continue;
    const payload = record(row.payload);
    const anchor =
      kind === "agent_died"
        ? (record(payload?.recipient_anchor) ?? record(payload?.agent_anchor))
        : record(payload?.agent_anchor);
    const presentationKey = identifier(anchor?.presentation_key);
    const publicAgentId = identifier(anchor?.public_agent_id);
    if (
      presentationKey === null ||
      publicAgentId === null ||
      seen.has(presentationKey)
    ) {
      continue;
    }
    seen.add(presentationKey);
    subjects.push(Object.freeze({ presentationKey, publicAgentId }));
  }
  return Object.freeze(subjects);
}

/**
 * Return the screen side ("left" or "right") where a team's home is drawn.
 *
 * Oracle View keeps today's owner: match_summary.teams display_side
 * (matchTeamSide). Agent POV never reads the match summary for arena drawing, so it
 * compares the mean x of each team's spawn pads in the authorized scene (pads lie on
 * their team's home side); without usable pads Team A is left. teamId is 1 or 2.
 * Pure.
 *
 * @param {Record<string, any>} presentation
 * @param {Record<string, any>} scene
 * @param {number} teamId
 * @param {boolean} agentVisual
 * @returns {"left" | "right"}
 */
function audienceTeamSide(presentation, scene, teamId, agentVisual) {
  if (!agentVisual) {
    return matchTeamSide(presentation.match_summary, teamId) === "right"
      ? "right"
      : "left";
  }
  /**
   * Return the mean world x of team's spawn pads in the authorized scene, or null
   * when the team has no pad with a usable position. Pure.
   *
   * @param {number} team
   */
  const meanX = (team) => {
    const xs = array(scene.spawn_pads)
      .filter((pad) => integer(record(pad)?.team_id) === team)
      .map((pad) => point(record(pad)?.position)?.[0])
      .filter((x) => typeof x === "number");
    return xs.length === 0 ? null : xs.reduce((sum, x) => sum + x, 0) / xs.length;
  };
  const own = meanX(teamId);
  const other = meanX(teamId === 1 ? 2 : 1);
  if (own === null || other === null || own === other) {
    return teamId === 1 ? "left" : "right";
  }
  return own < other ? "left" : "right";
}

/**
 * Return whether one optional derived family is switched on, treating an unregistered
 * paint part as off.
 *
 * visualFilters is the page's filter state; family names a DERIVED_EVENT_PAINT_PARTS
 * entry. When the look module that draws the family has not registered its tag yet,
 * visual-filters.js throws RangeError; this returns false so the family stays data
 * only and is never drawn by accident. Other errors (a malformed state) propagate.
 *
 * @param {Record<string, boolean>} visualFilters
 * @param {keyof typeof DERIVED_EVENT_PAINT_PARTS} family
 */
function derivedFamilyEnabled(visualFilters, family) {
  try {
    return isVisualPaintPartEnabled(
      visualFilters,
      DERIVED_EVENT_PAINT_PARTS[family].part,
    );
  } catch (error) {
    if (error instanceof RangeError) return false;
    throw error;
  }
}

/**
 * @typedef {Readonly<{
 *   eventId: string,
 *   presentationKey: string,
 *   publicAgentId: string,
 *   victim: Readonly<Record<string, any>>,
 *   killers: ReadonlyArray<Readonly<Record<string, any>>>,
 *   killingTeamId: number | null,
 *   homeFall: boolean | null,
 *   points: number | null,
 *   ultimate: Readonly<{classId: number, publicAgentId: string}> | null,
 *   at: number,
 *   world: readonly [number, number] | null,
 * }>} AuthorizedFall
 */

/**
 * Name the falls this audience may show, with only the killers it may name.
 *
 * input.rows are the audience's own rows; input.sceneByKey its authorized bodies
 * (Agent POV: seen bodies plus local corpses); input.trajectories the joined phase
 * anchors (null when the inventory has none); input.schedule the beat schedule;
 * input.agentVisual is true for Agent POV; input.presentation is the frame (Oracle
 * only reads match_summary.deaths as a fallback for killers).
 *
 * Rules:
 * - A victim is named only from an agent_died event in the audience's own rows
 *   whose body is in the authorized scene (a seen body or an overlay corpse).
 * - Oracle View: killers are the sources of lethal_damage_contribution rows for the
 *   victim; if there are none, the match summary's credited contributors.
 * - Agent POV: a killer is named only when the team saw its blow land: a damaging
 *   Basic or Ultimate activation (ability_activated) on the victim whose source was
 *   seen at the start of the tick. This includes the Eyes Closed rule: a source that
 *   left sight this tick (no successor) is still named, marked lastSeen, so its
 *   killing blow is drawn from a Last Seen ring. A killer's team is its scene team,
 *   or the victim's opposing team for a source that left sight (blows only damage
 *   enemies). match_summary.deaths is never read.
 * - homeFall is the host's home_fall flag on the agent_died event (true, false, or
 *   null when the recording has no flag); points is 2, 1 or null to match.
 * - ultimate names a damaging Ultimate (Charge, Freezing Trap or Crippling Poison)
 *   that struck the victim this tick, as {classId, publicAgentId} of its caster, for
 *   the herald's Ultimate Takedown. Oracle View: an ultimate ability_activated event
 *   on the victim whose source also has a lethal_damage_contribution row to it.
 *   Agent POV (no contribution rows): a damaging Ultimate activation on the victim
 *   from a source the team saw at the start of the tick, the same seen blow that
 *   names a killer. The host records no single last hit, so when several qualify the
 *   first in event order is used. null when none qualifies, and always null when the
 *   killers came only from the match summary.
 * Return frozen falls ordered like the beat schedule (by public ID), each with its
 * Fall beat time and END world spot. Pure.
 *
 * @param {{
 *   presentation: Record<string, any>,
 *   rows: ReadonlyArray<Readonly<{id: string, kind: string, vocabulary: string, payload: any}>>,
 *   sceneByKey: Map<string, Record<string, any>>,
 *   trajectories: Map<string, Record<string, any>> | null,
 *   schedule: BeatSchedule,
 *   agentVisual: boolean,
 *   transitionId: string,
 * }} input
 * @returns {ReadonlyArray<AuthorizedFall>}
 */
function authorizedFalls(input) {
  const { presentation, rows, sceneByKey, trajectories, schedule, agentVisual } = input;
  /**
   * Format one public identity with the frame's display ID.
   *
   * agent supplies public_agent_id, class_id and team_id. Return the shared
   * canonical identity labels plus publicAgentId, classId and teamId (frozen).
   * No position is read.
   *
   * @param {Record<string, any>} agent
   */
  const identity = (agent) =>
    Object.freeze({
      ...canonicalAgentIdentity({
        ...agent,
        display_agent_id: authorizedPresentationAgentDisplayId(
          presentation,
          agent.public_agent_id,
        ),
      }),
      publicAgentId: String(agent.public_agent_id),
      classId: integer(agent.class_id),
      teamId: integer(agent.team_id),
    });
  const scenePublic = new Map(
    [...sceneByKey.values()].map((agent) => [identifier(agent.public_agent_id), agent]),
  );
  const summaryDeaths = agentVisual ? [] : array(presentation.match_summary?.deaths);
  /** @type {AuthorizedFall[]} */
  const falls = [];
  for (const entry of schedule.falls) {
    const victimRow = rows.find(
      (row) =>
        row.kind === "agent_died" &&
        identifier(
          (record(row.payload?.recipient_anchor) ?? record(row.payload?.agent_anchor))
            ?.presentation_key,
        ) === entry.presentationKey,
    );
    const victimAgent = sceneByKey.get(entry.presentationKey);
    const victimSnapshot = authorizedIdentitySnapshot(victimAgent);
    if (!victimRow || !victimAgent || victimSnapshot === null) continue;
    /** @type {Map<string, Record<string, any>>} */
    const killers = new Map();
    /** @type {Readonly<{classId: number, publicAgentId: string}> | null} */
    let ultimate = null;
    if (agentVisual) {
      for (const row of rows) {
        if (row.kind !== "ability_activated" || row.vocabulary !== "event") continue;
        const recipientAnchor = record(row.payload?.recipient_anchor);
        if (identifier(recipientAnchor?.public_agent_id) !== entry.publicAgentId)
          continue;
        const sourceAnchor = record(row.payload?.source_anchor);
        const sourceKey = identifier(sourceAnchor?.presentation_key);
        const sourcePublicId = identifier(sourceAnchor?.public_agent_id);
        const trajectory =
          sourceKey === null || trajectories === null
            ? null
            : trajectoryForAuthorizedAnchor(
                trajectories,
                sourceAnchor,
                "transition_start",
              );
        const classId = integer(trajectory?.agent_class_id);
        if (trajectory === null || sourcePublicId === null || classId === null)
          continue;
        const token =
          row.payload?.ability_component === "ultimate"
            ? ultimateTokenFromClassId(classId, row.payload)
            : resolveVisualToken(
                "activation",
                classId === 5 ? "basic_heal" : "basic_damage",
                row.payload,
              );
        if (activationImpactSemantic(token.tokenId) !== "damage") continue;
        if (ultimate === null && row.payload?.ability_component === "ultimate") {
          ultimate = Object.freeze({ classId, publicAgentId: sourcePublicId });
        }
        const seenSource = scenePublic.get(sourcePublicId);
        const sourceTeam = integer(seenSource?.team_id) ?? 3 - victimSnapshot.team_id;
        killers.set(
          sourcePublicId,
          Object.freeze({
            ...identity({
              public_agent_id: sourcePublicId,
              class_id: classId,
              team_id: sourceTeam,
            }),
            presentationKey: sourceKey,
            lastSeen: trajectory.successor == null,
          }),
        );
      }
    } else {
      for (const row of rows) {
        if (row.kind !== "lethal_damage_contribution") continue;
        if (
          identifier(row.payload?.recipient_anchor?.public_agent_id) !==
          entry.publicAgentId
        ) {
          continue;
        }
        const sourcePublicId = identifier(row.payload?.source_anchor?.public_agent_id);
        const source = sourcePublicId === null ? null : scenePublic.get(sourcePublicId);
        const snapshot = authorizedIdentitySnapshot(source);
        if (sourcePublicId === null || snapshot === null) continue;
        killers.set(
          sourcePublicId,
          Object.freeze({
            ...identity(snapshot),
            presentationKey: snapshot.presentation_key,
            lastSeen: false,
          }),
        );
      }
      // Ultimate Takedown: an ultimate on the victim from a lethal contributor.
      for (const row of rows) {
        if (ultimate !== null) break;
        if (row.kind !== "ability_activated" || row.vocabulary !== "event") continue;
        if (row.payload?.ability_component !== "ultimate") continue;
        if (
          identifier(row.payload?.recipient_anchor?.public_agent_id) !==
          entry.publicAgentId
        ) {
          continue;
        }
        const casterId = identifier(row.payload?.source_anchor?.public_agent_id);
        const caster = casterId === null ? undefined : killers.get(casterId);
        if (caster === undefined || casterId === null) continue;
        const token = ultimateTokenFromClassId(caster.classId, row.payload);
        if (activationImpactSemantic(token.tokenId) !== "damage") continue;
        ultimate = Object.freeze({ classId: caster.classId, publicAgentId: casterId });
      }
      if (killers.size === 0) {
        const death = summaryDeaths.find(
          (candidate) => candidate?.public_agent_id === entry.publicAgentId,
        );
        for (const contributor of array(death?.contributors)) {
          const sourcePublicId = identifier(contributor?.public_agent_id);
          if (sourcePublicId === null) continue;
          killers.set(
            sourcePublicId,
            Object.freeze({
              ...identity(contributor),
              presentationKey:
                identifier(scenePublic.get(sourcePublicId)?.presentation_key) ?? null,
              lastSeen: false,
            }),
          );
        }
      }
    }
    const orderedKillers = Object.freeze(
      [...killers.entries()]
        .sort(([left], [right]) => comparePublicAgentIds(left, right))
        .map(([, killer]) => killer),
    );
    const killerTeams = new Set(orderedKillers.map((killer) => killer.teamId));
    const flag = victimRow.payload?.home_fall;
    const homeFall = typeof flag === "boolean" ? flag : null;
    falls.push(
      Object.freeze({
        eventId: victimRow.id,
        presentationKey: entry.presentationKey,
        publicAgentId: entry.publicAgentId,
        victim: Object.freeze({
          ...identity(victimSnapshot),
          presentationKey: entry.presentationKey,
        }),
        killers: orderedKillers,
        killingTeamId:
          killerTeams.size === 1
            ? [...killerTeams][0]
            : killerTeams.size === 0
              ? 3 - victimSnapshot.team_id
              : null,
        homeFall,
        points: homeFall === null ? null : homeFall ? 2 : 1,
        ultimate,
        at: entry.at,
        world: point(victimAgent.position),
      }),
    );
  }
  return Object.freeze(falls);
}

/**
 * @typedef {Readonly<{
 *   match: Readonly<{
 *     before: readonly [number, number],
 *     after: readonly [number, number],
 *     pointsToWin: number | null,
 *     tick: number | null,
 *     tickLimit: number | null,
 *   }> | null,
 *   freshStart: boolean,
 *   largestLeadBefore: number,
 *   runs: ReadonlyArray<Readonly<{teamId: 1 | 2, takedowns: number}>>,
 *   rescues: ReadonlyArray<Readonly<{champion: Readonly<Record<string, any>>, byUltimate: boolean}>>,
 *   combos: ReadonlyArray<Readonly<{
 *     casters: ReadonlyArray<Readonly<{classId: number, publicAgentId: string, teamId: number | null}>>,
 *     target: Readonly<Record<string, any>> | null,
 *     fell: boolean,
 *   }>>,
 *   wiped: ReadonlyArray<Readonly<{teamId: 1 | 2, members: ReadonlyArray<Readonly<Record<string, any>>>}>>,
 * }>} HeraldTickFacts
 *   The herald row's moment facts for one tick and one audience
 *   (heraldTickFacts); looks/heralds.js heraldMoments turns them into moments.
 */

/** Herald facts of a tick with none: no match facts, no history, no moments. */
const NO_HERALD_FACTS = /** @type {HeraldTickFacts} */ (
  Object.freeze({
    match: null,
    freshStart: false,
    largestLeadBefore: 0,
    runs: Object.freeze([]),
    rescues: Object.freeze([]),
    combos: Object.freeze([]),
    wiped: Object.freeze([]),
  })
);

/**
 * Gather the herald row's moment facts for one tick and one audience (pure).
 *
 * input.presentation is the frame; input.falls its named falls
 * (authorizedFalls); input.rows its own incoming rows; input.sceneByKey its
 * authorized bodies (Agent POV: the seen bodies and local corpses);
 * input.trajectories its joined phase anchors (null when the inventory has
 * none); input.agentVisual true for Agent POV. Return a frozen
 * HeraldTickFacts:
 * - match: the public scoreboard of a Team Deathmatch frame (the scores at the
 *   start and end of the tick from authorizedPresentationMatchBeatFacts,
 *   First To, the tick and the draw tick from matchSummaryView), the same
 *   numbers every view's scorebug shows; null for another task or no summary.
 * - freshStart, largestLeadBefore, runs ({teamId, takedowns}): copied from the
 *   frame's own match_moments host fact (viewer/match_moments.py: in Agent POV
 *   counted only from the falls the team saw); false, 0 and [] when the host
 *   sent none.
 * - rescues ({champion, byUltimate}): the host fact's rescues whose champion is
 *   in this audience's scene (the host already kept, in Agent POV, only
 *   rescues the team saw whole); champion is its identity record.
 * - combos ({casters, target, fell}): champions struck this tick by the
 *   damaging Ultimates (Charge, Freezing Trap, Crippling Poison) of two or more
 *   casters, from this audience's own ability_activated rows. Agent POV counts
 *   a blow only when the team saw its caster and its target at the start of
 *   the tick, and, when either charged this tick (a Warrior's Ultimate), at
 *   its end too (store struck rule). target is the struck champion's identity
 *   (null when it is not in the scene); fell whether it is a named fall.
 * - wiped ({teamId, members}): a team that lost a named fall this tick and
 *   whose every configured-active champion (the public roster,
 *   publicRosterIdentities) is in this audience's scene as fallen (alive
 *   false or a corpse). In Agent POV a champion the team cannot see is not in
 *   the scene, so the team is not wiped there (the store's rule).
 * Identity records carry the shared canonical labels plus publicAgentId,
 * classId and teamId, with the frame's display IDs (null display IDs when
 * presentation is not a normalized frame, which also gives match null). No
 * position, no researcher view and no hidden match_summary row is read.
 * Exported for the plan builder's unit tests; buildChoreographyPlan is its
 * one caller in the product.
 *
 * @param {{
 *   presentation: Record<string, any>,
 *   falls: ReadonlyArray<AuthorizedFall>,
 *   rows: ReadonlyArray<Readonly<{id: string, kind: string, vocabulary: string, payload: any}>>,
 *   sceneByKey: Map<string, Record<string, any>>,
 *   trajectories: Map<string, Record<string, any>> | null,
 *   agentVisual: boolean,
 * }} input
 * @returns {HeraldTickFacts}
 */
export function heraldTickFacts(input) {
  const { presentation, falls, rows, sceneByKey, trajectories, agentVisual } = input;
  const host = record(presentation.match_moments);
  const scenePublic = new Map(
    [...sceneByKey.values()].map((agent) => [identifier(agent.public_agent_id), agent]),
  );
  /**
   * Format one scene body as an identity record with the frame's display ID.
   *
   * @param {Record<string, any>} agent
   */
  const identityOf = (agent) =>
    Object.freeze({
      ...canonicalAgentIdentity({
        ...agent,
        display_agent_id: authorizedPresentationAgentDisplayId(
          presentation,
          agent.public_agent_id,
        ),
      }),
      publicAgentId: String(agent.public_agent_id),
      classId: integer(agent.class_id),
      teamId: integer(agent.team_id),
    });
  const summary = matchSummaryView(presentation);
  const beat = authorizedPresentationMatchBeatFacts(presentation);
  const match =
    summary !== null && beat !== null && summary.taskMode === 1
      ? Object.freeze({
          before: beat.scores_before,
          after: beat.scores_after,
          pointsToWin: integer(summary.threshold),
          tick: integer(summary.tick?.current),
          tickLimit: integer(summary.tick?.limit),
        })
      : null;
  const runs = array(host?.runs).flatMap((fact) => {
    const teamId = integer(record(fact)?.team_id);
    const takedowns = integer(record(fact)?.takedowns);
    return (teamId === 1 || teamId === 2) && takedowns !== null && takedowns > 0
      ? [Object.freeze({ teamId: /** @type {1 | 2} */ (teamId), takedowns })]
      : [];
  });
  const rescues = array(host?.rescues).flatMap((fact) => {
    const agent = scenePublic.get(identifier(record(fact)?.public_agent_id));
    return agent && authorizedIdentitySnapshot(agent) !== null
      ? [
          Object.freeze({
            champion: identityOf(agent),
            byUltimate: record(fact)?.by_ultimate === true,
          }),
        ]
      : [];
  });
  // Ultimate Combo: damaging Ultimates of two or more casters on one target.
  /** @type {Map<string, {target: string, casters: Map<string, Readonly<{classId: number, publicAgentId: string, teamId: number | null}>>}>} */
  const struck = new Map();
  const ultimates = rows.filter(
    (row) =>
      row.kind === "ability_activated" &&
      row.vocabulary === "event" &&
      row.payload?.ability_component === "ultimate",
  );
  /** @param {unknown} anchor */
  const classOf = (anchor) => {
    const key = identifier(record(anchor)?.presentation_key);
    const body = key === null ? undefined : sceneByKey.get(key);
    const trajectory =
      trajectories === null || key === null ? null : (trajectories.get(key) ?? null);
    return integer(body?.class_id) ?? integer(trajectory?.agent_class_id);
  };
  const charged = new Set(
    ultimates
      .filter((row) => classOf(row.payload?.source_anchor) === 2)
      .map((row) => identifier(record(row.payload?.source_anchor)?.presentation_key)),
  );
  /** @param {unknown} anchor */
  const seenAtBlow = (anchor) => {
    if (!agentVisual) return true;
    const key = identifier(record(anchor)?.presentation_key);
    const trajectory =
      trajectories === null || key === null ? null : (trajectories.get(key) ?? null);
    return (
      trajectory?.transition_start != null &&
      (trajectory?.successor != null || !charged.has(key))
    );
  };
  for (const row of ultimates) {
    const source = record(row.payload?.source_anchor);
    const target = record(row.payload?.recipient_anchor);
    const sourceId = identifier(source?.public_agent_id);
    const targetId = identifier(target?.public_agent_id);
    const classId = classOf(source);
    if (sourceId === null || targetId === null || sourceId === targetId) continue;
    if (classId === null) continue;
    const token = ultimateTokenFromClassId(classId, row.payload);
    if (activationImpactSemantic(token.tokenId) !== "damage") continue;
    if (!seenAtBlow(source) || !seenAtBlow(target)) continue;
    const entry = struck.get(targetId) ?? { target: targetId, casters: new Map() };
    entry.casters.set(
      sourceId,
      Object.freeze({
        classId,
        publicAgentId: sourceId,
        teamId: integer(scenePublic.get(sourceId)?.team_id),
      }),
    );
    struck.set(targetId, entry);
  }
  const combos = [...struck.values()]
    .filter((entry) => entry.casters.size >= 2)
    .sort((left, right) => comparePublicAgentIds(left.target, right.target))
    .map((entry) => {
      const body = scenePublic.get(entry.target);
      return Object.freeze({
        casters: Object.freeze(
          [...entry.casters.values()].sort((left, right) =>
            comparePublicAgentIds(left.publicAgentId, right.publicAgentId),
          ),
        ),
        target:
          body !== undefined && authorizedIdentitySnapshot(body) !== null
            ? identityOf(body)
            : null,
        fell: falls.some((fall) => fall.publicAgentId === entry.target),
      });
    });
  // Team Wiped: every active member seen fallen, one of them on this tick.
  const roster = publicRosterIdentities(presentation);
  /** @type {Array<Readonly<{teamId: 1 | 2, members: ReadonlyArray<Readonly<Record<string, any>>>}>>} */
  const wiped = [];
  for (const teamId of /** @type {Array<1 | 2>} */ ([1, 2])) {
    if (!falls.some((fall) => fall.victim.teamId === teamId)) continue;
    const members = roster.filter((row) => row.teamId === teamId);
    const bodies = members.map((row) => scenePublic.get(row.publicAgentId));
    if (
      members.length > 0 &&
      bodies.every(
        (body) =>
          body !== undefined &&
          authorizedIdentitySnapshot(body) !== null &&
          (body.alive === false || body.life_state === "corpse"),
      )
    ) {
      wiped.push(
        Object.freeze({
          teamId,
          members: Object.freeze(
            bodies.map((body) => identityOf(/** @type {Record<string, any>} */ (body))),
          ),
        }),
      );
    }
  }
  return Object.freeze({
    match,
    freshStart: host?.fresh_start === true,
    largestLeadBefore: integer(host?.largest_lead_before) ?? 0,
    runs: Object.freeze(runs),
    rescues: Object.freeze(rescues),
    combos: Object.freeze(combos),
    wiped: Object.freeze(wiped),
  });
}

/**
 * Build the plan rows of the five derived event families for one tick.
 *
 * input.falls are this audience's named falls (authorizedFalls); input.rows its own
 * rows; input.sceneByKey and input.trajectories its authorized bodies and anchors;
 * input.schedule the beat schedule; input.surface the projection (null keeps every
 * row non-spatial); input.visualFilters the filter state; input.agentVisual true for
 * Agent POV; input.teamSide maps a team ID to its home side; input.streaks is the
 * frame's own champion_streaks host fact (the host's count for this frame's view:
 * in Agent POV only takedowns the team saw), or null or absent when the host did
 * not count; input.heraldFacts is this tick's heraldTickFacts (the herald row's
 * moment facts), or null or absent for none.
 *
 * Families (each row has kind, eventId, eventType, paintParts.effect, spatial,
 * presentationSuppressed, persistent, phaseStart and phaseEnd in real ms):
 * - home_fall: one row per victim team whose seen fall carries the host's
 *   home_fall flag. It starts at that team's first such fall and lasts 600 ms.
 *   Agent POV: only falls the team saw (the host sends the flag only for those).
 * - fallen_feed_row: one row per named fall: victim, named killers, killing team,
 *   points, ultimate (the damaging Ultimate caster {classId, publicAgentId} or
 *   null, as on herald_card falls) and the scoring team's side. It enters at the victim's Fall beat time
 *   and stays to the end of the Fall beat.
 * - herald_card: one row per tick with named falls, and on a tick without one
 *   only when looks/heralds.js cardMoments names a moment for it. count equals
 *   the named falls (it never counts a fall the team did not see); byTeam counts
 *   them per killing team. Each falls entry carries victim, killers,
 *   killingTeamId, points and ultimate (the fall's damaging Ultimate caster
 *   {classId, publicAgentId} or null; see authorizedFalls) so looks/heralds.js
 *   can name an Ultimate Takedown. streaks lists [{champion, takedowns}] for the
 *   Champion Streaks switch: one entry per input.streaks fact (every 5th
 *   takedown without falling: 5, 10, 15, 20), in host order, whose champion is a killer of one of this
 *   tick's named falls; champion is that killer's identity record and takedowns
 *   the fact's number. A fact naming anyone else is dropped, so the card never
 *   names a champion this audience did not see strike. match, freshStart,
 *   largestLeadBefore, runs, rescues, combos and wiped are input.heraldFacts
 *   (heraldTickFacts; absent: none). The browser never counts history. The row
 *   enters at the first Fall beat (schedule.scoreAt when this audience saw no
 *   fall), or at the tick's settled time (schedule.explanationEnd) when every
 *   moment it names waits for it (heralds.js momentEntersAtSettle: Rescue,
 *   Ticks Left, a Combo whose target did not fall), and it is kept in the
 *   paused still. budget is heralds.js rowBudget(row, visualFilters): the
 *   paint allowance of the card it names (planEventBudget charges it).
 * - last_seen (Agent POV only): one row per body that left sight this tick (start
 *   anchor and no successor), at its START spot, kept in the paused still. teamId
 *   comes from the public roster (input.teamByPublicId: the identity
 *   directory's team of each public ID, the same fact the Unseen portraits
 *   show in both views), so the ring takes that team's light; null when the
 *   roster does not name the body.
 * - focus_tally: a target struck this tick by two or more enemy sources, at most
 *   four, busiest first then by public ID. Agent POV counts a blow only when its
 *   source and target were seen at both ends of the tick. Anchored at the target's
 *   END spot; it appears at Impact.
 * A family whose paint part is off or unregistered stays in the plan as data only.
 * Return a frozen array of frozen rows. Pure; reads no researcher data and no
 * match summary.
 *
 * @param {{
 *   falls: ReadonlyArray<AuthorizedFall>,
 *   rows: ReadonlyArray<Readonly<{id: string, kind: string, vocabulary: string, payload: any}>>,
 *   sceneByKey: Map<string, Record<string, any>>,
 *   trajectories: Map<string, Record<string, any>> | null,
 *   schedule: BeatSchedule,
 *   surface: ProjectionSurface | null,
 *   visualFilters: Record<string, boolean>,
 *   agentVisual: boolean,
 *   transitionId: string,
 *   teamSide: (teamId: number) => "left" | "right",
 *   streaks?: ReadonlyArray<Readonly<Record<string, any>>> | null,
 *   teamByPublicId?: ReadonlyMap<string, 1 | 2> | null,
 *   heraldFacts?: HeraldTickFacts | null,
 * }} input
 * @returns {ReadonlyArray<Readonly<Record<string, any>>>}
 */
function derivedFamilyEvents(input) {
  const { falls, rows, sceneByKey, trajectories, schedule, surface } = input;
  const { visualFilters, agentVisual, transitionId, teamSide } = input;
  /** @type {Readonly<Record<string, any>>[]} */
  const planned = [];
  /**
   * Return the common switch fields for a row of one family.
   *
   * family is a DERIVED_EVENT_PAINT_PARTS key; anchored says whether the row has a
   * screen anchor (or needs none). Return paintParts, spatial and
   * presentationSuppressed.
   *
   * @param {keyof typeof DERIVED_EVENT_PAINT_PARTS} family
   * @param {boolean} anchored
   */
  const paint = (family, anchored) => {
    const enabled = derivedFamilyEnabled(visualFilters, family);
    return {
      paintParts: Object.freeze({ effect: enabled }),
      presentationSuppressed: !enabled,
      spatial: enabled && anchored,
    };
  };
  const fallEnd = schedule.fall?.[1] ?? null;

  // Home Fall: the host's home_fall flag only, never a guess from positions.
  for (const teamId of [1, 2]) {
    const homeFalls = falls.filter(
      (fall) => fall.homeFall === true && fall.victim.teamId === teamId,
    );
    if (homeFalls.length === 0) continue;
    const at = homeFalls[0].at;
    planned.push(
      Object.freeze({
        eventId: `${transitionId}:home-fall:${teamId}`,
        eventType: "agent_died",
        kind: "home_fall",
        teamId,
        teamSide: teamSide(teamId),
        victims: Object.freeze(homeFalls.map((fall) => fall.victim)),
        count: homeFalls.length,
        ...paint("home_fall", true),
        persistent: false,
        phaseStart: at,
        phaseEnd: at + BEAT_CLOCK.endHold,
      }),
    );
  }

  // Fallen feed rows: one per named fall, only killers this audience may name.
  for (const fall of falls) {
    const scoringTeam = fall.killingTeamId ?? 3 - fall.victim.teamId;
    planned.push(
      Object.freeze({
        eventId: `${transitionId}:fallen:${fall.publicAgentId}`,
        eventType: "agent_died",
        kind: "fallen_feed_row",
        victim: fall.victim,
        victimPresentationKey: fall.presentationKey,
        killers: fall.killers,
        killingTeamId: fall.killingTeamId,
        points: fall.points,
        homeFall: fall.homeFall,
        ultimate: fall.ultimate,
        teamId: scoringTeam,
        teamSide: teamSide(scoringTeam),
        ...paint("fallen_feed_row", true),
        persistent: false,
        phaseStart: fall.at,
        phaseEnd: Math.max(fallEnd ?? fall.at + READABLE_OUTCOME_DWELL_MS, fall.at),
      }),
    );
  }

  // Herald card: counts only the falls it names. A tick without a named fall
  // gets a row only when the moment finder names a moment (a Rescue, an
  // Ultimate Combo, a run, Match Point, Scores Level or Ticks Left).
  {
    /** @type {Record<string, number>} */
    const byTeam = { 1: 0, 2: 0 };
    for (const fall of falls) {
      const team = fall.killingTeamId ?? 3 - fall.victim.teamId;
      byTeam[team] = (byTeam[team] ?? 0) + 1;
    }
    // Champion Streaks: the host counts (per view); a fact names a killer of a
    // named fall this tick, so only identities this audience already names.
    const killersById = new Map(
      falls.flatMap((fall) =>
        fall.killers.map((killer) => [killer.publicAgentId, killer]),
      ),
    );
    const streaks = array(input.streaks).flatMap((fact) => {
      const champion = killersById.get(identifier(record(fact)?.public_agent_id) ?? "");
      const takedowns = integer(record(fact)?.takedowns);
      return champion && takedowns !== null && takedowns > 0
        ? [Object.freeze({ champion, takedowns })]
        : [];
    });
    const facts = input.heraldFacts ?? NO_HERALD_FACTS;
    const row = {
      eventId: `${transitionId}:herald`,
      eventType: falls.length > 0 ? "agent_died" : "match_moment",
      kind: "herald_card",
      count: falls.length,
      byTeam: Object.freeze({ 1: byTeam[1], 2: byTeam[2] }),
      falls: Object.freeze(
        falls.map((fall) =>
          Object.freeze({
            victim: fall.victim,
            killers: fall.killers,
            killingTeamId: fall.killingTeamId,
            points: fall.points,
            ultimate: fall.ultimate,
          }),
        ),
      ),
      streaks: Object.freeze(streaks),
      match: facts.match,
      freshStart: facts.freshStart,
      largestLeadBefore: facts.largestLeadBefore,
      runs: facts.runs,
      rescues: facts.rescues,
      combos: facts.combos,
      wiped: facts.wiped,
    };
    // The moments this row's card names (looks/heralds.js owns the rules): a
    // card enters at the first Fall beat (or the score time when this view saw
    // no fall), unless it names only moments that wait for the tick to
    // settle (Rescue, Ticks Left, a Combo whose target did not fall).
    const named = cardMoments(row, visualFilters);
    if (falls.length > 0 || named.length > 0) {
      const fallAt = falls[0]?.at ?? schedule.scoreAt;
      const phaseStart =
        named.length > 0 && named.every(momentEntersAtSettle)
          ? schedule.explanationEnd
          : fallAt;
      planned.push(
        Object.freeze({
          ...row,
          ...paint("herald_card", true),
          persistent: true,
          phaseStart,
          phaseEnd: Math.max(phaseStart, schedule.explanationEnd),
          // The paint allowance of this row's own card (planEventBudget).
          budget: rowBudget(row, visualFilters),
        }),
      );
    }
  }

  // Last Seen rings: Agent POV bodies that left sight this tick.
  if (agentVisual && trajectories !== null) {
    for (const trajectory of trajectories.values()) {
      if (trajectory.transition_start == null || trajectory.successor != null) continue;
      const world = authorizedWorldPoint(trajectory.transition_start);
      const enabled = derivedFamilyEnabled(visualFilters, "last_seen");
      const anchor = enabled ? project(world, surface) : null;
      planned.push(
        Object.freeze({
          eventId: `${transitionId}:last-seen:${trajectory.agent_public_agent_id}`,
          eventType: "agent_left_sight",
          kind: "last_seen",
          presentationKey: trajectory.agent_presentation_key,
          publicAgentId: trajectory.agent_public_agent_id,
          classId: integer(trajectory.agent_class_id),
          teamId:
            input.teamByPublicId?.get(String(trajectory.agent_public_agent_id)) ?? null,
          world,
          anchor,
          ...paint("last_seen", anchor !== null),
          persistent: true,
          phaseStart: 0,
          phaseEnd: schedule.choreographyEnd,
        }),
      );
    }
  }

  // Focus Tally: a target struck by two or more enemies this tick.
  /** @type {Map<string, {target: Record<string, any>, sources: Set<string>, team: number}>} */
  const tallies = new Map();
  for (const row of rows) {
    if (row.kind !== "ability_activated" || row.vocabulary !== "event") continue;
    const sourceAnchor = record(row.payload?.source_anchor);
    const targetAnchor = record(row.payload?.recipient_anchor);
    const source = sceneByKey.get(identifier(sourceAnchor?.presentation_key) ?? "");
    const target = sceneByKey.get(identifier(targetAnchor?.presentation_key) ?? "");
    if (!source || !target || source === target) continue;
    if (agentVisual) {
      const sourceTrajectory = trajectories?.get(source.presentation_key);
      const targetTrajectory = trajectories?.get(target.presentation_key);
      if (
        sourceTrajectory?.transition_start == null ||
        sourceTrajectory?.successor == null ||
        targetTrajectory?.transition_start == null ||
        targetTrajectory?.successor == null
      ) {
        continue;
      }
    }
    const sourceTeam = integer(source.team_id);
    const targetTeam = integer(target.team_id);
    if (sourceTeam === null || targetTeam === null || sourceTeam === targetTeam)
      continue;
    const key = String(target.public_agent_id);
    const tally = tallies.get(key) ?? { target, sources: new Set(), team: sourceTeam };
    tally.sources.add(String(source.public_agent_id));
    tallies.set(key, tally);
  }
  const fellKeys = new Set(falls.map((fall) => fall.presentationKey));
  const tallyRows = [...tallies.values()]
    .filter((tally) => tally.sources.size >= 2)
    .sort(
      (left, right) =>
        right.sources.size - left.sources.size ||
        comparePublicAgentIds(
          String(left.target.public_agent_id),
          String(right.target.public_agent_id),
        ),
    )
    .slice(0, 4);
  for (const tally of tallyRows) {
    const world = point(tally.target.position);
    const enabled = derivedFamilyEnabled(visualFilters, "focus_tally");
    const anchor = enabled ? project(world, surface) : null;
    planned.push(
      Object.freeze({
        eventId: `${transitionId}:focus:${tally.target.public_agent_id}`,
        eventType: "ability_activated",
        kind: "focus_tally",
        targetPresentationKey: tally.target.presentation_key,
        targetPublicAgentId: tally.target.public_agent_id,
        sourcePublicAgentIds: Object.freeze(
          [...tally.sources].sort(comparePublicAgentIds),
        ),
        attackingTeamId: tally.team,
        count: tally.sources.size,
        lethal: fellKeys.has(tally.target.presentation_key),
        world,
        anchor,
        anchorRadius:
          anchor !== null && surface
            ? surface.worldLengthToScreen(Number(tally.target.radius) || 0.5)
            : null,
        ...paint("focus_tally", anchor !== null),
        persistent: true,
        phaseStart: schedule.windows.impact[0],
        phaseEnd: schedule.choreographyEnd,
      }),
    );
  }
  return Object.freeze(planned);
}

/**
 * @typedef {Readonly<{
 *   presentationKey: string,
 *   publicAgentId: string,
 *   teamId: number | null,
 *   alive: boolean,
 *   start: readonly [number, number] | null,
 *   postCharge: readonly [number, number] | null,
 *   end: readonly [number, number],
 *   charged: boolean,
 *   died: boolean,
 *   returned: boolean,
 *   glide: boolean,
 *   dimShift: boolean,
 * }>} PlanBody
 */

/**
 * Describe where each authorized body began and ended the tick, for glides.
 *
 * sceneByKey holds the audience's bodies (Agent POV: seen bodies and local corpses);
 * trajectories the joined phase anchors, or null. falls and returns are the
 * audience's own seen deaths and returns (beat schedule entries). Return one frozen
 * record per scene body in scene order with world positions: start (transition
 * start, or null when not disclosed), postCharge (Oracle only, where a Charge rush
 * stopped, or null), end (the body's END spot), and whether it glides in play:
 * glide is true only when start is known, the body did not return to a pad this
 * tick (returns never travel) and it moved. dimShift is true for a gliding body
 * whose last leg (from the post-Charge spot or start to END) cuts into a stone of
 * obstacles (the public map stones, default none): the choreographer then dims it
 * out and back in rather than sliding it over the stone. In Agent POV a body that
 * came into sight this tick has no start, so it never glides out of the dark. Pure.
 *
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @param {Map<string, Record<string, any>> | null} trajectories
 * @param {BeatSchedule} schedule
 * @param {ReadonlyArray<unknown>} [obstacles]
 * @returns {ReadonlyArray<PlanBody>}
 */
function planBodies(sceneByKey, trajectories, schedule, obstacles = []) {
  const fallen = new Set(schedule.falls.map((entry) => entry.presentationKey));
  const returned = new Set(schedule.returns.map((entry) => entry.presentationKey));
  /** @type {PlanBody[]} */
  const bodies = [];
  for (const [key, agent] of sceneByKey) {
    const end = point(agent.position);
    const publicId = identifier(agent.public_agent_id);
    if (end === null || publicId === null) continue;
    const trajectory = trajectories?.get(key) ?? null;
    const start = authorizedWorldPoint(trajectory?.transition_start);
    const post = authorizedWorldPoint(trajectory?.post_charge);
    const charged =
      start !== null &&
      post !== null &&
      Math.hypot(post[0] - start[0], post[1] - start[1]) > 1e-6;
    const moved =
      start !== null && Math.hypot(end[0] - start[0], end[1] - start[1]) > 1e-6;
    bodies.push(
      Object.freeze({
        presentationKey: key,
        publicAgentId: publicId,
        teamId: integer(agent.team_id),
        alive: agent.alive === true || agent.life_state === "alive",
        start,
        postCharge: charged ? post : null,
        end,
        charged,
        died: fallen.has(key),
        returned: returned.has(key),
        glide: start !== null && !returned.has(key) && (moved || charged),
        dimShift:
          start !== null &&
          !returned.has(key) &&
          (moved || charged) &&
          cutsStone(
            charged ? post : start,
            end,
            obstacles,
            finiteNumber(agent.radius) ?? 0.5,
          ),
      }),
    );
  }
  return Object.freeze(bodies);
}

/**
 * Build the camera's input for one tick from the audience's own scene and events.
 *
 * bodies are planBodies records; events the plan rows after filtering and layout
 * (only their world anchors are read through rows); rows the audience's own rows;
 * trajectories its joined anchors. Return a frozen record:
 * - bodies: every living, falling or returning body with its START (or null) and
 *   END world spots and state ("living", "falling" or "returning"). Corpses from
 *   earlier ticks are left out. Everyone In View fits these spots.
 * - fights: one entry per Basic or Ultimate activation with the world spots of its
 *   source and target (START for a source that left sight), and one per seen fall
 *   at the victim's END spot. Battle Cam frames these.
 * In Agent POV everything comes from the seen scene and visual_events, so the camera
 * never leans toward a champion the team cannot see. Pure.
 *
 * @param {ReadonlyArray<PlanBody>} bodies
 * @param {ReadonlyArray<Readonly<{id: string, kind: string, vocabulary: string, payload: any}>>} rows
 * @param {Map<string, Record<string, any>> | null} trajectories
 * @param {"researcher" | "agent_pov" | null} audience
 */
function planCamera(bodies, rows, trajectories, audience) {
  const cameraBodies = bodies
    .filter((body) => body.alive || body.died || body.returned)
    .map((body) =>
      Object.freeze({
        presentationKey: body.presentationKey,
        publicAgentId: body.publicAgentId,
        start: body.returned ? null : body.start,
        end: body.end,
        state: body.died ? "falling" : body.returned ? "returning" : "living",
      }),
    );
  /** @type {Readonly<Record<string, any>>[]} */
  const fights = [];
  /**
   * Return the drawn world spot of one anchor: its successor when disclosed,
   * otherwise its transition start (a body that left sight).
   *
   * @param {unknown} rawAnchor
   */
  const spotOf = (rawAnchor) => {
    const key = identifier(record(rawAnchor)?.presentation_key);
    const trajectory = key === null ? null : (trajectories?.get(key) ?? null);
    return (
      authorizedWorldPoint(trajectory?.successor) ??
      authorizedWorldPoint(trajectory?.transition_start)
    );
  };
  for (const row of rows) {
    if (row.vocabulary !== "event") continue;
    if (row.kind === "ability_activated") {
      const spots = [
        spotOf(row.payload?.source_anchor),
        spotOf(row.payload?.recipient_anchor),
      ].filter((spot) => spot !== null);
      if (spots.length > 0) {
        fights.push(
          Object.freeze({
            eventId: row.id,
            kind: "blow",
            points: Object.freeze(spots),
          }),
        );
      }
    } else if (row.kind === "agent_died") {
      const spot = spotOf(row.payload?.recipient_anchor ?? row.payload?.agent_anchor);
      if (spot !== null) {
        fights.push(
          Object.freeze({
            eventId: row.id,
            kind: "fall",
            points: Object.freeze([spot]),
          }),
        );
      }
    }
  }
  return Object.freeze({
    audience,
    bodies: Object.freeze(cameraBodies),
    fights: Object.freeze(fights),
  });
}

/** The Mage's Burst status (x1.5 damage): the Arcane Blast During Burst condition. */
const MAGE_BURST_STATUS_ID = "mage_burst_damage_amplification";

/** Status rows that prove a status was already active before this tick ended. */
const STATUS_ACTIVE_BEFORE_ROWS = new Set([
  "status_aged_to_zero",
  "status_broken_by_damage",
  "status_refreshed_or_extended",
  "status_cleared_by_new_death",
]);

/**
 * Name the champions whose Mage Burst was active during this tick, so their
 * Basic draws with the Burst look (switch kindled_blast, "Arcane Blast During
 * Burst": the Comet Bolt, or a thicker cleave from point-blank).
 *
 * A champion counts when this audience has its own proof that Burst was on at
 * the start of the tick:
 * - one of this audience's status rows (aged to zero, broken, refreshed or
 *   cleared by death) names Burst on that champion;
 * - the END scene shows Burst on that champion and no row of this tick applied
 *   it fresh (a fresh Burst was not on when the tick began);
 * - Agent POV only: the team's own start-of-tick observation
 *   (presentation.latest_events with summary_kind
 *   "shared_obs_recipient_observation_deltas") shows Burst on that champion.
 *   This is what the team saw on the board when it chose its orders.
 *
 * Agent POV boundary: it reads only this audience's rows, its END scene and
 * its team's own observation. It never reads researcher_space or
 * match_summary. It only styles a blow that visual_events already admitted;
 * it never adds a body, a spot or a blow.
 *
 * @param {Record<string, any>} presentation The normalized presentation frame.
 * @param {ReadonlyArray<Readonly<{kind: string, vocabulary: string, payload: Readonly<Record<string, any>>}>>} rows
 *   This audience's incoming rows (authorizedPresentationIncomingRows).
 * @param {Map<string, Record<string, any>>} sceneByKey This audience's END scene
 *   agents by presentation key.
 * @param {boolean} agentVisualInventory True in Agent POV (fog-filtered
 *   visual_events); false in Oracle View.
 * @returns {ReadonlySet<string>} Presentation keys of the kindled champions
 *   (empty when none). Pure; inputs unchanged.
 */
function kindledSourceKeys(presentation, rows, sceneByKey, agentVisualInventory) {
  /** @type {Set<string>} */
  const keys = new Set();
  /** @type {Set<string>} */
  const freshlyApplied = new Set();
  for (const row of rows) {
    if (row.vocabulary !== "event") continue;
    const event = record(row.payload);
    if (identifier(event?.status_id) !== MAGE_BURST_STATUS_ID) continue;
    const key = identifier(record(event?.recipient_anchor)?.presentation_key);
    if (key === null) continue;
    if (STATUS_ACTIVE_BEFORE_ROWS.has(row.kind)) keys.add(key);
    else if (row.kind === "status_applied") freshlyApplied.add(key);
  }
  for (const [key, agent] of sceneByKey) {
    if (freshlyApplied.has(key) && !keys.has(key)) continue;
    if (
      array(agent.statuses).some(
        (status) => identifier(record(status)?.status_id) === MAGE_BURST_STATUS_ID,
      )
    ) {
      keys.add(key);
    }
  }
  const startView = record(presentation.latest_events);
  if (
    agentVisualInventory &&
    startView?.summary_kind === "shared_obs_recipient_observation_deltas"
  ) {
    for (const delta of array(startView.deltas)) {
      const start = record(record(delta)?.start_observation);
      const key = identifier(start?.presentation_key);
      if (
        key !== null &&
        array(start?.statuses).some(
          (status) => identifier(record(status)?.status_id) === MAGE_BURST_STATUS_ID,
        )
      ) {
        keys.add(key);
      }
    }
  }
  return keys;
}

/**
 * The empty damage-source list of a health row with no recorded sources.
 *
 * @type {ReadonlyArray<Readonly<Record<string, any>>>}
 */
const NO_DAMAGE_SOURCES = Object.freeze([]);

/**
 * Collect each recipient's recorded per-attacker damage (Oracle View only).
 *
 * rows are this audience's own incoming rows; sceneByKey its authorized bodies.
 * Every "source_damage_output" event row with a recipient adds one entry to its
 * recipient's list: {sourcePresentationKey, sourceClassId (from the source's
 * scene body, or null), amount} where amount is source_modified_damage_output
 * times recipient_damage_modifier (the recorded effective damage, in health
 * points). Rows missing a key or a finite value are skipped. Return a Map from
 * recipient presentation key to a frozen list in row order. Call it only for
 * Oracle View: Agent POV rows carry no source amounts. Pure; nothing is
 * recomputed from class rules.
 *
 * @param {ReadonlyArray<Record<string, any>>} rows
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @returns {Map<string, ReadonlyArray<Readonly<Record<string, any>>>>}
 */
function recordedDamageSources(rows, sceneByKey) {
  /** @type {Map<string, Array<Readonly<Record<string, any>>>>} */
  const byRecipient = new Map();
  for (const row of rows) {
    if (row.vocabulary !== "event" || row.kind !== "source_damage_output") continue;
    const event = record(row.payload);
    const recipientKey = identifier(record(event?.recipient_anchor)?.presentation_key);
    const sourceKey = identifier(record(event?.source_anchor)?.presentation_key);
    const output = finiteNumber(event?.source_modified_damage_output);
    const modifier = finiteNumber(event?.recipient_damage_modifier);
    if (
      recipientKey === null ||
      sourceKey === null ||
      output === null ||
      modifier === null
    ) {
      continue;
    }
    const list = byRecipient.get(recipientKey) ?? [];
    list.push(
      Object.freeze({
        sourcePresentationKey: sourceKey,
        sourceClassId: integer(sceneByKey.get(sourceKey)?.class_id),
        amount: output * modifier,
      }),
    );
    byRecipient.set(recipientKey, list);
  }
  return new Map([...byRecipient].map(([key, list]) => [key, Object.freeze(list)]));
}

/**
 * Build incoming explanations from the normalized presentation authority.
 *
 * presentation is a recognized frame. surface supplies optional pixel projection
 * and allocator bounds; visualFilters is the local paint state. Use Agent visual_events
 * or Oracle latest_events, not outgoing inspection (in Agent POV the team's own
 * start-of-tick observation in latest_events only decides whether an admitted Mage
 * Basic is kindled; see kindledSourceKeys). Join serialized anchors and
 * preserve recorded atomic identity; legacy observation deltas remain noncausal and
 * feed-only, while recipient health cues can show endpoint changes explicitly.
 *
 * Return a frozen plan with session/epoch/authority/content/paint identities, phases,
 * ordered rows and resource bounds, or null when required facts/joins/projection are
 * unavailable. Reference-only researcher data may add joined display identity or
 * Ultimate help; local fog-authorized anchors own all world-space cues. Global death
 * announcements use nonspatial match-summary facts in a HUD corner. Paint filters
 * run before allocation. Allocator/token errors may propagate. No DOM, game command,
 * random draw or source-record mutation occurs.
 *
 * @param {Record<string, any>} presentation
 * @param {ProjectionSurface | null} surface
 * @param {Record<string, boolean>} visualFilters
 * @returns {Readonly<ChoreographyPlan> | null}
 */
function buildAuthorizedPresentationChoreographyPlan(
  presentation,
  surface,
  visualFilters,
) {
  const audience = authorizedPresentationAudience(presentation);
  const latest = record(
    audience === "agent_pov" ? presentation.visual_events : presentation.latest_events,
  );
  const scene = authorizedPresentationSceneView(presentation);
  const researcherScene = authorizedPresentationResearcherSceneView(presentation);
  if (!latest || !scene || !researcherScene) {
    return null;
  }
  const transitionId =
    scientificIdentity(latest.incoming_transition_id) ??
    scientificIdentity(latest.incoming_recipient_transition_id);
  const simulatorStep =
    integer(latest.incoming_successor_simulator_step_count) ??
    integer(presentation.simulator_step_count);
  if (transitionId === null || simulatorStep === null) {
    return null;
  }
  const rows = authorizedPresentationIncomingRows(presentation);
  /** @type {Map<string, Record<string, any>>} */
  const sceneByKey = new Map();
  const scenePublicIds = new Set();
  for (const candidate of array(scene.agents)) {
    const agent = record(candidate);
    const key = identifier(agent?.presentation_key);
    const publicId = identifier(agent?.public_agent_id);
    if (
      !agent ||
      key === null ||
      publicId === null ||
      sceneByKey.has(key) ||
      scenePublicIds.has(publicId)
    ) {
      return null;
    }
    sceneByKey.set(key, agent);
    scenePublicIds.add(publicId);
  }
  /** @type {Map<string, Record<string, any>>} */
  const researcherAgentByPublicId = new Map();
  for (const candidate of array(researcherScene.agents)) {
    const agent = record(candidate);
    const publicId = identifier(agent?.public_agent_id);
    if (
      !agent ||
      publicId === null ||
      authorizedIdentitySnapshot(agent) === null ||
      researcherAgentByPublicId.has(publicId)
    ) {
      return null;
    }
    researcherAgentByPublicId.set(publicId, agent);
  }
  /** @type {Map<number, Record<string, any>>} */
  const classMechanicsById = new Map();
  // Ultimate help is researcher-space reference material. Resolve it from the
  // authoritative researcher catalog, while all bodies, endpoints, and event
  // admission continue to come exclusively from the fog-authorized scene.
  for (const candidate of array(researcherScene.class_mechanics)) {
    const mechanics = record(candidate);
    const classId = integer(mechanics?.class_id);
    if (mechanics === null || classId === null || classMechanicsById.has(classId)) {
      return null;
    }
    classMechanicsById.set(classId, mechanics);
  }
  const causalVisualInventory =
    latest.summary_kind === "replay_incoming_inventory" ||
    latest.summary_kind === "agent_pov_fog_filtered_visual_events";
  const agentVisualInventory =
    latest.summary_kind === "agent_pov_fog_filtered_visual_events";
  const kindledKeys = kindledSourceKeys(
    presentation,
    rows,
    sceneByKey,
    agentVisualInventory,
  );
  const damageSourcesByRecipient = agentVisualInventory
    ? new Map()
    : recordedDamageSources(rows, sceneByKey);
  const actorInputSuccessorKeys = agentVisualInventory
    ? new Set(
        array(presentation.current_endpoint?.parts?.scene?.agents)
          .map((agent) => identifier(record(agent)?.presentation_key))
          .filter((key) => key !== null),
      )
    : null;
  const deathOverlaySuccessorKeys = agentVisualInventory
    ? new Set(
        rows
          .filter((row) => row.kind === "agent_died")
          .map((row) =>
            identifier(record(row.payload)?.recipient_anchor?.presentation_key),
          )
          .filter(
            (key) =>
              key !== null &&
              actorInputSuccessorKeys instanceof Set &&
              !actorInputSuccessorKeys.has(key),
          ),
      )
    : new Set();
  const trajectories = causalVisualInventory
    ? authorizedTrajectoryMap(
        latest,
        sceneByKey,
        actorInputSuccessorKeys,
        deathOverlaySuccessorKeys,
      )
    : null;
  if (causalVisualInventory && trajectories === null) {
    return null;
  }
  // The beat clock reads only this audience's own death and return events, so a
  // death the Agent POV team did not see changes no time anywhere in the plan.
  const schedule = beatSchedule({
    falls: beatSubjects(rows, "agent_died"),
    returns: beatSubjects(rows, "agent_respawned"),
    visualFilters,
  });
  /**
   * Return a team's home side for this audience (see audienceTeamSide).
   *
   * @param {number} teamId
   */
  const teamSide = (teamId) =>
    audienceTeamSide(presentation, scene, teamId, agentVisualInventory);
  const statusCompositions = authorizedStatusCompositions(
    rows,
    transitionId,
    sceneByKey,
    researcherAgentByPublicId,
    agentVisualInventory,
    trajectories,
    surface,
    visualFilters,
  );
  if (statusCompositions === null) {
    return null;
  }
  /** @type {Record<string, any>[]} */
  const planned = [];
  const plannedStatusCompositions = new Set();
  const healthLaneByKey = new Map();
  for (const row of rows) {
    const event = row.payload;
    const common = {
      eventId: row.id,
      eventType: row.kind,
      transitionId,
      authorityVocabulary: row.vocabulary,
    };
    if (row.vocabulary === "observation_delta") {
      planned.push(
        Object.freeze({
          ...common,
          kind: "feed_only",
          spatial: false,
          noncausal: true,
        }),
      );
      continue;
    }
    if (row.vocabulary === "recipient_cue") {
      const recipientKey = presentation.recipient_presentation_key;
      const recipientAgent = sceneByKey.get(recipientKey);
      const recipientIdentity = authorizedIdentitySnapshot(recipientAgent);
      const recipientWorld = point(recipientAgent?.position);
      const healthBefore = finiteNumber(event.start_health);
      const healthAfter = finiteNumber(event.successor_health);
      if (
        row.kind === "own_health_changed" &&
        recipientWorld &&
        healthBefore !== null &&
        healthAfter !== null
      ) {
        const delta = healthAfter - healthBefore;
        const outcome = delta < 0 ? "damage" : delta > 0 ? "healing" : "unchanged";
        const paintDecision = authorizedPaintDecision(
          { kind: "net_health", outcome },
          visualFilters,
        );
        if (paintDecision === null) {
          return null;
        }
        const recipient = paintDecision.enabled
          ? project(recipientWorld, surface)
          : null;
        if (paintDecision.enabled && recipient === null) {
          planned.push(
            Object.freeze({
              ...common,
              kind: "feed_only",
              spatial: false,
              noncausal: true,
            }),
          );
          continue;
        }
        planned.push(
          Object.freeze({
            ...common,
            kind: "net_health",
            recipientPresentationKey: recipientKey,
            recipientPublicAgentId: presentation.recipient_public_agent_id,
            recipientIdentity,
            recipient,
            netDelta: delta,
            healthBefore,
            healthAfter,
            maximumHealth: finiteNumber(recipientAgent?.maximum_health),
            recipientRadius: paintDecision.enabled
              ? projectedAgentRadius(surface, sceneByKey, recipientKey)
              : 0,
            totalDamage: null,
            totalHealing: null,
            damageSources: NO_DAMAGE_SOURCES,
            outcome,
            lane: paintDecision.enabled ? 0 : null,
            paintParts: paintDecision.paintParts,
            presentationSuppressed: !paintDecision.enabled,
            spatial: paintDecision.enabled,
            noncausal: true,
            presentationKind: "successor_observation",
            phaseStart: CHOREOGRAPHY_PHASES.povSuccessorObservationStart,
            phaseEnd: CHOREOGRAPHY_PHASES.total,
          }),
        );
      } else {
        planned.push(
          Object.freeze({
            ...common,
            kind: "feed_only",
            spatial: false,
            noncausal: true,
          }),
        );
      }
      continue;
    }

    if (STATUS_EVENT_TYPES.has(row.kind)) {
      const composition = statusCompositions.get(row.id);
      if (!composition) {
        return null;
      }
      if (!plannedStatusCompositions.has(composition)) {
        planned.push(composition);
        plannedStatusCompositions.add(composition);
      }
      continue;
    }

    if (
      row.vocabulary === "event" &&
      (row.kind === "team_deathmatch_score_changed" ||
        row.kind === "team_deathmatch_completed")
    ) {
      planned.push(
        Object.freeze({
          ...common,
          kind: "feed_only",
          spatial: false,
        }),
      );
      continue;
    }

    const sourceAnchor = record(event.source_anchor);
    const recipientAnchor = record(event.recipient_anchor);
    const agentAnchor = record(event.agent_anchor);
    if (row.kind === "ability_activated") {
      if (trajectories === null) {
        return null;
      }
      const sourceTrajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        sourceAnchor,
        "transition_start",
      );
      const targetTrajectory =
        recipientAnchor === null
          ? null
          : trajectoryForAuthorizedAnchor(
              trajectories,
              recipientAnchor,
              "transition_start",
            );
      const sourceKey = identifier(sourceAnchor?.presentation_key);
      const targetKey = identifier(recipientAnchor?.presentation_key);
      if (
        sourceTrajectory === null ||
        (recipientAnchor !== null && targetTrajectory === null)
      ) {
        return null;
      }
      const sourceAgent = sourceKey === null ? null : sceneByKey.get(sourceKey);
      const sourceIdentity = authorizedIdentitySnapshot(sourceAgent);
      const sourceClassId = agentVisualInventory
        ? integer(sourceTrajectory.agent_class_id)
        : integer(sourceAgent?.class_id);
      if (
        sourceKey === null ||
        sourceClassId === null ||
        (!agentVisualInventory && !sourceAgent) ||
        (sourceAgent != null &&
          identifier(sourceAgent.public_agent_id) !==
            identifier(sourceTrajectory.agent_public_agent_id))
      ) {
        return null;
      }
      const component = event.ability_component;
      const authorizedClassMechanics =
        component === "ultimate"
          ? (classMechanicsById.get(sourceClassId) ?? null)
          : null;
      const token =
        component === "ultimate"
          ? ultimateTokenFromClassId(sourceClassId, event)
          : resolveVisualToken(
              "activation",
              sourceClassId === 5 ? "basic_heal" : "basic_damage",
              event,
            );
      const impactSemantic = activationImpactSemantic(token.tokenId);
      const paintParts = authorizedPaintParts(
        { kind: "activation", component, impactSemantic },
        visualFilters,
      );
      if (paintParts === null) {
        return null;
      }
      const abilityEnabled = paintParts.ability === true;
      const semanticEnabled = paintParts.semantic === true;
      const targetAgent = targetKey === null ? null : sceneByKey.get(targetKey);
      const targetIdentity = authorizedIdentitySnapshot(targetAgent);
      if (
        (targetKey === null) !== (targetTrajectory === null) ||
        (targetKey !== null &&
          ((!agentVisualInventory && !targetAgent) ||
            (targetAgent != null &&
              identifier(targetAgent.public_agent_id) !==
                identifier(targetTrajectory?.agent_public_agent_id))))
      ) {
        return null;
      }
      const chargeEndpoints = token.tokenId === "warrior_charge";
      const sourceEndpointPhase =
        chargeEndpoints ||
        (agentVisualInventory &&
          (sourceTrajectory.successor == null ||
            !actorInputSuccessorKeys?.has(sourceKey)))
          ? "transition_start"
          : "successor";
      const targetEndpointPhase =
        targetTrajectory === null
          ? null
          : chargeEndpoints ||
              (agentVisualInventory &&
                (targetTrajectory.successor == null ||
                  targetKey === null ||
                  !actorInputSuccessorKeys?.has(targetKey)))
            ? "transition_start"
            : "successor";
      const authorizedSource = abilityEnabled
        ? authorizedAnchor(sourceTrajectory[sourceEndpointPhase], surface)
        : null;
      const authorizedTarget =
        abilityEnabled || semanticEnabled
          ? authorizedAnchor(
              targetEndpointPhase === null
                ? null
                : targetTrajectory?.[targetEndpointPhase],
              surface,
            )
          : null;
      if (
        (abilityEnabled && authorizedSource === null) ||
        ((abilityEnabled || semanticEnabled) &&
          targetTrajectory !== null &&
          authorizedTarget === null)
      ) {
        return null;
      }
      const source = authorizedSource;
      const target = authorizedTarget;
      const routed = Boolean(abilityEnabled && source && target);
      // Look sizing and END spots (CSS px), projected only for a drawn activation
      // (a switched-off one reads no geometry). Agent POV: an END spot exists only
      // for a body this audience saw at END; the post-Charge spot is Oracle View only.
      const drawn = abilityEnabled || semanticEnabled;
      const lookSurface = drawn ? surface : null;
      const unitPx = lookSurface ? lookSurface.worldLengthToScreen(1) : null;
      const chargeStart = chargeEndpoints
        ? authorizedWorldPoint(sourceTrajectory.transition_start)
        : null;
      const chargeEnd =
        chargeEndpoints && !agentVisualInventory
          ? authorizedWorldPoint(sourceTrajectory.post_charge)
          : null;
      planned.push(
        Object.freeze({
          ...common,
          kind: "activation",
          component,
          tokenId: token.tokenId,
          token,
          impactSemantic,
          lane: component === "ultimate" ? 1 : 0,
          sourcePresentationKey: sourceKey,
          sourcePublicAgentId: sourceTrajectory.agent_public_agent_id,
          sourceIdentity,
          sourceDisclosure:
            sourceIdentity !== null
              ? "public"
              : agentVisualInventory
                ? "redacted"
                : "unavailable",
          sourceClassId,
          sourceClass: classTokenFromId(sourceClassId),
          authorizedClassMechanics,
          targetPresentationKey: targetKey,
          targetPublicAgentId: targetTrajectory?.agent_public_agent_id ?? null,
          recipientIdentity: targetIdentity,
          targetDisclosure:
            targetKey === null
              ? "target_none"
              : targetIdentity !== null
                ? "public"
                : agentVisualInventory
                  ? "redacted"
                  : "unavailable",
          source,
          target,
          sourceEndpointPhase,
          targetEndpointPhase,
          endpointPhase:
            targetEndpointPhase === null || sourceEndpointPhase === targetEndpointPhase
              ? sourceEndpointPhase
              : null,
          route: null,
          paintParts,
          spatial:
            (abilityEnabled && source !== null) || (semanticEnabled && target !== null),
          presentationKind: routed
            ? "routed"
            : !abilityEnabled && semanticEnabled && target
              ? "target_only_impact"
              : "source_local",
          sourceRadius: projectedAgentRadius(lookSurface, sceneByKey, sourceKey),
          targetRadius:
            targetKey === null
              ? 0
              : projectedAgentRadius(lookSurface, sceneByKey, targetKey),
          pixelsPerUnit: unitPx,
          unitPx,
          // Kindled: the source's Burst was on this tick (kindledSourceKeys). The
          // blow's amount has no incoming row today, so tooltips give no amount.
          sourceKindled: sourceKey !== null && kindledKeys.has(sourceKey),
          outputAmount: null,
          chargeLanding:
            chargeEnd === null
              ? null
              : authorizedAnchor(sourceTrajectory.post_charge, lookSurface),
          chargeTravel:
            chargeStart !== null && chargeEnd !== null
              ? Math.hypot(chargeEnd[0] - chargeStart[0], chargeEnd[1] - chargeStart[1])
              : null,
          sourceEnd: authorizedAnchor(sourceTrajectory.successor, lookSurface),
          targetEnd:
            targetTrajectory === null
              ? null
              : authorizedAnchor(targetTrajectory.successor, lookSurface),
          phaseStart: CHOREOGRAPHY_PHASES.v2AbilityStart,
          phaseImpact: CHOREOGRAPHY_PHASES.v2HealthResolutionStart - 20,
          phaseEnd: CHOREOGRAPHY_PHASES.v2HealthResolutionStart,
        }),
      );
      continue;
    }
    if (row.kind === "recipient_health_resolution") {
      if (trajectories === null) {
        return null;
      }
      const recipientTrajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        recipientAnchor,
        "transition_start",
      );
      const recipientKey = identifier(recipientAnchor?.presentation_key);
      const before = finiteNumber(event.transition_start_health);
      const after = finiteNumber(event.health_after_combat_resolution);
      const delta = finiteNumber(event.realized_net_health_change);
      const recipientAgent =
        recipientKey === null ? null : sceneByKey.get(recipientKey);
      const recipientIdentity = authorizedIdentitySnapshot(recipientAgent);
      const recipientWorld = authorizedWorldPoint(recipientTrajectory?.successor);
      if (
        recipientTrajectory === null ||
        recipientKey === null ||
        recipientWorld === null ||
        before === null ||
        after === null ||
        delta === null
      ) {
        return null;
      }
      const outcome = delta < 0 ? "damage" : delta > 0 ? "healing" : "unchanged";
      const paintDecision = authorizedPaintDecision(
        { kind: "net_health", outcome },
        visualFilters,
      );
      if (paintDecision === null) {
        return null;
      }
      const recipient = paintDecision.enabled ? project(recipientWorld, surface) : null;
      if (paintDecision.enabled && recipient === null) {
        return null;
      }
      const lane = paintDecision.enabled
        ? (healthLaneByKey.get(recipientKey) ?? 0)
        : null;
      if (lane !== null) {
        healthLaneByKey.set(recipientKey, lane + 1);
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "net_health",
          recipientPresentationKey: recipientKey,
          recipientPublicAgentId: recipientTrajectory.agent_public_agent_id,
          recipientIdentity,
          recipient,
          netDelta: delta,
          healthBefore: before,
          healthAfter: after,
          maximumHealth: finiteNumber(recipientAgent?.maximum_health),
          recipientRadius: paintDecision.enabled
            ? projectedAgentRadius(surface, sceneByKey, recipientKey)
            : 0,
          totalDamage: agentVisualInventory
            ? null
            : finiteNumber(event.total_effective_damage),
          totalHealing: agentVisualInventory
            ? null
            : finiteNumber(event.total_effective_healing),
          damageSources: agentVisualInventory
            ? NO_DAMAGE_SOURCES
            : (damageSourcesByRecipient.get(recipientKey) ?? NO_DAMAGE_SOURCES),
          outcome,
          lane,
          paintParts: paintDecision.paintParts,
          presentationSuppressed: !paintDecision.enabled,
          spatial: lane !== null,
          phaseStart: CHOREOGRAPHY_PHASES.v2HealthResolutionStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2CountdownAndRegenStart,
        }),
      );
      continue;
    }
    if (row.kind === "charge_phase_displacement") {
      if (trajectories === null) {
        return null;
      }
      const startAnchor = record(event.start_anchor);
      const endAnchor = record(event.end_anchor);
      const trajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        startAnchor,
        "transition_start",
      );
      const endTrajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        endAnchor,
        "post_charge",
      );
      const key = identifier(startAnchor?.presentation_key);
      const startWorld = authorizedWorldPoint(trajectory?.transition_start);
      const endWorld = authorizedWorldPoint(endTrajectory?.post_charge);
      if (
        trajectory === null ||
        endTrajectory !== trajectory ||
        key === null ||
        startWorld === null ||
        endWorld === null
      ) {
        return null;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "feed_only",
          spatial: false,
        }),
      );
      continue;
    }
    if (row.kind === "ordinary_movement_phase_displacement") {
      if (trajectories === null) {
        return null;
      }
      const trajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        event.start_anchor,
        "post_charge",
      );
      const endTrajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        event.end_anchor,
        "successor",
      );
      if (trajectory === null || endTrajectory !== trajectory) {
        return null;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "feed_only",
          spatial: false,
        }),
      );
      continue;
    }
    if (row.kind === "action_rejected") {
      const actorAnchor = record(event.actor_anchor);
      planned.push(
        Object.freeze({
          ...common,
          kind: "feed_only",
          feedKind: "rejected_action",
          actorPresentationKey: actorAnchor?.presentation_key ?? null,
          actorPublicAgentId:
            actorAnchor?.public_agent_id ??
            event.actor_identity?.public_agent_id ??
            null,
          component: event.rejection_component,
          presentationSuppressed: true,
          spatial: false,
        }),
      );
      continue;
    }
    if (row.kind === "agent_left_combat") {
      if (trajectories === null) {
        return null;
      }
      const trajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        agentAnchor,
        "successor",
      );
      const recipientWorld = authorizedWorldPoint(trajectory?.successor);
      const recipientAgent =
        trajectory === null ? null : sceneByKey.get(trajectory.agent_presentation_key);
      const recipientIdentity = authorizedIdentitySnapshot(recipientAgent);
      const maximumHealth = finiteNumber(recipientAgent?.maximum_health);
      const regenerationFraction = finiteNumber(
        recipientAgent?.out_of_combat_health_regeneration_fraction_per_step,
      );
      const regenerationPerTick =
        maximumHealth === null || regenerationFraction === null
          ? null
          : maximumHealth * regenerationFraction;
      if (
        trajectory === null ||
        recipientWorld === null ||
        recipientIdentity === null ||
        regenerationPerTick === null ||
        !Number.isFinite(regenerationPerTick)
      ) {
        return null;
      }
      const paintParts = authorizedPaintParts(
        { kind: "status_lifecycle", lifecycle: "expired" },
        visualFilters,
      );
      if (paintParts === null) {
        return null;
      }
      const enabled = Object.values(paintParts).some((part) => part === true);
      const recipient = enabled ? project(recipientWorld, surface) : null;
      if (enabled && recipient === null) {
        return null;
      }
      const status = resolveVisualToken("status", "in_combat", event);
      const lifecycle = resolveVisualToken("lifecycle", "expired", event);
      planned.push(
        Object.freeze({
          ...common,
          kind: "status_lifecycle",
          tokenId: status.tokenId,
          token: status,
          lifecycle: lifecycle.tokenId,
          lifecycleToken: lifecycle,
          recipientPresentationKey: trajectory.agent_presentation_key,
          recipientPublicAgentId: trajectory.agent_public_agent_id,
          recipientIdentity,
          outOfCombatRegenerationPerTick: regenerationPerTick,
          recipient,
          sourcePresentationKey: null,
          sourcePublicAgentId: null,
          applicationSources: Object.freeze([]),
          durationBefore: 1,
          durationAfter: 0,
          lane: enabled ? 0 : null,
          laneCount: enabled ? 1 : 0,
          statusLayoutOrder: row.payload.ordinal,
          atomicEventIds: Object.freeze([row.id]),
          applicationEventIds: Object.freeze([]),
          paintParts,
          presentationSuppressed: !enabled,
          spatial: enabled,
          phaseStart: CHOREOGRAPHY_PHASES.v2StatusStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2ShieldStart,
        }),
      );
      continue;
    }
    if (row.kind === "combat_countdown_reset") {
      planned.push(
        Object.freeze({
          ...common,
          kind: "feed_only",
          spatial: false,
        }),
      );
      continue;
    }
    if (row.kind === "respawn_wave_occurred") {
      const waveIdentity = authorizedTeamWaveIdentity(event.team_anchor, teamSide);
      const paintDecision = authorizedPaintDecision(
        { kind: "semantic_pulse", cueSemantic: row.kind },
        visualFilters,
      );
      if (waveIdentity === null || paintDecision === null) {
        return null;
      }
      // The team wave is page chrome (the scorebug's Wave Chip), never an arena
      // mark, so it has no anchor and reserves no arena box.
      planned.push(
        Object.freeze({
          ...common,
          kind: "semantic_pulse",
          cueSemantic: row.kind,
          anchor: null,
          teamIndex: waveIdentity.teamIndex,
          teamId: waveIdentity.teamId,
          teamSide: waveIdentity.teamSide,
          label: waveIdentity.label,
          paintParts: paintDecision.paintParts,
          presentationSuppressed: !paintDecision.enabled,
          spatial: false,
          persistent: paintDecision.enabled,
          phaseStart: CHOREOGRAPHY_PHASES.v2RespawnWaveStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2RespawnStart,
        }),
      );
      continue;
    }
    if (row.kind === "spawn_shield_expired") {
      let presentationAnchor = agentAnchor;
      if (trajectories !== null) {
        const trajectory = trajectoryForAuthorizedAnchor(
          trajectories,
          agentAnchor,
          "successor",
        );
        if (trajectory === null) {
          return null;
        }
        presentationAnchor = trajectory.successor;
      }
      const paintDecision = authorizedPaintDecision(
        {
          kind: "status_lifecycle",
          lifecycle: "expired",
          cueSemantic: row.kind,
        },
        visualFilters,
      );
      const recipientWorld = authorizedWorldPoint(presentationAnchor);
      const recipientKey = identifier(presentationAnchor?.presentation_key);
      const recipientAgent =
        recipientKey === null ? null : sceneByKey.get(recipientKey);
      const recipientIdentity = authorizedIdentitySnapshot(recipientAgent);
      if (
        paintDecision === null ||
        recipientWorld === null ||
        recipientIdentity === null
      ) {
        return null;
      }
      const recipient = paintDecision.enabled ? project(recipientWorld, surface) : null;
      if (paintDecision.enabled && recipient === null) {
        return null;
      }
      const lifecycle = resolveVisualToken("lifecycle", "expired", event);
      planned.push(
        Object.freeze({
          ...common,
          kind: "status_lifecycle",
          cueSemantic: row.kind,
          tokenId: SPAWN_SHIELD_STATUS_TOKEN.tokenId,
          token: SPAWN_SHIELD_STATUS_TOKEN,
          lifecycle: lifecycle.tokenId,
          lifecycleToken: lifecycle,
          recipientPresentationKey: presentationAnchor?.presentation_key ?? null,
          recipientPublicAgentId: presentationAnchor?.public_agent_id ?? null,
          recipientIdentity,
          recipient,
          // The body's radius in CSS px, so the Ward Lift ring fits it.
          anchorRadius:
            paintDecision.enabled && surface && recipientAgent
              ? surface.worldLengthToScreen(finiteNumber(recipientAgent.radius) ?? 0.5)
              : null,
          sourcePresentationKey: null,
          sourcePublicAgentId: null,
          applicationSources: Object.freeze([]),
          durationBefore: null,
          durationAfter: 0,
          lane: paintDecision.enabled ? 0 : null,
          laneCount: paintDecision.enabled ? 1 : 0,
          statusLayoutOrder: row.payload.ordinal,
          atomicEventIds: Object.freeze([row.id]),
          applicationEventIds: Object.freeze([]),
          paintParts: paintDecision.paintParts,
          presentationSuppressed: !paintDecision.enabled,
          spatial: paintDecision.enabled,
          phaseStart: CHOREOGRAPHY_PHASES.v2ShieldStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2RespawnWaveStart,
        }),
      );
      continue;
    }
    if (
      [
        "health_regenerated",
        "cooldown_started",
        "cooldown_ready",
        "agent_died",
        "agent_respawned",
      ].includes(row.kind)
    ) {
      const anchor = agentAnchor ?? recipientAnchor;
      let presentationAnchor = anchor;
      if (
        row.kind === "health_regenerated" ||
        row.kind === "cooldown_started" ||
        row.kind === "cooldown_ready"
      ) {
        if (trajectories === null) {
          return null;
        }
        const trajectory = trajectoryForAuthorizedAnchor(
          trajectories,
          agentAnchor,
          "transition_start",
        );
        if (trajectory === null) {
          return null;
        }
        presentationAnchor = trajectory.successor;
      } else if (trajectories !== null) {
        const trajectory = trajectoryForAuthorizedAnchor(
          trajectories,
          anchor,
          "successor",
        );
        if (trajectory === null) {
          return null;
        }
        presentationAnchor = trajectory.successor;
      }
      const kind =
        row.kind === "health_regenerated" ? "regeneration" : "semantic_pulse";
      const paintDecision = authorizedPaintDecision(
        { kind, cueSemantic: row.kind },
        visualFilters,
      );
      const presentationWorld = authorizedWorldPoint(presentationAnchor);
      if (paintDecision === null || presentationWorld === null) {
        return null;
      }
      const projectedAnchor = paintDecision.enabled
        ? project(presentationWorld, surface)
        : null;
      const presentationKey = identifier(presentationAnchor?.presentation_key);
      const presentationAgent =
        presentationKey === null ? null : sceneByKey.get(presentationKey);
      const recipientIdentity = authorizedIdentitySnapshot(presentationAgent);
      if (
        recipientIdentity === null ||
        (paintDecision.enabled && projectedAnchor === null)
      ) {
        return null;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind,
          cueSemantic: row.kind,
          anchor: projectedAnchor,
          recipient: row.kind === "health_regenerated" ? projectedAnchor : null,
          agentPresentationKey: presentationAnchor?.presentation_key ?? null,
          agentPublicAgentId: presentationAnchor?.public_agent_id ?? null,
          recipientPresentationKey: presentationAnchor?.presentation_key ?? null,
          recipientPublicAgentId: presentationAnchor?.public_agent_id ?? null,
          recipientIdentity,
          value:
            row.kind === "health_regenerated"
              ? finiteNumber(event.actual_health_regenerated)
              : null,
          // The body's radius in CSS px, so the Fall Ring and Homecoming fit it.
          anchorRadius:
            paintDecision.enabled && surface && presentationAgent
              ? surface.worldLengthToScreen(
                  finiteNumber(presentationAgent.radius) ?? 0.5,
                )
              : null,
          // The host's double-scoring flag (Agent POV: sent only for a seen fall).
          homeFall: row.kind === "agent_died" && event.home_fall === true,
          paintParts: paintDecision.paintParts,
          presentationSuppressed: !paintDecision.enabled,
          persistent:
            paintDecision.enabled &&
            (row.kind === "agent_died" || row.kind === "agent_respawned"),
          spatial: paintDecision.enabled,
          phaseStart: CHOREOGRAPHY_PHASES.v2CountdownAndRegenStart,
          phaseEnd: CHOREOGRAPHY_PHASES.total,
        }),
      );
      continue;
    }
    planned.push(
      Object.freeze({
        ...common,
        kind: "feed_only",
        spatial: false,
        noncausal: row.vocabulary !== "event",
      }),
    );
  }
  const falls = authorizedFalls({
    presentation,
    rows,
    sceneByKey,
    trajectories,
    schedule,
    agentVisual: agentVisualInventory,
    transitionId,
  });
  // Death Announcer cards. Oracle View keeps the match summary's credited deaths
  // (Oracle facts with no world position). Agent POV never reads
  // match_summary.deaths: its cards hold only the falls the team saw and the
  // killers it may name, so a hidden death adds no card and no time.
  const deaths = agentVisualInventory ? [] : array(presentation.match_summary?.deaths);
  /** @type {ReadonlyArray<{member: Readonly<Record<string, any>>, sideTeam: number}>} */
  const announcedDeaths = agentVisualInventory
    ? falls.map((fall) => ({
        member: Object.freeze({
          ...fall.victim,
          killingTeamId: fall.killingTeamId,
          contributors: fall.killers,
          points: fall.points,
          homeFall: fall.homeFall === true,
        }),
        sideTeam: fall.killingTeamId ?? fall.victim.teamId,
      }))
    : deaths.map((death) => {
        /**
         * Format a death-HUD public identity using the presentation's display-ID
         * mapping.
         *
         * agent supplies already authorized match-summary identity fields. Return the
         * shared canonical identity result; it adds no world position or actor-input
         * permission.
         */
        const identity = (/** @type {Record<string, any>} */ agent) =>
          canonicalAgentIdentity({
            ...agent,
            display_agent_id: authorizedPresentationAgentDisplayId(
              presentation,
              agent.public_agent_id,
            ),
          });
        return {
          member: Object.freeze({
            ...identity(death),
            killingTeamId: death.killing_team_id ?? null,
            points: death.home_fall === true ? 2 : 1,
            homeFall: death.home_fall === true,
            contributors:
              death.contributors == null
                ? null
                : Object.freeze(death.contributors.map(identity)),
          }),
          sideTeam: death.killing_team_id ?? death.team_id,
        };
      });
  // The death announcement is a data row for the feed and its explanation; the
  // Fallen panel in the side rail draws the falls, so it has no arena anchor
  // and reserves no arena box.
  if (
    announcedDeaths.length > 0 &&
    isVisualPaintPartEnabled(visualFilters, {
      surface: "transient",
      kind: "death_announcement",
    })
  ) {
    for (const sideId of [1, 2]) {
      const members = announcedDeaths
        .filter((entry) => entry.sideTeam === sideId)
        .map((entry) => entry.member);
      if (members.length === 0) continue;
      // Historical deaths remain neutral when attribution was not recorded.
      // Mixed evidence shares a side bucket, never a third card or a queue.
      const teamId = members.every((member) => member.killingTeamId === sideId)
        ? sideId
        : null;
      const side = teamSide(sideId);
      planned.push(
        Object.freeze({
          eventId: `${transitionId}:death-announcement:${sideId}`,
          eventType: "agent_died",
          kind: "semantic_pulse",
          cueSemantic: "death_announcement",
          teamId,
          teamIndex: teamId === null ? null : teamId - 1,
          teamSide: side,
          sideId,
          label:
            teamId === null ? "Agent Deaths" : `Team ${teamId === 1 ? "A" : "B"} Kills`,
          members: Object.freeze(members),
          anchor: null,
          spatial: false,
          persistent: false,
        }),
      );
    }
  }
  // Rows built from several events (Fallen feed, heralds, Home Fall, Last Seen,
  // Focus Tally) live beside plan.events, which keeps one row per incoming event.
  const familyRows = derivedFamilyEvents({
    falls,
    rows,
    sceneByKey,
    trajectories,
    schedule,
    surface,
    visualFilters,
    agentVisual: agentVisualInventory,
    transitionId,
    teamSide,
    // The frame's own host count: Agent POV frames carry only their team's.
    streaks: presentation.champion_streaks,
    // The public roster's team of each public ID (allowed in Agent POV, as
    // ruling L10 allows the public episode limit): which team a champion is on
    // is a match setting every audience already sees in the scorebug's
    // portraits, never a hidden fact about where it is or what it does.
    teamByPublicId: new Map(
      publicRosterIdentities(presentation).map((row) => [
        row.publicAgentId,
        row.teamId,
      ]),
    ),
    // The herald row's moment facts: the frame's own match_moments host fact,
    // the public scoreboard, and this audience's own blows and bodies.
    heraldFacts: heraldTickFacts({
      presentation,
      falls,
      rows,
      sceneByKey,
      trajectories,
      agentVisual: agentVisualInventory,
    }),
  });
  const paintFiltered = applyAuthorizedVisualFilters(planned, visualFilters);
  const laidOut = layoutCrossPhaseEvents(paintFiltered, surface, sceneByKey);
  if (laidOut === null) {
    return null;
  }
  const scheduled = scheduleChoreography(laidOut, schedule);
  const bodies = planBodies(
    sceneByKey,
    trajectories,
    schedule,
    array(record(presentation.scene)?.map?.obstacles),
  );
  const budgeted = applyPlanBudget(scheduled.events, familyRows, bodies, schedule);
  const events = budgeted.events;
  return Object.freeze({
    epochKey: JSON.stringify([presentation.session_id, transitionId]),
    authorizationKey: JSON.stringify([
      presentation.session_id,
      presentation.authority_epoch,
      presentation.presentation_kind,
      presentation.recipient_presentation_key,
    ]),
    // The audience key names who is watching (session, view kind and Agent POV
    // recipient) without the per-frame authority epoch, so a step glide can tell
    // whether two frames show the same audience.
    audienceKey: JSON.stringify([
      presentation.session_id,
      presentation.presentation_kind,
      presentation.recipient_presentation_key ?? null,
    ]),
    audience,
    // Agent POV hashes only its own rows: a hidden death never changes this.
    fingerprint: `${events.length}:${hashText(JSON.stringify([...rows.map(({ payload }) => payload), ...deaths]))}`,
    paintKey: visualFilterPaintKey(visualFilters),
    transitionId,
    simulatorStep,
    phases: scheduled.phases,
    beats: schedule,
    events,
    derivedEvents: budgeted.derivedEvents,
    bodies,
    camera: planCamera(bodies, rows, trajectories, audience),
    bounds: budgeted.bounds,
  });
}

/**
 * The plan-wide resource caps.
 *
 * nodes, animations and persistentNodes cap the painter's SVG nodes, its event
 * animations and the nodes a persistent cue keeps after settling: 1024 each
 * (owner, 4 Oct 2026; 512 before). The plan counts each row's largest possible
 * drawing, so a busy tick could ask for more than 512 while it really draws far
 * fewer, and the budget then dropped its Herald Card and Fallen rows (map 47
 * ticks 105, 110, 225 and 245 asked for 535 to 637). At 1024 they keep them; the
 * measured cost is a few more arena nodes and no slower frames. The recorded worst
 * case of real recordings and every browser fixture frame fit under the caps, see
 * tests/plan-budget.test.js.
 * overrideAnimations caps the choreographer's own WAAPI overrides on durable nodes
 * (body glides, beat holds, cue fades and the camera ease).
 */
export const PLAN_RESOURCE_CAPS = Object.freeze({
  nodes: 1024,
  animations: 1024,
  persistentNodes: 1024,
  overrideAnimations: 96,
});

/**
 * The allowance of a drawn plan row whose kind has no look module of its own: 30
 * nodes, 4 animations and 7 kept nodes, the shared earlier per-event allowance.
 */
const UNDRAWN_ROW_BUDGET = Object.freeze({
  nodesPerRecord: 30,
  animationsPerRecord: 4,
  persistentNodesPerRecord: 7,
});

/**
 * The allowance of a status-change row that no look draws: 0 nodes, 0 animations
 * and 0 kept nodes. V5 Rune Zones shows a status that starts, ends, breaks or is
 * cleared on the body itself (its status marks), so statuses.js records() gives
 * nothing for such a "status_lifecycle" row and the painter draws no group for it.
 * Charging it an allowance made busy ticks drop herald cards for nothing.
 */
const UNDRAWN_STATUS_CHANGE_BUDGET = Object.freeze({
  nodesPerRecord: 0,
  animationsPerRecord: 0,
  persistentNodesPerRecord: 0,
});

/**
 * Optional derived families, in the order they are dropped when a tick is over budget.
 * Today's event kinds are never dropped: over budget drops optional sub-parts and
 * never blanks the tick.
 */
const OPTIONAL_FAMILY_DROP_ORDER = Object.freeze([
  "focus_tally",
  "last_seen",
  "herald_card",
  "home_fall",
  "fallen_feed_row",
]);

/**
 * Durable overrides one beat may need: per seen fall (life, ash and token before
 * and after), per seen return (17: the body and its panel row before and after,
 * plus the set-down waits of its Command Ring, up to 4 Order Studs, 2 thread
 * lines, 3 Step Ghost circles and up to 3 Ranges rings), and
 * the score and HUD holds of one tick, plus the cue fades and the camera ease.
 * perBodyLook covers each body's look-owned overrides (a Dim Shift's second
 * animation, its Start Spot or afterimage, its hit flash and shiver, its Impact
 * life change); sightAndDim covers the Agent POV sight veil states and glints and
 * Death Dimming.
 */
const OVERRIDE_BUDGET = Object.freeze({
  perFall: 6,
  perReturn: 17,
  scoreAndHud: 8,
  cueFadesAndCamera: 4,
  perBodyLook: 4,
  sightAndDim: 8,
});

/**
 * Return the budget record of the look module that draws one plan row.
 *
 * event is one row. The mapping (kept in step with the painter's dispatch):
 * Basic activation -> basics.js, Ultimate activation -> ultimates.js, net health
 * and regeneration -> health-text.js, a Spawn Shield expiry (the Ward Lift,
 * cueSemantic "spawn_shield_expired") and cooldown pulses -> statuses.js, every
 * other status lifecycle row -> UNDRAWN_STATUS_CHANGE_BUDGET (no look draws it), deaths, Death Announcer cards, respawn waves, returns and Fallen
 * feed rows -> requiems.js (a return adds its waystone Call, spawns.js CALL_BUDGET), Home Fall -> red-zone.js, herald cards -> heralds.js
 * (the row's own `budget`, heralds.js rowBudget for the card it names, else BUDGET),
 * Last Seen rings -> fog.js, Focus Tally -> focus-tally.js; every other row uses
 * UNDRAWN_ROW_BUDGET. Each record has nodesPerRecord, animationsPerRecord and
 * persistentNodesPerRecord. Pure.
 *
 * @param {Record<string, any>} event
 */
export function planEventBudget(event) {
  switch (event.kind) {
    case "activation":
      return event.component === "ultimate" ? ULTIMATE_BUDGET : BASIC_BUDGET;
    case "net_health":
    case "regeneration":
      return HEALTH_TEXT_BUDGET;
    case "status_lifecycle":
      // Only the Ward Lift draws a status change off the body marks.
      return event.cueSemantic === "spawn_shield_expired"
        ? STATUS_BUDGET
        : UNDRAWN_STATUS_CHANGE_BUDGET;
    case "semantic_pulse":
      if (event.cueSemantic === "agent_respawned") {
        // The return's Homecoming plus its waystone's Call.
        return Object.freeze({
          nodesPerRecord:
            REQUIEM_BUDGET.nodesPerRecord + WAYSTONE_CALL_BUDGET.nodesPerRecord,
          animationsPerRecord:
            REQUIEM_BUDGET.animationsPerRecord +
            WAYSTONE_CALL_BUDGET.animationsPerRecord,
          // A return is a kept (persistent) row, and its Call is drawn in the same
          // kept group, so every Call node counts as kept too.
          persistentNodesPerRecord:
            REQUIEM_BUDGET.persistentNodesPerRecord +
            WAYSTONE_CALL_BUDGET.nodesPerRecord,
        });
      }
      return event.cueSemantic === "cooldown_ready" ||
        event.cueSemantic === "cooldown_started"
        ? STATUS_BUDGET
        : REQUIEM_BUDGET;
    case "fallen_feed_row":
      return REQUIEM_BUDGET;
    case "home_fall":
      return RED_ZONE_BUDGET;
    case "herald_card":
      // The row's own card allowance (heralds.js rowBudget), set by the plan.
      return record(event.budget) ?? HERALD_BUDGET;
    case "last_seen":
      return FOG_BUDGET;
    case "focus_tally":
      return FOCUS_TALLY_BUDGET;
    default:
      return UNDRAWN_ROW_BUDGET;
  }
}

/**
 * Count the plan's resource demand from its rows and bodies.
 *
 * events are plan.events (one row per incoming event, plus Death Announcer cards);
 * derivedEvents are plan.derivedEvents (the derived rows); bodies the plan's body
 * records; schedule its beat schedule. Return the uncapped sums:
 * - nodes: 3 roots plus the node allowance of every drawn (spatial) row. The painter
 *   skips rows that are not spatial (feed-only, filtered off or suppressed), so they
 *   add nothing; before the beat clock every row counted, which pinned busy ticks at the cap;
 * - animations: 3 clocks plus each drawn row's animation allowance;
 * - persistentNodes: the kept nodes of every drawn persistent row;
 * - overrideAnimations: per body one glide plus its look overrides
 *   (OVERRIDE_BUDGET.perBodyLook), plus OVERRIDE_BUDGET for the seen falls,
 *   returns, score and HUD, cue fades and camera, and the sight veil and dimming.
 * Pure.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 * @param {ReadonlyArray<Record<string, any>>} derivedEvents
 * @param {ReadonlyArray<PlanBody>} bodies
 * @param {BeatSchedule} schedule
 */
export function planResourceDemand(events, derivedEvents, bodies, schedule) {
  let nodes = 3;
  let animations = 3;
  let persistentNodes = 0;
  for (const event of [...events, ...derivedEvents]) {
    // The painter draws only spatial rows, so only they can add nodes.
    if (!event.spatial) continue;
    const budget = planEventBudget(event);
    nodes += budget.nodesPerRecord;
    animations += budget.animationsPerRecord;
    if (event.persistent) {
      persistentNodes += budget.persistentNodesPerRecord;
    }
  }
  const overrideAnimations =
    bodies.length * (1 + OVERRIDE_BUDGET.perBodyLook) +
    OVERRIDE_BUDGET.perFall * schedule.falls.length +
    OVERRIDE_BUDGET.perReturn * schedule.returns.length +
    OVERRIDE_BUDGET.scoreAndHud +
    OVERRIDE_BUDGET.cueFadesAndCamera +
    OVERRIDE_BUDGET.sightAndDim;
  return Object.freeze({ nodes, animations, persistentNodes, overrideAnimations });
}

/**
 * Keep a tick inside PLAN_RESOURCE_CAPS by dropping optional derived families first.
 *
 * events, derivedEvents, bodies and schedule are as for planResourceDemand. While
 * the demand is over a cap, the drawn rows of the next optional family in
 * OPTIONAL_FAMILY_DROP_ORDER become data only (spatial false, droppedForBudget
 * true); plan.events rows are never dropped, so the tick is never blanked. Return
 * frozen {events, derivedEvents, bounds}, where bounds is the demand clipped to the
 * caps (nodes, animations, persistentNodes, overrideAnimations). A tick whose
 * incoming events alone exceed a cap keeps the capped bound, exactly as before the beat clock.
 * buildChoreographyPlan calls it; it is exported for the budget tests. Pure.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 * @param {ReadonlyArray<Record<string, any>>} derivedEvents
 * @param {ReadonlyArray<PlanBody>} bodies
 * @param {BeatSchedule} schedule
 */
export function applyPlanBudget(events, derivedEvents, bodies, schedule) {
  let derived = derivedEvents;
  let demand = planResourceDemand(events, derived, bodies, schedule);
  /**
   * Return whether a demand is over the node, animation or kept-node cap. Pure.
   *
   * @param {ReturnType<typeof planResourceDemand>} value
   */
  const over = (value) =>
    value.nodes > PLAN_RESOURCE_CAPS.nodes ||
    value.animations > PLAN_RESOURCE_CAPS.animations ||
    value.persistentNodes > PLAN_RESOURCE_CAPS.persistentNodes;
  for (const family of OPTIONAL_FAMILY_DROP_ORDER) {
    if (!over(demand)) break;
    derived = derived.map((event) =>
      event.kind === family && event.spatial
        ? Object.freeze({ ...event, spatial: false, droppedForBudget: true })
        : event,
    );
    demand = planResourceDemand(events, derived, bodies, schedule);
  }
  return Object.freeze({
    events,
    derivedEvents: Object.freeze([...derived]),
    bounds: Object.freeze({
      nodes: Math.min(demand.nodes, PLAN_RESOURCE_CAPS.nodes),
      animations: Math.min(demand.animations, PLAN_RESOURCE_CAPS.animations),
      persistentNodes: Math.min(
        demand.persistentNodes,
        PLAN_RESOURCE_CAPS.persistentNodes,
      ),
      overrideAnimations: Math.min(
        demand.overrideAnimations,
        PLAN_RESOURCE_CAPS.overrideAnimations,
      ),
    }),
  });
}

/**
 * Build a frozen combat explanation from an already-authorized frame.
 *
 * frame must satisfy the normalized presentation envelope check; otherwise return
 * null. surface defaults to null and supplies screen projection, viewport bounds
 * and protected regions. visualFilters defaults to DEFAULT_VISUAL_FILTER_STATE.
 * Return a plan, or null when required joins/projection are unavailable. The plan
 * carries: epochKey, authorizationKey, audienceKey (session, view kind and Agent
 * POV recipient), audience, fingerprint, paintKey, transitionId, simulatorStep;
 * phases (event windows in real ms, see scheduleChoreography); beats (the tick's
 * beatSchedule); events (filtered ordered rows: one per incoming event, plus Death
 * Announcer cards); derivedEvents (rows built from several events: Fallen feed,
 * herald, Home Fall, Last Seen and Focus Tally; see derivedFamilyEvents) with their own
 * times; bodies (planBodies: START, post-Charge and END spots for glides); camera
 * (planCamera: the camera's input); and bounds (nodes, animations,
 * persistentNodes and overrideAnimations, see applyPlanBudget).
 *
 * Only incoming presentation facts are used. No outgoing draft/accepted action is
 * read as an event, no hidden geometry is inferred, and inputs are unchanged.
 * Agent POV: every record, time and bound comes from the seen scene and
 * visual_events, so changing a hidden champion's researcher_space or match_summary
 * row changes nothing in the plan. The shared facts read are the public
 * roster's team of each public ID (the identity directory), which colors a
 * Last Seen ring in its team's light and tells Team Wiped who is on a team,
 * and the public scoreboard of the herald cards (scores before and after the
 * tick, First To, the tick and the draw tick); every audience sees both in
 * the scorebug, and neither is a fact about where a champion is or what it
 * does. The herald row's history facts come from the frame's own per-view
 * match_moments host fact. Allocator or dependent validation errors can
 * propagate. Render policy is owned by the controller/painter; this builder does
 * not accept or change that policy.
 *
 * @param {unknown} frame
 * @param {ProjectionSurface | null} surface
 * @param {Record<string, boolean>} visualFilters
 * @returns {Readonly<ChoreographyPlan> | null}
 */
export function buildChoreographyPlan(
  frame,
  surface = null,
  visualFilters = DEFAULT_VISUAL_FILTER_STATE,
) {
  if (isAuthorizedPresentationFrame(frame)) {
    return buildAuthorizedPresentationChoreographyPlan(frame, surface, visualFilters);
  }
  return null;
}

/**
 * @typedef {Readonly<{
 *   presentationKey: string,
 *   publicAgentId: string,
 *   before: number | null,
 *   after: number | null,
 *   at: number,
 *   held: boolean,
 *   shown: number | null,
 * }>} HeldLife
 * @typedef {Readonly<{
 *   publicAgentId: string,
 *   seen: boolean,
 *   at: number,
 *   back: boolean,
 * }>} HeldReturn
 */

/**
 * Tell the HUD and panels which value to show at one moment of the beat clock.
 *
 * beatView is the one owner of held values (the dramatic hold): life reaches 0 at a
 * champion's Fall beat, any other life change shows at Impact, the score changes at
 * the Fall beat and a returning champion comes back at the Return beat. Everything is read from the current frame only;
 * there is no memory of an earlier frame, so a seek, a skip or Oracle -> Agent POV
 * -> Oracle gives the same answer as a fresh render.
 *
 * frame is a normalized presentation frame (anything else returns null). stage is
 * "start" (before any beat), "end" (the END state, the default and what the DOM
 * holds) or a number: real ms on the beat clock at playback rate 1 in normal motion
 * (for reduced motion divide by the schedule's reducedScale first).
 * options.visualFilters is the page's filter state (default: switch defaults); it
 * decides whether there is a Fall or Return beat, exactly as the plan does.
 *
 * Returns a frozen record:
 * - audience: "researcher" (Oracle View) or "agent_pov"; schedule: the beatSchedule.
 * - life: one entry per fall the audience saw (Oracle: latest_events; Agent POV:
 *   visual_events): before is the life at the start of the tick
 *   (transition_start_health of its health resolution, or null when the frame has
 *   none), after its life after the tick (0 for a fall), at its Fall beat time,
 *   held whether stage is before at, and shown the value to draw now. A death the
 *   team did not see has no entry and no time.
 * - hits: one entry per other life change the audience saw this tick (a hit or a
 *   heal that did not end in a fall): a recipient_health_resolution row of its own
 *   rows whose transition_start_health and health_after_combat_resolution are both
 *   known and differ. before and after are those two values, at is the Impact beat
 *   (BEAT_CLOCK.impactBlow on this schedule's clock), held and shown as for life.
 *   A champion appears once (its first row); a change the audience has no row for
 *   has no entry, so in Agent POV nothing unseen is ever held.
 * - score: before and after (Team A then Team B; from the adapter's
 *   authorizedPresentationMatchBeatFacts: match_summary.previous_scores and scores,
 *   before equals after when the host sent no previous_scores; both null without a
 *   match summary), changed, at (the Fall beat start, the same on every tick so it
 *   never reveals a hidden death in Agent POV), held and shown.
 * - returns: one entry per returning champion from the same adapter facts (Oracle
 *   View: match_summary.returned_agent_ids; Agent POV: only the agent_respawned rows
 *   of its visual_events, so an unseen return is never listed). seen says whether
 *   this audience has the return's event; at is its Return beat time when seen,
 *   otherwise the end of the beats the audience saw; back says whether it shows as
 *   returned at stage.
 * Agent POV: the score is the public scoreboard; nothing here is drawn on the arena,
 * and every time depends only on seen events. Pure: reads the frame, writes nothing.
 *
 * @param {unknown} frame
 * @param {"start" | "end" | number} [stage]
 * @param {{visualFilters?: unknown}} [options]
 * @returns {Readonly<{
 *   audience: "researcher" | "agent_pov",
 *   schedule: BeatSchedule,
 *   life: ReadonlyArray<HeldLife>,
 *   hits: ReadonlyArray<HeldLife>,
 *   score: Readonly<{
 *     before: ReadonlyArray<number> | null,
 *     after: ReadonlyArray<number> | null,
 *     changed: boolean,
 *     at: number,
 *     held: boolean,
 *     shown: ReadonlyArray<number> | null,
 *   }>,
 *   returns: ReadonlyArray<HeldReturn>,
 * }> | null}
 */
export function beatView(frame, stage = "end", options = {}) {
  if (!isAuthorizedPresentationFrame(frame)) return null;
  const audience = authorizedPresentationAudience(frame);
  if (audience === null) return null;
  const time =
    stage === "start"
      ? Number.NEGATIVE_INFINITY
      : stage === "end"
        ? Number.POSITIVE_INFINITY
        : typeof stage === "number" && !Number.isNaN(stage)
          ? stage
          : Number.POSITIVE_INFINITY;
  const rows = authorizedPresentationIncomingRows(frame);
  const schedule = beatSchedule({
    falls: beatSubjects(rows, "agent_died"),
    returns: beatSubjects(rows, "agent_respawned"),
    visualFilters: options.visualFilters ?? DEFAULT_VISUAL_FILTER_STATE,
  });
  const scene = authorizedPresentationSceneView(frame);
  const sceneByKey = new Map(
    array(scene?.agents).map((agent) => [
      identifier(record(agent)?.presentation_key),
      agent,
    ]),
  );
  const life = Object.freeze(
    schedule.falls.map((entry) => {
      const resolution = rows.find(
        (row) =>
          row.kind === "recipient_health_resolution" &&
          row.vocabulary === "event" &&
          identifier(record(row.payload?.recipient_anchor)?.presentation_key) ===
            entry.presentationKey,
      );
      const before = finiteNumber(resolution?.payload?.transition_start_health);
      const after =
        finiteNumber(resolution?.payload?.health_after_combat_resolution) ??
        finiteNumber(sceneByKey.get(entry.presentationKey)?.current_health);
      const held = time < entry.at;
      return Object.freeze({
        presentationKey: entry.presentationKey,
        publicAgentId: entry.publicAgentId,
        before,
        after,
        at: entry.at,
        held,
        shown: held && before !== null ? before : after,
      });
    }),
  );
  const fallKeys = new Set(schedule.falls.map((entry) => entry.presentationKey));
  const impactAt = beatRealTime(BEAT_CLOCK.impactBlow, schedule.glide, schedule.lead);
  /** @type {Set<string>} */
  const hitKeys = new Set();
  const hits = Object.freeze(
    rows.flatMap((row) => {
      if (row.kind !== "recipient_health_resolution" || row.vocabulary !== "event") {
        return [];
      }
      const anchor = record(row.payload?.recipient_anchor);
      const presentationKey = identifier(anchor?.presentation_key);
      const publicAgentId = identifier(anchor?.public_agent_id);
      if (
        presentationKey === null ||
        publicAgentId === null ||
        fallKeys.has(presentationKey) ||
        hitKeys.has(presentationKey)
      ) {
        return [];
      }
      const before = finiteNumber(row.payload?.transition_start_health);
      const after = finiteNumber(row.payload?.health_after_combat_resolution);
      if (before === null || after === null || before === after) return [];
      hitKeys.add(presentationKey);
      const held = time < impactAt;
      return [
        Object.freeze({
          presentationKey,
          publicAgentId,
          before,
          after,
          at: impactAt,
          held,
          shown: held ? before : after,
        }),
      ];
    }),
  );
  // The adapter owns reading the held facts for this audience; beatView owns when
  // each one shows.
  const facts = authorizedPresentationMatchBeatFacts(frame);
  const scoreBefore = facts ? facts.scores_before : null;
  const scoreAfter = facts ? facts.scores_after : null;
  const changed = facts?.score_changed === true;
  const scoreHeld = changed && time < schedule.scoreAt;
  const seenReturns = new Map(
    schedule.returns.map((entry) => [entry.publicAgentId, entry]),
  );
  const returns = Object.freeze(
    (facts?.returned_public_agent_ids ?? []).map((publicAgentId) => {
      const seen = seenReturns.get(publicAgentId);
      const at = seen?.at ?? schedule.explanationEnd;
      return Object.freeze({
        publicAgentId,
        seen: seen !== undefined,
        at,
        back: time >= at,
      });
    }),
  );
  return Object.freeze({
    audience,
    schedule,
    life,
    hits,
    score: Object.freeze({
      before: scoreBefore,
      after: scoreAfter,
      changed,
      at: schedule.scoreAt,
      held: scoreHeld,
      shown: scoreHeld ? scoreBefore : scoreAfter,
    }),
    returns,
  });
}
