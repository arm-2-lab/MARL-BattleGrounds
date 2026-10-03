/**
 * @file The Basic Attacks look, Cleave And Bloom: how every Basic (Arcane
 * Blast, Shield Bash, Serrated Arrows, Backstab and Lesser Heal) is drawn in
 * the BattleClient, the DevClient combat screen and the Replay Viewer.
 *
 * What it draws (all in the class colour of the champion that struck):
 * - Close blows are forward cleaves, filled arcs that leave the attacker's rim
 *   and end just outside the target's rim: the Warrior's broad bronze sweep
 *   with a Forge White edge and three sparks, the Rogue's two thin jade flicks,
 *   the Mage's starlit arc, the Hunter's serrated swipe and the Priest's soft
 *   gold arc at 58%.
 * - Further away, Arcane Blast is one 2 px violet bolt with a Starcore centre
 *   line that lands in a white eight-point star; Serrated Arrows is a 1.5 px
 *   green streak with fletching bars that lands in the Shard Fan; Lesser Heal
 *   is a gold ribbon that lands in the Dawn Bloom. A Priest healing itself shows
 *   four small blooms on its diagonals. There are no arrowheads and no plus
 *   signs.
 * - Point blank: when a Mage, Hunter or Priest strikes with the two rims under
 *   0.55 world units apart, its Basic is drawn as its own cleave; with the
 *   Close-Range Strokes switch off it shows its landing alone.
 * - Arcane Blast During Burst (switch kindled_blast): while a Mage's Burst is
 *   active its bolt is the Comet Bolt. That is a brighter 2.5 px violet bolt
 *   with a Starcore core, a comet head that rides the bolt's tip and then rests
 *   under the star, the larger star, and a ring that flashes open around the
 *   star. A point-blank Burst blow keeps a thicker cleave and the larger star.
 *   With the switch off, or without Burst, Arcane Blast is the plain bolt.
 *   Soft Glow: a faint wide line under bolts, streaks and ribbons while
 *   playing, never in a still.
 * - Small champions get less detail (the detail ladder): from 22 px radius
 *   everything; 14-22 px no sparks and two fletching bars; 10-14 px no glows or
 *   centre lines, thin strokes and 70% landings; under 10 px a 3 px class dot.
 *
 * Who calls it: `choreography-painter.js` sends every activation whose
 * component is not "ultimate" here. It calls records({event}), then
 * build(records, doc, mount) into its groups, tooltip(record) for the hover
 * card and update(nodes, records) after a resize or camera change.
 * Strokes, bolts and cleaves go in the route group (the Strike Band layer:
 * above stones, under bodies); landings, sparks and dots go in the event group
 * (the foreground events layer); a moved landing's leader goes in the
 * connector group. The Visual Key draws the same pictures through
 * detachedEventMount.
 *
 * Timing: the looks are authored in blow time (the store's milliseconds at
 * 1x). The bolt draws in 80-180 ms and drains 420-540 ms, landings land at
 * 240-340 ms, and everything has faded by 600 ms. Blow times are turned into
 * real milliseconds from the event's own phaseImpact (blow 360 ms) and phaseEnd
 * (blow 760 ms), which the plan already placed on the beat clock, so Body Glide
 * and Longer Tick are followed without importing the plan. Each animation is a
 * Web Animations description (fill "backwards", first frame invisible) that
 * the choreographer starts; the DOM always holds the still picture: cleaves at
 * 80%, bolts, streaks and ribbons at 50% (the Comet Bolt and its head at 65%,
 * its ring at 55%), landings at 80%. Reduced motion is opacity only, within
 * 150 ms. No timers and no CSS animations.
 *
 * Geometry rule (anchors at END): every blow joins the two points the plan
 * gives (`event.source`, `event.target`, already projected to CSS pixels).
 * Under Body Glide those are where the fighters ended the tick, so a blow can
 * look longer than its reach while the imprint keeps the true start. Where the
 * allocator moved the landing off its anchor (`impactCue`), the whole landing
 * and the cleave end move with it and the allocator's leader is drawn.
 *
 * Agent POV boundary: records() reads only the one plan event it is given. The
 * plan admitted it for the audience, and in Agent POV it anchors a champion
 * that left sight this tick at its Last Seen spot (the tick's start), so such
 * a blow starts at its Last Seen ring and never shows the hidden END. A
 * redacted champion is named by class only in the tooltip. Nothing here reads
 * the researcher view, the match summary, a body or any other event.
 */
import { CLASS_IDENTITIES, exactAuthorizedAgentIdentityV1 } from "../agent-identity.js";
import { classAbilityNameV1 } from "../class-documentation.js";
import { explainActivation } from "../explanations.js";
import { createSemanticDescriptor } from "../tooltip.js";
import {
  animationSpec,
  appendAllocatedLeader,
  assignLayoutPlacement,
  detachedEventMount,
  EVENT_BUDGET,
  effectNode,
  explainChoreographyEvent,
  mountNodes,
  paintAwareExplanationEvent,
  paintPartEnabled,
  paintParts,
  setEffectAttributes,
  syncAllocatedLeader,
} from "./look-support.js";
import { ACTIVATION_FOOTPRINTS } from "./strike-routes.js";

/**
 * @typedef {import("./look-support.js").JsonRecord} JsonRecord
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").EventMount} EventMount
 * @typedef {import("./look-support.js").AnimationSpec} AnimationSpec
 * @typedef {[number, number]} Vec
 * @typedef {{
 *   pivot: Vec,
 *   polygons: Record<string, Vec[][]>,
 *   circles: Record<string, Array<[Vec, number]>>,
 *   lines: Record<string, Array<[Vec, Vec]>>,
 * }} LandingShape
 *   A landing in its own frame: the pivot it scales about, and its filled
 *   polygons, circles and lines by role (CSS pixels).
 * @typedef {{
 *   key: string,
 *   tag: string,
 *   attrs: Record<string, string | number | null>,
 *   children?: DrawNode[],
 *   motion?: string,
 *   playOnly?: boolean,
 * }} DrawNode
 *   One SVG node to draw. key is its stable `data-basic-part` name inside the
 *   event; motion names the animation it gets while playing; playOnly nodes
 *   (glows and sparks) are left out of a still.
 */

/**
 * Basic footprints in CSS pixels, read by the plan's collision allocator: the
 * impact boxes of a Basic damage or heal blow, the mark used when only the
 * meaning paint is on, and the route padding kept clear around a Basic route.
 * The landings, cleaves and bolts drawn here fit inside these boxes at the
 * 22 px detail level (a star reaches about 1.64 radii from the target center;
 * the Comet Bolt's ring reaches 0.44 radii plus 1.5 px from the star's center).
 */
export const FOOTPRINT = Object.freeze({
  activation: Object.freeze({
    basic_damage: ACTIVATION_FOOTPRINTS.activation.basic_damage,
    basic_heal: ACTIVATION_FOOTPRINTS.activation.basic_heal,
  }),
  semanticOnlyImpact: Object.freeze({ width: 24, height: 24 }),
  route: Object.freeze({
    activationPathPadding: ACTIVATION_FOOTPRINTS.route.activationPathPadding.default,
    activationMarkerPadding: ACTIVATION_FOOTPRINTS.route.activationMarkerPadding,
    activationCompactMarkerPadding:
      ACTIVATION_FOOTPRINTS.route.activationCompactMarkerPadding,
  }),
});

/**
 * The plan's shared per-event allowance: 30 nodes and 4 animations. A Basic
 * times its own marks, so the painter adds no fade of its own. The Comet Bolt
 * (Arcane Blast during Burst) is the busiest: at most 20 nodes and 4 animations
 * (the bolt, the comet head, the star and the ring with its flash). Every other
 * Basic uses at most 16 nodes and 2 animations (a stroke or cleave plus a
 * landing, twin flick or sparks). A moved landing adds one leader node.
 */
export const BUDGET = EVENT_BUDGET;

/**
 * The Basic's paint parts and the switch that owns each one. The ability and
 * meaning marks belong to Basic Ability Effects (`basic_ability_effects`).
 * Three finer parts each belong to their own switch: `glow` to Soft Glow
 * (`soft_glow`), `close_range` to Close-Range Strokes (`close_range_strokes`)
 * and `kindled` to Arcane Blast During Burst (`kindled_blast`). The plan may
 * resolve these three into `event.paintParts.glow`, `.close_range` and
 * `.kindled`; records() also reads them from `input.filters`.
 */
export const PAINT_PARTS = paintParts([
  [
    { surface: "transient", kind: "activation", component: "basic", part: "ability" },
    "basic_ability_effects",
  ],
  [
    { surface: "transient", kind: "activation", component: "basic", part: "semantic" },
    "basic_ability_effects",
  ],
  [
    { surface: "transient", kind: "activation", component: "basic", part: "glow" },
    "soft_glow",
  ],
  [
    {
      surface: "transient",
      kind: "activation",
      component: "basic",
      part: "close_range",
    },
    "close_range_strokes",
  ],
  [
    { surface: "transient", kind: "activation", component: "basic", part: "kindled" },
    "kindled_blast",
  ],
]);

/**
 * Rims closer than this many world units are point blank. A point-blank Mage,
 * Hunter or Priest blow is drawn as its class cleave while Close-Range Strokes
 * is on, and as its landing alone while it is off; it never draws a line.
 */
export const CLOSE_RANGE_UNITS = 0.55;
/** A champion's body radius in world units, used only when the event gives no size. */
const BODY_RADIUS_UNITS = 0.5;
/** Body radius in CSS pixels used when neither the event nor a transform gives one. */
const FALLBACK_RADIUS_PX = 14;

/**
 * Blow-time windows in milliseconds at 1x (the store's numbers). bolt: draw-in
 * and drain from the Mage's end; streak: draw-in and fade; ribbon: unfurl and
 * fade; landing: the one Impact beat (open by openEnd for the Shard Fan and
 * Dawn Bloom) and the fade; cleaveFade, sparks and blessing windows; reduced:
 * the reduced-motion envelope (fade in, hold, gone); gone: everything has
 * faded by this time.
 */
export const BLOW_TIMING = Object.freeze({
  bolt: Object.freeze({ drawStart: 80, drawEnd: 180, drainStart: 420, drainEnd: 540 }),
  streak: Object.freeze({ drawStart: 80, drawEnd: 160, fadeStart: 420, fadeEnd: 500 }),
  ribbon: Object.freeze({ growStart: 80, growEnd: 240, fadeStart: 420, fadeEnd: 600 }),
  landing: Object.freeze({
    start: 240,
    openEnd: 290,
    end: 340,
    fadeStart: 480,
    fadeEnd: 600,
  }),
  cleaveFade: Object.freeze({ start: 480, end: 600 }),
  sparks: Object.freeze({ start: 240, end: 360 }),
  blessing: Object.freeze({
    growStart: 80,
    growEnd: 240,
    fadeStart: 480,
    fadeEnd: 600,
  }),
  reduced: Object.freeze({ fadeIn: 70, holdEnd: 120, end: 150 }),
  gone: 600,
});

/** Opacity while playing (peak) and in a still, per kind of mark. */
export const BLOW_OPACITY = Object.freeze({
  peak: Object.freeze({ stroke: 0.85, cleave: 0.9, landing: 0.85 }),
  still: Object.freeze({ stroke: 0.5, cleave: 0.8, landing: 0.8 }),
});

/**
 * The Comet Bolt: Arcane Blast while the Mage's Burst is active, with the
 * Arcane Blast During Burst switch on. Sizes are CSS pixels at the full detail
 * level (the 10-14 px level draws the head and ring at 70%, a thin bolt and no
 * core line), and the ring sizes are shares of the target radius R.
 * - core, center, glow: widths of the violet bolt, its Starcore core and its
 *   Soft Glow line (the glow has square ends).
 * - peak, still: the bolt's and the head's opacity while playing and in a still.
 * - head: the comet head. ball and ballCore are radii of the violet ball and
 *   its Starcore core; tailCore is the half width of the Starcore tail (the
 *   violet tail is as wide as the ball, at 55%); the tail is half the head's
 *   travel, held between tailMin and tailMax, and the Starcore tail is
 *   coreShare of it.
 * - ring: the ring around the star's center. radius and flash are shares of R
 *   (the flash circle sits inside the growing ring, so it is a little larger
 *   than it looks at its brightest); outer and inner are the violet and
 *   Starcore line widths; still is its opacity in a still.
 */
export const COMET = Object.freeze({
  core: 2.5,
  center: 1.2,
  glow: 9,
  peak: 0.95,
  still: 0.65,
  head: Object.freeze({
    ball: 4.2,
    ballCore: 2.2,
    tailCore: 1.9,
    tailMin: 2,
    tailMax: 56,
    coreShare: 0.7,
  }),
  ring: Object.freeze({
    radius: 0.44,
    flash: 0.465,
    outer: 3,
    inner: 1.2,
    still: 0.55,
  }),
});

/**
 * One cleave per class. a: start angle off the line to the landing (degrees);
 * k: first bow as a share of the chord; w: widest width in px at a 23 px
 * radius; prof: width profile; grow: blow-time window it grows in (ms);
 * easing: its growth easing. The Rogue adds a second, thinner flick (twin).
 */
const CLEAVE = Object.freeze({
  warrior: { a: 80, k: 0.4, w: 7, prof: "blunt", grow: [150, 240], easing: "ease-in" },
  rogue: {
    a: 68,
    k: 0.34,
    w: 3,
    prof: "blade",
    grow: [190, 235],
    easing: "ease-out",
    twin: { a: 36, k: 0.2, w: 2.4, grow: [250, 295] },
  },
  mage: { a: 62, k: 0.34, w: 3.2, prof: "blade", grow: [120, 220], easing: "ease-out" },
  hunter: { a: 52, k: 0.3, w: 3.6, prof: "blade", grow: [170, 220], easing: "linear" },
  priest: { a: 72, k: 0.4, w: 6, prof: "soft", grow: [90, 250], easing: "ease-in-out" },
});
/** Samples along a cleave's centre line. */
const CLEAVE_SAMPLES = 20;
/** The class landing after a ranged or point-blank blow (Warrior and Rogue have none). */
const LANDING_BY_CLASS = Object.freeze({
  mage: "star",
  hunter: "fan",
  priest: "bloom",
});
/** The ranged stroke of each ranged class. */
const STROKE_BY_CLASS = Object.freeze({
  mage: "bolt",
  hunter: "streak",
  priest: "ribbon",
});
/** The Shard Fan's four shards: [angle off the landing direction (deg), length in radii]. */
const FAN_SHARDS = Object.freeze([
  [-42, 0.36],
  [-15, 0.5],
  [13, 0.46],
  [40, 0.33],
]);
/** Self-heal blooms sit on these diagonals (degrees) at this scale. */
const BLESSING_ANGLES = Object.freeze([-45, 45, 135, 225]);
const BLESSING_SCALE = 0.62;
const DEG = Math.PI / 180;

/* ---------- Small vector helpers (CSS pixels) ---------- */

/** @param {Vec} a @param {Vec} b @returns {Vec} */
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
/** @param {Vec} a @param {Vec} b @returns {Vec} */
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
/** @param {Vec} a @param {number} k @returns {Vec} */
const mul = (a, k) => [a[0] * k, a[1] * k];
/** @param {Vec} a */
const len = (a) => Math.hypot(a[0], a[1]);
/** @param {Vec} a @param {Vec} b */
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
/** @param {Vec} a @param {Vec} b @param {number} t @returns {Vec} */
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
/** @param {Vec} p @param {number} angle radians @returns {Vec} */
const rot = (p, angle) => [
  p[0] * Math.cos(angle) - p[1] * Math.sin(angle),
  p[0] * Math.sin(angle) + p[1] * Math.cos(angle),
];
/** @param {Vec} c @param {number} r @param {number} angle radians @returns {Vec} */
const polar = (c, r, angle) => [c[0] + r * Math.cos(angle), c[1] + r * Math.sin(angle)];
/** The screen-left side of a direction (a fixed side, never a facing). @param {Vec} u @returns {Vec} */
const leftOf = (u) => [u[1], -u[0]];
/** @param {number} value @param {number} low @param {number} high */
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
/** @param {Vec} a @returns {Vec} */
function norm(a) {
  const length = len(a);
  return length > 1e-9 ? [a[0] / length, a[1] / length] : [1, 0];
}
/** Round to 0.01 px so records stay stable plain data. @param {number} value */
const r2 = (value) => Math.round(value * 100) / 100 + 0;
/** @param {Vec} p @returns {Vec} */
const pt = (p) => [r2(p[0]), r2(p[1])];
/** @param {Vec} p */
const xy = (p) => `${r2(p[0])} ${r2(p[1])}`;
/** A closed polygon path. @param {Vec[]} points */
const polygonPath = (points) => `M${points.map(xy).join("L")}Z`;
/** A full circle as two arcs (a path, so several share one node). @param {Vec} c @param {number} r */
const circlePath = (c, r) =>
  `M${r2(c[0] - r)} ${r2(c[1])}a${r2(r)} ${r2(r)} 0 1 0 ${r2(2 * r)} 0a${r2(r)} ${r2(r)} 0 1 0 ${r2(-2 * r)} 0`;

/**
 * Read a screen point from a plan anchor.
 *
 * value is a plan point {x, y} in CSS pixels. Return [x, y], or null when it is
 * missing or not finite. Pure.
 *
 * @param {unknown} value
 * @returns {Vec | null}
 */
function screenVec(value) {
  if (value === null || typeof value !== "object") return null;
  const { x, y } = /** @type {{x?: unknown, y?: unknown}} */ (value);
  return typeof x === "number" &&
    typeof y === "number" &&
    Number.isFinite(x) &&
    Number.isFinite(y)
    ? [x, y]
    : null;
}

/**
 * Return a positive finite number, or null.
 *
 * @param {unknown} value
 * @returns {number | null}
 */
function positive(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : null;
}

/**
 * Read one of the Basic's finer switches.
 *
 * input.filters (the page's visual filter state, a map of switch id to
 * boolean) wins when it holds filterId; otherwise event.paintParts[partKey]
 * when the plan resolved it; otherwise the switch's default, on. Pure.
 *
 * @param {JsonRecord} input
 * @param {JsonRecord} event
 * @param {string} filterId one of soft_glow, close_range_strokes, kindled_blast
 * @param {string} partKey the matching paint-part key: glow, close_range, kindled
 * @returns {boolean}
 */
function switchOn(input, event, filterId, partKey) {
  const fromFilters = input.filters?.[filterId];
  if (typeof fromFilters === "boolean") return fromFilters;
  const fromPlan = event.paintParts?.[partKey];
  return typeof fromPlan === "boolean" ? fromPlan : true;
}

/**
 * Return the detail level for a target body radius in CSS pixels.
 *
 * 3 (22 px or more): everything. 2 (14-22): no sparks, two fletching bars.
 * 1 (10-14): no glows or centre lines, thin strokes, 70% landings.
 * 0 (under 10): a 3 px class dot only. Pure.
 *
 * @param {number} radius
 * @returns {0 | 1 | 2 | 3}
 */
export function detailLevel(radius) {
  return radius >= 22 ? 3 : radius >= 14 ? 2 : radius >= 10 ? 1 : 0;
}

/**
 * Work out the body radii and the pixels per world unit for one event.
 *
 * Sizes come, in order, from event.targetRadius / event.sourceRadius and
 * event.pixelsPerUnit (CSS pixels, when the plan supplies them), from
 * input.transform.worldLengthToScreen, from a successor-anchored route end
 * (the plan clips a route 3 px outside the target rim), and finally a 14 px
 * fallback. Champions are 0.5 world units in radius, so one size gives the
 * other. A source with no size (a champion out of sight) uses the target's.
 * Pure.
 *
 * @param {JsonRecord} input
 * @param {JsonRecord} event
 * @param {Vec | null} target
 * @returns {{sourceRadius: number, targetRadius: number, pixelsPerUnit: number}}
 */
function bodyScale(input, event, target) {
  const transformLength = input.transform?.worldLengthToScreen;
  const fromTransform =
    typeof transformLength === "function"
      ? positive(Number(transformLength(BODY_RADIUS_UNITS)))
      : null;
  const ppuGiven = positive(event.pixelsPerUnit);
  const routeEnd = screenVec(event.route?.end);
  const fromRoute =
    routeEnd &&
    target &&
    event.targetEndpointPhase === "successor" &&
    event.route?.kind === "curve" &&
    event.route?.close !== true
      ? positive(len(sub(routeEnd, target)) - 3)
      : null;
  const targetRadius =
    positive(event.targetRadius) ??
    fromTransform ??
    (ppuGiven === null ? null : ppuGiven * BODY_RADIUS_UNITS) ??
    fromRoute ??
    FALLBACK_RADIUS_PX;
  const sourceRadius = positive(event.sourceRadius) ?? targetRadius;
  const pixelsPerUnit = ppuGiven ?? targetRadius / BODY_RADIUS_UNITS;
  return { sourceRadius, targetRadius, pixelsPerUnit };
}

/**
 * Return the label the tooltip uses for one champion.
 *
 * identity is the plan's authorized identity snapshot or null. A valid one
 * gives "Mage · ID 5" (agent-identity.js). Without one, a known classId gives
 * the class name alone, and otherwise "Out Of Sight" when redacted is true or
 * "Unknown Champion". Pure.
 *
 * @param {unknown} identity
 * @param {unknown} classId
 * @param {boolean} redacted
 */
function championLabel(identity, classId, redacted) {
  const exact = exactAuthorizedAgentIdentityV1(identity);
  if (exact !== null) return exact.title;
  const classLabel =
    typeof classId === "number"
      ? CLASS_IDENTITIES[/** @type {1|2|3|4|5} */ (classId)]?.label
      : undefined;
  if (classLabel) return classLabel;
  return redacted ? "Out Of Sight" : "Unknown Champion";
}

/**
 * Format an amount the way the store does: 10 or more as a whole number,
 * smaller as one decimal. @param {number} value
 */
function amountText(value) {
  return value >= 10
    ? String(Math.round(value))
    : (Math.round(value * 10) / 10).toFixed(1);
}

/**
 * Build the one-line tooltip title of a Basic.
 *
 * For example "Arcane Blast · Mage · ID 0 → Mage · ID 5 · 15 Damage". The
 * amount is added only when the event carries `outputAmount` (a finite number
 * of health points); a Priest's reads "Healing". A point-blank blow adds
 * "From Right Beside Its Target". Pure.
 *
 * @param {JsonRecord} event
 * @param {string} basicName
 * @param {boolean} self
 * @param {boolean} closeRange
 * @param {string | null} classKey
 */
function tooltipLine(event, basicName, self, closeRange, classKey) {
  const source = championLabel(
    event.sourceIdentity,
    event.sourceClassId,
    event.sourceDisclosure === "redacted",
  );
  const target = self
    ? "Itself"
    : championLabel(
        event.recipientIdentity,
        null,
        event.targetDisclosure === "redacted",
      );
  const parts = [basicName, `${source} → ${target}`];
  const amount = event.outputAmount;
  if (typeof amount === "number" && Number.isFinite(amount) && amount >= 0) {
    parts.push(`${amountText(amount)} ${classKey === "priest" ? "Healing" : "Damage"}`);
  }
  if (closeRange) parts.push("From Right Beside Its Target");
  return parts.join(" · ");
}

/* ---------- Landing shapes (store numbers, in the landing's own frame) ---------- */

/**
 * Return one landing's shape in its own frame: x points out of the target,
 * the target centre is the origin and lengths are in CSS pixels.
 *
 * type is "star" (the Mage's eight-point star, long rays 0.3 radii or 0.36
 * when kindled), "fan" (the Shard Fan: four slim yew shards with pale spines
 * and a pale spark at the hit point) or "bloom" (the Dawn Bloom: six round gold
 * petals round a Dawnlight heart). R is the target radius. Return {pivot,
 * parts}, where parts are [role, kind, points or circles]. Pure.
 *
 * @param {"star" | "fan" | "bloom"} type
 * @param {number} R
 * @param {boolean} kindled
 * @returns {LandingShape}
 */
function landingShape(type, R, kindled) {
  if (type === "star") {
    const long = kindled ? 0.36 : 0.3;
    /** @type {Array<[number, number]>} */
    const rays = Array.from({ length: 8 }, (_, i) => [i * 45, i % 2 ? 0.16 : long]);
    const reach = Math.max(...rays.map(([a, L]) => -L * Math.cos(a * DEG)));
    /** @type {Vec} */
    const c = [(1.04 + reach) * R, 0];
    /** @type {Vec[]} */
    const poly = [];
    rays.forEach(([a, L], i) => {
      const next = rays[(i + 1) % rays.length][0];
      const mid = (a + ((next - a + 360) % 360 || 360) / 2) * DEG;
      poly.push(add(c, mul([Math.cos(a * DEG), Math.sin(a * DEG)], L * R)));
      poly.push(add(c, mul([Math.cos(mid), Math.sin(mid)], 0.06 * R)));
    });
    return { pivot: c, polygons: { star: [poly] }, circles: {}, lines: {} };
  }
  if (type === "fan") {
    /** @type {Vec} */
    const o = [1.1 * R, 0];
    /** @type {Vec[][]} */
    const shards = [];
    /** @type {Array<[Vec, Vec]>} */
    const spines = [];
    for (const [a, L] of FAN_SHARDS) {
      /** @type {Vec} */
      const d = [Math.cos(a * DEG), Math.sin(a * DEG)];
      /** @type {Vec} */
      const n = [-d[1], d[0]];
      const base = add(o, mul(d, 0.07 * R));
      const tip = add(o, mul(d, (0.07 + L) * R));
      const mid = lerp(base, tip, 0.3);
      const w = 0.06 * R;
      shards.push([base, add(mid, mul(n, w)), tip, sub(mid, mul(n, w))]);
      spines.push([base, lerp(base, tip, 0.62)]);
    }
    return {
      pivot: o,
      polygons: { shards },
      circles: { spark: [[o, 0.1 * R]] },
      lines: { spines },
    };
  }
  const petalRadius = 0.105;
  const petalDistance = 0.19;
  /** @type {Vec} */
  const c = [(1.04 + petalDistance * Math.cos(30 * DEG) + petalRadius) * R, 0];
  /** @type {Array<[Vec, number]>} */
  const petals = [];
  for (let i = 0; i < 6; i += 1) {
    const a = (30 + 60 * i) * DEG;
    petals.push([
      add(c, mul([Math.cos(a), Math.sin(a)], petalDistance * R)),
      petalRadius * R,
    ]);
  }
  return {
    pivot: c,
    polygons: {},
    circles: { petals, heart: [[c, 0.08 * R]] },
    lines: {},
  };
}

/**
 * Turn a landing shape into path strings around a chosen origin.
 *
 * shape comes from landingShape; angle (radians) turns the landing to face its
 * attacker; scale (1, 0.7 or the blessing 0.62) shrinks it about its pivot;
 * offset moves the pivot from the target centre (pixels). Points are written
 * relative to origin. Return a map of role to SVG path data. Pure.
 *
 * @param {LandingShape} shape
 * @param {number} angle
 * @param {number} scale
 * @param {Vec} offset
 * @param {Vec} origin
 * @returns {Record<string, string>}
 */
function landingPaths(shape, angle, scale, offset, origin) {
  /** @param {Vec} q @returns {Vec} */
  const place = (q) =>
    sub(
      add(
        add(rot(shape.pivot, angle), rot(mul(sub(q, shape.pivot), scale), angle)),
        offset,
      ),
      origin,
    );
  /** @type {Record<string, string>} */
  const paths = {};
  for (const [role, polygons] of Object.entries(shape.polygons)) {
    paths[role] = polygons.map((polygon) => polygonPath(polygon.map(place))).join("");
  }
  for (const [role, circles] of Object.entries(shape.circles)) {
    paths[role] = circles.map(([c, r]) => circlePath(place(c), r * scale)).join("");
  }
  for (const [role, lines] of Object.entries(shape.lines)) {
    paths[role] = lines.map(([a, b]) => `M${xy(place(a))}L${xy(place(b))}`).join("");
  }
  return paths;
}

/* ---------- Forward cleaves (store rules) ---------- */

/**
 * Return a cleave's width at fraction s along it, for profile prof: "blunt"
 * widens to a blunt end (Warrior), "soft" swells in the middle (Priest),
 * "blade" tapers at both ends. Pure.
 *
 * @param {string} prof
 * @param {number} s
 */
function cleaveProfile(prof, s) {
  if (prof === "blunt") return 0.18 + 0.82 * s ** 0.7;
  if (prof === "soft") return Math.sin(Math.PI * clamp(s, 0, 1));
  return Math.sin(Math.PI * clamp(s, 0, 1) ** 1.35);
}

/**
 * Sample one swept arc from the attacker's rim to the cleave's end.
 *
 * A is the attacker centre and Rs its radius; end is the cleave's end point;
 * a (degrees) turns the start off the line to the end, toward side sv; k bows
 * the arc by k times the chord (capped at 0.8 Rs plus a little, so a long
 * cleave between fighters that stepped apart stays a gentle arc); W is the
 * widest width and prof the profile. Return the start point, sampled centre
 * points, outward normals and widths. Pure.
 *
 * @param {Vec} A
 * @param {Vec} end
 * @param {number} Rs
 * @param {number} a
 * @param {number} k
 * @param {Vec} sv
 * @param {number} W
 * @param {string} prof
 */
function sweepCurve(A, end, Rs, a, k, sv, W, prof) {
  const u = norm(sub(end, A));
  let n = leftOf(u);
  if (dot(n, sv) < 0) n = mul(n, -1);
  const start = add(
    A,
    mul(norm(add(mul(u, Math.cos(a * DEG)), mul(n, Math.sin(a * DEG)))), Rs + 2),
  );
  const chord = sub(end, start);
  let bowSide = leftOf(norm(chord));
  if (dot(bowSide, sv) < 0) bowSide = mul(bowSide, -1);
  const bow = Math.min(k * len(chord), 0.8 * Rs + (k - 0.3) * 20);
  const control = add(lerp(start, end, 0.5), mul(bowSide, bow));
  /** @type {Vec[]} */
  const points = [];
  /** @type {Vec[]} */
  const normals = [];
  /** @type {number[]} */
  const widths = [];
  for (let i = 0; i < CLEAVE_SAMPLES; i += 1) {
    const s = i / (CLEAVE_SAMPLES - 1);
    const m = 1 - s;
    points.push(add(add(mul(start, m * m), mul(control, 2 * s * m)), mul(end, s * s)));
    const tangent = add(mul(sub(control, start), 2 * m), mul(sub(end, control), 2 * s));
    let normal = leftOf(norm(tangent));
    if (dot(normal, bowSide) < 0) normal = mul(normal, -1);
    normals.push(normal);
    widths.push(W * cleaveProfile(prof, s));
  }
  return { start, points, normals, widths };
}

/**
 * Say whether a cleave grazes either fighter (then it bows more).
 *
 * Return true when a middle sample comes within the fighter's radius plus 1 px
 * plus half the width. Other bodies and the floor edge are not known to this
 * per-event look, so they are not checked. Pure.
 *
 * @param {ReturnType<typeof sweepCurve>} curve
 * @param {Vec} A
 * @param {Vec} T
 * @param {number} Rs
 * @param {number} Rt
 */
function sweepGrazes(curve, A, T, Rs, Rt) {
  for (let i = 2; i < CLEAVE_SAMPLES - 2; i += 1) {
    const p = curve.points[i];
    const half = curve.widths[i] / 2;
    if (len(sub(p, A)) < Rs + 1 + half || len(sub(p, T)) < Rt + 1 + half) return true;
  }
  return false;
}

/**
 * Return a cleave's filled outline relative to its own start point.
 * @param {ReturnType<typeof sweepCurve>} curve
 */
function sweepOutline(curve) {
  /** @type {Vec[]} */
  const left = [];
  /** @type {Vec[]} */
  const right = [];
  curve.points.forEach((p, i) => {
    const half = curve.widths[i] / 2;
    left.push(sub(add(p, mul(curve.normals[i], half)), curve.start));
    right.push(sub(sub(p, mul(curve.normals[i], half)), curve.start));
  });
  return polygonPath([...left, ...right.reverse()]);
}

/**
 * Return a line along a cleave between fractions a and b, on the centre
 * (inset 0) or inset px in from the outer edge, relative to its start point.
 *
 * @param {ReturnType<typeof sweepCurve>} curve
 * @param {number} a
 * @param {number} b
 * @param {number} inset
 */
function sweepLine(curve, a, b, inset) {
  const last = CLEAVE_SAMPLES - 1;
  const first = Math.max(0, Math.round(a * last));
  const final = Math.min(last, Math.round(b * last));
  if (final <= first) return "";
  let d = "";
  for (let i = first; i <= final; i += 1) {
    const p = inset
      ? add(
          curve.points[i],
          mul(curve.normals[i], Math.max(0, curve.widths[i] / 2 - inset)),
        )
      : curve.points[i];
    d += `${i === first ? "M" : "L"}${xy(sub(p, curve.start))}`;
  }
  return d;
}

/**
 * Return the outer-edge point, outward normal and forward direction at s.
 * @param {ReturnType<typeof sweepCurve>} curve
 * @param {number} s
 */
function sweepEdge(curve, s) {
  const i = Math.round(s * (CLEAVE_SAMPLES - 1));
  return {
    P: add(curve.points[i], mul(curve.normals[i], curve.widths[i] / 2)),
    N: curve.normals[i],
    D: norm(
      sub(
        curve.points[Math.min(CLEAVE_SAMPLES - 1, i + 1)],
        curve.points[Math.max(0, i - 1)],
      ),
    ),
  };
}

/**
 * Plan one class's cleave from the attacker to the landing's end point.
 *
 * classKey picks the class cleave; A/Rs are the attacker centre and radius,
 * T/Rt the target's; end is just outside the target rim where the landing
 * sits; kindled thickens the Mage's. The bend picks the side the end sits on,
 * else the attacker's screen-left; it bows more until it clears both
 * fighters. level 2 or more adds the class accents. Return frozen drawing data
 * ({part, origin, d, accents, grow, easing}) for the main cleave and the
 * Rogue's twin, or [] when the end is too close to the attacker to draw an
 * arc. Pure.
 *
 * @param {string} classKey
 * @param {Vec} A
 * @param {number} Rs
 * @param {Vec} T
 * @param {number} Rt
 * @param {Vec} end
 * @param {boolean} kindled
 * @param {number} level
 */
function cleaveData(classKey, A, Rs, T, Rt, end, kindled, level) {
  const spec = CLEAVE[/** @type {keyof typeof CLEAVE} */ (classKey)];
  if (!spec || len(sub(end, A)) < Rs + 5) return [];
  const sizeScale = clamp(Rt / 23, 0.7, 1.4);
  const W = spec.w * sizeScale * (kindled ? 1.3 : 1);
  const axisLeft = leftOf(norm(sub(T, A)));
  const offset = dot(sub(end, A), axisLeft);
  const sides =
    Math.abs(offset) > 2 ? [Math.sign(offset), -Math.sign(offset)] : [1, -1];
  /** @type {ReturnType<typeof sweepCurve> | null} */
  let chosen = null;
  /** @type {Vec} */
  let side = axisLeft;
  for (const sign of sides) {
    const sv = mul(axisLeft, sign);
    for (let k = spec.k; k <= 0.8 + 1e-9; k += 0.08) {
      const curve = sweepCurve(A, end, Rs, spec.a, k, sv, W, spec.prof);
      if (chosen === null) {
        chosen = curve;
        side = sv;
      }
      if (!sweepGrazes(curve, A, T, Rs, Rt)) {
        chosen = curve;
        side = sv;
        break;
      }
    }
    if (chosen && !sweepGrazes(chosen, A, T, Rs, Rt)) break;
  }
  const main = /** @type {ReturnType<typeof sweepCurve>} */ (chosen);
  /** @type {Array<{role: string, d: string, kind: "line" | "fill"}>} */
  const accents = [];
  if (level >= 2) {
    if (classKey === "warrior") {
      accents.push({ role: "edge", kind: "line", d: sweepLine(main, 0.42, 1, 0.7) });
    } else if (classKey === "mage") {
      accents.push({ role: "core", kind: "line", d: sweepLine(main, 0.14, 0.9, 0) });
      let stars = "";
      for (const s of [0.38, 0.66]) {
        const edge = sweepEdge(main, s);
        const c = sub(add(edge.P, mul(edge.N, 3.2)), main.start);
        const r = 2.4;
        const q = 0.55;
        stars += polygonPath([
          add(c, [0, -r]),
          add(c, [q, -q]),
          add(c, [r, 0]),
          add(c, [q, q]),
          add(c, [0, r]),
          add(c, [-q, q]),
          add(c, [-r, 0]),
          add(c, [-q, -q]),
        ]);
      }
      accents.push({ role: "stars", kind: "fill", d: stars });
    } else if (classKey === "hunter") {
      let teeth = "";
      for (const s of [0.34, 0.54, 0.74]) {
        const edge = sweepEdge(main, s);
        const before = sweepEdge(main, s - 0.05);
        const after = sweepEdge(main, s + 0.05);
        const tip = add(add(edge.P, mul(edge.N, 2.8)), mul(edge.D, -2.4));
        teeth += polygonPath([before.P, tip, after.P].map((p) => sub(p, main.start)));
      }
      accents.push({ role: "teeth", kind: "fill", d: teeth });
    } else if (classKey === "priest") {
      accents.push({ role: "core", kind: "line", d: sweepLine(main, 0.1, 0.93, 0) });
    } else if (classKey === "rogue") {
      accents.push({ role: "spine", kind: "line", d: sweepLine(main, 0.16, 0.86, 0) });
    }
  }
  /** @type {Array<Readonly<Record<string, any>>>} */
  const cleaves = [
    Object.freeze({
      part: "main",
      origin: pt(main.start),
      d: sweepOutline(main),
      accents: Object.freeze(accents.map((accent) => Object.freeze(accent))),
      grow: Object.freeze([...spec.grow]),
      easing: spec.easing,
    }),
  ];
  if ("twin" in spec && spec.twin) {
    const twin = sweepCurve(
      A,
      end,
      Rs,
      spec.twin.a,
      spec.twin.k,
      side,
      spec.twin.w * sizeScale,
      spec.prof,
    );
    cleaves.push(
      Object.freeze({
        part: "twin",
        origin: pt(twin.start),
        d: sweepOutline(twin),
        accents: Object.freeze(
          level >= 2
            ? [
                Object.freeze({
                  role: "spine",
                  kind: /** @type {"line"} */ ("line"),
                  d: sweepLine(twin, 0.18, 0.84, 0),
                }),
              ]
            : [],
        ),
        grow: Object.freeze([...spec.twin.grow]),
        easing: spec.easing,
      }),
    );
  }
  return cleaves;
}

/* ---------- records ---------- */

/**
 * Return the drawing record of a Basic activation, or [] for any other event.
 *
 * input.event is one admitted plan event with screen geometry (CSS pixels):
 * source and target are the two champions' centres at the blow's anchor (END,
 * or the Last Seen spot in Agent POV for a champion that left sight); route
 * and impactCue come from the allocator. Optional inputs:
 * - input.filters: the visual filter state; it decides Soft Glow,
 *   Close-Range Strokes and Arcane Blast During Burst (else event.paintParts,
 *   else on).
 * - input.transform: a LookTransform, used for body size when the event
 *   carries none.
 * - event.targetRadius, event.sourceRadius, event.pixelsPerUnit: body radii and
 *   pixels per world unit, when the plan supplies them.
 * - event.sourceKindled: true while the Mage's Burst is active.
 * - event.outputAmount: the blow's damage or healing, for the tooltip.
 *
 * The one record is frozen plain data: kind "basic_blow"; event; classKey
 * ("mage" ... "priest" or null); shape ("bolt", "streak", "ribbon", "cleave",
 * "landing_only", "self_heal", "dot", "meaning_dot" or "none"); closeRange;
 * fromLastSeen (the source is drawn at its Last Seen spot); kindled (a Mage
 * blow under Burst with Arcane Blast During Burst on); glow (Soft Glow on and
 * detail level 2 or more); level (0-3); rimGapUnits; tooltipText; stroke (null
 * or {type, origin, angle in degrees, length, core, centre, glow widths in px,
 * ctrl, bars}); comet (null, or for a kindled bolt the Comet Bolt's extra
 * parts: {head, tail, scale, ringRadius, flashRadius}, where head is how far
 * along the bolt, in px from its origin, the head rests under the star, tail
 * the head's tail length in px, scale 1 or 0.7 for the 10-14 px level, and
 * ringRadius and flashRadius the ring's radii in px around landing.origin);
 * cleaves (list); sparks (null or {origin, d}); landing (null or {type, origin,
 * paths}); blessings (null or {origin, paths}); dot (null or {at, r}); leader
 * (the allocator's leader or null). A kindled bolt is the Comet Bolt (core
 * 2.5 px, `centre` 1.2 px and glow 9 px from the 14 px level up; thin with no
 * `centre` line below it); a kindled cleave is 1.3 times as wide and a kindled star
 * has longer rays. Turning Basic Ability Effects off gives shape "none"
 * (nothing drawn), or "meaning_dot" when only the meaning paint is on. Pure:
 * no DOM, no clock.
 *
 * @param {import("./look-support.js").LookInput} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const event = input?.event;
  if (event?.kind !== "activation" || event.component === "ultimate") {
    return Object.freeze([]);
  }
  const options = /** @type {JsonRecord} */ (input);
  const classId = Number.isInteger(event.sourceClassId) ? event.sourceClassId : null;
  const classKey =
    classId === null
      ? null
      : (CLASS_IDENTITIES[/** @type {1|2|3|4|5} */ (classId)]?.tokenId ?? null);
  const basicName = classAbilityNameV1(classId, "basic") ?? "Basic";
  const ability = paintPartEnabled(event, "ability");
  const semantic = paintPartEnabled(event, "semantic");
  const closeOn = switchOn(options, event, "close_range_strokes", "close_range");
  const glowOn = switchOn(options, event, "soft_glow", "glow");
  const kindledOn = switchOn(options, event, "kindled_blast", "kindled");
  const S = screenVec(event.source);
  const T = screenVec(event.target);
  const self =
    typeof event.targetPresentationKey === "string" &&
    event.targetPresentationKey === event.sourcePresentationKey;
  const {
    sourceRadius: Rs,
    targetRadius: Rt,
    pixelsPerUnit,
  } = bodyScale(options, event, T);
  const level = detailLevel(Rt);
  const kindled = classKey === "mage" && kindledOn && event.sourceKindled === true;
  const fromLastSeen = event.sourceEndpointPhase === "transition_start";
  const ranged = classKey === "mage" || classKey === "hunter" || classKey === "priest";
  const rimGapUnits =
    S && T && !self ? (len(sub(T, S)) - Rs - Rt) / pixelsPerUnit : null;
  const closeRange =
    ranged && closeOn && rimGapUnits !== null && rimGapUnits < CLOSE_RANGE_UNITS;
  /** @type {Record<string, any>} */
  const record = {
    kind: "basic_blow",
    event,
    classKey,
    shape: "none",
    closeRange,
    fromLastSeen,
    kindled,
    glow: glowOn && level >= 2,
    level,
    rimGapUnits: rimGapUnits === null ? null : r2(rimGapUnits),
    tooltipText: tooltipLine(event, basicName, self, closeRange, classKey),
    stroke: null,
    comet: null,
    cleaves: Object.freeze([]),
    sparks: null,
    landing: null,
    blessings: null,
    dot: null,
    leader: event.impactLeader ?? null,
  };
  const finish = () => Object.freeze([Object.freeze(record)]);
  if (!ability) {
    if (semantic && T) {
      record.shape = "meaning_dot";
      record.dot = Object.freeze({ at: pt(polar(T, Rt + 3, -90 * DEG)), r: 1.5 });
    }
    return finish();
  }
  if (!T || classKey === null || (!S && !self)) return finish();
  const source = S ?? T;
  const toSource = sub(source, T);
  const angle =
    self || len(toSource) < 1e-6 ? -90 * DEG : Math.atan2(toSource[1], toSource[0]);
  if (level === 0) {
    record.shape = "dot";
    record.dot = Object.freeze({ at: pt(polar(T, Rt + 2, angle)), r: 1.5 });
    return finish();
  }
  const sizeScale = level === 1 ? 0.7 : 1;
  if (self) {
    if (classKey !== "priest") {
      record.shape = "dot";
      record.dot = Object.freeze({ at: pt(polar(T, Rt + 2, angle)), r: 1.5 });
      return finish();
    }
    const shape = landingShape("bloom", Rt, false);
    /** @type {Record<string, string>} */
    const paths = { petals: "", heart: "" };
    for (const degrees of BLESSING_ANGLES) {
      const each = landingPaths(
        shape,
        degrees * DEG,
        BLESSING_SCALE * sizeScale,
        [0, 0],
        [0, 0],
      );
      paths.petals += each.petals;
      paths.heart += each.heart;
    }
    record.shape = "self_heal";
    record.blessings = Object.freeze({ origin: pt(T), paths: Object.freeze(paths) });
    return finish();
  }
  const impactCue = screenVec(event.impactCue);
  const routeEnd = screenVec(event.route?.end);
  /** @type {Vec} */
  const shift = impactCue && routeEnd ? sub(impactCue, routeEnd) : [0, 0];
  const isCleave = closeRange || classKey === "warrior" || classKey === "rogue";
  const strokeOk = ranged && rimGapUnits !== null && rimGapUnits >= CLOSE_RANGE_UNITS;
  if (isCleave && S) {
    const end = add(polar(T, 1.1 * Rt, angle), shift);
    const cleaves = cleaveData(classKey, S, Rs, T, Rt, end, kindled, level);
    if (cleaves.length === 0) {
      record.shape = "dot";
      record.dot = Object.freeze({ at: pt(polar(T, Rt + 2, angle)), r: 1.5 });
      return finish();
    }
    record.shape = "cleave";
    record.cleaves = Object.freeze(cleaves);
    if (classKey === "warrior" && level >= 3) {
      let d = "";
      for (const spread of [-20, 0, 20]) {
        const a = angle + spread * DEG;
        d += `M${xy(sub(polar(T, 1.36 * Rt, a), T))}L${xy(sub(polar(T, 1.56 * Rt, a), T))}`;
      }
      record.sparks = Object.freeze({ origin: pt(add(T, shift)), d });
    }
  } else if (strokeOk && S) {
    const u = norm(sub(T, S));
    const A = add(S, mul(u, Rs + 2));
    const B = sub(T, mul(u, 1.04 * Rt));
    const length = len(sub(B, A));
    const type = /** @type {"bolt" | "streak" | "ribbon"} */ (
      STROKE_BY_CLASS[/** @type {keyof typeof STROKE_BY_CLASS} */ (classKey)]
    );
    const thin = level === 1;
    const bow = Math.min(0.12 * length, 0.3 * pixelsPerUnit);
    record.shape = type;
    record.stroke = Object.freeze({
      type,
      origin: pt(A),
      angle: r2(Math.atan2(u[1], u[0]) / DEG),
      length: r2(length),
      // A kindled bolt is the Comet Bolt (COMET); a plain bolt is 2 px with a
      // 1 px center line.
      core:
        type === "bolt"
          ? thin
            ? 1.5
            : kindled
              ? COMET.core
              : 2
          : type === "streak"
            ? 1.5
            : thin
              ? 1.5
              : 2,
      centre:
        thin || type === "streak" ? 0 : type === "bolt" && kindled ? COMET.center : 1,
      glow: type === "bolt" ? (kindled ? COMET.glow : 6) : type === "streak" ? 4.5 : 7,
      ctrl: type === "ribbon" ? pt([length / 2, -2 * bow]) : null,
      bars: Object.freeze(
        type === "streak" ? (level >= 3 ? [3, 6, 9] : level === 2 ? [3, 7] : []) : [],
      ),
    });
  } else {
    record.shape = "landing_only";
  }
  const landingType =
    LANDING_BY_CLASS[/** @type {keyof typeof LANDING_BY_CLASS} */ (classKey)];
  if (landingType) {
    const shape = landingShape(landingType, Rt, kindled);
    const pivot = add(add(T, rot(shape.pivot, angle)), shift);
    record.landing = Object.freeze({
      type: landingType,
      origin: pt(pivot),
      paths: Object.freeze(landingPaths(shape, angle, sizeScale, add(T, shift), pivot)),
    });
    if (record.shape === "bolt" && kindled) {
      // The Comet Bolt's head rests under the star's center. The bolt ends
      // 1.04 R from the target center and the star's center is shape.pivot[0]
      // from it, so the head stops that difference short of the bolt's end.
      const head = Math.max(0, record.stroke.length - (shape.pivot[0] - 1.04 * Rt));
      record.comet = Object.freeze({
        head: r2(head),
        tail: r2(clamp(0.5 * head, COMET.head.tailMin, COMET.head.tailMax)),
        scale: sizeScale,
        ringRadius: r2(COMET.ring.radius * Rt * sizeScale),
        flashRadius: r2(COMET.ring.flash * Rt * sizeScale),
      });
    }
  }
  return finish();
}

/* ---------- build, update ---------- */

/**
 * Describe the Comet Bolt's head, in the bolt's own frame (x runs along the
 * bolt from its origin).
 *
 * comet is record.comet. The outer group sits comet.head px along the bolt,
 * under the star. Its inner group (motion "comet-head", 65% in a still) holds
 * a violet tail at 55% and a violet ball, with a Starcore tail and core on top,
 * all at comet.scale; the tail points back toward the Mage. While playing, the
 * motion carries the head out along the bolt's growing tip. Pure.
 *
 * @param {Readonly<Record<string, number>>} comet
 * @returns {DrawNode}
 */
function cometHead(comet) {
  const k = comet.scale;
  const ball = r2(COMET.head.ball * k);
  const core = r2(COMET.head.tailCore * k);
  return {
    key: "comet-head",
    tag: "g",
    attrs: { class: "basic-comet__head", transform: `translate(${comet.head} 0)` },
    children: [
      {
        key: "comet-head-move",
        tag: "g",
        motion: "comet-head",
        attrs: { opacity: COMET.still },
        children: [
          {
            key: "comet-tail",
            tag: "path",
            attrs: {
              class: "basic-comet__tail",
              d: `M0 ${-ball}L${-comet.tail} 0L0 ${ball}Z`,
            },
          },
          {
            key: "comet-ball",
            tag: "circle",
            attrs: { class: "basic-comet__ball", cx: 0, cy: 0, r: ball },
          },
          {
            key: "comet-tail-core",
            tag: "path",
            attrs: {
              class: "basic-comet__tail-core",
              d: `M0 ${-core}L${r2(-comet.tail * COMET.head.coreShare)} 0L0 ${core}Z`,
            },
          },
          {
            key: "comet-ball-core",
            tag: "circle",
            attrs: {
              class: "basic-comet__core",
              cx: 0,
              cy: 0,
              r: r2(COMET.head.ballCore * k),
            },
          },
        ],
      },
    ],
  };
}

/**
 * Describe the Comet Bolt's ring around the star.
 *
 * comet is record.comet, origin the star's center (record.landing.origin, CSS
 * pixels) and classKey the striking class. The inner group (motion
 * "comet-ring", 55% in a still) holds a violet ring with a Starcore line on it
 * and, while playing only, a Starcore flash disc. The group's fill-opacity is
 * 0, so the flash shows only while the motion lights it; the ring circles have
 * no fill. Pure.
 *
 * @param {Readonly<Record<string, number>>} comet
 * @param {Vec} origin
 * @param {string} classKey
 * @returns {DrawNode}
 */
function cometRing(comet, origin, classKey) {
  const k = comet.scale;
  return {
    key: "comet-ring",
    tag: "g",
    attrs: {
      class: "basic-comet",
      "data-basic-class": classKey,
      transform: `translate(${origin[0]} ${origin[1]})`,
    },
    children: [
      {
        key: "comet-ring-grow",
        tag: "g",
        motion: "comet-ring",
        attrs: { opacity: COMET.ring.still, "fill-opacity": 0 },
        children: [
          {
            key: "comet-flash",
            tag: "circle",
            playOnly: true,
            attrs: { class: "basic-comet__flash", cx: 0, cy: 0, r: comet.flashRadius },
          },
          {
            key: "comet-ring-outer",
            tag: "circle",
            attrs: {
              class: "basic-comet__ring",
              cx: 0,
              cy: 0,
              r: comet.ringRadius,
              "stroke-width": r2(COMET.ring.outer * k),
            },
          },
          {
            key: "comet-ring-inner",
            tag: "circle",
            attrs: {
              class: "basic-comet__ring-core",
              cx: 0,
              cy: 0,
              r: comet.ringRadius,
              "stroke-width": r2(COMET.ring.inner * k),
            },
          },
        ],
      },
    ],
  };
}

/**
 * Describe the SVG nodes of one record.
 *
 * record comes from records(). Return {band, over}: band nodes go in the route
 * group (Strike Band: strokes, cleaves and the Comet Bolt's head), over nodes
 * in the event group (landings, the Comet Bolt's ring, sparks, blessings and
 * dots). Every node has a stable key; glows, sparks and the Comet Bolt's flash
 * are marked playOnly. Pure.
 *
 * @param {LookRecord} record
 * @returns {{band: DrawNode[], over: DrawNode[]}}
 */
function drawing(record) {
  const classKey = record.classKey ?? "none";
  /** @type {DrawNode[]} */
  const band = [];
  /** @type {DrawNode[]} */
  const over = [];
  const stroke = record.stroke;
  if (stroke) {
    const straight = `M0 0L${stroke.length} 0`;
    const d =
      stroke.type === "ribbon"
        ? `M0 0Q${stroke.ctrl[0]} ${stroke.ctrl[1]} ${stroke.length} 0`
        : stroke.type === "streak"
          ? straight +
            stroke.bars
              .filter((/** @type {number} */ bar) => bar <= stroke.length)
              .map((/** @type {number} */ bar) => `M${bar} 2.5L${bar} -2.5`)
              .join("")
          : straight;
    /** @type {DrawNode[]} */
    const lines = [];
    if (record.glow) {
      lines.push({
        key: "stroke-glow",
        tag: "path",
        playOnly: true,
        attrs: {
          class: "basic-blow__glow",
          d: stroke.type === "streak" ? straight : d,
          "stroke-width": stroke.glow,
        },
      });
    }
    lines.push({
      key: "stroke-core",
      tag: "path",
      attrs: { class: "basic-blow__core", d, "stroke-width": stroke.core },
    });
    if (stroke.centre > 0) {
      lines.push({
        key: "stroke-centre",
        tag: "path",
        attrs: { class: "basic-blow__centre", d, "stroke-width": stroke.centre },
      });
    }
    lines.push({
      key: "stroke-hit",
      tag: "path",
      attrs: { class: "basic-blow__hit", d: stroke.type === "ribbon" ? d : straight },
    });
    const comet = record.comet;
    band.push({
      key: "stroke",
      tag: "g",
      attrs: {
        class: `basic-blow basic-blow--${stroke.type}${comet ? " basic-blow--comet" : ""}`,
        "data-basic-class": classKey,
        transform: `translate(${stroke.origin[0]} ${stroke.origin[1]}) rotate(${stroke.angle})`,
      },
      children: [
        {
          key: "stroke-draw",
          tag: "g",
          motion: comet ? "comet-bolt" : stroke.type,
          attrs: {
            class: "basic-blow__draw",
            opacity: comet ? COMET.still : BLOW_OPACITY.still.stroke,
          },
          children: lines,
        },
        ...(comet ? [cometHead(comet)] : []),
      ],
    });
  }
  for (const cleave of record.cleaves) {
    band.push({
      key: `cleave-${cleave.part}`,
      tag: "g",
      attrs: {
        class: `basic-cleave basic-cleave--${cleave.part}`,
        "data-basic-class": classKey,
        "data-kindled": record.kindled ? "true" : null,
        transform: `translate(${cleave.origin[0]} ${cleave.origin[1]})`,
      },
      children: [
        {
          key: `cleave-${cleave.part}-grow`,
          tag: "g",
          motion: `cleave-${cleave.part}`,
          attrs: { class: "basic-cleave__grow", opacity: BLOW_OPACITY.still.cleave },
          children: [
            {
              key: `cleave-${cleave.part}-fill`,
              tag: "path",
              attrs: { class: "basic-cleave__fill", d: cleave.d },
            },
            ...cleave.accents.map((/** @type {JsonRecord} */ accent) => ({
              key: `cleave-${cleave.part}-${accent.role}`,
              tag: "path",
              attrs: {
                class: `basic-cleave__${accent.role} basic-cleave__accent--${accent.kind}`,
                d: accent.d,
              },
            })),
          ],
        },
      ],
    });
  }
  if (record.sparks) {
    over.push({
      key: "sparks",
      tag: "g",
      attrs: {
        class: "basic-sparks",
        transform: `translate(${record.sparks.origin[0]} ${record.sparks.origin[1]})`,
      },
      children: [
        {
          key: "sparks-flick",
          tag: "path",
          motion: "sparks",
          playOnly: true,
          attrs: { class: "basic-sparks__flick", d: record.sparks.d, opacity: 0 },
        },
      ],
    });
  }
  const landing = record.landing;
  if (landing) {
    over.push({
      key: "landing",
      tag: "g",
      attrs: {
        class: `combat-impact basic-landing basic-landing--${landing.type}`,
        "data-basic-class": classKey,
        transform: `translate(${landing.origin[0]} ${landing.origin[1]})`,
      },
      children: [
        {
          key: "landing-stamp",
          tag: "g",
          motion: `landing-${landing.type}`,
          attrs: { class: "basic-landing__stamp", opacity: BLOW_OPACITY.still.landing },
          children: Object.entries(landing.paths).map(([role, d]) => ({
            key: `landing-${role}`,
            tag: "path",
            attrs: { class: `basic-landing__${role}`, d },
          })),
        },
      ],
    });
    if (record.comet) {
      over.push(cometRing(record.comet, landing.origin, classKey));
    }
  }
  if (record.blessings) {
    const origin = record.blessings.origin;
    over.push({
      key: "blessings",
      tag: "g",
      attrs: {
        class: "combat-impact basic-landing basic-landing--blessings",
        "data-basic-class": classKey,
        transform: `translate(${origin[0]} ${origin[1]})`,
      },
      children: [
        {
          key: "blessings-bloom",
          tag: "g",
          motion: "blessings",
          attrs: { class: "basic-landing__stamp", opacity: BLOW_OPACITY.still.landing },
          children: Object.entries(record.blessings.paths).map(([role, d]) => ({
            key: `blessings-${role}`,
            tag: "path",
            attrs: { class: `basic-landing__${role}`, d },
          })),
        },
      ],
    });
  }
  if (record.dot) {
    over.push({
      key: "dot",
      tag: "circle",
      motion: "dot",
      attrs: {
        class: `combat-impact basic-dot${record.shape === "meaning_dot" ? " basic-dot--meaning" : ""}`,
        "data-basic-class": classKey,
        cx: record.dot.at[0],
        cy: record.dot.at[1],
        r: record.dot.r,
        opacity: BLOW_OPACITY.still.landing,
      },
    });
  }
  return { band, over };
}

/**
 * Turn a blow time (ms at 1x) into real milliseconds for one event.
 *
 * The plan puts the event's phaseImpact at blow 360 ms and phaseEnd at blow
 * 760 ms on the beat clock (Body Glide and Longer Tick included); between and
 * beyond those the mapping is a straight line. Without them the plan's
 * impactStart and settleStart phases are used, and failing those real time is
 * blow time. Pure.
 *
 * @param {JsonRecord} event
 * @param {JsonRecord} plan
 * @returns {(blow: number) => number}
 */
export function blowClock(event, plan) {
  const impact = Number(event.phaseImpact ?? plan?.phases?.impactStart ?? 360);
  const end = Number(event.phaseEnd ?? plan?.phases?.settleStart ?? 760);
  const slope =
    Number.isFinite(impact) && Number.isFinite(end) && end > impact
      ? (end - impact) / 400
      : 1;
  const base = Number.isFinite(impact) ? impact : 360;
  return (blow) => Math.max(0, base + (blow - 360) * slope);
}

/**
 * Return the keyframes of one motion, as [blow ms, frame] pairs.
 *
 * motion names what moves: "bolt" (draw-in from the Mage's end, drain into the
 * star), "streak", "ribbon", "cleave-main" / "cleave-twin" (grow out of the
 * attacker), "sparks", "landing-star" (stamp 1.2 to 1), "landing-fan" and
 * "landing-bloom" (open 0.45 to 1), "blessings" and "dot". The Comet Bolt adds
 * "comet-bolt" (the bolt's timing at 95%), "comet-head" (rides the bolt's tip
 * out from the Mage, rests under the star, gone when the bolt has drained) and
 * "comet-ring" (opens from 34% to full size around the star at 240-340 ms, a
 * white flash inside it at 240-320 ms, gone by 440 ms). record gives the cleave
 * windows and the Comet Bolt's sizes. Every list starts invisible and ends
 * invisible at or after blow 600 ms. Pure.
 *
 * @param {string} motion
 * @param {LookRecord} record
 * @returns {Array<[number, Keyframe]>}
 */
function motionFrames(motion, record) {
  const peak = BLOW_OPACITY.peak;
  const T = BLOW_TIMING;
  if (motion === "comet-head") {
    const length = Math.max(Number(record.stroke?.length ?? 0), 1e-6);
    const head = Number(record.comet?.head ?? 0);
    // The bolt's tip eases out about like 1 - (1 - t)^2, so it passes the
    // head's resting spot at about this share of its draw-in.
    const share = 1 - Math.sqrt(Math.max(0, 1 - head / length));
    const reach = Math.max(
      T.bolt.drawStart + 10,
      T.bolt.drawStart + (T.bolt.drawEnd - T.bolt.drawStart) * share,
    );
    return [
      [
        T.bolt.drawStart,
        { opacity: 0, transform: `translateX(${r2(-head)}px)`, easing: "ease-out" },
      ],
      [T.bolt.drawStart + 10, { opacity: COMET.peak }],
      [reach, { transform: "translateX(0px)" }],
      [T.bolt.drainEnd, { opacity: COMET.peak, transform: "translateX(0px)" }],
      [T.bolt.drainEnd + 1, { opacity: 0 }],
      [T.gone, { opacity: 0 }],
    ];
  }
  if (motion === "comet-ring") {
    // The flash shares this animation through the group's fill-opacity. Its
    // values are the flash's own brightness (60% at 256 ms, 38% at 280 ms, 19%
    // at 300 ms, gone at 320 ms) divided by the ring's opacity at that moment.
    return [
      [
        T.landing.start,
        { opacity: 0, transform: "scale(0.34)", fillOpacity: 0, easing: "ease-out" },
      ],
      [255, { opacity: 1, transform: "scale(0.5)" }],
      [256, { fillOpacity: 0.6 }],
      [280, { fillOpacity: 0.41 }],
      [300, { fillOpacity: 0.22 }],
      [320, { fillOpacity: 0 }],
      [T.landing.end, { opacity: 0.7, transform: "scale(1)" }],
      [440, { opacity: 0, transform: "scale(1)" }],
      [T.gone, { opacity: 0 }],
    ];
  }
  if (motion === "bolt" || motion === "comet-bolt") {
    const strokePeak = motion === "bolt" ? peak.stroke : COMET.peak;
    return [
      [
        T.bolt.drawStart,
        {
          opacity: 0,
          transform: "scaleX(0.001)",
          transformOrigin: "0% 50%",
          easing: "ease-out",
        },
      ],
      [T.bolt.drawStart + 10, { opacity: strokePeak }],
      [T.bolt.drawEnd, { transform: "scaleX(1)", transformOrigin: "0% 50%" }],
      [
        T.bolt.drainStart,
        { transform: "scaleX(1)", transformOrigin: "100% 50%", easing: "ease-in" },
      ],
      [
        T.bolt.drainEnd,
        {
          opacity: strokePeak,
          transform: "scaleX(0.001)",
          transformOrigin: "100% 50%",
        },
      ],
      [T.bolt.drainEnd + 1, { opacity: 0 }],
      [T.gone, { opacity: 0 }],
    ];
  }
  if (motion === "streak") {
    return [
      [T.streak.drawStart, { opacity: 0, transform: "scaleX(0.001)" }],
      [T.streak.drawStart + 5, { opacity: peak.stroke }],
      [T.streak.drawEnd, { transform: "scaleX(1)" }],
      [T.streak.fadeStart, { opacity: peak.stroke }],
      [T.streak.fadeEnd, { opacity: 0, transform: "scaleX(1)" }],
      [T.gone, { opacity: 0 }],
    ];
  }
  if (motion === "ribbon") {
    return [
      [
        T.ribbon.growStart,
        { opacity: 0, transform: "scale(0.6)", easing: "ease-in-out" },
      ],
      [T.ribbon.growEnd, { opacity: peak.stroke, transform: "scale(1)" }],
      [T.ribbon.fadeStart, { opacity: peak.stroke }],
      [T.ribbon.fadeEnd, { opacity: 0, transform: "scale(1)" }],
    ];
  }
  if (motion === "cleave-main" || motion === "cleave-twin") {
    const part = motion === "cleave-main" ? "main" : "twin";
    const cleave = record.cleaves.find(
      (/** @type {JsonRecord} */ each) => each.part === part,
    );
    const [start, end] = cleave ? cleave.grow : [150, 240];
    return [
      [
        start,
        { opacity: 0, transform: "scale(0.15)", easing: cleave?.easing ?? "ease-out" },
      ],
      [Math.min(start + 15, end), { opacity: peak.cleave }],
      [end, { transform: "scale(1)" }],
      [T.cleaveFade.start, { opacity: peak.cleave }],
      [T.cleaveFade.end, { opacity: 0, transform: "scale(1)" }],
    ];
  }
  if (motion === "sparks") {
    return [
      [T.sparks.start, { opacity: 0, transform: "scale(1)" }],
      [T.sparks.start + 10, { opacity: 1 }],
      [T.sparks.end, { opacity: 0, transform: "scale(1.08)" }],
      [T.gone, { opacity: 0 }],
    ];
  }
  if (motion === "landing-star") {
    return [
      [T.landing.start, { opacity: 0, transform: "scale(1.2)" }],
      [T.landing.end, { opacity: peak.landing, transform: "scale(1)" }],
      [T.landing.fadeStart, { opacity: peak.landing }],
      [T.landing.fadeEnd, { opacity: 0, transform: "scale(1)" }],
    ];
  }
  if (motion === "landing-fan" || motion === "landing-bloom") {
    return [
      [T.landing.start, { opacity: 0, transform: "scale(0.45)", easing: "ease-out" }],
      [T.landing.openEnd, { opacity: peak.landing }],
      [T.landing.end, { transform: "scale(1)" }],
      [T.landing.fadeStart, { opacity: peak.landing }],
      [T.landing.fadeEnd, { opacity: 0, transform: "scale(1)" }],
    ];
  }
  if (motion === "blessings") {
    return [
      [
        T.blessing.growStart,
        { opacity: 0, transform: "scale(0.85)", easing: "ease-out" },
      ],
      [T.blessing.growEnd, { opacity: peak.landing, transform: "scale(1)" }],
      [T.blessing.fadeStart, { opacity: peak.landing }],
      [T.blessing.fadeEnd, { opacity: 0, transform: "scale(1)" }],
    ];
  }
  return [
    [T.landing.start, { opacity: 0 }],
    [T.landing.start + 60, { opacity: peak.landing }],
    [T.landing.fadeStart, { opacity: peak.landing }],
    [T.landing.fadeEnd, { opacity: 0 }],
  ];
}

/**
 * Describe the animation of one moving node, or null when nothing animates.
 *
 * element is the node; motion its motion name; record its blow; mount gives
 * the plan, paint options and the event. A settled install or motion "off"
 * animates nothing (the DOM still is the picture). Reduced motion fades the
 * node in over 70 ms, holds to 120 ms and is gone by 150 ms, starting at the
 * event's scaled phaseStart, with no movement (the Comet Bolt's parts peak at
 * 95%, and its flash stays dark because only the full motion lights it).
 * Otherwise the motion's blow-time keyframes are placed on the real clock
 * (blowClock); the animation lasts until the later of blow 600 ms and the
 * event's phaseEnd, so the node stays hidden until the event's own phase has
 * ended. fill is "backwards": before it starts it shows its first, invisible
 * frame, and when it ends the DOM still returns. Return the AnimationSpec (id
 * `mbg:<epoch>:<event>:basic-<motion>`).
 *
 * @param {Element} element
 * @param {string} motion
 * @param {LookRecord} record
 * @param {EventMount} mount
 * @returns {AnimationSpec | null}
 */
function motionSpec(element, motion, record, mount) {
  const { plan, options } = mount;
  if (options.settled || options.motionMode === "off") return null;
  const event = record.event;
  if (options.motionMode === "reduced") {
    const scale =
      Number(plan.phases?.reducedTotal ?? 220) /
      Math.max(Number(plan.phases?.total ?? 900), 1);
    const peak = motion.startsWith("cleave")
      ? BLOW_OPACITY.peak.cleave
      : motion.startsWith("landing") || motion === "blessings" || motion === "dot"
        ? BLOW_OPACITY.peak.landing
        : motion === "sparks"
          ? 0
          : motion.startsWith("comet")
            ? COMET.peak
            : BLOW_OPACITY.peak.stroke;
    const R = BLOW_TIMING.reduced;
    return animationSpec(
      element,
      [
        { opacity: 0 },
        { opacity: peak, offset: R.fadeIn / R.end },
        { opacity: peak, offset: R.holdEnd / R.end },
        { opacity: 0 },
      ],
      {
        delay: Math.max(0, Number(event.phaseStart ?? 0) * scale),
        duration: R.end,
        fill: "backwards",
      },
      plan,
      event,
      `basic-${motion}`,
    );
  }
  const real = blowClock(event, plan);
  const frames = motionFrames(motion, record);
  const startReal = real(frames[0][0]);
  const endReal = Math.max(
    real(Math.max(BLOW_TIMING.gone, frames[frames.length - 1][0])),
    Number.isFinite(Number(event.phaseEnd)) ? Number(event.phaseEnd) : 0,
  );
  const duration = endReal - startReal;
  if (!(duration > 0)) return null;
  /** @type {Keyframe[]} */
  const keyframes = frames.map(([blow, frame]) => ({
    ...frame,
    offset: clamp((real(blow) - startReal) / duration, 0, 1),
  }));
  if (Number(keyframes.at(-1)?.offset ?? 1) < 1) {
    keyframes.push({ opacity: 0, offset: 1 });
  }
  return animationSpec(
    element,
    keyframes,
    { delay: startReal, duration, fill: "backwards" },
    plan,
    event,
    `basic-${motion}`,
  );
}

/**
 * Create one described node and its children under parent.
 *
 * doc creates SVG nodes; node is a DrawNode; still leaves out playOnly nodes.
 * Each node gets data-basic-part = its key. Return a list of [element, motion]
 * for every created node that has a motion. DOM changes only.
 *
 * @param {Document} doc
 * @param {Element} parent
 * @param {DrawNode} node
 * @param {boolean} still
 * @returns {Array<[Element, string]>}
 */
function createNode(doc, parent, node, still) {
  if (still && node.playOnly) return [];
  const element = effectNode(doc, node.tag, {
    ...node.attrs,
    "data-basic-part": node.key,
  });
  parent.append(element);
  /** @type {Array<[Element, string]>} */
  const moving = node.motion ? [[element, node.motion]] : [];
  for (const child of node.children ?? []) {
    moving.push(...createNode(doc, element, child, still));
  }
  return moving;
}

/**
 * Draw Basic records into the painter's groups.
 *
 * records come from records(); doc creates the nodes; mount (optional) is the
 * painter's EventMount. Without mount, detached settled groups are made (no
 * animation), for pictures such as the Visual Key. Strokes, cleaves and the
 * Comet Bolt's head go in mount.underlay (the Strike Band), or in mount.group
 * when there is no route group; landings, the Comet Bolt's ring, sparks,
 * blessings and dots go in mount.group; the allocator's leader for a moved
 * landing goes in mount.connector. A still (settled, or motion "off") draws
 * the DOM still only, with no glows, sparks or flash; otherwise one animation
 * description per moving node is added to mount.animationSpecs (at most 4 for
 * the Comet Bolt, at most 2 for every other Basic). The landing group keeps the
 * `combat-impact` class and the allocator's placement data. Export-safe SVG
 * only: flat fills, strokes, opacity and transforms. Return the groups drawn
 * into ([group, connector] plus the route group when there is one), or [] for
 * no records.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @param {EventMount} [mount]
 * @returns {SVGElement[]}
 */
export function build(records, doc, mount) {
  const record = records[0];
  if (record === undefined) {
    return [];
  }
  const target = mount ?? detachedEventMount(doc, record.event);
  const still = target.options.settled || target.options.motionMode === "off";
  for (const each of records) {
    const { band, over } = drawing(each);
    /** @type {Array<[Element, string]>} */
    const moving = [];
    for (const node of band) {
      moving.push(...createNode(doc, target.underlay ?? target.group, node, still));
    }
    for (const node of over) {
      moving.push(...createNode(doc, target.group, node, still));
    }
    const landing = target.group.querySelector('[data-basic-part="landing"]');
    if (landing) {
      const event = each.event;
      assignLayoutPlacement(
        landing,
        event.impactLayoutKey,
        event.impactBounds,
        event.impactDisposition,
        event.impactCueCollisionFree,
      );
      appendAllocatedLeader(
        doc,
        target.connector,
        "combat-cue__leader combat-cue__leader--impact",
        each.leader,
      );
    }
    for (const [element, motion] of moving) {
      const spec = motionSpec(element, motion, each, target);
      if (spec) target.animationSpecs.push(spec);
    }
  }
  return mountNodes(target);
}

/**
 * Write one described node's attributes onto its existing element, and its
 * children's, matched by data-basic-part inside root. Missing elements are
 * skipped (nothing is created). DOM changes only.
 *
 * @param {Element} root
 * @param {DrawNode} node
 */
function syncNode(root, node) {
  const element = root.querySelector(`[data-basic-part="${node.key}"]`);
  if (element) setEffectAttributes(element, node.attrs);
  for (const child of node.children ?? []) syncNode(root, child);
}

/**
 * Move a Basic's nodes to its new geometry (after a resize or camera change).
 *
 * nodes are [group, connector, route group or nothing] as build returned;
 * records holds the event's new record from records(). Every existing part is
 * rewritten in place (positions, paths, widths) by its data-basic-part key,
 * and the landing's placement data and leader follow. Running animations keep
 * going: their keyframes are relative (scale, opacity and percentage
 * origins), so they stay right on the new geometry. The one exception is the
 * Comet Bolt's head while it travels out (blow 80 ms until it reaches the
 * star): its start offset is in pixels, so for that tick it starts from the
 * old distance; from then on it rests at its new spot. Parts that would appear
 * or vanish (a changed detail level or shape) are not added or removed; the
 * next installed frame draws them. Return undefined.
 *
 * @param {ReadonlyArray<Element | null>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  const record = records[0];
  const [group, connector = null, underlay = null] = nodes;
  if (record === undefined || !group) {
    return;
  }
  const { band, over } = drawing(record);
  for (const node of band) syncNode(underlay ?? group, node);
  for (const node of over) syncNode(group, node);
  const landing = group.querySelector('[data-basic-part="landing"]');
  if (landing) {
    const event = record.event;
    assignLayoutPlacement(
      landing,
      event.impactLayoutKey,
      event.impactBounds,
      event.impactDisposition,
      event.impactCueCollisionFree,
    );
  }
  if (connector?.querySelector(".combat-cue__leader--impact")) {
    syncAllocatedLeader(connector, ".combat-cue__leader--impact", record.leader);
  }
}

/**
 * Return the tooltip for a Basic record.
 *
 * With ability paint on, the activation explanation from `explanations.js`
 * (who used which Basic on whom, the Basic's description) under the blow's
 * own title, for example "Arcane Blast · Mage · ID 0 → Mage · ID 5 · 15
 * Damage"; with it off, the shared event explanation of the meaning mark.
 * Both read only the plan event, so in Agent POV a champion the team did not
 * see is named by class only ("Mage") or "Out Of Sight". Return a semantic
 * tooltip descriptor (`tooltip.js`).
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  const view = paintAwareExplanationEvent(record.event);
  if (!paintPartEnabled(record.event, "ability")) {
    return explainChoreographyEvent(view);
  }
  const base = explainActivation(view);
  return createSemanticDescriptor({
    kind: base.kind,
    id: base.id,
    title: record.tooltipText ?? base.title,
    tone: base.tone,
    accent: base.accent,
    summary: base.summary === null ? base.title : base.summary,
    rows: base.rows,
    sections: base.sections,
    metadata: base.metadata,
    anchor: base.anchor,
  });
}
