/**
 * @file The Statuses And Auras look (V5 Rune Zones) for all three clients.
 *
 * It draws:
 * - Rune Zones: status marks inside each living champion's token, at most 3
 *   per body, all inside 0.72R (R = body radius): the crown (Charge (Stun) and
 *   Crippling Poison (Stun): 3 stars; Freezing Trap (Stun): a hexagon shell
 *   and 4 crystals), the roots (Charge (Slow) and Crippling Poison (Slow): a
 *   chain of 5 beads; Serrated Arrows (Slow): a thread), Crippling Poison
 *   (Healing Reduction) veins in the corners, Blessing of Freedom (Slow
 *   Resistance) wings at the sides (only when it matters) and Burst (Damage
 *   Boost) runes. The code below keeps the design store's short names for
 *   these drawings (sapped, frozen, hamstrung, snagged, withering, unbound,
 *   kindled); players only ever read the status names. Counted pieces go out
 *   one per tick and stay as faint ghosts (part "status-marks").
 * - The Rune Ring at 0.68R (same part): six 5-tick arcs at 55% fill as the
 *   Ultimate charges; when ready, a 1.5u ring and 4 gems.
 * - Thaw (a wet ring) and Shatter (a crack across the token at Impact and six
 *   shards at Settle) when a Freezing Trap ends or breaks, kept for that tick.
 *   Switch freezing_trap_break: off, the ice simply cuts out at Settle.
 * - Status Clear On Death (switch status_clear_on_death, same part): a
 *   champion that falls this tick keeps its tick-start marks and Rune Ring
 *   until its Fall beat. On, they turn to Ash and fade over 160 ms; off, they
 *   cut out at the Fall.
 * - Barrier Held (switch barrier_held, on by default, same part): a bronze
 *   collar closes just outside the rim of a champion that lived only because
 *   another champion's Guardian's Barrier (a Warrior's aura) cut the damage it
 *   took. It flashes at Impact and rests faint for the tick, so the paused
 *   frame shows a faint collar.
 * - Ward Circles: each Warrior's and Mage's aura reach (2.0 units), solid, in
 *   the class color, Team B double-lined, with studs (part "aura-fields").
 *   Aura Glints (switch aura_glints, off by default): during play, one stud
 *   glints at Impact on a tick when that aura helped an ally inside it. Barrier
 *   Held also flashes the Warrior's studs. Both are play only.
 * - The Ward Lift (switch spawn_shield_expiry, through the event painter): a
 *   dashed silver ring lifts off a champion at Settle on the tick its Spawn
 *   Shield ends. Play only; the paused frame does not show it.
 * - Status Plates (status and aura docks, off by default) and the old Ultimate
 *   cooldown docks, in the `durable-status-modifier` layer, at the places
 *   `scene.js` chose with the shared dock allocator (`layout.js`); and plain
 *   pulses (Ultimate Ready) through the event painter.
 * - No off-body status-change badge: a status that starts, ends, breaks or is
 *   cleared shows only on the body (its Rune Zones mark installs or goes out at
 *   Settle; Thaw and Shatter for a Freezing Trap). records gives nothing for a
 *   "status_lifecycle" plan event other than a Spawn Shield expiry, so the
 *   painter draws no group, ring or leader for it. The event feed still lists
 *   the change.
 * - statusTray: the 2 chips + "+N" status tray for the HUD and cards.
 * `scene.js` and `choreography-painter.js` call records, then build, and update
 * on resize or camera change; fitDockValues keeps dock numbers readable.
 *
 * Timing: the DOM always holds the END state. A mark or ring piece that
 * changes this tick is drawn with data-beat-hold (before or after side): at
 * "settle" for installs and expiries, "impact" for the Shatter crack, the
 * Barrier Held collar and the glints, and "fall" (keyed by the champion) for
 * a falling champion's marks. The choreographer turns these into
 * fill-backwards opacity overrides, or a one-off flash for a node with
 * data-beat-pulse. No timers or CSS animations here.
 *
 * Boundary (Agent POV): records reads only what it is given: the audience's
 * own arena scene and its own events (Oracle View latest_events, Agent POV
 * visual_events), the audience's own tick-start spots (Agent POV
 * trajectories), allocator placements and one plan event. Only bodies in the
 * given scene get marks, so an unseen champion has none. In Oracle View Barrier
 * Held and Aura Glints read the host's exact damage rows; Agent POV gets no
 * such rows, so there they are worked out from champions the team saw at both
 * ends of the tick and never from an unseen one. Nothing here imports a
 * researcher helper or reads researcher_space or the match summary. Colors
 * come from CSS tokens (src/styles/statuses.css). Export-safe SVG only: flat
 * fills, strokes, dashes, opacity and transforms.
 */
import { formatCompactDisplayNumber, formatDisplayNumber } from "../display.js";
import {
  explainAura,
  explainCooldown,
  explainModifier,
  explainOverflow,
  explainPovOverflow,
  explainPovStatus,
  explainStatus,
} from "../explanations.js";
import { createSvgIcon, statusSigilKey } from "../icons.js";
import { registerTooltipOwner } from "../tooltip.js";
import {
  classTokenFromId,
  resolveVisualToken,
  statusTokenIdFromCatalogId,
  ultimateTokenFromClassId,
} from "../vocabulary.js";
import {
  agentIdentity,
  appendAllocatedLeader,
  asArray,
  assignLayoutPlacement,
  cssIdentifier,
  detachedEventMount,
  displayIdentityAttributes,
  displayIdentityRecord,
  EVENT_BUDGET,
  effectNode,
  explainChoreographyEvent,
  finiteNumber,
  isRecord,
  mountNodes,
  paintAwareExplanationEvent,
  paintParts,
  screenPoint,
  setDisplayIdentityData,
  setEffectAttributes,
  setNodeAttributes,
  svgNode,
  syncAllocatedLeader,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").JsonRecord} JsonRecord
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {import("./look-support.js").EventMount} EventMount
 * @typedef {{cellWidth: number, cellHeight: number, cellGap: number}} DockDimensions
 * @typedef {(
 *   | {kind: "status", status: unknown, recipient: JsonRecord, sourceAgents: ReadonlyArray<JsonRecord>}
 *   | {kind: "pov-status", status: unknown, recipient: JsonRecord}
 *   | {kind: "modifier", modifier: unknown, recipient: JsonRecord}
 *   | {kind: "overflow", items: ReadonlyArray<unknown>, dockKind: "status" | "modifier", recipient: JsonRecord, sourceAgents: ReadonlyArray<JsonRecord>}
 *   | {kind: "pov-overflow", items: ReadonlyArray<unknown>, recipient: JsonRecord}
 * )} DockTooltipInput
 *   The plain data a dock tooltip is built from, chosen by the caller.
 */

/**
 * Sizes in CSS pixels: status, Ultimate cooldown and modifier dock cells (the
 * allocator places docks from these) and the plain pulse box. A status change
 * has no box: it shows on the body, so the plan reserves no arena space for it.
 */
export const FOOTPRINT = Object.freeze({
  statusDock: Object.freeze({ cellWidth: 28, cellHeight: 18, cellGap: 2 }),
  cooldownDock: Object.freeze({ cellWidth: 38, cellHeight: 18, cellGap: 2 }),
  modifierDock: Object.freeze({ cellWidth: 42, cellHeight: 16, cellGap: 2 }),
  pulseCue: Object.freeze({ width: 62, height: 62 }),
});

/** The plan's shared per-event allowance; docks are bounded by nine cells each. */
export const BUDGET = EVENT_BUDGET;

/** Every paint part this look draws, with the filter that switches it. */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "aura_field" }, "aura_fields"],
  [{ surface: "durable", kind: "aura_modifier_badge" }, "aura_modifier_badges"],
  [{ surface: "durable", kind: "duration_status_badge" }, "duration_status_badges"],
  [{ surface: "durable", kind: "pov_duration_status_badge" }, "duration_status_badges"],
  [{ surface: "durable", kind: "cooldown_badge" }, "cooldown_effects"],
  [{ surface: "transient", kind: "cooldown", semantic: "started" }, "cooldown_effects"],
  [{ surface: "transient", kind: "cooldown", semantic: "ready" }, "cooldown_effects"],
  [
    {
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "applied",
      part: "effect",
    },
    "status_application",
  ],
  [
    {
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "expired",
      part: "effect",
    },
    "natural_status_expiry",
  ],
  [
    {
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "trap_broken",
      part: "effect",
    },
    "freezing_trap_break",
  ],
  [
    {
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "cleared_by_death",
      part: "effect",
    },
    "status_clear_on_death",
  ],
  [{ surface: "transient", kind: "spawn_shield_expiry" }, "spawn_shield_expiry"],
  [{ surface: "durable", kind: "status_ash" }, "status_clear_on_death"],
  [{ surface: "durable", kind: "barrier_held" }, "barrier_held"],
  [{ surface: "durable", kind: "aura_glint" }, "aura_glints"],
]);

/**
 * Rune Zones numbers (store skins/statuses-and-auras.js, option rune-zones,
 * sizes at body radius R in CSS pixels; u = R / 23). One owner: change a
 * number here, never in a caller.
 *
 * - maxMarks: at most 3 status marks per body (priority Frozen, Sapped,
 *   Withering, Hamstrung or Snagged, Kindled, Unbound).
 * - ghost: opacity of a spent piece (the silhouette never changes).
 * - ringRadius 0.68R: the Rune Ring; six arcs of ringSegmentTicks (5) ticks
 *   over ringTicks (30, every class's Ultimate cooldown); unlit track at
 *   trackOpacity, lit arcs at litOpacity; ready: a 1.5u ring and 4 gems.
 * - markRadius 0.72R: every mark stays inside it.
 * - installFadeMs / expiryFadeMs / meltFadeMs / thawFadeMs / shatterFadeMs:
 *   how long a mark takes to appear, go out, lose a crystal, thaw or shatter,
 *   counted from the Settle beat.
 * - crackFadeMs, crackOpacity: the Shatter crack comes in over 60 ms at Impact
 *   and stays at 80% for the tick (spec: Ice Crack, 1 px at 80%).
 * - giveWayMs, ashFadeMs: Status Clear On Death. At a champion's Fall its
 *   marks give way in 53 ms (the first third) while their Ash copy flashes in
 *   and fades by 160 ms (store: Ash over 160 ms from the Fall).
 * - collarRadius (1.06R), collarRestOpacity (35%), collarInMs (80 ms) and
 *   collarFlashMs (640 ms): Barrier Held. The collar closes at Impact, flashes
 *   to full and rests faint for the rest of the tick (store kf 240-900 ms).
 * - glintMs: an Aura Glint or a Barrier Held stud flash lasts 120 ms from
 *   Impact (store 240-360 ms).
 * - wardLiftRadius (0.95R), wardLiftOpacity (60%), wardLiftPieces (24): the
 *   dashed silver ring of the Ward Lift (store wardLift); its circumference
 *   is cut into 24 equal pieces, alternately dash and gap (12 dashes).
 * - wardRadiusUnits: the aura reach the Ward Circle shows (2.0 world units;
 *   the circle itself uses each field's own radius, which is 2.0). Barrier Held
 *   and Agent POV Aura Glints count a champion as covered when its tick-start
 *   center is within this reach of the emitter's.
 * - barrierMultiplier: Guardian's Barrier multiplies the damage a covered
 *   champion takes by 0.85 for each covering Warrior.
 */
export const RUNE_ZONES = Object.freeze({
  maxMarks: 3,
  ghost: 0.22,
  ringRadius: 0.68,
  markRadius: 0.72,
  ringTicks: 30,
  ringSegments: 6,
  ringSegmentTicks: 5,
  trackOpacity: 0.12,
  litOpacity: 0.55,
  installFadeMs: 200,
  expiryFadeMs: 200,
  meltFadeMs: 300,
  thawFadeMs: 480,
  shatterFadeMs: 60,
  crackFadeMs: 60,
  crackOpacity: 0.8,
  giveWayMs: 53,
  ashFadeMs: 160,
  collarRadius: 1.06,
  collarRestOpacity: 0.35,
  collarInMs: 80,
  collarFlashMs: 640,
  glintMs: 120,
  wardLiftRadius: 0.95,
  wardLiftOpacity: 0.6,
  wardLiftPieces: 24,
  wardRadiusUnits: 2,
  barrierMultiplier: 0.85,
  hamstrungBelow: 0.85,
});

/** Status tray order (most limiting first) for statusTray. */
const TRAY_ORDER = Object.freeze([
  "stun_hunter_trap",
  "stun_warrior_charge",
  "stun_rogue_poison",
  "slow_warrior_charge",
  "slow_rogue_poison",
  "slow_hunter_basic",
  "anti_heal_rogue_poison",
  "priest_freedom",
  "mage_burst",
]);

/** Slow token -> [mark kind, default movement multiplier]. */
/** @type {Readonly<Record<string, readonly ["charge" | "poison" | "snag", number]>>} */
const SLOW_KINDS = Object.freeze({
  slow_warrior_charge: ["charge", 0.5],
  slow_rogue_poison: ["poison", 0.5],
  slow_hunter_basic: ["snag", 0.85],
});

/** Mark kind -> paint (a class color token in statuses.css). */
const KIND_PAINT = Object.freeze({
  charge: "warrior",
  poison: "rogue",
  snag: "hunter",
});

/** Class id -> class name for paints. */
const CLASS_PAINT = Object.freeze({
  1: "mage",
  2: "warrior",
  3: "hunter",
  4: "rogue",
  5: "priest",
});

/**
 * @typedef {{
 *   frozen: number | null,
 *   sapped: "charge" | "poison" | null,
 *   roots: {kind: "charge" | "poison" | "snag", hamstrung: boolean, beads: number} | null,
 *   withering: number,
 *   kindled: number,
 *   unbound: boolean,
 * }} RuneMarks
 *   One body's marks. frozen: lit crystals 0-4 of the Freezing Trap stun, or
 *   null when not frozen (0 means frozen with every crystal spent or unknown).
 *   sapped: the Charge or Poison stun stars (null: none). roots: the strongest
 *   slow (hamstrung chain with 1-5 lit beads, or the Snagged thread). withering
 *   and kindled: lit veins (0-4) and runes (0-5); 0 means none. unbound: the
 *   Freedom wings, shown only when Freedom matters (slowed below x0.85, not
 *   stunned, no Spawn Shield).
 */

/**
 * @typedef {{
 *   id: string,
 *   tag: "path" | "circle",
 *   attributes: Readonly<Record<string, string | number>>,
 *   paint: string,
 *   opacity: number,
 * }} RunePiece
 *   One drawn element of a body's marks, in the body's own frame (center 0,0,
 *   CSS pixels, y down). id is stable across START and END so the two can be
 *   compared; paint is a space-separated list of `rune-*` classes.
 */

/**
 * @typedef {RunePiece & {
 *   side: "steady" | "before" | "after",
 *   fadeMs: number,
 *   fromOpacity: number,
 *   beat?: "settle" | "impact" | "fall",
 *   pulse?: number,
 *   batch?: string,
 * }} RuneDrawPiece
 *   A piece with its beat: "steady" is the same at START and END; "before" is
 *   shown from tick start and goes out at its beat; "after" comes in at its
 *   beat. beat names that beat: "settle" (the default), "impact" (blow time
 *   260 ms) or "fall" (this champion's Fall beat). fadeMs is the change length
 *   in ms (1 is a cut); fromOpacity the opacity before the change. pulse, when
 *   set, makes an "after" piece a one-off flash instead: it rests at opacity 0
 *   and rises to pulse and back to 0 over fadeMs from its beat (normal motion
 *   only), so the paused frame never shows it. batch, when set, names a group
 *   of pieces that share one beat: build draws them inside one `g.rune-batch`
 *   that carries the beat (from the batch's first piece), and each piece keeps
 *   only its own resting opacity. A batch costs one beat override instead of
 *   one per piece. A batched "before" or "after" piece's fromOpacity and pulse
 *   are the whole batch's (1 means each piece's own opacity).
 */

/** Round to hundredths for SVG numbers. */
const f = (/** @type {number} */ value) => Math.round(value * 100) / 100;
/** Degrees to radians. */
const DEG = Math.PI / 180;
/**
 * Point at radius r and angle deg (0 = 12 o'clock, clockwise), y down.
 *
 * @param {number} r
 * @param {number} deg
 * @returns {[number, number]}
 */
const pol = (r, deg) => [r * Math.sin(deg * DEG), -r * Math.cos(deg * DEG)];
/** "x,y" text for a point. */
const pt = (/** @type {[number, number]} */ p) => `${f(p[0])},${f(p[1])}`;
/**
 * Clockwise arc path of radius r from angle a0 to a1 (degrees).
 *
 * @param {number} r
 * @param {number} a0
 * @param {number} a1
 */
const arcD = (r, a0, a1) =>
  `M${pt(pol(r, a0))} A${f(r)},${f(r)} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${pt(pol(r, a1))}`;
/**
 * Four-point star path centered at c, half height h, waist w.
 *
 * @param {[number, number]} c
 * @param {number} h
 * @param {number} w
 */
const star4 = (c, h, w) =>
  `M${f(c[0])},${f(c[1] - h)} L${f(c[0] + w)},${f(c[1] - w)} L${f(c[0] + h)},${f(c[1])} L${f(c[0] + w)},${f(c[1] + w)} L${f(c[0])},${f(c[1] + h)} L${f(c[0] - w)},${f(c[1] + w)} L${f(c[0] - h)},${f(c[1])} L${f(c[0] - w)},${f(c[1] - w)} Z`;
/**
 * Diamond path centered at c with half size h.
 *
 * @param {[number, number]} c
 * @param {number} h
 */
const diamond = (c, h) =>
  `M${f(c[0])},${f(c[1] - h)} L${f(c[0] + h)},${f(c[1])} L${f(c[0])},${f(c[1] + h)} L${f(c[0] - h)},${f(c[1])} Z`;

/**
 * A tapered, waving Withering vein from the rim inward (store veinPath).
 *
 * @param {number} t Angle in degrees.
 * @param {number} rOut
 * @param {number} rIn
 * @param {number} wOut
 * @param {number} wIn
 * @param {number} amp
 * @param {number} sign
 */
function veinPath(t, rOut, rIn, wOut, wIn, amp, sign) {
  const d = [Math.sin(t * DEG), -Math.cos(t * DEG)];
  const n = [Math.cos(t * DEG), Math.sin(t * DEG)];
  /** @type {[number, number][]} */
  const left = [];
  /** @type {[number, number][]} */
  const right = [];
  for (let j = 0; j <= 10; j += 1) {
    const q = j / 10;
    const r = rOut - (rOut - rIn) * q;
    const hw = wOut + (wIn - wOut) * q;
    const off = amp * Math.sin(2 * Math.PI * q) * sign;
    const cx = d[0] * r + n[0] * off;
    const cy = d[1] * r + n[1] * off;
    left.push([cx + n[0] * hw, cy + n[1] * hw]);
    right.push([cx - n[0] * hw, cy - n[1] * hw]);
  }
  return `M${left.map(pt).join(" L")} L${right.reverse().map(pt).join(" L")} Z`;
}

/**
 * @typedef {{
 *   R: number, u: number, ringR: number, ringArcs: string[],
 *   gemAt: [number, number][], gemH: number,
 *   starAt: [number, number][], starH: number, starW: number, orbit: string,
 *   hex: string, facets: string, crystalAt: [number, number][], crystalL: number,
 *   chain: string, beadAt: [number, number][], beadR: number, snag: string,
 *   veins: string[], wings: string[], runes: string[], shards: string, thawR: number,
 *   crack: string, collarR: number, collarRivets: string,
 * }} RuneGeometry
 *   Rune Zones geometry for one body radius, in CSS pixels around the body
 *   center (y down): Rune Ring radius and arcs, gem spots, star, crystal and
 *   bead spots, the mark paths, the Shatter crack (store polyline -9,-6 ...
 *   9,5 at R 23, scaled by u) and the Barrier Held collar (radius 1.06R and
 *   twelve 3 px rivets pointing in, every 30 degrees).
 */

/** Geometry cache by body radius rounded to hundredths. @type {Map<number, RuneGeometry>} */
const GEOMETRY = new Map();

/**
 * Return the Rune Zones geometry for body radius R (CSS pixels), cached by R
 * rounded to hundredths. Every number is the store's (sizes at R 23, scaled
 * by u = R / 23). Pure.
 *
 * @param {number} R
 * @returns {RuneGeometry}
 */
function runeGeometry(R) {
  const key = f(R);
  const cached = GEOMETRY.get(key);
  if (cached) return cached;
  const u = R / 23;
  const ringR = RUNE_ZONES.ringRadius * R;
  const half = Math.max(1.5, 2 * u) / ringR / DEG / 2;
  const c = 0.7 * R;
  const s3 = Math.sqrt(3) / 2;
  const rc = 0.6 * R;
  const b = pol(rc, 180);
  const runeAngles = [36, 108, 180, 252, 324];
  const veinAngles = [310, 50, 120, 240];
  /** @param {number} sx */
  const wing = (sx) =>
    `M${f(sx * 11.4 * u)},${f(-5 * u)} Q${f(sx * 17.4 * u)},0 ${f(sx * 11.4 * u)},${f(5 * u)} Q${f(sx * 13.2 * u)},0 ${f(sx * 11.4 * u)},${f(-5 * u)} Z`;
  const geometry = Object.freeze({
    R,
    u,
    ringR,
    ringArcs: [0, 1, 2, 3, 4, 5].map((k) =>
      arcD(ringR, 60 * k + half, 60 * k + 60 - half),
    ),
    gemAt: [0, 90, 180, 270].map((t) => pol(ringR, t)),
    gemH: 0.096 * R,
    starAt: [-20, 0, 20].map((t) => pol(0.62 * R, t)),
    starH: 0.115 * R,
    starW: 0.036 * R,
    orbit: arcD(0.62 * R, -34, 34),
    hex: `M0,${f(-c)} L${f(s3 * c)},${f(-c / 2)} L${f(s3 * c)},${f(c / 2)} L0,${f(c)} L${f(-s3 * c)},${f(c / 2)} L${f(-s3 * c)},${f(-c / 2)} Z`,
    facets: `M${f(-6.97 * u)},${f(-12.07 * u)} L0,${f(-8 * u)} L${f(6.97 * u)},${f(-12.07 * u)} M${f(-13.94 * u)},0 L${f(-9 * u)},${f(2 * u)} M${f(13.94 * u)},0 L${f(9 * u)},${f(2 * u)}`,
    crystalAt: [-33, -11, 11, 33].map((t) => pol(0.64 * R, t)),
    crystalL: 0.085 * R,
    chain: arcD(rc, 144, 216),
    beadAt: [144, 162, 180, 198, 216].map((t) => pol(rc, t)),
    beadR: Math.max(1, 0.072 * R),
    snag: `${arcD(rc, 158, 202)} M${f(-1.7 * u)},${f(b[1] + 1.7 * u)} L0,${f(b[1])} L${f(1.7 * u)},${f(b[1] + 1.7 * u)}`,
    veins: veinAngles.map((t) =>
      veinPath(t, 0.705 * R, 0.52 * R, 1.25 * u, 0.2 * u, 0.04 * R, t > 180 ? -1 : 1),
    ),
    wings: [wing(1), wing(-1)],
    runes: runeAngles.map((t) => `M${pt(pol(0.56 * R, t))} L${pt(pol(0.735 * R, t))}`),
    shards: [0, 60, 120, 180, 240, 300]
      .map((t, i) => {
        const ri = 1.07 * R;
        const tip = ri + (i % 2 ? 3.2 : 4.5) * u;
        return `M${pt(pol(ri, t))} L${pt(pol(1.15 * R, t + 4))} L${pt(pol(tip + 1.2 * u, t))} L${pt(pol(1.15 * R, t - 4))} Z`;
      })
      .join(" "),
    thawR: 0.7 * R,
    crack: `M${[
      [-9, -6],
      [-4, -2],
      [-1, -4],
      [2, 1],
      [6, 0],
      [9, 5],
    ]
      .map(([x, y]) => `${f(x * u)},${f(y * u)}`)
      .join(" L")}`,
    collarR: RUNE_ZONES.collarRadius * R,
    collarRivets: Array.from({ length: 12 }, (_, i) => i * 30)
      .map(
        (t) =>
          `M${pt(pol(RUNE_ZONES.collarRadius * R, t))} L${pt(pol(RUNE_ZONES.collarRadius * R - 3, t))}`,
      )
      .join(" "),
  });
  GEOMETRY.set(key, geometry);
  return geometry;
}

/**
 * Return a status row's vocabulary token id ("stun_hunter_trap" and the like)
 * from its catalog id (token_id or status_id), or null for a non-record.
 *
 * @param {unknown} status
 * @returns {string | null}
 */
function statusToken(status) {
  if (!isRecord(status)) return null;
  const raw = status.token_id ?? status.status_id;
  return typeof raw === "string" ? statusTokenIdFromCatalogId(raw) : null;
}

/**
 * Return a status row's ticks left (duration, else remaining_duration) as a
 * non-negative integer, or null when it is missing or not a number.
 *
 * @param {unknown} status
 * @returns {number | null}
 */
function statusTicks(status) {
  if (!isRecord(status)) return null;
  const raw = status.duration ?? status.remaining_duration;
  return Number.isFinite(raw) ? Math.max(0, Math.floor(Number(raw))) : null;
}

/**
 * @typedef {{token: string, ticks: number | null, magnitude: number | null}} MarkStatus
 *   One status reduced to what the marks need: vocabulary token, ticks left
 *   (null = unknown) and movement multiplier (null = the token's default).
 */

/**
 * Return the Rune Zones marks for one living body (store marksOf).
 *
 * statuses are MarkStatus rows; spawnShield is true while the body has its
 * Spawn Shield. Unknown tokens, In Combat and Spawn Shield draw no mark.
 * Counts are capped at the piece counts (4 crystals, 5 beads, 4 veins, 5
 * runes). Pure.
 *
 * @param {ReadonlyArray<MarkStatus>} statuses
 * @param {boolean} spawnShield
 * @returns {RuneMarks}
 */
export function runeMarksFromStatuses(statuses, spawnShield) {
  const pieces = (/** @type {number | null} */ ticks, /** @type {number} */ max) =>
    ticks === null ? 0 : Math.min(max, Math.max(0, ticks));
  const trap = statuses.find((status) => status.token === "stun_hunter_trap");
  const daze = statuses
    .filter(
      (status) =>
        status.token === "stun_warrior_charge" || status.token === "stun_rogue_poison",
    )
    .at(-1);
  const slows = statuses
    .filter((status) => Object.hasOwn(SLOW_KINDS, status.token))
    .map((status) => {
      const [kind, magnitude] =
        SLOW_KINDS[/** @type {keyof typeof SLOW_KINDS} */ (status.token)];
      return {
        kind: /** @type {"charge" | "poison" | "snag"} */ (kind),
        magnitude: status.magnitude ?? magnitude,
        ticks: status.ticks ?? 0,
      };
    });
  /** @type {RuneMarks["roots"]} */
  let roots = null;
  if (slows.length > 0) {
    const strong = slows.reduce((a, b) =>
      b.magnitude < a.magnitude || (b.magnitude === a.magnitude && b.ticks > a.ticks)
        ? b
        : a,
    );
    roots = Object.freeze({
      kind: strong.kind,
      hamstrung: strong.magnitude < RUNE_ZONES.hamstrungBelow,
      beads: Math.min(5, Math.max(...slows.map((slow) => slow.ticks))),
    });
  }
  const product = Math.max(
    0.2,
    slows.reduce((total, slow) => total * slow.magnitude, 1),
  );
  const stunned = Boolean(trap || daze);
  const freedom = statuses.some((status) => status.token === "priest_freedom");
  const withering = statuses.find(
    (status) => status.token === "anti_heal_rogue_poison",
  );
  const kindled = statuses.find((status) => status.token === "mage_burst");
  return Object.freeze({
    frozen: trap ? pieces(trap.ticks, 4) : null,
    sapped: trap
      ? null
      : daze
        ? daze.token === "stun_warrior_charge"
          ? "charge"
          : "poison"
        : null,
    roots,
    withering: withering ? Math.max(1, pieces(withering.ticks, 4)) : 0,
    kindled: kindled ? Math.max(1, pieces(kindled.ticks, 5)) : 0,
    unbound: freedom && product < RUNE_ZONES.hamstrungBelow && !stunned && !spawnShield,
  });
}

/**
 * Keep at most RUNE_ZONES.maxMarks marks, in the store's priority: Frozen or
 * Sapped (the crown), Withering, Hamstrung or Snagged, Kindled, Unbound.
 * Return a new frozen RuneMarks; marks over the limit are dropped (the agent
 * details still list every status). Pure.
 *
 * @param {RuneMarks} marks
 * @returns {RuneMarks}
 */
export function runeZones(marks) {
  let room = Number(RUNE_ZONES.maxMarks);
  const take = (/** @type {boolean} */ present) => {
    if (!present || room === 0) return false;
    room -= 1;
    return true;
  };
  const crown = take(marks.frozen !== null || marks.sapped !== null);
  const withering = take(marks.withering > 0);
  const roots = take(marks.roots !== null);
  const kindled = take(marks.kindled > 0);
  const unbound = take(marks.unbound);
  return Object.freeze({
    frozen: crown ? marks.frozen : null,
    sapped: crown ? marks.sapped : null,
    withering: withering ? marks.withering : 0,
    roots: roots ? marks.roots : null,
    kindled: kindled ? marks.kindled : 0,
    unbound,
  });
}

/**
 * Return how many Rune Ring arcs are lit for an Ultimate cooldown: 6 (ready)
 * at 0 ticks or less, else floor((30 - ticks) / 5) clamped to 0-5; null for a
 * missing cooldown. Pure.
 *
 * @param {number | null} ticks Ultimate cooldown ticks left.
 * @returns {number | null}
 */
export function runeRingLitArcs(ticks) {
  if (ticks === null || !Number.isFinite(ticks)) return null;
  if (ticks <= 0) return RUNE_ZONES.ringSegments;
  return Math.min(
    RUNE_ZONES.ringSegments - 1,
    Math.max(
      0,
      Math.floor((RUNE_ZONES.ringTicks - ticks) / RUNE_ZONES.ringSegmentTicks),
    ),
  );
}

/**
 * Make one frozen RunePiece.
 *
 * @param {string} id
 * @param {"path" | "circle"} tag
 * @param {Record<string, string | number>} attributes
 * @param {string} paint
 * @param {number} [opacity]
 * @returns {RunePiece}
 */
function piece(id, tag, attributes, paint, opacity = 1) {
  return Object.freeze({
    id,
    tag,
    attributes: Object.freeze(attributes),
    paint,
    opacity: f(opacity),
  });
}

/**
 * Return the pieces of one body's Rune Ring and marks (store rzLocal, still
 * frame), in drawing order. geometry from runeGeometry; classPaint the body's
 * class name; litArcs from runeRingLitArcs (null: no ring); zones from
 * runeZones; switches.ring and switches.marks are the Cooldown Effects and
 * Status Marks switches. Pure.
 *
 * @param {RuneGeometry} g
 * @param {string} classPaint
 * @param {number | null} litArcs
 * @param {RuneMarks} zones
 * @param {{ring: boolean, marks: boolean}} switches
 * @returns {RunePiece[]}
 */
function bodyPieces(g, classPaint, litArcs, zones, switches) {
  /** @type {RunePiece[]} */
  const out = [];
  const ghost = RUNE_ZONES.ghost;
  if (switches.ring && litArcs !== null) {
    if (litArcs >= RUNE_ZONES.ringSegments) {
      out.push(
        piece(
          "ring.ready",
          "circle",
          { r: f(g.ringR), "stroke-width": f(1.5 * g.u) },
          `rune-stroke--${classPaint}`,
        ),
      );
      // A gem hides where a mark sits at 12, 3, 6 or 9 o'clock.
      const crown = zones.frozen !== null || zones.sapped !== null;
      const off = [crown, zones.unbound, zones.roots !== null, zones.unbound];
      g.gemAt.forEach((at, i) => {
        if (off[i]) return;
        out.push(
          piece(
            `ring.gem.${i}`,
            "path",
            { d: diamond(at, g.gemH), "stroke-width": 0.75 },
            `rune-fill--${classPaint} rune-stroke--ink`,
          ),
        );
      });
    } else {
      out.push(
        piece(
          "ring.track",
          "path",
          { d: g.ringArcs.join(" "), "stroke-width": f(Math.max(1, g.u)) },
          `rune-stroke--${classPaint}`,
          RUNE_ZONES.trackOpacity,
        ),
      );
      for (let i = 0; i < litArcs; i += 1) {
        out.push(
          piece(
            `ring.arc.${i}`,
            "path",
            { d: g.ringArcs[i], "stroke-width": f(Math.max(1, g.u)) },
            `rune-stroke--${classPaint}`,
            RUNE_ZONES.litOpacity,
          ),
        );
      }
    }
  }
  if (!switches.marks) return out;
  // Kindled runes (sides of the ring), lit clockwise from 36 degrees.
  if (zones.kindled > 0) {
    g.runes.forEach((d, i) => {
      const lit = i < zones.kindled;
      out.push(
        piece(
          `kindled.${i}.ink`,
          "path",
          { d, "stroke-width": f(3.2 * g.u) },
          "rune-stroke--ink",
          lit ? 1 : ghost,
        ),
      );
      out.push(
        piece(
          `kindled.${i}`,
          "path",
          { d, "stroke-width": f(lit ? 2 * g.u : 1.8 * g.u) },
          "rune-stroke--starcore",
          lit ? 1 : ghost,
        ),
      );
    });
  }
  // Withering veins in the four corners.
  if (zones.withering > 0) {
    g.veins.forEach((d, i) => {
      const lit = i < zones.withering;
      out.push(
        piece(
          `withering.${i}.ink`,
          "path",
          { d, "stroke-width": 1.2 },
          "rune-fill--ink rune-stroke--ink",
          lit ? 1 : ghost,
        ),
      );
      out.push(
        piece(`withering.${i}`, "path", { d }, "rune-fill--rogue", lit ? 1 : ghost),
      );
    });
  }
  // Roots: the Hamstrung chain with five beads, or the Snagged thread.
  if (zones.roots?.hamstrung) {
    const paint = KIND_PAINT[zones.roots.kind];
    out.push(
      piece(
        `roots.chain.ink`,
        "path",
        { d: g.chain, "stroke-width": f(2.8 * g.u) },
        "rune-stroke--ink",
        0.55,
      ),
    );
    out.push(
      piece(
        `roots.chain.${paint}`,
        "path",
        { d: g.chain, "stroke-width": f(1.4 * g.u) },
        `rune-stroke--${paint}`,
        0.75,
      ),
    );
    g.beadAt.forEach((at, i) => {
      out.push(
        piece(
          `roots.bead.${i}.${paint}`,
          "circle",
          { cx: f(at[0]), cy: f(at[1]), r: f(g.beadR), "stroke-width": 0.8 },
          `rune-fill--${paint} rune-stroke--ink`,
          i < (zones.roots?.beads ?? 0) ? 1 : ghost,
        ),
      );
    });
  } else if (zones.roots) {
    out.push(
      piece(
        "roots.snag.ink",
        "path",
        { d: g.snag, "stroke-width": f(3.4 * g.u) },
        "rune-stroke--ink",
        0.7,
      ),
    );
    out.push(
      piece(
        "roots.snag",
        "path",
        { d: g.snag, "stroke-width": f(2 * g.u) },
        "rune-stroke--hunter",
      ),
    );
  }
  // Unbound wings at the sides.
  if (zones.unbound) {
    g.wings.forEach((d, i) => {
      out.push(
        piece(
          `unbound.${i}`,
          "path",
          { d, "stroke-width": 0.6 },
          "rune-fill--priest rune-stroke--amber",
        ),
      );
    });
  }
  // The crown: Frozen shell and crystals, or the Sapped stars.
  if (zones.frozen !== null) {
    out.push(
      piece(
        "frozen.shell",
        "path",
        { d: g.hex, "stroke-width": f(g.u), "fill-opacity": 0.3 },
        "rune-fill--frostglass rune-stroke--rime",
      ),
    );
    out.push(
      piece(
        "frozen.facets",
        "path",
        { d: g.facets, "stroke-width": f(0.75 * g.u) },
        "rune-stroke--rime",
        0.6,
      ),
    );
    g.crystalAt.forEach((at, i) => {
      const L = g.crystalL;
      const d = `M${f(at[0])},${f(at[1] - L)} V${f(at[1] + L)} M${f(at[0] - 0.866 * L)},${f(at[1] - L / 2)} L${f(at[0] + 0.866 * L)},${f(at[1] + L / 2)} M${f(at[0] - 0.866 * L)},${f(at[1] + L / 2)} L${f(at[0] + 0.866 * L)},${f(at[1] - L / 2)}`;
      const level = i < (zones.frozen ?? 0) ? 1 : ghost;
      out.push(
        piece(
          `frozen.crystal.${i}.ink`,
          "path",
          { d, "stroke-width": f(2.4 * g.u) },
          "rune-stroke--ink",
          level,
        ),
      );
      out.push(
        piece(
          `frozen.crystal.${i}`,
          "path",
          { d, "stroke-width": f(1.2 * g.u) },
          "rune-stroke--rime",
          level,
        ),
      );
    });
  } else if (zones.sapped) {
    const paint = KIND_PAINT[zones.sapped];
    out.push(
      piece(
        `sapped.orbit.${paint}`,
        "path",
        { d: g.orbit, "stroke-width": f(0.9 * g.u) },
        `rune-stroke--${paint}`,
        0.45,
      ),
    );
    g.starAt.forEach((at, i) => {
      const d = star4(at, g.starH, g.starW);
      out.push(
        piece(
          `sapped.star.${i}.ink`,
          "path",
          { d, "stroke-width": 1.5 },
          "rune-fill--ink rune-stroke--ink",
        ),
      );
      out.push(
        piece(`sapped.star.${i}.${paint}`, "path", { d }, `rune-fill--${paint}`),
      );
    });
  }
  return out;
}

/**
 * @typedef {{
 *   applied: Set<string>,
 *   expired: Set<string>,
 *   broken: Set<string>,
 *   cleared: Set<string>,
 *   cooldownReady: boolean,
 *   cooldownStarted: boolean,
 *   died: boolean,
 *   respawned: boolean,
 *   shieldExpired: boolean,
 * }} BodyTickEvents
 *   What this audience's own events say happened to one body this tick:
 *   status tokens applied or refreshed, aged out, broken by damage and cleared
 *   by its death; whether its Ultimate became ready or was spent; and whether
 *   it fell (agent_died), came back (agent_respawned) or lost its Spawn Shield
 *   (spawn_shield_expired) this tick.
 */

/**
 * Group this audience's own raw events (Oracle `latest_events`, Agent POV
 * `visual_events`) by recipient presentation key. Reads event_kind,
 * recipient_anchor / agent_anchor presentation_key and status_id; rows without
 * a key are skipped. Pure.
 *
 * @param {ReadonlyArray<unknown>} events
 * @returns {Map<string, BodyTickEvents>}
 */
function eventsByBody(events) {
  /** @type {Map<string, BodyTickEvents>} */
  const byBody = new Map();
  for (const event of events) {
    if (!isRecord(event)) continue;
    const anchor = isRecord(event.recipient_anchor)
      ? event.recipient_anchor
      : isRecord(event.agent_anchor)
        ? event.agent_anchor
        : null;
    const key =
      typeof anchor?.presentation_key === "string" ? anchor.presentation_key : null;
    if (key === null) continue;
    let entry = byBody.get(key);
    if (!entry) {
      entry = {
        applied: new Set(),
        expired: new Set(),
        broken: new Set(),
        cleared: new Set(),
        cooldownReady: false,
        cooldownStarted: false,
        died: false,
        respawned: false,
        shieldExpired: false,
      };
      byBody.set(key, entry);
    }
    const kind = event.event_kind;
    const token =
      typeof event.status_id === "string"
        ? statusTokenIdFromCatalogId(event.status_id)
        : null;
    if (
      (kind === "status_applied" || kind === "status_refreshed_or_extended") &&
      token
    ) {
      entry.applied.add(token);
    } else if (kind === "status_aged_to_zero" && token) {
      entry.expired.add(token);
    } else if (kind === "status_broken_by_damage" && token) {
      entry.broken.add(token);
    } else if (kind === "status_cleared_by_new_death" && token) {
      entry.cleared.add(token);
    } else if (kind === "cooldown_ready") {
      entry.cooldownReady = true;
    } else if (kind === "cooldown_started") {
      entry.cooldownStarted = true;
    } else if (kind === "agent_died") {
      entry.died = true;
    } else if (kind === "agent_respawned") {
      entry.respawned = true;
    } else if (kind === "spawn_shield_expired") {
      entry.shieldExpired = true;
    }
  }
  return byBody;
}

/**
 * Return the presentation key of an event anchor, or null.
 *
 * @param {unknown} anchor An event anchor such as recipient_anchor.
 * @returns {string | null}
 */
function anchorKey(anchor) {
  return isRecord(anchor) && typeof anchor.presentation_key === "string"
    ? anchor.presentation_key
    : null;
}

/**
 * Return an anchor's world spot [x, y] when it was taken at tick start
 * (phase "transition_start", or no phase), else null.
 *
 * @param {unknown} anchor
 * @returns {[number, number] | null}
 */
function anchorStartSpot(anchor) {
  if (!isRecord(anchor) || !Array.isArray(anchor.position)) return null;
  if (anchor.phase !== undefined && anchor.phase !== "transition_start") return null;
  const [x, y] = anchor.position;
  return Number.isFinite(x) && Number.isFinite(y) ? [Number(x), Number(y)] : null;
}

/**
 * Return every body's tick-start spot this audience knows, in world units, by
 * presentation key.
 *
 * trajectories: the audience's own phase trajectories (Agent POV
 * visual_events.agent_phase_trajectories; rows {agent_presentation_key,
 * transition_start: {position}}); null or missing gives none. events: the same
 * audience's raw event rows; every anchor taken at tick start adds its body.
 * A trajectory wins over an event anchor for the same key. In Agent POV both
 * sources hold only bodies the team saw at tick start. Pure.
 *
 * @param {unknown} trajectories
 * @param {ReadonlyArray<unknown>} events
 * @returns {Map<string, [number, number]>}
 */
function startSpots(trajectories, events) {
  /** @type {Map<string, [number, number]>} */
  const spots = new Map();
  for (const event of events) {
    if (!isRecord(event)) continue;
    for (const field of ["source_anchor", "recipient_anchor", "agent_anchor"]) {
      const key = anchorKey(event[field]);
      const spot = anchorStartSpot(event[field]);
      if (key !== null && spot !== null) spots.set(key, spot);
    }
  }
  for (const row of asArray(trajectories)) {
    if (!isRecord(row)) continue;
    const key =
      typeof row.agent_presentation_key === "string"
        ? row.agent_presentation_key
        : anchorKey(row.transition_start);
    const spot = anchorStartSpot(row.transition_start);
    if (key !== null && spot !== null) spots.set(key, spot);
  }
  return spots;
}

/**
 * Say whether a scene body could give or take an aura at tick start: alive
 * then (alive at END and not brought back this tick, or fallen this tick) and
 * not under its Spawn Shield (none at END, or one that ended this tick).
 *
 * @param {JsonRecord} agent The body's END row in the audience's scene.
 * @param {BodyTickEvents | undefined} events Its own events this tick.
 * @returns {boolean}
 */
function activeAtStart(agent, events) {
  const alive = events?.died === true || (agent.alive === true && !events?.respawned);
  const shielded =
    events?.shieldExpired === true || finiteNumber(agent.spawn_shield_remaining) > 0;
  return alive && !shielded;
}

/**
 * Return whether two world spots are within the aura reach (2.0 units, with a
 * tiny tolerance), as the store's cover test does.
 *
 * @param {readonly [number, number]} a
 * @param {readonly [number, number]} b
 */
function withinReach(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]) <= RUNE_ZONES.wardRadiusUnits + 1e-9;
}

/**
 * @typedef {{
 *   events: ReadonlyArray<unknown>,
 *   agents: ReadonlyArray<unknown>,
 *   byBody: Map<string, BodyTickEvents>,
 *   spots: Map<string, [number, number]>,
 *   pov: boolean,
 * }} AuraTick
 *   One audience's facts for the aura moments of one tick: its raw events,
 *   its scene's agent rows (END), its events grouped by body (eventsByBody),
 *   its tick-start spots (startSpots) and whether it is Agent POV.
 */

/**
 * Find the Barrier Held moments of one tick (store barrierHeld).
 *
 * A champion is held when it took damage this tick, is alive at END, and the
 * damage would have reached its tick-start life (less its healing) without the
 * Guardian's Barrier of at least one other champion. Return a Map from the held
 * champion's presentation key to the presentation keys of those other
 * Warriors (the walls).
 *
 * Oracle View reads the host's exact rows: source_damage_output (the damage
 * before the recipient's barrier and the covering Warriors) and
 * recipient_health_resolution (tick-start life, damage, healing, life after).
 * A champion's own barrier stays applied. Agent POV gets no damage-source rows,
 * so it works the moment out as the store does: the walls are the scene's other
 * living, unshielded allied Warriors whose tick-start spot was within 2.0 units
 * of the champion's, and the damage without them is the damage taken divided by
 * 0.85 per wall. Agent POV health rows carry only the net life change, so the
 * damage taken is the net loss with no healing, the smallest damage the tick
 * can explain. Only champions the team saw count, and a heal can only hide a
 * moment, so Agent POV can miss a moment but never shows one that did not
 * happen. Pure.
 *
 * @param {AuraTick} tick
 * @returns {Map<string, string[]>}
 */
function barrierHeldMoments(tick) {
  /** @type {Map<string, JsonRecord>} */
  const agentByKey = new Map();
  for (const agent of tick.agents) {
    if (isRecord(agent) && typeof agent.presentation_key === "string") {
      agentByKey.set(agent.presentation_key, agent);
    }
  }
  /** @type {Map<string, {unbarred: number, walls: Set<string>}>} */
  const exact = new Map();
  if (!tick.pov) {
    for (const event of tick.events) {
      if (!isRecord(event) || event.event_kind !== "source_damage_output") continue;
      const recipient = anchorKey(event.recipient_anchor);
      if (recipient === null) continue;
      const covering = asArray(event.warrior_mitigation_aura_covering_emitters)
        .map(anchorKey)
        .filter((key) => key !== null);
      const entry = exact.get(recipient) ?? { unbarred: 0, walls: new Set() };
      const own = covering.includes(recipient) ? RUNE_ZONES.barrierMultiplier : 1;
      entry.unbarred += finiteNumber(event.source_modified_damage_output) * own;
      for (const key of covering) if (key !== recipient) entry.walls.add(key);
      exact.set(recipient, entry);
    }
  }
  /** @type {Map<string, string[]>} */
  const held = new Map();
  for (const event of tick.events) {
    if (!isRecord(event) || event.event_kind !== "recipient_health_resolution")
      continue;
    const key = anchorKey(event.recipient_anchor);
    const agent = key === null ? undefined : agentByKey.get(key);
    if (key === null || agent === undefined || agent.alive !== true) continue;
    const before = finiteNumber(event.transition_start_health, Number.NaN);
    // Agent POV rows carry only the net change, never the gross damage and
    // healing: the net loss with no healing is the smallest damage the tick
    // can explain, so a moment found that way really happened.
    const gross = Number.isFinite(event.total_effective_damage);
    const damage = gross
      ? finiteNumber(event.total_effective_damage)
      : Math.max(0, -finiteNumber(event.realized_net_health_change));
    const healing = gross ? finiteNumber(event.total_effective_healing) : 0;
    const after = finiteNumber(event.health_after_combat_resolution);
    if (!(damage > 0) || !(after > 0) || !Number.isFinite(before)) continue;
    /** @type {string[]} */
    let walls = [];
    let unbarred = damage;
    if (!tick.pov) {
      const entry = exact.get(key);
      if (entry === undefined) continue;
      walls = [...entry.walls];
      unbarred = entry.unbarred;
    } else {
      const spot = anchorStartSpot(event.recipient_anchor) ?? tick.spots.get(key);
      if (!spot) continue;
      for (const [wallKey, wall] of agentByKey) {
        const wallSpot = tick.spots.get(wallKey);
        if (
          wallKey === key ||
          wall.class_id !== 2 ||
          wall.team_id !== agent.team_id ||
          wallSpot === undefined ||
          !activeAtStart(wall, tick.byBody.get(wallKey)) ||
          !withinReach(spot, wallSpot)
        ) {
          continue;
        }
        walls.push(wallKey);
      }
      unbarred = damage / RUNE_ZONES.barrierMultiplier ** walls.length;
    }
    if (walls.length > 0 && unbarred - healing >= before) held.set(key, walls);
  }
  return held;
}

/**
 * Find the Aura Glints of one tick (store glintStud): for each aura emitter,
 * the first champion its aura helped this tick, with both tick-start spots.
 *
 * A Sorcerer's Empowerment (Mage) helps an ally inside it that deals damage
 * to an enemy; a Guardian's Barrier (Warrior) helps an ally inside it that
 * takes damage from an enemy. The emitter itself never counts. Oracle View
 * reads the covering emitters on the host's source_damage_output rows, in row
 * order. Agent POV derives the same from the team's own ability_activated rows
 * (source and recipient seen at tick start, on opposite teams), the scene's
 * teams and tick-start spots: the ally must be the emitter's teammate, alive
 * and unshielded at tick start and within 2.0 units of the emitter. Return a
 * Map from the emitter's presentation key to {from, to}: the emitter's and the
 * helped ally's tick-start spots in world units. Pure.
 *
 * @param {AuraTick} tick
 * @returns {Map<string, {from: [number, number], to: [number, number]}>}
 */
function auraGlintMoments(tick) {
  /** @type {Map<string, {from: [number, number], to: [number, number]}>} */
  const glints = new Map();
  if (!tick.pov) {
    for (const event of tick.events) {
      if (!isRecord(event) || event.event_kind !== "source_damage_output") continue;
      /** @type {Array<[unknown, unknown]>} */
      const pairs = [
        ...asArray(event.mage_damage_aura_covering_emitters).map(
          (emitter) =>
            /** @type {[unknown, unknown]} */ ([emitter, event.source_anchor]),
        ),
        ...asArray(event.warrior_mitigation_aura_covering_emitters).map(
          (emitter) =>
            /** @type {[unknown, unknown]} */ ([emitter, event.recipient_anchor]),
        ),
      ];
      for (const [emitter, helped] of pairs) {
        const key = anchorKey(emitter);
        const from = anchorStartSpot(emitter);
        const to = anchorStartSpot(helped);
        if (key === null || from === null || to === null) continue;
        if (key === anchorKey(helped) || glints.has(key)) continue;
        glints.set(key, { from, to });
      }
    }
    return glints;
  }
  /** @type {Map<string, JsonRecord>} */
  const agentByKey = new Map();
  for (const agent of tick.agents) {
    if (isRecord(agent) && typeof agent.presentation_key === "string") {
      agentByKey.set(agent.presentation_key, agent);
    }
  }
  const hits = tick.events.filter(
    (event) =>
      isRecord(event) &&
      event.event_kind === "ability_activated" &&
      anchorKey(event.source_anchor) !== null &&
      anchorKey(event.recipient_anchor) !== null,
  );
  for (const [emitterKey, emitter] of agentByKey) {
    const from = tick.spots.get(emitterKey);
    if (
      (emitter.class_id !== 1 && emitter.class_id !== 2) ||
      from === undefined ||
      !activeAtStart(emitter, tick.byBody.get(emitterKey))
    ) {
      continue;
    }
    for (const hit of /** @type {JsonRecord[]} */ (hits)) {
      const source = agentByKey.get(
        /** @type {string} */ (anchorKey(hit.source_anchor)),
      );
      const target = agentByKey.get(
        /** @type {string} */ (anchorKey(hit.recipient_anchor)),
      );
      if (!source || !target || source.team_id === target.team_id) continue;
      const helpedAnchor =
        emitter.class_id === 1 ? hit.source_anchor : hit.recipient_anchor;
      const helped = emitter.class_id === 1 ? source : target;
      const helpedKey = /** @type {string} */ (anchorKey(helpedAnchor));
      const to = anchorStartSpot(helpedAnchor) ?? tick.spots.get(helpedKey);
      if (
        helpedKey === emitterKey ||
        helped.team_id !== emitter.team_id ||
        !to ||
        !activeAtStart(helped, tick.byBody.get(helpedKey)) ||
        !withinReach(from, to)
      ) {
        continue;
      }
      glints.set(emitterKey, { from, to });
      break;
    }
  }
  return glints;
}

/**
 * Compare the START and END pieces and give each its beat (see
 * RuneDrawPiece). A piece equal at both ends is steady; one only at END comes
 * in at Settle (fade installMs); one only at START goes out at Settle (fade
 * from fadeOut(id)); a changed piece is drawn twice: END version "after",
 * START version "before". Pure.
 *
 * @param {ReadonlyArray<RunePiece>} start
 * @param {ReadonlyArray<RunePiece>} end
 * @param {number} installMs
 * @param {(id: string) => number} fadeOut
 * @returns {RuneDrawPiece[]}
 */
function beatPieces(start, end, installMs, fadeOut) {
  const startById = new Map(start.map((entry) => [entry.id, entry]));
  const endIds = new Set(end.map((entry) => entry.id));
  /** @type {RuneDrawPiece[]} */
  const out = [];
  for (const entry of end) {
    const earlier = startById.get(entry.id);
    const same =
      earlier !== undefined &&
      earlier.opacity === entry.opacity &&
      earlier.paint === entry.paint &&
      JSON.stringify(earlier.attributes) === JSON.stringify(entry.attributes);
    if (same) {
      out.push(
        Object.freeze({
          ...entry,
          side: "steady",
          fadeMs: 1,
          fromOpacity: entry.opacity,
        }),
      );
      continue;
    }
    out.push(
      Object.freeze({ ...entry, side: "after", fadeMs: installMs, fromOpacity: 0 }),
    );
    if (earlier !== undefined) {
      out.push(
        Object.freeze({
          ...earlier,
          id: `${earlier.id}.before`,
          side: "before",
          fadeMs: fadeOut(earlier.id),
          fromOpacity: earlier.opacity,
        }),
      );
    }
  }
  for (const entry of start) {
    if (!endIds.has(entry.id)) {
      out.push(
        Object.freeze({
          ...entry,
          side: "before",
          fadeMs: fadeOut(entry.id),
          fromOpacity: entry.opacity,
        }),
      );
    }
  }
  return out;
}

/**
 * @typedef {{
 *   statusMarks?: boolean,
 *   cooldownEffects?: boolean,
 *   statusApplication?: boolean,
 *   naturalStatusExpiry?: boolean,
 *   freezingTrapBreak?: boolean,
 *   statusClearOnDeath?: boolean,
 *   barrierHeld?: boolean,
 * }} RuneSwitches
 *   The Visual Filter switches this part reads, with their V5 defaults:
 *   status_marks, cooldown_effects, status_application,
 *   natural_status_expiry, freezing_trap_break and barrier_held are on unless
 *   false; status_clear_on_death is off unless true.
 */

/**
 * Rebuild one body's tick-start statuses from the current frame only.
 *
 * endStatuses are its END statuses (empty for a champion that fell this
 * tick); events are its own events this tick. An END status not applied or
 * refreshed this tick had one more tick at START. A status cleared by its
 * death (and not applied this tick) was there with its ticks unknown (null:
 * its counted pieces show as ghosts, see runeMarksFromStatuses). An aged-out
 * status had 1 tick; a broken Freezing Trap had its shell (0 lit crystals). A
 * status applied or refreshed this tick counts as new, so it is not in START.
 * Return a new array. Pure.
 *
 * @param {ReadonlyArray<MarkStatus>} endStatuses
 * @param {BodyTickEvents | undefined} events
 * @returns {MarkStatus[]}
 */
function startStatusesOf(endStatuses, events) {
  /** @type {MarkStatus[]} */
  const start = [];
  for (const status of endStatuses) {
    if (events?.applied.has(status.token)) continue;
    start.push({ ...status, ticks: status.ticks === null ? null : status.ticks + 1 });
  }
  for (const token of events?.cleared ?? []) {
    if (events?.applied.has(token) || events?.expired.has(token)) continue;
    if (events?.broken.has(token)) continue;
    start.push({ token, ticks: null, magnitude: null });
  }
  for (const token of events?.expired ?? []) {
    start.push({ token, ticks: 1, magnitude: null });
  }
  for (const token of events?.broken ?? []) {
    start.push({ token, ticks: 0, magnitude: null });
  }
  return start;
}

/**
 * Return the Ultimate cooldown a body had at tick start, from its END
 * cooldown (ticks, or null when unknown) and its own events: 0 when it spent
 * its Ultimate this tick, 1 when the Ultimate became ready this tick,
 * otherwise one tick more than END while charging (0 stays 0). Pure.
 *
 * @param {number | null} cooldown
 * @param {BodyTickEvents | undefined} events
 * @returns {number | null}
 */
function startCooldownOf(cooldown, events) {
  if (cooldown === null) return null;
  if (events?.cooldownStarted) return 0;
  if (events?.cooldownReady) return 1;
  return cooldown > 0 ? cooldown + 1 : 0;
}

/**
 * Turn a piece's paint to Ash: every `rune-fill--*` and `rune-stroke--*`
 * class except the Ink outline becomes `--ash` (store tint M.ash). Pure.
 *
 * @param {string} paint A space-separated list of `rune-*` classes.
 * @returns {string}
 */
function ashPaint(paint) {
  return paint
    .split(" ")
    .map((name) =>
      /^rune-(fill|stroke)--/u.test(name) && !name.endsWith("--ink")
        ? name.replace(/--[a-z-]+$/u, "--ash")
        : name,
    )
    .join(" ");
}

/**
 * Return the beat-timed pieces of a champion that fell this tick (store
 * perBody, falling case).
 *
 * startPieces are its tick-start marks and Rune Ring (bodyPieces). They form
 * one batch ("fall", see RuneDrawPiece) drawn "before" the champion's own Fall
 * beat (keyed by the champion by the caller): the batch shows from tick start
 * and goes out at the Fall, each piece at its own opacity. With Status Clear On
 * Death on (clearOn) the batch gives way over giveWayMs while an Ash copy (batch
 * "fall.ash", paint ashPaint, id suffix ".ash") flashes in to each piece's own
 * opacity and fades by ashFadeMs; off, the batch simply cuts at the Fall. Two
 * batches cost two beat overrides however many pieces they hold. A Fall the
 * audience did not see has no beat, so the choreographer shows END (no marks)
 * at once. Return a new array. Pure.
 *
 * @param {ReadonlyArray<RunePiece>} startPieces
 * @param {boolean} clearOn
 * @returns {RuneDrawPiece[]}
 */
function fallingPieces(startPieces, clearOn) {
  const marks = startPieces.map((entry) =>
    Object.freeze({
      ...entry,
      batch: "fall",
      side: /** @type {const} */ ("before"),
      beat: /** @type {const} */ ("fall"),
      fadeMs: clearOn ? RUNE_ZONES.giveWayMs : 1,
      fromOpacity: 1,
    }),
  );
  if (!clearOn) return marks;
  return [
    ...marks,
    ...startPieces.map((entry) =>
      Object.freeze({
        ...entry,
        id: `${entry.id}.ash`,
        paint: ashPaint(entry.paint),
        batch: "fall.ash",
        side: /** @type {const} */ ("after"),
        beat: /** @type {const} */ ("fall"),
        fadeMs: RUNE_ZONES.ashFadeMs,
        fromOpacity: 0,
        pulse: 1,
      }),
    ),
  ];
}

/**
 * Return the Barrier Held collar pieces for one body (store barrierLayer): a
 * bronze 1.25 px circle at 1.06R and twelve 1.5 px rivets, twice. The batch
 * "barrier" closes in at Impact over collarInMs and rests at collarRestOpacity
 * for the tick, so the paused frame shows a faint collar; the batch
 * "barrier.flash" (ids ending ".flash", full opacity) flashes once at Impact
 * over collarFlashMs in normal motion. Two beat overrides per collar. Pure.
 *
 * @param {RuneGeometry} g
 * @returns {RuneDrawPiece[]}
 */
function collarPieces(g) {
  /** @param {string} suffix @param {number} opacity */
  const parts = (suffix, opacity) => [
    piece(
      `barrier.collar${suffix}`,
      "circle",
      { r: f(g.collarR), "stroke-width": 1.25 },
      "rune-stroke--warrior",
      opacity,
    ),
    piece(
      `barrier.rivets${suffix}`,
      "path",
      { d: g.collarRivets, "stroke-width": 1.5 },
      "rune-stroke--warrior",
      opacity,
    ),
  ];
  return [
    ...parts("", RUNE_ZONES.collarRestOpacity).map((entry) =>
      Object.freeze({
        ...entry,
        batch: "barrier",
        side: /** @type {const} */ ("after"),
        beat: /** @type {const} */ ("impact"),
        fadeMs: RUNE_ZONES.collarInMs,
        fromOpacity: 0,
      }),
    ),
    ...parts(".flash", 1).map((entry) =>
      Object.freeze({
        ...entry,
        batch: "barrier.flash",
        side: /** @type {const} */ ("after"),
        beat: /** @type {const} */ ("impact"),
        fadeMs: RUNE_ZONES.collarFlashMs,
        fromOpacity: 0,
        pulse: 1,
      }),
    ),
  ];
}

/**
 * Return the Rune Zones records (records part "status-marks"): one per living
 * body in the audience's arena scene that has something to draw, plus one per
 * champion that fell this tick while this audience watched (its tick-start
 * marks wait for its Fall).
 *
 * input.scene: the authorized arena scene for this audience (Agent POV: the
 * POV scene, so only seen bodies get marks). input.transform projects
 * positions and radii (a missing radius is 0.5 world units). input.events:
 * this audience's own raw events of this tick (Oracle `latest_events`, Agent
 * POV `visual_events`; default none). input.trajectories: Agent POV only, the
 * same summary's agent_phase_trajectories (tick-start spots of seen bodies;
 * default none, which leaves Agent POV without Barrier Held).
 * input.startSeen(presentationKey): whether the body was in this audience's
 * view at tick start; default true in Oracle View and false in Agent POV, so a
 * body coming into sight shows its END state with no install, expiry, collar
 * or Fall from the dark. input.switches: RuneSwitches. Bodies with no
 * presentation key give no record, and so does a dead body that did not fall
 * this tick in this audience's view.
 *
 * Each record is {kind: "status-marks", presentationKey, center (CSS px, END
 * spot), radius (CSS px), classPaint, pieces (RuneDrawPiece[])}. START is
 * rebuilt from the current frame only (startStatusesOf, startCooldownOf). A
 * living body gets its install, count and expiry beats at Settle, the Thaw or
 * Shatter echo (the Shatter crack at Impact) and, when Barrier Held is on and
 * the moment happened (barrierHeldMoments), its collar. A champion that fell
 * gets its START pieces through fallingPieces (Ash or a cut at its Fall). Ash
 * (runes burned while held) needs history and is not drawn. Pure data: no DOM.
 *
 * @param {Record<string, any>} input
 * @returns {ReadonlyArray<LookRecord>}
 */
function statusMarkRecords(input) {
  const scene = input.scene;
  const transform = input.transform;
  if (!isRecord(scene) || !transform) return Object.freeze([]);
  const switches = /** @type {RuneSwitches} */ (input.switches ?? {});
  const on = (/** @type {boolean | undefined} */ value) => value !== false;
  const drawSwitches = {
    ring: on(switches.cooldownEffects),
    marks: on(switches.statusMarks),
  };
  const barrierOn = on(switches.barrierHeld);
  const clearOn = switches.statusClearOnDeath === true;
  if (!drawSwitches.ring && !drawSwitches.marks && !barrierOn) {
    return Object.freeze([]);
  }
  const povAudience = scene.audience === "agent_pov" || input.audience === "agent_pov";
  /** @type {(key: string) => boolean} */
  const startSeen =
    typeof input.startSeen === "function" ? input.startSeen : () => !povAudience;
  const rows = Array.isArray(input.events) ? input.events : [];
  const byBody = eventsByBody(rows);
  const held = barrierOn
    ? barrierHeldMoments({
        events: rows,
        agents: asArray(scene.agents),
        byBody,
        spots: startSpots(input.trajectories, rows),
        pov: povAudience,
      })
    : new Map();
  const installMs = on(switches.statusApplication) ? RUNE_ZONES.installFadeMs : 1;
  const expiryOn = on(switches.naturalStatusExpiry);
  const breakOn = on(switches.freezingTrapBreak);
  /** @type {LookRecord[]} */
  const out = [];
  for (const agent of asArray(scene.agents)) {
    if (!isRecord(agent)) continue;
    const key =
      typeof agent.presentation_key === "string" ? agent.presentation_key : null;
    if (key === null) continue;
    const events = byBody.get(key);
    const known = startSeen(key);
    const falling = agent.alive !== true && events?.died === true && known;
    if (agent.alive !== true && !falling) continue;
    const radius = transform.worldLengthToScreen(finiteNumber(agent.radius, 0.5));
    if (!(radius > 0)) continue;
    const g = runeGeometry(radius);
    const classPaint = CLASS_PAINT[/** @type {1} */ (agent.class_id)] ?? "neutral";
    /** @type {MarkStatus[]} */
    const endStatuses = [];
    for (const status of asArray(agent.statuses)) {
      const token = statusToken(status);
      if (token === null) continue;
      endStatuses.push({
        token,
        ticks: statusTicks(status),
        magnitude: Number.isFinite(status?.magnitude) ? Number(status.magnitude) : null,
      });
    }
    const cooldown = Number.isFinite(agent.ultimate_cooldown)
      ? Number(agent.ultimate_cooldown)
      : Number.isFinite(agent.ultimate_cooldown_remaining)
        ? Number(agent.ultimate_cooldown_remaining)
        : null;
    /** @type {RuneDrawPiece[]} */
    let pieces;
    if (falling) {
      // A champion cannot fall under its Spawn Shield, so START had none.
      const startZones = runeZones(
        runeMarksFromStatuses(startStatusesOf(endStatuses, events), false),
      );
      pieces = fallingPieces(
        bodyPieces(
          g,
          classPaint,
          runeRingLitArcs(startCooldownOf(cooldown, events)),
          startZones,
          drawSwitches,
        ),
        clearOn,
      );
    } else {
      const shield = finiteNumber(agent.spawn_shield_remaining) > 0;
      const endZones = runeZones(runeMarksFromStatuses(endStatuses, shield));
      const endPieces = bodyPieces(
        g,
        classPaint,
        runeRingLitArcs(cooldown),
        endZones,
        drawSwitches,
      );
      if (known) {
        const startZones = runeZones(
          runeMarksFromStatuses(startStatusesOf(endStatuses, events), shield),
        );
        const startPieces = bodyPieces(
          g,
          classPaint,
          runeRingLitArcs(startCooldownOf(cooldown, events)),
          startZones,
          drawSwitches,
        );
        const broken = Boolean(events?.broken.has("stun_hunter_trap"));
        const thawed = Boolean(events?.expired.has("stun_hunter_trap"));
        pieces = beatPieces(startPieces, endPieces, installMs, (id) => {
          if (id.startsWith("frozen.") && broken)
            return breakOn ? RUNE_ZONES.shatterFadeMs : 1;
          if (!expiryOn) return 1;
          if (id.startsWith("frozen.") && thawed) return RUNE_ZONES.thawFadeMs;
          if (id.startsWith("frozen.crystal")) return RUNE_ZONES.meltFadeMs;
          return RUNE_ZONES.expiryFadeMs;
        });
        // Thaw and Shatter echoes stay for this tick (they are part of END).
        if (drawSwitches.marks && thawed && expiryOn && endZones.frozen === null) {
          pieces.push(
            Object.freeze({
              ...piece(
                "thaw.ring",
                "circle",
                { r: f(g.thawR), "stroke-width": f(1.25 * g.u) },
                "rune-stroke--frostglass",
                0.28,
              ),
              side: /** @type {const} */ ("after"),
              fadeMs: installMs,
              fromOpacity: 0,
            }),
          );
        }
        if (drawSwitches.marks && broken && breakOn) {
          // The breaking blow cracks the ice at Impact; the shards fly at Settle.
          pieces.push(
            Object.freeze({
              ...piece(
                "shatter.crack",
                "path",
                { d: g.crack, "stroke-width": f(1.1 * g.u) },
                "rune-stroke--ice-crack",
                RUNE_ZONES.crackOpacity,
              ),
              side: /** @type {const} */ ("after"),
              beat: /** @type {const} */ ("impact"),
              fadeMs: RUNE_ZONES.crackFadeMs,
              fromOpacity: 0,
            }),
            Object.freeze({
              ...piece(
                "shatter.shards",
                "path",
                { d: g.shards, "stroke-width": 0.5 },
                "rune-fill--rime rune-stroke--ink",
                0.45,
              ),
              side: /** @type {const} */ ("after"),
              fadeMs: 120,
              fromOpacity: 0,
            }),
          );
        }
        if (held.has(key)) pieces.push(...collarPieces(g));
      } else {
        pieces = endPieces.map((entry) =>
          Object.freeze({
            ...entry,
            side: /** @type {const} */ ("steady"),
            fadeMs: 1,
            fromOpacity: entry.opacity,
          }),
        );
      }
    }
    if (pieces.length === 0) continue;
    out.push(
      Object.freeze({
        kind: "status-marks",
        presentationKey: key,
        center: screenPoint(agent.position, transform),
        radius,
        classPaint,
        pieces: Object.freeze(pieces),
      }),
    );
  }
  return Object.freeze(out);
}

/**
 * Build one body's marks group (record kind "status-marks").
 *
 * Return a detached `g.status-marks` translated to the body center, holding
 * one element per piece with class `rune-piece` plus its paint classes,
 * data-rune-piece (the piece id) and its END opacity. A non-steady piece gets
 * data-beat-hold (its beat: "settle" by default, "impact" or "fall"),
 * data-beat-side, data-beat-key (the presentation key) and data-beat-fade
 * (ms). A "before" piece is drawn at opacity 0 with data-beat-from (its START
 * opacity); an "after" piece with data-beat-from 0. A pulse piece is drawn at
 * opacity 0 with data-beat-pulse (its peak) and no data-beat-from: the
 * choreographer flashes it once in normal motion. Pieces with a batch go,
 * in order, into one `g.rune-batch` per batch (data-rune-batch names it) that
 * carries those beat attributes and the "before" or pulse opacity 0, while
 * each piece inside keeps its own resting opacity and no beat attributes.
 * These become fill-backwards overrides, so the DOM always holds END. The
 * group is aria-hidden and takes no pointer input (the champion's own card
 * lists every status). Export-safe SVG only.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement}
 */
function buildStatusMarks(doc, record) {
  const group = svgNode(doc, "g", {
    class: "status-marks",
    "data-marks-for": record.presentationKey,
    "data-class": record.classPaint,
    "data-radius": f(record.radius),
    transform: `translate(${f(record.center.x)} ${f(record.center.y)})`,
    "aria-hidden": "true",
  });
  /** @type {Map<string, SVGElement>} */
  const batches = new Map();
  /** @param {RuneDrawPiece} entry @returns {Record<string, unknown>} */
  const beatAttributes = (entry) => {
    const pulse = Number.isFinite(entry.pulse) ? Number(entry.pulse) : null;
    return {
      "data-beat-hold": entry.beat ?? "settle",
      "data-beat-side": entry.side,
      "data-beat-key": record.presentationKey,
      "data-beat-from": pulse === null ? entry.fromOpacity : null,
      "data-beat-fade": entry.fadeMs,
      "data-beat-pulse": pulse === null ? null : f(pulse),
    };
  };
  /** @param {RuneDrawPiece} entry */
  const hiddenAtEnd = (entry) =>
    entry.side === "before" || Number.isFinite(entry.pulse);
  for (const entry of /** @type {RuneDrawPiece[]} */ (record.pieces)) {
    if (typeof entry.batch === "string" && entry.side !== "steady") {
      let holder = batches.get(entry.batch);
      if (holder === undefined) {
        holder = svgNode(doc, "g", {
          class: "rune-batch",
          "data-rune-batch": entry.batch,
          opacity: hiddenAtEnd(entry) ? 0 : null,
          ...beatAttributes(entry),
        });
        batches.set(entry.batch, holder);
        group.append(holder);
      }
      holder.append(
        svgNode(doc, entry.tag, {
          ...entry.attributes,
          class: `rune-piece ${entry.paint}`,
          "data-rune-piece": entry.id,
          opacity: entry.opacity < 1 ? entry.opacity : null,
        }),
      );
      continue;
    }
    const node = svgNode(doc, entry.tag, {
      ...entry.attributes,
      class: `rune-piece ${entry.paint}`,
      "data-rune-piece": entry.id,
      opacity: hiddenAtEnd(entry) ? 0 : entry.opacity < 1 ? entry.opacity : null,
    });
    if (entry.side !== "steady") setNodeAttributes(node, beatAttributes(entry));
    group.append(node);
  }
  return group;
}

/**
 * Return the Ward Circle records for the aura fields the caller allows
 * (records part "aura-fields"; the switch aura_fields is off by default).
 *
 * input.fields: the scene's aura fields to draw (living sources only, already
 * filtered; Agent POV: only fields in the POV scene). input.agents (optional):
 * the same scene's agents, used to find a source's team (Team B rings are
 * double-lined) and, for the moments below, teams, life and shields.
 * input.transform projects; input.tooltipInput(field) gives plain tooltip
 * data.
 *
 * The moments (play only, drawn by fillWardMarks): input.events is this
 * audience's own raw events of this tick (default none), input.trajectories
 * its Agent POV tick-start spots (default none), input.audience "agent_pov"
 * picks the Agent POV rules (default Oracle View) and input.switches is
 * {auraGlints (default false), barrierHeld (default true)}. With Aura Glints
 * on, an emitter whose aura helped an ally this tick (auraGlintMoments) gets
 * glintStud, the stud (0-11, clockwise from 12 o'clock on screen) nearest the
 * direction from the emitter's tick-start spot to the ally's; else null. With
 * Barrier Held on, a Warrior whose barrier held an ally this tick
 * (barrierHeldMoments) gets barrierFlash true.
 *
 * Each record is {kind: "aura-field", center, radius (CSS px; the field's own
 * reach, 2.0 units), token, identity, classKey ("mage" or "warrior" from
 * source_class_id, else null), teamB, glintStud, barrierFlash,
 * tooltipInput}. Pure.
 *
 * @param {Record<string, any>} input
 * @returns {ReadonlyArray<LookRecord>}
 */
function wardCircleRecords(input) {
  if (!input.transform) return Object.freeze([]);
  const agents = asArray(input.agents);
  /** @type {Map<string, number>} */
  const teamByKey = new Map();
  for (const agent of agents) {
    if (isRecord(agent) && typeof agent.presentation_key === "string") {
      teamByKey.set(agent.presentation_key, Number(agent.team_id));
    }
  }
  const switches = isRecord(input.switches) ? input.switches : {};
  const glintOn = switches.auraGlints === true;
  const barrierOn = switches.barrierHeld !== false;
  const rows = Array.isArray(input.events) ? input.events : [];
  /** @type {AuraTick} */
  const tick = {
    events: rows,
    agents,
    byBody: eventsByBody(rows),
    spots: startSpots(input.trajectories, rows),
    pov: input.audience === "agent_pov",
  };
  const glints = glintOn && rows.length > 0 ? auraGlintMoments(tick) : new Map();
  const walls = new Set(
    barrierOn && rows.length > 0 ? [...barrierHeldMoments(tick).values()].flat() : [],
  );
  const tooltipInput = input.tooltipInput ?? ((/** @type {unknown} */ row) => row);
  /** @type {LookRecord[]} */
  const out = [];
  for (const row of asArray(input.fields)) {
    if (!isRecord(row)) continue;
    const radius = input.transform.worldLengthToScreen(finiteNumber(row.radius));
    if (!(radius > 0)) continue;
    const classKey =
      row.source_class_id === 1 ? "mage" : row.source_class_id === 2 ? "warrior" : null;
    const emitter = String(row.source_presentation_key);
    const glint = classKey === null ? undefined : glints.get(emitter);
    let glintStud = null;
    if (glint !== undefined) {
      const from = screenPoint(glint.from, input.transform);
      const to = screenPoint(glint.to, input.transform);
      const angle = (Math.atan2(to.x - from.x, -(to.y - from.y)) / DEG + 360) % 360;
      glintStud = Math.round(angle / 30) % 12;
    }
    out.push(
      Object.freeze({
        kind: "aura-field",
        center: screenPoint(row.center, input.transform),
        radius,
        token: typeof row.token_id === "string" ? row.token_id : null,
        identity: Object.freeze({
          presentation_key: row.presentation_key,
          global_slot: row.global_slot,
        }),
        classKey,
        teamB: teamByKey.get(emitter) === 2,
        glintStud,
        barrierFlash: classKey === "warrior" && walls.has(emitter),
        tooltipInput: tooltipInput(row),
      }),
    );
  }
  return Object.freeze(out);
}

/**
 * Return the stud path of a Ward Circle of radius ra (CSS px): twelve radial
 * ticks 5.5 px long for the Warrior's Guardian's Barrier, twelve small
 * four-point stars for the Mage's Sorcerer's Empowerment. Pure.
 *
 * @param {number} ra
 * @param {string | null} classKey
 * @returns {string}
 */
function wardStuds(ra, classKey) {
  const angles = Array.from({ length: 12 }, (_, i) => i * 30);
  return classKey === "warrior"
    ? angles.map((t) => `M${pt(pol(ra, t))} L${pt(pol(ra - 5.5, t))}`).join(" ")
    : angles.map((t) => star4(pol(ra, t), 2.6, 0.8)).join(" ");
}

/**
 * Build one Ward Circle: the focusable `circle.aura-field` ring (its tooltip
 * owner) and a `g.ward-circle__marks` group with the Team B inner line, the
 * studs and the play-only Aura Glint and Barrier Held flashes (see
 * fillWardMarks). A record whose tooltip is null is left out. Return [circle,
 * marks] or []. Registers the tooltip owner (DOM side effect through
 * tooltip.js).
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement[]}
 */
function buildWardCircle(doc, record) {
  const descriptor = /** @type {{title: string} | null} */ (tooltip(record));
  if (descriptor === null) return [];
  const circle = svgNode(doc, "circle", {
    class: "aura-field ward-circle",
    cx: record.center.x,
    cy: record.center.y,
    r: record.radius,
    role: "img",
    tabindex: "0",
    "aria-label": descriptor.title,
    "data-token": record.token,
    "data-class": record.classKey,
  });
  setDisplayIdentityData(circle, record.identity);
  registerTooltipOwner(circle, descriptor);
  const marks = svgNode(doc, "g", {
    class: "ward-circle__marks",
    "data-class": record.classKey,
    "aria-hidden": "true",
  });
  fillWardMarks(doc, marks, record);
  return [circle, marks];
}

/**
 * Replace the children of a Ward Circle marks group for record (its center,
 * radius, class, team and moments). Children: the Team B inner line, the
 * studs and, play only, the moments. An Aura Glint (record.glintStud) is a
 * Forge White 2 px rivet across the Warrior's stud or a larger Starcore star
 * on the Mage's; a Barrier Held flash (record.barrierFlash) lights all twelve
 * Warrior rivets in Forge White. Both rest at opacity 0 and carry
 * data-beat-hold "impact" and data-beat-pulse 1 over glintMs, so the
 * choreographer flashes them once at Impact in normal motion and the paused
 * frame never shows them. Mutates marks only.
 *
 * @param {Document} doc
 * @param {SVGElement} marks
 * @param {LookRecord} record
 */
function fillWardMarks(doc, marks, record) {
  marks.setAttribute(
    "transform",
    `translate(${f(record.center.x)} ${f(record.center.y)})`,
  );
  const ra = record.radius;
  /** @type {SVGElement[]} */
  const children = [];
  if (record.teamB && ra > 3) {
    children.push(
      svgNode(doc, "circle", { class: "ward-circle__inner", r: f(ra - 3) }),
    );
  }
  children.push(
    svgNode(doc, "path", {
      class: `ward-circle__studs ward-circle__studs--${record.classKey ?? "neutral"}`,
      d: wardStuds(ra, record.classKey),
    }),
  );
  const flash = {
    opacity: 0,
    "data-beat-hold": "impact",
    "data-beat-side": "after",
    "data-beat-pulse": 1,
    "data-beat-fade": RUNE_ZONES.glintMs,
  };
  if (record.barrierFlash === true) {
    children.push(
      svgNode(doc, "path", {
        class: "ward-circle__barrier-flash",
        d: Array.from({ length: 12 }, (_, i) => i * 30)
          .map((t) => `M${pt(pol(ra + 0.5, t))} L${pt(pol(ra - 6, t))}`)
          .join(" "),
        ...flash,
      }),
    );
  }
  if (Number.isInteger(record.glintStud)) {
    const t = Number(record.glintStud) * 30;
    children.push(
      svgNode(doc, "path", {
        class: `ward-circle__glint ward-circle__glint--${record.classKey}`,
        d:
          record.classKey === "warrior"
            ? `M${pt(pol(ra + 1, t))} L${pt(pol(ra - 6.5, t))}`
            : star4(pol(ra, t), 3.6, 1.1),
        "data-stud": record.glintStud,
        ...flash,
      }),
    );
  }
  marks.replaceChildren(...children);
}

/**
 * Return the status tray of one champion for the HUD and cards: at most limit
 * chips (default 2, plan P15) and how many more statuses there are ("+N").
 *
 * statuses: the champion's status rows from the audience's own scene (Agent
 * POV: only a seen champion's). In Combat, Spawn Shield and unknown tokens are
 * left out. Chips are ordered most limiting first, as the Champion Card orders
 * its rows (Freezing Trap (Stun), the Charge and Crippling Poison stuns, the
 * slows, Crippling Poison (Healing Reduction), Blessing of Freedom (Slow
 * Resistance), Burst (Damage Boost)), then by more ticks left. Each chip is
 * {tokenId, sigilKey (icons.js), ticks (null when unknown)}. Return frozen
 * {chips, hiddenCount}. Pure; names come from symbols.js at the caller.
 *
 * @param {ReadonlyArray<unknown>} statuses
 * @param {number} [limit]
 * @returns {Readonly<{chips: ReadonlyArray<{tokenId: string, sigilKey: string | null, ticks: number | null}>, hiddenCount: number}>}
 */
export function statusTray(statuses, limit = 2) {
  const rows = asArray(statuses)
    .map((status) => ({ tokenId: statusToken(status), ticks: statusTicks(status) }))
    .filter(
      (row) =>
        row.tokenId !== null &&
        TRAY_ORDER.includes(/** @type {string} */ (row.tokenId)),
    )
    .sort(
      (a, b) =>
        TRAY_ORDER.indexOf(/** @type {string} */ (a.tokenId)) -
          TRAY_ORDER.indexOf(/** @type {string} */ (b.tokenId)) ||
        (b.ticks ?? 0) - (a.ticks ?? 0),
    );
  const shown = rows.slice(0, Math.max(0, limit));
  return Object.freeze({
    chips: Object.freeze(
      shown.map((row) =>
        Object.freeze({
          tokenId: /** @type {string} */ (row.tokenId),
          sigilKey: statusSigilKey(row.tokenId),
          ticks: row.ticks,
        }),
      ),
    ),
    hiddenCount: rows.length - shown.length,
  });
}

/** Semantic pulses owned by the Requiems look; every other pulse is drawn here. */
const REQUIEM_PULSES = new Set([
  "agent_died",
  "agent_respawned",
  "respawn_wave_occurred",
  "death_announcement",
]);

/**
 * Return this look's drawing records for one part of the arena or one event.
 *
 * input.part picks what to build:
 * - "status-marks": the Rune Zones and Rune Ring of every living body in
 *   input.scene, a falling champion's tick-start marks and the Barrier Held
 *   collar; see statusMarkRecords for input.transform, input.events,
 *   input.trajectories, input.startSeen and input.switches. Records are
 *   {kind: "status-marks", presentationKey, center, radius, classPaint,
 *   pieces}.
 * - "aura-fields": the Ward Circles; see wardCircleRecords (input.fields,
 *   input.agents, input.transform, input.tooltipInput, and for the moments
 *   input.events, input.trajectories, input.audience and input.switches).
 *   Records are {kind: "aura-field", center, radius, token, identity,
 *   classKey, teamB, glintStud, barrierFlash, tooltipInput} in CSS pixels.
 * - "fact-docks": input.docks lists {placement, dockKind ("status" or
 *   "modifier"), dimensions} from the dock allocator; input.ownerByLayoutSlot
 *   maps a layout slot to its arena agent row; input.audience is "researcher"
 *   or "agent_pov"; input.tooltipInputs.cell(item, owner, dockKind) and
 *   input.tooltipInputs.overflow(items, owner, dockKind) give tooltip data.
 *   Records are {kind: "fact-dock", dockKind, placement, dimensions,
 *   ownerAgent, audience, cellTooltipInputs, overflowTooltipInput}.
 * - "cooldown-docks": input.docks lists allocator placements and
 *   input.ownerByLayoutSlot their owners; records are {kind: "cooldown-dock",
 *   placement, ownerAgent}.
 * Without part, input.event is one plan event: a "semantic_pulse" event the
 * Requiems look does not own gives {kind: "semantic-pulse", event}; a
 * "status_lifecycle" event for an ended Spawn Shield (cueSemantic
 * "spawn_shield_expired", with a projected recipient) gives {kind:
 * "ward-lift", event}. Anything else gives [], including every other
 * "status_lifecycle" event: V5 Rune Zones shows a status change on the body
 * ("status-marks"), never as an off-body badge, so the painter skips that
 * event. Pure data: no DOM (the caller's callbacks must be pure).
 *
 * @param {Record<string, any>} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  if (input.part === "aura-fields") {
    return wardCircleRecords(input);
  }
  if (input.part === "status-marks") {
    return statusMarkRecords(input);
  }
  if (input.part === "fact-docks") {
    /** @type {ReadonlyMap<number, JsonRecord>} */
    const owners = input.ownerByLayoutSlot ?? new Map();
    return Object.freeze(
      (input.docks ?? []).map(
        (
          /** @type {{placement: any, dockKind: "status" | "modifier", dimensions: DockDimensions}} */ dock,
        ) => {
          const ownerAgent = owners.get(dock.placement.globalSlot) ?? {};
          return Object.freeze({
            kind: "fact-dock",
            dockKind: dock.dockKind,
            placement: dock.placement,
            dimensions: dock.dimensions,
            ownerAgent,
            audience: input.audience,
            cellTooltipInputs: Object.freeze(
              dock.placement.visibleStatuses.map((/** @type {unknown} */ rawItem) =>
                input.tooltipInputs.cell(
                  isRecord(rawItem) ? rawItem : {},
                  ownerAgent,
                  dock.dockKind,
                ),
              ),
            ),
            overflowTooltipInput:
              dock.placement.hiddenCount > 0
                ? input.tooltipInputs.overflow(
                    dock.placement.hiddenStatuses,
                    ownerAgent,
                    dock.dockKind,
                  )
                : null,
          });
        },
      ),
    );
  }
  if (input.part === "cooldown-docks") {
    /** @type {ReadonlyMap<number, JsonRecord>} */
    const owners = input.ownerByLayoutSlot ?? new Map();
    return Object.freeze(
      (input.docks ?? []).map((/** @type {any} */ placement) =>
        Object.freeze({
          kind: "cooldown-dock",
          placement,
          ownerAgent: owners.get(placement.globalSlot) ?? {},
        }),
      ),
    );
  }
  const event = input.event;
  // A status change has no off-body badge: V5 Rune Zones shows it on the
  // body itself (status-marks install or go out at the Settle beat).
  if (event?.kind === "semantic_pulse" && !REQUIEM_PULSES.has(event.cueSemantic)) {
    return Object.freeze([Object.freeze({ kind: "semantic-pulse", event })]);
  }
  // The one status change drawn off the marks: the Ward Lift when a Spawn
  // Shield ends (the plan admits the row only while spawn_shield_expiry is on).
  if (
    event?.kind === "status_lifecycle" &&
    event.cueSemantic === "spawn_shield_expired" &&
    isRecord(event.recipient)
  ) {
    return Object.freeze([Object.freeze({ kind: "ward-lift", event })]);
  }
  return Object.freeze([]);
}

/**
 * Build nodes for this look's records.
 *
 * records come from records(); doc creates nodes. Aura records become two
 * nodes each: a focusable `circle.aura-field.ward-circle` with its tooltip and
 * a `g.ward-circle__marks` (Team B inner line and studs); a record whose
 * tooltip is null is left out. Status-mark records become one
 * `g.status-marks` group each (see buildStatusMarks; mount it in the body's
 * glide wrapper so it rides the body glide). Fact-dock and cooldown-dock records become one dock group
 * each, with cells, leaders and tooltips. Semantic-pulse records draw into
 * mount (the painter's EventMount; without it, new detached settled groups). Return the
 * new top-level nodes in record order (for event records: the groups drawn
 * into). A ward-lift record draws its ring into mount the same way (see
 * buildWardLift). Registers tooltip owners and pointerdown listeners;
 * export-safe SVG.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @param {EventMount} [mount]
 * @returns {SVGElement[]}
 */
export function build(records, doc, mount) {
  /** @type {SVGElement[]} */
  const nodes = [];
  /** @type {EventMount | null} */
  let eventMount = null;
  for (const record of records) {
    if (record.kind === "aura-field") {
      nodes.push(...buildWardCircle(doc, record));
    } else if (record.kind === "status-marks") {
      nodes.push(buildStatusMarks(doc, record));
    } else if (record.kind === "fact-dock") {
      nodes.push(buildFactDock(doc, record));
    } else if (record.kind === "cooldown-dock") {
      nodes.push(buildCooldownDock(doc, record));
    } else if (record.kind === "semantic-pulse") {
      eventMount ??= mount ?? detachedEventMount(doc, record.event);
      buildPulse(doc, eventMount.group, eventMount.connector, record.event);
    } else if (record.kind === "ward-lift") {
      eventMount ??= mount ?? detachedEventMount(doc, record.event);
      buildWardLift(doc, eventMount, record.event);
    }
  }
  return eventMount === null ? nodes : [...nodes, ...mountNodes(eventMount)];
}

/**
 * Draw the Ward Lift (store death-respawn-and-kills wardLift) into an event
 * mount: one dashed Ward Silver ring, 1 px, at 0.95R (R = event.anchorRadius,
 * the body radius in CSS px), 12 dashes, at 60% opacity, centered on the
 * event's projected recipient (`circle.ward-lift.spawn-shield-expiry__ring`).
 * The painter fades the event group in and out over the Settle window, so the
 * ring lifts off as the shield ends. The paused frame does not show it: with a
 * settled mount (mount.options.settled), or without a recipient or a positive
 * radius, nothing is drawn. Return undefined; appends to mount.group only.
 * Export-safe SVG (a flat dashed stroke).
 *
 * @param {Document} doc
 * @param {EventMount} mount
 * @param {JsonRecord} event
 */
function buildWardLift(doc, mount, event) {
  const radius = finiteNumber(event.anchorRadius) * RUNE_ZONES.wardLiftRadius;
  if (mount.options?.settled === true || !(radius > 0) || !isRecord(event.recipient)) {
    return;
  }
  const dash = f((2 * Math.PI * radius) / RUNE_ZONES.wardLiftPieces);
  mount.group.append(
    effectNode(doc, "circle", {
      class: "ward-lift spawn-shield-expiry__ring",
      cx: f(finiteNumber(event.recipient.x)),
      cy: f(finiteNumber(event.recipient.y)),
      r: f(radius),
      "stroke-width": 1,
      "stroke-dasharray": `${dash} ${dash}`,
      opacity: RUNE_ZONES.wardLiftOpacity,
      "aria-hidden": "true",
    }),
  );
}

/**
 * Create one status or modifier dock and its overflow cell.
 *
 * record is a fact-dock record: its placement gives ordered visible and hidden
 * items and pixel geometry, dockKind is "status" or "modifier", dimensions the
 * cell width, height and gap in CSS pixels, ownerAgent the arena row and audience
 * the explanation path. Return a detached dock group with exact value
 * metadata, icons and labels. Unknown values show "?" rather than a number.
 * Hidden items stay reachable through the overflow tooltip; a layout slot never
 * becomes a public identity.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement}
 */
function buildFactDock(doc, record) {
  const placement = record.placement;
  const kind = record.dockKind;
  const dimensions = record.dimensions;
  const audience = record.audience;
  const ownerAgent = record.ownerAgent;
  const compact = "compactFallback" in placement && placement.compactFallback === true;
  const compactAttributes = compact
    ? { "data-compact-fallback": true, "data-kind": kind, tabindex: "0" }
    : {};
  const group = svgNode(doc, "g", {
    class: `${kind}-dock`,
    "data-zone": `${kind}-dock`,
    ...displayIdentityAttributes(ownerAgent),
    "data-anchor": placement.anchor,
    "data-expanded": placement.expanded,
    "data-collision-free": placement.collisionFree,
    "data-visible-count": placement.visibleCount,
    "data-hidden-count": placement.hiddenCount,
  });
  if (
    placement.hiddenCount > 0 ||
    placement.anchor !== "north" ||
    placement.tangentShift !== 0
  ) {
    group.append(
      svgNode(doc, "line", {
        class: `dock-leader ${kind}-dock__leader`,
        x1: placement.leader.start.x,
        y1: placement.leader.start.y,
        x2: placement.leader.end.x,
        y2: placement.leader.end.y,
        "aria-hidden": "true",
      }),
    );
  }

  for (const [index, rawItem] of placement.visibleStatuses.entries()) {
    const item = isRecord(rawItem) ? rawItem : {};
    const token = resolveVisualToken(
      kind,
      item.token_id,
      audience === "agent_pov" && kind === "status" ? undefined : item,
    );
    const column = index % placement.columns;
    const row = Math.floor(index / placement.columns);
    const x =
      placement.bounds.left + column * (dimensions.cellWidth + dimensions.cellGap);
    const y = placement.bounds.top + row * (dimensions.cellHeight + dimensions.cellGap);
    const value =
      kind === "status"
        ? String(
            Number.isInteger(item.duration) && item.duration > 0 ? item.duration : "?",
          )
        : Number.isFinite(item.multiplier)
          ? `×${formatDisplayNumber(item.multiplier)}`
          : "×?";
    const accessibleValue =
      kind === "status"
        ? `duration ${value}`
        : Number.isFinite(item.multiplier)
          ? `multiplier ${formatDisplayNumber(item.multiplier)}`
          : "multiplier unknown";
    const supportedStatusDuration =
      kind === "status" &&
      Number.isInteger(item.duration) &&
      item.duration >= 1 &&
      item.duration <= 5;
    const cell = svgNode(doc, "g", {
      class: `${kind}-cell${compact ? " required-dock-fallback" : ""}`,
      ...compactAttributes,
      role: "img",
      "aria-label": `${token.accessibleName}, ${accessibleValue}, ${agentIdentity(ownerAgent)}`,
      "data-zone": `${kind}-cell`,
      ...displayIdentityAttributes(ownerAgent),
      "data-token": token.cssKey,
      "data-token-id": token.tokenId,
      "data-index": index,
      "data-numeric-layout": supportedStatusDuration ? "compartments" : "measured",
      "data-supported-duration": kind === "status" ? supportedStatusDuration : null,
      "data-duration":
        kind === "status" && Number.isInteger(item.duration) ? item.duration : null,
      "data-multiplier":
        kind === "modifier" && Number.isFinite(item.multiplier)
          ? item.multiplier
          : null,
    });
    if (kind === "status") {
      cell.dataset.sourceClass = classTokenFromId(item.source_class_id).cssKey;
    }
    const box = svgNode(doc, "rect", {
      class: `${kind}-cell__box`,
      x,
      y,
      width: dimensions.cellWidth,
      height: dimensions.cellHeight,
      rx: 5,
    });
    const icon = createSvgIcon(doc, token.glyphKey, {
      className: `${kind}-cell__icon`,
    });
    const iconSize = kind === "modifier" ? 9 : kind === "status" ? 12 : 10;
    setNodeAttributes(icon, {
      x: x + (kind === "modifier" ? 3 : 2),
      y: y + (dimensions.cellHeight - iconSize) / 2,
      width: iconSize,
      height: iconSize,
    });
    const text = svgNode(doc, "text", {
      class: `${kind}-cell__value`,
      x: x + dimensions.cellWidth - 3,
      y: y + dimensions.cellHeight / 2,
    });
    text.textContent = value;
    const compartments =
      kind === "status"
        ? [
            svgNode(doc, "rect", {
              class: "status-cell__icon-compartment",
              x: x + 2,
              y: y + 2,
              width: 12,
              height: dimensions.cellHeight - 4,
              "aria-hidden": "true",
            }),
            svgNode(doc, "rect", {
              class: "status-cell__value-compartment",
              x: x + 16,
              y: y + 2,
              width: 9,
              height: dimensions.cellHeight - 4,
              "aria-hidden": "true",
            }),
          ]
        : [];
    registerTooltipOwner(
      cell,
      tooltip(
        Object.freeze({
          kind: "dock-cell",
          tooltipInput: record.cellTooltipInputs[index],
        }),
      ),
    );
    cell.append(box, ...compartments, icon, text);
    group.append(cell);
  }

  if (placement.hiddenCount > 0) {
    const index = placement.visibleCount;
    const column = index % placement.columns;
    const row = Math.floor(index / placement.columns);
    const x =
      placement.bounds.left + column * (dimensions.cellWidth + dimensions.cellGap);
    const y = placement.bounds.top + row * (dimensions.cellHeight + dimensions.cellGap);
    const hiddenLabels = placement.hiddenStatuses.map(
      (/** @type {unknown} */ rawItem) => {
        const item = isRecord(rawItem) ? rawItem : {};
        const token = resolveVisualToken(
          kind,
          item.token_id,
          audience === "agent_pov" && kind === "status" ? undefined : item,
        );
        return kind === "status"
          ? `${token.accessibleName}, duration ${item.duration ?? "unknown"}`
          : `${token.accessibleName}, multiplier ${formatDisplayNumber(item.multiplier)}`;
      },
    );
    const overflow = svgNode(doc, "g", {
      class: `${kind}-overflow${compact ? " required-dock-fallback" : ""}`,
      ...compactAttributes,
      role: "img",
      "aria-label": `${placement.overflowLabel} hidden ${kind} cues for ${agentIdentity(ownerAgent)}: ${hiddenLabels.join("; ")}`,
      "data-zone": `${kind}-overflow`,
      ...displayIdentityAttributes(ownerAgent),
      "data-hidden-count": placement.hiddenCount,
      "data-owner-label": agentIdentity(ownerAgent),
    });
    registerTooltipOwner(
      overflow,
      tooltip(
        Object.freeze({
          kind: "dock-overflow",
          tooltipInput: record.overflowTooltipInput,
        }),
      ),
    );
    const overflowLabel = svgNode(doc, "text", {
      class: `${kind}-overflow__label`,
      x: x + dimensions.cellWidth / 2,
      y: y + dimensions.cellHeight / 2,
    });
    overflowLabel.textContent = placement.overflowLabel;
    overflow.append(
      svgNode(doc, "rect", {
        class: `${kind}-cell__box`,
        x,
        y,
        width: dimensions.cellWidth,
        height: dimensions.cellHeight,
        rx: 5,
      }),
      overflowLabel,
    );
    group.append(overflow);
  }
  return group;
}

/**
 * Create one class-specific Ultimate cooldown dock from its placement.
 *
 * record is a cooldown-dock record: its placement holds one visible class and
 * tick item, and ownerAgent the arena row. Positive whole ticks show exactly
 * (fitDockValues may shorten them later); invalid ticks show "?". A missing
 * explanation gives an aria-hidden empty group. Otherwise return a detached
 * group with icon and value compartments and a tooltip, plus compact fallback
 * metadata when the allocator used the fallback.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement}
 */
function buildCooldownDock(doc, record) {
  const placement = record.placement;
  const dimensions = FOOTPRINT.cooldownDock;
  const fallbackPlacement = placement.compactFallback === true;
  const rawItem = placement.visibleStatuses[0];
  const item = isRecord(rawItem) ? rawItem : {};
  const ticks =
    Number.isInteger(item.ticks) && item.ticks > 0 ? Number(item.ticks) : "?";
  const classToken = classTokenFromId(item.classId);
  const token = ultimateTokenFromClassId(item.classId);
  const ownerAgent = record.ownerAgent;
  const group = svgNode(doc, "g", {
    class: "cooldown-dock",
    "data-zone": "cooldown-dock",
    ...displayIdentityAttributes(ownerAgent),
    "data-class": classToken.cssKey,
    "data-anchor": placement.anchor,
    "data-expanded": fallbackPlacement ? true : placement.expanded,
    "data-collision-free": placement.collisionFree,
    "data-visible-count": 1,
    "data-hidden-count": 0,
  });
  const explanation = tooltip(
    Object.freeze({ kind: "cooldown-cell", ownerAgent, ticks }),
  );
  if (explanation === null) {
    group.setAttribute("aria-hidden", "true");
    return group;
  }
  if (
    fallbackPlacement ||
    placement.anchor !== "north" ||
    placement.tangentShift !== 0
  ) {
    group.append(
      svgNode(doc, "line", {
        class: "dock-leader cooldown-dock__leader",
        x1: placement.leader.start.x,
        y1: placement.leader.start.y,
        x2: placement.leader.end.x,
        y2: placement.leader.end.y,
        "aria-hidden": "true",
      }),
    );
  }

  const x = placement.bounds.left;
  const y = placement.bounds.top;
  const accessibleTicks =
    typeof ticks === "number"
      ? `${ticks} ${ticks === 1 ? "tick" : "ticks"} remaining`
      : "remaining ticks unknown";
  const cell = svgNode(doc, "g", {
    class: "cooldown-cell",
    role: "img",
    tabindex: fallbackPlacement ? "0" : null,
    "aria-label": `${token.accessibleName} cooldown, ${accessibleTicks}, ${agentIdentity(ownerAgent)}`,
    "data-zone": "ultimate-cooldown",
    ...displayIdentityAttributes(ownerAgent),
    "data-layout-key": fallbackPlacement ? placement.layoutKey : null,
    "data-kind": fallbackPlacement ? "cooldown" : null,
    "data-compact-fallback": fallbackPlacement ? "true" : null,
    "data-owner-label": fallbackPlacement ? agentIdentity(ownerAgent) : null,
    "data-class": classToken.cssKey,
    "data-token": token.cssKey,
    "data-token-id": token.tokenId,
    "data-ticks": ticks,
    "data-numeric-layout": "compartments",
  });
  const box = svgNode(doc, "rect", {
    class: "cooldown-cell__box",
    x,
    y,
    width: dimensions.cellWidth,
    height: dimensions.cellHeight,
    rx: 5,
  });
  const iconCompartment = svgNode(doc, "rect", {
    class: "cooldown-cell__icon-compartment",
    x: x + 2,
    y: y + 2,
    width: 14,
    height: dimensions.cellHeight - 4,
    "aria-hidden": "true",
  });
  const valueCompartment = svgNode(doc, "rect", {
    class: "cooldown-cell__value-compartment",
    x: x + 19,
    y: y + 2,
    width: 16,
    height: dimensions.cellHeight - 4,
    "aria-hidden": "true",
  });
  const icon = createSvgIcon(doc, token.glyphKey, {
    className: "cooldown-cell__icon",
  });
  setNodeAttributes(icon, {
    x: x + 3,
    y: y + 3,
    width: 12,
    height: 12,
  });
  const value = svgNode(doc, "text", {
    class: "cooldown-cell__value",
    x: x + dimensions.cellWidth - 3,
    y: y + dimensions.cellHeight / 2,
  });
  value.textContent = String(ticks);
  registerTooltipOwner(cell, explanation);
  cell.append(box, iconCompartment, valueCompartment, icon, value);
  group.append(cell);
  return group;
}

/**
 * Keep dock values readable using the browser's measured text and icon sizes.
 *
 * layer is the `durable-status-modifier` layer. For each status, cooldown and
 * modifier cell: a status duration of 1 to 5 keeps fixed compartments;
 * otherwise a colliding icon is hidden and, if still needed, the shown number
 * is shortened. Exact values stay in data attributes, labels and tooltips. Reads
 * browser layout (getBBox, getBoundingClientRect) and changes the cells; cells
 * missing a part are skipped. Return undefined; measurement errors propagate.
 * `scene.js` runs it after building docks and again when fonts are ready.
 *
 * @param {Element} layer
 */
export function fitDockValues(layer) {
  for (const cell of layer.querySelectorAll(
    ".status-cell, .cooldown-cell, .modifier-cell",
  )) {
    const kind = cell.classList.contains("status-cell")
      ? "status"
      : cell.classList.contains("cooldown-cell")
        ? "cooldown"
        : "modifier";
    const box = cell.querySelector(`.${kind}-cell__box`);
    const icon = cell.querySelector(`.${kind}-cell__icon`);
    const value = cell.querySelector(`.${kind}-cell__value`);
    if (
      !(box instanceof SVGGraphicsElement) ||
      !(icon instanceof SVGGraphicsElement) ||
      !(value instanceof SVGGraphicsElement)
    ) {
      continue;
    }
    const supportedStatusDuration =
      kind === "status" && /^[1-5]$/u.test(value.textContent ?? "");
    if (supportedStatusDuration) {
      cell.setAttribute("data-numeric-layout", "compartments");
      cell.removeAttribute("data-icon-suppressed");
      cell.removeAttribute("data-numeric-fallback");
      icon.removeAttribute("hidden");
      continue;
    }
    const boxBounds = box.getBBox();
    const iconScreenBounds = icon.getBoundingClientRect();
    const valueScreenBounds = value.getBoundingClientRect();
    let valueBounds = value.getBBox();
    if (iconScreenBounds.right + 2 > valueScreenBounds.left) {
      cell.setAttribute("data-icon-suppressed", "true");
      cell.setAttribute("data-numeric-fallback", "true");
      cell.setAttribute("data-numeric-layout", "measured-fallback");
      icon.setAttribute("hidden", "");
      value.setAttribute("x", String(boxBounds.x + boxBounds.width / 2));
      valueBounds = value.getBBox();
    }
    const availableWidth = Math.max(boxBounds.width - 6, 1);
    if (valueBounds.width > availableWidth) {
      const exactValue =
        kind === "status"
          ? Number(cell.getAttribute("data-duration"))
          : kind === "cooldown"
            ? Number(cell.getAttribute("data-ticks"))
            : Number(cell.getAttribute("data-multiplier"));
      const prefix = kind === "modifier" ? "×" : "";
      value.textContent = `${prefix}${formatCompactDisplayNumber(exactValue)}`;
      value.removeAttribute("textLength");
      value.removeAttribute("lengthAdjust");
      cell.setAttribute("data-numeric-layout", "compact-measured-fallback");
      cell.setAttribute("data-visible-value-abbreviated", "true");
    }
  }
}

/**
 * Draw a plain pulse (for example Ultimate Ready) at its cue or anchor.
 *
 * doc creates nodes in group and connector; event gives the anchor, meaning,
 * optional value and layout. A "cooldown_started" pulse or a missing anchor
 * draws nothing. Add the allocated leader, then a ring and core, a tick mark
 * for Ultimate Ready and a "+N" value when the event has a finite value. A
 * primary-button press on the hit circle does not reach the arena. Return
 * undefined.
 *
 * @param {Document} doc
 * @param {SVGElement} group
 * @param {SVGElement} connector
 * @param {JsonRecord} event
 */
function buildPulse(doc, group, connector, event) {
  if (event.cueSemantic === "cooldown_started") {
    return;
  }
  if (!event.anchor) {
    return;
  }
  appendAllocatedLeader(
    doc,
    connector,
    "combat-cue__leader combat-cue__leader--semantic",
    event.cueLeader,
  );
  const pulse = effectNode(doc, "g", {
    class: `combat-semantic-pulse combat-semantic-pulse--${cssIdentifier(event.cueSemantic)}`,
    "data-semantic": event.cueSemantic,
  });
  assignLayoutPlacement(
    pulse,
    event.cueLayoutKey,
    event.cueBounds,
    event.cueDisposition,
    event.cueCollisionFree,
  );
  const pulseHit = effectNode(doc, "circle", {
    class: "combat-semantic-pulse__hit",
    r: 31,
  });
  pulseHit.addEventListener("pointerdown", (pointerEvent) => {
    if (pointerEvent.button === 0) {
      pointerEvent.stopPropagation();
    }
  });
  pulse.append(
    pulseHit,
    effectNode(doc, "circle", {
      class: "combat-semantic-pulse__ring",
      r: 19,
    }),
    effectNode(doc, "circle", {
      class: "combat-semantic-pulse__core",
      r: 10,
    }),
  );
  if (event.cueSemantic === "cooldown_ready") {
    pulse.append(
      effectNode(doc, "path", {
        class: "combat-semantic-pulse__mark",
        d: "M -7 0 L -2 6 L 8 -7",
      }),
    );
  }
  if (Number.isFinite(event.value)) {
    const value = effectNode(doc, "text", {
      class: "combat-semantic-pulse__value",
      x: 0,
      y: 29,
    });
    value.textContent = `+${formatDisplayNumber(event.value)}`;
    pulse.append(value);
  }
  group.append(pulse);
  const anchor = event.cue ?? event.anchor;
  setEffectAttributes(pulse, {
    transform: `translate(${anchor.x} ${anchor.y})`,
  });
}

/**
 * Move this look's nodes to new records (reprojection).
 *
 * nodes and records come in matching order. Aura records take the [circle,
 * marks] pairs build returned: each circle gets its new center and radius and
 * its marks are redrawn. Status-mark groups move to the new body center; a
 * group whose body radius changed is replaced in the DOM by a fresh one (the
 * old node object is detached, so callers that keep nodes should rebuild).
 * Event records take nodes [group, connector, ...] as build
 * returned: a plain pulse moves to its cue (or anchor) and syncs its leader; a
 * Ward Lift ring moves to its recipient. Dock records are not moved here:
 * `scene.js` lays out and rebuilds docks on every render. Return undefined.
 *
 * @param {ReadonlyArray<Element | null>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  const first = records[0];
  if (first === undefined) {
    return;
  }
  if (first.kind === "aura-field") {
    // build gave [circle, marks] per record, in record order.
    records.forEach((record, index) => {
      const circle = nodes[index * 2];
      const marks = nodes[index * 2 + 1];
      if (!circle || !marks) return;
      setNodeAttributes(/** @type {SVGElement} */ (circle), {
        cx: record.center.x,
        cy: record.center.y,
        r: record.radius,
      });
      fillWardMarks(marks.ownerDocument, /** @type {SVGElement} */ (marks), record);
    });
    return;
  }
  if (first.kind === "status-marks") {
    records.forEach((record, index) => {
      const group = nodes[index];
      if (!group) return;
      if (group.getAttribute("data-radius") === String(f(record.radius))) {
        group.setAttribute(
          "transform",
          `translate(${f(record.center.x)} ${f(record.center.y)})`,
        );
        return;
      }
      group.replaceWith(buildStatusMarks(group.ownerDocument, record));
    });
    return;
  }
  const [group, connector = null] = nodes;
  if (!group) {
    return;
  }
  const event = first.event;
  if (first.kind === "ward-lift") {
    const ring = group.querySelector(".ward-lift");
    if (ring && isRecord(event.recipient)) {
      ring.setAttribute("cx", String(f(finiteNumber(event.recipient.x))));
      ring.setAttribute("cy", String(f(finiteNumber(event.recipient.y))));
    }
    return;
  }
  if (first.kind === "semantic-pulse") {
    const pulse = group.querySelector(".combat-semantic-pulse");
    if (pulse && event.anchor) {
      const cue = event.cue ?? event.anchor;
      pulse.setAttribute("transform", `translate(${cue.x} ${cue.y})`);
      assignLayoutPlacement(
        pulse,
        event.cueLayoutKey,
        event.cueBounds,
        event.cueDisposition,
        event.cueCollisionFree,
      );
    }
    syncAllocatedLeader(connector, ".combat-cue__leader--semantic", event.cueLeader);
  }
}

/**
 * Return the tooltip for one of this look's records.
 *
 * - aura-field: the aura explanation from its tooltip input {field,
 *   sourceAgent, audience};
 * - dock-cell and dock-overflow (made inside build): the status, POV status,
 *   modifier, overflow or POV overflow explanation named by the input's kind;
 * - cooldown-cell (made inside build): the Ultimate cooldown explanation for
 *   the owner and ticks, or null when the owner cannot be explained;
 * - semantic-pulse and ward-lift: the shared event explanation (for the Ward
 *   Lift, the expired Spawn Shield).
 * Any other record returns null. Agent POV: only what the inputs hold.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  if (record.kind === "aura-field") {
    const input = record.tooltipInput;
    return explainAura(input.field, input.sourceAgent, input.audience);
  }
  if (record.kind === "dock-cell" || record.kind === "dock-overflow") {
    const input = /** @type {DockTooltipInput} */ (record.tooltipInput);
    if (input.kind === "status") {
      return explainStatus(input.status, input.recipient, [...input.sourceAgents]);
    }
    if (input.kind === "pov-status") {
      return explainPovStatus(input.status, input.recipient);
    }
    if (input.kind === "modifier") {
      return explainModifier(input.modifier, input.recipient);
    }
    if (input.kind === "overflow") {
      return explainOverflow(input.items, input.dockKind, input.recipient, [
        ...input.sourceAgents,
      ]);
    }
    return explainPovOverflow(input.items, input.recipient);
  }
  if (record.kind === "cooldown-cell") {
    return explainCooldown(
      {
        ...displayIdentityRecord(record.ownerAgent),
        ultimate_cooldown: record.ticks,
      },
      record.ownerAgent,
    );
  }
  if (record.kind === "semantic-pulse" || record.kind === "ward-lift") {
    return explainChoreographyEvent(paintAwareExplanationEvent(record.event));
  }
  return null;
}
