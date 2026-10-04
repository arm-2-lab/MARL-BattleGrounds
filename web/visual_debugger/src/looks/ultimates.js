/**
 * @file The Ultimates look (V5 "Vanguard, Rime, Kindling" and "Nightshade
 * Thrust, Dawn Pillar"): how every Ultimate activation is drawn on the arena.
 *
 * - Charge (Warrior): a tapered bronze war-path from the Warrior's START rim to
 *   its landing that grows with the rush, a Forge White head, a crunch star on
 *   the target's END rim, an open bronze clasp with two rivets, rush rings while
 *   bodies jump, a grey "Charge" label at the START spot and a 35% Bronze Shade
 *   echo of the path when the Warrior travelled 1.25 units or more.
 * - Freezing Trap (Hunter): a dashed cord between the two champions and a
 *   hexagon snare of Rime that falls from 1.4R onto the foe.
 * - Burst (Mage): eight violet slivers fold into the Mage, then her star flares.
 * - Crippling Poison (Rogue): a silver stiletto drawn back and driven in, a tip
 *   flash, a venom seep on the rim, a resting blade at 70% and a 30% tether.
 * - Holy Word: Salvation (Priest): a telegraph ring, a golden Gilded Bolt from
 *   the Priest to the ally (a 7 px Dawnlight edge, a 4.5 px gold body and a 2 px
 *   Dawnlight core, with a 12 px Soft Glow while playing), three rings that
 *   collapse inward, a Dawnlight flood with eight rays, and in a still the bolt
 *   at 70% and a 35% echo ring open at 12 o'clock.
 *
 * Lasting marks (the Frozen shell with its icicles, the Burst runes, ember and
 * under-glow, Crack, Shatter and Thaw) belong to the status scene, which
 * `statuses.js` draws at Settle from authorized status rows. This file exports
 * their shared shapes (frostShellShape, iceCrackPath, iceShardsPath,
 * thawGhostPath, burstRuneShapes) so both pictures stay the same.
 *
 * Who calls it: `choreography-painter.js` sends every activation whose
 * component is "ultimate" here. It calls records({event}), build(records, doc,
 * mount) into its event group (above bodies) and route group (the Strike Band,
 * under bodies), and update(nodes, records) on a resize or camera change.
 *
 * Timing comes from the plan's beat schedule (mount.plan.beats): each mark is
 * authored in blow time and mapped through the schedule's published strike,
 * impact and settle windows, and the Charge rush uses its chargeRush window. The
 * DOM always holds the END picture (the settled still). Every movement is a Web
 * Animations override with fill "backwards" handed to the choreographer through
 * mount.animationSpecs; nothing here starts a timer or a CSS animation. In a
 * played tick the choreographer keeps a transient mark hidden once a cue that
 * ends at opacity 0 has finished (choreography.js endsHidden), so the still
 * does not come back for the rest of the tick.
 *
 * Agent POV boundary: records reads only the one plan event it is given, which
 * the plan already admitted for this audience. It never reads a researcher
 * view, a match summary or a body. Extra rules on top of the plan:
 * the Charge path needs both START anchors; a hidden Warrior leaves only the
 * four-arc Unseen Clasp on the target (no path, crunch or label); a target not
 * seen at END gets no clasp; the Trap cord needs both champions; Poison and
 * Salvation are drawn only when both champions were seen at END (both endpoint
 * phases "successor"), otherwise nothing at all.
 *
 * Export safety: only flat fills, strokes, opacity and transforms. No ids,
 * gradients, masks, filters, url() or <image>.
 */
import { explainActivation } from "../explanations.js";
import {
  assignLayoutPlacement,
  detachedEventMount,
  effectNode,
  explainChoreographyEvent,
  mountNodes,
  paintAwareExplanationEvent,
  paintPartEnabled,
  paintParts,
  semanticImpactGlyph,
  setEffectAttributes,
} from "./look-support.js";
import { ACTIVATION_FOOTPRINTS } from "./strike-routes.js";

/**
 * @typedef {import("./look-support.js").JsonRecord} JsonRecord
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").EventMount} EventMount
 * @typedef {import("./look-support.js").AnimationSpec} AnimationSpec
 * @typedef {readonly [number, number]} Vec
 *   A screen point or direction in CSS pixels, [x, y], y down.
 * @typedef {Readonly<{
 *   tag: string,
 *   attrs: Readonly<Record<string, string | number>>,
 *   text?: string,
 *   children?: ReadonlyArray<UltNode>,
 * }>} UltNode
 *   One SVG node to draw, as plain data. attrs always holds a "data-ult-key"
 *   naming the node's job (for example "charge.clasp"); keys are unique within
 *   one Ultimate. attrs.opacity is the node's END (settled still) opacity.
 * @typedef {Readonly<{
 *   t: number | "rushStart" | "rushEnd",
 *   clock?: "part" | "real",
 *   o: number,
 *   x?: string,
 *   e?: string,
 * }>} UltFrame
 *   One keyframe. t is a time in ms: blow time when clock is "part" (default),
 *   real tick time when "real", or the start or end of the plan's Charge rush
 *   window. In a reduced list t is ms of the 220 ms reduced tick. o is opacity
 *   (0-1); x is a CSS transform such as "scale(1.4)" (required on every frame of
 *   a cue that moves); e is the easing of the segment after this frame.
 * @typedef {Readonly<{key: string, frames: ReadonlyArray<UltFrame>}>} UltCue
 *   The animation of the node with data-ult-key key.
 * @typedef {Readonly<{
 *   glide: ReadonlyArray<UltCue>,
 *   plain: ReadonlyArray<UltCue>,
 *   reduced: ReadonlyArray<UltCue>,
 * }>} UltCues
 *   The cues for Body Glide on (glide), Body Glide off (plain) and reduced
 *   motion (reduced). Motion off and settled installs use none.
 * @typedef {Readonly<{
 *   key: string,
 *   keyframes: Keyframe[],
 *   options: KeyframeAnimationOptions,
 * }>} UltTrack
 *   One browser animation description in real ms: the node key, its keyframes
 *   and its timing (always fill "backwards", easing linear between frames).
 */

/**
 * Ultimate footprints in CSS pixels, kept equal to the allocator table in
 * `strike-routes.js` (the plan reads them for collision layout): each token's
 * impact box (flare extents fit inside), the caster mark of a source-local
 * Ultimate, the Charge label box, the meaning-only marks drawn when ability
 * paint is off, and the route padding the allocator keeps clear. The V5 marks
 * sit on the champions' own bodies (inside 1.6R), so these boxes did not change.
 */
export const FOOTPRINT = Object.freeze({
  activation: Object.freeze({
    holy_word: ACTIVATION_FOOTPRINTS.activation.holy_word,
    hunter_trap: ACTIVATION_FOOTPRINTS.activation.hunter_trap,
    rogue_poison: ACTIVATION_FOOTPRINTS.activation.rogue_poison,
    warrior_charge: ACTIVATION_FOOTPRINTS.activation.warrior_charge,
    mage_burst: ACTIVATION_FOOTPRINTS.activation.mage_burst,
    local: ACTIVATION_FOOTPRINTS.activation.local,
  }),
  chargeOwnership: ACTIVATION_FOOTPRINTS.chargeOwnership,
  semanticOnlyImpact: Object.freeze({
    hunter_trap: Object.freeze({ width: 16, height: 16 }),
    warrior_charge: Object.freeze({ width: 18, height: 18 }),
    default: Object.freeze({ width: 24, height: 24 }),
  }),
  route: Object.freeze({
    holyWordPathPadding: ACTIVATION_FOOTPRINTS.route.activationPathPadding.holy_word,
    chargePathPadding: ACTIVATION_FOOTPRINTS.route.chargePathPadding,
    chargeMarkerPadding: ACTIVATION_FOOTPRINTS.route.chargeMarkerPadding,
  }),
});

/**
 * The node and animation allowance of one Ultimate, read by the plan's resource
 * bounds: at most 30 SVG nodes (the busiest, Salvation, draws 26 including its
 * two roots, 23 with Soft Glow off) and 10 animations (Salvation draws 9, a
 * long Charge with jumping bodies 10). Nothing an Ultimate draws is kept as a
 * persistent node.
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 30,
  animationsPerRecord: 10,
  persistentNodesPerRecord: 0,
});

/**
 * Tells the painter that this look times its own marks (true). The painter
 * should then add no whole-group fade for Ultimate events, so the still marks
 * stay after the tick. Read only by `choreography-painter.js`.
 */
export const OWNS_TIMING = true;

/** Ultimate Ability Effects switches the Ultimate's ability mark and its meaning mark. */
export const PAINT_PARTS = paintParts([
  [
    {
      surface: "transient",
      kind: "activation",
      component: "ultimate",
      part: "ability",
    },
    "ultimate_ability_effects",
  ],
  [
    {
      surface: "transient",
      kind: "activation",
      component: "ultimate",
      part: "semantic",
    },
    "ultimate_ability_effects",
  ],
]);

/**
 * Body radius in CSS pixels used when the event gives none (a 0.5-unit champion
 * at about 40 px per unit). The store designed every size at 23 px.
 */
export const DEFAULT_BODY_RADIUS_PX = 20;

/** The Ultimate variant drawn for each Ultimate token id. */
const VARIANT_BY_TOKEN = Object.freeze({
  warrior_charge: "charge",
  hunter_trap: "trap",
  mage_burst: "burst",
  rogue_poison: "poison",
  holy_word: "salvation",
});

const DEG = Math.PI / 180;

/**
 * Round a number to 2 decimals for SVG attributes. Pure.
 *
 * @param {number} value
 * @returns {number}
 */
function r2(value) {
  return Math.round(value * 100) / 100;
}

/**
 * Read a screen point {x, y} (or [x, y]) as a Vec, or null when it is not finite.
 *
 * @param {unknown} value
 * @returns {Vec | null}
 */
function vec(value) {
  if (Array.isArray(value)) {
    return Number.isFinite(value[0]) && Number.isFinite(value[1])
      ? [value[0], value[1]]
      : null;
  }
  const point = /** @type {{x?: unknown, y?: unknown} | null} */ (value);
  return point && Number.isFinite(point.x) && Number.isFinite(point.y)
    ? [Number(point.x), Number(point.y)]
    : null;
}

/** @param {Vec} a @param {Vec} b @returns {Vec} */
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
/** @param {Vec} a @param {Vec} b @returns {Vec} */
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
/** @param {Vec} a @param {number} s @returns {Vec} */
const mul = (a, s) => [a[0] * s, a[1] * s];
/** @param {Vec} a */
const len = (a) => Math.hypot(a[0], a[1]);
/** @param {Vec} a @param {Vec} b @param {number} q @returns {Vec} */
const lerp = (a, b, q) => [a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q];
/** @param {Vec} a @returns {Vec} */
const norm = (a) => {
  const length = len(a);
  return length > 1e-9 ? [a[0] / length, a[1] / length] : [1, 0];
};
/** @param {Vec} c @param {number} radius @param {number} angle screen radians @returns {Vec} */
const polar = (c, radius, angle) => [
  c[0] + radius * Math.cos(angle),
  c[1] + radius * Math.sin(angle),
];
/**
 * A point at a clock angle (radians from 12 o'clock, clockwise) around c.
 *
 * @param {Vec} c @param {number} radius @param {number} angle @returns {Vec}
 */
const clock = (c, radius, angle) => [
  c[0] + radius * Math.sin(angle),
  c[1] - radius * Math.cos(angle),
];
/** @param {Vec} p */
const pt = (p) => `${r2(p[0])} ${r2(p[1])}`;
/** @param {ReadonlyArray<Vec>} points */
const poly = (points) => `M ${points.map(pt).join(" L ")} Z`;
/** @param {Vec} c @param {number} radius A closed circle as path data. */
const circlePath = (c, radius) =>
  `M ${pt([c[0] - radius, c[1]])} A ${r2(radius)} ${r2(radius)} 0 1 0 ${pt([c[0] + radius, c[1]])} A ${r2(radius)} ${r2(radius)} 0 1 0 ${pt([c[0] - radius, c[1]])} Z`;
/** @param {Vec} at @param {number} [angle] radians */
const frame = (at, angle = 0) =>
  angle === 0
    ? `translate(${pt(at)})`
    : `translate(${pt(at)}) rotate(${r2(angle / DEG)})`;

/**
 * Build one frozen UltNode. Pure.
 *
 * @param {string} tag SVG tag name.
 * @param {string} key The node's data-ult-key.
 * @param {Record<string, string | number | null | undefined>} attrs Other
 *   attributes; null and undefined values are left out.
 * @param {ReadonlyArray<UltNode | null>} [children] Child nodes; null is skipped.
 * @param {string} [text] Text content (for <text>).
 * @returns {UltNode}
 */
function node(tag, key, attrs, children, text) {
  /** @type {Record<string, string | number>} */
  const clean = { "data-ult-key": key };
  for (const [name, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) {
      clean[name] = typeof value === "number" ? r2(value) : value;
    }
  }
  return Object.freeze({
    tag,
    attrs: Object.freeze(clean),
    ...(children
      ? {
          children: Object.freeze(
            /** @type {UltNode[]} */ (children.filter((child) => child !== null)),
          ),
        }
      : {}),
    ...(text === undefined ? {} : { text }),
  });
}

/**
 * Freeze a cue list. Pure.
 *
 * @param {ReadonlyArray<[string, ReadonlyArray<UltFrame>] | null>} rows [key, frames] pairs; null rows are skipped.
 * @returns {ReadonlyArray<UltCue>}
 */
function cues(rows) {
  return Object.freeze(
    rows
      .filter((row) => row !== null)
      .map((row) => {
        const [key, frames] = /** @type {[string, ReadonlyArray<UltFrame>]} */ (row);
        return Object.freeze({
          key,
          frames: Object.freeze(frames.map((each) => Object.freeze({ ...each }))),
        });
      }),
  );
}

/**
 * The flat-top hexagon of the Frozen shell and its four icicles, around center.
 *
 * center is the frozen champion's END centre and radius its body radius (CSS
 * pixels). count (default 4, clamped 0-4) is how many icicles hang from the flat
 * top edge, one per tick of stun left. Return frozen {shell, icicles} path data:
 * shell at 0.60R (Frostglass fill about 30%, Rime edge), icicles as small Rime
 * triangles 3k long (k = radius / 23). Pure; the status scene draws it.
 *
 * @param {{x: number, y: number}} center
 * @param {number} radius
 * @param {number} [count]
 * @returns {Readonly<{shell: string, icicles: string}>}
 */
export function frostShellShape(center, radius, count = 4) {
  const c = vec(center) ?? [0, 0];
  const k = radius / 23;
  const rs = 0.6 * radius;
  const corners = Array.from({ length: 6 }, (_, i) => polar(c, rs, i * 60 * DEG));
  const top = c[1] - rs * Math.sin(60 * DEG);
  const xs = [-3.9, -1.3, 1.3, 3.9];
  const icicles = xs
    .slice(0, Math.max(0, Math.min(4, Math.trunc(count))))
    .map((dx) => {
      const x = c[0] + dx * k;
      return poly([
        [x - k, top],
        [x + k, top],
        [x, top + 3 * k],
      ]);
    })
    .join(" ");
  return Object.freeze({ shell: poly(corners), icicles });
}

/**
 * The crack across a breaking Frozen shell: a five-point zig-zag from the
 * shell corner nearest the breaking blow, through the centre, to the far corner.
 *
 * center and radius as in frostShellShape; direction is the screen angle
 * (radians) toward the nearest authorized breaking source. With no seen source
 * pass the 1 o'clock corner (-60 degrees). Return path data. Pure.
 *
 * @param {{x: number, y: number}} center
 * @param {number} radius
 * @param {number} direction
 * @returns {string}
 */
export function iceCrackPath(center, radius, direction) {
  const c = vec(center) ?? [0, 0];
  const k = radius / 23;
  const rs = 0.6 * radius;
  let best = 0;
  let bestCos = -2;
  for (let i = 0; i < 6; i += 1) {
    const cosine = Math.cos(i * 60 * DEG - direction);
    if (cosine > bestCos) {
      bestCos = cosine;
      best = i;
    }
  }
  const a = polar(c, rs, best * 60 * DEG);
  const b = polar(c, rs, (best + 3) * 60 * DEG);
  const axis = norm(sub(b, a));
  const normal = /** @type {Vec} */ ([-axis[1], axis[0]]);
  const length = len(sub(b, a));
  const points = [a];
  for (const [q, offset] of [
    [0.25, -2.4],
    [0.5, 2.4],
    [0.75, -2.4],
  ]) {
    points.push(add(add(a, mul(axis, length * q)), mul(normal, offset * k)));
  }
  points.push(b);
  return `M ${points.map(pt).join(" L ")}`;
}

/**
 * Six tilted Rime shards resting just outside the rim after a shatter.
 *
 * center and radius as in frostShellShape. Return path data of six small
 * triangles at 1.15R, at 30 + 60n degrees, each tilted 25 degrees. Pure.
 *
 * @param {{x: number, y: number}} center
 * @param {number} radius
 * @returns {string}
 */
export function iceShardsPath(center, radius) {
  const c = vec(center) ?? [0, 0];
  const k = radius / 23;
  const lengths = [5.4, 4.4, 5.8, 4.8, 5.6, 4.2];
  const bases = [3, 2.6, 3.2, 2.8, 3.1, 2.6];
  const parts = [];
  for (let i = 0; i < 6; i += 1) {
    const angle = (30 + 60 * i) * DEG;
    const at = polar(c, 1.15 * radius, angle);
    const tilt = angle + (i % 2 ? -25 : 25) * DEG;
    const along = /** @type {Vec} */ ([Math.cos(tilt), Math.sin(tilt)]);
    const across = /** @type {Vec} */ ([-along[1], along[0]]);
    const half = (lengths[i] * k) / 2;
    const base = (bases[i] * k) / 2;
    const tip = add(at, mul(along, half));
    const back = sub(at, mul(along, half));
    parts.push(poly([tip, add(back, mul(across, base)), sub(back, mul(across, base))]));
  }
  return parts.join(" ");
}

/**
 * The dashed ghost a thawed shell leaves: the shell's hexagon outline.
 *
 * center and radius as in frostShellShape. Return path data (draw it with a
 * Frostglass 1k stroke dashed 2k 2k at 45%). Pure.
 *
 * @param {{x: number, y: number}} center
 * @param {number} radius
 * @returns {string}
 */
export function thawGhostPath(center, radius) {
  const c = vec(center) ?? [0, 0];
  return poly(
    Array.from({ length: 6 }, (_, i) => polar(c, 0.6 * radius, i * 60 * DEG)),
  );
}

/**
 * The Burst runes, ember and under-glow on a Mage under Burst (Damage Boost).
 *
 * center and radius as in frostShellShape; count (default 5, clamped 0-5) is
 * how many runes are still lit, one per tick left. Runes are Starcore
 * eight-point stars on the 0.68R Rune Ring, set as a flat-top pentagon (36, 108,
 * 180, 252 and 324 degrees from 12 o'clock) so the Crown stays clear. Return
 * frozen {runes, ember, emberRadius, glowRadius}: rune path data, the ember
 * centre and radius (0.3R, Starcore at 16%, with a 0.11R core) and the violet
 * under-glow radius (0.5R). Pure; the status scene draws it.
 *
 * @param {{x: number, y: number}} center
 * @param {number} radius
 * @param {number} [count]
 * @returns {Readonly<{runes: string, ember: Readonly<{x: number, y: number}>, emberRadius: number, glowRadius: number}>}
 */
export function burstRuneShapes(center, radius, count = 5) {
  const c = vec(center) ?? [0, 0];
  const k = radius / 23;
  const lit = Math.max(0, Math.min(5, Math.trunc(count)));
  const runes = [];
  for (let n = 0; n < lit; n += 1) {
    const at = clock(c, 0.68 * radius, (36 + 72 * n) * DEG);
    const points = [];
    for (let i = 0; i < 16; i += 1) {
      const reach = (i % 4 === 0 ? 1.9 : i % 2 === 0 ? 1.25 : 0.6) * k;
      points.push(polar(at, reach, (i * 22.5 - 90) * DEG));
    }
    runes.push(poly(points));
  }
  return Object.freeze({
    runes: runes.join(" "),
    ember: Object.freeze({ x: c[0], y: c[1] }),
    emberRadius: r2(0.3 * radius),
    glowRadius: r2(0.5 * radius),
  });
}

/**
 * Read a positive pixel radius from an event field, or the default.
 *
 * @param {unknown} value
 * @param {number} fallback
 * @returns {number}
 */
function radiusOf(value, fallback) {
  return Number.isFinite(value) && Number(value) > 0 ? Number(value) : fallback;
}

/**
 * Return the drawing record for an Ultimate activation, or [] when nothing may be drawn.
 *
 * input.event is one admitted plan event in CSS pixels; any other event (or a
 * missing one) gives []. The event fields read are: kind, component, tokenId,
 * presentationKind, paintParts (ability, semantic and the optional softGlow;
 * absent means all on), source and target (screen points; the plan puts them
 * at END, except the Charge which keeps both START spots), sourceEndpointPhase
 * and targetEndpointPhase ("successor" means seen at END), source and target
 * presentation keys (a self-cast Salvation has no bolt), and these optional
 * fields (absent means the fallback shown):
 * - sourceRadius, targetRadius: body radii in px (DEFAULT_BODY_RADIUS_PX).
 * - unitPx: px per world unit (2 x the Warrior's radius, a 0.5-unit body).
 * - chargeLanding: where the Charge rush stopped (Oracle only). Without it the
 *   war-path ends at the target's near rim, the recorded relation.
 * - chargeTravel: rush length in world units (the drawn length / unitPx).
 * - sourceEnd, targetEnd: END spots for the Charge clasp; null means "not seen
 *   at END" (no clasp on an unseen target), undefined means "use START".
 *
 * Every Ultimate activation gives exactly one frozen record (this look owns
 * the event; any other event gives []). The record is {kind: "ultimate",
 * variant, tokenId, event, band, over, cues} plus a few facts per variant
 * (hasPath, isLong, travel, unseenClasp, tether, selfCast): variant is
 * "charge", "trap", "burst", "poison", "salvation", "plain" (meaning mark
 * only, when ability paint is off) or "none" (nothing may be drawn: a Poison
 * or Salvation with a champion not seen at END, a Charge or Trap with no
 * target, a Burst with no caster, or every paint part off); band and over are
 * UltNode lists for the Strike Band (under bodies) and the event layer (above
 * bodies), both empty for "none"; cues holds the glide, plain and reduced
 * animation lists. Pure: no DOM, no timers, the event is not changed.
 *
 * @param {{event?: JsonRecord | null}} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const event = input.event;
  if (event?.kind !== "activation" || event.component !== "ultimate") {
    return Object.freeze([]);
  }
  const ability = paintPartEnabled(event, "ability");
  const semantic = paintPartEnabled(event, "semantic");
  const softGlow =
    !event.paintParts ||
    typeof event.paintParts !== "object" ||
    event.paintParts.softGlow !== false;
  const variant = /** @type {Record<string, string>} */ (VARIANT_BY_TOKEN)[
    String(event.tokenId)
  ];
  /** @type {Omit<LookRecord, "kind" | "variant" | "tokenId" | "event"> | null} */
  let drawing = null;
  if (ability && variant === "charge") drawing = chargeDrawing(event);
  else if (ability && variant === "trap") drawing = trapDrawing(event);
  else if (ability && variant === "burst") drawing = burstDrawing(event);
  else if (ability && variant === "poison") drawing = poisonDrawing(event, softGlow);
  else if (ability && variant === "salvation")
    drawing = salvationDrawing(event, softGlow);
  else if (semantic && vec(event.target)) drawing = plainDrawing(event);
  const chosen = drawing === null ? "none" : ability && variant ? variant : "plain";
  if (drawing === null) {
    drawing = {
      band: Object.freeze([]),
      over: Object.freeze([]),
      cues: Object.freeze({ glide: cues([]), plain: cues([]), reduced: cues([]) }),
    };
  }
  return Object.freeze([
    Object.freeze({
      kind: "ultimate",
      variant: chosen,
      tokenId: event.tokenId ?? null,
      event,
      ...drawing,
    }),
  ]);
}

/**
 * The meaning-only mark (damage or healing) at the target, for the switch-off rule.
 *
 * @param {JsonRecord} event
 */
function plainDrawing(event) {
  const target = /** @type {Vec} */ (vec(event.target));
  return {
    band: Object.freeze([]),
    over: Object.freeze([
      node("g", "plain.at", { transform: frame(target), class: "ult-plain" }, [
        node("circle", "plain.hit", { class: "ult-hit-area", r: 12 }),
      ]),
    ]),
    cues: Object.freeze({ glide: cues([]), plain: cues([]), reduced: cues([]) }),
    semantic: event.impactSemantic ?? null,
  };
}

/**
 * The Charge drawing: war-path, echo, head, crunch, clasp, rush rings and label.
 *
 * @param {JsonRecord} event
 */
function chargeDrawing(event) {
  const start = vec(event.source);
  const targetStart = vec(event.target);
  if (targetStart === null) {
    return null;
  }
  const rS = radiusOf(event.sourceRadius, DEFAULT_BODY_RADIUS_PX);
  const R = radiusOf(event.targetRadius, rS);
  const k = R / 23;
  const kS = rS / 23;
  const targetEnd = event.targetEnd === undefined ? targetStart : vec(event.targetEnd);
  /** @type {UltNode[]} */
  const band = [];
  /** @type {UltNode[]} */
  const over = [];
  /** @type {Array<[string, UltFrame[]] | null>} */
  const glide = [];
  /** @type {Array<[string, UltFrame[]] | null>} */
  const plain = [];
  /** @type {Array<[string, UltFrame[]] | null>} */
  const reduced = [];
  if (start === null) {
    // Hidden Warrior: only the Unseen Clasp, which names no direction.
    if (targetEnd === null) return null;
    const rr = 1.15 * R;
    const arcs = [45, 135, 225, 315]
      .map((centre) => {
        const a = polar([0, 0], rr, (centre - 10) * DEG);
        const b = polar([0, 0], rr, (centre + 10) * DEG);
        return `M ${pt(a)} A ${r2(rr)} ${r2(rr)} 0 0 1 ${pt(b)}`;
      })
      .join(" ");
    over.push(
      node("g", "charge.claspAt", { transform: frame(targetEnd) }, [
        node("circle", "charge.hit", { class: "ult-hit-area", r: rr }),
        node("g", "charge.clasp", { class: "ult-charge__clasp-group", opacity: 0.8 }, [
          node("path", "charge.claspArcs", {
            class: "ult-charge__clasp ult-charge__clasp--unseen",
            d: arcs,
            "stroke-width": Math.max(1, 3 * k),
          }),
        ]),
      ]),
    );
    const claspIn = [
      { t: 79, o: 0 },
      { t: 80, o: 0 },
      { t: 180, o: 1 },
      { t: 560, o: 0.8 },
    ];
    glide.push(["charge.clasp", claspIn]);
    plain.push([
      "charge.clasp",
      [
        { t: 359, o: 0 },
        { t: 360, o: 0 },
        { t: 460, o: 1 },
        { t: 760, o: 0.8 },
      ],
    ]);
    reduced.push([
      "charge.clasp",
      [
        { t: 60, o: 0 },
        { t: 140, o: 1 },
        { t: 220, o: 0.8 },
      ],
    ]);
    return {
      band: Object.freeze(band),
      over: Object.freeze(over),
      cues: Object.freeze({
        glide: cues(glide),
        plain: cues(plain),
        reduced: cues(reduced),
      }),
      unseenClasp: true,
    };
  }
  const u = norm(sub(targetStart, start));
  const landing = vec(event.chargeLanding) ?? sub(targetStart, mul(u, R + rS));
  const unitPx = radiusOf(event.unitPx, rS / 0.5);
  const travel = Number.isFinite(event.chargeTravel)
    ? Number(event.chargeTravel)
    : len(sub(landing, start)) / unitPx;
  const p0 = add(start, mul(u, rS));
  const p1 = add(landing, mul(u, rS));
  const length = len(sub(p1, p0));
  const hasPath = travel >= 0.25 && length > 1.5;
  const isLong = travel >= 1.25;
  const angle = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
  const c0 = Math.max(1.25, 2 * kS);
  const c1 = Math.max(2.5, 4 * kS);
  /**
   * A tapered band from x = 0 (width w0) to x = length (width w1). Pure.
   *
   * @param {number} w0
   * @param {number} w1
   */
  const taper = (w0, w1) =>
    poly([
      [0, w0 / 2],
      [length, w1 / 2],
      [length, -w1 / 2],
      [0, -w0 / 2],
    ]);
  if (hasPath) {
    band.push(
      node("line", "charge.hit", {
        class: "ult-hit",
        x1: start[0],
        y1: start[1],
        x2: p1[0],
        y2: p1[1],
      }),
      node("g", "charge.frame", { transform: frame(p0, angle) }, [
        isLong
          ? node("path", "charge.echo", {
              class: "ult-charge__echo",
              d: taper(c0, c1),
              opacity: 0.35,
            })
          : null,
        node("g", "charge.grow", { class: "ult-charge__grow", opacity: 0 }, [
          node("path", "charge.glow", {
            class: "ult-charge__glow ult-soft-glow",
            d: taper(3 * c0, 3 * c1),
            opacity: 0.18,
          }),
          node("path", "charge.core", {
            class: "ult-charge__core",
            d: taper(c0, c1),
            opacity: 0.9,
          }),
          node("line", "charge.hair", {
            class: "ult-charge__hair",
            x1: 0,
            y1: 0,
            x2: length,
            y2: 0,
          }),
        ]),
      ]),
    );
    const headWidth = 4 * kS;
    const headLength = Math.min(23 * kS, length);
    over.push(
      node("g", "charge.headFrame", { transform: frame(p0, angle) }, [
        node("g", "charge.head", { opacity: 0 }, [
          node("line", "charge.headLine", {
            class: "ult-charge__head",
            x1: -headLength + headWidth / 2,
            y1: 0,
            x2: -headWidth / 2,
            y2: 0,
            "stroke-width": headWidth,
          }),
        ]),
      ]),
    );
    const grow = `scaleX(1)`;
    glide.push(
      [
        "charge.grow",
        [
          { t: "rushStart", o: 1, x: "scaleX(0.001)" },
          { t: "rushEnd", o: 1, x: grow },
          { t: 380, o: 1, x: grow },
          { t: 560, o: 0, x: grow },
        ],
      ],
      [
        "charge.head",
        [
          { t: "rushStart", o: 1, x: "translateX(0px)" },
          { t: "rushEnd", o: 1, x: `translateX(${r2(length)}px)` },
          { t: 120, o: 0, x: `translateX(${r2(length)}px)` },
        ],
      ],
    );
    plain.push(
      [
        "charge.grow",
        [
          { t: 79, o: 0, x: "scaleX(0.001)" },
          { t: 80, o: 1, x: "scaleX(0.001)", e: "ease-in" },
          { t: 360, o: 1, x: grow },
          { t: 580, o: 1, x: grow },
          { t: 760, o: 0, x: grow },
        ],
      ],
      [
        "charge.head",
        [
          { t: 79, o: 0, x: "translateX(0px)" },
          { t: 80, o: 1, x: "translateX(0px)", e: "ease-in" },
          { t: 360, o: 1, x: `translateX(${r2(length)}px)` },
          { t: 420, o: 0, x: `translateX(${r2(length)}px)` },
        ],
      ],
    );
    reduced.push([
      "charge.grow",
      [
        { t: 0, o: 0 },
        { t: 80, o: 1 },
        { t: 120, o: 1 },
        { t: 220, o: 0 },
      ],
    ]);
    if (isLong) {
      glide.push([
        "charge.echo",
        [
          { t: 380, o: 0 },
          { t: 560, o: 0.35 },
        ],
      ]);
      plain.push([
        "charge.echo",
        [
          { t: 580, o: 0 },
          { t: 760, o: 0.35 },
        ],
      ]);
      reduced.push([
        "charge.echo",
        [
          { t: 120, o: 0 },
          { t: 220, o: 0.35 },
        ],
      ]);
      // Rush rings mark the rush only while bodies jump (Body Glide off).
      const rings =
        travel >= 2
          ? [
              [0.25, 500, 0.1],
              [0.5, 530, 0.2],
              [0.75, 555, 0.3],
            ]
          : [[0.5, 530, 0.3]];
      rings.forEach(([q, at, peak], index) => {
        const centre = lerp(start, landing, q);
        band.push(
          node("circle", `charge.ring${index}`, {
            class: "ult-charge__ring",
            cx: centre[0],
            cy: centre[1],
            r: 0.93 * rS,
            "stroke-width": Math.max(1, kS),
            opacity: 0,
          }),
        );
        plain.push([
          `charge.ring${index}`,
          [
            { t: at - 1, o: 0 },
            { t: at, o: peak },
            { t: at + 200, o: 0 },
          ],
        ]);
      });
    }
  }
  if (targetEnd !== null) {
    const sourceEnd =
      event.sourceEnd === undefined || event.sourceEnd === null
        ? (vec(event.chargeLanding) ?? start)
        : (vec(event.sourceEnd) ?? start);
    const toward = sub(sourceEnd, targetEnd);
    const face = len(toward) > 1 ? norm(toward) : /** @type {Vec} */ ([-u[0], -u[1]]);
    const theta = Math.atan2(face[1], face[0]);
    const rr = 1.15 * R;
    const contact = add(targetEnd, mul(face, R));
    const spikes = [];
    for (let i = 0; i < 12; i += 1) {
      spikes.push(polar([0, 0], i % 2 ? 0.6 * R : 5 * k, theta + i * 30 * DEG));
    }
    let arcs = "";
    let rivets = "";
    for (const sign of [1, -1]) {
      const a = polar([0, 0], rr, theta + sign * 28 * DEG);
      const b = polar([0, 0], rr, theta + sign * 56 * DEG);
      arcs += `M ${pt(a)} A ${r2(rr)} ${r2(rr)} 0 0 ${sign > 0 ? 1 : 0} ${pt(b)} `;
      rivets += `${circlePath(b, 1.4 * k)} `;
    }
    over.push(
      node("g", "charge.crunchAt", { transform: frame(contact) }, [
        node("g", "charge.crunch", { opacity: 0 }, [
          node("path", "charge.crunchStar", {
            class: "ult-charge__crunch",
            d: poly(spikes),
          }),
        ]),
      ]),
      node("g", "charge.claspAt", { transform: frame(targetEnd) }, [
        node("circle", "charge.targetHit", { class: "ult-hit-area", r: rr }),
        node("g", "charge.clasp", { class: "ult-charge__clasp-group", opacity: 0.8 }, [
          node("path", "charge.claspArcs", {
            class: "ult-charge__clasp",
            d: arcs.trim(),
            "stroke-width": Math.max(1, 3 * k),
          }),
          node("path", "charge.rivets", {
            class: "ult-charge__rivet",
            d: rivets.trim(),
          }),
        ]),
      ]),
    );
    glide.push(
      [
        "charge.crunch",
        [
          { t: 79, o: 0, x: "scale(1.3)" },
          { t: 80, o: 1, x: "scale(1.3)" },
          { t: 180, o: 1, x: "scale(1)" },
          { t: 340, o: 0, x: "scale(1)" },
        ],
      ],
      [
        "charge.clasp",
        [
          { t: 80, o: 0, x: "scale(1.2)" },
          { t: 180, o: 1, x: "scale(1)" },
          { t: 560, o: 0.8, x: "scale(1)" },
        ],
      ],
    );
    plain.push(
      [
        "charge.crunch",
        [
          { t: 359, o: 0, x: "scale(1.3)" },
          { t: 360, o: 1, x: "scale(1.3)" },
          { t: 460, o: 1, x: "scale(1)" },
          { t: 620, o: 0, x: "scale(1)" },
        ],
      ],
      [
        "charge.clasp",
        [
          { t: 360, o: 0, x: "scale(1.2)" },
          { t: 460, o: 1, x: "scale(1)" },
          { t: 760, o: 0.8, x: "scale(1)" },
        ],
      ],
    );
    reduced.push(
      [
        "charge.crunch",
        [
          { t: 60, o: 0 },
          { t: 140, o: 0.6 },
          { t: 220, o: 0 },
        ],
      ],
      [
        "charge.clasp",
        [
          { t: 60, o: 0 },
          { t: 140, o: 1 },
          { t: 220, o: 0.8 },
        ],
      ],
    );
  }
  const size = Math.max(10, 11 * kS);
  over.push(
    node(
      "text",
      "charge.label",
      {
        class: "ult-charge__label",
        x: start[0],
        y: start[1] + rS + 4 + size,
        "font-size": size,
        "text-anchor": "middle",
        opacity: 1,
      },
      undefined,
      "Charge",
    ),
  );
  glide.push([
    "charge.label",
    [
      { t: 380, o: 0 },
      { t: 500, o: 1 },
    ],
  ]);
  plain.push([
    "charge.label",
    [
      { t: 580, o: 0 },
      { t: 700, o: 1 },
    ],
  ]);
  reduced.push([
    "charge.label",
    [
      { t: 140, o: 0 },
      { t: 220, o: 1 },
    ],
  ]);
  return {
    band: Object.freeze(band),
    over: Object.freeze(over),
    cues: Object.freeze({
      glide: cues(glide),
      plain: cues(plain),
      reduced: cues(reduced),
    }),
    hasPath,
    isLong,
    travel: r2(travel),
    unseenClasp: false,
  };
}

/**
 * The Freezing Trap drawing: the dashed cord and the falling hexagon snare.
 *
 * @param {JsonRecord} event
 */
function trapDrawing(event) {
  const source = vec(event.source);
  const target = vec(event.target);
  if (target === null) {
    return null;
  }
  const R = radiusOf(event.targetRadius, DEFAULT_BODY_RADIUS_PX);
  const rS = radiusOf(event.sourceRadius, R);
  const k = R / 23;
  /** @type {UltNode[]} */
  const band = [];
  /** @type {Array<[string, UltFrame[]]>} */
  const shared = [];
  /** @type {Array<[string, UltFrame[]]>} */
  const reduced = [];
  if (source !== null) {
    const u = norm(sub(target, source));
    const a = add(source, mul(u, rS + 3));
    const b = sub(target, mul(u, R + 3));
    if (len(sub(b, a)) > 2) {
      band.push(
        node("line", "trap.hit", {
          class: "ult-hit",
          x1: source[0],
          y1: source[1],
          x2: target[0],
          y2: target[1],
        }),
        node("line", "trap.cord", {
          class: "ult-trap__cord",
          x1: a[0],
          y1: a[1],
          x2: b[0],
          y2: b[1],
          "stroke-width": Math.max(1, k),
          "stroke-dasharray": `${r2(3 * k)} ${r2(2 * k)}`,
          opacity: 0.7,
        }),
      );
      shared.push([
        "trap.cord",
        [
          { t: 80, o: 0 },
          { t: 120, o: 0.4 },
          { t: 240, o: 0.7 },
        ],
      ]);
      reduced.push([
        "trap.cord",
        [
          { t: 0, o: 0 },
          { t: 80, o: 0.7 },
        ],
      ]);
    }
  }
  const corners = Array.from({ length: 6 }, (_, i) => polar([0, 0], R, i * 60 * DEG));
  let snare = poly(corners);
  for (const corner of corners) {
    snare += ` M ${pt(corner)} L ${pt(add(corner, mul(norm(sub([0, 0], corner)), 3 * k)))}`;
  }
  const over = [
    node("g", "trap.snareAt", { transform: frame(target) }, [
      node("circle", "trap.targetHit", { class: "ult-hit-area", r: R }),
      node("g", "trap.snare", { opacity: 0 }, [
        node("path", "trap.snarePath", {
          class: "ult-trap__snare",
          d: snare,
          "stroke-width": Math.max(1, 1.5 * k),
        }),
      ]),
    ]),
  ];
  shared.push([
    "trap.snare",
    [
      { t: 80, o: 0, x: "scale(1.4)" },
      { t: 120, o: 0.8, x: "scale(1.36)", e: "ease-in" },
      { t: 240, o: 0.8, x: "scale(1)" },
      { t: 241, o: 1, x: "scale(1)" },
      { t: 300, o: 1, x: "scale(0.92)" },
      { t: 520, o: 0, x: "scale(0.92)" },
    ],
  ]);
  reduced.push([
    "trap.snare",
    [
      { t: 0, o: 0 },
      { t: 80, o: 0.8 },
      { t: 140, o: 0.8 },
      { t: 220, o: 0 },
    ],
  ]);
  return {
    band: Object.freeze(band),
    over: Object.freeze(over),
    cues: Object.freeze({
      glide: cues(shared),
      plain: cues(shared),
      reduced: cues(reduced),
    }),
  };
}

/**
 * The Burst drawing: eight slivers folding into the Mage, then the star flare.
 *
 * @param {JsonRecord} event
 */
function burstDrawing(event) {
  const caster = vec(event.source);
  if (caster === null) {
    return null;
  }
  const R = radiusOf(event.sourceRadius, DEFAULT_BODY_RADIUS_PX);
  const k = R / 23;
  let bright = "";
  let dim = "";
  let heads = "";
  for (let i = 0; i < 8; i += 1) {
    const angle = (22.5 + 45 * i - 90) * DEG;
    const across = /** @type {Vec} */ ([-Math.sin(angle), Math.cos(angle)]);
    const head = polar([0, 0], 1.2 * R, angle);
    const tail = polar([0, 0], 1.55 * R, angle);
    const mid = lerp(head, tail, 0.45);
    const hw = 0.9 * k;
    bright += `${poly([add(head, mul(across, hw)), add(mid, mul(across, 0.55 * hw)), sub(mid, mul(across, 0.55 * hw)), sub(head, mul(across, hw))])} `;
    dim += `${poly([add(mid, mul(across, 0.55 * hw)), tail, sub(mid, mul(across, 0.55 * hw))])} `;
    heads += `${circlePath(head, 1.15 * k)} `;
  }
  const star = [];
  for (let i = 0; i < 16; i += 1) {
    const reach = (i % 4 === 0 ? 0.55 : i % 2 === 0 ? 0.36 : 0.17) * R;
    star.push(polar([0, 0], reach, (i * 22.5 - 90) * DEG));
  }
  const over = [
    node("g", "burst.at", { transform: frame(caster) }, [
      node("circle", "burst.hit", { class: "ult-hit-area", r: R }),
      node("g", "burst.slivers", { opacity: 0 }, [
        node("path", "burst.tails", { class: "ult-burst__sliver-tail", d: dim.trim() }),
        node("path", "burst.bodies", { class: "ult-burst__sliver", d: bright.trim() }),
        node("path", "burst.heads", {
          class: "ult-burst__sliver-head",
          d: heads.trim(),
        }),
      ]),
      node("path", "burst.flare", {
        class: "ult-burst__flare",
        d: poly(star),
        opacity: 0,
      }),
    ]),
  ];
  const shared = cues([
    [
      "burst.slivers",
      [
        { t: 80, o: 0, x: "scale(1)" },
        { t: 160, o: 0.4, x: "scale(0.91)" },
        { t: 161, o: 0.85, x: "scale(0.91)" },
        { t: 280, o: 0.85, x: "scale(0.45)" },
        { t: 320, o: 0, x: "scale(0.21)" },
      ],
    ],
    [
      "burst.flare",
      [
        { t: 319, o: 0 },
        { t: 320, o: 0 },
        { t: 360, o: 1 },
        { t: 440, o: 0 },
      ],
    ],
  ]);
  return {
    band: Object.freeze([]),
    over: Object.freeze(over),
    cues: Object.freeze({
      glide: shared,
      plain: shared,
      reduced: cues([
        [
          "burst.flare",
          [
            { t: 40, o: 0 },
            { t: 90, o: 1 },
            { t: 140, o: 0 },
          ],
        ],
      ]),
    }),
  };
}

/**
 * True when both champions of a paired Ultimate were seen at END (the plan sets
 * the endpoint phase to "successor"); a missing phase (older plans, pictures
 * drawn outside the arena) counts as seen.
 *
 * @param {JsonRecord} event
 */
function seenAtBothEnds(event) {
  const ok = (/** @type {unknown} */ phase) =>
    phase === undefined || phase === "successor";
  return ok(event.sourceEndpointPhase) && ok(event.targetEndpointPhase);
}

/**
 * The Crippling Poison drawing: stiletto strike, tip flash, venom seep, resting
 * blade and tether.
 *
 * @param {JsonRecord} event
 * @param {boolean} softGlow
 */
function poisonDrawing(event, softGlow) {
  const rogue = vec(event.source);
  const victim = vec(event.target);
  if (rogue === null || victim === null || !seenAtBothEnds(event)) {
    return null;
  }
  const R = radiusOf(event.targetRadius, DEFAULT_BODY_RADIUS_PX);
  const rS = radiusOf(event.sourceRadius, R);
  const unitPx = radiusOf(event.unitPx, R / 0.5);
  const u = norm(sub(rogue, victim));
  const angle = Math.atan2(u[1], u[0]);
  const shift = 0.62 - 0.28;
  /** @param {number} v Distance from the victim's centre in R, before the shift. */
  const X = (v) => (v + shift) * R;
  const bw = 0.11 * R;
  const full = R >= 14;
  /**
   * A capsule between two local points as path data. Pure.
   *
   * @param {Vec} p @param {Vec} q @param {number} width
   */
  const capsule = (p, q, width) => {
    const half = Math.max(0.05, width / 2);
    const along = norm(sub(q, p));
    const n = /** @type {Vec} */ ([-along[1] * half, along[0] * half]);
    return `M ${pt(add(p, n))} L ${pt(add(q, n))} A ${r2(half)} ${r2(half)} 0 0 0 ${pt(sub(q, n))} L ${pt(sub(p, n))} A ${r2(half)} ${r2(half)} 0 0 0 ${pt(add(p, n))} Z`;
  };
  const tri = poly([
    [X(1.05), -bw],
    [X(0.28), 0],
    [X(1.05), bw],
  ]);
  const guard = capsule([X(1.09), -0.3 * R], [X(1.09), 0.3 * R], Math.max(2, 0.09 * R));
  const pommel = full ? circlePath([X(1.42), 0], Math.max(1.8, 0.09 * R)) : "";
  const gripWidth = Math.max(2, 0.1 * R);
  const deep =
    `${tri} ${full ? capsule([X(1.13), 0], [X(1.34), 0], gripWidth - 2) : ""}`.trim();
  const silver = `${tri} ${guard} ${pommel}`.trim();
  const ink =
    `${silver} ${full ? capsule([X(1.13), 0], [X(1.34), 0], gripWidth) : ""}`.trim();
  const pommelEnd = (1.51 + shift) * R;
  const startD = len(sub(rogue, victim)) - rS;
  const gapUnits = (len(sub(rogue, victim)) - rS - R) / unitPx;
  const tether = gapUnits >= 0.3 && startD - pommelEnd > 2;
  const wound = mul(u, R);
  const woundClock = Math.atan2(u[0], -u[1]);
  let seep = "";
  for (const sign of [-1, 1]) {
    const outer = [];
    const inner = [];
    for (let i = 0; i <= 10; i += 1) {
      const q = i / 10;
      const at = woundClock + sign * 50 * DEG * q;
      outer.push(clock([0, 0], 1.17 * R, at));
      inner.push(clock([0, 0], 1.17 * R - 0.12 * R * (1 - q), at));
    }
    seep += `M ${outer.map(pt).join(" L ")} L ${inner.reverse().map(pt).join(" L ")} Z `;
  }
  const band = [
    node("line", "poison.hit", {
      class: "ult-hit",
      x1: rogue[0],
      y1: rogue[1],
      x2: victim[0],
      y2: victim[1],
    }),
    node("g", "poison.restFrame", { transform: frame(victim, angle) }, [
      node("g", "poison.rest", { opacity: 0.7 }, [
        node("path", "poison.restDeep", { class: "ult-poison__deep", d: deep }),
        node("path", "poison.restSilver", { class: "ult-poison__silver", d: silver }),
      ]),
      tether
        ? node("line", "poison.tether", {
            class: "ult-poison__tether",
            x1: pommelEnd,
            y1: 0,
            x2: startD,
            y2: 0,
            opacity: 0.3,
          })
        : null,
    ]),
  ];
  const over = [
    node("g", "poison.strikeFrame", { transform: frame(victim, angle) }, [
      node("g", "poison.strike", { opacity: 0 }, [
        node("path", "poison.ink", { class: "ult-poison__ink", d: ink }),
        softGlow
          ? node("path", "poison.glow", {
              class: "ult-poison__glow ult-soft-glow",
              d: tri,
              opacity: 0.18,
            })
          : null,
        node("path", "poison.deep", { class: "ult-poison__deep", d: deep }),
        node("path", "poison.silver", { class: "ult-poison__silver", d: silver }),
        node("line", "poison.line", {
          class: "ult-poison__line",
          x1: X(1.0),
          y1: 0,
          x2: X(0.48),
          y2: 0,
        }),
      ]),
    ]),
    node("g", "poison.at", { transform: frame(victim) }, [
      node("circle", "poison.targetHit", { class: "ult-hit-area", r: R }),
      node("circle", "poison.flash", {
        class: "ult-poison__flash",
        cx: wound[0],
        cy: wound[1],
        r: Math.max(2.2, 0.13 * R),
        opacity: 0,
      }),
      node("g", "poison.seep", { opacity: 0 }, [
        node("path", "poison.seepInk", {
          class: "ult-poison__seep-ink",
          d: seep.trim(),
          opacity: 0.8,
        }),
        node("path", "poison.seepJade", { class: "ult-poison__seep", d: seep.trim() }),
      ]),
    ]),
  ];
  const drawn = `translateX(${r2(0.5 * R)}px)`;
  const shared = cues([
    [
      "poison.strike",
      [
        { t: 160, o: 0, x: drawn },
        { t: 190, o: 0.4, x: drawn },
        { t: 240, o: 0.4, x: drawn, e: "cubic-bezier(.55,0,1,.45)" },
        { t: 320, o: 1, x: "translateX(0px)" },
        { t: 380, o: 1, x: "translateX(0px)" },
        { t: 460, o: 0, x: "translateX(0px)" },
      ],
    ],
    [
      "poison.flash",
      [
        { t: 320, o: 0 },
        { t: 342, o: 1 },
        { t: 380, o: 0 },
      ],
    ],
    [
      "poison.seep",
      [
        { t: 360, o: 0 },
        { t: 440, o: 0.9 },
        { t: 560, o: 0.9 },
        { t: 640, o: 0 },
      ],
    ],
    [
      "poison.rest",
      [
        { t: 380, o: 0 },
        { t: 460, o: 0.7 },
      ],
    ],
    tether
      ? [
          "poison.tether",
          [
            { t: 160, o: 0 },
            { t: 240, o: 0.3 },
          ],
        ]
      : null,
  ]);
  return {
    band: Object.freeze(band),
    over: Object.freeze(over),
    cues: Object.freeze({
      glide: shared,
      plain: shared,
      reduced: cues([
        [
          "poison.strike",
          [
            { t: 40, o: 0 },
            { t: 100, o: 1 },
            { t: 160, o: 1 },
            { t: 220, o: 0 },
          ],
        ],
        [
          "poison.flash",
          [
            { t: 100, o: 0 },
            { t: 130, o: 0.8 },
            { t: 160, o: 0 },
          ],
        ],
        [
          "poison.seep",
          [
            { t: 100, o: 0 },
            { t: 160, o: 0.9 },
            { t: 220, o: 0 },
          ],
        ],
        [
          "poison.rest",
          [
            { t: 160, o: 0 },
            { t: 220, o: 0.7 },
          ],
        ],
        tether
          ? [
              "poison.tether",
              [
                { t: 40, o: 0 },
                { t: 100, o: 0.3 },
              ],
            ]
          : null,
      ]),
    }),
    tether,
  };
}

/**
 * The Holy Word: Salvation drawing: telegraph and syllable rings, the golden
 * Gilded Bolt, flood, rays and the echo ring.
 *
 * event is one admitted Salvation activation in CSS pixels (source is the
 * Priest, target the ally, both at END). The Strike Band gets the bolt from the
 * Priest's rim to the ally's rim: a group "salv.core" (70% in a still) holding
 * a Dawnlight edge, a gold body and a Dawnlight core, which grows out from the
 * Priest, plus, when softGlow is true, a "salv.bright" group with a wide gold
 * glow that shows only while playing. A self-cast, or an ally whose rim is
 * within 1 px of the Priest's, has no bolt. The echo ring, the three rings,
 * the flood and the rays at the ally do not depend on the bolt. Return {band,
 * over, cues, selfCast}, or null when the Priest's point is missing, or when
 * either endpoint phase is given and is not "successor" (that champion was not
 * seen at END; a null phase counts as not seen, an absent one as seen). A
 * missing ally point draws as a self-cast at the Priest. Pure.
 *
 * @param {JsonRecord} event
 * @param {boolean} softGlow
 */
function salvationDrawing(event, softGlow) {
  const priest = vec(event.source);
  const ally = vec(event.target) ?? priest;
  if (priest === null || ally === null || !seenAtBothEnds(event)) {
    return null;
  }
  const selfCast =
    vec(event.target) === null ||
    (event.sourcePresentationKey != null &&
      event.sourcePresentationKey === event.targetPresentationKey) ||
    len(sub(ally, priest)) < 1;
  const R = radiusOf(event.targetRadius, DEFAULT_BODY_RADIUS_PX);
  const rS = radiusOf(event.sourceRadius, R);
  /** @type {UltNode[]} */
  const band = [];
  /** @type {Array<[string, UltFrame[]] | null>} */
  const motion = [];
  /** @type {Array<[string, UltFrame[]] | null>} */
  const reduced = [];
  const u = selfCast ? /** @type {Vec} */ ([0, -1]) : norm(sub(ally, priest));
  if (!selfCast) {
    const from = add(priest, mul(u, rS));
    const to = sub(ally, mul(u, R));
    const length = len(sub(to, from));
    if (length > 1) {
      band.push(
        node("line", "salv.hit", {
          class: "ult-hit",
          x1: priest[0],
          y1: priest[1],
          x2: ally[0],
          y2: ally[1],
        }),
        node("g", "salv.frame", { transform: frame(from, Math.atan2(u[1], u[0])) }, [
          node("g", "salv.grow", {}, [
            // The golden Gilded Bolt: a Dawnlight edge, a gold body and a
            // Dawnlight core, kept at 70% in a still.
            node("g", "salv.core", { opacity: 0.7 }, [
              node("line", "salv.boltEdge", {
                class: "ult-salv__bolt-edge",
                x1: 0,
                y1: 0,
                x2: length,
                y2: 0,
              }),
              node("line", "salv.boltBody", {
                class: "ult-salv__bolt-body",
                x1: 0,
                y1: 0,
                x2: length,
                y2: 0,
              }),
              node("line", "salv.boltCore", {
                class: "ult-salv__bolt-core",
                x1: 0,
                y1: 0,
                x2: length,
                y2: 0,
              }),
            ]),
            // Its Soft Glow brightens the bolt while playing; with Soft Glow off
            // there is no bright group and no cue for it.
            softGlow
              ? node("g", "salv.bright", { opacity: 0 }, [
                  node("line", "salv.glow", {
                    class: "ult-salv__glow ult-soft-glow",
                    x1: 0,
                    y1: 0,
                    x2: length,
                    y2: 0,
                    opacity: 0.21,
                  }),
                ])
              : null,
          ]),
        ]),
      );
      motion.push(
        [
          "salv.grow",
          [
            { t: 160, o: 1, x: "scaleX(0.001)", e: "ease-out" },
            { t: 320, o: 1, x: "scaleX(1)" },
          ],
        ],
        [
          "salv.core",
          [
            { t: 159, o: 0 },
            { t: 160, o: 1 },
            { t: 640, o: 1 },
            { t: 960, o: 0.7 },
          ],
        ],
        softGlow
          ? [
              "salv.bright",
              [
                { t: 159, o: 0 },
                { t: 160, o: 1 },
                { t: 500, o: 1 },
                { t: 640, o: 0 },
              ],
            ]
          : null,
      );
      reduced.push(
        [
          "salv.core",
          [
            { t: 40, o: 0 },
            { t: 100, o: 1 },
            { t: 160, o: 1 },
            { t: 220, o: 0.7 },
          ],
        ],
        softGlow
          ? [
              "salv.bright",
              [
                { t: 40, o: 0 },
                { t: 100, o: 1 },
                { t: 160, o: 1 },
                { t: 220, o: 0 },
              ],
            ]
          : null,
      );
    }
  }
  const gap = 25 * DEG;
  const echoR = 1.15 * R;
  band.push(
    node("path", "salv.echo", {
      class: "ult-salv__echo",
      d: `M ${pt(clock(ally, echoR, gap))} A ${r2(echoR)} ${r2(echoR)} 0 1 1 ${pt(clock(ally, echoR, 2 * Math.PI - gap))}`,
      opacity: 0.35,
    }),
  );
  motion.push([
    "salv.echo",
    [
      { t: 520, o: 0 },
      { t: 640, o: 0.35 },
    ],
  ]);
  reduced.push([
    "salv.echo",
    [
      { t: 160, o: 0 },
      { t: 220, o: 0.35 },
    ],
  ]);
  const toPriest = selfCast ? /** @type {Vec} */ ([0, 0]) : mul(u, -R);
  let rays = "";
  for (let i = 0; i < 8; i += 1) {
    const at = (22.5 + 45 * i) * DEG;
    rays += `M ${pt(clock([0, 0], 1.08 * R, at))} L ${pt(clock([0, 0], 1.2 * R, at))} `;
  }
  const over = [
    node("g", "salv.at", { transform: frame(ally) }, [
      node("circle", "salv.targetHit", { class: "ult-hit-area", r: R }),
      ...[0, 1, 2].map((index) =>
        node("g", `salv.ring${index}`, { opacity: 0 }, [
          node("circle", `salv.ringCircle${index}`, { class: "ult-salv__ring", r: R }),
        ]),
      ),
      node("g", "salv.flood", { opacity: 0 }, [
        node("circle", "salv.floodOuter", {
          class: "ult-salv__flood",
          cx: toPriest[0],
          cy: toPriest[1],
          r: 0.56 * R,
        }),
        node("circle", "salv.floodInner", {
          class: "ult-salv__flood",
          cx: toPriest[0],
          cy: toPriest[1],
          r: 0.33 * R,
          opacity: 0.55,
        }),
      ]),
      node("g", "salv.rays", { opacity: 0 }, [
        softGlow
          ? node("path", "salv.rayGlow", {
              class: "ult-salv__ray-glow ult-soft-glow",
              d: rays.trim(),
              opacity: 0.18,
            })
          : null,
        node("path", "salv.rayLines", { class: "ult-salv__ray", d: rays.trim() }),
      ]),
    ]),
  ];
  motion.push(
    [
      "salv.ring0",
      [
        { t: 80, o: 0, x: "scale(1.6)" },
        { t: 160, o: 0.4, x: "scale(1.6)" },
        { t: 320, o: 0.4, x: "scale(1.6)", e: "ease-in" },
        { t: 362, o: 0.9, x: "scale(1.5)", e: "ease-in" },
        { t: 440, o: 0, x: "scale(1)" },
      ],
    ],
    ...[1, 2].map((index) => {
      const at = 320 + 40 * index;
      return /** @type {[string, UltFrame[]]} */ ([
        `salv.ring${index}`,
        [
          { t: at, o: 0, x: "scale(1.6)", e: "ease-in" },
          { t: at + 42, o: 0.9, x: "scale(1.5)", e: "ease-in" },
          { t: at + 120, o: 0, x: "scale(1)" },
        ],
      ]);
    }),
    [
      "salv.flood",
      [
        { t: 380, o: 0 },
        { t: 440, o: 0.48 },
        { t: 460, o: 0.48 },
        { t: 620, o: 0 },
      ],
    ],
    [
      "salv.rays",
      [
        { t: 380, o: 0 },
        { t: 420, o: 1 },
        { t: 540, o: 1 },
        { t: 620, o: 0 },
      ],
    ],
  );
  reduced.push(
    [
      "salv.ring0",
      [
        { t: 0, o: 0, x: "scale(1.6)" },
        { t: 60, o: 0.4, x: "scale(1.6)" },
        { t: 100, o: 0.4, x: "scale(1.6)" },
        { t: 160, o: 0, x: "scale(1.6)" },
      ],
    ],
    [
      "salv.ring1",
      [
        { t: 100, o: 0 },
        { t: 160, o: 0.9 },
        { t: 220, o: 0 },
      ],
    ],
    [
      "salv.flood",
      [
        { t: 100, o: 0 },
        { t: 160, o: 0.34 },
        { t: 220, o: 0 },
      ],
    ],
    [
      "salv.rays",
      [
        { t: 100, o: 0 },
        { t: 160, o: 1 },
        { t: 220, o: 0 },
      ],
    ],
  );
  const shared = cues(motion);
  return {
    band: Object.freeze(band),
    over: Object.freeze(over),
    cues: Object.freeze({ glide: shared, plain: shared, reduced: cues(reduced) }),
    selfCast,
  };
}

/**
 * Convert a cue time to real tick ms through the plan's beat schedule. Pure.
 *
 * time is blow ms ("part" clock), real ms ("real" clock) or a Charge rush
 * marker; beats is the plan's BeatSchedule or null. With Body Glide on, blow
 * time from 80 ms is mapped piecewise linearly through the schedule's own
 * published knots (80 -> strike start, 240 -> impact start, 420 -> settle start,
 * 900 -> choreography end, then one to one), so Longer Tick and Body Glide keep
 * one owner. With Body Glide off, or with no schedule, blow time is real time.
 * The rush markers read windows.chargeRush (default 0-110 ms).
 *
 * @param {UltFrame["t"]} time
 * @param {"part" | "real"} timeClock
 * @param {JsonRecord | null} beats
 * @returns {number}
 */
function realTime(time, timeClock, beats) {
  if (time === "rushStart" || time === "rushEnd") {
    const rush = Array.isArray(beats?.windows?.chargeRush)
      ? beats.windows.chargeRush
      : [0, 110];
    return Number(time === "rushStart" ? rush[0] : rush[1]);
  }
  if (timeClock === "real" || !beats?.glide) {
    return time;
  }
  const windows = beats.windows ?? {};
  const knots = [
    [80, Number(windows.strike?.[0])],
    [240, Number(windows.impact?.[0])],
    [420, Number(windows.settle?.[0])],
    [900, Number(beats.choreographyEnd)],
  ];
  if (knots.some(([, real]) => !Number.isFinite(real))) {
    return time;
  }
  if (time <= knots[0][0]) return knots[0][1] - (knots[0][0] - time);
  for (let index = 1; index < knots.length; index += 1) {
    const [partStart, realStart] = knots[index - 1];
    const [partEnd, realEnd] = knots[index];
    if (time <= partEnd) {
      return (
        realStart + ((time - partStart) * (realEnd - realStart)) / (partEnd - partStart)
      );
    }
  }
  return knots[3][1] + (time - knots[3][0]);
}

/**
 * Turn a record's cues into browser animation descriptions in real ms.
 *
 * record is one record from records(); beats is the plan's BeatSchedule (or
 * null: Body Glide off, real time equals blow time); motionMode is "normal",
 * "reduced" or "off". Return a frozen list of UltTrack (key, keyframes,
 * options {delay, duration, easing "linear", fill "backwards"}): the glide cues
 * when beats.glide is true, the plain cues otherwise, the reduced cues in
 * reduced motion (their 220 ms clock is scaled to the plan's reduced
 * choreography length, choreographyEnd x reducedScale), and nothing for "off"
 * or an unknown mode. Before its first frame a node shows the first keyframe;
 * after its last frame the DOM's END value shows again, unless the cue ended at
 * opacity 0 on a transient event (the choreographer then keeps it hidden until
 * the tick is replaced). Cues with no positive
 * duration are left out. Pure.
 *
 * @param {LookRecord} record
 * @param {JsonRecord | null | undefined} beats
 * @param {string} motionMode
 * @returns {ReadonlyArray<UltTrack>}
 */
export function animationTracks(record, beats, motionMode) {
  const schedule = beats ?? null;
  /** @type {ReadonlyArray<UltCue> | undefined} */
  const list =
    motionMode === "reduced"
      ? record.cues?.reduced
      : motionMode === "normal"
        ? schedule?.glide
          ? record.cues?.glide
          : record.cues?.plain
        : undefined;
  if (!list) {
    return Object.freeze([]);
  }
  const reducedEnd =
    Number.isFinite(schedule?.choreographyEnd) &&
    Number.isFinite(schedule?.reducedScale)
      ? Number(schedule?.choreographyEnd) * Number(schedule?.reducedScale)
      : 220;
  /** @type {UltTrack[]} */
  const tracks = [];
  for (const cue of list) {
    const times = cue.frames.map((each) =>
      motionMode === "reduced"
        ? (Number(each.t) * reducedEnd) / 220
        : realTime(each.t, each.clock ?? "part", schedule),
    );
    const first = Math.min(...times);
    const last = Math.max(...times);
    const duration = last - first;
    if (!(duration > 0)) continue;
    const keyframes = cue.frames.map((each, index) => {
      /** @type {Keyframe} */
      const keyframe = {
        offset: r2(((times[index] - first) / duration) * 1000) / 1000,
        opacity: each.o,
      };
      if (each.x !== undefined) keyframe.transform = each.x;
      if (each.e !== undefined) keyframe.easing = each.e;
      return keyframe;
    });
    keyframes[0].offset = 0;
    keyframes[keyframes.length - 1].offset = 1;
    tracks.push(
      Object.freeze({
        key: cue.key,
        keyframes,
        options: Object.freeze({
          delay: r2(first),
          duration: r2(duration),
          easing: "linear",
          fill: /** @type {FillMode} */ ("backwards"),
        }),
      }),
    );
  }
  return Object.freeze(tracks);
}

/**
 * Create the DOM nodes of an UltNode tree.
 *
 * doc creates SVG; tree is one UltNode; byKey receives every created element
 * by its data-ult-key. Return the root element. DOM changes only.
 *
 * @param {Document} doc
 * @param {UltNode} tree
 * @param {Map<string, Element>} byKey
 * @returns {SVGElement}
 */
function createTree(doc, tree, byKey) {
  const element = effectNode(doc, tree.tag, tree.attrs);
  byKey.set(String(tree.attrs["data-ult-key"]), element);
  if (tree.text !== undefined) {
    element.textContent = tree.text;
  }
  for (const child of tree.children ?? []) {
    element.append(createTree(doc, child, byKey));
  }
  return element;
}

/**
 * Collect every element under root that carries data-ult-key, by key.
 *
 * @param {Element} root
 * @param {Map<string, Element>} byKey
 * @returns {Map<string, Element>}
 */
function collectKeys(root, byKey = new Map()) {
  const key = root.getAttribute?.("data-ult-key");
  if (key) byKey.set(key, root);
  for (const child of Array.from(root.children ?? [])) {
    collectKeys(child, byKey);
  }
  return byKey;
}

/**
 * List the keys of an UltNode tree in order.
 *
 * @param {ReadonlyArray<UltNode>} trees
 * @param {string[]} [into]
 * @returns {string[]}
 */
function treeKeys(trees, into = []) {
  for (const tree of trees) {
    into.push(String(tree.attrs["data-ult-key"]));
    treeKeys(tree.children ?? [], into);
  }
  return into;
}

/**
 * Draw Ultimate records into the painter's groups.
 *
 * records come from records(); doc creates nodes; mount (optional) is the
 * painter's EventMount. Without mount, new detached settled groups are made
 * (no animation; used by the Visual Key). For each record a `g.ult` root with
 * data-ult-variant goes into mount.group (above bodies) and a `g.ult-band` root
 * into mount.underlay, the Strike Band (or at the start of mount.group when the
 * event has no route group). The nodes are drawn at their END (settled) look.
 * Unless mount.options.settled is true or motionMode is "off", one animation
 * description per cue (animationTracks with mount.plan.beats) is pushed onto
 * mount.animationSpecs with the id mbg:<epoch>:<eventId>:ult.<key>. The
 * "plain" variant draws the shared damage or healing mark at the target. Layout
 * metadata from the plan's impact allocation is copied onto the root. A "none"
 * record draws nothing (its groups stay empty). Return
 * the groups drawn into ([group, connector] plus the route group when present);
 * [] for no records (doc is then not used). Export-safe SVG only.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @param {EventMount} [mount]
 * @returns {SVGElement[]}
 */
export function build(records, doc, mount) {
  const first = records[0];
  if (first === undefined) {
    return [];
  }
  const target = mount ?? detachedEventMount(doc, first.event);
  for (const record of records) {
    if (record.variant === "none") continue;
    /** @type {Map<string, Element>} */
    const byKey = new Map();
    const root = effectNode(doc, "g", {
      class: `ult ult--${record.variant}`,
      "data-ult-root": "over",
      "data-ult-variant": record.variant,
    });
    for (const tree of record.over) {
      root.append(createTree(doc, tree, byKey));
    }
    if (record.variant === "plain") {
      byKey.get("plain.at")?.append(semanticImpactGlyph(doc, record.semantic));
    }
    assignLayoutPlacement(
      root,
      record.event.impactLayoutKey,
      record.event.impactBounds,
      record.event.impactDisposition,
      record.event.impactCueCollisionFree,
    );
    if (record.band.length > 0) {
      const band = effectNode(doc, "g", {
        class: `ult-band ult-band--${record.variant}`,
        "data-ult-root": "band",
      });
      for (const tree of record.band) {
        band.append(createTree(doc, tree, byKey));
      }
      if (target.underlay) {
        target.underlay.append(band);
      } else {
        target.group.append(band);
      }
    }
    target.group.append(root);
    if (target.options.settled || target.options.motionMode === "off") {
      continue;
    }
    for (const track of animationTracks(
      record,
      target.plan?.beats ?? null,
      target.options.motionMode,
    )) {
      const element = byKey.get(track.key);
      if (!element) continue;
      target.animationSpecs.push(
        Object.freeze({
          element,
          keyframes: track.keyframes,
          options: track.options,
          id: `mbg:${target.plan?.epochKey}:${record.event.eventId}:ult.${track.key}`,
        }),
      );
    }
  }
  return mountNodes(target);
}

/**
 * Move an Ultimate's nodes to its new geometry (reprojection).
 *
 * nodes are [group, connector, route group or nothing] as build returned;
 * records holds the event's new record. When the new drawing has the same node
 * keys, every node's attributes are rewritten in place, so running animations
 * keep their elements and progress. When the keys differ (for example the echo
 * appears at a new zoom), the look's own roots are rebuilt at END with no
 * animation. Nothing else in the groups is touched. Return undefined.
 *
 * @param {ReadonlyArray<Element | null>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  const record = records[0];
  const [group, , underlay = null] = nodes;
  if (record === undefined || !group) {
    return;
  }
  const roots = [group, underlay].flatMap((container) =>
    container
      ? Array.from(container.children ?? []).filter((child) =>
          child.hasAttribute?.("data-ult-root"),
        )
      : [],
  );
  /** @type {Map<string, Element>} */
  const existing = new Map();
  for (const root of roots) collectKeys(root, existing);
  const wanted = treeKeys([...record.over, ...record.band]);
  const sameShape =
    wanted.length === existing.size && wanted.every((key) => existing.has(key));
  if (sameShape) {
    /** @param {ReadonlyArray<UltNode>} trees */
    const apply = (trees) => {
      for (const tree of trees) {
        const element = existing.get(String(tree.attrs["data-ult-key"]));
        if (element) setEffectAttributes(element, tree.attrs);
        apply(tree.children ?? []);
      }
    };
    apply(record.over);
    apply(record.band);
    return;
  }
  const doc = group.ownerDocument;
  for (const root of roots) root.remove();
  build(records, doc, {
    group: /** @type {SVGElement} */ (group),
    connector: /** @type {SVGElement} */ (nodes[1] ?? group),
    underlay: /** @type {SVGElement | null} */ (underlay),
    plan: { epochKey: "reprojected", phases: {} },
    options: { motionMode: "off", settled: true },
    animationSpecs: [],
  });
}

/**
 * Return the tooltip for an Ultimate record.
 *
 * With ability paint on, the activation explanation (title such as "Charge
 * (Warrior Ultimate)"); with it off, the shared event explanation of the
 * meaning mark. Both read only the plan event, so in Agent POV they name only
 * what the plan admitted.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  const view = paintAwareExplanationEvent(record.event);
  return paintPartEnabled(record.event, "ability")
    ? explainActivation(view)
    : explainChoreographyEvent(view);
}
