/**
 * @file The Fog And Vision look: in Agent POV, the Sight Reach veil, the
 * recipient rim, the Last Seen ring and the Spotted Glint. Oracle View draws
 * nothing.
 *
 * What it draws (store part "Fog And Vision", plan P18 and P19):
 * - Sight Reach veil (`pov_sight_veil`): the floor OUTSIDE the sight reach
 *   circles of the frame's living sensors is darkened with Ink at 45%. The lit
 *   floor gets a faint 4% Lift and a thin Reach Edge where a circle is not
 *   inside another one. In live play (input.live) the numbers are the
 *   BattleClient vision's: Ink at 52%, a 5% Lift, the own edge at 45%
 *   (FOG_LOOK_LIVE). The vision also darkens the sight shadows that stones
 *   cast; this look never tests stone (amendment A17 keeps line of sight out
 *   of the browser), so in live play it paints the host's merged team sight
 *   (`scene.team_sight`, stone shadows included) when the scene carries it,
 *   and the reach circles otherwise (the Replay Viewer). The veil is made of exact pieces cut where the shape
 *   changes, with straight floor sides and SVG arc circle sides, so its edge
 *   is smooth with no row steps (no masks, clip paths, gradients or filters).
 *   It sits in the `fog` layer, under the stones.
 * - Recipient Rim (`pov_recipient_rim`): a thin silver ring at 0.98R on the
 *   champion whose eyes the view uses (the scene row with relation "self").
 * - Spotted Glint (`sight_change_marks`, play only): a ring that shrinks onto
 *   an enemy that came into sight this tick. Its END look is invisible.
 * - Last Seen ring (`sight_change_marks`): a dotted ring at 0.93R with a hollow
 *   class emblem where a champion was at the start of a tick in which it left
 *   sight. It is drawn from the plan's derived `last_seen` rows (one per body
 *   whose trajectory has a start and no successor), kept in the paused still.
 *   As the store draws it: the ring (1.5 px, dots 0.1 5.4) is in the lost
 *   champion's team light at 45% (`data-team` from the row's teamId; Moon
 *   Silver when the row names no team), and the emblem is the store's hollow
 *   ghost emblem (GHOST_EMBLEMS, line art copied from the store, never the
 *   filled sigil), 0.6R wide, a 1 px Moon Silver line at 60%.
 *
 * Who calls it: `scene.js` calls records/build for the `fog` layer (the
 * default part "under") and, once wired, for the part "over" (rim and glint,
 * drawn above the bodies). `choreography-painter.js` calls records/build with
 * a derived `last_seen` row and an EventMount. The choreographer calls
 * `sightAnimationSpecs` for the beat-timed fades of this look's nodes.
 *
 * Timing: the DOM always holds the END state. A veil "state" (the light at one
 * point of the tick) is a node whose END opacity is 1 only for the last state.
 * `sightAnimationSpecs` returns Web Animations overrides (fill "backwards") that
 * show earlier states on named beats: the glide (Stride), each sensor's Fall
 * beat and each Return beat. Nothing here starts a timer or a CSS animation.
 *
 * Boundary (Agent POV): records read only the audience's own arena scene, the
 * frame's authorized sensor list (`current_endpoint.parts.authorized_sensor_sources`),
 * the audience's own visual-event trajectories and the audience's own beat
 * schedule. They never read a researcher view or a match summary. The veil is
 * a reach display, NOT a line-of-sight shape: stone is never tested here
 * (amendment A17 forbids rebuilding line of sight in the browser). Only living
 * sensors give light; dead ones give none. Hidden champions are never drawn:
 * the rim, glint and Last Seen ring use only bodies and positions this
 * audience saw.
 */
import { CLASS_IDENTITIES } from "../agent-identity.js";
import { createSemanticDescriptor, registerTooltipOwner } from "../tooltip.js";
import {
  asArray,
  finiteNumber,
  isRecord,
  mountNodes,
  paintParts,
  svgNode,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").LookInput} LookInput
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {import("./look-support.js").EventMount} EventMount
 * @typedef {import("./look-support.js").AnimationSpec} AnimationSpec
 * @typedef {Record<string, any>} JsonRecord
 * @typedef {Readonly<{
 *   presentationKey: string,
 *   publicAgentId: string,
 *   x: number,
 *   y: number,
 *   r: number,
 *   own: boolean,
 * }>} SightCircle
 *   One living sensor's sight reach in world units: centre (x, y), radius r
 *   (its `observation_radius`), and own true for the recipient's own circle.
 * @typedef {Readonly<{kind: "stride" | "fall" | "return", key: string | null}>} SightBeat
 *   A named beat that changes the veil: "stride" (the glide at the start of a
 *   tick, key null), "fall" or "return" (keyed by the champion's presentation
 *   key, as in the plan's beat schedule).
 */

/**
 * The fixed numbers of this look, from the store (Fog And Vision, Stone
 * Shadows) and plan P18. Lengths ending in Factor multiply the body radius R in
 * CSS pixels; Ms values are milliseconds of 1x tick time; opacities are 0..1.
 */
/**
 * The Night Veil's opacities in live play (the BattleClient and the DevClient
 * combat screen), as the approved vision page draws them (`veil` in
 * vision_template.html): the unseen floor in Ink at 52%, a 5% Lift on the lit
 * floor, the focused champion's Reach Edge at 45% and its teammates' dashed
 * edges at 30%. The Replay Viewer keeps FOG_LOOK's store numbers.
 */
export const FOG_LOOK_LIVE = Object.freeze({
  veilOpacity: 0.52,
  liftOpacity: 0.05,
  ownEdgeOpacity: 0.45,
  alliedEdgeOpacity: 0.3,
});

export const FOG_LOOK = Object.freeze({
  veilOpacity: 0.45,
  liftOpacity: 0.04,
  ownEdgeOpacity: 0.4,
  alliedEdgeOpacity: 0.3,
  /** Largest angle between two Reach Edge points, in degrees. */
  edgeStepDegrees: 2,
  rimRadiusFactor: 0.98,
  rimOpacity: 0.9,
  /** The rim steps aside when R is under this many CSS pixels. */
  rimMinRadius: 10,
  lastSeenRadiusFactor: 0.93,
  lastSeenOpacity: 0.45,
  lastSeenEmblemFactor: 0.6,
  lastSeenEmblemOpacity: 0.6,
  /** Under this R (CSS pixels) the Last Seen mark is the ring only. */
  lastSeenEmblemMinRadius: 14,
  /** R used when a caller gives no transform (the store's 23 px token). */
  defaultBodyRadiusPx: 23,
  /** A body with no radius uses this many world units, as champions.js does. */
  defaultBodyRadiusWorld: 0.5,
  glintFromFactor: 1.35,
  glintPeakOpacity: 0.5,
  fallFadeInMs: 200,
  returnFadeInMs: 150,
  reducedFadeInMs: 110,
  reducedMarkFadeMs: 50,
  reducedGlintMs: 110,
  stepFadeMs: 200,
});

/**
 * Node and animation allowance of one derived `last_seen` row (the plan's
 * budget unit for this look): a group, the dotted ring and the ghost emblem (an
 * SVG of at most 2 shapes; the allowance still covers the 4 shapes the filled
 * sigil once drew, measured on the map 47 replay), plus margin; one fade-in;
 * all kept in the paused still.
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 8,
  animationsPerRecord: 1,
  persistentNodesPerRecord: 8,
});

/**
 * Allocator footprint in CSS pixels. The Last Seen ring needs a square of
 * twice its radius at the store's R (23 px): 2 x 0.93 x 23, rounded up.
 */
export const FOOTPRINT = Object.freeze({
  lastSeen: Object.freeze({ width: 43, height: 43 }),
});

/**
 * The switchable parts of this look. The `last_seen` tag is exactly the plan's
 * DERIVED_EVENT_PAINT_PARTS tag.
 *
 * @type {ReadonlyArray<import("./look-support.js").PaintPartRegistration>}
 */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "sight_veil" }, "pov_sight_veil"],
  [{ surface: "durable", kind: "recipient_rim" }, "pov_recipient_rim"],
  [{ surface: "durable", kind: "spotted_glint" }, "sight_change_marks"],
  [{ surface: "transient", kind: "last_seen" }, "sight_change_marks"],
]);

/**
 * Return whether a visual filter is on. A missing state or a missing switch
 * counts as on (all three fog switches default to on).
 *
 * @param {LookInput} input
 * @param {string} id
 * @returns {boolean}
 */
function switchOn(input, id) {
  const filters = input.filters;
  return !(isRecord(filters) && filters[id] === false);
}

/**
 * Return a world point [x, y] from a position, or null when it is not two
 * finite numbers.
 *
 * @param {unknown} position
 * @returns {readonly [number, number] | null}
 */
function worldPoint(position) {
  if (!Array.isArray(position) || position.length < 2) return null;
  const [x, y] = position;
  return typeof x === "number" &&
    Number.isFinite(x) &&
    typeof y === "number" &&
    Number.isFinite(y)
    ? [x, y]
    : null;
}

/**
 * Round a CSS-pixel value to 0.01 px for short, stable path text.
 *
 * @param {number} value
 * @returns {number}
 */
function px(value) {
  return Math.round(value * 100) / 100;
}

/**
 * @typedef {Readonly<{x: number}> | Readonly<{cx: number, cy: number, r: number, side: -1 | 1}>} ReachEdge
 *   One side of a veil piece, in world units: a straight vertical side at
 *   world x (the floor edge or the edge of the whole floor), or the left
 *   (side -1) or right (side 1) half of a reach circle centred at (cx, cy)
 *   with radius r.
 * @typedef {Readonly<{y0: number, y1: number, left: ReachEdge, right: ReachEdge}>} ReachPiece
 *   One dark or lit piece of the floor between rows y0 < y1 (world units). At
 *   every y in [y0, y1] the piece runs from its left side to its right side.
 */

/**
 * Return the world x of a piece side at world height y. A circle side outside
 * its circle's height range gives the circle's centre x. Pure.
 *
 * @param {ReachEdge} edge
 * @param {number} y
 * @returns {number}
 */
function edgeX(edge, y) {
  if ("x" in edge) return edge.x;
  const dy = y - edge.cy;
  return edge.cx + edge.side * Math.sqrt(Math.max(0, edge.r * edge.r - dy * dy));
}

/**
 * Return the sorted world heights where the veil's shape changes: the floor's
 * top and bottom, each circle's top and bottom, the points where two circles
 * cross, and the points where a circle crosses the floor's left or right edge.
 * Between two neighbouring heights every lit run keeps the same left and right
 * owner. Only heights inside [0, height] are kept; near-equal heights (within
 * 1e-9) count once. Pure.
 *
 * @param {ReadonlyArray<SightCircle>} circles
 * @param {number} width
 * @param {number} height
 * @returns {number[]}
 */
function reachBreaks(circles, width, height) {
  const ys = [0, height];
  circles.forEach((a, index) => {
    ys.push(a.y - a.r, a.y + a.r);
    for (const wall of [0, width]) {
      const dx = wall - a.x;
      if (Math.abs(dx) < a.r) {
        const half = Math.sqrt(a.r * a.r - dx * dx);
        ys.push(a.y - half, a.y + half);
      }
    }
    for (const b of circles.slice(index + 1)) {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.hypot(dx, dy);
      if (d === 0 || d > a.r + b.r || d < Math.abs(a.r - b.r)) continue;
      const along = (a.r * a.r - b.r * b.r + d * d) / (2 * d);
      const across = Math.sqrt(Math.max(0, a.r * a.r - along * along));
      const my = a.y + (along * dy) / d;
      ys.push(my + (across * dx) / d, my - (across * dx) / d);
    }
  });
  const sorted = ys.filter((y) => y >= 0 && y <= height).sort((p, q) => p - q);
  /** @type {number[]} */
  const unique = [];
  for (const y of sorted) {
    if (unique.length === 0 || y - unique[unique.length - 1] > 1e-9) unique.push(y);
  }
  return unique;
}

/**
 * Split the floor into the exact dark and lit pieces of the Sight Reach veil,
 * in world units.
 *
 * circles are the living sensors' reach circles (world units); width and
 * height are the floor size in world units (the floor is [0, width] x
 * [0, height]). The floor is cut into bands at the heights where the shape
 * changes (see reachBreaks); inside a band each piece is bounded by straight
 * sides or by circle halves, so the lit floor is exactly the union of the
 * circles clipped to the floor, with smooth round edges and no row steps.
 * Return {dark, lit}: arrays of ReachPiece, left to right inside each band,
 * bands from y 0 upward. The pieces never overlap and together cover the whole
 * floor. No circles gives one dark piece over the whole floor ("Eyes Closed").
 * A floor of zero or negative size gives two empty arrays. Pure; this is a
 * reach display only and never tests stone.
 *
 * @param {ReadonlyArray<SightCircle>} circles
 * @param {number} width
 * @param {number} height
 * @returns {{dark: ReachPiece[], lit: ReachPiece[]}}
 */
export function sightReachRuns(circles, width, height) {
  /** @type {ReachPiece[]} */
  const dark = [];
  /** @type {ReachPiece[]} */
  const lit = [];
  if (!(width > 0 && height > 0)) return { dark, lit };
  /** @type {ReachEdge} */
  const floorLeft = { x: 0 };
  /** @type {ReachEdge} */
  const floorRight = { x: width };
  const breaks = reachBreaks(circles, width, height);
  for (let index = 0; index + 1 < breaks.length; index += 1) {
    const y0 = breaks[index];
    const y1 = breaks[index + 1];
    const middle = (y0 + y1) / 2;
    /** @type {Array<{from: number, to: number, left: ReachEdge, right: ReachEdge}>} */
    const spans = [];
    for (const circle of circles) {
      const dy = middle - circle.y;
      if (Math.abs(dy) >= circle.r) continue;
      const half = Math.sqrt(circle.r * circle.r - dy * dy);
      const from = circle.x - half;
      const to = circle.x + half;
      if (to <= 0 || from >= width) continue;
      spans.push({
        from: Math.max(0, from),
        to: Math.min(width, to),
        left:
          from <= 0 ? floorLeft : { cx: circle.x, cy: circle.y, r: circle.r, side: -1 },
        right:
          to >= width
            ? floorRight
            : { cx: circle.x, cy: circle.y, r: circle.r, side: 1 },
      });
    }
    spans.sort((p, q) => p.from - q.from);
    /** @type {typeof spans} */
    const merged = [];
    for (const span of spans) {
      const last = merged[merged.length - 1];
      if (last && span.from <= last.to) {
        if (span.to > last.to) {
          last.to = span.to;
          last.right = span.right;
        }
      } else {
        merged.push({ ...span });
      }
    }
    /** @type {ReachEdge} */
    let cursor = floorLeft;
    let cursorX = 0;
    for (const span of merged) {
      if (span.from > cursorX) dark.push({ y0, y1, left: cursor, right: span.left });
      lit.push({ y0, y1, left: span.left, right: span.right });
      cursor = span.right;
      cursorX = span.to;
    }
    if (cursorX < width) dark.push({ y0, y1, left: cursor, right: floorRight });
  }
  return { dark, lit };
}

/**
 * Return whether world point (x, y) is lit by the pieces of sightReachRuns.
 *
 * runs is a sightReachRuns result. A point on a piece's edge, or within 1e-9
 * world units of it, counts as lit. Pure; used by tests and by callers that
 * check a body sits in the light.
 *
 * @param {{lit: ReadonlyArray<ReachPiece>}} runs
 * @param {number} x
 * @param {number} y
 * @returns {boolean}
 */
export function isLitByRuns(runs, x, y) {
  const slack = 1e-9;
  return runs.lit.some((piece) => {
    if (y < piece.y0 - slack || y > piece.y1 + slack) return false;
    const at = Math.min(Math.max(y, piece.y0), piece.y1);
    return x >= edgeX(piece.left, at) - slack && x <= edgeX(piece.right, at) + slack;
  });
}

/**
 * Project veil pieces into one path (CSS pixels): one closed subpath per
 * piece, straight sides as lines and circle sides as SVG arcs (no masks, clip
 * paths or filters); a piece with two straight sides is a plain
 * "M x y H x V y H x Z" rectangle. Every subpath runs the same way round (top-left, top-right,
 * bottom-right, bottom-left in world units), so pieces that touch fill without
 * hairline seams under the default nonzero fill rule. The arc radius uses
 * transform.worldLengthToScreen, so the projection must scale x and y alike.
 * Pure; returns "" for no pieces.
 *
 * @param {ReadonlyArray<ReachPiece>} pieces
 * @param {LookTransform} transform
 * @returns {string}
 */
function piecePath(pieces, transform) {
  /** @param {ReachEdge} edge @param {number} y */
  const screen = (edge, y) => transform.worldToScreen([edgeX(edge, y), y]);
  /**
   * @param {ReachEdge} edge
   * @param {number} from
   * @param {number} to
   */
  const side = (edge, from, to) => {
    const end = screen(edge, to);
    if ("x" in edge) return `L${px(end.x)} ${px(end.y)}`;
    const start = screen(edge, from);
    const mid = screen(edge, (from + to) / 2);
    const turn =
      (mid.x - start.x) * (end.y - mid.y) - (mid.y - start.y) * (end.x - mid.x);
    const radius = px(transform.worldLengthToScreen(edge.r));
    return `A${radius} ${radius} 0 0 ${turn > 0 ? 1 : 0} ${px(end.x)} ${px(end.y)}`;
  };
  const parts = [];
  for (const piece of pieces) {
    const start = screen(piece.left, piece.y0);
    const topRight = screen(piece.right, piece.y0);
    if ("x" in piece.left && "x" in piece.right) {
      // A plain rectangle keeps the short H/V form.
      const far = screen(piece.right, piece.y1);
      parts.push(
        `M${px(start.x)} ${px(start.y)}H${px(far.x)}V${px(far.y)}H${px(start.x)}Z`,
      );
      continue;
    }
    parts.push(
      `M${px(start.x)} ${px(start.y)}L${px(topRight.x)} ${px(topRight.y)}` +
        side(piece.right, piece.y0, piece.y1) +
        side({ x: edgeX(piece.left, piece.y1) }, piece.y1, piece.y1) +
        side(piece.left, piece.y1, piece.y0) +
        "Z",
    );
  }
  return parts.join("");
}

/**
 * Return the Reach Edge pieces of one circle as angle intervals (radians): the
 * parts of its rim that are on the floor and not strictly inside another
 * circle. An identical earlier circle covers a later one, so a shared rim is
 * drawn once. Pure.
 *
 * @param {ReadonlyArray<SightCircle>} circles
 * @param {number} index
 * @param {number} width
 * @param {number} height
 * @returns {Array<[number, number]>}
 */
function edgeIntervals(circles, index, width, height) {
  const circle = circles[index];
  const twoPi = Math.PI * 2;
  const cuts = [0, twoPi];
  /** @param {number} angle */
  const addCut = (angle) => {
    const wrapped = ((angle % twoPi) + twoPi) % twoPi;
    cuts.push(wrapped);
  };
  for (let other = 0; other < circles.length; other += 1) {
    if (other === index) continue;
    const peer = circles[other];
    const dx = peer.x - circle.x;
    const dy = peer.y - circle.y;
    const distance = Math.hypot(dx, dy);
    if (
      distance === 0 ||
      distance >= circle.r + peer.r ||
      distance <= Math.abs(circle.r - peer.r)
    ) {
      continue;
    }
    const base = Math.atan2(dy, dx);
    const spread = Math.acos(
      (circle.r * circle.r + distance * distance - peer.r * peer.r) /
        (2 * circle.r * distance),
    );
    addCut(base - spread);
    addCut(base + spread);
  }
  for (const [lineX, lineY] of [
    [0, null],
    [width, null],
    [null, 0],
    [null, height],
  ]) {
    if (lineX !== null && Math.abs(lineX - circle.x) < circle.r) {
      const angle = Math.acos((lineX - circle.x) / circle.r);
      addCut(angle);
      addCut(-angle);
    }
    if (lineY !== null && Math.abs(lineY - circle.y) < circle.r) {
      const angle = Math.asin((lineY - circle.y) / circle.r);
      addCut(angle);
      addCut(Math.PI - angle);
    }
  }
  cuts.sort((a, b) => a - b);
  /** @type {Array<[number, number]>} */
  const kept = [];
  for (let cut = 0; cut + 1 < cuts.length; cut += 1) {
    const start = cuts[cut];
    const end = cuts[cut + 1];
    if (end - start < 1e-9) continue;
    const middle = (start + end) / 2;
    const x = circle.x + circle.r * Math.cos(middle);
    const y = circle.y + circle.r * Math.sin(middle);
    const onFloor = x >= -1e-9 && x <= width + 1e-9 && y >= -1e-9 && y <= height + 1e-9;
    const covered = circles.some((peer, other) => {
      if (other === index) return false;
      const same = peer.x === circle.x && peer.y === circle.y && peer.r === circle.r;
      if (same) return other < index;
      return Math.hypot(x - peer.x, y - peer.y) < peer.r - 1e-9;
    });
    if (!onFloor || covered) continue;
    const last = kept[kept.length - 1];
    if (last && Math.abs(last[1] - start) < 1e-9) {
      last[1] = end;
    } else {
      kept.push([start, end]);
    }
  }
  // Join a piece that ends at 2 pi with one that starts at 0.
  if (
    kept.length > 1 &&
    kept[0][0] < 1e-9 &&
    Math.abs(kept[kept.length - 1][1] - twoPi) < 1e-9
  ) {
    const tail = /** @type {[number, number]} */ (kept.pop());
    kept[0] = [tail[0] - twoPi, kept[0][1]];
  }
  return kept;
}

/**
 * Return the Reach Edge paths of a state: {own, allied} polyline path text in
 * CSS pixels ("" when nothing is drawn). own holds the recipient's edge pieces,
 * allied every other sensor's.
 *
 * @param {ReadonlyArray<SightCircle>} circles
 * @param {number} width
 * @param {number} height
 * @param {LookTransform} transform
 * @returns {{own: string, allied: string}}
 */
function edgePaths(circles, width, height, transform) {
  /** @type {string[]} */
  const own = [];
  /** @type {string[]} */
  const allied = [];
  const stepRadians = (FOG_LOOK.edgeStepDegrees * Math.PI) / 180;
  circles.forEach((circle, index) => {
    for (const [start, end] of edgeIntervals(circles, index, width, height)) {
      const count = Math.max(1, Math.ceil((end - start) / stepRadians));
      const points = [];
      for (let step = 0; step <= count; step += 1) {
        const angle = start + ((end - start) * step) / count;
        const point = transform.worldToScreen([
          circle.x + circle.r * Math.cos(angle),
          circle.y + circle.r * Math.sin(angle),
        ]);
        points.push(`${px(point.x)} ${px(point.y)}`);
      }
      (circle.own ? own : allied).push(`M${points.join("L")}`);
    }
  });
  return { own: own.join(""), allied: allied.join("") };
}

/** Geometry cache: one entry per (circles, floor, projection). */
const GEOMETRY_CACHE_LIMIT = 64;
/** @type {Map<string, Readonly<{veilPath: string, liftPath: string, ownEdgePath: string, alliedEdgePath: string}>>} */
const geometryCache = new Map();

/**
 * Return the veil, lift and edge path text of one light state, cached per
 * circles, floor size and projection (so a paused frame, a re-render of the
 * same frame and a repeated view cost one build). The cache keeps the last 64
 * entries and holds only strings. Pure from the caller's view.
 *
 * @param {ReadonlyArray<SightCircle>} circles
 * @param {number} width
 * @param {number} height
 * @param {LookTransform} transform
 */
function stateGeometry(circles, width, height, transform) {
  const origin = transform.worldToScreen([0, 0]);
  const unit = transform.worldToScreen([1, 1]);
  const key = JSON.stringify([
    circles.map((c) => [c.x, c.y, c.r, c.own]),
    width,
    height,
    origin.x,
    origin.y,
    unit.x,
    unit.y,
  ]);
  const cached = geometryCache.get(key);
  if (cached) return cached;
  const runs = sightReachRuns(circles, width, height);
  const edges = edgePaths(circles, width, height, transform);
  const geometry = Object.freeze({
    veilPath: piecePath(runs.dark, transform),
    liftPath: piecePath(runs.lit, transform),
    ownEdgePath: edges.own,
    alliedEdgePath: edges.allied,
  });
  if (geometryCache.size >= GEOMETRY_CACHE_LIMIT) {
    const oldest = geometryCache.keys().next().value;
    if (oldest !== undefined) geometryCache.delete(oldest);
  }
  geometryCache.set(key, geometry);
  return geometry;
}

/**
 * Return the scene rows that are this frame's authorized sensors.
 *
 * sensorSources is `current_endpoint.parts.authorized_sensor_sources` of a
 * SharedObs Agent POV frame (rows with source_public_agent_id); null,
 * undefined or a non-array means NoSharedObs, where the only sensor is the
 * recipient (the scene row with relation "self"). Rows are returned dead or
 * alive, recipient first, then by public agent ID.
 *
 * @param {ReadonlyArray<JsonRecord>} agents
 * @param {unknown} sensorSources
 * @returns {JsonRecord[]}
 */
function sensorAgents(agents, sensorSources) {
  const ids = new Set(
    Array.isArray(sensorSources)
      ? sensorSources
          .filter(isRecord)
          .map((row) => row.source_public_agent_id)
          .filter((id) => typeof id === "string")
      : agents
          .filter((agent) => agent.relation === "self")
          .map((agent) => agent.public_agent_id),
  );
  return agents
    .filter(
      (agent) =>
        typeof agent.public_agent_id === "string" &&
        typeof agent.presentation_key === "string" &&
        ids.has(agent.public_agent_id),
    )
    .sort((a, b) => {
      const rank = (/** @type {JsonRecord} */ agent) =>
        agent.relation === "self" ? 0 : 1;
      return (
        rank(a) - rank(b) ||
        String(a.public_agent_id).localeCompare(String(b.public_agent_id))
      );
    });
}

/**
 * Return the presentation keys and times of a schedule's falls or returns,
 * sorted by time (then by key).
 *
 * @param {unknown} rows
 * @returns {Array<{key: string, at: number}>}
 */
function beatRows(rows) {
  return asArray(rows)
    .filter((row) => isRecord(row) && typeof row.presentationKey === "string")
    .map((row) => ({ key: String(row.presentationKey), at: finiteNumber(row.at) }))
    .sort((a, b) => a.at - b.at || a.key.localeCompare(b.key));
}

/**
 * Build the light states of one tick: START (living sensors at their start
 * spots), then the glide (Stride) to END spots, then each sensor's Fall beat
 * (its light goes out) and Return beat (its light opens on its pad), in beat
 * order. The last state always equals END (living sensors at END spots).
 * Neighbouring identical states are merged.
 *
 * @param {LookInput} input
 * @param {JsonRecord[]} sensors
 * @returns {Array<{circles: SightCircle[], arrive: SightBeat | null}>}
 */
function lightStates(input, sensors) {
  const schedule = isRecord(input.schedule) ? input.schedule : {};
  const falls = beatRows(schedule.falls);
  const returns = beatRows(schedule.returns);
  const fallKeys = new Set(falls.map((row) => row.key));
  const returnKeys = new Set(returns.map((row) => row.key));
  /** @type {Map<string, JsonRecord>} */
  const trajectories = new Map(
    asArray(input.trajectories)
      .filter((row) => isRecord(row) && typeof row.agent_presentation_key === "string")
      .map((row) => [String(row.agent_presentation_key), row]),
  );
  /**
   * @param {JsonRecord} agent
   * @param {"start" | "end"} stage
   * @returns {SightCircle | null}
   */
  const circleFor = (agent, stage) => {
    const trajectory = trajectories.get(agent.presentation_key);
    const start =
      stage === "start" && isRecord(trajectory?.transition_start)
        ? worldPoint(trajectory.transition_start.position)
        : null;
    const point = start ?? worldPoint(agent.position);
    const r = finiteNumber(agent.observation_radius);
    if (point === null || r <= 0) return null;
    return Object.freeze({
      presentationKey: String(agent.presentation_key),
      publicAgentId: String(agent.public_agent_id),
      x: point[0],
      y: point[1],
      r,
      own: agent.relation === "self",
    });
  };
  /** @param {JsonRecord} agent */
  const aliveAtEnd = (agent) => agent.life_state === "alive";
  /** @param {JsonRecord} agent */
  const aliveAtStart = (agent) =>
    (aliveAtEnd(agent) && !returnKeys.has(agent.presentation_key)) ||
    fallKeys.has(agent.presentation_key);
  /**
   * @param {(agent: JsonRecord) => boolean} lit
   * @param {"start" | "end"} stage
   */
  const circles = (lit, stage) =>
    /** @type {SightCircle[]} */ (
      sensors
        .filter(lit)
        .map((agent) => circleFor(agent, stage))
        .filter((circle) => circle !== null)
    );
  /** @type {Array<{circles: SightCircle[], arrive: SightBeat | null}>} */
  const states = [];
  if (trajectories.size > 0) {
    states.push({ circles: circles(aliveAtStart, "start"), arrive: null });
  }
  states.push({
    circles: circles(aliveAtStart, "end"),
    arrive: states.length > 0 ? { kind: "stride", key: null } : null,
  });
  const lit = new Set(
    sensors.filter(aliveAtStart).map((agent) => String(agent.presentation_key)),
  );
  const timeline = [
    ...falls.map((row) => ({ ...row, kind: /** @type {const} */ ("fall") })),
    ...returns.map((row) => ({ ...row, kind: /** @type {const} */ ("return") })),
  ].sort(
    (a, b) =>
      a.at - b.at ||
      (a.kind === b.kind ? a.key.localeCompare(b.key) : a.kind === "fall" ? -1 : 1),
  );
  for (const beat of timeline) {
    const agent = sensors.find((row) => row.presentation_key === beat.key);
    if (!agent) continue;
    if (beat.kind === "fall") lit.delete(beat.key);
    else if (aliveAtEnd(agent)) lit.add(beat.key);
    else continue;
    states.push({
      circles: circles((row) => lit.has(String(row.presentation_key)), "end"),
      arrive: { kind: beat.kind, key: beat.key },
    });
  }
  // The last state is END by definition.
  states[states.length - 1].circles = circles(aliveAtEnd, "end");
  /** @param {SightCircle[]} list */
  const signature = (list) => JSON.stringify(list.map((c) => [c.x, c.y, c.r, c.own]));
  /** @type {Array<{circles: SightCircle[], arrive: SightBeat | null}>} */
  const merged = [];
  for (const state of states) {
    const previous = merged[merged.length - 1];
    if (previous && signature(previous.circles) === signature(state.circles)) {
      continue;
    }
    merged.push(state);
  }
  return merged;
}

/**
 * Return the path text of one ring or polyline of world points, projected.
 *
 * @param {ReadonlyArray<unknown>} points World [x, y] pairs (y up).
 * @param {LookTransform} transform This render's projection.
 * @param {boolean} closed True to close the run with "Z" (a ring).
 * @returns {string} "" when fewer than two usable points.
 */
function runPath(points, transform, closed) {
  const projected = points
    .filter(
      (point) =>
        Array.isArray(point) &&
        point.length === 2 &&
        point.every((value) => typeof value === "number" && Number.isFinite(value)),
    )
    .map((point) => transform.worldToScreen(/** @type {[number, number]} */ (point)));
  if (projected.length < 2) return "";
  return `M${projected.map((point) => `${px(point.x)} ${px(point.y)}`).join("L")}${closed ? "Z" : ""}`;
}

/**
 * Return the veil, lift and edge paths of the host's merged team sight.
 *
 * teamSight is the live Agent POV scene's `team_sight` (world units): its
 * `lit_rings` are closed outlines of the lit floor; `own_edges` and
 * `allied_edges` are Reach Edge polylines. The veil path is the floor
 * rectangle (corners rounded as the vision's veil, 6 of its 50-per-unit
 * pixels) plus every ring, and the lift path is the rings alone; both are
 * filled even-odd, so the veil covers exactly the unseen floor, stone
 * shadows included, with flat export-safe shapes (no mask). Nothing here
 * tests stone: the host did (amendment A17).
 *
 * @param {JsonRecord} teamSight The scene's team_sight row.
 * @param {number} width Floor width in world units.
 * @param {number} height Floor height in world units.
 * @param {LookTransform} transform This render's projection.
 * @returns {{veilPath: string, liftPath: string, ownEdgePath: string, alliedEdgePath: string, fillRule: "evenodd"}}
 */
function teamSightGeometry(teamSight, width, height, transform) {
  const a = transform.worldToScreen([0, height]);
  const b = transform.worldToScreen([width, 0]);
  const left = Math.min(a.x, b.x);
  const right = Math.max(a.x, b.x);
  const top = Math.min(a.y, b.y);
  const bottom = Math.max(a.y, b.y);
  const r = Math.min(transform.worldLengthToScreen(6 / 50), (right - left) / 2);
  const floor =
    `M${px(left + r)} ${px(top)}H${px(right - r)}A${px(r)} ${px(r)} 0 0 1 ${px(right)} ${px(top + r)}` +
    `V${px(bottom - r)}A${px(r)} ${px(r)} 0 0 1 ${px(right - r)} ${px(bottom)}` +
    `H${px(left + r)}A${px(r)} ${px(r)} 0 0 1 ${px(left)} ${px(bottom - r)}` +
    `V${px(top + r)}A${px(r)} ${px(r)} 0 0 1 ${px(left + r)} ${px(top)}Z`;
  const rings = asArray(teamSight.lit_rings)
    .map((ring) => runPath(asArray(ring), transform, true))
    .join("");
  const edges = (/** @type {unknown} */ runs) =>
    asArray(runs)
      .map((run) => runPath(asArray(run), transform, false))
      .join("");
  return {
    veilPath: floor + rings,
    liftPath: rings,
    ownEdgePath: edges(teamSight.own_edges),
    alliedEdgePath: edges(teamSight.allied_edges),
    fillRule: "evenodd",
  };
}

/**
 * Return the Sight Reach veil record of an Agent POV scene, or null.
 *
 * @param {LookInput} input
 * @param {JsonRecord[]} agents
 * @param {JsonRecord} scene
 * @param {LookTransform} transform
 * @returns {LookRecord | null}
 */
function veilRecord(input, agents, scene, transform) {
  const map = isRecord(scene.map) ? scene.map : {};
  const width = finiteNumber(map.width);
  const height = finiteNumber(map.height);
  if (!(width > 0 && height > 0)) return null;
  const look = input.live === true ? FOG_LOOK_LIVE : FOG_LOOK;
  const opacities = Object.freeze({
    veil: look.veilOpacity,
    lift: look.liftOpacity,
    ownEdge: look.ownEdgeOpacity,
    alliedEdge: look.alliedEdgeOpacity,
  });
  if (input.live === true && isRecord(scene.team_sight)) {
    return Object.freeze({
      kind: "sight-veil",
      part: "under",
      width,
      height,
      opacities,
      states: Object.freeze([
        Object.freeze({
          index: 0,
          circles: Object.freeze([]),
          arrive: null,
          leave: null,
          endOpacity: 1,
          ...teamSightGeometry(scene.team_sight, width, height, transform),
        }),
      ]),
    });
  }
  const sensors = sensorAgents(agents, input.sensorSources);
  const states = lightStates(input, sensors);
  return Object.freeze({
    kind: "sight-veil",
    part: "under",
    width,
    height,
    opacities,
    states: Object.freeze(
      states.map((state, index) => {
        const geometry = stateGeometry(state.circles, width, height, transform);
        const next = states[index + 1];
        return Object.freeze({
          index,
          circles: Object.freeze([...state.circles]),
          arrive: state.arrive === null ? null : Object.freeze({ ...state.arrive }),
          leave: next?.arrive ? Object.freeze({ ...next.arrive }) : null,
          endOpacity: index === states.length - 1 ? 1 : 0,
          ...geometry,
        });
      }),
    ),
  });
}

/**
 * Return the body radius of a scene row in CSS pixels (missing radius: 0.5
 * world units, as champions.js draws it).
 *
 * @param {JsonRecord} agent
 * @param {LookTransform} transform
 * @returns {number}
 */
function bodyRadius(agent, transform) {
  return transform.worldLengthToScreen(
    finiteNumber(agent.radius, FOG_LOOK.defaultBodyRadiusWorld),
  );
}

/**
 * Return the "over" records: the recipient rim and the Spotted Glints.
 *
 * @param {LookInput} input
 * @param {JsonRecord[]} agents
 * @param {JsonRecord} scene
 * @param {LookTransform} transform
 * @returns {LookRecord[]}
 */
function overRecords(input, agents, scene, transform) {
  /** @type {LookRecord[]} */
  const out = [];
  const selection = isRecord(scene.selection) ? scene.selection : {};
  if (switchOn(input, "pov_recipient_rim")) {
    const self = agents.find((agent) => agent.relation === "self");
    const point = self ? worldPoint(self.position) : null;
    const radius = self ? bodyRadius(self, transform) : 0;
    const shielded = self ? finiteNumber(self.spawn_shield_remaining) > 0 : false;
    const controlled =
      self !== undefined &&
      selection.controlled_presentation_key === self.presentation_key;
    if (
      self &&
      point !== null &&
      self.life_state === "alive" &&
      !shielded &&
      !controlled &&
      radius >= FOG_LOOK.rimMinRadius
    ) {
      const center = transform.worldToScreen(point);
      out.push(
        Object.freeze({
          kind: "recipient-rim",
          part: "over",
          presentationKey: String(self.presentation_key),
          publicAgentId: String(self.public_agent_id),
          center: Object.freeze({ x: px(center.x), y: px(center.y) }),
          radius: px(radius * FOG_LOOK.rimRadiusFactor),
        }),
      );
    }
  }
  if (switchOn(input, "sight_change_marks")) {
    for (const row of asArray(input.trajectories)) {
      if (!isRecord(row) || row.transition_start != null || !isRecord(row.successor)) {
        continue;
      }
      const agent = agents.find(
        (candidate) => candidate.presentation_key === row.agent_presentation_key,
      );
      if (agent?.relation !== "opponent" || agent.life_state !== "alive") {
        continue;
      }
      const point = worldPoint(agent.position);
      if (point === null) continue;
      const center = transform.worldToScreen(point);
      out.push(
        Object.freeze({
          kind: "spotted-glint",
          part: "over",
          presentationKey: String(agent.presentation_key),
          publicAgentId: String(agent.public_agent_id),
          center: Object.freeze({ x: px(center.x), y: px(center.y) }),
          radius: px(bodyRadius(agent, transform)),
        }),
      );
    }
  }
  return out;
}

/**
 * Return the Last Seen record of one derived `last_seen` plan row, or null.
 *
 * @param {LookInput} input
 * @param {JsonRecord} event
 * @returns {LookRecord | null}
 */
function lastSeenRecord(input, event) {
  if (event.kind !== "last_seen" || event.presentationSuppressed === true) return null;
  if (!switchOn(input, "sight_change_marks")) return null;
  const anchor = isRecord(event.anchor) ? event.anchor : null;
  if (anchor === null || !Number.isFinite(anchor.x) || !Number.isFinite(anchor.y)) {
    return null;
  }
  // A body drawn at END (for example its corpse) needs no Last Seen ring.
  if (
    isRecord(input.scene) &&
    asArray(input.scene.agents).some(
      (agent) => isRecord(agent) && agent.presentation_key === event.presentationKey,
    )
  ) {
    return null;
  }
  const transform = isRecord(input.transform)
    ? /** @type {LookTransform} */ (input.transform)
    : null;
  const bodyR =
    transform !== null
      ? transform.worldLengthToScreen(FOG_LOOK.defaultBodyRadiusWorld)
      : FOG_LOOK.defaultBodyRadiusPx;
  const classId = Number(event.classId);
  const classIdentity = Object.hasOwn(CLASS_IDENTITIES, classId)
    ? CLASS_IDENTITIES[/** @type {1 | 2 | 3 | 4 | 5} */ (classId)]
    : null;
  const teamId = event.teamId === 1 || event.teamId === 2 ? event.teamId : null;
  return Object.freeze({
    kind: "last-seen",
    eventId: String(event.eventId ?? ""),
    presentationKey: String(event.presentationKey ?? ""),
    publicAgentId: String(event.publicAgentId ?? ""),
    classLabel: classIdentity?.label ?? null,
    glyphKey: classIdentity ? `class-${classIdentity.tokenId}` : null,
    teamId,
    center: Object.freeze({ x: px(Number(anchor.x)), y: px(Number(anchor.y)) }),
    radius: px(bodyR * FOG_LOOK.lastSeenRadiusFactor),
    emblemSize:
      bodyR >= FOG_LOOK.lastSeenEmblemMinRadius && classIdentity
        ? px(bodyR * FOG_LOOK.lastSeenEmblemFactor)
        : 0,
  });
}

/**
 * Return this look's drawing records. Pure data: no DOM, no clock.
 *
 * Two forms:
 * - Event form (`input.event` set): one derived `last_seen` plan row gives one
 *   "last-seen" record {kind, eventId, presentationKey, publicAgentId,
 *   classLabel, glyphKey, teamId, center, radius, emblemSize} (CSS pixels;
 *   center is the row's projected START spot). None when the row is not
 *   last_seen, is suppressed, has no anchor, `sight_change_marks` is off, or
 *   input.scene (optional) still holds that body at END. input.transform
 *   (optional) sizes the ring from a 0.5-unit body; without it R is 23 px.
 *   teamId is the row's team (1 or 2), taken by the plan from the public
 *   roster, so the ring is in that team's light as the store draws it (main
 *   session ruling P5, 3 October 2026); null (a body the roster does not
 *   name) gives neutral Moon Silver.
 * - Scene form: input.scene (the audience's arena scene), input.transform,
 *   input.audience (default scene.audience). Nothing unless the audience is
 *   "agent_pov". input.part picks the layer: "under" (default; the fog layer)
 *   gives one "sight-veil" record when `pov_sight_veil` is on; "over" (above
 *   the bodies) gives the "recipient-rim" record when `pov_recipient_rim` is
 *   on, and one "spotted-glint" record per living opponent whose trajectory
 *   has a successor and no start (it came into sight) when
 *   `sight_change_marks` is on.
 *   Further fields: input.filters (visual filter state; a missing switch is
 *   on); input.sensorSources (the frame's
 *   `current_endpoint.parts.authorized_sensor_sources`; null or absent means
 *   NoSharedObs, so only the recipient gives light); input.trajectories (the
 *   frame's `visual_events.agent_phase_trajectories`; null or absent means no
 *   tick, so the veil has only its END state and there are no glints);
 *   input.schedule (the audience's beat schedule, `plan.beats` or
 *   `beatView(frame).schedule`; its falls and returns time the light changes;
 *   absent means no beat changes).
 *
 * input.live (optional, default false): true in the BattleClient and the
 * DevClient combat screen, where the veil takes the vision's opacities
 * (FOG_LOOK_LIVE); otherwise FOG_LOOK's. In live play, when the scene
 * carries the host's merged team sight (`scene.team_sight`), the veil is that
 * shape, stone shadows included, as one END state (no beat states; the light
 * changes with the next frame), with fillRule "evenodd" on the state.
 * Otherwise it is built from the sensors' reach circles as before.
 *
 * The "sight-veil" record is {kind, part, width, height, opacities, states}:
 * width and height in world units; opacities {veil, lift, ownEdge,
 * alliedEdge} (0 to 1, the group and edge opacities the build writes);
 * states in time order, each {index, circles (world
 * units), arrive, leave (SightBeat or null), endOpacity (1 only for the last,
 * END, state), veilPath, liftPath, ownEdgePath, alliedEdgePath (CSS-pixel path
 * text, "" when empty)}. Only living sensors give light; with none the whole
 * floor is dark. The rim record is {kind, part, presentationKey,
 * publicAgentId, center, radius}; it steps aside while the recipient is dead,
 * under its Spawn Shield, is the controlled champion, or R is under 10 px. A
 * glint record is {kind, part, presentationKey, publicAgentId, center, radius}
 * (radius is R; the glint starts at 1.35R).
 *
 * Agent POV limits: reads only the given scene, sensor list, trajectories and
 * schedule (the audience's own); never a researcher view or match summary;
 * never tests stone. Results are frozen; geometry is cached (see
 * stateGeometry), so equal inputs give equal output.
 *
 * @param {LookInput} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  if (!isRecord(input)) return Object.freeze([]);
  if (isRecord(input.event)) {
    const record = lastSeenRecord(input, input.event);
    return Object.freeze(record === null ? [] : [record]);
  }
  const scene = isRecord(input.scene) ? input.scene : null;
  const transform = isRecord(input.transform)
    ? /** @type {LookTransform} */ (input.transform)
    : null;
  const audience = input.audience ?? scene?.audience ?? null;
  if (scene === null || transform === null || audience !== "agent_pov") {
    return Object.freeze([]);
  }
  const agents = asArray(scene.agents).filter(isRecord);
  if (input.part === "over") {
    return Object.freeze(overRecords(input, agents, scene, transform));
  }
  if (!switchOn(input, "pov_sight_veil")) return Object.freeze([]);
  const veil = veilRecord(input, agents, scene, transform);
  return Object.freeze(veil === null ? [] : [veil]);
}

/**
 * Write a veil state's beat attributes on its node (read by
 * sightAnimationSpecs): data-fog-state (index), data-fog-arrive-kind and
 * -key, data-fog-leave-kind and -key, and the END opacity attribute.
 *
 * @param {Element} element
 * @param {JsonRecord} state
 */
function setStateAttributes(element, state) {
  element.setAttribute("data-fog-state", String(state.index));
  element.setAttribute("opacity", String(state.endOpacity));
  for (const [side, beat] of [
    ["arrive", state.arrive],
    ["leave", state.leave],
  ]) {
    if (isRecord(beat)) {
      element.setAttribute(`data-fog-${side}-kind`, String(beat.kind));
      if (typeof beat.key === "string") {
        element.setAttribute(`data-fog-${side}-key`, beat.key);
      } else {
        element.removeAttribute(`data-fog-${side}-key`);
      }
    } else {
      element.removeAttribute(`data-fog-${side}-kind`);
      element.removeAttribute(`data-fog-${side}-key`);
    }
  }
}

/**
 * Build the Sight Reach veil group of a "sight-veil" record.
 *
 * @param {Document} doc
 * @param {JsonRecord} record
 * @returns {SVGElement}
 */
function buildVeil(doc, record) {
  const root = svgNode(doc, "g", {
    class: "fog-sight",
    "data-fog-kind": "sight-veil",
    "aria-hidden": "true",
  });
  const opacities = isRecord(record.opacities) ? record.opacities : {};
  const veil = svgNode(doc, "g", {
    class: "fog-veil",
    opacity: opacities.veil ?? FOG_LOOK.veilOpacity,
  });
  const lift = svgNode(doc, "g", {
    class: "fog-lift",
    opacity: opacities.lift ?? FOG_LOOK.liftOpacity,
  });
  const edge = svgNode(doc, "g", { class: "fog-edge" });
  for (const state of record.states) {
    const fillRule = state.fillRule === "evenodd" ? "evenodd" : "nonzero";
    const veilState = svgNode(doc, "path", {
      class: "fog-veil-path",
      d: state.veilPath,
      "fill-rule": fillRule,
    });
    const liftState = svgNode(doc, "path", {
      class: "fog-lift-path",
      d: state.liftPath,
      "fill-rule": fillRule,
    });
    const edgeState = svgNode(doc, "g", { class: "fog-edge-state" });
    edgeState.append(
      svgNode(doc, "path", {
        class: "fog-edge-own",
        d: state.ownEdgePath,
        fill: "none",
        opacity: opacities.ownEdge ?? FOG_LOOK.ownEdgeOpacity,
      }),
      svgNode(doc, "path", {
        class: "fog-edge-allied",
        d: state.alliedEdgePath,
        fill: "none",
        opacity: opacities.alliedEdge ?? FOG_LOOK.alliedEdgeOpacity,
      }),
    );
    for (const element of [veilState, liftState, edgeState]) {
      setStateAttributes(element, state);
    }
    veil.append(veilState);
    lift.append(liftState);
    edge.append(edgeState);
  }
  root.append(veil, lift, edge);
  return root;
}

/**
 * Build a recipient rim or a Spotted Glint circle.
 *
 * @param {Document} doc
 * @param {JsonRecord} record
 * @returns {SVGElement}
 */
function buildOverCircle(doc, record) {
  const glint = record.kind === "spotted-glint";
  return svgNode(doc, "circle", {
    class: glint ? "fog-glint" : "fog-rim",
    "data-fog-kind": record.kind,
    "data-presentation-key": record.presentationKey,
    cx: record.center.x,
    cy: record.center.y,
    r: record.radius,
    fill: "none",
    // The glint's END look is invisible; its fade is a play-only override.
    opacity: glint ? 0 : FOG_LOOK.rimOpacity,
    "aria-hidden": "true",
  });
}

/**
 * Return the store's 16-point Mage star (skins/fog-and-vision.js STAR) as path
 * data in the 24-unit emblem box: long points 10.5, middle 6 and short 2.5
 * units from the center, the first long point straight up. Pure.
 *
 * @returns {string}
 */
function ghostStar() {
  let d = "";
  for (let k = 0; k < 16; k += 1) {
    const a = (-90 + k * 22.5) * (Math.PI / 180);
    const r = k % 4 === 0 ? 10.5 : k % 2 === 0 ? 6 : 2.5;
    d += `${k ? "L" : "M"}${(12 + r * Math.cos(a)).toFixed(2)} ${(12 + r * Math.sin(a)).toFixed(2)}`;
  }
  return `${d}Z`;
}

/**
 * Return the store's eight short Priest sun rays (skins/fog-and-vision.js
 * PRIEST_RAYS) as path data: from 9.4 to 11 units out, every 45 degrees from
 * 22.5. Pure.
 *
 * @returns {string}
 */
function ghostRays() {
  let d = "";
  for (let k = 0; k < 8; k += 1) {
    const a = (k * 45 + 22.5) * (Math.PI / 180);
    d += `M${(12 + 9.4 * Math.cos(a)).toFixed(2)} ${(12 + 9.4 * Math.sin(a)).toFixed(2)}L${(12 + 11 * Math.cos(a)).toFixed(2)} ${(12 + 11 * Math.sin(a)).toFixed(2)}`;
  }
  return d;
}

/**
 * @typedef {readonly [string, Readonly<Record<string, string | number>>]} GhostShape
 *   One shape of a ghost emblem: its SVG tag and its attributes.
 */

/**
 * Return one frozen ghost-emblem shape. Pure.
 *
 * @param {string} tag The SVG tag ("path" or "circle").
 * @param {Record<string, string | number>} attributes Its geometry.
 * @returns {GhostShape}
 */
function ghostShape(tag, attributes) {
  return /** @type {GhostShape} */ (Object.freeze([tag, Object.freeze(attributes)]));
}

/**
 * The Last Seen ghost emblems, copied from the store pick
 * (skins/fog-and-vision.js EMBLEM, "Codex shapes, outline only"): one hollow
 * line drawing per class token id in a 24-unit box, as [tag, attributes]
 * shapes. buildLastSeen draws them unfilled in Moon Silver.
 *
 * @type {Readonly<Record<"mage" | "warrior" | "hunter" | "rogue" | "priest", ReadonlyArray<GhostShape>>>}
 */
const GHOST_EMBLEMS = Object.freeze({
  mage: Object.freeze([
    ghostShape("path", { d: ghostStar() }),
    ghostShape("circle", { cx: 12, cy: 12, r: 1.2 }),
  ]),
  warrior: Object.freeze([
    ghostShape("path", {
      d: "M12 3.2 19.2 5.6V11.4C19.2 16 16.2 19.2 12 21 7.8 19.2 4.8 16 4.8 11.4V5.6Z",
    }),
    ghostShape("circle", { cx: 12, cy: 11.4, r: 2.4 }),
  ]),
  hunter: Object.freeze([
    ghostShape("circle", { cx: 12, cy: 17.4, r: 2.6 }),
    ghostShape("path", {
      d: "M10.4 15.5C8.3 12.9 7.2 9.6 7.6 4.4M7.9 10.2 4.4 8.5M7.6 6.9 5.1 4.3M13.6 15.5C15.7 12.9 16.8 9.6 16.4 4.4M16.1 10.2 19.6 8.5M16.4 6.9 18.9 4.3",
    }),
  ]),
  rogue: Object.freeze([
    ghostShape("path", {
      d: "M10.25 15.45 18.5 5.5 8.55 13.75ZM11.5 16.7 7.3 12.5M9.4 14.6 5.5 18.5M13.75 15.45 5.5 5.5 15.45 13.75ZM12.5 16.7 16.7 12.5M14.6 14.6 18.5 18.5",
    }),
  ]),
  priest: Object.freeze([
    ghostShape("circle", { cx: 12, cy: 12, r: 8.2 }),
    ghostShape("path", {
      d: `M10.4 6.4H13.6V10.4H17.6V13.6H13.6V17.6H10.4V13.6H6.4V10.4H10.4Z${ghostRays()}`,
    }),
  ]),
});

/**
 * Put a Last Seen ghost emblem (an svg with a 24-unit viewBox) on its record.
 *
 * Sets x, y, width and height so the emblem is record.emblemSize CSS pixels
 * square and centered on record.center, and a stroke width of 24 /
 * emblemSize viewBox units, so its outline is 1 CSS pixel at any size (the
 * store's `hollowEmblem` stroke). Return undefined; changes the node only.
 *
 * @param {Element} emblem
 * @param {JsonRecord} record
 */
function placeLastSeenEmblem(emblem, record) {
  const size = record.emblemSize;
  emblem.setAttribute("x", String(px(record.center.x - size / 2)));
  emblem.setAttribute("y", String(px(record.center.y - size / 2)));
  emblem.setAttribute("width", String(size));
  emblem.setAttribute("height", String(size));
  emblem.setAttribute("stroke-width", String(px(24 / size)));
}

/**
 * Draw one Last Seen mark into an event mount and queue its fade-in.
 *
 * @param {Document} doc
 * @param {JsonRecord} record
 * @param {EventMount | null | undefined} mount
 * @returns {SVGElement}
 */
function buildLastSeen(doc, record, mount) {
  const group = svgNode(doc, "g", {
    class: "fog-last-seen",
    "data-fog-kind": "last-seen",
    "data-presentation-key": record.presentationKey,
    "data-team": record.teamId ?? "none",
    role: "img",
    tabindex: "0",
  });
  group.append(
    svgNode(doc, "circle", {
      class: "fog-last-seen-ring",
      cx: record.center.x,
      cy: record.center.y,
      r: record.radius,
      fill: "none",
      opacity: FOG_LOOK.lastSeenOpacity,
    }),
  );
  const shapes =
    typeof record.glyphKey === "string"
      ? GHOST_EMBLEMS[
          /** @type {keyof typeof GHOST_EMBLEMS} */ (record.glyphKey.slice(6))
        ]
      : undefined;
  if (record.emblemSize > 0 && shapes !== undefined) {
    // The store's ghost emblem: always hollow line art, never the filled sigil.
    const emblem = svgNode(doc, "svg", {
      class: "fog-last-seen-emblem",
      viewBox: "0 0 24 24",
      fill: "none",
      stroke: "currentColor",
      "stroke-linejoin": "round",
      "stroke-linecap": "round",
      opacity: FOG_LOOK.lastSeenEmblemOpacity,
      "aria-hidden": "true",
      "data-emblem": record.glyphKey.slice(6),
    });
    for (const [tag, attributes] of shapes)
      emblem.append(svgNode(doc, tag, attributes));
    placeLastSeenEmblem(emblem, record);
    group.append(emblem);
  }
  const descriptor = tooltip(record);
  if (descriptor !== null) {
    group.setAttribute("aria-label", descriptor.title);
    registerTooltipOwner(group, descriptor);
  }
  if (mount) {
    const spec = lastSeenFade(group, record, mount);
    if (spec !== null) mount.animationSpecs.push(spec);
  }
  return group;
}

/**
 * Return the fade-in of a Last Seen mark, or null for a still.
 *
 * Normal motion: over the second half of the glide window (the body leaves
 * sight as it moves), or 0-160 ms when Body Glide is off. Reduced motion: a
 * 50 ms fade at the tick start. Motion off, a settled mount or a static
 * policy: none. fill "backwards" holds the hidden look before the fade; the
 * DOM keeps the visible END.
 *
 * @param {Element} group
 * @param {JsonRecord} record
 * @param {EventMount} mount
 * @returns {AnimationSpec | null}
 */
function lastSeenFade(group, record, mount) {
  const options = mount.options ?? {};
  if (
    options.motionMode === "off" ||
    options.settled === true ||
    options.renderPolicy === "replay_static"
  ) {
    return null;
  }
  const glide = mount.plan?.beats?.windows?.glide;
  let delay = 0;
  let duration = 160;
  if (options.motionMode === "reduced") {
    duration = FOG_LOOK.reducedMarkFadeMs;
  } else if (Array.isArray(glide) && glide.length === 2) {
    const half = (Number(glide[1]) - Number(glide[0])) / 2;
    delay = Number(glide[0]) + half;
    duration = Math.max(1, half);
  }
  return Object.freeze({
    element: group,
    keyframes: [{ opacity: 0 }, { opacity: 1 }],
    options: Object.freeze({ delay, duration, fill: "backwards", easing: "linear" }),
    id: `mbg:${mount.plan?.epochKey ?? "detached"}:${record.eventId}:last-seen`,
  });
}

/**
 * Build export-safe SVG nodes for records (flat fills, strokes, opacity and
 * the emblem's plain shapes; no gradient, mask, filter, url() or image).
 *
 * records come from records(); doc creates the nodes. Scene records return
 * one node each: a "sight-veil" gives a g.fog-sight (veil, lift and edge
 * groups, one child per light state, beat attributes set for
 * sightAnimationSpecs); a "recipient-rim" gives circle.fog-rim; a
 * "spotted-glint" gives circle.fog-glint at END opacity 0. "last-seen"
 * records are drawn into mount.group when mount (an EventMount) is given and
 * the mount's nodes are returned (its fade is pushed to mount.animationSpecs);
 * without a mount each returns its own g.fog-last-seen. Every Last Seen group
 * gets its tooltip registered through tooltip.js. An empty list returns [] and
 * touches no DOM (doc may then be null). The caller places the nodes; the
 * veil and the Last Seen marks have no pointer hit area except the mark.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @param {EventMount} [mount]
 * @returns {SVGElement[]}
 */
export function build(records, doc, mount) {
  if (records.length === 0) return [];
  /** @type {SVGElement[]} */
  const nodes = [];
  let drewIntoMount = false;
  for (const record of records) {
    if (record.kind === "sight-veil") {
      nodes.push(buildVeil(doc, record));
    } else if (record.kind === "recipient-rim" || record.kind === "spotted-glint") {
      nodes.push(buildOverCircle(doc, record));
    } else if (record.kind === "last-seen") {
      const group = buildLastSeen(doc, record, mount);
      if (mount) {
        mount.group.append(group);
        drewIntoMount = true;
      } else {
        nodes.push(group);
      }
    }
  }
  return drewIntoMount && mount ? mountNodes(mount) : nodes;
}

/**
 * Move existing nodes to new records (reprojection after a resize or camera
 * change). nodes is what build returned for the same kinds of records, in the
 * same order; each node is matched to the record of the same kind (and, for
 * circles, the same presentation key). null and undefined entries are skipped:
 * the choreography painter passes [group, connector, underlay] and allows a
 * missing connector or underlay. Path data, centres and radii are rewritten,
 * and a Last Seen emblem gets its new place, size and 1 px stroke; beat
 * attributes are left alone. A node with no matching record is left
 * unchanged. Return undefined. Never throws for missing nodes. Changes the DOM
 * only. Agent POV: it only moves marks build already drew from authorized rows;
 * it reads nothing new.
 *
 * @param {ReadonlyArray<Element | null | undefined>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  const all = nodes
    .filter((node) => node !== null && node !== undefined)
    .flatMap((node) => [node, ...node.querySelectorAll("[data-fog-kind]")]);
  for (const node of all) {
    const kind = node.getAttribute("data-fog-kind");
    const key = node.getAttribute("data-presentation-key");
    const record = records.find(
      (row) =>
        row.kind === kind &&
        (key === null ||
          row.presentationKey === undefined ||
          row.presentationKey === key),
    );
    if (!record) continue;
    if (kind === "sight-veil") {
      for (const state of record.states) {
        const selector = `[data-fog-state="${state.index}"]`;
        node.querySelector(`.fog-veil ${selector}`)?.setAttribute("d", state.veilPath);
        node.querySelector(`.fog-lift ${selector}`)?.setAttribute("d", state.liftPath);
        const edge = node.querySelector(`.fog-edge ${selector}`);
        edge?.querySelector(".fog-edge-own")?.setAttribute("d", state.ownEdgePath);
        edge
          ?.querySelector(".fog-edge-allied")
          ?.setAttribute("d", state.alliedEdgePath);
      }
    } else if (kind === "recipient-rim" || kind === "spotted-glint") {
      node.setAttribute("cx", String(record.center.x));
      node.setAttribute("cy", String(record.center.y));
      node.setAttribute("r", String(record.radius));
    } else if (kind === "last-seen") {
      const ring = node.querySelector(".fog-last-seen-ring");
      ring?.setAttribute("cx", String(record.center.x));
      ring?.setAttribute("cy", String(record.center.y));
      ring?.setAttribute("r", String(record.radius));
      const emblem = node.querySelector(".fog-last-seen-emblem");
      if (emblem && record.emblemSize > 0) placeLastSeenEmblem(emblem, record);
    }
  }
}

/**
 * Return the tooltip descriptor of a record (through tooltip.js), or null for
 * an unknown kind. Only Last Seen marks register theirs on the arena; the
 * other kinds have no hover on the arena. The text names no champion the audience cannot
 * see: a Last Seen card names only the class it saw.
 *
 * @param {LookRecord} record
 * @returns {ReturnType<typeof createSemanticDescriptor> | null}
 */
export function tooltip(record) {
  /** @type {Record<string, [string, string]>} */
  const copy = {
    "sight-veil": [
      "Sight Reach",
      "In Agent POV the dark floor is beyond the sight reach of your team's living champions. It shows reach, not true sight: stone also blocks sight inside the circle, and champions just back on their pads stay hidden.",
    ],
    "recipient-rim": ["Agent POV", "You see through this champion's eyes."],
    "spotted-glint": ["Came Into Sight", "This champion came into sight this tick."],
    "last-seen": [
      "Last Seen",
      `${record.classLabel ? `A ${record.classLabel} was` : "A champion was"} here at the start of this tick, then your team lost sight of it. It may have moved out of reach, behind stone, or fallen.`,
    ],
  };
  const entry = copy[String(record.kind)];
  if (!entry) return null;
  return createSemanticDescriptor({
    kind: "visibility",
    id: `fog:${record.kind}:${record.publicAgentId ?? "view"}`,
    title: entry[0],
    tone: "neutral",
    accent: "none",
    summary: entry[1],
    rows: [],
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "pointer",
  });
}

/**
 * Return the real time (ms) of a veil beat in a plan's beat schedule, or null
 * when the schedule has no such beat (then the change is at once).
 *
 * @param {JsonRecord} beats
 * @param {string | null} kind
 * @param {string | null} key
 * @returns {number | null}
 */
function beatTime(beats, kind, key) {
  if (kind === "stride") {
    const glide = beats.windows?.glide;
    return Array.isArray(glide) ? finiteNumber(glide[0]) : null;
  }
  const rows = kind === "fall" ? beats.falls : kind === "return" ? beats.returns : null;
  const row = asArray(rows).find(
    (entry) => isRecord(entry) && entry.presentationKey === key,
  );
  return row && Number.isFinite(row.at) ? Number(row.at) : null;
}

/**
 * Return the fade-in length (ms) of a veil beat; the old state fades out over
 * twice this, so unchanged night never thins while the two cross.
 *
 * @param {JsonRecord} beats
 * @param {string | null} kind
 * @param {boolean} reduced
 * @returns {number}
 */
function fadeIn(beats, kind, reduced) {
  if (reduced) return FOG_LOOK.reducedFadeInMs;
  if (kind === "stride") {
    const glide = beats.windows?.glide;
    return Array.isArray(glide)
      ? Math.max(1, (finiteNumber(glide[1]) - finiteNumber(glide[0])) / 2)
      : 1;
  }
  return kind === "fall" ? FOG_LOOK.fallFadeInMs : FOG_LOOK.returnFadeInMs;
}

/**
 * Return the Web Animations overrides that time this look's nodes in one
 * render, for the choreographer to start (ids `mbg:<epoch>:fog:<n>`).
 *
 * root is the element searched (the battlefield SVG or the fog layer and the
 * over layer); plan is the choreography plan (epochKey and beats, a
 * BeatSchedule); options.motionMode is "normal", "reduced" or "off";
 * options.renderPolicy is "live_once", "replay_animated", "replay_static" or
 * "replay_step"; options.step is "forward" for a +1 step (anything else counts
 * as no glide).
 *
 * Played tick (live_once or replay_animated, motion on): each veil state
 * shows from its arrive beat to its leave beat. The new state fades in (half
 * the glide for the Stride, 200 ms at a Fall beat, 150 ms at a Return beat)
 * and the old one fades out over twice that. A beat the schedule lacks (for
 * example Death Effects off) changes at once. Reduced motion scales beat
 * times by beats.reducedScale and uses 110 ms fades. Each Spotted Glint plays
 * over the Settle window: opacity 0 to 50% to 0 while it shrinks from 1.35R
 * to R (reduced: one 110 ms blink, no shrink). A +1 step (replay_step with
 * step "forward", normal motion): the first state fades out and the END state
 * fades in over 200 ms; the rest snaps. Anything else returns []. Every
 * override uses fill "backwards", so cancelling or finishing it shows END.
 * Reads the DOM only; changes nothing.
 *
 * @param {ParentNode | null} root
 * @param {JsonRecord | null} plan
 * @param {{motionMode?: string, renderPolicy?: string, step?: string | null}} [options]
 * @returns {AnimationSpec[]}
 */
export function sightAnimationSpecs(root, plan, options = {}) {
  if (root === null || !isRecord(plan) || !isRecord(plan.beats)) return [];
  const motion = options.motionMode ?? "normal";
  const policy = options.renderPolicy ?? "replay_static";
  if (motion === "off") return [];
  const beats = plan.beats;
  /** @type {AnimationSpec[]} */
  const specs = [];
  /**
   * @param {Element} element
   * @param {Keyframe[]} keyframes
   * @param {KeyframeAnimationOptions} timing
   */
  const push = (element, keyframes, timing) => {
    specs.push(
      Object.freeze({
        element,
        keyframes,
        options: Object.freeze({ fill: "backwards", easing: "linear", ...timing }),
        id: `mbg:${plan.epochKey ?? "detached"}:fog:${specs.length}`,
      }),
    );
  };
  const states = [...root.querySelectorAll("[data-fog-state]")];
  if (policy === "replay_step") {
    if (motion !== "normal" || options.step !== "forward") return [];
    const indexes = states.map((element) =>
      Number(element.getAttribute("data-fog-state")),
    );
    const last = Math.max(-1, ...indexes);
    if (last < 1) return [];
    for (const element of states) {
      const index = Number(element.getAttribute("data-fog-state"));
      if (index === 0) {
        push(element, [{ opacity: 1 }, { opacity: 0 }], {
          duration: FOG_LOOK.stepFadeMs,
        });
      } else if (index === last) {
        push(element, [{ opacity: 0 }, { opacity: 1 }], {
          duration: FOG_LOOK.stepFadeMs,
        });
      }
    }
    return specs;
  }
  if (policy !== "live_once" && policy !== "replay_animated") return [];
  const reduced = motion === "reduced";
  const scale = reduced ? finiteNumber(beats.reducedScale, 1) : 1;
  for (const element of states) {
    const arriveKind = element.getAttribute("data-fog-arrive-kind");
    const leaveKind = element.getAttribute("data-fog-leave-kind");
    const end = finiteNumber(Number(element.getAttribute("opacity")), 1);
    /** @type {Array<[number, number]>} */
    const points = [];
    let cursor = 0;
    if (arriveKind === null) {
      points.push([0, 1]);
    } else {
      const at = beatTime(
        beats,
        arriveKind,
        element.getAttribute("data-fog-arrive-key"),
      );
      if (at === null) {
        points.push([0, 1]);
      } else {
        const start = at * scale;
        points.push(
          [0, 0],
          [start, 0],
          [start + fadeIn(beats, arriveKind, reduced), 1],
        );
        cursor = start + fadeIn(beats, arriveKind, reduced);
      }
    }
    if (leaveKind !== null) {
      const at = beatTime(beats, leaveKind, element.getAttribute("data-fog-leave-key"));
      if (at === null) {
        points.push([cursor, 0]);
      } else {
        const start = Math.max(cursor, at * scale);
        points.push([start, 1], [start + 2 * fadeIn(beats, leaveKind, reduced), 0]);
      }
    }
    const total = points[points.length - 1][0];
    if (total <= 0 || points.every(([, value]) => value === end)) continue;
    push(
      element,
      points.map(([time, value]) => ({ offset: time / total, opacity: value })),
      { duration: total },
    );
  }
  const settle = beats.windows?.settle;
  if (Array.isArray(settle)) {
    for (const glint of root.querySelectorAll('[data-fog-kind="spotted-glint"]')) {
      const from = finiteNumber(settle[0]) * scale;
      if (reduced) {
        push(
          glint,
          [{ opacity: 0 }, { opacity: FOG_LOOK.glintPeakOpacity }, { opacity: 0 }],
          { delay: from, duration: FOG_LOOK.reducedGlintMs },
        );
        continue;
      }
      const peakScale = (1 + FOG_LOOK.glintFromFactor) / 2;
      push(
        glint,
        [
          { opacity: 0, transform: `scale(${FOG_LOOK.glintFromFactor})` },
          { opacity: FOG_LOOK.glintPeakOpacity, transform: `scale(${peakScale})` },
          { opacity: 0, transform: "scale(1)" },
        ],
        {
          delay: from,
          duration: Math.max(1, finiteNumber(settle[1]) * scale - from),
        },
      );
    }
  }
  return specs;
}
