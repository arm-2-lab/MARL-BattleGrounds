/**
 * @file The Atmosphere look (Moonlit Night): Moon Motes, the Last Watch Veil
 * (Time of Night) and Death Dimming.
 *
 * - Moon Motes (`moon_motes`): six seeded pairs of faint Moon Mote dots (radius
 *   0.05 world units, at least 1.2 px), an exact mirror about the floor's middle,
 *   at rest at 18%. Below 20 screen px per unit no motes are drawn. They mean
 *   nothing in the game. The store's slow breathing and drift are not ported:
 *   the motes always rest (as the store shows them in live play and in a still),
 *   so the air holds still through the dramatic hold by construction.
 * - Time of Night (`time_of_night`): over the last fifth of the episode the floor
 *   greys in four steps, Last Watch Grey at 2.5%, 4%, 5.5% and 7% from ticks
 *   ceil(0.80 L), ceil(0.85 L), ceil(0.90 L) and ceil(0.95 L) of an episode limit
 *   L. On the tick a step changes, the new veil is beat-held to the Settle point
 *   (`data-beat-hold="score"`, a 400 ms cross-fade); a still shows the step for
 *   its tick.
 * - Death Dimming (`death_dimming`, part "dim"): during the Fall beat of a death
 *   this audience saw, Ink dims the floor to 28% (rise 120 ms, hold, release over
 *   max(300 ms, 40% of the beat), ending with the beat) with two 0.16-unit
 *   windows over the Red Zone rule lines. Normal motion only; never in a still.
 *   Its END opacity is 0; `dimAnimation` describes the browser animation for the
 *   choreographer's animation factory (fill "backwards").
 * All three are off by default and on in the Broadcast preset (plan P27).
 *
 * Who calls it: the renderer (motes and veil in a layer under the Red Zone; the
 * dim between the stones and the Strike Band) and the choreographer (the dim's
 * animation). `visual-filters.js` reads PAINT_PARTS.
 *
 * Boundary: the air reads no body. records reads the map record (size, stones,
 * Red Zone), spawn-pad spots, the public tick and episode limit and the Fall
 * beat window the caller takes from this audience's own beat schedule (Agent
 * POV: only seen deaths). It never reads a researcher view or a match summary.
 * No timers or CSS animations.
 */
import {
  asArray,
  finiteNumber,
  isRecord,
  paintParts,
  setNodeAttributes,
  svgNode,
} from "./look-support.js";
import { FLOOR_CORNER_RADIUS } from "./map-skins.js";

/**
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 */

/**
 * Store numbers for Moonlit Night (design store `skins/atmosphere-and-weather.js`).
 * Names ending in `Units` are world units; times are ms.
 */
export const FOOTPRINT = Object.freeze({
  motePairs: 6,
  moteUnits: 0.05,
  moteMinRadius: 1.2,
  moteOpacity: 0.18,
  motesBelow: 20,
  veilFractions: Object.freeze([0.8, 0.85, 0.9, 0.95]),
  veilOpacities: Object.freeze([0.025, 0.04, 0.055, 0.07]),
  veilFade: 400,
  dimPeak: 0.28,
  dimRise: 120,
  dimWindowUnits: 0.16,
});

/** Node allowance per record: the motes group holds 12 dots. No durable animations. */
export const BUDGET = Object.freeze({
  nodesPerRecord: 13,
  animationsPerRecord: 1,
  persistentNodesPerRecord: 0,
});

/** The three Moonlit Night switches, one paint part each. */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "moon_motes" }, "moon_motes"],
  [{ surface: "durable", kind: "last_watch_veil" }, "time_of_night"],
  [{ surface: "transient", kind: "death_dimming" }, "death_dimming"],
]);

/**
 * Return a seeded random generator (mulberry32), the store's generator. Pure
 * apart from its own state.
 *
 * @param {number} seed
 */
function mulberry32(seed) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Hash text to a 32-bit number (FNV-1a), as the store does for scenario ids. Pure.
 *
 * @param {string} text
 */
function hashText(text) {
  let h = 0x811c9dc5;
  for (const character of text) {
    h ^= character.charCodeAt(0);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * Return the world distance from (x, y) to a recorded stone's edge. Pure.
 *
 * @param {Record<string, any>} stone
 * @param {number} x
 * @param {number} y
 */
function stoneDistance(stone, x, y) {
  const [cx, cy] = asArray(stone.center).map((value) => finiteNumber(value));
  if (stone.kind === "pillar")
    return Math.hypot(x - cx, y - cy) - finiteNumber(stone.radius);
  if (stone.kind !== "wall") return Number.POSITIVE_INFINITY;
  const theta = finiteNumber(stone.theta);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const lx = (x - cx) * c + (y - cy) * s;
  const ly = -(x - cx) * s + (y - cy) * c;
  return Math.hypot(
    Math.max(Math.abs(lx) - finiteNumber(stone.width) / 2, 0),
    Math.max(Math.abs(ly) - finiteNumber(stone.height) / 2, 0),
  );
}

/**
 * Return the seeded left-twin mote spots (world units) for a map: up to six,
 * each with its twin at (width - x, y) clear of stones (0.4), spawn pads (1.02),
 * the left rule line (0.5) and each other (1.2). The seed is the map id times
 * 2654435761 (a text id is hashed first). Pure.
 *
 * @param {number | string} mapId
 * @param {number} width
 * @param {number} height
 * @param {ReadonlyArray<Record<string, any>>} stones
 * @param {ReadonlyArray<readonly [number, number]>} pads
 * @param {number} depth Left rule line x (0 when none).
 */
export function moteSpots(mapId, width, height, stones, pads, depth) {
  const base =
    typeof mapId === "number" && Number.isFinite(mapId)
      ? mapId * 2654435761
      : Math.imul(hashText(String(mapId)), 2654435761 | 0);
  const random = mulberry32((base + 97) >>> 0);
  const half = width / 2;
  /** @type {Array<readonly [number, number]>} */
  const spots = [];
  let tries = 0;
  while (spots.length < FOOTPRINT.motePairs && tries < 5000) {
    tries += 1;
    const x = 0.4 + random() * (half - 0.6 - 0.4);
    const y = 0.4 + random() * (height - 0.8);
    const twin = width - x;
    if (
      stones.some(
        (stone) =>
          stoneDistance(stone, x, y) < 0.4 || stoneDistance(stone, twin, y) < 0.4,
      )
    ) {
      continue;
    }
    if (
      pads.some(
        ([px, py]) =>
          Math.hypot(x - px, y - py) < 1.02 || Math.hypot(twin - px, y - py) < 1.02,
      )
    ) {
      continue;
    }
    if (depth > 0 && Math.abs(x - depth) < 0.5) continue;
    if (spots.some(([sx, sy]) => Math.hypot(x - sx, y - sy) < 1.2)) continue;
    spots.push([x, y]);
    random();
  }
  return spots;
}

/**
 * Return the Last Watch Veil opacity for a tick: 0 before ceil(0.8 L), then
 * 0.025, 0.04, 0.055 or 0.07. A missing tick or limit gives 0. Pure.
 *
 * @param {number} tick Public tick of the frame (0 or more).
 * @param {number} limit Episode limit L in ticks (positive).
 */
export function veilOpacity(tick, limit) {
  if (!(Number.isFinite(tick) && Number.isFinite(limit) && limit > 0)) return 0;
  let opacity = 0;
  FOOTPRINT.veilFractions.forEach((fraction, index) => {
    if (tick >= Math.ceil(fraction * limit)) opacity = FOOTPRINT.veilOpacities[index];
  });
  return opacity;
}

/**
 * Round to 0.001 for stable path text. Pure.
 *
 * @param {number} value
 */
function fixed(value) {
  return Number(value.toFixed(3));
}

/**
 * Return the rounded floor rectangle as path text. Pure.
 *
 * @param {{left: number, top: number, width: number, height: number}} box
 * @param {number} r Corner radius in px.
 */
function floorPath(box, r) {
  const { left: x, top: y, width: w, height: h } = box;
  return `M${fixed(x + r)} ${fixed(y)}H${fixed(x + w - r)}A${r} ${r} 0 0 1 ${fixed(x + w)} ${fixed(y + r)}V${fixed(y + h - r)}A${r} ${r} 0 0 1 ${fixed(x + w - r)} ${fixed(y + h)}H${fixed(x + r)}A${r} ${r} 0 0 1 ${fixed(x)} ${fixed(y + h - r)}V${fixed(y + r)}A${r} ${r} 0 0 1 ${fixed(x + r)} ${fixed(y)}Z`;
}

/**
 * Return the drawing records of Moonlit Night.
 *
 * input.transform projects (mapBounds is the floor); input.map is the map record
 * (width, height, obstacles, red_zone; default input.scene.map); input.mapId
 * names the map for the mote seed (a number or text; default a hash of the
 * map's size and stones); input.scene gives spawn-pad spots; input.filters is
 * the visual filter state (null: every Moonlit Night switch at its default, off);
 * input.tick is the frame's public tick and input.episodeLimit the episode limit
 * L in ticks (either missing: no veil); input.part is "dim" for Death Dimming,
 * else the motes and veil; input.fallBeat is this audience's Fall beat [start,
 * end] in ms (null: no seen death); input.motionMode is "normal", "reduced" or
 * "off" (default "normal").
 *
 * Return a frozen list in screen px:
 * - Default part: {kind: "moon-motes", dots: [{cx, cy}], r, opacity} when Moon
 *   Motes is on and the scale is 20 px per unit or more; {kind:
 *   "last-watch-veil", x, y, width, height, rx, opacity, fromOpacity (null when
 *   the step did not change this tick)} when Time of Night is on and the veil
 *   is above 0.
 * - Part "dim": {kind: "death-dim", d, start, end} when Death Dimming is on, a
 *   Fall beat exists and motion is normal (d is the floor with two windows over
 *   the Red Zone rule lines while red_zone_floors is on; fill-rule evenodd).
 * Pure data: no DOM.
 *
 * @param {{
 *   transform?: LookTransform | null,
 *   map?: Record<string, any> | null,
 *   mapId?: number | string | null,
 *   scene?: Record<string, any> | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   tick?: number | null,
 *   episodeLimit?: number | null,
 *   part?: "dim" | null,
 *   fallBeat?: readonly [number, number] | null,
 *   motionMode?: "normal" | "reduced" | "off" | null,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const transform = input.transform;
  const map = input.map ?? input.scene?.map ?? null;
  if (!transform) {
    return Object.freeze([]);
  }
  const filters = input.filters ?? null;
  const box = transform.mapBounds;
  const ppu = transform.worldLengthToScreen(1);
  const width = finiteNumber(map?.width, transform.worldWidth);
  const height = finiteNumber(map?.height, transform.worldHeight);
  const redZone = isRecord(map?.red_zone) ? map.red_zone : null;
  /** @type {LookRecord[]} */
  const result = [];
  if (input.part === "dim") {
    const beat = asArray(input.fallBeat).map((value) =>
      finiteNumber(value, Number.NaN),
    );
    if (
      filters?.death_dimming !== true ||
      beat.length !== 2 ||
      !beat.every(Number.isFinite) ||
      !(beat[1] > beat[0]) ||
      (input.motionMode ?? "normal") !== "normal"
    ) {
      return Object.freeze([]);
    }
    let d = floorPath(box, FLOOR_CORNER_RADIUS);
    if (redZone && filters?.red_zone_floors !== false) {
      const half = (FOOTPRINT.dimWindowUnits / 2) * ppu;
      for (const key of ["team_a_x_range", "team_b_x_range"]) {
        const range = asArray(redZone[key]).map((value) =>
          finiteNumber(value, Number.NaN),
        );
        if (range.length !== 2 || !range.every(Number.isFinite)) continue;
        for (const edge of range) {
          if (edge <= 1e-6 || edge >= width - 1e-6) continue;
          const x = transform.worldToScreen([edge, 0]).x;
          d += `M${fixed(x - half)} ${fixed(box.top)}V${fixed(box.top + box.height)}H${fixed(x + half)}V${fixed(box.top)}Z`;
        }
      }
    }
    result.push(Object.freeze({ kind: "death-dim", d, start: beat[0], end: beat[1] }));
    return Object.freeze(result);
  }
  if (filters?.moon_motes === true && ppu >= FOOTPRINT.motesBelow) {
    const stones = asArray(map?.obstacles).filter(isRecord);
    const pads = /** @type {Array<readonly [number, number]>} */ (
      /** @type {unknown} */ (
        asArray(input.scene?.spawn_pads)
          .map((pad) =>
            asArray(isRecord(pad) ? pad.position : null).map((value) =>
              finiteNumber(value, Number.NaN),
            ),
          )
          .filter((point) => point.length === 2 && point.every(Number.isFinite))
      )
    );
    const leftEdges = redZone
      ? ["team_a_x_range", "team_b_x_range"]
          .map((key) =>
            asArray(redZone[key]).map((value) => finiteNumber(value, Number.NaN)),
          )
          .filter(
            (range) =>
              range.length === 2 && range[0] <= 1e-6 && Number.isFinite(range[1]),
          )
          .map((range) => range[1])
      : [];
    const mapId =
      input.mapId ??
      `${width}x${height}:${JSON.stringify(stones.map((stone) => [stone.kind, stone.center, stone.radius, stone.width, stone.height, stone.theta]))}`;
    const spots = moteSpots(mapId, width, height, stones, pads, leftEdges[0] ?? 0);
    const dots = spots.flatMap(([x, y]) => {
      const left = transform.worldToScreen([x, y]);
      const right = transform.worldToScreen([width - x, y]);
      return [
        Object.freeze({ cx: left.x, cy: left.y }),
        Object.freeze({ cx: right.x, cy: right.y }),
      ];
    });
    if (dots.length > 0) {
      result.push(
        Object.freeze({
          kind: "moon-motes",
          dots: Object.freeze(dots),
          r: Math.max(FOOTPRINT.moteUnits * ppu, FOOTPRINT.moteMinRadius),
          opacity: FOOTPRINT.moteOpacity,
        }),
      );
    }
  }
  if (filters?.time_of_night === true) {
    const tick = finiteNumber(input.tick, Number.NaN);
    const limit = finiteNumber(input.episodeLimit, Number.NaN);
    const opacity = veilOpacity(tick, limit);
    if (opacity > 0) {
      const before = veilOpacity(tick - 1, limit);
      result.push(
        Object.freeze({
          kind: "last-watch-veil",
          x: box.left,
          y: box.top,
          width: box.width,
          height: box.height,
          rx: FLOOR_CORNER_RADIUS,
          opacity,
          fromOpacity: before === opacity ? null : before,
        }),
      );
    }
  }
  return Object.freeze(result);
}

/**
 * Write one record's geometry onto its node (build and update). DOM only.
 *
 * @param {Element} node
 * @param {LookRecord} record
 */
function writeGeometry(node, record) {
  if (record.kind === "moon-motes") {
    for (const [index, dot] of [...node.children].entries()) {
      const spot = record.dots[index];
      if (spot) setNodeAttributes(dot, { cx: spot.cx, cy: spot.cy, r: record.r });
    }
  } else if (record.kind === "last-watch-veil") {
    setNodeAttributes(node, {
      x: record.x,
      y: record.y,
      width: record.width,
      height: record.height,
      rx: record.rx,
    });
  } else if (record.kind === "death-dim") {
    setNodeAttributes(node, { d: record.d });
  }
}

/**
 * Build the Moonlit Night nodes, in record order.
 *
 * records come from records(); doc creates the nodes. "moon-motes" gives a
 * `g.moon-motes` of `circle.moon-mote` dots at the group opacity; "last-watch-veil"
 * gives a `rect.last-watch-veil` with its fill opacity, beat-held to the Settle
 * point (data-beat-hold "score", side "after", data-beat-from the old step's
 * share, data-beat-fade 400) when the step changed this tick; "death-dim" gives a
 * `path.death-dim` with fill-rule evenodd (an attribute) at END opacity 0.
 * Every node is aria-hidden and ignores the pointer. Export-safe: flat fills and
 * opacity; no ids. Return the nodes.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @returns {SVGElement[]}
 */
export function build(records, doc) {
  /** @type {SVGElement[]} */
  const nodes = [];
  for (const record of records) {
    /** @type {SVGElement} */
    let node;
    if (record.kind === "moon-motes") {
      node = svgNode(doc, "g", {
        class: "moon-motes",
        opacity: record.opacity,
        "aria-hidden": "true",
      });
      for (const _dot of record.dots)
        node.append(svgNode(doc, "circle", { class: "moon-mote" }));
    } else if (record.kind === "last-watch-veil") {
      node = svgNode(doc, "rect", {
        class: "last-watch-veil",
        "fill-opacity": record.opacity,
        "aria-hidden": "true",
      });
      if (record.fromOpacity !== null) {
        setNodeAttributes(node, {
          "data-beat-hold": "score",
          "data-beat-side": "after",
          "data-beat-from": fixed(record.fromOpacity / record.opacity),
          "data-beat-fade": FOOTPRINT.veilFade,
        });
      }
    } else if (record.kind === "death-dim") {
      node = svgNode(doc, "path", {
        class: "death-dim",
        "fill-rule": "evenodd",
        opacity: 0,
        "aria-hidden": "true",
      });
    } else {
      continue;
    }
    writeGeometry(node, record);
    nodes.push(node);
  }
  return nodes;
}

/**
 * Move existing nodes to new records (reprojection). nodes and records are in
 * the same order and kinds. Return undefined; DOM change only.
 *
 * @param {ReadonlyArray<Element>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  for (const [index, node] of nodes.entries()) {
    const record = records[index];
    if (record !== undefined) writeGeometry(node, record);
  }
}

/**
 * Describe the Death Dimming animation for the choreographer's animation factory.
 *
 * element is the dim's `path.death-dim`; record its "death-dim" record; plan
 * supplies epochKey for the animation id. Return {element, keyframes, options,
 * id}: opacity 0 to 0.28 over 120 ms (ease-out), held, then released over
 * max(300 ms, 40% of the beat) (ease-in-out) to 0 at the beat's end; delay is the
 * beat start, fill "backwards", so cancelling or finishing shows END (0). Pure.
 *
 * @param {Element} element
 * @param {LookRecord} record
 * @param {{epochKey?: string}} plan
 */
export function dimAnimation(element, record, plan) {
  const duration = record.end - record.start;
  const release = Math.max(300, 0.4 * duration);
  const rise = Math.min(FOOTPRINT.dimRise / duration, 1);
  const hold = Math.max(rise, (duration - release) / duration);
  return Object.freeze({
    element,
    keyframes: [
      { opacity: 0, easing: "ease-out" },
      { opacity: FOOTPRINT.dimPeak, offset: rise },
      { opacity: FOOTPRINT.dimPeak, offset: hold, easing: "ease-in-out" },
      { opacity: 0 },
    ],
    options: Object.freeze({ delay: record.start, duration, fill: "backwards" }),
    id: `mbg:${plan.epochKey ?? "air"}:death-dim`,
  });
}

/**
 * Return the tooltip for a record: the air has no tips, so always null.
 *
 * @param {LookRecord} _record
 * @returns {null}
 */
export function tooltip(_record) {
  return null;
}
