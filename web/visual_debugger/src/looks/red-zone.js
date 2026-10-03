/**
 * @file The Red Zone look (Grave Ground) in the `red-zone` layer: each team's home
 * zone, the same on both ends and with no team colour.
 *
 * What it draws, bottom to top:
 * - an Ash floor (17%) over each recorded strip, its outer corners following the
 *   rounded floor;
 * - on each strip's inner edge (the exact rule edge) a Grave Bone border (1.25 px
 *   at 60%) over a 3 px Ink keyline (50%), broken 0.16 world units around a small
 *   gravestone (0.2 by 0.27 units) at every y = k + 0.5; the border also stops
 *   where a stone stands on the line (the store's stop rule);
 * - a skull rune joined to a 12 px "×2" mark near the top and the bottom of each
 *   home, as the store draws them: on the store's clear spots (off stones, spawn
 *   pads and the gravestone border), the skull on the home's back side and the
 *   "×2" toward the border. In live play (input.vision, the BattleClient and
 *   the DevClient) the mark, its clear box and the border strokes are scaled
 *   with the field (VISION_FIELD), so they keep the store's size beside the
 *   gravestones;
 * - Home Fall: when this audience saw a fall worth 2 points in a home, that home
 *   lights once at its Fall beat: extra Ash (+5%), the border and filled
 *   gravestones in Grave Bone at 35%. These nodes hold the END look and carry
 *   `data-beat-hold="fall"` (side "after", from 0, a 120 ms fade), so the
 *   choreographer hides them until the beat; a paused frame shows them at 35%.
 * Below 20 screen px per world unit the gravestones drop, the border becomes one
 * plain line (2 px bone over 4 px ink) and each home keeps one 14 px mark
 * (Replay Viewer only; live play always draws the full layout).
 * With Red Zone Floors off a plain thin Grave Bone line marks each inner edge
 * (plan P26): the rule edge is never hidden.
 *
 * Who calls it: `scene.js` calls records, then build, and puts the nodes just
 * above the floor, under the spawns, bodies and effects. `visual-filters.js`
 * reads PAINT_PARTS; `choreography-plan.js` reads BUDGET for its Home Fall rows.
 *
 * Boundary: the zone is the same in every view. records reads the recorded
 * `map.red_zone` strips, the map size, the recorded stones and spawn-pad spots
 * (public map facts), the projection and the filter switches. The Home Fall
 * reads only `homeFallIds`, which the caller takes from this audience's own
 * events (Oracle View: latest_events; Agent POV: visual_events, where the host
 * sends the flag only for seen falls), and keeps an ID only when that champion
 * is in this audience's scene. It never reads a researcher view or a match
 * summary. No timers or CSS animations: the beat clock owns the timing.
 */
import { teamLabelFromId } from "../agent-identity.js";
import { createSymbolDescriptor, registerTooltipOwner } from "../tooltip.js";
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
 * @typedef {readonly [number, number]} Pair
 */

/**
 * Store numbers for Grave Ground (design store `skins/red-zone.js`). Names ending
 * in `Units` are world units; the rest are screen px or opacities.
 */
export const FOOTPRINT = Object.freeze({
  floorCornerRadius: FLOOR_CORNER_RADIUS,
  ashOpacity: 0.17,
  borderWidth: 1.25,
  borderOpacity: 0.6,
  keylineWidth: 3,
  keylineOpacity: 0.5,
  liteBorderWidth: 2,
  liteKeylineWidth: 4,
  liteBelow: 20,
  stoneWidthUnits: 0.2,
  stoneHeightUnits: 0.27,
  stoneBreakUnits: 0.16,
  markSize: 12,
  liteMarkSize: 14,
  homeFallAsh: 0.05,
  homeFallBone: 0.35,
  homeFallFade: 120,
});

/**
 * The BattleClient vision page's field scale: 50 px per world unit. Live play
 * (records input.vision) multiplies the border and gravestone strokes and the
 * skull-and-"×2" mark (its size and its clear box) by the field's pixels per
 * unit over this, so they keep their size beside the field at any window
 * size; the Replay Viewer keeps the store's fixed pixel sizes.
 */
export const VISION_FIELD = Object.freeze({
  pixelsPerUnit: 50,
});

/**
 * Node allowance. A durable strip record builds at most 7 nodes (the skull mark:
 * a group of 4 paths and one text). A Home Fall row of the plan (kind
 * "home_fall") is drawn here by beat-held durable nodes, never by the painter, so
 * it adds no painter animation; its 3 nodes are counted for the plan bound.
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 7,
  animationsPerRecord: 0,
  persistentNodesPerRecord: 0,
});

/**
 * Red Zone Floors switches the Grave Ground (off: the plain edge line). Red Zone
 * Home Fall switches the Home Fall glow; this is the tag the plan's derived
 * "home_fall" family asks about.
 */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "red_zone_floor" }, "red_zone_floors"],
  [{ surface: "transient", kind: "home_fall" }, "red_zone_home_fall"],
]);

/**
 * Read one switch from a visual filter state (null: the default). Pure.
 *
 * @param {Readonly<Record<string, boolean>> | null | undefined} filters
 * @param {string} id
 * @param {boolean} fallback
 */
function switchOn(filters, id, fallback) {
  const value = filters?.[id];
  return typeof value === "boolean" ? value : fallback;
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
 * Return the world distance from (x, y) to a recorded stone's edge (negative
 * inside). Rows that are not pillars or walls give Infinity. Pure.
 *
 * @param {Record<string, any>} stone
 * @param {number} x
 * @param {number} y
 */
function stoneDistance(stone, x, y) {
  const [cx, cy] = asArray(stone.center).map((value) => finiteNumber(value));
  if (stone.kind === "pillar") {
    return Math.hypot(x - cx, y - cy) - finiteNumber(stone.radius);
  }
  if (stone.kind !== "wall") return Number.POSITIVE_INFINITY;
  const theta = finiteNumber(stone.theta);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const dx = x - cx;
  const dy = y - cy;
  const lx = dx * c + dy * s;
  const ly = -dx * s + dy * c;
  return Math.hypot(
    Math.max(Math.abs(lx) - finiteNumber(stone.width) / 2, 0),
    Math.max(Math.abs(ly) - finiteNumber(stone.height) / 2, 0),
  );
}

/**
 * Return each team's strip from the recorded Red Zone: team 1 from
 * team_a_x_range, team 2 from team_b_x_range, clipped to [0, width]. A strip
 * touching exactly one map edge has an inner edge (x of the rule line) and dir
 * (-1 when its home is the left end, +1 the right end). A collapsed strip is
 * left out. Pure.
 *
 * @param {Record<string, any>} redZone
 * @param {number} width
 */
function teamStrips(redZone, width) {
  const eps = 1e-6;
  /** @type {Array<{teamId: number, x0: number, x1: number, atLeft: boolean, atRight: boolean, inner: number | null, dir: number, back: number}>} */
  const strips = [];
  for (const [teamId, key] of /** @type {Array<[number, string]>} */ ([
    [1, "team_a_x_range"],
    [2, "team_b_x_range"],
  ])) {
    const range = asArray(redZone[key]).map((value) => finiteNumber(value, Number.NaN));
    if (range.length !== 2 || range.some((value) => !Number.isFinite(value))) continue;
    const x0 = Math.min(Math.max(Math.min(range[0], range[1]), 0), width);
    const x1 = Math.min(Math.max(Math.max(range[0], range[1]), 0), width);
    if (x1 - x0 <= eps) continue;
    const atLeft = x0 <= eps;
    const atRight = x1 >= width - eps;
    const inner = atLeft && !atRight ? x1 : atRight && !atLeft ? x0 : null;
    const dir = atLeft && !atRight ? -1 : atRight && !atLeft ? 1 : 0;
    strips.push({
      teamId,
      x0,
      x1,
      atLeft,
      atRight,
      inner,
      dir,
      back: dir < 0 ? x0 : x1,
    });
  }
  return strips;
}

/**
 * Return the strip's floor path: a rectangle whose outer corners (on the map
 * edge) follow the floor's rounded corners. Pure.
 *
 * @param {LookTransform} transform
 * @param {number} x0
 * @param {number} x1
 * @param {number} height World height.
 * @param {boolean} roundLeft
 * @param {boolean} roundRight
 */
function stripPath(transform, x0, x1, height, roundLeft, roundRight) {
  const topLeft = transform.worldToScreen([x0, height]);
  const bottomRight = transform.worldToScreen([x1, 0]);
  const left = Math.min(topLeft.x, bottomRight.x);
  const right = Math.max(topLeft.x, bottomRight.x);
  const top = Math.min(topLeft.y, bottomRight.y);
  const bottom = Math.max(topLeft.y, bottomRight.y);
  const limit = Math.min((right - left) / 2, (bottom - top) / 2);
  const rl = roundLeft ? Math.min(FLOOR_CORNER_RADIUS, limit) : 0;
  const rr = roundRight ? Math.min(FLOOR_CORNER_RADIUS, limit) : 0;
  let d = `M${fixed(left + rl)} ${fixed(top)}H${fixed(right - rr)}`;
  if (rr) d += `A${fixed(rr)} ${fixed(rr)} 0 0 1 ${fixed(right)} ${fixed(top + rr)}`;
  d += `V${fixed(bottom - rr)}`;
  if (rr) d += `A${fixed(rr)} ${fixed(rr)} 0 0 1 ${fixed(right - rr)} ${fixed(bottom)}`;
  d += `H${fixed(left + rl)}`;
  if (rl) d += `A${fixed(rl)} ${fixed(rl)} 0 0 1 ${fixed(left)} ${fixed(bottom - rl)}`;
  d += `V${fixed(top + rl)}`;
  if (rl) d += `A${fixed(rl)} ${fixed(rl)} 0 0 1 ${fixed(left + rl)} ${fixed(top)}`;
  return `${d}Z`;
}

/**
 * Apply the store's stop rule on the rule line x = X: return its free runs
 * ([y0, y1] world pairs, sampled every 0.02 units, where no stone is within c =
 * max(0.1, 4 px) of the line) and the gravestone spots (y = k + 0.5 clear by
 * c + 0.12). Pure.
 *
 * @param {number} lineX
 * @param {number} height
 * @param {ReadonlyArray<Record<string, any>>} stones
 * @param {number} ppu Screen px per world unit.
 */
function borderGeometry(lineX, height, stones, ppu) {
  const clearance = Math.max(0.1, 4 / ppu);
  /** @param {number} y */
  const distance = (y) =>
    stones.reduce(
      (best, stone) => Math.min(best, stoneDistance(stone, lineX, y)),
      Infinity,
    );
  /** @type {Array<[number, number]>} */
  const runs = [];
  /** @type {[number, number] | null} */
  let run = null;
  const steps = Math.max(1, Math.round(height / 0.02));
  for (let index = 0; index <= steps; index += 1) {
    const y = (height * index) / steps;
    if (distance(y) >= clearance) {
      if (run === null) run = [y, y];
      else run[1] = y;
    } else if (run !== null) {
      runs.push(run);
      run = null;
    }
  }
  if (run !== null) runs.push(run);
  /** @type {number[]} */
  const stoneSpots = [];
  for (let k = 0; k + 0.5 <= height; k += 1) {
    if (distance(k + 0.5) >= clearance + 0.12) stoneSpots.push(k + 0.5);
  }
  return { runs, stoneSpots };
}

/**
 * Find clear spots for a home's marks (world points): a top one and a bottom one,
 * or only the top one when they would sit within one unit. The mark's box (28 by
 * 16 px, 32 by 18 px when lite, times scale) must stay inside the strip and the
 * floor, 0.15 units from every stone and 0.77 units from every spawn pad. This
 * is the store's clearSpots (skins/red-zone.js). scale (default 1) multiplies
 * every pixel size here (the box, the 12 px inset and the 10 px edge), so live
 * play's field-scaled mark gets the store's spots in world units. Pure.
 *
 * @param {{x0: number, x1: number, inner: number | null, dir: number, back: number, atLeft: boolean}} strip
 * @param {{inner: number | null, dir: number} | undefined} other The other team's strip.
 * @param {number} height
 * @param {ReadonlyArray<Record<string, any>>} stones
 * @param {ReadonlyArray<Pair>} pads
 * @param {number} ppu
 * @param {boolean} lite
 * @param {number} [scale] Mark scale (live play: the field's px per unit over 50).
 * @returns {Pair[]}
 */
function markSpots(strip, other, height, stones, pads, ppu, lite, scale = 1) {
  const halfWidth = ((lite ? 32 : 28) * scale) / 2 / ppu;
  const halfHeight = ((lite ? 18 : 16) * scale) / 2 / ppu;
  const inset = Math.max(0.5, (12 * scale) / ppu);
  const edge = Math.max(0.45, (10 * scale) / ppu);
  /** @param {number} x @param {number} y */
  const clearAt = (x, y) => {
    if (
      x - halfWidth < strip.x0 - 1e-6 ||
      x + halfWidth > strip.x1 + 1e-6 ||
      y - halfHeight < 0 ||
      y + halfHeight > height
    ) {
      return false;
    }
    const points = [
      [x, y],
      [x - halfWidth, y - halfHeight],
      [x + halfWidth, y - halfHeight],
      [x - halfWidth, y + halfHeight],
      [x + halfWidth, y + halfHeight],
      [x, y - halfHeight],
      [x, y + halfHeight],
      [x - halfWidth, y],
      [x + halfWidth, y],
    ];
    for (const stone of stones) {
      for (const [px, py] of points) {
        if (
          stoneDistance(stone, px, py) < 0.15 ||
          stoneDistance(stone, px, py + 0.06) < 0.15
        ) {
          return false;
        }
      }
    }
    for (const [qx, qy] of pads) {
      for (const [px, py] of points) {
        if (Math.hypot(px - qx, py - qy) < 0.77) return false;
      }
    }
    return true;
  };
  let xs;
  if (strip.inner !== null) {
    const inner = strip.inner;
    xs = [0, 0.5, 1].map((extra) => inner + strip.dir * (inset + extra));
    if (
      other?.inner != null &&
      Math.abs(xs[0] - (other.inner + other.dir * inset)) < 1
    ) {
      xs = [0, 0.5, 1].map((extra) => strip.back - strip.dir * (1.5 + extra));
    }
  } else {
    xs = [0, 0.5, 1].map(
      (extra) => strip.back - (strip.atLeft ? -1 : 1) * (1.5 + extra),
    );
  }
  xs = xs.filter(
    (x) => Math.abs(x - strip.back) >= 0.3 && x > strip.x0 && x < strip.x1,
  );
  /** @param {number} from @param {number} to @returns {Pair | null} */
  const search = (from, to) => {
    const step = from < to ? 0.25 : -0.25;
    for (const x of xs) {
      for (let y = from; step > 0 ? y <= to + 1e-6 : y >= to - 1e-6; y += step) {
        if (clearAt(x, y)) return [x, y];
      }
    }
    return null;
  };
  const top = search(height - edge, height / 2);
  const bottom = search(edge, height / 2);
  /** @type {Pair[]} */
  const spots = [];
  if (top) spots.push(top);
  if (bottom && !(top && Math.abs(top[1] - bottom[1]) < 1)) spots.push(bottom);
  return lite ? spots.slice(0, 1) : spots;
}

/**
 * Return a small gravestone (rounded top) centred on (x, y), w by h px. Pure.
 *
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 */
function gravestonePath(x, y, w, h) {
  const r = w / 2;
  const top = y - h / 2;
  const bottom = y + h / 2;
  return `M${fixed(x - r)} ${fixed(bottom)}V${fixed(top + r)}A${fixed(r)} ${fixed(r)} 0 0 1 ${fixed(x + r)} ${fixed(top + r)}V${fixed(bottom)}Z`;
}

/**
 * Return the skull rune's path data, s px tall, centred near (x, y): head
 * (cranium and jaw), eyes and teeth. Pure.
 *
 * @param {number} x
 * @param {number} y
 * @param {number} s
 */
function skullPaths(x, y, s) {
  /** @param {number} cx @param {number} cy @param {number} r */
  const circle = (cx, cy, r) =>
    `M${fixed(cx - r)} ${fixed(cy)}A${fixed(r)} ${fixed(r)} 0 1 1 ${fixed(cx + r)} ${fixed(cy)}A${fixed(r)} ${fixed(r)} 0 1 1 ${fixed(cx - r)} ${fixed(cy)}Z`;
  const jx = x - 0.22 * s;
  const jy = y + 0.12 * s;
  const jw = 0.44 * s;
  const jh = 0.3 * s;
  const rx = 0.06 * s;
  const head =
    circle(x, y - 0.1 * s, 0.36 * s) +
    `M${fixed(jx + rx)} ${fixed(jy)}H${fixed(jx + jw - rx)}A${fixed(rx)} ${fixed(rx)} 0 0 1 ${fixed(jx + jw)} ${fixed(jy + rx)}V${fixed(jy + jh - rx)}A${fixed(rx)} ${fixed(rx)} 0 0 1 ${fixed(jx + jw - rx)} ${fixed(jy + jh)}` +
    `H${fixed(jx + rx)}A${fixed(rx)} ${fixed(rx)} 0 0 1 ${fixed(jx)} ${fixed(jy + jh - rx)}V${fixed(jy + rx)}A${fixed(rx)} ${fixed(rx)} 0 0 1 ${fixed(jx + rx)} ${fixed(jy)}Z`;
  const eyes =
    circle(x - 0.14 * s, y - 0.08 * s, 0.1 * s) +
    circle(x + 0.14 * s, y - 0.08 * s, 0.1 * s);
  const teeth = `M${fixed(x - 0.07 * s)} ${fixed(y + 0.3 * s)}v${fixed(0.12 * s)}M${fixed(x + 0.07 * s)} ${fixed(y + 0.3 * s)}v${fixed(0.12 * s)}`;
  return { head, eyes, teeth, toothWidth: Math.max(0.6, 0.06 * s) };
}

/**
 * Return the drawing records for both homes.
 *
 * input.transform projects and gives the scale; input.map gives the world width
 * (and optional height and obstacles; height defaults to the transform's world
 * height); input.redZone is the recorded `map.red_zone` record ({team_a_x_range,
 * team_b_x_range}) or null for a map without a Red Zone (then return []);
 * input.filters is the visual filter state (null: defaults, red_zone_floors and
 * red_zone_home_fall on); input.scene is this audience's arena scene, read only
 * for `spawn_pads` positions (mark spots) and `agents` (Home Fall victims);
 * input.homeFallIds lists the public IDs of this tick's falls worth 2 points
 * that this audience saw (default none).
 *
 * Return a frozen list, all numbers in screen px:
 * - Red Zone Floors off: {kind: "red-zone-line", teamId, x1, y1, x2, y2} per
 *   strip with an inner edge.
 * - Otherwise: {kind: "grave-ash", d} for both floors; per lit team
 *   {kind: "home-fall-ash", teamId, d, beatKey, opacity: 0.05}; per strip with an
 *   inner edge {kind: "grave-border", teamId, lineX, runs, stones, bars, lite}
 *   (runs: vertical line path; stones: gravestone path, empty when lite; bars:
 *   their cross-bars); per lit team {kind: "home-fall-border", teamId, runs,
 *   stones, beatKey, opacity: 0.35, lite}; then {kind: "grave-mark", teamId, x,
 *   y, dir, size} per mark spot (x, y: the mark's center; dir: -1 for a left
 *   home, +1 for a right home; size: the "×2" font size, 12 px or 14 px when
 *   lite). With input.vision true (live play: the BattleClient and the
 *   DevClient) widthScale is the field's pixels per unit over 50: the border
 *   records carry it for their strokes, and each mark's size and clear box
 *   are multiplied by it (the spots stay the store's in world units). Without
 *   it (the default; the Replay Viewer) widthScale is 1.
 * beatKey is the presentation key of the team's first seen home fall in public
 * ID order (the same order as the Fall beat). Pure data: no DOM. Agent POV: the
 * homes are identical in every view; a Home Fall needs a seen victim.
 *
 * @param {{
 *   transform?: LookTransform | null,
 *   map?: {width?: number, height?: number, obstacles?: unknown} | null,
 *   redZone?: Readonly<{team_a_x_range: readonly number[], team_b_x_range: readonly number[]}> | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   scene?: Record<string, any> | null,
 *   homeFallIds?: ReadonlyArray<string> | null,
 *   vision?: boolean,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const transform = input.transform;
  const redZone = input.redZone;
  if (!transform || !isRecord(redZone)) {
    return Object.freeze([]);
  }
  const width = finiteNumber(input.map?.width, transform.worldWidth);
  const height = finiteNumber(input.map?.height, transform.worldHeight);
  const strips = teamStrips(redZone, width);
  const ppu = transform.worldLengthToScreen(1);
  const vision = input.vision === true;
  const lite = !vision && ppu < FOOTPRINT.liteBelow;
  const widthScale = vision ? ppu / VISION_FIELD.pixelsPerUnit : 1;
  /** @param {number} x */
  const screenX = (x) => transform.worldToScreen([x, 0]).x;
  /** @param {number} y */
  const screenY = (y) => transform.worldToScreen([0, y]).y;
  /** @type {LookRecord[]} */
  const result = [];
  if (!switchOn(input.filters, "red_zone_floors", true)) {
    for (const strip of strips) {
      if (strip.inner === null) continue;
      const x = screenX(strip.inner);
      result.push(
        Object.freeze({
          kind: "red-zone-line",
          teamId: strip.teamId,
          x1: x,
          y1: screenY(height),
          x2: x,
          y2: screenY(0),
        }),
      );
    }
    return Object.freeze(result);
  }
  const stones = asArray(input.map?.obstacles).filter(isRecord);
  const pads = /** @type {Pair[]} */ (
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
  // Home Fall: only victims this audience saw (their body is in its scene).
  /** @type {Map<number, string>} */
  const beatKeyByTeam = new Map();
  if (switchOn(input.filters, "red_zone_home_fall", true)) {
    const sceneAgents = new Map(
      asArray(input.scene?.agents)
        .filter((agent) => isRecord(agent) && typeof agent.public_agent_id === "string")
        .map((agent) => [String(agent.public_agent_id), agent]),
    );
    const ids = [
      ...new Set(asArray(input.homeFallIds).filter((id) => typeof id === "string")),
    ].sort((left, right) => left.localeCompare(right, "en", { numeric: true }));
    for (const id of ids) {
      const agent = sceneAgents.get(id);
      const teamId = Number(agent?.team_id);
      if (
        (teamId === 1 || teamId === 2) &&
        typeof agent?.presentation_key === "string" &&
        !beatKeyByTeam.has(teamId)
      ) {
        beatKeyByTeam.set(teamId, agent.presentation_key);
      }
    }
  }
  result.push(
    Object.freeze({
      kind: "grave-ash",
      d: strips
        .map((strip) =>
          stripPath(transform, strip.x0, strip.x1, height, strip.atLeft, strip.atRight),
        )
        .join(""),
    }),
  );
  for (const strip of strips) {
    const beatKey = beatKeyByTeam.get(strip.teamId);
    if (beatKey !== undefined) {
      result.push(
        Object.freeze({
          kind: "home-fall-ash",
          teamId: strip.teamId,
          d: stripPath(
            transform,
            strip.x0,
            strip.x1,
            height,
            strip.atLeft,
            strip.atRight,
          ),
          beatKey,
          opacity: FOOTPRINT.homeFallAsh,
        }),
      );
    }
  }
  for (const strip of strips) {
    if (strip.inner === null) continue;
    const geometry = borderGeometry(strip.inner, height, stones, ppu);
    let runs = geometry.runs;
    if (!lite) {
      for (const y of geometry.stoneSpots) {
        runs = runs.flatMap(([a, b]) =>
          y + FOOTPRINT.stoneBreakUnits < a || y - FOOTPRINT.stoneBreakUnits > b
            ? [[a, b]]
            : [
                [a, y - FOOTPRINT.stoneBreakUnits],
                [y + FOOTPRINT.stoneBreakUnits, b],
              ],
        );
      }
    }
    const x = screenX(strip.inner);
    const runPath = runs
      .filter(([a, b]) => b - a > 0.05)
      .map(([a, b]) => `M${fixed(x)} ${fixed(screenY(b))}V${fixed(screenY(a))}`)
      .join("");
    const stonePath = lite
      ? ""
      : geometry.stoneSpots
          .map((y) =>
            gravestonePath(
              x,
              screenY(y),
              transform.worldLengthToScreen(FOOTPRINT.stoneWidthUnits),
              transform.worldLengthToScreen(FOOTPRINT.stoneHeightUnits),
            ),
          )
          .join("");
    const barHalf = transform.worldLengthToScreen(0.07);
    const bars = lite
      ? ""
      : geometry.stoneSpots
          .map(
            (y) => `M${fixed(x - barHalf)} ${fixed(screenY(y))}H${fixed(x + barHalf)}`,
          )
          .join("");
    result.push(
      Object.freeze({
        kind: "grave-border",
        teamId: strip.teamId,
        lineX: x,
        runs: runPath,
        stones: stonePath,
        bars,
        lite,
        widthScale,
      }),
    );
    const beatKey = beatKeyByTeam.get(strip.teamId);
    if (beatKey !== undefined) {
      result.push(
        Object.freeze({
          kind: "home-fall-border",
          teamId: strip.teamId,
          runs: runPath,
          stones: stonePath,
          beatKey,
          opacity: FOOTPRINT.homeFallBone,
          lite,
        }),
      );
    }
  }
  // The skull and "×2" marks are the store's in every client: the skull joined
  // to the "×2" on clear spots off the stones, pads and gravestones. Live play
  // scales the mark (and its clear box) with the field, as it scales the
  // border, so the mark keeps the store's size beside the gravestones.
  for (const strip of strips) {
    const other = strips.find((candidate) => candidate.teamId !== strip.teamId);
    const spots = markSpots(strip, other, height, stones, pads, ppu, lite, widthScale);
    for (const [wx, wy] of spots) {
      const point = transform.worldToScreen([wx, wy]);
      result.push(
        Object.freeze({
          kind: "grave-mark",
          teamId: strip.teamId,
          x: point.x,
          y: point.y,
          dir: strip.dir,
          size: (lite ? FOOTPRINT.liteMarkSize : FOOTPRINT.markSize) * widthScale,
        }),
      );
    }
  }
  return Object.freeze(result);
}

/**
 * Write the beat-hold attributes of a Home Fall node: shown from the Fall beat
 * of beatKey with a 120 ms fade from 0 to its END opacity. DOM change only.
 *
 * @param {Element} node
 * @param {LookRecord} record
 */
function holdForFall(node, record) {
  setNodeAttributes(node, {
    opacity: record.opacity,
    "data-beat-hold": "fall",
    "data-beat-side": "after",
    "data-beat-key": record.beatKey,
    "data-beat-from": 0,
    "data-beat-fade": FOOTPRINT.homeFallFade,
  });
}

/**
 * Build one record's group of nodes (empty array for an unknown kind). DOM only.
 *
 * @param {LookRecord} record
 * @param {Document} doc
 * @returns {SVGElement | null}
 */
function buildOne(record, doc) {
  switch (record.kind) {
    case "red-zone-line":
      return svgNode(doc, "line", { class: "red-zone-line", "aria-hidden": "true" });
    case "grave-ash":
      return svgNode(doc, "path", {
        class: "red-zone-floor grave-ash",
        "aria-hidden": "true",
      });
    case "home-fall-ash": {
      const node = svgNode(doc, "path", {
        class: "home-fall-ash",
        "aria-hidden": "true",
      });
      holdForFall(node, record);
      return node;
    }
    case "grave-border": {
      const group = svgNode(doc, "g", {
        class: "grave-border",
        "data-team-id": record.teamId,
        "aria-hidden": "true",
      });
      group.append(
        svgNode(doc, "path", { class: "grave-border__keyline" }),
        svgNode(doc, "path", { class: "grave-border__bone" }),
        svgNode(doc, "path", { class: "grave-border__stone-shadow" }),
        svgNode(doc, "path", { class: "grave-border__stone" }),
        svgNode(doc, "path", { class: "grave-border__bar" }),
      );
      return group;
    }
    case "home-fall-border": {
      const group = svgNode(doc, "g", {
        class: "home-fall-border",
        "data-team-id": record.teamId,
        "aria-hidden": "true",
      });
      group.append(
        svgNode(doc, "path", { class: "home-fall-border__line" }),
        svgNode(doc, "path", { class: "home-fall-border__stone" }),
      );
      holdForFall(group, record);
      return group;
    }
    case "grave-mark": {
      const group = svgNode(doc, "g", {
        class: "grave-mark",
        "data-team-id": record.teamId,
        role: "img",
        "aria-label": "x2 Mark",
      });
      group.append(
        svgNode(doc, "path", { class: "grave-mark__halo" }),
        svgNode(doc, "path", { class: "grave-mark__head" }),
        svgNode(doc, "path", { class: "grave-mark__eyes" }),
        svgNode(doc, "path", { class: "grave-mark__teeth" }),
        svgNode(doc, "text", {
          class: "grave-mark__text",
          "text-anchor": "middle",
          "dominant-baseline": "central",
        }),
      );
      const text = group.lastChild;
      if (text) text.textContent = "×2";
      registerTooltipOwner(group, tooltip(record));
      return group;
    }
    default:
      return null;
  }
}

/**
 * Write one record's geometry onto its node or group (build and update). DOM only.
 *
 * @param {Element} node
 * @param {LookRecord} record
 */
function writeGeometry(node, record) {
  const parts = [...node.children];
  switch (record.kind) {
    case "red-zone-line":
      setNodeAttributes(node, {
        x1: record.x1,
        y1: record.y1,
        x2: record.x2,
        y2: record.y2,
      });
      break;
    case "grave-ash":
    case "home-fall-ash":
      setNodeAttributes(node, { d: record.d });
      break;
    case "grave-border": {
      const scale = Number.isFinite(record.widthScale) ? record.widthScale : 1;
      const lineWidth =
        (record.lite ? FOOTPRINT.liteBorderWidth : FOOTPRINT.borderWidth) * scale;
      const keyWidth =
        (record.lite ? FOOTPRINT.liteKeylineWidth : FOOTPRINT.keylineWidth) * scale;
      setNodeAttributes(parts[0], { d: record.runs || null, "stroke-width": keyWidth });
      setNodeAttributes(parts[1], {
        d: record.runs || null,
        "stroke-width": lineWidth,
      });
      setNodeAttributes(parts[2], { d: record.stones || null });
      setNodeAttributes(parts[3], { d: record.stones || null });
      setNodeAttributes(parts[4], { d: record.bars || null });
      break;
    }
    case "home-fall-border":
      setNodeAttributes(parts[0], {
        d: record.runs || null,
        "stroke-width": record.lite ? FOOTPRINT.liteBorderWidth : FOOTPRINT.borderWidth,
      });
      setNodeAttributes(parts[1], { d: record.stones || null });
      break;
    case "grave-mark": {
      // The store's graveMarks: the skull (1.05 x size tall) on the home's back
      // side and the "×2" toward the border, joined around the mark's center.
      const offset = record.size * 0.95 * (record.dir < 0 ? 1 : -1);
      const skull = skullPaths(record.x - offset * 0.55, record.y, record.size * 1.05);
      setNodeAttributes(parts[0], { d: skull.head });
      setNodeAttributes(parts[1], { d: skull.head });
      setNodeAttributes(parts[2], { d: skull.eyes });
      setNodeAttributes(parts[3], { d: skull.teeth, "stroke-width": skull.toothWidth });
      setNodeAttributes(parts[4], {
        x: record.x + offset * 0.5,
        y: record.y,
        "font-size": record.size,
      });
      break;
    }
    default:
      break;
  }
}

/**
 * Build the Grave Ground nodes, in record order, for the `red-zone` layer.
 *
 * records come from records(); doc creates the nodes. Kinds become: a
 * `line.red-zone-line`; a `path.grave-ash`; a beat-held `path.home-fall-ash`; a
 * `g.grave-border` (keyline, bone line, gravestone shadow, gravestone and
 * cross-bar paths); a beat-held `g.home-fall-border` (line and filled
 * gravestones); a `g.grave-mark` (skull halo, head, eyes, teeth and the "×2"
 * text) with the "x2 Mark" Symbol Tip registered through `tooltip.js` (a DOM side
 * effect). Beat-held nodes carry data-beat-hold "fall", data-beat-side "after",
 * data-beat-key, data-beat-from 0 and data-beat-fade 120 and hold their END
 * opacity. Export-safe: flat fills, strokes and opacity; no ids, clip paths or
 * gradients. Return the nodes.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @returns {SVGElement[]}
 */
export function build(records, doc) {
  /** @type {SVGElement[]} */
  const nodes = [];
  for (const record of records) {
    const node = buildOne(record, doc);
    if (node === null) continue;
    writeGeometry(node, record);
    nodes.push(node);
  }
  return nodes;
}

/**
 * Move existing Grave Ground nodes to new records (reprojection).
 *
 * nodes are build()'s nodes and records the new records in the same order and
 * kinds. Write the new geometry; beat holds and tooltips are unchanged. Return
 * undefined; DOM change only.
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
 * Return the Symbol Tip for a record: the "x2 Mark" tip for a mark, null for
 * anything else (floors, borders and lines have no tip). Pure; the same in every
 * view.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  if (record?.kind !== "grave-mark") {
    return null;
  }
  return createSymbolDescriptor({
    symbol: "red_zone_x2",
    id: `red-zone-mark:${record.teamId}`,
    facts: [
      { label: "Home Of", value: teamLabelFromId(record.teamId) ?? "Unavailable" },
    ],
  });
}
