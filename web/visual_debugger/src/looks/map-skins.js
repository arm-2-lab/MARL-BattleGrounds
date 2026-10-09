/**
 * @file The Map Skins look (Realm Skins): the arena floor in the `map` layer.
 *
 * It draws, bottom to top: the Night Surround apron; the floor (the map's tale
 * floor from `map-skin-data.js` when its layout matches, otherwise its realm
 * kit's flat floor with at most one sparse inlay); Moonlight (a soft pool over
 * the middle, a sinking hollow on Tidewrack); the Floor Grid (coursed stone
 * joints on whole units and a dashed seam at the mirror line, broken 0.25
 * units around thin walls lying on them); the Spawn Waystones floor marks
 * (Hearth Seals on active pads and Home Gates just outside the floor edge);
 * the Arena Rim parapet when on; the Survey Frame coordinates when on; and the
 * floor edge line (`rect.map-boundary`). The live clients' combat debugger asks
 * for a one-unit grid (`showUnitGrid`); it replaces the coursed joints there.
 *
 * Callers: `scene.js` calls records, then build, and puts the nodes at the
 * bottom of the arena, under the Red Zone and everything else. `obstacles.js`
 * may call taleStoneRecords, then build, to draw a map's tale stones.
 * `visual-filters.js` reads PAINT_PARTS. The maintainers' map-skin converter
 * imports mapGeometryKey so the data and the browser share one key rule.
 *
 * Static: nothing here moves or reads a beat. records caches its result by
 * map, geometry, pads, Red Zone, switches, audience, viewport and camera
 * frame, and build caches its nodes by records, so the renderer's frequent
 * re-renders reuse the same nodes. Every node is export-safe: flat fills,
 * strokes, opacity and transforms only.
 *
 * Boundary: the floor is the same in every view. records reads only the map
 * record of the arena scene (size, stones, Red Zone strips), the spawn pads'
 * positions, teams and whether each is active, the map's technical name and
 * the switches. It never reads a champion, a researcher view or a match
 * summary. One view rule only: Map 47's Legion Mist is left out while the
 * Sight Reach veil shows in Agent POV, so a still never suggests a vision
 * limit that the veil does not draw.
 */
import {
  asArray,
  finiteNumber,
  isRecord,
  paintParts,
  setNodeAttributes,
  svgNode,
} from "./look-support.js";
import {
  MAP_SKIN_ENTRIES,
  MAP_SKIN_KITS,
  MAP_SKIN_MATERIALS,
  MAX_TALE_NODES,
  mapTaleSkin,
  TALE_FLOOR_SWITCHES,
  TALE_REFERENCE_SCALE,
  TALE_STONE_SWITCHES,
  taleFloorTree,
  taleStoneTree,
} from "./map-skin-data.js";

/**
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {import("./map-skin-data.js").TaleNode} TaleNode
 * @typedef {{x: number, y: number, team: "A" | "B"}} FloorPad
 * @typedef {{
 *   kind: "pillar" | "wall",
 *   x: number,
 *   y: number,
 *   r: number,
 *   w: number,
 *   h: number,
 *   theta: number,
 * }} FloorStone
 */

/**
 * Legacy fixed corner radius of the floor, in CSS pixels, kept only as a
 * listed store number (FOOTPRINT here and in `red-zone.js`). The Realm Skins
 * floor and the Red Zone strips both use floorCornerRadius.
 */
export const FLOOR_CORNER_RADIUS = 8;

/**
 * Return the floor's corner radius in CSS pixels at a given scale.
 *
 * scale is pixels per world unit (a positive number). The radius is 0.12 world
 * units, never under 2 pixels, as in the store's Realm Skins floor. Pure.
 *
 * @param {number} scale
 * @returns {number}
 */
export function floorCornerRadius(scale) {
  return Math.max(2, 0.12 * finiteNumber(scale));
}

/**
 * Drawing sizes: the legacy floor corner radius in CSS pixels and the floor's
 * real corner radius in world units (see floorCornerRadius).
 */
export const FOOTPRINT = Object.freeze({
  floorCornerRadius: FLOOR_CORNER_RADIUS,
  floorCornerRadiusUnits: 0.12,
});

/**
 * One node per record, except a tale record, which holds a whole stored tree
 * (at most MAX_TALE_NODES nodes plus its group). No animations.
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: MAX_TALE_NODES + 1,
  animationsPerRecord: 0,
  persistentNodesPerRecord: 0,
});

/** The floor parts and the switch that controls each. */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "map_skin" }, "map_skins"],
  [{ surface: "durable", kind: "moonlight" }, "moonlight"],
  [{ surface: "durable", kind: "floor_grid" }, "floor_grid"],
  [{ surface: "durable", kind: "survey_frame" }, "survey_frame"],
  [{ surface: "durable", kind: "arena_rim" }, "arena_rim"],
]);

/** Moon Pool bands: [width, height] in world units and opacity, outer first. */
const MOON_BANDS = Object.freeze([
  [19.2, 9.4, 0.006],
  [17.06, 8.35, 0.011],
  [16.06, 7.86, 0.017],
  [15.22, 7.45, 0.022],
  [14.46, 7.08, 0.028],
  [13.73, 6.72, 0.034],
  [12.99, 6.36, 0.039],
  [12.23, 5.99, 0.045],
  [11.4, 5.58, 0.05],
  [10.4, 5.09, 0.056],
]);

/** Floor vignette bands: [inset in world units, opacity], outer first. */
const VIGNETTE_BANDS = Object.freeze([
  [0.7, 0.1],
  [0.4, 0.12],
  [0.18, 0.16],
]);

/** How many recent records results and built node sets to keep. */
const CACHE_SIZE = 8;

/** @type {Map<string, ReadonlyArray<LookRecord>>} */
const RECORDS_CACHE = new Map();

/** @type {WeakMap<ReadonlyArray<LookRecord>, {doc: Document, nodes: SVGElement[]}>} */
const BUILT = new WeakMap();

/** @type {WeakMap<Element, ReadonlyArray<LookRecord>>} */
const BUILT_OWNER = new WeakMap();

/** @type {Map<string, string> | null} geometry key -> technical name. */
let geometryIndex = null;

/**
 * Round to two decimals, as the store's SVG writer does.
 *
 * @param {number} value
 * @returns {number}
 */
function f(value) {
  return Math.round(value * 100) / 100;
}

/**
 * The store's deterministic texture noise (xorshift), seeded.
 *
 * @param {number} seed
 * @returns {() => number} Numbers in [0, 1) on each call.
 */
function seededNoise(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return ((state >>> 0) % 100000) / 100000;
  };
}

/**
 * Return the geometry key of a map record: one short text that is equal for
 * two maps exactly when their size and stones are the same.
 *
 * map is the arena scene's `map` record: width and height (world units) and
 * obstacles, each {kind: "pillar" | "wall", center: [x, y], radius, width,
 * height, theta} (world units and radians). Stone order does not matter.
 * Numbers are rounded to 3 decimals (theta to 4), so float32 recordings match
 * the float64 catalogue. Return "fnv1a:" plus 8 hex digits; a missing or
 * malformed map returns the key of an empty 0 x 0 map. Pure; the converter
 * uses the same function, so the data and the browser agree.
 *
 * @param {unknown} map
 * @returns {string}
 */
export function mapGeometryKey(map) {
  const record = isRecord(map) ? map : {};
  const rows = asArray(record.obstacles)
    .filter(isRecord)
    .map((obstacle) => {
      const center = asArray(obstacle.center);
      const parts = [
        obstacle.kind === "pillar" ? "p" : "w",
        finiteNumber(center[0]).toFixed(3),
        finiteNumber(center[1]).toFixed(3),
        finiteNumber(obstacle.radius).toFixed(3),
        finiteNumber(obstacle.width).toFixed(3),
        finiteNumber(obstacle.height).toFixed(3),
        finiteNumber(obstacle.theta).toFixed(4),
      ];
      return parts.join(",");
    })
    .sort();
  const text = `${finiteNumber(record.width).toFixed(3)}x${finiteNumber(record.height).toFixed(3)}|${rows.join(";")}`;
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return `fnv1a:${hash.toString(16).padStart(8, "0")}`;
}

/**
 * Find the skin entry for a map: by technical name first, else by geometry.
 *
 * mapKey is the technical name (a string) or null. map is the scene's map
 * record. A known name wins. With no known name, a catalogue layout with at
 * least one stone and the same geometry key is used (an empty floor is never
 * matched by geometry alone). Return {key, entry, geometry, layoutMatches}
 * where layoutMatches is true when the entry has a tale and its catalogue
 * layout equals this map; key and entry are null when nothing matches. Pure.
 *
 * @param {unknown} mapKey
 * @param {unknown} map
 * @returns {{key: string | null, entry: ReturnType<typeof mapTaleSkin>, geometry: string, layoutMatches: boolean}}
 */
export function resolveMapSkin(mapKey, map) {
  const geometry = mapGeometryKey(map);
  let key = mapTaleSkin(mapKey) ? /** @type {string} */ (mapKey) : null;
  if (key === null && asArray(isRecord(map) ? map.obstacles : null).length > 0) {
    if (geometryIndex === null) {
      /** @type {Map<string, string>} */
      const counts = new Map();
      geometryIndex = new Map();
      for (const [name, entry] of Object.entries(MAP_SKIN_ENTRIES)) {
        counts.set(entry.geometry, (counts.get(entry.geometry) ?? "") + name);
        geometryIndex.set(entry.geometry, name);
      }
      for (const [geometryKey, names] of counts) {
        if (names !== geometryIndex.get(geometryKey)) {
          geometryIndex.delete(geometryKey);
        }
      }
    }
    key = geometryIndex.get(geometry) ?? null;
  }
  const entry = mapTaleSkin(key);
  return {
    key,
    entry,
    geometry,
    layoutMatches: entry?.tale === true && entry.geometry === geometry,
  };
}

/**
 * Switches this file reads that are off when the caller leaves them out. They
 * match the registry defaults in `visual-filters.js`; this file cannot import
 * that registry, because the registry imports `red-zone.js`, which imports
 * this file (an import loop that breaks when this file loads first).
 */
const OFF_WHEN_MISSING = new Set(["survey_frame", "arena_rim", "pov_sight_veil"]);

/**
 * Read one switch: a boolean in filters wins; a missing switch is on, except
 * the ones in OFF_WHEN_MISSING.
 *
 * @param {unknown} filters
 * @param {string} id
 * @returns {boolean}
 */
function switchOn(filters, id) {
  const value = isRecord(filters) ? filters[id] : undefined;
  if (typeof value === "boolean") {
    return value;
  }
  return !OFF_WHEN_MISSING.has(id);
}

/**
 * Turn scene stones into plain world-unit shapes.
 *
 * @param {unknown} map
 * @returns {FloorStone[]}
 */
function stonesOf(map) {
  return asArray(isRecord(map) ? map.obstacles : null)
    .filter(isRecord)
    .map((obstacle) => {
      const center = asArray(obstacle.center);
      return {
        kind: obstacle.kind === "pillar" ? "pillar" : "wall",
        x: finiteNumber(center[0]),
        y: finiteNumber(center[1]),
        r: finiteNumber(obstacle.radius),
        w: finiteNumber(obstacle.width),
        h: finiteNumber(obstacle.height),
        theta: finiteNumber(obstacle.theta),
      };
    });
}

/**
 * Active pads with their team (team_id 1 = Team A, 2 = Team B).
 *
 * @param {unknown} spawnPads
 * @returns {FloorPad[]}
 */
function padsOf(spawnPads) {
  /** @type {FloorPad[]} */
  const pads = [];
  for (const pad of asArray(spawnPads)) {
    if (!isRecord(pad) || pad.configured_active === false) {
      continue;
    }
    const position = asArray(pad.position);
    const x = position[0];
    const y = position[1];
    if (typeof x === "number" && Number.isFinite(x) && typeof y === "number") {
      if (Number.isFinite(y) && (pad.team_id === 1 || pad.team_id === 2)) {
        pads.push({ x, y, team: pad.team_id === 1 ? "A" : "B" });
      }
    }
  }
  return pads;
}

/**
 * Thin walls (under 0.1 units thick) as world segments [ax, ay, bx, by].
 *
 * @param {FloorStone[]} stones
 * @returns {Array<[number, number, number, number]>}
 */
function railsOf(stones) {
  /** @type {Array<[number, number, number, number]>} */
  const rails = [];
  for (const stone of stones) {
    if (stone.kind !== "wall" || Math.min(stone.w, stone.h) >= 0.1) {
      continue;
    }
    const length = Math.max(stone.w, stone.h);
    const angle = stone.theta + (stone.w < stone.h ? Math.PI / 2 : 0);
    const dx = (Math.cos(angle) * length) / 2;
    const dy = (Math.sin(angle) * length) / 2;
    rails.push([stone.x - dx, stone.y - dy, stone.x + dx, stone.y + dy]);
  }
  return rails;
}

/**
 * Intervals a line at axis value c must skip around rails lying on it.
 *
 * @param {Array<[number, number, number, number]>} rails
 * @param {"x" | "y"} axis
 * @param {number} c
 * @returns {Array<[number, number]>}
 */
function breaksFor(rails, axis, c) {
  /** @type {Array<[number, number]>} */
  const intervals = [];
  for (const [ax, ay, bx, by] of rails) {
    if (axis === "x" && Math.abs(ax - c) < 2e-3 && Math.abs(bx - c) < 2e-3) {
      intervals.push([Math.min(ay, by) - 0.25, Math.max(ay, by) + 0.25]);
    }
    if (axis === "y" && Math.abs(ay - c) < 2e-3 && Math.abs(by - c) < 2e-3) {
      intervals.push([Math.min(ax, bx) - 0.25, Math.max(ax, bx) + 0.25]);
    }
  }
  return intervals;
}

/**
 * Pieces of [lo, hi] left after the breaks; leftovers under 0.3 units drop.
 *
 * @param {number} lo
 * @param {number} hi
 * @param {Array<[number, number]>} intervals
 * @returns {Array<[number, number]>}
 */
function pieces(lo, hi, intervals) {
  /** @type {Array<[number, number]>} */
  let segments = [[lo, hi]];
  for (const [a, b] of intervals) {
    /** @type {Array<[number, number]>} */
    const next = [];
    for (const [s, e] of segments) {
      if (b <= s || a >= e) {
        next.push([s, e]);
        continue;
      }
      if (a > s) {
        next.push([s, a]);
      }
      if (b < e) {
        next.push([b, e]);
      }
    }
    segments = next;
  }
  if (intervals.length === 0) {
    return segments;
  }
  return segments.filter(([s, e]) => e - s >= 0.3 || (s === lo && e === hi));
}

/**
 * The drawing frame of one render: screen floor rectangle and helpers.
 *
 * @param {LookTransform} transform
 */
function frameOf(transform) {
  const bounds = transform.mapBounds;
  const viewport = transform.viewportBounds;
  const ppu = finiteNumber(transform.scale);
  const W = finiteNumber(transform.worldWidth);
  const H = finiteNumber(transform.worldHeight);
  return {
    ppu,
    W,
    H,
    x0: bounds.left,
    y0: bounds.top,
    x1: bounds.right,
    y1: bounds.bottom,
    vx: viewport.left,
    vy: viewport.top,
    vw: viewport.width,
    vh: viewport.height,
    rx: floorCornerRadius(ppu),
    margin: Math.max(
      4,
      Math.min(
        bounds.left - viewport.left,
        bounds.top - viewport.top,
        viewport.right - bounds.right,
        viewport.bottom - bounds.bottom,
      ),
    ),
    /** @param {number} x */
    X: (x) => bounds.left + x * ppu,
    /** @param {number} y */
    Y: (y) => bounds.top + (H - y) * ppu,
  };
}

/** @typedef {ReturnType<typeof frameOf>} FloorFrame */

/**
 * A rounded rectangle path; ccw draws the opposite winding (a hole).
 *
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {number} radius
 * @param {boolean} [ccw]
 * @returns {string}
 */
function roundedRect(x, y, w, h, radius, ccw = false) {
  const r = Math.max(0, Math.min(radius, w / 2, h / 2));
  if (!ccw) {
    return `M${f(x + r)},${f(y)}H${f(x + w - r)}A${f(r)},${f(r)} 0 0 1 ${f(x + w)},${f(y + r)}V${f(y + h - r)}A${f(r)},${f(r)} 0 0 1 ${f(x + w - r)},${f(y + h)}H${f(x + r)}A${f(r)},${f(r)} 0 0 1 ${f(x)},${f(y + h - r)}V${f(y + r)}A${f(r)},${f(r)} 0 0 1 ${f(x + r)},${f(y)}Z`;
  }
  return `M${f(x + r)},${f(y)}A${f(r)},${f(r)} 0 0 0 ${f(x)},${f(y + r)}V${f(y + h - r)}A${f(r)},${f(r)} 0 0 0 ${f(x + r)},${f(y + h)}H${f(x + w - r)}A${f(r)},${f(r)} 0 0 0 ${f(x + w)},${f(y + h - r)}V${f(y + r)}A${f(r)},${f(r)} 0 0 0 ${f(x + w - r)},${f(y)}Z`;
}

/**
 * A disc path centred on (cx, cy) with radius r, in pixels.
 *
 * @param {number} cx
 * @param {number} cy
 * @param {number} r
 * @returns {string}
 */
function disc(cx, cy, r) {
  return `M${f(cx - r)},${f(cy)}a${f(r)},${f(r)} 0 1,0 ${f(2 * r)},0a${f(r)},${f(r)} 0 1,0 ${f(-2 * r)},0Z`;
}

/**
 * Path of one floor line along an axis, cut around rails.
 *
 * @param {FloorFrame} g
 * @param {Array<[number, number, number, number]>} rails
 * @param {"x" | "y"} axis
 * @param {number} c
 * @param {number} lo
 * @param {number} hi
 * @returns {string}
 */
function lineSegments(g, rails, axis, c, lo, hi) {
  let d = "";
  for (const [s, e] of pieces(lo, hi, breaksFor(rails, axis, c))) {
    d +=
      axis === "x"
        ? `M${f(g.X(c))},${f(g.Y(e))}V${f(g.Y(s))}`
        : `M${f(g.X(s))},${f(g.Y(c))}H${f(g.X(e))}`;
  }
  return d;
}

/**
 * Distance from a world point to the nearest stone edge or pad seal.
 *
 * @param {FloorStone[]} stones
 * @param {FloorPad[]} pads
 * @param {number} px
 * @param {number} py
 * @returns {number}
 */
function clearance(stones, pads, px, py) {
  let best = 1e9;
  for (const stone of stones) {
    let distance;
    if (stone.kind === "pillar") {
      distance = Math.hypot(px - stone.x, py - stone.y) - stone.r;
    } else {
      const c = Math.cos(-stone.theta);
      const s = Math.sin(-stone.theta);
      const lx = (px - stone.x) * c - (py - stone.y) * s;
      const ly = (px - stone.x) * s + (py - stone.y) * c;
      const qx = Math.abs(lx) - stone.w / 2;
      const qy = Math.abs(ly) - stone.h / 2;
      distance =
        Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0);
    }
    best = Math.min(best, distance);
  }
  for (const pad of pads) {
    best = Math.min(best, Math.hypot(px - pad.x, py - pad.y) - 0.6);
  }
  return best;
}

/**
 * Make one frozen shape record.
 *
 * @param {string} kind
 * @param {string} tag
 * @param {string} className
 * @param {Record<string, string | number>} attributes
 * @param {string} [text]
 * @returns {LookRecord}
 */
function shape(kind, tag, className, attributes, text) {
  return Object.freeze({
    kind,
    tag,
    className,
    attributes: Object.freeze({ ...attributes, "pointer-events": "none" }),
    ...(text === undefined ? {} : { text }),
  });
}

/**
 * Moonlight records: the Moon Pool bands, or one sinking hollow on Tidewrack.
 *
 * @param {FloorFrame} g
 * @param {(typeof MAP_SKIN_KITS)[string]} kit
 * @returns {LookRecord[]}
 */
function moonlightRecords(g, kit) {
  const cx = g.X(g.W / 2);
  const cy = g.Y(g.H / 2);
  /** @type {(w: number, h: number, ccw?: boolean) => string} */
  const oval = (w, h, ccw = false) => {
    const rx = (w / 2) * g.ppu;
    const ry = (h / 2) * g.ppu;
    const sweep = ccw ? 0 : 1;
    return `M${f(cx - rx)},${f(cy)}a${f(rx)},${f(ry)} 0 1,${sweep} ${f(2 * rx)},0a${f(rx)},${f(ry)} 0 1,${sweep} ${f(-2 * rx)},0Z`;
  };
  if (kit.abyss) {
    return kit.pool === kit.floor
      ? []
      : [
          shape("moonlight", "path", "map-skin-moonlight map-skin-moonlight--hollow", {
            d: oval(18, 9),
            fill: kit.pool,
            opacity: 0.22,
          }),
        ];
  }
  return MOON_BANDS.map(([w, h, opacity], index) => {
    const inner = MOON_BANDS[index + 1];
    return shape("moonlight", "path", "map-skin-moonlight", {
      d: `${oval(w, h)}${inner ? ` ${oval(inner[0], inner[1], true)}` : ""}`,
      fill: MAP_SKIN_MATERIALS.moon,
      opacity,
    });
  });
}

/**
 * Floor vignette records: three dark bands just inside the floor edge.
 *
 * @param {FloorFrame} g
 * @returns {LookRecord[]}
 */
function vignetteRecords(g) {
  const w = g.x1 - g.x0;
  const h = g.y1 - g.y0;
  return VIGNETTE_BANDS.map(([band, opacity]) => {
    const b = band * g.ppu;
    return shape("vignette", "path", "map-skin-vignette", {
      d: `${roundedRect(g.x0, g.y0, w, h, g.rx)} ${roundedRect(g.x0 + b, g.y0 + b, w - 2 * b, h - 2 * b, Math.max(0, g.rx - b), true)}`,
      opacity,
    });
  });
}

/**
 * Coursed flagstone joints: course lines on whole y, 2 x 1 stones offset per
 * course, ley joints at x 5 and 15 and y 5. Fine joints only at 20 px a unit.
 *
 * @param {FloorFrame} g
 * @param {(typeof MAP_SKIN_KITS)[string]} kit
 * @param {Array<[number, number, number, number]>} rails
 * @returns {LookRecord[]}
 */
function jointRecords(g, kit, rails) {
  let minor = "";
  let major = "";
  const dense = g.ppu >= 20;
  for (let y = 1; y < g.H; y += 1) {
    const d = lineSegments(g, rails, "y", y, 0, g.W);
    if (y === 5) {
      major += d;
    } else if (dense) {
      minor += d;
    }
  }
  if (dense) {
    for (let k = 0; k < g.H; k += 1) {
      for (let x = 1; x < g.W; x += 1) {
        if (x === 5 || x === 10 || x === 15 || x % 2 !== k % 2) {
          continue;
        }
        minor += lineSegments(g, rails, "x", x, k, k + 1);
      }
    }
  }
  for (const x of [5, 15]) {
    if (x < g.W) {
      major += lineSegments(g, rails, "x", x, 0, g.H);
    }
  }
  /** @type {LookRecord[]} */
  const out = [];
  for (const [d, opacity, modifier] of /** @type {const} */ ([
    [minor, 0.4, "minor"],
    [major, 0.7, "major"],
  ])) {
    if (d) {
      out.push(
        shape("joints", "path", `map-skin-joints map-skin-joints--${modifier}`, {
          d,
          fill: "none",
          stroke: kit.joint,
          "stroke-opacity": opacity,
          "stroke-width": 0.75,
        }),
      );
    }
  }
  return out;
}

/**
 * The dashed ley seam at the mirror line (0.5 units on, 0.5 off).
 *
 * @param {FloorFrame} g
 * @param {(typeof MAP_SKIN_KITS)[string]} kit
 * @param {Array<[number, number, number, number]>} rails
 * @returns {LookRecord[]}
 */
function seamRecords(g, kit, rails) {
  const intervals = breaksFor(rails, "x", g.W / 2);
  let d = "";
  for (let y = g.H; y > 0; y -= 1) {
    for (const [a, b] of pieces(y - 0.5, y, intervals)) {
      d += `M${f(g.X(g.W / 2))},${f(g.Y(b))}V${f(g.Y(a))}`;
    }
  }
  return d
    ? [
        shape("ley-seam", "path", "map-skin-seam", {
          d,
          fill: "none",
          stroke: kit.seam,
          "stroke-width": 1,
        }),
      ]
    : [];
}

/**
 * The one-unit grid of the combat debugger: one line per whole unit, majors
 * every 5 units and the mirror line marked. Lines are whole (no rail cuts).
 *
 * @param {FloorFrame} g
 * @returns {LookRecord[]}
 */
function unitGridRecords(g) {
  /** @type {LookRecord[]} */
  const out = [];
  if (!Number.isInteger(g.W) || !Number.isInteger(g.H)) {
    return out;
  }
  /** @param {number} value @param {number} mid */
  const rank = (value, mid) =>
    value === mid
      ? " map-grid-line--mid"
      : value % 5 === 0
        ? " map-grid-line--major"
        : "";
  for (let x = 1; x < g.W; x += 1) {
    out.push(
      shape(
        "grid-line",
        "line",
        `map-grid-line map-grid-line--vertical${rank(x, g.W / 2)}`,
        { x1: f(g.X(x)), y1: f(g.Y(0)), x2: f(g.X(x)), y2: f(g.Y(g.H)) },
      ),
    );
  }
  for (let y = 1; y < g.H; y += 1) {
    out.push(
      shape(
        "grid-line",
        "line",
        `map-grid-line map-grid-line--horizontal${rank(y, -1)}`,
        {
          x1: f(g.X(0)),
          y1: f(g.Y(y)),
          x2: f(g.X(g.W)),
          y2: f(g.Y(y)),
        },
      ),
    );
  }
  return out;
}

/**
 * Hearth Seals: a flat r 0.36 disc on each active pad (from 20 px a unit).
 *
 * @param {FloorFrame} g
 * @param {(typeof MAP_SKIN_KITS)[string]} kit
 * @param {FloorPad[]} pads
 * @returns {LookRecord[]}
 */
function sealRecords(g, kit, pads) {
  if (g.ppu < 20 || pads.length === 0) {
    return [];
  }
  const d = pads.map((pad) => disc(g.X(pad.x), g.Y(pad.y), 0.36 * g.ppu)).join("");
  return [shape("seals", "path", "map-skin-seals", { d, fill: kit.seal })];
}

/**
 * Home Gates: a 3 px band in the pad team's colour just outside the floor
 * edge beside each active pad; Team B's band is split by a 1 px ink line.
 *
 * @param {FloorFrame} g
 * @param {FloorPad[]} pads
 * @param {number} offset Pixels outside the floor edge.
 * @returns {LookRecord[]}
 */
function gateRecords(g, pads, offset) {
  /** @type {LookRecord[]} */
  const out = [];
  for (const team of /** @type {const} */ (["A", "B"])) {
    let d = "";
    for (const pad of pads) {
      if (pad.team !== team) {
        continue;
      }
      const x = pad.x < g.W / 2 ? g.x0 - offset : g.x1 + offset;
      d += `M${f(x)},${f(g.Y(pad.y + 0.4))}V${f(g.Y(pad.y - 0.4))}`;
    }
    if (!d) {
      continue;
    }
    const lower = team.toLowerCase();
    out.push(
      shape("gate", "path", `map-skin-gate map-skin-gate--${lower}`, {
        d,
        fill: "none",
        "stroke-opacity": 0.55,
        "stroke-width": 3,
      }),
    );
    if (team === "B") {
      out.push(
        shape("gate-split", "path", "map-skin-gate-split", {
          d,
          fill: "none",
          "stroke-width": 1,
        }),
      );
    }
  }
  return out;
}

/**
 * Arena Rim parapet: a carved band outside the floor with coping joints and an
 * outer lip. Return its records and its width in pixels.
 *
 * @param {FloorFrame} g
 * @param {(typeof MAP_SKIN_KITS)[string]} kit
 * @returns {{records: LookRecord[], width: number}}
 */
function parapetRecords(g, kit) {
  const width = Math.min(Math.max(0.28 * g.ppu, 6), 16, g.margin - 1);
  const W = g.x1 - g.x0;
  const H = g.y1 - g.y0;
  const outer = roundedRect(
    g.x0 - width,
    g.y0 - width,
    W + 2 * width,
    H + 2 * width,
    g.rx + width * 0.35,
  );
  let joints = "";
  for (let x = 0; x <= g.W; x += 2) {
    joints += `M${f(g.X(x))},${f(g.y0 - width)}V${f(g.y0)}M${f(g.X(x))},${f(g.y1)}V${f(g.y1 + width)}`;
  }
  for (let y = 0; y <= g.H; y += 2) {
    joints += `M${f(g.x0 - width)},${f(g.Y(y))}H${f(g.x0)}M${f(g.x1)},${f(g.Y(y))}H${f(g.x1 + width)}`;
  }
  return {
    width,
    records: [
      shape("parapet", "path", "map-skin-rim", {
        d: `${outer} ${roundedRect(g.x0, g.y0, W, H, g.rx, true)}`,
        fill: kit.rim,
      }),
      shape("parapet-joints", "path", "map-skin-rim-joints", {
        d: joints,
        fill: "none",
        stroke: kit.rimJoint,
        "stroke-width": 0.75,
      }),
      shape("parapet-lip", "path", "map-skin-rim-lip", {
        d: outer,
        fill: "none",
        "stroke-width": 1,
      }),
    ],
  };
}

/**
 * The thin north shade the Arena Rim casts on the floor's top edge.
 *
 * @param {FloorFrame} g
 * @returns {LookRecord}
 */
function northShadeRecord(g) {
  const h = Math.max(1, 0.08 * g.ppu);
  const corner = Math.min(g.rx, h);
  return shape("north-shade", "path", "map-skin-north-shade", {
    d: `M${f(g.x0 + g.rx)},${f(g.y0)}H${f(g.x1 - g.rx)}Q${f(g.x1)},${f(g.y0)} ${f(g.x1)},${f(g.y0 + corner)}V${f(g.y0 + h)}H${f(g.x0)}V${f(g.y0 + corner)}Q${f(g.x0)},${f(g.y0)} ${f(g.x0 + g.rx)},${f(g.y0)}Z`,
    fill: MAP_SKIN_MATERIALS.shade,
    opacity: 0.4,
  });
}

/**
 * The kit's one sparse inlay (Deepwood roots, Dunes ripples, Tidewrack
 * barnacles, Forge plate bolts), seeded by the catalogue map, built on the
 * left half and mirrored. Only from 30 px a unit, and never on a stone or pad.
 *
 * @param {FloorFrame} g
 * @param {string} kitName
 * @param {(typeof MAP_SKIN_KITS)[string]} kit
 * @param {number} seedId
 * @param {FloorStone[]} stones
 * @param {FloorPad[]} pads
 * @returns {LookRecord[]}
 */
function inlayRecords(g, kitName, kit, seedId, stones, pads) {
  if (!kit.inlay || g.ppu < 30) {
    return [];
  }
  const random = seededNoise(1000 + seedId * 7919);
  /** @param {number} a @param {number} b */
  const uniform = (a, b) => a + (b - a) * random();
  /** @param {number} x @param {number} y */
  const inField = (x, y) =>
    x >= 0.15 && x <= g.W - 0.15 && y >= 0.15 && y <= g.H - 0.15;
  /** @param {number[][]} points */
  const clear = (points) =>
    points.every(([x, y]) => inField(x, y) && clearance(stones, pads, x, y) >= 0.3);
  /** @param {number} x */
  const mirror = (x) => g.W - x;
  /** @param {number} x @param {number} y */
  const point = (x, y) => `${f(g.X(x))},${f(g.Y(y))}`;
  if (kitName === "deepwood") {
    let d = "";
    let count = 0;
    let tries = 0;
    /** @type {Array<[number, boolean]>} */
    const used = [];
    while (count < 5 && tries < 800) {
      tries += 1;
      const x = Math.floor(uniform(1, 9)) + uniform(0.28, 0.72);
      const top = random() < 0.5;
      const y0 = top ? g.H : 0;
      const sign = top ? -1 : 1;
      const length = uniform(0.8, 1.2);
      const bend = (random() < 0.5 ? -1 : 1) * 0.14;
      if (used.some(([ux, ut]) => ut === top && Math.abs(ux - x) < 1.1)) {
        continue;
      }
      const control = [
        [x, y0],
        [x + bend, y0 + sign * 0.35 * length],
        [x - bend, y0 + sign * 0.7 * length],
        [x + bend * 0.5, y0 + sign * length],
      ];
      const samples = [];
      for (let j = 2; j <= 12; j += 1) {
        const u = j / 12;
        const v = 1 - u;
        samples.push(
          [0, 1].map(
            (i) =>
              v * v * v * control[0][i] +
              3 * v * v * u * control[1][i] +
              3 * v * u * u * control[2][i] +
              u * u * u * control[3][i],
          ),
        );
      }
      if (!clear(samples)) {
        continue;
      }
      used.push([x, top]);
      for (const flip of [false, true]) {
        const q = control.map(([px, py]) => [flip ? mirror(px) : px, py]);
        d += `M${point(q[0][0], q[0][1])}C${point(q[1][0], q[1][1])} ${point(q[2][0], q[2][1])} ${point(q[3][0], q[3][1])}`;
      }
      count += 1;
    }
    return d
      ? [
          shape("inlay", "path", "map-skin-inlay", {
            d,
            fill: "none",
            stroke: kit.inlay,
            "stroke-opacity": 0.45,
            "stroke-width": 1,
            "stroke-linecap": "round",
          }),
        ]
      : [];
  }
  if (kitName === "dunes") {
    let d = "";
    let count = 0;
    let tries = 0;
    /** @param {number[]} a @param {number[]} b @param {number[]} c */
    const quad = (a, b, c) => {
      const out = [];
      for (let j = 0; j <= 12; j += 1) {
        const u = j / 12;
        const v = 1 - u;
        out.push([
          v * v * a[0] + 2 * v * u * b[0] + u * u * c[0],
          v * v * a[1] + 2 * v * u * b[1] + u * u * c[1],
        ]);
      }
      return out;
    };
    /** @type {number[][]} */
    const boxes = [];
    while (count < 3 && tries < 600) {
      tries += 1;
      const length = uniform(1.6, 2.4);
      const x = uniform(1.2, 9.6 - length);
      const y = random() < 0.5 ? uniform(0.8, 2.38) : uniform(7.6, 9.18);
      if (
        boxes.some(
          ([bx, by, bl]) =>
            x < bx + bl + 0.4 && bx < x + length + 0.4 && Math.abs(by - y) < 0.8,
        )
      ) {
        continue;
      }
      const arcs = [0, 1].map((k) => {
        const yy = y + k * 0.22;
        return [
          [x, yy],
          [x + length / 2, yy - 0.1],
          [x + length, yy],
        ];
      });
      if (!arcs.every((arc) => clear(quad(arc[0], arc[1], arc[2])))) {
        continue;
      }
      boxes.push([x, y, length]);
      for (const arc of arcs) {
        for (const flip of [false, true]) {
          const q = arc.map(([px, py]) => [flip ? mirror(px) : px, py]);
          d += `M${point(q[0][0], q[0][1])}Q${point(q[1][0], q[1][1])} ${point(q[2][0], q[2][1])}`;
        }
      }
      count += 1;
    }
    return d
      ? [
          shape("inlay", "path", "map-skin-inlay", {
            d,
            fill: "none",
            stroke: kit.inlay,
            "stroke-opacity": 0.5,
            "stroke-width": 0.75,
            "stroke-linecap": "round",
          }),
        ]
      : [];
  }
  if (kitName === "tidewrack") {
    let d = "";
    const radius = Math.max(0.05 * g.ppu, 1);
    for (const [cx, cy] of [
      [1.3, 1.1],
      [1.6, 8.8],
    ]) {
      let count = 0;
      let tries = 0;
      /** @type {number[][]} */
      const got = [];
      while (count < 4 && tries < 300) {
        tries += 1;
        const px = cx + uniform(-0.6, 0.9);
        const py = cy + uniform(-0.5, 0.5);
        if (
          !inField(px, py) ||
          clearance(stones, pads, px, py) < 0.3 ||
          got.some(([a, b]) => Math.hypot(a - px, b - py) < 0.14)
        ) {
          continue;
        }
        got.push([px, py]);
        for (const x of [px, mirror(px)]) {
          d += disc(g.X(x), g.Y(py), radius);
        }
        count += 1;
      }
    }
    return d
      ? [
          shape("inlay", "path", "map-skin-inlay", {
            d,
            fill: "none",
            stroke: kit.inlay,
            "stroke-opacity": 0.5,
            "stroke-width": 0.75,
          }),
        ]
      : [];
  }
  if (kitName === "forge") {
    let d = "";
    const size = Math.max(0.18 * g.ppu, 3);
    for (const x of [2.5, 7.5]) {
      for (const y of [2.5, 7.5]) {
        if (
          clearance(stones, [], x, y) < 0.3 ||
          clearance(stones, [], mirror(x), y) < 0.3
        ) {
          continue;
        }
        for (const px of [x, mirror(x)]) {
          d += `M${f(g.X(px) - size / 2)},${f(g.Y(y) - size / 2)}h${f(size)}v${f(size)}h${f(-size)}Z`;
        }
      }
    }
    return d
      ? [
          shape("inlay", "path", "map-skin-inlay", {
            d,
            fill: kit.inlay,
            stroke: MAP_SKIN_MATERIALS.boltEdge,
            "stroke-width": 0.75,
          }),
        ]
      : [];
  }
  return [];
}

/**
 * Starfall apron stars: still, neutral specks in the top and bottom margins.
 *
 * @param {FloorFrame} g
 * @param {(typeof MAP_SKIN_KITS)[string]} kit
 * @param {number} seedId
 * @returns {LookRecord[]}
 */
function apronStarRecords(g, kit, seedId) {
  if (!kit.inlay) {
    return [];
  }
  const random = seededNoise(2000 + seedId * 104729);
  /** @param {number} a @param {number} b */
  const uniform = (a, b) => a + (b - a) * random();
  const above = g.y0 - g.vy - 2;
  const below = g.vy + g.vh - g.y1 - 2;
  let d = "";
  for (let i = 0; i < 7; i += 1) {
    const x = uniform(0.3, 9.7);
    const up = i % 2 === 0;
    const space = up ? above : below;
    const offset = uniform(6, Math.max(6.5, Math.min(space, 40)));
    const r = uniform(0.8, 1.2);
    if (space < 8) {
      continue;
    }
    const y = up ? g.y0 - offset : g.y1 + offset;
    for (const px of [x, g.W - x]) {
      d += disc(g.X(px), y, r);
    }
  }
  return d
    ? [
        shape("apron-stars", "path", "map-skin-apron-stars", {
          d,
          fill: kit.inlay,
          opacity: 0.4,
        }),
      ]
    : [];
}

/**
 * Survey Frame: world coordinates on the left and bottom margins, y up.
 *
 * @param {FloorFrame} g
 * @returns {LookRecord[]}
 */
function surveyFrameRecords(g) {
  const labelGap = 6;
  const tick = 4;
  let size = 11;
  while (size > 7.5 && 2 * 0.56 * size > g.x0 - g.vx - labelGap - 1) {
    size -= 0.5;
  }
  size = Math.min(size, Math.max(7.5, g.margin - 7.5));
  let d = "";
  for (let x = 0; x <= g.W; x += 1) {
    d += `M${f(g.X(x))},${f(g.y1 + 1)}V${f(g.y1 + 1 + tick)}`;
  }
  for (let y = 0; y <= g.H; y += 2) {
    d += `M${f(g.x0 - 5)},${f(g.Y(y))}H${f(g.x0 - 1)}`;
  }
  /** @type {LookRecord[]} */
  const out = [
    shape("ruler-ticks", "path", "map-skin-ruler-ticks", {
      d,
      fill: "none",
      "stroke-opacity": 0.55,
      "stroke-width": 1,
    }),
  ];
  const baseline = g.y1 + 1 + tick + 1.5 + size * 0.5;
  for (let x = 0; x <= g.W; x += 5) {
    out.push(
      shape(
        "ruler-label",
        "text",
        "map-skin-ruler-label",
        { x: f(g.X(x)), y: f(baseline), "font-size": f(size), "text-anchor": "middle" },
        String(x),
      ),
    );
  }
  for (let y = 0; y <= g.H; y += 2) {
    out.push(
      shape(
        "ruler-label",
        "text",
        "map-skin-ruler-label",
        {
          x: f(g.x0 - labelGap),
          y: f(g.Y(y)),
          "font-size": f(size),
          "text-anchor": "end",
        },
        String(y),
      ),
    );
  }
  return out;
}

/**
 * The switch mix index of a stored tale tree (bit k = k-th switch on).
 *
 * @param {ReadonlyArray<string>} switchIds
 * @param {(id: string) => boolean} isOn
 * @returns {number}
 */
function mixIndex(switchIds, isOn) {
  let mix = 0;
  for (const [bit, id] of switchIds.entries()) {
    if (isOn(id)) {
      mix |= 1 << bit;
    }
  }
  return mix;
}

/**
 * Red Zone depth in world units from the map's recorded strips, or 0.
 *
 * @param {unknown} map
 * @returns {number}
 */
function redZoneDepth(map) {
  const zone = isRecord(map) && isRecord(map.red_zone) ? map.red_zone : null;
  if (zone === null) {
    return 0;
  }
  if (typeof zone.depth === "number" && Number.isFinite(zone.depth)) {
    return zone.depth;
  }
  const range = asArray(zone.team_a_x_range);
  return Math.abs(finiteNumber(range[1]) - finiteNumber(range[0]));
}

/**
 * The tale group's placement: reference pixels to this render's pixels.
 *
 * @param {FloorFrame} g
 * @returns {string}
 */
function taleTransform(g) {
  const k = g.ppu / TALE_REFERENCE_SCALE;
  return `translate(${f(g.x0)} ${f(g.y0)}) scale(${Math.round(k * 1e6) / 1e6})`;
}

/**
 * Return the floor drawing records for one render.
 *
 * input fields (all optional):
 * - transform: this render's projection (LookTransform, with the camera
 *   frame). Without it, or without a positive map size, return [].
 * - map: the arena scene's `map` record: width and height (world units),
 *   obstacles (stones, used for the geometry key, the rail cuts and the kit
 *   inlay) and red_zone (its depth picks the tale's homeland ground).
 * - spawnPads: the scene's `spawn_pads` (position, team_id 1 or 2 and
 *   configured_active are read; inactive pads get no seal or gate).
 * - mapKey: the map's technical name, or null. With no name the skin is found
 *   by an exact catalogue layout match (see resolveMapSkin).
 * - filters: the visual filter state. A missing id reads as on, except
 *   survey_frame, arena_rim and pov_sight_veil (off), as in the registry.
 *   map_skins (off: plain Moonstone kit, no tale, no inlay, no stars),
 *   moonlight, floor_grid, spawn_waystones (seals, gates and the tale's
 *   waystone beds), red_zone_floors (the tale's quieter homeland ground),
 *   survey_frame (hides the Arena Rim while on), arena_rim, pov_sight_veil.
 * - audience: "researcher" (Oracle View) or "agent_pov". The floor is the same
 *   for both, except Map 47's Legion Mist, left out in Agent POV while
 *   pov_sight_veil is on.
 * - showUnitGrid (default false): the combat debugger's one-unit grid in place
 *   of the coursed joints (still under floor_grid).
 *
 * Return a frozen array of frozen records in paint order. A shape record is
 * {kind, tag, className, attributes, text?} in CSS pixels; a tale record is
 * {kind: "tale", layer: "floor", mapKey, mix, attributes: {transform}}, where
 * mix picks the stored tree's nodes (bits of switches the tale ignores are
 * zero, so views that differ only in those switches get equal records). The same inputs return the same array
 * object (a small cache keyed by every value read). Pure data: no DOM.
 *
 * @param {{
 *   transform?: LookTransform | null,
 *   map?: Record<string, any> | null,
 *   spawnPads?: unknown,
 *   mapKey?: string | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   audience?: "researcher" | "agent_pov" | null,
 *   showUnitGrid?: boolean,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const transform = input?.transform;
  const width = finiteNumber(input?.map?.width);
  const height = finiteNumber(input?.map?.height);
  if (!transform || !(width > 0) || !(height > 0)) {
    return Object.freeze([]);
  }
  const filters = input.filters;
  /** @param {string} id */
  const isOn = (id) => switchOn(filters, id);
  const pads = padsOf(input.spawnPads);
  const resolved = resolveMapSkin(input.mapKey ?? null, input.map);
  const veil = input.audience === "agent_pov" && isOn("pov_sight_veil");
  const depth = redZoneDepth(input.map);
  const bounds = transform.mapBounds;
  const viewport = transform.viewportBounds;
  const switchText = [
    "map_skins",
    "moonlight",
    "floor_grid",
    "spawn_waystones",
    "red_zone_floors",
    "survey_frame",
    "arena_rim",
  ]
    .map((id) => (isOn(id) ? "1" : "0"))
    .join("");
  const cacheKey = [
    resolved.key,
    resolved.geometry,
    width,
    height,
    depth,
    veil,
    input.showUnitGrid === true,
    switchText,
    transform.scale,
    bounds.left,
    bounds.top,
    viewport.left,
    viewport.top,
    viewport.width,
    viewport.height,
    pads.map((pad) => `${pad.x},${pad.y},${pad.team}`).join(";"),
  ].join("|");
  const cached = RECORDS_CACHE.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  const g = frameOf(transform);
  const stones = stonesOf(input.map);
  const rails = railsOf(stones);
  const skins = isOn("map_skins");
  const grid = isOn("floor_grid");
  const waystones = isOn("spawn_waystones");
  const ruler = isOn("survey_frame");
  const rim = isOn("arena_rim") && !ruler;
  const kitName = skins && resolved.entry ? resolved.entry.kit : "moonstone";
  const kit = MAP_SKIN_KITS[kitName] ?? MAP_SKIN_KITS.moonstone;
  const seedId = resolved.entry?.mapId ?? 0;
  const tale = skins && resolved.layoutMatches && resolved.key !== null;

  /** @type {LookRecord[]} */
  const out = [
    shape("apron", "rect", "map-skin-apron", {
      x: f(g.vx - g.vw),
      y: f(g.vy - g.vh),
      width: f(g.vw * 3),
      height: f(g.vh * 3),
    }),
  ];
  const parapet = rim ? parapetRecords(g, kit) : null;
  if (tale && resolved.entry !== null) {
    const homeland =
      isOn("red_zone_floors") &&
      depth > 0 &&
      Math.abs(depth - resolved.entry.redZoneDepth) < 1e-6;
    const mix =
      mixIndex(TALE_FLOOR_SWITCHES, (id) =>
        id === "pov_sight_veil" ? veil : id === "red_zone_floors" ? homeland : isOn(id),
      ) & (resolved.entry.floorSwitches ?? -1);
    out.push(
      Object.freeze({
        kind: "tale",
        layer: "floor",
        mapKey: resolved.key,
        mix,
        attributes: Object.freeze({
          class: "map-skin-tale",
          transform: taleTransform(g),
          "pointer-events": "none",
        }),
      }),
    );
    if (isOn("moonlight") && !kit.abyss) {
      out.push(...moonlightRecords(g, kit));
    }
    if (grid) {
      out.push(
        ...(input.showUnitGrid === true
          ? unitGridRecords(g)
          : jointRecords(g, kit, rails)),
      );
    }
    if (rim) {
      out.push(northShadeRecord(g));
    }
    if (waystones) {
      out.push(...sealRecords(g, kit, pads));
    }
  } else {
    out.push(
      shape("floor", "path", "map-skin-floor", {
        d: roundedRect(g.x0, g.y0, g.x1 - g.x0, g.y1 - g.y0, g.rx),
        fill: kit.floor,
      }),
    );
    if (isOn("moonlight")) {
      out.push(...moonlightRecords(g, kit), ...vignetteRecords(g));
    }
    if (rim) {
      out.push(northShadeRecord(g));
    }
    if (skins) {
      out.push(...inlayRecords(g, kitName, kit, seedId, stones, pads));
    }
    if (grid) {
      out.push(
        ...(input.showUnitGrid === true
          ? unitGridRecords(g)
          : jointRecords(g, kit, rails)),
      );
    }
    if (waystones) {
      out.push(...sealRecords(g, kit, pads));
    }
    if (grid && input.showUnitGrid !== true) {
      out.push(...seamRecords(g, kit, rails));
    }
  }
  if (parapet) {
    out.push(...parapet.records);
  }
  if (skins && kitName === "starfall" && !ruler && !rim) {
    out.push(...apronStarRecords(g, kit, seedId));
  }
  if (waystones) {
    out.push(...gateRecords(g, pads, parapet ? parapet.width / 2 : 4));
  }
  out.push(
    shape("edge", "rect", "map-boundary map-skin-edge", {
      x: f(g.x0),
      y: f(g.y0),
      width: f(g.x1 - g.x0),
      height: f(g.y1 - g.y0),
      rx: f(g.rx),
      fill: "none",
      stroke: MAP_SKIN_MATERIALS.edge,
      "stroke-width": 1.5,
    }),
  );
  if (ruler) {
    out.push(...surveyFrameRecords(g));
  }
  const result = Object.freeze(out);
  RECORDS_CACHE.set(cacheKey, result);
  if (RECORDS_CACHE.size > CACHE_SIZE) {
    RECORDS_CACHE.delete(/** @type {string} */ (RECORDS_CACHE.keys().next().value));
  }
  return result;
}

/**
 * Build the SVG nodes of one stored tale tree for a switch mix.
 *
 * @param {Document} doc
 * @param {ReadonlyArray<TaleNode>} nodes
 * @param {number} mix
 * @param {Element} parent
 */
function appendTaleNodes(doc, nodes, mix, parent) {
  for (const [tag, attributes, mask, children] of nodes) {
    if (mask !== -1 && (mask & (1 << mix)) === 0) {
      continue;
    }
    const element = svgNode(doc, tag, attributes);
    if (children) {
      appendTaleNodes(doc, children, mix, element);
    }
    parent.append(element);
  }
}

/**
 * Build one tale group (floor or stones) for a tale record.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement | null}
 */
function taleGroup(doc, record) {
  const tree =
    record.layer === "stones"
      ? taleStoneTree(record.mapKey)
      : taleFloorTree(record.mapKey);
  if (tree === null) {
    return null;
  }
  const group = svgNode(doc, "g", record.attributes);
  appendTaleNodes(doc, tree, record.mix, group);
  return group;
}

/**
 * Build export-safe SVG nodes for floor records.
 *
 * records come from records() (or taleStoneRecords()); doc creates the nodes.
 * A shape record becomes one element with its class, attributes and text; a
 * tale record becomes a `g.map-skin-tale` (or `g.map-skin-tale-stones`) holding
 * its stored tree's nodes for the record's switch mix. Records of another kind
 * are skipped. Return the nodes in record order; the caller puts them in the
 * `map` layer. Building the same records array again for the same doc returns
 * the same nodes (in a new array), so a re-render does not rebuild the floor.
 * No tooltip, storage or timer.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @returns {SVGElement[]}
 */
export function build(records, doc) {
  if (records.length === 0) {
    return [];
  }
  const built = BUILT.get(records);
  if (built !== undefined && built.doc === doc) {
    return [...built.nodes];
  }
  /** @type {SVGElement[]} */
  const nodes = [];
  for (const record of records) {
    if (record.kind === "tale") {
      const group = taleGroup(doc, record);
      if (group !== null) {
        nodes.push(group);
      }
    } else if (typeof record.tag === "string") {
      const element = svgNode(doc, record.tag, {
        class: record.className,
        ...record.attributes,
      });
      if (typeof record.text === "string") {
        element.textContent = record.text;
      }
      nodes.push(element);
    }
  }
  BUILT.set(records, { doc, nodes });
  if (nodes[0]) {
    BUILT_OWNER.set(nodes[0], records);
  }
  return [...nodes];
}

/**
 * Move existing floor nodes to new records (reprojection).
 *
 * nodes are build()'s nodes and records the new records, in the same order
 * and of the same kinds (the caller rebuilds when the count or a kind
 * changes). Write the new attributes and text; a tale group only gets its new
 * placement. The build cache then serves these nodes for the new records only.
 * Return undefined. DOM change only.
 *
 * @param {ReadonlyArray<Element>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  for (const [index, node] of nodes.entries()) {
    const record = records[index];
    if (!record) {
      continue;
    }
    if (record.kind === "tale") {
      setNodeAttributes(node, { transform: record.attributes.transform });
    } else {
      setNodeAttributes(node, { class: record.className, ...record.attributes });
      if (typeof record.text === "string") {
        node.textContent = record.text;
      }
    }
  }
  const first = nodes[0];
  if (first !== undefined) {
    const previous = BUILT_OWNER.get(first);
    if (previous !== undefined) {
      BUILT.delete(previous);
    }
    const doc = first.ownerDocument;
    if (doc !== null && records.length > 0) {
      BUILT.set(records, { doc, nodes: /** @type {SVGElement[]} */ ([...nodes]) });
      BUILT_OWNER.set(first, records);
    }
  }
}

/**
 * Return the tooltip for a floor record. The floor has none, so return null.
 *
 * @param {LookRecord} _record
 * @returns {null}
 */
export function tooltip(_record) {
  return null;
}

/**
 * Return the tale stone record for a map, for `obstacles.js` to draw.
 *
 * input takes the same transform, map, mapKey and filters as records(). The
 * record exists only when map_skins and obstacle_skins are on and the map has
 * a tale whose catalogue layout equals this map (so every stone is drawn on
 * its true footprint). obstacle_relief, obstacle_shadows and red_zone_floors
 * pick the stored tree's nodes. Return a frozen array holding zero or one
 * frozen {kind: "tale", layer: "stones", mapKey, mix, attributes} record;
 * build() turns it into a `g.map-skin-tale-stones`. Pure data; the stones are
 * the same in every view. The stored stones carry no tooltip or hit shape:
 * `obstacles.js` keeps its own hover targets.
 *
 * @param {{
 *   transform?: LookTransform | null,
 *   map?: Record<string, any> | null,
 *   mapKey?: string | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function taleStoneRecords(input) {
  const transform = input?.transform;
  if (!transform || !isRecord(input.map)) {
    return Object.freeze([]);
  }
  /** @param {string} id */
  const isOn = (id) => switchOn(input.filters, id);
  const resolved = resolveMapSkin(input.mapKey ?? null, input.map);
  if (
    !isOn("map_skins") ||
    !isOn("obstacle_skins") ||
    !resolved.layoutMatches ||
    resolved.entry === null ||
    resolved.key === null
  ) {
    return Object.freeze([]);
  }
  const depth = redZoneDepth(input.map);
  const homeland =
    isOn("red_zone_floors") &&
    depth > 0 &&
    Math.abs(depth - resolved.entry.redZoneDepth) < 1e-6;
  const mix =
    mixIndex(TALE_STONE_SWITCHES, (id) =>
      id === "red_zone_floors" ? homeland : isOn(id),
    ) & (resolved.entry.stoneSwitches ?? -1);
  return Object.freeze([
    Object.freeze({
      kind: "tale",
      layer: "stones",
      mapKey: resolved.key,
      mix,
      attributes: Object.freeze({
        class: "map-skin-tale-stones",
        transform: taleTransform(frameOf(transform)),
        "pointer-events": "none",
      }),
    }),
  ]);
}
