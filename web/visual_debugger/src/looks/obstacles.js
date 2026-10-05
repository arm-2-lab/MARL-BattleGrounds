/**
 * @file The Obstacles look (Standing Stones): pillars, walls and thin walls in the
 * `obstacle` layer.
 *
 * What it draws, bottom to top, inside one `g.standing-stones` group per render:
 * - a soft shadow under every stone (Obstacle Shadows): Stone Shadow at 55%,
 *   moved 0.06 world units down the screen, kept inside the floor by a plain
 *   floor-sized window (a nested `<svg>`, no clip path or id);
 * - thin walls (shorter side under 0.1 units) as blades: a dark Stone Edge
 *   stroke, a Stone Cap core and, with Obstacle Relief, lit End Stones on free
 *   ends, plus an invisible 10 px line that carries the tooltip;
 * - pillar and wall bodies: Standing Stone or Basalt Wall fill with a 1.5 px Stone
 *   Edge drawn inside the true footprint (its outer side is the recorded edge);
 * - with Obstacle Relief: a Wall Top (inset 0.12 units, lifted 0.04 units up the
 *   screen) and a Stone Cap crescent on the upper half of each pillar.
 * Below 22 screen px per world unit the caps, tops and End Stones are dropped
 * and the lines thin (detail ladder 40 / 22 px per unit, as in the design store).
 * With Obstacle Skins off every stone is a plain flat shape on its true footprint
 * (plan P26): nothing about the stone is hidden.
 *
 * The tooltip (a Symbol Tip) uses the design store's plain words (spec
 * `specs/obstacles.json`, Tooltip): the title "Pillar", "Wall" or "Thin Wall";
 * the rows Obstacle ID (a plain number), Radius, Length or Width and Height in
 * world units, Turned and Center; the rule "Blocks movement and sight. A
 * Warrior's Charge can pass over it. Bodies never block sight."; and for a thin
 * wall the note "True width 0.01 units. Drawn thicker so you can see it." No
 * internal ID such as "base-sensor-obstacle-3" ever reaches the player.
 *
 * Who calls it: `scene.js` calls records, then build, and puts the nodes above
 * the floor, the Red Zone and the sight veil, and below the Strike Band and the
 * bodies. `visual-filters.js` reads PAINT_PARTS. Colors come from the stone
 * tokens in `styles/obstacles.css`.
 *
 * Boundary: stones are the same in every view (the map record is public).
 * records reads only `map.obstacles`, the projection and the visual filter
 * switches; never a champion, a researcher view or a match summary. No collision
 * or sight is computed. Static: no timers, no animations.
 */
import { createSymbolDescriptor, registerTooltipOwner } from "../tooltip.js";
import {
  asArray,
  finiteNumber,
  isRecord,
  setNodeAttributes,
  svgNode,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {readonly [number, number]} Pair
 */

/**
 * Store numbers for the Standing Stones (design store `skins/obstacles.js`).
 * Lengths named `...Units` are world units; the rest are screen px.
 */
export const FOOTPRINT = Object.freeze({
  thinWallUnits: 0.1,
  shadowDropUnits: 0.06,
  shadowOpacity: 0.55,
  wallTopInsetUnits: 0.12,
  wallTopLiftUnits: 0.04,
  bodyLine: 1.5,
  bodyLineSmall: 1,
  detailFull: 40,
  detailRelief: 22,
  hitLine: 10,
  endToleranceUnits: 0.02,
});

/**
 * Node allowance: one record (a whole map's stones) builds at most one group,
 * one shadow window and path, three blade paths, the bodies, hit lines, tops and
 * caps. 64 covers the largest recorded maps. No animations.
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 64,
  animationsPerRecord: 0,
  persistentNodesPerRecord: 0,
});

/**
 * Stones are a game fact, so they are always drawn: no paint part is switched
 * off. The three Standing Stones switches (obstacle_skins, obstacle_relief,
 * obstacle_shadows) change only how they look and are read by records().
 */
/** @type {ReadonlyArray<import("./look-support.js").PaintPartRegistration>} */
export const PAINT_PARTS = Object.freeze([]);

/**
 * The one rule sentence every stone's tooltip carries (the design store's
 * Obstacles spec, word for word).
 */
export const STONE_RULE =
  "Blocks movement and sight. A Warrior's Charge can pass over it. Bodies never block sight.";

/**
 * The plain title of each stone kind: the design store's tooltip names.
 */
export const STONE_TITLES = Object.freeze({
  pillar: "Pillar",
  wall: "Wall",
  thin: "Thin Wall",
});

/**
 * Read one switch from a visual filter state.
 *
 * filters is the page's filter state or null; id is a switch id; fallback is
 * its documented default. Return the switch's boolean when present, otherwise
 * fallback. Pure.
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
 * Round a screen number to 0.001 px for stable, short path text. Pure.
 *
 * @param {number} value
 */
function fixed(value) {
  return Number(value.toFixed(3));
}

/**
 * Return polygon path text "M x,y L ... Z" for screen points. Pure.
 *
 * @param {ReadonlyArray<Pair>} points
 */
function polygonPath(points) {
  return `M${points.map(([x, y]) => `${fixed(x)},${fixed(y)}`).join("L")}Z`;
}

/**
 * Return a full circle as two arcs of path text. Pure.
 *
 * @param {number} cx
 * @param {number} cy
 * @param {number} r
 */
function circlePath(cx, cy, r) {
  return `M${fixed(cx - r)},${fixed(cy)}a${fixed(r)},${fixed(r)} 0 1 0 ${fixed(2 * r)},0a${fixed(r)},${fixed(r)} 0 1 0 ${fixed(-2 * r)},0Z`;
}

/**
 * Return the four screen corners of a recorded wall, inset by inset world units.
 *
 * wall has center [x, y], width, height and theta (radians, world y up);
 * transform projects. Return null when the inset leaves no rectangle. Pure.
 *
 * @param {Record<string, any>} wall
 * @param {LookTransform} transform
 * @param {number} inset
 * @returns {Pair[] | null}
 */
function wallCorners(wall, transform, inset) {
  const [x, y] = asArray(wall.center).map((value) => finiteNumber(value));
  const theta = finiteNumber(wall.theta);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const hx = finiteNumber(wall.width) / 2 - inset;
  const hy = finiteNumber(wall.height) / 2 - inset;
  if (!(hx > 0) || !(hy > 0)) return null;
  return [
    [-hx, -hy],
    [hx, -hy],
    [hx, hy],
    [-hx, hy],
  ].map(([lx, ly]) => {
    const point = transform.worldToScreen([x + lx * c - ly * s, y + lx * s + ly * c]);
    return /** @type {Pair} */ ([point.x, point.y]);
  });
}

/**
 * Return the world distance from point p to segment a-b. Pure.
 *
 * @param {Pair} p
 * @param {Pair} a
 * @param {Pair} b
 */
function segmentDistance(p, a, b) {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const length2 = abx * abx + aby * aby;
  const t =
    length2 > 0
      ? Math.max(0, Math.min(1, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / length2))
      : 0;
  return Math.hypot(p[0] - a[0] - t * abx, p[1] - a[1] - t * aby);
}

/**
 * Return whether world point p lies inside a recorded wall grown by tol units.
 *
 * @param {Pair} p
 * @param {Record<string, any>} wall
 * @param {number} tol
 */
function insideWall(p, wall, tol) {
  const [x, y] = asArray(wall.center).map((value) => finiteNumber(value));
  const theta = finiteNumber(wall.theta);
  const c = Math.cos(-theta);
  const s = Math.sin(-theta);
  const dx = p[0] - x;
  const dy = p[1] - y;
  return (
    Math.abs(dx * c - dy * s) <= finiteNumber(wall.width) / 2 + tol &&
    Math.abs(dx * s + dy * c) <= finiteNumber(wall.height) / 2 + tol
  );
}

/**
 * Classify one end of a thin wall: "edge" (on the map edge), "joined" (it meets
 * another stone) or "free". Pure.
 *
 * @param {Pair} end World point.
 * @param {Record<string, any>} self The thin wall's own row.
 * @param {ReadonlyArray<Record<string, any>>} rows Every stone row.
 * @param {{width: number, height: number}} size World size of the map.
 * @param {ReadonlyArray<{row: Record<string, any>, a: Pair, b: Pair}>} thins
 */
function endKind(end, self, rows, size, thins) {
  const tol = FOOTPRINT.endToleranceUnits;
  if (
    end[0] <= tol ||
    end[0] >= size.width - tol ||
    end[1] <= tol ||
    end[1] >= size.height - tol
  ) {
    return "edge";
  }
  for (const row of rows) {
    if (row === self) continue;
    const center = asArray(row.center).map((value) => finiteNumber(value));
    if (
      row.kind === "pillar" &&
      Math.hypot(end[0] - center[0], end[1] - center[1]) <=
        finiteNumber(row.radius) + tol
    ) {
      return "joined";
    }
    if (row.kind === "wall") {
      const thin = thins.find((entry) => entry.row === row);
      if (
        thin
          ? segmentDistance(end, thin.a, thin.b) <= tol + 0.01
          : insideWall(end, row, tol)
      ) {
        return "joined";
      }
    }
  }
  return "free";
}

/**
 * Return the ends of a thin wall's center line in world units: along its longer
 * side, turned by theta. Pure.
 *
 * @param {Record<string, any>} wall
 */
function thinWallEnds(wall) {
  const [x, y] = asArray(wall.center).map((value) => finiteNumber(value));
  const width = finiteNumber(wall.width);
  const height = finiteNumber(wall.height);
  const theta = finiteNumber(wall.theta);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const along = height >= width;
  const length = Math.max(width, height);
  /** @param {number} lx @param {number} ly @returns {Pair} */
  const turn = (lx, ly) => [x + lx * c - ly * s, y + lx * s + ly * c];
  return {
    a: along ? turn(0, -length / 2) : turn(-length / 2, 0),
    b: along ? turn(0, length / 2) : turn(length / 2, 0),
    length,
  };
}

/**
 * Format the stone's turn for the tooltip: "0°" or "30° (0.524 rad)". Pure.
 *
 * @param {number} theta Radians.
 */
function turnText(theta) {
  if (!theta) return "0°";
  return `${Math.round((theta * 180) / Math.PI)}° (${Number(theta.toFixed(3))} rad)`;
}

/**
 * Return a stone's plain number, for its tooltip's Obstacle ID row and its
 * screen-reader label ("Pillar 3").
 *
 * The number is the recorded obstacle slot that ends the row's obstacle_id:
 * "base-sensor-obstacle-3" (Agent POV), "obstacle-3" (Oracle View) and
 * "pov-obstacle-3" all give "3", so every view names a stone the same way.
 * When the ID is missing or ends in no digits, the number is index, the row's
 * place in map.obstacles counted from 0 (as the design store numbers stones).
 * Pure.
 *
 * @param {Record<string, any>} row The recorded stone.
 * @param {number} index The row's place in map.obstacles, from 0.
 * @returns {string} Digits only, for example "3".
 */
function stoneNumber(row, index) {
  const match = /(\d+)$/u.exec(
    typeof row.obstacle_id === "string" ? row.obstacle_id : "",
  );
  return match ? String(Number(match[1])) : String(index);
}

/**
 * Write a length in world units for a tooltip: "1 unit", "0.5 units",
 * "2 units". The value is rounded to 3 decimals; a value that is not a finite
 * number counts as 0. Pure.
 *
 * @param {number} value World units.
 * @returns {string}
 */
function unitsText(value) {
  const rounded = Number(finiteNumber(value).toFixed(3));
  return `${rounded} ${rounded === 1 ? "unit" : "units"}`;
}

/**
 * Return the tooltip rows for one recorded stone, in the design store's order:
 * Obstacle ID; then Radius (a pillar), Length (a thin wall) or Width and
 * Height (a wall), in world units; then Turned (walls and thin walls only,
 * "0°" or "45° (0.785 rad)"); then Center "(x, y)" in world units. Pure.
 *
 * @param {"pillar" | "wall" | "thin"} kind Which stone shape the row is.
 * @param {Record<string, any>} row The recorded stone.
 * @param {string} number The stone's plain number (stoneNumber).
 * @param {number | null} length Thin wall length in world units, else null.
 * @returns {ReadonlyArray<{label: string, value: string}>}
 */
function stoneFacts(kind, row, number, length) {
  const center = asArray(row.center).map((value) =>
    Number(finiteNumber(value).toFixed(3)),
  );
  /** @type {Array<{label: string, value: string}>} */
  const facts = [{ label: "Obstacle ID", value: number }];
  if (kind === "pillar") {
    facts.push({ label: "Radius", value: unitsText(row.radius) });
  } else if (kind === "thin") {
    facts.push({ label: "Length", value: unitsText(length ?? 0) });
  } else {
    facts.push(
      { label: "Width", value: unitsText(row.width) },
      { label: "Height", value: unitsText(row.height) },
    );
  }
  if (kind !== "pillar") {
    facts.push({ label: "Turned", value: turnText(finiteNumber(row.theta)) });
  }
  facts.push({ label: "Center", value: `(${center[0] ?? 0}, ${center[1] ?? 0})` });
  return Object.freeze(facts.map((fact) => Object.freeze(fact)));
}

/**
 * Return the thin-wall note for its tooltip: "True width 0.01 units. Drawn
 * thicker so you can see it." The width is the wall's shorter recorded side
 * in world units. Pure.
 *
 * @param {Record<string, any>} row The recorded thin wall.
 * @returns {string}
 */
function thinWallNote(row) {
  const width = Math.min(finiteNumber(row.width), finiteNumber(row.height));
  return `True width ${unitsText(width)}. Drawn thicker so you can see it.`;
}

/**
 * Return the drawing records for every recorded stone.
 *
 * input.map.obstacles holds the recorded stones (pillar rows: center [x, y] and
 * radius; wall rows: center, width, height and theta in radians, world y up);
 * input.map.width and input.map.height are the world size (defaults: the
 * transform's world size); input.transform projects them and gives the screen
 * scale; input.filters is the visual filter state (null means every switch at
 * its default: obstacle_skins, obstacle_relief and obstacle_shadows on).
 *
 * Return a frozen list (empty without a transform):
 * - Skins off: one {kind: "plain-pillar", cx, cy, r, label, title, facts} or
 *   {kind: "plain-wall", points, label, title, facts (and note for a thin
 *   wall)} per stone, on the true footprint.
 * - Skins on: one {kind: "stone-shadow", window, d} record when Obstacle Shadows
 *   is on and there is a stone (window is the floor rectangle; d the merged
 *   shadow path in window coordinates); one {kind: "blade", edge, core, ends,
 *   edgeWidth, coreWidth, hit, label, title, note, facts} per thin wall; one
 *   {kind: "pillar", cx, cy, r, lineWidth, label, title, facts} or {kind:
 *   "wall", points, lineWidth, label, title, facts} per body; then with
 *   Obstacle Relief one {kind: "wall-top", d} and one {kind: "pillar-cap", d}
 *   record when any exist.
 * All numbers are screen px. title is the store's plain name ("Pillar",
 * "Wall" or "Thin Wall", STONE_TITLES); label is the title and the stone's
 * plain number ("Pillar 3", the screen-reader name; the number is the recorded
 * obstacle slot, see stoneNumber); facts are the tooltip rows (stoneFacts);
 * note is the thin-wall line "True width 0.01 units. Drawn thicker so you can
 * see it.". Rows that are not objects, or have another kind or no size, are
 * skipped. Pure data: no DOM. Agent POV: identical records in every view (the
 * number ignores the view's ID prefix).
 *
 * @param {{
 *   map?: {obstacles?: unknown, width?: number, height?: number} | null,
 *   transform?: LookTransform | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const transform = input.transform;
  if (!transform) {
    return Object.freeze([]);
  }
  const recorded = asArray(input.map?.obstacles);
  const rows = recorded.filter(
    (row) => isRecord(row) && (row.kind === "pillar" || row.kind === "wall"),
  );
  const size = {
    width: finiteNumber(input.map?.width, transform.worldWidth),
    height: finiteNumber(input.map?.height, transform.worldHeight),
  };
  const ppu = transform.worldLengthToScreen(1);
  const skins = switchOn(input.filters, "obstacle_skins", true);
  const relief =
    switchOn(input.filters, "obstacle_relief", true) && ppu >= FOOTPRINT.detailRelief;
  const shadows = switchOn(input.filters, "obstacle_shadows", true);
  const lineWidth =
    ppu >= FOOTPRINT.detailRelief ? FOOTPRINT.bodyLine : FOOTPRINT.bodyLineSmall;
  const edgeWidth =
    ppu >= FOOTPRINT.detailFull ? 6 : ppu >= FOOTPRINT.detailRelief ? 4.5 : 3;
  const coreWidth =
    ppu >= FOOTPRINT.detailFull ? 3 : ppu >= FOOTPRINT.detailRelief ? 1.5 : 1;
  const endLength =
    ppu >= FOOTPRINT.detailFull ? 4 : ppu >= FOOTPRINT.detailRelief ? 3 : 0;
  const thins = rows
    .filter(
      (row) =>
        row.kind === "wall" &&
        finiteNumber(row.width) > 0 &&
        finiteNumber(row.height) > 0 &&
        Math.min(finiteNumber(row.width), finiteNumber(row.height)) <
          FOOTPRINT.thinWallUnits,
    )
    .map((row) => ({ row, ...thinWallEnds(row) }));
  /** @type {LookRecord[]} */
  const result = [];
  /** @type {LookRecord[]} */
  const bodies = [];
  /** @type {LookRecord[]} */
  const blades = [];
  const drop = Math.max(1, FOOTPRINT.shadowDropUnits * ppu);
  const bounds = transform.mapBounds;
  let shadow = "";
  let tops = "";
  let caps = "";
  for (const row of rows) {
    const number = stoneNumber(row, recorded.indexOf(row));
    const center = transform.worldToScreen(
      /** @type {Pair} */ (
        /** @type {unknown} */ (asArray(row.center).map((value) => finiteNumber(value)))
      ),
    );
    if (row.kind === "pillar") {
      const radius = finiteNumber(row.radius);
      if (!(radius > 0)) continue;
      const r = transform.worldLengthToScreen(radius);
      const facts = stoneFacts("pillar", row, number, null);
      const title = STONE_TITLES.pillar;
      if (!skins) {
        bodies.push(
          Object.freeze({
            kind: "plain-pillar",
            cx: center.x,
            cy: center.y,
            r,
            label: `${title} ${number}`,
            title,
            facts,
          }),
        );
        continue;
      }
      bodies.push(
        Object.freeze({
          kind: "pillar",
          cx: center.x,
          cy: center.y,
          r: Math.max(0.5, r - lineWidth / 2),
          lineWidth,
          label: `${title} ${number}`,
          title,
          facts,
        }),
      );
      shadow += circlePath(center.x - bounds.left, center.y + drop - bounds.top, r);
      const outer = r - 1.5;
      if (relief && outer >= 3) {
        const inner = 1.075 * outer;
        caps += `M${fixed(center.x - outer)},${fixed(center.y)}A${fixed(outer)},${fixed(outer)} 0 0 1 ${fixed(center.x + outer)},${fixed(center.y)}A${fixed(inner)},${fixed(inner)} 0 0 0 ${fixed(center.x - outer)},${fixed(center.y)}Z`;
      }
      continue;
    }
    const width = finiteNumber(row.width);
    const height = finiteNumber(row.height);
    if (!(width > 0) || !(height > 0)) continue;
    const thin = thins.find((entry) => entry.row === row);
    if (thin) {
      const facts = stoneFacts("thin", row, number, thin.length);
      const title = STONE_TITLES.thin;
      const note = thinWallNote(row);
      const a = transform.worldToScreen(thin.a);
      const b = transform.worldToScreen(thin.b);
      if (!skins) {
        const corners = wallCorners(row, transform, 0);
        if (corners)
          bodies.push(
            Object.freeze({
              kind: "plain-wall",
              points: polygonPath(corners),
              label: `${title} ${number}`,
              title,
              note,
              facts,
            }),
          );
        continue;
      }
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const length = Math.hypot(dx, dy);
      if (length < 1e-6) continue;
      const u = [dx / length, dy / length];
      const n = [-u[1], u[0]];
      const half = edgeWidth / 2;
      const kindA = endKind(thin.a, row, rows, size, thins);
      const kindB = endKind(thin.b, row, rows, size, thins);
      /** @param {{x: number, y: number}} p @param {number} k @returns {Pair} */
      const along = (p, k) => [p.x + u[0] * k, p.y + u[1] * k];
      const edgeA = kindA === "joined" ? along(a, -half) : along(a, 0);
      const edgeB = kindB === "joined" ? along(b, half) : along(b, 0);
      const coreA = kindA === "joined" ? edgeA : along(a, 1.5);
      const coreB = kindB === "joined" ? edgeB : along(b, -1.5);
      /** @type {string[]} */
      const ends = [];
      if (relief && endLength > 0 && length >= 2 * (1.5 + endLength) + 2) {
        for (const [
          kind,
          from,
          sign,
        ] of /** @type {Array<[string, {x: number, y: number}, number]>} */ ([
          [kindA, a, 1],
          [kindB, b, -1],
        ])) {
          if (kind !== "free") continue;
          const p = along(from, sign * 1.5);
          const q = along(from, sign * (1.5 + endLength));
          ends.push(`M${fixed(p[0])},${fixed(p[1])}L${fixed(q[0])},${fixed(q[1])}`);
        }
      }
      blades.push(
        Object.freeze({
          kind: "blade",
          edge: `M${fixed(edgeA[0])},${fixed(edgeA[1])}L${fixed(edgeB[0])},${fixed(edgeB[1])}`,
          core:
            Math.hypot(coreB[0] - coreA[0], coreB[1] - coreA[1]) > 0.5
              ? `M${fixed(coreA[0])},${fixed(coreA[1])}L${fixed(coreB[0])},${fixed(coreB[1])}`
              : "",
          ends: ends.join(""),
          edgeWidth,
          coreWidth,
          hit: Object.freeze({ x1: a.x, y1: a.y, x2: b.x, y2: b.y }),
          label: `${title} ${number}`,
          title,
          note,
          facts,
        }),
      );
      const quad = /** @type {Pair[]} */ ([
        [edgeA[0] + n[0] * half, edgeA[1] + n[1] * half],
        [edgeB[0] + n[0] * half, edgeB[1] + n[1] * half],
        [edgeB[0] - n[0] * half, edgeB[1] - n[1] * half],
        [edgeA[0] - n[0] * half, edgeA[1] - n[1] * half],
      ]);
      shadow += polygonPath(
        quad.map(([x, y]) => [x - bounds.left, y + drop - bounds.top]),
      );
      continue;
    }
    const facts = stoneFacts("wall", row, number, null);
    const title = STONE_TITLES.wall;
    if (!skins) {
      const corners = wallCorners(row, transform, 0);
      if (corners)
        bodies.push(
          Object.freeze({
            kind: "plain-wall",
            points: polygonPath(corners),
            label: `${title} ${number}`,
            title,
            facts,
          }),
        );
      continue;
    }
    const corners = wallCorners(row, transform, lineWidth / 2 / ppu);
    const footprint = wallCorners(row, transform, 0);
    if (!corners || !footprint) continue;
    bodies.push(
      Object.freeze({
        kind: "wall",
        points: polygonPath(corners),
        lineWidth,
        label: `${title} ${number}`,
        title,
        facts,
      }),
    );
    shadow += polygonPath(
      footprint.map(([x, y]) => [x - bounds.left, y + drop - bounds.top]),
    );
    const short = Math.min(width, height);
    if (relief && short >= 0.4 && (short - 0.24) * ppu >= 4) {
      const top = wallCorners(row, transform, FOOTPRINT.wallTopInsetUnits);
      const lift = -FOOTPRINT.wallTopLiftUnits * ppu;
      if (top) tops += polygonPath(top.map(([x, y]) => [x, y + lift]));
    }
  }
  if (skins && shadows && shadow) {
    result.push(
      Object.freeze({
        kind: "stone-shadow",
        window: Object.freeze({
          x: bounds.left,
          y: bounds.top,
          width: bounds.width,
          height: bounds.height,
        }),
        d: shadow,
        opacity: FOOTPRINT.shadowOpacity,
      }),
    );
  }
  result.push(...blades, ...bodies);
  if (tops) result.push(Object.freeze({ kind: "wall-top", d: tops }));
  if (caps) result.push(Object.freeze({ kind: "pillar-cap", d: caps }));
  return Object.freeze(result);
}

/**
 * Write one record's geometry onto its node. Used by build and update. Returns
 * undefined; changes the DOM only.
 *
 * @param {Element} node
 * @param {LookRecord} record
 */
function writeGeometry(node, record) {
  switch (record.kind) {
    case "stone-shadow": {
      setNodeAttributes(node, record.window);
      const path = node.firstElementChild;
      if (path)
        setNodeAttributes(path, { d: record.d, "fill-opacity": record.opacity });
      break;
    }
    case "blade": {
      const [edge, core, ends, hit] = [...node.children];
      if (edge)
        setNodeAttributes(edge, { d: record.edge, "stroke-width": record.edgeWidth });
      if (core)
        setNodeAttributes(core, {
          d: record.core || null,
          "stroke-width": record.coreWidth,
        });
      if (ends)
        setNodeAttributes(ends, {
          d: record.ends || null,
          "stroke-width": record.coreWidth,
        });
      if (hit) setNodeAttributes(hit, record.hit);
      break;
    }
    case "pillar":
    case "plain-pillar":
      setNodeAttributes(node, {
        cx: record.cx,
        cy: record.cy,
        r: record.r,
        "data-detail": record.lineWidth === FOOTPRINT.bodyLineSmall ? "fine" : null,
      });
      break;
    case "wall":
    case "plain-wall":
      setNodeAttributes(node, {
        d: record.points,
        "data-detail": record.lineWidth === FOOTPRINT.bodyLineSmall ? "fine" : null,
      });
      break;
    default:
      setNodeAttributes(node, { d: record.d });
  }
}

/**
 * Build the stone nodes, in record order, for the `obstacle` layer.
 *
 * records come from records(); doc creates the nodes. Each record gives one
 * node: a nested `svg.stone-shadow-window` holding one `path.stone-shadow`; a
 * `g.stone-blade` holding its edge, core and End Stone paths and a transparent
 * `line.stone-blade__hit` (focusable, with the tooltip); a `circle.obstacle` or
 * `path.obstacle` body (focusable, role "img", aria-label and tooltip); and the
 * `path.stone-wall-top` and `path.stone-cap` relief paths (pointer-events none).
 * Tooltips are registered through `tooltip.js` (a DOM side effect). Export-safe:
 * flat fills and strokes only, no ids, clip paths or gradients. Return the
 * nodes.
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
    if (record.kind === "stone-shadow") {
      node = svgNode(doc, "svg", {
        class: "stone-shadow-window",
        "aria-hidden": "true",
      });
      node.append(svgNode(doc, "path", { class: "stone-shadow" }));
    } else if (record.kind === "blade") {
      node = svgNode(doc, "g", { class: "stone-blade" });
      const hit = svgNode(doc, "line", {
        class: "obstacle stone-blade__hit",
        role: "img",
        tabindex: "0",
        "aria-label": record.label,
      });
      node.append(
        svgNode(doc, "path", { class: "stone-blade__edge", "aria-hidden": "true" }),
        svgNode(doc, "path", { class: "stone-blade__core", "aria-hidden": "true" }),
        svgNode(doc, "path", { class: "stone-blade__end", "aria-hidden": "true" }),
        hit,
      );
      registerTooltipOwner(hit, tooltip(record));
    } else if (record.kind === "pillar" || record.kind === "plain-pillar") {
      node = svgNode(doc, "circle", {
        class: `obstacle ${record.kind === "pillar" ? "standing-stone standing-stone--pillar" : "plain-stone"}`,
        role: "img",
        tabindex: "0",
        "aria-label": record.label,
      });
      registerTooltipOwner(node, tooltip(record));
    } else if (record.kind === "wall" || record.kind === "plain-wall") {
      node = svgNode(doc, "path", {
        class: `obstacle ${record.kind === "wall" ? "standing-stone standing-stone--wall" : "plain-stone"}`,
        role: "img",
        tabindex: "0",
        "aria-label": record.label,
      });
      registerTooltipOwner(node, tooltip(record));
    } else if (record.kind === "wall-top" || record.kind === "pillar-cap") {
      node = svgNode(doc, "path", {
        class: record.kind === "wall-top" ? "stone-wall-top" : "stone-cap",
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
 * Move existing stone nodes to new records (reprojection after a resize or a
 * camera change).
 *
 * nodes are build()'s nodes and records the new records in the same order (the
 * same map and switches, so the same kinds). Write the new geometry and line
 * widths. Tooltips are unchanged. Return undefined; DOM change only.
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
 * Return the Symbol Tip for a stone record, or null for any other record.
 *
 * record is a pillar, wall, blade or plain stone record from records(). The
 * tip is the "Stone" symbol titled with the store's plain name ("Pillar",
 * "Wall" or "Thin Wall"), with the line STONE_RULE ("Blocks movement and
 * sight. A Warrior's Charge can pass over it. Bodies never block sight."),
 * the rows Obstacle ID (a plain number), size (Radius, Length, or Width and
 * Height, in world units), Turned and Center, the symbol's own "What It
 * Means" row, and for a thin wall the Note "True width 0.01 units. Drawn
 * thicker so you can see it.". It carries no internal ID. Pure; the same in
 * every view.
 *
 * @param {LookRecord} record
 * @returns {ReturnType<typeof createSymbolDescriptor> | null}
 */
export function tooltip(record) {
  if (!Array.isArray(record?.facts)) {
    return null;
  }
  return createSymbolDescriptor({
    symbol: "stone",
    id: `obstacle:${record.label}`,
    title: typeof record.title === "string" ? record.title : null,
    line: STONE_RULE,
    facts: record.facts,
    note: typeof record.note === "string" ? record.note : null,
  });
}
