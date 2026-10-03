/**
 * @file The Team Identity look (Homeland Brackets): a bracket outside each home
 * end, in the `team-identity` layer.
 *
 * Each team's bracket runs outside the floor along its home end and turns along
 * both long sides as far as that team's recorded Red Zone reaches (only the end
 * when the map has no Red Zone). Team A is one 2 px line; Team B is two 1 px
 * lines 3 px apart (Team B is always double-lined). Both are in the team ring
 * colour at 70%. A crest with the letter "A" or "B" breaks the end line at
 * mid-height. The store's two sizes fit the room outside the floor (FOOTPRINT,
 * markSize): with a 24 px margin (the Replay Viewer) the 16 px crest 14 px out;
 * with less (the BattleClient and the DevClient pad their field by 16/1032 of
 * its width) the store stage's 14 px crest 8.5 px out, scaled down below a
 * 16.4 px margin, so the crest and both lines always stay whole inside the arena
 * box. In Agent POV the POV team's crest gets a 1 px Moon Silver rim (the POV
 * Home Mark). The brackets follow the recorded side of each team (its Red
 * Zone, else its spawn pads), never the screen side; when the record is unclear
 * nothing is drawn. Static: no timers, no animations. With Home End Marks off
 * nothing is drawn (team colour still shows on every token ring).
 *
 * Who calls it: `scene.js` calls records and build each render and puts the
 * nodes in the `team-identity` layer, outside the floor. `visual-filters.js`
 * reads PAINT_PARTS.
 *
 * Boundary: records reads only the map record (`red_zone`), the spawn-pad
 * positions of the arena scene, the projection (and its margin), the filter
 * switches and the POV
 * team the caller names; the same in every view except the POV Home Mark. It
 * never reads a champion's position, a researcher view or a match summary.
 */
import { teamIdentityFromId } from "../agent-identity.js";
import { createSymbolDescriptor, registerTooltipOwner } from "../tooltip.js";
import {
  asArray,
  finiteNumber,
  isRecord,
  paintParts,
  setNodeAttributes,
  svgNode,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 */

/**
 * Store numbers for the Homeland Brackets (design store `skins/team-identity.js`).
 * All lengths are screen px. The store has two sizes of the same mark, chosen
 * by the room outside the floor (the margin):
 * - `product`: the spec's 16 px crest centered 14 px outside the floor, with
 *   the line broken 11 px either side of it. It needs a margin of at least
 *   `product.margin` (24 px), as the Replay Viewer's 24 px pad gives.
 * - `stage`: the store stage's 14 px crest centered 8.5 px outside the floor,
 *   broken 10 px either side, made for the store's 16.4 px margin
 *   (`stage.margin`). The BattleClient and the DevClient pad their field by
 *   16/1032 of its width (about 14 px at 1440 x 900), so they use this size,
 *   and below 16.4 px they draw it scaled by margin / 16.4, so the crest and
 *   both of Team B's lines stay whole inside the arena box.
 * teamBOffset is half the 3 px gap between Team B's two lines (scaled with the
 * stage size); the lines stay 2 px (Team A) and 1 px (Team B) wide at 70%.
 */
export const FOOTPRINT = Object.freeze({
  teamBOffset: 1.5,
  opacity: 0.7,
  product: Object.freeze({
    margin: 24,
    axis: 14,
    crestBreak: 11,
    crest: Object.freeze({
      disc: 5,
      ring: 6.5,
      ringWidth: 3,
      gapWidth: 1,
      outer: 8,
      letter: 11,
      baseline: 3.9,
      keyline: 0.7,
    }),
  }),
  stage: Object.freeze({
    margin: 16.4,
    axis: 8.5,
    crestBreak: 10,
    crest: Object.freeze({
      disc: 4.6,
      ring: 5.75,
      ringWidth: 2.5,
      gapWidth: 1,
      outer: 7,
      letter: 9.6,
      baseline: 3.4,
      keyline: 0.7,
    }),
  }),
});

/**
 * @typedef {Readonly<{
 *   disc: number,
 *   ring: number,
 *   ringWidth: number,
 *   gapWidth: number,
 *   outer: number,
 *   letter: number,
 *   baseline: number,
 *   keyline: number,
 * }>} CrestSize
 *   One crest's sizes in screen px: the field disc radius, the ring radius and
 *   stroke width, Team B's Ink gap stroke width, the outer radius, the letter's
 *   font size and baseline drop, and the Ink keyline width (the keyline disc
 *   has radius outer + keyline).
 * @typedef {Readonly<{
 *   axis: number,
 *   crestBreak: number,
 *   teamBOffset: number,
 *   crest: CrestSize,
 * }>} MarkSize
 *   The bracket axis (px outside the floor), the line break either side of the
 *   crest center, half of Team B's line gap and the crest sizes.
 */

/**
 * Return the room (screen px) outside the floor for the marks: the smallest gap
 * between the shown frame and the viewport edge on any side. The frame is the
 * transform's camera frame when it has one (so a camera move never changes the
 * marks' size), else the floor (mapBounds). Pure.
 *
 * @param {LookTransform & {cameraFrame?: {minX: number, minY: number, maxX: number, maxY: number}}} transform
 * @returns {number} The margin in CSS px (may be 0 or negative for a frame that
 *   fills or overflows the viewport).
 */
function markMargin(transform) {
  const view = transform.viewportBounds;
  const frame = transform.cameraFrame;
  let box = transform.mapBounds;
  if (
    frame &&
    [frame.minX, frame.minY, frame.maxX, frame.maxY].every(Number.isFinite)
  ) {
    const a = transform.worldToScreen([frame.minX, frame.maxY]);
    const b = transform.worldToScreen([frame.maxX, frame.minY]);
    box = {
      left: Math.min(a.x, b.x),
      top: Math.min(a.y, b.y),
      right: Math.max(a.x, b.x),
      bottom: Math.max(a.y, b.y),
      width: Math.abs(b.x - a.x),
      height: Math.abs(b.y - a.y),
    };
  }
  if (!view) return FOOTPRINT.product.margin;
  return Math.min(
    box.left - view.left,
    box.top - view.top,
    view.right - box.right,
    view.bottom - box.bottom,
  );
}

/**
 * Return the mark size for a margin (px): the product size from
 * FOOTPRINT.product.margin up, else the store stage size, scaled by
 * margin / FOOTPRINT.stage.margin below that margin (never above 1, never
 * below 0). Every length scales except the bracket line widths. Pure.
 *
 * @param {number} margin
 * @returns {MarkSize}
 */
export function markSize(margin) {
  if (margin >= FOOTPRINT.product.margin) {
    const { axis, crestBreak, crest } = FOOTPRINT.product;
    return Object.freeze({
      axis,
      crestBreak,
      teamBOffset: FOOTPRINT.teamBOffset,
      crest,
    });
  }
  const k = Math.min(1, Math.max(0, margin) / FOOTPRINT.stage.margin);
  const stage = FOOTPRINT.stage;
  const crest = /** @type {CrestSize} */ (
    Object.freeze(
      Object.fromEntries(
        Object.entries(stage.crest).map(([name, value]) => [name, fixed(value * k)]),
      ),
    )
  );
  return Object.freeze({
    axis: fixed(stage.axis * k),
    crestBreak: fixed(stage.crestBreak * k),
    teamBOffset: fixed(FOOTPRINT.teamBOffset * k),
    crest,
  });
}

/** Node allowance per team record: the bracket path and a crest of up to 6 nodes. */
export const BUDGET = Object.freeze({
  nodesPerRecord: 8,
  animationsPerRecord: 0,
  persistentNodesPerRecord: 0,
});

/** Home End Marks switches the brackets and crests. */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "home_end_mark" }, "home_end_marks"],
]);

/**
 * Round to 0.001 for stable path text. Pure.
 *
 * @param {number} value
 */
function fixed(value) {
  return Number(value.toFixed(3));
}

/**
 * Return each team's recorded home side ("left" or "right") and Red Zone range,
 * or null when the record does not decide it. The Red Zone ranges decide first;
 * without them the mean x of each team's spawn pads. Pure.
 *
 * @param {Record<string, any> | null | undefined} redZone
 * @param {ReadonlyArray<unknown>} pads
 * @param {number} width World width.
 */
function homeSides(redZone, pads, width) {
  /** @param {unknown} range */
  const middle = (range) => {
    const values = asArray(range).map((value) => finiteNumber(value, Number.NaN));
    return values.length === 2 && values.every(Number.isFinite) && values[1] > values[0]
      ? (values[0] + values[1]) / 2
      : null;
  };
  let a = isRecord(redZone) ? middle(redZone.team_a_x_range) : null;
  let b = isRecord(redZone) ? middle(redZone.team_b_x_range) : null;
  const zones = a !== null && b !== null;
  if (!zones) {
    /** @param {number} teamId */
    const meanX = (teamId) => {
      const xs = pads
        .filter((pad) => isRecord(pad) && Number(pad.team_id) === teamId)
        .map((pad) =>
          finiteNumber(
            asArray(/** @type {Record<string, any>} */ (pad).position)[0],
            Number.NaN,
          ),
        )
        .filter(Number.isFinite);
      return xs.length === 0 ? null : xs.reduce((sum, x) => sum + x, 0) / xs.length;
    };
    a = meanX(1);
    b = meanX(2);
  }
  if (a === null || b === null) return null;
  const sideA = a > width / 2 ? "right" : "left";
  const sideB = b > width / 2 ? "right" : "left";
  if (sideA === sideB) return null;
  return {
    1: {
      side: sideA,
      range: zones
        ? asArray(/** @type {Record<string, any>} */ (redZone).team_a_x_range)
        : null,
    },
    2: {
      side: sideB,
      range: zones
        ? asArray(/** @type {Record<string, any>} */ (redZone).team_b_x_range)
        : null,
    },
  };
}

/**
 * Return one record per team's home end.
 *
 * input.transform projects (mapBounds is the floor rectangle in screen px; the
 * room between the shown frame and viewportBounds picks the mark size, see
 * markSize);
 * input.map is the map record (its `red_zone`; default input.scene.map);
 * input.scene is the arena scene (only `spawn_pads` positions are read, as a
 * fallback for the home side); input.filters is the visual filter state (null:
 * home_end_marks on); input.audience is "researcher" or "agent_pov";
 * input.povTeamId is the POV recipient's team (1 or 2), used only in Agent POV
 * for the POV Home Mark (default none).
 *
 * Return a frozen list (empty when the switch is off, there is no transform or
 * the record does not decide the sides) of {kind: "homeland-bracket", teamId,
 * side ("left" | "right"), d (bracket path), lineWidth (2 for Team A, 1 for Team
 * B), crestX, crestY, crest (the CrestSize to draw), letter ("A" | "B"), pov
 * (boolean)}, Team B first. All in screen px. Pure data: no DOM.
 *
 * @param {{
 *   transform?: LookTransform | null,
 *   map?: Record<string, any> | null,
 *   scene?: Record<string, any> | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   audience?: "researcher" | "agent_pov" | null,
 *   povTeamId?: number | null,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const transform = input.transform;
  const map = input.map ?? input.scene?.map ?? null;
  if (!transform || input.filters?.home_end_marks === false || (!map && !input.scene)) {
    return Object.freeze([]);
  }
  const sides = homeSides(
    map?.red_zone,
    asArray(input.scene?.spawn_pads),
    transform.worldWidth,
  );
  if (sides === null) {
    return Object.freeze([]);
  }
  const bounds = transform.mapBounds;
  const middle = (bounds.top + bounds.bottom) / 2;
  const size = markSize(markMargin(transform));
  const gap = size.crestBreak;
  /** @type {LookRecord[]} */
  const result = [];
  for (const teamId of /** @type {Array<1 | 2>} */ ([2, 1])) {
    const { side, range } = sides[teamId];
    const left = side === "left";
    const edge = left ? bounds.left : bounds.right;
    const out = left ? -1 : 1;
    const values = range?.map((value) => finiteNumber(value, Number.NaN)) ?? [];
    const inner =
      values.length === 2 && values.every(Number.isFinite)
        ? transform.worldToScreen([left ? values[1] : values[0], 0]).x
        : null;
    const offsets =
      teamId === 1
        ? [size.axis]
        : [size.axis - size.teamBOffset, size.axis + size.teamBOffset];
    let d = "";
    for (const offset of offsets) {
      const x = edge + out * offset;
      if (inner !== null) {
        d += `M${fixed(inner)} ${fixed(bounds.top - offset)}H${fixed(x)}V${fixed(middle - gap)}`;
        d += `M${fixed(x)} ${fixed(middle + gap)}V${fixed(bounds.bottom + offset)}H${fixed(inner)}`;
      } else {
        d += `M${fixed(x)} ${fixed(bounds.top)}V${fixed(middle - gap)}M${fixed(x)} ${fixed(middle + gap)}V${fixed(bounds.bottom)}`;
      }
    }
    result.push(
      Object.freeze({
        kind: "homeland-bracket",
        teamId,
        side,
        d,
        lineWidth: teamId === 1 ? 2 : 1,
        crestX: edge + out * size.axis,
        crestY: middle,
        crest: size.crest,
        letter: teamId === 1 ? "A" : "B",
        pov: input.audience === "agent_pov" && input.povTeamId === teamId,
      }),
    );
  }
  return Object.freeze(result);
}

/**
 * Write one record's geometry onto its group (build and update): the bracket
 * path and its width, then every crest part's center, radius, stroke width and
 * the letter's place and font size from record.crest (the CrestSize the record
 * was made with; FOOTPRINT.product.crest when a record has none). DOM only.
 *
 * @param {Element} group
 * @param {LookRecord} record
 */
function writeGeometry(group, record) {
  /** @type {CrestSize} */
  const c = record.crest ?? FOOTPRINT.product.crest;
  const [line, crest] = [...group.children];
  setNodeAttributes(line, { d: record.d, "stroke-width": record.lineWidth });
  if (!crest) return;
  const center = { cx: record.crestX, cy: record.crestY };
  for (const part of [...crest.children]) {
    const name = part.getAttribute("class") ?? "";
    if (name === "home-crest__text") {
      setNodeAttributes(part, {
        x: record.crestX,
        y: fixed(record.crestY + c.baseline),
        "font-size": c.letter,
      });
    } else if (name === "home-crest__keyline") {
      setNodeAttributes(part, { ...center, r: fixed(c.outer + c.keyline) });
    } else if (name === "home-crest__field") {
      setNodeAttributes(part, { ...center, r: c.disc });
    } else if (name === "home-crest__ring") {
      setNodeAttributes(part, { ...center, r: c.ring, "stroke-width": c.ringWidth });
    } else if (name === "home-crest__gap") {
      setNodeAttributes(part, { ...center, r: c.ring, "stroke-width": c.gapWidth });
    } else if (name === "home-crest__pov") {
      setNodeAttributes(part, {
        ...center,
        r: fixed(c.outer + c.keyline / 2),
        "stroke-width": c.keyline,
      });
    } else {
      setNodeAttributes(part, center);
    }
  }
}

/**
 * Build one `g.homeland-bracket` per record: the bracket path and a crest group
 * (Ink keyline disc, team field disc, ring, Team B's Ink gap ring, the letter,
 * and the Moon Silver rim when pov), sized from record.crest by writeGeometry.
 * The crest carries the team's Symbol Tip, registered through `tooltip.js` (a
 * DOM side effect). Export-safe: flat fills and strokes; no ids or gradients.
 * Return the groups in record order.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @returns {SVGElement[]}
 */
export function build(records, doc) {
  /** @type {SVGElement[]} */
  const nodes = [];
  for (const record of records) {
    if (record.kind !== "homeland-bracket") continue;
    const team = record.teamId === 1 ? "a" : "b";
    const group = svgNode(doc, "g", {
      class: `homeland-bracket homeland-bracket--team-${team}`,
      "data-team-id": record.teamId,
      "data-home-side": record.side,
    });
    const crest = svgNode(doc, "g", {
      class: "home-crest",
      role: "img",
      "aria-label": `${teamIdentityFromId(record.teamId)?.label ?? "Team"} Home End`,
    });
    crest.append(
      svgNode(doc, "circle", { class: "home-crest__keyline" }),
      svgNode(doc, "circle", { class: "home-crest__field" }),
      svgNode(doc, "circle", { class: "home-crest__ring" }),
    );
    if (record.teamId === 2) {
      crest.append(svgNode(doc, "circle", { class: "home-crest__gap" }));
    }
    const letter = svgNode(doc, "text", {
      class: "home-crest__text",
      "text-anchor": "middle",
    });
    letter.textContent = record.letter;
    crest.append(letter);
    if (record.pov) {
      crest.append(svgNode(doc, "circle", { class: "home-crest__pov" }));
    }
    group.append(
      svgNode(doc, "path", { class: "homeland-bracket__line", "aria-hidden": "true" }),
      crest,
    );
    registerTooltipOwner(crest, tooltip(record));
    writeGeometry(group, record);
    nodes.push(group);
  }
  return nodes;
}

/**
 * Move existing bracket groups to new records (reprojection after a resize or a
 * camera change), resizing each crest to its record's CrestSize. nodes and
 * records are in the same order. Return undefined; DOM change only.
 *
 * @param {ReadonlyArray<Element>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  for (const [index, node] of nodes.entries()) {
    const record = records[index];
    if (record?.kind === "homeland-bracket") writeGeometry(node, record);
  }
}

/**
 * Return the team's Symbol Tip for a bracket record, or null for any other
 * record. The line says this is the team's home end and that the bracket reaches
 * as far as its Red Zone; the POV team's tip adds "Your team's home end." Pure.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  if (record?.kind !== "homeland-bracket") {
    return null;
  }
  const label = teamIdentityFromId(record.teamId)?.label ?? "This team";
  return createSymbolDescriptor({
    symbol: record.teamId === 1 ? "team_a" : "team_b",
    id: `home-end:${record.teamId}`,
    line: `${label}'s home end. The bracket reaches as far as its Red Zone. Paired games swap ends.`,
    note: record.pov ? "Your team's home end." : null,
  });
}
