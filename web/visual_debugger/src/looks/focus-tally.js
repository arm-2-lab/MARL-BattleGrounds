/**
 * @file The Focus Tally look: a short curved row of beads under a champion hit
 * by two or more enemies on this tick, one bead per attacker.
 *
 * It owns the drawing of the plan's `focus_tally` rows (switch `focus_fire`,
 * Diagnostics, off by default). `choreography-plan.js` makes the rows (at most
 * four per tick, at the target's END spot, from Impact to the end of the tick,
 * kept in the paused still) and `choreography-painter.js` asks this look to
 * draw each one into the event's group above the bodies.
 *
 * Geometry (store spec, analysis-overlays): N beads (2 to 5) on a ring of 1.5R
 * around the target's centre, each a 20 degree arc with 7 degree gaps, the row
 * centred on 6 o'clock; a 1 px thread arc from 8 degrees before the first bead
 * to 8 degrees after the last; a 5.5 px Ink Shadow under-copy of the beads. The
 * beads take the attacking team's light colour. Detail ladder (body radius R
 * in CSS px): beads 3 px at R 22 or more, 2.5 px from R 14, and 2 px with no
 * thread below R 14. An invisible 8 px hit arc over the bead row (thread
 * span) lets the pointer open the tally's tip; it sits outside the body, so
 * the body under it keeps its clicks.
 *
 * Boundary: records reads only the one admitted plan row it is given. In Agent
 * POV the plan counts a blow only when its source and target were seen, and the
 * host sends no activation from a hidden attacker, so a hidden attacker never
 * earns a bead. It never reads a researcher view or a match summary. Export-safe
 * SVG only: flat strokes, opacity and no fills.
 */
import { createSemanticDescriptor } from "../tooltip.js";
import {
  detachedEventMount,
  finiteNumber,
  isRecord,
  mountNodes,
  paintParts,
  setEffectAttributes,
  svgNode,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").LookInput} LookInput
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").EventMount} EventMount
 */

/** Body radius in CSS px used when a row carries no `anchorRadius`. */
export const DEFAULT_BODY_RADIUS_PX = 14;

/** The tally ring's radius as a share of the body radius (1.5R). */
export const RING_SHARE = 1.5;

/** Each bead's arc length and the gap between beads, in degrees. */
export const BEAD_DEGREES = Object.freeze({ bead: 20, gap: 7, threadOverhang: 8 });

/** Fewest and most beads one tally draws. */
export const BEAD_LIMITS = Object.freeze({ fewest: 2, most: 5 });

/** No allocator footprint: the beads sit just outside the body, on its ring. */
export const FOOTPRINT = Object.freeze({});

/**
 * Nodes per record, counted the way the choreography painter counts a kept
 * row (its event group plus everything inside it): the painter's event group,
 * the `g.focus-tally` group, the under-copy, the thread, the beads path and
 * the hit arc (6). The painter drops the empty connector, so it adds nothing.
 * The painter fades the event group in (1 animation). Rows are persistent
 * (kept in the paused still), so all 6 nodes are retained. A budget below the
 * drawn count made the painter refuse the whole tick ("exceeded the planned
 * persistent node bound").
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 6,
  animationsPerRecord: 1,
  persistentNodesPerRecord: 6,
});

/** The hit arc's stroke width in CSS px (an invisible pointer target). */
export const HIT_STROKE_PX = 8;

/** The Focus Tally switch (`focus_fire`) owns the plan's `focus_tally` rows. */
export const PAINT_PARTS = paintParts([
  [{ surface: "transient", kind: "focus_tally" }, "focus_fire"],
]);

/**
 * Return the bead arcs for N beads, in degrees clockwise from 12 o'clock.
 *
 * count is clamped to BEAD_LIMITS. The row is centred on 180 degrees (6
 * o'clock): the first bead starts at 180 - (27N - 7) / 2. N = 2 gives
 * [156.5, 176.5] and [183.5, 203.5]. Pure.
 *
 * @param {number} count
 * @returns {ReadonlyArray<readonly [number, number]>}
 */
export function beadAngles(count) {
  const n = Math.min(
    BEAD_LIMITS.most,
    Math.max(BEAD_LIMITS.fewest, Math.trunc(finiteNumber(count))),
  );
  const step = BEAD_DEGREES.bead + BEAD_DEGREES.gap;
  const first = 180 - (step * n - BEAD_DEGREES.gap) / 2;
  return Object.freeze(
    Array.from({ length: n }, (_, index) =>
      Object.freeze(
        /** @type {[number, number]} */ ([
          first + index * step,
          first + index * step + BEAD_DEGREES.bead,
        ]),
      ),
    ),
  );
}

/**
 * Return the bead stroke width and whether the thread shows, by body radius.
 *
 * radius is the body radius in CSS px: 3 px beads with a thread at 22 px or
 * more, 2.5 px with a thread from 14 px, 2 px with no thread below 14 px.
 *
 * @param {number} radius
 * @returns {{beadWidth: number, thread: boolean}}
 */
export function detailLadder(radius) {
  if (radius >= 22) return { beadWidth: 3, thread: true };
  if (radius >= 14) return { beadWidth: 2.5, thread: true };
  return { beadWidth: 2, thread: false };
}

/**
 * Return the screen point at angle degrees clockwise from 12 o'clock.
 *
 * @param {{x: number, y: number}} centre
 * @param {number} radius CSS px.
 * @param {number} degrees
 * @returns {{x: number, y: number}}
 */
function pointAt(centre, radius, degrees) {
  const radians = (degrees * Math.PI) / 180;
  return {
    x: round(centre.x + radius * Math.sin(radians)),
    y: round(centre.y - radius * Math.cos(radians)),
  };
}

/**
 * Round to two decimals so the path text stays short and stable.
 *
 * @param {number} value
 * @returns {number}
 */
function round(value) {
  return Math.round(value * 100) / 100;
}

/**
 * Return an SVG arc path drawn clockwise from start to end degrees.
 *
 * @param {{x: number, y: number}} centre
 * @param {number} radius CSS px.
 * @param {number} start Degrees clockwise from 12 o'clock.
 * @param {number} end Degrees, larger than start.
 * @returns {string}
 */
function arcPath(centre, radius, start, end) {
  const from = pointAt(centre, radius, start);
  const to = pointAt(centre, radius, end);
  const large = end - start > 180 ? 1 : 0;
  return `M ${from.x} ${from.y} A ${round(radius)} ${round(radius)} 0 ${large} 1 ${to.x} ${to.y}`;
}

/**
 * Return this look's drawing records for one admitted plan row.
 *
 * input.event must be a `focus_tally` row whose paint part is on
 * (`paintParts.effect` not false) with a screen `anchor` {x, y} (CSS px).
 * Its `count` (distinct attackers, clamped to 2 to 5), `attackingTeamId`
 * (1 Team A, 2 Team B), `targetPublicAgentId`, `lethal` and optional
 * `anchorRadius` (the target's body radius in CSS px; default
 * DEFAULT_BODY_RADIUS_PX) are read. Return one frozen {kind: "focus-tally",
 * event, centre, radius, ringRadius, count, teamKey, beadWidth, beadsPath,
 * underPath, threadPath, hitPath} record, or [] for any other input.
 * threadPath is null when the ladder drops the thread; hitPath, the arc of
 * the hit target, always spans the thread's angles. Pure data: no DOM.
 *
 * @param {LookInput} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const event = input?.event;
  if (
    !isRecord(event) ||
    event.kind !== "focus_tally" ||
    event.paintParts?.effect === false ||
    !isRecord(event.anchor)
  ) {
    return Object.freeze([]);
  }
  const centre = { x: finiteNumber(event.anchor.x), y: finiteNumber(event.anchor.y) };
  const anchorRadius = Number(event.anchorRadius);
  const radius =
    Number.isFinite(anchorRadius) && anchorRadius > 0
      ? anchorRadius
      : DEFAULT_BODY_RADIUS_PX;
  const ringRadius = radius * RING_SHARE;
  const angles = beadAngles(event.count);
  const ladder = detailLadder(radius);
  const beadsPath = angles
    .map(([start, end]) => arcPath(centre, ringRadius, start, end))
    .join(" ");
  const first = angles[0][0];
  const last = angles[angles.length - 1][1];
  const span = arcPath(
    centre,
    ringRadius,
    first - BEAD_DEGREES.threadOverhang,
    last + BEAD_DEGREES.threadOverhang,
  );
  return Object.freeze([
    Object.freeze({
      kind: "focus-tally",
      event,
      centre: Object.freeze(centre),
      radius,
      ringRadius,
      count: angles.length,
      teamKey: event.attackingTeamId === 1 ? "team-a" : "team-b",
      beadWidth: ladder.beadWidth,
      beadsPath,
      underPath: beadsPath,
      threadPath: ladder.thread ? span : null,
      hitPath: span,
    }),
  ]);
}

/**
 * Draw Focus Tally records into the painter's event mount.
 *
 * records come from records(); doc creates nodes; mount is the painter's
 * EventMount (a detached one is made when absent, for pictures outside the
 * arena). Each record becomes one `g.focus-tally` in mount.group (data-team,
 * data-count, data-target-agent-id, data-lethal) holding, bottom to top, the
 * Ink Shadow under-copy, the thread (when the ladder keeps it), the beads and
 * the invisible hit arc (`focus-tally__hit`, HIT_STROKE_PX wide; the only part
 * that takes the pointer). Widths are written as attributes; colours and
 * pointer rules come from focus-tally.css. The
 * painter fades the group in at Impact; nothing here animates. Return the
 * mount's containers (mountNodes), or [] for no records.
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
  for (const each of records) {
    const group = svgNode(doc, "g", {
      class: "focus-tally",
      "data-team": each.teamKey,
      "data-count": each.count,
      "data-target-agent-id": String(each.event.targetPublicAgentId ?? ""),
      "data-lethal": String(each.event.lethal === true),
      "aria-hidden": "true",
    });
    group.append(
      svgNode(doc, "path", {
        class: "focus-tally__under",
        d: each.underPath,
        "stroke-width": 5.5,
      }),
    );
    if (each.threadPath !== null) {
      group.append(
        svgNode(doc, "path", {
          class: "focus-tally__thread",
          d: each.threadPath,
          "stroke-width": 1,
        }),
      );
    }
    group.append(
      svgNode(doc, "path", {
        class: "focus-tally__beads",
        d: each.beadsPath,
        "stroke-width": each.beadWidth,
      }),
      svgNode(doc, "path", {
        class: "focus-tally__hit",
        d: each.hitPath,
        "stroke-width": HIT_STROKE_PX,
      }),
    );
    target.group.append(group);
  }
  return mountNodes(target);
}

/**
 * Move drawn tallies to new records (reprojection after a resize or camera
 * change).
 *
 * nodes are the containers build returned (the event group first); records
 * come from records() for the same event. Each `g.focus-tally` in the group
 * gets its record's new paths (under-copy, thread, beads and hit arc) and
 * bead width, in order. Return undefined. DOM changes only.
 *
 * @param {ReadonlyArray<Element>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  const groups = nodes[0]?.querySelectorAll?.("g.focus-tally") ?? [];
  for (const [index, group] of [...groups].entries()) {
    const record = records[index];
    if (record === undefined) break;
    const under = group.querySelector(".focus-tally__under");
    const thread = group.querySelector(".focus-tally__thread");
    const beads = group.querySelector(".focus-tally__beads");
    const hit = group.querySelector(".focus-tally__hit");
    if (under) setEffectAttributes(under, { d: record.underPath });
    if (hit) setEffectAttributes(hit, { d: record.hitPath });
    if (thread) setEffectAttributes(thread, { d: record.threadPath ?? "" });
    if (beads) {
      setEffectAttributes(beads, {
        d: record.beadsPath,
        "stroke-width": record.beadWidth,
      });
    }
  }
}

/**
 * Return the Symbol Tip descriptor for a Focus Tally record, or null for any
 * other record.
 *
 * It says how many attackers and which team struck, that it is a plain count of
 * hits and not the Focus Fire Concentration metric, and that only visible
 * blows count (in Agent POV only blows the POV team saw).
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  if (record?.kind !== "focus-tally") {
    return null;
  }
  const team = record.teamKey === "team-a" ? "Team A" : "Team B";
  return createSemanticDescriptor({
    kind: "focus-tally",
    id: `focus-tally:${record.event.eventId ?? record.event.targetPublicAgentId ?? ""}`,
    title: `Focus Tally · ${record.count} Attackers`,
    tone: "information",
    accent: "none",
    summary: `${record.count} champions of ${team} hit this champion on this tick, one bead each. It is a plain count of hits, not the Focus Fire Concentration metric. Only visible blows are counted.`,
    rows: [],
    sections: [],
    metadata: Object.freeze({ compact: true, full: true }),
    anchor: "element",
  });
}
