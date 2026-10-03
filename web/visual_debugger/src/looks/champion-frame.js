/**
 * @file The Champion Frame: every mark of selection and inspection, on the
 * arena and in the frame under it.
 *
 * On the arena (SVG, export-safe) it draws:
 * - Selection Moonlight: one flat Moon Silver disc at 6% around the selected
 *   living champion, R + 12 px, made by buildSelectionMarks as a separate node
 *   (`circle.selection-moonlight`) so `scene.js` can lay it on the floor under
 *   rings, stones and bodies (switch `selection_moonlight`);
 * - Silver Brackets: four L corners on a square of half-size R + clamp(0.52R,
 *   6, 12) px around the selected target, over an Ink Shadow keyline, at 45% on
 *   a fallen champion, one ring at R + 4 px when R is under 10 px (switch
 *   `target_selection_visuals`); live play (BattleClient and DevClient) uses
 *   the BattleClient vision's brackets instead (VISION_BRACKET: half-size
 *   1.48 R, arms 0.45 R, strokes scaled by the field's --mark-scale); the
 *   controlled halo, a Moon Silver ring at R + 8 px;
 * - Reach Rings: the inspected champion's sight, Basic and Ultimate reach from
 *   the host's own radii (`scene.ranges`), in the `debug-range` layer, each
 *   with a stroke hit ring (Ranges button, or switch `reach_rings` for the
 *   selected champion; in live play `scene.js` draws them only with both
 *   Ranges and `reach_rings` on);
 * - the Next Action Thread: a straight dashed line from the actor's rim + 3 px
 *   to 6 px short of the target's rim, and a dot (Basic) or diamond (Ultimate)
 *   at the target's 6 o'clock, R + 9 px below its centre; a self-cast draws only
 *   the glyph under the actor (switch `next_action_thread`).
 * It draws no Basic or Ultimate lane pills ("0/B", "1/U") on the arena in any
 * client. Runefield shows legality on the deck's Action slots and the Command
 * Ring with its Order Studs (`looks/hotbar.js`), and the read-only Replay
 * Viewer shows none. `scene.js` calls records, then build, and update for the
 * retained marks.
 *
 * Under the arena (DOM, Replay Viewer only, plan P10) it builds the Champion
 * Frame: a hint row ("Select An Agent To Inspect · Click A Body Or A Team
 * Rosters Chip") until a champion is selected, then its token as a portrait (drawn by
 * the Champions look), its label, life, Ultimate, every status by the status
 * rule, Recent Ticks (this tick only) and Next Target. main.js reads the
 * frame's facts with the adapter into a plain ChampionFrameInput, then calls
 * championFrameView and renderChampionFrame (switch `selected_agent_frame`).
 * This file never imports the adapter or reads a frame itself.
 *
 * Agent POV boundary:
 * - Arena marks read only the authorized arena scene (`scene.ranges`,
 *   `scene.pending_route`, `scene.agents`) and the Champions look's records. In Agent POV that scene holds only what the POV
 *   team may see, so a selected champion out of sight has no record and its
 *   marks lapse; they return when it is seen again (the selection itself is
 *   kept by the caller). The adapter gives a pending route only for the POV
 *   agent's own next action. Nothing here draws legality, so the selected
 *   pair's mask facts (`scene.selected_legality`) never reach the arena.
 * - The docked frame is a researcher panel (owner rule, 1 Oct, row 3): in the
 *   Replay Viewer's Agent POV the caller passes the selected champion's full
 *   record, shown under the "Full Record" chip. Only its Recent Ticks cell is
 *   limited to the audience's own events; it says "Not Visible" when the POV
 *   team did not see the champion. Nothing in the frame is drawn on or over
 *   the arena, and the arena marks never use the frame's inputs. The
 *   BattleClient and DevClient never mount it.
 */
import { canonicalAgentIdentity, tickCountLabel } from "../agent-identity.js";
import { classAbilityNameV1 } from "../class-documentation.js";
import { formatDisplayNumber } from "../display.js";
import { createLifeBar, lifeBarModel } from "../life-bar.js";
import { explainPendingRoute, explainRange } from "../explanations.js";
import { statusLineV1 } from "../semantic-vocabulary.js";
import { statusOrderIndex } from "../symbols.js";
import { createSemanticDescriptor, registerTooltipOwner } from "../tooltip.js";
import { classTokenFromId, statusTokenIdFromCatalogId } from "../vocabulary.js";
import * as champions from "./champions.js";
import {
  asArray,
  buildCircleNodes,
  circleRecords,
  displayIdentityAttributes,
  finiteNumber,
  isRecord,
  paintParts,
  screenPoint,
  setNodeAttributes,
  svgNode,
  updateCircleNodes,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").JsonRecord} JsonRecord
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {{
 *   selectionRoot: SVGElement,
 *   controlledHalo: SVGElement,
 *   selectedKeyline: SVGElement,
 *   selectedReticle: SVGElement,
 *   moonlight: SVGElement,
 * }} SelectionNodes
 *   One champion's retained selection marks. selectionRoot holds the halo, the
 *   bracket keyline and the brackets; moonlight is a separate floor node that
 *   `scene.js` places in a layer under the stones and bodies.
 */

/**
 * Arena sizes this look reserves: none. It draws its Champion Frame as page
 * chrome under the arena and no longer draws lane pills, so the renderer
 * reserves no legality dock (the empty record keeps the look contract).
 */
export const FOOTPRINT = Object.freeze({});

/**
 * Nodes per record: a reach ring with its hit ring and owner (3), the next-action
 * thread (line, glyph group of three and hit path: 5) or the selection marks
 * (root, halo, keyline, brackets and moonlight: 5).
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 5,
  animationsPerRecord: 0,
  persistentNodesPerRecord: 0,
});

/**
 * Switches for the arena marks: Target Selection Visuals (brackets), Selection
 * Moonlight, Next Action Thread and Reach Rings. The `selected_pair_legality`
 * part stays registered under Target Selection Visuals so the switch list and
 * `scene.js` keep their names, but it paints nothing now: this look draws no
 * lane pills. The docked frame's switch (`selected_agent_frame`) is an
 * interface switch with no paint part.
 */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "selection_reticle" }, "target_selection_visuals"],
  [{ surface: "durable", kind: "selected_pair_legality" }, "target_selection_visuals"],
  [{ surface: "durable", kind: "selection_moonlight" }, "selection_moonlight"],
  [{ surface: "durable", kind: "next_action_thread" }, "next_action_thread"],
  [{ surface: "durable", kind: "reach_ring" }, "reach_rings"],
]);

/** Selection Moonlight and the bracket square reach this far past the body (CSS px). */
export const MOONLIGHT_OFFSET_PX = 12;

/** The controlled halo sits this far outside the body (CSS px). */
export const HALO_OFFSET_PX = 8;

/** Next Action Thread gaps: from the actor's rim, and short of the target's rim (CSS px). */
export const THREAD_GAPS_PX = Object.freeze({ start: 3, end: 6 });

/** The next-action glyph's centre sits this far below the body's rim (CSS px). */
export const GLYPH_DROP_PX = 9;

/**
 * Return how far the bracket square sits outside a body of radius R (CSS px).
 *
 * The detail ladder: 0.52R, kept between 6 and 12 px (12 px at R 23).
 *
 * @param {number} radius Body radius in CSS pixels.
 * @returns {number}
 */
export function bracketOffset(radius) {
  return Math.min(12, Math.max(6, 0.52 * finiteNumber(radius)));
}

/**
 * The BattleClient vision's corner brackets, as shares of the body radius R:
 * the square's half-size is R + 12 vision px at R 25 (so 1.48 R) and each arm
 * is 0.45 R (vision_template.html, renderArena). Live play scales them with
 * the body, as the vision's drawing scales with its field.
 */
export const VISION_BRACKET = Object.freeze({ halfShare: 1 + 12 / 25, armShare: 0.45 });

/**
 * Return the Silver Brackets path around a selected body.
 *
 * centerX/centerY and radius are the body's screen centre and radius (CSS px).
 * shape "ladder" (the default; the Replay Viewer): from R 10 px up, four L
 * corners on a square of half-size R + bracketOffset(R), each arm 0.38 of that
 * half-size, open in the middle of each side; under R 10 px the corners become
 * one ring at R + 4 px. shape "vision" (live play: the BattleClient and the
 * DevClient): four L corners on a square of half-size 1.48 R with arms 0.45 R
 * (VISION_BRACKET), at any radius. Pure; returns an SVG path.
 *
 * @param {number} centerX
 * @param {number} centerY
 * @param {number} radius
 * @param {"ladder" | "vision"} [shape] Which bracket rule; default "ladder".
 * @returns {string}
 */
export function bracketPath(centerX, centerY, radius, shape = "ladder") {
  const vision = shape === "vision";
  if (!vision && radius < 10) {
    const r = radius + 4;
    return `M ${centerX - r} ${centerY} A ${r} ${r} 0 1 0 ${centerX + r} ${centerY} A ${r} ${r} 0 1 0 ${centerX - r} ${centerY} Z`;
  }
  const half = vision
    ? radius * VISION_BRACKET.halfShare
    : radius + bracketOffset(radius);
  const arm = vision ? radius * VISION_BRACKET.armShare : half * 0.38;
  const left = centerX - half;
  const right = centerX + half;
  const top = centerY - half;
  const bottom = centerY + half;
  return [
    `M ${left + arm} ${top} H ${left} V ${top + arm}`,
    `M ${right - arm} ${top} H ${right} V ${top + arm}`,
    `M ${left} ${bottom - arm} V ${bottom} H ${left + arm}`,
    `M ${right} ${bottom - arm} V ${bottom} H ${right - arm}`,
  ].join(" ");
}

/**
 * Return this look's drawing records for one part of the arena.
 *
 * input.part picks what to build:
 * - "reach-rings": input.scene (its ranges, agents and selection) and
 *   input.transform give {kind: "reach-ring", ...} circle records in CSS
 *   pixels, tinted by the owner's class. With input.showRanges not false (the
 *   Ranges button, default on) every range row is drawn. With showRanges false
 *   and input.showReachRings true (switch `reach_rings`, default false here so
 *   a caller that does not pass it keeps today's behaviour), only the rows of
 *   the selected champion (scene.selection.selected_presentation_key) are
 *   drawn; otherwise [].
 * - "next-action-route": input.scene.pending_route and input.transform give one
 *   {kind: "next-action-route", path, glyph, lane, legal, ids and keys,
 *   routeKind, tooltipInput} record, or [] with no route or when
 *   input.showNextActionThread is false (switch `next_action_thread`, default
 *   on). path is the straight thread ("M x y L x y") or null for a self-cast
 *   or bodies too close for a line; glyph is {x, y, shape: "dot" | "diamond"}
 *   in CSS px; routeKind is "thread" or "self".
 * - "selection-marks": input.champions (the Champions look's records),
 *   input.showSelectionReticle (default false), input.showSelectionMoonlight
 *   (default true) and input.bracketShape ("vision" in live play, else
 *   "ladder", the default; see bracketPath) give one {kind: "selection-marks",
 *   identityKey, agent, center, radius, controlled, selected, alive,
 *   showReticle, showMoonlight, bracketShape} record per champion.
 *   showMoonlight is true only for the selected living champion with the
 *   switch on.
 * - "lane-pills" is retired and gives [], whatever the input: the arena shows
 *   no "0/B" or "1/U" pills in any client (the deck's Action slots and the
 *   Command Ring carry legality; the Replay Viewer shows none).
 * Anything else gives []. Pure data: no DOM.
 *
 * Agent POV: input.scene and input.champions must come from the POV arena
 * scene; a champion the team cannot see has no record, so its marks lapse.
 *
 * @param {Record<string, any>} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  if (input.part === "reach-rings") {
    if (!input.transform || !isRecord(input.scene)) {
      return Object.freeze([]);
    }
    const allRanges = input.showRanges !== false;
    if (!allRanges && input.showReachRings !== true) {
      return Object.freeze([]);
    }
    const selectedKey = isRecord(input.scene.selection)
      ? input.scene.selection.selected_presentation_key
      : null;
    const rangeRows = asArray(input.scene.ranges).filter(
      (row) =>
        allRanges ||
        (typeof selectedKey === "string" &&
          isRecord(row) &&
          row.presentation_key === selectedKey),
    );
    const agents = asArray(input.scene.agents).filter(
      (agent) => isRecord(agent) && typeof agent.presentation_key === "string",
    );
    const classByIdentity = new Map(
      agents.map((agent) => [
        String(agent.presentation_key),
        classTokenFromId(agent.class_id).cssKey,
      ]),
    );
    const agentByPresentationKey = new Map(
      agents.map((agent) => [String(agent.presentation_key), agent]),
    );
    return circleRecords(rangeRows, input.transform, {
      kind: "reach-ring",
      tokenAttribute: "kind",
      classByIdentity,
      tooltipInput: (row) =>
        Object.freeze({
          range: row,
          owner:
            (typeof row.presentation_key === "string"
              ? agentByPresentationKey.get(row.presentation_key)
              : null) ?? null,
        }),
    });
  }
  if (input.part === "next-action-route") {
    return input.showNextActionThread === false
      ? Object.freeze([])
      : nextActionRouteRecords(input.scene, input.transform);
  }
  if (input.part === "selection-marks") {
    return Object.freeze(
      asArray(input.champions).map((champion) =>
        Object.freeze({
          kind: "selection-marks",
          identityKey: champion.identityKey,
          agent: champion.agent,
          center: champion.center,
          radius: champion.radius,
          controlled: champion.controlled,
          selected: champion.selected,
          alive: champion.agent?.alive !== false,
          showReticle: input.showSelectionReticle === true,
          bracketShape: input.bracketShape === "vision" ? "vision" : "ladder",
          showMoonlight:
            input.showSelectionMoonlight !== false &&
            champion.selected === true &&
            champion.agent?.alive !== false,
        }),
      ),
    );
  }
  return Object.freeze([]);
}

/**
 * Return the straight thread between two bodies, or null when they are too close.
 *
 * source and target are screen centres; startGap and endGap are distances from
 * each centre (CSS px) where the line starts and stops (the actor's radius + 3
 * and the target's radius + 6). Return {x1, y1, x2, y2}, or null when the
 * centres are not finite or not farther apart than the two gaps. Pure.
 *
 * @param {{x: number, y: number}} source
 * @param {{x: number, y: number}} target
 * @param {number} startGap
 * @param {number} endGap
 * @returns {{x1: number, y1: number, x2: number, y2: number} | null}
 */
export function threadSegment(source, target, startGap, endGap) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.hypot(dx, dy);
  if (!Number.isFinite(distance) || distance <= startGap + endGap) {
    return null;
  }
  const ux = dx / distance;
  const uy = dy / distance;
  return {
    x1: source.x + ux * startGap,
    y1: source.y + uy * startGap,
    x2: target.x - ux * endGap,
    y2: target.y - uy * endGap,
  };
}

/**
 * Return the Next Action Thread record for a scene, or [] without a route.
 *
 * scene.pending_route owns the disclosed source and target anchors, radii,
 * lane and legality; transform projects them. The thread is a straight line
 * from the actor's rim + 3 px to 6 px short of the target's rim (null for a
 * self-cast, or when the rims nearly touch). The glyph sits R + 9 px below the
 * target's centre (the actor's own for a self-cast): a dot for lane 0 (Basic),
 * a diamond for lane 1 (Ultimate). The tooltip input names the route and the
 * matching scene agents (same presentation key and public ID). No target
 * position, legality or accepted action is inferred. In Agent POV the adapter
 * gives a pending route only for the POV agent's own next action.
 *
 * @param {unknown} scene
 * @param {LookTransform | null | undefined} transform
 * @returns {ReadonlyArray<LookRecord>}
 */
function nextActionRouteRecords(scene, transform) {
  const route =
    isRecord(scene) && isRecord(scene.pending_route) ? scene.pending_route : null;
  if (!route || !transform || !isRecord(scene)) {
    return Object.freeze([]);
  }
  const source = screenPoint(route.source_anchor, transform);
  const target = screenPoint(route.target_anchor, transform);
  const sourceRadius = transform.worldLengthToScreen(finiteNumber(route.source_radius));
  const targetRadius = transform.worldLengthToScreen(finiteNumber(route.target_radius));
  const selfCast =
    route.source_public_agent_id === route.target_public_agent_id &&
    route.source_presentation_key === route.target_presentation_key;
  const segment = selfCast
    ? null
    : threadSegment(
        source,
        target,
        sourceRadius + THREAD_GAPS_PX.start,
        targetRadius + THREAD_GAPS_PX.end,
      );
  const host = selfCast ? source : target;
  const hostRadius = selfCast ? sourceRadius : targetRadius;
  const sourceAgent = asArray(scene.agents).find(
    (agent) =>
      isRecord(agent) &&
      agent.presentation_key === route.source_presentation_key &&
      agent.public_agent_id === route.source_public_agent_id,
  );
  const recipientAgent = asArray(scene.agents).find(
    (agent) =>
      isRecord(agent) &&
      agent.presentation_key === route.target_presentation_key &&
      agent.public_agent_id === route.target_public_agent_id,
  );
  return Object.freeze([
    Object.freeze({
      kind: "next-action-route",
      path:
        segment === null
          ? null
          : `M ${segment.x1} ${segment.y1} L ${segment.x2} ${segment.y2}`,
      glyph: Object.freeze({
        x: host.x,
        y: host.y + hostRadius + GLYPH_DROP_PX,
        shape: route.lane === 1 ? "diamond" : "dot",
      }),
      lane: String(route.lane ?? 0),
      legal: String(Boolean(route.legal)),
      sourceAgentId: String(route.source_public_agent_id ?? ""),
      targetAgentId: String(route.target_public_agent_id ?? ""),
      sourceSlot: Number.isInteger(route.source_global_slot)
        ? String(route.source_global_slot)
        : null,
      targetSlot: Number.isInteger(route.target_global_slot)
        ? String(route.target_global_slot)
        : null,
      sourcePresentationKey:
        typeof route.source_presentation_key === "string"
          ? route.source_presentation_key
          : null,
      targetPresentationKey:
        typeof route.target_presentation_key === "string"
          ? route.target_presentation_key
          : null,
      routeKind: selfCast ? "self" : "thread",
      tooltipInput: Object.freeze({ route, sourceAgent, recipientAgent }),
    }),
  ]);
}

/**
 * Build nodes for this look's records.
 *
 * records come from records(); doc creates nodes. Reach rings become ring and
 * hit-ring owner groups with tooltips (left out when the tooltip is null); each
 * owner carries a "set-down" beat hold keyed to its champion's presentation
 * key (data-beat-hold, data-beat-key, data-beat-side "after", a 1 ms cut), so
 * on the tick a champion returns its rings appear only when it is set down on
 * its pad; on any other tick the choreographer finds no such beat and the
 * rings show at once. A
 * next-action route becomes [thread line (none for a self-cast), glyph group,
 * focusable hit path] with its tooltip. Selection marks become one
 * SelectionNodes bundle's root (see buildSelectionMarks; its separate moonlight
 * node is not returned here). Any other record kind, including an old
 * "lane-pills" record, builds nothing. Return the new top-level nodes in record
 * order. Registers tooltip owners.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @returns {SVGElement[]}
 */
export function build(records, doc) {
  /** @type {SVGElement[]} */
  const nodes = [];
  for (const record of records) {
    if (record.kind === "reach-ring") {
      const key = record.identity?.presentation_key;
      for (const node of buildCircleNodes(doc, [record], {
        className: "range-ring",
        tokenAttribute: "kind",
        tooltip,
        withStrokeHitRegion: true,
      })) {
        // A returning champion's rings wait for its set-down on the pad, as
        // its token, ward ring and Command Ring do; the choreographer applies
        // this only on a tick when that champion returns.
        if (typeof key === "string" && key !== "") {
          node.setAttribute("data-beat-hold", "set-down");
          node.setAttribute("data-beat-key", key);
          node.setAttribute("data-beat-side", "after");
          node.setAttribute("data-beat-fade", "1");
        }
        nodes.push(node);
      }
    } else if (record.kind === "next-action-route") {
      nodes.push(...buildNextActionRoute(doc, record));
    } else if (record.kind === "selection-marks") {
      nodes.push(buildSelectionMarks([record], doc)[0].selectionRoot);
    }
  }
  return nodes;
}

/**
 * Return the diamond path of the given full width centred on (x, y) (CSS px).
 *
 * @param {number} x
 * @param {number} y
 * @param {number} width
 * @returns {string}
 */
function diamondPath(x, y, width) {
  const half = width / 2;
  return `M ${x} ${y - half} L ${x + half} ${y} L ${x} ${y + half} L ${x - half} ${y} Z`;
}

/**
 * Build the Next Action Thread nodes: the line (when there is one), the glyph
 * group and the focusable hit path.
 *
 * The line is `path.pending-route` with the route's data attributes (lane,
 * legal, IDs, slots, keys, route kind). The glyph is `g.next-action-glyph`
 * holding an Ink Shadow under-copy and the Moon Silver mark: a 6 px dot or a
 * 9 px diamond. The hit path follows the line, or rings the glyph for a
 * self-cast, and carries the route tooltip. Return [line?, glyph, hit].
 * Registers one tooltip owner. Export-safe SVG only.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement[]}
 */
function buildNextActionRoute(doc, record) {
  /** @type {SVGElement[]} */
  const nodes = [];
  const data = {
    "data-lane": record.lane,
    "data-legal": record.legal,
    "data-source-agent-id": record.sourceAgentId,
    "data-target-agent-id": record.targetAgentId,
    "data-source-slot": record.sourceSlot,
    "data-target-slot": record.targetSlot,
    "data-source-presentation-key": record.sourcePresentationKey,
    "data-target-presentation-key": record.targetPresentationKey,
    "data-route-kind": record.routeKind,
  };
  if (record.path !== null) {
    nodes.push(
      svgNode(doc, "path", { class: "pending-route", d: record.path, ...data }),
    );
  }
  const { x, y, shape } = record.glyph;
  const glyph = svgNode(doc, "g", {
    class: "next-action-glyph",
    "data-shape": shape,
    ...data,
    "aria-hidden": "true",
  });
  glyph.append(
    shape === "diamond"
      ? svgNode(doc, "path", {
          class: "next-action-glyph__under",
          d: diamondPath(x, y, 9),
        })
      : svgNode(doc, "circle", {
          class: "next-action-glyph__under",
          cx: x,
          cy: y,
          r: 4.5,
        }),
    shape === "diamond"
      ? svgNode(doc, "path", {
          class: "next-action-glyph__mark",
          d: diamondPath(x, y, 9),
        })
      : svgNode(doc, "circle", {
          class: "next-action-glyph__mark",
          cx: x,
          cy: y,
          r: 3,
        }),
  );
  nodes.push(glyph);
  const hitPath = svgNode(doc, "path", {
    class: "pending-route-hit",
    d:
      record.path ??
      `M ${x - 8} ${y} A 8 8 0 1 0 ${x + 8} ${y} A 8 8 0 1 0 ${x - 8} ${y} Z`,
    role: "img",
    tabindex: "0",
  });
  const routeDescriptor = /** @type {NonNullable<ReturnType<typeof tooltip>>} */ (
    tooltip(record)
  );
  hitPath.setAttribute("aria-label", routeDescriptor.title);
  registerTooltipOwner(hitPath, routeDescriptor);
  nodes.push(hitPath);
  return nodes;
}

/**
 * Create the retained selection marks for champions.
 *
 * records are "selection-marks" records; doc creates nodes. Each bundle has a
 * `g.agent-selection` root (aria-hidden, with the champion's identity
 * attributes) holding the controlled halo circle, the Ink Shadow bracket
 * keyline (`path.selected-reticle-keyline`) and the Silver Brackets
 * (`path.selected-reticle`), plus a separate `circle.selection-moonlight` with
 * the same identity attributes, not yet in any layer: `scene.js` puts it in a
 * floor layer. Nodes have no geometry until updateSelectionMarks runs. Return
 * the bundles in record order. No tooltip, storage or timer.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @returns {SelectionNodes[]}
 */
export function buildSelectionMarks(records, doc) {
  return records.map((record) => {
    const agent = record.agent;
    const selectionRoot = svgNode(doc, "g", {
      class: "agent-selection",
      ...displayIdentityAttributes(agent),
      "data-zone": "selection",
      "aria-hidden": "true",
    });
    const controlledHalo = svgNode(doc, "circle", {
      class: "controlled-halo",
    });
    const selectedKeyline = svgNode(doc, "path", {
      class: "selected-reticle-keyline",
    });
    const selectedReticle = svgNode(doc, "path", {
      class: "selected-reticle",
    });
    selectionRoot.append(controlledHalo, selectedKeyline, selectedReticle);
    const moonlight = svgNode(doc, "circle", {
      class: "selection-moonlight",
      ...displayIdentityAttributes(agent),
      "data-zone": "selection-moonlight",
      "aria-hidden": "true",
    });
    return {
      selectionRoot,
      controlledHalo,
      selectedKeyline,
      selectedReticle,
      moonlight,
    };
  });
}

/**
 * Refresh retained selection marks from their new records.
 *
 * nodes and records are parallel arrays. The halo is a circle 8 px outside the
 * body, shown only for the controlled champion. The brackets and their keyline
 * (bracketPath, in the record's bracketShape) show only for the selected
 * target with Target Selection Visuals on, and carry data-fallen="true" on a
 * fallen champion (drawn at 45% by CSS) and data-shape="vision" in live play
 * (the vision's stroke weights, scaled by the field's --mark-scale).
 * The moonlight is a disc R + 12 px, shown only when the record's
 * showMoonlight is true. A bundle without keyline or moonlight (an older
 * caller) is updated where it has nodes. Return undefined. DOM changes only.
 *
 * @param {ReadonlyArray<Partial<SelectionNodes> & {controlledHalo: SVGElement, selectedReticle: SVGElement}>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function updateSelectionMarks(nodes, records) {
  for (const [index, bundle] of nodes.entries()) {
    const record = records[index];
    if (record === undefined) {
      break;
    }
    const { center, radius } = record;
    setNodeAttributes(bundle.controlledHalo, {
      cx: center.x,
      cy: center.y,
      r: radius + HALO_OFFSET_PX,
      hidden: record.controlled ? null : "",
    });
    const bracket = {
      d: bracketPath(center.x, center.y, radius, record.bracketShape),
      hidden: record.selected && record.showReticle ? null : "",
      "data-fallen": record.alive === false ? "true" : null,
      "data-shape": record.bracketShape === "vision" ? "vision" : null,
    };
    setNodeAttributes(bundle.selectedReticle, bracket);
    if (bundle.selectedKeyline) {
      setNodeAttributes(bundle.selectedKeyline, bracket);
    }
    if (bundle.moonlight) {
      setNodeAttributes(bundle.moonlight, {
        cx: center.x,
        cy: center.y,
        r: radius + MOONLIGHT_OFFSET_PX,
        hidden: record.showMoonlight === true ? null : "",
      });
    }
  }
}

/**
 * Move this look's nodes to new records (reprojection).
 *
 * nodes and records come in matching order and kinds. Reach rings get their
 * new centre and radius; selection marks take SelectionNodes bundles (see
 * updateSelectionMarks). A next-action route is rebuilt by
 * `scene.js` on each render, so update leaves them alone. Return undefined.
 *
 * @param {ReadonlyArray<any>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  const first = records[0];
  if (first?.kind === "reach-ring") {
    updateCircleNodes(nodes, records);
  } else if (first?.kind === "selection-marks") {
    updateSelectionMarks(nodes, records);
  }
}

/**
 * Return the tooltip for one of this look's records.
 *
 * - reach-ring: the range explanation for the ring and its owner;
 * - next-action-route: the pending route explanation with its source and
 *   recipient agents;
 * Any other record returns null.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  if (record.kind === "reach-ring") {
    return explainRange(record.tooltipInput.range, record.tooltipInput.owner);
  }
  if (record.kind === "next-action-route") {
    const input = record.tooltipInput;
    return explainPendingRoute(input.route, {
      sourceAgent: input.sourceAgent,
      recipientAgent: input.recipientAgent,
    });
  }
  return null;
}

/** The hint row shown under the arena while no champion is selected. */
export const CHAMPION_FRAME_HINT =
  "Select An Agent To Inspect · Click A Body Or A Team Rosters Chip";

/** Next Target on the last frame of a replay, where no next tick exists. */
export const FINAL_FRAME_NEXT = "No Next Tick (Final Frame)";

/** The portrait's body radius in CSS pixels (a 46 x 46 px token). */
export const PORTRAIT_RADIUS_PX = 23;

/** The frame's tooltip rows show on compact and full cards. */
const FRAME_TIP_METADATA = Object.freeze({ compact: true, full: true });

/**
 * @typedef {Readonly<{
 *   state: "empty",
 *   hint: string,
 * }>} EmptyChampionFrameView
 *   Nothing (or no known champion) is selected: the frame shows only the hint.
 * @typedef {Readonly<{
 *   state: "seen" | "not-visible" | "no-tick",
 *   ability: "basic" | "ultimate" | null,
 *   damage: number | null,
 *   heal: number | null,
 *   rest: number | null,
 *   net: boolean,
 *   povOnly: boolean,
 * }>} RecentTickCell
 *   This tick's cell, from the audience's own incoming events. "no-tick" is
 *   frame zero (no incoming tick); "not-visible" is an Agent POV tick on which
 *   the POV team did not see the champion (no numbers). damage and heal are
 *   total effective amounts (Oracle View) or, when net is true (Agent POV,
 *   whose events carry only the net change), the net loss or gain; null means
 *   none recorded. rest is health regenerated by resting. povOnly is true when
 *   the cell is limited to what the POV team saw.
 * @typedef {Readonly<{
 *   state: "selected",
 *   chip: string | null,
 *   agent: Readonly<Record<string, any>>,
 *   title: string,
 *   teamLabel: string | null,
 *   teamKey: "team-a" | "team-b" | null,
 *   classKey: string,
 *   alive: boolean,
 *   lifeText: string,
 *   life: import("../life-bar.js").LifeBarModel | null,
 *   ultimateText: string,
 *   ultimateReady: boolean,
 *   statuses: ReadonlyArray<Readonly<{tokenId: string, line: string}>>,
 *   recent: RecentTickCell,
 *   nextText: string | null,
 *   finalFrame: boolean,
 * }>} SelectedChampionFrameView
 *   One selected champion's facts from the record the caller passed. chip is
 *   the Full Record label in the Replay Viewer's Agent POV, else null.
 *   lifeText is the exact life ("43.15 / 80", or "Fallen") for the life tip;
 *   life is the shared Life bar's model (`life-bar.js`), or null for a fallen
 *   champion or a record with no life numbers. nextText is null outside a
 *   replay.
 * @typedef {EmptyChampionFrameView | SelectedChampionFrameView} ChampionFrameView
 * @typedef {(
 *   action: Readonly<Record<string, any>>,
 *   classId: number | null | undefined,
 *   targetMap: ReadonlyArray<string | null> | null,
 *   actorPublicId: string | null,
 * ) => Readonly<{
 *   ability: string | null,
 *   noCombat: boolean,
 *   targetPublicId: string | null,
 *   targetCode: number | null,
 *   itself: boolean,
 * }>} DescribeRecordedAction
 *   `describeRecordedAction` from `panels.js`, the one owner of action words.
 */

/**
 * @typedef {Readonly<{
 *   selectedPublicAgentId: string | null | undefined,
 *   agents: ReadonlyArray<unknown>,
 *   chip: string | null,
 *   audience: "researcher" | "agent_pov",
 *   incomingRows: ReadonlyArray<Readonly<{payload: Readonly<Record<string, any>>}>>,
 *   hasIncomingTick: boolean,
 *   povSceneAgentIds: ReadonlyArray<string> | null,
 *   replay: boolean,
 *   upcomingActionRows: ReadonlyArray<unknown> | null,
 *   describeAction: DescribeRecordedAction,
 * }>} ChampionFrameInput
 *   What the caller (main.js, Replay Viewer only) reads from the installed
 *   frame for the docked Champion Frame. This look never reads a frame itself:
 *   - selectedPublicAgentId: the selected champion's public ID ("agent-slot-5"),
 *     or null for no selection.
 *   - agents: the researcher record's agents
 *     (authorizedPresentationResearcherSceneView(frame).agents): the full record,
 *     also in Agent POV, because the docked frame is a researcher panel.
 *   - chip: the Full Record words from `fullRecordLabel` in `panels.js` (the
 *     one owner of those words), or null outside the Replay Viewer's Agent POV.
 *   - audience: the frame's audience.
 *   - incomingRows: authorizedPresentationIncomingRows(frame), the audience's
 *     own incoming events (Oracle latest_events, Agent POV visual_events).
 *   - hasIncomingTick: false on frame zero (no incoming event branch).
 *   - povSceneAgentIds: in Agent POV the public IDs in the POV arena scene
 *     (authorizedPresentationSceneView(frame).agents); null in Oracle View.
 *   - replay: true for a replay frame.
 *   - upcomingActionRows: the recorded upcoming transition's action_rows
 *     (researcher branch first), or null when there is none (the final frame).
 *   - describeAction: `describeRecordedAction` from `panels.js`.
 */

/**
 * Return the Champion Frame's facts for one frame and selection. Pure.
 *
 * input is a ChampionFrameInput. No selection, or an ID missing from
 * input.agents, gives the empty view (the hint row). Otherwise the facts come
 * from that agent's record: label "Mage · ID 5", team, life "33.15 / 80" (or
 * "Fallen"), Ultimate ("Ultimate Ready" or "Burst · Ready In 28 Ticks"), every
 * status in Champion Card order (stuns, then slows, then the rest) by the
 * status rule (statusLineV1: the card row's words, the source named by class
 * only), the Recent Ticks cell (recentTickCell: only the audience's own events) and
 * Next Target (nextActionText). The Replay Viewer alone mounts this frame (plan
 * P10); the arena marks of this look never use these inputs.
 *
 * @param {ChampionFrameInput} input
 * @returns {ChampionFrameView}
 * @throws {TypeError} When input.describeAction is not a function.
 */
export function championFrameView(input) {
  if (typeof input?.describeAction !== "function") {
    throw new TypeError("championFrameView needs the describeAction function.");
  }
  const empty = Object.freeze({ state: "empty", hint: CHAMPION_FRAME_HINT });
  const selectedId = input.selectedPublicAgentId;
  if (typeof selectedId !== "string") {
    return empty;
  }
  const agents = asArray(input.agents).filter(isRecord);
  const agent = agents.find((each) => each.public_agent_id === selectedId);
  if (agent === undefined) {
    return empty;
  }
  const identity = canonicalAgentIdentity(agent);
  const alive = agent.alive !== false;
  const health = finiteNumber(agent.current_health, 0);
  const maximum = finiteNumber(agent.max_health ?? agent.maximum_health, 0);
  const ultimateName = classAbilityNameV1(agent.class_id, "ultimate") ?? "Ultimate";
  const cooldown = agent.ultimate_cooldown;
  const ultimateReady = Number.isInteger(cooldown) && cooldown <= 0;
  const next = nextActionText(input, agent, agents);
  return Object.freeze({
    state: "selected",
    chip: typeof input.chip === "string" ? input.chip : null,
    agent,
    title: identity.title,
    teamLabel: identity.teamLabel ?? null,
    teamKey: agent.team_id === 1 ? "team-a" : agent.team_id === 2 ? "team-b" : null,
    classKey: classTokenFromId(agent.class_id).cssKey,
    alive,
    lifeText: alive
      ? `${formatDisplayNumber(health)} / ${formatDisplayNumber(maximum)}`
      : "Fallen",
    life: alive && maximum > 0 ? lifeBarModel(health, maximum) : null,
    ultimateText: ultimateReady
      ? "Ultimate Ready"
      : Number.isInteger(cooldown)
        ? `${ultimateName} · Ready In ${tickCountLabel(cooldown)}`
        : ultimateName,
    ultimateReady,
    statuses: Object.freeze(
      asArray(agent.statuses)
        .filter(isRecord)
        .map((status) =>
          Object.freeze({
            tokenId: statusTokenIdFromCatalogId(status.status_id ?? status.token_id),
            line: frameStatusLine(status, agent),
          }),
        )
        .sort((a, b) => statusOrderIndex(a.tokenId) - statusOrderIndex(b.tokenId)),
    ),
    recent: recentTickCell(input, selectedId),
    nextText: next.text,
    finalFrame: next.final,
  });
}

/**
 * Write one status line by the status rule (statusLineV1), in the Champion
 * Card row's words: "Charge (Slow) · 5 ticks left · It moves at ×0.5 speed.
 * From a Warrior." The source is named by class only (owner ruling, 2
 * October 2026), so the line never names an Agent ID. The status's catalog ID
 * (status_id) is turned into its display token first
 * (statusTokenIdFromCatalogId). Pure.
 *
 * @param {Readonly<Record<string, any>>} status One status of agent.
 * @param {Readonly<Record<string, any>>} agent The champion with the status
 *   (its class gives "From its own aura.").
 * @returns {string}
 */
function frameStatusLine(status, agent) {
  return statusLineV1({
    tokenId: statusTokenIdFromCatalogId(status.status_id ?? status.token_id),
    magnitude: status.magnitude,
    remainingTicks: status.remaining_duration ?? status.duration,
    recipientClassId: agent.class_id,
  });
}

/**
 * Return this tick's Recent Ticks cell for one champion. Pure.
 *
 * Reads only input.incomingRows (the audience's own events). Without an
 * incoming tick (frame zero) the cell is "no-tick". In Agent POV a champion
 * missing from input.povSceneAgentIds gives "not-visible" with no numbers.
 * Otherwise: ability_activated with this champion as source sets ability (an
 * Ultimate wins over a Basic); recipient_health_resolution for it sets damage
 * and heal (Oracle View totals, or the Agent POV net change with net true);
 * health_regenerated for it sets rest. Champions are joined by public ID.
 *
 * @param {ChampionFrameInput} input
 * @param {string} publicAgentId
 * @returns {RecentTickCell}
 */
function recentTickCell(input, publicAgentId) {
  const pov = input.audience === "agent_pov";
  const cell = {
    state: /** @type {RecentTickCell["state"]} */ ("seen"),
    ability: /** @type {RecentTickCell["ability"]} */ (null),
    damage: /** @type {number | null} */ (null),
    heal: /** @type {number | null} */ (null),
    rest: /** @type {number | null} */ (null),
    net: false,
    povOnly: pov,
  };
  if (input.hasIncomingTick !== true) {
    return Object.freeze({ ...cell, state: "no-tick" });
  }
  if (pov && !asArray(input.povSceneAgentIds).includes(publicAgentId)) {
    return Object.freeze({ ...cell, state: "not-visible" });
  }
  for (const row of asArray(input.incomingRows)) {
    const event = isRecord(row) && isRecord(row.payload) ? row.payload : {};
    const kind = event.event_kind;
    if (
      kind === "ability_activated" &&
      anchorId(event.source_anchor) === publicAgentId
    ) {
      cell.ability =
        event.ability_component === "ultimate" || cell.ability === "ultimate"
          ? "ultimate"
          : "basic";
    } else if (
      kind === "recipient_health_resolution" &&
      anchorId(event.recipient_anchor) === publicAgentId
    ) {
      if (
        Number.isFinite(event.total_effective_damage) ||
        Number.isFinite(event.total_effective_healing)
      ) {
        cell.damage = positiveOrNull(event.total_effective_damage);
        cell.heal = positiveOrNull(event.total_effective_healing);
      } else if (Number.isFinite(event.realized_net_health_change)) {
        cell.net = true;
        cell.damage = positiveOrNull(-event.realized_net_health_change);
        cell.heal = positiveOrNull(event.realized_net_health_change);
      }
    } else if (
      kind === "health_regenerated" &&
      anchorId(event.agent_anchor) === publicAgentId
    ) {
      cell.rest = positiveOrNull(event.actual_health_regenerated);
    }
  }
  return Object.freeze(cell);
}

/**
 * Return anchor.public_agent_id when anchor is a record, else null.
 *
 * @param {unknown} anchor
 * @returns {unknown}
 */
function anchorId(anchor) {
  return isRecord(anchor) ? anchor.public_agent_id : null;
}

/**
 * Return value when it is a finite number above zero, else null.
 *
 * @param {unknown} value
 * @returns {number | null}
 */
function positiveOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : null;
}

/**
 * Return the Next Target words for one champion. Pure.
 *
 * Outside a replay (input.replay false): {text: null, final: false}. In a
 * replay without input.upcomingActionRows: the final frame, FINAL_FRAME_NEXT.
 * Otherwise the champion's row (by actor_public_agent_id) is described by
 * input.describeAction: "Next · No Action" without a row or without combat,
 * "Next · ◆ Burst · Self" for a self-cast, "Next · ● Arcane Blast → Hunter ·
 * ID 7" with a target (its label from agents, its public ID when it is not
 * there, or "Target N" when the target is not mapped), and "Next · ◆ Burst"
 * for an Ultimate without a target.
 *
 * @param {ChampionFrameInput} input
 * @param {Readonly<Record<string, any>>} agent
 * @param {ReadonlyArray<Readonly<Record<string, any>>>} agents
 * @returns {{text: string | null, final: boolean}}
 */
function nextActionText(input, agent, agents) {
  if (input.replay !== true) {
    return { text: null, final: false };
  }
  if (!Array.isArray(input.upcomingActionRows)) {
    return { text: FINAL_FRAME_NEXT, final: true };
  }
  const row = input.upcomingActionRows.find(
    (/** @type {unknown} */ each) =>
      isRecord(each) && each.actor_public_agent_id === agent.public_agent_id,
  );
  if (!isRecord(row) || !isRecord(row.accepted_action)) {
    return { text: "Next · No Action", final: false };
  }
  const described = input.describeAction(
    row.accepted_action,
    agent.class_id,
    Array.isArray(row.target_action_recipient_public_agent_id_by_id)
      ? row.target_action_recipient_public_agent_id_by_id
      : null,
    agent.public_agent_id,
  );
  if (described.noCombat || described.ability === null) {
    return { text: "Next · No Action", final: false };
  }
  const glyph = row.accepted_action.use_ultimate_action === 1 ? "◆" : "●";
  const head = `Next · ${glyph} ${described.ability}`;
  if (described.itself) {
    return { text: `${head} · Self`, final: false };
  }
  if (described.targetPublicId !== null) {
    const target = agents.find(
      (each) => each.public_agent_id === described.targetPublicId,
    );
    return {
      text: `${head} → ${target === undefined ? described.targetPublicId : canonicalAgentIdentity(target).title}`,
      final: false,
    };
  }
  if (described.targetCode !== null) {
    return { text: `${head} → Target ${described.targetCode}`, final: false };
  }
  return { text: head, final: false };
}

/**
 * Create one HTML element with a class and optional text.
 *
 * @param {Document} doc
 * @param {string} tagName
 * @param {string} className
 * @param {string | null} [text]
 * @returns {HTMLElement}
 */
function htmlNode(doc, tagName, className, text = null) {
  const element = doc.createElement(tagName);
  element.className = className;
  if (text !== null) {
    element.textContent = text;
  }
  return element;
}

/**
 * Draw the champion's own token as a 46 x 46 px portrait.
 *
 * agent is the selected champion's record. The Champions look draws it
 * (records, build, update) on a small standalone SVG with the champion's
 * centre at (23, 23) and its body radius at 23 px, so the portrait matches the
 * arena token, including the Ashen Token of a fallen champion. If that look
 * cannot draw it, a plain token (team ring, class colour) is drawn instead. The
 * portrait is aria-hidden and not focusable. Return the svg element.
 *
 * @param {Document} doc
 * @param {Readonly<Record<string, any>>} agent
 * @returns {SVGSVGElement}
 */
export function buildPortrait(doc, agent) {
  const size = PORTRAIT_RADIUS_PX * 2;
  const svg = /** @type {SVGSVGElement} */ (
    svgNode(doc, "svg", {
      class: "champion-frame__portrait-token",
      viewBox: `0 0 ${size} ${size}`,
      width: size,
      height: size,
      "aria-hidden": "true",
      focusable: "false",
    })
  );
  const centre = Array.isArray(agent.position) ? agent.position : [0, 0];
  const scale = PORTRAIT_RADIUS_PX / Math.max(finiteNumber(agent.radius, 0.5), 0.01);
  const bounds = Object.freeze({
    left: 0,
    top: 0,
    right: size,
    bottom: size,
    width: size,
    height: size,
  });
  /** @type {LookTransform} */
  const transform = {
    scale,
    worldWidth: size / scale,
    worldHeight: size / scale,
    viewportBounds: bounds,
    mapBounds: bounds,
    worldToScreen: (point) => {
      const [x, y] = Array.isArray(point)
        ? point
        : [/** @type {any} */ (point).x, /** @type {any} */ (point).y];
      return {
        x: PORTRAIT_RADIUS_PX + (finiteNumber(x) - finiteNumber(centre[0])) * scale,
        y: PORTRAIT_RADIUS_PX - (finiteNumber(y) - finiteNumber(centre[1])) * scale,
      };
    },
    worldLengthToScreen: (length) => length * scale,
  };
  try {
    const tokenRecords = champions.records({
      scene: { agents: [agent], selection: {} },
      transform,
    });
    const nodes = champions.build(tokenRecords, doc);
    champions.update(nodes, tokenRecords);
    if (nodes.length === 0) {
      throw new RangeError("no portrait token");
    }
    const root = nodes[0].root;
    root.removeAttribute("tabindex");
    root.removeAttribute("role");
    root.setAttribute("aria-hidden", "true");
    svg.append(root);
  } catch {
    svg.replaceChildren(
      svgNode(doc, "circle", {
        class: "champion-frame__portrait-fallback",
        cx: PORTRAIT_RADIUS_PX,
        cy: PORTRAIT_RADIUS_PX,
        r: PORTRAIT_RADIUS_PX - 2,
        "data-team": agent.team_id === 1 ? "team-a" : "team-b",
        "data-class": classTokenFromId(agent.class_id).cssKey,
      }),
    );
  }
  return svg;
}

/**
 * Return the Recent Ticks cell's words: its glyph ("●" Basic, "◆" Ultimate,
 * "·" no ability, "" when there are no numbers), its label ("This Tick", "Not
 * Visible" or "No Tick Yet") and the plain sentence its tooltip shows. Pure.
 *
 * @param {RecentTickCell} cell
 * @returns {{glyph: string, label: string, summary: string}}
 */
function recentCellWords(cell) {
  if (cell.state === "no-tick") {
    return {
      glyph: "",
      label: "No Tick Yet",
      summary: "This is the start of the match, so no tick has been played yet.",
    };
  }
  if (cell.state === "not-visible") {
    return {
      glyph: "",
      label: "Not Visible",
      summary: "The POV team did not see this champion on this tick.",
    };
  }
  const glyph =
    cell.ability === "ultimate" ? "◆" : cell.ability === "basic" ? "●" : "·";
  const action =
    cell.ability === "ultimate"
      ? "It used its Ultimate."
      : cell.ability === "basic"
        ? "It used its Basic."
        : "It used no ability.";
  const amounts = cell.net
    ? "Agent POV records only the net change in life, so a loss and a heal on one tick show as one number."
    : "Damage taken and healing received are shown on separate lines, never netted.";
  return {
    glyph,
    label: "This Tick",
    summary: `${action} ${amounts}${cell.povOnly ? " Only what the POV team saw is counted." : ""}`,
  };
}

/**
 * Fill the Champion Frame container from a ChampionFrameView.
 *
 * container is the frame's slot under the arena (`#champion-frame`); its
 * children are replaced, data-state is set to "empty", "selected" or
 * "fallen", and data-class to the champion's class CSS key (removed when
 * empty). The empty view shows the hint row. The selected view shows, left
 * to right: the Full Record chip (when view.chip is set), the portrait
 * (buildPortrait), the name block (label, team, the shared Life bar of
 * `life-bar.js` or "Fallen", Ultimate), the status list (one line per status
 * by the status rule, or "No Statuses"), the Recent Ticks cell and Next
 * Target. Nothing here animates, starts a timer or touches storage.
 * Shows and hides are the caller's job (the `selected_agent_frame` switch).
 * Registers Symbol Tip owners on the life, Ultimate and Recent Ticks parts
 * through tooltip.js. Return undefined.
 *
 * @param {HTMLElement} container
 * @param {ChampionFrameView} view
 */
export function renderChampionFrame(container, view) {
  const doc = container.ownerDocument;
  container.classList.add("champion-frame");
  if (view.state === "empty") {
    container.dataset.state = "empty";
    delete container.dataset.class;
    container.replaceChildren(htmlNode(doc, "p", "champion-frame__hint", view.hint));
    return;
  }
  container.dataset.state = view.alive ? "selected" : "fallen";
  container.dataset.class = view.classKey;
  const parts = [];
  if (view.chip !== null) {
    parts.push(htmlNode(doc, "span", "champion-frame__chip", view.chip));
  }
  const portrait = htmlNode(doc, "div", "champion-frame__portrait");
  portrait.append(buildPortrait(doc, view.agent));
  parts.push(portrait);

  const name = htmlNode(doc, "div", "champion-frame__name");
  const heading = htmlNode(doc, "p", "champion-frame__title");
  heading.append(htmlNode(doc, "strong", "champion-frame__label", view.title));
  if (view.teamLabel !== null) {
    const team = htmlNode(doc, "span", "champion-frame__team", view.teamLabel);
    if (view.teamKey !== null) team.dataset.team = view.teamKey;
    heading.append(team);
  }
  const life = htmlNode(doc, "div", "champion-frame__life");
  life.append(
    view.life === null
      ? htmlNode(doc, "span", "champion-frame__life-text", view.lifeText)
      : createLifeBar(doc, view.life),
  );
  registerTooltipOwner(
    life,
    frameTip(
      "life",
      "Life",
      view.alive
        ? `Life now: ${view.lifeText}.`
        : "This champion has fallen and waits for its team's next return.",
    ),
  );
  const ultimate = htmlNode(doc, "p", "champion-frame__ultimate", view.ultimateText);
  ultimate.dataset.ready = String(view.ultimateReady);
  registerTooltipOwner(
    ultimate,
    frameTip(
      "ultimate",
      "Ultimate",
      view.ultimateReady
        ? "The Ultimate can be used on the next tick."
        : `${view.ultimateText}: the ticks until the Ultimate can be used again.`,
    ),
  );
  name.append(heading, life, ultimate);
  parts.push(name);

  const statuses = htmlNode(doc, "ul", "champion-frame__statuses");
  statuses.setAttribute("aria-label", "Statuses");
  if (view.statuses.length === 0) {
    statuses.append(htmlNode(doc, "li", "champion-frame__status-none", "No Statuses"));
  }
  for (const status of view.statuses) {
    const item = htmlNode(doc, "li", "champion-frame__status", status.line);
    item.dataset.token = status.tokenId;
    statuses.append(item);
  }
  parts.push(statuses);

  const words = recentCellWords(view.recent);
  const recent = htmlNode(doc, "div", "champion-frame__recent");
  recent.append(htmlNode(doc, "span", "champion-frame__eyebrow", "Recent Ticks"));
  const cell = htmlNode(doc, "div", "champion-frame__cell");
  cell.dataset.state = view.recent.state;
  cell.append(htmlNode(doc, "span", "champion-frame__cell-label", words.label));
  if (view.recent.state === "seen") {
    const glyph = htmlNode(doc, "span", "champion-frame__cell-glyph", words.glyph);
    glyph.dataset.ability = view.recent.ability ?? "none";
    cell.append(
      glyph,
      htmlNode(
        doc,
        "span",
        "champion-frame__cell-damage",
        view.recent.damage === null
          ? "–"
          : `−${formatDisplayNumber(view.recent.damage, { maximumFractionDigits: 0 })}`,
      ),
    );
    const heal = view.recent.heal ?? view.recent.rest;
    const healLine = htmlNode(
      doc,
      "span",
      "champion-frame__cell-heal",
      heal === null
        ? "–"
        : `+${formatDisplayNumber(heal, { maximumFractionDigits: 0 })}`,
    );
    healLine.dataset.rest = String(
      view.recent.heal === null && view.recent.rest !== null,
    );
    cell.append(healLine);
  }
  registerTooltipOwner(
    cell,
    frameTip("recent", `Recent Ticks · ${words.label}`, words.summary),
  );
  recent.append(cell);
  parts.push(recent);

  if (view.nextText !== null) {
    const next = htmlNode(doc, "p", "champion-frame__next", view.nextText);
    next.dataset.final = String(view.finalFrame);
    registerTooltipOwner(
      next,
      frameTip(
        "next",
        "Next Target",
        view.finalFrame
          ? "This is the last frame of the replay, so there is no next tick."
          : "What this champion did on the next tick, from the recording.",
      ),
    );
    parts.push(next);
  }
  container.replaceChildren(...parts);
}

/**
 * Return one frozen tooltip descriptor (kind "champion-frame", anchored to the
 * element, shown on compact and full cards) for a part of the frame.
 *
 * @param {string} id Short part name, for the descriptor id.
 * @param {string} title The card title, in Title Case.
 * @param {string} summary One plain sentence or two.
 * @returns {ReturnType<typeof createSemanticDescriptor>}
 * @throws {TypeError} From createSemanticDescriptor for an empty title.
 */
function frameTip(id, title, summary) {
  return createSemanticDescriptor({
    kind: "champion-frame",
    id: `champion-frame:${id}`,
    title,
    tone: "information",
    accent: "none",
    summary,
    rows: [],
    sections: [],
    metadata: FRAME_TIP_METADATA,
    anchor: "element",
  });
}
