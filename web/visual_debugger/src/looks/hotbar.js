/**
 * @file The Champion's Hotbar look: the plan marks of live play in the
 * BattleClient and the DevClient combat screen, drawn as the approved vision
 * page draws them (`vision_template.html`, renderArena). Plans are Moon
 * Silver and hollow; the blows themselves are drawn later, in class colors,
 * by the combat looks.
 *
 * What it draws, around the champion you are ordering, bottom to top (the
 * vision's own paint order, all under the bodies):
 * - Command Ring: one silver ring at 1.45 body radii (36.25 px on the
 *   vision's 25 px body), 2 px at 90% with a 6% silver disc. It means "whose
 *   hands" (the Agent POV ring at 0.98 radii means "whose eyes"). A fallen
 *   champion keeps an ashen ring at 45%. Every other Manual champion the
 *   scene draws (One On Each Team) gets the same ring dimmed to 45% with no
 *   disc, and its studs at 65% when its staged lane is known.
 * - Order Studs: four marks on the ring at the diagonals. Silver dots
 *   (r 2.2) for No Combat, class-color dots (r 3.2) for the Basic, class-color
 *   diamonds (half 4.5) for the Ultimate, hollow class-color diamonds for the
 *   Mage's Burst, and hollow gray marks when the host refuses the armed
 *   action. Each dot is its own `<circle>`, as the vision draws it (a circle
 *   made of path arcs lands its edge on different pixels); the four diamonds
 *   share one path. Switch `command_marks`.
 * - Reach Marks: a dashed silver ring (radius 1.45 body radii, 1.25 px,
 *   dashes 2 3, 60%) around each body the armed action may target now, read
 *   from the host's target mask (No Combat shows the Basic's targets; the
 *   Burst has none). A Manual champion's ring (yours or another's) is 5 px
 *   wider, so it clears that champion's Command Ring. Switch
 *   `reach_marks`.
 * - Order Thread: one straight dashed silver line, over a dark underlay, from
 *   your champion's edge to 4 px short of its target's edge, only while the
 *   armed action is allowed on that target (Basic 1.5 px, dashes 5 4, 75%;
 *   Ultimate 2.5 px, dashes 6 4, 85%). Switch `next_action_thread` (the
 *   thread is the live form of the Next Action Thread).
 * - Look Ring: when the champion in focus is a policy's (Policies Only, or
 *   Watch Only on a policy teammate), only a soft silver ring 12 px outside
 *   its body (7% disc, 1 px edge at 55%) and no plan marks: nobody orders it.
 *   Switch `selection_moonlight` (the ring is Selection Moonlight's live form).
 * - Step Ghost: a dotted silver circle the size of your champion where your
 *   planned step ends if nothing blocks it, over a faint silver disc and a
 *   dark ring. Hidden while a Warrior Charge is armed (the Charge decides
 *   where the Warrior lands) and while the step is Stay. Switch `step_ghost`.
 * The marks are a still picture of the current plan: nothing here animates,
 * so the DOM always holds the END state. While a tick plays, the choreographer
 * moves the marks on a champion with its glide and fades the thread and Step
 * Ghost in once the champions have landed (glideTiming), so no mark shows at
 * a landing spot before its champion gets there. Each mark names the champion it
 * belongs to (returnKey) through a "set-down" beat hold: on the tick that
 * champion returns, the choreographer keeps the mark hidden until the
 * champion is set down on its waystone, so no Command Ring, stud, thread,
 * Reach Mark or Step Ghost shows at the waystone before the champion does;
 * on any other tick the hold does nothing.
 *
 * Step end: the host's own facts give it. The draft inspection gives your
 * planned step (`draft_action.move_action`) and the scene gives your
 * champion's place and its effective speed (`effective_movement_speed`, which
 * Core's shared speed rule computes, so slows, stuns and the spawn shield are
 * in it). The end is place + speed x the step's compass direction,
 * MOVE_DIRECTIONS, a copy of Core's fixed table
 * (`core/axis_mappings.py`, UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION);
 * `tests/test_battle_client_step_ghost.py` checks the copy and the result
 * against Core's own planned movement.
 *
 * Sizes: every size is the vision's size on its 25 px body times (body radius
 * on screen / 25), so the marks keep the vision's proportions at any window
 * size (FOOTPRINT, sizeScale). Line widths and dash lengths live in
 * `styles/hotbar.css` and grow the same way through the battlefield's
 * `--mark-scale` variable (markScale).
 *
 * Who calls it: `scene.js` calls `records` then `build` on every render and
 * puts the nodes in the body layer's plan-marks group, under every body. It
 * passes the Manual slots from the transport frame's combat configuration
 * (when the page gives them) and the frame's own top-level staged Manual
 * rows. Only
 * live frames have a controlled champion
 * (`scene.selection.controlled_presentation_key`), so the Replay Viewer gets
 * no records.
 *
 * Boundary and Agent POV: `records` reads only the arena scene of the frame's
 * audience and that same audience's live inspection (`input.inspection`, from
 * `authorizedPresentationInspectionState`, never the researcher view). Every
 * mark sits on a body that the scene draws, or on your own champion's own
 * step, so in Agent POV no mark points at or rings a champion the team cannot
 * see. Reach and legality come only from the host's masks; nothing is worked
 * out from distances. Every node is export-safe (flat fills, strokes and
 * opacity), and none of these marks enters PNG provenance.
 */
import { classIdentityFromId } from "../agent-identity.js";
import {
  asArray,
  finiteNumber,
  isRecord,
  paintParts,
  screenPoint,
  setNodeAttributes,
  svgNode,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").LookInput} LookInput
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 */

/**
 * The sizes of the hotbar marks as the vision page draws them on its 25 px
 * body (VISION_BODY_RADIUS), in that drawing's pixels unless named otherwise.
 * The live field is drawn at the window's size, so every size here is
 * multiplied by (body radius on screen / 25) before it is drawn: the marks
 * keep the vision's proportions at any window size (a 12 px gap on the
 * vision's 25 px body is a 10.2 px gap on a 21.2 px body).
 * `bandRadiusFactor` is the Command Ring radius as a multiple of the body
 * radius on screen; the collision allocator keeps chips off this band.
 * `studDotRadius`, `studNoCombatRadius` and `studDiamondHalf` size the Order
 * Studs. `ownReachExtra` widens the Reach Mark on every Manual champion
 * (yours and any other this scene draws a Command Ring for) so it clears that
 * Command Ring, as the vision's `isManual(b) ? BAND + 5 : BAND` does. `threadTargetGap` is how far short of the
 * target's edge the Order Thread stops; it starts at your champion's edge.
 * `ghostRingInset` is how far inside the Step Ghost's disc its rings lie.
 * `lookRingGap` is how far outside the body the Look Ring lies (the
 * vision's R + 12). A thread shorter than `threadMinimumLength` screen
 * pixels is not drawn. A Step Ghost nearer than `ghostMinimumOffset` screen
 * pixels to your champion is not drawn (a stunned champion does not move).
 */
export const FOOTPRINT = Object.freeze({
  bandRadiusFactor: 1.45,
  studAnglesDegrees: Object.freeze([45, 135, 225, 315]),
  studDotRadius: 3.2,
  studNoCombatRadius: 2.2,
  studDiamondHalf: 4.5,
  ownReachExtra: 5,
  threadTargetGap: 4,
  threadMinimumLength: 1,
  ghostRingInset: 1,
  ghostMinimumOffset: 1,
  lookRingGap: 12,
});

/**
 * Node allowance per record: each record draws exactly one SVG node (a ring
 * circle, one studs path, one thread line, one reach ring or one Step Ghost
 * circle). This look starts no animation of its own; on a tick its champion
 * returns, the choreographer gives each node one beat-hold override (counted
 * in the plan's override allowance, not here).
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 1,
  animationsPerRecord: 0,
  persistentNodesPerRecord: 0,
});

/**
 * The hotbar's three switchable paint parts. The Command Ring and the Order
 * Thread have no switch (they are the plain plan marks a switch falls back to).
 *
 * @type {ReadonlyArray<import("./look-support.js").PaintPartRegistration>}
 */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "command_marks" }, "command_marks"],
  [{ surface: "durable", kind: "reach_marks" }, "reach_marks"],
  [{ surface: "durable", kind: "step_ghost" }, "step_ghost"],
]);

/**
 * The compass direction of each step, indexed by `move_action` 0 to 8, in
 * world units (y counts up): Stay, North, South, East, West, North-East,
 * North-West, South-East, South-West. A copy of Core's fixed table
 * UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION (`core/axis_mappings.py`);
 * diagonals are 0.7071067690849304 (Core's float32 value) on each axis.
 * `tests/test_battle_client_step_ghost.py` checks that this copy equals Core.
 */
export const MOVE_DIRECTIONS = Object.freeze([
  Object.freeze([0, 0]),
  Object.freeze([0, 1]),
  Object.freeze([0, -1]),
  Object.freeze([1, 0]),
  Object.freeze([-1, 0]),
  Object.freeze([0.7071067690849304, 0.7071067690849304]),
  Object.freeze([-0.7071067690849304, 0.7071067690849304]),
  Object.freeze([0.7071067690849304, -0.7071067690849304]),
  Object.freeze([-0.7071067690849304, -0.7071067690849304]),
]);

/** Class id of the Mage (Burst: an Ultimate with no target) and the Warrior (Charge). */
const MAGE_CLASS_ID = 1;
const WARRIOR_CLASS_ID = 2;

/** The body radius, in pixels, of the vision page's own drawing; FOOTPRINT sizes are for it. */
export const VISION_BODY_RADIUS = 25;

/**
 * Return how much bigger (or smaller) than the vision's drawing a body is
 * drawn now: its radius on screen over VISION_BODY_RADIUS. Multiply a
 * FOOTPRINT size by it to draw that size in screen pixels. Pure.
 *
 * @param {number} radius The body radius on screen, in CSS pixels (positive).
 * @returns {number} The scale (1 on the vision's own 25 px body).
 */
function sizeScale(radius) {
  return radius / VISION_BODY_RADIUS;
}

/** Pixels per world unit of the vision page's own field drawing (1032 x 532, 16 px pad). */
export const VISION_PIXELS_PER_UNIT = 50;

/**
 * Return the stroke scale of the live field: how many times bigger than the
 * vision's own drawing one world unit is drawn now (its pixels per unit over
 * VISION_PIXELS_PER_UNIT). `scene.js` puts it in the battlefield's
 * `--mark-scale` CSS variable, and the plan and reach line widths and dash
 * lengths in `styles/hotbar.css` and `styles/battleclient/arena.css` are
 * multiplied by it, so a 2.5 px Charge thread is 2.6 px at 1440 x 900, as
 * the vision's (its field is 1.0407 times its drawing there). Mark positions
 * and radii scale through sizeScale instead. Pure; reads only the transform.
 *
 * @param {Pick<LookTransform, "worldLengthToScreen">} transform The
 *   renderer's viewport transform for this render (world units to screen
 *   pixels); only its worldLengthToScreen is read.
 * @returns {number} The scale, rounded to 4 decimals (1 when the transform
 *   gives no positive finite length).
 */
export function markScale(transform) {
  const perUnit = transform.worldLengthToScreen(1);
  if (!Number.isFinite(perUnit) || perUnit <= 0) return 1;
  return Math.round((perUnit / VISION_PIXELS_PER_UNIT) * 10000) / 10000;
}

/**
 * Return a scene body's shown Agent ID (its global slot, 0 to 9), or null.
 *
 * The id is the body's `display_agent_id` when it is one digit: the same
 * number its token and portrait show, in Agent POV too. It is never worked out
 * from anything hidden. Pure.
 *
 * @param {Record<string, any>} agent An arena scene agent.
 * @returns {number | null}
 */
function shownSlot(agent) {
  return /^\d$/u.test(String(agent.display_agent_id ?? ""))
    ? Number(agent.display_agent_id)
    : null;
}

/**
 * Return the armed lane of one staged Manual draft row, or null without one.
 *
 * row is a `pending_joint_action.action_rows` entry; its `pending_action`
 * holds the submitted triple: `use_ultimate_action` 1 means the Ultimate is
 * armed, else a `target_action` above 0 means the Basic, else No Combat.
 * Pure.
 *
 * @param {Record<string, any> | null | undefined} row The staged row, or null.
 * @returns {"none" | "basic" | "ultimate" | null}
 */
function pendingLane(row) {
  const action = isRecord(row?.pending_action) ? row.pending_action : null;
  if (action === null) return null;
  if (action.use_ultimate_action === 1) return "ultimate";
  return Number.isInteger(action.target_action) && action.target_action > 0
    ? "basic"
    : "none";
}

/**
 * Return the other Manual champions this scene draws, each with its screen
 * place and staged lane, in shown-slot order.
 *
 * A body counts as Manual when its shown slot is in `manualSlots` (the
 * transport frame's combat configuration), or, with no `manualSlots`, when
 * one of `pendingRows` names it (the staged Manual drafts; their keys are
 * the Oracle View's, so this finds nothing in Agent POV). Only bodies in
 * `agents` (the audience's own scene) are returned, so in Agent POV a Manual
 * champion the team cannot see gets no mark. `lane` comes from the row whose
 * presentation key and public id both match the body, else null (the ring
 * is drawn without studs). The champion keyed `skipKey` is left out. Pure.
 *
 * @param {ReadonlyArray<Record<string, any>>} agents The scene's bodies.
 * @param {LookTransform} transform This render's projection.
 * @param {ReadonlyArray<number> | null} manualSlots The Manual slots, or null.
 * @param {ReadonlyArray<Record<string, any>>} pendingRows Staged Manual rows.
 * @param {string} skipKey The presentation key to leave out.
 * @returns {Array<{agent: Record<string, any>, center: {x: number, y: number, r: number}, slot: number | null, lane: "none" | "basic" | "ultimate" | null}>}
 */
function otherManualBodies(agents, transform, manualSlots, pendingRows, skipKey) {
  const rowByKey = new Map(
    pendingRows.map((row) => [String(row.actor_presentation_key), row]),
  );
  const out = [];
  for (const agent of agents) {
    if (agent.presentation_key === skipKey) continue;
    const slot = shownSlot(agent);
    const candidate = rowByKey.get(String(agent.presentation_key));
    const row =
      candidate !== undefined &&
      candidate.actor_public_agent_id === agent.public_agent_id
        ? candidate
        : null;
    const manual =
      manualSlots !== null ? slot !== null && manualSlots.includes(slot) : row !== null;
    if (!manual) continue;
    const center = projectAgent(agent, transform);
    if (center === null) continue;
    out.push({ agent, center, slot, lane: pendingLane(row) });
  }
  return out.sort((a, b) => (a.slot ?? -1) - (b.slot ?? -1));
}

/**
 * Return the dimmed Command Ring and Order Studs of a Manual champion that is
 * not in focus, as the vision draws every Manual champion it is not ordering
 * (ring at 45% with no disc, studs at 65%, so No Combat dots at 58.5%: the
 * `away` look, styles/hotbar.css). A fallen one
 * keeps only its ashen ring. With no known lane the studs are left out (Agent
 * POV has no other champion's draft). Pure.
 *
 * @param {{agent: Record<string, any>, center: {x: number, y: number, r: number}, lane: "none" | "basic" | "ultimate" | null}} body
 *   One entry of otherManualBodies.
 * @param {Record<string, any>} filters The switch state (`command_marks`).
 * @returns {LookRecord[]} The ring record, then no stud records (fallen, no
 *   known lane or `command_marks` off), one diamond record or four dot records.
 */
function awayMarks(body, filters) {
  const { agent, center, lane } = body;
  const alive = agent.alive === true || agent.life_state === "alive";
  const band = round(center.r * FOOTPRINT.bandRadiusFactor);
  /** @type {LookRecord[]} */
  const out = [
    Object.freeze({
      kind: "command-ring",
      identityKey: String(agent.presentation_key),
      cx: round(center.x),
      cy: round(center.y),
      r: band,
      alive,
      away: true,
      returnKey: String(agent.presentation_key),
    }),
  ];
  if (alive && lane !== null && filters.command_marks !== false) {
    out.push(...studsRecords(agent, center, lane, true, true));
  }
  return out;
}

/**
 * Round a pixel value to two decimals so records stay small and stable.
 *
 * @param {number} value Any finite number.
 * @returns {number}
 */
function round(value) {
  return Math.round(value * 100) / 100;
}

/**
 * Round a pixel value to three decimals, for the small Order Studs.
 *
 * Why the studs need more than `round`'s two decimals: the fitted live field
 * is drawn smaller than the vision's (0.842 CSS px per vision px at
 * 1440 x 900), so a 0.01 px step is coarser than the vision's own 0.01 grid,
 * and Chrome places the edge of a 3 px dot on a fixed sub-pixel grid. A
 * 0.003 px rounding nudge on one stud center moved 15 edge pixels by up to
 * 59 color levels in the equal-scale picture; three decimals keep every stud
 * within 0.0015 px of its exact place. Pure.
 *
 * @param {number} value Any finite number, in pixels.
 * @returns {number} The value rounded to the nearest 0.001.
 */
function roundFine(value) {
  return Math.round(value * 1000) / 1000;
}

/**
 * Return the point at `radius` pixels from (cx, cy) at a compass angle
 * (0 = up, 90 = right), as the vision's `pt` does, rounded to 0.001 px
 * (roundFine). Only the Order Studs use it. Pure.
 *
 * @param {number} cx Center x in pixels.
 * @param {number} cy Center y in pixels.
 * @param {number} radius Distance in pixels.
 * @param {number} degrees Compass angle in degrees.
 * @returns {[number, number]} The point's x and y in pixels.
 */
function polar(cx, cy, radius, degrees) {
  const angle = (degrees * Math.PI) / 180;
  return [
    roundFine(cx + radius * Math.sin(angle)),
    roundFine(cy - radius * Math.cos(angle)),
  ];
}

/**
 * Read a scene agent's screen center and screen radius, or null when it has
 * no usable position.
 *
 * @param {Record<string, any>} agent An arena scene agent.
 * @param {LookTransform} transform This render's projection.
 * @returns {{x: number, y: number, r: number} | null}
 */
function projectAgent(agent, transform) {
  if (!Array.isArray(agent.position) || agent.position.length !== 2) return null;
  const point = screenPoint(agent.position, transform);
  const r = transform.worldLengthToScreen(finiteNumber(agent.radius));
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !(r > 0)) {
    return null;
  }
  return { x: point.x, y: point.y, r };
}

/**
 * Return the controlled champion's own draft inspection, or null.
 *
 * @param {Record<string, any> | null} inspection The audience's live draft
 *   inspection, or null.
 * @param {string} controlledKey The controlled champion's presentation key.
 * @returns {Record<string, any> | null} The inspection when it belongs to the
 *   controlled champion, else null.
 */
function ownInspection(inspection, controlledKey) {
  return isRecord(inspection) && inspection.actor_presentation_key === controlledKey
    ? inspection
    : null;
}

/**
 * Work out the armed lane and whether the host allows it, for the
 * controlled champion.
 *
 * The draft comes from `inspection` only when the inspection belongs to the
 * controlled champion; otherwise the scene's `selected_legality` is used.
 * With neither, the lane is "none". Legality is the host's answer for the
 * armed pair when a target is set (`selected_legality.armed_pair_legal`), else
 * the draft's own `draft_legality` flags; No Combat is always allowed.
 *
 * @param {Record<string, any>} scene The arena scene.
 * @param {Record<string, any> | null} own The controlled champion's own draft
 *   inspection, or null.
 * @returns {{lane: "none" | "basic" | "ultimate", legal: boolean}}
 */
function armedPlan(scene, own) {
  const legality = isRecord(scene.selected_legality) ? scene.selected_legality : null;
  /** @type {"none" | "basic" | "ultimate"} */
  let lane = "none";
  const draftLane = own?.draft_action?.armed_lane;
  if (draftLane === "basic" || draftLane === "ultimate") {
    lane = draftLane;
  } else if (draftLane === undefined && legality !== null) {
    lane =
      legality.armed_lane === 0
        ? "basic"
        : legality.armed_lane === 1
          ? "ultimate"
          : "none";
  }
  if (lane === "none") return { lane, legal: true };
  if (legality !== null && typeof legality.armed_pair_legal === "boolean") {
    return { lane, legal: legality.armed_pair_legal };
  }
  const draftLegality = isRecord(own?.draft_legality) ? own.draft_legality : null;
  if (draftLegality !== null) {
    return {
      lane,
      legal:
        draftLegality.armed_lane_is_legal !== false &&
        draftLegality.combat_pair_is_legal !== false,
    };
  }
  return { lane, legal: true };
}

/**
 * Return the world point where the controlled champion's planned step ends
 * if nothing blocks it, or null when there is no step to show.
 *
 * The step is the draft's `move_action` (1 to 8; 0 is Stay and gives null).
 * The end is the champion's scene `position` plus its scene
 * `effective_movement_speed` (world units per tick; Core already applied
 * slows, stuns and the spawn shield) times MOVE_DIRECTIONS[move]. A missing
 * or non-finite fact gives null. It never reads another champion.
 *
 * @param {Record<string, any>} controlled The controlled champion's scene row.
 * @param {Record<string, any> | null} own Its own draft inspection, or null.
 * @returns {[number, number] | null} World x and y (y up), or null.
 */
function plannedStepEnd(controlled, own) {
  const move = own?.draft_action?.move_action;
  const direction = Number.isInteger(move) ? MOVE_DIRECTIONS[move] : undefined;
  const speed = controlled.effective_movement_speed;
  const position = controlled.position;
  if (
    direction === undefined ||
    move === 0 ||
    typeof speed !== "number" ||
    !Number.isFinite(speed) ||
    !Array.isArray(position) ||
    position.length !== 2 ||
    !position.every((value) => typeof value === "number" && Number.isFinite(value))
  ) {
    return null;
  }
  return [position[0] + speed * direction[0], position[1] + speed * direction[1]];
}

/**
 * Return the Order Studs records of one champion: four studs on its Command
 * Ring, shaped by the armed lane ("no-combat" small dots, "basic" dots,
 * "ultimate" or "burst" diamonds, "refused" when the host refuses the armed
 * pair). The studs sit at the four diagonals of the exact Command Ring
 * radius (center.r times FOOTPRINT.bandRadiusFactor, not the ring record's
 * two-decimal radius). Stud centers and sizes keep three decimals
 * (roundFine), so the small dots land on the same pixels as the vision's at
 * the fitted field's size. Pure.
 *
 * @param {Record<string, any>} champion The scene body (its class_id).
 * @param {{x: number, y: number, r: number}} center Its center and radius on
 *   screen (px), unrounded; the stud sizes scale with r (sizeScale) and the
 *   ring radius is 1.45 r.
 * @param {string} lane The armed lane: "none", "basic" or "ultimate".
 * @param {boolean} legal False when the host refuses the armed pair.
 * @param {boolean} away True for a Manual champion that is not in focus
 *   (Watch Only): the studs are drawn dimmed.
 * @returns {LookRecord[]} Diamonds ("ultimate", "burst", or "refused" on the
 *   Ultimate lane): one record `{kind: "order-studs", shape: "diamond", cx,
 *   cy, d, style, classKey, away}` whose path `d` draws all four. Dots
 *   ("no-combat", "basic", or "refused" on the Basic lane): four records
 *   `{kind: "order-studs", shape: "dot", cx, cy, x, y, r, style, classKey,
 *   away}`, one per stud in the order 45, 135, 225 and 315 degrees, each drawn
 *   as an SVG `<circle>` at (x, y) with radius r (screen px), because the
 *   vision draws its dots as circles and Chrome rasterizes a circle element's
 *   edge differently from a circle made of path arcs. cx and cy are always the
 *   champion's center.
 */
function studsRecords(champion, center, lane, legal, away) {
  // The exact ring radius, not the ring record's two-decimal one (roundFine).
  const band = center.r * FOOTPRINT.bandRadiusFactor;
  const burst = lane === "ultimate" && champion.class_id === MAGE_CLASS_ID;
  const style = !legal
    ? "refused"
    : burst
      ? "burst"
      : lane === "basic"
        ? "basic"
        : lane === "ultimate"
          ? "ultimate"
          : "no-combat";
  const diamond =
    style === "burst" ||
    style === "ultimate" ||
    (style === "refused" && lane === "ultimate");
  const size = roundFine(
    sizeScale(center.r) *
      (diamond
        ? FOOTPRINT.studDiamondHalf
        : style === "no-combat"
          ? FOOTPRINT.studNoCombatRadius
          : FOOTPRINT.studDotRadius),
  );
  const classKey = classIdentityFromId(champion.class_id)?.tokenId ?? "unknown";
  if (!diamond) {
    return FOOTPRINT.studAnglesDegrees.map((degrees) => {
      const [x, y] = polar(center.x, center.y, band, degrees);
      return Object.freeze({
        kind: "order-studs",
        shape: "dot",
        cx: round(center.x),
        cy: round(center.y),
        x,
        y,
        r: size,
        style,
        classKey,
        away,
        returnKey: String(champion.presentation_key),
      });
    });
  }
  const d = FOOTPRINT.studAnglesDegrees
    .map((degrees) => {
      const [x, y] = polar(center.x, center.y, band, degrees);
      return `M${x} ${roundFine(y - size)}L${roundFine(x + size)} ${y}L${x} ${roundFine(y + size)}L${roundFine(x - size)} ${y}Z`;
    })
    .join("");
  return [
    Object.freeze({
      kind: "order-studs",
      shape: "diamond",
      cx: round(center.x),
      cy: round(center.y),
      d,
      style,
      classKey,
      away,
      returnKey: String(champion.presentation_key),
    }),
  ];
}

/**
 * Return this look's drawing records: the hotbar marks of the champion you
 * are ordering, as plain frozen data in screen pixels.
 *
 * input fields read:
 * - `scene` (required): the arena scene of the frame's audience. Its
 *   `selection.controlled_presentation_key` names the champion you order
 *   (live only); `agents` gives bodies (with `effective_movement_speed`);
 *   `pending_route` gives the planned target's anchors; `selected_legality`
 *   gives the host's answer for the armed pair.
 * - `transform` (required): this render's projection.
 * - `inspection` (optional, default null): the same audience's live draft
 *   inspection (`authorizedPresentationInspectionState(frame).inspection`).
 *   Its `draft_action` (`armed_lane`, `move_action`) and `decision_mask` give
 *   the studs, the Reach Marks and the Step Ghost. Without it the studs follow
 *   `selected_legality` and no Reach Marks or Step Ghost are drawn.
 * - `filters` (optional, default all on): `command_marks: false` drops the
 *   studs; `reach_marks: false` drops the Reach Marks; `step_ghost: false`
 *   drops the Step Ghost; `selection_moonlight: false` drops the Look Ring
 *   (its live form: the soft ring round a looked-at champion);
 *   `next_action_thread: false` drops the Order Thread (the live form of the
 *   Next Action Thread: the line from your champion to its target).
 * - `lookedAtKey` (optional, default null): the presentation key of the
 *   champion the player looks at when it is not the champion being ordered
 *   (Watch Only in Agent POV: the page maps the transport frame's
 *   `focused_public_agent_id` to a body of this scene). When that body is in
 *   `scene.agents`, the records are Watch Only's: the soft `look-ring` round
 *   it, then your champion's Command Ring and studs dimmed (`away: true`), as
 *   the vision draws a Manual champion that is not in focus; no Reach Marks,
 *   Order Thread or Step Ghost. A key not in the scene is ignored, so an
 *   unseen champion never gets a mark.
 * - `manualSlots` (optional, default null): the shown slots (0 to 9) of every
 *   Manual champion, from the transport frame's combat configuration. With
 *   null, the Manual champions are the ones `pendingRows` names.
 * - `pendingRows` (optional, default []): the frame's own top-level
 *   `pending_joint_action.action_rows` (the staged Manual drafts; never the
 *   researcher view's copy). They give another Manual champion's staged lane.
 *   Every other Manual champion this scene draws gets a dimmed Command Ring
 *   (and its studs when its lane is known), as the vision draws One On Each
 *   Team: the champion you order at full strength, the other at 45%.
 *
 * When the inspection is a policy's read-only view (`inspection_kind`
 * "policy_read_only": Policies Only, or Watch Only on a policy's champion),
 * the records are `look-ring` ({identityKey, cx, cy, r}: the soft ring
 * 12 vision pixels outside the body) and the dimmed rings of the Manual
 * champions this scene draws, as the vision draws a looked-at champion.
 * Every size scales with the body's radius on screen (sizeScale), so the
 * marks keep the vision's proportions at any window size.
 *
 * Records, bottom to top (the vision's paint order); every record also has
 * returnKey, the presentation key of the champion whose Return it waits for
 * (see returnHold):
 * `command-ring` ({identityKey, cx, cy, r, alive, away}),
 * `order-studs` (style "no-combat", "basic", "ultimate", "burst" or
 * "refused": one `shape: "diamond"` record {cx, cy, d, style, classKey, away}
 * whose path d draws all four diamonds, or four `shape: "dot"` records {cx,
 * cy, x, y, r, style, classKey, away}, one circle each; see studsRecords),
 * `reach-mark` ({identityKey, cx, cy, r}), `order-thread-ink` and
 * `order-thread` (both {x1, y1, x2, y2, ultimate, targetKey}, targetKey the
 * thread's target's presentation key), then `step-ghost-disc`,
 * `step-ghost-ink` and `step-ghost` (each {cx, cy, r, moveAction}). A dead
 * champion keeps an ashen ring and no other mark. No controlled champion in
 * the scene gives [].
 *
 * Agent POV: only bodies in `scene.agents` are ringed or joined, so a target
 * or another Manual champion the team cannot see gets no mark; the Step
 * Ghost is your own champion's own step. Pure: no DOM, no storage, no timers.
 *
 * @param {LookInput} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const scene = isRecord(input?.scene) ? input.scene : null;
  const transform = input?.transform ?? null;
  const controlledKey = scene?.selection?.controlled_presentation_key;
  if (scene === null || !transform || typeof controlledKey !== "string") {
    return Object.freeze([]);
  }
  const agents = asArray(scene.agents).filter(isRecord);
  const controlled = agents.find((agent) => agent.presentation_key === controlledKey);
  const center = controlled ? projectAgent(controlled, transform) : null;
  if (!controlled || center === null) return Object.freeze([]);
  const filters = isRecord(input.filters) ? input.filters : {};
  const inspection = isRecord(input.inspection) ? input.inspection : null;
  const own = ownInspection(inspection, controlledKey);
  const band = round(center.r * FOOTPRINT.bandRadiusFactor);
  const scale = sizeScale(center.r);
  const others = otherManualBodies(
    agents,
    transform,
    Array.isArray(input.manualSlots) ? input.manualSlots : null,
    asArray(input.pendingRows).filter(isRecord),
    controlledKey,
  );
  const alive = controlled.alive === true || controlled.life_state === "alive";
  /** @type {LookRecord[]} */
  const out = [];
  const lookedAtKey =
    typeof input.lookedAtKey === "string" && input.lookedAtKey !== controlledKey
      ? input.lookedAtKey
      : null;
  const lookedAt =
    lookedAtKey === null
      ? null
      : (agents.find((agent) => agent.presentation_key === lookedAtKey) ?? null);
  const lookedAtCenter = lookedAt ? projectAgent(lookedAt, transform) : null;
  // Selection Moonlight's live form is the Look Ring: off drops the ring only.
  const lookRingOn = filters.selection_moonlight !== false;
  if (lookedAt !== null && lookedAtCenter !== null) {
    // Watch Only: a policy's champion (or a seen enemy) is in focus. The
    // vision rings it softly and keeps your champion's ring and studs dimmed.
    if (lookRingOn)
      out.push(
        Object.freeze({
          kind: "look-ring",
          identityKey: String(lookedAtKey),
          cx: round(lookedAtCenter.x),
          cy: round(lookedAtCenter.y),
          r: round(
            lookedAtCenter.r + sizeScale(lookedAtCenter.r) * FOOTPRINT.lookRingGap,
          ),
          returnKey: String(lookedAtKey),
        }),
      );
    // Every Manual champion, in shown-slot order, dimmed: yours with your
    // own draft's studs, any other with its staged lane.
    const yours = {
      slot: shownSlot(controlled),
      marks: () => {
        /** @type {LookRecord[]} */
        const marks = [
          Object.freeze({
            kind: "command-ring",
            identityKey: controlledKey,
            cx: round(center.x),
            cy: round(center.y),
            r: band,
            alive,
            away: true,
            returnKey: controlledKey,
          }),
        ];
        if (alive && filters.command_marks !== false) {
          const { lane, legal } = armedPlan(scene, own);
          marks.push(...studsRecords(controlled, center, lane, legal, true));
        }
        return marks;
      },
    };
    const dimmed = [
      ...others
        .filter((body) => body.agent.presentation_key !== lookedAtKey)
        .map((body) => ({ slot: body.slot, marks: () => awayMarks(body, filters) })),
      yours,
    ].sort((a, b) => (a.slot ?? -1) - (b.slot ?? -1));
    for (const entry of dimmed) out.push(...entry.marks());
    return Object.freeze(out);
  }
  if (own !== null && own.inspection_kind !== "live_draft_action") {
    // A policy's champion in focus (Policies Only, or Watch Only on a policy
    // teammate): nobody orders it from here, so the vision draws the soft
    // Look Ring instead of a Command Ring, and no plan marks.
    if (lookRingOn)
      out.push(
        Object.freeze({
          kind: "look-ring",
          identityKey: controlledKey,
          cx: round(center.x),
          cy: round(center.y),
          r: round(center.r + scale * FOOTPRINT.lookRingGap),
          returnKey: controlledKey,
        }),
      );
    for (const body of others) out.push(...awayMarks(body, filters));
    return Object.freeze(out);
  }
  // The vision draws every Manual champion's ring and studs in slot order:
  // the one you order at full strength, any other dimmed.
  const ownSlot = shownSlot(controlled);
  const { lane, legal } = armedPlan(scene, own);
  for (const body of others) {
    if ((body.slot ?? -1) < (ownSlot ?? -1)) out.push(...awayMarks(body, filters));
  }
  out.push(
    Object.freeze({
      kind: "command-ring",
      identityKey: controlledKey,
      cx: round(center.x),
      cy: round(center.y),
      r: band,
      alive,
      returnKey: controlledKey,
    }),
  );
  if (alive && filters.command_marks !== false) {
    out.push(...studsRecords(controlled, center, lane, legal, false));
  }
  for (const body of others) {
    if ((body.slot ?? -1) >= (ownSlot ?? -1)) out.push(...awayMarks(body, filters));
  }
  if (!alive) return Object.freeze(out);
  const burst = lane === "ultimate" && controlled.class_id === MAGE_CLASS_ID;

  if (filters.reach_marks !== false && !burst && own !== null) {
    const mask = isRecord(own.decision_mask) ? own.decision_mask : null;
    const ownMask =
      mask !== null && mask.owner_presentation_key === controlledKey ? mask : null;
    const laneIndex = lane === "ultimate" ? 1 : 0;
    const pairs = asArray(ownMask?.target_use_ultimate_joint_mask);
    const manualKeys = new Set([
      String(controlledKey),
      ...others.map((body) => String(body.agent.presentation_key)),
    ]);
    for (const target of asArray(ownMask?.target_actions).filter(isRecord)) {
      const targetAction = Number(target.target_action);
      if (
        !Number.isInteger(targetAction) ||
        targetAction === 0 ||
        target.target_kind !== "visible_authorized_agent" ||
        asArray(pairs[targetAction])[laneIndex] !== true
      ) {
        continue;
      }
      const body = agents.find(
        (agent) =>
          agent.presentation_key === target.target_presentation_key &&
          agent.public_agent_id === target.target_public_agent_id,
      );
      const point = body ? projectAgent(body, transform) : null;
      if (!body || point === null) continue;
      // Like the vision (`isManual(b) ? BAND + 5 : BAND`), every Manual
      // champion's Reach Mark clears its Command Ring, not only yours.
      const extra = manualKeys.has(String(body.presentation_key))
        ? sizeScale(point.r) * FOOTPRINT.ownReachExtra
        : 0;
      out.push(
        Object.freeze({
          kind: "reach-mark",
          identityKey: String(body.presentation_key),
          cx: round(point.x),
          cy: round(point.y),
          r: round(point.r * FOOTPRINT.bandRadiusFactor + extra),
          returnKey: String(body.presentation_key),
        }),
      );
    }
  }

  const route = isRecord(scene.pending_route) ? scene.pending_route : null;
  if (
    route !== null &&
    lane !== "none" &&
    legal &&
    !burst &&
    route.source_presentation_key === controlledKey &&
    route.target_presentation_key !== controlledKey &&
    agents.some((agent) => agent.presentation_key === route.target_presentation_key)
  ) {
    const source = screenPoint(route.source_anchor, transform);
    const target = screenPoint(route.target_anchor, transform);
    const sourceRadius = transform.worldLengthToScreen(
      finiteNumber(route.source_radius),
    );
    const targetRadius = transform.worldLengthToScreen(
      finiteNumber(route.target_radius),
    );
    const length = Math.hypot(target.x - source.x, target.y - source.y);
    const startGap = sourceRadius;
    const endGap = targetRadius + scale * FOOTPRINT.threadTargetGap;
    if (
      Number.isFinite(length) &&
      length - startGap - endGap >= FOOTPRINT.threadMinimumLength
    ) {
      const ux = (target.x - source.x) / length;
      const uy = (target.y - source.y) / length;
      const line = {
        x1: round(source.x + ux * startGap),
        y1: round(source.y + uy * startGap),
        x2: round(target.x - ux * endGap),
        y2: round(target.y - uy * endGap),
        ultimate: lane === "ultimate",
        returnKey: controlledKey,
        targetKey: String(route.target_presentation_key),
      };
      if (filters.next_action_thread !== false) {
        out.push(Object.freeze({ kind: "order-thread-ink", ...line }));
        out.push(Object.freeze({ kind: "order-thread", ...line }));
      }
    }
  }

  const charging = lane === "ultimate" && controlled.class_id === WARRIOR_CLASS_ID;
  const stepEnd =
    filters.step_ghost !== false && !charging ? plannedStepEnd(controlled, own) : null;
  if (stepEnd !== null) {
    const point = screenPoint(stepEnd, transform);
    if (
      Number.isFinite(point.x) &&
      Number.isFinite(point.y) &&
      Math.hypot(point.x - center.x, point.y - center.y) >= FOOTPRINT.ghostMinimumOffset
    ) {
      const ghost = {
        cx: round(point.x),
        cy: round(point.y),
        moveAction: Number(own?.draft_action?.move_action),
        returnKey: controlledKey,
      };
      const inner = round(center.r - scale * FOOTPRINT.ghostRingInset);
      out.push(
        Object.freeze({ kind: "step-ghost-disc", ...ghost, r: round(center.r) }),
      );
      out.push(Object.freeze({ kind: "step-ghost-ink", ...ghost, r: inner }));
      out.push(Object.freeze({ kind: "step-ghost", ...ghost, r: inner }));
    }
  }
  return Object.freeze(out);
}

/**
 * Return the SVG tag and attributes for one record: its mark (markSpec), its
 * Return hold (returnHold) and its glide timing (glideTiming). Null for an
 * unknown kind.
 *
 * @param {LookRecord} record One hotbar record.
 * @returns {{tag: string, attributes: Record<string, unknown>} | null}
 */
function nodeSpec(record) {
  const spec = markSpec(record);
  return spec === null
    ? null
    : {
        tag: spec.tag,
        attributes: {
          ...spec.attributes,
          ...returnHold(record),
          ...glideTiming(record),
        },
      };
}

/** The marks that show what comes next; they wait for the landing. */
const LANDING_WAIT_KINDS = new Set([
  "order-thread",
  "order-thread-ink",
  "step-ghost",
  "step-ghost-ink",
  "step-ghost-disc",
]);

/**
 * Return the attributes that time a mark against its champions' glides.
 *
 * While a tick plays, each body glides from where it started to where it
 * ends, but the marks are drawn where the plan stands at the end. So a mark
 * on a champion (Command Ring, Look Ring, Order Studs, Reach Mark) carries
 * data-glide-follow with record.returnKey and rides that champion's glide,
 * and a mark that shows what comes next (Order Thread and its underlay, Step
 * Ghost and its disc and ink) carries data-landing-wait with the keys it
 * touches (returnKey, and the thread's targetKey) and fades in once those
 * champions have landed (choreography.js). Neither shows at a landing spot
 * before its champion arrives. A record without returnKey gets {}. Pure.
 *
 * @param {LookRecord} record One hotbar record.
 * @returns {Record<string, string>}
 */
function glideTiming(record) {
  if (typeof record.returnKey !== "string" || record.returnKey === "") return {};
  if (!LANDING_WAIT_KINDS.has(record.kind)) {
    return { "data-glide-follow": record.returnKey };
  }
  const keys = [record.returnKey, record.targetKey].filter(
    (key) => typeof key === "string" && key !== "",
  );
  return { "data-landing-wait": [...new Set(keys)].join(" ") };
}

/**
 * Return the beat-hold attributes that keep a mark hidden until its champion
 * is set down on its waystone.
 *
 * record.returnKey is the presentation key of the champion the mark belongs
 * to (the ringed or looked-at champion, the studs' owner, the body a Reach
 * Mark sits on, or your champion for its Order Thread and Step Ghost). The
 * mark gets data-beat-hold "set-down" on that key, side "after", a 1 ms cut:
 * the choreographer (choreography.js beatHoldSpecs) hides it until that
 * champion's set-down time when the champion returns this tick, and adds no
 * hold at all when it does not (a key with no Return beat in the plan). So a
 * returning champion's Command Ring, studs and plan marks never show at the
 * waystone before it does. A cut, not a fade, because each mark's opacity
 * lives in `styles/hotbar.css`. A record without returnKey gets {}. Pure.
 *
 * @param {LookRecord} record One hotbar record.
 * @returns {Record<string, string>}
 */
function returnHold(record) {
  if (typeof record.returnKey !== "string" || record.returnKey === "") return {};
  return {
    "data-beat-hold": "set-down",
    "data-beat-key": record.returnKey,
    "data-beat-side": "after",
    "data-beat-fade": "1",
  };
}

/**
 * Return the SVG tag and attributes of one record's mark (no beat hold).
 *
 * Classes carry the paint (`styles/hotbar.css`); `data-hotbar` names the mark
 * for tests and tools ("ring", "look", "studs", "reach", "thread-ink",
 * "thread", "ghost-disc", "ghost-ink", "ghost"). The Look Ring also carries
 * `selection-moonlight`, the class the live page's look ring has always had.
 * An `order-studs` record with `shape: "dot"` becomes one `<circle>` (cx, cy,
 * r from its x, y, r); a diamond record becomes one `<path>` of four.
 *
 * @param {LookRecord} record One hotbar record.
 * @returns {{tag: string, attributes: Record<string, unknown>} | null} Null
 *   for an unknown kind.
 */
function markSpec(record) {
  if (record.kind === "command-ring") {
    return {
      tag: "circle",
      attributes: {
        class: `${record.alive ? "hotbar-ring" : "hotbar-ring hotbar-ring--fallen"}${record.away ? " hotbar-ring--away" : ""}`,
        "data-hotbar": "ring",
        "data-identity-key": record.identityKey,
        cx: record.cx,
        cy: record.cy,
        r: record.r,
      },
    };
  }
  if (record.kind === "look-ring") {
    return {
      tag: "circle",
      attributes: {
        class: "hotbar-look selection-moonlight",
        "data-hotbar": "look",
        "data-identity-key": record.identityKey,
        cx: record.cx,
        cy: record.cy,
        r: record.r,
      },
    };
  }
  if (record.kind === "order-studs") {
    const attributes = {
      class: `hotbar-studs hotbar-studs--${record.style} hotbar-studs--${record.classKey}${record.away ? " hotbar-studs--away" : ""}`,
      "data-hotbar": "studs",
      "data-style": record.style,
    };
    return record.shape === "dot"
      ? {
          tag: "circle",
          attributes: { ...attributes, cx: record.x, cy: record.y, r: record.r },
        }
      : { tag: "path", attributes: { ...attributes, d: record.d } };
  }
  if (record.kind === "order-thread" || record.kind === "order-thread-ink") {
    const ink = record.kind === "order-thread-ink";
    const base = ink ? "hotbar-thread-ink" : "hotbar-thread";
    return {
      tag: "line",
      attributes: {
        class: record.ultimate ? `${base} ${base}--ultimate` : base,
        "data-hotbar": ink ? "thread-ink" : "thread",
        x1: record.x1,
        y1: record.y1,
        x2: record.x2,
        y2: record.y2,
      },
    };
  }
  if (record.kind === "reach-mark") {
    return {
      tag: "circle",
      attributes: {
        class: "hotbar-reach",
        "data-hotbar": "reach",
        "data-identity-key": record.identityKey,
        cx: record.cx,
        cy: record.cy,
        r: record.r,
      },
    };
  }
  if (
    record.kind === "step-ghost" ||
    record.kind === "step-ghost-ink" ||
    record.kind === "step-ghost-disc"
  ) {
    const part = record.kind === "step-ghost" ? "ghost" : record.kind.slice(5);
    return {
      tag: "circle",
      attributes: {
        class: record.kind === "step-ghost" ? "hotbar-ghost" : `hotbar-${part}`,
        "data-hotbar": part,
        "data-move-action": record.moveAction,
        cx: record.cx,
        cy: record.cy,
        r: record.r,
      },
    };
  }
  return null;
}

/**
 * Build one export-safe SVG node per record (circle, path or line with flat
 * strokes and fills set by `styles/hotbar.css`). Records of an unknown kind
 * are skipped. Nodes are detached; the caller places them. Every node is
 * `pointer-events: none` through its CSS class, so clicks reach the body under
 * it.
 *
 * @param {ReadonlyArray<LookRecord>} records Records from `records`.
 * @param {Document} doc The document that creates the nodes.
 * @returns {SVGElement[]} New nodes in record order (empty for no records).
 */
export function build(records, doc) {
  /** @type {SVGElement[]} */
  const nodes = [];
  for (const record of records) {
    const spec = nodeSpec(record);
    if (spec !== null) nodes.push(svgNode(doc, spec.tag, spec.attributes));
  }
  return nodes;
}

/**
 * Move existing nodes to new records after a resize or camera change. nodes
 * and records are parallel arrays from one `build` call; a node whose record
 * has a different SVG tag or kind is left alone (the caller should rebuild).
 * Changes DOM attributes only.
 *
 * @param {ReadonlyArray<Element>} nodes Nodes from `build`.
 * @param {ReadonlyArray<LookRecord>} records New records for the same marks.
 * @returns {void}
 */
export function update(nodes, records) {
  for (let index = 0; index < Math.min(nodes.length, records.length); index += 1) {
    const spec = nodeSpec(records[index]);
    if (spec !== null && nodes[index].localName === spec.tag) {
      setNodeAttributes(nodes[index], spec.attributes);
    }
  }
}

/** Plain explanations for each hotbar mark, keyed by record kind. */
const RING_HELP = Object.freeze({
  title: "Command Ring",
  summary:
    "The silver ring marks the champion in your hands: your next orders are for it.",
});
const THREAD_HELP = Object.freeze({
  title: "Order Thread",
  summary: "The dashed line joins your champion to the target of its armed action.",
});
const GHOST_HELP = Object.freeze({
  title: "Step Ghost",
  summary:
    "The dotted circle shows where your champion ends its step if nothing blocks it.",
});
const MARK_HELP = Object.freeze({
  "command-ring": RING_HELP,
  "look-ring": Object.freeze({
    title: "Look Ring",
    summary:
      "The soft silver ring marks the champion you are looking at. A policy plays it; you cannot order it.",
  }),
  "order-studs": Object.freeze({
    title: "Order Studs",
    summary:
      "Four marks on the ring show the armed action: silver dots for No Combat, dots for the Basic, diamonds for the Ultimate. Gray hollow marks mean the game does not allow it now.",
  }),
  "order-thread": THREAD_HELP,
  "order-thread-ink": THREAD_HELP,
  "reach-mark": Object.freeze({
    title: "Reach Mark",
    summary: "The game allows the armed action on this body right now.",
  }),
  "step-ghost": GHOST_HELP,
  "step-ghost-ink": GHOST_HELP,
  "step-ghost-disc": GHOST_HELP,
});

/**
 * Return the hover card for one hotbar record, or null for any other record.
 * The card names the mark and says what it means; it names no champion, so it
 * is the same in Oracle View and Agent POV.
 *
 * @param {LookRecord} record A record from `records`.
 * @returns {Readonly<Record<string, any>> | null} A legality tooltip descriptor.
 */
export function tooltip(record) {
  const help = isRecord(record)
    ? MARK_HELP[/** @type {keyof typeof MARK_HELP} */ (record.kind)]
    : undefined;
  if (help === undefined) return null;
  return Object.freeze({
    kind: "legality",
    id: `hotbar:${record.kind}`,
    title: help.title,
    tone: "neutral",
    accent: "none",
    summary: help.summary,
    rows: Object.freeze([]),
    sections: Object.freeze([]),
    metadata: Object.freeze({ compact: true, full: true }),
    anchor: "element",
  });
}
