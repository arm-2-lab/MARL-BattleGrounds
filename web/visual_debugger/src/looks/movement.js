/**
 * @file The Movement look (Scholar's Step) for all three clients: the marks a
 * champion's move leaves on the floor, in the `imprints` layer (under bodies).
 *
 * What it draws, all at the tick's END state (the DOM always holds END):
 * - Start Spot (switch last_move_ghosts, "Start Spots"): a faint solid ring
 *   (opacity 0.28) where a champion began the tick, a 1 px hairline (0.30) from
 *   there to where it ended, and for a Charge a small pin (0.5) where the rush
 *   stopped, with the hairline bent through it. Only for a move of 0.25 world
 *   units or more. A leg that cuts into stone is dotted (it is a Dim Shift).
 * - Phantom Afterimage (switch phantom_afterimage, needs Body Glide): a smoke
 *   copy of a darting Rogue's token at its start spot. It is drawn at opacity 0
 *   (paused frames never show it) and only an animation makes it visible.
 * - Focus Trail (switch focus_trail): eight fading dots along the selected
 *   champion's path of this tick only, start to end. There is no history of
 *   earlier ticks (plan P17).
 *
 * The glide itself is not drawn here: the choreographer moves each body's
 * `g.agent-glide` wrapper with a temporary transform. This file gives the
 * choreographer three helpers: `cutsStone` (does a straight move pass through
 * stone, so the body should dim and shift instead of gliding),
 * `dimShiftSpecs` (the Dim Shift animation of one body wrapper) and
 * `animationSpecs` (how this look's marks fade in during a played tick or a
 * one-tick step). Every animation uses fill "backwards" only, so cancelling it
 * shows END. No timers and no CSS animations.
 *
 * Callers: `scene.js` calls `records`, `build` and `update` for the `imprints`
 * layer, with `movementSource(events, audience)` (the audience's own event
 * inventory) for the trajectories and returns;
 * `choreography.js` calls `animationSpecs`, `cutsStone` and `dimShiftSpecs`.
 * `visual-filters.js` reads `PAINT_PARTS`.
 *
 * Agent POV boundary: records read only the audience's own arena scene and the
 * audience's own events (Oracle View: latest_events; Agent POV: visual_events).
 * A champion the team cannot see at the end of the tick, or whose start spot
 * was not disclosed, gets no Start Spot, no afterimage and no trail. Agent POV
 * trajectories never carry a post-Charge spot, so no Charge pin is drawn there.
 * Nothing here reads the researcher view or the match summary.
 */
import { canonicalAgentIdentity } from "../agent-identity.js";
import { formatDisplayNumber } from "../display.js";
import { createSemanticDescriptor, registerTooltipOwner } from "../tooltip.js";
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
 * @typedef {import("./look-support.js").AnimationSpec} AnimationSpec
 * @typedef {import("./look-support.js").ScreenPoint} ScreenPoint
 * @typedef {readonly [number, number]} WorldPoint
 */

/**
 * The store's Scholar's Step numbers (`skins/movement-and-flow.js`).
 * - ghostMin: a move shorter than this (world units) leaves no Start Spot.
 * - dartMin: a Rogue move at least this long (world units) leaves an afterimage.
 * - cutMin: a straight leg whose body cuts this deep into stone (world units) is
 *   a Dim Shift.
 * - bodyRadius: the body radius used by the stone test and by rings when a
 *   champion has no radius (world units).
 * - ringScale: the Start Spot ring radius as a share of the body's radius.
 * - ringOpacity, hairlineOpacity, pinOpacity: END opacities (0.35 drawn at 0.8
 *   ink for the solid ring gives 0.28).
 * - pinRadius: the Charge pin radius in CSS pixels.
 * - minHairline: a hairline leg shorter than this many CSS pixels is skipped.
 * - trailDots: Focus Trail dot count; trailWorldRadius its newest dot's radius
 *   in world units (never under trailMinRadius CSS pixels), shrinking by
 *   trailRadiusDecay and fading by trailOpacityDecay per older dot from
 *   trailOpacity.
 * - waymarkRingFrom, waymarkHairlineFrom, waymarkMs: with Body Glide off, a
 *   played tick's ring and hairline arrive brighter and settle over 400 ms.
 * - markFadeMs: reduced motion and the played glide fade marks in over 120 ms.
 * - dimShare: a Dim Shift dims for this share of its leg, then brightens for
 *   the rest; dimOpacity is the body's lowest opacity.
 * - smoke: the afterimage's fill and rim hold and fade times (real ms at 1x,
 *   counted from the start of the glide).
 * - fallFadeMs: how long a fallen champion's trail takes to fade at its Fall.
 */
export const MOVEMENT = Object.freeze({
  ghostMin: 0.25,
  dartMin: 0.95,
  cutMin: 0.15,
  bodyRadius: 0.5,
  ringScale: 0.93,
  ringOpacity: 0.28,
  hairlineOpacity: 0.3,
  pinOpacity: 0.5,
  pinRadius: 1.2,
  minHairline: 1,
  trailDots: 8,
  trailWorldRadius: 0.12,
  trailMinRadius: 1.5,
  trailRadiusDecay: 0.88,
  trailOpacity: 0.5,
  trailOpacityDecay: 0.8,
  waymarkRingFrom: 0.48,
  waymarkHairlineFrom: 0.5,
  waymarkMs: 400,
  markFadeMs: 120,
  dimShare: 0.4,
  dimOpacity: 0.3,
  smoke: Object.freeze({ fillHold: 130, fillEnd: 290, rimHold: 180, rimEnd: 340 }),
  fallFadeMs: 300,
});

/** The Rogue's class ID: only a Rogue leaves a Phantom Afterimage. */
const ROGUE_CLASS_ID = 4;

/** Marks sit under the bodies and never enter the collision allocator. */
export const FOOTPRINT = Object.freeze({});

/**
 * Per record: a Start Spot has at most a group, two hairline legs, a pin, a
 * ring and a hover circle (6 nodes, 4 animations); an afterimage a group and
 * three circles (4 nodes, 3 animations); a Focus Trail two groups and eight
 * dots (10 nodes, 1 animation). Durable: nothing is kept after a render.
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 10,
  animationsPerRecord: 4,
  persistentNodesPerRecord: 0,
});

/** The switch for each mark this look draws. The Dim Shift belongs to Body Glide. */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "start_spot" }, "last_move_ghosts"],
  [{ surface: "durable", kind: "phantom_afterimage" }, "phantom_afterimage"],
  [{ surface: "durable", kind: "focus_trail" }, "focus_trail"],
  [{ surface: "durable", kind: "dim_shift" }, "stride_glide"],
]);

/**
 * Read one movement switch from a filter state.
 *
 * filters is the page's visual filter state or null; id is a switch ID. A
 * missing state or a missing key uses the switch default, which is on for every
 * switch this look reads. Only an exact false turns a switch off. Pure.
 *
 * @param {Readonly<Record<string, boolean>> | null | undefined} filters
 * @param {string} id
 * @returns {boolean}
 */
function switchOn(filters, id) {
  return !isRecord(filters) || filters[id] !== false;
}

/**
 * Return a world point [x, y] from a value, or null.
 *
 * value should be an array of two finite numbers; anything else gives null.
 * Pure.
 *
 * @param {unknown} value
 * @returns {WorldPoint | null}
 */
function worldPoint(value) {
  if (!Array.isArray(value) || value.length < 2) return null;
  const x = Number(value[0]);
  const y = Number(value[1]);
  return Number.isFinite(x) && Number.isFinite(y) ? Object.freeze([x, y]) : null;
}

/**
 * Return the distance between two world points, in world units. Pure.
 *
 * @param {WorldPoint} left
 * @param {WorldPoint} right
 */
function distance(left, right) {
  return Math.hypot(right[0] - left[0], right[1] - left[1]);
}

/**
 * Return how far a point lies outside one stone (negative inside), in world units.
 *
 * obstacle is a scene stone: {kind: "pillar", center, radius} or {kind: "wall",
 * center, width, height, theta} (theta in radians). Any other row returns
 * +Infinity (never a cut). Pure.
 *
 * @param {WorldPoint} point
 * @param {Record<string, any>} obstacle
 * @returns {number}
 */
function signedStoneDistance(point, obstacle) {
  const center = worldPoint(obstacle.center);
  if (center === null) return Number.POSITIVE_INFINITY;
  const dx = point[0] - center[0];
  const dy = point[1] - center[1];
  if (obstacle.kind === "pillar") {
    return Math.hypot(dx, dy) - finiteNumber(obstacle.radius);
  }
  if (obstacle.kind !== "wall") return Number.POSITIVE_INFINITY;
  const theta = finiteNumber(obstacle.theta);
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const localX = cos * dx + sin * dy;
  const localY = -sin * dx + cos * dy;
  const qx = Math.abs(localX) - finiteNumber(obstacle.width) / 2;
  const qy = Math.abs(localY) - finiteNumber(obstacle.height) / 2;
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0);
}

/**
 * Say whether a body moving straight from one spot to another passes through stone.
 *
 * This is a display choice only (glide or Dim Shift), never a game rule check:
 * the game moved the body around or through stone by its own rules, and the
 * browser only avoids drawing a glide that crosses stone. from and to are world
 * points [x, y]; obstacles is the scene's map.obstacles list (pillars and walls;
 * other rows are ignored); bodyRadius is the body's radius in world units
 * (default 0.5). The straight leg is sampled every 0.04 units (at least 8
 * steps). Return true when the body cuts at least 0.15 units into any stone.
 * Return false for a leg shorter than a millionth of a unit, bad points or no
 * stones. Pure. Agent POV: stones are public map facts, so this reads nothing
 * hidden.
 *
 * @param {unknown} from
 * @param {unknown} to
 * @param {unknown} obstacles
 * @param {number} [bodyRadius]
 * @returns {boolean}
 */
export function cutsStone(from, to, obstacles, bodyRadius = MOVEMENT.bodyRadius) {
  const start = worldPoint(from);
  const end = worldPoint(to);
  const stones = asArray(obstacles).filter(isRecord);
  if (start === null || end === null || stones.length === 0) return false;
  const length = distance(start, end);
  if (length < 1e-6) return false;
  const steps = Math.max(8, Math.ceil(length / 0.04));
  for (let index = 0; index <= steps; index += 1) {
    const share = index / steps;
    const point = /** @type {WorldPoint} */ ([
      start[0] + (end[0] - start[0]) * share,
      start[1] + (end[1] - start[1]) * share,
    ]);
    for (const stone of stones) {
      if (bodyRadius - signedStoneDistance(point, stone) >= MOVEMENT.cutMin) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Collect this look's frame inputs from the audience's own event inventory.
 *
 * events is that inventory, chosen by the caller from the normalized frame:
 * Oracle View passes frame.latest_events, Agent POV passes frame.visual_events
 * (never the other one). audience is "researcher" or "agent_pov"; in Agent POV
 * the inventory must be the fog-filtered one (summary_kind
 * "agent_pov_fog_filtered_visual_events"), otherwise nothing is returned.
 * Return a frozen {trajectories, returned}: trajectories is the inventory's
 * agent_phase_trajectories list (empty when missing), and returned lists the
 * presentation keys from the agent_anchor of each event whose event_kind is
 * "agent_respawned" (a return to a pad is not a move). A non-object gives empty
 * lists. Pure: no DOM; it reads only what it is given, so in Agent POV an unseen
 * return is simply not listed.
 *
 * @param {unknown} events
 * @param {"researcher" | "agent_pov" | null | undefined} audience
 * @returns {Readonly<{trajectories: ReadonlyArray<unknown>, returned: ReadonlyArray<string>}>}
 */
export function movementSource(events, audience) {
  const empty = Object.freeze({
    trajectories: Object.freeze([]),
    returned: Object.freeze([]),
  });
  if (!isRecord(events)) return empty;
  if (
    audience === "agent_pov" &&
    events.summary_kind !== "agent_pov_fog_filtered_visual_events"
  ) {
    return empty;
  }
  const returned = new Set();
  for (const event of asArray(events.events)) {
    if (!isRecord(event) || event.event_kind !== "agent_respawned") continue;
    const key = isRecord(event.agent_anchor)
      ? event.agent_anchor.presentation_key
      : null;
    if (typeof key === "string" && key.length > 0) returned.add(key);
  }
  return Object.freeze({
    trajectories: Object.freeze([...asArray(events.agent_phase_trajectories)]),
    returned: Object.freeze([...returned]),
  });
}

/**
 * Turn a scene team ID into the CSS team letter.
 *
 * 1 is Team A ("a"), 2 is Team B ("b"); anything else is null (neutral marks).
 *
 * @param {unknown} teamId
 * @returns {"a" | "b" | null}
 */
function teamLetter(teamId) {
  return teamId === 1 ? "a" : teamId === 2 ? "b" : null;
}

/**
 * Return this look's drawing records for one render.
 *
 * input (see LookInput) reads:
 * - scene: the audience's authorized arena scene. Bodies come from
 *   scene.agents (presentation_key, public_agent_id, class_id, team_id,
 *   position, radius in world units, life_state), the stones from
 *   scene.map.obstacles and the Focus Trail's champion from
 *   scene.selection.selected_presentation_key.
 * - transform: this render's projection (camera included). Without a scene or
 *   transform the result is empty.
 * - audience: "researcher" (Oracle View) or "agent_pov"; defaults to
 *   scene.audience. In Agent POV a trajectory's post_charge is ignored.
 * - filters: the visual filter state (default: every switch on).
 *   last_move_ghosts, phantom_afterimage, focus_trail and stride_glide are
 *   read; an Afterimage also needs stride_glide (Body Glide).
 * - trajectories: the audience's own agent_phase_trajectories (from
 *   movementSource). Each row joins its successor to a scene body by
 *   presentation key; a row whose body is not in the scene, whose successor
 *   is missing or differs from the body's position, or whose transition_start
 *   is missing draws nothing.
 * - returned: presentation keys of champions that returned to a pad this tick
 *   (from movementSource); they draw nothing.
 *
 * Records, all positions and radii in CSS pixels, frozen, in scene order:
 * - {kind: "start_spot", presentationKey, publicAgentId, label, team, center,
 *   radius, legs: [{from, to, cut, leg: "rush" | "stride"}], pin, charged, cut,
 *   path, startWorld, endWorld, opacity: {ring, hairline, pin}}: only with
 *   last_move_ghosts on and a move of at least 0.25 world units. pin is the
 *   post-Charge spot or null; cut says the last leg passes through stone (a
 *   Dim Shift; its leg is dotted). path is the world length of the move.
 * - {kind: "phantom_afterimage", presentationKey, center, radius}: a Rogue
 *   that moved at least 0.95 world units, with phantom_afterimage and
 *   stride_glide on.
 * - {kind: "focus_trail", presentationKey, team, fallen, dots: [{x, y, r,
 *   opacity}]}: the selected champion's eight dots, oldest (at the start spot)
 *   first, with focus_trail on and a move of at least 0.25 world units.
 *   fallen says the champion fell this tick (its trail is hidden at END and
 *   fades at its Fall beat in play).
 * Pure data: no DOM, inputs unchanged. Agent POV: only bodies in the POV scene
 * with a disclosed start spot ever give a record.
 *
 * @param {LookInput} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const scene = isRecord(input?.scene) ? input.scene : null;
  const transform = input?.transform ?? null;
  if (scene === null || !transform) return Object.freeze([]);
  const audience = input.audience ?? scene.audience;
  const filters = input.filters ?? null;
  const ghosts = switchOn(filters, "last_move_ghosts");
  const afterimages =
    switchOn(filters, "phantom_afterimage") && switchOn(filters, "stride_glide");
  const trail = switchOn(filters, "focus_trail");
  const returned = new Set(asArray(input.returned));
  const trajectoryByKey = new Map();
  for (const row of asArray(input.trajectories)) {
    if (isRecord(row) && typeof row.agent_presentation_key === "string") {
      trajectoryByKey.set(row.agent_presentation_key, row);
    }
  }
  const obstacles = isRecord(scene.map) ? scene.map.obstacles : null;
  const selectedKey = isRecord(scene.selection)
    ? scene.selection.selected_presentation_key
    : null;
  /** @type {LookRecord[]} */
  const result = [];
  for (const agent of asArray(scene.agents)) {
    if (!isRecord(agent) || typeof agent.presentation_key !== "string") continue;
    const key = agent.presentation_key;
    const trajectory = trajectoryByKey.get(key);
    if (!trajectory || returned.has(key)) continue;
    const end = worldPoint(agent.position);
    const successor = worldPoint(trajectory.successor?.position);
    const start = worldPoint(trajectory.transition_start?.position);
    if (end === null || successor === null || start === null) continue;
    if (distance(end, successor) > 1e-6) continue;
    const rawPost =
      audience === "agent_pov" ? null : worldPoint(trajectory.post_charge?.position);
    const post = rawPost !== null && distance(start, rawPost) > 1e-4 ? rawPost : null;
    const legStart = post ?? start;
    const path = (post === null ? 0 : distance(start, post)) + distance(legStart, end);
    if (path < MOVEMENT.ghostMin) continue;
    const worldRadius = finiteNumber(agent.radius, MOVEMENT.bodyRadius);
    const bodyRadius = transform.worldLengthToScreen(
      worldRadius > 0 ? worldRadius : MOVEMENT.bodyRadius,
    );
    const team = teamLetter(agent.team_id);
    const center = screenPoint(start, transform);
    const cut = cutsStone(legStart, end, obstacles, MOVEMENT.bodyRadius);
    if (ghosts && path >= MOVEMENT.ghostMin) {
      /** @type {Array<Readonly<{from: ScreenPoint, to: ScreenPoint, cut: boolean, leg: string}>>} */
      const legs = [];
      if (post !== null) {
        legs.push(hairlineLeg(start, post, transform, false, "rush"));
      }
      legs.push(hairlineLeg(legStart, end, transform, cut, "stride"));
      result.push(
        Object.freeze({
          kind: "start_spot",
          presentationKey: key,
          publicAgentId: String(agent.public_agent_id ?? ""),
          label: canonicalAgentIdentity(agent).title,
          team,
          center,
          radius: bodyRadius * MOVEMENT.ringScale,
          legs: Object.freeze(
            legs.filter(
              (leg) =>
                Math.hypot(leg.to.x - leg.from.x, leg.to.y - leg.from.y) >=
                MOVEMENT.minHairline,
            ),
          ),
          pin: post === null ? null : screenPoint(post, transform),
          charged: post !== null,
          cut,
          path,
          startWorld: start,
          endWorld: end,
          opacity: Object.freeze({
            ring: MOVEMENT.ringOpacity,
            hairline: MOVEMENT.hairlineOpacity,
            pin: MOVEMENT.pinOpacity,
          }),
        }),
      );
    }
    if (afterimages && agent.class_id === ROGUE_CLASS_ID && path >= MOVEMENT.dartMin) {
      result.push(
        Object.freeze({
          kind: "phantom_afterimage",
          presentationKey: key,
          center,
          radius: bodyRadius,
        }),
      );
    }
    if (trail && key === selectedKey && path >= MOVEMENT.ghostMin) {
      result.push(
        Object.freeze({
          kind: "focus_trail",
          presentationKey: key,
          team,
          fallen: agent.life_state !== undefined && agent.life_state !== "alive",
          dots: trailDots(post === null ? [start, end] : [start, post, end], transform),
        }),
      );
    }
  }
  return Object.freeze(result);
}

/**
 * Project one hairline leg into screen points.
 *
 * from and to are world points; cut marks a leg through stone (drawn dotted);
 * leg names it ("rush" for the Charge's first leg, else "stride"). Pure.
 *
 * @param {WorldPoint} from
 * @param {WorldPoint} to
 * @param {LookTransform} transform
 * @param {boolean} cut
 * @param {"rush" | "stride"} leg
 */
function hairlineLeg(from, to, transform, cut, leg) {
  return Object.freeze({
    from: screenPoint(from, transform),
    to: screenPoint(to, transform),
    cut,
    leg,
  });
}

/**
 * Place the Focus Trail's eight dots along this tick's path.
 *
 * points is the world path (start, the post-Charge spot when there is one,
 * end). Dot k (k = 0 to 7) sits k/8 of the way along the path by length, so the
 * first dot is on the start spot and none is on the end spot (the body is
 * there). Its age i = 7 - k sets its look: radius max(1.5 px, 0.12 world units
 * x 0.88^i) and opacity 0.5 x 0.8^i, so the newest dot is the biggest and
 * brightest. Return a frozen array of frozen {x, y, r, opacity}, oldest first,
 * in CSS pixels. Pure.
 *
 * @param {ReadonlyArray<WorldPoint>} points
 * @param {LookTransform} transform
 */
function trailDots(points, transform) {
  const lengths = points.slice(1).map((point, index) => distance(points[index], point));
  const total = lengths.reduce((sum, length) => sum + length, 0);
  const count = MOVEMENT.trailDots;
  const baseRadius = transform.worldLengthToScreen(MOVEMENT.trailWorldRadius);
  const dots = [];
  for (let k = 0; k < count; k += 1) {
    let along = (total * k) / count;
    let index = 0;
    while (index < lengths.length - 1 && along > lengths[index]) {
      along -= lengths[index];
      index += 1;
    }
    const share = lengths[index] > 0 ? Math.min(1, along / lengths[index]) : 0;
    const from = points[index];
    const to = points[index + 1];
    const screen = screenPoint(
      [from[0] + (to[0] - from[0]) * share, from[1] + (to[1] - from[1]) * share],
      transform,
    );
    const age = count - 1 - k;
    dots.push(
      Object.freeze({
        x: screen.x,
        y: screen.y,
        r: Math.max(
          MOVEMENT.trailMinRadius,
          baseRadius * MOVEMENT.trailRadiusDecay ** age,
        ),
        opacity: MOVEMENT.trailOpacity * MOVEMENT.trailOpacityDecay ** age,
      }),
    );
  }
  return Object.freeze(dots);
}

/**
 * Build export-safe SVG nodes for records, one top node per record, in order.
 *
 * records come from records(); doc creates the nodes (it is not read when
 * records is empty). Every node is flat fill or stroke with opacity attributes:
 * no gradients, masks, filters, url() or images. Colours come from CSS classes
 * and the data-team attribute (styles/movement.css). Structure:
 * - start_spot: g.movement-start-spot[data-movement-kind="start_spot"] holding
 *   its hairline legs (line.movement-hairline, data-leg, data-cut, and
 *   pathLength 1 on solid legs only so a dotted stone leg keeps its dots),
 *   its pin (circle.movement-pin), the ring
 *   (circle.movement-start-ring) and a paintless hover circle
 *   (circle.movement-start-hit) that owns the "Start Spot" tooltip.
 * - phantom_afterimage: g.movement-afterimage holding three circles at
 *   opacity 0 (fill, core ring, rim); only an animation shows them.
 * - focus_trail: g.movement-focus-trail-hold (for a champion that fell this
 *   tick it carries the Fall beat hold: data-beat-hold "fall", side "before",
 *   END opacity 0) holding g.movement-focus-trail and its dots.
 * Each top node carries data-movement-kind and data-presentation-key, which
 * animationSpecs reads. Return the new detached nodes. Side effects: registers
 * tooltip owners through tooltip.js. Agent POV: draws only what records gave.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @returns {SVGElement[]}
 */
export function build(records, doc) {
  if (records.length === 0) return [];
  /** @type {SVGElement[]} */
  const nodes = [];
  for (const record of records) {
    const group = svgNode(doc, "g", {
      "data-movement-kind": record.kind,
      "data-presentation-key": record.presentationKey,
    });
    if (record.kind === "start_spot") {
      setNodeAttributes(group, {
        class: "movement-start-spot",
        "data-team": record.team,
      });
      for (const leg of record.legs) {
        group.append(
          svgNode(doc, "line", {
            class: "movement-hairline",
            "data-leg": leg.leg,
            "data-cut": leg.cut ? "true" : "false",
            // A solid leg unrolls in play by its dash, measured as a share of
            // its length; a dotted stone leg keeps real dash lengths.
            pathLength: leg.cut ? null : 1,
            opacity: record.opacity.hairline,
          }),
        );
      }
      if (record.pin !== null) {
        group.append(
          svgNode(doc, "circle", {
            class: "movement-pin",
            r: MOVEMENT.pinRadius,
            opacity: record.opacity.pin,
          }),
        );
      }
      group.append(
        svgNode(doc, "circle", {
          class: "movement-start-ring",
          opacity: record.opacity.ring,
        }),
        svgNode(doc, "circle", { class: "movement-start-hit" }),
      );
      const descriptor = tooltip(record);
      if (descriptor !== null) {
        registerTooltipOwner(
          /** @type {Element} */ (group.lastElementChild),
          descriptor,
        );
      }
    } else if (record.kind === "phantom_afterimage") {
      setNodeAttributes(group, { class: "movement-afterimage" });
      group.append(
        svgNode(doc, "circle", {
          class: "movement-afterimage-fill",
          "data-afterimage": "fill",
          opacity: 0,
        }),
        svgNode(doc, "circle", {
          class: "movement-afterimage-core",
          "data-afterimage": "core",
          opacity: 0,
        }),
        svgNode(doc, "circle", {
          class: "movement-afterimage-rim",
          "data-afterimage": "rim",
          opacity: 0,
        }),
      );
    } else if (record.kind === "focus_trail") {
      setNodeAttributes(group, {
        class: "movement-focus-trail-hold",
        "data-team": record.team,
        ...(record.fallen
          ? {
              "data-beat-hold": "fall",
              "data-beat-key": record.presentationKey,
              "data-beat-side": "before",
              "data-beat-from": 1,
              "data-beat-fade": MOVEMENT.fallFadeMs,
              opacity: 0,
            }
          : {}),
      });
      const inner = svgNode(doc, "g", { class: "movement-focus-trail" });
      for (const dot of record.dots) {
        inner.append(
          svgNode(doc, "circle", { class: "movement-trail-dot", opacity: dot.opacity }),
        );
      }
      group.append(inner);
    } else {
      continue;
    }
    place(group, record);
    nodes.push(group);
  }
  return nodes;
}

/**
 * Write one record's positions and sizes onto the nodes build() made for it.
 *
 * node is the record's top group; record its (new) record of the same kind and
 * shape. Children are matched in build order. Return undefined. Changes only
 * position and size attributes (DOM side effect).
 *
 * @param {Element} node
 * @param {LookRecord} record
 */
function place(node, record) {
  if (record.kind === "start_spot") {
    const lines = node.querySelectorAll("line.movement-hairline");
    for (const [index, leg] of record.legs.entries()) {
      const line = lines[index];
      if (!line) break;
      setNodeAttributes(line, {
        x1: leg.from.x,
        y1: leg.from.y,
        x2: leg.to.x,
        y2: leg.to.y,
      });
    }
    const pin = node.querySelector("circle.movement-pin");
    if (pin && record.pin !== null) {
      setNodeAttributes(pin, { cx: record.pin.x, cy: record.pin.y });
    }
    for (const circle of node.querySelectorAll(
      "circle.movement-start-ring, circle.movement-start-hit",
    )) {
      setNodeAttributes(circle, {
        cx: record.center.x,
        cy: record.center.y,
        r: record.radius,
      });
    }
  } else if (record.kind === "phantom_afterimage") {
    for (const circle of node.querySelectorAll("circle")) {
      const part = circle.getAttribute("data-afterimage");
      setNodeAttributes(circle, {
        cx: record.center.x,
        cy: record.center.y,
        r: record.radius * (part === "core" ? 0.56 : MOVEMENT.ringScale),
      });
    }
  } else if (record.kind === "focus_trail") {
    const dots = node.querySelectorAll("circle.movement-trail-dot");
    for (const [index, dot] of record.dots.entries()) {
      const circle = dots[index];
      if (!circle) break;
      setNodeAttributes(circle, { cx: dot.x, cy: dot.y, r: dot.r });
    }
  }
}

/**
 * Move existing nodes to new records after a resize or camera change.
 *
 * nodes are the top nodes build() returned; records the new records for the
 * same frame, in the same order and with the same kinds (a new frame needs a
 * new build). Each node gets its record's positions and sizes; a node whose
 * record is missing or of another kind is left alone, and extra records are
 * ignored. Return undefined. DOM side effect only; tooltips and opacities stay.
 *
 * @param {ReadonlyArray<Element>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  for (const [index, node] of nodes.entries()) {
    const record = records[index];
    if (record === undefined) break;
    if (node.getAttribute("data-movement-kind") !== record.kind) continue;
    place(node, record);
  }
}

/**
 * Return the "Start Spot" tooltip descriptor for a start_spot record.
 *
 * record is a records() result. A start_spot gives a tooltip.js semantic
 * descriptor: title "Start Spot", a plain summary, and rows Champion (for
 * example "Mage · ID 5"), Started At and Ended At (world x, y) and Moved (world
 * units, "1 unit" when the shown number is 1, else "N units"). Any other kind (afterimages and trails have no tooltip) returns null.
 * Pure. Agent POV: the record exists only for a champion the team saw at both
 * ends of the tick, so nothing hidden is named.
 *
 * @param {LookRecord} record
 * @returns {import("../tooltip.js").AnyTooltipDescriptor | null}
 */
export function tooltip(record) {
  if (!isRecord(record) || record.kind !== "start_spot") return null;
  const both = Object.freeze({ compact: true, full: true });
  const spot = (/** @type {WorldPoint} */ point) =>
    `${formatDisplayNumber(point[0])}, ${formatDisplayNumber(point[1])}`;
  return createSemanticDescriptor({
    kind: "movement",
    id: `start-spot:${record.presentationKey}`,
    title: "Start Spot",
    tone: "neutral",
    accent: "none",
    summary: record.cut
      ? "Where this champion stood when the tick began. The game judged its blows from here. Its path passed through stone, so it dims and shifts instead of gliding."
      : "Where this champion stood when the tick began. The game judged its blows from here; the body shows where it ended.",
    rows: [
      { label: "Champion", value: record.label, metadata: both },
      { label: "Started At", value: spot(record.startWorld), metadata: both },
      { label: "Ended At", value: spot(record.endWorld), metadata: both },
      {
        label: "Moved",
        // The word follows the shown, rounded number: "1 unit", "0.5 units".
        value: `${formatDisplayNumber(record.path)} ${formatDisplayNumber(record.path) === "1" ? "unit" : "units"}`,
        metadata: both,
      },
    ],
    sections: [],
    metadata: both,
    anchor: "pointer",
  });
}

/**
 * Read a node's END opacity from its opacity attribute (default 1). Pure.
 *
 * @param {Element} element
 */
function endOpacity(element) {
  const value = Number(element.getAttribute("opacity"));
  return element.hasAttribute("opacity") && Number.isFinite(value) ? value : 1;
}

/**
 * Describe how this look's marks enter during a played tick or a one-tick step.
 *
 * root is the element holding this look's nodes (the `imprints` layer or the
 * battlefield SVG); only nodes with data-movement-kind are read. options:
 * - epochKey: the plan's epoch key, for ids `mbg:<epochKey>:movement:<n>`.
 * - mode: "played" (a tick playing on the beat clock) or "step" (a one-tick
 *   replay_step glide).
 * - motionMode: "normal", "reduced" or "off" ("off" returns []).
 * - beats: the plan's beat schedule (plan.beats). Played ticks read
 *   beats.glide (Body Glide on) and beats.windows.glide and
 *   beats.windows.chargeRush (real ms); missing windows use 0-260 and 0-110.
 * - stepMs: the step glide length in ms (BEAT_CLOCK.stepGlide, 240), for
 *   mode "step".
 * Rules (store Scholar's Step numbers):
 * - Played, normal motion, Body Glide on: rings fade in over 0-120 ms; solid
 *   hairline legs unroll with the body (a Charge's rush leg during the rush,
 *   the stride leg after it, to the glide's end); a dotted (stone) leg appears
 *   when the Dim Shift cuts over (0.4 of its leg); a Charge pin appears when
 *   the rush ends; the Focus Trail fades in over 0-120 ms; a Rogue's
 *   afterimage fill holds 0.7 until 130 ms and fades by 290 ms, its core ring
 *   (0.56) likewise, its rim holds 0.8 until 180 ms and fades by 340 ms.
 * - Played, normal motion, Body Glide off: bodies snap, so rings arrive at
 *   0.48 and hairlines at 0.5 and settle to END over 400 ms; no afterimage.
 * - Reduced motion (played or step): marks fade in over 120 ms; no afterimage.
 * - Step, normal motion: marks fade in over the step glide (stepMs); no
 *   afterimage.
 * Every spec uses fill "backwards" and no forward fill, so ending or
 * cancelling it shows the END look the DOM holds. Return a new array of
 * AnimationSpec records; nothing is started (the choreographer starts and
 * counts them). Reads the DOM only. Agent POV: only nodes built from the
 * audience's own records exist, so nothing hidden is animated.
 *
 * @param {ParentNode | null | undefined} root
 * @param {{
 *   epochKey: string,
 *   mode: "played" | "step",
 *   motionMode: "normal" | "reduced" | "off",
 *   beats?: Record<string, any> | null,
 *   stepMs?: number,
 * }} options
 * @returns {AnimationSpec[]}
 */
export function animationSpecs(root, options) {
  if (!root || typeof root.querySelectorAll !== "function") return [];
  const { epochKey, mode, motionMode } = options;
  if (motionMode === "off") return [];
  const beats = isRecord(options.beats) ? options.beats : null;
  const glideWindow = asArray(beats?.windows?.glide);
  const rushWindow = asArray(beats?.windows?.chargeRush);
  const glideEnd = finiteNumber(glideWindow[1], 260);
  const rushEnd = finiteNumber(rushWindow[1], 110);
  const stepMs = finiteNumber(options.stepMs, 240);
  const played = mode === "played";
  const glideOn = beats?.glide !== false;
  /** @type {AnimationSpec[]} */
  const specs = [];
  const push = (
    /** @type {Element} */ element,
    /** @type {Keyframe[]} */ keyframes,
    /** @type {KeyframeAnimationOptions} */ timing,
  ) => {
    specs.push({
      element,
      keyframes,
      options: { ...timing, fill: "backwards" },
      id: `mbg:${epochKey}:movement:${specs.length}`,
    });
  };
  const fadeIn = (/** @type {Element} */ element, /** @type {number} */ duration) =>
    push(element, [{ opacity: 0 }, { opacity: endOpacity(element) }], {
      delay: 0,
      duration,
    });
  const simpleFade =
    motionMode === "reduced" ? MOVEMENT.markFadeMs : !played ? stepMs : null;
  for (const node of root.querySelectorAll("[data-movement-kind]")) {
    const kind = node.getAttribute("data-movement-kind");
    if (kind === "start_spot") {
      const marks = [...node.querySelectorAll(".movement-hairline, .movement-pin")];
      const ring = node.querySelector(".movement-start-ring");
      if (simpleFade !== null) {
        for (const mark of [...marks, ring]) if (mark) fadeIn(mark, simpleFade);
        continue;
      }
      if (!glideOn) {
        if (ring) {
          push(
            ring,
            [
              { opacity: MOVEMENT.waymarkRingFrom, easing: "ease-out" },
              { opacity: endOpacity(ring) },
            ],
            {
              delay: 0,
              duration: MOVEMENT.waymarkMs,
            },
          );
        }
        for (const line of node.querySelectorAll(".movement-hairline")) {
          push(
            line,
            [
              { opacity: MOVEMENT.waymarkHairlineFrom, easing: "ease-out" },
              { opacity: endOpacity(line) },
            ],
            { delay: 0, duration: MOVEMENT.waymarkMs },
          );
        }
        continue;
      }
      if (ring) fadeIn(ring, MOVEMENT.markFadeMs);
      const charged = node.querySelector(".movement-pin") !== null;
      const strideStart = charged ? rushEnd : 0;
      for (const line of node.querySelectorAll(".movement-hairline")) {
        const rush = line.getAttribute("data-leg") === "rush";
        if (line.getAttribute("data-cut") === "true") {
          const hidden = strideStart + MOVEMENT.dimShare * (glideEnd - strideStart);
          push(line, [{ opacity: 0 }, { opacity: 0 }], { delay: 0, duration: hidden });
          continue;
        }
        push(line, [{ strokeDasharray: "0 1" }, { strokeDasharray: "1 0" }], {
          delay: rush ? 0 : strideStart,
          duration: Math.max(1, rush ? rushEnd : glideEnd - strideStart),
        });
      }
      const pin = node.querySelector(".movement-pin");
      if (pin)
        push(pin, [{ opacity: 0 }, { opacity: 0 }], { delay: 0, duration: rushEnd });
    } else if (kind === "focus_trail") {
      const inner = node.querySelector(".movement-focus-trail");
      if (inner) fadeIn(inner, simpleFade ?? MOVEMENT.markFadeMs);
    } else if (kind === "phantom_afterimage") {
      if (!played || motionMode !== "normal" || !glideOn) continue;
      const { fillHold, fillEnd, rimHold, rimEnd } = MOVEMENT.smoke;
      for (const circle of node.querySelectorAll("[data-afterimage]")) {
        const part = circle.getAttribute("data-afterimage");
        const rim = part === "rim";
        const peak = rim ? 0.8 : part === "core" ? 0.56 : 0.7;
        const hold = rim ? rimHold : fillHold;
        const end = rim ? rimEnd : fillEnd;
        push(
          circle,
          [
            { opacity: peak, offset: 0 },
            { opacity: peak, offset: hold / end, easing: "ease-in-out" },
            { opacity: 0, offset: 1 },
          ],
          { delay: 0, duration: end },
        );
      }
    }
  }
  return specs;
}

/**
 * Describe the Dim Shift of one body: it dims, cuts across the stone, then brightens.
 *
 * Used instead of a glide when the body's last leg passes through stone
 * (cutsStone). element is the body's g.agent-glide wrapper; points are its
 * world spots in order ([start, end], or [start, post-Charge, end] for a
 * Charge whose rush still glides first); offsets are the matching shares of
 * duration (the same offsets the glide would use, first 0 and last 1);
 * worldToScreen projects world points to CSS pixels; duration is the glide
 * length in ms (260 played, 240 for a step); idPrefix starts both ids
 * (`<idPrefix>:dim-shift:move` and `:fade`). The body holds at the start of
 * its last leg while its opacity falls from 1 to 0.3 over 0.4 of that leg,
 * then it jumps to END (the DOM's own place) and its opacity returns to 1 by
 * the end. The move is a translate added on top of the wrapper's own transform
 * (composite "add"), like a glide. Both specs use fill "backwards" only, so
 * END shows once they end or are cancelled. The body never vanishes. Return
 * [] when a point does not project to a finite screen spot. Pure.
 *
 * @param {Element} element
 * @param {ReadonlyArray<WorldPoint>} points
 * @param {ReadonlyArray<number>} offsets
 * @param {(point: WorldPoint) => ScreenPoint} worldToScreen
 * @param {number} duration
 * @param {string} idPrefix
 * @returns {AnimationSpec[]}
 */
export function dimShiftSpecs(
  element,
  points,
  offsets,
  worldToScreen,
  duration,
  idPrefix,
) {
  if (points.length < 2 || offsets.length !== points.length || !(duration > 0)) {
    return [];
  }
  const end = worldToScreen(points[points.length - 1]);
  /** @type {string[]} */
  const shifts = [];
  for (const world of points.slice(0, -1)) {
    const screen = worldToScreen(world);
    const dx = Math.round((screen.x - end.x) * 1000) / 1000;
    const dy = Math.round((screen.y - end.y) * 1000) / 1000;
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) return [];
    shifts.push(`translate(${dx + 0}px, ${dy + 0}px)`);
  }
  const legStart = offsets[offsets.length - 2] * duration;
  const cutAt = legStart + MOVEMENT.dimShare * (duration - legStart);
  /** @type {Keyframe[]} */
  const moveFrames = shifts.map((transform, index) => ({
    offset: cutAt > 0 ? Math.min(1, (offsets[index] * duration) / cutAt) : 0,
    transform,
  }));
  moveFrames.push({ offset: 1, transform: shifts[shifts.length - 1] });
  return [
    {
      element,
      keyframes: moveFrames,
      options: {
        delay: 0,
        duration: Math.max(1, cutAt),
        fill: "backwards",
        composite: "add",
      },
      id: `${idPrefix}:dim-shift:move`,
    },
    {
      element,
      keyframes: [
        { offset: 0, opacity: 1 },
        { offset: legStart / duration, opacity: 1 },
        { offset: cutAt / duration, opacity: MOVEMENT.dimOpacity },
        { offset: 1, opacity: 1 },
      ],
      options: { delay: 0, duration, fill: "backwards" },
      id: `${idPrefix}:dim-shift:fade`,
    },
  ];
}
