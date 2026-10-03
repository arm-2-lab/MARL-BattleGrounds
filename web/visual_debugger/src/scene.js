/**
 * @file Render durable, already-authorized battlefield facts into retained SVG layers.
 *
 * BattlefieldRenderer owns the arena SVG: one camera wrapper group
 * (`g.battlefield-camera`) holding the fixed layer stack BATTLEFIELD_LAYER_ORDER,
 * bottom to top. Each render projects the authorized arena scene through one
 * viewport transform (with the camera frame; Whole Map by default) and asks the
 * look modules under `src/looks/` for their drawing: records(input) gives plain
 * data, build(records, doc) gives export-safe SVG nodes, and this renderer puts
 * them in their layers. Durable looks mounted here: map skins (with tale
 * stones over the Standing Stones), atmosphere, Red Zone, spawn waystones,
 * Homeland Brackets, aura Ward Circles (with their played-tick moments, Aura
 * Glints and the Barrier Held stud flash), movement marks, reach rings and the
 * selection moonlight, the next-action thread (or the live hotbar), the Agent POV
 * sight veil, obstacles, Death Dimming, champions, corpse marks, Rune Zones and
 * the recipient rim (both inside each body's glide wrapper) and the hotbar
 * (in the body layer's plan-marks group, under every body). It
 * keeps the retained body nodes (each body inside an animatable `g.agent-glide`
 * wrapper for the choreographer's glides), lays out the status, modifier and
 * lane-pill docks with the shared allocator (`layout.js`; the Rune Ring replaced
 * the old Ultimate cooldown docks) and publishes a narrow choreographySurface
 * (with the herald layer, this render's transform, switch state and arena scene
 * for the event looks).
 *
 * Smooth play: a repaint changes only what changed. Each layer keeps every
 * node that is equal to its new drawing (replaceLayer), and a body already in
 * its place is never removed and re-inserted (placeGroupsInOrder); a body
 * that falls or returns moves once, straight between the living and dead
 * groups, and no other body moves with it. So the running glide and beat
 * animations of a tick survive any repaint and nothing is torn down between
 * ticks.
 *
 * Live play (`product_kind` "combat_debugger", the BattleClient and the
 * DevClient combat screen) follows the approved vision page: the field has
 * the vision's 16/1032 pad, the reach rings are Basic and Ultimate only and
 * shown only with both Ranges (G) and the Reach Rings switch on, the target
 * gets the vision's corner brackets (champion-frame bracketPath "vision"), and
 * the layers stack as the vision draws them (see #arrangeLiveLayers). Its
 * drawing is sized by the field's exact box (liveFieldSize, so one drawing
 * unit is one CSS pixel), and the battlefield carries `--mark-scale`
 * (setMarkScale, hotbar.markScale), which grows the plan, reach, veil-edge and
 * bracket line widths with the field as the vision's lines grow. The Replay
 * Viewer keeps its whole-pixel size, 24 px pad, three reach rings and layer
 * order, and has no `--mark-scale`.
 *
 * The transient controller owns the children it installs
 * in the `strike-band`, `transient-events` and `heralds` layers; the Frame
 * Explanation controller owns the `explain` layer.
 *
 * Boundary: bodies, rings, docks and every tooltip over the arena come only
 * from the authorized arena scene of the current audience and that audience's
 * own events (Oracle View: latest_events; Agent POV: visual_events, the POV
 * trajectories and the authorized sensor sources). In Agent POV that is the POV
 * scene alone: this file never reads the global researcher space or the match
 * summary's rows (the adapter scopes the double-scoring fall IDs it uses), so
 * an aura or status tooltip can name only a champion the team can see, and a POV
 * status card never borrows a researcher row's source. The browser draws
 * supplied masks and events; it does not decide legality, advance games or
 * fetch data.
 */
import {
  authorizedPresentationCurrentTick,
  authorizedPresentationEpisodeLimit,
  authorizedPresentationIncomingRows,
  authorizedPresentationInspectionState,
  authorizedPresentationMatchBeatFacts,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import { beatView } from "./choreography-plan.js";
import { formatCompactDisplayNumber, formatDisplayNumber } from "./display.js";
import { explainPovAgent } from "./explanations.js";
import { createSvgIcon } from "./icons.js";
import {
  createViewportTransform,
  layoutRequiredDocks,
  layoutStatusDocks,
} from "./layout.js";
import * as atmosphere from "./looks/atmosphere.js";
import * as championFrame from "./looks/champion-frame.js";
import * as champions from "./looks/champions.js";
import * as fog from "./looks/fog.js";
import * as hotbar from "./looks/hotbar.js";
import {
  agentIdentity,
  asArray,
  finiteNumber,
  isRecord,
  screenPoint,
  setNodeAttributes,
  svgNode,
} from "./looks/look-support.js";
import * as mapSkins from "./looks/map-skins.js";
import * as movement from "./looks/movement.js";
import * as obstacles from "./looks/obstacles.js";
import * as redZone from "./looks/red-zone.js";
import * as requiems from "./looks/requiems.js";
import * as spawns from "./looks/spawns.js";
import * as statuses from "./looks/statuses.js";
import * as teamIdentity from "./looks/team-identity.js";
import { registerTooltipOwner } from "./tooltip.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualPaintPartEnabled,
} from "./visual-filters.js";
import { classTokenFromId, resolveVisualToken, teamTokenFromId } from "./vocabulary.js";

// Dock cell sizes in CSS pixels. The looks that draw the docks own them; this
// renderer reads them so its dock allocator reserves exactly what is drawn.
const STATUS_DOCK_DIMENSIONS = statuses.FOOTPRINT.statusDock;
const COOLDOWN_DOCK_DIMENSIONS = statuses.FOOTPRINT.cooldownDock;
const MODIFIER_DOCK_DIMENSIONS = statuses.FOOTPRINT.modifierDock;

/**
 * Return the durable paint-part tag a look module registers for kind.
 *
 * look is a look module; kind is a durable part kind such as "aura_field". Each
 * look owns its PAINT_PARTS, so this renderer asks the same tag the filters use.
 * Throw TypeError when the look does not register that durable kind.
 *
 * @param {{PAINT_PARTS: ReadonlyArray<{part: Readonly<Record<string, string>>, filter: string}>}} look
 * @param {string} kind
 * @returns {Readonly<Record<string, string>>}
 */
function durablePaintPart(look, kind) {
  const registration = look.PAINT_PARTS.find(
    ({ part }) => part.surface === "durable" && part.kind === kind,
  );
  if (registration === undefined) {
    throw new TypeError(`No look registers the durable paint part ${kind}.`);
  }
  return registration.part;
}

// The durable paint-part tags this renderer asks the visual filters about, each
// read from the look that registers it.
const DURABLE_VISUAL_PAINT_PARTS = Object.freeze({
  auraFields: durablePaintPart(statuses, "aura_field"),
  auraModifierBadges: durablePaintPart(statuses, "aura_modifier_badge"),
  durationStatusBadges: durablePaintPart(statuses, "duration_status_badge"),
  povDurationStatusBadges: durablePaintPart(statuses, "pov_duration_status_badge"),
  spawnShield: durablePaintPart(champions, "spawn_shield"),
  cooldownBadges: durablePaintPart(statuses, "cooldown_badge"),
  selectionReticle: durablePaintPart(championFrame, "selection_reticle"),
  selectedPairLegality: durablePaintPart(championFrame, "selected_pair_legality"),
  redZoneFloors: durablePaintPart(redZone, "red_zone_floor"),
});

/**
 * The arena layers, bottom to top, inside the camera wrapper group:
 * map (floor and tale skin), atmosphere (Moon Motes and the Last Watch Veil),
 * red-zone, spawns, team-identity (brackets outside the floor), aura, imprints
 * (where bodies started), debug-range (selection moonlight, then reach rings),
 * pending-route (next action), fog (Agent POV sight veil, under stones),
 * obstacle, death-dim (Death Dimming over the floor and stones), strike-band
 * (activation routes: above stones, under bodies), body (selection brackets,
 * corpses, corpse marks, then living bodies), over (rings and marks on bodies),
 * selection-legality, transient-events, durable-status-modifier (docks),
 * heralds (Herald Cards, above every body), explain (Frame Explanation marks)
 * and accessible-labels.
 */
export const BATTLEFIELD_LAYER_ORDER = Object.freeze([
  "map",
  "atmosphere",
  "red-zone",
  "spawns",
  "team-identity",
  "aura",
  "imprints",
  "debug-range",
  "pending-route",
  "fog",
  "obstacle",
  "death-dim",
  "strike-band",
  "body",
  "over",
  "selection-legality",
  "transient-events",
  "durable-status-modifier",
  "heralds",
  "explain",
  "accessible-labels",
]);

/** Class of the one camera wrapper group that holds every arena layer. */
export const BATTLEFIELD_CAMERA_CLASS = "battlefield-camera";

/**
 * @typedef {Record<string, any>} JsonRecord
 */
/**
 * @typedef {import("./looks/champions.js").ChampionNodes &
 *   import("./looks/champion-frame.js").SelectionNodes} AgentNodes
 */

/**
 * @typedef {ReturnType<typeof createViewportTransform>} ViewportTransform
 * @typedef {import("./layout.js").CameraFrame} CameraFrame
 * @typedef {{
 *   layer: SVGElement,
 *   routeLayer: SVGElement,
 *   heraldLayer: SVGElement,
 *   battlefield: SVGSVGElement,
 *   ownerDocument: Document,
 *   viewportKey: string,
 *   viewportBounds: Rectangle,
 *   protectedRects: ReadonlyArray<ProtectedRegion>,
 *   worldToScreen: (
 *     point: {x: number, y: number} | readonly [number, number],
 *   ) => {x: number, y: number},
 *   worldLengthToScreen: (length: number) => number,
 *   lookTransform: ViewportTransform,
 *   visualFilters: Readonly<Record<string, boolean>>,
 *   scene: JsonRecord | null,
 * }} ChoreographySurface
 * @typedef {{
 *   agent: JsonRecord,
 *   identityKey: number | string,
 *   layoutSlot: number,
 *   globalSlot: number | null,
 *   presentationKey: string | null,
 *   center: {x: number, y: number},
 *   radius: number,
 *   controlled: boolean,
 *   selected: boolean,
 *   statuses: any[],
 * }} ProjectedAgent
 * @typedef {{
 *   left: number,
 *   top: number,
 *   right: number,
 *   bottom: number,
 *   width: number,
 *   height: number,
 * }} Rectangle
 * @typedef {Rectangle & {
 *   layoutKey: string,
 *   bounds: Rectangle,
 *   protectedKind: "body" | "status" | "cooldown" | "modifier" | "legality",
 *   ownerPresentationKey: string | null,
 * }} ProtectedRegion
 * @typedef {{
 *   showAuraFields: boolean,
 *   showAuraModifierBadges: boolean,
 *   showDurationStatusBadges: boolean,
 *   showPovDurationStatusBadges: boolean,
 *   showSpawnShield: boolean,
 *   showCooldownBadges: boolean,
 *   showSelectionReticle: boolean,
 *   showSelectedPairLegality: boolean,
 *   showRedZoneFloors: boolean,
 * }} DurableVisualPolicy
 * @typedef {import("./looks/statuses.js").DockTooltipInput} DockTooltipInput
 */

/**
 * Package a durable drawing region with an explicit semantic owner.
 *
 * protectedKind names body/status/cooldown/modifier/legality; ownerIdentity gives
 * its stable local key; bounds is a pixel rectangle; ownerPresentationKey may be
 * null for legacy slot ownership. placementIdentity defaults to ownerIdentity and
 * separates multiple docks belonging to one body. Return a frozen region with a
 * copied frozen bounds record and compatibility top-level coordinates. No geometry
 * or ownership is inferred from proximity.
 *
 * @param {"body" | "status" | "cooldown" | "modifier" | "legality"} protectedKind
 * @param {string} ownerIdentity
 * @param {Rectangle} bounds
 * @param {string | null} ownerPresentationKey
 * @param {string} [placementIdentity]
 * @returns {ProtectedRegion}
 */
function choreographyProtectedRegion(
  protectedKind,
  ownerIdentity,
  bounds,
  ownerPresentationKey,
  placementIdentity = ownerIdentity,
) {
  const frozenBounds = Object.freeze({ ...bounds });
  return Object.freeze({
    layoutKey: JSON.stringify([
      "durable",
      protectedKind,
      ownerIdentity,
      placementIdentity,
    ]),
    bounds: frozenBounds,
    ...frozenBounds,
    protectedKind,
    ownerPresentationKey,
  });
}

/**
 * Resolve local filter state into the nine durable paint switches.
 *
 * state is validated/classified by the visual-filter authority. Return a frozen
 * policy for aura, badges, shield, selection, pair-legality and Red Zone floor
 * rendering. No accepted scene facts or input filter object are changed.
 *
 * @param {unknown} state
 * @returns {Readonly<DurableVisualPolicy>}
 */
function durableVisualPolicy(state) {
  return Object.freeze({
    showAuraFields: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.auraFields,
    ),
    showAuraModifierBadges: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.auraModifierBadges,
    ),
    showDurationStatusBadges: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.durationStatusBadges,
    ),
    showPovDurationStatusBadges: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.povDurationStatusBadges,
    ),
    showSpawnShield: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.spawnShield,
    ),
    showCooldownBadges: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.cooldownBadges,
    ),
    showSelectionReticle: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.selectionReticle,
    ),
    showSelectedPairLegality: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.selectedPairLegality,
    ),
    showRedZoneFloors: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.redZoneFloors,
    ),
  });
}

/**
 * Read an own enumerable data property without invoking an accessor.
 *
 * value may be unknown and key names the property. Return its stored value, or
 * undefined for missing/nonenumerable/accessor properties, nonobjects or a thrown
 * property-descriptor trap. Proxy traps may run while obtaining the descriptor;
 * errors are caught. This helper performs no mutation.
 *
 * @param {unknown} value
 * @param {string} key
 * @returns {unknown}
 */
function ownEnumerableDataValue(value, key) {
  try {
    if (typeof value !== "object" || value === null) return undefined;
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (
      descriptor === undefined ||
      !descriptor.enumerable ||
      !Object.hasOwn(descriptor, "value")
    ) {
      return undefined;
    }
    return descriptor.value;
  } catch {
    return undefined;
  }
}

/**
 * Read the authorized display scene from a recognized presentation frame.
 *
 * frame may be unknown; localInspectedPresentationKey defaults to undefined and is
 * forwarded for local inspection selection. Return the adapter view or null for an
 * unrecognized frame. The adapter owns allowed inspection and disclosure rules.
 *
 * @param {unknown} frame
 * @param {string | null | undefined} [localInspectedPresentationKey]
 * @returns {JsonRecord | null}
 */
function frameScene(frame, localInspectedPresentationKey = undefined) {
  return isAuthorizedPresentationFrame(frame)
    ? authorizedPresentationSceneView(frame, localInspectedPresentationKey)
    : null;
}

/**
 * Create a detached named SVG group for the battlefield layer stack.
 *
 * doc creates the group; name sets data-layer; attributes (default {}) are
 * written after it and may override it. Return the group; the constructor
 * decides the order.
 *
 * @param {Document} doc
 * @param {string} name
 * @param {Record<string, unknown>} [attributes]
 * @returns {SVGElement}
 */
function createLayer(doc, name, attributes = {}) {
  return svgNode(doc, "g", {
    "data-layer": name,
    ...attributes,
  });
}

/**
 * Return the live field's exact size in CSS pixels, or null to keep the
 * whole-pixel size.
 *
 * `clientWidth` and `clientHeight` round to whole pixels (553.64 becomes 554),
 * so a viewBox built from them is a hair bigger than the box and the browser
 * shrinks and shifts the whole drawing to fit (0.9994 times, 0.35 px right at
 * 1440 x 900), which moves every line off the vision's pixels. The exact box
 * from `getBoundingClientRect` maps one drawing unit to one CSS pixel. It is
 * used only when the element can report a positive box (a laid-out page);
 * test documents and hidden fields get null.
 *
 * @param {SVGSVGElement} battlefield The arena SVG.
 * @returns {{width: number, height: number} | null} The box size in CSS px,
 *   or null when the element has no box to read.
 */
function liveFieldSize(battlefield) {
  if (typeof battlefield.getBoundingClientRect !== "function") return null;
  const box = battlefield.getBoundingClientRect();
  const width = finiteNumber(box?.width);
  const height = finiteNumber(box?.height);
  return width > 0 && height > 0 ? { width, height } : null;
}

/**
 * Set or clear the live field's `--mark-scale` CSS variable, which
 * `styles/hotbar.css` and `styles/battleclient/arena.css` multiply every plan
 * and reach line's stroke width and dash length by (the vision's lines on its
 * 50 px per unit drawing, grown with the field). It writes only when the value
 * changes, so an ordinary repaint makes no style mutation (the smooth-play
 * check counts mutations between ticks). Uses the CSS object model, which the
 * page's style policy allows. Side effect: changes the element's inline style.
 *
 * @param {SVGSVGElement} battlefield The arena SVG.
 * @param {number | null} scale The scale (from hotbar.markScale), or null to
 *   remove the variable (Replay Viewer, or no scene).
 * @returns {void}
 */
function setMarkScale(battlefield, scale) {
  const style = /** @type {CSSStyleDeclaration | undefined} */ (
    /** @type {{style?: CSSStyleDeclaration}} */ (battlefield).style
  );
  if (style === undefined || typeof style.getPropertyValue !== "function") return;
  const next = scale === null ? "" : String(scale);
  if (style.getPropertyValue("--mark-scale") === next) return;
  if (next === "") style.removeProperty("--mark-scale");
  else style.setProperty("--mark-scale", next);
}

/**
 * Return whether a freshly built node carries a hover card or keyboard focus.
 *
 * A look registers its tooltip on the node it builds (`registerTooltipOwner`),
 * and the registration cannot be copied to another node. Every such owner in
 * the looks carries an `aria-label` or a `tabindex`, so this test finds them.
 *
 * @param {Element} node A node from a look's build().
 * @returns {boolean} True when the node or a child is a tooltip or focus owner.
 */
function carriesTooltipOwner(node) {
  return (
    node.hasAttribute("aria-label") ||
    node.hasAttribute("tabindex") ||
    node.querySelector("[aria-label], [tabindex]") !== null
  );
}

/**
 * Write the look nodes for one layer, keeping every drawn node that did not
 * change.
 *
 * layer is the target group; nodes are the look's build() result for this
 * render, in paint order. Each old child that is equal to the new node at the
 * same place (`isEqualNode`: same tag, attributes and children) stays in the
 * DOM, so a repaint of the same tick (a deck click, a phase change, a poll)
 * or an unchanged stone, pad or floor on a new tick never tears the layer
 * down. That keeps running CSS and Web Animations on those nodes, and it is
 * the smooth-play rule: between ticks only what changed is replaced. A node
 * that differs, or that carries a hover card or focus (its registration lives
 * on the new node only), replaces the old one in place; extra old nodes are
 * removed and extra new ones appended. Return undefined. DOM change only.
 *
 * @param {Element} layer The layer group (`g[data-layer]` or a body group).
 * @param {ReadonlyArray<Element>} nodes The new nodes, bottom to top.
 */
function replaceLayer(layer, nodes) {
  reconcileNodes(layer, [...layer.children], nodes);
}

/**
 * Turn the drawn nodes old into nodes inside parent, keeping unchanged ones.
 *
 * old are parent's current nodes for this look, in paint order; nodes are the
 * new ones. The rules are replaceLayer's: an equal node without a hover card
 * stays, a different one replaces the old node in place, extra old nodes are
 * removed and extra new ones appended to parent. Return undefined. DOM change
 * only.
 *
 * @param {Element} parent The group that holds the nodes.
 * @param {ReadonlyArray<Element>} old The current nodes, bottom to top.
 * @param {ReadonlyArray<Element>} nodes The new nodes, bottom to top.
 */
function reconcileNodes(parent, old, nodes) {
  if (old.length === 0) {
    if (nodes.length > 0) parent.append(...nodes);
    return;
  }
  for (const [index, node] of nodes.entries()) {
    const current = old[index];
    if (current === undefined) {
      parent.append(node);
    } else if (
      current !== node &&
      (carriesTooltipOwner(node) || !current.isEqualNode(node))
    ) {
      current.replaceWith(node);
    }
  }
  for (const extra of old.slice(nodes.length)) extra.remove();
}

/**
 * Put nodes into a parent in the given order, moving only what is out of
 * place.
 *
 * Appending a node that is already in place still removes and re-inserts it,
 * which restarts its CSS animations and shows as a teardown to any observer.
 * This is placeGroupsInOrder for one group; see there for the rules. Return
 * undefined. DOM change only.
 *
 * @param {Element} parent The group that holds the nodes.
 * @param {ReadonlyArray<Element>} nodes The wanted children, first to last.
 */
function placeInOrder(parent, nodes) {
  placeGroupsInOrder([[parent, nodes]]);
}

/**
 * Put nodes into several groups at once, moving each node at most once.
 *
 * groups is a list of [parent, nodes] pairs: nodes are the wanted children
 * of parent, first to last; a node may come from another of the groups (a
 * champion that falls moves from the living group to the dead group). First
 * every group is walked in its wanted order: a child that no group of this
 * parent wants is stepped over, a wanted node already at its place stays,
 * and only a node out of place is moved (insertBefore), straight from where
 * it is to its place. Then, in each parent, the children that are not in its
 * nodes and still sit before a placed node go right after the last placed
 * node, keeping their relative order. So one fall moves one body, not every
 * body after it, and a smooth glide keeps all other nodes (an observer sees
 * one removal for the one move). Return undefined. DOM change only; no game
 * value is read.
 *
 * @param {ReadonlyArray<readonly [Element, ReadonlyArray<Element>]>} groups
 *   The [parent, wanted children] pairs, in any order.
 */
function placeGroupsInOrder(groups) {
  for (const [parent, nodes] of groups) {
    const wanted = new Set(nodes);
    let cursor = parent.firstElementChild;
    for (const node of nodes) {
      while (cursor !== null && cursor !== node && !wanted.has(cursor)) {
        cursor = cursor.nextElementSibling;
      }
      if (cursor === node) {
        cursor = node.nextElementSibling;
      } else {
        parent.insertBefore(node, cursor);
      }
    }
  }
  for (const [parent, nodes] of groups) {
    const last = nodes.at(-1);
    if (last === undefined || last.parentNode !== parent) continue;
    const wanted = new Set(nodes);
    /** @type {Element[]} */
    const before = [];
    for (let child = parent.firstElementChild; child !== null && child !== last; ) {
      const next = child.nextElementSibling;
      if (!wanted.has(child)) before.push(child);
      child = next;
    }
    if (before.length > 0) last.after(...before);
  }
}

/**
 * Return the arena padding, in CSS pixels, for one render.
 *
 * The BattleClient and the DevClient combat screen (`product_kind`
 * "combat_debugger") draw the field as the approved vision page does: its
 * field drawing is 1032 px wide with a 16 px pad (50 px per world unit on a
 * 20 by 10 map), so the pad is 16/1032 of the arena width and every body,
 * mark and stone lands at the vision's place at any window size. The Replay
 * Viewer keeps its fixed 24 px pad.
 *
 * @param {unknown} frame The installed presentation (its product_kind is read).
 * @param {number} viewportWidth The arena width in CSS pixels.
 * @returns {number} The pad on every side, in CSS pixels.
 */
function arenaPadding(frame, viewportWidth) {
  return isRecord(frame) && frame.product_kind === "combat_debugger"
    ? (viewportWidth * 16) / 1032
    : 24;
}

/**
 * Return the raw event rows inside one audience's incoming summary record.
 *
 * Rune Zones (looks/statuses.js) reads these rows to learn what happened to
 * each body this tick: a status applied, refreshed, aged out or broken by
 * damage (the Freezing Trap's Shatter), and an Ultimate made ready or spent.
 * It needs the rows, not the summary record around them.
 *
 * @param {unknown} summary This audience's own incoming summary, as
 *   audienceSources gives it: Oracle View `latest_events` or Agent POV
 *   `visual_events` (a record whose `events` array holds the rows), or null on
 *   a frame with no transition into it.
 * @returns {ReadonlyArray<unknown>} The summary's `events` array itself (not a
 *   copy), or an empty array when the summary is missing or has no array. Agent
 *   POV: the rows are the POV team's own authorized rows; nothing is added.
 *   Pure: reads the record, writes nothing.
 */
export function incomingEventRows(summary) {
  return isRecord(summary) && Array.isArray(summary.events) ? summary.events : [];
}

/**
 * Gather the audience-scoped inputs the durable looks share for one render.
 *
 * frame is the normalized presentation; scene is its authorized arena scene for
 * the same audience; visualFilters is the page's switch state. Return a frozen
 * record:
 * - events: this audience's own incoming summary record (Oracle View
 *   latest_events, Agent POV visual_events), or null. Its raw event rows sit
 *   in its `events` array; a look that reads rows (Rune Zones) gets that array.
 * - trajectories: Agent POV only, the visual_events agent_phase_trajectories
 *   array (null in Oracle View or when missing); the sight veil reads them.
 * - startSeen(presentationKey): whether a body was seen at the start of the tick
 *   (Oracle View: always true; Agent POV: its POV trajectory has a start spot).
 * - sensorSources: Agent POV only, current_endpoint.parts.authorized_sensor_sources
 *   (null: the recipient alone, as for NoSharedObs).
 * - schedule: the tick's beat schedule from beatView (null when unavailable).
 * - homeFallIds: public IDs of this tick's double-scoring falls this audience saw
 *   (the adapter scopes them; Agent POV: seen falls only).
 * - povTeamId: Agent POV only, the team (1 or 2) of the POV recipient, read from
 *   the POV scene (the recipient always sees itself); otherwise null.
 * - tick: the public tick after this transition; on a frame with no transition
 *   into it, the presentation's own public tick (authorizedPresentationCurrentTick);
 *   null when neither is known.
 * Agent POV boundary: nothing here reads researcher_space or the match summary's
 * hidden rows. Pure: reads the frame, writes nothing.
 *
 * @param {unknown} frame
 * @param {JsonRecord} scene
 * @param {Readonly<Record<string, boolean>>} visualFilters
 */
function audienceSources(frame, scene, visualFilters) {
  const agentPov = scene.audience === "agent_pov";
  const presentation = isRecord(frame) ? frame : {};
  const events = agentPov
    ? isRecord(presentation.visual_events)
      ? presentation.visual_events
      : null
    : isRecord(presentation.latest_events)
      ? presentation.latest_events
      : null;
  const trajectories =
    agentPov && Array.isArray(events?.agent_phase_trajectories)
      ? /** @type {JsonRecord[]} */ (events.agent_phase_trajectories)
      : null;
  const seenAtStart = new Set(
    (trajectories ?? [])
      .filter(
        (trajectory) => isRecord(trajectory) && trajectory.transition_start != null,
      )
      .map((trajectory) => String(trajectory.agent_presentation_key)),
  );
  const parts = isRecord(presentation.current_endpoint)
    ? presentation.current_endpoint.parts
    : null;
  const sensorSources =
    agentPov && isRecord(parts) && Array.isArray(parts.authorized_sensor_sources)
      ? parts.authorized_sensor_sources
      : null;
  const view = beatView(frame, "end", { visualFilters });
  const facts = authorizedPresentationMatchBeatFacts(frame);
  const recipientId = isRecord(presentation.authority)
    ? presentation.authority.recipient_public_agent_id
    : null;
  const recipient = agentPov
    ? asArray(scene.agents).find(
        (agent) => isRecord(agent) && agent.public_agent_id === recipientId,
      )
    : undefined;
  const povTeamId =
    isRecord(recipient) && (recipient.team_id === 1 || recipient.team_id === 2)
      ? recipient.team_id
      : null;
  // The tick after this transition; on a frame with no transition into it (a
  // fresh start or an installed start such as tick 299) the presentation's own
  // public tick, so a still shows the Time Of Night step for its tick.
  const tick = Number.isInteger(events?.incoming_successor_simulator_step_count)
    ? Number(events?.incoming_successor_simulator_step_count)
    : authorizedPresentationCurrentTick(frame);
  return Object.freeze({
    events,
    trajectories,
    /**
     * Say whether the body with presentationKey was seen at the start of the tick.
     *
     * @param {string} presentationKey
     */
    startSeen: (presentationKey) => !agentPov || seenAtStart.has(presentationKey),
    sensorSources,
    schedule: view?.schedule ?? null,
    homeFallIds: facts?.home_fall_public_agent_ids ?? [],
    povTeamId,
    tick,
  });
}

/**
 * Own the durable SVG layers for an authoritative debugger or replay frame.
 *
 * The constructor replaces the supplied SVG's children with one camera wrapper
 * group holding the fixed layer stack (BATTLEFIELD_LAYER_ORDER). Later render
 * calls update durable layers while keeping the SVG root, the camera wrapper
 * and both transient layers for CombatChoreographer. Bodies keep stable node
 * identities where possible, each inside its glide wrapper. The class owns
 * projection, layer placement and dock layout, not simulator rules or fetching;
 * the look modules own how each part is drawn.
 */
export class BattlefieldRenderer {
  /**
   * Install the battlefield layers into caller-supplied browser elements.
   *
   * The elements object requires battlefield (SVGSVGElement) and empty
   * (HTMLElement). Replace battlefield's children with the camera wrapper and
   * its layers, start the retained-node and lookup maps and schedule dock number
   * fitting after battlefield.ownerDocument.fonts.ready. The caller owns the
   * elements and coordinates transient cleanup. A browser DOM and font API are
   * required; unsupported element operations throw.
   *
   * @param {{
   *   battlefield: SVGSVGElement,
   *   empty: HTMLElement,
   * }} elements
   */
  constructor({ battlefield, empty }) {
    this.battlefield = battlefield;
    this.empty = empty;
    /** @type {ViewportTransform | null} */
    this.transform = null;
    /**
     * The switch state and arena scene of the last successful render, which the
     * choreography surface hands to the event looks (same audience as the scene).
     * @type {Readonly<Record<string, boolean>>}
     */
    this.lastVisualFilterState = DEFAULT_VISUAL_FILTER_STATE;
    /** @type {JsonRecord | null} */
    this.lastScene = null;
    /** @type {ReadonlyArray<ProtectedRegion>} */
    this.choreographyProtectedRects = Object.freeze([]);
    /** @type {Readonly<{
     *   base: ReadonlyArray<ProtectedRegion>,
     *   legality: ReadonlyArray<ProtectedRegion>,
     *   status: ReadonlyArray<ProtectedRegion>,
     * }>} */
    this.choreographyProtectedRectGroups = Object.freeze({
      base: Object.freeze([]),
      legality: Object.freeze([]),
      status: Object.freeze([]),
    });
    /** @type {ReadonlyMap<string, JsonRecord>} */
    this.agentByPresentationKey = new Map();
    /** @type {ReadonlyMap<number, JsonRecord>} */
    this.agentByLayoutSlot = new Map();
    /** @type {ReadonlyMap<string, JsonRecord>} */
    this.sceneAgentByPublicId = new Map();

    const doc = battlefield.ownerDocument;
    const map = createLayer(doc, "map", { "aria-hidden": "true" });
    const atmosphereLayer = createLayer(doc, "atmosphere", { "aria-hidden": "true" });
    const redZoneLayer = createLayer(doc, "red-zone", { "aria-hidden": "true" });
    const spawnsLayer = createLayer(doc, "spawns", { "aria-hidden": "true" });
    const teamIdentityLayer = createLayer(doc, "team-identity", {
      "aria-hidden": "true",
    });
    const aura = createLayer(doc, "aura", { "aria-hidden": "true" });
    const imprints = createLayer(doc, "imprints", { "aria-hidden": "true" });
    const debugRange = createLayer(doc, "debug-range", {
      "aria-label": "Authorized ability and effect ranges",
    });
    // Selection moonlight sits first, under the reach rings, stones and bodies.
    this.selectionMoonlight = createLayer(doc, "selection-moonlight", {
      "aria-hidden": "true",
    });
    this.rangeCues = createLayer(doc, "range-cues", { "aria-hidden": "true" });
    debugRange.append(this.selectionMoonlight, this.rangeCues);
    const pendingRoute = createLayer(doc, "pending-route", { "aria-hidden": "true" });
    const fogLayer = createLayer(doc, "fog", { "aria-hidden": "true" });
    const obstacle = createLayer(doc, "obstacle", { "aria-label": "Map obstacles" });
    const deathDim = createLayer(doc, "death-dim", { "aria-hidden": "true" });
    const strikeBand = createLayer(doc, "strike-band", {
      "aria-label": "Authorized combat event routes",
    });
    const body = createLayer(doc, "body", { "aria-label": "Authorized agents" });
    this.bodyLayers = {
      dead: svgNode(doc, "g", { "data-body-state": "dead" }),
      corpseMarks: svgNode(doc, "g", {
        "data-body-state": "corpse-marks",
        "aria-hidden": "true",
      }),
      living: svgNode(doc, "g", { "data-body-state": "living" }),
    };
    // Selection brackets sit under every body, so a neighbour covers them
    // (live play moves them over the bodies; see #arrangeLiveLayers).
    this.selectionCues = createLayer(doc, "selection-cues", {
      "aria-hidden": "true",
    });
    // The live hotbar's plan marks (Command Ring, studs, Reach Marks, Order
    // Thread, Step Ghost) lie under every body, as the vision draws them.
    this.planMarks = createLayer(doc, "plan-marks", { "aria-hidden": "true" });
    body.append(
      this.selectionCues,
      this.planMarks,
      this.bodyLayers.dead,
      this.bodyLayers.corpseMarks,
      this.bodyLayers.living,
    );
    const over = createLayer(doc, "over", { "aria-hidden": "true" });
    const selectionLegality = createLayer(doc, "selection-legality", {
      "aria-label": "Selection and exact actor-owned legality",
    });
    this.legalityCues = createLayer(doc, "legality-cues");
    selectionLegality.append(this.legalityCues);
    const durableStatusModifier = createLayer(doc, "durable-status-modifier", {
      "aria-label": "Durable status, cooldown, and modifier cues",
    });
    const transientEvents = createLayer(doc, "transient-events", {
      "aria-label": "Authorized combat event summaries",
    });
    const heraldsLayer = createLayer(doc, "heralds", {
      "aria-label": "Match moments",
    });
    const explain = createLayer(doc, "explain", {
      "aria-label": "Frame Explanation marks",
    });
    const accessibleLabels = createLayer(doc, "accessible-labels");

    this.layers = {
      map,
      atmosphere: atmosphereLayer,
      redZone: redZoneLayer,
      spawns: spawnsLayer,
      teamIdentity: teamIdentityLayer,
      aura,
      imprints,
      debugRange,
      pendingRoute,
      fog: fogLayer,
      obstacle,
      deathDim,
      strikeBand,
      body,
      over,
      selectionLegality,
      transientEvents,
      durableStatusModifier,
      heralds: heraldsLayer,
      explain,
      accessibleLabels,
    };

    /** @type {Map<number | string, AgentNodes>} */
    this.agentNodes = new Map();
    /**
     * Glide wrapper of each drawn body by presentation key, so marks that ride
     * a body (Rune Zones, the Agent POV recipient rim) join its glide.
     * @type {Map<string, SVGElement>}
     */
    this.wrapperByPresentationKey = new Map();
    /** @type {Map<string, SVGElement>} */
    this.observedBodyNodes = new Map();

    /**
     * The one camera wrapper group. A camera change may ease with a temporary
     * transform on it; its data-camera-frame names the current world frame.
     */
    this.cameraGroup = svgNode(doc, "g", { class: BATTLEFIELD_CAMERA_CLASS });
    this.cameraGroup.append(
      map,
      atmosphereLayer,
      redZoneLayer,
      spawnsLayer,
      teamIdentityLayer,
      aura,
      imprints,
      debugRange,
      pendingRoute,
      fogLayer,
      obstacle,
      deathDim,
      strikeBand,
      body,
      over,
      selectionLegality,
      transientEvents,
      durableStatusModifier,
      heraldsLayer,
      explain,
      accessibleLabels,
    );
    battlefield.replaceChildren(this.cameraGroup);
    void battlefield.ownerDocument.fonts.ready.then(() => {
      statuses.fitDockValues(this.layers.durableStatusModifier);
    });
  }

  /**
   * Paint the durable facts of a normalized authorized frame.
   *
   * frame may be unknown. options.offline defaults to false and changes only the
   * empty-state text; showRanges defaults to true; localInspectedPresentationKey
   * defaults to undefined; visualFilterState defaults to
   * DEFAULT_VISUAL_FILTER_STATE. Optional renderPolicy is copied to SVG metadata
   * without changing game time. options.cameraFrame is the camera's world
   * rectangle {minX, minY, maxX, maxY} (world units); null or absent means the
   * Whole Map. A frame changes only where world points land on screen: every
   * pixel size (docks, cues, strokes) stays the same.
   *
   * Return true after rendering a recognized researcher or agent_pov scene with
   * a positive map size. Otherwise clear durable state, show an unavailable
   * message and return false; transient layers stay controller-owned. Use the
   * element's measured viewport (live play: its exact box, liveFieldSize;
   * otherwise its whole-pixel client size) or a map-based fallback, with 24 px
   * padding (live play: the vision's 16/1032 pad). Live play also sets the
   * battlefield's `--mark-scale` (setMarkScale); other products clear it.
   * Ask each look module for its records and nodes (passing the switch state,
   * the audience, this audience's own events and beat schedule from beatView;
   * see audienceSources), keep the retained body nodes, put the Rune Zones and
   * the recipient rim inside each body's glide wrapper, lay out the status,
   * modifier and lane-pill docks (no Ultimate cooldown docks: the Rune Ring shows
   * the cooldown) and publish protected regions. The `explain` layer is left to
   * the Frame Explanation controller. Bodies, marks and tooltips use only the
   * authorized arena scene of the frame's audience (Agent POV: the POV scene; no
   * researcher row is read). This changes the DOM and renderer caches (the
   * stored switch state and scene the choreography surface hands to the event
   * looks). Layout, filter, camera-frame (TypeError or RangeError for a bad
   * frame) and browser errors propagate; no network request or game command is
   * sent.
   *
   * Agent POV: every look reads the POV arena scene and visual_events, so no
   * unseen champion is drawn and no timing depends on one; the camera frame must
   * come from the same audience's scene.
   *
   * @param {unknown} frame
   * @param {{
   *   offline?: boolean,
   *   showRanges?: boolean,
   *   localInspectedPresentationKey?: string | null,
   *   visualFilterState?: Readonly<Record<string, boolean>>,
   *   renderPolicy?: "live_once" | "replay_animated" | "replay_static",
   *   cameraFrame?: CameraFrame | null,
   *   focusedPublicAgentId?: string | null,
   *   manualSlots?: ReadonlyArray<number> | null,
   * }} [options] focusedPublicAgentId (live Agent POV only, default null) is
   *   the transport frame's focused_public_agent_id: the champion the player
   *   looks at when it is not the one being ordered. The hotbar then draws
   *   Watch Only's Look Ring round its body, only when this audience's scene
   *   draws that body. manualSlots (live only, default null) is the transport
   *   frame's `combat_configuration.manual_slots` (shown slots 0 to 9 of the
   *   Manual champions); the hotbar dims a Command Ring round every other
   *   Manual champion this scene draws. With null, the Manual champions are
   *   the ones the frame's own top-level `pending_joint_action` names (Oracle
   *   View only; Agent POV has no such rows, so no other ring shows).
   * @returns {boolean} Whether the frame contained a paintable scene.
   */
  render(frame, options = {}) {
    const visualFilterState =
      options.visualFilterState === undefined
        ? DEFAULT_VISUAL_FILTER_STATE
        : options.visualFilterState;
    const visualPolicy = durableVisualPolicy(visualFilterState);
    const scene = frameScene(frame, options.localInspectedPresentationKey);
    const map = isRecord(scene?.map) ? scene.map : null;
    const width = finiteNumber(map?.width);
    const height = finiteNumber(map?.height);

    if (
      !scene ||
      !map ||
      (scene.audience !== "researcher" && scene.audience !== "agent_pov") ||
      width <= 0 ||
      height <= 0
    ) {
      this.#clearDurableScene();
      this.battlefield.removeAttribute("viewBox");
      this.battlefield.removeAttribute("data-preset");
      this.battlefield.removeAttribute("data-audience");
      this.battlefield.removeAttribute("data-render-policy");
      setMarkScale(this.battlefield, null);
      this.battlefield.setAttribute(
        "aria-label",
        "Battlefield unavailable; no authorized scene was returned.",
      );
      this.empty.hidden = false;
      this.empty.textContent = options.offline
        ? "The local debugger service is unavailable. Commands are not being retried."
        : "No authorized battlefield scene was returned.";
      this.transform = null;
      this.sceneAgentByPublicId = new Map();
      return false;
    }

    // Live play sizes the drawing by the field's exact box (see liveFieldSize);
    // the Replay Viewer keeps its whole-pixel size.
    const exact =
      isRecord(frame) && frame.product_kind === "combat_debugger"
        ? liveFieldSize(this.battlefield)
        : null;
    const viewportWidth =
      exact !== null
        ? exact.width
        : this.battlefield.clientWidth > 0
          ? this.battlefield.clientWidth
          : Math.max(width * 40, 320);
    const viewportHeight =
      exact !== null
        ? exact.height
        : this.battlefield.clientHeight > 0
          ? this.battlefield.clientHeight
          : Math.max(height * 40, 240);
    const transform = createViewportTransform({
      worldWidth: width,
      worldHeight: height,
      viewportWidth,
      viewportHeight,
      padding: arenaPadding(frame, viewportWidth),
      cameraFrame: options.cameraFrame ?? null,
    });
    this.transform = transform;
    this.battlefield.setAttribute("viewBox", `0 0 ${viewportWidth} ${viewportHeight}`);
    this.battlefield.dataset.preset = "analysis";
    this.battlefield.dataset.audience = scene.audience;
    if (options.renderPolicy === undefined) {
      this.battlefield.removeAttribute("data-render-policy");
    } else {
      this.battlefield.dataset.renderPolicy = options.renderPolicy;
    }
    const cameraFrame = transform.cameraFrame;
    this.cameraGroup.dataset.cameraFrame = [
      cameraFrame.minX,
      cameraFrame.minY,
      cameraFrame.maxX,
      cameraFrame.maxY,
    ].join(" ");
    const audienceLabel = scene.audience === "researcher" ? "Oracle View" : "Agent POV";
    this.battlefield.setAttribute(
      "aria-label",
      `${audienceLabel} battlefield, ${width} by ${height}.`,
    );
    this.empty.hidden = true;

    const doc = this.battlefield.ownerDocument;
    const audience = /** @type {"researcher" | "agent_pov"} */ (scene.audience);
    const sources = audienceSources(frame, scene, visualFilterState);
    // The BattleClient and the DevClient combat screen (see the file notes).
    const liveProduct = isRecord(frame) && frame.product_kind === "combat_debugger";
    if (liveProduct) this.#arrangeLiveLayers();
    setMarkScale(this.battlefield, liveProduct ? hotbar.markScale(transform) : null);
    this.lastVisualFilterState = visualFilterState;
    this.lastScene = scene;
    replaceLayer(
      this.layers.map,
      mapSkins.build(
        mapSkins.records({
          transform,
          map,
          spawnPads: scene.spawn_pads,
          // The technical map name lives only in match_summary, which no arena
          // drawing reads; the skin is found by its exact layout instead.
          mapKey: null,
          filters: visualFilterState,
          audience,
          showUnitGrid: isRecord(frame) && frame.product_kind === "combat_debugger",
        }),
        doc,
      ),
    );
    replaceLayer(
      this.layers.atmosphere,
      atmosphere.build(
        atmosphere.records({
          transform,
          map,
          scene,
          filters: visualFilterState,
          tick: sources.tick,
          // The episode limit is a public constant fixed before the game (the
          // same in both audiences); the adapter reads only that one field, so
          // the Last Watch Veil works in Oracle View and Agent POV alike.
          episodeLimit: authorizedPresentationEpisodeLimit(frame),
        }),
        doc,
      ),
    );
    replaceLayer(
      this.layers.redZone,
      redZone.build(
        redZone.records({
          transform,
          map,
          redZone: isRecord(map?.red_zone)
            ? /** @type {Readonly<{team_a_x_range: number[], team_b_x_range: number[]}>} */ (
                map.red_zone
              )
            : null,
          filters: visualFilterState,
          scene,
          homeFallIds: sources.homeFallIds,
          // Live play draws the BattleClient vision's Grave Ground marks.
          vision: liveProduct,
        }),
        doc,
      ),
    );
    replaceLayer(
      this.layers.spawns,
      spawns.build(
        spawns.records({
          scene,
          transform,
          audience,
          filters: visualFilterState,
          falls: sources.schedule?.falls ?? [],
          returns: sources.schedule?.returns ?? [],
        }),
        doc,
      ),
    );
    replaceLayer(
      this.layers.teamIdentity,
      teamIdentity.build(
        teamIdentity.records({
          scene,
          map,
          transform,
          audience,
          filters: visualFilterState,
          povTeamId: sources.povTeamId,
        }),
        doc,
      ),
    );
    this.agentByPresentationKey = new Map(
      asArray(scene.agents)
        .filter(
          (agent) => isRecord(agent) && typeof agent.presentation_key === "string",
        )
        .map((agent) => [String(agent.presentation_key), agent]),
    );
    this.sceneAgentByPublicId = new Map(
      asArray(scene.agents)
        .filter((agent) => isRecord(agent) && typeof agent.public_agent_id === "string")
        .map((agent) => [String(agent.public_agent_id), agent]),
    );
    replaceLayer(
      this.layers.aura,
      statuses.build(
        statuses.records({
          part: "aura-fields",
          transform,
          agents: asArray(scene.agents),
          fields: visualPolicy.showAuraFields
            ? asArray(scene.aura_fields).filter(
                (field) => isRecord(field) && field.source_alive === true,
              )
            : [],
          tooltipInput: (/** @type {JsonRecord} */ record) =>
            this.#auraTooltipInput(record, audience),
          // The ring moments of a played tick (Aura Glints, the Barrier Held
          // stud flash) come from this audience's own rows and, in Agent POV,
          // its own tick-start spots only.
          audience,
          events: incomingEventRows(sources.events),
          trajectories: sources.trajectories,
          switches: {
            auraGlints: visualFilterState.aura_glints === true,
            barrierHeld: visualFilterState.barrier_held !== false,
          },
        }),
        doc,
      ),
    );
    replaceLayer(
      this.layers.imprints,
      movement.build(
        movement.records({
          scene,
          transform,
          audience,
          filters: visualFilterState,
          ...movement.movementSource(sources.events, audience),
        }),
        doc,
      ),
    );
    // Live play rings the target with corner brackets only, as the vision
    // does: the selected body there is the draft's target, so no moonlight.
    const selectionMoonlight =
      !liveProduct &&
      isVisualPaintPartEnabled(visualFilterState, {
        surface: "durable",
        kind: "selection_moonlight",
      });
    // Live play (BattleClient and DevClient) shows the vision's two reach
    // rings, Basic and Ultimate (an Ultimate with no reach, the Mage's Burst,
    // has none); sight is the Agent POV veil's job. The Replay Viewer keeps
    // its three rings.
    const reachScene = liveProduct
      ? {
          ...scene,
          ranges: asArray(scene.ranges).filter(
            (range) =>
              isRecord(range) &&
              (range.kind === "basic" || range.kind === "ultimate") &&
              finiteNumber(range.radius) > 0,
          ),
        }
      : scene;
    replaceLayer(
      this.rangeCues,
      championFrame.build(
        championFrame.records({
          part: "reach-rings",
          scene: reachScene,
          transform,
          // Live play: Reach Rings is the Ranges rings' own switch (G shows
          // and hides them while it is on; off hides them whatever G says),
          // and no ring is drawn for a selected target with Ranges off, as on
          // the vision page. The Replay Viewer keeps the selected agent's
          // rings with Ranges off.
          showRanges:
            options.showRanges !== false &&
            (!liveProduct ||
              isVisualPaintPartEnabled(visualFilterState, {
                surface: "durable",
                kind: "reach_ring",
              })),
          showReachRings:
            !liveProduct &&
            isVisualPaintPartEnabled(visualFilterState, {
              surface: "durable",
              kind: "reach_ring",
            }),
        }),
        doc,
      ),
    );
    // The live Command Ring and Order Thread (hotbar) replace the controlled
    // halo's route in live play, so the next-action thread is drawn only when
    // the hotbar draws no command ring.
    const focusedId =
      typeof options.focusedPublicAgentId === "string"
        ? options.focusedPublicAgentId
        : null;
    const lookedAtBody =
      focusedId === null
        ? null
        : (asArray(scene.agents).find(
            (agent) => isRecord(agent) && agent.public_agent_id === focusedId,
          ) ?? null);
    const hotbarRecords = hotbar.records({
      scene,
      transform,
      audience,
      filters: visualFilterState,
      lookedAtKey:
        typeof lookedAtBody?.presentation_key === "string"
          ? lookedAtBody.presentation_key
          : null,
      inspection: isAuthorizedPresentationFrame(frame)
        ? authorizedPresentationInspectionState(frame).inspection
        : null,
      manualSlots: Array.isArray(options.manualSlots) ? options.manualSlots : null,
      // The frame's own top-level staged Manual rows; never researcher_space.
      pendingRows:
        isAuthorizedPresentationFrame(frame) &&
        isRecord(frame.pending_joint_action) &&
        Array.isArray(frame.pending_joint_action.action_rows)
          ? frame.pending_joint_action.action_rows
          : [],
    });
    const hotbarDrawsCommand = hotbarRecords.some(
      (record) => record.kind === "command-ring" || record.kind === "order-thread",
    );
    replaceLayer(
      this.layers.pendingRoute,
      hotbarDrawsCommand
        ? []
        : championFrame.build(
            championFrame.records({
              part: "next-action-route",
              scene,
              transform,
              showNextActionThread: isVisualPaintPartEnabled(visualFilterState, {
                surface: "durable",
                kind: "next_action_thread",
              }),
            }),
            doc,
          ),
    );
    const fogInput = {
      scene,
      transform,
      audience,
      live: liveProduct,
      filters: visualFilterState,
      sensorSources: sources.sensorSources,
      trajectories: sources.trajectories,
      schedule: sources.schedule,
    };
    replaceLayer(this.layers.fog, fog.build(fog.records(fogInput), doc));
    // Standing Stones keep their hover targets and tips; on a tale-skinned map
    // the tale's own stones are painted over them (no pointer input, no tips),
    // on the same exact layout.
    replaceLayer(this.layers.obstacle, [
      ...obstacles.build(
        obstacles.records({ map, transform, filters: visualFilterState }),
        doc,
      ),
      ...mapSkins.build(
        mapSkins.taleStoneRecords({
          transform,
          map,
          mapKey: null,
          filters: visualFilterState,
        }),
        doc,
      ),
    ]);
    const dimRecords = atmosphere.records({
      part: "dim",
      transform,
      map,
      filters: visualFilterState,
      fallBeat: sources.schedule?.fall ?? null,
      motionMode: "normal",
    });
    const dimNodes = atmosphere.build(dimRecords, doc);
    for (const [index, node] of dimNodes.entries()) {
      // The choreographer reads the beat window to start the dimming on a
      // played tick; the node itself rests at its END opacity 0.
      const record = dimRecords[index];
      if (node instanceof SVGElement && record !== undefined) {
        node.dataset.dimStart = String(record.start);
        node.dataset.dimEnd = String(record.end);
      }
    }
    replaceLayer(this.layers.deathDim, dimNodes);
    const projectedAgents = this.#renderAgents(
      frame,
      scene,
      transform,
      visualPolicy,
      visualFilterState,
      selectionMoonlight,
      sources.startSeen,
    );
    this.agentByLayoutSlot = new Map(
      projectedAgents.map((projected) => [projected.layoutSlot, projected.agent]),
    );
    replaceLayer(
      this.bodyLayers.corpseMarks,
      requiems.build(
        requiems.records({
          part: "corpse-marks",
          scene,
          transform,
          audience,
          filters: visualFilterState,
        }),
        doc,
      ),
    );
    replaceLayer(this.planMarks, hotbar.build(hotbarRecords, doc));
    // Rune Zones and the Rune Ring ride each body's glide wrapper, after the token.
    /** @type {Element[]} */
    const overNodes = [];
    const statusSwitches = {
      statusMarks: visualFilterState.status_marks !== false,
      cooldownEffects: visualFilterState.cooldown_effects !== false,
      statusApplication: visualFilterState.status_application !== false,
      naturalStatusExpiry: visualFilterState.natural_status_expiry !== false,
      freezingTrapBreak: visualFilterState.freezing_trap_break !== false,
      statusClearOnDeath: visualFilterState.status_clear_on_death === true,
      barrierHeld: visualFilterState.barrier_held !== false,
    };
    const statusRecords = statuses.records({
      part: "status-marks",
      scene,
      transform,
      audience,
      events: incomingEventRows(sources.events),
      trajectories: sources.trajectories,
      startSeen: sources.startSeen,
      switches: statusSwitches,
    });
    const statusNodes = statuses.build(statusRecords, doc);
    // Marks that ride a body's glide wrapper (Rune Zones, the recipient rim),
    // gathered per wrapper and then reconciled after the token and shield.
    /** @type {Map<Element, Element[]>} */
    const riders = new Map();
    const ride = (/** @type {Element} */ wrapper, /** @type {Element} */ node) => {
      const list = riders.get(wrapper);
      if (list) list.push(node);
      else riders.set(wrapper, [node]);
    };
    for (const [index, node] of statusNodes.entries()) {
      const key = statusRecords[index]?.presentationKey;
      const wrapper = typeof key === "string" ? this.#glideWrapperByKey(key) : null;
      if (wrapper) {
        ride(wrapper, node);
      } else {
        overNodes.push(node);
      }
    }
    for (const node of fog.build(fog.records({ ...fogInput, part: "over" }), doc)) {
      const key = node instanceof SVGElement ? node.dataset.presentationKey : undefined;
      const wrapper =
        node.classList.contains("fog-rim") && typeof key === "string"
          ? this.#glideWrapperByKey(key)
          : null;
      if (wrapper) {
        ride(wrapper, node);
      } else {
        overNodes.push(node);
      }
    }
    for (const nodes of this.agentNodes.values()) {
      reconcileNodes(
        nodes.wrapper,
        [
          ...nodes.wrapper.querySelectorAll(
            ":scope > .status-marks, :scope > .fog-rim",
          ),
        ],
        riders.get(nodes.wrapper) ?? [],
      );
    }
    replaceLayer(this.layers.over, overNodes);
    this.#renderObservedBodies(
      scene,
      transform,
      visualPolicy.showPovDurationStatusBadges,
    );
    this.#renderStatusDocks(projectedAgents, transform, {
      showModifiers: visualPolicy.showAuraModifierBadges,
      showStatuses: visualPolicy.showDurationStatusBadges,
      // The Rune Ring on each body replaces the old Ultimate cooldown docks.
      showCooldowns: false,
      audience: scene.audience,
    });
    this.#refreshChoreographyProtectedRects();

    this.layers.accessibleLabels.replaceChildren();
    return true;
  }

  /**
   * Refresh protected regions for older callers and return false.
   *
   * _active is kept for compatibility and ignored. Durable facts are not hidden
   * by transient activity; the shared allocator reserves their regions.
   *
   * @param {boolean} _active
   * @returns {boolean} Whether the effective compact-active state changed.
   */
  setCompactActiveCombat(_active) {
    this.#refreshChoreographyProtectedRects();
    return false;
  }

  /**
   * Convert an SVG-local pointer point into debugger world_x/world_y.
   *
   * point must have finite x/y and a current transform must exist; otherwise
   * return null. The current camera frame is included, so a click lands on the
   * world point under the pointer. Return a new coordinate record without
   * clamping it to the map. This sends no command and does not check whether
   * the position is actionable.
   *
   * @param {{x: number, y: number}} point
   * @returns {{world_x: number, world_y: number} | null}
   */
  toWorldPoint(point) {
    if (!this.transform || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      return null;
    }
    const world = this.transform.screenToWorld(point);
    return {
      world_x: world.x,
      world_y: world.y,
    };
  }

  /**
   * Return the glide wrapper of one drawn body, or null.
   *
   * identityKey is the body's retained-node key (its display_key, or its integer
   * global slot when it has none); the same value is written as the wrapper's
   * data-glide-key. Return the `g.agent-glide` group holding the body token and
   * its spawn-shield chip, or null when that body is not drawn (in Agent POV an
   * unseen champion is never drawn). The choreographer may set a temporary
   * transform on it for a glide; the DOM keeps the END position.
   *
   * @param {number | string} identityKey
   * @returns {SVGElement | null}
   */
  bodyGlideWrapper(identityKey) {
    return this.agentNodes.get(identityKey)?.wrapper ?? null;
  }

  /**
   * Return the narrow layer and projection contract for transient choreography.
   *
   * Return null before a valid durable scene exists. Otherwise return a frozen
   * surface with the foreground layer (`transient-events`), the route layer
   * (`strike-band`), the herald layer (`heralds`, above every body), this
   * render's viewport transform (lookTransform), switch state (visualFilters) and
   * arena scene (scene; Agent POV: the POV scene) for the event looks, the
   * battlefield SVG root (where the choreographer looks for
   * the `g.agent-glide` body wrappers; only drawn bodies are there, so in Agent
   * POV an unseen champion is never found), ownerDocument, the visible map bounds, an immutable
   * protected-region snapshot and a viewportKey that includes those regions and
   * the map bounds (which change with the camera frame). Projection functions
   * close over this render's transform. Callers may append only their own
   * children to layer and routeLayer and must project authorized event anchors
   * instead of borrowing current body positions. Other layers stay owned by
   * this renderer.
   *
   * @returns {Readonly<ChoreographySurface> | null}
   */
  choreographySurface() {
    const transform = this.transform;
    if (!transform) {
      return null;
    }
    const protectedLayoutKey = this.choreographyProtectedRects
      .map((region) =>
        [
          region.bounds.left,
          region.bounds.top,
          region.bounds.right,
          region.bounds.bottom,
        ]
          .map((value) => Number(value).toFixed(3))
          .concat(region.layoutKey)
          .join(","),
      )
      .join(";");
    /** @type {ChoreographySurface} */
    const surface = {
      layer: this.layers.transientEvents,
      routeLayer: this.layers.strikeBand,
      heraldLayer: this.layers.heralds,
      battlefield: this.battlefield,
      ownerDocument: this.battlefield.ownerDocument,
      viewportKey: [
        transform.worldWidth,
        transform.worldHeight,
        transform.viewportBounds.width,
        transform.viewportBounds.height,
        transform.mapBounds.left,
        transform.mapBounds.top,
        transform.mapBounds.right,
        transform.mapBounds.bottom,
        protectedLayoutKey,
      ].join(":"),
      viewportBounds: Object.freeze({ ...transform.visibleMapBounds }),
      protectedRects: this.choreographyProtectedRects,
      /**
       * Project a supplied authorized world point with this surface's captured transform.
       *
       * point is an x/y object or coordinate pair. Return screen coordinates; no body
       * lookup, authority expansion or DOM update occurs.
       */
      worldToScreen: (point) => transform.worldToScreen(point),
      /**
       * Convert a supplied world length with this surface's captured scale.
       *
       * length uses the transform's numeric contract. Return a screen-pixel length without
       * changing renderer state.
       */
      worldLengthToScreen: (length) => transform.worldLengthToScreen(length),
      lookTransform: transform,
      visualFilters: this.lastVisualFilterState,
      scene: this.lastScene,
    };
    return Object.freeze(surface);
  }

  /**
   * Remove durable drawing children and reset retained scene and projection caches.
   *
   * Clear every durable layer (map, atmosphere, Red Zone, spawns, brackets,
   * aura, imprints, selection moonlight, ranges, route, fog, obstacles, Death
   * Dimming, bodies, corpse marks, plan marks, over, selection, legality,
   * docks, heralds,
   * Frame Explanation and labels), dock metadata and the retained node maps, and
   * drop the camera frame mark. Keep the SVG root, the camera wrapper, the fixed
   * layers and the transient route and event children; the animation controller
   * clears its own. Return undefined.
   */
  #clearDurableScene() {
    this.layers.map.replaceChildren();
    this.layers.atmosphere.replaceChildren();
    this.layers.redZone.replaceChildren();
    this.layers.spawns.replaceChildren();
    this.layers.teamIdentity.replaceChildren();
    this.layers.aura.replaceChildren();
    this.layers.imprints.replaceChildren();
    this.selectionMoonlight.replaceChildren();
    this.rangeCues.replaceChildren();
    this.layers.pendingRoute.replaceChildren();
    this.layers.fog.replaceChildren();
    this.layers.obstacle.replaceChildren();
    this.layers.deathDim.replaceChildren();
    this.bodyLayers.dead.replaceChildren();
    this.bodyLayers.corpseMarks.replaceChildren();
    this.bodyLayers.living.replaceChildren();
    this.layers.over.replaceChildren();
    this.planMarks.replaceChildren();
    this.selectionCues.replaceChildren();
    this.legalityCues.replaceChildren();
    this.layers.durableStatusModifier.replaceChildren();
    this.layers.durableStatusModifier.removeAttribute("data-suppressed-status-slots");
    this.layers.durableStatusModifier.removeAttribute("data-suppressed-cooldown-slots");
    this.layers.durableStatusModifier.removeAttribute("data-suppressed-modifier-slots");
    this.layers.durableStatusModifier.removeAttribute("data-compacted-required-docks");
    this.layers.durableStatusModifier.removeAttribute(
      "data-suppressed-status-presentation-keys",
    );
    this.layers.durableStatusModifier.removeAttribute(
      "data-suppressed-cooldown-presentation-keys",
    );
    this.layers.durableStatusModifier.removeAttribute(
      "data-suppressed-modifier-presentation-keys",
    );
    this.layers.durableStatusModifier.removeAttribute(
      "data-compacted-required-presentations",
    );
    this.layers.heralds.replaceChildren();
    this.layers.explain.replaceChildren();
    this.layers.accessibleLabels.replaceChildren();
    this.cameraGroup.removeAttribute("data-camera-frame");
    this.agentNodes.clear();
    this.wrapperByPresentationKey.clear();
    this.observedBodyNodes.clear();
    this.agentByPresentationKey = new Map();
    this.agentByLayoutSlot = new Map();
    this.sceneAgentByPublicId = new Map();
    this.choreographyProtectedRects = Object.freeze([]);
    this.choreographyProtectedRectGroups = Object.freeze({
      base: Object.freeze([]),
      legality: Object.freeze([]),
      status: Object.freeze([]),
    });
    this.transform = null;
    this.lastScene = null;
  }

  /**
   * Publish a frozen copy of current body/dock/legality protected regions.
   *
   * Read base, status and legality groups in that order and copy nested bounds.
   * Update choreographyProtectedRects; return undefined without recomputing layout.
   */
  #refreshChoreographyProtectedRects() {
    const groups = this.choreographyProtectedRectGroups;
    this.choreographyProtectedRects = Object.freeze(
      [...groups.base, ...groups.status, ...groups.legality].map((region) =>
        Object.freeze({
          ...region,
          bounds: Object.freeze({ ...region.bounds }),
        }),
      ),
    );
  }

  /**
   * Return the tooltip input for one aura field, with its joined source row.
   *
   * record is an authorized aura field; audience is the scene audience. When the
   * field names a source public ID that an agent of the current arena scene
   * carries, the explanation record carries that source's presentation key and
   * public ID and the source row. In Agent POV the scene is the POV scene, so an
   * unseen source gives sourceAgent null and the card names no source. Return {field, sourceAgent, audience} for the Statuses look.
   * Reads the field without calling getters. No body or position is added.
   *
   * @param {JsonRecord} record
   * @param {"researcher" | "agent_pov"} audience
   * @returns {{field: JsonRecord, sourceAgent: JsonRecord | null, audience: string}}
   */
  #auraTooltipInput(record, audience) {
    const sourcePublicAgentId = ownEnumerableDataValue(
      record,
      "source_public_agent_id",
    );
    const sourceAgent =
      typeof sourcePublicAgentId === "string"
        ? (this.sceneAgentByPublicId.get(sourcePublicAgentId) ?? null)
        : null;
    const explanationRecord =
      sourceAgent === null
        ? record
        : {
            ...record,
            source_presentation_key: sourceAgent.presentation_key,
            source_public_agent_id: sourceAgent.public_agent_id,
          };
    return { field: explanationRecord, sourceAgent, audience };
  }

  /**
   * Stack the arena layers as the vision page does, for live play.
   *
   * The BattleClient and the DevClient combat screen call this on each render
   * (it moves nothing once in place): the reach rings and the look ring
   * (`debug-range`) go over the sight veil and the stones, the plan marks lie
   * under every body, and the target's corner brackets (`selection-cues`) go
   * over the bodies. The Replay Viewer never calls it and keeps the
   * constructor's order (BATTLEFIELD_LAYER_ORDER). Moving a layer keeps its
   * nodes, attributes and running animations. Return undefined. DOM order only.
   */
  #arrangeLiveLayers() {
    const { debugRange, obstacle, body } = this.layers;
    if (obstacle.nextElementSibling !== debugRange) obstacle.after(debugRange);
    placeInOrder(body, [
      this.planMarks,
      this.bodyLayers.dead,
      this.bodyLayers.corpseMarks,
      this.bodyLayers.living,
      this.selectionCues,
    ]);
  }

  /**
   * Return the glide wrapper of a drawn body by its presentation key, or null.
   *
   * presentationKey is the body's audience-scoped presentation key. Return the
   * `g.agent-glide` group of that body from this render, or null when the body
   * is not drawn (in Agent POV an unseen champion is never drawn). Read only.
   *
   * @param {string} presentationKey
   * @returns {SVGElement | null}
   */
  #glideWrapperByKey(presentationKey) {
    return this.wrapperByPresentationKey.get(presentationKey) ?? null;
  }

  /**
   * Keep each drawn body's retained nodes and return their projected records.
   *
   * frame is the normalized presentation (its audience's incoming rows give the
   * beat cues); scene supplies accepted agents and selection; transform projects
   * positions and radii; visualPolicy controls the spawn shield and corner
   * marks; visualFilterState is the page's switch state (the champion look and
   * the beat clock read it); showMoonlight says whether Selection Moonlight is
   * on; startSeen is the audience's seen-at-tick-start test from
   * audienceSources (Oracle View: always true; Agent POV: the body has a
   * tick-start spot in the team's own trajectories), so a Class Seal or a
   * Spawn Ward's lift shows its earlier look only on a body the audience saw
   * at the start of the tick. Ask the Champions look for one record per
   * drawable body, with its beat cues from this audience's rows and beatView
   * (Agent POV: visual_events only), remove bodies that are gone, build nodes for new ones (body token in its
   * glide wrapper and the selection marks), update every bundle, register the
   * token tooltip and put each wrapper in the dead or living group (every
   * corpse under every living body, scene order inside each), with its shield
   * after the token. A wrapper, token, shield or selection node already in its
   * place is not moved, and a body that changes group moves once
   * (placeGroupsInOrder), so a repaint tears nothing down. Marks
   * that ride a wrapper (Rune Zones, the recipient rim) are left for render,
   * which reconciles them after the token and shield. The selection
   * brackets go in the body layer's first group (under every body) and the
   * moonlight in the debug-range layer's first group. Return the champion
   * records as ProjectedAgent rows (screen center and radius, with separate
   * global, presentation and layout identities). Layout fallback indices never
   * become public slot facts. Source records stay unchanged.
   *
   * @param {unknown} frame
   * @param {JsonRecord} scene
   * @param {ViewportTransform} transform
   * @param {Readonly<DurableVisualPolicy>} visualPolicy
   * @param {Readonly<Record<string, boolean>>} visualFilterState
   * @param {boolean} showMoonlight
   * @param {(presentationKey: string) => boolean} startSeen
   * @returns {ProjectedAgent[]}
   */
  #renderAgents(
    frame,
    scene,
    transform,
    visualPolicy,
    visualFilterState,
    showMoonlight,
    startSeen,
  ) {
    const doc = this.battlefield.ownerDocument;
    const championRecords = champions.records({
      scene,
      transform,
      showSpawnShield: visualPolicy.showSpawnShield,
      visualFilters: visualFilterState,
      startSeen,
      beats: isAuthorizedPresentationFrame(frame)
        ? {
            rows: authorizedPresentationIncomingRows(frame),
            view: beatView(frame, "end", { visualFilters: visualFilterState }),
          }
        : undefined,
    });
    const nextIdentities = new Set(championRecords.map((record) => record.identityKey));
    for (const [identityKey, nodes] of this.agentNodes) {
      if (!nextIdentities.has(identityKey)) {
        nodes.wrapper.remove();
        nodes.selectionRoot.remove();
        nodes.moonlight?.remove();
        this.agentNodes.delete(identityKey);
      }
    }
    this.wrapperByPresentationKey.clear();
    const selectionRecords = championFrame.records({
      part: "selection-marks",
      champions: championRecords,
      showSelectionReticle: visualPolicy.showSelectionReticle,
      showSelectionMoonlight: showMoonlight,
      // Live play draws the BattleClient vision's corner brackets.
      bracketShape:
        isRecord(frame) && frame.product_kind === "combat_debugger"
          ? "vision"
          : "ladder",
    });

    // Wanted children of each group, in paint order. A body that is already in
    // place is never removed and re-inserted, and a body that falls or returns
    // moves once between the two groups (placeGroupsInOrder), so a repaint
    // keeps every other body node and its running glide (smooth play).
    /** @type {Element[]} */
    const living = [];
    /** @type {Element[]} */
    const dead = [];
    /** @type {Element[]} */
    const selectionRoots = [];
    /** @type {Element[]} */
    const moonlights = [];
    for (const [index, record] of championRecords.entries()) {
      const selectionRecord = selectionRecords[index];
      let nodes = this.agentNodes.get(record.identityKey);
      if (!nodes) {
        nodes = {
          ...champions.build([record], doc)[0],
          ...championFrame.buildSelectionMarks([selectionRecord], doc)[0],
        };
        this.agentNodes.set(record.identityKey, nodes);
      }
      champions.update([nodes], [record]);
      championFrame.updateSelectionMarks([nodes], [selectionRecord]);
      registerTooltipOwner(nodes.root, champions.tooltip(record));

      (record.agent.alive ? living : dead).push(nodes.wrapper);
      // Inside the wrapper: the token first, then its shield.
      if (nodes.wrapper.firstElementChild !== nodes.root) {
        nodes.wrapper.insertBefore(nodes.root, nodes.wrapper.firstChild);
      }
      if (visualPolicy.showSpawnShield && nodes.shieldRoot !== null) {
        if (nodes.root.nextElementSibling !== nodes.shieldRoot) {
          nodes.root.after(nodes.shieldRoot);
        }
      } else if (nodes.shieldRoot?.parentNode) {
        nodes.shieldRoot.remove();
      }
      if (typeof record.presentationKey === "string") {
        this.wrapperByPresentationKey.set(record.presentationKey, nodes.wrapper);
      }
      selectionRoots.push(nodes.selectionRoot);
      if (nodes.moonlight) moonlights.push(nodes.moonlight);
    }
    placeGroupsInOrder([
      [this.bodyLayers.living, living],
      [this.bodyLayers.dead, dead],
    ]);
    placeInOrder(this.selectionCues, selectionRoots);
    placeInOrder(this.selectionMoonlight, moonlights);
    return /** @type {ProjectedAgent[]} */ (
      /** @type {unknown} */ ([...championRecords])
    );
  }

  /**
   * Draw recipient-observed bodies without assigning undisclosed global slots.
   *
   * scene.observed_bodies must carry an ally/enemy relation, integer observation_row
   * and exactly matching observation_key. transform projects their authorized values;
   * showDurationStatusBadges controls status paint and matching accessible text.
   * Retain nodes by relation-row key, remove absent bodies and update hover/keyboard
   * inspection. Use the same corpse/living paint groups as ordinary agents so later
   * observed corpses cannot hide living actors. These hit regions do not become
   * target/control regions. Return undefined; no hidden body identity or position
   * is invented. The authorized adapter never sets `scene.observed_bodies` today,
   * so this older path draws nothing; it stays until its last browser check moves.
   *
   * @param {JsonRecord} scene
   * @param {ViewportTransform} transform
   * @param {boolean} showDurationStatusBadges
   */
  #renderObservedBodies(scene, transform, showDurationStatusBadges) {
    const doc = this.battlefield.ownerDocument;
    const bodies = asArray(scene.observed_bodies).filter(
      (body) =>
        isRecord(body) &&
        (body.relation === "ally" || body.relation === "enemy") &&
        Number.isInteger(body.observation_row) &&
        body.observation_key === `${body.relation}:${body.observation_row}`,
    );
    const keys = new Set(bodies.map((body) => String(body.observation_key)));
    for (const [key, node] of this.observedBodyNodes) {
      if (!keys.has(key)) {
        node.remove();
        this.observedBodyNodes.delete(key);
      }
    }

    for (const body of bodies) {
      const key = String(body.observation_key);
      const center = screenPoint(body.position, transform);
      const radius = transform.worldLengthToScreen(finiteNumber(body.radius, 0.5));
      const classToken = classTokenFromId(body.class_id);
      const teamToken = teamTokenFromId(body.team_id);
      let root = this.observedBodyNodes.get(key);
      if (!root) {
        root = svgNode(doc, "g", {
          class: "agent pov-observed-body",
          tabindex: "0",
          role: "img",
          "data-observation-key": key,
        });
        root.append(
          svgNode(doc, "circle", {
            class: "agent-body",
            "data-zone": "observed-body",
          }),
          svgNode(doc, "circle", {
            class: "agent-team-ring",
            "data-zone": "observed-team",
          }),
          svgNode(doc, "circle", {
            class: "agent-health-track",
            "data-zone": "observed-health",
          }),
          svgNode(doc, "circle", {
            class: "agent-health",
            "data-zone": "observed-health",
            pathLength: 100,
          }),
          createSvgIcon(this.battlefield.ownerDocument, classToken.glyphKey, {
            className: "agent-class-icon",
          }),
          svgNode(doc, "text", { class: "pov-observed-body__label" }),
          svgNode(doc, "g", {
            class: "pov-observed-body__statuses",
            "data-zone": "observed-statuses",
          }),
        );
        this.observedBodyNodes.set(key, root);
      }

      root.dataset.relation = body.relation;
      root.dataset.observationRow = String(body.observation_row);
      root.dataset.publicAgentId = String(body.public_agent_id);
      root.dataset.team = teamToken.cssKey;
      root.dataset.class = classToken.cssKey;
      root.dataset.alive = String(Boolean(body.alive));
      const observedStatuses = asArray(body.statuses);
      const paintedStatuses = showDurationStatusBadges ? observedStatuses : [];
      const statusDescriptions = paintedStatuses.map((status) => {
        const token = resolveVisualToken("status", status.token_id);
        return `${token.accessibleName}, ${formatDisplayNumber(status.duration)} ticks`;
      });
      root.dataset.statusCount = String(observedStatuses.length);
      root.setAttribute(
        "aria-label",
        [
          `${body.relation} observation row ${body.observation_row}`,
          agentIdentity(body),
          classToken.label,
          `life state ${body.alive ? "alive" : "corpse"}`,
          `health ${formatDisplayNumber(body.current_health)} of ${formatDisplayNumber(body.max_health)}`,
          `effective speed ${formatDisplayNumber(body.effective_movement_speed)}`,
          showDurationStatusBadges
            ? statusDescriptions.length === 0
              ? "no persistent statuses"
              : `statuses ${statusDescriptions.join(", ")}`
            : null,
        ]
          .filter((part) => part !== null)
          .join(", "),
      );

      const healthRadius = Math.max(radius - 4, radius * 0.7);
      const healthRatio = Math.max(
        0,
        Math.min(
          1,
          finiteNumber(body.current_health) /
            Math.max(finiteNumber(body.max_health, 1), Number.EPSILON),
        ),
      );
      const [bodyCircle, teamRing, healthTrack, health, icon, label, statusGroup] =
        root.children;
      setNodeAttributes(/** @type {SVGElement} */ (bodyCircle), {
        cx: center.x,
        cy: center.y,
        r: radius,
      });
      setNodeAttributes(/** @type {SVGElement} */ (teamRing), {
        cx: center.x,
        cy: center.y,
        r: radius,
      });
      setNodeAttributes(/** @type {SVGElement} */ (healthTrack), {
        cx: center.x,
        cy: center.y,
        r: healthRadius,
      });
      setNodeAttributes(/** @type {SVGElement} */ (health), {
        cx: center.x,
        cy: center.y,
        r: healthRadius,
        "stroke-dasharray": `${healthRatio * 100} ${100 - healthRatio * 100}`,
        transform: `rotate(-90 ${center.x} ${center.y})`,
      });
      const iconSize = Math.max(14, Math.min(radius * 0.95, 28));
      setNodeAttributes(/** @type {SVGElement} */ (icon), {
        x: center.x - iconSize / 2,
        y: center.y - iconSize * 0.62,
        width: iconSize,
        height: iconSize,
      });
      setNodeAttributes(/** @type {SVGElement} */ (label), {
        x: center.x,
        y: center.y + radius + 15,
      });
      label.textContent = `${body.relation === "ally" ? "ALLY" : "ENEMY"} ${body.observation_row}`;
      const statusCellWidth = 18;
      const statusCellHeight = 15;
      const statusColumns = Math.min(paintedStatuses.length, 5);
      const statusNodes = paintedStatuses.map((status, index) => {
        const token = resolveVisualToken("status", status.token_id);
        const rowIndex = Math.floor(index / 5);
        const columnIndex = index % 5;
        const rowCount = Math.min(5, paintedStatuses.length - rowIndex * 5);
        const x =
          center.x - (rowCount * statusCellWidth) / 2 + columnIndex * statusCellWidth;
        const y = center.y - radius - 19 - rowIndex * statusCellHeight;
        const effectClass = classTokenFromId(status.source_class_id);
        const cell = svgNode(doc, "g", {
          class: "pov-observed-status",
          transform: `translate(${x} ${y})`,
          role: "img",
          "aria-label": `${token.accessibleName}, duration ${formatDisplayNumber(status.duration)} ticks`,
          "data-token-id": token.tokenId,
          "data-duration": status.duration,
          "data-status-feature-index": status.status_feature_index,
          "data-effect-class-id": status.source_class_id,
          "data-effect-class": effectClass.cssKey,
        });
        const statusIcon = createSvgIcon(
          this.battlefield.ownerDocument,
          token.glyphKey,
          { className: "pov-observed-status__icon" },
        );
        setNodeAttributes(statusIcon, {
          x: 2,
          y: 2,
          width: 8,
          height: 8,
        });
        cell.append(
          svgNode(doc, "rect", {
            class: "pov-observed-status__box",
            x: 0,
            y: 0,
            width: statusCellWidth - 2,
            height: statusCellHeight - 1,
            rx: 3,
          }),
          statusIcon,
          svgNode(doc, "text", {
            class: "pov-observed-status__duration",
            x: 12,
            y: 11,
          }),
        );
        const duration = cell.lastElementChild;
        if (duration) {
          duration.textContent = formatCompactDisplayNumber(status.duration);
        }
        registerTooltipOwner(
          cell,
          statuses.tooltip(
            Object.freeze({
              kind: "dock-cell",
              tooltipInput: this.#statusTooltipInput(
                status,
                {
                  presentation_key: body.presentation_key,
                  public_agent_id: body.public_agent_id,
                  class_id: body.class_id,
                  team_id: body.team_id,
                },
                "agent_pov",
              ),
            }),
          ),
        );
        return cell;
      });
      statusGroup.replaceChildren(...statusNodes);
      if (showDurationStatusBadges) {
        statusGroup.setAttribute("data-columns", String(statusColumns));
      } else {
        statusGroup.removeAttribute("data-columns");
      }
      registerTooltipOwner(
        root,
        explainPovAgent(body, { controlled: false, selected: false }),
      );
      (body.alive ? this.bodyLayers.living : this.bodyLayers.dead).append(root);
    }
  }

  /**
   * Clear the selected pair's legality layer.
   *
   * The arena no longer draws Basic/Ultimate lane pills (the Champion Frame and
   * the Tactics Turn deck show what is allowed), so this only empties the
   * legality layer and drops its old suppression marks. It reserves no arena
   * space. Returns an empty ProtectedRegion array, so callers keep their shape.
   *
   * @returns {ProtectedRegion[]}
   */
  #renderSelectedLegality() {
    this.legalityCues.replaceChildren();
    this.legalityCues.removeAttribute("data-suppressed-slot");
    this.legalityCues.removeAttribute("data-suppressed-presentation-key");
    return [];
  }

  /**
   * Place separate current status, cooldown and modifier docks.
   *
   * projectedAgents contains disclosed current bodies; transform supplies pixel
   * bounds. policy supplies three show switches and audience. Required
   * selected/controlled statuses and positive cooldowns are placed first, then
   * optional statuses and nonneutral modifiers. The legality layer is only
   * cleared (no lane pills are drawn and no legality dock is reserved). Small
   * viewports use compact/overflow representations; the shared allocators own collision
   * choices. The Statuses look draws each dock from its placement. Replace durable dock
   * DOM, publish suppression/compaction metadata and protected regions, then fit numeric
   * cells. Return undefined. Python-provided status order, values and actor identity
   * remain the semantic authority.
   *
   * @param {ProjectedAgent[]} projectedAgents
   * @param {ViewportTransform} transform
   * @param {{
   *   showModifiers: boolean,
   *   showStatuses: boolean,
   *   showCooldowns: boolean,
   *   audience: "researcher" | "agent_pov",
   * }} policy
   */
  #renderStatusDocks(projectedAgents, transform, policy) {
    const compactMinimumViewport =
      transform.viewportBounds.width <= 600 && transform.viewportBounds.height <= 420;
    const requiredStatusSlots = new Set(
      policy.showStatuses
        ? projectedAgents
            .filter(
              (agent) =>
                agent.statuses.length > 0 && (agent.controlled || agent.selected),
            )
            .map(({ layoutSlot }) => layoutSlot)
        : [],
    );
    const requiredDockRequests = [
      ...projectedAgents
        .filter(({ layoutSlot }) => requiredStatusSlots.has(layoutSlot))
        .map((agent) => ({
          layoutKey: `status:${agent.layoutSlot}`,
          globalSlot: agent.layoutSlot,
          publicAgentId: agent.agent.public_agent_id,
          statuses: agent.statuses,
          dockOptions: {
            ...STATUS_DOCK_DIMENSIONS,
            dockGap: 5,
            requiredVisibleLimit: compactMinimumViewport ? 0 : 9,
          },
          fallbackDockOptions: {
            cellWidth: 32,
            cellHeight: 16,
            cellGap: 0,
            dockGap: 3,
          },
          priority: 0,
        })),
      ...projectedAgents.flatMap((agent) => {
        const ticks = agent.agent.ultimate_cooldown;
        if (!policy.showCooldowns || !Number.isInteger(ticks) || Number(ticks) <= 0) {
          return [];
        }
        return [
          {
            layoutKey: `cooldown:${agent.layoutSlot}`,
            globalSlot: agent.layoutSlot,
            publicAgentId: agent.agent.public_agent_id,
            statuses: Object.freeze([
              Object.freeze({
                classId: agent.agent.class_id,
                ticks: Number(ticks),
                publicAgentId: agent.agent.public_agent_id,
              }),
            ]),
            dockOptions: {
              ...COOLDOWN_DOCK_DIMENSIONS,
              dockGap: 5,
            },
            fallbackDockOptions: {
              ...COOLDOWN_DOCK_DIMENSIONS,
              dockGap: 5,
            },
            priority: 1,
          },
        ];
      }),
    ];
    const requiredDockLayout = layoutRequiredDocks(
      {
        agents: projectedAgents.map((agent) => ({
          globalSlot: agent.layoutSlot,
          center: agent.center,
          radius: agent.radius,
          statuses: Object.freeze([]),
          controlled: agent.controlled,
          selected: agent.selected,
        })),
        requests: requiredDockRequests,
        viewport: transform.viewportBounds,
      },
      {
        bodyPadding: 4,
        selectionAllowance: 12,
        dockGap: 5,
      },
    );
    const requiredStatusDocks = requiredDockLayout.docks.filter(
      ({ compactFallback, layoutKey }) =>
        !compactFallback && layoutKey.startsWith("status:"),
    );
    const cooldownDocks = requiredDockLayout.docks.filter(({ layoutKey }) =>
      layoutKey.startsWith("cooldown:"),
    );
    const compactRequiredDocks = requiredDockLayout.docks.filter(
      ({ compactFallback, layoutKey }) =>
        compactFallback && layoutKey.startsWith("status:"),
    );
    const requiredDockRects = requiredDockLayout.docks.map(({ bounds }) => bounds);
    const optionalStatusLayout = layoutStatusDocks(
      {
        agents: projectedAgents.map((agent) => ({
          globalSlot: agent.layoutSlot,
          center: agent.center,
          radius: agent.radius,
          statuses:
            policy.showStatuses && !requiredStatusSlots.has(agent.layoutSlot)
              ? agent.statuses
              : Object.freeze([]),
          controlled: agent.controlled,
          selected: agent.selected,
        })),
        viewport: transform.viewportBounds,
        reservedRects: requiredDockRects,
      },
      {
        bodyPadding: 4,
        selectionAllowance: 12,
        ...STATUS_DOCK_DIMENSIONS,
        dockGap: 5,
        ordinaryVisibleLimit: compactMinimumViewport ? 0 : 9,
      },
    );
    const suppressedStatusSlots = [
      ...requiredDockLayout.suppressedLayoutKeys
        .filter((layoutKey) => layoutKey.startsWith("status:"))
        .map((layoutKey) => Number(layoutKey.slice("status:".length))),
      ...optionalStatusLayout.suppressedGlobalSlots,
    ].sort((left, right) => left - right);
    const suppressedCooldownSlots = requiredDockLayout.suppressedLayoutKeys
      .filter((layoutKey) => layoutKey.startsWith("cooldown:"))
      .map((layoutKey) => Number(layoutKey.slice("cooldown:".length)))
      .sort((left, right) => left - right);
    const statusLayout = {
      docks: [...requiredStatusDocks, ...optionalStatusLayout.docks].sort(
        (left, right) => left.globalSlot - right.globalSlot,
      ),
      protectedBodies: requiredDockLayout.protectedBodies,
      placementOrder: [
        ...requiredStatusDocks.map(({ globalSlot }) => globalSlot),
        ...optionalStatusLayout.placementOrder,
      ],
      suppressedGlobalSlots: suppressedStatusSlots,
    };
    const cooldownLayout = {
      docks: cooldownDocks,
      suppressedGlobalSlots: suppressedCooldownSlots,
    };
    const statusRects = statusLayout.docks.map(({ bounds }) => bounds);
    const cooldownRects = cooldownDocks.map(({ bounds }) => bounds);
    const compactRequiredRects = compactRequiredDocks.map(({ bounds }) => bounds);
    const legalityRects = this.#renderSelectedLegality();
    const modifierLayout = policy.showModifiers
      ? layoutStatusDocks(
          {
            agents: projectedAgents.map((agent) => ({
              globalSlot: agent.layoutSlot,
              center: agent.center,
              radius: agent.radius,
              statuses: asArray(agent.agent.modifiers).filter(
                (modifier) =>
                  !isRecord(modifier) ||
                  typeof modifier.multiplier !== "number" ||
                  !Number.isFinite(modifier.multiplier) ||
                  modifier.multiplier !== 1,
              ),
              controlled: agent.controlled,
              selected: agent.selected,
            })),
            viewport: transform.viewportBounds,
            reservedRects: [
              ...statusLayout.protectedBodies.map(({ bounds }) => bounds),
              ...statusRects,
              ...cooldownRects,
              ...compactRequiredRects,
              ...legalityRects,
            ],
          },
          {
            bodyPadding: 4,
            selectionAllowance: 12,
            ...MODIFIER_DOCK_DIMENSIONS,
            dockGap: 5,
          },
        )
      : { docks: [], suppressedGlobalSlots: [] };

    const doc = this.battlefield.ownerDocument;
    /** @type {{
     *   cell: (item: JsonRecord, owner: JsonRecord, dockKind: "status" | "modifier") => DockTooltipInput,
     *   overflow: (items: ReadonlyArray<unknown>, owner: JsonRecord, dockKind: "status" | "modifier") => DockTooltipInput,
     * }} */
    const tooltipInputs = {
      cell: (item, owner, dockKind) =>
        dockKind === "status"
          ? this.#statusTooltipInput(item, owner, policy.audience)
          : { kind: "modifier", modifier: item, recipient: owner },
      overflow: (items, owner, dockKind) =>
        policy.audience === "agent_pov" && dockKind === "status"
          ? this.#statusOverflowTooltipInput(items, owner)
          : {
              kind: "overflow",
              items,
              dockKind,
              recipient: owner,
              sourceAgents: [...this.agentByLayoutSlot.values()],
            },
    };
    /**
     * Ask the Statuses look to draw fact docks of one kind at their placements.
     *
     * placements are allocator docks; dockKind is status or modifier; dimensions
     * gives each placement's cell size in pixels. Return the dock groups in order.
     *
     * @param {ReadonlyArray<any>} placements
     * @param {"status" | "modifier"} dockKind
     * @param {(placement: any) => {cellWidth: number, cellHeight: number, cellGap: number}} dimensions
     */
    const factDockNodes = (placements, dockKind, dimensions) =>
      statuses.build(
        statuses.records({
          part: "fact-docks",
          docks: placements.map((placement) => ({
            placement,
            dockKind,
            dimensions: dimensions(placement),
          })),
          ownerByLayoutSlot: this.agentByLayoutSlot,
          audience: policy.audience,
          tooltipInputs,
        }),
        doc,
      );
    const modifierNodes = factDockNodes(
      modifierLayout.docks,
      "modifier",
      () => MODIFIER_DOCK_DIMENSIONS,
    );
    const statusNodes = factDockNodes(
      statusLayout.docks,
      "status",
      () => STATUS_DOCK_DIMENSIONS,
    );
    const cooldownNodes = statuses.build(
      statuses.records({
        part: "cooldown-docks",
        docks: [
          ...cooldownLayout.docks.filter(({ compactFallback }) => compactFallback),
          ...cooldownLayout.docks.filter(({ compactFallback }) => !compactFallback),
        ],
        ownerByLayoutSlot: this.agentByLayoutSlot,
      }),
      doc,
    );
    const compactRequiredNodes = factDockNodes(
      compactRequiredDocks,
      "status",
      (placement) => ({
        cellWidth: placement.bounds.width,
        cellHeight: placement.bounds.height,
        cellGap: 0,
      }),
    );
    const usesPresentationKeys = projectedAgents.some(
      ({ presentationKey }) => presentationKey !== null,
    );
    /**
     * Map internal layout indices to already disclosed presentation keys.
     *
     * slots is an ordered index array; omit missing/nonstring keys. Return a new array
     * for suppression metadata, without exposing local indices as Agent POV slot facts.
     */
    const presentationKeysForLayoutSlots = (
      /** @type {ReadonlyArray<number>} */ slots,
    ) =>
      slots.flatMap((layoutSlot) => {
        const key = this.agentByLayoutSlot.get(layoutSlot)?.presentation_key;
        return typeof key === "string" ? [key] : [];
      });
    if (usesPresentationKeys) {
      this.layers.durableStatusModifier.removeAttribute("data-suppressed-status-slots");
      this.layers.durableStatusModifier.removeAttribute(
        "data-suppressed-cooldown-slots",
      );
      this.layers.durableStatusModifier.removeAttribute(
        "data-suppressed-modifier-slots",
      );
      this.layers.durableStatusModifier.removeAttribute(
        "data-compacted-required-docks",
      );
      if (policy.showStatuses) {
        this.layers.durableStatusModifier.dataset.suppressedStatusPresentationKeys =
          presentationKeysForLayoutSlots(statusLayout.suppressedGlobalSlots).join(",");
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-status-presentation-keys",
        );
      }
      if (policy.showCooldowns) {
        this.layers.durableStatusModifier.dataset.suppressedCooldownPresentationKeys =
          presentationKeysForLayoutSlots(cooldownLayout.suppressedGlobalSlots).join(
            ",",
          );
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-cooldown-presentation-keys",
        );
      }
      if (policy.showModifiers) {
        this.layers.durableStatusModifier.dataset.suppressedModifierPresentationKeys =
          presentationKeysForLayoutSlots(modifierLayout.suppressedGlobalSlots).join(
            ",",
          );
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-modifier-presentation-keys",
        );
      }
      this.layers.durableStatusModifier.dataset.compactedRequiredPresentations =
        requiredDockLayout.compactedLayoutKeys
          .flatMap((layoutKey) => {
            const [kind, rawLayoutSlot] = layoutKey.split(":");
            const key = this.agentByLayoutSlot.get(
              Number(rawLayoutSlot),
            )?.presentation_key;
            return typeof key === "string" ? [`${kind}:${key}`] : [];
          })
          .join(",");
    } else {
      this.layers.durableStatusModifier.removeAttribute(
        "data-suppressed-status-presentation-keys",
      );
      this.layers.durableStatusModifier.removeAttribute(
        "data-suppressed-cooldown-presentation-keys",
      );
      this.layers.durableStatusModifier.removeAttribute(
        "data-suppressed-modifier-presentation-keys",
      );
      this.layers.durableStatusModifier.removeAttribute(
        "data-compacted-required-presentations",
      );
      if (policy.showStatuses) {
        this.layers.durableStatusModifier.dataset.suppressedStatusSlots =
          statusLayout.suppressedGlobalSlots.join(",");
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-status-slots",
        );
      }
      if (policy.showCooldowns) {
        this.layers.durableStatusModifier.dataset.suppressedCooldownSlots =
          cooldownLayout.suppressedGlobalSlots.join(",");
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-cooldown-slots",
        );
      }
      this.layers.durableStatusModifier.dataset.compactedRequiredDocks =
        requiredDockLayout.compactedLayoutKeys.join(",");
      if (policy.showModifiers) {
        this.layers.durableStatusModifier.dataset.suppressedModifierSlots =
          modifierLayout.suppressedGlobalSlots.join(",");
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-modifier-slots",
        );
      }
    }
    this.layers.durableStatusModifier.replaceChildren(
      ...modifierNodes,
      ...compactRequiredNodes,
      ...cooldownNodes,
      ...statusNodes,
    );
    statuses.fitDockValues(this.layers.durableStatusModifier);
    /**
     * Resolve an internal layout slot to its protected-region identity.
     *
     * globalSlot is the allocator index. Return a frozen ownerPresentationKey and
     * ownerIdentity; a legacy slot key is used only when no presentation key exists.
     *
     * @param {number} globalSlot
     */
    const protectedOwner = (globalSlot) => {
      const key = this.agentByLayoutSlot.get(globalSlot)?.presentation_key;
      const ownerPresentationKey = typeof key === "string" ? key : null;
      return Object.freeze({
        ownerPresentationKey,
        ownerIdentity: ownerPresentationKey ?? `slot:${globalSlot}`,
      });
    };
    /**
     * Package one allocator-owned body or dock rectangle for choreography.
     *
     * kind names the region, globalSlot is its internal owner index, bounds is its
     * pixel rectangle, and placementIdentity distinguishes docks. Return a frozen
     * region using the disclosed owner key when available; no spatial owner is guessed.
     *
     * @param {"body" | "status" | "cooldown" | "modifier"} kind
     * @param {number} globalSlot
     * @param {Rectangle} bounds
     * @param {string} placementIdentity
     */
    const protectedDock = (kind, globalSlot, bounds, placementIdentity) => {
      const owner = protectedOwner(globalSlot);
      return choreographyProtectedRegion(
        kind,
        owner.ownerIdentity,
        bounds,
        owner.ownerPresentationKey,
        placementIdentity,
      );
    };
    const bodyProtectedRegions = statusLayout.protectedBodies.map(
      ({ globalSlot, bounds }) =>
        protectedDock("body", globalSlot, bounds, `body:${globalSlot}`),
    );
    const cooldownProtectedRegions = cooldownDocks.map(
      ({ globalSlot, bounds, layoutKey, compactFallback }) =>
        protectedDock(
          "cooldown",
          globalSlot,
          bounds,
          compactFallback ? `compact:${layoutKey}` : layoutKey,
        ),
    );
    const compactRequiredStatusRegions = compactRequiredDocks
      .filter(({ layoutKey }) => layoutKey.startsWith("status:"))
      .map(({ globalSlot, bounds, layoutKey }) =>
        protectedDock("status", globalSlot, bounds, `compact:${layoutKey}`),
      );
    const modifierProtectedRegions = modifierLayout.docks.map(
      ({ globalSlot, bounds }) =>
        protectedDock("modifier", globalSlot, bounds, `modifier:${globalSlot}`),
    );
    const statusProtectedRegions = statusLayout.docks.map((placement) =>
      protectedDock(
        "status",
        placement.globalSlot,
        placement.bounds,
        "layoutKey" in placement && typeof placement.layoutKey === "string"
          ? placement.layoutKey
          : `status:${placement.globalSlot}`,
      ),
    );
    this.choreographyProtectedRectGroups = Object.freeze({
      base: Object.freeze([
        ...bodyProtectedRegions,
        ...cooldownProtectedRegions,
        ...modifierProtectedRegions,
      ]),
      legality: Object.freeze(legalityRects),
      status: Object.freeze([
        ...statusProtectedRegions,
        ...compactRequiredStatusRegions,
      ]),
    });
    this.#refreshChoreographyProtectedRects();
  }

  /**
   * Choose the tooltip input of one drawn status.
   *
   * rawStatus is the status as the current arena scene carries it;
   * localRecipient carries its public ID and local presentation key; audience
   * is "researcher" (Oracle View) or "agent_pov". Oracle View explains the
   * status with the scene agents as possible sources. Agent POV always uses the
   * limited POV status card built from the POV scene alone: it never joins a
   * researcher row, so it cannot name a source the team did not see. Return the
   * DockTooltipInput for the Statuses look. This admits no status, reads no
   * researcher data and adds no geometry.
   *
   * @param {unknown} rawStatus The drawn status record.
   * @param {JsonRecord} localRecipient The champion that carries it.
   * @param {"researcher" | "agent_pov"} audience The scene audience.
   * @returns {DockTooltipInput}
   */
  #statusTooltipInput(rawStatus, localRecipient, audience) {
    return audience === "agent_pov"
      ? { kind: "pov-status", status: rawStatus, recipient: localRecipient }
      : {
          kind: "status",
          status: rawStatus,
          recipient: localRecipient,
          sourceAgents: [...this.agentByLayoutSlot.values()],
        };
  }

  /**
   * Choose the tooltip input for all hidden statuses of one Agent POV dock.
   *
   * rawStatuses is the ordered hidden list as the POV scene carries it and
   * localRecipient supplies identity. The whole list uses the POV overflow
   * card; no researcher row is joined, so no unseen source can be named.
   * Return the DockTooltipInput without changing statuses or admitting new
   * visible facts.
   *
   * @param {ReadonlyArray<unknown>} rawStatuses The statuses hidden behind "+N".
   * @param {JsonRecord} localRecipient The champion that carries them.
   * @returns {DockTooltipInput}
   */
  #statusOverflowTooltipInput(rawStatuses, localRecipient) {
    return { kind: "pov-overflow", items: rawStatuses, recipient: localRecipient };
  }
}
