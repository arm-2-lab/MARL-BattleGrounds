/**
 * @file Paint authorized combat plans into retained SVG and animation descriptions.
 *
 * SvgChoreographyPainter owns its three installation roots: foreground events
 * (in the `transient-events` layer), connectors and route underlays (both in the
 * `strike-band` layer, above stones and under bodies). For each admitted plan
 * event it makes the event's group, connector and route groups with their
 * metadata, then sends the event to the look module that draws it (its
 * records(), build(records, doc, mount) and later update(nodes, records)):
 * `looks/basics.js` and `looks/ultimates.js` for activations,
 * `looks/health-text.js` for health changes and rest heals, `looks/statuses.js`
 * for status changes and plain pulses, `looks/requiems.js` for falls and
 * returns (with the waystone Call from `looks/spawns.js`). It also paints the
 * plan's derived rows: Herald Cards (`looks/heralds.js`, in their own root in
 * the `heralds` layer above every body), Last Seen rings (`looks/fog.js`) and
 * Focus Tally beads (`looks/focus-tally.js`). The death card, respawn wave and
 * Fallen feed rows are page chrome (requiems.js gives no arena record for them)
 * and Home Fall glows are drawn by the durable Red Zone look, so none of them is
 * painted here. It registers each event's tooltip from that look and returns
 * animation descriptions for CombatChoreographer to run. Looks that time their
 * own marks (Ultimates, falls, returns, heralds, Last Seen, Focus Tally, Basics)
 * get no generic group fade.
 *
 * The one thing it remembers between plans is the herald row of the tick it
 * last painted for one audience (carryFromEarlierTicks), so a Herald Card holds
 * through the next tick. Only a move of exactly one tick for the same audience
 * keeps it; a seek, skip, view or recipient change forgets it, so such a paint
 * equals a fresh render.
 *
 * All event identity, positions, phases and layout come from the supplied plan
 * (or, for held rows, an earlier plan of the same audience); this module does
 * not query simulator state, infer hidden bodies or submit actions. Agent POV:
 * it draws only events the plans admitted for the audience.
 */
import { createSemanticDescriptor, registerTooltipOwner } from "./tooltip.js";
import * as basics from "./looks/basics.js";
import * as focusTally from "./looks/focus-tally.js";
import * as fog from "./looks/fog.js";
import * as healthText from "./looks/health-text.js";
import * as heralds from "./looks/heralds.js";
import {
  animationSpec,
  assignLayoutKey,
  cssIdentifier,
  effectNode,
  setEffectAttributes,
} from "./looks/look-support.js";
import * as requiems from "./looks/requiems.js";
import * as spawns from "./looks/spawns.js";
import * as statuses from "./looks/statuses.js";
import * as ultimates from "./looks/ultimates.js";

// The shared event tooltip lives with the looks; it stays importable from here.
export { explainChoreographyEvent } from "./looks/look-support.js";

/**
 * The event looks, in the order the painter asks them. Each look's records()
 * says whether it draws an event; no two looks claim the same event.
 */
const EVENT_LOOKS = Object.freeze([
  basics,
  ultimates,
  healthText,
  statuses,
  requiems,
  heralds,
  fog,
  focusTally,
]);

/** The derived plan rows this painter draws (the others are data only). */
const PAINTED_DERIVED_KINDS = new Set(["herald_card", "last_seen", "focus_tally"]);

/** Pulse semantics whose look times every mark itself (no generic group fade). */
const SELF_TIMED_PULSES = new Set([
  "agent_died",
  "agent_respawned",
  "death_announcement",
]);

/**
 * @typedef {{
 *   plan: JsonRecord,
 *   surface: JsonRecord,
 *   visualFilters: Readonly<Record<string, boolean>> | null,
 *   carry: TickCarry,
 * }} PaintContext
 */

/**
 * Return the records() input one look needs for one event.
 *
 * look is the event's look module; event is the plan row; context carries the
 * plan, the surface (its lookTransform, visualFilters, scene and
 * viewportBounds, all of the same audience) and the carry from this audience's
 * previous tick (see carryFromEarlierTicks). Basics read the switch state and
 * transform; health numbers read the plan, the switch state (Saved Beat, Last
 * Breath, Battle Text Detail and the Champions life style) and the transform
 * (whether the Beyond Death tag fits); falls and returns read the plan, the
 * transform and the switch state (Kill Credit Glint rings the plan's named
 * killers at their END body spots); heralds read the visible arena, the END
 * body spots to avoid, the switches and the previous tick's herald row
 * (`previous`, so a card holds through the next tick); Last Seen rings read
 * the transform, switches and the audience scene (a body still drawn at END
 * gets no ring). Others read only the event. Every input belongs to this
 * audience (Agent POV: its own plan and scene). Pure.
 *
 * @param {typeof EVENT_LOOKS[number]} look
 * @param {JsonRecord} event
 * @param {PaintContext} context
 * @returns {any}
 */
function lookInput(look, event, context) {
  const { plan, surface, visualFilters } = context;
  if (look === basics) {
    return { event, filters: visualFilters, transform: surface.lookTransform ?? null };
  }
  if (look === healthText || look === requiems) {
    return {
      event,
      plan,
      filters: visualFilters,
      transform: surface.lookTransform ?? null,
    };
  }
  if (look === heralds) {
    const avoid = (Array.isArray(plan.bodies) ? plan.bodies : [])
      .filter((body) => body && Array.isArray(body.end))
      .map((body) => {
        const center = surface.worldToScreen(body.end);
        return { x: center.x, y: center.y, radius: surface.worldLengthToScreen(0.5) };
      });
    return {
      event,
      bounds: surface.viewportBounds,
      previous: context.carry.previousHerald,
      avoid,
      filters: visualFilters,
    };
  }
  if (look === fog) {
    return {
      event,
      transform: { worldLengthToScreen: surface.worldLengthToScreen },
      filters: visualFilters,
      scene: surface.scene ?? null,
    };
  }
  return { event };
}

/**
 * Return whether a look times every one of its marks for this event.
 *
 * Such looks push their own fill-backwards animations, so the painter shows
 * their groups at once and adds no generic fade. Pure.
 *
 * @param {typeof EVENT_LOOKS[number]} look
 * @param {JsonRecord} event
 */
function selfTimed(look, event) {
  if (look === ultimates || look === basics) return true;
  if (look === heralds || look === fog || look === focusTally) return true;
  return (
    look === requiems &&
    event.kind === "semantic_pulse" &&
    SELF_TIMED_PULSES.has(String(event.cueSemantic))
  );
}

/**
 * Return the look module that draws event, or null when no look draws it.
 *
 * event is one admitted plan event. A "herald_card" row always goes to
 * looks/heralds.js (its card may come from the previous tick's row). For other
 * rows each look's records({event}) owns the rule for which events it draws; the
 * first look with a record wins. Return that look, or null for kinds no look
 * draws (the painter then skips the event).
 *
 * @param {JsonRecord} event
 * @returns {typeof EVENT_LOOKS[number] | null}
 */
function eventLook(event) {
  // A herald row whose own falls make no card can still draw the held card.
  if (event?.kind === "herald_card") return heralds;
  for (const look of EVENT_LOOKS) {
    if (look.records({ event }).length > 0) {
      return look;
    }
  }
  return null;
}

/**
 * What the painter remembers about the tick it last painted for one audience.
 *
 * audienceKey is the plan's audienceKey (session, view kind and Agent POV
 * recipient); step its simulatorStep; herald that tick's drawn herald_card row,
 * or null when it had none. The row comes from that plan, so it belongs to that
 * plan's audience.
 *
 * @typedef {Readonly<{
 *   audienceKey: string,
 *   step: number,
 *   herald: JsonRecord | null,
 *   previousHerald: JsonRecord | null,
 * }>} TickMemory
 * previousHerald is the herald row this tick itself carried (kept so a repaint
 * of the same tick carries the same row).
 */

/**
 * What one tick's paint takes from the same audience's previous tick.
 *
 * previousHerald is the herald_card row of exactly one tick before, or null.
 * heldRows are extra rows the painter draws because this tick's plan has none
 * of its own: a held Herald Card row (kind "herald_card", no falls, held true,
 * eventId `<transitionId>:herald:held`) when the plan has no herald row but the
 * previous tick had one; looks/heralds.js draws it as the previous card.
 *
 * @typedef {Readonly<{
 *   previousHerald: JsonRecord | null,
 *   heldRows: ReadonlyArray<JsonRecord>,
 * }>} TickCarry
 */

/** The carry of a tick with nothing remembered: no previous row, no held rows. */
const NO_CARRY = /** @type {TickCarry} */ (
  Object.freeze({ previousHerald: null, heldRows: Object.freeze([]) })
);

/**
 * Work out what a plan's paint carries from the same audience's previous tick.
 *
 * A Herald Card enters at the Fall beat and holds through the next tick, but a
 * plan only holds its own tick. So the painter remembers the herald row of the
 * tick it last painted and hands it back here as `previous` for looks/heralds.js.
 *
 * memory is the painter's last TickMemory, or null for none. plan is the plan
 * about to be painted; its audienceKey (string) and simulatorStep (integer) say
 * whose tick it is.
 *
 * Rules: the same audienceKey and exactly one tick later carries memory's herald
 * row; the same audienceKey and the same simulatorStep (a repaint of the same
 * tick) carries what that tick carried before; anything else (a seek, a skip
 * back or forward, a new episode, a view or recipient change, or a plan without
 * audienceKey or simulatorStep) carries nothing, so the paint equals a fresh
 * render.
 *
 * Agent POV: memory only returns a row from a plan with the same audience key,
 * which the plan already limited to the falls that team saw; a recipient or
 * view change clears it, so Oracle -> POV -> Oracle paints fresh.
 *
 * Returns {memory, carry}: memory is the TickMemory to keep for the next paint
 * (null when the plan has no audienceKey or simulatorStep) and carry the
 * TickCarry for this paint (NO_CARRY when nothing is carried). Pure: inputs are
 * not changed; no errors are thrown for missing or odd fields.
 *
 * @param {TickMemory | null} memory
 * @param {JsonRecord} plan
 * @returns {{memory: TickMemory | null, carry: TickCarry}}
 */
export function carryFromEarlierTicks(memory, plan) {
  const audienceKey = typeof plan?.audienceKey === "string" ? plan.audienceKey : null;
  const step = Number.isInteger(plan?.simulatorStep)
    ? Number(plan.simulatorStep)
    : null;
  if (audienceKey === null || step === null) {
    return { memory: null, carry: NO_CARRY };
  }
  const sameAudience = memory?.audienceKey === audienceKey;
  const previousHerald = !sameAudience
    ? null
    : step === memory.step + 1
      ? memory.herald
      : step === memory.step
        ? memory.previousHerald
        : null;
  const derived = Array.isArray(plan.derivedEvents) ? plan.derivedEvents : [];
  // Only a drawn row counts: a row that is not spatial (filtered off or dropped
  // for budget) was never shown, so it is never held.
  const herald =
    derived.find(
      (/** @type {JsonRecord} */ row) => row?.kind === "herald_card" && row.spatial,
    ) ?? null;
  const ownRow = derived.some(
    (/** @type {JsonRecord} */ row) => row?.kind === "herald_card",
  );
  const heldRows =
    previousHerald !== null && !ownRow
      ? [
          Object.freeze({
            eventId: `${plan.transitionId}:herald:held`,
            eventType: "agent_died",
            kind: "herald_card",
            count: 0,
            byTeam: Object.freeze({ 1: 0, 2: 0 }),
            falls: Object.freeze([]),
            spatial: true,
            persistent: previousHerald.persistent === true,
            phaseStart: 0,
            phaseEnd: Number(
              plan.beats?.explanationEnd ?? previousHerald.phaseEnd ?? 0,
            ),
            held: true,
            heldFromEventId: previousHerald.eventId,
          }),
        ]
      : [];
  return {
    memory: Object.freeze({ audienceKey, step, herald, previousHerald }),
    carry:
      previousHerald === null
        ? NO_CARRY
        : Object.freeze({ previousHerald, heldRows: Object.freeze(heldRows) }),
  };
}

/**
 * @typedef {Record<string, any>} JsonRecord
 * @typedef {{
 *   motionMode: "normal" | "reduced" | "off",
 *   renderPolicy: "live_once" | "replay_animated" | "replay_static",
 *   settled: boolean,
 *   persistentOnly: boolean,
 *   retainTransientOnSettle?: boolean,
 * }} PainterOptions
 * @typedef {{
 *   element: Element,
 *   keyframes: Keyframe[] | PropertyIndexedKeyframes,
 *   options: KeyframeAnimationOptions,
 *   id: string,
 * }} AnimationSpec
 * @typedef {{
 *   root: SVGElement,
 *   connectorRoot: SVGElement,
 *   routeRoot: SVGElement,
 *   heraldRoot: SVGElement,
 *   eventNodes: Map<
 *     string,
 *     {
 *       group: SVGElement,
 *       connector: SVGElement | null,
 *       underlay: SVGElement | null,
 *       event: JsonRecord,
 *     }
 *   >,
 *   animationSpecs: readonly AnimationSpec[],
 *   nodeCount: number,
 *   persistentNodeCount: number,
 *   heldNodeCount: number,
 *   heldPersistentNodeCount: number,
 *   carry: TickCarry,
 *   motionMode: "normal" | "reduced" | "off",
 *   renderPolicy: "live_once" | "replay_animated" | "replay_static",
 *   retainTransientOnSettle: boolean,
 * }} PainterInstallation
 */

/**
 * Own retained SVG for one controller's authorized combat explanations.
 *
 * install appends four owned roots (events, connectors, routes and heralds) and
 * returns their mutable installation record. The painter remembers the herald
 * row of the tick it last painted for one audience (see carryFromEarlierTicks),
 * so a Herald Card holds through the next tick; a seek, skip, view or recipient
 * change forgets it.
 * settle keeps persistent or explicitly retained explanations; clear removes only
 * these roots. reproject updates supported geometry in place so browser animations
 * keep their element identity. The controller owns actual animation handles/clocks.
 */
export class SvgChoreographyPainter {
  /**
   * The rows of the ticks this painter last painted, or null. Set only by a
   * successful install; read by the next install (see carryFromEarlierTicks).
   *
   * @type {TickMemory | null}
   */
  #tickMemory = null;

  /**
   * Build and append one bounded SVG installation from an authorized plan.
   *
   * plan supplies ordered events, derived rows, identity, timing and resource
   * bounds. surface gives ownerDocument, foreground layer, optional routeLayer
   * (defaults to layer), optional heraldLayer (defaults to layer), viewportKey and,
   * for the looks, optional lookTransform, visualFilters and scene (all of the
   * plan's audience). options supplies motionMode, renderPolicy, settled and
   * persistentOnly; retainTransientOnSettle is honored only when initially settled.
   *
   * Paint plan.events, then the held row carried from this audience's previous
   * tick (a held Herald Card; see carryFromEarlierTicks), then the drawn derived
   * rows (herald_card, last_seen, focus_tally). A herald row also gets the
   * previous tick's herald row. Skip nonspatial/unknown rows and, in persistent-only
   * mode, nonpersistent rows.
   * Return a mutable installation with its roots, event map (held rows included),
   * frozen animation specs, counted nodes and carry. nodeCount and
   * persistentNodeCount count the plan's own rows (which plan.bounds covers);
   * heldNodeCount and heldPersistentNodeCount count the held rows, which are
   * bounded by heralds.BUDGET.nodesPerRecord per held row instead (a held row
   * over it is dropped, never the tick). Specs are not started here. Exceeding
   * planned node/animation bounds throws RangeError before the roots are appended. DOM/descriptor errors propagate.
   * Side effect: a successful install replaces the painter's tick memory with this
   * plan's rows. The caller owns clear/settle and must remove an old authority
   * before installing a replacement.
   *
   * @param {JsonRecord} plan
   * @param {JsonRecord} surface
   * @param {PainterOptions} options
   * @returns {PainterInstallation}
   */
  install(plan, surface, options) {
    const root = effectNode(surface.ownerDocument, "g", {
      class: "combat-choreography",
      role: "group",
      "aria-label": "Authorized combat event summaries",
      "data-epoch-key": plan.epochKey,
      "data-authorization-key": plan.authorizationKey,
      "data-event-fingerprint": plan.fingerprint,
      "data-paint-key": plan.paintKey,
      "data-motion-mode": options.motionMode,
      "data-render-policy": options.renderPolicy,
      "data-state":
        options.settled || options.motionMode === "off" ? "settled" : "playing",
      "data-viewport-key": surface.viewportKey,
    });
    const routeRoot = effectNode(surface.ownerDocument, "g", {
      class: "combat-choreography-routes",
      role: "group",
      "aria-label": "Authorized combat event routes",
      "data-epoch-key": plan.epochKey,
      "data-authorization-key": plan.authorizationKey,
      "data-event-fingerprint": plan.fingerprint,
      "data-paint-key": plan.paintKey,
      "data-motion-mode": options.motionMode,
      "data-render-policy": options.renderPolicy,
      "data-state":
        options.settled || options.motionMode === "off" ? "settled" : "playing",
      "data-viewport-key": surface.viewportKey,
    });
    const heraldRoot = effectNode(surface.ownerDocument, "g", {
      class: "combat-choreography-heralds",
      role: "group",
      "aria-label": "Match moments",
      "data-epoch-key": plan.epochKey,
      "data-render-policy": options.renderPolicy,
      "data-viewport-key": surface.viewportKey,
    });
    const connectorRoot = effectNode(surface.ownerDocument, "g", {
      class: "combat-choreography-connectors",
      "aria-hidden": "true",
      "data-epoch-key": plan.epochKey,
      "data-authorization-key": plan.authorizationKey,
      "data-event-fingerprint": plan.fingerprint,
      "data-paint-key": plan.paintKey,
      "data-motion-mode": options.motionMode,
      "data-render-policy": options.renderPolicy,
      "data-state":
        options.settled || options.motionMode === "off" ? "settled" : "playing",
      "data-viewport-key": surface.viewportKey,
    });
    /** @type {Map<
     *   string,
     *   {
     *     group: SVGElement,
     *     connector: SVGElement | null,
     *     underlay: SVGElement | null,
     *     event: JsonRecord,
     *   }
     * >} */
    const eventNodes = new Map();
    /** @type {AnimationSpec[]} */
    const animationSpecs = [];

    const visualFilters = surface.visualFilters ?? null;
    const { memory, carry } = carryFromEarlierTicks(this.#tickMemory, plan);
    /** @type {PaintContext} */
    const context = { plan, surface, visualFilters, carry };
    const derived = (
      Array.isArray(plan.derivedEvents) ? plan.derivedEvents : []
    ).filter((/** @type {JsonRecord} */ event) =>
      PAINTED_DERIVED_KINDS.has(event.kind),
    );
    for (const event of [...plan.events, ...carry.heldRows, ...derived]) {
      if (
        !event.spatial ||
        event.kind === "unknown" ||
        (options.persistentOnly && !event.persistent)
      ) {
        continue;
      }
      const rendered = this.#renderEvent(
        surface.ownerDocument,
        event,
        context,
        options,
        animationSpecs,
      );
      if (!rendered) {
        continue;
      }
      const { group, connector, underlay } = rendered;
      eventNodes.set(event.eventId, { group, connector, underlay, event });
      if (connector) {
        connectorRoot.append(connector);
      }
      if (underlay) {
        routeRoot.append(underlay);
      }
      (event.kind === "herald_card" ? heraldRoot : root).append(group);
    }
    const heldIds = new Set(carry.heldRows.map((row) => row.eventId));
    /** @type {(entry: {group: SVGElement, connector: SVGElement | null, underlay: SVGElement | null}) => number} */
    const entryNodes = ({ group, connector, underlay }) =>
      group.querySelectorAll("*").length +
      1 +
      (connector ? connector.querySelectorAll("*").length + 1 : 0) +
      (underlay ? underlay.querySelectorAll("*").length + 1 : 0);
    let heldNodeCount = 0;
    let heldPersistentNodeCount = 0;
    for (const [eventId, entry] of eventNodes) {
      if (!heldIds.has(eventId)) continue;
      const count = entryNodes(entry);
      // A held card is a courtesy: over its look budget it is dropped, never the tick.
      if (count > heralds.BUDGET.nodesPerRecord) {
        entry.group.remove();
        entry.connector?.remove();
        entry.underlay?.remove();
        eventNodes.delete(eventId);
        continue;
      }
      heldNodeCount += count;
      if (entry.event.persistent) heldPersistentNodeCount += count;
    }
    const nodeCount =
      root.querySelectorAll("*").length +
      connectorRoot.querySelectorAll("*").length +
      routeRoot.querySelectorAll("*").length +
      heraldRoot.querySelectorAll("*").length +
      3 -
      heldNodeCount;
    const persistentNodeCount = Array.from(eventNodes.entries())
      .filter(([eventId, { event }]) => event.persistent && !heldIds.has(eventId))
      .reduce((count, [, entry]) => count + entryNodes(entry), 0);
    if (nodeCount > Number(plan.bounds?.nodes ?? 512)) {
      throw new RangeError("choreography painter exceeded the planned node bound.");
    }
    if (animationSpecs.length > Number(plan.bounds?.animations ?? 512)) {
      throw new RangeError(
        "choreography painter exceeded the planned animation bound.",
      );
    }
    if (persistentNodeCount > Number(plan.bounds?.persistentNodes ?? 64)) {
      // Name what was kept, so an over-budget look is easy to find.
      const kept = Array.from(eventNodes.values())
        .filter(({ event }) => event.persistent && !heldIds.has(event.eventId))
        .map(
          ({ event, group, connector, underlay }) =>
            `${event.kind}${event.cueSemantic ? `:${event.cueSemantic}` : ""}=${
              group.querySelectorAll("*").length +
              1 +
              (connector ? connector.querySelectorAll("*").length + 1 : 0) +
              (underlay ? underlay.querySelectorAll("*").length + 1 : 0)
            }`,
        )
        .join(", ");
      throw new RangeError(
        `choreography painter exceeded the planned persistent node bound (${persistentNodeCount} > ${Number(plan.bounds?.persistentNodes ?? 64)}: ${kept}).`,
      );
    }
    (surface.routeLayer ?? surface.layer).append(connectorRoot, routeRoot);
    surface.layer.append(root);
    (surface.heraldLayer ?? surface.layer).append(heraldRoot);
    /** @type {PainterInstallation} */
    const installation = {
      root,
      connectorRoot,
      routeRoot,
      heraldRoot,
      eventNodes,
      animationSpecs: Object.freeze(animationSpecs),
      nodeCount,
      persistentNodeCount,
      heldNodeCount,
      heldPersistentNodeCount,
      carry,
      motionMode: options.motionMode,
      renderPolicy: options.renderPolicy,
      retainTransientOnSettle:
        options.settled && options.retainTransientOnSettle === true,
    };
    if (options.settled) {
      this.settle(installation);
    }
    this.#tickMemory = memory;
    return installation;
  }

  /**
   * Remove only the four roots owned by installation.
   *
   * installation may be null; _reason is an optional compatibility argument and is
   * unused. Return undefined. This removes DOM but does not cancel animation handles
   * or clear the installation object; the controller owns those resources.
   *
   * @param {PainterInstallation | null} installation
   * @param {string} [_reason]
   */
  clear(installation, _reason) {
    installation?.root?.remove();
    installation?.connectorRoot?.remove();
    installation?.routeRoot?.remove();
    installation?.heraldRoot?.remove();
  }

  /**
   * Mark this installation settled and remove its nonretained transient nodes.
   *
   * installation is the mutable result of install. Persistent events, or all events
   * when retainTransientOnSettle is true, become fully opaque with settled metadata.
   * Other event groups/connectors/underlays are removed from DOM and eventNodes.
   * Return undefined; animation handles and original count/spec fields are unchanged.
   *
   * @param {PainterInstallation} installation
   */
  settle(installation) {
    installation.root.dataset.state = "settled";
    installation.connectorRoot.dataset.state = "settled";
    installation.routeRoot.dataset.state = "settled";
    for (const [eventId, record] of installation.eventNodes) {
      const group = record.group;
      const persistent = group.dataset.persistent === "true";
      if (persistent || installation.retainTransientOnSettle) {
        group.dataset.settled = "true";
        group.setAttribute("opacity", "1");
        if (record.connector) {
          record.connector.dataset.settled = "true";
          record.connector.setAttribute("opacity", "1");
        }
        if (record.underlay) {
          record.underlay.dataset.settled = "true";
          record.underlay.setAttribute("opacity", "1");
        }
      } else {
        group.remove();
        record.connector?.remove();
        record.underlay?.remove();
        installation.eventNodes.delete(eventId);
      }
    }
  }

  /**
   * Apply a new layout to retained event elements without restarting clocks.
   *
   * installation is active, plan describes the same authorized content with updated
   * geometry, and surface provides the new viewportKey. Update viewport metadata and
   * supported event geometry; remove existing events that are now absent/nonspatial.
   * Held rows keep the carry they were installed with (installation.carry), so a
   * held card keeps the layout of the tick it came from.
   * No new event subtree or animation is installed here. Return undefined. The caller
   * owns identity compatibility and any rebuild needed for changed content.
   *
   * @param {PainterInstallation} installation
   * @param {JsonRecord} plan
   * @param {JsonRecord} surface
   */
  reproject(installation, plan, surface) {
    installation.root.dataset.viewportKey = surface.viewportKey;
    installation.connectorRoot.dataset.viewportKey = surface.viewportKey;
    installation.routeRoot.dataset.viewportKey = surface.viewportKey;
    const carry = installation.carry ?? NO_CARRY;
    const nextEvents = /** @type {JsonRecord[]} */ ([
      ...plan.events,
      ...carry.heldRows,
      ...(Array.isArray(plan.derivedEvents) ? plan.derivedEvents : []),
    ]);
    const nextById = new Map(nextEvents.map((event) => [event.eventId, event]));
    /** @type {PaintContext} */
    const context = {
      plan,
      surface,
      visualFilters: surface.visualFilters ?? null,
      carry,
    };
    for (const [eventId, record] of installation.eventNodes) {
      const next = nextById.get(eventId);
      if (!next?.spatial) {
        record.group.remove();
        record.connector?.remove();
        record.underlay?.remove();
        installation.eventNodes.delete(eventId);
        continue;
      }
      this.#updateGeometry(
        record.group,
        record.connector,
        record.underlay,
        next,
        context,
      );
      record.event = next;
    }
  }

  /**
   * Create one event subtree and append its animation descriptions.
   *
   * ownerDocument creates SVG; event is an admitted spatial plan row; context
   * carries the plan (phases) and the surface the looks read; options supplies
   * motion/render/settlement choices; animationSpecs is the caller's mutable output
   * array. Make the event, connector and route groups with
   * their metadata, then let the event's look module draw into them (see eventLook).
   * Return {group, connector, underlay}, with null unused layers, or null for rows
   * no look draws, for cooldown-start rows and for an Ultimate the audience may not
   * see drawn (variant "none"). A return also gets its waystone's Call
   * (looks/spawns.js) in the same group. Register the look's semantic tooltip and
   * layout metadata. Reduced motion rescales phase times; settled and off modes add
   * no generic fade specs, and neither do looks that time their own marks (see
   * selfTimed), whose groups show at once. No Web Animation is started here.
   *
   * @param {Document} ownerDocument
   * @param {JsonRecord} event
   * @param {PaintContext} context
   * @param {PainterOptions} options
   * @param {AnimationSpec[]} animationSpecs
   * @returns {{
   *   group: SVGElement,
   *   connector: SVGElement | null,
   *   underlay: SVGElement | null,
   * } | null}
   */
  #renderEvent(ownerDocument, event, context, options, animationSpecs) {
    const plan = context.plan;
    if (event.kind === "semantic_pulse" && event.cueSemantic === "cooldown_started") {
      return null;
    }
    const look = eventLook(event);
    if (look === null) {
      return null;
    }
    const records = look.records(lookInput(look, event, context));
    if (records.length === 0 || records[0]?.variant === "none") {
      // Nothing the audience may see: no empty focusable group, no tooltip.
      return null;
    }
    const group = effectNode(ownerDocument, "g", {
      class: `combat-effect combat-effect--${cssIdentifier(event.kind)}`,
      opacity: options.settled || options.motionMode === "off" ? 1 : 0,
      "data-event-id": event.eventId,
      "data-event-type": event.eventType,
      "data-phase": phaseFor(event),
      "data-persistent": Boolean(event.persistent),
    });
    assignSlot(group, "source", event.sourceSlot);
    assignSlot(group, "target", event.targetSlot);
    assignSlot(group, "recipient", event.recipientSlot);
    assignSlot(group, "actor", event.actorSlot);
    assignPresentationKey(group, "source", event.sourcePresentationKey);
    assignPresentationKey(group, "target", event.targetPresentationKey);
    assignPresentationKey(group, "recipient", event.recipientPresentationKey);
    assignPresentationKey(group, "actor", event.actorPresentationKey);
    if (typeof event.tokenId === "string") {
      group.dataset.tokenId = event.tokenId;
    }
    if (typeof event.sourceClass?.cssKey === "string") {
      group.dataset.sourceClass = event.sourceClass.cssKey;
    }
    if (typeof event.outcome === "string") {
      group.dataset.outcome = event.outcome;
    }
    if (typeof event.impactSemantic === "string") {
      group.dataset.impactSemantic = event.impactSemantic;
    }
    if (typeof event.lifecycle === "string") {
      group.dataset.lifecycle = event.lifecycle;
    }
    if (Number.isInteger(event.lane)) {
      group.dataset.lane = String(event.lane);
    }
    if (Number.isInteger(event.laneCount)) {
      group.dataset.laneCount = String(event.laneCount);
    }
    if (typeof event.component === "string") {
      group.dataset.component = event.component;
    }
    if (typeof event.cueSemantic === "string") {
      group.dataset.cueSemantic = event.cueSemantic;
    }
    if (Number.isInteger(event.teamId)) {
      group.dataset.teamId = String(event.teamId);
    }
    if (Number.isInteger(event.teamIndex)) {
      group.dataset.teamIndex = String(event.teamIndex);
    }
    if (typeof event.teamSide === "string") {
      group.dataset.teamSide = event.teamSide;
    }
    if (typeof event.label === "string") {
      group.dataset.label = event.label;
    }
    if (Number.isInteger(event.agentSlot)) {
      group.dataset.agentSlot = String(event.agentSlot);
    }
    assignPresentationKey(group, "agent", event.agentPresentationKey);
    if (typeof event.movementMaskValue === "boolean") {
      group.dataset.movementMaskValue = String(event.movementMaskValue);
    }
    if (typeof event.pairMaskValue === "boolean") {
      group.dataset.pairMaskValue = String(event.pairMaskValue);
    }
    if (Number.isInteger(event.durationBefore)) {
      group.dataset.durationBefore = String(event.durationBefore);
    }
    if (Number.isInteger(event.durationAfter)) {
      group.dataset.durationAfter = String(event.durationAfter);
    }
    if (Array.isArray(event.applicationEventIds)) {
      group.dataset.applicationEventIds = JSON.stringify(event.applicationEventIds);
    }
    if (Array.isArray(event.atomicEventIds)) {
      group.dataset.atomicEventIds = JSON.stringify(event.atomicEventIds);
    }
    const connector = effectNode(ownerDocument, "g", {
      class: `combat-connector-effect combat-connector-effect--${cssIdentifier(event.kind)}`,
      "aria-hidden": "true",
      opacity: options.settled || options.motionMode === "off" ? 1 : 0,
    });
    copyEventMetadata(group, connector);
    const underlay = event.route
      ? effectNode(ownerDocument, "g", {
          class: `combat-route-effect combat-route-effect--${cssIdentifier(event.kind)}`,
          "aria-hidden": "true",
          opacity: options.settled || options.motionMode === "off" ? 1 : 0,
        })
      : null;
    if (underlay) {
      copyEventMetadata(group, underlay);
      assignLayoutKey(underlay, event.routeLayoutKey ?? event.route?.layoutKey);
      if (Number.isInteger(event.routeLane ?? event.route?.lane)) {
        underlay.dataset.lane = String(event.routeLane ?? event.route.lane);
      }
    }

    const mount = {
      group,
      connector,
      underlay,
      plan,
      options,
      animationSpecs,
    };
    look.build(/** @type {any} */ (records), ownerDocument, mount);
    if (event.kind === "semantic_pulse" && event.cueSemantic === "agent_respawned") {
      const transform = context.surface.lookTransform ?? null;
      spawns.build(
        spawns.records({
          part: "call",
          event,
          transform,
          filters: context.visualFilters,
        }),
        ownerDocument,
        mount,
      );
    }
    this.#syncRouteBridgeGaps(underlay, event.route);
    this.#registerEventExplanation(group, underlay, event, look, records);
    this.#applySpatialDisposition(group, event);
    const renderedConnector = connector.childElementCount > 0 ? connector : null;

    if (selfTimed(look, event)) {
      group.setAttribute("opacity", "1");
      connector.setAttribute("opacity", "1");
      underlay?.setAttribute("opacity", "1");
      return { group, connector: renderedConnector, underlay };
    }
    if (!options.settled && options.motionMode !== "off") {
      const reduced = options.motionMode === "reduced";
      const authoredPhaseStart = Number(event.phaseStart ?? 0);
      const authoredPhaseEnd = Number(event.phaseEnd ?? plan.phases.total);
      const reducedScale =
        Number(plan.phases.reducedTotal ?? 220) /
        Math.max(Number(plan.phases.total ?? 900), 1);
      const phaseStart = reduced
        ? authoredPhaseStart * reducedScale
        : authoredPhaseStart;
      const phaseEnd = reduced ? authoredPhaseEnd * reducedScale : authoredPhaseEnd;
      const activationHasImpact =
        event.kind === "activation" && group.querySelector(".combat-impact") !== null;
      const targets = underlay
        ? event.kind === "activation"
          ? [{ element: underlay, part: "route" }]
          : [
              { element: underlay, part: "route" },
              { element: group, part: "group" },
            ]
        : activationHasImpact
          ? []
          : [{ element: group, part: "group" }];
      if (renderedConnector) {
        targets.push({ element: renderedConnector, part: "connector" });
      }
      if (event.kind === "activation" && (underlay || activationHasImpact)) {
        group.setAttribute("opacity", "1");
      }
      const duration = phaseEnd - phaseStart;
      if (!(duration > 0)) {
        return { group, connector: renderedConnector, underlay };
      }
      for (const target of targets) {
        animationSpecs.push(
          animationSpec(
            target.element,
            eventKeyframes(event, options.motionMode),
            {
              delay: phaseStart,
              duration,
              easing: "ease-out",
              fill: "both",
            },
            plan,
            event,
            target.part,
          ),
        );
      }
    }
    return { group, connector: renderedConnector, underlay };
  }

  /**
   * Bind the look's tooltip to a visible event and its optional route group.
   *
   * group is the keyboard-focusable owner; underlay may be null; event already
   * holds authorized facts and filter pieces; look is the event's look module and
   * records its records for this event. The look's tooltip(record) picks the
   * activation, health or shared event explanation; a look that gives no tip
   * leaves the group unfocusable and hidden from assistive tech. Otherwise set the
   * accessible label and register both hit surfaces (the route copy is an
   * "accepted-route" for activations). Return undefined.
   * Identity and data validation errors propagate; nothing is looked up.
   *
   * @param {SVGElement} group
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   * @param {typeof EVENT_LOOKS[number]} look
   * @param {ReadonlyArray<any>} records the look's records for this event
   */
  #registerEventExplanation(group, underlay, event, look, records) {
    const explanation =
      /** @type {ReturnType<typeof createSemanticDescriptor> | null} */ (
        look.tooltip(records[0])
      );
    if (explanation === null || explanation === undefined) {
      // A look with no tip (for example a Focus Tally bead) stays unfocusable.
      group.setAttribute("aria-hidden", "true");
      return;
    }
    setEffectAttributes(group, {
      role: "img",
      tabindex: "0",
      "aria-label": explanation.title,
    });
    registerTooltipOwner(group, explanation);
    if (underlay) {
      registerTooltipOwner(
        underlay,
        createSemanticDescriptor({
          ...explanation,
          kind: event.kind === "activation" ? "accepted-route" : explanation.kind,
          id: `${explanation.id}:route`,
          anchor: "pointer",
        }),
      );
    }
  }

  /**
   * Replace crossing backplates while preserving the continuous route hit path.
   *
   * underlay may be null and route may omit bridgeGaps. Remove prior backplates, then
   * add circles only for finite positive-gap allocator records before the visible path.
   * Invalid gap rows are skipped. Return undefined; no crossings are discovered here.
   *
   * @param {SVGElement | null} underlay
   * @param {JsonRecord | null | undefined} route
   */
  #syncRouteBridgeGaps(underlay, route) {
    if (!underlay) {
      return;
    }
    for (const bridge of underlay.querySelectorAll(".combat-route__bridge-backplate")) {
      bridge.remove();
    }
    const visibleRoute = underlay.querySelector(".combat-route__path");
    if (!(visibleRoute instanceof SVGElement) || !Array.isArray(route?.bridgeGaps)) {
      return;
    }
    for (const bridgeGap of route.bridgeGaps) {
      if (
        !bridgeGap ||
        !Number.isFinite(bridgeGap.at?.x) ||
        !Number.isFinite(bridgeGap.at?.y) ||
        !Number.isFinite(bridgeGap.gap) ||
        bridgeGap.gap <= 0
      ) {
        continue;
      }
      const bridge = effectNode(underlay.ownerDocument, "circle", {
        class: "combat-route__bridge-backplate",
        "aria-hidden": "true",
        cx: bridgeGap.at.x,
        cy: bridgeGap.at.y,
        r: bridgeGap.gap / 2,
        "data-bridge-with-layout-key": bridgeGap.withLayoutKey,
        "data-bridge-gap": bridgeGap.gap,
      });
      visibleRoute.before(bridge);
    }
  }

  /**
   * Update supported retained event geometry from a compatible new plan row.
   *
   * group, optional connector and optional underlay are existing SVG nodes; event
   * holds the new allocated points and route metadata. Write the display
   * disposition and route layout key, let the event's look move its own nodes
   * (its update(nodes, records)), then redo the route crossing backplates.
   * Existing animations keep their elements. Return undefined; a content or
   * authority change needs a new installation instead.
   *
   * @param {SVGElement} group
   * @param {SVGElement | null} connector
   * @param {SVGElement | null} underlay
   * @param {JsonRecord} event
   * @param {PaintContext} context the new plan and surface the looks read
   */
  #updateGeometry(group, connector, underlay, event, context) {
    this.#applySpatialDisposition(group, event);
    if (underlay) {
      assignLayoutKey(underlay, event.routeLayoutKey ?? event.route?.layoutKey);
      if (Number.isInteger(event.routeLane ?? event.route?.lane)) {
        underlay.dataset.lane = String(event.routeLane ?? event.route.lane);
      }
    }
    const look = eventLook(event);
    look?.update(
      /** @type {any} */ ([group, connector, underlay]),
      /** @type {any} */ (look.records(lookInput(look, event, context))),
    );
    this.#syncRouteBridgeGaps(underlay, event.route);
  }

  /**
   * Publish the plan's display disposition and clear stale hidden attributes.
   *
   * group is retained SVG; event supplies cue, impact or source disposition in that
   * priority order, defaulting to rendered. Set data-spatial-disposition and remove
   * visibility/aria-hidden. Return undefined. This helper records layout state; it
   * does not suppress a row or grant new spatial authority.
   *
   * @param {SVGElement} group
   * @param {Record<string, any>} event
   */
  #applySpatialDisposition(group, event) {
    const disposition = String(
      event.cueDisposition ??
        event.impactDisposition ??
        event.sourceDisposition ??
        "rendered",
    );
    group.dataset.spatialDisposition = disposition;
    group.removeAttribute("visibility");
    group.removeAttribute("aria-hidden");
  }
}

/**
 * Attach an integer value as the role-specific internal slot dataset field.
 *
 * group is an SVG node and role names source/target/etc. Nonintegers do nothing;
 * no range/roster validation is performed. This metadata never supplies display IDs.
 *
 * @param {SVGElement} group
 * @param {string} role
 * @param {unknown} value
 */
function assignSlot(group, role, value) {
  if (Number.isInteger(value)) {
    group.dataset[`${role}Slot`] = String(value);
  }
}

/**
 * Attach a nonempty string value to the role-specific presentation-key field.
 *
 * group is SVG and role names the relationship. Other values do nothing; old values
 * are not removed. The caller already owns disclosure and key validity.
 *
 * @param {SVGElement} group
 * @param {string} role
 * @param {unknown} value
 */
function assignPresentationKey(group, role, value) {
  if (typeof value === "string" && value) {
    group.dataset[`${role}PresentationKey`] = value;
  }
}

/**
 * Copy every data-* attribute from source to target.
 *
 * Both are owned SVG nodes. Other attributes and absent source fields are not copied
 * or removed. Return undefined; the source metadata must already be safe to disclose.
 *
 * @param {SVGElement} source
 * @param {SVGElement} target
 */
function copyEventMetadata(source, target) {
  for (const attribute of source.attributes) {
    if (attribute.name.startsWith("data-")) {
      target.setAttribute(attribute.name, attribute.value);
    }
  }
}

/**
 * Return the event's semantic DOM phase label.
 *
 * event kind/semantic selects regeneration, pulse, outcome, target-only impact or
 * activation. This label is for presentation inspection, not simulator phase logic.
 *
 * @param {Record<string, any>} event
 */
function phaseFor(event) {
  if (event.kind === "regeneration") {
    return "health_regenerated";
  }
  if (event.kind === "semantic_pulse") {
    return String(event.cueSemantic ?? "semantic");
  }
  if (event.kind === "net_health" || event.kind === "status_lifecycle") {
    return "outcome";
  }
  if (event.kind === "activation" && event.presentationKind === "target_only_impact") {
    return "impact";
  }
  return "activation";
}

/**
 * Return opacity keyframes for the planned event and motion mode.
 *
 * event.persistent decides whether the final opacity remains one, except normal
 * net-health fades always end at zero. motionMode is normal/reduced/off; the caller
 * decides whether to schedule these frames. No browser animation starts here.
 *
 * @param {JsonRecord} event
 * @param {"normal" | "reduced" | "off"} motionMode
 * @returns {Keyframe[]}
 */
function eventKeyframes(event, motionMode) {
  if (event.kind === "net_health" && motionMode === "normal") {
    return [
      { opacity: 0 },
      { opacity: 1, offset: 0.18 },
      { opacity: 1, offset: 0.72 },
      { opacity: 0 },
    ];
  }
  return [
    { opacity: 0 },
    { opacity: 1, offset: 0.18 },
    { opacity: 1, offset: event.persistent ? 1 : 0.72 },
    { opacity: event.persistent ? 1 : 0 },
  ];
}
