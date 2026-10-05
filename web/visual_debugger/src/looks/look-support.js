/**
 * @file Shared SVG helpers for the look modules under `src/looks/`.
 *
 * Each look module (one per Runefield part, for example `champions.js` or
 * `basics.js`) owns how its part is drawn. This file owns the small pieces they
 * all need: SVG node creation, attribute writing, identity attributes, allocator
 * metadata and leader lines, the shared event explanation, paint-part checks and
 * animation descriptions. `scene.js`, `choreography-painter.js` and the look
 * modules import it; it imports only shared label and tooltip owners.
 *
 * Two attribute writers exist on purpose, because the durable renderer and the
 * event painter always differed: `setNodeAttributes` (durable parts) writes
 * `false` as the text "false", while `setEffectAttributes` (event parts) removes
 * an attribute whose value is `false`. Moving code between files keeps its own
 * writer so the drawn SVG stays byte for byte the same.
 *
 * Boundary: nothing here reads a frame, a researcher view or a match summary.
 * The helpers only format and draw the values a caller passes in, so Agent POV
 * limits stay with the caller that chose those values. Every node built here is
 * export-safe: flat fills and strokes, opacity and transforms only.
 */
import {
  canonicalAgentIdentity,
  exactAuthorizedAgentIdentityV1,
} from "../agent-identity.js";
import { formatDisplayNumber } from "../display.js";
import { explainDeathAnnouncement } from "../explanations.js";
import { statusLifecyclePresentation } from "../semantic-vocabulary.js";
import { createSemanticDescriptor, registerTooltipOwner } from "../tooltip.js";
import { resolveVisualToken } from "../vocabulary.js";

/** The SVG namespace every look node is created in. */
export const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

/**
 * @typedef {Record<string, any>} JsonRecord
 * @typedef {{x: number, y: number}} ScreenPoint
 * @typedef {{
 *   left: number,
 *   top: number,
 *   right: number,
 *   bottom: number,
 *   width: number,
 *   height: number,
 * }} Rectangle
 * @typedef {{
 *   scale: number,
 *   worldWidth: number,
 *   worldHeight: number,
 *   viewportBounds: Rectangle,
 *   mapBounds: Rectangle,
 *   worldToScreen: (point: {x: number, y: number} | readonly [number, number]) =>
 *     ScreenPoint,
 *   screenToWorld?: (point: {x: number, y: number} | readonly [number, number]) =>
 *     ScreenPoint,
 *   worldLengthToScreen: (length: number) => number,
 * }} LookTransform
 *   The projection for one render (see `createViewportTransform` in `layout.js`).
 *   It already includes the camera frame, so looks draw fixed pixel sizes around
 *   projected points and never scale them by the camera zoom.
 * @typedef {{
 *   scene?: JsonRecord | null,
 *   event?: JsonRecord | null,
 *   transform?: LookTransform | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   beat?: string | null,
 *   audience?: "researcher" | "agent_pov" | null,
 *   [extra: string]: any,
 * }} LookInput
 *   What `records(input)` reads. `scene` is the authorized arena scene for the
 *   current audience (never the researcher view); `event` is one admitted plan
 *   event; `transform` is this render's projection; `filters` is the visual
 *   filter state; `beat` names the beat stage ("end" is the only stage drawn
 *   before the beat clock lands, and the DOM always holds the END state);
 *   `audience` is "researcher" (Oracle View) or "agent_pov". A look may name
 *   extra fields it needs; each look documents its own.
 * @typedef {Readonly<Record<string, any>>} LookRecord
 *   One drawing record: plain frozen data with a `kind` field. It never holds a
 *   DOM node.
 * @typedef {Readonly<{
 *   part: Readonly<Record<string, string>>,
 *   filter: string,
 * }>} PaintPartRegistration
 *   One independently switchable paint part (`part`, the tag the renderers ask
 *   about) and the visual filter id that switches it (`filter`).
 * @typedef {Readonly<{
 *   nodesPerRecord: number,
 *   animationsPerRecord: number,
 *   persistentNodesPerRecord: number,
 * }>} LookBudget
 *   The node and animation allowance of one record, used by the plan's resource
 *   bounds. Durable looks count nodes only.
 * @typedef {{
 *   element: Element,
 *   keyframes: Keyframe[] | PropertyIndexedKeyframes,
 *   options: KeyframeAnimationOptions,
 *   id: string,
 * }} AnimationSpec
 *   A browser animation description that the choreographer starts later.
 * @typedef {{
 *   group: SVGElement,
 *   connector: SVGElement,
 *   underlay: SVGElement | null,
 *   plan: JsonRecord,
 *   options: {
 *     motionMode: "normal" | "reduced" | "off",
 *     renderPolicy?: "live_once" | "replay_animated" | "replay_static",
 *     settled: boolean,
 *   },
 *   animationSpecs: AnimationSpec[],
 * }} EventMount
 *   Where an event look draws: the painter's event group (foreground layer), its
 *   connector group and optional route group (Strike Band layer), the plan for
 *   timing, the paint options and the list that receives animation descriptions.
 */

/** Budget of a look that draws nothing yet. */
export const EMPTY_BUDGET = Object.freeze({
  nodesPerRecord: 0,
  animationsPerRecord: 0,
  persistentNodesPerRecord: 0,
});

/**
 * The plan's shared per-event allowance for event looks: 30 nodes and 4
 * animations per event, summed over a tick (`choreography-plan.js` bounds), and
 * 7 retained nodes for a persistent semantic pulse.
 */
export const EVENT_BUDGET = Object.freeze({
  nodesPerRecord: 30,
  animationsPerRecord: 4,
  persistentNodesPerRecord: 7,
});

/**
 * Return whether value is a non-null object other than an array.
 *
 * This shape check does not validate a whole record or grant any right to show it.
 *
 * @param {unknown} value
 * @returns {value is JsonRecord}
 */
export function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return value unchanged when it is an array, otherwise a new empty array.
 *
 * No element is checked, copied or changed.
 *
 * @param {unknown} value
 * @returns {any[]}
 */
export function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * Return a finite number unchanged, otherwise fallback (default zero).
 *
 * Numeric strings are not converted. The fallback is returned as given.
 *
 * @param {unknown} value
 * @param {number} [fallback]
 * @returns {number}
 */
export function finiteNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Project a world coordinate array through a render's transform.
 *
 * point should be a world [x, y] array; a missing or non-array point, or a
 * nonfinite part, falls back to zero. transform owns the world-to-screen
 * conversion (and the camera frame). Return the projected screen point in CSS
 * pixels. This tolerant drawing helper is not input validation.
 *
 * @param {unknown} point
 * @param {LookTransform} transform
 * @returns {ScreenPoint}
 */
export function screenPoint(point, transform) {
  const values = Array.isArray(point) ? point : [0, 0];
  return transform.worldToScreen([finiteNumber(values[0]), finiteNumber(values[1])]);
}

/**
 * Create a detached SVG node for a durable part.
 *
 * doc creates the node; tagName picks the SVG element; attributes (default {})
 * are written by setNodeAttributes, so `false` becomes the text "false". Return
 * the node. DOM errors propagate. The caller decides where it goes.
 *
 * @param {Document} doc
 * @param {string} tagName
 * @param {Record<string, unknown>} [attributes]
 * @returns {SVGElement}
 */
export function svgNode(doc, tagName, attributes = {}) {
  const element = /** @type {SVGElement} */ (
    doc.createElementNS(SVG_NAMESPACE, tagName)
  );
  setNodeAttributes(element, attributes);
  return element;
}

/**
 * Write attributes on a durable part's node.
 *
 * A null or undefined value removes that attribute; every other value, `false`
 * included, is written as text. Return undefined. This changes the DOM and does
 * not sanitize names or values.
 *
 * @param {Element} element
 * @param {Record<string, unknown>} attributes
 */
export function setNodeAttributes(element, attributes) {
  for (const [name, value] of Object.entries(attributes)) {
    if (value === null || value === undefined) {
      element.removeAttribute(name);
    } else {
      element.setAttribute(name, String(value));
    }
  }
}

/**
 * Create a detached SVG node for an event part.
 *
 * doc creates the node; tagName picks the SVG element; attributes (default {})
 * are written by setEffectAttributes, so `false` removes the attribute. Return
 * the node. DOM errors propagate. The caller decides where it goes.
 *
 * @param {Document} doc
 * @param {string} tagName
 * @param {Record<string, string | number | boolean | null | undefined>} [attributes]
 * @returns {SVGElement}
 */
export function effectNode(doc, tagName, attributes = {}) {
  const element = /** @type {SVGElement} */ (
    doc.createElementNS(SVG_NAMESPACE, tagName)
  );
  setEffectAttributes(element, attributes);
  return element;
}

/**
 * Write attributes on an event part's node.
 *
 * A null, undefined or false value removes that attribute; every other value is
 * written as text. Return undefined. This changes the DOM and does not sanitize
 * names or values.
 *
 * @param {Element} element
 * @param {Record<string, string | number | boolean | null | undefined>} attributes
 */
export function setEffectAttributes(element, attributes) {
  for (const [name, value] of Object.entries(attributes)) {
    if (value === null || value === undefined || value === false) {
      element.removeAttribute(name);
    } else {
      element.setAttribute(name, String(value));
    }
  }
}

/**
 * Format the public identity an agent row already carries ("Agent ID 5").
 *
 * agent may be any value. Return canonicalAgentIdentity(agent).publicIdentity.
 * Layout or global slots are never turned into a public ID here.
 *
 * @param {unknown} agent
 * @returns {string}
 */
export function agentIdentity(agent) {
  return canonicalAgentIdentity(agent).publicIdentity;
}

/**
 * Write only the identity actually present on record into element metadata.
 *
 * A nonempty presentation_key wins and removes data-slot; otherwise an integer
 * global_slot is written and data-presentation-key is removed. With neither,
 * nothing changes. Return undefined. The caller supplies an accepted display row.
 *
 * @param {SVGElement} element
 * @param {JsonRecord} record
 */
export function setDisplayIdentityData(element, record) {
  if (
    typeof record.presentation_key === "string" &&
    record.presentation_key.length > 0
  ) {
    element.dataset.presentationKey = record.presentation_key;
    element.removeAttribute("data-slot");
  } else if (Number.isInteger(record.global_slot)) {
    element.dataset.slot = String(record.global_slot);
    element.removeAttribute("data-presentation-key");
  }
}

/**
 * Return identity attributes from record without inventing a slot.
 *
 * A string presentation_key wins (even an empty one); otherwise an integer
 * global_slot gives data-slot; with neither return an empty object. The input
 * is unchanged; the normalizer owns identity validity.
 *
 * @param {JsonRecord} record
 * @returns {Record<string, string | number>}
 */
export function displayIdentityAttributes(record) {
  return typeof record.presentation_key === "string"
    ? { "data-presentation-key": record.presentation_key }
    : Number.isInteger(record.global_slot)
      ? { "data-slot": record.global_slot }
      : {};
}

/**
 * Copy only the accepted public ID and optional string presentation key.
 *
 * agent supplies those facts. Return a new tooltip input with no layout or global
 * slot, so a renderer-local index can never become an Agent POV fact.
 *
 * @param {JsonRecord} agent
 * @returns {JsonRecord}
 */
export function displayIdentityRecord(agent) {
  return {
    ...(typeof agent.presentation_key === "string"
      ? { presentation_key: agent.presentation_key }
      : {}),
    public_agent_id: agent.public_agent_id,
  };
}

/**
 * Return a CSS name suffix for a lowercase letter, digit and underscore string.
 *
 * Underscores become hyphens. Any other value returns "unknown"; no HTML or CSS
 * selector text is evaluated.
 *
 * @param {unknown} value
 * @returns {string}
 */
export function cssIdentifier(value) {
  return typeof value === "string" && /^[a-z0-9_]+$/.test(value)
    ? value.replaceAll("_", "-")
    : "unknown";
}

/**
 * Attach a nonempty string as element's data-layout-key.
 *
 * Other values do nothing and never clear an older key. Display metadata only.
 *
 * @param {Element} element
 * @param {unknown} value
 */
export function assignLayoutKey(element, value) {
  if (typeof value === "string" && value.length > 0) {
    element.setAttribute("data-layout-key", value);
  }
}

/**
 * Publish allocator metadata on the visible owner of a cue.
 *
 * element receives a nonempty layoutKey, finite bounds edges (CSS pixels), an
 * optional nonempty disposition and a Boolean collisionFree. Missing or invalid
 * pieces are skipped, never cleared. Return undefined. Bounds describe the
 * reserved cue rectangle; decorative leaders may reach outside it. No layout is
 * solved here.
 *
 * @param {Element} element
 * @param {unknown} layoutKey
 * @param {unknown} bounds
 * @param {unknown} disposition
 * @param {unknown} collisionFree
 */
export function assignLayoutPlacement(
  element,
  layoutKey,
  bounds,
  disposition,
  collisionFree,
) {
  assignLayoutKey(element, layoutKey);
  if (bounds && typeof bounds === "object") {
    const rectangle = /** @type {Record<string, unknown>} */ (bounds);
    for (const edge of ["left", "top", "right", "bottom"]) {
      const value = rectangle[edge];
      if (Number.isFinite(value)) {
        element.setAttribute(`data-layout-${edge}`, String(value));
      }
    }
  }
  if (typeof disposition === "string" && disposition.length > 0) {
    element.setAttribute("data-layout-disposition", disposition);
  }
  if (typeof collisionFree === "boolean") {
    element.setAttribute("data-layout-collision-free", String(collisionFree));
  }
}

/**
 * Return whether value gives finite start and end x/y numbers.
 *
 * Extra fields are ignored. A shape check only, not a layout or rights check.
 *
 * @param {unknown} value
 * @returns {value is {start: {x: number, y: number}, end: {x: number, y: number}}}
 */
export function isAllocatedLeader(value) {
  if (value === null || typeof value !== "object") {
    return false;
  }
  const leader = /** @type {Record<string, any>} */ (value);
  return (
    Number.isFinite(leader.start?.x) &&
    Number.isFinite(leader.start?.y) &&
    Number.isFinite(leader.end?.x) &&
    Number.isFinite(leader.end?.y)
  );
}

/**
 * Return the start and end points of a valid leader, or an empty array.
 *
 * leader needs four finite coordinates. The array is new; its points are the
 * caller's own objects.
 *
 * @param {unknown} leader
 * @returns {Array<{x: number, y: number}>}
 */
function allocatedLeaderPoints(leader) {
  if (!isAllocatedLeader(leader)) {
    return [];
  }
  const candidate = /** @type {Record<string, any>} */ (leader);
  return [candidate.start, candidate.end];
}

/**
 * Write a valid leader as local line ends and inspection metadata.
 *
 * element is an SVG line; leader has finite start and end; origin is subtracted
 * from both so absolute screen points become local ones. Set x1/y1/x2/y2, record
 * the points as JSON in data-leader-points and remove any path d. Throw
 * TypeError for invalid leader points. Inputs are unchanged.
 *
 * @param {SVGElement} element
 * @param {Record<string, any>} leader
 * @param {{x: number, y: number}} origin
 */
function setAllocatedLeaderGeometry(element, leader, origin) {
  const points = allocatedLeaderPoints(leader).map(({ x, y }) => ({
    x: x - origin.x,
    y: y - origin.y,
  }));
  const first = points[0];
  const last = points.at(-1);
  if (!first || !last) {
    throw new TypeError("Allocated leader geometry requires two finite points.");
  }
  element.setAttribute("data-leader-points", JSON.stringify(points));
  element.removeAttribute("d");
  setEffectAttributes(element, {
    x1: first.x,
    y1: first.y,
    x2: last.x,
    y2: last.y,
  });
}

/**
 * Create and append a leader line for valid allocator end points.
 *
 * doc creates the line under parent with className. leader must hold finite
 * start and end; otherwise nothing is added and null is returned. origin
 * (default {x: 0, y: 0}) turns absolute screen points into the parent's local
 * points. Return the new line.
 *
 * @param {Document} doc
 * @param {Element} parent
 * @param {string} className
 * @param {unknown} leader
 * @param {{x: number, y: number}} [origin]
 * @returns {SVGElement | null}
 */
export function appendAllocatedLeader(
  doc,
  parent,
  className,
  leader,
  origin = { x: 0, y: 0 },
) {
  if (!isAllocatedLeader(leader)) {
    return null;
  }
  const element = effectNode(doc, "line", {
    class: className,
    "aria-hidden": "true",
  });
  setAllocatedLeaderGeometry(element, leader, origin);
  parent.append(element);
  return element;
}

/**
 * Update or hide an existing leader under parent.
 *
 * parent may be null; selector finds an existing SVG element. An invalid leader
 * hides it; a missing element is not created. A non-line element is swapped for
 * an SVG line with the same class. origin defaults to zero. Return undefined;
 * throw TypeError if the swapped element is not SVG.
 *
 * @param {Element | null} parent
 * @param {string} selector
 * @param {unknown} leader
 * @param {{x: number, y: number}} [origin]
 */
export function syncAllocatedLeader(parent, selector, leader, origin = { x: 0, y: 0 }) {
  if (parent === null) {
    return;
  }
  let element = parent.querySelector(selector);
  if (!(element instanceof SVGElement)) {
    return;
  }
  if (!isAllocatedLeader(leader)) {
    element.setAttribute("visibility", "hidden");
    return;
  }
  if (element.localName !== "line") {
    const replacement = effectNode(element.ownerDocument, "line", {
      class: element.getAttribute("class"),
      "aria-hidden": "true",
    });
    if (!(replacement instanceof SVGElement)) {
      throw new TypeError("Allocated leader replacement is not SVG geometry.");
    }
    element.replaceWith(replacement);
    element = replacement;
  }
  element.setAttribute("visibility", "visible");
  setAllocatedLeaderGeometry(/** @type {SVGElement} */ (element), leader, origin);
}

/**
 * Squeeze long visible SVG text without changing its full text.
 *
 * element holds the text, text gives its character count, maximumLength is a
 * width in CSS pixels and naturalCharacterLimit is the longest text left as it
 * is. Set textLength and lengthAdjust only when both limits ask for it;
 * otherwise remove them. Return undefined. Tooltip text stays complete.
 *
 * @param {Element} element
 * @param {string} text
 * @param {number} maximumLength
 * @param {number} naturalCharacterLimit
 */
export function constrainCompactText(
  element,
  text,
  maximumLength,
  naturalCharacterLimit,
) {
  if ([...text].length > naturalCharacterLimit && maximumLength > 0) {
    setEffectAttributes(element, {
      textLength: maximumLength,
      lengthAdjust: "spacingAndGlyphs",
    });
    return;
  }
  element.removeAttribute("textLength");
  element.removeAttribute("lengthAdjust");
}

/**
 * Return whether an event's named paint part is switched on.
 *
 * An event without an object paintParts keeps the old all-on behaviour.
 * Otherwise only an exact true at part turns it on; a missing key is off. No
 * filter is changed.
 *
 * @param {JsonRecord} event
 * @param {string} part
 * @returns {boolean}
 */
export function paintPartEnabled(event, part) {
  if (!event.paintParts || typeof event.paintParts !== "object") {
    return true;
  }
  return event.paintParts[part] === true;
}

/**
 * Return a shallow view of an event that matches what is actually drawn.
 *
 * event is unchanged. When an activation's ability paint is off, drop its token
 * and use the impact meaning as eventType. When a health change's recipient text
 * is off, clear recipientPublicAgentId in the view. Refresh and reapply status
 * changes use the ordinary Applied explanation. Otherwise return event itself.
 * This picks display wording; it never removes the original identity.
 *
 * @param {JsonRecord} event
 * @returns {JsonRecord}
 */
export function paintAwareExplanationEvent(event) {
  if (event.kind === "activation" && !paintPartEnabled(event, "ability")) {
    return {
      ...event,
      eventType: `${event.impactSemantic ?? "activation"}_effect`,
      token: null,
      tokenId: null,
    };
  }
  if (event.kind === "net_health" && !paintPartEnabled(event, "recipientText")) {
    return {
      ...event,
      recipientPublicAgentId: null,
    };
  }
  if (
    event.kind === "status_lifecycle" &&
    ["refreshed", "reapplied", "trap_broken_and_reapplied"].includes(event.lifecycle)
  ) {
    return {
      ...event,
      eventType: "status_applied",
      lifecycle: "applied",
      lifecycleToken: resolveVisualToken("lifecycle", "applied", event),
    };
  }
  return event;
}

/**
 * Format a public agent label through the shared identity owner.
 *
 * publicAgentId is the fallback; identity (default null) wins when given and is
 * the whole identity input. Return the plain public identity text. Internal slot
 * numbers are never used as the fallback.
 *
 * @param {unknown} publicAgentId
 * @param {unknown} [identity]
 * @returns {string}
 */
export function formatAgentIdentity(publicAgentId, identity = null) {
  return canonicalAgentIdentity(identity ?? { public_agent_id: publicAgentId })
    .publicIdentity;
}

/**
 * Return the exact authorized identity title, or null.
 *
 * value is checked by exactAuthorizedAgentIdentityV1. A missing or invalid
 * identity is never replaced by a slot or a guessed source.
 *
 * @param {unknown} value
 * @returns {string | null}
 */
function authorizedIdentityTitle(value) {
  return exactAuthorizedAgentIdentityV1(value)?.title ?? null;
}

/**
 * Turn a machine event name into a simple title.
 *
 * value defaults to "event" when nullish; it is turned into text, underscores
 * become spaces and word initials become capitals. Adds no event facts.
 *
 * @param {unknown} value
 * @returns {string}
 */
function humanizeEventName(value) {
  return String(value ?? "event")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

/**
 * Return frozen title and summary text for an admitted event.
 *
 * event.cueSemantic picks death, return, team wave or finite regeneration
 * wording. Other events use a humanized eventType and a general sentence. No
 * event is detected or rebuilt here.
 *
 * @param {JsonRecord} event
 * @returns {Readonly<{title: string, summary: string}>}
 */
function semanticEventCopy(event) {
  if (event.cueSemantic === "agent_died") {
    return Object.freeze({
      title: "Fallen",
      summary: "This champion fell this tick.",
    });
  }
  if (event.cueSemantic === "agent_respawned") {
    return Object.freeze({
      title: "Returned",
      summary: "This champion came back at its waystone this tick.",
    });
  }
  if (event.cueSemantic === "respawn_wave_occurred") {
    return Object.freeze({
      title:
        typeof event.label === "string" && event.label.trim()
          ? event.label
          : "Team Respawn",
      summary: "This team's fallen champions came back this tick.",
    });
  }
  if (event.cueSemantic === "health_regenerated" && Number.isFinite(event.value)) {
    return Object.freeze({
      title: "Life Regained",
      summary: `It regained ${formatDisplayNumber(event.value)} life while out of combat.`,
    });
  }
  return Object.freeze({
    title: humanizeEventName(event.eventType),
    summary: "This happened this tick.",
  });
}

/**
 * Build the shared tooltip for an admitted plan event.
 *
 * event supplies identity, kind, tokens, lifecycle and optional application
 * sources; the drawn-paint view (paintAwareExplanationEvent) is applied first.
 * A death announcement uses its own explanation. Source rows appear only for
 * valid exact public identities, and several applications need every source
 * title known. Return the shared semantic descriptor with title, summary and
 * Source/Recipient rows. Slot IDs may key DOM records but never become display
 * names. The input is unchanged; descriptor and identity validation errors may
 * propagate. `choreography-painter.js` re-exports this function.
 *
 * Agent POV: it shows only identities the plan already admitted for this
 * audience; it never looks anything up.
 *
 * @param {JsonRecord} event
 */
export function explainChoreographyEvent(event) {
  const visibleEvent = paintAwareExplanationEvent(event);
  if (visibleEvent.cueSemantic === "death_announcement") {
    return explainDeathAnnouncement(visibleEvent);
  }
  const semanticCopy = semanticEventCopy(visibleEvent);
  const lifecycleCopy =
    visibleEvent.kind === "status_lifecycle"
      ? statusLifecyclePresentation(visibleEvent.tokenId, visibleEvent.lifecycle)
      : null;
  const title = String(
    lifecycleCopy?.title ??
      visibleEvent.lifecycleToken?.label ??
      visibleEvent.token?.label ??
      semanticCopy.title,
  );
  const rows = [];
  const applicationSources =
    visibleEvent.lifecycle === "applied" &&
    Array.isArray(visibleEvent.applicationSources)
      ? visibleEvent.applicationSources
      : [];
  const applicationSourceIdentities = applicationSources.map((source) =>
    authorizedIdentityTitle(source?.sourceIdentity),
  );
  const distinctApplicationSourceIdentities = [...new Set(applicationSourceIdentities)];
  if (
    distinctApplicationSourceIdentities.length > 0 &&
    applicationSourceIdentities.every((identity) => identity !== null)
  ) {
    rows.push({
      label: distinctApplicationSourceIdentities.length === 1 ? "Source" : "Sources",
      value: distinctApplicationSourceIdentities.join("; "),
      metadata: { compact: true, full: true },
    });
  }
  const directSourceIdentity =
    visibleEvent.kind === "activation" && applicationSources.length === 0
      ? authorizedIdentityTitle(visibleEvent.sourceIdentity)
      : null;
  const recipientIdentity = authorizedIdentityTitle(visibleEvent.recipientIdentity);
  for (const [label, value] of [
    ["Source", directSourceIdentity],
    ["Recipient", recipientIdentity],
  ]) {
    if (value !== null) {
      rows.push({
        label,
        value,
        metadata: { compact: true, full: true },
      });
    }
  }
  const semanticOwnerKey = [
    visibleEvent.kind,
    visibleEvent.eventType,
    visibleEvent.tokenId,
    visibleEvent.lifecycle,
    visibleEvent.actorPresentationKey,
    visibleEvent.sourcePresentationKey,
    ...applicationSources.map((source) => source?.sourcePresentationKey),
    visibleEvent.recipientPresentationKey,
    visibleEvent.agentPresentationKey,
    visibleEvent.actorPublicAgentId,
    visibleEvent.sourcePublicAgentId,
    ...applicationSources.map((source) => source?.sourcePublicAgentId),
    visibleEvent.recipientPublicAgentId,
    visibleEvent.agentPublicAgentId,
  ]
    .filter((value) => typeof value === "string" && value.length > 0)
    .join(":");
  const regenerationPerTick = Number(visibleEvent.outOfCombatRegenerationPerTick);
  const summary =
    visibleEvent.kind === "status_lifecycle" &&
    visibleEvent.tokenId === "in_combat" &&
    visibleEvent.lifecycle === "expired" &&
    Number.isFinite(regenerationPerTick)
      ? `It can regain up to ${formatDisplayNumber(regenerationPerTick)} life per tick while it stays out of combat.`
      : lifecycleCopy !== null
        ? lifecycleCopy.summary
        : String(
            visibleEvent.lifecycleToken?.accessibleName ??
              visibleEvent.token?.accessibleName ??
              semanticCopy.summary,
          );
  return createSemanticDescriptor({
    kind: "event",
    id: `semantic-event:${semanticOwnerKey || "unclassified"}`,
    title,
    tone: "information",
    accent: "none",
    summary,
    rows,
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "pointer",
  });
}

/**
 * Package a stable animation description without starting it.
 *
 * element is the target, keyframes the browser frames, options the browser
 * timing (milliseconds), and plan/event/part form the stable id
 * `mbg:<epochKey>:<eventId>:<part>`. Return a frozen description; options is
 * frozen in place while element and keyframes stay shared. The choreographer
 * starts and owns the real browser animation.
 *
 * @param {Element} element
 * @param {Keyframe[] | PropertyIndexedKeyframes} keyframes
 * @param {KeyframeAnimationOptions} options
 * @param {Record<string, any>} plan
 * @param {Record<string, any>} event
 * @param {string} part
 * @returns {Readonly<AnimationSpec>}
 */
export function animationSpec(element, keyframes, options, plan, event, part) {
  return Object.freeze({
    element,
    keyframes,
    options: Object.freeze(options),
    id: `mbg:${plan.epochKey}:${event.eventId}:${part}`,
  });
}

/**
 * Turn a list of today's paint-part tags into frozen PaintPartRegistration rows.
 *
 * rows pairs each tag (surface, kind and the other tag fields) with the filter
 * id that switches it. Return a frozen array of frozen {part, filter} rows in the
 * given order. `visual-filters.js` reads every look's rows and refuses a tag that
 * two looks claim.
 *
 * @param {ReadonlyArray<readonly [Record<string, string>, string]>} rows
 * @returns {ReadonlyArray<PaintPartRegistration>}
 */
export function paintParts(rows) {
  return Object.freeze(
    rows.map(([part, filter]) =>
      Object.freeze({ part: Object.freeze({ ...part }), filter }),
    ),
  );
}

/**
 * Project authorized circle rows (aura fields or range rings) into drawing records.
 *
 * rows are the scene's own records with a world `center` [x, y] and a world
 * `radius`; transform is this render's projection; options.kind names the
 * records' kind (for example "aura-field"); options.tokenAttribute names the
 * text field copied to data-kind ("kind") or data-token ("token_id");
 * options.classByIdentity maps a presentation key to a class CSS key (default
 * empty); options.tooltipInput turns a row into the plain data its look's
 * tooltip needs (default: the row itself). Rows that are not objects, or whose
 * screen radius is zero or less, are skipped. Return a frozen array of frozen
 * records {kind, center, radius, token, identity, classKey, tooltipInput} in
 * CSS pixels. Pure: no DOM, and the rows are not changed. No range or sight is
 * computed here.
 *
 * @param {ReadonlyArray<unknown>} rows
 * @param {LookTransform} transform
 * @param {{
 *   kind: string,
 *   tokenAttribute: "kind" | "token_id",
 *   classByIdentity?: ReadonlyMap<number | string, string>,
 *   tooltipInput?: (row: JsonRecord) => unknown,
 * }} options
 * @returns {ReadonlyArray<LookRecord>}
 */
export function circleRecords(rows, transform, options) {
  const classByIdentity = options.classByIdentity ?? new Map();
  const tooltipInput = options.tooltipInput ?? ((row) => row);
  const tokenAttribute = options.tokenAttribute;
  /** @type {LookRecord[]} */
  const records = [];
  for (const row of rows) {
    if (!isRecord(row)) {
      continue;
    }
    const center = screenPoint(row.center, transform);
    const radius = transform.worldLengthToScreen(finiteNumber(row.radius));
    if (radius <= 0) {
      continue;
    }
    const identity =
      typeof row.presentation_key === "string" ? row.presentation_key : null;
    const classKey = identity === null ? undefined : classByIdentity.get(identity);
    records.push(
      Object.freeze({
        kind: options.kind,
        center,
        radius,
        token: typeof row[tokenAttribute] === "string" ? row[tokenAttribute] : null,
        identity: Object.freeze({
          presentation_key: row.presentation_key,
          global_slot: row.global_slot,
        }),
        classKey: classKey ?? null,
        tooltipInput: tooltipInput(row),
      }),
    );
  }
  return Object.freeze(records);
}

/**
 * Build circle nodes from circleRecords output, with optional tooltips.
 *
 * doc creates the nodes. options.className is the circle class;
 * options.tokenAttribute picks data-kind ("kind") or data-token ("token_id");
 * options.tooltip is the look's tooltip function (record to descriptor, or null
 * to leave a circle out) or null for circles with no tooltip at all;
 * options.withStrokeHitRegion (default false) wraps each circle with a separate
 * transparent hit circle in an owner group `<className>-owner`. Return the nodes
 * in record order, ready for the caller's layer. Registers tooltip owners (a DOM
 * side effect through `tooltip.js`); no game range or sight is computed.
 *
 * @param {Document} doc
 * @param {ReadonlyArray<LookRecord>} records
 * @param {{
 *   className: string,
 *   tokenAttribute: "kind" | "token_id",
 *   tooltip: ((record: LookRecord) => unknown) | null,
 *   withStrokeHitRegion?: boolean,
 * }} options
 * @returns {SVGElement[]}
 */
export function buildCircleNodes(doc, records, options) {
  const { className, tokenAttribute, tooltip } = options;
  const withStrokeHitRegion = options.withStrokeHitRegion === true;
  const circles = [];
  for (const record of records) {
    const circle = svgNode(doc, "circle", {
      class: className,
      cx: record.center.x,
      cy: record.center.y,
      r: record.radius,
      role: tooltip === null ? null : "img",
      tabindex: tooltip === null ? null : "0",
    });
    if (typeof record.token === "string") {
      if (tokenAttribute === "kind") {
        circle.dataset.kind = record.token;
      } else {
        circle.dataset.token = record.token;
      }
    }
    setDisplayIdentityData(circle, record.identity);
    if (record.classKey) {
      circle.dataset.class = record.classKey;
    }
    if (tooltip === null) {
      circles.push(circle);
      continue;
    }
    if (!withStrokeHitRegion) {
      const descriptor = /** @type {{title: string} | null} */ (tooltip(record));
      if (descriptor === null) {
        continue;
      }
      circle.setAttribute("aria-label", descriptor.title);
      registerTooltipOwner(circle, descriptor);
      circles.push(circle);
      continue;
    }
    const owner = svgNode(doc, "g", {
      class: `${className}-owner`,
    });
    const hitRegion = svgNode(doc, "circle", {
      class: `${className}-hit`,
      cx: record.center.x,
      cy: record.center.y,
      r: record.radius,
      "aria-hidden": "true",
    });
    if (typeof record.token === "string") {
      hitRegion.dataset[tokenAttribute === "kind" ? "kind" : "token"] = record.token;
    }
    setDisplayIdentityData(hitRegion, record.identity);
    owner.append(circle, hitRegion);
    const descriptor = /** @type {{title: string} | null} */ (tooltip(record));
    if (descriptor === null) {
      continue;
    }
    owner.setAttribute("role", "img");
    owner.setAttribute("tabindex", "0");
    owner.setAttribute("aria-label", descriptor.title);
    registerTooltipOwner(owner, descriptor);
    circles.push(owner);
  }
  return circles;
}

/**
 * Move existing circle nodes to new circle records (reprojection).
 *
 * nodes are the elements buildCircleNodes returned (a circle, or an owner group
 * holding the circle and its hit circle); records are the matching new records
 * in the same order. Each circle and hit circle gets the new cx, cy and r. Extra
 * nodes or records are left alone. Return undefined. Tooltips are unchanged.
 *
 * @param {ReadonlyArray<Element>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function updateCircleNodes(nodes, records) {
  for (const [index, node] of nodes.entries()) {
    const record = records[index];
    if (record === undefined) {
      break;
    }
    const circles = node.localName === "circle" ? [node] : [...node.children];
    for (const circle of circles) {
      setNodeAttributes(circle, {
        cx: record.center.x,
        cy: record.center.y,
        r: record.radius,
      });
    }
  }
}

/**
 * Create a detached damage, healing or neutral mark (no number).
 *
 * doc creates SVG. value "damage" gives a minus, "healing" a plus, and anything
 * else a neutral diamond. Return the `g.combat-impact__semantic` group; no amount
 * or outcome is worked out from this mark. Used by the Basic, Ultimate and
 * health text looks.
 *
 * @param {Document} doc
 * @param {unknown} value
 * @returns {SVGElement}
 */
export function semanticImpactGlyph(doc, value) {
  const semantic = value === "damage" || value === "healing" ? value : "neutral";
  const group = effectNode(doc, "g", {
    class: `combat-impact__semantic combat-impact__semantic--${semantic}`,
  });
  if (semantic === "neutral") {
    group.append(
      effectNode(doc, "path", {
        d: "M 0 -6 L 6 0 0 6 -6 0 Z",
      }),
    );
    return group;
  }
  group.append(
    effectNode(doc, "line", {
      x1: -6,
      y1: 0,
      x2: 6,
      y2: 0,
    }),
  );
  if (semantic === "healing") {
    group.append(
      effectNode(doc, "line", {
        x1: 0,
        y1: -6,
        x2: 0,
        y2: 6,
      }),
    );
  }
  return group;
}

/**
 * Update the line joining a moved cue to its disclosed recipient.
 *
 * connector may be null; event supplies cueLeader, or recipient and cue points
 * (CSS pixels). A valid allocated leader wins; otherwise the existing line runs
 * from recipient to cue. Without those points its x ends are removed. Return
 * undefined. No new anchor is made and the recipient is not moved.
 *
 * @param {Element | null} connector
 * @param {JsonRecord} event
 */
export function updateCueLeader(connector, event) {
  const leader = connector?.querySelector(".combat-cue__leader") ?? null;
  const recipient = event.recipient;
  const cue = event.cue;
  if (leader && isAllocatedLeader(event.cueLeader)) {
    syncAllocatedLeader(connector, ".combat-cue__leader", event.cueLeader);
    return;
  }
  if (!leader || !recipient || !cue) {
    leader?.removeAttribute("x1");
    leader?.removeAttribute("x2");
    return;
  }
  setEffectAttributes(leader, {
    visibility: "visible",
    x1: recipient.x,
    y1: recipient.y,
    x2: cue.x,
    y2: cue.y,
  });
}

/**
 * Make detached groups for an event look used without the painter.
 *
 * doc creates the groups; event says whether a route group is needed (when it
 * has a route). The groups are settled with motion off and no plan timing, so
 * nothing animates. Return the EventMount to pass to an event look's build.
 * Used by pictures drawn outside the arena and by tests.
 *
 * @param {Document} doc
 * @param {JsonRecord} event
 * @returns {EventMount}
 */
export function detachedEventMount(doc, event) {
  return {
    group: effectNode(doc, "g", { class: "combat-effect" }),
    connector: effectNode(doc, "g", { class: "combat-connector-effect" }),
    underlay: event.route
      ? effectNode(doc, "g", { class: "combat-route-effect" })
      : null,
    plan: { epochKey: "detached", phases: {} },
    options: { motionMode: "off", settled: true },
    animationSpecs: [],
  };
}

/**
 * Return the containers an event look drew into, as build's result.
 *
 * mount is the EventMount used. Return [group, connector] plus the route group
 * when there is one.
 *
 * @param {EventMount} mount
 * @returns {SVGElement[]}
 */
export function mountNodes(mount) {
  return mount.underlay === null
    ? [mount.group, mount.connector]
    : [mount.group, mount.connector, mount.underlay];
}
