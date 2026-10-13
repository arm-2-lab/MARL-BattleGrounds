/**
 * @file Test-only support for the V5 Health and Combat Text add-on tests
 * (`v5-saved-beat`, `v5-last-breath` and `v5-battle-text-detail`). It holds a
 * tiny fake SVG document (enough for `src/looks/health-text.js` build and
 * update: attributes, dataset, text, children, class lookups and markup), plain
 * plan rows shaped like the plan's health rows (with the look inputs
 * maximumHealth, recipientRadius and the Oracle totals and damage sources),
 * Oracle View and Agent POV plans, a simple projection (40 px per world unit)
 * and a draw helper that builds one event into a painter-like mount (optionally
 * with that projection) and reports its nodes and animation specs.
 */
import * as healthText from "../src/looks/health-text.js";
import { DEFAULT_VISUAL_FILTER_STATE } from "../src/visual-filters.js";

export class FakeElement {
  /** @param {FakeDocument} doc @param {string} tagName */
  constructor(doc, tagName) {
    this.ownerDocument = doc;
    this.tagName = tagName;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    /** @type {Record<string, string>} */
    this.dataset = {};
    this.textContent = "";
  }
  /** @param {string} name @param {unknown} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  /** @param {string} name */
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    this.children.push(...nodes);
  }
  addEventListener() {}
  /** @returns {FakeElement[]} */
  all() {
    return [this, ...this.children.flatMap((child) => child.all())];
  }
  /** @param {string} selector */
  querySelectorAll(selector) {
    const name = selector.replace(/^\./u, "");
    return this.all()
      .slice(1)
      .filter((node) =>
        (node.getAttribute("class") ?? "").split(/\s+/u).includes(name),
      );
  }
  /** @param {string} selector */
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }
  /** @returns {string} */
  markup() {
    const attributes = [...this.attributes, ...Object.entries(this.dataset)]
      .map(([key, value]) => ` ${key}="${value}"`)
      .join("");
    return `<${this.tagName}${attributes}>${this.textContent}${this.children
      .map((child) => child.markup())
      .join("")}</${this.tagName}>`;
  }
}

export class FakeDocument {
  /** @param {string} _namespace @param {string} tagName */
  createElementNS(_namespace, tagName) {
    return new FakeElement(this, tagName);
  }
}

export const RECIPIENT_KEY = "body-recipient";
export const MAGE_KEY = "body-mage";
export const HUNTER_KEY = "body-hunter";

/**
 * @param {Record<string, any>} [overrides]
 * @returns {Record<string, any>}
 */
export function healthRow(overrides = {}) {
  const before = overrides.healthBefore ?? 80;
  const after = overrides.healthAfter ?? 50;
  return {
    eventId: "event-health",
    kind: "net_health",
    recipientPresentationKey: RECIPIENT_KEY,
    recipient: { x: 300, y: 200 },
    cue: { x: 300, y: 168 },
    cueBounds: { left: 280, top: 159, right: 320, bottom: 177 },
    netDelta: after - before,
    healthBefore: before,
    healthAfter: after,
    outcome: after < before ? "damage" : after > before ? "healing" : "unchanged",
    maximumHealth: 100,
    recipientRadius: 20,
    totalDamage: Math.max(0, before - after),
    totalHealing: Math.max(0, after - before),
    damageSources: [],
    paintParts: { effect: true, battleText: true, recipientText: true },
    phaseStart: 460,
    phaseEnd: 1100,
    ...overrides,
  };
}

/**
 * @param {"researcher" | "agent_pov"} audience
 * @param {{start?: [number, number] | null, returned?: boolean, events?: any[]}} [body]
 * @returns {Record<string, any>}
 */
export function planFor(audience, body = {}) {
  return {
    epochKey: "epoch",
    audience,
    events: body.events ?? [],
    bodies: [
      {
        presentationKey: RECIPIENT_KEY,
        start: body.start === undefined ? [1, 1] : body.start,
        returned: body.returned ?? false,
      },
    ],
    beats: { blowSpeed: 1, windows: { settle: [620, 1100] } },
    phases: { total: 1100, reducedTotal: 220 },
  };
}

/**
 * @param {Record<string, boolean>} switches
 * @returns {Record<string, boolean>}
 */
export function filtersWith(switches) {
  return { ...DEFAULT_VISUAL_FILTER_STATE, ...switches };
}

/**
 * @param {Record<string, any>} event
 * @param {Record<string, any>} plan
 * @param {Record<string, boolean> | null} filters
 * @param {{motionMode?: string, settled?: boolean, transform?: Record<string, any>}} [options]
 */
export function draw(event, plan, filters, options = {}) {
  const doc = /** @type {any} */ (new FakeDocument());
  /** @type {any[]} */
  const animationSpecs = [];
  const mount = /** @type {any} */ ({
    group: doc.createElementNS("", "g"),
    connector: doc.createElementNS("", "g"),
    underlay: null,
    plan,
    options: {
      motionMode: options.motionMode ?? "normal",
      settled: options.settled ?? false,
    },
    animationSpecs,
  });
  const records = healthText.records({
    event,
    plan,
    filters,
    transform: options.transform ?? null,
  });
  healthText.build(records, doc, mount);
  const group = /** @type {FakeElement} */ (mount.group);
  return {
    records,
    group,
    connector: /** @type {FakeElement} */ (mount.connector),
    animationSpecs,
    nodeCount: group.all().length,
  };
}

export const TRANSFORM = Object.freeze({
  worldToScreen: (/** @type {[number, number]} */ point) => ({
    x: point[0] * 40,
    y: point[1] * 40,
  }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 40,
  viewportBounds: Object.freeze({ left: 0, top: 0, right: 1000, bottom: 600 }),
});

export const EXPORT_UNSAFE =
  /url\(|gradient|<mask|<filter|<image|<pattern|mix-blend| id="| style="/iu;
