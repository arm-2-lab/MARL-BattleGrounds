/**
 * @file Check the Priest's cross-only emblem on its Class Seal (`src/icons.js`
 * createSvgIcon option onClassSeal, used by `src/looks/champions.js`).
 *
 * Contracts checked: with onClassSeal, the Priest emblem in the sigil style
 * draws only its two cross bars (keyline and body, no sun ring, rays, jewels or
 * filigree) at every master (micro, standard, display), keeps the ash paint's
 * 55%, and carries data-icon-variant "cross"; the outline style draws one
 * 12-corner cross silhouette; every other class, and the plain style, ignore
 * the option. In the Champions look a Priest token shows the cross alone
 * exactly where a seal is drawn: on its living token and its falling held look
 * while Class Seals is on (from 18 px), over its corpse's ash seal outline,
 * never on a Fallen Step Aside coin, and never while Class Seals is off.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { authorizedPresentationSceneView } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { createSvgIcon } from "../src/icons.js";
import { createViewportTransform } from "../src/layout.js";
import * as champions from "../src/looks/champions.js";
import { DEFAULT_VISUAL_FILTER_STATE } from "../src/visual-filters.js";

class FakeElement {
  /** @param {FakeDocument} doc @param {string} tagName */
  constructor(doc, tagName) {
    this.ownerDocument = doc;
    this.tagName = tagName;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    /** @type {FakeElement | null} */
    this.parentElement = null;
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
  hasAttribute(name) {
    return this.attributes.has(name);
  }
  /** @param {string} name */
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  getBoundingClientRect() {
    return { x: 0, y: 0, width: 0, height: 0, left: 0, top: 0, right: 0, bottom: 0 };
  }
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    for (const node of nodes) {
      node.remove();
      node.parentElement = this;
      this.children.push(node);
    }
  }
  /** @param {...FakeElement} nodes */
  replaceChildren(...nodes) {
    for (const child of this.children) child.parentElement = null;
    this.children = [];
    this.append(...nodes);
  }
  /** @param {FakeElement} node */
  replaceWith(node) {
    const parent = this.parentElement;
    if (!parent) return;
    const index = parent.children.indexOf(this);
    node.remove();
    parent.children[index] = node;
    node.parentElement = parent;
    this.parentElement = null;
  }
  remove() {
    if (this.parentElement) {
      this.parentElement.children = this.parentElement.children.filter(
        (child) => child !== this,
      );
      this.parentElement = null;
    }
  }
  /** @param {FakeElement} other */
  isEqualNode(other) {
    return other instanceof FakeElement && other.markup() === this.markup();
  }
  addEventListener() {}
  blur() {}
  /** @returns {FakeElement[]} */
  all() {
    return [this, ...this.children.flatMap((child) => child.all())];
  }
  /** @param {string} className */
  find(className) {
    return this.all().filter((node) =>
      (node.getAttribute("class") ?? "").split(/\s+/u).includes(className),
    );
  }
  /** @returns {string} */
  markup() {
    const attributes = [
      ...this.attributes,
      ...Object.entries(this.dataset).map(([key, value]) => [
        `data-${key.replace(/[A-Z]/gu, (letter) => `-${letter.toLowerCase()}`)}`,
        value,
      ]),
    ]
      .map(([key, value]) => ` ${key}="${value}"`)
      .join("");
    return `<${this.tagName}${attributes}>${this.textContent}${this.children
      .map((child) => child.markup())
      .join("")}</${this.tagName}>`;
  }
}

class FakeDocument {
  constructor() {
    this.activeElement = null;
  }
  /** @param {string} _namespace @param {string} tagName */
  createElementNS(_namespace, tagName) {
    return new FakeElement(this, tagName);
  }
}

/** @param {string} key @param {Record<string, unknown>} options */
function icon(key, options) {
  return /** @type {FakeElement} */ (
    /** @type {unknown} */ (
      createSvgIcon(/** @type {any} */ (new FakeDocument()), key, options)
    )
  );
}

/**
 * Return every number pair of an absolute M/L/H/V path as [x, y] points.
 *
 * @param {string} d
 */
function points(d) {
  const numbers = (d.match(/-?\d+(?:\.\d+)?/gu) ?? []).map(Number);
  return numbers.flatMap((value, index) =>
    index % 2 === 0 ? [[value, numbers[index + 1]]] : [],
  );
}

test("onClassSeal draws the Priest's cross alone at every master", () => {
  for (const size of [10, 20, 40]) {
    const full = icon("emblem-priest", { style: "sigil", size });
    const cross = icon("emblem-priest", { style: "sigil", size, onClassSeal: true });
    assert.equal(full.dataset.iconVariant, undefined);
    assert.equal(cross.dataset.iconVariant, "cross");
    const body = cross.find("sigil-body");
    assert.equal(body.length, 1);
    const d = body[0].getAttribute("d") ?? "";
    assert.equal(d.split("M").length - 1, 2, `two bars at ${size} px`);
    for (const [x, y] of points(d)) {
      assert.ok(Math.abs(x - 12) <= 6 && Math.abs(y - 12) <= 6, `${x},${y}`);
    }
    const keyline = cross.find("sigil-keyline");
    assert.equal(keyline.length, 1);
    assert.equal(keyline[0].getAttribute("d"), d);
    assert.equal(cross.find("sigil-jewel").length, 0);
    assert.equal(cross.find("sigil-filigree").length, 0);
    const fullBody = full.find("sigil-body")[0].getAttribute("d") ?? "";
    assert.ok(fullBody.split("M").length - 1 > 2, "the full emblem keeps its sun");
    assert.ok(fullBody.includes(d.slice(0, d.indexOf("M", 1))), "bars come from it");
  }
  const ash = icon("class-priest", { size: 20, paint: "ash", onClassSeal: true });
  assert.equal(ash.find("sigil-body")[0].getAttribute("opacity"), "0.55");
});

test("the outline style draws the cross's one silhouette", () => {
  const outline = icon("emblem-priest", {
    style: "outline",
    size: 20,
    onClassSeal: true,
  });
  const paths = outline.find("sigil-outline");
  assert.equal(paths.length, 1);
  const d = paths[0].getAttribute("d") ?? "";
  assert.match(d, /^M[\d.]+ [\d.]+(?:[HV][\d.]+){11}Z$/u);
  assert.equal(outline.dataset.iconVariant, "cross");
});

test("other classes and the plain style ignore onClassSeal", () => {
  for (const key of [
    "emblem-mage",
    "emblem-warrior",
    "emblem-hunter",
    "emblem-rogue",
  ]) {
    assert.equal(
      icon(key, { size: 20, onClassSeal: true }).markup(),
      icon(key, { size: 20 }).markup(),
      key,
    );
  }
  assert.equal(
    icon("class-priest", { style: "plain", onClassSeal: true }).markup(),
    icon("class-priest", { style: "plain" }).markup(),
  );
});

test("a Priest token shows the cross exactly where its seal is drawn", async () => {
  const fixture = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  const frame = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(fixture.presentations.replay_oracle),
  );
  const scene = authorizedPresentationSceneView(frame);
  assert.ok(scene);
  const transform = createViewportTransform({
    worldWidth: scene.map.width,
    worldHeight: scene.map.height,
    viewportWidth: 960,
    viewportHeight: 600,
    padding: 24,
  });
  const priestId = "agent-slot-2";
  const sealsOn = {
    ...DEFAULT_VISUAL_FILTER_STATE,
    class_seals: true,
    fallen_step_aside: true,
  };
  /**
   * @param {Record<string, any>} sceneInput
   * @param {Record<string, boolean>} filters
   * @param {Record<string, any>} [beats]
   */
  const drawnPriest = (sceneInput, filters, beats) => {
    const record = champions
      .records({ scene: sceneInput, transform, visualFilters: filters, beats })
      .find((row) => row.agent.public_agent_id === priestId);
    assert.ok(record);
    const [bundle] = champions.build([record], /** @type {any} */ (new FakeDocument()));
    champions.update([bundle], [record]);
    return {
      record,
      root: /** @type {FakeElement} */ (/** @type {any} */ (bundle.root)),
    };
  };
  /** @param {FakeElement} root @param {string} className */
  const variants = (root, className) =>
    root.find(className).map((node) => node.dataset.iconVariant ?? "full");

  const living = drawnPriest(scene, sealsOn);
  assert.ok(living.record.seal);
  assert.ok(living.record.radius >= champions.CLASS_SEAL.minRadius);
  assert.deepEqual(variants(living.root, "agent-class-icon"), ["cross"]);
  const plain = drawnPriest(scene, DEFAULT_VISUAL_FILTER_STATE);
  assert.equal(plain.record.seal, null);
  assert.deepEqual(variants(plain.root, "agent-class-icon"), ["full"]);

  const priest = scene.agents.find(
    (/** @type {Record<string, any>} */ row) => row.public_agent_id === priestId,
  );
  /** @param {Record<string, any>} changes @param {Record<string, any>} [others] */
  const withPriest = (changes, others = {}) => ({
    ...scene,
    agents: scene.agents.map((/** @type {Record<string, any>} */ row) =>
      row.public_agent_id === priestId ? { ...row, ...changes } : { ...row, ...others },
    ),
  });
  const corpse = drawnPriest(withPriest({ alive: false }), sealsOn);
  assert.equal(corpse.record.state, "fallen");
  assert.equal(corpse.record.aside, null);
  assert.equal(corpse.root.find("champion-seal__ash").length, 1);
  assert.deepEqual(variants(corpse.root, "champion-ashen__emblem"), ["cross"]);

  const covered = withPriest({ alive: false });
  covered.agents = covered.agents.map((/** @type {Record<string, any>} */ row) =>
    row.public_agent_id === "agent-slot-0"
      ? { ...row, position: priest.position }
      : row,
  );
  const coin = drawnPriest(covered, sealsOn);
  assert.ok(coin.record.aside, "a standing champion covers the corpse");
  assert.equal(coin.root.find("champion-seal__ash").length, 0);
  assert.deepEqual(variants(coin.root, "champion-ashen__emblem"), ["full"]);

  const falling = drawnPriest(withPriest({ alive: false }), sealsOn, {
    rows: [],
    view: { life: [{ presentationKey: priest.presentation_key, before: 100 }] },
  });
  assert.equal(falling.record.state, "falling");
  const held = falling.root.find("champion-falling")[0];
  assert.deepEqual(variants(held, "agent-class-icon"), ["cross"]);
});
