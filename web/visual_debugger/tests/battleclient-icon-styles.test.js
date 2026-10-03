/**
 * @file Check the page-wide icon styles in src/icons.js that the Lore Emblems
 * and Status Icons From Marks switches set (setIconStyleDefaults and
 * defaultIconStyle): both default to the Heraldic Sigils; Lore Emblems off
 * draws every class emblem (`class-*` and `emblem-*`) in the "outline" style
 * (the emblem's silhouette as one thin line, never today's old glyph) and
 * leaves statuses alone; Status Icons From Marks off draws every status, aura,
 * life-change, In Combat, Rest and Unseen icon as its mark's shapes in thin
 * lines and leaves class emblems alone; outline icons have no fills or class
 * colours; team crests and ability icons follow neither switch; an explicit
 * style always wins; unknown values keep the sigil style; and the "plain"
 * style draws today's glyph, with Rest and Moon Dew as the plain drop
 * `status-rest` (never the unknown glyph).
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  createSvgIcon,
  defaultIconStyle,
  setIconStyleDefaults,
  statusSigilKey,
} from "../src/icons.js";

class FakeElement {
  /** @param {string} tag */
  constructor(tag) {
    this.tagName = tag;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    /** @type {Record<string, string>} */
    this.dataset = {};
  }
  /** @param {string} name @param {unknown} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    this.children.push(...nodes);
  }
}

const doc = /** @type {any} */ ({
  /** @param {string} _namespace @param {string} tag */
  createElementNS(_namespace, tag) {
    return new FakeElement(tag);
  },
});

/** @param {string} key @param {any} [options] */
const drawn = (key, options) => {
  const svg = /** @type {any} */ (createSvgIcon(doc, key, options));
  return {
    style: svg.dataset.iconStyle,
    icon: svg.dataset.icon,
    fill: svg.getAttribute("fill"),
    classes: svg.children.map((/** @type {any} */ c) => c.getAttribute("class")),
  };
};

const CLASS_KEYS = ["class-mage", "emblem-warrior", "class-hunter", "emblem-rogue"];
const STATUS_KEYS = [
  "status-stun",
  "sigil-frozen",
  "sigil-withering",
  "modifier-mitigation",
  "sigil-in-combat",
  "sigil-unseen",
  "sigil-life-applied",
  /** @type {string} */ (statusSigilKey("rest")),
  "sigil-moon-dew",
];
const NEITHER_KEYS = ["team-a", "sigil-crest-b", "sigil-ab-charge", "activation-burst"];

test("both switches default to the sigil style", () => {
  setIconStyleDefaults();
  for (const key of [...CLASS_KEYS, ...STATUS_KEYS, ...NEITHER_KEYS]) {
    assert.equal(defaultIconStyle(key), "sigil", key);
    assert.equal(drawn(key).style, "sigil", key);
  }
});

test("Lore Emblems off draws only class emblems as outlines", () => {
  setIconStyleDefaults({ classEmblems: "outline" });
  try {
    for (const key of CLASS_KEYS) {
      const result = drawn(key);
      assert.equal(result.style, "outline", key);
      assert.equal(result.fill, "none", key);
      assert.deepEqual(result.classes, ["sigil-outline"], key);
    }
    for (const key of [...STATUS_KEYS, ...NEITHER_KEYS]) {
      assert.equal(drawn(key).style, "sigil", key);
    }
  } finally {
    setIconStyleDefaults();
  }
});

test("Status Icons From Marks off draws only status-family icons as outlines", () => {
  setIconStyleDefaults({ statusIcons: "outline" });
  try {
    for (const key of STATUS_KEYS) {
      const result = drawn(key);
      assert.equal(result.style, "outline", key);
      assert.ok(result.classes.length > 0, key);
      assert.ok(
        result.classes.every((/** @type {string} */ name) => name === "sigil-outline"),
        key,
      );
    }
    for (const key of [...CLASS_KEYS, ...NEITHER_KEYS]) {
      assert.equal(drawn(key).style, "sigil", key);
    }
  } finally {
    setIconStyleDefaults();
  }
});

test("an explicit style wins and unknown values keep the sigil", () => {
  setIconStyleDefaults({ classEmblems: "outline", statusIcons: "outline" });
  try {
    assert.equal(drawn("class-mage", { style: "sigil" }).style, "sigil");
    assert.equal(drawn("status-stun", { style: "sigil" }).style, "sigil");
  } finally {
    setIconStyleDefaults();
  }
  setIconStyleDefaults({ classEmblems: "dotted", statusIcons: 1 });
  assert.equal(drawn("class-mage").style, "sigil");
  assert.equal(drawn("status-stun").style, "sigil");
  setIconStyleDefaults();
});

test("the plain style keeps today's glyphs and draws Rest as a plain drop", () => {
  assert.equal(drawn("class-mage", { style: "plain" }).icon, "class-mage");
  assert.equal(drawn("sigil-rest-vessel", { style: "plain" }).icon, "status-rest");
  assert.equal(drawn("sigil-moon-dew", { style: "plain" }).icon, "status-rest");
});
