/**
 * @file Check the Heraldic Sigils in src/icons.js: every old glyph key still
 * resolves (as a sigil and as its plain glyph); the five class emblems draw an
 * Ink keyline, a class-coloured body and their jewels; the plain style draws
 * today's monoline glyph; status and aura tokens map to the sigil of their
 * mark (Frozen, Sapped by class, Hamstrung by class, Snagged, Withering,
 * Unbound, Kindled, In Combat hourglass, Spawn Shield); In Combat and Unseen
 * glyphs exist; life-change glyphs are pip rails with no plus sign; the size
 * picks the micro, standard or display master; and sigil output is
 * export-safe (no url(), gradients, masks, filters, images or ids; colours
 * only through token classes in icons.css).
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  createSvgIcon,
  iconDefinition,
  KNOWN_GLYPH_KEYS,
  modifierSigilKey,
  sigilDefinition,
  sigilMasterForSize,
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

const OLD_KEYS = [
  "class-mage",
  "class-warrior",
  "class-hunter",
  "class-rogue",
  "class-priest",
  "team-a",
  "team-b",
  "status-stun",
  "status-slow",
  "status-anti-heal",
  "status-freedom",
  "status-burst",
  "status-spawn-shield",
  "activation-basic-damage",
  "activation-basic-heal",
  "activation-holy-word",
  "activation-burst",
  "activation-charge",
  "activation-trap",
  "activation-poison",
  "modifier-amplification",
  "modifier-mitigation",
  "lifecycle-applied",
  "lifecycle-refreshed",
  "lifecycle-decremented",
  "lifecycle-expired",
  "lifecycle-trap-broken",
  "lifecycle-death-clear",
  "lifecycle-ended",
  "lifecycle-trap-broken-reapplied",
  "combat-in-progress",
];

test("every old glyph key is kept and draws a sigil by default", () => {
  for (const key of OLD_KEYS) {
    assert.ok(KNOWN_GLYPH_KEYS.includes(key), key);
    assert.notEqual(iconDefinition(key).glyphKey, "unknown", key);
    assert.ok(sigilDefinition(key), key);
    const svg = /** @type {any} */ (createSvgIcon(doc, key));
    assert.equal(svg.dataset.icon, key);
    assert.equal(svg.dataset.iconStyle, "sigil");
    assert.ok(svg.children.length > 0, key);
  }
  assert.equal(sigilDefinition("unknown"), null);
  assert.equal(/** @type {any} */ (createSvgIcon(doc, "nope")).dataset.icon, "unknown");
});

test("class emblems are filled with an Ink keyline and class paint", () => {
  for (const name of ["mage", "warrior", "hunter", "rogue", "priest"]) {
    const svg = /** @type {any} */ (createSvgIcon(doc, `class-${name}`, { size: 20 }));
    const classes = svg.children.map((/** @type {any} */ child) =>
      child.getAttribute("class"),
    );
    assert.equal(classes[0], "sigil-keyline");
    assert.equal(svg.children[0].getAttribute("stroke-width"), "3");
    assert.match(classes[1], new RegExp(`sigil-paint--class-${name}`, "u"));
  }
  const micro = /** @type {any} */ (createSvgIcon(doc, "emblem-mage", { size: 10 }));
  assert.equal(micro.children[0].getAttribute("stroke-width"), "2");
  const ash = /** @type {any} */ (createSvgIcon(doc, "emblem-rogue", { paint: "ash" }));
  assert.match(ash.children[1].getAttribute("class"), /sigil-paint--ash/u);
  assert.equal(ash.children[1].getAttribute("opacity"), "0.55");
  const host = /** @type {any} */ (
    createSvgIcon(doc, "emblem-hunter", { paint: "host", teamId: 2 })
  );
  assert.match(host.children[1].getAttribute("class"), /sigil-paint--team-b/u);
  const ready = /** @type {any} */ (
    createSvgIcon(doc, "emblem-priest", { paint: "ready" })
  );
  assert.equal(ready.children[1].getAttribute("stroke-opacity"), "0.45");
});

test("the plain style draws today's monoline glyph", () => {
  const svg = /** @type {any} */ (
    createSvgIcon(doc, "class-hunter", { style: "plain" })
  );
  assert.equal(svg.dataset.iconStyle, "plain");
  assert.equal(svg.getAttribute("stroke"), "currentColor");
  assert.deepEqual(
    svg.children.map((/** @type {any} */ child) => child.tagName),
    ["path", "line", "line"],
  );
  assert.equal(iconDefinition("sigil-frozen").glyphKey, "status-stun");
  assert.equal(iconDefinition("emblem-mage").glyphKey, "class-mage");
});

test("status and aura tokens map to the sigil of their mark", () => {
  assert.deepEqual(
    [
      "stun_warrior_charge",
      "stun_rogue_poison",
      "stun_hunter_trap",
      "slow_warrior_charge",
      "slow_rogue_poison",
      "slow_hunter_basic",
      "anti_heal_rogue_poison",
      "priest_freedom",
      "mage_burst",
      "in_combat",
      "spawn_shield",
      "nope",
    ].map(statusSigilKey),
    [
      "sigil-dazed-warrior",
      "sigil-dazed-rogue",
      "sigil-frozen",
      "sigil-hobbled-warrior",
      "sigil-hobbled-rogue",
      "sigil-snagged",
      "sigil-withering",
      "sigil-unbound",
      "sigil-kindled",
      "sigil-in-combat",
      "sigil-moonward",
      null,
    ],
  );
  assert.equal(modifierSigilKey("mage_amplification"), "sigil-empowerment");
  assert.equal(modifierSigilKey("warrior_mitigation"), "sigil-barrier");
  for (const key of ["sigil-in-combat", "sigil-unseen", "sigil-life-applied"]) {
    assert.ok(KNOWN_GLYPH_KEYS.includes(key), key);
  }
  // Generic old keys draw the class colour in the caller's colour.
  const stun = /** @type {any} */ (createSvgIcon(doc, "status-stun"));
  assert.ok(
    stun.children.some((/** @type {any} */ child) =>
      /sigil-paint--current/u.test(child.getAttribute("class")),
    ),
  );
});

test("life-change glyphs are pip rails with no plus sign", () => {
  const applied = /** @type {any} */ (createSvgIcon(doc, "lifecycle-applied"));
  // A keyline under four moon-silver parts: the rail and three pips.
  assert.equal(applied.children.length, 5);
  assert.ok(
    applied.children
      .slice(1)
      .every((/** @type {any} */ child) =>
        /sigil-paint--moon/u.test(child.getAttribute("class")),
      ),
  );
});

test("size picks the emblem master", () => {
  assert.equal(sigilMasterForSize(10), "micro");
  assert.equal(sigilMasterForSize(14), "standard");
  assert.equal(sigilMasterForSize(32), "display");
  assert.equal(sigilMasterForSize(undefined), "standard");
});

test("sigil output and icons.css are export-safe", async () => {
  for (const key of KNOWN_GLYPH_KEYS) {
    const svg = /** @type {any} */ (createSvgIcon(doc, key, { size: 40 }));
    const text = JSON.stringify([
      ...svg.attributes,
      ...svg.children.map((/** @type {any} */ c) => [...c.attributes]),
    ]);
    assert.doesNotMatch(text, /url\(|Gradient|mask|filter|image|"id"/u, key);
    assert.doesNotMatch(text, /#[0-9a-f]{3,8}\b/iu, key);
  }
  const css = await readFile(
    new URL("../src/styles/icons.css", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(css, /#[0-9a-f]{3,8}\b/iu);
  assert.doesNotMatch(css, /url\(|animation\s*:|transition\s*:/u);
});
