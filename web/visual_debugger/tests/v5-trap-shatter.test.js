/**
 * @file Check Freezing Trap Break (switch freezing_trap_break, on by default)
 * in the Rune Zones look (src/looks/statuses.js). When damage breaks a seen
 * Freezing Trap: the Frozen pieces go out at Settle over 60 ms (a cut when the
 * switch is off); the Shatter crack comes in at Impact over 60 ms and rests at
 * 80%; the six shards come in at Settle and rest at 45%. With the switch off
 * neither echo is drawn. A body coming into sight in Agent POV (not seen at
 * tick start) gets no Shatter and no fade. build writes the Impact beat on the
 * crack and the Settle beat on the Frozen pieces; the SVG is export-safe; a
 * repeat render gives equal records; and the Shatter adds only two nodes.
 */
import assert from "node:assert/strict";
import test from "node:test";

import * as statuses from "../src/looks/statuses.js";

class FakeElement {
  /** @param {string} tag */
  constructor(tag) {
    this.tagName = tag;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    this.ownerDocument = fakeDocument;
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
  /** @returns {string} */
  serialize() {
    const attrs = [...this.attributes].map(([key, value]) => ` ${key}="${value}"`);
    return `<${this.tagName}${attrs.join("")}>${this.children.map((child) => child.serialize()).join("")}</${this.tagName}>`;
  }
}

const fakeDocument = {
  /** @param {string} _namespace @param {string} tag */
  createElementNS(_namespace, tag) {
    return new FakeElement(tag);
  },
};

const transform = /** @type {any} */ ({
  worldToScreen: (/** @type {number[]} */ point) => ({
    x: point[0] * 46,
    y: point[1] * 46,
  }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 46,
});

const KEY = "oracle_trapped_rogue";
const trapped = Object.freeze({
  presentation_key: KEY,
  class_id: 4,
  team_id: 2,
  alive: true,
  position: [2, 3],
  radius: 0.5,
  ultimate_cooldown: 12,
  spawn_shield_remaining: 0,
  statuses: [],
});
const brokenTrap = Object.freeze([
  {
    event_kind: "status_broken_by_damage",
    status_id: "hunter_trap_stun",
    recipient_anchor: { presentation_key: KEY },
  },
]);

/** @param {Record<string, any>} [overrides] */
function marks(overrides = {}) {
  return /** @type {any[]} */ (
    statuses.records({
      part: "status-marks",
      scene: { agents: [trapped] },
      transform,
      events: brokenTrap,
      ...overrides,
    })
  );
}

/** @param {any} record @param {string} id */
function pieceById(record, id) {
  return record.pieces.find((/** @type {any} */ entry) => entry.id === id);
}

test("a broken trap shatters: Frozen pieces out at Settle in 60 ms, crack at Impact, shards at Settle", () => {
  const [record] = marks();
  assert.ok(record);
  const frozen = record.pieces.filter((/** @type {any} */ entry) =>
    entry.id.startsWith("frozen."),
  );
  assert.ok(frozen.length >= 2);
  for (const entry of frozen) {
    assert.equal(entry.side, "before", entry.id);
    assert.equal(entry.fadeMs, statuses.RUNE_ZONES.shatterFadeMs, entry.id);
    assert.equal(entry.beat ?? "settle", "settle", entry.id);
  }
  const crack = pieceById(record, "shatter.crack");
  assert.deepEqual(
    [crack.side, crack.beat, crack.fadeMs, crack.opacity, crack.paint],
    ["after", "impact", 60, 0.8, "rune-stroke--ice-crack"],
  );
  const shards = pieceById(record, "shatter.shards");
  assert.deepEqual(
    [shards.side, shards.beat ?? "settle", shards.opacity],
    ["after", "settle", 0.45],
  );
});

test("Freezing Trap Break off cuts the ice at Settle and draws no crack or shards", () => {
  const [record] = marks({ switches: { freezingTrapBreak: false } });
  assert.ok(record);
  for (const entry of record.pieces) {
    if (entry.id.startsWith("frozen.")) assert.equal(entry.fadeMs, 1, entry.id);
  }
  assert.equal(pieceById(record, "shatter.crack"), undefined);
  assert.equal(pieceById(record, "shatter.shards"), undefined);
  // Status Marks off also leaves out the Shatter echo (it is part of the marks).
  const [ringOnly] = marks({ switches: { statusMarks: false } });
  assert.equal(pieceById(ringOnly, "shatter.crack"), undefined);
});

test("Agent POV: a body that comes into sight shows no Shatter and no fade from the dark", () => {
  const [record] = marks({
    scene: { audience: "agent_pov", agents: [trapped] },
    startSeen: () => false,
  });
  assert.ok(record);
  assert.equal(
    record.pieces.every((/** @type {any} */ entry) => entry.side === "steady"),
    true,
  );
  assert.equal(pieceById(record, "shatter.crack"), undefined);
  // An unseen champion is not in the POV scene: no record at all.
  assert.deepEqual(marks({ scene: { audience: "agent_pov", agents: [] } }), []);
});

test("build writes the crack's Impact beat, the Settle beats and export-safe SVG", () => {
  const records = marks();
  const [group] = /** @type {any[]} */ (
    statuses.build(records, /** @type {any} */ (fakeDocument))
  );
  const nodes = /** @type {FakeElement[]} */ (group.children);
  const crack = nodes.find(
    (node) => node.getAttribute("data-rune-piece") === "shatter.crack",
  );
  assert.ok(crack);
  assert.equal(crack.getAttribute("data-beat-hold"), "impact");
  assert.equal(crack.getAttribute("data-beat-side"), "after");
  assert.equal(crack.getAttribute("data-beat-fade"), "60");
  assert.equal(crack.getAttribute("opacity"), "0.8");
  const shell = nodes.find(
    (node) => node.getAttribute("data-rune-piece") === "frozen.shell",
  );
  assert.ok(shell);
  assert.equal(shell.getAttribute("data-beat-hold"), "settle");
  assert.equal(shell.getAttribute("data-beat-side"), "before");
  assert.equal(shell.getAttribute("opacity"), "0");
  const svg = group.serialize();
  assert.doesNotMatch(svg, /url\(|gradient|<mask|<filter|<image|mix-blend|\bid=/iu);
});

test("a repeat render gives equal records and the Shatter adds only two nodes", () => {
  assert.deepEqual(marks(), marks());
  const [withShatter] = marks();
  const [withoutShatter] = marks({ switches: { freezingTrapBreak: false } });
  assert.equal(withShatter.pieces.length - withoutShatter.pieces.length, 2);
  assert.ok(withShatter.pieces.length <= statuses.BUDGET.nodesPerRecord);
});
