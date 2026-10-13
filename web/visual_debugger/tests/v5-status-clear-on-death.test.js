/**
 * @file Check Status Clear on Death (switch status_clear_on_death, off by
 * default) in the Rune Zones look (src/looks/statuses.js). A champion that
 * falls this tick in this audience's view keeps its tick-start marks and Rune
 * Ring until its own Fall beat: tick-start statuses are rebuilt from this
 * tick's events only (cleared by the death and not applied this tick: ticks
 * unknown; aged out: 1 tick), and the Rune Ring had one more cooldown tick.
 * The pieces form one batch (one beat override): off, it cuts at the Fall
 * (beat "fall", 1 ms); on, it gives way over 53 ms while an Ash batch (Ink
 * outlines kept) flashes in to each piece's own opacity and fades by 160 ms. A
 * corpse that did not fall this tick, and in Agent POV a fall the team did not
 * watch from tick start, give no record. build puts each batch in one
 * g.rune-batch that carries the Fall beat keyed by the champion (the Ash batch
 * as a pulse) while its pieces keep only their own opacity; the SVG is
 * export-safe; a repeat render is equal; Ash at most doubles the pieces.
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

const KEY = "oracle_falling_rogue";
// A Rogue corpse at END: no statuses, its Ultimate still charging (27 ticks).
const corpse = Object.freeze({
  presentation_key: KEY,
  class_id: 4,
  team_id: 1,
  alive: false,
  position: [3, 2],
  radius: 0.5,
  ultimate_cooldown: 27,
  spawn_shield_remaining: 0,
  statuses: [],
});

/** @param {string} kind @param {string} [statusId] */
function row(kind, statusId) {
  return {
    event_kind: kind,
    recipient_anchor: { phase: "successor", presentation_key: KEY },
    ...(statusId === undefined ? {} : { status_id: statusId }),
  };
}

// It fell: its Crippling Poison (Healing Reduction) was cleared, its Crippling
// Poison (Stun) aged out this tick, and a fresh Charge (Slow) landed this tick.
const fallRows = Object.freeze([
  row("agent_died"),
  row("status_cleared_by_new_death", "rogue_poison_anti_heal"),
  row("status_aged_to_zero", "rogue_poison_stun"),
  row("status_applied", "warrior_charge_slow"),
  row("status_cleared_by_new_death", "warrior_charge_slow"),
]);

/** @param {Record<string, any>} [overrides] */
function marks(overrides = {}) {
  return /** @type {any[]} */ (
    statuses.records({
      part: "status-marks",
      scene: { agents: [corpse] },
      transform,
      events: fallRows,
      ...overrides,
    })
  );
}

/** @param {any} record @returns {string[]} */
function ids(record) {
  return record.pieces.map((/** @type {any} */ entry) => entry.id);
}

test("a falling champion keeps its tick-start marks and Rune Ring until its Fall", () => {
  const [record] = marks();
  assert.ok(record);
  assert.equal(record.presentationKey, KEY);
  const pieceIds = ids(record);
  // Withering (cleared, ticks unknown) and the Poison stars (aged out) were there
  // at tick start; the Charge chain landed this tick, so it was not.
  assert.ok(pieceIds.some((id) => id.startsWith("withering.")));
  assert.ok(pieceIds.some((id) => id.startsWith("sapped.star.")));
  assert.equal(
    pieceIds.some((id) => id.startsWith("roots.")),
    false,
  );
  // START cooldown 28: no lit arc yet, only the 12% track.
  assert.ok(pieceIds.includes("ring.track"));
  assert.equal(
    pieceIds.some((id) => id.startsWith("ring.arc.")),
    false,
  );
  // Off (the default): one batch cuts out at the champion's Fall.
  for (const entry of record.pieces) {
    assert.deepEqual(
      [entry.batch, entry.side, entry.beat, entry.fadeMs, entry.fromOpacity],
      ["fall", "before", "fall", 1, 1],
    );
  }
  assert.equal(
    pieceIds.some((id) => id.endsWith(".ash")),
    false,
  );
});

test("Status Clear on Death on turns the marks to Ash over 160 ms after the Fall", () => {
  const [off] = marks();
  const [on] = marks({ switches: { statusClearOnDeath: true } });
  const colored = on.pieces.filter(
    (/** @type {any} */ entry) => !entry.id.endsWith(".ash"),
  );
  const ash = on.pieces.filter((/** @type {any} */ entry) => entry.id.endsWith(".ash"));
  assert.equal(colored.length, off.pieces.length);
  assert.equal(ash.length, colored.length);
  for (const entry of colored) {
    assert.deepEqual(
      [entry.batch, entry.side, entry.beat, entry.fadeMs],
      ["fall", "before", "fall", statuses.RUNE_ZONES.giveWayMs],
    );
  }
  for (const entry of ash) {
    const source = colored.find(
      (/** @type {any} */ other) => `${other.id}.ash` === entry.id,
    );
    assert.ok(source, entry.id);
    assert.deepEqual(
      [entry.batch, entry.side, entry.beat, entry.fadeMs, entry.pulse, entry.opacity],
      ["fall.ash", "after", "fall", statuses.RUNE_ZONES.ashFadeMs, 1, source.opacity],
    );
    for (const name of entry.paint.split(" ")) {
      assert.ok(name.endsWith("--ash") || name.endsWith("--ink"), name);
    }
  }
  // Ash at most doubles the pieces of the falling body.
  assert.ok(on.pieces.length <= 2 * off.pieces.length);
});

test("no Fall, no record: a lying corpse, Agent POV from the dark, and both switches off", () => {
  assert.deepEqual(marks({ events: [] }), []);
  assert.deepEqual(
    marks({
      scene: { audience: "agent_pov", agents: [corpse] },
      startSeen: () => false,
    }),
    [],
  );
  const [watched] = marks({
    scene: { audience: "agent_pov", agents: [corpse] },
    startSeen: (/** @type {string} */ key) => key === KEY,
  });
  assert.ok(watched);
  // An unseen champion is not in the POV scene, so its fall draws nothing.
  assert.deepEqual(marks({ scene: { audience: "agent_pov", agents: [] } }), []);
  assert.deepEqual(
    marks({
      switches: { statusMarks: false, cooldownEffects: false, barrierHeld: false },
    }),
    [],
  );
});

test("build keys the Fall beat to the champion and writes the Ash flash", () => {
  const [record] = marks({ switches: { statusClearOnDeath: true } });
  const [group] = /** @type {any[]} */ (
    statuses.build([record], /** @type {any} */ (fakeDocument))
  );
  // Two batches, so two beat overrides however many pieces there are.
  const batches = /** @type {FakeElement[]} */ (group.children);
  assert.deepEqual(
    batches.map((node) => [
      node.getAttribute("class"),
      node.getAttribute("data-rune-batch"),
    ]),
    [
      ["rune-batch", "fall"],
      ["rune-batch", "fall.ash"],
    ],
  );
  const [colored, ash] = batches;
  assert.deepEqual(
    [
      colored.getAttribute("data-beat-hold"),
      colored.getAttribute("data-beat-key"),
      colored.getAttribute("data-beat-side"),
      colored.getAttribute("data-beat-from"),
      colored.getAttribute("data-beat-fade"),
      colored.getAttribute("opacity"),
    ],
    ["fall", KEY, "before", "1", "53", "0"],
  );
  assert.deepEqual(
    [
      ash.getAttribute("data-beat-hold"),
      ash.getAttribute("data-beat-key"),
      ash.getAttribute("data-beat-side"),
      ash.getAttribute("data-beat-pulse"),
      ash.getAttribute("data-beat-from"),
      ash.getAttribute("data-beat-fade"),
      ash.getAttribute("opacity"),
    ],
    ["fall", KEY, "after", "1", null, "160", "0"],
  );
  assert.equal(colored.children.length, ash.children.length);
  assert.equal(colored.children.length, record.pieces.length / 2);
  for (const node of [...colored.children, ...ash.children]) {
    assert.equal(node.getAttribute("data-beat-hold"), null);
    assert.match(String(node.getAttribute("class")), /^rune-piece /u);
  }
  for (const node of ash.children) {
    assert.match(String(node.getAttribute("data-rune-piece")), /\.ash$/u);
    for (const name of String(node.getAttribute("class")).split(" ").slice(1)) {
      assert.ok(name.endsWith("--ash") || name.endsWith("--ink"), name);
    }
  }
  assert.doesNotMatch(
    group.serialize(),
    /url\(|gradient|<mask|<filter|<image|mix-blend|\bid=/iu,
  );
});

test("a repeat render is equal and living bodies keep their own beats", () => {
  assert.deepEqual(marks(), marks());
  const living = { ...corpse, alive: true, presentation_key: "oracle_living" };
  const records = /** @type {any[]} */ (
    statuses.records({
      part: "status-marks",
      scene: { agents: [living] },
      transform,
      events: [],
      switches: { statusClearOnDeath: true },
    })
  );
  assert.equal(records.length, 1);
  assert.equal(
    records[0].pieces.some((/** @type {any} */ entry) => entry.beat === "fall"),
    false,
  );
});
