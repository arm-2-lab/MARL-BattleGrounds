/**
 * @file Check Barrier Held (switch barrier_held, on by default) in the Rune
 * Zones look (src/looks/statuses.js). In Oracle View the host's exact rows
 * decide it: an ally that survived damage which, without another champion's
 * Guardian's Barrier, would have reached its tick-start life (less healing)
 * gets a bronze collar: one batch closes at Impact over 80 ms and rests at 35%
 * (the paused frame's faint collar) and a second batch at full strength
 * flashes once over 640 ms (two beat overrides in all). The
 * covering Warrior's Ward Circle gets a play-only stud flash. Not held: a blow
 * the barrier did not decide, a Warrior covered only by itself, the switch off.
 * Agent POV gets no damage-source rows, so the moment is worked out from seen
 * champions only (tick-start spots, 2.0 units, damage / 0.85 per wall); its
 * health rows carry only the net change, which counts as damage with no
 * healing (a heal can hide a moment, never make one up). An unseen Warrior or
 * one without a tick-start spot counts for nothing, and the Oracle damage rows
 * are never read there. build puts each batch in one g.rune-batch carrying
 * the Impact beat, the SVG is export-safe, a repeat render is equal, and the
 * collar adds four pieces in two batches.
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
    /** @type {Record<string, string>} */
    this.dataset = {};
    this.ownerDocument = fakeDocument;
    this.parentElement = null;
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
  /** @param {string} name */
  hasAttribute(name) {
    return this.attributes.has(name);
  }
  getBoundingClientRect() {
    return { x: 0, y: 0, width: 0, height: 0, left: 0, top: 0, right: 0, bottom: 0 };
  }
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    this.children.push(...nodes);
  }
  /** @param {...FakeElement} nodes */
  replaceChildren(...nodes) {
    this.children = [...nodes];
  }
  /** @param {string} _type @param {unknown} _listener */
  addEventListener(_type, _listener) {}
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

/** @param {string} key @param {number} classId @param {number} teamId @param {number[]} position */
function body(key, classId, teamId, position) {
  return {
    presentation_key: key,
    class_id: classId,
    team_id: teamId,
    alive: true,
    position,
    radius: 0.5,
    ultimate_cooldown: 0,
    spawn_shield_remaining: 0,
    statuses: [],
  };
}

const ally = body("ally", 3, 1, [0, 0]);
const wall = body("wall", 2, 1, [1, 0]);
const enemy = body("enemy", 4, 2, [3, 0]);

/** @param {string} key @param {number[]} position */
function startAnchor(key, position) {
  return { phase: "transition_start", presentation_key: key, position };
}

/** @param {number} before */
function healthRow(before) {
  return {
    event_kind: "recipient_health_resolution",
    recipient_anchor: startAnchor("ally", [0, 0]),
    transition_start_health: before,
    total_effective_damage: 12.71,
    total_effective_healing: 0,
    health_after_combat_resolution: before - 12.71,
  };
}

/** @param {number} before @param {number} net */
function povHealthRow(before, net) {
  return {
    event_kind: "recipient_health_resolution",
    recipient_anchor: startAnchor("ally", [0, 0]),
    transition_start_health: before,
    health_after_combat_resolution: before + net,
    realized_net_health_change: net,
  };
}

// The host's exact row: 14.95 damage before the barrier, cut by the Warrior.
const damageRow = Object.freeze({
  event_kind: "source_damage_output",
  source_anchor: startAnchor("enemy", [3, 0]),
  recipient_anchor: startAnchor("ally", [0, 0]),
  raw_damage_output: 13,
  source_modified_damage_output: 14.95,
  recipient_damage_modifier: 0.85,
  mage_damage_aura_covering_emitters: [],
  warrior_mitigation_aura_covering_emitters: [startAnchor("wall", [1, 0])],
});

const fields = Object.freeze([
  {
    presentation_key: "field_wall",
    source_presentation_key: "wall",
    source_class_id: 2,
    aura_id: "warrior_damage_mitigation",
    token_id: "warrior_damage_mitigation",
    center: [1, 0],
    radius: 2,
  },
  {
    presentation_key: "field_mage",
    source_presentation_key: "mage",
    source_class_id: 1,
    aura_id: "mage_damage_amplification",
    token_id: "mage_damage_amplification",
    center: [0, 1],
    radius: 2,
  },
]);

/** @param {Record<string, any>} overrides */
function marks(overrides) {
  return /** @type {any[]} */ (
    statuses.records({
      part: "status-marks",
      scene: { agents: [ally, wall, enemy] },
      transform,
      ...overrides,
    })
  );
}

/** @param {any[]} records @param {string} key */
function collarOf(records, key) {
  const record = records.find((entry) => entry.presentationKey === key);
  return (record?.pieces ?? []).filter((/** @type {any} */ entry) =>
    entry.id.startsWith("barrier."),
  );
}

/** @param {Record<string, any>} overrides */
function circles(overrides) {
  return /** @type {any[]} */ (
    statuses.records({
      part: "aura-fields",
      transform,
      fields,
      agents: [ally, wall, enemy],
      tooltipInput: (/** @type {unknown} */ row) => ({
        field: row,
        sourceAgent: null,
        audience: "researcher",
      }),
      ...overrides,
    })
  );
}

test("Oracle View: a life the barrier decided gets the collar, timed at Impact", () => {
  const records = marks({ events: [damageRow, healthRow(14.25)] });
  const collar = collarOf(records, "ally");
  assert.deepEqual(
    collar.map((/** @type {any} */ entry) => [
      entry.id,
      entry.batch,
      entry.side,
      entry.beat,
      entry.fadeMs,
      entry.pulse ?? null,
      entry.opacity,
      entry.paint,
    ]),
    [
      [
        "barrier.collar",
        "barrier",
        "after",
        "impact",
        80,
        null,
        0.35,
        "rune-stroke--warrior",
      ],
      [
        "barrier.rivets",
        "barrier",
        "after",
        "impact",
        80,
        null,
        0.35,
        "rune-stroke--warrior",
      ],
      [
        "barrier.collar.flash",
        "barrier.flash",
        "after",
        "impact",
        640,
        1,
        1,
        "rune-stroke--warrior",
      ],
      [
        "barrier.rivets.flash",
        "barrier.flash",
        "after",
        "impact",
        640,
        1,
        1,
        "rune-stroke--warrior",
      ],
    ],
  );
  // Nothing on the Warrior or the enemy.
  assert.deepEqual(collarOf(records, "wall"), []);
  assert.deepEqual(collarOf(records, "enemy"), []);
  // The Warrior's Ward Circle flashes its studs; the Mage's does not.
  const [wallCircle, mageCircle] = circles({ events: [damageRow, healthRow(14.25)] });
  assert.equal(wallCircle.barrierFlash, true);
  assert.equal(mageCircle.barrierFlash, false);
});

test("not held: the barrier was not the difference, only self-cover, or the switch is off", () => {
  assert.deepEqual(collarOf(marks({ events: [damageRow, healthRow(20)] }), "ally"), []);
  const selfOnly = {
    ...damageRow,
    recipient_anchor: startAnchor("wall", [1, 0]),
  };
  const wallHealth = {
    ...healthRow(14.25),
    recipient_anchor: startAnchor("wall", [1, 0]),
  };
  assert.deepEqual(collarOf(marks({ events: [selfOnly, wallHealth] }), "wall"), []);
  const off = { barrierHeld: false };
  assert.deepEqual(
    collarOf(marks({ events: [damageRow, healthRow(14.25)], switches: off }), "ally"),
    [],
  );
  const [wallCircle] = circles({
    events: [damageRow, healthRow(14.25)],
    switches: off,
  });
  assert.equal(wallCircle.barrierFlash, false);
});

test("Agent POV works it out from seen champions only and never reads damage rows", () => {
  const trajectories = [
    { agent_presentation_key: "ally", transition_start: startAnchor("ally", [0, 0]) },
    { agent_presentation_key: "wall", transition_start: startAnchor("wall", [1, 0]) },
  ];
  const pov = {
    scene: { audience: "agent_pov", agents: [ally, wall, enemy] },
    startSeen: () => true,
  };
  // 12.71 / 0.85 = 14.95 would have taken all 14.25 life: held.
  assert.equal(
    collarOf(marks({ ...pov, events: [healthRow(14.25)], trajectories }), "ally")
      .length,
    4,
  );
  // Real Agent POV rows: only the net change (-6.8 of 6.89 life; 6.8 / 0.85 = 8).
  assert.equal(
    collarOf(
      marks({ ...pov, events: [povHealthRow(6.89, -6.8)], trajectories }),
      "ally",
    ).length,
    4,
  );
  // A smaller net loss the barrier did not decide (5 / 0.85 < 6.89): no collar.
  assert.deepEqual(
    collarOf(marks({ ...pov, events: [povHealthRow(6.89, -5)], trajectories }), "ally"),
    [],
  );
  // The Warrior unseen (not in the POV scene): no wall, no collar.
  assert.deepEqual(
    collarOf(
      marks({
        scene: { audience: "agent_pov", agents: [ally, enemy] },
        startSeen: () => true,
        events: [damageRow, healthRow(14.25)],
        trajectories,
      }),
      "ally",
    ),
    [],
  );
  // No tick-start spot for the Warrior: no wall either, even with the
  // Oracle-only damage row present (Agent POV never reads it).
  assert.deepEqual(
    collarOf(
      marks({ ...pov, events: [damageRow, healthRow(14.25)], trajectories: [] }),
      "ally",
    ),
    [],
  );
  // Out of reach at tick start (2.5 units): no wall.
  const far = [
    trajectories[0],
    { agent_presentation_key: "wall", transition_start: startAnchor("wall", [2.5, 0]) },
  ];
  assert.deepEqual(
    collarOf(marks({ ...pov, events: [healthRow(14.25)], trajectories: far }), "ally"),
    [],
  );
});

test("build writes the Impact beats, rests faint and flashes once; export-safe SVG", () => {
  const records = marks({ events: [damageRow, healthRow(14.25)] });
  const nodes = /** @type {any[]} */ (
    statuses.build(records, /** @type {any} */ (fakeDocument))
  );
  const group = nodes.find((node) => node.getAttribute("data-marks-for") === "ally");
  const batches = /** @type {FakeElement[]} */ (group.children).filter(
    (node) => node.getAttribute("class") === "rune-batch",
  );
  assert.deepEqual(
    batches.map((node) => [
      node.getAttribute("data-rune-batch"),
      node.getAttribute("data-beat-hold"),
      node.getAttribute("data-beat-side"),
      node.getAttribute("data-beat-key"),
      node.getAttribute("data-beat-from"),
      node.getAttribute("data-beat-pulse"),
      node.getAttribute("data-beat-fade"),
      node.getAttribute("opacity"),
    ]),
    [
      ["barrier", "impact", "after", "ally", "0", null, "80", null],
      ["barrier.flash", "impact", "after", "ally", null, "1", "640", "0"],
    ],
  );
  // Inside a batch each piece keeps only its own resting opacity.
  const [rest, flash] = batches;
  assert.deepEqual(
    rest.children.map((node) => [
      node.getAttribute("opacity"),
      node.getAttribute("data-beat-hold"),
    ]),
    [
      ["0.35", null],
      ["0.35", null],
    ],
  );
  assert.deepEqual(
    flash.children.map((node) => node.getAttribute("opacity")),
    [null, null],
  );
  const ring = /** @type {any[]} */ (
    statuses.build(
      circles({ events: [damageRow, healthRow(14.25)] }),
      /** @type {any} */ (fakeDocument),
    )
  );
  const marksGroup = ring[1];
  const studFlash = marksGroup.children.find(
    (/** @type {FakeElement} */ node) =>
      node.getAttribute("class") === "ward-circle__barrier-flash",
  );
  assert.ok(studFlash);
  assert.deepEqual(
    [
      studFlash.getAttribute("opacity"),
      studFlash.getAttribute("data-beat-hold"),
      studFlash.getAttribute("data-beat-pulse"),
      studFlash.getAttribute("data-beat-fade"),
    ],
    ["0", "impact", "1", "120"],
  );
  for (const node of [group, marksGroup]) {
    assert.doesNotMatch(
      node.serialize(),
      /url\(|gradient|<mask|<filter|<image|mix-blend|\bid=/iu,
    );
  }
});

test("a repeat render is equal and the collar adds four pieces in two batches", () => {
  const input = { events: [damageRow, healthRow(14.25)] };
  assert.deepEqual(marks(input), marks(input));
  const held = marks(input).find((entry) => entry.presentationKey === "ally");
  const plain = marks({ ...input, switches: { barrierHeld: false } }).find(
    (entry) => entry.presentationKey === "ally",
  );
  assert.equal(held.pieces.length - (plain?.pieces.length ?? 0), 4);
  assert.equal(
    new Set(
      held.pieces
        .filter((/** @type {any} */ entry) => entry.id.startsWith("barrier."))
        .map((/** @type {any} */ entry) => entry.batch),
    ).size,
    2,
  );
  assert.ok(held.pieces.length <= statuses.BUDGET.nodesPerRecord);
});
