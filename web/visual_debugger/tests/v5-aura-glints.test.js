/**
 * @file Check Aura Glints (switch aura_glints, off by default) on the Ward
 * Circles of the Rune Zones look (src/looks/statuses.js). With the switch on,
 * an aura that helped an ally inside it this tick glints one stud: the stud
 * (0-11, clockwise from 12 o'clock on screen) nearest the direction from the
 * emitter's tick-start spot to the ally's. A Mage's aura helps an ally that
 * deals damage; a Warrior's helps an ally that takes damage; the emitter never
 * counts itself. Oracle View reads the host's covering emitters; Agent POV
 * works it out from its own ability rows, the scene's teams and its tick-start
 * spots, so an unseen ally, an emitter without a tick-start spot or an ally out
 * of reach gives no glint, and the Oracle damage rows are never read there.
 * build draws a play-only flash (opacity 0, Impact beat, pulse 1, 120 ms); the
 * SVG is export-safe; a repeat render is equal; a glint adds one node.
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
    spawn_shield_remaining: 0,
    statuses: [],
  };
}

// Screen y grows down: the Mage's ally stands north of it, the Warrior's
// ally east of it.
const mage = body("mage", 1, 1, [0, 0]);
const striker = body("striker", 4, 1, [0, -1]);
const warrior = body("warrior", 2, 1, [5, 0]);
const guarded = body("guarded", 3, 1, [6, 0]);
const enemy = body("enemy", 5, 2, [3, 3]);
const agents = Object.freeze([mage, striker, warrior, guarded, enemy]);

/** @param {string} key @param {number[]} position */
function startAnchor(key, position) {
  return { phase: "transition_start", presentation_key: key, position };
}

const fields = Object.freeze([
  {
    presentation_key: "field_mage",
    source_presentation_key: "mage",
    source_class_id: 1,
    aura_id: "mage_damage_amplification",
    token_id: "mage_damage_amplification",
    center: [0, 0],
    radius: 2,
  },
  {
    presentation_key: "field_warrior",
    source_presentation_key: "warrior",
    source_class_id: 2,
    aura_id: "warrior_damage_mitigation",
    token_id: "warrior_damage_mitigation",
    center: [5, 0],
    radius: 2,
  },
]);

const oracleRows = Object.freeze([
  {
    event_kind: "source_damage_output",
    source_anchor: startAnchor("striker", [0, -1]),
    recipient_anchor: startAnchor("enemy", [3, 3]),
    source_modified_damage_output: 10,
    recipient_damage_modifier: 1,
    mage_damage_aura_covering_emitters: [startAnchor("mage", [0, 0])],
    warrior_mitigation_aura_covering_emitters: [],
  },
  {
    event_kind: "source_damage_output",
    source_anchor: startAnchor("enemy", [3, 3]),
    recipient_anchor: startAnchor("guarded", [6, 0]),
    source_modified_damage_output: 8,
    recipient_damage_modifier: 0.85,
    mage_damage_aura_covering_emitters: [],
    warrior_mitigation_aura_covering_emitters: [startAnchor("warrior", [5, 0])],
  },
]);

/** @param {Record<string, any>} overrides */
function circles(overrides) {
  return /** @type {any[]} */ (
    statuses.records({
      part: "aura-fields",
      transform,
      fields,
      agents,
      tooltipInput: (/** @type {unknown} */ row) => ({
        field: row,
        sourceAgent: null,
        audience: "researcher",
      }),
      ...overrides,
    })
  );
}

const ON = Object.freeze({ auraGlints: true });

test("Oracle View: each aura glints the stud facing the ally it helped", () => {
  const [mageCircle, warriorCircle] = circles({ events: oracleRows, switches: ON });
  assert.equal(mageCircle.glintStud, 0);
  assert.equal(warriorCircle.glintStud, 3);
  // Off by default: no glint at all.
  const [mageOff, warriorOff] = circles({ events: oracleRows });
  assert.deepEqual([mageOff.glintStud, warriorOff.glintStud], [null, null]);
  // The emitter itself never counts: the Mage's own blow under its own aura.
  const selfRow = {
    ...oracleRows[0],
    source_anchor: startAnchor("mage", [0, 0]),
  };
  const [mageSelf] = circles({ events: [selfRow], switches: ON });
  assert.equal(mageSelf.glintStud, null);
});

test("Agent POV derives glints from seen champions only and never reads damage rows", () => {
  const hits = [
    {
      event_kind: "ability_activated",
      ability_component: "basic",
      source_anchor: startAnchor("striker", [0, -1]),
      recipient_anchor: startAnchor("enemy", [3, 3]),
    },
    {
      event_kind: "ability_activated",
      ability_component: "basic",
      source_anchor: startAnchor("enemy", [3, 3]),
      recipient_anchor: startAnchor("guarded", [6, 0]),
    },
  ];
  const trajectories = [
    { agent_presentation_key: "mage", transition_start: startAnchor("mage", [0, 0]) },
    {
      agent_presentation_key: "warrior",
      transition_start: startAnchor("warrior", [5, 0]),
    },
  ];
  const pov = { audience: "agent_pov", switches: ON, trajectories };
  const [mageCircle, warriorCircle] = circles({ ...pov, events: hits });
  assert.deepEqual([mageCircle.glintStud, warriorCircle.glintStud], [0, 3]);
  // Agent POV never reads the Oracle damage rows.
  const [mageRows, warriorRows] = circles({ ...pov, events: oracleRows });
  assert.deepEqual([mageRows.glintStud, warriorRows.glintStud], [null, null]);
  // The helped ally unseen (not in the POV scene): no glint.
  const [mageHidden] = circles({
    ...pov,
    events: hits,
    agents: agents.filter((agent) => agent.presentation_key !== "striker"),
  });
  assert.equal(mageHidden.glintStud, null);
  // No tick-start spot for the emitter: no glint.
  const [mageNoSpot] = circles({ ...pov, events: hits, trajectories: [] });
  assert.equal(mageNoSpot.glintStud, null);
  // Out of reach at tick start (2.5 units): no glint.
  const far = [
    { ...hits[0], source_anchor: startAnchor("striker", [0, -2.5]) },
    hits[1],
  ];
  const [mageFar] = circles({ ...pov, events: far });
  assert.equal(mageFar.glintStud, null);
});

test("build draws a play-only flash on the glinting stud; export-safe SVG", () => {
  const nodes = /** @type {any[]} */ (
    statuses.build(
      circles({ events: oracleRows, switches: ON }),
      /** @type {any} */ (fakeDocument),
    )
  );
  const [mageMarks, warriorMarks] = [nodes[1], nodes[3]];
  for (const [group, kind, stud] of [
    [mageMarks, "mage", "0"],
    [warriorMarks, "warrior", "3"],
  ]) {
    const glint = group.children.find(
      (/** @type {FakeElement} */ node) =>
        node.getAttribute("class") === `ward-circle__glint ward-circle__glint--${kind}`,
    );
    assert.ok(glint, kind);
    assert.deepEqual(
      [
        glint.getAttribute("opacity"),
        glint.getAttribute("data-beat-hold"),
        glint.getAttribute("data-beat-side"),
        glint.getAttribute("data-beat-pulse"),
        glint.getAttribute("data-beat-fade"),
        glint.getAttribute("data-stud"),
      ],
      ["0", "impact", "after", "1", "120", stud],
    );
    assert.doesNotMatch(
      group.serialize(),
      /url\(|gradient|<mask|<filter|<image|mix-blend|\bid=/iu,
    );
  }
});

test("a repeat render is equal and a glint adds one node to its Ward Circle", () => {
  const input = { events: oracleRows, switches: ON };
  assert.deepEqual(circles(input), circles(input));
  const withGlint = /** @type {any[]} */ (
    statuses.build(circles(input), /** @type {any} */ (fakeDocument))
  );
  const without = /** @type {any[]} */ (
    statuses.build(circles({ events: oracleRows }), /** @type {any} */ (fakeDocument))
  );
  assert.equal(withGlint[1].children.length - without[1].children.length, 1);
});
