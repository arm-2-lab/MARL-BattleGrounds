/**
 * @file Checks the Spawns and Bases look (Moon Waystones, `src/looks/spawns.js`):
 * records() states and opacities per claimed pad (Dormant 22%, Vigil 40% with
 * pips when Vigil Pips is off, quiet when it is on, Wake 60%), the rim gap facing
 * the nearer floor edge, Team B's extra inner ring, beat holds for a return and a
 * fall this tick, Spawn Waystones off (a plain ring), the Agent POV rule (a pad
 * whose champion is unseen is Dormant with no pips and no champion in its tip),
 * the Call event part (only for an agent_respawned pulse, three animations with
 * fill "backwards" in normal motion, no blaze in reduced motion, none when
 * settled), the "Spawn Pad" tip, and build() export safety.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as spawns from "../src/looks/spawns.js";

class FakeElement {
  /** @param {string} tagName */
  constructor(tagName) {
    this.tagName = tagName;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    this.textContent = "";
    /** @type {FakeElement | null} */
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
  hasAttribute(name) {
    return this.attributes.has(name);
  }
  /** @param {string} name */
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    this.children.push(...nodes);
  }
  /** @returns {FakeElement | null} */
  get firstElementChild() {
    return this.children[0] ?? null;
  }
  /** @returns {FakeElement | null} */
  get lastChild() {
    return this.children.at(-1) ?? null;
  }
  getBoundingClientRect() {
    return { left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0 };
  }
}

const doc = /** @type {Document} */ (
  /** @type {unknown} */ ({
    createElementNS: (/** @type {string} */ _namespace, /** @type {string} */ tag) =>
      new FakeElement(tag),
  })
);

/** @param {FakeElement} node @returns {FakeElement[]} */
function walk(node) {
  return [node, ...node.children.flatMap(walk)];
}

/** @param {ReadonlyArray<unknown>} nodes */
function assertExportSafe(nodes) {
  for (const node of nodes.flatMap((root) => walk(/** @type {FakeElement} */ (root)))) {
    assert.ok(
      ![
        "linearGradient",
        "radialGradient",
        "mask",
        "filter",
        "image",
        "clipPath",
        "pattern",
      ].includes(node.tagName),
      node.tagName,
    );
    assert.equal(node.attributes.has("id"), false);
    for (const value of node.attributes.values()) {
      assert.equal(value.includes("url("), false, value);
    }
  }
}

const transform = createViewportTransform({
  worldWidth: 20,
  worldHeight: 10,
  viewportWidth: 1440,
  viewportHeight: 900,
  padding: 24,
});

/** @param {number} team @param {number} slot @param {number} x @param {string | null} key @param {boolean} alive */
function pad(team, slot, x, key, alive) {
  return {
    team_id: team,
    team_local_slot: slot,
    position: [x, 2 + slot * 2],
    configured_active: true,
    currently_alive: alive,
    assigned_presentation_key: key,
    assigned_public_agent_id:
      key === null ? null : `agent-slot-${slot + (team - 1) * 5}`,
    spawn_shield_remaining: 0,
  };
}

const SCENE = Object.freeze({
  audience: "researcher",
  agents: [
    {
      presentation_key: "a0",
      public_agent_id: "agent-slot-0",
      class_id: 1,
      display_agent_id: "0",
      team_id: 1,
    },
    {
      presentation_key: "a1",
      public_agent_id: "agent-slot-1",
      class_id: 2,
      display_agent_id: "1",
      team_id: 1,
    },
    {
      presentation_key: "a2",
      public_agent_id: "agent-slot-2",
      class_id: 3,
      display_agent_id: "2",
      team_id: 1,
    },
    {
      presentation_key: "b0",
      public_agent_id: "agent-slot-5",
      class_id: 4,
      display_agent_id: "5",
      team_id: 2,
    },
  ],
  spawn_pads: [
    pad(1, 0, 1.5, "a0", true),
    pad(1, 1, 1.5, "a1", false),
    pad(1, 2, 1.5, "a2", true),
    pad(2, 0, 18.5, "b0", true),
    {
      ...pad(2, 1, 18.5, null, false),
      configured_active: false,
      assigned_public_agent_id: null,
    },
  ],
  respawn_waves: [
    { team_id: 1, team_index: 0, period_steps: 5, countdown_steps: 2 },
    { team_id: 2, team_index: 1, period_steps: 5, countdown_steps: 0 },
  ],
});

/** @param {Record<string, any>} input */
function recordsOf(input = {}) {
  return spawns.records({ scene: SCENE, transform, audience: "researcher", ...input });
}

test("each claimed pad is Dormant, Vigil or Wake at the store opacities", () => {
  const records = recordsOf({
    filters: { vigil_pips: false },
    returns: [{ presentationKey: "a2" }],
  });
  assert.equal(records.length, 4, "an inactive slot has no waystone");
  assert.deepEqual(
    records.map((record) => [record.slot, record.teamId, record.state, record.opacity]),
    [
      [0, 1, "dormant", 0.22],
      [1, 1, "vigil", 0.4],
      [2, 1, "wake", 0.6],
      [0, 2, "dormant", 0.22],
    ],
  );
  const vigil = records[1];
  assert.equal(
    (vigil.pips.lit.match(/M/g) ?? []).length,
    3,
    "countdown 2 means 3 ticks left",
  );
  assert.equal((vigil.pips.hollow.match(/M/g) ?? []).length, 2);
  assert.equal(vigil.keystoneOpacity, 0);
  assert.deepEqual(
    { ...records[2].hold },
    { kind: "return", key: "a2", fromOpacity: 0.4 / 0.6, keystoneHeld: false },
  );
});

test("while Vigil Pips counts on the body a waiting stone stays quiet", () => {
  const records = recordsOf({ falls: ["a1"] });
  const vigil = records[1];
  assert.equal(vigil.state, "vigil");
  assert.equal(vigil.opacity, 0.22);
  assert.equal(vigil.pips, null);
  assert.equal(vigil.hold, null, "a quiet stone does not change on the fall");
  assert.equal(vigil.keystoneOpacity, 0.22);
});

test("a fall this tick holds the Dormant look until the Fall beat", () => {
  const records = recordsOf({ falls: ["a1"], filters: { vigil_pips: false } });
  assert.deepEqual(
    { ...records[1].hold },
    { kind: "fall", key: "a1", fromOpacity: 0.22 / 0.4, keystoneHeld: true },
  );
  const [, group] = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (spawns.build(records, doc))
  );
  const [rune, keystone, pips] = group.children;
  assert.equal(rune.getAttribute("data-beat-hold"), "fall");
  assert.equal(rune.getAttribute("data-beat-side"), "after");
  assert.equal(rune.getAttribute("data-beat-fade"), "300");
  assert.equal(keystone.getAttribute("data-beat-side"), "before");
  assert.equal(keystone.getAttribute("opacity"), "0");
  assert.equal(pips.getAttribute("data-beat-from"), "0");
});

test("the rim gap faces the nearer floor edge and Team B has a second inner ring", () => {
  const records = recordsOf();
  const r = 0.56 * transform.worldLengthToScreen(1);
  const left = records[0];
  const start = [
    left.cx + r * Math.cos((220 * Math.PI) / 180),
    left.cy + r * Math.sin((220 * Math.PI) / 180),
  ];
  assert.ok(
    left.body.startsWith(
      `M${Number(start[0].toFixed(3))} ${Number(start[1].toFixed(3))}`,
    ),
  );
  const right = records[3];
  const rightStart = [
    right.cx + r * Math.cos((40 * Math.PI) / 180),
    right.cy + r * Math.sin((40 * Math.PI) / 180),
  ];
  assert.ok(
    right.body.startsWith(
      `M${Number(rightStart[0].toFixed(3))} ${Number(rightStart[1].toFixed(3))}`,
    ),
  );
  assert.equal((left.body.match(/a/g) ?? []).length, 2, "Team A: one inner ring");
  assert.equal((right.body.match(/a/g) ?? []).length, 4, "Team B: two inner rings");
});

test("Spawn Waystones off draws a plain ring on every claimed pad", () => {
  const records = recordsOf({ filters: { spawn_waystones: false } });
  assert.deepEqual(
    records.map((record) => record.kind),
    ["spawn-pad-plain", "spawn-pad-plain", "spawn-pad-plain", "spawn-pad-plain"],
  );
});

test("Agent POV: a pad whose champion is unseen is Dormant with no pips and no champion", () => {
  const pov = {
    ...SCENE,
    audience: "agent_pov",
    agents: SCENE.agents.filter(
      (agent) => agent.team_id === 1 && agent.presentation_key !== "a1",
    ),
  };
  const records = spawns.records({
    scene: pov,
    transform,
    audience: "agent_pov",
    filters: { vigil_pips: false },
    returns: ["b0"],
  });
  const fallen = records.find((record) => record.slot === 1 && record.teamId === 1);
  const enemy = records.find((record) => record.teamId === 2);
  for (const record of [fallen, enemy]) {
    assert.equal(record?.state, "dormant");
    assert.equal(record?.seen, false);
    assert.equal(record?.pips, null);
    assert.equal(record?.hold, null);
    const tip = /** @type {Record<string, any>} */ (
      spawns.tooltip(/** @type {any} */ (record))
    );
    const text = JSON.stringify(tip);
    assert.equal(text.includes("Returns Here"), false);
    assert.equal(text.includes("Waiting"), false);
    assert.ok(text.includes("Your team cannot see this pad's champion now."));
  }
  const seenTip = JSON.stringify(spawns.tooltip(/** @type {any} */ (records[0])));
  assert.ok(seenTip.includes("Mage · ID 0"));
  assert.equal(spawns.tooltip(/** @type {any} */ ({ kind: "anything" })), null);
});

test("the Call answers only an agent_respawned pulse and is timed from its Return beat", () => {
  const event = {
    kind: "semantic_pulse",
    cueSemantic: "agent_respawned",
    anchor: { x: 100, y: 200 },
    recipientIdentity: { team_id: 2 },
    phaseStart: 2400,
  };
  assert.deepEqual(
    spawns.records({ part: "call", event: { ...event, cueSemantic: "agent_died" } }),
    [],
  );
  assert.deepEqual(
    spawns.records({ event }),
    [],
    "without part the look reads the scene only",
  );
  assert.deepEqual(
    spawns.records({ part: "call", event, filters: { respawn_wave: false } }),
    [],
  );
  const records = spawns.records({ part: "call", event, transform });
  assert.equal(records[0].teamId, 2);
  /** @param {"normal" | "reduced" | "off"} motionMode @param {boolean} settled */
  const mount = (motionMode, settled) => ({
    group: new FakeElement("g"),
    connector: new FakeElement("g"),
    underlay: null,
    plan: { epochKey: "e1" },
    options: { motionMode, settled },
    animationSpecs: /** @type {any[]} */ ([]),
  });
  const normal = mount("normal", false);
  const nodes = spawns.build(records, doc, /** @type {any} */ (normal));
  assert.equal(normal.group.children.length, 1);
  assertExportSafe(nodes);
  assert.equal(normal.animationSpecs.length, 3);
  assert.ok(normal.animationSpecs.every((spec) => spec.options.fill === "backwards"));
  assert.deepEqual(
    normal.animationSpecs.map((spec) => spec.options.delay),
    [2400, 2460, 2460],
  );
  assert.equal(normal.animationSpecs[0].keyframes[1].opacity, 0.9);
  for (const node of walk(/** @type {any} */ (nodes[0])).slice(1)) {
    if (node.hasAttribute("opacity"))
      assert.equal(node.getAttribute("opacity"), "0", "END is invisible");
  }
  const reduced = mount("reduced", false);
  spawns.build(records, doc, /** @type {any} */ (reduced));
  assert.equal(reduced.animationSpecs.length, 2, "reduced motion: no blaze");
  const settled = mount("normal", true);
  spawns.build(records, doc, /** @type {any} */ (settled));
  assert.equal(settled.animationSpecs.length, 0);
});

test("waystones build export-safe with a Spawn Pad tip on each hit circle", () => {
  const records = recordsOf({ filters: { vigil_pips: false } });
  const nodes = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (spawns.build(records, doc))
  );
  assertExportSafe(nodes);
  const hits = nodes
    .flatMap(walk)
    .filter((node) => node.getAttribute("class") === "waystone__hit");
  assert.equal(hits.length, 4);
  const moved = spawns.records({
    scene: SCENE,
    transform: createViewportTransform({
      worldWidth: 20,
      worldHeight: 10,
      viewportWidth: 960,
      viewportHeight: 600,
      padding: 24,
    }),
    filters: { vigil_pips: false },
  });
  spawns.update(/** @type {any} */ (nodes), moved);
  assert.equal(hits[0].getAttribute("cx"), String(moved[0].cx));
});
