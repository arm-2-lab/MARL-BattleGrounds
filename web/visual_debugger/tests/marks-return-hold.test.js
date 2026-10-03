/**
 * @file Check that nothing of a returning champion shows at its waystone
 * before its Return beat (src/looks/champions.js and src/looks/hotbar.js).
 * Champions: a returning champion's Return hold sits on its glide wrapper
 * (data-beat-hold "return", its presentation key, side "after", from 0, the
 * 300 ms set-down fade, or a 1 ms cut with Resurrection Effects off), so its
 * token, its Spawn Ward and the marks that ride the wrapper wait together;
 * the token itself carries no Return hold, and a champion that does not
 * return gets no hold. Hotbar: every record names the champion it belongs
 * to (returnKey: the ringed or looked-at champion, the studs' owner, the body
 * a Reach Mark sits on, or your champion for its Order Thread and Step
 * Ghost) and every built node carries a "set-down" hold on that key (side
 * "after", a 1 ms cut), which the choreographer applies only on a tick that
 * champion returns.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as champions from "../src/looks/champions.js";
import * as hotbar from "../src/looks/hotbar.js";

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
    this.classList = { add: () => {} };
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
  addEventListener() {}
  blur() {}
  getBoundingClientRect() {
    return { x: 0, y: 0, width: 0, height: 0, left: 0, top: 0, right: 0, bottom: 0 };
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

const transform = createViewportTransform({
  worldWidth: 20,
  worldHeight: 10,
  viewportWidth: 1000,
  viewportHeight: 500,
  padding: 0,
});

/** @param {string} id @param {number} x @param {Record<string, any>} [more] */
function agent(id, x, more = {}) {
  return {
    display_key: `d-${id}`,
    presentation_key: `pk-${id}`,
    public_agent_id: id,
    team_id: 1,
    class_id: 2,
    position: [x, 5],
    radius: 0.5,
    alive: true,
    life_state: "alive",
    current_health: 200,
    max_health: 200,
    spawn_shield_remaining: 0,
    effective_movement_speed: 1,
    ...more,
  };
}

/** @param {boolean} setDown */
function drawnChampions(setDown) {
  const scene = {
    audience: "researcher",
    agents: [agent("a1", 18, { spawn_shield_remaining: 3 }), agent("a2", 10)],
    selection: {},
  };
  const rows = champions.records({
    scene,
    transform,
    showSpawnShield: true,
    visualFilters: { resurrection_effects: setDown },
    beats: {
      rows: [],
      view: { life: [], returns: [{ publicAgentId: "a1", seen: true }] },
    },
  });
  const doc = /** @type {any} */ (new FakeDocument());
  const nodes = champions.build(rows, doc);
  champions.update(nodes, rows);
  return { rows, nodes };
}

test("a returning champion's Return hold sits on its glide wrapper", () => {
  const { rows, nodes } = drawnChampions(true);
  assert.equal(rows[0].state, "returning");
  assert.equal(rows[1].state, "living");
  const wrapper = /** @type {any} */ (nodes[0].wrapper);
  assert.equal(wrapper.getAttribute("data-beat-hold"), "return");
  assert.equal(wrapper.getAttribute("data-beat-key"), "pk-a1");
  assert.equal(wrapper.getAttribute("data-beat-side"), "after");
  assert.equal(wrapper.getAttribute("data-beat-from"), "0");
  assert.equal(
    wrapper.getAttribute("data-beat-fade"),
    String(champions.BEAT_LENGTHS.setDown),
  );
  // The token and its ward ride the wrapper; neither keeps a hold of its own.
  const root = /** @type {any} */ (nodes[0].root);
  assert.equal(root.getAttribute("data-beat-hold"), null);
  assert.equal(root.getAttribute("opacity"), "0.8");
  assert.ok(nodes[0].shieldRoot, "the returning champion is warded");
  assert.equal(
    /** @type {any} */ (nodes[0].shieldRoot).getAttribute("data-beat-hold"),
    null,
  );
  // A champion that does not return gets no hold.
  const other = /** @type {any} */ (nodes[1].wrapper);
  assert.equal(other.getAttribute("data-beat-hold"), null);
  assert.equal(other.getAttribute("data-beat-from"), null);
  // Resurrection Effects off: the same hold is a cut.
  const cut = /** @type {any} */ (drawnChampions(false).nodes[0].wrapper);
  assert.equal(cut.getAttribute("data-beat-fade"), "1");
});

test("every hotbar mark waits for its own champion's set-down", () => {
  const you = agent("a1", 8, { presentation_key: "you" });
  const mate = agent("a2", 12, { presentation_key: "mate" });
  const foe = agent("b1", 10, { presentation_key: "foe", team_id: 2 });
  const scene = {
    agents: [you, mate, foe],
    selection: { controlled_presentation_key: "you" },
    pending_route: {
      source_presentation_key: "you",
      target_presentation_key: "foe",
      source_anchor: you.position,
      target_anchor: foe.position,
      source_radius: 0.5,
      target_radius: 0.5,
    },
  };
  const inspection = {
    inspection_kind: "live_draft_action",
    actor_presentation_key: "you",
    draft_action: { armed_lane: "basic", move_action: 1 },
    decision_mask: {
      owner_presentation_key: "you",
      target_actions: [
        {
          target_action: 1,
          target_kind: "visible_authorized_agent",
          target_presentation_key: "foe",
          target_public_agent_id: "b1",
        },
      ],
      target_use_ultimate_joint_mask: [
        [true, false],
        [true, false],
      ],
    },
  };
  const records = hotbar.records({
    scene,
    transform,
    inspection,
    manualSlots: null,
    pendingRows: [
      {
        actor_presentation_key: "mate",
        actor_public_agent_id: "a2",
        pending_action: { target_action: 0, use_ultimate_action: 0 },
      },
    ],
  });
  const kinds = new Set(records.map((record) => record.kind));
  for (const kind of [
    "command-ring",
    "order-studs",
    "reach-mark",
    "order-thread",
    "step-ghost",
  ]) {
    assert.ok(kinds.has(kind), kind);
  }
  const owners = new Map([
    ["command-ring", (/** @type {any} */ r) => r.identityKey],
    ["look-ring", (/** @type {any} */ r) => r.identityKey],
    ["reach-mark", (/** @type {any} */ r) => r.identityKey],
  ]);
  for (const record of records) {
    const expected = owners.get(record.kind)?.(record);
    if (expected !== undefined) assert.equal(record.returnKey, expected, record.kind);
    else if (record.kind !== "order-studs")
      assert.equal(record.returnKey, "you", record.kind);
    else assert.ok(record.returnKey === "you" || record.returnKey === "mate");
  }
  assert.ok(records.some((r) => r.kind === "reach-mark" && r.returnKey === "foe"));
  assert.ok(records.some((r) => r.kind === "command-ring" && r.returnKey === "mate"));
  assert.ok(records.some((r) => r.kind === "order-studs" && r.returnKey === "mate"));
  const doc = /** @type {any} */ (new FakeDocument());
  const nodes = /** @type {any[]} */ (hotbar.build(records, doc));
  assert.equal(nodes.length, records.length);
  for (const [index, node] of nodes.entries()) {
    assert.equal(node.getAttribute("data-beat-hold"), "set-down");
    assert.equal(node.getAttribute("data-beat-key"), records[index].returnKey);
    assert.equal(node.getAttribute("data-beat-side"), "after");
    assert.equal(node.getAttribute("data-beat-fade"), "1");
    // A cut with no opacity attribute: the mark's own CSS opacity stays its END.
    assert.equal(node.getAttribute("opacity"), null);
  }
});
