/**
 * @file Check that a returning champion's Ranges rings wait for its set-down
 * on the pad (src/looks/champion-frame.js build): every reach-ring node of a
 * champion with a presentation key carries the "set-down" beat hold keyed to
 * that champion (data-beat-hold "set-down", data-beat-key, side "after", a
 * 1 ms cut), which the choreographer applies only on a tick that champion
 * returns, so no Ranges ring shows at its waystone before the Return beat; a
 * ring with no presentation key gets no hold.
 */
import assert from "node:assert/strict";
import test from "node:test";

import * as frame from "../src/looks/champion-frame.js";

/** @returns {any} */
function fakeDocument() {
  /** @param {string} tag @returns {any} */
  const node = (tag) => {
    /** @type {Map<string, string>} */
    const attributes = new Map();
    /** @type {any} */
    const element = {
      tagName: tag,
      children: [],
      parentElement: null,
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
      dataset: {},
      style: {},
      classList: { add() {}, remove() {}, toggle() {} },
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      hasAttribute: (/** @type {string} */ name) => attributes.has(name),
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      addEventListener() {},
      append: (/** @type {any[]} */ ...nodes) => element.children.push(...nodes),
    };
    return element;
  };
  return {
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      node(tag),
    createElement: (/** @type {string} */ tag) => node(tag),
  };
}

/** @type {any} */
const transform = {
  worldToScreen: (/** @type {any} */ point) => ({ x: point[0] * 20, y: point[1] * 20 }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 20,
};

test("a champion's Ranges rings carry its set-down hold", () => {
  const scene = {
    agents: [
      {
        presentation_key: "w",
        public_agent_id: "agent-slot-1",
        class_id: 2,
        team_id: 1,
      },
    ],
    selection: { selected_presentation_key: "w" },
    ranges: [
      {
        presentation_key: "w",
        public_agent_id: "agent-slot-1",
        kind: "basic",
        center: [18, 5],
        radius: 1.5,
      },
      {
        presentation_key: "w",
        public_agent_id: "agent-slot-1",
        kind: "ultimate",
        center: [18, 5],
        radius: 5.75,
      },
    ],
  };
  const rings = frame.records({ part: "reach-rings", scene, transform });
  assert.equal(rings.length, 2);
  const nodes = frame.build(rings, fakeDocument());
  assert.ok(nodes.length >= 2);
  for (const node of nodes) {
    assert.equal(node.getAttribute("data-beat-hold"), "set-down");
    assert.equal(node.getAttribute("data-beat-key"), "w");
    assert.equal(node.getAttribute("data-beat-side"), "after");
    assert.equal(node.getAttribute("data-beat-fade"), "1");
  }
  const keyless = rings.map((ring) => ({
    ...ring,
    identity: { ...ring.identity, presentation_key: undefined },
  }));
  for (const node of frame.build(keyless, fakeDocument())) {
    assert.equal(node.getAttribute("data-beat-hold"), null);
  }
});
