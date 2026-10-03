/**
 * @file Check that the Focus Tally's node budget (src/looks/focus-tally.js
 * BUDGET) covers what it really draws, counted the way the choreography
 * painter counts a kept row (choreography-painter.js: the event group, every
 * node inside it, and the connector only when it holds something). With the
 * thread kept and five beads, one tally keeps 5 nodes, and the plan's
 * persistent bound for one focus_tally row (planResourceDemand) is at least
 * that, so turning Focus Fire on never makes the painter refuse the tick
 * ("exceeded the planned persistent node bound").
 */
import assert from "node:assert/strict";
import test from "node:test";

import { planEventBudget, planResourceDemand } from "../src/choreography-plan.js";
import * as tally from "../src/looks/focus-tally.js";

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
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      append: (/** @type {any[]} */ ...nodes) => {
        element.children.push(...nodes);
      },
      get childElementCount() {
        return element.children.length;
      },
    };
    return element;
  };
  return {
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      node(tag),
  };
}

/** @param {any} element @returns {number} */
function descendants(element) {
  return element.children.reduce(
    (/** @type {number} */ sum, /** @type {any} */ child) =>
      sum + 1 + descendants(child),
    0,
  );
}

test("one drawn Focus Tally fits its budget and the plan's persistent bound", () => {
  const doc = fakeDocument();
  const records = tally.records({
    event: {
      eventId: "t:focus:agent-slot-5",
      kind: "focus_tally",
      targetPublicAgentId: "agent-slot-5",
      sourcePublicAgentIds: ["0", "1", "2", "3", "4"],
      attackingTeamId: 1,
      count: 5,
      lethal: true,
      anchor: { x: 244.1, y: 177.8 },
      anchorRadius: 23,
      paintParts: { effect: true },
      persistent: true,
    },
  });
  assert.equal(records.length, 1);
  assert.notEqual(records[0].threadPath, null);
  const mount = {
    group: doc.createElementNS("", "g"),
    connector: doc.createElementNS("", "g"),
    underlay: null,
    plan: { epochKey: "e", phases: {} },
    options: { motionMode: "off", settled: true },
    animationSpecs: [],
  };
  tally.build(records, doc, /** @type {any} */ (mount));
  // The painter's count: the event group plus its nodes; an empty connector
  // is dropped (renderedConnector is null), so it adds nothing.
  const kept =
    1 +
    descendants(mount.group) +
    (mount.connector.childElementCount > 0 ? 1 + descendants(mount.connector) : 0);
  assert.equal(kept, 5);
  assert.equal(tally.BUDGET.persistentNodesPerRecord, kept);
  assert.ok(tally.BUDGET.nodesPerRecord >= kept);
  assert.equal(planEventBudget({ kind: "focus_tally" }), tally.BUDGET);
  const demand = planResourceDemand(
    [],
    [{ kind: "focus_tally", spatial: true, persistent: true }],
    [],
    /** @type {any} */ ({ falls: [], returns: [] }),
  );
  assert.ok(demand.persistentNodes >= kept);
});
