/**
 * @file Check that the Champion Frame look draws no lane pills.
 *
 * Contract: the old Basic and Ultimate lane pills ("0/B", "1/U") are gone from
 * the arena in every client, the Replay Viewer included. records() with part
 * "lane-pills" gives no records even with full legality, owner and placement,
 * build() makes no node from an old "lane-pills" record, and the look's node
 * budget no longer counts the pills. Legality lives on the deck's Action slots
 * and the Command Ring instead.
 */
import assert from "node:assert/strict";
import test from "node:test";

import * as championFrame from "../src/looks/champion-frame.js";

test("lane-pills gives no records and builds no nodes", () => {
  const legality = {
    owner_presentation_key: "agent-a",
    owner_public_agent_id: "A1",
    basic_available: true,
    ultimate_available: true,
    armed_lane: 0,
    armed_pair_legal: true,
  };
  const owner = { agent: { public_agent_id: "A1" }, presentationKey: "agent-a" };
  const placement = {
    bounds: { left: 10, top: 10, right: 73, bottom: 28 },
    anchor: "north",
    tangentShift: 0,
    collisionFree: true,
  };
  assert.deepEqual(
    championFrame.records({ part: "lane-pills", legality, owner, placement }),
    [],
  );
  assert.deepEqual(
    championFrame.build(
      [{ kind: "lane-pills", legality, ownerAgent: owner.agent, placement }],
      /** @type {any} */ (null),
    ),
    [],
  );
  assert.equal(championFrame.BUDGET.nodesPerRecord, 5);
});
