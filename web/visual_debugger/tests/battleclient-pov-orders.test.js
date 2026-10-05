/**
 * @file Check which staged Manual orders the BattleClient deck and rail may
 * show (src/tactics-turn.js), on the real presentation fixtures:
 * - shownManualIdsOf: in Agent POV only the Manual champions on the POV
 *   recipient's team (ruling L18: another team's staged order is never shown
 *   in Agent POV), in roster order; an empty list when the POV scene does not
 *   draw the recipient; Oracle View keeps every Manual champion. It reads only
 *   the recipient, the POV scene and the identity rows, so changing the
 *   researcher branch's staged orders changes nothing.
 * - servedManualOrdersOf: in Oracle View, each Manual champion's staged order
 *   from the frame's own pending_joint_action rows (move, lane, and the target
 *   decoded like core/axis_mappings.py: 1 to 5 the actor's own team, 6 to 10
 *   the other team, 0 none), so a fresh page and a page that switched views
 *   agree; in Agent POV the viewing team's own staged orders from the
 *   researcher branch (ruling P1, 3 Oct 2026: your own orders always show),
 *   never another team's even when its IDs are passed in, and nothing
 *   without the shown IDs.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  authorizedPresentationIdentityRows,
  authorizedPresentationSceneView,
} from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { servedManualOrdersOf, shownManualIdsOf } from "../src/tactics-turn.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @param {string} name @param {(raw: any) => void} [edit] */
async function frameOf(name, edit = () => undefined) {
  const raw = structuredClone(fixture.presentations[name]);
  edit(raw);
  return /** @type {Record<string, any>} */ (
    await normalizeAuthorizedPresentationFrameV1(raw)
  );
}

/** @param {Record<string, any>} frame */
function partsOf(frame) {
  return {
    sceneAgents: /** @type {any[]} */ (
      authorizedPresentationSceneView(frame)?.agents ?? []
    ),
    rosterRows: /** @type {any[]} */ ([...authorizedPresentationIdentityRows(frame)]),
  };
}

const ALL_MANUAL = [
  "agent-slot-0",
  "agent-slot-1",
  "agent-slot-2",
  "agent-slot-5",
  "agent-slot-6",
];

test("Agent POV shows only the recipient team's Manual champions", async () => {
  const pov = await frameOf("live_no_shared_obs_agent_pov");
  assert.equal(pov.authority.recipient_public_agent_id, "agent-slot-0");
  const { sceneAgents, rosterRows } = partsOf(pov);
  assert.deepEqual(shownManualIdsOf(pov, sceneAgents, rosterRows, ALL_MANUAL), [
    "agent-slot-0",
    "agent-slot-1",
    "agent-slot-2",
  ]);
  // Only another team's Manual champions: nothing is shown.
  assert.deepEqual(
    shownManualIdsOf(pov, sceneAgents, rosterRows, ["agent-slot-5", "agent-slot-6"]),
    [],
  );
  // A scene without the recipient shows nothing rather than guessing a team.
  const withoutRecipient = sceneAgents.filter(
    (agent) => agent.public_agent_id !== "agent-slot-0",
  );
  assert.deepEqual(shownManualIdsOf(pov, withoutRecipient, rosterRows, ALL_MANUAL), []);
  // The researcher branch's staged orders play no part.
  const scrambled = await frameOf("live_no_shared_obs_agent_pov", (raw) => {
    for (const row of raw.researcher_space.pending_joint_action.action_rows) {
      if (row.actor_public_agent_id === "agent-slot-0") continue;
      row.pending_action = { move_action: 3, target_action: 4, use_ultimate_action: 1 };
    }
  });
  const changed = partsOf(scrambled);
  assert.deepEqual(
    shownManualIdsOf(scrambled, changed.sceneAgents, changed.rosterRows, ALL_MANUAL),
    shownManualIdsOf(pov, sceneAgents, rosterRows, ALL_MANUAL),
  );
});

test("Oracle View shows every Manual champion", async () => {
  const oracle = await frameOf("live_oracle");
  const { sceneAgents, rosterRows } = partsOf(oracle);
  assert.deepEqual(
    shownManualIdsOf(oracle, sceneAgents, rosterRows, ALL_MANUAL),
    ALL_MANUAL,
  );
  assert.deepEqual(shownManualIdsOf(null, [], [], ALL_MANUAL), ALL_MANUAL);
});

test("Oracle View reads each Manual champion's staged order from its own pending rows", async () => {
  const oracle = await frameOf("live_oracle", (raw) => {
    const byId = new Map(
      raw.pending_joint_action.action_rows.map((/** @type {any} */ row) => [
        row.actor_public_agent_id,
        row,
      ]),
    );
    // Team A actor, other team's first row: Basic on agent-slot-5.
    byId.get("agent-slot-1").pending_action = {
      move_action: 3,
      target_action: 6,
      use_ultimate_action: 0,
    };
    // Team B actor, own team's second row: Ultimate on agent-slot-6.
    byId.get("agent-slot-5").pending_action = {
      move_action: 1,
      target_action: 2,
      use_ultimate_action: 1,
    };
    // Team B actor, other team's third row: Basic on agent-slot-2.
    byId.get("agent-slot-6").pending_action = {
      move_action: 0,
      target_action: 8,
      use_ultimate_action: 0,
    };
  });
  const orders = servedManualOrdersOf(oracle, partsOf(oracle).rosterRows);
  assert.deepEqual(orders.get("agent-slot-1"), {
    moveAction: 3,
    lane: "basic",
    targetPublicId: "agent-slot-5",
  });
  assert.deepEqual(orders.get("agent-slot-5"), {
    moveAction: 1,
    lane: "ultimate",
    targetPublicId: "agent-slot-6",
  });
  assert.deepEqual(orders.get("agent-slot-6"), {
    moveAction: 0,
    lane: "basic",
    targetPublicId: "agent-slot-2",
  });
  assert.deepEqual(orders.get("agent-slot-2"), {
    moveAction: 0,
    lane: "none",
    targetPublicId: null,
  });
  assert.equal(orders.size, 5);
});

test("Agent POV serves only the viewing team's own staged orders", async () => {
  const pov = await frameOf("live_no_shared_obs_agent_pov", (raw) => {
    for (const row of raw.researcher_space.pending_joint_action.action_rows) {
      // Team A actor 1: Basic on the other team's first row (agent-slot-5).
      if (row.actor_public_agent_id === "agent-slot-1") {
        row.pending_action = {
          move_action: 3,
          target_action: 6,
          use_ultimate_action: 0,
        };
      }
      // Team B actor 5: an order Agent POV of Team A must never show.
      if (row.actor_public_agent_id === "agent-slot-5") {
        row.pending_action = {
          move_action: 1,
          target_action: 2,
          use_ultimate_action: 1,
        };
      }
    }
  });
  const { sceneAgents, rosterRows } = partsOf(pov);
  const shown = shownManualIdsOf(pov, sceneAgents, rosterRows, ALL_MANUAL);
  const orders = servedManualOrdersOf(pov, rosterRows, shown);
  assert.deepEqual(
    [...orders.keys()],
    ["agent-slot-0", "agent-slot-1", "agent-slot-2"],
  );
  assert.deepEqual(orders.get("agent-slot-1"), {
    moveAction: 3,
    lane: "basic",
    targetPublicId: "agent-slot-5",
  });
  assert.deepEqual(orders.get("agent-slot-0"), {
    moveAction: 0,
    lane: "none",
    targetPublicId: null,
  });
  // Another team's IDs passed in are still never served in Agent POV.
  const all = servedManualOrdersOf(pov, rosterRows, ALL_MANUAL);
  assert.equal(all.has("agent-slot-5"), false);
  assert.equal(all.has("agent-slot-6"), false);
  assert.equal(all.size, 3);
  // Without the shown IDs Agent POV serves nothing.
  assert.equal(servedManualOrdersOf(pov, rosterRows).size, 0);
  assert.equal(servedManualOrdersOf(pov, rosterRows, []).size, 0);
  // The shared-sight Agent POV fixture serves its recipient's team only.
  const shared = await frameOf("live_shared_obs_agent_pov");
  const sharedParts = partsOf(shared);
  const ids = sharedParts.rosterRows.map((row) => row.public_agent_id);
  assert.deepEqual(
    [...servedManualOrdersOf(shared, sharedParts.rosterRows, ids).keys()],
    ["0", "1", "2", "3", "4"],
  );
  assert.equal(servedManualOrdersOf(null, []).size, 0);
});
