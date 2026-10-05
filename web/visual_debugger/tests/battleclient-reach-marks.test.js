/**
 * @file Check the Reach Mark size rule of the Champion's Hotbar
 * (src/looks/hotbar.js), as the BattleClient vision page draws it
 * (`isManual(b) ? BAND + 5 : BAND`): a Reach Mark around any Manual champion,
 * yours or another team's, is 5 vision pixels (scaled by body radius / 25)
 * wider than the Command Ring band, so it clears that champion's ring; a
 * Reach Mark around a policy's champion sits on the band.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as hotbar from "../src/looks/hotbar.js";

/**
 * @param {string} key
 * @param {number} x
 * @param {string} slot
 * @param {number} team
 */
function agent(key, x, slot, team) {
  return {
    presentation_key: key,
    public_agent_id: `${key}-id`,
    position: [x, 5],
    radius: 0.5,
    alive: true,
    life_state: "alive",
    team_id: team,
    class_id: 1,
    display_agent_id: slot,
  };
}

/** @param {string} key @param {number} action */
function target(key, action) {
  return {
    target_action: action,
    target_kind: "visible_authorized_agent",
    target_presentation_key: key,
    target_public_agent_id: `${key}-id`,
  };
}

test("every Manual champion's Reach Mark clears its Command Ring", () => {
  const scene = {
    agents: [
      agent("you", 4, "1", 0),
      agent("mage", 8, "5", 1),
      agent("bot", 12, "6", 1),
    ],
    selection: { controlled_presentation_key: "you" },
  };
  const transform = createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: 1000,
    viewportHeight: 500,
    padding: 0,
  });
  const inspection = {
    actor_presentation_key: "you",
    inspection_kind: "live_draft_action",
    draft_action: { armed_lane: "basic" },
    decision_mask: {
      owner_presentation_key: "you",
      target_actions: [target("mage", 1), target("bot", 2)],
      target_use_ultimate_joint_mask: [
        [true, false],
        [true, false],
        [true, false],
      ],
    },
  };
  const records = hotbar.records({ scene, transform, inspection, manualSlots: [1, 5] });
  const band = records.find(
    (r) => r.kind === "command-ring" && r.identityKey === "you",
  );
  const marks = new Map(
    records.filter((r) => r.kind === "reach-mark").map((r) => [r.identityKey, r.r]),
  );
  assert.ok(band);
  const radius = 0.5 * 50;
  const extra = (radius / 25) * hotbar.FOOTPRINT.ownReachExtra;
  assert.ok(Math.abs(Number(marks.get("mage")) - (band.r + extra)) < 0.02);
  assert.ok(Math.abs(Number(marks.get("bot")) - band.r) < 0.02);
  // Without the Mage in the Manual slots its mark sits on the band.
  const policy = hotbar.records({ scene, transform, inspection, manualSlots: [1] });
  const mage = policy.find((r) => r.kind === "reach-mark" && r.identityKey === "mage");
  assert.ok(Math.abs(Number(mage?.r) - band.r) < 0.02);
});
