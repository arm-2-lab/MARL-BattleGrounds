/**
 * @file Check two HUD record contracts in looks/hud.js records().
 *
 * Contracts checked:
 * - ID order: each team lists its champions in shown-ID order in the Replay
 *   Viewer's Agent POV (seen, unseen and fallen mixed, so a frame never moves
 *   when sight changes; the old seen_grouping switch is gone and a stale
 *   value of it changes nothing), in Oracle View and in the BattleClient.
 * - Full Record chip: the words name the POV agent with the same chip text as
 *   its own strip frame ("Agent 0", never the raw public ID such as
 *   "agent-slot-0"), even when displayIdOf cannot name it, and read
 *   "This Agent" when the POV agent is unknown.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { records } from "../src/looks/hud.js";
import { hudSourcesFromFrame } from "../src/match-summary.js";

/** @param {Record<string, any>} overrides */
function agent(overrides) {
  return {
    public_agent_id: "agent-slot-0",
    presentation_key: "key-0",
    team_id: 1,
    class_id: 3,
    display_agent_id: "0",
    life_state: "alive",
    alive: true,
    current_health: 100,
    maximum_health: 100,
    ultimate_cooldown_remaining: 5,
    spawn_shield_remaining: 0,
    statuses: [],
    ...overrides,
  };
}

/** @param {number} slot @param {Record<string, any>} [overrides] */
const slotAgent = (slot, overrides = {}) =>
  agent({
    public_agent_id: `agent-slot-${slot}`,
    presentation_key: `key-${slot}`,
    display_agent_id: String(slot),
    team_id: slot < 5 ? 1 : 2,
    ...overrides,
  });

const fallen = { alive: false, life_state: "dead", current_health: 0 };
// Team A: 0 seen, 1 unseen, 2 seen but fallen, 3 seen. Team B: 5 unseen, 6 seen.
const everyone = [
  slotAgent(0),
  slotAgent(1),
  slotAgent(2, fallen),
  slotAgent(3),
  slotAgent(5),
  slotAgent(6),
];
const seenBodies = [slotAgent(0), slotAgent(2, fallen), slotAgent(3), slotAgent(6)];

/**
 * @param {Partial<import("../src/looks/hud.js").HudSources>} overrides
 * @returns {import("../src/looks/hud.js").HudSources}
 */
function sources(overrides) {
  return {
    product: "replay_viewer",
    audience: "agent_pov",
    sceneAgents: seenBodies,
    researcherAgents: everyone,
    identities: [],
    mechanics: [],
    waves: [],
    visualEvents: [],
    teams: [{ display_name: "learned-a" }, { display_name: "learned-b" }],
    deaths: [],
    controlledKey: null,
    recipientPublicAgentId: "agent-slot-0",
    returnedFallback: [],
    displayIdOf: () => null,
    ...overrides,
  };
}

/** @param {ReadonlyArray<Record<string, any>>} list */
const shownOrder = (list) =>
  list.filter((record) => record.kind === "champion").map((record) => record.displayId);

/** @param {ReadonlyArray<Record<string, any>>} list */
const label = (list) =>
  list.find((record) => record.kind === "hud_view")?.fullRecordLabel;

test("Agent POV, Oracle View and the BattleClient all keep shown-ID order", () => {
  const idOrder = ["0", "1", "2", "3", "5", "6"];
  assert.deepEqual(shownOrder(records({ sources: sources({}) })), idOrder);
  assert.deepEqual(
    shownOrder(records({ sources: sources({}), filters: { seen_grouping: true } })),
    idOrder,
  );
  assert.deepEqual(
    shownOrder(
      records({
        sources: sources({
          audience: "researcher",
          sceneAgents: everyone,
          researcherAgents: null,
          recipientPublicAgentId: null,
        }),
      }),
    ),
    idOrder,
  );
  const identities = everyone.map((body) => ({
    publicAgentId: body.public_agent_id,
    teamId: /** @type {1 | 2} */ (body.team_id),
    teamLocalSlot: Number(body.display_agent_id) % 5,
    classId: body.class_id,
  }));
  assert.deepEqual(
    shownOrder(
      records({
        sources: sources({
          product: "combat_debugger",
          researcherAgents: null,
          identities,
          displayIdOf: (id) => id.replace("agent-slot-", ""),
        }),
      }),
    ),
    idOrder,
  );
});

test("the Full Record chip names the POV agent like its strip frame", () => {
  const list = records({ sources: sources({}) });
  assert.equal(label(list), "Full Record · Not Limited To Agent 0's View");
  const own = list.find(
    (record) => record.kind === "champion" && record.publicAgentId === "agent-slot-0",
  );
  assert.equal(label(list), `Full Record · Not Limited To ${own?.chipText}'s View`);
  assert.doesNotMatch(String(label(list)), /agent-slot/u);
  assert.equal(
    label(records({ sources: sources({ recipientPublicAgentId: null }) })),
    "Full Record · Not Limited To This Agent's View",
  );
  assert.equal(
    label(
      records({ sources: sources({ researcherAgents: null, audience: "researcher" }) }),
    ),
    null,
  );
});

test("on a recorded Replay Viewer Agent POV frame the chip matches the POV agent's frame", async () => {
  const fixture = JSON.parse(
    readFileSync(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  const frame = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(fixture.presentations.replay_no_shared_obs_agent_pov),
  );
  const gathered = hudSourcesFromFrame(frame);
  assert.ok(gathered);
  const list = records({ sources: gathered });
  const own = list.find(
    (record) =>
      record.kind === "champion" &&
      record.publicAgentId === gathered?.recipientPublicAgentId,
  );
  assert.ok(own);
  assert.equal(label(list), `Full Record · Not Limited To ${own.chipText}'s View`);
  assert.match(String(label(list)), /^Full Record · Not Limited To Agent \d's View$/u);
});
