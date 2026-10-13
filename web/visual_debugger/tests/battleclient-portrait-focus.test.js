/**
 * @file Check who a click on a live scorebug portrait reaches in Agent POV
 * (liveScorebugModel, src/looks/hud.js), in One on Each Team (a Manual
 * champion on each team): an unseen enemy that is not yours keeps its
 * refusal sentence ("The Warrior (ID 6) is out of your team's sight. Agent
 * POV hides where it is and how it fares."), while your own Manual champion
 * on the other team, unseen by this view, has no refusal (a click selects it
 * and the host then shows its view). It stays identity only: label "Unseen",
 * no life and no hold.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { liveControllers, liveScorebugModel, records } from "../src/looks/hud.js";

const CLASSES = [1, 2, 3, 4, 5];

/** @param {number} id */
function agent(id) {
  return {
    public_agent_id: String(id),
    display_agent_id: String(id),
    presentation_key: `k${id}`,
    team_id: id < 5 ? 1 : 2,
    class_id: CLASSES[id % 5],
    alive: true,
    life_state: "alive",
    current_health: 100,
    maximum_health: 100,
    max_health: 100,
    ultimate_cooldown_remaining: 0,
    statuses: [],
    spawn_shield_remaining: 0,
    position: [id, id],
    radius: 0.5,
  };
}

const identities = Array.from({ length: 10 }, (_, id) =>
  Object.freeze({
    publicAgentId: String(id),
    teamId: /** @type {1 | 2} */ (id < 5 ? 1 : 2),
    teamLocalSlot: id % 5,
    classId: CLASSES[id % 5],
  }),
);

const EACH_TEAM = liveControllers({
  manual_slots: [1, 5],
  slot_labels: [
    "Recurrent MAPPO",
    "Manual",
    "Recurrent MAPPO",
    "Recurrent MAPPO",
    "Recurrent MAPPO",
    "Manual",
    "tdm-beta",
    "tdm-beta",
    "tdm-beta",
    "tdm-beta",
  ],
});

test("your unseen Manual champion on the other team can be clicked; an unseen enemy cannot", () => {
  const input = {
    product: /** @type {const} */ ("combat_debugger"),
    audience: /** @type {const} */ ("agent_pov"),
    sceneAgents: Array.from({ length: 5 }, (_, id) => agent(id)),
    researcherAgents: null,
    identities,
    mechanics: [],
    waves: [],
    visualEvents: [],
    teams: [],
    deaths: [],
    controlledKey: "k1",
    recipientPublicAgentId: "1",
    returnedFallback: [],
    displayIdOf: (/** @type {string} */ id) => id,
  };
  const result = liveScorebugModel(
    records({ sources: input, filters: {} }),
    input,
    EACH_TEAM,
  );
  const portraits = result.teams.flatMap((team) => team.portraits);
  const yours = portraits.find((entry) => entry.publicAgentId === "5");
  const enemy = portraits.find((entry) => entry.publicAgentId === "6");
  assert.ok(yours && enemy);
  assert.equal(yours.state, "unseen");
  assert.equal(yours.label, "Unseen");
  assert.equal(yours.life, null);
  assert.equal(yours.hold, null);
  assert.equal(yours.refusal, null);
  assert.equal(enemy.state, "unseen");
  assert.equal(
    enemy.refusal,
    "The Warrior (ID 6) is out of your team's sight. Agent POV hides where it is and how it fares.",
  );
});
