/**
 * @file Check agent-identity.js as the one owner of label forms: the champion
 * label "Mage · ID 5" with no team word, the chip text "Agent 5", the source
 * name "Hunter ID 7", the Out of Sight chip "Warrior · ID 6 · Out of Sight",
 * the frozen class and team tables (Team A and Team B only), integer-only class
 * and team lookups, and tick counts "1 Tick"/"2 Ticks" for labels and
 * "1 tick"/"2 ticks" inside sentences.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  CLASS_IDENTITIES,
  canonicalAgentIdentity,
  classIdentityFromId,
  exactAuthorizedAgentIdentityV1,
  TEAM_IDENTITIES,
  teamIdentityFromId,
  teamLabelFromId,
  tickCountLabel,
  tickCountText,
} from "../src/agent-identity.js";

test("labels follow the owner's forms and never name a team", () => {
  const identity = exactAuthorizedAgentIdentityV1({
    presentation_key: "scene:warrior",
    public_agent_id: "agent-slot-6",
    display_agent_id: "6",
    class_id: 2,
    team_id: 2,
  });
  assert.deepEqual(identity, {
    presentationKey: "scene:warrior",
    publicAgentId: "agent-slot-6",
    classId: 2,
    teamId: 2,
    title: "Warrior · ID 6",
    publicIdentity: "Agent ID 6",
    chipText: "Agent 6",
    sourceName: "Warrior ID 6",
    outOfSightLabel: "Warrior · ID 6 · Out of Sight",
    classLabel: "Warrior",
    teamLabel: "Team B",
    accent: "warrior",
  });
  assert.equal(Object.isFrozen(identity), true);
  for (const field of ["title", "chipText", "sourceName", "outOfSightLabel"]) {
    assert.doesNotMatch(
      String(identity?.[/** @type {keyof typeof identity} */ (field)]),
      /Team|Azure|Crimson|Host|Blue|Red/u,
    );
  }
  // A longer public ID is shown as given when no one-digit display ID exists.
  assert.equal(
    canonicalAgentIdentity({ public_agent_id: "seat/12", class_id: 5, team_id: 1 })
      .title,
    "Priest · ID seat/12",
  );
  // A display ID must be one digit; anything else falls back to the public ID.
  assert.equal(
    canonicalAgentIdentity({
      public_agent_id: "agent-slot-3",
      display_agent_id: "13",
      class_id: 4,
      team_id: 1,
    }).chipText,
    "Agent agent-slot-3",
  );
});

test("class and team tables are frozen and only integer IDs find a row", () => {
  assert.deepEqual(
    Object.values(CLASS_IDENTITIES).map(({ classId, tokenId, label }) => [
      classId,
      tokenId,
      label,
    ]),
    [
      [1, "mage", "Mage"],
      [2, "warrior", "Warrior"],
      [3, "hunter", "Hunter"],
      [4, "rogue", "Rogue"],
      [5, "priest", "Priest"],
    ],
  );
  assert.deepEqual(
    Object.values(TEAM_IDENTITIES).map(({ teamId, tokenId, label, letter }) => [
      teamId,
      tokenId,
      label,
      letter,
    ]),
    [
      [1, "team_a", "Team A", "A"],
      [2, "team_b", "Team B", "B"],
    ],
  );
  assert.equal(Object.isFrozen(CLASS_IDENTITIES), true);
  assert.equal(Object.isFrozen(TEAM_IDENTITIES), true);
  for (const row of [
    ...Object.values(CLASS_IDENTITIES),
    ...Object.values(TEAM_IDENTITIES),
  ]) {
    assert.equal(Object.isFrozen(row), true);
  }
  assert.equal(classIdentityFromId(3), CLASS_IDENTITIES[3]);
  assert.equal(teamIdentityFromId(1), TEAM_IDENTITIES[1]);
  assert.equal(teamLabelFromId(2), "Team B");
  for (const invalid of [0, 6, "1", 1.5, null, undefined, "constructor", {}]) {
    assert.equal(classIdentityFromId(invalid), null);
  }
  for (const invalid of [0, 3, "1", 1.5, null, undefined, "constructor"]) {
    assert.equal(teamIdentityFromId(invalid), null);
    assert.equal(teamLabelFromId(invalid), null);
  }
});

test("tick counts use the singular only for exactly one", () => {
  assert.deepEqual(
    [0, 1, 2, 30, -1].map((count) => [tickCountLabel(count), tickCountText(count)]),
    [
      ["0 Ticks", "0 ticks"],
      ["1 Tick", "1 tick"],
      ["2 Ticks", "2 ticks"],
      ["30 Ticks", "30 ticks"],
      ["-1 Ticks", "-1 ticks"],
    ],
  );
  for (const invalid of [1.5, "1", null, undefined, Number.NaN]) {
    assert.equal(tickCountLabel(invalid), null);
    assert.equal(tickCountText(invalid), null);
  }
});
