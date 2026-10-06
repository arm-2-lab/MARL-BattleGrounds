/**
 * @file Check the class card and ability titles built from real host numbers in
 * the authorized presentation fixture: every certified class card lists its
 * named Basic ("Basic Name", "Basic Description") before the Ultimate; the
 * Hunter's Serrated Arrows carry the slow that used to be its passive, and its
 * passive is Far Sight with the host's observation radius; activation titles
 * read "Arcane Blast (Mage Basic)" and "Burst (Mage Ultimate)", and without
 * class mechanics a Basic's summary is the plain "The Priest used Lesser Heal
 * this tick.".
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { explainActivation, explainClassDocumentation } from "../src/explanations.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);
const scene = fixture.presentations.replay_oracle.current_endpoint.scene;

/** @param {number} classId */
function classRows(classId) {
  const owner = scene.agents.find(
    (/** @type {Record<string, any>} */ agent) => agent.class_id === classId,
  );
  const mechanics = scene.class_mechanics.find(
    (/** @type {Record<string, any>} */ row) => row.class_id === classId,
  );
  const descriptor = explainClassDocumentation(owner, mechanics);
  assert.ok(descriptor);
  const section = descriptor.sections.find(
    (/** @type {Record<string, any>} */ candidate) =>
      candidate.title === "Class Mechanics",
  );
  assert.ok(section);
  return {
    owner,
    mechanics,
    rows: new Map(section.rows.map((row) => [row.label, row.value])),
  };
}

test("each class card names its Basic with host numbers", () => {
  assert.deepEqual(
    [1, 2, 3, 4, 5].map((classId) => {
      const { rows } = classRows(classId);
      return [rows.get("Basic Name"), rows.get("Basic Description")];
    }),
    [
      [
        "Arcane Blast",
        "Arcane Blast hits one enemy in reach with a bolt of arcane energy. It is the strongest Basic in the game. Damage bonuses and reductions affect the damage dealt.",
      ],
      [
        "Shield Bash",
        "Shield Bash hits one enemy close to the Warrior. Damage bonuses and reductions affect the damage dealt.",
      ],
      [
        "Serrated Arrows",
        "Serrated Arrows hit one enemy in reach. Each hit also slows that enemy, causing a 15% movement reduction (×0.85) for 1 tick. A new hit starts the slow again.",
      ],
      [
        "Backstab",
        "Backstab hits one enemy close to the Rogue. Damage bonuses and reductions affect the damage dealt.",
      ],
      [
        "Lesser Heal",
        "Lesser Heal heals one ally in reach, or the Priest itself, and gives it Blessing of Freedom. Healing reductions and the maximum-health limit affect the healing received.",
      ],
    ],
  );
  const hunter = classRows(3).rows;
  assert.equal(hunter.get("Passive Name"), "Far Sight");
  assert.equal(
    hunter.get("Passive Description"),
    "Far Sight lets the Hunter see everything within a radius of 6.5 around it, farther than any other class. Stone still blocks its sight.",
  );
  assert.equal(hunter.get("Observation Radius"), "6.5");
  const labels = [...classRows(1).rows.keys()];
  assert.ok(labels.indexOf("Basic Name") < labels.indexOf("Ultimate Name"));
  assert.equal(labels.indexOf("Basic Description"), labels.indexOf("Basic Name") + 1);
});

test("activation titles use the ability label form", () => {
  const mage = classRows(1);
  const basic = explainActivation({
    component: "basic",
    tokenId: "basic_damage",
    sourceClassId: 1,
    authorizedClassMechanics: mage.mechanics,
    sourcePublicAgentId: mage.owner.public_agent_id,
    targetPublicAgentId: "target-for-tooltip-proof",
  });
  assert.equal(basic.title, "Arcane Blast (Mage Basic)");
  assert.equal(basic.summary, mage.rows.get("Basic Description"));
  const ultimate = explainActivation({
    component: "ultimate",
    tokenId: "mage_burst",
    sourceClassId: 1,
    authorizedClassMechanics: mage.mechanics,
    sourcePublicAgentId: mage.owner.public_agent_id,
  });
  assert.equal(ultimate.title, "Burst (Mage Ultimate)");
  // Without class mechanics the Basic keeps its name and a plain sentence.
  const bare = explainActivation({
    component: "basic",
    tokenId: "basic_heal",
    sourceClassId: 5,
  });
  assert.equal(bare.title, "Lesser Heal (Priest Basic)");
  assert.equal(bare.summary, "The Priest used Lesser Heal this tick.");
  // An unknown class falls back to the plain token label.
  assert.equal(
    explainActivation({ component: "basic", tokenId: "basic_damage", sourceClassId: 9 })
      .title,
    "Basic Damage",
  );
});
