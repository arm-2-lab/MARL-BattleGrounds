/**
 * @file Check exact aura explanation text ("Sorcerer's Empowerment (Damage Boost)",
 * "Guardian's Barrier (Damage Mitigation)", plain-English effects and the Damage Bonus
 * and Damage Reduction row labels) and unavailable results for unknown
 * semantic inputs.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { auraPresentation } from "../src/semantic-vocabulary.js";

test("aura vocabulary owns only the exact qualitative C2 copy", () => {
  const cases = [
    [
      "mage_damage_amplification",
      {
        fieldTitle: "Sorcerer's Empowerment (Damage Boost)",
        recipientTitle: "Sorcerer's Empowerment (Damage Boost)",
        fieldEffect:
          "This Mage's aura makes the Mage itself, and teammates in its reach without a spawn shield, deal more damage.",
        aggregateEffect:
          "This agent is within reach of at least one Mage's Sorcerer's Empowerment.",
        fieldEffectLabel: "Damage Bonus",
        aggregateEffectLabel: "Total Damage Bonus",
        accent: "mage",
        effectKind: "damage_dealt",
      },
    ],
    [
      "warrior_damage_mitigation",
      {
        fieldTitle: "Guardian's Barrier (Damage Mitigation)",
        recipientTitle: "Guardian's Barrier (Damage Mitigation)",
        fieldEffect:
          "This Warrior's aura makes the Warrior itself, and teammates in its reach without a spawn shield, take less damage.",
        aggregateEffect:
          "This agent is within reach of at least one Warrior's Guardian's Barrier.",
        fieldEffectLabel: "Damage Reduction",
        aggregateEffectLabel: "Total Damage Reduction",
        accent: "warrior",
        effectKind: "damage_received",
      },
    ],
  ];
  for (const [auraId, expected] of cases) {
    const presentation = auraPresentation(auraId);
    assert.deepEqual(presentation, expected);
    assert.equal(Object.isFrozen(presentation), true);
  }
  assert.equal(
    auraPresentation("mage_amplification"),
    auraPresentation("mage_damage_amplification"),
  );
  assert.equal(
    auraPresentation(" warrior_mitigation "),
    auraPresentation("warrior_damage_mitigation"),
  );
});

test("unknown aura inputs retain no class guide or source fallback", () => {
  for (const auraId of ["", "paladin_aura", null, 7]) {
    const presentation = auraPresentation(auraId);
    assert.equal(presentation.fieldTitle, "Recorded Aura Field");
    assert.equal(presentation.recipientTitle, "Recorded Aura");
    assert.equal(presentation.accent, "none");
    assert.equal(Object.hasOwn(presentation, "ultimateName"), false);
    assert.equal(Object.hasOwn(presentation, "source"), false);
  }
});
