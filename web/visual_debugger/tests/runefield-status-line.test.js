/**
 * @file Check the status words in semantic-vocabulary.js: the class and action
 * lane that applies each status (none for Spawn Shield, In Combat and unknown
 * statuses), and the one-line status rule statusLineV1, which is the Champion
 * Card row in one line: "Name (Effect) · N ticks left · sentence with the
 * exact number. From a <Class>." (owner rulings, 2 October 2026). The source
 * is named by class only (never an Agent ID or "a hidden source"), an aura of
 * the champion's own class reads "From its own aura.", the time words follow
 * statusTimeText ("This tick", "Now", "While near"), and a missing number, a
 * non-finite number or a non-integer or negative tick count is left out.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  statusLineV1,
  statusPresentation,
  statusTimeText,
} from "../src/semantic-vocabulary.js";

test("each status names the class and lane that applies it, and none for sourceless ones", () => {
  assert.deepEqual(
    [
      "stun_warrior_charge",
      "stun_hunter_trap",
      "stun_rogue_poison",
      "slow_warrior_charge",
      "slow_hunter_basic",
      "slow_rogue_poison",
      "anti_heal_rogue_poison",
      "priest_freedom",
      "mage_burst",
      "spawn_shield",
      "in_combat",
      "future_status",
    ].map((tokenId) => {
      const presentation = statusPresentation(tokenId);
      return [tokenId, presentation.sourceClassId, presentation.sourceComponent];
    }),
    [
      ["stun_warrior_charge", 2, "ultimate"],
      ["stun_hunter_trap", 3, "ultimate"],
      ["stun_rogue_poison", 4, "ultimate"],
      ["slow_warrior_charge", 2, "ultimate"],
      ["slow_hunter_basic", 3, "basic"],
      ["slow_rogue_poison", 4, "ultimate"],
      ["anti_heal_rogue_poison", 4, "ultimate"],
      ["priest_freedom", 5, "basic"],
      ["mage_burst", 1, "ultimate"],
      ["spawn_shield", null, null],
      ["in_combat", null, null],
      ["future_status", null, null],
    ],
  );
  const unknown = statusPresentation("future_status");
  assert.equal(unknown.title, "Recorded Status");
  assert.equal(unknown.magnitudeKind, "none");
  assert.equal(Object.isFrozen(unknown), true);
});

test("the status line is the card row in one line and leaves out missing parts", () => {
  assert.equal(
    statusLineV1({ tokenId: "slow_hunter_basic", magnitude: 0.85, remainingTicks: 1 }),
    "Serrated Arrows (Slow) · 1 tick left · It moves at ×0.85 speed. From a Hunter.",
  );
  assert.equal(
    statusLineV1({
      tokenId: "anti_heal_rogue_poison",
      magnitude: 0.5,
      remainingTicks: 3,
    }),
    "Crippling Poison (Healing Reduction) · 3 ticks left · Heals and rest give it ×0.5 healing. From a Rogue.",
  );
  assert.equal(
    statusLineV1({ tokenId: "mage_burst", magnitude: 1.5, remainingTicks: 0 }),
    "Burst (Damage Boost) · 0 ticks left · Its attacks deal ×1.5 damage. From its own Ultimate.",
  );
  assert.equal(
    statusLineV1({ tokenId: "priest_freedom", magnitude: 0.85, remainingTicks: 1 }),
    "Blessing of Freedom (Slow Resistance) · 1 tick left · Slows can't drop it below 85% speed. From a Priest.",
  );
  assert.equal(
    statusLineV1({ tokenId: "stun_warrior_charge", remainingTicks: 1 }),
    "Charge (Stun) · 1 tick left · It can't move or act. From a Warrior.",
  );
  // A non-finite number, a non-integer or negative tick count are left out.
  assert.equal(
    statusLineV1({
      tokenId: "slow_hunter_basic",
      magnitude: 0.85,
      remainingTicks: 1.5,
    }),
    "Serrated Arrows (Slow) · It moves at ×0.85 speed. From a Hunter.",
  );
  assert.equal(
    statusLineV1({
      tokenId: "slow_hunter_basic",
      magnitude: Number.NaN,
      remainingTicks: -1,
    }),
    "Serrated Arrows (Slow) · It moves a little slower. From a Hunter.",
  );
  assert.equal(
    statusLineV1({ tokenId: "in_combat", remainingTicks: 2 }),
    "In Combat · 2 ticks left · It fought recently, so it can't rest yet.",
  );
  // An aura reads "From its own aura." on a champion of its own class.
  assert.equal(
    statusLineV1({
      tokenId: "mage_amplification",
      magnitude: 1.15,
      recipientClassId: 1,
    }),
    "Sorcerer's Empowerment (Damage Boost) · While near · It deals ×1.15 damage near a Mage. From its own aura.",
  );
  assert.equal(
    statusLineV1({
      tokenId: "warrior_mitigation",
      magnitude: 0.85,
      recipientClassId: 3,
    }),
    "Guardian's Barrier (Damage Mitigation) · While near · It takes ×0.85 damage near a Warrior. From a Warrior's aura.",
  );
  assert.equal(
    statusLineV1({ tokenId: null }),
    "Recorded Status · The recording names a status this viewer does not recognize.",
  );
  // No line names an Agent ID or a hidden source.
  for (const tokenId of ["stun_hunter_trap", "slow_rogue_poison", "priest_freedom"]) {
    assert.doesNotMatch(
      statusLineV1({ tokenId, magnitude: 0.5, remainingTicks: 2 }),
      /\bID\b|hidden/u,
    );
  }
  assert.deepEqual(
    [
      statusTimeText("stun_broken", null),
      statusTimeText("rest", null),
      statusTimeText("warrior_mitigation", null),
      statusTimeText("slow_warrior_charge", 1),
      statusTimeText("slow_warrior_charge", null),
    ],
    ["This tick", "Now", "While near", "1 tick left", ""],
  );
});
