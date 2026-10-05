/**
 * @file Check the class-name and lore owner in class-documentation.js: callings
 * (The Astral Walker, The Undying, The Sentinel, The Shadowblade, The
 * Lightbringer), Basic names with the Mage's short form "Blast", Ultimate and
 * passive names (the Hunter's passive no longer shares the name Serrated
 * Arrows), the ability name for each action lane, and class lore that carries
 * no number and no team, team colour, side or host.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  classAbilityNameV1,
  classLoreV1,
  classNamesV1,
  resolveClassDocumentationV1,
} from "../src/class-documentation.js";

const PROFILE = Object.freeze({
  availability_kind: "available",
  profile_id: "marl_battlegrounds.class_documentation.canonical_v1",
});

/** @param {unknown} value */
function assertRecursivelyFrozen(value) {
  if (value === null || typeof value !== "object") return;
  assert.equal(Object.isFrozen(value), true);
  for (const child of Object.values(value)) assertRecursivelyFrozen(child);
}

/** @param {unknown} value @returns {string[]} */
function allStrings(value) {
  if (typeof value === "string") return [value];
  if (value === null || typeof value !== "object") return [];
  return Object.values(value).flatMap(allStrings);
}

/** @type {Record<number, Record<string, string>>} */
const GUIDE_VALUES = {
  1: {
    burstDuration: "a",
    burstDamageEffect: "b",
    auraRadius: "c",
    perEmitterDamageAmplificationEffect: "d",
    damageAmplificationCeiling: "e",
  },
  2: {
    ultimateRawDamage: "a",
    chargeStunDuration: "b",
    chargeSlowEffect: "c",
    chargeSlowDuration: "d",
    auraRadius: "e",
    perEmitterDamageMitigationEffect: "f",
    damageMitigationFloor: "g",
  },
  3: {
    hunterBasicMovementEffect: "a",
    hunterBasicSlowDuration: "b",
    ultimateRawDamage: "c",
    trapStunDuration: "d",
    observationRadius: "e",
  },
  4: {
    ultimateRawDamage: "a",
    poisonStunDuration: "b",
    poisonSlowEffect: "c",
    poisonSlowDuration: "d",
    poisonAntiHealEffect: "e",
    poisonAntiHealDuration: "f",
    baseMovementSpeed: "g",
    outOfCombatDelay: "h",
  },
  5: { ultimateRawHealing: "a", freedomDuration: "b", freedomMovementFloor: "c" },
};

test("class names, callings and ability names have one owner and no profile gate", () => {
  assert.deepEqual(
    [1, 2, 3, 4, 5].map((classId) => classNamesV1(classId)),
    [
      {
        classId: 1,
        className: "Mage",
        calling: "The Astral Walker",
        basicName: "Arcane Blast",
        basicShortName: "Blast",
        ultimateName: "Burst",
        passiveName: "Sorcerer's Empowerment (Damage Boost)",
      },
      {
        classId: 2,
        className: "Warrior",
        calling: "The Undying",
        basicName: "Shield Bash",
        basicShortName: "Shield Bash",
        ultimateName: "Charge",
        passiveName: "Guardian's Barrier (Damage Mitigation)",
      },
      {
        classId: 3,
        className: "Hunter",
        calling: "The Sentinel",
        basicName: "Serrated Arrows",
        basicShortName: "Serrated Arrows",
        ultimateName: "Freezing Trap",
        passiveName: "Far Sight",
      },
      {
        classId: 4,
        className: "Rogue",
        calling: "The Shadowblade",
        basicName: "Backstab",
        basicShortName: "Backstab",
        ultimateName: "Crippling Poison",
        passiveName: "Phantom's Quickness",
      },
      {
        classId: 5,
        className: "Priest",
        calling: "The Lightbringer",
        basicName: "Lesser Heal",
        basicShortName: "Lesser Heal",
        ultimateName: "Holy Word: Salvation",
        passiveName: "Blessing of Freedom",
      },
    ],
  );
  for (const classId of [1, 2, 3, 4, 5]) {
    assertRecursivelyFrozen(classNamesV1(classId));
    assert.equal(
      classAbilityNameV1(classId, "basic"),
      classNamesV1(classId)?.basicName,
    );
    assert.equal(
      classAbilityNameV1(classId, "ultimate"),
      classNamesV1(classId)?.ultimateName,
    );
  }
  // The Hunter's Basic is Serrated Arrows, so its passive must have another name.
  assert.notEqual(classNamesV1(3)?.passiveName, classNamesV1(3)?.basicName);
  for (const invalid of [0, 6, "1", 1.5, null, undefined, "Mage"]) {
    assert.equal(classNamesV1(invalid), null);
    assert.equal(classLoreV1(invalid), null);
    assert.equal(classAbilityNameV1(invalid, "basic"), null);
  }
  for (const component of ["passive", "Basic", null]) {
    assert.equal(classAbilityNameV1(1, component), null);
  }
});

test("class lore carries no number and no team, colour side or host", () => {
  for (const classId of [1, 2, 3, 4, 5]) {
    const lore = classLoreV1(classId);
    assert.ok(lore);
    assertRecursivelyFrozen(lore);
    assert.deepEqual(Object.keys(lore), [
      "calling",
      "order",
      "home",
      "fantasyLine",
      "description",
      "guideRole",
    ]);
    const text = allStrings(lore).join(" ");
    assert.doesNotMatch(text, /\d/u);
    assert.doesNotMatch(
      text,
      /\b(team a|team b|azure|crimson|blue|red|host|left|right|warden|homeland|vigil|kindled|halves|one tick)\b/iu,
    );
    const guide = resolveClassDocumentationV1(PROFILE, classId, GUIDE_VALUES[classId]);
    assert.equal(guide?.overview, lore.description);
    assert.equal(guide?.calling, lore.calling);
    assert.equal(guide?.tacticalGuideRows[0].value, lore.guideRole);
  }
});
