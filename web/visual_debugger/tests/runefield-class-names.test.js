/**
 * @file Check the class-name owner in class-documentation.js: callings (The
 * Astral Walker, The Undying, The Sentinel, The Shadowblade, The Lightbringer),
 * Basic names with the Mage's short form "Blast", Ultimate and passive names
 * (the Hunter's passive no longer shares the name Serrated Arrows), the ability
 * name for each action lane, and that unknown classes and lanes return null.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  classAbilityNameV1,
  classLoreV1,
  classNamesV1,
} from "../src/class-documentation.js";

/** @param {unknown} value */
function assertRecursivelyFrozen(value) {
  if (value === null || typeof value !== "object") return;
  assert.equal(Object.isFrozen(value), true);
  for (const child of Object.values(value)) assertRecursivelyFrozen(child);
}

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
