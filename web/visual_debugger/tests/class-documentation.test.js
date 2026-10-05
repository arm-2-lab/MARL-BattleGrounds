/**
 * @file Check the five class guides, required profile identity and exact named value
 * substitution. Each guide now carries the class name, calling, order, fantasy
 * line, the lore description as its overview, the lore role as its Role row,
 * and a named Basic (Arcane Blast with the short form "Blast", Shield Bash,
 * Serrated Arrows, Backstab, Lesser Heal) beside the Ultimate and passive; the
 * Hunter's passive is the provisional Far Sight.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  requiredClassDocumentationValueNamesV1,
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

const cases = [
  {
    classId: 1,
    values: {
      burstDuration: "<Burst Duration>",
      burstDamageEffect: "<Burst Damage Effect>",
      auraRadius: "<Aura Radius>",
      perEmitterDamageAmplificationEffect: "<Per-Emitter Damage Amplification Effect>",
      damageAmplificationCeiling: "<Damage Amplification Ceiling>",
    },
    expected: {
      className: "Mage",
      calling: "The Astral Walker",
      order: "The Council of Caelmont",
      fantasyLine:
        "An archmage of Caelmont, the violet city that floats above the clouds, who speaks one word of star-lore and a star answers.",
      overview:
        "The Mage is an archmage of the Council of Caelmont, the violet city that floats above the clouds. It has the strongest Basic in the game but the lowest health, so its team must keep it safe. The Mage and allies close to it deal more damage. Its Ultimate, Burst, lights a star inside the Mage, and its own blows hit harder for the next few ticks.",
      tacticalGuideRows: [
        {
          label: "Role",
          value: "Artillery / Glass Cannon",
        },
        {
          label: "Primary Strength",
          value: "Highest Basic damage among damage-dealing classes.",
        },
        {
          label: "Primary Weakness",
          value: "Lowest maximum health; glass cannon.",
        },
      ],
      basic: {
        name: "Arcane Blast",
        shortName: "Blast",
        description:
          "Arcane Blast hits one enemy in reach with a bolt of starlight. It is the strongest Basic in the game. Damage bonuses and reductions affect the damage dealt.",
      },
      ultimate: {
        name: "Burst",
        description:
          "For <Burst Duration>, Burst applies <Burst Damage Effect> to this Mage's damage, starting on the next tick.",
      },
      passive: {
        name: "Sorcerer's Empowerment (Damage Boost)",
        description:
          "While alive and without a spawn shield, the Mage gives off Sorcerer's Empowerment. It strengthens the attacks of living teammates within <Aura Radius> who also have no spawn shield, including the Mage itself. Each aura gives <Per-Emitter Damage Amplification Effect>. Overlapping auras multiply their damage bonuses, up to <Damage Amplification Ceiling> times normal damage.",
      },
    },
  },
  {
    classId: 2,
    values: {
      ultimateRawDamage: "<Ultimate Raw Damage>",
      chargeStunDuration: "<Charge Stun Duration>",
      chargeSlowEffect: "<Charge Slow Effect>",
      chargeSlowDuration: "<Charge Slow Duration>",
      auraRadius: "<Aura Radius>",
      perEmitterDamageMitigationEffect: "<Per-Emitter Damage Mitigation Effect>",
      damageMitigationFloor: "<Damage Mitigation Floor>",
    },
    expected: {
      className: "Warrior",
      calling: "The Undying",
      order: "The Knights of Lornhold",
      fantasyLine:
        "A knight of fallen Lornhold whose kingdom died but whose oath did not: it plants the line and holds it.",
      overview:
        "The Warrior is a knight of Lornhold, a fallen kingdom whose knights still keep their oath to hold the line. It has the most health in the game, and allies close to it take less damage. Its Basic is weak, so it needs allies to finish fights. Its Ultimate, Charge, rushes it in one straight line up to a visible enemy, then hurts, stuns and slows that enemy without pushing it back.",
      tacticalGuideRows: [
        {
          label: "Role",
          value: "Tank / Guardian",
        },
        { label: "Primary Strength", value: "Highest maximum health." },
        {
          label: "Primary Weakness",
          value: "Second-lowest Basic damage among damage-dealing classes.",
        },
      ],
      basic: {
        name: "Shield Bash",
        shortName: "Shield Bash",
        description:
          "Shield Bash hits one enemy close to the Warrior. Damage bonuses and reductions affect the damage dealt.",
      },
      ultimate: {
        name: "Charge",
        description:
          "Charge moves the Warrior toward an enemy before normal movement. It deals <Ultimate Raw Damage> base damage, stuns for <Charge Stun Duration>, and applies <Charge Slow Effect> for <Charge Slow Duration>. Damage bonuses and reductions affect the damage dealt.",
      },
      passive: {
        name: "Guardian's Barrier (Damage Mitigation)",
        description:
          "While alive and without a spawn shield, the Warrior gives off Guardian's Barrier. It protects living teammates within <Aura Radius> who also have no spawn shield, including the Warrior itself. Each barrier reduces incoming damage by <Per-Emitter Damage Mitigation Effect>. When barriers overlap, each reduces the damage left after the others, down to <Damage Mitigation Floor> of the original incoming damage.",
      },
    },
  },
  {
    classId: 3,
    values: {
      hunterBasicMovementEffect: "<Hunter Basic Movement Effect>",
      hunterBasicSlowDuration: "<Hunter Basic Slow Duration>",
      ultimateRawDamage: "<Ultimate Raw Damage>",
      trapStunDuration: "<Trap Stun Duration>",
      observationRadius: "<Observation Radius>",
    },
    expected: {
      className: "Hunter",
      calling: "The Sentinel",
      order: "The Sentinels of the Lunewood",
      fantasyLine:
        "A Sentinel of the moonlit Lunewood who sees first, strikes from farthest and seals an enemy in frost.",
      overview:
        "The Hunter is a Sentinel of the Lunewood, an ancient moonlit forest whose night folk serve the Moon. It sees the farthest and its Basic reaches the farthest, but its arrows do the least damage and only slow their target a little. Its Ultimate, Freezing Trap, is thrown at one enemy and freezes it in place for the longest stun in the game. Any damage on a later tick breaks the ice early, so allies must choose: wait, or strike.",
      tacticalGuideRows: [
        { label: "Role", value: "Crowd Control / Scout" },
        {
          label: "Primary Strength",
          value: "Longest stun duration.",
        },
        {
          label: "Primary Weakness",
          value: "Lowest Basic damage among damage-dealing classes.",
        },
      ],
      basic: {
        name: "Serrated Arrows",
        shortName: "Serrated Arrows",
        description:
          "Serrated Arrows hit one enemy in reach. Each hit also slows that enemy, causing <Hunter Basic Movement Effect> for <Hunter Basic Slow Duration>. A new hit starts the slow again.",
      },
      ultimate: {
        name: "Freezing Trap",
        description:
          "Freezing Trap deals <Ultimate Raw Damage> base damage to one enemy and stuns it for <Trap Stun Duration>. Damage bonuses and reductions affect the damage dealt. Damage breaks an existing trap.",
      },
      passive: {
        name: "Far Sight",
        description:
          "Far Sight lets the Hunter see everything within <Observation Radius> around it, farther than any other class. Stone still blocks its sight.",
      },
    },
  },
  {
    classId: 4,
    values: {
      ultimateRawDamage: "<Ultimate Raw Damage>",
      poisonStunDuration: "<Poison Stun Duration>",
      poisonSlowEffect: "<Poison Slow Effect>",
      poisonSlowDuration: "<Poison Slow Duration>",
      poisonAntiHealEffect: "<Poison Anti-Heal Effect>",
      poisonAntiHealDuration: "<Poison Anti-Heal Duration>",
      baseMovementSpeed: "<Base Movement Speed>",
      outOfCombatDelay: "<Out-of-Combat Delay>",
    },
    expected: {
      className: "Rogue",
      calling: "The Shadowblade",
      order: "The Nightshade Office",
      fantasyLine:
        "An agent of the Nightshade Office, the crown's secret service of spies: a whisper, a cut, and a wound that will not heal.",
      overview:
        "The Rogue is an agent of the Nightshade Office, the secret service of spies and assassins who work for the crown of Galehaven. It moves faster than any other class, but it cannot turn invisible, blink or teleport: it wins with speed, stone cover and good timing. Its Ultimate, Crippling Poison, is the biggest single hit in the game; it also stuns, slows, and cuts the healing its target gets for a few ticks. The poison does no damage over time.",
      tacticalGuideRows: [
        { label: "Role", value: "Assassin / Flanker" },
        {
          label: "Primary Strength",
          value: "Highest base movement speed.",
        },
        {
          label: "Primary Weakness",
          value: "Short attack range with no gap closer.",
        },
      ],
      basic: {
        name: "Backstab",
        shortName: "Backstab",
        description:
          "Backstab hits one enemy close to the Rogue. Damage bonuses and reductions affect the damage dealt.",
      },
      ultimate: {
        name: "Crippling Poison",
        description:
          "Crippling Poison deals <Ultimate Raw Damage> base damage to one enemy, stuns it for <Poison Stun Duration>, and applies <Poison Slow Effect> for <Poison Slow Duration>. It also applies <Poison Anti-Heal Effect> to healing and out-of-combat regeneration for <Poison Anti-Heal Duration>. Damage bonuses and reductions affect the damage dealt.",
      },
      passive: {
        name: "Phantom's Quickness",
        description:
          "Phantom's Quickness gives the Rogue a base movement speed of <Base Movement Speed>, the highest in the game. After <Out-of-Combat Delay> out of combat, it regenerates health each tick at the rate shown under Out-of-Combat Regeneration.",
      },
    },
  },
  {
    classId: 5,
    values: {
      ultimateRawHealing: "<Ultimate Raw Healing>",
      freedomDuration: "<Freedom Duration>",
      freedomMovementFloor: "<Freedom Movement Floor>",
    },
    expected: {
      className: "Priest",
      calling: "The Lightbringer",
      order: "The Church of the Morning Light",
      fantasyLine:
        "A priest of the Church of the Morning Light, sent to keep one candle lit through the night battle.",
      overview:
        "The Priest is a Lightbringer of the Church of the Morning Light, sent by the Archbishop of Galehaven to keep a candle lit through the night battle. It is the only class that can heal, and it cannot deal damage at all, so its team must protect it. Its Basic, Lesser Heal, mends one ally or itself and briefly lets that ally shake off most of a strong slow (stuns still hold). Its Ultimate, Holy Word: Salvation, is a huge heal that can save an ally from a killing blow in the same tick, but it never brings back the fallen.",
      tacticalGuideRows: [
        { label: "Role", value: "Healer / Medic" },
        {
          label: "Primary Strength",
          value: "Only class with healing abilities.",
        },
        { label: "Primary Weakness", value: "Cannot deal damage." },
      ],
      basic: {
        name: "Lesser Heal",
        shortName: "Lesser Heal",
        description:
          "Lesser Heal heals one ally in reach, or the Priest itself, and gives it Blessing of Freedom. Healing reductions and the maximum-health limit affect the healing received.",
      },
      ultimate: {
        name: "Holy Word: Salvation",
        description:
          "Holy Word: Salvation heals one ally for <Ultimate Raw Healing>, before healing reductions and the maximum-health limit.",
      },
      passive: {
        name: "Blessing of Freedom",
        description:
          "Each successful Lesser Heal applies Blessing of Freedom to its target for <Freedom Duration>. The Priest can target itself. Freedom prevents slows from reducing movement below <Freedom Movement Floor>, but it does not prevent stuns.",
      },
    },
  },
];

test("the certified profile resolves the exact five numeric-free class guides", () => {
  for (const { classId, values, expected } of cases) {
    const requiredNames = requiredClassDocumentationValueNamesV1(PROFILE, classId);
    assert.deepEqual(requiredNames, Object.keys(values));
    assertRecursivelyFrozen(requiredNames);
    const before = structuredClone(values);
    const result = resolveClassDocumentationV1(PROFILE, classId, values);
    assert.deepEqual(result, expected);
    assert.deepEqual(values, before);
    assertRecursivelyFrozen(result);
    assert.equal(
      allStrings(result).some((value) => /\d/u.test(value)),
      false,
    );
    assert.equal(
      allStrings(result).some((value) => /\{\{/u.test(value)),
      false,
    );
  }
});

test("profile and class selection fail closed without name, row, or slot inference", () => {
  const values = cases[0].values;
  const invalidProfiles = [
    null,
    undefined,
    PROFILE.profile_id,
    { availability_kind: "unavailable" },
    { availability_kind: "available", profile_id: "future_profile" },
    { ...PROFILE, extra: true },
    { profile_id: PROFILE.profile_id },
  ];
  for (const profile of invalidProfiles) {
    assert.equal(requiredClassDocumentationValueNamesV1(profile, 1), null);
    assert.equal(resolveClassDocumentationV1(profile, 1, values), null);
  }
  for (const classId of [0, 6, "1", "Mage", true, 1.5, { class_id: 1 }, [1]]) {
    assert.equal(requiredClassDocumentationValueNamesV1(PROFILE, classId), null);
    assert.equal(resolveClassDocumentationV1(PROFILE, classId, values), null);
  }

  let reads = 0;
  const accessorProfile = { availability_kind: "available" };
  Object.defineProperty(accessorProfile, "profile_id", {
    enumerable: true,
    get() {
      reads += 1;
      return PROFILE.profile_id;
    },
  });
  assert.equal(requiredClassDocumentationValueNamesV1(accessorProfile, 1), null);
  assert.equal(resolveClassDocumentationV1(accessorProfile, 1, values), null);
  assert.equal(reads, 0);
});

test("interpolation requires the exact named preformatted value record", () => {
  const valid = cases[4].values;
  const missing = { ...valid };
  delete missing.freedomDuration;
  const accessor = { ...valid };
  let reads = 0;
  Object.defineProperty(accessor, "freedomDuration", {
    enumerable: true,
    get() {
      reads += 1;
      return valid.freedomDuration;
    },
  });
  const withSymbol = { ...valid };
  Object.defineProperty(withSymbol, Symbol("future"), {
    enumerable: true,
    value: "future",
  });
  for (const values of [
    null,
    [],
    missing,
    { ...valid, futureValue: "future" },
    { ...valid, freedomDuration: 3 },
    { ...valid, freedomDuration: " <Freedom Duration>" },
    { ...valid, freedomDuration: "<Freedom Duration>\n" },
    { ...valid, freedomDuration: "Unavailable" },
    { ...valid, freedomDuration: "Effect unavailable" },
    accessor,
    withSymbol,
  ]) {
    assert.equal(resolveClassDocumentationV1(PROFILE, 5, values), null);
  }
  assert.equal(reads, 0);
});
