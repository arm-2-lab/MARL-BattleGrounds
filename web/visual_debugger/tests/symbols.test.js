/**
 * @file Check the one symbol list in symbols.js: 58 frozen symbols in the nine
 * Visual Key groups with unique IDs and aliases; the owner's subtitle rule
 * (subtitles only for status effects, class callings and ability labels; every
 * other symbol shows its plain name); the fifteen approved status names in
 * STATUS_NAMES (owner, 2 October 2026: the cause's name, then the plain
 * effect, "Charge (Stun)" and the rest, in Champion Card order with a
 * class-only source sentence each), a two-cause mark listing both names (the
 * stun stars: Charge (Stun) and Crippling Poison (Stun)) and no old flavor
 * word (Sapped, Snagged, Moon Dew and the rest) left; class, Basic and
 * Ultimate names read from their owners; the keys V, B and F; the Sight Reach
 * veil sentence; explanations with no tuning numbers; number sources that name
 * real fields of the generated host schema; no retired or banned names; and
 * the lookup helpers, including the "Spawn Shield 3" chip text.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { CLASS_IDENTITIES } from "../src/agent-identity.js";
import { AUTHORIZED_PRESENTATION_SCHEMA_V1 } from "../src/authorized-presentation-schema.js";
import { classNamesV1 } from "../src/class-documentation.js";
import {
  STATUS_NAMES,
  SYMBOL_GROUPS,
  SYMBOLS,
  statusNameFor,
  statusOrderIndex,
  symbolById,
  symbolChipText,
  symbolForClass,
  symbolForLifecycleToken,
  symbolForModifierToken,
  symbolForStatusToken,
  symbolsInGroup,
} from "../src/symbols.js";
import {
  CANONICAL_STATUS_ORDER,
  CATALOG_STATUS_IDS,
  MODIFIER_TOKEN_IDS,
  resolveVisualToken,
} from "../src/vocabulary.js";

const SUBTITLED = new Map([
  ["class_mage", "The Astral Walker"],
  ["class_warrior", "The Undying"],
  ["class_hunter", "The Sentinel"],
  ["class_rogue", "The Shadowblade"],
  ["class_priest", "The Lightbringer"],
  ["stun_frozen", "Stun"],
  ["stun_broken", "Stun Broken"],
  ["slow_snagged", "Slow"],
  ["anti_heal", "Healing Reduction"],
  ["freedom", "Slow Resistance"],
  ["damage_boost", "Damage Boost"],
  ["spawn_shield", "Invulnerable"],
  ["rest", "Regeneration"],
  ["aura_empowerment", "Damage Boost"],
  ["aura_barrier", "Damage Mitigation"],
  ["basic_mage", "Mage Basic"],
  ["basic_warrior", "Warrior Basic"],
  ["basic_hunter", "Hunter Basic"],
  ["basic_rogue", "Rogue Basic"],
  ["basic_priest", "Priest Basic"],
  ["ult_mage", "Mage Ultimate"],
  ["ult_warrior", "Warrior Ultimate"],
  ["ult_hunter", "Hunter Ultimate"],
  ["ult_rogue", "Rogue Ultimate"],
  ["ult_priest", "Priest Ultimate"],
]);

/** The two status marks that serve two causes, with the names they list. */
const TWO_CAUSE_MARKS = new Map([
  ["stun_sapped", ["Charge (Stun)", "Crippling Poison (Stun)"]],
  ["slow_hamstrung", ["Charge (Slow)", "Crippling Poison (Slow)"]],
]);

const PLAIN_NAMES = [
  "Champion",
  "Start Spot",
  "In Combat",
  "Blow",
  "Fall",
  "Last Seen",
  "Fighters Standing",
  "More Statuses",
  "Agent POV",
  "Oracle View",
  "Whole Map",
  "Life",
  "Ultimate Charge",
  "Fallen",
  "Returns In",
  "Return",
  "Red Zone",
  "x2 Mark",
  "Spawn Pad",
  "Stone",
  "Out Of Sight",
  "Score",
  "Tick",
  "Chronicle",
  "Damage",
  "Healing",
  "Team A",
  "Team B",
  "Everyone In View",
  "Battle Cam",
  "Frame Explanation",
];

/** @param {unknown} value */
function assertRecursivelyFrozen(value) {
  if (value === null || typeof value !== "object") return;
  assert.equal(Object.isFrozen(value), true);
  for (const child of Object.values(value)) assertRecursivelyFrozen(child);
}

test("the list holds 58 frozen symbols in the nine Visual Key groups", () => {
  assert.deepEqual(SYMBOL_GROUPS, [
    "Bodies And Rings",
    "Classes",
    "Statuses",
    "Basics",
    "Ultimates",
    "Deaths And Returns",
    "Zones",
    "HUD Marks",
    "Camera And Keys",
  ]);
  assert.equal(SYMBOLS.length, 58);
  assertRecursivelyFrozen(SYMBOLS);
  assertRecursivelyFrozen(SYMBOL_GROUPS);
  const groupOrder = SYMBOLS.map((entry) => SYMBOL_GROUPS.indexOf(entry.group));
  assert.equal(groupOrder.includes(-1), false);
  assert.deepEqual(
    groupOrder,
    [...groupOrder].sort((a, b) => a - b),
  );
  const keys = SYMBOLS.flatMap((entry) => [entry.id, ...entry.aliases]);
  assert.equal(new Set(keys).size, keys.length);
  for (const group of SYMBOL_GROUPS) {
    assert.deepEqual(
      symbolsInGroup(group),
      SYMBOLS.filter((entry) => entry.group === group),
    );
    assert.ok(symbolsInGroup(group).length > 0, group);
  }
  assert.deepEqual(symbolsInGroup("Azure Host"), []);
});

test("only statuses, callings and ability labels carry a subtitle", () => {
  for (const entry of SYMBOLS) {
    const expected = SUBTITLED.get(entry.id) ?? null;
    assert.equal(entry.subtitle, expected, entry.id);
    if (TWO_CAUSE_MARKS.has(entry.id)) continue;
    assert.equal(
      entry.title,
      expected === null ? entry.name : `${entry.name} (${expected})`,
      entry.id,
    );
    assert.deepEqual(entry.names, [entry.title], entry.id);
  }
  for (const [id, names] of TWO_CAUSE_MARKS) {
    const entry = symbolById(id);
    assert.deepEqual(entry?.names, names, id);
    assert.equal(entry?.title, names.join(" / "), id);
  }
  for (const name of PLAIN_NAMES) {
    const matches = SYMBOLS.filter((entry) => entry.name === name);
    assert.equal(matches.length, 1, name);
    assert.equal(matches[0].subtitle, null, name);
    assert.equal(matches[0].title, name, name);
  }
});

test("status symbols carry the owner's status names", () => {
  assert.deepEqual(
    STATUS_NAMES.map((entry) => [entry.id, entry.title, entry.group, entry.from]),
    [
      ["stun_hunter_trap", "Freezing Trap (Stun)", "stun", "From a Hunter."],
      ["stun_warrior_charge", "Charge (Stun)", "stun", "From a Warrior."],
      ["stun_rogue_poison", "Crippling Poison (Stun)", "stun", "From a Rogue."],
      ["stun_broken", "Freezing Trap (Stun Broken)", "stun", "From a Hunter."],
      ["slow_warrior_charge", "Charge (Slow)", "slow", "From a Warrior."],
      ["slow_rogue_poison", "Crippling Poison (Slow)", "slow", "From a Rogue."],
      ["slow_hunter_basic", "Serrated Arrows (Slow)", "slow", "From a Hunter."],
      [
        "anti_heal_rogue_poison",
        "Crippling Poison (Healing Reduction)",
        "other",
        "From a Rogue.",
      ],
      [
        "priest_freedom",
        "Blessing of Freedom (Slow Resistance)",
        "other",
        "From a Priest.",
      ],
      ["mage_burst", "Burst (Damage Boost)", "other", "From its own Ultimate."],
      ["spawn_shield", "Spawn Shield (Invulnerable)", "other", null],
      ["in_combat", "In Combat", "other", null],
      ["rest", "Rest (Regeneration)", "other", null],
      [
        "mage_amplification",
        "Sorcerer's Empowerment (Damage Boost)",
        "other",
        "From a Mage's aura.",
      ],
      [
        "warrior_mitigation",
        "Guardian's Barrier (Damage Mitigation)",
        "other",
        "From a Warrior's aura.",
      ],
    ],
  );
  assertRecursivelyFrozen(STATUS_NAMES);
  for (const entry of STATUS_NAMES) {
    // Each name points at the mark that shows it, and that mark lists it.
    assert.ok(symbolById(entry.symbolId)?.names.includes(entry.title), entry.id);
    assert.equal(statusNameFor(entry.id), entry);
    assert.equal(statusOrderIndex(entry.id), STATUS_NAMES.indexOf(entry));
    // A sentence with a number has a plain twin without one.
    assert.equal(entry.sentence.includes("{amount}"), entry.amountKind !== "none");
    assert.doesNotMatch(entry.plainSentence, /\{amount\}|\d/u, entry.id);
  }
  // The two aura passives carry the approved aura names.
  assert.equal(
    classNamesV1(1)?.passiveName,
    statusNameFor("mage_amplification")?.title,
  );
  assert.equal(
    classNamesV1(2)?.passiveName,
    statusNameFor("warrior_mitigation")?.title,
  );
  assert.equal(statusNameFor("trap_broken")?.id, "stun_broken");
  assert.equal(statusNameFor("trap_broken_and_reapplied")?.id, "stun_broken");
  assert.equal(statusNameFor("rogue_anti_heal")?.id, "anti_heal_rogue_poison");
  assert.equal(statusNameFor("mage_damage_amplification")?.id, "mage_amplification");
  assert.equal(statusNameFor("hunter_basic_slow"), null);
  assert.equal(statusNameFor(7), null);
  assert.equal(statusOrderIndex("future_status"), STATUS_NAMES.length);
  assert.equal(
    symbolForLifecycleToken("trap_broken")?.title,
    "Freezing Trap (Stun Broken)",
  );
  assert.equal(symbolForLifecycleToken("applied"), null);
  assert.equal(symbolById("rest")?.title, "Rest (Regeneration)");
  // One owner: the display tokens show exactly the approved names.
  for (const tokenId of [...CANONICAL_STATUS_ORDER, "in_combat"]) {
    const token = resolveVisualToken("status", tokenId);
    assert.equal(token.label, statusNameFor(tokenId)?.title, tokenId);
    assert.equal(token.shortLabel, token.label, tokenId);
    assert.ok(symbolForStatusToken(tokenId)?.names.includes(token.label), tokenId);
  }
  for (const tokenId of MODIFIER_TOKEN_IDS) {
    const token = resolveVisualToken("modifier", tokenId);
    assert.equal(token.label, statusNameFor(tokenId)?.title, tokenId);
    assert.equal(token.shortLabel, statusNameFor(tokenId)?.name, tokenId);
  }
});

test("class and ability symbols read their names from their owners", () => {
  for (const identity of Object.values(CLASS_IDENTITIES)) {
    const names = classNamesV1(identity.classId);
    assert.ok(names);
    const classSymbol = symbolForClass(identity.classId);
    const basic = symbolForClass(identity.classId, "basic");
    const ultimate = symbolForClass(identity.classId, "ultimate");
    assert.equal(classSymbol?.title, `${identity.label} (${names.calling})`);
    assert.equal(basic?.title, `${names.basicName} (${identity.label} Basic)`);
    assert.equal(ultimate?.title, `${names.ultimateName} (${identity.label} Ultimate)`);
    for (const entry of [classSymbol, basic, ultimate]) {
      assert.equal(entry?.classId, identity.classId);
    }
  }
  assert.equal(symbolForClass(1, "basic")?.title, "Arcane Blast (Mage Basic)");
  assert.equal(symbolForClass(3)?.title, "Hunter (The Sentinel)");
  for (const invalid of [0, 6, "1", null, 1.5]) {
    assert.equal(symbolForClass(invalid), null);
  }
  assert.equal(
    symbolForClass(1, /** @type {"class"} */ (/** @type {unknown} */ ("passive"))),
    null,
  );
});

test("the camera and explanation symbols carry the keys V, B and F", () => {
  assert.deepEqual(
    SYMBOLS.filter((entry) => entry.key !== null).map((entry) => [
      entry.id,
      entry.name,
      entry.key,
    ]),
    [
      ["camera_everyone_in_view", "Everyone In View", "V"],
      ["camera_battle_cam", "Battle Cam", "B"],
      ["frame_explanation", "Frame Explanation", "F"],
    ],
  );
  assert.equal(symbolById("camera_whole_map")?.key, null);
  assert.equal(symbolById("camera_battle_director")?.id, "camera_battle_cam");
  assert.match(
    symbolById("camera_battle_cam")?.eli5 ?? "",
    /^Replay Viewer only: press B/u,
  );
});

test("the Sight Reach veil states that stone and fresh returns still hide champions", () => {
  const veil = symbolById("sight_veil");
  assert.equal(veil?.name, "Out Of Sight");
  assert.ok(
    veil?.eli5.endsWith(
      "Stone also blocks sight inside the circle, and champions just back on their pads stay hidden.",
    ),
  );
  assert.equal(symbolById("sight_reach"), veil);
});

test("explanations are short plain sentences with no tuning numbers", () => {
  // Only game facts the owner wrote into the look carry a digit: life 0 means a
  // fall, and a fall is worth 1 point, or 2 in a Red Zone (the x2 Mark).
  const allowedDigits = new Set([
    "life_bar",
    "fall",
    "red_zone",
    "red_zone_x2",
    "score",
  ]);
  for (const entry of SYMBOLS) {
    assert.ok(entry.eli5.length > 0, entry.id);
    assert.ok(entry.eli5.length <= 260, entry.id);
    assert.match(entry.eli5, /[.]$/u, entry.id);
    assert.equal(/\d/u.test(entry.eli5), allowedDigits.has(entry.id), entry.id);
    assert.doesNotMatch(entry.eli5, /x\d|×|%|\d+\.\d/u, entry.id);
  }
  assert.equal(
    symbolById("red_zone")?.eli5,
    "Your home zone: die here and the enemy scores 2.",
  );
});

test("no retired name, look name or team colour name is shown", () => {
  const shown = [
    ...SYMBOLS.map((entry) =>
      [entry.name, entry.subtitle, entry.title, ...entry.names, entry.eli5].join(" | "),
    ),
    ...STATUS_NAMES.map((entry) =>
      [entry.title, entry.sentence, entry.plainSentence, entry.from].join(" | "),
    ),
  ].join("\n");
  for (const banned of [
    /Azure/u,
    /Crimson/u,
    /\bHost\b/u,
    /Battle Director/u,
    /Explain This Frame/u,
    /Warden/iu,
    /Dazed/u,
    /Hobbled/u,
    /Sapped/u,
    /\bFrozen\b/u,
    /Shattered/u,
    /Snagged/u,
    /Hamstrung/u,
    /Withering/u,
    /Unbound/u,
    /Kindled/u,
    /Moon Dew/u,
    /Damage Aura/u,
    /Guard Aura/u,
    /Anti-Heal/u,
    /Moonward/u,
    /Grave Ground/u,
    /Ivory Arc/u,
    /Rune Ring/u,
    /Ashen Token/u,
    /The Token/u,
    /Battle-Warm/u,
    /The Stroke/u,
    /Requiem/u,
    /Ghost Ring/u,
    /Muster/u,
    /Night Veil/u,
    /First To 20/u,
    /Match Clock/u,
    /Score Tides/u,
    /Waystone/u,
    /Standing Stone/u,
    /Skull Rune/u,
    /One Team's Eyes/u,
    /Moon's Eye/u,
    /Still Camera/u,
    /Key [VBF]\)/u,
  ]) {
    assert.doesNotMatch(shown, banned);
  }
});

test("every number source names a real field of the generated host schema", () => {
  const defs = AUTHORIZED_PRESENTATION_SCHEMA_V1.$defs;
  const statusIds = new Set(CATALOG_STATUS_IDS);
  const auraIds = new Set(["mage_damage_amplification", "warrior_damage_mitigation"]);
  let count = 0;
  for (const entry of SYMBOLS) {
    for (const source of entry.numberSources) {
      count += 1;
      const record = defs[source.record];
      assert.ok(record, `${entry.id}: ${source.record}`);
      assert.ok(
        Object.hasOwn(record.properties ?? {}, source.field),
        `${entry.id}: ${source.record}.${source.field}`,
      );
      assert.ok(source.meaning.length > 0);
      assert.doesNotMatch(source.meaning, /\d/u, entry.id);
      const select = source.select ?? {};
      for (const key of Object.keys(select)) {
        assert.ok(["class_id", "status_id", "aura_id"].includes(key), key);
      }
      if (select.status_id !== undefined) {
        assert.equal(source.record, "AuthorizedClassStatusMechanicV1");
        assert.ok(statusIds.has(select.status_id), select.status_id);
      }
      if (select.aura_id !== undefined) {
        assert.equal(source.record, "AuthorizedClassAuraMechanicV1");
        assert.ok(auraIds.has(select.aura_id), select.aura_id);
      }
      if (select.class_id !== undefined) {
        assert.equal(source.record, "AuthorizedClassMechanicsV2");
        assert.ok([1, 2, 3, 4, 5].includes(select.class_id));
      }
    }
  }
  assert.ok(count >= 100, `only ${count} number sources`);
  for (const id of ["champion", "life_bar", "slow_snagged", "spawn_shield", "score"]) {
    assert.ok((symbolById(id)?.numberSources.length ?? 0) > 0, id);
  }
});

test("lookups accept IDs and aliases, and chips show the plain name with a count", () => {
  assert.equal(symbolById("champion")?.name, "Champion");
  assert.equal(symbolById(" token ")?.id, "champion");
  assert.equal(symbolById("stun_dazed")?.id, "stun_sapped");
  assert.equal(symbolById("slow_hobbled")?.id, "slow_hamstrung");
  for (const invalid of ["", "Champion", "constructor", "__proto__", null, 7, {}]) {
    assert.equal(symbolById(invalid), null);
  }
  for (const tokenId of [...CANONICAL_STATUS_ORDER, "spawn_shield", "in_combat"]) {
    assert.ok(symbolForStatusToken(tokenId), tokenId);
  }
  assert.equal(symbolForStatusToken("hunter_basic_slow"), null);
  assert.equal(symbolForStatusToken(null), null);
  assert.equal(symbolForModifierToken("future_aura"), null);
  assert.equal(symbolChipText("spawn_shield", 3), "Spawn Shield 3");
  assert.equal(symbolChipText("stun_sapped", 1), "Stun 1");
  assert.equal(symbolChipText("in_combat", 2), "In Combat 2");
  assert.equal(symbolChipText("spawn_shield"), "Spawn Shield");
  assert.equal(symbolChipText("spawn_shield", 2.5), "Spawn Shield");
  assert.equal(symbolChipText("future_symbol", 3), null);
});
