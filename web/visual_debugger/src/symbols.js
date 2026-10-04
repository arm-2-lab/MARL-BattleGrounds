/**
 * @file The one symbol list shared by every client.
 *
 * Each symbol a player can see (a ring, a status, a mark, a camera) has one
 * entry here: a stable ID, its group, its plain name, an optional subtitle,
 * one short plain-English explanation, the host fields its numbers come from,
 * and its key when it has one. The HUD, the panels, the hover and focus tips,
 * the Visual Key, Frame Explanation and the website's plain-English page read
 * this list, so a name is written once.
 *
 * Subtitle rule (owner, 1 October 2026): only statuses ("Charge (Stun)"),
 * class callings ("Hunter (The Sentinel)") and ability labels ("Arcane Blast
 * (Mage Basic)") carry a subtitle. Every other symbol shows its plain name
 * only, for example "Champion", "Red Zone" or "Whole Map".
 *
 * Status names (owner, 2 October 2026; "Don't give new names to things"): a
 * status is named by what causes it, then its plain effect in brackets:
 * "Charge (Stun)", "Serrated Arrows (Slow)", "Burst (Damage Boost)". In
 * Combat has no brackets. STATUS_NAMES below holds all fifteen names, one
 * plain sentence for each (with a place for its exact number) and where it
 * comes from by class ("From a Priest."; an aura on a champion of its own
 * class: "From its own aura."). Every surface reads them here: the
 * Champion Card, the Replay Viewer status tray, the Visual Key, Frame
 * Explanation, the status lines, tips and docs. A status mark that serves two
 * causes (the stun stars serve Charge and Crippling Poison) is one symbol
 * whose `names` lists both, so the Visual Key shows both under one picture.
 *
 * Names come from their owners: class names from `agent-identity.js`, callings
 * and ability names from `class-documentation.js`. This module owns the plain
 * names and subtitles of statuses and every other symbol. `semantic-vocabulary.js`
 * and `vocabulary.js` read status names from here.
 *
 * Numbers never live here. Each entry lists numberSources: the exact record and
 * field of the host's authorized presentation (the class mechanics catalogue,
 * the scene, the match summary or a recorded event) that holds the number at
 * run time. The explanations carry no tuning numbers. The only numbers in them
 * are game facts with no host field that the owner wrote into the look itself:
 * a fall is worth 1 point, or 2 in a Red Zone (the "x2 Mark"), and life 0 means
 * a fall. A status sentence in STATUS_NAMES holds the mark "{amount}" where the
 * caller's number goes (statusCardSentence in `semantic-vocabulary.js` fills it).
 *
 * Boundary: this is static text. It reads no frame, so it cannot leak anything
 * in Agent POV; the callers that pick a symbol for a mark own that choice.
 */
import { CLASS_IDENTITIES } from "./agent-identity.js";
import { classNamesV1 } from "./class-documentation.js";

/**
 * The nine symbol groups, in the order the Visual Key lists them.
 *
 * @type {readonly ["Bodies And Rings", "Classes", "Statuses", "Basics", "Ultimates", "Deaths And Returns", "Zones", "HUD Marks", "Camera And Keys"]}
 */
export const SYMBOL_GROUPS = Object.freeze(
  /** @type {const} */ ([
    "Bodies And Rings",
    "Classes",
    "Statuses",
    "Basics",
    "Ultimates",
    "Deaths And Returns",
    "Zones",
    "HUD Marks",
    "Camera And Keys",
  ]),
);

/** @typedef {(typeof SYMBOL_GROUPS)[number]} SymbolGroup */

/**
 * Where one number of a symbol comes from at run time.
 *
 * - record: the record name in the generated authorized presentation schema
 *   (`AUTHORIZED_PRESENTATION_SCHEMA_V1.$defs`), for example
 *   "AuthorizedClassMechanicsV2".
 * - field: the field of that record, for example "observation_radius".
 * - select: which row of a list holds the number, when the record is one row
 *   of a list: class_id (1 to 5), status_id or aura_id (host catalogue IDs).
 *   Absent when every row applies (for example the current agent's own row).
 * - meaning: what the number means, in plain words.
 *
 * @typedef {Readonly<{
 *   record: string,
 *   field: string,
 *   select?: Readonly<{class_id?: 1 | 2 | 3 | 4 | 5, status_id?: string, aura_id?: string}>,
 *   meaning: string,
 * }>} SymbolNumberSource
 */

/**
 * One symbol.
 *
 * - id: the stable lookup ID (never shown).
 * - group: one of SYMBOL_GROUPS.
 * - name: the plain name, for example "Serrated Arrows". Always shown. A
 *   status mark that serves two causes has its shared effect here ("Stun").
 * - subtitle: the bracketed subtitle, for example "Slow", or null. Only
 *   statuses, callings and ability labels have one.
 * - title: name plus subtitle, for example "Serrated Arrows (Slow)", or the
 *   plain name. A mark that serves two causes joins its names with " / ":
 *   "Charge (Stun) / Crippling Poison (Stun)".
 * - names: every name the symbol stands for, in order: just [title] for most
 *   symbols, both status names for a two-cause mark. The Visual Key shows
 *   each on its own line.
 * - eli5: one short plain-English explanation with no tuning numbers.
 * - numberSources: where its numbers come from; empty when it has none.
 * - key: the keyboard key that toggles it ("V", "B" or "F"), or null.
 * - aliases: other IDs that find it (older store IDs and common words). Never
 *   shown. Each alias belongs to one symbol only.
 * - statusTokens, modifierTokens, lifecycleTokens: the browser display token IDs
 *   (from `vocabulary.js`) this symbol names. Empty when none.
 * - classId: the class this symbol belongs to (1 to 5), or null.
 *
 * @typedef {Readonly<{
 *   id: string,
 *   group: SymbolGroup,
 *   name: string,
 *   subtitle: string | null,
 *   title: string,
 *   names: readonly string[],
 *   eli5: string,
 *   numberSources: readonly SymbolNumberSource[],
 *   key: "V" | "B" | "F" | null,
 *   aliases: readonly string[],
 *   statusTokens: readonly string[],
 *   modifierTokens: readonly string[],
 *   lifecycleTokens: readonly string[],
 *   classId: 1 | 2 | 3 | 4 | 5 | null,
 * }>} SymbolEntry
 */

/**
 * The optional parts of a symbol before it is frozen.
 *
 * @typedef {{
 *   subtitle?: string | null,
 *   names?: readonly string[],
 *   numberSources?: readonly SymbolNumberSource[],
 *   key?: "V" | "B" | "F" | null,
 *   aliases?: readonly string[],
 *   statusTokens?: readonly string[],
 *   modifierTokens?: readonly string[],
 *   lifecycleTokens?: readonly string[],
 *   classId?: 1 | 2 | 3 | 4 | 5 | null,
 * }} SymbolOptions
 */

/**
 * Join a plain name and an optional subtitle into a symbol title.
 *
 * name is the plain name. subtitle is a nonempty string or null. Return
 * "Name (Subtitle)" or just "Name". No checks happen here; the list below owns
 * which symbols may have a subtitle.
 *
 * @param {string} name
 * @param {string | null} subtitle
 * @returns {string}
 */
function titleOf(name, subtitle) {
  return subtitle === null ? name : `${name} (${subtitle})`;
}

/**
 * Build one frozen number source.
 *
 * record and field name a host schema record and field. meaning says what the
 * number is. select, when given, picks one row of a list (class, status or
 * aura); it is frozen too. Nothing is checked here; the symbols test checks
 * every record and field against the generated schema.
 *
 * @param {string} record
 * @param {string} field
 * @param {string} meaning
 * @param {SymbolNumberSource["select"]} [select]
 * @returns {SymbolNumberSource}
 */
function source(record, field, meaning, select) {
  return Object.freeze(
    select === undefined
      ? { record, field, meaning }
      : { record, field, select: Object.freeze({ ...select }), meaning },
  );
}

/**
 * A number from one class's row of the class mechanics catalogue.
 *
 * @param {string} field A field of AuthorizedClassMechanicsV2.
 * @param {string} meaning What the number means.
 * @param {1 | 2 | 3 | 4 | 5} [classId] The class row; absent means every class.
 * @returns {SymbolNumberSource}
 */
function classMechanic(field, meaning, classId) {
  return source(
    "AuthorizedClassMechanicsV2",
    field,
    meaning,
    classId === undefined ? undefined : { class_id: classId },
  );
}

/**
 * A number from one status row of the class mechanics catalogue.
 *
 * @param {string} statusId The host status ID, for example "hunter_basic_slow".
 * @param {"duration_steps" | "magnitude" | "breaks_on_positive_damage"} field
 * @param {string} meaning What the number means.
 * @returns {SymbolNumberSource}
 */
function statusMechanic(statusId, field, meaning) {
  return source("AuthorizedClassStatusMechanicV1", field, meaning, {
    status_id: statusId,
  });
}

/**
 * A number from one aura row of the class mechanics catalogue.
 *
 * @param {string} auraId The host aura ID, for example "mage_damage_amplification".
 * @param {"radius" | "per_emitter_multiplier" | "clamp_value"} field
 * @param {string} meaning What the number means.
 * @returns {SymbolNumberSource}
 */
function auraMechanic(auraId, field, meaning) {
  return source("AuthorizedClassAuraMechanicV1", field, meaning, { aura_id: auraId });
}

/**
 * Freeze one symbol entry.
 *
 * id, group, name and eli5 are required. options supplies the subtitle (null
 * by default), names (default: just the title; give two or more for a status
 * mark that serves several causes, and the title then joins them with
 * " / "), numberSources, key, aliases, token lists (all empty by default)
 * and classId (null by default). Return a frozen SymbolEntry whose arrays are
 * frozen copies. Inputs are not changed.
 *
 * @param {string} id
 * @param {SymbolGroup} group
 * @param {string} name
 * @param {string} eli5
 * @param {SymbolOptions} [options]
 * @returns {SymbolEntry}
 */
function symbol(id, group, name, eli5, options = {}) {
  const subtitle = options.subtitle ?? null;
  const names =
    options.names !== undefined && options.names.length > 0
      ? options.names
      : [titleOf(name, subtitle)];
  return Object.freeze({
    id,
    group,
    name,
    subtitle,
    title: names.join(" / "),
    names: Object.freeze([...names]),
    eli5,
    numberSources: Object.freeze([...(options.numberSources ?? [])]),
    key: options.key ?? null,
    aliases: Object.freeze([...(options.aliases ?? [])]),
    statusTokens: Object.freeze([...(options.statusTokens ?? [])]),
    modifierTokens: Object.freeze([...(options.modifierTokens ?? [])]),
    lifecycleTokens: Object.freeze([...(options.lifecycleTokens ?? [])]),
    classId: options.classId ?? null,
  });
}

/**
 * Return the names of a class from their owner, failing loudly if missing.
 *
 * classId is 1 to 5. The names always exist for these IDs; the throw only
 * guards against a broken edit of `class-documentation.js` at load time.
 *
 * @param {1 | 2 | 3 | 4 | 5} classId
 * @throws {Error} When the class names are missing.
 */
function namesOf(classId) {
  const names = classNamesV1(classId);
  if (names === null) throw new Error(`Missing class names for class ${classId}.`);
  return names;
}

/** The five class IDs in canonical order (1 Mage … 5 Priest). */
const CLASS_IDS = /** @type {const} */ ([1, 2, 3, 4, 5]);

/** One plain-English explanation per class symbol, by class ID. */
const CLASS_EXPLANATIONS = Object.freeze({
  1: "It hits hardest from range but has the least life, so its team must keep it safe.",
  2: "It has the most life and shields nearby allies, but its own blows are weak.",
  3: "It sees and shoots the farthest; its arrows slow, and its trap freezes one enemy.",
  4: "The fastest champion, with the biggest single hit; it needs stone cover to get close.",
  5: "The only healer: it keeps allies alive but cannot deal any damage.",
});

/** One plain-English explanation per Basic symbol, by class ID. */
const BASIC_EXPLANATIONS = Object.freeze({
  1: "A violet bolt bursts into a white star on one enemy.",
  2: "A blow with the shield on one enemy close to the Warrior.",
  3: "An arrow that hurts a little and slows its target a little.",
  4: "A quick stab on one enemy close to the Rogue.",
  5: "Heals one ally, or the Priest itself, and briefly keeps its legs free of strong slows.",
});

/** One plain-English explanation per Ultimate symbol, by class ID. */
const ULTIMATE_EXPLANATIONS = Object.freeze({
  1: "The Mage lights a star inside itself: its own blows hit harder for the next few ticks.",
  2: "The Warrior rushes in a straight line to one enemy, then hurts, stuns and slows it.",
  3: "A trap of frost freezes one enemy for the longest stun in the game.",
  4: "The biggest single hit: it also stuns, slows and cuts the healing its target gets.",
  5: "A huge heal on one ally that can save it from a killing blow in the same tick.",
});

/**
 * Extra number sources of each Basic beyond its amount and reach: the slow of
 * Serrated Arrows and the Blessing of Freedom of Lesser Heal. Empty for the
 * others.
 *
 * @type {Readonly<Record<1 | 2 | 3 | 4 | 5, readonly SymbolNumberSource[]>>}
 */
const BASIC_EXTRA_SOURCES = Object.freeze({
  1: Object.freeze([]),
  2: Object.freeze([]),
  3: Object.freeze([
    statusMechanic("hunter_basic_slow", "magnitude", "Speed multiplier of the slow"),
    statusMechanic("hunter_basic_slow", "duration_steps", "Ticks the slow lasts"),
  ]),
  4: Object.freeze([]),
  5: Object.freeze([
    statusMechanic(
      "priest_blessing_of_freedom_movement_floor",
      "magnitude",
      "Lowest speed multiplier a slow can reach",
    ),
    statusMechanic(
      "priest_blessing_of_freedom_movement_floor",
      "duration_steps",
      "Ticks Blessing of Freedom lasts",
    ),
  ]),
});

/**
 * Extra number sources of each Ultimate beyond its amount, reach and recharge:
 * the statuses it applies. Empty for Holy Word: Salvation.
 *
 * @type {Readonly<Record<1 | 2 | 3 | 4 | 5, readonly SymbolNumberSource[]>>}
 */
const ULTIMATE_EXTRA_SOURCES = Object.freeze({
  1: Object.freeze([
    statusMechanic("mage_burst_damage_amplification", "magnitude", "Damage multiplier"),
    statusMechanic(
      "mage_burst_damage_amplification",
      "duration_steps",
      "Ticks the boost lasts",
    ),
  ]),
  2: Object.freeze([
    statusMechanic("warrior_charge_stun", "duration_steps", "Ticks the stun lasts"),
    statusMechanic("warrior_charge_slow", "magnitude", "Speed multiplier of the slow"),
    statusMechanic("warrior_charge_slow", "duration_steps", "Ticks the slow lasts"),
  ]),
  3: Object.freeze([
    statusMechanic("hunter_trap_stun", "duration_steps", "Ticks the stun lasts"),
  ]),
  4: Object.freeze([
    statusMechanic("rogue_poison_stun", "duration_steps", "Ticks the stun lasts"),
    statusMechanic("rogue_poison_slow", "magnitude", "Speed multiplier of the slow"),
    statusMechanic("rogue_poison_slow", "duration_steps", "Ticks the slow lasts"),
    statusMechanic("rogue_poison_anti_heal", "magnitude", "Healing multiplier"),
    statusMechanic(
      "rogue_poison_anti_heal",
      "duration_steps",
      "Ticks the healing reduction lasts",
    ),
  ]),
  5: Object.freeze([]),
});

/**
 * Build the five class symbols, each with its calling as the subtitle.
 *
 * @returns {SymbolEntry[]}
 */
function classSymbols() {
  return CLASS_IDS.map((classId) => {
    const names = namesOf(classId);
    const tokenId = CLASS_IDENTITIES[classId].tokenId;
    return symbol(
      `class_${tokenId}`,
      "Classes",
      names.className,
      CLASS_EXPLANATIONS[classId],
      {
        subtitle: names.calling,
        classId,
        aliases: [tokenId, `${tokenId}_class`],
        numberSources: [
          classMechanic("maximum_health", "Full life", classId),
          classMechanic("base_movement_speed", "Distance it moves per tick", classId),
          classMechanic("observation_radius", "How far it sees", classId),
          classMechanic(
            classId === 5 ? "basic_raw_healing" : "basic_raw_damage",
            classId === 5 ? "Basic healing" : "Basic damage",
            classId,
          ),
          classMechanic("basic_interaction_radius", "Basic reach", classId),
        ],
      },
    );
  });
}

/**
 * Build the five Basic symbols, labelled "Arcane Blast (Mage Basic)" and so on.
 *
 * @returns {SymbolEntry[]}
 */
function basicSymbols() {
  return CLASS_IDS.map((classId) => {
    const names = namesOf(classId);
    const tokenId = CLASS_IDENTITIES[classId].tokenId;
    const aliasName = names.basicName.toLowerCase().replaceAll(" ", "_");
    return symbol(
      `basic_${tokenId}`,
      "Basics",
      names.basicName,
      BASIC_EXPLANATIONS[classId],
      {
        subtitle: `${names.className} Basic`,
        classId,
        aliases: [aliasName, `${tokenId}_basic`],
        numberSources: [
          classMechanic(
            classId === 5 ? "basic_raw_healing" : "basic_raw_damage",
            classId === 5 ? "Base healing" : "Base damage",
            classId,
          ),
          classMechanic("basic_interaction_radius", "Reach", classId),
          ...BASIC_EXTRA_SOURCES[classId],
        ],
      },
    );
  });
}

/**
 * Build the five Ultimate symbols, labelled "Burst (Mage Ultimate)" and so on.
 *
 * @returns {SymbolEntry[]}
 */
function ultimateSymbols() {
  return CLASS_IDS.map((classId) => {
    const names = namesOf(classId);
    const tokenId = CLASS_IDENTITIES[classId].tokenId;
    const aliasName = names.ultimateName
      .toLowerCase()
      .replaceAll(":", "")
      .replaceAll(" ", "_");
    /** @type {SymbolNumberSource[]} */
    const amount =
      classId === 1
        ? []
        : [
            classMechanic(
              classId === 5 ? "ultimate_raw_healing" : "ultimate_raw_damage",
              classId === 5 ? "Base healing" : "Base damage",
              classId,
            ),
          ];
    return symbol(
      `ult_${tokenId}`,
      "Ultimates",
      names.ultimateName,
      ULTIMATE_EXPLANATIONS[classId],
      {
        subtitle: `${names.className} Ultimate`,
        classId,
        aliases: [aliasName, `${tokenId}_ultimate`],
        numberSources: [
          ...amount,
          classMechanic("ultimate_interaction_radius", "Reach", classId),
          classMechanic("ultimate_cooldown_steps", "Ticks to recharge", classId),
          ...ULTIMATE_EXTRA_SOURCES[classId],
        ],
      },
    );
  });
}

/**
 * Every symbol, in Visual Key order (grouped as in SYMBOL_GROUPS). Frozen.
 *
 * @type {readonly SymbolEntry[]}
 */
export const SYMBOLS = Object.freeze([
  symbol(
    "champion",
    "Bodies And Rings",
    "Champion",
    "Each circle is one champion, drawn at its real size; it sees all around itself, never only ahead.",
    {
      aliases: ["token", "body", "agent"],
      numberSources: [
        classMechanic("body_radius", "Body size (radius)"),
        classMechanic("observation_radius", "How far it sees"),
      ],
    },
  ),
  symbol(
    "team_a",
    "Bodies And Rings",
    "Team A",
    "A blue ring with one band marks Team A; the color only says which side a champion is on.",
    { aliases: ["team_a_crest"] },
  ),
  symbol(
    "team_b",
    "Bodies And Rings",
    "Team B",
    "A red ring with two bands marks Team B; here red means this team, never danger.",
    { aliases: ["team_b_crest"] },
  ),
  symbol(
    "life_bar",
    "Bodies And Rings",
    "Life",
    "The pale arc around a champion, and its bar in the HUD, show how much life it has left; at 0 it falls.",
    {
      aliases: ["life", "health", "life_arc", "health_ring", "hp"],
      numberSources: [
        source("AuthorizedAgentV1", "current_health", "Life now"),
        source("AuthorizedAgentV1", "maximum_health", "Full life"),
      ],
    },
  ),
  symbol(
    "rune_ring",
    "Bodies And Rings",
    "Ultimate Charge",
    "The ring fills while the Ultimate recharges, and a full glowing ring means the Ultimate is ready.",
    {
      aliases: [
        "ultimate_charge",
        "ultimate_ready",
        "cooldown",
        "ultimate_cooldown",
        "ready",
      ],
      numberSources: [
        source(
          "AuthorizedAgentV1",
          "ultimate_cooldown_remaining",
          "Ticks until the Ultimate is ready",
        ),
        classMechanic("ultimate_cooldown_steps", "Ticks a full recharge takes"),
      ],
    },
  ),
  symbol(
    "imprint",
    "Bodies And Rings",
    "Start Spot",
    "A faint ring and hairline show where a champion stood when the tick began; the body shows where it ended.",
    { aliases: ["start_spot", "ghost", "last_move"] },
  ),
  ...classSymbols(),
  symbol(
    "stun_sapped",
    "Statuses",
    "Stun",
    "Knocked dizzy: it cannot move or act until this wears off.",
    {
      names: ["Charge (Stun)", "Crippling Poison (Stun)"],
      aliases: ["stun_dazed", "sapped", "dazed"],
      statusTokens: ["stun_warrior_charge", "stun_rogue_poison"],
      numberSources: [
        statusMechanic(
          "warrior_charge_stun",
          "duration_steps",
          "Ticks a Charge stun lasts",
        ),
        statusMechanic(
          "rogue_poison_stun",
          "duration_steps",
          "Ticks a Crippling Poison stun lasts",
        ),
        source("AuthorizedStatusV1", "remaining_duration", "Ticks left"),
      ],
    },
  ),
  symbol(
    "stun_frozen",
    "Statuses",
    "Freezing Trap",
    "Sealed in ice: it cannot move or act, and damage on a later tick breaks the ice early.",
    {
      subtitle: "Stun",
      aliases: ["frozen"],
      statusTokens: ["stun_hunter_trap"],
      numberSources: [
        statusMechanic("hunter_trap_stun", "duration_steps", "Ticks the stun lasts"),
        statusMechanic(
          "hunter_trap_stun",
          "breaks_on_positive_damage",
          "Whether damage breaks it early",
        ),
        source("AuthorizedStatusV1", "remaining_duration", "Ticks left"),
      ],
    },
  ),
  symbol(
    "stun_broken",
    "Statuses",
    "Freezing Trap",
    "It took damage, so the ice of the Freezing Trap broke early.",
    {
      subtitle: "Stun Broken",
      aliases: ["shattered", "hunter_trap_broken", "status_broken_by_damage"],
      lifecycleTokens: ["trap_broken", "trap_broken_and_reapplied"],
      numberSources: [
        statusMechanic(
          "hunter_trap_stun",
          "breaks_on_positive_damage",
          "Whether damage breaks it early",
        ),
      ],
    },
  ),
  symbol(
    "slow_snagged",
    "Statuses",
    "Serrated Arrows",
    "A barbed arrow caught its legs, so it walks a little slower.",
    {
      subtitle: "Slow",
      aliases: ["snagged"],
      statusTokens: ["slow_hunter_basic"],
      numberSources: [
        statusMechanic("hunter_basic_slow", "magnitude", "Speed multiplier"),
        statusMechanic("hunter_basic_slow", "duration_steps", "Ticks the slow lasts"),
        source("AuthorizedStatusV1", "remaining_duration", "Ticks left"),
      ],
    },
  ),
  symbol(
    "slow_hamstrung",
    "Statuses",
    "Slow",
    "Its legs are hurt, so it walks much slower for a while.",
    {
      names: ["Charge (Slow)", "Crippling Poison (Slow)"],
      aliases: ["slow_hobbled", "hamstrung", "hobbled"],
      statusTokens: ["slow_warrior_charge", "slow_rogue_poison"],
      numberSources: [
        statusMechanic(
          "warrior_charge_slow",
          "magnitude",
          "Speed multiplier from Charge",
        ),
        statusMechanic(
          "warrior_charge_slow",
          "duration_steps",
          "Ticks a Charge slow lasts",
        ),
        statusMechanic(
          "rogue_poison_slow",
          "magnitude",
          "Speed multiplier from Crippling Poison",
        ),
        statusMechanic(
          "rogue_poison_slow",
          "duration_steps",
          "Ticks a Crippling Poison slow lasts",
        ),
        source("AuthorizedStatusV1", "remaining_duration", "Ticks left"),
      ],
    },
  ),
  symbol(
    "anti_heal",
    "Statuses",
    "Crippling Poison",
    "Venom in its blood: every heal and every rest mends it less.",
    {
      subtitle: "Healing Reduction",
      aliases: ["withering", "antiheal"],
      statusTokens: ["anti_heal_rogue_poison"],
      modifierTokens: ["rogue_anti_heal"],
      numberSources: [
        statusMechanic("rogue_poison_anti_heal", "magnitude", "Healing multiplier"),
        statusMechanic("rogue_poison_anti_heal", "duration_steps", "Ticks it lasts"),
        source("AuthorizedStatusV1", "remaining_duration", "Ticks left"),
      ],
    },
  ),
  symbol(
    "freedom",
    "Statuses",
    "Blessing of Freedom",
    "A Priest's blessing keeps its legs free: slows cannot drag it below most of its speed, but a stun still holds.",
    {
      subtitle: "Slow Resistance",
      aliases: ["unbound", "blessing_of_freedom"],
      statusTokens: ["priest_freedom"],
      modifierTokens: ["priest_freedom"],
      numberSources: [
        statusMechanic(
          "priest_blessing_of_freedom_movement_floor",
          "magnitude",
          "Lowest speed multiplier a slow can reach",
        ),
        statusMechanic(
          "priest_blessing_of_freedom_movement_floor",
          "duration_steps",
          "Ticks it lasts",
        ),
        source("AuthorizedStatusV1", "remaining_duration", "Ticks left"),
      ],
    },
  ),
  symbol(
    "damage_boost",
    "Statuses",
    "Burst",
    "The Mage's Burst: its own blows hit harder for a few ticks.",
    {
      subtitle: "Damage Boost",
      aliases: ["kindled", "damage_amplification"],
      statusTokens: ["mage_burst"],
      modifierTokens: ["mage_burst"],
      numberSources: [
        statusMechanic(
          "mage_burst_damage_amplification",
          "magnitude",
          "Damage multiplier",
        ),
        statusMechanic(
          "mage_burst_damage_amplification",
          "duration_steps",
          "Ticks it lasts",
        ),
        source("AuthorizedStatusV1", "remaining_duration", "Ticks left"),
      ],
    },
  ),
  symbol(
    "spawn_shield",
    "Statuses",
    "Spawn Shield",
    "Just back from a fall: enemies can neither see nor hit it, and it cannot strike yet.",
    {
      subtitle: "Invulnerable",
      aliases: ["shield", "spawn_shield_status"],
      statusTokens: ["spawn_shield"],
      numberSources: [
        source(
          "AuthorizedSpawnShieldMechanicsAvailableV2",
          "configured_duration_steps",
          "Ticks the shield lasts",
        ),
        source(
          "AuthorizedSpawnShieldMechanicsAvailableV2",
          "movement_speed",
          "Distance it moves per tick while shielded",
        ),
        source("AuthorizedAgentV1", "spawn_shield_remaining", "Ticks left"),
      ],
    },
  ),
  symbol(
    "in_combat",
    "Statuses",
    "In Combat",
    "It fought recently, so it cannot rest yet; the count shows how many quiet ticks are left before it can.",
    {
      aliases: ["combat"],
      statusTokens: ["in_combat"],
      numberSources: [
        source(
          "AuthorizedAgentV1",
          "steps_until_out_of_combat",
          "Ticks until it leaves combat",
        ),
        classMechanic(
          "out_of_combat_delay_steps",
          "Quiet ticks needed to leave combat",
        ),
      ],
    },
  ),
  symbol(
    "rest",
    "Statuses",
    "Rest",
    "After a few quiet ticks a champion mends a little life every tick, even while it moves.",
    {
      subtitle: "Regeneration",
      aliases: ["regeneration", "regen", "moon_dew"],
      numberSources: [
        classMechanic("out_of_combat_delay_steps", "Quiet ticks before rest starts"),
        classMechanic(
          "out_of_combat_health_regeneration_fraction_per_step",
          "Share of full life mended per tick",
        ),
        classMechanic("maximum_health", "Full life"),
      ],
    },
  ),
  symbol(
    "aura_empowerment",
    "Statuses",
    "Sorcerer's Empowerment",
    "Allies close to a Mage, and the Mage itself, hit a little harder.",
    {
      subtitle: "Damage Boost",
      aliases: ["sorcerers_empowerment", "empowerment"],
      modifierTokens: ["mage_amplification"],
      numberSources: [
        auraMechanic("mage_damage_amplification", "radius", "Reach of the aura"),
        auraMechanic(
          "mage_damage_amplification",
          "per_emitter_multiplier",
          "Damage multiplier from one Mage",
        ),
        auraMechanic(
          "mage_damage_amplification",
          "clamp_value",
          "Highest multiplier when auras overlap",
        ),
      ],
    },
  ),
  symbol(
    "aura_barrier",
    "Statuses",
    "Guardian's Barrier",
    "Allies close to a Warrior, and the Warrior itself, take a little less damage.",
    {
      subtitle: "Damage Mitigation",
      aliases: ["guardians_barrier", "barrier"],
      modifierTokens: ["warrior_mitigation"],
      numberSources: [
        auraMechanic("warrior_damage_mitigation", "radius", "Reach of the aura"),
        auraMechanic(
          "warrior_damage_mitigation",
          "per_emitter_multiplier",
          "Damage-taken multiplier from one Warrior",
        ),
        auraMechanic(
          "warrior_damage_mitigation",
          "clamp_value",
          "Lowest multiplier when auras overlap",
        ),
      ],
    },
  ),
  symbol(
    "blow",
    "Basics",
    "Blow",
    "Every blow hits one target and never misses; all blows of a tick land at the same moment, from where the champions stood when it began.",
    { aliases: ["stroke", "attack"] },
  ),
  symbol(
    "damage_number",
    "Basics",
    "Damage",
    "An ivory number with a minus sign is the life a champion lost this tick.",
    {
      aliases: ["damage", "damage_text"],
      numberSources: [
        source(
          "ReplayIncomingRecipientHealthResolutionEventV1",
          "total_effective_damage",
          "Damage taken this tick, after every bonus and reduction",
        ),
        source(
          "AgentPovVisualIncomingRecipientHealthResolutionEventV1",
          "realized_net_health_change",
          "Life change this tick that the team saw",
        ),
      ],
    },
  ),
  symbol(
    "heal_number",
    "Basics",
    "Healing",
    "A gold number with a plus sign is the life a champion got back this tick.",
    {
      aliases: ["healing", "heal", "heal_text"],
      numberSources: [
        source(
          "ReplayIncomingRecipientHealthResolutionEventV1",
          "total_effective_healing",
          "Healing received this tick",
        ),
        source(
          "AgentPovVisualIncomingRecipientHealthResolutionEventV1",
          "realized_net_health_change",
          "Life change this tick that the team saw",
        ),
      ],
    },
  ),
  ...basicSymbols(),
  ...ultimateSymbols(),
  symbol(
    "fall",
    "Deaths And Returns",
    "Fall",
    "A champion whose life reaches 0 falls, and the field holds still for a moment so you see it.",
    {
      aliases: ["death", "died", "kill", "agent_died"],
      numberSources: [source("MatchSummaryV1", "scores", "Team scores after the fall")],
    },
  ),
  symbol(
    "fallen",
    "Deaths And Returns",
    "Fallen",
    "A fallen champion waits where it fell until its team's next wave calls it home.",
    {
      aliases: ["corpse", "dead"],
      numberSources: [
        source(
          "AuthorizedRespawnWaveV1",
          "countdown_steps",
          "Ticks until the next wave",
        ),
        source("AuthorizedRespawnWaveV1", "period_steps", "Ticks between waves"),
      ],
    },
  ),
  symbol(
    "returns_in",
    "Deaths And Returns",
    "Returns In",
    "How many ticks until the fallen come back with their team's next wave.",
    {
      aliases: ["wave_countdown", "respawn_countdown"],
      numberSources: [
        source(
          "AuthorizedRespawnWaveV1",
          "countdown_steps",
          "Ticks until the next wave",
        ),
      ],
    },
  ),
  symbol(
    "respawn",
    "Deaths And Returns",
    "Return",
    "The fallen come back at full life on their team's spawn pads, under a Spawn Shield.",
    {
      aliases: ["returned", "respawned", "respawn_wave", "agent_respawned", "return"],
      numberSources: [
        source("AuthorizedRespawnWaveV1", "period_steps", "Ticks between waves"),
        source(
          "AuthorizedSpawnShieldMechanicsAvailableV2",
          "configured_duration_steps",
          "Ticks the Spawn Shield lasts",
        ),
      ],
    },
  ),
  symbol(
    "red_zone",
    "Zones",
    "Red Zone",
    "Your home zone: die here and the enemy scores 2.",
    {
      aliases: ["home_zone"],
      numberSources: [
        source("AuthorizedRedZoneV1", "depth", "How deep each Red Zone reaches"),
        source("AuthorizedRedZoneV1", "team_a_x_range", "Where Team A's Red Zone lies"),
        source("AuthorizedRedZoneV1", "team_b_x_range", "Where Team B's Red Zone lies"),
      ],
    },
  ),
  symbol(
    "red_zone_x2",
    "Zones",
    "x2 Mark",
    "A reminder on each Red Zone: a champion that falls here gives the enemy 2 points.",
    { aliases: ["x2", "double_point"] },
  ),
  symbol(
    "waystone",
    "Zones",
    "Spawn Pad",
    "Each team's home pads, where its fallen champions come back.",
    {
      aliases: ["spawn_pad", "spawn"],
      numberSources: [
        source("AuthorizedSpawnPadV1", "position", "Where the pad is"),
        source("AuthorizedRespawnWaveV1", "period_steps", "Ticks between waves"),
      ],
    },
  ),
  symbol(
    "stone",
    "Zones",
    "Stone",
    "Stones block both bodies and sight: nothing can walk or see through them.",
    {
      aliases: ["obstacle", "wall", "pillar"],
      numberSources: [
        source("AuthorizedObstacleV1", "center", "Where the stone stands"),
        source("AuthorizedObstacleV1", "radius", "Size of a round stone"),
        source("AuthorizedObstacleV1", "width", "Width of a wall"),
        source("AuthorizedObstacleV1", "height", "Height of a wall on the map"),
      ],
    },
  ),
  symbol(
    "sight_veil",
    "Zones",
    "Out Of Sight",
    "In Agent POV the ground is dark outside the reach of your team's sight. Stone also blocks sight inside the circle, and champions just back on their pads stay hidden.",
    {
      aliases: ["fog", "veil", "sight_reach", "pov_sight_veil"],
      numberSources: [
        source("AuthorizedAgentV1", "observation_radius", "How far each champion sees"),
        classMechanic("observation_radius", "How far each class sees"),
      ],
    },
  ),
  symbol(
    "last_seen",
    "Zones",
    "Last Seen",
    "In Agent POV a faint ring marks where your team last saw an enemy, on the tick it slipped out of sight.",
    { aliases: ["sight_change"] },
  ),
  symbol(
    "score",
    "HUD Marks",
    "Score",
    "A team scores 1 point for each enemy that falls, or 2 if it falls in its own Red Zone; the first team to the target score wins, and if time runs out first the match is a draw.",
    {
      aliases: ["score_bar", "scorebug", "scoreboard"],
      numberSources: [
        source("MatchSummaryV1", "scores", "Each team's points"),
        source("MatchSummaryV1", "score_threshold", "Points needed to win"),
        source("MatchSummaryV1", "episode_limit", "Last tick before a draw"),
      ],
    },
  ),
  symbol(
    "tick_clock",
    "HUD Marks",
    "Tick",
    "In each tick every champion acts once, all at the same moment.",
    {
      aliases: ["tick", "clock", "timer"],
      numberSources: [
        source("MatchSummaryV1", "simulator_step_count", "Ticks played so far"),
        source("MatchSummaryV1", "episode_limit", "Last tick before a draw"),
      ],
    },
  ),
  symbol(
    "fighters_standing",
    "HUD Marks",
    "Fighters Standing",
    "How many of a team's champions are alive right now.",
    {
      aliases: ["standing", "alive_count"],
      numberSources: [source("AuthorizedAgentV1", "life_state", "Alive or fallen")],
    },
  ),
  symbol(
    "more_statuses",
    "HUD Marks",
    "More Statuses",
    "Only a few status icons fit in a row; this shows how many more there are, and hovering it lists them all.",
    { aliases: ["more", "overflow"] },
  ),
  symbol(
    "event_track",
    "HUD Marks",
    "Event Track",
    "The Replay Viewer's timeline. A bar above the line is a Team A fall and a bar below it a Team B fall; a diamond is an Ultimate (filled when it deals damage); a small pip is a return. Click a tick to go there.",
    { aliases: ["timeline", "track", "scrubber"] },
  ),
  symbol(
    "pov_caption",
    "HUD Marks",
    "Agent POV",
    "You see only what one team can see: the fair view for playing with policies, and the BattleClient opens in it.",
    { aliases: ["agent_pov", "pov"] },
  ),
  symbol(
    "oracle_tag",
    "HUD Marks",
    "Oracle View",
    "You see every champion, even ones no team can see, and the Replay Viewer opens in it.",
    { aliases: ["oracle", "oracle_view"] },
  ),
  symbol(
    "camera_whole_map",
    "Camera And Keys",
    "Whole Map",
    "The camera shows the whole arena and stays still; every client opens on it, and pressing the key of the camera you are in again (V, or B in the Replay Viewer) comes back to it.",
    { aliases: ["whole_map", "camera_whole"] },
  ),
  symbol(
    "camera_everyone_in_view",
    "Camera And Keys",
    "Everyone In View",
    "Press V and the camera fits every champion on the field without cutting one off; press V again for the Whole Map.",
    { key: "V", aliases: ["everyone_in_view", "camera_everyone", "key_v"] },
  ),
  symbol(
    "camera_battle_cam",
    "Camera And Keys",
    "Battle Cam",
    "Replay Viewer only: press B and the camera follows the fights for you (in Agent POV, only fights your team saw); press B again for the Whole Map, or V for Everyone In View.",
    {
      key: "B",
      aliases: [
        "battle_cam",
        "camera_battle_director",
        "battle_director",
        "camera_director",
        "key_b",
      ],
    },
  ),
  symbol(
    "frame_explanation",
    "Camera And Keys",
    "Frame Explanation",
    "Press F and every mark of this tick gets a number, with a plain list of what happened; in the Replay Viewer it pauses first, and F again puts it away.",
    {
      key: "F",
      aliases: ["explain_this_frame", "explain_frame", "explanation", "key_f"],
    },
  ),
]);

/**
 * Every symbol ID and alias mapped to its symbol. Built once at load time; a
 * key used twice throws then, so a broken edit fails at once.
 *
 * @type {ReadonlyMap<string, SymbolEntry>}
 */
const SYMBOL_BY_LOOKUP = (() => {
  /** @type {Map<string, SymbolEntry>} */
  const lookup = new Map();
  for (const entry of SYMBOLS) {
    for (const key of [entry.id, ...entry.aliases]) {
      if (lookup.has(key)) {
        throw new Error(`Symbol lookup key ${key} is used twice.`);
      }
      lookup.set(key, entry);
    }
  }
  return lookup;
})();

/**
 * Build a lookup from one kind of token ID to its symbol.
 *
 * pick returns the token list of one entry. Each token may belong to one
 * symbol only.
 *
 * @param {(entry: SymbolEntry) => readonly string[]} pick
 * @returns {ReadonlyMap<string, SymbolEntry>}
 * @throws {Error} At load time if two symbols claim one token.
 */
function tokenLookup(pick) {
  /** @type {Map<string, SymbolEntry>} */
  const lookup = new Map();
  for (const entry of SYMBOLS) {
    for (const tokenId of pick(entry)) {
      if (lookup.has(tokenId)) {
        throw new Error(`Display token ${tokenId} is named by two symbols.`);
      }
      lookup.set(tokenId, entry);
    }
  }
  return lookup;
}

/** Status, modifier and lifecycle display token IDs mapped to their symbol. */
const SYMBOL_BY_STATUS_TOKEN = tokenLookup((entry) => entry.statusTokens);
const SYMBOL_BY_MODIFIER_TOKEN = tokenLookup((entry) => entry.modifierTokens);
const SYMBOL_BY_LIFECYCLE_TOKEN = tokenLookup((entry) => entry.lifecycleTokens);

/**
 * Find one symbol by its ID or by one of its aliases.
 *
 * idOrAlias is a string; outer spaces are ignored and case matters. An unknown
 * value, an empty string or a non-string returns null. The returned entry is
 * shared and frozen.
 *
 * @param {unknown} idOrAlias
 * @returns {SymbolEntry | null}
 */
export function symbolById(idOrAlias) {
  if (typeof idOrAlias !== "string") return null;
  return SYMBOL_BY_LOOKUP.get(idOrAlias.trim()) ?? null;
}

/**
 * Return every symbol in one group, in Visual Key order.
 *
 * group must be one of SYMBOL_GROUPS; any other value returns an empty array.
 * The result is a new frozen array of shared entries.
 *
 * @param {unknown} group
 * @returns {readonly SymbolEntry[]}
 */
export function symbolsInGroup(group) {
  return Object.freeze(SYMBOLS.filter((entry) => entry.group === group));
}

/**
 * Find the symbol that names one status display token.
 *
 * tokenId is a status token ID from `vocabulary.js` (for example
 * "slow_hunter_basic"), or "spawn_shield" or "in_combat". Host catalogue status
 * IDs (for example "hunter_basic_slow") are not accepted here; translate them
 * with `statusTokenIdFromCatalogId` first. Unknown values return null.
 *
 * @param {unknown} tokenId
 * @returns {SymbolEntry | null}
 */
export function symbolForStatusToken(tokenId) {
  return typeof tokenId === "string"
    ? (SYMBOL_BY_STATUS_TOKEN.get(tokenId.trim()) ?? null)
    : null;
}

/**
 * Find the symbol that names one aura or status modifier display token.
 *
 * tokenId is a modifier token ID from `vocabulary.js`, for example
 * "mage_amplification" (Sorcerer's Empowerment) or "rogue_anti_heal" (Crippling
 * Poison (Healing Reduction)). Unknown
 * values return null.
 *
 * @param {unknown} tokenId
 * @returns {SymbolEntry | null}
 */
export function symbolForModifierToken(tokenId) {
  return typeof tokenId === "string"
    ? (SYMBOL_BY_MODIFIER_TOKEN.get(tokenId.trim()) ?? null)
    : null;
}

/**
 * Find the symbol that names one status lifecycle display token.
 *
 * Only "trap_broken" and "trap_broken_and_reapplied" have their own symbol
 * (Freezing Trap (Stun Broken)); every other lifecycle returns null.
 *
 * @param {unknown} tokenId
 * @returns {SymbolEntry | null}
 */
export function symbolForLifecycleToken(tokenId) {
  return typeof tokenId === "string"
    ? (SYMBOL_BY_LIFECYCLE_TOKEN.get(tokenId.trim()) ?? null)
    : null;
}

/**
 * Find a class symbol, or one of its ability symbols.
 *
 * classId must be an integer 1 (Mage) to 5 (Priest). part chooses "class"
 * (the default; "Mage (The Astral Walker)"), "basic" ("Arcane Blast (Mage
 * Basic)") or "ultimate" ("Burst (Mage Ultimate)"). Any other input returns
 * null.
 *
 * @param {unknown} classId
 * @param {"class" | "basic" | "ultimate"} [part]
 * @returns {SymbolEntry | null}
 */
export function symbolForClass(classId, part = "class") {
  if (!Number.isInteger(classId) || !Object.hasOwn(CLASS_IDENTITIES, Number(classId))) {
    return null;
  }
  const tokenId = CLASS_IDENTITIES[/** @type {1 | 2 | 3 | 4 | 5} */ (classId)].tokenId;
  const prefix =
    part === "class"
      ? "class"
      : part === "basic"
        ? "basic"
        : part === "ultimate"
          ? "ult"
          : null;
  return prefix === null ? null : symbolById(`${prefix}_${tokenId}`);
}

/**
 * Write a symbol's short chip text: its plain name, then a count.
 *
 * idOrAlias finds the symbol as symbolById does. count is an optional integer,
 * for example the ticks left: symbolChipText("spawn_shield", 3) gives
 * "Spawn Shield 3". A missing or non-integer count gives just the plain name.
 * The subtitle is never part of a chip. An unknown symbol returns null.
 *
 * @param {unknown} idOrAlias
 * @param {unknown} [count]
 * @returns {string | null}
 */
export function symbolChipText(idOrAlias, count) {
  const entry = symbolById(idOrAlias);
  if (entry === null) return null;
  return Number.isInteger(count) ? `${entry.name} ${count}` : entry.name;
}

/**
 * Where a status sits on the Champion Card: "stun" rows come first, then
 * "slow" rows, then "other" rows (owner, 2 October 2026).
 *
 * @typedef {"stun" | "slow" | "other"} StatusRowGroup
 */

/**
 * How a status's time reads on the Champion Card: "ticks" ("5 ticks left"),
 * "this_tick" ("This tick", for a one-tick event), "now" ("Now", for Rest)
 * or "while_near" ("While near", for an aura).
 *
 * @typedef {"ticks" | "this_tick" | "now" | "while_near"} StatusTimeKind
 */

/**
 * What the number in a status sentence is: "none" (the sentence has no
 * number), "multiplier" (written "×1.5"), "percent" (a share of speed,
 * written "85%") or "life" (life mended per tick, written "1.6").
 *
 * @typedef {"none" | "multiplier" | "percent" | "life"} StatusAmountKind
 */

/**
 * One status as every surface names and explains it.
 *
 * - id: the lookup ID. A status display token from `vocabulary.js`
 *   ("stun_warrior_charge"), "spawn_shield", "in_combat", "rest",
 *   "stun_broken" (a Freezing Trap that damage broke this tick) or an aura
 *   modifier token ("mage_amplification", "warrior_mitigation").
 * - name: what causes it, an existing name ("Charge", "Serrated Arrows").
 * - effect: its plain effect ("Stun", "Slow"), or null (In Combat).
 * - title: "Name (Effect)", or the name alone when effect is null.
 * - symbolId: the symbol of the mark that shows it (SYMBOLS).
 * - group: its place on the card (StatusRowGroup).
 * - sourceClassId: the class (1 Mage to 5 Priest) whose ability or aura gives
 *   it, or null when no champion gives it (Spawn Shield, In Combat, Rest).
 * - from: where it comes from, by class only, as a sentence ("From a
 *   Warrior.", "From its own Ultimate.", "From a Mage's aura."), or null.
 *   Never an Agent ID, so the same words are true in Agent POV and Oracle View.
 * - fromOwn: what `from` reads instead when the champion that has it is of
 *   the source class itself, or null when `from` always applies. Only the two
 *   auras have one: a Mage's own Sorcerer's Empowerment and a Warrior's own
 *   Guardian's Barrier read "From its own aura." (owner, 2 October 2026).
 * - sentence: what it does, in one plain sentence. "{amount}" marks where the
 *   exact number goes; statusCardSentence in `semantic-vocabulary.js` fills it.
 * - plainSentence: the same sentence for when the number is missing.
 * - amountKind: how the number is written (StatusAmountKind).
 * - time: how its time reads (StatusTimeKind).
 * - paint: the color of its row's drain line: a class ("mage", "warrior",
 *   "hunter", "rogue", "priest"), "rime" (ice), "moon" (Spawn Shield),
 *   "neutral" (In Combat) or "dew" (Rest).
 *
 * @typedef {Readonly<{
 *   id: string,
 *   name: string,
 *   effect: string | null,
 *   title: string,
 *   symbolId: string,
 *   group: StatusRowGroup,
 *   sourceClassId: 1 | 2 | 3 | 4 | 5 | null,
 *   from: string | null,
 *   fromOwn: string | null,
 *   sentence: string,
 *   plainSentence: string,
 *   amountKind: StatusAmountKind,
 *   time: StatusTimeKind,
 *   paint: string,
 * }>} StatusName
 */

/**
 * Freeze one StatusName. The title is built from name and effect; fromOwn
 * defaults to null; every other field is copied. Inputs are not changed.
 *
 * @param {Omit<StatusName, "title" | "fromOwn"> & {fromOwn?: string | null}} fields
 * @returns {StatusName}
 */
function statusName(fields) {
  return Object.freeze({
    fromOwn: null,
    ...fields,
    title: titleOf(fields.name, fields.effect),
  });
}

/** What an aura's `from` reads on a champion of the aura's own class. */
const OWN_AURA = "From its own aura.";

/** The words of every stun sentence. */
const STUN_SENTENCE = "It can't move or act.";

/** The words of every slow sentence; "{amount}" is the speed multiplier. */
const SLOW_SENTENCE = "It moves at {amount} speed.";

/**
 * The fifteen approved status names (owner, 2 October 2026), in Champion Card
 * order: stuns first, then slows, then the rest. Frozen.
 *
 * @type {readonly StatusName[]}
 */
export const STATUS_NAMES = Object.freeze([
  statusName({
    id: "stun_hunter_trap",
    name: "Freezing Trap",
    effect: "Stun",
    symbolId: "stun_frozen",
    group: "stun",
    sourceClassId: 3,
    from: "From a Hunter.",
    sentence: `${STUN_SENTENCE} Damage breaks the ice.`,
    plainSentence: `${STUN_SENTENCE} Damage breaks the ice.`,
    amountKind: "none",
    time: "ticks",
    paint: "rime",
  }),
  statusName({
    id: "stun_warrior_charge",
    name: "Charge",
    effect: "Stun",
    symbolId: "stun_sapped",
    group: "stun",
    sourceClassId: 2,
    from: "From a Warrior.",
    sentence: STUN_SENTENCE,
    plainSentence: STUN_SENTENCE,
    amountKind: "none",
    time: "ticks",
    paint: "warrior",
  }),
  statusName({
    id: "stun_rogue_poison",
    name: "Crippling Poison",
    effect: "Stun",
    symbolId: "stun_sapped",
    group: "stun",
    sourceClassId: 4,
    from: "From a Rogue.",
    sentence: STUN_SENTENCE,
    plainSentence: STUN_SENTENCE,
    amountKind: "none",
    time: "ticks",
    paint: "rogue",
  }),
  statusName({
    id: "stun_broken",
    name: "Freezing Trap",
    effect: "Stun Broken",
    symbolId: "stun_broken",
    group: "stun",
    sourceClassId: 3,
    from: "From a Hunter.",
    sentence: "Damage broke the ice early.",
    plainSentence: "Damage broke the ice early.",
    amountKind: "none",
    time: "this_tick",
    paint: "rime",
  }),
  statusName({
    id: "slow_warrior_charge",
    name: "Charge",
    effect: "Slow",
    symbolId: "slow_hamstrung",
    group: "slow",
    sourceClassId: 2,
    from: "From a Warrior.",
    sentence: SLOW_SENTENCE,
    plainSentence: "It moves slower.",
    amountKind: "multiplier",
    time: "ticks",
    paint: "warrior",
  }),
  statusName({
    id: "slow_rogue_poison",
    name: "Crippling Poison",
    effect: "Slow",
    symbolId: "slow_hamstrung",
    group: "slow",
    sourceClassId: 4,
    from: "From a Rogue.",
    sentence: SLOW_SENTENCE,
    plainSentence: "It moves slower.",
    amountKind: "multiplier",
    time: "ticks",
    paint: "rogue",
  }),
  statusName({
    id: "slow_hunter_basic",
    name: "Serrated Arrows",
    effect: "Slow",
    symbolId: "slow_snagged",
    group: "slow",
    sourceClassId: 3,
    from: "From a Hunter.",
    sentence: SLOW_SENTENCE,
    plainSentence: "It moves a little slower.",
    amountKind: "multiplier",
    time: "ticks",
    paint: "hunter",
  }),
  statusName({
    id: "anti_heal_rogue_poison",
    name: "Crippling Poison",
    effect: "Healing Reduction",
    symbolId: "anti_heal",
    group: "other",
    sourceClassId: 4,
    from: "From a Rogue.",
    sentence: "Heals and rest give it {amount} healing.",
    plainSentence: "Heals and rest mend it less.",
    amountKind: "multiplier",
    time: "ticks",
    paint: "rogue",
  }),
  statusName({
    id: "priest_freedom",
    name: "Blessing of Freedom",
    effect: "Slow Resistance",
    symbolId: "freedom",
    group: "other",
    sourceClassId: 5,
    from: "From a Priest.",
    sentence: "Slows can't drop it below {amount} speed.",
    plainSentence: "Slows can't drop it below most of its speed.",
    amountKind: "percent",
    time: "ticks",
    paint: "priest",
  }),
  statusName({
    id: "mage_burst",
    name: "Burst",
    effect: "Damage Boost",
    symbolId: "damage_boost",
    group: "other",
    sourceClassId: 1,
    from: "From its own Ultimate.",
    sentence: "Its attacks deal {amount} damage.",
    plainSentence: "Its attacks deal more damage.",
    amountKind: "multiplier",
    time: "ticks",
    paint: "mage",
  }),
  statusName({
    id: "spawn_shield",
    name: "Spawn Shield",
    effect: "Invulnerable",
    symbolId: "spawn_shield",
    group: "other",
    sourceClassId: null,
    from: null,
    sentence: "It can't be seen or hit, and can't strike yet.",
    plainSentence: "It can't be seen or hit, and can't strike yet.",
    amountKind: "none",
    time: "ticks",
    paint: "moon",
  }),
  statusName({
    id: "in_combat",
    name: "In Combat",
    effect: null,
    symbolId: "in_combat",
    group: "other",
    sourceClassId: null,
    from: null,
    sentence: "It fought recently, so it can't rest yet.",
    plainSentence: "It fought recently, so it can't rest yet.",
    amountKind: "none",
    time: "ticks",
    paint: "neutral",
  }),
  statusName({
    id: "rest",
    name: "Rest",
    effect: "Regeneration",
    symbolId: "rest",
    group: "other",
    sourceClassId: null,
    from: null,
    sentence: "It mends {amount} life every tick.",
    plainSentence: "It mends a little life every tick.",
    amountKind: "life",
    time: "now",
    paint: "dew",
  }),
  statusName({
    id: "mage_amplification",
    name: "Sorcerer's Empowerment",
    effect: "Damage Boost",
    symbolId: "aura_empowerment",
    group: "other",
    sourceClassId: 1,
    from: "From a Mage's aura.",
    fromOwn: OWN_AURA,
    sentence: "It deals {amount} damage near a Mage.",
    plainSentence: "It hits a little harder near a Mage.",
    amountKind: "multiplier",
    time: "while_near",
    paint: "mage",
  }),
  statusName({
    id: "warrior_mitigation",
    name: "Guardian's Barrier",
    effect: "Damage Mitigation",
    symbolId: "aura_barrier",
    group: "other",
    sourceClassId: 2,
    from: "From a Warrior's aura.",
    fromOwn: OWN_AURA,
    sentence: "It takes {amount} damage near a Warrior.",
    plainSentence: "It takes a little less damage near a Warrior.",
    amountKind: "multiplier",
    time: "while_near",
    paint: "warrior",
  }),
]);

/**
 * Other IDs that name the same status: the two Freezing Trap lifecycle tokens,
 * the Crippling Poison modifier token of `vocabulary.js` and the two host aura
 * IDs. (The priest_freedom and mage_burst modifier tokens already equal their
 * status IDs.) Never shown.
 */
const STATUS_NAME_ALIASES = Object.freeze({
  trap_broken: "stun_broken",
  trap_broken_and_reapplied: "stun_broken",
  rogue_anti_heal: "anti_heal_rogue_poison",
  mage_damage_amplification: "mage_amplification",
  warrior_damage_mitigation: "warrior_mitigation",
});

/** STATUS_NAMES by ID, built once. */
const STATUS_NAME_BY_ID = new Map(STATUS_NAMES.map((entry) => [entry.id, entry]));

/**
 * Find the approved name of one status.
 *
 * id is a StatusName ID (see StatusName), a Freezing Trap lifecycle token
 * ("trap_broken", "trap_broken_and_reapplied"), a modifier token
 * ("rogue_anti_heal", "priest_freedom", "mage_burst") or a host aura ID
 * ("mage_damage_amplification", "warrior_damage_mitigation"). Outer spaces
 * are ignored. Host catalogue status IDs ("hunter_basic_slow") are not
 * accepted; translate them with `statusTokenIdFromCatalogId` first. An
 * unknown value returns null. The entry is shared and frozen. Pure.
 *
 * @param {unknown} id
 * @returns {StatusName | null}
 * @example statusNameFor("slow_hunter_basic")?.title // "Serrated Arrows (Slow)"
 */
export function statusNameFor(id) {
  if (typeof id !== "string") return null;
  const key = id.trim();
  const target = Object.hasOwn(STATUS_NAME_ALIASES, key)
    ? STATUS_NAME_ALIASES[/** @type {keyof typeof STATUS_NAME_ALIASES} */ (key)]
    : key;
  return STATUS_NAME_BY_ID.get(target) ?? null;
}

/**
 * Return a status's place in Champion Card order: its index in STATUS_NAMES
 * (stuns, then slows, then the rest), or STATUS_NAMES.length for an unknown
 * ID, so unknown statuses go last. Pure.
 *
 * @param {unknown} id Any ID statusNameFor accepts.
 * @returns {number}
 */
export function statusOrderIndex(id) {
  const entry = statusNameFor(id);
  return entry === null ? STATUS_NAMES.length : STATUS_NAMES.indexOf(entry);
}
