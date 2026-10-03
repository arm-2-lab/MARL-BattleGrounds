/**
 * @file Own the class callings, ability names, lore and class guides.
 *
 * This module is the one owner of:
 *
 * - Callings: The Astral Walker (Mage), The Undying (Warrior), The Sentinel
 *   (Hunter), The Shadowblade (Rogue), The Lightbringer (Priest).
 * - Basic names: Arcane Blast (short form "Blast"), Shield Bash, Serrated
 *   Arrows, Backstab, Lesser Heal. Ultimate and passive names.
 * - Lore: each class's order, home, fantasy line and description. Lore never
 *   carries a number or a team: any order can fight for Team A or Team B.
 * - The class guide for the supported documentation profile: overview, role
 *   rows, and Basic, Ultimate and passive descriptions.
 *
 * Class names ("Mage" …) come from `agent-identity.js`. Guide numbers never
 * live here: they arrive as named, already formatted strings that
 * `explanations.js` builds from the host's class mechanics catalogue. Unknown
 * profiles, classes, or missing or extra values return no guide.
 *
 * Callers: `explanations.js` (class cards and ability titles), `symbols.js`
 * (class and ability symbols), `vocabulary.js` (Ultimate token labels),
 * `semantic-vocabulary.js` (status effect text) and `source-attribution.js`
 * (ability names in source wording). Nothing here reads a frame, so there is
 * no Agent POV concern: names and lore are public facts.
 */
import { CLASS_IDENTITIES } from "./agent-identity.js";

const CANONICAL_PROFILE_ID = "marl_battlegrounds.class_documentation.canonical_v1";
const TEMPLATE_TOKEN = /\{\{([A-Za-z][A-Za-z0-9]*)\}\}/gu;

/**
 * Return a frozen guide row containing label and value exactly as supplied.
 * Both arguments are strings owned by this module. No validation or formatting
 * is performed; the returned row can be shared by class definitions.
 *
 * @param {string} label The row label, in Title Case.
 * @param {string} value The row value, plain text with no numbers.
 * @returns {Readonly<{label: string, value: string}>}
 */
function tacticalRow(label, value) {
  return Object.freeze({ label, value });
}

/**
 * The authored text of one class, before it is frozen.
 *
 * @typedef {{
 *   classId: 1 | 2 | 3 | 4 | 5,
 *   calling: string,
 *   order: string,
 *   home: string,
 *   fantasyLine: string,
 *   description: string,
 *   guideRole: string,
 *   tacticalGuideRows: readonly Readonly<{label: string, value: string}>[],
 *   basicName: string,
 *   basicShortName: string,
 *   basicTemplate: string,
 *   ultimateName: string,
 *   ultimateTemplate: string,
 *   passiveName: string,
 *   passiveTemplate: string,
 * }} ClassDefinitionInput
 */

/**
 * Freeze one authored class definition and discover its interpolation keys.
 *
 * value supplies the class's lore, names and three description templates
 * (Basic, Ultimate, passive). Scan the templates in that order for named
 * double-brace tokens, keep the first occurrence of each name and drop
 * repeats. Add className from `agent-identity.js`. Return a frozen record with
 * frozen copies of the row and key arrays. Row objects are reused; their
 * creator owns freezing. Inputs are not validated or changed.
 *
 * @param {ClassDefinitionInput} value
 */
function definition(value) {
  const requiredValueNames = [
    ...new Set(
      [value.basicTemplate, value.ultimateTemplate, value.passiveTemplate].flatMap(
        (template) => [...template.matchAll(TEMPLATE_TOKEN)].map((match) => match[1]),
      ),
    ),
  ];
  return Object.freeze({
    ...value,
    className: CLASS_IDENTITIES[value.classId].label,
    tacticalGuideRows: Object.freeze([...value.tacticalGuideRows]),
    requiredValueNames: Object.freeze(requiredValueNames),
  });
}

const CLASS_DOCUMENTATION = Object.freeze({
  1: definition({
    classId: 1,
    calling: "The Astral Walker",
    order: "The Council of Caelmont",
    home: "Caelmont, a violet city of scholars that floats above the clouds",
    fantasyLine:
      "An archmage of Caelmont, the violet city that floats above the clouds, who speaks one word of star-lore and a star answers.",
    description:
      "The Mage is an archmage of the Council of Caelmont, the violet city that floats above the clouds. It has the strongest Basic in the game but the lowest health, so its team must keep it safe. The Mage and allies close to it deal more damage. Its Ultimate, Burst, lights a star inside the Mage, and its own blows hit harder for the next few ticks.",
    guideRole: "Artillery / Glass Cannon",
    tacticalGuideRows: [
      tacticalRow("Role", "Artillery / Glass Cannon"),
      tacticalRow(
        "Primary Strength",
        "Highest Basic damage among damage-dealing classes.",
      ),
      tacticalRow("Primary Weakness", "Lowest maximum health; glass cannon."),
    ],
    basicName: "Arcane Blast",
    basicShortName: "Blast",
    basicTemplate:
      "Arcane Blast hits one enemy in reach with a bolt of starlight. It is the strongest Basic in the game. Damage bonuses and reductions affect the damage dealt.",
    ultimateName: "Burst",
    ultimateTemplate:
      "For {{burstDuration}}, Burst applies {{burstDamageEffect}} to this Mage's damage, starting on the next tick.",
    passiveName: "Sorcerer's Empowerment (Damage Boost)",
    passiveTemplate:
      "While alive and without a spawn shield, the Mage gives off Sorcerer's Empowerment. It strengthens the attacks of living teammates within {{auraRadius}} who also have no spawn shield, including the Mage itself. Each aura gives {{perEmitterDamageAmplificationEffect}}. Overlapping auras multiply their damage bonuses, up to {{damageAmplificationCeiling}} times normal damage.",
  }),
  2: definition({
    classId: 2,
    calling: "The Undying",
    order: "The Knights of Lornhold",
    home: "Lornhold, a human kingdom that fell long ago; its knights remain",
    fantasyLine:
      "A knight of fallen Lornhold whose kingdom died but whose oath did not: it plants the line and holds it.",
    description:
      "The Warrior is a knight of Lornhold, a fallen kingdom whose knights still keep their oath to hold the line. It has the most health in the game, and allies close to it take less damage. Its Basic is weak, so it needs allies to finish fights. Its Ultimate, Charge, rushes it in one straight line up to a visible enemy, then hurts, stuns and slows that enemy without pushing it back.",
    guideRole: "Tank / Guardian",
    tacticalGuideRows: [
      tacticalRow("Role", "Tank / Guardian"),
      tacticalRow("Primary Strength", "Highest maximum health."),
      tacticalRow(
        "Primary Weakness",
        "Second-lowest Basic damage among damage-dealing classes.",
      ),
    ],
    basicName: "Shield Bash",
    basicShortName: "Shield Bash",
    basicTemplate:
      "Shield Bash hits one enemy close to the Warrior. Damage bonuses and reductions affect the damage dealt.",
    ultimateName: "Charge",
    ultimateTemplate:
      "Charge moves the Warrior toward an enemy before normal movement. It deals {{ultimateRawDamage}} base damage, stuns for {{chargeStunDuration}}, and applies {{chargeSlowEffect}} for {{chargeSlowDuration}}. Damage bonuses and reductions affect the damage dealt.",
    passiveName: "Guardian's Barrier (Damage Mitigation)",
    passiveTemplate:
      "While alive and without a spawn shield, the Warrior gives off Guardian's Barrier. It protects living teammates within {{auraRadius}} who also have no spawn shield, including the Warrior itself. Each barrier reduces incoming damage by {{perEmitterDamageMitigationEffect}}. When barriers overlap, each reduces the damage left after the others, down to {{damageMitigationFloor}} of the original incoming damage.",
  }),
  3: definition({
    classId: 3,
    calling: "The Sentinel",
    order: "The Sentinels of the Lunewood",
    home: "The Lunewood, an ancient moonlit forest of a tall, long-lived night folk",
    fantasyLine:
      "A Sentinel of the moonlit Lunewood who sees first, strikes from farthest and seals an enemy in frost.",
    description:
      "The Hunter is a Sentinel of the Lunewood, an ancient moonlit forest whose night folk serve the Moon. It sees the farthest and its Basic reaches the farthest, but its arrows do the least damage and only slow their target a little. Its Ultimate, Freezing Trap, is thrown at one enemy and freezes it in place for the longest stun in the game. Any damage on a later tick breaks the ice early, so allies must choose: wait, or strike.",
    guideRole: "Crowd Control / Scout",
    tacticalGuideRows: [
      tacticalRow("Role", "Crowd Control / Scout"),
      tacticalRow("Primary Strength", "Longest stun duration."),
      tacticalRow(
        "Primary Weakness",
        "Lowest Basic damage among damage-dealing classes.",
      ),
    ],
    basicName: "Serrated Arrows",
    basicShortName: "Serrated Arrows",
    basicTemplate:
      "Serrated Arrows hit one enemy in reach. Each hit also slows that enemy, causing {{hunterBasicMovementEffect}} for {{hunterBasicSlowDuration}}. A new hit starts the slow again.",
    ultimateName: "Freezing Trap",
    ultimateTemplate:
      "Freezing Trap deals {{ultimateRawDamage}} base damage to one enemy and stuns it for {{trapStunDuration}}. Damage bonuses and reductions affect the damage dealt. Damage breaks an existing trap.",
    // Provisional name: "Serrated Arrows" became the Basic, so the passive
    // needs its own name. The owner has been asked to confirm or rename it.
    passiveName: "Far Sight",
    passiveTemplate:
      "Far Sight lets the Hunter see everything within {{observationRadius}} around it, farther than any other class. Stone still blocks its sight.",
  }),
  4: definition({
    classId: 4,
    calling: "The Shadowblade",
    order: "The Nightshade Office",
    home: "Galehaven, a living human kingdom; the Office serves its crown in secret",
    fantasyLine:
      "An agent of the Nightshade Office, the crown's secret service of spies: a whisper, a cut, and a wound that will not heal.",
    description:
      "The Rogue is an agent of the Nightshade Office, the secret service of spies and assassins who work for the crown of Galehaven. It moves faster than any other class, but it cannot turn invisible, blink or teleport: it wins with speed, stone cover and good timing. Its Ultimate, Crippling Poison, is the biggest single hit in the game; it also stuns, slows, and cuts the healing its target gets for a few ticks. The poison does no damage over time.",
    guideRole: "Assassin / Flanker",
    tacticalGuideRows: [
      tacticalRow("Role", "Assassin / Flanker"),
      tacticalRow("Primary Strength", "Highest base movement speed."),
      tacticalRow("Primary Weakness", "Short attack range with no gap closer."),
    ],
    basicName: "Backstab",
    basicShortName: "Backstab",
    basicTemplate:
      "Backstab hits one enemy close to the Rogue. Damage bonuses and reductions affect the damage dealt.",
    ultimateName: "Crippling Poison",
    ultimateTemplate:
      "Crippling Poison deals {{ultimateRawDamage}} base damage to one enemy, stuns it for {{poisonStunDuration}}, and applies {{poisonSlowEffect}} for {{poisonSlowDuration}}. It also applies {{poisonAntiHealEffect}} to healing and out-of-combat regeneration for {{poisonAntiHealDuration}}. Damage bonuses and reductions affect the damage dealt.",
    passiveName: "Phantom's Quickness",
    passiveTemplate:
      "Phantom's Quickness gives the Rogue a base movement speed of {{baseMovementSpeed}}, the highest in the game. After {{outOfCombatDelay}} out of combat, it regenerates health each tick at the rate shown under Out-of-Combat Regeneration.",
  }),
  5: definition({
    classId: 5,
    calling: "The Lightbringer",
    order: "The Church of the Morning Light",
    home: "Galehaven, where the Archbishop keeps the church's great cathedral",
    fantasyLine:
      "A priest of the Church of the Morning Light, sent to keep one candle lit through the night battle.",
    description:
      "The Priest is a Lightbringer of the Church of the Morning Light, sent by the Archbishop of Galehaven to keep a candle lit through the night battle. It is the only class that can heal, and it cannot deal damage at all, so its team must protect it. Its Basic, Lesser Heal, mends one ally or itself and briefly lets that ally shake off most of a strong slow (stuns still hold). Its Ultimate, Holy Word: Salvation, is a huge heal that can save an ally from a killing blow in the same tick, but it never brings back the fallen.",
    guideRole: "Healer / Medic",
    tacticalGuideRows: [
      tacticalRow("Role", "Healer / Medic"),
      tacticalRow("Primary Strength", "Only class with healing abilities."),
      tacticalRow("Primary Weakness", "Cannot deal damage."),
    ],
    basicName: "Lesser Heal",
    basicShortName: "Lesser Heal",
    basicTemplate:
      "Lesser Heal heals one ally in reach, or the Priest itself, and gives it Blessing of Freedom. Healing reductions and the maximum-health limit affect the healing received.",
    ultimateName: "Holy Word: Salvation",
    ultimateTemplate:
      "Holy Word: Salvation heals one ally for {{ultimateRawHealing}}, before healing reductions and the maximum-health limit.",
    passiveName: "Blessing of Freedom",
    passiveTemplate:
      "Each successful Lesser Heal applies Blessing of Freedom to its target for {{freedomDuration}}. The Priest can target itself. Freedom prevents slows from reducing movement below {{freedomMovementFloor}}, but it does not prevent stuns.",
  }),
});

/**
 * Return the shared definition for an integer class ID 1–5, or null.
 *
 * No profile check happens here; names and lore are public and contain no
 * numbers. Strings, other numbers and non-integers return null.
 *
 * @param {unknown} classId
 */
function definitionForClassId(classId) {
  return Number.isSafeInteger(classId) &&
    /** @type {number} */ (classId) >= 1 &&
    /** @type {number} */ (classId) <= 5
    ? CLASS_DOCUMENTATION[/** @type {1 | 2 | 3 | 4 | 5} */ (classId)]
    : null;
}

/**
 * Return the names of one class and its three abilities.
 *
 * classId must be an integer from 1 (Mage) to 5 (Priest); anything else
 * returns null. The result is a new frozen record:
 *
 * - className: "Mage" … "Priest" (from `agent-identity.js`).
 * - calling: for example "The Astral Walker". A symbol shows it as the
 *   subtitle "Mage (The Astral Walker)".
 * - basicName and basicShortName: for example "Arcane Blast" and "Blast".
 *   Only the Mage has a shorter form; the others repeat the full name.
 * - ultimateName and passiveName. The Mage's and Warrior's passives are
 *   auras, so their passive names are the approved status names of
 *   `symbols.js` STATUS_NAMES ("Sorcerer's Empowerment (Damage Boost)",
 *   "Guardian's Barrier (Damage Mitigation)"; tests/symbols.test.js keeps
 *   them equal).
 *
 * Names carry no numbers, so no documentation profile is needed. This never
 * reads a frame and grants no information about any agent.
 *
 * @param {unknown} classId
 * @returns {Readonly<{
 *   classId: 1 | 2 | 3 | 4 | 5,
 *   className: string,
 *   calling: string,
 *   basicName: string,
 *   basicShortName: string,
 *   ultimateName: string,
 *   passiveName: string,
 * }> | null}
 */
export function classNamesV1(classId) {
  const selected = definitionForClassId(classId);
  if (selected === null) return null;
  return Object.freeze({
    classId: selected.classId,
    className: selected.className,
    calling: selected.calling,
    basicName: selected.basicName,
    basicShortName: selected.basicShortName,
    ultimateName: selected.ultimateName,
    passiveName: selected.passiveName,
  });
}

/**
 * Return the name of one class's ability for one action lane.
 *
 * classId must be an integer 1–5. component must be "basic" (the Basic, for
 * example "Arcane Blast") or "ultimate" (for example "Burst"). Any other input
 * returns null. Used to write "from Hunter ID 7's Serrated Arrows".
 *
 * @param {unknown} classId
 * @param {unknown} component
 * @returns {string | null}
 */
export function classAbilityNameV1(classId, component) {
  const names = classNamesV1(classId);
  if (names === null) return null;
  if (component === "basic") return names.basicName;
  if (component === "ultimate") return names.ultimateName;
  return null;
}

/**
 * Return the lore of one class.
 *
 * classId must be an integer 1–5; anything else returns null. The result is a
 * new frozen record with calling, order (the class's original organization),
 * home, fantasyLine (one sentence), description (two to four plain sentences
 * for class guides and hover cards) and guideRole (short role words such as
 * "Artillery / Glass Cannon"). Lore carries no numbers and never names a team,
 * a team colour or a side of the screen.
 *
 * @param {unknown} classId
 * @returns {Readonly<{
 *   calling: string,
 *   order: string,
 *   home: string,
 *   fantasyLine: string,
 *   description: string,
 *   guideRole: string,
 * }> | null}
 */
export function classLoreV1(classId) {
  const selected = definitionForClassId(classId);
  if (selected === null) return null;
  return Object.freeze({
    calling: selected.calling,
    order: selected.order,
    home: selected.home,
    fantasyLine: selected.fantasyLine,
    description: selected.description,
    guideRole: selected.guideRole,
  });
}

/**
 * Copy an exact plain record of display strings, or return null.
 *
 * value must have exactly expectedKeys as own enumerable data properties,
 * with no symbol keys or getters. Each value must be a nonempty string with no
 * outer whitespace, line breaks or whole-word unavailable text. Read property
 * descriptors, not getters. Return a new null-prototype record (not frozen);
 * invalid input and inspection errors return null. No numeric parsing occurs.
 *
 * @param {unknown} value
 * @param {readonly string[]} expectedKeys
 * @returns {Readonly<Record<string, string>> | null}
 */
function exactStringRecord(value, expectedKeys) {
  try {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return null;
    const keys = Reflect.ownKeys(value);
    if (
      keys.length !== expectedKeys.length ||
      keys.some((key) => typeof key !== "string") ||
      expectedKeys.some((key) => !keys.includes(key))
    ) {
      return null;
    }
    const descriptors = Object.getOwnPropertyDescriptors(value);
    /** @type {Record<string, string>} */
    const snapshot = Object.create(null);
    for (const key of expectedKeys) {
      const descriptor = descriptors[key];
      if (!descriptor || !("value" in descriptor) || !descriptor.enumerable) {
        return null;
      }
      const item = descriptor.value;
      if (
        typeof item !== "string" ||
        item.length === 0 ||
        item.trim() !== item ||
        /[\r\n]/u.test(item) ||
        /\bunavailable\b/iu.test(item)
      ) {
        return null;
      }
      snapshot[key] = item;
    }
    return snapshot;
  } catch {
    return null;
  }
}

/**
 * Replace every named double-brace token in template with values[name].
 * The caller must already have checked that every required string exists.
 * Return a new string without changing either argument; this helper does not
 * validate, format numerical values or escape HTML.
 *
 * @param {string} template
 * @param {Readonly<Record<string, string>>} values
 * @returns {string}
 */
function interpolate(template, values) {
  return template.replace(TEMPLATE_TOKEN, (_token, name) => values[name]);
}

/**
 * Select the shared definition for a recognized profile and class ID.
 *
 * documentationProfile must contain exactly availability_kind=available and
 * the supported profile_id. classId must be a safe integer from 1 to 5. Return
 * the frozen definition or null; no fallback by class name or global slot is
 * allowed. This checks profile fields, not a separate cryptographic proof.
 *
 * @param {unknown} documentationProfile
 * @param {unknown} classId
 */
function certifiedDefinition(documentationProfile, classId) {
  const profile = exactStringRecord(documentationProfile, [
    "availability_kind",
    "profile_id",
  ]);
  if (
    profile?.availability_kind !== "available" ||
    profile.profile_id !== CANONICAL_PROFILE_ID
  ) {
    return null;
  }
  return definitionForClassId(classId);
}

/**
 * Return the ordered string names needed to render one class guide.
 *
 * documentationProfile and classId use certifiedDefinition's exact checks.
 * The names come from the Basic, Ultimate and passive templates, in that
 * order, each name once. Success returns the shared frozen key array; an
 * unknown profile or class returns null. No quantities are looked up and no
 * argument is changed.
 *
 * @param {unknown} documentationProfile
 * @param {unknown} classId
 * @returns {readonly string[] | null}
 */
export function requiredClassDocumentationValueNamesV1(documentationProfile, classId) {
  return certifiedDefinition(documentationProfile, classId)?.requiredValueNames ?? null;
}

/**
 * Render the supported class guide using exact authorized display strings.
 *
 * documentationProfile and classId select the definition. authorizedValues
 * must contain exactly that definition's named interpolation strings; missing,
 * extra, empty or unavailable values return null, as does an unknown profile.
 * Success returns a new frozen guide:
 *
 * - className, calling, order and fantasyLine: names and lore, no numbers.
 * - overview: the class description from the lore.
 * - tacticalGuideRows: shared frozen Role, Primary Strength and Primary
 *   Weakness rows.
 * - basic: {name, shortName, description}; ultimate and passive: {name,
 *   description}. Descriptions hold the supplied values in place of their
 *   tokens.
 *
 * It never invents numbers and never changes its inputs. The caller owns the
 * right to show the supplied facts.
 *
 * @param {unknown} documentationProfile
 * @param {unknown} classId
 * @param {unknown} authorizedValues
 * @returns {Readonly<{
 *   className: string,
 *   calling: string,
 *   order: string,
 *   fantasyLine: string,
 *   overview: string,
 *   tacticalGuideRows: readonly Readonly<{label: string, value: string}>[],
 *   basic: Readonly<{name: string, shortName: string, description: string}>,
 *   ultimate: Readonly<{name: string, description: string}>,
 *   passive: Readonly<{name: string, description: string}>,
 * }> | null}
 */
export function resolveClassDocumentationV1(
  documentationProfile,
  classId,
  authorizedValues,
) {
  const selected = certifiedDefinition(documentationProfile, classId);
  if (selected === null) return null;
  const values = exactStringRecord(authorizedValues, selected.requiredValueNames);
  if (values === null) return null;
  return Object.freeze({
    className: selected.className,
    calling: selected.calling,
    order: selected.order,
    fantasyLine: selected.fantasyLine,
    overview: selected.description,
    tacticalGuideRows: selected.tacticalGuideRows,
    basic: Object.freeze({
      name: selected.basicName,
      shortName: selected.basicShortName,
      description: interpolate(selected.basicTemplate, values),
    }),
    ultimate: Object.freeze({
      name: selected.ultimateName,
      description: interpolate(selected.ultimateTemplate, values),
    }),
    passive: Object.freeze({
      name: selected.passiveName,
      description: interpolate(selected.passiveTemplate, values),
    }),
  });
}
