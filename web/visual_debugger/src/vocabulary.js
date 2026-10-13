/**
 * @file Map recorded semantic IDs to browser labels, glyphs and CSS keys.
 *
 * This module owns visual identities: which glyph, CSS key and fallback letter
 * each class, team, status, activation, modifier and lifecycle token uses. It
 * does not own names. Class and team names come from `agent-identity.js`;
 * ability names from `class-documentation.js`; status, aura and "Freezing Trap
 * (Stun Broken)" names from `symbols.js` STATUS_NAMES. A status token's label
 * and short label are both its approved name ("Serrated Arrows (Slow)"), so
 * two causes of one effect never share a label.
 *
 * Definitions are display-only: this module does not calculate durations,
 * ranges, action legality or combat results, and it reads no frame, so it
 * cannot decide what Agent POV may show. Callers supply validated kinds and
 * authorized label text. Callers include `explanations.js`,
 * `source-attribution.js`, `scene.js`, `panels.js`, the choreography modules
 * and both frame normalizers.
 */
import {
  CLASS_IDENTITIES,
  classIdentityFromId,
  TEAM_IDENTITIES,
  teamIdentityFromId,
} from "./agent-identity.js";
import { classNamesV1 } from "./class-documentation.js";
import { statusNameFor } from "./symbols.js";

/**
 * @typedef {"class" | "team" | "status" | "activation" | "modifier" | "lifecycle"} VisualTokenKind
 */

/**
 * @typedef {{
 *   tokenId: string,
 *   label: string,
 *   shortLabel: string,
 *   accessibleName: string,
 *   glyphKey: string,
 *   cssKey: string,
 *   fallback: string,
 * }} VisualToken
 */

/**
 * @typedef {Readonly<Record<string, Readonly<VisualToken>>>} VisualTokenRegistry
 */

/**
 * @typedef {{
 *   label?: unknown,
 *   short_label?: unknown,
 *   shortLabel?: unknown,
 *   accessible_name?: unknown,
 *   accessibleName?: unknown,
 * }} TokenPayload
 */

/**
 * Freeze and return definition itself. It must already be a complete
 * VisualToken. This does not copy or validate fields, so the owning registry
 * must not expect to edit the object after this call.
 *
 * @param {VisualToken} definition
 * @returns {Readonly<VisualToken>}
 */
function token(definition) {
  return Object.freeze(definition);
}

/**
 * Build the five class tokens from the class table in `agent-identity.js`.
 *
 * Each class keeps its display name from that owner. This module adds only the
 * visual identity: glyph key "class-<id>", CSS key "<id>" and a one-letter
 * fallback (the first letter of the name). The short label is the full name.
 *
 * @returns {VisualTokenRegistry}
 */
function classTokens() {
  return Object.freeze(
    Object.fromEntries(
      Object.values(CLASS_IDENTITIES).map(({ tokenId, label }) => [
        tokenId,
        token({
          tokenId,
          label,
          shortLabel: label,
          accessibleName: `${label} class`,
          glyphKey: `class-${tokenId}`,
          cssKey: tokenId,
          fallback: label[0],
        }),
      ]),
    ),
  );
}

/**
 * Build the two team tokens from the team table in `agent-identity.js`.
 *
 * Labels are "Team A" and "Team B" only; the short label and fallback are the
 * team letter. Glyph and CSS keys are "team-a" and "team-b".
 *
 * @returns {VisualTokenRegistry}
 */
function teamTokens() {
  return Object.freeze(
    Object.fromEntries(
      Object.values(TEAM_IDENTITIES).map(({ tokenId, label, letter }) => [
        tokenId,
        token({
          tokenId,
          label,
          shortLabel: letter,
          accessibleName: `${label}, solid outline`,
          glyphKey: tokenId.replace("_", "-"),
          cssKey: tokenId.replace("_", "-"),
          fallback: letter,
        }),
      ]),
    ),
  );
}

const CLASS_TOKENS = classTokens();

const TEAM_TOKENS = teamTokens();

/**
 * Write where a status comes from, for one class and lane.
 *
 * classId is 1 to 5 and component "basic" or "ultimate". The result reads "a
 * Warrior's Charge"; with own true it reads "the Mage's own Burst", for a
 * status an agent gives itself. Names come from `class-documentation.js`; a
 * missing name throws at load time.
 *
 * @param {1 | 2 | 3 | 4 | 5} classId
 * @param {"basic" | "ultimate"} component
 * @param {boolean} [own] True for a status the agent gives itself. Default false.
 * @returns {string}
 * @throws {Error} When the class names are missing.
 */
function abilitySource(classId, component, own = false) {
  const names = classNamesV1(classId);
  if (names === null) throw new Error(`Missing class names for class ${classId}.`);
  const ability = component === "basic" ? names.basicName : names.ultimateName;
  return own
    ? `the ${names.className}'s own ${ability}`
    : `a ${names.className}'s ${ability}`;
}

/**
 * Build one status token from its approved name in `symbols.js` STATUS_NAMES.
 *
 * tokenId is the status display token. The label and the short label are the
 * approved name ("Serrated Arrows (Slow)"; the cause's name alone would be
 * ambiguous, as Charge gives both a stun and a slow) and the accessible name
 * is that name plus where it came from ("Serrated Arrows (Slow) from a
 * Hunter's Serrated Arrows"). glyphKey, cssKey and fallback are the visual
 * identity. A missing name throws at load time.
 *
 * @param {string} tokenId
 * @param {string | null} from The source phrase, or null for none.
 * @param {string} glyphKey
 * @param {string} cssKey
 * @param {string} fallback
 * @returns {Readonly<VisualToken>}
 * @throws {Error} When tokenId has no approved name.
 */
function statusToken(tokenId, from, glyphKey, cssKey, fallback) {
  const named = statusNameFor(tokenId);
  if (named === null) throw new Error(`Status ${tokenId} has no approved name.`);
  return token({
    tokenId,
    label: named.title,
    shortLabel: named.title,
    accessibleName: from === null ? named.title : `${named.title} from ${from}`,
    glyphKey,
    cssKey,
    fallback,
  });
}

const STATUS_TOKENS = Object.freeze({
  in_combat: statusToken("in_combat", null, "combat-in-progress", "in-combat", "C"),
  stun_warrior_charge: statusToken(
    "stun_warrior_charge",
    abilitySource(2, "ultimate"),
    "status-stun",
    "stun-warrior-charge",
    "CS",
  ),
  stun_hunter_trap: statusToken(
    "stun_hunter_trap",
    abilitySource(3, "ultimate"),
    "status-stun",
    "stun-hunter-trap",
    "T",
  ),
  stun_rogue_poison: statusToken(
    "stun_rogue_poison",
    abilitySource(4, "ultimate"),
    "status-stun",
    "stun-rogue-poison",
    "PS",
  ),
  slow_warrior_charge: statusToken(
    "slow_warrior_charge",
    abilitySource(2, "ultimate"),
    "status-slow",
    "slow-warrior-charge",
    "CS",
  ),
  slow_hunter_basic: statusToken(
    "slow_hunter_basic",
    abilitySource(3, "basic"),
    "status-slow",
    "slow-hunter-basic",
    "HS",
  ),
  slow_rogue_poison: statusToken(
    "slow_rogue_poison",
    abilitySource(4, "ultimate"),
    "status-slow",
    "slow-rogue-poison",
    "PS",
  ),
  anti_heal_rogue_poison: statusToken(
    "anti_heal_rogue_poison",
    abilitySource(4, "ultimate"),
    "status-anti-heal",
    "anti-heal-rogue-poison",
    "AH",
  ),
  priest_freedom: statusToken(
    "priest_freedom",
    abilitySource(5, "basic"),
    "status-freedom",
    "priest-freedom",
    "F",
  ),
  mage_burst: statusToken(
    "mage_burst",
    abilitySource(1, "ultimate", true),
    "status-burst",
    "mage-burst",
    "B",
  ),
});

/**
 * Return the name of one class's Ultimate from `class-documentation.js`.
 *
 * @param {1 | 2 | 3 | 4 | 5} classId
 * @returns {string}
 * @throws {Error} When the class names are missing.
 */
function ultimateName(classId) {
  const names = classNamesV1(classId);
  if (names === null) throw new Error(`Missing class names for class ${classId}.`);
  return names.ultimateName;
}

const ACTIVATION_TOKENS = Object.freeze({
  basic_damage: token({
    tokenId: "basic_damage",
    label: "Basic Damage",
    shortLabel: "Basic",
    accessibleName: "Basic damage",
    glyphKey: "activation-basic-damage",
    cssKey: "basic-damage",
    fallback: "B",
  }),
  basic_heal: token({
    tokenId: "basic_heal",
    label: classNamesV1(5)?.basicName ?? "Basic Healing",
    shortLabel: "Basic",
    accessibleName: `Priest's ${classNamesV1(5)?.basicName ?? "Basic Healing"}`,
    glyphKey: "activation-basic-heal",
    cssKey: "basic-heal",
    fallback: "B",
  }),
  holy_word: token({
    tokenId: "holy_word",
    label: ultimateName(5),
    shortLabel: "Salvation",
    accessibleName: `Priest's ${ultimateName(5)}`,
    glyphKey: "activation-holy-word",
    cssKey: "holy-word",
    fallback: "U",
  }),
  mage_burst: token({
    tokenId: "mage_burst",
    label: ultimateName(1),
    shortLabel: ultimateName(1),
    accessibleName: `Mage's ${ultimateName(1)}`,
    glyphKey: "activation-burst",
    cssKey: "mage-burst",
    fallback: "U",
  }),
  warrior_charge: token({
    tokenId: "warrior_charge",
    label: ultimateName(2),
    shortLabel: ultimateName(2),
    accessibleName: `Warrior's ${ultimateName(2)}`,
    glyphKey: "activation-charge",
    cssKey: "warrior-charge",
    fallback: "U",
  }),
  hunter_trap: token({
    tokenId: "hunter_trap",
    label: ultimateName(3),
    shortLabel: ultimateName(3),
    accessibleName: `Hunter's ${ultimateName(3)}`,
    glyphKey: "activation-trap",
    cssKey: "hunter-trap",
    fallback: "U",
  }),
  rogue_poison: token({
    tokenId: "rogue_poison",
    label: ultimateName(4),
    shortLabel: ultimateName(4),
    accessibleName: `Rogue's ${ultimateName(4)}`,
    glyphKey: "activation-poison",
    cssKey: "rogue-poison",
    fallback: "U",
  }),
});

/**
 * Build one aura or status modifier token from its approved name in
 * `symbols.js` STATUS_NAMES.
 *
 * tokenId is the modifier display token. The label is the approved name
 * ("Sorcerer's Empowerment (Damage Boost)"), the short label the cause's name
 * ("Sorcerer's Empowerment", shown on chips as "Sorcerer's Empowerment
 * ×1.15") and the accessible name the label plus what the number changes. A
 * missing name throws at load time.
 *
 * @param {string} tokenId
 * @param {string} changes What the modifier's number changes, for example "damage dealt".
 * @param {string} glyphKey
 * @param {string} cssKey
 * @param {string} fallback
 * @returns {Readonly<VisualToken>}
 * @throws {Error} When tokenId has no symbol.
 */
function modifierToken(tokenId, changes, glyphKey, cssKey, fallback) {
  const named = statusNameFor(tokenId);
  if (named === null) throw new Error(`Modifier ${tokenId} has no approved name.`);
  return token({
    tokenId,
    label: named.title,
    shortLabel: named.name,
    accessibleName: `${named.title} modifier on ${changes}`,
    glyphKey,
    cssKey,
    fallback,
  });
}

const MODIFIER_TOKENS = Object.freeze({
  mage_amplification: modifierToken(
    "mage_amplification",
    "damage dealt",
    "modifier-amplification",
    "mage-amplification",
    "AMP",
  ),
  warrior_mitigation: modifierToken(
    "warrior_mitigation",
    "damage received",
    "modifier-mitigation",
    "warrior-mitigation",
    "MIT",
  ),
  rogue_anti_heal: modifierToken(
    "rogue_anti_heal",
    "healing received",
    "status-anti-heal",
    "rogue-anti-heal",
    "AH",
  ),
  priest_freedom: modifierToken(
    "priest_freedom",
    "the lowest movement speed",
    "status-freedom",
    "priest-freedom",
    "F",
  ),
  mage_burst: modifierToken(
    "mage_burst",
    "damage dealt",
    "status-burst",
    "mage-burst",
    "B",
  ),
});

/** The "Freezing Trap (Stun Broken)" name, read once for the trap lifecycle tokens. */
const STUN_BROKEN = statusNameFor("trap_broken");

const LIFECYCLE_TOKENS = Object.freeze({
  applied: token({
    tokenId: "applied",
    label: "Applied",
    shortLabel: "Applied",
    accessibleName: "Status applied",
    glyphKey: "lifecycle-applied",
    cssKey: "applied",
    fallback: "+",
  }),
  refreshed: token({
    tokenId: "refreshed",
    label: "Refreshed",
    shortLabel: "Refreshed",
    accessibleName: "Status refreshed or extended",
    glyphKey: "lifecycle-refreshed",
    cssKey: "refreshed",
    fallback: "R",
  }),
  decremented: token({
    tokenId: "decremented",
    label: "Aged",
    shortLabel: "Age",
    accessibleName: "Status has one tick less left",
    glyphKey: "lifecycle-decremented",
    cssKey: "decremented",
    fallback: "-",
  }),
  expired: token({
    tokenId: "expired",
    label: "Expired",
    shortLabel: "Expired",
    accessibleName: "Status expired naturally",
    glyphKey: "lifecycle-expired",
    cssKey: "expired",
    fallback: "E",
  }),
  trap_broken: token({
    tokenId: "trap_broken",
    label: STUN_BROKEN?.title ?? "Freezing Trap (Stun Broken)",
    shortLabel: STUN_BROKEN?.effect ?? "Stun Broken",
    accessibleName: `${STUN_BROKEN?.title ?? "Freezing Trap (Stun Broken)"}: damage ended the Freezing Trap stun early`,
    glyphKey: "lifecycle-trap-broken",
    cssKey: "trap-broken",
    fallback: "X",
  }),
  cleared_by_death: token({
    tokenId: "cleared_by_death",
    label: "Cleared on Death",
    shortLabel: "Death Clear",
    accessibleName: "Status cleared because the champion fell",
    glyphKey: "lifecycle-death-clear",
    cssKey: "cleared-by-death",
    fallback: "D",
  }),
  cleared_unclassified: token({
    tokenId: "cleared_unclassified",
    label: "Status Ended",
    shortLabel: "End",
    accessibleName: "Status ended; the record does not say why",
    glyphKey: "lifecycle-ended",
    cssKey: "cleared-unclassified",
    fallback: "?",
  }),
  trap_broken_and_reapplied: token({
    tokenId: "trap_broken_and_reapplied",
    label: `${STUN_BROKEN?.title ?? "Freezing Trap (Stun Broken)"}, Then Reapplied`,
    shortLabel: "Break+",
    accessibleName: "Freezing Trap stun was broken by damage, then applied again",
    glyphKey: "lifecycle-trap-broken-reapplied",
    cssKey: "trap-broken-and-reapplied",
    fallback: "X+",
  }),
  reapplied: token({
    tokenId: "reapplied",
    label: "Reapplied",
    shortLabel: "Reapplied",
    accessibleName: "Status reapplied",
    glyphKey: "lifecycle-applied",
    cssKey: "reapplied",
    fallback: "+",
  }),
});

/**
 * CP2 catalog IDs are scientific wire identity; status tokens are stable
 * presentation vocabulary. Keep the translation explicit and version-bound
 * rather than relying on word order or string rewriting.
 */
/** @type {Readonly<Record<string, string>>} */
const STATUS_TOKEN_BY_CATALOG_ID = Object.freeze({
  warrior_charge_slow: "slow_warrior_charge",
  hunter_basic_slow: "slow_hunter_basic",
  rogue_poison_slow: "slow_rogue_poison",
  warrior_charge_stun: "stun_warrior_charge",
  hunter_trap_stun: "stun_hunter_trap",
  rogue_poison_stun: "stun_rogue_poison",
  rogue_poison_anti_heal: "anti_heal_rogue_poison",
  mage_burst_damage_amplification: "mage_burst",
  priest_blessing_of_freedom_movement_floor: "priest_freedom",
});

/**
 * Authorized aura IDs retain their catalog identity on the wire. Translate
 * only at the display-vocabulary boundary, just like catalog status IDs.
 */
/** @type {Readonly<Record<string, string>>} */
const MODIFIER_TOKEN_BY_CATALOG_AURA_ID = Object.freeze({
  mage_damage_amplification: "mage_amplification",
  warrior_damage_mitigation: "warrior_mitigation",
});

/**
 * The five class token IDs in class-ID order: mage, warrior, hunter, rogue,
 * priest. Read from `agent-identity.js`.
 *
 * @type {readonly string[]}
 */
export const CLASS_TOKEN_IDS = Object.freeze(
  Object.values(CLASS_IDENTITIES).map(({ tokenId }) => tokenId),
);

/**
 * The two team token IDs: team_a, team_b. Read from `agent-identity.js`.
 *
 * @type {readonly string[]}
 */
export const TEAM_TOKEN_IDS = Object.freeze(
  Object.values(TEAM_IDENTITIES).map(({ tokenId }) => tokenId),
);

export const CANONICAL_STATUS_ORDER = Object.freeze([
  "stun_warrior_charge",
  "stun_hunter_trap",
  "stun_rogue_poison",
  "slow_warrior_charge",
  "slow_hunter_basic",
  "slow_rogue_poison",
  "anti_heal_rogue_poison",
  "priest_freedom",
  "mage_burst",
]);

export const ACTIVATION_TOKEN_IDS = Object.freeze([
  "basic_damage",
  "basic_heal",
  "holy_word",
  "mage_burst",
  "warrior_charge",
  "hunter_trap",
  "rogue_poison",
]);

export const MODIFIER_TOKEN_IDS = Object.freeze([
  "mage_amplification",
  "warrior_mitigation",
  "rogue_anti_heal",
  "priest_freedom",
  "mage_burst",
]);

export const LIFECYCLE_TOKEN_IDS = Object.freeze([
  "applied",
  "refreshed",
  "decremented",
  "expired",
  "trap_broken",
  "cleared_by_death",
  "cleared_unclassified",
  "trap_broken_and_reapplied",
  "reapplied",
]);

export const CATALOG_STATUS_IDS = Object.freeze(
  Object.keys(STATUS_TOKEN_BY_CATALOG_ID),
);

/** @type {Readonly<Record<VisualTokenKind, VisualTokenRegistry>>} */
const TOKEN_REGISTRIES = Object.freeze({
  class: CLASS_TOKENS,
  team: TEAM_TOKENS,
  status: STATUS_TOKENS,
  activation: ACTIVATION_TOKENS,
  modifier: MODIFIER_TOKENS,
  lifecycle: LIFECYCLE_TOKENS,
});

/** @type {Readonly<Record<number, string>>} */
const CLASS_ID_TO_ULTIMATE_TOKEN = Object.freeze({
  1: "mage_burst",
  2: "warrior_charge",
  3: "hunter_trap",
  4: "rogue_poison",
  5: "holy_word",
});

/** @type {Readonly<Record<string, "damage" | "healing" | "neutral" | "local">>} */
const ACTIVATION_IMPACT_SEMANTICS = Object.freeze({
  basic_damage: "damage",
  basic_heal: "healing",
  holy_word: "healing",
  mage_burst: "local",
  warrior_charge: "damage",
  hunter_trap: "damage",
  rogue_poison: "damage",
});

/**
 * Return trimmed nonempty string value, or null for another/empty value.
 * No length, markup or authority check is performed; callers use the result
 * as text within already authorized presentation data.
 *
 * @param {unknown} value
 * @returns {string | null}
 */
function displayString(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/**
 * Return an object payload as-is, or {} for null/nonobject input. Arrays
 * also pass this shallow check. No copy, freeze or accessor validation occurs;
 * callers provide trusted/normalized display data.
 *
 * @param {unknown} payload
 * @returns {TokenPayload}
 */
function tokenPayload(payload) {
  return typeof payload === "object" && payload !== null
    ? /** @type {TokenPayload} */ (payload)
    : {};
}

/**
 * Return tokenId as a trimmed nonempty string, or unknown otherwise.
 * This does not check registry membership or change the case of an ID.
 *
 * @param {unknown} tokenId
 * @returns {string}
 */
function normalizedTokenId(tokenId) {
  return displayString(tokenId) ?? "unknown";
}

/**
 * Create a frozen neutral visual token retaining tokenId and naming kind
 * in its accessible text. Use unknown glyph/CSS keys and question-mark labels.
 * Both inputs are caller-owned display identifiers; no scientific meaning is
 * inferred from them.
 *
 * @param {VisualTokenKind} kind
 * @param {string} tokenId
 * @returns {Readonly<VisualToken>}
 */
function unknownToken(kind, tokenId) {
  return token({
    tokenId,
    label: "Unknown",
    shortLabel: "?",
    accessibleName: `Unknown ${kind} visual token ${tokenId}`,
    glyphKey: "unknown",
    cssKey: "unknown",
    fallback: "?",
  });
}

/**
 * Apply optional payload labels to a frozen definition without changing
 * its glyph, CSS or token identity. Use label, then snake-case short_label/
 * accessible_name before camel-case alternatives; invalid/empty strings fall
 * back to the definition. Return the original for no change, or a newly
 * frozen shallow copy. This does not validate payload authority or escape HTML.
 *
 * @param {Readonly<VisualToken>} definition
 * @param {unknown} payload
 * @returns {Readonly<VisualToken>}
 */
function withPayloadProse(definition, payload) {
  const record = tokenPayload(payload);
  const label = displayString(record.label) ?? definition.label;
  const shortLabel =
    displayString(record.short_label) ??
    displayString(record.shortLabel) ??
    definition.shortLabel;
  const accessibleName =
    displayString(record.accessible_name) ??
    displayString(record.accessibleName) ??
    definition.accessibleName;

  if (
    label === definition.label &&
    shortLabel === definition.shortLabel &&
    accessibleName === definition.accessibleName
  ) {
    return definition;
  }
  return token({
    ...definition,
    label,
    shortLabel,
    accessibleName,
  });
}

/**
 * Resolve tokenId in the registry for kind and apply optional payload prose.
 *
 * kind must be class, team, status, activation, modifier or lifecycle; an
 * unsupported kind is not handled and may throw TypeError. Normalize tokenId
 * and apply declared status/aura aliases. Unlisted IDs receive a neutral
 * unknown definition. payload is optional and may change labels only, never
 * glyph/CSS selection. Return a frozen token, possibly shared. No input
 * mutation or simulator lookup occurs.
 *
 * @param {VisualTokenKind} kind
 * @param {unknown} tokenId
 * @param {unknown} [payload]
 * @returns {Readonly<VisualToken>}
 */
export function resolveVisualToken(kind, tokenId, payload) {
  const wireId = normalizedTokenId(tokenId);
  const normalized =
    kind === "status" && Object.hasOwn(STATUS_TOKEN_BY_CATALOG_ID, wireId)
      ? STATUS_TOKEN_BY_CATALOG_ID[wireId]
      : kind === "modifier" && Object.hasOwn(MODIFIER_TOKEN_BY_CATALOG_AURA_ID, wireId)
        ? MODIFIER_TOKEN_BY_CATALOG_AURA_ID[wireId]
        : wireId;
  const registry = TOKEN_REGISTRIES[kind];
  const definition = Object.hasOwn(registry, normalized)
    ? registry[normalized]
    : unknownToken(kind, normalized);
  return withPayloadProse(definition, payload);
}

/**
 * Map normalized statusId through the declared catalog-status aliases.
 * Return the unchanged normalized ID when unlisted, or unknown for invalid/
 * empty input. This preserves future IDs for the ordinary unknown-token view
 * rather than inferring a status from a class or slot.
 *
 * @param {unknown} statusId
 */
export function statusTokenIdFromCatalogId(statusId) {
  const normalized = normalizedTokenId(statusId);
  return Object.hasOwn(STATUS_TOKEN_BY_CATALOG_ID, normalized)
    ? STATUS_TOKEN_BY_CATALOG_ID[normalized]
    : normalized;
}

/**
 * Resolve integer classId 1–5 to its display token, with optional payload
 * labels. Other integers retain a class_id_N unknown identity; nonintegers
 * use unknown. Return the frozen token from resolveVisualToken without
 * copying class mechanics or granting access to an agent.
 *
 * @param {unknown} classId
 * @param {unknown} [payload]
 * @returns {Readonly<VisualToken>}
 */
export function classTokenFromId(classId, payload) {
  const numericId =
    typeof classId === "number" && Number.isInteger(classId) ? classId : null;
  const tokenId =
    numericId === null ? undefined : classIdentityFromId(numericId)?.tokenId;
  return resolveVisualToken(
    "class",
    tokenId ?? (numericId === null ? "unknown" : `class_id_${numericId}`),
    payload,
  );
}

/**
 * Resolve a class's Ultimate display token for cooldown/activation cues.
 * Integer classId 1–5 selects the declared Ultimate; other integers retain an
 * ultimate_for_class_id_N unknown identity and nonintegers use unknown.
 * Optional payload affects labels only. Return a frozen display token; no
 * ability availability or damage is calculated.
 *
 * @param {unknown} classId
 * @param {unknown} [payload]
 * @returns {Readonly<VisualToken>}
 */
export function ultimateTokenFromClassId(classId, payload) {
  const numericId =
    typeof classId === "number" && Number.isInteger(classId) ? classId : null;
  const tokenId =
    numericId === null ? undefined : CLASS_ID_TO_ULTIMATE_TOKEN[numericId];
  return resolveVisualToken(
    "activation",
    tokenId ?? (numericId === null ? "unknown" : `ultimate_for_class_id_${numericId}`),
    payload,
  );
}

/**
 * Look up the visual impact category for a normalized activation tokenId.
 * Known IDs return damage, healing or local; an absent lookup returns neutral.
 * This classifies an already validated activation, not its damage amount or
 * recipient attribution. The current direct object lookup does not guard
 * inherited property names such as constructor; callers must not pass those
 * as arbitrary unknown IDs. This helper does not validate raw wire input.
 *
 * @param {unknown} tokenId
 * @returns {"damage" | "healing" | "neutral" | "local"}
 */
export function activationImpactSemantic(tokenId) {
  const normalized = normalizedTokenId(tokenId);
  return ACTIVATION_IMPACT_SEMANTICS[normalized] ?? "neutral";
}

/**
 * Resolve integer teamId 1 or 2 to Team A/B display metadata. Other integers
 * retain a team_id_N unknown identity; nonintegers use unknown. Optional
 * payload may override display labels. Return a frozen token without changing
 * team ownership or copying simulator rules.
 *
 * @param {unknown} teamId
 * @param {unknown} [payload]
 * @returns {Readonly<VisualToken>}
 */
export function teamTokenFromId(teamId, payload) {
  const numericId =
    typeof teamId === "number" && Number.isInteger(teamId) ? teamId : null;
  const tokenId =
    numericId === null ? undefined : teamIdentityFromId(numericId)?.tokenId;
  return resolveVisualToken(
    "team",
    tokenId ?? (numericId === null ? "unknown" : `team_id_${numericId}`),
    payload,
  );
}
