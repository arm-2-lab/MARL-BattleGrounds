/**
 * @file Supply status and aura words for browser explanations.
 *
 * This module owns the plain-English effect text of each status and aura, the
 * status lifecycle titles ("Applied Serrated Arrows (Slow)"), the Champion
 * Card's status sentence (statusCardSentence): the plain sentence from
 * `symbols.js` STATUS_NAMES with the exact number written in, then where the
 * status comes from by class ("It moves at ×0.5 speed." and "From a
 * Warrior."; an aura on a champion of its own class: "From its own aura."),
 * its time words (statusTimeText: "5 ticks left", "This tick", "Now", "While
 * near") and the one-line status rule (statusLineV1), which is the card row in
 * one line:
 *
 *   "Serrated Arrows (Slow) · 1 tick left · It moves at ×0.85 speed. From a Hunter."
 *
 * Every status display uses these words (owner rulings, 2 October 2026): the
 * card, Frame Explanation's status items, the agent details panel, the Replay
 * Viewer's Champion Frame and the status tips. Sources are named by class
 * only, never by Agent ID and never "a hidden source", so the words are the
 * same in Agent POV and Oracle View.
 *
 * Status and aura names (the cause's name, then the plain effect in brackets)
 * come from `symbols.js` STATUS_NAMES, and ability names from
 * `class-documentation.js`, so each name has one owner. The frozen
 * definitions here hold no tuning numbers: callers pass exact durations,
 * radii and effect values from their authorized presentation.
 *
 * Callers: `explanations.js`, `source-attribution.js` (the applying class of
 * each status), the choreography painter, the Champion Card (`tooltip.js`),
 * Frame Explanation and the HUD and panel modules. Unknown tokens receive
 * neutral labels or no lifecycle result.
 */
import { tickCountText } from "./agent-identity.js";
import { classAbilityNameV1, classNamesV1 } from "./class-documentation.js";
import { formatDisplayNumber } from "./display.js";
import { statusNameFor } from "./symbols.js";

/** @typedef {"mage" | "warrior" | "hunter" | "rogue" | "priest" | "none"} StatusAccent */
/** @typedef {"none" | "movement_multiplier" | "healing_multiplier" | "damage_multiplier" | "movement_floor"} StatusMagnitudeKind */

/**
 * One status's display definition.
 *
 * - title: name plus subtitle, for example "Serrated Arrows (Slow)".
 * - name: the cause's name, for example "Serrated Arrows".
 * - subtitle: the plain effect, for example "Slow", or null (In Combat has
 *   none).
 * - effect: plain-English effect text with no numbers.
 * - accent: the class colour of the source, or "none".
 * - magnitudeKind: which kind of number the caller may show beside it.
 * - positiveDamageBreak: true when damage can end it early.
 * - sourceClassId and sourceComponent: the class (1 to 5) and the action lane
 *   ("basic" or "ultimate") whose ability applies it, or null for statuses
 *   with no source (Spawn Shield, In Combat, unknown).
 *
 * @typedef {Readonly<{
 *   title: string,
 *   name: string,
 *   subtitle: string | null,
 *   effect: string,
 *   accent: StatusAccent,
 *   magnitudeKind: StatusMagnitudeKind,
 *   positiveDamageBreak: boolean,
 *   sourceClassId: 1 | 2 | 3 | 4 | 5 | null,
 *   sourceComponent: "basic" | "ultimate" | null,
 * }>} StatusPresentation
 */

/**
 * Write "a Warrior's Charge" for a class ID and action lane.
 *
 * classId is 1 to 5 and component is "basic" or "ultimate". The names come
 * from `class-documentation.js`. A missing name throws at load time, which
 * only a broken edit of that module can cause.
 *
 * @param {1 | 2 | 3 | 4 | 5} classId
 * @param {"basic" | "ultimate"} component
 * @returns {string}
 * @throws {Error} When the class or ability name is missing.
 */
function abilityPhrase(classId, component) {
  const className = classNamesV1(classId)?.className;
  const ability = classAbilityNameV1(classId, component);
  if (className === undefined || ability === null) {
    throw new Error(`Missing ability name for class ${classId} ${component}.`);
  }
  return `a ${className}'s ${ability}`;
}

/**
 * Freeze one status display definition.
 *
 * tokenId selects the approved status name (`symbols.js` statusNameFor) the
 * status shows; a missing name throws at load time. effect is authored text
 * with no numbers.
 * accent picks the source class colour and magnitudeKind the kind of number a
 * caller may show. source names the applying class and lane, or null.
 * positiveDamageBreak defaults to false and marks effects that damage can end.
 *
 * @param {string} tokenId
 * @param {string} effect
 * @param {StatusAccent} accent
 * @param {StatusMagnitudeKind} magnitudeKind
 * @param {Readonly<{classId: 1 | 2 | 3 | 4 | 5, component: "basic" | "ultimate"}> | null} source
 * @param {boolean} [positiveDamageBreak]
 * @returns {StatusPresentation}
 * @throws {Error} When tokenId has no approved status name.
 */
function statusProfile(
  tokenId,
  effect,
  accent,
  magnitudeKind,
  source,
  positiveDamageBreak = false,
) {
  const named = statusNameFor(tokenId);
  if (named === null) throw new Error(`Status ${tokenId} has no approved name.`);
  return Object.freeze({
    title: named.title,
    name: named.name,
    subtitle: named.effect,
    effect,
    accent,
    magnitudeKind,
    positiveDamageBreak,
    sourceClassId: source?.classId ?? null,
    sourceComponent: source?.component ?? null,
  });
}

/** The shared end of every stun effect sentence, after "A Warrior's Charge". */
const STUN_EFFECT =
  "stops this agent from moving or acting until the stun ends. Collisions can still push its body.";

/**
 * Every known status definition by display token ID. Frozen and shared.
 *
 * @type {Readonly<Record<string, StatusPresentation>>}
 */
const STATUS_PRESENTATION = Object.freeze({
  spawn_shield: statusProfile(
    "spawn_shield",
    "While the Spawn Shield lasts, this agent cannot be hurt, enemies cannot see or target it, auras skip it, and it can only move. It can move through other agents; it collides with them again at the end of the shield's last tick.",
    "none",
    "none",
    null,
  ),
  in_combat: statusProfile(
    "in_combat",
    "Shows how many ticks remain before this agent leaves combat. Taking part in combat starts the count again.",
    "none",
    "none",
    null,
  ),
  stun_warrior_charge: statusProfile(
    "stun_warrior_charge",
    `${capitalize(abilityPhrase(2, "ultimate"))} ${STUN_EFFECT}`,
    "warrior",
    "none",
    { classId: 2, component: "ultimate" },
  ),
  stun_hunter_trap: statusProfile(
    "stun_hunter_trap",
    `${capitalize(abilityPhrase(3, "ultimate"))} ${STUN_EFFECT} Damage on a later tick breaks it early.`,
    "hunter",
    "none",
    { classId: 3, component: "ultimate" },
    true,
  ),
  stun_rogue_poison: statusProfile(
    "stun_rogue_poison",
    `${capitalize(abilityPhrase(4, "ultimate"))} ${STUN_EFFECT}`,
    "rogue",
    "none",
    { classId: 4, component: "ultimate" },
  ),
  slow_warrior_charge: statusProfile(
    "slow_warrior_charge",
    `${capitalize(abilityPhrase(2, "ultimate"))} slows this agent's movement until the slow ends.`,
    "warrior",
    "movement_multiplier",
    { classId: 2, component: "ultimate" },
  ),
  slow_hunter_basic: statusProfile(
    "slow_hunter_basic",
    `${capitalize(abilityPhrase(3, "basic"))} slow this agent's movement until the slow ends. A new hit starts it again.`,
    "hunter",
    "movement_multiplier",
    { classId: 3, component: "basic" },
  ),
  slow_rogue_poison: statusProfile(
    "slow_rogue_poison",
    `${capitalize(abilityPhrase(4, "ultimate"))} slows this agent's movement until the slow ends.`,
    "rogue",
    "movement_multiplier",
    { classId: 4, component: "ultimate" },
  ),
  anti_heal_rogue_poison: statusProfile(
    "anti_heal_rogue_poison",
    `${capitalize(abilityPhrase(4, "ultimate"))} cuts the healing this agent receives, and its out-of-combat regeneration, until it ends.`,
    "rogue",
    "healing_multiplier",
    { classId: 4, component: "ultimate" },
  ),
  priest_freedom: statusProfile(
    "priest_freedom",
    `${capitalize(abilityPhrase(5, "basic"))} on a teammate, or on the Priest itself, gives Blessing of Freedom: until it ends, slows cannot pull this agent's movement below a set floor. It does not stop a stun.`,
    "priest",
    "movement_floor",
    { classId: 5, component: "basic" },
  ),
  mage_burst: statusProfile(
    "mage_burst",
    "This Mage's own Burst makes its outgoing damage higher until it ends.",
    "mage",
    "damage_multiplier",
    { classId: 1, component: "ultimate" },
  ),
});

/**
 * Return text with its first letter in upper case. Used to start a sentence
 * with "A Warrior's Charge". Empty text is returned unchanged.
 *
 * @param {string} text
 * @returns {string}
 */
function capitalize(text) {
  return text.length === 0 ? text : `${text[0].toUpperCase()}${text.slice(1)}`;
}

/**
 * One aura's display definition for its field (the circle on the ground) and
 * for an agent inside it.
 *
 * @typedef {Readonly<{
 *   fieldTitle: string,
 *   recipientTitle: string,
 *   fieldEffect: string,
 *   aggregateEffect: string,
 *   fieldEffectLabel: string,
 *   aggregateEffectLabel: string,
 *   accent: "mage" | "warrior" | "none",
 *   effectKind: "damage_dealt" | "damage_received" | "generic",
 * }>} AuraPresentation
 */

/**
 * Freeze the words for one aura definition.
 *
 * modifierTokenId selects the approved name ("Sorcerer's Empowerment (Damage
 * Boost)", from `symbols.js` statusNameFor) that serves both the field and
 * recipient cards; a missing name throws at load time. fieldEffect and aggregateEffect describe the two views;
 * fieldEffectLabel and aggregateEffectLabel name their number rows. accent
 * selects Mage or Warrior and effectKind tells damage dealt from damage
 * received. Return a frozen record without any number.
 *
 * @param {string} modifierTokenId
 * @param {string} fieldEffect
 * @param {string} aggregateEffect
 * @param {string} fieldEffectLabel
 * @param {string} aggregateEffectLabel
 * @param {"mage" | "warrior"} accent
 * @param {"damage_dealt" | "damage_received"} effectKind
 * @returns {AuraPresentation}
 * @throws {Error} When modifierTokenId has no approved name.
 */
function auraProfile(
  modifierTokenId,
  fieldEffect,
  aggregateEffect,
  fieldEffectLabel,
  aggregateEffectLabel,
  accent,
  effectKind,
) {
  const named = statusNameFor(modifierTokenId);
  if (named === null) throw new Error(`Aura ${modifierTokenId} has no approved name.`);
  return Object.freeze({
    fieldTitle: named.title,
    recipientTitle: named.title,
    fieldEffect,
    aggregateEffect,
    fieldEffectLabel,
    aggregateEffectLabel,
    accent,
    effectKind,
  });
}

/**
 * The two aura definitions by host aura ID. Frozen and shared.
 *
 * @type {Readonly<Record<string, AuraPresentation>>}
 */
const AURA_PRESENTATION = Object.freeze({
  mage_damage_amplification: auraProfile(
    "mage_amplification",
    "This Mage's aura makes the Mage itself, and teammates in its reach without a spawn shield, deal more damage.",
    "This agent is within reach of at least one Mage's Sorcerer's Empowerment.",
    "Damage Bonus",
    "Total Damage Bonus",
    "mage",
    "damage_dealt",
  ),
  warrior_damage_mitigation: auraProfile(
    "warrior_mitigation",
    "This Warrior's aura makes the Warrior itself, and teammates in its reach without a spawn shield, take less damage.",
    "This agent is within reach of at least one Warrior's Guardian's Barrier.",
    "Damage Reduction",
    "Total Damage Reduction",
    "warrior",
    "damage_received",
  ),
});

/** Modifier token IDs accepted in place of the host aura IDs. */
const AURA_ID_ALIASES = Object.freeze({
  mage_amplification: "mage_damage_amplification",
  warrior_mitigation: "warrior_damage_mitigation",
});

/** The word put before a status title for each lifecycle ("Applied …"). */
const STATUS_LIFECYCLE_PREFIX = Object.freeze({
  applied: "Applied",
  refreshed: "Refreshed",
  decremented: "Aged",
  expired: "Expired",
  trap_broken: "Broken",
  cleared_by_death: "Cleared On Death",
  cleared_unclassified: "Ended",
  trap_broken_and_reapplied: "Broken, Then Reapplied",
  reapplied: "Reapplied",
});

/**
 * The neutral definition for a status this viewer does not know.
 *
 * @type {StatusPresentation}
 */
const UNKNOWN_STATUS = Object.freeze({
  title: "Recorded Status",
  name: "Recorded Status",
  subtitle: null,
  effect: "The recording names a status this viewer does not recognize.",
  accent: "none",
  magnitudeKind: "none",
  positiveDamageBreak: false,
  sourceClassId: null,
  sourceComponent: null,
});

/**
 * Return a frozen status definition for tokenId.
 *
 * tokenId is a status display token from `vocabulary.js` (for example
 * "slow_hunter_basic"), "spawn_shield" or "in_combat". String input is trimmed
 * before lookup. Unknown or non-string input receives a neutral "Recorded
 * Status" record with no magnitude kind, no source and no damage-break flag.
 * Known definitions are shared. This lookup does not decide whether an actor
 * may see the status, and it supplies no duration.
 *
 * @param {unknown} tokenId
 * @returns {StatusPresentation}
 */
export function statusPresentation(tokenId) {
  const key = typeof tokenId === "string" ? tokenId.trim() : "";
  return Object.hasOwn(STATUS_PRESENTATION, key)
    ? STATUS_PRESENTATION[key]
    : UNKNOWN_STATUS;
}

/**
 * Build a frozen lifecycle title and summary, or return null for an unknown
 * lifecycle.
 *
 * lifecycleTokenId is trimmed and must name a known lifecycle. statusTokenId
 * selects a status definition, with the neutral fallback. Titles put the
 * lifecycle word before the status title ("Applied Serrated Arrows (Slow)",
 * "Cleared Serrated Arrows (Slow) On Death"). Three exact cases have their own
 * words, compared on the untrimmed status input: an expired "in_combat" is
 * "Out of Combat"; a "trap_broken" "stun_hunter_trap" is "Freezing Trap (Stun
 * Broken)"; and a "trap_broken_and_reapplied" "stun_hunter_trap" is "Freezing
 * Trap (Stun Broken), Then Reapplied". Expiry and death clearing have a null summary; a broken
 * trap says why it ended; every other summary is the status effect text.
 * Inputs are unchanged; visibility and duration checks belong to the caller.
 *
 * @param {unknown} statusTokenId
 * @param {unknown} lifecycleTokenId
 * @returns {Readonly<{title: string, summary: string | null}> | null}
 */
export function statusLifecyclePresentation(statusTokenId, lifecycleTokenId) {
  const lifecycleKey =
    typeof lifecycleTokenId === "string" ? lifecycleTokenId.trim() : "";
  if (!Object.hasOwn(STATUS_LIFECYCLE_PREFIX, lifecycleKey)) {
    return null;
  }
  const status = statusPresentation(statusTokenId);
  const trapBreak =
    statusTokenId === "stun_hunter_trap" &&
    (lifecycleKey === "trap_broken" || lifecycleKey === "trap_broken_and_reapplied");
  const brokenTitle = statusNameFor("stun_broken")?.title ?? "Stun Broken";
  const title = trapBreak
    ? lifecycleKey === "trap_broken"
      ? brokenTitle
      : `${brokenTitle}, Then Reapplied`
    : lifecycleKey === "cleared_by_death"
      ? `Cleared ${status.title} On Death`
      : lifecycleKey === "expired" && statusTokenId === "in_combat"
        ? "Out of Combat"
        : `${STATUS_LIFECYCLE_PREFIX[/** @type {keyof typeof STATUS_LIFECYCLE_PREFIX} */ (lifecycleKey)]} ${status.title}`;
  const summary =
    lifecycleKey === "expired" || lifecycleKey === "cleared_by_death"
      ? null
      : lifecycleKey === "trap_broken" && statusTokenId === "stun_hunter_trap"
        ? "The Freezing Trap stun ended early because this agent took damage."
        : status.effect;
  return Object.freeze({
    title,
    summary,
  });
}

/**
 * Return a frozen aura definition, with a neutral fallback.
 *
 * auraId is trimmed when it is a string. The two modifier token IDs
 * "mage_amplification" and "warrior_mitigation" are accepted as aliases of the
 * host aura IDs "mage_damage_amplification" and "warrior_damage_mitigation".
 * Unknown values return generic "Recorded Aura" words. No radius, multiplier,
 * emitter identity or permission is inferred. Known definitions are shared.
 *
 * @param {unknown} auraId
 * @returns {AuraPresentation}
 */
export function auraPresentation(auraId) {
  const rawKey = typeof auraId === "string" ? auraId.trim() : "";
  const key = Object.hasOwn(AURA_ID_ALIASES, rawKey)
    ? AURA_ID_ALIASES[/** @type {keyof typeof AURA_ID_ALIASES} */ (rawKey)]
    : rawKey;
  return Object.hasOwn(AURA_PRESENTATION, key)
    ? AURA_PRESENTATION[key]
    : Object.freeze({
        fieldTitle: "Recorded Aura Field",
        recipientTitle: "Recorded Aura",
        fieldEffect: "The recording names an aura this viewer does not recognize.",
        aggregateEffect:
          "This agent has an aura effect this viewer does not recognize.",
        fieldEffectLabel: "Aura Effect",
        aggregateEffectLabel: "Total Aura Effect",
        accent: "none",
        effectKind: "generic",
      });
}

/**
 * Write how much time a status has left, the way the Champion Card's row
 * reads it (owner, 2 October 2026).
 *
 * id is any ID `symbols.js` statusNameFor accepts. Its time kind decides the
 * words: "ticks" gives "1 tick left" or "5 ticks left" (empty when ticks is
 * not a whole number of at least 0), "this_tick" gives "This tick" (Freezing
 * Trap (Stun Broken)), "now" gives "Now" (Rest) and "while_near" gives "While
 * near" (an aura). An unknown id reads like a "ticks" status. Pure.
 *
 * @param {unknown} id
 * @param {unknown} ticks Whole ticks left, or null.
 * @returns {string} The words, or "" when there is nothing to say.
 * @example statusTimeText("slow_warrior_charge", 5) // "5 ticks left"
 */
export function statusTimeText(id, ticks) {
  const time = statusNameFor(id)?.time ?? "ticks";
  if (time === "this_tick") return "This tick";
  if (time === "now") return "Now";
  if (time === "while_near") return "While near";
  return Number.isInteger(ticks) && /** @type {number} */ (ticks) >= 0
    ? `${tickCountText(ticks)} left`
    : "";
}

/**
 * Write one status as a single line, in the same words as its Champion Card
 * row (owner rulings, 2 October 2026: every status display uses the card's
 * approved names and names the source by class only).
 *
 * Example: "Charge (Slow) · 5 ticks left · It moves at ×0.5 speed. From a
 * Warrior." Parts, joined by " · ":
 *
 * 1. The status title (`symbols.js` statusNameFor: any of the fifteen
 *    approved names, auras, Rest and Freezing Trap (Stun Broken) included);
 *    an unknown token gives "Recorded Status".
 * 2. The time left (statusTimeText), only when it has words.
 * 3. What it does with its exact number and where it comes from by class
 *    (statusCardSentence of options.tokenId, options.magnitude and
 *    options.recipientClassId).
 *
 * The source is never an Agent ID and never "a hidden source", so the line is
 * the same in Agent POV and Oracle View. Numbers are copied from the caller's
 * authorized record and only formatted. Callers: Frame Explanation's status
 * items, the agent details panel (`panels.js`) and the Replay Viewer's
 * Champion Frame. Pure.
 *
 * @param {Readonly<{
 *   tokenId: unknown,
 *   magnitude?: unknown,
 *   remainingTicks?: unknown,
 *   recipientClassId?: unknown,
 * }>} options tokenId is the status's display token (statusTokenIdFromCatalogId
 *   turns a host status_id into one); magnitude is its recorded number (a
 *   multiplier, or Blessing of Freedom's lowest speed multiplier), default
 *   none; remainingTicks is the whole ticks left, default none;
 *   recipientClassId is the class (1 Mage to 5 Priest) of the champion that
 *   has it, so an aura of its own class reads "From its own aura."; default
 *   none.
 * @returns {string}
 */
export function statusLineV1(options) {
  const title = statusNameFor(options.tokenId)?.title ?? "Recorded Status";
  const time = statusTimeText(options.tokenId, options.remainingTicks);
  const what = statusCardSentence(
    options.tokenId,
    options.magnitude ?? null,
    options.recipientClassId ?? null,
  ).line;
  return [title, time, what].filter((part) => part !== "").join(" · ");
}

/**
 * Write the exact number of a status sentence the way the Champion Card
 * shows it, or null when amount is missing or not a finite number.
 *
 * kind "multiplier" gives "×1.5" (up to two decimals), "percent" gives "85%"
 * (amount 0.85, rounded to a whole percent) and "life" gives "1.6" (up to one
 * decimal). kind "none" always gives null. Pure.
 *
 * @param {import("./symbols.js").StatusAmountKind} kind
 * @param {unknown} amount
 * @returns {string | null}
 */
function statusAmountWords(kind, amount) {
  if (kind === "none" || typeof amount !== "number" || !Number.isFinite(amount)) {
    return null;
  }
  if (kind === "percent") {
    return `${formatDisplayNumber(amount * 100, { maximumFractionDigits: 0 })}%`;
  }
  if (kind === "life") {
    return formatDisplayNumber(amount, { maximumFractionDigits: 1 });
  }
  return `×${formatDisplayNumber(amount)}`;
}

/**
 * Write what one status does, for the Champion Card's second line, the status
 * lines (statusLineV1) and the status tips.
 *
 * id is any ID `symbols.js` statusNameFor accepts ("slow_warrior_charge",
 * "rest", "mage_amplification", ...). amount is the number the sentence needs,
 * copied from the audience's own record: a speed or damage or healing
 * multiplier (the status row's `magnitude`, or an aura modifier's
 * `multiplier`), Blessing of Freedom's lowest speed multiplier (0.85 reads
 * "85%") or the life Rest mends per tick. null, a missing value or a status
 * with no number uses the sentence without a number. recipientClassId is the
 * class (1 Mage to 5 Priest) of the champion that has the status: when it is
 * the status's own source class and the status has a `fromOwn` (the two
 * auras), the source reads "From its own aura." (owner, 2 October 2026).
 *
 * Returns {sentence, from, line}: sentence is "It moves at ×0.5 speed."; from
 * is where it comes from by class ("From a Warrior.", "From its own
 * Ultimate.", "From its own aura."), or null for a status no champion gives
 * (Spawn Shield, In Combat, Rest); line is both joined by a space. An unknown
 * id gives {sentence: "The recording names a status this viewer does not
 * recognize.", from: null, line: the same}. Sources are named by class only,
 * never by Agent ID, so the words are the same in Agent POV and Oracle View
 * (owner, 2 October 2026). Pure.
 *
 * @param {unknown} id
 * @param {unknown} [amount] Default null.
 * @param {unknown} [recipientClassId] Default null (no "own" wording).
 * @returns {Readonly<{sentence: string, from: string | null, line: string}>}
 * @example statusCardSentence("mage_burst", 1.5).line
 * // "Its attacks deal ×1.5 damage. From its own Ultimate."
 * @example statusCardSentence("mage_amplification", 1.15, 1).from
 * // "From its own aura."
 */
export function statusCardSentence(id, amount = null, recipientClassId = null) {
  const named = statusNameFor(id);
  if (named === null) {
    const unknown = "The recording names a status this viewer does not recognize.";
    return Object.freeze({ sentence: unknown, from: null, line: unknown });
  }
  const words = statusAmountWords(named.amountKind, amount);
  const sentence =
    words === null || !named.sentence.includes("{amount}")
      ? named.plainSentence
      : named.sentence.replace("{amount}", words);
  const from =
    named.fromOwn !== null &&
    named.sourceClassId !== null &&
    recipientClassId === named.sourceClassId
      ? named.fromOwn
      : named.from;
  return Object.freeze({
    sentence,
    from,
    line: from === null ? sentence : `${sentence} ${from}`,
  });
}
