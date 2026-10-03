/**
 * @file Own every agent, class and team label the browser clients show.
 *
 * This module is the one owner of label forms in every client (the
 * BattleClient and the Replay Viewer):
 *
 * - The champion label "Mage · ID 5" (no team word; the scorebug heading is the
 *   only place a team name appears beside a champion).
 * - The token-chip text "Agent 5" (the chip itself draws the class emblem in its
 *   team ring).
 * - The source name "Hunter ID 7", used in "from Hunter ID 7's Serrated Arrows".
 * - The Out Of Sight chip text "Warrior · ID 6 · Out Of Sight".
 * - The plain accessible identity "Agent ID 5".
 * - Class names (Mage, Warrior, Hunter, Rogue, Priest), team names ("Team A",
 *   "Team B" only) and tick counts ("1 Tick", "2 Ticks" for labels; "1 tick",
 *   "2 ticks" inside sentences).
 * - The Full Record chip words ("Full Record · Not Limited To Agent 3's View").
 *
 * `vocabulary.js` builds its class and team display tokens from the two tables
 * here, and `class-documentation.js` reads the class names, so each name has one
 * owner. Callers are `explanations.js`, `source-attribution.js`, `scene.js`,
 * `panels.js`, the choreography modules and the adapter.
 *
 * Boundary: this module formats labels from fields the caller already has the
 * right to show. It never derives an ID from a global slot, a position or a
 * private simulator record, and it reads fields without calling getters.
 * Agent POV limits belong to the caller: it must pass only identities its
 * audience may see. Source-to-agent joins belong to `source-attribution.js`.
 */

/**
 * One class's identity: its class ID (1 to 5), its stable token ID (also the
 * CSS accent key, for example "mage") and its display name ("Mage").
 *
 * @typedef {Readonly<{
 *   classId: 1 | 2 | 3 | 4 | 5,
 *   tokenId: "mage" | "warrior" | "hunter" | "rogue" | "priest",
 *   label: "Mage" | "Warrior" | "Hunter" | "Rogue" | "Priest",
 * }>} ClassIdentity
 */

/**
 * One team's identity: its team ID (1 or 2), its stable token ID, its display
 * name and the single letter used on compact marks.
 *
 * @typedef {Readonly<{
 *   teamId: 1 | 2,
 *   tokenId: "team_a" | "team_b",
 *   label: "Team A" | "Team B",
 *   letter: "A" | "B",
 * }>} TeamIdentity
 */

/**
 * The display forms of one agent identity. Every field is plain text.
 *
 * - title: the champion label, for example "Mage · ID 5".
 * - publicIdentity: the plain accessible identity, for example "Agent ID 5".
 * - chipText: the token-chip text, for example "Agent 5".
 * - sourceName: the name used in source wording, for example "Mage ID 5".
 * - outOfSightLabel: the Out Of Sight chip text, for example
 *   "Mage · ID 5 · Out Of Sight".
 * - classLabel: "Mage" … "Priest", or "Unknown".
 * - teamLabel: "Team A", "Team B" or "Unknown". Labels above never carry it.
 * - accent: the class accent key, or "none" for an unknown class.
 *
 * A missing or invalid ID gives "ID unavailable", "Agent ID unavailable" and
 * "Agent unavailable" in the matching forms.
 *
 * @typedef {Readonly<{
 *   title: string,
 *   publicIdentity: string,
 *   chipText: string,
 *   sourceName: string,
 *   outOfSightLabel: string,
 *   classLabel: string,
 *   teamLabel: string,
 *   accent: "none" | "mage" | "warrior" | "hunter" | "rogue" | "priest",
 * }>} AgentIdentityLabels
 */

/**
 * The five classes by class ID. Frozen; shared by every caller. The order of
 * the keys (1 to 5) is the canonical class order.
 *
 * @type {Readonly<Record<1 | 2 | 3 | 4 | 5, ClassIdentity>>}
 */
export const CLASS_IDENTITIES = Object.freeze({
  1: Object.freeze({ classId: 1, tokenId: "mage", label: "Mage" }),
  2: Object.freeze({ classId: 2, tokenId: "warrior", label: "Warrior" }),
  3: Object.freeze({ classId: 3, tokenId: "hunter", label: "Hunter" }),
  4: Object.freeze({ classId: 4, tokenId: "rogue", label: "Rogue" }),
  5: Object.freeze({ classId: 5, tokenId: "priest", label: "Priest" }),
});

/**
 * The two teams by team ID. Frozen; shared by every caller. "Team A" and
 * "Team B" are the only team names any client shows.
 *
 * @type {Readonly<Record<1 | 2, TeamIdentity>>}
 */
export const TEAM_IDENTITIES = Object.freeze({
  1: Object.freeze({ teamId: 1, tokenId: "team_a", label: "Team A", letter: "A" }),
  2: Object.freeze({ teamId: 2, tokenId: "team_b", label: "Team B", letter: "B" }),
});

const AUTHORIZED_IDENTITY_KEYS = Object.freeze([
  "presentation_key",
  "public_agent_id",
  "class_id",
  "team_id",
]);

/**
 * Look up one class by its class ID.
 *
 * classId must be an integer from 1 (Mage) to 5 (Priest). Strings, other
 * numbers, null and undefined return null. No class is guessed from a name,
 * a slot or a position. The returned record is shared and frozen.
 *
 * @param {unknown} classId
 * @returns {ClassIdentity | null}
 */
export function classIdentityFromId(classId) {
  return Number.isInteger(classId) &&
    Object.hasOwn(CLASS_IDENTITIES, /** @type {number} */ (classId))
    ? CLASS_IDENTITIES[/** @type {1 | 2 | 3 | 4 | 5} */ (classId)]
    : null;
}

/**
 * Look up one team by its team ID.
 *
 * teamId must be the integer 1 (Team A) or 2 (Team B). Any other value
 * returns null. The returned record is shared and frozen.
 *
 * @param {unknown} teamId
 * @returns {TeamIdentity | null}
 */
export function teamIdentityFromId(teamId) {
  return Number.isInteger(teamId) &&
    Object.hasOwn(TEAM_IDENTITIES, /** @type {number} */ (teamId))
    ? TEAM_IDENTITIES[/** @type {1 | 2} */ (teamId)]
    : null;
}

/**
 * Return the team name for teamId: "Team A" for 1, "Team B" for 2.
 *
 * Any other value returns null, so a caller can choose its own "Unknown" or
 * "Unavailable" text. This never returns a colour or host name.
 *
 * @param {unknown} teamId
 * @returns {"Team A" | "Team B" | null}
 */
export function teamLabelFromId(teamId) {
  return teamIdentityFromId(teamId)?.label ?? null;
}

/**
 * Write a tick count for use inside a sentence: "1 tick", "3 ticks".
 *
 * count must be an integer. Only exactly 1 is singular; 0 and negative counts
 * are plural ("0 ticks"). A non-integer returns null. Ticks are simulator
 * decision steps; no seconds are converted.
 *
 * @param {unknown} count
 * @returns {string | null}
 */
export function tickCountText(count) {
  if (!Number.isInteger(count)) return null;
  return `${count} ${count === 1 ? "tick" : "ticks"}`;
}

/**
 * Write a tick count for a label or chip: "1 Tick", "3 Ticks".
 *
 * Same rules as tickCountText, with Title Case for labels. A non-integer
 * returns null.
 *
 * @param {unknown} count
 * @returns {string | null}
 */
export function tickCountLabel(count) {
  if (!Number.isInteger(count)) return null;
  return `${count} ${count === 1 ? "Tick" : "Ticks"}`;
}

/**
 * Write the Full Record chip words: "Full Record · Not Limited To Agent 3's
 * View".
 *
 * chipText is the POV agent's token-chip text ("Agent 3"), or null, undefined
 * or blank when the page does not know it; then the words name "This Agent".
 * The one owner of these words: the HUD strip, the rail's chips (ledger.js) and
 * panels.js all call it, so they always agree. Pure.
 *
 * Agent POV: name only the POV agent itself, never another champion.
 *
 * @param {string | null | undefined} chipText
 * @returns {string}
 */
export function fullRecordChipWords(chipText) {
  const name =
    typeof chipText === "string" && chipText.trim() !== "" ? chipText : "This Agent";
  return `Full Record · Not Limited To ${name}'s View`;
}

/**
 * Return display labels from rawAgent's own data fields.
 *
 * rawAgent may be any value. Read public_agent_id, class_id, team_id and
 * optional display_agent_id without calling property getters. Invalid fields
 * independently become "Unknown", "ID unavailable" or the "none" accent. A
 * one-digit display_agent_id takes precedence over the public ID. Return a
 * frozen AgentIdentityLabels record (title "Mage · ID 5" and the other forms).
 * This formats labels only: it does not certify information rights, and it does
 * not change the input. In Agent POV the caller must pass only an agent its
 * audience may see.
 *
 * @param {unknown} rawAgent
 * @returns {AgentIdentityLabels}
 */
export function canonicalAgentIdentity(rawAgent) {
  const fields = identityDataFields(rawAgent);
  return formatIdentity(
    fields?.public_agent_id,
    fields?.class_id,
    fields?.team_id,
    fields?.display_agent_id,
  );
}

/**
 * Check a plain identity record and return its frozen display projection.
 *
 * rawAgent must have own enumerable data fields presentation_key,
 * public_agent_id, class_id and team_id. Class IDs are integers 1–5 and team IDs
 * are 1–2. IDs are nonempty strings of at most 512 characters, with no outer
 * whitespace or line breaks. Extra string-named fields are allowed; symbol
 * keys and getter-backed required fields are rejected. A valid optional
 * display_agent_id (one digit) may override the shown ID. Invalid input returns
 * null. The result carries every AgentIdentityLabels form plus presentationKey,
 * publicAgentId, classId and teamId. Matching these fields alone does not
 * establish a source-attribution join; the caller still owns that join and the
 * Agent POV limits.
 *
 * @param {unknown} rawAgent
 * @returns {Readonly<{
 *   presentationKey: string,
 *   publicAgentId: string,
 *   classId: 1 | 2 | 3 | 4 | 5,
 *   teamId: 1 | 2,
 *   title: string,
 *   publicIdentity: string,
 *   chipText: string,
 *   sourceName: string,
 *   outOfSightLabel: string,
 *   classLabel: string,
 *   teamLabel: string,
 *   accent: "mage" | "warrior" | "hunter" | "rogue" | "priest",
 * }> | null}
 */
export function exactAuthorizedAgentIdentityV1(rawAgent) {
  const fields = exactAuthorizedIdentityFields(rawAgent);
  if (fields === null) return null;
  const presentationKey = exactIdentifier(fields.presentation_key);
  const publicAgentId = exactIdentifier(fields.public_agent_id);
  const classId = fields.class_id;
  const teamId = fields.team_id;
  const classIdentity = Number.isSafeInteger(classId)
    ? classIdentityFromId(classId)
    : null;
  const teamIdentity = Number.isSafeInteger(teamId) ? teamIdentityFromId(teamId) : null;
  if (
    presentationKey === null ||
    publicAgentId === null ||
    classIdentity === null ||
    teamIdentity === null
  ) {
    return null;
  }
  const identity = formatIdentity(
    publicAgentId,
    classId,
    teamId,
    fields.display_agent_id,
  );
  return Object.freeze({
    presentationKey,
    publicAgentId,
    classId: classIdentity.classId,
    teamId: teamIdentity.teamId,
    title: identity.title,
    publicIdentity: identity.publicIdentity,
    chipText: identity.chipText,
    sourceName: identity.sourceName,
    outOfSightLabel: identity.outOfSightLabel,
    classLabel: identity.classLabel,
    teamLabel: identity.teamLabel,
    accent: /** @type {"mage" | "warrior" | "hunter" | "rogue" | "priest"} */ (
      identity.accent
    ),
  });
}

/**
 * Create the frozen label forms from four unchecked values.
 *
 * publicAgentId supplies the shown ID unless displayId is one decimal digit.
 * classId 1–5 selects a class name and accent; teamId 1–2 selects Team A or
 * Team B. Invalid parts get neutral "Unknown" or "unavailable" text on their
 * own. No coercion, input change, source lookup or authority check occurs.
 *
 * @param {unknown} publicAgentId The public agent ID string, or anything invalid.
 * @param {unknown} classId The class ID, 1 to 5.
 * @param {unknown} teamId The team ID, 1 or 2.
 * @param {unknown} displayId An optional one-digit display ID string.
 * @returns {AgentIdentityLabels}
 */
function formatIdentity(publicAgentId, classId, teamId, displayId) {
  const shownId =
    typeof displayId === "string" && /^[0-9]$/u.test(displayId)
      ? displayId
      : exactIdentifier(publicAgentId);
  const classIdentity = classIdentityFromId(classId);
  const classLabel = classIdentity?.label ?? "Unknown";
  const teamLabel = teamLabelFromId(teamId) ?? "Unknown";
  const idText = shownId === null ? "ID unavailable" : `ID ${shownId}`;
  return Object.freeze({
    title: `${classLabel} · ${idText}`,
    publicIdentity: shownId === null ? "Agent ID unavailable" : `Agent ID ${shownId}`,
    chipText: shownId === null ? "Agent unavailable" : `Agent ${shownId}`,
    sourceName: `${classLabel} ${idText}`,
    outOfSightLabel: `${classLabel} · ${idText} · Out Of Sight`,
    classLabel,
    teamLabel,
    accent: classIdentity?.tokenId ?? /** @type {const} */ ("none"),
  });
}

/**
 * Accept a nonempty string of at most 512 characters with no outer whitespace
 * or CR/LF. value is returned unchanged when valid; all other inputs return
 * null. The function performs no registry lookup or identity comparison.
 *
 * @param {unknown} value
 * @returns {string | null}
 */
function exactIdentifier(value) {
  return typeof value === "string" &&
    value.length > 0 &&
    value.length <= 512 &&
    value.trim() === value &&
    !/[\r\n]/u.test(value)
    ? value
    : null;
}

/**
 * Read four own data fields from value without calling their getters.
 *
 * Reject null, arrays and nonobjects. Copy public_agent_id, class_id, team_id
 * and display_agent_id into a new null-prototype record; absent/getter-backed
 * fields become undefined. The returned record is not frozen. Descriptor
 * inspection errors return null. Field validity and authority are not checked.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, unknown>> | null}
 */
function identityDataFields(value) {
  try {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    const descriptors = Object.getOwnPropertyDescriptors(value);
    /** @type {Record<string, unknown>} */
    const fields = Object.create(null);
    for (const key of ["public_agent_id", "class_id", "team_id", "display_agent_id"]) {
      const descriptor = descriptors[key];
      fields[key] =
        descriptor && Object.hasOwn(descriptor, "value") ? descriptor.value : undefined;
    }
    return fields;
  } catch {
    return null;
  }
}

/**
 * Read required identity data from a plain object or return null.
 *
 * value must have Object.prototype or a null prototype, no symbol keys and
 * four own enumerable data fields named in AUTHORIZED_IDENTITY_KEYS. Extra
 * string fields are allowed. Copy those fields and an eligible display_agent_id
 * into a new null-prototype record, without freezing it or changing value.
 * Inspection errors return null; field values are checked by the caller.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, unknown>> | null}
 */
function exactAuthorizedIdentityFields(value) {
  try {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return null;
    const keys = Reflect.ownKeys(value);
    if (
      keys.some((key) => typeof key !== "string") ||
      AUTHORIZED_IDENTITY_KEYS.some((key) => !keys.includes(key))
    ) {
      return null;
    }
    const descriptors = Object.getOwnPropertyDescriptors(value);
    /** @type {Record<string, unknown>} */
    const fields = Object.create(null);
    for (const key of AUTHORIZED_IDENTITY_KEYS) {
      const descriptor = descriptors[key];
      if (
        !descriptor ||
        !Object.hasOwn(descriptor, "value") ||
        !descriptor.enumerable
      ) {
        return null;
      }
      fields[key] = descriptor.value;
    }
    const display = descriptors.display_agent_id;
    if (display && Object.hasOwn(display, "value") && display.enumerable) {
      fields.display_agent_id = display.value;
    }
    return fields;
  } catch {
    return null;
  }
}
