/**
 * @file The Champions look (Life Vessels): each champion's token in the `body` layer.
 *
 * A living champion is a coin that fits inside its true body circle: a team ring
 * (Team B's ring is split by a thin Ink line), a vessel that fills with the team
 * color from the bottom to a height equal to its life, an ivory waterline (Cinder
 * at 30% life or less), a faint class halo and the upright class emblem (an
 * outline emblem while Lore Emblems is off; icons.js picks the style). A fallen
 * champion is a quiet Ashen Token (no red X). A champion under its Spawn Shield
 * looks warded: a ring of 12 dashes, a pale veil and up to three silver diamonds
 * counting the shield ticks (this replaces the old shield chip). On the tick its
 * Spawn Shield ends, a champion the audience saw at tick start keeps its ward
 * until Settle; then the band closes: the ward gives way to the solid team ring
 * and the token comes up from 80% to full (over 200 ms with Spawn Shield Expiry
 * on, a cut without it; store `skins/champions.js` wardState). The dashed ring
 * that lifts off is the Statuses look's Ward Lift. There is no Team B
 * chevron. Each token sits in an animatable wrapper `g.agent-glide`
 * (data-glide-key) that also holds its ward group, so the choreographer can glide a
 * body and its ward together.
 *
 * Class Seals (switch class_seals, off by default): from a
 * body radius of 18 px, each champion stands on its class seal instead of the
 * plain halo: a Mage starglass octagon, a Warrior bronze boss with eight rivets,
 * a Hunter greenwood quatrefoil, a Rogue steel cushion lozenge or a Priest gold
 * sun (the Priest's emblem then shows its cross alone, because the seal is its
 * sun: icons.js createSvgIcon's onClassSeal option, also on its held living look
 * and over its ash outline). From 22 px the seal also carries its engraving. A seal is dormant while the
 * Ultimate recharges and awakens in its own material when the Ultimate is ready
 * (only while Cooldown Effects is on). It steps back (fainter outline, no
 * engraving) while a Rune Zones mark is on the token (Status Marks on). A fallen
 * champion keeps a cold ash outline of its seal. The store and spec are
 * `skins/champions.js` (Class Seals) and `specs/champions.json`.
 *
 * Timing (the beat clock): the DOM always holds the END state. Nodes that show an
 * earlier look carry the choreographer's beat-hold attributes (data-beat-hold,
 * data-beat-key, data-beat-side, data-beat-fade): the life level changes at Impact
 * ("impact"), a dying champion keeps its living look until its Fall beat ("fall")
 * and a returning champion is set down at its Return beat ("return": the hold is
 * on its glide wrapper, from 0, so the token, its ward and every mark riding in
 * the wrapper wait together and nothing of it shows at the waystone before the
 * beat); a lifting Spawn Ward holds until Settle ("settle"). A seal goes
 * dormant at Impact when its champion spends the Ultimate ("impact"), and awakens
 * or steps back under a new mark at Settle ("settle"). The 80 ms hit flash, the
 * gold or Moon Dew pour and the Mage seal's awakening flare carry
 * data-beat-pulse; the hitstop shiver carries data-beat-shiver. This file never
 * starts a timer or a CSS animation.
 *
 * Callers: `scene.js` calls records once per render, build for a champion it has
 * not drawn yet, then update, and puts each wrapper in the dead or living group
 * (every corpse under every living body). `visual-filters.js` reads PAINT_PARTS.
 *
 * Boundary (Agent POV): records reads only the authorized arena scene for this
 * audience (`scene.agents`, `scene.selection`, `scene.spawn_shield_mechanics`) and
 * the plain beat facts its caller passes in: that same audience's incoming rows
 * (Oracle View: latest_events; Agent POV: visual_events, as
 * `authorizedPresentationIncomingRows` returns them), its `beatView` result and,
 * optionally, its startSeen test (which bodies the audience saw at tick start).
 * Like every look it imports neither the presentation adapter nor the
 * normalizer. In Agent POV the scene holds only bodies the POV team can see (plus
 * locally seen corpses), so an unseen champion has no record, no node and no hit
 * area, and no timing here depends on an unseen champion. Missing life draws a
 * dotted ring, never a guessed level. A seal reads only its own row's Ultimate
 * cooldown and statuses (a missing cooldown gives a dormant seal) and its beat
 * cues come only from the audience's own rows. It never reads a researcher view
 * or a match summary. The Rune Zones mark rules come from `statuses.js`
 * (runeMarksFromStatuses and runeZones), so a seal steps back exactly when that
 * look draws a mark.
 */
import { formatDisplayNumber } from "../display.js";
import {
  createSpawnShieldView,
  explainAgent,
  explainPovAgent,
} from "../explanations.js";
import { createSvgIcon, defaultIconStyle } from "../icons.js";
import { registerTooltipOwner } from "../tooltip.js";
import { DEFAULT_VISUAL_FILTER_STATE } from "../visual-filters.js";
import {
  classTokenFromId,
  statusTokenIdFromCatalogId,
  teamTokenFromId,
} from "../vocabulary.js";
import {
  agentIdentity,
  asArray,
  displayIdentityAttributes,
  finiteNumber,
  isRecord,
  paintParts,
  screenPoint,
  setNodeAttributes,
  svgNode,
} from "./look-support.js";
import { runeMarksFromStatuses, runeZones } from "./statuses.js";

/**
 * @typedef {import("./look-support.js").JsonRecord} JsonRecord
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {{
 *   wrapper: SVGElement,
 *   root: SVGElement,
 *   body: SVGElement,
 *   life: SVGElement,
 *   flash: SVGElement,
 *   inner: SVGElement,
 *   halo: SVGElement,
 *   seal: SVGElement,
 *   classIcon: SVGSVGElement,
 *   classLetter: SVGElement,
 *   teamRing: SVGElement,
 *   ringSplit: SVGElement,
 *   fallen: SVGElement,
 *   falling: SVGElement,
 *   shieldRoot: SVGElement | null,
 * }} ChampionNodes
 *   One champion's retained nodes. wrapper is the glide group; root is the
 *   `g.agent` token. body (`.agent-body`) is the hit circle: the groove under the
 *   vessel for a living champion, the ash disc for a fallen one. life, seal,
 *   fallen and falling are containers whose children are rebuilt on every update
 *   (seal, `g.champion-seal` inside the inner group between the halo and the
 *   emblem, is empty while no Class Seal shows). shieldRoot is the ward group
 *   (`g.agent-spawn-shield`) or, on the tick a seen ward lifts, the lift group
 *   (`g.agent-ward-lift`, holding the ward's earlier look), null while neither
 *   shows (update makes it when one shows and drops it when it ends);
 *   `scene.js` appends it after the token inside the wrapper.
 * @typedef {Readonly<{
 *   R: number,
 *   ringWidth: number,
 *   ringRadius: number,
 *   splitWidth: number,
 *   doubleBand: boolean,
 *   vesselRadius: number,
 *   lifeWidth: number,
 *   lifeRadius: number,
 *   haloRadius: number,
 *   emblemSize: number,
 *   coinRadius: number,
 *   sealRadius: number,
 * }>} TokenGeometry
 *   Token sizes in CSS pixels for body radius R (see tokenGeometry).
 * @typedef {"mage" | "warrior" | "hunter" | "rogue" | "priest"} SealClass
 *   The class CSS key of a champion that has a Class Seal.
 * @typedef {Readonly<{awake: boolean, dim: boolean}>} SealLook
 *   One still look of a seal. awake: the seal is awakened (the Ultimate is
 *   ready and Cooldown Effects is on); dim: the seal steps back because a Rune
 *   Zones mark is on the token (Status Marks on).
 * @typedef {Readonly<{
 *   classKey: SealClass,
 *   engraved: boolean,
 *   start: SealLook,
 *   end: SealLook,
 *   awakeBeat: "impact" | "settle" | null,
 *   dimBeat: "settle" | null,
 * }>} ClassSeal
 *   A champion record's Class Seal (see records). engraved is true from a body
 *   radius of 22 px. end is the END look; start the look before this tick's
 *   beats (equal to end when there is no cue). awakeBeat names the beat at which
 *   awake changes: "impact" when the champion spent its Ultimate this tick,
 *   "settle" when its Ultimate became ready, null for no change. dimBeat is
 *   "settle" when dim changes this tick, otherwise null. For a "falling" record
 *   start is the held living look shown until the Fall beat and end is dormant
 *   and undimmed; for a "fallen" record both are dormant and undimmed (only the
 *   ash outline is drawn).
 */

/** Life at or below this share of the maximum shows as Cinder (Low Life Warning). */
export const LOW_LIFE_SHARE = 0.3;

/** Fallen Step Aside: a standing champion closer than this (world units) covers a corpse. */
export const STEP_ASIDE_DISTANCE = 0.6;

/** Fallen Step Aside: the ash coin's gap to the covering champion's rim, in CSS pixels. */
const COIN_GAP = 3;

/** Hitstop shiver amplitude by class key, in CSS pixels (the Warrior never flinches). */
export const SHIVER_PIXELS = Object.freeze({
  mage: 1.5,
  warrior: 0,
  hunter: 1,
  rogue: 1,
  priest: 1,
});

/**
 * Beat-clock lengths this look writes into its nodes, in ms at playback rate 1:
 * flash (the hit flash), shiver and lethalShiver (the hitstop on an Ultimate or a
 * killing blow), fall (the cross-fade to ash at the Fall beat), setDown (the fade
 * in at the Return beat when Resurrection Effects is on) and pour (the gold or
 * Moon Dew pour into the vessel).
 */
export const BEAT_LENGTHS = Object.freeze({
  flash: 80,
  shiver: 90,
  lethalShiver: 120,
  shiverDelay: 20,
  fall: 300,
  setDown: 300,
  pour: 460,
});

/**
 * The Class Seals numbers (store `skins/champions.js` seal and awake; spec
 * `specs/champions.json`, Class Seals). Sizes are body radii in CSS px and
 * times are ms at playback rate 1:
 * - minRadius 18: no seal below it (the plain halo shows); engraveRadius 22: no
 *   engraving below it.
 * - spend 120: the awakened look goes dark at Impact when the Ultimate is spent.
 * - awaken 240: the cross-fade to the awakened look at Settle when the Ultimate
 *   becomes ready.
 * - dim 200: the step back (or forward) at Settle when a mark comes or goes.
 * - flare 160: the Mage's Starcore facet flare as the seal awakens (play only).
 * - rivet 120 and rivetStep 30: each Warrior rivet pair lights over 120 ms, the
 *   next pair from the bottom 30 ms later (see drawSeal for how the beat clock
 *   plays this).
 * - ashOpacity 0.35: the cold ash seal outline on a fallen champion.
 */
export const CLASS_SEAL = Object.freeze({
  minRadius: 18,
  engraveRadius: 22,
  spend: 120,
  awaken: 240,
  dim: 200,
  flare: 160,
  rivet: 120,
  rivetStep: 30,
  ashOpacity: 0.35,
});

/**
 * The Spawn Ward's lift on the tick its Spawn Shield ends (store
 * `skins/champions.js` wardState and living; spec `specs/champions.json`):
 * the ward (veil, dashed ring and `marks` diamond, the last shield tick) holds
 * until Settle, then gives way to the solid team ring over `fade` ms (at
 * playback rate 1) while the token comes up from `fromOpacity` to full. With
 * Spawn Shield Expiry off the change is a `cut` (1 ms) at Settle.
 */
export const WARD_LIFT = Object.freeze({
  fade: 200,
  cut: 1,
  fromOpacity: 0.8,
  marks: 1,
});

/** The classes that have a Class Seal, by class CSS key. */
const SEAL_CLASSES = Object.freeze(
  new Set(["mage", "warrior", "hunter", "rogue", "priest"]),
);

/**
 * The token is drawn inside the body circle, so it needs no allocator room of
 * its own: insideBody is true and the extra room around the body is 0 px.
 */
export const FOOTPRINT = Object.freeze({
  token: Object.freeze({ insideBody: true, outsideMargin: 0 }),
});

/**
 * Nodes per champion, not counting the strokes inside icon glyphs: wrapper, token,
 * hit circle, about 12 life nodes, flash, inner group with halo, emblem and letter,
 * ring and split, about 6 ash nodes, about 12 nodes for a falling champion's held
 * living look and 5 ward nodes, plus the Class Seal: its group and at most 13
 * nodes on a tick where it changes (the look before, the look between Impact and
 * Settle, the END look, and the Mage flare or the four Warrior rivet pairs).
 * The lifting Spawn Ward reuses the 5 ward nodes (a ward and a lift never show
 * together). Up to 14 beat-clock overrides per champion: 6 for the life before
 * and after, pour, flash, shiver and the fall or return pair, 6 for the seal,
 * and on the tick a Spawn Ward lifts 4 more (the held ward, the solid ring,
 * Team B's split and the token's opacity); only a living champion lifts, so the
 * fall or return pair is then absent (6 - 2 + 6 + 4 = 14).
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 58,
  animationsPerRecord: 14,
  persistentNodesPerRecord: 0,
});

/**
 * One durable paint part per switch this look owns. Spawn Shield switches the
 * ward; Class Seals switches the class seal; the others switch the token's look,
 * its life drawing and its beat cues. Cooldown Effects and Status Marks, which
 * the seals also follow, are owned (and listed) by the Statuses look.
 */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "spawn_shield" }, "spawn_shield"],
  [{ surface: "durable", kind: "life_vessel" }, "life_vessels"],
  [{ surface: "durable", kind: "life_lost_layer" }, "recent_health_change"],
  [{ surface: "durable", kind: "low_life_waterline" }, "low_life_warning"],
  [{ surface: "durable", kind: "hit_reaction" }, "hit_reactions"],
  [{ surface: "durable", kind: "rest_pour" }, "regeneration_effects"],
  [{ surface: "durable", kind: "champion_set_down" }, "resurrection_effects"],
  [{ surface: "durable", kind: "fallen_step_aside" }, "fallen_step_aside"],
  [{ surface: "durable", kind: "class_letter" }, "class_letters"],
  [{ surface: "durable", kind: "life_segment" }, "life_segments"],
  [{ surface: "durable", kind: "class_seal" }, "class_seals"],
]);

/**
 * The switch ids this look reads (each with its registry default).
 * cooldown_effects and status_marks belong to the Statuses look; the seals
 * only follow them (no awakening while Cooldown Effects is off, no step back
 * while Status Marks is off). spawn_shield_expiry belongs to the Statuses
 * look's Ward Lift; the ward's band only follows it (a 200 ms close, or a cut
 * while it is off).
 */
const SWITCH_IDS = Object.freeze([
  "life_vessels",
  "recent_health_change",
  "low_life_warning",
  "hit_reactions",
  "regeneration_effects",
  "resurrection_effects",
  "fallen_step_aside",
  "class_letters",
  "life_segments",
  "class_seals",
  "cooldown_effects",
  "status_marks",
  "spawn_shield_expiry",
]);

/**
 * Choose a stable retained-node key from an accepted display row.
 *
 * A nonempty agent.display_key wins; otherwise an integer global_slot, or null.
 * This is a DOM join key, not a public label or a new slot fact.
 *
 * @param {JsonRecord} agent
 * @returns {number | string | null}
 */
export function agentDisplayIdentity(agent) {
  if (typeof agent.display_key === "string" && agent.display_key.length > 0) {
    return agent.display_key;
  }
  return Number.isInteger(agent.global_slot) ? Number(agent.global_slot) : null;
}

/**
 * Return the integer global_slot on agent, otherwise fallback.
 *
 * fallback is a renderer-local layout index. It may place a POV body but must
 * never be shown as an undisclosed simulator global slot.
 *
 * @param {JsonRecord} agent
 * @param {number} fallback
 * @returns {number}
 */
function agentLayoutSlot(agent, fallback) {
  return Number.isInteger(agent.global_slot) ? Number(agent.global_slot) : fallback;
}

/**
 * Return the token sizes for a body of radius R CSS pixels.
 *
 * R is the projected body radius (any finite number; values below 0 count as 0).
 * From R 10 up the team ring is max(3.5, 0.155R) wide (below that max(1.5, 0.2R))
 * and Team B's ring is split by an Ink line max(1, 0.25 x ring width) wide. The
 * vessel fills the circle inside the ring (radius R - ring width). The ivory life
 * ring used when Life Vessels is off is max(1.5, 0.1R) wide just inside the team
 * ring. The class halo has radius 0.34R; the emblem box is clamp(0.8R, 9, 26) px;
 * the Fallen Step Aside ash coin has radius 0.46R. sealRadius is the outer edge
 * of a Class Seal: the store's status band edge less 0.4 px, that is R minus the
 * ring width, the life width, 2 x max(0.75, 0.03R), 0.09R and 0.4 (10.8 at R 20,
 * 13.16 at R 23, 21.02 at R 36; never below 0), so a seal never reaches the life
 * ring or the team ring. Every part stays inside the body circle. Pure; the
 * store's numbers (`skins/champions.js` geo).
 *
 * @param {number} R
 * @returns {TokenGeometry}
 */
export function tokenGeometry(R) {
  const radius = Math.max(0, finiteNumber(R));
  const ringWidth =
    radius >= 10 ? Math.max(3.5, 0.155 * radius) : Math.max(1.5, 0.2 * radius);
  const lifeWidth = Math.max(1.5, 0.1 * radius);
  const gap = Math.max(0.75, 0.03 * radius);
  return Object.freeze({
    R: radius,
    ringWidth,
    ringRadius: Math.max(0, radius - ringWidth / 2),
    splitWidth: Math.max(1, 0.25 * ringWidth),
    doubleBand: radius >= 10,
    vesselRadius: Math.max(0, radius - ringWidth),
    lifeWidth,
    lifeRadius: Math.max(0, radius - ringWidth - lifeWidth / 2),
    haloRadius: 0.34 * radius,
    emblemSize: Math.min(26, Math.max(9, 0.8 * radius)),
    coinRadius: 0.46 * radius,
    sealRadius: Math.max(
      0,
      radius - ringWidth - lifeWidth - 2 * gap - 0.09 * radius - 0.4,
    ),
  });
}

/**
 * Read one switch from the page's filter state, falling back to its default.
 *
 * state is the visual filter state or null/undefined (every switch at its
 * registry default); id is a registry switch id. A non-boolean entry also falls
 * back to the default. Pure.
 *
 * @param {unknown} state
 * @param {string} id
 * @returns {boolean}
 */
function switchOn(state, id) {
  if (isRecord(state) && typeof state[id] === "boolean") {
    return Boolean(state[id]);
  }
  return (
    /** @type {Record<string, boolean>} */ (DEFAULT_VISUAL_FILTER_STATE)[id] === true
  );
}

/**
 * Collect this audience's life facts for the tick from its own incoming rows.
 *
 * rows are the audience's incoming rows as `authorizedPresentationIncomingRows`
 * returns them ({kind, vocabulary, payload}; Oracle View latest_events, Agent POV
 * only its visual_events); anything that is not an array gives empty results and
 * rows that are not "event" vocabulary are skipped. Return {changes, ultimate}:
 * changes maps a presentation key to {before, after, rest} health points from
 * its recipient_health_resolution (transition_start_health,
 * health_after_combat_resolution) and adds rest points from health_regenerated;
 * ultimate is the set of presentation keys an Ultimate (ability_component
 * "ultimate") reached this tick; shieldEnded is the set of presentation keys
 * whose Spawn Shield ended this tick (a spawn_shield_expired row's agent_anchor).
 * Pure; reads the rows only.
 *
 * @param {unknown} rows
 * @returns {{
 *   changes: Map<string, {before: number | null, after: number | null, rest: number}>,
 *   ultimate: Set<string>,
 *   shieldEnded: Set<string>,
 * }}
 */
function tickLifeFacts(rows) {
  /** @type {Map<string, {before: number | null, after: number | null, rest: number}>} */
  const changes = new Map();
  /** @type {Set<string>} */
  const ultimate = new Set();
  /** @type {Set<string>} */
  const shieldEnded = new Set();
  for (const row of Array.isArray(rows) ? rows : []) {
    if (!isRecord(row) || row.vocabulary !== "event") continue;
    const payload = row.payload;
    if (row.kind === "recipient_health_resolution") {
      const key = payload?.recipient_anchor?.presentation_key;
      if (typeof key !== "string") continue;
      const entry = changes.get(key) ?? { before: null, after: null, rest: 0 };
      const before = Number(payload.transition_start_health);
      const after = Number(payload.health_after_combat_resolution);
      entry.before = Number.isFinite(before) ? before : entry.before;
      entry.after = Number.isFinite(after) ? after : entry.after;
      changes.set(key, entry);
    } else if (row.kind === "health_regenerated") {
      const key = payload?.agent_anchor?.presentation_key;
      const amount = Number(payload?.actual_health_regenerated);
      if (typeof key !== "string" || !Number.isFinite(amount) || amount <= 0) continue;
      const entry = changes.get(key) ?? { before: null, after: null, rest: 0 };
      entry.rest += amount;
      changes.set(key, entry);
    } else if (
      row.kind === "ability_activated" &&
      payload?.ability_component === "ultimate"
    ) {
      const key = payload?.recipient_anchor?.presentation_key;
      if (typeof key === "string") ultimate.add(key);
    } else if (row.kind === "spawn_shield_expired") {
      const key = payload?.agent_anchor?.presentation_key;
      if (typeof key === "string") shieldEnded.add(key);
    }
  }
  return { changes, ultimate, shieldEnded };
}

/**
 * Return a life share in 0..1, or null when it cannot be known.
 *
 * value and maximum are health points; null comes back when either is not finite
 * or maximum is not above 0. Pure.
 *
 * @param {unknown} value
 * @param {unknown} maximum
 * @returns {number | null}
 */
function lifeShare(value, maximum) {
  const current = Number(value);
  const max = Number(maximum);
  if (value === null || value === undefined || !Number.isFinite(current)) return null;
  if (!Number.isFinite(max) || max <= 0) return null;
  return Math.max(0, Math.min(1, current / max));
}

/**
 * @typedef {{
 *   spent: boolean,
 *   readied: boolean,
 *   applied: Set<string>,
 *   expired: Set<string>,
 *   broken: Set<string>,
 *   cleared: Set<string>,
 * }} SealTickFacts
 *   What one audience's own rows say happened to one body this tick, for its
 *   Class Seal: spent (a cooldown_started row: it used its Ultimate, so the
 *   Ultimate was ready at tick start), readied (a cooldown_ready row: its
 *   Ultimate became ready), and the status tokens applied or refreshed, aged out,
 *   broken by damage and cleared by its death.
 * @typedef {{token: string, ticks: number | null, magnitude: number | null}} SealMarkStatus
 *   One status reduced to what the Rune Zones mark rules read (the Statuses
 *   look's MarkStatus): vocabulary token, whole ticks left (null when unknown)
 *   and movement multiplier (null for the token's default).
 */

/**
 * Which SealTickFacts set each status event kind fills (by event kind).
 */
const STATUS_FACT_FIELDS = Object.freeze({
  status_applied: /** @type {const} */ ("applied"),
  status_refreshed_or_extended: /** @type {const} */ ("applied"),
  status_aged_to_zero: /** @type {const} */ ("expired"),
  status_broken_by_damage: /** @type {const} */ ("broken"),
  status_cleared_by_new_death: /** @type {const} */ ("cleared"),
});

/**
 * Collect this audience's Class Seal facts for the tick from its own rows.
 *
 * rows are the same incoming rows tickLifeFacts reads ({kind, vocabulary,
 * payload}); anything that is not an array gives an empty map, and rows that are
 * not "event" vocabulary or have no payload record are skipped. A row belongs to
 * the body named by its payload's recipient_anchor (status rows), else its
 * agent_anchor (cooldown rows), presentation_key. Status tokens come from
 * status_id through statusTokenIdFromCatalogId, as the Statuses look reads them.
 * Return a new map from presentation key to SealTickFacts (only keys some row
 * names). Pure; reads the rows only.
 *
 * @param {unknown} rows
 * @returns {Map<string, SealTickFacts>}
 */
function tickSealFacts(rows) {
  /** @type {Map<string, SealTickFacts>} */
  const facts = new Map();
  for (const row of Array.isArray(rows) ? rows : []) {
    if (!isRecord(row) || row.vocabulary !== "event" || !isRecord(row.payload)) {
      continue;
    }
    const payload = row.payload;
    const anchor = isRecord(payload.recipient_anchor)
      ? payload.recipient_anchor
      : isRecord(payload.agent_anchor)
        ? payload.agent_anchor
        : null;
    const key = anchor?.presentation_key;
    if (typeof key !== "string") continue;
    let entry = facts.get(key);
    if (entry === undefined) {
      entry = {
        spent: false,
        readied: false,
        applied: new Set(),
        expired: new Set(),
        broken: new Set(),
        cleared: new Set(),
      };
      facts.set(key, entry);
    }
    const token =
      typeof payload.status_id === "string"
        ? statusTokenIdFromCatalogId(payload.status_id)
        : null;
    const field =
      typeof row.kind === "string" && Object.hasOwn(STATUS_FACT_FIELDS, row.kind)
        ? STATUS_FACT_FIELDS[/** @type {keyof typeof STATUS_FACT_FIELDS} */ (row.kind)]
        : null;
    if (row.kind === "cooldown_started") {
      entry.spent = true;
    } else if (row.kind === "cooldown_ready") {
      entry.readied = true;
    } else if (field !== null && token !== null) {
      entry[field].add(token);
    }
  }
  return facts;
}

/**
 * Reduce a scene agent's status rows to the Rune Zones mark inputs.
 *
 * statuses are the agent's own status rows (token_id or status_id, duration or
 * remaining_duration, magnitude); a row that is not a record or has no string
 * id is skipped. Return new SealMarkStatus rows in the same order: ticks is the
 * whole ticks left (at least 0) or null, magnitude the finite multiplier or
 * null. Pure.
 *
 * @param {ReadonlyArray<unknown>} statuses
 * @returns {SealMarkStatus[]}
 */
function markStatuses(statuses) {
  /** @type {SealMarkStatus[]} */
  const out = [];
  for (const status of statuses) {
    if (!isRecord(status)) continue;
    const id = status.token_id ?? status.status_id;
    if (typeof id !== "string") continue;
    const ticks = status.duration ?? status.remaining_duration;
    out.push({
      token: statusTokenIdFromCatalogId(id),
      ticks: Number.isFinite(ticks) ? Math.max(0, Math.floor(Number(ticks))) : null,
      magnitude: Number.isFinite(status.magnitude) ? Number(status.magnitude) : null,
    });
  }
  return out;
}

/**
 * Say whether the Statuses look draws at least one Rune Zones mark on a body.
 *
 * statuses are SealMarkStatus rows; spawnShield says whether the body has its
 * Spawn Shield (Blessing of Freedom's wings never show under it). The answer uses
 * the Statuses look's own rules (runeMarksFromStatuses, then runeZones), so it is
 * true exactly when that look draws a crown, roots, veins, runes or wings. Pure.
 *
 * @param {ReadonlyArray<SealMarkStatus>} statuses
 * @param {boolean} spawnShield
 * @returns {boolean}
 */
function marksOnToken(statuses, spawnShield) {
  if (statuses.length === 0) return false;
  const marks = runeZones(runeMarksFromStatuses(statuses, spawnShield));
  return (
    marks.frozen !== null ||
    marks.sapped !== null ||
    marks.roots !== null ||
    marks.withering > 0 ||
    marks.kindled > 0 ||
    marks.unbound
  );
}

/**
 * Rebuild a body's statuses at tick start from its END statuses and its facts.
 *
 * end are the END SealMarkStatus rows; facts (optional) are its SealTickFacts.
 * Rows applied or refreshed this tick are left out (they arrive at Settle), and
 * a status that aged out (1 tick left), broke (0 ticks) or, with withCleared,
 * was cleared by the body's death (ticks unknown) is added back, as the Statuses
 * look rebuilds its START marks. Return a new array. Pure.
 *
 * @param {ReadonlyArray<SealMarkStatus>} end
 * @param {SealTickFacts | undefined} facts
 * @param {boolean} withCleared
 * @returns {SealMarkStatus[]}
 */
function startStatuses(end, facts, withCleared) {
  if (facts === undefined) return [...end];
  /** @param {Set<string>} tokens @param {number | null} ticks */
  const back = (tokens, ticks) =>
    [...tokens].map((token) => ({ token, ticks, magnitude: null }));
  return [
    ...end.filter((status) => !facts.applied.has(status.token)),
    ...back(facts.expired, 1),
    ...back(facts.broken, 0),
    ...(withCleared ? back(facts.cleared, null) : []),
  ];
}

/**
 * Work out one champion record's Class Seal, or null when no seal is drawn.
 *
 * context holds: agent (the audience's scene row), state (the record state),
 * classKey (class CSS key), radius (body radius, CSS px), look ("heraldic" or
 * "plain"), on (the switch values records read), facts (this body's
 * SealTickFacts, or undefined) and known (true when the audience saw the body
 * at tick start, so its earlier look may be shown). Null comes back when Class
 * Seals is off, the radius is under CLASS_SEAL.minRadius or the
 * class has no seal.
 *
 * awake (END) is true when Cooldown Effects is on and the row's Ultimate cooldown
 * (ultimate_cooldown, else ultimate_cooldown_remaining) is known and 0 or less;
 * a missing cooldown gives a dormant seal. dim is true when Status Marks is on
 * and the Statuses look draws a mark on the body (marksOnToken). For a living
 * champion seen at tick start, a spent Ultimate means the seal was awake before
 * Impact (awakeBeat "impact"), a readied one that it was dormant until Settle
 * (awakeBeat "settle"), and the statuses rebuilt by startStatuses give the dim
 * before Settle (dimBeat "settle" when it differs). A returning champion, or a
 * body not seen at start, shows END with no cue. A falling champion's start is
 * its held living look (awake from its cooldown; dim from its statuses before
 * the death, including those cleared by it, when seen at start). A fallen
 * champion's seal is dormant and undimmed: only its ash outline is drawn. Return
 * a frozen ClassSeal. Pure.
 *
 * @param {{
 *   agent: JsonRecord,
 *   state: "living" | "returning" | "falling" | "fallen",
 *   classKey: string,
 *   radius: number,
 *   look: "heraldic" | "plain",
 *   on: Record<string, boolean>,
 *   facts: SealTickFacts | undefined,
 *   known: boolean,
 * }} context
 * @returns {ClassSeal | null}
 */
function classSeal(context) {
  const { agent, state, classKey, radius, on, facts, known } = context;
  if (
    context.look !== "heraldic" ||
    !on.class_seals ||
    !(radius >= CLASS_SEAL.minRadius) ||
    !SEAL_CLASSES.has(classKey)
  ) {
    return null;
  }
  const cooldown = Number.isFinite(agent.ultimate_cooldown)
    ? Number(agent.ultimate_cooldown)
    : Number.isFinite(agent.ultimate_cooldown_remaining)
      ? Number(agent.ultimate_cooldown_remaining)
      : null;
  const ready = on.cooldown_effects && cooldown !== null && cooldown <= 0;
  const shield = finiteNumber(agent.spawn_shield_remaining) > 0;
  const end = markStatuses(asArray(agent.statuses));
  const dimFor = (/** @type {ReadonlyArray<SealMarkStatus>} */ statuses) =>
    on.status_marks && marksOnToken(statuses, shield);
  const quiet = Object.freeze({ awake: false, dim: false });
  /**
   * @param {SealLook} start
   * @param {SealLook} finish
   * @param {"impact" | "settle" | null} awakeBeat
   * @param {"settle" | null} dimBeat
   * @returns {ClassSeal}
   */
  const make = (start, finish, awakeBeat, dimBeat) =>
    Object.freeze({
      classKey: /** @type {SealClass} */ (classKey),
      engraved: radius >= CLASS_SEAL.engraveRadius,
      start,
      end: finish,
      awakeBeat,
      dimBeat,
    });
  if (state === "fallen") return make(quiet, quiet, null, null);
  if (state === "falling") {
    const held = Object.freeze({
      awake: ready,
      dim: dimFor(known ? startStatuses(end, facts, true) : end),
    });
    return make(held, quiet, null, null);
  }
  const finish = Object.freeze({ awake: ready, dim: dimFor(end) });
  if (state !== "living" || !known) return make(finish, finish, null, null);
  const startAwake = facts?.spent
    ? on.cooldown_effects
    : facts?.readied
      ? false
      : finish.awake;
  const start = Object.freeze({
    awake: startAwake,
    dim: dimFor(startStatuses(end, facts, false)),
  });
  return make(
    start.awake === finish.awake && start.dim === finish.dim ? finish : start,
    finish,
    start.awake === finish.awake ? null : facts?.spent ? "impact" : "settle",
    start.dim === finish.dim ? null : "settle",
  );
}

/**
 * Return one projected record per drawable champion in the arena scene.
 *
 * input.scene is the authorized arena scene for this audience; input.transform
 * projects positions and radii (a missing radius is 0.5 world units);
 * input.showSpawnShield (default false) says whether the Spawn Shield switch is
 * on; input.visualFilters (optional) is the page's filter state, read for the
 * switches life_vessels, recent_health_change, low_life_warning,
 * hit_reactions, regeneration_effects, resurrection_effects, fallen_step_aside,
 * class_letters, life_segments, class_seals, cooldown_effects, status_marks and
 * spawn_shield_expiry (null or missing: each at its default);
 * input.beats (optional) carries this tick's plain beat facts for the same
 * audience: beats.rows is `authorizedPresentationIncomingRows(frame)` and
 * beats.view is `beatView(frame, "end", {visualFilters})` (its life entries name
 * the falls this audience saw, with the life before each fall; its returns name
 * the returns, of which only seen ones count). Without beats every champion
 * shows its END look with no beat cue. input.startSeen(presentationKey)
 * (optional) says whether the audience saw that body at tick start, the same test
 * the Statuses look gets; by default true in Oracle View and false in Agent POV.
 * Only a body seen at start shows its seal's earlier look before a beat, or its
 * Spawn Ward until the ward lifts.
 *
 * Rows that are not objects or have no display identity are skipped, in scene
 * order. Each frozen record keeps {kind: "champion", agent, identityKey,
 * layoutSlot, globalSlot, presentationKey, center, radius, controlled, selected,
 * statuses, audience, spawnShieldMechanics, showSpawnShield} (center and radius
 * in CSS pixels; layoutSlot never shown as a public fact) and adds:
 * - classKey and teamKey (CSS keys), geometry (tokenGeometry(radius)).
 * - look: always "heraldic" (Heraldic Tokens is part of the game, not a switch;
 *   the "plain" branches below are kept only until the M12b clean-up);
 *   lifeStyle: "vessel" or "arc" (the ivory life ring when Life Vessels is off).
 * - state: "living", "fallen", "falling" (this audience saw it die this tick: it
 *   keeps its living look until its Fall beat) or "returning" (this audience saw
 *   it return this tick: it is set down at its Return beat).
 * - life: {known, share (END, 0..1 or null), before (share before the beat, or
 *   null when there is no beat cue), change ("loss", "gain", "rest" or "none"),
 *   low (END share at or under 30% and Low Life Warning on), lowBefore}. In
 *   Agent POV missing health gives known false and share null.
 * - hit: null, or {flash (80 ms flash, Hit Reactions on and life lost), shiver
 *   (px; on an Ultimate's or a killing blow's target, by class; 0 for the
 *   Warrior), lethal}.
 * - ward: null, or {ticks, marks (1..3 diamonds)} for a living champion with
 *   shield ticks left while Spawn Shield is on.
 * - wardLift: null, or {fade} (ms: WARD_LIFT.fade with Spawn Shield Expiry on,
 *   WARD_LIFT.cut off) for a "living" champion whose Spawn Shield ended this
 *   tick (its own spawn_shield_expired row in beats.rows) while Spawn Shield is
 *   on, when it has a presentation key, no ward at END and the audience saw it
 *   at tick start: its ward holds until Settle, then the band closes.
 * - aside: null, or {x, y, radius} (CSS px) where a fallen champion's ash coin is
 *   drawn because a standing champion's END center is within 0.6 world units of
 *   it (Fallen Step Aside on). The coin goes straight away from the standing
 *   champion (straight down when the two nearly meet), turning in 30 degree
 *   steps to stay clear of other bodies and coins and inside the map.
 * - seal: null, or the ClassSeal (see classSeal) while Class Seals is on, the
 *   body radius is at least 18 px and the class is known.
 * - switches: {recent, lowLife, regeneration, setDown, letters, segments}.
 *
 * Agent POV: only bodies in the POV scene get a record, and every beat cue comes
 * from the POV team's own visual events, so nothing here depends on an unseen
 * champion. Pure data: no DOM.
 *
 * @param {{
 *   scene?: JsonRecord | null,
 *   transform?: LookTransform | null,
 *   showSpawnShield?: boolean,
 *   visualFilters?: Readonly<Record<string, boolean>> | null,
 *   beats?: {
 *     rows?: ReadonlyArray<unknown>,
 *     view?: {
 *       life?: ReadonlyArray<{presentationKey: string, before: number | null}>,
 *       returns?: ReadonlyArray<{publicAgentId: string, seen: boolean}>,
 *     } | null,
 *   } | null,
 *   startSeen?: ((presentationKey: string) => boolean) | null,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const scene = input.scene;
  const transform = input.transform;
  if (!isRecord(scene) || !transform) {
    return Object.freeze([]);
  }
  const filters = input.visualFilters ?? null;
  const on = Object.fromEntries(SWITCH_IDS.map((id) => [id, switchOn(filters, id)]));
  const agents = asArray(scene.agents).filter(
    (agent) => isRecord(agent) && agentDisplayIdentity(agent) !== null,
  );
  const selection = isRecord(scene.selection) ? scene.selection : {};
  const facts = tickLifeFacts(input.beats?.rows);
  const sealFacts = tickSealFacts(input.beats?.rows);
  const povAudience = scene.audience === "agent_pov";
  const startSeen =
    typeof input.startSeen === "function"
      ? input.startSeen
      : (/** @type {string} */ _key) => !povAudience;
  const beats = input.beats?.view ?? null;
  const falls = new Map(
    (beats?.life ?? []).map((entry) => [entry.presentationKey, entry]),
  );
  const returnedIds = new Set(
    (beats?.returns ?? [])
      .filter((entry) => entry.seen === true)
      .map((entry) => entry.publicAgentId),
  );
  const switches = Object.freeze({
    recent: on.recent_health_change,
    lowLife: on.low_life_warning,
    regeneration: on.regeneration_effects,
    setDown: on.resurrection_effects,
    letters: on.class_letters,
    segments: on.life_segments,
  });
  // Heraldic Tokens is part of the game, not a switch (owner, 3 Oct 2026).
  const look = "heraldic";
  const lifeStyle = look === "heraldic" && on.life_vessels ? "vessel" : "arc";

  /** @type {Array<Record<string, any>>} */
  const drafts = [];
  for (const [index, agent] of agents.entries()) {
    const identityKey = agentDisplayIdentity(agent);
    if (identityKey === null) continue;
    const globalSlot = Number.isInteger(agent.global_slot)
      ? Number(agent.global_slot)
      : null;
    const presentationKey =
      typeof agent.presentation_key === "string" ? agent.presentation_key : null;
    const controlled =
      globalSlot !== null
        ? globalSlot === selection.controlled_global_slot
        : presentationKey === selection.controlled_presentation_key;
    const selected =
      globalSlot !== null
        ? globalSlot === selection.selected_global_slot
        : presentationKey === selection.selected_presentation_key;
    const radius = transform.worldLengthToScreen(finiteNumber(agent.radius, 0.5));
    const classToken = classTokenFromId(agent.class_id);
    const teamToken = teamTokenFromId(agent.team_id);
    const alive = agent.alive === true;
    const fall = presentationKey === null ? undefined : falls.get(presentationKey);
    const returning =
      alive &&
      typeof agent.public_agent_id === "string" &&
      returnedIds.has(agent.public_agent_id);
    const state = alive
      ? returning
        ? "returning"
        : "living"
      : fall !== undefined
        ? "falling"
        : "fallen";
    const share = lifeShare(agent.current_health, agent.max_health);
    const change =
      presentationKey === null ? undefined : facts.changes.get(presentationKey);
    /** @type {number | null} */
    let before = null;
    /** @type {"loss" | "gain" | "rest" | "none"} */
    let kindOfChange = "none";
    if (state === "falling") {
      before = lifeShare(fall?.before, agent.max_health);
      kindOfChange = "loss";
    } else if (state === "living" && share !== null && change !== undefined) {
      const startPoints =
        change.before ??
        (change.rest > 0 ? finiteNumber(agent.current_health) - change.rest : null);
      const startShare = lifeShare(startPoints, agent.max_health);
      if (startShare !== null && Math.abs(startShare - share) > 0.0005) {
        before = startShare;
        kindOfChange =
          share < startShare
            ? "loss"
            : change.before !== null &&
                change.after !== null &&
                change.after > change.before
              ? "gain"
              : "rest";
      }
    }
    const lethal = state === "falling";
    const ultimateTarget =
      presentationKey !== null && facts.ultimate.has(presentationKey);
    const lost = kindOfChange === "loss";
    const shiverPixels =
      lost && (lethal || ultimateTarget)
        ? (SHIVER_PIXELS[
            /** @type {keyof typeof SHIVER_PIXELS} */ (classToken.cssKey)
          ] ?? 1)
        : 0;
    const hit =
      on.hit_reactions && lost
        ? Object.freeze({ flash: true, shiver: shiverPixels, lethal })
        : null;
    const shieldTicks = Number.isInteger(agent.spawn_shield_remaining)
      ? Math.max(0, Number(agent.spawn_shield_remaining))
      : 0;
    const ward =
      input.showSpawnShield === true && alive && shieldTicks > 0
        ? Object.freeze({ ticks: shieldTicks, marks: Math.min(3, shieldTicks) })
        : null;
    const wardLift =
      input.showSpawnShield === true &&
      state === "living" &&
      ward === null &&
      presentationKey !== null &&
      facts.shieldEnded.has(presentationKey) &&
      startSeen(presentationKey) === true
        ? Object.freeze({
            fade: on.spawn_shield_expiry ? WARD_LIFT.fade : WARD_LIFT.cut,
          })
        : null;
    const low = (/** @type {number | null} */ value) =>
      on.low_life_warning && value !== null && value > 0 && value <= LOW_LIFE_SHARE;
    drafts.push({
      kind: "champion",
      agent,
      identityKey,
      layoutSlot: agentLayoutSlot(agent, index),
      globalSlot,
      presentationKey,
      center: screenPoint(agent.position, transform),
      radius,
      controlled,
      selected,
      statuses: asArray(agent.statuses),
      audience: scene.audience,
      spawnShieldMechanics: scene.spawn_shield_mechanics,
      showSpawnShield: input.showSpawnShield === true,
      classKey: classToken.cssKey,
      teamKey: teamToken.cssKey,
      geometry: tokenGeometry(radius),
      look,
      lifeStyle,
      state,
      life: Object.freeze({
        known: share !== null,
        share,
        before,
        change: kindOfChange,
        low: low(share),
        lowBefore: low(before),
      }),
      hit,
      ward,
      wardLift,
      aside: null,
      seal: classSeal({
        agent,
        state,
        classKey: classToken.cssKey,
        radius,
        look,
        on,
        facts: presentationKey === null ? undefined : sealFacts.get(presentationKey),
        known: presentationKey !== null && startSeen(presentationKey) === true,
      }),
      switches,
    });
  }
  if (on.fallen_step_aside) {
    placeAshCoins(drafts, transform);
  }
  return Object.freeze(drafts.map((draft) => Object.freeze(draft)));
}

/**
 * Give each covered corpse its Fallen Step Aside coin spot (mutates drafts).
 *
 * drafts are the records under construction; transform gives the map bounds and
 * the world-to-pixel scale. A corpse ("fallen" or "falling") is covered when a
 * standing champion's END center is within STEP_ASIDE_DISTANCE world units of it;
 * its coin (radius 0.46R) then sits at R + coin + 3 px from that champion's center,
 * first straight away from it (straight down when the centers are under 0.2 units
 * apart), then turning by 30 degree steps until it clears every standing body,
 * every uncovered corpse, every earlier coin and the map edge (first spot if none
 * clears). Stones are not checked. Return undefined.
 *
 * @param {Array<Record<string, any>>} drafts
 * @param {LookTransform} transform
 */
function placeAshCoins(drafts, transform) {
  const unit = Math.max(transform.worldLengthToScreen(1), Number.EPSILON);
  const standing = drafts.filter(
    (draft) => draft.state === "living" || draft.state === "returning",
  );
  const corpses = drafts.filter(
    (draft) => draft.state === "fallen" || draft.state === "falling",
  );
  if (standing.length === 0 || corpses.length === 0) return;
  /** @type {Array<{x: number, y: number, r: number}>} */
  const blockers = standing.map((draft) => ({
    x: draft.center.x,
    y: draft.center.y,
    r: draft.radius,
  }));
  /** @type {Map<Record<string, any>, Record<string, any>>} */
  const coveredBy = new Map();
  for (const corpse of corpses) {
    let best = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const body of standing) {
      const distance = Math.hypot(
        body.center.x - corpse.center.x,
        body.center.y - corpse.center.y,
      );
      if (distance < STEP_ASIDE_DISTANCE * unit && distance < bestDistance) {
        best = body;
        bestDistance = distance;
      }
    }
    if (best === null) {
      blockers.push({ x: corpse.center.x, y: corpse.center.y, r: corpse.radius - 1 });
    } else {
      coveredBy.set(corpse, best);
    }
  }
  const bounds = transform.mapBounds ?? null;
  for (const [corpse, body] of coveredBy) {
    const coin = corpse.geometry.coinRadius;
    const vx = corpse.center.x - body.center.x;
    const vy = corpse.center.y - body.center.y;
    const distance = Math.hypot(vx, vy);
    const base = distance >= 0.2 * unit ? Math.atan2(vy, vx) : Math.PI / 2;
    const need = body.radius + coin + COIN_GAP;
    /** @type {{x: number, y: number} | null} */
    let first = null;
    /** @type {{x: number, y: number} | null} */
    let chosen = null;
    for (const turn of [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]) {
      const angle = base + (turn * Math.PI) / 180;
      const spot = {
        x: body.center.x + need * Math.cos(angle),
        y: body.center.y + need * Math.sin(angle),
      };
      first ??= spot;
      if (
        bounds !== null &&
        (spot.x < bounds.left + coin + 2 ||
          spot.x > bounds.right - coin - 2 ||
          spot.y < bounds.top + coin + 2 ||
          spot.y > bounds.bottom - coin - 2)
      ) {
        continue;
      }
      if (
        blockers.some(
          (blocker) =>
            Math.hypot(spot.x - blocker.x, spot.y - blocker.y) <
            blocker.r + coin + COIN_GAP - 0.01,
        )
      ) {
        continue;
      }
      chosen = spot;
      break;
    }
    const spot = chosen ?? first;
    if (spot !== null) {
      corpse.aside = Object.freeze({ x: spot.x, y: spot.y, radius: coin });
      blockers.push({ x: spot.x, y: spot.y, r: coin });
    }
  }
}

/**
 * Create the retained node bundle for each champion record.
 *
 * records come from records(); doc creates the nodes. Each bundle is a glide
 * wrapper `g.agent-glide` (data-glide-key = the record's identityKey) holding the
 * `g.agent` token: the hit circle `.agent-body`, the life group, the hit flash,
 * the inner group (halo, the Class Seal group `g.champion-seal`, emblem
 * `svg.agent-class-icon`, letter), the team ring
 * and its Team B split, the fallen group and the falling group. The ward group is
 * made later by update only while a ward shows. Nodes have no geometry until update
 * runs. Return the bundles in record order. No tooltip, storage or timer.
 * Export-safe SVG only (flat fills, strokes, opacity and transforms).
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @returns {ChampionNodes[]}
 */
export function build(records, doc) {
  return records.map((record) => {
    const agent = record.agent;
    const wrapper = svgNode(doc, "g", {
      class: "agent-glide",
      "data-glide-key": String(record.identityKey),
    });
    const root = svgNode(doc, "g", {
      class: "agent",
      tabindex: "-1",
      role: "img",
      ...displayIdentityAttributes(agent),
    });
    const body = svgNode(doc, "circle", { class: "agent-body", "data-zone": "body" });
    const life = svgNode(doc, "g", {
      class: "champion-life",
      "data-zone": "health",
      "aria-hidden": "true",
    });
    const flash = svgNode(doc, "circle", {
      class: "champion-flash",
      opacity: 0,
      "aria-hidden": "true",
    });
    const inner = svgNode(doc, "g", { class: "champion-inner", "aria-hidden": "true" });
    const halo = svgNode(doc, "circle", { class: "champion-halo" });
    const seal = svgNode(doc, "g", { class: "champion-seal" });
    const classIcon = createSvgIcon(doc, "unknown", { className: "agent-class-icon" });
    const classLetter = svgNode(doc, "text", { class: "agent-class-letter" });
    inner.append(halo, seal, classIcon, classLetter);
    const teamRing = svgNode(doc, "circle", {
      class: "agent-team-ring",
      "data-zone": "team",
    });
    const ringSplit = svgNode(doc, "circle", {
      class: "champion-ring-split",
      "aria-hidden": "true",
    });
    const fallen = svgNode(doc, "g", {
      class: "champion-fallen",
      "aria-hidden": "true",
    });
    const falling = svgNode(doc, "g", {
      class: "champion-falling",
      "aria-hidden": "true",
    });
    root.append(body, life, flash, inner, teamRing, ringSplit, fallen, falling);
    wrapper.append(root);
    return {
      wrapper,
      root,
      body,
      life,
      flash,
      inner,
      halo,
      seal,
      classIcon,
      classLetter,
      teamRing,
      ringSplit,
      fallen,
      falling,
      shieldRoot: null,
    };
  });
}

/**
 * Refresh each retained bundle from its new record.
 *
 * nodes and records are parallel arrays (bundle i belongs to record i). For each
 * pair, write the token's data attributes and accessible label, then draw the
 * END look for the record's state: the living token (hit circle, life, flash,
 * inner group with its Class Seal or halo, ring) or the Ashen Token (or its
 * Fallen Step Aside coin, with the seal's ash outline on the full token), the
 * held living look of a falling champion, and the ward (made now if needed;
 * blurred and dropped when no ward shows) or, on the tick a seen ward lifts, its
 * held earlier look (drawWard). The seal group is emptied for a dead champion
 * and whenever the record has no seal. The living parts (rings, flash, halo, class
 * emblem and letter) get their position, size and emblem in every state, even
 * while hidden, so a retained bundle holds exactly the DOM that build() plus one
 * update() makes for the same record (a later Return or view switch never shows
 * a stale emblem). Beat-hold attributes are written or removed on
 * every call, so a node never keeps a cue from an earlier tick; a returning
 * champion's Return hold is written on its glide wrapper (data-beat-hold
 * "return", data-beat-from 0, its set-down fade), so the token, the ward and
 * the marks scene.js puts in the wrapper all wait for that beat. Life shares are
 * clipped to 0..1 only for drawing. Return undefined. DOM changes only: no game
 * value, legality or selection changes; the ward's tooltip is registered through
 * `tooltip.js`.
 *
 * @param {ChampionNodes[]} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  for (const [index, bundle] of nodes.entries()) {
    const record = records[index];
    if (record !== undefined) {
      updateChampion(bundle, record);
    }
  }
}

/**
 * Write attributes on node, removing any whose value is null or undefined.
 *
 * A thin alias of setNodeAttributes kept for reading clarity. Return undefined.
 *
 * @param {Element} node
 * @param {Record<string, unknown>} attributes
 */
function put(node, attributes) {
  setNodeAttributes(/** @type {SVGElement} */ (node), attributes);
}

/**
 * Give group exactly the children nodes, keeping every old child that already
 * equals the new one at the same place.
 *
 * This is the smooth-play rule of scene.js's replaceLayer, applied inside a
 * token: a tick whose life, ward or fallen look is unchanged keeps the same
 * nodes (so running animations and the choreographer's holds stay on them),
 * and a tick that changes one part replaces only that part in place. Equal
 * means `isEqualNode` (same tag, attributes and children); a node without
 * isEqualNode (a test double) is always replaced. Extra old children
 * are removed and extra new ones appended. Return undefined. DOM change only;
 * no node here carries a hover card, so none needs a fresh registration.
 *
 * @param {Element} group The token part to fill (life, fallen, falling or ward).
 * @param {ReadonlyArray<Element>} children The new children, bottom to top.
 */
function setChildren(group, children) {
  const old = [...group.children];
  for (const [index, node] of children.entries()) {
    const current = old[index];
    if (current === undefined) {
      group.append(node);
    } else if (
      current !== node &&
      !(typeof current.isEqualNode === "function" && current.isEqualNode(node))
    ) {
      current.replaceWith(node);
    }
  }
  for (const extra of old.slice(children.length)) extra.remove();
}

/**
 * Return beat-hold attributes for a node shown on one side of a beat.
 *
 * kind is "impact", "settle", "fall" or "return"; key is the champion's
 * presentation key in the audience scene (the choreographer times "impact" and
 * "settle" without it, but it is still written); side is "before" (the earlier
 * look, drawn at END opacity 0) or "after" (the END look); fade is the
 * cross-fade in ms (1 for a cut). When key is null, every attribute is null, so
 * the node shows END at once. Pure.
 *
 * @param {"impact" | "settle" | "fall" | "return"} kind
 * @param {string | null} key
 * @param {"before" | "after"} side
 * @param {number} fade
 * @returns {Record<string, string | null>}
 */
function beatAttributes(kind, key, side, fade) {
  if (key === null) {
    return {
      "data-beat-hold": null,
      "data-beat-key": null,
      "data-beat-side": null,
      "data-beat-fade": null,
    };
  }
  return {
    "data-beat-hold": kind,
    "data-beat-key": key,
    "data-beat-side": side,
    "data-beat-fade": String(fade),
  };
}

/**
 * Return the vessel's level y (absolute CSS px) for a life share.
 *
 * cy is the token center's y; vesselRadius the vessel radius; share in 0..1
 * (0 the bottom, 1 the top). Pure.
 *
 * @param {number} cy
 * @param {number} vesselRadius
 * @param {number} share
 */
function levelY(cy, vesselRadius, share) {
  return cy + vesselRadius * (1 - 2 * share);
}

/**
 * Return the half chord of the vessel circle at a level, in CSS px.
 *
 * @param {number} vesselRadius
 * @param {number} offset level y minus the center's y
 */
function halfChord(vesselRadius, offset) {
  return Math.sqrt(Math.max(0, vesselRadius * vesselRadius - offset * offset));
}

/**
 * Return a rounded number string for SVG geometry (3 decimals). Pure.
 *
 * @param {number} value
 */
function n(value) {
  return String(Math.round(value * 1000) / 1000);
}

/**
 * Create the vessel fill below a level: the team field from the bottom up.
 *
 * doc creates the node; center is the token center (CSS px); vesselRadius the
 * vessel radius; share 0..1. A share of 0.001 or less gives null (nothing to
 * fill); 0.999 or more gives a full circle; otherwise a closed chord-and-arc
 * path. className names the node. Export-safe; color comes from CSS.
 *
 * @param {Document} doc
 * @param {{x: number, y: number}} center
 * @param {number} vesselRadius
 * @param {number} share
 * @param {string} className
 * @returns {SVGElement | null}
 */
function fillBelow(doc, center, vesselRadius, share, className) {
  if (share <= 0.001) return null;
  if (share >= 0.999) {
    return svgNode(doc, "circle", {
      class: className,
      cx: n(center.x),
      cy: n(center.y),
      r: n(vesselRadius),
    });
  }
  const y = levelY(center.y, vesselRadius, share);
  const c = halfChord(vesselRadius, y - center.y);
  const large = y < center.y ? 1 : 0;
  return svgNode(doc, "path", {
    class: className,
    d: `M${n(center.x - c)} ${n(y)} L${n(center.x + c)} ${n(y)} A${n(vesselRadius)} ${n(vesselRadius)} 0 ${large} 1 ${n(center.x - c)} ${n(y)} Z`,
  });
}

/**
 * Create the band of the vessel between two levels (lost layer or pour).
 *
 * doc creates the node; center and vesselRadius as fillBelow; top and bottom are
 * shares with top above bottom (otherwise null). className names the node and
 * opacity is its END opacity. Export-safe.
 *
 * @param {Document} doc
 * @param {{x: number, y: number}} center
 * @param {number} vesselRadius
 * @param {number} top
 * @param {number} bottom
 * @param {string} className
 * @param {number} opacity
 * @returns {SVGElement | null}
 */
function bandBetween(doc, center, vesselRadius, top, bottom, className, opacity) {
  if (top - bottom <= 0.001) return null;
  const yTop = levelY(center.y, vesselRadius, top);
  const yBottom = levelY(center.y, vesselRadius, bottom);
  const cTop = halfChord(vesselRadius, yTop - center.y);
  const cBottom = halfChord(vesselRadius, yBottom - center.y);
  const r = n(vesselRadius);
  return svgNode(doc, "path", {
    class: className,
    opacity: n(opacity),
    d: [
      `M${n(center.x - cTop)} ${n(yTop)} L${n(center.x + cTop)} ${n(yTop)}`,
      `A${r} ${r} 0 0 1 ${n(center.x + cBottom)} ${n(yBottom)}`,
      `L${n(center.x - cBottom)} ${n(yBottom)}`,
      `A${r} ${r} 0 0 1 ${n(center.x - cTop)} ${n(yTop)} Z`,
    ].join(" "),
  });
}

/**
 * Create the waterline across the vessel at a level, or null.
 *
 * share must be strictly between 0 and 1, and the chord at least 2 px, or null
 * comes back. dashed draws the faint dashed old level (Recent Health Change);
 * otherwise the 1.25 px ivory line, Cinder when low is true. Export-safe; the
 * dash pattern is a presentation attribute.
 *
 * @param {Document} doc
 * @param {{x: number, y: number}} center
 * @param {number} vesselRadius
 * @param {number} share
 * @param {{dashed?: boolean, low?: boolean}} options
 * @returns {SVGElement | null}
 */
function waterline(doc, center, vesselRadius, share, options) {
  if (!(share > 0 && share < 1)) return null;
  const y = levelY(center.y, vesselRadius, share);
  const c = halfChord(vesselRadius, y - center.y);
  if (c < 2) return null;
  const d = `M${n(center.x - (c - 1))} ${n(y)} H${n(center.x + (c - 1))}`;
  return options.dashed
    ? svgNode(doc, "path", {
        class: "champion-life__old-level",
        d,
        "stroke-width": 1,
        "stroke-dasharray": "2 2",
        opacity: 0.5,
      })
    : svgNode(doc, "path", {
        class: "champion-life__waterline",
        d,
        "stroke-width": 1.25,
        "stroke-linecap": "round",
        opacity: 0.9,
        "data-low": options.low ? "true" : "false",
      });
}

/**
 * Create the ivory life ring (Life Vessels off, or the plain token), or null.
 *
 * share 0..1 fills clockwise from 12 o'clock on the ring at geometry.lifeRadius
 * with width geometry.lifeWidth; low draws it Cinder. A share of 0.0005 or less
 * gives null. Export-safe (pathLength and a dash pattern).
 *
 * @param {Document} doc
 * @param {{x: number, y: number}} center
 * @param {TokenGeometry} geometry
 * @param {number} share
 * @param {boolean} low
 * @returns {SVGElement | null}
 */
function lifeArc(doc, center, geometry, share, low) {
  if (share <= 0.0005) return null;
  const length = Math.min(100, share * 100);
  return svgNode(doc, "circle", {
    class: "champion-life__arc",
    cx: n(center.x),
    cy: n(center.y),
    r: n(geometry.lifeRadius),
    "stroke-width": n(geometry.lifeWidth),
    pathLength: 100,
    "stroke-dasharray": `${n(length)} ${n(100 - length)}`,
    transform: `rotate(-90 ${n(center.x)} ${n(center.y)})`,
    "data-low": low ? "true" : "false",
  });
}

/**
 * Create thin Ink notches across the life ring every 20 health (Life Segments).
 *
 * Only for the ring style and bodies of radius 18 px or more; otherwise null.
 * maximum is the champion's maximum health. Export-safe.
 *
 * @param {Document} doc
 * @param {{x: number, y: number}} center
 * @param {TokenGeometry} geometry
 * @param {number} maximum
 * @returns {SVGElement | null}
 */
function lifeNotches(doc, center, geometry, maximum) {
  const count = Math.round(maximum / 20);
  if (geometry.R < 18 || !(count > 1)) return null;
  const inner = geometry.lifeRadius - geometry.lifeWidth / 2;
  const outer = geometry.lifeRadius + geometry.lifeWidth / 2;
  const parts = [];
  for (let step = 1; step < count; step += 1) {
    const angle = (2 * Math.PI * step) / count;
    const sx = Math.sin(angle);
    const sy = -Math.cos(angle);
    parts.push(
      `M${n(center.x + inner * sx)} ${n(center.y + inner * sy)} L${n(center.x + outer * sx)} ${n(center.y + outer * sy)}`,
    );
  }
  return svgNode(doc, "path", {
    class: "champion-life__notches",
    d: parts.join(" "),
    "stroke-width": 1,
  });
}

/**
 * Create the life drawing of one champion at one moment, as a list of nodes.
 *
 * record gives the style ("vessel" or "arc"), geometry and switches; center is
 * where to draw; share is the life to draw (0..1) or null when unknown (a dotted
 * ring, never a guessed level); low draws the waterline or ring in Cinder;
 * lostFrom (optional) is the earlier share whose loss shows as the lost layer
 * (vessel only, Recent Health Change on). Return the nodes in paint order.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @param {{x: number, y: number}} center
 * @param {number | null} share
 * @param {boolean} low
 * @param {number | null} [lostFrom]
 * @returns {SVGElement[]}
 */
function lifeLook(doc, record, center, share, low, lostFrom = null) {
  /** @type {TokenGeometry} */
  const geometry = record.geometry;
  /** @type {Array<SVGElement | null>} */
  const parts = [];
  if (share === null) {
    parts.push(
      svgNode(doc, "circle", {
        class: "champion-life__unknown",
        cx: n(center.x),
        cy: n(center.y),
        r: n(Math.max(0, geometry.vesselRadius - 1.5)),
        "stroke-width": 1.25,
        "stroke-dasharray": "1.5 2.5",
        "stroke-linecap": "round",
      }),
    );
    return /** @type {SVGElement[]} */ (parts.filter(Boolean));
  }
  if (record.lifeStyle === "vessel") {
    const vessel = geometry.vesselRadius;
    parts.push(fillBelow(doc, center, vessel, share, "champion-life__field"));
    if (lostFrom !== null && lostFrom > share && record.switches.recent) {
      parts.push(
        bandBetween(doc, center, vessel, lostFrom, share, "champion-life__lost", 0.14),
        waterline(doc, center, vessel, lostFrom, { dashed: true }),
      );
    }
    parts.push(waterline(doc, center, vessel, share, { low }));
  } else {
    parts.push(
      svgNode(doc, "circle", {
        class: "champion-life__track",
        cx: n(center.x),
        cy: n(center.y),
        r: n(geometry.lifeRadius),
        "stroke-width": n(geometry.lifeWidth),
      }),
      lifeArc(doc, center, geometry, share, low),
    );
    if (record.switches.segments) {
      parts.push(
        lifeNotches(doc, center, geometry, finiteNumber(record.agent.max_health)),
      );
    }
  }
  return /** @type {SVGElement[]} */ (parts.filter(Boolean));
}

/**
 * Rebuild the life group of a living or returning champion (END plus its cue).
 *
 * With no change (or no known earlier share) the END life is drawn directly.
 * Otherwise an "impact" pair: a "before" group with the earlier level (END
 * opacity 0) and an "after" group with the END level (and the lost layer); a gain
 * or rest also gets a pour band (Dawn Gold for a heal, Moon Dew for Rest; Rest
 * only with Regeneration Effects on) at END opacity 0 with data-beat-pulse 0.45.
 * Return undefined; sets the group's children, keeping unchanged ones
 * (see setChildren).
 *
 * @param {SVGElement} group
 * @param {LookRecord} record
 */
function drawLivingLife(group, record) {
  const doc = group.ownerDocument;
  const { life, center, presentationKey } = record;
  const children = [];
  if (life.known && record.lifeStyle === "vessel") {
    children.push(
      svgNode(doc, "circle", {
        class: "champion-life__groove",
        cx: n(center.x),
        cy: n(center.y),
        r: n(record.geometry.vesselRadius),
      }),
    );
  }
  if (!life.known || life.before === null || life.change === "none") {
    children.push(...lifeLook(doc, record, center, life.share, life.low));
    setChildren(group, children);
    return;
  }
  const before = svgNode(doc, "g", {
    class: "champion-life__before",
    opacity: 0,
    ...beatAttributes("impact", presentationKey, "before", 1),
  });
  before.append(...lifeLook(doc, record, center, life.before, life.lowBefore));
  const after = svgNode(doc, "g", {
    class: "champion-life__after",
    ...beatAttributes("impact", presentationKey, "after", 1),
  });
  after.append(
    ...lifeLook(
      doc,
      record,
      center,
      life.share,
      life.low,
      life.change === "loss" ? life.before : null,
    ),
  );
  children.push(before, after);
  const pours =
    record.lifeStyle === "vessel" &&
    record.switches.recent &&
    (life.change === "gain" ||
      (life.change === "rest" && record.switches.regeneration));
  if (pours) {
    const pour = bandBetween(
      doc,
      center,
      record.geometry.vesselRadius,
      life.share,
      life.before,
      "champion-life__pour",
      0,
    );
    if (pour !== null) {
      put(pour, {
        "data-tone": life.change === "rest" ? "rest" : "heal",
        "data-beat-pulse": "0.45",
        ...beatAttributes("impact", presentationKey, "after", BEAT_LENGTHS.pour),
      });
      children.push(pour);
    }
  }
  setChildren(group, children);
}

/**
 * @typedef {Readonly<{
 *   plate: string,
 *   table: string,
 *   facets: string,
 *   rivets: string,
 *   pairs: ReadonlyArray<string>,
 *   midribs: string,
 *   honing: string,
 *   ring: string,
 * }>} SealPaths
 *   One class's seal paths at one body radius, in token-local CSS px (center
 *   0,0, y down). plate is the seal outline every class has; the others are ""
 *   (pairs empty) unless the class uses them: Mage table (the inner octagon) and
 *   facets (eight lines), Warrior rivets (all eight) and pairs (the four mirror
 *   pairs, bottom pair first), Hunter midribs, Rogue honing line, Priest ring.
 */

/** Seal paths by class and exact body radius (see classSealPaths). */
const SEAL_PATHS = new Map();

/** The seal path cache is cleared when it holds more entries than this. */
const SEAL_PATH_LIMIT = 64;

/**
 * Return the point at radius r and angle degrees clockwise from 12 o'clock.
 *
 * r is in CSS px; the point is [x, y] in token-local px with y down. Pure.
 *
 * @param {number} r
 * @param {number} degrees
 * @returns {[number, number]}
 */
function sealPoint(r, degrees) {
  const t = (degrees * Math.PI) / 180;
  return [r * Math.sin(t), -r * Math.cos(t)];
}

/**
 * Return an SVG path for a full circle at (cx, cy) with radius r (two arcs).
 *
 * Pure; numbers are rounded to 3 decimals.
 *
 * @param {number} cx
 * @param {number} cy
 * @param {number} r
 * @returns {string}
 */
function circlePath(cx, cy, r) {
  return `M${n(cx - r)} ${n(cy)}a${n(r)} ${n(r)} 0 1 0 ${n(2 * r)} 0a${n(r)} ${n(r)} 0 1 0 ${n(-2 * r)} 0Z`;
}

/**
 * Return the Class Seal paths of one class at one body radius.
 *
 * classKey is a SealClass; R is the body radius in CSS px. Every seal fits inside
 * Rs = tokenGeometry(R).sealRadius and is mirror-symmetric about both axes, so
 * nothing on it points (store `sealGeom`, spec Class Seals):
 * - mage: a flat-topped octagon (vertices at 22.5 + 45k degrees), an inner table
 *   octagon at 0.6Rs and eight facet lines joining them.
 * - warrior: a boss circle at 0.95Rs and eight rivets of radius max(0.8, 0.045R)
 *   on it at 22.5 + 45k degrees, in mirror pairs from the bottom.
 * - hunter: a quatrefoil (lobe radius 0.6Rs, cusps at 0.812Rs on the diagonals)
 *   and four midribs from 0.52Rs to 0.86Rs at 12, 3, 6 and 9 o'clock.
 * - rogue: a cushion lozenge (points at Rs on 12, 3, 6 and 9 o'clock, sides
 *   arcs of radius 1.712Rs) and a straight honing lozenge of half-diagonal 0.74Rs.
 * - priest: a 24-point sun (tips at Rs, valleys at 0.84Rs) and a ring at 0.7Rs.
 * Paths are built once per class and exact radius (never a rounded radius, so
 * Oracle View and Agent POV never share an entry that the other drew) and kept in
 * a small cache that is cleared past 64 entries. Return the frozen SealPaths.
 *
 * @param {SealClass} classKey
 * @param {number} R
 * @returns {SealPaths}
 */
export function classSealPaths(classKey, R) {
  const cacheKey = `${classKey}|${R}`;
  const cached = SEAL_PATHS.get(cacheKey);
  if (cached !== undefined) return cached;
  const Rs = tokenGeometry(R).sealRadius;
  const P = (/** @type {[number, number]} */ point) => `${n(point[0])} ${n(point[1])}`;
  /** @type {Record<string, any>} */
  const paths = {
    plate: "",
    table: "",
    facets: "",
    rivets: "",
    pairs: [],
    midribs: "",
    honing: "",
    ring: "",
  };
  if (classKey === "mage") {
    const angles = [0, 1, 2, 3, 4, 5, 6, 7].map((k) => 22.5 + 45 * k);
    const outer = angles.map((degrees) => sealPoint(Rs, degrees));
    const inner = angles.map((degrees) => sealPoint(0.6 * Rs, degrees));
    paths.plate = `M${outer.map(P).join(" L")} Z`;
    paths.table = `M${inner.map(P).join(" L")} Z`;
    paths.facets = inner.map((point, k) => `M${P(point)} L${P(outer[k])}`).join(" ");
  } else if (classKey === "warrior") {
    const boss = 0.95 * Rs;
    const rivet = Math.max(0.8, 0.045 * Math.max(0, finiteNumber(R)));
    paths.plate = circlePath(0, 0, boss);
    paths.pairs = [0, 1, 2, 3].map((pair) => {
      const degrees = 157.5 - 45 * pair;
      const left = sealPoint(boss, degrees);
      const right = sealPoint(boss, 360 - degrees);
      return (
        circlePath(left[0], left[1], rivet) + circlePath(right[0], right[1], rivet)
      );
    });
    paths.rivets = paths.pairs.join("");
  } else if (classKey === "hunter") {
    const lobe = n(0.6 * Rs);
    const arc = `A${lobe} ${lobe} 0`;
    const cusps = [45, 135, 225, 315].map((degrees) => sealPoint(0.812 * Rs, degrees));
    paths.plate = [
      `M0 ${n(-Rs)}`,
      `${arc} 0 1 ${P(cusps[0])}`,
      `${arc} 1 1 ${P(cusps[1])}`,
      `${arc} 1 1 ${P(cusps[2])}`,
      `${arc} 1 1 ${P(cusps[3])}`,
      `${arc} 0 1 0 ${n(-Rs)} Z`,
    ].join(" ");
    paths.midribs = [0, 90, 180, 270]
      .map(
        (degrees) =>
          `M${P(sealPoint(0.52 * Rs, degrees))} L${P(sealPoint(0.86 * Rs, degrees))}`,
      )
      .join(" ");
  } else if (classKey === "rogue") {
    const bulge = n(1.712 * Rs);
    const arc = `A${bulge} ${bulge} 0 0 1`;
    paths.plate = `M0 ${n(-Rs)} ${arc} ${n(Rs)} 0 ${arc} 0 ${n(Rs)} ${arc} ${n(-Rs)} 0 ${arc} 0 ${n(-Rs)} Z`;
    const h = 0.74 * Rs;
    paths.honing = `M0 ${n(-h)} L${n(h)} 0 L0 ${n(h)} L${n(-h)} 0 Z`;
  } else {
    const points = [];
    for (let k = 0; k < 24; k += 1) {
      points.push(sealPoint(k % 2 ? 0.84 * Rs : Rs, 15 * k));
    }
    paths.plate = `M${points.map(P).join(" L")} Z`;
    paths.ring = circlePath(0, 0, 0.7 * Rs);
  }
  paths.pairs = Object.freeze(paths.pairs);
  const frozen = /** @type {SealPaths} */ (Object.freeze(paths));
  if (SEAL_PATHS.size >= SEAL_PATH_LIMIT) SEAL_PATHS.clear();
  SEAL_PATHS.set(cacheKey, frozen);
  return frozen;
}

/**
 * Return the translate transform that puts token-local px at center. Pure.
 *
 * @param {{x: number, y: number}} center
 * @returns {string}
 */
function sealTranslate(center) {
  return `translate(${n(center.x)} ${n(center.y)})`;
}

/**
 * Create one still look of a Class Seal as a detached group.
 *
 * doc creates the nodes; seal is the record's ClassSeal; R the body radius (CSS
 * px); center the token center; look the SealLook to draw; withLitRivets
 * (default true) false leaves out an awakened Warrior's lit rivets (drawSeal then
 * adds them pair by pair). Return `g.champion-seal__look` translated to center,
 * with data-class, data-awake and data-dim, holding (store `seal`):
 * - the plate: class color fill at 8% (20% for an awakened Hunter or Priest)
 *   and a 1 px outline (Warrior 1.8 px) at 45%, 100% awakened, 20% dimmed and
 *   50% awakened and dimmed;
 * - Mage: the table filled at 22% when awakened; the table and facet lines
 *   (0.75 px, 30%, 60% awakened) when engraved and not dimmed;
 * - Warrior, at any size: bronze rivets at 60% (30% dimmed) and, when awakened,
 *   Forge White rivets at 100% (50% dimmed);
 * - Hunter: midribs (0.75 px, 30%, 85% awakened) when engraved and not dimmed;
 * - Rogue: the honing line when engraved and not dimmed: Blade Silver at 30%
 *   while dormant, the class color at 100% when awakened;
 * - Priest: the linen ring (0.75 px, 25%, 60% awakened) when engraved and not
 *   dimmed.
 * Colors come from CSS (the class color through currentColor). Export-safe:
 * flat fills, strokes, opacity and one transform.
 *
 * @param {Document} doc
 * @param {ClassSeal} seal
 * @param {number} R
 * @param {{x: number, y: number}} center
 * @param {SealLook} look
 * @param {boolean} [withLitRivets]
 * @returns {SVGElement}
 */
function sealLookNode(doc, seal, R, center, look, withLitRivets = true) {
  const paths = classSealPaths(seal.classKey, R);
  const { awake, dim } = look;
  const outline = dim ? (awake ? 0.5 : 0.2) : awake ? 1 : 0.45;
  const warm =
    awake && (seal.classKey === "hunter" || seal.classKey === "priest") ? 0.12 : 0;
  const engraved = seal.engraved && !dim;
  const group = svgNode(doc, "g", {
    class: "champion-seal__look",
    transform: sealTranslate(center),
    "data-class": seal.classKey,
    "data-awake": String(awake),
    "data-dim": String(dim),
  });
  /** @type {SVGElement[]} */
  const parts = [
    svgNode(doc, "path", {
      class: "champion-seal__plate",
      d: paths.plate,
      "fill-opacity": n(0.08 + warm),
      "stroke-width": seal.classKey === "warrior" ? 1.8 : 1,
      "stroke-opacity": n(outline),
      "stroke-linejoin": "round",
    }),
  ];
  /** @param {string} d @param {number} opacity @param {string} [className] */
  const engraving = (d, opacity, className = "champion-seal__engraving") =>
    svgNode(doc, "path", {
      class: className,
      d,
      "stroke-width": 0.75,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      opacity: n(opacity),
    });
  if (seal.classKey === "mage") {
    if (awake) {
      parts.push(
        svgNode(doc, "path", {
          class: "champion-seal__table",
          d: paths.table,
          "fill-opacity": 0.22,
        }),
      );
    }
    if (engraved)
      parts.push(engraving(`${paths.table} ${paths.facets}`, awake ? 0.6 : 0.3));
  } else if (seal.classKey === "warrior") {
    // The rivets are the Warrior's silhouette: they stay at any size and only
    // dim to half under a mark.
    const rivetOpacity = dim ? 0.5 : 1;
    parts.push(
      svgNode(doc, "path", {
        class: "champion-seal__rivets",
        d: paths.rivets,
        opacity: n(0.6 * rivetOpacity),
      }),
    );
    if (awake && withLitRivets) {
      parts.push(
        svgNode(doc, "path", {
          class: "champion-seal__rivets-lit",
          d: paths.rivets,
          opacity: n(rivetOpacity),
        }),
      );
    }
  } else if (seal.classKey === "hunter") {
    if (engraved) parts.push(engraving(paths.midribs, awake ? 0.85 : 0.3));
  } else if (seal.classKey === "rogue") {
    if (engraved) {
      parts.push(
        awake
          ? engraving(paths.honing, 1)
          : engraving(paths.honing, 0.3, "champion-seal__honing"),
      );
    }
  } else if (engraved) {
    parts.push(engraving(paths.ring, awake ? 0.6 : 0.25));
  }
  group.append(...parts);
  return group;
}

/**
 * Rebuild the Class Seal group of a living or returning champion.
 *
 * group is the token's `g.champion-seal`; record its champion record. With no
 * seal, or a dead champion, the group is emptied. With no cue (no awakeBeat and no
 * dimBeat, or no presentation key) it holds the END look alone. Otherwise the
 * earlier look is a "before" group (END opacity 0) and the END look an "after"
 * group, timed by the beat clock:
 * - a spent Ultimate: the awakened look goes dark at Impact over 120 ms;
 * - a readied Ultimate: the seal awakens at Settle over 240 ms (with a mark
 *   change on the same tick, both change together);
 * - a mark that comes or goes: the seal steps back or forward at Settle over
 *   200 ms;
 * - a spent Ultimate and a mark change: three looks; the middle one (dormant,
 *   with the start's dim) sits in `g.champion-seal__stage`, shown from Impact
 *   (120 ms) and hidden again at Settle (200 ms).
 * An awakening also plays its class flourish: the Mage's facets flare Starcore
 * (data-beat-pulse, peak 0.95, or 0.33 without engraving, over 160 ms; normal
 * motion only, so a still and reduced motion never show it) and the Warrior's
 * lit rivets come in pair by pair from the bottom. The beat clock has no
 * per-node delay for a hold, so each pair starts at Settle and takes 120, 150,
 * 180 and 210 ms (the store lights each pair over 120 ms, 30 ms after the one
 * below); the Hunter's midribs fade in with the look instead of also growing
 * from 0.6 to 1 (opacity only, as in the store's reduced motion). Return
 * undefined; sets the group's children, keeping unchanged ones (setChildren).
 *
 * @param {SVGElement} group
 * @param {LookRecord} record
 */
function drawSeal(group, record) {
  /** @type {ClassSeal | null} */
  const seal = record.seal;
  const living = record.state === "living" || record.state === "returning";
  if (seal === null || !living) {
    setChildren(group, []);
    return;
  }
  const doc = group.ownerDocument;
  const { center, presentationKey: key } = record;
  const R = record.radius;
  const { start, end, awakeBeat, dimBeat } = seal;
  /** @param {SealLook} look @param {boolean} [lit] */
  const draw = (look, lit = true) => sealLookNode(doc, seal, R, center, look, lit);
  if (key === null || (awakeBeat === null && dimBeat === null)) {
    setChildren(group, [draw(end)]);
    return;
  }
  if (awakeBeat === "impact" && dimBeat === "settle") {
    const before = draw(start);
    put(before, {
      opacity: 0,
      ...beatAttributes("impact", key, "before", CLASS_SEAL.spend),
    });
    const middle = draw({ awake: end.awake, dim: start.dim });
    put(middle, {
      opacity: 0,
      ...beatAttributes("settle", key, "before", CLASS_SEAL.dim),
    });
    const stage = svgNode(doc, "g", {
      class: "champion-seal__stage",
      ...beatAttributes("impact", key, "after", CLASS_SEAL.spend),
    });
    stage.append(middle);
    const after = draw(end);
    put(after, beatAttributes("settle", key, "after", CLASS_SEAL.dim));
    setChildren(group, [before, stage, after]);
    return;
  }
  const awakening = awakeBeat === "settle";
  const kind = awakeBeat === "impact" ? "impact" : "settle";
  const fade =
    awakeBeat === "impact"
      ? CLASS_SEAL.spend
      : awakening
        ? CLASS_SEAL.awaken
        : CLASS_SEAL.dim;
  const staggered = awakening && seal.classKey === "warrior";
  const before = draw(start);
  put(before, { opacity: 0, ...beatAttributes(kind, key, "before", fade) });
  const after = draw(end, !staggered);
  put(after, beatAttributes(kind, key, "after", fade));
  /** @type {SVGElement[]} */
  const children = [before, after];
  const paths = classSealPaths(seal.classKey, R);
  if (awakening && seal.classKey === "mage") {
    const peak = 0.95 * (seal.engraved && !end.dim ? 1 : 0.35);
    children.push(
      svgNode(doc, "path", {
        class: "champion-seal__flare",
        d: paths.facets,
        transform: sealTranslate(center),
        "stroke-width": 1,
        "stroke-linecap": "round",
        opacity: 0,
        "data-beat-pulse": n(peak),
        ...beatAttributes("settle", key, "after", CLASS_SEAL.flare),
      }),
    );
  }
  if (staggered) {
    for (const [pair, d] of paths.pairs.entries()) {
      children.push(
        svgNode(doc, "path", {
          class: "champion-seal__rivets-lit",
          d,
          transform: sealTranslate(center),
          opacity: n(end.dim ? 0.5 : 1),
          ...beatAttributes(
            "settle",
            key,
            "after",
            CLASS_SEAL.rivet + CLASS_SEAL.rivetStep * pair,
          ),
        }),
      );
    }
  }
  setChildren(group, children);
}

/**
 * Create the cold ash outline of a fallen champion's Class Seal, or null.
 *
 * doc creates the node; record is a dead champion's record. Null comes back when
 * the record has no seal or its corpse is drawn as a Fallen Step Aside coin (the
 * small coin carries no seal). Otherwise a `path.champion-seal__ash`: the seal's
 * plate outline in Ash, 1 px at 35%, no fill, centered on the corpse (store
 * `ashen`). Export-safe.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement | null}
 */
function sealAshNode(doc, record) {
  /** @type {ClassSeal | null} */
  const seal = record.seal;
  if (seal === null || record.aside !== null) return null;
  return svgNode(doc, "path", {
    class: "champion-seal__ash",
    d: classSealPaths(seal.classKey, record.radius).plate,
    transform: sealTranslate(record.center),
    "stroke-width": 1,
    "stroke-linejoin": "round",
    opacity: CLASS_SEAL.ashOpacity,
  });
}

/**
 * Rebuild the Ashen Token (or its Fallen Step Aside coin) of a dead champion.
 *
 * group receives the ash rim, Team B's inner rim, the cold ash outline of its
 * Class Seal when the record has one (sealAshNode; never on the coin) and the
 * ash emblem (the icons.js "ash" paint, gray at 55%; a Priest over its ash
 * seal outline shows its cross alone, icons.js onClassSeal); body (the
 * hit circle) becomes the ash disc. record.aside moves the drawing to the coin
 * spot with the coin radius. For a falling champion the ash group fades in at
 * its Fall beat. Return undefined.
 *
 * @param {ChampionNodes} nodes
 * @param {LookRecord} record
 */
function drawFallen(nodes, record) {
  const doc = nodes.root.ownerDocument;
  /** @type {TokenGeometry} */
  const geometry = record.geometry;
  const coin = record.aside;
  const center = coin ?? record.center;
  const radius = coin ? coin.radius : geometry.R;
  put(nodes.body, {
    cx: n(center.x),
    cy: n(center.y),
    r: n(Math.max(0, coin ? radius : radius - 1)),
    "fill-opacity": coin ? 0.92 : 0.75,
    "data-aside": coin ? "true" : null,
  });
  const rim = svgNode(doc, "circle", {
    class: "champion-ashen__rim",
    cx: n(center.x),
    cy: n(center.y),
    r: n(Math.max(0, radius - (coin ? 0.6 : 0.5))),
    "stroke-width": coin ? 1.2 : 1,
    opacity: coin ? 0.7 : 0.45,
  });
  const parts = [rim];
  if (record.teamKey === "team-b" && geometry.doubleBand) {
    parts.push(
      svgNode(doc, "circle", {
        class: "champion-ashen__rim",
        cx: n(center.x),
        cy: n(center.y),
        r: n(Math.max(0, radius - 3)),
        "stroke-width": coin ? 0.9 : 1,
        opacity: coin ? 0.6 : 0.45,
      }),
    );
  }
  const ashSeal = sealAshNode(doc, record);
  if (ashSeal !== null) parts.push(ashSeal);
  const size = coin ? Math.max(9, 0.55 * geometry.emblemSize) : geometry.emblemSize;
  // The "ash" paint (icons.js) is the gray 55% fallen emblem.
  const emblem = createSvgIcon(doc, classTokenFromId(record.agent.class_id).glyphKey, {
    className: "champion-ashen__emblem",
    paint: "ash",
    size,
    onClassSeal: ashSeal !== null,
  });
  put(emblem, {
    x: n(center.x - size / 2),
    y: n(center.y - size / 2),
    width: n(size),
    height: n(size),
  });
  parts.push(emblem);
  setChildren(nodes.fallen, parts);
  put(nodes.fallen, {
    display: null,
    ...(record.state === "falling"
      ? beatAttributes("fall", record.presentationKey, "after", BEAT_LENGTHS.fall)
      : beatAttributes("fall", null, "after", 1)),
  });
}

/**
 * Rebuild a falling champion's held living look (shown until its Fall beat).
 *
 * The group holds a full living token at the corpse spot with its life before
 * the fall (a dotted ring when that life is unknown) and its Class Seal's held
 * look (seal.start, no cue of its own; a Priest's emblem then shows its cross
 * alone) or, without a seal, the halo, drawn at END opacity 0 with the "fall"
 * before-side hold. For any other state the group is emptied. Return undefined.
 *
 * @param {ChampionNodes} nodes
 * @param {LookRecord} record
 */
function drawFalling(nodes, record) {
  if (record.state !== "falling") {
    nodes.falling.replaceChildren();
    put(nodes.falling, { opacity: null, ...beatAttributes("fall", null, "before", 1) });
    return;
  }
  const doc = nodes.root.ownerDocument;
  const { center } = record;
  /** @type {TokenGeometry} */
  const geometry = record.geometry;
  const share = record.life.before;
  const parts = [
    svgNode(doc, "circle", {
      class: "champion-falling__ground",
      cx: n(center.x),
      cy: n(center.y),
      r: n(geometry.R),
    }),
    ...lifeLook(doc, record, center, share, record.life.lowBefore),
  ];
  /** @type {ClassSeal | null} */
  const seal = record.seal;
  if (seal !== null) {
    parts.push(sealLookNode(doc, seal, record.radius, center, seal.start));
  } else if (record.look === "heraldic") {
    parts.push(
      svgNode(doc, "circle", {
        class: "champion-halo",
        cx: n(center.x),
        cy: n(center.y),
        r: n(geometry.haloRadius),
        opacity: 0.14,
      }),
    );
  }
  const emblem = createSvgIcon(doc, classTokenFromId(record.agent.class_id).glyphKey, {
    className: "agent-class-icon",
    size: geometry.emblemSize,
    onClassSeal: seal !== null,
  });
  put(emblem, {
    x: n(center.x - geometry.emblemSize / 2),
    y: n(center.y - geometry.emblemSize / 2),
    width: n(geometry.emblemSize),
    height: n(geometry.emblemSize),
  });
  parts.push(emblem, ...ringNodes(doc, record, center, false));
  setChildren(nodes.falling, parts);
  put(nodes.falling, {
    opacity: 0,
    ...beatAttributes("fall", record.presentationKey, "before", BEAT_LENGTHS.fall),
  });
}

/**
 * Create a team ring (and Team B's Ink split) as detached nodes.
 *
 * dashed draws the ward's ring of 12 dashes (pathLength 12, dashes 0.62 with
 * gaps 0.38, offset 0.31). Return [ring] or [ring, split]. Export-safe.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @param {{x: number, y: number}} center
 * @param {boolean} dashed
 * @returns {SVGElement[]}
 */
function ringNodes(doc, record, center, dashed) {
  /** @type {TokenGeometry} */
  const geometry = record.geometry;
  const dash = dashed
    ? { pathLength: 12, "stroke-dasharray": "0.62 0.38", "stroke-dashoffset": 0.31 }
    : {};
  const ring = svgNode(doc, "circle", {
    class: dashed ? "agent-spawn-shield__ring" : "champion-ring",
    cx: n(center.x),
    cy: n(center.y),
    r: n(geometry.ringRadius),
    "stroke-width": n(geometry.ringWidth),
    ...dash,
  });
  if (record.teamKey !== "team-b" || !geometry.doubleBand) return [ring];
  const split = svgNode(doc, "circle", {
    class: dashed ? "agent-spawn-shield__split" : "champion-ring-split",
    cx: n(center.x),
    cy: n(center.y),
    r: n(geometry.ringRadius),
    "stroke-width": n(geometry.splitWidth),
    ...dash,
  });
  return [ring, split];
}

/**
 * Return the ward's diamond path (up to three silver diamonds on the ring).
 *
 * marks 1 sits at 12 o'clock, 2 at -60 and 60 degrees, 3 at 0, 120 and 240
 * degrees (mirror-symmetric). Each diamond spans the ring width less 1.2 px along
 * the radius and max(1.6, 0.08R) px across. Pure; returns an SVG path string.
 *
 * @param {{x: number, y: number}} center
 * @param {TokenGeometry} geometry
 * @param {number} marks
 */
function diamondPath(center, geometry, marks) {
  /** @type {Record<number, number[]>} */
  const angles = { 1: [0], 2: [-60, 60], 3: [0, 120, 240] };
  const along = geometry.ringWidth / 2 - 0.6;
  const across = Math.max(1.6, 0.08 * geometry.R);
  return (angles[Math.min(3, Math.max(1, marks))] ?? [0])
    .map((degrees) => {
      const t = (degrees * Math.PI) / 180;
      const ux = Math.sin(t);
      const uy = -Math.cos(t);
      const vx = Math.cos(t);
      const vy = Math.sin(t);
      const cx = center.x + ux * geometry.ringRadius;
      const cy = center.y + uy * geometry.ringRadius;
      return [
        `M${n(cx + ux * along)} ${n(cy + uy * along)}`,
        `L${n(cx + vx * across)} ${n(cy + vy * across)}`,
        `L${n(cx - ux * along)} ${n(cy - uy * along)}`,
        `L${n(cx - vx * across)} ${n(cy - vy * across)} Z`,
      ].join(" ");
    })
    .join(" ");
}

/**
 * Create the detached ward group of one champion.
 *
 * doc creates the nodes; agent gives only its display identity. Return the
 * `g.agent-spawn-shield` group (role img, data-zone spawn-shield). Its veil,
 * dashed ring and diamonds are drawn by drawWard.
 *
 * @param {Document} doc
 * @param {JsonRecord} agent
 * @returns {SVGElement}
 */
function createWardRoot(doc, agent) {
  return svgNode(doc, "g", {
    class: "agent-spawn-shield",
    role: "img",
    "data-zone": "spawn-shield",
    ...displayIdentityAttributes(agent),
  });
}

/**
 * Create the detached group that holds a lifting Spawn Ward's earlier look.
 *
 * doc creates the node; record is a champion record with a wardLift. Return a
 * `g.agent-ward-lift` group (data-team, aria-hidden, no pointer input, no
 * tooltip and no display identity, so no shield query or hover finds it). Its
 * veil, dashed ring and diamond are drawn by drawWard.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement}
 */
function createWardLiftRoot(doc, record) {
  return svgNode(doc, "g", {
    class: "agent-ward-lift",
    "aria-hidden": "true",
    "pointer-events": "none",
    "data-team": record.teamKey,
  });
}

/**
 * Drop a bundle's ward group (`nodes.shieldRoot`), blurring it first if it has focus.
 *
 * The bundle then matches a freshly built token that never had a ward
 * (scene.js appends shieldRoot only while it is not null). Return undefined;
 * changes `nodes` and the DOM only.
 *
 * @param {ChampionNodes} nodes
 */
function dropWardRoot(nodes) {
  const shield = nodes.shieldRoot;
  if (shield === null) return;
  if (shield.ownerDocument.activeElement === shield) {
    /** @type {SVGElement & {blur?: () => void}} */ (shield).blur?.();
  }
  shield.remove();
  nodes.shieldRoot = null;
}

/**
 * Draw or hide the Spawn Ward and keep the token's solid ring in step.
 *
 * With a ward (a living champion with shield ticks while Spawn Shield is on) the
 * ward group holds the pale veil (opacity 0.16), the ring of 12 dashes (Team B
 * split too) and the silver diamonds, and the solid team ring is hidden
 * (updateChampion sets the token's 80% opacity). On the tick a seen ward lifts
 * (record.wardLift with a presentation key), `nodes.shieldRoot` is instead a
 * `g.agent-ward-lift` group (createWardLiftRoot) with the same veil, dashed ring
 * and WARD_LIFT.marks diamond, drawn at END opacity 0 with the "settle"
 * before-side hold, and the solid ring (and Team B's split) carry the matching
 * after-side hold, both over wardLift.fade ms: the band closes at Settle
 * (updateChampion brings the token from 80% to full at the same beat). Without
 * either, the group is dropped (dropWardRoot) and the solid ring shows for a
 * living champion; a dead champion never shows the solid ring. A ward group and
 * a lift group are never reused for each other: each is made fresh when it
 * next shows. The ward's tooltip is the shared Spawn Shield explanation
 * (`createSpawnShieldView`); the lift group has none. Return undefined; changes
 * `nodes` and the DOM only.
 *
 * @param {ChampionNodes} nodes
 * @param {LookRecord} record
 */
function drawWard(nodes, record) {
  const doc = nodes.root.ownerDocument;
  const view = record.showSpawnShield
    ? createSpawnShieldView(record.agent, record.spawnShieldMechanics)
    : null;
  const warded = record.ward !== null && view?.active === true;
  const key = record.presentationKey;
  /** @type {{fade: number} | null} */
  const lift = !warded && key !== null ? record.wardLift : null;
  const living = record.state === "living" || record.state === "returning";
  const ringHold =
    lift === null
      ? beatAttributes("settle", null, "after", 1)
      : beatAttributes("settle", key, "after", lift.fade);
  put(nodes.teamRing, { display: warded || !living ? "none" : null, ...ringHold });
  put(nodes.ringSplit, {
    display:
      warded || !living || !(record.teamKey === "team-b" && record.geometry.doubleBand)
        ? "none"
        : null,
    ...ringHold,
  });
  // Keep the group only when it is the kind this tick needs (a ward, a lift
  // or none); otherwise drop it, so the next one is made fresh.
  const wanted = warded ? "ward" : lift !== null ? "lift" : null;
  const current =
    nodes.shieldRoot === null
      ? null
      : nodes.shieldRoot.getAttribute("class") === "agent-ward-lift"
        ? "lift"
        : "ward";
  if (current !== wanted) dropWardRoot(nodes);
  if (lift !== null) {
    if (nodes.shieldRoot === null) {
      nodes.shieldRoot = createWardLiftRoot(doc, record);
    }
    const { center } = record;
    setChildren(nodes.shieldRoot, [
      svgNode(doc, "circle", {
        class: "agent-spawn-shield__veil",
        cx: n(center.x),
        cy: n(center.y),
        r: n(record.geometry.vesselRadius),
        opacity: 0.16,
      }),
      ...ringNodes(doc, record, center, true),
      svgNode(doc, "path", {
        class: "agent-spawn-shield__diamonds",
        d: diamondPath(center, record.geometry, WARD_LIFT.marks),
        "stroke-width": 1,
        "stroke-linejoin": "round",
        "paint-order": "stroke",
      }),
    ]);
    put(nodes.shieldRoot, {
      "data-team": record.teamKey,
      opacity: 0,
      ...beatAttributes("settle", key, "before", lift.fade),
    });
    return;
  }
  if (!warded) return;
  if (nodes.shieldRoot === null) {
    nodes.shieldRoot = createWardRoot(doc, record.agent);
  }
  const shield = nodes.shieldRoot;
  const { center } = record;
  const veil = svgNode(doc, "circle", {
    class: "agent-spawn-shield__veil",
    cx: n(center.x),
    cy: n(center.y),
    r: n(record.geometry.vesselRadius),
    opacity: 0.16,
  });
  const diamonds = svgNode(doc, "path", {
    class: "agent-spawn-shield__diamonds",
    d: diamondPath(center, record.geometry, record.ward.marks),
    "stroke-width": 1,
    "stroke-linejoin": "round",
    "paint-order": "stroke",
  });
  setChildren(shield, [veil, ...ringNodes(doc, record, center, true), diamonds]);
  put(shield, {
    hidden: null,
    display: null,
    tabindex: "0",
    "aria-label": view?.shieldAriaLabel ?? null,
    "data-ticks": String(record.ward.ticks),
    "data-team": record.teamKey,
  });
  shield.removeAttribute("aria-description");
  if (view !== null) registerTooltipOwner(shield, view.descriptor);
}

/**
 * Refresh one champion bundle (the body of update).
 *
 * nodes is the mutable bundle and record its new champion record. See update.
 *
 * @param {ChampionNodes} nodes
 * @param {LookRecord} record
 */
function updateChampion(nodes, record) {
  const { agent, center, controlled, selected } = record;
  /** @type {TokenGeometry} */
  const geometry = record.geometry;
  const doc = nodes.root.ownerDocument;
  const classToken = classTokenFromId(agent.class_id);
  const teamToken = teamTokenFromId(agent.team_id);
  const living = record.state === "living" || record.state === "returning";
  const stepsUntilOutOfCombat =
    Number.isInteger(agent.steps_until_out_of_combat) &&
    agent.steps_until_out_of_combat >= 0
      ? Number(agent.steps_until_out_of_combat)
      : 0;
  const spawnShieldRemaining = Number.isInteger(agent.spawn_shield_remaining)
    ? Math.max(0, Number(agent.spawn_shield_remaining))
    : 0;
  const dataset = nodes.root.dataset;
  dataset.team = teamToken.cssKey;
  dataset.class = classToken.cssKey;
  dataset.alive = String(Boolean(agent.alive));
  dataset.controlled = String(controlled);
  dataset.selected = String(selected);
  dataset.combatStatus = stepsUntilOutOfCombat > 0 ? "IC" : "OOC";
  dataset.stepsUntilOutOfCombat = String(stepsUntilOutOfCombat);
  dataset.spawnShieldRemaining = String(spawnShieldRemaining);
  dataset.respawnedOnIncomingTransition = String(record.state === "returning");
  dataset.look = record.look;
  dataset.lifeStyle = record.lifeStyle;
  dataset.life = record.life.known ? "known" : "unknown";
  dataset.lowLife = String(record.life.low === true);
  dataset.state = record.state;
  const healthText = record.life.known
    ? `health ${formatDisplayNumber(agent.current_health)} of ${formatDisplayNumber(agent.max_health)}`
    : "health unknown";
  nodes.root.setAttribute(
    "aria-label",
    [
      agentIdentity(agent),
      classToken.label,
      teamToken.label,
      healthText,
      agent.alive ? "alive" : "dead",
      record.ward !== null ? "spawn shield" : null,
      controlled ? "controlled actor" : null,
      selected ? "selected target" : null,
    ]
      .filter((part) => part !== null)
      .join(", "),
  );

  // The returning champion is set down at its Return beat. The hold sits on the
  // glide wrapper, so everything that rides with the body (the token, its
  // Spawn Ward and the Rune Zones and rim scene.js puts in the wrapper) stays
  // hidden at the waystone until that beat, as the store's plain waystone does.
  // The token's END opacity is 0.8 under a ward, 1 otherwise. A lifting ward
  // brings the token from 80% up to full at Settle, with the band (drawWard).
  const returning = record.state === "returning" && record.presentationKey !== null;
  put(nodes.wrapper, {
    "data-beat-from": returning ? 0 : null,
    ...beatAttributes(
      "return",
      returning ? record.presentationKey : null,
      "after",
      record.switches.setDown ? BEAT_LENGTHS.setDown : 1,
    ),
  });
  const lifting = record.wardLift !== null && record.presentationKey !== null;
  put(nodes.root, {
    opacity: record.ward !== null ? 0.8 : null,
    "data-beat-from": lifting ? WARD_LIFT.fromOpacity : null,
    ...(lifting
      ? beatAttributes("settle", record.presentationKey, "after", record.wardLift.fade)
      : beatAttributes("return", null, "after", 1)),
  });

  drawFalling(nodes, record);
  // The hidden living parts (rings, flash, halo, emblem, letter) get the same
  // geometry and emblem in every state. A fallen token then holds exactly what
  // a freshly built one holds, so a later Return never shows a stale emblem.
  put(nodes.teamRing, {
    cx: n(center.x),
    cy: n(center.y),
    r: n(geometry.ringRadius),
    "stroke-width": n(geometry.ringWidth),
  });
  put(nodes.ringSplit, {
    cx: n(center.x),
    cy: n(center.y),
    r: n(geometry.ringRadius),
    "stroke-width": n(geometry.splitWidth),
  });

  // Hit flash: END is no flash and no offset.
  put(nodes.flash, {
    cx: n(center.x),
    cy: n(center.y),
    r: n(geometry.vesselRadius),
    opacity: 0,
  });
  // Class halo or Class Seal, and the upright emblem (Class Letters: a smaller
  // emblem over a letter). A seal replaces the halo; a dead champion's seal
  // group is empty (its corpse carries the ash outline instead).
  put(nodes.halo, {
    cx: n(center.x),
    cy: n(center.y),
    r: n(geometry.haloRadius),
    opacity: 0.14,
    display: record.look === "heraldic" && record.seal === null ? null : "none",
  });
  drawSeal(nodes.seal, record);
  const letters = record.switches.letters === true;
  const iconSize = letters ? 0.5 * geometry.R : geometry.emblemSize;
  // icons.js picks a simpler or richer master by size and the page-wide style
  // (Lore Emblems), and a Priest on its Class Seal shows its cross alone, so
  // the emblem is made again only when its class key, size, style or seal
  // changes.
  const onSeal = record.seal !== null;
  const iconKey = `${classToken.glyphKey}@${n(iconSize)}:${defaultIconStyle(classToken.glyphKey)}${onSeal ? ":seal" : ""}`;
  if (nodes.classIcon.getAttribute("data-emblem-key") !== iconKey) {
    const replacement = createSvgIcon(doc, classToken.glyphKey, {
      className: "agent-class-icon",
      size: iconSize,
      onClassSeal: onSeal,
    });
    replacement.setAttribute("data-emblem-key", iconKey);
    nodes.classIcon.replaceWith(replacement);
    nodes.classIcon = replacement;
  }
  const iconY = letters ? -0.18 * geometry.R : 0;
  put(nodes.classIcon, {
    x: n(center.x - iconSize / 2),
    y: n(center.y + iconY - iconSize / 2),
    width: n(iconSize),
    height: n(iconSize),
  });
  put(nodes.classLetter, {
    x: n(center.x),
    y: n(center.y + 0.37 * geometry.R),
    "font-size": n(Math.min(11, Math.max(8, 0.4 * geometry.R))),
    "text-anchor": "middle",
    display: letters ? null : "none",
  });
  nodes.classLetter.textContent = classToken.fallback;
  if (!living) {
    for (const node of [
      nodes.life,
      nodes.flash,
      nodes.inner,
      nodes.teamRing,
      nodes.ringSplit,
    ]) {
      put(node, { display: "none" });
    }
    nodes.life.replaceChildren();
    clearHitCues(nodes);
    drawFallen(nodes, record);
    drawWard(nodes, record);
    return;
  }
  nodes.fallen.replaceChildren();
  put(nodes.fallen, { display: "none", ...beatAttributes("fall", null, "after", 1) });
  for (const node of [nodes.life, nodes.flash, nodes.inner, nodes.teamRing]) {
    put(node, { display: null });
  }
  put(nodes.body, {
    cx: n(center.x),
    cy: n(center.y),
    r: n(geometry.R),
    "fill-opacity": null,
    "data-aside": null,
  });
  drawLivingLife(nodes.life, record);
  // Hitstop shiver for a living champion hit this tick.
  clearHitCues(nodes);
  if (record.hit !== null && record.presentationKey !== null) {
    put(nodes.flash, {
      "data-beat-pulse": "0.08",
      ...beatAttributes("impact", record.presentationKey, "after", BEAT_LENGTHS.flash),
    });
    if (record.hit.shiver > 0) {
      put(nodes.inner, {
        "data-beat-shiver": n(record.hit.shiver),
        "data-beat-shiver-key": record.presentationKey,
        "data-beat-shiver-at": "impact",
        "data-beat-delay": String(BEAT_LENGTHS.shiverDelay),
        "data-beat-fade": String(
          record.hit.lethal ? BEAT_LENGTHS.lethalShiver : BEAT_LENGTHS.shiver,
        ),
      });
    }
  }

  drawWard(nodes, record);
}

/**
 * Remove every hit cue attribute from the flash and the inner group.
 *
 * Return undefined; DOM only.
 *
 * @param {ChampionNodes} nodes
 */
function clearHitCues(nodes) {
  put(nodes.flash, {
    "data-beat-pulse": null,
    ...beatAttributes("impact", null, "after", 1),
  });
  put(nodes.inner, {
    "data-beat-shiver": null,
    "data-beat-shiver-key": null,
    "data-beat-shiver-at": null,
    "data-beat-delay": null,
    "data-beat-fade": null,
  });
}

/**
 * Return the token tooltip for a champion record.
 *
 * Oracle View uses the full agent explanation; Agent POV uses the POV agent
 * explanation (which shows only fields of the POV scene row), with the record's
 * controlled and selected flags. Pure; returns a tooltip descriptor.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  return record.audience === "agent_pov"
    ? explainPovAgent(record.agent, {
        controlled: record.controlled,
        selected: record.selected,
      })
    : explainAgent(record.agent);
}
