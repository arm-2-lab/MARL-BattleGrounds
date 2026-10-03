/**
 * @file The Class Requiems look (Runefield part "Death, Respawn And Kills"):
 * how champions fall and come home, the Fallen panel and the Wave Chip.
 *
 * It owns these drawings:
 * - Fall (`agent_died` pulse): a thin team Fall Ring at 1.0 to 1.3 times the
 *   body radius, the class requiem at the Fall beat (Mage: a Starcore flare and
 *   eight cracks; Warrior: the rim splits; Hunter: umber leaves; Rogue: smoke,
 *   with the blades going last; Priest: eight rays go out one by one), and an
 *   "x2" tag when the host marked the fall `home_fall` (it scored double).
 * - Kill Credit Glint (the `kill_credit_glint` switch, off by default, part of
 *   a fall): at the Fall beat every living champion that shared the kill gets
 *   the same thin ring in its class color, 1.06 times the body radius, fading
 *   from 70% to nothing over 240 ms. It lies in the Strike Band layer (under
 *   the bodies, so a neighbor covers it), at most 4 per fall, never in a paused
 *   still, and only with Death Effects on.
 * - Homecoming (`agent_respawned` pulse): a ring of moonlight narrows onto the
 *   pad at the Return beat, and the old corpse lifts away: a copy of the
 *   corpse as the frame before drew it (its Ashen Token or Fallen Step Aside
 *   coin, Class Seal outline, relic and last Vigil Pip) stays where it fell
 *   from the start of the tick and fades out over 200 ms from the Return beat
 *   (the spec's 900-1100 ms), while a 1 px Ash hairline ring lifts off it
 *   (grows to 1.2 times and fades over 300 ms). Both sit in the Strike Band
 *   layer under the bodies, never in a paused still, and only where the
 *   audience saw the corpse when the tick began.
 * - Corpse marks (durable, `records({part: "corpse-marks", ...})`): an ash
 *   relic of how each class fell and the Vigil Pips (ticks until the team's
 *   next wave, plus one) on every corpse in the scene.
 *
 * Page chrome (DOM, never on or over the arena, never in the battlefield SVG,
 * so never in a PNG export):
 * - Fallen panel: fallenFeedHistory(memory, plan) carries the falls from tick
 *   to tick, then fallenFeedRecords({tick, rows, displayIdOf}) and
 *   buildFallenFeed(doc, records) give the "Fallen · Last 8 Ticks" panel that
 *   sits at the top of the side rail: the last 8 ticks' falls, newest first,
 *   at most 4 rows; each row shows who shared the fall ("+"), a Basic dot or
 *   Ultimate diamond when known, an arrow, the fallen champion and the points.
 * - Wave Chip: buildWaveChip(doc, teamId, beatKey) gives "Team A Returns",
 *   which the scorebug puts under that team's name on its home side.
 * The `death_announcement` and `respawn_wave_occurred` plan pulses therefore
 * draw nothing on the arena: records({event}) gives [] for them, and the plan
 * gives them no anchor and no arena box.
 *
 * Callers: `choreography-painter.js` sends the fall and return pulses here
 * through records({event, plan, transform, filters}), build(records, doc,
 * mount) and update(nodes, records); records({event}) alone still names the
 * fall or return, with no glints. The renderer (`scene.js`) draws the corpse marks through
 * records({part: "corpse-marks", scene, transform, audience, filters}) and
 * build(records, doc). main.js mounts the Fallen panel in #fallen-feed at the
 * top of the side rail, and match-summary.js puts the Wave Chip in the scorebug.
 *
 * Timing: the DOM always holds the END state (Fall Ring at 35%, relics, pips,
 * "x2"; every passing effect at opacity 0). Each moment before END is a Web
 * Animations description pushed to `mount.animationSpecs` with
 * `fill: "backwards"` and no forward fill, timed from the event's Fall or
 * Return beat (`phaseStart`, real milliseconds). Corpse marks of this tick's
 * victim use the choreographer's beat hold (`data-beat-hold="fall"`), so they
 * appear at the Fall beat. No timers and no CSS animations.
 *
 * Agent POV boundary: records reads only the one plan event it is given (with,
 * for glints and a return's old corpse, that audience's own plan), or the
 * authorized arena scene it is given. The old corpse copy needs the plan
 * body's START spot, which Agent POV gives only for a body the team saw when
 * the tick began, so an unseen corpse is never drawn or lifted; its look comes
 * from the Champions look's pure records() fed only those plan bodies. The plan builder already keeps only the falls, returns and killers this
 * audience saw, and the POV scene holds only seen bodies, so an unseen champion
 * gets no requiem, no relic and no row, and an unseen killer is left out of its
 * row with no mark. A glint needs the plan's own named killer of this fall (in
 * Agent POV, one whose blow on the victim the team saw land) standing alive in
 * the plan's END bodies, so no glint ever marks an unseen champion, a killer
 * that left sight, or a blow the team did not see. Nothing here imports a
 * researcher helper or reads a researcher view or match summary. The Fallen
 * panel shows only the rows its caller passes, which must already be limited
 * to the audience (the plan's fallen_feed_row rows are). Returns are public,
 * so the Wave Chip is the same in both views.
 *
 * Export safety: flat fills, strokes, opacity and transforms only; no
 * gradients, masks, filters, url() or images. Colors come from CSS tokens
 * through classes in `src/styles/requiems.css`. Glints rest at opacity 0 and
 * are never built in a settled still, so a PNG export never shows one.
 */
import { classIdentityFromId, teamLabelFromId } from "../agent-identity.js";
import { explainDeathAnnouncement } from "../explanations.js";
import { createSvgIcon } from "../icons.js";
import { createSemanticDescriptor, registerTooltipOwner } from "../tooltip.js";
import { classTokenFromId } from "../vocabulary.js";
import { CLASS_SEAL, records as championRecords, classSealPaths } from "./champions.js";
import {
  animationSpec,
  asArray,
  detachedEventMount,
  effectNode,
  explainChoreographyEvent,
  finiteNumber,
  isRecord,
  mountNodes,
  paintAwareExplanationEvent,
  paintParts,
  screenPoint,
  setEffectAttributes,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").JsonRecord} JsonRecord
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {import("./look-support.js").EventMount} EventMount
 * @typedef {import("./look-support.js").AnimationSpec} AnimationSpec
 * @typedef {"mage" | "warrior" | "hunter" | "rogue" | "priest"} ClassKey
 * @typedef {"a" | "b"} TeamKey
 * @typedef {{x: number, y: number}} Point
 * @typedef {{
 *   tick: number,
 *   victim: JsonRecord,
 *   victimPresentationKey?: string | null,
 *   killers?: ReadonlyArray<JsonRecord> | null,
 *   killingTeamId?: number | null,
 *   points?: number | null,
 *   ultimate?: Readonly<{classId: number, publicAgentId: string}> | null,
 * }} FallenFeedSource
 *   One fall a Fallen panel caller knows about; see fallenFeedRecords.
 * @typedef {Readonly<{
 *   publicAgentId: string | null,
 *   displayId: string | null,
 *   classKey: ClassKey | null,
 *   teamKey: TeamKey | null,
 *   title: string,
 * }>} FallenFeedPerson
 *   A champion named in a Fallen row (shown ID null when the caller gave no
 *   displayIdOf or the ID is unknown).
 * @typedef {Readonly<{
 *   tick: number,
 *   ageTicks: number,
 *   beatKey: string | null,
 *   victim: FallenFeedPerson,
 *   killers: ReadonlyArray<FallenFeedPerson>,
 *   ability: Readonly<{kind: "basic" | "ultimate", classKey: ClassKey | null}> | null,
 *   points: Readonly<{text: string, teamKey: TeamKey | null}> | null,
 *   killingTeamId: 1 | 2 | null,
 *   label: string,
 * }>} FallenFeedRecord
 *   One Fallen panel row; see fallenFeedRecords.
 * @typedef {Readonly<{
 *   publicAgentId: string,
 *   classKey: ClassKey | null,
 *   center: Readonly<Point>,
 * }>} KillCreditGlint
 *   One Kill Credit Glint of a fall: the living champion that shared the kill
 *   (its public ID and class, which picks the ring color) and its END center
 *   in CSS px, where the ring is drawn.
 * @typedef {Readonly<{
 *   center: Readonly<Point>,
 *   radius: number,
 *   aside: Readonly<{x: number, y: number, radius: number}> | null,
 *   classKey: ClassKey,
 *   teamKey: TeamKey,
 *   doubleBand: boolean,
 *   emblemSize: number,
 *   glyphKey: string,
 *   sealClass: string | null,
 *   relic: boolean,
 *   pips: number,
 * }>} OldCorpse
 *   The old corpse of a returning champion, as the frame before the return
 *   drew it (see oldCorpseOf): center is the corpse spot (the champion's START
 *   on the return tick) and radius its body radius, both in CSS px; aside is
 *   its Fallen Step Aside coin {x, y, radius} in CSS px, or null when no
 *   standing champion covered it; classKey and teamKey name the champion;
 *   doubleBand is true for a Team B token with the double ash rim;
 *   emblemSize is the token's emblem box (CSS px) and glyphKey the class
 *   glyph of `icons.js`; sealClass is the class key of the Class Seal whose
 *   ash outline it carried, or null; relic says whether its ash relic showed
 *   (Death Effects on) and pips how many Vigil Pips it carried (1, the last
 *   corpse frame's count, or 0 with Vigil Pips off).
 */

/**
 * Sizes in CSS pixels: the fall and return hit box. The team respawn wave has
 * no arena box: the plan never makes it spatial.
 */
export const FOOTPRINT = Object.freeze({
  lifecycleRing: Object.freeze({ width: 68, height: 68 }),
});

/**
 * Node and animation allowance of one record. A fall draws at most about 20
 * nodes (Priest: eight rays and the ember) and starts at most 12 animations
 * (eight rays, the ember, the ring, its glow and the "x2" tag). With Kill Credit
 * Glint on, a played fall adds at most GLINTS_PER_FALL rings (one node and one
 * animation each) to its Strike Band group, so it may start 16 animations. Its
 * paused still keeps at most 5 nodes (group, hit circle, ring, glow, "x2") and
 * no glint, but a played fall or return is a kept (persistent) row whose whole
 * drawing stays to the end of the tick, so its kept allowance (24) covers the
 * played drawing too: at most 17 nodes in the event group plus 5 in the Strike
 * Band group (the group and four rings). A played return draws 5 nodes in the
 * event group (group, hit circle, ring group, glow, ring) and at most 13 in
 * the Strike Band group (the old corpse: group, disc, two rims, seal outline,
 * an emblem of up to 4 nodes, relic and pips; the lift: group and ring), 18
 * kept nodes and 4 animations in all.
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 30,
  animationsPerRecord: 16,
  persistentNodesPerRecord: 24,
});

/** Most Kill Credit Glints one fall draws (the store's limit). */
export const GLINTS_PER_FALL = 4;

/**
 * Body radius in CSS pixels used when an event carries no `anchorRadius` (a
 * 0.5-unit body on a typical arena). The plan builder may add the true value.
 */
export const DEFAULT_BODY_RADIUS_PX = 14;

/** Most rows the Fallen panel shows, newest first (store shell part: 4). */
export const FEED_MAX_ROWS = 4;

/** Ticks the Fallen panel looks back over: this tick and the 7 before it. */
export const FEED_WINDOW_TICKS = 8;

/**
 * Width in CSS px one character of a shown ID takes in a Fallen row: the IDs
 * are 9 px bold (requiems.css `.fallen-feed__id`), about 0.62 em per digit.
 * The row advances by it, so the next "+" or the points never overlap an ID.
 */
export const FEED_ID_ADVANCE_PX = 5.6;

/** The XHTML namespace, for the page chrome nodes (Fallen panel, Wave Chip). */
const HTML_NAMESPACE = "http://www.w3.org/1999/xhtml";

/**
 * Every paint part this look owns, with the filter that switches it. The
 * death_announcement, fallen_feed_row and respawn_wave parts draw nothing on
 * the arena; they stay registered so their switches (Fallen Feed, Respawn
 * Wave) keep gating the plan rows and the page chrome built here. The
 * kill_credit_glint part is the glint rings a played fall adds (see records).
 */
export const PAINT_PARTS = paintParts([
  [{ surface: "transient", kind: "death_effect" }, "death_effects"],
  [{ surface: "transient", kind: "kill_credit_glint" }, "kill_credit_glint"],
  [{ surface: "transient", kind: "death_announcement" }, "death_announcer"],
  [{ surface: "transient", kind: "fallen_feed_row" }, "death_announcer"],
  [{ surface: "transient", kind: "respawn_wave" }, "respawn_wave"],
  [{ surface: "transient", kind: "resurrection_effect" }, "resurrection_effects"],
  [{ surface: "durable", kind: "requiem_relic" }, "death_effects"],
  [{ surface: "durable", kind: "vigil_pips" }, "vigil_pips"],
]);

/** The pulse meanings this look draws, mapped to their record kinds. */
const RECORD_KIND_BY_SEMANTIC = Object.freeze({
  agent_died: "fall",
  agent_respawned: "homecoming",
});

/**
 * Store requiem times are written on an 880 ms Fall clock: the Fall beat is
 * 880, so a store time x lands at (Fall beat + x - 880) real milliseconds.
 */
const STORE_FALL_CLOCK = 880;

/** Priest rays in the order they go out (degrees clockwise from 12 o'clock). */
const PRIEST_RAY_ORDER = Object.freeze([0, 180, 90, 270, 45, 225, 135, 315]);

/** The store's ease-out (its EASE.out), as a browser easing for keyframes. */
const STORE_EASE_OUT = "cubic-bezier(0, 0, 0.58, 1)";

/** The store's ease-in (its EASE.in), as a browser easing for keyframes. */
const STORE_EASE_IN = "cubic-bezier(0.42, 0, 1, 1)";

/**
 * Return a drawing record list for one plan event or one corpse-marks request.
 *
 * Accepted inputs:
 * - `{event}` where event is one plan event. A "semantic_pulse" whose
 *   cueSemantic is "agent_died" gives a "fall" record and "agent_respawned" a
 *   "homecoming" record. Anything else gives [], including the
 *   "death_announcement" and "respawn_wave_occurred" pulses: the Fallen panel
 *   and the Wave Chip are page chrome (fallenFeedRecords, buildFallenFeed,
 *   buildWaveChip), never drawn on the arena.
 *   A "fall" record holds the event, anchor (CSS px), radius (event
 *   `anchorRadius`, default DEFAULT_BODY_RADIUS_PX), classKey, teamKey,
 *   homeFall (true only when the event carries `homeFall === true`) and
 *   glints, the fall's KillCreditGlint list (see creditGlints; [] unless
 *   filters.kill_credit_glint is true and plan and transform are given).
 *   plan is this audience's choreography plan (its fallen_feed_row killers and
 *   END bodies are read), transform this render's projection and filters the
 *   switch state; all three are optional and only feed the glints.
 *   A "homecoming" record holds the same event, anchor (the pad), radius,
 *   classKey, teamKey and homeFall (false), plus oldCorpse: the returning
 *   champion's OldCorpse, drawn so the old corpse lifts away at the Return
 *   beat (see oldCorpseOf), or null without plan and transform, with
 *   Resurrection Effects off, or when the audience did not see the corpse at
 *   the start of the tick.
 * - `{part: "corpse-marks", scene, transform, audience?, filters?}` gives one
 *   "corpse-marks" record per corpse (`alive === false` or life_state
 *   "corpse") of the given authorized arena scene, with its END centre and
 *   radius in CSS px, classKey, teamKey, presentationKey, relic (true unless
 *   filters.death_effects is false) and pips (ticks until its team's wave plus
 *   one, from scene.respawn_waves countdown_steps; 0 when vigil_pips is off or
 *   the wave is missing). A corpse with neither relic nor pips is skipped.
 *   In Agent POV pass the POV arena scene: it holds only seen bodies.
 * Pure data: no DOM, inputs unchanged; returns a frozen array of frozen records.
 *
 * @param {{
 *   event?: JsonRecord | null,
 *   plan?: JsonRecord | null,
 *   part?: string,
 *   scene?: JsonRecord | null,
 *   transform?: LookTransform | null,
 *   audience?: "researcher" | "agent_pov" | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  if (input?.part === "corpse-marks") {
    return corpseMarkRecords(input);
  }
  const event = input?.event;
  if (event?.kind !== "semantic_pulse") {
    return Object.freeze([]);
  }
  const kind = Object.hasOwn(RECORD_KIND_BY_SEMANTIC, event.cueSemantic)
    ? RECORD_KIND_BY_SEMANTIC[
        /** @type {keyof typeof RECORD_KIND_BY_SEMANTIC} */ (event.cueSemantic)
      ]
    : null;
  if (kind === null) {
    return Object.freeze([]);
  }
  if (kind === "fall" || kind === "homecoming") {
    const identity = isRecord(event.recipientIdentity) ? event.recipientIdentity : {};
    return Object.freeze([
      Object.freeze({
        kind,
        event,
        anchor: event.anchor ?? null,
        radius: bodyRadius(event),
        classKey: classKeyOf(identity.class_id ?? identity.classId),
        teamKey: teamKeyOf(identity.team_id ?? identity.teamId),
        homeFall: kind === "fall" && event.homeFall === true,
        ...(kind === "fall"
          ? { glints: creditGlints(input, event) }
          : { oldCorpse: oldCorpseOf(input, event) }),
      }),
    ]);
  }
  return Object.freeze([]);
}

/**
 * Return the Kill Credit Glints of one fall (pure data).
 *
 * input is the records() input: filters.kill_credit_glint must be true and
 * filters.death_effects not false (the glint is part of the fall), and plan
 * (this audience's choreography plan) and transform (this render's
 * projection) must be given; otherwise the result is []. event is the fall's
 * "agent_died" pulse; its recipientPublicAgentId names the victim.
 *
 * Who glints: the killers of this victim's `fallen_feed_row` row in
 * plan.derivedEvents (the row is kept as data even when Death Announcer is off
 * or the row was dropped for budget). The plan already chose them for this
 * audience: in Oracle View every champion that shared the kill; in Agent POV
 * only a champion whose damaging blow on the victim the team saw land. A
 * killer is drawn only when it stands alive in plan.bodies at END (matched by
 * public ID) and is not marked lastSeen (it left sight this tick), so a glint
 * never marks a body the audience cannot see. Killers keep the plan's public
 * ID order; at most GLINTS_PER_FALL are kept.
 *
 * Returns a frozen array of frozen KillCreditGlint records: publicAgentId,
 * classKey (from the killer's class ID, null when unknown) and center, the
 * body's END spot projected through transform (CSS px). Inputs are not
 * changed; no match summary or researcher data is read.
 *
 * @param {{
 *   plan?: JsonRecord | null,
 *   transform?: LookTransform | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 * }} input
 * @param {JsonRecord} event
 * @returns {ReadonlyArray<KillCreditGlint>}
 */
function creditGlints(input, event) {
  const filters = input?.filters ?? null;
  const plan = input?.plan;
  const transform = input?.transform;
  const victimId =
    typeof event.recipientPublicAgentId === "string"
      ? event.recipientPublicAgentId
      : null;
  if (
    filters?.kill_credit_glint !== true ||
    filters.death_effects === false ||
    !isRecord(plan) ||
    !transform ||
    victimId === null
  ) {
    return Object.freeze([]);
  }
  const row = asArray(plan.derivedEvents).find(
    (candidate) =>
      isRecord(candidate) &&
      candidate.kind === "fallen_feed_row" &&
      isRecord(candidate.victim) &&
      candidate.victim.publicAgentId === victimId,
  );
  if (!row) {
    return Object.freeze([]);
  }
  /** @type {Map<string, JsonRecord>} */
  const bodies = new Map();
  for (const body of asArray(plan.bodies)) {
    if (isRecord(body) && typeof body.publicAgentId === "string") {
      bodies.set(body.publicAgentId, body);
    }
  }
  /** @type {KillCreditGlint[]} */
  const glints = [];
  for (const killer of asArray(row.killers)) {
    if (glints.length >= GLINTS_PER_FALL) break;
    if (!isRecord(killer) || killer.lastSeen === true) continue;
    const publicAgentId =
      typeof killer.publicAgentId === "string" ? killer.publicAgentId : null;
    const body = publicAgentId === null ? undefined : bodies.get(publicAgentId);
    if (
      publicAgentId === null ||
      publicAgentId === victimId ||
      body === undefined ||
      body.alive !== true ||
      !Array.isArray(body.end)
    ) {
      continue;
    }
    const center = screenPoint(body.end, transform);
    glints.push(
      Object.freeze({
        publicAgentId,
        classKey: classKeyOfIdentity(killer),
        center: Object.freeze({ x: center.x, y: center.y }),
      }),
    );
  }
  return Object.freeze(glints);
}

/** The display key of the old corpse in oldCorpseOf's rebuilt start scene. */
const OLD_CORPSE_KEY = "old-corpse";

/**
 * Return the old corpse of one return as the frame before it drew it, or null.
 *
 * input is the records() input: plan (this audience's choreography plan, whose
 * `bodies` give each body's START and END spot), transform (this render's
 * projection) and filters (the switch state) must be given, and
 * filters.resurrection_effects must not be false. event is the return's
 * "agent_respawned" pulse; its recipientPresentationKey names the champion,
 * recipientIdentity its class and team, and anchorRadius its body radius in CSS
 * px (default 0.5 world units).
 *
 * The champion must be a plan body with returned true and a START spot. In
 * Agent POV the plan gives START only for a body the team saw when the tick
 * began, so an unseen corpse gets no copy and nothing marks its spot. The look
 * is the Champions look's own (`champions.js` records): this rebuilds the
 * start of the tick from plan.bodies (each body with a START spot, standing
 * when it was alive then: it fell this tick, or it is alive and did not
 * return) and asks that look for the corpse's record, so the copy has the same
 * size, Class Seal outline and Fallen Step Aside coin as the corpse on the frame
 * before. A body that left the team's sight this tick is not in plan.bodies,
 * so it cannot push the coin aside here. Every body has radius 0.5 world units
 * except the corpse (from anchorRadius). relic follows Death Effects and pips
 * is 1 (the last corpse frame's Vigil count) unless Vigil Pips is off.
 *
 * Returns a frozen OldCorpse (CSS px). Pure: inputs unchanged; no researcher
 * data or match summary is read.
 *
 * @param {{
 *   plan?: JsonRecord | null,
 *   transform?: LookTransform | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 * }} input
 * @param {JsonRecord} event
 * @returns {OldCorpse | null}
 */
function oldCorpseOf(input, event) {
  const plan = input?.plan;
  const transform = input?.transform;
  const filters = input?.filters ?? null;
  const key =
    typeof event.recipientPresentationKey === "string"
      ? event.recipientPresentationKey
      : null;
  if (
    filters?.resurrection_effects === false ||
    !isRecord(plan) ||
    !transform ||
    key === null
  ) {
    return null;
  }
  const bodies = asArray(plan.bodies).filter(isRecord);
  const own = bodies.find((body) => body.presentationKey === key);
  if (own?.returned !== true || !Array.isArray(own.start)) {
    return null;
  }
  const identity = isRecord(event.recipientIdentity) ? event.recipientIdentity : {};
  const classId = Number(identity.class_id ?? identity.classId);
  const teamId = Number(identity.team_id ?? identity.teamId);
  const classKey = classKeyOf(classId);
  const teamKey = teamKeyOf(teamId);
  if (classKey === null || teamKey === null) {
    return null;
  }
  const unit = transform.worldLengthToScreen(1);
  const anchorRadius = Number(event.anchorRadius);
  const radius =
    Number.isFinite(anchorRadius) && anchorRadius > 0 && unit > 0
      ? anchorRadius / unit
      : 0.5;
  /** @type {JsonRecord[]} */
  const agents = [];
  for (const body of bodies) {
    if (!Array.isArray(body.start)) continue;
    if (body === own) {
      agents.push({
        display_key: OLD_CORPSE_KEY,
        class_id: classId,
        team_id: teamId,
        alive: false,
        life_state: "corpse",
        position: body.start,
        radius,
      });
      continue;
    }
    const standing =
      body.died === true || (body.alive === true && body.returned !== true);
    agents.push({
      display_key: `start:${String(body.presentationKey)}`,
      team_id: body.teamId,
      alive: standing,
      life_state: standing ? "alive" : "corpse",
      position: body.start,
      radius: 0.5,
    });
  }
  const corpse = championRecords({
    scene: { agents },
    transform,
    visualFilters: filters,
  }).find((each) => each.identityKey === OLD_CORPSE_KEY);
  if (corpse === undefined) {
    return null;
  }
  return Object.freeze({
    center: Object.freeze({ x: corpse.center.x, y: corpse.center.y }),
    radius: corpse.radius,
    aside: corpse.aside ?? null,
    classKey,
    teamKey,
    doubleBand: teamKey === "b" && corpse.geometry.doubleBand === true,
    emblemSize: corpse.geometry.emblemSize,
    glyphKey: classTokenFromId(classId).glyphKey,
    sealClass: corpse.seal?.classKey ?? null,
    relic: filters?.death_effects !== false,
    pips: filters?.vigil_pips === false ? 0 : 1,
  });
}

/**
 * Return the event's body radius in CSS pixels.
 *
 * event.anchorRadius (a positive finite number) wins; otherwise
 * DEFAULT_BODY_RADIUS_PX. Pure.
 *
 * @param {JsonRecord} event
 * @returns {number}
 */
function bodyRadius(event) {
  const radius = Number(event.anchorRadius);
  return Number.isFinite(radius) && radius > 0 ? radius : DEFAULT_BODY_RADIUS_PX;
}

/**
 * Return the class key ("mage" ... "priest") for a class ID 1-5, or null.
 *
 * @param {unknown} classId
 * @returns {ClassKey | null}
 */
function classKeyOf(classId) {
  const identity = classIdentityFromId(Number(classId));
  return identity === null ? null : /** @type {ClassKey} */ (identity.tokenId);
}

/**
 * Return "a" for team 1, "b" for team 2, else null.
 *
 * @param {unknown} teamId
 * @returns {TeamKey | null}
 */
function teamKeyOf(teamId) {
  const id = Number(teamId);
  return id === 1 ? "a" : id === 2 ? "b" : null;
}

/**
 * Return the team ID (1 or 2) of an identity record, or null.
 *
 * identity may carry teamId, team_id or a teamLabel "Team A"/"Team B". Pure.
 *
 * @param {unknown} identity
 * @returns {1 | 2 | null}
 */
function teamIdOf(identity) {
  if (!isRecord(identity)) return null;
  const id = Number(identity.teamId ?? identity.team_id);
  if (id === 1 || id === 2) return id;
  return identity.teamLabel === teamLabelFromId(1)
    ? 1
    : identity.teamLabel === teamLabelFromId(2)
      ? 2
      : null;
}

/**
 * Return the class key of an identity record, or null.
 *
 * identity may carry classId, class_id or an accent ("mage" ... "priest").
 *
 * @param {unknown} identity
 * @returns {ClassKey | null}
 */
function classKeyOfIdentity(identity) {
  if (!isRecord(identity)) return null;
  const byId = classKeyOf(identity.classId ?? identity.class_id);
  if (byId !== null) return byId;
  return ["mage", "warrior", "hunter", "rogue", "priest"].includes(identity.accent)
    ? /** @type {ClassKey} */ (identity.accent)
    : null;
}

/**
 * Build the corpse-mark records of one scene (relic and Vigil Pips).
 *
 * See records for the input. Return a frozen array; [] without a scene or
 * transform. Pure.
 *
 * @param {{
 *   scene?: JsonRecord | null,
 *   transform?: LookTransform | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
function corpseMarkRecords(input) {
  const { scene, transform } = input;
  if (!isRecord(scene) || !transform) {
    return Object.freeze([]);
  }
  const filters = input.filters ?? {};
  const relicOn = filters.death_effects !== false;
  const pipsOn = filters.vigil_pips !== false;
  /** @type {Map<number, number>} */
  const countdownByTeam = new Map();
  for (const wave of asArray(scene.respawn_waves)) {
    if (isRecord(wave) && Number.isInteger(wave.countdown_steps)) {
      countdownByTeam.set(Number(wave.team_id), Number(wave.countdown_steps));
    }
  }
  /** @type {LookRecord[]} */
  const result = [];
  for (const agent of asArray(scene.agents)) {
    if (!isRecord(agent)) continue;
    const corpse = agent.alive === false || agent.life_state === "corpse";
    const presentationKey =
      typeof agent.presentation_key === "string" ? agent.presentation_key : null;
    const classKey = classKeyOf(agent.class_id);
    const teamKey = teamKeyOf(agent.team_id);
    if (!corpse || presentationKey === null || classKey === null || teamKey === null) {
      continue;
    }
    const countdown = countdownByTeam.get(Number(agent.team_id));
    const pips = pipsOn && countdown !== undefined ? countdown + 1 : 0;
    if (!relicOn && pips === 0) continue;
    result.push(
      Object.freeze({
        kind: "corpse-marks",
        presentationKey,
        publicAgentId: String(agent.public_agent_id ?? ""),
        classKey,
        teamKey,
        center: screenPoint(agent.position, transform),
        radius: transform.worldLengthToScreen(finiteNumber(agent.radius, 0.5)),
        relic: relicOn,
        pips,
      }),
    );
  }
  return Object.freeze(result);
}

/**
 * Draw records: fall and return records into the painter's groups, or corpse
 * marks.
 *
 * records come from records(); doc creates the nodes (it may be null only when
 * records is empty). For event records, mount is the painter's EventMount; when
 * absent, detached settled groups are made. Return the event groups drawn into
 * ([group, connector] or with the route group), or for corpse marks one
 * `g.requiem-corpse-marks` per record. An event without an anchor draws nothing.
 *
 * Side effects: appends SVG nodes (a played fall's Kill Credit Glints and a
 * played return's old corpse and lift ring go into mount.connector, the Strike
 * Band layer under the bodies; everything else into mount.group); registers
 * pointerdown listeners on fall and return hit circles; pushes animation
 * descriptions (fill "backwards") into mount.animationSpecs only in normal or
 * reduced motion when the mount is not settled. A settled mount or motion off
 * draws no glint and no old corpse. Export-safe SVG.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @param {EventMount} [mount]
 * @returns {SVGElement[]}
 */
export function build(records, doc, mount) {
  const record = records[0];
  if (record === undefined) {
    return [];
  }
  if (record.kind === "corpse-marks") {
    return records.map((each) => buildCorpseMarks(doc, each));
  }
  const target = mount ?? detachedEventMount(doc, record.event);
  for (const each of records) {
    const event = each.event;
    if (!event.anchor) {
      continue;
    }
    if (each.kind === "fall") {
      buildFall(doc, target, each);
    } else if (each.kind === "homecoming") {
      buildHomecoming(doc, target, each);
      buildOldCorpse(doc, target, each);
    }
  }
  return mountNodes(target);
}

/**
 * Return how this mount animates: null when it does not (settled or motion
 * off), else {reduced, scale} where scale maps plan times to reduced motion.
 *
 * @param {EventMount} mount
 * @returns {{reduced: boolean, scale: number} | null}
 */
function motionOf(mount) {
  const { options, plan } = mount;
  if (options.settled || options.motionMode === "off") return null;
  if (options.motionMode !== "reduced") return { reduced: false, scale: 1 };
  const beatScale = Number(plan.beats?.reducedScale);
  const phaseScale =
    Number(plan.phases?.reducedTotal ?? 220) /
    Math.max(Number(plan.phases?.total ?? 900), 1);
  return {
    reduced: true,
    scale: Number.isFinite(beatScale) && beatScale > 0 ? beatScale : phaseScale,
  };
}

/**
 * Push one animation description that starts at the beat.
 *
 * mount receives it; element is the target; keyframes are browser keyframes
 * whose first frame is the look before the beat; beatAt is the beat in real ms
 * (already scaled for reduced motion); duration is in ms; part names the piece
 * (unique within the event). Uses fill "backwards" so the DOM's END shows once
 * the animation finishes or is cancelled. Return undefined.
 *
 * @param {EventMount} mount
 * @param {JsonRecord} event
 * @param {Element} element
 * @param {Keyframe[]} keyframes
 * @param {number} beatAt
 * @param {number} duration
 * @param {string} part
 * @param {string} [easing] default "linear".
 */
function pushBeatAnimation(
  mount,
  event,
  element,
  keyframes,
  beatAt,
  duration,
  part,
  easing = "linear",
) {
  if (!(duration > 0)) return;
  mount.animationSpecs.push(
    animationSpec(
      element,
      keyframes,
      { delay: Math.max(0, beatAt), duration, easing, fill: "backwards" },
      mount.plan,
      event,
      part,
    ),
  );
}

/**
 * Build opacity keyframes from [time, opacity] pairs over a window.
 *
 * points are [ms from the window start, opacity] in time order; duration is the
 * window length in ms. Return keyframes with offsets 0-1. Pure.
 *
 * @param {ReadonlyArray<readonly [number, number]>} points
 * @param {number} duration
 * @param {Record<string, string>} [extra] optional property per offset 0 and 1:
 *   {from, to, property} merged into the first and last frames.
 * @returns {Keyframe[]}
 */
function opacityFrames(points, duration, extra) {
  const frames = points.map(([time, opacity]) => ({
    offset: Math.min(1, Math.max(0, time / duration)),
    opacity,
  }));
  if (extra) {
    /** @type {Record<string, any>} */ (frames[0])[extra.property] = extra.from;
    /** @type {Record<string, any>} */ (frames[frames.length - 1])[extra.property] =
      extra.to;
  }
  return frames;
}

/**
 * Format a number for an SVG attribute with at most two decimals. Pure.
 *
 * @param {number} value
 * @returns {string}
 */
function fmt(value) {
  return String(Math.round(value * 100) / 100);
}

/**
 * Return the point at radius r and angle deg (clockwise from 12 o'clock) around
 * the origin, with screen y pointing down. Pure.
 *
 * @param {number} r
 * @param {number} deg
 * @returns {[number, number]}
 */
function polar(r, deg) {
  const rad = (deg * Math.PI) / 180;
  return [r * Math.sin(rad), -r * Math.cos(rad)];
}

/**
 * Return path data for a full circle of radius r around (cx, cy). Pure.
 *
 * @param {number} cx
 * @param {number} cy
 * @param {number} r
 * @returns {string}
 */
function ringPath(cx, cy, r) {
  return `M${fmt(cx - r)} ${fmt(cy)}a${fmt(r)} ${fmt(r)} 0 1 0 ${fmt(2 * r)} 0a${fmt(r)} ${fmt(r)} 0 1 0 ${fmt(-2 * r)} 0`;
}

/**
 * Return path data for a straight segment between two points. Pure.
 *
 * @param {readonly [number, number]} from
 * @param {readonly [number, number]} to
 * @returns {string}
 */
function segmentPath(from, to) {
  return `M${fmt(from[0])} ${fmt(from[1])}L${fmt(to[0])} ${fmt(to[1])}`;
}

/**
 * Return path data for an almond (a leaf fleck) centred on q, its long axis
 * along the radial angle deg, length l and half width w (CSS px). Pure.
 *
 * @param {readonly [number, number]} q
 * @param {number} deg
 * @param {number} l
 * @param {number} w
 * @returns {string}
 */
function almondPath(q, deg, l, w) {
  const rad = (deg * Math.PI) / 180;
  const u = [Math.sin(rad), -Math.cos(rad)];
  const v = [Math.cos(rad), Math.sin(rad)];
  const a = [q[0] - (u[0] * l) / 2, q[1] - (u[1] * l) / 2];
  const b = [q[0] + (u[0] * l) / 2, q[1] + (u[1] * l) / 2];
  return `M${fmt(a[0])} ${fmt(a[1])}Q${fmt(q[0] + v[0] * w)} ${fmt(q[1] + v[1] * w)} ${fmt(b[0])} ${fmt(b[1])}Q${fmt(q[0] - v[0] * w)} ${fmt(q[1] - v[1] * w)} ${fmt(a[0])} ${fmt(a[1])}Z`;
}

/**
 * Return path data for the Warrior's split rim: two half arcs of radius r
 * around the origin, parted by `part` px each. Pure.
 *
 * @param {number} r
 * @param {number} part
 * @returns {string}
 */
function rimHalvesPath(r, part) {
  const right = (/** @type {number} */ deg) => {
    const [x, y] = polar(r, deg);
    return [x + part, y];
  };
  const left = (/** @type {number} */ deg) => {
    const [x, y] = polar(r, deg);
    return [x - part, y];
  };
  const a0 = right(3);
  const a1 = right(177);
  const b0 = left(183);
  const b1 = left(357);
  return `M${fmt(a0[0])} ${fmt(a0[1])}A${fmt(r)} ${fmt(r)} 0 0 1 ${fmt(a1[0])} ${fmt(a1[1])}M${fmt(b0[0])} ${fmt(b0[1])}A${fmt(r)} ${fmt(r)} 0 0 1 ${fmt(b1[0])} ${fmt(b1[1])}`;
}

/**
 * Return path data for one half of the Warrior's rim: the arc of radius r
 * around the origin from 3 to 177 degrees (right, true) or from 183 to 357
 * degrees (left, false), clockwise from 12 o'clock, so the two halves leave a
 * 4-degree gap at 12 and 6 o'clock. Pure.
 *
 * @param {number} r
 * @param {boolean} right
 * @returns {string}
 */
function rimHalfPath(r, right) {
  const [x0, y0] = polar(r, right ? 3 : 183);
  const [x1, y1] = polar(r, right ? 177 : 357);
  return `M${fmt(x0)} ${fmt(y0)}A${fmt(r)} ${fmt(r)} 0 0 1 ${fmt(x1)} ${fmt(y1)}`;
}

/**
 * Append a class emblem (the shared icon) centred on the origin.
 *
 * doc creates it; parent receives it; classKey picks the glyph
 * ("class-<key>"); size is its width and height in CSS px; className is added
 * (a plain icon takes its colour from it); opacity is its END opacity; (cx, cy)
 * is its centre; icon passes the shared icon owner's options: style ("sigil",
 * the default, or "plain" for a one-colour monoline glyph in currentColor),
 * paint ("class", "host", "ash" or "flare"; see `icons.js`) and teamId (1 or
 * 2, for "host"). Return the icon element.
 *
 * @param {Document} doc
 * @param {Element} parent
 * @param {ClassKey | null} classKey
 * @param {number} size
 * @param {string} className
 * @param {number} opacity
 * @param {number} [cx] centre x, default 0.
 * @param {number} [cy] centre y, default 0.
 * @param {{style?: "sigil" | "plain", paint?: "class" | "host" | "ash" | "flare", teamId?: 1 | 2}} [icon]
 *   default {} (the class-coloured sigil).
 */
function appendEmblem(
  doc,
  parent,
  classKey,
  size,
  className,
  opacity,
  cx = 0,
  cy = 0,
  icon = {},
) {
  const node = createSvgIcon(doc, classKey === null ? "unknown" : `class-${classKey}`, {
    ...icon,
    size,
    className,
  });
  node.setAttribute("x", fmt(cx - size / 2));
  node.setAttribute("y", fmt(cy - size / 2));
  node.setAttribute("width", fmt(size));
  node.setAttribute("height", fmt(size));
  node.setAttribute("opacity", fmt(opacity));
  parent.append(node);
  return node;
}

/**
 * Draw one fall: hit circle, Fall Ring, optional "x2" tag and, in motion, the
 * class requiem.
 *
 * doc creates nodes under mount.group; record is a "fall" record. END: the
 * team Fall Ring at 1.3 x radius and 35% opacity (Team B doubled 3 px inside),
 * its glow at 0, the "x2" tag at 70% (home falls only), every requiem piece at
 * 0. In normal motion from the Fall beat (event.phaseStart): the ring grows
 * from 0.77 to 1 and fades from 60% to 35% over 400 ms with a glow; the
 * requiem plays on the store's 880 ms clock (particles only at radius >= 14).
 * Reduced motion: the ring fades in over 220 ms; only the Priest's rays go out
 * together. In both motion modes the record's Kill Credit Glints play from the
 * same beat (see buildCreditGlints). Settled or motion off: END only, with no
 * glint. Return undefined.
 *
 * @param {Document} doc
 * @param {EventMount} mount
 * @param {LookRecord} record
 */
function buildFall(doc, mount, record) {
  const { event, radius, classKey, teamKey } = record;
  const team = teamKey ?? "a";
  const root = effectNode(doc, "g", {
    class: `combat-lifecycle-ring combat-lifecycle-ring--death requiem requiem--team-${team}${classKey ? ` requiem--${classKey}` : ""}`,
    "data-lifecycle-ring": "death",
    "data-requiem-class": classKey ?? "unknown",
    transform: `translate(${fmt(record.anchor.x)} ${fmt(record.anchor.y)})`,
  });
  const hit = effectNode(doc, "circle", {
    class: "combat-lifecycle-ring__hit",
    r: fmt(Math.max(radius * 1.3 + 4, 18)),
  });
  hit.addEventListener("pointerdown", (pointerEvent) => {
    if (pointerEvent.button === 0) {
      pointerEvent.stopPropagation();
    }
  });
  root.append(hit);
  const ringRadius = 1.3 * radius;
  const doubled = team === "b" && radius >= 14;
  const thin = radius < 14 || doubled;
  const glow = effectNode(doc, "path", {
    class: "requiem__fall-glow",
    d: ringPath(0, 0, doubled ? ringRadius - 1.5 : ringRadius),
    "stroke-width": doubled ? 6 : thin ? 3 : 4.5,
    opacity: 0,
  });
  const ring = effectNode(doc, "path", {
    class: "requiem__fall-ring",
    d: doubled
      ? ringPath(0, 0, ringRadius) + ringPath(0, 0, ringRadius - 3)
      : ringPath(0, 0, ringRadius),
    "stroke-width": thin ? 1 : 1.5,
    opacity: 0.35,
  });
  root.append(glow, ring);
  /** @type {SVGElement | null} */
  let x2 = null;
  if (record.homeFall) {
    x2 = effectNode(doc, "text", {
      class: "requiem__x2",
      x: 0,
      y: fmt(ringRadius + 8),
      opacity: 0.7,
    });
    x2.textContent = "×2";
    root.append(x2);
  }
  mount.group.append(root);
  const motion = motionOf(mount);
  if (motion === null) {
    return;
  }
  const beatAt = Number(event.phaseStart ?? 0) * motion.scale;
  buildCreditGlints(doc, mount, record, beatAt, motion.reduced);
  if (motion.reduced) {
    pushBeatAnimation(
      mount,
      event,
      ring,
      [{ opacity: 0 }, { opacity: 0.35 }],
      beatAt,
      220,
      "fall-ring",
    );
    if (x2) {
      pushBeatAnimation(
        mount,
        event,
        x2,
        [{ opacity: 0 }, { opacity: 0.7 }],
        beatAt,
        120,
        "x2",
      );
    }
    if (classKey === "priest") {
      const rays = effectNode(doc, "path", {
        class: "requiem__rays",
        d: PRIEST_RAY_ORDER.map((deg) =>
          segmentPath(polar(0.5 * radius, deg), polar(0.6 * radius, deg)),
        ).join(""),
        "stroke-width": 1.25,
        opacity: 0,
      });
      root.append(rays);
      pushBeatAnimation(
        mount,
        event,
        rays,
        [{ opacity: 0 }, { opacity: 1, offset: 0.001 }, { opacity: 0 }],
        beatAt,
        220,
        "requiem-rays",
      );
    }
    return;
  }
  pushBeatAnimation(
    mount,
    event,
    ring,
    [
      { offset: 0, opacity: 0, transform: "scale(0.77)" },
      { offset: 0.125, opacity: 0.57, transform: "scale(0.85)" },
      { offset: 1, opacity: 0.35, transform: "scale(1)" },
    ],
    beatAt,
    400,
    "fall-ring",
    "ease-out",
  );
  pushBeatAnimation(
    mount,
    event,
    glow,
    [
      { offset: 0, opacity: 0, transform: "scale(0.77)" },
      { offset: 0.125, opacity: 0.18, transform: "scale(0.85)" },
      { offset: 1, opacity: 0, transform: "scale(1)" },
    ],
    beatAt,
    400,
    "fall-glow",
    "ease-out",
  );
  if (x2) {
    pushBeatAnimation(
      mount,
      event,
      x2,
      [{ opacity: 0 }, { opacity: 0.7 }],
      beatAt,
      120,
      "x2",
    );
  }
  buildRequiem(doc, mount, root, record, beatAt);
}

/**
 * Draw the class requiem pieces (END opacity 0) and their beat animations.
 *
 * doc creates nodes under root (centred on the fall); record gives classKey
 * and radius R (CSS px); beatAt is the Fall beat in real ms. Times follow the
 * store's 880 ms Fall clock (a store time x plays at beatAt + x - 880), and
 * each piece follows the V5 Class Requiems spec (specs/death-respawn-and-kills
 * .json, "Class Requiems"); where the store code differs, the spec wins:
 * - Mage, The Star Goes Out: a Starcore flare of the emblem (0 to 100% over
 *   880-940, out over 940-1060); eight 0.75 px cracks from 0.1R to 0.62R
 *   between the star's rays that grow from the centre (scale 0 to 1, ease-out)
 *   over 940-1000, hold at 80% and fade over 1180-1480; six Aether Violet motes
 *   (r max(0.8, 0.05R) at 0.5R) that scale about the centre from 1 to 2.2 and
 *   fade from 70% over 940-1440.
 * - Warrior, The Shield Splits: a 1 px crack from 0.62R above the centre to
 *   0.62R below that grows top to bottom over 880-1000 and fades over
 *   1160-1400; the bronze rim's two halves at 0.56R part by 0.75 px each over
 *   1000-1160 (ease-in) and fade to half by 1500, where the ash relic takes
 *   over; crack and halves settle to scale 0.97 over 1000-1120; the ash dust
 *   ring widens from 1.0R to 1.2R and fades from 40% over 1040-1480.
 * - Hunter, Leaves On Frost: an Umber copy of the emblem (in over 880-1030,
 *   out over 1130-1380); five leaves (0.11R by 0.05R, at 0.55R) and five
 *   Rime dots (r 0.8 px) that scale about the centre from 1 to 2.1 and fade
 *   (leaves from 80%, rime from 70%) over 940-1440.
 * - Rogue, Into Smoke: the smoke ring (880-1000) and disc (1000-1120) creep
 *   in at 70%; three puffs (r 0.22R at 0.9R) each swell about its own centre
 *   from 0.6 to 1.3 and fade from 60% over 900-1300; the jade blades hold
 *   until 1180 and go last, with the smoke, by 1380.
 * - Priest, The Last Candle: eight rays go out one by one, 60 ms apart and
 *   each over 120 ms, starting at 880 in the order 12, 6, 3, 9, 1:30, 7:30,
 *   4:30 and 10:30 o'clock; the ember glows over 1300-1380, holds to 1440 and
 *   dims by 1680.
 * Particles (motes, leaves, rime, puffs, dust) are drawn only when R >= 14.
 * Every animation starts hidden (opacity 0) so nothing shows before the
 * beat, and uses fill "backwards", so the DOM's END (opacity 0) shows after.
 * Return undefined.
 *
 * @param {Document} doc
 * @param {EventMount} mount
 * @param {Element} root
 * @param {LookRecord} record
 * @param {number} beatAt
 */
function buildRequiem(doc, mount, root, record, beatAt) {
  const { event, classKey } = record;
  const radius = Number(record.radius);
  const particles = radius >= 14;
  const emblemSize = Math.min(26, Math.max(9, 0.8 * radius));
  /**
   * Append a path with class, d and stroke width at END opacity 0.
   *
   * @param {string} className
   * @param {string} d
   * @param {number} [strokeWidth]
   * @param {Element} [parent] default root.
   */
  const piece = (className, d, strokeWidth, parent = root) => {
    const node = effectNode(doc, "path", {
      class: className,
      d,
      ...(strokeWidth === undefined ? {} : { "stroke-width": fmt(strokeWidth) }),
      opacity: 0,
    });
    parent.append(node);
    return node;
  };
  /**
   * Animate node's opacity through store-clock [time, opacity] points.
   *
   * @param {Element} node
   * @param {ReadonlyArray<readonly [number, number]>} points store times (ms).
   * @param {string} part
   */
  const play = (node, points, part) => {
    const start = points[0][0];
    const end = points[points.length - 1][0];
    const relative = points.map(
      ([time, opacity]) => /** @type {const} */ ([time - start, opacity]),
    );
    pushBeatAnimation(
      mount,
      event,
      node,
      opacityFrames([[0, 0], ...relative], end - start),
      beatAt + (start - STORE_FALL_CLOCK),
      end - start,
      part,
    );
  };
  /**
   * Animate node through store-clock keyframes [time, properties], in time
   * order. properties holds browser keyframe values (opacity, transform and
   * an optional easing for the interval that starts there). The first frame
   * must hide the node (opacity 0).
   *
   * @param {Element} node
   * @param {ReadonlyArray<readonly [number, Keyframe]>} frames store times (ms).
   * @param {string} part
   */
  const animate = (node, frames, part) => {
    const start = frames[0][0];
    const end = frames[frames.length - 1][0];
    pushBeatAnimation(
      mount,
      event,
      node,
      frames.map(([time, values]) => ({
        offset: (time - start) / (end - start),
        ...values,
      })),
      beatAt + (start - STORE_FALL_CLOCK),
      end - start,
      part,
    );
  };
  /**
   * Sample a drift (scale about the centre with the store's ease-out) and a
   * fade (the store's ease-in) over store times from..to: six keyframes at
   * p = 0, 0.2, 0.4, 0.6, 0.8 and 1, with opacity peak x (1 - easeIn(p)) x
   * ramp and scale 1 + (scaleTo - 1) x easeOut(p) (each piece's own scale
   * from scaleFrom first). rampMs (default 0) eases the opacity in from 0
   * over its first rampMs ms, as the store's leaves do.
   *
   * @param {number} from
   * @param {number} to
   * @param {number} peak
   * @param {number} scaleFrom
   * @param {number} scaleTo
   * @param {number} [rampMs]
   * @returns {Array<readonly [number, Keyframe]>}
   */
  const drift = (from, to, peak, scaleFrom, scaleTo, rampMs = 0) =>
    [0, 0.2, 0.4, 0.6, 0.8, 1].map((p) => {
      const time = from + p * (to - from);
      const ramp = rampMs > 0 ? Math.min(1, (time - from) / rampMs) : 1;
      return /** @type {const} */ ([
        time,
        {
          opacity: Math.round(peak * ramp * (1 - storeEaseIn(p)) * 1000) / 1000,
          transform: `scale(${fmt(scaleFrom + (scaleTo - scaleFrom) * storeEaseOut(p))})`,
        },
      ]);
    });
  if (classKey === "mage") {
    const flare = appendEmblem(
      doc,
      root,
      "mage",
      emblemSize,
      "requiem__emblem requiem__emblem--starcore",
      0,
      0,
      0,
      { paint: "flare" },
    );
    play(
      flare,
      [
        [880, 0],
        [940, 1],
        [1060, 0],
      ],
      "requiem-flare",
    );
    const cracks = piece(
      "requiem__cracks",
      Array.from({ length: 8 }, (_, k) =>
        segmentPath(
          polar(0.1 * radius, 22.5 + 45 * k),
          polar(0.62 * radius, 22.5 + 45 * k),
        ),
      ).join(""),
      0.75,
    );
    animate(
      cracks,
      [
        [880, { opacity: 0, transform: "scale(0)" }],
        [939, { opacity: 0, transform: "scale(0)" }],
        [940, { opacity: 0.8, transform: "scale(0)", easing: STORE_EASE_OUT }],
        [1000, { opacity: 0.8, transform: "scale(1)" }],
        [1180, { opacity: 0.8, transform: "scale(1)" }],
        [1480, { opacity: 0, transform: "scale(1)" }],
      ],
      "requiem-cracks",
    );
    if (particles) {
      const motes = piece(
        "requiem__motes",
        Array.from({ length: 6 }, (_, k) => {
          const [x, y] = polar(0.5 * radius, 30 + 60 * k);
          return ringPath(x, y, Math.max(0.8, 0.05 * radius));
        }).join(""),
      );
      animate(
        motes,
        [
          [880, { opacity: 0, transform: "scale(1)" }],
          [939, { opacity: 0, transform: "scale(1)" }],
          ...drift(940, 1440, 0.7, 1, 2.2),
        ],
        "requiem-motes",
      );
    }
  } else if (classKey === "warrior") {
    // The crack and the rim halves settle together (scale 1 to 0.97).
    const shield = effectNode(doc, "g", { class: "requiem__shield" });
    root.append(shield);
    animate(
      shield,
      [
        [880, { opacity: 0, transform: "scale(1)" }],
        [881, { opacity: 1, transform: "scale(1)" }],
        [1000, { opacity: 1, transform: "scale(1)" }],
        [1120, { opacity: 1, transform: "scale(0.97)" }],
        [1501, { opacity: 1, transform: "scale(0.97)" }],
      ],
      "requiem-shield",
    );
    /** @type {ReadonlyArray<readonly [string, number]>} */
    const halves = [
      ["right", 1],
      ["left", -1],
    ];
    for (const [side, sign] of halves) {
      const half = piece(
        `requiem__rim requiem__rim--${side}`,
        rimHalfPath(0.56 * radius, side === "right"),
        2,
        shield,
      );
      animate(
        half,
        [
          [880, { opacity: 0, transform: "translate(0px, 0px)" }],
          [930, { opacity: 1, transform: "translate(0px, 0px)" }],
          [
            1000,
            {
              opacity: 0.93,
              transform: "translate(0px, 0px)",
              easing: STORE_EASE_IN,
            },
          ],
          [1160, { opacity: 0.84, transform: `translate(${fmt(0.75 * sign)}px, 0px)` }],
          [1500, { opacity: 0.5, transform: `translate(${fmt(0.75 * sign)}px, 0px)` }],
          [1501, { opacity: 0, transform: `translate(${fmt(0.75 * sign)}px, 0px)` }],
        ],
        `requiem-rim-${side}`,
      );
    }
    const crack = piece(
      "requiem__crack",
      segmentPath([0, -0.62 * radius], [0, 0.62 * radius]),
      1,
      shield,
    );
    animate(
      crack,
      [
        [880, { opacity: 0, transform: "scale(1, 0)" }],
        [881, { opacity: 1, transform: "scale(1, 0)" }],
        [1000, { opacity: 1, transform: "scale(1, 1)" }],
        [1160, { opacity: 1, transform: "scale(1, 1)" }],
        [1400, { opacity: 0, transform: "scale(1, 1)" }],
      ],
      "requiem-crack",
    );
    if (particles) {
      const dust = piece("requiem__ash-ring", ringPath(0, 0, 1.2 * radius), 1);
      animate(
        dust,
        [
          [1040, { opacity: 0, transform: "scale(0.83)" }],
          [1041, { opacity: 0.4, transform: "scale(0.83)" }],
          [1480, { opacity: 0, transform: "scale(1)" }],
        ],
        "requiem-ash",
      );
    }
  } else if (classKey === "hunter") {
    const leaf = appendEmblem(
      doc,
      root,
      "hunter",
      emblemSize,
      "requiem__emblem requiem__emblem--umber",
      0,
      0,
      0,
      { style: "plain" },
    );
    play(
      leaf,
      [
        [880, 0],
        [1030, 1],
        [1130, 1],
        [1380, 0],
      ],
      "requiem-umber",
    );
    if (particles) {
      const reach = 0.55 * radius;
      const leaves = piece(
        "requiem__leaves",
        Array.from({ length: 5 }, (_, i) =>
          almondPath(polar(reach, 72 * i), 72 * i, 0.11 * radius, 0.05 * radius),
        ).join(""),
      );
      animate(
        leaves,
        [
          [880, { opacity: 0, transform: "scale(1)" }],
          ...drift(940, 1440, 0.8, 1, 2.1, 50),
        ],
        "requiem-leaves",
      );
      const rime = piece(
        "requiem__rime",
        Array.from({ length: 5 }, (_, i) => {
          const [x, y] = polar(reach, 36 + 72 * i);
          return ringPath(x, y, 0.8);
        }).join(""),
      );
      animate(
        rime,
        [
          [880, { opacity: 0, transform: "scale(1)" }],
          ...drift(940, 1440, 0.7, 1, 2.1, 50),
        ],
        "requiem-rime",
      );
    }
  } else if (classKey === "rogue") {
    const ring = piece(
      "requiem__smoke-ring",
      ringPath(0, 0, 0.82 * radius),
      0.2 * radius,
    );
    play(
      ring,
      [
        [880, 0],
        [1000, 0.7],
        [1180, 0.7],
        [1380, 0],
      ],
      "requiem-smoke-ring",
    );
    const disc = piece("requiem__smoke-disc", ringPath(0, 0, 0.62 * radius));
    play(
      disc,
      [
        [880, 0],
        [1000, 0],
        [1120, 0.7],
        [1180, 0.7],
        [1380, 0],
      ],
      "requiem-smoke",
    );
    if (particles) {
      // Each puff swells about its own centre (fill-box origin, requiems.css).
      [0, 120, 240].forEach((deg, index) => {
        const [x, y] = polar(0.9 * radius, deg);
        const puff = piece(
          "requiem__puffs requiem__puff",
          ringPath(x, y, 0.22 * radius),
          1,
        );
        animate(
          puff,
          [
            [880, { opacity: 0, transform: "scale(0.6)" }],
            ...drift(900, 1300, 0.6, 0.6, 1.3, 80),
          ],
          `requiem-puff-${index}`,
        );
      });
    }
    // The blades go last: the emblem stays until the smoke has gone.
    const blades = appendEmblem(
      doc,
      root,
      "rogue",
      emblemSize,
      "requiem__emblem requiem__emblem--rogue",
      0,
      0,
      0,
      { paint: "class" },
    );
    play(
      blades,
      [
        [880, 0],
        [900, 1],
        [1180, 1],
        [1380, 0],
      ],
      "requiem-blades",
    );
  } else if (classKey === "priest") {
    PRIEST_RAY_ORDER.forEach((deg, index) => {
      const ray = piece(
        "requiem__ray",
        segmentPath(polar(0.5 * radius, deg), polar(0.6 * radius, deg)),
        1.25,
      );
      // Lit at the beat, each ray goes out over 120 ms, 60 ms after the last.
      const out = 880 + 60 * index;
      play(
        ray,
        index === 0
          ? [
              [880, 0],
              [885, 1],
              [1000, 0],
            ]
          : [
              [880, 0],
              [885, 1],
              [out, 1],
              [out + 120, 0],
            ],
        `requiem-ray-${index}`,
      );
    });
    const ember = effectNode(doc, "g", { class: "requiem__ember", opacity: 0 });
    ember.append(
      effectNode(doc, "path", { d: ringPath(0, 0, 0.24 * radius), opacity: 0.18 }),
      effectNode(doc, "path", { d: ringPath(0, 0, 0.1 * radius) }),
    );
    root.append(ember);
    play(
      ember,
      [
        [880, 0],
        [1300, 0],
        [1380, 1],
        [1440, 1],
        [1680, 0],
      ],
      "requiem-ember",
    );
  }
}

/**
 * Draw a played fall's Kill Credit Glints and their beat animations.
 *
 * doc creates the nodes; mount.connector (the Strike Band group, under the
 * bodies) receives one `path.requiem-glint` per record.glints entry, in order:
 * a ring of radius 1.06 x record.radius (CSS px) around the glint's END
 * center, 1.25 px wide, in its class color (class
 * `requiem-glint--<classKey>`), carrying `data-glint-agent` (the public ID, so
 * update can find it). Its END opacity is 0. beatAt is the Fall beat in real
 * ms (already scaled for reduced motion). Normal motion (reduced false): from
 * the beat the ring shows at 70% and fades linearly to 0 over 240 ms while it
 * grows from 0.95 to 1 times its size with the store's ease-out (cubic-bezier
 * 0, 0, 0.58, 1). Reduced motion: the same fade over 200 ms in place, with no
 * growth. Before the beat (fill "backwards") it stays at opacity 0. Nothing is
 * drawn for an empty glint list. Return undefined.
 *
 * @param {Document} doc
 * @param {EventMount} mount
 * @param {LookRecord} record
 * @param {number} beatAt
 * @param {boolean} reduced
 */
function buildCreditGlints(doc, mount, record, beatAt, reduced) {
  const glints = /** @type {ReadonlyArray<KillCreditGlint>} */ (asArray(record.glints));
  const r = 1.06 * Number(record.radius);
  glints.forEach((glint, index) => {
    const node = effectNode(doc, "path", {
      class: `requiem-glint requiem-glint--${glint.classKey ?? "unknown"}`,
      "data-glint-agent": glint.publicAgentId,
      d: ringPath(glint.center.x, glint.center.y, r),
      "stroke-width": 1.25,
      opacity: 0,
    });
    mount.connector.append(node);
    /** @type {Keyframe[]} */
    const frames = reduced
      ? [
          { offset: 0, opacity: 0 },
          { offset: 0.001, opacity: 0.7 },
          { offset: 1, opacity: 0 },
        ]
      : [
          { offset: 0, opacity: 0, transform: "scale(0.95)" },
          { offset: 0.001, opacity: 0.7, transform: "scale(0.95)" },
          ...[0.25, 0.5, 0.75, 1].map((p) => ({
            offset: p,
            opacity: Number(fmt(0.7 * (1 - p))),
            transform: `scale(${(0.95 + 0.05 * storeEaseOut(p)).toFixed(4)})`,
          })),
        ];
    pushBeatAnimation(
      mount,
      record.event,
      node,
      frames,
      beatAt,
      reduced ? 200 : 240,
      `kill-credit-glint-${index}`,
    );
  });
}

/**
 * Return a CSS cubic-bezier(x1, 0, x2, 1) curve at progress p (0 to 1;
 * values outside are clamped): the store's ease-out (x1 0, x2 0.58) and
 * ease-in (x1 0.42, x2 1) are both of this form. Solved by bisection on the
 * curve's x to about 1e-6. Pure.
 *
 * @param {number} p
 * @param {number} x1
 * @param {number} x2
 * @returns {number}
 */
function storeCurve(p, x1, x2) {
  const x = Math.min(1, Math.max(0, p));
  /** @param {number} t @param {number} a @param {number} b */
  const bezier = (t, a, b) =>
    3 * (1 - t) * (1 - t) * t * a + 3 * (1 - t) * t * t * b + t ** 3;
  let low = 0;
  let high = 1;
  for (let step = 0; step < 24; step += 1) {
    const mid = (low + high) / 2;
    if (bezier(mid, x1, x2) < x) low = mid;
    else high = mid;
  }
  return bezier((low + high) / 2, 0, 1);
}

/**
 * Return the store's ease-out curve, CSS cubic-bezier(0, 0, 0.58, 1), at
 * progress p (0 to 1; clamped). The store grows each glint, and drifts the
 * requiem particles, along it. Pure.
 *
 * @param {number} p
 * @returns {number}
 */
function storeEaseOut(p) {
  return storeCurve(p, 0, 0.58);
}

/**
 * Return the store's ease-in curve, CSS cubic-bezier(0.42, 0, 1, 1), at
 * progress p (0 to 1; clamped). The store fades the requiem particles along
 * it (opacity peak x (1 - easeIn)). Pure.
 *
 * @param {number} p
 * @returns {number}
 */
function storeEaseIn(p) {
  return storeCurve(p, 0.42, 1);
}

/**
 * Draw one return: the hit circle and, in motion, the Homecoming ring.
 *
 * doc creates nodes under mount.group; record is a "homecoming" record at the
 * pad (END position). END: nothing visible (the ring group at opacity 0). From
 * the Return beat (event.phaseStart) in normal motion the ring narrows from 1.6
 * to 0.9 x radius (60-420 ms) and fades out by 780 ms; reduced motion shows a
 * fixed 0.9 x radius ring for 420 ms. Return undefined.
 *
 * @param {Document} doc
 * @param {EventMount} mount
 * @param {LookRecord} record
 */
function buildHomecoming(doc, mount, record) {
  const { event, radius } = record;
  const team = record.teamKey ?? "a";
  const root = effectNode(doc, "g", {
    class: `combat-lifecycle-ring combat-lifecycle-ring--resurrection homecoming homecoming--team-${team}`,
    "data-lifecycle-ring": "resurrection",
    transform: `translate(${fmt(record.anchor.x)} ${fmt(record.anchor.y)})`,
  });
  const hit = effectNode(doc, "circle", {
    class: "combat-lifecycle-ring__hit",
    r: fmt(Math.max(radius + 4, 18)),
  });
  hit.addEventListener("pointerdown", (pointerEvent) => {
    if (pointerEvent.button === 0) {
      pointerEvent.stopPropagation();
    }
  });
  const ring = effectNode(doc, "g", { class: "homecoming__rings", opacity: 0 });
  const d = ringPath(0, 0, 0.9 * radius);
  ring.append(
    effectNode(doc, "path", {
      class: "homecoming__glow",
      d,
      "stroke-width": 4.5,
      opacity: 0.18,
    }),
    effectNode(doc, "path", {
      class: "homecoming__ring",
      d,
      "stroke-width": 1.5,
      opacity: 0.95,
    }),
  );
  root.append(hit, ring);
  mount.group.append(root);
  const motion = motionOf(mount);
  if (motion === null) {
    return;
  }
  const beatAt = Number(event.phaseStart ?? 0) * motion.scale;
  if (motion.reduced) {
    pushBeatAnimation(
      mount,
      event,
      ring,
      opacityFrames(
        [
          [0, 0],
          [100, 0.8],
          [420, 0],
        ],
        420,
      ),
      beatAt,
      420,
      "homecoming",
    );
    return;
  }
  const grow = 1.6 / 0.9;
  pushBeatAnimation(
    mount,
    event,
    ring,
    [
      { offset: 0, opacity: 0, transform: `scale(${fmt(grow)})` },
      { offset: 60 / 780, opacity: 1, transform: `scale(${fmt(grow)})` },
      { offset: 420 / 780, opacity: 1, transform: "scale(1)", easing: "linear" },
      { offset: 500 / 780, opacity: 1, transform: "scale(1)" },
      { offset: 1, opacity: 0, transform: "scale(1)" },
    ],
    beatAt,
    780,
    "homecoming",
    "ease-in",
  );
}

/**
 * Draw a return's old corpse lifting away (the store's corpse lift).
 *
 * doc creates nodes in mount.connector (the Strike Band layer, under every
 * body, so a champion standing on the spot still covers it); record is a
 * "homecoming" record. Nothing is drawn when record.oldCorpse is null, or when
 * the mount is settled or motion is off (a paused still shows no lift). Two
 * groups are made, both at END opacity 0 and placed at the corpse spot:
 * - `g.requiem-old-corpse`: the corpse as the frame before drew it
 *   (oldCorpseParts: its Ashen Token or Fallen Step Aside coin, its Class Seal
 *   outline, ash relic and last Vigil Pip). It shows from the start of the
 *   tick, so the corpse does not vanish when the tick arrives, and fades out
 *   over 200 ms from the Return beat (event.phaseStart; 160 ms in reduced
 *   motion): the spec's 900-1100 ms on its 880 ms Fall clock.
 * - `g.requiem-old-corpse__lift-group`: one 1 px Ash hairline ring at 0.95
 *   times the body radius around the corpse spot. At the Return beat it
 *   appears at 50% and fades out over 300 ms while it grows to 1.2 times
 *   (ease-out; reduced motion: no growth), the store's `lift`.
 * Both are aria-hidden and take no pointer. Pushes 2 or 3 fill "backwards"
 * animations. Export-safe SVG. Return undefined.
 *
 * @param {Document} doc
 * @param {EventMount} mount
 * @param {LookRecord} record
 */
function buildOldCorpse(doc, mount, record) {
  /** @type {OldCorpse | null} */
  const corpse = record.oldCorpse ?? null;
  const motion = motionOf(mount);
  if (corpse === null || motion === null) {
    return;
  }
  const event = record.event;
  const place = `translate(${fmt(corpse.center.x)} ${fmt(corpse.center.y)})`;
  const group = effectNode(doc, "g", {
    class: `requiem-old-corpse requiem-old-corpse--team-${corpse.teamKey}`,
    "aria-hidden": "true",
    transform: place,
    opacity: 0,
  });
  group.append(...oldCorpseParts(doc, corpse));
  const lift = effectNode(doc, "g", {
    class: "requiem-old-corpse__lift-group",
    "aria-hidden": "true",
    transform: place,
    opacity: 0,
  });
  const ring = effectNode(doc, "path", {
    class: "requiem-old-corpse__lift",
    d: ringPath(0, 0, 0.95 * corpse.radius),
    "stroke-width": 1,
  });
  lift.append(ring);
  mount.connector.append(group, lift);
  const beatAt = Number(event.phaseStart ?? 0) * motion.scale;
  pushBeatAnimation(
    mount,
    event,
    group,
    [{ opacity: 1 }, { opacity: 0 }],
    beatAt,
    motion.reduced ? 160 : 200,
    "old-corpse",
  );
  pushBeatAnimation(
    mount,
    event,
    lift,
    [
      { offset: 0, opacity: 0 },
      { offset: 0.001, opacity: 0.5 },
      { offset: 1, opacity: 0 },
    ],
    beatAt,
    300,
    "old-corpse-lift",
  );
  if (!motion.reduced) {
    pushBeatAnimation(
      mount,
      event,
      ring,
      [{ transform: "scale(1)" }, { transform: "scale(1.2)" }],
      beatAt,
      300,
      "old-corpse-lift-grow",
      STORE_EASE_OUT,
    );
  }
}

/**
 * Return the drawing of an old corpse, centered on 0, 0 (its corpse spot).
 *
 * It repeats the Champions look's Ashen Token (`champions.js` drawFallen) and
 * this look's corpse marks (buildCorpseMarks) with the same numbers, so the
 * copy matches the corpse the frame before drew: the ash disc (radius R - 1 at
 * 75% fill, or the coin radius at 92% for a Fallen Step Aside coin), the team
 * ash rim (R - 0.5, 1 px at 45%; on a coin radius - 0.6, 1.2 px at 70%), Team
 * B's inner rim 3 px inside (1 px at 45%; on a coin 0.9 px at 60%), the Class
 * Seal's ash plate outline (1 px at 35%, never on a coin), the ash emblem
 * (`icons.js` "ash" paint, the token's emblem box, on a coin max(9, 0.55 x
 * box)), and at the corpse spot itself the ash relic (from R 14 px, with
 * relic on) and the Vigil Pips (from R 10 px). corpse is an OldCorpse; doc
 * creates the nodes. Return them in paint order. Pure apart from node
 * creation.
 *
 * @param {Document} doc
 * @param {OldCorpse} corpse
 * @returns {Element[]}
 */
function oldCorpseParts(doc, corpse) {
  const coin = corpse.aside;
  const cx = coin ? coin.x - corpse.center.x : 0;
  const cy = coin ? coin.y - corpse.center.y : 0;
  const radius = coin ? coin.radius : corpse.radius;
  /** @type {Element[]} */
  const parts = [
    effectNode(doc, "circle", {
      class: "requiem-old-corpse__disc",
      cx: fmt(cx),
      cy: fmt(cy),
      r: fmt(Math.max(0, coin ? radius : radius - 1)),
      "fill-opacity": coin ? 0.92 : 0.75,
    }),
    effectNode(doc, "circle", {
      class: "requiem-old-corpse__rim",
      cx: fmt(cx),
      cy: fmt(cy),
      r: fmt(Math.max(0, radius - (coin ? 0.6 : 0.5))),
      "stroke-width": coin ? 1.2 : 1,
      opacity: coin ? 0.7 : 0.45,
    }),
  ];
  if (corpse.doubleBand) {
    parts.push(
      effectNode(doc, "circle", {
        class: "requiem-old-corpse__rim",
        cx: fmt(cx),
        cy: fmt(cy),
        r: fmt(Math.max(0, radius - 3)),
        "stroke-width": coin ? 0.9 : 1,
        opacity: coin ? 0.6 : 0.45,
      }),
    );
  }
  const seal = corpse.sealClass !== null && coin === null;
  if (seal) {
    parts.push(
      effectNode(doc, "path", {
        class: "requiem-old-corpse__seal",
        d: classSealPaths(/** @type {any} */ (corpse.sealClass), corpse.radius).plate,
        "stroke-width": 1,
        "stroke-linejoin": "round",
        opacity: CLASS_SEAL.ashOpacity,
      }),
    );
  }
  const size = coin ? Math.max(9, 0.55 * corpse.emblemSize) : corpse.emblemSize;
  const emblem = createSvgIcon(doc, corpse.glyphKey, {
    className: "requiem-old-corpse__emblem",
    paint: "ash",
    size,
    onClassSeal: seal,
  });
  emblem.setAttribute("x", fmt(cx - size / 2));
  emblem.setAttribute("y", fmt(cy - size / 2));
  emblem.setAttribute("width", fmt(size));
  emblem.setAttribute("height", fmt(size));
  parts.push(emblem);
  if (corpse.relic && corpse.radius >= 14) {
    parts.push(relicNode(doc, corpse.classKey, corpse.radius));
  }
  if (corpse.pips > 0 && corpse.radius >= 10) {
    parts.push(pipsNode(doc, corpse.pips, corpse.radius));
  }
  return parts;
}

/**
 * Return a team crest: one ring for Team A, two rings for Team B.
 *
 * doc creates it; (cx, cy) is its centre and r its outer radius in CSS px;
 * team picks the ring colour class; className is added. Return the group.
 *
 * @param {Document} doc
 * @param {number} cx
 * @param {number} cy
 * @param {number} r
 * @param {TeamKey} team
 * @param {string} className
 */
function crestNode(doc, cx, cy, r, team, className) {
  const crest = effectNode(doc, "path", {
    class: `${className} requiem-crest requiem-crest--team-${team}`,
    d:
      team === "b"
        ? ringPath(cx, cy, r) + ringPath(cx, cy, Math.max(1, r - 2.6))
        : ringPath(cx, cy, r),
    "stroke-width": team === "b" ? 1.2 : 1.6,
  });
  return crest;
}

/**
 * Return the Fallen panel's row records for one tick (pure data, no DOM).
 *
 * The Fallen panel is page chrome at the top of the side rail ("Fallen · Last
 * 8 Ticks"). It is never drawn on or over the arena and never enters the
 * battlefield SVG, so PNG exports never carry it.
 *
 * input.tick is the tick on screen (an integer; anything else gives []).
 * input.rows lists the falls the caller knows about, each a FallenFeedSource:
 * tick (the integer tick of the fall), victim (identity record: publicAgentId,
 * classId, teamId, title), victimPresentationKey (string or null; used only on
 * the fall's own tick to hold the row to the Fall beat), killers (the named
 * killers' identity records; null or empty names nobody), killingTeamId (1, 2
 * or null), points (1 or 2, or null when unknown) and ultimate (the damaging
 * Ultimate caster {classId, publicAgentId}, null when only Basics did it, or
 * absent when the caller does not know). Rows from the plan's fallen_feed_row
 * events (choreography-plan.js) fit this shape once the caller adds tick.
 * input.displayIdOf (optional, default none) maps a public agent ID to its
 * shown ID ("0" to "9"); without it no ID is written beside an emblem.
 *
 * Window: only rows with tick - FEED_WINDOW_TICKS < row.tick <= tick count
 * (never a future fall), newest first; rows of one tick keep their input
 * order. At most FEED_MAX_ROWS rows are returned.
 *
 * Each record: {tick, ageTicks (0 for this tick), beatKey (the victim's
 * presentation key when ageTicks is 0, else null), victim {publicAgentId,
 * displayId, classKey, teamKey, title}, killers [{publicAgentId, displayId,
 * classKey, teamKey, title}], ability (null when ultimate is absent or no
 * killer is named; {kind: "basic"} when ultimate is null; {kind: "ultimate",
 * classKey} for the caster's class), points ({text "+1" or "+2", teamKey of
 * the scoring team} or null), killingTeamId, label (plain words for screen
 * readers)}.
 *
 * Agent POV: the caller must pass rows already limited to the audience (the
 * plan's fallen_feed_row rows are: only falls the team saw, only killers it
 * saw). This function adds nothing: an unnamed killer gets no mark at all.
 * It reads no researcher view and no match summary. Inputs are not changed;
 * returns a frozen array of frozen records.
 *
 * @param {{
 *   tick: number,
 *   rows?: ReadonlyArray<FallenFeedSource> | null,
 *   displayIdOf?: ((publicAgentId: string) => string | null) | null,
 * }} input
 * @returns {ReadonlyArray<FallenFeedRecord>}
 */
export function fallenFeedRecords(input) {
  const tick = Number(input?.tick);
  if (!Number.isInteger(tick)) {
    return Object.freeze([]);
  }
  const displayIdOf =
    typeof input.displayIdOf === "function" ? input.displayIdOf : () => null;
  const shown = asArray(input.rows)
    .map((row, order) => ({ row, order }))
    .filter(
      ({ row }) =>
        isRecord(row) &&
        isRecord(row.victim) &&
        Number.isInteger(row.tick) &&
        row.tick <= tick &&
        row.tick > tick - FEED_WINDOW_TICKS,
    )
    .sort((left, right) => right.row.tick - left.row.tick || left.order - right.order)
    .slice(0, FEED_MAX_ROWS);
  return Object.freeze(
    shown.map(({ row }) =>
      feedRecord(/** @type {FallenFeedSource} */ (row), tick, displayIdOf),
    ),
  );
}

/**
 * @typedef {Readonly<{
 *   audienceKey: string,
 *   tick: number,
 *   earlier: ReadonlyArray<FallenFeedSource>,
 *   current: ReadonlyArray<FallenFeedSource>,
 * }>} FallenFeedMemory
 *   What the Fallen panel remembers between paints of one audience: the falls
 *   of earlier ticks still inside the window, and this tick's own falls.
 */

/**
 * Carry the Fallen panel's falls from tick to tick.
 *
 * memory is the FallenFeedMemory of the last paint, or null. plan is the
 * choreography plan about to be shown (choreographer.plan), or null; its
 * audienceKey (string), simulatorStep (the tick, an integer) and its
 * fallen_feed_row derivedEvents are read. Rows are kept even when the Fallen
 * Feed switch is off, so turning it on shows the recent falls.
 *
 * Rules (the same as the herald carry in choreography-painter.js): the same
 * audienceKey one tick later keeps the remembered falls of the last
 * FEED_WINDOW_TICKS ticks and adds this tick's; the same audienceKey and the
 * same tick (a repaint, resize or switch change) gives the same rows as
 * before; anything else (a seek, a skip, a new episode, a view or recipient
 * change, or no plan) starts again from this tick's falls only, so the panel
 * equals a fresh render.
 *
 * Returns {memory, tick, rows}: memory to keep (null without a plan, audience
 * key or tick), tick (the plan's tick or null) and rows, FallenFeedSource rows
 * with their tick, ready for fallenFeedRecords.
 *
 * Agent POV: the plan's fallen_feed_row rows hold only falls the team saw and
 * only killers it saw, and a different audience key never reuses memory, so
 * Oracle -> POV -> Oracle shows a fresh panel. Pure: inputs are not changed.
 *
 * @param {FallenFeedMemory | null} memory
 * @param {JsonRecord | null} plan
 * @returns {{memory: FallenFeedMemory | null, tick: number | null, rows: ReadonlyArray<FallenFeedSource>}}
 */
export function fallenFeedHistory(memory, plan) {
  const audienceKey = typeof plan?.audienceKey === "string" ? plan.audienceKey : null;
  const tick = Number.isInteger(plan?.simulatorStep)
    ? Number(plan?.simulatorStep)
    : null;
  if (audienceKey === null || tick === null) {
    return { memory: null, tick: null, rows: Object.freeze([]) };
  }
  const current = Object.freeze(
    asArray(plan?.derivedEvents)
      .filter((row) => isRecord(row) && row.kind === "fallen_feed_row")
      .map((row) =>
        Object.freeze({
          tick,
          victim: row.victim,
          victimPresentationKey:
            typeof row.victimPresentationKey === "string"
              ? row.victimPresentationKey
              : null,
          killers: row.killers ?? null,
          killingTeamId: row.killingTeamId ?? null,
          points: row.points ?? null,
          ...(Object.hasOwn(row, "ultimate") ? { ultimate: row.ultimate ?? null } : {}),
        }),
      ),
  );
  const same = memory?.audienceKey === audienceKey;
  /** @type {ReadonlyArray<FallenFeedSource>} */
  const earlier = !same
    ? Object.freeze([])
    : tick === memory.tick + 1
      ? Object.freeze(
          [...memory.earlier, ...memory.current].filter(
            (row) => row.tick > tick - FEED_WINDOW_TICKS,
          ),
        )
      : tick === memory.tick
        ? memory.earlier
        : Object.freeze([]);
  const next = Object.freeze({ audienceKey, tick, earlier, current });
  return { memory: next, tick, rows: Object.freeze([...earlier, ...current]) };
}

/**
 * Turn one fall into a Fallen panel record (pure; see fallenFeedRecords).
 *
 * row is one fall inside the window; tick the tick on screen; displayIdOf maps
 * a public ID to its shown ID or null. The scoring team is killingTeamId (1 or
 * 2), else the victim's other team, else unknown (points then carry no team).
 *
 * @param {FallenFeedSource} row
 * @param {number} tick
 * @param {(publicAgentId: string) => string | null} displayIdOf
 * @returns {FallenFeedRecord}
 */
function feedRecord(row, tick, displayIdOf) {
  const person = (/** @type {JsonRecord} */ identity) => {
    const publicAgentId =
      typeof identity.publicAgentId === "string" ? identity.publicAgentId : null;
    const shownId = publicAgentId === null ? null : displayIdOf(publicAgentId);
    return Object.freeze({
      publicAgentId,
      displayId: typeof shownId === "string" ? shownId : null,
      classKey: classKeyOfIdentity(identity),
      teamKey: teamKeyOf(teamIdOf(identity)),
      title: String(identity.title ?? "A champion"),
    });
  };
  const victim = person(row.victim);
  const killers = Object.freeze(asArray(row.killers).filter(isRecord).map(person));
  const victimTeam = teamIdOf(row.victim);
  const killingTeamId =
    row.killingTeamId === 1 || row.killingTeamId === 2 ? row.killingTeamId : null;
  const scoringTeam =
    killingTeamId ??
    (victimTeam === null ? null : /** @type {1 | 2} */ (3 - victimTeam));
  const pointsValue = Number(row.points);
  const caster = isRecord(row.ultimate) ? row.ultimate : null;
  /** @type {FallenFeedRecord["ability"]} */
  let ability = null;
  if (killers.length > 0 && row.ultimate === null) {
    ability = Object.freeze({ kind: "basic", classKey: null });
  } else if (killers.length > 0 && caster !== null) {
    ability = Object.freeze({ kind: "ultimate", classKey: classKeyOf(caster.classId) });
  }
  const ageTicks = tick - row.tick;
  const names = killers.map((killer) => killer.title);
  return Object.freeze({
    tick: row.tick,
    ageTicks,
    beatKey:
      ageTicks === 0 && typeof row.victimPresentationKey === "string"
        ? row.victimPresentationKey
        : null,
    victim,
    killers,
    ability,
    points:
      pointsValue === 1 || pointsValue === 2
        ? Object.freeze({ text: `+${pointsValue}`, teamKey: teamKeyOf(scoringTeam) })
        : null,
    killingTeamId,
    label:
      names.length > 0
        ? `Tick ${row.tick}: ${names.join(" + ")} took down ${victim.title}`
        : `Tick ${row.tick}: ${victim.title} fell`,
  });
}

/**
 * Build the Fallen panel for the side rail (page chrome, never the arena).
 *
 * doc creates the nodes; records come from fallenFeedRecords. Return a new,
 * detached `section.fallen-feed` the caller mounts at the top of the rail
 * (and replaces on every frame, filter or view change). It holds a header
 * ("Fallen" and "Last 8 Ticks"), then either "No one has fallen in the last 8
 * ticks." or one focusable `li.fallen-feed__row` per record. A row's line is
 * one small inline SVG drawn as in the store's shell part (11 px emblems):
 * each named killer's class emblem in its team colour with its shown ID,
 * joined by "+"; the ability mark (a grey dot for Basics, a diamond for an
 * Ultimate) when known; an arrow; the fallen champion's grey emblem on a
 * grave-stone disc ringed in its team colour (Team B double-ringed) with its
 * shown ID; and the points in the scoring team's colour. "Tick N" follows as
 * text. A row of this tick carries the Fall beat hold (data-beat-hold "fall",
 * data-beat-key the victim's presentation key, data-beat-side "after",
 * data-beat-fade 160), so a choreographer that lists this panel among its
 * beatHoldRoots fades it in at that champion's Fall beat; older rows simply
 * show.
 *
 * Side effects: creates DOM nodes and registers one tooltip owner per row
 * (the shared death explanation; when the audience named no killer, only who
 * fell and the killing team). No timers, storage or listeners of its own.
 * Agent POV: shows exactly the records given (see fallenFeedRecords).
 *
 * @param {Document} doc
 * @param {ReadonlyArray<FallenFeedRecord>} records
 * @returns {HTMLElement}
 */
export function buildFallenFeed(doc, records) {
  const panel = htmlNode(doc, "section", {
    class: "fallen-feed",
    "aria-label": "Fallen · Last 8 Ticks",
  });
  const head = htmlNode(doc, "header", { class: "fallen-feed__head" });
  head.append(
    htmlNode(doc, "span", { class: "fallen-feed__title" }, "Fallen"),
    htmlNode(doc, "span", { class: "fallen-feed__window" }, "Last 8 Ticks"),
  );
  panel.append(head);
  if (records.length === 0) {
    panel.append(
      htmlNode(
        doc,
        "p",
        { class: "fallen-feed__empty" },
        "No one has fallen in the last 8 ticks.",
      ),
    );
    return panel;
  }
  const list = htmlNode(doc, "ol", { class: "fallen-feed__rows" });
  for (const record of records) {
    const row = htmlNode(doc, "li", {
      class: "fallen-feed__row",
      tabindex: "0",
      "aria-label": record.label,
      "data-tick": record.tick,
      "data-age-ticks": record.ageTicks,
    });
    if (record.beatKey !== null) {
      row.setAttribute("data-beat-hold", "fall");
      row.setAttribute("data-beat-key", record.beatKey);
      row.setAttribute("data-beat-side", "after");
      row.setAttribute("data-beat-fade", "160");
    }
    row.append(
      feedLineNode(doc, record),
      htmlNode(doc, "span", { class: "fallen-feed__tick" }, `Tick ${record.tick}`),
    );
    registerTooltipOwner(row, fallenRowTooltip(record));
    list.append(row);
  }
  panel.append(list);
  return panel;
}

/**
 * Draw one Fallen row's line as a small inline SVG (store sizes, CSS px).
 *
 * Layout from the store's feedRow: emblems 11 px on a 15 px row; each shown
 * ID (9 px bold) starts 1.5 px after its emblem (2.5 px after the victim's
 * ringed disc) and takes FEED_ID_ADVANCE_PX per character, so a two-digit ID
 * gets its full width; then a 2 px gap and a 9 px slot for the "+" (12 px,
 * centred), so no ID ever runs into a "+"; the ability mark 8 px wide after
 * a 3 px gap; the arrow 15 px wide; the victim disc 12 px across; then the
 * points ("+1", 12 px) 4 px after the victim's ID. Return the svg.
 * Export-safe nodes only (it is page chrome, but it follows the same rule).
 *
 * @param {Document} doc
 * @param {FallenFeedRecord} record
 * @returns {SVGElement}
 */
function feedLineNode(doc, record) {
  const size = 11;
  const height = 15;
  const cy = height / 2;
  const line = effectNode(doc, "svg", {
    class: "fallen-feed__line",
    "aria-hidden": "true",
    focusable: "false",
  });
  let x = 0;
  record.killers.forEach((killer, index) => {
    if (index > 0) {
      line.append(feedText(doc, "fallen-feed__plus", x + 4.5, cy, "+"));
      x += 9;
    }
    appendEmblem(
      doc,
      line,
      killer.classKey,
      size,
      `fallen-feed__emblem fallen-feed__emblem--team-${killer.teamKey ?? "a"}`,
      1,
      x + size / 2,
      cy,
      { paint: "host", teamId: killer.teamKey === "b" ? 2 : 1 },
    );
    x += size;
    if (killer.displayId !== null) {
      line.append(feedText(doc, "fallen-feed__id", x + 1.5, cy, killer.displayId));
      x += 1.5 + killer.displayId.length * FEED_ID_ADVANCE_PX;
    }
    x += 2;
  });
  if (record.ability !== null) {
    x += 3;
    line.append(
      record.ability.kind === "ultimate"
        ? effectNode(doc, "path", {
            class: `fallen-feed__ability fallen-feed__ability--ultimate fallen-feed__ability--${record.ability.classKey ?? "unknown"}`,
            d: `M${fmt(x + 2.5)} ${fmt(cy - 3)}L${fmt(x + 5.5)} ${fmt(cy)}L${fmt(x + 2.5)} ${fmt(cy + 3)}L${fmt(x - 0.5)} ${fmt(cy)}Z`,
          })
        : effectNode(doc, "circle", {
            class: "fallen-feed__ability fallen-feed__ability--basic",
            cx: fmt(x + 2.5),
            cy: fmt(cy),
            r: 1.5,
          }),
    );
    x += 8;
  }
  line.append(
    effectNode(doc, "path", {
      class: "fallen-feed__arrow",
      d: `M${fmt(x)} ${fmt(cy)}H${fmt(x + 8)}M${fmt(x + 8)} ${fmt(cy - 2.4)}L${fmt(x + 11)} ${fmt(cy)}L${fmt(x + 8)} ${fmt(cy + 2.4)}`,
      "stroke-width": 1,
    }),
  );
  x += 15;
  const victimTeam = record.victim.teamKey ?? "a";
  const centre = x + size / 2;
  line.append(
    effectNode(doc, "path", {
      class: "fallen-feed__victim-field",
      d: ringPath(centre, cy, size / 2 + 0.5),
    }),
    crestNode(doc, centre, cy, size / 2 + 0.5, victimTeam, "fallen-feed__victim-ring"),
  );
  appendEmblem(
    doc,
    line,
    record.victim.classKey,
    size * 0.72,
    "fallen-feed__emblem fallen-feed__emblem--fallen",
    1,
    centre,
    cy,
    { paint: "ash" },
  );
  x += size;
  if (record.victim.displayId !== null) {
    line.append(feedText(doc, "fallen-feed__id", x + 2.5, cy, record.victim.displayId));
    x += 2.5 + record.victim.displayId.length * FEED_ID_ADVANCE_PX;
  }
  x += 4;
  if (record.points !== null) {
    line.append(
      feedText(
        doc,
        `fallen-feed__points fallen-feed__points--team-${record.points.teamKey ?? "a"}`,
        x,
        cy,
        record.points.text,
      ),
    );
    x += record.points.text.length * 7.5;
  }
  const width = fmt(Math.max(x, 1));
  setEffectAttributes(line, {
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
  });
  return line;
}

/**
 * Return one SVG text node of a Fallen row: class, left x and centre y (CSS
 * px) and its words. Pure apart from creating the node.
 *
 * @param {Document} doc
 * @param {string} className
 * @param {number} x
 * @param {number} y
 * @param {string} words
 * @returns {SVGElement}
 */
function feedText(doc, className, x, y, words) {
  const node = effectNode(doc, "text", {
    class: className,
    x: fmt(x),
    y: fmt(y + 0.5),
  });
  node.textContent = words;
  return node;
}

/**
 * Return the tooltip descriptor of one Fallen panel row.
 *
 * When the record names at least one killer, the shared death explanation
 * lists them as Kill Contributors. When the audience named no killer (for
 * example an Agent POV fall whose attacker the team did not see), the card
 * says only who fell, on which tick, and the killing team, with no killer row
 * and no "Unseen" mark. Pure apart from building the descriptor.
 *
 * @param {FallenFeedRecord} record
 */
function fallenRowTooltip(record) {
  const eventId = `fallen:${record.tick}:${record.victim.publicAgentId ?? record.victim.title}`;
  if (record.killers.length > 0) {
    return explainDeathAnnouncement({
      eventId,
      label: `Fallen On Tick ${record.tick}`,
      members: [
        {
          title: record.victim.title,
          killingTeamId: record.killingTeamId,
          contributors: record.killers.map((killer) => ({ title: killer.title })),
        },
      ],
    });
  }
  const killingTeam = teamLabelFromId(Number(record.killingTeamId));
  return createSemanticDescriptor({
    kind: "event",
    id: `death-announcement:${eventId}`,
    title: record.victim.title,
    tone: "information",
    accent: "none",
    summary: `This champion fell on tick ${record.tick}.`,
    rows:
      killingTeam === null
        ? []
        : [
            {
              label: "Killing Team",
              value: killingTeam,
              metadata: { compact: true, full: true },
            },
          ],
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "pointer",
  });
}

/**
 * Build the Wave Chip for the scorebug (page chrome, never the arena).
 *
 * doc creates the nodes; teamId is the returning team (2 means Team B, any
 * other value Team A). beatKey (optional, default null) is the presentation
 * key of one champion of that team returning this tick. Return a new, detached
 * `span.wave-chip.wave-chip--team-a|b`: a pill with the team crest (Team B
 * double-ringed, as a small inline SVG) and "Team A Returns". The caller (the
 * scorebug owner) puts it under that team's name on its home side, only on a
 * tick when that team's wave brings champions back. With a beatKey the chip
 * carries the Return beat hold (data-beat-hold "return", data-beat-side
 * "after", data-beat-fade 120), so a choreographer that lists the scorebug
 * among its beatHoldRoots fades it in at that Return beat; without one it
 * simply shows. Returns are public facts, so the chip is the same in Oracle
 * View and Agent POV. Side effects: creates DOM nodes only.
 *
 * @param {Document} doc
 * @param {number} teamId
 * @param {string | null} [beatKey]
 * @returns {HTMLElement}
 */
export function buildWaveChip(doc, teamId, beatKey = null) {
  const id = Number(teamId) === 2 ? 2 : 1;
  const team = id === 2 ? "b" : "a";
  const words = `${teamLabelFromId(id)} Returns`;
  const chip = htmlNode(doc, "span", {
    class: `wave-chip wave-chip--team-${team}`,
    "data-team-id": id,
    role: "status",
    "aria-label": words,
  });
  if (typeof beatKey === "string" && beatKey.length > 0) {
    chip.setAttribute("data-beat-hold", "return");
    chip.setAttribute("data-beat-key", beatKey);
    chip.setAttribute("data-beat-side", "after");
    chip.setAttribute("data-beat-fade", "120");
  }
  const crest = effectNode(doc, "svg", {
    class: "wave-chip__crest-icon",
    width: 8,
    height: 8,
    viewBox: "0 0 8 8",
    "aria-hidden": "true",
    focusable: "false",
  });
  crest.append(crestNode(doc, 4, 4, 2.8, team, "wave-chip__crest"));
  chip.append(crest, htmlNode(doc, "span", { class: "wave-chip__label" }, words));
  return chip;
}

/**
 * Create a detached HTML element (XHTML namespace, so it works in any
 * document), write its attributes as text and, when given, its words.
 *
 * @param {Document} doc
 * @param {string} tagName
 * @param {Record<string, string | number>} attributes
 * @param {string} [words] default none.
 * @returns {HTMLElement}
 */
function htmlNode(doc, tagName, attributes, words) {
  const element = /** @type {HTMLElement} */ (
    doc.createElementNS(HTML_NAMESPACE, tagName)
  );
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  if (words !== undefined) {
    element.textContent = words;
  }
  return element;
}

/**
 * Draw one corpse's durable marks: the ash relic and the Vigil Pips.
 *
 * doc creates the nodes; record is a "corpse-marks" record. The group carries
 * the beat hold (data-beat-hold "fall", data-beat-key = presentation key,
 * data-beat-side "after", data-beat-fade 300), so on the tick of this fall the
 * choreographer reveals it at the Fall beat; on other ticks it simply shows.
 * Relic (radius >= 14 only): Mage eight short cracks, Warrior split rim, Hunter
 * a leaf, Rogue three blade rings, Priest one ember, in ash at 50% (Priest 70%).
 * Pips (radius >= 10 only): up to 5 silver dots under the corpse on an arc at
 * 0.95 x radius, 24 degrees apart; above 5 the count as a number (radius >= 14).
 * Return the `g.requiem-corpse-marks` group (not yet attached).
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement}
 */
function buildCorpseMarks(doc, record) {
  const group = effectNode(doc, "g", {
    class: `requiem-corpse-marks requiem-corpse-marks--team-${record.teamKey}`,
    "data-presentation-key": record.presentationKey,
    "data-beat-hold": "fall",
    "data-beat-key": record.presentationKey,
    "data-beat-side": "after",
    "data-beat-fade": 300,
    transform: `translate(${fmt(record.center.x)} ${fmt(record.center.y)})`,
  });
  const radius = Number(record.radius);
  if (record.relic && radius >= 14) {
    group.append(relicNode(doc, record.classKey, radius));
  }
  if (record.pips > 0 && radius >= 10) {
    group.append(pipsNode(doc, record.pips, radius));
  }
  return group;
}

/**
 * Return the ash relic path of one class at radius (CSS px), centred on 0, 0.
 *
 * @param {Document} doc
 * @param {ClassKey} classKey
 * @param {number} radius
 * @returns {SVGElement}
 */
function relicNode(doc, classKey, radius) {
  const r = 0.56 * radius;
  if (classKey === "mage") {
    return effectNode(doc, "path", {
      class: "requiem-relic requiem-relic--stroke",
      d: Array.from({ length: 8 }, (_, k) =>
        segmentPath(
          polar(0.46 * radius, 22.5 + 45 * k),
          polar(0.62 * radius, 22.5 + 45 * k),
        ),
      ).join(""),
      "stroke-width": 0.75,
      opacity: 0.5,
    });
  }
  if (classKey === "warrior") {
    return effectNode(doc, "path", {
      class: "requiem-relic requiem-relic--stroke",
      d: rimHalvesPath(r, 0.75),
      "stroke-width": 1.5,
      opacity: 0.5,
    });
  }
  if (classKey === "hunter") {
    return effectNode(doc, "path", {
      class: "requiem-relic requiem-relic--fill",
      d: almondPath(polar(r, 180), 0, 0.2 * radius, 0.09 * radius),
      opacity: 0.5,
    });
  }
  if (classKey === "rogue") {
    return effectNode(doc, "path", {
      class: "requiem-relic requiem-relic--stroke",
      d: [0, 120, 240]
        .map((deg) => {
          const [x, y] = polar(r, deg);
          return ringPath(x, y, 0.09 * radius);
        })
        .join(""),
      "stroke-width": 0.75,
      opacity: 0.5,
    });
  }
  const [x, y] = polar(r, 0);
  return effectNode(doc, "path", {
    class: "requiem-relic requiem-relic--fill",
    d: ringPath(x, y, 0.1 * radius),
    opacity: 0.7,
  });
}

/**
 * Return the Vigil Pips for count ticks at radius (CSS px), centred on 0, 0.
 *
 * Up to 5 dots on an arc at 0.95 x radius centred at 6 o'clock, 24 degrees
 * apart, dot radius max(1.25, 0.065 x radius), at 90%. More than 5: the count
 * as a number at 6 o'clock (radius >= 14), else an empty group.
 *
 * @param {Document} doc
 * @param {number} count
 * @param {number} radius
 * @returns {SVGElement}
 */
function pipsNode(doc, count, radius) {
  const arc = 0.95 * radius;
  if (count > 5) {
    const [x, y] = polar(arc, 180);
    const text = effectNode(doc, "text", {
      class: "vigil-pips vigil-pips__count",
      x: fmt(x),
      y: fmt(y),
      opacity: 0.81,
      "data-pips": count,
    });
    text.textContent = radius >= 14 ? String(count) : "";
    return text;
  }
  const dot = Math.max(1.25, 0.065 * radius);
  return effectNode(doc, "path", {
    class: "vigil-pips",
    "data-pips": count,
    d: Array.from({ length: count }, (_, i) => {
      const [x, y] = polar(arc, 180 + (i - (count - 1) / 2) * 24);
      return ringPath(x, y, dot);
    }).join(""),
    opacity: 0.9,
  });
}

/**
 * Move nodes to new geometry (reprojection after resize or camera change).
 *
 * For event records, nodes are [group, connector, ...] as build returned and
 * records holds the event's new record: a fall or return moves its ring to the
 * new anchor, and each Kill Credit Glint in the connector moves to the new END
 * center of the glint record with the same public ID (a glint with no new
 * record keeps its old place; a running animation keeps its element). A
 * return's old corpse and its lift ring move to the new record's corpse spot,
 * and the corpse's drawing is rebuilt at the new size inside its group (the
 * group, which carries the fade, is kept); with no oldCorpse they stay. For
 * corpse-mark records, nodes are the groups build returned, in the same order;
 * each moves to its record's center. Return undefined.
 *
 * @param {ReadonlyArray<Element | null>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  const record = records[0];
  if (record === undefined) {
    return;
  }
  if (record.kind === "corpse-marks") {
    records.forEach((each, index) => {
      nodes[index]?.setAttribute(
        "transform",
        `translate(${fmt(each.center.x)} ${fmt(each.center.y)})`,
      );
    });
    return;
  }
  const [group, connector] = nodes;
  const event = record.event;
  const ring = group?.querySelector(".combat-lifecycle-ring");
  if (ring && event.anchor) {
    ring.setAttribute(
      "transform",
      `translate(${fmt(event.anchor.x)} ${fmt(event.anchor.y)})`,
    );
  }
  if (record.kind === "homecoming" && connector) {
    /** @type {OldCorpse | null} */
    const corpse = record.oldCorpse ?? null;
    if (corpse === null) return;
    const place = `translate(${fmt(corpse.center.x)} ${fmt(corpse.center.y)})`;
    for (const node of connector.querySelectorAll(".requiem-old-corpse")) {
      node.setAttribute("transform", place);
      node.replaceChildren(...oldCorpseParts(node.ownerDocument, corpse));
    }
    for (const node of connector.querySelectorAll(".requiem-old-corpse__lift-group")) {
      node.setAttribute("transform", place);
      node
        .querySelector(".requiem-old-corpse__lift")
        ?.setAttribute("d", ringPath(0, 0, 0.95 * corpse.radius));
    }
    return;
  }
  if (record.kind !== "fall" || !connector) {
    return;
  }
  const r = 1.06 * Number(record.radius);
  const glints = /** @type {ReadonlyArray<KillCreditGlint>} */ (asArray(record.glints));
  for (const node of connector.querySelectorAll(".requiem-glint")) {
    const glint = glints.find(
      (each) => each.publicAgentId === node.getAttribute("data-glint-agent"),
    );
    if (glint) {
      node.setAttribute("d", ringPath(glint.center.x, glint.center.y, r));
    }
  }
}

/**
 * Return the tooltip for a record.
 *
 * Fall and return records use the shared event explanation. Corpse marks return null: the champion's own token
 * owns its tooltip. Reads only the record.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  if (!isRecord(record?.event)) {
    return null;
  }
  return explainChoreographyEvent(paintAwareExplanationEvent(record.event));
}
