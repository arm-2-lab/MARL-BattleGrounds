/**
 * @file The Health and Combat Text look (Ivory Arc and Numbers): one life number per
 * changed body, plus the three V5 add-ons Saved Beat, Last Breath and Battle Text
 * Detail.
 *
 * A health change ("net_health" plan event) or a rest heal ("regeneration" plan
 * event) draws one short number beside where its champion ended the tick:
 * Life Ivory "−20" for a loss, Dawn Gold "+12" for a heal, Moon Dew "+3" for
 * Rest. It is 13 px, or 15 px when an Ultimate reached the champion or the blow
 * killed it. The number is rounded the store's way (lifeNumberText: "−41",
 * "−3.3", "<0.1"); the tooltip keeps the exact value. It sits at the cue the
 * plan's collision allocator chose for it (event.cue), and no leader line joins
 * it to the body: the number must stand beside its own body, so how near the
 * cue is belongs to the plan (`choreography-plan.js`) and this look's FOOTPRINT
 * and lifeNumberCueBox, which size the box the plan reserves. There is no "NET",
 * no ID and no "HP unchanged": a body whose life did not change gets no number.
 * The number pops 4 px upward at its event's phaseStart (the plan owns that time
 * on the beat clock) through the choreographer's animation list, with fill
 * "backwards", so the DOM holds END.
 * The life change inside the token is the Champions look's job
 * (`looks/champions.js`, Life Vessels), including the Moon Dew pour of Rest.
 *
 * The V5 add-ons (store `skins/health-and-combat-text.js`, option
 * ivory-arc-and-numbers, and its specs), each off by default:
 * - Saved Beat (saved_beat): when a heal kept a champion alive through blows that
 *   would have killed it (recorded damage at least its start health, healing
 *   above 0, alive after), "Saved" in Dawn Gold stacks 3 px above its number
 *   (alone in the number's place when the number is 0) and its rim flashes Dawn
 *   Gold twice in play. Oracle View only: Agent POV rows carry no damage or
 *   healing totals.
 * - Last Breath (last_breath): when a living champion lost at least as much life
 *   this tick as it has left, the empty part of its life smolders in Cinder at
 *   25%: the empty top of the vessel (Life Vessels) or the empty life ring (the
 *   ring look). The store draws nothing on a vessel; this client draws the same
 *   recolor of the empty part there, so the switch works under the default look.
 *   It fades in at Settle in play and is present in the still.
 * - Battle Text Detail (battle_text_detail): numbers are sized by the share of
 *   the champion's life the change took (under 10% 13 px, 10-30% 15 px, 30% or
 *   more 18 px; never smaller than the plain size), pop 6 px lower and drift up
 *   into place over 600 ms, and in Oracle View carry a share underline (one bar
 *   per attacker in its class color, sized by its recorded share, 3 px for an
 *   Ultimate, Rime for a Freezing Trap, then one Dawn Gold bar for healing) and
 *   a "Beyond Death N" tag 4 px right of a killing blow's number that overshot.
 *   The plan reserves room for the bigger number, its underline and "Saved"
 *   (lifeNumberCueBox), not for the tag, which would push the number far from
 *   its body; so, as the store sheds extras in a crowd, the tag is drawn only
 *   when its box covers no other body, no other number's box and stays inside
 *   the view (beyondDeathClear). Agent POV: the size and drift only.
 *
 * `choreography-painter.js` sends both event kinds here: it calls
 * records({event, plan, filters, transform}), build(records, doc, mount) into its
 * own groups, and update(nodes, records) on resize or camera change. build
 * re-reads the plan from mount.plan, so the Agent POV rules below hold even when
 * records was called with the event alone. The add-ons read their switches from
 * input.filters, else from the event's resolved paint parts (savedBeat,
 * lastBreath, battleTextDetail), else from the registry defaults (all off). They
 * read the plan row's look inputs maximumHealth and recipientRadius and, in
 * Oracle View, totalDamage, totalHealing and damageSources; a missing input draws
 * the plain, truthful number without that add-on. They refine the health event,
 * so they draw only while Scrolling Battle Text keeps that event drawn.
 *
 * Boundary (Agent POV): records reads only the one plan event and, when given,
 * that audience's own plan (plan.audience, plan.bodies, plan.events), the page's
 * switches and the render's projection (only to test the Beyond Death tag's
 * room against the plan's own bodies and boxes). In Agent POV a number (and a
 * Last Breath) shows only for a champion the team saw at both ends of the tick
 * (its plan body has a start), never on a tick it came into sight or returned
 * to its pad (no "+100" on a respawn). A number that shows has Oracle View's
 * color, on both teams: Life Ivory for a loss, Dawn Gold for a heal, Moon Dew
 * for Rest. The color tells the team nothing new: its own rows already carry a
 * seen champion's heal and its Rest as separate events, and the Life Vessel
 * already pours them in gold and Moon Dew. Every Agent POV number is 13 px
 * unless Battle Text Detail sizes it from the observed change (the Ultimate size
 * would reveal an unseen caster). Saved Beat, the share underline and Beyond
 * Death are never drawn there. Nothing here reads a researcher view, a match
 * summary or a body outside the plan.
 */
import { explainNetHealth } from "../explanations.js";
import { DEFAULT_VISUAL_FILTER_STATE } from "../visual-filters.js";
import { classTokenFromId } from "../vocabulary.js";
import { tokenGeometry } from "./champions.js";
import {
  animationSpec,
  asArray,
  assignLayoutPlacement,
  detachedEventMount,
  EVENT_BUDGET,
  effectNode,
  explainChoreographyEvent,
  isRecord,
  mountNodes,
  paintAwareExplanationEvent,
  paintPartEnabled,
  paintParts,
  setEffectAttributes,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").JsonRecord} JsonRecord
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").EventMount} EventMount
 * @typedef {"small" | "medium" | "large"} LifeWeight
 *   Battle Text Detail's severity tier: the share of the champion's maximum life
 *   the change took, under 10% ("small"), 10% to under 30% ("medium") or 30% and
 *   more ("large").
 * @typedef {Readonly<{
 *   text: string,
 *   tone: "damage" | "heal" | "rest",
 *   size: 13 | 15 | 18,
 *   weight?: LifeWeight,
 * }>} LifeNumber
 *   One drawn number: text (for example "−20" or "+12"), tone (the color, the
 *   same in both views: damage Life Ivory, heal Dawn Gold, rest Moon Dew), size
 *   in px, and weight, the Battle Text Detail tier that sized it (absent while
 *   that switch is off, for a Rest number, or when the champion's maximum life
 *   is unknown).
 * @typedef {Readonly<{
 *   savedBeat: boolean,
 *   lastBreath: boolean,
 *   detail: boolean,
 *   lifeStyle: "vessel" | "arc",
 * }>} HealthSwitches
 *   The switches this look reads for one event: Saved Beat, Last Breath and
 *   Battle Text Detail, and the Champions look's life style ("vessel" when
 *   Life Vessels is on, else "arc", the ivory life ring), which decides where
 *   Last Breath smolders.
 * @typedef {Readonly<{
 *   x: number,
 *   y: number,
 *   radius: number,
 *   share: number,
 *   lifeStyle: "vessel" | "arc",
 * }>} BodyMark
 *   Where the body marks go, in CSS pixels: the champion's END center (x, y), its
 *   body radius, its life share after the tick (0..1) and its life style.
 * @typedef {Readonly<{
 *   amount: number,
 *   paint: "mage" | "warrior" | "hunter" | "rogue" | "priest" | "trap" | "heal",
 *   ultimate: boolean,
 * }>} UnderlineBar
 *   One share underline bar: amount (recorded effective damage or healing, in
 *   health points), paint (the attacker's class key, "trap" for a Freezing
 *   Trap, "heal" for the healing bar) and whether an Ultimate dealt it (3 px).
 * @typedef {Readonly<{
 *   saved: boolean,
 *   lastBreath: boolean,
 *   body: BodyMark | null,
 *   underline: ReadonlyArray<UnderlineBar> | null,
 *   beyondDeath: string | null,
 *   drift: boolean,
 * }>} HealthMarks
 *   The add-on marks of one event: saved ("Saved" and the rim flash), lastBreath
 *   (the smolder), body (their geometry, or null when the plan row lacks it, so
 *   neither body mark is drawn), underline (the share bars, or null),
 *   beyondDeath (the tag text, or null; records() keeps it only when
 *   beyondDeathClear says it fits) and drift (Battle Text Detail's 6 px
 *   drift for the number).
 */

/** Number sizes in CSS pixels: ordinary, for an Ultimate's or a killing blow's target, and Battle Text Detail's top tier. */
export const NUMBER_SIZES = Object.freeze({ ordinary: 13, large: 15, severe: 18 });

/** The pop: the number rises this many CSS pixels into place over popMs (popLargeMs at 15 px). */
export const NUMBER_POP = Object.freeze({ rise: 4, popMs: 80, popLargeMs: 120 });

/**
 * Battle Text Detail numbers (store numberSize, popState, groupSpec and groupSvg):
 * mediumShare and largeShare are the life shares that start the 15 px and 18 px
 * tiers; drift is the extra rise in CSS px and driftMs its length, during which the
 * number settles from scale 1.15 (15 px) or 1.2 (18 px) over the first 120 ms;
 * underlineOffset is the underline's distance below the number's baseline;
 * backingWidth, basicBar and ultimateBar are stroke widths; barGap is the gap
 * between bars and minimumBar the shortest bar; tagSize, tagGap and tagOpacity
 * place the 11 px "Beyond Death" tag 4 px right of the number, on its baseline,
 * at 75%; tagClearance is the gap the tag keeps from other numbers' boxes;
 * overkillMinimum is the smallest overshoot that gets a tag; sourceTolerance is
 * how far the recorded per-source rows may miss the recorded total and still be
 * drawn (store recordedRows).
 */
export const DETAIL = Object.freeze({
  mediumShare: 0.1,
  largeShare: 0.3,
  drift: 6,
  driftMs: 600,
  underlineOffset: 3.5,
  backingWidth: 5,
  basicBar: 2,
  ultimateBar: 3,
  barGap: 1.5,
  minimumBar: 4,
  tagSize: 11,
  tagGap: 4,
  tagClearance: 3,
  tagOpacity: 0.75,
  overkillMinimum: 0.05,
  sourceTolerance: 0.05,
});

/**
 * Saved Beat (store groupSpec, groupSvg and overLayer): the 13 px word stacks gap
 * px above the number, rises rise px over riseMs and fades in over fadeMs; the
 * rim is a strokeWidth px Dawn Gold ring at rimShare of the body radius less
 * half its stroke, and each of its two flashes takes flashMs at blow speed 1,
 * rising over flashRise of it.
 */
export const SAVED = Object.freeze({
  size: 13,
  gap: 3,
  rise: 3,
  riseMs: 200,
  fadeMs: 80,
  rimShare: 0.98,
  strokeWidth: 1.5,
  flashMs: 100,
  flashRise: 0.3,
});

/** Last Breath (store bandPieces): Cinder at this opacity, fading in over fadeMs at Settle. */
export const LAST_BREATH = Object.freeze({ opacity: 0.25, fadeMs: 200 });

/**
 * Cue rectangles in CSS pixels, centred on the cue point, for the plan's
 * collision allocator: netEffect (a health change with only its effect part on),
 * netText (the number: the longest rounded text, such as "−200" or "+9.9", at
 * 15 px with its 1.5 px Ink outline, as the store measures it), unchangedNetText
 * (an unchanged body: no number is drawn, so only a small box), regenerationEffect
 * and regenerationText (the Rest number's box as left, top, right and bottom
 * offsets from the cue point; it is centred like netText). The plan keeps every
 * cue at least half this box's diagonal away from the body, so a box larger than
 * the number pushes the number away from its body. The V5 add-ons grow a
 * number's box per event through lifeNumberCueBox.
 */
export const FOOTPRINT = Object.freeze({
  netEffect: Object.freeze({ width: 24, height: 20 }),
  netText: Object.freeze({ width: 40, height: 18 }),
  unchangedNetText: Object.freeze({ width: 24, height: 20 }),
  regenerationEffect: Object.freeze({ width: 24, height: 20 }),
  regenerationText: Object.freeze({ left: -20, top: -9, right: 20, bottom: 9 }),
});

/**
 * The plan's shared per-event allowance (30 nodes, 4 animations). An event draws
 * at most 15 nodes (group, hit box, Last Breath, rim, number set, number, backing,
 * five bars, tag and "Saved") and 4 animations (the number's pop, the "Saved"
 * rise, the rim flashes and the Last Breath fade).
 */
export const BUDGET = EVENT_BUDGET;

/**
 * Scrolling Battle Text switches every health-change part and the Rest number;
 * Regeneration Effects switches the Rest effect part (the Moon Dew pour itself
 * is drawn in the token by the Champions look). The V5 add-ons each own one part
 * per health outcome (saved_beat, last_breath and battle_text_detail), and Battle
 * Text Detail also owns the Rest number's detail part (its drift). The plan
 * resolves them into the event's paint parts savedBeat, lastBreath and
 * battleTextDetail; they refine a drawn number and never keep a switched-off one
 * drawn.
 */
export const PAINT_PARTS = paintParts([
  ...["damage", "healing", "unchanged"].flatMap(
    (outcome) =>
      /** @type {Array<[Record<string, string>, string]>} */ ([
        [
          { surface: "transient", kind: "net_health", outcome, part: "effect" },
          "scrolling_battle_text",
        ],
        [
          { surface: "transient", kind: "net_health", outcome, part: "battle_text" },
          "scrolling_battle_text",
        ],
        [
          { surface: "transient", kind: "net_health", outcome, part: "recipient_text" },
          "scrolling_battle_text",
        ],
        [
          { surface: "transient", kind: "net_health", outcome, part: "saved_beat" },
          "saved_beat",
        ],
        [
          { surface: "transient", kind: "net_health", outcome, part: "last_breath" },
          "last_breath",
        ],
        [
          {
            surface: "transient",
            kind: "net_health",
            outcome,
            part: "battle_text_detail",
          },
          "battle_text_detail",
        ],
      ]),
  ),
  [
    { surface: "transient", kind: "regeneration", part: "effect" },
    "regeneration_effects",
  ],
  [
    { surface: "transient", kind: "regeneration", part: "battle_text" },
    "scrolling_battle_text",
  ],
  [
    { surface: "transient", kind: "regeneration", part: "battle_text_detail" },
    "battle_text_detail",
  ],
]);

/** Approximate advance of one Atkinson Hyperlegible 700 character, in em. */
const CHARACTER_EM = 0.58;

/** The number's alphabetic baseline below its central line, in em (store "my"). */
const BASELINE_EM = 0.36;

/** The number's cap top above its central line, in em (store box top 0.74 em above the baseline). */
const CAP_TOP_EM = 0.38;

/** The ink outline around drawn text, in CSS px (half the 3 px stroke). */
const HALO = 1.5;

/**
 * Return the audience of a plan event: "researcher", "agent_pov" or null.
 *
 * plan.audience wins when it is one of the two names. Without a plan, a
 * NoSharedObs recipient cue ("recipient_cue" vocabulary) is Agent POV and
 * anything else is unknown (null). Pure.
 *
 * @param {JsonRecord} event
 * @param {JsonRecord | null | undefined} plan
 * @returns {"researcher" | "agent_pov" | null}
 */
function eventAudience(event, plan) {
  if (plan?.audience === "researcher" || plan?.audience === "agent_pov") {
    return plan.audience;
  }
  return event.authorityVocabulary === "recipient_cue" ? "agent_pov" : null;
}

/**
 * Return value when it is a finite number, else null. Numeric strings are not converted. Pure.
 *
 * @param {unknown} value
 * @returns {number | null}
 */
function finiteOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Round the size of a life change the store's way (health-and-combat-text fmtAbs).
 *
 * size is a finite, nonnegative amount in health points. Return "<0.1" below
 * 0.1; a whole number when the size is 9.95 or more, or within 0.05 of a whole
 * number ("41" for 41.05, "2" for 1.98); otherwise one decimal place ("3.3" for
 * 3.28). Pure.
 *
 * @param {number} size
 * @returns {string}
 */
function lifeAmountText(size) {
  if (size < 0.1) return "<0.1";
  return size >= 9.95 || Math.abs(size - Math.round(size)) < 0.05
    ? String(Math.round(size))
    : size.toFixed(1);
}

/**
 * Format a signed life number the store's way, short but never hiding a change.
 *
 * delta is the recorded finite change in health points (negative for a loss).
 * Return "−x" (U+2212) or "+x", where x is the size of the change rounded like
 * the design store (health-and-combat-text fmtAbs): a whole number when the size
 * is 9.95 or more, or within 0.05 of a whole number ("−41" for 41.05, "+2" for
 * 1.98); otherwise one decimal place ("−3.3" for 3.28, "−6.8"); and "<0.1" for a
 * size below 0.1 (so a tiny change still shows). Return null for zero, null,
 * undefined or a non-finite delta (no number). Formats only; nothing is
 * recomputed, and the tooltip keeps the exact value.
 *
 * @param {unknown} delta
 * @returns {string | null}
 */
export function lifeNumberText(delta) {
  const value = Number(delta);
  if (delta === null || delta === undefined || !Number.isFinite(value) || value === 0) {
    return null;
  }
  const visible = lifeAmountText(Math.abs(value));
  return value < 0 ? `−${visible}` : `+${visible}`;
}

/**
 * Return Battle Text Detail's tier for a health change, or null.
 *
 * event is a "net_health" plan event. The tier is |netDelta| over the row's
 * maximumHealth (the champion's public maximum life): 30% or more "large", 10%
 * or more "medium", else "small". Return null when either value is missing or
 * the maximum is not above 0. Pure.
 *
 * @param {JsonRecord} event
 * @returns {LifeWeight | null}
 */
export function lifeWeight(event) {
  const delta = finiteOrNull(event.netDelta);
  const maximum = finiteOrNull(event.maximumHealth);
  if (delta === null || maximum === null || !(maximum > 0)) return null;
  const share = Math.abs(delta) / maximum;
  if (share >= DETAIL.largeShare) return "large";
  return share >= DETAIL.mediumShare ? "medium" : "small";
}

/**
 * Return the px size of a Battle Text Detail tier: 13, 15 or 18. Pure.
 *
 * @param {LifeWeight} weight
 * @returns {13 | 15 | 18}
 */
function weightSize(weight) {
  if (weight === "large") return NUMBER_SIZES.severe;
  return weight === "medium" ? NUMBER_SIZES.large : NUMBER_SIZES.ordinary;
}

/**
 * Decide the number one health or rest event shows, or null for none.
 *
 * event is a "net_health" or "regeneration" plan event; plan (optional) is its
 * plan; options.detail (default false) says Battle Text Detail is on. Rules:
 * - No number when the change is zero or unknown, when its battle text is off
 *   (paint part battleText), or when the champion returned to its pad this tick
 *   (plan.bodies returned).
 * - Tone (every audience, both teams): rest for a "regeneration" event, else
 *   damage for a loss and heal for a gain. It reads only the event's own kind and
 *   sign, so Agent POV and an unknown audience (no plan, and an event that is not
 *   a "recipient_cue" row) get the same color as Oracle View.
 * - Oracle View: 15 px when the change killed the champion (healthAfter 0) or an
 *   Ultimate activation in plan.events targets it, else 13 px.
 * - Agent POV (or an unknown audience): only when the champion's plan body has a
 *   start (seen at both ends; an unknown audience needs no body); 13 px.
 * - Battle Text Detail (health changes only, both views): weight is lifeWeight
 *   and the size is the larger of the tier's size and the plain size above (Agent
 *   POV therefore never gets the Ultimate floor). Without a known tier, and for a
 *   Rest number, the plain size stays and weight is absent.
 * Return a frozen LifeNumber or null. Pure.
 *
 * @param {JsonRecord} event
 * @param {JsonRecord | null | undefined} plan
 * @param {{detail?: boolean}} [options]
 * @returns {LifeNumber | null}
 */
export function lifeNumber(event, plan, options = {}) {
  if (!paintPartEnabled(event, "battleText")) return null;
  const isRest = event.kind === "regeneration";
  const delta = isRest
    ? Number(event.value)
    : event.outcome === "unchanged"
      ? 0
      : Number(event.netDelta);
  const text = lifeNumberText(delta);
  if (text === null) return null;
  const key = event.recipientPresentationKey ?? null;
  const bodies = Array.isArray(plan?.bodies) ? plan.bodies : null;
  const body = bodies?.find((entry) => entry?.presentationKey === key) ?? null;
  if (body?.returned === true) return null;
  const audience = eventAudience(event, plan);
  if (audience === "agent_pov" && (body === null || body.start == null)) {
    return null;
  }
  /** @type {LifeNumber["tone"]} */
  const tone = isRest ? "rest" : delta < 0 ? "damage" : "heal";
  /** @type {13 | 15} */
  let plain = NUMBER_SIZES.ordinary;
  if (audience === "researcher") {
    const lethal = !isRest && Number(event.healthAfter) === 0;
    const ultimate = (Array.isArray(plan?.events) ? plan.events : []).some(
      (other) =>
        other?.kind === "activation" &&
        other.component === "ultimate" &&
        key !== null &&
        other.targetPresentationKey === key,
    );
    plain = lethal || ultimate ? NUMBER_SIZES.large : NUMBER_SIZES.ordinary;
  }
  const weight = options.detail === true && !isRest ? lifeWeight(event) : null;
  if (weight === null) {
    return Object.freeze({ text, tone, size: plain });
  }
  return Object.freeze({
    text,
    tone,
    size: /** @type {13 | 15 | 18} */ (Math.max(plain, weightSize(weight))),
    weight,
  });
}

/**
 * Read one switch for an event: the page's filters, else the plan, else the registry.
 *
 * filters is the page's visual filter state (or null); a boolean entry for
 * filterId wins. Otherwise event.paintParts[partKey] when the plan resolved it
 * (partKey null skips this step). Otherwise the switch's registry default
 * (`visual-filters.js` DEFAULT_VISUAL_FILTER_STATE). Pure.
 *
 * @param {Readonly<Record<string, unknown>> | null | undefined} filters
 * @param {JsonRecord} event
 * @param {string} filterId
 * @param {string | null} partKey
 * @returns {boolean}
 */
function switchOn(filters, event, filterId, partKey) {
  const fromFilters = isRecord(filters) ? filters[filterId] : undefined;
  if (typeof fromFilters === "boolean") return fromFilters;
  const fromPlan = partKey === null ? undefined : event.paintParts?.[partKey];
  if (typeof fromPlan === "boolean") return fromPlan;
  return (
    /** @type {Record<string, boolean>} */ (DEFAULT_VISUAL_FILTER_STATE)[filterId] ===
    true
  );
}

/**
 * Resolve the switches this look reads for one event (see HealthSwitches).
 *
 * filters is the page's filter state or null; event the plan event (its paint
 * parts are the fallback for the three add-ons). lifeStyle follows the Champions
 * look's own rule (`looks/champions.js` records: a vessel when Life Vessels is
 * on). Return a frozen record. Pure.
 *
 * @param {Readonly<Record<string, unknown>> | null | undefined} filters
 * @param {JsonRecord} event
 * @returns {HealthSwitches}
 */
export function healthSwitches(filters, event) {
  const vessel = switchOn(filters, event, "life_vessels", null);
  return Object.freeze({
    savedBeat: switchOn(filters, event, "saved_beat", "savedBeat"),
    lastBreath: switchOn(filters, event, "last_breath", "lastBreath"),
    detail: switchOn(filters, event, "battle_text_detail", "battleTextDetail"),
    lifeStyle: vessel ? "vessel" : "arc",
  });
}

/**
 * Return whether a heal kept this champion alive through lethal blows (Saved Beat).
 *
 * event is a plan event. True only for a "net_health" row that carries the
 * recorded Oracle totals (totalDamage, totalHealing) with healthBefore and
 * healthAfter, where healing is above 0, healthBefore minus totalDamage is 0 or
 * less (the blows alone would have killed it) and healthAfter is above 0 (it
 * lived). Agent POV rows carry no totals, so they are never saved. Pure; it
 * compares recorded values and recomputes nothing (store recsOf rec.saved).
 *
 * @param {JsonRecord} event
 * @returns {boolean}
 */
export function savedBeatHeld(event) {
  if (event?.kind !== "net_health") return false;
  const damage = finiteOrNull(event.totalDamage);
  const healing = finiteOrNull(event.totalHealing);
  const before = finiteOrNull(event.healthBefore);
  const after = finiteOrNull(event.healthAfter);
  return (
    damage !== null &&
    healing !== null &&
    before !== null &&
    after !== null &&
    healing > 0 &&
    before - damage <= 0 &&
    after > 0
  );
}

/**
 * Return whether this health change is a Last Breath warning.
 *
 * event is a plan event. True only for a "net_health" row whose champion is
 * alive after the tick (healthAfter above 0) and lost life (netDelta below 0) of
 * at least what it has left (−netDelta at least healthAfter): one more tick like
 * this ends it. In Agent POV the values are the observed health at both ends.
 * Pure (store recsOf rec.lastBreath).
 *
 * @param {JsonRecord} event
 * @returns {boolean}
 */
export function lastBreathHeld(event) {
  if (event?.kind !== "net_health") return false;
  const net = finiteOrNull(event.netDelta);
  const after = finiteOrNull(event.healthAfter);
  return net !== null && after !== null && after > 0 && net < 0 && -net >= after;
}

/**
 * Return the share underline bars of an Oracle health change, or null.
 *
 * event is a "net_health" row with totalDamage, totalHealing and damageSources
 * (each {sourcePresentationKey, sourceClassId, amount}: the recorded effective
 * damage of one attacker on this champion); plan.events supplies the Ultimate
 * join (an Ultimate activation from that attacker targeting this champion).
 * Return null unless damage is above 0 and the source amounts add up to the
 * recorded total within DETAIL.sourceTolerance (store recordedRows: no guessed
 * shares). Otherwise one bar per attacker, largest first (ties by class ID, then
 * key), in its class color ("trap" for a Hunter's Ultimate), then one "heal" bar
 * when healing is above 0. Pure; frozen output.
 *
 * @param {JsonRecord} event
 * @param {JsonRecord | null | undefined} plan
 * @returns {ReadonlyArray<UnderlineBar> | null}
 */
function underlineBars(event, plan) {
  const damage = finiteOrNull(event.totalDamage);
  const healing = finiteOrNull(event.totalHealing) ?? 0;
  const sources = asArray(event.damageSources)
    .filter((source) => isRecord(source) && finiteOrNull(source.amount) !== null)
    .map((source) => ({
      key:
        typeof source.sourcePresentationKey === "string"
          ? source.sourcePresentationKey
          : "",
      classId: Number.isInteger(source.sourceClassId)
        ? Number(source.sourceClassId)
        : 0,
      amount: Number(source.amount),
    }));
  if (damage === null || !(damage > 0) || sources.length === 0) return null;
  const sum = sources.reduce((total, source) => total + source.amount, 0);
  if (Math.abs(sum - damage) > DETAIL.sourceTolerance) return null;
  const key = event.recipientPresentationKey ?? null;
  const events = Array.isArray(plan?.events) ? plan.events : [];
  sources.sort(
    (left, right) =>
      right.amount - left.amount ||
      left.classId - right.classId ||
      (left.key < right.key ? -1 : left.key > right.key ? 1 : 0),
  );
  /** @type {UnderlineBar[]} */
  const bars = sources.map((source) => {
    const ultimate = events.some(
      (other) =>
        other?.kind === "activation" &&
        other.component === "ultimate" &&
        other.sourcePresentationKey === source.key &&
        key !== null &&
        other.targetPresentationKey === key,
    );
    const classKey = /** @type {string} */ (classTokenFromId(source.classId).cssKey);
    const paint = /** @type {UnderlineBar["paint"]} */ (
      classKey === "hunter" && ultimate ? "trap" : classKey
    );
    return Object.freeze({ amount: source.amount, paint, ultimate });
  });
  if (healing > 0) {
    bars.push(Object.freeze({ amount: healing, paint: "heal", ultimate: false }));
  }
  return Object.freeze(bars);
}

/**
 * Return the body geometry for Saved Beat's rim and Last Breath, or null.
 *
 * event supplies recipient (the END center, CSS px), recipientRadius (CSS px),
 * maximumHealth and healthAfter (health points); lifeStyle is the Champions
 * look's life style. Return null when any is missing or the radius or maximum is
 * not above 0. Pure; frozen output.
 *
 * @param {JsonRecord} event
 * @param {"vessel" | "arc"} lifeStyle
 * @returns {BodyMark | null}
 */
function bodyMark(event, lifeStyle) {
  const x = finiteOrNull(event.recipient?.x);
  const y = finiteOrNull(event.recipient?.y);
  const radius = finiteOrNull(event.recipientRadius);
  const maximum = finiteOrNull(event.maximumHealth);
  const after = finiteOrNull(event.healthAfter);
  if (
    x === null ||
    y === null ||
    radius === null ||
    maximum === null ||
    after === null
  ) {
    return null;
  }
  if (!(radius > 0) || !(maximum > 0)) return null;
  return Object.freeze({
    x,
    y,
    radius,
    share: Math.min(1, Math.max(0, after / maximum)),
    lifeStyle,
  });
}

/**
 * Decide the V5 add-on marks of one health or rest event.
 *
 * event is the plan event, plan its audience's plan (or null), switches the
 * resolved HealthSwitches and number the event's LifeNumber (or null). Rules:
 * - Only "net_health" rows get saved, lastBreath, underline and beyondDeath; a
 *   returned champion (plan.bodies returned) gets nothing.
 * - saved: Saved Beat on, Oracle View, and savedBeatHeld.
 * - lastBreath: Last Breath on and lastBreathHeld, in Oracle View, or in Agent
 *   POV when the champion's plan body has a start (seen at both ends of the tick).
 *   An unknown audience gets none.
 * - underline and beyondDeath: Battle Text Detail on, Oracle View and a number
 *   drawn; beyondDeath is beyondDeathText.
 * - drift: Battle Text Detail on and a number drawn (both views and Rest).
 * - body: bodyMark when saved or lastBreath, else null.
 * Return a frozen HealthMarks. Pure.
 *
 * @param {JsonRecord} event
 * @param {JsonRecord | null | undefined} plan
 * @param {HealthSwitches} switches
 * @param {LifeNumber | null} number
 * @returns {HealthMarks}
 */
export function healthMarks(event, plan, switches, number) {
  const audience = eventAudience(event, plan);
  const key = event.recipientPresentationKey ?? null;
  const body = Array.isArray(plan?.bodies)
    ? (plan.bodies.find((entry) => entry?.presentationKey === key) ?? null)
    : null;
  const health = event.kind === "net_health" && body?.returned !== true;
  const oracle = audience === "researcher";
  const seen =
    oracle || (audience === "agent_pov" && body !== null && body.start != null);
  const saved = health && oracle && switches.savedBeat && savedBeatHeld(event);
  const lastBreath = health && seen && switches.lastBreath && lastBreathHeld(event);
  const detail = switches.detail && number !== null;
  const tag = beyondDeathText(event);
  return Object.freeze({
    saved,
    lastBreath,
    body: saved || lastBreath ? bodyMark(event, switches.lifeStyle) : null,
    underline: health && oracle && detail ? underlineBars(event, plan) : null,
    beyondDeath: health && oracle && detail ? tag : null,
    drift: detail,
  });
}

/**
 * Return the Beyond Death tag of a killing blow that overshot, or null.
 *
 * event is a plan row. When it is a "net_health" row whose change killed the
 * champion (healthAfter 0) and whose recorded totalDamage exceeded healthBefore
 * by DETAIL.overkillMinimum or more, return "Beyond Death N" with N the overshoot
 * rounded like lifeNumberText (store groupSpec). Agent POV rows carry no
 * totalDamage, so they never get one. Pure.
 *
 * @param {JsonRecord} event
 * @returns {string | null}
 */
function beyondDeathText(event) {
  if (event?.kind !== "net_health") return null;
  const before = finiteOrNull(event.healthBefore);
  const damage = finiteOrNull(event.totalDamage);
  if (before === null || damage === null || finiteOrNull(event.healthAfter) !== 0) {
    return null;
  }
  const overkill = damage - before;
  return overkill >= DETAIL.overkillMinimum
    ? `Beyond Death ${lifeAmountText(overkill)}`
    : null;
}

/**
 * Return the extra cue box the V5 add-ons need for a health number, or null.
 *
 * event is a "net_health" plan row whose paint parts are resolved
 * (`choreography-plan.js` paintedOutcomeCueDimensions adds this box to the
 * FOOTPRINT box). With battleTextDetail on, the box covers the number at its
 * Battle Text Detail size (lifeWeight; 15 px is already covered) and, when the
 * row carries recorded damage sources, the share underline below it. With
 * savedBeat on and savedBeatHeld, it also covers "Saved" stacked 3 px above the
 * number (or alone when the number is 0). The Beyond Death tag is not reserved
 * (see beyondDeathClear). Return null when no
 * add-on applies, or Battle Text Detail is on with no number to draw (the plain
 * FOOTPRINT box is then unchanged), else a frozen {left, top, right, bottom} in
 * CSS px around the cue point. Pure.
 *
 * @param {JsonRecord} event
 * @returns {Readonly<{left: number, top: number, right: number, bottom: number}> | null}
 */
export function lifeNumberCueBox(event) {
  if (event?.kind !== "net_health") return null;
  const parts = isRecord(event.paintParts) ? event.paintParts : {};
  const detail = parts.battleTextDetail === true;
  const saved = parts.savedBeat === true && savedBeatHeld(event);
  const text = event.outcome === "unchanged" ? null : lifeNumberText(event.netDelta);
  if (!saved && (!detail || text === null)) return null;
  const weight = detail ? lifeWeight(event) : null;
  const size = weight === "large" ? NUMBER_SIZES.severe : NUMBER_SIZES.large;
  const width = text === null ? 0 : estimatedTextWidth(text, size);
  let half = text === null ? 0 : width / 2 + HALO;
  let top = text === null ? 0 : -(CAP_TOP_EM * size + HALO);
  let bottom = text === null ? 0 : 0.42 * size + HALO;
  if (detail && text !== null && asArray(event.damageSources).length > 0) {
    bottom =
      BASELINE_EM * size + DETAIL.underlineOffset + DETAIL.backingWidth / 2 + HALO;
  }
  if (saved) {
    const savedHalf = estimatedTextWidth("Saved", SAVED.size) / 2 + HALO;
    half = Math.max(half, savedHalf);
    top =
      text === null
        ? -(CAP_TOP_EM * SAVED.size + HALO)
        : top - SAVED.gap - (CAP_TOP_EM + 0.42) * SAVED.size;
    if (text === null) bottom = 0.42 * SAVED.size + HALO;
  }
  return Object.freeze({ left: -half, top, right: half, bottom });
}

/**
 * Return the Beyond Death tag's box beside a drawn number, in CSS px.
 *
 * x and y are the number's cue (its center); number is the drawn LifeNumber and
 * tag the tag text. The tag starts DETAIL.tagGap px right of the number's
 * estimated right edge, shares its baseline and keeps the 1.5 px Ink halo.
 * Return {left, top, right, bottom}. Pure.
 *
 * @param {number} x
 * @param {number} y
 * @param {LifeNumber} number
 * @param {string} tag
 * @returns {{left: number, top: number, right: number, bottom: number}}
 */
function beyondDeathBox(x, y, number, tag) {
  const left = x + estimatedTextWidth(number.text, number.size) / 2 + DETAIL.tagGap;
  const baseline = y + BASELINE_EM * number.size;
  return {
    left: left - HALO,
    top: baseline - (CAP_TOP_EM + BASELINE_EM) * DETAIL.tagSize - HALO,
    right: left + estimatedTextWidth(tag, DETAIL.tagSize) + HALO,
    bottom: baseline + 0.06 * DETAIL.tagSize + HALO,
  };
}

/**
 * Return whether the Beyond Death tag fits beside its number (the store sheds it otherwise).
 *
 * event is the plan row (its cue, recipientPresentationKey and recipientRadius);
 * plan its audience's plan (plan.bodies END spots in world units, plan.events
 * with their reserved cueBounds); number the drawn LifeNumber; tag the tag text;
 * transform the render's projection (worldToScreen, optional viewportBounds).
 * True only when the tag's box (beyondDeathBox) touches no other plan body
 * (projected END spot, the recipient's radius; corpses count), no other row's
 * cueBounds grown by DETAIL.tagClearance px, and stays inside
 * transform.viewportBounds when given. Without a cue, a transform or a plan the
 * tag does not fit (false), so the plain number stays. Pure.
 *
 * @param {JsonRecord} event
 * @param {JsonRecord | null | undefined} plan
 * @param {LifeNumber} number
 * @param {string} tag
 * @param {JsonRecord | null | undefined} transform
 * @returns {boolean}
 */
export function beyondDeathClear(event, plan, number, tag, transform) {
  const x = finiteOrNull(event.cue?.x);
  const y = finiteOrNull(event.cue?.y);
  if (
    x === null ||
    y === null ||
    !plan ||
    typeof transform?.worldToScreen !== "function"
  ) {
    return false;
  }
  const box = beyondDeathBox(x, y, number, tag);
  const view = transform.viewportBounds;
  if (
    isRecord(view) &&
    (box.left < view.left ||
      box.right > view.right ||
      box.top < view.top ||
      box.bottom > view.bottom)
  ) {
    return false;
  }
  const radius = finiteOrNull(event.recipientRadius) ?? 0;
  const key = event.recipientPresentationKey ?? null;
  for (const body of asArray(plan.bodies)) {
    if (!Array.isArray(body?.end) || body.presentationKey === key) continue;
    const center = transform.worldToScreen(body.end);
    const dx = Math.max(box.left - center.x, 0, center.x - box.right);
    const dy = Math.max(box.top - center.y, 0, center.y - box.bottom);
    if (dx * dx + dy * dy < radius * radius) return false;
  }
  for (const other of asArray(plan.events)) {
    const bounds = other?.cueBounds;
    if (other?.eventId === event.eventId || !isRecord(bounds)) continue;
    const pad = DETAIL.tagClearance;
    if (
      box.left < bounds.right + pad &&
      bounds.left - pad < box.right &&
      box.top < bounds.bottom + pad &&
      bounds.top - pad < box.bottom
    ) {
      return false;
    }
  }
  return true;
}

/**
 * Estimate the drawn width of text at a font size, in CSS px.
 *
 * text is the string; size the font size in px. Atkinson Hyperlegible 700 draws
 * digits, signs and letters about CHARACTER_EM wide, so this is characters times
 * size times CHARACTER_EM. An estimate: the look has no font metrics when it
 * builds. Pure.
 *
 * @param {string} text
 * @param {number} size
 * @returns {number}
 */
function estimatedTextWidth(text, size) {
  return [...text].length * size * CHARACTER_EM;
}

/**
 * Return the drawing record for a health change or rest heal, or [] otherwise.
 *
 * input.event is one admitted plan event; input.plan (optional) is its plan;
 * input.filters (optional) is the page's visual filter state; input.transform
 * (optional) is the render's projection, read only to test the Beyond Death
 * tag's room. A "net_health" event gives {kind: "net-health", event, number,
 * switches, marks, tagClear}; a "regeneration" event gives {kind:
 * "regeneration", ...} with the same fields; anything else gives []. switches is
 * healthSwitches(filters, event); number is lifeNumber(event, plan, {detail})
 * (the number to draw, or null: the event is still owned by this look, so its
 * hit box and tooltip stay); marks is healthMarks with beyondDeath kept only
 * when tagClear; tagClear is beyondDeathClear (false when there is no tag).
 * Pure data: no DOM.
 *
 * @param {{
 *   event?: JsonRecord | null,
 *   plan?: JsonRecord | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   transform?: JsonRecord | null,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const event = input.event;
  if (event?.kind !== "net_health" && event?.kind !== "regeneration") {
    return Object.freeze([]);
  }
  const plan = input.plan ?? null;
  const switches = healthSwitches(input.filters ?? null, event);
  const number = lifeNumber(event, plan, { detail: switches.detail });
  const marks = healthMarks(event, plan, switches, number);
  const tagClear =
    number !== null &&
    marks.beyondDeath !== null &&
    beyondDeathClear(event, plan, number, marks.beyondDeath, input.transform);
  return Object.freeze([
    Object.freeze({
      kind: event.kind === "net_health" ? "net-health" : "regeneration",
      event,
      number,
      switches,
      marks: tagClear ? marks : Object.freeze({ ...marks, beyondDeath: null }),
      tagClear,
    }),
  ]);
}

/**
 * Draw health-change and rest numbers, with their add-ons, into the painter's groups.
 *
 * records come from records(); doc creates nodes; mount (optional) is the
 * painter's EventMount (without it, new detached settled groups are made). When
 * mount.plan is present the number and marks are decided again with that plan
 * and the record's switches, so the Agent POV rules always apply; the Beyond
 * Death tag is drawn only when the record says it fits (tagClear). Each event gets
 * a hit box (a primary-button press on it does not reach the arena; class
 * combat-number__hit plus combat-net__hit or combat-regeneration__hit), then in
 * this order: the Last Breath smolder on the body (`path.combat-last-breath`,
 * data-style "vessel" or "arc"), the Saved Beat rim
 * (`circle.combat-saved-rim`, END opacity 0, only while playing), and the number
 * as `text.combat-number` with data-tone, data-size and, with Battle Text Detail,
 * data-weight, centred on the plan's cue. When the number has extras (share
 * underline, Beyond Death tag or "Saved") they share a `g.combat-number-set`
 * with it; "Saved" alone (`text.combat-saved`) stands at the cue when the number
 * is 0. mount.group gets data-net-delta (health change) or data-value (rest).
 * In normal motion, when mount is not settled, these animations are added to
 * mount.animationSpecs (fill "backwards"; the DOM holds END): the number's (or
 * set's) 4 px pop at the event's phaseStart, or with Battle Text Detail a 600 ms
 * pop and 6 px drift; the "Saved" rise; the rim's two flashes; the Last Breath
 * fade at Settle. Reduced motion adds only the rim's one steady flash. Return the
 * groups drawn into. No leader line is drawn: mount.connector stays empty.
 * Export-safe SVG only (flat fills and strokes from CSS tokens, opacity, dash
 * patterns, transforms, text with paint-order); installs pointerdown listeners.
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
  const target = mount ?? detachedEventMount(doc, record.event);
  for (const each of records) {
    const switches = /** @type {HealthSwitches} */ (
      each.switches ?? healthSwitches(null, each.event)
    );
    const plan = mount?.plan ? /** @type {JsonRecord} */ (mount.plan) : null;
    const number = plan
      ? lifeNumber(each.event, plan, { detail: switches.detail })
      : /** @type {LifeNumber | null} */ (each.number);
    const decided = plan
      ? healthMarks(each.event, plan, switches, number)
      : /** @type {HealthMarks} */ (
          each.marks ?? healthMarks(each.event, null, switches, number)
        );
    // The tag's room was tested in records() with the render's projection.
    const marks =
      each.tagClear === true || decided.beyondDeath === null
        ? decided
        : { ...decided, beyondDeath: null };
    buildNumber(doc, target, each.event, number, marks);
  }
  return mountNodes(target);
}

/**
 * Return whether the mount is drawing a played (not settled) installation.
 *
 * target.options.settled true or motionMode "off" means settled; a mount with
 * no plan is settled too. Pure.
 *
 * @param {EventMount} target
 * @returns {boolean}
 */
function playing(target) {
  const options = /** @type {Record<string, any>} */ (target.options ?? {});
  return (
    Boolean(target.plan) && options.settled !== true && options.motionMode !== "off"
  );
}

/**
 * Draw one event's hit box, body marks and number, and queue their animations.
 *
 * doc creates nodes; target is the event mount; event the plan event; number
 * the LifeNumber or null; marks the HealthMarks. A missing recipient draws
 * nothing. Return undefined.
 *
 * @param {Document} doc
 * @param {EventMount} target
 * @param {JsonRecord} event
 * @param {LifeNumber | null} number
 * @param {HealthMarks} marks
 */
function buildNumber(doc, target, event, number, marks) {
  const { group } = target;
  if (!event.recipient) {
    return;
  }
  if (event.kind === "net_health") {
    group.dataset.netDelta = String(event.netDelta);
  } else {
    group.dataset.value = String(event.value);
  }
  // The hit box keeps the event's old class (combat-net__hit or
  // combat-regeneration__hit) so pointer and hover checks still find it.
  const hit = effectNode(doc, "rect", {
    class: `combat-number__hit ${event.kind === "net_health" ? "combat-net__hit" : "combat-regeneration__hit"}`,
    "aria-hidden": "true",
  });
  hit.addEventListener("pointerdown", (pointerEvent) => {
    if (pointerEvent.button === 0) {
      pointerEvent.stopPropagation();
    }
  });
  group.append(hit);
  if (marks.lastBreath && marks.body !== null) {
    const smolder = effectNode(doc, "path", {
      class: "combat-last-breath",
      "data-style": marks.body.lifeStyle,
      opacity: LAST_BREATH.opacity,
    });
    group.append(smolder);
    queueLastBreath(target, event, smolder);
  }
  if (marks.saved && marks.body !== null && playing(target)) {
    const rim = effectNode(doc, "circle", {
      class: "combat-saved-rim",
      "stroke-width": SAVED.strokeWidth,
      opacity: 0,
    });
    group.append(rim);
    queueRimFlashes(target, event, rim);
  }
  const extras =
    number !== null &&
    (marks.underline !== null || marks.beyondDeath !== null || marks.saved);
  /** @type {SVGElement | null} */
  let savedWord = null;
  if (marks.saved) {
    savedWord = effectNode(doc, "text", {
      class: "combat-saved",
      "font-size": SAVED.size,
      "font-weight": 700,
      "text-anchor": "middle",
      "dominant-baseline": "central",
      "stroke-width": 3,
      "stroke-linejoin": "round",
      "paint-order": "stroke",
    });
    savedWord.textContent = "Saved";
  }
  if (number !== null) {
    const text = effectNode(doc, "text", {
      class: "combat-number",
      "data-tone": number.tone,
      "data-size": String(number.size),
      "data-weight": number.weight ?? null,
      "font-size": number.size,
      "font-weight": 700,
      "text-anchor": "middle",
      "dominant-baseline": "central",
      "stroke-width": 3,
      "stroke-linejoin": "round",
      "paint-order": "stroke",
    });
    text.textContent = number.text;
    if (extras) {
      const set = effectNode(doc, "g", { class: "combat-number-set" });
      set.append(text, ...underlineNodes(doc, marks.underline));
      if (marks.beyondDeath !== null) {
        const tag = effectNode(doc, "text", {
          class: "combat-number__tag",
          "font-size": DETAIL.tagSize,
          "font-weight": 700,
          "fill-opacity": DETAIL.tagOpacity,
          "text-anchor": "start",
          "dominant-baseline": "central",
          "stroke-width": 3,
          "stroke-linejoin": "round",
          "paint-order": "stroke",
        });
        tag.textContent = marks.beyondDeath;
        set.append(tag);
      }
      if (savedWord !== null) set.append(savedWord);
      group.append(set);
      queuePop(target, event, set, number, marks.drift);
    } else {
      group.append(text);
      queuePop(target, event, text, number, marks.drift);
    }
  } else if (savedWord !== null) {
    group.append(savedWord);
  }
  if (savedWord !== null) {
    queueSavedRise(target, event, savedWord);
  }
  placeNumber(group, event, number, marks);
}

/**
 * Create the share underline nodes: one Ink backing line, then one line per bar.
 *
 * doc creates nodes; bars the UnderlineBar list or null (no nodes). Each bar line
 * carries class combat-underline__bar, data-paint (class key, "trap" or "heal")
 * and stroke-width 2 or 3 (Ultimate); placeNumber sets their geometry.
 * Export-safe. Return the nodes in drawing order.
 *
 * @param {Document} doc
 * @param {ReadonlyArray<UnderlineBar> | null} bars
 * @returns {SVGElement[]}
 */
function underlineNodes(doc, bars) {
  if (bars === null || bars.length === 0) return [];
  return [
    effectNode(doc, "line", {
      class: "combat-underline__backing",
      "stroke-width": DETAIL.backingWidth,
      "stroke-linecap": "round",
    }),
    ...bars.map((bar) =>
      effectNode(doc, "line", {
        class: "combat-underline__bar",
        "data-paint": bar.paint,
        "stroke-width": bar.ultimate ? DETAIL.ultimateBar : DETAIL.basicBar,
        "stroke-linecap": "butt",
      }),
    ),
  ];
}

/**
 * Return the store's ease-out curve, cubic-bezier(0, 0, 0.58, 1), at progress p.
 *
 * p is clamped to 0..1. The bezier's x is solved by bisection (24 steps), then
 * its y is returned. Pure.
 *
 * @param {number} p
 * @returns {number}
 */
function easeOut(p) {
  const x = Math.min(1, Math.max(0, p));
  // Control points (0, 0) and (0.58, 1): x(t) = 3(1-t)t^2 * 0.58 + t^3 and
  // y(t) = 3(1-t)t^2 + t^3.
  const bx = (/** @type {number} */ t) => 3 * (1 - t) * t * t * 0.58 + t * t * t;
  const by = (/** @type {number} */ t) => 3 * (1 - t) * t * t + t * t * t;
  let low = 0;
  let high = 1;
  for (let step = 0; step < 24; step += 1) {
    const middle = (low + high) / 2;
    if (bx(middle) < x) low = middle;
    else high = middle;
  }
  return by((low + high) / 2);
}

/**
 * Return the progress x at which the ease-out curve (easeOut) reaches value y.
 *
 * y is clamped to 0..1. The bezier's parameter is solved for y by bisection (24
 * steps), then its x is returned. Pure; the inverse of easeOut.
 *
 * @param {number} y
 * @returns {number}
 */
function easeOutInverse(y) {
  const target = Math.min(1, Math.max(0, y));
  const bx = (/** @type {number} */ t) => 3 * (1 - t) * t * t * 0.58 + t * t * t;
  const by = (/** @type {number} */ t) => 3 * (1 - t) * t * t + t * t * t;
  let low = 0;
  let high = 1;
  for (let step = 0; step < 24; step += 1) {
    const middle = (low + high) / 2;
    if (by(middle) < target) low = middle;
    else high = middle;
  }
  return bx((low + high) / 2);
}

/**
 * Where the painter shows a health group at full strength, as shares of the
 * event's [phaseStart, phaseEnd] window: `choreography-painter.js` fades a
 * net_health group in to 1 at keyframe 0.18 and out from keyframe 0.72 under an
 * "ease-out" timing over the whole window, so in real time full strength runs
 * from about 0.113 to about 0.533 of the window. The Saved Beat rim flashes
 * inside it.
 */
export const GROUP_FULL_STRENGTH = Object.freeze({
  from: easeOutInverse(0.18),
  to: easeOutInverse(0.72),
});

/**
 * Return the Battle Text Detail pop-and-drift keyframes (store popState).
 *
 * big is the starting scale (1, 1.15 or 1.2). Over DETAIL.driftMs the number
 * fades in over the first 80 ms, rises 4 px (ease-out-quad over 80 ms) plus
 * DETAIL.drift px (ease-out over the whole drift), and settles its scale over
 * 120 ms (ease-out); the curve is sampled at 0, 20, 40, 60, 80, 120, 200, 300,
 * 450 and 600 ms with linear steps between. Pure; returns new keyframes.
 *
 * @param {number} big
 * @returns {Keyframe[]}
 */
function driftKeyframes(big) {
  return [0, 20, 40, 60, 80, 120, 200, 300, 450, 600].map((ms) => {
    const pop = Math.min(1, ms / NUMBER_POP.popMs);
    const rise =
      NUMBER_POP.rise * (1 - (1 - (1 - pop) * (1 - pop))) +
      DETAIL.drift * (1 - easeOut(ms / DETAIL.driftMs));
    const scale = big - (big - 1) * easeOut(ms / 120);
    return {
      offset: ms / DETAIL.driftMs,
      opacity: Math.round(pop * 1000) / 1000,
      transform: `translate(0px, ${Math.round(rise * 1000) / 1000}px) scale(${Math.round(scale * 10000) / 10000})`,
    };
  });
}

/**
 * Add the number's pop to the painter's animation list (normal motion only).
 *
 * target.options.settled or a motionMode other than "normal" adds nothing.
 * Without drift, one spec rises element (the number, or its set) 4 px into place
 * and fades it in over 80 ms (120 ms at 15 px, which also settles from 1.15x).
 * With drift (Battle Text Detail), one 600 ms spec of driftKeyframes (from 1.2x
 * at 18 px). Both start at event.phaseStart in real ms, with fill "backwards" so
 * the DOM holds END. Return undefined.
 *
 * @param {EventMount} target
 * @param {JsonRecord} event
 * @param {SVGElement} element
 * @param {LifeNumber} number
 * @param {boolean} drift
 */
function queuePop(target, event, element, number, drift) {
  const options = /** @type {Record<string, any>} */ (target.options ?? {});
  const specs = /** @type {unknown[] | undefined} */ (target.animationSpecs);
  if (!Array.isArray(specs) || !target.plan || options.settled === true) return;
  if (options.motionMode !== "normal") return;
  const delay = Math.max(0, Number(event.phaseStart ?? 0));
  const plan = /** @type {Record<string, any>} */ (target.plan);
  if (drift) {
    const big =
      number.size === NUMBER_SIZES.severe
        ? 1.2
        : number.size === NUMBER_SIZES.large
          ? 1.15
          : 1;
    specs.push(
      animationSpec(
        element,
        driftKeyframes(big),
        { delay, duration: DETAIL.driftMs, easing: "linear", fill: "backwards" },
        plan,
        event,
        "number-pop",
      ),
    );
    return;
  }
  const large = number.size === NUMBER_SIZES.large;
  specs.push(
    animationSpec(
      element,
      [
        {
          opacity: 0,
          transform: `translate(0px, ${NUMBER_POP.rise}px) scale(${large ? 1.15 : 1})`,
        },
        { opacity: 1, transform: "translate(0px, 0px) scale(1)" },
      ],
      {
        delay,
        duration: large ? NUMBER_POP.popLargeMs : NUMBER_POP.popMs,
        easing: "ease-out",
        fill: "backwards",
      },
      plan,
      event,
      "number-pop",
    ),
  );
}

/**
 * Add the "Saved" word's own rise (normal motion only, store groupSvg).
 *
 * At event.phaseStart the word fades in over SAVED.fadeMs and rises SAVED.rise px
 * over SAVED.riseMs (ease-out), fill "backwards". Settled or other motion adds
 * nothing. Return undefined.
 *
 * @param {EventMount} target
 * @param {JsonRecord} event
 * @param {SVGElement} word
 */
function queueSavedRise(target, event, word) {
  const options = /** @type {Record<string, any>} */ (target.options ?? {});
  const specs = /** @type {unknown[] | undefined} */ (target.animationSpecs);
  if (!Array.isArray(specs) || !playing(target) || options.motionMode !== "normal")
    return;
  const fade = SAVED.fadeMs / SAVED.riseMs;
  specs.push(
    animationSpec(
      word,
      [
        { offset: 0, opacity: 0, transform: `translate(0px, ${SAVED.rise}px)` },
        {
          offset: fade,
          opacity: 1,
          transform: `translate(0px, ${Math.round(SAVED.rise * (1 - easeOut(fade)) * 1000) / 1000}px)`,
        },
        { offset: 1, opacity: 1, transform: "translate(0px, 0px)" },
      ],
      {
        delay: Math.max(0, Number(event.phaseStart ?? 0)),
        duration: SAVED.riseMs,
        easing: "linear",
        fill: "backwards",
      },
      /** @type {Record<string, any>} */ (target.plan),
      event,
      "saved-rise",
    ),
  );
}

/**
 * Add the Saved Beat rim's flashes (store overLayer: exactly two flashes).
 *
 * The flashes play inside the window where the painter shows the health group at
 * full strength (GROUP_FULL_STRENGTH of the event's [phaseStart, phaseEnd]
 * window, about 11% to 53%). Normal motion: two Dawn Gold flashes,
 * each SAVED.flashMs divided by plan.beats.blowSpeed (default 1) long, rising
 * over its first 30%; the first starts as the group reaches full strength and the
 * second ends as the group starts to fade. Reduced motion: one steady flash over
 * the same window scaled by plan.phases.reducedTotal / plan.phases.total. A
 * window shorter than two flashes, or a missing phaseEnd, adds nothing. fill
 * "backwards"; the rim's DOM opacity stays 0 (no rim in the still). Return
 * undefined.
 *
 * @param {EventMount} target
 * @param {JsonRecord} event
 * @param {SVGElement} rim
 */
function queueRimFlashes(target, event, rim) {
  const options = /** @type {Record<string, any>} */ (target.options ?? {});
  const specs = /** @type {unknown[] | undefined} */ (target.animationSpecs);
  if (!Array.isArray(specs) || !playing(target)) return;
  const plan = /** @type {Record<string, any>} */ (target.plan);
  const start = Number(event.phaseStart ?? 0);
  const end = Number(event.phaseEnd);
  if (!Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) return;
  const window = end - start;
  const from = start + GROUP_FULL_STRENGTH.from * window;
  const to = start + GROUP_FULL_STRENGTH.to * window;
  if (options.motionMode === "reduced") {
    const scale =
      Number(plan.phases?.reducedTotal ?? 220) /
      Math.max(Number(plan.phases?.total ?? 900), 1);
    specs.push(
      animationSpec(
        rim,
        [{ opacity: 1 }, { opacity: 1 }],
        { delay: from * scale, duration: (to - from) * scale, fill: "none" },
        plan,
        event,
        "saved-rim",
      ),
    );
    return;
  }
  if (options.motionMode !== "normal") return;
  const speed = Number(plan.beats?.blowSpeed ?? 1);
  const flash = SAVED.flashMs / (Number.isFinite(speed) && speed > 0 ? speed : 1);
  const span = to - from;
  if (!(span >= 2 * flash)) return;
  const at = (/** @type {number} */ ms) => ms / span;
  specs.push(
    animationSpec(
      rim,
      [
        { offset: 0, opacity: 0 },
        { offset: at(SAVED.flashRise * flash), opacity: 1 },
        { offset: at(flash), opacity: 0 },
        { offset: at(span - flash), opacity: 0 },
        { offset: at(span - flash + SAVED.flashRise * flash), opacity: 1 },
        { offset: 1, opacity: 0 },
      ],
      { delay: from, duration: span, easing: "linear", fill: "backwards" },
      plan,
      event,
      "saved-rim",
    ),
  );
}

/**
 * Add the Last Breath fade-in at Settle (normal motion only, store bandPieces).
 *
 * The smolder fades from 0 to LAST_BREATH.opacity over LAST_BREATH.fadeMs divided
 * by plan.beats.blowSpeed, starting at the plan's Settle window start
 * (plan.beats.windows.settle[0], real ms), fill "backwards"; the DOM holds the
 * END opacity. Without that window nothing is added (the smolder shows with the
 * group). Return undefined.
 *
 * @param {EventMount} target
 * @param {JsonRecord} event
 * @param {SVGElement} smolder
 */
function queueLastBreath(target, event, smolder) {
  const options = /** @type {Record<string, any>} */ (target.options ?? {});
  const specs = /** @type {unknown[] | undefined} */ (target.animationSpecs);
  if (!Array.isArray(specs) || !playing(target) || options.motionMode !== "normal")
    return;
  const plan = /** @type {Record<string, any>} */ (target.plan);
  const settle = Number(plan.beats?.windows?.settle?.[0]);
  if (!Number.isFinite(settle)) return;
  const speed = Number(plan.beats?.blowSpeed ?? 1);
  specs.push(
    animationSpec(
      smolder,
      [{ opacity: 0 }, { opacity: LAST_BREATH.opacity }],
      {
        delay: Math.max(0, settle),
        duration:
          LAST_BREATH.fadeMs / (Number.isFinite(speed) && speed > 0 ? speed : 1),
        easing: "ease-out",
        fill: "backwards",
      },
      plan,
      event,
      "last-breath",
    ),
  );
}

/**
 * Return a rounded number for SVG geometry (3 decimals). Pure.
 *
 * @param {number} value
 * @returns {number}
 */
function n(value) {
  return Math.round(value * 1000) / 1000;
}

/**
 * Return the Last Breath path for a body.
 *
 * body is the BodyMark. "vessel": the closed region of the vessel above the
 * waterline (the vessel radius from `looks/champions.js` tokenGeometry), a
 * full circle when the share is 0.001 or less. "arc": the life ring from the
 * share's end clockwise to 12 o'clock (ring radius and width from tokenGeometry),
 * as an open arc path. Return {d, strokeWidth} (strokeWidth null for the
 * vessel's fill), or null when the life is full (nothing empty). Pure.
 *
 * @param {BodyMark} body
 * @returns {{d: string, strokeWidth: number | null} | null}
 */
function lastBreathShape(body) {
  if (body.share >= 0.999) return null;
  const geometry = tokenGeometry(body.radius);
  if (body.lifeStyle === "vessel") {
    const radius = geometry.vesselRadius;
    if (!(radius > 0)) return null;
    if (body.share <= 0.001) {
      return {
        d: `M${n(body.x - radius)} ${n(body.y)} A${n(radius)} ${n(radius)} 0 1 1 ${n(body.x + radius)} ${n(body.y)} A${n(radius)} ${n(radius)} 0 1 1 ${n(body.x - radius)} ${n(body.y)} Z`,
        strokeWidth: null,
      };
    }
    const y = body.y + radius * (1 - 2 * body.share);
    const offset = y - body.y;
    const chord = Math.sqrt(Math.max(0, radius * radius - offset * offset));
    const large = y > body.y ? 1 : 0;
    return {
      d: `M${n(body.x - chord)} ${n(y)} L${n(body.x + chord)} ${n(y)} A${n(radius)} ${n(radius)} 0 ${large} 0 ${n(body.x - chord)} ${n(y)} Z`,
      strokeWidth: null,
    };
  }
  const radius = geometry.lifeRadius;
  if (!(radius > 0)) return null;
  /**
   * Return the ring point at a share clockwise from 12 o'clock.
   *
   * @param {number} share
   */
  const point = (share) => {
    const angle = 2 * Math.PI * share;
    return [body.x + radius * Math.sin(angle), body.y - radius * Math.cos(angle)];
  };
  const [x0, y0] = point(body.share);
  const [x1, y1] = point(1);
  const large = 1 - body.share > 0.5 ? 1 : 0;
  return {
    d:
      body.share <= 0.001
        ? `M${n(x1)} ${n(y1)} A${n(radius)} ${n(radius)} 0 1 1 ${n(body.x)} ${n(body.y + radius)} A${n(radius)} ${n(radius)} 0 1 1 ${n(x1)} ${n(y1)}`
        : `M${n(x0)} ${n(y0)} A${n(radius)} ${n(radius)} 0 ${large} 1 ${n(x1)} ${n(y1)}`,
    strokeWidth: geometry.lifeWidth,
  };
}

/**
 * Place an event's hit box, number, its extras and its body marks.
 *
 * group belongs to the event; event gives recipient, cue bounds and position and
 * lane (CSS pixels); number and marks are the event's drawing (number may be
 * null). Without a recipient do nothing. Invalid bounds hide the hit box; a
 * missing cue position falls back to above the recipient (24 px plus 18 px per
 * lane). The number is centred on the cue; its underline sits
 * DETAIL.underlineOffset px below its baseline, its Beyond Death tag
 * DETAIL.tagGap px right of it on its baseline, and "Saved" SAVED.gap px above it
 * (or on the cue
 * when there is no number). The Last Breath path and the rim follow marks.body.
 * No leader line is drawn or moved (event.cueLeader is ignored). Return
 * undefined.
 *
 * @param {SVGElement} group
 * @param {JsonRecord} event
 * @param {LifeNumber | null} number
 * @param {HealthMarks | null} marks
 */
function placeNumber(group, event, number, marks) {
  if (!event.recipient) {
    return;
  }
  assignLayoutPlacement(
    group,
    event.cueLayoutKey,
    event.cueBounds,
    event.cueDisposition,
    event.cueCollisionFree,
  );
  const hit = group.querySelector(".combat-number__hit");
  if (hit) {
    const bounds = event.cueBounds;
    if (
      Number.isFinite(bounds?.left) &&
      Number.isFinite(bounds?.top) &&
      Number.isFinite(bounds?.right) &&
      Number.isFinite(bounds?.bottom) &&
      bounds.right >= bounds.left &&
      bounds.bottom >= bounds.top
    ) {
      setEffectAttributes(hit, {
        visibility: "visible",
        x: bounds.left,
        y: bounds.top,
        width: bounds.right - bounds.left,
        height: bounds.bottom - bounds.top,
      });
    } else {
      hit.setAttribute("visibility", "hidden");
    }
  }
  const lane = Number.isFinite(event.lane) ? Number(event.lane) : 0;
  const x = event.cue?.x ?? event.recipient.x;
  const y = event.cue?.y ?? event.recipient.y - 24 - lane * 18;
  const text = group.querySelector(".combat-number");
  if (text) {
    setEffectAttributes(text, { x, y });
  }
  const saved = group.querySelector(".combat-saved");
  if (saved) {
    const size = number?.size ?? 0;
    setEffectAttributes(saved, {
      x,
      y: number === null ? y : n(y - CAP_TOP_EM * size - SAVED.gap - 0.42 * SAVED.size),
    });
  }
  if (number !== null) {
    placeExtras(group, x, y, number, marks?.underline ?? null);
  }
  const body = marks?.body ?? null;
  const smolder = group.querySelector(".combat-last-breath");
  if (smolder) {
    const shape = body === null ? null : lastBreathShape(body);
    setEffectAttributes(smolder, {
      visibility: shape === null ? "hidden" : "visible",
      d: shape?.d ?? null,
      "stroke-width": shape?.strokeWidth ?? null,
    });
  }
  const rim = group.querySelector(".combat-saved-rim");
  if (rim && body !== null) {
    setEffectAttributes(rim, {
      cx: n(body.x),
      cy: n(body.y),
      r: n(Math.max(0, body.radius * SAVED.rimShare - SAVED.strokeWidth / 2)),
    });
  }
}

/**
 * Place a number's share underline and Beyond Death tag around the cue (x, y).
 *
 * group holds them (class combat-underline__backing, combat-underline__bar and
 * combat-number__tag); number is the drawn LifeNumber; bars the UnderlineBar
 * list (or null). The underline spans the larger of the number's estimated width
 * and the room the bars need (DETAIL.minimumBar each, DETAIL.barGap between),
 * centred under it; each bar's length is its share of that span (at least
 * DETAIL.minimumBar, the longer bars giving up the excess), and the Ink backing
 * reaches 1 px past both ends (store groupSvg). The tag starts DETAIL.tagGap px
 * right of the number's estimated edge, on the number's baseline
 * (beyondDeathBox). Return undefined.
 *
 * @param {SVGElement} group
 * @param {number} x
 * @param {number} y
 * @param {LifeNumber} number
 * @param {ReadonlyArray<UnderlineBar> | null} bars
 */
function placeExtras(group, x, y, number, bars) {
  const width = estimatedTextWidth(number.text, number.size);
  const baseline = y + BASELINE_EM * number.size;
  const tag = group.querySelector(".combat-number__tag");
  if (tag) {
    setEffectAttributes(tag, {
      x: n(x + width / 2 + DETAIL.tagGap),
      y: n(baseline - BASELINE_EM * DETAIL.tagSize),
    });
  }
  if (bars === null || bars.length === 0) return;
  const count = bars.length;
  const span = Math.max(width, count * DETAIL.minimumBar + (count - 1) * DETAIL.barGap);
  const available = span - DETAIL.barGap * (count - 1);
  const total = bars.reduce((sum, bar) => sum + bar.amount, 0) || 1;
  let lengths = bars.map((bar) =>
    Math.max(DETAIL.minimumBar, (available * bar.amount) / total),
  );
  const over = lengths.reduce((sum, length) => sum + length, 0) - available;
  if (over > 0) {
    const room =
      lengths.reduce(
        (sum, length) =>
          sum + (length > DETAIL.minimumBar ? length - DETAIL.minimumBar : 0),
        0,
      ) || 1;
    lengths = lengths.map((length) =>
      length > DETAIL.minimumBar
        ? length - (over * (length - DETAIL.minimumBar)) / room
        : length,
    );
  }
  const lineY = n(baseline + DETAIL.underlineOffset);
  const left = x - span / 2;
  const backing = group.querySelector(".combat-underline__backing");
  if (backing) {
    setEffectAttributes(backing, {
      x1: n(left - 1),
      y1: lineY,
      x2: n(left + span + 1),
      y2: lineY,
    });
  }
  const lines = [...group.querySelectorAll(".combat-underline__bar")];
  let at = left;
  lines.forEach((line, index) => {
    const length = lengths[index] ?? 0;
    setEffectAttributes(line, { x1: n(at), y1: lineY, x2: n(at + length), y2: lineY });
    at += length + DETAIL.barGap;
  });
}

/**
 * Move health-change or rest nodes to their new geometry (reprojection).
 *
 * nodes are [group, connector, ...] as build returned (only group is used: the
 * connector holds nothing of this look); records holds the event's new record
 * (its number and marks, decided with the same plan). The hit box, number,
 * extras and body marks move to the new cue and body; no text or node is
 * changed or added. Return undefined. Animations keep their elements.
 *
 * @param {ReadonlyArray<Element | null>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  const record = records[0];
  const [group] = nodes;
  if (record === undefined || !group) {
    return;
  }
  placeNumber(
    /** @type {SVGElement} */ (group),
    record.event,
    /** @type {LifeNumber | null} */ (record.number ?? null),
    /** @type {HealthMarks | null} */ (record.marks ?? null),
  );
}

/**
 * Return the tooltip for a health-change or rest record.
 *
 * A health change with its battle or recipient text on uses the health
 * explanation; otherwise (and for a rest heal) the shared event explanation.
 * Both read only the plan event. Pure; returns a tooltip descriptor.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  const event = record.event;
  const view = paintAwareExplanationEvent(event);
  return event.kind === "net_health" &&
    (paintPartEnabled(event, "battleText") || paintPartEnabled(event, "recipientText"))
    ? explainNetHealth(view)
    : explainChoreographyEvent(view);
}
