/**
 * @file The Herald Cards look (Runefield part "Match Moments"): the cards that
 * name a match's big moments over the top of the arena.
 *
 * It owns these drawings, all export-safe SVG:
 * - Herald plate (from the plan's `herald_card` row): a 40 px plate with
 *   swallowtail ends, a 2 px rule on its top edge (the scoring team's color
 *   at 70%, or Moon Silver at 50% for a neutral moment), its glyphs, an 11 px
 *   eyebrow and an 18 px title. heraldMoments(row, filters) is the moment
 *   finder: it reads only the row's facts and returns the tick's moments in
 *   the store's headline order (RANK). The first is the card's moment; up to
 *   two more join its eyebrow as short words ("Team A +2 · Trade"). The
 *   moments, with the store's words and glyphs:
 *   - Penta Takedown: one team lost five champions this tick (the fallen;
 *     "TEAM A +5 · TEAM B WIPED" / "Penta Takedown"; tier 1). It heads the
 *     card above Team Wiped, which a five-champion team always reaches too.
 *   - Team Wiped: every active champion of one team is fallen after this
 *     tick and one of them fell on it (fallen medallions, one per member;
 *     "TEAM B" / "Team Wiped"; a tier-1 moment). In another card's eyebrow it
 *     reads "Team B Wiped".
 *   - First Blood: the first tick with a fall, only in a match that began at
 *     tick 0 with 0-0 (the row's freshStart host fact): the fallen, "TEAM B
 *     +1" or "TRADED" when both teams lost one / "First Blood"; tier 1.
 *   - Match Point: a team's score rose this tick to exactly 1 point short of
 *     First To (its crest; "TEAM A NEEDS 1 POINT" or "BOTH TEAMS NEED 1
 *     POINT" / "Match Point"; neutral; tier 1). 2 points short is not Match
 *     Point.
 *   - Double, Triple or Quadruple Takedown: one team lost two, three or four
 *     this tick (the fallen; "TEAM A +2").
 *   - Takedown runs: a team's 5th, 10th, 15th or 20th (every 5th) takedown
 *     in a row with no loss of its own (the row's runs host fact): its crest
 *     and the words, no dots; "TEAM B" / "5 Unanswered". Every run ranks as
 *     the big run (runBig); "run" (under 5) stays for older facts.
 *   - Champion Streak (the `champion_streaks` switch, off by default): a
 *     champion that has now taken part in 5, 10, 15 or 20 (every 5th)
 *     takedowns without falling (the row's `streaks` facts): its living
 *     medallion and the words, no dots; "TEAM A MAGE" / "10 Takedowns
 *     Without Falling". Kill credit is shared, so the words say "took part",
 *     never "killed".
 *   - Rescue: a Priest's heal kept a champion alive through blows that would
 *     have killed it this tick (the row's rescues host fact): its living
 *     medallion in a Dawn Gold ring with a gold "+"; "HOLY WORD: SALVATION"
 *     or "LESSER HEAL" / "Rescue".
 *   - Ultimate Takedown: a fall whose damaging Ultimate (Charge, Freezing
 *     Trap or Crippling Poison) struck it (the caster's rune diamond and the
 *     fallen; "CHARGE · TEAM A +1").
 *   - Trade: both teams lost champions this tick (the fallen of each team
 *     with a divider; "TEAM A LOST 1 · TEAM B LOST 1"; neutral).
 *   - Scores Level (the owner's "Comeback"): the scores became level this
 *     tick after one team had led by 3 or more (the row's largestLeadBefore
 *     host fact): both teams' crests; "7 – 7" / "Scores Level"; neutral.
 *   - Ultimate Combo: two or more damaging Ultimates struck the same
 *     champion this tick: one rune diamond per caster; "CHARGE + FREEZING
 *     TRAP" / "Ultimate Combo"; neutral.
 *   - Ticks Left: 30 ticks before the draw tick while no team has reached
 *     First To (a Moon Silver crescent; "DRAW IF NO TEAM REACHES 20" / "30
 *     Ticks Left"; neutral). The owner removed the 10, 5, 4, 3, 2 and 1
 *     cards and the countdown chip (4 Oct 2026).
 *   A plain single takedown gets no card (its Fallen feed row tells it). With
 *   the Death Announcer switch off, the moments born from a fall (First
 *   Blood, Multi Takedown, Team Wiped, Trade, Ultimate Takedown, runs and
 *   streaks) are left out, as in the store. A tier-1 card (Team Wiped, First
 *   Blood, Match Point) also draws the Herald Rule: a 1 px line in its rule
 *   color at 35% across the arena's top band, behind the card. Each card's
 *   group carries `data-herald-moments`, the kinds it names in order (for
 *   example "multi trade"). Every moment card is this plate: there is no
 *   one-line band or chip form (owner, 4 Oct 2026). The plate enters at the
 *   row's phaseStart (the plan's: the first Fall beat, or the tick's settled
 *   time when only a Rescue, Ticks Left or a Combo without a fall is named,
 *   or the later time the painter's card queue gives it) and, when the
 *   caller passes the previous tick's row, holds in the same spot; a held
 *   card with an exitAt fades out at that time (the queue's hand-over).
 * - Lineup card (frame 0): the map, each team's crest and the medallions of
 *   its champions this audience may see, and the match rules.
 * - Verdict card (the final frame, in the Replay Viewer and, at game over, in
 *   the BattleClient and DevClient), the store's full Herald Cards verdict:
 *   the basis words, two team columns (crest, team, controller name, score
 *   and result; the left column is the team whose home end is on the left),
 *   the lead line of a horizon draw, Honors and the Match Summary and Show
 *   Arena buttons. Its band form (Show Arena, or the `moment_full_cards`
 *   switch off) is a herald plate: crest, score, "–", score, crest, the basis
 *   words and the result.
 * With `moment_full_cards` off the lineup shows as a one-line band and the
 * verdict as its herald plate. With
 * `match_moments` off nothing is drawn (the scorebug still shows score and
 * result).
 *
 * Placement (Sliding Top Placement, owner 4 Oct 2026): a herald plate always
 * hugs the top edge of the arena's visible area. It tries spots across the
 * whole width and picks the one that covers the fewest champions, then the
 * least champion area, then the one nearest the centre. Only when every
 * full-size spot covers someone does it try smaller sizes, never so small
 * that its smallest words go under 9 px, and it never turns into another
 * form. It never covers page chrome (such as the view tag) while a spot
 * clear of it exists. A card never silently vanishes: when no spot fits
 * anywhere it still draws at its smallest readable size, centred. A row drawn
 * again with the same arena rectangle (the next tick's hold, a repaint) keeps
 * the spot it was first given.
 *
 * Callers: `choreography-painter.js` (for `herald_card` plan rows, through
 * records({event, bounds, previous, avoid, chrome}), build(records, doc, mount)
 * and update(nodes, records); it owns the card queue and reads
 * MINIMUM_CARD_MS, CARD_FADE_OUT_MS and cardMoments); the clients' renderer
 * for the lineup and verdict through records({part, ...}) and
 * build(records, doc), plus
 * verdictHonours(matchHonors) (the frame's host fact turned into the
 * verdict's Honors rows), verdictButtons(record) and buildVerdictActions(doc,
 * record, onPress) for the verdict's two working buttons, and
 * buildMatchSummarySheet(doc, matchHonors, options) for the sheet its Match
 * Summary button opens (page chrome, never over the arena).
 *
 * Timing: the DOM always holds the END state (the card fully shown, or, for a
 * held card with an exitAt, hidden). The entry (opacity 0 to 1 over 240 ms,
 * unfurling from 92% width, medallions settling from 115%; the tier-1 Herald
 * Rule grows from the card's centre over 320 ms) and a held card's exit
 * (opacity 1 to 0 over CARD_FADE_OUT_MS from its exitAt) are Web Animations
 * descriptions with `fill: "backwards"` pushed to `mount.animationSpecs`,
 * timed from the row's `phaseStart` (real ms). Reduced motion fades only, over
 * 120 ms. No timers and no CSS animations.
 *
 * Agent POV boundary: a herald reads only the plan row it is given (the plan
 * builder names only the falls the team saw, and counts only those), or the
 * authorized arena scene and the public match facts the caller passes. A
 * streak is drawn only from the row's own `streaks` facts, so the row's owner
 * must count, in Agent POV, only the takedowns the team saw (blows it saw land
 * on falls it saw) and reset a count only on a fall it saw; this look never
 * counts history itself. The same holds for the row's runs, rescues,
 * freshStart and largestLeadBefore (the host's per-view `match_moments`
 * fact, viewer/match_moments.py), its combos and wiped teams (the plan's,
 * from the blows and bodies the team saw) and its public match facts (scores
 * before and after the tick, First To, the tick and the draw tick, which
 * every view's scorebug shows). The verdict's Honors come only from the
 * caller, which passes the frame's own host fact (`match_honors`): Agent POV
 * lists every champion too, because Honors come only once the game is over
 * and nothing is left to hide then. Spots avoid only the bodies
 * the caller passes (in Agent POV only the POV scene's bodies), so an unseen
 * champion never moves a card. Nothing here imports a researcher helper or
 * reads a researcher view or match summary.
 */
import { classIdentityFromId, teamLabelFromId } from "../agent-identity.js";
import { classAbilityNameV1 } from "../class-documentation.js";
import { createSvgIcon } from "../icons.js";
import { createSemanticDescriptor } from "../tooltip.js";
import {
  animationSpec,
  asArray,
  effectNode,
  isRecord,
  paintParts,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").JsonRecord} JsonRecord
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").EventMount} EventMount
 * @typedef {import("./look-support.js").Rectangle} Rectangle
 * @typedef {"a" | "b"} TeamKey
 * @typedef {{classKey: string | null, teamKey: TeamKey, fallen: boolean}} Medallion
 * @typedef {Readonly<{kind: string, [field: string]: any}>} HeraldMoment
 *   One moment heraldMoments found: its kind ("penta", "wiped", "firstBlood",
 *   "matchPoint", "multi", "multiBig", "run", "runBig", "streak", "rescue",
 *   "ultTakedown", "trade", "level", "combo" or "moonset") and its fields
 *   (listed at heraldMoments).
 * @typedef {Readonly<{
 *   champion: Readonly<{
 *     publicAgentId?: string,
 *     classId: number,
 *     teamId: number,
 *     title?: string,
 *   }>,
 *   takedowns: number,
 * }>} ChampionStreakFact
 *   One champion streak a herald_card row may carry in `streaks`: the living
 *   champion (its public identity: class ID 1-5, team ID 1 or 2 and its
 *   title, such as "Mage · ID 3") and takedowns, the number of takedowns it
 *   has now taken part in without falling (a positive integer; the store
 *   fires at 5 and 8). The row's owner decides who has a streak.
 * @typedef {Readonly<{
 *   key: "plate",
 *   width: number,
 *   height: number,
 *   minScale: number,
 * }>} BandForm
 *   How a top-band card is drawn: key "plate" (the 40 px plate, the only
 *   form since 4 Oct 2026), its unscaled drawn box in CSS px (tails
 *   included) and the smallest scale at which its smallest words stay
 *   readable (MIN_TEXT_PX).
 * @typedef {Readonly<{
 *   form: "plate",
 *   scale: number,
 *   left: number,
 *   top: number,
 *   width: number,
 *   height: number,
 *   covered: number,
 *   hit: number,
 *   distance: number,
 * }>} BandSpot
 *   Where a top-band card is drawn: its form, scale and drawn box in CSS px
 *   (left, top, width, height: tails included, after scaling), covered (how
 *   many champion circles it touches), hit (how much champion area it
 *   covers, px squared) and distance (from its centre to the band's centre,
 *   px). Spots compare by covered, then hit, then distance.
 * @typedef {Readonly<{
 *   id: "summary" | "show_arena",
 *   label: string,
 *   left: number,
 *   top: number,
 *   width: number,
 *   height: number,
 * }>} VerdictButton
 *   One Verdict card button and its box in CSS px in the arena's coordinates
 *   (the herald layer's), after the card's placement and scale.
 */

/**
 * Most medallions a herald plate draws. Only a tick with more than 7 falls
 * of one card's champions (a First Blood, Team Wiped or Multi Takedown of 8
 * to 10) reaches it; the eyebrow still gives the real counts, and the card
 * stays inside BUDGET.
 */
export const PLATE_MEDALLIONS_MAX = 7;

/** Most tied leaders one Honors cell draws before "+N". */
const HONOR_MEDALLIONS_MAX = 6;

/**
 * Smaller sizes a Verdict column's controller name tries, in CSS px, before it
 * is cut with "…" (heralds.css has a class for each: herald__name--13 and
 * herald__name--11). The full size is 15 px.
 */
const VERDICT_NAME_SIZES = Object.freeze([13, 11]);

/** Smallest words a card may shrink to, in CSS px (the store's 9 px). */
const MIN_TEXT_PX = 9;

/** Scales the band search tries, largest first (the store's list). */
const BAND_SCALES = Object.freeze([1, 0.9, 0.8, 0.7, 0.6, 0.5]);

/**
 * Sizes in CSS pixels of the plate, the one-line band and the full cards.
 * plate.medallionStep is the step between overlapping medallions (20: a 6 px
 * overlap) and plate.wipedStep Team Wiped's (18: an 8 px overlap, the
 * store's). The band sizes are the folded Lineup's one-line band.
 */
export const FOOTPRINT = Object.freeze({
  plate: Object.freeze({
    height: 40,
    padLeft: 8,
    padRight: 14,
    gap: 9,
    tailWidth: 14,
    tailHeight: 30,
    tailReach: 10,
    medallion: 26,
    medallionStep: 20,
    wipedStep: 18,
    ruleInset: 10,
    minScale: 0.75,
  }),
  band: Object.freeze({
    height: 22,
    pad: 10,
    gap: 6,
    crest: 12,
    tailReach: 7,
  }),
  fullCard: Object.freeze({ width: 372, pad: 14, medallion: 28 }),
  verdict: Object.freeze({
    padX: 12,
    padTop: 9,
    padBottom: 9,
    honorMedallion: 22,
    honorGap: 8,
    buttonHeight: 24,
    buttonGap: 8,
    buttonPadX: 10,
    minScale: 0.55,
  }),
  topGap: 8,
});

/**
 * Node and animation allowance of one card. A plate draws 10 nodes of its own
 * (group, body, inner group, two tails, panel, rule, glyph group, eyebrow and
 * title), 4 or 5 per medallion (the field-and-ring path, a Team B living
 * medallion's ink ring, and the class sigil: 3 nodes, 4 for the Mage), one
 * path per divider, diamond or crest, 7 or 8 for a Rescue
 * medallion (its medallion, gold ring and "+"), 2 for the Ticks Left
 * crescent, and 2 for a tier-1 card's Herald Rule (its group and line). The
 * most it draws (a tier-1 First Blood naming PLATE_MEDALLIONS_MAX fallen
 * Mages at 5 nodes each: 47) stays within 48 nodes. It starts at most 3
 * animations (card, glyphs, Herald Rule; a held card's exit fade 1). A
 * paused card keeps all of them, so
 * the persistent allowance equals the node allowance. The painter drops a
 * held card over this allowance, so it must cover the largest card.
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 48,
  animationsPerRecord: 3,
  persistentNodesPerRecord: 48,
});

/**
 * Every paint part this look draws, with the filter that switches it. The
 * herald_streak part is the Champion Streak moment on a herald plate.
 */
export const PAINT_PARTS = paintParts([
  [{ surface: "transient", kind: "herald_card" }, "match_moments"],
  [{ surface: "transient", kind: "herald_streak" }, "champion_streaks"],
  [{ surface: "transient", kind: "herald_lineup" }, "match_moments"],
  [{ surface: "transient", kind: "herald_verdict" }, "match_moments"],
]);

/** The Verdict card's two buttons, in order. */
export const VERDICT_BUTTONS = Object.freeze([
  Object.freeze({ id: "summary", label: "Match Summary" }),
  Object.freeze({ id: "show_arena", label: "Show Arena" }),
]);

/** Fallback band when neither bounds nor an anchor is given (CSS px). */
const DEFAULT_BOUNDS = Object.freeze({
  left: 0,
  top: 0,
  right: 960,
  bottom: 600,
  width: 960,
  height: 600,
});

/**
 * Names for 2, 3, 4 and 5 takedowns of one team in one tick ("Double
 * Takedown" ... "Penta Takedown"). A team has at most five champions; any
 * larger count would read "Multi Takedown".
 */
const MULTI_NAMES = Object.freeze(["", "", "Double", "Triple", "Quadruple", "Penta"]);

/**
 * The Minimum Card Time (owner, 4 Oct 2026): a moment card stays on screen at
 * least this long, in replay time ms (2 s at 1x, 1 s at 2x), before the next
 * card may replace it. The painter's card queue (choreography-painter.js)
 * owns the waiting; this look only draws the times it is given.
 */
export const MINIMUM_CARD_MS = 2000;

/**
 * How long a card being replaced takes to fade out, in replay time ms. The
 * next card starts to fade in only after it, so two cards are never on screen.
 */
export const CARD_FADE_OUT_MS = 150;

/** The XHTML namespace, for the verdict's working buttons (page chrome). */
const HTML_NAMESPACE = "http://www.w3.org/1999/xhtml";

/**
 * A herald row's spot, kept for the row's life: the arena rectangle key it
 * was placed in and the spot. Keyed by the row object itself, so a new plan
 * (a seek, a view change) always places afresh.
 *
 * @type {WeakMap<object, {boundsKey: string, spot: BandSpot}>}
 */
const SPOT_BY_ROW = new WeakMap();

/**
 * Estimate a bold text line's width in CSS px (no DOM measuring).
 *
 * text is the words; size the font size in px; caps true for the uppercase
 * eyebrow (adds its 0.12 em letter spacing). About 0.6 em per bold character.
 * Pure.
 *
 * @param {string} text
 * @param {number} size
 * @param {boolean} [caps] default false.
 * @returns {number}
 */
export function textWidth(text, size, caps = false) {
  const length = String(text).length;
  return length * size * 0.6 + (caps ? 0.12 * size * length : 0);
}

/**
 * Return the drawing records for one herald request.
 *
 * Every form returns [] when `filters.match_moments` is false. Accepted inputs:
 * - `{event, bounds?, previous?, avoid?, chrome?, filters?}`: event is a plan
 *   row of kind "herald_card" ({eventId, count, byTeam, falls[{victim,
 *   killers, killingTeamId, points, ultimate?}], streaks?, match?,
 *   freshStart?, largestLeadBefore?, runs?, rescues?, combos?, wiped?,
 *   phaseStart, phaseEnd, paintParts}; see heraldMoments for the fact
 *   fields). It gives one plate for the moments heraldMoments(event, filters)
 *   finds (see heraldRecord), placed as the file's Placement note says.
 *   A row with no moment (a plain single fall) gives []. With
 *   `filters.champion_streaks` true, the row's `streaks` (ChampionStreakFact
 *   list; absent or invalid entries are skipped) also count; with the switch
 *   false or absent they are ignored. With `filters.death_announcer` false
 *   the moments born from a fall are left out. When the row gives no card,
 *   previous (an earlier herald_card row) gives its card again with held
 *   true (no entry animation), in the spot it was first given when the
 *   bounds are the same. The row's exitAt (a finite number of ms from the
 *   tick's start, set by the painter's card queue) makes the card drawn,
 *   its own or the held one, fade out then. A row whose paintParts.effect is false gives [].
 * - `{part: "lineup", scene, bounds?, filters?, mapTitle, mapLabel?,
 *   pointsToWin?, tickLimit?, redZoneDoubles?, leftTeamId?}`: the lineup card.
 *   Medallions come only from scene.agents (the authorized arena scene: in
 *   Agent POV only the champions the team sees). leftTeamId (default 1) is the
 *   team drawn on the left.
 * - `{part: "verdict", product, verdict, bounds?, filters?, leftTeamId?,
 *   tucked?, avoid?, chrome?}`: product must be "replay_viewer" or
 *   "combat_debugger" (the BattleClient and DevClient at game over); any
 *   other value gives []. verdict is {scores: {1: number, 2: number},
 *   winnerTeamId: 1 | 2 | null, tick, basis: "threshold" | "horizon" |
 *   "tie", pointsToWin, teams?: {1?: {name, short?}, 2?: {name, short?}}
 *   (each team's controller line, shown under its team word: the full line
 *   name ("You + Recurrent MAPPO") when it fits the column at 15 px, else
 *   the short line ("You + 4 Policies") when given, else the full line at 13
 *   or 11 px, cut with "…" only when it still does not fit),
 *   honours?: [{label, value, leaders: [{classId, teamId}]}] (at most 3 are
 *   drawn; an older row with classId and teamId and no leaders names one
 *   leader)}. The caller passes the frame's own Honors (every champion in
 *   both views: they come only once the game is over). leftTeamId (1 or 2,
 *   default 1) is the team whose home end is on the left of the arena: its
 *   column is drawn on the left. tucked true (the page's Show Arena) or
 *   `filters.moment_full_cards` false gives its herald plate in the top band.
 * bounds is the arena's visible rectangle in CSS px (default a 960 x 600 band
 * at the origin); event.anchor ({x, y}: top centre) moves a herald's band
 * centre and top. avoid (optional) lists the champion circles {x, y, radius}
 * in CSS px that this audience may see: a card covers as few of them as it
 * can. In Agent POV pass only the POV scene's bodies. chrome (optional) lists
 * rectangles {left, top, right, bottom} in CSS px of page chrome over the
 * arena (such as the view tag): a card never covers one unless no spot is
 * left. Pure data apart from the per-row spot memory (see SPOT_BY_ROW);
 * returns a frozen array of frozen records.
 *
 * @param {{
 *   event?: JsonRecord | null,
 *   previous?: JsonRecord | null,
 *   part?: string,
 *   scene?: JsonRecord | null,
 *   bounds?: Rectangle | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   audience?: "researcher" | "agent_pov" | null,
 *   product?: string | null,
 *   avoid?: ReadonlyArray<{x: number, y: number, radius: number}> | null,
 *   chrome?: ReadonlyArray<{left: number, top: number, right: number, bottom: number}> | null,
 *   tucked?: boolean,
 *   [extra: string]: any,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const filters = input?.filters ?? {};
  if (filters.match_moments === false) {
    return Object.freeze([]);
  }
  const bounds = isRecord(input?.bounds)
    ? /** @type {Rectangle} */ (input.bounds)
    : DEFAULT_BOUNDS;
  const full = filters.moment_full_cards !== false;
  const avoid = asArray(input?.avoid).filter(isRecord);
  const chrome = asArray(input?.chrome).filter(isRecord);
  if (input?.part === "lineup") {
    return freezeOne(lineupRecord(input, bounds, full));
  }
  if (input?.part === "verdict") {
    return input.product === "replay_viewer" || input.product === "combat_debugger"
      ? freezeOne(
          verdictRecord(input, bounds, full && input.tucked !== true, avoid, chrome),
        )
      : Object.freeze([]);
  }
  // The painter passes avoid; only its spots are kept for the row's life.
  const keep = Array.isArray(input?.avoid);
  const exitAt = isRecord(input?.event) ? input.event.exitAt : null;
  const current = heraldRecord(
    input?.event,
    bounds,
    false,
    avoid,
    chrome,
    filters,
    keep,
    Number.isFinite(exitAt) ? Number(exitAt) : null,
  );
  if (current !== null) {
    return Object.freeze([current]);
  }
  return freezeOne(
    heraldRecord(
      input?.previous,
      bounds,
      true,
      avoid,
      chrome,
      filters,
      keep,
      Number.isFinite(exitAt) ? Number(exitAt) : null,
    ),
  );
}

/**
 * Return a frozen one-record array, or a frozen empty array for null.
 *
 * @param {LookRecord | null} record
 * @returns {ReadonlyArray<LookRecord>}
 */
function freezeOne(record) {
  return Object.freeze(record === null ? [] : [record]);
}

/**
 * Return "a" for team 1 and "b" for team 2 (anything else "a").
 *
 * @param {unknown} teamId
 * @returns {TeamKey}
 */
function teamKey(teamId) {
  return Number(teamId) === 2 ? "b" : "a";
}

/**
 * Return a medallion from an identity record (classId/class_id/accent and
 * teamId/team_id). fallen picks the Grave Stone look.
 *
 * @param {unknown} identity
 * @param {boolean} fallen
 * @returns {Medallion}
 */
function medallionOf(identity, fallen) {
  const value = isRecord(identity) ? identity : {};
  const classIdentity = classIdentityFromId(Number(value.classId ?? value.class_id));
  return Object.freeze({
    classKey:
      classIdentity?.tokenId ??
      (["mage", "warrior", "hunter", "rogue", "priest"].includes(value.accent)
        ? value.accent
        : null),
    teamKey: teamKey(value.teamId ?? value.team_id),
    fallen,
  });
}

/**
 * Return the herald slot: the middle half of the band at the top of bounds,
 * or a band of the same width centred on anchor.x (and starting at anchor.y)
 * when an anchor is given.
 *
 * @param {Rectangle} bounds
 * @param {unknown} anchor
 * @returns {{x0: number, x1: number, top: number}}
 */
function slotOf(bounds, anchor) {
  const half = bounds.width / 4;
  const centre =
    isRecord(anchor) && Number.isFinite(anchor.x)
      ? Number(anchor.x)
      : bounds.left + bounds.width / 2;
  const top =
    isRecord(anchor) && Number.isFinite(anchor.y)
      ? Number(anchor.y)
      : bounds.top + FOOTPRINT.topGap;
  return { x0: centre - half, x1: centre + half, top };
}

/**
 * Return the valid Champion Streak facts of a herald_card row (pure).
 *
 * event.streaks is read as a ChampionStreakFact list. An entry is kept only
 * when it is an object whose takedowns is a positive integer and whose
 * champion has a class ID 1-5 and a team ID 1 or 2; others are skipped. The
 * row's order is kept. Return a new array (empty when none is valid).
 *
 * @param {JsonRecord} event
 * @returns {ChampionStreakFact[]}
 */
function streakFacts(event) {
  return /** @type {ChampionStreakFact[]} */ (
    asArray(event.streaks).filter(
      (streak) =>
        isRecord(streak) &&
        Number.isInteger(streak.takedowns) &&
        streak.takedowns > 0 &&
        isRecord(streak.champion) &&
        classIdentityFromId(Number(streak.champion.classId)) !== null &&
        teamLabelFromId(Number(streak.champion.teamId)) !== null,
    )
  );
}

/**
 * The store's headline order (match-moments.js RANK): when several moments
 * share a tick, the lowest number heads the card and the next two join its
 * eyebrow. The Verdict (0) is drawn by the clients' renderer, not here.
 * Penta Takedown (0.5, owner 4 Oct 2026) heads the Team Wiped it causes.
 */
export const MOMENT_RANK = Object.freeze({
  penta: 0.5,
  wiped: 1,
  firstBlood: 2,
  matchPoint: 3,
  multiBig: 4,
  runBig: 5,
  streak: 5.5,
  multi: 6,
  rescue: 7,
  run: 8,
  ultTakedown: 9,
  trade: 10,
  level: 11,
  combo: 12,
  moonset: 13,
});

/**
 * Moments born from a fall (the store's FALL_KINDS). They are part of the
 * death announcement, so they show only while the Death Announcer switch
 * (`death_announcer`) is on.
 */
const FALL_KINDS = Object.freeze(
  new Set([
    "firstBlood",
    "multi",
    "multiBig",
    "penta",
    "wiped",
    "trade",
    "ultTakedown",
    "run",
    "runBig",
    "streak",
  ]),
);

/**
 * Ticks before the draw tick on which a Ticks Left card is born (store
 * moonset). Only 30 since the owner removed the 10, 5, 4, 3, 2 and 1 cards
 * (4 Oct 2026).
 */
const TICKS_LEFT_MARKS = Object.freeze([30]);

/** The lead a team must have held before the scores became level (Scores Level). */
const COMEBACK_LEAD = 3;

/** The moments one card names: its own and at most two more in its eyebrow. */
const MOMENTS_PER_CARD = 3;

/** The Priest's class ID, whose heals a Rescue names. */
const PRIEST_CLASS_ID = 5;

/**
 * Read a [Team A, Team B] score pair, or null when it is not two whole
 * numbers 0 or more. Pure.
 *
 * @param {unknown} value
 * @returns {readonly [number, number] | null}
 */
function scorePair(value) {
  const pair = asArray(value);
  return pair.length === 2 &&
    pair.every((score) => Number.isInteger(score) && score >= 0)
    ? /** @type {readonly [number, number]} */ ([Number(pair[0]), Number(pair[1])])
    : null;
}

/**
 * Return a team ID 1 or 2 from a value, or null for anything else. Pure.
 *
 * @param {unknown} value
 * @returns {1 | 2 | null}
 */
function teamIdOf(value) {
  const team = Number(value);
  return team === 1 || team === 2 ? team : null;
}

/**
 * Find the match moments a herald_card row names, in headline order (pure).
 *
 * event is a plan row of kind "herald_card"; anything else gives []. Its
 * fields (every one optional; a missing or invalid field adds no moment):
 * falls (the named falls: {victim, killers, killingTeamId, points (1, 2 or
 * null), ultimate ({classId, publicAgentId} or null)}), streaks
 * (ChampionStreakFact list), match ({before, after: [Team A, Team B] scores
 * at the start and end of the tick, pointsToWin, tick, tickLimit}: public
 * scoreboard facts), freshStart (true when the match began at tick 0 with
 * 0-0), largestLeadBefore (the largest lead at the start of this tick or
 * earlier), runs ([{teamId, takedowns}] reached this tick), rescues
 * ([{champion: identity, byUltimate}]), combos ([{casters: [{classId,
 * publicAgentId, teamId}], target: identity, fell}]) and wiped ([{teamId,
 * members: identity[]}]: teams whose every active champion is seen fallen
 * after the tick). An identity is {classId 1-5, teamId 1 or 2, title?,
 * publicAgentId?}.
 *
 * filters is the page's switch state (null or absent: defaults).
 * champion_streaks true adds streaks; death_announcer false leaves out every
 * moment born from a fall (FALL_KINDS). Return a frozen array of frozen
 * moments sorted by MOMENT_RANK (ties keep the order below), each
 * {kind, ...}:
 * - wiped {teamId (the scoring team), fallenTeamId, victims}; only when the
 *   fallen team also lost a champion this tick;
 * - firstBlood {teamId (the scoring team, null when both teams lost one),
 *   traded, victims, points (the fallen team's points when every fall's
 *   points are known, else null)}; only with freshStart, a named fall and
 *   a 0-0 start of the tick;
 * - matchPoint {teamIds, needs}: a team whose score rose to exactly 1 short
 *   of pointsToWin (needs is always 1);
 * - multi (2), multiBig (3 or 4) or penta (5) {teamId, victims, n, points}:
 *   one team lost two or more;
 * - run or runBig (5 or more) {teamId, takedowns};
 * - streak {champion, takedowns} (switch on, and only on a tick with a
 *   named fall);
 * - rescue {champion, byUltimate};
 * - ultTakedown {teamId, classKey (the caster's), ability, victim, points}:
 *   one per fall with an ultimate;
 * - trade {lost: {1, 2}, victims (Team A's first), dividerAfter};
 * - level {score}: scores level after the tick, not before it, and
 *   largestLeadBefore at least COMEBACK_LEAD;
 * - combo {casters [{classId, classKey}], abilities, target, fell}: two or
 *   more casters;
 * - moonset {left, pointsToWin}: tickLimit minus tick is in
 *   TICKS_LEFT_MARKS and no team has reached pointsToWin.
 *
 * Agent POV: every fact comes from the row, which the plan and the host
 * already limited to what the viewing team saw (and the public scoreboard),
 * so this finder never names an unseen champion.
 *
 * @param {unknown} event
 * @param {Readonly<Record<string, boolean>> | null} [filters] Default null.
 * @returns {ReadonlyArray<HeraldMoment>}
 */
export function heraldMoments(event, filters = null) {
  if (!isRecord(event) || event.kind !== "herald_card") return Object.freeze([]);
  const switches = isRecord(filters) ? filters : {};
  const falls = asArray(event.falls).filter((fall) => isRecord(fall?.victim));
  /** @param {JsonRecord} fall */
  const lostTeam = (fall) =>
    Number(fall.victim.teamId ?? fall.victim.team_id) === 2 ? 2 : 1;
  /** @type {Record<1 | 2, JsonRecord[]>} */
  const lostBy = { 1: [], 2: [] };
  for (const fall of falls) lostBy[lostTeam(fall)].push(fall);
  /** @param {JsonRecord[]} list */
  const pointsOf = (list) =>
    list.length > 0 && list.every((fall) => fall.points === 1 || fall.points === 2)
      ? list.reduce((sum, fall) => sum + Number(fall.points), 0)
      : null;
  const match = isRecord(event.match) ? event.match : null;
  const before = scorePair(match?.before);
  const after = scorePair(match?.after);
  const pointsToWin =
    Number.isInteger(match?.pointsToWin) && Number(match?.pointsToWin) > 0
      ? Number(match?.pointsToWin)
      : null;
  /** @type {Array<Record<string, any>>} */
  const found = [];
  for (const wiped of asArray(event.wiped)) {
    const fallenTeamId = teamIdOf(isRecord(wiped) ? wiped.teamId : null);
    const victims = asArray(isRecord(wiped) ? wiped.members : null).filter(isRecord);
    if (fallenTeamId === null || victims.length === 0) continue;
    if (lostBy[fallenTeamId].length === 0) continue;
    found.push({ kind: "wiped", teamId: 3 - fallenTeamId, fallenTeamId, victims });
  }
  if (
    event.freshStart === true &&
    falls.length > 0 &&
    before !== null &&
    before[0] === 0 &&
    before[1] === 0
  ) {
    const traded = lostBy[1].length > 0 && lostBy[2].length > 0;
    const teamId = traded ? null : lostBy[2].length > 0 ? 1 : 2;
    found.push({
      kind: "firstBlood",
      teamId,
      traded,
      victims: falls.map((fall) => fall.victim),
      points: teamId === null ? null : pointsOf(lostBy[teamId === 1 ? 2 : 1]),
    });
  }
  if (pointsToWin !== null && before !== null && after !== null) {
    const teamIds = /** @type {Array<1 | 2>} */ ([1, 2]).filter(
      (team) =>
        after[team - 1] > before[team - 1] && after[team - 1] === pointsToWin - 1,
    );
    if (teamIds.length > 0) {
      found.push({
        kind: "matchPoint",
        teamIds,
        needs: 1,
      });
    }
  }
  for (const lost of /** @type {Array<1 | 2>} */ ([1, 2])) {
    const list = lostBy[lost];
    if (list.length < 2) continue;
    found.push({
      kind: list.length >= 5 ? "penta" : list.length >= 3 ? "multiBig" : "multi",
      teamId: 3 - lost,
      victims: list.map((fall) => fall.victim),
      n: list.length,
      points: pointsOf(list),
    });
  }
  for (const run of asArray(event.runs)) {
    const teamId = teamIdOf(isRecord(run) ? run.teamId : null);
    const takedowns = isRecord(run) ? run.takedowns : null;
    if (teamId === null || !Number.isInteger(takedowns) || Number(takedowns) <= 0) {
      continue;
    }
    found.push({
      kind: Number(takedowns) >= 5 ? "runBig" : "run",
      teamId,
      takedowns: Number(takedowns),
    });
  }
  if (switches.champion_streaks === true && falls.length > 0) {
    for (const streak of streakFacts(event)) {
      found.push({
        kind: "streak",
        champion: streak.champion,
        takedowns: streak.takedowns,
      });
    }
  }
  for (const rescue of asArray(event.rescues)) {
    const champion = isRecord(rescue) ? rescue.champion : null;
    if (
      !isRecord(champion) ||
      classIdentityFromId(Number(champion.classId)) === null ||
      teamIdOf(champion.teamId) === null
    ) {
      continue;
    }
    found.push({ kind: "rescue", champion, byUltimate: rescue?.byUltimate === true });
  }
  for (const fall of falls) {
    const ultimate = isRecord(fall.ultimate) ? fall.ultimate : null;
    const ability =
      ultimate === null
        ? null
        : classAbilityNameV1(Number(ultimate.classId), "ultimate");
    if (ability === null) continue;
    found.push({
      kind: "ultTakedown",
      teamId: 3 - lostTeam(fall),
      classKey: classIdentityFromId(Number(ultimate?.classId))?.tokenId ?? null,
      ability,
      victim: fall.victim,
      points: pointsOf([fall]),
    });
  }
  if (lostBy[1].length > 0 && lostBy[2].length > 0) {
    found.push({
      kind: "trade",
      lost: { 1: lostBy[1].length, 2: lostBy[2].length },
      victims: [...lostBy[1], ...lostBy[2]].map((fall) => fall.victim),
      dividerAfter: lostBy[1].length,
    });
  }
  if (
    before !== null &&
    after !== null &&
    after[0] === after[1] &&
    before[0] !== before[1] &&
    Number(event.largestLeadBefore) >= COMEBACK_LEAD
  ) {
    found.push({ kind: "level", score: after[0] });
  }
  for (const combo of asArray(event.combos)) {
    if (!isRecord(combo)) continue;
    const casters = asArray(combo.casters)
      .filter(isRecord)
      .map((caster) => ({
        classId: Number(caster.classId),
        classKey: classIdentityFromId(Number(caster.classId))?.tokenId ?? null,
        ability: classAbilityNameV1(Number(caster.classId), "ultimate"),
      }))
      .filter((caster) => caster.ability !== null);
    if (casters.length < 2) continue;
    found.push({
      kind: "combo",
      casters: casters.map(({ classId, classKey }) => ({ classId, classKey })),
      abilities: casters.map((caster) => caster.ability),
      target: isRecord(combo.target) ? combo.target : null,
      fell: combo.fell === true,
    });
  }
  if (
    pointsToWin !== null &&
    after !== null &&
    Number.isInteger(match?.tick) &&
    Number.isInteger(match?.tickLimit)
  ) {
    const left = Number(match?.tickLimit) - Number(match?.tick);
    if (TICKS_LEFT_MARKS.includes(left) && Math.max(after[0], after[1]) < pointsToWin) {
      found.push({ kind: "moonset", left, pointsToWin });
    }
  }
  const announced = switches.death_announcer !== false;
  return Object.freeze(
    found
      .map((moment, order) => ({ moment, order }))
      .filter(({ moment }) => announced || !FALL_KINDS.has(moment.kind))
      .sort(
        (left, right) =>
          MOMENT_RANK[/** @type {keyof typeof MOMENT_RANK} */ (left.moment.kind)] -
            MOMENT_RANK[/** @type {keyof typeof MOMENT_RANK} */ (right.moment.kind)] ||
          left.order - right.order,
      )
      .map(({ moment }) => /** @type {HeraldMoment} */ (Object.freeze(moment))),
  );
}

/**
 * Return whether a moment enters at the tick's settled time rather than at
 * the Fall beat (the store's SETTLE_KINDS: Rescue and Ticks Left, and an
 * Ultimate Combo whose target did not fall). The plan times a card from the
 * earliest moment it names. Pure.
 *
 * @param {HeraldMoment} moment
 * @returns {boolean}
 */
export function momentEntersAtSettle(moment) {
  return (
    moment.kind === "rescue" ||
    moment.kind === "moonset" ||
    (moment.kind === "combo" && moment.fell !== true)
  );
}

/**
 * Return the moments one card names: the headline and at most two more, in
 * headline order (heraldMoments' first MOMENTS_PER_CARD). Pure.
 *
 * @param {unknown} event
 * @param {Readonly<Record<string, boolean>> | null} [filters] Default null.
 * @returns {ReadonlyArray<HeraldMoment>}
 */
export function cardMoments(event, filters = null) {
  return Object.freeze(heraldMoments(event, filters).slice(0, MOMENTS_PER_CARD));
}

/**
 * Upper bounds of the nodes one glyph draws on a plate (see BUDGET): a fallen
 * medallion 5 (its ring path and a sigil of at most 4 nodes), a living one 6
 * (plus a Team B ink ring), a Rescue medallion 8 (a living one, its gold ring
 * and "+"), the Ticks Left crescent 2 and a diamond, crest or divider 1.
 */
const GLYPH_NODES_MAX = Object.freeze({
  fallen: 5,
  living: 6,
  rescue: 8,
  crescent: 2,
  path: 1,
});

/**
 * Return the paint allowance of one herald_card row's own card, for the
 * plan's budget (pure).
 *
 * event is the row and filters the page's switch state (see heraldMoments).
 * When the row names moments, the allowance is an upper bound of the card it
 * draws: a plate's 10 nodes of its own, 2 for a tier-1 Herald Rule and
 * GLYPH_NODES_MAX per glyph (at least 8, a one-line band's most), plus 1 for
 * the card's own group, which the painter counts as a kept node too (without
 * it a card that used its whole allowance failed the tick: 13 > 12 on Match
 * Point near the end of a match, 3 Oct 2026), with 3 animations for a tier-1
 * plate and 2 for another (a band or chip uses 1);
 * the kept allowance equals the node allowance (a paused card keeps every
 * node). A row that names no moment may still draw the previous tick's held
 * card, so it gets the whole BUDGET. Return {nodesPerRecord,
 * animationsPerRecord, persistentNodesPerRecord}, frozen. Busy ticks then
 * keep a small card (a run's card is only its crest and words) instead of
 * dropping it for the largest card's allowance.
 *
 * @param {unknown} event
 * @param {Readonly<Record<string, boolean>> | null} [filters] Default null.
 * @returns {Readonly<{nodesPerRecord: number, animationsPerRecord: number, persistentNodesPerRecord: number}>}
 */
export function rowBudget(event, filters = null) {
  const moments = cardMoments(event, filters);
  if (moments.length === 0) return BUDGET;
  const spec = cardSpec(moments);
  const medallions = spec.medallions.slice(0, PLATE_MEDALLIONS_MAX);
  const glyphs =
    medallions.reduce(
      (sum, medallion) =>
        sum + (medallion.fallen ? GLYPH_NODES_MAX.fallen : GLYPH_NODES_MAX.living),
      0,
    ) +
    (spec.dividerAfter === null ? 0 : GLYPH_NODES_MAX.path) +
    (spec.diamondClassKey ? GLYPH_NODES_MAX.path : 0) +
    spec.crests.length * GLYPH_NODES_MAX.path +
    (spec.rescue === null ? 0 : GLYPH_NODES_MAX.rescue) +
    (spec.crescent === null ? 0 : GLYPH_NODES_MAX.crescent) +
    spec.diamonds.length * GLYPH_NODES_MAX.path;
  // The card's own group counts as one kept node in the painter.
  const nodes = Math.min(
    BUDGET.nodesPerRecord,
    Math.max(8, 10 + (spec.tier1 ? 2 : 0) + glyphs) + 1,
  );
  return Object.freeze({
    nodesPerRecord: nodes,
    animationsPerRecord: spec.tier1 ? 3 : 2,
    persistentNodesPerRecord: nodes,
  });
}

/**
 * Return "Team A" or "Team B" for a team ID (anything else ""). Pure.
 *
 * @param {unknown} teamId
 * @returns {string}
 */
function teamWords(teamId) {
  return teamLabelFromId(Number(teamId)) ?? "";
}

/**
 * Return " +N" for known points, or "" for null. Pure.
 *
 * @param {number | null} points
 * @returns {string}
 */
function plusWords(points) {
  return points === null ? "" : ` +${points}`;
}

/**
 * Return "N Ticks Left" ("1 Tick Left"). Pure.
 *
 * @param {number} left
 * @returns {string}
 */
function ticksLeftWords(left) {
  return `${left} ${left === 1 ? "Tick" : "Ticks"} Left`;
}

/**
 * Return a moment's short words when it joins another card's eyebrow (the
 * store's chipWords): "◆ Charge Takedown", "Trade", "5 Unanswered", "Double
 * Takedown", "Penta Takedown", "Scores Level", "◆◆ Ultimate Combo", "30
 * Ticks Left", "Match Point", "Rescue", "Team A Mage 5 Takedowns", "Team B
 * Wiped" or "First Blood". Pure.
 *
 * @param {HeraldMoment} moment
 * @returns {string}
 */
function chipWords(moment) {
  switch (moment.kind) {
    case "ultTakedown":
      return `◆ ${moment.ability} Takedown`;
    case "trade":
      return "Trade";
    case "run":
    case "runBig":
      return `${moment.takedowns} Unanswered`;
    case "multi":
    case "multiBig":
    case "penta":
      return `${MULTI_NAMES[moment.n] || "Multi"} Takedown`;
    case "level":
      return "Scores Level";
    case "combo":
      return `${"◆".repeat(moment.casters.length)} Ultimate Combo`;
    case "moonset":
      return ticksLeftWords(moment.left);
    case "matchPoint":
      return "Match Point";
    case "rescue":
      return "Rescue";
    case "streak":
      return `${teamWords(moment.champion.teamId)} ${classIdentityFromId(Number(moment.champion.classId))?.label ?? "Champion"} ${moment.takedowns} Takedowns`.trim();
    case "wiped":
      return `${teamWords(moment.fallenTeamId)} Wiped`.trim();
    case "firstBlood":
      return "First Blood";
    default:
      return "";
  }
}

/**
 * @typedef {Readonly<{
 *   kinds: ReadonlyArray<string>,
 *   title: string,
 *   eyebrow: string,
 *   ruleTeam: TeamKey | null,
 *   tier1: boolean,
 *   diamonds: ReadonlyArray<string | null>,
 *   medallions: ReadonlyArray<Medallion>,
 *   medallionStep: number,
 *   dividerAfter: number | null,
 *   diamondClassKey: string | null,
 *   crests: ReadonlyArray<Readonly<{teamKey: TeamKey, size: number}>>,
 *   rescue: Medallion | null,
 *   crescent: Readonly<{left: number}> | null,
 *   streak: Readonly<{title: string, takedowns: number}> | null,
 * }>} HeraldCardSpec
 *   What one card draws (cardSpec): the moment kinds it names in order, its
 *   title and eyebrow (chips joined with " · "), its rule team (null for a
 *   neutral Moon Silver rule), whether it is a tier-1 moment (Herald Rule),
 *   and its glyphs: diamonds (an Ultimate Combo's caster class keys, one rune
 *   diamond each), medallions (stepping medallionStep px, a divider before
 *   index dividerAfter), the Ultimate rune diamond, crests ({teamKey, size}),
 *   the Rescue medallion and the Ticks Left crescent. streak names a
 *   heading streak's champion for the tooltip.
 */

/**
 * Build what one card draws from its moments (the store's heraldSpec). Pure.
 *
 * moments are cardMoments' result (non-empty); the first heads the card and
 * the rest join its eyebrow as chipWords. Return the HeraldCardSpec.
 *
 * @param {ReadonlyArray<HeraldMoment>} moments
 * @returns {HeraldCardSpec}
 */
function cardSpec(moments) {
  const [head, ...chips] = moments;
  const tail = chips.map((chip) => ` · ${chipWords(chip)}`).join("");
  const base = {
    kinds: Object.freeze(moments.map((moment) => moment.kind)),
    ruleTeam: /** @type {TeamKey | null} */ (null),
    tier1: false,
    diamonds: /** @type {ReadonlyArray<string | null>} */ ([]),
    medallions: /** @type {ReadonlyArray<Medallion>} */ ([]),
    medallionStep: FOOTPRINT.plate.medallionStep,
    dividerAfter: /** @type {number | null} */ (null),
    diamondClassKey: /** @type {string | null} */ (null),
    crests: /** @type {ReadonlyArray<{teamKey: TeamKey, size: number}>} */ ([]),
    rescue: /** @type {Medallion | null} */ (null),
    crescent: /** @type {{left: number} | null} */ (null),
    streak: /** @type {{title: string, takedowns: number} | null} */ (null),
  };
  /** @param {ReadonlyArray<JsonRecord>} victims */
  const fallen = (victims) => victims.map((victim) => medallionOf(victim, true));
  switch (head.kind) {
    case "wiped":
      return Object.freeze({
        ...base,
        title: "Team Wiped",
        eyebrow: `${teamWords(head.fallenTeamId)}${tail}`,
        ruleTeam: teamKey(head.teamId),
        tier1: true,
        medallions: fallen(head.victims),
        medallionStep: FOOTPRINT.plate.wipedStep,
      });
    case "firstBlood":
      return Object.freeze({
        ...base,
        title: "First Blood",
        eyebrow: `${head.traded ? "Traded" : `${teamWords(head.teamId)}${plusWords(head.points)}`}${tail}`,
        ruleTeam: head.teamId === null ? null : teamKey(head.teamId),
        tier1: true,
        medallions: fallen(head.victims),
      });
    case "matchPoint":
      return Object.freeze({
        ...base,
        title: "Match Point",
        eyebrow: `${head.teamIds.length > 1 ? "Both Teams Need" : `${teamWords(head.teamIds[0])} Needs`} ${head.needs} ${head.needs === 1 ? "Point" : "Points"}${tail}`,
        tier1: true,
        crests: head.teamIds.map((/** @type {number} */ team) => ({
          teamKey: teamKey(team),
          size: 22,
        })),
      });
    case "penta":
      return Object.freeze({
        ...base,
        title: `${MULTI_NAMES[head.n] || "Multi"} Takedown`,
        eyebrow: `${teamWords(head.teamId)}${plusWords(head.points)}${tail}`,
        ruleTeam: teamKey(head.teamId),
        tier1: true,
        medallions: fallen(head.victims),
        medallionStep: FOOTPRINT.plate.wipedStep,
      });
    case "multi":
    case "multiBig":
      return Object.freeze({
        ...base,
        title: `${MULTI_NAMES[head.n] || "Multi"} Takedown`,
        eyebrow: `${teamWords(head.teamId)}${plusWords(head.points)}${tail}`,
        ruleTeam: teamKey(head.teamId),
        medallions: fallen(head.victims),
      });
    case "run":
    case "runBig":
      return Object.freeze({
        ...base,
        title: `${head.takedowns} Unanswered`,
        eyebrow: `${teamWords(head.teamId)}${tail}`,
        ruleTeam: teamKey(head.teamId),
        crests: [{ teamKey: teamKey(head.teamId), size: 18 }],
      });
    case "streak": {
      const champion = head.champion;
      const classLabel =
        classIdentityFromId(Number(champion.classId))?.label ?? "Champion";
      const teamLabel = teamWords(champion.teamId);
      return Object.freeze({
        ...base,
        title: `${head.takedowns} Takedowns Without Falling`,
        eyebrow: `${`${teamLabel} ${classLabel}`.trim()}${tail}`,
        ruleTeam: teamKey(champion.teamId),
        medallions: [medallionOf(champion, false)],
        streak: {
          title: String(champion.title ?? `${teamLabel} ${classLabel}`.trim()),
          takedowns: head.takedowns,
        },
      });
    }
    case "rescue":
      return Object.freeze({
        ...base,
        title: "Rescue",
        eyebrow: `${classAbilityNameV1(PRIEST_CLASS_ID, head.byUltimate ? "ultimate" : "basic") ?? "Priest Heal"}${tail}`,
        ruleTeam: teamKey(head.champion.teamId),
        rescue: medallionOf(head.champion, false),
      });
    case "ultTakedown":
      return Object.freeze({
        ...base,
        title: "Ultimate Takedown",
        eyebrow: `${head.ability} · ${teamWords(head.teamId)}${plusWords(head.points)}${tail}`,
        ruleTeam: teamKey(head.teamId),
        medallions: [medallionOf(head.victim, true)],
        diamondClassKey: head.classKey,
      });
    case "trade":
      return Object.freeze({
        ...base,
        title: "Trade",
        eyebrow: `${teamWords(1)} Lost ${head.lost[1]} · ${teamWords(2)} Lost ${head.lost[2]}${tail}`,
        medallions: fallen(head.victims),
        dividerAfter: head.dividerAfter,
      });
    case "level":
      return Object.freeze({
        ...base,
        title: "Scores Level",
        eyebrow: `${head.score} – ${head.score}${tail}`,
        crests: [
          { teamKey: teamKey(1), size: 18 },
          { teamKey: teamKey(2), size: 18 },
        ],
      });
    case "combo":
      return Object.freeze({
        ...base,
        title: "Ultimate Combo",
        eyebrow: `${head.abilities.join(" + ")}${tail}`,
        diamonds: head.casters.map(
          (/** @type {{classKey: string | null}} */ caster) => caster.classKey,
        ),
      });
    default:
      return Object.freeze({
        ...base,
        title: ticksLeftWords(head.left),
        eyebrow: `Draw If No Team Reaches ${head.pointsToWin}${tail}`,
        crescent: { left: head.left },
      });
  }
}

/**
 * Return a key of an arena rectangle (left, top, width, height, rounded to
 * 0.1 px) for SPOT_BY_ROW. Pure.
 *
 * @param {Rectangle} bounds
 * @returns {string}
 */
function boundsKeyOf(bounds) {
  return [bounds.left, bounds.top, bounds.width, bounds.height]
    .map((value) => Math.round(Number(value) * 10))
    .join(",");
}

/**
 * @typedef {Readonly<{
 *   type: "diamond" | "crest" | "medallion" | "divider" | "rescue" | "crescent",
 *   x: number,
 *   [field: string]: any,
 * }>} PlacedGlyph
 *   One glyph of a herald plate at its left edge x in unscaled CSS px from the
 *   plate's left (see glyphLayout for each type's fields).
 */

/**
 * Lay out a plate's glyphs left to right from its left padding (pure).
 *
 * spec is the card's HeraldCardSpec; medallions is spec.medallions cut to
 * PLATE_MEDALLIONS_MAX and dividerAfter its divider index (or null). Order
 * and sizes (unscaled CSS px): an Ultimate Combo's rune diamonds and the
 * Ultimate Takedown's rune diamond (11 px each, then a 6 px gap),
 * crests (spec size each, 4 px apart), the Rescue medallion (24 px inside a
 * 30 px gold ring), the Ticks Left crescent (22 px), the medallions (26 px,
 * one every spec.medallionStep px, so they overlap; a 1 px divider takes 15
 * px before index dividerAfter). Return {items, width}: the PlacedGlyph list
 * ("diamond" {classKey}, "crest" {teamKey, size}, "rescue" {medallion},
 * "crescent" {left}, "divider", "medallion" {medallion}) and the glyph
 * block's width (the last glyph's right edge minus the padding).
 *
 * @param {HeraldCardSpec} spec
 * @param {ReadonlyArray<Medallion>} medallions
 * @param {number | null} dividerAfter
 * @returns {{items: ReadonlyArray<PlacedGlyph>, width: number}}
 */
function glyphLayout(spec, medallions, dividerAfter) {
  const size = FOOTPRINT.plate;
  /** @type {PlacedGlyph[]} */
  const items = [];
  /** @type {number} */
  let x = size.padLeft;
  let end = x;
  for (const classKey of spec.diamonds) {
    items.push({ type: "diamond", x, classKey });
    end = x + 11;
    x += 17;
  }
  if (spec.diamondClassKey) {
    items.push({ type: "diamond", x, classKey: spec.diamondClassKey });
    end = x + 11;
    x += 17;
  }
  spec.crests.forEach((crest, index) => {
    if (index > 0) x += 4;
    items.push({ type: "crest", x, teamKey: crest.teamKey, size: crest.size });
    x += crest.size;
    end = x;
  });
  if (spec.rescue !== null) {
    items.push({ type: "rescue", x, medallion: spec.rescue });
    x += 30;
    end = x;
  }
  if (spec.crescent !== null) {
    items.push({ type: "crescent", x, left: spec.crescent.left });
    x += 22;
    end = x;
  }
  medallions.forEach((medallion, index) => {
    if (dividerAfter !== null && index === dividerAfter) {
      items.push({ type: "divider", x });
      x += 15;
    }
    items.push({ type: "medallion", x, medallion });
    end = x + size.medallion;
    x += spec.medallionStep;
  });
  return {
    items: Object.freeze(items.map((item) => Object.freeze(item))),
    width: end - size.padLeft,
  };
}

/**
 * Build the herald card record for one herald_card row, or null.
 *
 * event is the row (or null); bounds the arena rectangle; held true when the
 * row is drawn again after its own tick (no entry animation). avoid lists
 * champion circles {x, y, radius} and chrome page-chrome rectangles {left,
 * top, right, bottom}, all in CSS px (see placeOnBand). filters is the page's
 * switch state (see heraldMoments). keep true stores the spot for the row's
 * life (SPOT_BY_ROW); a row already placed in the same bounds keeps its spot
 * whatever avoid says. exitAt is when the card fades out, in ms from the
 * tick's start, or null to stay the whole tick (the painter's card queue sets
 * it on a held card, or on a card that enters and leaves in one tick).
 *
 * The card names cardMoments(event, filters) (null when there are none). It
 * is always a "herald-plate" record (owner, 4 Oct 2026: one plate style)
 * with title, eyebrow, ruleTeam, medallions (at most PLATE_MEDALLIONS_MAX),
 * dividerAfter, diamondClassKey, glyphs (glyphLayout's PlacedGlyph list) and
 * glyphWidth; a tier-1 plate (Penta Takedown, Team Wiped, First Blood, Match
 * Point) also carries tierRule {x0, x1, y}: the Herald Rule across the band
 * (4 px inside each side of bounds) at the plate's vertical centre, in the
 * card's own unscaled coordinates. Every record carries kinds (the moment
 * kinds it names, in order), fallen (every named fall's title, for the
 * tooltip), streak ({title, takedowns}, only when a streak heads it),
 * phaseStart (the row's), exitAt (a finite number or null), place "top" and
 * its drawn box (x, y, scale, width, height). Pure apart from the spot memory.
 *
 * @param {unknown} event
 * @param {Rectangle} bounds
 * @param {boolean} held
 * @param {ReadonlyArray<JsonRecord>} avoid
 * @param {ReadonlyArray<JsonRecord>} chrome
 * @param {Readonly<Record<string, boolean>>} filters
 * @param {boolean} keep
 * @param {number | null} [exitAt] default null.
 * @returns {LookRecord | null}
 */
function heraldRecord(
  event,
  bounds,
  held,
  avoid,
  chrome,
  filters,
  keep,
  exitAt = null,
) {
  if (!isRecord(event) || event.kind !== "herald_card") return null;
  if (isRecord(event.paintParts) && event.paintParts.effect === false) return null;
  const moments = cardMoments(event, filters);
  if (moments.length === 0) return null;
  const spec = cardSpec(moments);
  const size = FOOTPRINT.plate;
  const medallions = spec.medallions.slice(0, PLATE_MEDALLIONS_MAX);
  const dividerAfter =
    spec.dividerAfter === null || spec.dividerAfter >= medallions.length
      ? null
      : spec.dividerAfter;
  const layout = glyphLayout(spec, medallions, dividerAfter);
  const textBlock = Math.max(
    textWidth(spec.eyebrow, 11, true),
    textWidth(spec.title, 18),
  );
  const width = size.padLeft + layout.width + size.gap + textBlock + size.padRight;
  /** @type {BandForm} */
  const form = {
    key: "plate",
    width: width + 2 * size.tailReach,
    height: size.height,
    minScale: Math.max(size.minScale, MIN_TEXT_PX / 11),
  };
  const boundsKey = boundsKeyOf(bounds);
  const kept = SPOT_BY_ROW.get(/** @type {object} */ (event));
  let spot = kept !== undefined && kept.boundsKey === boundsKey ? kept.spot : null;
  if (spot === null) {
    spot = placeOnBand(bounds, slotOf(bounds, event.anchor), form, avoid, chrome);
    if (keep) SPOT_BY_ROW.set(/** @type {object} */ (event), { boundsKey, spot });
  }
  const falls = asArray(event.falls).filter((fall) => isRecord(fall?.victim));
  const x = spot.left + size.tailReach * spot.scale;
  return Object.freeze({
    kind: "herald-plate",
    eventId: String(event.eventId ?? "herald"),
    event,
    held,
    phaseStart: Number(event.phaseStart ?? 0),
    exitAt: Number.isFinite(exitAt) ? Number(exitAt) : null,
    title: spec.title,
    eyebrow: spec.eyebrow,
    ruleTeam: spec.ruleTeam,
    kinds: spec.kinds,
    fallen: Object.freeze(
      falls.map((fall) => String(fall.victim.title ?? "A champion")),
    ),
    ...(spec.streak === null ? {} : { streak: Object.freeze({ ...spec.streak }) }),
    place: "top",
    medallions: Object.freeze(medallions),
    dividerAfter,
    diamondClassKey: spec.diamondClassKey,
    glyphs: layout.items,
    glyphWidth: layout.width,
    ...(spec.tier1
      ? {
          tierRule: Object.freeze({
            x0: (bounds.left + 4 - x) / spot.scale,
            x1: (bounds.right - 4 - x) / spot.scale,
            y: size.height / 2,
          }),
        }
      : {}),
    width,
    height: size.height,
    scale: spot.scale,
    x,
    y: spot.top,
  });
}

/**
 * Choose the spot for a card in the arena's top band: Sliding Top Placement
 * (owner, 4 Oct 2026). Pure.
 *
 * bounds is the arena rectangle; slot comes from slotOf (its top is the
 * card's top, so the card hugs the top edge, and the middle of x0 and x1 is
 * the band's centre); form the card's BandForm; avoid the champion circles
 * {x, y, radius} and chrome the page-chrome rectangles {left, top, right,
 * bottom}, all in CSS px. The card slides along the whole band (4 px inside
 * each side): at full size first, and only when every full-size spot covers
 * a champion at smaller sizes from BAND_SCALES, never below form.minScale.
 * Spots compare by how many champions they touch, then how much champion
 * area they cover, then the larger size, then the distance from the centre
 * (see spotFor for the spots tried and the chrome rule). When the card is
 * wider than the band even at its smallest readable size, it is centred at
 * that size, so a card never vanishes. Return the BandSpot.
 *
 * @param {Rectangle} bounds
 * @param {{x0: number, x1: number, top: number}} slot
 * @param {BandForm} form
 * @param {ReadonlyArray<JsonRecord>} avoid
 * @param {ReadonlyArray<JsonRecord>} chrome
 * @returns {BandSpot}
 */
function placeOnBand(bounds, slot, form, avoid, chrome) {
  const span = /** @type {const} */ ([bounds.left + 4, bounds.right - 4]);
  const centre = (slot.x0 + slot.x1) / 2;
  /** @type {BandSpot | null} */
  let best = null;
  for (const scale of BAND_SCALES) {
    if (scale < form.minScale - 1e-6) break;
    const spot = spotFor(form, scale, span, slot.top, centre, avoid, chrome);
    if (spot !== null && (best === null || spotBefore(spot, best))) best = spot;
    if (best !== null && best.covered === 0) break;
  }
  if (best !== null) return best;
  const width = form.width * form.minScale;
  const height = form.height * form.minScale;
  const left = centre - width / 2;
  const cover = coverOf(
    [left - 2, slot.top - 2, left + width + 2, slot.top + height + 2],
    avoid,
  );
  return Object.freeze({
    form: form.key,
    scale: form.minScale,
    left,
    top: slot.top,
    width,
    height,
    covered: cover.covered,
    hit: cover.hit,
    distance: 0,
  });
}

/**
 * Return whether spot a goes before spot b: fewer champions touched, then
 * less champion area covered (to 1 px squared), then a larger scale, then
 * nearer the band's centre. Pure.
 *
 * @param {BandSpot} a
 * @param {BandSpot} b
 * @returns {boolean}
 */
function spotBefore(a, b) {
  if (a.covered !== b.covered) return a.covered < b.covered;
  if (Math.round(a.hit) !== Math.round(b.hit)) return a.hit < b.hit;
  if (a.scale !== b.scale) return a.scale > b.scale;
  return a.distance < b.distance;
}

/**
 * Find the best spot for a card of one scale along the band (pure).
 *
 * form is the BandForm and scale its size factor; range [lo, hi] the band's
 * x range in CSS px; top the card's top; centre the band's centre x; avoid
 * and chrome as in placeOnBand. Page chrome that reaches the card's height
 * cuts the range into free stretches; the card tries, in every stretch wide
 * enough for it, the left edges from the stretch's start to its end in steps
 * of 4 px, plus the spot centred on the band (moved inside the stretch) and
 * both ends. Only when no stretch is wide enough is chrome ignored, so the
 * card never covers chrome while a clear spot exists. Return the spot that
 * goes first by spotBefore, or null when the card is wider than the range.
 *
 * @param {BandForm} form
 * @param {number} scale
 * @param {readonly [number, number]} range
 * @param {number} top
 * @param {number} centre
 * @param {ReadonlyArray<JsonRecord>} avoid
 * @param {ReadonlyArray<JsonRecord>} chrome
 * @returns {BandSpot | null}
 */
function spotFor(form, scale, range, top, centre, avoid, chrome) {
  const [lo, hi] = range;
  const width = form.width * scale;
  const height = form.height * scale;
  if (width > hi - lo) return null;
  const cuts = chrome
    .filter((rect) => Number(rect.top) < top + height && Number(rect.bottom) > top)
    .map((rect) => [Number(rect.left), Number(rect.right)])
    .sort((left, right) => left[0] - right[0]);
  /** @type {Array<[number, number]>} */
  const free = [];
  let x = lo;
  for (const [cutLeft, cutRight] of cuts) {
    if (cutLeft > x) free.push([x, Math.min(cutLeft, hi)]);
    x = Math.max(x, cutRight);
  }
  if (hi > x) free.push([x, hi]);
  const stretches = free.filter(([a, z]) => z - a >= width);
  /** @type {BandSpot | null} */
  let best = null;
  for (const [a, z] of stretches.length > 0 ? stretches : [[lo, hi]]) {
    const lefts = [Math.min(Math.max(centre - width / 2, a), z - width), a, z - width];
    for (let left = a; left <= z - width; left += 4) lefts.push(left);
    for (const left of lefts) {
      const cover = coverOf(
        [left - 2, top - 2, left + width + 2, top + height + 2],
        avoid,
      );
      /** @type {BandSpot} */
      const spot = {
        form: form.key,
        scale,
        left,
        top,
        width,
        height,
        covered: cover.covered,
        hit: cover.hit,
        distance: Math.abs(left + width / 2 - centre),
      };
      if (best === null || spotBefore(spot, best)) best = spot;
    }
  }
  return best === null ? null : Object.freeze(best);
}

/**
 * Return how many champions a card box touches and how much of their area it
 * covers (pure): box is [left, top, right, bottom]; each avoid circle counts
 * as its square (centre +- radius + 3 px). covered is the number of squares
 * the box overlaps and hit the summed overlap in px squared. Circles without
 * finite numbers are skipped.
 *
 * @param {readonly [number, number, number, number]} box
 * @param {ReadonlyArray<JsonRecord>} avoid
 * @returns {{covered: number, hit: number}}
 */
function coverOf(box, avoid) {
  let covered = 0;
  let hit = 0;
  for (const circle of avoid) {
    const cx = Number(circle.x);
    const cy = Number(circle.y);
    const r = Number(circle.radius ?? 0) + 3;
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(r)) continue;
    const w = Math.min(box[2], cx + r) - Math.max(box[0], cx - r);
    const h = Math.min(box[3], cy + r) - Math.max(box[1], cy - r);
    if (w > 0 && h > 0) {
      covered += 1;
      hit += w * h;
    }
  }
  return { covered, hit };
}

/**
 * Build a one-line band record (22 px) centred in slot (pure).
 *
 * kind names the record; common holds shared fields; text is the line;
 * crests lists team crests drawn before the text. The band is scaled down to
 * fit the slot (no lower limit; its words stay whole).
 *
 * @param {string} kind
 * @param {Record<string, any>} common
 * @param {string} text
 * @param {{x0: number, x1: number, top: number}} slot
 * @param {ReadonlyArray<TeamKey>} crests
 * @returns {LookRecord}
 */
function bandRecord(kind, common, text, slot, crests) {
  const band = FOOTPRINT.band;
  const width =
    2 * band.pad + crests.length * (band.crest + band.gap) + textWidth(text, 13);
  const slotWidth = slot.x1 - slot.x0;
  const scale = Math.min(1, slotWidth / (width + 14));
  const centre = (slot.x0 + slot.x1) / 2;
  return Object.freeze({
    kind,
    ...common,
    text,
    crests: Object.freeze([...crests]),
    width,
    height: band.height,
    scale,
    x: centre - (width * scale) / 2,
    y: slot.top,
  });
}

/**
 * Build the lineup record (full card, or band when full cards are off).
 *
 * See records for the input. The full card is 372 px wide (scaled to fit the
 * arena width minus 16 px). Pure.
 *
 * @param {Record<string, any>} input
 * @param {Rectangle} bounds
 * @param {boolean} full
 * @returns {LookRecord | null}
 */
function lineupRecord(input, bounds, full) {
  const mapTitle = String(input.mapTitle ?? "").trim();
  if (mapTitle.length === 0) return null;
  const leftTeamId = Number(input.leftTeamId) === 2 ? 2 : 1;
  const rightTeamId = leftTeamId === 1 ? 2 : 1;
  const agents = asArray(isRecord(input.scene) ? input.scene.agents : null).filter(
    isRecord,
  );
  /**
   * Medallions of one team's champions in the given scene, in public ID order.
   *
   * @param {number} teamId
   */
  const teamMedallions = (teamId) =>
    Object.freeze(
      agents
        .filter((agent) => Number(agent.team_id) === teamId)
        .sort((left, right) =>
          String(left.public_agent_id).localeCompare(
            String(right.public_agent_id),
            "en",
            {
              numeric: true,
            },
          ),
        )
        .map((agent) =>
          medallionOf(agent, agent.alive === false || agent.life_state === "corpse"),
        ),
    );
  const rules = [];
  if (Number.isInteger(input.pointsToWin)) {
    rules.push(
      `First to ${input.pointsToWin} Wins${Number.isInteger(input.tickLimit) ? ` · Draw at Tick ${input.tickLimit}` : ""}`,
    );
  }
  if (input.redZoneDoubles === true) {
    rules.push("A Death in Its Own Red Zone Scores 2");
  }
  const common = {
    eventId: "herald:lineup",
    event: null,
    held: false,
    phaseStart: 0,
    title: mapTitle,
    eyebrow: String(input.mapLabel ?? "Team Deathmatch"),
    ruleTeam: null,
    fallen: Object.freeze([]),
  };
  const slot = slotOf(bounds, null);
  if (!full) {
    return bandRecord(
      "herald-band",
      common,
      mapTitle,
      { x0: bounds.left + 8, x1: bounds.right - 8, top: slot.top },
      [teamKey(leftTeamId), teamKey(rightTeamId)],
    );
  }
  const card = FOOTPRINT.fullCard;
  const height =
    10 + 13 + 2 + 27 + 17 + 13 + 6 + card.medallion + 9 + 17 * rules.length + 10;
  const scale = Math.min(1, (bounds.width - 16) / card.width);
  return Object.freeze({
    kind: "herald-lineup",
    ...common,
    rules: Object.freeze(rules),
    columns: Object.freeze([
      Object.freeze({
        teamKey: teamKey(leftTeamId),
        teamId: leftTeamId,
        medallions: teamMedallions(leftTeamId),
      }),
      Object.freeze({
        teamKey: teamKey(rightTeamId),
        teamId: rightTeamId,
        medallions: teamMedallions(rightTeamId),
      }),
    ]),
    width: card.width,
    height,
    scale,
    x: bounds.left + bounds.width / 2 - (card.width * scale) / 2,
    y: slot.top,
  });
}

/**
 * Return the verdict's words: {eyebrow, results {1, 2}, lead (or null)}.
 *
 * verdict is the caller's final facts (see records). Pure.
 *
 * @param {JsonRecord} verdict
 */
function verdictWords(verdict) {
  const target = Number(verdict.pointsToWin);
  const tick = Number(verdict.tick);
  const scores = isRecord(verdict.scores) ? verdict.scores : {};
  const winner =
    Number(verdict.winnerTeamId) === 2
      ? 2
      : Number(verdict.winnerTeamId) === 1
        ? 1
        : null;
  if (verdict.basis === "horizon" || winner === null) {
    const tied = verdict.basis === "tie";
    return {
      eyebrow: tied
        ? `Tick ${tick} · Both Reached ${target} · Tied`
        : `Tick ${tick} · No Team Reached ${target}`,
      results: { 1: "Draw", 2: "Draw" },
      lead:
        !tied && Number(scores[1]) !== Number(scores[2])
          ? `A Lead at Tick ${tick} Wins Nothing`
          : null,
    };
  }
  const both = Number(scores[1]) >= target && Number(scores[2]) >= target;
  return {
    eyebrow: both
      ? `Tick ${tick} · Both Reached ${target} · ${teamLabelFromId(winner)} Scored More`
      : `Tick ${tick} · ${teamLabelFromId(winner)} Reached ${target}`,
    results: {
      1: winner === 1 ? "Victory" : "Defeat",
      2: winner === 2 ? "Victory" : "Defeat",
    },
    lead: null,
  };
}

/**
 * Cut words to fit a width at a bold font size, ending with "…" (pure).
 *
 * text is the words; size the font size in px; width the room in px (the
 * textWidth estimate). Return text unchanged when it fits, else its longest
 * start that fits with "…" added ("" for no room).
 *
 * @param {string} text
 * @param {number} size
 * @param {number} width
 * @returns {string}
 */
function cutToWidth(text, size, width) {
  if (textWidth(text, size) <= width) return text;
  const room = Math.floor(width / (size * 0.6)) - 1;
  return room <= 0 ? "" : `${text.slice(0, room).trimEnd()}…`;
}

/**
 * Return a Verdict value as shown: a finite number rounded to a whole number,
 * anything else as its text (pure).
 *
 * @param {unknown} value
 * @returns {string}
 */
function honourValueText(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? String(Math.round(value))
    : String(value ?? "");
}

/**
 * Build the verdict record: the full card with Honors and its two buttons,
 * or (full false) its band form in the top band. Pure.
 *
 * See records for the input. Full card (the store's Herald Cards verdict, 372
 * px wide; unscaled CSS px from its top): 9 px padding; the basis eyebrow;
 * two team columns (16 px crest and team word, then the controller name in
 * 15 px when any team has one, then the 32 px score in a 40 px box and the
 * 18 px result 8 px after it); the 13 px lead line of a horizon draw with a
 * lead; when Honors exist a 1 px divider, the "HONORS" eyebrow and one cell
 * per honor (its leaders' 22 px living medallions, 16 px apart, up to
 * HONOR_MEDALLIONS_MAX and then "+N"; the 11 px label; the 15 px value);
 * then the Match Summary and Show Arena buttons (24 px tall, 8 px apart,
 * centred); 9 px padding. Columns follow leftTeamId: the left column is the
 * team whose home end is on the left. The card is centred on the arena and
 * vertically in the room under the top band (store fullBox), scaled to fit
 * (between FOOTPRINT.verdict.minScale and 1). The record carries layout (the
 * y of each line), columns, honours and buttons (card-local boxes; see
 * verdictButtons). Band form: a herald plate ("herald-verdict-plate": crest,
 * score, "–", score, crest; the basis eyebrow; the result title, with the
 * lead after it), placed by placeOnBand with avoid and chrome (it has no
 * one-line form since 4 Oct 2026).
 *
 * @param {Record<string, any>} input
 * @param {Rectangle} bounds
 * @param {boolean} full
 * @param {ReadonlyArray<JsonRecord>} avoid
 * @param {ReadonlyArray<JsonRecord>} chrome
 * @returns {LookRecord | null}
 */
function verdictRecord(input, bounds, full, avoid, chrome) {
  const verdict = isRecord(input.verdict) ? input.verdict : null;
  if (verdict === null || !isRecord(verdict.scores)) return null;
  const words = verdictWords(verdict);
  const leftTeamId = Number(input.leftTeamId) === 2 ? 2 : 1;
  const rightTeamId = leftTeamId === 1 ? 2 : 1;
  const winnerTeam = Number(verdict.winnerTeamId);
  const common = {
    eventId: "herald:verdict",
    event: null,
    held: false,
    phaseStart: 0,
    title:
      winnerTeam === 1 || winnerTeam === 2
        ? `${teamLabelFromId(winnerTeam)} Victory`
        : "Draw",
    eyebrow: words.eyebrow,
    ruleTeam: null,
    fallen: Object.freeze([]),
  };
  const scoreOf = (/** @type {1 | 2} */ team) => Number(verdict.scores[team]);
  if (!full) {
    // The folded Verdict is always its herald plate (one plate style, owner
    // 4 Oct 2026), placed by the same Sliding Top Placement as moment cards.
    const size = FOOTPRINT.plate;
    const glyphWidth = 116;
    const titleWidth =
      textWidth(common.title, 18) +
      (words.lead ? textWidth(` · ${words.lead}`, 13) : 0);
    const width =
      size.padLeft +
      glyphWidth +
      size.gap +
      Math.max(textWidth(words.eyebrow, 11, true), titleWidth) +
      size.padRight;
    const spot = placeOnBand(
      bounds,
      slotOf(bounds, null),
      {
        key: "plate",
        width: width + 2 * size.tailReach,
        height: size.height,
        minScale: Math.max(size.minScale, MIN_TEXT_PX / 11),
      },
      avoid,
      chrome,
    );
    return Object.freeze({
      kind: "herald-verdict-plate",
      ...common,
      lead: words.lead,
      sides: Object.freeze([
        Object.freeze({ teamKey: teamKey(leftTeamId), score: scoreOf(leftTeamId) }),
        Object.freeze({ teamKey: teamKey(rightTeamId), score: scoreOf(rightTeamId) }),
      ]),
      glyphWidth,
      width,
      height: size.height,
      scale: spot.scale,
      x: spot.left + size.tailReach * spot.scale,
      y: spot.top,
      place: "top",
    });
  }
  const layoutSize = FOOTPRINT.verdict;
  const cardWidth = FOOTPRINT.fullCard.width;
  const columnWidth = (cardWidth - 2 * layoutSize.padX) / 2 - 4;
  const teams = isRecord(verdict.teams) ? verdict.teams : {};
  // A name that does not fit at 15 px uses its short line when the caller
  // gives one ("2 Policies"); without one (a single policy) it shrinks to 13
  // or 11 px and is cut with "…" only if it still does not fit, so a team
  // keeps its policy's own name (owner, 4 Oct 2026).
  const nameOf = (/** @type {1 | 2} */ team) => {
    const entry = isRecord(teams[team]) ? teams[team] : {};
    const name = String(entry.name ?? "").trim();
    const short = String(entry.short ?? "").trim();
    if (textWidth(name, 15) <= columnWidth) return { text: name, size: 15 };
    if (short !== "") return { text: cutToWidth(short, 15, columnWidth), size: 15 };
    for (const size of VERDICT_NAME_SIZES) {
      if (textWidth(name, size) <= columnWidth) return { text: name, size };
    }
    return { text: cutToWidth(name, 11, columnWidth), size: 11 };
  };
  const names = {
    [leftTeamId]: nameOf(leftTeamId),
    [rightTeamId]: nameOf(rightTeamId),
  };
  const named = names[1].text !== "" || names[2].text !== "";
  const honours = asArray(verdict.honours)
    .filter((row) => isRecord(row) && typeof row.label === "string")
    .slice(0, 3)
    .map((row) => {
      const leaders = asArray(row.leaders).filter(isRecord);
      const people =
        leaders.length > 0
          ? leaders
          : row.classId !== undefined || row.class_id !== undefined
            ? [row]
            : [];
      const shown = people.slice(0, HONOR_MEDALLIONS_MAX);
      return Object.freeze({
        label: String(row.label),
        value: honourValueText(row.value),
        medallions: Object.freeze(shown.map((person) => medallionOf(person, false))),
        more: people.length - shown.length,
      });
    });
  let y = layoutSize.padTop;
  const eyebrowBaseline = y + 10;
  y += 13 + 5;
  const crestY = y + 8;
  y += 16 + 2;
  const nameBaseline = named ? y + 14 : null;
  if (named) y += 18 + 2;
  const scoreBaseline = y + 29;
  y += 36;
  /** @type {number | null} */
  let leadBaseline = null;
  if (words.lead) {
    leadBaseline = y + 4 + 13;
    y += 4 + 17;
  }
  /** @type {number | null} */
  let divider = null;
  /** @type {number | null} */
  let honoursEyebrow = null;
  /** @type {number | null} */
  let honourTop = null;
  if (honours.length > 0) {
    divider = y + 7;
    y += 7 + 1 + 5;
    honoursEyebrow = y + 10;
    y += 13 + 4;
    honourTop = y;
    y += layoutSize.honorMedallion + 2 + 13 + 2 + 17;
  }
  y += layoutSize.buttonGap;
  const buttonTop = y;
  y += layoutSize.buttonHeight + layoutSize.padBottom;
  const height = y;
  const buttonWidths = VERDICT_BUTTONS.map(
    (button) => textWidth(button.label, 13) + 2 * layoutSize.buttonPadX + 2,
  );
  const buttonsWidth =
    buttonWidths.reduce((sum, width) => sum + width, 0) +
    layoutSize.buttonGap * (VERDICT_BUTTONS.length - 1);
  let buttonLeft = (cardWidth - buttonsWidth) / 2;
  const buttons = VERDICT_BUTTONS.map((button, index) => {
    const box = Object.freeze({
      id: button.id,
      label: button.label,
      left: buttonLeft,
      top: buttonTop,
      width: buttonWidths[index],
      height: layoutSize.buttonHeight,
    });
    buttonLeft += buttonWidths[index] + layoutSize.buttonGap;
    return box;
  });
  const top = bounds.top + FOOTPRINT.topGap;
  const room = Math.max(40, bounds.bottom - 6 - top);
  const scale = Math.min(
    1,
    Math.max(
      layoutSize.minScale,
      Math.min(room / height, (bounds.width - 16) / cardWidth),
    ),
  );
  return Object.freeze({
    kind: "herald-verdict",
    ...common,
    lead: words.lead,
    columns: Object.freeze(
      [leftTeamId, rightTeamId].map((team) =>
        Object.freeze({
          teamKey: teamKey(team),
          teamId: team,
          score: scoreOf(/** @type {1 | 2} */ (team)),
          result: words.results[/** @type {1 | 2} */ (team)],
          name: names[team].text,
          nameSize: names[team].size,
        }),
      ),
    ),
    honours: Object.freeze(honours),
    layout: Object.freeze({
      eyebrowBaseline,
      crestY,
      nameBaseline,
      scoreBaseline,
      leadBaseline,
      divider,
      honoursEyebrow,
      honourTop,
    }),
    buttons: Object.freeze(buttons),
    width: cardWidth,
    height,
    scale,
    x: bounds.left + bounds.width / 2 - (cardWidth * scale) / 2,
    y: top + Math.max(0, (room - height * scale) / 2),
  });
}

/**
 * Draw herald records.
 *
 * records come from records(); doc creates nodes (it may be null only when
 * records is empty). With mount (the painter's EventMount) the card goes into
 * mount.group and, when the mount animates and the record is not held, its
 * entry animations go to mount.animationSpecs (fill "backwards"). Without a
 * mount each card is a new detached group. Return [group, connector] with a
 * mount, else the card groups. Export-safe SVG; no listeners (the verdict's
 * buttons are drawn here, and made to work by buildVerdictActions).
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @param {EventMount} [mount]
 * @returns {SVGElement[]}
 */
export function build(records, doc, mount) {
  if (records.length === 0) {
    return [];
  }
  /** @type {SVGElement[]} */
  const cards = [];
  for (const record of records) {
    const card =
      record.kind === "herald-plate"
        ? buildPlate(doc, record, mount)
        : record.kind === "herald-lineup"
          ? buildLineup(doc, record)
          : record.kind === "herald-verdict"
            ? buildVerdict(doc, record)
            : record.kind === "herald-verdict-plate"
              ? buildVerdictPlate(doc, record)
              : buildBand(doc, record);
    if (mount) {
      mount.group.append(card);
    }
    cards.push(card);
  }
  return mount ? [mount.group, mount.connector] : cards;
}

/**
 * Format a number for an SVG attribute (at most two decimals). Pure.
 *
 * @param {number} value
 * @returns {string}
 */
function fmt(value) {
  return String(Math.round(value * 100) / 100);
}

/**
 * Return path data for a full circle around (cx, cy) with radius r. Pure.
 *
 * @param {number} cx
 * @param {number} cy
 * @param {number} r
 */
function ringPath(cx, cy, r) {
  return `M${fmt(cx - r)} ${fmt(cy)}a${fmt(r)} ${fmt(r)} 0 1 0 ${fmt(2 * r)} 0a${fmt(r)} ${fmt(r)} 0 1 0 ${fmt(-2 * r)} 0`;
}

/**
 * Append a text node with class and words at (x, y).
 *
 * @param {Document} doc
 * @param {Element} parent
 * @param {string} className
 * @param {string} words
 * @param {number} x
 * @param {number} y
 * @param {Record<string, string | number>} [extra]
 */
function appendText(doc, parent, className, words, x, y, extra = {}) {
  const node = effectNode(doc, "text", {
    class: className,
    x: fmt(x),
    y: fmt(y),
    ...extra,
  });
  node.textContent = words;
  parent.append(node);
  return node;
}

/**
 * Append a team crest (one ring Team A, two rings Team B) at (cx, cy).
 *
 * @param {Document} doc
 * @param {Element} parent
 * @param {number} cx
 * @param {number} cy
 * @param {number} r outer radius in CSS px.
 * @param {TeamKey} team
 */
function appendCrest(doc, parent, cx, cy, r, team) {
  parent.append(
    effectNode(doc, "path", {
      class: `herald__crest herald__crest--team-${team}`,
      d:
        team === "b"
          ? ringPath(cx, cy, r) + ringPath(cx, cy, Math.max(1, r - 2.6))
          : ringPath(cx, cy, r),
      "stroke-width": team === "b" ? 1.5 : 2,
    }),
  );
}

/**
 * Append one medallion of diameter size centred at (cx, cy).
 *
 * Living: one path whose fill is the team field and whose 2.5 px stroke is
 * the team ring (r 10.75 of 26), plus, for Team B, a thin ink ring over it;
 * then the class sigil in its class colour (icon paint "class"). Fallen: one
 * Grave Stone path with a 1 px team ring at 55% (Team B doubled inside, r
 * 9.4 of 26), then the sigil in ash (icon paint "ash") at 70%. The ring path
 * carries `herald__medallion` and its team (and fallen) modifiers, so each
 * medallion draws 4 nodes (5 for a living Team B one). Return undefined.
 *
 * @param {Document} doc
 * @param {Element} parent
 * @param {Medallion} medallion
 * @param {number} cx
 * @param {number} cy
 * @param {number} size
 */
function appendMedallion(doc, parent, medallion, cx, cy, size) {
  const k = size / 26;
  const team = medallion.teamKey;
  const base = `herald__medallion herald__medallion--team-${team}`;
  if (medallion.fallen) {
    parent.append(
      effectNode(doc, "path", {
        class: `${base} herald__medallion--fallen`,
        d:
          team === "b"
            ? ringPath(cx, cy, 11.5 * k) + ringPath(cx, cy, 9.4 * k)
            : ringPath(cx, cy, 11.5 * k),
        "stroke-width": 1,
      }),
    );
  } else {
    parent.append(
      effectNode(doc, "path", {
        class: base,
        d: ringPath(cx, cy, 10.75 * k),
        "stroke-width": fmt(2.5 * k),
      }),
    );
    if (team === "b") {
      parent.append(
        effectNode(doc, "path", {
          class: "herald__medallion-ink",
          d: ringPath(cx, cy, 10.75 * k),
          "stroke-width": fmt(0.9 * k),
        }),
      );
    }
  }
  const emblemSize = 12.6 * k;
  const icon = createSvgIcon(
    doc,
    medallion.classKey === null ? "unknown" : `class-${medallion.classKey}`,
    {
      size: emblemSize,
      paint: medallion.fallen ? "ash" : "class",
      className: `herald__emblem herald__emblem--${medallion.fallen ? "fallen" : (medallion.classKey ?? "unknown")}`,
    },
  );
  icon.setAttribute("x", fmt(cx - emblemSize / 2));
  icon.setAttribute("y", fmt(cy - emblemSize / 2));
  icon.setAttribute("width", fmt(emblemSize));
  icon.setAttribute("height", fmt(emblemSize));
  if (medallion.fallen) icon.setAttribute("opacity", "0.7");
  parent.append(icon);
}

/**
 * Return the card's outer group placed at (x, y) and scaled.
 *
 * The group carries data-herald-kind (the record kind), data-herald-title,
 * data-herald-moments (the moment kinds a herald card names, in order, such
 * as "multi trade"; absent for the lineup, verdict and state chip), role img
 * and an aria-label of the title and eyebrow.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @param {string} kindClass
 */
function cardGroup(doc, record, kindClass) {
  return effectNode(doc, "g", {
    class: `herald ${kindClass}${record.ruleTeam ? ` herald--team-${record.ruleTeam}` : " herald--neutral"}`,
    "data-herald-kind": record.kind,
    "data-herald-title": record.title,
    "data-herald-moments": asArray(record.kinds).join(" ") || false,
    role: "img",
    "aria-label": record.eyebrow ? `${record.title}. ${record.eyebrow}` : record.title,
    transform: `translate(${fmt(record.x)} ${fmt(record.y)}) scale(${fmt(record.scale)})`,
  });
}

/**
 * Append a plate's placed glyphs to parent (record.glyphs; see glyphLayout).
 *
 * glyphs is the PlacedGlyph list; mid the plate's vertical centre in its own
 * px. Each type draws: "diamond" an 11 px rune diamond in the caster's class
 * color (`herald__diamond--<class>`); "crest" a team crest (one ring Team A,
 * two Team B) of its size; "rescue" a living medallion (24 px) inside a
 * 1.5 px Dawn Gold ring of 30 px with a small gold "+" at its top right;
 * "crescent" the Ticks Left moon (a 1 px Moon Silver circle at 50% and its
 * lit part, ticks left out of 30); "divider" a 1 px line; "medallion" one 26
 * px medallion. Return undefined.
 *
 * @param {Document} doc
 * @param {Element} parent
 * @param {ReadonlyArray<JsonRecord>} glyphs
 * @param {number} mid
 */
function appendGlyphs(doc, parent, glyphs, mid) {
  const size = FOOTPRINT.plate;
  for (const glyph of glyphs) {
    const x = Number(glyph.x);
    switch (glyph.type) {
      case "diamond":
        parent.append(
          effectNode(doc, "path", {
            class: `herald__diamond herald__diamond--${glyph.classKey ?? "unknown"}`,
            d: `M${fmt(x + 5.5)} ${fmt(mid - 4.7)}L${fmt(x + 10.2)} ${fmt(mid)}L${fmt(x + 5.5)} ${fmt(mid + 4.7)}L${fmt(x + 0.8)} ${fmt(mid)}Z`,
          }),
        );
        break;
      case "crest":
        appendCrest(
          doc,
          parent,
          x + Number(glyph.size) / 2,
          mid,
          Number(glyph.size) / 2 - 1,
          glyph.teamKey === "b" ? "b" : "a",
        );
        break;
      case "rescue":
        appendMedallion(doc, parent, glyph.medallion, x + 15, mid, 24);
        parent.append(
          effectNode(doc, "path", {
            class: "herald__rescue-ring",
            d: ringPath(x + 15, mid, 14.25),
            "stroke-width": 1.5,
          }),
        );
        appendText(doc, parent, "herald__rescue-plus", "+", x + 25, mid - 8);
        break;
      case "crescent": {
        const cx = x + 11;
        const lit = Math.min(1, Math.max(0.04, Number(glyph.left) / 30));
        const xm = 20 - 18 * lit;
        const terminator =
          xm >= 11
            ? `A${fmt(xm - 11)} 9 0 0 0 ${fmt(cx)} ${fmt(mid - 9)}`
            : `A${fmt(11 - xm)} 9 0 0 1 ${fmt(cx)} ${fmt(mid - 9)}`;
        parent.append(
          effectNode(doc, "path", {
            class: "herald__crescent-rim",
            d: ringPath(cx, mid, 9),
            "stroke-width": 1,
            opacity: 0.5,
          }),
          effectNode(doc, "path", {
            class: "herald__crescent",
            d: `M${fmt(cx)} ${fmt(mid - 9)}A9 9 0 0 1 ${fmt(cx)} ${fmt(mid + 9)}${terminator}Z`,
          }),
        );
        break;
      }
      case "divider":
        parent.append(
          effectNode(doc, "path", {
            class: "herald__divider",
            d: `M${fmt(x + 13)} ${fmt(mid - 9)}V${fmt(mid + 9)}`,
            "stroke-width": 1,
          }),
        );
        break;
      case "medallion":
        appendMedallion(
          doc,
          parent,
          glyph.medallion,
          x + size.medallion / 2,
          mid,
          size.medallion,
        );
        break;
      default:
        break;
    }
  }
}

/**
 * Return the path data of a tier-1 record's Herald Rule: a straight line from
 * tierRule.x0 to tierRule.x1 at tierRule.y, in the card's own unscaled px.
 * Empty when the record has no tierRule. Pure.
 *
 * @param {LookRecord} record
 * @returns {string}
 */
function tierRulePath(record) {
  const rule = isRecord(record.tierRule) ? record.tierRule : null;
  return rule === null
    ? ""
    : `M${fmt(Number(rule.x0))} ${fmt(Number(rule.y))}H${fmt(Number(rule.x1))}`;
}

/**
 * Draw a herald plate and, when mount animates, its entry animations.
 *
 * Layout (unscaled CSS px, origin at the plate's top-left): swallowtails reach
 * 10 px out at both ends (14 x 30, 5 px down); the 40 px panel (96%) with a
 * 2 px team rule along its top edge (10 px in, 70%; neutral 50%); the glyphs
 * at the left (record.glyphs, laid out by glyphLayout and drawn by
 * appendGlyphs: the Ultimate diamond, crests, the Rescue medallion, the
 * Ticks Left crescent, medallions with a Trade's divider); the
 * 11 px uppercase eyebrow and the 18 px title stacked on the right. A tier-1
 * record's Herald Rule (record.tierRule) is drawn first, behind the plate: a
 * 1 CSS px line at 35% in the rule color across the band. With an animating
 * mount and a record that is not held, the plate unfurls (opacity 0 to 1,
 * 92% to full width, 240 ms; opacity only over 120 ms in reduced motion),
 * the glyphs fade in after it and the Herald Rule grows from the card's
 * centre over 320 ms, all from record.phaseStart (times the plan's reduced
 * scale in reduced motion). A record with a finite exitAt (the painter's card
 * queue handing over to the next card) also fades the whole card out over
 * CARD_FADE_OUT_MS from exitAt, fill "backwards": the choreographer then
 * keeps it hidden for the rest of the tick (its painter row is not
 * persistent). Return the card group.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @param {EventMount} [mount]
 * @returns {SVGElement}
 */
function buildPlate(doc, record, mount) {
  const size = FOOTPRINT.plate;
  const card = cardGroup(doc, record, "herald--plate");
  if (record.held) card.setAttribute("data-herald-held", "true");
  const body = effectNode(doc, "g", {
    class: "herald__body",
    transform: `translate(${fmt(record.width / 2)} 0)`,
  });
  const inner = effectNode(doc, "g", {
    transform: `translate(${fmt(-record.width / 2)} 0)`,
  });
  const tailTop = (size.height - size.tailHeight) / 2;
  const tail = `M${size.tailWidth} 0H0L6 ${size.tailHeight / 2}L0 ${size.tailHeight}H${size.tailWidth}Z`;
  inner.append(
    effectNode(doc, "path", {
      class: "herald__tail",
      d: tail,
      transform: `translate(${-size.tailReach} ${tailTop})`,
    }),
    effectNode(doc, "path", {
      class: "herald__tail",
      d: tail,
      transform: `translate(${fmt(record.width + size.tailReach)} ${tailTop}) scale(-1 1)`,
    }),
    effectNode(doc, "rect", {
      class: "herald__panel",
      x: 0,
      y: 0,
      width: fmt(record.width),
      height: size.height,
      rx: 8,
    }),
    effectNode(doc, "rect", {
      class: "herald__rule",
      x: size.ruleInset,
      y: -1,
      width: fmt(Math.max(0, record.width - 2 * size.ruleInset)),
      height: 2,
      opacity: record.ruleTeam ? 0.7 : 0.5,
    }),
  );
  const glyphs = effectNode(doc, "g", { class: "herald__glyphs" });
  appendGlyphs(doc, glyphs, asArray(record.glyphs), size.height / 2);
  inner.append(glyphs);
  const textX = size.padLeft + record.glyphWidth + size.gap;
  appendText(doc, inner, "herald__eyebrow", record.eyebrow.toUpperCase(), textX, 15);
  appendText(doc, inner, "herald__title", record.title, textX, 32);
  body.append(inner);
  /** @type {SVGElement | null} */
  let tier = null;
  if (isRecord(record.tierRule)) {
    // The Herald Rule runs behind the plate, so it is the card's first child.
    tier = effectNode(doc, "g", { class: "herald__tier" });
    tier.append(
      effectNode(doc, "path", {
        class: "herald__tier-rule",
        d: tierRulePath(record),
        "stroke-width": fmt(1 / Math.max(0.01, Number(record.scale) || 1)),
        opacity: 0.35,
      }),
    );
    card.append(tier);
  }
  card.append(body);
  if (
    mount &&
    Number.isFinite(record.exitAt) &&
    !mount.options.settled &&
    mount.options.motionMode !== "off"
  ) {
    mount.animationSpecs.push(
      animationSpec(
        card,
        [{ opacity: 1 }, { opacity: 0 }],
        {
          delay: Math.max(0, Number(record.exitAt)),
          duration: CARD_FADE_OUT_MS,
          easing: "ease-in",
          fill: "backwards",
        },
        mount.plan,
        record.event ?? { eventId: record.eventId },
        "herald-exit",
      ),
    );
  }
  if (mount && !record.held) {
    const { options, plan } = mount;
    if (!options.settled && options.motionMode !== "off") {
      const reduced = options.motionMode === "reduced";
      const scale = reduced ? Number(plan.beats?.reducedScale ?? 1) || 1 : 1;
      const delay = Math.max(0, record.phaseStart * scale);
      const event = record.event ?? { eventId: record.eventId };
      mount.animationSpecs.push(
        animationSpec(
          body,
          reduced
            ? [{ opacity: 0 }, { opacity: 1 }]
            : [
                {
                  opacity: 0,
                  transform: `translate(${fmt(record.width / 2)}px, 0px) scale(0.92, 1)`,
                },
                {
                  opacity: 1,
                  transform: `translate(${fmt(record.width / 2)}px, 0px) scale(1, 1)`,
                },
              ],
          {
            delay,
            duration: reduced ? 120 : 240,
            easing: "cubic-bezier(0.33, 1, 0.68, 1)",
            fill: "backwards",
          },
          plan,
          event,
          "herald-card",
        ),
      );
      if (!reduced) {
        mount.animationSpecs.push(
          animationSpec(
            glyphs,
            [
              { opacity: 0, offset: 0 },
              { opacity: 0, offset: 1 / 3 },
              { opacity: 1, offset: 1 },
            ],
            { delay, duration: 240, easing: "ease-out", fill: "backwards" },
            plan,
            event,
            "herald-glyphs",
          ),
        );
      }
      if (tier !== null) {
        const centre = record.width / 2;
        mount.animationSpecs.push(
          animationSpec(
            tier,
            reduced
              ? [{ opacity: 0 }, { opacity: 1 }]
              : [
                  {
                    transform: `translate(${fmt(centre)}px, 0px) scale(0, 1) translate(${fmt(-centre)}px, 0px)`,
                  },
                  {
                    transform: `translate(${fmt(centre)}px, 0px) scale(1, 1) translate(${fmt(-centre)}px, 0px)`,
                  },
                ],
            {
              delay,
              duration: reduced ? 120 : 320,
              easing: "cubic-bezier(0.33, 1, 0.68, 1)",
              fill: "backwards",
            },
            plan,
            event,
            "herald-rule",
          ),
        );
      }
    }
  }
  return card;
}

/**
 * Draw a one-line band: the Lineup's folded form ("herald-band", with
 * `moment_full_cards` off). Moment cards and the folded Verdict are always
 * plates since 4 Oct 2026, so this is the only band left.
 *
 * 22 px tall panel (radius 6) with small swallowtails and a 2 px rule along
 * its top (neutral, at 50%; a team's color at 70% when record.ruleTeam is
 * set), then the crests (record.crests) and the 13 px words (record.text).
 * No entry animation. Return the card group.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement}
 */
function buildBand(doc, record) {
  const band = FOOTPRINT.band;
  const card = cardGroup(doc, record, "herald--band");
  if (record.held) card.setAttribute("data-herald-held", "true");
  const tail = "M10 0H0L4 8L0 16H10Z";
  card.append(
    effectNode(doc, "path", {
      class: "herald__tail",
      d: tail,
      transform: "translate(-7 3)",
    }),
    effectNode(doc, "path", {
      class: "herald__tail",
      d: tail,
      transform: `translate(${fmt(record.width + 7)} 3) scale(-1 1)`,
    }),
    effectNode(doc, "rect", {
      class: "herald__panel",
      x: 0,
      y: 0,
      width: fmt(record.width),
      height: band.height,
      rx: 6,
    }),
    effectNode(doc, "rect", {
      class: "herald__rule",
      x: 8,
      y: -1,
      width: fmt(Math.max(0, record.width - 16)),
      height: 2,
      opacity: record.ruleTeam ? 0.7 : 0.5,
    }),
  );
  let x = band.pad;
  const mid = band.height / 2;
  for (const team of asArray(record.crests)) {
    appendCrest(doc, card, x + band.crest / 2, mid, band.crest / 2 - 1, team);
    x += band.crest + band.gap;
  }
  appendText(doc, card, "herald__band-text", record.text, x, mid + 0.5);
  return card;
}

/**
 * Draw the full lineup card: eyebrow, map title, a divider, two team columns
 * (crest, "TEAM A" eyebrow, medallions) and the rules lines. Return the group.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement}
 */
function buildLineup(doc, record) {
  const card = cardGroup(doc, record, "herald--lineup");
  const { width, pad, medallion } = FOOTPRINT.fullCard;
  card.append(
    effectNode(doc, "rect", {
      class: "herald__panel",
      x: 0,
      y: 0,
      width,
      height: fmt(record.height),
      rx: 8,
    }),
  );
  appendText(
    doc,
    card,
    "herald__eyebrow herald__centre",
    record.eyebrow.toUpperCase(),
    width / 2,
    20,
  );
  appendText(
    doc,
    card,
    "herald__title herald__title--large herald__centre",
    record.title,
    width / 2,
    46,
  );
  card.append(
    effectNode(doc, "rect", {
      class: "herald__line",
      x: pad,
      y: 64,
      width: width - 2 * pad,
      height: 1,
    }),
  );
  const columnTop = 82;
  record.columns.forEach(
    (/** @type {JsonRecord} */ column, /** @type {number} */ index) => {
      const right = index === 1;
      const edge = right ? width - pad : pad;
      const sign = right ? -1 : 1;
      appendCrest(doc, card, edge + sign * 8, columnTop, 7, column.teamKey);
      appendText(
        doc,
        card,
        `herald__eyebrow${right ? " herald__end" : ""}`,
        String(teamLabelFromId(column.teamId)).toUpperCase(),
        edge + sign * 22,
        columnTop + 1,
      );
      column.medallions.forEach(
        (/** @type {Medallion} */ each, /** @type {number} */ order) => {
          appendMedallion(
            doc,
            card,
            each,
            edge + sign * (medallion / 2 + order * (medallion + 3)),
            columnTop + 12 + medallion / 2,
            medallion,
          );
        },
      );
    },
  );
  card.append(
    effectNode(doc, "rect", {
      class: "herald__line",
      x: width / 2,
      y: columnTop - 8,
      width: 1,
      height: medallion + 26,
    }),
  );
  record.rules.forEach((/** @type {string} */ line, /** @type {number} */ index) => {
    appendText(
      doc,
      card,
      "herald__rules herald__centre",
      line,
      width / 2,
      columnTop + medallion + 30 + index * 17,
    );
  });
  return card;
}

/**
 * Draw the full verdict card from its record (see verdictRecord for the
 * layout): panel, basis eyebrow, two team columns (crest, team word,
 * controller name in the team's light color, 32 px score in the team color,
 * 18 px result: Victory and Defeat in UI Text, Draw in Moon Silver), the lead
 * line, then the Honors divider, eyebrow and cells (leaders' living
 * medallions, label, value), then the Match Summary and Show Arena buttons
 * (13 px Moon Silver words on a Moon Silver outline at 45%). Every button
 * group carries data-verdict-action (its id). Return the group.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement}
 */
function buildVerdict(doc, record) {
  const card = cardGroup(doc, record, "herald--verdict");
  const { width } = FOOTPRINT.fullCard;
  const size = FOOTPRINT.verdict;
  const pad = size.padX;
  const layout = record.layout;
  card.append(
    effectNode(doc, "rect", {
      class: "herald__panel",
      x: 0,
      y: 0,
      width,
      height: fmt(record.height),
      rx: 8,
    }),
  );
  appendText(
    doc,
    card,
    "herald__eyebrow herald__centre",
    record.eyebrow.toUpperCase(),
    width / 2,
    layout.eyebrowBaseline,
  );
  record.columns.forEach(
    (/** @type {JsonRecord} */ column, /** @type {number} */ index) => {
      const right = index === 1;
      const edge = right ? width - pad : pad;
      const sign = right ? -1 : 1;
      const end = right ? " herald__end" : "";
      appendCrest(doc, card, edge + sign * 8, layout.crestY, 7, column.teamKey);
      appendText(
        doc,
        card,
        `herald__eyebrow${end}`,
        String(teamLabelFromId(column.teamId)).toUpperCase(),
        edge + sign * 22,
        layout.crestY + 4,
      );
      if (layout.nameBaseline !== null && column.name !== "") {
        const size =
          Number(column.nameSize) === 15 ? "" : ` herald__name--${column.nameSize}`;
        appendText(
          doc,
          card,
          `herald__name herald__name--team-${column.teamKey}${size}${end}`,
          column.name,
          edge,
          layout.nameBaseline,
        );
      }
      appendText(
        doc,
        card,
        `herald__score herald__score--team-${column.teamKey}${end}`,
        String(column.score),
        edge,
        layout.scoreBaseline,
      );
      appendText(
        doc,
        card,
        `herald__result${column.result === "Draw" ? " herald__result--draw" : ""}${end}`,
        column.result,
        edge + sign * 48,
        layout.scoreBaseline,
      );
    },
  );
  if (record.lead && layout.leadBaseline !== null) {
    appendText(
      doc,
      card,
      "herald__lead herald__centre",
      record.lead,
      width / 2,
      layout.leadBaseline,
    );
  }
  if (record.honours.length > 0 && layout.divider !== null) {
    card.append(
      effectNode(doc, "rect", {
        class: "herald__line",
        x: pad,
        y: fmt(layout.divider),
        width: width - 2 * pad,
        height: 1,
      }),
    );
    appendText(
      doc,
      card,
      "herald__eyebrow herald__centre",
      "HONORS",
      width / 2,
      layout.honoursEyebrow,
    );
    const count = record.honours.length;
    const cell = (width - 2 * pad - size.honorGap * (count - 1)) / count;
    const r = size.honorMedallion / 2;
    record.honours.forEach(
      (/** @type {JsonRecord} */ honour, /** @type {number} */ index) => {
        const cx = pad + index * (cell + size.honorGap) + cell / 2;
        const shown = honour.medallions.length;
        const moreText = honour.more > 0 ? `+${honour.more}` : "";
        const moreWidth = moreText === "" ? 0 : 4 + textWidth(moreText, 11);
        const groupWidth =
          shown === 0 ? 0 : size.honorMedallion + (shown - 1) * 16 + moreWidth;
        let x = cx - groupWidth / 2 + r;
        const cy = layout.honourTop + r;
        for (const medallion of honour.medallions) {
          appendMedallion(doc, card, medallion, x, cy, size.honorMedallion);
          x += 16;
        }
        if (moreText !== "") {
          appendText(
            doc,
            card,
            "herald__honour-more",
            moreText,
            x - 16 + r + 4,
            cy + 4,
          );
        }
        appendText(
          doc,
          card,
          "herald__honour-label herald__centre",
          honour.label,
          cx,
          layout.honourTop + size.honorMedallion + 2 + 10,
        );
        appendText(
          doc,
          card,
          "herald__honour-value herald__centre",
          honour.value,
          cx,
          layout.honourTop + size.honorMedallion + 2 + 13 + 2 + 13,
        );
      },
    );
  }
  for (const button of record.buttons) {
    const group = effectNode(doc, "g", {
      class: "herald__button",
      "data-verdict-action": button.id,
    });
    group.append(
      effectNode(doc, "rect", {
        class: "herald__button-frame",
        x: fmt(button.left + 0.5),
        y: fmt(button.top + 0.5),
        width: fmt(button.width - 1),
        height: fmt(button.height - 1),
        rx: 8,
      }),
    );
    appendText(
      doc,
      group,
      "herald__button-text herald__centre",
      button.label,
      button.left + button.width / 2,
      button.top + button.height / 2 + 4.5,
    );
    card.append(group);
  }
  return card;
}

/**
 * Draw the verdict's band form: a herald plate with the left team's crest and
 * score, "–", the right team's score and crest (18 px scores in the team
 * colors), then the basis eyebrow and the result title (Draw in Moon Silver,
 * followed by the lead line in 13 px when there is one), with a neutral rule.
 * No entry animation (the page draws it as it stands). Return the group.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @returns {SVGElement}
 */
function buildVerdictPlate(doc, record) {
  const size = FOOTPRINT.plate;
  const card = cardGroup(doc, record, "herald--plate herald--verdict-plate");
  const tailTop = (size.height - size.tailHeight) / 2;
  const tail = `M${size.tailWidth} 0H0L6 ${size.tailHeight / 2}L0 ${size.tailHeight}H${size.tailWidth}Z`;
  card.append(
    effectNode(doc, "path", {
      class: "herald__tail",
      d: tail,
      transform: `translate(${-size.tailReach} ${tailTop})`,
    }),
    effectNode(doc, "path", {
      class: "herald__tail",
      d: tail,
      transform: `translate(${fmt(record.width + size.tailReach)} ${tailTop}) scale(-1 1)`,
    }),
    effectNode(doc, "rect", {
      class: "herald__panel",
      x: 0,
      y: 0,
      width: fmt(record.width),
      height: size.height,
      rx: 8,
    }),
    effectNode(doc, "rect", {
      class: "herald__rule",
      x: size.ruleInset,
      y: -1,
      width: fmt(Math.max(0, record.width - 2 * size.ruleInset)),
      height: 2,
      opacity: 0.5,
    }),
  );
  const mid = size.height / 2;
  const [left, right] = record.sides;
  const x0 = size.padLeft;
  appendCrest(doc, card, x0 + 9, mid, 8, left.teamKey);
  appendText(
    doc,
    card,
    `herald__band-score herald__score--team-${left.teamKey} herald__end`,
    String(left.score),
    x0 + 48,
    mid + 6,
  );
  appendText(doc, card, "herald__dash herald__centre", "–", x0 + 58, mid + 4.5);
  appendText(
    doc,
    card,
    `herald__band-score herald__score--team-${right.teamKey}`,
    String(right.score),
    x0 + 68,
    mid + 6,
  );
  appendCrest(doc, card, x0 + 107, mid, 8, right.teamKey);
  const textX = x0 + record.glyphWidth + size.gap;
  appendText(doc, card, "herald__eyebrow", record.eyebrow.toUpperCase(), textX, 15);
  const title = appendText(
    doc,
    card,
    `herald__title${record.title === "Draw" ? " herald__result--draw" : ""}`,
    record.title,
    textX,
    32,
  );
  if (record.lead) {
    const lead = effectNode(doc, "tspan", { class: "herald__lead" });
    lead.textContent = ` · ${record.lead}`;
    title.append(lead);
  }
  return card;
}

/**
 * Return a full Verdict card's buttons with their boxes in the arena's CSS
 * px (the herald layer's coordinates): each card-local box moved by the
 * card's x and y and multiplied by its scale. Any other record gives [].
 * Pure.
 *
 * @param {LookRecord} record
 * @returns {ReadonlyArray<VerdictButton>}
 */
export function verdictButtons(record) {
  if (!isRecord(record) || record.kind !== "herald-verdict") {
    return Object.freeze([]);
  }
  const scale = Number(record.scale);
  return Object.freeze(
    asArray(record.buttons)
      .filter(isRecord)
      .map((button) =>
        Object.freeze({
          id: /** @type {VerdictButton["id"]} */ (button.id),
          label: String(button.label),
          left: Number(record.x) + Number(button.left) * scale,
          top: Number(record.y) + Number(button.top) * scale,
          width: Number(button.width) * scale,
          height: Number(button.height) * scale,
        }),
      ),
  );
}

/**
 * Build the working Match Summary and Show Arena buttons of a full Verdict
 * card.
 *
 * doc creates the nodes (page chrome, HTML); record is a "herald-verdict"
 * record; onPress(id) is called with "summary" or "show_arena" when the
 * player clicks a button or presses Enter or Space on it. Return a new,
 * detached `div.herald-verdict-actions` that the caller lays over the arena
 * with its top-left corner at the herald layer's origin (the same CSS px as
 * bounds): one clear, wordless `button` (type "button", data-verdict-action,
 * its label as its accessible name, aria-label) over each drawn button
 * (verdictButtons), placed through its style object (left, top, width and
 * height in px; the page's Content-Security-Policy refuses style
 * attributes). The card's SVG draws the look; these give the keyboard focus
 * ring and the click. Any other record gives an empty container. Side
 * effects: creates DOM nodes and one click listener per button; no timers or
 * storage. Agent POV: carries no match fact.
 *
 * @param {Document} doc
 * @param {LookRecord} record
 * @param {(id: VerdictButton["id"]) => void} onPress
 * @returns {HTMLElement}
 */
export function buildVerdictActions(doc, record, onPress) {
  const container = htmlNode(doc, "div", {
    class: "herald-verdict-actions",
    role: "group",
    "aria-label": "Verdict",
  });
  for (const button of verdictButtons(record)) {
    // No words of its own: the card's SVG shows them, and a page's button
    // color rules can never make them show twice.
    const node = htmlNode(doc, "button", {
      type: "button",
      class: "herald-verdict-actions__button",
      "data-verdict-action": button.id,
      "aria-label": button.label,
    });
    // The page's Content-Security-Policy refuses style attributes, so the box
    // is set through the element's style object instead.
    node.style.left = `${fmt(button.left)}px`;
    node.style.top = `${fmt(button.top)}px`;
    node.style.width = `${fmt(button.width)}px`;
    node.style.height = `${fmt(button.height)}px`;
    node.addEventListener("click", () => onPress(button.id));
    container.append(node);
  }
  return container;
}

/**
 * Turn a frame's `match_honors` host fact into the verdict's honours rows.
 *
 * matchHonors is the frame's own field (viewer.match_summary.MatchHonorsV1 on
 * the wire: {honors: [{metric, label, value, leaders: [{public_agent_id,
 * team_id, class_id}]}], ...}) or null. Return the rows records({part:
 * "verdict"}) draws: [{label, value, leaders: [{classId, teamId,
 * publicAgentId}]}], in the host's order; [] for null or a malformed fact.
 * Pure: reads only the given host fact (the host lists every champion in
 * both views, since Honors come only once the game is over).
 *
 * @param {unknown} matchHonors
 * @returns {ReadonlyArray<Readonly<{label: string, value: number, leaders: ReadonlyArray<Readonly<{classId: number, teamId: number, publicAgentId: string}>>}>>}
 */
export function verdictHonours(matchHonors) {
  if (!isRecord(matchHonors)) return Object.freeze([]);
  return Object.freeze(
    asArray(matchHonors.honors)
      .filter(
        (row) =>
          isRecord(row) && typeof row.label === "string" && Number.isFinite(row.value),
      )
      .map((row) =>
        Object.freeze({
          label: String(row.label),
          value: Number(row.value),
          leaders: Object.freeze(
            asArray(row.leaders)
              .filter(isRecord)
              .map((leader) =>
                Object.freeze({
                  classId: Number(leader.class_id),
                  teamId: Number(leader.team_id),
                  publicAgentId: String(leader.public_agent_id ?? ""),
                }),
              ),
          ),
        }),
      ),
  );
}

/**
 * Build the Match Summary sheet the Verdict card's Match Summary button
 * opens (page chrome, HTML; the store's Match Summary Sheet).
 *
 * doc creates the nodes; matchHonors is the frame's own `match_honors` host
 * fact (see verdictHonours; its labels give the row words from the metric
 * catalog and its champions the columns). options: displayIdOf maps a public
 * ID to its shown ID (null leaves the ID out; default no IDs); leftTeamId (1
 * or 2, default 1) is the team listed first, as the Verdict's left column;
 * onClose is called by the Close button. Return a new, detached
 * `section.match-summary-sheet` (aria-label "Match Summary"): a header
 * ("Match Summary", the view's scope words "Every Champion" or "Your Team's
 * Champions", and Close), then one table with one row per metric (Kill
 * Contributions, Deaths, Damage Done, Effective Healing Done, Red Zone
 * Deaths, Ultimate Ability Activations, in the host's order) and, for each
 * team in the fact, a total column ("Team A") then one column per champion
 * ("Mage · ID 0"). It is short enough to fit in the BattleClient deck's
 * place below the arena. An unknown Kill Contributions reads "—". Numbers
 * use tabular figures. A null or malformed fact gives a sheet that says "No
 * Match Summary for This Game.". Side effects: creates DOM nodes and one
 * click listener; no timers or storage. Agent POV: shows exactly the fact
 * given; the host lists every champion in both views once the game is over
 * (a fact for one team reads "Your Team's Champions").
 *
 * @param {Document} doc
 * @param {unknown} matchHonors
 * @param {{
 *   displayIdOf?: ((publicAgentId: string) => string | null) | null,
 *   leftTeamId?: 1 | 2,
 *   onClose?: (() => void) | null,
 * }} [options]
 * @returns {HTMLElement}
 */
export function buildMatchSummarySheet(doc, matchHonors, options = {}) {
  const sheet = htmlNode(doc, "section", {
    class: "match-summary-sheet",
    "aria-label": "Match Summary",
  });
  const header = htmlNode(doc, "header", { class: "match-summary-sheet__head" });
  const fact = isRecord(matchHonors) ? matchHonors : null;
  header.append(
    htmlNode(doc, "h3", { class: "match-summary-sheet__title" }, "Match Summary"),
    htmlNode(
      doc,
      "span",
      { class: "match-summary-sheet__scope" },
      fact === null
        ? ""
        : fact.team_id === 1 || fact.team_id === 2
          ? "Your Team's Champions"
          : "Every Champion",
    ),
  );
  const close = htmlNode(
    doc,
    "button",
    { type: "button", class: "match-summary-sheet__close" },
    "Close",
  );
  close.addEventListener("click", () => options.onClose?.());
  header.append(close);
  sheet.append(header);
  const labels = asArray(fact?.labels).filter(isRecord);
  const champions = asArray(fact?.champions).filter(isRecord);
  if (labels.length === 0 || champions.length === 0) {
    sheet.append(
      htmlNode(
        doc,
        "p",
        { class: "match-summary-sheet__empty" },
        "No Match Summary for This Game.",
      ),
    );
    return sheet;
  }
  const displayIdOf =
    typeof options.displayIdOf === "function" ? options.displayIdOf : () => null;
  const first = options.leftTeamId === 2 ? 2 : 1;
  /** @type {Array<{name: string, className: string, valueOf: (metric: string) => string}>} */
  const columns = [];
  for (const teamId of [first, 3 - first]) {
    const members = champions.filter((row) => Number(row.team_id) === teamId);
    if (members.length === 0) continue;
    columns.push({
      name: String(teamLabelFromId(teamId)),
      className: `match-summary-sheet__team match-summary-sheet__team--${teamKey(teamId)}`,
      valueOf: (metric) =>
        members.some((row) => !Number.isFinite(row[metric]))
          ? "—"
          : String(members.reduce((sum, row) => sum + Number(row[metric]), 0)),
    });
    for (const row of members) {
      const label = classIdentityFromId(Number(row.class_id))?.label ?? "Champion";
      const shown = displayIdOf(String(row.public_agent_id));
      columns.push({
        name: shown === null ? label : `${label} · ID ${shown}`,
        className: "match-summary-sheet__champion",
        valueOf: (metric) => (Number.isFinite(row[metric]) ? String(row[metric]) : "—"),
      });
    }
  }
  const table = htmlNode(doc, "table", { class: "match-summary-sheet__table" });
  const head = htmlNode(doc, "thead", {});
  const headRow = htmlNode(doc, "tr", {});
  headRow.append(htmlNode(doc, "td", {}));
  for (const column of columns) {
    headRow.append(
      htmlNode(doc, "th", { scope: "col", class: column.className }, column.name),
    );
  }
  head.append(headRow);
  table.append(head);
  const body = htmlNode(doc, "tbody", {});
  for (const label of labels) {
    const row = htmlNode(doc, "tr", {});
    row.append(htmlNode(doc, "th", { scope: "row" }, String(label.label)));
    for (const column of columns) {
      row.append(
        htmlNode(
          doc,
          "td",
          { class: column.className },
          column.valueOf(String(label.metric)),
        ),
      );
    }
    body.append(row);
  }
  table.append(body);
  sheet.append(table);
  return sheet;
}

/**
 * Return a new HTML element with the given attributes and, optionally, words.
 *
 * @param {Document} doc
 * @param {string} tagName
 * @param {Record<string, string | number>} attributes
 * @param {string} [words]
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
 * Move cards to new geometry (reprojection after resize or camera change).
 *
 * nodes are what build returned: [group, connector] with a mount (the cards
 * are the group's `.herald` children, in record order), or the card groups.
 * Each card moves to its record's x, y and scale, and a tier-1 card's Herald
 * Rule is redrawn across the record's band (tierRulePath, with a 1 CSS px
 * stroke at the new scale). Return undefined.
 *
 * @param {ReadonlyArray<Element | null>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  if (records.length === 0 || nodes.length === 0) {
    return;
  }
  const first = nodes[0];
  const cards =
    first && !first.classList.contains("herald")
      ? [...first.querySelectorAll(":scope > .herald")]
      : nodes;
  records.forEach((record, index) => {
    const card = cards[index];
    card?.setAttribute(
      "transform",
      `translate(${fmt(record.x)} ${fmt(record.y)}) scale(${fmt(record.scale)})`,
    );
    // The Herald Rule spans the band, so it is redrawn for the new rectangle.
    const rule = card?.querySelector?.(".herald__tier-rule") ?? null;
    if (rule !== null && isRecord(record.tierRule)) {
      rule.setAttribute("d", tierRulePath(record));
      rule.setAttribute(
        "stroke-width",
        fmt(1 / Math.max(0.01, Number(record.scale) || 1)),
      );
    }
  });
}

/**
 * Return the tooltip descriptor for a herald record, or null for anything else.
 *
 * Title is the card's title; the summary its eyebrow (or a plain line); a
 * Champion Streak card first adds a "Streak" row in plain words ("Mage · ID 3
 * took part in 5 takedowns without falling. Kill credit is shared."); a
 * "Fallen" row lists the falls the card names. Reads only the record.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  if (
    !isRecord(record) ||
    ![
      "herald-plate",
      "herald-band",
      "herald-lineup",
      "herald-verdict",
      "herald-verdict-plate",
    ].includes(String(record.kind))
  ) {
    return null;
  }
  const fallen = asArray(record.fallen);
  const streak = isRecord(record.streak) ? record.streak : null;
  return createSemanticDescriptor({
    kind: "event",
    id: `herald:${record.eventId}`,
    title: String(record.title),
    tone: "information",
    accent: "none",
    summary: record.eyebrow ? String(record.eyebrow) : "A moment of the match.",
    rows: [
      ...(streak === null
        ? []
        : [
            {
              label: "Streak",
              value: `${String(streak.title)} took part in ${Number(streak.takedowns)} takedowns without falling. Kill credit is shared.`,
              metadata: { compact: true, full: true },
            },
          ]),
      ...(fallen.length > 0
        ? [
            {
              label: "Fallen",
              value: fallen.join("; "),
              metadata: { compact: true, full: true },
            },
          ]
        : []),
    ],
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "pointer",
  });
}
