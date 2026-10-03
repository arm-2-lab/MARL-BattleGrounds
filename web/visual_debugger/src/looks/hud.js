/**
 * @file The HUD look (Arena Frames): the party frames of both teams and the
 * HUD's one Tab stop. The score strip beside them is painted by
 * `match-summary.js`.
 *
 * Live play (BattleClient and DevClient combat screen) draws the approved
 * vision page's scorebug instead (#scorebug in index.html): this file fills
 * its hosts and portraits (liveScorebugModel, pure; paintLiveScorebug, DOM),
 * and match-summary.js fills its scores, bars and middle. Each host shows its
 * crest, team name, the "Yours" tag (Team A, the researcher's side, unless
 * every Manual slot is on Team B) and a controller line from the launch's
 * combat configuration (liveControllers, liveControllerRoster,
 * controllerLine, yoursTeam): the names when they fit ("You + Recurrent
 * MAPPO"), else "You + 4 Policies"; long names cut with "…"; hovering the
 * line drops down the team's full list (controller-names.js, the one owner
 * of controller names). Each portrait is a button: the champion's own
 * arena token (Champions and Statuses looks, drawn at 13 px), a life bar
 * (Ivory, Cinder at 30% or less) and "You", "ID 5" or, for an enemy the
 * Agent POV team cannot see, "Unseen" with a dashed ring and no life. A click
 * focuses a seen champion (onSelect) or explains an unseen one (onRefuse).
 * The Arena Frames below are then not drawn.
 *
 * What it draws (plan P1, P12, P15, P21, P23): per team, Team A on the left,
 * a header (crest, "Team A" or "Team B", the policy name as a controller
 * line in the Replay Viewer, "Standing 4 of 5")
 * and one frame per champion. A frame has a sigil (team disc, class emblem and
 * the Ultimate Charge ring, full with four notches when Ready), "Agent N" over
 * the class name, a life bar with its number (whole from 10 up, one decimal
 * below 10; Cinder at 30% or less), and a tray of 2 status chips plus "+N".
 * A fallen frame shows "Returns In N" (the team's wave countdown + 1) and the
 * equal-size emblems of the champions credited with the fall, on the tick of
 * the fall only. A shielded frame shows "Spawn Shield N"; on its return tick
 * it shows "Returned" instead.
 *
 * Who calls it: the integrator mounts `createHud` once (main.js) and calls its
 * `render(hudSourcesFromFrame(frame), ...)` on every installed frame, filter
 * change and view change, BEFORE the choreographer plans the tick, so the
 * beat-held nodes exist when it reads the DOM. This module never reads a frame:
 * `hudSourcesFromFrame` in match-summary.js gathers its facts, so no import of
 * this look reaches the presentation adapter. The pure `records` is also used
 * by the Agent POV leak test.
 *
 * Beats (plan 4.2): the DOM always holds the END state. Values that must wait
 * come only from the caller's `beatView(frame, "end", {visualFilters})`
 * (choreography-plan.js), never from an earlier frame: a champion that falls
 * this tick gets a "before" node with its life from before the blow
 * (data-beat-hold="fall"); a champion that returns gets a "before" node with
 * its fallen look (data-beat-hold="return"); Standing changes with them. The
 * choreographer shows each "before" node until its beat. Statuses, Ready and
 * life changes without a fall show END at once (there is no Settle hold kind).
 *
 * Boundary (Agent POV, plan P23 and owner rules s5, s11):
 * - BattleClient and DevClient (product "combat_debugger"): every frame reads
 *   only the POV arena scene and the POV visual events. A champion the POV
 *   team cannot see (an enemy, or a NoSharedObs teammate) reads "Unseen" with
 *   no life, statuses, Ultimate, shield, wave or credit. To know which
 *   champions exist, its sources carry the identity directory's identity
 *   fields only (public ID, team, class, slot); never roster health or
 *   positions.
 * - Replay Viewer ("replay_viewer"): the frames are a docked researcher panel,
 *   so in Agent POV they list every champion's facts under a "Full Record"
 *   chip; every tip and the Champion Card still follow Agent POV: the card
 *   (tooltip.js, the same card the arena body and the live hover card show)
 *   is built from the POV scene row, and an unseen champion gets the reduced
 *   card ("Unseen · Team A cannot see it now.") while its chip tips say only
 *   "Your team cannot see this champion now".
 *   With the Seen And Not Seen switch on, each team lists seen champions
 *   first, then unseen, then fallen; off, ID order.
 * - Kill credit in Agent POV names only killers the team saw (the caller's
 *   fallen_feed_row list from the choreography plan); with no list, Agent POV
 *   names none. Oracle View may fall back to the match summary's deaths
 *   (sources.deaths, empty in Agent POV).
 * Oracle -> Agent POV -> Oracle is a fresh render: nothing is kept between
 * renders except which HUD symbol holds the one Tab stop and, in live play,
 * each party's last paint (partyPaintKey, pure). That paint only lets a
 * repaint that would draw the very same portraits keep its nodes (so a
 * running beat hold survives a repaint of the same tick); any change
 * rebuilds the party.
 *
 * Export: the HUD is page chrome. It is never inside the battlefield SVG, so
 * PNG exports never carry it.
 */
import {
  classIdentityFromId,
  fullRecordChipWords,
  teamLabelFromId,
} from "../agent-identity.js";
import { classNamesV1 } from "../class-documentation.js";
import {
  controllerLineParts,
  controllerOfChampion,
  controllerRoster,
  NO_CONTROLLER_ROSTER,
  paintControllerLine,
  publishControllerRoster,
  rosterTeam,
} from "../controller-names.js";
import {
  formatDisplayNumber,
  formatLifeNumber,
  LOW_LIFE_FRACTION,
} from "../display.js";
import { createSvgIcon } from "../icons.js";
import { createLifeBar, lifeBarModel } from "../life-bar.js";
import {
  statusCardSentence,
  statusPresentation,
  statusTimeText,
} from "../semantic-vocabulary.js";
import { statusOrderIndex, symbolById, symbolChipText } from "../symbols.js";
import {
  championCardFactsFromAgent,
  createChampionCardDescriptor,
  createSemanticDescriptor,
  hoverCardSubject,
  registerTooltipOwner,
} from "../tooltip.js";
import { classTokenFromId, resolveVisualToken } from "../vocabulary.js";
import * as champions from "./champions.js";
import { EMPTY_BUDGET, SVG_NAMESPACE } from "./look-support.js";
import * as statuses from "./statuses.js";

/**
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 */

/**
 * Sizes in CSS pixels (store spec "arena-frames"). The HUD is DOM chrome
 * outside the arena, so these never enter the arena's collision allocator.
 */
export const FOOTPRINT = Object.freeze({
  headerHeight: 24,
  frameHeight: 34,
  sigil: 28,
  sigilRingRadius: 12.5,
  sigilDiscRadius: 10.5,
  emblem: 16,
  nameColumn: 72,
  lifeBar: Object.freeze({ width: 150, height: 8 }),
  lifeNumberBox: 52,
  chip: Object.freeze({ width: 30, height: 18 }),
  creditEmblem: 14,
});

/** The HUD draws no arena nodes, so the plan's budget counts nothing for it. */
export const BUDGET = EMPTY_BUDGET;

/**
 * No arena paint parts: the HUD's switches (scorebug, team_rosters,
 * death_announcer, fighters_standing, respawn_wave, symbol_explanations) are
 * interface switches read directly from the filter state by `records`.
 *
 * @type {ReadonlyArray<import("./look-support.js").PaintPartRegistration>}
 */
export const PAINT_PARTS = Object.freeze([]);

/** Status chips shown in a frame's tray before "+N" (plan P15). */
export const TRAY_CHIP_LIMIT = 2;

/**
 * Life at or below this share of the maximum is drawn in Cinder: the shared
 * low-life warning line (LOW_LIFE_FRACTION of `display.js`, 30%), which the
 * Champion Card's red life bar uses too.
 */
export const CINDER_FRACTION = LOW_LIFE_FRACTION;

/** What an unseen champion's frame reads in the BattleClient's Agent POV. */
export const UNSEEN_LABEL = "Unseen";

/** The chip word over the Replay Viewer's frames in Agent POV. */
export const FULL_RECORD_LABEL = "Full Record";

/** What a tip says about a champion the Agent POV team cannot see now. */
export const LIMITED_TIP_TEXT = "Your team cannot see this champion now.";

const COMPACT_AND_FULL = Object.freeze({ compact: true, full: true });

/**
 * Format a life value the way the HUD prints it: the shared life number rule
 * of `display.js` (whole and rounded down from 10 up, one decimal rounded up
 * below 10), re-exported here for the HUD's callers.
 */
export { formatLifeNumber };

/**
 * Build a life record from current and maximum life.
 *
 * Return {current, maximum, fraction (0 to 1), text (formatLifeNumber),
 * low (true at CINDER_FRACTION or less)}, frozen, or null when current is not
 * a finite number. maximum may be missing; the fraction is then 1 for any life
 * above 0. Pure.
 *
 * @param {unknown} current
 * @param {unknown} maximum
 */
function lifeRecord(current, maximum) {
  if (typeof current !== "number" || !Number.isFinite(current)) return null;
  const max = typeof maximum === "number" && maximum > 0 ? maximum : null;
  const fraction =
    max === null ? (current > 0 ? 1 : 0) : Math.max(0, Math.min(1, current / max));
  return Object.freeze({
    current,
    maximum: max,
    fraction,
    text: formatLifeNumber(Math.max(0, current)),
    low: fraction <= CINDER_FRACTION,
  });
}

/**
 * The exact life words for tips: "153.4 of 200", or "153.4" with no maximum
 * (up to two decimals, display.js). Pure.
 *
 * @param {Readonly<Record<string, any>>} life See lifeRecord.
 * @returns {string}
 */
function exactLife(life) {
  const current = formatDisplayNumber(life.current);
  return life.maximum === null
    ? current
    : `${current} of ${formatDisplayNumber(life.maximum)}`;
}

/**
 * Read a record's own value at key, or undefined for a non-object. Pure.
 *
 * @param {unknown} value
 * @param {string} key
 * @returns {any}
 */
function field(value, key) {
  return value !== null && typeof value === "object"
    ? /** @type {Record<string, any>} */ (value)[key]
    : undefined;
}

/**
 * @typedef {{
 *   sources?: HudSources | null,
 *   beat?: Readonly<Record<string, any>> | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   falls?: ReadonlyArray<Readonly<Record<string, any>>> | null,
 * }} HudInput
 *   sources: the frame's facts from hudSourcesFromFrame (match-summary.js);
 *   null or absent draws nothing.
 *   beat: `beatView(frame, "end", {visualFilters})` from choreography-plan.js,
 *   or null/absent for no holds.
 *   filters: the visual filter state; a missing switch counts as on (every
 *   switch read here is on by default). Switches read: team_rosters,
 *   fighters_standing, death_announcer, respawn_wave, symbol_explanations and
 *   seen_grouping (Replay Viewer Agent POV order only).
 *   falls: the choreography plan's fallen_feed_row derived events for this
 *   frame ({victim: {publicAgentId}, killers: [{publicAgentId, classId,
 *   teamId, title}]}), already limited to what the audience saw. Absent:
 *   Oracle View uses sources.deaths; Agent POV names nobody.
 */

/**
 * @typedef {{
 *   product: "replay_viewer" | "combat_debugger",
 *   audience: "researcher" | "agent_pov",
 *   sceneAgents: ReadonlyArray<Record<string, any>>,
 *   researcherAgents: ReadonlyArray<Record<string, any>> | null,
 *   identities: ReadonlyArray<Readonly<{publicAgentId: string, teamId: 1 | 2, teamLocalSlot: number, classId: number | null}>>,
 *   spawnShieldMechanics?: Record<string, any> | null,
 *   mechanics: ReadonlyArray<Record<string, any>>,
 *   waves: ReadonlyArray<Record<string, any>>,
 *   visualEvents: ReadonlyArray<Record<string, any>>,
 *   teams: ReadonlyArray<Record<string, any>>,
 *   deaths: ReadonlyArray<Record<string, any>>,
 *   controlledKey: string | null,
 *   recipientPublicAgentId: string | null,
 *   returnedFallback: ReadonlyArray<string>,
 *   trapBreakIds?: ReadonlyArray<string>,
 *   displayIdOf: (publicAgentId: string) => string | null,
 *   controllerRoster?: import("../controller-names.js").ControllerRoster | null,
 * }} HudSources
 *   Everything the HUD reads, gathered from one frame by
 *   hudSourcesFromFrame in match-summary.js.
 *   sceneAgents: the audience's arena scene agents (Agent POV: only seen
 *   bodies). researcherAgents: the Replay Viewer's full record in Agent POV,
 *   else null. identities: identity fields of every champion in the
 *   BattleClient and DevClient (else []), used for the "Unseen" rows in Agent
 *   POV and, in both views, for each champion's command slot in the live
 *   scorebug. spawnShieldMechanics: the scene's Spawn Shield mechanics, for
 *   the live portraits' ward and the Champion Card's Spawn Shield drain line
 *   (null or absent when the scene has none). mechanics: class mechanics (Ultimate
 *   cooldown steps). waves: scene.respawn_waves (public wave clocks).
 *   visualEvents: Agent POV visual_events.events, else []. teams and deaths:
 *   match_summary.teams and match_summary.deaths (deaths only in Oracle View,
 *   else []). controlledKey: the scene's controlled presentation key
 *   (BattleClient and DevClient: the champion whose orders the player sees,
 *   or a policy's champion the player looks at), else null.
 *   recipientPublicAgentId: the
 *   Agent POV recipient, else null. returnedFallback: champions returning
 *   this tick by the adapter's beat facts, used only without a beat view.
 *   trapBreakIds: public IDs whose Freezing Trap broke this tick in the
 *   audience's own events (the Champion Card's Freezing Trap (Stun Broken)
 *   row); absent means none. displayIdOf: the shown ID ("0" to "9") for a
 *   public ID, or null. controllerRoster: the Replay Viewer's one controller
 *   model (controllerRosterFromFrame in match-summary.js: each champion's
 *   recorded controller name, or each team's recorded name for old
 *   replays, with its champions), which names the Team Rosters strip's
 *   teams and the Champion Card's "played by"; null or absent in live play,
 *   where the scorebug builds the model from the launch's combat
 *   configuration instead.
 */

/**
 * Return the HUD's drawing records. Pure data, no DOM.
 *
 * input.sources comes from hudSourcesFromFrame (match-summary.js), the one
 * place the HUD's facts are read from a frame; this module never reads a
 * frame, so the Agent POV leak test can check it on its own.
 *
 * Rows: Oracle View draws every scene agent. The Replay Viewer in Agent POV
 * draws its full record (researcherAgents), marking which ones the team sees.
 * The BattleClient and DevClient in Agent POV draw the seen scene agents plus
 * one identity-only "unseen" row for every other champion in identities.
 *
 * Records, in display order:
 * 1. one `hud_view` record: {product, audience ("researcher" Oracle View or
 *    "agent_pov"), fullRecord (true in the Replay Viewer's Agent POV),
 *    fullRecordLabel ("Full Record · Not Limited To Agent 0's View", where
 *    "Agent 0" is the POV agent's own champion chipText, so the chip and the
 *    strip always agree; "...Not Limited To This Agent's View" when the POV
 *    agent is unknown or not in the rows; null when fullRecord is false),
 *    rosters (team_rosters switch), tips (symbol_explanations switch)};
 * 2. per team, Team A first: one `team_header` record {teamId, side ("left"
 *    or "right"), label, name, controllers (the team in
 *    sources.controllerRoster, or null when absent), standing {alive,
 *    total} | null (null when the
 *    switch is off or the team has an unseen champion), standingHold {before,
 *    kind "fall" | "return", beatKey} | null}, followed by its `champion`
 *    records in shown-ID order. Exception: in the Replay Viewer's Agent POV
 *    with the seen_grouping switch on (the default), each team lists the
 *    living champions its POV team sees first, then those it does not see,
 *    then the fallen, each group in shown-ID order (see seenGroupRank).
 * A `champion` record has: teamId, side, publicAgentId, displayId, chipText
 * ("Agent 5"), classId, className, calling, classToken ("mage" ...), state
 * ("alive", "fallen" or "unseen"), seen (in the audience's arena scene), you
 * (the champion you order, BattleClient and DevClient only), selectable,
 * life (see lifeRecord) | null, ultimate {ready, fraction | null, remaining}
 * | null, tray {chips, more, moreLines, iconStyle} | null, spawnShield (ticks) | null,
 * returned (true on its return tick, with Respawn Wave on), returnsIn (ticks)
 * | null, credits (killers' {publicAgentId, classId, teamId, label}, only on
 * the tick of the fall, with Fallen Feed on), fallHold {beatKey, life} |
 * null, returnHold {beatKey, returnsIn} | null, tipLimited and card (the
 * Champion Card facts of tooltip.js, ChampionCardFacts: read from the
 * audience's own scene row through championCardFactsFromAgent, so it is the
 * card the arena body shows; the reduced "unseen" facts when the audience
 * cannot see the champion now).
 * An "unseen" record carries identity only: every fact field is null or
 * empty. Return a frozen array; only the hud_view record when there are no
 * champions.
 *
 * @param {HudInput} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const sources = input?.sources ?? null;
  if (sources === null) return Object.freeze([]);
  const options = input;
  const { product, audience, sceneAgents, mechanics } = sources;
  const filters = options.filters ?? {};
  const on = (/** @type {string} */ id) => filters[id] !== false;
  const beat = options.beat ?? null;
  const sceneById = new Map(sceneAgents.map((agent) => [agent.public_agent_id, agent]));
  const researcherAgents = audience === "agent_pov" ? sources.researcherAgents : null;
  // The Agent POV team, for the reduced card's "Team A cannot see it now."
  const povTeamId =
    audience !== "agent_pov" || sources.recipientPublicAgentId === null
      ? null
      : ([...sceneAgents, ...(researcherAgents ?? [])].find(
          (agent) => agent?.public_agent_id === sources.recipientPublicAgentId,
        )?.team_id ?? null);
  const povTeam =
    povTeamId === 1 || povTeamId === 2 ? teamLabelFromId(povTeamId) : null;
  /**
   * Who plays a champion, for the Champion Card: in the Replay Viewer its
   * recorded controller from the one controller model (its own name when
   * the replay records per-champion names, else its team's name), falling
   * back to the team name from the match summary; null in live play (its
   * card reads slot labels).
   *
   * @param {number} teamId
   * @param {string} publicAgentId
   * @returns {string | null}
   */
  const playedBy = (teamId, publicAgentId) => {
    if (product !== "replay_viewer") return null;
    const name =
      controllerOfChampion(sources.controllerRoster ?? NO_CONTROLLER_ROSTER, {
        publicAgentId,
      })?.controller.name ?? sources.teams[teamId - 1]?.display_name;
    return typeof name === "string" && name !== "" ? name : null;
  };

  /** @type {Array<{agent: Record<string, any> | null, identity: Record<string, any> | null, seen: boolean}>} */
  let rows;
  if (audience === "researcher") {
    rows = sceneAgents.map((agent) => ({ agent, identity: null, seen: true }));
  } else if (researcherAgents !== null) {
    rows = researcherAgents.map((agent) => ({
      agent,
      identity: null,
      seen: sceneById.has(agent.public_agent_id),
    }));
  } else {
    rows = sceneAgents.map((agent) => ({ agent, identity: null, seen: true }));
    for (const identity of sources.identities) {
      if (!sceneById.has(identity.publicAgentId)) {
        rows.push({ agent: null, identity, seen: false });
      }
    }
  }

  const youId =
    product !== "combat_debugger"
      ? null
      : (sceneAgents.find((agent) => agent.presentation_key === sources.controlledKey)
          ?.public_agent_id ?? sources.recipientPublicAgentId);
  const returnedIds = new Set(
    beat !== null
      ? (Array.isArray(beat.returns) ? beat.returns : []).map(
          (/** @type {any} */ entry) => entry.publicAgentId,
        )
      : sources.returnedFallback,
  );

  /** @type {LookRecord[]} */
  const champions = rows.map((row) => {
    const agent = row.agent;
    const publicAgentId = String(agent?.public_agent_id ?? row.identity?.publicAgentId);
    const teamId = Number(agent?.team_id ?? row.identity?.teamId);
    const classId = Number(agent?.class_id ?? row.identity?.classId);
    const displayId =
      agent?.display_agent_id ?? sources.displayIdOf(publicAgentId) ?? publicAgentId;
    const names = classNamesV1(classId);
    const classIdentity = classIdentityFromId(classId);
    const base = {
      kind: "champion",
      teamId,
      side: teamId === 1 ? "left" : "right",
      publicAgentId,
      displayId: String(displayId),
      chipText: `Agent ${displayId}`,
      classId: classIdentity?.classId ?? null,
      className: names?.className ?? classIdentity?.label ?? "Unknown Class",
      calling: names?.calling ?? null,
      classToken: classIdentity?.tokenId ?? "unknown",
      seen: row.seen,
      you: youId !== null && publicAgentId === youId,
    };
    if (agent === null) {
      return Object.freeze({
        ...base,
        state: "unseen",
        selectable: false,
        life: null,
        ultimate: null,
        tray: null,
        spawnShield: null,
        returned: false,
        returnsIn: null,
        credits: Object.freeze([]),
        fallHold: null,
        returnHold: null,
        tipLimited: true,
        card: unseenCardFacts(base, povTeam, playedBy(teamId, publicAgentId)),
      });
    }
    const alive = agent.alive === true || agent.life_state === "alive";
    const maximum = agent.maximum_health ?? agent.max_health;
    const life = lifeRecord(alive ? agent.current_health : 0, maximum);
    const steps = mechanics.find(
      (entry) => entry?.class_id === classId,
    )?.ultimate_cooldown_steps;
    const remaining = agent.ultimate_cooldown_remaining ?? agent.ultimate_cooldown;
    const ultimate =
      alive && Number.isInteger(remaining)
        ? Object.freeze({
            ready: remaining === 0,
            remaining,
            fraction:
              Number.isInteger(steps) && steps > 0
                ? Math.max(0, Math.min(1, 1 - remaining / steps))
                : null,
          })
        : null;
    const lifeEntry = (Array.isArray(beat?.life) ? beat.life : []).find(
      (/** @type {any} */ entry) => entry?.publicAgentId === publicAgentId,
    );
    const fallHold =
      !alive && lifeEntry !== undefined && typeof lifeEntry.before === "number"
        ? Object.freeze({
            beatKey: String(lifeEntry.presentationKey),
            life: lifeRecord(lifeEntry.before, maximum),
          })
        : null;
    const returnEntry = (
      Array.isArray(beat?.schedule?.returns) ? beat.schedule.returns : []
    ).find((/** @type {any} */ entry) => entry?.publicAgentId === publicAgentId);
    const returned = alive && returnedIds.has(publicAgentId);
    const returnHold =
      returned && returnEntry !== undefined
        ? Object.freeze({ beatKey: String(returnEntry.presentationKey), returnsIn: 1 })
        : null;
    const countdown = sources.waves.find(
      (wave) => wave?.team_id === teamId,
    )?.countdown_steps;
    const shield = agent.spawn_shield_remaining;
    const credits =
      !alive && on("death_announcer")
        ? creditsFor(sources, publicAgentId, options.falls ?? null)
        : Object.freeze([]);
    const cardAgent =
      audience === "agent_pov" ? (sceneById.get(publicAgentId) ?? null) : agent;
    const tray = alive
      ? trayRecord(agent.statuses, {
          cardAgent,
          iconStyle: on("status_marks_icons") ? "sigil" : "outline",
        })
      : null;
    const record = {
      ...base,
      state: alive ? "alive" : "fallen",
      selectable: true,
      life,
      ultimate,
      tray,
      spawnShield: alive && Number.isInteger(shield) && shield > 0 ? shield : null,
      returned: returned && on("respawn_wave"),
      returnsIn: !alive && Number.isInteger(countdown) ? countdown + 1 : null,
      credits,
      fallHold,
      returnHold,
      tipLimited: audience === "agent_pov" && cardAgent === null,
    };
    return Object.freeze({
      ...record,
      card:
        cardAgent === null
          ? unseenCardFacts(base, povTeam, playedBy(teamId, publicAgentId))
          : championCardFactsFromAgent(cardAgent, {
              classMechanics: mechanics,
              spawnShieldSteps:
                sources.spawnShieldMechanics?.configured_duration_steps ?? null,
              trapBroke: (sources.trapBreakIds ?? []).includes(publicAgentId),
              // The card writes its own "Fallen · Returns In N" from the
              // same public wave clock as returnsIn above.
              respawnWaves: sources.waves,
              you: base.you,
              playedBy: base.you ? null : playedBy(teamId, publicAgentId),
            }),
    });
  });

  const grouped =
    product === "replay_viewer" && audience === "agent_pov" && on("seen_grouping");
  champions.sort(
    (a, b) =>
      a.teamId - b.teamId ||
      (grouped ? seenGroupRank(a) - seenGroupRank(b) : 0) ||
      Number(a.displayId) - Number(b.displayId) ||
      String(a.publicAgentId).localeCompare(String(b.publicAgentId)),
  );
  const recipientChip =
    sources.recipientPublicAgentId === null
      ? null
      : (champions.find(
          (champion) => champion.publicAgentId === sources.recipientPublicAgentId,
        )?.chipText ?? null);
  /** @type {LookRecord[]} */
  const output = [
    Object.freeze({
      kind: "hud_view",
      product,
      audience,
      fullRecord: researcherAgents !== null,
      fullRecordLabel:
        researcherAgents === null ? null : fullRecordChipWords(recipientChip),
      rosters: on("team_rosters"),
      tips: on("symbol_explanations"),
    }),
  ];
  for (const teamId of [1, 2]) {
    const members = champions.filter((champion) => champion.teamId === teamId);
    if (members.length === 0) continue;
    const anyUnseen = members.some((member) => member.state === "unseen");
    const alive = members.filter((member) => member.state === "alive").length;
    const falls = members.filter((member) => member.fallHold !== null);
    const returns = members.filter((member) => member.returnHold !== null);
    const before = alive + falls.length - returns.length;
    const holdSource = falls[0]?.fallHold ?? returns[0]?.returnHold ?? null;
    const standing =
      on("fighters_standing") && !anyUnseen
        ? Object.freeze({ alive, total: members.length })
        : null;
    const team = sources.teams[teamId - 1];
    output.push(
      Object.freeze({
        kind: "team_header",
        teamId,
        side: teamId === 1 ? "left" : "right",
        label: teamLabelFromId(teamId) ?? `Team ${teamId}`,
        name: typeof team?.display_name === "string" ? team.display_name : null,
        controllers:
          sources.controllerRoster === null || sources.controllerRoster === undefined
            ? null
            : rosterTeam(sources.controllerRoster, teamId),
        standing,
        standingHold:
          standing !== null && holdSource !== null && before !== alive
            ? Object.freeze({
                before: Object.freeze({ alive: before, total: members.length }),
                kind: falls.length > 0 ? "fall" : "return",
                beatKey: holdSource.beatKey,
              })
            : null,
      }),
      ...members,
    );
  }
  return Object.freeze(output);
}

/**
 * Name the champions credited with a fall this tick, for a fallen frame.
 *
 * falls (the plan's fallen_feed_row rows, already limited to the audience)
 * wins when given. Without it, sources.deaths (match_summary.deaths, filled in
 * Oracle View only) gives the victim's contributors; Agent POV names nobody.
 * Return a frozen list of {publicAgentId, classId, teamId, label ("Hunter ·
 * ID 7")}, in the given order. Pure.
 *
 * @param {HudSources} sources
 * @param {string} victimId
 * @param {ReadonlyArray<Readonly<Record<string, any>>> | null} falls
 */
function creditsFor(sources, victimId, falls) {
  /** @type {ReadonlyArray<any>} */
  let killers = [];
  if (Array.isArray(falls)) {
    const row = falls.find(
      (entry) => field(field(entry, "victim"), "publicAgentId") === victimId,
    );
    killers = Array.isArray(row?.killers) ? row.killers : [];
  } else if (sources.audience === "researcher") {
    const death = sources.deaths.find((entry) => entry?.public_agent_id === victimId);
    killers = (Array.isArray(death?.contributors) ? death.contributors : []).map(
      (/** @type {any} */ contributor) => ({
        publicAgentId: contributor.public_agent_id,
        classId: contributor.class_id,
        teamId: contributor.team_id,
      }),
    );
  }
  return Object.freeze(
    killers.flatMap((killer) => {
      const publicAgentId = field(killer, "publicAgentId");
      const classId = field(killer, "classId");
      const teamId = field(killer, "teamId");
      if (typeof publicAgentId !== "string") return [];
      const shown = sources.displayIdOf(publicAgentId) ?? publicAgentId;
      const label =
        typeof field(killer, "title") === "string"
          ? field(killer, "title")
          : `${classIdentityFromId(classId)?.label ?? "Unknown"} · ID ${shown}`;
      return [
        Object.freeze({
          publicAgentId,
          classId: Number.isInteger(classId) ? classId : null,
          teamId: teamId === 1 || teamId === 2 ? teamId : null,
          label,
        }),
      ];
    }),
  );
}

/**
 * Build a frame's status tray from its durable statuses.
 *
 * statuses is the drawn agent's normalized status list (token_id holds the
 * catalog status ID). In Combat is left out. Chips are ordered as the
 * Champion Card orders its rows (`symbols.js` statusOrderIndex: stuns first,
 * then slows, then the rest), then by more ticks left. Each chip: {tokenId
 * (display token), name and title (the approved name, "Serrated Arrows
 * (Slow)"), glyphKey, cssKey, count (ticks left) | null, colorClass (the
 * causer's class token, or "rime" for a Freezing Trap stun), time ("3 ticks
 * left", or "" without a count) and line (the Champion Card's second line:
 * what it does with its number, then where it comes from by class, from
 * statusCardSentence)}. The tray keeps the first TRAY_CHIP_LIMIT chips; more
 * is how many are left out and moreLines their full lines ("Charge (Slow) · 3
 * ticks left · It moves at ×0.5 speed. From a Warrior."). iconStyle is
 * context.iconStyle: "sigil" (default; the chip glyph is a small copy of the
 * mark on the body) or "outline" (the Status Icons From Marks switch,
 * status_marks_icons, is off: the same mark drawn as thin lines, never
 * today's old glyph). Return null for no statuses.
 *
 * Sources are named by class only (owner, 2 October 2026), so Agent POV and
 * Oracle View read the same words. When context.cardAgent is null (an Agent
 * POV champion the team cannot see) the line is the plain name only: its tips
 * then say only LIMITED_TIP_TEXT. Pure.
 *
 * @param {unknown} statuses
 * @param {{cardAgent: Record<string, any> | null, iconStyle?: "sigil" | "outline"}} context
 */
function trayRecord(statuses, context) {
  const list = (Array.isArray(statuses) ? statuses : []).flatMap((status) => {
    const token = resolveVisualToken("status", field(status, "token_id"));
    if (token.tokenId === "in_combat") return [];
    const profile = statusPresentation(token.tokenId);
    const count = field(status, "remaining_duration");
    const ticks = Number.isInteger(count) ? /** @type {number} */ (count) : null;
    return [
      Object.freeze({
        tokenId: token.tokenId,
        name: token.shortLabel,
        title: token.label,
        glyphKey: token.glyphKey,
        cssKey: token.cssKey,
        count: ticks,
        colorClass:
          token.tokenId === "stun_hunter_trap"
            ? "rime"
            : (classIdentityFromId(profile.sourceClassId)?.tokenId ?? "neutral"),
        time: ticks === null ? "" : `${ticks} ${ticks === 1 ? "tick" : "ticks"} left`,
        line:
          context.cardAgent === null
            ? token.label
            : statusCardSentence(token.tokenId, field(status, "magnitude")).line,
      }),
    ];
  });
  if (list.length === 0) return null;
  list.sort(
    (a, b) =>
      statusOrderIndex(a.tokenId) - statusOrderIndex(b.tokenId) ||
      (b.count ?? 0) - (a.count ?? 0),
  );
  return Object.freeze({
    chips: Object.freeze(list.slice(0, TRAY_CHIP_LIMIT)),
    more: Math.max(0, list.length - TRAY_CHIP_LIMIT),
    moreLines: Object.freeze(
      list
        .slice(TRAY_CHIP_LIMIT)
        .map((chip) =>
          [chip.title, chip.time, chip.line]
            .filter(
              (part, index) => part !== "" && (index !== 2 || part !== chip.title),
            )
            .join(" · "),
        ),
    ),
    iconStyle: context.iconStyle === "outline" ? "outline" : "sigil",
  });
}

/**
 * Return a champion's place in the Seen And Not Seen grouping (the
 * seen_grouping switch, Replay Viewer Agent POV only): 0 for a living
 * champion the POV team sees, 1 for one it does not see, 2 for a fallen
 * champion. records() sorts by team, then this rank, then shown ID. Agent POV:
 * it reads only the record's state and seen fields, which say what the POV
 * team saw, so the order shows nothing beyond the strip itself. Pure.
 *
 * @param {LookRecord} record A `champion` record built in records().
 * @returns {0 | 1 | 2}
 */
function seenGroupRank(record) {
  if (record.state === "fallen") return 2;
  return record.seen === true ? 0 : 1;
}

/**
 * The reduced Champion Card facts of a champion the audience cannot see now
 * (Replay Viewer Agent POV): its class name and calling, shown ID, team and
 * who plays it, with "Unseen · Team A cannot see it now." (unseenBy is the
 * Agent POV team's label, or null for "none of your team"). No life,
 * Ultimate, status, source or credit. Pure.
 *
 * @param {{className: string, calling: string | null, classId: number | null, displayId: string, teamId: number}} base
 * @param {string | null} unseenBy The Agent POV team's label, or null.
 * @param {string | null} playedBy Who plays it, or null.
 * @returns {import("../tooltip.js").ChampionCardFacts}
 */
function unseenCardFacts(base, unseenBy, playedBy) {
  return Object.freeze({
    state: /** @type {const} */ ("unseen"),
    name: base.className,
    calling: base.calling,
    classId: /** @type {1 | 2 | 3 | 4 | 5 | null} */ (
      [1, 2, 3, 4, 5].includes(base.classId ?? 0) ? base.classId : null
    ),
    displayId: base.displayId,
    teamId: base.teamId === 1 || base.teamId === 2 ? base.teamId : null,
    life: null,
    ultimate: null,
    statuses: Object.freeze([]),
    note: null,
    you: false,
    playedBy,
    yourTarget: false,
    unseenBy,
  });
}

/**
 * Return the Champion Card tip descriptor for a champion record, or null for
 * any other record.
 *
 * The descriptor is tooltip.js createChampionCardDescriptor of the record's
 * card facts, so the shared tooltip controller draws the same Champion Card
 * as the arena body and the live hover card. In Agent POV a champion the team
 * cannot see gets the reduced card (name, calling, crest, ID, "Unseen").
 * Creates a frozen descriptor; no DOM.
 *
 * @param {LookRecord} record
 * @returns {ReturnType<typeof createChampionCardDescriptor> | null}
 */
export function tooltip(record) {
  if (record?.kind !== "champion") return null;
  return createChampionCardDescriptor(
    record.card,
    `champion:frame:${record.publicAgentId}`,
  );
}

/**
 * Make one frozen HUD tip descriptor. accent is a class token or "none".
 *
 * @param {string} id
 * @param {string} title
 * @param {string} summary
 * @param {ReadonlyArray<ReadonlyArray<string>>} rows Each row is [label, value].
 * @param {string} [accent] Defaults to "none".
 */
function descriptor(id, title, summary, rows, accent = "none") {
  return createSemanticDescriptor({
    kind: "hud",
    id: `hud:${id}`,
    title,
    tone: "information",
    accent: ["mage", "warrior", "hunter", "rogue", "priest"].includes(accent)
      ? accent
      : "none",
    summary,
    rows: rows.map((row) => ({
      label: row[0],
      value: row[1],
      metadata: COMPACT_AND_FULL,
    })),
    sections: [],
    metadata: COMPACT_AND_FULL,
    anchor: "element",
  });
}

/**
 * Make node a HUD symbol (data-hud-symbol, tabindex -1) and, when tips is
 * true, register desc as its shared tip. Changes node and the tooltip
 * registry only.
 *
 * @param {Element} node
 * @param {string} id
 * @param {boolean} tips
 * @param {() => ReturnType<typeof createSemanticDescriptor> | null} desc
 */
function symbol(node, id, tips, desc) {
  node.setAttribute("data-hud-symbol", id);
  node.setAttribute("tabindex", "-1");
  if (!tips) return;
  const made = desc();
  if (made !== null) registerTooltipOwner(node, made);
}

/**
 * Create an element with a class name and optional text. Attaches nothing.
 *
 * @param {Document} doc
 * @param {string} tag
 * @param {string} className
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function el(doc, tag, className, text) {
  const node = doc.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Mark node as a beat "before" node (choreography.js contract): it shows the
 * value before the beat and is drawn at its END opacity 0. Hidden from
 * screen readers. Changes node only.
 *
 * @param {HTMLElement} node
 * @param {"fall" | "return"} kind
 * @param {string} beatKey The champion's presentation key in the audience's events.
 */
function holdBefore(node, kind, beatKey) {
  node.setAttribute("data-beat-hold", kind);
  node.setAttribute("data-beat-key", beatKey);
  node.setAttribute("data-beat-side", "before");
  node.setAttribute("opacity", "0");
  node.style.opacity = "0";
  node.setAttribute("aria-hidden", "true");
}

/**
 * Mark node as the beat "after" node: the END value, faded in at the beat.
 *
 * @param {HTMLElement} node
 * @param {"fall" | "return"} kind
 * @param {string} beatKey
 */
function holdAfter(node, kind, beatKey) {
  node.setAttribute("data-beat-hold", kind);
  node.setAttribute("data-beat-key", beatKey);
  node.setAttribute("data-beat-side", "after");
}

/**
 * Draw a sigil: the team disc, the Ultimate Charge ring and the class emblem.
 *
 * state "alive" draws the charge arc (fraction of the ring at 70% opacity over
 * a 12% track) or, when Ready, the full ring with four inward notches;
 * "fallen" draws a Grave Stone disc and an Ash ring; "unseen" a hollow ring
 * at 12% and a dimmed emblem. Export-safe SVG (flat fills and strokes only).
 * Returns a new span holding the ring SVG and the emblem SVG.
 *
 * @param {Document} doc
 * @param {Record<string, any>} record A champion record.
 * @param {"alive" | "fallen" | "unseen"} state
 * @returns {HTMLElement}
 */
function sigil(doc, record, state) {
  const box = el(doc, "span", `arena-frame__sigil arena-frame__sigil--${state}`);
  const svg = doc.createElementNS(SVG_NAMESPACE, "svg");
  svg.setAttribute("viewBox", "0 0 28 28");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "arena-frame__ring");
  const circle = (/** @type {number} */ r, /** @type {string} */ className) => {
    const node = doc.createElementNS(SVG_NAMESPACE, "circle");
    node.setAttribute("cx", "14");
    node.setAttribute("cy", "14");
    node.setAttribute("r", String(r));
    node.setAttribute("class", className);
    svg.append(node);
    return node;
  };
  circle(FOOTPRINT.sigilDiscRadius, "arena-frame__disc");
  circle(FOOTPRINT.sigilRingRadius, "arena-frame__track");
  const ultimate = record.ultimate;
  if (state === "alive" && ultimate !== null) {
    if (ultimate.ready) {
      circle(FOOTPRINT.sigilRingRadius, "arena-frame__charge is-ready");
      for (const angle of [0, 90, 180, 270]) {
        const notch = doc.createElementNS(SVG_NAMESPACE, "line");
        notch.setAttribute("x1", "14");
        notch.setAttribute("y1", "1.5");
        notch.setAttribute("x2", "14");
        notch.setAttribute("y2", "4.5");
        notch.setAttribute("transform", `rotate(${angle} 14 14)`);
        notch.setAttribute("class", "arena-frame__notch");
        svg.append(notch);
      }
    } else if (ultimate.fraction !== null && ultimate.fraction > 0) {
      const arc = circle(FOOTPRINT.sigilRingRadius, "arena-frame__charge");
      const length = 2 * Math.PI * FOOTPRINT.sigilRingRadius;
      arc.setAttribute(
        "stroke-dasharray",
        `${(length * ultimate.fraction).toFixed(2)} ${length.toFixed(2)}`,
      );
      arc.setAttribute("transform", "rotate(-90 14 14)");
    }
  }
  box.append(
    svg,
    createSvgIcon(doc, `class-${record.classToken}`, {
      className: "arena-frame__emblem",
    }),
  );
  return box;
}

/**
 * Draw a frame's life as the one shared Life bar (`life-bar.js`: "Life", a
 * thin bar colored green, yellow or red, and "77 / 80"), in a
 * `span.arena-frame__life` holder. A life with no maximum counts its own
 * value as full. The Team Rosters strip hides the label and number for room
 * (styles.css); the frame's tip carries the exact life.
 *
 * @param {Document} doc
 * @param {Readonly<Record<string, any>>} life See lifeRecord.
 * @returns {HTMLElement}
 */
function lifeBar(doc, life) {
  const box = el(doc, "span", "arena-frame__life");
  // lifeRecord keeps only a finite current life, so the model always exists.
  const model = lifeBarModel(life.current, life.maximum ?? Math.max(life.current, 0));
  if (model !== null) box.append(createLifeBar(doc, model));
  return box;
}

/**
 * Draw a fallen frame's body: "Returns In N" and the credited emblems joined
 * by "+".
 *
 * @param {Document} doc
 * @param {Record<string, any>} record
 * @param {number | null} returnsIn
 * @param {boolean} tips
 * @param {boolean} withCredits False for the "before" copy of a return.
 * @returns {HTMLElement}
 */
function fallenBody(doc, record, returnsIn, tips, withCredits) {
  const body = el(doc, "span", "arena-frame__fallen");
  const text = el(
    doc,
    "span",
    "arena-frame__returns-in",
    returnsIn === null ? "Fallen" : `Returns In ${returnsIn}`,
  );
  body.append(text);
  if (withCredits) {
    symbol(text, `returns:${record.publicAgentId}`, tips, () =>
      descriptor(
        `returns:${record.publicAgentId}`,
        symbolById("returns_in")?.title ?? "Returns In",
        symbolById("returns_in")?.eli5 ?? "",
        [
          [
            "Returns In",
            returnsIn === null
              ? "Unknown"
              : `${returnsIn} ${returnsIn === 1 ? "Tick" : "Ticks"}`,
          ],
          ...(record.credits.length > 0
            ? [
                [
                  "Felled By",
                  record.credits.map((/** @type {any} */ c) => c.label).join(" + "),
                ],
              ]
            : []),
        ],
      ),
    );
    record.credits.forEach((/** @type {any} */ credit, /** @type {number} */ index) => {
      if (index > 0) body.append(el(doc, "span", "arena-frame__credit-join", "+"));
      const token = classIdentityFromId(credit.classId)?.tokenId ?? "unknown";
      const emblem = createSvgIcon(doc, `class-${token}`, {
        className: `arena-frame__credit arena-frame__credit--team-${credit.teamId === 2 ? "b" : "a"}`,
        accessibleName: credit.label,
      });
      body.append(emblem);
    });
  }
  return body;
}

/**
 * Draw a living frame's body: life bar, tray chips, "+N", and the Spawn Shield
 * or Returned chip.
 *
 * @param {Document} doc
 * @param {Record<string, any>} record
 * @param {boolean} tips
 * @param {boolean} interactive False for "before" copies (no symbols or tips).
 * @param {Readonly<Record<string, any>> | null} [lifeOverride] Life to show instead.
 * @returns {HTMLElement}
 */
function aliveBody(doc, record, tips, interactive, lifeOverride = null) {
  const body = el(doc, "span", "arena-frame__alive");
  const life = lifeOverride ?? record.life;
  if (life !== null) {
    const bar = lifeBar(doc, life);
    if (interactive) {
      symbol(bar, `life:${record.publicAgentId}`, tips, () =>
        record.tipLimited
          ? descriptor(`life:${record.publicAgentId}`, "Life", LIMITED_TIP_TEXT, [])
          : descriptor(
              `life:${record.publicAgentId}`,
              symbolById("life_bar")?.title ?? "Life",
              symbolById("life_bar")?.eli5 ?? "",
              [["Life", exactLife(life)]],
            ),
      );
    }
    body.append(bar);
  }
  if (lifeOverride !== null) return body;
  const tray = el(doc, "span", "arena-frame__tray");
  (record.tray?.chips ?? []).forEach(
    (/** @type {any} */ chip, /** @type {number} */ index) => {
      const node = el(
        doc,
        "span",
        `arena-frame__chip arena-frame__chip--${chip.colorClass}`,
      );
      node.append(
        createSvgIcon(doc, chip.glyphKey, {
          className: "arena-frame__chip-glyph",
          style: record.tray?.iconStyle === "outline" ? "outline" : "sigil",
        }),
      );
      if (chip.count !== null)
        node.append(el(doc, "span", "arena-frame__chip-count", String(chip.count)));
      node.setAttribute(
        "aria-label",
        chip.count === null ? chip.name : `${chip.name} ${chip.count}`,
      );
      if (interactive) {
        // The chip's tip says what the Champion Card's row says: the approved
        // name, then what it does and where it comes from, then the time left.
        symbol(node, `chip:${record.publicAgentId}:${index}`, tips, () =>
          record.tipLimited
            ? descriptor(
                `chip:${record.publicAgentId}:${index}`,
                chip.title,
                LIMITED_TIP_TEXT,
                [],
                chip.colorClass,
              )
            : descriptor(
                `chip:${record.publicAgentId}:${index}`,
                chip.title,
                chip.line,
                chip.time === "" ? [] : [["Time Left", chip.time]],
                chip.colorClass,
              ),
        );
      }
      tray.append(node);
    },
  );
  if ((record.tray?.more ?? 0) > 0) {
    const more = el(doc, "span", "arena-frame__more", `+${record.tray.more}`);
    if (interactive) {
      symbol(more, `more:${record.publicAgentId}`, tips, () =>
        descriptor(
          `more:${record.publicAgentId}`,
          symbolById("more_statuses")?.title ?? "More Statuses",
          record.tipLimited
            ? LIMITED_TIP_TEXT
            : (symbolById("more_statuses")?.eli5 ?? ""),
          record.tipLimited
            ? []
            : record.tray.moreLines.map((/** @type {string} */ line) => [
                "Status",
                line,
              ]),
        ),
      );
    }
    tray.append(more);
  }
  if (record.returned || record.spawnShield !== null) {
    const text = record.returned
      ? "Returned"
      : (symbolChipText("spawn_shield", record.spawnShield) ?? "Spawn Shield");
    const chip = el(
      doc,
      "span",
      `arena-frame__ward${record.returned ? " is-returned" : ""}`,
      text,
    );
    if (interactive) {
      const id = record.returned ? "respawn" : "spawn_shield";
      // The Spawn Shield chip's tip says what the Champion Card's row says,
      // as the status chips do: the approved name, the card's sentence and
      // the time left ("3 ticks left"). "Returned" keeps its symbol's own
      // words, with the shield's time left when it has one.
      symbol(chip, `ward:${record.publicAgentId}`, tips, () =>
        descriptor(
          `ward:${record.publicAgentId}`,
          symbolById(id)?.title ?? text,
          record.tipLimited
            ? LIMITED_TIP_TEXT
            : record.returned
              ? (symbolById(id)?.eli5 ?? "")
              : statusCardSentence("spawn_shield").line,
          record.tipLimited || record.spawnShield === null
            ? []
            : [
                [
                  record.returned ? "Spawn Shield" : "Time Left",
                  statusTimeText("spawn_shield", record.spawnShield),
                ],
              ],
        ),
      );
    }
    tray.append(chip);
  }
  body.append(tray);
  return body;
}

/**
 * Build one champion frame (an li element) from its record.
 *
 * @param {Document} doc
 * @param {Record<string, any>} record
 * @param {boolean} tips
 * @returns {HTMLElement}
 */
function frameNode(doc, record, tips) {
  const item = el(
    doc,
    "li",
    `arena-frame arena-frame--${record.state} arena-frame--class-${record.classToken}${record.you ? " is-you" : ""}`,
  );
  item.dataset.publicAgentId = record.publicAgentId;
  item.dataset.teamId = String(record.teamId);
  item.dataset.state = record.state;
  const sigilCell = el(doc, "span", "arena-frame__sigil-cell");
  sigilCell.append(sigil(doc, record, record.state));
  if (record.fallHold !== null) {
    const before = sigil(doc, record, "alive");
    before.classList.add("arena-frame__layer");
    holdBefore(before, "fall", record.fallHold.beatKey);
    sigilCell.append(before);
  } else if (record.returnHold !== null) {
    const before = sigil(doc, record, "fallen");
    before.classList.add("arena-frame__layer");
    holdBefore(before, "return", record.returnHold.beatKey);
    sigilCell.append(before);
  }
  const names = el(doc, "span", "arena-frame__names");
  const agent = el(doc, "span", "arena-frame__agent", record.chipText);
  if (record.you) agent.append(el(doc, "span", "arena-frame__you", "You"));
  names.append(agent, el(doc, "span", "arena-frame__class", record.className));
  const bodyCell = el(doc, "span", "arena-frame__body");
  if (record.state === "unseen") {
    bodyCell.append(el(doc, "span", "arena-frame__unseen", UNSEEN_LABEL));
  } else if (record.state === "fallen") {
    const after = fallenBody(doc, record, record.returnsIn, tips, true);
    bodyCell.append(after);
    if (record.fallHold !== null) {
      holdAfter(after, "fall", record.fallHold.beatKey);
      const before = aliveBody(doc, record, tips, false, record.fallHold.life);
      before.classList.add("arena-frame__layer");
      holdBefore(before, "fall", record.fallHold.beatKey);
      bodyCell.append(before);
    }
  } else {
    const after = aliveBody(doc, record, tips, true);
    bodyCell.append(after);
    if (record.returnHold !== null) {
      holdAfter(after, "return", record.returnHold.beatKey);
      const before = fallenBody(doc, record, record.returnHold.returnsIn, tips, false);
      before.classList.add("arena-frame__layer");
      holdBefore(before, "return", record.returnHold.beatKey);
      bodyCell.append(before);
    }
  }
  item.append(sigilCell, names, bodyCell);
  symbol(item, `frame:${record.publicAgentId}`, tips, () => tooltip(record));
  item.setAttribute(
    "aria-label",
    record.state === "unseen"
      ? `${record.chipText}, ${record.className}, ${UNSEEN_LABEL}`
      : record.state === "fallen"
        ? `${record.chipText}, ${record.className}, Fallen${record.returnsIn === null ? "" : `, Returns In ${record.returnsIn}`}`
        : `${record.chipText}, ${record.className}${record.life?.text ? `, Life ${record.life.text}` : ""}`,
  );
  return item;
}

/**
 * Build the HUD's DOM nodes from its records.
 *
 * Returns new, unattached HTML elements in order: the "Full Record" chip
 * (Replay Viewer Agent POV only), then one dock per team (div.arena-frames__dock,
 * Team A first): its header and an ordered list of frames. Every symbol
 * (frame, life bar, chip, "+N", Spawn Shield or Returned chip, "Returns In",
 * crest, Standing) carries data-hud-symbol and tabindex -1, and gets its
 * shared tip when the hud_view record says tips are on. Beat pairs follow the
 * choreography.js contract (data-beat-hold, data-beat-key, data-beat-side;
 * "before" nodes drawn at opacity 0). A header's team name
 * (span.arena-frames__team-name) whose record has controllers is remembered,
 * so createHud's render paints it as the team's controller line once it is
 * in the page. Returns [] when the records hold no champion or the Team
 * Rosters switch is off. Side effects: tooltip registry entries for the new
 * nodes. Not used inside the arena SVG.
 *
 * @param {ReadonlyArray<LookRecord>} hudRecords From records().
 * @param {Document} doc
 * @returns {HTMLElement[]}
 */
export function build(hudRecords, doc) {
  const view = hudRecords.find((record) => record.kind === "hud_view");
  if (!view?.rosters) return [];
  const tips = view.tips === true;
  /** @type {HTMLElement[]} */
  const nodes = [];
  if (view.fullRecord) {
    const chip = el(
      doc,
      "p",
      "arena-frames__full-record",
      view.fullRecordLabel ?? FULL_RECORD_LABEL,
    );
    chip.title =
      "These frames show every champion's facts. The arena and its tips show only what your team can see.";
    nodes.push(chip);
  }
  /** @type {HTMLElement | null} */
  let list = null;
  for (const record of hudRecords) {
    if (record.kind === "team_header") {
      const dock = el(
        doc,
        "div",
        `arena-frames__dock arena-frames__dock--${record.teamId === 1 ? "a" : "b"}`,
      );
      dock.dataset.teamId = String(record.teamId);
      const header = el(doc, "div", "arena-frames__header");
      const crest = el(doc, "span", "arena-frames__crest");
      crest.append(
        createSvgIcon(doc, record.teamId === 1 ? "team-a" : "team-b", {
          className: "arena-frames__crest-icon",
        }),
      );
      symbol(crest, `team:${record.teamId}`, tips, () =>
        descriptor(
          `team:${record.teamId}`,
          record.label,
          symbolById(record.teamId === 1 ? "team_a" : "team_b")?.eli5 ?? "",
          record.name ? [["Policy", record.name]] : [],
        ),
      );
      header.append(crest, el(doc, "span", "arena-frames__eyebrow", record.label));
      if (record.name) {
        const name = el(doc, "span", "arena-frames__team-name", record.name);
        // createHud's render paints it as the team's controller line once it
        // is in the page (a line needs its real width to fit).
        if (record.controllers?.controllers.length > 0)
          teamNameLines.set(name, record.controllers);
        header.append(name);
      }
      if (record.standing !== null) {
        const standing = el(doc, "span", "arena-frames__standing");
        const after = el(
          doc,
          "span",
          "arena-frames__standing-value",
          `Standing ${record.standing.alive} of ${record.standing.total}`,
        );
        standing.append(after);
        if (record.standingHold !== null) {
          holdAfter(after, record.standingHold.kind, record.standingHold.beatKey);
          const before = el(
            doc,
            "span",
            "arena-frames__standing-value arena-frame__layer",
            `Standing ${record.standingHold.before.alive} of ${record.standingHold.before.total}`,
          );
          holdBefore(before, record.standingHold.kind, record.standingHold.beatKey);
          standing.append(before);
        }
        symbol(standing, `standing:${record.teamId}`, tips, () =>
          descriptor(
            `standing:${record.teamId}`,
            symbolById("fighters_standing")?.title ?? "Fighters Standing",
            symbolById("fighters_standing")?.eli5 ?? "",
            [["Standing", `${record.standing.alive} of ${record.standing.total}`]],
          ),
        );
        header.append(standing);
      }
      list = /** @type {HTMLElement} */ (el(doc, "ol", "arena-frames__list"));
      dock.append(header, list);
      nodes.push(dock);
    } else if (record.kind === "champion" && list !== null) {
      list.append(frameNode(doc, record, tips));
    }
  }
  return nodes;
}

/**
 * Team names build() made that render paints as controller lines, with
 * their team in the one controller model.
 *
 * @type {WeakMap<Element, import("../controller-names.js").ControllerTeam>}
 */
const teamNameLines = new WeakMap();

// ---------------------------------------------------------------------------
// The live scorebug: hosts and portraits (BattleClient and DevClient)

/**
 * The document event main.js sends with every installed live frame's combat
 * configuration (detail.manual_slots and detail.slot_labels, among others).
 */
export const COMBAT_CONFIGURATION_EVENT = "marl-combat-configuration-installed";

/** A portrait token's square SVG side in CSS px (radius 13, as the vision's). */
export const PORTRAIT_TOKEN_SIZE = 28;

/** A portrait token's radius in CSS px. */
const PORTRAIT_TOKEN_RADIUS = 13;

/**
 * @typedef {Readonly<{
 *   manualSlots: ReadonlyArray<number>,
 *   slotLabels: ReadonlyArray<string | null>,
 * }>} LiveControllers
 *   Who plays each command slot. A command slot is (team_id - 1) * 5 +
 *   team_local_slot: 0 to 4 for Team A, 5 to 9 for Team B. manualSlots lists
 *   the slots a human orders (Manual); slotLabels names each slot's
 *   controller in slot order ("Manual", "Recurrent MAPPO", "tdm-beta"), null
 *   for a slot with no champion.
 */

/** No combat configuration yet: nobody is Manual and no slot has a label. */
export const NO_CONTROLLERS = Object.freeze({
  manualSlots: Object.freeze([]),
  slotLabels: Object.freeze([]),
});

/**
 * Read the controller facts from a combat configuration.
 *
 * configuration is the detail of COMBAT_CONFIGURATION_EVENT (or the
 * transport's combat_configuration): manual_slots (whole numbers 0 to 9) and
 * slot_labels (ten strings or nulls). Anything else gives NO_CONTROLLERS;
 * malformed entries are dropped. Returns a new frozen LiveControllers. Pure.
 *
 * Agent POV: who plays each slot is set before the game starts and is the
 * same in both views, so it reveals nothing the POV team cannot see.
 *
 * @param {unknown} configuration
 * @returns {LiveControllers}
 */
export function liveControllers(configuration) {
  const manual = field(configuration, "manual_slots");
  const labels = field(configuration, "slot_labels");
  if (!Array.isArray(manual) || !Array.isArray(labels)) return NO_CONTROLLERS;
  return Object.freeze({
    manualSlots: Object.freeze(
      manual.filter((slot) => Number.isInteger(slot) && slot >= 0 && slot < 10),
    ),
    slotLabels: Object.freeze(
      labels.map((label) => (typeof label === "string" ? label : null)),
    ),
  });
}

/**
 * Build both teams' controllers, the one shared model of controller-names.js
 * (controllerRoster), from the launch's controller facts and the identity
 * directory.
 *
 * controllers gives manual_slots and slot_labels; identities are
 * hudSourcesFromFrame's identity rows (publicRosterIdentities) and
 * displayIdOf its shown-ID reader. These are the same facts
 * controllerRosterFromFrame (match-summary.js) reads for the Tactics Turn
 * rail, so the scorebug lines, the hover lists and the rail's Policies list
 * always agree. With no identities (before the first frame), every slot with
 * a label counts as one champion of unknown class. Pure.
 *
 * @param {LiveControllers} controllers
 * @param {ReadonlyArray<Readonly<{publicAgentId: string, teamId: 1 | 2, teamLocalSlot: number, classId: number | null}>>} [identities]
 *   Default none.
 * @param {((publicAgentId: string) => string | null) | null} [displayIdOf]
 *   Default none: the command slot is shown as the Agent ID.
 * @returns {import("../controller-names.js").ControllerRoster}
 */
export function liveControllerRoster(controllers, identities = [], displayIdOf = null) {
  return controllerRoster({
    manualSlots: controllers.manualSlots,
    slotLabels: controllers.slotLabels,
    champions: identities.map((identity) => ({
      ...identity,
      displayId: displayIdOf?.(identity.publicAgentId) ?? null,
    })),
  });
}

/**
 * Build one team's full controller line: "You" when the team has a Manual
 * champion, then each different policy name of its other slots in slot
 * order, joined by " + ". Examples: "You + Recurrent MAPPO" (you play one
 * champion, a policy the rest), "tdm-beta" (policies only), "You" (you play
 * them all). Names come from the launch as they are. Returns "" when no slot
 * of the team has a label and none is Manual (for example before the
 * configuration arrives). Pure.
 *
 * These are the words of the shared model (liveControllerRoster, then
 * controllerLineParts). The scorebug shows them only when they fit; else it
 * shows the short line ("You + 4 Policies"; paintControllerLine in
 * controller-names.js). They are also the line's aria-label.
 *
 * @param {1 | 2} teamId
 * @param {LiveControllers} controllers
 * @returns {string}
 */
export function controllerLine(teamId, controllers) {
  return controllerLineParts(liveControllerRoster(controllers).teams[teamId - 1]).join(
    " + ",
  );
}

/**
 * Say which team carries the scorebug's "Yours" tag.
 *
 * The vision marks Team A, the researcher's side, in every mode (one Manual
 * champion, one on each team, or Policies Only, where Team A is still the
 * researcher's policy). Team B gets it only when every Manual slot is on
 * Team B. Pure.
 *
 * @param {LiveControllers} controllers
 * @returns {1 | 2}
 */
export function yoursTeam(controllers) {
  const manual = controllers.manualSlots;
  return manual.length > 0 && manual.every((slot) => slot >= 5) ? 2 : 1;
}

/**
 * @typedef {Readonly<{fraction: number, low: boolean, current: number}>} PortraitLife
 *   A portrait's life bar: fraction 0 to 1 of the maximum, low true at
 *   CINDER_FRACTION or less (Cinder fill, else Ivory), current in hit points.
 */

/**
 * @typedef {Readonly<{
 *   publicAgentId: string,
 *   displayId: string,
 *   teamId: number,
 *   classId: number | null,
 *   className: string,
 *   state: "alive" | "fallen" | "unseen",
 *   manual: boolean,
 *   focus: boolean,
 *   life: PortraitLife | null,
 *   label: string,
 *   labelKind: "you" | "id" | "unseen",
 *   ariaLabel: string,
 *   refusal: string | null,
 *   hold: Readonly<{
 *     kind: "fall" | "return",
 *     beatKey: string,
 *     state: "alive" | "fallen",
 *     life: PortraitLife,
 *   }> | null,
 *   record: LookRecord,
 * }>} PortraitModel
 *   One scorebug portrait. displayId is the shown ID ("0" to "9", the
 *   vision's data-slot). state "unseen" (Agent POV, an enemy the POV team
 *   cannot see) carries identity only: life null, no hold. manual: a human
 *   orders it (label "You"). focus: the champion the player looks at (never
 *   an unseen one). label: "You", "ID 5" or "Unseen". refusal: for an unseen
 *   portrait that is not your Manual champion, the status line sentence a
 *   click gives ("The Warrior (ID 6) is out of your team's sight. Agent POV
 *   hides where it is and how it fares."); else null (an unseen Manual
 *   champion, yours on the other team, is selected by a click). hold: the look before this tick's beat (kind
 *   "fall": alive with the life from before the blow; kind "return": fallen),
 *   or null. record: the HUD champion record, for the Champion Card tip.
 */

/**
 * @typedef {Readonly<{
 *   yoursTeam: 1 | 2,
 *   lines: readonly [string, string],
 *   roster: import("../controller-names.js").ControllerRoster,
 *   rosters: boolean,
 *   teams: ReadonlyArray<Readonly<{teamId: 1 | 2, portraits: ReadonlyArray<PortraitModel>}>>,
 * }>} LiveScorebugModel
 *   The live scorebug's hosts and portraits: which team shows "Yours", the
 *   two full controller lines (Team A first; the screen shows each only when
 *   it fits, else its short line), the one controller model they come from
 *   (roster; the hover lists and the rail's Policies list use the same
 *   model), whether the portraits show (the Team Rosters switch) and each
 *   team's portraits in shown-ID order, Team A first.
 */

/**
 * Turn the HUD records into the live scorebug's hosts and portraits. Pure: no
 * DOM.
 *
 * hudRecords is records({sources, beat, filters}) for the same sources;
 * sources is hudSourcesFromFrame(frame) (match-summary.js) of a BattleClient
 * or DevClient frame; controllers is the launch's LiveControllers
 * (NO_CONTROLLERS before the configuration arrives).
 *
 * Rules (the approved vision page): every champion a human orders is
 * "You"; others read "ID n". The focus look goes to the champion the player
 * looks at (sources.controlledKey: the one whose orders show, or a policy's
 * champion in Watch Only), never to an unseen one. In live play the scene's
 * selected key names the target, so it is not used here. A
 * fallen champion keeps its label with an empty life bar. A champion that
 * falls this tick holds its living look with its life from before the blow
 * until its Fall beat; one that returns holds its fallen look until its
 * Return beat (owner, 1 Oct 2026: life bars and the score wait for the Fall
 * beat; a returning champion reappears at its Return beat).
 *
 * Agent POV: an unseen champion's portrait comes from identity fields only
 * (class, team, shown ID, command slot) and reads "Unseen", with no life,
 * hold or focus; its click sentence names the viewing team ("your team"
 * when you order champions, else "Team A" or "Team B", the team whose eyes
 * the view uses). Oracle View lists everyone. Oracle -> Agent POV -> Oracle
 * gives exactly a fresh model: nothing is kept between calls.
 *
 * @param {ReadonlyArray<LookRecord>} hudRecords
 * @param {HudSources} sources
 * @param {LiveControllers} controllers
 * @returns {LiveScorebugModel}
 */
export function liveScorebugModel(hudRecords, sources, controllers) {
  const view = hudRecords.find((record) => record.kind === "hud_view");
  const champions = hudRecords.filter((record) => record.kind === "champion");
  const slotOf = new Map(
    sources.identities.map((identity) => [
      identity.publicAgentId,
      (identity.teamId - 1) * 5 + identity.teamLocalSlot,
    ]),
  );
  const keyOwner = (/** @type {string | null | undefined} */ key) =>
    typeof key === "string"
      ? (sources.sceneAgents.find((agent) => agent.presentation_key === key)
          ?.public_agent_id ?? null)
      : null;
  const controlledId = keyOwner(sources.controlledKey);
  const focusId = controlledId;
  const teamOfId = (/** @type {string | null} */ id) =>
    id === null
      ? null
      : (champions.find((record) => record.publicAgentId === id)?.teamId ?? null);
  const anyUnseen = (/** @type {number} */ teamId) =>
    champions.some((record) => record.teamId === teamId && record.state === "unseen");
  const viewingTeam =
    teamOfId(sources.recipientPublicAgentId) ??
    teamOfId(controlledId) ??
    ([1, 2].find((teamId) => !anyUnseen(teamId) && anyUnseen(3 - teamId)) || null);
  const viewers =
    controllers.manualSlots.length > 0 || viewingTeam === null
      ? "your team"
      : (teamLabelFromId(viewingTeam) ?? "your team");
  /** @param {Readonly<Record<string, any>> | null} life */
  const portraitLife = (life) =>
    life === null
      ? null
      : Object.freeze({
          fraction: Number(life.fraction),
          low: life.low === true,
          current: Number(life.current),
        });
  const fallenLife = Object.freeze({ fraction: 0, low: true, current: 0 });
  const portraits = champions.map((record) => {
    const slot = slotOf.get(record.publicAgentId) ?? Number(record.displayId);
    const manual = controllers.manualSlots.includes(slot);
    const unseen = record.state === "unseen";
    const teamWords = teamLabelFromId(record.teamId) ?? "Unknown Team";
    /** @type {PortraitModel["hold"]} */
    let hold = null;
    if (record.fallHold !== null) {
      hold = Object.freeze({
        kind: "fall",
        beatKey: String(record.fallHold.beatKey),
        state: "alive",
        life: portraitLife(record.fallHold.life) ?? fallenLife,
      });
    } else if (record.returnHold !== null) {
      hold = Object.freeze({
        kind: "return",
        beatKey: String(record.returnHold.beatKey),
        state: "fallen",
        life: fallenLife,
      });
    }
    return Object.freeze({
      publicAgentId: record.publicAgentId,
      displayId: record.displayId,
      teamId: record.teamId,
      classId: record.classId,
      className: record.className,
      state: record.state,
      manual,
      focus: !unseen && record.publicAgentId === focusId,
      life: unseen
        ? null
        : record.state === "fallen"
          ? fallenLife
          : (portraitLife(record.life) ?? fallenLife),
      label: unseen ? UNSEEN_LABEL : manual ? "You" : `ID ${record.displayId}`,
      labelKind: unseen ? "unseen" : manual ? "you" : "id",
      ariaLabel: unseen
        ? `${record.className}, Agent ID ${record.displayId}, unseen by ${viewers}`
        : `${record.className}, Agent ID ${record.displayId}, ${teamWords}${manual ? ", yours" : ""}${record.state === "fallen" ? ", fallen" : ""}`,
      // Your own Manual champion on the other team (One On Each Team) is
      // unseen by this view but still yours: a click gives it the orders.
      refusal:
        unseen && !manual
          ? `The ${record.className} (ID ${record.displayId}) is out of ${viewers}'s sight. Agent POV hides where it is and how it fares.`
          : null,
      hold: unseen ? null : hold,
      record,
    });
  });
  const roster = liveControllerRoster(
    controllers,
    sources.identities,
    sources.displayIdOf,
  );
  return Object.freeze({
    yoursTeam: yoursTeam(controllers),
    lines: /** @type {readonly [string, string]} */ (
      Object.freeze(roster.teams.map((team) => controllerLineParts(team).join(" + ")))
    ),
    roster,
    rosters: view?.rosters !== false,
    teams: Object.freeze(
      /** @type {const} */ ([1, 2]).map((teamId) =>
        Object.freeze({
          teamId,
          portraits: Object.freeze(
            portraits.filter((portrait) => portrait.teamId === teamId),
          ),
        }),
      ),
    ),
  });
}

/**
 * @typedef {{
 *   filters: Readonly<Record<string, boolean>> | null,
 *   tips: boolean,
 *   sceneById: ReadonlyMap<string, Record<string, any>>,
 *   spawnShieldMechanics: Record<string, any> | null,
 *   cardFacts: ((subject: import("../tooltip.js").HoverCardSubject) => import("../tooltip.js").ChampionCardFacts | null) | null,
 * }} PortraitContext
 *   What drawing a portrait needs besides its model: the visual filter state
 *   (null: defaults), whether Symbol Explanations is on (the Champion Card
 *   tip), the audience's scene agents by public ID (Agent POV: seen ones
 *   only), the scene's Spawn Shield mechanics, and the live hover card's own
 *   facts builder (HudOptions.cardFacts), or null.
 */

/**
 * Paint the live scorebug's hosts and portraits into its fixed markup.
 *
 * scorebug is #scorebug from index.html (.sb-host.a and .sb-host.b, each with
 * a span.yours; #ctl-a, #ctl-b; #party-a, #party-b). model is
 * liveScorebugModel's result; context see PortraitContext. It shows the
 * "Yours" tag on model.yoursTeam only, paints both controller lines from
 * model.roster (paintControllerLine: the names when they fit in the line's
 * real width, else "You + 4 Policies"; long names cut with "…"; hovering a
 * line drops down its team's list), publishes model.roster for those lists,
 * hides the parties when Team Rosters is off and rebuilds each party's
 * portraits (portraitNode).
 *
 * Repaints: a party whose portraits would draw exactly what it already
 * shows (partyPaintKey equal to the last paint) keeps its nodes. The page
 * repaints a tick while it plays (a new server revision of the same tick,
 * or the combat configuration event that comes with each frame); rebuilt
 * nodes would drop the choreographer's running Fall and Return holds, so a
 * falling champion's portrait would show its fallen look before the Fall
 * beat. Any change rebuilds that whole party.
 *
 * Side effects: DOM inside scorebug, tooltip registrations, the last paint
 * per party (paintedParties), the controller lines' resize observer and the
 * hover list's document listeners (controller-names.js); no storage or
 * network. Missing parts are skipped.
 *
 * @param {HTMLElement} scorebug
 * @param {LiveScorebugModel} model
 * @param {PortraitContext} context
 * @returns {void}
 */
function paintLiveScorebug(scorebug, model, context) {
  const doc = scorebug.ownerDocument;
  for (const team of model.teams) {
    const key = team.teamId === 1 ? "a" : "b";
    const yours = scorebug.querySelector(`.sb-host.${key} .yours`);
    if (yours instanceof HTMLElement) yours.hidden = model.yoursTeam !== team.teamId;
    const line = scorebug.querySelector(`#ctl-${key}`);
    if (line instanceof HTMLElement)
      paintControllerLine(line, model.roster.teams[team.teamId - 1], {
        fitRoot: scorebug,
      });
    const party = scorebug.querySelector(`#party-${key}`);
    if (!(party instanceof HTMLElement)) continue;
    party.hidden = !model.rosters;
    const painted = partyPaintKey(team.portraits, context);
    if (painted !== null && paintedParties.get(party) === painted) continue;
    party.replaceChildren(
      ...team.portraits.map((portrait) => portraitNode(doc, portrait, context)),
    );
    if (painted === null) paintedParties.delete(party);
    else paintedParties.set(party, painted);
  }
  publishControllerRoster(doc, model.roster);
}

/**
 * The last paint of each live party (#party-a, #party-b), as partyPaintKey's
 * text. paintLiveScorebug skips a party whose new key equals it;
 * clearLiveScorebug forgets it. Weak: a removed party is forgotten with it.
 *
 * @type {WeakMap<HTMLElement, string>}
 */
const paintedParties = new WeakMap();

/**
 * Write one text that names everything a party's portraits draw: each
 * portrait model (its record and card facts included), the audience's scene
 * row each token is drawn from, and the drawing context (switch state,
 * Symbol Explanations, Spawn Shield mechanics, whether the hover card's
 * facts builder is given). Equal text means portraitNode would build the same
 * nodes. Pure. Returns null when the facts cannot be written as JSON (the
 * party is then always rebuilt).
 *
 * @param {ReadonlyArray<PortraitModel>} portraits One party's portraits.
 * @param {PortraitContext} context
 * @returns {string | null}
 */
export function partyPaintKey(portraits, context) {
  try {
    return JSON.stringify([
      context.filters,
      context.tips,
      context.spawnShieldMechanics,
      context.cardFacts !== null,
      portraits.map((portrait) => [
        portrait,
        context.sceneById.get(portrait.publicAgentId) ?? null,
      ]),
    ]);
  } catch {
    return null;
  }
}

/**
 * Empty the live scorebug's hosts and portraits (no frame installed). Hides
 * both "Yours" tags, paints both controller lines empty and publishes the
 * empty roster, so an open hover list closes. It also forgets each party's
 * last paint, so the next frame draws its portraits fresh. Side effects: DOM
 * inside scorebug, paintedParties and the controller-name state of its
 * document.
 *
 * @param {HTMLElement} scorebug
 * @returns {void}
 */
function clearLiveScorebug(scorebug) {
  for (const key of ["a", "b"]) {
    const yours = scorebug.querySelector(`.sb-host.${key} .yours`);
    if (yours instanceof HTMLElement) yours.hidden = true;
    const line = scorebug.querySelector(`#ctl-${key}`);
    if (line instanceof HTMLElement)
      paintControllerLine(line, NO_CONTROLLER_ROSTER.teams[key === "a" ? 0 : 1]);
    const party = scorebug.querySelector(`#party-${key}`);
    if (party) {
      party.replaceChildren();
      paintedParties.delete(/** @type {HTMLElement} */ (party));
    }
  }
  publishControllerRoster(scorebug.ownerDocument, NO_CONTROLLER_ROSTER);
}

/**
 * Build one portrait as the vision draws it, plus its beat-held earlier look.
 *
 * The END portrait is a button.por (class "unseen" for an unseen champion,
 * "focus" for the focused one) with data-slot (shown ID), data-public-agent-id
 * and data-state, an aria-label, the token SVG (portraitToken, or
 * unseenToken), the life bar (span.lb with an `i` whose inline width is the
 * life share, class "low" for Cinder; span.lb.none for an unseen champion)
 * and the label span.pl ("You" with class "you", "ID 5", or "Unseen"). With
 * Symbol Explanations on it registers the Champion Card tip, a screen-reader
 * copy of the visible hover card: from context.cardFacts when given (the hover
 * card's own facts, so both read the same), else tooltip(record) (the reduced
 * card for an unseen champion). An unseen portrait also carries
 * data-refusal, the sentence a click gives.
 *
 * With a hold, returns span.por-stack holding the END portrait (data-beat-hold
 * kind, data-beat-key, data-beat-side "after") and after it the earlier look
 * (side "before", opacity 0 at END, aria-hidden, not focusable, no tip),
 * drawn in the same place by scorebug.css; each one's life bar carries the
 * same hold mark as its portrait. Without a hold it returns the END portrait.
 * The
 * choreographer swaps them at the beat when #scorebug is one of its
 * beatHoldRoots. Creates nodes only.
 *
 * @param {Document} doc
 * @param {PortraitModel} portrait
 * @param {PortraitContext} context
 * @returns {HTMLElement}
 */
function portraitNode(doc, portrait, context) {
  const node = portraitButton(doc, portrait, portrait.state, portrait.life, context);
  if (context.tips) {
    // The screen-reader copy of the Champion Card reads exactly as the
    // visible hover card: the same facts from the same builder.
    const facts = context.cardFacts?.(hoverCardSubject(node)) ?? null;
    const desc =
      facts === null
        ? tooltip(portrait.record)
        : createChampionCardDescriptor(
            facts,
            `champion:frame:${portrait.publicAgentId}`,
          );
    if (desc !== null) registerTooltipOwner(node, desc);
  }
  if (portrait.state === "unseen" && portrait.refusal !== null) {
    node.dataset.refusal = portrait.refusal;
  }
  if (portrait.hold === null) return node;
  const before = portraitButton(
    doc,
    portrait,
    portrait.hold.state,
    portrait.hold.life,
    context,
  );
  before.tabIndex = -1;
  holdBefore(before, portrait.hold.kind, portrait.hold.beatKey);
  holdAfter(node, portrait.hold.kind, portrait.hold.beatKey);
  // The life bars carry the same hold, so each bar's own opacity changes at
  // the beat too (the owner's rule names the life bar).
  const beforeBar = before.querySelector(".lb");
  const endBar = node.querySelector(".lb");
  if (beforeBar instanceof HTMLElement)
    holdBefore(beforeBar, portrait.hold.kind, portrait.hold.beatKey);
  if (endBar instanceof HTMLElement)
    holdAfter(endBar, portrait.hold.kind, portrait.hold.beatKey);
  const stack = el(doc, "span", "por-stack");
  stack.append(node, before);
  return stack;
}

/**
 * Build one portrait button for a look: state and life (see portraitNode).
 * The button carries data-slot, data-public-agent-id, data-class-id (1 to 5,
 * when the class is known) and data-state.
 * The token's agent is the audience's scene agent with its life state and
 * health set to the look's (a fall's earlier look is alive at its life before
 * the blow; a return's is fallen). Creates nodes only.
 *
 * @param {Document} doc
 * @param {PortraitModel} portrait
 * @param {"alive" | "fallen" | "unseen"} state
 * @param {PortraitLife | null} life
 * @param {PortraitContext} context
 * @returns {HTMLButtonElement}
 */
function portraitButton(doc, portrait, state, life, context) {
  const button = /** @type {HTMLButtonElement} */ (doc.createElement("button"));
  button.type = "button";
  button.className = `por${state === "unseen" ? " unseen" : ""}${portrait.focus ? " focus" : ""}`;
  button.dataset.slot = portrait.displayId;
  button.dataset.publicAgentId = portrait.publicAgentId;
  // The class (1 to 5) is static roster identity, so an Unseen portrait's
  // hover card can name it without showing anything Agent POV hides.
  if (portrait.classId !== null) button.dataset.classId = String(portrait.classId);
  button.dataset.state = state;
  button.setAttribute("aria-label", portrait.ariaLabel);
  if (state === "unseen") {
    button.append(unseenToken(doc, portrait), el(doc, "span", "lb none"));
  } else {
    const agent = context.sceneById.get(portrait.publicAgentId) ?? {
      public_agent_id: portrait.publicAgentId,
      team_id: portrait.teamId,
      class_id: portrait.classId,
      radius: 0.5,
      position: [0, 0],
    };
    const alive = state === "alive";
    const look = {
      ...agent,
      alive,
      life_state: alive ? "alive" : "dead",
      current_health: alive ? (life?.current ?? agent.current_health) : 0,
    };
    const bar = el(doc, "span", "lb");
    const fill = doc.createElement("i");
    fill.style.width = `${(100 * (alive ? (life?.fraction ?? 0) : 0)).toFixed(1)}%`;
    if (alive && life?.low === true) fill.className = "low";
    bar.append(fill);
    button.append(portraitToken(doc, look, context), bar);
  }
  const label = el(doc, "span", portrait.labelKind === "you" ? "pl you" : "pl");
  label.textContent = portrait.label;
  button.append(label);
  return button;
}

/**
 * Draw a seen champion's token for its portrait (championTokenSvg at 28 px
 * square, radius 13 px, with the portrait's switches and Spawn Shield
 * mechanics). Returns an svg.tok (decorative); see championTokenSvg.
 *
 * @param {Document} doc
 * @param {Readonly<Record<string, any>>} agent A scene agent record.
 * @param {PortraitContext} context
 * @returns {SVGSVGElement}
 */
function portraitToken(doc, agent, context) {
  return championTokenSvg(doc, agent, {
    size: PORTRAIT_TOKEN_SIZE,
    radius: PORTRAIT_TOKEN_RADIUS,
    filters: context.filters,
    spawnShieldMechanics: context.spawnShieldMechanics,
  });
}

/**
 * Draw one seen champion's arena token alone in a small square SVG: the
 * arena's own token (Champions look: Heraldic Token with Life Vessel, or the
 * plain token when those switches are off; the Ashen Token when fallen; the
 * Spawn Ward while shielded and Spawn Shield is on) with its Statuses look
 * marks (the Rune Ring, with 4 gems when the Ultimate is ready, and the Rune
 * Zones status marks; the switches cooldown_effects and status_marks),
 * centered. The scorebug portraits and the BattleClient deck (Your Champion,
 * the target chip, Allowed Now and the Manual list) use it, so a token switch
 * changes the token everywhere it is drawn.
 *
 * agent is a scene agent record (position, radius, team_id, class_id,
 * presentation_key, alive, current_health, max_health,
 * spawn_shield_remaining, ultimate_cooldown_remaining, statuses). The token is
 * drawn with no selection, no events and no beat cues, so it always shows its
 * END look.
 * Its parts are aria-hidden and not focusable. If the look cannot draw it, a
 * plain team ring is drawn instead. Returns an svg.tok (decorative). Creates
 * nodes and may register the ward's tip (pointer events are off on the token
 * in scorebug.css, so the portrait's own tip shows). Agent POV: pass only a
 * champion of this audience's own scene; the token shows nothing the record
 * does not hold.
 *
 * @param {Document} doc
 * @param {Readonly<Record<string, any>>} agent A scene agent record.
 * @param {{size: number, radius: number, filters: Readonly<Record<string, boolean>> | null, spawnShieldMechanics: Record<string, any> | null}} options
 *   size: the SVG's square side in CSS px; radius: the token radius in CSS
 *   px; filters: the visual filter state (null: defaults);
 *   spawnShieldMechanics: the scene's Spawn Shield mechanics (null: none).
 * @returns {SVGSVGElement}
 */
export function championTokenSvg(doc, agent, options) {
  const size = options.size;
  const middle = size / 2;
  const svg = /** @type {SVGSVGElement} */ (doc.createElementNS(SVG_NAMESPACE, "svg"));
  for (const [name, value] of Object.entries({
    class: "tok",
    width: size,
    height: size,
    viewBox: `0 0 ${size} ${size}`,
    "aria-hidden": "true",
    focusable: "false",
  })) {
    svg.setAttribute(name, String(value));
  }
  const center = Array.isArray(agent.position) ? agent.position : [0, 0];
  const radius = Number(agent.radius);
  const scale = options.radius / Math.max(Number.isFinite(radius) ? radius : 0.5, 0.01);
  const bounds = Object.freeze({
    left: 0,
    top: 0,
    right: size,
    bottom: size,
    width: size,
    height: size,
  });
  /** @type {import("./look-support.js").LookTransform} */
  const transform = {
    scale,
    worldWidth: size / scale,
    worldHeight: size / scale,
    viewportBounds: bounds,
    mapBounds: bounds,
    worldToScreen: (point) => {
      const [x, y] = Array.isArray(point)
        ? point
        : [/** @type {any} */ (point).x, /** @type {any} */ (point).y];
      return {
        x: middle + (Number(x) - Number(center[0])) * scale,
        y: middle - (Number(y) - Number(center[1])) * scale,
      };
    },
    worldLengthToScreen: (length) => length * scale,
  };
  try {
    const tokenRecords = champions.records({
      scene: {
        agents: [agent],
        selection: {},
        spawn_shield_mechanics: options.spawnShieldMechanics,
      },
      transform,
      visualFilters: options.filters,
      showSpawnShield: options.filters?.spawn_shield !== false,
    });
    const nodes = champions.build(tokenRecords, doc);
    champions.update(nodes, tokenRecords);
    const bundle = nodes[0];
    if (bundle === undefined) throw new RangeError("no portrait token");
    const filters = options.filters ?? {};
    const marks = statuses.build(
      statuses.records({
        part: "status-marks",
        scene: { agents: [agent] },
        transform,
        events: [],
        startSeen: () => false,
        switches: {
          statusMarks: filters.status_marks !== false,
          cooldownEffects: filters.cooldown_effects !== false,
          statusApplication: filters.status_application !== false,
          naturalStatusExpiry: filters.natural_status_expiry !== false,
          freezingTrapBreak: filters.freezing_trap_break !== false,
        },
      }),
      doc,
    );
    for (const part of [bundle.root, bundle.shieldRoot, ...marks]) {
      if (part === null) continue;
      part.removeAttribute("tabindex");
      part.removeAttribute("role");
      part.setAttribute("aria-hidden", "true");
      svg.append(part);
    }
  } catch {
    const ring = doc.createElementNS(SVG_NAMESPACE, "circle");
    for (const [name, value] of Object.entries({
      class: "tok__fallback",
      cx: middle,
      cy: middle,
      r: options.radius - 2,
      "data-team": agent.team_id === 1 ? "team-a" : "team-b",
    })) {
      ring.setAttribute(name, String(value));
    }
    svg.replaceChildren(ring);
  }
  return svg;
}

/**
 * Draw an unseen champion's token for its portrait, as the vision's
 * unseen_token: a dark disc, a dashed ring in its team color (the team is
 * public) and its class emblem in ash at 38% (the roster is public). No life,
 * Ultimate, status or ward, so nothing hidden reaches the HUD. Returns a
 * 28 px svg.tok.unseen-tok (decorative). Creates nodes only.
 *
 * @param {Document} doc
 * @param {PortraitModel} portrait
 * @returns {SVGSVGElement}
 */
function unseenToken(doc, portrait) {
  const size = PORTRAIT_TOKEN_SIZE;
  const middle = size / 2;
  const svg = /** @type {SVGSVGElement} */ (doc.createElementNS(SVG_NAMESPACE, "svg"));
  for (const [name, value] of Object.entries({
    class: "tok unseen-tok",
    width: size,
    height: size,
    viewBox: `0 0 ${size} ${size}`,
    "aria-hidden": "true",
    focusable: "false",
  })) {
    svg.setAttribute(name, String(value));
  }
  const disc = doc.createElementNS(SVG_NAMESPACE, "circle");
  const ring = doc.createElementNS(SVG_NAMESPACE, "circle");
  for (const [node, attributes] of /** @type {const} */ ([
    [
      disc,
      {
        class: "unseen-tok__disc",
        cx: middle,
        cy: middle,
        r: PORTRAIT_TOKEN_RADIUS - 1,
      },
    ],
    [
      ring,
      {
        class: "unseen-tok__ring",
        cx: middle,
        cy: middle,
        r: PORTRAIT_TOKEN_RADIUS - 1.25,
        "data-team": portrait.teamId === 1 ? "team-a" : "team-b",
      },
    ],
  ])) {
    for (const [name, value] of Object.entries(attributes)) {
      node.setAttribute(name, String(value));
    }
  }
  const side = 1.05 * PORTRAIT_TOKEN_RADIUS;
  const emblem = createSvgIcon(doc, classTokenFromId(portrait.classId).glyphKey, {
    className: "unseen-tok__emblem",
    size: side,
  });
  emblem.setAttribute("x", String(middle - side / 2));
  emblem.setAttribute("y", String(middle - side / 2));
  emblem.setAttribute("width", String(side));
  emblem.setAttribute("height", String(side));
  svg.append(disc, ring, emblem);
  return svg;
}

/**
 * Reproject existing nodes after a resize or camera change. The HUD is page
 * chrome laid out by CSS, so this does nothing. Returns undefined.
 *
 * @param {ReadonlyArray<Element>} _nodes
 * @param {ReadonlyArray<LookRecord>} _records
 */
export function update(_nodes, _records) {}

/**
 * @typedef {{
 *   frames: HTMLElement,
 *   extraRoots?: ReadonlyArray<HTMLElement> | null,
 *   onSelect?: ((publicAgentId: string) => void) | null,
 *   onRefuse?: ((sentence: string) => void) | null,
 *   cardFacts?: ((subject: import("../tooltip.js").HoverCardSubject) => import("../tooltip.js").ChampionCardFacts | null) | null,
 * }} HudOptions
 *   frames: the container for the party frames (#party-frames, or the
 *   Replay Viewer's Team Rosters strip). extraRoots: other HUD containers
 *   whose data-hud-symbol marks share the one Tab stop, in Tab order before
 *   the frames (the score strip's #match-scoreboard); default none. onSelect:
 *   called with a champion's public ID when a frame or a live scorebug
 *   portrait is clicked, or Enter or Space is pressed on it (the Roster row's
 *   command); unseen frames and portraits never call it; default none.
 *   onRefuse: called with the status line sentence when an unseen live
 *   portrait is clicked ("The Warrior (ID 6) is out of your team's sight.
 *   Agent POV hides where it is and how it fares."); default none (the click
 *   then does nothing). cardFacts: live play only, the visible hover card's
 *   facts builder (shell.js championCardFacts over the installed frame),
 *   called with a portrait's hoverCardSubject; each live portrait's
 *   screen-reader copy of the Champion Card is built from it, so it reads
 *   exactly as the visible card (same Unseen sentence, same "Played by");
 *   default none (the copy then comes from the HUD record's own card facts).
 */

/**
 * Mount the HUD on its containers and return its controller.
 *
 * render(sources, {filters, beat, falls}) draws the party frames from
 * hudSourcesFromFrame(frame) (match-summary.js) and returns the records it
 * drew. Call renderMatchSummary for the strip first, then render, so the Tab
 * stop is settled across both; call both before the choreographer plans the
 * tick so its beat holds find the "before" nodes (pass the containers as
 * surface.beatHoldRoots). render hides the frames container (hidden = true)
 * when there is nothing to draw or Team Rosters is off.
 *
 * The whole HUD is one Tab stop: exactly one symbol across the extra roots and
 * the frames has tabindex 0 (the last one focused, else the first). The arrow
 * keys move between symbols, Home and End jump to the ends, Escape leaves
 * (blurs) the HUD, and Enter or Space on a frame selects it. Those keys never
 * reach the tick-stepping handlers while the HUD has focus (preventDefault and
 * stopPropagation); keys with Ctrl, Alt or Meta pass through. destroy()
 * removes the listeners.
 *
 * Live play (BattleClient and DevClient combat screen): when the document has
 * the vision scorebug (#scorebug) and the sources are a "combat_debugger"
 * frame, render draws the scorebug's hosts and portraits instead
 * (liveScorebugModel, then paintLiveScorebug) and leaves the frames container
 * empty and hidden. Portraits are native buttons in the page's Tab order (not
 * the HUD's one Tab stop). A click on a seen portrait calls onSelect; on an
 * unseen one, onRefuse with its sentence. Who plays each slot comes from
 * COMBAT_CONFIGURATION_EVENT on the document (main.js sends it with every
 * installed live frame); a new configuration repaints the last sources at
 * once. render(null) empties the hosts and portraits too.
 *
 * Controller names (controller-names.js): the live scorebug's two lines
 * and, in the Replay Viewer, each Team Rosters header's team name are
 * controller lines from the one controller model (live: liveControllerRoster;
 * Replay Viewer: sources.controllerRoster). Each shows its names when they
 * fit in its real width, else the short line ("You + 4 Policies", "1
 * Policy"); any long name is cut with "…"; hovering a line drops down the
 * team's list ("Team A Controllers").
 *
 * Side effects: DOM of the frames container and of #scorebug's hosts and
 * parties, tabindex of HUD symbols in every root, tooltip registrations,
 * keydown, focusin and click listeners, one document listener for the
 * configuration event, and the controller lines' resize observer and hover
 * list listeners (controller-names.js, its 120 ms hide delay). No storage
 * or network.
 *
 * @param {HudOptions} options
 * @returns {Readonly<{render: (sources: HudSources | null, renderOptions?: Omit<HudInput, "sources">) => ReadonlyArray<LookRecord>, destroy: () => void}>}
 */
export function createHud(options) {
  const frames = options.frames;
  const roots = [...(options.extraRoots ?? []), frames];
  /** @type {string | null} */
  let focusedId = null;

  /** Every visible HUD symbol across the roots, in Tab order (before nodes and hidden containers left out). */
  const symbols = () =>
    roots
      .flatMap((root) => [
        .../** @type {NodeListOf<HTMLElement>} */ (
          root.querySelectorAll("[data-hud-symbol]")
        ),
      ])
      .filter(
        (node) =>
          !node.closest("[hidden]") && node.getAttribute("aria-hidden") !== "true",
      );

  /** Give exactly one symbol tabindex 0: the remembered one, else the first. */
  const settleTabStop = () => {
    const all = symbols();
    const current = all.find((node) => node.dataset.hudSymbol === focusedId) ?? all[0];
    for (const node of all) node.tabIndex = node === current ? 0 : -1;
    return { all, current };
  };

  /**
   * Move, select or leave on the HUD keys; ignore every other key and any key
   * with Ctrl, Alt or Meta. Handled keys get preventDefault and stopPropagation.
   *
   * @param {KeyboardEvent} event
   */
  const onKeyDown = (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    const target = /** @type {HTMLElement | null} */ (
      event.target instanceof Element ? event.target.closest("[data-hud-symbol]") : null
    );
    if (target === null) return;
    const all = symbols();
    const index = all.indexOf(target);
    /** @type {HTMLElement | undefined} */
    let next;
    if (event.key === "ArrowRight" || event.key === "ArrowDown")
      next = all[Math.min(all.length - 1, index + 1)];
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp")
      next = all[Math.max(0, index - 1)];
    else if (event.key === "Home") next = all[0];
    else if (event.key === "End") next = all[all.length - 1];
    else if (event.key === "Escape") {
      target.blur();
    } else if (
      (event.key === "Enter" || event.key === " ") &&
      target.matches("li.arena-frame")
    ) {
      select(target);
    } else {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (next) {
      focusedId = next.dataset.hudSymbol ?? null;
      settleTabStop();
      next.focus();
    }
  };

  /**
   * Call onSelect with a frame's public ID, unless the frame is unseen.
   *
   * @param {Element} item An li.arena-frame.
   */
  const select = (item) => {
    const id = item.getAttribute("data-public-agent-id");
    if (id !== null && item.getAttribute("data-state") !== "unseen")
      options.onSelect?.(id);
  };

  /**
   * Select the frame that holds the click target, if any.
   *
   * @param {Event} event
   */
  const onClick = (event) => {
    const item =
      event.target instanceof Element ? event.target.closest("li.arena-frame") : null;
    if (item) select(item);
  };

  /**
   * Remember the focused symbol and move the one Tab stop to it.
   *
   * @param {FocusEvent} event
   */
  const onFocusIn = (event) => {
    const target =
      event.target instanceof Element
        ? event.target.closest("[data-hud-symbol]")
        : null;
    if (target instanceof HTMLElement || target instanceof SVGElement) {
      focusedId = target.getAttribute("data-hud-symbol");
      settleTabStop();
    }
  };

  for (const root of roots) {
    root.addEventListener("keydown", onKeyDown);
    root.addEventListener("focusin", onFocusIn);
  }
  frames.addEventListener("click", onClick);
  frames.classList.add("arena-frames");
  frames.setAttribute("aria-label", "Arena Frames");

  // The live scorebug (BattleClient and DevClient); null on documents without one.
  const page = frames.ownerDocument;
  const scorebug =
    typeof page?.getElementById === "function" ? page.getElementById("scorebug") : null;
  /** @type {LiveControllers} */
  let controllers = NO_CONTROLLERS;
  /** @type {{sources: HudSources, renderOptions: Omit<HudInput, "sources">} | null} */
  let lastLive = null;

  /**
   * Draw the live scorebug's hosts and portraits from one frame's sources.
   *
   * @param {HudSources} sources
   * @param {Omit<HudInput, "sources">} renderOptions
   * @returns {ReadonlyArray<LookRecord>} The HUD records drawn from.
   */
  const drawLive = (sources, renderOptions) => {
    const hudRecords = records({ ...renderOptions, sources });
    const view = hudRecords.find((record) => record.kind === "hud_view");
    paintLiveScorebug(
      /** @type {HTMLElement} */ (scorebug),
      liveScorebugModel(hudRecords, sources, controllers),
      {
        filters: renderOptions.filters ?? null,
        tips: view?.tips === true,
        sceneById: new Map(
          sources.sceneAgents.map((agent) => [String(agent.public_agent_id), agent]),
        ),
        spawnShieldMechanics: sources.spawnShieldMechanics ?? null,
        cardFacts: options.cardFacts ?? null,
      },
    );
    return hudRecords;
  };

  /**
   * Keep the launch's controller facts and repaint the last live sources.
   *
   * @param {Event} event A COMBAT_CONFIGURATION_EVENT (detail: the configuration).
   */
  const onConfiguration = (event) => {
    controllers = liveControllers(/** @type {CustomEvent} */ (event).detail);
    if (lastLive !== null) drawLive(lastLive.sources, lastLive.renderOptions);
  };

  /**
   * Select a clicked seen portrait (onSelect) or explain a clicked unseen one
   * (onRefuse). An unseen portrait without a refusal is your own Manual
   * champion on the other team: it is selected too, and the host then shows
   * that champion's view. Clicks on a held earlier look never count.
   *
   * @param {Event} event
   */
  const onPortraitClick = (event) => {
    const portrait =
      event.target instanceof Element ? event.target.closest("button.por") : null;
    if (portrait === null || portrait.getAttribute("data-beat-side") === "before")
      return;
    if (portrait.getAttribute("data-state") === "unseen") {
      const sentence = portrait.getAttribute("data-refusal");
      if (sentence) {
        options.onRefuse?.(sentence);
        return;
      }
    }
    const id = portrait.getAttribute("data-public-agent-id");
    if (id !== null) options.onSelect?.(id);
  };

  if (scorebug !== null) {
    page.addEventListener(COMBAT_CONFIGURATION_EVENT, onConfiguration);
    scorebug.addEventListener("click", onPortraitClick);
  }

  return Object.freeze({
    render(/** @type {HudSources | null} */ sources, renderOptions = {}) {
      const doc = frames.ownerDocument;
      if (
        scorebug !== null &&
        (sources === null || sources.product === "combat_debugger")
      ) {
        frames.replaceChildren();
        frames.hidden = true;
        if (sources === null) {
          lastLive = null;
          clearLiveScorebug(scorebug);
          return Object.freeze([]);
        }
        lastLive = { sources, renderOptions };
        return drawLive(sources, renderOptions);
      }
      const hadFocus = roots.some((root) => root.contains(doc.activeElement));
      const hudRecords = records({ ...renderOptions, sources });
      const nodes = build(hudRecords, doc);
      frames.replaceChildren(...nodes);
      frames.hidden = nodes.length === 0;
      // The Replay Viewer's team names are controller lines: fitted now that
      // they are in the page; then the hover lists read the same model.
      for (const name of frames.querySelectorAll(".arena-frames__team-name")) {
        const team = teamNameLines.get(name);
        if (team)
          paintControllerLine(/** @type {HTMLElement} */ (name), team, {
            fitRoot: frames,
          });
      }
      if (sources?.controllerRoster)
        publishControllerRoster(doc, sources.controllerRoster);
      const view = hudRecords.find((record) => record.kind === "hud_view");
      frames.dataset.audience = view?.audience ?? "";
      frames.dataset.product = view?.product ?? "";
      const { current } = settleTabStop();
      if (hadFocus && current) current.focus();
      return hudRecords;
    },
    destroy() {
      for (const root of roots) {
        root.removeEventListener("keydown", onKeyDown);
        root.removeEventListener("focusin", onFocusIn);
      }
      frames.removeEventListener("click", onClick);
      if (scorebug !== null) {
        page.removeEventListener(COMBAT_CONFIGURATION_EVENT, onConfiguration);
        scorebug.removeEventListener("click", onPortraitClick);
      }
    },
  });
}
