/**
 * @file Build and paint the score strip (the scorebug) from an installed
 * authorized presentation.
 *
 * Owns: matchSummaryView (pure labels and numbers from match_summary),
 * matchTeamSide (the recorded spawn side, read by choreography-plan.js for
 * arena drawing in Oracle View), renderMatchSummary (the DOM of the slim
 * strip in #match-scoreboard in the Replay Viewer, or, in live play, the
 * match facts of the vision scorebug #scorebug: scores, "/ 20", the First To
 * bars, "Tick 91 of 300" (in a replay, "of" its last recorded tick), "First
 * to 20", "Respawn Wave in N Ticks" and the
 * side Team A sits on, through the pure liveScorebugView) and
 * hudSourcesFromFrame with
 * publicRosterIdentities and controllerRosterFromFrame (the one controller
 * model of controller-names.js for a frame; the Replay Viewer's team names,
 * the Arena Frames and the live Tactics Turn rail paint from it) (the
 * one place the HUD's party frames read a frame;
 * looks/hud.js draws them from these sources and never reads a frame itself,
 * so the arena looks' import walk never reaches the adapter). main.js calls
 * them once per installed frame, filter change or view change.
 *
 * The Replay Viewer's strip shows, Team A on the left: each team's crest,
 * "Team A" or "Team B", its policy name (its controller line: shown when it
 * fits, else "1 Policy", long names cut with "…", the team's controller list
 * on hover), its score in a fixed box with "/ N"
 * beside it, a First to N bar of segments (N = match_summary.score_threshold)
 * with "+N" when a team goes past it, the team's wave pips (the public respawn
 * clock, when the caller passes the scene's respawn_waves), and in the
 * center "Tick N of E" (simulator_step_count, and E the replay's last recorded
 * tick; see tickLine). With the Scorebug switch off it falls back to a plain text
 * score line. Combat diagnostics show names and the tick, never scores.
 *
 * Momentum (switch hud_momentum, off by default; the V5 Arena Frames pick,
 * broadcast-hud.js segBar): in the Replay Viewer strip and the live scorebug
 * alike, the points a team scored in the last 10 ticks fill in its light
 * color over a Moon Silver underline; older points keep the team color. The
 * baseline is the host's public fact match_summary.scores_ten_ticks_ago
 * (momentumBaselineFromFrame); the browser keeps no score history. With the
 * switch off, or a frame without that fact, every earned point looks the same.
 *
 * Boundary: the strip reads only match_summary, which holds public match
 * facts (task, map, team names, scores, the scores 10 ticks ago, tick), plus
 * what the caller passes in its options (the public wave clock); it never
 * reads the researcher space. The live scorebug reads the same public facts,
 * the same in every view. hudSourcesFromFrame keeps the HUD's Agent POV rules: the
 * audience scene and visual events for what the team sees; the researcher
 * view only for the Replay Viewer's docked full record; identity fields only
 * (never health or positions) to list unseen champions in the BattleClient;
 * match_summary deaths only in Oracle View. It does not infer scores or winners from
 * agents drawn on the battlefield. The score waits for the Fall beat: when a
 * beat view says the score changed this tick, the old number and bar are drawn
 * as "before" nodes (data-beat-hold="score") that the choreographer shows until
 * the Fall beat; the DOM itself always holds the END value. A live repaint
 * with an unchanged scorebug view leaves its DOM alone, so a hold that is
 * running keeps its nodes (renderLiveScorebug).
 */
import { teamLabelFromId } from "./agent-identity.js";
import {
  authorizedPresentationAgentDisplayId,
  authorizedPresentationAudience,
  authorizedPresentationIncomingRows,
  authorizedPresentationMatchBeatFacts,
  authorizedPresentationResearcherSceneView,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import {
  controllerRoster,
  NO_CONTROLLER_ROSTER,
  paintControllerLine,
  publishControllerRoster,
  rosterTeam,
} from "./controller-names.js";
import { createSvgIcon } from "./icons.js";
import { symbolById } from "./symbols.js";
import { createSemanticDescriptor, registerTooltipOwner } from "./tooltip.js";

/** The most segments the First To bar draws; a higher threshold draws one fill. */
export const SCORE_BAR_MAX_SEGMENTS = 40;

/**
 * Read a normalized match's display side, using the historical layout if absent.
 * teamId is 1 or 2. This reads recorded metadata, never current actor positions.
 * The Replay Viewer's score strip does not use it (Team A is always on the
 * left there); arena drawing in Oracle View and the live scorebug's side
 * (liveScorebugView, through matchSummaryView) do.
 * @param {Record<string, any> | null | undefined} match
 * @param {number} teamId
 * @returns {"left" | "right"}
 */
export function matchTeamSide(match, teamId) {
  return match?.teams?.[teamId - 1]?.display_side ?? (teamId === 1 ? "left" : "right");
}

/**
 * Return the last recorded tick of a replay presentation, or null.
 *
 * A replay frame's source names the shown frame (source_frame_index) and the
 * replay's last frame (source_final_frame_index); each recorded frame is one
 * tick later than the one before, so the last tick is the shown tick plus the
 * frames still to come. This is the length the Rune Scrubber shows ("Tick 118
 * of 204"), also for a replay saved part way through a BattleClient game.
 * Live frames (the game's end is not known yet) and frames without these
 * indices give null. Pure.
 *
 * @param {unknown} presentation A normalized presentation frame.
 * @param {number} current The shown tick (match_summary.simulator_step_count).
 * @returns {number | null}
 */
function replayLastTick(presentation, current) {
  const frame = /** @type {Record<string, any>} */ (presentation);
  if (frame?.viewer_mode !== "replay") return null;
  const at = frame.source?.source_frame_index;
  const last = frame.source?.source_final_frame_index;
  if (!Number.isInteger(at) || !Number.isInteger(last) || last < at) return null;
  return current + (last - at);
}

/**
 * Format the tick line from a match summary.
 *
 * match supplies simulator_step_count (the current tick, an integer 0 or more)
 * and optional episode_limit (the tick at which the game ends in a draw).
 * presentation is the frame the summary came from. In the Replay Viewer the
 * line reads "Tick N of E", E being the replay's last recorded tick
 * (replayLastTick: where this game actually ended or was saved). In live play
 * it reads "Tick N of L", L being the draw limit, because the end is not known
 * yet; "Tick N" when neither is known. The draw limit is kept as `limit`
 * either way (the Draw At tip and the time-running-out cards read it). Return
 * null when the tick itself is missing. Pure.
 *
 * @param {Record<string, any>} match
 * @param {unknown} presentation
 * @returns {{current: number, limit: number | null, last: number | null, text: string} | null}
 */
function tickLine(match, presentation) {
  const current = Number.isInteger(match.simulator_step_count)
    ? match.simulator_step_count
    : null;
  if (current === null) return null;
  const limit =
    Number.isInteger(match.episode_limit) && match.episode_limit > 0
      ? match.episode_limit
      : null;
  const last = replayLastTick(presentation, current);
  const shownEnd = last ?? limit;
  return {
    current,
    limit,
    last,
    text: shownEnd === null ? `Tick ${current}` : `Tick ${current} of ${shownEnd}`,
  };
}

/**
 * Build labels from a recognized presentation's match_summary, or return null.
 *
 * presentation may be any input. Unrecognized frames and missing summaries
 * produce null. Recognized records supply the map name/split, Team A/B names,
 * policy/checkpoint references, outcome and the tick. TDM adds scores and the
 * threshold; combat diagnostics omit scores. The returned outer record and team
 * records are frozen; the teams array is newly allocated but not frozen. No DOM
 * or input record is changed. Normalization owns the validity of nested facts.
 * Team labels are "Team A" for the first recorded team and "Team B" for the
 * second, from agent-identity.js (no color words). Only match_summary is
 * read; the scoreboard sits beside the arena and shows public match facts.
 *
 * Fields: task ("TDM" or "Combat diagnostic"), taskMode (0 or 1), map (display
 * name with "(Test Map)" style split, or null), threshold (TDM points to win,
 * or null), tick ({current, limit, last, text} or null; see tickLine: a
 * replay reads "Tick N of" its last recorded tick, live play "of" the draw
 * limit) and teams, in
 * recorded order (Team A first). Each team has teamId, side (the recorded spawn
 * side, "left" or "right"), label, name, score ("6/7" text or null),
 * scoreValue (the number after this tick, or null), result ("VICTORY",
 * "DEFEAT", "DRAW" or null while playing) and provenance (policy IDs and
 * checkpoint digests, one per line).
 *
 * @param {unknown} presentation
 * @returns {Readonly<Record<string, any>> | null} The frozen labels, or null
 *   when no summary is installed.
 */
export function matchSummaryView(presentation) {
  const match = isAuthorizedPresentationFrame(presentation)
    ? presentation.match_summary
    : null;
  if (!match) return null;
  const tdm = match.task_mode === 1;
  return Object.freeze({
    task: tdm ? "TDM" : "Combat diagnostic",
    taskMode: match.task_mode,
    map: match.map
      ? `${match.map.display_name}${match.map.split ? ` (${match.map.split[0].toUpperCase()}${match.map.split.slice(1)} Map)` : ""}`
      : null,
    threshold: tdm ? match.score_threshold : null,
    tick: tickLine(match, presentation),
    teams: match.teams.map(
      (/** @type {Record<string, any>} */ team, /** @type {number} */ index) =>
        Object.freeze({
          teamId: team.team_id,
          side: matchTeamSide(match, team.team_id),
          label: teamLabelFromId(index + 1) ?? `Team ${index + 1}`,
          name: team.display_name,
          score: tdm ? `${match.scores[index]}/${match.score_threshold}` : null,
          scoreValue: tdm ? match.scores[index] : null,
          result:
            match.outcome === "draw"
              ? "DRAW"
              : match.outcome === "team_a_win"
                ? index === 0
                  ? "VICTORY"
                  : "DEFEAT"
                : match.outcome === "team_b_win"
                  ? index === 1
                    ? "VICTORY"
                    : "DEFEAT"
                  : null,
          provenance: [...team.policy_ids, ...team.checkpoint_digests].join("\n"),
        }),
    ),
  });
}

/**
 * @typedef {{
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   beat?: Readonly<{score?: Readonly<{before: unknown, after: unknown, changed: boolean}>}> | null,
 *   waves?: ReadonlyArray<Record<string, any>> | null,
 * }} MatchSummaryRenderOptions
 *   filters is the visual filter state (default: every switch at its default,
 *   so the strip is on and tips are on). Only "scorebug" (false: plain text
 *   score line), "symbol_explanations" (false: no tips) and "hud_momentum"
 *   (true: Momentum marks the points scored in the last 10 ticks, in the
 *   Replay Viewer strip and the live scorebug; absent or false: no
 *   Momentum) are read. beat is
 *   the frame's `beatView(frame, "end", {visualFilters})` result from
 *   choreography-plan.js, or null/absent for no beat holds (the score then
 *   shows its END value at once). waves (default none) is the scene's
 *   respawn_waves list ({team_index 0 or 1, period_steps, countdown_steps});
 *   with it each team's strip gets its wave pips.
 */

/**
 * Replace the supplied scoreboard DOM with the current match strip.
 *
 * elements supplies root (#match-scoreboard), task (the center text node,
 * #match-task) and two team containers, Team A first then Team B. Team A
 * always goes in the first (left) container, whatever its spawn side (plan
 * P12). presentation is passed to matchSummaryView. If it gives null, hide the
 * root, write "Task unavailable" and clear both teams. Otherwise show the strip
 * (see the file description) or, with options.filters.scorebug false, today's
 * plain text line ("Task mode: TDM · Map: …", team label, name, "Score: 6/7",
 * result). In the strip each team's score is a `.match-scoreboard__score`
 * element whose text is the END score only; on a scoring tick a separate
 * "before" node with the old score and bar is added for the Fall beat hold.
 * Policy references become the team label's title tooltip. Each team's
 * name is its controller line from the one controller model
 * (controllerRosterFromFrame, published for the hover lists with
 * publishControllerRoster): the name when it fits, cut with "…" past the
 * fixed name width, "1 Policy" when it does not fit, and hovering it drops
 * down the team's list ("Team A Controllers") with its champions.
 *
 * Every client (the BattleClient, the DevClient combat screen and the Replay
 * Viewer): when the page has the vision scorebug (#scorebug), the strip is
 * not drawn. The root is hidden and the task and team containers are
 * emptied. The scorebug is shown (index.html keeps it hidden until the
 * first call), and the scores, bars, tick, First To, wave line and side go
 * into it (liveScorebugView, then renderLiveScorebug; see there for the Fall beat
 * hold). options.waves,
 * options.filters (only "scorebug" and "hud_momentum") and
 * options.beat.score are read. With
 * "hud_momentum" true the bars mark the points scored since the host's
 * match_summary.scores_ten_ticks_ago (momentumBaselineFromFrame), in the
 * strip and the live scorebug alike. looks/hud.js fills its hosts and
 * portraits.
 *
 * Side effects: replaces the children of the team containers and the task
 * node, toggles root.hidden and the "match-strip" class, sets data-team and
 * data-side on the containers, and registers shared tooltips (tooltip.js) on
 * the crest, score, bar and tick when Symbol Explanations is on. Those strip
 * marks carry data-hud-symbol and tabindex -1 so the HUD's one Tab stop can
 * move to them with the arrow keys. The team names are fitted to their real
 * width now and again when the strip's width changes (controller-names.js;
 * one ResizeObserver and the hover list's listeners per document). With a
 * scorebug it changes #scorebug's DOM and hidden flag instead (no tooltips).
 * It keeps no state
 * between calls. No other timers, no storage or network. The caller
 * must supply matching containers; DOM shape is not validated. Agent POV: the
 * strip shows only public match facts, so it is the same in every view.
 *
 * @param {{root: HTMLElement, task: HTMLElement, teams: HTMLElement[]}} elements
 * @param {unknown} presentation
 * @param {MatchSummaryRenderOptions} [options] Defaults to {}.
 */
export function renderMatchSummary(elements, presentation, options = {}) {
  const summary = matchSummaryView(presentation);
  const scorebug = liveScorebugRoot(elements.root);
  if (scorebug !== null) {
    // Every client draws the vision scorebug; the old strip stays empty.
    scorebug.hidden = false;
    elements.root.hidden = true;
    elements.root.classList.remove("match-strip");
    elements.task.replaceChildren();
    for (const team of elements.teams) team.replaceChildren();
    const beatScore = options.beat?.score ?? null;
    renderLiveScorebug(
      scorebug,
      liveScorebugView(summary, {
        waves: options.waves ?? null,
        filters: options.filters ?? null,
        beforeScores:
          beatScore?.changed === true && Array.isArray(beatScore.before)
            ? beatScore.before
            : null,
        recentBaseline: momentumBaselineFromFrame(presentation, options.filters),
      }),
    );
    return;
  }
  elements.root.hidden = summary === null;
  if (summary === null) {
    elements.root.classList.remove("match-strip");
    elements.task.textContent = "Task unavailable";
    for (const team of elements.teams) team.replaceChildren();
    return;
  }
  const filters = options.filters ?? null;
  const strip = filters?.scorebug !== false;
  const tips = filters?.symbol_explanations !== false;
  elements.root.classList.toggle("match-strip", strip);
  elements.root.dataset.taskMode = String(summary.taskMode);
  if (strip) {
    renderStripTask(elements.task, summary, tips);
  } else {
    elements.task.textContent = `Task mode: ${summary.task}${summary.map ? ` · Map: ${summary.map}` : ""}`;
  }
  const beatScore = options.beat?.score ?? null;
  const momentum = momentumBaselineFromFrame(presentation, filters);
  // The one controller model; the team names below and the Team Rosters
  // strip (looks/hud.js) both paint from it, and its hover lists read it.
  const roster = controllerRosterFromFrame(presentation, null);
  summary.teams.forEach(
    (/** @type {Record<string, any>} */ team, /** @type {number} */ index) => {
      const root = elements.teams[team.teamId === 1 ? 0 : 1] ?? elements.teams[index];
      if (!root) return;
      const controllers = rosterTeam(roster, team.teamId);
      root.dataset.team = String(team.teamId);
      root.dataset.side = team.side;
      root.classList.toggle("match-scoreboard__team--a", team.teamId === 1);
      root.classList.toggle("match-scoreboard__team--b", team.teamId === 2);
      if (strip) {
        const before =
          beatScore?.changed === true && Array.isArray(beatScore.before)
            ? beatScore.before[index]
            : null;
        const waves = Array.isArray(options.waves) ? options.waves : [];
        renderStripTeam(root, team, summary.threshold, {
          before:
            Number.isInteger(before) && before !== team.scoreValue ? before : null,
          recentBaseline: momentum?.[team.teamId - 1] ?? null,
          tips,
          wave:
            waves.find(
              (/** @type {Record<string, any>} */ wave) =>
                wave?.team_index === team.teamId - 1,
            ) ?? null,
        });
      } else {
        renderPlainTeam(root, team);
      }
      paintTeamName(root, controllers, elements.root);
    },
  );
  if (elements.root.ownerDocument)
    publishControllerRoster(elements.root.ownerDocument, roster);
}

/**
 * Paint a Replay Viewer team's name (span.match-scoreboard__policy, already
 * in the page) as its controller line: the name when it fits, cut with "…"
 * past the fixed name width, "1 Policy" when even the cut name does not fit,
 * and the team's full list on hover (paintControllerLine). A team the roster
 * has no controllers for keeps the recorded name as written (CSS still cuts
 * it with "…"). Side effects: the name's DOM and attributes.
 *
 * @param {HTMLElement} root One team's strip or plain line.
 * @param {import("./controller-names.js").ControllerTeam | null} team
 * @param {HTMLElement} fitRoot The whole strip (#match-scoreboard).
 * @returns {void}
 */
function paintTeamName(root, team, fitRoot) {
  if (team === null || typeof root.querySelector !== "function") return;
  const name = /** @type {HTMLElement | null} */ (
    root.querySelector(".match-scoreboard__policy")
  );
  if (name !== null && team.controllers.length > 0)
    paintControllerLine(name, team, { fitRoot });
}

/**
 * The name a saved game records for a champion a person played. Every client
 * labels a Manual slot "Manual" (normalize_slot_ownership in
 * _battle_client/model.py requires it), and that label is what the replay
 * keeps as the slot's controller name.
 */
const RECORDED_MANUAL_NAME = "Manual";

/**
 * Build the one shared controller model (controllerRoster in
 * controller-names.js) for one installed presentation.
 *
 * Live play (BattleClient and DevClient; combatConfiguration is the transport
 * frame's combat_configuration): manual_slots and slot_labels name who plays
 * each command slot, and the identity directory (publicRosterIdentities, with
 * each champion's shown Agent ID) says which champions exist. The live
 * scorebug (looks/hud.js) builds the same model from the same two facts
 * (the configuration event's manual_slots and slot_labels, and
 * hudSourcesFromFrame's identities and displayIdOf), so every live surface
 * shows the same controllers.
 *
 * Replay Viewer (product_kind "replay_viewer"; combatConfiguration ignored):
 * each champion is played by its recorded controller name
 * (match_summary.slot_controller_names, ten names in slot order, exactly as
 * recorded), the same per-champion list the live clients showed. A Manual
 * slot reads "Manual", never "You": no slot is the viewer's own here. Each
 * slot recorded as RECORDED_MANUAL_NAME is passed as recordedManualSlots,
 * so its controller is flagged manual: it is listed first, as "You" is in
 * live play, and the short line counts it as the person, not as a policy
 * ("Manual + 3 Policies"). The replay keeps only the name, so a policy
 * recorded under the exact name "Manual" is flagged the same way. Old
 * replays without slot names fall back to the recorded team names
 * (match_summary.teams[].display_name), which then play every champion of
 * their team. The champions come from the identity directory, else the
 * audience's researcher scene, else its arena scene (identity fields only:
 * team, slot, class, shown ID).
 *
 * Pure: reads only identity fields and the match setup, never life, place
 * or status, so Agent POV and Oracle View give the same model. Returns
 * NO_CONTROLLER_ROSTER for anything that is not an authorized presentation.
 *
 * @param {unknown} presentation The installed authorized presentation.
 * @param {unknown} combatConfiguration Live play's
 *   transport.combat_configuration; null or anything else in the Replay
 *   Viewer.
 * @returns {import("./controller-names.js").ControllerRoster}
 */
export function controllerRosterFromFrame(presentation, combatConfiguration) {
  if (!isAuthorizedPresentationFrame(presentation)) return NO_CONTROLLER_ROSTER;
  /** @param {string} id */
  const shown = (id) => authorizedPresentationAgentDisplayId(presentation, id);
  const identities = publicRosterIdentities(presentation).map((identity) => ({
    ...identity,
    displayId: shown(identity.publicAgentId),
  }));
  if (field(presentation, "product_kind") !== "replay_viewer") {
    return controllerRoster({
      manualSlots: field(combatConfiguration, "manual_slots") ?? null,
      slotLabels: field(combatConfiguration, "slot_labels") ?? null,
      champions: identities,
    });
  }
  const teams = field(presentation.match_summary, "teams");
  const slotNames = field(presentation.match_summary, "slot_controller_names");
  const scene =
    authorizedPresentationResearcherSceneView(presentation) ??
    authorizedPresentationSceneView(presentation);
  const sceneAgents = Array.isArray(scene?.agents) ? scene.agents : [];
  return controllerRoster({
    // Recorded per-champion names win; old replays have none (null).
    slotLabels: Array.isArray(slotNames) ? slotNames : null,
    recordedManualSlots: Array.isArray(slotNames)
      ? slotNames.flatMap((name, slot) => (name === RECORDED_MANUAL_NAME ? [slot] : []))
      : null,
    teamNames: [1, 2].map((teamId) =>
      Array.isArray(teams)
        ? field(
            teams.find((team) => field(team, "team_id") === teamId),
            "display_name",
          )
        : null,
    ),
    champions:
      identities.length > 0
        ? identities
        : sceneAgents.map((agent) => ({
            publicAgentId: field(agent, "public_agent_id") ?? null,
            teamId: field(agent, "team_id"),
            teamLocalSlot: field(agent, "team_local_slot"),
            classId: field(agent, "class_id"),
            displayId:
              field(agent, "display_agent_id") ??
              (typeof field(agent, "public_agent_id") === "string"
                ? shown(field(agent, "public_agent_id"))
                : null),
          })),
  });
}

/**
 * Most segments one team's live First To bar draws (the vision draws 20 for
 * First to 20). A larger threshold draws this many, each worth
 * threshold / LIVE_SEGMENT_LIMIT points.
 */
export const LIVE_SEGMENT_LIMIT = 40;

/**
 * How many ticks back Momentum looks: a point scored in the last 10 ticks is
 * "recent" (the store's Arena Frames bar, broadcast-hud.js momentumFrom).
 * The host owns the window and the history (MOMENTUM_TICKS in
 * viewer/match_summary.py, sent as match_summary.scores_ten_ticks_ago); this
 * copy only writes the number into the words, such as the tip's "Last 10
 * Ticks" row.
 */
export const MOMENTUM_TICKS = 10;

/**
 * Return the scores Momentum compares against, or null for no Momentum.
 *
 * presentation is the installed presentation; filters is the visual filter
 * state (null or absent: every switch at its default, so Momentum is off).
 * Returns the frozen [Team A, Team B] match_summary.scores_ten_ticks_ago
 * when filters.hud_momentum is true, the frame is an authorized presentation
 * of a TDM game (task_mode 1) and the field holds two whole numbers, each 0
 * or more and at most that team's score. Returns null with the switch off,
 * for a combat diagnostic, for a frame from before the field existed (null
 * or absent) and for a malformed pair, so no point is shown as recent. Each
 * team's points above its baseline are the recent ones.
 *
 * Agent POV: scores are public match facts, the same in Agent POV and Oracle
 * View, so the result never depends on the view and says nothing about any
 * champion. Pure: reads match_summary only, keeps nothing between calls.
 *
 * @param {unknown} presentation
 * @param {Readonly<Record<string, boolean>> | null | undefined} filters
 * @returns {readonly [number, number] | null}
 */
export function momentumBaselineFromFrame(presentation, filters) {
  if (filters?.hud_momentum !== true) return null;
  const match = isAuthorizedPresentationFrame(presentation)
    ? presentation.match_summary
    : null;
  if (match?.task_mode !== 1) return null;
  const baseline = match.scores_ten_ticks_ago;
  const scores = match.scores;
  if (
    !Array.isArray(baseline) ||
    !Array.isArray(scores) ||
    baseline.length !== 2 ||
    scores.length !== 2 ||
    ![0, 1].every(
      (index) =>
        Number.isInteger(baseline[index]) &&
        Number.isInteger(scores[index]) &&
        baseline[index] >= 0 &&
        baseline[index] <= scores[index],
    )
  )
    return null;
  return /** @type {readonly [number, number]} */ (
    Object.freeze([baseline[0], baseline[1]])
  );
}

/**
 * @typedef {Readonly<{
 *   teamId: 1 | 2,
 *   key: "a" | "b",
 *   value: number | null,
 *   scoreText: string,
 *   goalText: string,
 *   segments: number,
 *   lit: number,
 *   recentFrom: number,
 *   before: Readonly<{scoreText: string, lit: number}> | null,
 * }>} LiveScoreTeam
 *   One team's score in the live scorebug. teamId 1 is Team A and key "a"
 *   names its elements (#score-a, #seg-a). value is the END score after this
 *   tick (null for a combat diagnostic, which has no score). scoreText is what
 *   #score-a shows: the number ("2"), or with the Scorebug switch off the plain
 *   line "Score: 2/20"; "" with no score. goalText is "/ 20" ("" when plain or
 *   with no threshold). segments is how many bar segments to draw (0 when
 *   plain or with no threshold) and lit how many of them are earned.
 *   recentFrom is Momentum's first recent segment: segments recentFrom to
 *   lit - 1 were earned in the last 10 ticks (MOMENTUM_TICKS). It equals
 *   segments (no recent segment) with Momentum off, with no known earlier
 *   score, in the plain score or with no threshold. before is the score
 *   before this tick's Fall beat, {scoreText, lit} in the same form, when the
 *   score changed this tick; else null. The held old bar marks segments
 *   recentFrom to before.lit - 1 as recent.
 */

/**
 * @typedef {Readonly<{
 *   available: boolean,
 *   teamASide: "left" | "right",
 *   plain: boolean,
 *   teams: ReadonlyArray<LiveScoreTeam>,
 *   tickText: string,
 *   firstToText: string,
 *   waveText: string,
 * }>} LiveScorebugView
 *   What the live scorebug (#scorebug) shows; see liveScorebugView.
 */

/**
 * Work out the live scorebug's scores, middle and side. Pure: no DOM.
 *
 * The live scorebug is the BattleClient's (and the DevClient combat
 * screen's) top strip from the approved vision page. This function gives its
 * match facts; looks/hud.js fills its hosts and portraits.
 *
 * summary is matchSummaryView(presentation), or null when no match summary is
 * installed (then available is false and every text is ""). options (default
 * {}) holds:
 * - waves: the scene's respawn_waves rows ({team_index 0 or 1,
 *   countdown_steps}), or null/absent for no wave line.
 * - filters: the visual filter state, or null for the defaults. Only
 *   "scorebug" is read: false gives the plain text score ("Score: 2/20") with
 *   no "/ 20" and no bar (the switch's own help text).
 * - beforeScores: [Team A, Team B] scores before this tick, when the frame's
 *   beat view says the score changed this tick (beatView(...).score.before);
 *   null/absent for no Fall beat hold.
 * - recentBaseline: Momentum (switch hud_momentum, off by default). The
 *   [Team A, Team B] scores 10 ticks ago (momentumBaselineFromFrame, the
 *   host's match_summary.scores_ten_ticks_ago), or null/absent for Momentum
 *   off or unknown. Each team's segments earned since then are recent
 *   (recentFrom); a baseline at or above the score gives none.
 *
 * Returns a frozen LiveScorebugView:
 * - teamASide: the side of Team A's spawn end ("left" or "right", from the
 *   recorded display_side), so the scorebug matches the field. The vision
 *   puts each team on the side of its own Red Zone; paired games swap ends,
 *   so the sides swap too.
 * - teams: Team A then Team B (see LiveScoreTeam). Scores at or above the
 *   threshold light every segment.
 * - tickText: "Tick 91 of 300" live, or "Tick 91 of 156" in a replay that
 *   ended at tick 156 (summary.tick.text; "" when unknown).
 * - firstToText: "First to 20" ("" for a combat diagnostic).
 * - waveText: "Respawn Wave in 4 Ticks": the next wave of either team. Core
 *   brings champions back on the transition that starts with a countdown of
 *   0, so the wave comes in countdown + 1 ticks, the same number a fallen
 *   champion's "Returns In" uses ("1 Tick" for one). "" with no wave rows.
 *
 * Agent POV: everything here is a public match fact (scores, threshold, tick,
 * draw tick, spawn sides and the public wave clocks), the same number in Agent
 * POV and Oracle View, so it reveals no champion the POV team cannot see.
 *
 * @param {Readonly<Record<string, any>> | null} summary
 * @param {{
 *   waves?: ReadonlyArray<Record<string, any>> | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   beforeScores?: ReadonlyArray<unknown> | null,
 *   recentBaseline?: ReadonlyArray<number> | null,
 * }} [options] Defaults to {}.
 * @returns {LiveScorebugView}
 */
export function liveScorebugView(summary, options = {}) {
  const plain = options.filters?.scorebug === false;
  if (summary === null) {
    return Object.freeze({
      available: false,
      teamASide: "right",
      plain,
      teams: Object.freeze([]),
      tickText: "",
      firstToText: "",
      waveText: "",
    });
  }
  const threshold = Number.isInteger(summary.threshold) ? summary.threshold : null;
  const segments =
    plain || threshold === null || threshold <= 0
      ? 0
      : Math.min(threshold, LIVE_SEGMENT_LIMIT);
  /** @param {number} value */
  const litFor = (value) =>
    segments === 0
      ? 0
      : Math.floor(
          (segments * Math.min(Math.max(0, value), threshold ?? 0)) / (threshold ?? 1),
        );
  /** @param {number} value */
  const scoreTextFor = (value) =>
    plain ? `Score: ${value}/${threshold}` : String(value);
  const teams = [1, 2].flatMap((teamId) => {
    const team = summary.teams.find(
      (/** @type {Record<string, any>} */ entry) => entry.teamId === teamId,
    );
    if (team === undefined) return [];
    const value = Number.isInteger(team.scoreValue) ? team.scoreValue : null;
    const old = options.beforeScores?.[teamId - 1];
    const baseline = options.recentBaseline?.[teamId - 1];
    const before =
      value !== null && threshold !== null && Number.isInteger(old) && old !== value
        ? Object.freeze({
            scoreText: scoreTextFor(/** @type {number} */ (old)),
            lit: litFor(/** @type {number} */ (old)),
          })
        : null;
    return [
      Object.freeze({
        teamId: /** @type {1 | 2} */ (teamId),
        key: /** @type {"a" | "b"} */ (teamId === 1 ? "a" : "b"),
        value,
        scoreText: value === null || threshold === null ? "" : scoreTextFor(value),
        goalText: plain || threshold === null ? "" : `/ ${threshold}`,
        segments,
        lit: value === null ? 0 : litFor(value),
        recentFrom:
          value === null || !Number.isInteger(baseline)
            ? segments
            : Math.min(segments, litFor(/** @type {number} */ (baseline))),
        before,
      }),
    ];
  });
  const countdowns = (Array.isArray(options.waves) ? options.waves : [])
    .map((wave) => wave?.countdown_steps)
    .filter((value) => Number.isInteger(value) && value >= 0);
  const ticks = countdowns.length === 0 ? null : Math.min(...countdowns) + 1;
  const teamA = summary.teams.find(
    (/** @type {Record<string, any>} */ entry) => entry.teamId === 1,
  );
  return Object.freeze({
    available: true,
    teamASide: teamA?.side === "left" ? "left" : "right",
    plain,
    teams: Object.freeze(teams),
    tickText: summary.tick?.text ?? "",
    firstToText: threshold === null ? "" : `First to ${threshold}`,
    waveText:
      ticks === null
        ? ""
        : `Respawn Wave in ${ticks} ${ticks === 1 ? "Tick" : "Ticks"}`,
  });
}

/**
 * Return the page's scorebug (#scorebug), else null.
 *
 * anchor is any element of the page (its document is searched). Every client
 * draws the BattleClient vision's scorebug: the BattleClient, the DevClient
 * combat screen and, since the Replay Viewer took the BattleClient's look
 * (owner, 3 Oct 2026: one score display at the top), the Replay Viewer too.
 * A document without an element with id "scorebug", or without
 * getElementById (unit tests with a fake DOM), gives null, so the old strip
 * in #match-scoreboard is drawn instead. Pure apart from the DOM lookup.
 *
 * @param {HTMLElement} anchor
 * @returns {HTMLElement | null}
 */
function liveScorebugRoot(anchor) {
  const doc = anchor.ownerDocument;
  if (typeof doc?.getElementById !== "function") return null;
  return doc.getElementById("scorebug");
}

/**
 * The last live scorebug view painted into each #scorebug, as its JSON text
 * (renderLiveScorebug). A repaint with an equal view leaves the DOM alone, so
 * a running Fall beat hold keeps its nodes. Weak: a removed scorebug is
 * forgotten with it.
 *
 * @type {WeakMap<HTMLElement, string>}
 */
const paintedLiveScorebugs = new WeakMap();

/**
 * Paint the live scorebug's match facts into its fixed markup.
 *
 * scorebug is #scorebug from index.html, holding .sb-score.a and .sb-score.b
 * (each with #score-a or #score-b, span.of and #seg-a or #seg-b) and .sb-mid
 * (b, a span and span.wave). view is liveScorebugView's result. It writes:
 * data-team-a-side ("left" mirrors the scorebug in scorebug.css) and
 * data-plain ("true" with the Scorebug switch off) on the scorebug; each
 * score's text, "/ 20" and bar (span.segbar children `i`, class "on" when
 * earned, plus "recent" from the team's recentFrom on and "newest" on the
 * last recent one, for Momentum); the tick, First To and wave lines (an
 * empty line is hidden).
 *
 * Fall beat (owner, 1 Oct 2026: the score waits for the Fall beat, like the
 * life bars): when a team's score changed this tick, its #score element and
 * bar become the "after" nodes of a "score" beat hold (opacity 0 until the
 * beat), and a "before" copy of the old number (span.n before #score) and of
 * the old bar (span.segbar after the bar) is drawn on top until the beat
 * (data-beat-hold "score", data-beat-side "before", opacity 0 at END,
 * aria-hidden). The choreographer times these only when #scorebug is one of
 * its beatHoldRoots; otherwise the END score shows at once.
 *
 * Repaints: a call whose view equals the last view painted into this
 * scorebug (same JSON, so the same tick, scores, holds and lines) changes
 * nothing. The page repaints a tick while it plays, for example when a new
 * server revision of the same tick arrives as the replay recording moves on.
 * Rebuilding the "before" nodes then would drop the choreographer's running
 * hold: the old number would vanish at once while the new one still waits
 * for its beat, so the score showed no number for about a second (hand play,
 * 3 Oct 2026). Any other view rebuilds: earlier "before" nodes and hold marks
 * are removed first.
 *
 * Side effects: DOM inside scorebug only, and the last painted view per
 * scorebug (paintedLiveScorebugs). Missing parts are skipped. No timers,
 * storage, network or tooltips.
 *
 * @param {HTMLElement} scorebug
 * @param {LiveScorebugView} view
 * @returns {void}
 */
function renderLiveScorebug(scorebug, view) {
  const painted = JSON.stringify(view);
  if (paintedLiveScorebugs.get(scorebug) === painted) return;
  paintedLiveScorebugs.set(scorebug, painted);
  const doc = scorebug.ownerDocument;
  scorebug.dataset.teamASide = view.teamASide;
  scorebug.dataset.plain = String(view.plain);
  for (const key of /** @type {const} */ (["a", "b"])) {
    const box = scorebug.querySelector(`.sb-score.${key}`);
    const score = box?.querySelector(`#score-${key}`);
    const goal = box?.querySelector(".of");
    const bar = box?.querySelector(`#seg-${key}`);
    if (
      !box ||
      !(score instanceof HTMLElement) ||
      !goal ||
      !(bar instanceof HTMLElement)
    )
      continue;
    for (const old of box.querySelectorAll('[data-beat-side="before"]')) old.remove();
    clearLiveHold(score);
    clearLiveHold(bar);
    const team = view.teams.find((entry) => entry.key === key) ?? null;
    score.textContent = team?.scoreText ?? "";
    goal.textContent = team?.goalText ?? "";
    const recentFrom = team?.recentFrom ?? team?.segments ?? 0;
    bar.replaceChildren(
      ...segmentNodes(doc, team?.segments ?? 0, team?.lit ?? 0, recentFrom),
    );
    if (team?.before == null) continue;
    const oldScore = doc.createElement("span");
    oldScore.className = "n num";
    oldScore.textContent = team.before.scoreText;
    holdBefore(oldScore, "score");
    score.before(oldScore);
    const oldBar = doc.createElement("span");
    oldBar.className = "segbar";
    oldBar.replaceChildren(
      ...segmentNodes(doc, team.segments, team.before.lit, recentFrom),
    );
    holdBefore(oldBar, "score");
    bar.after(oldBar);
    for (const node of [score, bar]) {
      node.dataset.beatHold = "score";
      node.dataset.beatSide = "after";
    }
  }
  const middle = scorebug.querySelector(".sb-mid");
  const lines = [
    [middle?.querySelector("b"), view.tickText],
    [middle?.querySelector("span:not(.wave)"), view.firstToText],
    [middle?.querySelector("span.wave"), view.waveText],
  ];
  for (const [node, text] of lines) {
    if (!(node instanceof HTMLElement)) continue;
    node.textContent = /** @type {string} */ (text);
    node.hidden = text === "";
  }
}

/**
 * Build count bar segments (`i` elements), the first lit of them with class
 * "on".
 *
 * Momentum: earned segments from recentFrom on (points scored in the last
 * 10 ticks) also get class "recent", and the newest of them (index lit - 1)
 * gets "newest" too, so scorebug.css can draw them in the team's light color
 * over one unbroken Moon Silver underline. recentFrom at or above lit marks
 * none. Creates nodes only. Returns a new array.
 *
 * @param {Document} doc
 * @param {number} count Segments to draw, 0 or more.
 * @param {number} lit Earned segments, 0 to count.
 * @param {number} recentFrom First recent segment; count or more for none.
 * @returns {HTMLElement[]}
 */
function segmentNodes(doc, count, lit, recentFrom) {
  return Array.from({ length: count }, (_, index) => {
    const segment = doc.createElement("i");
    if (index < lit) segment.className = "on";
    if (index < lit && index >= recentFrom) {
      segment.classList.add("recent");
      if (index === lit - 1) segment.classList.add("newest");
    }
    return segment;
  });
}

/**
 * Remove the "score" beat-hold marks a live score or bar got on an earlier
 * scoring tick, so it shows at once. Changes node only.
 *
 * @param {HTMLElement} node
 * @returns {void}
 */
function clearLiveHold(node) {
  delete node.dataset.beatHold;
  delete node.dataset.beatSide;
}

/**
 * Write today's plain text team line (Scorebug switch off) into root.
 *
 * Children: strong.match-scoreboard__team-label ("Team A", with the
 * provenance as its title), span .match-scoreboard__policy (the name; the
 * caller then paints it as the controller line), then for TDM
 * strong.match-scoreboard__score ("Score: 6/7") and any final result.
 * Replaces root's children; nothing else changes.
 *
 * @param {HTMLElement} root
 * @param {Record<string, any>} team One team of matchSummaryView.
 */
function renderPlainTeam(root, team) {
  const doc = root.ownerDocument;
  const title = doc.createElement("strong");
  title.className = "match-scoreboard__team-label";
  title.textContent = team.label;
  // The provenance sits on the label: hovering the name opens the team's
  // controller list instead (paintTeamName).
  title.title = team.provenance;
  const name = doc.createElement("span");
  name.className = "match-scoreboard__policy";
  name.textContent = team.name;
  root.replaceChildren(title, name);
  if (team.score !== null) {
    const score = doc.createElement("strong");
    score.className = "match-scoreboard__score";
    score.textContent = `Score: ${team.score}`;
    root.append(score);
  }
  appendResult(root, team);
}

/**
 * Append the final result plate ("VICTORY", "DEFEAT" or "DRAW") when known.
 *
 * @param {HTMLElement} root
 * @param {Record<string, any>} team One team of matchSummaryView.
 */
function appendResult(root, team) {
  if (team.result === null) return;
  const result = root.ownerDocument.createElement("strong");
  result.className = `match-scoreboard__result match-scoreboard__result--${team.result.toLowerCase()}`;
  result.textContent = team.result;
  root.append(result);
}

/**
 * Write the center of the strip: the tick line and the map name.
 *
 * task receives span.match-strip__tick ("Tick 118 of 204", a HUD symbol with
 * the Tick tip) and span.match-strip__map (the map name, muted), or "Tick
 * unavailable" when the summary has no tick. Replaces task's children.
 *
 * @param {HTMLElement} task
 * @param {Readonly<Record<string, any>>} summary From matchSummaryView.
 * @param {boolean} tips True to register the Tick tip.
 */
function renderStripTask(task, summary, tips) {
  const doc = task.ownerDocument;
  const tick = doc.createElement("span");
  tick.className = "match-strip__tick";
  tick.textContent = summary.tick?.text ?? "Tick unavailable";
  markSymbol(tick, "tick", tips, "tick_clock", [
    ["Tick", summary.tick?.text ?? "Unavailable"],
    ...(summary.tick?.limit != null
      ? [["Draw At", `Tick ${summary.tick.limit}, if nobody reached the winning score`]]
      : []),
  ]);
  const children = [tick];
  if (summary.map) {
    const map = doc.createElement("span");
    map.className = "match-strip__map";
    map.textContent = summary.map;
    children.push(map);
  }
  task.replaceChildren(...children);
}

/**
 * Write one team's strip content into root.
 *
 * Order in the DOM (Team B is mirrored by CSS): crest, label and name, score box, the
 * goal ("/ 20", aria-hidden, since the score's label already says "of 20"),
 * First To bar, wave pips, "+N" overshoot, result. For a combat diagnostic
 * only the crest, label and name are drawn. options.before is the
 * score before this tick when it changed (else null); it adds the Fall-beat
 * "before" nodes. options.recentBaseline (default null) is Momentum: this
 * team's score 10 ticks ago (momentumBaselineFromFrame), or null with the
 * switch off or the fact unknown; with it the bar marks the points above it
 * as recent (scoreBar), the held "before" bar too, and the score and bar
 * tips gain a "Last 10 Ticks" row ("2 points, the bright segments" or "No
 * points"). options.wave is this team's respawn_waves row or null (no
 * pips).
 *
 * @param {HTMLElement} root
 * @param {Record<string, any>} team One team of matchSummaryView.
 * @param {number | null} threshold Points to win, or null for a diagnostic.
 * @param {{
 *   before: number | null,
 *   recentBaseline?: number | null,
 *   tips: boolean,
 *   wave?: Record<string, any> | null,
 * }} options
 */
function renderStripTeam(root, team, threshold, options) {
  const doc = root.ownerDocument;
  const crest = createSvgIcon(doc, team.teamId === 1 ? "team-a" : "team-b", {
    className: "match-strip__crest",
  });
  const crestBox = doc.createElement("span");
  crestBox.className = "match-strip__crest-box";
  crestBox.append(crest);
  markSymbol(
    crestBox,
    `crest:${team.teamId}`,
    options.tips,
    team.teamId === 1 ? "team_a" : "team_b",
    [
      ["Team", team.label],
      ["Policy", team.name],
    ],
  );
  const names = doc.createElement("span");
  names.className = "match-strip__names";
  const title = doc.createElement("strong");
  title.className = "match-scoreboard__team-label";
  title.textContent = team.label;
  // The provenance sits on the label: hovering the name opens the team's
  // controller list instead (paintTeamName).
  title.title = team.provenance;
  const name = doc.createElement("span");
  name.className = "match-scoreboard__policy";
  name.textContent = team.name;
  names.append(title, name);
  root.replaceChildren(crestBox, names);
  if (team.scoreValue === null || threshold === null) {
    appendResult(root, team);
    return;
  }
  const value = /** @type {number} */ (team.scoreValue);
  const scoreBox = doc.createElement("span");
  scoreBox.className = "match-strip__score-box";
  const score = doc.createElement("strong");
  score.className = "match-scoreboard__score";
  score.textContent = String(value);
  score.setAttribute("aria-label", `Score ${value} of ${threshold}`);
  scoreBox.append(score);
  const recentBaseline = options.recentBaseline ?? null;
  const barBox = doc.createElement("span");
  barBox.className = "match-strip__bar-box";
  barBox.append(scoreBar(doc, value, threshold, "match-strip__bar", recentBaseline));
  if (options.before !== null) {
    const before = doc.createElement("span");
    before.className = "match-strip__score-before";
    before.textContent = String(options.before);
    holdBefore(before, "score");
    scoreBox.append(before);
    const beforeBar = scoreBar(
      doc,
      options.before,
      threshold,
      "match-strip__bar match-strip__bar--before",
      recentBaseline,
    );
    holdBefore(beforeBar, "score");
    barBox.append(beforeBar);
  }
  const recentPoints = recentBaseline === null ? null : value - recentBaseline;
  const scoreRows = [
    ["Score", `${value} of ${threshold}`],
    ["To Win", `${Math.max(0, threshold - value)} more`],
    ...(recentPoints === null
      ? []
      : [
          [
            `Last ${MOMENTUM_TICKS} Ticks`,
            recentPoints <= 0
              ? "No points"
              : `${recentPoints} ${recentPoints === 1 ? "point" : "points"}, the bright segments`,
          ],
        ]),
  ];
  markSymbol(scoreBox, `score:${team.teamId}`, options.tips, "score", scoreRows);
  markSymbol(barBox, `bar:${team.teamId}`, options.tips, "score", scoreRows);
  const goal = doc.createElement("span");
  goal.className = "match-strip__goal";
  goal.setAttribute("aria-hidden", "true");
  goal.textContent = `/ ${threshold}`;
  root.append(scoreBox, goal, barBox);
  const pips = wavePips(doc, options.wave ?? null);
  if (pips !== null) root.append(pips);
  if (threshold > 0 && value > threshold) {
    const over = doc.createElement("span");
    over.className = "match-strip__overshoot";
    over.textContent = `+${value - threshold}`;
    root.append(over);
  }
  appendResult(root, team);
}

/**
 * Build one team's wave pips: the public respawn clock under its First To bar.
 *
 * wave is the team's scene respawn_waves row (period_steps and countdown_steps,
 * whole numbers) or null. The row of pips has one pip per tick of the period
 * (1 to 10; period_steps outside that range or missing gives no pips). Lit pips
 * count the ticks gone toward the next wave: period - min(countdown, period - 1),
 * so a countdown of 0 (the wave comes next tick) lights them all. Returns a new
 * `span.match-strip__waves` (role "img", aria-label "Next Wave in N Ticks",
 * N = countdown + 1) of `span.match-strip__wave` children, lit ones with class
 * "is-lit", or null. The wave clock is a public fact, the same in Oracle View
 * and Agent POV. Creates DOM nodes only.
 *
 * @param {Document} doc
 * @param {Record<string, any> | null} wave
 * @returns {HTMLElement | null}
 */
function wavePips(doc, wave) {
  const period = Number(wave?.period_steps);
  const countdown = Number(wave?.countdown_steps);
  if (
    !Number.isInteger(period) ||
    period < 1 ||
    period > 10 ||
    !Number.isInteger(countdown) ||
    countdown < 0
  ) {
    return null;
  }
  const lit = period - Math.min(countdown, period - 1);
  const row = doc.createElement("span");
  row.className = "match-strip__waves";
  row.setAttribute("role", "img");
  const ticks = countdown + 1;
  row.setAttribute(
    "aria-label",
    `Next Wave in ${ticks} ${ticks === 1 ? "Tick" : "Ticks"}`,
  );
  for (let index = 0; index < period; index += 1) {
    const pip = doc.createElement("span");
    pip.className = index < lit ? "match-strip__wave is-lit" : "match-strip__wave";
    row.append(pip);
  }
  return row;
}

/**
 * Build one First To bar: up to SCORE_BAR_MAX_SEGMENTS segments, lit up to
 * value (capped at threshold), or a single fill for a larger threshold.
 *
 * Returns a new span (role "img", aria-label "6 of 20 points to win") with
 * span.match-strip__segment children (class "is-lit" when earned) or one
 * span.match-strip__fill whose inline width is the earned share. A threshold
 * of 0 gives an empty bar. Creates nodes only; nothing is attached.
 *
 * Momentum: recentBaseline (default null: no Momentum) is the team's score
 * 10 ticks ago. Earned segments from min(recentBaseline, threshold) on also
 * get class "is-recent"; the oldest of them also "is-recent-first" and the
 * newest "is-recent-last", so hud.css draws them in the team's light color
 * over one unbroken Moon Silver underline. A larger threshold splits the
 * fill in two: span.match-strip__fill up to the baseline, then
 * span.match-strip__fill.match-strip__fill--recent for the recent share.
 * With recent points the aria-label adds ", 2 in the last 10 ticks". A
 * baseline at or above the lit points marks nothing.
 *
 * @param {Document} doc
 * @param {number} value Points earned, 0 or more.
 * @param {number} threshold Points to win, 0 or more.
 * @param {string} className
 * @param {number | null} [recentBaseline] Score 10 ticks ago, or null.
 * @returns {HTMLElement}
 */
function scoreBar(doc, value, threshold, className, recentBaseline = null) {
  const bar = doc.createElement("span");
  bar.className = className;
  bar.setAttribute("role", "img");
  const lit = Math.min(value, threshold);
  const recentFrom =
    recentBaseline === null ? lit : Math.min(Math.max(0, recentBaseline), lit);
  const recent = lit - recentFrom;
  bar.setAttribute(
    "aria-label",
    `${value} of ${threshold} points to win${recent > 0 ? `, ${recent} in the last ${MOMENTUM_TICKS} ticks` : ""}`,
  );
  if (threshold > 0 && threshold <= SCORE_BAR_MAX_SEGMENTS) {
    for (let index = 0; index < threshold; index += 1) {
      const segment = doc.createElement("span");
      segment.className =
        index < lit ? "match-strip__segment is-lit" : "match-strip__segment";
      if (index < lit && index >= recentFrom) {
        segment.classList.add("is-recent");
        if (index === recentFrom) segment.classList.add("is-recent-first");
        if (index === lit - 1) segment.classList.add("is-recent-last");
      }
      bar.append(segment);
    }
  } else if (threshold > 0) {
    const fill = doc.createElement("span");
    fill.className = "match-strip__fill";
    fill.style.width = `${(100 * recentFrom) / threshold}%`;
    bar.append(fill);
    if (recent > 0) {
      const hot = doc.createElement("span");
      hot.className = "match-strip__fill match-strip__fill--recent";
      hot.style.width = `${(100 * recent) / threshold}%`;
      bar.append(hot);
    }
  }
  return bar;
}

/**
 * Mark node as a Fall-beat "before" node of kind (the choreographer contract
 * in choreography.js): data-beat-hold=kind, data-beat-side="before", drawn at
 * its END opacity 0 (attribute and inline style), and hidden from screen
 * readers. Changes node only.
 *
 * @param {HTMLElement} node
 * @param {"score"} kind
 */
function holdBefore(node, kind) {
  node.dataset.beatHold = kind;
  node.dataset.beatSide = "before";
  node.setAttribute("opacity", "0");
  node.style.opacity = "0";
  node.setAttribute("aria-hidden", "true");
}

/**
 * Make node one HUD symbol: data-hud-symbol=id, tabindex -1 (the HUD's roving
 * focus picks the one Tab stop) and, when tips is true and the symbol list has
 * symbolId, register the shared tip: the symbol's plain name as title, its
 * one-line meaning as summary, and rows (label, value) with this strip's own
 * numbers. Changes node and the tooltip registry only.
 *
 * @param {HTMLElement} node
 * @param {string} id Unique within the HUD, for example "score:1".
 * @param {boolean} tips
 * @param {string} symbolId An id from symbols.js, for example "score".
 * @param {ReadonlyArray<ReadonlyArray<string>>} rows Each row is [label, value].
 */
function markSymbol(node, id, tips, symbolId, rows) {
  node.dataset.hudSymbol = id;
  node.tabIndex = -1;
  const symbol = symbolById(symbolId);
  if (!tips || symbol === null) return;
  registerTooltipOwner(
    node,
    createSemanticDescriptor({
      kind: "hud",
      id: `hud:${id}`,
      title: symbol.title,
      tone: "information",
      accent: "none",
      summary: symbol.eli5,
      rows: rows.map((row) => ({
        label: row[0],
        value: row[1],
        metadata: { compact: true, full: true },
      })),
      sections: [],
      metadata: { compact: true, full: true },
      anchor: "element",
    }),
  );
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
 * Return the public identity rows (identity fields only) of every configured
 * champion, for the BattleClient's Agent POV "Unseen" portraits and the live
 * scorebug's command slots ((team_id - 1) * 5 + team_local_slot, which the
 * combat configuration's manual_slots and slot_labels index).
 *
 * Reads the identity directory (researcher_space.identity_directory, else
 * current_endpoint.identity_directory, the same directory the adapter uses for
 * shown IDs) and copies only public_agent_id, team_id (1 or 2),
 * team_local_slot (0 to 4) and class_id (1 to 5, or null) of rows marked
 * configured_active. It never reads roster health, statuses or positions.
 * Return a frozen array ordered by team then slot, or [] when the frame has no
 * directory. Pure.
 *
 * @param {unknown} frame
 * @returns {ReadonlyArray<Readonly<{publicAgentId: string, teamId: 1 | 2, teamLocalSlot: number, classId: number | null}>>}
 */
export function publicRosterIdentities(frame) {
  const directory =
    field(field(frame, "researcher_space"), "identity_directory") ??
    field(field(frame, "current_endpoint"), "identity_directory");
  const rows = field(directory, "identities");
  if (!Array.isArray(rows)) return Object.freeze([]);
  /** @type {Array<Readonly<{publicAgentId: string, teamId: 1 | 2, teamLocalSlot: number, classId: number | null}>>} */
  const identities = [];
  for (const row of rows) {
    const publicAgentId = field(row, "public_agent_id");
    const teamId = field(row, "team_id");
    const slot = field(row, "team_local_slot");
    const classId = field(row, "class_id");
    if (
      field(row, "configured_active") !== true ||
      typeof publicAgentId !== "string" ||
      (teamId !== 1 && teamId !== 2) ||
      !Number.isInteger(slot)
    ) {
      continue;
    }
    identities.push(
      Object.freeze({
        publicAgentId,
        teamId,
        teamLocalSlot: slot,
        classId: Number.isInteger(classId) ? classId : null,
      }),
    );
  }
  identities.sort((a, b) => a.teamId - b.teamId || a.teamLocalSlot - b.teamLocalSlot);
  return Object.freeze(identities);
}

/**
 * Gather the HUD's party-frame sources from one frame, or return null for a
 * non-frame or a frame with no known audience.
 *
 * This is the only place the HUD's facts are read from a frame. It reads the
 * audience scene (authorizedPresentationSceneView: in Agent POV only the seen
 * bodies, plus its Spawn Shield mechanics), the frame's scene class
 * mechanics and respawn waves, Agent POV visual events, the audience's own
 * incoming events (authorizedPresentationIncomingRows: only which champions'
 * Freezing Trap broke this tick, for the Champion Card), match_summary teams (and match_summary
 * deaths in Oracle View only), the identity fields of every champion
 * (publicRosterIdentities, for the BattleClient and DevClient in both views:
 * the "Unseen" portraits in Agent POV and, in both views, each champion's
 * command slot for the scorebug's "You" marks and controller lines), for the
 * Replay Viewer the one controller model (controllerRosterFromFrame; the
 * Arena Frames and the scorebug name its teams from it) and,
 * for the Replay Viewer in Agent POV only, the researcher scene view (its
 * docked full record). The product comes from frame.product_kind ("replay_viewer", otherwise
 * "combat_debugger", which includes the DevClient). Pure; the returned
 * displayIdOf closure reads the same frame later.
 *
 * @param {unknown} frame
 * @returns {import("./looks/hud.js").HudSources | null}
 */
export function hudSourcesFromFrame(frame) {
  if (!isAuthorizedPresentationFrame(frame)) return null;
  const audience = authorizedPresentationAudience(frame);
  if (audience === null) return null;
  const product =
    frame.product_kind === "replay_viewer" ? "replay_viewer" : "combat_debugger";
  const scene = authorizedPresentationSceneView(frame);
  const researcher =
    audience === "agent_pov" && product === "replay_viewer"
      ? authorizedPresentationResearcherSceneView(frame)
      : null;
  const list = (/** @type {unknown} */ value) =>
    /** @type {ReadonlyArray<Record<string, any>>} */ (
      Array.isArray(value) ? value : []
    );
  const facts = authorizedPresentationMatchBeatFacts(frame);
  return {
    product,
    audience,
    sceneAgents: list(scene?.agents),
    researcherAgents: researcher === null ? null : list(researcher.agents),
    identities:
      product === "combat_debugger" ? publicRosterIdentities(frame) : Object.freeze([]),
    controllerRoster:
      product === "replay_viewer" ? controllerRosterFromFrame(frame, null) : null,
    spawnShieldMechanics: scene?.spawn_shield_mechanics ?? null,
    mechanics: [
      ...list(frame.scene?.class_mechanics),
      ...list(researcher?.class_mechanics),
    ],
    waves: list(frame.scene?.respawn_waves),
    visualEvents: audience === "agent_pov" ? list(frame.visual_events?.events) : [],
    teams: list(frame.match_summary?.teams),
    deaths: audience === "researcher" ? list(frame.match_summary?.deaths) : [],
    controlledKey:
      product === "combat_debugger" &&
      typeof scene?.selection?.controlled_presentation_key === "string"
        ? scene.selection.controlled_presentation_key
        : null,
    recipientPublicAgentId:
      audience === "agent_pov" && typeof frame.recipient_public_agent_id === "string"
        ? frame.recipient_public_agent_id
        : null,
    returnedFallback: facts?.returned_public_agent_ids ?? [],
    trapBreakIds: authorizedPresentationIncomingRows(frame).flatMap((row) =>
      row.vocabulary === "event" &&
      row.kind === "status_broken_by_damage" &&
      row.payload?.status_id === "hunter_trap_stun" &&
      typeof row.payload?.recipient_anchor?.public_agent_id === "string"
        ? [row.payload.recipient_anchor.public_agent_id]
        : [],
    ),
    displayIdOf: (publicAgentId) =>
      authorizedPresentationAgentDisplayId(frame, publicAgentId),
  };
}
