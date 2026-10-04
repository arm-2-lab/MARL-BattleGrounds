/**
 * @file The one owner of controller names in all three clients: the
 * BattleClient, the DevClient combat screen and the Replay Viewer.
 *
 * A controller is whoever chooses a champion's actions: "You" (a Manual slot)
 * or a policy, named by the name it carries. The names come from the server
 * as they are: a built-in has a fixed name ("tdm-alpha", "tdm-beta",
 * "tdm-gamma", "random"), a saved checkpoint uses its saved display name or
 * else its method name ("Recurrent MAPPO"), a researcher's System uses its
 * own name. This file never invents, shortens or rewrites a name; long names
 * are cut only on screen, by CSS.
 *
 * What it owns:
 * - The one shared model (controllerRoster, pure). Per team, Team A first:
 *   each controller once, the person first ("You" in live play, a replay's
 *   recorded "Manual"), then each policy in the order of its first
 *   champion's slot, each with its champions (class and Agent ID) in slot
 *   order. The scorebug line, both teams' hover lists and the
 *   BattleClient rail's Policies list are all drawn from it, so they always
 *   name the same controllers with the same full names, champion counts and
 *   order.
 * - The words (pure): the full line ("You + Recurrent MAPPO",
 *   controllerLineParts), the short line ("You + 4 Policies", "4 Policies",
 *   "1 Policy", and in the Replay Viewer "Manual + 4 Policies";
 *   collapsedControllerLine), "1 Champion" / "4 Champions"
 *   (championCountWords), the rail's rows (policyListRows) and the hover
 *   list's content (controllerListModel).
 * - The page parts (DOM): paintControllerLine shows a line's names when they
 *   fit in the line's real rendered width, else the short line, and fits it
 *   again when its space changes (fitControllerLine). Every name sits in a
 *   span.controller-name that CSS cuts with "…" when it is wider than the
 *   fixed name width (--controller-name-max in
 *   src/styles/controller-names.css). controllerNameHtml,
 *   controllerNameNode, markControllerName and paintNamedSentence (a
 *   "played by <name>" sentence such as the deck's status line) make a name
 *   elsewhere follow the same rule. Hovering or focusing a line, or a cut name, drops down the
 *   team's full list ("Team A Controllers") under it: one shared panel
 *   (#controller-list) kept inside the window at every width.
 *
 * Who calls it: looks/hud.js (the scorebug lines in every client and the
 * Replay Viewer's Arena Frames headers), match-summary.js (the old score strip),
 * tactics-turn.js (the rail's Policies list, the focus chip, Watch Only and
 * the Class Guide note) and panels.js (the Developer drawer's Roster card).
 * Each caller builds the model with controllerRoster from the same facts
 * (the launch's combat configuration and the identity directory in live
 * play, the recorded per-champion names in the Replay Viewer, or its
 * recorded team names for old replays) and publishes it with
 * publishControllerRoster, which the hover lists read.
 *
 * Agent POV: who plays each champion is set before the game and is the same
 * in both views, so names and counts reveal nothing hidden. The model reads
 * identity fields only (team, slot, class, shown ID), never life, place or
 * status, and Agent POV and Oracle View show the same lists.
 *
 * Side effects: the pure functions touch nothing. The page parts write only
 * the elements they are given, one shared panel appended to the page body on
 * first use, document listeners for pointer, focus, key, scroll and resize
 * events (installed once per document), one ResizeObserver per document and
 * a 120 ms hide delay. No storage or network.
 */
import { classNamesV1 } from "./class-documentation.js";
import { createSvgIcon } from "./icons.js";
import { classTokenFromId } from "./vocabulary.js";

/** The ID of the one shared hover list panel. */
export const CONTROLLER_LIST_ID = "controller-list";

/** How long the pointer may be away from a line and its list before it closes, in ms. */
export const CONTROLLER_LIST_HIDE_MS = 120;

/** The space kept between the list panel and the window edges, in CSS px. */
const WINDOW_MARGIN = 8;

/** The list panel's widest width, in CSS px (the approved picture's width). */
const PANEL_WIDTH = 360;

/**
 * @typedef {Readonly<{
 *   publicAgentId: string | null,
 *   slot: number | null,
 *   classId: number | null,
 *   className: string,
 *   displayId: string,
 * }>} ControllerChampion
 *   One champion a controller plays. slot is its command slot (0 to 4 for
 *   Team A, 5 to 9 for Team B), or null when unknown. classId is 1 to 5, or
 *   null when unknown; className is "Mage", "Warrior", "Hunter", "Rogue" or
 *   "Priest", or "Champion" when the class is unknown. displayId is the shown
 *   Agent ID ("5").
 */

/**
 * @typedef {Readonly<{
 *   name: string,
 *   you: boolean,
 *   manual: boolean,
 *   champions: ReadonlyArray<ControllerChampion>,
 * }>} TeamController
 *   One controller of a team. name is "You" when you is true (live play:
 *   the viewer plays these champions), else the full name exactly as sent or
 *   recorded. manual is true when a person played these champions in a
 *   saved game (the Replay Viewer's recorded Manual slots, see
 *   recordedManualSlots); name then stays the recorded name ("Manual"). you
 *   and manual are never both true, and live play never sets manual. A
 *   controller with neither is a policy. champions has at least one entry, in
 *   slot order.
 */

/**
 * @typedef {Readonly<{
 *   teamId: 1 | 2,
 *   label: "Team A" | "Team B",
 *   controllers: ReadonlyArray<TeamController>,
 * }>} ControllerTeam
 *   One team's controllers: the person first ("You" when any champion is
 *   Manual in live play, or a replay's recorded "Manual"), then each policy
 *   in the order of its first champion's slot. Empty when no
 *   champion of the team has a known controller (for example before the
 *   launch's configuration arrives).
 */

/**
 * @typedef {Readonly<{teams: readonly [ControllerTeam, ControllerTeam]}>} ControllerRoster
 *   The one shared model: Team A, then Team B.
 */

/**
 * @typedef {{
 *   publicAgentId?: string | null,
 *   teamId: unknown,
 *   slot?: unknown,
 *   teamLocalSlot?: unknown,
 *   classId?: unknown,
 *   displayId?: unknown,
 * }} RosterChampionInput
 *   One champion's identity fields. teamId is 1 or 2. Give its command slot
 *   either as slot (0 to 9) or as teamLocalSlot (0 to 4, which gives
 *   (teamId - 1) * 5 + teamLocalSlot). classId is 1 to 5. displayId is the
 *   shown Agent ID; when missing, the command slot is shown, as live play
 *   does. Rows with another team are skipped.
 */

/**
 * @typedef {{
 *   champions?: ReadonlyArray<RosterChampionInput> | null,
 *   manualSlots?: ReadonlyArray<unknown> | null,
 *   recordedManualSlots?: ReadonlyArray<unknown> | null,
 *   slotLabels?: ReadonlyArray<unknown> | null,
 *   teamNames?: ReadonlyArray<unknown> | null,
 * }} ControllerRosterInput
 *   What the model is built from. Live play (BattleClient and DevClient)
 *   gives manualSlots and slotLabels, the launch's combat configuration
 *   (manual_slots: command slots a human orders; slot_labels: ten names or
 *   nulls in slot order), and champions, the identity directory's rows
 *   (publicRosterIdentities in match-summary.js). The Replay Viewer gives
 *   slotLabels from the replay's recorded per-champion names (the match
 *   summary's slot_controller_names) and no manualSlots, so a Manual slot
 *   is named "Manual", not "You". It also gives recordedManualSlots, the
 *   command slots (0 to 9) a person played in the saved game: their
 *   controller keeps its recorded name and gets manual true. Live play
 *   never gives it. It is read only with slotLabels; default none (no slot
 *   is a recorded person). An old replay without them gives
 *   teamNames instead (Team A's and Team B's recorded names, the match
 *   summary's display names), and every champion of a team is played by its
 *   team's name. slotLabels wins over teamNames when both are given. With no
 *   champions, live play lists one champion of unknown class for every slot
 *   with a name.
 */

/**
 * The empty model: Team A and Team B with no controllers (before the launch's
 * configuration arrives, or for anything that is not a presentation).
 *
 * @type {ControllerRoster}
 */
export const NO_CONTROLLER_ROSTER = Object.freeze({
  teams: /** @type {readonly [ControllerTeam, ControllerTeam]} */ (
    Object.freeze([
      Object.freeze({
        teamId: /** @type {const} */ (1),
        label: /** @type {const} */ ("Team A"),
        controllers: Object.freeze([]),
      }),
      Object.freeze({
        teamId: /** @type {const} */ (2),
        label: /** @type {const} */ ("Team B"),
        controllers: Object.freeze([]),
      }),
    ])
  ),
});

/**
 * Whether value is a whole command slot, 0 to 9. Pure.
 *
 * @param {unknown} value
 * @returns {value is number}
 */
function isSlot(value) {
  return Number.isInteger(value) && Number(value) >= 0 && Number(value) < 10;
}

/**
 * A usable name: a string with something other than spaces in it. Pure.
 *
 * @param {unknown} value
 * @returns {value is string}
 */
function isName(value) {
  return typeof value === "string" && value.trim() !== "";
}

/**
 * Turn one identity row into a ControllerChampion and its team, or null when
 * its team is not 1 or 2. Pure.
 *
 * @param {RosterChampionInput} row
 * @returns {{teamId: 1 | 2, champion: ControllerChampion} | null}
 */
function championFromRow(row) {
  const teamId = row?.teamId;
  if (teamId !== 1 && teamId !== 2) return null;
  const slot = isSlot(row.slot)
    ? row.slot
    : Number.isInteger(row.teamLocalSlot) &&
        Number(row.teamLocalSlot) >= 0 &&
        Number(row.teamLocalSlot) < 5
      ? (teamId - 1) * 5 + Number(row.teamLocalSlot)
      : null;
  const classId =
    Number.isInteger(row.classId) &&
    Number(row.classId) >= 1 &&
    Number(row.classId) <= 5
      ? Number(row.classId)
      : null;
  const shown =
    typeof row.displayId === "string" && row.displayId !== ""
      ? row.displayId
      : Number.isInteger(row.displayId)
        ? String(row.displayId)
        : slot === null
          ? "?"
          : String(slot);
  return {
    teamId,
    champion: Object.freeze({
      publicAgentId: typeof row.publicAgentId === "string" ? row.publicAgentId : null,
      slot,
      classId,
      className: classNamesV1(classId)?.className ?? "Champion",
      displayId: shown,
    }),
  };
}

/**
 * Build the one shared model of both teams' controllers.
 *
 * Each champion is played by "You" when its command slot is in manualSlots;
 * else by its slot's name in slotLabels, flagged manual when the slot is also
 * in recordedManualSlots (a person played it in a saved game); else (no
 * slotLabels) by its team's name in teamNames. A champion with none of these
 * is left out. Each team lists the person first ("You", or a recorded
 * manual controller), then each policy once, in the order of its first
 * champion's slot; two slots with the same name and the same flags share one
 * controller. Each controller lists its champions in slot order (shown ID
 * order when the slot is unknown). Names are kept exactly as given.
 *
 * Pure: reads only its input, returns a new frozen ControllerRoster (both
 * teams always present, possibly with no controllers). Malformed entries are
 * skipped, never thrown on.
 *
 * Agent POV: the input is identity fields and the match setup, which every
 * view shows, so the model is the same in Agent POV and Oracle View.
 *
 * @param {ControllerRosterInput | null | undefined} input
 * @returns {ControllerRoster}
 *
 * @example
 * controllerRoster({
 *   manualSlots: [1],
 *   slotLabels: ["Recurrent MAPPO", "Manual", "Recurrent MAPPO", null, null,
 *     "tdm-beta", "tdm-beta", null, null, null],
 *   champions: [{teamId: 1, teamLocalSlot: 0, classId: 1}, ...],
 * }).teams[0].controllers.map((c) => c.name); // ["You", "Recurrent MAPPO"]
 */
export function controllerRoster(input) {
  const manual = new Set(
    Array.isArray(input?.manualSlots) ? input.manualSlots.filter(isSlot) : [],
  );
  const recordedManual = new Set(
    Array.isArray(input?.recordedManualSlots)
      ? input.recordedManualSlots.filter(isSlot)
      : [],
  );
  const labels = Array.isArray(input?.slotLabels) ? input.slotLabels : null;
  const teamNames = Array.isArray(input?.teamNames) ? input.teamNames : null;
  /** @type {Array<{teamId: 1 | 2, champion: ControllerChampion}>} */
  let rows = [];
  for (const row of Array.isArray(input?.champions) ? input.champions : []) {
    const entry = championFromRow(row);
    if (entry !== null) rows.push(entry);
  }
  if (rows.length === 0 && labels !== null) {
    rows = labels.flatMap((label, slot) => {
      if (!isName(label) || slot > 9) return [];
      const entry = championFromRow({ teamId: slot < 5 ? 1 : 2, slot });
      return entry === null ? [] : [entry];
    });
  }
  /**
   * Who plays one champion, or null when nobody is known.
   *
   * @param {1 | 2} teamId
   * @param {ControllerChampion} champion
   * @returns {{name: string, you: boolean, manual: boolean} | null}
   */
  const playerOf = (teamId, champion) => {
    if (labels !== null) {
      if (champion.slot === null) return null;
      if (manual.has(champion.slot)) return { name: "You", you: true, manual: false };
      const label = labels[champion.slot];
      return isName(label)
        ? { name: label, you: false, manual: recordedManual.has(champion.slot) }
        : null;
    }
    const name = teamNames?.[teamId - 1];
    return isName(name) ? { name, you: false, manual: false } : null;
  };
  const order = (
    /** @type {ControllerChampion} */ a,
    /** @type {ControllerChampion} */ b,
  ) =>
    (a.slot ?? 99) - (b.slot ?? 99) ||
    Number(a.displayId) - Number(b.displayId) ||
    a.displayId.localeCompare(b.displayId);
  const teams = /** @type {const} */ ([1, 2]).map((teamId) => {
    const members = rows
      .filter((row) => row.teamId === teamId)
      .map((row) => row.champion)
      .sort(order);
    /** @type {Array<{name: string, you: boolean, manual: boolean, champions: ControllerChampion[]}>} */
    const controllers = [];
    for (const champion of members) {
      const player = playerOf(teamId, champion);
      if (player === null) continue;
      const entry = controllers.find(
        (row) =>
          row.you === player.you &&
          row.manual === player.manual &&
          row.name === player.name,
      );
      if (entry) entry.champions.push(champion);
      else controllers.push({ ...player, champions: [champion] });
    }
    // The person first ("You", or a replay's recorded Manual); a stable sort
    // keeps the policies in first-slot order. Live play never sets manual.
    controllers.sort((a, b) => Number(b.you || b.manual) - Number(a.you || a.manual));
    return Object.freeze({
      teamId,
      label: teamId === 1 ? "Team A" : "Team B",
      controllers: Object.freeze(
        controllers.map((row) =>
          Object.freeze({
            name: row.name,
            you: row.you,
            manual: row.manual,
            champions: Object.freeze(row.champions),
          }),
        ),
      ),
    });
  });
  return Object.freeze({
    teams: /** @type {readonly [ControllerTeam, ControllerTeam]} */ (
      /** @type {unknown} */ (Object.freeze(teams))
    ),
  });
}

/**
 * Return one team of a roster. Pure.
 *
 * @param {ControllerRoster} roster
 * @param {unknown} teamId 1 (Team A) or 2 (Team B).
 * @returns {ControllerTeam | null} The team, or null for another teamId.
 */
export function rosterTeam(roster, teamId) {
  return teamId === 1 || teamId === 2 ? roster.teams[teamId - 1] : null;
}

/**
 * Find who plays one champion, by its public ID or its command slot. Pure.
 *
 * @param {ControllerRoster} roster
 * @param {{publicAgentId?: unknown, slot?: unknown}} who At least one of
 *   publicAgentId (matched first) or slot (0 to 9).
 * @returns {Readonly<{team: ControllerTeam, controller: TeamController}> | null}
 *   The champion's team and controller, or null when the roster does not list
 *   it.
 */
export function controllerOfChampion(roster, who) {
  for (const team of roster.teams) {
    for (const controller of team.controllers) {
      const found = controller.champions.some(
        (champion) =>
          (typeof who.publicAgentId === "string" &&
            champion.publicAgentId === who.publicAgentId) ||
          (isSlot(who.slot) && champion.slot === who.slot),
      );
      if (found) return Object.freeze({ team, controller });
    }
  }
  return null;
}

/**
 * Write a champion count: "1 Champion", "4 Champions". Pure.
 *
 * @param {number} count Whole number, 0 or more; only exactly 1 is singular.
 * @returns {string}
 */
export function championCountWords(count) {
  return `${count} Champion${count === 1 ? "" : "s"}`;
}

/**
 * The full line's names, in order: "You" first when the team has a Manual
 * champion (a replay's recorded "Manual" likewise comes first), then each
 * policy's full name. Joined by " + " this is the full line ("You + Recurrent
 * MAPPO"). Empty for a team with no controllers. Pure.
 *
 * @param {ControllerTeam} team
 * @returns {ReadonlyArray<string>}
 */
export function controllerLineParts(team) {
  return Object.freeze(team.controllers.map((controller) => controller.name));
}

/**
 * The short line used when the full line does not fit: "You + 4 Policies",
 * "You + 1 Policy", "4 Policies", "1 Policy", "You" (you play every
 * champion) or "" (no controllers). Pure.
 *
 * In the Replay Viewer no champion is yours. A controller flagged manual (a
 * person played it in the saved game) is counted as that person, not as a
 * policy, and keeps its recorded name: "Manual + 3 Policies", or "Manual"
 * alone. Only the flag decides this, never the name, so in live play (which
 * never sets manual) a policy that happens to be named "Manual" still counts
 * as a policy.
 *
 * @param {ControllerTeam} team
 * @returns {string}
 */
export function collapsedControllerLine(team) {
  const person = team.controllers.find(
    (controller) => controller.you || controller.manual,
  );
  const human = person === undefined ? null : person.you ? "You" : person.name;
  const policies = team.controllers.filter(
    (controller) => !controller.you && !controller.manual,
  ).length;
  if (policies === 0) return human ?? "";
  const words = `${policies} ${policies === 1 ? "Policy" : "Policies"}`;
  return human === null ? words : `${human} + ${words}`;
}

/**
 * The BattleClient rail's Policies list: every policy of both teams, Team A
 * first, in the roster's order (the hover lists' rows without "You"). Pure.
 *
 * @param {ControllerRoster} roster
 * @returns {ReadonlyArray<Readonly<{teamId: 1 | 2, name: string, count: number, countWords: string}>>}
 *   name is the full name (never cut); count is the number of champions it
 *   plays and countWords says it ("4 Champions").
 */
export function policyListRows(roster) {
  return Object.freeze(
    roster.teams.flatMap((team) =>
      team.controllers
        .filter((controller) => !controller.you)
        .map((controller) =>
          Object.freeze({
            teamId: team.teamId,
            name: controller.name,
            count: controller.champions.length,
            countWords: championCountWords(controller.champions.length),
          }),
        ),
    ),
  );
}

/**
 * @typedef {Readonly<{
 *   title: string,
 *   teamId: 1 | 2,
 *   rows: ReadonlyArray<Readonly<{
 *     name: string,
 *     you: boolean,
 *     countWords: string,
 *     champions: ReadonlyArray<Readonly<{classId: number | null, text: string}>>,
 *   }>>,
 * }>} ControllerListModel
 *   The hover list's content. title is "Team A Controllers" or "Team B
 *   Controllers". One row per controller in the roster's order: its full
 *   name (never cut; "You" for the human), countWords ("2 Champions") and
 *   each champion it plays as its class ID (for the class icon) and text
 *   ("Mage · ID 0").
 */

/**
 * Build the hover list's content for one team. Pure.
 *
 * @param {ControllerTeam} team
 * @returns {ControllerListModel}
 */
export function controllerListModel(team) {
  return Object.freeze({
    title: `${team.label} Controllers`,
    teamId: team.teamId,
    rows: Object.freeze(
      team.controllers.map((controller) =>
        Object.freeze({
          name: controller.name,
          you: controller.you,
          countWords: championCountWords(controller.champions.length),
          champions: Object.freeze(
            controller.champions.map((champion) =>
              Object.freeze({
                classId: champion.classId,
                text: `${champion.className} · ID ${champion.displayId}`,
              }),
            ),
          ),
        }),
      ),
    ),
  });
}

/**
 * Choose what a line shows. Pure.
 *
 * @param {ControllerTeam} team
 * @param {boolean} fullFits True when the full line fits in its space.
 * @returns {Readonly<{mode: "full" | "short", text: string}>} mode "full"
 *   shows the names (text is the full line, "You + Recurrent MAPPO"); mode
 *   "short" shows the short line ("You + 4 Policies"). A team with only "You"
 *   or no controllers is always "full".
 */
export function controllerLineChoice(team, fullFits) {
  const full = controllerLineParts(team).join(" + ");
  const short = collapsedControllerLine(team);
  return fullFits || short === full
    ? Object.freeze({ mode: /** @type {const} */ ("full"), text: full })
    : Object.freeze({ mode: /** @type {const} */ ("short"), text: short });
}

// ---------------------------------------------------------------------------
// The page parts: lines, names and the one hover list.

/**
 * @typedef {{
 *   roster: ControllerRoster,
 *   panel: HTMLElement | null,
 *   trigger: Element | null,
 *   hideTimer: ReturnType<typeof setTimeout> | null,
 *   lines: Set<HTMLElement>,
 *   observer: ResizeObserver | null,
 *   widths: WeakMap<Element, number>,
 *   pointer: {x: number, y: number} | null,
 *   anchor: {left: number, top: number} | null,
 * }} PageState
 *   One document's controller-name state: the last published roster, the
 *   shared panel (null until first shown), the element the open list belongs
 *   to (null when closed), the pending hide, the lines to fit again when
 *   their space changes, the resize observer, the widths last fitted at,
 *   the pointer's last place in window coordinates (null before it moves)
 *   and where the open list's element was when the list was placed (null
 *   when closed).
 */

/** @type {WeakMap<Document, PageState>} */
const pages = new WeakMap();

/** The team each painted line shows. @type {WeakMap<Element, ControllerTeam>} */
const lineTeams = new WeakMap();

/**
 * Return a document's state, installing its listeners on first use.
 *
 * Listeners (once per document): pointerover and focusin open the list for a
 * marked element (data-controller-team); pointermove and pointerover keep
 * the pointer's place (to find a repainted line under it, see
 * publishControllerRoster); pointerout and focusout close it
 * after CONTROLLER_LIST_HIDE_MS unless the pointer or focus is on the
 * element or the panel; Escape, a scroll outside the panel that moves the
 * open list's element, and a window resize close it at once. Also fits every line again once the page's fonts
 * have loaded. A value that is not an object (no document, as in a Node test
 * with a bare stand-in) gets a fresh state that is not kept; a stand-in
 * without addEventListener gets a kept state with no listeners. Side
 * effects: those listeners only.
 *
 * @param {Document} doc
 * @returns {PageState}
 */
function pageState(doc) {
  const known = doc !== null && typeof doc === "object" ? pages.get(doc) : undefined;
  if (known) return known;
  /** @type {PageState} */
  const page = {
    roster: NO_CONTROLLER_ROSTER,
    panel: null,
    trigger: null,
    hideTimer: null,
    lines: new Set(),
    observer: null,
    widths: new WeakMap(),
    pointer: null,
    anchor: null,
  };
  if (doc === null || typeof doc !== "object") return page;
  pages.set(doc, page);
  if (typeof doc.addEventListener !== "function") return page;
  /** @param {EventTarget | null} target @returns {Element | null} */
  const markedFrom = (target) =>
    target instanceof Element ? target.closest("[data-controller-team]") : null;
  /** @param {EventTarget | null} target */
  const insideList = (target) =>
    target instanceof Node &&
    ((page.panel?.contains(target) ?? false) ||
      (page.trigger?.contains(target) ?? false));
  doc.addEventListener(
    "pointermove",
    (event) => {
      page.pointer = { x: event.clientX, y: event.clientY };
    },
    { passive: true },
  );
  doc.addEventListener("pointerover", (event) => {
    page.pointer = { x: event.clientX, y: event.clientY };
    const marked = markedFrom(event.target);
    if (marked !== null) openList(doc, marked);
    else if (insideList(event.target)) cancelHide(page);
  });
  doc.addEventListener("pointerout", (event) => {
    if (page.trigger === null || insideList(event.relatedTarget)) return;
    if (insideList(event.target)) scheduleHide(doc);
  });
  doc.addEventListener("focusin", (event) => {
    const marked = markedFrom(event.target);
    if (marked !== null) openList(doc, marked);
  });
  doc.addEventListener("focusout", (event) => {
    if (page.trigger !== null && !insideList(event.relatedTarget)) scheduleHide(doc);
  });
  doc.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && page.trigger !== null) hideList(doc);
  });
  doc.addEventListener(
    "scroll",
    (event) => {
      if (page.trigger === null || insideList(event.target)) return;
      // Close only when the scroll moved the list's element away from where
      // the list was placed (a scroll that left it in place keeps the list).
      const box = page.trigger.getBoundingClientRect();
      if (
        page.anchor === null ||
        Math.abs(box.left - page.anchor.left) > 1 ||
        Math.abs(box.top - page.anchor.top) > 1
      )
        hideList(doc);
    },
    true,
  );
  doc.defaultView?.addEventListener("resize", () => hideList(doc));
  doc.fonts?.ready.then(() => refitLines(doc)).catch(() => {});
  return page;
}

/**
 * Keep the model the hover lists read for this document. Every caller that
 * paints controller names publishes the roster it painted from (all build it
 * with controllerRoster from the same facts), after painting its names. An
 * open list keeps its place and its pending hide, and its content is
 * redrawn from the new roster. When its line was replaced by a repaint (the
 * Replay Viewer rebuilds its Arena Frames on every frame), the list
 * moves to the marked element of the same team now under the pointer, or,
 * while the pointer is on the list, to the repainted twin of its line (same
 * first class, same team); it closes when there is none or its team has no
 * controllers now. Side effects: the
 * document's controller-name state, and the open panel's DOM and place.
 *
 * @param {Document} doc
 * @param {ControllerRoster} roster
 * @returns {void}
 */
export function publishControllerRoster(doc, roster) {
  const page = pageState(doc);
  page.roster = roster;
  if (page.trigger === null || page.panel === null) return;
  let trigger = page.trigger;
  if (!trigger.isConnected) {
    const teamMark = trigger.getAttribute("data-controller-team");
    const at =
      page.pointer !== null && typeof doc.elementFromPoint === "function"
        ? doc.elementFromPoint(page.pointer.x, page.pointer.y)
        : null;
    // The pointer on the list itself: its line's repainted twin (same first
    // class, same team). Else the marked element under the pointer.
    const firstClass = trigger.classList.item(0);
    const twin =
      at !== null && page.panel.contains(at)
        ? firstClass !== null && /^[\w-]+$/u.test(firstClass)
          ? doc.querySelector(`.${firstClass}[data-controller-team="${teamMark}"]`)
          : null
        : (at?.closest("[data-controller-team]") ?? null);
    if (twin === null || twin.getAttribute("data-controller-team") !== teamMark) {
      hideList(doc);
      return;
    }
    trigger = twin;
    page.trigger = trigger;
    if (trigger instanceof HTMLElement) trigger.dataset.controllerOpen = "true";
  }
  const team = rosterTeam(roster, Number(trigger.getAttribute("data-controller-team")));
  if (team === null || team.controllers.length === 0) {
    hideList(doc);
    return;
  }
  fillControllerList(page.panel, controllerListModel(team));
  const box = trigger.getBoundingClientRect();
  page.anchor = { left: box.left, top: box.top };
  placeControllerList(page.panel, box, viewSize(doc));
}

/**
 * The window's size in CSS px, without the scroll bar.
 *
 * @param {Document} doc
 * @returns {{width: number, height: number}}
 */
function viewSize(doc) {
  return {
    width: doc.documentElement.clientWidth || doc.defaultView?.innerWidth || 0,
    height: doc.documentElement.clientHeight || doc.defaultView?.innerHeight || 0,
  };
}

/**
 * Read the roster last published for this document (NO_CONTROLLER_ROSTER
 * before any). Pure apart from installing the document's listeners on first
 * use.
 *
 * @param {Document} doc
 * @returns {ControllerRoster}
 */
export function publishedControllerRoster(doc) {
  return pageState(doc).roster;
}

/**
 * Mark an element as a controller name or line of one team, so hovering or
 * focusing it drops down that team's list. mode "line" (a scorebug or team
 * line) always opens it; mode "name" (a name inside other words, such as
 * "Played By ...") opens it only while CSS cuts the name with "…". Side
 * effects: data-controller-team and data-controller-list on the element, and
 * the document's listeners on first use.
 *
 * @param {HTMLElement} element
 * @param {1 | 2} teamId
 * @param {"line" | "name"} [mode] Default "name".
 * @returns {void}
 */
export function markControllerName(element, teamId, mode = "name") {
  element.dataset.controllerTeam = String(teamId);
  element.dataset.controllerList = mode;
  pageState(element.ownerDocument);
}

/**
 * Write one controller name as HTML, for page parts built from HTML text
 * (the Tactics Turn deck and rail): a span.controller-name, cut with "…" by
 * CSS when wider than the fixed name width, marked so hovering it while cut
 * drops down its team's list (see markControllerName). teamId null gives the
 * plain cut name with no list. The name is escaped. Pure.
 *
 * @param {string} name The full name ("Recurrent MAPPO").
 * @param {1 | 2 | null} teamId
 * @param {string} [end] Punctuation that closes the sentence right after the
 *   name ("."), kept inside the span so a cut name reads "Recurrent MAPPO ·
 *   Pool…" with no lone period after the "…"; default "" (none).
 * @returns {string}
 */
export function controllerNameHtml(name, teamId, end = "") {
  const text = `${name}${end}`
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
  const marks =
    teamId === 1 || teamId === 2
      ? ` data-controller-team="${teamId}" data-controller-list="name"`
      : "";
  return `<span class="controller-name"${marks}>${text}</span>`;
}

/**
 * Make one controller name node (span.controller-name, cut with "…" by CSS
 * when wider than the fixed name width), marked for its team's hover list
 * when teamId is 1 or 2 (see markControllerName; mode "name"). Creates a node
 * only.
 *
 * @param {Document} doc
 * @param {string} name The full name.
 * @param {1 | 2 | null} teamId
 * @param {string} [end] Closing punctuation kept inside the span (see
 *   controllerNameHtml); default "".
 * @returns {HTMLSpanElement}
 */
export function controllerNameNode(doc, name, teamId, end = "") {
  const node = /** @type {HTMLSpanElement} */ (doc.createElement("span"));
  node.className = "controller-name";
  node.textContent = `${name}${end}`;
  if (teamId === 1 || teamId === 2) markControllerName(node, teamId, "name");
  return node;
}

/**
 * Write a sentence that may name a controller ("Looking at the Warrior (ID
 * 1), played by Recurrent MAPPO.") into element, as one inline run.
 *
 * When the text has "played by " (any letter case) followed by the full
 * name of a controller in the roster last published for the document
 * (longest name first), that name becomes a controllerNameNode: cut with "…"
 * past the fixed name width, its team's list on hover while cut; a period
 * right after the name goes inside its span. The run is
 * one span, so a flex container (the deck's status line) still wraps it as
 * one text. Other text is written as plain text. Nothing is rewritten when
 * element already shows exactly this text. Side effects: element's
 * children.
 *
 * @param {HTMLElement} element
 * @param {string} text
 * @returns {void}
 */
export function paintNamedSentence(element, text) {
  const doc = element.ownerDocument;
  const roster = publishedControllerRoster(doc);
  const lower = text.toLowerCase();
  const at = lower.indexOf("played by ");
  /** @type {{name: string, teamId: 1 | 2} | null} */
  let found = null;
  if (at >= 0) {
    const rest = text.slice(at + "played by ".length);
    for (const team of roster.teams) {
      for (const controller of team.controllers) {
        if (
          !controller.you &&
          rest.startsWith(controller.name) &&
          (found === null || controller.name.length > found.name.length)
        )
          found = { name: controller.name, teamId: team.teamId };
      }
    }
  }
  if (found === null) {
    if (element.textContent !== text) element.textContent = text;
    return;
  }
  if (element.textContent === text && element.querySelector(".controller-name")) return;
  const start = at + "played by ".length;
  const after = text.slice(start + found.name.length);
  // A period right after the name stays inside the cut span (no lone ".").
  const end = after.startsWith(".") ? "." : "";
  const run = doc.createElement("span");
  run.append(
    text.slice(0, start),
    controllerNameNode(doc, found.name, found.teamId, end),
    after.slice(end.length),
  );
  element.replaceChildren(run);
}

/**
 * Paint one team's controller line into element and keep it fitted.
 *
 * element is the line's own box (the scorebug's #ctl-a or #ctl-b, or a
 * Replay Viewer team name); CSS must give it a bounded width with overflow
 * hidden and no wrapping (class controller-line in controller-names.css
 * does). It gets class controller-line, aria-label (the full line, so a
 * screen reader always hears every name), the team's marks for the hover
 * list (mode "line") and its content from fitControllerLine. fitRoot, the
 * element whose width decides the line's space (default element's parent),
 * is watched with a ResizeObserver, and the line is fitted again whenever
 * fitRoot's width changes (for example a window resize, or the scorebug
 * becoming visible). Side effects: element's DOM and attributes, the
 * observer, the document's listeners on first use.
 *
 * @param {HTMLElement} element
 * @param {ControllerTeam} team
 * @param {{fitRoot?: Element | null}} [options]
 * @returns {void}
 */
export function paintControllerLine(element, team, options = {}) {
  const doc = element.ownerDocument;
  const page = pageState(doc);
  lineTeams.set(element, team);
  element.classList.add("controller-line");
  const full = controllerLineParts(team).join(" + ");
  if (full === "") {
    element.removeAttribute("aria-label");
    delete element.dataset.controllerTeam;
    delete element.dataset.controllerList;
  } else {
    element.setAttribute("aria-label", `${team.label} Controllers: ${full}`);
    markControllerName(element, team.teamId, "line");
  }
  for (const line of page.lines) if (!line.isConnected) page.lines.delete(line);
  page.lines.add(element);
  fitControllerLine(element);
  const root = options.fitRoot ?? element.parentElement;
  const Observer = doc.defaultView?.ResizeObserver;
  if (root && typeof Observer === "function") {
    page.observer ??= new Observer((entries) => {
      let changed = false;
      for (const entry of entries) {
        const width = Math.round(entry.contentRect.width);
        if (page.widths.get(entry.target) !== width) changed = true;
        page.widths.set(entry.target, width);
      }
      if (changed) refitLines(doc);
    });
    page.observer.observe(root);
  }
}

/**
 * Fit one painted line to its space, measured from the real rendered width.
 *
 * Writes the full line (each name in a span.controller-name, which CSS cuts
 * with "…" past the fixed name width; "You" stays whole; " + " between),
 * then compares the line's scrollWidth with its clientWidth. When the full
 * line overflows, it writes the short line instead ("You + 4 Policies").
 * Sets data-controller-line to "full" or "short". A line that is not painted
 * by paintControllerLine is left alone. Side effects: element's children and
 * one data attribute; reading the widths makes the browser lay out the page.
 *
 * @param {HTMLElement} element
 * @returns {void}
 */
export function fitControllerLine(element) {
  const team = lineTeams.get(element);
  if (!team) return;
  const doc = element.ownerDocument;
  /** @type {Array<Node | string>} */
  const nodes = [];
  team.controllers.forEach((controller, index) => {
    if (index > 0) nodes.push(" + ");
    nodes.push(controller.you ? "You" : controllerNameNode(doc, controller.name, null));
  });
  element.replaceChildren(...nodes);
  const fits = element.scrollWidth <= element.clientWidth + 0.5;
  const choice = controllerLineChoice(team, fits);
  if (choice.mode === "short") element.replaceChildren(choice.text);
  element.dataset.controllerLine = choice.mode;
}

/**
 * Fit every painted line of the document again (after a width change or once
 * the fonts load). Lines no longer in the page are forgotten. Side effects:
 * as fitControllerLine.
 *
 * @param {Document} doc
 * @returns {void}
 */
function refitLines(doc) {
  const page = pageState(doc);
  for (const line of page.lines) {
    if (line.isConnected) fitControllerLine(line);
    else page.lines.delete(line);
  }
}

/**
 * Stop a pending hide.
 *
 * @param {PageState} page
 * @returns {void}
 */
function cancelHide(page) {
  if (page.hideTimer !== null) clearTimeout(page.hideTimer);
  page.hideTimer = null;
}

/**
 * Close the list after CONTROLLER_LIST_HIDE_MS, so the pointer can move from
 * a line into its panel. Side effects: one timer.
 *
 * @param {Document} doc
 * @returns {void}
 */
function scheduleHide(doc) {
  const page = pageState(doc);
  cancelHide(page);
  page.hideTimer = setTimeout(() => hideList(doc), CONTROLLER_LIST_HIDE_MS);
}

/**
 * Close the list at once. Side effects: hides the panel and clears the open
 * element's data-controller-open mark.
 *
 * @param {Document} doc
 * @returns {void}
 */
export function hideList(doc) {
  const page = pageState(doc);
  cancelHide(page);
  if (page.trigger instanceof HTMLElement) delete page.trigger.dataset.controllerOpen;
  page.trigger = null;
  page.anchor = null;
  if (page.panel) page.panel.hidden = true;
}

/**
 * Open the team list for one marked element, under it and inside the window.
 *
 * Reads the team from the element's data-controller-team and the roster last
 * published for the document. A "name" mark opens only while the name is cut
 * (scrollWidth above clientWidth). A team with no controllers opens nothing.
 * Side effects: builds or reuses #controller-list, sets data-controller-open
 * on the element.
 *
 * @param {Document} doc
 * @param {Element} trigger
 * @returns {void}
 */
function openList(doc, trigger) {
  const page = pageState(doc);
  cancelHide(page);
  if (page.trigger === trigger && page.panel !== null && !page.panel.hidden) return;
  const teamId = Number(trigger.getAttribute("data-controller-team"));
  const team = rosterTeam(page.roster, teamId);
  const cut = trigger.scrollWidth > trigger.clientWidth + 0.5;
  if (
    team === null ||
    team.controllers.length === 0 ||
    (trigger.getAttribute("data-controller-list") !== "line" && !cut)
  ) {
    if (page.trigger !== trigger) return;
    hideList(doc);
    return;
  }
  if (page.trigger instanceof HTMLElement && page.trigger !== trigger)
    delete page.trigger.dataset.controllerOpen;
  page.trigger = trigger;
  if (trigger instanceof HTMLElement) trigger.dataset.controllerOpen = "true";
  if (page.panel === null || !page.panel.isConnected) {
    page.panel = doc.createElement("div");
    page.panel.id = CONTROLLER_LIST_ID;
    page.panel.className = "controller-list";
    page.panel.setAttribute("role", "tooltip");
    doc.body.append(page.panel);
  }
  page.panel.hidden = false;
  fillControllerList(page.panel, controllerListModel(team));
  const box = trigger.getBoundingClientRect();
  page.anchor = { left: box.left, top: box.top };
  placeControllerList(page.panel, box, viewSize(doc));
}

/**
 * Draw the list's content into panel, as the approved picture: a title with
 * the team crest ("Team A Controllers"), then one row per controller: its
 * full name (wraps, never cut; "You" in the moon color), its champion count
 * on the right, and under it each champion as its class icon and "Class · ID
 * n". Side effects: replaces panel's children.
 *
 * @param {HTMLElement} panel
 * @param {ControllerListModel} model
 * @returns {void}
 */
export function fillControllerList(panel, model) {
  const doc = panel.ownerDocument;
  /**
   * @param {string} tag
   * @param {string} className
   * @param {string} [text]
   */
  const el = (tag, className, text) => {
    const node = doc.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const title = el("p", "controller-list__title");
  title.append(
    createSvgIcon(doc, model.teamId === 1 ? "team-a" : "team-b", {
      className: "controller-list__crest",
      size: 14,
    }),
    model.title,
  );
  const rows = el("ul", "controller-list__rows");
  for (const row of model.rows) {
    const item = el("li", "controller-list__row");
    const champions = el("span", "controller-list__champions");
    for (const champion of row.champions) {
      const entry = el("span", "controller-list__champion");
      entry.append(
        createSvgIcon(doc, classTokenFromId(champion.classId).glyphKey, {
          className: "controller-list__class",
          size: 14,
        }),
        champion.text,
      );
      champions.append(entry);
    }
    item.append(
      el(
        "span",
        `controller-list__name${row.you ? " controller-list__name--you" : ""}`,
        row.name,
      ),
      el("span", "controller-list__count", row.countWords),
      champions,
    );
    rows.append(item);
  }
  panel.setAttribute("aria-label", model.title);
  panel.replaceChildren(title, rows);
}

/**
 * Place the panel under its trigger and keep it inside the window.
 *
 * Width: PANEL_WIDTH (360 px), or the window width less two margins when
 * narrower. Across: the panel's right edge meets the trigger's right edge
 * when the trigger sits in the right half of the window, else its left edge
 * meets the trigger's left edge; then it is moved to stay WINDOW_MARGIN (8
 * px) inside both window edges. Down: 6 px under the trigger; when the
 * space below is smaller than the space above and too small for the list,
 * it opens above instead. Its height is capped to the space it opens into
 * (it then scrolls). Side effects: the panel's inline position and size.
 *
 * @param {HTMLElement} panel
 * @param {{left: number, right: number, top: number, bottom: number}} anchor The
 *   trigger's box in window coordinates (getBoundingClientRect).
 * @param {{width: number, height: number}} view The window's size in CSS px.
 * @returns {void}
 */
export function placeControllerList(panel, anchor, view) {
  const width = Math.max(0, Math.min(PANEL_WIDTH, view.width - 2 * WINDOW_MARGIN));
  const rightHalf = (anchor.left + anchor.right) / 2 > view.width / 2;
  const wanted = rightHalf ? anchor.right - width : anchor.left;
  const left = Math.max(
    WINDOW_MARGIN,
    Math.min(wanted, view.width - width - WINDOW_MARGIN),
  );
  panel.style.width = `${width}px`;
  panel.style.left = `${Math.round(left)}px`;
  panel.style.maxHeight = "";
  panel.style.top = "0px";
  panel.style.bottom = "";
  const needed = panel.scrollHeight || 0;
  const below = view.height - anchor.bottom - 6 - WINDOW_MARGIN;
  const above = anchor.top - 6 - WINDOW_MARGIN;
  if (needed > below && above > below) {
    panel.style.top = "";
    panel.style.bottom = `${Math.round(view.height - anchor.top + 6)}px`;
    panel.style.maxHeight = `${Math.max(0, Math.floor(above))}px`;
  } else {
    panel.style.top = `${Math.round(anchor.bottom + 6)}px`;
    panel.style.maxHeight = `${Math.max(0, Math.floor(below))}px`;
  }
}
