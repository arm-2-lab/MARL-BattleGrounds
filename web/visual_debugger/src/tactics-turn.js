/**
 * @file The Tactics Turn: the BattleClient and DevClient combat screen's turn
 * deck, its status line and the three live rail cards. A player plans one
 * turn for the champion it orders and locks it in. Every tick is a turn;
 * nothing moves until Lock In, and there is no timer. It also sorts keyboard
 * commands into draft keys, submit keys and the rest; the draft keys come
 * from the one command set (command-set.js), so the deck and the keyboard
 * never disagree.
 *
 * Who calls it: the page start module (main.js) calls createTacticsTurn once,
 * giving it the page elements, the live page state and the page helpers it
 * needs. main.js then calls the returned renderers on every repaint and calls
 * bindCommandDeck once at startup. The pure helpers at the top (DRAFT_KEYS,
 * commandRole, the deferred-draft checks, the deck words and the SVG
 * builders) touch no DOM, so Node tests can import them.
 *
 * What it owns, as drawn on the approved vision page (index.html holds the
 * markup; this module fills it):
 *
 * - The deck (#command-deck): Your Champion (token, class, calling, Agent ID,
 *   "You Control This Agent", life, ‹ Prev and Next ›), 1 Step (the nine-key
 *   pad, X is Stay, and the name of the chosen step; the arrow keys do
 *   nothing), 2 Target (the target chip in its four forms: No Target Yet,
 *   Burst Needs No Target, a seen body with its life, and "Warrior · ID 6 ·
 *   Out Of Sight"; the × Esc clear button; the Allowed Now buttons, the
 *   vision's pills while they fit, else a grid of short buttons, fitPicks), 3 Action
 *   (None, the class's real Basic and Ultimate with reach, the Rune Ring and
 *   the Armed, Auto and Not Allowed tags, and a shake on a refused press) and
 *   4 Lock In (the order list and the button label Lock In Turn, Orders
 *   Sent, Policies Choosing… or Everyone Acts, plus Cancel Policy Work). The
 *   order list is the vision's full sentence for one or two Manual
 *   champions and one short row each (compactOrderParts) for three or more,
 *   so every line shows; each short row stays whole on one line
 *   (fitOrderRow: the step gives way first, then the text steps down to
 *   10 px; the ability and the target are never cut); the deck's
 *   data-orders ("one", "two" or "many",
 *   from the roster, never the view) lets deck.css keep the deck one height
 *   from turn to turn and across view switches.
 * - The deck overlays: Watch Only (a policy's champion is in view, with its
 *   life; Back To Your Warrior), Stopped (policy work was cancelled; Finish And Save; the
 *   phases then rest on You Plan with the policies Waiting) and
 *   the Watching bar when nobody is Manual (Run or Pause and Play One Turn).
 * - The status line (#status-line): the recording seal, the one message line
 *   (#msg; the vision's refusal sentences, written from the host's masks
 *   before anything is sent; the first sentence true for the roster mix,
 *   startGreeting; on the tick the game ends, the game-over step for the
 *   replay's save state, gameOverStep, said again when that state changes;
 *   once Exit was accepted, the closed sentence for the replay's save state,
 *   closedSentence; host notices in the player's words, plainHostNotice),
 *   Keys On or Keys Off, and Restart. The Recording words of the seal (and
 *   of main.js's Recording pill and popover) have one owner here,
 *   recordingWords.
 * - The rail cards: This Turn (#turn-card: tick, title, set pill, Guide /
 *   Hints / Off, Do This Now, the four phases You Plan · Lock In · Policies
 *   Choose · All Act, the checklist of the champions you play and the
 *   Policies block), the Class Guide (#class-card, hidden while the
 *   class_guide switch is off) and the How To Play card (#howto-card, shown
 *   at the Guide level, its lines true for the roster mix, howToPlayLines;
 *   Full Guide opens the help).
 *
 * Life: Your Champion's life, the target chip's and Watch Only's are the one
 * shared Life bar (`life-bar.js`: "Life", a thin bar colored green, yellow or
 * red, and "200 / 200"; owner, 2 October 2026), or words where the champion
 * has fallen or is out of sight (paintLife).
 *
 * Controller names: the Policies block, the "Played By" names (the deck's
 * chip, Watch Only and the Class Guide note) and the status line's "played
 * by ..." all come from the one controller model (controllerRosterFromFrame
 * in match-summary.js, built from the combat configuration and the identity
 * directory, the same facts the scorebug's lines and hover lists use;
 * controller-names.js owns the model and the words). The Policies block
 * lists every policy of both teams with its crest, its full name (wrapping,
 * never cut) and "N Champions", exactly the hover lists' rows without "You";
 * a "Played By" name is cut with "…" past the fixed name width and hovering
 * it while cut drops down its team's list.
 *
 * Phases come from real states: You Plan (nothing in flight), Lock In (a
 * Lock In this page sent is waiting for its answer), Policies Choose (the
 * host's system_operation is loading or thinking) and All Act (the host is
 * finishing the turn, or the page is playing the new tick's choreography).
 *
 * Boundary: this module never sends a request itself; it calls the dispatch
 * functions it is given, and Python checks every command again. It reads the
 * page state but never assigns it. The Replay Viewer has no deck: every deck
 * control stays disabled there and the rail cards stay hidden.
 *
 * Agent POV: everything the deck and the rail cards SHOW comes from the
 * frame's audience: the arena scene (authorizedPresentationSceneView) and the
 * audience's own live inspection of the champion in view
 * (authorizedPresentationInspectionState: its draft and its masks, which in
 * Agent POV name only bodies the team may know). Allowed Now lists only bodies
 * that scene draws; the target chip shows life only for a drawn body (a kept
 * target out of sight reads "Warrior · ID 6 · Out Of Sight" with no life or
 * token); class facts come from the scene's class_mechanics. The researcher
 * branch is read only as command plumbing that is never drawn: which champion
 * is in focus (an ID the host lets the POV team focus: its own champions or an
 * enemy it sees now), whether a policy plays it, the button legality tooltips
 * of the champion you order, the global command slot a click sends, and the
 * staged orders of your own team's Manual champions (your own orders, ruling
 * P1, 3 Oct 2026). The rail rows, the "N Of M Set" pill, Do This Now and the
 * Lock In order list show staged orders only for the viewing team's Manual
 * champions (shownManualIdsOf, ruling L18; another team's Manual champion
 * keeps its clickable rail row with no order, which reads "Out Of Sight"
 * when this view cannot see it and "In Sight · Order Hidden" when it can).
 * "Set" in the pill and the rail means the server holds an order that is not
 * the default Stay + No Combat (owner, 3 Oct 2026), so a reload and both
 * views agree. A champion not
 * in view takes its staged order from the server's pending rows
 * (servedManualOrdersOf: the Oracle frame's own rows, or in Agent POV the
 * researcher branch's rows of the viewing team only), so a view switch, a
 * fresh page and orders kept from earlier ticks all agree with the server.
 * A fallen champion is never counted or asked for orders, and a fall shows
 * in the deck and the rail at its Fall beat, not earlier (fallHoldsOf, from
 * choreography-plan.js beatView, which lists only falls the audience saw).
 * The match summary is read only for the episode tick limit
 * (authorizedPresentationEpisodeLimit), which every audience shares. Nothing
 * here draws on the arena. Agent POV and Oracle View change only what the
 * human sees; this module never feeds the view into a policy input.
 */

import { canonicalAgentIdentity, tickCountLabel } from "./agent-identity.js";
import {
  authorizedOracleCommandSlotForPresentationKey,
  authorizedOracleCommandSlotForPublicAgentId,
  authorizedPresentationAudience,
  authorizedPresentationController,
  authorizedPresentationEpisodeLimit,
  authorizedPresentationHasResearcherSpace,
  authorizedPresentationIdentityRows,
  authorizedPresentationInspectionState,
  authorizedPresentationResearcherInspectionState,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import { beatView } from "./choreography-plan.js";
import { classLoreV1, classNamesV1 } from "./class-documentation.js";
import { COMMAND_SET } from "./command-set.js";
import {
  controllerNameHtml,
  controllerOfChampion,
  NO_CONTROLLER_ROSTER,
  paintNamedSentence,
  policyListRows,
  publishControllerRoster,
} from "./controller-names.js";
import {
  keyboardCommand,
  recordingCommandDecision,
  targetSelectionCommand,
} from "./controls.js";
import { explainLegality } from "./explanations.js";
import { createSvgIcon } from "./icons.js";
import { createLifeBar, lifeBarModel } from "./life-bar.js";
import { championTokenSvg } from "./looks/hud.js";
import { controllerRosterFromFrame } from "./match-summary.js";
import { registerTooltipOwner } from "./tooltip.js";

// The command-set rows whose keys stage part of a draft action.
const DRAFT_COMMAND_ROW_IDS = new Set(["step", "stay", "combat_choice"]);

/**
 * Lower-case keyboard keys that stage part of a draft action, read from the one
 * command set (command-set.js): the eight Step keys (W A S D Q E Z C), X for
 * Stay and 0 / 1 / 2 for No Combat, Basic and Ultimate. The arrow keys are not
 * draft keys: they do nothing in live play. Compare against
 * `command.key.toLowerCase()`. Callers must not change the set. It decides
 * grouping only, never whether Python accepts the key.
 *
 * @type {ReadonlySet<string>}
 */
export const DRAFT_KEYS = new Set(
  COMMAND_SET.filter(({ id }) => DRAFT_COMMAND_ROW_IDS.has(id)).flatMap(({ keys }) =>
    keys.map((key) => key.toLowerCase()),
  ),
);

/**
 * Command types, other than keyboard keys, that may stage a draft before a
 * later Enter: a pointer click on the battlefield and a roster pick.
 *
 * @type {ReadonlySet<string>}
 */
export const DEFERRED_SUBMIT_PREPARATION_COMMAND_TYPES = new Set([
  "battlefield_pointer",
  "roster_selection",
]);

/**
 * Return whether a command can stage a draft before a later Enter.
 * Recognize the preparation command set plus draft, Tab, and Escape keys; this
 * does not check freshness, modifiers, or current command availability.
 * Pure: reads only `command`.
 *
 * @param {Record<string, unknown>} command A browser command record, for
 *   example `{command_type: "keyboard", key: "w", ...}` from keyboardCommand.
 * @returns {boolean} True for a battlefield pointer, a roster pick, a draft key,
 *   Tab or Escape (any letter case); false for everything else.
 */
export function commandPreparesDeferredSubmit(command) {
  if (DEFERRED_SUBMIT_PREPARATION_COMMAND_TYPES.has(String(command.command_type))) {
    return true;
  }
  if (command.command_type !== "keyboard" || typeof command.key !== "string") {
    return false;
  }
  const key = command.key.toLowerCase();
  return DRAFT_KEYS.has(key) || key === "tab" || key === "escape";
}

/**
 * Return whether a keyboard command belongs to the draft-preparation key group.
 * Other command types and non-string keys return false. Pure.
 *
 * @param {Record<string, unknown>} command A browser command record.
 * @returns {boolean} True for a keyboard draft key, Tab or Escape.
 */
export function isDeferredDraftKey(command) {
  return (
    command.command_type === "keyboard" &&
    typeof command.key === "string" &&
    commandPreparesDeferredSubmit(command)
  );
}

/**
 * Return whether a draft-preparation key can be retained during a pending request.
 * Require explicit false Ctrl/Alt/Meta/repeat flags; Shift is allowed only for Tab.
 * A missing flag counts as not false, so the key is not retained. Pure.
 *
 * @param {Record<string, unknown>} command A browser command record with
 *   `shift_key`, `ctrl_key`, `alt_key`, `meta_key` and `repeat` booleans.
 * @returns {boolean} True when the page may queue this key until the current
 *   request settles.
 */
export function isFreshDeferredDraft(command) {
  if (!isDeferredDraftKey(command)) {
    return false;
  }
  const key = String(command.key).toLowerCase();
  return (
    command.ctrl_key === false &&
    command.alt_key === false &&
    command.meta_key === false &&
    command.repeat === false &&
    (command.shift_key === false || key === "tab")
  );
}

/**
 * Classify keyboard commands as draft, interactive-submit, or null.
 * Recognize draft keys and the Lock In keys (Enter/Return/Space aliases and N,
 * which is one tick: Python treats an interactive N as Space) without deciding
 * legality. Pure: reads only `command`.
 *
 * @param {Record<string, unknown>} command A browser command record.
 * @returns {"draft" | "interactive-submit" | null} "draft" for a DRAFT_KEYS key,
 *   "interactive-submit" for Enter, Return, Space, "Spacebar" or N (any case),
 *   and null for any other key or any non-keyboard command.
 */
export function commandRole(command) {
  if (command.command_type !== "keyboard" || typeof command.key !== "string") {
    return null;
  }
  const key = command.key.toLowerCase();
  if (DRAFT_KEYS.has(key)) {
    return "draft";
  }
  if (
    key === "enter" ||
    key === "return" ||
    key === " " ||
    key === "spacebar" ||
    key === "n"
  ) {
    return "interactive-submit";
  }
  return null;
}

/**
 * The movement action names by action id, as the vision page writes them:
 * 0 Stay, 1 North, 2 South, 3 East, 4 West, 5 North-East, 6 North-West,
 * 7 South-East and 8 South-West. Frozen.
 *
 * @type {ReadonlyArray<string>}
 */
export const MOVE_NAMES = Object.freeze([
  "Stay",
  "North",
  "South",
  "East",
  "West",
  "North-East",
  "North-West",
  "South-East",
  "South-West",
]);

/**
 * The Step pad, keyed by each button's `data-key`: the key letter printed on
 * the button, its movement action id (0 to 8) and its name from MOVE_NAMES.
 * X is Stay. The arrow keys have no entry: they do nothing. Frozen.
 *
 * @type {Readonly<Record<string, Readonly<{letter: string, moveAction: number, name: string}>>>}
 */
export const STEP_PAD_KEYS = Object.freeze({
  q: Object.freeze({ letter: "Q", moveAction: 6, name: MOVE_NAMES[6] }),
  w: Object.freeze({ letter: "W", moveAction: 1, name: MOVE_NAMES[1] }),
  e: Object.freeze({ letter: "E", moveAction: 5, name: MOVE_NAMES[5] }),
  a: Object.freeze({ letter: "A", moveAction: 4, name: MOVE_NAMES[4] }),
  x: Object.freeze({ letter: "X", moveAction: 0, name: MOVE_NAMES[0] }),
  d: Object.freeze({ letter: "D", moveAction: 3, name: MOVE_NAMES[3] }),
  z: Object.freeze({ letter: "Z", moveAction: 8, name: MOVE_NAMES[8] }),
  s: Object.freeze({ letter: "S", moveAction: 2, name: MOVE_NAMES[2] }),
  c: Object.freeze({ letter: "C", moveAction: 7, name: MOVE_NAMES[7] }),
});

/**
 * Every fixed label of the deck and its rail cards, in one place, so tests
 * and the page agree. Title Case labels, as on the vision page. Frozen.
 */
export const DECK_LABELS = Object.freeze({
  youControl: "You Control This Agent",
  lockInButton: "Lock In Turn",
  lockInKeys: "Space or Enter",
  lockInHelp: "Space or Enter sends every order of yours at once.",
  ordersSent: "Orders Sent",
  policiesChoosingButton: "Policies Choosing…",
  everyoneActs: "Everyone Acts",
  cancelPolicyWork: "Cancel Policy Work",
  restart: "Restart",
  keysOn: "Keys On",
  keysOff: "Keys Off · Click The Arena",
  keysClosed: "Keys Off",
  noTargetYet: "No Target Yet",
  targetOutOfSight: "Your team cannot see it.",
  allowedNowEmpty: "Nothing is allowed right now.",
  yourTurn: "Your Turn",
  gameStopped: "Game Stopped",
  gameOver: "Game Over",
  closed: "Closed",
  outOfSight: "Out Of Sight",
  orderHidden: "In Sight · Order Hidden",
  policiesChoosing: "Policies Choosing",
  policiesOnly: "Policies Only",
  classGuide: "Class Guide",
});

/** The four phases of one turn, in order, as the This Turn card names them. */
export const TURN_PHASES = Object.freeze([
  "You Plan",
  "Lock In",
  "Policies Choose",
  "All Act",
]);

/** The same four phases while nobody is Manual (Policies Only). */
export const WATCH_PHASES = Object.freeze([
  "Paused",
  "Play",
  "Policies Choose",
  "All Act",
]);

/**
 * The This Turn help levels: "guide" (Do This Now, the dashed beacon on the
 * control it names and the How To Play card; the default), "hints" (Do This
 * Now only) and "off" (all three hidden). The choice is remembered per browser.
 */
export const GUIDE_MODES = /** @type {ReadonlyArray<"guide" | "hints" | "off">} */ (
  Object.freeze(["guide", "hints", "off"])
);

/**
 * Paint attributes for one SVG element in a class color, by class key. The
 * colors live in the page's palette tokens (styles.css owns --class-mage and
 * the rest); the element carries a .tc-<class> class (src/styles/battleclient/
 * base.css sets its CSS color) and paints with currentColor, so no script
 * holds a second copy of a palette color and the SVG stays flat (no style
 * attribute, gradient or url()).
 */
const CLASS_PAINT = Object.freeze({
  mage: "tc-mage",
  warrior: "tc-warrior",
  hunter: "tc-hunter",
  rogue: "tc-rogue",
  priest: "tc-priest",
});

/**
 * Return the attributes that paint one SVG element in a team or class color.
 *
 * Pure. A known team ("A", "B") or class key gives `class="tc-…"` plus
 * `<attribute>="currentColor"`; the .tc-… rules in
 * src/styles/battleclient/base.css set the color from the palette tokens.
 * Anything else gives the moon fallback color directly.
 *
 * @param {"fill" | "stroke"} attribute Which SVG paint attribute to set.
 * @param {string | undefined} key "A" or "B" for a team, or a class key
 *   ("mage", "warrior", "hunter", "rogue", "priest").
 * @returns {string} The attribute text, for example
 *   `class="tc-b" stroke="currentColor"`.
 */
function paintAttributes(attribute, key) {
  const name =
    key === "A"
      ? "tc-a"
      : key === "B"
        ? "tc-b"
        : /** @type {Record<string, string>} */ (CLASS_PAINT)[key ?? ""];
  return name
    ? `class="${name}" ${attribute}="currentColor"`
    : `${attribute}="${MOON}"`;
}

/**
 * The vision page's ability symbols by class key ("ab-…" ids in the page's
 * hidden SVG definitions) and the glyph icons.js draws when a symbol is not
 * on the page.
 */
const ABILITY_ICONS = Object.freeze({
  mage: {
    basic: "ab-lance",
    ultimate: "ab-burst",
    basicGlyph: "activation-basic-damage",
    ultimateGlyph: "activation-burst",
  },
  warrior: {
    basic: "ab-bash",
    ultimate: "ab-charge",
    basicGlyph: "activation-basic-damage",
    ultimateGlyph: "activation-charge",
  },
  hunter: {
    basic: "ab-arrows",
    ultimate: "ab-trap",
    basicGlyph: "activation-basic-damage",
    ultimateGlyph: "activation-trap",
  },
  rogue: {
    basic: "ab-twin",
    ultimate: "ab-poison",
    basicGlyph: "activation-basic-damage",
    ultimateGlyph: "activation-poison",
  },
  priest: {
    basic: "ab-mend",
    ultimate: "ab-salvation",
    basicGlyph: "activation-basic-heal",
    ultimateGlyph: "activation-holy-word",
  },
});

/** Rune Ring tick angles in lighting order (degrees, clockwise from north). */
const RUNE_ORDER = Object.freeze([
  165, 195, 135, 225, 105, 255, 75, 285, 45, 315, 15, 345,
]);

/** The vision's moonlight and ink. */
const MOON = "#C9D4E8";
const INK = "#0A0D14";

/**
 * Write a tick count: "1 tick" and "3 ticks", or "1 Tick" and "3 Ticks" in
 * Title Case. Pure.
 *
 * @param {number} count Whole ticks (decision steps).
 * @param {boolean} [title] True for Title Case. Default false.
 * @returns {string}
 */
export function ticksText(count, title = false) {
  return `${count} ${title ? "Tick" : "tick"}${count === 1 ? "" : "s"}`;
}

/**
 * Write a reach or a sight radius in world units with at least one decimal
 * ("1.5", "3.0", "5.75"). Pure.
 *
 * @param {unknown} value A finite number of world units.
 * @returns {string | null} Null for a value that is not a finite number.
 */
export function reachText(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  const hundredths = Math.round(number * 100);
  return hundredths % 10 === 0
    ? (hundredths / 100).toFixed(1)
    : (hundredths / 100).toFixed(2);
}

/**
 * Escape text for HTML: `&`, `<`, `>` and `"`. Pure.
 *
 * @param {unknown} value Any value; it is turned into a string first.
 * @returns {string}
 */
export function escapeHtml(value) {
  return String(value).replace(
    /[&<>"]/gu,
    (character) =>
      /** @type {Record<string, string>} */ ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
      })[character] ?? character,
  );
}

/**
 * Return the plain facts of one champion for the deck and the rail, read from
 * one arena-scene agent or one roster agent.
 *
 * Pure. In Agent POV pass only an agent from the POV arena scene when its
 * life may be shown; a roster agent carries no life, so lifeText is null.
 *
 * @param {unknown} agent An agent record (`class_id`, `team_id`,
 *   `public_agent_id`, `display_agent_id`, `current_health`, `maximum_health`
 *   or `max_health`, `life_state`), or anything else.
 * @returns {Readonly<{label: string, name: string, number: string,
 *   nameWithId: string, className: string, calling: string, classKey: string,
 *   team: "A" | "B" | null, lifeText: string | null,
 *   life: import("./life-bar.js").LifeBarModel | null, alive: boolean}> | null}
 *   label is "Mage · ID 5"; nameWithId is "Mage (ID 5)"; number is "5";
 *   lifeText is "62 / 100" (rounded) and life the shared Life bar's model
 *   (`life-bar.js` lifeBarModel), both null when the agent has no finite life
 *   with a maximum above 0. Null for a non-record.
 */
export function championFacts(agent) {
  if (agent === null || typeof agent !== "object" || Array.isArray(agent)) {
    return null;
  }
  const record = /** @type {Record<string, any>} */ (agent);
  const identity = canonicalAgentIdentity(record);
  const names = classNamesV1(record.class_id);
  const health = Number(record.current_health);
  const maximum = Number(record.maximum_health ?? record.max_health);
  const finite = Number.isFinite(health) && Number.isFinite(maximum) && maximum > 0;
  const number = identity.publicIdentity.replace(/^Agent ID /u, "");
  const name = names?.className ?? identity.classLabel;
  return Object.freeze({
    label: identity.title,
    name,
    number,
    nameWithId: `${name} (ID ${number})`,
    className: name,
    calling: names?.calling ?? "",
    classKey: identity.accent,
    team: record.team_id === 1 ? "A" : record.team_id === 2 ? "B" : null,
    lifeText: finite ? `${Math.round(health)} / ${Math.round(maximum)}` : null,
    life: finite ? lifeBarModel(health, maximum) : null,
    alive: record.life_state === "alive" || record.alive === true,
  });
}

/**
 * Return what the target chip shows for the staged target.
 *
 * Pure. `sceneAgents` must be the arena scene agents of the frame's audience,
 * so in Agent POV a target the team cannot see is "out_of_sight": it keeps its
 * pick and its name (from `rosterAgents`, the static match roster) but shows
 * no life, token or position.
 *
 * @param {unknown} target The draft target record (`target_kind`
 *   "no_target", "visible_authorized_agent" or "axis_only_authorized_agent",
 *   and `target_public_agent_id`), or null for none.
 * @param {ReadonlyArray<Record<string, any>>} sceneAgents Arena scene agents.
 * @param {ReadonlyArray<Record<string, any>>} rosterAgents Roster agents
 *   (class and ID only) used to name an out-of-sight pick. Default [].
 * @returns {Readonly<{state: "none"} | {state: "seen", facts: ReturnType<typeof championFacts>}
 *   | {state: "out_of_sight", label: string, facts: ReturnType<typeof championFacts>}>}
 *   For out_of_sight, facts come from the roster (class and ID, no life).
 */
export function targetChipView(target, sceneAgents, rosterAgents = []) {
  const record =
    target !== null && typeof target === "object" && !Array.isArray(target)
      ? /** @type {Record<string, any>} */ (target)
      : null;
  if (
    record === null ||
    record.target_kind === "no_target" ||
    typeof record.target_public_agent_id !== "string"
  ) {
    return Object.freeze({ state: "none" });
  }
  const seen = sceneAgents.find(
    (agent) => agent?.public_agent_id === record.target_public_agent_id,
  );
  if (seen !== undefined) {
    return Object.freeze({ state: "seen", facts: championFacts(seen) });
  }
  const known = rosterAgents.find(
    (agent) => agent?.public_agent_id === record.target_public_agent_id,
  );
  const named = known ?? record;
  return Object.freeze({
    state: "out_of_sight",
    label: canonicalAgentIdentity(named).outOfSightLabel,
    facts: championFacts({
      class_id: named.class_id,
      team_id: named.team_id,
      public_agent_id: record.target_public_agent_id,
      display_agent_id: named.display_agent_id,
    }),
  });
}

/**
 * List the Allowed Now target buttons from the audience's decision mask.
 *
 * A target is listed when the mask names a visible agent that the audience's
 * arena scene draws and the host allows the Basic or the Ultimate on it now.
 * The champion itself is listed only when the mask allows it (a Priest healing
 * itself). Pure; nothing is worked out from distances. Rows are sorted by
 * their shown ID, as on the vision page.
 *
 * @param {unknown} mask The decision mask (`target_actions`,
 *   `target_use_ultimate_joint_mask`), or anything else for none.
 * @param {ReadonlyArray<Record<string, any>>} sceneAgents Arena scene agents
 *   of the frame's audience.
 * @returns {ReadonlyArray<Readonly<{targetAction: number, publicAgentId:
 *   string, label: string, number: string, name: string, team: "A" | "B" |
 *   null, classKey: string, basic: boolean, ultimate: boolean}>>}
 */
export function allowedNowTargets(mask, sceneAgents) {
  const record =
    mask !== null && typeof mask === "object" && !Array.isArray(mask)
      ? /** @type {Record<string, any>} */ (mask)
      : null;
  if (record === null || !Array.isArray(record.target_actions)) {
    return Object.freeze([]);
  }
  const pairs = Array.isArray(record.target_use_ultimate_joint_mask)
    ? record.target_use_ultimate_joint_mask
    : [];
  /** @type {Array<Readonly<{targetAction: number, publicAgentId: string, label: string, number: string, name: string, team: "A" | "B" | null, classKey: string, basic: boolean, ultimate: boolean}>>} */
  const rows = [];
  for (const target of record.target_actions) {
    const targetAction = Number(target?.target_action);
    if (
      !Number.isInteger(targetAction) ||
      targetAction === 0 ||
      target?.target_kind !== "visible_authorized_agent"
    ) {
      continue;
    }
    const pair = Array.isArray(pairs[targetAction]) ? pairs[targetAction] : [];
    const basic = pair[0] === true;
    const ultimate = pair[1] === true;
    const body = sceneAgents.find(
      (agent) => agent?.public_agent_id === target.target_public_agent_id,
    );
    if ((!basic && !ultimate) || body === undefined) continue;
    const facts = championFacts(body);
    if (facts === null) continue;
    rows.push(
      Object.freeze({
        targetAction,
        publicAgentId: String(target.target_public_agent_id),
        label: facts.label,
        number: facts.number,
        name: facts.name,
        team: facts.team,
        classKey: facts.classKey,
        basic,
        ultimate,
      }),
    );
  }
  rows.sort((left, right) => Number(left.number) - Number(right.number));
  return Object.freeze(rows);
}

/**
 * Write the Rune Ring text for an Ultimate cooldown: "Ready" at 0, else the
 * ticks left ("1 Tick", "3 Ticks"). Anything that is not a whole number of
 * ticks gives null (nothing to show, for example an unseen champion).
 *
 * @param {unknown} cooldown Ticks until the Ultimate is ready.
 * @returns {string | null}
 */
export function runeRingText(cooldown) {
  if (!Number.isInteger(cooldown) || Number(cooldown) < 0) return null;
  return cooldown === 0 ? "Ready" : tickCountLabel(cooldown);
}

/**
 * Return how many of the Rune Ring's twelve ticks are lit, as on the vision
 * page: all twelve when the Ultimate is ready, otherwise two for every five
 * ticks already waited out of the cooldown period.
 *
 * Pure.
 *
 * @param {unknown} cooldown Ticks until the Ultimate is ready (0 or more).
 * @param {unknown} period The class's Ultimate cooldown in ticks (30 today).
 * @returns {number} 0 to 12; 0 for input that is not a whole number.
 */
export function runeRingLit(cooldown, period) {
  if (!Number.isInteger(cooldown) || Number(cooldown) < 0) return 0;
  if (cooldown === 0) return 12;
  const total = Number.isInteger(period) && Number(period) > 0 ? Number(period) : 30;
  return Math.max(0, Math.min(12, 2 * Math.floor((total - Number(cooldown)) / 5)));
}

/**
 * Return the effective action lane of a draft: "basic" only when a Basic is
 * armed on a target, "ultimate" when the Ultimate is armed, else "none".
 *
 * Pure.
 *
 * @param {unknown} draft The draft action (`armed_lane`, `target_action`), or
 *   null for no draft.
 * @returns {"none" | "basic" | "ultimate"}
 */
export function effectiveLane(draft) {
  if (draft === null || typeof draft !== "object") return "none";
  const record = /** @type {Record<string, any>} */ (draft);
  if (record.armed_lane === "ultimate") return "ultimate";
  if (record.armed_lane === "basic" && Number(record.target_action) > 0) return "basic";
  return "none";
}

/**
 * Return the three action slots (None, the class's Basic, its Ultimate) as
 * the vision page draws them.
 *
 * Pure.
 *
 * @param {{
 *   classId: unknown,
 *   mechanics: Record<string, any> | null,
 *   cooldown: unknown,
 *   lane: "none" | "basic" | "ultimate",
 *   autoArmed: boolean,
 *   basicOk: boolean,
 *   ultimateOk: boolean,
 *   terminal?: boolean,
 * }} options mechanics is the class's class_mechanics row (reach and
 *   cooldown period), or null; cooldown is the champion's Ultimate cooldown in
 *   ticks, or null when unknown; lane is effectiveLane of its draft; autoArmed
 *   is true when the game armed the Basic by itself; basicOk and ultimateOk
 *   say whether the host allows that lane now (on the picked target, or with
 *   no target for an Ultimate that takes none). terminal (default false) is
 *   true once the game is over: then neither lane is allowed, so an unarmed
 *   Basic or Ultimate reads Not Allowed whatever the host's last mask said.
 * @returns {ReadonlyArray<Readonly<{lane: "none" | "basic" | "ultimate",
 *   key: string, name: string, sub: string, ready: string | null, armed:
 *   boolean, no: boolean, tag: "Armed" | "Auto" | "Not Allowed" | null,
 *   lit: number}>>} Always three slots. sub is "Only Move", "Reach 1.5" or "No
 *   Target"; ready is "Ready" or "In 3 Ticks" (Ultimate only); lit is the
 *   number of lit Rune Ring ticks (Ultimate only, else 0).
 */
export function actionSlotViews(options) {
  const names = classNamesV1(options.classId);
  const mechanics = options.mechanics;
  const basicReach = reachText(mechanics?.basic_interaction_radius);
  const ultimateRadius = Number(mechanics?.ultimate_interaction_radius);
  const ultimateNeedsNoTarget =
    mechanics?.ultimate_target_mode === "target_none" ||
    (Number.isFinite(ultimateRadius) && ultimateRadius <= 0);
  const cooldown = Number.isInteger(options.cooldown) ? Number(options.cooldown) : null;
  const basicArmed = options.lane === "basic";
  const ultimateArmed = options.lane === "ultimate";
  const basicOk = options.basicOk && options.terminal !== true;
  const ultimateOk = options.ultimateOk && options.terminal !== true;
  return Object.freeze([
    Object.freeze({
      lane: /** @type {const} */ ("none"),
      key: "0",
      name: "No Combat",
      sub: "Only Move",
      ready: null,
      armed: options.lane === "none",
      no: false,
      tag: options.lane === "none" ? /** @type {const} */ ("Armed") : null,
      lit: 0,
    }),
    Object.freeze({
      lane: /** @type {const} */ ("basic"),
      key: "1",
      name: names?.basicName ?? "Basic",
      sub: basicReach === null ? "" : `Reach ${Number(basicReach).toFixed(1)}`,
      ready: null,
      armed: basicArmed,
      no: !basicArmed && !basicOk,
      tag: basicArmed
        ? options.autoArmed
          ? /** @type {const} */ ("Auto")
          : /** @type {const} */ ("Armed")
        : basicOk
          ? null
          : /** @type {const} */ ("Not Allowed"),
      lit: 0,
    }),
    Object.freeze({
      lane: /** @type {const} */ ("ultimate"),
      key: "2",
      name: names?.ultimateName ?? "Ultimate",
      sub: ultimateNeedsNoTarget
        ? "No Target"
        : reachText(ultimateRadius) === null
          ? ""
          : `Reach ${reachText(ultimateRadius)}`,
      ready:
        cooldown === null
          ? null
          : cooldown === 0
            ? "Ready"
            : `In ${ticksText(cooldown, true)}`,
      armed: ultimateArmed,
      no: !ultimateArmed && !ultimateOk,
      tag: ultimateArmed
        ? /** @type {const} */ ("Armed")
        : ultimateOk
          ? null
          : /** @type {const} */ ("Not Allowed"),
      lit: runeRingLit(cooldown, mechanics?.ultimate_cooldown_steps),
    }),
  ]);
}

/**
 * Write one champion's staged order as one plain sentence, in the vision
 * page's words, for 4 Lock In and the checklist.
 *
 * Pure. Examples: "Charge the Mage (ID 5), then stay where you land.",
 * "Shield Bash on the Mage (ID 5), then step North.", "Step West, with No
 * Combat." and "Stay, with No Combat (the default)." A fallen champion reads
 * "None. Your Warrior is fallen and returns in 3 ticks."
 *
 * @param {{
 *   touched: boolean,
 *   moveAction: unknown,
 *   lane: "none" | "basic" | "ultimate",
 *   classId: unknown,
 *   ultimateNeedsNoTarget: boolean,
 *   targetName: string | null,
 *   fallen?: {className: string, returnsIn: number | null} | null,
 * }} order touched is false while the order is still the default one;
 *   moveAction is the movement id 0 to 8; targetName is "Mage (ID 5)" or
 *   null; fallen is set while the champion is fallen (returnsIn in ticks, or
 *   null when unknown).
 * @returns {string}
 */
export function orderSentence(order) {
  if (order.fallen) {
    const back =
      order.fallen.returnsIn === null
        ? "and returns on the next respawn wave"
        : `and returns in ${ticksText(order.fallen.returnsIn)}`;
    return `None. Your ${order.fallen.className} is fallen ${back}.`;
  }
  if (!order.touched) return "Stay, with No Combat (the default).";
  const moveId = Number.isInteger(order.moveAction) ? Number(order.moveAction) : 0;
  const moveName = MOVE_NAMES[moveId] ?? "Stay";
  const names = classNamesV1(order.classId);
  if (order.lane === "none" || (order.lane === "basic" && order.targetName === null)) {
    return `${moveId === 0 ? "Stay" : `Step ${moveName}`}, with No Combat.`;
  }
  const basicName = names?.basicName ?? "Basic";
  const ultimateName = names?.ultimateName ?? "Ultimate";
  const action =
    order.lane === "basic"
      ? `${basicName} on the ${order.targetName}`
      : order.ultimateNeedsNoTarget || order.targetName === null
        ? `${ultimateName} (no target)`
        : `${ultimateName} the ${order.targetName}`;
  const charge = order.lane === "ultimate" && names?.className === "Warrior";
  const tail =
    moveId === 0
      ? charge
        ? "stay where you land"
        : "stay where you are"
      : `step ${moveName}${charge ? " from where you land" : ""}`;
  return `${action}, then ${tail}.`;
}

/**
 * Return the turn phase from real states.
 *
 * Pure. "plan" once the game is stopped (stopped: the host asks for a fresh
 * start), whatever else is true: no turn can be played any more, so the
 * phases show You Plan and the policies Waiting, as on the vision page after
 * Cancel Policy Work, even while the cancelled work still winds down
 * ("cancelling"). Otherwise "choosing" while the host's policy work is
 * loading, thinking or cancelling, and between turns while Run plays
 * (running: the policies are always about to choose); "act" while it is
 * finishing, or while the page plays the new tick (choreographyBlocking);
 * "sent" while a Lock In this page sent waits for its answer; otherwise
 * "plan".
 *
 * @param {{operationState: unknown, lockSent: boolean, choreographyBlocking:
 *   boolean, running?: boolean, stopped?: boolean}} input operationState is
 *   the host's `system_operation.state` ("loading", "thinking",
 *   "cancelling", "finishing", "failed") or undefined when no policy work is
 *   out. running is true while Run plays (default false). stopped is the
 *   frame's `fresh_launch_required` (default false).
 * @returns {"plan" | "sent" | "choosing" | "act"}
 */
export function turnPhase(input) {
  if (input.stopped === true) return "plan";
  if (
    input.operationState === "loading" ||
    input.operationState === "thinking" ||
    input.operationState === "cancelling"
  ) {
    return "choosing";
  }
  if (input.operationState === "finishing" || input.choreographyBlocking) return "act";
  if (input.lockSent) return "sent";
  return input.running ? "choosing" : "plan";
}

/**
 * Return the This Turn card title and the Lock In button label for a phase.
 * Once Exit was accepted (closed) the title and the button read "Closed",
 * whatever else is true, because no turn can be played any more. At game
 * over (terminal) the disabled button reads "Game Over", even while a
 * restart waits for the replay to save, instead of a turn phase such as
 * "Everyone Acts".
 *
 * Pure.
 *
 * @param {{phase: "plan" | "sent" | "choosing" | "act", stopped: boolean,
 *   terminal: boolean, watchMode: boolean, running: boolean, closed?:
 *   boolean}} input closed is true after an accepted Exit (default false).
 * @returns {Readonly<{title: string, lockLabel: string}>}
 */
export function turnTitles(input) {
  const lockLabel = input.closed
    ? DECK_LABELS.closed
    : input.terminal
      ? DECK_LABELS.gameOver
      : input.phase === "sent"
        ? DECK_LABELS.ordersSent
        : input.phase === "choosing"
          ? DECK_LABELS.policiesChoosingButton
          : input.phase === "act"
            ? DECK_LABELS.everyoneActs
            : DECK_LABELS.lockInButton;
  const title = input.closed
    ? DECK_LABELS.closed
    : input.stopped
      ? DECK_LABELS.gameStopped
      : input.terminal
        ? DECK_LABELS.gameOver
        : input.watchMode
          ? input.running
            ? "Watching · Running"
            : "Watching · Paused"
          : input.phase === "plan"
            ? DECK_LABELS.yourTurn
            : input.phase === "sent"
              ? DECK_LABELS.ordersSent
              : input.phase === "choosing"
                ? DECK_LABELS.policiesChoosing
                : DECK_LABELS.everyoneActs;
  return Object.freeze({ title, lockLabel });
}

/**
 * Write the game-over step: the rail's Do This Now at game over and the
 * status line on the tick the game ends. It names only what really exists
 * for the replay's save state, so it never offers Finish And Save once the
 * game has saved itself.
 *
 * - No recording (null or not an object): "The game is over. Press Restart
 *   for a new match."
 * - lifecycle "saved" or "reviewing": the game is saved; Open Replay is in
 *   the Saved menu at the top.
 * - lifecycle "recording" with finish_available true: Finish And Save is
 *   offered (its button shows).
 * - lifecycle "sealed" or "finalized_unsaved": the capture is closed and the
 *   server is saving the replay in the background: "Saving the replay…" (the
 *   Recording pill and seal say "Saving Replay…" then).
 * - lifecycle "persistence_failed": the save failed; Retry Save and Save As
 *   are in the Save Failed menu at the top.
 * - anything else (for example "discarded"): Restart only.
 *
 * Pure.
 *
 * @param {unknown} recording The frame's recording status (`lifecycle`,
 *   `finish_available`), or null when the match records nothing.
 * @returns {string}
 */
export function gameOverStep(recording) {
  const status =
    recording !== null && typeof recording === "object" && !Array.isArray(recording)
      ? /** @type {Record<string, unknown>} */ (recording)
      : null;
  const lifecycle = status?.lifecycle;
  if (lifecycle === "saved" || lifecycle === "reviewing") {
    return "The game is over and saved. Press Restart for a new match, or click Saved at the top, then Open Replay.";
  }
  if (lifecycle === "recording" && status?.finish_available === true) {
    return "The game is over. Press Restart for a new match, or Finish And Save.";
  }
  if (lifecycle === "sealed" || lifecycle === "finalized_unsaved") {
    return "The game is over. Saving the replay…";
  }
  if (lifecycle === "persistence_failed") {
    return "The game is over, but the replay did not save. Click Save Failed at the top, then Retry Save or Save As.";
  }
  return "The game is over. Press Restart for a new match.";
}

/**
 * Write the sentence the status line and Do This Now show once Exit was
 * accepted, true for the replay's save state. The server saves every played
 * tick before it closes, so only these three cases exist:
 *
 * - lifecycle "saved" or "reviewing": "The match is closed. The game so far
 *   is saved. You can close this tab."
 * - lifecycle "recording" (Exit before the first tick: the server closes
 *   without writing a replay, as for Ctrl-C): "The match is closed. No tick
 *   was played, so no replay was saved. You can close this tab."
 * - no recording (null: the match records nothing) or any other lifecycle:
 *   "The match is closed. You can close this tab."
 *
 * Pure.
 *
 * @param {unknown} recording The recording status that came with the Exit
 *   answer (`lifecycle`), or null when the match records nothing.
 * @returns {string}
 */
export function closedSentence(recording) {
  const lifecycle =
    recording !== null && typeof recording === "object" && !Array.isArray(recording)
      ? /** @type {Record<string, unknown>} */ (recording).lifecycle
      : null;
  if (lifecycle === "saved" || lifecycle === "reviewing") {
    return "The match is closed. The game so far is saved. You can close this tab.";
  }
  if (lifecycle === "recording") {
    return "The match is closed. No tick was played, so no replay was saved. You can close this tab.";
  }
  return "The match is closed. You can close this tab.";
}

/**
 * Write the Recording pill, the Recording popover's state line and the
 * status line's seal for one recording state, in the vision's words. The one
 * owner of these words: main.js paints the pill and the popover with it
 * (renderRecordingControls) and the deck paints the seal (renderSeal).
 *
 * - "recording": pill "Recording · Tick 91", popover "0 Ticks Kept Since
 *   Tick 91" (ticks kept, then the tick capture began), seal "Recording ·
 *   Tick 91 of 300" ("of 300" only when limit is known).
 * - "sealed" or "finalized_unsaved" (the game is over and the server saves
 *   the replay in the background): "Saving Replay…", "Saving Replay… · 300
 *   Ticks", "Saving Replay…".
 * - "saved" or "reviewing": "Saved", "Saved · 91 Ticks", "Saved · 91 Ticks".
 * - "persistence_failed": "Save Failed · Retry", "Save Failed · 91 Ticks
 *   Kept", "Save Failed · Retry".
 * - closed (Exit was accepted) while the recording is still "recording" with
 *   no tick kept: the server closed without a replay (Exit before the first
 *   tick), so "No Replay", "No Replay · No Tick Was Played", "No Replay".
 * - any other lifecycle (for example "discarded"): "Recording · Discarded"
 *   in all three, with "· N Ticks" on the popover.
 *
 * Pure.
 *
 * @param {Readonly<Record<string, any>>} recording The recording status
 *   (`lifecycle` and `captured_transition_count`, a count of kept ticks; a
 *   value that is not an integer reads 0).
 * @param {number} tick The installed tick (simulator steps), a public fact.
 * @param {number | null} limit The episode's tick limit, or null when unknown.
 * @param {boolean} [closed] True once Exit was accepted. Default false.
 * @returns {Readonly<{pill: string, popover: string, seal: string}>}
 */
export function recordingWords(recording, tick, limit, closed = false) {
  const kept = Number.isInteger(recording.captured_transition_count)
    ? Number(recording.captured_transition_count)
    : 0;
  const ticks = `${kept} Tick${kept === 1 ? "" : "s"}`;
  const lifecycle = String(recording.lifecycle);
  /** @param {string} pill @param {string} popover @param {string} seal */
  const words = (pill, popover, seal) => Object.freeze({ pill, popover, seal });
  if (lifecycle === "saved" || lifecycle === "reviewing") {
    return words("Saved", `Saved · ${ticks}`, `Saved · ${ticks}`);
  }
  if (lifecycle === "persistence_failed") {
    return words(
      "Save Failed · Retry",
      `Save Failed · ${ticks} Kept`,
      "Save Failed · Retry",
    );
  }
  if (lifecycle === "sealed" || lifecycle === "finalized_unsaved") {
    return words("Saving Replay…", `Saving Replay… · ${ticks}`, "Saving Replay…");
  }
  if (lifecycle === "recording" && closed && kept === 0) {
    return words("No Replay", "No Replay · No Tick Was Played", "No Replay");
  }
  if (lifecycle === "recording") {
    return words(
      `Recording · Tick ${tick}`,
      `${ticks} Kept Since Tick ${Math.max(0, tick - kept)}`,
      `Recording · Tick ${tick}${limit === null ? "" : ` of ${limit}`}`,
    );
  }
  const label = `Recording · ${lifecycle
    .replaceAll("_", " ")
    .replace(/\b\w/gu, (letter) => letter.toUpperCase())}`;
  return words(label, `${label} · ${ticks}`, label);
}

/**
 * Return the player's words for a host notice the status line shows, or the
 * notice unchanged when it has none.
 *
 * - "The episode ended and its replay was saved." becomes the saved
 *   game-over step (gameOverStep).
 * - "The episode ended, but replay saving failed; ..." becomes the failed
 *   game-over step.
 * - Host and page notices in developer words become plain sentences: a
 *   refused step, Basic, Ultimate or target ("... unavailable in the current
 *   action mask ...") says "... is not allowed ... Your order is unchanged.";
 *   "Agent g5 is unavailable in this view." says the champion cannot be
 *   picked in this view; "Repeated submission input ignored." says the Lock
 *   In was already sent; "Scientific controls are fenced ..." and "The
 *   episode is terminal; reset to continue." say why orders cannot change;
 *   a stale tab, a command-ID conflict or a duplicate says the page shows the
 *   latest state and nothing was sent twice; "Debugger shutdown requested."
 *   says the BattleClient is closing.
 *
 * Pure.
 *
 * @param {string} text The notice as the host or the page wrote it.
 * @returns {string}
 */
export function plainHostNotice(text) {
  if (text === "The episode ended and its replay was saved.") {
    return gameOverStep({ lifecycle: "saved" });
  }
  if (text.startsWith("The episode ended, but replay saving failed")) {
    return gameOverStep({ lifecycle: "persistence_failed" });
  }
  if (/^Movement action \d+ is unavailable in the current action mask/u.test(text)) {
    return "That step is not allowed right now. Your order is unchanged.";
  }
  if (text.startsWith("Basic requires a selected target;")) {
    return "Your Basic needs a target first. Your order is unchanged.";
  }
  const lane =
    /^(.+) is unavailable for the selected target in the current action mask/u.exec(
      text,
    );
  if (lane !== null) {
    return `${lane[1]} is not allowed on this target right now. Your order is unchanged.`;
  }
  if (/^Target action \d+ is unavailable in the current authorized POV\./u.test(text)) {
    return "That target cannot be picked in Agent POV right now. Your order is unchanged.";
  }
  if (/^Agent g\d+ is unavailable in this view\./u.test(text)) {
    return "That champion cannot be picked in this view.";
  }
  if (text === "Repeated submission input ignored.") {
    return "Your Lock In was already sent for this turn.";
  }
  if (text.startsWith("Scientific controls are fenced")) {
    return "This game is saved and closed, so orders cannot change. Press R to play again.";
  }
  if (text === "The episode is terminal; reset to continue.") {
    return "The game is over, so orders cannot change. Press R to play again.";
  }
  if (
    text.startsWith("This tab was stale.") ||
    text.startsWith("The service rejected a command-ID conflict.") ||
    text.startsWith("A mixed authority GET was discarded")
  ) {
    return "The game had moved on, so the page now shows the latest state. Nothing was sent twice.";
  }
  if (
    text === "Duplicate command recognized; it was not applied again." ||
    text === "Command already processed; current frame returned."
  ) {
    return "That input was already applied once, so it was not applied again.";
  }
  if (text === "Debugger shutdown requested.") {
    return "Closing the BattleClient.";
  }
  return text;
}

/**
 * Write the first status-line sentence of a game, true for its roster mix.
 *
 * - Nobody Manual (watchMode): "Nobody is yours this game. The match starts
 *   paused."
 * - One Manual champion on each team: "You play one champion on each team.
 *   Each keeps its own order until you lock in."
 * - Several Manual champions: how many on each team ("You play five champions
 *   on Team A. ..."; with both teams, "You play two champions on Team A and
 *   one on Team B. ...").
 * - One Manual champion: the view's own sentence, Agent POV ("you see only
 *   what your team sees") or Oracle View.
 *
 * Pure.
 *
 * @param {{watchMode: boolean, teamA: number, teamB: number, pov: boolean}} input
 *   teamA and teamB count the Manual champions on each team (0 to 5); pov is
 *   true in Agent POV.
 * @returns {string}
 */
export function startGreeting(input) {
  const KEEP = "Each keeps its own order until you lock in.";
  const total = input.teamA + input.teamB;
  if (input.watchMode || total === 0) {
    return "Nobody is yours this game. The match starts paused.";
  }
  if (input.teamA === 1 && input.teamB === 1) {
    return `You play one champion on each team. ${KEEP}`;
  }
  if (total > 1) {
    /** @param {number} count @param {boolean} [noun] */
    const count = (count, noun = true) =>
      `${NUMBER_WORDS[count] ?? String(count)}${noun ? ` champion${count === 1 ? "" : "s"}` : ""}`;
    const part =
      input.teamA > 0 && input.teamB > 0
        ? `${count(input.teamA)} on Team A and ${count(input.teamB, false)} on Team B`
        : input.teamA > 0
          ? `${count(input.teamA)} on Team A`
          : `${count(input.teamB)} on Team B`;
    return `You play ${part}. ${KEEP}`;
  }
  return input.pov
    ? "Agent POV: you see only what your team sees. Your orders stay private."
    : "Oracle View: every champion is shown, even those your team cannot see.";
}

/** Number words for counts of champions (0 to 10). */
const NUMBER_WORDS = Object.freeze([
  "no",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
]);

/**
 * The How To Play card's lines for one Manual champion, exactly as on the
 * approved vision page (index.html holds the same markup).
 */
const HOW_TO_PLAY_STEP =
  "Step with <kbd>Q</kbd><kbd>W</kbd><kbd>E</kbd> <kbd>A</kbd><kbd>D</kbd> <kbd>Z</kbd><kbd>S</kbd><kbd>C</kbd>. <kbd>X</kbd> is Stay.";
const HOW_TO_PLAY_TARGET =
  "Hold <kbd>Shift</kbd> and click a body with a <b>dashed ring</b>. Then pick <kbd>0</kbd> <kbd>1</kbd> <kbd>2</kbd>.";

/**
 * Return the How To Play card's lines (HTML of each list item's text), true
 * for the roster mix.
 *
 * - One Manual champion: the vision page's four lines ("The silver ring is
 *   your champion. Policies play the rest.", the step keys, Shift + click and
 *   0 1 2, "Space locks in. Everyone acts at once.").
 * - Several Manual champions: "The silver rings are your champions." and
 *   "Policies play the rest." (or "You play every champion." when no policy
 *   plays), the same step and target lines, "Space locks in all your orders.
 *   Everyone acts at once." and a fifth line on how to order another
 *   champion (click its row above).
 * - Nobody Manual (Policies Only): what really works there: Run, Play One
 *   Turn, Pause and looking at a champion; no step, target or Lock In keys.
 *
 * Pure. The markup uses only <b> and <kbd> with fixed words.
 *
 * @param {{manualCount: number, policyCount: number}} input manualCount
 *   counts the Manual champions of both teams (0 to 10); policyCount counts
 *   the champions a policy plays.
 * @returns {string[]} One HTML string per line, in order.
 */
export function howToPlayLines(input) {
  if (input.manualCount <= 0) {
    return [
      "Policies play every champion. Nobody is yours this game.",
      "Press <b>Run</b> to watch them play, or <b>Play One Turn</b> to step once.",
      "<b>Pause</b> stops after the current turn.",
      "Click a portrait or a body to look at that champion.",
    ];
  }
  if (input.manualCount === 1) {
    return [
      "The <b>silver ring</b> is your champion. Policies play the rest.",
      HOW_TO_PLAY_STEP,
      HOW_TO_PLAY_TARGET,
      "<kbd>Space</kbd> locks in. Everyone acts at once.",
    ];
  }
  return [
    `The <b>silver rings</b> are your champions. ${
      input.policyCount > 0 ? "Policies play the rest." : "You play every champion."
    }`,
    HOW_TO_PLAY_STEP,
    HOW_TO_PLAY_TARGET,
    "<kbd>Space</kbd> locks in all your orders. Everyone acts at once.",
    "Click a champion's row above to give it orders.",
  ];
}

/**
 * Write the status-line sentence for a new tick, or null to keep the line.
 *
 * terminal: the new tick ended the game; then the sentence is the game-over
 * step for the replay's save state (gameOverStep(recording)), whoever played
 * the tick, so the Lock In or Run sentence ("Orders sent. The policies choose
 * next.") never outlives the game. mine: this page's own Lock In or Play One
 * Turn played the tick. watchMode: Policies Only (nobody is yours). A tick
 * this page did not ask for (Run, another tab) before the end gives null.
 * Pure.
 *
 * @param {{terminal: boolean, mine: boolean, watchMode: boolean, recording?:
 *   unknown}} input recording is the frame's recording status, or null or
 *   omitted when the match records nothing.
 * @returns {string | null}
 */
export function tickPlayedAnnouncement(input) {
  if (input.terminal) return gameOverStep(input.recording ?? null);
  if (!input.mine) return null;
  return input.watchMode
    ? "The turn played. The match is paused again."
    : "The tick played. Every step is back to Stay, every target stays picked, and Basic armed itself only where it is allowed.";
}

/**
 * Pick the one next step for Do This Now and the control it names, in the
 * vision page's words.
 *
 * Pure. `beacon` is a CSS selector of the control the Guide level outlines,
 * or null when no single control applies. It never asks for a target when
 * nothing is allowed (allowedCount 0): it says no target is allowed and
 * points to Step and Lock In (or to the Ultimate when one needs no target and
 * is allowed now). At game over it says gameOverStep(recording). Once Exit
 * was accepted (closed) it says closedSentence(recording) ("The match is
 * closed. ..."), whatever else is true, because no control on the page works
 * any more.
 *
 * @param {{
 *   closed?: boolean,
 *   stopped: boolean,
 *   terminal: boolean,
 *   phase: "plan" | "sent" | "choosing" | "act",
 *   watchMode: boolean,
 *   running: boolean,
 *   focusedManual: boolean,
 *   lastManualName: string,
 *   fallen: {className: string, returnsIn: number | null} | null,
 *   targetSet: boolean,
 *   lane: "none" | "basic" | "ultimate",
 *   ultimateOkOnTarget: boolean,
 *   ultimateNeedsNoTarget: boolean,
 *   ultimateName: string,
 *   untouchedOther: {publicAgentId: string, className: string} | null,
 *   allowedCount?: number,
 *   noTargetUltimateOk?: boolean,
 *   recording?: unknown,
 * }} input lastManualName names the champion Back returns to ("Warrior");
 *   untouchedOther is another living champion you play whose order is still
 *   the default, or null (a fallen champion is never named: it cannot take
 *   orders). allowedCount is how many Allowed Now buttons show (default: one
 *   or more, as before). noTargetUltimateOk is true when an Ultimate that
 *   needs no target is allowed now (default false). recording is the frame's
 *   recording status for the game-over and closed sentences (default
 *   none). closed is true after an accepted Exit (default false).
 * @returns {Readonly<{text: string, beacon: string | null}>}
 */
export function doThisNow(input) {
  /** @param {string} text @param {string | null} beacon */
  const step = (text, beacon) => Object.freeze({ text, beacon });
  if (input.closed === true) {
    return step(closedSentence(input.recording ?? null), null);
  }
  if (input.stopped) {
    return step(
      "A fresh start is needed. Save the game so far, or exit.",
      "#stopped-save",
    );
  }
  if (input.terminal) {
    return step(gameOverStep(input.recording ?? null), "#reset-button");
  }
  // While Run plays turn after turn, the one useful step is Pause.
  if (input.watchMode && input.running) {
    return step("Watching. Press Pause to stop after this turn.", "#live-run-button");
  }
  if (input.phase === "sent" || input.phase === "choosing") {
    return step(
      "The policies are choosing now, from what their own champions may see. They never see your orders.",
      null,
    );
  }
  if (input.phase === "act") {
    return step(
      "Everyone acts at once: blows and heals, then Charges, then every step.",
      null,
    );
  }
  if (input.watchMode) {
    return step(
      "Press Run to watch the policies play, or Play One Turn to step once.",
      "#live-run-button",
    );
  }
  if (!input.focusedManual) {
    return step(
      `You are looking at a policy's champion. Press Back To Your ${input.lastManualName} to give orders.`,
      "#wo-back",
    );
  }
  if (input.fallen) {
    const back =
      input.fallen.returnsIn === null
        ? "on the next respawn wave"
        : `in ${ticksText(input.fallen.returnsIn)}`;
    return step(
      `Your ${input.fallen.className} is fallen. It returns ${back}. Press Lock In Turn to play on.`,
      "#submit-turn-button",
    );
  }
  const nothingAllowed = input.allowedCount === 0;
  if (!input.targetSet && input.lane === "none") {
    if (nothingAllowed) {
      return input.ultimateNeedsNoTarget && input.noTargetUltimateOk === true
        ? step(
            `No target is allowed right now. Press 2 for ${input.ultimateName}, or pick a step and lock in.`,
            "#ultimate-button",
          )
        : step(
            "No target is allowed right now. Pick a step, then press Lock In Turn.",
            "#pad",
          );
    }
    return step(
      "Pick a target: hold Shift and click a body with a dashed ring, or press a button under Allowed Now.",
      "#command-target-select .pick",
    );
  }
  if (input.lane === "none" && input.targetSet) {
    if (!input.ultimateNeedsNoTarget && input.ultimateOkOnTarget) {
      return step(
        `Press 2 for ${input.ultimateName}, or lock in with No Combat.`,
        "#ultimate-button",
      );
    }
    if (nothingAllowed) {
      return step(
        "Nothing is allowed on your target right now. Pick a step, then lock in with No Combat.",
        "#pad",
      );
    }
    return step(
      "Pick a body under Allowed Now, or lock in with No Combat.",
      "#command-target-select .pick",
    );
  }
  if (input.untouchedOther) {
    return step(
      `Your ${input.untouchedOther.className} still has the default order: Stay, No Combat. Click its row below to give it orders, or lock in as it is.`,
      `.cl-row[data-slot="${input.untouchedOther.publicAgentId}"]`,
    );
  }
  return step(
    "Your order is ready. Press Lock In Turn. Then everyone acts at once.",
    "#submit-turn-button",
  );
}

/**
 * Decide, before anything is sent, whether the deck refuses an order input,
 * and with which sentence (the vision page's words).
 *
 * Pure. A refused input is never sent: the deck shows the sentence in the
 * status line (or nothing, when `text` is null) and shakes the control.
 *
 * @param {{
 *   intent: {kind: "step", moveAction: number} | {kind: "lane", lane: 0 | 1 | 2}
 *     | {kind: "target"} | {kind: "lock"},
 *   stopped: boolean,
 *   terminal: boolean,
 *   phase: "plan" | "sent" | "choosing" | "act",
 *   watchMode: boolean,
 *   focusedManual: boolean,
 *   fallen: boolean,
 *   moveAllowed: boolean,
 *   classId: unknown,
 *   ultimateNeedsNoTarget: boolean,
 *   hasTarget: boolean,
 *   basicOk: boolean,
 *   ultimateOk: boolean,
 *   cooldown: number | null,
 * }} input moveAllowed is the movement mask value of the step; basicOk and
 *   ultimateOk are the host's pair-mask values for the picked target (for an
 *   Ultimate that takes no target, ultimateOk is the value with no target);
 *   cooldown is the Ultimate cooldown in ticks, or null when unknown.
 * @returns {Readonly<{refused: boolean, text: string | null}>} refused true
 *   means do not send. text null with refused true is a silent refusal (a
 *   turn is already in flight).
 */
export function deckRefusal(input) {
  /** @param {string | null} text */
  const refuse = (text) => Object.freeze({ refused: true, text });
  const accept = Object.freeze({ refused: false, text: null });
  if (input.stopped) {
    return refuse(
      "A fresh start is needed, so orders cannot change. You can look around, save or exit.",
    );
  }
  if (input.terminal) {
    return refuse("The game is over, so orders cannot change. Press R to play again.");
  }
  if (input.phase === "sent" || input.phase === "choosing") return refuse(null);
  if (input.intent.kind === "lock") return accept;
  if (input.phase !== "plan") return refuse(null);
  if (!input.focusedManual) {
    return refuse(
      input.watchMode
        ? "Nobody is yours this game. You can look, not command."
        : "This champion is played by a policy. You can look, not command. Pick one of yours to give orders.",
    );
  }
  if (input.fallen) return refuse("A fallen champion cannot act until it returns.");
  const names = classNamesV1(input.classId);
  const basicName = names?.basicName ?? "Basic";
  const ultimateName = names?.ultimateName ?? "Ultimate";
  if (input.intent.kind === "step") {
    return input.moveAllowed
      ? accept
      : refuse(
          `Step ${MOVE_NAMES[input.intent.moveAction] ?? ""} is not allowed right now. Your order is unchanged.`,
        );
  }
  if (input.intent.kind !== "lane") return accept;
  const later =
    input.cooldown !== null && input.cooldown > 0
      ? ` It is ready in ${ticksText(input.cooldown)}.`
      : "";
  if (input.intent.lane === 1) {
    if (!input.hasTarget)
      return refuse(`${basicName} needs a target first. Your order is unchanged.`);
    if (!input.basicOk) {
      return refuse(
        `${basicName} is not allowed on this target right now. Your order is unchanged.`,
      );
    }
    return accept;
  }
  if (input.intent.lane === 2) {
    if (input.ultimateNeedsNoTarget) {
      return input.ultimateOk
        ? accept
        : refuse(
            `${ultimateName} is not allowed right now. Your order is unchanged.${later}`,
          );
    }
    if (!input.hasTarget) {
      return refuse(`${ultimateName} needs a target first. Your order is unchanged.`);
    }
    if (!input.ultimateOk) {
      return refuse(
        `${ultimateName} is not allowed on this target right now. Your order is unchanged.${later}`,
      );
    }
  }
  return accept;
}

/**
 * Write the status-line sentence for an accepted order input, in the vision
 * page's words. The deck shows it at once, before the host answers; a host
 * refusal replaces it.
 *
 * Pure.
 *
 * @param {{
 *   intent: {kind: "step", moveAction: number} | {kind: "lane", lane: 0 | 1 | 2}
 *     | {kind: "target", basicOk: boolean, ultimateOk: boolean, targetName: string}
 *     | {kind: "clear", viaKey: boolean, editable: boolean} | {kind: "lock"},
 *   classId: unknown,
 *   ultimateNeedsNoTarget: boolean,
 *   targetName: string | null,
 *   watchMode: boolean,
 * }} input targetName is the picked target "Mage (ID 5)" before the input.
 * @returns {string}
 */
export function deckAnnouncement(input) {
  const names = classNamesV1(input.classId);
  const basicName = names?.basicName ?? "Basic";
  const ultimateName = names?.ultimateName ?? "Ultimate";
  const intent = input.intent;
  if (intent.kind === "step") {
    return intent.moveAction === 0
      ? "Step: Stay."
      : `Step: ${MOVE_NAMES[intent.moveAction]}.`;
  }
  if (intent.kind === "lock") {
    return input.watchMode
      ? "Playing one turn."
      : "Orders sent. The policies choose next.";
  }
  if (intent.kind === "clear") {
    if (!intent.viaKey) return "Target cleared.";
    return intent.editable
      ? "Target cleared. Keys are off until you click the arena again."
      : "Keys are off until you click the arena again.";
  }
  if (intent.kind === "target") {
    if (intent.basicOk) {
      return `Target: ${intent.targetName}. ${basicName} armed itself, because it is allowed (Auto).`;
    }
    if (!input.ultimateNeedsNoTarget && intent.ultimateOk) {
      return `Target: ${intent.targetName}. ${basicName} is not allowed on it, so your action is No Combat. Press 2 for ${ultimateName}.`;
    }
    return `Target: ${intent.targetName}. Nothing is allowed on it right now, so your action is No Combat.`;
  }
  if (intent.lane === 0) {
    return "No Combat: your champion only moves this tick. Your target stays picked.";
  }
  if (intent.lane === 1) return `${basicName} is armed on the ${input.targetName}.`;
  if (input.ultimateNeedsNoTarget) {
    return `${ultimateName} is armed. It has no target, so your target was cleared.`;
  }
  return `${ultimateName} is armed on the ${input.targetName}.`;
}

/**
 * Return the Class Guide of one class, written from the host's real
 * class_mechanics in the vision page's words.
 *
 * Pure. Numbers come only from `mechanics`; names, calling and lore come from
 * class-documentation.js. Statuses are found by their family and the action
 * part that applies them.
 *
 * @param {unknown} classId Class ID 1 to 5.
 * @param {Record<string, any> | null} mechanics The class's class_mechanics
 *   row (`maximum_health`, `base_movement_speed`, `observation_radius`,
 *   `basic_*`, `ultimate_*`, `status_mechanics`), or null.
 * @returns {Readonly<{name: string, calling: string, lore: string, stats:
 *   string | null, basicName: string, basicText: string | null, ultimateName:
 *   string, ultimateText: string | null, ultimateEvery: number | null}> | null}
 *   Null for an unknown class. The texts and stats are null without mechanics.
 */
export function classGuideView(classId, mechanics) {
  const names = classNamesV1(classId);
  if (names === null) return null;
  const lore = classLoreV1(classId)?.fantasyLine ?? "";
  if (mechanics === null || typeof mechanics !== "object") {
    return Object.freeze({
      name: names.className,
      calling: names.calling,
      lore,
      stats: null,
      basicName: names.basicName,
      basicText: null,
      ultimateName: names.ultimateName,
      ultimateText: null,
      ultimateEvery: null,
    });
  }
  const m = mechanics;
  const statuses = Array.isArray(m.status_mechanics) ? m.status_mechanics : [];
  /** @param {string} part @param {string} family */
  const status = (part, family) =>
    statuses.find(
      (row) => row?.source_action_component === part && row?.family === family,
    ) ?? null;
  /** @param {unknown} value */
  const whole = (value) => String(Math.round(Number(value) * 100) / 100);
  /** @param {any} row */
  const slowWords = (row) =>
    Number(row?.magnitude) === 0.5 ? "half speed" : `×${whole(row?.magnitude)} speed`;
  /** @param {any} row */
  const healWords = (row) =>
    Number(row?.magnitude) === 0.5
      ? "half healing"
      : `×${whole(row?.magnitude)} healing`;
  /** @param {any} row */
  const lasts = (row) => ticksText(Number(row?.duration_steps ?? 0));
  const basicReach = reachText(m.basic_interaction_radius) ?? "";
  const ultimateReach = reachText(m.ultimate_interaction_radius) ?? "";
  const basicDamage = whole(m.basic_raw_damage);
  const ultimateDamage = whole(m.ultimate_raw_damage);
  const key = classNamesV1(classId)?.className.toLowerCase();
  /** @type {string} */
  let basicText;
  /** @type {string} */
  let ultimateText;
  if (key === "mage") {
    const burst = status("ultimate", "damage_amplification");
    basicText = `${basicDamage} damage to one visible enemy within ${basicReach}.`;
    ultimateText = `No target. It costs your action this tick, then your damage is ×${whole(burst?.magnitude)} for the next ${lasts(burst)}.`;
  } else if (key === "warrior") {
    const stun = status("ultimate", "stun");
    const slow = status("ultimate", "slow");
    basicText = `${basicDamage} damage to one enemy within ${basicReach}.`;
    ultimateText = `Rush in one straight line at a visible enemy within ${ultimateReach}: ${ultimateDamage} damage, a ${whole(stun?.duration_steps)}-tick stun and ${slowWords(slow)} for ${lasts(slow)}. No knockback. Your own step still runs after you land.`;
  } else if (key === "hunter") {
    const slow = status("basic", "slow");
    const stun = status("ultimate", "stun");
    basicText = `${basicDamage} damage to one enemy within ${basicReach}${slow ? `, and ${slowWords(slow)} for ${lasts(slow)}` : ""}.`;
    ultimateText = `Thrown at one visible enemy within ${ultimateReach}: ${ultimateDamage} damage and a ${whole(stun?.duration_steps)}-tick stun.${stun?.breaks_on_positive_damage ? " Later damage breaks it early." : ""}`;
  } else if (key === "rogue") {
    const stun = status("ultimate", "stun");
    const slow = status("ultimate", "slow");
    const antiHeal = status("ultimate", "anti_heal");
    basicText = `${basicDamage} damage to one enemy within ${basicReach}.`;
    ultimateText = `One enemy within ${ultimateReach}: ${ultimateDamage} damage, a ${whole(stun?.duration_steps)}-tick stun, ${slowWords(slow)} for ${lasts(slow)} and ${healWords(antiHeal)} for ${lasts(antiHeal)}.`;
  } else {
    const freedom = status("basic", "movement_floor");
    basicText = `Heals one ally or itself for ${whole(m.basic_raw_healing)} within ${basicReach}${freedom ? `, plus Blessing of Freedom for ${lasts(freedom)}` : ""}.`;
    ultimateText = `Heals one ally or itself for ${whole(m.ultimate_raw_healing)} within ${ultimateReach}.`;
  }
  const every = Number.isInteger(m.ultimate_cooldown_steps)
    ? Number(m.ultimate_cooldown_steps)
    : null;
  return Object.freeze({
    name: names.className,
    calling: names.calling,
    lore,
    stats: `Life ${whole(m.maximum_health)} · Speed ${Number(m.base_movement_speed).toFixed(1)} · Sight ${Number(m.observation_radius).toFixed(1)}${every === null ? "" : ` · Ultimate every ${every} ticks`}`,
    basicName: names.basicName,
    basicText,
    ultimateName: names.ultimateName,
    ultimateText,
    ultimateEvery: every,
  });
}

/**
 * Return one ability's icon markup for a Class Guide row: the vision page's
 * hidden symbol (`<use href="#ab-…">`) when the page has it, else the
 * icons.js glyph drawn as markup. Reads the document only to look the
 * symbol up and to build the glyph; changes nothing on the page.
 *
 * @param {Document} doc The page.
 * @param {string} symbolId Vision symbol id, for example "ab-lance".
 * @param {string} glyphKey icons.js fallback glyph, for example
 *   "activation-basic-damage".
 * @returns {string} Trusted SVG markup with aria-hidden="true".
 */
function abilityIconHtml(doc, symbolId, glyphKey) {
  if (doc.getElementById(symbolId)) {
    return `<svg viewBox="0 0 24 24" aria-hidden="true"><use href="#${symbolId}"/></svg>`;
  }
  const icon = createSvgIcon(doc, glyphKey, {});
  icon.setAttribute("aria-hidden", "true");
  return icon.outerHTML;
}

/**
 * Write the Class Guide card's inner HTML for one class, in the vision
 * page's words: the "Class Guide" eyebrow, the class name with its calling,
 * the lore line, the stats line, then the Basic row and the Ultimate row,
 * each with its icon, name, kind and what it does (classGuideView). One
 * owner for both clients that show it: the live rail's #class-card
 * (renderClassGuide adds its Allowed lines and its "Played by" note) and the
 * Replay Viewer's Agent tab (main.js renders the selected champion's guide).
 *
 * @param {Document} doc The page, for the ability icons (abilityIconHtml).
 * @param {unknown} classId Class ID 1 to 5.
 * @param {Record<string, any> | null} mechanics The class's class_mechanics
 *   row from the shown scene, or null (the guide then has names and lore
 *   but no numbers).
 * @param {{cooldown?: number | null, basicNow?: string, ultimateNow?: string,
 *   note?: string}} [extras] cooldown: ticks until the Ultimate is ready (0
 *   reads "Ultimate · Ready", 3 reads "Ultimate · Ready In 3 Ticks"; null or
 *   absent reads "Ultimate"). basicNow, ultimateNow: markup the caller has
 *   already escaped, added after the Basic and the Ultimate words. note:
 *   escaped markup added at the end of the card. Each defaults to "".
 * @returns {string} The card's inner HTML. An unknown class gives the short
 *   "Pick a champion to read its guide." card. Pure apart from the icon
 *   lookup.
 */
export function classGuideCardHtml(doc, classId, mechanics, extras = {}) {
  const guide = classGuideView(classId, mechanics);
  if (guide === null) {
    return '<p class="eyebrow">Class Guide</p><p class="stats">Pick a champion to read its guide.</p>';
  }
  const icons =
    /** @type {Record<string, {basic: string, ultimate: string, basicGlyph: string, ultimateGlyph: string}>} */ (
      ABILITY_ICONS
    )[guide.name.toLowerCase()] ?? null;
  const cooldown = extras.cooldown ?? null;
  const readiness =
    cooldown === null
      ? "Ultimate"
      : `Ultimate · ${cooldown === 0 ? "Ready" : `Ready In ${ticksText(cooldown, true)}`}`;
  return `<p class="eyebrow">Class Guide</p><h3>${escapeHtml(guide.name)}<span>${escapeHtml(guide.calling)}</span></h3><p class="lore">${escapeHtml(guide.lore)}</p>${
    guide.stats === null ? "" : `<p class="stats">${escapeHtml(guide.stats)}</p>`
  }<ul class="abl"><li>${icons ? abilityIconHtml(doc, icons.basic, icons.basicGlyph) : ""}<div><b>${escapeHtml(guide.basicName)}</b><em>Basic</em>${
    guide.basicText === null ? "" : `<p>${escapeHtml(guide.basicText)}</p>`
  }${extras.basicNow ?? ""}</div></li><li>${icons ? abilityIconHtml(doc, icons.ultimate, icons.ultimateGlyph) : ""}<div><b>${escapeHtml(guide.ultimateName)}</b><em>${escapeHtml(readiness)}</em>${
    guide.ultimateText === null ? "" : `<p>${escapeHtml(guide.ultimateText)}</p>`
  }${extras.ultimateNow ?? ""}</div></li></ul>${extras.note ?? ""}`;
}

/**
 * Write the checklist line of one champion you play: "Step: Stay · Action:
 * Charge → Mage 5".
 *
 * Pure.
 *
 * @param {{moveAction: unknown, lane: "none" | "basic" | "ultimate", classId:
 *   unknown, ultimateNeedsNoTarget: boolean, targetShort: string | null}} row
 *   targetShort is the target as "Mage 5", or null.
 * @returns {string}
 */
export function checklistLine(row) {
  const parts = orderWords(row);
  return `Step: ${parts.step} · Action: ${parts.action}`;
}

/**
 * Return the short parts of an order: the step's name ("Stay",
 * "North-West"), the action ("No Combat", "Arcane Blast → Warrior 6",
 * "Burst"), and the action split into its ability ("No Combat", "Arcane
 * Blast", "Burst") and its target ("Warrior 6", or null). The one owner of
 * these words for checklistLine and compactOrderParts.
 *
 * Pure.
 *
 * @param {{moveAction: unknown, lane: "none" | "basic" | "ultimate", classId:
 *   unknown, ultimateNeedsNoTarget: boolean, targetShort: string | null}} row
 *   As checklistLine.
 * @returns {{step: string, action: string, ability: string, target: string |
 *   null}}
 */
function orderWords(row) {
  const names = classNamesV1(row.classId);
  const moveId = Number.isInteger(row.moveAction) ? Number(row.moveAction) : 0;
  const [ability, target] =
    row.lane === "basic" && row.targetShort !== null
      ? [names?.basicName ?? "Basic", row.targetShort]
      : row.lane === "ultimate"
        ? row.ultimateNeedsNoTarget || row.targetShort === null
          ? [names?.ultimateName ?? "Ultimate", null]
          : [names?.ultimateName ?? "Ultimate", row.targetShort]
        : ["No Combat", null];
  return {
    step: MOVE_NAMES[moveId] ?? "Stay",
    action: target === null ? ability : `${ability} → ${target}`,
    ability,
    target,
  };
}

/**
 * Write the words for a fallen champion that takes no orders until its
 * team's respawn wave: "Fallen · Returns In 3 Ticks" (Your Champion's life
 * and the rail's checklist row), or with `short` "Fallen · Returns In 3" (a
 * short Lock In row, the Champion Card's form). "Fallen" alone when the wave
 * clock is unknown (returnsIn null).
 *
 * Pure.
 *
 * @param {{returnsIn: number | null}} fallen Ticks until the champion is back
 *   (1 means after the next Lock In), or null when unknown.
 * @param {boolean} [short] Leave out the word "Ticks". Default false.
 * @returns {string}
 */
export function fallenWords(fallen, short = false) {
  if (fallen.returnsIn === null) return "Fallen";
  return short
    ? `Fallen · Returns In ${fallen.returnsIn}`
    : `Fallen · Returns In ${ticksText(fallen.returnsIn, true)}`;
}

/**
 * Return the three parts of one short Lock In row (the rows the deck shows
 * when you play three or more champions): `main` (the action, or the step
 * and No Combat), `target` ("Warrior 6", or null) and `tail` (the step when
 * the champion also acts, or null). The deck draws each row whole on one
 * line (fitOrderRow): a row too long for the column drops the tail first,
 * then its text steps down; the ability's name and the target are never cut.
 * compactOrderLine joins them.
 *
 * - Fallen: main "Fallen · Returns In 3" (fallenWords, short; "Fallen" when
 *   the respawn wave is unknown).
 * - The default order: main "Stay · No Combat".
 * - No Combat: main "North · No Combat".
 * - An action: main "Arcane Blast", target "Warrior 6", tail "North-West"
 *   (no tail while it stays).
 *
 * Pure.
 *
 * @param {{touched: boolean, moveAction: unknown, lane: "none" | "basic" |
 *   "ultimate", classId: unknown, ultimateNeedsNoTarget: boolean, targetShort:
 *   string | null, fallen?: {className: string, returnsIn: number | null} |
 *   null}} row touched is false while the order is the default one (Stay,
 *   No Combat), as the server holds it;
 *   targetShort is the target as "Mage 5", or null; fallen is set while the
 *   champion is fallen (returnsIn in ticks, or null when unknown).
 * @returns {{main: string, target: string | null, tail: string | null}}
 */
export function compactOrderParts(row) {
  if (row.fallen) {
    return { main: fallenWords(row.fallen, true), target: null, tail: null };
  }
  if (!row.touched) return { main: "Stay · No Combat", target: null, tail: null };
  const words = orderWords(row);
  if (words.ability === "No Combat" && words.target === null) {
    return { main: `${words.step} · No Combat`, target: null, tail: null };
  }
  return {
    main: words.ability,
    target: words.target,
    tail: words.step === "Stay" ? null : words.step,
  };
}

/**
 * Write one short line of the Lock In list when you play three or more
 * champions (compactOrderParts joined): "Arcane Blast → Warrior 6",
 * "Charge → Mage 5 · North", "North · No Combat", "Stay · No Combat" for an
 * untouched order, and "Fallen · Returns In 3" ("Fallen" when the respawn
 * wave is unknown) for a fallen champion. The full sentence
 * (orderSentence) stays in the row's title; the rail's checklist shows the
 * same words with their labels (checklistLine).
 *
 * Pure.
 *
 * @param {Parameters<typeof compactOrderParts>[0]} row As compactOrderParts.
 * @returns {string}
 */
export function compactOrderLine(row) {
  const parts = compactOrderParts(row);
  return `${parts.main}${parts.target === null ? "" : ` → ${parts.target}`}${
    parts.tail === null ? "" : ` · ${parts.tail}`
  }`;
}

/**
 * Return whether an order counts as Set in the "N Of M Set" pill and the
 * rail's Set or Default tag: the server holds an order for the champion
 * that is not the default one, Stay with No Combat (owner, 3 Oct 2026). A
 * kept target with No Combat is still the default (the server sends no
 * target for it), and Stay + No Combat reads Default even when it was
 * chosen on purpose, so a freshly loaded page and both views agree.
 *
 * Pure.
 *
 * @param {{moveAction: number, lane: "none" | "basic" | "ultimate"} | null}
 *   order The server's staged order (moveAction 0 is Stay; lane is the armed
 *   action), or null when the server sent none.
 * @returns {boolean} True for Set, false for Default (and for null).
 */
export function orderIsSet(order) {
  return order !== null && (order.moveAction !== 0 || order.lane !== "none");
}

/**
 * Return the line of another team's Manual champion's rail row in Agent POV,
 * a row that shows no order (ruling L18): "Out Of Sight" when this view
 * cannot see the champion (the owner's wording, 3 Oct 2026; the row then
 * shows nothing else about it), else "In Sight · Order Hidden". Never "Not
 * Observed".
 *
 * Pure.
 *
 * @param {boolean} seen True when the audience's scene draws the champion.
 * @returns {string}
 */
export function hiddenOrderLine(seen) {
  return seen ? DECK_LABELS.orderHidden : DECK_LABELS.outOfSight;
}

/**
 * The text sizes, in CSS pixels, a short Lock In row tries after its step
 * was dropped (owner, 3 Oct 2026): half a pixel at a time from 11.5 down to
 * 10, the smallest. deck.css draws the row at 12 px first.
 */
const ORDER_ROW_SIZES = Object.freeze([11.5, 11, 10.5, 10]);

/**
 * The letter spacings, in em, a short Lock In row tries after 10 px, only
 * when it is still too long there (the longest ability on the longest
 * target, such as "Priest: Holy Word: Salvation → Hunter 2", needs about
 * 196 px in the 190 px column; -0.02em is enough for it).
 */
const ORDER_ROW_SPACINGS = Object.freeze([-0.01, -0.02, -0.03, -0.04, -0.05]);

/**
 * Fit one short Lock In row (`p.order-row`, three or more Manual champions)
 * whole on one line, so its ability's name and its target are never cut
 * (owner, 3 Oct 2026). In order, it stops at the first form that fits the
 * row's own width (scrollWidth no wider than clientWidth):
 *
 * 1. the row as deck.css draws it (12 px, with its step);
 * 2. the step (`.order-tail`) hidden: the step gives way first;
 * 3. the text at 11.5, 11, 10.5 and then 10 px (ORDER_ROW_SIZES);
 * 4. at 10 px, its letters a little closer, -0.01em to -0.05em
 *    (ORDER_ROW_SPACINGS), which keeps the space after the name;
 * 5. last, the gaps between its parts closed up too (column-gap 0.1em).
 *
 * Each call starts again from step 1, so a row that becomes shorter grows
 * back. The row's title keeps the full sentence with the step. A row that is
 * still too long after step 5 stays at the last form and deck.css clips it;
 * no row of the five classes' abilities and targets is that long in the
 * 190 px column.
 *
 * Side effects: sets or removes the row's inline font-size, column-gap and
 * letter-spacing (through the style object, which the page's security policy
 * allows) and the step's `hidden`. Reads layout (scrollWidth, clientWidth)
 * once per form tried. Works on any element with that shape.
 *
 * @param {HTMLElement} row One `.order-row` paragraph.
 * @returns {void}
 */
export function fitOrderRow(row) {
  const tail = /** @type {HTMLElement | null} */ (row.querySelector(".order-tail"));
  for (const property of ["font-size", "column-gap", "letter-spacing"]) {
    row.style.removeProperty(property);
  }
  if (tail) tail.hidden = false;
  const fits = () => row.scrollWidth <= row.clientWidth;
  if (fits()) return;
  if (tail) {
    tail.hidden = true;
    if (fits()) return;
  }
  for (const size of ORDER_ROW_SIZES) {
    row.style.setProperty("font-size", `${size}px`);
    if (fits()) return;
  }
  for (const spacing of ORDER_ROW_SPACINGS) {
    row.style.setProperty("letter-spacing", `${spacing}em`);
    if (fits()) return;
  }
  row.style.setProperty("column-gap", "0.1em");
}

/**
 * Fit the Allowed Now buttons (#command-target-select) into 2 Target, which
 * never makes the deck taller (deck.css, contain: size): the vision's pills,
 * each with its name and abilities ("Mage · ID 0 Arcane Blast"), while they
 * all show without scrolling; three or more that would scroll become the
 * two-column grid of short buttons (data-count "many": token and "Mage · ID
 * 0"; each button's title still names its abilities), which scrolls only
 * past six. One or two buttons always keep the pill form.
 *
 * Side effects: sets picks.dataset.count to the number of buttons, reads
 * layout once (scrollHeight, clientHeight) and, when three or more pills
 * overflow, sets it to "many". Works on any element with that shape.
 *
 * @param {HTMLElement} picks The Allowed Now list.
 * @returns {void}
 */
export function fitPicks(picks) {
  const count = picks.querySelectorAll(".pick").length;
  picks.dataset.count = String(count);
  if (count >= 3 && picks.scrollHeight > picks.clientHeight + 1) {
    picks.dataset.count = "many";
  }
}

/**
 * Fit every short Lock In row of a list (fitOrderRow on each `.order-row`).
 * Rows of the one- and two-champion forms are full sentences that wrap, so
 * they are left alone.
 *
 * @param {ParentNode} list The Lock In list (#order-line).
 * @returns {void}
 */
export function fitOrderRows(list) {
  for (const row of list.querySelectorAll(".order-row")) {
    fitOrderRow(/** @type {HTMLElement} */ (row));
  }
}

/**
 * Return the Manual champions whose staged orders the deck and the rail may
 * show for one installed frame (ruling L18: another team's staged order is
 * never shown in Agent POV).
 *
 * In Agent POV (authorizedPresentationAudience is "agent_pov") the result
 * keeps only the Manual champions on the viewing team: the team of the POV
 * recipient (`presentation.authority.recipient_public_agent_id`) as the
 * audience's own scene draws it. When the scene does not draw the recipient,
 * the result is empty, so nothing is shown rather than guessed. In Oracle
 * View, in the Replay Viewer and for anything else, the result is
 * `manualIds` unchanged. Team membership comes from the identity rows (the
 * static roster: class, team and ID only), never from a researcher fact
 * about orders, so switching the view and switching focus give the same list
 * as a freshly loaded page.
 *
 * Pure: reads its arguments only and never changes them.
 *
 * @param {unknown} presentation The installed presentation.
 * @param {ReadonlyArray<Readonly<Record<string, any>>>} sceneAgents The
 *   audience scene's agent records (authorizedPresentationSceneView agents).
 * @param {ReadonlyArray<{public_agent_id: string, agent: Readonly<Record<string,
 *   any>>}>} rosterRows Identity rows (authorizedPresentationIdentityRows):
 *   each row's agent carries its team_id (1 for Team A, 2 for Team B).
 * @param {ReadonlyArray<string>} manualIds Public IDs of every Manual
 *   champion of both teams, in roster order.
 * @returns {string[]} The public IDs to show, in the order of `manualIds`.
 */
export function shownManualIdsOf(presentation, sceneAgents, rosterRows, manualIds) {
  if (authorizedPresentationAudience(presentation) !== "agent_pov") {
    return [...manualIds];
  }
  // An "agent_pov" audience means an approved frame, which always has authority.
  const recipientId = /** @type {Record<string, any>} */ (presentation).authority
    ?.recipient_public_agent_id;
  const recipient = sceneAgents.find(
    (agent) => typeof recipientId === "string" && agent.public_agent_id === recipientId,
  );
  const viewTeam = recipient?.team_id;
  if (viewTeam !== 1 && viewTeam !== 2) return [];
  return manualIds.filter(
    (publicId) =>
      rosterRows.find((row) => row.public_agent_id === publicId)?.agent?.team_id ===
      viewTeam,
  );
}

/**
 * Return the staged order of each Manual champion this view may show, as the
 * server holds it, so a freshly loaded page and a page that has just switched
 * views or focus show the same orders (rule: Oracle -> POV -> Oracle equals a
 * fresh render), and orders kept from earlier ticks (the server keeps every
 * target and an armed Basic from tick to tick) show as they really are.
 *
 * Oracle View: reads the live presentation's own
 * `pending_joint_action.action_rows`, one row per Manual champion of both
 * teams. Agent POV: reads the same rows from the presentation's live
 * researcher branch (`researcher_space.pending_joint_action`), but keeps only
 * the actors listed in `shownIds` that are on the POV recipient's team
 * (team from the identity rows). Your own team's staged orders are your own
 * orders, so Agent POV shows them (main-session ruling P1, 3 Oct 2026); another
 * team's staged orders are never returned in Agent POV (ruling L18). A kept
 * target is named by class and ID only; nothing here carries a position or a
 * life.
 *
 * Each row is {actor_public_agent_id, pending_action {move_action,
 * target_action, use_ultimate_action}}, as the host sends them for Lock In.
 * The lane is "ultimate" when use_ultimate_action is 1, "basic" when
 * target_action is above 0, else "none" (the host sends 0, 0 for an unarmed
 * order). target_action is decoded the way Core's axis mapping does it
 * (core/axis_mappings.py): 0 is no target, 1 to 5 are the actor's own team in
 * slot order and 6 to 10 the other team in slot order; the slot becomes a
 * public ID through the identity rows' command_global_slot (0 to 4 Team A,
 * 5 to 9 Team B).
 *
 * Returns an empty map for the Replay Viewer, an unmarked value, a missing
 * field, Agent POV without a researcher branch or without `shownIds`, and an
 * Agent POV recipient the identity rows do not list; a malformed row is
 * skipped. Pure: reads its arguments only.
 *
 * @param {unknown} presentation The installed presentation.
 * @param {ReadonlyArray<{public_agent_id: string, command_global_slot: number
 *   | null, agent?: Readonly<Record<string, any>>}>} rosterRows Identity rows
 *   (authorizedPresentationIdentityRows); each row's agent carries its team_id
 *   (1 for Team A, 2 for Team B).
 * @param {ReadonlyArray<string> | null} [shownIds] Agent POV only: the public
 *   IDs whose orders this view may show (shownManualIdsOf: the viewing team's
 *   Manual champions). Default null, which in Agent POV returns nothing.
 *   Oracle View ignores it.
 * @returns {Map<string, {moveAction: number, lane: "none" | "basic" |
 *   "ultimate", targetPublicId: string | null}>} Keyed by the actor's public
 *   ID.
 */
export function servedManualOrdersOf(presentation, rosterRows, shownIds = null) {
  /** @type {Map<string, {moveAction: number, lane: "none" | "basic" | "ultimate", targetPublicId: string | null}>} */
  const orders = new Map();
  const audience = authorizedPresentationAudience(presentation);
  if (audience === null) return orders;
  const value = /** @type {Record<string, any>} */ (presentation);
  if (value.viewer_mode !== "live") return orders;
  /** @type {unknown} */
  let rows = null;
  /** @type {Set<string> | null} */
  let allowed = null;
  if (audience === "researcher") {
    rows = value.pending_joint_action?.action_rows;
  } else {
    if (!authorizedPresentationHasResearcherSpace(presentation) || !shownIds) {
      return orders;
    }
    const recipientId = value.authority?.recipient_public_agent_id;
    const teamOf = (/** @type {unknown} */ publicId) =>
      rosterRows.find((row) => row.public_agent_id === publicId)?.agent?.team_id;
    const viewTeam = teamOf(recipientId);
    if (viewTeam !== 1 && viewTeam !== 2) return orders;
    allowed = new Set(shownIds.filter((publicId) => teamOf(publicId) === viewTeam));
    rows = value.researcher_space.pending_joint_action?.action_rows;
  }
  if (!Array.isArray(rows)) return orders;
  /** @type {Map<string, number>} */
  const slotOf = new Map();
  /** @type {Map<number, string>} */
  const idAt = new Map();
  for (const row of rosterRows) {
    if (!Number.isInteger(row.command_global_slot)) continue;
    const slot = Number(row.command_global_slot);
    slotOf.set(row.public_agent_id, slot);
    idAt.set(slot, row.public_agent_id);
  }
  for (const row of rows) {
    const publicId = row?.actor_public_agent_id;
    const action = row?.pending_action;
    const slot = typeof publicId === "string" ? slotOf.get(publicId) : undefined;
    if (
      typeof publicId !== "string" ||
      (allowed !== null && !allowed.has(publicId)) ||
      slot === undefined ||
      action === null ||
      typeof action !== "object" ||
      ![action.move_action, action.target_action, action.use_ultimate_action].every(
        Number.isInteger,
      )
    ) {
      continue;
    }
    const targetAction = Number(action.target_action);
    const ownBase = slot < 5 ? 0 : 5;
    const otherBase = 5 - ownBase;
    const targetSlot =
      targetAction >= 1 && targetAction <= 5
        ? ownBase + targetAction - 1
        : targetAction >= 6 && targetAction <= 10
          ? otherBase + targetAction - 6
          : null;
    orders.set(publicId, {
      moveAction: Number(action.move_action),
      lane:
        action.use_ultimate_action === 1
          ? "ultimate"
          : targetAction > 0
            ? "basic"
            : "none",
      targetPublicId: targetSlot === null ? null : (idAt.get(targetSlot) ?? null),
    });
  }
  return orders;
}

/**
 * Draw a team crest as SVG markup, as on the vision page: a ring in the team
 * color with a faint center (Team B adds a thin ink ring).
 *
 * Pure. Export-safe SVG (flat fills and strokes). The team color is painted
 * through the .tc-a and .tc-b classes (see paintAttributes), so it shows only
 * on a page that loads src/styles/battleclient/base.css.
 *
 * @param {"A" | "B"} team
 * @param {14 | 16} size Pixel size: 14 beside an Agent ID, 16 in headings.
 * @returns {string}
 */
export function crestSvg(team, size) {
  const half = size / 2;
  const ring = size === 16 ? 6 : 5;
  const dot = size === 16 ? 2.7 : 2.25;
  const ink =
    team === "B"
      ? `<circle cx="${half.toFixed(1)}" cy="${half.toFixed(1)}" r="${ring}" fill="none" stroke="${INK}" stroke-width="1"/>`
      : "";
  return `<svg class="crest" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" aria-hidden="true"><circle cx="${half.toFixed(1)}" cy="${half.toFixed(1)}" r="${ring}" fill="none" ${paintAttributes("stroke", team)} stroke-width="3"/>${ink}<circle cx="${half.toFixed(1)}" cy="${half.toFixed(1)}" r="${dot}" ${paintAttributes("fill", team)} opacity="0.35"/></svg>`;
}

/**
 * Return a point on a circle, clockwise from north, as the vision page does.
 *
 * @param {number} center Center x and y.
 * @param {number} radius Radius.
 * @param {number} degrees Angle clockwise from north.
 * @returns {[number, number]}
 */
function ringPoint(center, radius, degrees) {
  const angle = (degrees * Math.PI) / 180;
  return [center + radius * Math.sin(angle), center - radius * Math.cos(angle)];
}

/**
 * Draw a champion token as SVG markup, as on the vision page: the team disc
 * (Team B with its flat top and ivory bar), the class glyph, the twelve
 * Ultimate ticks (size "M" only) and the team ring.
 *
 * Pure. Export-safe SVG. The caller supplies the glyph markup, so this
 * function needs no page to build it. The team ring and the class colors are
 * painted through the .tc-… classes (see paintAttributes), so they show only
 * on a page that loads src/styles/battleclient/base.css.
 *
 * @param {{classKey: string, team: "A" | "B", size: "M" | "XS", lit?: number,
 *   ready?: boolean, faded?: boolean, glyph: string}} token lit is the number
 *   of lit Ultimate ticks (0 to 12, default 0); ready adds the four ready
 *   diamonds; faded draws the token at 45% (a fallen champion); glyph is the
 *   class glyph markup placed in the glyph box (see tokenGlyphBox).
 * @returns {string}
 */
export function tokenSvg(token) {
  const medium = token.size === "M";
  const box = medium ? 46 : 20;
  const c = box / 2;
  const outer = medium ? 20.25 : 8.1;
  const inner = medium ? 18.5 : 7.2;
  /** @param {"fill" | "stroke"} attribute */
  const classPaint = (attribute) => paintAttributes(attribute, token.classKey);
  const disc = token.team === "A" ? "#11264A" : "#3B141B";
  const parts = [
    `<circle cx="${c.toFixed(2)}" cy="${c.toFixed(2)}" r="${outer.toFixed(2)}" fill="${disc}"/>`,
    `<circle cx="${c.toFixed(2)}" cy="${c.toFixed(2)}" r="${inner.toFixed(2)}" fill="#0D1119"/>`,
  ];
  if (token.team === "A") {
    parts.push(
      `<circle cx="${c.toFixed(2)}" cy="${c.toFixed(2)}" r="${inner.toFixed(2)}" fill="${disc}"/>`,
    );
  } else {
    const top = medium ? 5.89 : 3.34;
    const left = medium ? 15.97 : 7.26;
    const right = medium ? 30.03 : 12.74;
    parts.push(
      `<path d="M${left.toFixed(2)} ${top.toFixed(2)} L${right.toFixed(2)} ${top.toFixed(2)} A${inner.toFixed(2)} ${inner.toFixed(2)} 0 1 1 ${left.toFixed(2)} ${top.toFixed(2)} Z" fill="${disc}"/>`,
      `<path d="M${(left + 1).toFixed(2)} ${top.toFixed(2)} H${(right - 1).toFixed(2)}" stroke="#F1E7DA" stroke-width="${medium ? 1.25 : 1}" stroke-linecap="round" opacity="0.9"/>`,
    );
  }
  if (medium) {
    parts.push(
      `<circle cx="23.00" cy="23.00" r="7.48" ${classPaint("fill")} opacity="0.14"/>`,
    );
  }
  parts.push(token.glyph);
  if (medium) {
    const lit = Math.max(0, Math.min(12, Number(token.lit ?? 0)));
    /** @param {ReadonlyArray<number>} angles */
    const ticks = (angles) =>
      angles
        .map((degrees) => {
          const [x1, y1] = ringPoint(23, 13.57, degrees);
          const [x2, y2] = ringPoint(23, 15.55, degrees);
          return `M${x1.toFixed(2)} ${y1.toFixed(2)} L${x2.toFixed(2)} ${y2.toFixed(2)}`;
        })
        .join(" ");
    const litAngles = RUNE_ORDER.slice(0, lit);
    const dimAngles = RUNE_ORDER.slice(lit);
    if (dimAngles.length > 0) {
      parts.push(
        `<path d="${ticks(dimAngles)}" ${classPaint("stroke")} stroke-width="1.21" opacity="0.22"/>`,
      );
    }
    if (litAngles.length > 0) {
      if (token.ready) {
        parts.push(
          `<path d="${ticks(litAngles)}" ${classPaint("stroke")} stroke-width="3.63" opacity="0.18"/>`,
        );
      }
      parts.push(
        `<path d="${ticks(litAngles)}" ${classPaint("stroke")} stroke-width="1.21" opacity="${token.ready ? 1 : 0.9}"/>`,
      );
    }
    if (token.ready) {
      const diamonds = [0, 90, 180, 270]
        .map((degrees) => {
          const [x, y] = ringPoint(23, 14.56, degrees);
          const across = degrees % 180 === 0;
          const w = across ? 1.1 : 1.29;
          const h = across ? 1.29 : 1.1;
          return `M${x.toFixed(2)} ${(y - h).toFixed(2)} L${(x + w).toFixed(2)} ${y.toFixed(2)} L${x.toFixed(2)} ${(y + h).toFixed(2)} L${(x - w).toFixed(2)} ${y.toFixed(2)} Z`;
        })
        .join(" ");
      parts.push(
        `<path d="${diamonds}" ${classPaint("fill")} stroke="${INK}" stroke-width="1" paint-order="stroke"/>`,
      );
    }
  }
  parts.push(
    `<circle cx="${c.toFixed(2)}" cy="${c.toFixed(2)}" r="${outer.toFixed(2)}" fill="none" ${paintAttributes("stroke", token.team)} stroke-width="${medium ? "3.50" : "1.80"}"/>`,
  );
  if (token.team === "B" && medium) {
    parts.push(
      `<circle cx="23.00" cy="23.00" r="20.25" fill="none" stroke="${INK}" stroke-width="1.00"/>`,
    );
  }
  return `<svg class="tok" width="${box}" height="${box}" viewBox="0 0 ${box} ${box}" aria-hidden="true"><g opacity="${token.faded ? 0.45 : 1}">${parts.join("")}</g></svg>`;
}

/**
 * The glyph box of a token size: where the class glyph sits, in the token's
 * own units.
 *
 * @param {"M" | "XS"} size
 * @returns {{at: number, width: number}}
 */
export function tokenGlyphBox(size) {
  return size === "M" ? { at: 13.32, width: 19.36 } : { at: 5.05, width: 9.9 };
}

/**
 * The title of another team's Manual champion's rail row in Agent POV: why it
 * shows no order (ruling L18) and what a click does.
 */
const HIDDEN_ORDER_TITLE =
  "Agent POV shows one team's orders at a time. Click to give this champion orders.";

/** The vision's out-of-sight token: an empty disc with a dashed ring. */
const UNSEEN_TOKEN = `<svg class="tok" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8" fill="#141A26"/><circle cx="10" cy="10" r="7.75" fill="none" stroke="${MOON}" stroke-width="1.5" stroke-dasharray="2.2 2.6" opacity=".45"/></svg>`;

/**
 * Page elements the deck paints. main.js finds them once at startup by ID; the
 * deck never looks them up again. The vision parts (#ag-name, #tchip, #msg,
 * #turn-card and the rest) are found once by bindCommandDeck.
 *
 * @typedef {object} TacticsTurnElements
 * @property {Element | null} commandDeck The `#command-deck` disclosure, or null
 *   when the page has no deck. Every deck button lives inside it.
 * @property {any} commandControlledActor `#command-controlled-actor`: a hidden
 *   legacy line that still names the champion being ordered ("Mage · ID 5").
 * @property {any} commandTargetSelect `#command-target-select`: the Allowed Now
 *   group of `.pick` buttons.
 * @property {any} policyTargetMasks `#policy-target-masks`: hidden legacy part.
 * @property {any} policyTargetMaskRows `#policy-target-mask-rows`: its list.
 * @property {any} noCombatButton `#no-combat-button`.
 * @property {any} basicButton `#basic-button`.
 * @property {any} ultimateButton `#ultimate-button`.
 * @property {any} submitTurnButton `#submit-turn-button`: "Lock In Turn".
 * @property {any} commandCommitTitle `#command-commit-title` (hidden legacy).
 * @property {any} commandCommitSummary `#command-commit-summary` (hidden legacy).
 * @property {any} liveRunButton `#live-run-button`: Run or Pause.
 * @property {any} liveRunStatus `#live-run-status` (hidden legacy).
 * @property {any} liveControlHelp `#live-control-help` (hidden legacy).
 * @property {any} liveMatchRules `#live-match-rules` (hidden legacy).
 * @property {any} resetButton `#reset-button`: "Restart R" in the status line.
 */

/**
 * Return whether live play is running: the page's own Run is on, or the
 * server reports run_state "running". Pure.
 *
 * @param {{pageRun?: boolean, frame: Record<string, any> | null}} state Page state.
 * @returns {boolean}
 */
export function liveRunning(state) {
  return state.pageRun === true || state.frame?.run_state === "running";
}

/**
 * The live page state that main.js owns. The deck reads these fields at call
 * time and never assigns them.
 *
 * @typedef {object} TacticsTurnState
 * @property {Record<string, any> | null} frame Installed transport frame, or
 *   null before the first frame.
 * @property {Readonly<Record<string, any>> | null} presentation Installed
 *   authorized presentation, or null while authority is pending.
 * @property {boolean} busy True while one page request is in flight.
 * @property {boolean} offline True after a connection failure.
 * @property {boolean} resyncRequired True until Reconnect installs fresh authority.
 * @property {boolean} shuttingDown True after Exit was accepted.
 * @property {Readonly<Record<string, any>> | null} [exitRecording] The
 *   recording status that came with the accepted Exit answer, or null (no
 *   Exit yet, or the match records nothing). It is newer than the installed
 *   frame's when the answer's frame could not be installed. Default null.
 * @property {boolean} [pageRun] True while the page's own Run plays one turn
 *   after another in Policies Only (main.js toggleLiveRun); the server's
 *   run_state then stays "paused". Default false.
 */

/**
 * Everything createTacticsTurn needs from main.js. Each function keeps the
 * contract documented beside its definition in main.js.
 *
 * @typedef {object} TacticsTurnContext
 * @property {TacticsTurnElements} elements Deck elements (see above).
 * @property {TacticsTurnState} state Live page state, read only.
 * @property {() => boolean} isReplayMode True when the launcher product is the
 *   Replay Viewer.
 * @property {(frame: unknown) => boolean} isTerminal True when a transport frame
 *   is at its terminal or final-frame boundary.
 * @property {(value: unknown) => value is Record<string, any>} isRecord Shape
 *   guard for a plain object.
 * @property {(value: unknown) => any[]} asArray The array itself, or a new empty
 *   array for anything else.
 * @property {(publicAgentId: unknown) => string} [agentIdentity] Not read
 *   any more: the deck names champions "Mage · ID 5" from agent-identity.js.
 *   Optional, so main.js may keep passing it.
 * @property {() => boolean} installedAuthorityIsCoherent True when the installed
 *   transport and presentation form one trusted pair.
 * @property {() => boolean} liveScriptedInspectionOnly True when the installed
 *   live presentation is scripted, inspection-only playback.
 * @property {() => boolean} recordingScientificControlsFenced True when the
 *   recording has left its recording lifecycle and game controls are fenced.
 * @property {(command: Record<string, unknown>, frame: Record<string, any> | null)
 *   => {allowed: boolean, notice: string | null}} modeAvailability The browser's
 *   advisory gate for one command at the installed frame.
 * @property {(presentation: unknown, presentationKey: unknown)
 *   => Readonly<Record<string, any>> | null} authorizedResearcherAgentForPresentationKey
 *   Find an agent by opaque key in the researcher list, or null.
 * @property {(element: Element) => void} clearPresentationTooltipOwner Remove a
 *   control's presentation tooltip and its accessible description.
 * @property {(command: Record<string, unknown>) => Promise<void>} dispatchCommand
 *   Send one live command through every page gate.
 * @property {(command: Record<string, unknown>,
 *   options?: {restoreFocusTo?: HTMLElement | null}) => void} dispatchCommandFromDraftControl
 *   Give the battlefield keyboard focus, then send one live command.
 * @property {() => void} scheduleLivePoll Schedule the next conditional live read.
 * @property {() => void} [toggleLiveRun] Start or stop the page's own Run in
 *   Policies Only (main.js). When absent, Run and Pause send the server's
 *   `run` and `pause` commands instead.
 * @property {(message: string, level?: string) => void} setNotice Store a notice
 *   for the next connection repaint.
 * @property {() => void} renderConnection Repaint the connection pill and notice.
 * @property {() => Readonly<Record<string, boolean>> | null} [visualFilters]
 *   The page's visual filter state, read each time the deck draws a champion
 *   token (the V5 token follows the token switches, as on the arena and the
 *   scorebug). Optional; when absent the tokens use the switch defaults.
 * @property {() => boolean} [classGuideEnabled] True while the Class Guide
 *   switch (class_guide) is on. Optional; when absent the Class Guide card always
 *   shows. When it returns false the card is hidden and not filled.
 */

/**
 * Messages main.js writes for routine work (a request waiting for its answer,
 * a load). They never replace a sentence the deck wrote.
 */
const ROUTINE_MESSAGES = new Set([
  "",
  "Waiting for the game to answer…",
  "Loading the current match…",
]);

/**
 * Create the live deck renderers for one page.
 *
 * Call once at page startup, after the page state exists and before anything
 * can repaint (the choreographer and the replay controller repaint through
 * renderCommandAvailability). The returned functions read `context.state` each
 * time they run, so they always paint the current frame. Creating the deck has
 * no side effects; only the returned functions touch the DOM.
 *
 * @param {TacticsTurnContext} context Page elements, state and helpers from
 *   main.js. Functions are captured once; pass stable functions or wrappers.
 * @returns {Readonly<{
 *   policyControlledActor: () => Readonly<{label: string}> | null,
 *   policyControllerBlocksActionEdit: (command: Record<string, unknown>) => boolean,
 *   renderCommandAvailability: () => void,
 *   renderSystemOperation: () => void,
 *   bindCommandDeck: () => void,
 *   showStatusMessage: (message: string, level?: string) => void,
 * }>} The deck API. See each function's own documentation.
 */
export function createTacticsTurn(context) {
  const {
    elements,
    state,
    isReplayMode,
    isTerminal,
    isRecord,
    asArray,
    installedAuthorityIsCoherent,
    liveScriptedInspectionOnly,
    recordingScientificControlsFenced,
    modeAvailability,
    authorizedResearcherAgentForPresentationKey,
    clearPresentationTooltipOwner,
    dispatchCommand,
    dispatchCommandFromDraftControl,
    scheduleLivePoll,
  } = context;
  const classGuideEnabled =
    typeof context.classGuideEnabled === "function"
      ? context.classGuideEnabled
      : () => true;
  const visualFilters =
    typeof context.visualFilters === "function" ? context.visualFilters : () => null;

  /**
   * Read the selected actor's actual slot owner from the paired accepted frame.
   * Return null for Manual or unavailable ownership. A System team may contain
   * Manual slots; its whole-team runtime name does not grant or remove control.
   *
   * Reads the audience's inspection of the installed presentation and the
   * transport's combat configuration. Agent POV: the controller label comes
   * from the match setup the player chose, not from sight; the inspected agent
   * itself is whichever agent the host says is in view now.
   *
   * @returns {Readonly<{label: string}> | null} The policy's display label (for
   *   example "Reactive TDM ALPHA"), or null for a Manual slot, no selection or
   *   no installed authority.
   */
  function policyControlledActor() {
    const inspection = authorizedPresentationResearcherInspectionState(
      state.presentation,
    ).inspection;
    if (!isRecord(inspection)) return null;
    const controller = authorizedPresentationController(
      state.presentation,
      state.frame?.combat_configuration,
      inspection.actor_public_agent_id,
    );
    if (controller === null || controller.manual) return null;
    return Object.freeze({ label: controller.label });
  }

  /**
   * Return whether a command would edit an automatically controlled actor action.
   * Block battlefield pointers, target selection, draft keys, and Escape while such
   * an actor is selected. Other commands keep their normal availability checks.
   *
   * Side effect: when it returns true it remembers the vision's refusal
   * sentence, so the warning main.js shows next in the status line reads "This
   * champion is played by a policy. You can look, not command. Pick one of
   * yours to give orders." (or, with nobody Manual, "Nobody is yours this
   * game. You can look, not command.").
   *
   * @param {Record<string, unknown>} command A browser command record.
   * @returns {boolean} True when a policy controls the selected agent and the
   *   command would change its staged action; the page then refuses it locally.
   */
  function policyControllerBlocksActionEdit(command) {
    if (policyControlledActor() === null) {
      return false;
    }
    let blocks = false;
    if (command.command_type === "battlefield_pointer") {
      // A plain click looks at a body; only Shift + click (a target pick) or a
      // right click (clear target) would edit the policy's order. A pointer
      // command without its Shift flag is treated as an edit.
      blocks = command.shift_key !== false || command.button === "secondary";
    } else if (
      command.command_type === "roster_selection" &&
      command.role === "target"
    ) {
      blocks = true;
    } else if (command.command_type === "keyboard" && typeof command.key === "string") {
      const key = command.key.toLowerCase();
      blocks = DRAFT_KEYS.has(key) || key === "escape";
    }
    if (blocks) {
      const manual = state.frame?.combat_configuration?.manual_slots;
      pendingRefusal =
        Array.isArray(manual) && manual.length === 0
          ? "Nobody is yours this game. You can look, not command."
          : "This champion is played by a policy. You can look, not command. Pick one of yours to give orders.";
    }
    return blocks;
  }

  /**
   * Mark a draft button selected or unselected in aria-pressed and its dataset.
   * This changes presentation state only, not the pending simulator action.
   *
   * @param {HTMLButtonElement} button A deck button.
   * @param {boolean} selected Whether the staged draft uses this choice.
   * @returns {void}
   */
  function setDraftSelection(button, selected) {
    button.setAttribute("aria-pressed", String(selected));
    button.dataset.selected = String(selected);
  }

  /**
   * Paint an action choice availability while keeping its explanation focusable.
   * Set aria-disabled and attach the supplied descriptor, or build a legality tooltip
   * from explanation when descriptor is null (the default). Python still checks
   * the exact action mask before accepting an edit. The button stays focusable
   * (it is not `disabled` here), so keyboard users can read why, and a click on
   * it shows the deck's refusal sentence instead of sending.
   *
   * @param {HTMLButtonElement} button A deck button.
   * @param {boolean} available Whether the installed mask allows this choice now.
   * @param {string} explanation Plain sentence for the tooltip.
   * @param {unknown} [descriptor] A ready tooltip descriptor; null (the default)
   *   builds a simple legality descriptor from `explanation`.
   * @returns {void}
   */
  function setAuthoritativeAvailability(
    button,
    available,
    explanation,
    descriptor = null,
  ) {
    button.setAttribute("aria-disabled", String(!available));
    button.dataset.authoritativeAvailable = String(available);
    button.dataset.tooltipKind = "legality";
    button.dataset.tooltipText = explanation;
    registerTooltipOwner(
      button,
      descriptor ?? {
        kind: "legality",
        id: `command:${button.id || button.dataset.key || "choice"}`,
        title:
          button.getAttribute("aria-label") ??
          button.textContent?.trim() ??
          "Pending choice",
        tone: available ? "positive" : "warning",
        accent: "none",
        summary: explanation,
        rows: [
          {
            label: "Status",
            value: available ? "True" : "False",
            metadata: { compact: true, full: true },
          },
        ],
        sections: [],
        metadata: { compact: true, full: true },
        anchor: "element",
      },
    );
  }

  /**
   * Mark a command button unavailable and remove its old scientific tooltip.
   * Use this when no exact authorized owner exists; generic help must not preserve
   * or invent an owner-specific legality claim.
   *
   * @param {HTMLButtonElement} button A deck button.
   * @returns {void}
   */
  function clearOwnerBoundLegalityAvailability(button) {
    button.setAttribute("aria-disabled", "true");
    button.dataset.authoritativeAvailable = "false";
    clearPresentationTooltipOwner(button);
  }

  /**
   * Paint the pad and the three action buttons' availability and legality
   * tooltips for the champion being ordered.
   *
   * Command plumbing only: it reads the researcher inspection (the global
   * draft and masks of the champion you order, so the owner check can use the
   * researcher agent list) and never writes text the deck shows; the deck's
   * words and tags come from renderTacticsTurn, which also takes
   * aria-disabled off the three action slots again (a Not Allowed slot stays
   * pressable and refuses with a sentence). Agent POV: the tooltips name only
   * the champion you order and whether its lanes are allowed.
   *
   * @param {Readonly<Record<string, any>>} presentation The installed presentation.
   * @returns {void}
   */
  function renderDraftState(presentation) {
    const inspectionState =
      authorizedPresentationResearcherInspectionState(presentation);
    const inspection = isRecord(inspectionState.inspection)
      ? inspectionState.inspection
      : {};
    const mask = isRecord(inspection.decision_mask) ? inspection.decision_mask : {};
    const pending = isRecord(inspection.draft_action) ? inspection.draft_action : {};
    const policyReadOnly = inspectionState.state_kind === "live_policy";
    const controlledCandidate = authorizedResearcherAgentForPresentationKey(
      presentation,
      inspection.actor_presentation_key,
    );
    const controlledOwner =
      controlledCandidate !== null &&
      controlledCandidate.public_agent_id === inspection.actor_public_agent_id
        ? controlledCandidate
        : null;
    const controlledIdentity =
      typeof controlledOwner?.public_agent_id === "string"
        ? canonicalAgentIdentity(controlledOwner).title
        : null;
    elements.commandControlledActor.textContent =
      controlledIdentity ?? "No Champion Selected";
    const controlledSlot =
      controlledOwner === null
        ? null
        : authorizedOracleCommandSlotForPresentationKey(
            presentation,
            controlledOwner.presentation_key,
          );
    if (
      authorizedPresentationAudience(presentation) === "researcher" &&
      Number.isInteger(controlledSlot)
    ) {
      elements.commandControlledActor.dataset.controlledSlot = String(controlledSlot);
    } else {
      delete elements.commandControlledActor.dataset.controlledSlot;
    }
    const pendingMove = Number(pending.move_action);
    const movementMask = asArray(mask.movement_action_mask);
    const movementButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
      elements.commandDeck?.querySelectorAll("button[data-move-action]") ?? []
    );
    for (const button of movementButtons) {
      const moveAction = Number(button.dataset.moveAction);
      const available = movementMask[moveAction];
      setDraftSelection(button, moveAction === pendingMove);
      setAuthoritativeAvailability(
        button,
        available === true,
        available === true
          ? `${button.getAttribute("aria-label") ?? "Movement"} is allowed now.`
          : `${button.getAttribute("aria-label") ?? "Movement"} is not allowed now.`,
      );
    }
    const targetAction = Number.isInteger(pending.target_action)
      ? Number(pending.target_action)
      : 0;
    const pairMask = asArray(mask.target_use_ultimate_joint_mask)[targetAction];
    const pairs = asArray(mask.target_use_ultimate_joint_mask);
    const basicAvailable = policyReadOnly
      ? pairs.some((pair, index) => index > 0 && asArray(pair)[0] === true)
      : targetAction > 0 && asArray(pairMask)[0] === true;
    const ultimateAvailable = policyReadOnly
      ? pairs.some((pair) => asArray(pair)[1] === true)
      : asArray(pairMask)[1] === true;
    setAuthoritativeAvailability(
      elements.noCombatButton,
      true,
      policyReadOnly
        ? "The policy may choose No Combat. It has not chosen its action yet."
        : "No Combat is always allowed. Your step still happens when you lock in.",
    );
    const legality =
      controlledOwner === null
        ? null
        : {
            owner_presentation_key: controlledOwner.presentation_key,
            owner_public_agent_id: controlledOwner.public_agent_id,
            lane_0_available: basicAvailable,
            lane_1_available: ultimateAvailable,
            basic_available: basicAvailable,
            ultimate_available: ultimateAvailable,
          };
    const basicLegality =
      legality === null ? null : explainLegality(legality, 0, controlledOwner);
    const ultimateLegality =
      legality === null ? null : explainLegality(legality, 1, controlledOwner);
    if (basicLegality === null) {
      clearOwnerBoundLegalityAvailability(elements.basicButton);
    } else {
      setAuthoritativeAvailability(
        elements.basicButton,
        basicAvailable,
        `The Basic is ${basicAvailable ? "" : "not "}allowed this tick.`,
        basicLegality,
      );
    }
    if (ultimateLegality === null) {
      clearOwnerBoundLegalityAvailability(elements.ultimateButton);
    } else {
      setAuthoritativeAvailability(
        elements.ultimateButton,
        ultimateAvailable,
        `The Ultimate is ${ultimateAvailable ? "" : "not "}allowed this tick.`,
        ultimateLegality,
      );
    }
  }

  /**
   * Apply current live/replay, authority, lifecycle, and controller gates to commands.
   * Paint a valid editable draft when available, otherwise clear its controls.
   * Automatic-controller action edits stay disabled while allowed submission and
   * inspection commands keep their own gates.
   *
   * Runs on every page repaint and on every choreography state change. Lock In
   * Turn sends Enter; while a turn is in flight its label reads Orders Sent,
   * Policies Choosing… or Everyone Acts and it is disabled. In scripted
   * playback the button becomes "Advance scripted frame" and sends N, and the
   * Watching bar's Play One Turn (the deck shown when nobody is Manual) also
   * sends N, stays enabled until the script ends and shows N as its key, so
   * the control on screen always advances the next scripted frame. In the
   * Replay Viewer every deck control is disabled. Ends by calling
   * renderTacticsTurn for the champion, target, action slots, overlays,
   * status line and rail cards. Side effects: DOM text, `disabled`, `aria-*`
   * and `data-*` attributes and tooltips on the deck and the rail cards; no
   * request is sent.
   *
   * @returns {void}
   */
  function renderCommandAvailability() {
    trackChanges(deckData());
    const presentation = state.presentation;
    const inspectionState =
      authorizedPresentationResearcherInspectionState(presentation);
    const editableDraft = inspectionState.state_kind === "live_editable";
    const policyInspection = inspectionState.state_kind === "live_policy";
    const scriptedAdvance = liveScriptedInspectionOnly();
    const disabled =
      state.busy ||
      !state.frame ||
      !isAuthorizedPresentationFrame(presentation) ||
      !installedAuthorityIsCoherent() ||
      state.shuttingDown ||
      state.resyncRequired ||
      state.offline;
    const scientificFenced = recordingScientificControlsFenced();
    elements.submitTurnButton.dataset.key = scriptedAdvance ? "n" : "Enter";
    const watchSubmit = deckParts?.watchSubmit ?? null;
    if (watchSubmit) {
      // Space and Enter are refused in scripted playback; N is its one tick.
      watchSubmit.dataset.key = scriptedAdvance ? "n" : "Enter";
      const watchKey = watchSubmit.querySelector("kbd");
      if (watchKey) watchKey.textContent = scriptedAdvance ? "N" : "Space";
    }
    if (scriptedAdvance) {
      elements.submitTurnButton.textContent = "Advance scripted frame";
      elements.submitTurnButton.removeAttribute("aria-label");
    } else {
      // The vision's two lines: <b id="lock-label"> (the phase label) and its
      // keys. Rebuilt only when scripted playback replaced them.
      if (!elements.submitTurnButton.querySelector("#lock-label")) {
        const label = document.createElement("b");
        label.id = "lock-label";
        label.textContent = DECK_LABELS.lockInButton;
        const keys = document.createElement("span");
        keys.textContent = DECK_LABELS.lockInKeys;
        elements.submitTurnButton.replaceChildren(label, keys);
        if (deckParts) deckParts.lockLabel = label;
      }
      elements.submitTurnButton.setAttribute("aria-label", DECK_LABELS.lockInButton);
    }
    elements.commandCommitTitle.textContent = scriptedAdvance
      ? "Advance the registered script"
      : stagedOrderSentence(presentation);
    elements.commandCommitSummary.textContent = scriptedAdvance
      ? "One authoritative scripted transition"
      : DECK_LABELS.lockInHelp;
    if (isReplayMode()) {
      if (elements.commandDeck) {
        for (const button of elements.commandDeck.querySelectorAll("button")) {
          /** @type {HTMLButtonElement} */ (button).disabled = true;
        }
      }
      renderTacticsTurn();
      return;
    }
    elements.policyTargetMasks.hidden = true;
    if ((editableDraft || policyInspection) && presentation) {
      renderDraftState(presentation);
    } else {
      for (const button of elements.commandDeck?.querySelectorAll(
        "button[data-move-action], #no-combat-button, #basic-button, #ultimate-button",
      ) ?? []) {
        const actionButton = /** @type {HTMLButtonElement} */ (button);
        setDraftSelection(actionButton, false);
        clearOwnerBoundLegalityAvailability(actionButton);
      }
      elements.commandControlledActor.textContent = "No Champion Selected";
      delete elements.commandControlledActor.dataset.controlledSlot;
    }
    const policyControlled = policyControlledActor();
    const policyControllerReadOnly = policyControlled !== null;
    elements.commandDeck?.setAttribute(
      "data-policy-controller-read-only",
      String(policyControllerReadOnly),
    );
    picksDisabled =
      disabled ||
      !editableDraft ||
      Boolean(state.frame?.system_operation) ||
      Boolean(state.frame?.fresh_launch_required) ||
      scientificFenced ||
      isTerminal(state.frame) ||
      policyControllerReadOnly;
    const turnInFlight = currentPhase() !== "plan";
    if (elements.commandDeck) {
      const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
        elements.commandDeck.querySelectorAll("button[data-key]")
      );
      for (const button of buttons) {
        const command = keyboardCommand(button.dataset.key ?? "", {
          shiftKey: button.dataset.shift === "true",
        });
        const mode = modeAvailability(command, state.frame);
        const recordingDecision = recordingCommandDecision(state.frame ?? {}, command);
        const submits =
          button === elements.submitTurnButton || button.id === "watch-submit";
        const enabledByInspection =
          editableDraft || policyInspection || (scriptedAdvance && submits);
        button.disabled =
          disabled ||
          Boolean(state.frame?.system_operation) ||
          (liveRunning(state) && submits) ||
          (turnInFlight && submits && !scriptedAdvance) ||
          scientificFenced ||
          !enabledByInspection ||
          policyControllerBlocksActionEdit(command) ||
          (scriptedAdvance && isTerminal(state.frame)) ||
          recordingDecision.action === "block" ||
          !mode.allowed;
      }
    }
    pendingRefusal = null;
    renderTacticsTurn();
  }

  /**
   * Show the policy worker status and the shared Run or Pause state, without
   * starting another turn.
   *
   * Reads the installed transport frame (`system_operation`, `run_state`,
   * `combat_configuration`, `fresh_launch_required`, `terminal`) and the
   * page's own Run flag (`state.pageRun`, see liveRunning). Hides Run
   * when the match has a Manual slot, is scripted, or the product is the
   * Replay Viewer. Cancel Policy Work shows only while the policies choose
   * (loading or thinking).
   *
   * Side effects: changes text, `hidden`, `disabled` and `aria-pressed` on the
   * deck's run controls; replaces the Cancel Policy Work button's `onclick`,
   * which sends one `cancel_system` command for the operation shown now;
   * repaints the deck and rail (renderTacticsTurn); and calls
   * scheduleLivePoll, which may start or clear one live read timer.
   *
   * @returns {void}
   */
  function renderSystemOperation() {
    const button = document.getElementById("cancel-system-button");
    const status = document.getElementById("system-operation-status");
    const operation = isRecord(state.frame) ? state.frame.system_operation : null;
    const choosing = ["loading", "thinking"].includes(operation?.state);
    if (button instanceof HTMLButtonElement) {
      button.hidden = !choosing || isReplayMode();
      button.disabled = state.busy || !choosing;
      button.onclick = () => {
        if (operation)
          void dispatchCommand({
            command_type: "cancel_system",
            operation_id: operation.operation_id,
          });
      };
    }
    if (status) {
      /** @type {Record<string, string>} */
      const labels = {
        loading: "Policies Loading",
        thinking: "Policies Choosing",
        finishing: "Policies Finishing",
        cancelling: "Cancelling Policy Work",
        failed: "Policy Cleanup Failed",
      };
      status.textContent = operation ? (labels[operation.state] ?? "") : "";
    }
    const manual = state.frame?.combat_configuration?.manual_slots;
    const running = liveRunning(state);
    const scripted = liveScriptedInspectionOnly();
    elements.liveRunButton.hidden =
      isReplayMode() || scripted || !Array.isArray(manual) || manual.length !== 0;
    elements.liveRunButton.textContent = running ? "Pause" : "Run";
    elements.liveRunButton.setAttribute("aria-pressed", String(running));
    // Pause of the page's own Run only clears a page flag, so it stays
    // available even while that Run's turn is out.
    elements.liveRunButton.disabled =
      (state.busy && state.pageRun !== true) ||
      state.offline ||
      state.resyncRequired ||
      state.shuttingDown ||
      !installedAuthorityIsCoherent() ||
      Boolean(state.frame?.fresh_launch_required) ||
      (!running &&
        (Boolean(operation) ||
          isTerminal(state.frame) ||
          recordingScientificControlsFenced()));
    elements.liveRunStatus.textContent = running ? "Running" : "Paused";
    renderTacticsTurn();
    scheduleLivePoll();
  }

  /** Where the This Turn help level is remembered in this browser. */
  const GUIDE_STORAGE_KEY = "marl-bgs.tactics-turn.guide";

  /**
   * The vision parts bindCommandDeck finds once, or null before it runs (or
   * when the page has no vision deck). Renderers skip missing nodes.
   *
   * @type {null | {
   *   deck: HTMLElement,
   *   agTok: HTMLElement | null, agName: HTMLElement | null, agEpi: HTMLElement | null,
   *   agMeta: HTMLElement | null, agChip: HTMLElement | null, agLife: HTMLElement | null,
   *   pad: HTMLElement | null, padNote: HTMLElement | null,
   *   tchip: HTMLElement | null, picks: HTMLElement | null,
   *   basicIcon: Element | null, basicName: HTMLElement | null, basicSub: HTMLElement | null,
   *   ultIcon: Element | null, ultName: HTMLElement | null, ultSub: HTMLElement | null,
   *   ultReady: HTMLElement | null, runeRing: Element | null,
   *   orderLine: HTMLElement | null, lockLabel: HTMLElement | null,
   *   watchOnly: HTMLElement | null, woTok: HTMLElement | null, woName: HTMLElement | null,
   *   woMeta: HTMLElement | null, woLife: HTMLElement | null, woBack: HTMLElement | null,
   *   stopped: HTMLElement | null,
   *   wbTok: HTMLElement | null, wbState: HTMLElement | null, wbNote: HTMLElement | null,
   *   watchSubmit: HTMLButtonElement | null,
   *   statusLine: HTMLElement | null, recSeal: HTMLElement | null, recSealText: HTMLElement | null,
   *   msg: HTMLElement | null, keysChip: HTMLElement | null, keysText: HTMLElement | null,
   *   arena: HTMLElement | null,
   *   turnCard: HTMLElement | null, turnEyebrow: HTMLElement | null, turnTitle: HTMLElement | null,
   *   setPill: HTMLElement | null, guideLevel: HTMLElement | null, doThis: HTMLElement | null,
   *   doThisText: HTMLElement | null, phases: HTMLElement | null, checklist: HTMLElement | null,
   *   classCard: HTMLElement | null, howtoCard: HTMLElement | null,
   * }}
   */
  let deckParts = null;

  /** True while the Allowed Now buttons must stay disabled (set each repaint). */
  let picksDisabled = true;

  /**
   * The status line's current sentence, its kind ("info" or "refuse"),
   * whether the deck wrote it and when (ms), and the vision refusal to show in
   * place of main.js's next policy warning (set by
   * policyControllerBlocksActionEdit).
   */
  let statusMessage = { text: "", kind: "info", fromDeck: false, at: 0 };

  /**
   * How long (ms) a refusal the deck just wrote outranks the page's late
   * info messages: the answer to an earlier order (for example "Step:
   * Stay.") must not hide the refusal of the key pressed after it.
   */
  const REFUSAL_HOLD_MS = 1500;

  /**
   * Sentences the deck announced for accepted inputs before its newest
   * refusal. When the page repeats one of them later (the host's answer to
   * that earlier input), it does not hide the refusal. A sentence leaves this
   * set when the deck announces it again.
   *
   * @type {Set<string>}
   */
  const echoesBeforeRefusal = new Set();

  /** Sentences announced since the newest refusal (moved into the set above). */
  /** @type {Set<string>} */
  const announcedSinceRefusal = new Set();
  /** @type {string | null} */
  let pendingRefusal = null;

  /**
   * The Lock In this page sent and is waiting on: its tick, its time (ms) and
   * whether nobody is Manual. Cleared when the tick moves on.
   *
   * @type {{tick: number, at: number, watch: boolean} | null}
   */
  let lockSent = null;

  /**
   * The tick that this page's last Lock In produced; while the page plays
   * that tick (the choreography's submission gate is closed) the phase is All
   * Act. Null before any Lock In.
   *
   * @type {number | null}
   */
  let lockResolvedTick = null;

  /**
   * How long All Act shows at least once a turn has played (ms). The new
   * tick's choreography usually covers it; with reduced motion the gate opens
   * at once, and this keeps the phase readable for a moment.
   */
  const ACT_HOLD_MS = 500;

  /**
   * How long (ms) Orders Sent shows before a Lock In still waiting for its
   * answer counts as Policies Choose (see currentPhase).
   */
  const SENT_BEAT_MS = 120;

  /** Time (ms) until which All Act shows after a played turn; 0 for none. */
  let actUntil = 0;

  /**
   * The tick this page saw arrive as a played turn (your Lock In, Play One
   * Turn or Run) and the page time (Date.now(), ms) it arrived, so the deck
   * can hold a fall until its Fall beat (fallHoldsOf). Null before any
   * played turn and for a tick the page only loaded (that tick shows its END
   * state at once, as the arena does).
   *
   * @type {{tick: number, at: number} | null}
   */
  let tickPlay = null;

  /** The tick whose Fall beat repaint is already scheduled, or null. */
  /** @type {number | null} */
  let fallRepaintTick = null;

  /**
   * What the deck remembers per champion you play, for this tick only:
   * whether you gave it an order input on this page (touched; it never makes
   * the champion Set, which is server truth, see manualRow, and only tells
   * whether an order the page cannot see may be unknown), whether you armed
   * its Basic yourself (explicitBasic: then the tag is Armed, not Auto) and
   * its last draft seen while it was in view (for the checklist and order
   * line when the server sent no staged row for it). `carried` is the draft
   * remembered on an earlier tick: after a tick the host puts the step back
   * to Stay but may keep the target and arm the Basic, so a carried target
   * means the deck cannot know that order until the champion is in view
   * again.
   *
   * @type {Map<string, {tick: number, touched: boolean, explicitBasic: boolean,
   *   draft: {moveAction: number, lane: "none" | "basic" | "ultimate",
   *   targetPublicId: string | null} | null, carried: {moveAction: number,
   *   lane: "none" | "basic" | "ultimate", targetPublicId: string | null} |
   *   null}>}
   */
  const edits = new Map();

  /**
   * Facts of the previous repaint, used to write announcements when they
   * change: the episode, the tick, the champion in view, the run state,
   * whether a fresh launch is needed and the game-over step last said
   * (gameOverStep; null before game over), so the step is said again when
   * the replay's save state changes.
   *
   * @type {{episode: unknown, tick: unknown, focus: unknown, running: boolean,
   *   stopped: boolean, started: boolean, gameOverStep: string | null}}
   */
  const previous = {
    episode: null,
    tick: null,
    focus: null,
    running: false,
    stopped: false,
    started: false,
    gameOverStep: null,
  };

  /** The Manual champion you ordered last (public ID); Back returns to it. */
  /** @type {string | null} */
  let lastManualId = null;

  /**
   * Last HTML written per part and the first node it made, so repaints skip
   * unchanged markup but notice when another module replaced the children.
   *
   * @type {WeakMap<Element, {html: string, first: ChildNode | null}>}
   */
  const writtenHtml = new WeakMap();

  /**
   * Read the remembered This Turn help level. Storage can be missing or can
   * throw (private windows, blocked site data); then "guide" (the default) is
   * used.
   *
   * @returns {"guide" | "hints" | "off"}
   */
  function readGuideMode() {
    try {
      const value = globalThis.localStorage?.getItem(GUIDE_STORAGE_KEY);
      return value === "hints" || value === "off" ? value : "guide";
    } catch {
      return "guide";
    }
  }

  /**
   * Remember the This Turn help level in this browser. A storage error is
   * ignored: the level still applies until the page reloads.
   *
   * @param {"guide" | "hints" | "off"} mode
   * @returns {void}
   */
  function writeGuideMode(mode) {
    try {
      globalThis.localStorage?.setItem(GUIDE_STORAGE_KEY, mode);
    } catch {
      // Storage is optional; the choice still holds for this page.
    }
  }

  let guideMode = readGuideMode();

  /**
   * Write markup into an element only when it differs from what the deck
   * wrote there last, so repaints do not rebuild unchanged parts.
   *
   * @param {Element | null} element The part, or null to do nothing.
   * @param {string} html Markup built from escaped text.
   * @returns {void}
   */
  function setHtml(element, html) {
    if (!element) return;
    const last = writtenHtml.get(element);
    if (last && last.html === html && last.first === element.firstChild) return;
    element.innerHTML = html;
    writtenHtml.set(element, { html, first: element.firstChild });
  }

  /**
   * What the short Lock In rows were last fitted for (fitShortOrderRows):
   * their markup, the list's width in CSS pixels and the list's first node
   * (another module replacing the rows makes a new one). html is "" while
   * the list shows full sentences.
   *
   * @type {{html: string, width: number, first: ChildNode | null}}
   */
  let orderFit = { html: "", width: -1, first: null };

  /**
   * What Allowed Now was last fitted for (fitAllowedNow): its first node and
   * its box in CSS pixels. A repaint that kept the buttons and the box reads
   * layout once and changes nothing.
   *
   * @type {{first: ChildNode | null, width: number, height: number}}
   */
  let picksFit = { first: null, width: -1, height: -1 };

  /**
   * Fit Allowed Now (fitPicks) when its buttons or its box changed since the
   * last fit.
   *
   * @param {boolean} [force] Fit even when nothing changed (fonts loaded).
   *   Default false.
   * @returns {void}
   */
  function fitAllowedNow(force = false) {
    const picks = deckParts?.picks;
    if (!(picks instanceof HTMLElement)) return;
    if (
      !force &&
      picksFit.first === picks.firstChild &&
      picksFit.width === picks.clientWidth &&
      picksFit.height === picks.clientHeight
    ) {
      return;
    }
    fitPicks(picks);
    picksFit = {
      first: picks.firstChild,
      width: picks.clientWidth,
      height: picks.clientHeight,
    };
  }

  /**
   * Fit the short Lock In rows whole on one line (fitOrderRows) when their
   * markup, the list's width or the list's nodes changed since the last fit,
   * so an ordinary repaint reads layout once and changes nothing. With html
   * "" (the list shows full sentences) it only forgets the last fit.
   *
   * @param {string} html The rows' markup as the deck just wrote it, or "".
   * @param {boolean} [force] Fit even when nothing changed (fonts loaded).
   *   Default false.
   * @returns {void}
   */
  function fitShortOrderRows(html, force = false) {
    const list = deckParts?.orderLine;
    if (html === "" || !(list instanceof HTMLElement)) {
      orderFit = { html: "", width: -1, first: null };
      return;
    }
    const width = list.clientWidth;
    if (
      !force &&
      orderFit.html === html &&
      orderFit.width === width &&
      orderFit.first === list.firstChild
    ) {
      return;
    }
    fitOrderRows(list);
    orderFit = { html, width, first: list.firstChild };
  }

  /**
   * Set an element's text only when it changed.
   *
   * @param {Element | null} element The part, or null to do nothing.
   * @param {string} text Plain text.
   * @returns {void}
   */
  function setText(element, text) {
    if (element && element.textContent !== text) element.textContent = text;
  }

  /**
   * Show a champion's life in host: the one shared Life bar (`life-bar.js`:
   * "Life", a thin bar colored green, yellow or red, and "77 / 80"; owner, 2
   * October 2026) when words is null and model is given, else words as a
   * muted `span.life-words` ("Fallen · Returns In 3 Ticks", "Out Of Sight"), or
   * nothing for "". The bar is rebuilt on every call, so it always shows the
   * current life. Agent POV: callers pass words, never a model, for a
   * champion the team cannot see.
   *
   * @param {Element | null} host The life slot (#ag-life, #wo-life or the
   *   target chip's .tchip-life), or null to do nothing.
   * @param {string | null} words The words to show instead of the bar, or
   *   null for the bar.
   * @param {import("./life-bar.js").LifeBarModel | null} model The bar's model.
   * @returns {void}
   */
  function paintLife(host, words, model) {
    if (!host) return;
    if (words === null && model !== null) {
      host.replaceChildren(createLifeBar(document, model));
      return;
    }
    if (!words) {
      host.replaceChildren();
      return;
    }
    const line = document.createElement("span");
    line.className = "life-words";
    line.textContent = words;
    host.replaceChildren(line);
  }

  /**
   * Return the class glyph markup for a token: the vision's `#sig-<class>`
   * symbol when the page has it, else the class emblem from icons.js.
   *
   * @param {string} classKey "mage" … "priest".
   * @param {"M" | "XS"} size Token size.
   * @returns {string} SVG markup placed in the token's glyph box.
   */
  function glyphMarkup(classKey, size) {
    const box = tokenGlyphBox(size);
    if (document.getElementById(`sig-${classKey}`)) {
      return `<use href="#sig-${classKey}" x="${box.at}" y="${box.at}" width="${box.width}" height="${box.width}"/>`;
    }
    const icon = createSvgIcon(document, `class-${classKey}`, {
      className: `tok-glyph tok-glyph--${classKey}`,
    });
    icon.setAttribute("x", String(box.at));
    icon.setAttribute("y", String(box.at));
    icon.setAttribute("width", String(box.width));
    icon.setAttribute("height", String(box.width));
    return icon.outerHTML;
  }

  /**
   * Return a champion token's markup for one roster or scene agent.
   *
   * With a scene agent of this audience's own scene (it has a position and a
   * radius), the token is the arena's own V5 token (hud.js championTokenSvg:
   * the Champions and Statuses looks, following the token switches), 46 px
   * square at radius 20.25 for size "M" and 20 px at radius 8.1 for "XS", so
   * a token switch changes the deck's tokens too. Without one (a roster-only
   * champion), the vision's flat token (tokenSvg) is drawn. Agent POV: callers
   * pass only scene agents, never a champion the team cannot see.
   *
   * @param {ReturnType<typeof championFacts>} facts The champion's facts.
   * @param {"M" | "XS"} size Token size.
   * @param {{lit?: number, ready?: boolean, faded?: boolean}} [extra] Rune
   *   ticks and fallen look of the flat token (size "M" only).
   * @param {{agent?: Record<string, any> | null, scene?: Record<string, any> | null}} [source]
   *   The champion's scene agent (default none) and the scene (its Spawn
   *   Shield mechanics; default none).
   * @returns {string} Empty for unknown facts.
   */
  function championToken(facts, size, extra = {}, source = {}) {
    if (facts === null || facts.team === null || facts.classKey === "none") return "";
    const agent = source.agent ?? null;
    if (
      isRecord(agent) &&
      Array.isArray(agent.position) &&
      Number.isFinite(Number(agent.radius))
    ) {
      const svg = championTokenSvg(document, agent, {
        size: size === "M" ? 46 : 20,
        radius: size === "M" ? 20.25 : 8.1,
        filters: visualFilters(),
        spawnShieldMechanics: isRecord(source.scene?.spawn_shield_mechanics)
          ? source.scene.spawn_shield_mechanics
          : null,
      });
      return svg.outerHTML;
    }
    return tokenSvg({
      classKey: facts.classKey,
      team: facts.team,
      size,
      glyph: glyphMarkup(facts.classKey, size),
      ...extra,
    });
  }

  /**
   * Return the scene agent with this public ID from the deck's data, or null.
   * A champion whose fall still waits for its Fall beat (data.fallHolds) is
   * returned as a copy that is still alive with its life from before the
   * tick, so the deck's and the rail's tokens fall at the Fall beat, with the
   * arena. The scene's own record is never changed.
   *
   * @param {{sceneAgents: ReadonlyArray<Record<string, any>>, fallHolds?:
   *   ReadonlyMap<string, {before: number | null, until: number}>}} data Deck
   *   data (deckData); sceneAgents are this audience's own scene agents, so in
   *   Agent POV an unseen champion is never found.
   * @param {unknown} publicId A public agent ID such as "agent_5".
   * @returns {Record<string, any> | null}
   */
  function sceneAgentById(data, publicId) {
    if (typeof publicId !== "string") return null;
    const agent =
      data.sceneAgents.find((candidate) => candidate?.public_agent_id === publicId) ??
      null;
    const held = data.fallHolds?.get(publicId);
    if (agent === null || held === undefined) return agent;
    // The V5 token reads `alive` (looks/champions.js); life_state is what
    // the deck reads.
    return {
      ...agent,
      alive: true,
      life_state: "alive",
      current_health: held.before ?? agent.current_health,
    };
  }

  /**
   * Point an ability icon at the vision's `#ab-…` symbol, or, when the page
   * has no such symbol, draw the icons.js glyph beside it in the same box.
   *
   * @param {Element | null} use The `<use>` element (#basic-icon or #ult-icon).
   * @param {string} symbolId The vision symbol id, for example "ab-charge".
   * @param {string} glyphKey The icons.js glyph to draw instead.
   * @returns {void}
   */
  function setAbilityIcon(use, symbolId, glyphKey) {
    if (!use) return;
    const parent = use.parentElement;
    const old = parent?.querySelector("[data-icon-fallback]");
    if (document.getElementById(symbolId)) {
      if (use.getAttribute("href") !== `#${symbolId}`)
        use.setAttribute("href", `#${symbolId}`);
      old?.remove();
      return;
    }
    use.removeAttribute("href");
    if (old?.getAttribute("data-icon-fallback") === glyphKey) return;
    old?.remove();
    const icon = createSvgIcon(document, glyphKey, {});
    icon.setAttribute("data-icon-fallback", glyphKey);
    for (const name of ["x", "y", "width", "height"]) {
      const value = use.getAttribute(name);
      if (value !== null) icon.setAttribute(name, value);
    }
    if (!use.hasAttribute("width")) {
      icon.setAttribute("width", "24");
      icon.setAttribute("height", "24");
    }
    parent?.insertBefore(icon, use);
  }

  /**
   * Return the champions whose fall on the tick now playing has not reached
   * its Fall beat yet, so the deck and the rail say "fallen" at the same
   * moment the arena shows the fall and the scorebug changes the score
   * (owner: life and score wait for the Fall beat).
   *
   * Times come from the one owner of held values, beatView in
   * choreography-plan.js: each fall the audience saw has its Fall beat time
   * `at` (real ms from the start of the tick at normal motion). The tick
   * starts when this page saw it arrive as a played turn (tickPlay, set by
   * trackChanges). Reduced motion multiplies the time by the schedule's
   * reducedScale; motion "off" (the page's data-motion-mode) holds nothing,
   * as the choreographer does. A tick the page only loaded holds nothing.
   *
   * Agent POV: beatView lists only falls the audience saw, so nothing here
   * depends on an unseen champion. Reads the clock (Date.now()) and the
   * page's motion mode; changes nothing.
   *
   * @param {unknown} presentation The installed presentation.
   * @param {number | null} tick The installed tick.
   * @returns {Map<string, {before: number | null, until: number}>} Keyed by
   *   public ID: the life before the tick (null when the frame has none) and
   *   the page time (ms) of the Fall beat. Empty when nothing is held.
   */
  function fallHoldsOf(presentation, tick) {
    /** @type {Map<string, {before: number | null, until: number}>} */
    const holds = new Map();
    if (tickPlay === null || tick === null || tickPlay.tick !== tick) return holds;
    const motion = document.documentElement.dataset.motionMode;
    if (motion === "off") return holds;
    const filters = visualFilters();
    const view = beatView(
      presentation,
      "end",
      filters ? { visualFilters: filters } : {},
    );
    if (view === null) return holds;
    const scale = motion === "reduced" ? Number(view.schedule.reducedScale) || 1 : 1;
    const now = Date.now();
    for (const entry of view.life) {
      const until = tickPlay.at + entry.at * scale;
      if (now < until) {
        holds.set(entry.publicAgentId, { before: entry.before, until });
      }
    }
    return holds;
  }

  /**
   * Gather what the deck and the rail show, from the installed frame.
   *
   * Reads the audience's arena scene and its own live inspection (the draft,
   * target and masks of the champion whose view it is), the static roster
   * (identity only), the transport frame's run facts and the episode tick
   * limit. `live` is false in the Replay Viewer and before a live frame is
   * installed; then the other fields are empty.
   *
   * Which champion is in focus: in Agent POV the host keeps your Manual
   * champion's view when you look at a policy's champion, so the audience's
   * inspection still describes your champion. The focus ID then comes from
   * the transport frame's focused_public_agent_id (sent only in Agent POV
   * while the focus is not the recipient), else from the researcher
   * inspection, as command plumbing only: the host lets the
   * POV team focus only its own champions and enemies it sees now, and every
   * fact the deck shows about the focused champion comes from the audience's
   * scene (a body that scene does not draw shows its class and ID only).
   * `draft`, `target` and `mask` are the focused champion's only when the
   * audience's inspection is about it; otherwise they are null (Watch Only).
   * `manualIds` lists every Manual champion of both teams (command plumbing:
   * is the focus yours, the start greeting). `shownManualIds` is the part
   * the page may SHOW staged orders for (shownManualIdsOf): in Agent POV only
   * the viewing team's Manual champions (ruling L18), in Oracle View all of
   * them. The "N Of M Set" pill, Do This Now's "still has the default order"
   * hint, the Lock In order list and the rail rows' orders read only
   * `shownManualIds`; the rail still lists every Manual champion, the others
   * with no order ("Out Of Sight" or "In Sight · Order Hidden"). `servedOrders` holds the server's staged order of
   * every champion in `shownManualIds` (servedManualOrdersOf: in Oracle View
   * every Manual champion, in Agent POV the viewing team's own, ruling P1);
   * a row not in view prefers it to this page's memory, so orders kept from
   * earlier ticks show in both views. `fallHolds` lists the champions whose
   * fall this tick is still waiting for its Fall beat (fallHoldsOf): until
   * then the deck and the rail show them as they were before the tick.
   * `recording` is the frame's recording status (lifecycle and the save
   * actions), or null when the match records nothing; once Exit was accepted
   * it is the Exit answer's own status (state.exitRecording) when main.js
   * kept one. `closed` is true once Exit was accepted (state.shuttingDown):
   * the page reads no more frames.
   * `controllers` is the one controller model (controllerRosterFromFrame in
   * match-summary.js, from the combat configuration and the identity
   * directory, the facts the scorebug's lines use), NO_CONTROLLER_ROSTER
   * outside live play; the Policies list and every "Played By" name read it.
   */
  function deckData() {
    const presentation = state.presentation;
    const frame = isRecord(state.frame) ? state.frame : null;
    const live =
      !isReplayMode() &&
      frame !== null &&
      isAuthorizedPresentationFrame(presentation) &&
      presentation?.viewer_mode === "live";
    const scene = live ? authorizedPresentationSceneView(presentation) : null;
    const sceneAgents = asArray(scene?.agents).filter(isRecord);
    const rosterRows = live ? authorizedPresentationIdentityRows(presentation) : [];
    const rosterAgents = rosterRows.map((row) => row.agent);
    const inspectionState = live
      ? authorizedPresentationInspectionState(presentation)
      : null;
    const inspection = isRecord(inspectionState?.inspection)
      ? inspectionState.inspection
      : null;
    const viewId =
      typeof inspection?.actor_public_agent_id === "string"
        ? inspection.actor_public_agent_id
        : null;
    const lookedAt = live
      ? authorizedPresentationResearcherInspectionState(presentation).inspection
      : null;
    const povFocus =
      live && typeof frame?.focused_public_agent_id === "string"
        ? frame.focused_public_agent_id
        : null;
    const focusId =
      povFocus ??
      (typeof lookedAt?.actor_public_agent_id === "string"
        ? lookedAt.actor_public_agent_id
        : viewId);
    const ownView = viewId !== null && viewId === focusId;
    const viewDraft = isRecord(inspection?.draft_action)
      ? inspection.draft_action
      : null;
    const focusedScene =
      sceneAgents.find((agent) => agent.public_agent_id === focusId) ?? null;
    const focusedRoster =
      rosterAgents.find((agent) => agent?.public_agent_id === focusId) ?? null;
    const configuration = frame?.combat_configuration;
    const manualSlots = asArray(configuration?.manual_slots);
    /** @param {unknown} publicId */
    const controllerOf = (publicId) =>
      live
        ? authorizedPresentationController(presentation, configuration, publicId)
        : null;
    const manualIds = rosterRows
      .filter((row) => controllerOf(row.public_agent_id)?.manual === true)
      .map((row) => row.public_agent_id);
    const shownManualIds = shownManualIdsOf(
      presentation,
      sceneAgents,
      rosterRows,
      manualIds,
    );
    const servedOrders = live
      ? servedManualOrdersOf(presentation, rosterRows, shownManualIds)
      : servedManualOrdersOf(null, []);
    // The one controller model (controller-names.js): the rail's Policies
    // list and every "Played By" name come from it, as the scorebug's lines
    // and hover lists do.
    const controllers = live
      ? controllerRosterFromFrame(presentation, configuration)
      : NO_CONTROLLER_ROSTER;
    const mechanics = asArray(scene?.class_mechanics).filter(isRecord);
    /** @param {unknown} classId */
    const mechanicsOf = (classId) =>
      mechanics.find((row) => row.class_id === classId) ?? null;
    const operationState = isRecord(frame?.system_operation)
      ? frame.system_operation.state
      : undefined;
    const tick = Number.isInteger(frame?.simulator_step_count)
      ? Number(frame?.simulator_step_count)
      : null;
    const waves = asArray(scene?.respawn_waves).filter(isRecord);
    return {
      live,
      presentation,
      frame,
      scene,
      sceneAgents,
      rosterRows,
      rosterAgents,
      inspection,
      editable: ownView && inspectionState?.state_kind === "live_editable",
      viewId,
      viewDraft,
      viewTarget: isRecord(inspection?.draft_target) ? inspection.draft_target : null,
      viewMask: isRecord(inspection?.decision_mask) ? inspection.decision_mask : null,
      focusId,
      focusedScene,
      focusedRoster,
      draft: ownView ? viewDraft : null,
      target:
        ownView && isRecord(inspection?.draft_target) ? inspection.draft_target : null,
      mask:
        ownView && isRecord(inspection?.decision_mask)
          ? inspection.decision_mask
          : null,
      manualIds,
      shownManualIds,
      servedOrders,
      fallHolds: live ? fallHoldsOf(presentation, tick) : new Map(),
      // After an accepted Exit, the Exit answer's own recording status is
      // newer than the installed frame's (Exit waits for a running save).
      recording:
        state.shuttingDown === true && isRecord(state.exitRecording)
          ? state.exitRecording
          : isRecord(frame?.recording)
            ? frame.recording
            : null,
      watchMode:
        live && Array.isArray(configuration?.manual_slots) && manualSlots.length === 0,
      controllerOf,
      controllers,
      mechanicsOf,
      operationState,
      tick,
      limit: live ? authorizedPresentationEpisodeLimit(presentation) : null,
      stopped: Boolean(frame?.fresh_launch_required),
      terminal: frame !== null && isTerminal(frame),
      closed: state.shuttingDown === true,
      running: liveRunning(state),
      /**
       * Ticks until the team's next respawn wave brings its fallen champions
       * back, or null when the frame has no wave row for the team. Core
       * brings them back on the transition that starts with a countdown of
       * 0, so this is countdown_steps + 1 (a countdown of 1 at tick 98 means
       * back at tick 100: "2 ticks"), the same number as the portraits'
       * "Returns In" and the scorebug's "Respawn Wave In N Ticks". The wave
       * clocks are public match facts, the same in Agent POV and Oracle View.
       *
       * @param {unknown} teamId Team 1 (Team A) or 2 (Team B).
       * @returns {number | null}
       */
      waveIn: (teamId) => {
        const wave = waves.find((row) => row.team_id === teamId);
        return Number.isInteger(wave?.countdown_steps)
          ? Number(wave?.countdown_steps) + 1
          : null;
      },
    };
  }

  /**
   * Return the current turn phase (turnPhase) from the installed frame, this
   * page's pending Lock In and the choreography's submission gate. A Lock In
   * still waiting for its answer after SENT_BEAT_MS, in a match with policy
   * champions, counts as Policies Choose even before the page has read the
   * host's "thinking" state (the host runs the policies while it answers).
   * A stopped game (the frame's `fresh_launch_required`) is always "plan".
   *
   * @returns {"plan" | "sent" | "choosing" | "act"}
   */
  function currentPhase() {
    const frame = isRecord(state.frame) ? state.frame : null;
    const operationState = isRecord(frame?.system_operation)
      ? frame.system_operation.state
      : undefined;
    const tick = frame?.simulator_step_count;
    const sent =
      lockSent !== null &&
      lockSent.tick === tick &&
      (state.busy || Date.now() - lockSent.at < 1500);
    // A fast host can finish the policies' turn before the page reads its
    // "thinking" state. While a Lock In is still waiting for its answer in a
    // match with policy champions, the policies are choosing: say so after a
    // short Orders Sent beat.
    const configuration = frame?.combat_configuration;
    const hasPolicies =
      asArray(configuration?.manual_slots).length <
      asArray(configuration?.slot_labels).length;
    const inferredChoosing =
      sent &&
      state.busy &&
      hasPolicies &&
      lockSent !== null &&
      Date.now() - lockSent.at >= SENT_BEAT_MS;
    return turnPhase({
      stopped: Boolean(frame?.fresh_launch_required),
      running: liveRunning(state),
      operationState:
        inferredChoosing && operationState === undefined ? "thinking" : operationState,
      lockSent: sent,
      // The submission gate also closes for other replays of a tick (for
      // example the first frame), so it counts only for a tick a Lock In or
      // Run produced.
      choreographyBlocking:
        (document.documentElement.dataset.submissionBlocked === "true" &&
          (lockSent !== null ||
            (lockResolvedTick !== null && lockResolvedTick === tick) ||
            liveRunning(state))) ||
        (lockResolvedTick === tick && Date.now() < actUntil),
    });
  }

  /**
   * Return the deck's per-tick memory of one champion you play.
   *
   * @param {string} publicId Public agent ID.
   * @param {number | null} tick Current tick.
   * @returns {{tick: number, touched: boolean, explicitBasic: boolean, draft:
   *   {moveAction: number, lane: "none" | "basic" | "ultimate", targetPublicId:
   *   string | null} | null, carried: {moveAction: number, lane: "none" |
   *   "basic" | "ultimate", targetPublicId: string | null} | null}}
   */
  function editOf(publicId, tick) {
    const entry = edits.get(publicId);
    if (entry && entry.tick === (tick ?? -1)) return entry;
    const fresh = {
      tick: tick ?? -1,
      touched: false,
      explicitBasic: false,
      draft: null,
      carried: entry ? (entry.draft ?? entry.carried) : null,
    };
    edits.set(publicId, fresh);
    return fresh;
  }

  /**
   * Return the picked target as facts and its mask pair, from the audience's
   * inspection: of the focused champion (default), or of the champion whose
   * view the audience's inspection is (`ofView`, your Manual champion while
   * you look at a policy's champion).
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @param {boolean} [ofView] True for the view's champion. Default false.
   * @returns {{chip: ReturnType<typeof targetChipView>, targetAction: number,
   *   basicOk: boolean, ultimateOk: boolean, noTargetUltimateOk: boolean,
   *   name: string | null, short: string | null, publicId: string | null}}
   */
  function pickedTarget(data, ofView = false) {
    const draft = ofView ? data.viewDraft : data.draft;
    const mask = ofView ? data.viewMask : data.mask;
    const chip = targetChipView(
      ofView ? data.viewTarget : data.target,
      data.sceneAgents,
      data.rosterAgents,
    );
    const targetAction = Number.isInteger(draft?.target_action)
      ? Number(draft?.target_action)
      : 0;
    const pairs = asArray(mask?.target_use_ultimate_joint_mask);
    const pair = asArray(pairs[targetAction]);
    const facts = chip.state === "none" ? null : chip.facts;
    return {
      chip,
      targetAction,
      basicOk: targetAction > 0 && pair[0] === true,
      ultimateOk: targetAction > 0 && pair[1] === true,
      noTargetUltimateOk: asArray(pairs[0])[1] === true,
      name: facts ? facts.nameWithId : null,
      short: facts ? `${facts.name} ${facts.number}` : null,
      publicId:
        chip.state === "none"
          ? null
          : String((ofView ? data.viewTarget : data.target)?.target_public_agent_id),
    };
  }

  /**
   * Whether the class's Ultimate takes no target (the Mage's Burst).
   *
   * @param {Record<string, any> | null} mechanics The class_mechanics row.
   * @param {unknown} classId Class ID, used when mechanics are missing.
   * @returns {boolean}
   */
  function ultimateNeedsNoTarget(mechanics, classId) {
    if (mechanics) return mechanics.ultimate_target_mode === "target_none";
    return classId === 1;
  }

  /**
   * Return whether the champion you order is fallen, and when it returns
   * (returnsIn: ticks until its team's respawn wave, from data.waveIn, so 1
   * means it is back after the next Lock In; null with no wave row). A fall
   * still waiting for its Fall beat (data.fallHolds) is not fallen yet.
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @returns {{className: string, returnsIn: number | null} | null}
   */
  function fallenOf(data) {
    const agent = data.focusedScene;
    if (agent === null || agent.life_state === "alive") return null;
    if (!data.manualIds.includes(String(data.focusId))) return null;
    if (data.fallHolds.has(String(data.focusId))) return null;
    const facts = championFacts(agent);
    return {
      className: facts?.name ?? "Champion",
      returnsIn: data.waveIn(agent.team_id),
    };
  }

  /**
   * Show one sentence in the status line. The line always keeps room for two
   * lines, so a new sentence never moves the page.
   *
   * @param {string} text The sentence ("" clears it).
   * @param {"info" | "refuse"} kind "refuse" draws the Alert edge.
   * @param {boolean} fromDeck True when the deck wrote it (routine page
   *   messages then never replace it).
   * @returns {void}
   */
  function say(text, kind, fromDeck) {
    if (fromDeck && kind === "refuse") {
      for (const echo of announcedSinceRefusal) echoesBeforeRefusal.add(echo);
      announcedSinceRefusal.clear();
    } else if (fromDeck && text !== "") {
      announcedSinceRefusal.add(text);
      echoesBeforeRefusal.delete(text);
    }
    statusMessage = { text, kind, fromDeck, at: Date.now() };
    paintMessage();
  }

  /**
   * Paint the status line's message from statusMessage. A controller named
   * after "played by" is cut with "…" past the fixed name width
   * (paintNamedSentence in controller-names.js), so the line keeps to two
   * lines.
   *
   * @returns {void}
   */
  function paintMessage() {
    const msg = deckParts?.msg;
    if (!msg) return;
    // A "played by <name>" sentence cuts the name with "…" like every
    // controller name, so a long name never pushes the line past two lines.
    paintNamedSentence(msg, statusMessage.text);
    msg.classList.toggle("refuse", statusMessage.kind === "refuse");
    msg.dataset.level = statusMessage.kind === "refuse" ? "warning" : "info";
  }

  /**
   * Show one plain message from the page in the status line. Messages only
   * repeat what the host or the page said; they never guess a cause.
   *
   * Routine messages ("", "Waiting for the game to answer…", "Loading the
   * current match…") never replace a sentence the deck wrote for your last
   * input, and no info message replaces a refusal the deck wrote in the last
   * 1.5 s or repeats a sentence the deck announced before that refusal (both
   * are the late answer to an earlier order). A warning right after
   * the deck refused a policy's champion shows the vision's refusal sentence
   * instead of the page's wording. Warnings and errors draw the Alert edge.
   * A host notice with a plainer player sentence is shown in those words
   * (plainHostNotice), so no developer words reach the line.
   *
   * @param {string} message The text ("" clears the line unless the deck
   *   wrote it).
   * @param {string} [level] "info" (default), "warning" or "error".
   * @returns {void}
   */
  function showStatusMessage(message, level = "info") {
    const text = plainHostNotice(String(message ?? ""));
    const warning = level === "warning" || level === "error";
    if (warning && pendingRefusal !== null) {
      const refusal = pendingRefusal;
      pendingRefusal = null;
      say(refusal, "refuse", true);
      return;
    }
    const freshRefusal =
      statusMessage.kind === "refuse" &&
      statusMessage.fromDeck &&
      (Date.now() - statusMessage.at < REFUSAL_HOLD_MS ||
        echoesBeforeRefusal.has(text));
    if (
      !warning &&
      (freshRefusal || (ROUTINE_MESSAGES.has(text) && statusMessage.fromDeck))
    ) {
      paintMessage();
      return;
    }
    say(text, warning ? "refuse" : "info", !ROUTINE_MESSAGES.has(text));
  }

  /**
   * Restart a control's shake animation (a refused press).
   *
   * @param {Element | null | undefined} element The control, or nothing.
   * @returns {void}
   */
  function shake(element) {
    if (!(element instanceof HTMLElement)) return;
    element.classList.remove("shake");
    void element.offsetWidth;
    element.classList.add("shake");
  }

  /**
   * Turn a key or a button into a deck intent, or null when the deck has no
   * opinion about it. Enter, Space and N (one tick, the same Lock In as
   * Space in live play) are the "lock" intent; scripted playback skips the
   * deck's checks (acceptIntent), so there N stays the next frame.
   *
   * @param {string} key The key or the button's `data-key` ("w", "1", " ",
   *   "Enter", "n", "Escape").
   * @returns {{kind: "step", moveAction: number} | {kind: "lane", lane: 0 | 1 | 2}
   *   | {kind: "clear"} | {kind: "lock"} | null}
   */
  function intentForKey(key) {
    const lower = key.toLowerCase();
    const step = STEP_PAD_KEYS[lower];
    if (step) return { kind: "step", moveAction: step.moveAction };
    if (lower === "0" || lower === "1" || lower === "2") {
      return { kind: "lane", lane: /** @type {0 | 1 | 2} */ (Number(lower)) };
    }
    if (lower === "escape") return { kind: "clear" };
    if (lower === "enter" || lower === " " || lower === "spacebar" || lower === "n")
      return { kind: "lock" };
    return null;
  }

  /**
   * Check one order input against the deck's refusals (deckRefusal) using the
   * installed frame. On refusal, show the sentence, shake `element` and return
   * false; otherwise remember the input (touched, explicit Basic, Lock In),
   * show its announcement and return true. While a request is out
   * (state.busy), a 1 or 2 skips the target checks: main.js may queue it behind
   * a target pick that is not shown yet and judges it against that order.
   *
   * @param {{kind: "step", moveAction: number} | {kind: "lane", lane: 0 | 1 | 2}
   *   | {kind: "clear"} | {kind: "lock"} | {kind: "target", publicId: string}} intent
   * @param {{element?: Element | null, viaKey?: boolean}} [options] element
   *   is the control to shake; viaKey is true for a key press.
   * @returns {boolean} True when the input may be sent.
   */
  function acceptIntent(intent, options = {}) {
    const data = deckData();
    if (!data.live || liveScriptedInspectionOnly()) return true;
    const phase = currentPhase();
    const focusedManual =
      data.focusId !== null && data.manualIds.includes(data.focusId);
    const classId = data.focusedScene?.class_id ?? data.focusedRoster?.class_id;
    const mechanics = data.mechanicsOf(classId);
    const noTarget = ultimateNeedsNoTarget(mechanics, classId);
    const picked = pickedTarget(data);
    const cooldown = Number.isInteger(data.focusedScene?.ultimate_cooldown_remaining)
      ? Number(data.focusedScene?.ultimate_cooldown_remaining)
      : null;
    if (intent.kind === "clear") {
      const editable =
        focusedManual && phase === "plan" && !data.stopped && !data.terminal;
      if (editable && picked.chip.state !== "none")
        touch(data, { explicitBasic: false });
      say(
        deckAnnouncement({
          intent: { kind: "clear", viaKey: options.viaKey === true, editable },
          classId,
          ultimateNeedsNoTarget: noTarget,
          targetName: picked.name,
          watchMode: data.watchMode,
        }),
        "info",
        true,
      );
      return true;
    }
    // While an earlier order is still out, main.js may queue this key behind
    // a target pick that is not shown yet, so the target checks wait for it
    // (main.js judges the key against the order it will actually change).
    const targetPending = state.busy === true && intent.kind === "lane";
    const decision = deckRefusal({
      intent:
        intent.kind === "target"
          ? { kind: "target" }
          : /** @type {{kind: "step", moveAction: number} | {kind: "lane", lane: 0 | 1 | 2} | {kind: "lock"}} */ (
              intent
            ),
      stopped: data.stopped,
      terminal: data.terminal,
      phase,
      watchMode: data.watchMode,
      focusedManual,
      fallen: fallenOf(data) !== null,
      moveAllowed:
        intent.kind !== "step" ||
        asArray(data.mask?.movement_action_mask)[intent.moveAction] === true,
      classId,
      ultimateNeedsNoTarget: noTarget,
      hasTarget: targetPending || picked.chip.state !== "none",
      basicOk: targetPending || picked.basicOk,
      ultimateOk:
        targetPending || (noTarget ? picked.noTargetUltimateOk : picked.ultimateOk),
      cooldown,
    });
    if (decision.refused) {
      if (decision.text !== null) say(decision.text, "refuse", true);
      shake(options.element ?? null);
      return false;
    }
    if (intent.kind === "lock") {
      lockSent = { tick: Number(data.tick), at: Date.now(), watch: data.watchMode };
      // Repaint once the Orders Sent beat is over (Policies Choose).
      setTimeout(renderCommandAvailability, SENT_BEAT_MS + 10);
      const button = data.watchMode
        ? deckParts?.watchSubmit
        : elements.submitTurnButton;
      if (button instanceof HTMLElement) {
        button.classList.remove("sweep");
        void button.offsetWidth;
        button.classList.add("sweep");
      }
      say(
        deckAnnouncement({
          intent: { kind: "lock" },
          classId,
          ultimateNeedsNoTarget: noTarget,
          targetName: picked.name,
          watchMode: data.watchMode,
        }),
        "info",
        true,
      );
      renderTacticsTurn();
      return true;
    }
    if (intent.kind === "target") {
      const target = asArray(data.mask?.target_actions).find(
        (row) => isRecord(row) && row.target_public_agent_id === intent.publicId,
      );
      const pair = asArray(
        asArray(data.mask?.target_use_ultimate_joint_mask)[
          Number(target?.target_action)
        ],
      );
      const body =
        data.sceneAgents.find((agent) => agent.public_agent_id === intent.publicId) ??
        data.rosterAgents.find((agent) => agent?.public_agent_id === intent.publicId);
      const facts = championFacts(body);
      touch(data, { explicitBasic: false });
      say(
        deckAnnouncement({
          intent: {
            kind: "target",
            basicOk: pair[0] === true,
            ultimateOk: pair[1] === true,
            targetName: facts?.nameWithId ?? "that champion",
          },
          classId,
          ultimateNeedsNoTarget: noTarget,
          targetName: picked.name,
          watchMode: data.watchMode,
        }),
        "info",
        true,
      );
      return true;
    }
    touch(data, intent.kind === "lane" ? { explicitBasic: intent.lane === 1 } : {});
    say(
      deckAnnouncement({
        intent,
        classId,
        ultimateNeedsNoTarget: noTarget,
        targetName: picked.name,
        watchMode: data.watchMode,
      }),
      "info",
      true,
    );
    return true;
  }

  /**
   * Remember that you gave the champion in view an order input this tick
   * (its Set or Default still comes from the server's order, manualRow).
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @param {{explicitBasic?: boolean}} change explicitBasic true when you
   *   armed the Basic yourself (key 1), false when a target pick or a clear
   *   lets the game arm it; omitted keeps the old value.
   * @returns {void}
   */
  function touch(data, change) {
    if (data.focusId === null) return;
    const entry = editOf(data.focusId, data.tick);
    entry.touched = true;
    if (change.explicitBasic !== undefined) entry.explicitBasic = change.explicitBasic;
  }

  /**
   * Write announcements for changes the page did not start with a deck input:
   * a new game (startGreeting, true for the roster mix), a new tick (the
   * game-over step when that tick ends the game), a new champion in view, Run
   * stopping, policy work being cancelled, and at game over a change of the
   * replay's save state (the game-over step is said again, gameOverStep, but
   * only while the line still shows the old step). Also keeps the per-tick
   * memory, the last Manual champion and the played-tick clock (tickPlay) up
   * to date.
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @returns {boolean} True when this repaint saw a played turn arrive (the
   *   caller then gathers the facts again, so a fall held until its Fall beat
   *   is held from the tick's first paint).
   */
  function trackChanges(data) {
    if (!data.live) return false;
    let played = false;
    const episode = data.frame?.episode_id;
    if (episode !== previous.episode) {
      previous.episode = episode;
      previous.started = false;
      previous.gameOverStep = null;
      edits.clear();
      lockSent = null;
      lockResolvedTick = null;
      lastManualId = null;
      tickPlay = null;
    }
    const focusedManual =
      data.focusId !== null && data.manualIds.includes(data.focusId);
    if (focusedManual) lastManualId = data.focusId;
    // A new setup can make the last Manual champion a policy's: fall back to
    // the first Manual champion (or none).
    // In Agent POV it stays on the viewing team (shownManualIds).
    if (lastManualId === null || !data.shownManualIds.includes(lastManualId))
      lastManualId = data.shownManualIds[0] ?? null;
    if (!previous.started) {
      previous.started = true;
      previous.tick = data.tick;
      previous.focus = data.focusId;
      previous.running = data.running;
      previous.stopped = data.stopped;
      const teamOf = (/** @type {string} */ publicId) =>
        data.rosterRows.find((row) => row.public_agent_id === publicId)?.agent?.team_id;
      say(
        startGreeting({
          watchMode: data.watchMode,
          teamA: data.manualIds.filter((publicId) => teamOf(publicId) === 1).length,
          teamB: data.manualIds.filter((publicId) => teamOf(publicId) === 2).length,
          pov: authorizedPresentationAudience(data.presentation) === "agent_pov",
        }),
        "info",
        true,
      );
      previous.gameOverStep = data.terminal ? gameOverStep(data.recording) : null;
      return false;
    }
    if (data.stopped && !previous.stopped) {
      say(
        "Policy work cancelled. The game so far is kept. A fresh start from Python is needed to play on.",
        "info",
        true,
      );
    } else if (data.tick !== previous.tick) {
      const mine = lockSent !== null;
      lockSent = null;
      if (mine || data.running) {
        // Hold All Act briefly, then repaint once the hold ends.
        lockResolvedTick = data.tick;
        actUntil = Date.now() + ACT_HOLD_MS;
        setTimeout(renderCommandAvailability, ACT_HOLD_MS + 20);
        // The tick's choreography starts now: falls wait for their Fall beat.
        if (data.tick !== null) {
          tickPlay = { tick: data.tick, at: Date.now() };
          played = true;
        }
      }
      const sentence = tickPlayedAnnouncement({
        terminal: data.terminal,
        mine,
        watchMode: data.watchMode,
        recording: data.recording,
      });
      if (sentence !== null) say(sentence, "info", true);
    } else if (data.focusId !== previous.focus && data.focusId !== null) {
      const facts = championFacts(data.focusedScene ?? data.focusedRoster);
      if (facts !== null) {
        say(
          focusedManual
            ? `Giving orders to your ${facts.name}.`
            : `Looking at the ${facts.nameWithId}, played by ${
                controllerOfChampion(data.controllers, {
                  publicAgentId: data.focusId,
                  slot: authorizedOracleCommandSlotForPublicAgentId(
                    data.presentation,
                    data.focusId,
                  ),
                })?.controller.name ?? "a policy"
              }.`,
          "info",
          true,
        );
      }
    } else if (previous.running && !data.running && data.watchMode && !data.terminal) {
      // At game over Run stops by itself; the game-over sentence stays.
      say("Paused after the current turn.", "info", true);
    }
    // At game over the step follows the replay's save state ("Saving the
    // replay…", then saved), but never replaces a newer sentence.
    const overStep = data.terminal ? gameOverStep(data.recording) : null;
    if (
      overStep !== null &&
      previous.gameOverStep !== null &&
      overStep !== previous.gameOverStep &&
      statusMessage.text === previous.gameOverStep
    ) {
      say(overStep, "info", true);
    }
    previous.gameOverStep = overStep;
    previous.tick = data.tick;
    previous.focus = data.focusId;
    previous.running = data.running;
    previous.stopped = data.stopped;
    return played;
  }

  /**
   * Write the staged order of the champion in view as one sentence, for the
   * hidden legacy commit title.
   *
   * @param {unknown} _presentation The installed presentation (read through
   *   state; kept for the call site's clarity).
   * @returns {string}
   */
  function stagedOrderSentence(_presentation) {
    const data = deckData();
    if (!data.live || data.focusId === null) return "No order to lock in yet.";
    return orderFor(data, data.focusId);
  }

  /**
   * Return one Manual champion's order sentence (orderSentence) from the
   * same source manualRow uses: the audience's inspection when it is in
   * view, else the server's staged order in Oracle View, else the deck's
   * memory of this tick.
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @param {string} publicId The champion's public ID.
   * @returns {string}
   */
  function orderFor(data, publicId) {
    const row = manualRow(data, publicId);
    return orderSentence({
      touched: row.touched,
      moveAction: row.moveAction,
      lane: row.lane,
      classId: row.classId,
      ultimateNeedsNoTarget: ultimateNeedsNoTarget(
        data.mechanicsOf(row.classId),
        row.classId,
      ),
      targetName: row.targetName,
      fallen: row.fallen,
    });
  }

  /**
   * Return the identity facts (class, calling, team, Agent ID; life only for
   * a body the audience's scene draws) of one champion, from the audience's
   * scene or, for a body that scene does not draw, the static roster. Reads
   * no staged order. Pure.
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @param {string} publicId The champion's public ID.
   * @returns {ReturnType<typeof championFacts>}
   */
  function manualChampionFacts(data, publicId) {
    return championFacts(
      data.sceneAgents.find((agent) => agent.public_agent_id === publicId) ??
        data.rosterAgents.find((agent) => agent?.public_agent_id === publicId),
    );
  }

  /**
   * Return what the deck knows about one champion you play: its class, its
   * order, whether you changed it and whether it is fallen. The order comes
   * from the audience's inspection when the champion is in view; else from
   * the server's staged order (data.servedOrders: every Manual champion in
   * Oracle View, the viewing team's own in Agent POV); else from this page's
   * memory of this tick. Callers pass only data.shownManualIds, never another
   * team's champion in Agent POV. "Set" (touched) is server truth (owner, 3
   * Oct 2026): the order the server holds is not the default (Stay, No
   * Combat), so a freshly loaded page and both views count the same; Stay +
   * No Combat reads Default even when chosen on purpose, and a target kept
   * with No Combat is still the default order (the server sends no target
   * for it). `fallen` is set once its fall has reached its Fall beat
   * (data.fallHolds); a fallen champion takes no orders, so callers never
   * count it as Set or Default.
   *
   * Agent POV: a target is named by class and ID only ("Mage (ID 5)"), never
   * with life or position.
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @param {string} publicId The champion's public ID.
   * @returns {{publicId: string, facts: ReturnType<typeof championFacts>,
   *   classId: unknown, moveAction: number, lane: "none" | "basic" |
   *   "ultimate", targetName: string | null, targetShort: string | null,
   *   touched: boolean, known: boolean, fallen: {className: string,
   *   returnsIn: number | null} | null}}
   */
  function manualRow(data, publicId) {
    const scene = data.sceneAgents.find((agent) => agent.public_agent_id === publicId);
    const roster = data.rosterAgents.find(
      (agent) => agent?.public_agent_id === publicId,
    );
    const facts = championFacts(scene ?? roster);
    const classId = (scene ?? roster)?.class_id;
    const entry = editOf(publicId, data.tick);
    /** @param {string | null} targetId */
    const targetNames = (targetId) => {
      if (targetId === null) return { name: null, short: null };
      const target =
        data.sceneAgents.find((agent) => agent.public_agent_id === targetId) ??
        data.rosterAgents.find((agent) => agent?.public_agent_id === targetId);
      const targetFacts = championFacts(
        target === undefined
          ? null
          : {
              class_id: target.class_id,
              team_id: target.team_id,
              public_agent_id: target.public_agent_id,
              display_agent_id: target.display_agent_id,
            },
      );
      return targetFacts === null
        ? { name: null, short: null }
        : {
            name: targetFacts.nameWithId,
            short: `${targetFacts.name} ${targetFacts.number}`,
          };
    };
    // A fall waits for its Fall beat (data.fallHolds), as the arena does.
    const fallen =
      scene !== undefined &&
      scene.life_state !== "alive" &&
      !data.fallHolds.has(publicId)
        ? {
            className: facts?.name ?? "Champion",
            returnsIn: data.waveIn(scene.team_id),
          }
        : null;
    if (publicId === data.viewId && data.viewDraft !== null) {
      const picked = pickedTarget(data, true);
      const moveAction = Number.isInteger(data.viewDraft.move_action)
        ? Number(data.viewDraft.move_action)
        : 0;
      const lane = effectiveLane(data.viewDraft);
      entry.draft = { moveAction, lane, targetPublicId: picked.publicId };
      return {
        publicId,
        facts,
        classId,
        moveAction,
        lane,
        targetName: picked.name,
        targetShort: picked.short,
        // Set is server truth: the server's order is not Stay + No Combat.
        touched: orderIsSet({ moveAction, lane }),
        known: true,
        fallen,
      };
    }
    // The server's staged order (both views: in Agent POV the viewing team's
    // own, ruling P1), so a fresh page and a page that switched views agree
    // and orders kept from earlier ticks show. This page's memory of the
    // tick is used only when the server sent no row.
    const served = data.servedOrders.get(publicId) ?? null;
    const draft = served ?? entry.draft;
    const names = targetNames(draft?.targetPublicId ?? null);
    return {
      publicId,
      facts,
      classId,
      moveAction: draft?.moveAction ?? 0,
      lane: draft?.lane ?? "none",
      targetName: names.name,
      targetShort: names.short,
      // Set is server truth: the server's order is not Stay + No Combat.
      touched: orderIsSet(draft),
      known:
        served !== null ||
        draft !== null ||
        (!entry.touched && entry.carried?.targetPublicId == null),
      fallen,
    };
  }

  /**
   * Write the How To Play card's lines for this game's roster mix
   * (howToPlayLines), so the card never says "Policies play the rest" when
   * you play every champion, or asks for keys that do nothing in Policies
   * Only. With one Manual champion it writes the vision page's own lines.
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @returns {void}
   */
  function paintHowToPlay(data) {
    const list = deckParts?.howtoCard?.querySelector("ol") ?? null;
    if (!list) return;
    const lines = howToPlayLines({
      manualCount: data.manualIds.length,
      policyCount: Math.max(0, data.rosterRows.length - data.manualIds.length),
    });
    setHtml(list, lines.map((line) => `<li><span>${line}</span></li>`).join(""));
  }

  /**
   * Repaint the deck, the overlays, the status line and the rail cards from
   * the installed frame. Runs at the end of renderCommandAvailability and
   * renderSystemOperation; does nothing before bindCommandDeck. In the Replay
   * Viewer (or with no live frame) the rail cards and the status line are
   * hidden, except that after an accepted Exit the status line keeps its
   * closed sentence and seal. Once Exit was accepted the title and the pill
   * read "Closed" and no phase is current.
   *
   * Agent POV: life, reach, cooldowns and Allowed Now come only from the
   * audience's scene and inspection; a champion that scene does not draw
   * shows "Out Of Sight" and no life. Your own team's staged orders show in
   * both views (ruling P1); another team's never in Agent POV (L18).
   *
   * A champion that fell on the tick now playing shows as it was before the
   * tick until its Fall beat (fallHoldsOf: alive, with its life before the
   * tick); one repaint is scheduled for each Fall beat time of the tick.
   *
   * Side effects: DOM text, markup, `hidden`, classes, `data-*` and `aria-*`
   * changes in the deck, the status line and the rail cards (including the
   * deck's data-orders, which sizes the Lock In list in deck.css, and the How
   * To Play card's lines); moves the `.beacon` mark; toggles `.keys-on` on
   * #arena through paintKeys; may set one timer (the Fall beat repaint).
   *
   * @returns {void}
   */
  function renderTacticsTurn() {
    if (deckParts === null) return;
    const parts = deckParts;
    let data = deckData();
    // A turn that has just arrived holds its falls from its first paint.
    if (trackChanges(data)) data = deckData();
    if (data.fallHolds.size > 0 && fallRepaintTick !== data.tick) {
      fallRepaintTick = data.tick;
      // One repaint per Fall beat time (falls are 60 ms apart in the beat).
      for (const until of new Set(
        [...data.fallHolds.values()].map((hold) => hold.until),
      )) {
        setTimeout(renderCommandAvailability, Math.max(0, until - Date.now()) + 20);
      }
    }
    for (const card of [parts.turnCard, parts.classCard]) {
      if (card) card.hidden = !data.live;
    }
    // After an accepted live Exit the status line keeps the closed sentence
    // and the seal, even when the Exit answer left no installed frame.
    const closedLive = data.closed && !isReplayMode();
    if (parts.statusLine) parts.statusLine.hidden = !data.live && !closedLive;
    if (parts.howtoCard) parts.howtoCard.hidden = !data.live || guideMode !== "guide";
    if (!data.live) {
      if (closedLive) renderSeal(data);
      paintMessage();
      return;
    }
    paintHowToPlay(data);
    const phase = currentPhase();
    const focusedManual =
      data.focusId !== null && data.manualIds.includes(data.focusId);
    const deckState = data.watchMode
      ? "watching"
      : data.stopped
        ? "stopped"
        : !focusedManual
          ? "watchonly"
          : "play";
    parts.deck.dataset.state = deckState;
    if (parts.watchOnly) parts.watchOnly.hidden = deckState !== "watchonly";
    if (parts.stopped) parts.stopped.hidden = deckState !== "stopped";

    // Your Champion.
    const champion = data.focusedScene ?? data.focusedRoster;
    const facts = championFacts(champion);
    const seen = data.focusedScene !== null;
    const classId = champion?.class_id;
    const mechanics = data.mechanicsOf(classId);
    const cooldown = Number.isInteger(data.focusedScene?.ultimate_cooldown_remaining)
      ? Number(data.focusedScene?.ultimate_cooldown_remaining)
      : null;
    const fallen = fallenOf(data);
    const tokenExtra = {
      lit: runeRingLit(cooldown, mechanics?.ultimate_cooldown_steps),
      ready: cooldown === 0 && fallen === null,
      faded: fallen !== null,
    };
    const tokenM = championToken(facts, "M", tokenExtra, {
      agent: sceneAgentById(data, data.focusId),
      scene: data.scene,
    });
    setHtml(parts.agTok, tokenM);
    setText(parts.agName, facts?.name ?? "");
    setText(parts.agEpi, facts?.calling ?? "");
    setHtml(
      parts.agMeta,
      facts === null || facts.team === null
        ? ""
        : `${crestSvg(facts.team, 14)}Agent ID ${escapeHtml(facts.number)}`,
    );
    // Who plays the champion in view, from the one controller model; the
    // name is cut with "…" past the fixed name width, and hovering it while
    // cut drops down its team's controller list (controller-names.js).
    const focusedPlayer = controllerOfChampion(data.controllers, {
      publicAgentId: data.focusId,
      slot: data.live
        ? authorizedOracleCommandSlotForPublicAgentId(data.presentation, data.focusId)
        : null,
    });
    const controllerLabel =
      focusedPlayer === null || focusedPlayer.controller.you
        ? "A Policy"
        : focusedPlayer.controller.name;
    const controllerHtml = controllerNameHtml(
      controllerLabel,
      focusedPlayer?.team.teamId ?? null,
    );
    // The Class Guide note ends its first sentence with the name.
    const controllerSentenceHtml = controllerNameHtml(
      controllerLabel,
      focusedPlayer?.team.teamId ?? null,
      ".",
    );
    if (parts.agChip) {
      parts.agChip.className = `chip ${focusedManual ? "you" : "pol"}`;
      setHtml(
        parts.agChip,
        facts === null
          ? ""
          : focusedManual
            ? escapeHtml(DECK_LABELS.youControl)
            : `Played By&nbsp;${controllerHtml}`,
      );
    }
    // The champion in view's life: the shared Life bar (owner, 2 October
    // 2026), or words when it has fallen or is out of sight. The Watch Only
    // overlay shows the same.
    const lifeWords =
      facts === null
        ? ""
        : fallen !== null
          ? fallenWords(fallen)
          : seen && facts.life !== null
            ? null
            : "Out Of Sight";
    // Until its Fall beat a falling champion keeps its life from before the
    // tick, as the scorebug portraits do.
    const held = data.focusId === null ? undefined : data.fallHolds.get(data.focusId);
    const heldMaximum = Number(
      data.focusedScene?.maximum_health ?? data.focusedScene?.max_health,
    );
    const lifeModel =
      held !== undefined &&
      held.before !== null &&
      Number.isFinite(heldMaximum) &&
      heldMaximum > 0
        ? lifeBarModel(held.before, heldMaximum)
        : (facts?.life ?? null);
    paintLife(parts.agLife, lifeWords, lifeModel);

    // 1 Step.
    const draft = data.draft;
    const moveAction = Number.isInteger(draft?.move_action)
      ? Number(draft?.move_action)
      : 0;
    for (const key of parts.pad?.querySelectorAll(".key") ?? []) {
      const step = STEP_PAD_KEYS[/** @type {HTMLElement} */ (key).dataset.key ?? ""];
      const pressed = focusedManual && step?.moveAction === moveAction;
      key.setAttribute("aria-pressed", String(pressed));
    }
    setHtml(parts.padNote, `<b>${escapeHtml(MOVE_NAMES[moveAction] ?? "Stay")}</b>`);

    // 2 Target.
    const picked = pickedTarget(data);
    const noTarget = ultimateNeedsNoTarget(mechanics, classId);
    const lane = effectiveLane(draft);
    const clearButton =
      '<button type="button" class="xbtn" data-key="Escape" id="clear-target-button" aria-label="Clear target (Esc)" title="Clear Target · Esc">×<small>Esc</small></button>';
    if (parts.tchip) {
      if (picked.chip.state === "none") {
        parts.tchip.className = "tchip none";
        setHtml(
          parts.tchip,
          `<div><b>${DECK_LABELS.noTargetYet}</b>${
            noTarget && lane === "ultimate"
              ? `<span>${escapeHtml(classNamesV1(classId)?.ultimateName ?? "Ultimate")} needs no target.</span>`
              : "<span>Shift + click a body, or use a button below.</span>"
          }</div>`,
        );
      } else if (picked.chip.state === "out_of_sight") {
        parts.tchip.className = "tchip";
        setHtml(
          parts.tchip,
          `${UNSEEN_TOKEN}<div><b>${escapeHtml(picked.chip.label)}</b><span>${DECK_LABELS.targetOutOfSight}</span></div>${clearButton}`,
        );
      } else {
        const targetFacts = picked.chip.facts;
        parts.tchip.className = "tchip";
        // The target's life is the shared Life bar, under its name.
        setHtml(
          parts.tchip,
          `${championToken(
            targetFacts,
            "XS",
            {},
            {
              agent: sceneAgentById(data, picked.publicId),
              scene: data.scene,
            },
          )}<div><b>${escapeHtml(targetFacts?.label ?? "Champion")}</b><div class="tchip-life"></div></div>${clearButton}`,
        );
        paintLife(
          parts.tchip.querySelector(".tchip-life"),
          targetFacts?.life ? null : "",
          targetFacts?.life ?? null,
        );
      }
    }
    const names = classNamesV1(classId);
    const allowed = allowedNowTargets(data.mask, data.sceneAgents).filter(
      (row) => row.basic || (!noTarget && row.ultimate),
    );
    const stagedId = picked.publicId;
    // The vision's pills (name and abilities) while they fit; fitAllowedNow
    // below sets data-count and switches three or more that would scroll to
    // the two-column grid of short buttons (deck.css). Each button names its
    // abilities in its title.
    setHtml(
      parts.picks,
      allowed.length === 0
        ? `<span class="none">${DECK_LABELS.allowedNowEmpty}</span>`
        : allowed
            .map((row) => {
              const what = [
                row.basic ? (names?.basicName ?? "Basic") : null,
                row.ultimate && !noTarget ? (names?.ultimateName ?? "Ultimate") : null,
              ]
                .filter(Boolean)
                .join(" · ");
              const rowFacts = championFacts(
                data.sceneAgents.find(
                  (agent) => agent.public_agent_id === row.publicAgentId,
                ),
              );
              const title = ` title="${escapeHtml(`${row.name} · ID ${row.number}: ${what}`)}"`;
              return `<button type="button" class="pick" data-target="${escapeHtml(row.publicAgentId)}" aria-pressed="${row.publicAgentId === stagedId}" aria-label="${escapeHtml(`${row.name}, Agent ID ${row.number}, Team ${row.team ?? ""}: ${what}`)}"${title}>${championToken(rowFacts, "XS", {}, { agent: sceneAgentById(data, row.publicAgentId), scene: data.scene })}${escapeHtml(`${row.name} · ID ${row.number}`)} <em>${escapeHtml(what)}</em></button>`;
            })
            .join(""),
    );
    for (const pick of parts.picks?.querySelectorAll(".pick") ?? []) {
      /** @type {HTMLButtonElement} */ (pick).disabled = picksDisabled;
    }
    fitAllowedNow();
    parts.picks?.setAttribute("aria-disabled", String(picksDisabled));

    // 3 Action.
    const edit = data.focusId === null ? null : editOf(data.focusId, data.tick);
    const origin = draft?.arm_origin;
    const autoArmed =
      lane === "basic" &&
      (origin === "automatic" ||
        (origin === undefined && edit !== null && !edit.explicitBasic));
    const slots = actionSlotViews({
      classId,
      mechanics,
      cooldown,
      lane,
      autoArmed,
      basicOk: picked.basicOk,
      ultimateOk: noTarget ? picked.noTargetUltimateOk : picked.ultimateOk,
      terminal: data.terminal,
    });
    const icons =
      /** @type {Record<string, {basic: string, ultimate: string, basicGlyph: string, ultimateGlyph: string}>} */ (
        ABILITY_ICONS
      )[facts?.classKey ?? ""] ?? null;
    for (const [view, button] of /** @type {const} */ ([
      [slots[0], elements.noCombatButton],
      [slots[1], elements.basicButton],
      [slots[2], elements.ultimateButton],
    ])) {
      if (!(button instanceof HTMLElement)) continue;
      button.classList.toggle("armed", focusedManual && view.armed);
      button.classList.toggle("no", focusedManual && view.no);
      button.setAttribute("aria-pressed", String(focusedManual && view.armed));
      // As on the vision page, a Not Allowed slot stays pressable: a press
      // shows the refusal sentence and shakes the slot, so it is not marked
      // aria-disabled (its tag says Not Allowed in its accessible name).
      button.removeAttribute("aria-disabled");
      const tag = button.querySelector(".tag");
      if (tag instanceof HTMLElement) {
        const shown = focusedManual ? view.tag : null;
        tag.hidden = shown === null;
        setText(tag, shown ?? "");
      }
    }
    setText(parts.basicName, slots[1].name);
    setText(parts.basicSub, slots[1].sub);
    setText(parts.ultName, slots[2].name);
    setText(parts.ultSub, slots[2].sub);
    setText(parts.ultReady, slots[2].ready ?? "");
    if (icons) {
      setAbilityIcon(parts.basicIcon, icons.basic, icons.basicGlyph);
      setAbilityIcon(parts.ultIcon, icons.ultimate, icons.ultimateGlyph);
    }
    if (parts.runeRing) {
      const litPaint = paintAttributes("stroke", facts?.classKey);
      const lines = RUNE_ORDER.map((degrees, index) => {
        const [x1, y1] = ringPoint(24, 18, degrees);
        const [x2, y2] = ringPoint(24, 22, degrees);
        return `<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" ${index < slots[2].lit ? litPaint : 'stroke="#2A3448"'} stroke-width="2.4" stroke-linecap="round"/>`;
      }).join("");
      const dots =
        cooldown === 0
          ? [0, 90, 180, 270]
              .map((degrees) => {
                const [x, y] = ringPoint(24, 20, degrees);
                return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="1.3" fill="${MOON}"/>`;
              })
              .join("")
          : "";
      setHtml(parts.runeRing, cooldown === null ? "" : lines + dots);
    }

    // 4 Lock In. Agent POV lists only the viewing team's orders (L18). The
    // list's size and form follow the roster (every Manual champion of both
    // teams), not the view, so neither a turn nor a view switch changes the
    // arena's size: data-orders sizes the list (deck.css). One or two Manual
    // champions: the vision's full sentences ("Your order: ..." when this
    // view shows one). Three or more: one short row each
    // (compactOrderParts), the full sentence in its title, so every line
    // shows.
    const manualRows = data.shownManualIds.map((publicId) => manualRow(data, publicId));
    const shownCount = data.shownManualIds.length;
    const rosterCount = data.manualIds.length;
    parts.deck.dataset.orders =
      rosterCount <= 1 ? "one" : rosterCount === 2 ? "two" : "many";
    if (rosterCount > 2 && shownCount > 0) {
      const rowsHtml = manualRows
        .map((row) => {
          const short = compactOrderParts({
            touched: row.touched,
            moveAction: row.moveAction,
            lane: row.lane,
            classId: row.classId,
            ultimateNeedsNoTarget: ultimateNeedsNoTarget(
              data.mechanicsOf(row.classId),
              row.classId,
            ),
            targetShort: row.targetShort,
            fallen: row.fallen,
          });
          // Spaces between the parts keep the row's text whole; the row
          // stays whole on one line (fitOrderRow): the step gives way
          // first, then the text steps down; the title keeps the step.
          return `<p class="order-row" title="${escapeHtml(orderFor(data, row.publicId))}"><b>${row.facts?.team ? crestSvg(row.facts.team, 14) : ""}${escapeHtml(row.facts?.name ?? "Champion")}:</b> <span class="order-main">${escapeHtml(short.main)}</span>${
            short.target === null
              ? ""
              : ` <span class="order-target">→ ${escapeHtml(short.target)}</span>`
          }${short.tail === null ? "" : ` <span class="order-tail">· ${escapeHtml(short.tail)}</span>`}</p>`;
        })
        .join("");
      setHtml(parts.orderLine, rowsHtml);
      fitShortOrderRows(rowsHtml);
    } else if (shownCount === 2) {
      setHtml(
        parts.orderLine,
        manualRows
          .map(
            (row) =>
              `<p><b>${row.facts?.team ? crestSvg(row.facts.team, 14) : ""}${escapeHtml(row.facts?.name ?? "Champion")}:</b> ${escapeHtml(orderFor(data, row.publicId))}</p>`,
          )
          .join(""),
      );
    } else if (data.shownManualIds.length === 1) {
      setHtml(
        parts.orderLine,
        `<p><b>Your order:</b> ${escapeHtml(orderFor(data, data.shownManualIds[0]))}</p>`,
      );
    } else {
      setHtml(parts.orderLine, "");
    }
    // Full sentences wrap and are never fitted: forget the last row fit.
    if (!(rosterCount > 2 && shownCount > 0)) fitShortOrderRows("");
    const titles = turnTitles({
      phase,
      stopped: data.stopped,
      terminal: data.terminal,
      watchMode: data.watchMode,
      running: data.running,
      closed: data.closed,
    });
    if (!liveScriptedInspectionOnly()) {
      setText(parts.lockLabel, titles.lockLabel);
      elements.submitTurnButton.setAttribute("aria-label", titles.lockLabel);
    }

    // Watch Only and Watching.
    const lastFacts = championFacts(
      data.sceneAgents.find((agent) => agent.public_agent_id === lastManualId) ??
        data.rosterAgents.find((agent) => agent?.public_agent_id === lastManualId),
    );
    const lastName = lastFacts?.name ?? "Champion";
    if (deckState === "watchonly") {
      setHtml(parts.woTok, tokenM);
      setText(parts.woName, facts === null ? "" : `${facts.name} · ${facts.calling}`);
      setHtml(
        parts.woMeta,
        facts === null || facts.team === null
          ? ""
          : `${crestSvg(facts.team, 14)}Agent ID ${escapeHtml(facts.number)} · Played By ${controllerHtml}`,
      );
      paintLife(parts.woLife, lifeWords, facts?.life ?? null);
      setText(parts.woBack, `Back To Your ${lastName}`);
    }
    // main.js disables every deck button while no authority is installed and
    // only data-key buttons are re-enabled by the availability pass, so the
    // overlay buttons are re-enabled here.
    if (parts.woBack instanceof HTMLButtonElement)
      parts.woBack.disabled = lastManualId === null;
    const stoppedSave = document.getElementById("stopped-save");
    if (stoppedSave instanceof HTMLButtonElement)
      stoppedSave.disabled = deckState !== "stopped";
    if (deckState === "watching") {
      setHtml(parts.wbTok, tokenM);
      setText(
        parts.wbState,
        data.terminal
          ? DECK_LABELS.gameOver
          : data.running
            ? phase === "choosing"
              ? "Running · Policies Choosing"
              : phase === "act"
                ? "Running · All Act"
                : "Running"
            : phase === "plan" || phase === "sent"
              ? "Paused"
              : phase === "choosing"
                ? DECK_LABELS.policiesChoosing
                : "All Act",
      );
      setText(
        parts.wbNote,
        `${facts === null ? "" : `Looking at the ${facts.nameWithId}. `}${data.terminal ? "The game is over." : "Every match starts paused. Run plays one turn at a time. Pause stops after the current turn."}`,
      );
    }

    // Status line.
    renderSeal(data);
    paintMessage();

    // This Turn, Class Guide and the beacon.
    renderThisTurn(data, {
      phase,
      titles,
      focusedManual,
      fallen,
      lane,
      picked,
      noTarget,
      classId,
      lastName,
      manualRows,
      allowedCount: allowed.length,
    });
    renderClassGuide(data, {
      classId,
      mechanics,
      cooldown,
      focusedManual,
      controllerHtml: controllerSentenceHtml,
    });
  }

  /**
   * Paint the recording seal with recordingWords (the one owner of the
   * Recording words, shared with main.js's pill and popover): "Recording ·
   * Tick 91 of 300" while recording, "Saving Replay…" while the server saves
   * a finished game (lifecycle "sealed" or "finalized_unsaved"), "Saved · 91
   * Ticks" once saved, "Save Failed · Retry" after a failed save and, once
   * Exit closed a game before its first tick, "No Replay"; hidden when the
   * match records nothing. It reads data.recording, so after an accepted
   * Exit it shows the Exit answer's own status, as the pill does.
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @returns {void}
   */
  function renderSeal(data) {
    const parts = deckParts;
    if (!parts?.recSeal) return;
    const recording = data.recording;
    parts.recSeal.hidden = recording === null;
    if (recording === null) return;
    parts.recSeal.classList.toggle(
      "saved",
      recording.lifecycle === "saved" || recording.lifecycle === "reviewing",
    );
    setText(
      parts.recSealText,
      recordingWords(recording, data.tick ?? 0, data.limit, data.closed).seal,
    );
  }

  /**
   * Repaint the This Turn card: tick, title, set pill, Guide / Hints / Off,
   * Do This Now (hidden at Off), the four phases, the checklist and the
   * Policies block; then move the Guide level's beacon. The set pill's
   * "N Of M Set" and Do This Now's untouched-champion hint count only
   * view.manualRows (data.shownManualIds: the viewing team's Manual
   * champions in Agent POV) that are not fallen: a fallen champion takes no
   * orders, so it is never counted or asked for orders. When every one of
   * them is fallen the pill reads "Fallen" (one champion) or "All Fallen".
   * Set means the server holds an order that is not the default (manualRow),
   * so the pill is the same after a reload and in both views.
   * At game over the pill reads "Game Over" and every phase is marked done
   * (none is current), since no turn is left to plan. Once Exit was accepted
   * (data.closed) the pill reads "Closed" and no phase is current or done.
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @param {{phase: "plan" | "sent" | "choosing" | "act", titles: {title:
   *   string, lockLabel: string}, focusedManual: boolean, fallen: {className:
   *   string, returnsIn: number | null} | null, lane: "none" | "basic" |
   *   "ultimate", picked: ReturnType<typeof pickedTarget>, noTarget: boolean,
   *   classId: unknown, lastName: string, manualRows:
   *   ReturnType<typeof manualRow>[], allowedCount: number}} view Facts the
   *   deck already worked out. allowedCount is how many Allowed Now buttons
   *   show for the champion in view.
   * @returns {void}
   */
  function renderThisTurn(data, view) {
    const parts = deckParts;
    if (!parts) return;
    setText(
      parts.turnEyebrow,
      `Tick ${data.tick ?? 0}${data.limit === null ? "" : ` Of ${data.limit}`}`,
    );
    setText(parts.turnTitle, view.titles.title);
    const orderable = view.manualRows.filter((row) => row.fallen === null);
    if (parts.setPill) {
      parts.setPill.className = data.watchMode ? "setpill watch" : "setpill";
      setText(
        parts.setPill,
        data.closed
          ? DECK_LABELS.closed
          : data.terminal
            ? "Game Over"
            : data.watchMode
              ? DECK_LABELS.policiesOnly
              : orderable.length === 0 && view.manualRows.length > 0
                ? view.manualRows.length === 1
                  ? "Fallen"
                  : "All Fallen"
                : `${orderable.filter((row) => row.touched).length} Of ${orderable.length} Set`,
      );
    }
    for (const button of parts.guideLevel?.querySelectorAll("button[data-level]") ??
      []) {
      button.setAttribute(
        "aria-pressed",
        String(/** @type {HTMLElement} */ (button).dataset.level === guideMode),
      );
    }
    const untouched = orderable.find(
      (row) => row.publicId !== data.focusId && !row.touched,
    );
    const next = doThisNow({
      closed: data.closed,
      stopped: data.stopped,
      terminal: data.terminal,
      phase: view.phase,
      watchMode: data.watchMode,
      running: data.running,
      focusedManual: view.focusedManual,
      lastManualName: view.lastName,
      fallen: view.fallen,
      targetSet: view.picked.chip.state !== "none",
      lane: view.lane,
      ultimateOkOnTarget: view.picked.ultimateOk,
      ultimateNeedsNoTarget: view.noTarget,
      ultimateName: classNamesV1(view.classId)?.ultimateName ?? "Ultimate",
      untouchedOther:
        untouched === undefined
          ? null
          : {
              publicAgentId: untouched.publicId,
              className: untouched.facts?.name ?? "Champion",
            },
      allowedCount: view.allowedCount,
      noTargetUltimateOk: view.picked.noTargetUltimateOk,
      recording: data.recording,
    });
    if (parts.doThis) parts.doThis.hidden = guideMode === "off";
    setText(parts.doThisText, next.text);
    // At game over no phase is current: every turn has been played. Once
    // Exit was accepted none is current or done: no turn is played any more.
    const index = data.closed
      ? -1
      : data.terminal
        ? TURN_PHASES.length
        : { plan: 0, sent: 1, choosing: 2, act: 3 }[view.phase];
    const labels = data.watchMode ? WATCH_PHASES : TURN_PHASES;
    parts.phases?.querySelectorAll("li").forEach((item, position) => {
      setText(item, labels[position] ?? "");
      item.className = position === index ? "on" : position < index ? "done" : "";
    });
    renderChecklist(data, view);
    for (const marked of document.querySelectorAll(".beacon")) {
      marked.classList.remove("beacon");
    }
    if (guideMode === "guide" && next.beacon) {
      const target = document.querySelector(next.beacon);
      if (target instanceof HTMLElement && target.offsetParent !== null) {
        target.classList.add("beacon");
      }
    }
  }

  /**
   * Repaint the checklist: one row per champion you play (data.manualIds), by
   * team. A row whose order this view may show (view.manualRows, built from
   * data.shownManualIds) has its step, its action and Set or Default, or,
   * once its fall has reached its Fall beat, "Fallen · Returns In N Ticks"
   * with a Fallen tag and data-fallen="true" (it takes no orders); any
   * other row (Agent POV: another team's Manual champion, ruling L18) has
   * data-observed="false", no order and no Set or Default, yet a click on it
   * still gives that champion orders. When this view cannot see it
   * (data-sight="out") it shows only the champion's class and calling, the
   * unseen token and "Out Of Sight" (owner, 3 Oct 2026); when this view sees
   * it (data-sight="in") its token from the audience's scene and "In Sight ·
   * Order Hidden". Its title says why (HIDDEN_ORDER_TITLE). Then the Policies block: every
   * policy of both teams from the one controller model (policyListRows of
   * data.controllers: team crest, full name that wraps and is never cut,
   * "· 4 Champions"; exactly the hover lists' rows without "You"), a note
   * and Waiting, Choosing… or Done. With nobody Manual it says Nobody Is
   * Yours. Also publishes data.controllers for the hover lists
   * (controller-names.js).
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @param {{phase: "plan" | "sent" | "choosing" | "act", manualRows:
   *   ReturnType<typeof manualRow>[]}} view Facts the deck already worked out.
   * @returns {void}
   */
  function renderChecklist(data, view) {
    const parts = deckParts;
    if (!parts?.checklist) return;
    let html = "";
    if (data.watchMode) {
      html +=
        '<p class="cl-head">Nobody Is Yours</p><p class="cl-note">Both teams are played by policies. Click any champion to look at it.</p>';
    } else {
      // Every Manual champion keeps its row (a click gives it orders); a row
      // whose order this view may not show (Agent POV: another team's,
      // ruling L18) has no order and no Set or Default. Out of this view's
      // sight it reads "Out Of Sight" (owner, 3 Oct 2026) with the unseen
      // token and nothing else; in sight it reads "In Sight · Order Hidden".
      const knownById = new Map(view.manualRows.map((row) => [row.publicId, row]));
      for (const team of /** @type {const} */ (["A", "B"])) {
        const ids = data.manualIds.filter(
          (publicId) => manualChampionFacts(data, publicId)?.team === team,
        );
        if (ids.length === 0) continue;
        html += `<p class="cl-head">${crestSvg(team, 16)}Team ${team}</p>`;
        for (const publicId of ids) {
          const row = knownById.get(publicId);
          if (row === undefined) {
            const facts = manualChampionFacts(data, publicId);
            const agent = sceneAgentById(data, publicId);
            html += `<button type="button" class="cl-row${publicId === data.focusId ? " focus" : ""}" data-slot="${escapeHtml(publicId)}" data-observed="false" data-sight="${agent === null ? "out" : "in"}" title="${escapeHtml(HIDDEN_ORDER_TITLE)}">${agent === null ? UNSEEN_TOKEN : championToken(facts, "XS", {}, { agent, scene: data.scene })}<span><b>${escapeHtml(`${facts?.name ?? "Champion"} · ${facts?.calling ?? ""}`)}</b><span>${escapeHtml(hiddenOrderLine(agent !== null))}</span></span></button>`;
            continue;
          }
          if (row.fallen) {
            // A fallen champion takes no orders: it says when it returns.
            html += `<button type="button" class="cl-row${row.publicId === data.focusId ? " focus" : ""}" data-slot="${escapeHtml(row.publicId)}" data-fallen="true">${championToken(row.facts, "XS", {}, { agent: sceneAgentById(data, row.publicId), scene: data.scene })}<span><b>${escapeHtml(`${row.facts?.name ?? "Champion"} · ${row.facts?.calling ?? ""}`)}</b><span>${escapeHtml(fallenWords(row.fallen))}</span></span><em class="cl-state fallen">Fallen</em></button>`;
            continue;
          }
          const line = row.known
            ? checklistLine({
                moveAction: row.moveAction,
                lane: row.lane,
                classId: row.classId,
                ultimateNeedsNoTarget: ultimateNeedsNoTarget(
                  data.mechanicsOf(row.classId),
                  row.classId,
                ),
                targetShort: row.targetShort,
              })
            : "Step: Stay · Action: Click To See";
          html += `<button type="button" class="cl-row${row.publicId === data.focusId ? " focus" : ""}" data-slot="${escapeHtml(row.publicId)}">${championToken(row.facts, "XS", {}, { agent: sceneAgentById(data, row.publicId), scene: data.scene })}<span><b>${escapeHtml(`${row.facts?.name ?? "Champion"} · ${row.facts?.calling ?? ""}`)}</b><span>${escapeHtml(line)}</span></span><em class="cl-state${row.touched ? "" : " def"}">${row.touched ? "Set" : "Default"}</em></button>`;
        }
      }
    }
    // Every policy of both teams from the one controller model: the same
    // names, counts and order as the scorebug's hover lists, without "You".
    // Names wrap and are never cut.
    if (parts.checklist.ownerDocument)
      publishControllerRoster(parts.checklist.ownerDocument, data.controllers);
    const busy = view.phase === "choosing";
    html += `<div class="cl-pol"><p class="cl-head">Policies</p>${policyListRows(
      data.controllers,
    )
      .map(
        (row) =>
          `<div class="row cl-pol__row">${crestSvg(row.teamId === 1 ? "A" : "B", 16)}<span class="cl-pol__name">${escapeHtml(row.name)}<span class="cl-pol__count"> · ${escapeHtml(row.countWords)}</span></span></div>`,
      )
      .join("")}<div class="row"><span class="note">${
      data.watchMode
        ? "They choose each turn you play."
        : "They choose after you lock in, and never see your orders."
    }</span><span class="st${busy ? " busy" : ""}">${busy ? "Choosing…" : view.phase === "act" ? "Done" : "Waiting"}</span></div></div>`;
    setHtml(parts.checklist, html);
  }

  /**
   * Repaint the Class Guide of the champion in view: calling, lore line,
   * stats, and its Basic and Ultimate with the host's numbers and, for a
   * champion you play, where each is allowed now (no such line after game
   * over, when nothing is allowed). Hidden while the Class Guide switch is
   * off.
   *
   * Agent POV: "Allowed on" lists come from the audience's own masks, so they
   * name only bodies the team may know.
   *
   * @param {ReturnType<typeof deckData>} data This repaint's facts.
   * @param {{classId: unknown, mechanics: Record<string, any> | null,
   *   cooldown: number | null, focusedManual: boolean, controllerHtml:
   *   string}} view Facts the deck already worked out. controllerHtml is the
   *   controller's name and the period that ends its sentence, as
   *   controllerNameHtml writes it (escaped, cut with "…" past the fixed
   *   name width).
   * @returns {void}
   */
  function renderClassGuide(data, view) {
    const parts = deckParts;
    if (!parts?.classCard) return;
    // The Class Guide switch (class_guide) off hides the card and skips its fill.
    const guideOn = classGuideEnabled();
    const guideCard = parts.classCard;
    if (guideCard) guideCard.hidden = !guideOn;
    if (!guideOn) return;
    if (classNamesV1(view.classId) === null) {
      setHtml(parts.classCard, classGuideCardHtml(document, view.classId, null));
      return;
    }
    // After game over nothing is allowed, so no "Allowed" line is shown.
    const mine = view.focusedManual && !data.stopped && !data.terminal;
    const pairs = asArray(data.mask?.target_use_ultimate_joint_mask);
    const listed = allowedNowTargets(data.mask, data.sceneAgents);
    /** @param {"basic" | "ultimate"} which */
    const allowedOn = (which) =>
      listed
        .filter((row) => (which === "basic" ? row.basic : row.ultimate))
        .map((row) => `${row.name} ${row.number}`)
        .join(", ");
    const basicList = allowedOn("basic");
    const basicNow = !mine
      ? ""
      : basicList
        ? `<span class="now ok">Allowed on: ${escapeHtml(basicList)}</span>`
        : '<span class="now no">Not allowed on anyone right now</span>';
    let ultimateNow = "";
    if (mine) {
      if (ultimateNeedsNoTarget(view.mechanics, view.classId)) {
        ultimateNow =
          asArray(pairs[0])[1] === true
            ? '<span class="now ok">Allowed now</span>'
            : `<span class="now no">Not allowed now${view.cooldown ? ` · Ready in ${ticksText(view.cooldown)}` : ""}</span>`;
      } else {
        const ultimateList = allowedOn("ultimate");
        ultimateNow = ultimateList
          ? `<span class="now ok">Allowed on: ${escapeHtml(ultimateList)}</span>`
          : '<span class="now no">Not allowed on anyone right now</span>';
      }
    }
    const note = !view.focusedManual
      ? `<p class="stats cg-note">Played by ${view.controllerHtml} ${data.watchMode ? "It chooses each turn" : "It chooses after you lock in"}, from what its own champions may see.</p>`
      : data.stopped
        ? '<p class="stats cg-note">You play this champion. The game is stopped, so its orders cannot change.</p>'
        : "";
    setHtml(
      parts.classCard,
      classGuideCardHtml(document, view.classId, view.mechanics, {
        cooldown: view.cooldown,
        basicNow,
        ultimateNow,
        note,
      }),
    );
  }

  /**
   * Repaint Keys On or Keys Off from the focused element, and outline the
   * arena (#arena.keys-on) while the battlefield has the keys. Once Exit was
   * accepted (state.shuttingDown) no key works any more: the chip reads
   * "Keys Off" (DECK_LABELS.keysClosed, with no invitation to click the
   * arena) and the arena has no keys-on frame, as main.js paintKeysState
   * paints it.
   *
   * @returns {void}
   */
  function paintKeys() {
    const parts = deckParts;
    if (!parts) return;
    const closed = state.shuttingDown === true;
    const on = !closed && document.activeElement?.id === "battlefield";
    parts.keysChip?.classList.toggle("off", !on);
    setText(
      parts.keysText,
      on ? DECK_LABELS.keysOn : closed ? DECK_LABELS.keysClosed : DECK_LABELS.keysOff,
    );
    parts.arena?.classList.toggle("keys-on", on);
  }

  /**
   * Send the focus to one champion you play (Back To Your Warrior, a checklist
   * row) through the same roster command a portrait click sends.
   *
   * @param {string | null} publicId The champion's public ID.
   * @returns {void}
   */
  function focusChampion(publicId) {
    if (publicId === null) return;
    const row = authorizedPresentationIdentityRows(state.presentation).find(
      (identity) => identity.public_agent_id === publicId,
    );
    const slot =
      row?.command_global_slot ??
      authorizedOracleCommandSlotForPublicAgentId(state.presentation, publicId);
    if (!Number.isInteger(slot)) return;
    dispatchCommandFromDraftControl({
      command_type: "roster_selection",
      role: "control",
      global_slot: slot,
    });
  }

  /**
   * Find the vision parts once and connect their listeners: the Guide / Hints
   * / Off buttons (stored per browser), Keys On or Off (focus listeners), the
   * deck's order-input refusals (a capture listener for keys on the
   * battlefield and Shift + click on a body), the mousedown guard that keeps
   * the keys on while a deck button is pressed, and delegated clicks for the
   * parts the deck writes (Allowed Now, × Esc, checklist rows, Back, Finish
   * And Save on Stopped). Full Guide (#full-guide) belongs to the help owner
   * in main.js, which opens How To Play.
   *
   * Side effects: listeners on the document and on deck parts; reads
   * localStorage. Missing parts are skipped, so a page without the vision
   * deck is left alone.
   *
   * @returns {void}
   */
  function installVisionDeck() {
    const deck = elements.commandDeck;
    if (!(deck instanceof HTMLElement) || deck.dataset.visionDeck === "bound") return;
    deck.dataset.visionDeck = "bound";
    /** @param {string} id */
    const part = (id) => document.getElementById(id);
    deckParts = {
      deck,
      agTok: part("ag-tok"),
      agName: part("ag-name"),
      agEpi: part("ag-epi"),
      agMeta: part("ag-meta"),
      agChip: part("ag-chip"),
      agLife: part("ag-life"),
      pad: part("pad"),
      padNote: part("pad-note"),
      tchip: part("tchip"),
      picks: part("command-target-select"),
      basicIcon: part("basic-icon"),
      basicName: part("basic-name"),
      basicSub: part("basic-sub"),
      ultIcon: part("ult-icon"),
      ultName: part("ult-name"),
      ultSub: part("ult-sub"),
      ultReady: part("ult-ready"),
      runeRing: part("rune-ring"),
      orderLine: part("order-line"),
      lockLabel: part("lock-label"),
      watchOnly: part("watch-only"),
      woTok: part("wo-tok"),
      woName: part("wo-name"),
      woMeta: part("wo-meta"),
      woLife: part("wo-life"),
      woBack: part("wo-back"),
      stopped: part("stopped"),
      wbTok: part("wb-tok"),
      wbState: part("wb-state"),
      wbNote: part("wb-note"),
      watchSubmit: /** @type {HTMLButtonElement | null} */ (part("watch-submit")),
      statusLine: part("status-line"),
      recSeal: part("rec-seal"),
      recSealText: part("rec-seal-text"),
      msg: part("msg"),
      keysChip: part("keys-chip"),
      keysText: part("keys-text"),
      arena: part("arena"),
      turnCard: part("turn-card"),
      turnEyebrow: document.querySelector("#turn-card .rc-head .eyebrow"),
      turnTitle: part("turn-title"),
      setPill: part("set-pill"),
      guideLevel: part("guide-level"),
      doThis: part("do-this"),
      doThisText: part("do-this-text"),
      phases: part("phases"),
      checklist: part("checklist"),
      classCard: part("class-card"),
      howtoCard: part("howto-card"),
    };

    for (const button of deckParts.guideLevel?.querySelectorAll("button[data-level]") ??
      []) {
      button.addEventListener("click", () => {
        const level = /** @type {HTMLElement} */ (button).dataset.level;
        if (level !== "guide" && level !== "hints" && level !== "off") return;
        guideMode = level;
        writeGuideMode(level);
        renderTacticsTurn();
      });
    }

    document.addEventListener("focusin", paintKeys);
    document.addEventListener("focusout", () => queueMicrotask(paintKeys));
    paintKeys();

    // Order inputs on the battlefield: refuse before anything is sent.
    document.addEventListener(
      "keydown",
      (event) => {
        const target = /** @type {Element | null} */ (event.target);
        if (target?.id !== "battlefield" || isReplayMode()) return;
        if (event.ctrlKey || event.altKey || event.metaKey) return;
        if (event.shiftKey && event.key !== "Tab") return;
        const intent = intentForKey(event.key);
        if (intent === null) return;
        if (intent.kind === "lock" && event.repeat) return;
        const element =
          intent.kind === "lane"
            ? [elements.noCombatButton, elements.basicButton, elements.ultimateButton][
                intent.lane
              ]
            : intent.kind === "step"
              ? deckParts?.pad?.querySelector(
                  `.key[data-key="${event.key.toLowerCase()}"]`,
                )
              : null;
        if (!acceptIntent(intent, { element, viaKey: true })) {
          event.preventDefault();
          event.stopImmediatePropagation();
        }
      },
      true,
    );
    document.addEventListener(
      "pointerdown",
      (event) => {
        const target = /** @type {Element | null} */ (event.target);
        if (!event.shiftKey || event.button !== 0 || isReplayMode()) return;
        if (!target?.closest("#battlefield")) return;
        // Arena bodies carry their audience-scoped presentation key; the
        // deck turns it into a public ID through the audience's own scene.
        const body = target.closest(
          ".agent[data-presentation-key], [data-public-agent-id]",
        );
        const key = body?.getAttribute("data-presentation-key");
        const publicId =
          body?.getAttribute("data-public-agent-id") ??
          deckData().sceneAgents.find((agent) => agent.presentation_key === key)
            ?.public_agent_id;
        if (typeof publicId !== "string") return;
        if (!acceptIntent({ kind: "target", publicId })) {
          event.preventDefault();
          event.stopImmediatePropagation();
        }
      },
      true,
    );
    // Keep the keys on while a deck control is pressed with the mouse.
    document.addEventListener(
      "mousedown",
      (event) => {
        if (document.activeElement?.id !== "battlefield") return;
        const target = /** @type {Element | null} */ (event.target);
        if (
          target?.closest(
            ".deck button, .dstatus button, .por, .cl-row, #view-switch button, #view-select button, #guide-level button",
          )
        ) {
          event.preventDefault();
        }
      },
      true,
    );

    deck.addEventListener("click", (event) => {
      const target = /** @type {Element | null} */ (event.target);
      const pick = target?.closest(".pick");
      if (pick instanceof HTMLButtonElement && !pick.disabled) {
        const publicId = pick.dataset.target ?? "";
        if (!acceptIntent({ kind: "target", publicId }, { element: pick })) return;
        const slot = authorizedOracleCommandSlotForPublicAgentId(
          state.presentation,
          publicId,
        );
        const command = Number.isInteger(slot)
          ? targetSelectionCommand(String(slot))
          : null;
        if (command) dispatchCommandFromDraftControl(command, { restoreFocusTo: null });
        return;
      }
      const clear = target?.closest("#clear-target-button");
      if (clear instanceof HTMLButtonElement && !clear.disabled) {
        if (!acceptIntent({ kind: "clear" }, { element: clear })) return;
        dispatchCommandFromDraftControl(keyboardCommand("Escape"), {
          restoreFocusTo: null,
        });
        return;
      }
      if (target?.closest("#wo-back")) {
        focusChampion(lastManualId);
        return;
      }
      if (target?.closest("#stopped-save")) {
        void dispatchCommand({ command_type: "finish_and_review" });
      }
    });
    deckParts.checklist?.addEventListener("click", (event) => {
      const row = /** @type {Element | null} */ (event.target)?.closest(".cl-row");
      if (row instanceof HTMLElement) focusChampion(row.dataset.slot ?? null);
    });
  }

  /**
   * Connect the deck's controls to live commands. Call exactly once per page;
   * a second call would add every listener again.
   *
   * First it runs installVisionDeck (see there for its listeners). It
   * listens for window resizes and for the page's fonts to finish loading,
   * to fit the short Lock In rows and Allowed Now again (fitShortOrderRows,
   * fitAllowedNow). Then it
   * adds click listeners to Restart (`reset`), Run or Pause (the page's own
   * Run through context.toggleLiveRun when main.js gives it, else `run` or
   * `pause` with the installed session and run generation) and every deck
   * button with
   * `data-key` (the same keyboard command its key would send: the pad, the
   * three action slots, ‹ Prev and Next ›, Lock In Turn and Play One Turn).
   * A press the deck refuses (deckRefusal) sends nothing: the status line
   * shows the vision's sentence and the control shakes. Python still checks
   * every command.
   *
   * @returns {void}
   */
  function bindCommandDeck() {
    installVisionDeck();
    // The short Lock In rows are fitted to the column's width: fit them again
    // when the window size changes or the page's fonts finish loading.
    window.addEventListener("resize", () => {
      fitShortOrderRows(orderFit.html);
      fitAllowedNow();
    });
    void document.fonts?.ready.then(() => {
      fitShortOrderRows(orderFit.html, true);
      fitAllowedNow(true);
    });
    elements.resetButton.addEventListener("click", () => {
      dispatchCommand({ command_type: "reset" });
    });

    elements.liveRunButton.addEventListener("click", () => {
      if (elements.liveRunButton.disabled || isReplayMode() || !state.frame) return;
      const running = liveRunning(state);
      say(running ? "Pausing after the current turn." : "Running.", "info", true);
      if (context.toggleLiveRun) {
        context.toggleLiveRun();
        return;
      }
      void dispatchCommand(
        running
          ? {
              command_type: "pause",
              session_id: state.frame.session_id,
              run_generation: state.frame.run_generation,
            }
          : { command_type: "run" },
      );
    });

    if (elements.commandDeck) {
      const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
        elements.commandDeck.querySelectorAll("button[data-key]")
      );
      for (const button of buttons) {
        button.addEventListener("click", (event) => {
          const key = button.dataset.key ?? "";
          const intent = key === "Tab" ? null : intentForKey(key);
          if (intent !== null && !acceptIntent(intent, { element: button })) return;
          const restoreKeyboardFocus = event.detail === 0;
          dispatchCommandFromDraftControl(
            keyboardCommand(key, {
              shiftKey: button.dataset.shift === "true",
            }),
            {
              restoreFocusTo: restoreKeyboardFocus ? button : null,
            },
          );
        });
      }
    }
  }

  return Object.freeze({
    policyControlledActor,
    policyControllerBlocksActionEdit,
    renderCommandAvailability,
    renderSystemOperation,
    bindCommandDeck,
    showStatusMessage,
  });
}
