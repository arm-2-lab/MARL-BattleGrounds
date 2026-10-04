/**
 * @file Translate battlefield UI input into commands and recording notices.
 * Python remains the authority for command legality and recording lifecycle.
 * These browser helpers build requests and decide local focus/confirmation
 * behavior; they do not execute a simulator transition or persist a replay.
 *
 * Which live keys exist comes from the one command set (`command-set.js`): the
 * battlefield forwards only the keys that table sends to Python (N, one tick,
 * is among them; the arrows are not), opens How To Play on ?, and hands V and F to their
 * local handlers in the browser, never to Python. B does nothing in live play.
 * ? works at any time the battlefield has focus (also at game over and while a
 * command is out); R and G still reach the page at game over when the caller
 * allows them. A right click on the battlefield sends the secondary pointer
 * (Python clears the target), and the browser's own menu stays closed there.
 *
 * It also owns the status line's plain sentences for the player's own commands
 * (statusSentence): the vision page's words for steps, actions, targets,
 * focus, Lock In, views, Restart and refusals. They are pure text builders;
 * main.js decides when to show them. audienceSwitchKeepsTick decides when a
 * view switch shows the tick already played at its end instead of replaying
 * it, so orders stay open (main.js acts on it).
 *
 * nextOrderSlot decides where Tab and Shift+Tab (and the deck's ‹ Prev and
 * Next ›) give the orders in live play with Manual champions: only to the
 * champions you command that can take orders (living Manual champions; with
 * one Manual champion on each team, both), never to a policy's champion. It
 * is pure; main.js reads the facts and sends the focus change.
 */
import { forwardedFieldKeys, runLocalCommand } from "./command-set.js";
import {
  requiresSharedObs as isReactiveController,
  isTeamController,
} from "./system-controls.js";

// The keys the focused battlefield owns in live play: every key the command
// set forwards to Python from the field, plus ? for How To Play (local).
const GAME_KEYS = new Set([...forwardedFieldKeys("combat_debugger"), "?"]);

const RECORDING_LIFECYCLE_COMMANDS = new Set([
  "finish_and_review",
  "review_replay",
  "retry_save",
  "save_as",
  "confirm_discard_and_replace",
  "save_and_replace",
  "exit",
]);

const RECORDING_PRESENTATION_KEYS = new Set(["g", "?"]);

const RECORDING_SAVE_AS_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*\.marlbg-replay\.json$/u;

/**
 * @typedef {{
 *   shiftKey?: boolean,
 *   ctrlKey?: boolean,
 *   altKey?: boolean,
 *   metaKey?: boolean,
 * }} ModifierSource
 */

/**
 * Copy source's optional keyboard modifiers into snake-case boolean fields.
 * Missing values become false. Return a new mutable record without changing
 * source; this helper does not decide whether a shortcut belongs to the game.
 *
 * @param {ModifierSource} source
 * @returns {{
 *   shift_key: boolean,
 *   ctrl_key: boolean,
 *   alt_key: boolean,
 *   meta_key: boolean,
 * }}
 */
function modifierFields(source) {
  return {
    shift_key: Boolean(source.shiftKey),
    ctrl_key: Boolean(source.ctrlKey),
    alt_key: Boolean(source.altKey),
    meta_key: Boolean(source.metaKey),
  };
}

/**
 * Build a keyboard request from key and optional modifier/repeat flags.
 * All flags default to false and are converted to booleans; key is retained
 * exactly. Return a new mutable command record without interpreting the key
 * or checking action legality. Python handles the command's meaning.
 *
 * @param {string} key
 * @param {ModifierSource & {repeat?: boolean}} options
 */
export function keyboardCommand(
  key,
  {
    shiftKey = false,
    ctrlKey = false,
    altKey = false,
    metaKey = false,
    repeat = false,
  } = {},
) {
  return {
    command_type: "keyboard",
    key,
    shift_key: Boolean(shiftKey),
    ctrl_key: Boolean(ctrlKey),
    alt_key: Boolean(altKey),
    meta_key: Boolean(metaKey),
    repeat: Boolean(repeat),
  };
}

/**
 * Return whether rawPresentation says local animation must settle before
 * submission: submissionBlocked or paused is true, or animationCount is a
 * positive integer. Nonobject input returns false. This checks the supplied
 * local state only; it does not stop animation or send a command.
 *
 * @param {unknown} rawPresentation
 */
export function presentationRequiresSubmissionSettle(rawPresentation) {
  if (
    typeof rawPresentation !== "object" ||
    rawPresentation === null ||
    Array.isArray(rawPresentation)
  ) {
    return false;
  }
  const presentation = /** @type {Record<string, unknown>} */ (rawPresentation);
  return (
    presentation.submissionBlocked === true ||
    presentation.paused === true ||
    (Number.isInteger(presentation.animationCount) &&
      Number(presentation.animationCount) > 0)
  );
}

/**
 * Translate a validated selector string into a target command. Empty value
 * returns an Escape keyboard command; one digit 0–9 returns a roster_selection
 * request with that global slot. Other strings return null. No targetability
 * or permission is granted by this conversion.
 *
 * @param {string} value
 * @returns {Record<string, unknown> | null}
 */
export function targetSelectionCommand(value) {
  if (value === "") {
    return keyboardCommand("Escape");
  }
  if (!/^(0|[1-9])$/u.test(value)) {
    return null;
  }
  return {
    command_type: "roster_selection",
    role: "target",
    global_slot: Number(value),
  };
}

/**
 * Project an effective episode replacement from frame and command, or null.
 *
 * Recognize reset, a changed valid combat configuration, an available changed
 * scenario, or an unmodified R keyboard reset. Both teams accept the same five
 * controllers. The reactive controllers (reactive_tdm, scenario_5 and tdm_gamma)
 * require SharedObs whichever team uses them. Return a frozen exact
 * replacement request. This is an advisory browser check against the current
 * frame; Python repeats validation against the actual session. Inputs stay
 * unchanged and no reset or recording discard occurs here.
 *
 * @param {Record<string, any>} frame
 * @param {Record<string, unknown>} command
 * @returns {Readonly<Record<string, unknown>> | null}
 */
export function recordingReplacementCommand(frame, command) {
  if (command.command_type === "reset") {
    return Object.freeze({ command_type: "reset" });
  }
  if (command.command_type === "set_combat_configuration") {
    const installed = frame.combat_configuration;
    const teamAController = command.team_a_controller;
    const teamBController = command.team_b_controller;
    const informationMode = command.execution_information_mode;
    if (
      !installed ||
      typeof installed !== "object" ||
      Array.isArray(installed) ||
      !isTeamController(teamAController) ||
      !isTeamController(teamBController) ||
      (informationMode !== "shared_obs" && informationMode !== "no_shared_obs") ||
      ((isReactiveController(teamAController) ||
        isReactiveController(teamBController)) &&
        informationMode !== "shared_obs") ||
      (installed.team_a_controller === teamAController &&
        installed.team_b_controller === teamBController &&
        installed.execution_information_mode === informationMode)
    ) {
      return null;
    }
    return Object.freeze({
      command_type: "set_combat_configuration",
      team_a_controller: teamAController,
      team_b_controller: teamBController,
      execution_information_mode: informationMode,
    });
  }
  if (command.command_type === "scenario_switch") {
    const scenarioName = command.scenario_name;
    if (
      typeof scenarioName !== "string" ||
      scenarioName === frame.scenario?.name ||
      !Array.isArray(frame.available_scenarios) ||
      !frame.available_scenarios.some(
        (entry) => (typeof entry === "string" ? entry : entry?.name) === scenarioName,
      )
    ) {
      return null;
    }
    return Object.freeze({
      command_type: "scenario_switch",
      scenario_name: scenarioName,
    });
  }
  if (
    command.command_type !== "keyboard" ||
    typeof command.key !== "string" ||
    command.ctrl_key === true ||
    command.alt_key === true ||
    command.meta_key === true
  ) {
    return null;
  }
  const key = command.key.toLowerCase();
  if (key === "r" && command.shift_key !== true) {
    return Object.freeze({ command_type: "reset" });
  }
  return null;
}

/**
 * Return a frozen allow, block or confirm decision for command in frame.
 *
 * Missing recording state is allowed. Save/review waits for pending method work;
 * a failed method still permits saving its last accepted game. Exit remains
 * available for cancellation and graceful closeout. An
 * episode replacement (R or another restart) made while the game-over save
 * runs (lifecycle "sealed" or "finalized_unsaved") is allowed as a new
 * save_and_replace command wrapping it, with no dialog; otherwise it asks for
 * confirmation when discard_available is true, or blocks when restart_fenced
 * is true. Outside active recording, block other scientific commands while
 * allowing the declared display-only ones. Any other allow result retains the
 * command reference, so a caller sends `result.command` when it differs from
 * its own; notices explain other results. This does not send commands or
 * replace Python lifecycle checks.
 *
 * @param {Record<string, any>} frame
 * @param {Record<string, unknown>} command
 * @returns {Readonly<{
 *   action: "allow" | "block" | "confirm",
 *   command?: Record<string, unknown>,
 *   replacement?: Record<string, unknown>,
 *   notice?: string,
 * }>}
 */
export function recordingCommandDecision(frame, command) {
  if (["finish_system", "cancel_system"].includes(String(command.command_type)))
    return Object.freeze({ action: "allow", command });
  const status = frame.recording;
  if (!status || typeof status !== "object" || Array.isArray(status)) {
    return Object.freeze({ action: "allow", command });
  }
  if (RECORDING_LIFECYCLE_COMMANDS.has(String(command.command_type))) {
    if (
      command.command_type !== "exit" &&
      frame.system_operation &&
      !(frame.fresh_launch_required && frame.system_operation.state === "failed")
    ) {
      return Object.freeze({
        action: "block",
        notice: "Wait for the current turn and its recording to finish.",
      });
    }
    return Object.freeze({ action: "allow", command });
  }
  const replacement = recordingReplacementCommand(frame, command);
  if (replacement) {
    // The game-over save is already running ("Saving Replay…"): there is
    // nothing to choose, so the restart goes out as Save And Replace, which
    // the server holds until the save ends and then restarts.
    if (status.lifecycle === "sealed" || status.lifecycle === "finalized_unsaved") {
      return Object.freeze({
        action: "allow",
        command: Object.freeze({ command_type: "save_and_replace", replacement }),
      });
    }
    if (status.discard_available === true) {
      return Object.freeze({
        action: "confirm",
        replacement,
        notice: "Choose Save, Discard or Cancel before replacing this recorded game.",
      });
    }
    if (status.restart_fenced === true) {
      return Object.freeze({
        action: "block",
        notice: "Save or discard this recording before starting another match.",
      });
    }
    return Object.freeze({ action: "allow", command });
  }

  const presentationOnly =
    command.command_type === "set_view" ||
    command.command_type === "set_preset" ||
    (command.command_type === "roster_selection" && command.role === "control") ||
    (command.command_type === "keyboard" &&
      typeof command.key === "string" &&
      RECORDING_PRESENTATION_KEYS.has(command.key.toLowerCase()));
  if (status.lifecycle !== "recording" && !presentationOnly) {
    return Object.freeze({
      action: "block",
      notice:
        "Scientific controls are fenced because this recording is no longer capturing transitions.",
    });
  }
  return Object.freeze({ action: "allow", command });
}

/**
 * Return a frozen save_as request for a valid replay file name, or null.
 * value must be 20–160 characters, begin with an ASCII letter/digit, contain
 * only letters/digits/dot/underscore/hyphen and end in .marlbg-replay.json.
 * Paths and surrounding whitespace are rejected. No file is written here.
 *
 * @param {unknown} value
 * @returns {Readonly<{command_type: "save_as", file_name: string}> | null}
 */
export function recordingSaveAsCommand(value) {
  if (
    typeof value !== "string" ||
    value.length < 20 ||
    value.length > 160 ||
    !RECORDING_SAVE_AS_PATTERN.test(value)
  ) {
    return null;
  }
  return Object.freeze({ command_type: "save_as", file_name: value });
}

/**
 * Return true only when command is exit and object payload reports
 * result=shutdown_scheduled. Other payload shapes return false. This checks
 * the response marker; it does not close the browser or stop the service.
 *
 * @param {Record<string, unknown>} command
 * @param {unknown} payload
 */
export function commandResponseSchedulesShutdown(command, payload) {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    return false;
  }
  const response = /** @type {Record<string, unknown>} */ (payload);
  return command.command_type === "exit" && response.result === "shutdown_scheduled";
}

/**
 * Build a new keyboard request from event.key, modifiers and repeat.
 * Keep the raw key meaning for Python. Do not cancel the event or change focus.
 *
 * @param {KeyboardEvent} event
 */
function keyboardCommandFromEvent(event) {
  return {
    command_type: "keyboard",
    key: event.key,
    ...modifierFields(event),
    repeat: Boolean(event.repeat),
  };
}

/**
 * Return whether event.key is a live key the battlefield forwards to Python
 * or opens How To Play with (?). The set comes from the command set: W A S D
 * Q E Z C, X, 0 1 2, Space, Enter, N, R, G, Tab, Escape and ?. Arrow keys,
 * V, B and F return false (the arrows do nothing live; V and F are local
 * jobs). Ctrl, Alt or Meta combinations always
 * return false; Shift is allowed. One-character keys are lowercased for
 * lookup. No event is cancelled by this check.
 *
 * @param {{
 *   key: string,
 *   shiftKey?: boolean,
 *   ctrlKey?: boolean,
 *   altKey?: boolean,
 *   metaKey?: boolean,
 * }} event
 */
export function isDebuggerKey(event) {
  if (event.ctrlKey || event.altKey || event.metaKey) {
    return false;
  }
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  return GAME_KEYS.has(key);
}

/**
 * Return the global slot that Tab (direction 1) or Shift+Tab (direction -1)
 * gives the orders to in live play: the next champion you command that can
 * take orders, in slot order, wrapping around (owner's ruling, 3 October
 * 2026).
 *
 * A champion can take orders when it is Manual (in manualSlots) and is not in
 * fallenSlots. With one Manual champion on each team, both count. A policy's
 * champion is never chosen; the player looks at one by clicking its portrait
 * or body (Watch Only). When currentSlot is a policy's champion or a fallen
 * one, Tab goes to the first champion that can take orders after it in slot
 * order (Shift+Tab: the first before it). Pure: reads only its argument.
 *
 * Example: manualSlots [0, 1, 2, 3, 4], fallenSlots [2], currentSlot 1 and
 * direction 1 give 3; currentSlot 7 (an enemy) and direction 1 give 0;
 * direction -1 from 0 gives 4.
 *
 * @param {{
 *   manualSlots: ReadonlyArray<unknown>,
 *   currentSlot: number | null,
 *   fallenSlots?: ReadonlyArray<unknown>,
 *   direction: 1 | -1,
 * }} facts manualSlots: the global slots (0 to 9) of every Manual champion,
 *   in any order (non-integers are ignored). currentSlot: the global slot
 *   that has the orders, or that the player looks at now; null when unknown
 *   (Tab then picks the first slot, Shift+Tab the last). fallenSlots: Manual
 *   slots known to be fallen right now; default none. direction: 1 for Tab
 *   (next), -1 for Shift+Tab (previous).
 * @returns {number | null} The slot to give the orders to. It is currentSlot
 *   itself when that is the only champion that can take orders, and null when
 *   none can (no Manual slot, or every Manual champion is fallen).
 */
export function nextOrderSlot({
  manualSlots,
  currentSlot,
  fallenSlots = [],
  direction,
}) {
  const fallen = new Set(fallenSlots);
  const ready = [...new Set(manualSlots)]
    .filter(
      (slot) => typeof slot === "number" && Number.isInteger(slot) && !fallen.has(slot),
    )
    .map((slot) => Number(slot))
    .sort((left, right) => left - right);
  if (ready.length === 0) return null;
  const last = ready[ready.length - 1];
  if (currentSlot === null || !Number.isInteger(currentSlot)) {
    return direction < 0 ? last : ready[0];
  }
  if (direction < 0) {
    const before = ready.filter((slot) => slot < currentSlot);
    return before.length > 0 ? before[before.length - 1] : last;
  }
  return ready.find((slot) => slot > currentSlot) ?? ready[0];
}

/**
 * Convert event client-pixel coordinates into svg's local coordinates.
 * Use the inverse screen transform; return null when no transform exists.
 * Return a new x/y record without moving the pointer or DOM. The caller
 * checks world conversion and finiteness; matrix errors may propagate.
 *
 * @param {SVGSVGElement} svg
 * @param {PointerEvent} event
 * @returns {{x: number, y: number} | null}
 */
function pointInSvg(svg, event) {
  const matrix = svg.getScreenCTM();
  if (!matrix) {
    return null;
  }
  const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(
    matrix.inverse(),
  );
  return { x: point.x, y: point.y };
}

/**
 * Attach keyboard and primary-pointer listeners to one focusable battlefield.
 *
 * bindings.battlefield receives the listeners. toWorldPoint maps SVG points
 * to world_x/world_y or null. onCommand handles built requests; onHelp shows
 * help and onReleaseFocus follows Escape, even if its awaited command fails.
 * onPointerCommand defaults to a no-op and observes the pointer target/request
 * before submission. isInteractive defaults to true; onFencedCommand defaults
 * to false and may consume blocked keys; ownsFencedSpaceDefault defaults to
 * true to suppress page scrolling for a blocked Space key;
 * ownsFencedTabDefault defaults to true to keep a blocked Tab in the arena
 * (the Replay Viewer passes false: its own Tab steps agents). ownsLocalCommands
 * defaults to true; return false (in the Replay Viewer) to leave V, B and F to
 * the replay key handler.
 *
 * Local jobs: while ownsLocalCommands() is true, a key that is not a forwarded
 * live key (V or F; B has no live job) runs its registered local handler
 * through runLocalCommand("combat_debugger", ...) with context
 * {interactive: isInteractive()}, even while commands are fenced. When a
 * handler ran, the event's default and propagation are stopped and nothing is
 * sent to Python. With no handler registered the key does nothing. The local
 * handler owns its own Agent POV limits.
 *
 * Keys while orders are closed (isInteractive() is false): ? still calls
 * onHelp (How To Play is local and safe at any time, also at game over and
 * while a command is out). A forwarded key for which allowsWhileClosed(command)
 * returns true still goes to onCommand (main.js allows R and G at game over).
 * Every other key goes to onFencedCommand as before. A Tab that
 * onFencedCommand does not keep has its browser default stopped while
 * ownsFencedTabDefault() is true, so Tab never moves focus from the arena
 * onto the marks drawn inside it (stones, champions) while the policies
 * choose or at game over (owner, 3 Oct 2026); it does nothing there.
 *
 * Pointer input: the primary button sends a "primary" battlefield_pointer and
 * a right click (button 2) sends a "secondary" one at the same world point
 * (Python clears the staged target; it changes nothing else). Both focus the
 * battlefield first, so keys stay on. While orders are open the browser's
 * context menu is suppressed on the battlefield, and so is the menu that
 * follows a right press the battlefield took (sent, or kept by
 * onFencedCommand), although that press has already made the page busy.
 * While orders are closed
 * (isInteractive() is false, for example while an earlier order is out), a
 * Shift + click (a target pick) or a right click is built the same way and
 * offered to onFencedCommand, which may keep it to send after the request
 * (main.js keeps it in order with queued keys); a plain click is ignored.
 *
 * Only events targeted at the battlefield are captured. Pointer input focuses
 * it and rejects missing/nonfinite world coordinates. Ordinary command
 * promises are not awaited here; callers own their error handling. Return
 * undefined. No disposer is returned, so bind once per battlefield element
 * and release its owning DOM when finished. No document-level listener or
 * Python action logic is added.
 *
 * @param {{
 *   battlefield: SVGSVGElement,
 *   toWorldPoint: (point: {x: number, y: number}) =>
 *     {world_x: number, world_y: number} | null,
 *   onCommand: (command: Record<string, unknown>) => void | Promise<void>,
 *   onPointerCommand?: (
 *     target: EventTarget | null,
 *     command: Readonly<Record<string, unknown>>,
 *   ) => void,
 *   onHelp: () => void,
 *   onReleaseFocus: () => void,
 *   isInteractive?: () => boolean,
 *   onFencedCommand?: (command: Record<string, unknown>) => boolean,
 *   allowsWhileClosed?: (command: Readonly<Record<string, unknown>>) => boolean,
 *   ownsFencedSpaceDefault?: () => boolean,
 *   ownsFencedTabDefault?: () => boolean,
 *   ownsLocalCommands?: () => boolean,
 * }} bindings allowsWhileClosed defaults to "never": return true for a
 *   keyboard command that may still be sent while isInteractive() is false.
 */
export function bindBattlefieldControls({
  battlefield,
  toWorldPoint,
  onCommand,
  onPointerCommand = () => {},
  onHelp,
  onReleaseFocus,
  isInteractive = () => true,
  onFencedCommand = () => false,
  allowsWhileClosed = () => false,
  ownsFencedSpaceDefault = () => true,
  ownsFencedTabDefault = () => true,
  ownsLocalCommands = () => true,
}) {
  battlefield.addEventListener("keydown", async (event) => {
    if (
      event.target === battlefield &&
      !isDebuggerKey(event) &&
      ownsLocalCommands() &&
      runLocalCommand("combat_debugger", event, { interactive: isInteractive() })
    ) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (event.target !== battlefield || !isDebuggerKey(event)) {
      return;
    }
    if (event.key === "?" && ownsLocalCommands()) {
      // How To Play is local: it opens at any time, also at game over.
      event.preventDefault();
      event.stopPropagation();
      onHelp();
      return;
    }
    if (!isInteractive()) {
      const command = keyboardCommandFromEvent(event);
      if (allowsWhileClosed(command)) {
        event.preventDefault();
        event.stopPropagation();
        onCommand(command);
        return;
      }
      if (onFencedCommand(command)) {
        event.preventDefault();
        event.stopPropagation();
      } else if (event.key === " " && ownsFencedSpaceDefault()) {
        event.preventDefault();
      } else if (event.key === "Tab" && ownsFencedTabDefault()) {
        // Keep focus in the arena: the browser's own Tab would walk onto the
        // marks drawn inside it, where the arena's keys no longer work.
        event.preventDefault();
      }
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (event.key === "?") {
      onHelp();
      return;
    }
    if (event.key === "Escape") {
      try {
        await onCommand(keyboardCommandFromEvent(event));
      } finally {
        onReleaseFocus();
      }
      return;
    }
    onCommand(keyboardCommandFromEvent(event));
  });

  // The browser fires contextmenu after the right press's pointerdown, which
  // has already sent its command and made the page busy, so the press itself
  // records that the battlefield took it.
  let rightPressTaken = false;
  battlefield.addEventListener("contextmenu", (event) => {
    if (rightPressTaken || isInteractive()) {
      event.preventDefault();
    }
    rightPressTaken = false;
  });

  /**
   * Build the battlefield_pointer command for a pointer event at its world
   * point, or null when the point is outside the drawn field.
   *
   * @param {PointerEvent} event A primary or secondary pointerdown.
   * @returns {Record<string, unknown> | null}
   */
  const pointerCommand = (event) => {
    const svgPoint = pointInSvg(battlefield, event);
    if (!svgPoint) {
      return null;
    }
    const worldPoint = toWorldPoint(svgPoint);
    if (
      !worldPoint ||
      !Number.isFinite(worldPoint.world_x) ||
      !Number.isFinite(worldPoint.world_y)
    ) {
      return null;
    }
    return {
      command_type: "battlefield_pointer",
      world_x: worldPoint.world_x,
      world_y: worldPoint.world_y,
      button: event.button === 2 ? "secondary" : "primary",
      ...modifierFields(event),
    };
  };

  battlefield.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 && event.button !== 2) {
      return;
    }
    rightPressTaken = false;
    if (!isInteractive()) {
      if (event.button === 2 || event.shiftKey) {
        const command = pointerCommand(event);
        if (command !== null && onFencedCommand(command)) {
          event.preventDefault();
          rightPressTaken = event.button === 2;
        }
      }
      return;
    }
    event.preventDefault();
    rightPressTaken = event.button === 2;
    battlefield.focus({ preventScroll: true });
    const command = pointerCommand(event);
    if (command === null) {
      return;
    }
    onPointerCommand(event.target, command);
    onCommand(command);
  });
}

/**
 * The vision page's names for the nine movement ids (0 Stay, 1 North,
 * 2 South, 3 East, 4 West, 5 North-East, 6 North-West, 7 South-East,
 * 8 South-West), as Core numbers them.
 */
export const STEP_NAMES = Object.freeze([
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
 * @typedef {(
 *   | {kind: "step", move: number}
 *   | {kind: "no_combat"}
 *   | {kind: "armed", ability: string, target: string | null}
 *   | {kind: "burst_armed"}
 *   | {kind: "needs_target", ability: string}
 *   | {kind: "not_allowed", ability: string, onTarget: boolean, readyIn: number}
 *   | {kind: "target", target: string, basic: string, ultimate: string,
 *      auto: boolean, ultimateAllowed: boolean}
 *   | {kind: "cleared", keysOff: boolean}
 *   | {kind: "keys_off"}
 *   | {kind: "focus_own", className: string}
 *   | {kind: "nobody_ready"}
 *   | {kind: "look", target: string, controller: string}
 *   | {kind: "policy_champion", watch: boolean}
 *   | {kind: "stopped"}
 *   | {kind: "lock_in", watch: boolean}
 *   | {kind: "view", view: "pov" | "oracle", unseen: number, viewers: string,
 *      watch: boolean}
 *   | {kind: "ranges", shown: boolean}
 *   | {kind: "restart_cancelled"}
 *   | {kind: "work_cancelled"}
 *   | {kind: "running"}
 *   | {kind: "pausing"}
 *   | {kind: "paused"}
 * )} StatusEvent
 */

/**
 * Return the status line's plain sentence for one of the player's own
 * commands, in the vision page's exact words.
 *
 * Pure: it reads only its argument. Names come from the caller already
 * formatted ("Mage (ID 5)", "Shield Bash"); the caller owns Agent POV limits
 * (it must name only champions the viewing team sees, or the player's own).
 * Examples: {kind: "step", move: 1} gives "Step: North."; {kind:
 * "needs_target", ability: "Charge"} gives "Charge needs a target first. Your
 * order is unchanged."; {kind: "not_allowed", ability: "Shield Bash",
 * onTarget: true, readyIn: 0} gives "Shield Bash is not allowed on this target
 * right now. Your order is unchanged."; a readyIn above 0 adds "It is ready in
 * N ticks." ("1 tick" for one). {kind: "view", view: "pov", unseen: 2,
 * viewers: "your team", watch: false} gives "Agent POV: you see only what your
 * team sees. 2 enemy champions are out of sight." The game's first sentence
 * and the closed page's words are not here: tactics-turn.js owns them
 * (startGreeting, closedSentence), because they depend on the roster and the
 * replay's save state.
 *
 * @param {StatusEvent} event What happened. Unknown kinds return "".
 * @returns {string} One or more sentences, or "" for an unknown event.
 */
export function statusSentence(event) {
  const ticks = (/** @type {number} */ n) => `${n} tick${n === 1 ? "" : "s"}`;
  switch (event.kind) {
    case "step":
      return event.move === 0
        ? "Step: Stay."
        : `Step: ${STEP_NAMES[event.move] ?? "Stay"}.`;
    case "no_combat":
      return "No Combat: your champion only moves this tick. Your target stays picked.";
    case "armed":
      return event.target === null
        ? `${event.ability} is armed.`
        : `${event.ability} is armed on the ${event.target}.`;
    case "burst_armed":
      return "Burst is armed. It has no target, so your target was cleared.";
    case "needs_target":
      return `${event.ability} needs a target first. Your order is unchanged.`;
    case "not_allowed": {
      const later = event.readyIn > 0 ? ` It is ready in ${ticks(event.readyIn)}.` : "";
      return event.onTarget
        ? `${event.ability} is not allowed on this target right now. Your order is unchanged.${later}`
        : `${event.ability} is not allowed right now. Your order is unchanged.${later}`;
    }
    case "target":
      if (event.auto) {
        return `Target: ${event.target}. ${event.basic} armed itself, because it is allowed (Auto).`;
      }
      return event.ultimateAllowed
        ? `Target: ${event.target}. ${event.basic} is not allowed on it, so your action is No Combat. Press 2 for ${event.ultimate}.`
        : `Target: ${event.target}. Nothing is allowed on it right now, so your action is No Combat.`;
    case "cleared":
      return event.keysOff
        ? "Target cleared. Keys are off until you click the arena again."
        : "Target cleared.";
    case "keys_off":
      return "Keys are off until you click the arena again.";
    case "focus_own":
      return `Giving orders to your ${event.className}.`;
    case "nobody_ready":
      return "All your champions are fallen, so nobody can take orders until they return.";
    case "look":
      return `Looking at the ${event.target}, played by ${event.controller}.`;
    case "policy_champion":
      return event.watch
        ? "Nobody is yours this game. You can look, not command."
        : "This champion is played by a policy. You can look, not command. Pick one of yours to give orders.";
    case "stopped":
      return "A fresh start is needed, so orders cannot change. You can look around, save or exit.";
    case "lock_in":
      return event.watch
        ? "Playing one turn."
        : "Orders sent. The policies choose next.";
    case "view": {
      if (event.view === "oracle") {
        return `Oracle View: every champion is shown, even those ${event.viewers} cannot see.`;
      }
      const lead = `Agent POV: you see only what ${event.viewers} sees.`;
      if (event.unseen === 0) return `${lead} Every champion is in sight.`;
      const who = event.watch ? "" : "enemy ";
      return `${lead} ${event.unseen} ${who}champion${event.unseen === 1 ? " is" : "s are"} out of sight.`;
    }
    case "ranges":
      return event.shown ? "Reach rings shown." : "Reach rings hidden.";
    case "restart_cancelled":
      return "Restart cancelled. The game goes on.";
    case "work_cancelled":
      return "Policy work cancelled. The game so far is kept. A fresh start from Python is needed to play on.";
    case "running":
      return "Running. Press Pause to stop after the current turn.";
    case "pausing":
      return "Pausing after the current turn.";
    case "paused":
      return "Paused after the current turn.";
    default:
      return "";
  }
}

/**
 * Return whether a live tick the page already showed is only being shown to a
 * new audience, so the page should show it at its end and not play it again.
 *
 * Why: Agent POV and Oracle View carry different transition ids for the same
 * tick, so the choreographer would glide that tick again after a view switch,
 * and while it replays the deck counts the tick as All Act and drops order
 * keys without a word. A view change only changes what the human sees.
 *
 * True only when both records exist, they show the same simulator tick, their
 * audienceKey strings differ (view kind or Agent POV recipient) and both keys
 * name the same session (the first item of the JSON array audienceKey, as
 * choreography-plan.js builds it: [session_id, presentation_kind,
 * recipient_presentation_key]). False for the first tick shown (shown null), a
 * new tick, a new session (a restart) or the same audience. Pure; it reads
 * nothing hidden from an Agent POV team, only the two plans' keys.
 *
 * @param {{audienceKey: unknown, simulatorStep: unknown} | null} shown The
 *   audience key and tick of the last live plan shown, or null before the
 *   first.
 * @param {{audienceKey: unknown, simulatorStep: unknown} | null} next The same
 *   facts for the plan just installed, or null when there is none.
 * @returns {boolean}
 */
export function audienceSwitchKeepsTick(shown, next) {
  if (shown === null || next === null) return false;
  if (shown.audienceKey === next.audienceKey) return false;
  if (shown.simulatorStep !== next.simulatorStep) return false;
  /** @param {unknown} key @returns {unknown} The session id, or null. */
  const sessionOf = (key) => {
    try {
      const parsed = JSON.parse(String(key));
      return Array.isArray(parsed) ? parsed[0] : null;
    } catch {
      return null;
    }
  };
  const session = sessionOf(next.audienceKey);
  return (
    session !== null &&
    session !== undefined &&
    session === sessionOf(shown.audienceKey)
  );
}
