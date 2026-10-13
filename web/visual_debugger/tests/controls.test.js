/**
 * @file Check keyboard and pointer command routing, native browser shortcuts and
 * submission state labels. Live keys follow the one command set: the arrows,
 * V, B and F are never forwarded to Python, while N (one tick) is, in either
 * letter case; V and F run their registered local
 * handlers on the focused battlefield (even while commands are fenced) and stop
 * the event; B and unregistered local keys do nothing; ownsLocalCommands false
 * leaves V, B and F to the Replay Viewer. A right click on the battlefield
 * sends the secondary pointer (Python clears the target) and keeps the browser
 * menu closed while orders are open; while they are closed it stays native and
 * sends nothing. Recording restarts accept the BETA
 * (scenario_5) and GAMMA (tdm_gamma) controllers for either team, and reject
 * them under NoSharedObs whichever team uses them.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { registerLocalCommandHandler } from "../src/command-set.js";
import {
  bindBattlefieldControls,
  commandResponseSchedulesShutdown,
  isDebuggerKey,
  keyboardCommand,
  presentationRequiresSubmissionSettle,
  recordingCommandDecision,
  recordingReplacementCommand,
  recordingSaveAsCommand,
  targetSelectionCommand,
} from "../src/controls.js";

/** @param {string} key
 * @param {{repeat?: boolean, ctrlKey?: boolean}} [options] */
function cancelableKeydown(key, { repeat = false, ctrlKey = false } = {}) {
  const event = new Event("keydown", { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    key: { value: key },
    repeat: { value: repeat },
    shiftKey: { value: false },
    ctrlKey: { value: ctrlKey },
    altKey: { value: false },
    metaKey: { value: false },
  });
  return event;
}

/** @param {"pointerdown" | "contextmenu"} type
 * @param {number} [button] */
function cancelablePointerEvent(type, button = 2) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    button: { value: button },
    clientX: { value: 10 },
    clientY: { value: 20 },
    shiftKey: { value: false },
    ctrlKey: { value: false },
    altKey: { value: false },
    metaKey: { value: false },
  });
  return event;
}

test("keyboardCommand forwards raw key and modifier state without semantics", () => {
  assert.deepEqual(
    keyboardCommand("Enter", {
      shiftKey: true,
      repeat: true,
    }),
    {
      command_type: "keyboard",
      key: "Enter",
      shift_key: true,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: true,
    },
  );
});

test("Submit settlement covers gated, post-gate active, and paused explanations", () => {
  assert.equal(
    presentationRequiresSubmissionSettle({
      submissionBlocked: true,
      animationCount: 3,
      paused: false,
    }),
    true,
  );
  assert.equal(
    presentationRequiresSubmissionSettle({
      submissionBlocked: false,
      animationCount: 2,
      paused: false,
    }),
    true,
  );
  assert.equal(
    presentationRequiresSubmissionSettle({
      submissionBlocked: false,
      animationCount: 2,
      paused: true,
    }),
    true,
  );
  assert.equal(
    presentationRequiresSubmissionSettle({
      submissionBlocked: false,
      animationCount: 0,
      paused: false,
    }),
    false,
  );
  assert.equal(presentationRequiresSubmissionSettle(null), false);
});

test("debugger key capture leaves modified browser shortcuts native", () => {
  assert.equal(isDebuggerKey({ key: "r" }), true);
  assert.equal(isDebuggerKey({ key: "v" }), false);
  assert.equal(isDebuggerKey({ key: "x" }), true);
  assert.equal(isDebuggerKey({ key: "X" }), true);
  assert.equal(isDebuggerKey({ key: "0" }), true);
  assert.equal(isDebuggerKey({ key: "Tab", shiftKey: true }), true);
  assert.equal(isDebuggerKey({ key: "r", ctrlKey: true }), false);
  assert.equal(isDebuggerKey({ key: "p", ctrlKey: true }), false);
  assert.equal(isDebuggerKey({ key: "w", metaKey: true }), false);
  assert.equal(isDebuggerKey({ key: "ArrowLeft", altKey: true }), false);
  assert.equal(isDebuggerKey({ key: "F5" }), false);
  for (const retiredKey of ["[", "]", "p", "P"]) {
    assert.equal(isDebuggerKey({ key: retiredKey }), false);
  }
  // N is one tick: forwarded like Space, in either letter case.
  assert.equal(isDebuggerKey({ key: "n" }), true);
  assert.equal(isDebuggerKey({ key: "N", shiftKey: true }), true);
  assert.equal(isDebuggerKey({ key: "n", ctrlKey: true }), false);
  for (const key of ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]) {
    assert.equal(isDebuggerKey({ key }), false, key);
    assert.equal(isDebuggerKey({ key, shiftKey: true }), false, key);
  }
  for (const key of ["v", "V", "b", "B", "f", "F"]) {
    assert.equal(isDebuggerKey({ key }), false, key);
  }
  for (const key of ["w", "a", "s", "d", "q", "e", "z", "c", "x", "0", "1", "2"]) {
    assert.equal(isDebuggerKey({ key }), true, key);
  }
  for (const key of [" ", "Enter", "r", "g", "?", "Tab", "Escape"]) {
    assert.equal(isDebuggerKey({ key }), true, key);
  }
});

test("V and F reach their local handlers on the live battlefield and never Python", () => {
  const battlefield = new EventTarget();
  let interactive = true;
  let ownsLocal = true;
  /** @type {Record<string, unknown>[]} */
  const commands = [];
  /** @type {Record<string, unknown>[]} */
  const fencedCommands = [];
  /** @type {any[]} */
  const calls = [];
  let helpCount = 0;
  bindBattlefieldControls({
    battlefield: /** @type {any} */ (battlefield),
    toWorldPoint: () => null,
    onCommand: (command) => {
      commands.push(command);
    },
    onHelp: () => {
      helpCount += 1;
    },
    onReleaseFocus: () => {},
    isInteractive: () => interactive,
    onFencedCommand: (command) => {
      fencedCommands.push(command);
      return true;
    },
    ownsLocalCommands: () => ownsLocal,
  });

  // No handler registered yet: V does nothing and stays native.
  const early = cancelableKeydown("v");
  battlefield.dispatchEvent(early);
  assert.equal(early.defaultPrevented, false);

  const removeCamera = registerLocalCommandHandler(
    "camera_everyone_in_view",
    (/** @type {any} */ call) => {
      calls.push(call);
    },
  );
  const removeExplain = registerLocalCommandHandler(
    "frame_explanation",
    (/** @type {any} */ call) => {
      calls.push(call);
    },
  );
  try {
    let propagated = 0;
    const view = cancelableKeydown("v");
    view.stopPropagation = () => {
      propagated += 1;
    };
    battlefield.dispatchEvent(view);
    assert.equal(view.defaultPrevented, true);
    assert.equal(propagated, 1);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].product, "combat_debugger");
    assert.equal(calls[0].job, "everyone_in_view");
    assert.deepEqual(calls[0].context, { interactive: true });

    const explain = cancelableKeydown("F");
    battlefield.dispatchEvent(explain);
    assert.equal(explain.defaultPrevented, true);
    assert.equal(calls[1].job, "frame_explanation");

    // B has no live job, and the arrows are not live keys.
    for (const key of ["b", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]) {
      const event = cancelableKeydown(key);
      battlefield.dispatchEvent(event);
      assert.equal(event.defaultPrevented, false, key);
    }

    // While commands are fenced, V still runs locally and is never fenced.
    interactive = false;
    const fencedView = cancelableKeydown("v");
    battlefield.dispatchEvent(fencedView);
    assert.equal(fencedView.defaultPrevented, true);
    assert.deepEqual(calls[2].context, { interactive: false });
    assert.deepEqual(fencedCommands, []);

    // With Ctrl held, V stays a browser shortcut.
    interactive = true;
    const paste = cancelableKeydown("v", { ctrlKey: true });
    battlefield.dispatchEvent(paste);
    assert.equal(paste.defaultPrevented, false);

    // In the Replay Viewer the replay key handler owns V, B and F.
    ownsLocal = false;
    const replayView = cancelableKeydown("v");
    battlefield.dispatchEvent(replayView);
    assert.equal(replayView.defaultPrevented, false);
    assert.equal(calls.length, 3);

    // ? still opens How to Play through onHelp, not a local handler.
    ownsLocal = true;
    battlefield.dispatchEvent(cancelableKeydown("?"));
    assert.equal(helpCount, 1);
    assert.deepEqual(commands, []);
  } finally {
    removeCamera();
    removeExplain();
  }
});

test("battlefield-owned commands and accepted fenced input never become browser defaults", () => {
  const battlefield = new EventTarget();
  let interactive = false;
  let ownFencedCommand = true;
  let ownFencedSpaceDefault = true;
  /** @type {Record<string, unknown>[]} */
  const commands = [];
  /** @type {Record<string, unknown>[]} */
  const fencedCommands = [];
  bindBattlefieldControls({
    battlefield: /** @type {any} */ (battlefield),
    toWorldPoint: () => null,
    onCommand: (command) => {
      commands.push(command);
    },
    onHelp: () => {},
    onReleaseFocus: () => {},
    isInteractive: () => interactive,
    onFencedCommand: (command) => {
      fencedCommands.push(command);
      return ownFencedCommand;
    },
    ownsFencedSpaceDefault: () => ownFencedSpaceDefault,
  });

  const fencedSpace = cancelableKeydown(" ", { repeat: true });
  battlefield.dispatchEvent(fencedSpace);
  assert.equal(fencedSpace.defaultPrevented, true);
  assert.deepEqual(fencedCommands, [keyboardCommand(" ", { repeat: true })]);
  assert.deepEqual(commands, []);

  const fencedMovement = cancelableKeydown("w");
  battlefield.dispatchEvent(fencedMovement);
  assert.equal(fencedMovement.defaultPrevented, true);
  assert.deepEqual(fencedCommands, [
    keyboardCommand(" ", { repeat: true }),
    keyboardCommand("w"),
  ]);
  assert.deepEqual(commands, []);

  const fencedEnter = cancelableKeydown("Enter");
  battlefield.dispatchEvent(fencedEnter);
  assert.equal(fencedEnter.defaultPrevented, true);
  assert.deepEqual(fencedCommands, [
    keyboardCommand(" ", { repeat: true }),
    keyboardCommand("w"),
    keyboardCommand("Enter"),
  ]);
  assert.deepEqual(commands, []);

  const fencedRepeatEnter = cancelableKeydown("Enter", { repeat: true });
  battlefield.dispatchEvent(fencedRepeatEnter);
  assert.equal(fencedRepeatEnter.defaultPrevented, true);
  assert.deepEqual(fencedCommands, [
    keyboardCommand(" ", { repeat: true }),
    keyboardCommand("w"),
    keyboardCommand("Enter"),
    keyboardCommand("Enter", { repeat: true }),
  ]);
  assert.deepEqual(commands, []);

  ownFencedCommand = false;
  const nativeFencedEnter = cancelableKeydown("Enter");
  battlefield.dispatchEvent(nativeFencedEnter);
  assert.equal(nativeFencedEnter.defaultPrevented, false);
  assert.deepEqual(commands, []);

  const ownedFencedSpace = cancelableKeydown(" ");
  battlefield.dispatchEvent(ownedFencedSpace);
  assert.equal(ownedFencedSpace.defaultPrevented, true);
  ownFencedSpaceDefault = false;
  const nativeFencedSpace = cancelableKeydown(" ");
  battlefield.dispatchEvent(nativeFencedSpace);
  assert.equal(nativeFencedSpace.defaultPrevented, false);

  interactive = true;
  const ownedSpace = cancelableKeydown(" ");
  battlefield.dispatchEvent(ownedSpace);
  assert.equal(ownedSpace.defaultPrevented, true);
  assert.deepEqual(commands, [keyboardCommand(" ")]);

  const modifiedSpace = cancelableKeydown(" ", { ctrlKey: true });
  battlefield.dispatchEvent(modifiedSpace);
  assert.equal(modifiedSpace.defaultPrevented, false);
  assert.deepEqual(commands, [keyboardCommand(" ")]);
});

test("a right click sends the secondary pointer and keeps the browser menu closed", () => {
  const battlefield = /** @type {any} */ (new EventTarget());
  let focused = 0;
  battlefield.focus = () => {
    focused += 1;
  };
  battlefield.getScreenCTM = () => ({ inverse: () => ({}) });
  const savedPoint = Reflect.get(globalThis, "DOMPoint");
  Reflect.set(
    globalThis,
    "DOMPoint",
    class {
      /** @param {number} x @param {number} y */
      constructor(x, y) {
        this.x = x;
        this.y = y;
      }
      matrixTransform() {
        return { x: this.x, y: this.y };
      }
    },
  );
  let interactive = true;
  let pointerCalls = 0;
  /** @type {Record<string, unknown>[]} */
  const commands = [];
  bindBattlefieldControls({
    battlefield,
    toWorldPoint: (point) => ({ world_x: point.x / 10, world_y: point.y / 10 }),
    onCommand: (command) => {
      commands.push(command);
    },
    onPointerCommand: () => {
      pointerCalls += 1;
    },
    onHelp: () => {},
    onReleaseFocus: () => {},
    isInteractive: () => interactive,
  });
  try {
    const pointerdown = cancelablePointerEvent("pointerdown");
    const contextmenu = cancelablePointerEvent("contextmenu");
    battlefield.dispatchEvent(pointerdown);
    battlefield.dispatchEvent(contextmenu);
    assert.equal(pointerdown.defaultPrevented, true);
    assert.equal(contextmenu.defaultPrevented, true);
    assert.equal(focused, 1);
    assert.equal(pointerCalls, 1);
    assert.deepEqual(commands, [
      {
        command_type: "battlefield_pointer",
        world_x: 1,
        world_y: 2,
        button: "secondary",
        shift_key: false,
        ctrl_key: false,
        alt_key: false,
        meta_key: false,
      },
    ]);
    // While orders are closed the right click stays native and sends nothing.
    interactive = false;
    const closedDown = cancelablePointerEvent("pointerdown");
    const closedMenu = cancelablePointerEvent("contextmenu");
    battlefield.dispatchEvent(closedDown);
    battlefield.dispatchEvent(closedMenu);
    assert.equal(closedDown.defaultPrevented, false);
    assert.equal(closedMenu.defaultPrevented, false);
    assert.equal(commands.length, 1);
    // A middle click does nothing at any time.
    interactive = true;
    const middle = cancelablePointerEvent("pointerdown", 1);
    battlefield.dispatchEvent(middle);
    assert.equal(middle.defaultPrevented, false);
    assert.equal(commands.length, 1);
  } finally {
    Reflect.set(globalThis, "DOMPoint", savedPoint);
  }
});

test("target selection uses the same researcher-space command in both live audiences", () => {
  assert.deepEqual(targetSelectionCommand("7"), {
    command_type: "roster_selection",
    role: "target",
    global_slot: 7,
  });
  assert.deepEqual(targetSelectionCommand(""), {
    command_type: "keyboard",
    key: "Escape",
    shift_key: false,
    ctrl_key: false,
    alt_key: false,
    meta_key: false,
    repeat: false,
  });
  assert.equal(targetSelectionCommand("pov-target-action:7"), null);
  assert.equal(targetSelectionCommand("10"), null);
});

function recordingFrame({ lifecycle = "recording", captured = 1 } = {}) {
  const persistenceFailed = lifecycle === "persistence_failed";
  const finalized = [
    "sealed",
    "finalized_unsaved",
    "persistence_failed",
    "saved",
    "reviewing",
  ].includes(lifecycle);
  return {
    scenario: {
      name: "alpha",
      ordinary_movement_distance_scale: 0.5,
    },
    available_scenarios: [{ name: "alpha" }, { name: "bravo" }, { name: "charlie" }],
    recording: {
      schema_version: 1,
      lifecycle,
      captured_transition_count: captured,
      expected_transition_count: 5,
      completion_state: finalized ? "partial" : null,
      completion_reason: finalized ? "user_finish_and_review" : null,
      restart_fenced:
        !["saved", "reviewing"].includes(lifecycle) &&
        (captured > 0 || lifecycle !== "recording"),
      finish_available: lifecycle === "recording",
      review_available: lifecycle === "saved",
      retry_available: persistenceFailed,
      save_as_available: persistenceFailed,
      discard_available:
        (lifecycle === "recording" && captured > 0) ||
        ["sealed", "finalized_unsaved", "persistence_failed"].includes(lifecycle),
      persistence_error_code: persistenceFailed ? "publication_failed" : null,
    },
  };
}

test("recording restart resolution retains direct compatibility without scenario keys", () => {
  const frame = recordingFrame();
  assert.deepEqual(recordingReplacementCommand(frame, { command_type: "reset" }), {
    command_type: "reset",
  });
  assert.deepEqual(
    recordingReplacementCommand(frame, {
      command_type: "scenario_switch",
      scenario_name: "bravo",
    }),
    {
      command_type: "scenario_switch",
      scenario_name: "bravo",
    },
  );
  assert.equal(recordingReplacementCommand(frame, keyboardCommand("[")), null);
  assert.equal(recordingReplacementCommand(frame, keyboardCommand("]")), null);
  assert.equal(recordingReplacementCommand(frame, keyboardCommand("n")), null);
  assert.deepEqual(recordingReplacementCommand(frame, keyboardCommand("r")), {
    command_type: "reset",
  });
  assert.equal(
    recordingReplacementCommand(frame, keyboardCommand("R", { shiftKey: true })),
    null,
  );
  assert.equal(
    recordingReplacementCommand(frame, {
      command_type: "scenario_switch",
      scenario_name: "missing",
    }),
    null,
  );
});

test("captured recording prefixes require exact discard confirmation for replacement", () => {
  const frame = {
    ...recordingFrame(),
    combat_configuration: {
      team_a_controller: "manual",
      team_b_controller: "manual",
      execution_information_mode: "shared_obs",
    },
  };
  for (const command of [
    { command_type: "reset" },
    { command_type: "scenario_switch", scenario_name: "bravo" },
    {
      command_type: "set_combat_configuration",
      team_a_controller: "reactive_tdm",
      team_b_controller: "reactive_tdm",
      execution_information_mode: "shared_obs",
    },
    {
      command_type: "set_combat_configuration",
      team_a_controller: "random_valid",
      team_b_controller: "random_valid",
      execution_information_mode: "no_shared_obs",
    },
    {
      command_type: "set_combat_configuration",
      team_a_controller: "manual",
      team_b_controller: "scenario_5",
      execution_information_mode: "shared_obs",
    },
    {
      command_type: "set_combat_configuration",
      team_a_controller: "manual",
      team_b_controller: "tdm_gamma",
      execution_information_mode: "shared_obs",
    },
    {
      command_type: "set_combat_configuration",
      team_a_controller: "scenario_5",
      team_b_controller: "manual",
      execution_information_mode: "shared_obs",
    },
    {
      command_type: "set_combat_configuration",
      team_a_controller: "tdm_gamma",
      team_b_controller: "manual",
      execution_information_mode: "shared_obs",
    },
    keyboardCommand("r"),
  ]) {
    const decision = recordingCommandDecision(frame, command);
    assert.equal(decision.action, "confirm");
    assert.ok(decision.replacement);
    assert.equal(Object.isFrozen(decision.replacement), true);
  }

  const frameZero = recordingFrame({ captured: 0 });
  assert.equal(
    recordingCommandDecision(frameZero, { command_type: "reset" }).action,
    "allow",
  );
  assert.equal(
    recordingCommandDecision({ ...frame, recording: null }, { command_type: "reset" })
      .action,
    "allow",
  );
  assert.equal(
    recordingReplacementCommand(frame, {
      command_type: "set_combat_configuration",
      team_a_controller: "manual",
      team_b_controller: "manual",
      execution_information_mode: "shared_obs",
    }),
    null,
  );
  for (const invalid of [
    {
      team_a_controller: "scenario_5",
      team_b_controller: "manual",
      execution_information_mode: "no_shared_obs",
    },
    {
      team_a_controller: "manual",
      team_b_controller: "scenario_5",
      execution_information_mode: "no_shared_obs",
    },
    {
      team_a_controller: "tdm_gamma",
      team_b_controller: "manual",
      execution_information_mode: "no_shared_obs",
    },
    {
      team_a_controller: "tdm_gamma",
      team_b_controller: "tdm_gamma",
      execution_information_mode: "no_shared_obs",
    },
    {
      team_a_controller: "manual",
      team_b_controller: "tdm_gamma",
      execution_information_mode: "no_shared_obs",
    },
    {
      team_a_controller: "scenario_3",
      team_b_controller: "manual",
      execution_information_mode: "shared_obs",
    },
    {
      team_a_controller: "manual",
      team_b_controller: "scenario_3",
      execution_information_mode: "shared_obs",
    },
    ...["team_a_controller", "team_b_controller"].map((team) => ({
      team_a_controller: "manual",
      team_b_controller: "manual",
      execution_information_mode: "no_shared_obs",
      [team]: "reactive_tdm",
    })),
  ]) {
    assert.equal(
      recordingReplacementCommand(frame, {
        command_type: "set_combat_configuration",
        ...invalid,
      }),
      null,
    );
  }
  assert.deepEqual(
    recordingReplacementCommand(frame, {
      command_type: "set_combat_configuration",
      team_a_controller: "random_valid",
      team_b_controller: "reactive_tdm",
      execution_information_mode: "shared_obs",
    }),
    {
      command_type: "set_combat_configuration",
      team_a_controller: "random_valid",
      team_b_controller: "reactive_tdm",
      execution_information_mode: "shared_obs",
    },
  );
  assert.equal(
    recordingReplacementCommand(frame, {
      command_type: "set_combat_configuration",
      team_a_controller: "unknown_policy",
      team_b_controller: "manual",
      execution_information_mode: "shared_obs",
    }),
    null,
  );
  const gammaCommand = {
    command_type: "set_combat_configuration",
    team_a_controller: "random_valid",
    team_b_controller: "tdm_gamma",
    execution_information_mode: "shared_obs",
  };
  const gammaReplacement = recordingReplacementCommand(frame, gammaCommand);
  assert.deepEqual(gammaReplacement, gammaCommand);
  assert.equal(Object.isFrozen(gammaReplacement), true);
  const gammaFrame = {
    ...frame,
    combat_configuration: {
      team_a_controller: "random_valid",
      team_b_controller: "tdm_gamma",
      execution_information_mode: "shared_obs",
    },
  };
  assert.equal(recordingReplacementCommand(gammaFrame, gammaCommand), null);
  assert.deepEqual(
    recordingReplacementCommand(gammaFrame, {
      ...gammaCommand,
      team_b_controller: "scenario_5",
    }),
    { ...gammaCommand, team_b_controller: "scenario_5" },
  );
  for (const teamA of ["scenario_5", "tdm_gamma"]) {
    const teamACommand = { ...gammaCommand, team_a_controller: teamA };
    const teamAReplacement = recordingReplacementCommand(frame, teamACommand);
    assert.deepEqual(teamAReplacement, teamACommand);
    assert.equal(Object.isFrozen(teamAReplacement), true);
  }
  for (const command of [
    keyboardCommand("w"),
    keyboardCommand("Enter"),
    { command_type: "battlefield_pointer", world_x: 1, world_y: 2 },
    { command_type: "roster_selection", role: "target", global_slot: 1 },
  ]) {
    assert.equal(recordingCommandDecision(frame, command).action, "allow");
  }
});

test("closed recording lifecycles fence scientific controls but retain presentation and recovery", () => {
  const failed = recordingFrame({ lifecycle: "persistence_failed" });
  for (const command of [
    keyboardCommand("w"),
    { command_type: "battlefield_pointer", world_x: 1, world_y: 2 },
  ]) {
    assert.equal(recordingCommandDecision(failed, command).action, "block");
  }
  assert.equal(
    recordingCommandDecision(failed, { command_type: "reset" }).action,
    "confirm",
  );
  assert.equal(
    recordingCommandDecision(recordingFrame({ lifecycle: "saved" }), {
      command_type: "reset",
    }).action,
    "allow",
  );
  for (const command of [
    { command_type: "roster_selection", role: "control", global_slot: 0 },
    { command_type: "save_and_replace", replacement: { command_type: "reset" } },
    { command_type: "set_view", view_mode: "pov" },
    { command_type: "set_preset", preset: "analysis" },
    keyboardCommand("g"),
    { command_type: "retry_save" },
    { command_type: "save_as", file_name: "copy.marlbg-replay.json" },
    { command_type: "exit" },
  ]) {
    assert.equal(recordingCommandDecision(failed, command).action, "allow");
  }
});

test("Save As emits only the protocol basename and rejects path-shaped input", () => {
  assert.deepEqual(recordingSaveAsCommand("copy.marlbg-replay.json"), {
    command_type: "save_as",
    file_name: "copy.marlbg-replay.json",
  });
  for (const value of [
    "../copy.marlbg-replay.json",
    "/tmp/copy.marlbg-replay.json",
    ".hidden.marlbg-replay.json",
    "copy replay.marlbg-replay.json",
    "copy.json",
    "",
  ]) {
    assert.equal(recordingSaveAsCommand(value), null);
  }
});

test("only an accepted Exit schedules shutdown", () => {
  const exit = { command_type: "exit" };
  assert.equal(
    commandResponseSchedulesShutdown(exit, { result: "shutdown_scheduled" }),
    true,
  );
  assert.equal(commandResponseSchedulesShutdown(exit, { result: "no_op" }), false);
  assert.equal(
    commandResponseSchedulesShutdown(
      { command_type: "reset" },
      {
        result: "shutdown_scheduled",
      },
    ),
    false,
  );
});

test("recording waits for accepted evidence but allows failure recovery and Exit", () => {
  for (const state of ["loading", "thinking", "finishing", "cancelling"]) {
    const frame = {
      ...recordingFrame(),
      system_operation: { operation_id: "pending", state },
    };
    for (const command_type of [
      "finish_and_review",
      "review_replay",
      "retry_save",
      "save_as",
      "save_and_replace",
      "confirm_discard_and_replace",
    ]) {
      assert.equal(recordingCommandDecision(frame, { command_type }).action, "block");
    }
    assert.equal(
      recordingCommandDecision(frame, { command_type: "exit" }).action,
      "allow",
    );
  }
  const failed = {
    ...recordingFrame(),
    fresh_launch_required: true,
    system_operation: { operation_id: "failed", state: "failed" },
  };
  assert.equal(
    recordingCommandDecision(failed, { command_type: "finish_and_review" }).action,
    "allow",
  );
});
