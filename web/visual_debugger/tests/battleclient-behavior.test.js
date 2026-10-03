/**
 * @file Check the BattleClient's own behavior contracts that need no server.
 * (1) statusSentence writes the vision page's exact words for steps, actions,
 * targets (Auto, Press 2 For, Nothing Allowed), refusals ("needs a target
 * first", "not allowed on this target right now", "It is ready in N ticks"),
 * focus, Lock In, views, ranges, Restart, Run and Pause. (The first sentence
 * and the closed page's words have other owners: tactics-turn.js
 * startGreeting and closedSentence.)
 * (2) On the focused battlefield ? opens How To Play at any time (also while
 * orders are closed, as at game over) and sends nothing; a key that
 * allowsWhileClosed accepts (R and G at game over) still reaches onCommand
 * while orders are closed; any other key then goes to onFencedCommand.
 * (3) main.js wires the controls the vision page shows: How To Play on #htp,
 * the Recording popover's Finish And Save, Open Replay kept visible, the
 * Restart dialog's Save focus and "Restart cancelled" sentence (read from
 * its source), and the pill, popover and seal words, which main.js imports
 * from their one owner, recordingWords in tactics-turn.js.
 * (4) audienceSwitchKeepsTick says a view switch shows the tick already
 * played at its end (no replay, orders stay open) only for the same session
 * and tick with a different audience, never for a new tick, a restart, the
 * first tick or the same audience; main.js then skips the replay.
 * (5) Policies Only Run in main.js (read from its source): Run is paced by
 * the page with quiet Play One Turn commands (the line keeps "Running."); a
 * turn already out is waited for after Pause, so the loop ends with "Paused
 * after the current turn." only once that turn has landed and settled; a
 * quick Pause then Run never starts a second loop; Python's "The Systems are
 * choosing this turn." is said as the Lock In sentence; and an overtaken
 * Lock In answer keeps that sentence instead of a technical warning.
 * (6) A fresh unmodified G pressed while a request is out (as right after the
 * click that turns keys on) is queued in main.js, not dropped.
 * (7) While the page's own Run plays (also after Pause until its last turn has
 * settled), pageRunOwnsLockIn claims Space, Enter and N (any case, any
 * modifiers) and nothing else, and dispatchCommand drops such a command
 * silently unless it is Run's own quiet turn, so no second turn is sent and no
 * glide is cut short, as on the vision page.
 * (8) N is one tick, a Lock In key like Space: the focused battlefield
 * forwards N; main.js names the Lock In keys in one helper (isLockInKeyName:
 * Enter, Space, N) used by isFreshUnmodifiedEnter, the fenced-input queue and
 * pageRunOwnsLockIn; the browser's mode gate (commandRole) gives N Space's
 * "interactive-submit" role, so N is refused at game over with Space's
 * sentence; and liveKeyAllowedWhileClosed passes R and G but never Space
 * while orders are closed, and N only in scripted inspection, where it is the
 * next frame (the deck's Lock In button and the Watching bar's Play One Turn
 * send N there).
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  audienceSwitchKeepsTick,
  bindBattlefieldControls,
  isDebuggerKey,
  STEP_NAMES,
  statusSentence,
} from "../src/controls.js";
import { commandRole, recordingWords } from "../src/tactics-turn.js";

const MAIN = readFileSync(new URL("../src/main.js", import.meta.url), "utf8");

/** @param {string} name A top-level function declared in main.js. */
function mainFunction(name) {
  const source = new RegExp(
    `\\nfunction ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n\\}`,
    "u",
  ).exec(MAIN)?.[0];
  assert.ok(source, `main.js defines ${name}`);
  return source;
}

/** @param {string} key @param {boolean} [shiftKey] */
function keydown(key, shiftKey = false) {
  const event = new Event("keydown", { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    key: { value: key },
    repeat: { value: false },
    shiftKey: { value: shiftKey },
    ctrlKey: { value: false },
    altKey: { value: false },
    metaKey: { value: false },
  });
  return event;
}

test("status sentences use the vision page's exact words", () => {
  assert.deepEqual(STEP_NAMES.slice(5), [
    "North-East",
    "North-West",
    "South-East",
    "South-West",
  ]);
  /** @type {Array<[import("../src/controls.js").StatusEvent, string]>} */
  const cases = [
    [{ kind: "step", move: 1 }, "Step: North."],
    [{ kind: "step", move: 6 }, "Step: North-West."],
    [{ kind: "step", move: 0 }, "Step: Stay."],
    [
      { kind: "no_combat" },
      "No Combat: your champion only moves this tick. Your target stays picked.",
    ],
    [
      { kind: "armed", ability: "Charge", target: "Mage (ID 5)" },
      "Charge is armed on the Mage (ID 5).",
    ],
    [
      { kind: "burst_armed" },
      "Burst is armed. It has no target, so your target was cleared.",
    ],
    [
      { kind: "needs_target", ability: "Shield Bash" },
      "Shield Bash needs a target first. Your order is unchanged.",
    ],
    [
      { kind: "not_allowed", ability: "Shield Bash", onTarget: true, readyIn: 0 },
      "Shield Bash is not allowed on this target right now. Your order is unchanged.",
    ],
    [
      { kind: "not_allowed", ability: "Burst", onTarget: false, readyIn: 30 },
      "Burst is not allowed right now. Your order is unchanged. It is ready in 30 ticks.",
    ],
    [
      { kind: "not_allowed", ability: "Charge", onTarget: true, readyIn: 1 },
      "Charge is not allowed on this target right now. Your order is unchanged. It is ready in 1 tick.",
    ],
    [
      {
        kind: "target",
        target: "Mage (ID 5)",
        basic: "Shield Bash",
        ultimate: "Charge",
        auto: false,
        ultimateAllowed: true,
      },
      "Target: Mage (ID 5). Shield Bash is not allowed on it, so your action is No Combat. Press 2 for Charge.",
    ],
    [
      {
        kind: "target",
        target: "Hunter (ID 7)",
        basic: "Shield Bash",
        ultimate: "Charge",
        auto: false,
        ultimateAllowed: false,
      },
      "Target: Hunter (ID 7). Nothing is allowed on it right now, so your action is No Combat.",
    ],
    [
      {
        kind: "target",
        target: "Mage (ID 0)",
        basic: "Arcane Blast",
        ultimate: "Burst",
        auto: true,
        ultimateAllowed: false,
      },
      "Target: Mage (ID 0). Arcane Blast armed itself, because it is allowed (Auto).",
    ],
    [
      { kind: "cleared", keysOff: true },
      "Target cleared. Keys are off until you click the arena again.",
    ],
    [{ kind: "cleared", keysOff: false }, "Target cleared."],
    [{ kind: "focus_own", className: "Warrior" }, "Giving orders to your Warrior."],
    [
      { kind: "look", target: "Hunter (ID 2)", controller: "Recurrent MAPPO" },
      "Looking at the Hunter (ID 2), played by Recurrent MAPPO.",
    ],
    [
      { kind: "policy_champion", watch: false },
      "This champion is played by a policy. You can look, not command. Pick one of yours to give orders.",
    ],
    [
      { kind: "policy_champion", watch: true },
      "Nobody is yours this game. You can look, not command.",
    ],
    [
      { kind: "stopped" },
      "A fresh start is needed, so orders cannot change. You can look around, save or exit.",
    ],
    [{ kind: "lock_in", watch: false }, "Orders sent. The policies choose next."],
    [{ kind: "lock_in", watch: true }, "Playing one turn."],
    [
      { kind: "view", view: "oracle", unseen: 0, viewers: "your team", watch: false },
      "Oracle View: every champion is shown, even those your team cannot see.",
    ],
    [
      { kind: "view", view: "pov", unseen: 2, viewers: "your team", watch: false },
      "Agent POV: you see only what your team sees. 2 enemy champions are out of sight.",
    ],
    [
      { kind: "view", view: "pov", unseen: 1, viewers: "Team B", watch: true },
      "Agent POV: you see only what Team B sees. 1 champion is out of sight.",
    ],
    [
      { kind: "view", view: "pov", unseen: 0, viewers: "your team", watch: false },
      "Agent POV: you see only what your team sees. Every champion is in sight.",
    ],
    [{ kind: "ranges", shown: false }, "Reach rings hidden."],
    [{ kind: "restart_cancelled" }, "Restart cancelled. The game goes on."],
    [
      { kind: "work_cancelled" },
      "Policy work cancelled. The game so far is kept. A fresh start from Python is needed to play on.",
    ],
    [{ kind: "pausing" }, "Pausing after the current turn."],
    [{ kind: "paused" }, "Paused after the current turn."],
  ];
  for (const [event, words] of cases) {
    assert.equal(statusSentence(event), words, JSON.stringify(event));
  }
  assert.equal(statusSentence(/** @type {any} */ ({ kind: "unknown" })), "");
});

test("? opens How To Play at any time and R and G pass while orders are closed", () => {
  const battlefield = /** @type {any} */ (new EventTarget());
  let interactive = false;
  let help = 0;
  /** @type {Record<string, unknown>[]} */
  const sent = [];
  /** @type {Record<string, unknown>[]} */
  const fenced = [];
  bindBattlefieldControls({
    battlefield,
    toWorldPoint: () => null,
    onCommand: (command) => {
      sent.push(command);
    },
    onHelp: () => {
      help += 1;
    },
    onReleaseFocus: () => {},
    isInteractive: () => interactive,
    onFencedCommand: (command) => {
      fenced.push(command);
      return true;
    },
    allowsWhileClosed: (command) =>
      typeof command.key === "string" && ["r", "g"].includes(command.key.toLowerCase()),
  });
  const question = keydown("?", true);
  battlefield.dispatchEvent(question);
  assert.equal(help, 1);
  assert.equal(question.defaultPrevented, true);
  for (const key of ["r", "g"]) {
    const event = keydown(key);
    battlefield.dispatchEvent(event);
    assert.equal(event.defaultPrevented, true, key);
  }
  assert.deepEqual(
    sent.map((command) => command.key),
    ["r", "g"],
  );
  battlefield.dispatchEvent(keydown("w"));
  assert.deepEqual(
    fenced.map((command) => command.key),
    ["w"],
  );
  interactive = true;
  battlefield.dispatchEvent(keydown("?", true));
  assert.equal(help, 2);
  assert.equal(sent.length, 2);
});

test("main.js wires the vision's help, recording and Restart controls", () => {
  assert.match(MAIN, /howToPlay: requiredElement\("htp"\)/u);
  assert.match(MAIN, /onHelp: \(\) => toggleHowToPlay\(\)/u);
  assert.match(MAIN, /allowsWhileClosed: liveKeyAllowedWhileClosed/u);
  for (const button of [
    "elements.recordingFinishButton",
    "elements.recordingPopFinishButton",
    "elements.stoppedSaveButton",
  ]) {
    assert.ok(MAIN.includes(button), button);
  }
  assert.match(MAIN, /recordingPopover\.hidePopover\(\)/u);
  assert.match(MAIN, /recordingSaveReplaceButton\.focus\(/u);
  assert.match(MAIN, /kind: "restart_cancelled"/u);
  assert.match(MAIN, /\n {2}recordingWords,\n\} from "\.\/tactics-turn\.js";/u);
  assert.equal(
    recordingWords({ lifecycle: "persistence_failed" }, 91, 300).pill,
    "Save Failed · Retry",
  );
  const live = recordingWords(
    { lifecycle: "recording", captured_transition_count: 0 },
    91,
    300,
  );
  assert.equal(live.pill, "Recording · Tick 91");
  assert.equal(live.popover, "0 Ticks Kept Since Tick 91");
});

test("a view switch keeps the tick already played instead of replaying it", () => {
  const oracle = JSON.stringify(["s1", "oracle", null]);
  const pov = JSON.stringify(["s1", "agent_pov", "team-a"]);
  const shown = { audienceKey: pov, simulatorStep: 8 };
  assert.equal(
    audienceSwitchKeepsTick(shown, { audienceKey: oracle, simulatorStep: 8 }),
    true,
  );
  assert.equal(
    audienceSwitchKeepsTick(shown, {
      audienceKey: JSON.stringify(["s1", "agent_pov", "team-b"]),
      simulatorStep: 8,
    }),
    true,
  );
  // A new tick plays; the same audience is left to the choreographer.
  assert.equal(
    audienceSwitchKeepsTick(shown, { audienceKey: oracle, simulatorStep: 9 }),
    false,
  );
  assert.equal(
    audienceSwitchKeepsTick(shown, { audienceKey: pov, simulatorStep: 8 }),
    false,
  );
  // A restart (new session), the first tick shown and broken keys play as before.
  assert.equal(
    audienceSwitchKeepsTick(shown, {
      audienceKey: JSON.stringify(["s2", "oracle", null]),
      simulatorStep: 8,
    }),
    false,
  );
  assert.equal(
    audienceSwitchKeepsTick(null, { audienceKey: oracle, simulatorStep: 8 }),
    false,
  );
  assert.equal(audienceSwitchKeepsTick(shown, null), false);
  assert.equal(
    audienceSwitchKeepsTick(
      { audienceKey: "not json", simulatorStep: 8 },
      { audienceKey: oracle, simulatorStep: 8 },
    ),
    false,
  );
  // main.js settles the replay right after presenting the tick.
  assert.match(
    MAIN,
    /choreographer\.presentFrame\([\s\S]{0,160}\);\s*settleLiveAudienceSwitch\(choreographer\.plan\);/u,
  );
});

test("main.js runs Policies Only turns one at a time in the vision's words", () => {
  assert.match(
    MAIN,
    /dispatchCommand\(keyboardCommand\("Enter"\), \{\s*source: "control",\s*quiet: true,\s*keepPresentation: true,\s*\}\)/u,
  );
  assert.match(MAIN, /liveRunMayContinue\(\{ ignorePause: true \}\)/u);
  assert.match(MAIN, /if \(!liveRunLoopActive\) void runLiveTurns\(\);/u);
  assert.match(
    MAIN,
    /liveRunLoopActive = false;\s*liveRunHoldsNextTick = false;\s*const paused = !state\.pageRun;/u,
  );
  assert.match(
    MAIN,
    /notice === "The Systems are choosing this turn\."\) \{\s*return statusSentence\(\{ kind: "lock_in", watch: facts\?\.mode === "watch" \}\);/u,
  );
  assert.match(
    MAIN,
    /installOutcome\.resynchronized\s*\? isSubmissionCommand\(command\)\s*\? statusSentence\(\{ kind: "lock_in"/u,
  );
  assert.doesNotMatch(
    MAIN,
    /mixed presentation candidate was discarded, and one fresh GET pair was installed\."\s*: \(notice \?\? "Duplicate command recognized/u,
  );
  assert.equal(statusSentence({ kind: "lock_in", watch: true }), "Playing one turn.");
  assert.equal(
    statusSentence({ kind: "lock_in", watch: false }),
    "Orders sent. The policies choose next.",
  );
  assert.equal(statusSentence({ kind: "paused" }), "Paused after the current turn.");
});

test("main.js queues G pressed while a request is out", () => {
  assert.match(
    MAIN,
    /const rangesKey =\s*command\.command_type === "keyboard" &&[\s\S]{0,200}command\.key\.toLowerCase\(\) === "g"/u,
  );
  assert.match(
    MAIN,
    /\(targetEdit \|\| rangesKey \|\| isFreshDeferredDraft\(command\)\)/u,
  );
});

test("Space, Enter and N are ignored while the page's own Run plays", () => {
  const source = /function pageRunOwnsLockIn\(command\) \{[\s\S]*?\n\}/u.exec(
    MAIN,
  )?.[0];
  assert.ok(source, "main.js defines pageRunOwnsLockIn");
  /** @type {(active: boolean) => (command: Record<string, unknown>) => boolean} */
  const owns = (active) =>
    new Function(
      "liveRunLoopActive",
      `${mainFunction("isLockInKeyName")}\n${source}\nreturn pageRunOwnsLockIn;`,
    )(active);
  const running = owns(true);
  for (const key of ["Enter", "enter", " ", "n", "N"]) {
    assert.equal(running({ command_type: "keyboard", key }), true, key);
    assert.equal(
      running({ command_type: "keyboard", key, shift_key: true }),
      true,
      key,
    );
  }
  for (const key of ["w", "x", "1", "g", "Escape", "r", "b", "Spacebar"]) {
    assert.equal(running({ command_type: "keyboard", key }), false, key);
  }
  assert.equal(
    running({ command_type: "battlefield_pointer", button: "primary" }),
    false,
  );
  assert.equal(running({ command_type: "run" }), false);
  assert.equal(owns(false)({ command_type: "keyboard", key: " " }), false);
  assert.equal(owns(false)({ command_type: "keyboard", key: "Enter" }), false);
  assert.equal(owns(false)({ command_type: "keyboard", key: "n" }), false);
  // dispatchCommand drops them silently, before the focus queue and the busy
  // warning, unless the command is Run's own quiet turn.
  assert.match(
    MAIN,
    /if \(!quiet && pageRunOwnsLockIn\(command\)\) \{[^}]*?return;\s*\}\s*if \(state\.busy && !state\.shuttingDown && retainFocusChange\(command\)\) \{[^}]*?\}\s*if \(state\.busy \|\| state\.shuttingDown\)/u,
  );
});

test("N is forwarded from the field and is a Lock In key like Space in main.js", () => {
  assert.equal(isDebuggerKey({ key: "n" }), true);
  assert.equal(isDebuggerKey({ key: "N", shiftKey: true }), true);
  /** @type {(command: Record<string, unknown>) => boolean} */
  const fresh = new Function(
    `${mainFunction("isLockInKeyName")}\n${mainFunction("isFreshUnmodifiedEnter")}\nreturn isFreshUnmodifiedEnter;`,
  )();
  const plain = {
    command_type: "keyboard",
    shift_key: false,
    ctrl_key: false,
    alt_key: false,
    meta_key: false,
    repeat: false,
  };
  assert.equal(fresh({ ...plain, key: "Enter" }), true);
  assert.equal(fresh({ ...plain, key: " " }), true);
  assert.equal(fresh({ ...plain, key: "n" }), true);
  assert.equal(fresh({ ...plain, key: "N" }), true);
  assert.equal(fresh({ ...plain, key: "n", repeat: true }), false);
  assert.equal(fresh({ ...plain, key: "n", shift_key: true }), false);
  assert.equal(fresh({ ...plain, key: " ", repeat: true }), false);
  assert.equal(fresh({ ...plain, key: "Enter", ctrl_key: true }), false);
  assert.equal(fresh({ ...plain, key: "w" }), false);
  // The fenced-input path keeps only a following Lock In key.
  assert.match(
    MAIN,
    /if \(command\.command_type !== "keyboard" \|\| !isLockInKeyName\(command\.key\)\) \{\s*return false;\s*\}/u,
  );
  /** @type {(key: unknown) => boolean} */
  const lockKey = new Function(
    `${mainFunction("isLockInKeyName")}\nreturn isLockInKeyName;`,
  )();
  for (const key of ["Enter", "enter", " ", "n", "N"]) {
    assert.equal(lockKey(key), true, key);
  }
  for (const key of ["w", "g", "r", "Escape", "Spacebar", "m", 7, null]) {
    assert.equal(lockKey(key), false, String(key));
  }
  // The browser's mode gate uses commandRole, which gives N Space's role, so
  // N is refused at game over with Space's sentence.
  assert.equal(commandRole({ ...plain, key: "n" }), "interactive-submit");
  assert.equal(commandRole({ ...plain, key: "N" }), "interactive-submit");
  assert.equal(commandRole({ ...plain, key: " " }), "interactive-submit");
  assert.equal(commandRole({ ...plain, key: "w" }), "draft");
  assert.match(MAIN, /const role = commandRole\(command\);/u);
});

test("N passes the closed-orders key gate only in scripted inspection; R and G do", () => {
  /** @type {(scripted: boolean) => (command: Record<string, unknown>) => boolean} */
  const gate = (scripted) => {
    const stubs = {
      isReplayMode: () => false,
      isAuthorizedPresentationFrame: () => true,
      installedAuthorityIsCoherent: () => true,
      liveScriptedInspectionOnly: () => scripted,
      state: { presentation: {}, frame: {}, busy: false },
    };
    return new Function(
      ...Object.keys(stubs),
      `${mainFunction("liveKeyAllowedWhileClosed")}\nreturn liveKeyAllowedWhileClosed;`,
    )(...Object.values(stubs));
  };
  const allowed = gate(false);
  const scriptedAllowed = gate(true);
  /** @type {(k: string, extra?: Record<string, unknown>) => Record<string, unknown>} */
  const key = (k, extra = {}) => ({
    command_type: "keyboard",
    key: k,
    shift_key: false,
    ctrl_key: false,
    alt_key: false,
    meta_key: false,
    repeat: false,
    ...extra,
  });
  assert.equal(allowed(key("n")), false);
  assert.equal(allowed(key("N")), false);
  assert.equal(allowed(key(" ")), false);
  assert.equal(allowed(key("g")), true);
  assert.equal(allowed(key("r")), true);
  assert.equal(allowed(key("r", { shift_key: true })), false);
  assert.equal(allowed(key("g", { ctrl_key: true })), false);
  // In scripted inspection plain N is the next frame, so it passes there.
  assert.equal(scriptedAllowed(key("n")), true);
  assert.equal(scriptedAllowed(key("n", { shift_key: true })), false);
  assert.equal(scriptedAllowed(key(" ")), false);
  assert.equal(scriptedAllowed(key("g")), true);
  // Scripted inspection still lets N (its next frame) and G through.
  assert.match(MAIN, /command\.key !== "n" && command\.key !== "g"/u);
});
