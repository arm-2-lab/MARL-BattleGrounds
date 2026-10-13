/**
 * @file Check the one command set (plan section 4.5): the table matches the
 * plan row by row; a key does the same job in every client or nothing (no
 * BattleClient key has a different job in the Replay Viewer); B does nothing
 * outside the Replay Viewer; the arrows, V, B and F are never forwarded to
 * Python from the live battlefield, while N (one tick: Lock In, or the next
 * frame in scripted play) is; in the Replay Viewer N is the same as +1
 * ("Forward one tick, the same as +1"); V, B, F, G and N are the rows marked
 * skipsTextAndDialogs (they work from a focused button but not from a text field
 * or open dialog); V, B and F skip text fields and open dialogs and ignore Ctrl,
 * Alt and Meta; unapproved keys stay unbound; the
 * local-handler registry has one owner per handler, passes the call record and
 * honours a "not now" answer; and the help rows and text follow the table.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  COMMAND_PRODUCTS,
  COMMAND_SET,
  commandHelpRows,
  commandHelpText,
  forwardedFieldKeys,
  isTextOrDialogTarget,
  LOCAL_COMMAND_HANDLERS,
  PYTHON_HANDLER,
  registerLocalCommandHandler,
  resolveCommandKey,
  runLocalCommand,
  UNAPPROVED_KEYS,
} from "../src/command-set.js";

// The plan's table: id, keys, keycaps, shift, combat [job, help, route,
// handler, onField] or null, replay [job, help, route, handler] or null.
const PLAN_TABLE = [
  [
    "step",
    ["w", "a", "s", "d", "q", "e", "z", "c"],
    "W A S D Q E Z C",
    "any",
    ["step", "Step (staged)", "forwarded", "python", true],
    null,
  ],
  ["stay", ["x"], "X", "any", ["stay", "Stay", "forwarded", "python", true], null],
  [
    "combat_choice",
    ["0", "1", "2"],
    "0 / 1 / 2",
    "any",
    ["combat_choice", "No Combat / Basic / Ultimate", "forwarded", "python", true],
    null,
  ],
  [
    "go",
    [" "],
    "Space",
    "any",
    ["go", "Lock In (Policies Only: play one turn)", "forwarded", "python", true],
    ["go", "Play or pause", "local", "replay_play_pause"],
  ],
  [
    "lock_in",
    ["Enter"],
    "Enter",
    "any",
    ["lock_in", "Lock In", "forwarded", "python", true],
    null,
  ],
  [
    "one_tick",
    ["n"],
    "N",
    "any",
    [
      "one_tick",
      "One tick (Lock In; scripted play: next frame)",
      "forwarded",
      "python",
      true,
    ],
    ["one_tick", "Forward one tick, the same as +1", "local", "replay_one_tick"],
  ],
  [
    "tick_step",
    ["ArrowLeft", "ArrowRight"],
    "Arrow Left / Right",
    "up",
    null,
    ["tick_step", "Back / forward one tick (glides)", "local", "replay_tick_step"],
  ],
  [
    "tick_jump",
    ["ArrowLeft", "ArrowRight"],
    "Shift + Arrow Left / Right",
    "down",
    null,
    ["tick_jump", "Back / forward ten ticks (snaps)", "local", "replay_tick_jump"],
  ],
  ["arrow_up_down", ["ArrowUp", "ArrowDown"], "Arrow Up / Down", "any", null, null],
  [
    "restart",
    ["r"],
    "R",
    "any",
    ["restart", "Restart (Save, Discard, Cancel)", "forwarded", "python", true],
    null,
  ],
  [
    "ranges",
    ["g"],
    "G",
    "any",
    ["ranges", "Ranges", "forwarded", "python", true],
    ["ranges", "Ranges", "local", "ranges"],
  ],
  [
    "help",
    ["?"],
    "?",
    "any",
    ["help", "How to Play", "local", "help", true],
    ["help", "Help", "local", "help"],
  ],
  [
    "next_champion",
    ["Tab"],
    "Tab / Shift+Tab",
    "any",
    [
      "next_champion",
      "Your next / previous champion (Policies Only: every champion in sight)",
      "forwarded",
      "python",
      true,
    ],
    [
      "next_champion",
      "Next / previous agent while the field has focus",
      "local",
      "replay_next_agent",
    ],
  ],
  [
    "escape",
    ["Escape"],
    "Escape",
    "any",
    ["escape", "Clear target, then keys off", "forwarded", "python", true],
    ["escape", "Close, clear selection, leave the field", "local", "replay_escape"],
  ],
  [
    "everyone_in_view",
    ["v"],
    "V",
    "up",
    ["everyone_in_view", "Everyone in View", "local", "camera_everyone_in_view", true],
    ["everyone_in_view", "Everyone in View", "local", "camera_everyone_in_view"],
  ],
  [
    "battle_cam",
    ["b"],
    "B",
    "up",
    null,
    ["battle_cam", "Battle Cam", "local", "camera_battle_cam"],
  ],
  [
    "frame_explanation",
    ["f"],
    "F",
    "up",
    [
      "frame_explanation",
      "Frame Explanation (while waiting for orders)",
      "local",
      "frame_explanation",
      true,
    ],
    [
      "frame_explanation",
      "Frame Explanation (paused; in play it pauses first)",
      "local",
      "frame_explanation",
    ],
  ],
];

/** @param {string} key @param {Record<string, unknown>} [extra] */
function keyEvent(key, extra = {}) {
  return {
    key,
    shiftKey: false,
    ctrlKey: false,
    altKey: false,
    metaKey: false,
    repeat: false,
    ...extra,
  };
}

/** @param {string | null} match @param {object | null} [ownerDocument] */
function fakeTarget(match, ownerDocument = null) {
  return {
    closest: (/** @type {string} */ selector) =>
      match !== null && selector.includes(match) ? {} : null,
    ownerDocument,
  };
}

test("the command table encodes plan section 4.5 exactly", () => {
  assert.deepEqual(COMMAND_PRODUCTS, ["combat_debugger", "replay_viewer"]);
  assert.equal(PYTHON_HANDLER, "python");
  assert.deepEqual(
    COMMAND_SET.map(({ id, keys, keycaps, shift, jobs }) => [
      id,
      [...keys],
      keycaps,
      shift,
      jobs.combat_debugger === null
        ? null
        : [
            jobs.combat_debugger.job,
            jobs.combat_debugger.help,
            jobs.combat_debugger.route,
            jobs.combat_debugger.handler,
            jobs.combat_debugger.onField,
          ],
      jobs.replay_viewer === null
        ? null
        : [
            jobs.replay_viewer.job,
            jobs.replay_viewer.help,
            jobs.replay_viewer.route,
            jobs.replay_viewer.handler,
          ],
    ]),
    PLAN_TABLE,
  );
  assert.equal(Object.isFrozen(COMMAND_SET), true);
  for (const row of COMMAND_SET) {
    assert.equal(Object.isFrozen(row), true);
    assert.equal(Object.isFrozen(row.keys), true);
    assert.equal(Object.isFrozen(row.jobs), true);
    assert.equal(
      row.skipsTextAndDialogs,
      ["v", "b", "f", "g", "n"].includes(row.keys[0]),
    );
    for (const entry of Object.values(row.jobs)) {
      if (entry === null) continue;
      assert.equal(Object.isFrozen(entry), true);
      assert.equal(entry.route === "forwarded", entry.handler === PYTHON_HANDLER);
    }
  }
  assert.deepEqual(LOCAL_COMMAND_HANDLERS, [
    "camera_battle_cam",
    "camera_everyone_in_view",
    "frame_explanation",
    "help",
    "ranges",
    "replay_escape",
    "replay_next_agent",
    "replay_one_tick",
    "replay_play_pause",
    "replay_tick_jump",
    "replay_tick_step",
  ]);
});

test("no BattleClient key has a different job in the Replay Viewer", () => {
  for (const row of COMMAND_SET) {
    const combat = row.jobs.combat_debugger;
    const replay = row.jobs.replay_viewer;
    if (combat !== null && replay !== null) {
      assert.equal(combat.job, replay.job, row.id);
    }
  }
  // The same key and Shift state must also resolve to one job across products.
  const probes = COMMAND_SET.flatMap(({ keys }) =>
    keys.flatMap((key) => [keyEvent(key), keyEvent(key, { shiftKey: true })]),
  );
  for (const probe of probes) {
    const combat = resolveCommandKey("combat_debugger", probe);
    const replay = resolveCommandKey("replay_viewer", probe);
    if (combat !== null && replay !== null) {
      assert.equal(combat.entry.job, replay.entry.job, JSON.stringify(probe));
      assert.equal(combat.row, replay.row, JSON.stringify(probe));
    }
  }
  // Every key appears in exactly one row per Shift state.
  for (const probe of probes) {
    const rows = COMMAND_SET.filter(
      ({ keys, shift }) =>
        keys.includes(probe.key) &&
        !(shift === "up" && probe.shiftKey) &&
        !(shift === "down" && !probe.shiftKey),
    );
    assert.equal(rows.length <= 1, true, JSON.stringify(probe));
  }
});

test("B does nothing outside the Replay Viewer and never gets another job", () => {
  for (const key of ["b", "B"]) {
    for (const shiftKey of [false, true]) {
      assert.equal(
        resolveCommandKey("combat_debugger", keyEvent(key, { shiftKey })),
        null,
      );
    }
  }
  const replay = resolveCommandKey("replay_viewer", keyEvent("b"));
  assert.equal(replay?.entry.job, "battle_cam");
  assert.equal(
    COMMAND_SET.filter(({ keys }) => keys.includes("b"))
      .map(({ id }) => id)
      .join(),
    "battle_cam",
  );
  assert.equal(forwardedFieldKeys("combat_debugger").includes("b"), false);
  let calls = 0;
  const unregister = registerLocalCommandHandler("camera_battle_cam", () => {
    calls += 1;
  });
  try {
    assert.equal(runLocalCommand("combat_debugger", keyEvent("b")), false);
    assert.equal(calls, 0);
    assert.equal(runLocalCommand("replay_viewer", keyEvent("b")), true);
    assert.equal(calls, 1);
  } finally {
    unregister();
  }
});

test("arrows, V, B and F are never forwarded to Python from the live field; N is", () => {
  const forwarded = forwardedFieldKeys("combat_debugger");
  assert.deepEqual(forwarded, [
    "w",
    "a",
    "s",
    "d",
    "q",
    "e",
    "z",
    "c",
    "x",
    "0",
    "1",
    "2",
    " ",
    "Enter",
    "n",
    "r",
    "g",
    "Tab",
    "Escape",
  ]);
  for (const key of [
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "v",
    "b",
    "f",
  ]) {
    assert.equal(forwarded.includes(key), false, key);
  }
  for (const key of ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]) {
    for (const shiftKey of [false, true]) {
      assert.equal(
        resolveCommandKey("combat_debugger", keyEvent(key, { shiftKey })),
        null,
      );
    }
  }
  assert.deepEqual(forwardedFieldKeys("replay_viewer"), []);
  assert.equal(resolveCommandKey("replay_viewer", keyEvent("ArrowUp")), null);
  assert.equal(
    resolveCommandKey("replay_viewer", keyEvent("ArrowLeft"))?.entry.job,
    "tick_step",
  );
  assert.equal(
    resolveCommandKey("replay_viewer", keyEvent("ArrowRight", { shiftKey: true }))
      ?.entry.job,
    "tick_jump",
  );
});

test("modifiers, case and Shift select rows exactly", () => {
  for (const modifier of ["ctrlKey", "altKey", "metaKey"]) {
    for (const key of ["v", "b", "f", "w", "Tab", " ", "ArrowLeft"]) {
      for (const product of COMMAND_PRODUCTS) {
        assert.equal(
          resolveCommandKey(product, keyEvent(key, { [modifier]: true })),
          null,
        );
      }
    }
  }
  assert.equal(
    resolveCommandKey("combat_debugger", keyEvent("V"))?.entry.job,
    "everyone_in_view",
  );
  assert.equal(
    resolveCommandKey("combat_debugger", keyEvent("V", { shiftKey: true })),
    null,
  );
  assert.equal(
    resolveCommandKey("replay_viewer", keyEvent("F", { shiftKey: true })),
    null,
  );
  assert.equal(
    resolveCommandKey("combat_debugger", keyEvent("Tab", { shiftKey: true }))?.entry
      .job,
    "next_champion",
  );
  assert.equal(
    resolveCommandKey("combat_debugger", keyEvent("?", { shiftKey: true }))?.entry.job,
    "help",
  );
  assert.equal(resolveCommandKey("combat_debugger", keyEvent("F5")), null);
  assert.throws(() => resolveCommandKey("battle_client", keyEvent("v")), RangeError);
  for (const helper of [forwardedFieldKeys, commandHelpRows, commandHelpText]) {
    assert.throws(() => helper("devclient"), RangeError, helper.name);
  }
});

test("unapproved keys stay unbound in every client", () => {
  assert.deepEqual(UNAPPROVED_KEYS, [
    "h",
    "[",
    "]",
    "{",
    "}",
    "m",
    "<",
    ">",
    "Home",
    "End",
    "/",
  ]);
  for (const key of UNAPPROVED_KEYS) {
    assert.equal(
      COMMAND_SET.some(({ keys }) => keys.includes(key)),
      false,
      key,
    );
    for (const product of COMMAND_PRODUCTS) {
      for (const shiftKey of [false, true]) {
        assert.equal(resolveCommandKey(product, keyEvent(key, { shiftKey })), null);
      }
    }
  }
});

test("V, B and F skip text fields and open dialogs", () => {
  /** @type {any[]} */
  const calls = [];
  const unregister = registerLocalCommandHandler(
    "camera_everyone_in_view",
    (/** @type {any} */ call) => {
      calls.push(call);
    },
  );
  try {
    for (const blocker of [
      "input",
      "textarea",
      "select",
      "contenteditable",
      'role="textbox"',
      'role="combobox"',
      'role="spinbutton"',
      "dialog[open]",
    ]) {
      assert.equal(isTextOrDialogTarget(fakeTarget(blocker)), true, blocker);
      assert.equal(
        runLocalCommand(
          "replay_viewer",
          keyEvent("v", { target: fakeTarget(blocker) }),
        ),
        false,
        blocker,
      );
    }
    const modalDocument = {
      querySelector: (/** @type {string} */ s) => (s === "dialog:modal" ? {} : null),
    };
    assert.equal(isTextOrDialogTarget(fakeTarget(null, modalDocument)), true);
    const quietDocument = { querySelector: () => null };
    assert.equal(isTextOrDialogTarget(fakeTarget(null, quietDocument)), false);
    const brokenDocument = {
      querySelector: () => {
        throw new SyntaxError("unsupported selector");
      },
    };
    assert.equal(isTextOrDialogTarget(fakeTarget(null, brokenDocument)), false);
    for (const target of [null, undefined, {}, "battlefield"]) {
      assert.equal(isTextOrDialogTarget(target), false);
    }
    assert.deepEqual(calls, []);
    assert.equal(
      runLocalCommand(
        "replay_viewer",
        keyEvent("v", { target: fakeTarget(null, quietDocument) }),
      ),
      true,
    );
    assert.equal(calls.length, 1);
  } finally {
    unregister();
  }
});

test("the local-handler registry has one owner per job and passes the call record", () => {
  assert.throws(() => registerLocalCommandHandler("python", () => {}), RangeError);
  assert.throws(() => registerLocalCommandHandler("camera_zoom", () => {}), RangeError);
  assert.throws(
    () => registerLocalCommandHandler("frame_explanation", "nope"),
    TypeError,
  );
  /** @type {any[]} */
  const calls = [];
  let answer = /** @type {boolean | void} */ (undefined);
  const unregister = registerLocalCommandHandler(
    "frame_explanation",
    (/** @type {any} */ call) => {
      calls.push(call);
      return answer;
    },
  );
  try {
    assert.throws(
      () => registerLocalCommandHandler("frame_explanation", () => {}),
      /already registered/u,
    );
    const event = keyEvent("F", { repeat: true });
    assert.equal(
      runLocalCommand("combat_debugger", event, { interactive: false }),
      true,
    );
    assert.equal(calls.length, 1);
    const [call] = calls;
    assert.equal(Object.isFrozen(call), true);
    assert.equal(call.product, "combat_debugger");
    assert.equal(call.job, "frame_explanation");
    assert.equal(call.key, "f");
    assert.equal(call.shiftKey, false);
    assert.equal(call.repeat, true);
    assert.equal(call.event, event);
    assert.deepEqual(call.context, { interactive: false });
    answer = false;
    assert.equal(runLocalCommand("replay_viewer", keyEvent("f")), false);
    assert.equal(calls.length, 2);
    assert.equal(calls[1].product, "replay_viewer");
    // Forwarded keys never reach a local handler.
    assert.equal(runLocalCommand("combat_debugger", keyEvent("w")), false);
    assert.equal(runLocalCommand("combat_debugger", keyEvent("n")), false);
    assert.equal(calls.length, 2);
  } finally {
    unregister();
    unregister();
  }
  assert.equal(runLocalCommand("combat_debugger", keyEvent("f")), false);
  const second = registerLocalCommandHandler("frame_explanation", () => {});
  second();
});

test("help rows and text are generated from the table for each product", () => {
  const combat = commandHelpRows("combat_debugger");
  const replay = commandHelpRows("replay_viewer");
  assert.equal(Object.isFrozen(combat), true);
  assert.deepEqual(
    combat.map(({ id }) => id),
    PLAN_TABLE.filter((row) => row[4] !== null).map(([id]) => id),
  );
  assert.deepEqual(
    replay.map(({ id }) => id),
    PLAN_TABLE.filter((row) => row[5] !== null).map(([id]) => id),
  );
  assert.equal(
    combat.some(({ id }) => id === "battle_cam"),
    false,
  );
  assert.equal(
    combat.some(({ id }) => id.startsWith("tick_")),
    false,
  );
  assert.deepEqual(combat[0], {
    id: "step",
    keycaps: "W A S D Q E Z C",
    help: "Step (staged)",
    job: "step",
  });
  const text = commandHelpText("replay_viewer");
  assert.equal(text.split("\n").length, replay.length);
  assert.match(text, /^Space: Play or pause$/mu);
  assert.match(text, /^B: Battle Cam$/mu);
  assert.match(commandHelpText("combat_debugger"), /^V: Everyone in View$/mu);
  assert.doesNotMatch(commandHelpText("combat_debugger"), /Battle Cam|Arrow/u);
});
