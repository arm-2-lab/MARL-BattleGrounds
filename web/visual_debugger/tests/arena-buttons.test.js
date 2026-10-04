/**
 * @file Check the arena's on-screen key stack (src/arena-buttons.js
 * wireArenaButtons, and its markup in index.html): wiring shows the group;
 * a click on Everyone In View presses "v" through the one command set,
 * reaching the registered camera_everyone_in_view handler with the button as
 * the target, no modifier keys and the caller's context plus source
 * "arena_buttons"; in the Replay Viewer a click on Battle Cam presses "b"
 * (camera_battle_cam) and a click on Clear Selection presses "Escape"
 * (replay_escape), while live play leaves both hidden and unwired; V's and
 * B's lit state (aria-pressed) follows the camera slot's data-camera
 * ("everyone_in_view", "battle_cam") on sync; in the Replay Viewer V and B
 * hide while the camera slot is hidden; destroy removes the click listeners
 * and hides the group; and index.html has the group, hidden, with the four
 * key-only buttons in order (V, B, F, Esc: each named by its aria-label, its
 * only visible text the keycap; B and Esc hidden until wired) and Match
 * Summary.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { wireArenaButtons } from "../src/arena-buttons.js";
import { registerLocalCommandHandler } from "../src/command-set.js";

class FakeNode {
  constructor() {
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    this.hidden = false;
    /** @type {Map<string, Function[]>} */
    this.listeners = new Map();
  }
  /** @param {string} name @param {string} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  /** @param {string} type @param {Function} handler */
  addEventListener(type, handler) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), handler]);
  }
  /** @param {string} type @param {Function} handler */
  removeEventListener(type, handler) {
    this.listeners.set(
      type,
      (this.listeners.get(type) ?? []).filter((entry) => entry !== handler),
    );
  }
  click() {
    for (const handler of this.listeners.get("click") ?? []) handler({ type: "click" });
  }
}

/** @param {"combat_debugger" | "replay_viewer"} product */
function mount(product) {
  const parts = {
    group: new FakeNode(),
    everyoneInView: new FakeNode(),
    battleCam: new FakeNode(),
    clearSelection: new FakeNode(),
    cameraSlot: new FakeNode(),
  };
  parts.battleCam.hidden = true;
  parts.clearSelection.hidden = true;
  parts.group.hidden = true;
  const wired = wireArenaButtons(
    /** @type {any} */ ({ ...parts, product, context: () => ({ interactive: true }) }),
  );
  return { parts, wired };
}

test("Everyone In View presses V through the one command set", () => {
  /** @type {any[]} */
  const calls = [];
  const off = registerLocalCommandHandler(
    "camera_everyone_in_view",
    (/** @type {any} */ call) => {
      calls.push(call);
    },
  );
  try {
    for (const product of /** @type {const} */ (["replay_viewer", "combat_debugger"])) {
      calls.length = 0;
      const { parts, wired } = mount(product);
      assert.equal(parts.group.hidden, false, "wiring shows the group");
      parts.everyoneInView.click();
      assert.deepEqual(
        calls.map((call) => [call.product, call.key, call.shiftKey, call.repeat]),
        [[product, "v", false, false]],
      );
      assert.equal(calls[0].event.target, parts.everyoneInView);
      assert.deepEqual(calls[0].context, {
        interactive: true,
        source: "arena_buttons",
      });
      wired.destroy();
      assert.equal(parts.group.hidden, true);
      parts.everyoneInView.click();
      assert.equal(calls.length, 1, "destroy removes the click listener");
    }
  } finally {
    off();
  }
});

test("Everyone In View and Battle Cam light while they are the camera and hide with the switch", () => {
  const { parts, wired } = mount("replay_viewer");
  assert.equal(parts.everyoneInView.getAttribute("aria-pressed"), "false");
  assert.equal(parts.battleCam.getAttribute("aria-pressed"), "false");
  assert.equal(parts.battleCam.hidden, false);
  assert.equal(parts.clearSelection.hidden, false);
  parts.cameraSlot.setAttribute("data-camera", "everyone_in_view");
  wired.sync();
  assert.equal(parts.everyoneInView.getAttribute("aria-pressed"), "true");
  assert.equal(parts.battleCam.getAttribute("aria-pressed"), "false");
  parts.cameraSlot.setAttribute("data-camera", "battle_cam");
  wired.sync();
  assert.equal(parts.everyoneInView.getAttribute("aria-pressed"), "false");
  assert.equal(parts.battleCam.getAttribute("aria-pressed"), "true");
  assert.equal(parts.everyoneInView.hidden, false);
  parts.cameraSlot.hidden = true;
  wired.sync();
  assert.equal(parts.everyoneInView.hidden, true, "Camera Framing off hides it");
  assert.equal(parts.battleCam.hidden, true, "Camera Framing off hides it");
  assert.equal(parts.clearSelection.hidden, false, "Esc does not need the camera");
  wired.destroy();
});

test("Battle Cam presses B and Clear Selection presses Esc, in the Replay Viewer only", () => {
  /** @type {any[]} */
  const calls = [];
  const offB = registerLocalCommandHandler(
    "camera_battle_cam",
    (/** @type {any} */ call) => {
      calls.push(call);
    },
  );
  const offEscape = registerLocalCommandHandler(
    "replay_escape",
    (/** @type {any} */ call) => {
      calls.push(call);
    },
  );
  try {
    const { parts, wired } = mount("replay_viewer");
    parts.battleCam.click();
    parts.clearSelection.click();
    assert.deepEqual(
      calls.map((call) => [
        call.product,
        call.key,
        call.event.target === parts.battleCam,
      ]),
      [
        ["replay_viewer", "b", true],
        ["replay_viewer", "Escape", false],
      ],
    );
    assert.equal(calls[1].event.target, parts.clearSelection);
    wired.destroy();
    calls.length = 0;
    const live = mount("combat_debugger");
    assert.equal(live.parts.battleCam.hidden, true, "no Battle Cam in live play");
    assert.equal(live.parts.clearSelection.hidden, true, "no Clear Selection key live");
    live.parts.battleCam.click();
    live.parts.clearSelection.click();
    assert.equal(calls.length, 0, "live play wires neither");
    live.wired.destroy();
  } finally {
    offB();
    offEscape();
  }
});

test("index.html has the hidden group with the four key-only buttons in order", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const group = /<div id="arena-buttons"[^>]*>([\s\S]*?)<\/div>/u.exec(html);
  assert.ok(group, "the group is in the page");
  assert.match(group[0], /^<div id="arena-buttons"[^>]*\shidden>/u);
  const keys = [
    ...group[1].matchAll(
      /<button[^>]*id="([^"]+)"[^>]*aria-label="([^"]+)"[^>]*>(.*?)<\/button>/gu,
    ),
  ].map((match) => [match[1], match[2], match[3]]);
  assert.deepEqual(keys, [
    [
      "arena-everyone-in-view-button",
      "Everyone In View",
      '<kbd aria-hidden="true">V</kbd>',
    ],
    ["arena-battle-cam-button", "Battle Cam", '<kbd aria-hidden="true">B</kbd>'],
    [
      "arena-frame-explanation-button",
      "Frame Explanation",
      '<kbd aria-hidden="true">F</kbd>',
    ],
    [
      "arena-clear-selection-button",
      "Clear Selection",
      '<kbd aria-hidden="true">Esc</kbd>',
    ],
  ]);
  assert.match(group[1], /id="arena-battle-cam-button"[^>]*\shidden>/u);
  assert.match(group[1], /id="arena-clear-selection-button"[^>]*\shidden>/u);
  assert.match(
    group[1],
    /id="arena-match-summary-button"[^>]*>Match Summary<\/button>/u,
  );
});
