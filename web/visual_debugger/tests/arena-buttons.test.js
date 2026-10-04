/**
 * @file Check the arena's two on-screen keys (src/arena-buttons.js
 * wireArenaButtons, and their markup in index.html): wiring shows the group;
 * a click on Everyone In View presses "v" through the one command set,
 * reaching the registered camera_everyone_in_view handler with the button as
 * the target, no modifier keys and the caller's context plus source
 * "arena_buttons"; its lit state (aria-pressed) follows the camera slot's
 * data-camera ("everyone_in_view") on sync; in the Replay Viewer it hides
 * while the camera slot is hidden; destroy removes the click listener and
 * hides the group; and index.html has the group, hidden, with both buttons
 * (Everyone In View and Frame Explanation) and their keycaps.
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
    cameraSlot: new FakeNode(),
  };
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

test("Everyone In View lights while it is the camera and hides with the switch", () => {
  const { parts, wired } = mount("replay_viewer");
  assert.equal(parts.everyoneInView.getAttribute("aria-pressed"), "false");
  parts.cameraSlot.setAttribute("data-camera", "everyone_in_view");
  wired.sync();
  assert.equal(parts.everyoneInView.getAttribute("aria-pressed"), "true");
  parts.cameraSlot.setAttribute("data-camera", "battle_cam");
  wired.sync();
  assert.equal(parts.everyoneInView.getAttribute("aria-pressed"), "false");
  assert.equal(parts.everyoneInView.hidden, false);
  parts.cameraSlot.hidden = true;
  wired.sync();
  assert.equal(parts.everyoneInView.hidden, true, "Camera Framing off hides it");
  wired.destroy();
});

test("index.html has the hidden group with both buttons and their keycaps", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const group = /<div id="arena-buttons"[^>]*>([\s\S]*?)<\/div>/u.exec(html);
  assert.ok(group, "the group is in the page");
  assert.match(group[0], /^<div id="arena-buttons"[^>]*\shidden>/u);
  assert.match(
    group[1],
    /id="arena-everyone-in-view-button" aria-keyshortcuts="V"[^>]*>Everyone In View <kbd aria-hidden="true">V<\/kbd>/u,
  );
  assert.match(
    group[1],
    /id="arena-frame-explanation-button" aria-keyshortcuts="F"[^>]*>Frame Explanation <kbd aria-hidden="true">F<\/kbd>/u,
  );
});
