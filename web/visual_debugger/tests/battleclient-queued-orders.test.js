/**
 * @file Check that BattleClient order edits made while an earlier order is
 * still out are offered for queuing instead of being lost:
 * - bindBattlefieldControls (src/controls.js), while orders are closed
 *   (isInteractive() false), builds the usual battlefield_pointer command for
 *   a Shift + click (a target pick) and a right click (clear the target) and
 *   offers it to onFencedCommand; the event's default is prevented only when
 *   onFencedCommand keeps it. A plain click is never offered and sends
 *   nothing. Nothing goes to onCommand while orders are closed.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { bindBattlefieldControls } from "../src/controls.js";

/**
 * @param {number} button
 * @param {boolean} shiftKey
 */
function pointerdown(button, shiftKey) {
  const event = new Event("pointerdown", { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    button: { value: button },
    clientX: { value: 30 },
    clientY: { value: 40 },
    shiftKey: { value: shiftKey },
    ctrlKey: { value: false },
    altKey: { value: false },
    metaKey: { value: false },
  });
  return event;
}

test("closed orders offer Shift + click and right click to the queue, never a plain click", () => {
  const battlefield = /** @type {any} */ (new EventTarget());
  battlefield.focus = () => {};
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
  /** @type {Record<string, unknown>[]} */
  const sent = [];
  /** @type {Record<string, unknown>[]} */
  const offered = [];
  let keep = true;
  bindBattlefieldControls({
    battlefield,
    toWorldPoint: (point) => ({ world_x: point.x / 10, world_y: point.y / 10 }),
    onCommand: (command) => {
      sent.push(command);
    },
    onHelp: () => {},
    onReleaseFocus: () => {},
    isInteractive: () => false,
    onFencedCommand: (command) => {
      offered.push(command);
      return keep;
    },
  });
  try {
    const pick = pointerdown(0, true);
    battlefield.dispatchEvent(pick);
    assert.equal(pick.defaultPrevented, true);
    const clear = pointerdown(2, false);
    battlefield.dispatchEvent(clear);
    assert.equal(clear.defaultPrevented, true);
    const plain = pointerdown(0, false);
    battlefield.dispatchEvent(plain);
    assert.equal(plain.defaultPrevented, false);
    assert.deepEqual(offered, [
      {
        command_type: "battlefield_pointer",
        world_x: 3,
        world_y: 4,
        button: "primary",
        shift_key: true,
        ctrl_key: false,
        alt_key: false,
        meta_key: false,
      },
      {
        command_type: "battlefield_pointer",
        world_x: 3,
        world_y: 4,
        button: "secondary",
        shift_key: false,
        ctrl_key: false,
        alt_key: false,
        meta_key: false,
      },
    ]);
    keep = false;
    const declined = pointerdown(0, true);
    battlefield.dispatchEvent(declined);
    assert.equal(declined.defaultPrevented, false);
    assert.equal(offered.length, 3);
    assert.deepEqual(sent, []);
  } finally {
    Reflect.set(globalThis, "DOMPoint", savedPoint);
  }
});
