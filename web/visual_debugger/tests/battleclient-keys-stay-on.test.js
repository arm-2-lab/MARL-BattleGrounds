/**
 * @file Check the keys and clicks rules of the BattleClient (owner, 3 October
 * 2026):
 * - keepKeysOnForTopBarPress (main.js) prevents the mouse press default on a
 *   button inside the top bar, on the arena's two on-screen keys (Everyone In
 *   View and Frame Explanation), on the game-over Verdict card's two buttons
 *   and on the Match Summary sheet's Close (KEYS_KEPT_BUTTONS) only while
 *   the battlefield has focus and only
 *   outside the Replay Viewer, so keys stay on; it is installed on the
 *   document in the capture phase. giveKeysBackAfterTopBarDialog then gives
 *   focus back to the battlefield after a dialog closes, once, only when that
 *   press kept keys on and focus sits on the button or the page body.
 * - A view switch keeps keys on: savePresentationPreferenceBeforeClear moves
 *   focus to Help only for focus inside the battlefield or a disclosure
 *   panel, never for the battlefield element itself.
 * - unpinArenaCardAfterClick drops a card pinned on a mark inside the
 *   battlefield (hide, then refresh so the hover card stays) and keeps pins
 *   outside it and in the Replay Viewer; it listens to battlefield clicks.
 * - paintKeysState makes a stale keys-off status line true when keys turn
 *   on: the keys-off sentence empties, "Target cleared. Keys are off ..."
 *   becomes "Target cleared.", any other message and keys off stay. Once
 *   Exit was accepted (state.shuttingDown) the chip reads "Keys Off" with
 *   the battlefield focused, and the arena has no keys-on frame.
 * - deck.css lets Watch Only cover every deck column but Lock In, and never
 *   the deck's header line.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { statusSentence } from "../src/controls.js";

const MAIN = readFileSync(new URL("../src/main.js", import.meta.url), "utf8");
const DECK_CSS = readFileSync(
  new URL("../src/styles/battleclient/deck.css", import.meta.url),
  "utf8",
);

/** @param {string} name @returns {string} */
function functionSource(name) {
  const source = new RegExp(
    `\\nfunction ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n\\}`,
    "u",
  ).exec(MAIN)?.[0];
  assert.ok(source, `main.js defines ${name}`);
  return source;
}

const KEYS_KEPT_SOURCE =
  /\nconst KEYS_KEPT_BUTTONS =\s*"[^"]*";/u.exec(MAIN)?.[0] ?? "";

class FakeElement {
  /** @param {string | null} [match] @param {FakeElement[]} [inside] */
  constructor(match = null, inside = []) {
    this.match = match;
    this.inside = inside;
    /** @type {(() => void) | undefined} */
    this.focus = undefined;
  }

  /** @param {string} selector A selector or a comma-separated selector list. */
  closest(selector) {
    return selector
      .split(",")
      .map((one) => one.trim())
      .includes(/** @type {string} */ (this.match))
      ? this
      : null;
  }

  /** @param {unknown} other */
  contains(other) {
    return other === this || this.inside.includes(/** @type {FakeElement} */ (other));
  }
}

/**
 * @param {{replay?: boolean, active: unknown, battlefield: FakeElement}} setup
 */
function topBar({ replay = false, active, battlefield }) {
  const doc = { activeElement: active, body: {} };
  /** @type {Array<() => void>} */
  const timers = [];
  /** @type {unknown[]} */
  const focused = [];
  battlefield.focus = () => {
    focused.push(battlefield);
  };
  const [press, close] = new Function(
    "isReplayMode",
    "document",
    "elements",
    "Element",
    "HTMLDialogElement",
    "window",
    `let topBarPressKeptKeys = null;
${KEYS_KEPT_SOURCE}
${functionSource("keepKeysOnForTopBarPress")}
${functionSource("giveKeysBackAfterTopBarDialog")}
return [keepKeysOnForTopBarPress, giveKeysBackAfterTopBarDialog];`,
  )(() => replay, doc, { battlefield }, FakeElement, FakeDialog, {
    setTimeout: (/** @type {() => void} */ run) => timers.push(run),
  });
  return {
    /** @type {(event: Record<string, unknown>) => void} */ press,
    /** @type {(event: Record<string, unknown>) => void} */ close,
    doc,
    timers,
    focused,
  };
}

class FakeDialog {}

/** @param {unknown} target */
function pressOn(target) {
  const event = { target, prevented: false, preventDefault: () => {} };
  event.preventDefault = () => {
    event.prevented = true;
  };
  return event;
}

test("a top-bar button press keeps keys on only while they are on", () => {
  const battlefield = new FakeElement();
  const button = new FakeElement(".app-header button");
  const elsewhere = new FakeElement();
  const on = topBar({ active: battlefield, battlefield });
  const press = pressOn(button);
  on.press(press);
  assert.equal(press.prevented, true, "the button does not take focus");
  const other = pressOn(elsewhere);
  on.press(other);
  assert.equal(other.prevented, false, "a press outside the top bar is untouched");
  const off = pressOn(button);
  topBar({ active: new FakeElement(), battlefield }).press(off);
  assert.equal(off.prevented, false, "keys already off stay as they are");
  const replay = pressOn(button);
  topBar({ replay: true, active: battlefield, battlefield }).press(replay);
  assert.equal(replay.prevented, false, "the Replay Viewer is left alone");
  for (const selector of [
    ".arena-buttons button",
    ".herald-verdict-actions__button",
    ".match-summary-sheet__close",
  ]) {
    const verdictPress = pressOn(new FakeElement(selector));
    on.press(verdictPress);
    assert.equal(verdictPress.prevented, true, `${selector} keeps keys on`);
  }
  assert.match(
    MAIN,
    /document\.addEventListener\("mousedown", keepKeysOnForTopBarPress, true\);/u,
  );
  assert.match(
    MAIN,
    /document\.addEventListener\("close", giveKeysBackAfterTopBarDialog, true\);/u,
  );
});

test("a dialog opened from the top bar gives the keys back when it closes", () => {
  const battlefield = new FakeElement();
  const button = new FakeElement(".app-header button");
  const page = topBar({ active: battlefield, battlefield });
  page.press(pressOn(button));
  page.close({ target: new FakeDialog() });
  // The dialog's own close work hands focus to its button first.
  page.doc.activeElement = button;
  assert.equal(page.timers.length, 1);
  page.timers[0]();
  assert.deepEqual(page.focused, [battlefield], "keys are on again");
  page.close({ target: new FakeDialog() });
  assert.equal(page.timers.length, 1, "only once per press");

  const moved = topBar({ active: battlefield, battlefield });
  moved.press(pressOn(button));
  moved.close({ target: new FakeDialog() });
  moved.doc.activeElement = new FakeElement();
  moved.timers[0]();
  assert.deepEqual(moved.focused, [], "focus the player moved elsewhere stays");

  const keysOff = topBar({ active: new FakeElement(), battlefield });
  keysOff.press(pressOn(button));
  keysOff.close({ target: new FakeDialog() });
  assert.equal(keysOff.timers.length, 0, "keys that were off stay off");

  const notDialog = topBar({ active: battlefield, battlefield });
  notDialog.press(pressOn(button));
  notDialog.close({ target: new FakeElement() });
  assert.equal(notDialog.timers.length, 0, "only a dialog's close counts");
});

test("a view switch keeps focus on the battlefield element itself", () => {
  const source = functionSource("savePresentationPreferenceBeforeClear");
  assert.match(
    source,
    /active instanceof Element &&\s+active !== elements\.battlefield &&\s+\(elements\.battlefield\.contains\(active\)/u,
  );
});

/**
 * @param {{replay?: boolean, pinned: unknown, battlefield: FakeElement}} setup
 */
function unpinner({ replay = false, pinned, battlefield }) {
  /** @type {string[]} */
  const calls = [];
  const controller = {
    pinnedOwner: () => pinned,
    hide: () => calls.push("hide"),
    refresh: () => calls.push("refresh"),
  };
  /** @type {() => void} */
  const unpin = new Function(
    "isReplayMode",
    "tooltipController",
    "elements",
    `${functionSource("unpinArenaCardAfterClick")}\nreturn unpinArenaCardAfterClick;`,
  )(() => replay, controller, { battlefield });
  return { unpin, calls };
}

test("a click on the arena never leaves a card pinned on its marks", () => {
  const stone = new FakeElement();
  const battlefield = new FakeElement(null, [stone]);
  const inside = unpinner({ pinned: stone, battlefield });
  inside.unpin();
  assert.deepEqual(inside.calls, ["hide", "refresh"], "dropped, hover card kept");
  const outside = unpinner({ pinned: new FakeElement(), battlefield });
  outside.unpin();
  assert.deepEqual(outside.calls, [], "a pin outside the arena is kept");
  const none = unpinner({ pinned: null, battlefield });
  none.unpin();
  assert.deepEqual(none.calls, []);
  const replay = unpinner({ replay: true, pinned: stone, battlefield });
  replay.unpin();
  assert.deepEqual(replay.calls, [], "Replay Viewer pins are kept");
  assert.match(
    MAIN,
    /elements\.battlefield\.addEventListener\("click", unpinArenaCardAfterClick\);/u,
  );
});

test("Watch Only covers every deck column but Lock In", () => {
  assert.match(
    DECK_CSS,
    /\.deck\[data-state="watchonly"\] \.deck-body \{\s+position: relative;\s+\}/u,
  );
  // It covers the five columns' row from Your Champion to 3 Action, never
  // 4 Lock In or the deck's header line above it (owner, 4 Oct 2026).
  assert.match(
    DECK_CSS,
    /\.deck-body > div#watch-only \{\s+grid-row: agent-start \/ agent-end;\s+grid-column: agent-start \/ action-end;\s+inset: -4px -7px -12px -12px;\s+\}/u,
  );
  assert.match(
    DECK_CSS,
    /grid-template-areas:\s+"head head head head head"\s+"agent step target action lock";/u,
  );
});

/**
 * @param {{replay?: boolean, focused: boolean, shown: string, closed?: boolean}} setup
 * @returns {{paint: () => void, notices: unknown[], chip: Record<string, any>,
 *   text: {textContent: string}}}
 */
function keysPainter({ replay = false, focused, shown, closed = false }) {
  const battlefield = {};
  /** @type {unknown[]} */
  const notices = [];
  /** @param {string} name */
  const classes = (name) => ({
    name,
    /** @type {Record<string, boolean>} */
    on: {},
    /** @param {string} key @param {boolean} value */
    toggle(key, value) {
      this.on[key] = value;
    },
  });
  const chip = { classList: classes("chip") };
  const text = { textContent: "" };
  /** @type {() => void} */
  const paint = new Function(
    "isReplayMode",
    "document",
    "elements",
    "statusSentence",
    "clearNotice",
    "state",
    `${functionSource("paintKeysState")}\nreturn paintKeysState;`,
  )(
    () => replay,
    { activeElement: focused ? battlefield : {} },
    {
      battlefield,
      keysChip: chip,
      keysText: text,
      arena: { classList: classes("arena") },
      statusMessage: { textContent: shown },
    },
    statusSentence,
    (/** @type {string | undefined} */ text) => notices.push(text ?? ""),
    { shuttingDown: closed },
  );
  return { paint, notices, chip, text };
}

test("keys turning on make a stale keys-off status line true again", () => {
  const off = statusSentence({ kind: "keys_off" });
  const cleared = statusSentence({ kind: "cleared", keysOff: true });
  const plain = keysPainter({ focused: true, shown: off });
  plain.paint();
  assert.deepEqual(plain.notices, [""]);
  assert.equal(plain.chip.classList.on.off, false);
  const target = keysPainter({ focused: true, shown: cleared });
  target.paint();
  assert.deepEqual(target.notices, ["Target cleared."]);
  const other = keysPainter({ focused: true, shown: "Step: North." });
  other.paint();
  assert.deepEqual(other.notices, [], "any other message stays");
  const blurred = keysPainter({ focused: false, shown: off });
  blurred.paint();
  assert.deepEqual(blurred.notices, [], "keys off keep the sentence");
  assert.equal(blurred.chip.classList.on.off, true);
  assert.equal(blurred.text.textContent, "Keys Off · Click The Arena");
  const closed = keysPainter({ focused: true, shown: "", closed: true });
  closed.paint();
  assert.equal(closed.chip.classList.on.off, true, "no key works after Exit");
  assert.equal(closed.text.textContent, "Keys Off");
});
