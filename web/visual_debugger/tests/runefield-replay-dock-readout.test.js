/**
 * @file Check the Replay Viewer dock readout is plain words, as in the design store:
 * the position line reads "Tick 98 / 122" and the status line reads "Paused · Settled",
 * "Playing", "Seeking…" or "Offline". The status line never shows the frame index, the
 * speed or an incoming transition ID (those belong to the Frame tab's Technical Frame),
 * and data-transport-state keeps the exact transport state for tests and tools.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  REPLAY_TRANSPORT_STATES,
  renderReplayTimelineControls,
} from "../src/replay-controls.js";

class FakeElement {
  constructor() {
    this.attributes = new Map();
    this.disabled = false;
    this.value = "";
    this.min = "";
    this.max = "";
    this.step = "";
    this.textContent = "";
  }

  /** @param {string} name @param {unknown} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
}

function elements() {
  return {
    firstButton: new FakeElement(),
    backTenButton: new FakeElement(),
    previousButton: new FakeElement(),
    playPauseButton: new FakeElement(),
    nextButton: new FakeElement(),
    forwardTenButton: new FakeElement(),
    lastButton: new FakeElement(),
    slider: new FakeElement(),
    position: new FakeElement(),
    rateSelect: new FakeElement(),
    status: new FakeElement(),
    tickForFrameIndex: (/** @type {number} */ frameIndex) => frameIndex - 12,
  };
}

/**
 * @param {string} transportState
 * @param {boolean} playing
 * @param {boolean} [withCursor]
 */
function snapshot(transportState, playing, withCursor = true) {
  return /** @type {any} */ ({
    transportState,
    generation: 1,
    presentationIntent: null,
    cursor: withCursor ? { frame_index: 110, final_frame_index: 134 } : null,
    playing,
    requestPending: transportState === REPLAY_TRANSPORT_STATES.ADVANCING,
    presentationPending: false,
    connected: transportState !== REPLAY_TRANSPORT_STATES.OFFLINE,
    hidden: false,
    playbackRate: 1.25,
    pauseReason: null,
    atStart: false,
    atEnd: false,
  });
}

test("dock readout says the tick and a plain playback label, with no debug IDs", () => {
  const cases = [
    [REPLAY_TRANSPORT_STATES.SETTLED, false, true, "Paused · Settled"],
    [REPLAY_TRANSPORT_STATES.PLAYING, true, true, "Playing"],
    [REPLAY_TRANSPORT_STATES.ADVANCING, true, true, "Playing"],
    [REPLAY_TRANSPORT_STATES.ADVANCING, false, true, "Seeking…"],
    [REPLAY_TRANSPORT_STATES.OFFLINE, false, true, "Offline"],
    [REPLAY_TRANSPORT_STATES.SETTLED, false, false, "Offline"],
  ];
  for (const [transportState, playing, withCursor, label] of cases) {
    const dock = elements();
    renderReplayTimelineControls(
      /** @type {any} */ (dock),
      snapshot(String(transportState), Boolean(playing), Boolean(withCursor)),
    );
    assert.equal(dock.status.textContent, label);
    assert.equal(dock.status.value, label);
    assert.equal(dock.status.getAttribute("data-transport-state"), transportState);
    assert.doesNotMatch(dock.status.textContent, /Frame|Transition|transition|×|\d/u);
  }
  const settled = elements();
  renderReplayTimelineControls(
    /** @type {any} */ (settled),
    snapshot(REPLAY_TRANSPORT_STATES.SETTLED, false),
  );
  assert.equal(settled.position.textContent, "Tick 98 / 122");
  assert.equal(settled.rateSelect.value, "1.25");
});
