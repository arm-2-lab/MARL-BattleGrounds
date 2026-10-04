/**
 * @file Check the Rune Scrubber's one-dock contract (src/rune-scrubber.js): on the
 * Replay Viewer page createRuneScrubber moves the page's transport into one dock
 * (Row 1 the Event Track; Row 2 exactly Start, −10, −1, Play/Pause, +1, +10, End,
 * the readout, Speed, Export and the camera switch, with no Step Play, fight or
 * bookmark button anywhere), moves the artifact actions (moved, not copied) into the
 * Export menu, hides the old slider, readout and status line with sr-only, and stays
 * shown with no cursor; the Export menu opens and closes on its button, on Escape
 * (focus returns to Export), on a press outside it and after a press of one of its
 * buttons, and the Export button is never disabled; a plain click on the track
 * (pointerdown then pointerup, no move) pauses play first, then seeks to the tick
 * under the pointer, and a right-button press does nothing; and during a drag the
 * thumb follows the pointer and never steps back while earlier seeks land (one seek
 * in flight, the newest target queued).
 */
import assert from "node:assert/strict";
import test from "node:test";

import { ReplayPlaybackController } from "../src/replay-controls.js";
import { createRuneScrubber } from "../src/rune-scrubber.js";

class Node {
  /** @param {Doc} ownerDocument @param {string} tagName */
  constructor(ownerDocument, tagName) {
    this.ownerDocument = ownerDocument;
    this.tagName = tagName.toUpperCase();
    /** @type {Node[]} */
    this.children = [];
    /** @type {Node | null} */
    this.parentElement = null;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {Record<string, string>} */
    this.dataset = {};
    /** @type {Map<string, Array<(event: any) => void>>} */
    this.listeners = new Map();
    this.textContent = "";
    this.hidden = false;
    this.disabled = false;
    this.className = "";
    this.id = "";
    this.type = "";
    this.tabIndex = -1;
    this.classList = {
      /** @param {string} name */
      add: (name) => {
        this.className = `${this.className} ${name}`.trim();
      },
    };
  }
  get firstElementChild() {
    return this.children[0] ?? null;
  }
  get childNodes() {
    return this.children;
  }
  /** @param {string} name @param {unknown} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  /** @param {string} name */
  hasAttribute(name) {
    return this.attributes.has(name);
  }
  /** @param {string} name */
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  getBoundingClientRect() {
    return { left: 0, top: 0, width: 432, height: 36, right: 432, bottom: 36 };
  }
  /** @param {Node} node */
  detach(node) {
    node.parentElement?.children.splice(node.parentElement.children.indexOf(node), 1);
    node.parentElement = this;
  }
  /** @param {...Node} nodes */
  append(...nodes) {
    for (const node of nodes) {
      this.detach(node);
      this.children.push(node);
    }
  }
  /** @param {...Node} nodes */
  replaceChildren(...nodes) {
    for (const node of [...this.children]) node.parentElement = null;
    this.children = [];
    this.append(...nodes);
  }
  /** @param {string} type @param {(event: any) => void} handler */
  addEventListener(type, handler) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), handler]);
  }
  /** @param {string} type @param {any} [event] */
  fire(type, event = {}) {
    const full = { target: this, stopPropagation() {}, ...event };
    /** @type {Node | null} */
    let node = this;
    while (node !== null) {
      for (const handler of node.listeners.get(type) ?? []) handler(full);
      node = node.parentElement;
    }
    return full;
  }
  focus() {
    this.ownerDocument.activeElement = this;
  }
  /** @param {Node} other */
  contains(other) {
    /** @type {Node | null} */
    let node = other;
    while (node !== null && node !== this) node = node.parentElement;
    return node === this;
  }
  /** @param {string} selector */
  matches(selector) {
    if (selector.startsWith("#")) return this.id === selector.slice(1);
    if (selector.startsWith("."))
      return this.className.split(" ").includes(selector.slice(1));
    return this.tagName === selector.toUpperCase();
  }
  /** @param {string} selector @returns {Node | null} */
  closest(selector) {
    /** @type {Node | null} */
    let node = this;
    while (node !== null && !node.matches(selector)) node = node.parentElement;
    return node;
  }
  /** @param {string} selector @returns {Node | null} */
  querySelector(selector) {
    for (const child of this.children) {
      if (child.matches(selector)) return child;
      const found = child.querySelector(selector);
      if (found !== null) return found;
    }
    return null;
  }
  /** @returns {Node[]} */
  all() {
    return [this, ...this.children.flatMap((child) => child.all())];
  }
}

class Doc {
  constructor() {
    /** @type {Node | null} */
    this.activeElement = null;
    /** @type {Map<string, Array<(event: any) => void>>} */
    this.listeners = new Map();
  }
  /** @param {string} tag */
  createElement(tag) {
    return new Node(this, tag);
  }
  /** @param {string} _ns @param {string} tag */
  createElementNS(_ns, tag) {
    return new Node(this, tag);
  }
  /** @param {string} type @param {(event: any) => void} handler */
  addEventListener(type, handler) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), handler]);
  }
  /** @param {string} type @param {(event: any) => void} handler */
  removeEventListener(type, handler) {
    this.listeners.set(
      type,
      (this.listeners.get(type) ?? []).filter((entry) => entry !== handler),
    );
  }
  /** @param {string} type @param {any} event */
  fire(type, event) {
    for (const handler of this.listeners.get(type) ?? []) handler(event);
  }
}

/** Build the Replay Viewer dock markup that index.html holds. */
function replayPage() {
  const doc = new Doc();
  /** @param {string} tag @param {{id?: string, className?: string, text?: string}} props @param {Node[]} [kids] */
  const el = (tag, props, kids = []) => {
    const node = doc.createElement(tag);
    node.id = props.id ?? "";
    node.className = props.className ?? "";
    node.textContent = props.text ?? "";
    node.append(...kids);
    return node;
  };
  const transport = [
    ["replay-first-button", "Start"],
    ["replay-back-ten-button", "−10"],
    ["replay-previous-button", "−1"],
    ["replay-play-pause-button", "Play"],
    ["replay-next-button", "+1"],
    ["replay-forward-ten-button", "+10"],
    ["replay-last-button", "End"],
  ];
  const slot = el("div", { id: "chronicle" });
  slot.hidden = true;
  const artifactActions = el(
    "fieldset",
    { id: "replay-artifact-actions", className: "replay-artifact-actions" },
    [
      el("button", { id: "replay-export-png-button", text: "Export PNG" }),
      el("button", {
        id: "replay-download-metrics-button",
        text: "Download Metrics CSV",
      }),
      el("button", { id: "replay-episode-details-button", text: "Episode Details" }),
    ],
  );
  const timeline = el("section", { id: "replay-timeline" }, [
    artifactActions,
    el("fieldset", { className: "replay-timeline__transport" }, [
      el("legend", {}),
      ...transport.map(([id, text]) => el("button", { id, text })),
      el("label", { className: "replay-timeline__slider" }, [
        el("input", { id: "replay-frame-slider" }),
      ]),
      el("output", { id: "replay-frame-position" }),
      el("div", { id: "replay-camera-switch" }),
      el("label", { className: "replay-timeline__rate" }, [
        el("select", { id: "replay-playback-rate" }),
      ]),
      el("output", { id: "replay-transport-status" }),
    ]),
    slot,
  ]);
  return { doc, timeline, slot, artifactActions };
}

/** @param {{frame?: number, playing?: boolean}} [start] */
function fakeController(start = {}) {
  const calls = /** @type {string[]} */ ([]);
  const state = {
    cursor: { frame_index: start.frame ?? 3, final_frame_index: 20 },
    playing: start.playing ?? false,
    requestPending: false,
    connected: true,
    hidden: false,
    transportState: "SETTLED",
    pauseReason: /** @type {string | null} */ (null),
    playbackRate: 1,
  };
  return {
    state,
    calls,
    snapshot: () => ({ ...state }),
    /** @param {number} frameIndex */
    seek: (frameIndex) => {
      calls.push(`seek:${frameIndex}`);
      return Promise.resolve(true);
    },
    play: () => {
      calls.push("play");
    },
    /** @param {string} [reason] */
    pause: (reason) => {
      calls.push(`pause:${reason}`);
      state.playing = false;
      state.pauseReason = reason ?? null;
    },
  };
}

/** @param {Node} node */
const ids = (node) => node.children.map((child) => child.id || child.className);

test("the Replay Viewer transport is moved into one dock in the agreed order", () => {
  const { timeline, slot, artifactActions } = replayPage();
  const controller = fakeController();
  const view = createRuneScrubber(/** @type {any} */ (slot), {
    controller,
    loadMarks: () => new Promise(() => {}),
  });
  assert.equal(view.docked, true);
  assert.equal(slot.hidden, false);
  assert.match(slot.className, /\brune-scrubber\b/u);
  const row = /** @type {Node} */ (
    timeline.querySelector(".replay-timeline__transport")
  );
  assert.equal(row.parentElement, slot);
  assert.deepEqual(ids(slot), [
    "rune-scrubber__track",
    "replay-timeline__transport rune-scrubber__row",
  ]);
  assert.deepEqual(ids(row), [
    "",
    "replay-first-button",
    "replay-back-ten-button",
    "replay-previous-button",
    "replay-play-pause-button",
    "replay-next-button",
    "replay-forward-ten-button",
    "replay-last-button",
    "rune-scrubber__readout",
    "replay-timeline__rate",
    "rune-scrubber__export",
    "replay-camera-switch",
    "replay-timeline__slider sr-only",
    "replay-frame-position",
    "replay-transport-status",
  ]);
  // What a sighted viewer sees, left to right.
  const visible = row.children
    .filter((child) => child.tagName !== "LEGEND" && !/sr-only/u.test(child.className))
    .map((child) =>
      child.className === "replay-timeline__rate"
        ? "Speed"
        : child.className === "rune-scrubber__readout"
          ? "Readout"
          : child.className === "rune-scrubber__export"
            ? (child.querySelector("button")?.textContent ?? "")
            : child.id === "replay-camera-switch"
              ? "Camera"
              : child.textContent,
    );
  assert.deepEqual(visible, [
    "Start",
    "−10",
    "−1",
    "Play",
    "+1",
    "+10",
    "End",
    "Readout",
    "Speed",
    "Export",
    "Camera",
  ]);
  for (const node of slot.all()) {
    assert.doesNotMatch(
      `${node.id} ${node.className} ${node.textContent}`,
      /step|fight|bookmark|chapter|pin/iu,
    );
  }
  for (const id of ["#replay-frame-position", "#replay-transport-status"]) {
    assert.match(timeline.querySelector(id)?.className ?? "", /sr-only/u);
  }
  const menu = /** @type {Node} */ (slot.querySelector("#rune-scrubber-export-menu"));
  assert.equal(artifactActions.parentElement, menu, "moved, not copied");
  assert.equal(timeline.children.includes(artifactActions), false);
  assert.equal(
    timeline.all().filter((node) => node.id === "replay-export-png-button").length,
    1,
  );
  view.update({
    playback: { ...controller.snapshot(), cursor: null },
    audience: { kind: "oracle" },
    replayKey: "r1",
  });
  assert.equal(slot.hidden, false, "the docked transport never disappears");
  view.update({
    playback: controller.snapshot(),
    audience: { kind: "oracle" },
    visible: false,
  });
  assert.equal(slot.hidden, true);
  view.dispose();
});

test("the Export menu opens on its button and closes on Escape, outside presses and its own buttons", () => {
  const { doc, slot } = replayPage();
  const controller = fakeController();
  const view = createRuneScrubber(/** @type {any} */ (slot), {
    controller,
    loadMarks: () => new Promise(() => {}),
  });
  const button = /** @type {Node} */ (/** @type {unknown} */ (view.exportButton));
  const menu = /** @type {Node} */ (slot.querySelector("#rune-scrubber-export-menu"));
  const png = /** @type {Node} */ (slot.querySelector("#replay-export-png-button"));
  assert.equal(menu.hidden, true);
  assert.equal(button.getAttribute("aria-controls"), "rune-scrubber-export-menu");
  controller.state.playing = true;
  view.update({ playback: controller.snapshot(), audience: { kind: "oracle" } });
  assert.equal(button.disabled, false, "Export never greys out");

  button.fire("click");
  assert.equal(menu.hidden, false);
  assert.equal(button.getAttribute("aria-expanded"), "true");
  png.fire("keydown", { key: "Escape" });
  assert.equal(menu.hidden, true);
  assert.equal(button.getAttribute("aria-expanded"), "false");
  assert.equal(doc.activeElement, button, "Escape returns focus to Export");

  button.fire("click");
  doc.fire("pointerdown", { target: png });
  assert.equal(menu.hidden, false, "a press inside keeps it open");
  doc.fire("pointerdown", { target: slot.querySelector(".rune-scrubber__track") });
  assert.equal(menu.hidden, true, "a press outside closes it");
  assert.equal(doc.listeners.get("pointerdown")?.length, 0, "no listener left behind");

  button.fire("click");
  png.fire("click");
  assert.equal(menu.hidden, true, "a menu button closes the menu after its action");
  button.fire("click");
  button.fire("click");
  assert.equal(menu.hidden, true, "the button toggles");
  view.dispose();
});

test("a plain click on the Event Track pauses play first, then seeks to the tick under it", async () => {
  const { slot } = replayPage();
  const controller = fakeController({ frame: 3, playing: true });
  const view = createRuneScrubber(/** @type {any} */ (slot), {
    controller,
    loadMarks: () => new Promise(() => {}),
  });
  view.update({ playback: controller.snapshot(), audience: { kind: "oracle" } });
  const track = /** @type {Node} */ (slot.querySelector(".rune-scrubber__track"));
  // pointerdown redraws the track under the pointer (a browser then sends no
  // "click"), so the seek must come from pointerup alone.
  track.fire("pointerdown", { button: 0, pointerId: 1, clientX: 116 });
  track.fire("pointerup", { button: 0, pointerId: 1, clientX: 116 });
  await Promise.resolve();
  assert.deepEqual(controller.calls, ["pause:user_pause", "seek:5"]);
  track.fire("pointerdown", { button: 2, pointerId: 2, clientX: 316 });
  track.fire("pointerup", { button: 2, pointerId: 2, clientX: 316 });
  assert.equal(controller.calls.length, 2, "a right-button press does not seek");
  view.dispose();
});

test("the thumb follows a drag and never steps back while the seeks land", async () => {
  const { slot } = replayPage();
  /** @type {number[]} */
  const sent = [];
  /** @type {Array<(value: unknown) => void>} */
  const answers = [];
  /** @param {number} frame @param {number} generation */
  const at = (frame, generation) => ({
    schema_version: 1,
    frame_index: frame,
    final_frame_index: 20,
    cursor_generation: generation,
    choreography_generation: 1,
  });
  /** @type {ReturnType<typeof createRuneScrubber> | null} */
  let view = null;
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      sent.push(command.frame_index);
      return new Promise((resolve) => {
        answers.push(resolve);
      });
    },
    // The page passes every snapshot to the scrubber, as main.js does.
    onStateChange: (playback) =>
      view?.update({ playback, audience: { kind: "oracle" } }),
  });
  view = createRuneScrubber(/** @type {any} */ (slot), {
    controller,
    loadMarks: () => new Promise(() => {}),
  });
  controller.setConnected(true);
  controller.installCursor(at(3, 3));
  const thumbX = () =>
    Number(
      slot
        .all()
        .find((node) => node.getAttribute("class") === "rune-scrubber__thumb")
        ?.getAttribute("cx"),
    );
  assert.equal(thumbX(), 76, "frame 3 at 20 px per tick");
  const track = /** @type {Node} */ (slot.querySelector(".rune-scrubber__track"));
  /** @type {number[]} */
  const seen = [];
  track.fire("pointerdown", { button: 0, pointerId: 1, clientX: 116 });
  seen.push(thumbX());
  for (const clientX of [216, 316]) {
    track.fire("pointermove", { button: 0, pointerId: 1, clientX });
    seen.push(thumbX());
  }
  track.fire("pointerup", { button: 0, pointerId: 1, clientX: 316 });
  seen.push(thumbX());
  assert.deepEqual(sent, [10], "one seek in flight, the newest queued");
  // The first seek lands on frame 10: the thumb stays on the queued target 15.
  answers.shift()?.({ frame: { cursor: at(10, 4) } });
  for (let index = 0; index < 6; index += 1) await Promise.resolve();
  seen.push(thumbX());
  assert.deepEqual(sent, [10, 15]);
  answers.shift()?.({ frame: { cursor: at(15, 5) } });
  for (let index = 0; index < 6; index += 1) await Promise.resolve();
  seen.push(thumbX());
  assert.deepEqual(seen, [116, 216, 316, 316, 316, 316]);
  view.dispose();
});
