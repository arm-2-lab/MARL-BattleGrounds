/**
 * @file Check the Saga Chronicle's one-dock contract (src/chronicle.js): on the
 * Replay Viewer page createChronicle moves the page's transport into one dock
 * (Row 0 caption, camera switch and readout; Row 1 ribbon; Row 2 Start,
 * Previous Fight, −10, −1, Play, Step Play, +1, +10, Next Fight, End, Fights
 * Only, Bookmark, Speed and the artifact actions), hides the old slider,
 * readout and status line with sr-only, and stays shown with no cursor; and
 * Fights Only (off by default): fightsOnlyCut keeps play inside a fight and
 * its 2-tick lead-in, cuts to 2 ticks before the next fight otherwise, and the
 * view makes that cut (pause, seek, play) with the Chapter Card "Skipped
 * Ticks 3–12 · Fight 2".
 */
import assert from "node:assert/strict";
import test from "node:test";

import { createChronicle, fightsOnlyCut } from "../src/chronicle.js";

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
    /** @type {Record<string, string>} */
    this.style = {};
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
    return { left: 0, top: 0, width: 432, height: 56, right: 432, bottom: 56 };
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
  /** @param {Node} node @param {Node} before */
  insertBefore(node, before) {
    this.detach(node);
    this.children.splice(this.children.indexOf(before), 0, node);
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
}

class Doc {
  /** @param {string} tag */
  createElement(tag) {
    return new Node(this, tag);
  }
  /** @param {string} _ns @param {string} tag */
  createElementNS(_ns, tag) {
    return new Node(this, tag);
  }
}

/** Build the Replay Viewer dock markup that index.html holds. */
function replayPage() {
  const doc = new Doc();
  /** @param {string} tag @param {{id?: string, className?: string}} props @param {Node[]} [kids] */
  const el = (tag, props, kids = []) => {
    const node = doc.createElement(tag);
    node.id = props.id ?? "";
    node.className = props.className ?? "";
    node.append(...kids);
    return node;
  };
  const transportIds = [
    "replay-first-button",
    "replay-back-ten-button",
    "replay-previous-button",
    "replay-play-pause-button",
    "replay-next-button",
    "replay-forward-ten-button",
    "replay-last-button",
  ];
  const chronicle = el("div", { id: "chronicle" });
  chronicle.hidden = true;
  const timeline = el("section", { id: "replay-timeline" }, [
    el("fieldset", {
      id: "replay-artifact-actions",
      className: "replay-artifact-actions",
    }),
    el("fieldset", { className: "replay-timeline__transport" }, [
      el("legend", {}),
      ...transportIds.map((id) => el("button", { id })),
      el("label", { className: "replay-timeline__slider" }, [
        el("input", { id: "replay-frame-slider" }),
      ]),
      el("output", { id: "replay-frame-position" }),
      el("div", { id: "replay-camera-switch" }),
      el("label", { className: "replay-timeline__rate" }),
      el("output", { id: "replay-transport-status" }),
    ]),
    chronicle,
  ]);
  return { timeline, chronicle };
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
    playOneTick: () => true,
  };
}

/** Oracle marks with two one-tick fights: frame 2 and frame 15. */
function twoFightMarks() {
  /** @param {string} id @param {1 | 2} team */
  const fall = (id, team) => ({
    public_agent_id: id,
    team_id: team,
    class_id: 2,
    home_fall: false,
  });
  return {
    schema_version: 1,
    marks_kind: "replay_timeline_marks",
    audience: "oracle",
    recipient_public_agent_id: null,
    final_frame_index: 20,
    ticks: [
      { frame_index: 2, falls: [fall("agent-6", 2)], ultimates: [], returns: [] },
      {
        frame_index: 15,
        falls: [fall("agent-1", 1)],
        ultimates: [],
        returns: [],
      },
    ],
    fights: [1, 2].map((number) => ({
      fight_number: number,
      first_frame_index: number === 1 ? 2 : 15,
      last_frame_index: number === 1 ? 2 : 15,
      ultimate_count: 0,
      fallen_count: 1,
    })),
  };
}

async function settle() {
  for (let index = 0; index < 5; index += 1) await Promise.resolve();
}

/** @param {Node} node */
const ids = (node) => node.children.map((child) => child.id || child.className);

test("fightsOnlyCut keeps fights and lead-ins, cuts the gaps, and stops after the last fight", () => {
  const fights = [
    {
      fightNumber: 1,
      firstFrameIndex: 2,
      lastFrameIndex: 2,
      ultimateCount: 0,
      fallenCount: 1,
    },
    {
      fightNumber: 2,
      firstFrameIndex: 15,
      lastFrameIndex: 15,
      ultimateCount: 0,
      fallenCount: 1,
    },
  ];
  assert.equal(fightsOnlyCut(fights, 0), null);
  assert.equal(fightsOnlyCut(fights, 2), null);
  assert.equal(fightsOnlyCut(fights, 13), null);
  assert.equal(fightsOnlyCut(fights, 16), null);
  assert.deepEqual(fightsOnlyCut(fights, 3), {
    targetFrameIndex: 13,
    skippedFirst: 3,
    skippedLast: 12,
    fight: fights[1],
  });
  assert.equal(fightsOnlyCut(fights, 12)?.skippedLast, 12);
  assert.equal(fightsOnlyCut([], 5), null);
});

test("the Replay Viewer transport is moved into one dock in the agreed order", () => {
  const { timeline, chronicle } = replayPage();
  const controller = fakeController();
  const view = createChronicle(/** @type {any} */ (chronicle), {
    controller,
    loadMarks: () => new Promise(() => {}),
    storage: null,
    clipboard: null,
  });
  assert.equal(view.docked, true);
  assert.equal(chronicle.hidden, false);
  const row = /** @type {Node} */ (
    timeline.querySelector(".replay-timeline__transport")
  );
  assert.equal(row.parentElement, chronicle);
  assert.deepEqual(ids(chronicle), [
    "chronicle__caption-row",
    "chronicle__ribbon-host",
    "replay-timeline__transport",
    "chronicle-bookmarks",
  ]);
  assert.deepEqual(ids(row), [
    "",
    "replay-first-button",
    "chronicle-previous-fight-button",
    "replay-back-ten-button",
    "replay-previous-button",
    "replay-play-pause-button",
    "chronicle-step-play-button",
    "replay-next-button",
    "replay-forward-ten-button",
    "chronicle-next-fight-button",
    "replay-last-button",
    "chronicle-fights-only-button",
    "chronicle-bookmark-button",
    "replay-timeline__rate",
    "replay-artifact-actions",
    "replay-timeline__slider sr-only",
    "replay-frame-position",
    "replay-transport-status",
  ]);
  for (const id of ["#replay-frame-position", "#replay-transport-status"]) {
    assert.match(timeline.querySelector(id)?.className ?? "", /sr-only/u);
  }
  assert.deepEqual(ids(/** @type {Node} */ (chronicle.children[0])), [
    "chronicle-caption",
    "replay-camera-switch",
    "chronicle-readout",
  ]);
  view.update({
    playback: { ...controller.snapshot(), cursor: null },
    audience: { kind: "oracle" },
    replayKey: "r1",
  });
  assert.equal(chronicle.hidden, false);
  view.update({
    playback: controller.snapshot(),
    audience: { kind: "oracle" },
    visible: false,
  });
  assert.equal(chronicle.hidden, true);
  view.dispose();
});

test("Fights Only is off by default, then cuts autoplay to 2 ticks before the next fight", async () => {
  const { chronicle } = replayPage();
  const controller = fakeController({ frame: 3, playing: true });
  const view = createChronicle(/** @type {any} */ (chronicle), {
    controller,
    loadMarks: () => Promise.resolve(twoFightMarks()),
    storage: null,
    clipboard: null,
  });
  const oracle = /** @type {const} */ ({ kind: "oracle" });
  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  await settle();
  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  assert.deepEqual(controller.calls, []);
  assert.equal(view.buttons.fightsOnly.getAttribute("aria-pressed"), "false");

  /** @type {any} */ (view.buttons.fightsOnly).listeners.get("click")[0]({});
  assert.equal(view.buttons.fightsOnly.getAttribute("aria-pressed"), "true");
  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  await settle();
  assert.deepEqual(controller.calls, ["pause:user_pause", "seek:13", "play"]);
  const caption = /** @type {Node} */ (chronicle.querySelector("#chronicle-caption"));
  assert.equal(caption.textContent, "Skipped Ticks 3–12 · Fight 2");
  assert.equal(caption.dataset.kind, "card");
  view.dispose();
});
