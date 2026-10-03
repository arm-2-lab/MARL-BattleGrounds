/**
 * @file Check the Saga Chronicle (src/chronicle.js): the timeline marks payload
 * check (schema, audience and recipient, order and range), fight captions
 * ("Fight 2 Of 3 · Ticks 5–7", "Starts In 1 Tick", "Between Fights", and no
 * future under Hide Marks Ahead), Previous and Next Fight targets, the beat and
 * tick readout, ribbon records (frame-to-x, current tick split, veil, chapter
 * padding, switches), hover rows with "Team B Warrior · ID 6" labels, bookmark
 * storage (try/catch, 50 pins, 60-character notes, Copy List text), beat times
 * from a choreography schedule, the one-request-in-flight queue (a press pauses
 * first, a newer press replaces a queued one, quick presses add up), and the
 * Agent POV boundary: a view change drops the old marks at once, a payload for
 * another view is refused, Oracle -> Agent POV -> Oracle redraws exactly as a
 * fresh view, and the module reads no frame data and binds no keys.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  bookmarkListText,
  buildChronicleRibbon,
  CHRONICLE_BOOKMARK_LIMIT,
  CHRONICLE_BOOKMARK_STORAGE_PREFIX,
  ChronicleNavigator,
  chapterCaption,
  chronicleAgentLabel,
  chronicleAudienceKey,
  chronicleBeatsFromSchedule,
  chronicleReadout,
  chronicleRibbonRecords,
  createChronicle,
  marksMatchAudience,
  nextFightStart,
  normalizeTimelineMarks,
  previousFightStart,
  readBookmarks,
  setBookmarkNote,
  tickMarkRows,
  toggleBookmark,
  writeBookmarks,
} from "../src/chronicle.js";

/** @param {string} id @param {1 | 2} team @param {1 | 2 | 3 | 4 | 5} cls */
function agent(id, team, cls) {
  return { public_agent_id: id, team_id: team, class_id: cls };
}

/** @param {"oracle" | "agent_pov"} audience @param {string | null} recipient */
function rawMarks(audience = "oracle", recipient = null) {
  return {
    schema_version: 1,
    marks_kind: "replay_timeline_marks",
    audience,
    recipient_public_agent_id: recipient,
    final_frame_index: 20,
    ticks: [
      {
        frame_index: 2,
        falls: [{ ...agent("agent-6", 2, 2), home_fall: true }],
        ultimates: [],
        returns: [],
      },
      {
        frame_index: 5,
        falls: [],
        ultimates: [
          { ...agent("agent-1", 1, 4), damaging: true },
          { ...agent("agent-2", 1, 1), damaging: false },
        ],
        returns: [],
      },
      {
        frame_index: 7,
        falls: [{ ...agent("agent-1", 1, 4), home_fall: false }],
        ultimates: [],
        returns: [agent("agent-6", 2, 2)],
      },
      {
        frame_index: 15,
        falls: [{ ...agent("agent-7", 2, 3), home_fall: false }],
        ultimates: [],
        returns: [],
      },
    ],
    fights: [
      {
        fight_number: 1,
        first_frame_index: 2,
        last_frame_index: 2,
        ultimate_count: 0,
        fallen_count: 1,
      },
      {
        fight_number: 2,
        first_frame_index: 5,
        last_frame_index: 7,
        ultimate_count: 2,
        fallen_count: 1,
      },
      {
        fight_number: 3,
        first_frame_index: 15,
        last_frame_index: 15,
        ultimate_count: 0,
        fallen_count: 1,
      },
    ],
  };
}

test("timeline marks are checked and frozen", () => {
  const marks = normalizeTimelineMarks(rawMarks());
  assert.equal(marks.audience, "oracle");
  assert.equal(marks.ticks[0].falls[0].homeFall, true);
  assert.equal(marks.ticks[1].ultimates[0].damaging, true);
  assert.equal(marks.fights[1].lastFrameIndex, 7);
  assert.ok(Object.isFrozen(marks.ticks[0].falls[0]));
  const pov = normalizeTimelineMarks(rawMarks("agent_pov", "agent-3"));
  assert.equal(pov.recipientPublicAgentId, "agent-3");

  const bad = [
    { ...rawMarks(), extra: 1 },
    { ...rawMarks(), schema_version: 2 },
    rawMarks("oracle", "agent-3"),
    rawMarks("agent_pov", null),
    { ...rawMarks(), final_frame_index: 10 },
    { ...rawMarks(), ticks: [...rawMarks().ticks].reverse() },
    { ...rawMarks(), fights: [{ ...rawMarks().fights[0], fight_number: 2 }] },
    {
      ...rawMarks(),
      ticks: [
        {
          frame_index: 3,
          falls: [{ ...agent("agent-1", 1, 1), team_id: 3, home_fall: false }],
          ultimates: [],
          returns: [],
        },
      ],
    },
    {
      ...rawMarks(),
      ticks: [
        { frame_index: 3, falls: [], ultimates: [agent("agent-1", 1, 1)], returns: [] },
      ],
    },
  ];
  for (const payload of bad) {
    assert.throws(() => normalizeTimelineMarks(payload), TypeError);
  }
});

test("audience keys and matches keep views apart", () => {
  const oracle = normalizeTimelineMarks(rawMarks());
  const pov = normalizeTimelineMarks(rawMarks("agent_pov", "agent-3"));
  assert.equal(chronicleAudienceKey({ kind: "oracle" }), "oracle");
  assert.equal(
    chronicleAudienceKey({ kind: "agent_pov", recipientPublicAgentId: "agent-3" }),
    "agent_pov:agent-3",
  );
  assert.equal(marksMatchAudience(oracle, { kind: "oracle" }), true);
  assert.equal(marksMatchAudience(pov, { kind: "oracle" }), false);
  assert.equal(
    marksMatchAudience(oracle, { kind: "agent_pov", recipientPublicAgentId: null }),
    false,
  );
  assert.equal(
    marksMatchAudience(pov, { kind: "agent_pov", recipientPublicAgentId: "agent-4" }),
    false,
  );
  assert.equal(
    marksMatchAudience(pov, { kind: "agent_pov", recipientPublicAgentId: "agent-3" }),
    true,
  );
  assert.equal(chronicleAgentLabel("agent-3"), "Agent 3");
  assert.equal(chronicleAgentLabel("scout"), "Agent scout");
});

test("captions name the fight, the lead-in or the lull, and hide the future on request", () => {
  const marks = normalizeTimelineMarks(rawMarks());
  /** @param {number} frameIndex @param {boolean} [hide] */
  const text = (frameIndex, hide = false) =>
    chapterCaption({ marks, frameIndex, hideMarksAhead: hide }).text;
  assert.equal(text(6), "Fight 2 Of 3 · Ticks 5–7");
  assert.equal(text(2), "Fight 1 Of 3 · Tick 2");
  assert.equal(text(4), "Fight 2 · Starts In 1 Tick");
  assert.equal(text(13), "Fight 3 · Starts In 2 Ticks");
  assert.equal(text(10), "Between Fights");
  assert.equal(text(6, true), "Fight 2 · From Tick 5");
  assert.equal(text(4, true), "Between Fights");
  assert.equal(
    chapterCaption({ marks, frameIndex: 6, tickForFrameIndex: (frame) => frame + 100 })
      .text,
    "Fight 2 Of 3 · Ticks 105–107",
  );
  assert.deepEqual(chapterCaption({ marks: null, frameIndex: 6 }), {
    kind: "none",
    text: "",
    fight: null,
  });
});

test("previous and next fight land on a fight's first tick and stop at the ends", () => {
  const { fights } = normalizeTimelineMarks(rawMarks());
  assert.equal(nextFightStart(fights, 0), 2);
  assert.equal(nextFightStart(fights, 5), 15);
  assert.equal(nextFightStart(fights, 15), null);
  assert.equal(previousFightStart(fights, 6), 5);
  assert.equal(previousFightStart(fights, 5), 2);
  assert.equal(previousFightStart(fights, 2), null);
});

test("the readout says the beat and the tick", () => {
  const base = {
    cursor: { frame_index: 4, final_frame_index: 20 },
    playing: false,
    requestPending: false,
    connected: true,
    hidden: false,
    transportState: "SETTLED",
    pauseReason: null,
    playbackRate: 1,
  };
  assert.equal(chronicleReadout(base), "Settled · Tick 4 of 20");
  assert.equal(chronicleReadout({ ...base, playing: true }), "Playing · Tick 4 of 20");
  assert.equal(
    chronicleReadout({ ...base, requestPending: true }),
    "Moving · Tick 4 of 20",
  );
  assert.equal(
    chronicleReadout({ ...base, pauseReason: "user_pause" }),
    "Paused · Tick 4 of 20",
  );
  assert.equal(chronicleReadout(base, { showBeat: false }), "Tick 4 of 20");
  assert.equal(chronicleReadout({ ...base, cursor: null }), "Tick —");
});

test("ribbon records place frames, split the current tick and veil the ticks ahead", () => {
  const marks = normalizeTimelineMarks(rawMarks());
  const records = chronicleRibbonRecords({
    marks,
    frameIndex: 5,
    finalFrameIndex: 20,
    width: 432,
  });
  assert.equal(records.perTick, 20);
  assert.equal(records.playhead.x, 116);
  assert.equal(records.currentMarks.hasRunes, true);
  assert.equal(records.currentMarks.hasFalls, false);
  assert.equal(records.currentMarks.runes.length, 2);
  assert.equal(records.currentMarks.runes.at(-1)?.damaging, true);
  assert.match(records.currentMarks.runes[0].d, /^M118\.75 16\.5/u);
  assert.notEqual(records.staticMarks.fallB, "");
  assert.deepEqual(records.veil, { x: 106, width: 314 });
  assert.equal(records.chapters.length, 3);
  assert.equal(records.chapters[1].current, true);
  assert.deepEqual([records.chapters[0].x, records.chapters[0].width], [52, 8]);
  assert.equal(records.hitTicks.length, 4);

  const hidden = chronicleRibbonRecords({
    marks,
    frameIndex: 6,
    finalFrameIndex: 20,
    width: 432,
    hideMarksAhead: true,
    bookmarks: [
      { frameIndex: 3, note: "" },
      { frameIndex: 18, note: "" },
    ],
  });
  assert.equal(hidden.hitTicks.length, 2);
  assert.equal(hidden.chapters.length, 2);
  assert.equal(hidden.chapters[1].width, 28);
  assert.deepEqual(
    hidden.pins.map((pin) => pin.frameIndex),
    [3],
  );

  const plain = chronicleRibbonRecords({
    marks,
    frameIndex: 5,
    finalFrameIndex: 20,
    width: 432,
    showMarks: false,
    showChapters: false,
    showBookmarks: false,
    bookmarks: [{ frameIndex: 3, note: "" }],
  });
  assert.equal(plain.hitTicks.length, 0);
  assert.equal(plain.chapters.length, 0);
  assert.equal(plain.pins.length, 0);
  assert.equal(plain.staticMarks.fallA + plain.staticMarks.fallB, "");
  const empty = chronicleRibbonRecords({
    marks: null,
    frameIndex: 0,
    finalFrameIndex: 0,
    width: 10,
  });
  assert.equal(empty.width, 64);
  assert.equal(empty.veil, null);
});

test("hover rows name champions with Team A or Team B and ability names", () => {
  const marks = normalizeTimelineMarks(rawMarks());
  assert.deepEqual(tickMarkRows(marks.ticks[0]), [
    { label: "Fell", value: "Team B Warrior · ID 6 · Home Fall, +2" },
  ]);
  const rows = tickMarkRows(marks.ticks[1]);
  assert.equal(rows[0].value, "Team A Rogue · ID 1 · Crippling Poison");
  assert.equal(rows[1].value, "Team A Mage · ID 2 · Burst");
  assert.deepEqual(tickMarkRows(marks.ticks[2]).at(-1), {
    label: "Returned",
    value: "Team B Warrior · ID 6",
  });
});

test("bookmarks survive bad storage, cap at 50 and copy as plain text", () => {
  /** @type {Map<string, string>} */
  const store = new Map();
  const storage = {
    /** @param {string} key */
    getItem: (key) => store.get(key) ?? null,
    /** @param {string} key @param {string} value */
    setItem: (key, value) => {
      store.set(key, value);
    },
  };
  let pins = toggleBookmark([], 9).pins;
  pins = toggleBookmark(pins, 4).pins;
  pins = setBookmarkNote(pins, 9, "x".repeat(80));
  assert.deepEqual(
    pins.map((pin) => pin.frameIndex),
    [4, 9],
  );
  assert.equal(pins[1].note.length, 60);
  assert.equal(writeBookmarks(storage, "replay-1", pins), true);
  assert.ok(store.has(`${CHRONICLE_BOOKMARK_STORAGE_PREFIX}replay-1`));
  assert.deepEqual(readBookmarks(storage, "replay-1"), pins);
  assert.deepEqual(readBookmarks(storage, "replay-2"), []);
  assert.equal(toggleBookmark(pins, 4).change, "removed");
  const broken = {
    getItem: () => {
      throw new Error("blocked");
    },
    setItem: () => {
      throw new Error("blocked");
    },
  };
  assert.deepEqual(readBookmarks(broken, "replay-1"), []);
  assert.equal(writeBookmarks(broken, "replay-1", pins), false);
  store.set(`${CHRONICLE_BOOKMARK_STORAGE_PREFIX}bad`, "{not json");
  assert.deepEqual(readBookmarks(storage, "bad"), []);
  let full = /** @type {ReadonlyArray<{frameIndex: number, note: string}>} */ ([]);
  for (let frame = 0; frame < CHRONICLE_BOOKMARK_LIMIT; frame += 1) {
    full = toggleBookmark(full, frame).pins;
  }
  assert.equal(toggleBookmark(full, 99).change, "full");
  assert.equal(
    bookmarkListText(
      setBookmarkNote(toggleBookmark([], 3).pins, 3, " Poison "),
      (frame) => frame + 10,
    ),
    "Tick 13 · Poison",
  );
});

test("beat times come from the choreography schedule", () => {
  assert.deepEqual(
    chronicleBeatsFromSchedule({
      windows: { impact: [280, 600] },
      fall: [900, 1300],
      returnBeat: null,
    }),
    { impactMs: 280, fallMs: 900, returnMs: null },
  );
  assert.equal(chronicleBeatsFromSchedule(null), null);
});

/**
 * @param {{frame?: number, final?: number, playing?: boolean, requestPending?: boolean}} [start]
 */
function fakeController(start = {}) {
  const calls = /** @type {Array<string>} */ ([]);
  const state = {
    cursor: { frame_index: start.frame ?? 3, final_frame_index: start.final ?? 20 },
    playing: start.playing ?? false,
    requestPending: start.requestPending ?? false,
    connected: true,
    hidden: false,
    transportState: "SETTLED",
    pauseReason: /** @type {string | null} */ (null),
    playbackRate: 1,
  };
  const controller = {
    state,
    calls,
    snapshot: () => ({ ...state }),
    /** @param {number} frameIndex */
    seek: (frameIndex) => {
      calls.push(`seek:${frameIndex}`);
      state.requestPending = true;
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
    playOneTick: () => {
      calls.push("one_tick");
      return true;
    },
  };
  return controller;
}

test("buttons stay live: one request in flight, newest press queued, presses add up", async () => {
  const controller = fakeController({ playing: true });
  const navigator = new ChronicleNavigator(controller);
  assert.equal(navigator.seek(8), "sent");
  assert.deepEqual(controller.calls, ["pause:user_pause", "seek:8"]);
  assert.equal(navigator.baseFrame(), 8);
  assert.equal(navigator.seek(10), "queued");
  assert.equal(navigator.stepPlay(), "queued");
  assert.equal(navigator.seek(12, { thenPlay: true }), "queued");
  assert.equal(navigator.baseFrame(), 12);
  assert.equal(navigator.flush(), false);
  controller.state.requestPending = false;
  controller.state.cursor = { frame_index: 8, final_frame_index: 20 };
  assert.equal(navigator.flush(), true);
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(controller.calls.slice(2), ["seek:12", "play"]);
  controller.state.requestPending = false;
  assert.equal(navigator.stepPlay(), "sent");
  assert.equal(controller.calls.at(-1), "one_tick");
  controller.state.cursor = /** @type {any} */ (null);
  assert.equal(navigator.seek(1), "refused");
});

class FakeNode {
  /** @param {FakeDocument} ownerDocument @param {string} tagName */
  constructor(ownerDocument, tagName) {
    this.ownerDocument = ownerDocument;
    this.tagName = tagName.toUpperCase();
    /** @type {FakeNode[]} */
    this.childNodes = [];
    /** @type {FakeNode | null} */
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
    this.value = "";
    this.classList = {
      /** @param {string} name */
      add: (name) => {
        this.className = `${this.className} ${name}`.trim();
      },
    };
  }
  get firstElementChild() {
    return this.childNodes[0] ?? null;
  }
  /** @param {string} name @param {string} value */
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
  /** @param {...FakeNode} nodes */
  append(...nodes) {
    for (const node of nodes) {
      node.parentElement = this;
      this.childNodes.push(node);
    }
  }
  /** @param {...FakeNode} nodes */
  replaceChildren(...nodes) {
    this.childNodes = [];
    this.append(...nodes);
  }
  /** @param {string} type @param {(event: any) => void} handler */
  addEventListener(type, handler) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), handler]);
  }
  /** @param {string} type @param {any} [event] */
  fire(type, event = {}) {
    for (const handler of this.listeners.get(type) ?? [])
      handler({ stopPropagation() {}, ...event });
  }
  focus() {}
  /** @returns {string} */
  serialize() {
    const attributes = [...this.attributes.entries()]
      .filter(([name]) => name !== "aria-description" && name !== "data-tooltip-owner")
      .map(([name, value]) => `${name}="${value}"`)
      .join(" ");
    return `<${this.tagName} ${attributes}>${this.textContent}${this.childNodes
      .map((node) => node.serialize())
      .join("")}</${this.tagName}>`;
  }
  /** @param {(node: FakeNode) => boolean} predicate @returns {FakeNode[]} */
  findAll(predicate) {
    return [
      ...(predicate(this) ? [this] : []),
      ...this.childNodes.flatMap((node) => node.findAll(predicate)),
    ];
  }
}

class FakeDocument {
  /** @param {string} tag */
  createElement(tag) {
    return new FakeNode(this, tag);
  }
  /** @param {string} _ns @param {string} tag */
  createElementNS(_ns, tag) {
    return new FakeNode(this, tag);
  }
}

/** @param {{resolve: (value: unknown) => void}[]} pending */
function makeView(pending) {
  const doc = new FakeDocument();
  const container = doc.createElement("div");
  const controller = fakeController({ frame: 5 });
  const view = createChronicle(/** @type {any} */ (container), {
    controller,
    loadMarks: () =>
      new Promise((resolve) => {
        pending.push({ resolve });
      }),
    storage: null,
    clipboard: null,
  });
  return { container, controller, view };
}

/** @param {FakeNode} container */
function markPathCount(container) {
  return container.findAll(
    (node) =>
      node.tagName === "PATH" &&
      /chronicle__(fall|rune|pip)/u.test(node.getAttribute("class") ?? ""),
  ).length;
}

async function settle() {
  for (let index = 0; index < 5; index += 1) await Promise.resolve();
}

test("Agent POV: a view change drops old marks at once, refuses a foreign payload and redraws fresh", async () => {
  /** @type {{resolve: (value: unknown) => void}[]} */
  const pending = [];
  const { container, controller, view } = makeView(pending);
  const oracle = /** @type {const} */ ({ kind: "oracle" });
  const pov = /** @type {const} */ ({
    kind: "agent_pov",
    recipientPublicAgentId: "agent-3",
  });
  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  await settle();
  pending.shift()?.resolve(rawMarks());
  await settle();
  const freshOracle = container.serialize();
  assert.ok(markPathCount(container) > 0);
  assert.equal(
    container.findAll((node) => node.id === "chronicle-caption")[0].textContent,
    "Fight 2 Of 3 · Ticks 5–7",
  );

  view.update({ playback: controller.snapshot(), audience: pov, replayKey: "r1" });
  assert.equal(markPathCount(container), 0);
  assert.equal(
    container.findAll((node) => node.id === "chronicle-caption")[0].textContent,
    "Finding Fights As Seen By Agent 3",
  );
  await settle();
  pending.shift()?.resolve(rawMarks("oracle", null));
  await settle();
  assert.equal(markPathCount(container), 0);
  assert.equal(
    container.findAll((node) => node.id === "chronicle-caption")[0].textContent,
    "Fights Unavailable",
  );

  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  assert.equal(container.serialize(), freshOracle);
  view.dispose();
});

test("the Chronicle buttons seek through the queue and the view starts hidden", async () => {
  /** @type {{resolve: (value: unknown) => void}[]} */
  const pending = [];
  const { container, controller, view } = makeView(pending);
  assert.equal(view.buttons.nextFight.disabled, true);
  view.update({
    playback: controller.snapshot(),
    audience: { kind: "oracle" },
    replayKey: null,
  });
  await settle();
  pending.shift()?.resolve(rawMarks());
  await settle();
  assert.equal(container.hidden, false);
  assert.equal(view.buttons.bookmark.disabled, true);
  /** @type {any} */ (view.buttons.nextFight).fire("click");
  assert.equal(controller.calls.at(-1), "seek:15");
  /** @type {any} */ (view.buttons.previousFight).fire("click");
  assert.equal(controller.calls.at(-1), "seek:15");
  controller.state.requestPending = false;
  view.update({
    playback: controller.snapshot(),
    audience: { kind: "oracle" },
    replayKey: null,
  });
  assert.equal(controller.calls.at(-1), "seek:5");
  view.update({
    playback: { ...controller.snapshot(), cursor: null },
    audience: { kind: "oracle" },
  });
  assert.equal(container.hidden, true);
  view.dispose();
});

test("ribbon SVG is export-safe and the module reads no frame data and binds no keys", () => {
  const marks = normalizeTimelineMarks(rawMarks());
  const records = chronicleRibbonRecords({
    marks,
    frameIndex: 5,
    finalFrameIndex: 20,
    width: 432,
  });
  const built = buildChronicleRibbon(records, /** @type {any} */ (new FakeDocument()));
  const markup = /** @type {any} */ (built.svg).serialize();
  assert.doesNotMatch(markup, /gradient|mask|filter|url\(|<IMAGE/iu);
  assert.equal(built.tickHits.length, 4);
  assert.equal(built.chapterHits.length, 3);
  const source = readFileSync(new URL("../src/chronicle.js", import.meta.url), "utf8");
  assert.doesNotMatch(
    source,
    /researcher_space|match_summary|latest_events|visual_events|scene\./u,
  );
  assert.doesNotMatch(source, /(document|window|globalThis)\.addEventListener/u);
  assert.doesNotMatch(source, /registerLocalCommandHandler|command-set\.js/u);
});
