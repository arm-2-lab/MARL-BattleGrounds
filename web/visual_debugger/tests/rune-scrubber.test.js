/**
 * @file Check the Rune Scrubber (src/rune-scrubber.js): the timeline marks payload
 * check (schema, audience and recipient, order and range, and no fight list: a
 * payload with "fights" is refused), audience keys, the "Tick 4 of 20" readout and
 * the beat label ("Settled", "Playing", "Moving", "Paused" and "Paused · Strike"),
 * Event Track records (frame-to-x, the 36 px lanes from SCRUBBER_LANES, runes on the
 * groove with damaging ones last, the current tick split, the veil, switches), hover
 * rows with "Team B Warrior · ID 6" labels and titles such as "Tick 2 · Team B
 * Warrior Fell", beat times and beat names from a choreography schedule, track seeks
 * through the controller's one queue (a press pauses first, the thumb shows the
 * target in flight or queued, a button press adds to it, a track press replaces it,
 * queued presses cost one round trip, no cursor refuses), the
 * paused beat label of a mid-tick pause (held still like the arena), notices under
 * the readout (role="status", aria-live polite, the whole sentence, 4 s, a newer one
 * restarts the 4 s, an empty one or dispose ends it, unknown levels show as info),
 * and the Agent POV boundary: a view change drops the old marks at
 * once, a payload for another view is refused, Oracle -> Agent POV -> Oracle redraws
 * exactly as a fresh view, and the module reads no frame data, binds no keys and
 * uses no browser storage.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  buildScrubberTrack,
  createRuneScrubber,
  marksMatchAudience,
  normalizeTimelineMarks,
  SCRUBBER_LANES,
  SCRUBBER_TRACK_HEIGHT,
  ScrubberNavigator,
  scrubberAgentLabel,
  scrubberAudienceKey,
  scrubberBeatAt,
  scrubberBeatLabel,
  scrubberBeatsFromSchedule,
  scrubberReadout,
  scrubberTrackRecords,
  tickMarkRows,
  tickMarkTitle,
} from "../src/rune-scrubber.js";
import { ReplayPlaybackController } from "../src/replay-controls.js";

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
        returns: [agent("agent-6", 2, 2), agent("agent-7", 2, 3)],
      },
      {
        frame_index: 15,
        falls: [{ ...agent("agent-7", 2, 3), home_fall: false }],
        ultimates: [],
        returns: [],
      },
    ],
  };
}

const PLAYBACK = Object.freeze({
  cursor: { frame_index: 4, final_frame_index: 20 },
  playing: false,
  requestPending: false,
  connected: true,
  hidden: false,
  transportState: "SETTLED",
  pauseReason: null,
  playbackRate: 1,
});

test("timeline marks are checked and frozen, and a fight list is refused", () => {
  const marks = normalizeTimelineMarks(rawMarks());
  assert.equal(marks.audience, "oracle");
  assert.equal(marks.ticks[0].falls[0].homeFall, true);
  assert.equal(marks.ticks[1].ultimates[0].damaging, true);
  assert.ok(Object.isFrozen(marks.ticks[0].falls[0]));
  assert.equal("fights" in marks, false);
  const pov = normalizeTimelineMarks(rawMarks("agent_pov", "agent-3"));
  assert.equal(pov.recipientPublicAgentId, "agent-3");

  const bad = [
    { ...rawMarks(), extra: 1 },
    { ...rawMarks(), fights: [] },
    { ...rawMarks(), schema_version: 2 },
    rawMarks("oracle", "agent-3"),
    rawMarks("agent_pov", null),
    { ...rawMarks(), final_frame_index: 10 },
    { ...rawMarks(), ticks: [...rawMarks().ticks].reverse() },
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
  assert.equal(scrubberAudienceKey({ kind: "oracle" }), "oracle");
  assert.equal(
    scrubberAudienceKey({ kind: "agent_pov", recipientPublicAgentId: "agent-3" }),
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
  assert.equal(scrubberAgentLabel("agent-3"), "Agent 3");
  assert.equal(scrubberAgentLabel("scout"), "Agent scout");
});

test("the readout says the tick and the beat label says the beat", () => {
  assert.equal(scrubberReadout(PLAYBACK), "Tick 4 of 20");
  assert.equal(
    scrubberReadout(PLAYBACK, { tickForFrameIndex: (frame) => frame + 100 }),
    "Tick 104 of 120",
  );
  assert.equal(scrubberReadout({ ...PLAYBACK, cursor: null }), "Tick —");
  assert.equal(scrubberBeatLabel(PLAYBACK), "Settled");
  assert.equal(scrubberBeatLabel({ ...PLAYBACK, playing: true }, "Strike"), "Playing");
  assert.equal(scrubberBeatLabel({ ...PLAYBACK, requestPending: true }), "Moving");
  const paused = { ...PLAYBACK, pauseReason: "user_pause" };
  assert.equal(scrubberBeatLabel(paused), "Paused");
  assert.equal(scrubberBeatLabel(paused, "Strike"), "Paused · Strike");
  assert.equal(scrubberBeatLabel({ ...PLAYBACK, pauseReason: "user_seek" }), "Settled");
});

test("track records place frames, keep the lanes inside 36 px and veil the ticks ahead", () => {
  assert.equal(SCRUBBER_TRACK_HEIGHT, 36);
  const groove = SCRUBBER_LANES.groove;
  assert.equal(SCRUBBER_LANES.rune, groove.y + groove.height / 2);
  assert.ok(SCRUBBER_LANES.fallA.y + SCRUBBER_LANES.fallA.height <= groove.y);
  assert.ok(SCRUBBER_LANES.fallB.y >= groove.y + groove.height);
  assert.ok(
    SCRUBBER_LANES.pipA.y + SCRUBBER_LANES.pipA.height <= SCRUBBER_LANES.fallA.y,
  );
  assert.ok(
    SCRUBBER_LANES.pipB.y >= SCRUBBER_LANES.fallB.y + SCRUBBER_LANES.fallB.height,
  );
  assert.ok(SCRUBBER_LANES.pipB.y + SCRUBBER_LANES.pipB.height <= 36);
  const marks = normalizeTimelineMarks(rawMarks());
  const records = scrubberTrackRecords({
    marks,
    frameIndex: 5,
    finalFrameIndex: 20,
    width: 432,
  });
  assert.equal(records.height, 36);
  assert.equal(records.perTick, 20);
  assert.equal(records.playhead.x, 116);
  assert.equal(records.currentMarks.hasRunes, true);
  assert.equal(records.currentMarks.hasFalls, false);
  assert.equal(records.currentMarks.runes.length, 2);
  assert.equal(records.currentMarks.runes.at(-1)?.damaging, true, "damaging on top");
  assert.match(
    records.currentMarks.runes[0].d,
    /^M116 15\.5L118\.5 18/u,
    "on the groove",
  );
  assert.notEqual(records.staticMarks.fallB, "");
  assert.match(records.staticMarks.fallB, /^M55 22h2v9/u, "Team B below the groove");
  assert.match(records.staticMarks.fallA, /^M155 5h2v9/u, "Team A above the groove");
  assert.match(records.staticMarks.pipB, /^M154\.5 33h3v4/u, "two returns: 4 px");
  assert.deepEqual(records.veil, { x: 106, width: 314 });
  assert.equal(records.hitTicks.length, 4);

  const hidden = scrubberTrackRecords({
    marks,
    frameIndex: 6,
    finalFrameIndex: 20,
    width: 432,
    hideMarksAhead: true,
  });
  assert.equal(hidden.hitTicks.length, 2);

  const plain = scrubberTrackRecords({
    marks,
    frameIndex: 5,
    finalFrameIndex: 20,
    width: 432,
    showMarks: false,
  });
  assert.equal(plain.hitTicks.length, 0);
  assert.equal(plain.staticMarks.fallA + plain.staticMarks.fallB, "");
  const empty = scrubberTrackRecords({
    marks: null,
    frameIndex: 0,
    finalFrameIndex: 0,
    width: 10,
  });
  assert.equal(empty.width, 64);
  assert.equal(empty.veil, null);
});

test("hover rows and titles name champions with Team A or Team B and ability names", () => {
  const marks = normalizeTimelineMarks(rawMarks());
  assert.deepEqual(tickMarkRows(marks.ticks[0]), [
    { label: "Fell", value: "Team B Warrior · ID 6 · Home Fall, +2" },
  ]);
  const rows = tickMarkRows(marks.ticks[1]);
  assert.equal(rows[0].value, "Team A Rogue · ID 1 · Crippling Poison");
  assert.equal(rows[1].value, "Team A Mage · ID 2 · Burst");
  assert.deepEqual(tickMarkRows(marks.ticks[2]).at(-1), {
    label: "Returned",
    value: "Team B Hunter · ID 7",
  });
  assert.equal(tickMarkTitle(marks.ticks[0], 2), "Tick 2 · Team B Warrior Fell");
  assert.equal(
    tickMarkTitle(marks.ticks[1], 5),
    "Tick 5 · Team A Rogue Used Crippling Poison · 1 More",
  );
  assert.equal(
    tickMarkTitle(marks.ticks[2], 107),
    "Tick 107 · Team A Rogue Fell · 2 More",
  );
});

test("beat times and beat names come from the choreography schedule", () => {
  const beats = scrubberBeatsFromSchedule({
    windows: { strike: [80, 240], impact: [200, 600], settle: [760, 900] },
    fall: [900, 1300],
    returnBeat: null,
    tickMs: 1700,
  });
  assert.equal(beats?.impactMs, 200);
  assert.equal(beats?.fallMs, 900);
  assert.equal(beats?.returnMs, null);
  assert.equal(beats?.tickMs, 1700);
  assert.equal(scrubberBeatAt(beats, 30), "Muster");
  assert.equal(scrubberBeatAt(beats, 100), "Strike");
  assert.equal(scrubberBeatAt(beats, 220), "Impact", "the later beat wins an overlap");
  assert.equal(scrubberBeatAt(beats, 700), "Hold");
  assert.equal(scrubberBeatAt(beats, 1000), "Fall");
  assert.equal(scrubberBeatAt(beats, 1700), null, "the tick has settled");
  assert.equal(scrubberBeatAt(null, 100), null);
  assert.equal(scrubberBeatsFromSchedule(null), null);
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
  };
  return controller;
}

test("track seeks share the controller's one queue: pause first, the newest target wins", async () => {
  /** @type {Array<Record<string, any>>} */
  const commands = [];
  /** @type {Array<(value: unknown) => void>} */
  const answers = [];
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      commands.push(command);
      return new Promise((resolve) => {
        answers.push(resolve);
      });
    },
    waitForPresentation: () => new Promise(() => {}),
  });
  /** @param {number} frame @param {number} generation */
  const at = (frame, generation) => ({
    schema_version: 1,
    frame_index: frame,
    final_frame_index: 20,
    cursor_generation: generation,
    choreography_generation: 1,
  });
  controller.setConnected(true);
  controller.installCursor(at(3, 3));
  assert.equal(controller.play(), true);
  const navigator = new ScrubberNavigator(controller);
  assert.equal(navigator.seek(8), "sent");
  assert.equal(controller.snapshot().playing, false, "the press paused play first");
  assert.equal(navigator.baseFrame(), 8, "the thumb shows the seek in flight");
  void controller.next();
  assert.equal(navigator.baseFrame(), 9, "a +1 button press adds to it");
  assert.equal(navigator.seek(12), "queued");
  assert.equal(navigator.baseFrame(), 12, "a track press replaces the target");
  answers.shift()?.({ frame: { cursor: at(8, 4) } });
  for (let index = 0; index < 6; index += 1) await Promise.resolve();
  assert.deepEqual(
    commands.map((command) => command.frame_index),
    [8, 12],
    "one round trip for the queued presses",
  );
  answers.shift()?.({ frame: { cursor: at(12, 5) } });
  for (let index = 0; index < 6; index += 1) await Promise.resolve();
  assert.equal(navigator.baseFrame(), 12);
  assert.equal(controller.snapshot().targetFrameIndex, null);
  controller.setAuthorityPending("test");
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
    return { left: 0, top: 0, width: 432, height: 36, right: 432, bottom: 36 };
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
  focus() {}
  /** @returns {string} */
  serialize() {
    const attributes = [...this.attributes.entries()]
      .filter(([name]) => name !== "aria-description" && name !== "data-tooltip-owner")
      .map(([name, value]) => `${name}="${value}"`)
      .join(" ");
    const data = JSON.stringify(this.dataset);
    return `<${this.tagName} ${attributes} ${data}>${this.textContent}${this.childNodes
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
  addEventListener() {}
  removeEventListener() {}
}

/** @param {{resolve: (value: unknown) => void}[]} pending */
function makeView(pending) {
  const doc = new FakeDocument();
  const container = doc.createElement("div");
  const controller = fakeController({ frame: 5 });
  const view = createRuneScrubber(/** @type {any} */ (container), {
    controller,
    loadMarks: () =>
      new Promise((resolve) => {
        pending.push({ resolve });
      }),
  });
  return { container, controller, view };
}

/** @param {FakeNode} container */
function markPathCount(container) {
  return container.findAll(
    (node) =>
      node.tagName === "PATH" &&
      /rune-scrubber__(fall|rune|pip)/u.test(node.getAttribute("class") ?? ""),
  ).length;
}

/** @param {FakeNode} container @param {string} id */
function byId(container, id) {
  return container.findAll((node) => node.id === id)[0];
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
  const track = /** @type {FakeNode} */ (container.childNodes[0]);
  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  await settle();
  pending.shift()?.resolve(rawMarks());
  await settle();
  const freshOracle = container.serialize();
  assert.ok(markPathCount(container) > 0);
  assert.equal(track.dataset.marks, "ready");
  assert.equal(byId(container, "rune-scrubber-readout").textContent, "Tick 5 of 20");
  assert.equal(byId(container, "rune-scrubber-beat").textContent, "Settled");

  view.update({ playback: controller.snapshot(), audience: pov, replayKey: "r1" });
  assert.equal(markPathCount(container), 0);
  assert.equal(track.dataset.marks, "loading");
  await settle();
  pending.shift()?.resolve(rawMarks("oracle", null));
  await settle();
  assert.equal(markPathCount(container), 0);
  assert.equal(track.dataset.marks, "failed");

  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  assert.equal(container.serialize(), freshOracle);
  assert.equal(pending.length, 0, "the cached Oracle marks need no new request");
  view.update({
    playback: { ...controller.snapshot(), cursor: null },
    audience: oracle,
  });
  assert.equal(container.hidden, true, "an undocked view hides with no cursor");
  view.dispose();
});

test("a pause in the middle of a tick names the frozen beat and holds it", async (t) => {
  /** @type {{resolve: (value: unknown) => void}[]} */
  const pending = [];
  const { container, controller, view } = makeView(pending);
  const clock = { now: 1000 };
  t.mock.method(globalThis.performance, "now", () => clock.now);
  const oracle = /** @type {const} */ ({ kind: "oracle" });
  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  await settle();
  pending.shift()?.resolve(rawMarks());
  await settle();
  controller.state.playing = true;
  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  const beat = byId(container, "rune-scrubber-beat");
  assert.equal(beat.textContent, "Playing");
  view.presentTick(
    5,
    scrubberBeatsFromSchedule({
      windows: { strike: [80, 240], impact: [360, 600] },
      fall: null,
      returnBeat: null,
      tickMs: 1700,
    }),
    2,
  );
  clock.now = 1060;
  controller.state.playing = false;
  controller.state.pauseReason = "user_pause";
  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  assert.equal(beat.textContent, "Paused · Strike", "60 ms at 2x is 120 plan ms");
  clock.now = 9000;
  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  assert.equal(beat.textContent, "Paused · Strike", "the label holds still");
  controller.state.pauseReason = "user_seek";
  view.update({ playback: controller.snapshot(), audience: oracle, replayKey: "r1" });
  assert.equal(beat.textContent, "Settled");
  view.dispose();
});

test("a notice takes the beat label's line for 4 s, whole, then the beat label returns", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  /** @type {{resolve: (value: unknown) => void}[]} */
  const pending = [];
  const { container, controller, view } = makeView(pending);
  view.update({ playback: controller.snapshot(), audience: { kind: "oracle" } });
  const notice = byId(container, "rune-scrubber-notice");
  const readoutBox = /** @type {FakeNode} */ (
    byId(container, "rune-scrubber-beat").parentElement?.parentElement
  );
  assert.equal(notice.hidden, true);
  assert.equal(notice.getAttribute("role"), "status");
  assert.equal(notice.getAttribute("aria-live"), "polite");
  const sentence = "Exported a-very-long-file-name-for-tick-58.png. It stays whole.";
  view.showNotice(sentence, "success");
  assert.equal(notice.hidden, false);
  assert.equal(notice.textContent, sentence, "never cut");
  assert.equal(notice.dataset.level, "success");
  assert.equal(readoutBox.dataset.notice, "on");
  t.mock.timers.tick(3999);
  assert.equal(notice.hidden, false);
  view.showNotice("Replay navigation failed.", "error");
  assert.equal(notice.dataset.level, "error");
  t.mock.timers.tick(3999);
  assert.equal(notice.hidden, false, "a newer notice starts the 4 s again");
  t.mock.timers.tick(1);
  assert.equal(notice.hidden, true);
  assert.equal(notice.textContent, "");
  assert.equal("notice" in readoutBox.dataset, false, "the beat label is back");
  assert.equal(byId(container, "rune-scrubber-beat").textContent, "Settled");
  view.showNotice("Unknown level shows as info.", "chatty");
  assert.equal(notice.dataset.level, "info");
  view.showNotice("");
  assert.equal(notice.hidden, true, "an empty text ends the notice at once");
  view.showNotice("Shown at dispose.");
  view.dispose();
  view.showNotice("After dispose.");
  assert.equal(notice.textContent, "Shown at dispose.");
});

test("track SVG is export-safe and the module reads no frame data, binds no keys and stores nothing", () => {
  const marks = normalizeTimelineMarks(rawMarks());
  const records = scrubberTrackRecords({
    marks,
    frameIndex: 5,
    finalFrameIndex: 20,
    width: 432,
  });
  const built = buildScrubberTrack(records, /** @type {any} */ (new FakeDocument()));
  const markup = /** @type {any} */ (built.svg).serialize();
  assert.doesNotMatch(markup, /gradient|mask|filter|url\(|<IMAGE|style=/iu);
  assert.match(markup, /class="rune-scrubber__thumb" cx="116" cy="18" r="6"/u);
  assert.equal(built.tickHits.length, 4);
  const source = readFileSync(
    new URL("../src/rune-scrubber.js", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(
    source,
    /researcher_space|match_summary|latest_events|visual_events|scene\./u,
  );
  assert.doesNotMatch(
    source,
    /(doc|document|window|globalThis)\.addEventListener\(\s*"key/u,
  );
  assert.doesNotMatch(source, /registerLocalCommandHandler|command-set\.js/u);
  assert.doesNotMatch(source, /localStorage|sessionStorage|\.style\./u);
  assert.doesNotMatch(source, /previousFight|nextFight|fightsOnly|bookmark|stepPlay/iu);
});
