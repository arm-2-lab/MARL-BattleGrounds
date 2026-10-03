/**
 * @file Check the beat clock owned by choreography-plan.js and played by
 * choreography.js: the schedule numbers (bodies glide 0-260 ms, blows from 280 ms,
 * Settle to 1100 ms, the Fall beat at 1100 ms plus 60 ms per earlier fall ordered by
 * public ID, the 900 ms Return beat, the 600 ms end hold, Longer Tick 1700 ms and
 * 3200 ms with a death, 1500 ms and 3000 ms without it, and the short reduced-motion
 * copy); the beat order; Oracle View timing from latest_events only and Agent POV
 * timing from visual_events only (a death the team did not see changes no time, and
 * the score always changes at the Fall beat start); beatView's held life, score and
 * returns read from the current frame only; and the choreographer's beat holds,
 * which use fill "backwards" and never a forward fill, so the paused still shows END
 * and a cancelled, finished or replaced hold shows END too.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { CombatChoreographer } from "../src/choreography.js";
import {
  BEAT_CLOCK,
  beatRealTime,
  beatSchedule,
  beatView,
  buildChoreographyPlan,
} from "../src/choreography-plan.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  enableAllVisualFilters,
} from "../src/visual-filters.js";

const ALL_FILTERS = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);

/** @type {Readonly<Record<string, number>>} */
const PHASE_RANK = Object.freeze({
  ability_activated: 20,
  recipient_health_resolution: 40,
  agent_died: 90,
  lethal_damage_contribution: 90,
  status_applied: 100,
  agent_respawned: 120,
});

const surface = Object.freeze({
  worldToScreen: (/** @type {any} */ point) => ({
    x: Number(point.x ?? point[0]) * 10,
    y: 120 - Number(point.y ?? point[1]) * 10,
  }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 10,
  viewportBounds: Object.freeze({
    left: 0,
    top: 0,
    right: 800,
    bottom: 600,
    width: 800,
    height: 600,
  }),
  protectedRects: Object.freeze([]),
});

/** @returns {Promise<Record<string, any>>} */
async function fixture() {
  return JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
}

/** @param {Record<string, any>} inventory @param {string} publicId */
function trajectory(inventory, publicId) {
  const row = inventory.agent_phase_trajectories.find(
    (/** @type {Record<string, any>} */ candidate) =>
      candidate.agent_public_agent_id === publicId,
  );
  assert.ok(row, `missing trajectory ${publicId}`);
  return row;
}

/** @param {Record<string, any>} inventory @param {Record<string, any>[]} events
 * @param {string} separator @param {string} prefix */
function installEvents(inventory, events, separator, prefix) {
  const ordered = events
    .map(
      (event) =>
        /** @type {Record<string, any>} */ ({
          ...event,
          phase_rank: PHASE_RANK[event.event_kind],
        }),
    )
    .sort((left, right) => left.phase_rank - right.phase_rank)
    .map(
      (event, ordinal) =>
        /** @type {Record<string, any>} */ ({
          ...event,
          event_id: `${prefix}:${separator}:${String(ordinal).padStart(4, "0")}`,
          ordinal,
        }),
    );
  inventory.events = ordered;
  inventory.event_count = ordered.length;
  inventory.ordered_event_ids = ordered.map(({ event_id }) => event_id);
  inventory.ordered_event_kinds = ordered.map(({ event_kind }) => event_kind);
}

/** @param {{homeFall?: boolean, scores?: boolean}} [options] */
async function oracleDeathFrame(options = {}) {
  const raw = structuredClone((await fixture()).presentations.replay_oracle);
  const latest = raw.latest_events;
  const anchor = (/** @type {string} */ id) =>
    structuredClone(trajectory(latest, id).successor);
  installEvents(
    latest,
    [
      {
        event_kind: "agent_died",
        recipient_anchor: anchor("agent-slot-6"),
        home_fall: options.homeFall === true,
      },
      {
        event_kind: "lethal_damage_contribution",
        source_anchor: anchor("agent-slot-0"),
        recipient_anchor: anchor("agent-slot-6"),
        attributed_death_damage: 40,
      },
      { event_kind: "agent_died", recipient_anchor: anchor("agent-slot-5") },
      {
        event_kind: "agent_respawned",
        agent_anchor: anchor("agent-slot-1"),
        team_id: 1,
        realized_successor_position: [...anchor("agent-slot-1").position],
      },
    ],
    "event",
    latest.incoming_transition_id,
  );
  if (options.scores) {
    Object.assign(raw.match_summary, {
      task_mode: 1,
      score_threshold: 20,
      outcome: "in_progress",
      previous_scores: [3, 1],
      scores: [6, 1],
      returned_agent_ids: ["agent-slot-1"],
    });
  }
  return normalizeAuthorizedPresentationFrameV1(raw);
}

/** @param {boolean} seenDeath */
async function povFrame(seenDeath) {
  const raw = structuredClone(
    (await fixture()).presentations.replay_no_shared_obs_agent_pov,
  );
  const visual = raw.visual_events;
  if (seenDeath) {
    installEvents(
      visual,
      [
        ...visual.events,
        {
          event_kind: "agent_died",
          recipient_anchor: structuredClone(
            trajectory(visual, "agent-slot-1").successor,
          ),
        },
      ],
      "visual-event",
      visual.incoming_recipient_transition_id,
    );
  }
  return normalizeAuthorizedPresentationFrameV1(raw);
}

test("beat schedule numbers follow the beat clock with Longer Tick on and off", () => {
  const quiet = beatSchedule();
  assert.equal(quiet.glide, true);
  assert.equal(quiet.stepGlide, true);
  assert.equal(quiet.longTick, true);
  assert.equal(quiet.lead, 200);
  assert.equal(quiet.blowSpeed, 1);
  assert.deepEqual(quiet.windows, {
    stride: [0, 280],
    glide: [0, 260],
    chargeRush: [0, 110],
    strike: [280, 440],
    impact: [440, 620],
    settle: [620, 1100],
  });
  assert.equal(quiet.choreographyEnd, 1100);
  assert.equal(quiet.fall, null);
  assert.equal(quiet.returnBeat, null);
  assert.equal(quiet.scoreAt, 1100);
  assert.equal(quiet.explanationEnd, 1100);
  assert.deepEqual(quiet.endHold, [1100, 1700]);
  assert.equal(quiet.tickMs, 1700);
  assert.equal(quiet.reducedScale, 0.2);
  assert.equal(quiet.reducedTickMs, 820);

  const deaths = beatSchedule({
    falls: [
      { presentationKey: "k10", publicAgentId: "agent-slot-10" },
      { presentationKey: "k2", publicAgentId: "agent-slot-2" },
      { presentationKey: "k9", publicAgentId: "agent-slot-9" },
    ],
  });
  assert.deepEqual(deaths.fall, [1100, 2600]);
  assert.deepEqual(
    deaths.falls.map(({ publicAgentId, order, at }) => [publicAgentId, order, at]),
    [
      ["agent-slot-2", 0, 1100],
      ["agent-slot-9", 1, 1160],
      ["agent-slot-10", 2, 1220],
    ],
  );
  assert.equal(deaths.tickMs, 3200);
  assert.equal(deaths.scoreAt, 1100);

  const returns = beatSchedule({
    falls: [{ presentationKey: "a", publicAgentId: "agent-slot-1" }],
    returns: [{ presentationKey: "b", publicAgentId: "agent-slot-7" }],
  });
  assert.deepEqual(returns.returnBeat, [2600, 3500]);
  assert.deepEqual(
    returns.returns.map(({ at, setDownAt }) => [at, setDownAt]),
    [[2600, 2900]],
  );
  assert.equal(returns.tickMs, 4100);

  const off = beatSchedule({
    falls: [{ presentationKey: "a", publicAgentId: "agent-slot-1" }],
    visualFilters: { scholar_long_tick: false },
  });
  assert.equal(off.longTick, false);
  assert.equal(off.lead, 0);
  assert.deepEqual(
    [off.windows.strike, off.windows.impact, off.windows.settle].map((window) =>
      window.map(Math.round),
    ),
    [
      [280, 401],
      [401, 537],
      [537, 900],
    ],
  );
  assert.ok(Math.abs(off.blowSpeed - 820 / 620) < 1e-12);
  assert.deepEqual(off.fall, [900, 2400]);
  assert.equal(off.tickMs, 3000);
  assert.equal(
    beatSchedule({ visualFilters: { scholar_long_tick: false } }).tickMs,
    1500,
  );

  const still = beatSchedule({ visualFilters: { stride_glide: false } });
  assert.equal(still.glide, false);
  assert.equal(still.longTick, false);
  assert.equal(still.windows.glide, null);
  assert.equal(still.windows.stride, null);
  assert.deepEqual(still.windows.strike, [80, 240]);
  assert.deepEqual(still.windows.settle, [420, 900]);

  const noDeathBeat = beatSchedule({
    falls: [
      { presentationKey: "a", publicAgentId: "agent-slot-1" },
      { presentationKey: "b", publicAgentId: "agent-slot-2" },
    ],
    visualFilters: { death_effects: false },
  });
  assert.equal(noDeathBeat.fall, null);
  assert.deepEqual(
    noDeathBeat.falls.map(({ at }) => at),
    [1100, 1100],
  );
  assert.equal(noDeathBeat.tickMs, 1700);

  const noReturnBeat = beatSchedule({
    returns: [{ presentationKey: "b", publicAgentId: "agent-slot-7" }],
    visualFilters: { respawn_wave: false },
  });
  assert.equal(noReturnBeat.returnBeat, null);
  assert.deepEqual(
    noReturnBeat.returns.map(({ at, setDownAt }) => [at, setDownAt]),
    [[1080, 1080]],
  );
  const noSetDown = beatSchedule({
    returns: [{ presentationKey: "b", publicAgentId: "agent-slot-7" }],
    visualFilters: { resurrection_effects: false },
  });
  assert.deepEqual(
    noSetDown.returns.map(({ at, setDownAt }) => [at, setDownAt]),
    [[1100, 1100]],
  );
  assert.equal(deaths.reducedTickMs, 2600 * 0.2 + 600);
});

test("blow time bends around the glide and keeps its own speed with Longer Tick", () => {
  assert.deepEqual(
    [0, 30, 60, 70, 80, 360, 760, 900, 2400].map((part) =>
      beatRealTime(part, true, 200),
    ),
    [0, 40, 80, 180, 280, 560, 960, 1100, 2600],
  );
  assert.equal(beatRealTime(80, true, 0), 280);
  assert.equal(beatRealTime(900, true, 0), 900);
  assert.equal(beatRealTime(2400, true, 0), 2400);
  for (const part of [0, 80, 360, 900, 1500]) {
    assert.equal(beatRealTime(part, false, 200), part);
  }
  assert.equal(BEAT_CLOCK.glideEnd, 260);
  assert.equal(BEAT_CLOCK.blowStart, 280);
  assert.equal(BEAT_CLOCK.stepGlide, 240);
  assert.equal(BEAT_CLOCK.stepCueFadeEnd, 320);
});

test("beats follow one another: glide, strike, impact, settle, fall, return, end hold", () => {
  const schedule = beatSchedule({
    falls: [{ presentationKey: "a", publicAgentId: "agent-slot-1" }],
    returns: [{ presentationKey: "b", publicAgentId: "agent-slot-7" }],
  });
  const glide = /** @type {readonly [number, number]} */ (schedule.windows.glide);
  const fall = /** @type {readonly [number, number]} */ (schedule.fall);
  const returnBeat = /** @type {readonly [number, number]} */ (schedule.returnBeat);
  const order = [
    glide[0],
    glide[1],
    schedule.windows.strike[0],
    schedule.windows.impact[0],
    schedule.windows.settle[0],
    schedule.windows.settle[1],
    fall[0],
    fall[1],
    returnBeat[0],
    returnBeat[1],
    schedule.endHold[0],
    schedule.endHold[1],
  ];
  for (let index = 1; index < order.length; index += 1) {
    assert.ok(order[index] >= order[index - 1], `beat ${index} comes in order`);
  }
  assert.ok(glide[1] < schedule.windows.strike[0]);
  assert.equal(fall[0], schedule.windows.settle[1]);
  assert.equal(returnBeat[0], fall[1]);
  assert.equal(schedule.endHold[0], returnBeat[1]);
  assert.equal(schedule.tickMs, schedule.endHold[1]);
});

test("Oracle View times falls, returns and derived rows only from its own latest_events", async () => {
  const frame = await oracleDeathFrame({ homeFall: true });
  const plan = buildChoreographyPlan(frame, surface, ALL_FILTERS);
  assert.ok(plan);
  assert.deepEqual(plan.beats.fall, [1100, 2600]);
  assert.deepEqual(
    plan.beats.falls.map(({ publicAgentId, at }) => [publicAgentId, at]),
    [
      ["agent-slot-5", 1100],
      ["agent-slot-6", 1160],
    ],
  );
  assert.deepEqual(plan.beats.returnBeat, [2600, 3500]);
  assert.equal(plan.beats.tickMs, 4100);
  const deathPulses = plan.events.filter(
    (event) => event.cueSemantic === "agent_died" && event.kind === "semantic_pulse",
  );
  assert.deepEqual(
    deathPulses.map(({ agentPublicAgentId, phaseStart, phaseEnd }) => [
      agentPublicAgentId,
      phaseStart,
      phaseEnd,
    ]),
    [
      ["agent-slot-6", 1160, 2600],
      ["agent-slot-5", 1100, 2600],
    ],
  );
  const respawn = plan.events.find((event) => event.cueSemantic === "agent_respawned");
  assert.deepEqual([respawn?.phaseStart, respawn?.phaseEnd], [2600, 3500]);
  assert.equal(plan.phases.total, 3500);
  assert.equal(plan.phases.reducedTotal, 3500 * plan.beats.reducedScale);
  assert.equal(plan.phases.submissionRelease, 450);

  const feed = plan.derivedEvents.filter((event) => event.kind === "fallen_feed_row");
  assert.deepEqual(
    feed.map((row) => [
      row.victim.publicAgentId,
      row.killers.map(
        (/** @type {Record<string, any>} */ killer) => killer.publicAgentId,
      ),
      row.points,
      row.phaseStart,
    ]),
    [
      ["agent-slot-5", [], null, 1100],
      ["agent-slot-6", ["agent-slot-0"], 2, 1160],
    ],
  );
  const herald = plan.derivedEvents.find((event) => event.kind === "herald_card");
  assert.equal(herald?.count, 2);
  assert.equal(herald?.phaseStart, 1100);
  const homeFalls = plan.derivedEvents.filter((event) => event.kind === "home_fall");
  assert.deepEqual(
    homeFalls.map(({ teamId, count, phaseStart }) => [teamId, count, phaseStart]),
    [[2, 1, 1160]],
  );

  const withoutFlag = buildChoreographyPlan(
    await oracleDeathFrame(),
    surface,
    ALL_FILTERS,
  );
  assert.ok(withoutFlag);
  assert.equal(
    withoutFlag.derivedEvents.some((event) => event.kind === "home_fall"),
    false,
  );
  assert.deepEqual(withoutFlag.beats, plan.beats);
});

test("Agent POV times only the deaths its team saw and its score always waits for 1100 ms", async () => {
  const quiet = await povFrame(false);
  const seen = await povFrame(true);
  const quietPlan = buildChoreographyPlan(quiet, surface, ALL_FILTERS);
  const seenPlan = buildChoreographyPlan(seen, surface, ALL_FILTERS);
  assert.ok(quietPlan);
  assert.ok(seenPlan);
  assert.equal(quietPlan.beats.fall, null);
  assert.equal(quietPlan.beats.tickMs, 1700);
  assert.deepEqual(seenPlan.beats.fall, [1100, 2600]);
  assert.deepEqual(
    seenPlan.beats.falls.map(({ publicAgentId }) => publicAgentId),
    ["agent-slot-1"],
  );
  assert.equal(seenPlan.beats.tickMs, 3200);
  for (const frame of [quiet, seen]) {
    const view = beatView(frame, 0);
    assert.ok(view);
    assert.equal(view.audience, "agent_pov");
    assert.equal(view.score.at, 1100);
  }
  const seenView = beatView(seen, 1099);
  assert.ok(seenView);
  assert.deepEqual(
    seenView.life.map(({ publicAgentId, at, held }) => [publicAgentId, at, held]),
    [["agent-slot-1", 1100, true]],
  );
  const feed = seenPlan.derivedEvents.filter(
    (event) => event.kind === "fallen_feed_row",
  );
  assert.equal(feed.length, 1);
  assert.equal(
    seenPlan.derivedEvents.find((event) => event.kind === "herald_card")?.count,
    1,
  );
});

test("beatView holds life, score and returns until their beats from the current frame", async () => {
  const frame = await oracleDeathFrame({ scores: true });
  const start = beatView(frame, "start");
  const beforeFall = beatView(frame, 1099);
  const afterFirstFall = beatView(frame, 1100);
  const end = beatView(frame);
  assert.ok(start && beforeFall && afterFirstFall && end);
  assert.deepEqual(start.score.shown, [3, 1]);
  assert.deepEqual(beforeFall.score.shown, [3, 1]);
  assert.equal(beforeFall.score.held, true);
  assert.deepEqual(afterFirstFall.score.shown, [6, 1]);
  assert.deepEqual(end.score, {
    before: [3, 1],
    after: [6, 1],
    changed: true,
    at: 1100,
    held: false,
    shown: [6, 1],
  });
  assert.deepEqual(
    afterFirstFall.life.map(({ publicAgentId, held }) => [publicAgentId, held]),
    [
      ["agent-slot-5", false],
      ["agent-slot-6", true],
    ],
  );
  assert.ok(end.life.every((entry) => entry.held === false));
  assert.deepEqual(
    end.returns.map(({ publicAgentId, seen, at, back }) => [
      publicAgentId,
      seen,
      at,
      back,
    ]),
    [["agent-slot-1", true, 2600, true]],
  );
  assert.equal(beatView(frame, 2599)?.returns[0].back, false);
  assert.deepEqual(beatView(frame, "end"), end);
  assert.equal(beatView({ not: "a frame" }), null);
});

// Stand-ins for the durable DOM: only the attribute and selector reads the
// choreographer makes.
class FakeNode {
  /** @param {Record<string, string>} [attributes] @param {FakeNode[]} [children] */
  constructor(attributes = {}, children = []) {
    this.attributes = new Map(Object.entries(attributes));
    this.children = children;
  }

  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  /** @returns {FakeNode[]} */
  descendants() {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }

  /** @param {string} selector */
  querySelectorAll(selector) {
    const name = selector.slice(1, -1);
    return this.descendants().filter((node) => node.attributes.has(name));
  }

  /** @param {string} _selector */
  querySelector(_selector) {
    return (
      this.descendants().find(
        (node) =>
          node.attributes.get("class") === "agent" &&
          node.attributes.has("data-presentation-key"),
      ) ?? null
    );
  }
}

class RecordingFactory {
  constructor() {
    /** @type {Array<Record<string, any>>} */
    this.created = [];
  }

  /** @param {Record<string, any>} spec */
  create(spec) {
    const animation = {
      spec,
      id: spec.id,
      currentTime: 0,
      playbackRate: 1,
      state: "running",
      finished: new Promise(() => {}),
      pause() {
        animation.state = "paused";
      },
      play() {
        animation.state = "running";
      },
      finish() {
        animation.state = "finished";
      },
      cancel() {
        animation.state = "cancelled";
      },
    };
    this.created.push(animation);
    return animation;
  }

  /** @param {unknown} element @param {number} duration @param {string} id */
  createClock(element, duration, id) {
    return this.create({ element, keyframes: [], options: { duration }, id });
  }
}

const painter = {
  install: () => ({
    root: {},
    nodeCount: 3,
    persistentNodeCount: 0,
    animationSpecs: [],
  }),
  clear: () => {},
  settle: () => {},
  reproject: () => {},
};

test("beat holds use fill backwards, no forward fill, and every stop shows END", async () => {
  const frame = await oracleDeathFrame();
  const probe = buildChoreographyPlan(frame, surface, ALL_FILTERS);
  assert.ok(probe);
  const victim = probe.beats.falls[1];
  const before = new FakeNode({
    "data-beat-hold": "fall",
    "data-beat-key": victim.presentationKey,
    "data-beat-side": "before",
    "data-beat-fade": "300",
  });
  const after = new FakeNode({
    "data-beat-hold": "fall",
    "data-beat-key": victim.presentationKey,
    "data-beat-side": "after",
    opacity: "0.9",
  });
  const hiddenDeath = new FakeNode({
    "data-beat-hold": "fall",
    "data-beat-key": "a-champion-with-no-seen-fall",
    "data-beat-side": "before",
  });
  const score = new FakeNode({ "data-beat-hold": "score", "data-beat-side": "after" });
  const battlefield = new FakeNode({}, [before, after, hiddenDeath, score]);
  const domSurface = /** @type {any} */ ({
    ...surface,
    battlefield,
    layer: {},
    ownerDocument: {},
    viewportKey: "beat-test",
  });
  const factory = new RecordingFactory();
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (painter),
    animationFactory: /** @type {any} */ (factory),
    planBuilder: (value) => buildChoreographyPlan(value, surface, ALL_FILTERS),
  });

  controller.presentFrame(frame, domSurface, { renderPolicy: "replay_static" });
  assert.equal(factory.created.length, 0, "the paused still is END with no override");

  controller.presentFrame(frame, domSurface, {
    renderPolicy: "replay_animated",
    restartAnimated: true,
  });
  const holds = factory.created.filter(({ id }) => id.includes(":hold:"));
  assert.deepEqual(
    holds.map(({ spec }) => [
      spec.element === before ? "before" : spec.element === after ? "after" : "score",
      spec.keyframes,
      spec.options,
    ]),
    [
      [
        "before",
        [{ opacity: 1 }, { opacity: 0 }],
        { delay: 1160, duration: 300, fill: "backwards" },
      ],
      [
        "after",
        [{ opacity: 0 }, { opacity: 0.9 }],
        { delay: 1160, duration: 1, fill: "backwards" },
      ],
      [
        "score",
        [{ opacity: 0 }, { opacity: 1 }],
        { delay: 1100, duration: 1, fill: "backwards" },
      ],
    ],
  );
  for (const { spec } of factory.created) {
    assert.notEqual(spec.options.fill, "forwards");
    assert.notEqual(spec.options.fill, "both");
  }
  assert.ok(holds.every(({ id }) => id.startsWith("mbg:")));
  const cleanup = factory.created.find(({ id }) => id.endsWith(":cleanup"));
  assert.equal(cleanup?.spec.options.duration, 3500 + 600);

  controller.skip();
  assert.ok(holds.every(({ state }) => state === "finished" || state === "cancelled"));

  controller.presentFrame(frame, domSurface, { renderPolicy: "replay_static" });
  controller.presentFrame(frame, domSurface, {
    renderPolicy: "replay_animated",
    restartAnimated: true,
  });
  const second = factory.created.filter(({ id }) => id.includes(":hold:")).slice(-3);
  assert.equal(second.length, 3);
  assert.ok(second.every(({ state }) => state === "running"));
  controller.presentFrame(await povFrame(false), domSurface, {
    renderPolicy: "live_once",
  });
  assert.ok(
    second.every(({ state }) => state === "cancelled"),
    "a new frame ends beats at END",
  );

  const reducedFactory = new RecordingFactory();
  const reduced = new CombatChoreographer({
    painter: /** @type {any} */ (painter),
    animationFactory: /** @type {any} */ (reducedFactory),
    motionMode: "reduced",
    planBuilder: (value) => buildChoreographyPlan(value, surface, ALL_FILTERS),
  });
  reduced.presentFrame(frame, domSurface, { renderPolicy: "replay_animated" });
  const reducedBefore = reducedFactory.created.find(
    ({ spec }) => spec.element === before,
  );
  assert.equal(reducedBefore?.spec.options.delay, 1160 * probe.beats.reducedScale);
  assert.equal(
    reducedFactory.created.some(({ id }) => id.includes(":glide:")),
    false,
    "reduced motion never glides",
  );
  controller.dispose();
  reduced.dispose();
});
