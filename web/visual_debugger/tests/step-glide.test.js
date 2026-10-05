/**
 * @file Check body glides, the one-tick step and the marks that ride them.
 * Played ticks glide every moved body from its START spot to its END spot in
 * 0-260 ms (a charging Warrior reaches its post-Charge spot at 110 ms first) as
 * FLIP overrides on the body wrappers, with fill "backwards", composite "add"
 * and no gate on the 450 ms live release. A +1 (replay_step one tick forward)
 * plays its tick exactly as Play does (replay_animated, no 600 ms end hold),
 * lets a repaint's still request play on, becomes an ordinary played tick when
 * Play is pressed, jumps to its END when a further step comes, and settles into
 * the paused still; motion off or Step Glide off shows the still. A -1 rewinds on top of the still:
 * bodies glide 240 ms back (a Charge rush back in the first 140 ms), the cues fade
 * in from 240 ms to 320 ms and the clock (which paints nothing, in the browser
 * factory too) settles it paused; a champion that fell in the undone tick rises
 * first (200 ms, its wrapper fading in, everything else waiting), one that had
 * come back to its pad fades back to its corpse without gliding, and copies of
 * the corpse, the pad body and the old veil (inert: no keys, focus or labels)
 * fade out and are removed when the rewind ends or anything replaces it; a
 * snapshot is used once. A fast next -1 starts from where the running glide
 * draws the body; a resize mid-rewind drops the copies; an error while
 * settling a +1 into the still clears and frees every settlement waiter; a -1
 * onto a first frame publishes its running glide; a backward step never plays.
 * Marks with data-glide-follow ride their champion's glide
 * (same keyframes) in a played tick and a rewind, one a repaint replaced joins
 * at the glide's moment and only once (a landing wait likewise at the tick
 * clock's moment, ids ":rejoin:<n>"), waits are created before beat holds,
 * and data-landing-wait marks of a gliding
 * champion fade in at 260 ms (played) or 240 ms (rewind) while a still
 * champion's mark is left alone; a busy tick whose holds exceed the plan's
 * override allowance still keeps every rider, landing wait and veil beat. A first frame has no plan: a +1 from it plays
 * tick 1 and a -1 back onto it glides each body from its END back to START.
 * Jumps, a changed audience or recipient, a re-render of the same frame,
 * reduced motion and Agent POV bodies unseen in either frame never rewind. The
 * replay controller ends one-tick moves (+1 and -1, the arrow keys, the
 * slider's own keys and a slider drag) with the replay_step intent, snaps
 * jumps, Shift + arrows and slider clicks, makes N do exactly what +1 and Arrow
 * Right do (one tick forward, the same request), and sends V, B and F through
 * the one command set (F pauses play first).
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { BrowserAnimationFactory, CombatChoreographer } from "../src/choreography.js";
import { beatSchedule, buildChoreographyPlan } from "../src/choreography-plan.js";
import { registerLocalCommandHandler } from "../src/command-set.js";
import {
  bindReplayTimelineControls,
  ReplayPlaybackController,
  replaySeekCommand,
} from "../src/replay-controls.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  enableAllVisualFilters,
  setVisualFilterEnabled,
} from "../src/visual-filters.js";

const ALL_FILTERS = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);

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

  /** @param {string} name */
  hasAttribute(name) {
    return this.attributes.has(name);
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

/** @param {string[]} keys */
function battlefieldWithBodies(keys) {
  /** @type {Map<string, FakeNode>} */
  const wrappers = new Map();
  for (const key of keys) {
    wrappers.set(
      key,
      new FakeNode({ class: "agent-glide", "data-glide-key": `["s","${key}"]` }, [
        new FakeNode({ class: "agent", "data-presentation-key": key }),
      ]),
    );
  }
  return { battlefield: new FakeNode({}, [...wrappers.values()]), wrappers };
}

class ControlledFactory {
  constructor() {
    /** @type {Array<Record<string, any>>} */
    this.created = [];
  }

  /** @param {Record<string, any>} spec */
  create(spec) {
    /** @type {(value?: unknown) => void} */
    let resolve = () => {};
    const finished = new Promise((done) => {
      resolve = done;
    });
    const animation = {
      spec,
      id: spec.id,
      currentTime: /** @type {number | null} */ (0),
      playbackRate: 1,
      state: "running",
      finished,
      resolve,
      pause() {
        animation.state = "paused";
      },
      play() {
        animation.state = "running";
      },
      finish() {
        animation.state = "finished";
        resolve();
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

  /** @param {string} part */
  matching(part) {
    return this.created.filter(({ id }) => id.includes(part));
  }
}

function fakePainter() {
  return {
    install: () => ({
      root: { name: "root" },
      connectorRoot: { name: "connectors" },
      routeRoot: { name: "routes" },
      nodeCount: 3,
      persistentNodeCount: 0,
      animationSpecs: [],
    }),
    clear: () => {},
    settle: () => {},
    reproject: () => {},
  };
}

/** @param {string} key @param {readonly [number, number]} end @param {Record<string, any>} [extra] */
function body(key, end, extra = {}) {
  return {
    presentationKey: key,
    publicAgentId: `agent-${key}`,
    teamId: 1,
    alive: true,
    start: null,
    postCharge: null,
    end,
    charged: false,
    died: false,
    returned: false,
    glide: false,
    ...extra,
  };
}

/** @param {number} step @param {Array<Record<string, any>>} bodies @param {Record<string, any>} [extra] */
function stepPlan(step, bodies, extra = {}) {
  return {
    epochKey: `epoch-${step}`,
    authorizationKey: `authority-${step}`,
    audienceKey: "audience-researcher",
    fingerprint: `events-${step}`,
    paintKey: "paint",
    simulatorStep: step,
    phases: { total: 0, reducedTotal: 0, submissionRelease: 450 },
    beats: beatSchedule(),
    bodies,
    bounds: { nodes: 16, animations: 8, persistentNodes: 2, overrideAnimations: 40 },
    events: [],
    ...extra,
  };
}

/** @param {{motionMode?: "normal" | "reduced" | "off"}} [options] */
function stepHarness(options = {}) {
  const { battlefield, wrappers } = battlefieldWithBodies(["a", "b", "c", "d"]);
  const factory = new ControlledFactory();
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (factory),
    motionMode: options.motionMode ?? "normal",
    planBuilder: (frame) => /** @type {any} */ (frame).plan,
  });
  const domSurface = /** @type {any} */ ({
    ...surface,
    battlefield,
    layer: {},
    ownerDocument: {},
    viewportKey: "step-test",
  });
  /** @param {Record<string, any>} plan @param {"replay_static" | "replay_step" | "replay_animated" | "live_once"} policy */
  const show = (plan, policy) =>
    controller.presentFrame({ plan }, domSurface, { renderPolicy: policy });
  return { controller, factory, show, wrappers };
}

/** @param {Record<string, any>} animation */
function translations(animation) {
  return animation.spec.keyframes.map((/** @type {Record<string, any>} */ frame) => [
    frame.offset,
    frame.transform,
  ]);
}

test("played ticks glide every moved body START to END in 0-260 ms with the Charge rush first", async () => {
  const raw = structuredClone(
    JSON.parse(
      await readFile(
        new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
        "utf8",
      ),
    ).presentations.replay_oracle,
  );
  // Only the bodies move in this tick: no event anchors the old START spots.
  Object.assign(raw.latest_events, {
    events: [],
    event_count: 0,
    ordered_event_ids: [],
    ordered_event_kinds: [],
  });
  for (const row of raw.latest_events.agent_phase_trajectories) {
    if (row.agent_public_agent_id === "agent-slot-0") {
      row.transition_start.position = [1.0, 1.5];
      row.post_charge.position = [1.0, 1.5];
    }
    if (row.agent_public_agent_id === "agent-slot-1") {
      row.transition_start.position = [4.5, 3.75];
      row.post_charge.position = [2.5, 3.75];
    }
  }
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const plan = buildChoreographyPlan(frame, surface, ALL_FILTERS);
  assert.ok(plan);
  const moved = plan.bodies.filter((entry) => entry.glide);
  assert.deepEqual(
    moved.map(({ publicAgentId, charged }) => [publicAgentId, charged]),
    [
      ["agent-slot-0", false],
      ["agent-slot-1", true],
    ],
  );
  const { battlefield, wrappers } = battlefieldWithBodies(
    plan.bodies.map(({ presentationKey }) => presentationKey),
  );
  const factory = new ControlledFactory();
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (factory),
    planBuilder: (value) => buildChoreographyPlan(value, surface, ALL_FILTERS),
  });
  const domSurface = /** @type {any} */ ({
    ...surface,
    battlefield,
    layer: {},
    ownerDocument: {},
    viewportKey: "played",
  });
  controller.presentFrame(frame, domSurface, { renderPolicy: "live_once" });
  const glides = factory.matching(":glide:");
  assert.equal(glides.length, 2);
  const mage = glides.find(({ id }) => id.endsWith(":glide:agent-slot-0"));
  const warrior = glides.find(({ id }) => id.endsWith(":glide:agent-slot-1"));
  assert.ok(mage && warrior);
  assert.equal(mage.spec.element, wrappers.get(moved[0].presentationKey));
  assert.deepEqual(translations(mage), [
    [0, "translate(-5px, 0px)"],
    [1, "translate(0px, 0px)"],
  ]);
  assert.deepEqual(translations(warrior), [
    [0, "translate(30px, 0px)"],
    [110 / 260, "translate(10px, 0px)"],
    [1, "translate(0px, 0px)"],
  ]);
  for (const glide of glides) {
    assert.deepEqual(glide.spec.options, {
      delay: 0,
      duration: 260,
      fill: "backwards",
      composite: "add",
    });
  }
  assert.equal(controller.snapshot().submissionBlocked, false, "a glide never gates");
  const cleanup = factory.created.find(({ id }) => id.endsWith(":cleanup"));
  assert.equal(cleanup?.spec.options.duration, 260);

  const stillFactory = new ControlledFactory();
  const still = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (stillFactory),
    planBuilder: (value) =>
      buildChoreographyPlan(
        value,
        surface,
        setVisualFilterEnabled(ALL_FILTERS, "stride_glide", false),
      ),
  });
  still.presentFrame(frame, domSurface, { renderPolicy: "live_once" });
  assert.equal(stillFactory.matching(":glide:").length, 0, "Body Glide off snaps");
  controller.presentFrame(frame, domSurface, { renderPolicy: "replay_static" });
  assert.ok(glides.every(({ state }) => state === "cancelled"));
  controller.dispose();
  still.dispose();
});

/** @param {Record<string, any> | undefined} animation @param {Record<string, any>} [options] */
async function finishAndFlush(animation, options = {}) {
  animation?.finish();
  for (let turn = 0; turn < (options.turns ?? 3); turn += 1) await Promise.resolve();
}

test("+1 plays its tick as Play does, then settles into the paused still", async () => {
  const before = stepPlan(5, [body("a", [1, 1]), body("b", [5, 5])]);
  const forward = stepPlan(6, [
    body("a", [2, 1], { start: [1, 1], glide: true }),
    body("b", [7, 5], {
      start: [5, 5],
      glide: true,
      charged: true,
      postCharge: [6, 5],
    }),
  ]);
  const { controller, factory, show, wrappers } = stepHarness();
  show(before, "replay_static");
  show(forward, "replay_step");
  assert.equal(controller.snapshot().renderPolicy, "replay_animated");
  assert.equal(controller.snapshot().paused, false);
  assert.equal(factory.matching(":step-glide:").length, 0, "+1 never uses the rewind");
  const glides = factory.matching(":glide:");
  assert.deepEqual(
    glides.map(({ spec }) => spec.element),
    [wrappers.get("a"), wrappers.get("b")],
  );
  assert.ok(glides.every(({ spec }) => spec.options.duration === 260));
  const cleanup = factory.created.find(({ id }) => id === "mbg:epoch-6:cleanup");
  assert.equal(
    cleanup?.spec.options.duration,
    Math.max(Number(forward.beats.explanationEnd ?? 0), 260),
    "a stepped tick has no 600 ms end hold",
  );
  const created = factory.created.length;
  show(forward, "replay_static");
  assert.equal(factory.created.length, created, "a repaint lets the tick play on");
  assert.equal(controller.snapshot().renderPolicy, "replay_animated");
  await finishAndFlush(cleanup);
  assert.equal(controller.snapshot().renderPolicy, "replay_static");
  assert.equal(controller.snapshot().paused, true, "it settles into the paused still");
  assert.equal(controller.snapshot().animationCount, 0);
  controller.dispose();

  // Play pressed while the stepped tick plays: it ends as an ordinary played tick.
  const played = stepHarness();
  played.show(before, "replay_static");
  played.show(forward, "replay_step");
  played.show(forward, "replay_animated");
  await finishAndFlush(
    played.factory.created.find(({ id }) => id === "mbg:epoch-6:cleanup"),
  );
  assert.equal(played.controller.snapshot().renderPolicy, "replay_animated");
  played.controller.dispose();

  // A further +1 jumps the playing tick to its END and plays the next one.
  const quick = stepHarness();
  quick.show(before, "replay_static");
  quick.show(forward, "replay_step");
  const first = quick.factory.matching(":glide:");
  quick.show(
    stepPlan(7, [body("a", [3, 1], { start: [2, 1], glide: true }), body("b", [7, 5])]),
    "replay_step",
  );
  assert.ok(first.every(({ state }) => state === "cancelled"));
  assert.deepEqual(
    quick.factory
      .matching(":glide:")
      .filter(({ state }) => state === "running")
      .map(({ id }) => id),
    ["mbg:epoch-7:glide:agent-a"],
  );
  quick.controller.dispose();

  // Motion off has no played tick, and Step Glide off asks for none: +1 shows
  // the still at once.
  const off = stepHarness({ motionMode: "off" });
  off.show(before, "replay_static");
  off.show(forward, "replay_step");
  assert.equal(off.controller.snapshot().renderPolicy, "replay_static");
  off.controller.dispose();
  const noStep = stepHarness();
  noStep.show(before, "replay_static");
  noStep.show(
    { ...forward, beats: beatSchedule({ visualFilters: { step_glide: false } }) },
    "replay_step",
  );
  assert.equal(noStep.controller.snapshot().renderPolicy, "replay_static");
  assert.equal(noStep.factory.matching(":glide:").length, 0);
  noStep.controller.dispose();
});

test("-1 rewinds: bodies glide back 240 ms, then the cues fade in by 320 ms", async () => {
  const { controller, factory, show } = stepHarness();
  const before = stepPlan(5, [body("a", [1, 1]), body("b", [5, 5])]);
  const after = stepPlan(6, [
    body("a", [2, 1]),
    body("b", [7, 5], { charged: true, postCharge: [6, 5], start: [5, 5] }),
  ]);
  show(after, "replay_static");
  show(before, "replay_step");
  assert.equal(controller.snapshot().renderPolicy, "replay_static");
  const glides = factory.matching(":step-glide:");
  assert.deepEqual(
    glides.map((animation) => translations(animation)),
    [
      [
        [0, "translate(10px, 0px)"],
        [1, "translate(0px, 0px)"],
      ],
      [
        [0, "translate(20px, 0px)"],
        [140 / 240, "translate(10px, 0px)"],
        [1, "translate(0px, 0px)"],
      ],
    ],
  );
  assert.ok(
    glides.every(
      ({ spec }) =>
        spec.options.delay === 0 &&
        spec.options.duration === 240 &&
        spec.options.fill === "backwards" &&
        spec.options.composite === "add",
    ),
  );
  const fades = factory.matching(":step-cue:");
  assert.deepEqual(
    fades.map(({ spec }) => [spec.element.name, spec.keyframes, spec.options]),
    ["routes", "connectors", "root"].map((name) => [
      name,
      [{ opacity: 0 }, { opacity: 1 }],
      { delay: 240, duration: 80, fill: "backwards" },
    ]),
  );
  const clock = factory.created.find(({ id }) => id === "mbg:epoch-5:step");
  assert.equal(clock?.spec.options.duration, 320);
  assert.deepEqual(clock?.spec.keyframes, [], "the clock paints nothing");
  assert.equal(controller.snapshot().paused, false);
  await finishAndFlush(clock);
  assert.equal(controller.snapshot().paused, true, "the rewind settles into the still");
  assert.equal(controller.snapshot().animationCount, 0);
  assert.ok([...glides, ...fades].every(({ state }) => state === "cancelled"));
  const created = factory.created.length;
  show(before, "replay_step");
  assert.equal(
    factory.created.length,
    created,
    "a re-render of the shown frame never moves",
  );
  controller.dispose();
});

test("-1 across a fall raises the champion first and a return goes back to its corpse", async () => {
  const { controller, factory, show, wrappers } = stepHarness();
  // Tick 6: c fell where it ended; d came back to its pad.
  const after = stepPlan(6, [
    body("a", [2, 1]),
    body("c", [4, 3], { alive: false, died: true }),
    body("d", [0, 0], { returned: true }),
  ]);
  const before = stepPlan(5, [
    body("a", [1, 1]),
    body("c", [3, 3]),
    body("d", [6, 6], { alive: false }),
  ]);
  show(after, "replay_static");
  show(before, "replay_step");
  const rises = factory.matching(":rise:");
  assert.deepEqual(
    rises.map(({ spec }) => [spec.element, spec.keyframes, spec.options]),
    [wrappers.get("c"), wrappers.get("d")].map((element) => [
      element,
      [{ opacity: 0 }, { opacity: 1 }],
      { delay: 0, duration: 200, fill: "backwards" },
    ]),
    "the risen body and the corpse fade in where they stand",
  );
  const glides = factory.matching(":step-glide:");
  assert.deepEqual(
    glides.map(({ spec }) => spec.element),
    [wrappers.get("a"), wrappers.get("c")],
    "the champion that came back to its pad does not glide",
  );
  assert.ok(
    glides.every(({ spec }) => spec.options.delay === 200),
    "the move waits for the rise",
  );
  assert.deepEqual(translations(glides[1]), [
    [0, "translate(10px, 0px)"],
    [1, "translate(0px, 0px)"],
  ]);
  assert.ok(
    factory.matching(":step-cue:").every(({ spec }) => spec.options.delay === 440),
  );
  const clock = factory.created.find(({ id }) => id === "mbg:epoch-5:step");
  assert.equal(clock?.spec.options.duration, 520);
  await finishAndFlush(clock);
  assert.equal(controller.snapshot().paused, true);
  controller.dispose();
});

test("a step off and back onto a first frame with no plan plays forward and glides back to START", async () => {
  const { controller, factory, show, wrappers } = stepHarness();
  // The first frame: no tick leads into it, so it has no plan.
  show(/** @type {any} */ (null), "replay_static");
  const first = stepPlan(1, [
    body("a", [2, 1], { start: [1, 1], glide: true }),
    body("b", [7, 5], { start: [5, 5], glide: true }),
    body("c", [3, 3], { start: [3, 3], died: true, alive: false }),
    body("d", [9, 9]),
  ]);
  show(first, "replay_step");
  assert.equal(
    controller.snapshot().renderPolicy,
    "replay_animated",
    "+1 plays tick 1",
  );
  await finishAndFlush(factory.created.find(({ id }) => id === "mbg:epoch-1:cleanup"));
  assert.equal(controller.snapshot().paused, true);

  show(/** @type {any} */ (null), "replay_step");
  const back = factory
    .matching(":step-glide:")
    .filter(({ state }) => state === "running");
  assert.deepEqual(
    back.map(({ spec }) => spec.element),
    [wrappers.get("a"), wrappers.get("b")],
  );
  assert.deepEqual(translations(back[0]), [
    [0, "translate(10px, 0px)"],
    [1, "translate(0px, 0px)"],
  ]);
  const clock = factory.created.find(({ id }) => id === "mbg:epoch-1:step-back");
  assert.equal(clock?.spec.options.duration, 240);
  await finishAndFlush(clock);
  assert.equal(
    controller.snapshot().paused,
    true,
    "the step back settles into the still",
  );
  assert.ok(back.every(({ state }) => state === "cancelled"));
  controller.dispose();
});

test("rewinding fast starts the next -1 from where the running glide draws the body", () => {
  const { factory, show, controller } = stepHarness();
  show(stepPlan(3, [body("a", [4, 0])]), "replay_static");
  show(stepPlan(2, [body("a", [2, 0])]), "replay_step");
  const first = factory.matching(":step-glide:")[0];
  first.currentTime = 120;
  show(stepPlan(1, [body("a", [0, 0])]), "replay_step");
  assert.equal(first.state, "cancelled");
  const second = factory.matching(":step-glide:").at(-1);
  assert.ok(second);
  assert.match(second.id, /^mbg:epoch-1:/u);
  assert.deepEqual(translations(second), [
    [0, "translate(30px, 0px)"],
    [1, "translate(0px, 0px)"],
  ]);
  controller.dispose();
});

test("jumps, audience changes and unseen Agent POV bodies never rewind", () => {
  const cases = [
    ["a ten-tick jump back", stepPlan(1, [body("a", [9, 9])])],
    [
      "another audience or recipient",
      stepPlan(5, [body("a", [9, 9])], { audienceKey: "audience-pov-recipient-2" }),
    ],
    ["a body unseen in the later frame", stepPlan(5, [body("c", [9, 9])])],
    [
      "Step Glide off",
      stepPlan(5, [body("a", [9, 9])], {
        beats: beatSchedule({ visualFilters: { step_glide: false } }),
      }),
    ],
  ];
  for (const [label, next] of cases) {
    const { factory, show, controller } = stepHarness();
    show(stepPlan(6, [body("a", [1, 1]), body("b", [2, 2])]), "replay_static");
    show(/** @type {Record<string, any>} */ (next), "replay_step");
    assert.equal(factory.matching(":step-glide:").length, 0, String(label));
    controller.dispose();
  }
  const pov = stepHarness();
  pov.show(stepPlan(6, [body("a", [2, 1]), body("b", [2, 2])]), "replay_static");
  pov.show(stepPlan(5, [body("a", [1, 1])]), "replay_step");
  assert.deepEqual(
    pov.factory.matching(":step-glide:").map(({ id }) => id),
    ["mbg:epoch-5:step-glide:agent-a"],
    "a body unseen in the earlier frame is absent from it and never glides",
  );
  pov.controller.dispose();

  const reduced = stepHarness({ motionMode: "reduced" });
  reduced.show(stepPlan(6, [body("a", [2, 1])]), "replay_static");
  reduced.show(stepPlan(5, [body("a", [1, 1])]), "replay_step");
  assert.equal(
    reduced.factory.created.length,
    0,
    "reduced motion shows the still on -1",
  );
  reduced.controller.dispose();
});

test("marks ride their champion's glide and next-action marks wait for the landing", async () => {
  /** @param {Record<string, string>} attributes */
  const mark = (attributes) => new FakeNode(attributes);
  const { battlefield, wrappers } = battlefieldWithBodies(["a", "b"]);
  const ring = mark({ "data-glide-follow": "a" });
  const thread = mark({ "data-landing-wait": "a b" });
  const stillThread = mark({ "data-landing-wait": "b" });
  battlefield.children.push(ring, thread, stillThread);
  const factory = new ControlledFactory();
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (factory),
    planBuilder: (frame) => /** @type {any} */ (frame).plan,
  });
  const domSurface = /** @type {any} */ ({
    ...surface,
    battlefield,
    layer: {},
    ownerDocument: {},
    viewportKey: "riders",
  });
  /** @param {Record<string, any>} plan @param {string} policy */
  const show = (plan, policy) =>
    controller.presentFrame({ plan }, domSurface, {
      renderPolicy: /** @type {any} */ (policy),
    });
  show(stepPlan(5, [body("a", [1, 1]), body("b", [5, 5])]), "replay_static");
  const forward = stepPlan(6, [
    body("a", [2, 1], { start: [1, 1], glide: true }),
    body("b", [5, 5]),
  ]);
  show(forward, "replay_step");
  const bodyGlide = factory.created.find(
    ({ id }) => id === "mbg:epoch-6:glide:agent-a",
  );
  const ride = factory.created.find(
    ({ id }) => id === "mbg:epoch-6:glide:agent-a:follow:0",
  );
  assert.equal(bodyGlide?.spec.element, wrappers.get("a"));
  assert.equal(ride?.spec.element, ring, "the ring rides the champion");
  assert.deepEqual(ride?.spec.keyframes, bodyGlide?.spec.keyframes);
  const waits = factory.matching(":landing-wait:");
  assert.deepEqual(
    waits.map(({ spec }) => [spec.element, spec.keyframes, spec.options]),
    [
      [
        thread,
        [{ opacity: 0 }, { opacity: 1 }],
        { delay: 260, duration: 80, fill: "backwards" },
      ],
    ],
    "only the mark of a gliding champion waits",
  );

  // A repaint that replaces the ring mid-glide: the new ring joins the glide.
  const newRing = mark({ "data-glide-follow": "a" });
  battlefield.children.splice(battlefield.children.indexOf(ring), 1, newRing);
  /** @type {any} */ (bodyGlide).currentTime = 100;
  show(forward, "replay_static");
  const rejoin = factory.created.find(
    ({ id, spec }) => /:rejoin:\d+$/u.test(id) && spec.element === newRing,
  );
  assert.equal(rejoin?.spec.element, newRing);
  assert.equal(rejoin?.currentTime, 100, "it joins at the body glide's moment");
  assert.deepEqual(rejoin?.spec.keyframes, bodyGlide?.spec.keyframes);
  const count = factory.created.length;
  show(forward, "replay_static");
  assert.equal(
    factory.created.length,
    count,
    "a rider that rides is never given a second glide",
  );
  controller.dispose();

  // The rewind: the ring rides the glide back; the thread waits for the landing.
  const back = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (factory),
    planBuilder: (frame) => /** @type {any} */ (frame).plan,
  });
  back.presentFrame(
    { plan: stepPlan(6, [body("a", [2, 1]), body("b", [5, 5])]) },
    domSurface,
    { renderPolicy: "replay_static" },
  );
  back.presentFrame(
    { plan: stepPlan(5, [body("a", [1, 1]), body("b", [5, 5])]) },
    domSurface,
    { renderPolicy: "replay_step" },
  );
  assert.equal(
    factory.created.find(({ id }) => id === "mbg:epoch-5:step-glide:agent-a:follow:0")
      ?.spec.element,
    newRing,
  );
  assert.deepEqual(
    factory.created
      .filter(({ id }) => id.startsWith("mbg:epoch-5:landing-wait:"))
      .map(({ spec }) => [spec.element, spec.options.delay]),
    [[thread, 240]],
  );
  back.dispose();
});

test("a busy tick's holds never crowd out the riders, landing waits or veil beats", () => {
  const { battlefield } = battlefieldWithBodies(["a", "b"]);
  const ring = new FakeNode({ "data-glide-follow": "a" });
  const thread = new FakeNode({ "data-landing-wait": "a" });
  const veil = new FakeNode({
    "data-fog-state": "1",
    "data-fog-arrive-kind": "stride",
    opacity: "1",
  });
  const holds = Array.from(
    { length: 6 },
    () =>
      new FakeNode({
        "data-beat-hold": "impact",
        "data-beat-side": "after",
        opacity: "1",
      }),
  );
  battlefield.children.push(ring, thread, veil, ...holds);
  const factory = new ControlledFactory();
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (factory),
    planBuilder: (frame) => /** @type {any} */ (frame).plan,
  });
  const domSurface = /** @type {any} */ ({
    ...surface,
    battlefield,
    layer: {},
    ownerDocument: {},
    viewportKey: "busy",
  });
  const plan = stepPlan(
    6,
    [body("a", [2, 1], { start: [1, 1], glide: true }), body("b", [5, 5])],
    { bounds: { nodes: 16, animations: 8, persistentNodes: 2, overrideAnimations: 1 } },
  );
  controller.presentFrame({ plan }, domSurface, { renderPolicy: "replay_animated" });
  const on = (/** @type {unknown} */ element) =>
    factory.created.filter(({ spec }) => spec.element === element);
  assert.ok(on(ring).length > 0, "the ring rides");
  assert.ok(on(thread).length > 0, "the thread waits");
  assert.ok(on(veil).length > 0, "the veil keeps its beats");
  assert.ok(
    holds.filter((hold) => on(hold).length > 0).length < holds.length,
    "the allowance still bounds the holds",
  );
  controller.dispose();
});

test("a repaint mid-tick gives a replaced next-action mark its wait back, and waits come before holds", async () => {
  const { battlefield } = battlefieldWithBodies(["a", "b"]);
  const thread = new FakeNode({ "data-landing-wait": "a" });
  const hold = new FakeNode({
    "data-beat-hold": "impact",
    "data-beat-side": "after",
    opacity: "1",
  });
  battlefield.children.push(thread, hold);
  const factory = new ControlledFactory();
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (factory),
    planBuilder: (frame) => /** @type {any} */ (frame).plan,
  });
  const domSurface = /** @type {any} */ ({
    ...surface,
    battlefield,
    layer: {},
    ownerDocument: {},
    viewportKey: "waits",
  });
  const plan = stepPlan(6, [
    body("a", [2, 1], { start: [1, 1], glide: true }),
    body("b", [5, 5]),
  ]);
  controller.presentFrame({ plan }, domSurface, { renderPolicy: "replay_animated" });
  const order = factory.created.map(({ id }) => id);
  const firstWait = order.findIndex((id) => id.includes(":landing-wait:"));
  const firstHold = order.findIndex((id) => id.includes(":hold:"));
  assert.ok(firstWait >= 0 && firstHold >= 0 && firstWait < firstHold, "waits first");
  const clock = factory.created.find(({ id }) => id === "mbg:epoch-6:cleanup");
  /** @type {any} */ (clock).currentTime = 120;
  const newThread = new FakeNode({ "data-landing-wait": "a" });
  battlefield.children.splice(battlefield.children.indexOf(thread), 1, newThread);
  controller.presentFrame({ plan }, domSurface, { renderPolicy: "replay_animated" });
  const rejoin = factory.created.find(({ spec }) => spec.element === newThread);
  assert.match(String(rejoin?.id), /:landing-wait:rejoin:\d+$/u);
  assert.deepEqual(rejoin?.spec.options, {
    delay: 260,
    duration: 80,
    fill: "backwards",
  });
  assert.equal(rejoin?.currentTime, 120, "it joins at the tick clock's moment");
  controller.dispose();
});

test("a settle error never hangs, a -1 onto a first frame publishes, a resize drops copies, and a backward step never plays", async () => {
  // The still cannot be built at the end of a +1: the controller clears and
  // frees every settlement waiter instead of hanging.
  let failStill = false;
  const { battlefield } = battlefieldWithBodies(["a"]);
  const factory = new ControlledFactory();
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (factory),
    planBuilder: (frame, _surface, _filters, policy) => {
      if (failStill && policy === "replay_static") throw new Error("no still");
      return /** @type {any} */ (frame).plan;
    },
  });
  const domSurface = /** @type {any} */ ({
    ...surface,
    battlefield,
    layer: {},
    ownerDocument: {},
    viewportKey: "edges",
  });
  /** @param {Record<string, any> | null} plan @param {string} policy @param {Record<string, any>} [extra] */
  const show = (plan, policy, extra = {}) =>
    controller.presentFrame({ plan }, domSurface, {
      renderPolicy: /** @type {any} */ (policy),
      ...extra,
    });
  show(stepPlan(5, [body("a", [1, 1])]), "replay_static");
  show(stepPlan(6, [body("a", [2, 1], { start: [1, 1], glide: true })]), "replay_step");
  failStill = true;
  let settled = false;
  void controller.whenSettled().then(() => {
    settled = true;
  });
  await finishAndFlush(
    factory.created.find(({ id }) => id === "mbg:epoch-6:cleanup"),
    {
      turns: 6,
    },
  );
  assert.equal(settled, true);
  assert.equal(controller.snapshot().animationCount, 0);
  failStill = false;

  // A -1 onto a first frame (no plan) publishes that the glide runs.
  /** @type {Array<Record<string, any>>} */
  const published = [];
  const watched = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (factory),
    planBuilder: (frame) => /** @type {any} */ (frame).plan,
    onStateChange: (snapshot) => published.push(snapshot),
  });
  watched.presentFrame(
    { plan: stepPlan(1, [body("a", [2, 1], { start: [1, 1] })]) },
    domSurface,
    { renderPolicy: "replay_static" },
  );
  watched.presentFrame({ plan: null }, domSurface, { renderPolicy: "replay_step" });
  assert.equal(published.at(-1)?.paused, false, "the last news is the running glide");
  watched.dispose();

  // A backward step from a frame with no plan shows the still, never plays.
  const back = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (factory),
    planBuilder: (frame) => /** @type {any} */ (frame).plan,
  });
  back.presentFrame({ plan: null }, domSurface, { renderPolicy: "replay_static" });
  back.presentFrame({ plan: stepPlan(4, [body("a", [1, 1])]) }, domSurface, {
    renderPolicy: "replay_step",
    stepDirection: "backward",
  });
  assert.equal(back.snapshot().renderPolicy, "replay_static");
  back.dispose();
  controller.dispose();
});

// A small DOM stand-in for the rewind copies: tags, classes, attributes,
// parents, cloning and the few selectors the choreographer uses.
class MiniNode {
  /** @param {string} tag @param {Record<string, string>} [attributes] @param {MiniNode[]} [children] */
  constructor(tag, attributes = {}, children = []) {
    this.tagName = tag;
    this.attributes = new Map(Object.entries(attributes));
    /** @type {MiniNode[]} */
    this.children = [];
    /** @type {MiniNode | null} */
    this.parentElement = null;
    this.connectedRoot = false;
    for (const child of children) this.append(child);
    this.classList = {
      /** @param {string} name */
      contains: (name) => (this.getAttribute("class") ?? "").split(" ").includes(name),
      /** @param {string} from @param {string} to */
      replace: (from, to) =>
        this.setAttribute(
          "class",
          (this.getAttribute("class") ?? "")
            .split(" ")
            .map((name) => (name === from ? to : name))
            .join(" "),
        ),
    };
  }

  get isConnected() {
    /** @type {MiniNode | null} */
    let at = this;
    while (at?.parentElement) at = at.parentElement;
    return at?.connectedRoot === true;
  }

  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  /** @param {string} name */
  hasAttribute(name) {
    return this.attributes.has(name);
  }

  /** @param {string} name @param {string} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  /** @param {string} name */
  removeAttribute(name) {
    this.attributes.delete(name);
  }

  /** @param {MiniNode} child */
  append(child) {
    child.remove();
    child.parentElement = this;
    this.children.push(child);
  }

  /** @param {MiniNode} child @param {MiniNode | null} reference */
  insertBefore(child, reference) {
    child.remove();
    child.parentElement = this;
    const index = reference ? this.children.indexOf(reference) : -1;
    this.children.splice(index < 0 ? this.children.length : index, 0, child);
  }

  remove() {
    const parent = this.parentElement;
    if (parent) parent.children.splice(parent.children.indexOf(this), 1);
    this.parentElement = null;
  }

  /** @param {boolean} _deep @returns {MiniNode} */
  cloneNode(_deep) {
    return new MiniNode(
      this.tagName,
      Object.fromEntries(this.attributes),
      this.children.map((child) => child.cloneNode(true)),
    );
  }

  /** @returns {MiniNode[]} */
  descendants() {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }

  /** @param {string} selector */
  matches(selector) {
    if (selector === "*") return true;
    const parsed = /^(\w+)?(?:\.([\w-]+))?(?:\[([\w-]+)\])?$/u.exec(selector);
    if (!parsed) return false;
    const [, tag, className, attribute] = parsed;
    return (
      (!tag || this.tagName === tag) &&
      (!className || this.classList.contains(className)) &&
      (!attribute || this.attributes.has(attribute))
    );
  }

  /** @param {string} selector */
  querySelectorAll(selector) {
    if (selector.startsWith(":scope > ")) {
      const inner = selector.slice(9);
      return this.children.filter((child) => child.matches(inner));
    }
    return this.descendants().filter((node) => node.matches(selector));
  }

  /** @param {string} selector */
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }
}

test("a rewind fades copies of the corpse, the pad body and the old veil out, then removes them", async () => {
  /** @param {string} key @param {string} [extra] */
  const wrapper = (key, extra = "") =>
    new MiniNode(
      "g",
      { class: `agent-glide${extra}`, "data-glide-key": `["s","${key}"]` },
      [
        new MiniNode("g", {
          class: "agent",
          "data-presentation-key": key,
          tabindex: "0",
          "aria-label": key,
        }),
      ],
    );
  const living = new MiniNode("g", { "data-body-state": "living" }, [
    wrapper("a"),
    wrapper("d"),
  ]);
  const dead = new MiniNode("g", { "data-body-state": "dead" }, [wrapper("c")]);
  /** @param {string} end the END veil path */
  const sight = (end) =>
    new MiniNode("g", { class: "fog-sight", "data-fog-kind": "sight-veil" }, [
      new MiniNode("g", { class: "fog-veil" }, [
        new MiniNode("path", { "data-fog-state": "0", opacity: "0", d: "M0 0" }),
        new MiniNode("path", { "data-fog-state": "1", opacity: "1", d: end }),
      ]),
    ]);
  const fogLayer = new MiniNode("g", { "data-layer": "fog" }, [sight("M1 1")]);
  const battlefield = new MiniNode("svg", {}, [living, dead, fogLayer]);
  battlefield.connectedRoot = true;
  const factory = new ControlledFactory();
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (fakePainter()),
    animationFactory: /** @type {any} */ (factory),
    planBuilder: (frame) => /** @type {any} */ (frame).plan,
  });
  const domSurface = /** @type {any} */ ({
    ...surface,
    battlefield,
    layer: {},
    ownerDocument: {},
    viewportKey: "copies",
    bodyWrapper: (/** @type {string} */ key) =>
      battlefield
        .querySelectorAll("g.agent-glide[data-glide-key]")
        .find((node) => node.getAttribute("data-glide-key") === `["s","${key}"]`) ??
      null,
  });
  /** @param {Record<string, any>} plan @param {string} policy */
  const show = (plan, policy) =>
    controller.presentFrame({ plan }, domSurface, {
      renderPolicy: /** @type {any} */ (policy),
    });
  const tick6 = stepPlan(6, [
    body("a", [2, 1]),
    body("c", [4, 3], { alive: false, died: true }),
    body("d", [0, 0], { returned: true }),
  ]);
  const tick5 = stepPlan(5, [
    body("a", [1, 1]),
    body("c", [3, 3]),
    body("d", [6, 6], { alive: false }),
  ]);
  show(tick6, "replay_static");
  controller.captureStepSnapshot(domSurface);
  // The durable render of tick 5: c is alive again, d is a corpse again, the
  // veil has a new END.
  const risen = dead.children[0];
  const fallenAgain = living.children[1];
  living.append(risen);
  dead.append(fallenAgain);
  // scene.js replaces the whole sight group whenever the veil changes.
  fogLayer.children[0].remove();
  fogLayer.append(sight("M2 2"));
  show(tick5, "replay_step");
  const copies = battlefield.querySelectorAll("[data-rewind-copy]");
  assert.equal(copies.length, 3, "the corpse, pad body and old veil copies are back");
  const corpse = copies.find((node) => node.parentElement === dead);
  const onPad = copies.find((node) => node.parentElement === living);
  const oldVeil = copies.find((node) => node.parentElement === fogLayer);
  assert.ok(corpse && onPad && oldVeil);
  assert.equal(corpse.getAttribute("opacity"), "0");
  assert.equal(corpse.classList.contains("agent-glide"), false);
  assert.ok(
    [corpse, ...corpse.descendants()].every(
      (node) =>
        !node.hasAttribute("data-glide-key") &&
        !node.hasAttribute("data-presentation-key") &&
        !node.hasAttribute("tabindex") &&
        !node.hasAttribute("aria-label"),
    ),
    "a copy is inert",
  );
  assert.deepEqual(
    oldVeil.descendants().map((node) => node.getAttribute("d")),
    [null, "M0 0", "M1 1"],
    "the copy keeps the old veil shape",
  );
  assert.ok(
    oldVeil.descendants().every((node) => !node.hasAttribute("data-fog-state")),
  );
  assert.equal(
    fogLayer.children.indexOf(oldVeil),
    0,
    "the copy paints just under the new sight group",
  );
  const outgoing = factory.created.filter(({ spec }) =>
    [corpse, onPad, oldVeil].includes(spec.element),
  );
  assert.deepEqual(
    outgoing.map(({ spec }) => [
      spec.keyframes[0].opacity,
      spec.keyframes.at(-1).opacity,
    ]),
    [
      [1, 0],
      [1, 0],
      [1, 0],
    ],
  );
  await finishAndFlush(factory.created.find(({ id }) => id === "mbg:epoch-5:step"));
  assert.equal(
    battlefield.querySelectorAll("[data-rewind-copy]").length,
    0,
    "the copies are gone",
  );

  // Any change before the rewind ends removes them too.
  show(tick6, "replay_static");
  controller.captureStepSnapshot(domSurface);
  show(tick5, "replay_step");
  assert.ok(battlefield.querySelectorAll("[data-rewind-copy]").length > 0);
  show(stepPlan(9, [body("a", [5, 5])]), "replay_static");
  assert.equal(battlefield.querySelectorAll("[data-rewind-copy]").length, 0);
  // A resize during a rewind drops the copies (they keep the old projection).
  show(tick6, "replay_static");
  controller.captureStepSnapshot(domSurface);
  show(tick5, "replay_step");
  assert.ok(battlefield.querySelectorAll("[data-rewind-copy]").length > 0);
  controller.reproject(
    { plan: tick5 },
    { ...domSurface, viewportKey: "copies-resized" },
    {
      renderPolicy: "replay_static",
    },
  );
  assert.equal(battlefield.querySelectorAll("[data-rewind-copy]").length, 0);
  // A snapshot is used once: a later step without a new one puts no copy back.
  show(stepPlan(8, [body("a", [4, 5])]), "replay_step");
  assert.equal(battlefield.querySelectorAll("[data-rewind-copy]").length, 0);
  controller.dispose();
});

test("the browser clock paints nothing, so it never covers a fade on its element", () => {
  /** @type {Array<{keyframes: unknown, options: unknown}>} */
  const calls = [];
  const element = {
    /** @param {unknown} keyframes @param {unknown} options */
    animate(keyframes, options) {
      calls.push({ keyframes, options });
      return /** @type {any} */ ({ id: "" });
    },
  };
  const clock = new BrowserAnimationFactory().createClock(
    /** @type {any} */ (element),
    320,
    "mbg:e:step",
  );
  assert.equal(clock.id, "mbg:e:step");
  assert.deepEqual(calls, [
    { keyframes: [], options: { duration: 320, fill: "both" } },
  ]);
});

/** @param {number} frameIndex @param {number} finalFrameIndex
 * @param {number} cursorGeneration @param {number} choreographyGeneration */
function cursor(frameIndex, finalFrameIndex, cursorGeneration, choreographyGeneration) {
  return {
    schema_version: 1,
    frame_index: frameIndex,
    final_frame_index: finalFrameIndex,
    cursor_generation: cursorGeneration,
    choreography_generation: choreographyGeneration,
  };
}

/** @param {number} start @param {number} final */
function seekController(start, final) {
  /** @type {Array<Record<string, any>>} */
  const commands = [];
  let cursorGeneration = 10;
  let choreographyGeneration = 4;
  let index = start;
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      commands.push(command);
      cursorGeneration += 1;
      if (command.command_type === "next_frame") {
        index += 1;
        choreographyGeneration += 1;
      } else {
        index = command.frame_index;
      }
      return {
        frame: {
          cursor: cursor(index, final, cursorGeneration, choreographyGeneration),
        },
      };
    },
  });
  controller.setConnected(true);
  controller.installCursor(
    cursor(start, final, cursorGeneration, choreographyGeneration),
  );
  return { controller, commands };
}

test("the replay controller ends one-tick moves with the replay_step intent and snaps the rest", async () => {
  const { controller, commands } = seekController(5, 30);
  const stepOf = () => controller.snapshot().presentationIntent?.stepGlide ?? null;
  assert.equal(await controller.next(), true);
  assert.deepEqual(commands.at(-1), replaySeekCommand(6));
  assert.equal(stepOf(), "forward");
  assert.equal(controller.snapshot().presentationIntent?.renderPolicy, "replay_static");
  assert.equal(await controller.previous(), true);
  assert.equal(stepOf(), "backward");
  await controller.jump(10);
  assert.equal(stepOf(), null, "+10 snaps");
  await controller.jump(-10);
  assert.equal(stepOf(), null, "-10 snaps");
  await controller.seek(6, { glide: true });
  assert.equal(stepOf(), "forward");
  await controller.seek(9, { glide: true });
  assert.equal(stepOf(), null, "a slider move of three ticks snaps");
  await controller.seek(10);
  assert.equal(stepOf(), null, "a plain seek snaps");
  await controller.first();
  assert.equal(stepOf(), null);
  await controller.previous();
  assert.deepEqual(commands.at(-1), replaySeekCommand(0));
  assert.equal(stepOf(), null, "a same-frame move at the start snaps");
  await controller.last();
  assert.equal(stepOf(), null);
});

class FakeControl {
  constructor() {
    /** @type {Map<string, Set<(event: any) => void>>} */
    this.handlers = new Map();
    this.attributes = new Map();
    this.disabled = false;
    this.value = "";
    this.min = "";
    this.max = "";
    this.step = "";
    this.textContent = "";
    this.hidden = false;
  }

  /** @param {string} type @param {(event: any) => void} handler */
  addEventListener(type, handler) {
    const set = this.handlers.get(type) ?? new Set();
    set.add(handler);
    this.handlers.set(type, set);
  }

  /** @param {string} type @param {(event: any) => void} handler */
  removeEventListener(type, handler) {
    this.handlers.get(type)?.delete(handler);
  }

  /** @param {string} name @param {unknown} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  matches() {
    return false;
  }

  /** @param {string} type @param {Record<string, any>} [event] */
  dispatch(type, event = {}) {
    const full = {
      target: this,
      defaultPrevented: false,
      preventDefault() {
        full.defaultPrevented = true;
      },
      ...event,
    };
    for (const handler of this.handlers.get(type) ?? []) handler(full);
    return full;
  }
}

function controls() {
  return {
    root: new FakeControl(),
    keyboardTarget: new FakeControl(),
    keyboardAuthorityInstalled: () => true,
    keyboardEnabled: () => true,
    clearSelection: () => {},
    firstButton: new FakeControl(),
    backTenButton: new FakeControl(),
    previousButton: new FakeControl(),
    playPauseButton: new FakeControl(),
    nextButton: new FakeControl(),
    forwardTenButton: new FakeControl(),
    lastButton: new FakeControl(),
    slider: new FakeControl(),
    position: new FakeControl(),
    rateSelect: new FakeControl(),
    status: new FakeControl(),
    tickForFrameIndex: (/** @type {number} */ index) => index,
  };
}

async function flush() {
  for (let index = 0; index < 4; index += 1) await Promise.resolve();
}

test("buttons, arrows and the slider glide one tick and snap clicks and jumps", async () => {
  const { controller, commands } = seekController(15, 30);
  const elements = controls();
  const unbind = bindReplayTimelineControls(/** @type {any} */ (elements), controller);
  const stepOf = () => controller.snapshot().presentationIntent?.stepGlide ?? null;
  try {
    elements.nextButton.dispatch("click");
    await flush();
    assert.equal(stepOf(), "forward");
    elements.backTenButton.dispatch("click");
    await flush();
    assert.equal(stepOf(), null);
    elements.keyboardTarget.dispatch("keydown", {
      key: "ArrowLeft",
      target: elements.root,
    });
    await flush();
    assert.equal(stepOf(), "backward");
    elements.keyboardTarget.dispatch("keydown", {
      key: "ArrowRight",
      shiftKey: true,
      target: elements.root,
    });
    await flush();
    assert.equal(stepOf(), null, "Shift + arrow snaps");
    const at = Number(controller.snapshot().cursor?.frame_index);

    elements.slider.value = String(at + 1);
    elements.slider.dispatch("input");
    elements.slider.dispatch("change");
    await flush();
    assert.equal(stepOf(), "forward", "the slider's own arrow keys glide");

    const clicked = Number(controller.snapshot().cursor?.frame_index);
    elements.slider.dispatch("pointerdown");
    elements.slider.value = String(clicked + 1);
    elements.slider.dispatch("input");
    elements.slider.dispatch("pointerup");
    elements.slider.dispatch("change");
    await flush();
    assert.deepEqual(commands.at(-1), replaySeekCommand(clicked + 1));
    assert.equal(stepOf(), null, "a click on the track snaps");

    const dragged = Number(controller.snapshot().cursor?.frame_index);
    const sent = commands.length;
    elements.slider.dispatch("pointerdown");
    elements.slider.dispatch("pointermove");
    elements.slider.value = String(dragged + 1);
    elements.slider.dispatch("input");
    await flush();
    assert.equal(commands.length, sent + 1, "a drag scrubs one tick at a time");
    assert.equal(stepOf(), "forward");
    elements.slider.dispatch("pointerup");
    elements.slider.dispatch("change");
    await flush();
    assert.equal(commands.length, sent + 1, "the drag already reached the value");
  } finally {
    unbind();
  }
});

test("N does exactly what +1 and Arrow Right do: one gliding tick forward", async () => {
  /** @type {Array<{frame: number | undefined, step: string | null, command: Record<string, any> | undefined}>} */
  const results = [];
  for (const press of ["button", "arrow", "n", "N"]) {
    const { controller, commands } = seekController(7, 30);
    const elements = controls();
    const unbind = bindReplayTimelineControls(
      /** @type {any} */ (elements),
      controller,
    );
    try {
      if (press === "button") elements.nextButton.dispatch("click");
      else {
        const event = elements.keyboardTarget.dispatch("keydown", {
          key: press === "arrow" ? "ArrowRight" : press,
          shiftKey: press === "N",
          target: elements.root,
        });
        assert.equal(event.defaultPrevented, true, press);
      }
      await flush();
      results.push({
        frame: controller.snapshot().cursor?.frame_index,
        step: controller.snapshot().presentationIntent?.stepGlide ?? null,
        command: commands.at(-1),
      });
      assert.equal(commands.length, 1, `${press} sends one request`);
      assert.equal(controller.snapshot().playing, false);
    } finally {
      unbind();
    }
  }
  for (const result of results) {
    assert.deepEqual(result, {
      frame: 8,
      step: "forward",
      command: replaySeekCommand(8),
    });
  }
});

test("V, B and F reach their owners through the one command set and F pauses play first", async () => {
  const { controller } = seekController(1, 30);
  controller.waitForPresentation = () => new Promise(() => {});
  const elements = controls();
  /** @type {string[]} */
  const calls = [];
  const removers = [
    registerLocalCommandHandler("camera_everyone_in_view", () => {
      calls.push("everyone_in_view");
    }),
    registerLocalCommandHandler("camera_battle_cam", () => {
      calls.push("battle_cam");
    }),
    registerLocalCommandHandler("frame_explanation", () => {
      calls.push(`frame_explanation:${controller.snapshot().playing}`);
    }),
  ];
  const unbind = bindReplayTimelineControls(/** @type {any} */ (elements), controller);
  try {
    for (const key of ["v", "B"]) {
      const event = elements.keyboardTarget.dispatch("keydown", {
        key,
        shiftKey: key === "B",
        target: elements.root,
      });
      assert.equal(event.defaultPrevented, key === "v");
    }
    elements.keyboardTarget.dispatch("keydown", { key: "b", target: elements.root });
    assert.equal(controller.play(), true);
    assert.equal(controller.snapshot().playing, true);
    const explained = elements.keyboardTarget.dispatch("keydown", {
      key: "f",
      target: elements.root,
    });
    assert.equal(explained.defaultPrevented, true);
    assert.deepEqual(calls, [
      "everyone_in_view",
      "battle_cam",
      "frame_explanation:false",
    ]);
    assert.equal(controller.snapshot().pauseReason, "frame_explanation");
    elements.keyboardTarget.dispatch("keydown", {
      key: "v",
      ctrlKey: true,
      target: elements.root,
    });
    assert.equal(calls.length, 3, "Ctrl+V stays a browser shortcut");
  } finally {
    unbind();
    for (const remove of removers) remove();
  }
  const bound = bindReplayTimelineControls(/** @type {any} */ (controls()), controller);
  assert.throws(
    () => bindReplayTimelineControls(/** @type {any} */ (controls()), controller),
    TypeError,
    "each replay handler has exactly one owner",
  );
  bound();
});
