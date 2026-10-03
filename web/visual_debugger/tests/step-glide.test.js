/**
 * @file Check body glides and the one-tick step. Played ticks glide every moved body
 * from its START spot to its END spot in 0-260 ms (a charging Warrior reaches its
 * post-Charge spot at 110 ms first) as FLIP overrides on the body wrappers, with fill
 * "backwards", composite "add" and no gate on the 450 ms live release. The
 * replay_step intent glides each body 240 ms from where the previous frame showed it
 * to its new END spot for +1 and -1 (a Charge rush takes the first 100 ms forwards
 * and 140 ms backwards), a quick next step starts from where the running glide draws
 * the body (scrubbing), the tick's cues fade in from 240 ms to 320 ms and the still
 * then settles paused. Jumps, a changed audience or recipient, deaths, returns, a
 * re-render of the same frame, reduced motion and Agent POV bodies unseen in either
 * frame never glide. The replay controller ends one-tick moves (+1 and -1, the arrow
 * keys, the slider's own keys and a slider drag) with the replay_step intent, snaps
 * jumps, Shift + arrows and slider clicks, plays one tick on N, and sends V, B and F
 * through the one command set (F pauses play first).
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { CombatChoreographer } from "../src/choreography.js";
import { beatSchedule, buildChoreographyPlan } from "../src/choreography-plan.js";
import { registerLocalCommandHandler } from "../src/command-set.js";
import {
  bindReplayTimelineControls,
  REPLAY_TRANSPORT_STATES,
  ReplayPlaybackController,
  replayNavigationCommand,
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

test("a one-tick step glides 240 ms from the previous END to the new END and fades cues in by 320 ms", async () => {
  const { controller, factory, show, wrappers } = stepHarness();
  const before = stepPlan(5, [body("a", [1, 1]), body("b", [5, 5])]);
  const forward = stepPlan(6, [
    body("a", [2, 1]),
    body("b", [7, 5], { charged: true, postCharge: [6, 5], start: [5, 5] }),
  ]);
  show(before, "replay_static");
  show(forward, "replay_step");
  const glides = factory.matching(":step-glide:");
  assert.equal(glides.length, 2);
  assert.equal(glides[0].spec.element, wrappers.get("a"));
  assert.deepEqual(translations(glides[0]), [
    [0, "translate(-10px, 0px)"],
    [1, "translate(0px, 0px)"],
  ]);
  assert.deepEqual(translations(glides[1]), [
    [0, "translate(-20px, 0px)"],
    [100 / 240, "translate(-10px, 0px)"],
    [1, "translate(0px, 0px)"],
  ]);
  assert.ok(
    glides.every(
      ({ spec }) =>
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
  const clock = factory.created.find(({ id }) => id === "mbg:epoch-6:step");
  assert.equal(clock?.spec.options.duration, 320);
  assert.equal(controller.snapshot().renderPolicy, "replay_static");
  assert.equal(controller.snapshot().paused, false);
  clock?.finish();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(controller.snapshot().paused, true, "the step settles into the still");
  assert.equal(controller.snapshot().animationCount, 0);
  assert.ok([...glides, ...fades].every(({ state }) => state === "cancelled"));

  show(before, "replay_step");
  const backward = factory
    .matching(":step-glide:")
    .filter(({ id }) => id.startsWith("mbg:epoch-5:"));
  assert.deepEqual(
    backward.map((animation) => translations(animation)),
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
  const created = factory.created.length;
  show(before, "replay_step");
  assert.equal(
    factory.created.length,
    created,
    "a re-render of the shown frame never glides",
  );
  controller.dispose();
});

test("scrubbing starts the next step from where the running glide draws the body", () => {
  const { factory, show, controller } = stepHarness();
  show(stepPlan(1, [body("a", [0, 0])]), "replay_static");
  show(stepPlan(2, [body("a", [2, 0])]), "replay_step");
  const first = factory.matching(":step-glide:")[0];
  first.currentTime = 120;
  show(stepPlan(3, [body("a", [4, 0])]), "replay_step");
  assert.equal(first.state, "cancelled");
  const second = factory.matching(":step-glide:").at(-1);
  assert.ok(second);
  assert.match(second.id, /^mbg:epoch-3:/u);
  assert.deepEqual(translations(second), [
    [0, "translate(-30px, 0px)"],
    [1, "translate(0px, 0px)"],
  ]);
  controller.dispose();
});

test("jumps, audience changes, deaths, returns and unseen Agent POV bodies never glide", () => {
  const cases = [
    ["a ten-tick jump", stepPlan(15, [body("a", [9, 9])])],
    [
      "another audience or recipient",
      stepPlan(6, [body("a", [9, 9])], { audienceKey: "audience-pov-recipient-2" }),
    ],
    ["a death", stepPlan(6, [body("a", [9, 9], { alive: false, died: true })])],
    ["a return to a pad", stepPlan(6, [body("a", [9, 9], { returned: true })])],
    ["a body unseen in the earlier frame", stepPlan(6, [body("c", [9, 9])])],
    [
      "Step Glide off",
      stepPlan(6, [body("a", [9, 9])], {
        beats: beatSchedule({ visualFilters: { step_glide: false } }),
      }),
    ],
  ];
  for (const [label, next] of cases) {
    const { factory, show, controller } = stepHarness();
    show(stepPlan(5, [body("a", [1, 1]), body("b", [2, 2])]), "replay_static");
    show(/** @type {Record<string, any>} */ (next), "replay_step");
    assert.equal(factory.matching(":step-glide:").length, 0, String(label));
    controller.dispose();
  }
  const pov = stepHarness();
  pov.show(stepPlan(5, [body("a", [1, 1]), body("b", [2, 2])]), "replay_static");
  pov.show(stepPlan(6, [body("a", [2, 1])]), "replay_step");
  assert.deepEqual(
    pov.factory.matching(":step-glide:").map(({ id }) => id),
    ["mbg:epoch-6:step-glide:agent-a"],
    "a body that left sight is absent from the new frame and never glides",
  );
  pov.controller.dispose();

  const reduced = stepHarness({ motionMode: "reduced" });
  reduced.show(stepPlan(5, [body("a", [1, 1])]), "replay_static");
  reduced.show(stepPlan(6, [body("a", [2, 1])]), "replay_step");
  assert.equal(reduced.factory.created.length, 0, "reduced motion shows the still");
  reduced.controller.dispose();
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

test("N plays exactly one tick, then settles into the paused still", async () => {
  /** @type {(value?: unknown) => void} */
  let presented = () => {};
  const { controller, commands } = seekController(3, 4);
  controller.waitForPresentation = () =>
    new Promise((resolve) => {
      presented = resolve;
    });
  assert.equal(controller.playOneTick(), true);
  assert.deepEqual(commands, [replayNavigationCommand("next")]);
  assert.equal(
    controller.snapshot().presentationIntent?.renderPolicy,
    "replay_animated",
  );
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.PLAYING);
  assert.equal(controller.playOneTick(), false, "N waits while the tick plays");
  presented();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.SETTLED);
  assert.equal(controller.snapshot().pauseReason, "one_tick");
  assert.equal(commands.length, 1, "no second frame is requested");

  const middle = seekController(1, 9);
  middle.controller.waitForPresentation = () => Promise.resolve();
  assert.equal(middle.controller.playOneTick(), true);
  for (let flush = 0; flush < 6; flush += 1) await Promise.resolve();
  assert.equal(middle.controller.snapshot().pauseReason, "one_tick");
  assert.equal(
    middle.controller.snapshot().presentationIntent?.renderPolicy,
    "replay_static",
  );
  assert.equal(middle.commands.length, 1);
  assert.equal(middle.controller.snapshot().cursor?.frame_index, 2);

  const end = seekController(4, 4);
  assert.equal(end.controller.playOneTick(), false);
  assert.equal(end.controller.snapshot().pauseReason, "endpoint");
  assert.deepEqual(end.commands, []);
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
