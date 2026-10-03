/**
 * @file Check the Fog And Vision look (`src/looks/fog.js`): the Agent POV
 * Sight Reach veil, the recipient rim, the Spotted Glint and the Last Seen
 * ring. Contracts: the veil's pieces cover the floor exactly once and light
 * exactly the points inside a living sensor's reach circle (also for crossing
 * circles and circles cut by the floor edge), and the drawn edge uses SVG arcs
 * with no row steps; update() skips null and undefined nodes (the painter's
 * missing underlay) and still moves the other marks; Oracle
 * View, an empty input and switched-off parts draw nothing; on every Agent POV
 * fixture frame every living body other than the recipient sits in the lit
 * area of the END state; NoSharedObs uses only the recipient's circle and
 * SharedObs only living authorized sensors (dead sensors give no light); the
 * light states follow the Stride, Fall and Return beats with END last; the
 * rim steps aside when the recipient is shielded, controlled, dead or too
 * small; glints mark only opponents that came into sight; Last Seen rings come
 * from derived `last_seen` rows, sit at 0.93R and name no unseen champion;
 * build() is export-safe (no url(), gradient, mask, filter, clip path, image,
 * pattern or id; fill-rule as an attribute); and sightAnimationSpecs gives
 * fill "backwards" overrides only for played ticks and +1 steps.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { authorizedPresentationSceneView } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { createViewportTransform } from "../src/layout.js";
import * as fog from "../src/looks/fog.js";

const TRANSFORM = createViewportTransform({
  worldWidth: 20,
  worldHeight: 12,
  viewportWidth: 1440,
  viewportHeight: 900,
  padding: 24,
});

/**
 * @param {Record<string, any>} overrides
 * @returns {Record<string, any>}
 */
function agent(overrides) {
  return {
    presentation_key: `pov_${overrides.public_agent_id}`,
    life_state: "alive",
    observation_radius: 6,
    radius: 0.5,
    spawn_shield_remaining: 0,
    relation: "ally",
    ...overrides,
  };
}

/**
 * @param {Array<Record<string, any>>} agents
 * @param {Record<string, any>} [selection]
 */
function scene(agents, selection = {}) {
  return { audience: "agent_pov", map: { width: 20, height: 12 }, agents, selection };
}

const SELF = agent({ public_agent_id: "a0", relation: "self", position: [4, 4] });
const ALLY = agent({ public_agent_id: "a1", position: [14, 6] });
const ENEMY = agent({ public_agent_id: "b0", relation: "opponent", position: [7, 5] });

// A tiny SVG-like DOM: attributes, children, parents and the simple selectors
// fog.js uses ([attr], [attr="v"], .class, tag, and descendant chains).
function fakeDocument() {
  /** @param {string} tag @returns {any} */
  function node(tag) {
    /** @type {Map<string, string>} */
    const attributes = new Map();
    /** @type {any} */
    const element = {
      tagName: tag,
      attributes,
      children: /** @type {any[]} */ ([]),
      parentElement: null,
      dataset: /** @type {Record<string, string>} */ ({}),
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      hasAttribute: (/** @type {string} */ name) => attributes.has(name),
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
      addEventListener: () => {},
      removeEventListener: () => {},
      append: (/** @type {any[]} */ ...nodes) => {
        for (const child of nodes) {
          child.parentElement = element;
          element.children.push(child);
        }
      },
      querySelectorAll: (/** @type {string} */ selector) =>
        descendants(element).filter((candidate) => matches(candidate, selector)),
      querySelector: (/** @type {string} */ selector) =>
        element.querySelectorAll(selector)[0] ?? null,
    };
    return element;
  }
  return {
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      node(tag),
    createElement: (/** @type {string} */ tag) => node(tag),
    node,
  };
}

/** @param {any} element @returns {any[]} */
function descendants(element) {
  return element.children.flatMap((/** @type {any} */ child) => [
    child,
    ...descendants(child),
  ]);
}

/** @param {any} element @param {string} token */
function matchesSimple(element, token) {
  const pattern = /^\[([\w-]+)(?:="([^"]*)")?\]$/u.exec(token);
  if (pattern) {
    const value = element.getAttribute(pattern[1]);
    return pattern[2] === undefined ? value !== null : value === pattern[2];
  }
  if (token.startsWith(".")) {
    return (element.getAttribute("class") ?? "").split(" ").includes(token.slice(1));
  }
  return element.tagName === token;
}

/** @param {any} element @param {string} selector */
function matches(element, selector) {
  const parts = selector.trim().split(/\s+/u);
  if (!matchesSimple(element, /** @type {string} */ (parts.pop()))) return false;
  let ancestor = element.parentElement;
  while (parts.length > 0 && ancestor) {
    if (matchesSimple(ancestor, parts[parts.length - 1])) parts.pop();
    ancestor = ancestor.parentElement;
  }
  return parts.length === 0;
}

/** @param {any} element @returns {string} */
function serialize(element) {
  const attributes = [...element.attributes]
    .map(([name, value]) => ` ${name}="${value}"`)
    .join("");
  return `<${element.tagName}${attributes}>${element.children.map(serialize).join("")}</${element.tagName}>`;
}

/** @param {Record<string, any>} record */
function endState(record) {
  return record.states[record.states.length - 1];
}

test("reach pieces cover the floor once and light exactly the reach circles", () => {
  /** @param {string} id @param {number} x @param {number} y @param {number} r */
  const circle = (id, x, y, r) => ({
    presentationKey: id,
    publicAgentId: id,
    x,
    y,
    r,
    own: id === "a",
  });
  const cases = [
    [circle("a", 5, 5, 2)],
    [circle("a", 4, 5, 2), circle("b", 6.5, 5.5, 1.5)],
    [circle("a", 1, 9, 2.5), circle("b", 9.5, 1, 1), circle("c", 9.5, 1, 1)],
  ];
  for (const circles of cases) {
    const runs = fog.sightReachRuns(circles, 10, 10);
    /** @param {any} piece @param {number} x @param {number} y */
    const inside = (piece, x, y) =>
      fog.isLitByRuns({ lit: [piece] }, x, y) && !(y === piece.y0 || y === piece.y1);
    for (let x = 0.05; x < 10; x += 0.1) {
      for (let y = 0.05; y < 10; y += 0.1) {
        const reach = Math.min(
          ...circles.map((c) => Math.hypot(x - c.x, y - c.y) - c.r),
        );
        if (Math.abs(reach) < 1e-6) continue;
        const hits = [...runs.dark, ...runs.lit].filter((piece) => inside(piece, x, y));
        assert.equal(hits.length, 1, `${x},${y} lies in exactly one piece`);
        assert.equal(fog.isLitByRuns(runs, x, y), reach < 0, `${x},${y}`);
      }
    }
  }
  const closed = fog.sightReachRuns([], 10, 10);
  assert.deepEqual(closed.dark, [{ y0: 0, y1: 10, left: { x: 0 }, right: { x: 10 } }]);
  assert.deepEqual(closed.lit, []);
  assert.deepEqual(fog.sightReachRuns(cases[0], 0, 10), { dark: [], lit: [] });
});

test("the Sight Reach veil edge is drawn with arcs, not row steps", () => {
  const [veil] = fog.records({ scene: scene([SELF]), transform: TRANSFORM });
  assert.ok(veil);
  const end = veil.states[veil.states.length - 1];
  assert.match(end.veilPath, /A/);
  assert.match(end.liftPath, /A/);
  assert.ok((end.veilPath.match(/M/g) ?? []).length <= 4, end.veilPath);
});

test("update skips the painter's missing underlay and still moves the marks", () => {
  /** @type {Record<string, string>} */
  const attributes = { "data-fog-kind": "recipient-rim" };
  const rim = {
    querySelectorAll: () => [],
    getAttribute: (/** @type {string} */ name) => attributes[name] ?? null,
    setAttribute: (/** @type {string} */ name, /** @type {string} */ value) => {
      attributes[name] = value;
    },
  };
  const record = { kind: "recipient-rim", center: { x: 3, y: 4 }, radius: 5 };
  assert.doesNotThrow(() =>
    fog.update(
      /** @type {any} */ ([rim, null, undefined]),
      /** @type {any} */ ([record]),
    ),
  );
  assert.equal(attributes.cx, "3");
  assert.equal(attributes.cy, "4");
  assert.equal(attributes.r, "5");
});

test("Oracle View, empty input and switched-off parts draw nothing", () => {
  assert.deepEqual(fog.records({}), []);
  assert.deepEqual(
    fog.records({
      scene: { ...scene([SELF]), audience: "researcher" },
      transform: TRANSFORM,
    }),
    [],
  );
  assert.deepEqual(
    fog.records({
      scene: scene([SELF]),
      transform: TRANSFORM,
      audience: "researcher",
      part: "over",
    }),
    [],
  );
  const off = {
    pov_sight_veil: false,
    pov_recipient_rim: false,
    sight_change_marks: false,
  };
  assert.deepEqual(
    fog.records({ scene: scene([SELF]), transform: TRANSFORM, filters: off }),
    [],
  );
  assert.deepEqual(
    fog.records({
      scene: scene([SELF]),
      transform: TRANSFORM,
      filters: off,
      part: "over",
    }),
    [],
  );
  const row = {
    kind: "last_seen",
    eventId: "e",
    presentationKey: "pov_b0",
    anchor: { x: 1, y: 2 },
  };
  assert.deepEqual(fog.records({ event: row, filters: off }), []);
  assert.deepEqual(fog.build([], /** @type {any} */ (null)), []);
});

test("NoSharedObs lights only the recipient; SharedObs only living authorized sensors", () => {
  const deadAlly = agent({
    public_agent_id: "a2",
    position: [16, 2],
    life_state: "dead",
  });
  const agents = [SELF, ALLY, deadAlly, ENEMY];
  const [noShared] = fog.records({ scene: scene(agents), transform: TRANSFORM });
  assert.equal(noShared.kind, "sight-veil");
  assert.equal(noShared.states.length, 1);
  assert.deepEqual(
    endState(noShared).circles.map((/** @type {any} */ c) => c.publicAgentId),
    ["a0"],
  );
  const sensorSources = ["a0", "a1", "a2"].map((id) => ({
    source_public_agent_id: id,
    source_presentation_key: `pov_${id}`,
  }));
  const [shared] = fog.records({
    scene: scene(agents),
    transform: TRANSFORM,
    sensorSources,
  });
  assert.deepEqual(
    endState(shared).circles.map((/** @type {any} */ c) => [c.publicAgentId, c.own]),
    [
      ["a0", true],
      ["a1", false],
    ],
  );
  const recipientDead = { ...SELF, life_state: "dead" };
  const [closed] = fog.records({
    scene: scene([recipientDead, ENEMY]),
    transform: TRANSFORM,
  });
  assert.deepEqual(endState(closed).circles, []);
  assert.equal(endState(closed).liftPath, "");
  assert.match(endState(closed).veilPath, /^M[\d.]+ [\d.]+H[\d.]+V[\d.]+H[\d.]+Z$/u);
});

test("every living body on the Agent POV fixture frames sits in the lit END area", async () => {
  const fixture = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  const raws = [
    ...Object.values(fixture.presentations),
    ...Object.values(fixture.pairs).map(
      (pair) => /** @type {any} */ (pair).presentation,
    ),
    ...Object.values(fixture.state_cases),
  ];
  let checkedBodies = 0;
  let sharedFrames = 0;
  for (const raw of raws) {
    let frame;
    try {
      frame = await normalizeAuthorizedPresentationFrameV1(structuredClone(raw));
    } catch {
      continue;
    }
    const view = authorizedPresentationSceneView(frame);
    if (view?.audience !== "agent_pov") continue;
    const parts = /** @type {any} */ (frame).current_endpoint.parts;
    const sensorSources = parts.authorized_sensor_sources ?? null;
    if (sensorSources) sharedFrames += 1;
    const transform = createViewportTransform({
      worldWidth: view.map.width,
      worldHeight: view.map.height,
      viewportWidth: 1440,
      viewportHeight: 900,
      padding: 24,
    });
    const [veil] = fog.records({
      scene: view,
      transform,
      sensorSources,
      trajectories:
        /** @type {any} */ (frame).visual_events?.agent_phase_trajectories ?? null,
    });
    const end = endState(veil);
    const runs = fog.sightReachRuns(end.circles, view.map.width, view.map.height);
    const allowed = new Set(
      sensorSources
        ? sensorSources.map((/** @type {any} */ row) => row.source_public_agent_id)
        : [parts.recipient_public_agent_id],
    );
    for (const circle of end.circles) assert.ok(allowed.has(circle.publicAgentId));
    for (const body of view.agents) {
      if (body.relation === "self" || body.life_state !== "alive") continue;
      checkedBodies += 1;
      assert.ok(
        fog.isLitByRuns(runs, body.position[0], body.position[1]),
        body.public_agent_id,
      );
    }
  }
  assert.ok(checkedBodies > 0, "the fixtures hold living bodies");
  assert.ok(sharedFrames > 0, "the fixtures hold SharedObs frames");
});

test("light states follow the Stride, Fall and Return beats and end at END", () => {
  const dyingAlly = agent({
    public_agent_id: "a1",
    position: [14, 6],
    life_state: "dead",
  });
  const returner = agent({ public_agent_id: "a2", position: [1, 1] });
  const trajectories = [
    {
      agent_presentation_key: "pov_a0",
      transition_start: { position: [3, 4] },
      successor: { position: [4, 4] },
    },
    {
      agent_presentation_key: "pov_a1",
      transition_start: { position: [14, 6] },
      successor: { position: [14, 6] },
    },
  ];
  const schedule = {
    falls: [{ presentationKey: "pov_a1", at: 900 }],
    returns: [{ presentationKey: "pov_a2", at: 2500 }],
  };
  const sensorSources = ["a0", "a1", "a2"].map((id) => ({
    source_public_agent_id: id,
  }));
  const [veil] = fog.records({
    scene: scene([SELF, dyingAlly, returner]),
    transform: TRANSFORM,
    sensorSources,
    trajectories,
    schedule,
  });
  const summary = veil.states.map((/** @type {any} */ state) => [
    state.arrive?.kind ?? null,
    state.leave?.kind ?? null,
    state.endOpacity,
    state.circles.map((/** @type {any} */ c) => `${c.publicAgentId}@${c.x}`).join(","),
  ]);
  assert.deepEqual(summary, [
    [null, "stride", 0, "a0@3,a1@14"],
    ["stride", "fall", 0, "a0@4,a1@14"],
    ["fall", "return", 0, "a0@4"],
    ["return", null, 1, "a0@4,a2@1"],
  ]);
  const [still] = fog.records({ scene: scene([SELF]), transform: TRANSFORM });
  assert.equal(still.states.length, 1);
  assert.equal(still.states[0].arrive, null);
});

test("the rim marks the recipient at 0.98R and steps aside when it should", () => {
  const radius = TRANSFORM.worldLengthToScreen(0.5);
  const over = (/** @type {any[]} */ agents, selection = {}) =>
    fog.records({
      scene: scene(agents, selection),
      transform: TRANSFORM,
      part: "over",
    });
  const [rim] = over([SELF, ENEMY]);
  assert.equal(rim.kind, "recipient-rim");
  assert.equal(rim.presentationKey, "pov_a0");
  assert.ok(Math.abs(rim.radius - radius * 0.98) < 0.01);
  assert.deepEqual(over([{ ...SELF, spawn_shield_remaining: 3 }]), []);
  assert.deepEqual(over([{ ...SELF, life_state: "dead" }]), []);
  assert.deepEqual(over([SELF], { controlled_presentation_key: "pov_a0" }), []);
  assert.deepEqual(over([{ ...SELF, radius: 0.05 }]), []);
});

test("glints mark only living opponents that came into sight", () => {
  const ally = agent({ public_agent_id: "a1", position: [5, 5] });
  const trajectories = [
    {
      agent_presentation_key: "pov_b0",
      transition_start: null,
      successor: { position: [7, 5] },
    },
    {
      agent_presentation_key: "pov_a1",
      transition_start: null,
      successor: { position: [5, 5] },
    },
  ];
  const records = fog.records({
    scene: scene([SELF, ally, ENEMY]),
    transform: TRANSFORM,
    part: "over",
    trajectories,
    filters: { pov_recipient_rim: false },
  });
  assert.deepEqual(
    records.map((record) => [record.kind, record.publicAgentId]),
    [["spotted-glint", "b0"]],
  );
});

test("Last Seen rings come from last_seen rows and name no unseen champion", () => {
  const row = {
    kind: "last_seen",
    eventId: "t1:last-seen:b3",
    presentationKey: "pov_b3",
    publicAgentId: "b3",
    classId: 3,
    teamId: null,
    anchor: { x: 210.5, y: 140.25 },
  };
  const [mark] = fog.records({ event: row, transform: TRANSFORM });
  assert.equal(mark.kind, "last-seen");
  assert.deepEqual(mark.center, { x: 210.5, y: 140.25 });
  const bodyRadius = TRANSFORM.worldLengthToScreen(0.5);
  assert.ok(Math.abs(mark.radius - bodyRadius * 0.93) < 0.01);
  assert.equal(mark.teamId, null);
  assert.equal(mark.glyphKey, "class-hunter");
  const [plain] = fog.records({ event: row });
  assert.ok(Math.abs(plain.radius - 23 * 0.93) < 0.01);
  assert.deepEqual(fog.records({ event: { ...row, anchor: null } }), []);
  assert.deepEqual(fog.records({ event: { ...row, kind: "herald_card" } }), []);
  assert.deepEqual(
    fog.records({
      event: row,
      scene: scene([agent({ public_agent_id: "b3", position: [1, 1] })]),
    }),
    [],
  );
  const card = /** @type {any} */ (fog.tooltip(mark));
  assert.equal(card.title, "Last Seen");
  assert.match(card.summary, /^A Hunter was here at the start of this tick/u);
  assert.doesNotMatch(card.summary, /b3|ID|Team/u);
});

test("build() is export-safe and records each veil state's beats", () => {
  const doc = fakeDocument();
  const trajectories = [
    {
      agent_presentation_key: "pov_a0",
      transition_start: { position: [3, 4] },
      successor: { position: [4, 4] },
    },
    {
      agent_presentation_key: "pov_b0",
      transition_start: null,
      successor: { position: [7, 5] },
    },
  ];
  const input = { scene: scene([SELF, ENEMY]), transform: TRANSFORM, trajectories };
  const under = fog.build(fog.records(input), /** @type {any} */ (doc));
  const over = fog.build(
    fog.records({ ...input, part: "over" }),
    /** @type {any} */ (doc),
  );
  const mount = {
    group: doc.node("g"),
    connector: doc.node("g"),
    underlay: null,
    plan: { epochKey: "e1", beats: { windows: { glide: [0, 260] } } },
    options: { motionMode: "normal", settled: false },
    animationSpecs: /** @type {any[]} */ ([]),
  };
  const row = {
    kind: "last_seen",
    eventId: "t:ls",
    presentationKey: "pov_b9",
    classId: 1,
    anchor: { x: 50, y: 60 },
  };
  const mounted = fog.build(
    fog.records({ event: row, transform: TRANSFORM }),
    /** @type {any} */ (doc),
    /** @type {any} */ (mount),
  );
  assert.deepEqual(mounted, [mount.group, mount.connector]);
  assert.equal(mount.animationSpecs.length, 1);
  assert.equal(mount.animationSpecs[0].options.fill, "backwards");
  assert.equal(mount.animationSpecs[0].options.delay, 130);
  const svg = [...under, ...over, mount.group].map(serialize).join("");
  assert.doesNotMatch(
    svg,
    /url\(|gradient|<mask|<filter|<clipPath|<image|<pattern| id="/iu,
  );
  assert.match(svg, /fill-rule="nonzero"/u);
  assert.match(svg, /class="fog-veil" opacity="0.45"/u);
  assert.match(svg, /class="fog-glint"[^>]*opacity="0"/u);
  const states = [...under[0].querySelectorAll(".fog-veil [data-fog-state]")];
  assert.deepEqual(
    states.map((/** @type {any} */ s) => [
      s.getAttribute("data-fog-arrive-kind"),
      s.getAttribute("data-fog-leave-kind"),
      s.getAttribute("opacity"),
    ]),
    [
      [null, "stride", "0"],
      ["stride", null, "1"],
    ],
  );
  fog.update(
    under,
    fog.records({
      ...input,
      transform: createViewportTransform({
        worldWidth: 20,
        worldHeight: 12,
        viewportWidth: 960,
        viewportHeight: 600,
        padding: 24,
      }),
    }),
  );
  assert.notEqual(
    states[1].getAttribute("d"),
    fog.records(input)[0].states[1].veilPath,
  );
});

test("sightAnimationSpecs times played ticks and +1 steps only", () => {
  const doc = fakeDocument();
  const trajectories = [
    {
      agent_presentation_key: "pov_a0",
      transition_start: { position: [3, 4] },
      successor: { position: [4, 4] },
    },
    {
      agent_presentation_key: "pov_b0",
      transition_start: null,
      successor: { position: [7, 5] },
    },
  ];
  const input = { scene: scene([SELF, ENEMY]), transform: TRANSFORM, trajectories };
  const root = doc.node("g");
  root.append(...fog.build(fog.records(input), /** @type {any} */ (doc)));
  root.append(
    ...fog.build(fog.records({ ...input, part: "over" }), /** @type {any} */ (doc)),
  );
  const plan = {
    epochKey: "e7",
    beats: { windows: { glide: [0, 260], settle: [600, 1100] }, reducedScale: 0.25 },
  };
  const played = fog.sightAnimationSpecs(root, plan, {
    renderPolicy: "replay_animated",
  });
  // Three groups (veil, lift, edge) x two states, plus one glint.
  assert.equal(played.length, 7);
  for (const [index, spec] of played.entries()) {
    assert.equal(spec.id, `mbg:e7:fog:${index}`);
    assert.equal(spec.options.fill, "backwards");
  }
  const startFade = played.find(
    (spec) => spec.element.getAttribute("data-fog-state") === "0",
  );
  assert.deepEqual(
    /** @type {any} */ (startFade).keyframes.map((/** @type {any} */ k) => k.opacity),
    [1, 1, 0],
  );
  assert.equal(/** @type {any} */ (startFade).options.duration, 260);
  const glint = /** @type {any} */ (played[played.length - 1]);
  assert.equal(glint.options.delay, 600);
  assert.equal(glint.keyframes[1].opacity, 0.5);
  assert.deepEqual(
    fog.sightAnimationSpecs(root, plan, { renderPolicy: "replay_static" }),
    [],
  );
  assert.deepEqual(
    fog.sightAnimationSpecs(root, plan, {
      motionMode: "off",
      renderPolicy: "live_once",
    }),
    [],
  );
  const step = fog.sightAnimationSpecs(root, plan, {
    renderPolicy: "replay_step",
    step: "forward",
  });
  assert.equal(step.length, 6);
  assert.ok(step.every((spec) => spec.options.duration === 200));
  assert.deepEqual(
    fog.sightAnimationSpecs(root, plan, {
      renderPolicy: "replay_step",
      step: "backward",
    }),
    [],
  );
});
