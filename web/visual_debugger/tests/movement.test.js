/**
 * @file Checks the Movement look (src/looks/movement.js, Scholar's Step).
 *
 * Contracts checked:
 * - records(): a Start Spot (ring 0.28, hairline 0.30, Charge pin 0.5) only for
 *   a move of 0.25 world units or more, at the START spot, in CSS pixels; a
 *   Charge bends its hairline through a pin; a leg through stone is a dotted
 *   Dim Shift leg; returns draw nothing; a Rogue dart of 0.95 units or more
 *   gives a Phantom Afterimage (needs Body Glide); the Focus Trail is eight
 *   dots on this tick's own path for the selected champion only, oldest first,
 *   and a champion that fell this tick has its trail held for its Fall beat.
 * - Switches: last_move_ghosts, phantom_afterimage, focus_trail and
 *   stride_glide each remove only their own marks.
 * - Agent POV: a body not in the POV scene, or with no disclosed start spot,
 *   gives no record; a post-Charge spot is ignored; movementSource uses only
 *   the inventory it is given (Oracle latest_events, Agent POV visual_events),
 *   refuses a non-fog-filtered inventory in Agent POV, and lists returns only
 *   from agent_respawned events. On real fixture frames every Agent POV record
 *   belongs to a body in the POV scene.
 * - cutsStone(): the display-only swept-circle stone test.
 * - build()/update(): export-safe SVG (no url(), gradients, masks, filters,
 *   images or id attributes), END opacities in attributes, the afterimage at
 *   opacity 0, reprojection in place.
 * - animationSpecs() and dimShiftSpecs(): beat-clock timing, fill "backwards"
 *   only, and no specs when motion is off. In a played tick each Focus Trail
 *   dot (it knows its leg and share of the path) appears over 60 ms as the
 *   gliding body passes it, oldest first, never ahead. A -1 rewind ("step") fades every
 *   mark in once the bodies have landed (from stepMs over stepCueFadeMs), with
 *   no afterimage.
 * - tooltip(): the "Start Spot" descriptor, null for other kinds.
 *
 * FakeElement is a tiny SVG DOM, just enough for build, update,
 * animationSpecs and tooltip.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { authorizedPresentationSceneView } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { createViewportTransform } from "../src/layout.js";
import {
  animationSpecs,
  BUDGET,
  build,
  cutsStone,
  dimShiftSpecs,
  MOVEMENT,
  movementSource,
  PAINT_PARTS,
  records,
  tooltip,
  update,
} from "../src/looks/movement.js";

const transform = createViewportTransform({
  worldWidth: 20,
  worldHeight: 10,
  viewportWidth: 848,
  viewportHeight: 448,
  padding: 24,
});
const UNIT = transform.worldLengthToScreen(1);

/**
 * @param {string} key
 * @param {Record<string, any>} [extra]
 */
function agent(key, extra = {}) {
  return {
    presentation_key: key,
    public_agent_id: `agent-slot-${key}`,
    class_id: 1,
    team_id: 1,
    position: [5, 5],
    radius: 0.5,
    life_state: "alive",
    ...extra,
  };
}

/**
 * @param {string} key
 * @param {number[] | null} start
 * @param {number[] | null} end
 * @param {number[] | null} [post]
 */
function trajectory(key, start, end, post = null) {
  return {
    agent_presentation_key: key,
    agent_public_agent_id: `agent-slot-${key}`,
    transition_start:
      start === null ? null : { phase: "transition_start", position: start },
    ...(post === null ? {} : { post_charge: { phase: "post_charge", position: post } }),
    successor: end === null ? null : { phase: "successor", position: end },
  };
}

/**
 * @param {Record<string, any>} [overrides]
 * @returns {Record<string, any>}
 */
function input(overrides = {}) {
  return {
    scene: {
      audience: "researcher",
      map: { width: 20, height: 10, obstacles: [] },
      agents: [
        agent("walker", { position: [5, 5] }),
        agent("still", { position: [8, 5], team_id: 2 }),
        agent("charger", { position: [12, 5], class_id: 2 }),
        agent("rogue", { position: [15, 7], class_id: 4, team_id: 2 }),
      ],
      selection: { selected_presentation_key: "walker" },
    },
    transform,
    audience: "researcher",
    trajectories: [
      trajectory("walker", [4, 5], [5, 5]),
      trajectory("still", [7.9, 5], [8, 5]),
      trajectory("charger", [9, 5], [12, 5], [11.5, 5]),
      trajectory("rogue", [13.5, 7], [15, 7]),
    ],
    returned: [],
    ...overrides,
  };
}

/** @param {ReadonlyArray<Record<string, any>>} list @param {string} kind */
function ofKind(list, kind) {
  return list.filter((record) => record.kind === kind);
}

class FakeElement {
  /** @param {string} localName */
  constructor(localName) {
    this.localName = localName;
    this.tagName = localName;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    /** @type {FakeElement | null} */
    this.parentElement = null;
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
    return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 };
  }
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    for (const node of nodes) {
      node.parentElement = this;
      this.children.push(node);
    }
  }
  /** @returns {FakeElement | null} */
  get lastElementChild() {
    return this.children.at(-1) ?? null;
  }
  /** @returns {FakeElement[]} */
  descendants() {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }
  /** @param {string} selectors */
  querySelectorAll(selectors) {
    const tests = selectors.split(",").map((part) => matcher(part.trim()));
    return this.descendants().filter((node) => tests.some((check) => check(node)));
  }
  /** @param {string} selectors */
  querySelector(selectors) {
    return this.querySelectorAll(selectors)[0] ?? null;
  }
}

/** @param {string} selector */
function matcher(selector) {
  const attribute = selector.match(/^\[([a-z-]+)\]$/);
  if (attribute)
    return (/** @type {FakeElement} */ node) => node.hasAttribute(attribute[1]);
  const [tag, className] = selector.startsWith(".")
    ? ["", selector.slice(1)]
    : selector.split(".");
  return (/** @type {FakeElement} */ node) =>
    (tag === "" || node.localName === tag) &&
    (className === undefined ||
      (node.getAttribute("class") ?? "").split(" ").includes(className));
}

const doc = /** @type {Document} */ (
  /** @type {unknown} */ ({
    createElementNS: (/** @type {string} */ namespace, /** @type {string} */ name) => {
      assert.equal(namespace, "http://www.w3.org/2000/svg");
      return new FakeElement(name);
    },
  })
);

/** @param {FakeElement} node @returns {string} */
function markup(node) {
  const attributes = [...node.attributes].map(([name, value]) => `${name}="${value}"`);
  return `<${node.localName} ${attributes.join(" ")}>${node.children.map(markup).join("")}</${node.localName}>`;
}

/** @param {string} name */
async function fixturePresentation(name) {
  const fixture = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  return structuredClone(fixture.presentations[name] ?? fixture.state_cases[name]);
}

test("a Start Spot marks where a moved champion began, at 28% ring and 30% hairline", () => {
  const result = records(input());
  const spots = ofKind(result, "start_spot");
  assert.deepEqual(
    spots.map((spot) => spot.presentationKey),
    ["walker", "charger", "rogue"],
  );
  const walker = spots[0];
  assert.deepEqual(walker.center, transform.worldToScreen([4, 5]));
  assert.equal(walker.radius, 0.5 * UNIT * 0.93);
  assert.deepEqual(walker.opacity, { ring: 0.28, hairline: 0.3, pin: 0.5 });
  assert.equal(walker.legs.length, 1);
  assert.deepEqual(walker.legs[0].from, transform.worldToScreen([4, 5]));
  assert.deepEqual(walker.legs[0].to, transform.worldToScreen([5, 5]));
  assert.equal(walker.legs[0].cut, false);
  assert.equal(walker.pin, null);
  assert.match(walker.label, /^Mage · /);
  assert.equal(walker.team, "a");
  assert.equal(spots[2].team, "b");
  assert.ok(Object.isFrozen(result));
  for (const record of result) assert.ok(Object.isFrozen(record));
  // A move under 0.25 world units leaves no Start Spot.
  assert.equal(
    spots.some((spot) => spot.presentationKey === "still"),
    false,
  );
});

test("a Charge bends its hairline through a pin where the rush stopped", () => {
  const charger = ofKind(records(input()), "start_spot").find(
    (spot) => spot.presentationKey === "charger",
  );
  assert.ok(charger);
  assert.equal(charger.charged, true);
  assert.deepEqual(charger.pin, transform.worldToScreen([11.5, 5]));
  assert.deepEqual(
    charger.legs.map((/** @type {Record<string, any>} */ leg) => [
      leg.leg,
      leg.from,
      leg.to,
    ]),
    [
      ["rush", transform.worldToScreen([9, 5]), transform.worldToScreen([11.5, 5])],
      ["stride", transform.worldToScreen([11.5, 5]), transform.worldToScreen([12, 5])],
    ],
  );
  assert.equal(charger.path, 3);
});

test("a path through stone is a dotted Dim Shift leg and cutsStone is the one test", () => {
  const pillar = { kind: "pillar", obstacle_id: 1, center: [4.5, 5], radius: 0.3 };
  const wall = {
    kind: "wall",
    obstacle_id: 2,
    center: [10, 2],
    width: 0.4,
    height: 3,
    theta: 0,
  };
  assert.equal(cutsStone([4, 5], [5, 5], [pillar]), true);
  assert.equal(cutsStone([4, 6.5], [5, 6.5], [pillar]), false);
  assert.equal(cutsStone([9, 2], [11, 2], [wall]), true);
  assert.equal(cutsStone([9, 4.5], [11, 4.5], [wall]), false);
  // A 90-degree turn makes the same wall lie along x.
  assert.equal(cutsStone([10, 0], [10, 4], [{ ...wall, theta: Math.PI / 2 }]), true);
  assert.equal(cutsStone([4, 5], [4, 5], [pillar]), false);
  assert.equal(cutsStone([4, 5], [5, 5], []), false);
  assert.equal(cutsStone(null, [5, 5], [pillar]), false);
  const base = input();
  const walker = ofKind(
    records({
      ...base,
      scene: { ...base.scene, map: { ...base.scene.map, obstacles: [pillar] } },
    }),
    "start_spot",
  )[0];
  assert.equal(walker.cut, true);
  assert.equal(walker.legs[0].cut, true);
  // A dotted stone leg must keep real dash lengths: no pathLength on it.
  const [node] = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (build([walker], doc))
  );
  const line = node.querySelector(".movement-hairline");
  assert.equal(line?.getAttribute("data-cut"), "true");
  assert.equal(line?.hasAttribute("pathLength"), false);
});

test("a return to a pad and a body missing from the trajectories draw nothing", () => {
  const result = records(input({ returned: ["walker", "charger", "rogue"] }));
  assert.deepEqual(result, []);
  assert.deepEqual(records(input({ trajectories: [] })), []);
  assert.deepEqual(records({}), []);
  assert.deepEqual(records({ scene: input().scene }), []);
});

test("a darting Rogue leaves a Phantom Afterimage; Body Glide off removes it", () => {
  const afterimages = ofKind(records(input()), "phantom_afterimage");
  assert.deepEqual(
    afterimages.map((record) => record.presentationKey),
    ["rogue"],
  );
  assert.deepEqual(afterimages[0].center, transform.worldToScreen([13.5, 7]));
  assert.equal(afterimages[0].radius, 0.5 * UNIT);
  const short = input({
    trajectories: [trajectory("rogue", [14.2, 7], [15, 7])],
  });
  assert.deepEqual(ofKind(records(short), "phantom_afterimage"), []);
  assert.deepEqual(
    ofKind(records(input({ filters: { stride_glide: false } })), "phantom_afterimage"),
    [],
  );
});

test("the Focus Trail draws eight dots on this tick's own path for the selected champion", () => {
  const trails = ofKind(records(input()), "focus_trail");
  assert.equal(trails.length, 1);
  const [trail] = trails;
  assert.equal(trail.presentationKey, "walker");
  assert.equal(trail.fallen, false);
  assert.equal(trail.dots.length, 8);
  // Oldest dot first, on the start spot; the newest is 7/8 of the way along.
  assert.deepEqual(
    { x: trail.dots[0].x, y: trail.dots[0].y },
    transform.worldToScreen([4, 5]),
  );
  const newest = transform.worldToScreen([4 + 7 / 8, 5]);
  assert.ok(Math.abs(trail.dots[7].x - newest.x) < 1e-9);
  assert.equal(trail.dots[7].opacity, 0.5);
  assert.ok(Math.abs(trail.dots[0].opacity - 0.5 * 0.8 ** 7) < 1e-12);
  assert.equal(trail.dots[7].r, Math.max(1.5, 0.12 * UNIT));
  assert.ok(trail.dots[0].r >= 1.5);
  // A Charge's trail follows the rush, then the stride.
  const charged = records(
    input({
      scene: { ...input().scene, selection: { selected_presentation_key: "charger" } },
    }),
  );
  const chargeTrail = ofKind(charged, "focus_trail")[0];
  assert.deepEqual(
    { x: chargeTrail.dots[4].x, y: chargeTrail.dots[4].y },
    transform.worldToScreen([10.5, 5]),
  );
  // Each dot knows its leg (rush 0, stride 1) and how far along it sits.
  assert.equal(chargeTrail.legs, 2);
  assert.equal(trail.legs, 1);
  assert.deepEqual(
    trail.dots.map((/** @type {Record<string, number>} */ dot) => [dot.leg, dot.share]),
    trail.dots.map((/** @type {unknown} */ _dot, /** @type {number} */ k) => [
      0,
      k / 8,
    ]),
  );
  const chargeLegs = chargeTrail.dots.map(
    (/** @type {Record<string, number>} */ dot) => dot.leg,
  );
  assert.equal(chargeLegs[0], 0);
  assert.equal(chargeLegs.at(-1), 1);
  assert.ok(
    chargeLegs.every(
      (/** @type {number} */ leg, /** @type {number} */ k) =>
        k === 0 || leg >= chargeLegs[k - 1],
    ),
  );
  // No selection, no trail; a fallen champion's trail waits for its Fall beat.
  const none = input();
  none.scene.selection = { selected_presentation_key: null };
  assert.deepEqual(ofKind(records(none), "focus_trail"), []);
  const fallen = input();
  fallen.scene.agents[0] = agent("walker", { position: [5, 5], life_state: "dead" });
  assert.equal(ofKind(records(fallen), "focus_trail")[0].fallen, true);
});

test("each switch removes only its own marks", () => {
  const all = records(input());
  const count = (/** @type {ReadonlyArray<Record<string, any>>} */ list) =>
    ["start_spot", "phantom_afterimage", "focus_trail"].map(
      (kind) => ofKind(list, kind).length,
    );
  assert.deepEqual(count(all), [3, 1, 1]);
  assert.deepEqual(
    count(records(input({ filters: { last_move_ghosts: false } }))),
    [0, 1, 1],
  );
  assert.deepEqual(
    count(records(input({ filters: { phantom_afterimage: false } }))),
    [3, 0, 1],
  );
  assert.deepEqual(
    count(records(input({ filters: { focus_trail: false } }))),
    [3, 1, 0],
  );
  assert.deepEqual(
    count(records(input({ filters: { stride_glide: false } }))),
    [3, 0, 1],
  );
  assert.deepEqual(
    PAINT_PARTS.map(({ part, filter }) => [part.kind, filter]),
    [
      ["start_spot", "last_move_ghosts"],
      ["phantom_afterimage", "phantom_afterimage"],
      ["focus_trail", "focus_trail"],
      ["dim_shift", "stride_glide"],
    ],
  );
});

test("Agent POV: no record for an unseen body or an undisclosed start; no Charge pin", () => {
  const pov = input({
    audience: "agent_pov",
    trajectories: [
      // Came into sight this tick: no start spot.
      trajectory("walker", null, [5, 5]),
      // Left sight this tick: not in the scene.
      trajectory("hidden", [1, 1], null),
      // Agent POV never has a post-Charge spot; one sent anyway is ignored.
      trajectory("charger", [9, 5], [12, 5], [11.5, 5]),
      // A successor that does not match the scene body is refused.
      trajectory("rogue", [13.5, 7], [14, 7]),
    ],
  });
  pov.scene.audience = "agent_pov";
  const result = records(pov);
  assert.deepEqual(
    result.map((record) => [record.kind, record.presentationKey]),
    [["start_spot", "charger"]],
  );
  assert.equal(result[0].pin, null);
  assert.equal(result[0].charged, false);
  assert.equal(result[0].legs.length, 1);
});

test("movementSource reads only the audience's own events", async () => {
  const oracleRaw = await fixturePresentation("replay_oracle");
  for (const row of oracleRaw.latest_events.agent_phase_trajectories) {
    if (row.agent_public_agent_id === "agent-slot-0") {
      row.transition_start.position = [1.0, 1.5];
      row.post_charge.position = [1.0, 1.5];
    }
  }
  const oracle = await normalizeAuthorizedPresentationFrameV1(oracleRaw);
  const oracleSource = movementSource(
    /** @type {Record<string, any>} */ (oracle).latest_events,
    "researcher",
  );
  assert.deepEqual(
    oracleSource.trajectories,
    /** @type {Record<string, any>} */ (oracle).latest_events.agent_phase_trajectories,
  );
  assert.deepEqual(oracleSource.returned, []);
  const scene = authorizedPresentationSceneView(oracle);
  const spots = ofKind(
    records({ scene, transform, audience: "researcher", ...oracleSource }),
    "start_spot",
  );
  assert.deepEqual(
    spots.map((spot) => spot.publicAgentId),
    ["agent-slot-0"],
  );

  for (const name of [
    "replay_no_shared_obs_agent_pov",
    "replay_no_shared_agent_appearance",
    "replay_no_shared_agent_disappearance",
  ]) {
    const frame = await normalizeAuthorizedPresentationFrameV1(
      await fixturePresentation(name),
    );
    const source = movementSource(
      /** @type {Record<string, any>} */ (frame).visual_events,
      "agent_pov",
    );
    const value = /** @type {Record<string, any>} */ (frame);
    assert.deepEqual(
      source.trajectories,
      value.visual_events.agent_phase_trajectories,
      name,
    );
    assert.notDeepEqual(
      source.trajectories,
      value.latest_events?.agent_phase_trajectories ?? null,
      name,
    );
    const povScene = authorizedPresentationSceneView(frame);
    const sceneKeys = new Set(
      (povScene?.agents ?? []).map(
        (/** @type {Record<string, any>} */ row) => row.presentation_key,
      ),
    );
    for (const record of records({
      scene: povScene,
      transform,
      audience: "agent_pov",
      ...source,
    })) {
      assert.ok(sceneKeys.has(record.presentationKey), name);
    }
  }
  const empty = { trajectories: [], returned: [] };
  assert.deepEqual(movementSource(null, "researcher"), empty);
  // Agent POV refuses any inventory that is not the fog-filtered one.
  assert.deepEqual(
    movementSource(
      { summary_kind: "replay_incoming_inventory", agent_phase_trajectories: [{}] },
      "agent_pov",
    ),
    empty,
  );
  const returns = movementSource(
    {
      summary_kind: "agent_pov_fog_filtered_visual_events",
      events: [
        { event_kind: "agent_respawned", agent_anchor: { presentation_key: "pov_a" } },
        { event_kind: "agent_died", agent_anchor: { presentation_key: "pov_b" } },
      ],
    },
    "agent_pov",
  );
  assert.deepEqual(returns.returned, ["pov_a"]);
});

test("build draws export-safe SVG with END opacities and update reprojects in place", () => {
  const result = records(input());
  const nodes = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (build(result, doc))
  );
  assert.equal(nodes.length, result.length);
  const text = nodes.map(markup).join("");
  for (const banned of [
    "url(",
    "Gradient",
    "<mask",
    "<filter",
    "<image",
    " id=",
    "filter=",
  ]) {
    assert.equal(text.includes(banned), false, banned);
  }
  assert.equal(/#[0-9a-f]{3,8}/i.test(text), false, "no raw colours in nodes");
  const walker = nodes[0];
  assert.equal(walker.getAttribute("data-movement-kind"), "start_spot");
  assert.equal(walker.getAttribute("data-team"), "a");
  assert.equal(
    walker.querySelector(".movement-start-ring")?.getAttribute("opacity"),
    "0.28",
  );
  assert.equal(
    walker.querySelector(".movement-hairline")?.getAttribute("opacity"),
    "0.3",
  );
  const charger = nodes.find(
    (node) => node.getAttribute("data-presentation-key") === "charger",
  );
  assert.equal(charger?.querySelector(".movement-pin")?.getAttribute("opacity"), "0.5");
  assert.equal(charger?.querySelectorAll(".movement-hairline").length, 2);
  const afterimage = nodes.find(
    (node) => node.getAttribute("data-movement-kind") === "phantom_afterimage",
  );
  assert.ok(afterimage);
  assert.deepEqual(
    afterimage.children.map((circle) => circle.getAttribute("opacity")),
    ["0", "0", "0"],
  );
  const trail = nodes.find(
    (node) => node.getAttribute("data-movement-kind") === "focus_trail",
  );
  assert.equal(trail?.querySelectorAll(".movement-trail-dot").length, 8);
  assert.equal(trail?.hasAttribute("data-beat-hold"), false);
  for (const node of nodes)
    assert.ok(node.descendants().length + 1 <= BUDGET.nodesPerRecord);

  const moved = createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: 1648,
    viewportHeight: 848,
    padding: 24,
  });
  const next = records(input({ transform: moved }));
  update(/** @type {any} */ (nodes), next);
  const ring = walker.querySelector(".movement-start-ring");
  assert.equal(Number(ring?.getAttribute("cx")), moved.worldToScreen([4, 5]).x);
  assert.equal(Number(ring?.getAttribute("r")), next[0].radius);
  assert.deepEqual(build([], /** @type {any} */ (null)), []);
});

test("a fallen champion's trail is held for its Fall beat and hidden at END", () => {
  const fallen = input();
  fallen.scene.agents[0] = agent("walker", { position: [5, 5], life_state: "dead" });
  const nodes = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (build(ofKind(records(fallen), "focus_trail"), doc))
  );
  const hold = nodes[0];
  assert.equal(hold.getAttribute("data-beat-hold"), "fall");
  assert.equal(hold.getAttribute("data-beat-key"), "walker");
  assert.equal(hold.getAttribute("data-beat-side"), "before");
  assert.equal(hold.getAttribute("opacity"), "0");
  assert.equal(hold.getAttribute("data-beat-from"), "1");
});

test("animationSpecs follow the beat clock with fill backwards only", () => {
  const root = new FakeElement("g");
  root.append(
    .../** @type {FakeElement[]} */ (
      /** @type {unknown} */ (build(records(input()), doc))
    ),
  );
  const beats = {
    glide: true,
    windows: { glide: [0, 260], chargeRush: [0, 110] },
  };
  const played = animationSpecs(/** @type {any} */ (root), {
    epochKey: "e1",
    mode: "played",
    motionMode: "normal",
    beats,
  });
  for (const spec of played) {
    assert.equal(spec.options.fill, "backwards");
    assert.match(spec.id, /^mbg:e1:movement:\d+$/);
  }
  const byClass = (/** @type {string} */ name) =>
    played.filter((spec) =>
      /** @type {FakeElement} */ (/** @type {unknown} */ (spec.element))
        .getAttribute("class")
        ?.startsWith(name),
    );
  const rings = byClass("movement-start-ring");
  assert.equal(rings.length, 3);
  assert.deepEqual(rings[0].keyframes, [{ opacity: 0 }, { opacity: 0.28 }]);
  assert.equal(rings[0].options.duration, 120);
  const hairlines = byClass("movement-hairline");
  const rush = hairlines.find(
    (spec) =>
      /** @type {FakeElement} */ (/** @type {unknown} */ (spec.element)).getAttribute(
        "data-leg",
      ) === "rush",
  );
  assert.deepEqual(rush?.keyframes, [
    { strokeDasharray: "0 1" },
    { strokeDasharray: "1 0" },
  ]);
  assert.deepEqual([rush?.options.delay, rush?.options.duration], [0, 110]);
  const chargeStride = hairlines.filter(
    (spec) => spec.options.delay === 110 && spec.options.duration === 150,
  );
  assert.equal(chargeStride.length, 1);
  const pin = byClass("movement-pin")[0];
  assert.equal(pin.options.duration, 110);
  // Each Focus Trail dot appears as the gliding body passes it: dot k at k/8
  // of a plain path waits until the eased glide has covered that share.
  const dots = played.filter(
    (spec) =>
      /** @type {FakeElement} */ (/** @type {unknown} */ (spec.element)).getAttribute(
        "class",
      ) === "movement-trail-dot",
  );
  assert.equal(dots.length, 8);
  const delays = dots.map((spec) => Number(spec.options.delay));
  assert.equal(delays[0], 0);
  assert.ok(
    delays.every((delay, k) => k === 0 || delay > delays[k - 1]),
    "oldest first",
  );
  assert.ok(
    delays.every((delay) => delay < 260),
    "each dot is passed during the glide",
  );
  assert.ok(dots.every((spec) => spec.options.duration === 60));
  assert.equal(
    played.some(
      (spec) =>
        /** @type {FakeElement} */ (/** @type {unknown} */ (spec.element)).getAttribute(
          "class",
        ) === "movement-focus-trail",
    ),
    false,
    "the trail is not faded in as a whole while the body glides",
  );
  const smoke = played.filter((spec) =>
    /** @type {FakeElement} */ (/** @type {unknown} */ (spec.element)).hasAttribute(
      "data-afterimage",
    ),
  );
  assert.deepEqual(
    smoke.map((spec) => spec.options.duration),
    [MOVEMENT.smoke.fillEnd, MOVEMENT.smoke.fillEnd, MOVEMENT.smoke.rimEnd],
  );
  assert.equal(
    /** @type {Keyframe[]} */ (smoke[2].keyframes)[1].offset,
    MOVEMENT.smoke.rimHold / MOVEMENT.smoke.rimEnd,
  );

  // A -1 rewind fades every mark in once the bodies have landed: at 240 ms,
  // or later when fallen champions rise first (440 ms here).
  const back = animationSpecs(/** @type {any} */ (root), {
    epochKey: "e2b",
    mode: "step",
    motionMode: "normal",
    beats,
    stepMs: 440,
    stepCueFadeMs: 80,
  });
  assert.ok(back.length > 0);
  assert.equal(
    back.some((spec) =>
      /** @type {FakeElement} */ (/** @type {unknown} */ (spec.element)).hasAttribute(
        "data-afterimage",
      ),
    ),
    false,
  );
  for (const spec of back) {
    assert.equal(spec.options.delay, 440);
    assert.equal(spec.options.duration, 80);
    assert.equal(/** @type {Keyframe[]} */ (spec.keyframes)[0].opacity, 0);
  }
  const reduced = animationSpecs(/** @type {any} */ (root), {
    epochKey: "e3",
    mode: "played",
    motionMode: "reduced",
    beats,
  });
  assert.ok(reduced.every((spec) => spec.options.duration === 120));
  const snapped = animationSpecs(/** @type {any} */ (root), {
    epochKey: "e4",
    mode: "played",
    motionMode: "normal",
    beats: { ...beats, glide: false },
  });
  const snappedRing = snapped.find((spec) =>
    /** @type {FakeElement} */ (/** @type {unknown} */ (spec.element))
      .getAttribute("class")
      ?.startsWith("movement-start-ring"),
  );
  assert.deepEqual(snappedRing?.keyframes, [
    { opacity: 0.48, easing: "ease-out" },
    { opacity: 0.28 },
  ]);
  assert.equal(snappedRing?.options.duration, 400);
  assert.deepEqual(
    animationSpecs(/** @type {any} */ (root), {
      epochKey: "e5",
      mode: "played",
      motionMode: "off",
    }),
    [],
  );
  assert.deepEqual(
    animationSpecs(null, { epochKey: "e6", mode: "played", motionMode: "normal" }),
    [],
  );
});

test("dimShiftSpecs hold the body, dim it to 0.3, cut to END and brighten", () => {
  const element = /** @type {Element} */ (
    /** @type {unknown} */ (new FakeElement("g"))
  );
  const toScreen = (/** @type {readonly [number, number]} */ point) => ({
    x: point[0] * 10,
    y: point[1] * 10,
  });
  const [move, fade] = dimShiftSpecs(
    element,
    [
      [1, 1],
      [3, 1],
    ],
    [0, 1],
    toScreen,
    260,
    "mbg:e:glide:a",
  );
  assert.equal(move.options.fill, "backwards");
  assert.equal(move.options.composite, "add");
  assert.equal(move.options.duration, 0.4 * 260);
  assert.deepEqual(move.keyframes, [
    { offset: 0, transform: "translate(-20px, 0px)" },
    { offset: 1, transform: "translate(-20px, 0px)" },
  ]);
  assert.equal(fade.options.fill, "backwards");
  assert.deepEqual(fade.keyframes, [
    { offset: 0, opacity: 1 },
    { offset: 0, opacity: 1 },
    { offset: 0.4, opacity: 0.3 },
    { offset: 1, opacity: 1 },
  ]);
  const charged = dimShiftSpecs(
    element,
    [
      [1, 1],
      [2, 1],
      [3, 1],
    ],
    [0, 110 / 260, 1],
    toScreen,
    260,
    "mbg:e:glide:b",
  );
  const cutAt = 110 + 0.4 * 150;
  assert.equal(charged[0].options.duration, cutAt);
  assert.deepEqual(
    /** @type {Keyframe[]} */ (charged[0].keyframes).map((frame) => frame.transform),
    ["translate(-20px, 0px)", "translate(-10px, 0px)", "translate(-10px, 0px)"],
  );
  assert.equal(/** @type {Keyframe[]} */ (charged[0].keyframes)[1].offset, 110 / cutAt);
  assert.deepEqual(dimShiftSpecs(element, [[1, 1]], [0], toScreen, 260, "x"), []);
});

test("tooltip names the Start Spot and nothing else", () => {
  const result = records(input());
  const descriptor = /** @type {Record<string, any>} */ (tooltip(result[0]));
  assert.equal(descriptor.title, "Start Spot");
  assert.deepEqual(
    descriptor.rows.map((/** @type {Record<string, any>} */ row) => row.label),
    ["Champion", "Started At", "Ended At", "Moved"],
  );
  assert.equal(descriptor.rows[3].value, "1 unit");
  assert.equal(tooltip(ofKind(result, "focus_trail")[0]), null);
  assert.equal(tooltip({ kind: "anything" }), null);
});
