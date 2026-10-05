/**
 * @file Checks the Atmosphere look (Moonlit Night, `src/looks/atmosphere.js`):
 * every switch is off by default; Moon Motes are six seeded pairs mirrored about
 * the floor's middle at 18%, stable per map and dropped below 20 px per unit;
 * the Last Watch Veil steps at 80, 85, 90 and 95% of the episode limit (2.5% to
 * 7%) and is beat-held to the Settle point only on the tick its step changes;
 * Death Dimming draws only on a Fall beat this audience plays, in normal motion
 * (never reduced or without a seen fall, as in Agent POV), with windows over the
 * Red Zone rule lines, END opacity 0 and a 28% browser animation with fill
 * "backwards"; and build() export safety with fill-rule written as an attribute.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as atmosphere from "../src/looks/atmosphere.js";

class FakeElement {
  /** @param {string} tagName */
  constructor(tagName) {
    this.tagName = tagName;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    this.textContent = "";
    /** @type {FakeElement | null} */
    this.parentElement = null;
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
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    this.children.push(...nodes);
  }
  /** @returns {FakeElement | null} */
  get firstElementChild() {
    return this.children[0] ?? null;
  }
  /** @returns {FakeElement | null} */
  get lastChild() {
    return this.children.at(-1) ?? null;
  }
  getBoundingClientRect() {
    return { left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0 };
  }
}

const doc = /** @type {Document} */ (
  /** @type {unknown} */ ({
    createElementNS: (/** @type {string} */ _namespace, /** @type {string} */ tag) =>
      new FakeElement(tag),
  })
);

/** @param {FakeElement} node @returns {FakeElement[]} */
function walk(node) {
  return [node, ...node.children.flatMap(walk)];
}

/** @param {ReadonlyArray<unknown>} nodes */
function assertExportSafe(nodes) {
  for (const node of nodes.flatMap((root) => walk(/** @type {FakeElement} */ (root)))) {
    assert.ok(
      ![
        "linearGradient",
        "radialGradient",
        "mask",
        "filter",
        "image",
        "clipPath",
        "pattern",
      ].includes(node.tagName),
      node.tagName,
    );
    assert.equal(node.attributes.has("id"), false);
    for (const value of node.attributes.values()) {
      assert.equal(value.includes("url("), false, value);
    }
  }
}

/** @param {number} viewportWidth */
function transformFor(viewportWidth) {
  return createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth,
    viewportHeight: viewportWidth / 1.6,
    padding: 24,
  });
}

const MAP = Object.freeze({
  width: 20,
  height: 10,
  obstacles: [{ kind: "pillar", center: [10, 5], radius: 0.9 }],
  red_zone: { team_a_x_range: [0, 3], team_b_x_range: [17, 20] },
});
const ALL_ON = Object.freeze({
  moon_motes: true,
  time_of_night: true,
  death_dimming: true,
});

test("every Moonlit Night switch is off by default", () => {
  const transform = transformFor(1440);
  assert.deepEqual(
    atmosphere.records({ transform, map: MAP, tick: 290, episodeLimit: 300 }),
    [],
  );
  assert.deepEqual(
    atmosphere.records({ transform, map: MAP, part: "dim", fallBeat: [1100, 2600] }),
    [],
  );
});

test("Moon Motes are six mirrored pairs at 18%, stable per map, and drop below 20 px per unit", () => {
  const transform = transformFor(1440);
  const [motes] = atmosphere.records({
    transform,
    map: MAP,
    mapId: 47,
    filters: ALL_ON,
  });
  assert.equal(motes.kind, "moon-motes");
  assert.equal(motes.dots.length, 12);
  assert.equal(motes.opacity, 0.18);
  const middle = transform.worldToScreen([10, 0]).x;
  for (let index = 0; index < 12; index += 2) {
    const [left, right] = [motes.dots[index], motes.dots[index + 1]];
    assert.ok(Math.abs(left.cx - middle + (right.cx - middle)) < 1e-6);
    assert.equal(left.cy, right.cy);
  }
  const again = atmosphere.records({ transform, map: MAP, mapId: 47, filters: ALL_ON });
  assert.equal(JSON.stringify(again[0]), JSON.stringify(motes));
  const other = atmosphere.records({ transform, map: MAP, mapId: 12, filters: ALL_ON });
  assert.notEqual(JSON.stringify(other[0].dots), JSON.stringify(motes.dots));
  const small = atmosphere.records({
    transform: transformFor(360),
    map: MAP,
    mapId: 47,
    filters: ALL_ON,
  });
  assert.equal(
    small.some((record) => record.kind === "moon-motes"),
    false,
  );
});

test("the Last Watch Veil steps over the last fifth and holds only on a changing tick", () => {
  const transform = transformFor(960);
  const filters = { time_of_night: true };
  /** @param {number} tick */
  const veil = (tick) =>
    atmosphere
      .records({ transform, map: MAP, filters, tick, episodeLimit: 300 })
      .find((record) => record.kind === "last-watch-veil");
  assert.equal(veil(239), undefined);
  assert.deepEqual([veil(240)?.opacity, veil(240)?.fromOpacity], [0.025, 0]);
  assert.deepEqual([veil(241)?.opacity, veil(241)?.fromOpacity], [0.025, null]);
  assert.deepEqual([veil(255)?.opacity, veil(255)?.fromOpacity], [0.04, 0.025]);
  assert.equal(veil(299)?.opacity, 0.07);
  assert.equal(
    atmosphere.records({ transform, map: MAP, filters, tick: 290 }).length,
    0,
  );
  const [changing, steady] = [veil(255), veil(256)].map(
    (record) =>
      /** @type {FakeElement} */ (
        /** @type {unknown} */ (atmosphere.build([/** @type {any} */ (record)], doc)[0])
      ),
  );
  assert.equal(changing.getAttribute("data-beat-hold"), "score");
  assert.equal(changing.getAttribute("data-beat-from"), "0.625");
  assert.equal(changing.getAttribute("data-beat-fade"), "400");
  assert.equal(steady.hasAttribute("data-beat-hold"), false);
});

test("Death Dimming plays only on a seen Fall beat in normal motion", () => {
  const transform = transformFor(1440);
  const dim = atmosphere.records({
    transform,
    map: MAP,
    filters: ALL_ON,
    part: "dim",
    fallBeat: [1100, 2600],
  });
  assert.equal(dim.length, 1);
  assert.equal(
    (dim[0].d.match(/M/g) ?? []).length,
    3,
    "the floor and two rule-line windows",
  );
  assert.equal(
    atmosphere.records({
      transform,
      map: MAP,
      filters: ALL_ON,
      part: "dim",
      fallBeat: null,
    }).length,
    0,
    "no seen fall (an Agent POV team that saw no death): no dim",
  );
  assert.equal(
    atmosphere.records({
      transform,
      map: MAP,
      filters: ALL_ON,
      part: "dim",
      fallBeat: [1100, 2600],
      motionMode: "reduced",
    }).length,
    0,
  );
  const plain = atmosphere.records({
    transform,
    map: MAP,
    filters: { ...ALL_ON, red_zone_floors: false },
    part: "dim",
    fallBeat: [1100, 2600],
  });
  assert.equal((plain[0].d.match(/M/g) ?? []).length, 1);
  const [node] = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (atmosphere.build(dim, doc))
  );
  assert.equal(node.getAttribute("fill-rule"), "evenodd");
  assert.equal(node.getAttribute("opacity"), "0");
  const animation = atmosphere.dimAnimation(/** @type {any} */ (node), dim[0], {
    epochKey: "e7",
  });
  assert.equal(animation.options.delay, 1100);
  assert.equal(animation.options.duration, 1500);
  assert.equal(animation.options.fill, "backwards");
  assert.equal(animation.keyframes[1].opacity, 0.28);
  assert.equal(animation.keyframes[1].offset, 120 / 1500);
  assert.equal(animation.keyframes[2].offset, (1500 - 600) / 1500);
  assert.equal(animation.keyframes.at(-1)?.opacity, 0);
  assert.equal(animation.id, "mbg:e7:death-dim");
});

test("air nodes are export-safe and never read a body", () => {
  const transform = transformFor(1440);
  const input = {
    transform,
    map: MAP,
    mapId: 47,
    filters: ALL_ON,
    tick: 290,
    episodeLimit: 300,
  };
  const records = atmosphere.records(input);
  const withBodies = atmosphere.records(
    /** @type {any} */ ({
      ...input,
      scene: { agents: [{ position: [3, 3], presentation_key: "x" }] },
    }),
  );
  assert.equal(JSON.stringify(records), JSON.stringify(withBodies));
  const nodes = atmosphere.build(records, doc);
  assert.equal(nodes.length, 2);
  assertExportSafe(nodes);
  const moved = atmosphere.records({ ...input, transform: transformFor(960) });
  atmosphere.update(nodes, moved);
  const veil = /** @type {FakeElement} */ (/** @type {unknown} */ (nodes[1]));
  assert.equal(veil.getAttribute("width"), String(moved[1].width));
  assert.equal(atmosphere.tooltip(/** @type {any} */ (records[0])), null);
});
