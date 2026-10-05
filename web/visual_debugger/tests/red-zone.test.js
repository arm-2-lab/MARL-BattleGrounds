/**
 * @file Checks the Red Zone look (Grave Ground, `src/looks/red-zone.js`):
 * records() for both homes (Ash floor, Grave Bone border on the exact rule edge,
 * gravestones at y = k + 0.5, skull and "×2" marks) with a mirror test on both
 * spawn ends (identical look, no team colour); the stop rule around stones; the
 * detail ladder below 20 px per unit (plain line, one 14 px mark); Red Zone
 * Floors off (a plain line on each rule edge); the Home Fall glow (only from the
 * caller's home-fall IDs, only for a victim in this audience's scene, keyed to the
 * first seen fall in public ID order, held at 35% by data-beat-hold "fall") in
 * Oracle View and Agent POV, and its switch; the "x2 Mark" tip; and build()
 * export safety and beat-hold attributes.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as redZone from "../src/looks/red-zone.js";

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

const NORMAL = Object.freeze({ team_a_x_range: [0, 3], team_b_x_range: [17, 20] });
const SWAPPED = Object.freeze({ team_a_x_range: [17, 20], team_b_x_range: [0, 3] });
const MAP = Object.freeze({ width: 20, height: 10, obstacles: [] });
const SCENE = Object.freeze({
  spawn_pads: [
    { team_id: 1, position: [1.5, 5] },
    { team_id: 2, position: [18.5, 5] },
  ],
  agents: [
    { public_agent_id: "agent-slot-10", team_id: 2, presentation_key: "k10" },
    { public_agent_id: "agent-slot-2", team_id: 2, presentation_key: "k2" },
    { public_agent_id: "agent-slot-1", team_id: 1, presentation_key: "k1" },
  ],
});

/** @param {Record<string, any>} input */
function recordsOf(input) {
  return redZone.records({
    transform: transformFor(1440),
    map: MAP,
    scene: SCENE,
    ...input,
  });
}

/** @param {ReadonlyArray<Record<string, any>>} records @param {string} kind @param {number} teamId */
function find(records, kind, teamId) {
  return records.find((record) => record.kind === kind && record.teamId === teamId);
}

test("both homes look the same, mirrored, on either spawn end and with no team colour", () => {
  const transform = transformFor(1440);
  const middle = transform.worldToScreen([10, 0]).x;
  for (const zone of [NORMAL, SWAPPED]) {
    const records = recordsOf({ redZone: zone });
    const left = records.find(
      (record) => record.kind === "grave-border" && record.lineX < middle,
    );
    const right = records.find(
      (record) => record.kind === "grave-border" && record.lineX > middle,
    );
    assert.ok(left && right);
    assert.ok(
      Math.abs(left.lineX - middle + (right.lineX - middle)) < 1e-6,
      "rule lines mirror",
    );
    assert.equal(
      left.runs.replace(/M[\d.]+ /g, "M "),
      right.runs.replace(/M[\d.]+ /g, "M "),
    );
    assert.equal(left.stones.length > 0, true);
    const marks = records.filter((record) => record.kind === "grave-mark");
    assert.equal(marks.length, 4);
    const leftMarks = marks
      .filter((mark) => mark.x < middle)
      .map((mark) => [Number((middle - mark.x).toFixed(6)), mark.y, -mark.dir]);
    const rightMarks = marks
      .filter((mark) => mark.x > middle)
      .map((mark) => [Number((mark.x - middle).toFixed(6)), mark.y, mark.dir]);
    assert.deepEqual(leftMarks, rightMarks);
  }
  // The same home end draws the same picture whichever team owns it.
  const normal = recordsOf({ redZone: NORMAL });
  const swapped = recordsOf({ redZone: SWAPPED });
  assert.equal(
    find(normal, "grave-border", 1)?.runs,
    find(swapped, "grave-border", 2)?.runs,
  );
  const nodes = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (redZone.build(normal, doc))
  );
  assertExportSafe(nodes);
  const classes = nodes
    .flatMap(walk)
    .map((node) => node.getAttribute("class") ?? "")
    .join(" ");
  assert.equal(
    /team-a|team-b/.test(classes),
    false,
    "no team colour class on Grave Ground",
  );
});

test("the border sits on the exact rule edge and stops around a stone on the line", () => {
  const transform = transformFor(1440);
  const clear = find(recordsOf({ redZone: NORMAL }), "grave-border", 1);
  assert.equal(clear?.lineX, transform.worldToScreen([3, 0]).x);
  const blocked = find(
    recordsOf({
      redZone: NORMAL,
      map: { ...MAP, obstacles: [{ kind: "pillar", center: [3, 5], radius: 0.6 }] },
    }),
    "grave-border",
    1,
  );
  assert.ok(
    (blocked?.stones.match(/M/g) ?? []).length <
      (clear?.stones.match(/M/g) ?? []).length,
  );
  const gapTop = transform.worldToScreen([0, 5.7]).y;
  const gapBottom = transform.worldToScreen([0, 4.3]).y;
  for (const [, from, to] of blocked?.runs.matchAll(/M[\d.]+ ([\d.]+)V([\d.]+)/g) ??
    []) {
    const [top, bottom] = [Number(from), Number(to)].sort((a, b) => a - b);
    assert.ok(bottom <= gapTop || top >= gapBottom, "no run crosses the stone");
  }
});

test("below 20 px per unit the stones drop and each home keeps one 14 px mark", () => {
  const transform = transformFor(360);
  assert.ok(transform.worldLengthToScreen(1) < 20);
  const records = redZone.records({
    transform,
    map: MAP,
    scene: SCENE,
    redZone: NORMAL,
  });
  const border = find(records, "grave-border", 1);
  assert.equal(border?.lite, true);
  assert.equal(border?.stones, "");
  const marks = records.filter((record) => record.kind === "grave-mark");
  assert.equal(marks.length, 2);
  assert.ok(marks.every((mark) => mark.size === 14));
});

test("Red Zone Floors off leaves a plain line on each rule edge and no Home Fall", () => {
  const records = recordsOf({
    redZone: NORMAL,
    filters: { red_zone_floors: false },
    homeFallIds: ["agent-slot-2"],
  });
  assert.deepEqual(
    records.map((record) => [record.kind, record.teamId]),
    [
      ["red-zone-line", 1],
      ["red-zone-line", 2],
    ],
  );
  assert.deepEqual(recordsOf({ redZone: null }), []);
});

test("Home Fall lights the victim's home, keyed to its first seen fall in public ID order", () => {
  const records = recordsOf({
    redZone: NORMAL,
    homeFallIds: ["agent-slot-10", "agent-slot-2"],
  });
  const ash = find(records, "home-fall-ash", 2);
  const border = find(records, "home-fall-border", 2);
  assert.equal(ash?.beatKey, "k2");
  assert.equal(ash?.opacity, 0.05);
  assert.equal(border?.beatKey, "k2");
  assert.equal(border?.opacity, 0.35);
  assert.equal(find(records, "home-fall-border", 1), undefined);
  const nodes = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (redZone.build(records, doc))
  );
  const held = nodes.filter((node) => node.hasAttribute("data-beat-hold"));
  assert.equal(held.length, 2);
  for (const node of held) {
    assert.equal(node.getAttribute("data-beat-hold"), "fall");
    assert.equal(node.getAttribute("data-beat-side"), "after");
    assert.equal(node.getAttribute("data-beat-key"), "k2");
    assert.equal(node.getAttribute("data-beat-from"), "0");
    assert.equal(node.getAttribute("data-beat-fade"), "120");
  }
  assert.deepEqual(
    held.map((node) => node.getAttribute("opacity")),
    ["0.05", "0.35"],
  );
  const off = recordsOf({
    redZone: NORMAL,
    homeFallIds: ["agent-slot-2"],
    filters: { red_zone_home_fall: false },
  });
  assert.equal(
    off.some((record) => record.kind.startsWith("home-fall")),
    false,
  );
});

test("Agent POV: a home fall lights only when its victim is in this audience's scene", () => {
  const unseen = recordsOf({
    redZone: NORMAL,
    homeFallIds: ["agent-slot-7"],
    scene: { ...SCENE, audience: "agent_pov" },
  });
  assert.equal(
    unseen.some((record) => record.kind.startsWith("home-fall")),
    false,
  );
  const seen = recordsOf({
    redZone: NORMAL,
    homeFallIds: ["agent-slot-1"],
    scene: { ...SCENE, audience: "agent_pov" },
  });
  assert.equal(find(seen, "home-fall-border", 1)?.beatKey, "k1");
  const without = recordsOf({
    redZone: NORMAL,
    scene: { ...SCENE, audience: "agent_pov" },
  });
  assert.equal(
    without.some((record) => record.kind.startsWith("home-fall")),
    false,
  );
});

test("marks carry the x2 Mark tip; floors and borders have none", () => {
  const records = recordsOf({ redZone: NORMAL });
  const mark = records.find((record) => record.kind === "grave-mark");
  const tip = /** @type {Record<string, any>} */ (
    redZone.tooltip(/** @type {any} */ (mark))
  );
  assert.equal(tip.title, "x2 Mark");
  assert.equal(redZone.tooltip(/** @type {any} */ (records[0])), null);
  const reprojected = redZone.records({
    transform: transformFor(960),
    map: MAP,
    scene: SCENE,
    redZone: NORMAL,
  });
  const nodes = redZone.build(records, doc);
  redZone.update(nodes, reprojected);
  const line = /** @type {FakeElement} */ (/** @type {unknown} */ (nodes[1]))
    .children[1];
  assert.equal(line.getAttribute("d"), find(reprojected, "grave-border", 1)?.runs);
});
