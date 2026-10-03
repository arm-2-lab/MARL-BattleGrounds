/**
 * @file Check the live Grave Ground skull-and-"×2" marks
 * (src/looks/red-zone.js records with `vision: true`, the BattleClient and
 * the DevClient): they are the store's marks (skins/red-zone.js graveMarks
 * and clearSpots), not the vision page's split layout. Each mark is one
 * centered record with no separate skull or text centers, sized 12 px times
 * the field scale (px per unit over 50), on the same world spots the store
 * finds at its own scale, with its whole clear box off the gravestone border;
 * build() draws the skull joined to the "×2" (the skull on the home's back
 * side, the "×2" toward the border, their gap under one mark size). The
 * Replay Viewer's records keep the fixed 12 px marks and widthScale 1.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as redZone from "../src/looks/red-zone.js";

/** @param {number} width @param {number} pad */
function fieldTransform(width, pad) {
  return createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: width,
    viewportHeight: (width * 532) / 1032,
    padding: pad,
  });
}

const map = {
  width: 20,
  height: 10,
  obstacles: [
    { kind: "pillar", center: [4.6, 9.3], radius: 0.4 },
    { kind: "wall", center: [15.5, 0.6], width: 0.8, height: 0.3, theta: 0 },
  ],
};
const red = { team_a_x_range: [15, 20], team_b_x_range: [0, 5] };
const scene = {
  spawn_pads: [
    { team_id: 2, position: [1, 5] },
    { team_id: 1, position: [19, 5] },
  ],
};

/** @param {ReturnType<typeof fieldTransform>} transform @param {boolean} vision */
function marks(transform, vision) {
  return redZone
    .records({ transform, map, redZone: red, scene, vision })
    .filter((record) => record.kind === "grave-mark");
}

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
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      new FakeElement(tag),
  })
);

test("live marks are the store's centered marks, scaled with the field", () => {
  for (const width of [908, 1255]) {
    const transform = fieldTransform(width, (width * 16) / 1032);
    const ppu = transform.worldLengthToScreen(1);
    const s = ppu / 50;
    const live = marks(transform, true);
    assert.equal(live.length, 4);
    for (const mark of live) {
      assert.equal(mark.skullX, undefined);
      assert.equal(mark.textX, undefined);
      assert.ok(Math.abs(mark.size - 12 * s) < 1e-9);
      // The whole 28 px clear box (scaled) stays off the gravestones (0.1 units
      // either side of the border line).
      const border = transform.worldToScreen([mark.dir < 0 ? 5 : 15, 0]).x;
      assert.ok(Math.abs(mark.x - border) - 14 * s > 0.1 * ppu, `${width} ${mark.x}`);
    }
  }
});

test("live marks sit on the store's world spots", () => {
  // At 50 px per unit the field scale is 1, so the live spots are exactly the
  // store's (the Replay Viewer's records at the same scale).
  const transform = fieldTransform(1032, 16);
  assert.ok(Math.abs(transform.worldLengthToScreen(1) - 50) < 1e-9);
  const live = marks(transform, true).map((m) => [m.x, m.y, m.size]);
  const store = marks(transform, false).map((m) => [m.x, m.y, m.size]);
  assert.deepEqual(live, store);
  // A smaller field keeps the same world spots (box and inset scale together).
  const small = fieldTransform(908, (908 * 16) / 1032);
  const toWorld = (/** @type {Readonly<Record<string, any>>} */ m, t = small) => {
    const origin = t.worldToScreen([0, 0]);
    const ppu = t.worldLengthToScreen(1);
    return [(m.x - origin.x) / ppu, (origin.y - m.y) / ppu].map((v) =>
      Math.round(v * 1000),
    );
  };
  assert.deepEqual(
    marks(small, true).map((m) => toWorld(m)),
    marks(transform, true).map((m) => toWorld(m, transform)),
  );
});

test("build joins the skull to the ×2 as the store draws them", () => {
  const transform = fieldTransform(908, (908 * 16) / 1032);
  for (const mark of marks(transform, true)) {
    const [group] = /** @type {FakeElement[]} */ (
      /** @type {unknown} */ (redZone.build([mark], doc))
    );
    const text = group.children.find(
      (c) => c.getAttribute("class") === "grave-mark__text",
    );
    const head = group.children.find(
      (c) => c.getAttribute("class") === "grave-mark__head",
    );
    assert.ok(text && head);
    assert.equal(text.textContent, "×2");
    const textX = Number(text.getAttribute("x"));
    const headX = Number((head.getAttribute("d") ?? "").match(/^M(-?[\d.]+)/u)?.[1]);
    const skullCenter = headX + 0.36 * mark.size * 1.05;
    // The skull is on the home's back side, the ×2 toward the border.
    assert.equal(Math.sign(textX - skullCenter), -mark.dir);
    // Joined: the centers are about one mark size apart (store: 0.95 x 1.05).
    assert.ok(Math.abs(Math.abs(textX - skullCenter) - mark.size * 0.9975) < 0.01);
    assert.equal(Number(text.getAttribute("font-size")), mark.size);
  }
});

test("Replay Viewer marks keep the fixed 12 px size", () => {
  const transform = fieldTransform(1100, 24);
  const replay = redZone.records({ transform, map, redZone: red, scene });
  assert.ok(replay.filter((r) => r.kind === "grave-mark").every((r) => r.size === 12));
  assert.ok(
    replay.filter((r) => r.kind === "grave-border").every((r) => r.widthScale === 1),
  );
});
