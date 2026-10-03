/**
 * @file Check that the Home End Marks (src/looks/team-identity.js, Homeland
 * Brackets) are drawn whole inside the arena box, as the store draws them:
 * markSize picks the spec's product crest (16 px, 14 px out) from a 24 px
 * margin, the store stage crest (14 px, 8.5 px out) from 16.4 px, and that
 * stage crest scaled by margin / 16.4 below it. With the live field's pad
 * (16/1032 of the arena width) at the 1366, 1440 and 1920 wide arenas, the
 * A and B crests (keyline included) and both of Team B's bracket lines, on
 * the end and on the long sides, stay inside the viewport; the Replay
 * Viewer's 24 px pad keeps the product size; build() and update() draw every
 * crest part at the record's size.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as teamIdentity from "../src/looks/team-identity.js";

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

const map = {
  width: 20,
  height: 10,
  obstacles: [],
  red_zone: { team_a_x_range: [15, 20], team_b_x_range: [0, 5] },
};

/** @param {number} width @param {number} pad */
function live(width, pad) {
  return createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: width,
    viewportHeight: (width * 532) / 1032,
    padding: pad,
  });
}

/** @param {string} d @returns {number[][]} */
function points(d) {
  return [...d.matchAll(/(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)/gu)].map((m) => [
    Number(m[1]),
    Number(m[2]),
  ]);
}

test("markSize: product from 24 px, stage from 16.4 px, scaled below", () => {
  const product = teamIdentity.markSize(24);
  assert.equal(product.axis, 14);
  assert.equal(product.crest.outer, 8);
  const stage = teamIdentity.markSize(16.4);
  assert.equal(stage.axis, 8.5);
  assert.equal(stage.crestBreak, 10);
  assert.deepEqual(stage.crest, teamIdentity.FOOTPRINT.stage.crest);
  assert.deepEqual(teamIdentity.markSize(20), stage);
  const small = teamIdentity.markSize(14);
  const k = 14 / 16.4;
  assert.ok(Math.abs(small.axis - 8.5 * k) < 1e-3);
  assert.ok(Math.abs(small.crest.outer - 7 * k) < 1e-3);
  assert.ok(Math.abs(small.teamBOffset - 1.5 * k) < 1e-3);
});

test("live crests and both Team B lines stay inside the arena box", () => {
  // Arena widths measured on the real page: about 790 (1366 x 768), 908
  // (1440 x 900) and 1255 (1920 x 1080) CSS px.
  for (const width of [790, 908, 1255]) {
    const transform = live(width, (width * 16) / 1032);
    const view = transform.viewportBounds;
    const records = teamIdentity.records({ transform, map });
    assert.equal(records.length, 2);
    for (const record of records) {
      const reach = record.crest.outer + record.crest.keyline;
      assert.ok(
        record.crestX - reach >= view.left - 1e-6,
        `${width} ${record.letter} left`,
      );
      assert.ok(
        record.crestX + reach <= view.right + 1e-6,
        `${width} ${record.letter} right`,
      );
      // Clear of the floor: the crest never covers the floor edge.
      const floor = transform.mapBounds;
      assert.ok(
        record.side === "left"
          ? record.crestX + reach <= floor.left
          : record.crestX - reach >= floor.right,
        `${width} ${record.letter} floor`,
      );
      const half = record.lineWidth / 2;
      for (const [x, y] of points(record.d)) {
        assert.ok(x - half >= view.left - 1e-6 && x + half <= view.right + 1e-6);
        assert.ok(y - half >= view.top - 1e-6 && y + half <= view.bottom + 1e-6);
      }
    }
  }
});

test("the Replay Viewer's 24 px pad keeps the product crest", () => {
  const transform = live(1100, 24);
  const records = teamIdentity.records({ transform, map });
  const left = records.find((record) => record.side === "left");
  assert.ok(left);
  assert.equal(left.crestX, transform.mapBounds.left - 14);
  assert.equal(left.crest.outer, 8);
});

test("build and update draw every crest part at the record's size", () => {
  const records = teamIdentity.records({
    transform: live(908, (908 * 16) / 1032),
    map,
    audience: "agent_pov",
    povTeamId: 2,
  });
  const nodes = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (teamIdentity.build(records, doc))
  );
  /** @param {FakeElement} group @param {string} name */
  const part = (group, name) =>
    group.children[1].children.find((c) => c.getAttribute("class") === name);
  const teamB = records[0];
  assert.equal(
    Number(part(nodes[0], "home-crest__field")?.getAttribute("r")),
    teamB.crest.disc,
  );
  assert.equal(
    Number(part(nodes[0], "home-crest__ring")?.getAttribute("r")),
    teamB.crest.ring,
  );
  assert.equal(
    Number(part(nodes[0], "home-crest__gap")?.getAttribute("stroke-width")),
    teamB.crest.gapWidth,
  );
  assert.equal(
    Number(part(nodes[0], "home-crest__text")?.getAttribute("font-size")),
    teamB.crest.letter,
  );
  assert.ok(part(nodes[0], "home-crest__pov"));
  const wide = teamIdentity.records({
    transform: live(1600, 24),
    map,
    audience: "agent_pov",
    povTeamId: 2,
  });
  teamIdentity.update(/** @type {any} */ (nodes), wide);
  assert.equal(Number(part(nodes[0], "home-crest__ring")?.getAttribute("r")), 6.5);
  assert.equal(Number(part(nodes[0], "home-crest__keyline")?.getAttribute("r")), 8.7);
  assert.equal(
    Number(part(nodes[0], "home-crest__ring")?.getAttribute("cx")),
    wide[0].crestX,
  );
});
