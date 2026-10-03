/**
 * @file Checks the Obstacles look (Standing Stones, `src/looks/obstacles.js`):
 * records() geometry for pillars, walls and thin-wall blades (bodies drawn inside
 * the true footprint, the shadow 0.06 units down the screen, wall tops and
 * pillar caps, joined and free blade ends with End Stones); the detail ladder
 * below 22 px per unit; the three switches (Obstacle Skins off gives plain stones
 * on the true footprint, Obstacle Relief and Obstacle Shadows remove only their
 * parts); identical records in Oracle View and Agent POV; the Symbol Tip in the
 * store's plain words (title "Wall", Obstacle ID as a plain number, Width,
 * Height, Turned and the rule "Blocks movement and sight. A Warrior's Charge can
 * pass over it. Bodies never block sight."); and build()
 * export safety (no ids, url(), gradients, masks, filters or images), tooltips on
 * focusable stones and update() reprojection.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as obstacles from "../src/looks/obstacles.js";

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
  obstacles: [
    {
      obstacle_id: "p0",
      kind: "pillar",
      center: [10, 5],
      radius: 0.9,
      width: null,
      height: null,
      theta: 0,
    },
    {
      obstacle_id: "w1",
      kind: "wall",
      center: [6, 3],
      radius: null,
      width: 2,
      height: 0.6,
      theta: 0.5,
    },
    {
      obstacle_id: "t2",
      kind: "wall",
      center: [10, 7.4],
      radius: null,
      width: 0.01,
      height: 3,
      theta: 0,
    },
    {
      obstacle_id: "t3",
      kind: "wall",
      center: [14, 2],
      radius: null,
      width: 3,
      height: 0.01,
      theta: 0,
    },
  ],
});

test("default Standing Stones give shadow, blades, bodies inside the footprint, tops and caps", () => {
  const transform = transformFor(1440);
  const ppu = transform.worldLengthToScreen(1);
  assert.ok(ppu >= 40);
  const records = obstacles.records({ map: MAP, transform });
  assert.deepEqual(
    records.map((record) => record.kind),
    ["stone-shadow", "blade", "blade", "pillar", "wall", "wall-top", "pillar-cap"],
  );
  const pillar = records.find((record) => record.kind === "pillar");
  const center = transform.worldToScreen([10, 5]);
  assert.equal(pillar?.cx, center.x);
  assert.ok(Math.abs((pillar?.r ?? 0) - (0.9 * ppu - 0.75)) < 1e-9);
  const shadow = records[0];
  assert.equal(shadow.opacity, 0.55);
  const drop = 0.06 * ppu;
  const shadowStart = `M${Number((center.x - 0.9 * ppu - transform.mapBounds.left).toFixed(3))},${Number((center.y + drop - transform.mapBounds.top).toFixed(3))}`;
  assert.ok(
    shadow.d.includes(shadowStart),
    "the pillar shadow sits 0.06 units down the screen",
  );
  const [joined, free] = records.filter((record) => record.kind === "blade");
  assert.equal(joined.edgeWidth, 6);
  assert.equal(joined.coreWidth, 3);
  assert.notEqual(joined.ends, "", "a free end gets an End Stone");
  assert.notEqual(free.ends, "");
  assert.ok(Object.isFrozen(records));
});

test("a blade end that meets a stone is pushed half a width into it and gets no End Stone", () => {
  const transform = transformFor(1440);
  const map = {
    ...MAP,
    obstacles: [
      MAP.obstacles[0],
      {
        obstacle_id: "t9",
        kind: "wall",
        center: [10, 7.4],
        radius: null,
        width: 0.01,
        height: 3.2,
        theta: 0,
      },
    ],
  };
  const blade = obstacles
    .records({ map, transform })
    .find((record) => record.kind === "blade");
  const top = transform.worldToScreen([10, 9]);
  const bottom = transform.worldToScreen([10, 5.8]);
  // The lower end (y 5.8) lies inside the pillar: joined. The upper end is free.
  assert.ok(
    blade?.edge.startsWith(
      `M${Number(bottom.x.toFixed(3))},${Number((bottom.y + 3).toFixed(3))}`,
    ),
  );
  assert.equal((blade?.ends.match(/M/g) ?? []).length, 1);
  assert.ok(
    blade?.ends.startsWith(
      `M${Number(top.x.toFixed(3))},${Number((top.y + 1.5).toFixed(3))}`,
    ),
  );
});

test("below 22 px per unit caps, tops and End Stones drop and the lines thin", () => {
  const transform = transformFor(400);
  assert.ok(transform.worldLengthToScreen(1) < 22);
  const records = obstacles.records({ map: MAP, transform });
  assert.equal(
    records.some(
      (record) => record.kind === "wall-top" || record.kind === "pillar-cap",
    ),
    false,
  );
  const blade = records.find((record) => record.kind === "blade");
  assert.equal(blade?.edgeWidth, 3);
  assert.equal(blade?.ends, "");
  assert.equal(records.find((record) => record.kind === "pillar")?.lineWidth, 1);
});

test("each switch removes only its own parts; skins off keeps every stone on its true footprint", () => {
  const transform = transformFor(1440);
  const ppu = transform.worldLengthToScreen(1);
  const plain = obstacles.records({
    map: MAP,
    transform,
    filters: { obstacle_skins: false },
  });
  assert.deepEqual(
    plain.map((record) => record.kind),
    ["plain-pillar", "plain-wall", "plain-wall", "plain-wall"],
  );
  assert.ok(
    Math.abs(plain[0].r - 0.9 * ppu) < 1e-9,
    "the plain pillar is the true radius",
  );
  const flat = obstacles.records({
    map: MAP,
    transform,
    filters: { obstacle_relief: false },
  });
  assert.equal(
    flat.some((record) => record.kind === "wall-top" || record.kind === "pillar-cap"),
    false,
  );
  assert.ok(
    flat
      .filter((record) => record.kind === "blade")
      .every((record) => record.ends === ""),
  );
  assert.ok(flat.some((record) => record.kind === "stone-shadow"));
  const unshadowed = obstacles.records({
    map: MAP,
    transform,
    filters: { obstacle_shadows: false },
  });
  assert.equal(
    unshadowed.some((record) => record.kind === "stone-shadow"),
    false,
  );
  assert.equal(
    unshadowed.length,
    obstacles.records({ map: MAP, transform }).length - 1,
  );
});

test("stones are identical in Oracle View and Agent POV and never read a champion", () => {
  const transform = transformFor(960);
  const oracle = obstacles.records(
    /** @type {any} */ ({
      map: MAP,
      transform,
      audience: "researcher",
      scene: { agents: [{ x: 1 }] },
    }),
  );
  const pov = obstacles.records(
    /** @type {any} */ ({ map: MAP, transform, audience: "agent_pov", scene: null }),
  );
  assert.equal(JSON.stringify(oracle), JSON.stringify(pov));
});

test("the Stone tip names the rule, type, ID, size and turn", () => {
  const transform = transformFor(960);
  const records = obstacles.records({ map: MAP, transform });
  const wall = records.find((record) => record.kind === "wall");
  const descriptor = /** @type {Record<string, any>} */ (
    obstacles.tooltip(/** @type {any} */ (wall))
  );
  assert.equal(descriptor.summary, obstacles.STONE_RULE);
  assert.equal(
    obstacles.STONE_RULE,
    "Blocks movement and sight. A Warrior's Charge can pass over it. Bodies never block sight.",
  );
  assert.equal(descriptor.title, "Wall");
  const rows = Object.fromEntries(
    descriptor.rows.map((/** @type {any} */ row) => [row.label, row.value]),
  );
  assert.equal(rows["Obstacle ID"], "1");
  assert.equal(rows.Width, "2 units");
  assert.equal(rows.Height, "0.6 units");
  assert.equal(rows.Turned, "29° (0.5 rad)");
  assert.equal(obstacles.tooltip(/** @type {any} */ ({ kind: "wall-top" })), null);
});

test("build is export-safe, stones are focusable with tips, and update moves them", () => {
  const records = obstacles.records({ map: MAP, transform: transformFor(1440) });
  const nodes = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (obstacles.build(records, doc))
  );
  assert.equal(nodes.length, records.length);
  assertExportSafe(nodes);
  const focusable = nodes
    .flatMap(walk)
    .filter((node) => node.getAttribute("tabindex") === "0");
  assert.equal(focusable.length, 4, "two bodies and two blade hit lines");
  assert.ok(focusable.every((node) => node.hasAttribute("aria-description")));
  const moved = obstacles.records({ map: MAP, transform: transformFor(960) });
  obstacles.update(/** @type {any} */ (nodes), moved);
  const pillar = nodes[moved.findIndex((record) => record.kind === "pillar")];
  assert.equal(
    pillar.getAttribute("cx"),
    String(moved.find((record) => record.kind === "pillar")?.cx),
  );
});
