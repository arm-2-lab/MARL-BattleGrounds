/**
 * @file Checks the Team Identity look (Homeland Brackets,
 * `src/looks/team-identity.js`): records() follows each team's recorded home
 * side (its Red Zone, else its spawn pads; never the screen side) on both spawn
 * ends; Team A is one 2 px line and Team B two 1 px lines, 14 px outside the
 * floor and as deep as the team's Red Zone; the crest sits on the end line at
 * mid-height; an unclear record or Home End Marks off draws nothing; the POV
 * Home Mark rim appears only in Agent POV on the POV team's crest; and build()
 * export safety, Team B's gap ring and the team tip.
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

const transform = createViewportTransform({
  worldWidth: 20,
  worldHeight: 10,
  viewportWidth: 1440,
  viewportHeight: 900,
  padding: 24,
});

/** @param {Record<string, any>} redZone */
function mapWith(redZone) {
  return { width: 20, height: 10, obstacles: [], red_zone: redZone };
}

test("brackets follow the recorded home side on both spawn ends", () => {
  const bounds = transform.mapBounds;
  const normal = teamIdentity.records({
    transform,
    map: mapWith({ team_a_x_range: [0, 3], team_b_x_range: [17, 20] }),
  });
  assert.deepEqual(
    normal.map((record) => [
      record.teamId,
      record.side,
      record.lineWidth,
      record.letter,
    ]),
    [
      [2, "right", 1, "B"],
      [1, "left", 2, "A"],
    ],
  );
  assert.equal(normal[1].crestX, bounds.left - 14);
  assert.equal(normal[0].crestX, bounds.right + 14);
  assert.equal(normal[1].crestY, (bounds.top + bounds.bottom) / 2);
  const inner = transform.worldToScreen([3, 0]).x;
  assert.ok(
    normal[1].d.startsWith(
      `M${Number(inner.toFixed(3))} ${Number((bounds.top - 14).toFixed(3))}`,
    ),
  );
  assert.equal(
    (normal[0].d.match(/M/g) ?? []).length,
    4,
    "Team B: two lines, each broken by the crest",
  );
  assert.equal((normal[1].d.match(/M/g) ?? []).length, 2);
  const swapped = teamIdentity.records({
    transform,
    map: mapWith({ team_a_x_range: [17, 20], team_b_x_range: [0, 3] }),
  });
  assert.deepEqual(
    swapped.map((record) => [record.teamId, record.side]),
    [
      [2, "left"],
      [1, "right"],
    ],
  );
});

test("without a Red Zone the spawn pads decide and only the end line is drawn", () => {
  const scene = {
    map: { width: 20, height: 10, obstacles: [] },
    spawn_pads: [
      { team_id: 1, position: [18.5, 3] },
      { team_id: 2, position: [1.5, 3] },
    ],
  };
  const records = teamIdentity.records({ transform, scene });
  assert.deepEqual(
    records.map((record) => [record.teamId, record.side]),
    [
      [2, "left"],
      [1, "right"],
    ],
  );
  assert.equal(records[1].d.includes("H"), false);
  const unclear = {
    ...scene,
    spawn_pads: [
      { team_id: 1, position: [2, 3] },
      { team_id: 2, position: [3, 3] },
    ],
  };
  assert.deepEqual(teamIdentity.records({ transform, scene: unclear }), []);
  assert.deepEqual(
    teamIdentity.records({ transform, scene, filters: { home_end_marks: false } }),
    [],
  );
});

test("the POV Home Mark appears only in Agent POV on the POV team's crest", () => {
  const map = mapWith({ team_a_x_range: [0, 3], team_b_x_range: [17, 20] });
  const pov = teamIdentity.records({
    transform,
    map,
    audience: "agent_pov",
    povTeamId: 1,
  });
  assert.deepEqual(
    pov.map((record) => [record.teamId, record.pov]),
    [
      [2, false],
      [1, true],
    ],
  );
  const oracle = teamIdentity.records({
    transform,
    map,
    audience: "researcher",
    povTeamId: 1,
  });
  assert.ok(oracle.every((record) => record.pov === false));
  const nodes = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (teamIdentity.build(pov, doc))
  );
  assertExportSafe(nodes);
  const classes = (/** @type {FakeElement} */ node) =>
    walk(node).map((child) => child.getAttribute("class"));
  assert.ok(
    classes(nodes[0]).includes("home-crest__gap"),
    "Team B crest has the Ink gap ring",
  );
  assert.equal(classes(nodes[1]).includes("home-crest__gap"), false);
  assert.ok(classes(nodes[1]).includes("home-crest__pov"));
  assert.equal(classes(nodes[0]).includes("home-crest__pov"), false);
  const tip = /** @type {Record<string, any>} */ (
    teamIdentity.tooltip(/** @type {any} */ (pov[1]))
  );
  assert.ok(JSON.stringify(tip).includes("Team A's home end."));
  assert.equal(teamIdentity.tooltip(/** @type {any} */ ({ kind: "anything" })), null);
  const moved = teamIdentity.records({
    transform: createViewportTransform({
      worldWidth: 20,
      worldHeight: 10,
      viewportWidth: 960,
      viewportHeight: 600,
      padding: 24,
    }),
    map,
    audience: "agent_pov",
    povTeamId: 1,
  });
  teamIdentity.update(/** @type {any} */ (nodes), moved);
  assert.equal(nodes[1].children[0].getAttribute("d"), moved[1].d);
});
