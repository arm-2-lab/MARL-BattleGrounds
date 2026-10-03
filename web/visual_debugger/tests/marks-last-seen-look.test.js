/**
 * @file Check the Last Seen mark's look against the store pick
 * (src/looks/fog.js; store skins/fog-and-vision.js lastSeenMarks and
 * hollowEmblem; spec "dotted Host-Light ring at 45%, hollow Moon Silver
 * emblem at 60%"): the ring is 0.93R at opacity 0.45 and carries the lost
 * champion's team (`data-team` 1 or 2, which styles/fog.css paints in that
 * team's light; "none" when the row names no team); the ghost emblem is the
 * store's hollow line art for the class (the store's own circle and cross
 * shapes for the Priest, never the filled sigil), unfilled in currentColor,
 * 0.6R square, centered on the spot, at opacity 0.6 with a 1 CSS px stroke;
 * below a 14 px body radius only the ring is drawn; update() moves and
 * resizes the emblem with the ring.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as fog from "../src/looks/fog.js";

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
        all(element).filter((candidate) =>
          (candidate.getAttribute("class") ?? "")
            .split(" ")
            .includes(selector.slice(1)),
        ),
      querySelector: (/** @type {string} */ selector) =>
        element.querySelectorAll(selector)[0] ?? null,
    };
    return element;
  }
  return {
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      node(tag),
  };
}

/** @param {any} element @returns {any[]} */
function all(element) {
  return element.children.flatMap((/** @type {any} */ child) => [child, ...all(child)]);
}

const transform = createViewportTransform({
  worldWidth: 20,
  worldHeight: 10,
  viewportWidth: 908,
  viewportHeight: 468,
  padding: 14,
});

/** @param {number | null} teamId */
function row(teamId) {
  return {
    kind: "last_seen",
    eventId: "t1:last-seen:b2",
    presentationKey: "pov_b2",
    publicAgentId: "b2",
    classId: 5,
    teamId,
    anchor: { x: 300, y: 200 },
  };
}

test("the ring is the lost champion's team light at 45%, the emblem hollow", () => {
  const doc = fakeDocument();
  const bodyR = transform.worldLengthToScreen(0.5);
  for (const [teamId, expected] of /** @type {const} */ ([
    [2, "2"],
    [1, "1"],
    [null, "none"],
  ])) {
    const records = fog.records({ event: row(teamId), transform });
    const [group] = fog.build(records, /** @type {any} */ (doc));
    const node = /** @type {any} */ (group);
    assert.equal(node.getAttribute("data-team"), expected);
    const ring = node.querySelector(".fog-last-seen-ring");
    assert.equal(ring.getAttribute("opacity"), "0.45");
    assert.ok(Math.abs(Number(ring.getAttribute("r")) - 0.93 * bodyR) < 0.01);
    const emblem = node.querySelector(".fog-last-seen-emblem");
    assert.ok(emblem, "a ghost emblem from a 14 px body radius up");
    assert.equal(emblem.getAttribute("data-emblem"), "priest");
    assert.equal(emblem.getAttribute("fill"), "none");
    assert.equal(emblem.getAttribute("stroke"), "currentColor");
    assert.equal(emblem.getAttribute("viewBox"), "0 0 24 24");
    assert.equal(emblem.getAttribute("opacity"), "0.6");
    const size = Number(emblem.getAttribute("width"));
    assert.ok(Math.abs(size - 0.6 * bodyR) < 0.01);
    assert.equal(Number(emblem.getAttribute("height")), size);
    assert.ok(Math.abs(Number(emblem.getAttribute("x")) - (300 - size / 2)) < 0.01);
    // 24 viewBox units across `size` px: a 1 px stroke is 24 / size units.
    assert.ok(
      Math.abs(Number(emblem.getAttribute("stroke-width")) * (size / 24) - 1) < 0.01,
    );
    // The store's Priest: the sun circle (r 8.2) and one path of cross and rays,
    // with no fill of their own.
    const parts = all(emblem);
    assert.deepEqual(
      parts.map((/** @type {any} */ part) => part.tagName),
      ["circle", "path"],
    );
    assert.equal(parts[0].getAttribute("r"), "8.2");
    assert.match(parts[1].getAttribute("d"), /^M10\.4 6\.4H13\.6V10\.4H17\.6/u);
    assert.ok(
      parts.every((/** @type {any} */ part) => part.getAttribute("fill") === null),
    );
  }
});

test("a small body keeps the ring only, and update moves and resizes the emblem", () => {
  const doc = fakeDocument();
  const small = createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: 400,
    viewportHeight: 210,
    padding: 5,
  });
  const [tiny] = fog.records({ event: row(2), transform: small });
  assert.equal(tiny.emblemSize, 0);
  const [group] = fog.build([tiny], /** @type {any} */ (doc));
  assert.equal(/** @type {any} */ (group).querySelector(".fog-last-seen-emblem"), null);
  const [first] = fog.records({ event: row(2), transform });
  const [drawn] = fog.build([first], /** @type {any} */ (doc));
  const bigger = createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: 1255,
    viewportHeight: 647,
    padding: 19,
  });
  const moved = fog.records({
    event: { ...row(2), anchor: { x: 500, y: 250 } },
    transform: bigger,
  });
  fog.update([drawn], moved);
  const emblem = /** @type {any} */ (drawn).querySelector(".fog-last-seen-emblem");
  assert.equal(Number(emblem.getAttribute("width")), moved[0].emblemSize);
  assert.ok(
    Math.abs(Number(emblem.getAttribute("x")) - (500 - moved[0].emblemSize / 2)) < 0.01,
  );
});
