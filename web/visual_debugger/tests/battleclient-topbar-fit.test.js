/**
 * @file Check the BattleClient chrome that must never cut the map name or
 * cover the scorebug or the rail (owner, 3 Oct 2026).
 *
 * Contracts checked:
 * - placeLineSplit (src/shell.js): the place line's "Map 47" or
 *   "Scenario 1" is the part that always shows with the bold name, and
 *   "Team Deathmatch" is the part that may give way; matchPlaceTitle reads
 *   the name and that first part.
 * - The live top-bar styles (src/styles/battleclient/topbar.css): the place
 *   line is a grid whose second column (span.where__more) alone can shrink,
 *   and a cut second part (data-cut, set when placeRestIsCut in
 *   src/shell.js says its text is wider than its box) hides whole; the line
 *   gives way before the brand, whose product name
 *   is the only brand part that can shrink.
 * - visualFiltersDrawerTop (src/shell.js) and overlays.css: the Visual
 *   Filters drawer starts at the rail's first card (the rail's top plus its
 *   top padding), so it never covers the scorebug; bad input gives null.
 * - createCameraSwitch (src/looks/camera.js), live play: the V tip goes in
 *   the status line and takes the status line message's box (left, top,
 *   width, height), so it never covers the arena, the scorebug, the rail or
 *   a control; overlays.css draws it there.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { createCameraSwitch } from "../src/looks/camera.js";
import {
  matchPlaceParts,
  matchPlaceTitle,
  placeLineSplit,
  placeRestIsCut,
  visualFiltersDrawerTop,
} from "../src/shell.js";

/** @param {string} path */
const css = (path) => readFileSync(new URL(path, import.meta.url), "utf8");

test("the place line keeps the map name and lets Team Deathmatch give way", () => {
  const map = { map: { display_name: "Teutoburg Forest", map_id: 47 }, task_mode: 1 };
  const parts = /** @type {NonNullable<ReturnType<typeof matchPlaceParts>>} */ (
    matchPlaceParts(map)
  );
  assert.deepEqual(placeLineSplit(parts), {
    lead: ["Map 47"],
    more: ["Team Deathmatch"],
  });
  assert.equal(matchPlaceTitle(map), "Teutoburg Forest · Map 47");
  const tale = /** @type {NonNullable<ReturnType<typeof matchPlaceParts>>} */ (
    matchPlaceParts({
      map: { display_name: "x", technical_name: "tdm_scenario_1" },
      task_mode: 1,
    })
  );
  assert.equal(tale.name, "The Last Mending");
  assert.deepEqual(placeLineSplit(tale), {
    lead: ["Scenario 1"],
    more: ["Team Deathmatch"],
  });
  const custom = /** @type {NonNullable<ReturnType<typeof matchPlaceParts>>} */ (
    matchPlaceParts({ map: { display_name: "Custom Map", map_id: null }, task_mode: 0 })
  );
  assert.deepEqual(placeLineSplit(custom), { lead: [], more: [] });
  assert.ok(Object.isFrozen(placeLineSplit(parts).lead));
});

test("the live top bar gives way in order: Team Deathmatch, then the product name", () => {
  const topbar = css("../src/styles/battleclient/topbar.css");
  const rule = (/** @type {string} */ selector) => {
    const start = topbar.indexOf(`${selector} {`);
    assert.ok(start >= 0, `rule ${selector}`);
    return topbar.slice(start, topbar.indexOf("}", start));
  };
  const where = rule('html[data-product-kind="combat_debugger"] .bc .bc-top .where');
  assert.match(where, /display: grid;/u);
  assert.match(where, /grid-template-columns: auto minmax\(0, auto\);/u);
  assert.match(where, /flex: 0 1000000 auto;/u);
  assert.match(where, /min-width: auto;/u);
  assert.match(where, /overflow: visible;/u);
  const more = rule(
    'html[data-product-kind="combat_debugger"] .bc .bc-top .where__more',
  );
  assert.match(more, /text-overflow: ellipsis;/u);
  assert.match(more, /white-space: pre;/u);
  assert.match(
    rule(
      'html[data-product-kind="combat_debugger"] .bc .bc-top .where__more[data-cut]',
    ),
    /visibility: hidden;/u,
  );
  assert.equal(placeRestIsCut({ scrollWidth: 130, clientWidth: 80 }), true);
  assert.equal(placeRestIsCut({ scrollWidth: 130.4, clientWidth: 130 }), false);
  assert.equal(placeRestIsCut({ scrollWidth: 0, clientWidth: 0 }), false);
  assert.match(
    rule('html[data-product-kind="combat_debugger"] .bc .bc-top .where[hidden]'),
    /display: none;/u,
  );
  const brand = rule('html[data-product-kind="combat_debugger"] .bc .bc-top .brand');
  assert.match(brand, /grid-template-columns: auto auto minmax\(0, auto\);/u);
  assert.match(brand, /flex: 0 1 auto;/u);
  assert.match(
    rule('html[data-product-kind="combat_debugger"] .bc .bc-top .brand .bprod'),
    /text-overflow: ellipsis;/u,
  );
  assert.doesNotMatch(topbar, /#camera-tip/u, "the V tip is not in the top bar");
});

test("the Visual Filters drawer starts at the rail's first card", () => {
  assert.equal(visualFiltersDrawerTop(132.5, 12), 145);
  assert.equal(visualFiltersDrawerTop(132, Number.NaN), 132);
  assert.equal(visualFiltersDrawerTop(-40, 12), 0);
  assert.equal(visualFiltersDrawerTop(Number.NaN, 12), null);
  const overlays = css("../src/styles/battleclient/overlays.css");
  assert.match(overlays, /inset: var\(--vf-drawer-top, 144px\) 12px auto auto;/u);
  assert.doesNotMatch(overlays, /inset: 56px/u, "no longer right under the top bar");
});

/** A tiny DOM: the status line with its message, and the live camera slot. */
function liveDocument() {
  /** @type {any} */
  const doc = {
    defaultView: { addEventListener() {}, removeEventListener() {} },
    getElementById: (/** @type {string} */ id) =>
      ({ "status-line": statusLine, msg: message })[id] ?? null,
    createElement: (/** @type {string} */ tag) => node(tag),
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      node(tag),
  };
  /** @param {string} tag */
  function node(tag) {
    /** @type {Map<string, string>} */
    const attributes = new Map();
    /** @type {any} */
    const element = {
      ownerDocument: doc,
      tagName: tag,
      children: /** @type {any[]} */ ([]),
      parentElement: null,
      dataset: {},
      style: {},
      hidden: false,
      textContent: "",
      id: "",
      className: "",
      classList: { add() {}, toggle() {} },
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      addEventListener() {},
      closest: () => null,
      remove() {},
      append: (/** @type {any[]} */ ...nodes) => {
        for (const child of nodes) {
          child.parentElement = element;
          element.children.push(child);
        }
      },
      replaceChildren: (/** @type {any[]} */ ...nodes) => {
        element.children = [];
        element.append(...nodes);
      },
    };
    return element;
  }
  const statusLine = node("div");
  const message = node("p");
  Object.assign(message, {
    offsetLeft: 214,
    offsetTop: 6,
    offsetWidth: 575,
    offsetHeight: 35,
  });
  statusLine.append(message);
  return { doc, statusLine, slot: node("div") };
}

test("the live V tip covers only the status line's message", () => {
  const { statusLine, slot } = liveDocument();
  const camera = createCameraSwitch({
    slot,
    product: "combat_debugger",
    storage: null,
    timers: { setTimeout: () => 0, clearTimeout: () => {} },
    applyFrame: () => {},
    registerKeys: false,
  });
  try {
    const tip = statusLine.children.find(
      (/** @type {any} */ child) => child.id === "camera-tip",
    );
    assert.ok(tip, "the tip sits in the status line, not over the rail");
    assert.equal(tip.hidden, true);
    assert.equal(camera.pressKey("v"), true);
    assert.equal(tip.hidden, false);
    assert.equal(tip.textContent, "Everyone In View on · V: back to Whole Map");
    assert.deepEqual(
      [tip.style.left, tip.style.top, tip.style.width, tip.style.height],
      ["214px", "6px", "575px", "35px"],
    );
  } finally {
    camera.destroy();
  }
  const overlays = css("../src/styles/battleclient/overlays.css");
  assert.match(overlays, /\.bc \.dstatus > #camera-tip \{\n {2}position: absolute;/u);
  assert.match(
    overlays,
    /\.bc \.dstatus > #camera-tip\[hidden\] \{\n {2}display: none;/u,
  );
});
