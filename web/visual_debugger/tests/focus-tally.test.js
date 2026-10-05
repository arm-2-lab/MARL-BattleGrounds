/**
 * @file Check the Focus Tally look (`src/looks/focus-tally.js`).
 *
 * Contracts: beads are 20 degree arcs with 7 degree gaps centred on 6 o'clock
 * (N = 2 gives 156.5-176.5 and 183.5-203.5), 2 to 5 of them on a ring of 1.5R;
 * the detail ladder sets 3, 2.5 or 2 px beads and drops the thread below R 14
 * px; the thread runs 8 degrees past the first and last bead; records answer
 * only a painted `focus_tally` plan row with an anchor; the beads take the
 * attacking team's colour key; build draws an under-copy, thread, beads and an
 * 8 px hit arc over the thread span (so the pointer can open the tip) in the
 * painter's group with export-safe SVG, and update moves every path; the tooltip calls it a plain count
 * of visible hits. The `focus_fire` switch (off by default) owns the plan's
 * `focus_tally` paint part, so turning it on paints the rows the plan makes.
 * Agent POV: on the fixture recordings the look draws nothing that the plan
 * did not admit for that audience (it reads only the admitted row).
 */
import assert from "node:assert/strict";
import test from "node:test";

import { DERIVED_EVENT_PAINT_PARTS } from "../src/choreography-plan.js";
import * as tally from "../src/looks/focus-tally.js";
import {
  classifyVisualPaintPart,
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualPaintPartEnabled,
} from "../src/visual-filters.js";

/** @returns {any} */
function fakeDocument() {
  /** @param {string} tag @returns {any} */
  const node = (tag) => {
    /** @type {Map<string, string>} */
    const attributes = new Map();
    /** @type {any} */
    const element = {
      tagName: tag,
      children: [],
      parentElement: null,
      textContent: "",
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      append: (/** @type {any[]} */ ...nodes) => {
        for (const child of nodes) {
          child.parentElement = element;
          element.children.push(child);
        }
      },
      querySelectorAll: (/** @type {string} */ selector) => {
        const name = selector.replace(/^g\./u, "");
        /** @param {any} parent @returns {any[]} */
        const walk = (parent) =>
          parent.children.flatMap((/** @type {any} */ child) => [
            child,
            ...walk(child),
          ]);
        return walk(element).filter((child) =>
          (child.getAttribute("class") ?? "").split(" ").includes(name),
        );
      },
      querySelector: (/** @type {string} */ selector) =>
        element.querySelectorAll(selector.replace(/^\./u, ""))[0] ?? null,
      attributes,
    };
    return element;
  };
  return {
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      node(tag),
  };
}

/** @param {any} element @returns {string} */
function serialize(element) {
  const attributes = [...element.attributes.entries()]
    .map(([name, value]) => ` ${name}="${value}"`)
    .join("");
  return `<${element.tagName}${attributes}>${element.textContent}${element.children.map(serialize).join("")}</${element.tagName}>`;
}

/** @param {Record<string, any>} [extra] */
function row(extra = {}) {
  return {
    eventId: "t:focus:agent-slot-5",
    kind: "focus_tally",
    targetPublicAgentId: "agent-slot-5",
    sourcePublicAgentIds: ["agent-slot-1", "agent-slot-2"],
    attackingTeamId: 1,
    count: 2,
    lethal: false,
    anchor: { x: 244.1, y: 177.8 },
    anchorRadius: 23,
    paintParts: { effect: true },
    persistent: true,
    ...extra,
  };
}

test("bead angles are centred on 6 o'clock with 7 degree gaps", () => {
  assert.deepEqual(tally.beadAngles(2), [
    [156.5, 176.5],
    [183.5, 203.5],
  ]);
  assert.deepEqual(tally.beadAngles(3), [
    [143, 163],
    [170, 190],
    [197, 217],
  ]);
  assert.deepEqual(tally.beadAngles(5)[0], [116, 136]);
  assert.equal(tally.beadAngles(9).length, 5);
  assert.equal(tally.beadAngles(1).length, 2);
});

test("the detail ladder sets bead widths and drops the thread on small bodies", () => {
  assert.deepEqual(tally.detailLadder(23), { beadWidth: 3, thread: true });
  assert.deepEqual(tally.detailLadder(16), { beadWidth: 2.5, thread: true });
  assert.deepEqual(tally.detailLadder(12), { beadWidth: 2, thread: false });
  const [small] = tally.records({ event: row({ anchorRadius: 12 }) });
  assert.equal(small.threadPath, null);
  assert.equal(small.beadWidth, 2);
});

test("records answer only painted focus_tally rows with an anchor", () => {
  const [record] = tally.records({ event: row() });
  assert.equal(record.kind, "focus-tally");
  assert.equal(record.ringRadius, 34.5);
  assert.equal(record.count, 2);
  assert.equal(record.teamKey, "team-a");
  // The mock recipe's first bead starts at about (257.9, 209.5) on r = 34.5.
  const [startX, startY] = record.beadsPath
    .split(" ")
    .slice(1, 3)
    .map((/** @type {string} */ value) => Number(value));
  assert.ok(Math.abs(startX - 257.9) < 0.1 && Math.abs(startY - 209.5) < 0.1);
  assert.match(record.beadsPath, / A 34\.5 34\.5 0 0 1 /u);
  assert.equal(record.beadsPath.match(/M /gu)?.length, 2);
  assert.match(String(record.threadPath), /^M /u);
  assert.equal(
    tally.records({ event: row({ attackingTeamId: 2 }) })[0].teamKey,
    "team-b",
  );
  assert.deepEqual(tally.records({ event: row({ kind: "last_seen" }) }), []);
  assert.deepEqual(tally.records({ event: row({ anchor: null }) }), []);
  assert.deepEqual(
    tally.records({ event: row({ paintParts: { effect: false } }) }),
    [],
  );
  assert.deepEqual(tally.records({}), []);
  assert.equal(
    tally.records({ event: row({ anchorRadius: undefined }) })[0].radius,
    tally.DEFAULT_BODY_RADIUS_PX,
  );
});

test("build draws an export-safe under-copy, thread and beads in the event group", () => {
  const doc = fakeDocument();
  const records = tally.records({ event: row({ count: 3, lethal: true }) });
  const mount = {
    group: doc.createElementNS("", "g"),
    connector: doc.createElementNS("", "g"),
    underlay: null,
    plan: { epochKey: "e", phases: {} },
    options: { motionMode: "off", settled: true },
    animationSpecs: [],
  };
  const nodes = tally.build(records, doc, /** @type {any} */ (mount));
  assert.deepEqual(nodes, [mount.group, mount.connector]);
  const [group] = mount.group.children;
  assert.equal(group.getAttribute("class"), "focus-tally");
  assert.equal(group.getAttribute("data-team"), "team-a");
  assert.equal(group.getAttribute("data-count"), "3");
  assert.equal(group.getAttribute("data-lethal"), "true");
  assert.deepEqual(
    group.children.map((/** @type {any} */ child) => child.getAttribute("class")),
    [
      "focus-tally__under",
      "focus-tally__thread",
      "focus-tally__beads",
      "focus-tally__hit",
    ],
  );
  assert.equal(group.children[0].getAttribute("stroke-width"), "5.5");
  assert.equal(group.children[2].getAttribute("stroke-width"), "3");
  assert.equal(group.children[3].getAttribute("stroke-width"), "8");
  assert.equal(group.children[3].getAttribute("d"), records[0].hitPath);
  assert.equal(records[0].hitPath, records[0].threadPath);
  assert.doesNotMatch(
    serialize(mount.group),
    /url\(|gradient|<mask|<filter|<image|<foreignObject| id="/iu,
  );
  assert.equal(mount.animationSpecs.length, 0);
  const moved = tally.records({
    event: row({ count: 3, anchor: { x: 10, y: 10 } }),
  });
  tally.update(nodes, moved);
  assert.equal(group.children[2].getAttribute("d"), moved[0].beadsPath);
  assert.equal(group.children[3].getAttribute("d"), moved[0].hitPath);
  assert.deepEqual(tally.build([], doc), []);
});

test("the tooltip calls the tally a plain count of visible hits", () => {
  const descriptor = /** @type {any} */ (
    tally.tooltip(tally.records({ event: row() })[0])
  );
  assert.equal(descriptor.title, "Focus Tally · 2 Attackers");
  assert.match(descriptor.summary, /Team A/u);
  assert.match(descriptor.summary, /not the Focus Fire Concentration metric/u);
  assert.match(descriptor.summary, /Only visible blows are counted/u);
  assert.equal(tally.tooltip({ kind: "anything" }), null);
});

test("the focus_fire switch owns the plan's focus_tally paint part, off by default", () => {
  const part = DERIVED_EVENT_PAINT_PARTS.focus_tally.part;
  assert.deepEqual(
    tally.PAINT_PARTS.map((entry) => ({ ...entry.part })),
    [{ ...part }],
  );
  assert.equal(classifyVisualPaintPart(part), "focus_fire");
  assert.equal(isVisualPaintPartEnabled(DEFAULT_VISUAL_FILTER_STATE, part), false);
  assert.equal(
    isVisualPaintPartEnabled(
      { ...DEFAULT_VISUAL_FILTER_STATE, focus_fire: true },
      part,
    ),
    true,
  );
});
