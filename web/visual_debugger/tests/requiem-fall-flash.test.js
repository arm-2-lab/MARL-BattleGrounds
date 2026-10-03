/**
 * @file Check the class requiem fall flash and the Fallen row spacing of
 * `src/looks/requiems.js` against the V5 Class Requiems spec (spec wins over
 * the store code). Mage: the eight cracks run from 0.1R to 0.62R, appear at
 * 80% at store time 940 and reach full length (scale 1) by 1000, then hold
 * and fade by 1480; the six motes start at 0.5R with radius max(0.8, 0.05R)
 * and scale about the centre to 2.2 while fading from 70% over 940-1440.
 * Warrior: the crack grows top to bottom (scaleY 0 to 1, top origin) over
 * 880-1000 and the two rim halves part by 0.75 px each by 1160. Hunter: the
 * leaves and rime start at 0.55R and scale to 2.1. Rogue: three separate
 * puffs each swell from 0.6 to 1.3 about their own centres. Priest: ray i is
 * lit at the beat and out by 1000 + 60i. Every animation still starts hidden
 * at or after the Fall beat. Fallen row: an ID never runs into the next "+"
 * or into the points, whatever its length.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import * as requiems from "../src/looks/requiems.js";

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
      dataset: {},
      textContent: "",
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      hasAttribute: (/** @type {string} */ name) => attributes.has(name),
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
      addEventListener: () => {},
      append: (/** @type {any[]} */ ...nodes) => {
        for (const child of nodes) {
          child.parentElement = element;
          element.children.push(child);
        }
      },
    };
    return element;
  };
  return {
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      node(tag),
  };
}

/** @param {any} element @returns {any[]} */
function descendants(element) {
  return element.children.flatMap((/** @type {any} */ child) => [
    child,
    ...descendants(child),
  ]);
}

/** @param {any} element @param {string} className @returns {any[]} */
function byClass(element, className) {
  return descendants(element).filter((child) =>
    (child.getAttribute("class") ?? "").split(" ").includes(className),
  );
}

const RADIUS = 20;
const BEAT = 900;

/** @param {number} classId */
function playedFall(classId) {
  const doc = fakeDocument();
  /** @type {any} */
  const mount = {
    group: doc.createElementNS("", "g"),
    connector: doc.createElementNS("", "g"),
    underlay: null,
    plan: { epochKey: "epoch", phases: { total: 1500, reducedTotal: 375 } },
    options: { motionMode: "normal", settled: false },
    animationSpecs: [],
  };
  const event = {
    kind: "semantic_pulse",
    cueSemantic: "agent_died",
    eventId: `death-${classId}`,
    anchor: { x: 100, y: 80 },
    anchorRadius: RADIUS,
    phaseStart: BEAT,
    phaseEnd: 2400,
    recipientIdentity: { class_id: classId, team_id: 2, public_agent_id: "7" },
  };
  requiems.build(requiems.records({ event }), /** @type {any} */ (doc), mount);
  /** @param {string} part @returns {any} */
  const spec = (part) => {
    const found = mount.animationSpecs.find((/** @type {any} */ each) =>
      each.id.endsWith(`:${part}`),
    );
    assert.ok(found, `${part} animates`);
    return found;
  };
  return { mount, spec };
}

/**
 * @param {any} spec
 * @param {number} storeMs
 */
function frameAt(spec, storeMs) {
  const offset = (storeMs - 880 - (spec.options.delay - BEAT)) / spec.options.duration;
  const frame = spec.keyframes.find(
    (/** @type {any} */ each) => Math.abs(each.offset - offset) < 1e-6,
  );
  assert.ok(frame, `${spec.id} has a keyframe at ${storeMs}`);
  return frame;
}

test("every requiem animation starts hidden at or after the Fall beat", () => {
  for (const classId of [1, 2, 3, 4, 5]) {
    const { mount } = playedFall(classId);
    for (const each of mount.animationSpecs) {
      assert.ok(each.options.delay >= BEAT, each.id);
      assert.equal(each.keyframes[0].opacity, 0, each.id);
      assert.equal(each.options.fill, "backwards", each.id);
    }
    assert.ok(mount.animationSpecs.length <= requiems.BUDGET.animationsPerRecord);
  }
});

test("the Mage's cracks reach full length by store time 1000 and its motes drift to 2.2", () => {
  const { mount, spec } = playedFall(1);
  const cracks = spec("requiem-cracks");
  assert.deepEqual(
    [frameAt(cracks, 939).opacity, frameAt(cracks, 940).opacity],
    [0, 0.8],
  );
  assert.equal(frameAt(cracks, 1000).transform, "scale(1)");
  assert.equal(frameAt(cracks, 1000).opacity, 0.8);
  assert.equal(frameAt(cracks, 1180).opacity, 0.8);
  assert.equal(frameAt(cracks, 1480).opacity, 0);
  const crackPath = byClass(mount.group, "requiem__cracks")[0].getAttribute("d");
  const lengths = [
    ...crackPath.matchAll(/M(-?[\d.]+) (-?[\d.]+)L(-?[\d.]+) (-?[\d.]+)/gu),
  ]
    .map((match) => match.slice(1).map(Number))
    .map(([x0, y0, x1, y1]) => [Math.hypot(x0, y0), Math.hypot(x1, y1)]);
  assert.equal(lengths.length, 8);
  for (const [inner, outer] of lengths) {
    assert.ok(Math.abs(inner - 0.1 * RADIUS) < 0.02);
    assert.ok(Math.abs(outer - 0.62 * RADIUS) < 0.02);
  }
  const motes = spec("requiem-motes");
  assert.equal(frameAt(motes, 940).opacity, 0.7);
  assert.equal(frameAt(motes, 940).transform, "scale(1)");
  assert.equal(frameAt(motes, 1440).transform, "scale(2.2)");
  assert.equal(frameAt(motes, 1440).opacity, 0);
  const motePath = byClass(mount.group, "requiem__motes")[0].getAttribute("d");
  const [first] = [...motePath.matchAll(/M(-?[\d.]+) (-?[\d.]+)a([\d.]+)/gu)];
  const r = Number(first[3]);
  const centreX = Number(first[1]) + r;
  assert.equal(r, Math.max(0.8, 0.05 * RADIUS));
  assert.ok(Math.abs(Math.hypot(centreX, Number(first[2])) - 0.5 * RADIUS) < 0.02);
});

test("the Warrior's crack grows from the top and its rim halves part", async () => {
  const { mount, spec } = playedFall(2);
  const crack = spec("requiem-crack");
  assert.equal(frameAt(crack, 881).transform, "scale(1, 0)");
  assert.equal(frameAt(crack, 1000).transform, "scale(1, 1)");
  assert.equal(frameAt(crack, 1400).opacity, 0);
  assert.equal(
    frameAt(spec("requiem-rim-right"), 1160).transform,
    "translate(0.75px, 0px)",
  );
  assert.equal(
    frameAt(spec("requiem-rim-left"), 1160).transform,
    "translate(-0.75px, 0px)",
  );
  assert.equal(frameAt(spec("requiem-shield"), 1120).transform, "scale(0.97)");
  assert.equal(byClass(mount.group, "requiem__rim").length, 2);
  const css = await readFile(
    new URL("../src/styles/requiems.css", import.meta.url),
    "utf8",
  );
  assert.match(css, /\.requiem__crack \{[^}]*transform-origin: 50% 0;/u);
  assert.match(css, /\.requiem__puffs \{[^}]*transform-origin: center;/u);
});

test("Hunter particles drift to 2.1, Rogue puffs swell one by one and Priest rays go out in turn", () => {
  const hunter = playedFall(3);
  for (const part of ["requiem-leaves", "requiem-rime"]) {
    assert.equal(frameAt(hunter.spec(part), 1440).transform, "scale(2.1)");
    assert.equal(frameAt(hunter.spec(part), 1440).opacity, 0);
  }
  const rogue = playedFall(4);
  assert.equal(byClass(rogue.mount.group, "requiem__puff").length, 3);
  for (const index of [0, 1, 2]) {
    const puff = rogue.spec(`requiem-puff-${index}`);
    assert.equal(frameAt(puff, 900).transform, "scale(0.6)");
    assert.equal(frameAt(puff, 1300).transform, "scale(1.3)");
  }
  const priest = playedFall(5);
  for (const index of [0, 1, 2, 3, 4, 5, 6, 7]) {
    const ray = priest.spec(`requiem-ray-${index}`);
    assert.equal(frameAt(ray, 885).opacity, 1);
    assert.equal(frameAt(ray, 1000 + 60 * index).opacity, 0);
  }
});

test("a Fallen row never runs an ID into a plus sign or the points", () => {
  const killers = ["10", "1", "7"].map((id, index) => ({
    publicAgentId: id,
    title: `Mage · ID ${id}`,
    classId: index + 1,
    teamId: 2,
  }));
  const [record] = requiems.fallenFeedRecords({
    tick: 92,
    rows: [
      {
        tick: 92,
        victim: { publicAgentId: "15", title: "Mage · ID 15", classId: 1, teamId: 1 },
        killers,
        killingTeamId: 2,
        points: 1,
        ultimate: null,
      },
    ],
    displayIdOf: (/** @type {string} */ id) => id,
  });
  const row = requiems.buildFallenFeed(/** @type {any} */ (fakeDocument()), [record]);
  const texts = descendants(row).filter((node) => node.tagName === "text");
  const plusHalfWidth = 3.6;
  let checked = 0;
  for (const [index, text] of texts.entries()) {
    if (!(text.getAttribute("class") ?? "").includes("fallen-feed__id")) continue;
    const end =
      Number(text.getAttribute("x")) +
      text.textContent.length * requiems.FEED_ID_ADVANCE_PX;
    const next = texts[index + 1];
    if (next === undefined) continue;
    const nextClass = next.getAttribute("class") ?? "";
    const nextLeft = nextClass.includes("fallen-feed__plus")
      ? Number(next.getAttribute("x")) - plusHalfWidth
      : Number(next.getAttribute("x"));
    assert.ok(nextLeft - end >= 1.5, `${text.textContent} clears ${next.textContent}`);
    checked += 1;
  }
  assert.equal(checked, 4);
});
