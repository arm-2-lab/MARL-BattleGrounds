/**
 * @file Check the Kill Credit Glint switch (`kill_credit_glint`) of the Class
 * Requiems look (`src/looks/requiems.js`). records() gives a fall no glint
 * with the switch off (the default), with Death Effects off, or without the
 * audience's plan and projection, and records({event}) alone keeps naming the
 * fall. Switched on, every killer of the victim's own fallen_feed_row row in
 * that plan that stands alive in the plan's END bodies gets one glint at its
 * projected END center, in the plan's public ID order, at most 4 per fall.
 * Agent POV: a killer the team did not see at END (not in the POV bodies), one
 * that left sight this tick (lastSeen) or one that fell is never marked, and a
 * killer the plan does not name gets nothing. build() puts the glints in the
 * Strike Band group under the bodies, at END opacity 0, as rings 1.06 times
 * the body radius and 1.25 px wide in the killer's class color, with one fill
 * "backwards" animation each from the Fall beat (70% fading to 0 over 240 ms
 * while growing from 0.95 to 1; reduced motion: the same fade over 200 ms with
 * no growth), and draws none in a settled still or with motion off. A played
 * Priest fall with four glints stays inside the look's BUDGET (nodes, kept
 * nodes and animations), the drawing is export-safe, update() moves each glint
 * to its new END center, the switch owns the registered kill_credit_glint
 * paint part, and the stylesheet colors every class from tokens.
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
      addEventListener: () => {},
      querySelector: (/** @type {string} */ selector) =>
        element.querySelectorAll(selector)[0] ?? null,
      querySelectorAll: (/** @type {string} */ selector) =>
        descendants(element).filter((/** @type {any} */ child) =>
          (child.getAttribute("class") ?? "").split(" ").includes(selector.slice(1)),
        ),
      append: (/** @type {any[]} */ ...nodes) => {
        for (const child of nodes) {
          child.parentElement = element;
          element.children.push(child);
        }
      },
      attributes,
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

/** @param {any} element @returns {string} */
function serialize(element) {
  const attributes = [...element.attributes.entries()]
    .map(([name, value]) => ` ${name}="${value}"`)
    .join("");
  return `<${element.tagName}${attributes}>${element.textContent}${element.children.map(serialize).join("")}</${element.tagName}>`;
}

/**
 * @param {{settled?: boolean, motionMode?: "normal" | "reduced" | "off"}} [options]
 * @returns {any}
 */
function mountFor(options = {}) {
  const doc = fakeDocument();
  return {
    doc,
    mount: {
      group: doc.createElementNS("", "g"),
      connector: doc.createElementNS("", "g"),
      underlay: null,
      plan: {
        epochKey: "epoch",
        phases: { total: 1500, reducedTotal: 375 },
        beats: { reducedScale: 0.25 },
      },
      options: {
        motionMode: options.motionMode ?? "normal",
        settled: options.settled ?? false,
      },
      animationSpecs: [],
    },
  };
}

/** @type {any} */
const transform = {
  worldToScreen: (/** @type {any} */ point) => ({ x: point[0] * 20, y: point[1] * 20 }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 20,
};

const ON = Object.freeze({ kill_credit_glint: true, death_effects: true });

/** @param {Record<string, any>} [extra] */
function fallPulse(extra = {}) {
  return {
    kind: "semantic_pulse",
    cueSemantic: "agent_died",
    eventId: "fall-7",
    anchor: { x: 100, y: 80 },
    anchorRadius: 16,
    phaseStart: 900,
    phaseEnd: 2400,
    recipientPublicAgentId: "7",
    recipientIdentity: { class_id: 5, team_id: 2, public_agent_id: "7" },
    ...extra,
  };
}

/**
 * @param {string} id
 * @param {number} classId
 * @param {Record<string, any>} [extra]
 */
function killer(id, classId, extra = {}) {
  return {
    publicAgentId: id,
    classId,
    teamId: 1,
    title: `Champion · ID ${id}`,
    presentationKey: `pk-${id}`,
    lastSeen: false,
    ...extra,
  };
}

/**
 * @param {string} id
 * @param {[number, number]} end
 * @param {boolean} [alive]
 */
function body(id, end, alive = true) {
  return { presentationKey: `pk-${id}`, publicAgentId: id, teamId: 1, alive, end };
}

/**
 * @param {any[]} killers
 * @param {any[]} bodies
 * @param {string} [victimId]
 */
function planWith(killers, bodies, victimId = "7") {
  return {
    epochKey: "epoch",
    derivedEvents: [
      {
        kind: "fallen_feed_row",
        eventId: `t:fallen:${victimId}`,
        spatial: false,
        victim: { publicAgentId: victimId, classId: 5, teamId: 2 },
        killers,
      },
    ],
    bodies: [...bodies, body(victimId, [6, 6], false)],
  };
}

const oraclePlan = planWith(
  [killer("0", 1), killer("2", 3), killer("3", 4)],
  [body("0", [1, 2]), body("2", [3, 4]), body("3", [5, 1])],
);

test("the glint is off by default and needs Death Effects, the plan and the projection", () => {
  for (const input of /** @type {any[]} */ ([
    { event: fallPulse() },
    { event: fallPulse(), plan: oraclePlan, transform, filters: {} },
    {
      event: fallPulse(),
      plan: oraclePlan,
      transform,
      filters: { kill_credit_glint: false },
    },
    {
      event: fallPulse(),
      plan: oraclePlan,
      transform,
      filters: { kill_credit_glint: true, death_effects: false },
    },
    { event: fallPulse(), transform, filters: ON },
    { event: fallPulse(), plan: oraclePlan, filters: ON },
  ])) {
    const records = requiems.records(input);
    assert.equal(records.length, 1);
    assert.equal(records[0].kind, "fall");
    assert.deepEqual(records[0].glints, []);
  }
});

test("switched on, every living named killer glints at its END center, at most 4", () => {
  const [fall] = requiems.records({
    event: fallPulse(),
    plan: oraclePlan,
    transform,
    filters: ON,
  });
  assert.ok(Object.isFrozen(fall.glints));
  assert.deepEqual(
    fall.glints.map((/** @type {any} */ glint) => [
      glint.publicAgentId,
      glint.classKey,
      glint.center.x,
      glint.center.y,
    ]),
    [
      ["0", "mage", 20, 40],
      ["2", "hunter", 60, 80],
      ["3", "rogue", 100, 20],
    ],
  );
  const five = planWith(
    ["0", "1", "2", "3", "4"].map((id) => killer(id, 2)),
    ["0", "1", "2", "3", "4"].map((id) => body(id, [1, 1])),
  );
  const [busy] = requiems.records({
    event: fallPulse(),
    plan: five,
    transform,
    filters: ON,
  });
  assert.equal(requiems.GLINTS_PER_FALL, 4);
  assert.deepEqual(
    busy.glints.map((/** @type {any} */ glint) => glint.publicAgentId),
    ["0", "1", "2", "3"],
  );
  // A return never carries glints, and another victim's row gives none.
  const [homecoming] = requiems.records({
    event: { ...fallPulse(), cueSemantic: "agent_respawned" },
    plan: oraclePlan,
    transform,
    filters: ON,
  });
  assert.equal(homecoming.glints, undefined);
  const [other] = requiems.records({
    event: fallPulse({ recipientPublicAgentId: "8" }),
    plan: oraclePlan,
    transform,
    filters: ON,
  });
  assert.deepEqual(other.glints, []);
});

test("Agent POV: an unseen, departed or fallen killer and an unnamed striker get no glint", () => {
  // The POV plan names killers whose blows the team saw; its bodies are only
  // the champions the team sees at END. Champion 4 struck but was not seen
  // striking, so the plan does not name it, although it stands in view.
  const povPlan = planWith(
    [
      killer("0", 1),
      killer("1", 2, { lastSeen: true }),
      killer("2", 3),
      killer("3", 4),
    ],
    [body("0", [1, 2]), body("3", [5, 1], false), body("4", [2, 2])],
  );
  const [fall] = requiems.records({
    event: fallPulse(),
    plan: povPlan,
    transform,
    filters: ON,
    audience: "agent_pov",
  });
  assert.deepEqual(
    fall.glints.map((/** @type {any} */ glint) => glint.publicAgentId),
    ["0"],
  );
  const { doc, mount } = mountFor();
  requiems.build([fall], doc, mount);
  const glints = byClass(mount.connector, "requiem-glint");
  assert.deepEqual(
    glints.map((node) => node.getAttribute("data-glint-agent")),
    ["0"],
  );
  assert.doesNotMatch(serialize(mount.connector), /data-glint-agent="[1234]"/u);
});

test("a played fall draws its glints under the bodies at END opacity 0 with beat animations", () => {
  const [fall] = requiems.records({
    event: fallPulse(),
    plan: oraclePlan,
    transform,
    filters: ON,
  });
  const { doc, mount } = mountFor();
  requiems.build([fall], doc, mount);
  const glints = byClass(mount.connector, "requiem-glint");
  assert.equal(glints.length, 3);
  assert.equal(byClass(mount.group, "requiem-glint").length, 0);
  const [first] = glints;
  assert.equal(first.getAttribute("opacity"), "0");
  assert.equal(first.getAttribute("stroke-width"), "1.25");
  assert.ok(first.getAttribute("class").split(" ").includes("requiem-glint--mage"));
  // A ring of radius 1.06 x 16 = 16.96 px around (20, 40).
  assert.equal(
    first.getAttribute("d"),
    "M3.04 40a16.96 16.96 0 1 0 33.92 0a16.96 16.96 0 1 0 -33.92 0",
  );
  const specs = mount.animationSpecs.filter((/** @type {any} */ spec) =>
    /kill-credit-glint-/u.test(spec.id),
  );
  assert.equal(specs.length, 3);
  for (const spec of specs) {
    assert.equal(spec.options.fill, "backwards");
    assert.equal(spec.options.delay, 900);
    assert.equal(spec.options.duration, 240);
    const frames = spec.keyframes;
    assert.deepEqual(frames[0], { offset: 0, opacity: 0, transform: "scale(0.95)" });
    assert.equal(frames[1].opacity, 0.7);
    assert.deepEqual(frames.at(-1), {
      offset: 1,
      opacity: 0,
      transform: "scale(1.0000)",
    });
    for (let index = 2; index < frames.length; index += 1) {
      assert.ok(frames[index].opacity < frames[index - 1].opacity);
    }
  }
  const reduced = mountFor({ motionMode: "reduced" });
  requiems.build([fall], reduced.doc, reduced.mount);
  const slow = reduced.mount.animationSpecs.filter((/** @type {any} */ spec) =>
    /kill-credit-glint-/u.test(spec.id),
  );
  assert.equal(slow.length, 3);
  for (const spec of slow) {
    assert.equal(spec.options.delay, 225);
    assert.equal(spec.options.duration, 200);
    assert.ok(spec.keyframes.every((/** @type {any} */ frame) => !frame.transform));
    assert.equal(spec.keyframes[0].opacity, 0);
  }
});

test("a paused still and motion off draw no glint", () => {
  const [fall] = requiems.records({
    event: fallPulse(),
    plan: oraclePlan,
    transform,
    filters: ON,
  });
  for (const options of [
    { settled: true },
    /** @type {{motionMode: "off"}} */ ({ motionMode: "off" }),
  ]) {
    const { doc, mount } = mountFor(options);
    requiems.build([fall], doc, mount);
    assert.equal(byClass(mount.connector, "requiem-glint").length, 0);
    assert.equal(mount.animationSpecs.length, 0);
  }
});

test("a played fall with four glints stays inside the look's budget and is export-safe", () => {
  const killers = ["0", "1", "2", "3", "4"].map((id) => killer(id, 4));
  const bodies = ["0", "1", "2", "3", "4"].map((id) => body(id, [1, 1]));
  for (const classId of [1, 2, 3, 4, 5]) {
    const [fall] = requiems.records({
      event: fallPulse({
        homeFall: true,
        recipientIdentity: { class_id: classId, team_id: 2, public_agent_id: "7" },
      }),
      plan: planWith(killers, bodies),
      transform,
      filters: ON,
    });
    assert.equal(fall.glints.length, 4);
    const { doc, mount } = mountFor();
    requiems.build([fall], doc, mount);
    // The painter counts each group and everything inside it.
    const kept =
      descendants(mount.group).length + 1 + descendants(mount.connector).length + 1;
    assert.ok(
      kept <= requiems.BUDGET.persistentNodesPerRecord,
      `class ${classId}: ${kept}`,
    );
    assert.ok(kept <= requiems.BUDGET.nodesPerRecord, `class ${classId}`);
    assert.ok(
      mount.animationSpecs.length <= requiems.BUDGET.animationsPerRecord,
      `class ${classId}: ${mount.animationSpecs.length}`,
    );
    const text = serialize(mount.connector) + serialize(mount.group);
    assert.doesNotMatch(text, /url\(|gradient|<mask|<filter|<image|<foreignObject/iu);
    assert.doesNotMatch(text, / id="/u);
  }
});

test("update moves each glint to its new END center", () => {
  const [fall] = requiems.records({
    event: fallPulse(),
    plan: oraclePlan,
    transform,
    filters: ON,
  });
  const { doc, mount } = mountFor();
  requiems.build([fall], doc, mount);
  /** @type {any} */
  const wide = {
    worldToScreen: (/** @type {any} */ point) => ({
      x: point[0] * 40,
      y: point[1] * 40,
    }),
    worldLengthToScreen: (/** @type {number} */ length) => length * 40,
  };
  const moved = requiems.records({
    event: fallPulse({ anchor: { x: 200, y: 160 } }),
    plan: oraclePlan,
    transform: wide,
    filters: ON,
  });
  requiems.update([mount.group, mount.connector], moved);
  const [first] = byClass(mount.connector, "requiem-glint");
  assert.match(first.getAttribute("d"), /^M23\.04 80a16\.96/u);
});

test("the switch owns its paint part and the stylesheet colors every class from tokens", async () => {
  assert.ok(
    requiems.PAINT_PARTS.some(
      (registration) =>
        registration.part.surface === "transient" &&
        registration.part.kind === "kill_credit_glint" &&
        registration.filter === "kill_credit_glint",
    ),
  );
  const css = await readFile(
    new URL("../src/styles/requiems.css", import.meta.url),
    "utf8",
  );
  for (const classKey of ["mage", "warrior", "hunter", "rogue", "priest"]) {
    assert.match(
      css,
      new RegExp(
        `\\.requiem-glint--${classKey} \\{\\s*stroke: var\\(--class-${classKey}\\);`,
        "u",
      ),
    );
  }
  assert.doesNotMatch(css, /url\(|gradient|filter:|mix-blend/iu);
});
