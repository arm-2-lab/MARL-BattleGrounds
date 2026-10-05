/**
 * @file Check the Class Requiems look (`src/looks/requiems.js`). records() is
 * pure data: a fall gives its anchor, radius (event anchorRadius or the 14 px
 * default), class, team and the "x2" home-fall flag only from the host's flag;
 * corpse marks come only from the given authorized scene (so an Agent POV
 * scene without a hidden corpse gives no mark), with Vigil Pips equal to the
 * team wave countdown plus one and each switch turned off dropping its piece;
 * the death_announcement and respawn_wave_occurred pulses draw nothing on the
 * arena. The Fallen panel (page chrome for the side rail) keeps only the last
 * 8 ticks' falls, never a future one, newest first, at most 4 rows; it claims a
 * Basic dot or Ultimate diamond only from the caller's ultimate fact, names no
 * unseen killer, holds this tick's rows to the victim's Fall beat and reads
 * "No one has fallen in the last 8 ticks." when empty.
 * build() keeps the END state in the DOM (Fall Ring at 35%, requiem pieces at
 * 0, relics and pips held to the Fall beat), pushes only fill "backwards"
 * animations timed from the beat (none when settled), and draws export-safe
 * SVG (no url(), gradients, masks, filters, images or ids). The Fallen feed
 * tag matches the plan's derived paint part.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { DERIVED_EVENT_PAINT_PARTS } from "../src/choreography-plan.js";
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
      listeners: [],
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      hasAttribute: (/** @type {string} */ name) => attributes.has(name),
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
      addEventListener: (/** @type {string} */ type) => element.listeners.push(type),
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

/** @param {any} root */
function assertExportSafe(root) {
  const text = serialize(root);
  assert.doesNotMatch(text, /url\(|gradient|<mask|<filter|<image|<foreignObject/iu);
  assert.doesNotMatch(text, / id="/u);
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

/** @param {number} classId @param {Record<string, any>} [extra] */
function deathPulse(classId, extra = {}) {
  return {
    kind: "semantic_pulse",
    cueSemantic: "agent_died",
    eventId: `death-${classId}`,
    anchor: { x: 100, y: 80 },
    phaseStart: 900,
    phaseEnd: 2400,
    recipientIdentity: { class_id: classId, team_id: 2, public_agent_id: "7" },
    ...extra,
  };
}

/** @type {any} */
const transform = {
  worldToScreen: (/** @type {any} */ point) => ({ x: point[0] * 20, y: point[1] * 20 }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 20,
};

/** @param {Array<Record<string, any>>} agents */
function sceneWith(agents) {
  return {
    agents,
    respawn_waves: [
      { team_index: 0, team_id: 1, period_steps: 5, countdown_steps: 3 },
      { team_index: 1, team_id: 2, period_steps: 7, countdown_steps: 6 },
    ],
  };
}

const corpseA = {
  presentation_key: "pk-a",
  public_agent_id: "2",
  class_id: 4,
  team_id: 1,
  life_state: "corpse",
  alive: false,
  position: [3, 4],
  radius: 0.75,
};
const corpseB = {
  ...corpseA,
  presentation_key: "pk-b",
  public_agent_id: "8",
  team_id: 2,
  class_id: 1,
};
const living = {
  ...corpseA,
  presentation_key: "pk-c",
  life_state: "alive",
  alive: true,
};

test("a fall record carries anchor, radius, class, team and the host's home-fall flag only", () => {
  const plain = requiems.records({ event: deathPulse(5) });
  assert.equal(plain.length, 1);
  assert.ok(Object.isFrozen(plain) && Object.isFrozen(plain[0]));
  assert.equal(plain[0].kind, "fall");
  assert.equal(plain[0].radius, requiems.DEFAULT_BODY_RADIUS_PX);
  assert.equal(plain[0].classKey, "priest");
  assert.equal(plain[0].teamKey, "b");
  assert.equal(plain[0].homeFall, false);
  const home = requiems.records({
    event: deathPulse(2, { homeFall: true, anchorRadius: 18 }),
  });
  assert.equal(home[0].homeFall, true);
  assert.equal(home[0].radius, 18);
  assert.equal(
    requiems.records({ event: deathPulse(2, { homeFall: "yes" }) })[0].homeFall,
    false,
  );
  assert.deepEqual(
    requiems.records({
      event: { kind: "semantic_pulse", cueSemantic: "cooldown_ready" },
    }),
    [],
  );
  assert.deepEqual(requiems.records({ event: { kind: "fallen_feed_row" } }), []);
  assert.deepEqual(requiems.records({}), []);
});

test("corpse marks come only from the given scene, with pips equal to countdown plus one", () => {
  const oracle = requiems.records({
    part: "corpse-marks",
    scene: sceneWith([corpseA, corpseB, living]),
    transform,
  });
  assert.deepEqual(
    oracle.map((record) => [record.presentationKey, record.pips, record.relic]),
    [
      ["pk-a", 4, true],
      ["pk-b", 7, true],
    ],
  );
  assert.deepEqual(oracle[0].center, { x: 60, y: 80 });
  assert.equal(oracle[0].radius, 15);
  // An Agent POV scene holds only seen bodies: the hidden corpse gets no mark.
  const pov = requiems.records({
    part: "corpse-marks",
    scene: sceneWith([corpseA, living]),
    transform,
    audience: "agent_pov",
  });
  assert.deepEqual(
    pov.map((record) => record.presentationKey),
    ["pk-a"],
  );
  const noPips = requiems.records({
    part: "corpse-marks",
    scene: sceneWith([corpseA]),
    transform,
    filters: { vigil_pips: false },
  });
  assert.equal(noPips[0].pips, 0);
  assert.equal(noPips[0].relic, true);
  assert.deepEqual(
    requiems.records({
      part: "corpse-marks",
      scene: sceneWith([corpseA]),
      transform,
      filters: { vigil_pips: false, death_effects: false },
    }),
    [],
  );
  assert.deepEqual(
    requiems.records({ part: "corpse-marks", scene: null, transform }),
    [],
  );
});

test("feed and wave pulses draw nothing on the arena", () => {
  for (const cueSemantic of ["death_announcement", "respawn_wave_occurred"]) {
    assert.deepEqual(
      requiems.records({
        event: {
          kind: "semantic_pulse",
          cueSemantic,
          eventId: cueSemantic,
          teamId: 2,
          teamSide: "right",
          anchor: { x: 800, y: 100 },
          members: [{ title: "Mage · ID 1", classId: 1, teamId: 1 }],
        },
      }),
      [],
    );
  }
});

test("the Fallen panel keeps the last 8 ticks, newest first, at most 4 rows, and names no unseen killer", () => {
  const victim = (/** @type {string} */ id) => ({
    publicAgentId: id,
    title: `Mage · ID ${id}`,
    classId: 1,
    teamId: 1,
  });
  const killer = { publicAgentId: "9", title: "Rogue · ID 9", classId: 4, teamId: 2 };
  const fall = (
    /** @type {number} */ tick,
    /** @type {string} */ id,
    /** @type {Record<string, any>} */ extra = {},
  ) => ({
    tick,
    victim: victim(id),
    victimPresentationKey: `pk-${id}`,
    killers: [killer],
    killingTeamId: 2,
    points: 1,
    ...extra,
  });
  const rows = [
    fall(102, "0"),
    fall(103, "1"),
    fall(110, "2", { killers: [], ultimate: null }),
    fall(108, "3", { points: 2, ultimate: { classId: 4, publicAgentId: "9" } }),
    fall(110, "4", { ultimate: null }),
    fall(111, "5"),
  ];
  const records = requiems.fallenFeedRecords({
    tick: 110,
    rows,
    displayIdOf: (/** @type {string} */ id) => id,
  });
  // Tick 102 is 8 ticks old (outside the window) and tick 111 is the future.
  assert.deepEqual(
    records.map((record) => [record.tick, record.victim.publicAgentId]),
    [
      [110, "2"],
      [110, "4"],
      [108, "3"],
      [103, "1"],
    ],
  );
  assert.equal(records.length, requiems.FEED_MAX_ROWS);
  assert.equal(requiems.FEED_WINDOW_TICKS, 8);
  const [unnamed, basic, ultimate, older] = records;
  assert.equal(unnamed.killers.length, 0);
  assert.equal(unnamed.ability, null);
  assert.equal(unnamed.label, "Tick 110: Mage · ID 2 fell");
  assert.equal(unnamed.beatKey, "pk-2");
  assert.deepEqual(basic.ability, { kind: "basic", classKey: null });
  assert.equal(basic.label, "Tick 110: Rogue · ID 9 took down Mage · ID 4");
  assert.deepEqual(ultimate.ability, { kind: "ultimate", classKey: "rogue" });
  assert.deepEqual(ultimate.points, { text: "+2", teamKey: "b" });
  assert.equal(ultimate.beatKey, null);
  // Without the caller's ultimate fact, no Basic or Ultimate mark is claimed.
  assert.equal(older.ability, null);
  assert.equal(older.ageTicks, 7);
  assert.equal(older.killers[0].displayId, "9");
  assert.deepEqual(requiems.fallenFeedRecords({ tick: Number.NaN, rows }), []);

  const doc = fakeDocument();
  const panel = requiems.buildFallenFeed(doc, records);
  assert.equal(panel.tagName, "section");
  assert.equal(panel.getAttribute("aria-label"), "Fallen · Last 8 Ticks");
  assert.equal(byClass(panel, "fallen-feed__title")[0].textContent, "Fallen");
  assert.equal(byClass(panel, "fallen-feed__window")[0].textContent, "Last 8 Ticks");
  const shown = byClass(panel, "fallen-feed__row");
  assert.equal(shown.length, 4);
  assert.equal(shown[0].getAttribute("data-beat-hold"), "fall");
  assert.equal(shown[0].getAttribute("data-beat-key"), "pk-2");
  assert.equal(shown[0].getAttribute("data-beat-side"), "after");
  assert.equal(shown[2].getAttribute("data-beat-hold"), null);
  assert.equal(byClass(shown[3], "fallen-feed__tick")[0].textContent, "Tick 103");
  assert.equal(byClass(shown[0], "fallen-feed__emblem--fallen").length, 1);
  assert.equal(byClass(shown[0], "fallen-feed__ability").length, 0);
  assert.equal(byClass(shown[1], "fallen-feed__ability--basic").length, 1);
  assert.equal(byClass(shown[2], "fallen-feed__ability--ultimate").length, 1);
  assert.equal(byClass(shown[2], "fallen-feed__points")[0].textContent, "+2");
  assert.doesNotMatch(serialize(shown[0]), /Unseen|\?|Rogue/u);
  for (const row of shown) assertExportSafe(row);
  const empty = requiems.buildFallenFeed(doc, []);
  assert.equal(
    byClass(empty, "fallen-feed__empty")[0].textContent,
    "No one has fallen in the last 8 ticks.",
  );
});

test("a settled fall holds END: ring at 35 percent, x2 only for a home fall, no animations", () => {
  const { doc, mount } = mountFor({ settled: true });
  requiems.build(
    requiems.records({ event: deathPulse(3, { homeFall: true }) }),
    doc,
    mount,
  );
  const ring = byClass(mount.group, "requiem__fall-ring")[0];
  assert.equal(ring.getAttribute("opacity"), "0.35");
  assert.equal(byClass(mount.group, "requiem__x2")[0].textContent, "×2");
  assert.equal(mount.animationSpecs.length, 0);
  assert.equal(byClass(mount.group, "requiem__leaves").length, 0);
  const plain = mountFor({ settled: true });
  requiems.build(requiems.records({ event: deathPulse(3) }), plain.doc, plain.mount);
  assert.equal(byClass(plain.mount.group, "requiem__x2").length, 0);
  assertExportSafe(mount.group);
});

test("a playing fall keeps requiem pieces at 0 and times fill-backwards animations from the beat", () => {
  for (const classId of [1, 2, 3, 4, 5]) {
    const { doc, mount } = mountFor();
    requiems.build(
      requiems.records({ event: deathPulse(classId, { anchorRadius: 16 }) }),
      doc,
      mount,
    );
    const specs = mount.animationSpecs;
    assert.ok(specs.length >= 3, `class ${classId} animates`);
    assert.ok(
      specs.length <= requiems.BUDGET.animationsPerRecord,
      `class ${classId} budget`,
    );
    assert.equal(
      new Set(specs.map((/** @type {any} */ spec) => spec.id)).size,
      specs.length,
    );
    for (const spec of specs) {
      assert.equal(spec.options.fill, "backwards");
      assert.ok(
        spec.options.delay >= 900,
        `${spec.id} starts at or after the Fall beat`,
      );
      const first = Array.isArray(spec.keyframes) ? spec.keyframes[0] : null;
      assert.equal(first?.opacity, 0, `${spec.id} is hidden before the beat`);
    }
    const pieces = descendants(mount.group).filter((node) =>
      /requiem__(cracks|motes|rim|crack|ash-ring|leaves|rime|smoke|puffs|ray|ember|emblem)/u.test(
        node.getAttribute("class") ?? "",
      ),
    );
    assert.ok(pieces.length > 0);
    for (const piece of pieces) {
      if (
        piece.getAttribute("opacity") !== null &&
        piece.parentElement.getAttribute("class") !== "requiem__ember"
      ) {
        assert.equal(piece.getAttribute("opacity"), "0", piece.getAttribute("class"));
      }
    }
    assert.ok(descendants(mount.group).length <= requiems.BUDGET.nodesPerRecord + 12);
    assertExportSafe(mount.group);
  }
  const priest = mountFor();
  requiems.build(requiems.records({ event: deathPulse(5) }), priest.doc, priest.mount);
  assert.equal(
    priest.mount.animationSpecs.filter((/** @type {any} */ spec) =>
      /requiem-ray-/u.test(spec.id),
    ).length,
    8,
  );
  const reduced = mountFor({ motionMode: "reduced" });
  requiems.build(
    requiems.records({ event: deathPulse(5) }),
    reduced.doc,
    reduced.mount,
  );
  assert.deepEqual(
    reduced.mount.animationSpecs.map((/** @type {any} */ spec) => [
      spec.options.delay,
      spec.options.duration,
    ]),
    [
      [225, 220],
      [225, 220],
    ],
  );
});

test("a return draws no END ring", () => {
  const { doc, mount } = mountFor({ settled: true });
  requiems.build(
    requiems.records({
      event: { ...deathPulse(1), cueSemantic: "agent_respawned", eventId: "back" },
    }),
    doc,
    mount,
  );
  assert.equal(
    byClass(mount.group, "homecoming__rings")[0].getAttribute("opacity"),
    "0",
  );
});

test("corpse marks hold to the Fall beat and draw relic and pips", () => {
  const doc = fakeDocument();
  const marks = requiems.records({
    part: "corpse-marks",
    scene: sceneWith([corpseA, { ...corpseB, radius: 0.8 }]),
    transform,
  });
  const nodes = requiems.build(marks, doc);
  assert.equal(nodes.length, 2);
  const [first, second] = nodes;
  assert.equal(first.getAttribute("data-beat-hold"), "fall");
  assert.equal(first.getAttribute("data-beat-key"), "pk-a");
  assert.equal(first.getAttribute("data-beat-side"), "after");
  assert.equal(byClass(first, "requiem-relic").length, 1);
  assert.equal(byClass(first, "vigil-pips")[0].getAttribute("data-pips"), "4");
  // Seven pips are too many for dots: the count is written instead.
  assert.equal(byClass(second, "vigil-pips__count")[0].textContent, "7");
  requiems.update(nodes, [marks[0], { ...marks[1], center: { x: 1, y: 2 } }]);
  assert.equal(second.getAttribute("transform"), "translate(1 2)");
  assert.equal(requiems.tooltip(marks[0]), null);
  for (const node of nodes) assertExportSafe(node);
});

test("the Fallen feed registers the plan's derived paint part tag", () => {
  const expected = DERIVED_EVENT_PAINT_PARTS.fallen_feed_row;
  assert.ok(
    requiems.PAINT_PARTS.some(
      (registration) =>
        JSON.stringify(registration.part) === JSON.stringify(expected.part) &&
        registration.filter === expected.filter,
    ),
  );
  assert.ok(Object.isFrozen(requiems.PAINT_PARTS));
});
