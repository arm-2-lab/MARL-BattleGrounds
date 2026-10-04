/**
 * @file Check the Basic Attacks look, Cleave And Bloom (`src/looks/basics.js`).
 * records() is pure data from one plan event: a ranged Mage, Hunter or Priest
 * draws its bolt, streak or ribbon from 2 px outside its rim to the target rim
 * and lands in its star, Shard Fan or Dawn Bloom; a Warrior or Rogue, or a
 * point-blank ranged blow (rims under 0.55 units), draws its class cleave; when
 * the two champions touch or overlap (even with their centres together) every
 * class still draws its full cleave, ending on the target's front shoulder
 * with the landing there, never a dot; with Close-Range Strokes off a
 * point-blank blow keeps its landing alone; a Priest
 * healing itself draws four blooms; the detail ladder drops to a class dot
 * under 10 px; Soft Glow adds glows in play only; Basic Ability Effects off
 * draws nothing (or the meaning dot). Arcane Blast During Burst: a Burst
 * Mage's bolt is the Comet Bolt (a 2.5 px violet bolt with a 1.2 px Starcore
 * core at 65% in a still and 95% in play, a comet head that rides the tip and
 * rests under the larger star, a ring with a flash around the star), within the
 * Basic budget (20 nodes, 4 animations) and export-safe; with the switch off it
 * draws exactly the plain bolt, a Mage without Burst is unchanged, and a
 * point-blank Burst blow keeps its thicker cleave. Agent POV: a blow from a
 * champion that left sight starts at the Last Seen spot the plan gave (the
 * Comet Bolt too) and its tooltip names the class only; records() reads
 * nothing but the event and the optional filters and transform. build() is
 * export-safe (no url(), gradients, masks, filters, images, ids or raw
 * colors), puts strokes, cleaves and the comet head in the route group and
 * landings and the comet ring in the event group, draws no glows, sparks or
 * flash in a still, and adds at most two animations (four for the Comet Bolt)
 * on the beat clock (fill "backwards", first frame invisible, gone by blow
 * 600 ms, Body Glide shifting them, reduced motion opacity only within
 * 150 ms). update() moves existing nodes. tooltip() titles the blow as
 * "Arcane Blast · Mage · ID 0 → Mage · ID 5 · 15 Damage". basics.css uses only
 * tokens and no CSS animation.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import * as basics from "../src/looks/basics.js";

/** @param {number} slot @param {number} classId @param {number} teamId */
function identity(slot, classId, teamId) {
  return {
    presentation_key: `body-${slot}`,
    public_agent_id: `agent-slot-${slot}`,
    display_agent_id: String(slot),
    class_id: classId,
    team_id: teamId,
  };
}

/** @param {Record<string, any>} [overrides] */
function blow(overrides = {}) {
  return {
    eventId: "event-1",
    kind: "activation",
    component: "basic",
    tokenId: "basic_damage",
    sourcePresentationKey: "body-0",
    sourcePublicAgentId: "agent-slot-0",
    sourceIdentity: identity(0, 1, 1),
    sourceDisclosure: "public",
    sourceClassId: 1,
    targetPresentationKey: "body-5",
    targetPublicAgentId: "agent-slot-5",
    recipientIdentity: identity(5, 1, 2),
    targetDisclosure: "public",
    sourceEndpointPhase: "successor",
    targetEndpointPhase: "successor",
    source: { x: 100, y: 200 },
    target: { x: 300, y: 200 },
    targetRadius: 23,
    sourceRadius: 23,
    pixelsPerUnit: 46,
    paintParts: { ability: true, semantic: true },
    presentationKind: "routed",
    route: null,
    phaseStart: 80,
    phaseImpact: 360,
    phaseEnd: 760,
    ...overrides,
  };
}

/** @param {Record<string, any>} event @param {Record<string, any>} [extra] */
function one(event, extra = {}) {
  const records = basics.records({ event, ...extra });
  assert.equal(records.length, 1);
  return /** @type {Record<string, any>} */ (records[0]);
}

class FakeElement {
  /** @param {string} tag */
  constructor(tag) {
    this.tagName = tag;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    /** @type {Record<string, string>} */
    this.style = {};
    this.ownerDocument = fakeDocument;
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
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    this.children.push(...nodes);
  }
  get childElementCount() {
    return this.children.length;
  }
  /** @returns {FakeElement[]} */
  descendants() {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }
  /** @param {string} selector */
  matches(selector) {
    const attribute = /^\[([\w-]+)="([^"]*)"\]$/u.exec(selector);
    if (attribute) return this.getAttribute(attribute[1]) === attribute[2];
    const className = /^\.([\w-]+)$/u.exec(selector);
    if (className) {
      return (this.getAttribute("class") ?? "").split(/\s+/u).includes(className[1]);
    }
    return false;
  }
  /** @param {string} selector */
  querySelector(selector) {
    return this.descendants().find((node) => node.matches(selector)) ?? null;
  }
  /** @param {string} selector */
  querySelectorAll(selector) {
    return this.descendants().filter((node) => node.matches(selector));
  }
  /** @returns {string} */
  serialize() {
    const attrs = [...this.attributes]
      .map(([key, value]) => ` ${key}="${value}"`)
      .join("");
    return `<${this.tagName}${attrs}>${this.children.map((child) => child.serialize()).join("")}</${this.tagName}>`;
  }
}

const fakeDocument = {
  /** @param {string} _namespace @param {string} tag */
  createElementNS(_namespace, tag) {
    return new FakeElement(tag);
  },
};

/** @param {Record<string, any>} [options] @param {Record<string, any>} [plan] */
function mount(options = {}, plan = {}) {
  return {
    group: new FakeElement("g"),
    connector: new FakeElement("g"),
    underlay: new FakeElement("g"),
    plan: { epochKey: "epoch", phases: { total: 900, reducedTotal: 220 }, ...plan },
    options: { motionMode: "normal", settled: false, ...options },
    /** @type {any[]} */
    animationSpecs: [],
  };
}

/** @param {Record<string, any>} record @param {Record<string, any>} target */
function draw(record, target) {
  basics.build(
    [record],
    /** @type {any} */ (fakeDocument),
    /** @type {any} */ (target),
  );
  return target;
}

/** @param {FakeElement} root @param {string} key */
function part(root, key) {
  return root.querySelector(`[data-basic-part="${key}"]`);
}

test("other events give no records, and an empty input draws nothing", () => {
  assert.deepEqual(basics.records({}), []);
  assert.deepEqual(basics.records({ event: { kind: "net_health" } }), []);
  assert.deepEqual(basics.records({ event: blow({ component: "ultimate" }) }), []);
  assert.deepEqual(basics.build([], /** @type {any} */ (null)), []);
});

test("Arcane Blast at range is one bolt from rim to rim landing in the star", () => {
  const record = one(blow({ outputAmount: 15 }));
  assert.equal(record.kind, "basic_blow");
  assert.equal(record.shape, "bolt");
  assert.equal(record.classKey, "mage");
  assert.equal(record.level, 3);
  assert.equal(record.closeRange, false);
  assert.equal(record.rimGapUnits, 3.35);
  assert.deepEqual(record.stroke.origin, [125, 200]);
  assert.equal(record.stroke.angle, 0);
  assert.equal(record.stroke.length, 151.08);
  assert.equal(record.stroke.core, 2);
  assert.equal(record.stroke.centre, 1);
  assert.equal(record.landing.type, "star");
  assert.deepEqual(record.landing.origin, [269.18, 200]);
  assert.equal(record.cleaves.length, 0);
  assert.equal(
    record.tooltipText,
    "Arcane Blast · Mage · ID 0 → Mage · ID 5 · 15 Damage",
  );
  assert.ok(Object.isFrozen(record) && Object.isFrozen(record.stroke));
  assert.equal(JSON.stringify(record), JSON.stringify(one(blow({ outputAmount: 15 }))));
});

test("point blank draws the class cleave, or the landing alone with the switch off", () => {
  const close = blow({ target: { x: 166, y: 200 } });
  const cleave = one(close);
  assert.equal(cleave.closeRange, true);
  assert.equal(cleave.shape, "cleave");
  assert.equal(cleave.stroke, null);
  assert.equal(cleave.landing.type, "star");
  assert.deepEqual(
    cleave.cleaves.map((/** @type {any} */ each) => each.part),
    ["main"],
  );
  assert.match(cleave.tooltipText, /From Right Beside Its Target$/u);
  const off = one(close, { filters: { close_range_strokes: false } });
  assert.equal(off.shape, "landing_only");
  assert.equal(off.stroke, null);
  assert.equal(off.cleaves.length, 0);
  assert.equal(off.landing.type, "star");
  const plan = one({
    ...close,
    paintParts: { ...close.paintParts, close_range: false },
  });
  assert.equal(plan.shape, "landing_only");
});

test("touching or overlapping champions still get every class's full cleave", () => {
  // Source at (100, 200), radius 23 px: rims touching, 4 px overlapping, 6 px
  // apart, and centres together (an authored setup), for every class.
  for (const sourceClassId of [1, 2, 3, 4, 5]) {
    for (const targetX of [146, 142, 152, 100]) {
      const record = one(blow({ sourceClassId, target: { x: targetX, y: 200 } }));
      assert.equal(record.shape, "cleave", `class ${sourceClassId} at x ${targetX}`);
      assert.equal(record.dot, null);
      const main = record.cleaves[0];
      const points = [...main.d.matchAll(/(-?[\d.]+) (-?[\d.]+)/gu)].map((m) => [
        main.origin[0] + Number(m[1]),
        main.origin[1] + Number(m[2]),
      ]);
      // A real stroke, not a tiny line: it spans at least one body radius.
      const xs = points.map((p) => p[0]);
      const ys = points.map((p) => p[1]);
      assert.ok(
        Math.hypot(
          Math.max(...xs) - Math.min(...xs),
          Math.max(...ys) - Math.min(...ys),
        ) > 23,
      );
      if (targetX !== 100) {
        // It ends on the target's front shoulder, above the line between them,
        // and the landing (Mage, Hunter, Priest) sits at that end.
        const end = points.reduce((best, p) =>
          Math.hypot(p[0] - targetX, p[1] - 200) <
          Math.hypot(best[0] - targetX, best[1] - 200)
            ? p
            : best,
        );
        assert.ok(end[1] < 200 - 10 && end[0] > 100 && end[0] < targetX);
        if (record.landing) {
          assert.ok(
            Math.hypot(
              record.landing.origin[0] - end[0],
              record.landing.origin[1] - end[1],
            ) < 16,
          );
        }
      }
    }
  }
  // Close-Range Strokes off still keeps a touching ranged blow's landing alone.
  const off = one(blow({ target: { x: 146, y: 200 } }), {
    filters: { close_range_strokes: false },
  });
  assert.equal(off.shape, "landing_only");
});

test("melee classes cleave: Warrior sweep with sparks, Rogue double flick", () => {
  const warrior = one(blow({ sourceClassId: 2, sourceIdentity: identity(0, 2, 1) }));
  assert.equal(warrior.shape, "cleave");
  assert.equal(warrior.landing, null);
  assert.ok(warrior.sparks);
  assert.deepEqual(
    warrior.cleaves[0].accents.map((/** @type {any} */ each) => each.role),
    ["edge"],
  );
  assert.deepEqual([...warrior.cleaves[0].grow], [150, 240]);
  assert.match(warrior.tooltipText, /^Shield Bash · Warrior · ID 0 → /u);
  const smaller = one(blow({ sourceClassId: 2, targetRadius: 16, sourceRadius: 16 }));
  assert.equal(smaller.level, 2);
  assert.equal(smaller.sparks, null);
  const rogue = one(blow({ sourceClassId: 4 }));
  assert.deepEqual(
    rogue.cleaves.map((/** @type {any} */ each) => [each.part, [...each.grow]]),
    [
      ["main", [190, 235]],
      ["twin", [250, 295]],
    ],
  );
  for (const each of [...warrior.cleaves, ...rogue.cleaves]) {
    assert.match(each.d, /^M[-\d. L]+Z$/u);
  }
});

test("Hunter streaks into the Shard Fan, Priest ribbons into the Dawn Bloom", () => {
  const hunter = one(blow({ sourceClassId: 3 }));
  assert.equal(hunter.shape, "streak");
  assert.deepEqual([...hunter.stroke.bars], [3, 6, 9]);
  assert.equal(hunter.landing.type, "fan");
  assert.deepEqual(Object.keys(hunter.landing.paths).sort(), [
    "shards",
    "spark",
    "spines",
  ]);
  assert.deepEqual(
    [...one(blow({ sourceClassId: 3, targetRadius: 16 })).stroke.bars],
    [3, 7],
  );
  const priest = one(blow({ sourceClassId: 5, tokenId: "basic_heal" }));
  assert.equal(priest.shape, "ribbon");
  assert.deepEqual(priest.stroke.ctrl, [75.54, -27.6]);
  assert.equal(priest.landing.type, "bloom");
  assert.equal(priest.landing.paths.petals.split("M").length - 1, 6);
  const self = one(
    blow({
      sourceClassId: 5,
      targetPresentationKey: "body-0",
      target: { x: 100, y: 200 },
      outputAmount: 12,
    }),
  );
  assert.equal(self.shape, "self_heal");
  assert.equal(self.blessings.paths.petals.split("M").length - 1, 24);
  assert.match(self.tooltipText, /→ Itself · 12 Healing$/u);
});

test("the detail ladder, Arcane Blast During Burst and Soft Glow follow their rules", () => {
  assert.deepEqual(
    [23, 22, 16, 12, 9].map((radius) => basics.detailLevel(radius)),
    [3, 3, 2, 1, 0],
  );
  const tiny = one(blow({ targetRadius: 8, sourceRadius: 8 }));
  assert.equal(tiny.shape, "dot");
  assert.equal(tiny.dot.r, 1.5);
  const thin = one(blow({ targetRadius: 12, sourceRadius: 12 }));
  assert.equal(thin.stroke.core, 1.5);
  assert.equal(thin.stroke.centre, 0);
  assert.equal(thin.glow, false);
  const kindled = one(blow({ sourceKindled: true }));
  assert.equal(kindled.kindled, true);
  assert.equal(kindled.stroke.core, 2.5);
  assert.equal(kindled.stroke.centre, 1.2);
  assert.equal(kindled.stroke.glow, 9);
  assert.notEqual(kindled.landing.paths.star, one(blow()).landing.paths.star);
  const plain = one(blow({ sourceKindled: true }), {
    filters: { kindled_blast: false },
  });
  assert.equal(plain.kindled, false);
  assert.equal(plain.comet, null);
  assert.equal(plain.stroke.core, 2);
  // The 10-14 px level keeps the thin bolt with no core line and draws the
  // comet head and ring at 70%.
  const small = one(blow({ sourceKindled: true, targetRadius: 12, sourceRadius: 12 }));
  assert.equal(small.stroke.core, 1.5);
  assert.equal(small.stroke.centre, 0);
  assert.equal(small.comet.scale, 0.7);
  assert.equal(small.comet.ringRadius, 3.7);
  assert.equal(one(blow()).glow, true);
  assert.equal(one(blow(), { filters: { soft_glow: false } }).glow, false);
});

test("Basic Ability Effects off draws nothing, or the meaning dot alone", () => {
  const none = one(blow({ paintParts: { ability: false, semantic: false } }));
  assert.equal(none.shape, "none");
  const target = draw(none, mount());
  assert.equal(target.group.childElementCount + target.underlay.childElementCount, 0);
  assert.equal(target.animationSpecs.length, 0);
  const meaning = one(blow({ paintParts: { ability: false, semantic: true } }));
  assert.equal(meaning.shape, "meaning_dot");
  assert.equal(meaning.stroke, null);
  assert.equal(meaning.landing, null);
});

test("Agent POV: a leaver's blow starts at its Last Seen spot and names its class only", () => {
  const event = blow({
    sourceEndpointPhase: "transition_start",
    sourceDisclosure: "redacted",
    sourceIdentity: null,
    sourceRadius: undefined,
    source: { x: 60, y: 120 },
  });
  const record = one(event);
  assert.equal(record.fromLastSeen, true);
  const u = [240 / Math.hypot(240, 80), 80 / Math.hypot(240, 80)];
  assert.deepEqual(record.stroke.origin, [
    Math.round((60 + u[0] * 25) * 100) / 100,
    Math.round((120 + u[1] * 25) * 100) / 100,
  ]);
  assert.match(record.tooltipText, /^Arcane Blast · Mage → Mage · ID 5$/u);
  assert.doesNotMatch(record.tooltipText, /ID 0/u);
  const hiddenTarget = one(
    blow({ recipientIdentity: null, targetDisclosure: "redacted" }),
  );
  assert.match(hiddenTarget.tooltipText, /→ Out Of Sight$/u);
  const guarded = { event };
  for (const name of ["scene", "researcherSpace", "researcher_space", "matchSummary"]) {
    Object.defineProperty(guarded, name, {
      get() {
        throw new Error(`records read ${name}`);
      },
    });
  }
  assert.equal(basics.records(guarded).length, 1);
  assert.equal(
    JSON.stringify(basics.records({ event: blow(), audience: "agent_pov" })),
    JSON.stringify(basics.records({ event: blow(), audience: "researcher" })),
  );
});

test("build is export-safe and puts strokes in the Strike Band and landings on top", () => {
  for (const classId of [1, 2, 3, 4, 5]) {
    for (const settled of [true, false]) {
      const target = draw(one(blow({ sourceClassId: classId })), mount({ settled }));
      const svg = [target.group, target.connector, target.underlay]
        .map((node) => node.serialize())
        .join("");
      assert.doesNotMatch(
        svg,
        /url\(|gradient|<mask|<filter|<image|clip-path|\sid=|#[0-9a-f]{3}/iu,
      );
      assert.doesNotMatch(svg, /style=/u);
      const band =
        target.underlay.querySelector('[data-basic-part="stroke"]') ??
        target.underlay.querySelector('[data-basic-part="cleave-main"]');
      assert.ok(band, `class ${classId} draws in the Strike Band`);
      if (classId === 1 || classId === 3 || classId === 5) {
        assert.ok(part(target.group, "landing")?.matches(".combat-impact"));
      }
    }
  }
  const still = draw(one(blow()), mount({ settled: true }));
  assert.equal(part(still.underlay, "stroke-glow"), null);
  assert.equal(part(still.underlay, "stroke-draw")?.getAttribute("opacity"), "0.5");
  assert.equal(part(still.group, "landing-stamp")?.getAttribute("opacity"), "0.8");
  assert.equal(still.animationSpecs.length, 0);
  const playing = draw(one(blow()), mount());
  assert.ok(part(playing.underlay, "stroke-glow"));
  const warriorStill = draw(one(blow({ sourceClassId: 2 })), mount({ settled: true }));
  assert.equal(part(warriorStill.group, "sparks-flick"), null);
  assert.equal(
    part(warriorStill.underlay, "cleave-main-grow")?.getAttribute("opacity"),
    "0.8",
  );
});

test("animations sit on the beat clock, start invisible and end by blow 600 ms", () => {
  const plain = draw(one(blow()), mount());
  assert.deepEqual(
    plain.animationSpecs.map((/** @type {any} */ spec) => spec.id),
    ["mbg:epoch:event-1:basic-bolt", "mbg:epoch:event-1:basic-landing-star"],
  );
  const [bolt, star] = plain.animationSpecs;
  assert.equal(bolt.options.delay, 80);
  assert.equal(bolt.options.duration, 680);
  assert.equal(bolt.options.fill, "backwards");
  assert.equal(bolt.keyframes[0].opacity, 0);
  assert.equal(bolt.keyframes.at(-1).opacity, 0);
  assert.equal(star.options.delay, 240);
  assert.equal(star.keyframes[0].opacity, 0);
  assert.equal(star.keyframes[0].transform, "scale(1.2)");
  const glide = draw(
    one(blow({ phaseStart: 280, phaseImpact: 560, phaseEnd: 960 })),
    mount(),
  );
  assert.equal(glide.animationSpecs[0].options.delay, 280);
  assert.equal(glide.animationSpecs[1].options.delay, 440);
  for (const classId of [2, 3, 4, 5]) {
    const target = draw(one(blow({ sourceClassId: classId })), mount());
    assert.ok(target.animationSpecs.length <= 2, `class ${classId}`);
    for (const spec of target.animationSpecs) {
      assert.equal(spec.keyframes[0].opacity, 0);
      assert.equal(spec.options.fill, "backwards");
    }
  }
  const reduced = draw(one(blow()), mount({ motionMode: "reduced" }));
  for (const spec of reduced.animationSpecs) {
    assert.equal(spec.options.duration, 150);
    assert.ok(
      spec.keyframes.every((/** @type {any} */ frame) => !("transform" in frame)),
    );
  }
  assert.equal(
    draw(one(blow()), mount({ motionMode: "off" })).animationSpecs.length,
    0,
  );
});

/** @param {Record<string, any>} target @returns {FakeElement[]} */
function drawnNodes(target) {
  return [target.group, target.connector, target.underlay].flatMap(
    (/** @type {FakeElement} */ node) => node.descendants(),
  );
}

/** @param {Record<string, any>} record @param {boolean} settled */
function picture(record, settled) {
  const target = draw(record, mount({ settled }));
  return (
    [target.group, target.connector, target.underlay]
      .map((node) => node.serialize())
      .join("") +
    JSON.stringify(
      target.animationSpecs.map((/** @type {any} */ spec) => [
        spec.id,
        spec.options,
        spec.keyframes,
      ]),
    )
  );
}

test("a Burst Mage's Arcane Blast is the Comet Bolt, export-safe and within the Basic budget", () => {
  const record = one(blow({ sourceKindled: true }));
  assert.equal(record.shape, "bolt");
  assert.deepEqual(record.comet, {
    head: 142.8,
    tail: 56,
    scale: 1,
    ringRadius: 10.12,
    flashRadius: 10.7,
  });
  // The head rests under the star: 1.4 radii out from the target center.
  assert.deepEqual(record.landing.origin, [267.8, 200]);
  assert.equal(
    Math.round((record.stroke.origin[0] + record.comet.head) * 100) / 100,
    record.landing.origin[0],
  );
  const playing = draw(record, mount());
  assert.equal(
    part(playing.underlay, "stroke")?.getAttribute("class"),
    "basic-blow basic-blow--bolt basic-blow--comet",
  );
  assert.equal(part(playing.underlay, "stroke-draw")?.getAttribute("opacity"), "0.65");
  assert.equal(
    part(playing.underlay, "stroke-glow")?.getAttribute("stroke-width"),
    "9",
  );
  assert.equal(
    part(playing.underlay, "stroke-core")?.getAttribute("stroke-width"),
    "2.5",
  );
  assert.equal(
    part(playing.underlay, "stroke-centre")?.getAttribute("stroke-width"),
    "1.2",
  );
  assert.ok(part(playing.underlay, "stroke-hit"), "the bolt keeps its hover line");
  assert.equal(
    part(playing.underlay, "comet-head")?.getAttribute("transform"),
    "translate(142.8 0)",
  );
  assert.equal(
    part(playing.underlay, "comet-head-move")?.getAttribute("opacity"),
    "0.65",
  );
  for (const key of [
    "comet-tail",
    "comet-ball",
    "comet-tail-core",
    "comet-ball-core",
  ]) {
    assert.ok(part(playing.underlay, key), key);
    assert.equal(part(playing.group, key), null, `${key} stays in the Strike Band`);
  }
  assert.equal(
    part(playing.underlay, "comet-tail")?.getAttribute("d"),
    "M0 -4.2L-56 0L0 4.2Z",
  );
  // The ring sits on the star, drawn after it, with its flash dark until played.
  assert.deepEqual(
    playing.group.children.map((/** @type {FakeElement} */ child) =>
      child.getAttribute("data-basic-part"),
    ),
    ["landing", "comet-ring"],
  );
  assert.equal(
    part(playing.group, "comet-ring")?.getAttribute("transform"),
    "translate(267.8 200)",
  );
  const grow = part(playing.group, "comet-ring-grow");
  assert.equal(grow?.getAttribute("opacity"), "0.55");
  assert.equal(grow?.getAttribute("fill-opacity"), "0");
  assert.equal(part(playing.group, "comet-flash")?.getAttribute("r"), "10.7");
  assert.equal(part(playing.group, "comet-ring-outer")?.getAttribute("r"), "10.12");
  assert.equal(
    part(playing.group, "comet-ring-outer")?.getAttribute("stroke-width"),
    "3",
  );
  assert.equal(
    part(playing.group, "comet-ring-inner")?.getAttribute("stroke-width"),
    "1.2",
  );
  assert.equal(drawnNodes(playing).length, 20);
  assert.ok(drawnNodes(playing).length <= basics.BUDGET.nodesPerRecord);
  assert.deepEqual(
    playing.animationSpecs.map((/** @type {any} */ spec) => spec.id),
    [
      "mbg:epoch:event-1:basic-comet-bolt",
      "mbg:epoch:event-1:basic-comet-head",
      "mbg:epoch:event-1:basic-landing-star",
      "mbg:epoch:event-1:basic-comet-ring",
    ],
  );
  assert.ok(playing.animationSpecs.length <= basics.BUDGET.animationsPerRecord);
  for (const settled of [false, true]) {
    const svg = picture(record, settled);
    assert.doesNotMatch(
      svg,
      /url\(|gradient|<mask|<filter|<image|clip-path|\sid=|#[0-9a-f]{3}|style=/iu,
    );
  }
  const still = draw(record, mount({ settled: true }));
  assert.equal(part(still.underlay, "stroke-glow"), null);
  assert.equal(part(still.group, "comet-flash"), null);
  assert.equal(part(still.underlay, "stroke-draw")?.getAttribute("opacity"), "0.65");
  assert.equal(
    part(still.underlay, "comet-head-move")?.getAttribute("opacity"),
    "0.65",
  );
  assert.equal(part(still.group, "comet-ring-grow")?.getAttribute("opacity"), "0.55");
  assert.equal(part(still.group, "landing-stamp")?.getAttribute("opacity"), "0.8");
  assert.equal(still.animationSpecs.length, 0);
  const moved = basics.records({
    event: blow({ sourceKindled: true, target: { x: 300, y: 260 } }),
  });
  basics.update([still.group, still.connector, still.underlay], moved);
  assert.equal(
    part(still.group, "comet-ring")?.getAttribute("transform"),
    `translate(${moved[0].landing.origin[0]} ${moved[0].landing.origin[1]})`,
  );
  assert.equal(
    part(still.underlay, "comet-head")?.getAttribute("transform"),
    `translate(${moved[0].comet.head} 0)`,
  );
});

test("the Comet Bolt plays on the beat clock and is gone by blow 600 ms", () => {
  const playing = draw(one(blow({ sourceKindled: true })), mount());
  const specs = new Map(
    playing.animationSpecs.map((/** @type {any} */ spec) => [
      spec.id.split(":")[3],
      spec,
    ]),
  );
  const bolt = specs.get("basic-comet-bolt");
  assert.equal(bolt.options.delay, 80);
  assert.equal(bolt.options.duration, 680);
  assert.equal(bolt.keyframes[1].opacity, 0.95);
  const head = specs.get("basic-comet-head");
  assert.equal(head.options.delay, 80);
  assert.equal(head.keyframes[0].transform, "translateX(-142.8px)");
  assert.ok(
    head.keyframes.some(
      (/** @type {any} */ frame) =>
        frame.transform === "translateX(0px)" && frame.offset < 100 / 680,
    ),
    "the head reaches the star while the bolt draws in",
  );
  assert.equal(specs.get("basic-landing-star").options.delay, 240);
  const ring = specs.get("basic-comet-ring");
  assert.equal(ring.options.delay, 240);
  assert.deepEqual(
    [
      ring.keyframes[0].opacity,
      ring.keyframes[0].transform,
      ring.keyframes[0].fillOpacity,
    ],
    [0, "scale(0.34)", 0],
  );
  /** @param {any} spec @param {string} property */
  const timeline = (spec, property) =>
    spec.keyframes
      .filter((/** @type {any} */ frame) => property in frame)
      .map((/** @type {any} */ frame) => [
        Math.round(spec.options.delay + frame.offset * spec.options.duration),
        frame[property],
      ]);
  assert.deepEqual(timeline(ring, "fillOpacity"), [
    [240, 0],
    [256, 0.6],
    [280, 0.41],
    [300, 0.22],
    [320, 0],
  ]);
  assert.deepEqual(timeline(ring, "transform"), [
    [240, "scale(0.34)"],
    [255, "scale(0.5)"],
    [340, "scale(1)"],
    [440, "scale(1)"],
  ]);
  for (const spec of specs.values()) {
    assert.equal(spec.options.fill, "backwards");
    assert.equal(spec.keyframes[0].opacity, 0);
    const lit = timeline(spec, "opacity").filter(
      (/** @type {[number, number]} */ [, value]) => value > 0,
    );
    assert.ok(Number(lit.at(-1)?.[0]) <= 600, spec.id);
    assert.equal(spec.keyframes.at(-1).opacity, 0);
  }
  const glide = draw(
    one(
      blow({ sourceKindled: true, phaseStart: 280, phaseImpact: 560, phaseEnd: 960 }),
    ),
    mount(),
  );
  assert.deepEqual(
    glide.animationSpecs.map((/** @type {any} */ spec) => spec.options.delay),
    [280, 280, 440, 440],
  );
  const reduced = draw(
    one(blow({ sourceKindled: true })),
    mount({ motionMode: "reduced" }),
  );
  assert.equal(reduced.animationSpecs.length, 4);
  for (const spec of reduced.animationSpecs) {
    assert.equal(spec.options.duration, 150);
    assert.ok(
      spec.keyframes.every(
        (/** @type {any} */ frame) =>
          !("transform" in frame) && !("fillOpacity" in frame),
      ),
    );
  }
  assert.equal(
    draw(one(blow({ sourceKindled: true })), mount({ motionMode: "off" }))
      .animationSpecs.length,
    0,
  );
});

test("switch off draws the plain Arcane Blast; no Burst, other classes and point blank keep their looks", () => {
  for (const settled of [false, true]) {
    const plain = picture(one(blow()), settled);
    assert.equal(
      picture(
        one(blow({ sourceKindled: true }), { filters: { kindled_blast: false } }),
        settled,
      ),
      plain,
    );
    const resolved = blow({ sourceKindled: true });
    assert.equal(
      picture(
        one({ ...resolved, paintParts: { ...resolved.paintParts, kindled: false } }),
        settled,
      ),
      plain,
    );
    assert.doesNotMatch(plain, /comet/u);
  }
  assert.equal(one(blow()).comet, null);
  for (const classId of [2, 3, 4, 5]) {
    const other = one(blow({ sourceClassId: classId, sourceKindled: true }));
    assert.equal(other.kindled, false, `class ${classId}`);
    assert.equal(other.comet, null, `class ${classId}`);
  }
  const close = one(blow({ sourceKindled: true, target: { x: 166, y: 200 } }));
  assert.equal(close.shape, "cleave");
  assert.equal(close.kindled, true);
  assert.equal(close.comet, null);
  const drawn = draw(close, mount());
  assert.equal(
    part(drawn.underlay, "cleave-main")?.getAttribute("data-kindled"),
    "true",
  );
  assert.equal(part(drawn.group, "comet-ring"), null);
  assert.equal(drawn.animationSpecs.length, 2);
});

test("Agent POV: the Comet Bolt reads only its event and starts at a leaver's Last Seen spot", () => {
  const leaver = {
    sourceEndpointPhase: "transition_start",
    sourceDisclosure: "redacted",
    sourceIdentity: null,
    sourceRadius: undefined,
    source: { x: 60, y: 120 },
  };
  const comet = one(blow({ ...leaver, sourceKindled: true }));
  assert.equal(comet.fromLastSeen, true);
  assert.ok(comet.comet);
  assert.deepEqual(comet.stroke.origin, one(blow(leaver)).stroke.origin);
  assert.deepEqual(
    comet.landing.origin,
    one(blow({ sourceKindled: true, ...leaver })).landing.origin,
  );
  const guarded = { event: blow({ sourceKindled: true }) };
  for (const name of ["scene", "researcherSpace", "researcher_space", "matchSummary"]) {
    Object.defineProperty(guarded, name, {
      get() {
        throw new Error(`records read ${name}`);
      },
    });
  }
  assert.ok(basics.records(guarded)[0].comet);
  assert.equal(
    JSON.stringify(
      basics.records({ event: blow({ sourceKindled: true }), audience: "agent_pov" }),
    ),
    JSON.stringify(
      basics.records({ event: blow({ sourceKindled: true }), audience: "researcher" }),
    ),
  );
});

test("update moves the drawn nodes to the new geometry", () => {
  const target = draw(one(blow()), mount({ settled: true }));
  const moved = basics.records({ event: blow({ target: { x: 300, y: 260 } }) });
  basics.update([target.group, target.connector, target.underlay], moved);
  assert.equal(
    part(target.group, "landing")?.getAttribute("transform"),
    `translate(${moved[0].landing.origin[0]} ${moved[0].landing.origin[1]})`,
  );
  assert.match(
    part(target.underlay, "stroke")?.getAttribute("transform") ?? "",
    new RegExp(`rotate\\(${moved[0].stroke.angle}\\)$`, "u"),
  );
});

test("the tooltip titles the blow with its labels and amount", () => {
  const descriptor = basics.tooltip(one(blow({ outputAmount: 15 })));
  assert.equal(
    descriptor?.title,
    "Arcane Blast · Mage · ID 0 → Mage · ID 5 · 15 Damage",
  );
  assert.ok(descriptor?.summary);
});

test("basics.css uses only colour tokens and no CSS animation", async () => {
  const css = await readFile(
    new URL("../src/styles/basics.css", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(
    css,
    /#[0-9a-f]{3,8}\b|rgb\(|animation[\w-]*\s*:|transition[\w-]*\s*:|@keyframes|url\(|gradient/iu,
  );
  for (const token of ["--class-mage", "--starcore", "--forge-white", "--dawnlight"]) {
    assert.ok(css.includes(`var(${token})`), token);
  }
});
