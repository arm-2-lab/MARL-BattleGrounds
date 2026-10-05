/**
 * @file Contracts of the V5 Ultimates look (src/looks/ultimates.js): records()
 * geometry for the Charge (war-path from START, 35% echo at 1.25 units, crunch,
 * open clasp facing the Warrior's END spot, "Charge" label), the Freezing Trap
 * (cord and hexagon snare), Burst (slivers and flare), Crippling Poison
 * (stiletto, resting blade, tether at 0.30 units) and Holy Word: Salvation
 * (the golden Gilded Bolt: a Dawnlight edge, a gold body and a Dawnlight core
 * in one 70% group, with no Starcore center line and a wide gold glow only
 * while playing; the telegraph and syllable rings, flood, rays and the echo
 * ring open at 12 o'clock keep their keys and timing); Agent POV rules (both START
 * anchors for the path, the four-arc Unseen Clasp for a hidden Warrior, no
 * clasp on a target unseen at END, Poison and Salvation only when both
 * champions were seen at END); switch-off rules (plain meaning mark, no
 * Soft Glow nodes); beat-clock timing through animationTracks (Charge rush
 * window, nothing else during the glide, Body Glide off, Longer Tick off,
 * reduced motion within 220 ms, motion off empty, fill "backwards"); build()
 * export safety, budgets, animation ids and settled installs; update() in
 * place; and the shared lasting-mark shapes.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { beatSchedule } from "../src/choreography-plan.js";
import * as ultimates from "../src/looks/ultimates.js";

/** @param {string} tag */
function fakeElement(tag) {
  /** @type {any} */
  const element = {
    tagName: tag,
    attributes: new Map(),
    children: /** @type {any[]} */ ([]),
    parent: null,
    textContent: "",
    ownerDocument: null,
    style: {},
    dataset: {},
    /** @param {string} name @param {string} value */
    setAttribute(name, value) {
      element.attributes.set(name, String(value));
    },
    /** @param {string} name */
    getAttribute(name) {
      return element.attributes.has(name) ? element.attributes.get(name) : null;
    },
    /** @param {string} name */
    hasAttribute(name) {
      return element.attributes.has(name);
    },
    /** @param {string} name */
    removeAttribute(name) {
      element.attributes.delete(name);
    },
    /** @param {...any} nodes */
    append(...nodes) {
      for (const each of nodes) {
        each.parent = element;
        element.children.push(each);
      }
    },
    remove() {
      if (element.parent) {
        element.parent.children = element.parent.children.filter(
          (/** @type {any} */ child) => child !== element,
        );
      }
    },
  };
  return element;
}

const doc = /** @type {any} */ ({
  createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
    fakeElement(tag),
});
doc.createElement = (/** @type {string} */ tag) => fakeElement(tag);
const createWithOwner = doc.createElementNS;
doc.createElementNS = (/** @type {string} */ ns, /** @type {string} */ tag) => {
  const element = createWithOwner(ns, tag);
  element.ownerDocument = doc;
  return element;
};

/** @param {any} root @returns {any[]} */
function descendants(root) {
  return [
    root,
    ...root.children.flatMap((/** @type {any} */ child) => descendants(child)),
  ];
}

/** @param {Record<string, any>} extra */
function charge(extra = {}) {
  return {
    kind: "activation",
    component: "ultimate",
    eventId: "e-charge",
    tokenId: "warrior_charge",
    sourcePresentationKey: "w",
    targetPresentationKey: "m",
    source: { x: 100, y: 100 },
    target: { x: 300, y: 100 },
    sourceRadius: 23,
    targetRadius: 23,
    unitPx: 46,
    paintParts: { ability: true, semantic: true },
    sourceEndpointPhase: "transition_start",
    targetEndpointPhase: "transition_start",
    ...extra,
  };
}

/** @param {string} tokenId @param {Record<string, any>} extra */
function paired(tokenId, extra = {}) {
  return {
    kind: "activation",
    component: "ultimate",
    eventId: `e-${tokenId}`,
    tokenId,
    sourcePresentationKey: "a",
    targetPresentationKey: "b",
    source: { x: 100, y: 200 },
    target: { x: 220, y: 200 },
    sourceRadius: 23,
    targetRadius: 23,
    unitPx: 46,
    paintParts: { ability: true, semantic: true },
    sourceEndpointPhase: "successor",
    targetEndpointPhase: "successor",
    ...extra,
  };
}

/** @param {any} record @returns {Map<string, any>} */
function nodesByKey(record) {
  /** @type {Map<string, any>} */
  const found = new Map();
  /** @param {ReadonlyArray<any>} trees */
  const walk = (trees) => {
    for (const tree of trees) {
      found.set(tree.attrs["data-ult-key"], tree);
      walk(tree.children ?? []);
    }
  };
  walk(record.over);
  walk(record.band);
  return found;
}

/** @param {Record<string, any> | null | undefined} event */
function drawsNothing(event) {
  const list = ultimates.records({ event });
  assert.equal(list.length, 1, "the look still owns the event");
  assert.equal(list[0].variant, "none");
  assert.deepEqual([list[0].band, list[0].over], [[], []]);
}

/** @param {Record<string, any>} event */
function only(event) {
  const list = ultimates.records({ event });
  assert.equal(list.length, 1);
  assert.ok(Object.isFrozen(list) && Object.isFrozen(list[0]));
  return /** @type {any} */ (list[0]);
}

test("records keeps only Ultimate activations", () => {
  assert.deepEqual(ultimates.records({}), []);
  assert.deepEqual(
    ultimates.records({ event: { kind: "activation", component: "basic" } }),
    [],
  );
  assert.deepEqual(ultimates.records({ event: { kind: "net_health" } }), []);
  drawsNothing({ kind: "activation", component: "ultimate" });
  assert.equal(only(charge()).variant, "charge");
  assert.equal(only(paired("hunter_trap")).variant, "trap");
  assert.equal(only(paired("rogue_poison")).variant, "poison");
  assert.equal(only(paired("holy_word")).variant, "salvation");
  assert.equal(
    only({ ...paired("mage_burst"), target: null, targetPresentationKey: null })
      .variant,
    "burst",
  );
});

test("a long Charge draws the war-path from the START rim, its echo, crunch, clasp and label", () => {
  const record = only(
    charge({ chargeLanding: { x: 254, y: 100 }, targetEnd: { x: 300, y: 130 } }),
  );
  assert.equal(record.hasPath, true);
  assert.equal(record.isLong, true);
  assert.equal(record.travel, 3.35);
  const nodes = nodesByKey(record);
  // The path frame starts on the Warrior's START rim, facing the landing.
  assert.equal(nodes.get("charge.frame").attrs.transform, "translate(123 100)");
  assert.equal(nodes.get("charge.echo").attrs.opacity, 0.35);
  assert.equal(nodes.get("charge.grow").attrs.opacity, 0);
  assert.equal(nodes.get("charge.hair").attrs.x2, 154);
  // Clasp on the target's END spot at 80%; crunch on the END rim facing the landing.
  assert.equal(nodes.get("charge.claspAt").attrs.transform, "translate(300 130)");
  assert.equal(nodes.get("charge.clasp").attrs.opacity, 0.8);
  assert.equal(nodes.get("charge.crunch").attrs.opacity, 0);
  const crunchAt = nodes.get("charge.crunchAt").attrs.transform;
  const [cx, cy] = crunchAt.match(/-?[\d.]+/gu).map(Number);
  assert.ok(Math.hypot(cx - 300, cy - 130) > 22.9 && cx < 300 && cy < 130);
  const label = nodes.get("charge.label");
  assert.equal(label.text, "Charge");
  assert.equal(label.attrs.x, 100);
  assert.ok(label.attrs.y > 100 + 23);
  assert.equal(label.attrs.opacity, 1);
  // Two rush rings? No: travel 3 gives three rings, unseen until play.
  assert.equal(nodes.get("charge.ring2").attrs.opacity, 0);
});

test("Charge path length rules: no path under 0.25 units, no echo under 1.25 units", () => {
  const near = only(charge({ chargeTravel: 0.1, chargeLanding: { x: 105, y: 100 } }));
  assert.equal(near.hasPath, false);
  assert.ok(!nodesByKey(near).has("charge.frame"));
  assert.ok(nodesByKey(near).has("charge.clasp"));
  assert.ok(nodesByKey(near).has("charge.label"));
  const mid = only(charge({ chargeTravel: 1, chargeLanding: { x: 146, y: 100 } }));
  assert.equal(mid.hasPath, true);
  assert.equal(mid.isLong, false);
  assert.ok(!nodesByKey(mid).has("charge.echo"));
  assert.ok(!nodesByKey(mid).has("charge.ring0"));
});

test("Agent POV Charge: no landing ends at the target's near rim; hidden Warrior gets the Unseen Clasp", () => {
  // No post-Charge anchor (Agent POV): the war-path ends at the target's near rim.
  const pov = only(charge());
  const hair = nodesByKey(pov).get("charge.hair");
  assert.equal(hair.attrs.x2, 300 - 23 - 123);
  // A hidden Warrior: four equal arcs, nothing else of the Warrior.
  const hidden = only(charge({ source: null }));
  assert.equal(hidden.unseenClasp, true);
  assert.deepEqual(hidden.band, []);
  const keys = [...nodesByKey(hidden).keys()];
  assert.deepEqual(keys.sort(), [
    "charge.clasp",
    "charge.claspArcs",
    "charge.claspAt",
    "charge.hit",
  ]);
  const arcs = nodesByKey(hidden).get("charge.claspArcs").attrs.d;
  assert.equal(arcs.match(/A /gu).length, 4);
  // A target not seen at END: no clasp and no crunch, the path from START stays.
  const gone = nodesByKey(only(charge({ targetEnd: null })));
  assert.ok(!gone.has("charge.clasp") && !gone.has("charge.crunch"));
  assert.ok(gone.has("charge.frame"));
  // No target at all: nothing.
  drawsNothing(charge({ target: null }));
});

test("Freezing Trap draws a dashed cord and a hexagon snare; a hidden Hunter leaves only the snare", () => {
  const nodes = nodesByKey(only(paired("hunter_trap")));
  const cord = nodes.get("trap.cord");
  assert.equal(cord.attrs.opacity, 0.7);
  assert.equal(cord.attrs.x1, 126);
  assert.equal(cord.attrs.x2, 194);
  assert.match(cord.attrs["stroke-dasharray"], /^3 2$/u);
  assert.equal(nodes.get("trap.snare").attrs.opacity, 0);
  assert.equal(nodes.get("trap.snarePath").attrs.d.match(/M /gu).length, 7);
  const hidden = nodesByKey(only(paired("hunter_trap", { source: null })));
  assert.ok(!hidden.has("trap.cord") && hidden.has("trap.snare"));
  drawsNothing(paired("hunter_trap", { target: null }));
});

test("Burst folds eight slivers into the Mage and flares; no caster means nothing", () => {
  const record = only({ ...paired("mage_burst"), target: null });
  const nodes = nodesByKey(record);
  assert.equal(nodes.get("burst.bodies").attrs.d.match(/M /gu).length, 8);
  assert.equal(nodes.get("burst.flare").attrs.opacity, 0);
  assert.deepEqual(record.band, []);
  drawsNothing({ ...paired("mage_burst"), source: null, target: null });
});

test("Poison and Salvation are drawn only when both champions were seen at END", () => {
  for (const tokenId of ["rogue_poison", "holy_word"]) {
    for (const phases of [
      { sourceEndpointPhase: "transition_start" },
      { targetEndpointPhase: "transition_start" },
    ]) {
      drawsNothing(paired(tokenId, phases));
    }
    drawsNothing(paired(tokenId, { source: null }));
  }
  // Oracle plans always mark both ends seen; pictures without phases are allowed.
  const {
    sourceEndpointPhase: _source,
    targetEndpointPhase: _target,
    ...bare
  } = paired("rogue_poison");
  assert.equal(only(bare).variant, "poison");
});

test("Poison rests its blade at 70% and adds a 30% tether only at 0.30 units or more", () => {
  const far = only(paired("rogue_poison"));
  const nodes = nodesByKey(far);
  assert.equal(nodes.get("poison.rest").attrs.opacity, 0.7);
  assert.equal(nodes.get("poison.strike").attrs.opacity, 0);
  assert.equal(nodes.get("poison.tether").attrs.opacity, 0.3);
  assert.equal(far.tether, true);
  // The victim's local frame points at the Rogue (angle 180 degrees here).
  assert.equal(
    nodes.get("poison.restFrame").attrs.transform,
    "translate(220 200) rotate(180)",
  );
  const near = only(paired("rogue_poison", { target: { x: 155, y: 200 } }));
  assert.equal(near.tether, false);
  assert.ok(!nodesByKey(near).has("poison.tether"));
});

test("Salvation keeps a 70% bolt and a 35% echo ring open at 12 o'clock; self-cast has no bolt", () => {
  const nodes = nodesByKey(only(paired("holy_word")));
  assert.equal(nodes.get("salv.core").attrs.opacity, 0.7);
  assert.equal(nodes.get("salv.bright").attrs.opacity, 0);
  const echo = nodes.get("salv.echo");
  assert.equal(echo.attrs.opacity, 0.35);
  // Starts 25 degrees right of 12 o'clock, on the ally, at 1.15R.
  const [x, y] = echo.attrs.d.match(/-?[\d.]+/gu).map(Number);
  assert.ok(Math.abs(Math.hypot(x - 220, y - 200) - 26.45) < 0.05);
  assert.ok(x > 220 && y < 200);
  for (const index of [0, 1, 2]) {
    assert.equal(nodes.get(`salv.ring${index}`).attrs.opacity, 0);
  }
  const self = only(
    paired("holy_word", { target: { x: 100, y: 200 }, targetPresentationKey: "a" }),
  );
  assert.equal(self.selfCast, true);
  assert.ok(!nodesByKey(self).has("salv.core"));
  assert.ok(nodesByKey(self).has("salv.echo"));
});

test("Salvation's line is the golden Gilded Bolt and its rings, flood, rays and echo ring keep their keys and timing", () => {
  const record = only(paired("holy_word"));
  const nodes = nodesByKey(record);
  const core = nodes.get("salv.core");
  assert.equal(core.tag, "g");
  assert.equal(core.attrs.opacity, 0.7);
  assert.deepEqual(
    core.children.map((/** @type {any} */ child) => [
      child.attrs["data-ult-key"],
      child.attrs.class,
    ]),
    [
      ["salv.boltEdge", "ult-salv__bolt-edge"],
      ["salv.boltBody", "ult-salv__bolt-body"],
      ["salv.boltCore", "ult-salv__bolt-core"],
    ],
  );
  // The three lines run the whole bolt, from the Priest's rim to the ally's rim.
  for (const line of core.children) {
    assert.equal(line.tag, "line");
    assert.equal(line.attrs.x2, 74);
  }
  assert.ok(!nodes.has("salv.centre"));
  assert.equal(nodes.get("salv.glow").attrs.opacity, 0.21);
  assert.equal(nodes.get("salv.glow").attrs.class, "ult-salv__glow ult-soft-glow");
  for (const key of [
    "salv.ring0",
    "salv.ring1",
    "salv.ring2",
    "salv.flood",
    "salv.rays",
    "salv.rayGlow",
    "salv.echo",
    "salv.targetHit",
  ]) {
    assert.ok(nodes.has(key), key);
  }
  const schedule = beatSchedule();
  const tracks = new Map(
    ultimates
      .animationTracks(record, schedule, "normal")
      .map((each) => [each.key, each]),
  );
  assert.deepEqual(
    [...tracks.keys()],
    [
      "salv.grow",
      "salv.core",
      "salv.bright",
      "salv.echo",
      "salv.ring0",
      "salv.ring1",
      "salv.ring2",
      "salv.flood",
      "salv.rays",
    ],
  );
  // The bolt keeps the line's beats: it brightens at 160 ms and rests at 70%.
  assert.deepEqual(
    record.cues.glide
      .find((/** @type {any} */ cue) => cue.key === "salv.core")
      .frames.map((/** @type {any} */ frame) => [frame.t, frame.o]),
    [
      [159, 0],
      [160, 1],
      [640, 1],
      [960, 0.7],
    ],
  );
  // Soft Glow off leaves the bolt and drops its bright group and that cue.
  const plain = only(
    paired("holy_word", {
      paintParts: { ability: true, semantic: true, softGlow: false },
    }),
  );
  assert.ok(nodesByKey(plain).has("salv.boltCore"));
  assert.ok(!nodesByKey(plain).has("salv.bright"));
  assert.ok(
    !ultimates
      .animationTracks(plain, schedule, "normal")
      .some((each) => each.key === "salv.bright"),
  );
});

test("switches: ability off keeps a plain meaning mark, all off draws nothing, Soft Glow off drops glow", () => {
  const plain = only(
    charge({
      paintParts: { ability: false, semantic: true },
      impactSemantic: "damage",
    }),
  );
  assert.equal(plain.variant, "plain");
  drawsNothing(charge({ paintParts: { ability: false, semantic: false } }));
  const glowKeys = ["poison.glow", "salv.glow", "salv.rayGlow"];
  for (const tokenId of ["rogue_poison", "holy_word"]) {
    const on = nodesByKey(only(paired(tokenId)));
    const off = nodesByKey(
      only(
        paired(tokenId, {
          paintParts: { ability: true, semantic: true, softGlow: false },
        }),
      ),
    );
    assert.ok(
      glowKeys.some((key) => on.has(key)),
      tokenId,
    );
    assert.ok(!glowKeys.some((key) => off.has(key)), tokenId);
  }
});

test("timing comes from the beat schedule: the Charge rides the rush, nothing else shows during the glide", () => {
  const schedule = beatSchedule();
  const record = only(charge({ chargeLanding: { x: 254, y: 100 } }));
  const tracks = new Map(
    ultimates
      .animationTracks(record, schedule, "normal")
      .map((each) => [each.key, each]),
  );
  const grow = tracks.get("charge.grow");
  assert.equal(grow?.options.delay, schedule.windows.chargeRush?.[0]);
  const rushEnd =
    Number(grow?.options.delay) +
    Number(grow?.options.duration) * Number(grow?.keyframes[1].offset);
  assert.ok(Math.abs(rushEnd - Number(schedule.windows.chargeRush?.[1])) < 0.5);
  // The crunch lands when every body stands on its END spot (the strike start).
  const crunch = tracks.get("charge.crunch");
  assert.equal(Number(crunch?.options.delay) + 1, schedule.windows.strike[0]);
  for (const track of tracks.values()) {
    assert.equal(track.options.fill, "backwards");
  }
  for (const tokenId of ["hunter_trap", "rogue_poison", "holy_word"]) {
    for (const track of ultimates.animationTracks(
      only(paired(tokenId)),
      schedule,
      "normal",
    )) {
      assert.ok(Number(track.options.delay) >= schedule.windows.strike[0], track.key);
    }
  }
  // Body Glide off: real time is blow time and the Charge crunches at 360 ms.
  const still = beatSchedule({ visualFilters: { stride_glide: false } });
  const plain = ultimates.animationTracks(record, still, "normal");
  assert.equal(plain.find((each) => each.key === "charge.crunch")?.options.delay, 359);
  assert.ok(plain.some((each) => each.key === "charge.ring0"));
  assert.ok(
    !ultimates
      .animationTracks(record, schedule, "normal")
      .some((each) => each.key === "charge.ring0"),
  );
  // Longer Tick off: blows play faster after the strike starts.
  const short = beatSchedule({ visualFilters: { scholar_long_tick: false } });
  const label = (/** @type {any} */ beats) =>
    ultimates
      .animationTracks(record, beats, "normal")
      .find((each) => each.key === "charge.label");
  assert.ok(
    Number(label(short)?.options.delay) < Number(label(schedule)?.options.delay),
  );
});

test("reduced motion fades within 220 ms with no movement; motion off animates nothing", () => {
  const schedule = beatSchedule();
  for (const event of [
    charge({ chargeLanding: { x: 254, y: 100 } }),
    paired("hunter_trap"),
    { ...paired("mage_burst"), target: null },
    paired("rogue_poison"),
    paired("holy_word"),
  ]) {
    const record = only(event);
    const tracks = ultimates.animationTracks(record, schedule, "reduced");
    assert.ok(tracks.length > 0, event.tokenId);
    for (const track of tracks) {
      assert.ok(Number(track.options.delay) + Number(track.options.duration) <= 220.01);
      const moves = new Set(track.keyframes.map((frame) => frame.transform ?? "none"));
      assert.ok(moves.size <= 1, `${track.key} does not move`);
    }
    assert.deepEqual(ultimates.animationTracks(record, schedule, "off"), []);
  }
});

test("build draws export-safe SVG within the budget and hands the choreographer backwards-filled animations", () => {
  const schedule = beatSchedule();
  const allowed = new Set(["g", "path", "line", "circle", "text"]);
  for (const event of [
    charge({ chargeLanding: { x: 254, y: 100 } }),
    charge({ chargeLanding: { x: 254, y: 100 }, source: null }),
    paired("hunter_trap"),
    { ...paired("mage_burst"), target: null },
    paired("rogue_poison"),
    paired("holy_word"),
    charge({
      paintParts: { ability: false, semantic: true },
      impactSemantic: "damage",
    }),
  ]) {
    const list = ultimates.records({ event });
    /** @type {any[]} */
    const animationSpecs = [];
    const mount = /** @type {any} */ ({
      group: fakeElement("g"),
      connector: fakeElement("g"),
      underlay: fakeElement("g"),
      plan: { epochKey: "epoch", beats: schedule, phases: {} },
      options: { motionMode: "normal", settled: false },
      animationSpecs,
    });
    const nodes = ultimates.build(list, doc, mount);
    assert.equal(nodes.length, 3);
    const drawn = [mount.group, mount.underlay].flatMap((root) =>
      descendants(root).slice(1),
    );
    assert.ok(
      drawn.length <= ultimates.BUDGET.nodesPerRecord,
      `${event.tokenId} ${drawn.length}`,
    );
    assert.ok(animationSpecs.length <= ultimates.BUDGET.animationsPerRecord);
    for (const element of drawn) {
      if (element.tagName === "g" && element.attributes.has("data-ult-root")) continue;
      assert.ok(
        allowed.has(element.tagName) || element.tagName === "path",
        element.tagName,
      );
      assert.ok(!element.attributes.has("id"), "no ids");
      for (const value of element.attributes.values()) {
        assert.doesNotMatch(value, /url\(|gradient|mask|filter|<image/iu);
      }
    }
    for (const spec of animationSpecs) {
      assert.equal(spec.options.fill, "backwards");
      assert.match(spec.id, new RegExp(`^mbg:epoch:${event.eventId}:ult\\.`, "u"));
    }
    // A settled install (paused frame, export) adds no animation.
    /** @type {any[]} */
    const settledSpecs = [];
    ultimates.build(list, doc, {
      ...mount,
      group: fakeElement("g"),
      underlay: fakeElement("g"),
      options: { motionMode: "normal", settled: true },
      animationSpecs: settledSpecs,
    });
    assert.deepEqual(settledSpecs, []);
  }
  // Strike Band marks go under the bodies; the rest above them.
  const mount = /** @type {any} */ ({
    group: fakeElement("g"),
    connector: fakeElement("g"),
    underlay: fakeElement("g"),
    plan: { epochKey: "x", phases: {} },
    options: { motionMode: "off", settled: true },
    animationSpecs: [],
  });
  ultimates.build(ultimates.records({ event: paired("rogue_poison") }), doc, mount);
  assert.equal(mount.underlay.children[0].getAttribute("data-ult-root"), "band");
  assert.equal(mount.group.children[0].getAttribute("data-ult-variant"), "poison");
  assert.deepEqual(ultimates.build([], /** @type {any} */ (null)), []);
});

test("update moves nodes in place and rebuilds only when the drawing changes shape", () => {
  const mount = /** @type {any} */ ({
    group: doc.createElementNS("", "g"),
    connector: fakeElement("g"),
    underlay: fakeElement("g"),
    plan: { epochKey: "x", phases: {} },
    options: { motionMode: "off", settled: true },
    animationSpecs: [],
  });
  const first = ultimates.records({ event: paired("holy_word") });
  ultimates.build(first, doc, mount);
  const ring = descendants(mount.group).find(
    (element) => element.getAttribute("data-ult-key") === "salv.at",
  );
  const moved = ultimates.records({
    event: paired("holy_word", {
      source: { x: 110, y: 210 },
      target: { x: 230, y: 210 },
    }),
  });
  ultimates.update([mount.group, mount.connector, mount.underlay], moved);
  const after = descendants(mount.group).find(
    (element) => element.getAttribute("data-ult-key") === "salv.at",
  );
  assert.equal(after, ring);
  assert.equal(after.getAttribute("transform"), "translate(230 210)");
  // A self-cast drops the bolt, so the look's roots are rebuilt.
  const self = ultimates.records({
    event: paired("holy_word", {
      target: { x: 100, y: 200 },
      targetPresentationKey: "a",
    }),
  });
  ultimates.update([mount.group, mount.connector, mount.underlay], self);
  const keys = descendants(mount.underlay).map((element) =>
    element.getAttribute("data-ult-key"),
  );
  assert.ok(!keys.includes("salv.core") && keys.includes("salv.echo"));
  assert.equal(mount.group.children.length, 1);
  assert.equal(ultimates.update([], first), undefined);
});

test("tooltips explain the Ultimate from the plan event only", () => {
  const descriptor = /** @type {any} */ (ultimates.tooltip(only(paired("holy_word"))));
  assert.equal(typeof descriptor?.title, "string");
  assert.ok(descriptor.title.length > 0);
});

test("lasting-mark shapes: shell with icicles, crack, six shards, thaw ghost and five runes", () => {
  const center = { x: 50, y: 50 };
  const shell = ultimates.frostShellShape(center, 23, 4);
  assert.equal(shell.shell.match(/M /gu)?.length, 1);
  assert.equal(shell.icicles.match(/M /gu)?.length, 4);
  assert.equal(
    ultimates.frostShellShape(center, 23, 2).icicles.match(/M /gu)?.length,
    2,
  );
  assert.equal(
    ultimates.frostShellShape(center, 23, 9).icicles.match(/M /gu)?.length,
    4,
  );
  assert.equal(ultimates.iceCrackPath(center, 23, 0).match(/L /gu)?.length, 4);
  assert.equal(ultimates.iceShardsPath(center, 23).match(/M /gu)?.length, 6);
  assert.match(ultimates.thawGhostPath(center, 23), /^M .* Z$/u);
  const runes = ultimates.burstRuneShapes(center, 23, 5);
  assert.equal(runes.runes.match(/M /gu)?.length, 5);
  assert.equal(ultimates.burstRuneShapes(center, 23, 0).runes, "");
  assert.deepEqual(runes.ember, { x: 50, y: 50 });
  assert.equal(runes.emberRadius, 6.9);
});
