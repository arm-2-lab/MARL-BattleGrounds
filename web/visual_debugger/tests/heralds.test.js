/**
 * @file Check the Herald Cards look (`src/looks/heralds.js`). records() is
 * pure data: a herald_card row names a multi takedown ("Double Takedown",
 * with "+N" only when every fall's points are known), a Trade (neutral rule)
 * or an Ultimate Takedown, gives no card for a plain single fall, counts only
 * the falls the row names (the plan already limits them to what an Agent POV
 * team saw), holds the previous tick's card when the current row gives none,
 * and gives nothing with Match Moments off or the row's paint part off. The
 * plate sits in the middle half of the top band, becomes a one-line band
 * when it cannot fit, and always stays in the top band: when it would touch
 * a champion it is told to avoid it moves sideways (or shrinks) along the
 * band, never to the bottom. The lineup draws medallions only from the given
 * authorized scene; the verdict shows in the Replay Viewer and at live game
 * over (BattleClient and DevClient), never for another product, and draws
 * exactly the Honours its caller passes (the host limits Agent POV Honours to
 * the viewing team); full cards turn into bands with Full Lineup And Verdict off; the
 * state chip reads "Match Point · Needs N" or "N Ticks Left". build() holds the
 * END state, pushes only fill "backwards" entry animations from the Fall beat
 * (none when settled or held) and draws export-safe SVG.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { DERIVED_EVENT_PAINT_PARTS } from "../src/choreography-plan.js";
import * as heralds from "../src/looks/heralds.js";

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
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      classList: {
        contains: (/** @type {string} */ name) =>
          (attributes.get("class") ?? "").split(" ").includes(name),
      },
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

const bounds = { left: 0, top: 0, right: 1200, bottom: 700, width: 1200, height: 700 };

/**
 * @param {number} victimTeam
 * @param {number | null} points
 * @param {string} id
 * @param {Record<string, any>} [extra]
 */
function fall(victimTeam, points, id, extra = {}) {
  return {
    victim: { title: `Warrior · ID ${id}`, classId: 2, teamId: victimTeam },
    killers: [],
    killingTeamId: 3 - victimTeam,
    points,
    ...extra,
  };
}

/** @param {any[]} falls */
function heraldRow(falls) {
  return {
    kind: "herald_card",
    eventId: "t:herald",
    count: falls.length,
    falls,
    paintParts: { effect: true },
    phaseStart: 960,
    phaseEnd: 2400,
  };
}

test("a herald names multi takedowns, trades and Ultimate takedowns, and no plain single fall", () => {
  const double = heralds.records({
    event: heraldRow([fall(2, 1, "6"), fall(2, 2, "7")]),
    bounds,
  });
  assert.equal(double.length, 1);
  assert.ok(Object.isFrozen(double) && Object.isFrozen(double[0]));
  assert.equal(double[0].kind, "herald-plate");
  assert.equal(double[0].title, "Double Takedown");
  assert.equal(double[0].eyebrow, "Team A +3");
  assert.equal(double[0].ruleTeam, "a");
  assert.equal(double[0].medallions.length, 2);
  const unknownPoints = heralds.records({
    event: heraldRow([fall(2, null, "6"), fall(2, 1, "7"), fall(2, 1, "8")]),
    bounds,
  });
  assert.equal(unknownPoints[0].title, "Triple Takedown");
  assert.equal(unknownPoints[0].eyebrow, "Team A");
  const trade = heralds.records({
    event: heraldRow([fall(1, 1, "1"), fall(2, 1, "6")]),
    bounds,
  });
  assert.equal(trade[0].title, "Trade");
  assert.equal(trade[0].eyebrow, "Team A Lost 1 · Team B Lost 1");
  assert.equal(trade[0].ruleTeam, null);
  assert.equal(trade[0].dividerAfter, 1);
  assert.deepEqual(
    heralds.records({ event: heraldRow([fall(2, 1, "6")]), bounds }),
    [],
  );
  const ultimate = heralds.records({
    event: heraldRow([fall(2, 1, "6", { ultimate: { classId: 5 } })]),
    bounds,
  });
  assert.equal(ultimate[0].title, "Ultimate Takedown");
  assert.match(ultimate[0].eyebrow, / · Team A \+1$/u);
  assert.equal(ultimate[0].diamondClassKey, "priest");
  // The card counts only the falls its row names (the plan keeps only seen falls).
  assert.equal(
    heralds.records({ event: heraldRow([fall(2, 1, "6"), fall(2, 1, "7")]) })[0].fallen
      .length,
    2,
  );
});

test("switches, paint parts and the previous tick decide whether a herald shows", () => {
  const row = heraldRow([fall(2, 1, "6"), fall(2, 1, "7")]);
  assert.deepEqual(
    heralds.records({ event: row, bounds, filters: { match_moments: false } }),
    [],
  );
  assert.deepEqual(
    heralds.records({ event: { ...row, paintParts: { effect: false } }, bounds }),
    [],
  );
  const held = heralds.records({
    event: heraldRow([fall(2, 1, "8")]),
    previous: row,
    bounds,
  });
  assert.equal(held[0].held, true);
  assert.equal(held[0].title, "Double Takedown");
  const fresh = heralds.records({ event: row, previous: row, bounds });
  assert.equal(fresh[0].held, false);
  assert.deepEqual(heralds.records({ event: { kind: "activation" } }), []);
  assert.deepEqual(heralds.records({}), []);
});

test("a plate sits in the middle half of the top band and becomes a band when it cannot fit", () => {
  const [plate] = heralds.records({
    event: heraldRow([fall(2, 1, "6"), fall(2, 1, "7")]),
    bounds,
  });
  const centre = plate.x + (plate.width * plate.scale) / 2;
  assert.ok(Math.abs(centre - 600) < 0.01);
  assert.equal(plate.y, heralds.FOOTPRINT.topGap);
  assert.equal(plate.height, 40);
  assert.ok(plate.x >= 300 - heralds.FOOTPRINT.plate.tailReach);
  // Too wide for the middle half, the plate looks along the whole band.
  const [smaller] = heralds.records({
    event: heraldRow([fall(2, 1, "6"), fall(2, 1, "7")]),
    bounds: { ...bounds, right: 260, width: 260 },
  });
  assert.equal(smaller.kind, "herald-plate");
  assert.equal(smaller.scale, 0.9);
  const narrow = { ...bounds, right: 200, width: 200 };
  const [band] = heralds.records({
    event: heraldRow([fall(2, 1, "6"), fall(2, 1, "7")]),
    bounds: narrow,
  });
  assert.equal(band.kind, "herald-band");
  assert.equal(band.text, "Double Takedown · Team A +2");
  assert.equal(band.height, 22);
  const top = { x: 600, y: 30, radius: 20 };
  const [moved] = heralds.records({
    event: heraldRow([fall(2, 1, "6"), fall(2, 1, "7")]),
    bounds,
    avoid: [top],
  });
  assert.equal(moved.place, "top");
  assert.equal(moved.y, heralds.FOOTPRINT.topGap);
  const left = moved.x - heralds.FOOTPRINT.plate.tailReach * moved.scale;
  const right =
    moved.x + (moved.width + heralds.FOOTPRINT.plate.tailReach) * moved.scale;
  assert.ok(right < 600 - 23 || left > 600 + 23, "moved sideways clear of the body");
  const [clear] = heralds.records({
    event: heraldRow([fall(2, 1, "6"), fall(2, 1, "7")]),
    bounds,
    avoid: [{ x: 100, y: 30, radius: 20 }],
  });
  assert.equal(clear.place, "top");
  assert.ok(Math.abs(clear.x + (clear.width * clear.scale) / 2 - 600) < 0.01);
});

test("the lineup shows only the champions in the given scene, as a card or a band", () => {
  const scene = {
    agents: [
      { public_agent_id: "2", class_id: 1, team_id: 1 },
      { public_agent_id: "1", class_id: 5, team_id: 1 },
    ],
  };
  const [card] = heralds.records({
    part: "lineup",
    scene,
    bounds,
    mapTitle: "Teutoburg Forest",
    mapLabel: "Team Deathmatch · Map 47",
    pointsToWin: 20,
    tickLimit: 300,
    redZoneDoubles: true,
  });
  assert.equal(card.kind, "herald-lineup");
  assert.deepEqual(
    card.columns[0].medallions.map(
      (/** @type {any} */ medallion) => medallion.classKey,
    ),
    ["priest", "mage"],
  );
  assert.equal(card.columns[1].medallions.length, 0);
  assert.deepEqual(card.rules, [
    "First To 20 Wins · Draw At Tick 300",
    "A Death In Its Own Red Zone Scores 2",
  ]);
  const [band] = heralds.records({
    part: "lineup",
    scene,
    bounds,
    mapTitle: "Teutoburg Forest",
    filters: { moment_full_cards: false },
  });
  assert.equal(band.kind, "herald-band");
  assert.deepEqual(band.crests, ["a", "b"]);
  assert.deepEqual(
    heralds.records({ part: "lineup", scene, bounds, mapTitle: "" }),
    [],
  );
});

test("the verdict shows in the Replay Viewer and at live game over, with the Honours its caller passes", () => {
  const verdict = {
    scores: { 1: 20, 2: 13 },
    winnerTeamId: 1,
    tick: 245,
    basis: "threshold",
    pointsToWin: 20,
    honours: [{ label: "Kill Contributions", value: "9", classId: 4, teamId: 1 }],
  };
  assert.deepEqual(
    heralds.records({ part: "verdict", product: "unknown_product", verdict, bounds }),
    [],
  );
  const [live] = heralds.records({
    part: "verdict",
    product: "combat_debugger",
    verdict,
    bounds,
  });
  assert.equal(live.kind, "herald-verdict");
  assert.equal(live.title, "Team A Victory");
  const [oracle] = heralds.records({
    part: "verdict",
    product: "replay_viewer",
    verdict,
    bounds,
  });
  assert.equal(oracle.eyebrow, "Tick 245 · Team A Reached 20");
  assert.equal(oracle.title, "Team A Victory");
  assert.deepEqual(
    oracle.columns.map((/** @type {any} */ column) => column.result),
    ["Victory", "Defeat"],
  );
  assert.equal(oracle.honours.length, 1);
  const [pov] = heralds.records({
    part: "verdict",
    product: "replay_viewer",
    audience: "agent_pov",
    verdict,
    bounds,
  });
  assert.equal(pov.honours.length, 1);
  const [draw] = heralds.records({
    part: "verdict",
    product: "replay_viewer",
    verdict: {
      ...verdict,
      winnerTeamId: null,
      basis: "horizon",
      scores: { 1: 19, 2: 11 },
      tick: 300,
    },
    bounds,
  });
  assert.equal(draw.eyebrow, "Tick 300 · No Team Reached 20");
  assert.equal(draw.lead, "A Lead At Tick 300 Wins Nothing");
});

test("the state chip names Match Point or the ticks left", () => {
  assert.equal(
    heralds.records({
      part: "state",
      state: { kind: "match_point", teamIds: [2], needs: 2 },
      bounds,
    })[0].text,
    "Match Point · Needs 2",
  );
  assert.equal(
    heralds.records({
      part: "state",
      state: { kind: "ticks_left", left: 1 },
      bounds,
    })[0].text,
    "1 Tick Left",
  );
  assert.deepEqual(
    heralds.records({ part: "state", state: { kind: "other" }, bounds }),
    [],
  );
});

test("build holds END, animates entry from the Fall beat only while playing, and is export safe", () => {
  const doc = fakeDocument();
  const records = heralds.records({
    event: heraldRow([fall(1, 1, "1"), fall(2, 1, "6")]),
    bounds,
  });
  /** @param {boolean} settled */
  const mountFor = (settled) => ({
    group: doc.createElementNS("", "g"),
    connector: doc.createElementNS("", "g"),
    underlay: null,
    plan: { epochKey: "e", phases: {}, beats: {} },
    options: { motionMode: /** @type {const} */ ("normal"), settled },
    animationSpecs: /** @type {any[]} */ ([]),
  });
  const playing = mountFor(false);
  heralds.build(records, doc, playing);
  assert.ok(playing.animationSpecs.length >= 1);
  for (const spec of playing.animationSpecs) {
    assert.equal(spec.options.fill, "backwards");
    assert.equal(spec.options.delay, 960);
    assert.equal(spec.keyframes[0].opacity, 0);
  }
  const card = byClass(playing.group, "herald")[0];
  assert.equal(card.getAttribute("data-herald-title"), "Trade");
  assert.equal(byClass(card, "herald__title")[0].textContent, "Trade");
  assert.equal(
    byClass(card, "herald__eyebrow")[0].textContent,
    "TEAM A LOST 1 · TEAM B LOST 1",
  );
  assert.equal(byClass(card, "herald__rule")[0].getAttribute("opacity"), "0.5");
  assert.equal(byClass(card, "herald__tail").length, 2);
  assert.equal(byClass(card, "herald__medallion").length, 2);
  assert.ok(descendants(card).length <= heralds.BUDGET.nodesPerRecord + 10);
  assertExportSafe(playing.group);
  const settled = mountFor(true);
  heralds.build(records, doc, settled);
  assert.equal(settled.animationSpecs.length, 0);
  const held = mountFor(false);
  heralds.build(
    heralds.records({
      event: heraldRow([]),
      previous: heraldRow([fall(2, 1, "6"), fall(2, 1, "7")]),
      bounds,
    }),
    doc,
    held,
  );
  assert.equal(held.animationSpecs.length, 0);
  const loose = heralds.build(
    [
      ...heralds.records({
        part: "state",
        state: { kind: "ticks_left", left: 9 },
        bounds,
      }),
      ...heralds.records({
        part: "verdict",
        product: "replay_viewer",
        verdict: {
          scores: { 1: 20, 2: 3 },
          winnerTeamId: 1,
          tick: 99,
          basis: "threshold",
          pointsToWin: 20,
        },
        bounds,
      }),
    ],
    doc,
  );
  assert.equal(loose.length, 2);
  heralds.update(loose, [
    { x: 5, y: 6, scale: 1 },
    { x: 7, y: 8, scale: 0.5 },
  ]);
  assert.equal(loose[1].getAttribute("transform"), "translate(7 8) scale(0.5)");
  for (const node of loose) assertExportSafe(node);
});

test("the herald registers the plan's derived paint part and gives tooltips only for cards", () => {
  const expected = DERIVED_EVENT_PAINT_PARTS.herald_card;
  assert.ok(
    heralds.PAINT_PARTS.some(
      (registration) =>
        JSON.stringify(registration.part) === JSON.stringify(expected.part) &&
        registration.filter === expected.filter,
    ),
  );
  assert.equal(heralds.tooltip({ kind: "anything" }), null);
  const [plate] = heralds.records({
    event: heraldRow([fall(2, 1, "6"), fall(2, 1, "7")]),
    bounds,
  });
  assert.equal(heralds.tooltip(plate)?.title, "Double Takedown");
});
