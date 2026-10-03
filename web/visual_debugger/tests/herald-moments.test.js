/**
 * @file Check the Herald Cards moment finder and its cards
 * (`src/looks/heralds.js`: heraldMoments, cardMoments, momentEntersAtSettle,
 * records and build). Contracts: the store's moments (Team Wiped, First
 * Blood, Match Point, Multi Takedown, takedown runs, Rescue, Ultimate
 * Takedown, Trade, Scores Level, Ultimate Combo, Ticks Left) are found from
 * the row's facts alone, in the store's headline order (RANK), with the
 * store's words and glyphs; a card names the headline and at most two more
 * moments in its eyebrow; First Blood needs the fresh start and a 0-0 start
 * of the tick, Scores Level a lead of 3 or more before, Match Point a score
 * that rose this tick; the Death Announcer switch off drops the moments born
 * from a fall; a plain single fall still gives no card; Rescue, Ticks Left and
 * a Combo whose target lived wait for the tick to settle; Scores Level and
 * Ultimate Combo are one-line chips; a tier-1 card draws its Herald Rule
 * behind the plate and grows it from the card's centre; every card names
 * its moments in data-herald-moments; a one-line band or chip fades in from
 * the row's phaseStart; the largest cards stay inside BUDGET and draw only
 * export-safe nodes; rowBudget, the row's allowance the plan charges, covers
 * every card it can draw plus the card's own group, which the painter counts
 * as a kept node (Match Point near the end of a match included), and the whole
 * BUDGET for a row that names no moment and may show a held card.
 */
import assert from "node:assert/strict";
import test from "node:test";

import * as heralds from "../src/looks/heralds.js";

const bounds = { left: 0, top: 0, right: 1200, bottom: 700, width: 1200, height: 700 };

/**
 * @param {number} victimTeam
 * @param {string} id
 * @param {Record<string, any>} [extra]
 */
function fall(victimTeam, id, extra = {}) {
  return {
    victim: { title: `Mage · ID ${id}`, classId: 1, teamId: victimTeam },
    killers: [],
    killingTeamId: 3 - victimTeam,
    points: 1,
    ...extra,
  };
}

/** @param {Record<string, any>} [facts] */
function row(facts = {}) {
  return {
    kind: "herald_card",
    eventId: "t:herald",
    falls: [],
    paintParts: { effect: true },
    phaseStart: 960,
    phaseEnd: 2400,
    ...facts,
  };
}

/**
 * @param {Record<string, any>} event
 * @param {Readonly<Record<string, boolean>> | null} [filters]
 */
function kinds(event, filters = null) {
  return heralds.heraldMoments(event, filters).map((moment) => moment.kind);
}

/** @param {Record<string, any>} event @returns {Record<string, any>} */
function card(event, filters = {}) {
  const [record] = heralds.records({ event, bounds, filters });
  assert.ok(record, "a card");
  return record;
}

const fresh = {
  before: [0, 0],
  after: [2, 0],
  pointsToWin: 20,
  tick: 24,
  tickLimit: 300,
};

test("First Blood heads a fresh match's first fall, with the other moments in its eyebrow", () => {
  const event = row({
    falls: [fall(2, "5"), fall(2, "6")],
    freshStart: true,
    match: fresh,
    runs: [{ teamId: 1, takedowns: 3 }],
  });
  assert.deepEqual(kinds(event), ["firstBlood", "multi", "run"]);
  const first = card(event);
  assert.equal(first.kind, "herald-plate");
  assert.equal(first.title, "First Blood");
  assert.equal(first.eyebrow, "Team A +2 · Double Takedown · 3 Unanswered");
  assert.equal(first.ruleTeam, "a");
  assert.ok(first.tierRule, "a tier-1 card has its Herald Rule");
  assert.deepEqual(first.kinds, ["firstBlood", "multi", "run"]);
  // Not a fresh start, or a tick that began above 0-0: no First Blood.
  assert.deepEqual(kinds({ ...event, freshStart: false }), ["multi", "run"]);
  assert.deepEqual(kinds({ ...event, match: { ...fresh, before: [1, 0] } }), [
    "multi",
    "run",
  ]);
  // Both teams lost one: "Traded", with the neutral rule.
  const traded = card(
    row({ falls: [fall(1, "1"), fall(2, "6")], freshStart: true, match: fresh }),
  );
  assert.equal(traded.title, "First Blood");
  assert.equal(traded.eyebrow, "Traded · Trade");
  assert.equal(traded.ruleTeam, null);
});

test("Team Wiped and Match Point outrank the takedowns, with their glyphs", () => {
  const wiped = card(
    row({
      falls: [fall(2, "5"), fall(2, "6")],
      wiped: [
        {
          teamId: 2,
          members: [
            { classId: 1, teamId: 2 },
            { classId: 2, teamId: 2 },
          ],
        },
      ],
    }),
  );
  assert.equal(wiped.title, "Team Wiped");
  assert.equal(wiped.eyebrow, "Team B · Double Takedown");
  assert.equal(wiped.ruleTeam, "a");
  assert.equal(wiped.medallions.length, 2);
  // Team Wiped's medallions overlap by 8 px (18 px apart).
  const placed = wiped.glyphs.filter(
    (/** @type {any} */ glyph) => glyph.type === "medallion",
  );
  assert.equal(placed[1].x - placed[0].x, 18);
  // A wiped team that lost nobody this tick is not named.
  assert.deepEqual(
    kinds(
      row({
        falls: [fall(1, "1")],
        wiped: [{ teamId: 2, members: [{ classId: 1, teamId: 2 }] }],
      }),
    ),
    [],
  );
  const point = card(
    row({
      falls: [fall(2, "5"), fall(2, "6")],
      match: {
        before: [16, 10],
        after: [18, 10],
        pointsToWin: 20,
        tick: 150,
        tickLimit: 300,
      },
    }),
  );
  assert.equal(point.title, "Match Point");
  assert.equal(point.eyebrow, "Team A Needs 2 Points · Double Takedown");
  assert.equal(point.ruleTeam, null);
  assert.deepEqual(
    point.glyphs.map((/** @type {any} */ glyph) => [glyph.type, glyph.size]),
    [["crest", 22]],
  );
  const both = heralds.heraldMoments(
    row({ match: { before: [18, 17], after: [19, 18], pointsToWin: 20, tick: 200 } }),
  );
  assert.deepEqual(both[0], { kind: "matchPoint", teamIds: [1, 2], needs: 1 });
  assert.equal(
    card(
      row({ match: { before: [18, 17], after: [19, 18], pointsToWin: 20, tick: 200 } }),
    ).eyebrow,
    "Both Teams Need 1 Point",
  );
  // A score that did not rise this tick is a state, not a new Match Point.
  assert.deepEqual(
    kinds(row({ match: { before: [19, 3], after: [19, 3], pointsToWin: 20 } })),
    [],
  );
});

test("runs, Rescue, Ultimate Takedown and Trade keep the store's order and words", () => {
  const big = card(
    row({ falls: [fall(2, "5"), fall(2, "6")], runs: [{ teamId: 1, takedowns: 5 }] }),
  );
  assert.equal(big.title, "5 Unanswered");
  assert.equal(big.eyebrow, "Team A · Double Takedown");
  assert.deepEqual(
    big.glyphs.map((/** @type {any} */ glyph) => glyph.type),
    ["crest"],
  );
  assert.equal("dots" in big, false);
  const rescue = card(
    row({
      rescues: [
        {
          champion: { classId: 3, teamId: 2, title: "Hunter · ID 7" },
          byUltimate: true,
        },
      ],
    }),
  );
  assert.equal(rescue.title, "Rescue");
  assert.equal(rescue.eyebrow, "Holy Word: Salvation");
  assert.equal(rescue.ruleTeam, "b");
  assert.equal(rescue.glyphs[0].type, "rescue");
  assert.equal(
    card(row({ rescues: [{ champion: { classId: 3, teamId: 2 }, byUltimate: false }] }))
      .eyebrow,
    "Lesser Heal",
  );
  const ultimate = card(
    row({
      falls: [
        fall(2, "6", { ultimate: { classId: 4, publicAgentId: "3" } }),
        fall(1, "1"),
      ],
    }),
  );
  assert.equal(ultimate.title, "Ultimate Takedown");
  assert.equal(ultimate.eyebrow, "Crippling Poison · Team A +1 · Trade");
  assert.equal(ultimate.diamondClassKey, "rogue");
  // A plain single fall is still told only by the Fallen feed.
  assert.deepEqual(
    heralds.records({ event: row({ falls: [fall(2, "6")] }), bounds }),
    [],
  );
});

test("Scores Level, Ultimate Combo and Ticks Left need their facts and wait as the store says", () => {
  const level = row({
    falls: [fall(1, "1")],
    match: {
      before: [8, 7],
      after: [8, 8],
      pointsToWin: 20,
      tick: 120,
      tickLimit: 300,
    },
    largestLeadBefore: 3,
  });
  const chip = card(level);
  assert.equal(chip.kind, "herald-chip");
  assert.equal(chip.text, "Scores Level · 8 – 8");
  assert.deepEqual(chip.kinds, ["level"]);
  assert.deepEqual(kinds({ ...level, largestLeadBefore: 2 }), []);
  const combo = row({
    combos: [
      {
        casters: [
          { classId: 2, publicAgentId: "1", teamId: 1 },
          { classId: 3, publicAgentId: "2", teamId: 1 },
        ],
        target: { classId: 1, teamId: 2 },
        fell: false,
      },
    ],
  });
  const comboChip = card(combo);
  assert.equal(comboChip.text, "Ultimate Combo · Charge + Freezing Trap");
  assert.deepEqual(comboChip.diamonds, ["warrior", "hunter"]);
  const [comboMoment] = heralds.heraldMoments(combo);
  assert.equal(heralds.momentEntersAtSettle(comboMoment), true);
  assert.equal(heralds.momentEntersAtSettle({ ...comboMoment, fell: true }), false);
  const moon = card(
    row({
      match: {
        before: [5, 9],
        after: [5, 9],
        pointsToWin: 20,
        tick: 270,
        tickLimit: 300,
      },
    }),
  );
  assert.equal(moon.title, "30 Ticks Left");
  assert.equal(moon.eyebrow, "Draw If No Team Reaches 20");
  assert.equal(moon.glyphs[0].type, "crescent");
  assert.equal(
    card(
      row({
        match: {
          before: [5, 9],
          after: [5, 9],
          pointsToWin: 20,
          tick: 299,
          tickLimit: 300,
        },
      }),
    ).title,
    "1 Tick Left",
  );
  assert.deepEqual(
    kinds(
      row({
        match: {
          before: [5, 9],
          after: [5, 9],
          pointsToWin: 20,
          tick: 271,
          tickLimit: 300,
        },
      }),
    ),
    [],
  );
  assert.deepEqual(
    kinds(
      row({
        match: {
          before: [20, 9],
          after: [20, 9],
          pointsToWin: 20,
          tick: 270,
          tickLimit: 300,
        },
      }),
    ),
    [],
  );
  for (const moment of heralds.heraldMoments(
    row({
      rescues: [{ champion: { classId: 1, teamId: 1 } }],
      match: {
        before: [5, 9],
        after: [5, 9],
        pointsToWin: 20,
        tick: 290,
        tickLimit: 300,
      },
    }),
  )) {
    assert.equal(heralds.momentEntersAtSettle(moment), true, moment.kind);
  }
});

test("the Death Announcer switch off drops the moments born from a fall", () => {
  const event = row({
    falls: [fall(2, "5"), fall(2, "6")],
    freshStart: true,
    match: {
      before: [0, 0],
      after: [2, 0],
      pointsToWin: 20,
      tick: 290,
      tickLimit: 300,
    },
    rescues: [{ champion: { classId: 1, teamId: 1 }, byUltimate: false }],
  });
  assert.deepEqual(kinds(event), ["firstBlood", "multi", "rescue", "moonset"]);
  assert.deepEqual(kinds(event, { death_announcer: false }), ["rescue", "moonset"]);
  assert.deepEqual(heralds.cardMoments(event).length, 3);
});

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
      classList: {
        contains: (/** @type {string} */ name) =>
          (attributes.get("class") ?? "").split(" ").includes(name),
      },
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      append: (/** @type {any[]} */ ...nodes) => element.children.push(...nodes),
      attributes,
    };
    return element;
  };
  return {
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      node(tag),
  };
}

/** @param {any} root @returns {any[]} */
function walk(root) {
  return [root, ...root.children.flatMap(walk)];
}

/** @param {boolean} animate @returns {{doc: any, mount: any}} */
function mount(animate) {
  const doc = fakeDocument();
  return {
    doc,
    mount: {
      group: doc.createElementNS("", "g"),
      connector: doc.createElementNS("", "g"),
      animationSpecs: /** @type {any[]} */ ([]),
      options: { settled: !animate, motionMode: "normal" },
      plan: { beats: { reducedScale: 0.25 } },
    },
  };
}

test("build draws the Herald Rule behind a tier-1 plate, names the moments and fades bands in", () => {
  const event = row({
    falls: [fall(2, "5"), fall(2, "6")],
    freshStart: true,
    match: fresh,
  });
  const { doc, mount: playing } = mount(true);
  heralds.build(heralds.records({ event, bounds }), doc, playing);
  const [plate] = playing.group.children;
  assert.equal(plate.getAttribute("data-herald-moments"), "firstBlood multi");
  assert.equal(plate.children[0].getAttribute("class"), "herald__tier");
  const rule = plate.children[0].children[0];
  assert.equal(rule.getAttribute("class"), "herald__tier-rule");
  assert.equal(rule.getAttribute("opacity"), "0.35");
  assert.deepEqual(
    playing.animationSpecs.map((/** @type {any} */ spec) => spec.id.split(":").at(-1)),
    ["herald-card", "herald-glyphs", "herald-rule"],
  );
  for (const spec of playing.animationSpecs) {
    assert.equal(spec.options.fill, "backwards");
    assert.equal(spec.options.delay, 960);
  }
  // A chip and a one-line band fade in from phaseStart; held ones do not.
  const chipRow = row({
    combos: [{ casters: [{ classId: 2 }, { classId: 4 }], fell: true }],
    phaseStart: 1400,
  });
  const chipped = mount(true);
  heralds.build(
    heralds.records({ event: chipRow, bounds }),
    chipped.doc,
    chipped.mount,
  );
  assert.equal(chipped.mount.animationSpecs.length, 1);
  assert.equal(chipped.mount.animationSpecs[0].options.delay, 1400);
  const narrow = { left: 0, top: 0, right: 200, bottom: 300, width: 200, height: 300 };
  const [band] = heralds.records({ event, bounds: narrow });
  assert.equal(band.kind, "herald-band");
  const banded = mount(true);
  heralds.build([band], banded.doc, banded.mount);
  assert.equal(banded.mount.animationSpecs.length, 1);
  const held = mount(true);
  heralds.build(
    heralds.records({ event: row(), previous: event, bounds }),
    held.doc,
    held.mount,
  );
  assert.equal(held.mount.animationSpecs.length, 0);
});

test("a row with no moment keeps the whole BUDGET for a held card; a run needs 12 nodes", () => {
  assert.equal(heralds.rowBudget(row({ falls: [fall(2, "6")] })), heralds.BUDGET);
  assert.deepEqual(heralds.rowBudget(row({ runs: [{ teamId: 2, takedowns: 5 }] })), {
    nodesPerRecord: 12,
    animationsPerRecord: 2,
    persistentNodesPerRecord: 12,
  });
  assert.equal(
    heralds.rowBudget(row({ falls: [fall(2, "5")], freshStart: true, match: fresh }))
      .animationsPerRecord,
    3,
  );
});

test("the largest cards stay inside BUDGET and draw only export-safe nodes", () => {
  const seven = Array.from({ length: 7 }, (_, index) => fall(2, String(5 + index)));
  const cases = [
    row({ falls: seven, freshStart: true, match: fresh }),
    row({
      falls: [
        ...seven.slice(0, 3),
        fall(1, "0"),
        fall(1, "1"),
        fall(1, "2"),
        fall(1, "3"),
      ],
    }),
    row({ falls: [...seven.slice(0, 4), fall(1, "0"), fall(1, "1"), fall(1, "2")] }),
    row({
      falls: [fall(2, "5")],
      wiped: [
        {
          teamId: 2,
          members: Array.from({ length: 5 }, () => ({ classId: 2, teamId: 2 })),
        },
      ],
    }),
    row({ rescues: [{ champion: { classId: 1, teamId: 2 }, byUltimate: true }] }),
    row({ runs: [{ teamId: 1, takedowns: 10 }] }),
    row({
      match: {
        before: [3, 9],
        after: [3, 9],
        pointsToWin: 20,
        tick: 290,
        tickLimit: 300,
      },
    }),
    row({ combos: [{ casters: [{ classId: 2 }, { classId: 3 }, { classId: 4 }] }] }),
    // Match Point held near the end with the Ticks Left count (3 Oct 2026).
    row({
      match: {
        before: [18, 19],
        after: [18, 19],
        pointsToWin: 20,
        tick: 297,
        tickLimit: 300,
      },
    }),
    row({
      match: {
        before: [18, 18],
        after: [18, 19],
        pointsToWin: 20,
        tick: 296,
        tickLimit: 300,
      },
    }),
  ];
  for (const event of cases) {
    const { doc, mount: playing } = mount(true);
    heralds.build(heralds.records({ event, bounds }), doc, playing);
    const nodes = walk(playing.group).slice(1);
    assert.ok(nodes.length <= heralds.BUDGET.nodesPerRecord, `${nodes.length} nodes`);
    assert.ok(playing.animationSpecs.length <= heralds.BUDGET.animationsPerRecord);
    // The plan charges the row only its own card's allowance, which covers it
    // and its own group (the painter counts the group as a kept node too).
    const allowance = heralds.rowBudget(event);
    assert.ok(
      nodes.length + 1 <= allowance.nodesPerRecord,
      `${nodes.length} + 1 > allowance`,
    );
    assert.ok(nodes.length + 1 <= allowance.persistentNodesPerRecord);
    assert.ok(playing.animationSpecs.length <= allowance.animationsPerRecord);
    for (const node of nodes) {
      assert.ok(
        ["g", "path", "rect", "text", "svg", "use", "circle"].includes(node.tagName),
        node.tagName,
      );
    }
  }
});
