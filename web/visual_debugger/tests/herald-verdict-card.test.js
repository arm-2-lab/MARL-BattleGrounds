/**
 * @file Check the store's game-over Verdict card and the herald plates' place
 * in `src/looks/heralds.js`. The full Verdict card puts the team whose home
 * end is on the left in the left column, shows each team's controller name
 * (a long one shrinks to 13 or 11 px, then is cut with "…"; a short line
 * such as "2 Policies" stands in only when the caller gives one, so a single
 * policy keeps its own name), the lead line only on a horizon draw with a
 * lead, an Honors cell per honor with every tied leader's medallion, and the
 * Match Summary and Show Arena buttons, centred over the arena and scaled to
 * fit; verdictButtons gives the buttons' boxes in arena px and
 * buildVerdictActions makes them real buttons that report their id. Tucked
 * (Show Arena) or with full cards off it is the band plate in the top band.
 * A herald plate lives only in the top band, keeps the spot it was first given
 * for its whole life (same row, same arena), never vanishes when no spot is
 * free (it stays a plate even when page chrome covers the whole band), and
 * stays inside BUDGET even with ten falls in one tick. The host's
 * `match_honors` fact turns into the verdict's Honors rows (verdictHonours),
 * and buildMatchSummarySheet lists one row per catalog label with a column for
 * each team's total and each champion (the Verdict's left team first), "—" for
 * unknown credit, and a working Close button.
 */
import assert from "node:assert/strict";
import test from "node:test";

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
      style: /** @type {Record<string, string>} */ ({}),
      textContent: "",
      listeners: /** @type {Array<[string, Function]>} */ ([]),
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      addEventListener: (/** @type {string} */ type, /** @type {Function} */ fn) =>
        element.listeners.push([type, fn]),
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

const bounds = { left: 0, top: 0, right: 908, bottom: 468, width: 908, height: 468 };

const DRAW = {
  scores: { 1: 13, 2: 17 },
  winnerTeamId: null,
  tick: 300,
  basis: "horizon",
  pointsToWin: 20,
  teams: { 1: { name: "You" }, 2: { name: "Recurrent MAPPO With a Very Long Name" } },
  honours: [
    {
      label: "Kill Contributions",
      value: 9,
      leaders: [
        { classId: 2, teamId: 2 },
        { classId: 4, teamId: 2 },
      ],
    },
    { label: "Damage Done", value: 412.4, leaders: [{ classId: 1, teamId: 1 }] },
    {
      label: "Effective Healing Done",
      value: 301,
      leaders: [{ classId: 5, teamId: 2 }],
    },
  ],
};

test("the full verdict follows the home ends, names controllers, honors ties and has two buttons", () => {
  const [card] = heralds.records({
    part: "verdict",
    product: "combat_debugger",
    verdict: DRAW,
    bounds,
    leftTeamId: 2,
  });
  assert.equal(card.kind, "herald-verdict");
  assert.equal(card.eyebrow, "Tick 300 · No Team Reached 20");
  assert.equal(card.lead, "A Lead at Tick 300 Wins Nothing");
  assert.deepEqual(
    card.columns.map((/** @type {any} */ column) => [
      column.teamId,
      column.score,
      column.result,
    ]),
    [
      [2, 17, "Draw"],
      [1, 13, "Draw"],
    ],
  );
  assert.equal(card.columns[1].name, "You");
  assert.equal(card.columns[1].nameSize, 15);
  assert.match(card.columns[0].name, /…$/u);
  assert.equal(card.columns[0].nameSize, 11);
  assert.ok(card.columns[0].name.length < DRAW.teams[2].name.length);
  // One policy keeps its own name, smaller before it is cut; a short line
  // ("2 Policies") stands in only when the caller gives one.
  const named = (/** @type {Record<string, any>} */ team) =>
    heralds
      .records({
        part: "verdict",
        product: "replay_viewer",
        verdict: { ...DRAW, teams: { 1: { name: "You" }, 2: team } },
        bounds,
      })[0]
      .columns.find((/** @type {any} */ column) => column.teamId === 2);
  const shrunk = named({ name: "IPPO at step 250,675,200" });
  assert.equal(shrunk.name, "IPPO at step 250,675,200");
  assert.ok(shrunk.nameSize < 15);
  assert.equal(
    named({ name: "IPPO at step 250,675,200 (run) + QMIX", short: "2 Policies" }).name,
    "2 Policies",
  );
  assert.deepEqual(
    card.honours.map((/** @type {any} */ honour) => [
      honour.label,
      honour.value,
      honour.medallions.length,
    ]),
    [
      ["Kill Contributions", "9", 2],
      ["Damage Done", "412", 1],
      ["Effective Healing Done", "301", 1],
    ],
  );
  // Centred on the arena, fully inside it, never under the minimum scale.
  assert.ok(Math.abs(card.x + (card.width * card.scale) / 2 - 454) < 0.01);
  assert.ok(card.y >= heralds.FOOTPRINT.topGap);
  assert.ok(card.y + card.height * card.scale <= bounds.bottom);
  const doc = fakeDocument();
  const [node] = heralds.build([card], doc);
  assert.equal(byClass(node, "herald__lead")[0].textContent, card.lead);
  assert.equal(byClass(node, "herald__honour-label").length, 3);
  assert.equal(byClass(node, "herald__name").length, 2);
  assert.deepEqual(
    byClass(node, "herald__button").map((group) =>
      group.getAttribute("data-verdict-action"),
    ),
    ["summary", "show_arena"],
  );
  assert.doesNotMatch(serialize(node), /url\(|gradient|<mask|<filter|<image| id="/iu);
  const boxes = heralds.verdictButtons(card);
  assert.deepEqual(
    boxes.map((box) => box.label),
    ["Match Summary", "Show Arena"],
  );
  for (const box of boxes) {
    assert.ok(
      box.left >= card.x && box.left + box.width <= card.x + card.width * card.scale,
    );
    assert.ok(box.top + box.height <= card.y + card.height * card.scale);
  }
  /** @type {string[]} */
  const pressed = [];
  const actions = heralds.buildVerdictActions(/** @type {any} */ (doc), card, (id) =>
    pressed.push(id),
  );
  const buttons = descendants(actions).filter((child) => child.tagName === "button");
  assert.equal(buttons.length, 2);
  assert.equal(buttons[1].getAttribute("aria-label"), "Show Arena");
  assert.equal(buttons[1].textContent, "");
  // Placed through the style object: the page's CSP refuses style attributes.
  assert.equal(buttons[0].getAttribute("style"), null);
  assert.equal(buttons[0].style.left, `${Math.round(boxes[0].left * 100) / 100}px`);
  assert.equal(buttons[1].style.height, `${Math.round(boxes[1].height * 100) / 100}px`);
  for (const button of buttons) {
    for (const [type, fn] of button.listeners) if (type === "click") fn();
  }
  assert.deepEqual(pressed, ["summary", "show_arena"]);
  assert.deepEqual(
    heralds.verdictButtons(/** @type {any} */ ({ kind: "herald-plate" })),
    [],
  );
});

test("Show Arena and full cards off give the band plate in the top band", () => {
  for (const input of [{ tucked: true }, { filters: { moment_full_cards: false } }]) {
    const [plate] = heralds.records({
      part: "verdict",
      product: "replay_viewer",
      verdict: DRAW,
      bounds,
      leftTeamId: 2,
      ...input,
    });
    assert.equal(plate.kind, "herald-verdict-plate");
    assert.equal(plate.y, heralds.FOOTPRINT.topGap);
    assert.deepEqual(
      plate.sides.map((/** @type {any} */ side) => [side.teamKey, side.score]),
      [
        ["b", 17],
        ["a", 13],
      ],
    );
    const [node] = heralds.build([plate], fakeDocument());
    assert.equal(
      byClass(node, "herald__lead")[0].textContent,
      " · A Lead at Tick 300 Wins Nothing",
    );
    assert.equal(heralds.tooltip(plate)?.title, "Draw");
  }
});

/** @param {number} count @param {boolean} trade */
function rowWith(count, trade) {
  return {
    kind: "herald_card",
    eventId: `t:herald:${count}:${trade}`,
    count,
    falls: Array.from({ length: count }, (_, index) => {
      const team = trade && index % 2 === 0 ? 1 : 2;
      return {
        victim: { title: `Mage · ID ${index}`, classId: (index % 5) + 1, teamId: team },
        killers: [],
        killingTeamId: 3 - team,
        points: 1,
      };
    }),
    paintParts: { effect: true },
    phaseStart: 960,
  };
}

test("a plate keeps its first spot, never vanishes and stays inside its budget", () => {
  const row = rowWith(2, false);
  const [born] = heralds.records({ event: row, bounds, avoid: [] });
  // The next tick holds the same row; new bodies under it do not move it.
  const [held] = heralds.records({
    event: {
      kind: "herald_card",
      falls: [],
      eventId: "next",
      paintParts: { effect: true },
    },
    previous: row,
    bounds,
    avoid: [{ x: born.x + 40, y: born.y + 10, radius: 20 }],
  });
  assert.equal(held.held, true);
  assert.equal(held.x, born.x);
  assert.equal(held.y, born.y);
  // Chrome over the whole top band leaves no free spot: the card still draws,
  // as the same plate (there is no one-line form).
  const [squeezed] = heralds.records({
    event: rowWith(2, false),
    bounds,
    avoid: [],
    chrome: [{ left: 0, top: 0, right: 908, bottom: 60 }],
  });
  assert.equal(squeezed.place, "top");
  assert.equal(squeezed.kind, "herald-plate");
  for (const [count, trade] of [
    [7, false],
    [10, true],
    [10, false],
  ]) {
    const [card] = heralds.records({
      event: rowWith(Number(count), Boolean(trade)),
      bounds,
    });
    const [node] = heralds.build([card], fakeDocument());
    assert.ok(card.medallions.length <= heralds.PLATE_MEDALLIONS_MAX);
    assert.ok(
      descendants(node).length + 1 <= heralds.BUDGET.nodesPerRecord,
      `${count} falls draw ${descendants(node).length + 1} nodes`,
    );
  }
});

test("the host's Honors fact feeds the verdict and the Match Summary sheet", () => {
  const fact = {
    schema_version: 1,
    team_id: 1,
    labels: [
      { metric: "kill_contributions", label: "Kill Contributions" },
      { metric: "deaths", label: "Deaths" },
      { metric: "damage_done", label: "Damage Done" },
      { metric: "effective_healing_done", label: "Effective Healing Done" },
      { metric: "red_zone_deaths", label: "Red Zone Deaths" },
      { metric: "ultimate_activations", label: "Ultimate Ability Activations" },
    ],
    honors: [
      {
        metric: "damage_done",
        label: "Damage Done",
        value: 412,
        leaders: [{ public_agent_id: "1", team_id: 1, class_id: 2 }],
      },
    ],
    champions: [
      {
        public_agent_id: "0",
        team_id: 1,
        class_id: 1,
        kill_contributions: null,
        deaths: 2,
        damage_done: 300,
        effective_healing_done: 0,
        red_zone_deaths: 1,
        ultimate_activations: 4,
      },
      {
        public_agent_id: "1",
        team_id: 1,
        class_id: 2,
        kill_contributions: null,
        deaths: 1,
        damage_done: 412,
        effective_healing_done: 0,
        red_zone_deaths: 0,
        ultimate_activations: 6,
      },
    ],
  };
  const rows = heralds.verdictHonours(fact);
  assert.deepEqual(rows, [
    {
      label: "Damage Done",
      value: 412,
      leaders: [{ classId: 2, teamId: 1, publicAgentId: "1" }],
    },
  ]);
  assert.deepEqual(heralds.verdictHonours(null), []);
  const [card] = heralds.records({
    part: "verdict",
    product: "combat_debugger",
    verdict: { ...DRAW, honours: rows },
    bounds,
  });
  assert.deepEqual(
    card.honours.map((/** @type {any} */ honour) => [honour.label, honour.value]),
    [["Damage Done", "412"]],
  );
  let closed = 0;
  const sheet = heralds.buildMatchSummarySheet(
    /** @type {any} */ (fakeDocument()),
    fact,
    {
      displayIdOf: (id) => id,
      onClose: () => {
        closed += 1;
      },
    },
  );
  const cells = (/** @type {any} */ row) =>
    row.children.map((/** @type {any} */ cell) => cell.textContent);
  const tableRows = descendants(sheet).filter((child) => child.tagName === "tr");
  assert.deepEqual(cells(tableRows[0]), [
    "",
    "Team A",
    "Mage · ID 0",
    "Warrior · ID 1",
  ]);
  assert.deepEqual(cells(tableRows[1]), ["Kill Contributions", "—", "—", "—"]);
  assert.deepEqual(cells(tableRows[3]), ["Damage Done", "712", "300", "412"]);
  assert.deepEqual(cells(tableRows[6]), [
    "Ultimate Ability Activations",
    "10",
    "4",
    "6",
  ]);
  assert.equal(tableRows.length, 7);
  assert.equal(
    byClass(sheet, "match-summary-sheet__scope")[0].textContent,
    "Your Team's Champions",
  );
  const close = descendants(sheet).find((child) => child.tagName === "button");
  for (const [type, fn] of close.listeners) if (type === "click") fn();
  assert.equal(closed, 1);
  const empty = heralds.buildMatchSummarySheet(
    /** @type {any} */ (fakeDocument()),
    null,
  );
  assert.equal(
    byClass(empty, "match-summary-sheet__empty")[0].textContent,
    "No Match Summary for This Game.",
  );
});
