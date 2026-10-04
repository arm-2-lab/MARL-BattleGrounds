/**
 * @file Check the Champion Streaks switch (`champion_streaks`) of the Herald
 * Cards look (`src/looks/heralds.js`). With the switch off (it is on by
 * default since 4 Oct 2026; these checks pass it explicitly) a
 * herald_card row's `streaks` facts are ignored and every card is exactly the
 * plain one: a single fall still gives no card. Switched on, the row's first
 * valid streak heads the card: the title "5 Takedowns Without Falling", the
 * eyebrow "Team A Mage", the champion's team rule and its living medallion,
 * with no dots (the host sends 5, 10, 15 or 20); further streaks and then the
 * tick's fall moment join the eyebrow, at most two ("Team B Rogue In 10
 * Takedowns", "Double Takedown", "Trade", "◆ Charge Takedown"). Invalid facts
 * are skipped, the previous tick's held card follows the same switch, a card
 * too wide for its spot stays the plate at a smaller readable size, and the
 * tooltip explains
 * the streak in plain words with shared credit. Agent POV: the look draws
 * only the streaks the row carries and counts no history itself, so a row
 * without a fact for a champion gives that champion no card. build() draws
 * no dots and stays inside the node budget with export-safe nodes, and the switch
 * owns the registered herald_streak paint part; heralds.css keeps no dot rules.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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

const bounds = { left: 0, top: 0, right: 1200, bottom: 700, width: 1200, height: 700 };
const ON = Object.freeze({ champion_streaks: true });

/**
 * @param {number} victimTeam
 * @param {string} id
 * @param {Record<string, any>} [extra]
 */
function fall(victimTeam, id, extra = {}) {
  return {
    victim: { title: `Warrior · ID ${id}`, classId: 2, teamId: victimTeam },
    killers: [],
    killingTeamId: 3 - victimTeam,
    points: 1,
    ...extra,
  };
}

/**
 * @param {number} classId
 * @param {number} teamId
 * @param {string} id
 * @param {number} takedowns
 */
function streak(classId, teamId, id, takedowns) {
  return {
    champion: { publicAgentId: id, classId, teamId, title: `Champion · ID ${id}` },
    takedowns,
  };
}

/** @param {any[]} falls @param {any[]} [streaks] */
function heraldRow(falls, streaks) {
  return {
    kind: "herald_card",
    eventId: "t:herald",
    count: falls.length,
    falls,
    ...(streaks === undefined ? {} : { streaks }),
    paintParts: { effect: true },
    phaseStart: 960,
    phaseEnd: 2400,
  };
}

test("off by default: streak facts are ignored and a single fall gives no card", () => {
  const row = heraldRow([fall(2, "6")], [streak(1, 1, "0", 5)]);
  assert.deepEqual(heralds.records({ event: row, bounds }), []);
  assert.deepEqual(
    heralds.records({ event: row, bounds, filters: { champion_streaks: false } }),
    [],
  );
  const double = heraldRow([fall(2, "6"), fall(2, "7")], [streak(1, 1, "0", 5)]);
  const [plain] = heralds.records({ event: double, bounds });
  assert.equal(plain.title, "Double Takedown");
  assert.equal(plain.eyebrow, "Team A +2");
  assert.equal("dots" in plain, false);
  assert.equal("streak" in plain, false);
});

test("switched on, a streak heads the card with the champion and its team rule, no dots", () => {
  const row = heraldRow([fall(2, "6")], [streak(1, 1, "0", 5)]);
  const [card] = heralds.records({ event: row, bounds, filters: ON });
  assert.equal(card.kind, "herald-plate");
  assert.equal(card.title, "5 Takedowns Without Falling");
  assert.equal(card.eyebrow, "Team A Mage");
  assert.equal(card.ruleTeam, "a");
  assert.deepEqual(card.medallions, [
    { classKey: "mage", teamKey: "a", fallen: false },
  ]);
  assert.equal("dots" in card, false);
  assert.deepEqual(card.streak, { title: "Champion · ID 0", takedowns: 5 });
  assert.deepEqual(card.fallen, ["Warrior · ID 6"]);
  // One 26 px medallion and nothing after it.
  assert.equal(card.glyphWidth, 26);
  const [eight] = heralds.records({
    event: heraldRow([fall(1, "1")], [streak(4, 2, "8", 10)]),
    bounds,
    filters: ON,
  });
  assert.equal(eight.title, "10 Takedowns Without Falling");
  assert.equal("dots" in eight, false);
  assert.equal(eight.ruleTeam, "b");
});

test("other moments of the tick join the eyebrow, at most two", () => {
  const [multi] = heralds.records({
    event: heraldRow([fall(2, "6"), fall(2, "7")], [streak(1, 1, "0", 5)]),
    bounds,
    filters: ON,
  });
  assert.equal(multi.eyebrow, "Team A Mage · Double Takedown");
  const [ultimate] = heralds.records({
    event: heraldRow(
      [fall(2, "6", { ultimate: { classId: 2, publicAgentId: "1" } })],
      [streak(2, 1, "1", 10)],
    ),
    bounds,
    filters: ON,
  });
  assert.equal(ultimate.eyebrow, "Team A Warrior · ◆ Charge Takedown");
  assert.equal(ultimate.diamondClassKey, null);
  const [crowded] = heralds.records({
    event: heraldRow(
      [fall(1, "1"), fall(2, "6")],
      [streak(1, 1, "0", 5), streak(4, 2, "8", 10), streak(3, 1, "2", 5)],
    ),
    bounds,
    filters: ON,
  });
  assert.equal(
    crowded.eyebrow,
    "Team A Mage · Team B Rogue 10 Takedowns · Team A Hunter 5 Takedowns",
  );
  const [trade] = heralds.records({
    event: heraldRow(
      [fall(1, "1"), fall(2, "6")],
      [streak(1, 1, "0", 5), streak(4, 2, "8", 10)],
    ),
    bounds,
    filters: ON,
  });
  assert.equal(trade.eyebrow, "Team A Mage · Team B Rogue 10 Takedowns · Trade");
});

test("invalid streak facts are skipped and the held card follows the switch", () => {
  const junk = heraldRow(
    [fall(2, "6")],
    [
      null,
      { champion: { classId: 9, teamId: 1 }, takedowns: 5 },
      { champion: { classId: 1, teamId: 3 }, takedowns: 5 },
      { champion: { classId: 1, teamId: 1 }, takedowns: 0 },
      { champion: { classId: 1, teamId: 1 }, takedowns: 2.5 },
      { takedowns: 5 },
    ],
  );
  assert.deepEqual(heralds.records({ event: junk, bounds, filters: ON }), []);
  const previous = heraldRow([fall(2, "6")], [streak(5, 2, "9", 5)]);
  const carrier = { ...heraldRow([]), falls: [], count: 0 };
  const [held] = heralds.records({ event: carrier, previous, bounds, filters: ON });
  assert.equal(held.held, true);
  assert.equal(held.title, "5 Takedowns Without Falling");
  assert.deepEqual(heralds.records({ event: carrier, previous, bounds }), []);
});

test("Agent POV: only the row's own facts are drawn, so an unseen streak gives no card", () => {
  // The plan builder gives an Agent POV row only the streaks the team saw.
  // Here the team saw the fall but holds no streak fact for its striker.
  const povRow = heraldRow([fall(2, "6", { killers: [{ publicAgentId: "4" }] })], []);
  assert.deepEqual(heralds.records({ event: povRow, bounds, filters: ON }), []);
  const seen = heraldRow([fall(2, "6")], [streak(3, 1, "2", 5)]);
  const [card] = heralds.records({ event: seen, bounds, filters: ON });
  assert.doesNotMatch(JSON.stringify(card), /"4"/u);
});

test("a card too wide for its spot stays the plate, at a smaller readable size", () => {
  const narrow = { left: 0, top: 0, right: 600, bottom: 400, width: 600, height: 400 };
  const [card] = heralds.records({
    event: heraldRow([fall(2, "6")], [streak(1, 1, "0", 10)]),
    bounds: narrow,
    filters: ON,
  });
  assert.equal(card.kind, "herald-plate");
  assert.equal(card.title, "10 Takedowns Without Falling");
  assert.equal(card.eyebrow, "Team A Mage");
  assert.ok(card.scale <= 1 && card.scale >= 9 / 11 - 1e-9);
  assert.deepEqual(card.streak, { title: "Champion · ID 0", takedowns: 10 });
});

test("build draws no dots, stays inside the budget with export-safe nodes, and the tooltip explains it", () => {
  const [card] = heralds.records({
    event: heraldRow([fall(2, "6"), fall(2, "7")], [streak(1, 1, "0", 5)]),
    bounds,
    filters: ON,
  });
  const doc = fakeDocument();
  const [node] = heralds.build([card], doc);
  assert.equal(byClass(node, "herald__streak-dots").length, 0);
  assert.equal(byClass(node, "herald__medallion--fallen").length, 0);
  assert.equal(
    byClass(node, "herald__title")[0].textContent,
    "5 Takedowns Without Falling",
  );
  assert.equal(
    byClass(node, "herald__eyebrow")[0].textContent,
    "TEAM A MAGE · DOUBLE TAKEDOWN",
  );
  assert.ok(descendants(node).length + 1 <= heralds.BUDGET.nodesPerRecord);
  const text = serialize(node);
  assert.doesNotMatch(text, /url\(|gradient|<mask|<filter|<image|<foreignObject/iu);
  assert.doesNotMatch(text, / id="/u);
  const tip = /** @type {any} */ (heralds.tooltip(card));
  assert.equal(tip.title, "5 Takedowns Without Falling");
  const rows = tip.rows.map((/** @type {any} */ row) => [row.label, row.value]);
  assert.deepEqual(rows[0], [
    "Streak",
    "Champion · ID 0 took part in 5 takedowns without falling. Kill credit is shared.",
  ]);
  assert.deepEqual(rows[1], ["Fallen", "Warrior · ID 6; Warrior · ID 7"]);
});

test("the switch owns its paint part and the styles keep no dot rules", async () => {
  assert.ok(
    heralds.PAINT_PARTS.some(
      (registration) =>
        registration.part.surface === "transient" &&
        registration.part.kind === "herald_streak" &&
        registration.filter === "champion_streaks",
    ),
  );
  const css = await readFile(
    new URL("../src/styles/heralds.css", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(css, /streak-dots/u);
});
