/**
 * @file Check the short Lock In rows the BattleClient deck shows when you play
 * three or more champions (src/tactics-turn.js compactOrderParts and
 * compactOrderLine, ruling P7, 3 Oct 2026: every controlled champion's line
 * shows): an untouched order reads "Stay · No Combat"; No Combat reads its
 * step first; an action reads its ability, then its target, then its step
 * only when it moves (the part that gives way first), with the same words as
 * the rail's checklist line (checklistLine, one owner of the words); an
 * Ultimate with no target names only the Ultimate; a fallen champion reads
 * "Fallen · Returns In N" (fallenWords, short; "Fallen" with no wave clock)
 * and never shows an order, while the rail and Your Champion use the long
 * form "Fallen · Returns In N Ticks".
 */
import assert from "node:assert/strict";
import test from "node:test";
import {
  checklistLine,
  compactOrderLine,
  compactOrderParts,
  fallenWords,
} from "../src/tactics-turn.js";

const ROW = /** @type {const} */ ({
  touched: true,
  moveAction: 0,
  lane: "none",
  classId: 1,
  ultimateNeedsNoTarget: true,
  targetShort: null,
  fallen: null,
});

test("an untouched order is Stay with No Combat", () => {
  assert.equal(
    compactOrderLine({
      ...ROW,
      touched: false,
      lane: "basic",
      targetShort: "Warrior 6",
    }),
    "Stay · No Combat",
  );
});

test("a set order names its action, target and step like the checklist", () => {
  const cases = /** @type {const} */ ([
    [
      { ...ROW, lane: "basic", targetShort: "Warrior 6" },
      { main: "Arcane Blast", target: "Warrior 6", tail: null },
      "Step: Stay · Action: Arcane Blast → Warrior 6",
    ],
    [
      { ...ROW, moveAction: 6, lane: "none" },
      { main: "North-West · No Combat", target: null, tail: null },
      "Step: North-West · Action: No Combat",
    ],
    [
      { ...ROW, moveAction: 1, lane: "ultimate" },
      { main: "Burst", target: null, tail: "North" },
      "Step: North · Action: Burst",
    ],
    [
      {
        ...ROW,
        classId: 5,
        ultimateNeedsNoTarget: false,
        moveAction: 7,
        lane: "ultimate",
        targetShort: "Warrior 1",
      },
      { main: "Holy Word: Salvation", target: "Warrior 1", tail: "South-East" },
      "Step: South-East · Action: Holy Word: Salvation → Warrior 1",
    ],
    [
      {
        ...ROW,
        classId: 2,
        ultimateNeedsNoTarget: false,
        lane: "basic",
        targetShort: null,
      },
      { main: "Stay · No Combat", target: null, tail: null },
      "Step: Stay · Action: No Combat",
    ],
  ]);
  for (const [row, parts, rail] of cases) {
    assert.deepEqual(compactOrderParts(row), parts);
    assert.equal(
      compactOrderLine(row),
      `${parts.main}${parts.target === null ? "" : ` → ${parts.target}`}${
        parts.tail === null ? "" : ` · ${parts.tail}`
      }`,
    );
    assert.equal(checklistLine(row), rail);
  }
  assert.equal(
    compactOrderLine({
      ...ROW,
      classId: 3,
      ultimateNeedsNoTarget: false,
      moveAction: 3,
      lane: "basic",
      targetShort: "Warrior 6",
    }),
    "Serrated Arrows → Warrior 6 · East",
  );
});

test("a fallen champion shows when it returns, never an order", () => {
  const fallen = { className: "Warrior", returnsIn: 3 };
  assert.equal(
    compactOrderLine({ ...ROW, lane: "basic", targetShort: "Mage 5", fallen }),
    "Fallen · Returns In 3",
  );
  assert.equal(
    compactOrderLine({ ...ROW, fallen: { className: "Warrior", returnsIn: null } }),
    "Fallen",
  );
  assert.equal(fallenWords({ returnsIn: 3 }), "Fallen · Returns In 3 Ticks");
  assert.equal(fallenWords({ returnsIn: 1 }), "Fallen · Returns In 1 Tick");
  assert.equal(fallenWords({ returnsIn: 1 }, true), "Fallen · Returns In 1");
  assert.equal(fallenWords({ returnsIn: null }), "Fallen");
});
