/**
 * @file Check the words the rail's This Turn checklist uses for each order
 * (src/tactics-turn.js checklistLine, the one owner of these words since the
 * deck lost its own order list, owner, 4 Oct 2026): an action reads its step,
 * then its ability and target ("Step: Stay · Action: Arcane Blast → Warrior
 * 6"); No Combat reads its step; an Ultimate with no target names only the
 * Ultimate; a Basic with no target reads No Combat. A fallen champion reads
 * "Fallen · Returns In N Ticks" ("1 Tick" for one, "Fallen" with no wave
 * clock; fallenWords).
 */
import assert from "node:assert/strict";
import test from "node:test";
import { checklistLine, fallenWords } from "../src/tactics-turn.js";

const ROW = /** @type {const} */ ({
  touched: true,
  moveAction: 0,
  lane: "none",
  classId: 1,
  ultimateNeedsNoTarget: true,
  targetShort: null,
  fallen: null,
});

test("a set order names its step, then its action and target", () => {
  const cases = /** @type {const} */ ([
    [
      { ...ROW, lane: "basic", targetShort: "Warrior 6" },
      "Step: Stay · Action: Arcane Blast → Warrior 6",
    ],
    [{ ...ROW, moveAction: 6, lane: "none" }, "Step: North-West · Action: No Combat"],
    [{ ...ROW, moveAction: 1, lane: "ultimate" }, "Step: North · Action: Burst"],
    [
      {
        ...ROW,
        classId: 5,
        ultimateNeedsNoTarget: false,
        moveAction: 7,
        lane: "ultimate",
        targetShort: "Warrior 1",
      },
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
      "Step: Stay · Action: No Combat",
    ],
  ]);
  for (const [row, rail] of cases) {
    assert.equal(checklistLine(row), rail);
  }
});

test("a fallen champion's words say when it returns", () => {
  assert.equal(fallenWords({ returnsIn: 3 }), "Fallen · Returns In 3 Ticks");
  assert.equal(fallenWords({ returnsIn: 1 }), "Fallen · Returns In 1 Tick");
  assert.equal(fallenWords({ returnsIn: null }), "Fallen");
});
