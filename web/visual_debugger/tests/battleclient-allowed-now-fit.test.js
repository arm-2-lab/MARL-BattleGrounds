/**
 * @file Check how Allowed Now fits 2 Target (src/tactics-turn.js fitPicks,
 * 3 Oct 2026): the buttons keep the vision's pill form, name and abilities,
 * while they all show without scrolling; three or more that would scroll
 * switch to the short two-column grid (data-count "many"); one or two
 * buttons never switch; each fit starts again from the pill form, so a list
 * that fits again goes back to pills.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { fitPicks } from "../src/tactics-turn.js";

/**
 * A stand-in for the Allowed Now list: `count` buttons, each `pillHeight`
 * px tall as a pill and 24 px in the grid, in a list `room` px tall.
 *
 * @param {number} count
 * @param {number} room
 * @param {number} [pillHeight]
 * @returns {any}
 */
function picksOf(count, room, pillHeight = 28) {
  const picks = {
    dataset: /** @type {Record<string, string>} */ ({}),
    querySelectorAll: () => Array.from({ length: count }),
    clientHeight: room,
    get scrollHeight() {
      const grid = picks.dataset.count === "many";
      const rows = grid ? Math.ceil(count / 2) : count;
      return rows * (grid ? 24 : pillHeight) + Math.max(0, rows - 1) * 4;
    },
  };
  return picks;
}

test("pills stay while every button shows", () => {
  const picks = picksOf(3, 100);
  fitPicks(picks);
  assert.equal(picks.dataset.count, "3");
});

test("three or more pills that would scroll become the short grid", () => {
  const picks = picksOf(4, 100);
  fitPicks(picks);
  assert.equal(picks.dataset.count, "many");
});

test("one or two buttons keep the pill form even when tight", () => {
  const picks = picksOf(2, 20);
  fitPicks(picks);
  assert.equal(picks.dataset.count, "2");
  const none = picksOf(0, 100);
  fitPicks(none);
  assert.equal(none.dataset.count, "0");
});

test("each fit starts from the pill form, so a list that fits again shows pills", () => {
  const picks = picksOf(4, 100);
  fitPicks(picks);
  assert.equal(picks.dataset.count, "many");
  picks.clientHeight = 200;
  fitPicks(picks);
  assert.equal(picks.dataset.count, "4");
});
