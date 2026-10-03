/**
 * @file Check the live arena marks that follow the BattleClient vision page
 * exactly: the corner brackets' "vision" shape (src/looks/champion-frame.js
 * bracketPath and the selection-marks records: half-size 1.48 R, arms 0.45 R,
 * at any radius, with the Replay Viewer's "ladder" shape unchanged), and the
 * Look Ring of the Champion's Hotbar (src/looks/hotbar.js), which the
 * Selection Moonlight switch turns off (`selection_moonlight: false` drops
 * only the ring; your champion's dimmed Command Ring stays). The live Grave
 * Ground marks follow the store, not the vision page, and are checked in
 * tests/marks-grave-ground-live.test.js.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as frame from "../src/looks/champion-frame.js";
import * as hotbar from "../src/looks/hotbar.js";

/** @param {string} path @returns {number[][]} */
function points(path) {
  return [...path.matchAll(/(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)/gu)].map((m) => [
    Number(m[1]),
    Number(m[2]),
  ]);
}

test("vision brackets sit 1.48 R out with 0.45 R arms at any radius", () => {
  for (const radius of [8, 22, 31]) {
    const xs = points(frame.bracketPath(100, 100, radius, "vision")).map(([x]) => x);
    const half = radius * (1 + 12 / 25);
    assert.ok(Math.abs(Math.min(...xs) - (100 - half)) < 1e-9);
    assert.ok(Math.abs(Math.max(...xs) - (100 + half)) < 1e-9);
    // The first corner starts one arm in from the left edge.
    assert.ok(Math.abs(xs[0] - (100 - half + 0.45 * radius)) < 1e-9);
  }
  assert.equal(frame.bracketPath(50, 50, 8), frame.bracketPath(50, 50, 8, "ladder"));
  assert.match(frame.bracketPath(50, 50, 8), /A /u);
});

test("selection-marks records carry the bracket shape, ladder by default", () => {
  const champions = [
    {
      identityKey: "k1",
      agent: { alive: true },
      center: { x: 10, y: 10 },
      radius: 20,
      controlled: false,
      selected: true,
    },
  ];
  const live = frame.records({
    part: "selection-marks",
    champions,
    showSelectionReticle: true,
    bracketShape: "vision",
  });
  const replay = frame.records({ part: "selection-marks", champions });
  assert.equal(live[0]?.bracketShape, "vision");
  assert.equal(replay[0]?.bracketShape, "ladder");
});

/** @param {Record<string, boolean>} filters */
function watchOnlyRecords(filters) {
  const agent = (/** @type {string} */ key, /** @type {number} */ x) => ({
    presentation_key: key,
    position: [x, 5],
    radius: 0.5,
    alive: true,
    life_state: "alive",
    team_id: 1,
    class_id: 2,
    display_agent_id: key === "you" ? "1" : "2",
  });
  const scene = {
    agents: [agent("you", 4), agent("mate", 8)],
    selection: { controlled_presentation_key: "you" },
  };
  const transform = createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: 1000,
    viewportHeight: 500,
    padding: 0,
  });
  return hotbar.records({ scene, transform, lookedAtKey: "mate", filters });
}

test("Selection Moonlight off drops the Look Ring and keeps the dimmed ring", () => {
  const on = watchOnlyRecords({});
  const off = watchOnlyRecords({ selection_moonlight: false });
  assert.equal(on.filter((r) => r.kind === "look-ring").length, 1);
  assert.equal(off.filter((r) => r.kind === "look-ring").length, 0);
  assert.deepEqual(
    off.map((r) => r.kind),
    on.filter((r) => r.kind !== "look-ring").map((r) => r.kind),
  );
  assert.ok(off.some((r) => r.kind === "command-ring" && r.away === true));
});
