/**
 * @file Check that every BattleClient return countdown tells the same truth.
 *
 * Core brings fallen champions back on the transition that starts with a
 * respawn-wave countdown of 0, so a countdown of N means "back in N + 1
 * ticks" (a recorded game: countdown 1 at tick 98, the Warrior back at tick
 * 100). Contracts: the scorebug's "Respawn Wave in N Ticks"
 * (liveScorebugView) says countdown + 1; the deck's waveIn (the "It returns
 * in N ticks" sentence and the "Fallen · Returns in N" agent card) and the
 * portraits' "Returns in N" (looks/hud.js) add the same 1, so the three
 * never disagree on one screen.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { liveScorebugView } from "../src/match-summary.js";

const DECK = readFileSync(new URL("../src/tactics-turn.js", import.meta.url), "utf8");
const HUD = readFileSync(new URL("../src/looks/hud.js", import.meta.url), "utf8");

test("the scorebug's wave says countdown + 1 ticks", () => {
  const summary = {
    threshold: 20,
    teams: [
      { teamId: 1, scoreValue: 6, side: "left" },
      { teamId: 2, scoreValue: 5, side: "right" },
    ],
    tick: { text: "Tick 98 of 300" },
  };
  const view = liveScorebugView(summary, {
    waves: [
      { team_id: 1, countdown_steps: 1 },
      { team_id: 2, countdown_steps: 1 },
    ],
  });
  assert.equal(view.waveText, "Respawn Wave in 2 Ticks");
  const due = liveScorebugView(summary, {
    waves: [{ team_id: 1, countdown_steps: 0 }],
  });
  assert.equal(due.waveText, "Respawn Wave in 1 Tick");
});

test("the deck and the portraits add the same 1 to the countdown", () => {
  assert.match(
    DECK,
    /waveIn: \(teamId\) => \{[\s\S]{0,200}?Number\(wave\?\.countdown_steps\) \+ 1/u,
  );
  assert.match(
    HUD,
    /returnsIn: !alive && Number\.isInteger\(countdown\) \? countdown \+ 1/u,
  );
});
