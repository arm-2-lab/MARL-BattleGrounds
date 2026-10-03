/**
 * @file Check the status line's sentence for a new tick
 * (src/tactics-turn.js tickPlayedAnnouncement): the tick that ends the game
 * always says the game-over step for the replay's save state (gameOverStep),
 * the same words as the rail's Do This Now at game over (doThisNow), whoever
 * played it, so "Orders sent. The policies choose next." never stays on
 * screen after the game; your own Lock In says the tick played; your Play One
 * Turn in Policies Only says the turn played; and a tick you did not ask for
 * before the end keeps the line (null).
 */
import assert from "node:assert/strict";
import test from "node:test";
import {
  doThisNow,
  gameOverStep,
  tickPlayedAnnouncement,
} from "../src/tactics-turn.js";

const RECORDING = { lifecycle: "recording", finish_available: true };
const GAME_OVER =
  "The game is over. Press Restart for a new match, or Finish And Save.";

test("the tick that ends the game says the game-over step", () => {
  assert.equal(gameOverStep(RECORDING), GAME_OVER);
  for (const mine of [true, false]) {
    for (const watchMode of [true, false]) {
      assert.equal(
        tickPlayedAnnouncement({
          terminal: true,
          mine,
          watchMode,
          recording: RECORDING,
        }),
        GAME_OVER,
      );
    }
  }
  const rail = doThisNow(
    /** @type {any} */ ({
      stopped: false,
      terminal: true,
      phase: "plan",
      recording: RECORDING,
    }),
  );
  assert.equal(rail.text, GAME_OVER);
});

test("other ticks keep their sentences", () => {
  assert.equal(
    tickPlayedAnnouncement({ terminal: false, mine: true, watchMode: true }),
    "The turn played. The match is paused again.",
  );
  assert.match(
    String(tickPlayedAnnouncement({ terminal: false, mine: true, watchMode: false })),
    /^The tick played\. Every step is back to Stay/u,
  );
  assert.equal(
    tickPlayedAnnouncement({ terminal: false, mine: false, watchMode: true }),
    null,
  );
});
