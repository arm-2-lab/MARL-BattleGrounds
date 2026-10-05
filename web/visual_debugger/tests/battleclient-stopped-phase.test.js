/**
 * @file Check the deck's turn phase once a game is stopped (src/tactics-turn.js
 * turnPhase and turnTitles): after Cancel Policy Work the host asks for a fresh
 * start (`fresh_launch_required`), and the phase is "plan" whatever else is
 * true, even while the cancelled policy work still winds down ("cancelling"),
 * a Lock In is still out, the tick still plays or Run is on. So, as on the
 * vision page, the phases show You Plan, the Lock In label reads "Lock In
 * Turn" and the title reads "Game Stopped". Without a stop, "cancelling"
 * still reads as Policies Choose.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { turnPhase, turnTitles } from "../src/tactics-turn.js";

test("a stopped game is always in the plan phase", () => {
  for (const operationState of [
    undefined,
    "loading",
    "thinking",
    "cancelling",
    "finishing",
    "failed",
  ]) {
    for (const flags of [
      { lockSent: false, choreographyBlocking: false, running: false },
      { lockSent: true, choreographyBlocking: true, running: true },
    ]) {
      assert.equal(turnPhase({ operationState, stopped: true, ...flags }), "plan");
    }
  }
  assert.equal(
    turnPhase({
      operationState: "cancelling",
      lockSent: false,
      choreographyBlocking: false,
      stopped: false,
    }),
    "choosing",
  );
  const titles = turnTitles({
    phase: turnPhase({
      operationState: "cancelling",
      lockSent: true,
      choreographyBlocking: false,
      stopped: true,
    }),
    stopped: true,
    terminal: false,
    watchMode: false,
    running: false,
  });
  assert.deepEqual({ ...titles }, { title: "Game Stopped", lockLabel: "Lock In Turn" });
});
