/**
 * @file Check that the BattleClient deck's words are true for every roster
 * mix (src/tactics-turn.js, main-session ruling P6, 3 Oct 2026):
 * - startGreeting: the first status-line sentence counts the Manual champions
 *   of each team ("You play five champions on Team A."), keeps the vision's
 *   sentence for one champion on each team, for Policies Only and for one
 *   champion in Agent POV or Oracle View.
 * - howToPlayLines: one Manual champion gets exactly the vision page's lines,
 *   the same lines index.html holds; several get "silver rings", "Policies
 *   play the rest." only when a policy plays, and how to order another
 *   champion; Policies Only names Run, Play One Turn and Pause and no order
 *   keys.
 * - gameOverStep: the game-over step offers Finish And Save only while that
 *   button works, says the game saved once it has, says "Saving the replay…"
 *   while the server saves in the background, and names Retry Save and Save
 *   As after a failed save.
 * - plainHostNotice: the host's game-over notices become the game-over
 *   step, and no notice in developer words ("action mask",
 *   "Agent g5", "fenced", "coherent latest pair", "Debugger") reaches the
 *   status line.
 * - doThisNow: it never asks for a target when no target is allowed.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  doThisNow,
  gameOverStep,
  howToPlayLines,
  plainHostNotice,
  startGreeting,
} from "../src/tactics-turn.js";

test("the first sentence counts your champions on each team", () => {
  const KEEP = "Each keeps its own order until you lock in.";
  assert.equal(
    startGreeting({ watchMode: false, teamA: 5, teamB: 0, pov: true }),
    `You play five champions on Team A. ${KEEP}`,
  );
  assert.equal(
    startGreeting({ watchMode: false, teamA: 0, teamB: 2, pov: false }),
    `You play two champions on Team B. ${KEEP}`,
  );
  assert.equal(
    startGreeting({ watchMode: false, teamA: 2, teamB: 1, pov: true }),
    `You play two champions on Team A and one on Team B. ${KEEP}`,
  );
  assert.equal(
    startGreeting({ watchMode: false, teamA: 1, teamB: 1, pov: true }),
    `You play one champion on each team. ${KEEP}`,
  );
  assert.equal(
    startGreeting({ watchMode: true, teamA: 0, teamB: 0, pov: true }),
    "Nobody is yours this game. The match starts paused.",
  );
  assert.equal(
    startGreeting({ watchMode: false, teamA: 1, teamB: 0, pov: true }),
    "Agent POV: you see only what your team sees. Your orders stay private.",
  );
  assert.equal(
    startGreeting({ watchMode: false, teamA: 0, teamB: 1, pov: false }),
    "Oracle View: every champion is shown, even those your team cannot see.",
  );
});

test("How To Play is the vision's card for one champion and true for the rest", async () => {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const card = html.slice(html.indexOf('id="howto-card"'));
  const list = card.slice(card.indexOf("<ol>") + 4, card.indexOf("</ol>"));
  const markup = [...list.matchAll(/<li><span>([\s\S]*?)<\/span><\/li>/gu)].map(
    (match) => match[1],
  );
  assert.deepEqual(howToPlayLines({ manualCount: 1, policyCount: 9 }), markup);
  const five = howToPlayLines({ manualCount: 5, policyCount: 5 });
  assert.equal(
    five[0],
    "The <b>silver rings</b> are your champions. Policies play the rest.",
  );
  assert.equal(five[1], markup[1]);
  assert.equal(five[2], markup[2]);
  assert.match(five[3], /locks in all your orders/u);
  assert.equal(five[4], "Click a champion's row above to give it orders.");
  assert.equal(
    howToPlayLines({ manualCount: 10, policyCount: 0 })[0],
    "The <b>silver rings</b> are your champions. You play every champion.",
  );
  const watch = howToPlayLines({ manualCount: 0, policyCount: 10 });
  assert.equal(watch[0], "Policies play every champion. Nobody is yours this game.");
  assert.match(watch.join(" "), /<b>Run<\/b>.*<b>Play One Turn<\/b>.*<b>Pause<\/b>/u);
  assert.doesNotMatch(watch.join(" "), /kbd|silver ring|locks in/u);
});

test("the game-over step names only what the save state offers", () => {
  assert.equal(
    gameOverStep({ lifecycle: "recording", finish_available: true }),
    "The game is over. Press Restart for a new match, or Finish And Save.",
  );
  for (const lifecycle of ["saved", "reviewing"]) {
    const step = gameOverStep({ lifecycle, finish_available: false });
    assert.equal(
      step,
      "The game is over and saved. Press Restart for a new match, or click Saved at the top, then Open Replay.",
    );
    assert.doesNotMatch(step, /Finish And Save/u);
  }
  for (const lifecycle of ["sealed", "finalized_unsaved"]) {
    assert.equal(gameOverStep({ lifecycle }), "The game is over. Saving the replay…");
  }
  assert.equal(
    gameOverStep({ lifecycle: "persistence_failed" }),
    "The game is over, but the replay did not save. Click Save Failed at the top, then Retry Save or Save As.",
  );
  for (const recording of [null, undefined, { lifecycle: "discarded" }, "x"]) {
    assert.equal(
      gameOverStep(recording),
      "The game is over. Press Restart for a new match.",
    );
  }
  // A recording that cannot finish any more never offers Finish And Save.
  assert.doesNotMatch(
    gameOverStep({ lifecycle: "recording", finish_available: false }),
    /Finish And Save/u,
  );
});

test("host notices read in the player's words and never go stale", () => {
  assert.equal(
    plainHostNotice("The episode ended and its replay was saved."),
    gameOverStep({ lifecycle: "saved" }),
  );
  assert.equal(
    plainHostNotice(
      "The episode ended, but replay saving failed; choose Retry Save or Save As.",
    ),
    gameOverStep({ lifecycle: "persistence_failed" }),
  );
  // Developer words never reach the status line.
  const developer = [
    "Movement action 3 is unavailable in the current action mask; pending action unchanged.",
    "Basic requires a selected target; target-none lane zero is the canonical no-combat tuple. Pending action unchanged.",
    "Ultimate is unavailable for the selected target in the current action mask; pending action unchanged.",
    "Target action 7 is unavailable in the current authorized POV.",
    "Agent g5 is unavailable in this view.",
    "Repeated submission input ignored.",
    "Scientific controls are fenced because this recording is no longer capturing transitions.",
    "The episode is terminal; reset to continue.",
    "This tab was stale. Its coherent latest pair was installed; the command was not retried.",
    "A mixed authority GET was discarded; one fresh pair was installed.",
    "Duplicate command recognized; it was not applied again.",
    "Command already processed; current frame returned.",
    "Debugger shutdown requested.",
  ];
  for (const notice of developer) {
    const plain = plainHostNotice(notice);
    assert.notEqual(plain, notice);
    assert.doesNotMatch(
      plain,
      /action mask|pending action|tuple|authorized|Agent g\d|fenced|terminal|coherent|GET|Debugger|episode/u,
    );
  }
  assert.equal(
    plainHostNotice(developer[2]),
    "Ultimate is not allowed on this target right now. Your order is unchanged.",
  );
  assert.equal(plainHostNotice("Step: Stay."), "Step: Stay.");
  assert.equal(plainHostNotice(""), "");
});

test("Do This Now never asks for a target when none is allowed", () => {
  const base = /** @type {const} */ ({
    stopped: false,
    terminal: false,
    phase: "plan",
    watchMode: false,
    running: false,
    focusedManual: true,
    lastManualName: "Priest",
    fallen: null,
    targetSet: false,
    lane: "none",
    ultimateOkOnTarget: false,
    ultimateNeedsNoTarget: false,
    ultimateName: "Holy Word: Salvation",
    untouchedOther: null,
  });
  const none = doThisNow({ ...base, allowedCount: 0 });
  assert.equal(
    none.text,
    "No target is allowed right now. Pick a step, then press Lock In Turn.",
  );
  assert.equal(none.beacon, "#pad");
  const burst = doThisNow({
    ...base,
    ultimateNeedsNoTarget: true,
    ultimateName: "Burst",
    allowedCount: 0,
    noTargetUltimateOk: true,
  });
  assert.equal(
    burst.text,
    "No target is allowed right now. Press 2 for Burst, or pick a step and lock in.",
  );
  assert.equal(burst.beacon, "#ultimate-button");
  const kept = doThisNow({ ...base, targetSet: true, allowedCount: 0 });
  assert.equal(
    kept.text,
    "Nothing is allowed on your target right now. Pick a step, then lock in with No Combat.",
  );
  for (const step of [none, burst, kept]) {
    assert.doesNotMatch(step.text, /Allowed Now|hold Shift/u);
  }
  // With a target allowed it still teaches Shift + click, as the vision does.
  assert.match(
    doThisNow({ ...base, allowedCount: 2 }).text,
    /^Pick a target: hold Shift/u,
  );
});
