/**
 * @file Check the BattleClient's words after Exit and its Set rule (owner and
 * main-session rulings, 3 Oct 2026), in src/tactics-turn.js and main.js:
 * - closedSentence says the match is closed and, from the Exit answer's
 *   recording status, that the game is saved, that no tick was played so no
 *   replay was saved (Exit before the first tick), or nothing about a replay
 *   when the match records nothing; Do This Now says the same once closed.
 * - recordingWords is the one owner of the Recording pill, popover and seal
 *   words, and once closed while still "recording" with no tick kept it says
 *   "No Replay" instead of a stale "Recording · Tick 0".
 * - turnTitles gives the title "Closed" after Exit, over game over and stop,
 *   and the disabled Lock In button reads "Closed" after Exit and "Game
 *   Over" at game over, never a turn phase such as "Everyone Acts".
 * - orderIsSet: Set only when the server's order is not Stay + No Combat (a
 *   kept target with No Combat is the default), so the pill is the same after
 *   a reload and in both views.
 * - hiddenOrderLine: another team's Manual champion reads "Out Of Sight" when
 *   the view cannot see it and "In Sight · Order Hidden" when it can, never
 *   "Not Observed".
 * - main.js: acceptExit keeps the Exit answer's recording status in
 *   state.exitRecording and says closedSentence; resendsAfterStale lets only
 *   Exit and Save And Replace be sent once more after a stale answer; Exit
 *   or R pressed while the game-over save runs says why nothing happens yet.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  closedSentence,
  DECK_LABELS,
  doThisNow,
  hiddenOrderLine,
  orderIsSet,
  recordingWords,
  turnTitles,
} from "../src/tactics-turn.js";

const MAIN = readFileSync(new URL("../src/main.js", import.meta.url), "utf8");

/** @param {string} name A top-level function declared in main.js. */
function mainFunction(name) {
  const source = new RegExp(
    `\\nfunction ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n\\}`,
    "u",
  ).exec(MAIN)?.[0];
  assert.ok(source, `main.js defines ${name}`);
  return source;
}

const SAVED = { lifecycle: "saved", captured_transition_count: 91 };
const OPEN_EMPTY = { lifecycle: "recording", captured_transition_count: 0 };

test("the closed sentence follows the replay's save state", () => {
  assert.equal(
    closedSentence(SAVED),
    "The match is closed. The game so far is saved. You can close this tab.",
  );
  assert.equal(
    closedSentence(OPEN_EMPTY),
    "The match is closed. No tick was played, so no replay was saved. You can close this tab.",
  );
  assert.equal(closedSentence(null), "The match is closed. You can close this tab.");
  const plan = {
    stopped: false,
    terminal: true,
    phase: /** @type {const} */ ("plan"),
    watchMode: false,
    running: false,
    focusedManual: true,
    lastManualName: "Warrior",
    fallen: null,
    targetSet: false,
    lane: /** @type {const} */ ("none"),
    ultimateOkOnTarget: false,
    ultimateNeedsNoTarget: false,
    ultimateName: "Charge",
    untouchedOther: null,
  };
  for (const recording of [SAVED, OPEN_EMPTY, null]) {
    assert.deepEqual(doThisNow({ ...plan, closed: true, recording }), {
      text: closedSentence(recording),
      beacon: null,
    });
  }
});

test("the Recording words say No Replay after an Exit before the first tick", () => {
  assert.deepEqual(recordingWords(OPEN_EMPTY, 0, 300, true), {
    pill: "No Replay",
    popover: "No Replay · No Tick Was Played",
    seal: "No Replay",
  });
  assert.deepEqual(recordingWords(OPEN_EMPTY, 0, 300), {
    pill: "Recording · Tick 0",
    popover: "0 Ticks Kept Since Tick 0",
    seal: "Recording · Tick 0 of 300",
  });
  assert.deepEqual(recordingWords(SAVED, 300, 300, true), {
    pill: "Saved",
    popover: "Saved · 91 Ticks",
    seal: "Saved · 91 Ticks",
  });
  assert.equal(
    recordingWords({ lifecycle: "sealed", captured_transition_count: 1 }, 300, 300)
      .popover,
    "Saving Replay… · 1 Tick",
  );
  assert.equal(
    recordingWords({ lifecycle: "discarded" }, 4, null).pill,
    "Recording · Discarded",
  );
});

test("the This Turn title reads Closed after Exit", () => {
  const base = {
    phase: /** @type {const} */ ("plan"),
    stopped: true,
    terminal: true,
    watchMode: false,
    running: false,
  };
  assert.equal(turnTitles({ ...base, closed: true }).title, DECK_LABELS.closed);
  assert.equal(DECK_LABELS.closed, "Closed");
  assert.equal(turnTitles(base).title, DECK_LABELS.gameStopped);
  assert.equal(turnTitles({ ...base, closed: true }).lockLabel, "Closed");
  const over = { ...base, stopped: false, phase: /** @type {const} */ ("act") };
  assert.equal(turnTitles(over).lockLabel, "Game Over");
});

test("Set means the server's order is not Stay with No Combat", () => {
  assert.equal(orderIsSet(null), false);
  assert.equal(orderIsSet({ moveAction: 0, lane: "none" }), false);
  assert.equal(orderIsSet({ moveAction: 1, lane: "none" }), true);
  assert.equal(orderIsSet({ moveAction: 0, lane: "basic" }), true);
  assert.equal(orderIsSet({ moveAction: 0, lane: "ultimate" }), true);
});

test("another team's champion reads Out Of Sight or In Sight, never Not Observed", () => {
  assert.equal(hiddenOrderLine(false), "Out Of Sight");
  assert.equal(hiddenOrderLine(true), "In Sight · Order Hidden");
  const rail = readFileSync(new URL("../src/tactics-turn.js", import.meta.url), "utf8");
  assert.doesNotMatch(rail, /NOT_OBSERVED_LABEL/u);
  assert.match(rail, /agent === null \? UNSEEN_TOKEN : championToken\(/u);
});

test("main.js keeps the Exit answer's recording and resends only Exit and Save", () => {
  const accept = mainFunction("acceptExit");
  assert.match(accept, /state\.shuttingDown = true;/u);
  assert.match(accept, /state\.exitRecording =/u);
  assert.match(accept, /setNotice\(closedSentence\(state\.exitRecording\), "info"\);/u);
  assert.match(accept, /paintKeysState\(\);/u);
  /** @type {(command: Record<string, unknown>) => boolean} */
  const resends = new Function(
    `${mainFunction("resendsAfterStale")}\nreturn resendsAfterStale;`,
  )();
  assert.equal(resends({ command_type: "exit" }), true);
  assert.equal(
    resends({
      command_type: "save_and_replace",
      replacement: { command_type: "reset" },
    }),
    true,
  );
  for (const command_type of ["keyboard", "reset", "finish_and_review", "set_view"]) {
    assert.equal(resends({ command_type }), false, command_type);
  }
  assert.doesNotMatch(MAIN, /kind: "closed"/u);
});

test("Exit or R pressed while the replay saves says why nothing happens yet", () => {
  assert.match(
    MAIN,
    /clearNotice\("The replay is still saving\. The match closes once it is saved\."\);/u,
  );
  assert.match(
    MAIN,
    /clearNotice\("The replay is still saving\. The new match starts once it is saved\."\);\n {4}return dispatchCommand\(recordingDecision\.command, \{/u,
  );
});
