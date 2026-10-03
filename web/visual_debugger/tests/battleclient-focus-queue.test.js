/**
 * @file Check that a focus change is never lost while a live request is out.
 * A portrait, roster or rail click that picks which champion the deck shows
 * (roster_selection, role "control") made while one live request is out joins
 * that request's queue (retainFocusChange) and is sent after it; a later focus
 * change replaces an earlier queued one; order keys already queued stay in
 * order; nothing is kept in the Replay Viewer, with no live request, while the
 * page must resync or is offline, after a queued Lock In, for a full queue or
 * for any other command; and dispatchCommand asks retainFocusChange before it
 * refuses a busy command with "A command is already in flight".
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const MAIN = readFileSync(new URL("../src/main.js", import.meta.url), "utf8");
const SOURCE = /\nfunction retainFocusChange\([^)]*\) \{[\s\S]*?\n\}/u.exec(MAIN)?.[0];

/**
 * @param {{
 *   replay?: boolean,
 *   transaction?: {deferredDrafts: Array<Record<string, unknown>>, deferredSubmit: unknown} | null,
 *   resyncRequired?: boolean,
 *   offline?: boolean,
 * }} [setup]
 */
function retainer({
  replay = false,
  transaction = { deferredDrafts: [], deferredSubmit: null },
  resyncRequired = false,
  offline = false,
} = {}) {
  assert.ok(SOURCE, "main.js defines retainFocusChange");
  /** @type {(command: Record<string, unknown>) => boolean} */
  const retain = new Function(
    "isReplayMode",
    "activeLiveCommandTransaction",
    "state",
    "MAX_DEFERRED_DRAFTS",
    `${SOURCE}\nreturn retainFocusChange;`,
  )(() => replay, transaction, { resyncRequired, offline }, 4);
  return {
    retain,
    transaction: /** @type {{deferredDrafts: Array<Record<string, unknown>>}} */ (
      transaction
    ),
  };
}

/** @param {number} slot @returns {Record<string, unknown>} */
const focus = (slot) => ({
  command_type: "roster_selection",
  role: "control",
  global_slot: slot,
});

test("a focus change made while a request is out is queued, and the last one wins", () => {
  const step = { command_type: "keyboard", key: "w" };
  const { retain, transaction } = retainer({
    transaction: { deferredDrafts: [step], deferredSubmit: null },
  });
  assert.equal(retain(focus(0)), true);
  assert.deepEqual(transaction.deferredDrafts, [step, focus(0)]);
  assert.equal(retain(focus(2)), true);
  assert.deepEqual(transaction.deferredDrafts, [step, focus(2)]);
});

test("nothing is kept where the queue cannot hold a focus change", () => {
  assert.equal(retainer({ replay: true }).retain(focus(0)), false);
  assert.equal(retainer({ transaction: null }).retain(focus(0)), false);
  assert.equal(retainer({ resyncRequired: true }).retain(focus(0)), false);
  assert.equal(retainer({ offline: true }).retain(focus(0)), false);
  const locked = retainer({
    transaction: { deferredDrafts: [], deferredSubmit: { key: "Enter" } },
  });
  assert.equal(locked.retain(focus(0)), false);
  assert.deepEqual(locked.transaction.deferredDrafts, []);
  const keys = ["w", "a", "1", "x"].map((key) => ({ command_type: "keyboard", key }));
  const full = retainer({
    transaction: { deferredDrafts: [...keys], deferredSubmit: null },
  });
  assert.equal(full.retain(focus(0)), false);
  assert.deepEqual(full.transaction.deferredDrafts, keys);
  const other = retainer();
  assert.equal(
    other.retain({ command_type: "roster_selection", role: "target", global_slot: 5 }),
    false,
  );
  assert.equal(other.retain({ command_type: "keyboard", key: "w" }), false);
  assert.deepEqual(other.transaction.deferredDrafts, []);
});

test("dispatchCommand queues a focus change before refusing a busy command", () => {
  assert.match(
    MAIN,
    /if \(state\.busy && !state\.shuttingDown && retainFocusChange\(command\)\) \{\s*renderConnection\(\);\s*return;\s*\}\s*if \(state\.busy \|\| state\.shuttingDown\) \{\s*setNotice\("A command is already in flight; no second command was sent\."/u,
  );
});
