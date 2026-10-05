/**
 * @file Check the plain words of the page's own control help and accessible
 * names (owner ruling M13, 3 Oct 2026): the control-help table in
 * src/main.js uses no developer words (authorized, immutable, captured,
 * transition, Python, artifact, serialized, clamped, deterministic, prefix,
 * wire), its rows read as the tips lane's word table gives them, and a
 * body's accessible label says what a click does ("Click to give it orders
 * or look at it."). The status lifecycle names in src/vocabulary.js and the
 * event cards in src/looks/look-support.js read as plain sentences.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { explainChoreographyEvent } from "../src/looks/look-support.js";
import { resolveVisualToken } from "../src/vocabulary.js";

const main = readFileSync(new URL("../src/main.js", import.meta.url), "utf8");
const table = main.slice(
  main.indexOf("const CONTROL_HELP = Object.freeze(["),
  main.indexOf("]);", main.indexOf("const CONTROL_HELP = Object.freeze([")),
);

test("the control-help table speaks plain English", () => {
  assert.ok(table.length > 1000, "found the table");
  assert.doesNotMatch(
    table,
    /authoriz|immutable|captured|transition|Python|artifact|serialized|clamped|deterministic|prefix|wire/iu,
  );
  for (const row of [
    '["#help-button", "How To Play", "Open the guide to the keys and controls."]',
    '"Close the client and stop its server safely."',
    '"Start a new match. If this game is not saved, you choose Save, Discard or Cancel first."',
    '"Drag to pick a tick; the replay jumps there when you let go."',
    '["#replay-first-button", "Start Replay Tick", "Go to tick 0."]',
    '["[data-key=\'Escape\']", "Clear Target", "Clear the target and turn keys off."]',
  ]) {
    assert.ok(table.includes(row), row);
  }
  assert.ok(main.includes(". Click to give it orders or look at it.`"));
});

test("status lifecycle names and event cards are plain sentences", () => {
  assert.equal(
    resolveVisualToken("lifecycle", "decremented").accessibleName,
    "Status has one tick less left",
  );
  assert.equal(
    resolveVisualToken("lifecycle", "cleared_unclassified").accessibleName,
    "Status ended; the record does not say why",
  );
  const card = explainChoreographyEvent({
    kind: "semantic_pulse",
    cueSemantic: "agent_respawned",
    eventType: "agent_respawned",
    eventId: "back",
  });
  assert.equal(card.title, "Returned");
  assert.equal(card.summary, "This champion came back at its waystone this tick.");
});
