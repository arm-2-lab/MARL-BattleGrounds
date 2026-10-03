/**
 * @file Check three page-start contracts in main.js by reading its source.
 * (1) The page draws the match moment cards from looks/heralds.js: the Lineup
 * card, the Verdict card (by product, through installedProductKind; the Replay
 * Viewer and live game over) and
 * the Match Point or Ticks Left state chip, in its own group in the renderer's
 * herald layer, after the choreographer presents each tick and after a resize.
 * Champions on the cards come only from the audience's authorized scene; the
 * match facts come from the public scoreboard reader (matchSummaryView), never
 * from researcher_space. (2) No load moves keyboard focus to the replay
 * timeline, so its help card is not open at rest. (3) Routine loads, waits and
 * frame updates make no top-bar notice; they go through clearNotice.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const mainUrl = new URL("../src/main.js", import.meta.url);

/**
 * @param {string} source
 * @param {string} name
 */
function functionBody(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} is missing`);
  const end = source.indexOf("\n}\n", start);
  return source.slice(start, end);
}

test("main draws the lineup, verdict and state cards from heralds", async () => {
  const source = await readFile(mainUrl, "utf8");
  assert.match(source, /import \* as heralds from "\.\/looks\/heralds\.js";/u);
  const records = functionBody(source, "matchMomentCardRecords");
  for (const part of ["lineup", "verdict", "state"]) {
    assert.match(records, new RegExp(`part: "${part}"`, "u"));
  }
  assert.match(records, /authorizedPresentationSceneView\(presentation\)/u);
  assert.match(records, /matchSummaryView\(presentation\)/u);
  assert.match(records, /product: installedProductKind\(\)/u);
  assert.doesNotMatch(records, /researcher_space|match_summary/u);
  const draw = functionBody(source, "renderMatchMomentCards");
  assert.match(draw, /matchMomentCardsGroup\?\.remove\(\)/u);
  assert.match(draw, /surface\.heraldLayer\.append\(group\)/u);
  assert.match(draw, /heralds\.build\(records, doc\)/u);
  const page = functionBody(source, "renderPage");
  assert.ok(
    page.indexOf("renderMatchMomentCards(presentationFrame)") >
      page.indexOf("choreographer.presentFrame("),
  );
  assert.match(
    functionBody(source, "scheduleBattlefieldResize"),
    /renderMatchMomentCards\(presentationFrame\)/u,
  );
  assert.match(
    functionBody(source, "clearPresentationAuthority"),
    /renderMatchMomentCards\(null\)/u,
  );
});

test("no load moves focus to the replay timeline", async () => {
  const source = await readFile(mainUrl, "utf8");
  assert.doesNotMatch(source, /replayTimeline\.focus\(/u);
});

test("routine loads and frame updates make no top-bar notice", async () => {
  const source = await readFile(mainUrl, "utf8");
  for (const routine of [
    "Authoritative frame updated.",
    "Read-only replay frame updated.",
    "Connected to the local product service.",
    "Read-only replay review is ready.",
    "Waiting for the authoritative Python response",
    "Waiting for the read-only replay response",
  ]) {
    assert.equal(source.includes(routine), false, routine);
  }
  const clear = functionBody(source, "clearNotice");
  assert.match(clear, /state\.notice = null;/u);
  assert.match(functionBody(source, "loadCurrentFrame"), /clearNotice\(\)/u);
});
