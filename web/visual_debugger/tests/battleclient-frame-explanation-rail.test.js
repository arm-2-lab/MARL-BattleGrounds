/**
 * @file Check where the BattleClient and DevClient combat screen show the
 * Frame Explanation list (F). src/main.js inserts #frame-explanation-list
 * after the hidden #this-turn slot, and index.html keeps that slot under the
 * How To Play card, below the bottom of a 1440 by 900 screen. The rail
 * stylesheet (src/styles/battleclient/rail.css) must therefore order the Your
 * Turn card first and the list right under it, dress the list like a rail
 * card, and scope both rules to the rail shared by live play and the Replay
 * Viewer (the Replay Viewer never shows #turn-card, so the order only acts in
 * live play).
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);
// Whitespace-insensitive: the formatter may break a long selector over lines.
const rail = (
  await readFile(new URL("src/styles/battleclient/rail.css", root), "utf8")
).replace(/\s+/gu, " ");
const main = await readFile(new URL("src/main.js", root), "utf8");
const page = await readFile(new URL("index.html", root), "utf8");

/** @param {string} selector */
function ruleBody(selector) {
  const start = rail.indexOf(`${selector} {`);
  assert.notEqual(start, -1, `rail.css has no rule for ${selector}`);
  return rail.slice(start, rail.indexOf("}", start));
}

test("the list still lands after the hidden slot under How To Play", () => {
  assert.match(main, /elements\.thisTurn\.after\(frameExplanationList\)/u);
  assert.match(main, /frameExplanationList\.id = "frame-explanation-list"/u);
  assert.match(
    page,
    /id="howto-card"[\s\S]*?<\/section>[\s\S]*?<section id="this-turn" hidden>/u,
  );
});

test("the rail puts Your Turn first and the list right under it", () => {
  const scope =
    'html:is([data-product-kind="combat_debugger"], [data-product-kind="replay_viewer"]) .bc .bc-rail';
  assert.match(ruleBody(`${scope} > #turn-card`), /order: -2;/u);
  const list = ruleBody(`${scope} > #frame-explanation-list`);
  assert.match(list, /order: -1;/u);
  assert.match(list, /padding: 12px;/u);
  assert.match(list, /border: 1px solid var\(--edge\);/u);
  assert.match(list, /border-radius: 8px;/u);
  assert.match(list, /background: var\(--panel\);/u);
});
