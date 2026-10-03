/**
 * @file Check the CSS rules that keep the live combat screen (the BattleClient
 * and the DevClient combat screen) on one screen at every desktop and laptop
 * size (owner, 2 Oct: 1280 x 720 up to 1920 x 1080): #bc is never taller than
 * the window (max-height 100dvh); its top bar and scorebug rows are
 * max-content, so the grid never shrinks them to their min-height and the
 * DevClient's setup line never runs over the scorebug; the stage row is at
 * least 420 px and takes the rest; the deck and the status line never shrink;
 * only the arena gets shorter, down to 120 px; and the rail adds no height
 * (height 0, min-height 100%) and scrolls on its own, so long rail content
 * never pushes Lock In or the status line off screen. The measured proof at
 * the six sizes is a browser check outside this file.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const SCOPE = 'html[data-product-kind="combat_debugger"]';

/** @param {string} name @returns {Promise<string>} */
async function battleClientCss(name) {
  const text = await readFile(
    new URL(`../src/styles/battleclient/${name}`, import.meta.url),
    "utf8",
  );
  return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\s+/g, " ");
}

/** @param {string} text @param {string} selector @returns {string} */
function ruleBody(text, selector) {
  const start = text.indexOf(`${SCOPE} ${selector} {`);
  assert.notEqual(start, -1, `missing rule ${selector}`);
  return text.slice(start, text.indexOf("}", start));
}

test("#bc is one window tall with max-content top rows and a 420 px stage floor", async () => {
  const body = ruleBody(await battleClientCss("layout.css"), ".bc");
  assert.match(
    body,
    /grid-template-rows: max-content max-content minmax\(420px, 1fr\);/,
  );
  assert.match(body, /max-height: 100dvh;/);
});

test("the deck and status line keep their size and the rail scrolls without adding height", async () => {
  const text = await battleClientCss("layout.css");
  assert.match(
    text,
    /\.bc-stage > :is\(#command-deck, #status-line\) \{ flex: none; \}/,
  );
  const rail = ruleBody(text, ".bc .bc-rail");
  assert.match(rail, /height: 0;/);
  assert.match(rail, /min-height: 100%;/);
  assert.match(rail, /overflow-y: auto;/);
});

test("only the arena shrinks, never below 120 px", async () => {
  const arena = ruleBody(await battleClientCss("arena.css"), ".bc #arena");
  assert.match(arena, /flex: 0 1 auto;/);
  assert.match(arena, /min-height: 120px;/);
});
