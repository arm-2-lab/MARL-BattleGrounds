/**
 * @file Check that public client startup and assets need no private editor code.
 * Browser tests separately prove live and replay startup with their actual servers.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("the public page has no private editor markup or startup import", async () => {
  const [markup, main, api, styles, summary] = await Promise.all(
    [
      "index.html",
      "src/main.js",
      "src/api.js",
      "styles.css",
      "src/match-summary.js",
    ].map((path) => readFile(new URL(path, root), "utf8")),
  );
  assert.doesNotMatch(markup, /devclient-|authoring-/u);
  assert.doesNotMatch(main, /dev-client\.js|authoring_available|devclient-/u);
  assert.doesNotMatch(api, /api\/dev\/authoring/u);
  assert.doesNotMatch(styles, /devclient|authoring/u);
  assert.doesNotMatch(summary, /taskSelect/u);
  assert.match(markup, /src="\/src\/main\.js"/u);
  assert.match(main, /combat_debugger: "MARL-BattleGrounds BattleClient"/u);
});
