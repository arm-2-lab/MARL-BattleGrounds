/**
 * @file Check that public client startup and assets need no private editor code,
 * including the page parts main.js starts (shell.js, tactics-turn.js,
 * filter-controls.js), and that shell.js owns the product titles: the tab
 * titles "MARL-BattleGrounds BattleClient" and "MARL-BattleGrounds Replay
 * Viewer", and the product names shown after the brand lockup.
 * Browser tests separately prove live and replay startup with their actual servers.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("the public page has no private editor markup or startup import", async () => {
  const [markup, main, api, styles, summary, shell, tacticsTurn, filterControls] =
    await Promise.all(
      [
        "index.html",
        "src/main.js",
        "src/api.js",
        "styles.css",
        "src/match-summary.js",
        "src/shell.js",
        "src/tactics-turn.js",
        "src/filter-controls.js",
      ].map((path) => readFile(new URL(path, root), "utf8")),
    );
  assert.doesNotMatch(markup, /devclient-|authoring-/u);
  for (const source of [main, shell, tacticsTurn, filterControls]) {
    assert.doesNotMatch(source, /dev-client\.js|authoring_available|devclient-/u);
  }
  assert.doesNotMatch(api, /api\/dev\/authoring/u);
  assert.doesNotMatch(styles, /devclient|authoring/u);
  assert.doesNotMatch(summary, /taskSelect/u);
  assert.match(markup, /src="\/src\/main\.js"/u);
  assert.match(shell, /combat_debugger: "MARL-BattleGrounds BattleClient"/u);
  assert.match(shell, /replay_viewer: "MARL-BattleGrounds Replay Viewer"/u);
  assert.match(
    shell,
    /combat_debugger: "BattleClient",\s*replay_viewer: "Replay Viewer"/u,
  );
});
