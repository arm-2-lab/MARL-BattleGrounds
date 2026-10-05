/**
 * @file Check that the Replay Viewer keeps raw error text for developers
 * (src/main.js logRawError; owner, 4 Oct 2026, Micro List M15). The page shows
 * only plain words, and logRawError writes the exact error to the browser
 * console as one line, "MARL-BGs <context>: <text>": an Error gives its name
 * and message, anything else its string form, and runs of whitespace (line
 * breaks included) become one space. main.js calls it where the page replaces
 * an error with plain words: a failed replay move (the playback controller's
 * onError), a failed PNG export, a failed download, and a host warning or
 * error notice that plainHostNotice rewrites. main.js cannot load outside a
 * browser page, so the test lifts logRawError's own source and reads the
 * call sites from the file.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const main = readFileSync(new URL("../src/main.js", import.meta.url), "utf8");

test("logRawError writes one console line with the context and the raw text", () => {
  const source = /function logRawError\(context, error\) \{[\s\S]*?\n\}/u.exec(
    main,
  )?.[0];
  assert.ok(source, "main.js defines logRawError");
  /** @type {unknown[][]} */
  const lines = [];
  const fakeConsole = { error: (/** @type {unknown[]} */ ...args) => lines.push(args) };
  const logRawError = new Function("console", `${source}\nreturn logRawError;`)(
    fakeConsole,
  );
  logRawError("PNG export failed", new TypeError("canvas\n  is   tainted"));
  logRawError("Host notice", "Movement action 3 is unavailable\tnow.");
  assert.deepEqual(lines, [
    ["MARL-BGs PNG export failed: TypeError: canvas is tainted"],
    ["MARL-BGs Host notice: Movement action 3 is unavailable now."],
  ]);
});

test("every place that shows plain words instead of an error also logs it", () => {
  for (const call of [
    'logRawError("Replay move failed", error);',
    'logRawError("PNG export failed", error);',
    'logRawError("Download failed", error);',
    'logRawError("Host notice", message);',
  ]) {
    assert.equal(main.split(call).length - 1, 1, `main.js calls ${call} once`);
  }
  const onError = main.indexOf("onError: (error) => {");
  assert.ok(
    onError > 0 && main.indexOf('logRawError("Replay move failed"', onError) > onError,
  );
  const hostLog = main.indexOf('logRawError("Host notice", message);');
  const translate = main.lastIndexOf(
    "const plain = plainHostNotice(message);",
    hostLog,
  );
  assert.ok(
    translate > 0 && hostLog - translate < 200,
    "logged right after the rewrite",
  );
});
