/**
 * @file Check the Palette choice (`src/palette.js`, `src/styles/palette.css`).
 *
 * Contracts: Moonlit is the default and first choice, High Contrast second;
 * applyPalette sets or removes `data-palette="high-contrast"` on the root and
 * refuses an unknown palette; currentPalette reads it back.
 * The High
 * Contrast stylesheet redefines only colour tokens under
 * `:root[data-palette="high-contrast"]`, uses the store's Full Moon Contrast
 * values, never touches `--export-backdrop`, `--export-backdrop-grid` or the
 * battlefield shell's background, and every token it sets exists in
 * styles.css or obstacles.css. One High Contrast switch (high_contrast) is
 * the palette control in all three clients, on by default (owner, 4 Oct
 * 2026); there is no Palette select and no switch named "palette".
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import * as palette from "../src/palette.js";
import { VISUAL_FILTER_IDS, visualFiltersForProduct } from "../src/visual-filters.js";

/** @returns {any} */
function fakeDocument() {
  /** @param {string} tag @returns {any} */
  const node = (tag) => {
    /** @type {Map<string, string>} */
    const attributes = new Map();
    /** @type {Map<string, Function>} */
    const listeners = new Map();
    /** @type {any} */
    const element = {
      tagName: tag,
      children: [],
      textContent: "",
      title: "",
      value: "",
      className: "",
      id: "",
      ownerDocument: null,
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      addEventListener: (
        /** @type {string} */ type,
        /** @type {Function} */ listener,
      ) => listeners.set(type, listener),
      dispatch: (/** @type {string} */ type) => listeners.get(type)?.(),
      append: (/** @type {any[]} */ ...nodes) => {
        element.children.push(...nodes);
      },
    };
    element.ownerDocument = doc;
    return element;
  };
  /** @type {any} */
  const doc = { createElement: (/** @type {string} */ tag) => node(tag) };
  doc.documentElement = node("html");
  return doc;
}

/** @param {string} css @param {string} selector @returns {Map<string, string>} */
function tokensIn(css, selector) {
  const start = css.indexOf(`${selector} {`);
  assert.notEqual(start, -1, `${selector} block exists`);
  const body = css.slice(start, css.indexOf("}", start));
  return new Map(
    [...body.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/gu)].map((match) => [
      match[1],
      match[2].trim(),
    ]),
  );
}

test("Moonlit is the default; applyPalette sets and clears the root attribute", () => {
  assert.deepEqual(palette.PALETTE_IDS, ["moonlit", "high_contrast"]);
  assert.equal(palette.DEFAULT_PALETTE, "moonlit");
  const doc = fakeDocument();
  const root = doc.documentElement;
  assert.equal(palette.currentPalette(root), "moonlit");
  assert.equal(palette.applyPalette(root, "high_contrast"), "high_contrast");
  assert.equal(root.getAttribute("data-palette"), "high-contrast");
  assert.equal(palette.currentPalette(root), "high_contrast");
  palette.applyPalette(root, "moonlit");
  assert.equal(root.getAttribute("data-palette"), null);
  assert.throws(() => palette.applyPalette(root, "print"), RangeError);
  root.setAttribute("data-palette", "something-else");
  assert.equal(palette.currentPalette(root), "moonlit");
});

test("High Contrast swaps colour tokens but never the export backdrop", async () => {
  const css = await readFile(
    new URL("../src/styles/palette.css", import.meta.url),
    "utf8",
  );
  const styles = await readFile(new URL("../styles.css", import.meta.url), "utf8");
  const obstacles = await readFile(
    new URL("../src/styles/obstacles.css", import.meta.url),
    "utf8",
  );
  const tokens = tokensIn(css, ':root[data-palette="high-contrast"]');
  assert.doesNotMatch(css, /--export-backdrop[a-z-]*\s*:/u);
  assert.doesNotMatch(css, /\.battlefield-shell\s*\{[^}]*background/u);
  for (const [name, value] of [
    ["--battlefield", "#0f1522"],
    ["--team-a", "#5ca0ff"],
    ["--team-b", "#ff5c5c"],
    ["--life-ivory", "#f7f2ea"],
    ["--moon-silver", "#e0e7f3"],
    ["--ui-muted", "#b3bdd0"],
    ["--panel", "#0c121e"],
    ["--stone-pillar", "#5c6983"],
    ["--class-mage", "#d2bcfa"],
  ]) {
    assert.equal(tokens.get(name), value, name);
  }
  const known = new Set(
    [...`${styles}\n${obstacles}`.matchAll(/^\s*(--[a-z0-9-]+)\s*:/gmu)].map(
      (match) => match[1],
    ),
  );
  for (const name of tokens.keys()) {
    assert.ok(
      known.has(name),
      `${name} is a token styles.css or obstacles.css defines`,
    );
  }
});

test("one High Contrast switch, on by default, in all three clients", () => {
  assert.ok(VISUAL_FILTER_IDS.includes("high_contrast"));
  assert.ok(!VISUAL_FILTER_IDS.includes(/** @type {any} */ ("palette")));
  for (const product of /** @type {const} */ (["combat_debugger", "replay_viewer"])) {
    const rows = visualFiltersForProduct(product).filter(
      ({ id }) => id === "high_contrast",
    );
    assert.equal(rows.length, 1, product);
    assert.equal(rows[0].label, "High Contrast");
    assert.equal(rows[0].defaultEnabled, true);
  }
  assert.equal("createPaletteControl" in palette, false);
});
