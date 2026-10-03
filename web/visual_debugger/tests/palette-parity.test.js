/**
 * @file Check that the palette has one owner and that every copy follows it.
 * The :root block of styles.css holds the Runefield V5 team, class, Grave
 * Bone and Withered Gold values. The static Matplotlib palette in
 * scene_geometry.py copies the team, class, floor and text colours exactly
 * and keeps no second copy of a team or class colour. No browser script
 * hard-codes a team or class colour. The PNG export backdrop token is exactly
 * #111827 with its 1 px grid at 17%, equals the export lock in
 * replay-export.js and paints .battlefield-shell through one rule; no look
 * stylesheet under src/ redefines the tokens or repaints the backdrop.
 */
import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

import { REPLAY_BATTLEFIELD_BACKGROUND_V1 } from "../src/replay-export.js";

const webRoot = new URL("../", import.meta.url);
const stylesUrl = new URL("styles.css", webRoot);
const sceneGeometryUrl = new URL(
  "../../src/marl_battlegrounds/rendering/scene_geometry.py",
  webRoot,
);

const V5_TEAM_AND_CLASS = Object.freeze({
  "--team-a": "#3d8bff",
  "--team-b": "#ef4b4b",
  "--class-mage": "#b99cff",
  "--class-warrior": "#d8904f",
  "--class-hunter": "#8cc84b",
  "--class-rogue": "#3fd0a0",
  "--class-priest": "#f5c95a",
});

const PYTHON_CONSTANT_TO_TOKEN = Object.freeze({
  _BACKGROUND: "--background",
  _BATTLEFIELD: "--battlefield",
  _TEXT: "--text",
  _MUTED: "--text-muted",
  _TEAM_A: "--team-a",
  _TEAM_B: "--team-b",
});

const RETIRED_TEAM_AND_CLASS = Object.freeze([
  "#3b82f6",
  "#f05a67",
  "#22d3ee",
  "#d18b47",
  "#84cc16",
  "#facc15",
]);

/** @param {string} css */
function withoutComments(css) {
  return css.replaceAll(/\/\*[\s\S]*?\*\//gu, "");
}

/** @param {string} css @returns {Map<string, string>} */
function rootTokens(css) {
  const body = /(?:^|\})\s*:root\s*\{([^{}]*)\}/u.exec(withoutComments(css))?.[1];
  assert.ok(body, "styles.css must have a :root token block");
  const tokens = new Map();
  for (const [, name, value] of body.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/gu)) {
    assert.equal(tokens.has(name), false, `${name} is declared twice in :root`);
    tokens.set(name, value.trim().toLowerCase());
  }
  return tokens;
}

/** @param {string} css @returns {{selector: string, body: string}[]} */
function innermostRules(css) {
  return [...withoutComments(css).matchAll(/([^{}]+)\{([^{}]*)\}/gu)].map(
    ([, selector, body]) => ({ selector: selector.trim(), body }),
  );
}

/** @param {{selector: string, body: string}} rule */
function setsShellBackground({ selector, body }) {
  return /[.#]battlefield-shell(?![\w-])/u.test(selector) && /\bbackground/u.test(body);
}

/** @param {string} source @returns {Map<string, string>} */
function pythonPalette(source) {
  const palette = new Map();
  for (const [, name, value] of source.matchAll(
    /^(_[A-Z_]+) = "(#[0-9A-Fa-f]{6})"$/gmu,
  )) {
    palette.set(name, value.toLowerCase());
  }
  const classBlock = /^_CLASS_COLORS = \{([^}]*)\}/mu.exec(source)?.[1];
  assert.ok(classBlock, "scene_geometry.py must define _CLASS_COLORS");
  for (const [, name, value] of classBlock.matchAll(
    /"([a-z]+)": "(#[0-9A-Fa-f]{6})"/gu,
  )) {
    palette.set(`_CLASS_COLORS.${name}`, value.toLowerCase());
  }
  return palette;
}

/** @param {string} color */
function rgbChannels(color) {
  const numbers = color.match(/\d+(?:\.\d+)?/gu)?.map(Number) ?? [];
  if (color.startsWith("#")) {
    return [1, 3, 5].map((index) => Number.parseInt(color.slice(index, index + 2), 16));
  }
  const [red, green, blue, alpha = 1] = numbers;
  return [red, green, blue, color.includes("%") ? alpha / 100 : alpha];
}

/** @param {URL} directory @param {string} suffix */
async function filesUnder(directory, suffix) {
  const names = await readdir(directory, { recursive: true });
  return names
    .filter((name) => name.endsWith(suffix))
    .map((name) => new URL(name, directory));
}

test("the :root token block holds the Runefield V5 colours", async () => {
  const tokens = rootTokens(await readFile(stylesUrl, "utf8"));
  for (const [name, value] of Object.entries(V5_TEAM_AND_CLASS)) {
    assert.equal(tokens.get(name), value, name);
  }
  assert.equal(tokens.get("--grave-bone"), "#d8ccb4");
  assert.equal(tokens.get("--grave-bone-print"), "#5a5346");
  assert.equal(tokens.get("--withered-gold"), "#b0b85f");
});

test("the static Matplotlib palette copies the CSS tokens exactly", async () => {
  const [css, python] = await Promise.all([
    readFile(stylesUrl, "utf8"),
    readFile(sceneGeometryUrl, "utf8"),
  ]);
  const tokens = rootTokens(css);
  const palette = pythonPalette(python);
  for (const [constant, token] of Object.entries(PYTHON_CONSTANT_TO_TOKEN)) {
    assert.equal(palette.get(constant), tokens.get(token), `${constant} vs ${token}`);
  }
  const classNames = [...palette.keys()]
    .filter((key) => key.startsWith("_CLASS_COLORS."))
    .map((key) => key.slice("_CLASS_COLORS.".length))
    .sort();
  assert.deepEqual(classNames, ["hunter", "mage", "priest", "rogue", "warrior"]);
  for (const name of classNames) {
    assert.equal(
      palette.get(`_CLASS_COLORS.${name}`),
      tokens.get(`--class-${name}`),
      name,
    );
  }
});

test("scene_geometry.py keeps one copy of each team and class colour", async () => {
  const python = (await readFile(sceneGeometryUrl, "utf8")).toLowerCase();
  for (const value of Object.values(V5_TEAM_AND_CLASS)) {
    assert.equal(python.split(value).length - 1, 1, value);
  }
  for (const value of RETIRED_TEAM_AND_CLASS) {
    assert.equal(python.includes(value), false, value);
  }
});

test("no browser script hard-codes a team or class colour", async () => {
  const scripts = await filesUnder(new URL("src/", webRoot), ".js");
  assert.ok(scripts.length > 0);
  const colours = Object.values(V5_TEAM_AND_CLASS);
  for (const url of scripts) {
    const source = (await readFile(url, "utf8")).toLowerCase();
    for (const colour of colours) {
      assert.equal(source.includes(colour), false, `${url.pathname} has ${colour}`);
    }
  }
});

test("the export backdrop token is #111827 with its locked grid", async () => {
  const css = await readFile(stylesUrl, "utf8");
  const tokens = rootTokens(css);
  assert.equal(tokens.get("--export-backdrop"), "#111827");
  assert.equal(tokens.get("--export-backdrop-grid"), "rgb(42 58 84 / 17%)");
  assert.deepEqual(
    rgbChannels(tokens.get("--export-backdrop") ?? ""),
    rgbChannels(REPLAY_BATTLEFIELD_BACKGROUND_V1.baseColor),
  );
  assert.deepEqual(
    rgbChannels(tokens.get("--export-backdrop-grid") ?? ""),
    rgbChannels(REPLAY_BATTLEFIELD_BACKGROUND_V1.gridColor),
  );
  assert.equal(REPLAY_BATTLEFIELD_BACKGROUND_V1.gridLineCssPixels, 1);
  assert.equal(REPLAY_BATTLEFIELD_BACKGROUND_V1.gridSpacingRem, 2);

  const shellBackgrounds = innermostRules(css).filter(setsShellBackground);
  assert.equal(shellBackgrounds.length, 1);
  const [{ selector, body }] = shellBackgrounds;
  assert.equal(selector, ".battlefield-shell");
  const background = /\bbackground\s*:\s*([^;]+);/u
    .exec(body)?.[1]
    .replaceAll(/\s+/gu, " ");
  assert.equal(
    background,
    "linear-gradient(var(--export-backdrop-grid) 1px, transparent 1px), " +
      "linear-gradient(90deg, var(--export-backdrop-grid) 1px, transparent 1px), " +
      "var(--export-backdrop)",
  );
  assert.match(body, /\bbackground-size\s*:\s*2rem 2rem\s*;/u);
});

test("no other stylesheet redefines or repaints the export backdrop", async () => {
  const stylesheets = [
    stylesUrl,
    ...(await filesUnder(new URL("src/", webRoot), ".css")),
  ];
  let declarations = 0;
  for (const url of stylesheets) {
    const css = withoutComments(await readFile(url, "utf8"));
    const found = css.match(/--export-backdrop(?:-grid)?\s*:/gu) ?? [];
    if (url.href !== stylesUrl.href) {
      assert.deepEqual(found, [], `${url.pathname} redefines the export backdrop`);
      assert.deepEqual(
        innermostRules(css).filter(setsShellBackground),
        [],
        `${url.pathname} paints the arena backdrop`,
      );
    }
    declarations += found.length;
  }
  assert.equal(declarations, 2);
});
