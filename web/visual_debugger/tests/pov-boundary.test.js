/**
 * @file Check the Agent POV boundary of the arena look modules and the camera by
 * reading their source. No file under `src/looks/` (which holds the camera look)
 * and not `layout.js` (which holds the camera frame math) may import an
 * `authorizedPresentationResearcher*` helper or the presentation adapter or
 * normalizer, or name `researcher_space` or `match_summary` (or their camel-case
 * forms) in code; the same holds for every module they import, followed through
 * the whole import graph. The renderer (`scene.js`) reads no researcher data
 * at all: arena tooltips (aura sources, status cards) come only from the
 * current audience's own scene, so an Agent POV card never names an unseen
 * source.
 */
import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SOURCE_ROOT = fileURLToPath(new URL("../src/", import.meta.url));
const FORBIDDEN_MODULES = new Set([
  "authorized-presentation-adapter.js",
  "authorized-presentation-normalizer.js",
]);
const FORBIDDEN_CODE = [
  /authorizedPresentationResearcher/u,
  /researcher_space/u,
  /researcherSpace/u,
  /match_summary/u,
  /matchSummary/u,
];

// Comments are dropped so documentation may name the boundary in plain words.
/** @param {string} source */
function codeOnly(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//gu, "")
    .split("\n")
    .map((line) => line.replace(/(^|[^:"'`])\/\/.*$/u, "$1"))
    .join("\n");
}

/** @param {string} source */
function importedPaths(source) {
  return [
    ...source.matchAll(
      /(?:import|export)\s+(?:[^"';]*?\sfrom\s+)?["'](\.{1,2}\/[^"']+)["']/gu,
    ),
    ...source.matchAll(/import\(\s*["'`](\.{1,2}\/[^"'`]+)["'`]\s*\)/gu),
  ].map((match) => match[1]);
}

/**
 * @param {ReadonlyArray<string>} entries
 * @returns {Promise<Map<string, string>>}
 */
async function importGraph(entries) {
  /** @type {Map<string, string>} */
  const seen = new Map();
  const pending = [...entries];
  while (pending.length > 0) {
    const path = /** @type {string} */ (pending.pop());
    if (seen.has(path)) {
      continue;
    }
    const source = await readFile(path, "utf8");
    seen.set(path, source);
    for (const imported of importedPaths(source)) {
      pending.push(resolve(dirname(path), imported));
    }
  }
  return seen;
}

test("look modules and the camera frame math never read researcher or match summary data", async () => {
  const lookFiles = (await readdir(resolve(SOURCE_ROOT, "looks")))
    .filter((name) => name.endsWith(".js"))
    .map((name) => resolve(SOURCE_ROOT, "looks", name));
  assert.ok(lookFiles.some((path) => path.endsWith("/looks/camera.js")));
  const graph = await importGraph([...lookFiles, resolve(SOURCE_ROOT, "layout.js")]);
  assert.ok(graph.size > lookFiles.length, "the walk follows shared imports");
  for (const [path, source] of graph) {
    const name = path.slice(SOURCE_ROOT.length);
    assert.equal(
      FORBIDDEN_MODULES.has(name),
      false,
      `${name} is reachable from a look or the camera`,
    );
    for (const imported of importedPaths(source)) {
      const target = resolve(dirname(path), imported).slice(SOURCE_ROOT.length);
      assert.equal(FORBIDDEN_MODULES.has(target), false, `${name} imports ${target}`);
    }
    const code = codeOnly(source);
    for (const pattern of FORBIDDEN_CODE) {
      assert.doesNotMatch(code, pattern, `${name} names ${pattern.source}`);
    }
  }
});

test("the renderer reads no researcher data for the arena", async () => {
  const scene = codeOnly(await readFile(resolve(SOURCE_ROOT, "scene.js"), "utf8"));
  assert.doesNotMatch(
    scene,
    /authorizedPresentationResearcher|researcherScene|researcherAgent/u,
    "scene.js never asks for the researcher scene, not even for tooltips",
  );
  assert.doesNotMatch(scene, /researcher_space|match_summary/u);
  assert.match(scene, /champions\.records\(\{\s*scene,/u);
  // Agent POV status cards are the POV-only kinds, with no joined source rows.
  assert.match(scene, /audience === "agent_pov"\s*\?\s*\{ kind: "pov-status"/u);
  assert.match(scene, /return \{ kind: "pov-overflow", items: rawStatuses/u);
});
