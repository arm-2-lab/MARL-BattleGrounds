/**
 * @file Check the Last Watch Veil's episode limit. The adapter's
 * authorizedPresentationEpisodeLimit returns match_summary.episode_limit (a
 * positive whole number of ticks) for an approved frame and the same number in
 * Oracle View and Agent POV, and null for an unapproved value or a missing, zero
 * or broken limit. With Time Of Night on, a tick near that limit gives a
 * last-watch-veil record in both audiences, and Oracle -> POV -> Oracle gives the
 * same veil. scene.js passes the helper's value to the atmosphere look and still
 * never names match_summary itself.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { authorizedPresentationEpisodeLimit } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { createViewportTransform } from "../src/layout.js";
import * as atmosphere from "../src/looks/atmosphere.js";

const FIXTURE = new URL("./fixtures/authorized-presentations-v1.json", import.meta.url);

/** @param {string} name */
async function frameOf(name) {
  const fixture = JSON.parse(await readFile(FIXTURE, "utf8"));
  return normalizeAuthorizedPresentationFrameV1(
    structuredClone(fixture.pairs[name].presentation),
  );
}

/** @param {unknown} frame @param {number} tick */
function veilFor(frame, tick) {
  const transform = createViewportTransform({
    worldWidth: 20,
    worldHeight: 12,
    viewportWidth: 960,
    viewportHeight: 600,
    padding: 24,
  });
  return atmosphere
    .records({
      transform,
      map: { width: 20, height: 12, obstacles: [], red_zone: null },
      filters: { time_of_night: true },
      tick,
      episodeLimit: authorizedPresentationEpisodeLimit(frame),
    })
    .find((record) => record.kind === "last-watch-veil");
}

test("the episode limit is the same public number in both audiences", async () => {
  const oracle = await frameOf("live_oracle");
  const pov = await frameOf("live_no_shared_obs_agent_pov");
  assert.equal(authorizedPresentationEpisodeLimit(oracle), 300);
  assert.equal(authorizedPresentationEpisodeLimit(pov), 300);
  assert.equal(authorizedPresentationEpisodeLimit(null), null);
  assert.equal(
    authorizedPresentationEpisodeLimit({ match_summary: { episode_limit: 300 } }),
    null,
  );
});

test("a missing, zero or broken limit gives null", async () => {
  const fixture = JSON.parse(await readFile(FIXTURE, "utf8"));
  for (const bad of [0, -5, 2.5, "300", null]) {
    const raw = structuredClone(fixture.pairs.live_oracle.presentation);
    raw.match_summary.episode_limit = bad;
    /** @type {unknown} */
    let frame = null;
    try {
      frame = await normalizeAuthorizedPresentationFrameV1(raw);
    } catch {
      // The normalizer may refuse a broken limit outright; that is also no limit.
      continue;
    }
    assert.equal(authorizedPresentationEpisodeLimit(frame), null, String(bad));
  }
});

test("Time Of Night draws the veil near the limit in both audiences", async () => {
  const oracle = await frameOf("live_oracle");
  const pov = await frameOf("live_no_shared_obs_agent_pov");
  assert.equal(veilFor(oracle, 100), undefined);
  const first = veilFor(oracle, 299);
  assert.ok(first, "Oracle View draws the veil near the limit");
  assert.deepEqual(veilFor(pov, 299), first, "Agent POV draws the same veil");
  assert.deepEqual(veilFor(oracle, 299), first, "Oracle -> POV -> Oracle is unchanged");
});

test("scene.js passes the adapter's limit and never names match_summary", async () => {
  const source = await readFile(new URL("../src/scene.js", import.meta.url), "utf8");
  const code = source.replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gmu, "");
  assert.match(code, /episodeLimit: authorizedPresentationEpisodeLimit\(frame\)/u);
  assert.doesNotMatch(code, /match_summary|researcher_space/u);
});
