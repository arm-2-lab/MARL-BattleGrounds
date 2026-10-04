/**
 * @file Check the browser half of the game-over Honors host fact: the frame
 * field match_honors (the Verdict card's Honors and Match Summary rows) passes
 * the normalizer only once the game is over, as the Python protocol requires.
 *
 * Contracts checked:
 * - A frame's own match_summary must show a finished game (outcome
 *   "team_a_win", "team_b_win" or "draw") for Honors to pass: Honors on a
 *   mid-game frame are refused in Oracle View and in Agent POV (where every
 *   champion's totals would name champions the team cannot see), and so are
 *   Honors on a frame without a match summary.
 * - The same Agent POV frame with a finished game passes with every
 *   champion's Honors (team_id null: nothing is left to hide at game over),
 *   and an absent or null field still passes on any frame.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

const REFUSED = /Match honors come only with a match summary of a finished game/u;

function everyChampionHonors() {
  return {
    schema_version: 1,
    team_id: null,
    labels: [],
    honors: [],
    champions: Array.from({ length: 10 }, (_, slot) => ({
      public_agent_id: `agent-slot-${slot}`,
      team_id: slot < 5 ? 1 : 2,
      class_id: (slot % 5) + 1,
      kill_contributions: 0,
      deaths: 0,
      damage_done: 0,
      effective_healing_done: 0,
      red_zone_deaths: 0,
      ultimate_activations: 0,
    })),
  };
}

/** @param {string} kind @param {Record<string, any> | null} [summary] */
function withHonors(kind, summary) {
  const raw = structuredClone(fixture.presentations[kind]);
  if (summary !== undefined) raw.match_summary = summary;
  raw.match_honors = everyChampionHonors();
  return raw;
}

/** @param {string} outcome */
function teamDeathmatchSummary(outcome) {
  return {
    ...structuredClone(fixture.presentations.live_shared_obs_agent_pov.match_summary),
    task_mode: 1,
    score_threshold: 1,
    scores: outcome === "team_a_win" ? [1, 0] : [0, 0],
    outcome,
  };
}

test("Honors on a frame from before game over are refused in both views", async () => {
  // The fixture frames are mid-game (Combat Diagnostic: not_applicable).
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(withHonors("replay_oracle")),
    REFUSED,
  );
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(withHonors("live_shared_obs_agent_pov")),
    REFUSED,
  );
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(
      withHonors("live_shared_obs_agent_pov", teamDeathmatchSummary("in_progress")),
    ),
    REFUSED,
  );
  // An Agent POV frame without a match summary cannot show the game ended.
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(
      withHonors("replay_no_shared_obs_agent_pov"),
    ),
    REFUSED,
  );
});

test("an Agent POV frame of a finished game passes with every champion's Honors", async () => {
  const frame = await normalizeAuthorizedPresentationFrameV1(
    withHonors("live_shared_obs_agent_pov", teamDeathmatchSummary("team_a_win")),
  );
  assert.equal(frame.match_honors?.team_id, null);
  assert.equal(frame.match_honors?.champions.length, 10);
  for (const kind of ["replay_oracle", "live_shared_obs_agent_pov"]) {
    const raw = structuredClone(fixture.presentations[kind]);
    raw.match_honors = null;
    await normalizeAuthorizedPresentationFrameV1(raw);
    delete raw.match_honors;
    await normalizeAuthorizedPresentationFrameV1(raw);
  }
});
