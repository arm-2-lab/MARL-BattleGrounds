/**
 * @file Check the normalizer's guard on the Momentum host fact
 * (match_summary.scores_ten_ticks_ago, each team's score 10 ticks ago).
 *
 * Contracts checked:
 * - A frame whose score 10 ticks ago is at most the score before this tick
 *   (previous_scores) and the score now is accepted, and so are older payloads
 *   without the field or with null.
 * - A score 10 ticks ago above the score before this tick, or above the score
 *   now when previous_scores is absent, is refused: a team's score never goes
 *   down.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/**
 * @param {Record<string, unknown>} facts
 * @returns {Record<string, any>}
 */
function teamDeathmatchFrame(facts) {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  Object.assign(raw.match_summary, {
    task_mode: 1,
    score_threshold: 20,
    outcome: "in_progress",
    scores: [3, 1],
    previous_scores: [2, 1],
    ...facts,
  });
  return raw;
}

test("a score 10 ticks ago at most the later scores is accepted", async () => {
  const frame = await normalizeAuthorizedPresentationFrameV1(
    teamDeathmatchFrame({ scores_ten_ticks_ago: [1, 0] }),
  );
  assert.deepEqual(frame.match_summary.scores_ten_ticks_ago, [1, 0]);
  const older = teamDeathmatchFrame({});
  delete older.match_summary.scores_ten_ticks_ago;
  await normalizeAuthorizedPresentationFrameV1(older);
  await normalizeAuthorizedPresentationFrameV1(
    teamDeathmatchFrame({ scores_ten_ticks_ago: null }),
  );
});

test("a score 10 ticks ago above a later score is refused", async () => {
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(
      teamDeathmatchFrame({ scores_ten_ticks_ago: [3, 1] }),
    ),
    /ten ticks ago must not exceed/u,
  );
  const noPrevious = teamDeathmatchFrame({ scores_ten_ticks_ago: [4, 1] });
  delete noPrevious.match_summary.previous_scores;
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(noPrevious),
    /ten ticks ago must not exceed/u,
  );
});
