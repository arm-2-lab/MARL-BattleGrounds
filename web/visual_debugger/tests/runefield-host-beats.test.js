/**
 * @file Check the browser half of the host beat facts and next-action ownership.
 * The normalizer accepts match_summary.previous_scores, returned_agent_ids and each
 * death's home_fall, keeps older payloads without them valid, and refuses a score that
 * went down, a repeated or unknown returning agent, and a home fall in a combat
 * diagnostic. authorizedPresentationMatchBeatFacts gives scores before and after in
 * every audience, takes Oracle returns from the match summary, and in Agent POV never
 * reads the global match returns or deaths. authorizedPresentationNextActionOwnership
 * marks every Oracle actor observed and, in Agent POV, only the recipient, with each
 * teammate reading "Not Observed" and no enemy listed.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  authorizedPresentationMatchBeatFacts,
  authorizedPresentationNextActionOwnership,
  NOT_OBSERVED_LABEL,
} from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @param {string} kind @returns {Record<string, any>} */
function rawFrame(kind) {
  return structuredClone(fixture.presentations[kind]);
}

/**
 * @param {Record<string, any>} raw
 * @returns {Record<string, any>}
 */
function teamDeathmatch(raw) {
  raw.match_summary.task_mode = 1;
  raw.match_summary.score_threshold = 20;
  raw.match_summary.outcome = "in_progress";
  return raw;
}

/** @param {Record<string, any>} raw @returns {Record<string, any>} */
function sceneAgents(raw) {
  return (raw.current_endpoint.parts ?? raw.current_endpoint).scene.agents;
}

test("older payloads without the beat fields still validate and hold no change", async () => {
  const raw = rawFrame("replay_oracle");
  delete raw.match_summary.previous_scores;
  delete raw.match_summary.returned_agent_ids;
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  assert.deepEqual(authorizedPresentationMatchBeatFacts(frame), {
    scores_before: [0, 0],
    scores_after: [0, 0],
    score_changed: false,
    returned_public_agent_ids: [],
    home_fall_public_agent_ids: [],
  });
});

test("Oracle beat facts read the score before the tick and the returns", async () => {
  const raw = teamDeathmatch(rawFrame("replay_oracle"));
  const victim = sceneAgents(raw).find(
    (/** @type {Record<string, any>} */ agent) => agent.team_id === 1,
  );
  raw.match_summary.scores = [3, 1];
  raw.match_summary.previous_scores = [1, 1];
  raw.match_summary.returned_agent_ids = ["agent-slot-5"];
  raw.match_summary.deaths = [
    {
      public_agent_id: victim.public_agent_id,
      team_id: victim.team_id,
      class_id: victim.class_id,
      killing_team_id: null,
      contributors: null,
      home_fall: true,
    },
  ];
  const facts = authorizedPresentationMatchBeatFacts(
    await normalizeAuthorizedPresentationFrameV1(raw),
  );
  assert.deepEqual(facts?.scores_before, [1, 1]);
  assert.deepEqual(facts?.scores_after, [3, 1]);
  assert.equal(facts?.score_changed, true);
  assert.deepEqual(facts?.returned_public_agent_ids, ["agent-slot-5"]);
  // Home falls come from the audience's own agent_died events, not the summary.
  assert.deepEqual(facts?.home_fall_public_agent_ids, []);
  assert.ok(Object.isFrozen(facts) && Object.isFrozen(facts?.scores_before));
});

test("Agent POV beat facts never read the global match returns or deaths", async () => {
  const raw = teamDeathmatch(rawFrame("live_shared_obs_agent_pov"));
  const roster = raw.researcher_space.roster_agents;
  const enemy = roster.find(
    (/** @type {Record<string, any>} */ agent) => agent.team_id === 2,
  );
  raw.match_summary.scores = [0, 2];
  raw.match_summary.previous_scores = [0, 0];
  raw.match_summary.returned_agent_ids = [enemy.public_agent_id];
  raw.match_summary.deaths = [
    {
      public_agent_id: enemy.public_agent_id,
      team_id: enemy.team_id,
      class_id: enemy.class_id,
      killing_team_id: null,
      contributors: null,
      home_fall: true,
    },
  ];
  const facts = authorizedPresentationMatchBeatFacts(
    await normalizeAuthorizedPresentationFrameV1(raw),
  );
  assert.deepEqual(facts?.scores_before, [0, 0]);
  assert.deepEqual(facts?.scores_after, [0, 2]);
  assert.deepEqual(facts?.returned_public_agent_ids, []);
  assert.deepEqual(facts?.home_fall_public_agent_ids, []);
});

test("impossible beat facts are refused", async () => {
  const roster = sceneAgents(fixture.presentations.replay_oracle);
  /** @type {Array<[string, (match: Record<string, any>) => void, RegExp]>} */
  const cases = [
    [
      "score went down",
      (match) => {
        match.task_mode = 1;
        match.score_threshold = 20;
        match.outcome = "in_progress";
        match.previous_scores = [1, 0];
      },
      /previous scores must not exceed/u,
    ],
    [
      "repeated return",
      (match) => {
        match.returned_agent_ids = ["agent-slot-5", "agent-slot-5"];
      },
      /Match returns must join unique identities/u,
    ],
    [
      "unknown return",
      (match) => {
        match.returned_agent_ids = ["nobody"];
      },
      /Match returns must join unique identities/u,
    ],
    [
      "combat diagnostic home fall",
      (match) => {
        match.task_mode = 0;
        match.score_threshold = 0;
        match.outcome = "not_applicable";
        match.deaths = [
          {
            public_agent_id: roster[0].public_agent_id,
            team_id: roster[0].team_id,
            class_id: roster[0].class_id,
            home_fall: true,
          },
        ];
      },
      /cannot claim a home fall/u,
    ],
  ];
  for (const [name, edit, message] of cases) {
    const raw = rawFrame("replay_oracle");
    edit(raw.match_summary);
    await assert.rejects(normalizeAuthorizedPresentationFrameV1(raw), message, name);
  }
});

test("Oracle View observes every actor's next action", async () => {
  const frame = await normalizeAuthorizedPresentationFrameV1(rawFrame("replay_oracle"));
  const rows = authorizedPresentationNextActionOwnership(frame);
  assert.deepEqual(
    rows.map((row) => [row.public_agent_id, row.observed, row.status_label]),
    sceneAgents(fixture.presentations.replay_oracle).map(
      (/** @type {Record<string, any>} */ agent) => [agent.public_agent_id, true, null],
    ),
  );
  assert.ok(Object.isFrozen(rows) && rows.every((row) => Object.isFrozen(row)));
});

test("Agent POV observes only the recipient; teammates read Not Observed", async () => {
  assert.equal(NOT_OBSERVED_LABEL, "Not Observed");
  for (const kind of [
    "replay_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
    "live_shared_obs_agent_pov",
    "live_no_shared_obs_agent_pov",
  ]) {
    const raw = rawFrame(kind);
    const frame = await normalizeAuthorizedPresentationFrameV1(raw);
    const recipient = raw.authority.recipient_public_agent_id;
    const recipientTeam = sceneAgents(raw).find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === recipient,
    ).team_id;
    const rows = authorizedPresentationNextActionOwnership(frame);
    assert.ok(rows.length > 1, kind);
    for (const row of rows) {
      const own = row.public_agent_id === recipient;
      assert.equal(row.observed, own, kind);
      assert.equal(row.status_label, own ? null : NOT_OBSERVED_LABEL, kind);
      assert.equal(row.actor_team, recipientTeam === 1 ? "team-a" : "team-b", kind);
    }
    assert.equal(rows.filter((row) => row.observed).length, 1, kind);
  }
  assert.deepEqual(authorizedPresentationNextActionOwnership({}), []);
  assert.equal(authorizedPresentationMatchBeatFacts({}), null);
});
