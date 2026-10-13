/**
 * @file Check the BattleClient's live scorebug models: liveScorebugView
 * (src/match-summary.js) and liveControllers, controllerLine, yoursTeam and
 * liveScorebugModel (src/looks/hud.js).
 *
 * Contracts checked:
 * - Scores: "/ 20", 20 segments lit up to the score, the side Team A sits on
 *   (its spawn end), "Tick 91 of 300" and "First to 20"; the Scorebug switch
 *   off gives the plain "Score: 2/20" with no goal and no bar; a scoring tick
 *   keeps the old score and bar for the Fall beat; no summary gives nothing.
 * - Respawn wave: "Respawn Wave in N Ticks" with N = the soonest countdown + 1
 *   (Core brings champions back on the transition that starts at countdown
 *   0), and "1 Tick" for one.
 * - Controllers: controllerLine gives a team's full words, "You + <policy>"
 *   for a team with a Manual champion, the policy label alone otherwise,
 *   distinct labels joined by " + " (the scorebug shows them only when they
 *   fit, else the short line, "You + 1 Policy"); the model's lines and its one
 *   controller model (roster, controller-names.js) agree; the "Yours" tag
 *   stays on Team A unless every Manual slot is on Team B; malformed
 *   configurations give no controllers.
 * - Portraits: every Manual champion reads "You", others "ID n"; the focus
 *   look follows the controlled champion; a fallen champion has an empty bar;
 *   Cinder at 30% life or less; a fall this tick holds the living look with
 *   the life from before the blow and a return holds the fallen look.
 * - Agent POV boundary: an unseen enemy's portrait reads "Unseen" with no
 *   life, hold or focus, and its click sentence names "your team" (or the
 *   viewing team in Policies Only); Oracle View -> Agent POV -> Oracle View
 *   gives exactly a fresh model.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  controllerLine,
  liveControllers,
  liveScorebugModel,
  NO_CONTROLLERS,
  records,
  yoursTeam,
} from "../src/looks/hud.js";
import {
  collapsedControllerLine,
  controllerLineParts,
} from "../src/controller-names.js";
import { liveScorebugView } from "../src/match-summary.js";

const CLASSES = [1, 2, 3, 4, 5];

/** @param {Record<string, any>} [overrides] */
function summary(overrides = {}) {
  return {
    threshold: 20,
    tick: { current: 91, limit: 300, text: "Tick 91 of 300" },
    teams: [
      { teamId: 1, side: "right", scoreValue: 2 },
      { teamId: 2, side: "left", scoreValue: 4 },
    ],
    ...overrides,
  };
}

/** @param {number} id @param {Record<string, any>} [overrides] */
function agent(id, overrides = {}) {
  return {
    public_agent_id: String(id),
    display_agent_id: String(id),
    presentation_key: `k${id}`,
    team_id: id < 5 ? 1 : 2,
    class_id: CLASSES[id % 5],
    alive: true,
    life_state: "alive",
    current_health: 100,
    maximum_health: 100,
    max_health: 100,
    ultimate_cooldown_remaining: 0,
    statuses: [],
    spawn_shield_remaining: 0,
    position: [id, id],
    radius: 0.5,
    ...overrides,
  };
}

const identities = Array.from({ length: 10 }, (_, id) =>
  Object.freeze({
    publicAgentId: String(id),
    teamId: /** @type {1 | 2} */ (id < 5 ? 1 : 2),
    teamLocalSlot: id % 5,
    classId: CLASSES[id % 5],
  }),
);

/** @param {Record<string, any>} [overrides] */
function sources(overrides = {}) {
  return {
    product: /** @type {const} */ ("combat_debugger"),
    audience: /** @type {"researcher" | "agent_pov"} */ ("researcher"),
    sceneAgents: Array.from({ length: 10 }, (_, id) => agent(id)),
    researcherAgents: null,
    identities,
    mechanics: [],
    waves: [],
    visualEvents: [],
    teams: [],
    deaths: [],
    controlledKey: "k1",
    recipientPublicAgentId: null,
    returnedFallback: [],
    displayIdOf: (/** @type {string} */ id) => id,
    ...overrides,
  };
}

const ONE_MANUAL = liveControllers({
  manual_slots: [1],
  slot_labels: [
    "Recurrent MAPPO",
    "Manual",
    "Recurrent MAPPO",
    "Recurrent MAPPO",
    "Recurrent MAPPO",
    "tdm-beta",
    "tdm-beta",
    "tdm-beta",
    "tdm-beta",
    "tdm-beta",
  ],
});

/** @param {ReturnType<typeof sources>} input @param {Record<string, any>} [options] */
function model(input, controllers = ONE_MANUAL, options = {}) {
  const hudRecords = records({ sources: input, filters: {}, ...options });
  return liveScorebugModel(hudRecords, input, controllers);
}

/** @param {ReturnType<typeof model>} result @param {string} id */
function portrait(result, id) {
  const found = result.teams
    .flatMap((team) => team.portraits)
    .find((entry) => entry.publicAgentId === id);
  assert.ok(found, `portrait ${id}`);
  return found;
}

test("the scorebug view gives scores, bars, side, tick and First To", () => {
  const view = liveScorebugView(summary());
  assert.equal(view.available, true);
  assert.equal(view.teamASide, "right");
  assert.equal(view.tickText, "Tick 91 of 300");
  assert.equal(view.firstToText, "First to 20");
  const [a, b] = view.teams;
  assert.deepEqual(
    [a.key, a.scoreText, a.goalText, a.segments, a.lit, a.before],
    ["a", "2", "/ 20", 20, 2, null],
  );
  assert.deepEqual([b.key, b.scoreText, b.segments, b.lit], ["b", "4", 20, 4]);
  const left = liveScorebugView(
    summary({
      teams: [
        { teamId: 1, side: "left", scoreValue: 25 },
        { teamId: 2, side: "right", scoreValue: 0 },
      ],
    }),
  );
  assert.equal(left.teamASide, "left");
  assert.equal(left.teams[0].lit, 20);
  assert.equal(left.teams[1].lit, 0);
});

test("the Scorebug switch off gives a plain score with no goal or bar", () => {
  const view = liveScorebugView(summary(), { filters: { scorebug: false } });
  assert.equal(view.plain, true);
  assert.deepEqual(
    [view.teams[0].scoreText, view.teams[0].goalText, view.teams[0].segments],
    ["Score: 2/20", "", 0],
  );
});

test("a scoring tick keeps the old score and bar for the Fall beat", () => {
  const view = liveScorebugView(summary(), { beforeScores: [1, 4] });
  assert.deepEqual(view.teams[0].before, { scoreText: "1", lit: 1 });
  assert.equal(view.teams[1].before, null);
});

test("the wave line counts the soonest wave as countdown + 1", () => {
  const view = (/** @type {number[]} */ countdowns) =>
    liveScorebugView(summary(), {
      waves: countdowns.map((countdown, index) => ({
        team_index: index,
        countdown_steps: countdown,
      })),
    }).waveText;
  assert.equal(view([3, 3]), "Respawn Wave in 4 Ticks");
  assert.equal(view([4, 0]), "Respawn Wave in 1 Tick");
  assert.equal(view([]), "");
});

test("no summary gives an empty scorebug", () => {
  const view = liveScorebugView(null);
  assert.equal(view.available, false);
  assert.deepEqual(view.teams, []);
  assert.equal(`${view.tickText}${view.firstToText}${view.waveText}`, "");
});

test("controller lines, the Yours tag and malformed configurations", () => {
  assert.equal(controllerLine(1, ONE_MANUAL), "You + Recurrent MAPPO");
  assert.equal(controllerLine(2, ONE_MANUAL), "tdm-beta");
  const each = liveControllers({
    manual_slots: [1, 5],
    slot_labels: ONE_MANUAL.slotLabels.map((label, slot) =>
      slot === 5 ? "Manual" : label,
    ),
  });
  assert.equal(controllerLine(2, each), "You + tdm-beta");
  assert.equal(yoursTeam(each), 1);
  const watch = liveControllers({
    manual_slots: [],
    slot_labels: ONE_MANUAL.slotLabels.map((label, slot) =>
      slot === 1 ? "Recurrent MAPPO" : label,
    ),
  });
  assert.equal(controllerLine(1, watch), "Recurrent MAPPO");
  assert.equal(yoursTeam(watch), 1);
  const mixed = liveControllers({
    manual_slots: [7],
    slot_labels: [
      ...ONE_MANUAL.slotLabels.slice(0, 5),
      "tdm-beta",
      "random",
      ...ONE_MANUAL.slotLabels.slice(7),
    ],
  });
  assert.equal(controllerLine(2, mixed), "You + tdm-beta + random");
  assert.equal(yoursTeam(mixed), 2);
  assert.equal(liveControllers({ manual_slots: "1" }), NO_CONTROLLERS);
  assert.equal(controllerLine(1, NO_CONTROLLERS), "");
});

test("portraits read You for every Manual champion and focus the controlled one", () => {
  const each = liveControllers({
    manual_slots: [1, 5],
    slot_labels: ONE_MANUAL.slotLabels,
  });
  const result = model(sources(), each);
  assert.deepEqual(
    result.teams.map((team) => team.portraits.map((entry) => entry.label)),
    [
      ["ID 0", "You", "ID 2", "ID 3", "ID 4"],
      ["You", "ID 6", "ID 7", "ID 8", "ID 9"],
    ],
  );
  assert.deepEqual(result.lines, ["You + Recurrent MAPPO", "You + tdm-beta"]);
  assert.deepEqual(
    result.roster.teams.map((team) => controllerLineParts(team).join(" + ")),
    result.lines,
  );
  assert.deepEqual(result.roster.teams.map(collapsedControllerLine), [
    "You + 1 Policy",
    "You + 1 Policy",
  ]);
  assert.equal(portrait(result, "1").focus, true);
  assert.equal(portrait(result, "0").focus, false);
  assert.match(
    portrait(result, "1").ariaLabel,
    /^Warrior, Agent ID 1, Team A, yours$/u,
  );
});

test("life bars: Cinder at 30% or less, empty when fallen", () => {
  const input = sources({
    sceneAgents: [
      agent(0, { current_health: 30 }),
      agent(1, { current_health: 31 }),
      agent(2, { alive: false, life_state: "dead", current_health: 0 }),
    ],
  });
  const result = model(input);
  assert.deepEqual(
    [portrait(result, "0").life?.low, portrait(result, "1").life?.low],
    [true, false],
  );
  assert.equal(portrait(result, "2").state, "fallen");
  assert.equal(portrait(result, "2").life?.fraction, 0);
  assert.equal(portrait(result, "2").label, "ID 2");
});

test("a fall holds the living look and a return holds the fallen look", () => {
  const fall = model(
    sources({
      sceneAgents: [
        agent(1),
        agent(5, { alive: false, life_state: "dead", current_health: 0 }),
      ],
    }),
    ONE_MANUAL,
    {
      beat: {
        life: [{ publicAgentId: "5", presentationKey: "k5", before: 40 }],
        returns: [],
        schedule: { returns: [] },
      },
    },
  );
  assert.deepEqual(portrait(fall, "5").hold, {
    kind: "fall",
    beatKey: "k5",
    state: "alive",
    life: { fraction: 0.4, low: false, current: 40 },
  });
  const back = model(sources({ sceneAgents: [agent(1)] }), ONE_MANUAL, {
    beat: {
      life: [],
      returns: [{ publicAgentId: "1", seen: true }],
      schedule: { returns: [{ publicAgentId: "1", presentationKey: "k1" }] },
    },
  });
  assert.equal(portrait(back, "1").hold?.kind, "return");
  assert.equal(portrait(back, "1").hold?.state, "fallen");
});

test("Agent POV: an unseen enemy is identity only and the round trip is fresh", () => {
  const seen = Array.from({ length: 10 }, (_, id) => agent(id)).filter(
    (entry) => !["6", "8"].includes(entry.public_agent_id),
  );
  const oracle = model(sources());
  const pov = model(
    sources({ audience: "agent_pov", sceneAgents: seen, recipientPublicAgentId: "1" }),
  );
  const unseen = portrait(pov, "6");
  assert.equal(unseen.state, "unseen");
  assert.equal(unseen.label, "Unseen");
  assert.equal(unseen.life, null);
  assert.equal(unseen.hold, null);
  assert.equal(unseen.focus, false);
  assert.equal(
    unseen.refusal,
    "The Warrior (ID 6) is out of your team's sight. Agent POV hides where it is and how it fares.",
  );
  assert.equal(unseen.ariaLabel, "Warrior, Agent ID 6, unseen by your team");
  const watch = model(
    sources({
      audience: "agent_pov",
      sceneAgents: seen,
      controlledKey: null,
      recipientPublicAgentId: "5",
    }),
    liveControllers({ manual_slots: [], slot_labels: ONE_MANUAL.slotLabels }),
  );
  assert.match(portrait(watch, "8").refusal ?? "", /out of Team B's sight/u);
  const again = model(sources());
  const strip = (/** @type {ReturnType<typeof model>} */ result) =>
    JSON.stringify(result, (key, value) => (key === "record" ? undefined : value));
  assert.equal(strip(again), strip(oracle));
});
