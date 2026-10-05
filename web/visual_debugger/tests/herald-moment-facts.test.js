/**
 * @file Check the herald row's moment facts in the plan builder
 * (`src/choreography-plan.js`: heraldTickFacts and the herald_card row of
 * buildChoreographyPlan). Contracts: the host's per-view match_moments fact
 * is copied (fresh start, largest earlier lead, runs) and its rescues are
 * kept only for champions in this audience's scene; an Ultimate Combo needs
 * two damaging Ultimates on one target, and in Agent POV only blows whose
 * caster and target the team saw at the start of the tick (a charging
 * Warrior also at its end); Team Wiped needs a named fall of the team and
 * every configured-active member in the scene as fallen, so an unseen member
 * keeps it off in Agent POV; the public scoreboard (scores before and after,
 * First To, tick and draw tick) rides on the row, so a fresh match's first
 * fall names First Blood at the Fall beat; a tick with no named fall gets a
 * row only when a moment is named, and a Combo whose target lived waits for
 * the tick to settle.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { buildChoreographyPlan, heraldTickFacts } from "../src/choreography-plan.js";
import * as heralds from "../src/looks/heralds.js";
import { DEFAULT_VISUAL_FILTER_STATE } from "../src/visual-filters.js";

/** @type {Readonly<Record<string, number>>} */
const PHASE_RANK = Object.freeze({
  ability_activated: 20,
  agent_died: 90,
  lethal_damage_contribution: 90,
});

const surface = Object.freeze({
  worldToScreen: (/** @type {any} */ point) => ({
    x: Number(point.x ?? point[0]) * 10,
    y: 120 - Number(point.y ?? point[1]) * 10,
  }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 10,
  viewportBounds: Object.freeze({
    left: 0,
    top: 0,
    right: 800,
    bottom: 600,
    width: 800,
    height: 600,
  }),
  protectedRects: Object.freeze([]),
});

/** @param {string} id @param {number} classId @param {number} teamId @param {string} [lifeState] */
function body(id, classId, teamId, lifeState = "alive") {
  return {
    presentation_key: `key-${id}`,
    public_agent_id: id,
    class_id: classId,
    team_id: teamId,
    life_state: lifeState,
  };
}

/** @param {string} source @param {string} target */
function ultimate(source, target) {
  return {
    id: `u:${source}:${target}`,
    kind: "ability_activated",
    vocabulary: "event",
    payload: {
      ability_component: "ultimate",
      source_anchor: { presentation_key: `key-${source}`, public_agent_id: source },
      recipient_anchor: { presentation_key: `key-${target}`, public_agent_id: target },
    },
  };
}

const ROSTER = {
  current_endpoint: {
    identity_directory: {
      identities: [
        ["agent-slot-0", 1, 0],
        ["agent-slot-1", 1, 1],
        ["agent-slot-5", 2, 0],
        ["agent-slot-6", 2, 1],
      ].map(([id, team, slot]) => ({
        public_agent_id: id,
        team_id: team,
        team_local_slot: slot,
        class_id: 1,
        configured_active: true,
      })),
    },
  },
};

/** @param {ReadonlyArray<Record<string, any>>} bodies */
function scene(bodies) {
  return new Map(bodies.map((agent) => [agent.presentation_key, agent]));
}

/** @type {any} */
const fallOfSix = Object.freeze({
  publicAgentId: "agent-slot-6",
  victim: { teamId: 2 },
});

test("host facts are copied, and rescues need a body in this audience's scene", () => {
  const facts = heraldTickFacts({
    presentation: {
      ...ROSTER,
      match_moments: {
        fresh_start: true,
        largest_lead_before: 4,
        runs: [{ team_id: 2, takedowns: 5 }],
        rescues: [
          {
            public_agent_id: "agent-slot-6",
            team_id: 2,
            class_id: 4,
            by_ultimate: true,
          },
          {
            public_agent_id: "agent-slot-9",
            team_id: 2,
            class_id: 5,
            by_ultimate: false,
          },
        ],
      },
    },
    falls: [],
    rows: [],
    sceneByKey: scene([body("agent-slot-6", 4, 2)]),
    trajectories: null,
    agentVisual: false,
  });
  assert.equal(facts.freshStart, true);
  assert.equal(facts.largestLeadBefore, 4);
  assert.deepEqual(facts.runs, [{ teamId: 2, takedowns: 5 }]);
  assert.equal(facts.rescues.length, 1);
  assert.equal(facts.rescues[0].champion.publicAgentId, "agent-slot-6");
  assert.equal(facts.rescues[0].byUltimate, true);
  assert.equal(facts.match, null);
  const none = heraldTickFacts({
    presentation: {},
    falls: [],
    rows: [],
    sceneByKey: new Map(),
    trajectories: null,
    agentVisual: false,
  });
  assert.deepEqual(
    [none.freshStart, none.largestLeadBefore, none.runs, none.rescues],
    [false, 0, [], []],
  );
});

test("an Ultimate Combo needs two damaging Ultimates, and in Agent POV seen blows only", () => {
  const bodies = scene([
    body("agent-slot-0", 1, 1),
    body("agent-slot-1", 2, 1),
    body("agent-slot-5", 3, 2),
    body("agent-slot-6", 4, 2),
  ]);
  const rows = [
    ultimate("agent-slot-5", "agent-slot-0"),
    ultimate("agent-slot-6", "agent-slot-0"),
    // A Mage's Burst names no damage, so it never makes a combo.
    ultimate("agent-slot-0", "agent-slot-6"),
    ultimate("agent-slot-1", "agent-slot-6"),
  ];
  const oracle = heraldTickFacts({
    presentation: ROSTER,
    falls: [],
    rows,
    sceneByKey: bodies,
    trajectories: null,
    agentVisual: false,
  });
  assert.equal(oracle.combos.length, 1);
  assert.deepEqual(
    oracle.combos[0].casters.map((caster) => [caster.publicAgentId, caster.classId]),
    [
      ["agent-slot-5", 3],
      ["agent-slot-6", 4],
    ],
  );
  assert.equal(oracle.combos[0].target?.publicAgentId, "agent-slot-0");
  assert.equal(oracle.combos[0].fell, false);
  /** @param {Record<string, boolean>} seenAtStart @param {Record<string, boolean>} seenAtEnd */
  const trajectories = (seenAtStart, seenAtEnd) =>
    new Map(
      ["agent-slot-0", "agent-slot-1", "agent-slot-5", "agent-slot-6"].map((id) => [
        `key-${id}`,
        {
          transition_start: seenAtStart[id] === false ? null : { x: 0 },
          successor: seenAtEnd[id] === false ? null : { x: 0 },
        },
      ]),
    );
  const unseen = heraldTickFacts({
    presentation: ROSTER,
    falls: [],
    rows,
    sceneByKey: bodies,
    trajectories: trajectories({ "agent-slot-6": false }, {}),
    agentVisual: true,
  });
  assert.equal(unseen.combos.length, 0);
  // A Warrior that charged out of sight is not seen when the blows land, so
  // blows on it do not count; seen at the end of the tick they do.
  const chargeRows = [
    ultimate("agent-slot-5", "agent-slot-1"),
    ultimate("agent-slot-6", "agent-slot-1"),
    ultimate("agent-slot-1", "agent-slot-6"),
  ];
  /** @param {Record<string, boolean>} seenAtEnd */
  const onCharger = (seenAtEnd) =>
    heraldTickFacts({
      presentation: ROSTER,
      falls: [],
      rows: chargeRows,
      sceneByKey: bodies,
      trajectories: trajectories({}, seenAtEnd),
      agentVisual: true,
    }).combos.length;
  assert.equal(onCharger({ "agent-slot-1": false }), 0);
  assert.equal(onCharger({}), 1);
});

test("Team Wiped needs a named fall and every active member seen fallen", () => {
  const fallen = [
    body("agent-slot-5", 3, 2, "corpse"),
    body("agent-slot-6", 4, 2, "corpse"),
  ];
  const wiped = heraldTickFacts({
    presentation: ROSTER,
    falls: [fallOfSix],
    rows: [],
    sceneByKey: scene([body("agent-slot-0", 1, 1), ...fallen]),
    trajectories: null,
    agentVisual: false,
  });
  assert.deepEqual(
    wiped.wiped.map((team) => [team.teamId, team.members.length]),
    [[2, 2]],
  );
  // Agent POV without the other member in sight: not wiped.
  const pov = heraldTickFacts({
    presentation: ROSTER,
    falls: [fallOfSix],
    rows: [],
    sceneByKey: scene([body("agent-slot-0", 1, 1), fallen[1]]),
    trajectories: null,
    agentVisual: true,
  });
  assert.deepEqual(pov.wiped, []);
  // No named fall of that team this tick: not a new Team Wiped.
  const quiet = heraldTickFacts({
    presentation: ROSTER,
    falls: [],
    rows: [],
    sceneByKey: scene(fallen),
    trajectories: null,
    agentVisual: false,
  });
  assert.deepEqual(quiet.wiped, []);
});

/** @returns {Promise<Record<string, any>>} */
async function oracleFrame() {
  const raw = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  return structuredClone(raw.presentations.replay_oracle);
}

/** @param {Record<string, any>} inventory @param {string} publicId
 * @param {"transition_start" | "successor"} phase */
function anchor(inventory, publicId, phase) {
  const row = inventory.agent_phase_trajectories.find(
    (/** @type {Record<string, any>} */ candidate) =>
      candidate.agent_public_agent_id === publicId,
  );
  assert.ok(row, `missing trajectory ${publicId}`);
  return structuredClone(row[phase]);
}

/** @param {Record<string, any>} inventory @param {Record<string, any>[]} events */
function installEvents(inventory, events) {
  /** @type {Record<string, any>[]} */
  const ordered = events
    .map((event) => ({ ...event, phase_rank: PHASE_RANK[event.event_kind] }))
    .sort((left, right) => left.phase_rank - right.phase_rank)
    .map((event, ordinal) => ({
      ...event,
      event_id: `${inventory.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
      ordinal,
    }));
  inventory.events = ordered;
  inventory.event_count = ordered.length;
  inventory.ordered_event_ids = ordered.map(({ event_id }) => event_id);
  inventory.ordered_event_kinds = ordered.map(({ event_kind }) => event_kind);
}

/** @param {Record<string, any>} raw @param {[number, number]} before @param {[number, number]} after */
function asTeamDeathmatch(raw, before, after) {
  Object.assign(raw.match_summary, {
    task_mode: 1,
    score_threshold: 20,
    outcome: "in_progress",
    episode_limit: 300,
    previous_scores: before,
    scores: after,
  });
  raw.match_moments = {
    fresh_start: true,
    largest_lead_before: 0,
    runs: [],
    rescues: [],
  };
}

test("a fresh match's first fall names First Blood at the Fall beat", async () => {
  const raw = await oracleFrame();
  asTeamDeathmatch(raw, [0, 0], [1, 0]);
  const latest = raw.latest_events;
  installEvents(latest, [
    {
      event_kind: "agent_died",
      recipient_anchor: anchor(latest, "agent-slot-6", "successor"),
      home_fall: false,
    },
    {
      event_kind: "lethal_damage_contribution",
      source_anchor: anchor(latest, "agent-slot-1", "successor"),
      recipient_anchor: anchor(latest, "agent-slot-6", "successor"),
      attributed_death_damage: 40,
    },
  ]);
  const plan = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(raw),
    surface,
    DEFAULT_VISUAL_FILTER_STATE,
  );
  assert.ok(plan);
  const herald = plan.derivedEvents.find((event) => event.kind === "herald_card");
  assert.ok(herald);
  assert.equal(herald.freshStart, true);
  assert.deepEqual(herald.match.before, [0, 0]);
  assert.deepEqual(herald.match.after, [1, 0]);
  assert.equal(herald.match.pointsToWin, 20);
  assert.equal(herald.phaseStart, plan.beats.falls[0].at);
  const [card] = heralds.records({
    event: herald,
    filters: DEFAULT_VISUAL_FILTER_STATE,
  });
  assert.equal(card.title, "First Blood");
  assert.equal(card.eyebrow, "Team A +1");
});

test("a tick with no fall gets a row only for a moment, timed by what it names", async () => {
  const raw = await oracleFrame();
  asTeamDeathmatch(raw, [0, 0], [0, 0]);
  const latest = raw.latest_events;
  installEvents(latest, [
    {
      event_kind: "ability_activated",
      ability_component: "ultimate",
      source_anchor: anchor(latest, "agent-slot-5", "transition_start"),
      recipient_anchor: anchor(latest, "agent-slot-0", "transition_start"),
    },
    {
      event_kind: "ability_activated",
      ability_component: "ultimate",
      source_anchor: anchor(latest, "agent-slot-6", "transition_start"),
      recipient_anchor: anchor(latest, "agent-slot-0", "transition_start"),
    },
  ]);
  const plan = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(raw),
    surface,
    DEFAULT_VISUAL_FILTER_STATE,
  );
  assert.ok(plan);
  const herald = plan.derivedEvents.find((event) => event.kind === "herald_card");
  assert.ok(herald, "a combo makes a row on a tick without a fall");
  assert.equal(herald.eventType, "match_moment");
  assert.equal(herald.count, 0);
  assert.equal(herald.phaseStart, plan.beats.explanationEnd);
  const [card] = heralds.records({ event: herald });
  assert.equal(card.kind, "herald-plate");
  assert.equal(card.title, "Ultimate Combo");
  assert.equal(card.eyebrow, "Freezing Trap + Crippling Poison");
  // The same tick with one Ultimate names nothing, so it has no row.
  const single = await oracleFrame();
  asTeamDeathmatch(single, [0, 0], [0, 0]);
  installEvents(single.latest_events, [latest.events[0]]);
  const quiet = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(single),
    surface,
    DEFAULT_VISUAL_FILTER_STATE,
  );
  assert.ok(quiet);
  assert.equal(
    quiet.derivedEvents.some((event) => event.kind === "herald_card"),
    false,
  );
});
