/**
 * @file Check the Agent POV boundary of the plan builder and beat clock. Over every
 * Agent POV frame of the browser fixture, plus a frame with a death the team saw,
 * changing the global researcher_space rows of champions the team cannot see
 * (their life and Ultimate cooldown) and the match_summary rows about them (a death
 * with its killers, a return) changes nothing: every plan record (events, derived
 * rows, bodies, camera input, bounds, fingerprint and keys), every beat time
 * and every beatView answer stays identical, with the default switches and with
 * every switch on. Oracle View -> Agent POV -> Oracle View then gives exactly the
 * painter installation and overrides of a fresh render, paused and in play.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { CombatChoreographer, ConsumedTransitionLedger } from "../src/choreography.js";
import { beatView, buildChoreographyPlan } from "../src/choreography-plan.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  enableAllVisualFilters,
} from "../src/visual-filters.js";

const FILTER_SETS = [
  DEFAULT_VISUAL_FILTER_STATE,
  enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE),
];

const surface = Object.freeze({
  worldToScreen: (/** @type {any} */ point) => ({
    x: 24 + Number(point.x ?? point[0]) * 40,
    y: 664 - Number(point.y ?? point[1]) * 40,
  }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 40,
  viewportBounds: Object.freeze({
    left: 24,
    top: 24,
    right: 1064,
    bottom: 664,
    width: 1040,
    height: 640,
  }),
  protectedRects: Object.freeze([]),
});

/** @returns {Promise<Record<string, any>>} */
async function fixture() {
  return JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
}

/** @param {Record<string, any>} data */
function agentPovRaws(data) {
  /** @type {Array<[string, Record<string, any>]>} */
  const raws = [];
  for (const [name, value] of Object.entries(data.presentations)) {
    raws.push([`presentations.${name}`, value]);
  }
  for (const group of ["pairs", "continuity_pairs"]) {
    for (const [name, value] of Object.entries(data[group])) {
      raws.push([`${group}.${name}`, /** @type {any} */ (value).presentation]);
    }
  }
  for (const group of ["state_cases", "red_zone_cases"]) {
    for (const [name, value] of Object.entries(data[group])) {
      raws.push([`${group}.${name}`, /** @type {Record<string, any>} */ (value)]);
    }
  }
  return raws.filter(
    ([, raw]) =>
      typeof raw.presentation_kind === "string" &&
      raw.presentation_kind.endsWith("agent_pov") &&
      raw.researcher_space,
  );
}

/** @param {Record<string, any>} raw */
function withSeenDeath(raw) {
  const copy = structuredClone(raw);
  const visual = copy.visual_events;
  const seenOther = visual.agent_phase_trajectories.find(
    (/** @type {Record<string, any>} */ row) =>
      row.successor &&
      row.transition_start &&
      row.agent_presentation_key !== copy.authority?.recipient_presentation_key,
  );
  assert.ok(seenOther, "the POV fixture sees one other champion at both ends");
  /** @type {Readonly<Record<string, number>>} */
  const rank = {
    ability_activated: 20,
    recipient_health_resolution: 40,
    agent_died: 90,
    status_applied: 100,
  };
  const events = [
    ...visual.events,
    {
      event_kind: "agent_died",
      recipient_anchor: structuredClone(seenOther.successor),
    },
  ]
    .map((event) => ({
      ...event,
      phase_rank: rank[event.event_kind] ?? event.phase_rank,
    }))
    .sort((left, right) => left.phase_rank - right.phase_rank)
    .map((event, ordinal) => ({
      ...event,
      ordinal,
      event_id: `${visual.incoming_recipient_transition_id}:visual-event:${String(ordinal).padStart(4, "0")}`,
    }));
  visual.events = events;
  visual.event_count = events.length;
  visual.ordered_event_ids = events.map(({ event_id }) => event_id);
  visual.ordered_event_kinds = events.map(({ event_kind }) => event_kind);
  return copy;
}

/** @param {Record<string, any>} raw */
function hiddenChampions(raw) {
  const seen = new Set([
    ...raw.current_endpoint.parts.scene.agents.map(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id,
    ),
    ...(raw.local_oracle_corpse_overlay?.corpse_observations ?? []).map(
      (/** @type {Record<string, any>} */ corpse) => corpse.public_agent_id,
    ),
  ]);
  return raw.researcher_space.roster_agents.filter(
    (/** @type {Record<string, any>} */ agent) => !seen.has(agent.public_agent_id),
  );
}

/**
 * @param {Record<string, any>} raw
 * @returns {Array<[string, Record<string, any>]>}
 */
function hiddenVariants(raw) {
  const hidden = hiddenChampions(raw);
  if (hidden.length === 0) return [];
  /** @type {Array<[string, Record<string, any>]>} */
  const variants = [];
  const life = structuredClone(raw);
  for (const agent of hiddenChampions(life)) {
    if (agent.life_state === "alive" && agent.current_health > 1) {
      agent.current_health = Math.max(1, Math.round(agent.current_health / 3));
    }
  }
  variants.push(["hidden life", life]);
  const cooldown = structuredClone(raw);
  for (const agent of hiddenChampions(cooldown)) {
    agent.ultimate_cooldown_remaining = 2;
  }
  variants.push(["hidden Ultimate cooldown", cooldown]);
  if (raw.match_summary) {
    const victim = hidden[0];
    const roster = raw.researcher_space.roster_agents;
    const killers = roster.filter(
      (/** @type {Record<string, any>} */ agent) => agent.team_id !== victim.team_id,
    );
    const death = structuredClone(raw);
    death.match_summary.deaths = [
      {
        public_agent_id: victim.public_agent_id,
        team_id: victim.team_id,
        class_id: victim.class_id,
        killing_team_id: 3 - victim.team_id,
        contributors: killers.map((/** @type {Record<string, any>} */ agent) => ({
          public_agent_id: agent.public_agent_id,
          team_id: agent.team_id,
          class_id: agent.class_id,
        })),
      },
    ];
    variants.push(["a hidden death with killers", death]);
    const returned = structuredClone(raw);
    returned.match_summary.returned_agent_ids = [victim.public_agent_id];
    variants.push(["a hidden return", returned]);
  }
  return variants;
}

/** @param {unknown} frame */
function boundaryView(frame) {
  return FILTER_SETS.map((filters) => ({
    plan: buildChoreographyPlan(frame, surface, filters),
    beats: [0, 1099, 1100, 1160, 2600, "end"].map((stage) =>
      beatView(frame, /** @type {any} */ (stage), { visualFilters: filters }),
    ),
  }));
}

test("hidden champions' researcher and match rows change no Agent POV plan, beat or camera input", async () => {
  const data = await fixture();
  const raws = agentPovRaws(data);
  const seenDeathBase = raws.find(
    ([name]) => name === "presentations.replay_no_shared_obs_agent_pov",
  );
  assert.ok(seenDeathBase);
  raws.push(["seen death", withSeenDeath(seenDeathBase[1])]);
  let compared = 0;
  let plannedFrames = 0;
  /** @type {Set<string>} */
  const variantKinds = new Set();
  for (const [name, raw] of raws) {
    const original = await normalizeAuthorizedPresentationFrameV1(structuredClone(raw));
    const expected = boundaryView(original);
    if (expected.some(({ plan }) => plan !== null)) plannedFrames += 1;
    for (const [label, variant] of hiddenVariants(raw)) {
      let frame;
      try {
        frame = await normalizeAuthorizedPresentationFrameV1(variant);
      } catch {
        continue;
      }
      assert.deepEqual(boundaryView(frame), expected, `${name}: ${label}`);
      compared += 1;
      variantKinds.add(label);
    }
  }
  assert.ok(compared >= 10, `compared ${compared} hidden-row variants`);
  assert.ok(plannedFrames >= 5, `${plannedFrames} Agent POV frames built a plan`);
  assert.deepEqual([...variantKinds].sort(), [
    "a hidden death with killers",
    "a hidden return",
    "hidden Ultimate cooldown",
    "hidden life",
  ]);
});

test("the seen death names its victim and only killers the team saw", async () => {
  const raw = withSeenDeath(
    (await fixture()).presentations.replay_no_shared_obs_agent_pov,
  );
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const plan = buildChoreographyPlan(
    frame,
    surface,
    enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE),
  );
  assert.ok(plan);
  const seenIds = new Set(
    raw.current_endpoint.parts.scene.agents.map(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id,
    ),
  );
  const startSeen = new Set(
    raw.visual_events.agent_phase_trajectories
      .filter((/** @type {Record<string, any>} */ row) => row.transition_start)
      .map((/** @type {Record<string, any>} */ row) => row.agent_public_agent_id),
  );
  for (const row of plan.derivedEvents) {
    if (row.kind !== "fallen_feed_row") continue;
    assert.ok(seenIds.has(row.victim.publicAgentId));
    for (const killer of row.killers) {
      assert.ok(
        startSeen.has(killer.publicAgentId),
        "a killer is named only when seen",
      );
    }
  }
  const announcements = plan.events.filter(
    (event) => event.cueSemantic === "death_announcement",
  );
  assert.equal(
    announcements.flatMap((event) => event.members).length,
    plan.beats.falls.length,
    "Agent POV cards hold only the falls the team saw",
  );
  const herald = plan.derivedEvents.find((row) => row.kind === "herald_card");
  assert.equal(herald?.count, plan.beats.falls.length);
  assert.ok(
    plan.camera.bodies.every((body) => seenIds.has(body.publicAgentId)),
    "the camera frames only seen bodies",
  );
});

class RecordingPainter {
  constructor() {
    /** @type {Array<[string, ...any[]]>} */
    this.calls = [];
  }

  /** @param {Record<string, any>} plan @param {unknown} _surface @param {Record<string, any>} options */
  install(plan, _surface, options) {
    this.calls.push(["install", plan, options]);
    return {
      root: {},
      nodeCount: 3,
      persistentNodeCount: 0,
      animationSpecs: options.settled
        ? []
        : [
            {
              element: {},
              keyframes: [],
              options: {},
              id: `mbg:${plan.epochKey}:event:group`,
            },
          ],
    };
  }

  /** @param {unknown} _installation @param {string} reason */
  clear(_installation, reason) {
    this.calls.push(["clear", reason]);
  }

  settle() {
    this.calls.push(["settle"]);
  }

  reproject() {
    this.calls.push(["reproject"]);
  }
}

class RecordingFactory {
  constructor() {
    /** @type {Array<Record<string, any>>} */
    this.created = [];
  }

  /** @param {Record<string, any>} spec */
  create(spec) {
    const animation = {
      spec,
      id: spec.id,
      currentTime: 0,
      playbackRate: 1,
      finished: new Promise(() => {}),
      pause() {},
      play() {},
      finish() {},
      cancel() {},
    };
    this.created.push(animation);
    return animation;
  }

  /** @param {unknown} element @param {number} duration @param {string} id */
  createClock(element, duration, id) {
    return this.create({ element, keyframes: [], options: { duration }, id });
  }
}

/** @param {ConsumedTransitionLedger} ledger */
function harness(ledger) {
  const painter = new RecordingPainter();
  const factory = new RecordingFactory();
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (painter),
    animationFactory: /** @type {any} */ (factory),
    ledger,
    planBuilder: (frame) =>
      buildChoreographyPlan(frame, surface, DEFAULT_VISUAL_FILTER_STATE),
  });
  return { controller, factory, painter };
}

test("Oracle View -> Agent POV -> Oracle View gives exactly a fresh render, paused and in play", async () => {
  const data = await fixture();
  const oracle = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(data.presentations.replay_oracle),
  );
  const pov = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(data.presentations.replay_no_shared_obs_agent_pov),
  );
  const domSurface = /** @type {any} */ ({
    ...surface,
    layer: {},
    ownerDocument: {},
    viewportKey: "boundary",
  });
  for (const policy of /** @type {const} */ (["replay_static", "live_once"])) {
    const ledger = new ConsumedTransitionLedger();
    const cycled = harness(ledger);
    cycled.controller.presentFrame(oracle, domSurface, { renderPolicy: policy });
    cycled.controller.presentFrame(pov, domSurface, { renderPolicy: policy });
    const before = cycled.factory.created.length;
    cycled.controller.presentFrame(oracle, domSurface, { renderPolicy: policy });
    const fresh = harness(ledger);
    fresh.controller.presentFrame(oracle, domSurface, { renderPolicy: policy });

    const lastInstall = cycled.painter.calls
      .filter(([kind]) => kind === "install")
      .at(-1);
    const freshInstall = fresh.painter.calls
      .filter(([kind]) => kind === "install")
      .at(-1);
    assert.deepEqual(lastInstall, freshInstall, policy);
    assert.deepEqual(
      cycled.factory.created.slice(before).map(({ spec }) => [spec.id, spec.options]),
      fresh.factory.created.map(({ spec }) => [spec.id, spec.options]),
      policy,
    );
    assert.equal(
      cycled.painter.calls.filter(([kind]) => kind === "clear").length,
      2,
      `${policy}: each view change clears before installing`,
    );
    const { epochKey, authorizationKey, fingerprint, paintKey, renderPolicy, paused } =
      cycled.controller.snapshot();
    const freshSnapshot = fresh.controller.snapshot();
    assert.deepEqual(
      { epochKey, authorizationKey, fingerprint, paintKey, renderPolicy, paused },
      {
        epochKey: freshSnapshot.epochKey,
        authorizationKey: freshSnapshot.authorizationKey,
        fingerprint: freshSnapshot.fingerprint,
        paintKey: freshSnapshot.paintKey,
        renderPolicy: freshSnapshot.renderPolicy,
        paused: freshSnapshot.paused,
      },
    );
    cycled.controller.dispose();
    fresh.controller.dispose();
  }
});
