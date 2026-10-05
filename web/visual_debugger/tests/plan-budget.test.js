/**
 * @file Check the plan's resource bounds. Every frame of the browser fixture
 * (presentations, pairs, continuity pairs, state cases, Red Zone cases and the
 * legacy compatibility case), in its own audience (Oracle View or Agent POV), with
 * the default switches and with every switch on, builds a plan whose node,
 * animation, kept-node and override bounds come from the look modules' BUDGET
 * exports and stay well under PLAN_RESOURCE_CAPS. Cue footprints come from the look
 * modules' FOOTPRINT exports (a return adds its waystone Call's CALL_BUDGET to
 * its Homecoming). The recorded worst case of real recordings (the three
 * sample replays and a full 5v5 test-map match, Oracle View and Agent POV of a
 * champion on each team, measured once with the plan builder and the real painter
 * in Chromium) stays under the caps with margin. A tick over budget drops optional derived families first and never
 * its incoming events, so a busy tick is never blanked. A status-change row that no
 * look draws (every one but the Ward Lift) costs nothing, so a busy tick with many
 * status changes keeps its herald card. The caps are 1024: the four busy ticks of
 * the map 47 tournament replay (episode 1002, Oracle View: ticks 105, 110, 225 and
 * 245, whose rows were recorded from the real plan on 4 Oct 2026) ask for 535 to
 * 637 nodes, more than the old 512, and keep their Herald Card and Fallen rows.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  applyPlanBudget,
  beatSchedule,
  buildChoreographyPlan,
  CHOREOGRAPHY_PAINT_FOOTPRINTS,
  PLAN_RESOURCE_CAPS,
  planEventBudget,
  planResourceDemand,
} from "../src/choreography-plan.js";
import * as basics from "../src/looks/basics.js";
import * as focusTally from "../src/looks/focus-tally.js";
import * as fog from "../src/looks/fog.js";
import * as healthText from "../src/looks/health-text.js";
import * as heralds from "../src/looks/heralds.js";
import * as redZone from "../src/looks/red-zone.js";
import * as spawns from "../src/looks/spawns.js";
import * as requiems from "../src/looks/requiems.js";
import * as statuses from "../src/looks/statuses.js";
import { ACTIVATION_FOOTPRINTS } from "../src/looks/strike-routes.js";
import * as ultimates from "../src/looks/ultimates.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  enableAllVisualFilters,
} from "../src/visual-filters.js";

// Recorded on 1 Oct 2026 from the three sample replays and one full 5v5 test-map
// match, in Oracle View and in Agent POV for a champion on each team: 969 frames,
// 1,914 plans. Plan demand is before the caps; painter counts are the nodes and
// animations Chromium really made with the bounds lifted.
const RECORDED_WORST_CASE = Object.freeze({
  planDemandNodes: 873,
  planAnimations: 119,
  planPersistentNodes: 35,
  planOverrideAnimations: 36,
  painterNodes: 327,
  painterAnimations: 67,
  painterPersistentNodes: 24,
  painterNodesPerDrawnRow: 13.5,
});

// Map 47 tournament replay (episode 1002), Oracle View: the rows the plan budget
// saw on the four ticks that lost their Herald Card under the old 512 cap, recorded
// from the real plan on 4 Oct 2026. Each row is [kind, component, cueSemantic,
// spatial, persistent, count]; herald is the card's own allowance (rowBudget).
const BUSY_TICKS = Object.freeze({
  105: {
    nodes: 620,
    herald: {
      nodesPerRecord: 17,
      animationsPerRecord: 2,
      persistentNodesPerRecord: 17,
    },
    events: [
      ["activation", "basic", null, true, false, 9],
      ["activation", "ultimate", null, true, false, 1],
      ["feed_only", null, null, false, false, 33],
      ["net_health", null, null, true, false, 8],
      ["semantic_pulse", null, "cooldown_started", false, false, 1],
      ["semantic_pulse", null, "agent_died", true, true, 1],
      ["status_lifecycle", null, null, true, false, 5],
      ["status_lifecycle", null, null, false, false, 1],
      ["semantic_pulse", null, "respawn_wave_occurred", false, true, 2],
      ["semantic_pulse", null, "death_announcement", false, false, 1],
    ],
    derived: [
      ["fallen_feed_row", null, null, true, false, 1],
      ["focus_tally", null, null, false, true, 2],
    ],
  },
  110: {
    nodes: 637,
    herald: {
      nodesPerRecord: 22,
      animationsPerRecord: 2,
      persistentNodesPerRecord: 22,
    },
    events: [
      ["activation", "basic", null, true, false, 8],
      ["feed_only", null, null, false, false, 29],
      ["net_health", null, null, true, false, 5],
      ["semantic_pulse", null, "cooldown_ready", true, false, 1],
      ["semantic_pulse", null, "agent_died", true, true, 2],
      ["status_lifecycle", null, null, false, false, 1],
      ["status_lifecycle", null, null, true, false, 2],
      ["semantic_pulse", null, "respawn_wave_occurred", false, true, 2],
      ["semantic_pulse", null, "agent_respawned", true, true, 2],
      ["semantic_pulse", null, "death_announcement", false, false, 2],
    ],
    derived: [
      ["fallen_feed_row", null, null, true, false, 2],
      ["focus_tally", null, null, false, true, 2],
    ],
  },
  225: {
    nodes: 542,
    herald: {
      nodesPerRecord: 17,
      animationsPerRecord: 2,
      persistentNodesPerRecord: 17,
    },
    events: [
      ["activation", "ultimate", null, true, false, 2],
      ["activation", "basic", null, true, false, 5],
      ["feed_only", null, null, false, false, 23],
      ["net_health", null, null, true, false, 4],
      ["status_lifecycle", null, null, true, false, 4],
      ["semantic_pulse", null, "cooldown_ready", true, false, 2],
      ["semantic_pulse", null, "cooldown_started", false, false, 2],
      ["semantic_pulse", null, "agent_died", true, true, 1],
      ["status_lifecycle", null, null, false, false, 2],
      ["semantic_pulse", null, "respawn_wave_occurred", false, true, 2],
      ["semantic_pulse", null, "agent_respawned", true, true, 2],
      ["semantic_pulse", null, "death_announcement", false, false, 1],
    ],
    derived: [
      ["fallen_feed_row", null, null, true, false, 1],
      ["focus_tally", null, null, false, true, 2],
    ],
  },
  245: {
    nodes: 535,
    herald: {
      nodesPerRecord: 22,
      animationsPerRecord: 2,
      persistentNodesPerRecord: 22,
    },
    events: [
      ["activation", "basic", null, true, false, 7],
      ["feed_only", null, null, false, false, 29],
      ["net_health", null, null, true, false, 6],
      ["semantic_pulse", null, "agent_died", true, true, 2],
      ["status_lifecycle", null, null, true, false, 4],
      ["semantic_pulse", null, "respawn_wave_occurred", false, true, 2],
      ["semantic_pulse", null, "death_announcement", false, false, 2],
    ],
    derived: [["fallen_feed_row", null, null, true, false, 2]],
  },
});

/**
 * @param {string} tick
 * @param {ReadonlyArray<ReadonlyArray<any>>} specs
 */
function busyRows(tick, specs) {
  return specs.flatMap(([kind, component, cueSemantic, spatial, persistent, count]) =>
    Array.from({ length: count }, (_, index) =>
      Object.freeze({
        eventId: `t${tick}-${kind}-${cueSemantic ?? component ?? "row"}-${index}`,
        kind,
        ...(component === null ? {} : { component }),
        ...(cueSemantic === null ? {} : { cueSemantic }),
        spatial,
        persistent,
      }),
    ),
  );
}

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
function fixtureFrames(data) {
  /** @type {Array<[string, Record<string, any>]>} */
  const frames = [];
  for (const [name, value] of Object.entries(data.presentations)) {
    frames.push([`presentations.${name}`, value]);
  }
  for (const group of ["pairs", "continuity_pairs"]) {
    for (const [name, value] of Object.entries(data[group])) {
      frames.push([`${group}.${name}`, /** @type {any} */ (value).presentation]);
    }
  }
  for (const group of ["state_cases", "red_zone_cases", "compatibility_cases"]) {
    for (const [name, value] of Object.entries(data[group])) {
      frames.push([`${group}.${name}`, /** @type {Record<string, any>} */ (value)]);
    }
  }
  return frames;
}

test("every fixture frame builds a plan whose bounds stay well under the caps", async () => {
  const data = await fixture();
  let plans = 0;
  let agentPovPlans = 0;
  for (const [name, raw] of fixtureFrames(data)) {
    const frame = await normalizeAuthorizedPresentationFrameV1(structuredClone(raw));
    for (const filters of [
      DEFAULT_VISUAL_FILTER_STATE,
      enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE),
    ]) {
      const plan = buildChoreographyPlan(frame, surface, filters);
      if (plan === null) continue;
      plans += 1;
      if (plan.audience === "agent_pov") agentPovPlans += 1;
      const demand = planResourceDemand(
        plan.events,
        plan.derivedEvents,
        plan.bodies,
        plan.beats,
      );
      for (const key of /** @type {const} */ ([
        "nodes",
        "animations",
        "persistentNodes",
        "overrideAnimations",
      ])) {
        assert.equal(
          plan.bounds[key],
          Math.min(demand[key], PLAN_RESOURCE_CAPS[key]),
          name,
        );
        assert.ok(plan.bounds[key] <= PLAN_RESOURCE_CAPS[key] / 2, `${name} ${key}`);
      }
      const drawn = [...plan.events, ...plan.derivedEvents].filter(
        (row) => row.spatial,
      );
      assert.equal(
        demand.nodes,
        3 + drawn.reduce((sum, row) => sum + planEventBudget(row).nodesPerRecord, 0),
        name,
      );
      assert.ok(plan.bodies.length <= 10, name);
    }
  }
  assert.ok(plans >= 20, `built ${plans} fixture plans`);
  assert.ok(agentPovPlans >= 8, `built ${agentPovPlans} Agent POV plans`);
});

test("budgets and footprints come from the look module that draws each part", () => {
  /** @type {Array<[Record<string, any>, unknown]>} */
  const cases = [
    [{ kind: "activation", component: "basic" }, basics.BUDGET],
    [{ kind: "activation", component: "ultimate" }, ultimates.BUDGET],
    [{ kind: "net_health" }, healthText.BUDGET],
    [{ kind: "regeneration" }, healthText.BUDGET],
    [
      { kind: "status_lifecycle", cueSemantic: "spawn_shield_expired" },
      statuses.BUDGET,
    ],
    [{ kind: "semantic_pulse", cueSemantic: "cooldown_ready" }, statuses.BUDGET],
    [{ kind: "semantic_pulse", cueSemantic: "agent_died" }, requiems.BUDGET],
    [{ kind: "semantic_pulse", cueSemantic: "death_announcement" }, requiems.BUDGET],
    [{ kind: "fallen_feed_row" }, requiems.BUDGET],
    [{ kind: "home_fall" }, redZone.BUDGET],
    [{ kind: "herald_card" }, heralds.BUDGET],
    [{ kind: "last_seen" }, fog.BUDGET],
    [{ kind: "focus_tally" }, focusTally.BUDGET],
  ];
  for (const [row, budget] of cases) {
    assert.equal(planEventBudget(row), budget, JSON.stringify(row));
  }
  // A return draws its Homecoming (requiems.js) and its waystone's Call
  // (spawns.js), so its allowance is the sum of both.
  assert.deepEqual(
    planEventBudget({ kind: "semantic_pulse", cueSemantic: "agent_respawned" }),
    {
      nodesPerRecord:
        requiems.BUDGET.nodesPerRecord + spawns.CALL_BUDGET.nodesPerRecord,
      animationsPerRecord:
        requiems.BUDGET.animationsPerRecord + spawns.CALL_BUDGET.animationsPerRecord,
      persistentNodesPerRecord:
        requiems.BUDGET.persistentNodesPerRecord + spawns.CALL_BUDGET.nodesPerRecord,
    },
  );
  // No look draws any other status change (Rune Zones shows it on the body), so
  // it costs nothing.
  for (const lifecycle of ["applied", "expired", "trap_broken", "cleared_by_death"]) {
    assert.deepEqual(planEventBudget({ kind: "status_lifecycle", lifecycle }), {
      nodesPerRecord: 0,
      animationsPerRecord: 0,
      persistentNodesPerRecord: 0,
    });
  }
  const fallback = planEventBudget({ kind: "feed_only" });
  assert.deepEqual(fallback, {
    nodesPerRecord: 30,
    animationsPerRecord: 4,
    persistentNodesPerRecord: 7,
  });
  assert.deepEqual(CHOREOGRAPHY_PAINT_FOOTPRINTS.activation, {
    ...basics.FOOTPRINT.activation,
    ...ultimates.FOOTPRINT.activation,
  });
  assert.equal(
    CHOREOGRAPHY_PAINT_FOOTPRINTS.chargeOwnership,
    ultimates.FOOTPRINT.chargeOwnership,
  );
  // The team respawn wave is page chrome: neither table keeps an arena box.
  assert.equal(Object.hasOwn(CHOREOGRAPHY_PAINT_FOOTPRINTS, "respawnWave"), false);
  assert.equal(Object.hasOwn(requiems.FOOTPRINT, "respawnWave"), false);
  assert.equal(CHOREOGRAPHY_PAINT_FOOTPRINTS.route, ACTIVATION_FOOTPRINTS.route);
});

test("the recorded worst case of real recordings stays under the caps with margin", () => {
  const worst = RECORDED_WORST_CASE;
  assert.ok(worst.painterNodes <= PLAN_RESOURCE_CAPS.nodes * 0.7);
  assert.ok(worst.painterPersistentNodes <= PLAN_RESOURCE_CAPS.persistentNodes * 0.1);
  assert.ok(worst.planAnimations <= PLAN_RESOURCE_CAPS.animations / 4);
  assert.ok(worst.planPersistentNodes <= PLAN_RESOURCE_CAPS.persistentNodes / 4);
  assert.ok(worst.planOverrideAnimations <= PLAN_RESOURCE_CAPS.overrideAnimations / 2);
  // Painter animations, its two clocks and every override still fit the browser limit.
  assert.ok(
    worst.painterAnimations + 2 + worst.planOverrideAnimations <=
      PLAN_RESOURCE_CAPS.animations / 4,
  );
  // The per-row allowance is a true upper bound with room to spare.
  assert.ok(
    planEventBudget({ kind: "activation" }).nodesPerRecord >=
      2 * worst.painterNodesPerDrawnRow,
  );
  // The busiest recorded tick's plan demand fits under the cap, so nothing on it is
  // dropped; its real nodes are far below the cap.
  assert.ok(worst.planDemandNodes <= PLAN_RESOURCE_CAPS.nodes);
  assert.ok(worst.painterNodes < PLAN_RESOURCE_CAPS.nodes / 2);
});

test("a tick over budget drops optional derived families first and never its incoming events", () => {
  const perEvent = planEventBudget({
    kind: "activation",
    component: "basic",
  }).nodesPerRecord;
  const perFallen = planEventBudget({ kind: "fallen_feed_row" }).nodesPerRecord;
  assert.ok(perEvent > 0 && perFallen > 0);
  const eventCount = Math.floor((PLAN_RESOURCE_CAPS.nodes - 3) / perEvent);
  const fallenCount =
    Math.ceil((PLAN_RESOURCE_CAPS.nodes - (3 + eventCount * perEvent)) / perFallen) + 1;
  const events = Array.from({ length: eventCount }, (_, index) =>
    Object.freeze({
      eventId: `event-${index}`,
      kind: "activation",
      component: "basic",
      spatial: true,
    }),
  );
  const derived = [
    ...Array.from({ length: fallenCount }, (_, index) =>
      Object.freeze({
        eventId: `fallen-${index}`,
        kind: "fallen_feed_row",
        spatial: true,
      }),
    ),
  ];
  const schedule = beatSchedule();
  const fitted = applyPlanBudget(events, derived, [], schedule);
  assert.deepEqual(fitted.events, events);
  assert.ok(fitted.events.every((row) => row.spatial === true));
  assert.ok(
    fitted.derivedEvents.every(
      (row) => row.spatial === false && row.droppedForBudget === true,
    ),
  );
  assert.equal(fitted.bounds.nodes, 3 + eventCount * perEvent);
  assert.ok(fitted.bounds.nodes <= PLAN_RESOURCE_CAPS.nodes);

  const roomy = applyPlanBudget(events.slice(0, 2), derived, [], schedule);
  assert.ok(roomy.derivedEvents.every((row) => row.spatial === true));
  assert.ok(roomy.derivedEvents.every((row) => row.droppedForBudget === undefined));

  const overflowing = Array.from({ length: eventCount + 4 }, (_, index) =>
    Object.freeze({ eventId: `busy-${index}`, kind: "net_health", spatial: true }),
  );
  const capped = applyPlanBudget(overflowing, [], [], schedule);
  assert.equal(capped.bounds.nodes, PLAN_RESOURCE_CAPS.nodes);
  assert.ok(
    capped.events.every((row) => row.spatial === true),
    "never blanks the tick",
  );
});

test("map 47's four busy ticks keep their Herald Card and Fallen rows", () => {
  const schedule = beatSchedule();
  for (const [tick, recorded] of Object.entries(BUSY_TICKS)) {
    const events = busyRows(tick, recorded.events);
    const derived = [
      ...busyRows(tick, recorded.derived),
      Object.freeze({
        eventId: `t${tick}-herald`,
        kind: "herald_card",
        spatial: true,
        persistent: true,
        budget: Object.freeze(recorded.herald),
      }),
    ];
    const demand = planResourceDemand(events, derived, [], schedule);
    assert.equal(
      demand.nodes,
      recorded.nodes,
      `tick ${tick} rebuilds the recorded demand`,
    );
    assert.ok(demand.nodes > 512, `tick ${tick} was over the old cap`);
    const fitted = applyPlanBudget(events, derived, [], schedule);
    const isCardOrFallen = (/** @type {Record<string, any>} */ row) =>
      row.kind === "herald_card" || row.kind === "fallen_feed_row";
    const kept = fitted.derivedEvents.filter(isCardOrFallen);
    assert.equal(kept.length, derived.filter(isCardOrFallen).length);
    for (const row of kept) {
      assert.equal(row.spatial, true, `tick ${tick} keeps ${row.eventId}`);
      assert.equal(row.droppedForBudget, undefined, `tick ${tick} drops nothing`);
    }
    assert.equal(fitted.bounds.nodes, recorded.nodes);
  }
});

test("undrawn status-change rows cost nothing, so a busy tick keeps its herald card", () => {
  const perEvent = planEventBudget({
    kind: "activation",
    component: "basic",
  }).nodesPerRecord;
  const perHerald = planEventBudget({ kind: "herald_card" }).nodesPerRecord;
  // Fill the cap with incoming Basics up to just room for one herald card.
  const eventCount = Math.floor((PLAN_RESOURCE_CAPS.nodes - 3 - perHerald) / perEvent);
  const basicsRows = Array.from({ length: eventCount }, (_, index) =>
    Object.freeze({
      eventId: `basic-${index}`,
      kind: "activation",
      component: "basic",
      spatial: true,
    }),
  );
  // Many status changes on the same tick (a replay frame with 2 or 3 per body).
  const statusRows = Array.from({ length: 30 }, (_, index) =>
    Object.freeze({
      eventId: `status-${index}`,
      kind: "status_lifecycle",
      lifecycle: index % 2 === 0 ? "applied" : "expired",
      spatial: true,
    }),
  );
  const herald = Object.freeze({
    eventId: "herald-0",
    kind: "herald_card",
    spatial: true,
  });
  const schedule = beatSchedule();
  const fitted = applyPlanBudget(
    [...basicsRows, ...statusRows],
    [herald],
    [],
    schedule,
  );
  assert.equal(fitted.derivedEvents[0].spatial, true);
  assert.equal(fitted.derivedEvents[0].droppedForBudget, undefined);
  assert.equal(fitted.bounds.nodes, 3 + eventCount * perEvent + perHerald);
  // A Ward Lift still pays its own allowance.
  const wardLift = Object.freeze({
    eventId: "ward-lift",
    kind: "status_lifecycle",
    cueSemantic: "spawn_shield_expired",
    spatial: true,
  });
  const withLift = planResourceDemand([wardLift], [], [], schedule);
  assert.equal(withLift.nodes, 3 + statuses.BUDGET.nodesPerRecord);
});
