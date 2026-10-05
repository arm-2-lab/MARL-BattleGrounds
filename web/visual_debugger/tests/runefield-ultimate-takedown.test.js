/**
 * @file Check that choreography-plan.js gives each herald_card fall the
 * `ultimate` caster that looks/heralds.js needs for an Ultimate Takedown.
 * Oracle View: an ultimate ability_activated event on the victim whose source
 * also has a lethal_damage_contribution row to it sets ultimate to
 * {classId, publicAgentId} and the herald names "Ultimate Takedown"; a Basic
 * blow, or an Ultimate from a source with no lethal contribution, leaves it
 * null and no single-fall card shows. Agent POV: only a damaging Ultimate from
 * a source the team saw at the start of the tick sets it; a non-damaging
 * Ultimate (Arcane Burst) leaves it null.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { buildChoreographyPlan } from "../src/choreography-plan.js";
import * as heralds from "../src/looks/heralds.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  enableAllVisualFilters,
} from "../src/visual-filters.js";

const ALL_FILTERS = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);

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

/** @param {string} name @returns {Promise<Record<string, any>>} */
async function presentation(name) {
  const raw = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  return structuredClone(raw.presentations[name]);
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

/** @param {Record<string, any>} inventory @param {Record<string, any>[]} events
 * @param {string} separator @param {string} prefix */
function installEvents(inventory, events, separator, prefix) {
  const ordered = events
    .map((event) => ({ ...event, phase_rank: PHASE_RANK[event.event_kind] }))
    .sort((left, right) => left.phase_rank - right.phase_rank)
    .map(
      (event, ordinal) =>
        /** @type {Record<string, any>} */ ({
          ...event,
          event_id: `${prefix}:${separator}:${String(ordinal).padStart(4, "0")}`,
          ordinal,
        }),
    );
  inventory.events = ordered;
  inventory.event_count = ordered.length;
  inventory.ordered_event_ids = ordered.map(({ event_id }) => event_id);
  inventory.ordered_event_kinds = ordered.map(({ event_kind }) => event_kind);
}

/** @param {{component: string, lethalSource: string}} options */
async function oracleHerald(options) {
  const raw = await presentation("replay_oracle");
  const latest = raw.latest_events;
  installEvents(
    latest,
    [
      {
        event_kind: "ability_activated",
        ability_component: options.component,
        source_anchor: anchor(latest, "agent-slot-1", "transition_start"),
        recipient_anchor: anchor(latest, "agent-slot-6", "transition_start"),
      },
      {
        event_kind: "agent_died",
        recipient_anchor: anchor(latest, "agent-slot-6", "successor"),
        home_fall: false,
      },
      {
        event_kind: "lethal_damage_contribution",
        source_anchor: anchor(latest, options.lethalSource, "successor"),
        recipient_anchor: anchor(latest, "agent-slot-6", "successor"),
        attributed_death_damage: 40,
      },
    ],
    "event",
    latest.incoming_transition_id,
  );
  const plan = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(raw),
    surface,
    ALL_FILTERS,
  );
  assert.ok(plan);
  const herald = plan.derivedEvents.find((event) => event.kind === "herald_card");
  assert.ok(herald);
  assert.equal(herald.falls.length, 1);
  return herald;
}

/** @param {string} sourceId */
async function povHerald(sourceId) {
  const raw = await presentation("replay_no_shared_obs_agent_pov");
  const visual = raw.visual_events;
  installEvents(
    visual,
    [
      {
        event_kind: "ability_activated",
        ability_component: "ultimate",
        source_anchor: anchor(visual, sourceId, "transition_start"),
        recipient_anchor: anchor(visual, "agent-slot-2", "transition_start"),
      },
      {
        event_kind: "agent_died",
        recipient_anchor: anchor(visual, "agent-slot-2", "successor"),
      },
    ],
    "visual-event",
    visual.incoming_recipient_transition_id,
  );
  const plan = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(raw),
    surface,
    ALL_FILTERS,
  );
  assert.ok(plan);
  const herald = plan.derivedEvents.find((event) => event.kind === "herald_card");
  assert.ok(herald);
  assert.equal(herald.falls.length, 1);
  return herald;
}

test("Oracle View names an Ultimate Takedown only for a lethal contributor's Ultimate", async () => {
  const herald = await oracleHerald({
    component: "ultimate",
    lethalSource: "agent-slot-1",
  });
  assert.deepEqual(herald.falls[0].ultimate, {
    classId: 2,
    publicAgentId: "agent-slot-1",
  });
  assert.ok(Object.isFrozen(herald.falls[0].ultimate));
  const card = heralds.records({ event: herald });
  assert.equal(card.length, 1);
  assert.equal(card[0].title, "Ultimate Takedown");
  assert.equal(card[0].diamondClassKey, "warrior");

  for (const options of [
    { component: "basic", lethalSource: "agent-slot-1" },
    { component: "ultimate", lethalSource: "agent-slot-0" },
  ]) {
    const plain = await oracleHerald(options);
    assert.equal(plain.falls[0].ultimate, null);
    assert.equal(heralds.records({ event: plain }).length, 0);
  }
});

test("Agent POV sets ultimate only from a seen damaging Ultimate", async () => {
  const seen = await povHerald("agent-slot-1");
  assert.deepEqual(seen.falls[0].ultimate, {
    classId: 2,
    publicAgentId: "agent-slot-1",
  });
  assert.equal(heralds.records({ event: seen })[0]?.title, "Ultimate Takedown");
  const burst = await povHerald("agent-slot-0");
  assert.equal(burst.falls[0].ultimate, null);
});
