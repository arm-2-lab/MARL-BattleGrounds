/**
 * @file Check the browser half of the Champion Streaks host fact: the frame
 * field champion_streaks (the host's count for this frame's own view) reaches
 * the herald card through choreography-plan.js.
 *
 * Contracts checked:
 * - The normalizer accepts the field, absent or null (older payloads), and
 *   refuses a champion named twice or one this view's own incoming events did
 *   not see at the start of the tick (Oracle View: latest_events; Agent POV:
 *   visual_events).
 * - Oracle View: a fact naming a killer of this tick's named fall gives the
 *   herald_card row streaks [{champion, takedowns}], champion being that
 *   killer's identity record, and looks/heralds.js heads the card "5
 *   Takedowns Without Falling" with the switch on. A fact naming a champion
 *   that is not a killer of a named fall is dropped; no field gives no streak.
 * - Agent POV: the row reads only its own frame's field, so a fact on the
 *   seen striker shows, and the same seen fall without the field shows none.
 *   Oracle View -> Agent POV -> Oracle View plans equal a fresh Oracle plan
 *   (the browser keeps no streak history).
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
const STREAKS_ON = Object.freeze({ champion_streaks: true });

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

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

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

// The Team A Warrior (agent-slot-1) takes down the Team B Mage (agent-slot-6).
function oracleFall() {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const latest = raw.latest_events;
  installEvents(
    latest,
    [
      {
        event_kind: "ability_activated",
        ability_component: "basic",
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
        source_anchor: anchor(latest, "agent-slot-1", "successor"),
        recipient_anchor: anchor(latest, "agent-slot-6", "successor"),
        attributed_death_damage: 40,
      },
    ],
    "event",
    latest.incoming_transition_id,
  );
  return raw;
}

// In Agent POV the team sees agent-slot-1 strike agent-slot-2, who falls.
function povFall() {
  const raw = structuredClone(fixture.presentations.replay_no_shared_obs_agent_pov);
  const visual = raw.visual_events;
  installEvents(
    visual,
    [
      {
        event_kind: "ability_activated",
        ability_component: "basic",
        source_anchor: anchor(visual, "agent-slot-1", "transition_start"),
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
  return raw;
}

/** @param {string} publicAgentId @param {number} takedowns */
function fact(publicAgentId, takedowns, teamId = 1, classId = 2) {
  return {
    public_agent_id: publicAgentId,
    team_id: teamId,
    class_id: classId,
    takedowns,
  };
}

/** @param {Record<string, any>} raw */
async function heraldRow(raw) {
  const plan = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(raw),
    surface,
    ALL_FILTERS,
  );
  assert.ok(plan);
  const herald = plan.derivedEvents.find((event) => event.kind === "herald_card");
  assert.ok(herald);
  return { plan, herald };
}

test("the normalizer accepts the field and refuses unseen or repeated champions", async () => {
  for (const value of [undefined, null, []]) {
    const raw = oracleFall();
    if (value === undefined) delete raw.champion_streaks;
    else raw.champion_streaks = value;
    await normalizeAuthorizedPresentationFrameV1(raw);
  }
  const twice = oracleFall();
  twice.champion_streaks = [fact("agent-slot-1", 5), fact("agent-slot-1", 8)];
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(twice),
    /Champion streaks must name/u,
  );
  const pov = povFall();
  const seen = new Set(
    pov.visual_events.agent_phase_trajectories
      .filter((/** @type {Record<string, any>} */ row) => row.transition_start != null)
      .map((/** @type {Record<string, any>} */ row) => row.agent_public_agent_id),
  );
  const unseen = pov.researcher_space.roster_agents.find(
    (/** @type {Record<string, any>} */ agent) => !seen.has(agent.public_agent_id),
  );
  assert.ok(unseen, "the Agent POV fixture has a champion the team did not see");
  pov.champion_streaks = [
    fact(unseen.public_agent_id, 5, unseen.team_id, unseen.class_id),
  ];
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(pov),
    /Champion streaks must name/u,
  );
});

test("Oracle View heads the herald with the host's streak on a named killer", async () => {
  const raw = oracleFall();
  raw.champion_streaks = [fact("agent-slot-1", 5)];
  const { herald } = await heraldRow(raw);
  const killer = herald.falls[0].killers[0];
  assert.equal(killer.publicAgentId, "agent-slot-1");
  assert.deepEqual(herald.streaks, [{ champion: killer, takedowns: 5 }]);
  assert.equal(herald.streaks[0].champion, killer);
  assert.ok(Object.isFrozen(herald.streaks) && Object.isFrozen(herald.streaks[0]));
  const [card] = heralds.records({ event: herald, filters: STREAKS_ON });
  assert.equal(card.title, "5 Takedowns Without Falling");

  const bystander = oracleFall();
  bystander.champion_streaks = [fact("agent-slot-0", 8, 1, 1)];
  assert.deepEqual((await heraldRow(bystander)).herald.streaks, []);
  assert.deepEqual((await heraldRow(oracleFall())).herald.streaks, []);
});

test("Agent POV reads only its own field and keeps no history", async () => {
  const raw = povFall();
  raw.champion_streaks = [fact("agent-slot-1", 8)];
  const { plan, herald } = await heraldRow(raw);
  assert.equal(herald.streaks.length, 1);
  assert.equal(herald.streaks[0].champion.publicAgentId, "agent-slot-1");
  assert.equal(herald.streaks[0].takedowns, 8);
  const [card] = heralds.records({ event: herald, filters: STREAKS_ON });
  assert.equal(card.title, "8 Takedowns Without Falling");

  // Only the frame's own field counts: without it the same seen fall has no
  // streak, whatever another view's frame carried.
  assert.deepEqual((await heraldRow(povFall())).herald.streaks, []);
  assert.deepEqual((await heraldRow(raw)).plan.derivedEvents, plan.derivedEvents);

  const oracle = oracleFall();
  oracle.champion_streaks = [fact("agent-slot-1", 5)];
  const fresh = (await heraldRow(oracle)).plan;
  await heraldRow(raw);
  const again = (await heraldRow(oracleFall())).plan;
  const back = (await heraldRow(oracle)).plan;
  assert.deepEqual(back.derivedEvents, fresh.derivedEvents);
  assert.notDeepEqual(again.derivedEvents, fresh.derivedEvents);
});
