/**
 * @file Check the arena's Arcane Blast During Burst input (the Comet Bolt):
 * choreography-plan.js sets an activation's sourceKindled only when this
 * audience has its own proof that the source's Mage Burst was on during the
 * tick. Oracle View: a Burst row on the source that proves Burst was already
 * on (aged to zero, or refreshed) sets it; no Burst row, or a Burst applied
 * fresh this tick, leaves it false.
 * Agent POV: the team's own start-of-tick observation (latest_events deltas)
 * showing Burst on a seen source sets it; the same frame without that Burst
 * leaves it false. With the Arcane Blast During Burst switch off the Basic's
 * kindled paint part is false, so the switch changes the drawn blow.
 *
 * It also checks that the live plan and reach lines grow with the field as the
 * vision's do: hotbar.markScale is the field's pixels per world unit over the
 * vision's 50 (1 for a transform with no length), every stroke width and dash
 * length in styles/hotbar.css is multiplied by --mark-scale, and the live
 * reach rings, veil edges and target brackets in styles/battleclient/arena.css
 * use the vision's sizes times --mark-scale, with flat dash ends on the Basic
 * ring and round ones on the Ultimate ring.
 *
 * Order Studs keep three decimals: on the fitted 1440 x 900 field (868.90625
 * x 447.921875 px) each Basic stud (one circle record of four) lies within
 * 0.0015 px of its exact
 * place (1.45 body radii from the body center at a diagonal) and each stud's
 * radius within 0.0005 px of 3.2 x (body radius / 25); two decimals missed
 * that by up to 0.005 px, enough to move a 3 px dot's edge pixels. Watch Only
 * No Combat studs show at 0.585 (the vision's 90% dots in a 65% group):
 * styles/hotbar.css gives `.hotbar-studs--no-combat.hotbar-studs--away` that
 * opacity after the 90% No Combat rule.
 *
 * Rune Zones gets the tick's raw event rows: scene.js incomingEventRows hands
 * looks/statuses.js the `events` array of the audience's incoming summary
 * (never the summary record itself, and an empty list when there is none), so
 * a Freezing Trap broken by a blow this tick draws its Frozen pieces at START
 * and shatters them over 60 ms (a 1 ms cut with the Freezing Trap Break switch
 * off). Given the summary record instead, no Shatter is drawn.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  authorizedPresentationInspectionState,
  authorizedPresentationSceneView,
} from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { buildChoreographyPlan } from "../src/choreography-plan.js";
import { createViewportTransform } from "../src/layout.js";
import * as hotbar from "../src/looks/hotbar.js";
import { markScale, VISION_PIXELS_PER_UNIT } from "../src/looks/hotbar.js";
import * as statuses from "../src/looks/statuses.js";
import { incomingEventRows } from "../src/scene.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  enableAllVisualFilters,
} from "../src/visual-filters.js";

const ALL_FILTERS = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
const BURST = "mage_burst_damage_amplification";

/** @type {Readonly<Record<string, number>>} */
const PHASE_RANK = Object.freeze({
  ability_activated: 20,
  status_aged_to_zero: 100,
  status_applied: 100,
  status_refreshed_or_extended: 100,
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

/** @param {Record<string, any>} plan @returns {Record<string, any>} */
function basicBlow(plan) {
  const blow = plan.events.find(
    (/** @type {Record<string, any>} */ event) =>
      event.kind === "activation" && event.component === "basic",
  );
  assert.ok(blow, "the plan has the Basic activation");
  return blow;
}

/** @param {string | null} burstRow @param {Record<string, boolean>} [filters] */
async function oracleBlow(burstRow, filters = ALL_FILTERS) {
  const raw = await presentation("replay_oracle");
  const latest = raw.latest_events;
  /** @type {Record<string, any>[]} */
  const events = [
    {
      event_kind: "ability_activated",
      ability_component: "basic",
      source_anchor: anchor(latest, "agent-slot-0", "transition_start"),
      recipient_anchor: anchor(latest, "agent-slot-5", "transition_start"),
    },
  ];
  if (burstRow !== null) {
    events.push({
      event_kind: burstRow,
      recipient_anchor: anchor(latest, "agent-slot-0", "successor"),
      status_channel: 7,
      status_id: BURST,
      ...(burstRow === "status_applied"
        ? { source_anchor: anchor(latest, "agent-slot-0", "successor") }
        : {}),
    });
  }
  installEvents(latest, events, "event", latest.incoming_transition_id);
  const plan = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(raw),
    surface,
    filters,
  );
  assert.ok(plan, "the Oracle plan builds");
  return basicBlow(plan);
}

/** @param {boolean} burstAtStart */
async function povBlow(burstAtStart) {
  const raw = await presentation("replay_shared_obs_agent_pov");
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
    ],
    "visual-event",
    visual.incoming_recipient_transition_id,
  );
  const delta = raw.latest_events.deltas.find(
    (/** @type {Record<string, any>} */ row) =>
      row.start_observation?.public_agent_id === "agent-slot-1",
  );
  assert.ok(delta, "the fixture has the team's start observation of agent-slot-1");
  if (burstAtStart) {
    delta.start_observation.statuses = [
      ...delta.start_observation.statuses,
      {
        status_channel: 7,
        status_id: BURST,
        family: "damage_amplification",
        configured_duration_steps: 5,
        remaining_duration: 5,
        mechanic_action_component: "ultimate",
        magnitude_kind: "damage_multiplier",
        magnitude: 1.5,
        breaks_on_positive_damage: false,
      },
    ];
  }
  const plan = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(raw),
    surface,
    ALL_FILTERS,
  );
  assert.ok(plan, "the Agent POV plan builds");
  return basicBlow(plan);
}

test("Oracle View kindles a Basic only when its own rows show Burst already on", async () => {
  assert.equal((await oracleBlow(null)).sourceKindled, false);
  for (const row of ["status_aged_to_zero", "status_refreshed_or_extended"]) {
    const blow = await oracleBlow(row);
    assert.equal(blow.sourceKindled, true, row);
    assert.equal(blow.paintParts.kindled, true);
  }
  assert.equal((await oracleBlow("status_applied")).sourceKindled, false);
});

test("Arcane Blast During Burst off turns the kindled paint part off", async () => {
  const off = await oracleBlow("status_aged_to_zero", {
    ...ALL_FILTERS,
    kindled_blast: false,
  });
  assert.equal(off.sourceKindled, true);
  assert.equal(off.paintParts.kindled, false);
});

test("Agent POV kindles a Basic from the team's own start observation", async () => {
  assert.equal((await povBlow(true)).sourceKindled, true);
  assert.equal((await povBlow(false)).sourceKindled, false);
});

/** @param {string} path @returns {Promise<string>} */
async function css(path) {
  return await readFile(new URL(path, import.meta.url), "utf8");
}

/**
 * @param {string} text
 * @param {string} selector
 * @param {boolean} [last]
 * @returns {string}
 */
function ruleBody(text, selector, last = false) {
  const key = `${selector} {`;
  const start = last ? text.lastIndexOf(key) : text.indexOf(key);
  assert.notEqual(start, -1, `missing rule ${selector}`);
  return text.slice(start, text.indexOf("}", start));
}

test("Basic Order Studs sit within 0.0015 px of their exact places on the fitted field", async () => {
  const frames = await Promise.all(
    ["live_oracle", "live_shared_obs_agent_pov"].map(async (name) =>
      normalizeAuthorizedPresentationFrameV1(await presentation(name)),
    ),
  );
  let checked = 0;
  for (const frame of frames) {
    const scene = /** @type {Record<string, any>} */ (
      authorizedPresentationSceneView(frame)
    );
    const inspection = /** @type {Record<string, any>} */ (
      authorizedPresentationInspectionState(frame).inspection
    );
    const transform = createViewportTransform({
      worldWidth: scene.map.width,
      worldHeight: scene.map.height,
      viewportWidth: 868.90625,
      viewportHeight: 447.921875,
      padding: 13.4713,
    });
    const studs = hotbar
      .records({
        scene,
        transform,
        audience: scene.audience,
        inspection: {
          ...inspection,
          draft_action: { ...inspection.draft_action, armed_lane: "basic" },
        },
      })
      .filter((record) => record.kind === "order-studs");
    assert.equal(
      studs.length,
      4,
      `${scene.audience}: the controlled champion has studs`,
    );
    for (const stud of studs) {
      assert.equal(stud.style, "basic");
      assert.equal(stud.shape, "dot");
    }
    const body = scene.agents.find(
      (/** @type {any} */ agent) =>
        agent.presentation_key === scene.selection.controlled_presentation_key,
    );
    const center = transform.worldToScreen(body.position);
    const r = transform.worldLengthToScreen(body.radius);
    const size = (3.2 * r) / 25;
    const reach = 1.45 * r * Math.SQRT1_2;
    const places = [
      [center.x + reach, center.y - reach],
      [center.x + reach, center.y + reach],
      [center.x - reach, center.y + reach],
      [center.x - reach, center.y - reach],
    ];
    // Each dot is one circle record: its center (x, y) and radius r.
    const drawn = studs.map((stud) => [stud.x, stud.y, stud.r]);
    // Two-decimal rounding would miss by more than the bound on this field.
    assert.ok(
      places
        .flat()
        .some((value) => Math.abs(Math.round(value * 100) / 100 - value) > 0.0015),
    );
    drawn.forEach(([x, y, radius], index) => {
      assert.ok(Math.abs(radius - size) <= 0.0005, `radius ${radius} vs ${size}`);
      assert.ok(
        Math.abs(x - places[index][0]) <= 0.0015,
        `x ${x} vs ${places[index][0]}`,
      );
      assert.ok(
        Math.abs(y - places[index][1]) <= 0.0015,
        `y ${y} vs ${places[index][1]}`,
      );
    });
    checked += 1;
  }
  assert.equal(checked, 2);
});

test("Watch Only No Combat studs show at 0.585, after the 90% No Combat rule", async () => {
  const text = await css("../src/styles/hotbar.css");
  const away = ruleBody(text, ".hotbar-studs--no-combat.hotbar-studs--away");
  assert.match(away, /opacity: 0\.585;/);
  assert.ok(
    text.indexOf(".hotbar-studs--no-combat.hotbar-studs--away {") >
      text.indexOf(".hotbar-studs--no-combat {"),
  );
  assert.match(ruleBody(text, ".hotbar-studs--away"), /opacity: 0\.65;/);
});

test("markScale is the field's pixels per world unit over the vision's 50", () => {
  assert.equal(VISION_PIXELS_PER_UNIT, 50);
  assert.equal(markScale({ worldLengthToScreen: (value) => value * 52.035 }), 1.0407);
  assert.equal(markScale({ worldLengthToScreen: (value) => value * 50 }), 1);
  assert.equal(markScale({ worldLengthToScreen: () => 0 }), 1);
  assert.equal(markScale({ worldLengthToScreen: () => Number.NaN }), 1);
});

test("every hotbar line width and dash grows with --mark-scale", async () => {
  const text = await css("../src/styles/hotbar.css");
  const values = [...text.matchAll(/stroke-(?:width|dasharray):([^;]+);/g)].map(
    (match) => match[1].trim(),
  );
  assert.ok(values.length >= 15);
  for (const value of values) {
    for (const part of value.split(/\s+(?=calc)/)) {
      assert.match(part, /^calc\([0-9.]+px \* var\(--mark-scale, 1\)\)$/);
    }
  }
  const ultimate = ruleBody(text, ".hotbar-thread--ultimate");
  assert.match(ultimate, /stroke-width: calc\(2\.5px \* var\(--mark-scale, 1\)\);/);
  assert.match(
    ultimate,
    /stroke-dasharray: calc\(6px \* var\(--mark-scale, 1\)\) calc\(4px \* var\(--mark-scale, 1\)\);/,
  );
  const ghost = ruleBody(text, ".hotbar-ghost");
  assert.match(
    ghost,
    /stroke-dasharray: calc\(2px \* var\(--mark-scale, 1\)\) calc\(3\.4px \* var\(--mark-scale, 1\)\);/,
  );
});

test("live reach rings, veil edges and brackets use the vision's sizes times --mark-scale", async () => {
  const text = (await css("../src/styles/battleclient/arena.css")).replace(/\s+/g, " ");
  const basic = ruleBody(text, '#battlefield .range-ring[data-kind="basic"]');
  assert.match(
    basic,
    /calc\(4px \* var\(--mark-scale, 1\)\) calc\(4px \* var\(--mark-scale, 1\)\)/,
  );
  assert.match(basic, /stroke-linecap: butt;/);
  const ultimate = ruleBody(text, '#battlefield .range-ring[data-kind="ultimate"]');
  assert.match(
    ultimate,
    /calc\(1px \* var\(--mark-scale, 1\)\) calc\(5px \* var\(--mark-scale, 1\)\)/,
  );
  assert.match(ultimate, /stroke-linecap: round;/);
  assert.match(
    ruleBody(text, "#battlefield .range-ring[data-kind]"),
    /stroke-width: calc\(1px \* var\(--mark-scale, 1\)\);/,
  );
  assert.match(
    ruleBody(text, "#battlefield .fog-edge-allied", true),
    /stroke-dasharray: calc\(4px \* var\(--mark-scale, 1\)\) calc\(4px \* var\(--mark-scale, 1\)\);/,
  );
  assert.match(
    ruleBody(text, "#battlefield .selected-reticle"),
    /stroke-width: calc\(2px \* var\(--mark-scale, 1\)\);/,
  );
  assert.match(
    ruleBody(text, "#battlefield .selected-reticle-keyline"),
    /stroke-width: calc\(4px \* var\(--mark-scale, 1\)\);/,
  );
});

function brokenTrapInput(/** @type {ReadonlyArray<unknown>} */ events, breakOn = true) {
  return {
    part: "status-marks",
    scene: {
      audience: "researcher",
      agents: [
        {
          alive: true,
          presentation_key: "agent-5",
          class_id: 1,
          radius: 0.5,
          position: [10, 6],
          statuses: [],
          ultimate_cooldown: 0,
          spawn_shield_remaining: 0,
        },
      ],
    },
    transform: surface,
    audience: "researcher",
    events,
    startSeen: () => true,
    switches: { freezingTrapBreak: breakOn },
  };
}

const TRAP_BROKEN_ROW = Object.freeze({
  event_kind: "status_broken_by_damage",
  status_id: "stun_hunter_trap",
  recipient_anchor: Object.freeze({ presentation_key: "agent-5" }),
});

/** @param {ReadonlyArray<Record<string, any>>} records */
function frozenPieces(records) {
  return records
    .flatMap((record) => record.pieces)
    .filter((piece) => String(piece.id).startsWith("frozen."));
}

test("incomingEventRows gives Rune Zones the summary's own rows", () => {
  const rows = [TRAP_BROKEN_ROW];
  assert.equal(incomingEventRows({ events: rows }), rows);
  assert.deepEqual(incomingEventRows(null), []);
  assert.deepEqual(incomingEventRows({ events: "not rows" }), []);
  assert.deepEqual(incomingEventRows([TRAP_BROKEN_ROW]), []);
});

test("a broken Freezing Trap shatters over 60 ms only when Rune Zones gets the rows", () => {
  const summary = { events: [TRAP_BROKEN_ROW] };
  const shattered = frozenPieces(
    statuses.records(brokenTrapInput(incomingEventRows(summary))),
  );
  assert.ok(shattered.length > 0, "the trap's Frozen pieces are drawn at START");
  for (const piece of shattered) {
    assert.equal(piece.side, "before");
    assert.equal(piece.fadeMs, 60);
  }
  const cut = frozenPieces(
    statuses.records(brokenTrapInput(incomingEventRows(summary), false)),
  );
  assert.ok(cut.length > 0);
  for (const piece of cut) assert.equal(piece.fadeMs, 1);
  // The summary record itself is not a row list: no START trap, no Shatter.
  assert.deepEqual(
    frozenPieces(statuses.records(brokenTrapInput(/** @type {any} */ (summary)))),
    [],
  );
});
