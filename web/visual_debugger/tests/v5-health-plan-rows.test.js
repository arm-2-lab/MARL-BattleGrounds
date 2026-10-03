/**
 * @file Check the health-change plan rows the V5 life-number add-ons read
 * (`src/choreography-plan.js` for Saved Beat, Last Breath and Battle Text
 * Detail in `src/looks/health-text.js`).
 *
 * Contracts checked: an Oracle View net_health row carries the recipient's
 * public maximumHealth, its projected recipientRadius in CSS px (0 when the
 * row is not drawn), the recorded totalDamage and totalHealing, and
 * damageSources (one {sourcePresentationKey, sourceClassId, amount} per
 * recorded source_damage_output row on it, amount = modified output times the
 * recipient's modifier); an Agent POV row carries null totals and no sources,
 * so it can never claim a save or name an attacker; its paint parts carry
 * savedBeat, lastBreath and battleTextDetail from their switches, which never
 * keep a switched-off number drawn; and the number's reserved cue box stays
 * at the plain 40 x 20 px with the add-ons off and grows for "Saved" and a
 * larger number when they are on.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { buildChoreographyPlan } from "../src/choreography-plan.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  setVisualFilterEnabled,
} from "../src/visual-filters.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

const surface = {
  worldToScreen: (/** @type {any} */ point) => {
    const x = "x" in point ? point.x : Number(point[0]);
    const y = "y" in point ? point.y : Number(point[1]);
    return { x: x * 10 + 100, y: 300 - y * 10 };
  },
  worldLengthToScreen: (/** @type {number} */ length) => length * 10,
  viewportBounds: { left: 0, top: 0, right: 800, bottom: 600, width: 800, height: 600 },
  protectedRects: [],
};

/**
 * One Oracle tick where agent-slot-2 hits agent-slot-1 for 40 and heals it for
 * 15, from 30 down to 5: the blows alone would have killed it (a Saved Beat).
 */
async function savedTick() {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const [activation, healing, resolution, status] = raw.latest_events.events;
  const id = (/** @type {number} */ ordinal) =>
    activation.event_id.replace(/0000$/u, String(ordinal).padStart(4, "0"));
  const damage = {
    event_id: id(1),
    event_kind: "source_damage_output",
    ordinal: 1,
    phase_rank: 30,
    source_anchor: structuredClone(activation.source_anchor),
    recipient_anchor: structuredClone(activation.recipient_anchor),
    raw_damage_output: 40,
    source_modified_damage_output: 50,
    recipient_damage_modifier: 0.8,
    mage_damage_aura_covering_emitters: [],
    warrior_mitigation_aura_covering_emitters: [],
  };
  Object.assign(healing, {
    event_id: id(2),
    ordinal: 2,
    raw_healing_output: 15,
    source_modified_healing_output: 15,
  });
  Object.assign(resolution, {
    event_id: id(3),
    ordinal: 3,
    transition_start_health: 30,
    health_after_combat_resolution: 5,
    realized_net_health_change: -25,
    total_effective_damage: 40,
    total_effective_healing: 15,
  });
  Object.assign(status, { event_id: id(4), ordinal: 4 });
  const events = [activation, damage, healing, resolution, status];
  raw.latest_events.events = events;
  raw.latest_events.event_count = events.length;
  raw.latest_events.ordered_event_ids = events.map((event) => event.event_id);
  raw.latest_events.ordered_event_kinds = events.map((event) => event.event_kind);
  return {
    frame: await normalizeAuthorizedPresentationFrameV1(raw),
    recipientKey: activation.recipient_anchor.presentation_key,
    sourceKey: activation.source_anchor.presentation_key,
  };
}

/** @param {Readonly<Record<string, any>> | null} plan */
function netRow(plan) {
  assert.ok(plan);
  const row = plan.events.find(
    (/** @type {Record<string, any>} */ event) => event.kind === "net_health",
  );
  assert.ok(row);
  return row;
}

/** @param {ReadonlyArray<string>} ids */
function switchedOn(ids) {
  let state = DEFAULT_VISUAL_FILTER_STATE;
  for (const id of ids) state = setVisualFilterEnabled(state, id, true);
  return state;
}

test("an Oracle health row carries the recorded totals and attackers", async () => {
  const { frame, recipientKey, sourceKey } = await savedTick();
  const row = netRow(
    buildChoreographyPlan(frame, surface, DEFAULT_VISUAL_FILTER_STATE),
  );
  assert.equal(row.recipientPresentationKey, recipientKey);
  assert.equal(row.maximumHealth, 200);
  assert.equal(row.recipientRadius, 5);
  assert.equal(row.totalDamage, 40);
  assert.equal(row.totalHealing, 15);
  assert.equal(row.damageSources.length, 1);
  assert.deepEqual(
    { ...row.damageSources[0] },
    {
      sourcePresentationKey: sourceKey,
      sourceClassId: 5,
      amount: 40,
    },
  );
  assert.ok(Object.isFrozen(row.damageSources));
  assert.deepEqual(
    { ...row.paintParts },
    {
      effect: true,
      battleText: true,
      recipientText: true,
      savedBeat: false,
      lastBreath: false,
      battleTextDetail: false,
    },
  );
  const hidden = netRow(
    buildChoreographyPlan(frame, surface, {
      ...DEFAULT_VISUAL_FILTER_STATE,
      scrolling_battle_text: false,
      saved_beat: true,
      battle_text_detail: true,
    }),
  );
  assert.equal(hidden.recipientRadius, 0, "a row that is not drawn has no radius");
  assert.equal(hidden.presentationSuppressed, true, "add-ons keep nothing drawn");
});

test("an Agent POV health row carries no totals and no attackers", async () => {
  for (const name of [
    "replay_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
  ]) {
    const frame = await normalizeAuthorizedPresentationFrameV1(
      structuredClone(fixture.presentations[name]),
    );
    const row = netRow(
      buildChoreographyPlan(
        frame,
        surface,
        switchedOn(["saved_beat", "last_breath", "battle_text_detail"]),
      ),
    );
    assert.equal(row.totalDamage, null, name);
    assert.equal(row.totalHealing, null, name);
    assert.deepEqual([...row.damageSources], [], name);
    assert.equal(row.maximumHealth, 200, name);
    assert.equal(row.paintParts.savedBeat, true, name);
  }
});

test("the reserved cue box is plain with the add-ons off and grows when they are on", async () => {
  const { frame } = await savedTick();
  const plain = netRow(
    buildChoreographyPlan(frame, surface, DEFAULT_VISUAL_FILTER_STATE),
  );
  assert.equal(plain.cueBounds.width, 40);
  assert.equal(plain.cueBounds.height, 20);
  const detailed = netRow(
    buildChoreographyPlan(
      frame,
      surface,
      switchedOn(["saved_beat", "battle_text_detail"]),
    ),
  );
  assert.equal(detailed.paintParts.savedBeat, true);
  assert.equal(detailed.paintParts.battleTextDetail, true);
  assert.ok(detailed.cueBounds.height > plain.cueBounds.height);
});
