/**
 * @file Check the Fallen panel's history from the server (requiems.js
 * fallenFeedHistory with the frame's recent_falls host fact) and the shared
 * damaging-blow table.
 *
 * Contracts checked:
 * - After a seek (no memory), the panel rows built from recent_falls equal the
 *   rows the browser's own tick-by-tick carry gives after playing the same
 *   ticks: same victims, killers in the plan's order, killing team, points
 *   and Ultimate mark (fallenFeedRecords output compared whole).
 * - The window: a fall stays for the 7 ticks after it and leaves on the 8th;
 *   a row for the shown tick or later is ignored (the plan draws the shown
 *   tick); an unknown victim is left out and an unknown killer is dropped.
 * - With recent_falls, a view change or a skip keeps the server's rows (no
 *   memory needed); without it (live play) the carry still works.
 * - The damaging-blow table (a Basic deals damage unless it is the Priest's;
 *   an Ultimate for the Warrior, Hunter and Rogue) equals the browser's
 *   activation tokens (vocabulary.js activationImpactSemantic) for all five
 *   classes. tests/test_viewer_recent_falls.py checks the same table against
 *   the server rule (viewer.champion_streaks), so a drift fails loudly.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { canonicalAgentIdentity } from "../src/agent-identity.js";
import { fallenFeedHistory, fallenFeedRecords } from "../src/looks/requiems.js";
import {
  activationImpactSemantic,
  resolveVisualToken,
  ultimateTokenFromClassId,
} from "../src/vocabulary.js";

// (Basic deals damage, Ultimate deals damage) per class ID. Keep equal to
// DAMAGING_BY_CLASS in tests/test_viewer_recent_falls.py.
const DAMAGING_BY_CLASS = Object.freeze({
  1: [true, false],
  2: [true, true],
  3: [true, true],
  4: [true, true],
  5: [false, false],
});

/** @type {Record<string, {public_agent_id: string, class_id: number, team_id: number, display_agent_id: string}>} */
const DIRECTORY = {
  2: { public_agent_id: "agent-2", class_id: 3, team_id: 1, display_agent_id: "2" },
  3: { public_agent_id: "agent-3", class_id: 4, team_id: 1, display_agent_id: "3" },
  5: { public_agent_id: "agent-5", class_id: 1, team_id: 2, display_agent_id: "5" },
  6: { public_agent_id: "agent-6", class_id: 2, team_id: 2, display_agent_id: "6" },
  7: { public_agent_id: "agent-7", class_id: 3, team_id: 2, display_agent_id: "7" },
};
const byPublicId = new Map(
  Object.values(DIRECTORY).map((row) => [row.public_agent_id, row]),
);

/** @param {string} id */
const identity = (id) => {
  const row = /** @type {any} */ (byPublicId.get(id));
  return {
    ...canonicalAgentIdentity(row),
    publicAgentId: id,
    classId: row.class_id,
    teamId: row.team_id,
  };
};

// The two falls as the plan gives them on their own tick.
const PLAN_ROWS = Object.freeze({
  117: [
    {
      kind: "fallen_feed_row",
      victim: { ...identity("agent-5"), presentationKey: "key-5" },
      victimPresentationKey: "key-5",
      killers: [
        { ...identity("agent-2"), presentationKey: "key-2", lastSeen: false },
        { ...identity("agent-3"), presentationKey: "key-3", lastSeen: false },
      ],
      killingTeamId: 1,
      points: 1,
      ultimate: { classId: 4, publicAgentId: "agent-3" },
    },
  ],
  118: [
    {
      kind: "fallen_feed_row",
      victim: { ...identity("agent-2"), presentationKey: "key-2" },
      victimPresentationKey: "key-2",
      killers: [
        { ...identity("agent-6"), presentationKey: "key-6", lastSeen: false },
        { ...identity("agent-7"), presentationKey: "key-7", lastSeen: true },
      ],
      killingTeamId: 2,
      points: 2,
      ultimate: null,
    },
  ],
});

// The same falls as the server's recent_falls records (killers by slot).
const RECENT_FALLS = Object.freeze([
  {
    tick: 117,
    victim_public_agent_id: "agent-5",
    killer_public_agent_ids: ["agent-3", "agent-2"],
    points: 1,
    ultimate_public_agent_id: "agent-3",
  },
  {
    tick: 118,
    victim_public_agent_id: "agent-2",
    killer_public_agent_ids: ["agent-6", "agent-7"],
    points: 2,
    ultimate_public_agent_id: null,
  },
]);

/** @param {number} step @param {string} [audienceKey] */
const plan = (step, audienceKey = "oracle") => ({
  audienceKey,
  simulatorStep: step,
  derivedEvents: /** @type {Record<number, any[]>} */ (PLAN_ROWS)[step] ?? [],
});

/** @param {string} id */
const identityOf = (id) => byPublicId.get(id) ?? null;
/** @param {string} id */
const displayIdOf = (id) => byPublicId.get(id)?.display_agent_id ?? null;

/** @param {ReturnType<typeof fallenFeedHistory>} history */
const shown = (history) =>
  fallenFeedRecords({
    tick: /** @type {number} */ (history.tick),
    rows: history.rows,
    displayIdOf,
  });

test("a seek shows the same Fallen rows from recent_falls as playing up to the tick", () => {
  for (const tick of [118, 119, 122, 124]) {
    let played = fallenFeedHistory(null, plan(115));
    for (let step = 116; step <= tick; step += 1) {
      played = fallenFeedHistory(played.memory, plan(step));
    }
    const sought = fallenFeedHistory(null, plan(tick), {
      rows: RECENT_FALLS,
      identityOf,
    });
    assert.deepEqual(shown(sought), shown(played), `tick ${tick}`);
    assert.ok(shown(sought).length > 0, `tick ${tick} shows rows`);
  }
  const atEdge = fallenFeedHistory(null, plan(125), { rows: RECENT_FALLS, identityOf });
  assert.deepEqual(
    atEdge.rows.map((row) => `${row.tick}:${row.victim.publicAgentId}`),
    ["118:agent-2"],
    "the fall of tick 117 leaves on tick 125",
  );
  const ultimate = shown(
    fallenFeedHistory(null, plan(120), { rows: RECENT_FALLS, identityOf }),
  ).find((row) => row.tick === 117);
  assert.deepEqual(ultimate?.ability, { kind: "ultimate", classKey: "rogue" });
});

test("recent_falls ignores the shown tick, unknown names and needs no memory", () => {
  const rows = [
    ...RECENT_FALLS,
    { ...RECENT_FALLS[0], tick: 120, victim_public_agent_id: "agent-5" },
    { ...RECENT_FALLS[0], tick: 119, victim_public_agent_id: "agent-404" },
    {
      ...RECENT_FALLS[1],
      tick: 119,
      victim_public_agent_id: "agent-3",
      killer_public_agent_ids: ["agent-404", "agent-6"],
    },
  ];
  const history = fallenFeedHistory(null, plan(120), { rows, identityOf });
  assert.deepEqual(
    history.rows.map((row) => `${row.tick}:${row.victim.publicAgentId}`),
    ["117:agent-5", "118:agent-2", "119:agent-3"],
  );
  assert.deepEqual(
    history.rows[2].killers?.map((killer) => killer.publicAgentId),
    ["agent-6"],
  );
  // A view change or a skip with recent_falls gives the server's rows at once.
  const other = fallenFeedHistory(history.memory, plan(122, "pov:agent-0"), {
    rows: RECENT_FALLS,
    identityOf,
  });
  assert.equal(other.rows.length, 2);
  // Live play (no recent_falls) still carries rows tick by tick.
  let live = fallenFeedHistory(null, plan(117));
  live = fallenFeedHistory(live.memory, plan(118));
  assert.equal(live.rows.length, 2);
});

test("the shared damaging-blow table equals the browser's activation tokens", () => {
  for (const [classText, [basic, ultimate]] of Object.entries(DAMAGING_BY_CLASS)) {
    const classId = Number(classText);
    const basicToken = resolveVisualToken(
      "activation",
      classId === 5 ? "basic_heal" : "basic_damage",
      {},
    );
    assert.equal(activationImpactSemantic(basicToken.tokenId) === "damage", basic);
    assert.equal(
      activationImpactSemantic(ultimateTokenFromClassId(classId, {}).tokenId) ===
        "damage",
      ultimate,
      `class ${classId} Ultimate`,
    );
  }
});
