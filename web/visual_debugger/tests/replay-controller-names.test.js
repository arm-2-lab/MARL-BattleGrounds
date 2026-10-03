/**
 * @file Check that the Replay Viewer names each champion's controller from the
 * replay's recorded per-champion names (match_summary.slot_controller_names),
 * exactly as recorded: controllerRosterFromFrame builds the one controller
 * model (scorebug line, "Team A Controllers" list, Team Rosters strip) from
 * them, a recorded "Manual" stays "Manual" (never "You"), and the Champion
 * Card's "played by" names the champion's own controller. A slot recorded as
 * "Manual" gets the manual flag: its controller is listed first, as "You" is
 * live, and when the names do not fit the short line counts it as the person,
 * not as a policy ("Manual + 3 Policies", or "Manual" alone). Only the flag
 * decides this: live play never sets it, so a live policy named "Manual"
 * still counts as a policy. An old replay without the names falls back to
 * each team's recorded name, as before. The fixture's replay frame has active
 * slots 0, 1, 2 and 5, 6.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  collapsedControllerLine,
  controllerLineParts,
  controllerRoster,
} from "../src/controller-names.js";
import { controllerRosterFromFrame } from "../src/match-summary.js";
import { championCardFacts } from "../src/shell.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

const LONG = "Recurrent MAPPO · Pool Seed 7 Best";

/** @param {ReadonlyArray<string | null> | null} names */
async function replayWith(names) {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  raw.match_summary.slot_controller_names = names;
  return normalizeAuthorizedPresentationFrameV1(raw);
}

/** @param {import("../src/controller-names.js").ControllerRoster} roster */
function listed(roster) {
  return roster.teams.map((team) =>
    team.controllers.map((controller) => [
      controller.name,
      controller.you,
      controller.manual,
      controller.champions.map((champion) => champion.slot),
    ]),
  );
}

test("recorded per-champion names build the Replay Viewer's controller model", async () => {
  const frame = await replayWith([
    "Manual",
    LONG,
    LONG,
    null,
    null,
    "tdm-beta",
    "tdm-beta",
    null,
    null,
    null,
  ]);
  const roster = controllerRosterFromFrame(frame, null);
  assert.deepEqual(listed(roster), [
    [
      ["Manual", false, true, [0]],
      [LONG, false, false, [1, 2]],
    ],
    [["tdm-beta", false, false, [5, 6]]],
  ]);
  assert.equal(collapsedControllerLine(roster.teams[0]), "Manual + 1 Policy");
  assert.equal(collapsedControllerLine(roster.teams[1]), "1 Policy");
  const human = controllerRosterFromFrame(
    await replayWith(["Manual", "Manual", "Manual", ...Array(7).fill(null)]),
    null,
  );
  assert.deepEqual(listed(human)[0], [["Manual", false, true, [0, 1, 2]]]);
  assert.equal(collapsedControllerLine(human.teams[0]), "Manual");
});

test("a recorded Manual slot is listed first and is never counted as a policy", async () => {
  const roster = controllerRosterFromFrame(
    await replayWith([
      LONG,
      "researcher_policy",
      "Manual",
      null,
      null,
      "tdm-beta",
      "tdm-beta",
      null,
      null,
      null,
    ]),
    null,
  );
  assert.deepEqual(listed(roster)[0], [
    ["Manual", false, true, [2]],
    [LONG, false, false, [0]],
    ["researcher_policy", false, false, [1]],
  ]);
  assert.deepEqual(controllerLineParts(roster.teams[0]), [
    "Manual",
    LONG,
    "researcher_policy",
  ]);
  assert.equal(collapsedControllerLine(roster.teams[0]), "Manual + 2 Policies");
  // A full five-champion team, as in a saved BattleClient game.
  const saved = controllerRoster({
    slotLabels: [
      "Manual",
      LONG,
      "tdm-beta",
      "researcher_policy",
      LONG,
      ...Array(5).fill(null),
    ],
    recordedManualSlots: [0],
  });
  assert.equal(collapsedControllerLine(saved.teams[0]), "Manual + 3 Policies");
});

test("live play never sets the manual flag, whatever a policy is named", () => {
  const live = controllerRoster({
    manualSlots: [0],
    slotLabels: ["Manual", "Manual", "tdm-beta", ...Array(7).fill(null)],
  });
  assert.equal(collapsedControllerLine(live.teams[0]), "You + 2 Policies");
  const policiesOnly = controllerRoster({
    manualSlots: [],
    slotLabels: ["tdm-beta", "Manual", "tdm-beta", ...Array(7).fill(null)],
  });
  assert.deepEqual(listed(policiesOnly)[0], [
    ["tdm-beta", false, false, [0, 2]],
    ["Manual", false, false, [1]],
  ]);
  assert.equal(collapsedControllerLine(policiesOnly.teams[0]), "2 Policies");
  assert.ok(
    [...live.teams, ...policiesOnly.teams].every((team) =>
      team.controllers.every((controller) => controller.manual === false),
    ),
  );
});

test("an old replay without slot names keeps one team name per team", async () => {
  const frame = await replayWith(null);
  assert.deepEqual(listed(controllerRosterFromFrame(frame, null)), [
    [["Policy 0 / Policy 1 / Policy 2", false, false, [0, 1, 2]]],
    [["Policy 5 / Policy 6", false, false, [5, 6]]],
  ]);
});

test("the Champion Card names the champion's own recorded controller", async () => {
  const names = [
    "Manual",
    LONG,
    "researcher_policy",
    null,
    null,
    "tdm-beta",
    "tdm-beta",
    null,
    null,
    null,
  ];
  /** @param {unknown} presentation @param {string} publicAgentId */
  const playedBy = (presentation, publicAgentId) =>
    championCardFacts({
      presentation: /** @type {Readonly<Record<string, any>>} */ (presentation),
      combatConfiguration: null,
      subject: { presentationKey: null, slot: null, publicAgentId, classId: null },
    })?.playedBy;
  const frame = await replayWith(names);
  assert.equal(playedBy(frame, "agent-slot-1"), LONG);
  assert.equal(playedBy(frame, "agent-slot-2"), "researcher_policy");
  assert.equal(playedBy(frame, "agent-slot-6"), "tdm-beta");
  const old = await replayWith(null);
  assert.equal(playedBy(old, "agent-slot-1"), "Policy 0 / Policy 1 / Policy 2");
});
