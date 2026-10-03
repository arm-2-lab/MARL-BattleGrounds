/**
 * @file Check the Tactics Turn deck's pure contracts (src/tactics-turn.js):
 * the Step pad prints the nine key letters W A S D Q E Z C X with the vision's
 * step names and no arrow is a draft key; the fixed labels are "Lock In Turn",
 * "Cancel Policy Work", "Restart", "Keys On" and "Keys Off · Click The Arena";
 * champions read "Mage · ID 5" with no team word; the target chip keeps an
 * out-of-sight pick as "Warrior · ID 6 · Out Of Sight" with no life; Allowed
 * Now lists only bodies the audience's arena scene draws and the audience's
 * own mask allows (checked on every live fixture frame, Oracle View and Agent
 * POV); the action slots carry the real Basic and Ultimate names, reach,
 * "Ready" or "In 3 Ticks", the lit Rune Ring ticks and the Armed, Auto and Not
 * Allowed tags (after game over an unarmed Basic and Ultimate read Not Allowed
 * even when the last mask allowed them); the order sentence in the vision's words; the turn phase from
 * real states; Do This Now with its beacon; the checklist line and the
 * Policies list rows (policyListRows of the one controller model in
 * src/controller-names.js: every policy of both teams with its full name and
 * "N Champions", Manual left out). It also checks that the module never
 * reads the raw transport HUD or the match summary by name, and reads the
 * researcher space in one place only: servedManualOrdersOf, for the viewing
 * team's own staged orders in Agent POV (ruling P1, 3 Oct 2026).
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { authorizedPresentationSceneView } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { authorizedPresentationInspectionState } from "../src/authorized-presentation-adapter.js";
import { controllerRoster, policyListRows } from "../src/controller-names.js";
import {
  DECK_LABELS,
  DRAFT_KEYS,
  GUIDE_MODES,
  STEP_PAD_KEYS,
  TURN_PHASES,
  actionSlotViews,
  allowedNowTargets,
  championFacts,
  checklistLine,
  doThisNow,
  effectiveLane,
  orderSentence,
  runeRingLit,
  runeRingText,
  targetChipView,
  turnPhase,
} from "../src/tactics-turn.js";

const MAGE = Object.freeze({
  public_agent_id: "agent-slot-5",
  display_agent_id: "5",
  class_id: 1,
  team_id: 2,
  current_health: 62.4,
  maximum_health: 100,
  life_state: "alive",
  basic_interaction_radius: 6,
  ultimate_interaction_radius: 0,
  ultimate_cooldown_remaining: 0,
});
const WARRIOR = Object.freeze({
  public_agent_id: "agent-slot-6",
  display_agent_id: "6",
  class_id: 2,
  team_id: 2,
});

test("the Step pad prints its nine key letters and arrows are not draft keys", () => {
  assert.deepEqual(Object.keys(STEP_PAD_KEYS).sort(), [..."acdeqswxz"].sort());
  for (const [key, { letter }] of Object.entries(STEP_PAD_KEYS)) {
    assert.equal(letter, key.toUpperCase());
    assert.ok(DRAFT_KEYS.has(key), `${key} is a draft key`);
  }
  for (const arrow of ["arrowup", "arrowdown", "arrowleft", "arrowright"]) {
    assert.equal(DRAFT_KEYS.has(arrow), false);
  }
  assert.equal(STEP_PAD_KEYS.x.name, "Stay");
  assert.equal(STEP_PAD_KEYS.q.name, "North-West");
  assert.equal(STEP_PAD_KEYS.q.moveAction, 6);
});

test("the deck's fixed labels use the V5 names", () => {
  assert.equal(DECK_LABELS.lockInButton, "Lock In Turn");
  assert.equal(DECK_LABELS.cancelPolicyWork, "Cancel Policy Work");
  assert.equal(DECK_LABELS.restart, "Restart");
  assert.equal(DECK_LABELS.youControl, "You Control This Agent");
  assert.deepEqual(
    [DECK_LABELS.keysOn, DECK_LABELS.keysOff],
    ["Keys On", "Keys Off · Click The Arena"],
  );
  assert.deepEqual(TURN_PHASES, ["You Plan", "Lock In", "Policies Choose", "All Act"]);
  assert.deepEqual(GUIDE_MODES, ["guide", "hints", "off"]);
  for (const value of Object.values(DECK_LABELS)) {
    assert.doesNotMatch(value, /Submit|Arm Basic|Azure|Crimson|Blue|Red\b/u);
  }
});

test("champions read Class · ID n with no team word", () => {
  const facts = championFacts(MAGE);
  assert.equal(facts?.label, "Mage · ID 5");
  assert.equal(facts?.calling, "The Astral Walker");
  assert.equal(facts?.lifeText, "62 / 100");
  assert.doesNotMatch(String(facts?.label), /Team/u);
  assert.equal(championFacts(null), null);
  assert.equal(championFacts({ ...MAGE, current_health: undefined })?.lifeText, null);
});

test("the target chip shows life only for a drawn body and keeps an unseen pick", () => {
  assert.deepEqual(targetChipView(null, [MAGE]), { state: "none" });
  assert.deepEqual(
    targetChipView({ target_kind: "no_target", target_public_agent_id: "x" }, [MAGE]),
    { state: "none" },
  );
  const seen = targetChipView(
    { target_kind: "visible_authorized_agent", target_public_agent_id: "agent-slot-5" },
    [MAGE],
  );
  assert.equal(seen.state, "seen");
  assert.equal(seen.state === "seen" && seen.facts?.lifeText, "62 / 100");
  const unseen = targetChipView(
    {
      target_kind: "axis_only_authorized_agent",
      target_public_agent_id: "agent-slot-6",
    },
    [MAGE],
    [WARRIOR],
  );
  assert.equal(unseen.state, "out_of_sight");
  assert.equal(
    unseen.state === "out_of_sight" && unseen.label,
    "Warrior · ID 6 · Out Of Sight",
  );
  assert.equal(unseen.state === "out_of_sight" && unseen.facts?.lifeText, null);
});

test("Allowed Now lists only allowed bodies that the scene draws", () => {
  const mask = {
    target_actions: [
      { target_action: 0, target_kind: "no_target" },
      {
        target_action: 1,
        target_kind: "visible_authorized_agent",
        target_public_agent_id: "agent-slot-5",
      },
      {
        target_action: 2,
        target_kind: "visible_authorized_agent",
        target_public_agent_id: "agent-slot-6",
      },
      {
        target_action: 3,
        target_kind: "axis_only_authorized_agent",
        target_public_agent_id: "agent-slot-7",
      },
    ],
    target_use_ultimate_joint_mask: [
      [true, true],
      [true, false],
      [true, true],
      [true, true],
    ],
  };
  const rows = allowedNowTargets(mask, [MAGE]);
  assert.deepEqual(
    rows.map((row) => [row.publicAgentId, row.label, row.basic, row.ultimate]),
    [["agent-slot-5", "Mage · ID 5", true, false]],
  );
  assert.deepEqual(allowedNowTargets(null, [MAGE]), []);
  const closed = {
    ...mask,
    target_use_ultimate_joint_mask: mask.target_use_ultimate_joint_mask.map(() => [
      false,
      false,
    ]),
  };
  assert.deepEqual(allowedNowTargets(closed, [MAGE]), []);
});

test("on every live fixture frame Allowed Now names only bodies of that audience's scene", async () => {
  const fixture = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  let checked = 0;
  for (const raw of Object.values(fixture.presentations)) {
    const frame = await normalizeAuthorizedPresentationFrameV1(structuredClone(raw));
    if (frame.viewer_mode !== "live") continue;
    const scene = authorizedPresentationSceneView(frame);
    const inspection = authorizedPresentationInspectionState(frame).inspection;
    const sceneIds = new Set(
      scene?.agents.map((/** @type {any} */ agent) => agent.public_agent_id),
    );
    for (const row of allowedNowTargets(
      inspection?.decision_mask,
      scene?.agents ?? [],
    )) {
      assert.ok(sceneIds.has(row.publicAgentId), `${scene?.audience} ${row.label}`);
    }
    checked += 1;
  }
  assert.ok(checked >= 3);
});

test("action slots carry real names, reach, readiness, the Rune Ring and tags", () => {
  const warrior = {
    class_id: 2,
    basic_interaction_radius: 1.5,
    ultimate_interaction_radius: 5.75,
    ultimate_target_mode: "enemy",
    ultimate_cooldown_steps: 30,
  };
  const slots = actionSlotViews({
    classId: 2,
    mechanics: warrior,
    cooldown: 0,
    lane: "ultimate",
    autoArmed: false,
    basicOk: false,
    ultimateOk: true,
  });
  assert.deepEqual(
    slots.map((slot) => [slot.key, slot.name, slot.sub, slot.ready, slot.tag, slot.no]),
    [
      ["0", "No Combat", "Only Move", null, null, false],
      ["1", "Shield Bash", "Reach 1.5", null, "Not Allowed", true],
      ["2", "Charge", "Reach 5.75", "Ready", "Armed", false],
    ],
  );
  assert.equal(slots[2].lit, 12);
  const mage = actionSlotViews({
    classId: 1,
    mechanics: {
      basic_interaction_radius: 3,
      ultimate_target_mode: "target_none",
      ultimate_interaction_radius: 0,
      ultimate_cooldown_steps: 30,
    },
    cooldown: 30,
    lane: "basic",
    autoArmed: true,
    basicOk: true,
    ultimateOk: false,
  });
  assert.deepEqual(
    mage.map((slot) => [slot.sub, slot.ready, slot.tag]),
    [
      ["Only Move", null, null],
      ["Reach 3.0", null, "Auto"],
      ["No Target", "In 30 Ticks", "Not Allowed"],
    ],
  );
  assert.equal(mage[0].armed, false);
  const over = actionSlotViews({
    classId: 1,
    mechanics: {
      basic_interaction_radius: 3,
      ultimate_target_mode: "target_none",
      ultimate_interaction_radius: 0,
      ultimate_cooldown_steps: 30,
    },
    cooldown: 0,
    lane: "none",
    autoArmed: false,
    basicOk: true,
    ultimateOk: true,
    terminal: true,
  });
  assert.deepEqual(
    over.map((slot) => [slot.ready, slot.tag, slot.no]),
    [
      [null, "Armed", false],
      [null, "Not Allowed", true],
      ["Ready", "Not Allowed", true],
    ],
  );
  assert.equal(runeRingLit(16, 30), 4);
  assert.equal(runeRingLit(0, 30), 12);
  assert.equal(runeRingLit(30, 30), 0);
  assert.equal(runeRingText(0), "Ready");
  assert.equal(runeRingText(1), "1 Tick");
  assert.equal(runeRingText(-1), null);
  assert.equal(runeRingText(undefined), null);
  assert.equal(effectiveLane({ armed_lane: "basic", target_action: 0 }), "none");
  assert.equal(effectiveLane({ armed_lane: "basic", target_action: 6 }), "basic");
  assert.equal(effectiveLane({ armed_lane: "ultimate", target_action: 0 }), "ultimate");
  assert.equal(effectiveLane(null), "none");
});

test("the order sentence uses the vision's words", () => {
  const charge = {
    touched: true,
    moveAction: 0,
    lane: /** @type {const} */ ("ultimate"),
    classId: 2,
    ultimateNeedsNoTarget: false,
    targetName: "Mage (ID 5)",
  };
  assert.equal(
    orderSentence(charge),
    "Charge the Mage (ID 5), then stay where you land.",
  );
  assert.equal(
    orderSentence({ ...charge, moveAction: 1 }),
    "Charge the Mage (ID 5), then step North from where you land.",
  );
  assert.equal(
    orderSentence({ ...charge, lane: "basic", moveAction: 4 }),
    "Shield Bash on the Mage (ID 5), then step West.",
  );
  assert.equal(orderSentence({ ...charge, lane: "none" }), "Stay, with No Combat.");
  assert.equal(
    orderSentence({ ...charge, lane: "none", moveAction: 6 }),
    "Step North-West, with No Combat.",
  );
  assert.equal(
    orderSentence({ ...charge, touched: false }),
    "Stay, with No Combat (the default).",
  );
  assert.equal(
    orderSentence({
      ...charge,
      classId: 1,
      ultimateNeedsNoTarget: true,
      targetName: null,
    }),
    "Burst (no target), then stay where you are.",
  );
  assert.equal(
    orderSentence({ ...charge, fallen: { className: "Warrior", returnsIn: 3 } }),
    "None. Your Warrior is fallen and returns in 3 ticks.",
  );
});

test("turn phases and Do This Now follow real states", () => {
  const phase = (
    /** @type {unknown} */ operationState,
    lockSent = false,
    act = false,
  ) => turnPhase({ operationState, lockSent, choreographyBlocking: act });
  assert.equal(phase(undefined), "plan");
  assert.equal(phase(undefined, true), "sent");
  assert.equal(phase("thinking", true), "choosing");
  assert.equal(phase("loading"), "choosing");
  assert.equal(phase("finishing"), "act");
  assert.equal(phase(undefined, true, true), "act");
  assert.equal(
    turnPhase({
      operationState: undefined,
      lockSent: false,
      choreographyBlocking: false,
      running: true,
    }),
    "choosing",
  );
  const plan = {
    stopped: false,
    terminal: false,
    phase: /** @type {const} */ ("plan"),
    watchMode: false,
    running: false,
    focusedManual: true,
    lastManualName: "Warrior",
    fallen: null,
    targetSet: true,
    lane: /** @type {const} */ ("ultimate"),
    ultimateOkOnTarget: true,
    ultimateNeedsNoTarget: false,
    ultimateName: "Charge",
    untouchedOther: null,
  };
  assert.deepEqual(doThisNow(plan), {
    text: "Your order is ready. Press Lock In Turn. Then everyone acts at once.",
    beacon: "#submit-turn-button",
  });
  assert.equal(
    doThisNow({ ...plan, lane: "none", targetSet: false }).beacon,
    "#command-target-select .pick",
  );
  assert.deepEqual(doThisNow({ ...plan, lane: "none" }), {
    text: "Press 2 for Charge, or lock in with No Combat.",
    beacon: "#ultimate-button",
  });
  assert.equal(doThisNow({ ...plan, stopped: true }).beacon, "#stopped-save");
  assert.equal(doThisNow({ ...plan, phase: "choosing" }).beacon, null);
  assert.equal(doThisNow({ ...plan, watchMode: true }).beacon, "#live-run-button");
  assert.equal(
    doThisNow({ ...plan, focusedManual: false }).text,
    "You are looking at a policy's champion. Press Back To Your Warrior to give orders.",
  );
  assert.equal(
    doThisNow({ ...plan, untouchedOther: { publicAgentId: "5", className: "Mage" } })
      .beacon,
    '.cl-row[data-slot="5"]',
  );
  assert.match(
    doThisNow({ ...plan, fallen: { className: "Warrior", returnsIn: 2 } }).text,
    /Your Warrior is fallen\. It returns in 2 ticks\./u,
  );
});

test("the checklist line and the Policies list name only what you may know", () => {
  assert.equal(
    checklistLine({
      moveAction: 0,
      lane: "ultimate",
      classId: 2,
      ultimateNeedsNoTarget: false,
      targetShort: "Mage 5",
    }),
    "Step: Stay · Action: Charge → Mage 5",
  );
  assert.equal(
    checklistLine({
      moveAction: 3,
      lane: "none",
      classId: 1,
      ultimateNeedsNoTarget: true,
      targetShort: null,
    }),
    "Step: East · Action: No Combat",
  );
  assert.deepEqual(
    policyListRows(
      controllerRoster({
        manualSlots: [0],
        slotLabels: [
          "Manual",
          "Recurrent MAPPO",
          "Recurrent MAPPO",
          null,
          null,
          "tdm-beta",
          null,
          null,
          null,
          null,
        ],
      }),
    ).map((row) => `${row.teamId === 1 ? "A" : "B"} ${row.name} · ${row.countWords}`),
    ["A Recurrent MAPPO · 2 Champions", "B tdm-beta · 1 Champion"],
  );
});

test("the deck reads no raw HUD or match summary, and the researcher space only for own orders", async () => {
  const source = await readFile(
    new URL("../src/tactics-turn.js", import.meta.url),
    "utf8",
  );
  const code = source.replace(/\/\*[\s\S]*?\*\//gu, "").replace(/\/\/.*$/gmu, "");
  assert.doesNotMatch(code, /\.hud\b|match_summary/u);
  // One read of the researcher branch, inside servedManualOrdersOf.
  const reads = [...code.matchAll(/researcher_space/gu)];
  assert.equal(reads.length, 1);
  const start = code.indexOf("export function servedManualOrdersOf(");
  const end = code.indexOf("\nexport function ", start + 1);
  assert.ok(
    start >= 0 && (reads[0].index ?? -1) > start && (reads[0].index ?? -1) < end,
  );
  assert.match(code, /authorizedPresentationSceneView\(/u);
  assert.match(code, /authorizedPresentationInspectionState\(presentation\)/u);
});
