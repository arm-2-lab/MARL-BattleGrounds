/**
 * @file Check that Tab, Shift+Tab, ‹ Prev and Next › give the orders only to
 * your champions that can take orders (owner's ruling, 3 October 2026):
 * - nextOrderSlot (controls.js) walks the living Manual slots in slot order,
 *   wraps around, skips fallen ones, starts from a policy champion that is
 *   looked at (Watch Only), stays on the only ready champion, and returns
 *   null when every Manual champion is fallen or none exists.
 * - statusSentence says "All your champions are fallen, ..." for that case.
 * - The help line for Tab in the one command set says it is your next or
 *   previous champion (Policies Only: every champion in sight).
 * - main.js ownChampionTabStep turns a live Tab into a roster_selection
 *   (role "control") for that slot, reads the looked-at champion from
 *   focused_public_agent_id or the audience's inspection, treats a Manual
 *   champion the audience's scene draws as not alive as fallen, keeps a
 *   Manual champion the scene does not draw as ready, and passes every other
 *   command, Policies Only, the Replay Viewer and scripted inspection to
 *   Python unchanged; dispatchCommand asks it before any other check that
 *   reads the key, and keeps a queue behind a Tab that sends nothing.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { commandHelpRows } from "../src/command-set.js";
import { nextOrderSlot, statusSentence } from "../src/controls.js";

const MAIN = readFileSync(new URL("../src/main.js", import.meta.url), "utf8");
const STEP_SOURCE = /\nfunction ownChampionTabStep\([^)]*\) \{[\s\S]*?\n\}/u.exec(
  MAIN,
)?.[0];

const FIVE = [0, 1, 2, 3, 4];

test("Tab walks the Manual slots in order and wraps around", () => {
  assert.equal(nextOrderSlot({ manualSlots: FIVE, currentSlot: 0, direction: 1 }), 1);
  assert.equal(nextOrderSlot({ manualSlots: FIVE, currentSlot: 4, direction: 1 }), 0);
  assert.equal(nextOrderSlot({ manualSlots: FIVE, currentSlot: 0, direction: -1 }), 4);
  assert.equal(nextOrderSlot({ manualSlots: FIVE, currentSlot: 3, direction: -1 }), 2);
  assert.equal(
    nextOrderSlot({ manualSlots: [4, 2, 0, 3, 1], currentSlot: 1, direction: 1 }),
    2,
    "the order of manualSlots does not matter",
  );
});

test("Tab skips fallen champions and never lands on a policy champion", () => {
  assert.equal(
    nextOrderSlot({
      manualSlots: FIVE,
      currentSlot: 1,
      fallenSlots: [2],
      direction: 1,
    }),
    3,
  );
  assert.equal(
    nextOrderSlot({
      manualSlots: FIVE,
      currentSlot: 3,
      fallenSlots: [2, 1],
      direction: -1,
    }),
    0,
  );
  // From an enemy looked at in Watch Only (slot 7), Tab goes to the first
  // ready slot after it (wrapping to 0) and Shift+Tab to the last before it.
  assert.equal(nextOrderSlot({ manualSlots: FIVE, currentSlot: 7, direction: 1 }), 0);
  assert.equal(nextOrderSlot({ manualSlots: FIVE, currentSlot: 7, direction: -1 }), 4);
  for (const currentSlot of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) {
    for (const direction of /** @type {const} */ ([1, -1])) {
      const next = nextOrderSlot({ manualSlots: FIVE, currentSlot, direction });
      assert.ok(next !== null && FIVE.includes(next), `${currentSlot} ${direction}`);
    }
  }
});

test("one Manual champion on each team: Tab moves between the two", () => {
  assert.equal(nextOrderSlot({ manualSlots: [1, 5], currentSlot: 1, direction: 1 }), 5);
  assert.equal(nextOrderSlot({ manualSlots: [1, 5], currentSlot: 5, direction: 1 }), 1);
  assert.equal(
    nextOrderSlot({ manualSlots: [1, 5], currentSlot: 5, direction: -1 }),
    1,
  );
  assert.equal(
    nextOrderSlot({ manualSlots: [1, 5], currentSlot: 3, direction: 1 }),
    5,
    "from a teammate policy champion the next Manual slot is on the other team",
  );
});

test("the only ready champion stays; none ready gives null", () => {
  assert.equal(
    nextOrderSlot({
      manualSlots: FIVE,
      currentSlot: 2,
      fallenSlots: [0, 1, 3, 4],
      direction: 1,
    }),
    2,
  );
  assert.equal(nextOrderSlot({ manualSlots: [1], currentSlot: 1, direction: -1 }), 1);
  assert.equal(
    nextOrderSlot({
      manualSlots: FIVE,
      currentSlot: 2,
      fallenSlots: FIVE,
      direction: 1,
    }),
    null,
  );
  assert.equal(nextOrderSlot({ manualSlots: [], currentSlot: 2, direction: 1 }), null);
  assert.equal(
    nextOrderSlot({
      manualSlots: [1, "2", 2.5, null],
      currentSlot: null,
      direction: 1,
    }),
    1,
    "non-integers are ignored and an unknown current slot starts at the first",
  );
  assert.equal(
    nextOrderSlot({ manualSlots: FIVE, currentSlot: null, direction: -1 }),
    4,
  );
});

test("the status line and the key help say what Tab does", () => {
  assert.equal(
    statusSentence({ kind: "nobody_ready" }),
    "All your champions are fallen, so nobody can take orders until they return.",
  );
  const row = commandHelpRows("combat_debugger").find(
    ({ id }) => id === "next_champion",
  );
  assert.equal(
    row?.help,
    "Your next / previous champion (Policies Only: every champion in sight)",
  );
});

/**
 * @param {{
 *   manualSlots?: unknown[],
 *   sceneAgents?: Array<Record<string, unknown>>,
 *   focused?: string,
 *   actor?: string | null,
 *   replay?: boolean,
 *   scripted?: boolean,
 * }} [setup]
 * @returns {(command: Record<string, unknown>) => any}
 */
function tabStep({
  manualSlots = FIVE,
  sceneAgents,
  focused,
  actor = "0",
  replay = false,
  scripted = false,
} = {}) {
  assert.ok(STEP_SOURCE, "main.js defines ownChampionTabStep");
  const agents =
    sceneAgents ??
    FIVE.map((slot) => ({
      public_agent_id: String(slot),
      life_state: "alive",
      class_name: ["Mage", "Warrior", "Hunter", "Rogue", "Priest"][slot],
    }));
  const rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((slot) => ({
    public_agent_id: String(slot),
    command_global_slot: slot,
  }));
  const frame = {
    combat_configuration: { manual_slots: manualSlots },
    ...(focused === undefined ? {} : { focused_public_agent_id: focused }),
  };
  /** @param {unknown} value */
  const isRecord = (value) =>
    typeof value === "object" && value !== null && !Array.isArray(value);
  return new Function(
    "isReplayMode",
    "liveScriptedInspectionOnly",
    "isAuthorizedPresentationFrame",
    "isRecord",
    "asArray",
    "authorizedPresentationIdentityRows",
    "authorizedPresentationSceneView",
    "authorizedPresentationInspectionState",
    "nextOrderSlot",
    "state",
    `${STEP_SOURCE}\nreturn ownChampionTabStep;`,
  )(
    () => replay,
    () => scripted,
    () => true,
    isRecord,
    (/** @type {unknown} */ value) => (Array.isArray(value) ? value : []),
    () => rows,
    () => ({ agents }),
    () => ({ inspection: actor === null ? null : { actor_public_agent_id: actor } }),
    nextOrderSlot,
    { frame, presentation: {} },
  );
}

/** @param {boolean} [shift] */
const tab = (shift = false) => ({
  command_type: "keyboard",
  key: "Tab",
  shift_key: shift,
  ctrl_key: false,
  alt_key: false,
  meta_key: false,
  repeat: false,
});

test("main.js turns a live Tab into a focus change to your next champion", () => {
  assert.deepEqual(tabStep({ actor: "0" })(tab()), {
    kind: "send",
    command: { command_type: "roster_selection", role: "control", global_slot: 1 },
  });
  assert.deepEqual(tabStep({ actor: "0" })(tab(true)).command.global_slot, 4);
  // Watch Only in Agent POV: the looked-at enemy comes from the frame.
  assert.equal(tabStep({ actor: "4", focused: "6" })(tab()).command.global_slot, 0);
  // Oracle View Watch Only: the inspection is about the looked-at champion.
  assert.equal(tabStep({ actor: "8" })(tab(true)).command.global_slot, 4);
});

test("main.js skips champions its scene draws as fallen and keeps unseen ones", () => {
  const agents = [
    { public_agent_id: "0", life_state: "alive", class_name: "Mage" },
    { public_agent_id: "1", life_state: "corpse", class_name: "Warrior" },
    { public_agent_id: "2", life_state: "alive", class_name: "Hunter" },
  ];
  assert.equal(
    tabStep({ sceneAgents: agents, actor: "0" })(tab()).command.global_slot,
    2,
  );
  // One Manual champion on each team: Team B's Mage (slot 5) is not drawn by
  // Team A's Agent POV, so the page has no fact that it fell and keeps it.
  assert.equal(
    tabStep({ manualSlots: [1, 5], sceneAgents: agents, actor: "1" })(tab()).command
      .global_slot,
    5,
  );
  assert.deepEqual(
    tabStep({ manualSlots: [0, 1], sceneAgents: agents, actor: "0" })(tab()),
    {
      kind: "stay",
      className: "Mage",
    },
  );
  assert.deepEqual(
    tabStep({ manualSlots: [1], sceneAgents: agents, actor: "1" })(tab()),
    {
      kind: "none",
    },
  );
});

test("main.js leaves other commands and other modes to Python", () => {
  assert.deepEqual(tabStep()({ command_type: "keyboard", key: "w" }), { kind: "pass" });
  assert.deepEqual(tabStep()({ ...tab(), ctrl_key: true }), { kind: "pass" });
  assert.deepEqual(tabStep({ manualSlots: [] })(tab()), { kind: "pass" });
  assert.deepEqual(tabStep({ replay: true })(tab()), { kind: "pass" });
  assert.deepEqual(tabStep({ scripted: true })(tab()), { kind: "pass" });
});

test("dispatchCommand asks ownChampionTabStep and keeps a queue behind it", () => {
  const start = MAIN.indexOf("async function dispatchCommand(");
  const end = MAIN.indexOf("const beforeFacts = ownOrderFacts(", start);
  const head = MAIN.slice(start, end);
  const asked = head.indexOf("const tabStep = ownChampionTabStep(command);");
  assert.ok(asked > 0, "dispatchCommand asks for the Tab step");
  assert.ok(
    asked > head.indexOf("if (!state.frame || !state.presentation)"),
    "after the frame is known",
  );
  assert.ok(
    asked <
      head.indexOf(
        'authorizedPresentationAudience(state.presentation) === "agent_pov"',
      ),
    "before any other check reads the key",
  );
  assert.match(head, /return dispatchCommand\(tabStep\.command, \{/u);
  assert.match(head, /return dispatchCommand\(deferredDrafts\[0\], \{/u);
  assert.match(
    head,
    /return dispatchCommand\(deferredSubmit, \{ source: "field" \}\);/u,
  );
});
