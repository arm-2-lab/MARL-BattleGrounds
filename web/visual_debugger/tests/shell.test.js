/**
 * @file Check the page shell's pure helpers in Node, with no page. The rune
 * wordmark carves "MARL-BGs" from the ported store geometry (cap 11 is about
 * 80 px wide; the A's crossbar and the hyphen are the only lozenges, the
 * hyphen flat so the name reads "MARL-BGs", not "MARL·BGs") and
 * refuses a bad cap height. applyProductHeader writes the official tab title,
 * the product name after the lockup, the Help heading and the Help button
 * label with its ? keycap. The place title names a built-in scenario by its
 * tale ("The Last Mending · Scenario 1"), also from the recorded scenario map
 * layout name "tdm-scenario-N-map", and a registered map by name and ID.
 * The connection pill keeps its priority order with Title Case words
 * ("Connected", not "Online"; a lost server reads "Disconnected" before
 * "Resync Required"), and autoplay's own fetch (quietBusy) keeps "Connected"
 * while a user's press shows "Command in Flight". The Replay Viewer's POV agent chip names the
 * POV agent ("Hunter · Agent 2") and its picker lists only agents a roster row
 * could switch to, Team A first, with identity only (no life, place or
 * status). Popovers open under their button and stay inside the window. The
 * Title Card waits 300 ms.
 */
import assert from "node:assert/strict";
import test from "node:test";
import {
  applyProductHeader,
  BRAND_NAME,
  connectionPill,
  matchPlaceTitle,
  OFFICIAL_TITLE,
  PRODUCT_NAMES,
  PRODUCT_TITLES,
  popoverPlacement,
  povAgentChipView,
  publicAudienceLabel,
  runeWordmark,
  SCENARIO_TALE_TITLES,
  scenarioTale,
  TITLE_CARD_DELAY_MS,
} from "../src/shell.js";

test("the rune wordmark carves MARL-BGs from the store geometry", () => {
  assert.equal(BRAND_NAME, "MARL-BGs");
  assert.equal(OFFICIAL_TITLE, "MARL-BattleGrounds");
  const mark = runeWordmark(BRAND_NAME);
  assert.equal(mark.height, 11);
  assert.ok(Math.abs(mark.width - 80.41) < 0.01, String(mark.width));
  assert.ok(Math.abs(mark.strokeWidth - 1.98) < 1e-9);
  assert.ok(Math.abs(mark.keylineWidth - 3.48) < 1e-9);
  assert.ok(mark.strokePath.startsWith("M0.99 10.01L0.99 0.99L4.95 6.16L8.91 0.99"));
  // One stroke group per carved line: M(1) A(1) R(2) L(1) B(2) G(1) s(1).
  assert.equal(mark.strokePath.match(/M/gu)?.length, 9);
  assert.equal(
    mark.lozengePath,
    "M16.61 5.34L17.77 6.99L16.61 8.64L15.45 6.99ZM46.97 4.46L49.06 5.5L46.97 6.55L44.88 5.5Z",
  );
  assert.ok(Object.isFrozen(mark));

  const single = runeWordmark("A", 10);
  assert.equal(single.strokePath, "M0.9 9.1L4.1 0.9L7.3 9.1");
  assert.equal(single.width, 8.2);
  // A letter the table lacks advances like a space and draws nothing.
  assert.equal(runeWordmark("?", 10).strokePath, "");
  assert.equal(runeWordmark("?", 10).width, 4);
  for (const cap of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(() => runeWordmark("A", cap), RangeError);
  }
});

test("applyProductHeader writes the tab title, product name and Help labels", () => {
  assert.deepEqual(PRODUCT_TITLES, {
    combat_debugger: "MARL-BattleGrounds BattleClient",
    replay_viewer: "MARL-BattleGrounds Replay Viewer",
  });
  assert.deepEqual(PRODUCT_NAMES, {
    combat_debugger: "BattleClient",
    replay_viewer: "Replay Viewer",
  });
  assert.ok(Object.isFrozen(PRODUCT_NAMES));
  const previousDocument = Reflect.get(globalThis, "document");
  const fakeDocument = { title: "MARL-BattleGrounds" };
  Reflect.set(globalThis, "document", fakeDocument);
  try {
    for (const [productKind, title, name, heading, button] of /** @type {const} */ ([
      [
        "combat_debugger",
        "MARL-BattleGrounds BattleClient",
        "BattleClient",
        "How to Play",
        "How to Play ",
      ],
      [
        "replay_viewer",
        "MARL-BattleGrounds Replay Viewer",
        "Replay Viewer",
        "Replay Viewer Help",
        "Help ",
      ],
    ])) {
      const appTitle = { textContent: "" };
      const helpHeading = { textContent: "" };
      /** @type {unknown[]} */
      let helpChildren = [];
      const helpButton = {
        dataset: /** @type {Record<string, string>} */ ({}),
        ownerDocument: {
          createElement: (/** @type {string} */ tag) => ({ tag, textContent: "" }),
        },
        replaceChildren: (/** @type {unknown[]} */ ...children) => {
          helpChildren = children;
        },
      };
      applyProductHeader(
        /** @type {any} */ ({ appTitle, helpHeading, helpButton }),
        productKind,
      );
      assert.equal(fakeDocument.title, title);
      assert.equal(appTitle.textContent, name);
      assert.equal(helpHeading.textContent, heading);
      assert.deepEqual(helpChildren, [button, { tag: "kbd", textContent: "?" }]);
      assert.equal(helpButton.dataset.product, productKind);
    }
  } finally {
    Reflect.set(globalThis, "document", previousDocument);
  }
  assert.equal(publicAudienceLabel("researcher"), "Oracle View");
  assert.equal(publicAudienceLabel("agent_pov"), "Agent POV");
  assert.equal(publicAudienceLabel(undefined), "View Unavailable");
});

test("the place title names a scenario by its tale and a map by name and ID", () => {
  assert.equal(Object.keys(SCENARIO_TALE_TITLES).length, 8);
  assert.deepEqual(scenarioTale("scenario_1_HRP_vs_MRP"), {
    title: "The Last Mending",
    number: 1,
  });
  assert.deepEqual(scenarioTale("tdm_scenario_8"), {
    title: "The Waiting Hour",
    number: 8,
  });
  assert.deepEqual(scenarioTale("scenario_5"), {
    title: "Turning the Tide",
    number: 5,
  });
  // The layout name a live scenario game and its replay record (3 Oct 2026:
  // the BattleClient showed "Custom Map" for every scenario).
  assert.deepEqual(scenarioTale("tdm-scenario-1-map"), {
    title: "The Last Mending",
    number: 1,
  });
  assert.equal(
    matchPlaceTitle({
      map: {
        map_id: null,
        technical_name: "tdm-scenario-4-map",
        display_name: "Custom Map",
      },
    }),
    "The Gambit · Scenario 4",
  );
  for (const name of [
    "scenario_9",
    "tdm-scenario-12-map",
    "my_scenario_1",
    "teutoburg_forest",
    1,
    null,
  ]) {
    assert.equal(scenarioTale(name), null, String(name));
  }
  assert.equal(
    matchPlaceTitle({
      map: {
        map_id: 47,
        technical_name: "tdm_map_id_47_teutoburg_forest_test",
        display_name: "Teutoburg Forest",
        split: "test",
      },
    }),
    "Teutoburg Forest · Map 47",
  );
  assert.equal(
    matchPlaceTitle({
      map: {
        map_id: null,
        technical_name: "scenario_1_HRP_vs_MRP",
        display_name: "Map name unavailable",
        split: null,
      },
    }),
    "The Last Mending · Scenario 1",
  );
  assert.equal(
    matchPlaceTitle({
      map: {
        map_id: null,
        technical_name: "x",
        display_name: "Custom Map",
        split: null,
      },
    }),
    "Custom Map",
  );
  for (const summary of [null, {}, { map: null }, { map: { display_name: "" } }]) {
    assert.equal(matchPlaceTitle(summary), null);
  }
});

test("the connection pill keeps its priority order in Title Case", () => {
  const calm = {
    busy: false,
    offline: false,
    resyncRequired: false,
    shuttingDown: false,
    hasFrame: true,
  };
  assert.deepEqual(connectionPill(calm, null), {
    label: "Connected",
    status: "online",
  });
  assert.deepEqual(connectionPill({ ...calm, hasFrame: false }, null), {
    label: "Connecting",
    status: "loading",
  });
  assert.deepEqual(connectionPill({ ...calm, offline: true }, null), {
    label: "Disconnected",
    status: "offline",
  });
  assert.deepEqual(
    connectionPill({ ...calm, offline: true, resyncRequired: true }, null),
    {
      label: "Disconnected",
      status: "offline",
    },
  );
  assert.deepEqual(connectionPill({ ...calm, resyncRequired: true }, null), {
    label: "Resync Required",
    status: "offline",
  });
  const busy = { ...calm, busy: true, offline: true };
  assert.equal(connectionPill(busy, null).label, "Command in Flight");
  // Autoplay's own next-frame fetch keeps the pill as it was: no blink to
  // "Command in Flight" before every played tick; a user press still shows it.
  assert.deepEqual(connectionPill({ ...calm, busy: true, quietBusy: true }, null), {
    label: "Connected",
    status: "online",
  });
  assert.equal(
    connectionPill({ ...calm, busy: true, quietBusy: false }, null).label,
    "Command in Flight",
  );
  assert.equal(
    connectionPill(busy, { deferredDrafts: [{}], deferredSubmit: null }).label,
    "Input Queued",
  );
  assert.equal(
    connectionPill(busy, { deferredDrafts: [], deferredSubmit: null }).label,
    "Command in Flight",
  );
  assert.equal(
    connectionPill(busy, { deferredDrafts: [{}], deferredSubmit: {} }).label,
    "Submit Queued",
  );
  assert.deepEqual(connectionPill({ ...busy, shuttingDown: true }, null), {
    label: "Closed",
    status: "offline",
  });
});

test("the POV agent chip names the POV agent and picks only switchable agents", () => {
  /** @param {string} id @param {number} classId @param {number} teamId @param {string} kind @param {number | null} slot */
  const row = (id, classId, teamId, kind, slot) => ({
    public_agent_id: id,
    command_global_slot: slot,
    activation_kind: kind,
    agent: {
      presentation_key: `key-${id}`,
      public_agent_id: id,
      class_id: classId,
      team_id: teamId,
      health: 31,
      position: [3, 4],
    },
  });
  const rows = [
    row("6", 2, 2, "replay_pov_global", 5),
    row("2", 3, 1, "replay_pov_global", 1),
    row("1", 1, 1, "replay_pov_global", 0),
    row("9", 5, 2, "scene_agent", null),
  ];
  const view = povAgentChipView("2", rows);
  assert.ok(view);
  assert.equal(view.label, "Hunter · Agent 2");
  assert.equal(view.classLetter, "H");
  assert.equal(view.accent, "hunter");
  assert.equal(view.teamId, 1);
  assert.deepEqual(
    view.teams.map((team) => team.map((choice) => [choice.label, choice.current])),
    [
      [
        ["Hunter · Agent 2", true],
        ["Mage · Agent 1", false],
      ],
      [["Warrior · Agent 6", false]],
    ],
  );
  assert.deepEqual(view.teams[1][0].command, {
    command_type: "activate_replay_pov_agent",
    global_slot: 5,
    public_agent_id: "6",
  });
  // Identity only: a choice never carries life, place or statuses.
  for (const choice of view.teams.flat()) {
    assert.deepEqual(Object.keys(choice).sort(), [
      "accent",
      "classLetter",
      "command",
      "current",
      "label",
      "publicAgentId",
      "teamId",
    ]);
  }
  assert.ok(Object.isFrozen(view) && Object.isFrozen(view.teams[0]));
  assert.equal(povAgentChipView("4", rows), null);
  assert.equal(povAgentChipView(null, rows), null);
  assert.equal(povAgentChipView("2", []), null);
});

test("popovers open under their button and stay inside the window", () => {
  assert.deepEqual(popoverPlacement({ left: 100, right: 180, bottom: 40 }, 300, 1440), {
    top: 46,
    left: 100,
  });
  assert.deepEqual(
    popoverPlacement({ left: 1300, right: 1380, bottom: 40 }, 300, 1440),
    {
      top: 46,
      left: 1080,
    },
  );
  assert.deepEqual(popoverPlacement({ left: 2, right: 40, bottom: 40.4 }, 300, 960), {
    top: 46,
    left: 16,
  });
  assert.deepEqual(popoverPlacement({ left: 700, right: 760, bottom: 40 }, 900, 960), {
    top: 46,
    left: 16,
  });
  assert.equal(TITLE_CARD_DELAY_MS, 300);
});
