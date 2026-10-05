/**
 * @file Check the one owner of controller names (src/controller-names.js) and
 * the surfaces that share its model.
 *
 * Contracts checked:
 * - The model: "You" first, then each policy once in the order of its first
 *   champion's slot, its champions in slot order with "Class · ID n"; the
 *   Replay Viewer's team names play every champion of their team; malformed
 *   input gives empty teams.
 * - Fit: a painted line shows its full names when they fit in its measured
 *   width, else the short line; every name in the full line sits in a
 *   span.controller-name.
 * - Collapse words: "You + 4 Policies", "4 Policies" with no Manual slot,
 *   "1 Policy" and "You + 1 Policy" singular, "You" alone.
 * - Cut with "…": CSS cuts span.controller-name past --controller-name-max
 *   with text-overflow: ellipsis; controllerNameHtml escapes the name and
 *   marks it for its team's list; a "played by <name>." sentence
 *   (paintNamedSentence, the deck's status line) puts the name and its period
 *   in one cut span.
 * - Hover list content: "Team A Controllers", each controller's full name
 *   (never cut), "N Champion(s)" and its champions; the panel stays inside
 *   the window at 1280, 1440 and 1920 px wide and at a narrow width.
 * - Consistency: the scorebug's lines (liveScorebugModel in looks/hud.js),
 *   both hover lists and the rail's Policies list (policyListRows) come from
 *   the same model: on every live fixture frame, the HUD's roster equals
 *   controllerRosterFromFrame (match-summary.js) for the same facts, and the
 *   rail rows are exactly the list rows without "You".
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  championCountWords,
  collapsedControllerLine,
  controllerLineChoice,
  controllerLineParts,
  controllerListModel,
  controllerNameHtml,
  controllerOfChampion,
  controllerRoster,
  fitControllerLine,
  paintNamedSentence,
  publishControllerRoster,
  NO_CONTROLLER_ROSTER,
  paintControllerLine,
  placeControllerList,
  policyListRows,
} from "../src/controller-names.js";
import { liveControllers, liveScorebugModel, records } from "../src/looks/hud.js";
import {
  controllerRosterFromFrame,
  hudSourcesFromFrame,
} from "../src/match-summary.js";

const LONG = Object.freeze({
  mappo: "Recurrent MAPPO · Pool Seed 12259871 Best",
  qmix: "QMIX Mixer Ablation Without Agent IDs",
  llm: "My LLM Planner (Qwen3.5-4B, Every 4 Ticks)",
});

const STRESS = Object.freeze({
  manual_slots: [0],
  slot_labels: [
    "Manual",
    LONG.mappo,
    LONG.qmix,
    LONG.llm,
    "tdm-beta",
    LONG.mappo,
    "tdm-alpha",
    "tdm-alpha",
    "random",
    "tdm-beta",
  ],
});

const CHAMPIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((slot) => ({
  publicAgentId: `agent-slot-${slot}`,
  teamId: slot < 5 ? 1 : 2,
  teamLocalSlot: slot % 5,
  classId: (slot % 5) + 1,
}));

const roster = controllerRoster({
  manualSlots: STRESS.manual_slots,
  slotLabels: STRESS.slot_labels,
  champions: CHAMPIONS,
});

test("the model lists You first, then each policy once in slot order", () => {
  const [a, b] = roster.teams;
  assert.deepEqual(controllerLineParts(a), [
    "You",
    LONG.mappo,
    LONG.qmix,
    LONG.llm,
    "tdm-beta",
  ]);
  assert.deepEqual(controllerLineParts(b), [
    LONG.mappo,
    "tdm-alpha",
    "random",
    "tdm-beta",
  ]);
  assert.deepEqual(
    b.controllers[1].champions.map((c) => `${c.className} · ID ${c.displayId}`),
    ["Warrior · ID 6", "Hunter · ID 7"],
  );
  assert.equal(a.controllers[0].you, true);
  assert.equal(
    controllerOfChampion(roster, { publicAgentId: "agent-slot-8" })?.controller.name,
    "random",
  );
  assert.equal(controllerOfChampion(roster, { slot: 0 })?.controller.you, true);
  const replay = controllerRoster({
    teamNames: ["Team: Manual / Recurrent MAPPO", "tdm-beta"],
    champions: CHAMPIONS,
  });
  assert.deepEqual(
    replay.teams.map((team) =>
      team.controllers.map((c) => [c.name, c.champions.length]),
    ),
    [[["Team: Manual / Recurrent MAPPO", 5]], [["tdm-beta", 5]]],
  );
  assert.deepEqual(
    controllerRoster(/** @type {any} */ ({ manualSlots: "1", slotLabels: 3 })),
    NO_CONTROLLER_ROSTER,
  );
  // No identities yet: every named slot counts as one champion of unknown class.
  const bare = controllerRoster({ manualSlots: [5], slotLabels: STRESS.slot_labels });
  assert.deepEqual(controllerLineParts(bare.teams[1]), [
    "You",
    "tdm-alpha",
    "random",
    "tdm-beta",
  ]);
  assert.equal(bare.teams[1].controllers[0].champions[0].className, "Champion");
});

test("collapse words: You + N Policies, N Policies, singular and You alone", () => {
  assert.equal(collapsedControllerLine(roster.teams[0]), "You + 4 Policies");
  assert.equal(collapsedControllerLine(roster.teams[1]), "4 Policies");
  const one = controllerRoster({
    manualSlots: [0],
    slotLabels: [
      "Manual",
      LONG.llm,
      LONG.llm,
      LONG.llm,
      LONG.llm,
      ...Array(5).fill("tdm-beta"),
    ],
  });
  assert.equal(collapsedControllerLine(one.teams[0]), "You + 1 Policy");
  assert.equal(collapsedControllerLine(one.teams[1]), "1 Policy");
  const you = controllerRoster({
    manualSlots: [0, 1],
    slotLabels: ["Manual", "Manual"],
  });
  assert.equal(collapsedControllerLine(you.teams[0]), "You");
  assert.equal(collapsedControllerLine(NO_CONTROLLER_ROSTER.teams[0]), "");
  assert.deepEqual(controllerLineChoice(roster.teams[0], true), {
    mode: "full",
    text: `You + ${LONG.mappo} + ${LONG.qmix} + ${LONG.llm} + tdm-beta`,
  });
  assert.deepEqual(controllerLineChoice(roster.teams[0], false), {
    mode: "short",
    text: "You + 4 Policies",
  });
  assert.deepEqual(controllerLineChoice(you.teams[0], false), {
    mode: "full",
    text: "You",
  });
  assert.equal(championCountWords(1), "1 Champion");
  assert.equal(championCountWords(4), "4 Champions");
});

// A line stand-in: 7 px per character, a fixed client width.
/** @param {number} clientWidth */
function fakeLine(clientWidth) {
  /** @type {any[]} */
  let children = [];
  const attributes = new Map();
  const doc = {
    createElement: () => {
      /** @type {any} */
      const node = { className: "", dataset: {}, textContent: "" };
      return node;
    },
  };
  /** @type {any} */
  const line = {
    ownerDocument: doc,
    parentElement: null,
    isConnected: true,
    dataset: {},
    classList: { add: () => {} },
    clientWidth,
    get scrollWidth() {
      return 7 * line.text.length;
    },
    get text() {
      return children
        .map((child) => (typeof child === "string" ? child : child.textContent))
        .join("");
    },
    get children() {
      return children;
    },
    /** @param {...any} nodes */
    replaceChildren(...nodes) {
      children = nodes;
    },
    /** @param {string} name @param {string} value */
    setAttribute(name, value) {
      attributes.set(name, value);
    },
    /** @param {string} name */
    removeAttribute(name) {
      attributes.delete(name);
    },
    /** @param {string} name */
    getAttribute(name) {
      return attributes.get(name) ?? null;
    },
  };
  return line;
}

test("fit: the full line when it fits its measured width, else the short line", () => {
  const short = controllerRoster({
    manualSlots: [0],
    slotLabels: ["Manual", "Recurrent MAPPO", "Recurrent MAPPO", null, null],
  }).teams[0];
  const wide = fakeLine(400);
  paintControllerLine(wide, short);
  assert.equal(wide.dataset.controllerLine, "full");
  assert.equal(wide.text, "You + Recurrent MAPPO");
  assert.equal(
    wide.getAttribute("aria-label"),
    "Team A Controllers: You + Recurrent MAPPO",
  );
  assert.equal(wide.dataset.controllerTeam, "1");
  assert.equal(wide.dataset.controllerList, "line");
  // The name sits in span.controller-name, which CSS cuts with "…".
  assert.equal(wide.children[2].className, "controller-name");
  const narrow = fakeLine(120);
  paintControllerLine(narrow, roster.teams[0]);
  assert.equal(narrow.dataset.controllerLine, "short");
  assert.equal(narrow.text, "You + 4 Policies");
  // The full words stay the accessible name.
  assert.match(
    narrow.getAttribute("aria-label") ?? "",
    /QMIX Mixer Ablation Without Agent IDs/u,
  );
  narrow.clientWidth = 9999;
  fitControllerLine(narrow);
  assert.equal(narrow.dataset.controllerLine, "full");
  const empty = fakeLine(100);
  paintControllerLine(empty, NO_CONTROLLER_ROSTER.teams[1]);
  assert.equal(empty.text, "");
  assert.equal(empty.dataset.controllerTeam, undefined);
});

test("cut with an ellipsis: the CSS rule and the escaped, marked name", async () => {
  const css = await readFile(
    new URL("../src/styles/controller-names.css", import.meta.url),
    "utf8",
  );
  const rule = /\.controller-name \{([^}]*)\}/u.exec(css)?.[1] ?? "";
  assert.match(rule, /text-overflow: ellipsis/u);
  assert.match(rule, /white-space: nowrap/u);
  assert.match(rule, /max-width: min\(100%, var\(--controller-name-max\)\)/u);
  assert.match(css, /--controller-name-max: \d+px/u);
  const styles = await readFile(new URL("../styles.css", import.meta.url), "utf8");
  assert.match(styles, /@import url\("\/src\/styles\/controller-names\.css"\);/u);
  assert.equal(
    controllerNameHtml('A <b> & "C"', 2),
    '<span class="controller-name" data-controller-team="2" data-controller-list="name">A &lt;b&gt; &amp; &quot;C&quot;</span>',
  );
  assert.equal(
    controllerNameHtml("tdm-beta", null),
    '<span class="controller-name">tdm-beta</span>',
  );
  assert.equal(
    controllerNameHtml("tdm-beta", 1, "."),
    '<span class="controller-name" data-controller-team="1" data-controller-list="name">tdm-beta.</span>',
  );
});

test("a played-by sentence cuts the named controller and keeps its period inside", () => {
  /** @type {any} */
  const doc = {
    createElement: () => {
      /** @type {any[]} */
      const children = [];
      return {
        className: "",
        dataset: {},
        children,
        textContent: "",
        /** @param {...any} nodes */
        append(...nodes) {
          children.push(...nodes);
        },
      };
    },
  };
  publishControllerRoster(doc, roster);
  /** @type {any} */
  const msg = {
    ownerDocument: doc,
    textContent: "",
    querySelector: () => null,
    /** @type {any[]} */
    children: [],
    /** @param {...any} nodes */
    replaceChildren(...nodes) {
      msg.children = nodes;
    },
  };
  paintNamedSentence(msg, `Looking at the Hunter (ID 2), played by ${LONG.qmix}.`);
  const run = msg.children[0];
  assert.equal(run.children[0], "Looking at the Hunter (ID 2), played by ");
  assert.equal(run.children[1].className, "controller-name");
  assert.equal(run.children[1].textContent, `${LONG.qmix}.`);
  assert.equal(run.children[1].dataset.controllerTeam, "1");
  assert.equal(run.children[2], "");
  // No known controller after "played by": plain text.
  paintNamedSentence(msg, "Looking at the Mage (ID 5), played by a policy.");
  assert.equal(msg.textContent, "Looking at the Mage (ID 5), played by a policy.");
});

test("the hover list: title, full names, counts and champions", () => {
  const model = controllerListModel(roster.teams[0]);
  assert.equal(model.title, "Team A Controllers");
  assert.deepEqual(
    model.rows.map((row) => [
      row.name,
      row.you,
      row.countWords,
      row.champions.map((c) => c.text),
    ]),
    [
      ["You", true, "1 Champion", ["Mage · ID 0"]],
      [LONG.mappo, false, "1 Champion", ["Warrior · ID 1"]],
      [LONG.qmix, false, "1 Champion", ["Hunter · ID 2"]],
      [LONG.llm, false, "1 Champion", ["Rogue · ID 3"]],
      ["tdm-beta", false, "1 Champion", ["Priest · ID 4"]],
    ],
  );
  assert.equal(controllerListModel(roster.teams[1]).title, "Team B Controllers");
  assert.equal(controllerListModel(roster.teams[1]).rows[1].countWords, "2 Champions");
});

test("the list panel stays inside the window at every width", () => {
  for (const [width, height] of [
    [1280, 800],
    [1440, 900],
    [1920, 1080],
    [320, 480],
  ]) {
    for (const anchor of [
      { left: width - 140, right: width - 30, top: 80, bottom: 98 },
      { left: 20, right: 160, top: 80, bottom: 98 },
      { left: width - 60, right: width + 40, top: height - 30, bottom: height - 12 },
    ]) {
      /** @type {any} */
      const panel = { style: {}, scrollHeight: 260 };
      placeControllerList(panel, anchor, { width, height });
      const left = Number.parseFloat(panel.style.left);
      const panelWidth = Number.parseFloat(panel.style.width);
      assert.ok(
        left >= 8 && left + panelWidth <= width - 8,
        `${width} ${JSON.stringify(anchor)}`,
      );
      if (panel.style.top !== "") {
        const top = Number.parseFloat(panel.style.top);
        assert.ok(top + Number.parseFloat(panel.style.maxHeight) <= height - 8);
      } else {
        const bottom = Number.parseFloat(panel.style.bottom);
        assert.ok(height - bottom - Number.parseFloat(panel.style.maxHeight) >= 8);
      }
    }
  }
});

test("one model: the scorebug, the hover lists and the rail agree on every live fixture frame", async () => {
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
    const sources = hudSourcesFromFrame(frame);
    if (sources === null) continue;
    const hud = liveScorebugModel(
      records({ sources }),
      sources,
      liveControllers(STRESS),
    );
    const shared = controllerRosterFromFrame(frame, STRESS);
    assert.deepEqual(hud.roster, shared);
    assert.deepEqual(
      hud.lines,
      shared.teams.map((team) => controllerLineParts(team).join(" + ")),
    );
    const listRows = shared.teams.flatMap((team) =>
      controllerListModel(team)
        .rows.filter((row) => !row.you)
        .map((row) => [team.teamId, row.name, row.countWords]),
    );
    assert.deepEqual(
      policyListRows(shared).map((row) => [row.teamId, row.name, row.countWords]),
      listRows,
    );
    // The identity directory gives every champion its class.
    assert.ok(
      shared.teams[0].controllers[0].champions.every((c) => c.classId !== null),
    );
    checked += 1;
  }
  assert.ok(checked >= 3);
});
