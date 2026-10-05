/**
 * @file Check Scribe's Ledger (src/ledger.js) and the ledger look of the metric
 * rows (src/metrics-panel.js): subject chips come only from the catalog's own
 * IDs and never change which rows, labels, units, values, tooltips or CSV
 * names appear (the before-and-after inventory stays identical, with or
 * without chips); with no agent list every subject is plain host text with no
 * chip; the rail builds
 * six tabs (Roster, Agent, Metrics, Actions, Visuals, Frame), each with its
 * registered hover help, moves today's
 * cards into their panels by ID (the roster into Roster, the Champion Frame
 * into Agent while the long agent details card stays out of the rail, the
 * Technical Frame and the Replay File facts into Frame), keeps live regions
 * out of hidden tabs, opens the Metrics card only while its tab is shown and
 * moves with Left, Right, Home and End; the Scope and View switches mirror the
 * native selects and fire their change events; the ledger header (Scope, Topic,
 * View and Find A Measurement) sits just before the Metrics card as its own
 * block, so only the card under it scrolls; the Technical Frame sorts its
 * facts into This Frame, This Game and This File with Copy on ids and cuts
 * long ids in the middle; the Full Record chip appears only in Agent POV,
 * names the POV agent with the same words as panels.js fullRecordLabel (never
 * the raw public ID), and outside Agent POV is left with no text, help text or
 * tooltip registration, exactly as a fresh render; the Match Summary gives the verdict on the final frame
 * and never shows kills or deaths from the ending before it.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  fullRecordChip,
  LEDGER_TAB_CONTENT,
  LEDGER_TABS,
  ledgerMatchSummary,
  middleTruncate,
  mountLedgerRail,
  renderTechnicalFrame,
  technicalFrameGroups,
} from "../src/ledger.js";
import {
  METRIC_SECTION_TITLES,
  metricSubjectParts,
  metricValueText,
  renderChoiceSwitch,
  renderMetricRows,
} from "../src/metrics-panel.js";
import { fullRecordLabel } from "../src/panels.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @returns {any} */
function fakeDocument() {
  /** @type {any} */
  const doc = {};
  /** @param {string} tag */
  function node(tag) {
    const attributes = new Map();
    /** @type {Map<string, Function[]>} */
    const listeners = new Map();
    const classes = new Set();
    /** @type {any} */
    const element = {
      ownerDocument: doc,
      tagName: tag.toUpperCase(),
      children: /** @type {any[]} */ ([]),
      parentElement: /** @type {any} */ (null),
      dataset: /** @type {Record<string, string>} */ ({}),
      hidden: false,
      open: false,
      disabled: false,
      value: "",
      _text: "",
      get id() {
        return attributes.get("id") ?? "";
      },
      set id(value) {
        attributes.set("id", String(value));
      },
      get className() {
        return [...classes].join(" ");
      },
      set className(value) {
        classes.clear();
        for (const name of String(value).split(/\s+/u).filter(Boolean))
          classes.add(name);
      },
      classList: {
        add: (/** @type {string} */ name) => classes.add(name),
        contains: (/** @type {string} */ name) => classes.has(name),
      },
      get textContent() {
        return (
          this._text +
          this.children.map((/** @type {any} */ child) => child.textContent).join("")
        );
      },
      set textContent(value) {
        this._text = String(value);
        for (const child of this.children) child.parentElement = null;
        this.children = [];
      },
      get tabIndex() {
        return Number(attributes.get("tabindex") ?? -1);
      },
      set tabIndex(value) {
        attributes.set("tabindex", String(value));
      },
      get nextElementSibling() {
        const siblings = this.parentElement?.children ?? [];
        return siblings[siblings.indexOf(this) + 1] ?? null;
      },
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) => {
        if (name === "class") element.className = String(value);
        else attributes.set(name, String(value));
      },
      getAttribute: (/** @type {string} */ name) =>
        name === "class" ? element.className : (attributes.get(name) ?? null),
      hasAttribute: (/** @type {string} */ name) => attributes.has(name),
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 1, height: 1 }),
      addEventListener(/** @type {string} */ type, /** @type {Function} */ listener) {
        listeners.set(type, [...(listeners.get(type) ?? []), listener]);
      },
      dispatchEvent(/** @type {{type: string}} */ event) {
        for (const listener of listeners.get(event.type) ?? []) listener(event);
        return true;
      },
      click() {
        if (!this.disabled) this.dispatchEvent({ type: "click" });
      },
      focus() {
        doc.activeElement = this;
      },
      remove() {
        if (!this.parentElement) return;
        const siblings = this.parentElement.children;
        siblings.splice(siblings.indexOf(this), 1);
        this.parentElement = null;
      },
      append(/** @type {any[]} */ ...items) {
        for (const item of items) {
          item.remove();
          item.parentElement = this;
          this.children.push(item);
        }
      },
      prepend(/** @type {any[]} */ ...items) {
        for (const item of [...items].reverse()) {
          item.remove();
          item.parentElement = this;
          this.children.unshift(item);
        }
      },
      after(/** @type {any} */ item) {
        item.remove();
        const siblings = this.parentElement.children;
        item.parentElement = this.parentElement;
        siblings.splice(siblings.indexOf(this) + 1, 0, item);
      },
      before(/** @type {any} */ item) {
        item.remove();
        const siblings = this.parentElement.children;
        item.parentElement = this.parentElement;
        siblings.splice(siblings.indexOf(this), 0, item);
      },
      replaceChildren(/** @type {any[]} */ ...items) {
        this.textContent = "";
        this.append(...items);
      },
      querySelector(/** @type {string} */ selector) {
        return this.querySelectorAll(selector)[0] ?? null;
      },
      querySelectorAll(/** @type {string} */ selector) {
        return all(this).filter(
          (/** @type {any} */ item) => item !== this && matches(item, selector),
        );
      },
    };
    return element;
  }
  /** @param {any} root @returns {any[]} */
  const all = (root) => [root, ...root.children.flatMap(all)];
  /** @param {any} item @param {string} selector */
  function matches(item, selector) {
    const forLabel = /^label\[for="(.+)"\]$/u.exec(selector);
    if (forLabel)
      return item.tagName === "LABEL" && item.getAttribute("for") === forLabel[1];
    if (selector === ".sr-only[aria-live]") {
      return item.classList.contains("sr-only") && item.hasAttribute("aria-live");
    }
    throw new Error(`fake selector ${selector}`);
  }
  doc.body = node("body");
  doc.activeElement = null;
  doc.createElement = node;
  doc.createElementNS = (/** @type {string} */ _ns, /** @type {string} */ tag) =>
    node(tag);
  doc.getElementById = (/** @type {string} */ id) =>
    all(doc.body).find((item) => item.id === id) ?? null;
  doc.querySelector = (/** @type {string} */ selector) =>
    doc.body.querySelector(selector);
  return doc;
}

/** @param {any} doc @param {any} parent @param {string} tag @param {string} [id] */
function add(doc, parent, tag, id) {
  const item = doc.createElement(tag);
  if (id) item.id = id;
  parent.append(item);
  return item;
}

function inspectorPage() {
  const doc = fakeDocument();
  globalThis.document = doc;
  const aside = add(doc, doc.body, "aside");
  const rail = add(doc, aside, "div", "replay-rail");
  rail.hidden = true;
  add(doc, rail, "div", "replay-rail-tabs").hidden = true;
  for (const tab of LEDGER_TABS) add(doc, rail, "section", tab.panelId).hidden = true;
  add(doc, aside, "details", "roster-details");
  add(doc, aside, "section", "champion-frame");
  add(doc, aside, "details", "agent-details");
  add(doc, aside, "div", "replay-details");
  const metrics = add(doc, aside, "details", "evaluation-metrics");
  add(doc, metrics, "summary").textContent = "TDM Evaluation Metrics";
  const body = add(doc, metrics, "div");
  body.className = "disclosure-body";
  const scopeLabel = add(doc, body, "label");
  scopeLabel.setAttribute("for", "metric-scope");
  const scope = add(doc, body, "select", "metric-scope");
  scope.value = "cursor";
  const topicLabel = add(doc, body, "label");
  topicLabel.setAttribute("for", "metric-selection");
  add(doc, body, "select", "metric-selection");
  const viewField = add(doc, body, "div", "metric-view-field");
  add(doc, viewField, "select", "metric-view").value = "totals";
  add(doc, body, "div", "metric-search-area");
  add(doc, body, "section", "metric-rows");
  for (const words of [
    "Metrics summarize the captured game.",
    "Up to Current Tick...",
  ]) {
    const note = add(doc, body, "p");
    note.className = "section-note";
    note.textContent = words;
  }
  add(doc, aside, "details", "pending-turn-details");
  const latest = add(doc, aside, "details", "latest-transition-details");
  const announcement = add(doc, latest, "p", "accepted-announcement");
  announcement.className = "sr-only";
  announcement.setAttribute("aria-live", "polite");
  add(doc, aside, "details", "visual-filters");
  add(doc, aside, "details", "technical-frame-details");
  return doc;
}

const agents = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5].map((class_id, slot) => ({
  slot,
  class_id,
  class_name: ["", "Mage", "Warrior", "Hunter", "Rogue", "Priest"][class_id],
  team_id: slot < 5 ? 1 : 2,
  active: true,
}));

/** @param {Record<string, any>} fields */
function statistic(fields) {
  return {
    applicable: true,
    unit: "count",
    status: null,
    subtitle: null,
    description: "What it counts.",
    guidance: "Higher is more.",
    missing_when: "The game has not reached it.",
    numerator: null,
    denominator: null,
    topic_order: { damage_done: [fields.order] },
    locations: [{ topic: "damage_done", view: "totals" }],
    valid: true,
    value: 3,
    ...fields,
  };
}

const summary = {
  scope: "cursor",
  topics: [{ name: "damage_done", description: "Damage done." }],
  statistics: [
    statistic({
      name: "episode_length",
      scope: "episode",
      subjects: [],
      subject: "Episode",
      label: "Episode Length",
      unit: "steps",
      order: 0,
    }),
    statistic({
      name: "team_b_damage",
      scope: "team",
      subjects: [2],
      subject: "Team B",
      label: "Damage Done",
      unit: "health",
      order: 1,
      value: 1234.5,
    }),
    statistic({
      name: "agent_3_damage",
      scope: "agent",
      subjects: [3],
      subject: "Agent ID 3 · Rogue · Team A",
      label: "Damage Done",
      order: 2,
      valid: false,
      value: null,
    }),
    statistic({
      name: "team_a_to_agent_7_damage",
      scope: "team_recipient",
      subject_role: "recipient",
      subjects: [1, 7],
      subject: "Team A → Agent ID 7 · Warrior · Team B",
      label: "Damage Received",
      unit: "fraction",
      order: 3,
      value: 0.123456,
    }),
    statistic({
      name: "agent_2_to_agent_5_damage",
      scope: "source_recipient",
      subjects: [2, 5],
      subject: "Agent ID 2 · Hunter · Team A → Agent ID 5 · Mage · Team B",
      label: "Damage Done",
      order: 4,
    }),
    statistic({
      name: "agent_0_and_agent_1_distance",
      scope: "ally_pair",
      subjects: [0, 1],
      subject: "Agent ID 0 · Mage · Team A ↔ Agent ID 1 · Warrior · Team A",
      label: "Ally Distance",
      unit: "distance",
      order: 5,
    }),
  ],
};

/** @param {any} container */
function inventory(container) {
  /** @param {any} root @returns {any[]} */
  const all = (root) => [root, ...root.children.flatMap(all)];
  return all(container)
    .filter((item) => item.tagName === "TR")
    .map((row) => {
      if (row.className === "metric-section") return ["section", row.textContent];
      if (!row.dataset.metric) return ["header", row.textContent];
      const [subject, measure, value] = row.children;
      const exact = all(subject).find(
        (item) => item.className === "ledger-subject__text",
      );
      const label = measure.children[0];
      return [
        row.dataset.metric,
        exact ? exact.textContent : subject.textContent,
        label.textContent,
        label.getAttribute("aria-description"),
        measure.children.at(-1).textContent,
        value.textContent,
        value.dataset.value,
        value.getAttribute("aria-label"),
      ];
    });
}

/** @param {Record<string, any>} options */
function renderRows(options) {
  const doc = fakeDocument();
  globalThis.document = doc;
  const container = doc.createElement("section");
  const description = doc.createElement("p");
  renderMetricRows(container, "damage_done", "totals", summary, description, options);
  return { container, description };
}

test("subject chips come only from the catalog's own IDs", () => {
  const [episode, team, agent, received, directed, pair] = summary.statistics;
  assert.deepEqual(metricSubjectParts(episode, agents), []);
  assert.deepEqual(metricSubjectParts(team, agents), [{ kind: "team", teamId: 2 }]);
  assert.deepEqual(metricSubjectParts(agent, agents), [
    { kind: "agent", slot: 3, classId: 4, className: "Rogue", teamId: 1 },
  ]);
  assert.deepEqual(
    metricSubjectParts(received, agents).map((part) => part.kind),
    ["team", "agent"],
  );
  assert.deepEqual(
    metricSubjectParts(directed, agents).map((part) => [part.kind, part.teamId]),
    [
      ["agent", 1],
      ["agent", 2],
    ],
  );
  assert.equal(metricSubjectParts(pair, agents).length, 2);
  assert.deepEqual(metricSubjectParts(agent, []), []);
  assert.deepEqual(metricSubjectParts({ scope: "team", subjects: [3] }, agents), []);
  assert.equal(metricValueText(summary.statistics[1]), "1,234.5");
  assert.equal(metricValueText(summary.statistics[2]), "—");
  assert.equal(metricValueText(summary.statistics[3]), "0.1235");
});

test("the ledger look keeps every row, value, tooltip and CSV name of the table", () => {
  const plain = renderRows({});
  const chips = renderRows({ agents });
  const expected = inventory(plain.container);
  assert.deepEqual(inventory(chips.container), expected);
  assert.equal(
    chips.description.textContent,
    "Damage done. 5 of 6 measurements available.",
  );
  assert.deepEqual(
    expected.filter((row) => row[0] === "section").map((row) => row[1]),
    [
      METRIC_SECTION_TITLES[0],
      METRIC_SECTION_TITLES[1],
      METRIC_SECTION_TITLES[2],
      METRIC_SECTION_TITLES[3],
      METRIC_SECTION_TITLES[4],
    ],
  );
  assert.deepEqual(METRIC_SECTION_TITLES, [
    "Episode and Team Totals",
    "Agent Details",
    "Recipient Totals",
    "Source-to-Recipient Details",
    "Ally Pair Details",
  ]);
  const missing = expected.find((row) => row[0] === "agent_3_damage");
  assert.deepEqual(missing?.slice(4), [
    "Count",
    "—",
    "",
    "Unavailable. Possible reasons: The game has not reached it.",
  ]);
  assert.equal(expected.find((row) => row[0] === "team_b_damage")?.[4], "HP");
  // Chips show identity only, and the exact host words stay for screen readers.
  /** @param {any} root @returns {any[]} */
  const all = (root) => [root, ...root.children.flatMap(all)];
  const cell = (/** @type {any} */ container, /** @type {string} */ name) =>
    all(container).find((item) => item.dataset?.metric === name).children[0];
  const directed = cell(chips.container, "agent_2_to_agent_5_damage");
  assert.equal(directed.className, "ledger-subject");
  assert.equal(directed.children[0].getAttribute("aria-hidden"), "true");
  assert.equal(directed.children[0].textContent, "Agent 2→Agent 5");
  assert.equal(
    all(directed.children[0]).filter((item) => item.dataset?.team === "b").length,
    1,
  );
  assert.equal(
    cell(chips.container, "agent_3_damage").children[0].textContent,
    "Agent 3Rogue",
  );
  assert.equal(
    cell(chips.container, "agent_0_and_agent_1_distance").children[0].textContent,
    "Agent 0↔Agent 1",
  );
  assert.equal(cell(chips.container, "episode_length").textContent, "Episode");
  assert.equal(cell(plain.container, "team_b_damage").textContent, "Team B");
  assert.equal(cell(plain.container, "team_b_damage").children.length, 0);
});

test("the rail builds six tabs, moves the cards and keeps live regions shown", () => {
  const doc = inspectorPage();
  const rail = mountLedgerRail(doc);
  assert.ok(rail);
  assert.equal(mountLedgerRail(doc), rail);
  const tabs = doc.getElementById("replay-rail-tabs");
  assert.equal(tabs.getAttribute("role"), "tablist");
  assert.deepEqual(
    tabs.children.map((/** @type {any} */ button) => button.textContent),
    ["Roster", "Agent", "Metrics", "Actions", "Visuals", "Frame"],
  );
  // Each tab button carries its registered hover help.
  tabs.children.forEach((/** @type {any} */ button, /** @type {number} */ index) => {
    assert.equal(button.getAttribute("aria-description"), LEDGER_TABS[index].help);
    assert.equal(button.getAttribute("data-tooltip-owner"), "");
  });
  assert.equal(doc.getElementById("replay-rail").hidden, false);
  for (const tab of LEDGER_TABS) {
    const panel = doc.getElementById(tab.panelId);
    assert.equal(panel.getAttribute("role"), "tabpanel");
    for (const id of LEDGER_TAB_CONTENT[
      /** @type {keyof typeof LEDGER_TAB_CONTENT} */ (tab.id)
    ]) {
      assert.equal(doc.getElementById(id).parentElement, panel, id);
    }
  }
  const announcement = doc.getElementById("accepted-announcement");
  assert.equal(announcement.parentElement.className, "ledger-rail__live");
  assert.equal(announcement.parentElement.parentElement.id, "replay-rail");
  // Agent opens first; Metrics stays closed so no metric work starts.
  assert.equal(rail.activeTab(), "agent");
  assert.equal(doc.getElementById("replay-rail-agent").hidden, false);
  assert.equal(doc.getElementById("replay-rail-metrics").hidden, true);
  // The long agent details card no longer moves into the Agent tab.
  assert.equal(doc.getElementById("agent-details").parentElement.tagName, "ASIDE");
  assert.equal(
    doc.getElementById("champion-frame").parentElement.id,
    "replay-rail-agent",
  );
  assert.equal(
    doc.getElementById("replay-details").parentElement.id,
    "replay-rail-frame",
  );
  assert.equal(doc.getElementById("evaluation-metrics").open, false);
  tabs.children[2].click();
  assert.equal(rail.activeTab(), "metrics");
  assert.equal(doc.getElementById("evaluation-metrics").open, true);
  assert.equal(tabs.children[2].getAttribute("aria-selected"), "true");
  assert.equal(tabs.children[2].tabIndex, 0);
  assert.equal(tabs.children[1].tabIndex, -1);
  for (const [key, expected] of [
    ["ArrowRight", "actions"],
    ["End", "frame"],
    ["ArrowRight", "roster"],
    ["ArrowLeft", "frame"],
    ["Home", "roster"],
  ]) {
    tabs.dispatchEvent({ type: "keydown", key, preventDefault() {} });
    assert.equal(rail.activeTab(), expected, key);
    assert.equal(doc.activeElement.dataset.tab, expected);
  }
  assert.equal(doc.getElementById("evaluation-metrics").open, false);
  assert.equal(doc.getElementById("pending-turn-details").open, true);
  assert.equal(doc.getElementById("technical-frame-details").open, true);
});

test("the ledger header mirrors Scope and View and folds the reading notes", () => {
  const doc = inspectorPage();
  const rail = mountLedgerRail(doc, { initialTab: "metrics" });
  assert.equal(rail?.activeTab(), "metrics");
  const scope = doc.getElementById("metric-scope");
  /** @type {string[]} */
  const changes = [];
  scope.addEventListener("change", () => changes.push(scope.value));
  const scopeSwitch = doc.getElementById("metric-scope-switch");
  assert.equal(scope.getAttribute("aria-hidden"), "true");
  assert.equal(scope.tabIndex, -1);
  assert.deepEqual(
    scopeSwitch.children.map((/** @type {any} */ button) => button.textContent),
    ["Up To Current Tick", "Whole GameShows the ending too."],
  );
  scopeSwitch.children[1].click();
  assert.deepEqual(changes, ["final"]);
  assert.equal(scopeSwitch.children[1].getAttribute("aria-checked"), "true");
  scopeSwitch.children[1].click();
  assert.deepEqual(changes, ["final"]);
  // The header is its own block just before the Metrics card, so in the
  // Metrics tab only the card under it scrolls.
  const card = doc.getElementById("evaluation-metrics");
  const header = scopeSwitch.parentElement;
  assert.equal(header.className, "ledger-header");
  assert.equal(header.parentElement, card.parentElement);
  assert.equal(
    header.parentElement.children.indexOf(header) + 1,
    header.parentElement.children.indexOf(card),
  );
  const fold = doc.getElementById("metric-rows").nextElementSibling;
  assert.equal(fold.className, "ledger-notes");
  assert.equal(fold.children[0].textContent, "About These Numbers");
  assert.equal(fold.children.length, 3);
  assert.equal(doc.getElementById("ledger-match-summary").hidden, true);
  // A generic switch keeps its buttons (and focus) while labels change.
  const view = doc.getElementById("metric-view");
  const host = doc.createElement("div");
  renderChoiceSwitch(host, view, [
    { name: "totals", label: "Totals" },
    { name: "recipients", label: "By Recipient" },
  ]);
  const first = host.children[0];
  renderChoiceSwitch(host, view, [
    { name: "totals", label: "All Totals" },
    { name: "recipients", label: "By Recipient" },
  ]);
  assert.equal(host.children[0], first);
  assert.equal(first.textContent, "All Totals");
  assert.equal(first.getAttribute("aria-checked"), "true");
  view.disabled = true;
  renderChoiceSwitch(host, view, [
    { name: "totals", label: "Totals" },
    { name: "recipients", label: "By Recipient" },
  ]);
  assert.equal(host.children[1].disabled, true);
});

test("Technical Frame groups facts and offers Copy on ids", async () => {
  assert.equal(middleTruncate("short"), "short");
  const long = "episode-0001:transition:0123456789abcdef0123456789";
  const cut = middleTruncate(long);
  assert.equal(cut.length, 32);
  assert.ok(
    cut.startsWith("episode-0001:tra") && cut.endsWith("6789") && cut.includes("…"),
  );
  const facts = [
    { id: "episode", label: "Episode", value: "episode-1" },
    { id: "frame", label: "Frame", value: 4 },
    { id: "artifact_digest_prefix", label: "Artifact Digest Prefix", value: "cccc" },
    { id: "brand_new_fact", label: "New Fact", value: "x" },
    { id: "incoming_transition", label: "Incoming Transition", value: long },
    { id: "seeds", label: "Seeds", value: "Root 1 · Episode stream 2" },
  ];
  const groups = technicalFrameGroups(facts);
  assert.deepEqual(
    groups.map((group) => [
      group.title,
      group.facts.map((fact) => [fact.id, fact.copy]),
    ]),
    [
      [
        "This Frame",
        [
          ["frame", false],
          ["brand_new_fact", false],
          ["incoming_transition", true],
        ],
      ],
      [
        "This Game",
        [
          ["episode", true],
          ["seeds", false],
        ],
      ],
      ["This File", [["artifact_digest_prefix", true]]],
    ],
  );
  assert.equal(groups[0].facts[2].shown, cut);
  assert.equal(groups[0].facts[2].text, long);
  assert.deepEqual(technicalFrameGroups([]), []);
  const doc = fakeDocument();
  globalThis.document = doc;
  const container = doc.createElement("div");
  renderTechnicalFrame(container, []);
  assert.equal(container.textContent, "Technical Frame unavailable.");
  renderTechnicalFrame(container, facts);
  assert.deepEqual(
    container.children.map(
      (/** @type {any} */ section) => section.children[0].textContent,
    ),
    ["This Frame", "This Game", "This File"],
  );
  const transition = container.children[0].children[3];
  assert.equal(transition.dataset.technicalFact, "incoming_transition");
  assert.equal(transition.children[1].title, long);
  const copy = transition.children[2];
  assert.equal(copy.getAttribute("aria-label"), "Copy Incoming Transition");
  /** @type {string[]} */
  const copied = [];
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      clipboard: { writeText: async (/** @type {string} */ text) => copied.push(text) },
    },
  });
  copy.click();
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(copied, [long]);
  assert.equal(copy.textContent, "Copied");
});

test("the Full Record chip shows only in Agent POV and names the POV agent", async () => {
  const oracle = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(fixture.presentations.replay_oracle),
  );
  assert.equal(fullRecordChip(oracle), null);
  assert.equal(
    fullRecordChip(fixture.presentations.replay_no_shared_obs_agent_pov),
    null,
  );
  const pov = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(fixture.presentations.replay_no_shared_obs_agent_pov),
  );
  const chip = fullRecordChip(pov);
  assert.ok(chip);
  assert.match(chip.text, /^Full Record · Not Limited To Agent \d+'s View$/u);
  assert.equal(chip.text, fullRecordLabel(pov));
  assert.doesNotMatch(chip.text, /agent-\d/u);
  assert.match(chip.tooltip, /every agent/u);
  assert.doesNotMatch(chip.tooltip, /agent-\d/u);
  const doc = inspectorPage();
  const rail = mountLedgerRail(doc);
  const fresh = LEDGER_TABS.map(
    (tab) => doc.getElementById(tab.panelId).children[0].outerHTML,
  );
  rail?.update({ presentation: pov });
  const chips = LEDGER_TABS.map((tab) => doc.getElementById(tab.panelId).children[0]);
  assert.deepEqual(
    chips.map((node) => node.className === "ledger-full-record" && !node.hidden),
    [true, true, true, true, false, true],
  );
  assert.equal(chips[1].textContent, chip.text);
  assert.equal(chips[1].getAttribute("aria-description"), chip.tooltip);
  assert.equal(chips[1].hasAttribute("data-tooltip-owner"), true);
  rail?.update({ presentation: oracle });
  assert.equal(chips[1].hidden, true);
  assert.deepEqual(
    LEDGER_TABS.map((tab) => doc.getElementById(tab.panelId).children[0].outerHTML),
    fresh,
  );
});

test("the Match Summary gives the verdict at the end and never shows the ending early", async () => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  Object.assign(raw.match_summary, {
    task_mode: 1,
    score_threshold: 20,
    scores: [6, 2],
    outcome: "in_progress",
  });
  const progress = await normalizeAuthorizedPresentationFrameV1(raw);
  const rows = (/** @type {string} */ scope) => ({
    scope,
    statistics: ["score", "kills", "deaths", "return"].flatMap((stem, index) =>
      ["a", "b"].map((letter) => ({
        name: `team_${letter}_${stem}`,
        label: ["Score", "Total Kills", "Total Deaths", "Return"][index],
        unit: "count",
        valid: stem !== "return",
        value: stem === "return" ? null : 10 + index,
      })),
    ),
  });
  const cursor = ledgerMatchSummary(progress, rows("cursor"));
  assert.ok(cursor);
  assert.equal(cursor.heading, "Match In Progress · Up To Tick 1");
  assert.deepEqual(
    cursor.teams.map((team) => [
      team.label,
      team.values.map((value) => `${value.label} ${value.text}`),
    ]),
    [
      ["Team A", ["Score 10", "Total Kills 11", "Total Deaths 12", "Return —"]],
      ["Team B", ["Score 10", "Total Kills 11", "Total Deaths 12", "Return —"]],
    ],
  );
  assert.equal(cursor.teams[0].values[1].csv, "team_a_kills");
  const whole = ledgerMatchSummary(progress, rows("final"));
  assert.ok(whole);
  assert.deepEqual(
    whole.teams.map((team) =>
      team.values.map((value) => `${value.label} ${value.text}`),
    ),
    [["Score 6"], ["Score 2"]],
  );
  assert.match(String(whole.note), /never shows the ending early/u);
  assert.equal(ledgerMatchSummary(progress, null)?.note, null);
  for (const [outcome, scores, heading] of [
    ["team_b_win", [17, 20], "Team B Wins · 17 – 20 · Score Limit Reached · Tick 1"],
    [
      "draw",
      [19, 16],
      "Draw · 19 – 16 · No Team Reached 20 · A Lead At Tick 3 Wins Nothing",
    ],
  ]) {
    Object.assign(raw.match_summary, { outcome, scores });
    const done = await normalizeAuthorizedPresentationFrameV1(raw);
    const view = ledgerMatchSummary(done, rows("final"));
    assert.equal(view?.heading, heading);
    assert.equal(view?.teams[0].values.length, 4);
  }
  Object.assign(raw.match_summary, {
    task_mode: 0,
    outcome: "not_applicable",
    scores: [0, 0],
    score_threshold: 0,
  });
  assert.equal(
    ledgerMatchSummary(await normalizeAuthorizedPresentationFrameV1(raw), null),
    null,
  );
});
