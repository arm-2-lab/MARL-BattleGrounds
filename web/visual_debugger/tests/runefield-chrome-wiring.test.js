/**
 * @file Check the page chrome wiring added at integration. fullRecordChipWords is
 * the one owner of the Full Record words ("This Agent" when the POV agent is
 * unknown), and panels.js, ledger.js and the HUD all call it. The HUD tray
 * chips follow the Status Icons From Marks switch (status_marks_icons): sigils
 * when on, outline icons (the mark's shapes as thin lines) when off. The
 * Class Guide switch (class_guide) hides the live Class Guide card and the
 * Replay Viewer's Agent tab Class Guide, and the Match Summary switch
 * (ledger_summary) hides the
 * ledger's Match Summary, both wired from main.js. The scorebug draws "/ 20"
 * beside each score and the public wave pips under each bar.
 * requiems.fallenFeedHistory
 * carries the Fallen panel's falls one tick at a time for the same audience,
 * repeats them on a repaint, keeps only the last 8 ticks, and starts fresh on
 * a seek, a skip or a view or recipient change. main.js mounts the panel from
 * each plan through the choreographer's beforeBeatHolds hook, hides every painter
 * herald card (a plate: data-herald-moments) while the Verdict card or its
 * tucked plate shows, and draws no state chip (no countdown, no "Match Point ·
 * Needs 1"; owner, 4 Oct 2026). In the Replay Viewer a click on an
 * Arena Frame (selectAgentFromHud) sends that champion's Roster row command
 * through panels.pickAgent, so in Agent POV it moves the view to any
 * champion, as the roster does.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { fullRecordChipWords } from "../src/agent-identity.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { build, records } from "../src/looks/hud.js";
import { FEED_WINDOW_TICKS, fallenFeedHistory } from "../src/looks/requiems.js";
import { renderMatchSummary } from "../src/match-summary.js";

const FIXTURE = new URL("./fixtures/authorized-presentations-v1.json", import.meta.url);

/** A tiny DOM stand-in: enough for createSvgIcon and the scorebug builders. */
class Node {
  /** @param {string} tag */
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    /** @type {Node[]} */
    this.children = [];
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    this.text = "";
    /** @type {Record<string, string>} */
    this.style = {};
    this.ownerDocument = DOC;
    this.dataset = new Proxy(
      {},
      {
        get: (_t, key) =>
          this.attributes.get(
            `data-${String(key).replace(/[A-Z]/gu, (c) => `-${c.toLowerCase()}`)}`,
          ),
        set: (_t, key, value) => {
          this.attributes.set(
            `data-${String(key).replace(/[A-Z]/gu, (c) => `-${c.toLowerCase()}`)}`,
            String(value),
          );
          return true;
        },
      },
    );
    this.classList = {
      add: (/** @type {string[]} */ ...names) => {
        this.className = [...new Set([...this.classes(), ...names])].join(" ");
      },
      remove: (/** @type {string[]} */ ...names) => {
        this.className = this.classes()
          .filter((name) => !names.includes(name))
          .join(" ");
      },
      toggle: (/** @type {string} */ name, /** @type {boolean} */ force) =>
        force ? this.classList.add(name) : this.classList.remove(name),
      contains: (/** @type {string} */ name) => this.classes().includes(name),
    };
  }
  classes() {
    return this.className.split(/\s+/u).filter(Boolean);
  }
  get className() {
    return this.attributes.get("class") ?? "";
  }
  set className(value) {
    this.attributes.set("class", value);
  }
  /** @returns {string} */
  get textContent() {
    return this.text + this.children.map((child) => child.textContent).join("");
  }
  set textContent(value) {
    this.text = String(value);
    this.children = [];
  }
  get hidden() {
    return this.attributes.has("hidden");
  }
  set hidden(value) {
    if (value) this.attributes.set("hidden", "");
    else this.attributes.delete("hidden");
  }
  /** @param {string} value */
  set title(value) {
    this.attributes.set("title", String(value));
  }
  /** @param {number} value */
  set tabIndex(value) {
    this.attributes.set("tabindex", String(value));
  }
  /** @param {string} name @param {unknown} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  /** @param {string} name */
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  /** @param {...Node} nodes */
  append(...nodes) {
    this.children.push(...nodes);
  }
  /** @param {...Node} nodes */
  replaceChildren(...nodes) {
    this.text = "";
    this.children = [...nodes];
  }
  addEventListener() {}
  /** @returns {Node[]} */
  all() {
    return this.children.flatMap((child) => [child, ...child.all()]);
  }
  /** @param {string} name */
  byClass(name) {
    return this.all().filter((node) => node.classes().includes(name));
  }
}

const DOC = {
  /** @param {string} tag */
  createElement: (tag) => new Node(tag),
  /** @param {string} _ns @param {string} tag */
  createElementNS: (_ns, tag) => new Node(tag),
};

/** @param {string} name */
async function rawOf(name) {
  const fixture = JSON.parse(await readFile(FIXTURE, "utf8"));
  return structuredClone(fixture.presentations[name]);
}

/** @param {string} name */
async function source(name) {
  return readFile(new URL(`../src/${name}`, import.meta.url), "utf8");
}

test("the Full Record words have one owner that every caller uses", async () => {
  assert.equal(
    fullRecordChipWords("Agent 3"),
    "Full Record · Not Limited to Agent 3's View",
  );
  for (const unknown of [null, undefined, "", "  "]) {
    assert.equal(
      fullRecordChipWords(unknown),
      "Full Record · Not Limited to This Agent's View",
    );
  }
  for (const file of ["panels.js", "ledger.js", "looks/hud.js"]) {
    const code = await source(file);
    assert.match(code, /fullRecordChipWords\(/u, file);
    assert.doesNotMatch(code, /Not Limited to \$\{/iu, file);
  }
});

test("HUD tray chips follow the Status Icons From Marks switch", () => {
  const sources = {
    product: "replay_viewer",
    audience: "researcher",
    sceneAgents: [
      {
        public_agent_id: "agent-slot-0",
        presentation_key: "key-0",
        team_id: 1,
        class_id: 3,
        display_agent_id: "0",
        life_state: "alive",
        alive: true,
        current_health: 100,
        maximum_health: 100,
        ultimate_cooldown_remaining: 5,
        spawn_shield_remaining: 0,
        statuses: [{ token_id: "hunter_trap_stun", remaining_duration: 3 }],
      },
    ],
    researcherAgents: null,
    identities: [],
    mechanics: [],
    waves: [],
    visualEvents: [],
    teams: [{ display_name: "a" }, { display_name: "b" }],
    deaths: [],
    controlledKey: null,
    recipientPublicAgentId: null,
    returnedFallback: [],
    displayIdOf: (/** @type {string} */ id) => id.replace("agent-slot-", ""),
  };
  /** @param {Record<string, boolean>} filters */
  const chipStyles = (filters) => {
    const list = records({ sources: /** @type {any} */ (sources), filters });
    const champion = list.find((record) => record.kind === "champion");
    const root = new Node("div");
    root.append(.../** @type {any[]} */ (build(list, /** @type {any} */ (DOC))));
    return {
      record: champion?.tray?.iconStyle,
      drawn: root
        .byClass("arena-frame__chip-glyph")
        .map((node) => node.getAttribute("data-icon-style")),
    };
  };
  assert.deepEqual(chipStyles({ symbol_explanations: false }), {
    record: "sigil",
    drawn: ["sigil"],
  });
  assert.deepEqual(
    chipStyles({ symbol_explanations: false, status_marks_icons: false }),
    { record: "outline", drawn: ["outline"] },
  );
});

test("the Class Guide and Match Summary switches are wired to their cards", async () => {
  const main = await source("main.js");
  const tactics = await source("tactics-turn.js");
  const ledger = await source("ledger.js");
  assert.match(
    main,
    /classGuideEnabled: \(\) => isVisualFilterEnabled\(visualFilterState, "class_guide"\)/u,
  );
  assert.match(tactics, /const guideOn = classGuideEnabled\(\);/u);
  assert.match(tactics, /if \(guideCard\) guideCard\.hidden = !guideOn;/u);
  assert.match(
    tactics,
    /if \(!guideOn\) return;\s*if \(classNamesV1\(view\.classId\) === null\)/u,
  );
  // The Replay Viewer's Agent tab card obeys the same switch.
  assert.match(
    main,
    /!isVisualFilterEnabled\(visualFilterState, "class_guide"\)\) \{\s*card\.hidden = true;/u,
  );
  assert.match(
    main,
    /summaryEnabled: isVisualFilterEnabled\(visualFilterState, "ledger_summary"\)/u,
  );
  assert.match(
    ledger,
    /if \(matchSummary && input\.summaryEnabled === false\) \{[\s\S]*?renderLedgerMatchSummary\(matchSummary, null\);/u,
  );
});

test("the scorebug draws its goal and wave pips", async () => {
  const raw = await rawOf("replay_oracle");
  raw.match_summary.task_mode = 1;
  raw.match_summary.score_threshold = 20;
  raw.match_summary.outcome = "in_progress";
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const root = new Node("section");
  const task = new Node("strong");
  const teams = [new Node("div"), new Node("div")];
  renderMatchSummary(/** @type {any} */ ({ root, task, teams }), frame, {
    filters: { symbol_explanations: false },
    waves: [
      { team_index: 0, period_steps: 5, countdown_steps: 4 },
      { team_index: 1, period_steps: 5, countdown_steps: 0 },
    ],
  });
  for (const team of teams) {
    assert.deepEqual(
      team.byClass("match-strip__goal").map((node) => node.textContent),
      ["/ 20"],
    );
  }
  const lit = (/** @type {Node} */ team) =>
    team.byClass("match-strip__wave").filter((pip) => pip.classes().includes("is-lit"))
      .length;
  assert.equal(teams[0].byClass("match-strip__wave").length, 5);
  assert.equal(lit(teams[0]), 1);
  assert.equal(lit(teams[1]), 5);
  assert.equal(
    teams[1].byClass("match-strip__waves")[0].getAttribute("aria-label"),
    "Next Wave in 1 Tick",
  );
});

test("the Fallen panel carries falls tick by tick for one audience", () => {
  /** @param {string} audienceKey @param {number} step @param {string[]} victims */
  const plan = (audienceKey, step, victims) => ({
    audienceKey,
    simulatorStep: step,
    derivedEvents: victims.map((id) => ({
      kind: "fallen_feed_row",
      victim: { publicAgentId: id, teamId: 2, classId: 1, title: id },
      victimPresentationKey: `key-${id}`,
      killers: [],
      killingTeamId: 1,
      points: 1,
      ultimate: null,
    })),
  });
  const ids = (/** @type {ReturnType<typeof fallenFeedHistory>} */ result) =>
    result.rows.map((row) => `${row.tick}:${row.victim.publicAgentId}`);
  let result = fallenFeedHistory(null, plan("oracle", 10, ["a"]));
  assert.deepEqual(ids(result), ["10:a"]);
  result = fallenFeedHistory(result.memory, plan("oracle", 11, ["b"]));
  assert.deepEqual(ids(result), ["10:a", "11:b"]);
  assert.equal(result.rows[0].ultimate, null, "the ultimate field is carried");
  const repaint = fallenFeedHistory(result.memory, plan("oracle", 11, ["b"]));
  assert.deepEqual(ids(repaint), ["10:a", "11:b"], "a repaint gives the same rows");
  assert.deepEqual(
    ids(fallenFeedHistory(result.memory, plan("pov:agent-0", 12, []))),
    [],
    "a view change starts fresh",
  );
  assert.deepEqual(
    ids(fallenFeedHistory(result.memory, plan("oracle", 20, []))),
    [],
    "a seek starts fresh",
  );
  let walk = fallenFeedHistory(null, plan("oracle", 0, ["x"]));
  for (let step = 1; step <= FEED_WINDOW_TICKS; step += 1) {
    walk = fallenFeedHistory(walk.memory, plan("oracle", step, []));
  }
  assert.deepEqual(ids(walk), [], "a fall leaves after 8 ticks");
  assert.deepEqual(fallenFeedHistory(null, null), {
    memory: null,
    tick: null,
    rows: [],
  });
});

test("main.js mounts the Fallen panel from each plan and hides plates under the Verdict", async () => {
  const main = await source("main.js");
  const choreography = await source("choreography.js");
  assert.match(main, /beforeBeatHolds: renderFallenFeed,/u);
  assert.match(main, /elements\.fallenFeed,/u);
  assert.match(main, /isVisualFilterEnabled\(visualFilterState, "death_announcer"\)/u);
  assert.match(
    choreography,
    /surface\.beforeBeatHolds\(plan\);[\s\S]*?this\.painter\.install\(plan, surface, painterOptions\)/u,
  );
  assert.match(
    main,
    /const verdictShown = records\.some\(\s*\(row\) => row\.kind === "herald-verdict" \|\| row\.kind === "herald-verdict-plate",?\s*\);[\s\S]*?querySelectorAll\("\[data-herald-moments\]"\)[\s\S]*?card\.setAttribute\("visibility", "hidden"\)/u,
  );
  // There is no state chip any more (owner, 4 Oct 2026): no countdown and no
  // "Match Point · Needs 1"; only the Lineup and Verdict are drawn here.
  assert.doesNotMatch(main, /herald-state|part: "state"/u);
  assert.match(main, /if \(records\.length === 0\) return;/u);
});

test("a Replay Viewer Arena Frame click sends that champion's Roster row command", async () => {
  const main = await source("main.js");
  assert.match(
    main,
    /function selectAgentFromHud\(publicAgentId\) \{\s*if \(isReplayMode\(\)\) \{\s*panels\.pickAgent\(publicAgentId\);\s*return;\s*\}/u,
  );
  assert.match(
    main,
    /onSelect: \(publicAgentId\) => selectAgentFromHud\(publicAgentId\),/u,
  );
});
