/**
 * @file Check Momentum (switch hud_momentum, on by default since 4 Oct 2026; the V5 Arena
 * Frames pick, broadcast-hud.js segBar) in src/match-summary.js.
 *
 * Contracts checked:
 * - momentumBaselineFromFrame returns the host's public fact
 *   match_summary.scores_ten_ticks_ago only with the switch on, only for a
 *   TDM game, and only when each team's baseline is a whole number from 0 to
 *   its score; it returns null with the switch off or absent, for a frame from
 *   before the field existed, for a combat diagnostic and for a raw (not
 *   normalized) record. A frame whose baseline is above a later score is
 *   refused by the normalizer, so it never reaches the reader. The browser
 *   keeps no score history: the same frame always gives the same answer.
 * - liveScorebugView (the live scorebug) marks the segments above the
 *   baseline as recent (recentFrom) and none with Momentum off.
 * - The Replay Viewer strip: with the switch on, the END First To bar marks
 *   the points above the baseline "is-recent" (oldest "is-recent-first",
 *   newest "is-recent-last") and its label adds "2 in the last 10 ticks"; the
 *   Fall beat's held "before" bar marks its own recent points from the same
 *   baseline; a team with no recent point and the switch off mark none; the
 *   Scorebug switch off draws no bar; a threshold above 40 splits the fill
 *   into an older part and a recent part.
 * - Agent POV: scores are public, so an Agent POV frame with the same match
 *   facts draws the same strip, and Oracle View -> Agent POV -> Oracle View
 *   draws exactly what a fresh Oracle render draws.
 * - Momentum is page chrome: its switch is an "interface" switch in both
 *   products and is not one of the battlefield switches a PNG export records.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  liveScorebugView,
  MOMENTUM_TICKS,
  matchSummaryView,
  momentumBaselineFromFrame,
  renderMatchSummary,
} from "../src/match-summary.js";
import { replayPngExportableFilterIds } from "../src/replay-export.js";
import { VISUAL_FILTER_REGISTRY } from "../src/visual-filters.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

const ON = Object.freeze({ symbol_explanations: false, hud_momentum: true });
const OFF = Object.freeze({ symbol_explanations: false, hud_momentum: false });

/**
 * @param {string} name
 * @param {Record<string, any>} facts
 */
async function frameWith(name, facts) {
  const raw = structuredClone(fixture.presentations[name]);
  raw.match_summary = {
    ...structuredClone(fixture.presentations.replay_oracle.match_summary),
    episode_id: raw.match_summary?.episode_id ?? raw.source.episode_id,
    source_frame_index:
      raw.match_summary?.source_frame_index ?? raw.source.source_frame_index,
    simulator_step_count:
      raw.match_summary?.simulator_step_count ?? raw.source.source_simulator_step_count,
    task_mode: 1,
    score_threshold: 20,
    outcome: "in_progress",
    ...facts,
  };
  return normalizeAuthorizedPresentationFrameV1(raw);
}

// Team A scored twice in the last 10 ticks; Team B scored nothing.
const SCORED = Object.freeze({
  scores: [6, 2],
  previous_scores: [5, 2],
  scores_ten_ticks_ago: [4, 2],
});

class FakeElement {
  /** @param {FakeDocument} doc @param {string} tag */
  constructor(doc, tag) {
    this.ownerDocument = doc;
    this.tagName = tag.toUpperCase();
    /** @type {FakeElement[]} */
    this.children = [];
    /** @type {FakeElement | null} */
    this.parentNode = null;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {Record<string, string>} */
    this.style = {};
    this.text = "";
    this.tabIndex = -1;
    this.dataset = new Proxy(
      {},
      {
        get: (_t, key) =>
          this.getAttribute(
            `data-${String(key).replace(/[A-Z]/gu, (c) => `-${c.toLowerCase()}`)}`,
          ) ?? undefined,
        set: (_t, key, value) => {
          this.setAttribute(
            `data-${String(key).replace(/[A-Z]/gu, (c) => `-${c.toLowerCase()}`)}`,
            String(value),
          );
          return true;
        },
      },
    );
    this.classList = {
      add: (/** @type {string[]} */ ...names) => {
        const set = new Set(this.className.split(/\s+/u).filter(Boolean));
        for (const name of names) set.add(name);
        this.className = [...set].join(" ");
      },
      remove: (/** @type {string[]} */ ...names) => {
        this.className = this.className
          .split(/\s+/u)
          .filter((name) => name && !names.includes(name))
          .join(" ");
      },
      toggle: (/** @type {string} */ name, /** @type {boolean} */ force) =>
        force ? this.classList.add(name) : this.classList.remove(name),
      contains: (/** @type {string} */ name) =>
        this.className.split(/\s+/u).includes(name),
    };
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
    this.attributes.set("title", value);
  }
  /** @param {string} name @param {unknown} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    for (const node of nodes) {
      node.parentNode = this;
      this.children.push(node);
    }
  }
  /** @param {...FakeElement} nodes */
  replaceChildren(...nodes) {
    this.children = [];
    this.text = "";
    this.append(...nodes);
  }
  /** @returns {FakeElement[]} */
  descendants() {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }
  /** @param {string} className */
  all(className) {
    return this.descendants().filter((node) => node.classList.contains(className));
  }
  /** @returns {string} */
  serialize() {
    const attributes = [...this.attributes].sort().map(([k, v]) => `${k}=${v}`);
    return `<${this.tagName} ${attributes.join(" ")} ${JSON.stringify(this.style)}>${this.text}${this.children.map((child) => child.serialize()).join("")}</>`;
  }
}

class FakeDocument {
  /** @param {string} tag */
  createElement(tag) {
    return new FakeElement(this, tag);
  }
  /** @param {string} _ns @param {string} tag */
  createElementNS(_ns, tag) {
    return new FakeElement(this, tag);
  }
}

for (const name of ["Element", "HTMLElement", "SVGElement"]) {
  Reflect.set(globalThis, name, FakeElement);
}

function mount() {
  const doc = new FakeDocument();
  const strip = {
    root: doc.createElement("section"),
    task: doc.createElement("strong"),
    teams: [doc.createElement("div"), doc.createElement("div")],
  };
  strip.root.append(strip.task, ...strip.teams);
  return strip;
}

/** @param {FakeElement} team @returns {FakeElement[]} */
function bars(team) {
  return team.all("match-strip__bar");
}

/** @param {FakeElement} bar */
function recentOf(bar) {
  return bar.children.map((segment) =>
    ["is-recent", "is-recent-first", "is-recent-last"]
      .filter((name) => segment.classList.contains(name))
      .join("+"),
  );
}

test("momentumBaselineFromFrame reads only the host's public fact", async () => {
  assert.equal(MOMENTUM_TICKS, 10);
  const frame = await frameWith("replay_oracle", SCORED);
  assert.deepEqual(momentumBaselineFromFrame(frame, ON), [4, 2]);
  assert.ok(Object.isFrozen(momentumBaselineFromFrame(frame, ON)));
  // The same frame again gives the same answer: no history in the browser.
  assert.deepEqual(momentumBaselineFromFrame(frame, ON), [4, 2]);
  assert.equal(momentumBaselineFromFrame(frame, OFF), null);
  assert.equal(momentumBaselineFromFrame(frame, null), null);
  assert.equal(momentumBaselineFromFrame(frame, undefined), null);
  const old = await frameWith("replay_oracle", {
    ...SCORED,
    scores_ten_ticks_ago: null,
  });
  assert.equal(momentumBaselineFromFrame(old, ON), null);
  const missing = await frameWith("replay_oracle", { scores: [6, 2] });
  assert.equal(momentumBaselineFromFrame(missing, ON), null);
  // A baseline above a later score never reaches the reader: the normalizer
  // refuses the frame.
  await assert.rejects(
    frameWith("replay_oracle", { ...SCORED, scores_ten_ticks_ago: [7, 2] }),
    /ten ticks ago must not exceed/u,
  );
  const diagnostic = await frameWith("replay_oracle", {
    task_mode: 0,
    score_threshold: 0,
    outcome: "not_applicable",
    scores: [0, 0],
    previous_scores: [0, 0],
    scores_ten_ticks_ago: [0, 0],
  });
  assert.equal(momentumBaselineFromFrame(diagnostic, ON), null);
  const raw = structuredClone(fixture.presentations.replay_oracle);
  raw.match_summary = { ...raw.match_summary, ...SCORED, task_mode: 1 };
  assert.equal(momentumBaselineFromFrame(raw, ON), null);
});

test("the live scorebug marks the points above the baseline as recent", async () => {
  const frame = await frameWith("live_shared_obs_agent_pov", SCORED);
  const summary = matchSummaryView(frame);
  const on = liveScorebugView(summary, {
    recentBaseline: momentumBaselineFromFrame(frame, ON),
  });
  assert.deepEqual(
    on.teams.map((team) => [team.lit, team.recentFrom, team.segments]),
    [
      [6, 4, 20],
      [2, 2, 20],
    ],
  );
  const off = liveScorebugView(summary, {
    recentBaseline: momentumBaselineFromFrame(frame, OFF),
  });
  assert.deepEqual(
    off.teams.map((team) => team.recentFrom),
    [20, 20],
  );
});

test("the Replay Viewer strip marks recent points in the END and held bars", async () => {
  const frame = await frameWith("replay_oracle", SCORED);
  const beat = { score: { before: [5, 2], after: [6, 2], changed: true } };
  const strip = mount();
  renderMatchSummary(/** @type {any} */ (strip), frame, { filters: ON, beat });
  const [end, held] = bars(strip.teams[0]);
  assert.equal(end.getAttribute("data-beat-side"), null);
  assert.equal(held.getAttribute("data-beat-side"), "before");
  const expectedEnd = Array.from({ length: 20 }, (_, index) =>
    index === 4
      ? "is-recent+is-recent-first"
      : index === 5
        ? "is-recent+is-recent-last"
        : "",
  );
  assert.deepEqual(recentOf(end), expectedEnd);
  assert.equal(
    end.getAttribute("aria-label"),
    "6 of 20 points to win, 2 in the last 10 ticks",
  );
  const expectedHeld = Array.from({ length: 20 }, (_, index) =>
    index === 4 ? "is-recent+is-recent-first+is-recent-last" : "",
  );
  assert.deepEqual(recentOf(held), expectedHeld);
  // Team B scored nothing in the last 10 ticks.
  assert.ok(recentOf(bars(strip.teams[1])[0]).every((name) => name === ""));
  assert.equal(
    bars(strip.teams[1])[0].getAttribute("aria-label"),
    "2 of 20 points to win",
  );
  renderMatchSummary(/** @type {any} */ (strip), frame, { filters: OFF, beat });
  for (const bar of bars(strip.teams[0])) {
    assert.ok(recentOf(bar).every((name) => name === ""));
  }
  assert.equal(
    bars(strip.teams[0])[0].getAttribute("aria-label"),
    "6 of 20 points to win",
  );
  renderMatchSummary(/** @type {any} */ (strip), frame, {
    filters: { ...ON, scorebug: false },
  });
  assert.equal(bars(strip.teams[0]).length, 0);
});

test("a threshold above 40 splits the fill into older and recent points", async () => {
  const frame = await frameWith("replay_oracle", { ...SCORED, score_threshold: 50 });
  const strip = mount();
  renderMatchSummary(/** @type {any} */ (strip), frame, { filters: ON });
  const [bar] = bars(strip.teams[0]);
  assert.deepEqual(
    bar.children.map((fill) => [fill.className, fill.style.width]),
    [
      ["match-strip__fill", "8%"],
      ["match-strip__fill match-strip__fill--recent", "4%"],
    ],
  );
  renderMatchSummary(/** @type {any} */ (strip), frame, { filters: OFF });
  assert.deepEqual(
    bars(strip.teams[0])[0].children.map((fill) => [fill.className, fill.style.width]),
    [["match-strip__fill", "12%"]],
  );
});

test("Agent POV draws the same public Momentum and a round trip equals a fresh render", async () => {
  const oracle = await frameWith("replay_oracle", SCORED);
  const pov = await frameWith("live_shared_obs_agent_pov", SCORED);
  assert.deepEqual(
    momentumBaselineFromFrame(pov, ON),
    momentumBaselineFromFrame(oracle, ON),
  );
  const fresh = mount();
  renderMatchSummary(/** @type {any} */ (fresh), oracle, { filters: ON });
  const trip = mount();
  renderMatchSummary(/** @type {any} */ (trip), oracle, { filters: ON });
  renderMatchSummary(/** @type {any} */ (trip), pov, { filters: ON });
  const povBars = trip.teams.map((team) => bars(team).map(recentOf));
  renderMatchSummary(/** @type {any} */ (trip), oracle, { filters: ON });
  assert.deepEqual(
    povBars,
    fresh.teams.map((team) => bars(team).map(recentOf)),
  );
  assert.equal(trip.root.serialize(), fresh.root.serialize());
});

test("Momentum is page chrome, never a battlefield mark", () => {
  const row = VISUAL_FILTER_REGISTRY.find((entry) => entry.id === "hud_momentum");
  assert.ok(row);
  assert.equal(row.surface, "interface");
  assert.equal(row.defaultEnabled, true);
  assert.deepEqual([...row.products].sort(), ["combat_debugger", "replay_viewer"]);
  assert.equal(replayPngExportableFilterIds().includes("hud_momentum"), false);
});
