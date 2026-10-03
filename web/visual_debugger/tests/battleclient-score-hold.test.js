/**
 * @file Check that a repaint of the same tick keeps the live scorebug's Fall
 * beat holds (hand play, 3 Oct 2026: the score showed no number for about a
 * second at every Fall beat).
 *
 * Contracts checked:
 * - renderMatchSummary (src/match-summary.js), live play: a second paint
 *   with the same scorebug view leaves the DOM alone, so the "before" copy
 *   of the old score (span.n, data-beat-side "before") and of the old bar
 *   are the very same nodes the choreographer is timing, and #score-a stays
 *   the "after" node. A new tick rebuilds them, and a tick without a score
 *   change leaves no "before" node.
 * - partyPaintKey (src/looks/hud.js): equal portraits and drawing context
 *   give equal text (the party keeps its nodes, and its Fall and Return
 *   holds); a changed life, hold, scene row or switch gives different text
 *   (the party is rebuilt); facts that cannot be written as JSON give null
 *   (always rebuilt).
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { partyPaintKey } from "../src/looks/hud.js";
import { renderMatchSummary } from "../src/match-summary.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/**
 * @param {[number, number]} scores
 * @param {[number, number]} previous
 */
async function liveFrame(scores, previous) {
  const raw = structuredClone(fixture.presentations.live_shared_obs_agent_pov);
  raw.match_summary = {
    ...raw.match_summary,
    task_mode: 1,
    score_threshold: 20,
    outcome: "in_progress",
    scores,
    previous_scores: previous,
    scores_ten_ticks_ago: previous,
  };
  return normalizeAuthorizedPresentationFrameV1(raw);
}

class El {
  /** @param {Doc} doc @param {string} tag @param {string} [cls] @param {string} [id] */
  constructor(doc, tag, cls = "", id = "") {
    this.ownerDocument = doc;
    this.tagName = tag.toUpperCase();
    this.id = id;
    this.className = cls;
    /** @type {El[]} */
    this.children = [];
    /** @type {El | null} */
    this.parentNode = null;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {Record<string, string>} */
    this.style = {};
    this.hidden = false;
    this.text = "";
    const data = (/** @type {string | symbol} */ key) =>
      `data-${String(key).replace(/[A-Z]/gu, (c) => `-${c.toLowerCase()}`)}`;
    this.dataset = new Proxy(
      {},
      {
        get: (_t, key) => this.attributes.get(data(key)),
        set: (_t, key, value) => {
          this.attributes.set(data(key), String(value));
          return true;
        },
        deleteProperty: (_t, key) => {
          this.attributes.delete(data(key));
          return true;
        },
      },
    );
    this.classList = {
      remove: (/** @type {string} */ name) => {
        this.className = this.classes()
          .filter((entry) => entry !== name)
          .join(" ");
      },
    };
  }
  classes() {
    return this.className.split(/\s+/u).filter(Boolean);
  }
  /** @returns {string} */
  get textContent() {
    return this.text + this.children.map((child) => child.textContent).join("");
  }
  set textContent(value) {
    this.children = [];
    this.text = String(value);
  }
  /** @param {string} name @param {unknown} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  /** @param {...El} nodes */
  append(...nodes) {
    for (const node of nodes) {
      node.remove();
      node.parentNode = this;
      this.children.push(node);
    }
  }
  /** @param {...El} nodes */
  replaceChildren(...nodes) {
    for (const child of this.children) child.parentNode = null;
    this.children = [];
    this.text = "";
    this.append(...nodes);
  }
  remove() {
    const parent = this.parentNode;
    if (parent === null) return;
    parent.children = parent.children.filter((child) => child !== this);
    this.parentNode = null;
  }
  /** @param {El} node @param {number} offset */
  insert(node, offset) {
    const parent = /** @type {El} */ (this.parentNode);
    node.remove();
    node.parentNode = parent;
    parent.children.splice(parent.children.indexOf(this) + offset, 0, node);
  }
  /** @param {El} node */
  before(node) {
    this.insert(node, 0);
  }
  /** @param {El} node */
  after(node) {
    this.insert(node, 1);
  }
  /** @returns {El[]} */
  descendants() {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }
  /** @param {string} selector One compound selector: tag, #id, .class, [attr="v"], :not(.class). */
  matches(selector) {
    let rest = selector;
    const not = /:not\(\.([\w-]+)\)/u.exec(rest);
    if (not) {
      if (this.classes().includes(not[1])) return false;
      rest = rest.replace(not[0], "");
    }
    const attribute = /\[([\w-]+)="([^"]*)"\]/u.exec(rest);
    if (attribute) {
      if (this.getAttribute(attribute[1]) !== attribute[2]) return false;
      rest = rest.replace(attribute[0], "");
    }
    const id = /#([\w-]+)/u.exec(rest);
    if (id) {
      if (this.id !== id[1]) return false;
      rest = rest.replace(id[0], "");
    }
    const wanted = [...rest.matchAll(/\.([\w-]+)/gu)].map((match) => match[1]);
    rest = rest.replace(/\.([\w-]+)/gu, "");
    if (!wanted.every((name) => this.classes().includes(name))) return false;
    return rest === "" || rest.toUpperCase() === this.tagName;
  }
  /** @param {string} selector */
  querySelectorAll(selector) {
    return this.descendants().filter((node) => node.matches(selector));
  }
  /** @param {string} selector */
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }
}

class Doc {
  constructor() {
    /** @type {El | null} */
    this.scorebug = null;
  }
  /** @param {string} tag */
  createElement(tag) {
    return new El(this, tag);
  }
  /** @param {string} id */
  getElementById(id) {
    return id === "scorebug" ? this.scorebug : null;
  }
}

Reflect.set(globalThis, "HTMLElement", El);

/** Build the vision scorebug (index.html #scorebug) and the old strip parts. */
function mount() {
  const doc = new Doc();
  const scorebug = new El(doc, "div", "bc-score", "scorebug");
  for (const key of ["b", "a"]) {
    const box = new El(doc, "div", `sb-score ${key}`);
    const row = new El(doc, "div");
    row.append(new El(doc, "span", "n num", `score-${key}`), new El(doc, "span", "of"));
    box.append(row, new El(doc, "span", "segbar", `seg-${key}`));
    scorebug.append(box);
  }
  const middle = new El(doc, "div", "sb-mid");
  middle.append(
    new El(doc, "b", "num"),
    new El(doc, "span"),
    new El(doc, "span", "wave"),
  );
  scorebug.append(middle);
  doc.scorebug = scorebug;
  const elements = {
    root: new El(doc, "section"),
    task: new El(doc, "strong"),
    teams: [new El(doc, "div"), new El(doc, "div")],
  };
  return { scorebug, elements };
}

/** @param {El} scorebug */
function teamA(scorebug) {
  const box = /** @type {El} */ (scorebug.querySelector(".sb-score.a"));
  return {
    before: box.querySelectorAll('[data-beat-side="before"]'),
    score: /** @type {El} */ (box.querySelector("#score-a")),
  };
}

test("a repaint of the same tick keeps the held score nodes", async () => {
  const { scorebug, elements } = mount();
  const frame = await liveFrame([3, 4], [2, 4]);
  const beat = { score: { before: [2, 4], after: [3, 4], changed: true } };
  renderMatchSummary(/** @type {any} */ (elements), frame, { beat });
  const first = teamA(scorebug);
  assert.equal(first.score.textContent, "3");
  assert.equal(first.score.getAttribute("data-beat-side"), "after");
  assert.deepEqual(
    first.before.map((node) => node.className),
    ["n num", "segbar"],
    "the old number and the old bar are held",
  );
  assert.equal(first.before[0].textContent, "2");
  assert.equal(first.before[1].children.length, 20);

  // The page repaints the same tick (a new server revision while it plays).
  renderMatchSummary(/** @type {any} */ (elements), frame, { beat });
  const again = teamA(scorebug);
  assert.equal(again.before.length, 2);
  assert.ok(
    again.before.every((node, index) => node === first.before[index]),
    "the held nodes the choreographer times are the same nodes",
  );
  assert.ok(again.before.every((node) => node.parentNode !== null));
  assert.equal(again.score.getAttribute("data-beat-side"), "after");

  // A later paint with no score change rebuilds and holds nothing.
  const next = await liveFrame([3, 4], [3, 4]);
  renderMatchSummary(/** @type {any} */ (elements), next, {
    beat: { score: { before: [3, 4], after: [3, 4], changed: false } },
  });
  const later = teamA(scorebug);
  assert.equal(later.before.length, 0);
  assert.ok(first.before.every((node) => node.parentNode === null));
  assert.equal(later.score.getAttribute("data-beat-side"), null);
  assert.equal(later.score.textContent, "3");
});

test("a new score change rebuilds the held nodes", async () => {
  const { scorebug, elements } = mount();
  const beat = { score: { before: [2, 4], after: [3, 4], changed: true } };
  renderMatchSummary(/** @type {any} */ (elements), await liveFrame([3, 4], [2, 4]), {
    beat,
  });
  const first = teamA(scorebug).before;
  renderMatchSummary(/** @type {any} */ (elements), await liveFrame([4, 4], [3, 4]), {
    beat: { score: { before: [3, 4], after: [4, 4], changed: true } },
  });
  const second = teamA(scorebug).before;
  assert.equal(second.length, 2);
  assert.equal(second[0].textContent, "3");
  assert.ok(second.every((node) => !first.includes(node)));
});

/** @param {Record<string, any>} [overrides] */
function portrait(overrides = {}) {
  return {
    publicAgentId: "6",
    displayId: "6",
    teamId: 2,
    classId: 2,
    className: "Warrior",
    state: "fallen",
    manual: false,
    focus: false,
    life: { fraction: 0, low: true, current: 0 },
    label: "ID 6",
    labelKind: "id",
    ariaLabel: "Warrior, Agent ID 6, Team B, fallen",
    refusal: null,
    hold: {
      kind: "fall",
      beatKey: "k6",
      state: "alive",
      life: { fraction: 0.2, low: true, current: 40 },
    },
    record: { kind: "champion", publicAgentId: "6" },
    ...overrides,
  };
}

/** @param {Record<string, any>} [row] @param {Record<string, any>} [overrides] */
function context(row = { public_agent_id: "6", current_health: 0 }, overrides = {}) {
  return {
    filters: null,
    tips: false,
    sceneById: new Map([["6", row]]),
    spawnShieldMechanics: null,
    cardFacts: null,
    ...overrides,
  };
}

test("partyPaintKey names exactly what a party's portraits draw", () => {
  const key = partyPaintKey(
    [/** @type {any} */ (portrait())],
    /** @type {any} */ (context()),
  );
  assert.equal(typeof key, "string");
  assert.equal(
    partyPaintKey([/** @type {any} */ (portrait())], /** @type {any} */ (context())),
    key,
    "the same tick painted again keeps the party",
  );
  for (const changed of [
    partyPaintKey(
      [/** @type {any} */ (portrait({ hold: null }))],
      /** @type {any} */ (context()),
    ),
    partyPaintKey(
      [
        /** @type {any} */ (
          portrait({ life: { fraction: 0.5, low: false, current: 100 } })
        ),
      ],
      /** @type {any} */ (context()),
    ),
    partyPaintKey(
      [/** @type {any} */ (portrait())],
      /** @type {any} */ (
        context({ public_agent_id: "6", current_health: 0, statuses: [7] })
      ),
    ),
    partyPaintKey(
      [/** @type {any} */ (portrait())],
      /** @type {any} */ (context(undefined, { filters: { status_marks: false } })),
    ),
    partyPaintKey(
      [/** @type {any} */ (portrait({ focus: true }))],
      /** @type {any} */ (context()),
    ),
  ]) {
    assert.notEqual(changed, key);
  }
  /** @type {Record<string, any>} */
  const loop = { public_agent_id: "6" };
  loop.self = loop;
  assert.equal(
    partyPaintKey(
      [/** @type {any} */ (portrait())],
      /** @type {any} */ (context(loop)),
    ),
    null,
  );
});
