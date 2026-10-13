/**
 * @file Check the HUD (looks/hud.js, Arena Frames), its frame sources
 * (hudSourcesFromFrame in match-summary.js) and the score strip
 * (renderMatchSummary).
 *
 * Contracts checked:
 * - formatLifeNumber: whole numbers from 10 up (rounded down), one decimal below
 *   10 (rounded up), "0" only for exactly 0; Cinder at 30% life or less.
 * - Agent POV boundary: in the BattleClient and DevClient a champion the POV
 *   team cannot see becomes an identity-only "Unseen" record (no life,
 *   statuses, Ultimate, shield, wave, credit or position) whose Champion Card
 *   is the reduced card ("Unseen", no life, Ultimate or status), and its team
 *   shows no Standing; the Replay Viewer lists everyone under a Full Record
 *   chip while the Champion Card of an unseen champion is the reduced card
 *   naming the Agent POV team ("Team A cannot see it now.") and its chip tips
 *   are limited.
 * - Oracle View -> Agent POV -> Oracle View gives exactly a fresh set of records.
 * - Records: Team A first; the tray keeps 2 chips in the Champion Card's
 *   order (stuns, then slows, then the rest) plus "+N" whose lines read like
 *   the card's rows, and never In Combat; Freezing Trap chips use Rime; the
 *   Ultimate ring is full when
 *   Ready; fallen frames read "Returns In" countdown + 1; kill credit comes from
 *   the plan's fall rows, else Oracle's match_summary.deaths, and never from
 *   match_summary in Agent POV; the switches team_rosters, fighters_standing,
 *   death_announcer, respawn_wave and symbol_explanations act as documented.
 * - Beats: a fall this tick gives a "before" node with the life from before the
 *   blow (data-beat-hold="fall"), a hit or heal without a fall (beatView hits)
 *   gives a "before" life bar with the life from before the tick
 *   (data-beat-hold="impact", so the bar changes at Impact; a fallen champion's
 *   hit row makes no such hold), a return gives a "before" fallen node
 *   (data-beat-hold="return"), Standing changes with them, and a scoring tick
 *   gives a score "before" node (data-beat-hold="score"); the DOM keeps END.
 * - DOM: the whole HUD is one Tab stop with roving focus (arrows, Home, End,
 *   Escape), Enter selects a frame, unseen frames never select, and the strip
 *   puts Team A in the left container whatever its spawn side.
 * - Replay Viewer Arena Frames dock side: render writes data-team-a-side
 *   ("left" or "right", from Team A's recorded display side) on the frames
 *   container, so the team whose home end is on the arena's left gets the
 *   left dock, as on the scorebug.
 * - Replay Viewer thin band: on a page with #scorebug the docks have no
 *   header row (each dock is named "Team A Arena Frames"), and each team's
 *   "Standing N of M" moves under its name in the scorebug (#ctl-a's or
 *   #ctl-b's host), once per render; in Agent POV only the viewing team's
 *   count moves (hud_view.povTeamId).
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  build,
  CINDER_FRACTION,
  createHud,
  formatLifeNumber,
  records,
  tooltip,
} from "../src/looks/hud.js";
import {
  hudSourcesFromFrame,
  publicRosterIdentities,
  renderMatchSummary,
} from "../src/match-summary.js";
import { championCardOf } from "../src/tooltip.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @param {string} name */
async function frameOf(name) {
  return normalizeAuthorizedPresentationFrameV1(
    structuredClone(fixture.presentations[name]),
  );
}

/** @param {ReadonlyArray<Record<string, any>>} list @param {string} kind */
const ofKind = (list, kind) => list.filter((record) => record.kind === kind);

/**
 * @param {Partial<import("../src/looks/hud.js").HudSources>} overrides
 * @returns {import("../src/looks/hud.js").HudSources}
 */
function sources(overrides) {
  return {
    product: "replay_viewer",
    audience: "researcher",
    sceneAgents: [],
    researcherAgents: null,
    identities: [],
    mechanics: [{ class_id: 3, ultimate_cooldown_steps: 20 }],
    waves: [
      { team_id: 1, countdown_steps: 2 },
      { team_id: 2, countdown_steps: 4 },
    ],
    visualEvents: [],
    teams: [{ display_name: "learned-a" }, { display_name: "learned-b" }],
    deaths: [],
    controlledKey: null,
    recipientPublicAgentId: null,
    returnedFallback: [],
    displayIdOf: (id) => id.replace("agent-slot-", ""),
    ...overrides,
  };
}

/**
 * @param {import("../src/looks/hud.js").HudSources} gathered
 * @param {Omit<import("../src/looks/hud.js").HudInput, "sources">} [options]
 */
const recordsOf = (gathered, options = {}) =>
  records({ ...options, sources: gathered });

/** @param {Record<string, any>} overrides */
function agent(overrides) {
  return {
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
    statuses: [],
    ...overrides,
  };
}

test("life numbers are whole from 10 up and keep one decimal below 10", () => {
  assert.equal(formatLifeNumber(153.9), "153");
  assert.equal(formatLifeNumber(10), "10");
  assert.equal(formatLifeNumber(9.96), "10");
  assert.equal(formatLifeNumber(1.88), "1.9");
  assert.equal(formatLifeNumber(0.04), "0.1");
  assert.equal(formatLifeNumber(0), "0");
  assert.equal(formatLifeNumber(-1), null);
  assert.equal(formatLifeNumber(Number.NaN), null);
  assert.equal(CINDER_FRACTION, 0.3);
  const [low] = ofKind(
    recordsOf(sources({ sceneAgents: [agent({ current_health: 30 })] })),
    "champion",
  );
  assert.equal(low.life.low, true);
  const [high] = ofKind(
    recordsOf(sources({ sceneAgents: [agent({ current_health: 31 })] })),
    "champion",
  );
  assert.equal(high.life.low, false);
});

test("BattleClient Agent POV frames show unseen champions by identity only", async () => {
  for (const name of ["live_no_shared_obs_agent_pov", "live_shared_obs_agent_pov"]) {
    const frame = await frameOf(name);
    const list = records({ sources: hudSourcesFromFrame(frame) });
    const seen = new Set(
      frame.scene.agents.map((/** @type {any} */ body) => body.public_agent_id),
    );
    const champions = ofKind(list, "champion");
    assert.equal(champions.length, publicRosterIdentities(frame).length);
    for (const champion of champions) {
      assert.equal(champion.state === "unseen", !seen.has(champion.publicAgentId));
      if (champion.state !== "unseen") continue;
      for (const key of ["life", "ultimate", "tray", "spawnShield", "returnsIn"]) {
        assert.equal(champion[key], null, `${name} ${champion.publicAgentId} ${key}`);
      }
      assert.deepEqual(champion.credits, []);
      assert.equal(champion.fallHold, null);
      assert.equal(champion.returnHold, null);
      assert.equal(champion.selectable, false);
      const { card, ...identity } = champion;
      assert.doesNotMatch(JSON.stringify(identity), /health|position|status|Life/u);
      assert.deepEqual(
        [card.state, card.life, card.ultimate, card.statuses, card.note],
        ["unseen", null, null, [], null],
      );
      assert.doesNotMatch(JSON.stringify(card), /health|position|Life|tick/u);
      const tip = tooltip(champion);
      assert.equal(tip?.kind, "champion");
      assert.equal(championCardOf(tip)?.state, "unseen");
      assert.match(tip?.summary ?? "", /Unseen · Team [AB] cannot see it now\./u);
    }
    for (const header of ofKind(list, "team_header")) {
      const team = champions.filter((champion) => champion.teamId === header.teamId);
      if (team.some((champion) => champion.state === "unseen")) {
        assert.equal(header.standing, null);
      }
    }
    assert.equal(ofKind(list, "hud_view")[0].fullRecord, false);
  }
});

test("Replay Viewer Agent POV lists the full record but limits tips on unseen champions", async () => {
  const frame = await frameOf("replay_no_shared_obs_agent_pov");
  const list = records({ sources: hudSourcesFromFrame(frame) });
  assert.equal(ofKind(list, "hud_view")[0].fullRecord, true);
  assert.match(
    ofKind(list, "hud_view")[0].fullRecordLabel,
    /^Full Record · Not Limited to Agent \d's View$/u,
  );
  const seen = new Set(
    frame.scene.agents.map((/** @type {any} */ body) => body.public_agent_id),
  );
  const champions = ofKind(list, "champion");
  assert.equal(champions.length, frame.researcher_space.roster_agents.length);
  assert.ok(champions.some((champion) => !seen.has(champion.publicAgentId)));
  for (const champion of champions) {
    assert.notEqual(champion.state, "unseen");
    assert.equal(champion.tipLimited, !seen.has(champion.publicAgentId));
    if (!champion.tipLimited) continue;
    assert.equal(champion.card.state, "unseen");
    assert.equal(champion.card.life, null);
    assert.deepEqual(champion.card.statuses, []);
    assert.match(championCardOf(tooltip(champion))?.unseenText ?? "", /^Team [AB] /u);
    for (const chip of champion.tray?.chips ?? []) assert.equal(chip.line, chip.title);
  }
});

test("Oracle View -> Agent POV -> Oracle View gives a fresh render", async () => {
  const oracle = await frameOf("replay_oracle");
  const first = records({ sources: hudSourcesFromFrame(oracle) });
  records({
    sources: hudSourcesFromFrame(await frameOf("replay_shared_obs_agent_pov")),
  });
  records({
    sources: hudSourcesFromFrame(await frameOf("live_no_shared_obs_agent_pov")),
  });
  assert.deepEqual(records({ sources: hudSourcesFromFrame(oracle) }), first);
  const champions = ofKind(first, "champion");
  assert.deepEqual(
    champions.map((champion) => [champion.teamId, champion.side]),
    [...champions].sort((a, b) => a.teamId - b.teamId).map((c) => [c.teamId, c.side]),
  );
  assert.equal(champions[0].side, "left");
  assert.deepEqual(records({ sources: hudSourcesFromFrame(null) }), []);
});

test("the tray keeps two chips in priority order plus a count, and never In Combat", () => {
  const statuses = [
    { token_id: "hunter_basic_slow", remaining_duration: 1 },
    { token_id: "in_combat", remaining_duration: 3 },
    { token_id: "rogue_poison_anti_heal", remaining_duration: 4 },
    { token_id: "hunter_trap_stun", remaining_duration: 3 },
  ];
  const [champion] = ofKind(
    recordsOf(sources({ sceneAgents: [agent({ statuses })] })),
    "champion",
  );
  assert.deepEqual(
    champion.tray.chips.map((/** @type {any} */ chip) => [chip.tokenId, chip.count]),
    [
      ["stun_hunter_trap", 3],
      ["slow_hunter_basic", 1],
    ],
  );
  assert.equal(champion.tray.chips[0].colorClass, "rime");
  assert.equal(champion.tray.chips[0].title, "Freezing Trap (Stun)");
  assert.equal(champion.tray.chips[0].time, "3 ticks left");
  assert.equal(
    champion.tray.chips[0].line,
    "It can't move or act. Damage breaks the ice. From a Hunter.",
  );
  assert.equal(champion.tray.more, 1);
  assert.deepEqual(champion.tray.moreLines, [
    "Crippling Poison (Healing Reduction) · 4 ticks left · Heals and rest mend it less. From a Rogue.",
  ]);
  assert.equal(champion.ultimate.ready, false);
  assert.equal(champion.ultimate.fraction, 0.75);
  const [ready] = ofKind(
    recordsOf(sources({ sceneAgents: [agent({ ultimate_cooldown_remaining: 0 })] })),
    "champion",
  );
  assert.equal(ready.ultimate.ready, true);
});

test("fallen frames read Returns In and name only the killers the audience may name", () => {
  const victim = agent({
    public_agent_id: "agent-slot-5",
    presentation_key: "key-5",
    team_id: 2,
    display_agent_id: "5",
    life_state: "corpse",
    alive: false,
    current_health: 0,
  });
  const deaths = [
    {
      public_agent_id: "agent-slot-5",
      contributors: [{ public_agent_id: "agent-slot-0", team_id: 1, class_id: 1 }],
    },
  ];
  const [oracle] = ofKind(
    recordsOf(sources({ sceneAgents: [victim], deaths })),
    "champion",
  );
  assert.equal(oracle.state, "fallen");
  assert.equal(oracle.returnsIn, 5);
  assert.equal(oracle.life.text, "0");
  assert.deepEqual(
    oracle.credits.map((/** @type {any} */ credit) => credit.label),
    ["Mage · ID 0"],
  );
  const [pov] = ofKind(
    recordsOf(sources({ audience: "agent_pov", sceneAgents: [victim], deaths })),
    "champion",
  );
  assert.deepEqual(pov.credits, []);
  const falls = [
    {
      victim: { publicAgentId: "agent-slot-5" },
      killers: [{ publicAgentId: "agent-slot-2", classId: 5, teamId: 1 }],
    },
  ];
  const [named] = ofKind(
    recordsOf(sources({ audience: "agent_pov", sceneAgents: [victim], deaths }), {
      falls,
    }),
    "champion",
  );
  assert.deepEqual(
    named.credits.map((/** @type {any} */ credit) => credit.publicAgentId),
    ["agent-slot-2"],
  );
  const [quiet] = ofKind(
    recordsOf(sources({ sceneAgents: [victim], deaths }), {
      filters: { death_announcer: false },
    }),
    "champion",
  );
  assert.deepEqual(quiet.credits, []);
});

test("a fall and a return this tick hold their old look until the beat", () => {
  const fallen = agent({
    public_agent_id: "agent-slot-1",
    presentation_key: "key-1",
    display_agent_id: "1",
    life_state: "corpse",
    alive: false,
    current_health: 0,
  });
  const back = agent({ spawn_shield_remaining: 3 });
  const beat = {
    life: [{ publicAgentId: "agent-slot-1", presentationKey: "pov-1", before: 7.25 }],
    returns: [{ publicAgentId: "agent-slot-0" }],
    schedule: {
      returns: [{ publicAgentId: "agent-slot-0", presentationKey: "pov-0" }],
    },
    score: { before: [0, 0], after: [0, 0], changed: false },
  };
  const list = recordsOf(sources({ sceneAgents: [back, fallen] }), { beat });
  const [returned, fell] = ofKind(list, "champion");
  assert.deepEqual(fell.fallHold.beatKey, "pov-1");
  assert.equal(fell.fallHold.life.text, "7.3");
  assert.equal(returned.returned, true);
  assert.equal(returned.returnHold.beatKey, "pov-0");
  const [header] = ofKind(list, "team_header");
  assert.deepEqual(header.standing, { alive: 1, total: 2 });
  // One fall and one return balance out, so Standing needs no hold.
  assert.equal(header.standingHold, null);
  const [fallOnly] = ofKind(
    recordsOf(sources({ sceneAgents: [agent({}), fallen] }), {
      beat: { ...beat, returns: [], schedule: { returns: [] } },
    }),
    "team_header",
  );
  assert.deepEqual(fallOnly.standingHold, {
    before: { alive: 2, total: 2 },
    kind: "fall",
    beatKey: "pov-1",
  });
  const quiet = recordsOf(sources({ sceneAgents: [back] }), {
    beat,
    filters: { respawn_wave: false, fighters_standing: false, team_rosters: false },
  });
  assert.equal(ofKind(quiet, "champion")[0].returned, false);
  assert.equal(ofKind(quiet, "champion")[0].spawnShield, 3);
  assert.equal(ofKind(quiet, "team_header")[0].standing, null);
  assert.equal(ofKind(quiet, "hud_view")[0].rosters, false);
});

/* A tiny DOM stand-in: enough for the HUD's builders and its focus rules. */
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
    /** @type {Map<string, Function[]>} */
    this.listeners = new Map();
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
    return (
      this.text +
      this.children
        .map((/** @type {FakeElement} */ child) => child.textContent)
        .join("")
    );
  }
  set textContent(value) {
    this.text = String(value);
    this.children = [];
  }
  get tabIndex() {
    return Number(this.attributes.get("tabindex") ?? -1);
  }
  set tabIndex(value) {
    this.attributes.set("tabindex", String(value));
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
  /** @param {string} name */
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  /** @param {...(FakeElement | string)} nodes */
  append(...nodes) {
    for (const item of nodes) {
      // A string is a text node: an element with only text and no tag match.
      const node =
        typeof item === "string" ? new FakeElement(this.ownerDocument, "#text") : item;
      if (typeof item === "string") node.text = item;
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
  get parentElement() {
    return this.parentNode;
  }
  remove() {
    const parent = this.parentNode;
    if (parent) parent.children = parent.children.filter((child) => child !== this);
    this.parentNode = null;
  }
  /** @param {string} selector */
  matches(selector) {
    if (selector.startsWith("#")) return this.getAttribute("id") === selector.slice(1);
    const attribute = /^\[([a-z-]+)\]$/u.exec(selector);
    if (attribute) return this.attributes.has(attribute[1]);
    const [tag, className] = selector.split(".");
    return this.tagName === tag.toUpperCase() && this.classList.contains(className);
  }
  /** @param {string} selector */
  querySelectorAll(selector) {
    return this.descendants().filter((node) => node.matches(selector));
  }
  /** @param {string} selector @returns {FakeElement | null} */
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }
  /** @param {string} selector @returns {FakeElement | null} */
  closest(selector) {
    /** @type {FakeElement | null} */
    let node = this;
    while (node) {
      if (node.matches(selector)) return node;
      node = node.parentNode;
    }
    return null;
  }
  /** @param {FakeElement | null} other */
  contains(other) {
    return other !== null && (other === this || this.descendants().includes(other));
  }
  focus() {
    this.ownerDocument.activeElement = this;
  }
  blur() {
    if (this.ownerDocument.activeElement === this)
      this.ownerDocument.activeElement = null;
  }
  /** @param {string} type @param {Function} handler */
  addEventListener(type, handler) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), handler]);
  }
  /** @param {string} type @param {Function} handler */
  removeEventListener(type, handler) {
    this.listeners.set(
      type,
      (this.listeners.get(type) ?? []).filter((entry) => entry !== handler),
    );
  }
  /** @param {Record<string, any>} event */
  dispatch(event) {
    /** @type {Record<string, any>} */
    const full = {
      target: this,
      stopped: false,
      prevented: false,
      stopPropagation() {
        this.stopped = true;
      },
      preventDefault() {
        this.prevented = true;
      },
      ...event,
    };
    /** @type {FakeElement | null} */
    let node = this;
    while (node && !full.stopped) {
      for (const handler of node.listeners.get(full.type) ?? []) handler(full);
      node = node.parentNode;
    }
    return full;
  }
}

class FakeDocument {
  constructor() {
    /** @type {FakeElement | null} */
    this.activeElement = null;
  }
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

/** @param {FakeDocument} doc */
function mount(doc) {
  const frames = doc.createElement("section");
  const strip = {
    root: doc.createElement("section"),
    task: doc.createElement("strong"),
    teams: [doc.createElement("div"), doc.createElement("div")],
  };
  strip.root.append(strip.task, ...strip.teams);
  return { frames, strip };
}

const NO_TIPS = { symbol_explanations: false };

test("a hit without a fall holds the frame's life bar until Impact", () => {
  const doc = new FakeDocument();
  const hit = agent({
    public_agent_id: "agent-slot-6",
    presentation_key: "key-6",
    team_id: 2,
    display_agent_id: "6",
    current_health: 15,
    maximum_health: 200,
  });
  const fallen = agent({
    public_agent_id: "agent-slot-1",
    display_agent_id: "1",
    life_state: "corpse",
    alive: false,
    current_health: 0,
  });
  const beat = {
    life: [],
    hits: [
      { publicAgentId: "agent-slot-6", presentationKey: "k6", before: 34, after: 15 },
      // A fallen champion's hit row never makes a hit hold (its fall owns it).
      { publicAgentId: "agent-slot-1", presentationKey: "k1", before: 9, after: 0 },
    ],
    returns: [],
    schedule: { returns: [] },
  };
  const list = recordsOf(sources({ sceneAgents: [fallen, hit] }), {
    filters: NO_TIPS,
    beat,
  });
  const byId = new Map(ofKind(list, "champion").map((r) => [r.publicAgentId, r]));
  assert.equal(byId.get("agent-slot-6")?.hitHold?.beatKey, "k6");
  assert.equal(byId.get("agent-slot-6")?.hitHold?.life.text, "34");
  assert.equal(byId.get("agent-slot-6")?.life.text, "15", "the DOM keeps END");
  assert.equal(byId.get("agent-slot-1")?.hitHold, null);
  const root = doc.createElement("div");
  root.append(.../** @type {any[]} */ (build(list, /** @type {any} */ (doc))));
  const impact = root
    .querySelectorAll("[data-beat-hold]")
    .filter((node) => node.getAttribute("data-beat-hold") === "impact");
  const before = impact.filter(
    (node) => node.getAttribute("data-beat-side") === "before",
  );
  const after = impact.filter(
    (node) => node.getAttribute("data-beat-side") === "after",
  );
  assert.equal(before.length, 1);
  assert.equal(after.length, 1);
  assert.match(before[0].textContent, /34/u);
  assert.equal(before[0].getAttribute("opacity"), "0");
  assert.equal(before[0].getAttribute("aria-hidden"), "true");
  assert.match(after[0].textContent, /15/u);
  // No hit row: the life bar is drawn once, with no hold.
  const plain = recordsOf(sources({ sceneAgents: [hit] }), {
    filters: NO_TIPS,
    beat: { ...beat, hits: [] },
  });
  assert.equal(ofKind(plain, "champion")[0].hitHold, null);
});

test("built frames keep END in the DOM and add hidden before nodes for held beats", () => {
  const doc = new FakeDocument();
  const fallen = agent({
    public_agent_id: "agent-slot-1",
    display_agent_id: "1",
    life_state: "corpse",
    alive: false,
    current_health: 0,
  });
  const list = recordsOf(sources({ sceneAgents: [agent({}), fallen] }), {
    filters: NO_TIPS,
    beat: {
      life: [{ publicAgentId: "agent-slot-1", presentationKey: "k1", before: 40 }],
      returns: [],
      schedule: { returns: [] },
    },
  });
  const nodes = build(list, /** @type {any} */ (doc));
  assert.equal(nodes.length, 1);
  const root = doc.createElement("div");
  root.append(.../** @type {any[]} */ (nodes));
  const holds = root.querySelectorAll("[data-beat-hold]");
  const before = holds.filter(
    (node) => node.getAttribute("data-beat-side") === "before",
  );
  assert.ok(before.length >= 2);
  for (const node of before) {
    assert.equal(node.getAttribute("opacity"), "0");
    assert.equal(node.style.opacity, "0");
    assert.equal(node.getAttribute("data-beat-key"), "k1");
    assert.equal(node.getAttribute("aria-hidden"), "true");
  }
  const lifeBefore = before.find((node) => node.textContent.includes("40"));
  assert.ok(lifeBefore);
  const frame = root
    .querySelectorAll("[data-hud-symbol]")
    .find((node) => node.getAttribute("data-public-agent-id") === "agent-slot-1");
  assert.match(frame?.textContent ?? "", /Returns in 3/u);
  assert.deepEqual(
    build(
      recordsOf(sources({ sceneAgents: [agent({})] }), {
        filters: { team_rosters: false },
      }),
      /** @type {any} */ (doc),
    ),
    [],
  );
});

test("the HUD is one Tab stop with roving focus and selects only seen frames", async () => {
  const doc = new FakeDocument();
  const { frames, strip } = mount(doc);
  /** @type {string[]} */
  const selected = [];
  const hud = createHud({
    frames: /** @type {any} */ (frames),
    extraRoots: [/** @type {any} */ (strip.root)],
    onSelect: (id) => selected.push(id),
  });
  const live = await frameOf("live_no_shared_obs_agent_pov");
  renderMatchSummary(/** @type {any} */ (strip), live, { filters: NO_TIPS });
  hud.render(hudSourcesFromFrame(live), { filters: NO_TIPS });
  const symbols = [strip.root, frames].flatMap((root) =>
    root.querySelectorAll("[data-hud-symbol]"),
  );
  assert.ok(symbols.length > 5);
  assert.equal(symbols.filter((node) => node.tabIndex === 0).length, 1);
  const frameNodes = frames.querySelectorAll("li.arena-frame");
  const unseen = frameNodes.filter(
    (node) => node.getAttribute("data-state") === "unseen",
  );
  assert.ok(unseen.length > 0);
  for (const node of unseen) {
    assert.match(node.textContent, /Unseen/u);
    assert.doesNotMatch(node.textContent, /\d{2,}/u);
  }
  const first = frameNodes[0];
  first.focus();
  const moved = first.dispatch({ type: "keydown", key: "ArrowDown" });
  assert.equal(moved.prevented, true);
  assert.equal(moved.stopped, true);
  assert.notEqual(doc.activeElement, first);
  assert.equal(symbols.filter((node) => node.tabIndex === 0).length, 1);
  const end = first.dispatch({ type: "keydown", key: "End" });
  assert.equal(end.prevented, true);
  first.dispatch({ type: "keydown", key: "Enter" });
  unseen[0].dispatch({ type: "keydown", key: "Enter" });
  assert.deepEqual(selected, [first.getAttribute("data-public-agent-id")]);
  const active = /** @type {any} */ (doc.activeElement);
  active.dispatch({ type: "keydown", key: "Escape" });
  assert.equal(doc.activeElement, null);
  const ignored = first.dispatch({ type: "keydown", key: "ArrowDown", ctrlKey: true });
  assert.equal(ignored.prevented, false);
  hud.destroy();
});

test("the strip puts Team A left and holds a changed score until the Fall beat", async () => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  raw.match_summary.task_mode = 1;
  raw.match_summary.score_threshold = 20;
  raw.match_summary.scores = [6, 2];
  raw.match_summary.previous_scores = [6, 2];
  raw.match_summary.outcome = "in_progress";
  raw.match_summary.teams[0].display_side = "right";
  raw.match_summary.teams[1].display_side = "left";
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const doc = new FakeDocument();
  const { frames, strip } = mount(doc);
  renderMatchSummary(/** @type {any} */ (strip), frame, {
    filters: NO_TIPS,
    beat: { score: { before: [5, 2], after: [6, 2], changed: true } },
  });
  assert.equal(strip.teams[0].getAttribute("data-team"), "1");
  assert.equal(strip.teams[0].getAttribute("data-side"), "right");
  const [score] = strip.teams[0].querySelectorAll("strong.match-scoreboard__score");
  assert.equal(score.textContent, "6");
  const before = strip.teams[0]
    .querySelectorAll("[data-beat-hold]")
    .filter((node) => node.getAttribute("data-beat-hold") === "score");
  assert.equal(before.length, 2);
  assert.ok(before.every((node) => node.getAttribute("data-beat-side") === "before"));
  assert.equal(before[0].textContent, "5");
  assert.equal(strip.teams[1].querySelectorAll("[data-beat-hold]").length, 0);
  assert.match(strip.task.textContent, /^Tick 1 of 3/u);
  // 20 segments in the END bar and 20 in the held "before" bar.
  assert.equal(strip.teams[0].querySelectorAll("span.match-strip__segment").length, 40);
  renderMatchSummary(/** @type {any} */ (strip), frame, {
    filters: { ...NO_TIPS, scorebug: false },
  });
  assert.equal(
    strip.teams[0].querySelectorAll("strong.match-scoreboard__score")[0].textContent,
    "Score: 6/20",
  );
  assert.equal(frames.children.length, 0);
});

test("Replay Viewer frames record which side Team A's dock takes", async () => {
  for (const side of ["right", "left"]) {
    const raw = structuredClone(fixture.presentations.replay_oracle);
    raw.match_summary.teams[0].display_side = side;
    raw.match_summary.teams[1].display_side = side === "right" ? "left" : "right";
    const frame = await normalizeAuthorizedPresentationFrameV1(raw);
    const doc = new FakeDocument();
    const { frames } = mount(doc);
    const hud = createHud({ frames: /** @type {any} */ (frames) });
    hud.render(hudSourcesFromFrame(frame), { filters: NO_TIPS });
    assert.equal(/** @type {any} */ (frames).dataset.teamASide, side);
    assert.ok(frames.querySelectorAll("li.arena-frame").length > 0);
    hud.destroy();
  }
});

/** A page with the scorebug's two team hosts, as index.html has them. */
class ScorebugDocument extends FakeDocument {
  constructor() {
    super();
    this.scorebug = this.createElement("section");
    this.scorebug.setAttribute("id", "scorebug");
    /** @type {Record<string, FakeElement>} */
    this.hosts = {};
    for (const key of ["b", "a"]) {
      const host = this.createElement("div");
      const name = this.createElement("b");
      const line = this.createElement("span");
      line.setAttribute("id", `ctl-${key}`);
      host.append(name, line);
      this.scorebug.append(host);
      this.hosts[key] = host;
    }
  }
  /** @param {string} id */
  getElementById(id) {
    return id === "scorebug" ? this.scorebug : null;
  }
  addEventListener() {}
  removeEventListener() {}
}

test("Replay Viewer docks have no header row and Standing moves to the scorebug", async () => {
  for (const name of ["replay_oracle", "replay_no_shared_obs_agent_pov"]) {
    const doc = new ScorebugDocument();
    const { frames } = mount(doc);
    const hud = createHud({ frames: /** @type {any} */ (frames) });
    const sources = hudSourcesFromFrame(await frameOf(name));
    hud.render(sources, { filters: NO_TIPS });
    hud.render(sources, { filters: NO_TIPS });
    assert.equal(frames.querySelectorAll("div.arena-frames__header").length, 0);
    const docks = frames.querySelectorAll("div.arena-frames__dock");
    assert.equal(docks.length, 2);
    for (const dock of docks)
      assert.match(dock.getAttribute("aria-label") ?? "", /^Team [AB] Arena Frames$/u);
    const view = ofKind(records({ sources }), "hud_view")[0];
    for (const [key, teamId] of /** @type {const} */ ([
      ["a", 1],
      ["b", 2],
    ])) {
      const counts = doc.hosts[key].querySelectorAll("span.arena-frames__standing");
      const shown = view.audience !== "agent_pov" || view.povTeamId === teamId;
      assert.equal(counts.length, shown ? 1 : 0, `${name}: team ${teamId}`);
      if (shown) assert.match(counts[0].textContent, /^Standing \d+ of \d+/u);
    }
    if (view.audience === "agent_pov")
      assert.ok(view.povTeamId === 1 || view.povTeamId === 2);
    hud.destroy();
  }
});
