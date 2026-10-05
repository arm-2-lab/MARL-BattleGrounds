/**
 * @file Check Frame Explanation (src/looks/frame-explanation.js): the items of
 * one tick in tick order (one blow or heal item per recipient with its recorded
 * total and life, each new status its own item in the Champion Card row's
 * words with its source by class only, broken ice, falls with 1 or 2 points
 * from home_fall only, the score after the falls, returns and Ultimates
 * ready); Agent POV event lines name only champions in the POV scene ("a
 * hidden source", "a champion out of sight", no callout) while its status
 * items read exactly as in Oracle View, and it reads no researcher data;
 * Oracle -> Agent POV -> Oracle answers afresh;
 * callouts are export-safe numbered discs beside seen champions; the list
 * matches the items; and the controller owns the F key (Replay Viewer: F in
 * play pauses first, Play hides it; BattleClient: F while a command is in
 * flight is kept for that tick and opens at the first refresh once the field
 * waits for orders, a second F cancels it and a new tick drops it), the
 * button and its teardown.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  authorizedPresentationAudience,
  authorizedPresentationIncomingRows,
  authorizedPresentationMatchBeatFacts,
  authorizedPresentationSceneView,
} from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { runLocalCommand } from "../src/command-set.js";
import {
  build,
  createFrameExplanation,
  explainTick,
  records,
  renderFrameExplanationList,
  update,
} from "../src/looks/frame-explanation.js";

const SVG = "http://www.w3.org/2000/svg";

class FakeNode {
  /** @param {string} tagName @param {string | null} namespace @param {any} ownerDocument */
  constructor(tagName, namespace, ownerDocument) {
    this.tagName = tagName.toUpperCase();
    this.namespaceURI = namespace;
    this.ownerDocument = ownerDocument;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeNode[]} */
    this.children = [];
    /** @type {FakeNode | null} */
    this.parentNode = null;
    this.ownText = "";
    this.hidden = false;
    this.className = "";
    /** @type {Map<string, Set<Function>>} */
    this.listeners = new Map();
  }
  /** @returns {FakeNode | null} */
  get lastChild() {
    return this.children.at(-1) ?? null;
  }
  /** @returns {string} */
  get textContent() {
    return this.ownText + this.children.map((child) => child.textContent).join("");
  }
  set textContent(value) {
    this.children = [];
    this.ownText = String(value);
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
  /** @param {...FakeNode} nodes */
  append(...nodes) {
    for (const node of nodes) {
      node.parentNode?.children.splice(node.parentNode.children.indexOf(node), 1);
      node.parentNode = this;
      this.children.push(node);
    }
  }
  /** @param {...FakeNode} nodes */
  replaceChildren(...nodes) {
    this.children = [];
    this.ownText = "";
    this.append(...nodes);
  }
  remove() {
    this.parentNode?.children.splice(this.parentNode.children.indexOf(this), 1);
    this.parentNode = null;
  }
  /** @param {string} selector */
  querySelectorAll(selector) {
    assert.equal(selector, "[data-item]");
    /** @type {FakeNode[]} */
    const found = [];
    const walk = (/** @type {FakeNode} */ node) => {
      for (const child of node.children) {
        if (child.getAttribute("data-item") !== null) found.push(child);
        walk(child);
      }
    };
    walk(this);
    return found;
  }
  /** @param {string} type @param {Function} listener */
  addEventListener(type, listener) {
    this.listeners.set(type, (this.listeners.get(type) ?? new Set()).add(listener));
  }
  /** @param {string} type @param {Function} listener */
  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }
  /** @param {string} type */
  fire(type) {
    for (const listener of this.listeners.get(type) ?? []) listener({ target: this });
  }
  /** @returns {FakeNode[]} */
  descendants() {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }
}

function fakeDocument() {
  /** @type {any} */
  const doc = {};
  doc.createElement = (/** @type {string} */ tag) => new FakeNode(tag, null, doc);
  doc.createElementNS = (/** @type {string} */ namespace, /** @type {string} */ tag) =>
    new FakeNode(tag, namespace, doc);
  return doc;
}

const TRANSFORM = Object.freeze({
  scale: 10,
  worldWidth: 20,
  worldHeight: 10,
  viewportBounds: { left: 0, top: 0, right: 200, bottom: 100, width: 200, height: 100 },
  mapBounds: { left: 0, top: 0, right: 200, bottom: 100, width: 200, height: 100 },
  worldToScreen: (/** @type {any} */ point) => ({
    x: (Array.isArray(point) ? point[0] : point.x) * 10,
    y: 100 - (Array.isArray(point) ? point[1] : point.y) * 10,
  }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 10,
});

/** @param {number} slot @param {number} classId @param {number} teamId */
function agent(slot, classId, teamId, extra = {}) {
  return {
    presentation_key: `key_${slot}`,
    public_agent_id: `agent-slot-${slot}`,
    display_agent_id: String(slot),
    class_id: classId,
    team_id: teamId,
    position: [slot, 2],
    radius: 0.5,
    maximum_health: 80,
    statuses: [],
    ...extra,
  };
}

/** @param {number} slot */
function anchor(slot) {
  return {
    phase: "successor",
    position: [slot, 2],
    presentation_key: `key_${slot}`,
    public_agent_id: `agent-slot-${slot}`,
  };
}

const TICK_EVENTS = Object.freeze([
  {
    ordinal: 0,
    event_kind: "ability_activated",
    ability_component: "basic",
    source_anchor: anchor(7),
    recipient_anchor: anchor(6),
  },
  {
    ordinal: 1,
    event_kind: "ability_activated",
    ability_component: "ultimate",
    source_anchor: anchor(1),
    recipient_anchor: anchor(6),
  },
  {
    ordinal: 2,
    event_kind: "ability_activated",
    ability_component: "ultimate",
    source_anchor: anchor(5),
    recipient_anchor: null,
  },
  {
    ordinal: 3,
    event_kind: "recipient_health_resolution",
    recipient_anchor: anchor(6),
    total_effective_damage: 25,
    total_effective_healing: 0,
    realized_net_health_change: -25,
    transition_start_health: 60,
    health_after_combat_resolution: 35,
  },
  { ordinal: 4, event_kind: "cooldown_ready", agent_anchor: anchor(5) },
  {
    ordinal: 5,
    event_kind: "agent_died",
    recipient_anchor: anchor(4),
    home_fall: true,
  },
  {
    ordinal: 6,
    event_kind: "agent_died",
    recipient_anchor: anchor(3),
    home_fall: false,
  },
  {
    ordinal: 7,
    event_kind: "status_broken_by_damage",
    recipient_anchor: anchor(6),
    status_id: "hunter_trap_stun",
  },
  {
    ordinal: 8,
    event_kind: "status_applied",
    recipient_anchor: anchor(6),
    source_anchor: anchor(7),
    status_id: "hunter_basic_slow",
  },
  {
    ordinal: 9,
    event_kind: "status_applied",
    recipient_anchor: anchor(6),
    source_anchor: anchor(1),
    status_id: "warrior_charge_stun",
  },
  { ordinal: 10, event_kind: "agent_respawned", agent_anchor: anchor(2) },
]);

const TICK_AGENTS = Object.freeze([
  agent(1, 2, 1),
  agent(2, 5, 1),
  agent(3, 4, 1),
  agent(4, 1, 2),
  agent(5, 1, 1),
  agent(6, 4, 2, {
    statuses: [
      {
        status_id: "hunter_basic_slow",
        magnitude: 0.85,
        magnitude_kind: "movement_multiplier",
        remaining_duration: 1,
      },
      { status_id: "warrior_charge_stun", remaining_duration: 2 },
    ],
  }),
  agent(7, 3, 1),
]);

test("one tick is explained in order: blows, abilities, falls, score, statuses, returns", () => {
  const explanation = explained({
    audience: "researcher",
    agents: TICK_AGENTS,
    events: [...TICK_EVENTS].reverse(),
    score: { changed: true, after: [3, 1] },
    tick: 12,
  });
  assert.equal(explanation.tick, 12);
  assert.deepEqual(
    explanation.items.map((item) => [item.n, item.kind, item.text]),
    [
      [1, "ability", "Mage ID 5's Burst was used."],
      [
        2,
        "blow",
        "Hunter ID 7's Serrated Arrows and Warrior ID 1's Charge hit Rogue ID 6: 25 damage this tick. Life 60 → 35 of 80.",
      ],
      [3, "ultimate_ready", "Mage ID 5's Burst is ready again."],
      [
        4,
        "fall",
        "Mage ID 4 fell. Team A scores 2 points (x2: it fell in its own Red Zone).",
      ],
      [5, "fall", "Rogue ID 3 fell. Team B scores 1 point."],
      [6, "score", "Score: Team A 3, Team B 1."],
      [
        7,
        "broken_ice",
        "The ice around Rogue ID 6 broke: Freezing Trap (Stun Broken).",
      ],
      [
        8,
        "status",
        "Rogue ID 6: Serrated Arrows (Slow) · 1 tick left · It moves at ×0.85 speed. From a Hunter.",
      ],
      [
        9,
        "status",
        "Rogue ID 6: Charge (Stun) · 2 ticks left · It can't move or act. From a Warrior.",
      ],
      [
        10,
        "return",
        "Priest ID 2 came back on a spawn pad under Spawn Shield (Invulnerable).",
      ],
    ],
  );
  assert.equal(explanation.items[1].anchorKey, "key_6");
  assert.ok(Object.isFrozen(explanation.items[0]));
});

test("Agent POV names only champions its scene holds and uses the life change it saw", () => {
  const seen = TICK_AGENTS.filter(
    (row) => ![1, 4, 7].includes(Number(row.presentation_key.slice(4))),
  );
  const explanation = explained({
    audience: "agent_pov",
    agents: seen,
    events: TICK_EVENTS,
    score: { changed: true, after: [3, 1] },
    tick: 12,
  });
  const text = explanation.items.map((item) => item.text).join("\n");
  assert.doesNotMatch(text, /ID 1\b|ID 4\b|ID 7\b/u);
  // Event lines keep their sight rule: an unseen attacker is a hidden source.
  const events = explanation.items
    .filter((item) => item.kind !== "status")
    .map((item) => item.text)
    .join("\n");
  assert.doesNotMatch(events, /Hunter|Warrior/u);
  assert.match(events, /A hidden source hit Rogue ID 6/u);
  assert.match(text, /A champion out of sight fell\. The other team scores 2 points/u);
  // Status items name the source by class only, the same as in Oracle View.
  assert.deepEqual(
    explanation.items.filter((item) => item.kind === "status").map((item) => item.text),
    [
      "Rogue ID 6: Serrated Arrows (Slow) · 1 tick left · It moves at ×0.85 speed. From a Hunter.",
      "Rogue ID 6: Charge (Stun) · 2 ticks left · It can't move or act. From a Warrior.",
    ],
  );
  const blow = explanation.items.find((item) => item.kind === "blow");
  assert.equal(
    blow?.text,
    "A hidden source hit Rogue ID 6: 25 damage this tick. Life 60 → 35 of 80.",
  );
  const unseenFall = explanation.items.find((item) =>
    item.text.startsWith("A champion out of sight"),
  );
  assert.equal(unseenFall?.anchorKey, null, "no callout for an unseen champion");
});

test("a recipient untouched and unchanged is left out; a healed one reads as healing", () => {
  const explanation = explained({
    audience: "agent_pov",
    agents: [agent(2, 5, 1), agent(3, 4, 1)],
    events: [
      {
        ordinal: 0,
        event_kind: "recipient_health_resolution",
        recipient_anchor: anchor(2),
        realized_net_health_change: 0,
        transition_start_health: 80,
        health_after_combat_resolution: 80,
      },
      {
        ordinal: 1,
        event_kind: "recipient_health_resolution",
        recipient_anchor: anchor(3),
        realized_net_health_change: 8,
        transition_start_health: 40,
        health_after_combat_resolution: 48,
      },
    ],
    score: null,
    tick: null,
  });
  assert.deepEqual(
    explanation.items.map((item) => item.text),
    ["Rogue ID 3 took 8 healing this tick. Life 40 → 48 of 80."],
  );
});

/** @param {Parameters<typeof explainTick>[0]} input */
function explained(input) {
  const out = explainTick(input);
  assert.ok(out);
  return out;
}

// The caller's half of the boundary: the shown audience's own data only.
/** @param {unknown} frame */
function tickInput(frame) {
  const audience = authorizedPresentationAudience(frame);
  if (audience === null) return null;
  const beat = authorizedPresentationMatchBeatFacts(frame);
  const raw = /** @type {Record<string, any>} */ (frame);
  const summary = audience === "agent_pov" ? raw.visual_events : raw.latest_events;
  return {
    audience,
    agents: authorizedPresentationSceneView(frame)?.agents ?? [],
    events: authorizedPresentationIncomingRows(frame)
      .filter((row) => row.vocabulary === "event")
      .map((row) => row.payload),
    score:
      beat === null ? null : { changed: beat.score_changed, after: beat.scores_after },
    tick: summary?.incoming_successor_simulator_step_count ?? null,
  };
}

/** @param {unknown} frame */
function frameExplanationItems(frame) {
  return explainTick(tickInput(frame));
}

/** @returns {Promise<Record<string, any>>} */
async function fixture() {
  return JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
}

test("real frames: Agent POV reads no researcher data and Oracle -> POV -> Oracle is fresh", async () => {
  const data = await fixture();
  const oracleRaw = data.presentations.replay_oracle;
  const povRaw = data.presentations.replay_no_shared_obs_agent_pov;
  const oracle = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(oracleRaw),
  );
  const pov = await normalizeAuthorizedPresentationFrameV1(structuredClone(povRaw));
  const first = frameExplanationItems(oracle);
  const povItems = frameExplanationItems(pov);
  const again = frameExplanationItems(oracle);
  assert.deepEqual(again, first);
  assert.equal(first?.audience, "researcher");
  assert.equal(povItems?.audience, "agent_pov");
  assert.deepEqual(
    first?.items.map((item) => item.kind),
    ["heal", "status"],
  );
  assert.match(String(first?.items[0].text), /8 healing this tick/u);
  assert.match(String(povItems?.items[0].text), /no change in life/u);

  const scene = new Set(
    povRaw.current_endpoint.parts.scene.agents.map(
      (/** @type {any} */ row) => row.public_agent_id,
    ),
  );
  const changed = structuredClone(povRaw);
  for (const row of changed.researcher_space.roster_agents) {
    if (!scene.has(row.public_agent_id)) {
      row.current_health -= 7;
      row.ultimate_cooldown_remaining += 2;
    }
  }
  const changedPov = await normalizeAuthorizedPresentationFrameV1(changed);
  assert.deepEqual(frameExplanationItems(changedPov), povItems);
  const source = await readFile(
    new URL("../src/looks/frame-explanation.js", import.meta.url),
    "utf8",
  );
  const code = source.replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gmu, "");
  assert.doesNotMatch(
    code,
    /researcher_space|ResearcherSceneView|match_summary|authorized-presentation-adapter/u,
  );
  assert.equal(explainTick(null), null);
  assert.equal(explainTick(/** @type {any} */ ({ audience: "nobody" })), null);
  assert.deepEqual(records({}), []);
});

test("callouts sit beside seen champions as export-safe numbered discs", () => {
  const items = explained({
    audience: "researcher",
    agents: TICK_AGENTS,
    events: TICK_EVENTS,
    score: null,
    tick: 1,
  });
  assert.ok(items.items.length > 0);
  const doc = fakeDocument();
  const callouts = [
    {
      kind: "frame_explanation_callout",
      n: 1,
      itemKind: "blow",
      text: "x",
      anchorKey: "key_6",
      x: 70,
      y: 70,
    },
    {
      kind: "frame_explanation_callout",
      n: 2,
      itemKind: "status",
      text: "y",
      anchorKey: "key_6",
      x: 90,
      y: 70,
    },
  ];
  const nodes = /** @type {any[]} */ (build(callouts, doc));
  assert.equal(nodes.length, 2);
  assert.equal(nodes[0].getAttribute("class"), "frame-explanation-callout");
  assert.equal(nodes[0].getAttribute("data-item"), "1");
  assert.equal(nodes[0].getAttribute("transform"), "translate(70 70)");
  assert.equal(nodes[1].textContent, "2");
  for (const node of nodes.flatMap((group) => [group, ...group.descendants()])) {
    assert.equal(node.namespaceURI, SVG);
    assert.ok(["G", "CIRCLE", "TEXT"].includes(node.tagName));
    for (const value of node.attributes.values()) {
      assert.doesNotMatch(value, /url\(|gradient/iu);
    }
  }
  update(nodes, [{ ...callouts[0], x: 10, y: 12.345 }]);
  assert.equal(nodes[0].getAttribute("transform"), "translate(10 12.35)");
  assert.equal(nodes[1].getAttribute("transform"), "translate(90 70)");
});

test("records place discs from the scene through the transform", async () => {
  const data = await fixture();
  const oracle = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(data.presentations.replay_oracle),
  );
  const out = records({ tick: tickInput(oracle), transform: TRANSFORM });
  assert.deepEqual(
    out.map((record) => record.n),
    [1, 2],
  );
  assert.equal(out[1].x - out[0].x, 20, "two items on one champion sit side by side");
  assert.equal(out[0].y, out[1].y);
  assert.deepEqual(records({ tick: tickInput(oracle), transform: null }), []);
});

test("the list repeats the items with numbers and a caption", () => {
  const doc = fakeDocument();
  const host = /** @type {any} */ (doc.createElement("div"));
  const explanation = explained({
    audience: "agent_pov",
    agents: TICK_AGENTS,
    events: TICK_EVENTS.slice(0, 4),
    score: null,
    tick: 5,
  });
  renderFrameExplanationList(host, explanation);
  assert.equal(
    host.children[0].textContent,
    "Tick 5 · Agent POV · Only what your team saw",
  );
  const rows = host.children[1].children;
  assert.equal(rows.length, explanation.items.length);
  assert.equal(rows[0].getAttribute("data-item"), "1");
  assert.equal(rows[0].children[1].textContent, explanation.items[0].text);
  renderFrameExplanationList(host, { ...explanation, items: [] });
  assert.match(host.children[1].textContent, /Nothing happened/u);
  renderFrameExplanationList(host, null);
  assert.equal(host.children.length, 0);
});

test("the controller owns F: pause first in the Replay Viewer, wait for the field in combat", async () => {
  const data = await fixture();
  const frame = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(data.presentations.replay_oracle),
  );
  const doc = fakeDocument();
  const layer = doc.createElementNS(SVG, "g");
  const listHost = doc.createElement("div");
  const button = doc.createElement("button");
  button.hidden = true;
  let paused = false;
  /** @type {boolean[]} */
  const changes = [];
  const key = {
    key: "f",
    shiftKey: false,
    ctrlKey: false,
    altKey: false,
    metaKey: false,
    target: null,
  };
  const replay = createFrameExplanation({
    product: "replay_viewer",
    getTickInput: () => tickInput(frame),
    getTransform: () => TRANSFORM,
    getLayer: () => layer,
    listHost,
    button,
    isPaused: () => paused,
    pause: () => {
      paused = true;
    },
    onOpenChange: (open) => changes.push(open),
  });
  assert.equal(button.hidden, false);
  assert.equal(button.getAttribute("aria-keyshortcuts"), "F");
  assert.equal(runLocalCommand("replay_viewer", /** @type {any} */ (key)), true);
  assert.equal(paused, true, "F in play pauses first");
  assert.equal(replay.isOpen(), true);
  assert.equal(button.getAttribute("aria-pressed"), "true");
  assert.equal(layer.children.length, 2);
  assert.equal(listHost.hidden, false);
  replay.refresh();
  assert.equal(layer.children.length, 2, "a refresh redraws without piling up");
  paused = false;
  replay.refresh();
  assert.equal(replay.isOpen(), false, "Play hides it");
  assert.equal(layer.children.length, 0);
  paused = true;
  button.fire("click");
  assert.equal(replay.isOpen(), true);
  runLocalCommand("replay_viewer", /** @type {any} */ (key));
  assert.equal(replay.isOpen(), false, "F again puts it away");
  assert.deepEqual(changes, [true, false, true, false]);
  replay.destroy();
  replay.destroy();

  let waiting = false;
  let laterTicks = 0;
  const combat = createFrameExplanation({
    product: "combat_debugger",
    getTickInput: () => {
      const input = tickInput(frame);
      return input === null ? null : { ...input, tick: (input.tick ?? 0) + laterTicks };
    },
    getTransform: () => null,
    getLayer: () => null,
    canExplain: () => waiting,
  });
  // F while a command is in flight is kept for this tick, not opened yet.
  assert.equal(runLocalCommand("combat_debugger", /** @type {any} */ (key)), true);
  assert.equal(combat.isOpen(), false);
  combat.refresh();
  assert.equal(combat.isOpen(), false, "still busy: it keeps waiting");
  waiting = true;
  combat.refresh();
  assert.equal(combat.isOpen(), true, "the field waits again: it opens");
  combat.close();
  waiting = false;
  runLocalCommand("combat_debugger", /** @type {any} */ (key));
  runLocalCommand("combat_debugger", /** @type {any} */ (key));
  waiting = true;
  combat.refresh();
  assert.equal(combat.isOpen(), false, "a second F cancels the waiting request");
  waiting = false;
  runLocalCommand("combat_debugger", /** @type {any} */ (key));
  laterTicks = 1;
  waiting = true;
  combat.refresh();
  assert.equal(combat.isOpen(), false, "a new tick drops the waiting request");
  laterTicks = 0;
  assert.equal(runLocalCommand("combat_debugger", /** @type {any} */ (key)), true);
  assert.equal(combat.isOpen(), true);
  waiting = false;
  combat.refresh();
  assert.equal(combat.isOpen(), false);
  assert.throws(
    () =>
      createFrameExplanation({
        product: "combat_debugger",
        getTickInput: () => null,
        getTransform: () => null,
        getLayer: () => null,
      }),
    TypeError,
    "one F owner per page",
  );
  combat.destroy();
  assert.throws(
    () =>
      createFrameExplanation(
        /** @type {any} */ ({
          product: "nope",
          getTickInput() {},
          getTransform() {},
          getLayer() {},
        }),
      ),
    RangeError,
  );
});
