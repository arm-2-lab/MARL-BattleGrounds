/**
 * @file Check the Spawn Ward's band close in the Champions look
 * (`src/looks/champions.js`, switch spawn_shield_expiry, off by default).
 *
 * Contracts checked: on the tick a champion's own spawn_shield_expired row says
 * its Spawn Shield ended, a living champion the audience saw at tick start gets
 * wardLift {fade}: 200 ms with Spawn Shield Expiry on, a 1 ms cut while it is
 * off. There is no lift without the Spawn Shield switch, without the row, while
 * a ward still shows, or in Agent POV for a body not seen at tick start (the
 * caller's startSeen test turns it on). The bundle holds the ward's earlier
 * look in `g.agent-ward-lift` (veil, 12-dash ring and Team B's split, one
 * diamond; END opacity 0 with the "settle" before-side hold; aria-hidden, no
 * pointer input, no focus and never an `.agent-spawn-shield`). The solid team
 * ring, Team B's split and the token's 80% to full opacity carry the matching
 * after-side hold. The next quiet tick on the same nodes holds exactly a fresh
 * build's DOM, a ward that shows again gets a fresh ward group, and the markup
 * is export-safe.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { authorizedPresentationSceneView } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { createViewportTransform } from "../src/layout.js";
import * as champions from "../src/looks/champions.js";
import { DEFAULT_VISUAL_FILTER_STATE } from "../src/visual-filters.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @param {string} name */
async function loadScene(name) {
  const frame = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(fixture.presentations[name]),
  );
  const scene = authorizedPresentationSceneView(frame);
  assert.ok(scene);
  const transform = createViewportTransform({
    worldWidth: scene.map.width,
    worldHeight: scene.map.height,
    viewportWidth: 960,
    viewportHeight: 600,
    padding: 24,
  });
  return { scene, transform };
}

/** @param {string} key */
function shieldEndedRow(key) {
  return {
    kind: "spawn_shield_expired",
    vocabulary: "event",
    payload: { agent_anchor: { presentation_key: key, phase: "successor" } },
  };
}

class FakeElement {
  /** @param {FakeDocument} doc @param {string} tagName */
  constructor(doc, tagName) {
    this.ownerDocument = doc;
    this.tagName = tagName;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    /** @type {FakeElement | null} */
    this.parentElement = null;
    /** @type {Record<string, string>} */
    this.dataset = {};
    this.textContent = "";
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
  hasAttribute(name) {
    return this.attributes.has(name);
  }
  /** @param {string} name */
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  getBoundingClientRect() {
    return { x: 0, y: 0, width: 0, height: 0, left: 0, top: 0, right: 0, bottom: 0 };
  }
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    for (const node of nodes) {
      node.remove();
      node.parentElement = this;
      this.children.push(node);
    }
  }
  /** @param {...FakeElement} nodes */
  replaceChildren(...nodes) {
    for (const child of this.children) child.parentElement = null;
    this.children = [];
    this.append(...nodes);
  }
  /** @param {FakeElement} node */
  replaceWith(node) {
    const parent = this.parentElement;
    if (!parent) return;
    const index = parent.children.indexOf(this);
    node.remove();
    parent.children[index] = node;
    node.parentElement = parent;
    this.parentElement = null;
  }
  remove() {
    if (this.parentElement) {
      this.parentElement.children = this.parentElement.children.filter(
        (child) => child !== this,
      );
      this.parentElement = null;
    }
  }
  /** @param {FakeElement} other */
  isEqualNode(other) {
    return other instanceof FakeElement && other.markup() === this.markup();
  }
  addEventListener() {}
  blur() {}
  /** @returns {FakeElement[]} */
  all() {
    return [this, ...this.children.flatMap((child) => child.all())];
  }
  /** @param {string} className */
  find(className) {
    return this.all().filter((node) =>
      (node.getAttribute("class") ?? "").split(/\s+/u).includes(className),
    );
  }
  /** @returns {string} */
  markup() {
    const attributes = [
      ...this.attributes,
      ...Object.entries(this.dataset).map(([key, value]) => [
        `data-${key.replace(/[A-Z]/gu, (letter) => `-${letter.toLowerCase()}`)}`,
        value,
      ]),
    ]
      .map(([key, value]) => ` ${key}="${value}"`)
      .join("");
    return `<${this.tagName}${attributes}>${this.textContent}${this.children
      .map((child) => child.markup())
      .join("")}</${this.tagName}>`;
  }
}

class FakeDocument {
  constructor() {
    this.activeElement = null;
  }
  /** @param {string} _namespace @param {string} tagName */
  createElementNS(_namespace, tagName) {
    return new FakeElement(this, tagName);
  }
}

/**
 * @param {Record<string, any>} bundle
 * @returns {string}
 */
function bundleMarkup(bundle) {
  return [bundle.root, bundle.shieldRoot]
    .map((node) => (node === null ? "" : node.markup()))
    .join("");
}

const EXPIRY_ON = Object.freeze({
  ...DEFAULT_VISUAL_FILTER_STATE,
  spawn_shield_expiry: true,
});

test("a seen champion whose shield ended this tick gets a ward lift, timed by the switch", async () => {
  const { scene, transform } = await loadScene("replay_oracle");
  const agent = scene.agents.find(
    (/** @type {Record<string, any>} */ row) => row.public_agent_id === "agent-slot-5",
  );
  const key = agent.presentation_key;
  const beats = { rows: [shieldEndedRow(key)] };
  const liftOf = (/** @type {Record<string, any>} */ input) =>
    champions
      .records({ scene, transform, showSpawnShield: true, beats, ...input })
      .find((record) => record.presentationKey === key)?.wardLift;
  assert.equal(DEFAULT_VISUAL_FILTER_STATE.spawn_shield_expiry, false);
  assert.deepEqual({ ...liftOf({}) }, { fade: champions.WARD_LIFT.cut });
  assert.deepEqual(
    { ...liftOf({ visualFilters: EXPIRY_ON }) },
    { fade: champions.WARD_LIFT.fade },
  );
  assert.equal(liftOf({ showSpawnShield: false }), null);
  assert.equal(liftOf({ beats: { rows: [] } }), null);
  assert.equal(liftOf({ beats: undefined }), null);
  const warded = {
    ...scene,
    agents: scene.agents.map((/** @type {Record<string, any>} */ row) =>
      row.presentation_key === key ? { ...row, spawn_shield_remaining: 2 } : row,
    ),
  };
  assert.equal(liftOf({ scene: warded }), null, "a ward that still shows");
  const others = champions
    .records({ scene, transform, showSpawnShield: true, beats })
    .filter((record) => record.presentationKey !== key);
  assert.ok(others.every((record) => record.wardLift === null));
});

test("in Agent POV only a body seen at tick start lifts its ward", async () => {
  const { scene, transform } = await loadScene("replay_shared_obs_agent_pov");
  assert.equal(scene.audience, "agent_pov");
  const agent = scene.agents.find(
    (/** @type {Record<string, any>} */ row) => row.alive === true,
  );
  const key = agent.presentation_key;
  const input = {
    scene,
    transform,
    showSpawnShield: true,
    visualFilters: EXPIRY_ON,
    beats: { rows: [shieldEndedRow(key)] },
  };
  const liftOf = (/** @type {Record<string, any>} */ extra) =>
    champions
      .records({ ...input, ...extra })
      .find((record) => record.presentationKey === key)?.wardLift;
  assert.equal(liftOf({}), null, "Agent POV defaults to unseen at tick start");
  assert.equal(liftOf({ startSeen: () => false }), null);
  assert.deepEqual({ ...liftOf({ startSeen: () => true }) }, { fade: 200 });
});

test("the lift holds the ward until Settle, then the band closes", async () => {
  const { scene, transform } = await loadScene("replay_oracle");
  const agent = scene.agents.find(
    (/** @type {Record<string, any>} */ row) => row.public_agent_id === "agent-slot-5",
  );
  const key = agent.presentation_key;
  const input = { scene, transform, showSpawnShield: true, visualFilters: EXPIRY_ON };
  const pick = (/** @type {ReadonlyArray<Record<string, any>>} */ rows) => {
    const record = rows.find((row) => row.presentationKey === key);
    assert.ok(record);
    return record;
  };
  const lifting = pick(
    champions.records({ ...input, beats: { rows: [shieldEndedRow(key)] } }),
  );
  assert.equal(lifting.teamKey, "team-b");
  assert.equal(lifting.geometry.doubleBand, true);
  const doc = /** @type {any} */ (new FakeDocument());
  const [bundle] = champions.build([lifting], doc);
  champions.update([bundle], [lifting]);
  const lift = /** @type {FakeElement} */ (/** @type {any} */ (bundle.shieldRoot));
  assert.ok(lift);
  assert.equal(lift.getAttribute("class"), "agent-ward-lift");
  assert.equal(lift.find("agent-spawn-shield").length, 0);
  assert.equal(lift.getAttribute("aria-hidden"), "true");
  assert.equal(lift.getAttribute("pointer-events"), "none");
  assert.equal(lift.getAttribute("tabindex"), null);
  assert.equal(lift.getAttribute("role"), null);
  assert.equal(lift.getAttribute("data-team"), "team-b");
  assert.equal(lift.getAttribute("opacity"), "0");
  assert.equal(lift.getAttribute("data-beat-hold"), "settle");
  assert.equal(lift.getAttribute("data-beat-side"), "before");
  assert.equal(lift.getAttribute("data-beat-key"), key);
  assert.equal(lift.getAttribute("data-beat-fade"), "200");
  assert.equal(lift.find("agent-spawn-shield__veil").length, 1);
  assert.equal(lift.find("agent-spawn-shield__ring").length, 1);
  assert.equal(lift.find("agent-spawn-shield__split").length, 1);
  const diamonds = lift.find("agent-spawn-shield__diamonds");
  assert.equal(diamonds.length, 1);
  assert.equal((diamonds[0].getAttribute("d") ?? "").split("Z").length - 1, 1);
  for (const ring of [bundle.teamRing, bundle.ringSplit]) {
    assert.equal(ring.getAttribute("display"), null);
    assert.equal(ring.getAttribute("data-beat-hold"), "settle");
    assert.equal(ring.getAttribute("data-beat-side"), "after");
    assert.equal(ring.getAttribute("data-beat-fade"), "200");
  }
  assert.equal(bundle.root.getAttribute("opacity"), null, "END: full strength");
  assert.equal(
    bundle.root.getAttribute("data-beat-from"),
    String(champions.WARD_LIFT.fromOpacity),
  );
  assert.equal(bundle.root.getAttribute("data-beat-hold"), "settle");
  assert.equal(bundle.root.getAttribute("data-beat-side"), "after");
  const markup = bundleMarkup(bundle);
  assert.doesNotMatch(markup, /url\(|gradient|<mask|<filter|<pattern|<image/iu);
  assert.doesNotMatch(markup, / id="| style="/u);

  const cut = pick(
    champions.records({
      ...input,
      visualFilters: DEFAULT_VISUAL_FILTER_STATE,
      beats: { rows: [shieldEndedRow(key)] },
    }),
  );
  champions.update([bundle], [cut]);
  assert.equal(bundle.shieldRoot?.getAttribute("data-beat-fade"), "1");
  assert.equal(bundle.teamRing.getAttribute("data-beat-fade"), "1");

  const quiet = pick(champions.records({ ...input, beats: { rows: [] } }));
  champions.update([bundle], [quiet]);
  const fresh = champions.build([quiet], /** @type {any} */ (new FakeDocument()))[0];
  champions.update([fresh], [quiet]);
  assert.equal(bundle.shieldRoot, null);
  assert.equal(bundleMarkup(bundle), bundleMarkup(fresh));
});

test("a ward group and a lift group are never reused for each other", async () => {
  const { scene, transform } = await loadScene("replay_oracle");
  const agent = scene.agents.find(
    (/** @type {Record<string, any>} */ row) => row.public_agent_id === "agent-slot-1",
  );
  const key = agent.presentation_key;
  const wardedScene = {
    ...scene,
    agents: scene.agents.map((/** @type {Record<string, any>} */ row) =>
      row.presentation_key === key ? { ...row, spawn_shield_remaining: 1 } : row,
    ),
  };
  const pick = (/** @type {ReadonlyArray<Record<string, any>>} */ rows) => {
    const record = rows.find((row) => row.presentationKey === key);
    assert.ok(record);
    return record;
  };
  const base = { transform, showSpawnShield: true, visualFilters: EXPIRY_ON };
  const ward = pick(champions.records({ ...base, scene: wardedScene }));
  const lifting = pick(
    champions.records({ ...base, scene, beats: { rows: [shieldEndedRow(key)] } }),
  );
  assert.ok(ward.ward);
  assert.ok(lifting.wardLift);
  const doc = /** @type {any} */ (new FakeDocument());
  const [bundle] = champions.build([ward], doc);
  champions.update([bundle], [ward]);
  const wardRoot = bundle.shieldRoot;
  assert.equal(wardRoot?.getAttribute("class"), "agent-spawn-shield");
  assert.equal(wardRoot?.getAttribute("tabindex"), "0");
  champions.update([bundle], [lifting]);
  assert.notEqual(bundle.shieldRoot, wardRoot);
  assert.equal(bundle.shieldRoot?.getAttribute("class"), "agent-ward-lift");
  assert.equal(bundle.shieldRoot?.getAttribute("data-team"), "team-a");
  assert.equal(bundle.ringSplit.getAttribute("display"), "none", "Team A: no split");
  const liftRoot = bundle.shieldRoot;
  champions.update([bundle], [ward]);
  assert.notEqual(bundle.shieldRoot, liftRoot);
  assert.equal(bundle.shieldRoot?.getAttribute("class"), "agent-spawn-shield");
  assert.equal(bundle.teamRing.getAttribute("data-beat-hold"), null);
  assert.equal(bundle.root.getAttribute("opacity"), "0.8");
  assert.equal(bundle.root.getAttribute("data-beat-hold"), null);
});
