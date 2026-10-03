/**
 * @file Checks the V5 Class Seals in the Champions look (`src/looks/champions.js`).
 *
 * Contracts checked: classSealPaths and tokenGeometry(R).sealRadius give the
 * spec's seal shapes and radius (Mage octagon, Warrior boss and rivets, Hunter
 * quatrefoil, Rogue cushion lozenge, Priest sun); the switch class_seals is off
 * by default and then every token keeps its plain halo and an empty seal group;
 * switched on, every heraldic token of 18 px or more stands on its class seal
 * instead of the halo (engraved from 22 px), and a small token or an unknown
 * class falls back to the halo (a stale saved Heraldic Tokens Off is ignored,
 * because tokens are no longer a switch); a seal is
 * awakened only while the Ultimate cooldown is 0 and Cooldown Effects is on (a
 * missing cooldown is dormant) and steps back under a Rune Zones mark only while
 * Status Marks is on; the audience's own cooldown and status rows give the beat
 * cues (spend at Impact, awaken at Settle with the Mage flare pulse and the
 * Warrior's rivet pairs, a mark at Settle, and the three-look Impact plus Settle
 * tick), and a quiet tick on the same nodes holds exactly a fresh build's DOM; in
 * Agent POV no cue shows unless the caller says the body was seen at tick start,
 * and an unseen champion gets no record and no seal even when a row names it; a
 * fallen champion keeps a cold ash outline (never on its Fallen Step Aside coin)
 * and a falling one holds its seal in its held living look; the markup is
 * export-safe, the worst seal tick stays within BUDGET, and the paint part
 * class_seal belongs to class_seals.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  authorizedPresentationIncomingRows,
  authorizedPresentationSceneView,
} from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { beatView } from "../src/choreography-plan.js";
import { createViewportTransform } from "../src/layout.js";
import * as champions from "../src/looks/champions.js";
import {
  classifyVisualPaintPart,
  DEFAULT_VISUAL_FILTER_STATE,
} from "../src/visual-filters.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

const SEALS_ON = Object.freeze({ ...DEFAULT_VISUAL_FILTER_STATE, class_seals: true });

/** @param {string} name */
async function loadFrame(name) {
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
  const beats = {
    rows: authorizedPresentationIncomingRows(frame),
    view: beatView(frame, "end"),
  };
  return { frame, scene, transform, beats };
}

/**
 * @param {Record<string, any>} scene
 * @param {string} publicId
 * @param {Record<string, any>} fields
 */
function withAgent(scene, publicId, fields) {
  const copy = structuredClone(scene);
  copy.agents = copy.agents.map((/** @type {Record<string, any>} */ agent) =>
    agent.public_agent_id === publicId ? { ...agent, ...fields } : agent,
  );
  return copy;
}

/**
 * @param {ReadonlyArray<Record<string, any>>} rows
 * @param {string} publicId
 */
function recordFor(rows, publicId) {
  const row = rows.find((entry) => entry.agent.public_agent_id === publicId);
  assert.ok(row, publicId);
  return row;
}

/**
 * @param {Record<string, any>} scene
 * @param {string} publicId
 * @returns {string}
 */
function keyOf(scene, publicId) {
  const agent = scene.agents.find(
    (/** @type {Record<string, any>} */ row) => row.public_agent_id === publicId,
  );
  assert.ok(agent, publicId);
  return agent.presentation_key;
}

/**
 * @param {string} kind
 * @param {string} key
 * @param {Record<string, any>} [extra]
 */
function eventRow(kind, key, extra = {}) {
  const anchor = { presentation_key: key, phase: "successor" };
  const payload = kind.startsWith("cooldown_")
    ? { event_kind: kind, agent_anchor: anchor, ...extra }
    : { event_kind: kind, recipient_anchor: anchor, ...extra };
  return { id: `${kind}:${key}`, kind, vocabulary: "event", payload };
}

const SLOWED = Object.freeze({
  status_id: "warrior_charge_slow",
  token_id: "warrior_charge_slow",
  magnitude: 0.5,
  duration: 2,
});

class FakeElement {
  /**
   * @param {FakeDocument} doc
   * @param {string} tagName
   */
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
    this.classList = {
      add: (/** @type {string} */ name) =>
        this.setAttribute(
          "class",
          `${this.getAttribute("class") ?? ""} ${name}`.trim(),
        ),
    };
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
  getBoundingClientRect() {
    return { x: 0, y: 0, width: 0, height: 0, left: 0, top: 0, right: 0, bottom: 0 };
  }
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

/** @param {Record<string, any>} record */
function drawn(record) {
  const doc = /** @type {any} */ (new FakeDocument());
  const [bundle] = champions.build([record], doc);
  champions.update([bundle], [record]);
  return {
    bundle,
    root: /** @type {FakeElement} */ (/** @type {any} */ (bundle.root)),
  };
}

/** @param {string} markup */
function assertExportSafe(markup) {
  assert.doesNotMatch(
    markup,
    /url\(|gradient|<mask|<filter|<pattern|<clipPath|<image/iu,
  );
  assert.doesNotMatch(markup, / id="| style="| filter="|mix-blend/u);
}

test("the seal radius and the five seal shapes follow the spec", () => {
  assert.ok(Math.abs(champions.tokenGeometry(20).sealRadius - 10.8) < 0.005);
  assert.ok(Math.abs(champions.tokenGeometry(23).sealRadius - 13.165) < 0.005);
  assert.ok(Math.abs(champions.tokenGeometry(29.4).sealRadius - 17.09) < 0.01);
  assert.ok(Math.abs(champions.tokenGeometry(36).sealRadius - 21.02) < 0.005);
  for (const R of [18, 23, 36]) {
    const geometry = champions.tokenGeometry(R);
    assert.ok(geometry.sealRadius < geometry.lifeRadius - geometry.lifeWidth / 2);
  }
  const mage = champions.classSealPaths("mage", 23);
  assert.match(mage.plate, /^M5\.038 -12\.163 L12\.163 -5\.038 /u);
  assert.equal(mage.facets.split("M").length - 1, 8);
  const warrior = champions.classSealPaths("warrior", 23);
  assert.match(warrior.plate, /^M-12\.507 0a12\.507 12\.507 /u);
  assert.equal(warrior.pairs.length, 4);
  assert.match(warrior.rivets, /a1\.035 1\.035 /u);
  const hunter = champions.classSealPaths("hunter", 23);
  assert.match(hunter.plate, /^M0 -13\.165 A7\.899 7\.899 0 0 1 7\.559 -7\.559 /u);
  const rogue = champions.classSealPaths("rogue", 23);
  assert.match(rogue.plate, /^M0 -13\.165 A22\.538 22\.538 0 0 1 13\.165 0 /u);
  assert.match(rogue.honing, /^M0 -9\.742 L9\.742 0 /u);
  const priest = champions.classSealPaths("priest", 23);
  assert.match(priest.plate, /^M0 -13\.165 L2\.862 -10\.682 /u);
  assert.equal(priest.plate.split("L").length, 24);
  assert.equal(champions.classSealPaths("mage", 23), mage, "cached per radius");
  assert.notEqual(champions.classSealPaths("mage", 23.01), mage);
});

test("off by default the halo stays; switched on, each class stands on its seal", async () => {
  const { scene, transform } = await loadFrame("replay_oracle");
  const plain = champions.records({ scene, transform });
  assert.equal(DEFAULT_VISUAL_FILTER_STATE.class_seals, false);
  for (const record of plain) {
    assert.equal(record.seal, null);
    const { bundle, root } = drawn(record);
    assert.equal(bundle.halo.getAttribute("display"), null);
    assert.equal(root.find("champion-seal")[0].children.length, 0);
  }
  const sealed = champions.records({ scene, transform, visualFilters: SEALS_ON });
  const classes = new Set();
  for (const record of sealed) {
    assert.ok(record.radius >= 22);
    assert.deepEqual({ ...record.seal.end }, { awake: true, dim: false });
    assert.equal(record.seal.engraved, true);
    assert.equal(record.seal.awakeBeat, null);
    const { bundle, root } = drawn(record);
    assert.equal(bundle.halo.getAttribute("display"), "none");
    const looks = root.find("champion-seal__look");
    assert.equal(looks.length, 1);
    assert.equal(looks[0].getAttribute("data-class"), record.classKey);
    assert.equal(looks[0].getAttribute("data-awake"), "true");
    const plate = root.find("champion-seal__plate")[0];
    assert.equal(plate.getAttribute("stroke-opacity"), "1");
    assertExportSafe(root.markup());
    classes.add(record.classKey);
  }
  assert.deepEqual([...classes].sort(), [
    "hunter",
    "mage",
    "priest",
    "rogue",
    "warrior",
  ]);
  const priest = drawn(recordFor(sealed, "agent-slot-2")).root;
  assert.equal(
    priest.find("champion-seal__plate")[0].getAttribute("fill-opacity"),
    "0.2",
  );
  const warrior = drawn(recordFor(sealed, "agent-slot-1")).root;
  assert.equal(warrior.find("champion-seal__rivets-lit").length, 1);
  assert.equal(
    warrior.find("champion-seal__plate")[0].getAttribute("stroke-width"),
    "1.8",
  );
});

test("the seal falls back truthfully: small tokens and unknown classes", async () => {
  const { scene, transform } = await loadFrame("replay_oracle");
  // Heraldic Tokens is no longer a switch: a stale saved Off is ignored.
  const staleOff = champions.records({
    scene,
    transform,
    visualFilters: { ...SEALS_ON, heraldic_tokens: false },
  });
  assert.ok(staleOff.some((record) => record.seal !== null));
  const small = createViewportTransform({
    worldWidth: scene.map.width,
    worldHeight: scene.map.height,
    viewportWidth: 600,
    viewportHeight: 400,
    padding: 24,
  });
  const smallRows = champions.records({
    scene,
    transform: small,
    visualFilters: SEALS_ON,
  });
  assert.ok(smallRows[0].radius < 18);
  assert.ok(smallRows.every((record) => record.seal === null));
  const unknown = withAgent(scene, "agent-slot-0", { class_id: 9 });
  const unknownRow = recordFor(
    champions.records({ scene: unknown, transform, visualFilters: SEALS_ON }),
    "agent-slot-0",
  );
  assert.equal(unknownRow.seal, null);
  assert.equal(drawn(unknownRow).bundle.halo.getAttribute("display"), null);
  const middle = createViewportTransform({
    worldWidth: scene.map.width,
    worldHeight: scene.map.height,
    viewportWidth: 800,
    viewportHeight: 500,
    padding: 24,
  });
  const middleRow = champions.records({
    scene,
    transform: middle,
    visualFilters: SEALS_ON,
  })[0];
  assert.ok(middleRow.radius >= 18 && middleRow.radius < 22);
  assert.equal(middleRow.seal.engraved, false);
  assert.equal(drawn(middleRow).root.find("champion-seal__engraving").length, 0);
});

test("awake follows the cooldown and Cooldown Effects; dim follows Rune Zones marks", async () => {
  const { scene, transform } = await loadFrame("replay_oracle");
  const charging = withAgent(scene, "agent-slot-0", { ultimate_cooldown: 12 });
  assert.equal(
    recordFor(
      champions.records({ scene: charging, transform, visualFilters: SEALS_ON }),
      "agent-slot-0",
    ).seal.end.awake,
    false,
  );
  const missing = withAgent(scene, "agent-slot-0", {
    ultimate_cooldown: null,
    ultimate_cooldown_remaining: null,
  });
  const missingRow = recordFor(
    champions.records({ scene: missing, transform, visualFilters: SEALS_ON }),
    "agent-slot-0",
  );
  assert.equal(missingRow.seal.end.awake, false, "a missing cooldown is dormant");
  assert.equal(
    drawn(missingRow)
      .root.find("champion-seal__plate")[0]
      .getAttribute("stroke-opacity"),
    "0.45",
  );
  const noCooldown = champions.records({
    scene,
    transform,
    visualFilters: { ...SEALS_ON, cooldown_effects: false },
  });
  assert.ok(noCooldown.every((record) => record.seal.end.awake === false));

  const slowed = withAgent(scene, "agent-slot-5", { statuses: [SLOWED] });
  const dimRow = recordFor(
    champions.records({ scene: slowed, transform, visualFilters: SEALS_ON }),
    "agent-slot-5",
  );
  assert.deepEqual({ ...dimRow.seal.end }, { awake: true, dim: true });
  const dimRoot = drawn(dimRow).root;
  assert.equal(
    dimRoot.find("champion-seal__plate")[0].getAttribute("stroke-opacity"),
    "0.5",
  );
  assert.equal(dimRoot.find("champion-seal__engraving").length, 0, "engraving hidden");
  const marksOff = recordFor(
    champions.records({
      scene: slowed,
      transform,
      visualFilters: { ...SEALS_ON, status_marks: false },
    }),
    "agent-slot-5",
  );
  assert.equal(marksOff.seal.end.dim, false);
  // Blessing of Freedom alone draws no mark (its wings show only when it matters).
  assert.equal(
    recordFor(
      champions.records({ scene, transform, visualFilters: SEALS_ON }),
      "agent-slot-1",
    ).seal.end.dim,
    false,
  );
  const dimWarrior = recordFor(
    champions.records({
      scene: withAgent(scene, "agent-slot-1", { statuses: [SLOWED] }),
      transform,
      visualFilters: SEALS_ON,
    }),
    "agent-slot-1",
  );
  const rivets = drawn(dimWarrior).root;
  assert.equal(rivets.find("champion-seal__rivets")[0].getAttribute("opacity"), "0.3");
  assert.equal(
    rivets.find("champion-seal__rivets-lit")[0].getAttribute("opacity"),
    "0.5",
  );
});

test("the audience's own rows time the spend, the awakening and a new mark", async () => {
  const { scene, transform } = await loadFrame("replay_oracle");
  const mageKey = keyOf(scene, "agent-slot-0");
  const spentScene = withAgent(scene, "agent-slot-0", { ultimate_cooldown: 30 });
  const spent = recordFor(
    champions.records({
      scene: spentScene,
      transform,
      visualFilters: SEALS_ON,
      beats: { rows: [eventRow("cooldown_started", mageKey)] },
    }),
    "agent-slot-0",
  );
  assert.equal(spent.seal.awakeBeat, "impact");
  assert.equal(spent.seal.start.awake, true);
  const spentDrawn = drawn(spent);
  const [before, after] = spentDrawn.root.find("champion-seal__look");
  assert.equal(before.getAttribute("data-beat-hold"), "impact");
  assert.equal(before.getAttribute("data-beat-side"), "before");
  assert.equal(before.getAttribute("opacity"), "0");
  assert.equal(before.getAttribute("data-beat-fade"), "120");
  assert.equal(before.getAttribute("data-awake"), "true");
  assert.equal(after.getAttribute("data-beat-side"), "after");
  assert.equal(after.getAttribute("data-awake"), "false");

  const readied = recordFor(
    champions.records({
      scene,
      transform,
      visualFilters: SEALS_ON,
      beats: { rows: [eventRow("cooldown_ready", mageKey)] },
    }),
    "agent-slot-0",
  );
  assert.equal(readied.seal.awakeBeat, "settle");
  const readiedRoot = drawn(readied).root;
  const flare = readiedRoot.find("champion-seal__flare")[0];
  assert.equal(flare.getAttribute("data-beat-pulse"), "0.95");
  assert.equal(flare.getAttribute("data-beat-hold"), "settle");
  assert.equal(flare.getAttribute("opacity"), "0");
  assert.equal(
    readiedRoot.find("champion-seal__look")[1].getAttribute("data-beat-fade"),
    "240",
  );

  const warriorKey = keyOf(scene, "agent-slot-1");
  const warrior = recordFor(
    champions.records({
      scene,
      transform,
      visualFilters: SEALS_ON,
      beats: { rows: [eventRow("cooldown_ready", warriorKey)] },
    }),
    "agent-slot-1",
  );
  const pairs = drawn(warrior).root.find("champion-seal__rivets-lit");
  assert.deepEqual(
    pairs.map((pair) => pair.getAttribute("data-beat-fade")),
    ["120", "150", "180", "210"],
  );
  assert.ok(pairs.every((pair) => pair.getAttribute("data-beat-hold") === "settle"));

  const slowedScene = withAgent(scene, "agent-slot-5", { statuses: [SLOWED] });
  const hunterKey = keyOf(scene, "agent-slot-5");
  const marked = recordFor(
    champions.records({
      scene: slowedScene,
      transform,
      visualFilters: SEALS_ON,
      beats: {
        rows: [
          eventRow("status_applied", hunterKey, { status_id: "warrior_charge_slow" }),
        ],
      },
    }),
    "agent-slot-5",
  );
  assert.equal(marked.seal.dimBeat, "settle");
  assert.equal(marked.seal.start.dim, false);
  const markedLooks = drawn(marked).root.find("champion-seal__look");
  assert.equal(markedLooks[0].getAttribute("data-beat-hold"), "settle");
  assert.equal(markedLooks[0].getAttribute("data-beat-fade"), "200");
  assert.equal(markedLooks[1].getAttribute("data-dim"), "true");

  const both = recordFor(
    champions.records({
      scene: withAgent(spentScene, "agent-slot-0", {
        statuses: [{ status_id: "mage_burst_damage_amplification", duration: 3 }],
      }),
      transform,
      visualFilters: SEALS_ON,
      beats: {
        rows: [
          eventRow("cooldown_started", mageKey),
          eventRow("status_applied", mageKey, {
            status_id: "mage_burst_damage_amplification",
          }),
        ],
      },
    }),
    "agent-slot-0",
  );
  assert.equal(both.seal.awakeBeat, "impact");
  assert.equal(both.seal.dimBeat, "settle");
  const bothDrawn = drawn(both);
  const stage = bothDrawn.root.find("champion-seal__stage")[0];
  assert.equal(stage.getAttribute("data-beat-hold"), "impact");
  assert.equal(stage.getAttribute("data-beat-side"), "after");
  const middle = stage.find("champion-seal__look")[0];
  assert.equal(middle.getAttribute("data-beat-hold"), "settle");
  assert.equal(middle.getAttribute("data-beat-side"), "before");
  assert.equal(middle.getAttribute("data-awake"), "false");
  assert.equal(middle.getAttribute("data-dim"), "false");
  assertExportSafe(bothDrawn.root.markup());
  const holds = bothDrawn.root
    .all()
    .filter((node) => node.hasAttribute("data-beat-hold"));
  assert.ok(holds.length <= champions.BUDGET.animationsPerRecord);

  // A quiet tick on the same nodes holds exactly what a fresh build holds.
  const quiet = recordFor(
    champions.records({ scene: spentScene, transform, visualFilters: SEALS_ON }),
    "agent-slot-0",
  );
  champions.update([bothDrawn.bundle], [quiet]);
  assert.equal(bothDrawn.root.markup(), drawn(quiet).root.markup());
  assert.doesNotMatch(bothDrawn.root.markup(), /data-beat-hold/u);
});

test("switching Class Seals on and off on retained nodes equals a fresh render", async () => {
  const { scene, transform, beats } = await loadFrame("replay_oracle");
  const on = champions.records({ scene, transform, beats, visualFilters: SEALS_ON });
  const off = champions.records({ scene, transform, beats });
  for (const [index, record] of on.entries()) {
    const { bundle, root } = drawn(off[index]);
    champions.update([bundle], [record]);
    assert.equal(root.markup(), drawn(record).root.markup());
    champions.update([bundle], [off[index]]);
    assert.equal(root.markup(), drawn(off[index]).root.markup());
    assert.doesNotMatch(root.markup(), /champion-seal__/u);
  }
});

test("Agent POV shows only seen bodies' seals and no cue unless seen at start", async () => {
  const pov = await loadFrame("replay_shared_obs_agent_pov");
  const oracle = await loadFrame("replay_oracle");
  const povIds = new Set(
    pov.scene.agents.map(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id,
    ),
  );
  const unseen = oracle.scene.agents.filter(
    (/** @type {Record<string, any>} */ agent) => !povIds.has(agent.public_agent_id),
  );
  assert.ok(unseen.length > 0, "the fixture has champions the POV team cannot see");
  const rows = [
    ...unseen.map((/** @type {Record<string, any>} */ agent) =>
      eventRow("cooldown_ready", agent.presentation_key),
    ),
  ];
  const povRows = champions.records({
    scene: pov.scene,
    transform: pov.transform,
    visualFilters: SEALS_ON,
    beats: { rows },
  });
  assert.deepEqual(
    povRows.map((record) => record.agent.public_agent_id).sort(),
    [...povIds].sort(),
  );
  for (const record of povRows) {
    assert.equal(record.audience, "agent_pov");
    assert.ok(record.seal);
    assert.equal(record.seal.awakeBeat, null);
    assert.doesNotMatch(drawn(record).root.markup(), /data-beat-hold/u);
  }
  const key = povRows[0].presentationKey;
  const cueRows = [eventRow("cooldown_ready", key)];
  const unknownStart = champions.records({
    scene: pov.scene,
    transform: pov.transform,
    visualFilters: SEALS_ON,
    beats: { rows: cueRows },
  });
  assert.equal(unknownStart[0].seal.awakeBeat, null, "not known to be seen at start");
  const seenStart = champions.records({
    scene: pov.scene,
    transform: pov.transform,
    visualFilters: SEALS_ON,
    beats: { rows: cueRows },
    startSeen: (presentationKey) => presentationKey === key,
  });
  assert.equal(seenStart[0].seal.awakeBeat, "settle");
  const hidden = withAgent(pov.scene, povRows[0].agent.public_agent_id, {
    ultimate_cooldown: null,
    ultimate_cooldown_remaining: null,
  });
  const hiddenRow = champions.records({
    scene: hidden,
    transform: pov.transform,
    visualFilters: SEALS_ON,
  })[0];
  assert.equal(hiddenRow.seal.end.awake, false, "no authorized cooldown: dormant");
});

test("a fallen champion keeps a cold ash outline; a falling one holds its seal", async () => {
  const { scene, transform } = await loadFrame("replay_oracle");
  const dead = withAgent(scene, "agent-slot-6", {
    alive: false,
    life_state: "corpse",
    current_health: 0,
  });
  const fallen = recordFor(
    champions.records({ scene: dead, transform, visualFilters: SEALS_ON }),
    "agent-slot-6",
  );
  assert.equal(fallen.state, "fallen");
  const { bundle, root } = drawn(fallen);
  const ash = root.find("champion-seal__ash");
  assert.equal(ash.length, 1);
  assert.equal(ash[0].getAttribute("opacity"), "0.35");
  assert.ok(
    /** @type {FakeElement} */ (/** @type {any} */ (bundle.fallen)).children.includes(
      ash[0],
    ),
  );
  assert.equal(root.find("champion-seal")[0].children.length, 0);
  assertExportSafe(root.markup());
  const noSeals = recordFor(
    champions.records({ scene: dead, transform }),
    "agent-slot-6",
  );
  assert.equal(drawn(noSeals).root.find("champion-seal__ash").length, 0);

  const covered = withAgent(scene, "agent-slot-0", {
    alive: false,
    life_state: "corpse",
    current_health: 0,
    position: [1.6, 3.75],
  });
  const coin = recordFor(
    champions.records({ scene: covered, transform, visualFilters: SEALS_ON }),
    "agent-slot-0",
  );
  assert.ok(coin.aside);
  assert.equal(
    drawn(coin).root.find("champion-seal__ash").length,
    0,
    "no seal on the coin",
  );

  const key = keyOf(scene, "agent-slot-6");
  const falling = recordFor(
    champions.records({
      scene: dead,
      transform,
      visualFilters: SEALS_ON,
      beats: {
        rows: [
          eventRow("status_cleared_by_new_death", key, {
            status_id: "rogue_poison_anti_heal",
          }),
        ],
        view: { life: [{ presentationKey: key, before: 60 }], returns: [] },
      },
    }),
    "agent-slot-6",
  );
  assert.equal(falling.state, "falling");
  assert.deepEqual({ ...falling.seal.start }, { awake: true, dim: true });
  const fallingRoot = drawn(falling);
  const fallingGroup = /** @type {FakeElement} */ (
    /** @type {any} */ (fallingRoot.bundle.falling)
  );
  const held = fallingGroup.find("champion-seal__look");
  assert.equal(held.length, 1);
  assert.equal(held[0].getAttribute("data-beat-hold"), null);
  assert.equal(fallingGroup.find("champion-halo").length, 0);
});

test("the worst seal tick stays inside BUDGET and the paint part is class_seals", async () => {
  const { scene, transform } = await loadFrame("replay_oracle");
  const key = keyOf(scene, "agent-slot-1");
  const warded = withAgent(scene, "agent-slot-1", {
    ultimate_cooldown: 30,
    spawn_shield_remaining: 5,
    current_health: 100,
    statuses: [SLOWED],
  });
  const record = recordFor(
    champions.records({
      scene: warded,
      transform,
      visualFilters: SEALS_ON,
      showSpawnShield: true,
      beats: {
        rows: [
          eventRow("recipient_health_resolution", key, {
            transition_start_health: 150,
            health_after_combat_resolution: 100,
          }),
          eventRow("cooldown_started", key),
          eventRow("status_applied", key, { status_id: "warrior_charge_slow" }),
        ],
      },
    }),
    "agent-slot-1",
  );
  assert.equal(record.seal.awakeBeat, "impact");
  assert.equal(record.seal.dimBeat, "settle");
  assert.equal(record.life.change, "loss");
  assert.ok(record.ward);
  const { bundle } = drawn(record);
  const wrapper = /** @type {FakeElement} */ (/** @type {any} */ (bundle.wrapper));
  // Icon glyph strokes inside an svg emblem are not counted (see BUDGET).
  /** @type {(node: FakeElement) => number} */
  const countNodes = (node) =>
    node.tagName === "svg"
      ? 1
      : 1 + node.children.reduce((sum, child) => sum + countNodes(child), 0);
  const count = countNodes(wrapper);
  assert.ok(count <= champions.BUDGET.nodesPerRecord, `${count} nodes`);
  assert.deepEqual(
    champions.PAINT_PARTS.find((entry) => entry.filter === "class_seals")?.part,
    { surface: "durable", kind: "class_seal" },
  );
  assert.equal(
    classifyVisualPaintPart({ surface: "durable", kind: "class_seal" }),
    "class_seals",
  );
});
