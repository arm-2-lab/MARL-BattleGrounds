/**
 * @file Checks the Champions look (Life Vessels) in `src/looks/champions.js`.
 *
 * Contracts checked: tokenGeometry gives the store's token sizes (team ring
 * max(3.5, 0.155R) inside the footprint, vessel R minus the ring, halo 0.34R,
 * emblem clamp(0.8R, 9, 26), ash coin 0.46R); records() is pure, gives one record
 * per scene body with look "heraldic" and a vessel by default, and the ivory
 * life ring when Life Vessels is off (Heraldic Tokens is not a switch, so a
 * stale saved Off still gives "heraldic"); life before the Impact beat comes from the audience's own
 * recipient_health_resolution rows, with a hit flash, and a hitstop shiver only
 * on an Ultimate's or a killing blow's target (0 px for the Warrior); a champion
 * the audience saw die is "falling" and keeps its life before the fall until the
 * Fall beat; Low Life Warning marks 30% or less; Agent POV records hold only the
 * POV scene's bodies and missing health gives an unknown life (dotted ring);
 * Spawn Ward gives up to three diamond marks; Fallen Step Aside moves a covered
 * corpse's ash coin to the covering champion's rim; build() and update() make
 * export-safe SVG (no url(), gradient, mask, filter, pattern, clipPath, image,
 * style or id) and write the beat-hold attributes the choreographer reads (fall,
 * impact, pulse, shiver) and remove them again on a quiet tick; the module never
 * reads the researcher view or the match summary, and imports neither the
 * presentation adapter, the normalizer nor the plan (its caller passes the
 * audience's incoming rows and beatView result as plain beat facts).
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
import { DEFAULT_VISUAL_FILTER_STATE } from "../src/visual-filters.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/**
 * @param {Record<string, any>} branch
 * @param {Record<string, any>} event
 */
function insertEvent(branch, event) {
  const prefix = branch.events[0].event_id.replace(/\d{4}$/u, "");
  const at = branch.events.findIndex(
    (/** @type {Record<string, any>} */ row) => row.phase_rank > event.phase_rank,
  );
  branch.events.splice(at < 0 ? branch.events.length : at, 0, event);
  branch.events.forEach(
    (/** @type {Record<string, any>} */ row, /** @type {number} */ index) => {
      row.ordinal = index;
      row.event_id = `${prefix}${String(index).padStart(4, "0")}`;
    },
  );
  branch.ordered_event_ids = branch.events.map(
    (/** @type {Record<string, any>} */ row) => row.event_id,
  );
  branch.ordered_event_kinds = branch.events.map(
    (/** @type {Record<string, any>} */ row) => row.event_kind,
  );
  branch.event_count = branch.events.length;
}

/**
 * @param {string} name
 * @param {{start?: number, after?: number, died?: boolean, ultimate?: boolean}} [change]
 */
async function loadFrame(name, change) {
  const raw = structuredClone(fixture.presentations[name]);
  const branch = name.includes("agent_pov") ? raw.visual_events : raw.latest_events;
  if (change) {
    const resolution = branch.events.find(
      (/** @type {Record<string, any>} */ row) =>
        row.event_kind === "recipient_health_resolution",
    );
    const start = change.start ?? 200;
    const after = change.after ?? 150;
    resolution.transition_start_health = start;
    resolution.health_after_combat_resolution = after;
    resolution.realized_net_health_change = after - start;
    if ("total_effective_damage" in resolution) {
      resolution.total_effective_damage = Math.max(0, start - after);
      resolution.total_effective_healing = Math.max(0, after - start);
    }
    if (change.ultimate) {
      const activation = branch.events.find(
        (/** @type {Record<string, any>} */ row) =>
          row.event_kind === "ability_activated",
      );
      activation.ability_component = "ultimate";
    }
    if (change.died) {
      insertEvent(branch, {
        event_kind: "agent_died",
        phase_rank: 90,
        recipient_anchor: { ...resolution.recipient_anchor, phase: "successor" },
      });
    }
  }
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
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

/** @param {ReadonlyArray<Record<string, any>>} rows */
function drawn(rows) {
  const doc = /** @type {any} */ (new FakeDocument());
  const nodes = champions.build(rows, doc);
  champions.update(nodes, rows);
  return nodes.map((bundle) => {
    const wrapper = /** @type {any} */ (bundle.wrapper);
    if (bundle.shieldRoot) wrapper.append(bundle.shieldRoot);
    return { bundle, wrapper: /** @type {FakeElement} */ (wrapper) };
  });
}

/** @param {string} markup */
function assertExportSafe(markup) {
  assert.doesNotMatch(
    markup,
    /url\(|gradient|<mask|<filter|<pattern|<clipPath|<image/iu,
  );
  assert.doesNotMatch(markup, / id="| style="| filter="/u);
}

test("tokenGeometry gives the store's token sizes", () => {
  const twenty = champions.tokenGeometry(20);
  assert.equal(twenty.ringWidth, 3.5);
  assert.equal(twenty.ringRadius, 18.25);
  assert.equal(twenty.vesselRadius, 16.5);
  assert.equal(twenty.emblemSize, 16);
  assert.ok(Math.abs(twenty.haloRadius - 6.8) < 1e-9);
  assert.ok(Math.abs(twenty.coinRadius - 9.2) < 1e-9);
  assert.equal(twenty.doubleBand, true);
  const large = champions.tokenGeometry(29.4);
  assert.ok(Math.abs(large.ringWidth - 4.557) < 1e-9);
  assert.ok(Math.abs(large.vesselRadius - 24.843) < 1e-9);
  assert.equal(champions.tokenGeometry(40).emblemSize, 26);
  const small = champions.tokenGeometry(8);
  assert.equal(small.doubleBand, false);
  assert.equal(small.ringWidth, 1.6);
  for (const R of [8, 20, 29.4, 36]) {
    const geometry = champions.tokenGeometry(R);
    assert.ok(geometry.ringRadius + geometry.ringWidth / 2 <= R + 1e-9, "ring inside");
  }
});

test("records default to Life Vessels and fall back to a plain truthful token", async () => {
  const { scene, transform, beats } = await loadFrame("replay_oracle");
  const rows = champions.records({ scene, transform, showSpawnShield: true, beats });
  assert.equal(rows.length, scene.agents.length);
  assert.deepEqual(
    JSON.stringify(rows),
    JSON.stringify(
      champions.records({ scene, transform, showSpawnShield: true, beats }),
    ),
  );
  for (const row of rows) {
    assert.equal(row.look, "heraldic");
    assert.equal(row.lifeStyle, "vessel");
    assert.equal(row.state, "living");
    assert.equal(row.life.known, true);
    assert.equal(row.life.change, "none");
    assert.equal(row.ward, null);
  }
  const noVessels = champions.records({
    scene,
    transform,
    visualFilters: { ...DEFAULT_VISUAL_FILTER_STATE, life_vessels: false },
  });
  assert.ok(
    noVessels.every((row) => row.look === "heraldic" && row.lifeStyle === "arc"),
  );
  const stillHeraldic = champions.records({
    scene,
    transform,
    visualFilters: { ...DEFAULT_VISUAL_FILTER_STATE, heraldic_tokens: false },
  });
  assert.ok(stillHeraldic.every((row) => row.look === "heraldic"));
});

test("life holds its earlier level until Impact, with a flash and a class shiver", async () => {
  const { scene, transform, beats } = await loadFrame("replay_oracle", {
    start: 160,
    after: 110,
    ultimate: true,
  });
  const hurt = withAgent(scene, "agent-slot-1", { current_health: 110 });
  const rows = champions.records({ scene: hurt, transform, beats });
  const row = recordFor(rows, "agent-slot-1");
  assert.equal(row.life.change, "loss");
  assert.equal(row.life.before, 0.8);
  assert.equal(row.life.share, 0.55);
  assert.equal(row.classKey, "warrior");
  assert.deepEqual({ ...row.hit }, { flash: true, shiver: 0, lethal: false });
  const [{ bundle, wrapper }] = drawn([row]);
  const before = wrapper.find("champion-life__before")[0];
  const after = wrapper.find("champion-life__after")[0];
  assert.equal(before.getAttribute("data-beat-hold"), "impact");
  assert.equal(before.getAttribute("data-beat-side"), "before");
  assert.equal(before.getAttribute("opacity"), "0");
  assert.equal(before.getAttribute("data-beat-key"), row.presentationKey);
  assert.equal(after.getAttribute("data-beat-side"), "after");
  assert.equal(after.find("champion-life__lost").length, 1);
  assert.equal(after.find("champion-life__old-level").length, 1);
  assert.equal(bundle.flash.getAttribute("data-beat-pulse"), "0.08");
  assert.equal(bundle.flash.getAttribute("data-beat-fade"), "80");
  assert.equal(
    bundle.inner.getAttribute("data-beat-shiver"),
    null,
    "the Warrior never flinches",
  );
  assertExportSafe(wrapper.markup());

  const mage = withAgent(hurt, "agent-slot-1", { class_id: 1 });
  const mageRow = recordFor(
    champions.records({ scene: mage, transform, beats }),
    "agent-slot-1",
  );
  assert.equal(mageRow.hit.shiver, 1.5);
  const quiet = champions.records({
    scene: hurt,
    transform,
    beats,
    visualFilters: { ...DEFAULT_VISUAL_FILTER_STATE, hit_reactions: false },
  });
  assert.equal(recordFor(quiet, "agent-slot-1").hit, null);

  // A quiet tick on the same retained nodes removes every cue.
  const calm = recordFor(champions.records({ scene, transform }), "agent-slot-1");
  champions.update([bundle], [calm]);
  const markup = /** @type {any} */ (bundle.wrapper).markup();
  assert.doesNotMatch(markup, /data-beat-(hold|pulse|shiver)/u);
});

test("a seen death keeps the living look until the Fall beat, then ash", async () => {
  for (const name of ["replay_oracle", "replay_shared_obs_agent_pov"]) {
    const { scene, transform, beats } = await loadFrame(name, {
      start: 120,
      after: 0,
      died: true,
    });
    const dead = withAgent(scene, "agent-slot-1", {
      alive: false,
      life_state: "corpse",
      current_health: 0,
    });
    const row = recordFor(
      champions.records({ scene: dead, transform, beats }),
      "agent-slot-1",
    );
    assert.equal(row.state, "falling", name);
    assert.equal(row.life.before, 0.6, name);
    assert.equal(row.hit.lethal, true, name);
    const [{ bundle, wrapper }] = drawn([row]);
    assert.equal(bundle.falling.getAttribute("data-beat-hold"), "fall");
    assert.equal(bundle.falling.getAttribute("data-beat-side"), "before");
    assert.equal(bundle.falling.getAttribute("opacity"), "0");
    assert.equal(bundle.fallen.getAttribute("data-beat-side"), "after");
    assert.equal(bundle.fallen.getAttribute("data-beat-fade"), "300");
    assert.equal(bundle.teamRing.getAttribute("display"), "none");
    assert.doesNotMatch(
      wrapper.markup(),
      /dead-mark|team-marker/u,
      "no red X, no chevron",
    );
    assertExportSafe(wrapper.markup());

    const withoutFrame = recordFor(
      champions.records({ scene: dead, transform }),
      "agent-slot-1",
    );
    assert.equal(withoutFrame.state, "fallen");
    assert.equal(withoutFrame.life.before, null);
  }
});

test("low life, unknown life in Agent POV and the POV scene's bodies only", async () => {
  const { scene, transform } = await loadFrame("replay_oracle");
  const low = withAgent(scene, "agent-slot-2", { current_health: 30 });
  assert.equal(
    recordFor(champions.records({ scene: low, transform }), "agent-slot-2").life.low,
    true,
  );
  const lowOff = champions.records({
    scene: low,
    transform,
    visualFilters: { ...DEFAULT_VISUAL_FILTER_STATE, low_life_warning: false },
  });
  assert.equal(recordFor(lowOff, "agent-slot-2").life.low, false);

  const pov = await loadFrame("replay_shared_obs_agent_pov");
  const povRows = champions.records({
    scene: pov.scene,
    transform: pov.transform,
    beats: pov.beats,
  });
  assert.deepEqual(
    povRows.map((row) => row.presentationKey).sort(),
    pov.scene.agents
      .map((/** @type {Record<string, any>} */ agent) => agent.presentation_key)
      .sort(),
  );
  assert.ok(povRows.every((row) => row.audience === "agent_pov"));
  const hidden = withAgent(pov.scene, "agent-slot-2", { current_health: undefined });
  const unknown = recordFor(
    champions.records({ scene: hidden, transform: pov.transform }),
    "agent-slot-2",
  );
  assert.equal(unknown.life.known, false);
  assert.equal(unknown.life.share, null);
  const [{ wrapper }] = drawn([unknown]);
  assert.equal(wrapper.find("champion-life__unknown").length, 1);
  assert.equal(wrapper.find("champion-life__field").length, 0);
});

test("the Spawn Ward replaces the shield chip and counts up to three ticks", async () => {
  const { scene, transform } = await loadFrame("replay_oracle");
  const warded = withAgent(scene, "agent-slot-5", { spawn_shield_remaining: 5 });
  const row = recordFor(
    champions.records({ scene: warded, transform, showSpawnShield: true }),
    "agent-slot-5",
  );
  assert.deepEqual({ ...row.ward }, { ticks: 5, marks: 3 });
  const off = recordFor(
    champions.records({ scene: warded, transform, showSpawnShield: false }),
    "agent-slot-5",
  );
  assert.equal(off.ward, null);
  const [{ bundle, wrapper }] = drawn([row]);
  assert.ok(bundle.shieldRoot);
  assert.equal(
    wrapper.find("agent-spawn-shield__ring")[0].getAttribute("pathLength"),
    "12",
  );
  assert.equal(wrapper.find("agent-spawn-shield__diamonds").length, 1);
  assert.equal(bundle.teamRing.getAttribute("display"), "none");
  assert.equal(bundle.root.getAttribute("opacity"), "0.8");
  assert.doesNotMatch(wrapper.markup(), /agent-spawn-shield__chip/u);
  assertExportSafe(wrapper.markup());
});

test("Fallen Step Aside puts a covered corpse's coin at the standing champion's rim", async () => {
  const { scene, transform } = await loadFrame("replay_oracle");
  const covered = withAgent(scene, "agent-slot-0", {
    alive: false,
    life_state: "corpse",
    current_health: 0,
    position: [1.6, 3.75],
  });
  const rows = champions.records({ scene: covered, transform });
  const corpse = recordFor(rows, "agent-slot-0");
  const standing = recordFor(rows, "agent-slot-1");
  assert.ok(corpse.aside);
  const reach = Math.hypot(
    corpse.aside.x - standing.center.x,
    corpse.aside.y - standing.center.y,
  );
  assert.ok(
    Math.abs(reach - (standing.radius + corpse.geometry.coinRadius + 3)) < 1e-6,
  );
  const off = champions.records({
    scene: covered,
    transform,
    visualFilters: { ...DEFAULT_VISUAL_FILTER_STATE, fallen_step_aside: false },
  });
  assert.equal(recordFor(off, "agent-slot-0").aside, null);
  const [{ bundle }] = drawn([corpse]);
  assert.equal(bundle.body.getAttribute("data-aside"), "true");
});

test("the look reads only the arena scene and the audience's own rows", async () => {
  const source = await readFile(
    new URL("../src/looks/champions.js", import.meta.url),
    "utf8",
  );
  const code = source.replace(/\/\*\*[\s\S]*?\*\//gu, "").replace(/\/\/.*$/gmu, "");
  assert.doesNotMatch(
    code,
    /researcher_space|match_summary|ResearcherScene|researcherView/u,
  );
  assert.doesNotMatch(
    code,
    /from "\.\.\/(authorized-presentation-(adapter|normalizer)|choreography-plan)\.js"/u,
  );
  assert.doesNotMatch(
    code,
    /setTimeout|setInterval|requestAnimationFrame|\.animate\(/u,
  );
});
