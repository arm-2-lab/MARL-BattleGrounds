/**
 * @file Checks the Health And Combat Text look (Ivory Numbers) in `src/looks/health-text.js`.
 *
 * Contracts checked: lifeNumberText writes "−N" and "+N" with no "NET" and
 * nothing for no change, rounded the store's way ("−41" for 41.05, "−3.3" for
 * 3.28, "+2" for 1.98, "<0.1" below 0.1); in Oracle View a loss is an ivory "damage" number, a
 * heal a gold "heal" number and Rest a Moon Dew "rest" number, 13 px, or 15 px
 * when an Ultimate reached the champion or the change killed it; an unchanged
 * body and a switched-off battle text give no number; in Agent POV a number
 * shows only for a champion seen at both ends of the tick, never after it
 * returned to its pad, and always 13 px; on both teams an Agent POV heal and
 * Rest number equal Oracle View's exactly (text, gold "heal" or Moon Dew "rest"
 * tone, size) and draw the same data-tone, and a plan-less call (unknown
 * audience) uses the same color; a heal or Rest on a champion the team did not
 * see at both ends draws no number;
 * build() draws one export-safe `text.combat-number` with no ID and no leader
 * line (the connector group stays empty), re-decides the number from mount.plan, and queues one 4 px pop at the event's phaseStart with
 * fill "backwards" (none when settled); the FOOTPRINT keys the plan reads stay;
 * the module never reads the researcher view or the match summary.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { buildChoreographyPlan } from "../src/choreography-plan.js";
import { createViewportTransform } from "../src/layout.js";
import { authorizedPresentationSceneView } from "../src/authorized-presentation-adapter.js";
import * as healthText from "../src/looks/health-text.js";
import { DEFAULT_VISUAL_FILTER_STATE } from "../src/visual-filters.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/**
 * @param {string} name
 * @param {{start: number, after: number, ultimate?: boolean}} change
 * @param {Record<string, boolean>} [filters]
 */
async function planFor(name, change, filters = DEFAULT_VISUAL_FILTER_STATE) {
  const raw = structuredClone(fixture.presentations[name]);
  const branch = name.includes("agent_pov") ? raw.visual_events : raw.latest_events;
  const resolution = branch.events.find(
    (/** @type {Record<string, any>} */ row) =>
      row.event_kind === "recipient_health_resolution",
  );
  resolution.transition_start_health = change.start;
  resolution.health_after_combat_resolution = change.after;
  resolution.realized_net_health_change = change.after - change.start;
  if ("total_effective_damage" in resolution) {
    resolution.total_effective_damage = Math.max(0, change.start - change.after);
    resolution.total_effective_healing = Math.max(0, change.after - change.start);
  }
  if (change.ultimate) {
    const activation = branch.events.find(
      (/** @type {Record<string, any>} */ row) =>
        row.event_kind === "ability_activated",
    );
    activation.ability_component = "ultimate";
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
  const plan = buildChoreographyPlan(
    frame,
    {
      worldToScreen: transform.worldToScreen,
      worldLengthToScreen: transform.worldLengthToScreen,
      viewportBounds: transform.mapBounds,
      protectedRects: [],
    },
    filters,
  );
  assert.ok(plan, name);
  const event = plan.events.find(
    (/** @type {Record<string, any>} */ row) => row.kind === "net_health",
  );
  assert.ok(event, `${name} has a health change`);
  return { plan, event };
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
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    this.children.push(...nodes);
  }
  addEventListener() {}
  /** @returns {FakeElement[]} */
  all() {
    return [this, ...this.children.flatMap((child) => child.all())];
  }
  /** @param {string} selector */
  querySelector(selector) {
    const name = selector.replace(/^\./u, "");
    return (
      this.all().find((node) =>
        (node.getAttribute("class") ?? "").split(/\s+/u).includes(name),
      ) ?? null
    );
  }
  /** @returns {string} */
  markup() {
    const attributes = [...this.attributes, ...Object.entries(this.dataset)]
      .map(([key, value]) => ` ${key}="${value}"`)
      .join("");
    return `<${this.tagName}${attributes}>${this.textContent}${this.children
      .map((child) => child.markup())
      .join("")}</${this.tagName}>`;
  }
}

class FakeDocument {
  /** @param {string} _namespace @param {string} tagName */
  createElementNS(_namespace, tagName) {
    return new FakeElement(this, tagName);
  }
}

/**
 * @param {Record<string, any>} event
 * @param {Record<string, any>} plan
 * @param {Record<string, any>} options
 */
function drawEvent(event, plan, options) {
  const doc = /** @type {any} */ (new FakeDocument());
  const visible = { ...event, cueLeader: null };
  /** @type {any[]} */
  const animationSpecs = [];
  const mount = /** @type {any} */ ({
    group: doc.createElementNS("", "g"),
    connector: doc.createElementNS("", "g"),
    underlay: null,
    plan,
    options,
    animationSpecs,
  });
  healthText.build(healthText.records({ event: visible }), doc, mount);
  return { group: mount.group, connector: mount.connector, animationSpecs };
}

test("lifeNumberText writes plain signed numbers and nothing for no change", () => {
  assert.equal(healthText.lifeNumberText(-20), "−20");
  assert.equal(healthText.lifeNumberText(12), "+12");
  assert.equal(healthText.lifeNumberText(0), null);
  assert.equal(healthText.lifeNumberText(Number.NaN), null);
  assert.equal(healthText.lifeNumberText(null), null);
  assert.equal(healthText.lifeNumberText(0.001), "+<0.1");
  assert.equal(healthText.lifeNumberText(-41.05), "−41");
  assert.equal(healthText.lifeNumberText(-3.28), "−3.3");
  assert.equal(healthText.lifeNumberText(-6.8), "−6.8");
  assert.equal(healthText.lifeNumberText(1.98), "+2");
  assert.equal(healthText.lifeNumberText(9.96), "+10");
  assert.equal(healthText.lifeNumberText(-200), "−200");
});

test("Oracle View numbers use tone and size from the audience's own plan", async () => {
  const loss = await planFor("replay_oracle", { start: 200, after: 150 });
  assert.deepEqual(
    { ...healthText.lifeNumber(loss.event, loss.plan) },
    { text: "−50", tone: "damage", size: 13 },
  );
  const ultimate = await planFor("replay_oracle", {
    start: 200,
    after: 150,
    ultimate: true,
  });
  assert.equal(healthText.lifeNumber(ultimate.event, ultimate.plan)?.size, 15);
  const lethal = await planFor("replay_oracle", { start: 40, after: 0 });
  assert.equal(healthText.lifeNumber(lethal.event, lethal.plan)?.size, 15);
  const heal = await planFor("replay_oracle", { start: 150, after: 160 });
  assert.deepEqual(
    { ...healthText.lifeNumber(heal.event, heal.plan) },
    { text: "+10", tone: "heal", size: 13 },
  );
  const same = await planFor("replay_oracle", { start: 200, after: 200 });
  assert.equal(healthText.lifeNumber(same.event, same.plan), null);
  const rest = healthText.lifeNumber(
    { kind: "regeneration", value: 3, recipientPresentationKey: "k" },
    { audience: "researcher", bodies: [], events: [] },
  );
  assert.deepEqual({ ...rest }, { text: "+3", tone: "rest", size: 13 });
  const off = await planFor(
    "replay_oracle",
    { start: 200, after: 150 },
    { ...DEFAULT_VISUAL_FILTER_STATE, scrolling_battle_text: false },
  );
  assert.equal(healthText.lifeNumber(off.event, off.plan), null);
});

test("Agent POV losses are ivory, 13 px and only for bodies seen at both ends", async () => {
  for (const name of [
    "replay_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
  ]) {
    const loss = await planFor(name, { start: 200, after: 150, ultimate: true });
    assert.equal(loss.plan.audience, "agent_pov");
    const number = healthText.lifeNumber(loss.event, loss.plan);
    assert.deepEqual({ ...number }, { text: "−50", tone: "damage", size: 13 }, name);
    const key = loss.event.recipientPresentationKey;
    const arrived = {
      ...loss.plan,
      bodies: loss.plan.bodies.map((/** @type {Record<string, any>} */ body) =>
        body.presentationKey === key ? { ...body, start: null } : body,
      ),
    };
    assert.equal(healthText.lifeNumber(loss.event, arrived), null, `${name} arrival`);
    const returned = {
      ...loss.plan,
      bodies: loss.plan.bodies.map((/** @type {Record<string, any>} */ body) =>
        body.presentationKey === key ? { ...body, returned: true } : body,
      ),
    };
    assert.equal(healthText.lifeNumber(loss.event, returned), null, `${name} respawn`);
    assert.equal(
      healthText.lifeNumber(loss.event, { ...loss.plan, bodies: [] }),
      null,
      `${name} unseen body`,
    );
  }
});

test("Agent POV colors every number it shows like Oracle View, on both teams, and hides unseen heals", async () => {
  const oracle = await planFor("replay_oracle", { start: 150, after: 160 });
  const oracleHeal = { ...healthText.lifeNumber(oracle.event, oracle.plan) };
  assert.deepEqual(oracleHeal, { text: "+10", tone: "heal", size: 13 });
  const restOf = (/** @type {Record<string, any>} */ event) => ({
    ...event,
    kind: "regeneration",
    value: 3,
  });
  const oracleRest = { ...healthText.lifeNumber(restOf(oracle.event), oracle.plan) };
  assert.deepEqual(oracleRest, { text: "+3", tone: "rest", size: 13 });
  /**
   * @param {Record<string, any>} event
   * @param {Record<string, any>} plan
   */
  const drawnTone = (event, plan) =>
    drawEvent(event, plan, { motionMode: "normal", settled: true })
      .group.querySelector(".combat-number")
      ?.getAttribute("data-tone") ?? null;
  const oracleTones = {
    heal: drawnTone(oracle.event, oracle.plan),
    rest: drawnTone(restOf(oracle.event), oracle.plan),
  };
  assert.deepEqual(oracleTones, { heal: "heal", rest: "rest" });
  // A plan-less call (unknown audience) uses the same color rule.
  assert.deepEqual(
    { ...healthText.lifeNumber({ kind: "net_health", netDelta: 10 }, null) },
    oracleHeal,
  );
  assert.deepEqual(
    { ...healthText.lifeNumber({ kind: "regeneration", value: 3 }, null) },
    oracleRest,
  );
  for (const name of [
    "replay_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
  ]) {
    const heal = await planFor(name, { start: 150, after: 160 });
    assert.equal(heal.plan.audience, "agent_pov");
    const key = heal.event.recipientPresentationKey;
    /**
     * @param {Record<string, any>} plan
     * @param {string} wanted
     */
    const bodyOf = (plan, wanted) =>
      plan.bodies.find(
        (/** @type {Record<string, any>} */ entry) => entry.presentationKey === wanted,
      );
    // audienceKey is [session, view kind, POV recipient's presentation key].
    const ownTeam = bodyOf(heal.plan, JSON.parse(heal.plan.audienceKey)[2])?.teamId;
    assert.ok(ownTeam === 1 || ownTeam === 2, name);
    assert.equal(bodyOf(heal.plan, key)?.teamId, ownTeam, `${name} own champion`);
    const enemyPlan = {
      ...heal.plan,
      bodies: heal.plan.bodies.map((/** @type {Record<string, any>} */ entry) =>
        entry.presentationKey === key ? { ...entry, teamId: 3 - ownTeam } : entry,
      ),
    };
    /** @type {Array<[string, Record<string, any>]>} */
    const teams = [
      ["own team", heal.plan],
      ["other team", enemyPlan],
    ];
    for (const [team, plan] of teams) {
      const where = `${name} ${team}`;
      const rest = restOf(heal.event);
      assert.deepEqual(
        { ...healthText.lifeNumber(heal.event, plan) },
        oracleHeal,
        where,
      );
      assert.deepEqual({ ...healthText.lifeNumber(rest, plan) }, oracleRest, where);
      assert.equal(drawnTone(heal.event, plan), oracleTones.heal, where);
      assert.equal(drawnTone(rest, plan), oracleTones.rest, where);
      const arrived = {
        ...plan,
        bodies: plan.bodies.map((/** @type {Record<string, any>} */ entry) =>
          entry.presentationKey === key ? { ...entry, start: null } : entry,
        ),
      };
      assert.equal(
        healthText.lifeNumber(heal.event, arrived),
        null,
        `${where} arrival`,
      );
      assert.equal(drawnTone(heal.event, arrived), null, `${where} arrival`);
      const unseen = {
        ...plan,
        bodies: plan.bodies.filter(
          (/** @type {Record<string, any>} */ entry) => entry.presentationKey !== key,
        ),
      };
      assert.equal(healthText.lifeNumber(heal.event, unseen), null, `${where} unseen`);
      assert.equal(drawnTone(heal.event, unseen), null, `${where} unseen`);
      assert.equal(drawnTone(rest, unseen), null, `${where} unseen Rest`);
    }
  }
});

test("build draws one export-safe number and queues its pop on the beat clock", async () => {
  const { plan, event } = await planFor("replay_oracle", {
    start: 200,
    after: 150,
    ultimate: true,
  });
  const { group, connector, animationSpecs } = drawEvent(event, plan, {
    motionMode: "normal",
    settled: false,
  });
  const text = group.querySelector(".combat-number");
  assert.ok(text);
  assert.equal(connector.querySelector(".combat-cue__leader"), null);
  assert.equal(group.querySelector(".combat-cue__leader"), null);
  assert.equal(text.textContent, "−50");
  assert.equal(text.getAttribute("data-tone"), "damage");
  assert.equal(text.getAttribute("font-size"), "15");
  assert.equal(text.getAttribute("paint-order"), "stroke");
  const markup = group.markup();
  assert.doesNotMatch(markup, /NET|HP unchanged|ID \d|agent-slot/u);
  assert.doesNotMatch(markup, /url\(|gradient|<mask|<filter|<image| id="| style="/iu);
  assert.equal(animationSpecs.length, 1);
  assert.equal(animationSpecs[0].element, text);
  assert.equal(animationSpecs[0].options.fill, "backwards");
  assert.equal(
    animationSpecs[0].options.delay,
    Math.max(0, Number(event.phaseStart ?? 0)),
  );
  assert.equal(animationSpecs[0].options.duration, 120);
  const settled = drawEvent(event, plan, { motionMode: "normal", settled: true });
  assert.equal(settled.animationSpecs.length, 0);
  const reduced = drawEvent(event, plan, { motionMode: "reduced", settled: false });
  assert.equal(reduced.animationSpecs.length, 0);

  const povPlan = { ...plan, audience: "agent_pov", bodies: [] };
  const hidden = drawEvent(event, povPlan, { motionMode: "normal", settled: false });
  assert.equal(hidden.group.querySelector(".combat-number"), null);
  assert.ok(hidden.group.querySelector(".combat-number__hit"));
});

test("the footprint keys stay and the look reads no researcher view", async () => {
  assert.deepEqual(Object.keys(healthText.FOOTPRINT).sort(), [
    "netEffect",
    "netText",
    "regenerationEffect",
    "regenerationText",
    "unchangedNetText",
  ]);
  const source = await readFile(
    new URL("../src/looks/health-text.js", import.meta.url),
    "utf8",
  );
  const code = source.replace(/\/\*\*[\s\S]*?\*\//gu, "").replace(/\/\/.*$/gmu, "");
  assert.doesNotMatch(code, /researcher_space|match_summary|ResearcherScene/u);
  assert.doesNotMatch(code, /setTimeout|setInterval|requestAnimationFrame|NET /u);
});
