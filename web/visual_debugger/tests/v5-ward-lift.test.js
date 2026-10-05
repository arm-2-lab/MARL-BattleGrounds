/**
 * @file Check the Ward Lift (switch spawn_shield_expiry, off by default): when
 * a seen champion's Spawn Shield ends, the plan row (choreography-plan.js)
 * carries the body's radius in CSS px (anchorRadius) only while the switch is
 * on, and is suppressed with no geometry while it is off. The Rune Zones look
 * (src/looks/statuses.js) claims only that status_lifecycle row ("ward-lift"),
 * draws one dashed Ward Silver ring at 0.95R (12 dashes, 1 px, 60%) on the
 * recipient during play, nothing in a settled (paused) mount or without a
 * radius, moves the ring on update and explains it with the shared Spawn
 * Shield explanation. The ring is export-safe SVG, one node, inside the event
 * budget; a repeat plan is equal.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { buildChoreographyPlan } from "../src/choreography-plan.js";
import * as statuses from "../src/looks/statuses.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  enableAllVisualFilters,
  setVisualFilterEnabled,
} from "../src/visual-filters.js";

class FakeElement {
  /** @param {string} tag */
  constructor(tag) {
    this.tagName = tag;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    this.ownerDocument = fakeDocument;
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
  /**
   * @param {string} selector A single ".class" selector.
   * @returns {FakeElement | null}
   */
  querySelector(selector) {
    const name = selector.replace(/^\./u, "");
    for (const child of this.children) {
      if (String(child.getAttribute("class")).split(" ").includes(name)) return child;
      /** @type {FakeElement | null} */
      const found = child.querySelector(selector);
      if (found) return found;
    }
    return null;
  }
  /** @returns {string} */
  serialize() {
    const attrs = [...this.attributes].map(([key, value]) => ` ${key}="${value}"`);
    return `<${this.tagName}${attrs.join("")}>${this.children.map((child) => child.serialize()).join("")}</${this.tagName}>`;
  }
}

const fakeDocument = {
  /** @param {string} _namespace @param {string} tag */
  createElementNS(_namespace, tag) {
    return new FakeElement(tag);
  },
};

const surface = {
  worldToScreen: (/** @type {any} */ point) => {
    const x = "x" in point ? point.x : Number(point[0]);
    const y = "y" in point ? point.y : Number(point[1]);
    return { x: x * 10, y: 120 - y * 10 };
  },
  worldLengthToScreen: (/** @type {number} */ length) => length * 10,
  viewportBounds: { left: 0, top: 0, right: 800, bottom: 600, width: 800, height: 600 },
  protectedRects: [],
};

const ALL_ON = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);

/** Build one Oracle frame whose only event is a Spawn Shield ending. */
async function shieldFrame() {
  const fixture = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const trajectory = raw.latest_events.agent_phase_trajectories[0];
  const transitionId = raw.latest_events.incoming_transition_id;
  raw.latest_events.events = [
    {
      event_kind: "spawn_shield_expired",
      agent_anchor: structuredClone(trajectory.successor),
      event_id: `${transitionId}:event:0000`,
      ordinal: 0,
      phase_rank: 110,
    },
  ];
  raw.latest_events.event_count = 1;
  raw.latest_events.ordered_event_ids = [raw.latest_events.events[0].event_id];
  raw.latest_events.ordered_event_kinds = ["spawn_shield_expired"];
  const agent = raw.current_endpoint.scene.agents.find(
    (/** @type {Record<string, any>} */ row) =>
      row.presentation_key === trajectory.agent_presentation_key,
  );
  return { frame: await normalizeAuthorizedPresentationFrameV1(raw), agent };
}

/** @param {Record<string, any>} plan */
function shieldRow(plan) {
  return plan.events.find(
    (/** @type {Record<string, any>} */ event) =>
      event.cueSemantic === "spawn_shield_expired",
  );
}

/** @param {boolean} settled */
function mount(settled) {
  return {
    group: /** @type {any} */ (new FakeElement("g")),
    connector: /** @type {any} */ (new FakeElement("g")),
    underlay: null,
    plan: { epochKey: "test", phases: {} },
    options: { motionMode: /** @type {const} */ ("normal"), settled },
    animationSpecs: [],
  };
}

test("the plan row carries the body radius only while Spawn Shield Expiry is on", async () => {
  const { frame, agent } = await shieldFrame();
  const plan = buildChoreographyPlan(frame, /** @type {any} */ (surface), ALL_ON);
  assert.ok(plan);
  const row = shieldRow(plan);
  assert.ok(row);
  assert.equal(row.spatial, true);
  assert.equal(row.anchorRadius, (agent.radius ?? 0.5) * 10);
  const off = buildChoreographyPlan(
    frame,
    /** @type {any} */ (surface),
    setVisualFilterEnabled(ALL_ON, "spawn_shield_expiry", false),
  );
  assert.ok(off);
  const suppressed = shieldRow(off);
  assert.deepEqual(
    [suppressed.spatial, suppressed.presentationSuppressed, suppressed.anchorRadius],
    [false, true, null],
  );
  assert.deepEqual(
    buildChoreographyPlan(frame, /** @type {any} */ (surface), ALL_ON),
    plan,
  );
});

test("the look draws one dashed ring at 0.95R during play and explains it", async () => {
  const { frame } = await shieldFrame();
  const plan = buildChoreographyPlan(frame, /** @type {any} */ (surface), ALL_ON);
  const event = shieldRow(/** @type {any} */ (plan));
  const records = statuses.records({ event });
  assert.deepEqual(
    records.map((record) => record.kind),
    ["ward-lift"],
  );
  const live = mount(false);
  statuses.build(records, /** @type {any} */ (fakeDocument), /** @type {any} */ (live));
  const [ring] = live.group.children;
  assert.equal(live.group.children.length, 1);
  assert.ok(live.group.children.length <= statuses.BUDGET.nodesPerRecord);
  const radius = event.anchorRadius * 0.95;
  const dash = Math.round(((2 * Math.PI * radius) / 24) * 100) / 100;
  assert.deepEqual(
    [
      ring.tagName,
      ring.getAttribute("class"),
      ring.getAttribute("r"),
      ring.getAttribute("stroke-dasharray"),
      ring.getAttribute("stroke-width"),
      ring.getAttribute("opacity"),
      ring.getAttribute("cx"),
      ring.getAttribute("cy"),
    ],
    [
      "circle",
      "ward-lift spawn-shield-expiry__ring",
      String(Math.round(radius * 100) / 100),
      `${dash} ${dash}`,
      "1",
      "0.6",
      String(Math.round(event.recipient.x * 100) / 100),
      String(Math.round(event.recipient.y * 100) / 100),
    ],
  );
  assert.doesNotMatch(
    live.group.serialize(),
    /url\(|gradient|<mask|<filter|<image|mix-blend|\bid=|combat-lifecycle__/iu,
  );
  const explanation = /** @type {any} */ (statuses.tooltip(records[0]));
  assert.equal(explanation.title, "Expired Spawn Shield (Invulnerable)");
  statuses.update(
    [live.group, live.connector],
    [{ kind: "ward-lift", event: { ...event, recipient: { x: 11, y: 22 } } }],
  );
  assert.deepEqual([ring.getAttribute("cx"), ring.getAttribute("cy")], ["11", "22"]);
});

test("nothing is drawn paused, without a radius, or for other status changes", async () => {
  const { frame } = await shieldFrame();
  const plan = buildChoreographyPlan(frame, /** @type {any} */ (surface), ALL_ON);
  const event = shieldRow(/** @type {any} */ (plan));
  const paused = mount(true);
  statuses.build(
    statuses.records({ event }),
    /** @type {any} */ (fakeDocument),
    /** @type {any} */ (paused),
  );
  assert.equal(paused.group.children.length, 0);
  const noRadius = mount(false);
  statuses.build(
    statuses.records({ event: { ...event, anchorRadius: null } }),
    /** @type {any} */ (fakeDocument),
    /** @type {any} */ (noRadius),
  );
  assert.equal(noRadius.group.children.length, 0);
  for (const cueSemantic of ["status_applied", "status_aged_to_zero", undefined]) {
    assert.deepEqual(
      statuses.records({ event: { ...event, cueSemantic } }),
      [],
      String(cueSemantic),
    );
  }
  assert.deepEqual(statuses.records({ event: { ...event, recipient: null } }), []);
});
