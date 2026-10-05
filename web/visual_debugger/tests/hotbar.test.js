/**
 * @file Check the Champion's Hotbar look (src/looks/hotbar.js) on the live
 * fixture frames in Oracle View and Agent POV: the Command Ring sits at 1.45
 * body radii around the champion you order; the Order Studs show No Combat,
 * Basic, Ultimate, the Mage's Burst and a refused order (dots are four
 * `<circle>` records, as the vision draws them; diamonds are one path of
 * four); the straight Order
 * Thread (an ink underlay and the silver line) runs from your champion's edge
 * to 4 vision pixels (scaled by your champion's radius / 25) short of an
 * allowed, drawn target; Reach Marks ring only bodies the
 * host's mask allows for the armed action AND the audience's scene draws (so
 * Agent POV never marks an unseen champion); `command_marks`, `reach_marks`
 * and `step_ghost` off drop their marks and keep the ring, and
 * `next_action_thread` off drops the Order Thread; a Stay step draws
 * no Step Ghost (battleclient-arena.test.js checks the ghost itself); a
 * replay frame (no controlled champion) gives no records; built nodes are
 * export-safe (no url(), gradient, filter, mask or image); every mark has
 * a plain hover card; and a mark on a champion (ring, studs, Reach Mark)
 * carries data-glide-follow with its champion's key, while the Order Thread
 * and the Step Ghost carry data-landing-wait with the keys they touch (the
 * thread also its target's), so a played tick never shows them at a landing
 * spot early.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  authorizedPresentationInspectionState,
  authorizedPresentationSceneView,
} from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { createViewportTransform } from "../src/layout.js";
import * as hotbar from "../src/looks/hotbar.js";

/** @returns {Promise<Array<Record<string, any>>>} */
async function fixtureFrames() {
  const fixture = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  return Promise.all(
    Object.values(fixture.presentations).map((raw) =>
      normalizeAuthorizedPresentationFrameV1(structuredClone(raw)),
    ),
  );
}

/** @param {Record<string, any>} frame */
function inputFor(frame) {
  const scene = /** @type {Record<string, any>} */ (
    authorizedPresentationSceneView(frame)
  );
  const transform = createViewportTransform({
    worldWidth: scene.map.width,
    worldHeight: scene.map.height,
    viewportWidth: 960,
    viewportHeight: 600,
    padding: 24,
  });
  return {
    scene,
    transform,
    audience: scene.audience,
    inspection: /** @type {Record<string, any>} */ (
      authorizedPresentationInspectionState(frame).inspection
    ),
  };
}

/** @param {Record<string, any>} inspection @param {Record<string, any>} draft */
function withDraft(inspection, draft) {
  return { ...inspection, draft_action: { ...inspection.draft_action, ...draft } };
}

const frames = await fixtureFrames();
const live = frames.filter((frame) => frame.viewer_mode === "live");

test("the live fixtures cover Oracle View and Agent POV", () => {
  const audiences = new Set(live.map((frame) => inputFor(frame).audience));
  assert.deepEqual([...audiences].sort(), ["agent_pov", "researcher"]);
});

test("the Command Ring sits at 1.45 radii around the champion you order", () => {
  for (const frame of live) {
    const input = inputFor(frame);
    const records = hotbar.records(input);
    const ring = records.find((record) => record.kind === "command-ring");
    const key = input.scene.selection.controlled_presentation_key;
    const body = input.scene.agents.find(
      (/** @type {any} */ agent) => agent.presentation_key === key,
    );
    assert.ok(ring && body, input.audience);
    const centre = input.transform.worldToScreen(body.position);
    const radius = input.transform.worldLengthToScreen(body.radius);
    assert.ok(
      Math.abs(ring.cx - centre.x) < 0.01 && Math.abs(ring.cy - centre.y) < 0.01,
    );
    assert.ok(Math.abs(ring.r - radius * 1.45) < 0.01);
    assert.equal(hotbar.FOOTPRINT.bandRadiusFactor, 1.45);
    assert.equal(records.filter((record) => record.kind === "command-ring").length, 1);
    assert.ok(
      hotbar
        .records({
          ...input,
          inspection: withDraft(input.inspection, { move_action: 0 }),
        })
        .every((record) => !/ghost/u.test(String(record.kind))),
      "a Stay step draws no Step Ghost",
    );
  }
});

test("Reach Marks ring only allowed bodies that the audience's scene draws", () => {
  let marked = 0;
  for (const frame of live) {
    const input = inputFor(frame);
    const sceneKeys = new Set(
      input.scene.agents.map((/** @type {any} */ agent) => agent.presentation_key),
    );
    const mask = input.inspection.decision_mask;
    // The fixtures allow no target, so open every pair and add one allowed
    // body that the scene does not draw.
    const hiddenKey = "pov_hidden_body_for_test";
    const opened = {
      ...input.inspection,
      decision_mask: {
        ...mask,
        target_actions: [
          ...mask.target_actions,
          {
            target_action: mask.target_actions.length,
            target_kind: "visible_authorized_agent",
            target_presentation_key: hiddenKey,
            target_public_agent_id: "agent-slot-hidden",
          },
        ],
        target_use_ultimate_joint_mask: [
          ...mask.target_use_ultimate_joint_mask.map(() => [true, true]),
          [true, true],
        ],
      },
    };
    const expected = opened.decision_mask.target_actions
      .filter(
        (/** @type {any} */ target) =>
          target.target_action > 0 &&
          target.target_kind === "visible_authorized_agent" &&
          sceneKeys.has(target.target_presentation_key),
      )
      .map((/** @type {any} */ target) => target.target_presentation_key)
      .sort();
    const records = hotbar.records({ ...input, inspection: opened });
    const ringed = records
      .filter((record) => record.kind === "reach-mark")
      .map((record) => record.identityKey)
      .sort();
    assert.deepEqual(ringed, expected, input.audience);
    assert.ok(!ringed.includes(hiddenKey));
    marked += ringed.length;
    // The fixtures' own closed masks mark nothing.
    assert.equal(
      hotbar.records(input).filter((record) => record.kind === "reach-mark").length,
      0,
    );
  }
  assert.ok(marked > 0, "the opened masks show Reach Marks");
});

test("Order Studs show the armed action and a refusal", () => {
  const frame = live.find((candidate) => inputFor(candidate).audience === "researcher");
  assert.ok(frame);
  const input = inputFor(frame);
  /** @param {Record<string, any>} draft @param {Record<string, any>} [extra] */
  const studs = (draft, extra = {}) =>
    hotbar
      .records({ ...input, ...extra, inspection: withDraft(input.inspection, draft) })
      .find((record) => record.kind === "order-studs");
  assert.equal(studs({ armed_lane: "none" })?.style, "no-combat");
  assert.equal(studs({ armed_lane: "basic" })?.style, "basic");
  const controlledKey = input.scene.selection.controlled_presentation_key;
  const body = input.scene.agents.find(
    (/** @type {any} */ agent) => agent.presentation_key === controlledKey,
  );
  const ultimate = studs({ armed_lane: "ultimate" })?.style;
  assert.equal(ultimate, body.class_id === 1 ? "burst" : "ultimate");
  const refused = studs(
    { armed_lane: "basic" },
    {
      scene: {
        ...input.scene,
        selected_legality: { armed_lane: 0, armed_pair_legal: false },
      },
    },
  );
  assert.equal(refused?.style, "refused");
  // Dots are four circle records; diamonds are four studs in one path.
  const all = (/** @type {Record<string, any>} */ draft) =>
    hotbar
      .records({ ...input, inspection: withDraft(input.inspection, draft) })
      .filter((record) => record.kind === "order-studs");
  const dots = all({ armed_lane: "basic" });
  assert.equal(dots.length, 4);
  assert.ok(dots.every((stud) => stud.shape === "dot" && stud.r > 0));
  const doc = /** @type {any} */ ({
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) => ({
      localName: tag,
      setAttribute() {},
    }),
  });
  const dotNodes = /** @type {any[]} */ (hotbar.build(dots, doc));
  assert.ok(dotNodes.every((node) => node.localName === "circle"));
  const diamonds = all({ armed_lane: "ultimate" });
  assert.equal(diamonds.length, 1);
  assert.equal(diamonds[0].shape, "diamond");
  assert.equal((String(diamonds[0].d).match(/M/gu) ?? []).length, 4);
});

test("the Order Thread joins an allowed, drawn target and nothing else", () => {
  const frame = live.find((candidate) => inputFor(candidate).audience === "researcher");
  assert.ok(frame);
  const input = inputFor(frame);
  const key = input.scene.selection.controlled_presentation_key;
  const source = input.scene.agents.find(
    (/** @type {any} */ agent) => agent.presentation_key === key,
  );
  const target = input.scene.agents.find(
    (/** @type {any} */ agent) => agent.presentation_key !== key,
  );
  const route = {
    source_presentation_key: key,
    source_anchor: source.position,
    source_radius: source.radius,
    target_presentation_key: target.presentation_key,
    target_anchor: target.position,
    target_radius: target.radius,
  };
  /** @param {Record<string, any>} legality @param {Record<string, any>} [sceneExtra] */
  const thread = (legality, sceneExtra = {}) =>
    hotbar
      .records({
        ...input,
        scene: {
          ...input.scene,
          pending_route: route,
          selected_legality: legality,
          ...sceneExtra,
        },
        inspection: withDraft(input.inspection, { armed_lane: "basic" }),
      })
      .find((record) => record.kind === "order-thread");
  const line = thread({ armed_lane: 0, armed_pair_legal: true });
  const ink = hotbar
    .records({
      ...input,
      scene: {
        ...input.scene,
        pending_route: route,
        selected_legality: { armed_lane: 0, armed_pair_legal: true },
      },
      inspection: withDraft(input.inspection, { armed_lane: "basic" }),
    })
    .find((record) => record.kind === "order-thread-ink");
  const a = input.transform.worldToScreen(source.position);
  const b = input.transform.worldToScreen(target.position);
  const length = Math.hypot(b.x - a.x, b.y - a.y);
  // The 4 px target gap is the vision's, scaled by your champion's radius / 25.
  const gap = (4 * input.transform.worldLengthToScreen(source.radius)) / 25;
  if (
    length -
      input.transform.worldLengthToScreen(source.radius) -
      input.transform.worldLengthToScreen(target.radius) -
      gap >=
    1
  ) {
    assert.ok(line, "an allowed target gets a thread");
    assert.ok(ink, "the thread has its ink underlay");
    assert.deepEqual(
      [ink.x1, ink.y1, ink.x2, ink.y2],
      [line.x1, line.y1, line.x2, line.y2],
    );
    const startGap = Math.hypot(line.x1 - a.x, line.y1 - a.y);
    assert.ok(
      Math.abs(startGap - input.transform.worldLengthToScreen(source.radius)) < 0.02,
    );
    const endGap = Math.hypot(line.x2 - b.x, line.y2 - b.y);
    assert.ok(
      Math.abs(endGap - (input.transform.worldLengthToScreen(target.radius) + gap)) <
        0.02,
    );
  }
  assert.equal(thread({ armed_lane: 0, armed_pair_legal: false }), undefined);
  // Next Action Thread off drops the Order Thread and its ink, nothing else.
  const threadOff = hotbar.records({
    ...input,
    filters: { next_action_thread: false },
    scene: {
      ...input.scene,
      pending_route: route,
      selected_legality: { armed_lane: 0, armed_pair_legal: true },
    },
    inspection: withDraft(input.inspection, { armed_lane: "basic" }),
  });
  assert.equal(
    threadOff.some((record) => String(record.kind).startsWith("order-thread")),
    false,
  );
  assert.ok(threadOff.some((record) => record.kind === "command-ring"));
  const withoutTarget = input.scene.agents.filter(
    (/** @type {any} */ agent) => agent.presentation_key !== target.presentation_key,
  );
  assert.equal(
    thread({ armed_lane: 0, armed_pair_legal: true }, { agents: withoutTarget }),
    undefined,
  );
});

test("switches drop their own marks and keep the ring", () => {
  for (const frame of live) {
    const input = inputFor(frame);
    const off = hotbar.records({
      ...input,
      filters: { command_marks: false, reach_marks: false, step_ghost: false },
    });
    assert.deepEqual(
      off.map((record) => record.kind),
      ["command-ring"],
    );
  }
  const kinds = hotbar.PAINT_PARTS.map(({ part, filter }) => [part.kind, filter]);
  assert.deepEqual(kinds, [
    ["command_marks", "command_marks"],
    ["reach_marks", "reach_marks"],
    ["step_ghost", "step_ghost"],
  ]);
});

test("a replay frame or a missing scene gives no records", () => {
  for (const frame of frames.filter(
    (candidate) => candidate.viewer_mode === "replay",
  )) {
    assert.deepEqual(hotbar.records(inputFor(frame)), []);
  }
  assert.deepEqual(hotbar.records({}), []);
});

test("built nodes are export-safe and every mark has a plain hover card", () => {
  /** @type {Array<{tag: string, attributes: Map<string, string>}>} */
  const made = [];
  const doc = /** @type {any} */ ({
    createElementNS(/** @type {string} */ _ns, /** @type {string} */ tag) {
      const node = {
        localName: tag,
        attributes: new Map(),
        setAttribute(/** @type {string} */ name, /** @type {string} */ value) {
          this.attributes.set(name, value);
        },
        removeAttribute(/** @type {string} */ name) {
          this.attributes.delete(name);
        },
      };
      made.push({ tag, attributes: node.attributes });
      return node;
    },
  });
  for (const frame of live) {
    const input = inputFor(frame);
    const records = hotbar.records({
      ...input,
      inspection: withDraft(input.inspection, { armed_lane: "basic" }),
    });
    const nodes = hotbar.build(records, doc);
    assert.equal(nodes.length, records.length);
    hotbar.update(nodes, records);
    for (const record of records) {
      const card = hotbar.tooltip(record);
      assert.ok(card && card.title.length > 0 && card.summary.length > 0);
      assert.doesNotMatch(card.summary, /Team [AB]|Azure|Crimson/u);
    }
  }
  assert.ok(["circle", "path", "line"].includes(made[0].tag));
  for (const { tag, attributes } of made) {
    assert.ok(["circle", "path", "line"].includes(tag));
    for (const value of attributes.values()) {
      assert.doesNotMatch(value, /url\(|gradient|filter|mask/u);
    }
  }
  assert.equal(hotbar.tooltip({ kind: "anything" }), null);
});

test("marks on a champion ride its glide and the thread and Step Ghost wait for the landing", () => {
  const frame = live.find((candidate) => inputFor(candidate).audience === "researcher");
  assert.ok(frame);
  const input = inputFor(frame);
  const key = input.scene.selection.controlled_presentation_key;
  const source = input.scene.agents.find(
    (/** @type {any} */ agent) => agent.presentation_key === key,
  );
  const target = input.scene.agents.find(
    (/** @type {any} */ agent) => agent.presentation_key !== key,
  );
  const records = hotbar.records({
    ...input,
    scene: {
      ...input.scene,
      pending_route: {
        source_presentation_key: key,
        source_anchor: source.position,
        source_radius: source.radius,
        target_presentation_key: target.presentation_key,
        target_anchor: target.position,
        target_radius: target.radius,
      },
      selected_legality: { armed_lane: 0, armed_pair_legal: true },
    },
    inspection: withDraft(input.inspection, { armed_lane: "basic", move_action: 3 }),
  });
  /** @type {Array<Map<string, string>>} */
  const made = [];
  const doc = /** @type {any} */ ({
    createElementNS(/** @type {string} */ _ns, /** @type {string} */ tag) {
      const attributes = new Map();
      made.push(attributes);
      return {
        localName: tag,
        attributes,
        setAttribute(/** @type {string} */ name, /** @type {string} */ value) {
          attributes.set(name, value);
        },
        removeAttribute(/** @type {string} */ name) {
          attributes.delete(name);
        },
      };
    },
  });
  hotbar.build(records, doc);
  const kinds = new Set(records.map((record) => record.kind));
  for (const kind of ["command-ring", "order-thread", "step-ghost"]) {
    assert.ok(kinds.has(kind), kind);
  }
  for (const [index, record] of records.entries()) {
    const attributes = made[index];
    if (/^(order-thread|step-ghost)/u.test(record.kind)) {
      const waits = String(attributes.get("data-landing-wait")).split(" ");
      assert.ok(waits.includes(key), record.kind);
      if (record.kind.startsWith("order-thread")) {
        assert.ok(
          waits.includes(target.presentation_key),
          "the thread waits for its target too",
        );
      }
      assert.equal(attributes.has("data-glide-follow"), false);
    } else {
      assert.equal(attributes.get("data-glide-follow"), record.returnKey, record.kind);
      assert.equal(attributes.has("data-landing-wait"), false);
    }
  }
});
