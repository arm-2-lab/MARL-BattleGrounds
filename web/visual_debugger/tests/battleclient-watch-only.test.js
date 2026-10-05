/**
 * @file Check the BattleClient's Watch Only plumbing and its page-paced Run:
 * - The live transport's optional `focused_public_agent_id` (Agent POV: the
 *   champion the player looks at when it is not the recipient) passes
 *   normalizeSystemControls and systemTransportKeys when it is a public ID,
 *   is absent when the frame lacks it, and a malformed value is refused.
 * - The hotbar (src/looks/hotbar.js) given `lookedAtKey` of a body in the
 *   audience's scene draws Watch Only: the Look Ring 12 vision pixels (scaled
 *   by the body's radius / 25) outside that body
 *   first, then your champion's Command Ring and studs marked `away` (classes
 *   `hotbar-ring--away` and `hotbar-studs--away`), and no Reach Marks, Order
 *   Thread or Step Ghost. A key the scene does not draw (an unseen champion)
 *   or your own champion's key changes nothing.
 * - liveRunning is true while the page's own Run is on (`pageRun`) or the
 *   server reports run_state "running", and false otherwise.
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
import {
  normalizeSystemControls,
  systemTransportKeys,
} from "../src/system-controls.js";
import { liveRunning } from "../src/tactics-turn.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);
const frames = await Promise.all(
  Object.values(fixture.presentations).map((raw) =>
    normalizeAuthorizedPresentationFrameV1(structuredClone(raw)),
  ),
);
const live = frames.filter((frame) => frame.viewer_mode === "live");

/** @param {Record<string, any>} frame */
function inputFor(frame) {
  const scene = /** @type {Record<string, any>} */ (
    authorizedPresentationSceneView(frame)
  );
  const transform = createViewportTransform({
    worldWidth: scene.map.width,
    worldHeight: scene.map.height,
    viewportWidth: 1032,
    viewportHeight: 532,
    padding: 16,
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

function fakeSvgDocument() {
  return /** @type {any} */ ({
    createElementNS(/** @type {string} */ _ns, /** @type {string} */ tag) {
      return {
        localName: tag,
        attributes: new Map(),
        setAttribute(/** @type {string} */ name, /** @type {unknown} */ value) {
          this.attributes.set(name, String(value));
        },
        removeAttribute(/** @type {string} */ name) {
          this.attributes.delete(name);
        },
      };
    },
  });
}

test("the focused public ID is an optional, checked live transport fact", () => {
  const base = { run_state: "paused", fresh_launch_required: false };
  assert.equal(
    Object.hasOwn(normalizeSystemControls(base), "focused_public_agent_id"),
    false,
  );
  assert.deepEqual(systemTransportKeys(["frame_kind"], base), [
    "frame_kind",
    "fresh_launch_required",
    "run_state",
  ]);
  const focused = { ...base, focused_public_agent_id: "agent_2" };
  assert.equal(normalizeSystemControls(focused).focused_public_agent_id, "agent_2");
  assert.ok(
    systemTransportKeys(["frame_kind"], focused).includes("focused_public_agent_id"),
  );
  for (const bad of [null, "", 7, "agent 2", "<b>"]) {
    assert.throws(
      () => normalizeSystemControls({ ...base, focused_public_agent_id: bad }),
      TypeError,
    );
  }
});

test("Watch Only rings the looked-at body softly and dims your champion's marks", () => {
  let checked = 0;
  for (const frame of live) {
    const input = inputFor(frame);
    const controlledKey = input.scene.selection?.controlled_presentation_key;
    if (typeof controlledKey !== "string") continue;
    if (input.inspection?.inspection_kind === "policy_read_only") continue;
    const other = input.scene.agents.find(
      (/** @type {any} */ agent) => agent.presentation_key !== controlledKey,
    );
    if (!other) continue;
    checked += 1;
    const plain = hotbar.records(input);
    assert.deepEqual(hotbar.records({ ...input, lookedAtKey: controlledKey }), plain);
    assert.deepEqual(hotbar.records({ ...input, lookedAtKey: "not-in-scene" }), plain);
    const records = hotbar.records({ ...input, lookedAtKey: other.presentation_key });
    const kinds = records.map((record) => record.kind);
    assert.equal(kinds[0], "look-ring");
    assert.equal(kinds[1], "command-ring");
    assert.ok(
      kinds.every((kind) =>
        ["look-ring", "command-ring", "order-studs"].includes(kind),
      ),
    );
    const radius = input.transform.worldLengthToScreen(other.radius);
    assert.ok(Math.abs(records[0].r - (radius + (12 * radius) / 25)) < 0.01);
    assert.equal(records[0].identityKey, other.presentation_key);
    assert.equal(records[1].identityKey, controlledKey);
    assert.equal(records[1].away, true);
    for (const studs of records.filter((record) => record.kind === "order-studs")) {
      assert.equal(studs.away, true);
    }
    const nodes = /** @type {any[]} */ (hotbar.build(records, fakeSvgDocument()));
    assert.equal(nodes[0].attributes.get("class"), "hotbar-look selection-moonlight");
    assert.match(nodes[1].attributes.get("class"), /\bhotbar-ring--away\b/u);
    for (const node of nodes.slice(2)) {
      assert.match(node.attributes.get("class"), /\bhotbar-studs--away\b/u);
    }
  }
  assert.ok(checked > 0);
});

test("liveRunning follows the page's own Run or the server's run state", () => {
  assert.equal(liveRunning({ frame: null }), false);
  assert.equal(liveRunning({ frame: { run_state: "paused" } }), false);
  assert.equal(liveRunning({ pageRun: true, frame: { run_state: "paused" } }), true);
  assert.equal(liveRunning({ pageRun: false, frame: { run_state: "running" } }), true);
});
