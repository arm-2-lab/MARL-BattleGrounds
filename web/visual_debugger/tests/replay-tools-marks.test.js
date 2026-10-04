/**
 * @file Check the Replay Viewer's two Replay Tools marks in
 * src/looks/champion-frame.js and their host facts in the normalizer.
 *
 * Contracts checked:
 * - All Staged Orders ("staged-orders"): one Next Action Thread record per
 *   living actor with a combat order, marked staged: a Basic draws a dot under
 *   its target, an Ultimate a diamond, an Ultimate with no target (the Mage's
 *   Burst) a diamond under the actor itself; no combat order, a dead actor, a
 *   target missing from the scene and an actor the Next Action Thread already
 *   draws give nothing; off gives nothing. Built nodes carry
 *   data-staged-order="true" and the hit path takes no Tab stop.
 * - Agent POV: on the fixture's Agent POV frames the frame's own upcoming
 *   rows hold only the recipient, so every staged order is the recipient's.
 * - Next Step Ghost ("next-step-ghosts"): one dotted ring per spot whose
 *   champion is alive in the scene and moves at least 1 px; a champion missing
 *   from the scene (in Agent POV, an unseen one) or fallen gets none; off gives
 *   nothing. Built nodes are the live Step Ghost's ink and dotted rings.
 * - The normalizer accepts the fixture frames' next_end_spots and
 *   recent_falls, and refuses a spot naming a champion not alive in the
 *   frame's own scene, any spot on the final frame, and a recent fall outside
 *   the 7 ticks before the shown one.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { authorizedPresentationSceneView } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { build, records } from "../src/looks/champion-frame.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

const transform = /** @type {any} */ ({
  worldToScreen: (/** @type {number[]} */ [x, y]) => ({ x: x * 10, y: y * 10 }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 10,
});

/** @param {string} id @param {number[]} position @param {boolean} [alive] */
const body = (id, position, alive = true) => ({
  presentation_key: `key-${id}`,
  public_agent_id: id,
  position,
  radius: 0.4,
  alive,
  life_state: alive ? "alive" : "corpse",
});

const scene = {
  agents: [
    body("0", [1, 1]),
    body("1", [3, 1]),
    body("2", [5, 1]),
    body("3", [7, 1]),
    body("5", [1, 5]),
    body("6", [3, 5], false),
  ],
};
const AXIS = [null, "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/** @param {string} id @param {number} target @param {0 | 1} ultimate */
const row = (id, target, ultimate) => ({
  actor_presentation_key: `key-${id}`,
  actor_public_agent_id: id,
  target_action_recipient_public_agent_id_by_id: AXIS,
  submitted_action: {
    move_action: 0,
    target_action: target,
    use_ultimate_action: ultimate,
  },
  accepted_action: {
    move_action: 0,
    target_action: target,
    use_ultimate_action: ultimate,
  },
});

/** A tiny document for build(): elements with attributes and children. */
function fakeDocument() {
  const make = (/** @type {string} */ tag) => {
    /** @type {any} */
    const node = {
      tagName: tag,
      attributes: new Map(),
      children: [],
      setAttribute(/** @type {string} */ name, /** @type {any} */ value) {
        node.attributes.set(name, String(value));
      },
      getAttribute(/** @type {string} */ name) {
        return node.attributes.get(name) ?? null;
      },
      removeAttribute(/** @type {string} */ name) {
        node.attributes.delete(name);
      },
      hasAttribute(/** @type {string} */ name) {
        return node.attributes.has(name);
      },
      append(/** @type {any[]} */ ...nodes) {
        node.children.push(...nodes);
      },
      addEventListener() {},
      getBoundingClientRect: () => ({ x: 0, y: 0, width: 0, height: 0 }),
      parentElement: null,
    };
    return node;
  };
  return /** @type {any} */ ({
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      make(tag),
    createElement: make,
  });
}

test("All Staged Orders draws one thread per combat order and skips the rest", () => {
  const actionRows = [
    row("0", 6, 0), // Basic on "5": a dot under it.
    row("1", 0, 1), // Targetless Ultimate: a diamond under the actor.
    row("2", 0, 0), // No combat order.
    row("3", 3, 1), // Ultimate on "2", but the thread already draws "3".
    row("5", 10, 0), // Target "9" is not in the scene.
    row("6", 1, 0), // The actor has fallen.
  ];
  const input = {
    part: "staged-orders",
    scene,
    transform,
    actionRows,
    skipActorIds: ["3"],
    showStagedOrders: true,
  };
  const marks = records(input);
  assert.deepEqual(
    marks.map((mark) => [mark.sourceAgentId, mark.targetAgentId, mark.glyph.shape]),
    [
      ["0", "5", "dot"],
      ["1", "1", "diamond"],
    ],
  );
  assert.equal(marks[0].routeKind, "thread");
  assert.equal(marks[1].routeKind, "self");
  assert.ok(marks.every((mark) => mark.staged === true && mark.legal === "true"));
  assert.deepEqual(records({ ...input, showStagedOrders: false }), []);
  const nodes = build(marks, fakeDocument());
  const staged = nodes.filter(
    (node) => node.getAttribute("data-staged-order") === "true",
  );
  assert.equal(staged.length, nodes.length);
  const hits = nodes.filter(
    (node) => node.getAttribute("class") === "pending-route-hit",
  );
  assert.equal(hits.length, 2);
  assert.ok(hits.every((node) => !node.hasAttribute("tabindex")));
});

test("Agent POV upcoming rows hold only the recipient's own order", async () => {
  let checked = 0;
  for (const name of [
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    const frame = await normalizeAuthorizedPresentationFrameV1(
      structuredClone(fixture.presentations[name]),
    );
    const view = authorizedPresentationSceneView(frame);
    const rows = frame.upcoming_transition?.action_rows ?? [];
    const marks = records({
      part: "staged-orders",
      scene: view,
      transform,
      actionRows: rows,
      showStagedOrders: true,
    });
    const recipient = frame.authority.recipient_public_agent_id;
    assert.ok(
      rows.every(
        (/** @type {Record<string, any>} */ entry) =>
          entry.actor_public_agent_id === recipient,
      ),
    );
    assert.ok(
      marks.every((mark) => mark.sourceAgentId === recipient),
      name,
    );
    checked += rows.length;
  }
  assert.ok(checked > 0, "the fixture POV frames carry upcoming rows");
});

test("Next Step Ghost rings each moving living champion's next spot", () => {
  const spots = [
    { public_agent_id: "0", position: [1.5, 1] }, // Moves 5 px: a ghost.
    { public_agent_id: "1", position: [3.05, 1] }, // Moves 0.5 px: none.
    { public_agent_id: "9", position: [2, 2] }, // Not in the scene: none.
    { public_agent_id: "6", position: [4, 5] }, // Fallen: none.
  ];
  const input = {
    part: "next-step-ghosts",
    scene,
    transform,
    spots,
    showNextStepGhost: true,
  };
  const ghosts = records(input);
  assert.deepEqual(
    ghosts.map((ghost) => [ghost.publicAgentId, ghost.cx, ghost.cy, ghost.r]),
    [["0", 15, 10, 4]],
  );
  assert.deepEqual(records({ ...input, showNextStepGhost: false }), []);
  const nodes = build(ghosts, fakeDocument());
  assert.deepEqual(
    nodes.map((node) => [
      node.getAttribute("class"),
      node.getAttribute("data-next-step-ghost"),
    ]),
    [
      ["hotbar-ghost-ink", "ink"],
      ["hotbar-ghost", "ring"],
    ],
  );
});

test("the normalizer checks next_end_spots and recent_falls against the frame", async () => {
  const oracle = structuredClone(fixture.presentations.replay_oracle);
  const pov = structuredClone(fixture.presentations.replay_no_shared_obs_agent_pov);
  const living = (/** @type {any} */ frame) =>
    (frame.current_endpoint.scene ?? frame.current_endpoint.parts.scene).agents.filter(
      (/** @type {any} */ agent) => agent.life_state === "alive",
    );
  for (const frame of [oracle, pov]) {
    const agent = living(frame)[0];
    frame.next_end_spots = [
      { public_agent_id: agent.public_agent_id, position: [1, 1] },
    ];
    frame.recent_falls = [];
    await normalizeAuthorizedPresentationFrameV1(structuredClone(frame));
    const unknown = structuredClone(frame);
    unknown.next_end_spots = [{ public_agent_id: "agent-404", position: [1, 1] }];
    await assert.rejects(
      normalizeAuthorizedPresentationFrameV1(unknown),
      /Next end spots must name/u,
    );
    const late = structuredClone(frame);
    const tick = late.source.source_simulator_step_count;
    late.recent_falls = [
      {
        tick, // The shown tick itself belongs to the plan, not the history.
        victim_public_agent_id: agent.public_agent_id,
        killer_public_agent_ids: [],
        points: 1,
        ultimate_public_agent_id: null,
      },
    ];
    await assert.rejects(
      normalizeAuthorizedPresentationFrameV1(late),
      /Recent falls must lie/u,
    );
  }
  const final = structuredClone(fixture.state_cases.replay_oracle_final_unselected);
  assert.deepEqual(final.next_end_spots, []);
  await normalizeAuthorizedPresentationFrameV1(structuredClone(final));
  final.next_end_spots = [
    { public_agent_id: living(final)[0].public_agent_id, position: [1, 1] },
  ];
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(final),
    /final replay frame cannot carry next end spots/u,
  );
});
