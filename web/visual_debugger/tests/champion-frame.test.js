/**
 * @file Check the Champion Frame look (`src/looks/champion-frame.js`).
 *
 * Contracts: Silver Brackets sit on a square of half-size R + clamp(0.52R, 6,
 * 12) px with arms 0.38 of it, and become one ring at R + 4 px under R 10 px;
 * Selection Moonlight shows only for the selected living champion with its
 * switch on, as a separate floor node at R + 12 px; brackets carry
 * data-fallen on a fallen champion. The Next Action Thread is a straight line
 * from the actor's rim + 3 px to 6 px short of the target's rim, with a dot
 * (Basic) or diamond (Ultimate) R + 9 px below the target, a self-cast draws
 * only the glyph under the actor, no arrowhead is drawn, the switch turns it
 * off, and every node is export-safe. Reach Rings follow the Ranges button,
 * or with the Reach Rings switch only the selected champion's rings. The look
 * registers the selection paint parts with the visual filters. Agent POV:
 * arena marks come only from the POV scene, so an unseen champion gets no
 * marks; the docked frame shows the full record under the Full Record chip,
 * while its Recent Ticks cell reads only the audience's own events ("Not
 * Visible" for a champion the POV team did not see). The frame view gives the
 * exact hint row with no selection, names every status in the Champion Card
 * row's words with the source by class only, gives the shared Life bar's
 * model, reads "No Next Tick (Final Frame)" on the last frame, and needs the
 * describeAction helper. The look never reads a frame: the test composes its
 * plain input with the adapter as main.js does. renderChampionFrame builds the
 * hint or the selected frame, with the life drawn as the shared Life bar.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  authorizedPresentationAudience,
  authorizedPresentationHasResearcherSpace,
  authorizedPresentationIncomingRows,
  authorizedPresentationResearcherSceneView,
  authorizedPresentationSceneView,
} from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import * as frame from "../src/looks/champion-frame.js";
import { describeRecordedAction, fullRecordLabel } from "../src/panels.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualPaintPartEnabled,
  classifyVisualPaintPart,
} from "../src/visual-filters.js";

// The frame's input, read from a frame the way main.js does (see the OWNER-FILE
// note of this lane): the look itself never reads a frame.
/** @param {any} presentation @param {string | null} selectedPublicAgentId */
function frameInput(presentation, selectedPublicAgentId) {
  const pov = authorizedPresentationAudience(presentation) === "agent_pov";
  const upcoming = authorizedPresentationHasResearcherSpace(presentation)
    ? presentation.researcher_space.upcoming_transition
    : presentation.upcoming_transition;
  const events = pov ? presentation.visual_events : presentation.latest_events;
  return {
    selectedPublicAgentId,
    agents: authorizedPresentationResearcherSceneView(presentation)?.agents ?? [],
    chip: fullRecordLabel(presentation),
    audience: /** @type {"researcher" | "agent_pov"} */ (
      pov ? "agent_pov" : "researcher"
    ),
    incomingRows: authorizedPresentationIncomingRows(presentation),
    hasIncomingTick: events !== null && typeof events === "object",
    povSceneAgentIds: pov
      ? (authorizedPresentationSceneView(presentation)?.agents ?? []).map(
          (/** @type {any} */ agent) => agent.public_agent_id,
        )
      : null,
    replay: presentation.viewer_mode === "replay",
    upcomingActionRows:
      upcoming !== null &&
      typeof upcoming === "object" &&
      Array.isArray(upcoming.action_rows)
        ? upcoming.action_rows
        : null,
    describeAction: describeRecordedAction,
  };
}

/** @returns {any} */
function fakeDocument() {
  /** @param {string} tag @returns {any} */
  const node = (tag) => {
    /** @type {Map<string, string>} */
    const attributes = new Map();
    /** @type {any} */
    const element = {
      tagName: tag,
      children: [],
      parentElement: null,
      dataset: {},
      style: {},
      textContent: "",
      ownerDocument: null,
      get className() {
        return attributes.get("class") ?? "";
      },
      set className(value) {
        attributes.set("class", String(value));
      },
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      hasAttribute: (/** @type {string} */ name) => attributes.has(name),
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
      classList: {
        add: (/** @type {string} */ name) => {
          const names = new Set((attributes.get("class") ?? "").split(" "));
          names.add(name);
          attributes.set("class", [...names].filter(Boolean).join(" "));
        },
      },
      append: (/** @type {any[]} */ ...nodes) => {
        for (const child of nodes) {
          child.parentElement = element;
          element.children.push(child);
        }
      },
      replaceChildren: (/** @type {any[]} */ ...nodes) => {
        element.children = [];
        element.append(...nodes);
      },
      attributes,
    };
    element.ownerDocument = doc;
    return element;
  };
  /** @type {any} */
  const doc = {
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      node(tag),
    createElement: (/** @type {string} */ tag) => node(tag),
  };
  doc.documentElement = node("html");
  return doc;
}

/** @param {any} element @returns {any[]} */
function descendants(element) {
  return element.children.flatMap((/** @type {any} */ child) => [
    child,
    ...descendants(child),
  ]);
}

/** @param {any} element @param {string} className @returns {any[]} */
function byClass(element, className) {
  return [element, ...descendants(element)].filter((child) =>
    (child.getAttribute("class") ?? "").split(" ").includes(className),
  );
}

/** @param {any} element @returns {string} */
function serialize(element) {
  const attributes = [...element.attributes.entries()]
    .map(([name, value]) => ` ${name}="${value}"`)
    .join("");
  return `<${element.tagName}${attributes}>${element.textContent}${element.children.map(serialize).join("")}</${element.tagName}>`;
}

/** @param {string} path @returns {number[]} */
function numbers(path) {
  return [...path.matchAll(/-?\d+(?:\.\d+)?/gu)].map((match) => Number(match[0]));
}

const transform = {
  scale: 46,
  worldWidth: 20,
  worldHeight: 20,
  viewportBounds: { left: 0, top: 0, right: 920, bottom: 920, width: 920, height: 920 },
  mapBounds: { left: 0, top: 0, right: 920, bottom: 920, width: 920, height: 920 },
  worldToScreen: (/** @type {any} */ point) => {
    const [x, y] = Array.isArray(point) ? point : [point.x, point.y];
    return { x: x * 46, y: 920 - y * 46 };
  },
  worldLengthToScreen: (/** @type {number} */ length) => length * 46,
};

/** @param {Record<string, any>} [overrides] */
function route(overrides = {}) {
  return {
    source_presentation_key: "key-a",
    source_public_agent_id: "agent-slot-1",
    source_anchor: [2, 2],
    source_radius: 0.5,
    target_presentation_key: "key-b",
    target_public_agent_id: "agent-slot-6",
    target_anchor: [6, 2],
    target_radius: 0.5,
    lane: 0,
    legal: true,
    ...overrides,
  };
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

test("Silver Brackets follow the detail ladder and become a ring on small bodies", () => {
  assert.equal(frame.bracketOffset(23), 0.52 * 23);
  assert.equal(frame.bracketOffset(5), 6);
  assert.equal(frame.bracketOffset(40), 12);
  const path = frame.bracketPath(100, 100, 23);
  const half = 23 + 0.52 * 23;
  const [x0, y0, h0] = numbers(path);
  assert.ok(Math.abs(x0 - (100 - half + half * 0.38)) < 1e-9);
  assert.ok(Math.abs(y0 - (100 - half)) < 1e-9);
  assert.ok(Math.abs(h0 - (100 - half)) < 1e-9);
  assert.equal(path.match(/M /gu)?.length, 4);
  const ring = frame.bracketPath(50, 50, 8);
  assert.match(ring, / A 12 12 /u);
  assert.doesNotMatch(ring, / H /u);
});

test("selection marks show moonlight only for the selected living champion", () => {
  const champion = (/** @type {Record<string, any>} */ extra) => ({
    identityKey: "k",
    agent: { presentation_key: "k", public_agent_id: "agent-slot-1", alive: true },
    center: { x: 10, y: 20 },
    radius: 23,
    controlled: false,
    selected: true,
    ...extra,
  });
  const [lit] = frame.records({
    part: "selection-marks",
    champions: [champion({})],
    showSelectionReticle: true,
  });
  assert.equal(lit.showMoonlight, true);
  assert.equal(lit.alive, true);
  const [off] = frame.records({
    part: "selection-marks",
    champions: [champion({})],
    showSelectionMoonlight: false,
  });
  assert.equal(off.showMoonlight, false);
  const [other] = frame.records({
    part: "selection-marks",
    champions: [champion({ selected: false })],
  });
  assert.equal(other.showMoonlight, false);
  const [fallen] = frame.records({
    part: "selection-marks",
    champions: [
      champion({
        agent: { presentation_key: "k", public_agent_id: "agent-slot-1", alive: false },
      }),
    ],
    showSelectionReticle: true,
  });
  assert.equal(fallen.showMoonlight, false);
  assert.equal(fallen.alive, false);

  const doc = fakeDocument();
  const [bundle] = frame.buildSelectionMarks([lit], doc);
  frame.updateSelectionMarks([bundle], [lit]);
  assert.equal(bundle.moonlight.getAttribute("r"), "35");
  assert.equal(bundle.moonlight.getAttribute("hidden"), null);
  assert.equal(
    bundle.moonlight.parentElement,
    null,
    "moonlight waits for a floor layer",
  );
  assert.equal(bundle.selectedReticle.getAttribute("hidden"), null);
  assert.equal(
    bundle.selectedKeyline.getAttribute("d"),
    bundle.selectedReticle.getAttribute("d"),
  );
  assert.equal(bundle.controlledHalo.getAttribute("hidden"), "");
  frame.updateSelectionMarks([bundle], [fallen]);
  assert.equal(bundle.moonlight.getAttribute("hidden"), "");
  assert.equal(bundle.selectedReticle.getAttribute("data-fallen"), "true");
  assert.equal(bundle.selectedKeyline.getAttribute("data-fallen"), "true");
  assert.doesNotMatch(
    serialize(bundle.selectionRoot) + serialize(bundle.moonlight),
    /url\(|gradient|<mask|<filter|<image|filter=/iu,
  );
});

test("the Next Action Thread is a straight line with a 6 o'clock glyph", () => {
  const scene = { pending_route: route(), agents: [] };
  const [record] = frame.records({ part: "next-action-route", scene, transform });
  const [x1, y1, x2, y2] = numbers(record.path);
  // Actor at (92, 828), target at (276, 828), radius 23 px each.
  assert.deepEqual([x1, y1, x2, y2], [92 + 26, 828, 276 - 29, 828]);
  assert.deepEqual({ ...record.glyph }, { x: 276, y: 828 + 23 + 9, shape: "dot" });
  assert.equal(record.routeKind, "thread");
  const [ultimate] = frame.records({
    part: "next-action-route",
    scene: { pending_route: route({ lane: 1 }), agents: [] },
    transform,
  });
  assert.equal(ultimate.glyph.shape, "diamond");
  const [self] = frame.records({
    part: "next-action-route",
    scene: {
      pending_route: route({
        target_presentation_key: "key-a",
        target_public_agent_id: "agent-slot-1",
        target_anchor: [2, 2],
        lane: 1,
      }),
      agents: [],
    },
    transform,
  });
  assert.equal(self.path, null);
  assert.equal(self.routeKind, "self");
  assert.deepEqual({ ...self.glyph }, { x: 92, y: 828 + 23 + 9, shape: "diamond" });
  assert.deepEqual(
    frame.records({
      part: "next-action-route",
      scene,
      transform,
      showNextActionThread: false,
    }),
    [],
  );

  const doc = fakeDocument();
  const nodes = frame.build([record], doc);
  assert.deepEqual(
    nodes.map((node) => node.getAttribute("class")),
    ["pending-route", "next-action-glyph", "pending-route-hit"],
  );
  assert.equal(nodes[0].getAttribute("data-route-kind"), "thread");
  assert.equal(nodes.flatMap((node) => byClass(node, "pending-route-arrow")).length, 0);
  const selfNodes = frame.build([self], doc);
  assert.deepEqual(
    selfNodes.map((node) => node.getAttribute("class")),
    ["next-action-glyph", "pending-route-hit"],
  );
  assert.equal(selfNodes[0].getAttribute("data-shape"), "diamond");
  for (const node of [...nodes, ...selfNodes]) {
    assert.doesNotMatch(serialize(node), /url\(|gradient|<mask|<filter|<image/iu);
  }
});

test("reach rings follow Ranges, or the Reach Rings switch for the selection", () => {
  const scene = {
    agents: [
      { presentation_key: "a", public_agent_id: "agent-slot-1", class_id: 3 },
      { presentation_key: "b", public_agent_id: "agent-slot-6", class_id: 1 },
    ],
    selection: { selected_presentation_key: "a" },
    ranges: [
      { presentation_key: "a", kind: "basic", center: [2, 2], radius: 3.5 },
      { presentation_key: "b", kind: "basic", center: [6, 2], radius: 3 },
    ],
  };
  assert.equal(frame.records({ part: "reach-rings", scene, transform }).length, 2);
  assert.deepEqual(
    frame.records({ part: "reach-rings", scene, transform, showRanges: false }),
    [],
  );
  const selected = frame.records({
    part: "reach-rings",
    scene,
    transform,
    showRanges: false,
    showReachRings: true,
  });
  assert.equal(selected.length, 1);
  assert.equal(selected[0].identity.presentation_key, "a");
});

test("the selection paint parts are registered with the visual filters", () => {
  const filters = frame.PAINT_PARTS.map((row) => [row.part.kind, row.filter]);
  assert.deepEqual(filters, [
    ["selection_reticle", "target_selection_visuals"],
    ["selected_pair_legality", "target_selection_visuals"],
    ["selection_moonlight", "selection_moonlight"],
    ["next_action_thread", "next_action_thread"],
    ["reach_ring", "reach_rings"],
  ]);
  for (const row of frame.PAINT_PARTS) {
    assert.equal(classifyVisualPaintPart(row.part), row.filter);
    assert.equal(
      typeof isVisualPaintPartEnabled(DEFAULT_VISUAL_FILTER_STATE, row.part),
      "boolean",
    );
  }
});

test("Agent POV arena marks come only from the POV scene", async () => {
  const fx = await fixture();
  const pov = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(fx.presentations.replay_no_shared_obs_agent_pov),
  );
  const scene = /** @type {Record<string, any>} */ (
    authorizedPresentationSceneView(pov)
  );
  const seen = new Set(
    scene.agents.map((/** @type {any} */ agent) => agent.public_agent_id),
  );
  assert.ok(
    !seen.has("agent-slot-5"),
    "the fixture hides agent-slot-5 from the POV team",
  );
  const champions = scene.agents.map((/** @type {any} */ agent) => ({
    identityKey: agent.presentation_key,
    agent,
    center: { x: 0, y: 0 },
    radius: 20,
    controlled: false,
    selected: false,
  }));
  const marks = frame.records({ part: "selection-marks", champions });
  assert.ok(marks.every((mark) => seen.has(mark.agent.public_agent_id)));
  const rings = frame.records({ part: "reach-rings", scene, transform });
  assert.ok(
    rings.every((ring) =>
      scene.agents.some(
        (/** @type {any} */ agent) =>
          agent.presentation_key === ring.identity.presentation_key,
      ),
    ),
  );
});

test("the frame view: hint, full record chip, status rule, Recent Ticks and Next", async () => {
  const fx = await fixture();
  const oracle = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(fx.presentations.replay_oracle),
  );
  const pov = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(fx.presentations.replay_no_shared_obs_agent_pov),
  );
  const final = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(fx.state_cases.replay_oracle_final_selected),
  );
  assert.throws(
    () =>
      frame.championFrameView(
        /** @type {any} */ ({ ...frameInput(oracle, null), describeAction: null }),
      ),
    TypeError,
  );
  const empty = frame.championFrameView(frameInput(oracle, null));
  assert.deepEqual(empty, {
    state: "empty",
    hint: "Select An Agent To Inspect · Click A Body Or A Team Rosters Chip",
  });
  assert.equal(
    frame.championFrameView(frameInput(oracle, "agent-slot-99")).state,
    "empty",
  );

  const warrior = /** @type {any} */ (
    frame.championFrameView(frameInput(oracle, "agent-slot-1"))
  );
  assert.equal(warrior.state, "selected");
  assert.equal(warrior.chip, null);
  assert.equal(warrior.title, "Warrior · ID 1");
  assert.equal(warrior.teamLabel, "Team A");
  assert.equal(warrior.lifeText, "200 / 200");
  assert.deepEqual(warrior.life, {
    text: "200",
    maximumText: "200",
    fraction: 1,
    tone: "good",
  });
  assert.equal(warrior.ultimateText, "Ultimate Ready");
  // The Champion Card row's words, with the source by class only.
  assert.deepEqual(
    warrior.statuses.map((/** @type {any} */ status) => status.line),
    [
      "Blessing of Freedom (Slow Resistance) · 1 tick left · Slows can't drop it below 85% speed. From a Priest.",
    ],
  );
  assert.deepEqual(
    { ...warrior.recent },
    {
      state: "seen",
      ability: null,
      damage: null,
      heal: 8,
      rest: null,
      net: false,
      povOnly: false,
    },
  );
  assert.equal(warrior.nextText, "Next · No Action");
  assert.equal(
    /** @type {any} */ (frame.championFrameView(frameInput(oracle, "agent-slot-0")))
      .nextText,
    "Next · ◆ Burst",
  );

  const hidden = /** @type {any} */ (
    frame.championFrameView(frameInput(pov, "agent-slot-5"))
  );
  assert.equal(hidden.chip, "Full Record · Not Limited To Agent 0's View");
  assert.equal(hidden.title, "Hunter · ID 5", "the docked frame keeps the full record");
  assert.equal(hidden.recent.state, "not-visible");
  assert.equal(hidden.recent.damage, null);
  const seen = /** @type {any} */ (
    frame.championFrameView(frameInput(pov, "agent-slot-1"))
  );
  assert.equal(seen.recent.state, "seen");
  assert.equal(seen.recent.povOnly, true);
  assert.equal(seen.recent.heal, null, "Agent POV never borrows the Oracle totals");

  const last = /** @type {any} */ (
    frame.championFrameView(frameInput(final, "agent-slot-0"))
  );
  assert.equal(last.nextText, "No Next Tick (Final Frame)");
  assert.equal(last.finalFrame, true);
  assert.equal(last.ultimateText, "Burst · Ready In 29 Ticks");
});

test("renderChampionFrame builds the hint row or the selected frame", async () => {
  const fx = await fixture();
  const pov = await normalizeAuthorizedPresentationFrameV1(
    structuredClone(fx.presentations.replay_no_shared_obs_agent_pov),
  );
  const doc = fakeDocument();
  const container = doc.createElement("section");
  frame.renderChampionFrame(container, frame.championFrameView(frameInput(pov, null)));
  assert.equal(container.dataset.state, "empty");
  assert.equal(container.children.length, 1);
  assert.equal(container.children[0].textContent, frame.CHAMPION_FRAME_HINT);

  frame.renderChampionFrame(
    container,
    frame.championFrameView(frameInput(pov, "agent-slot-1")),
  );
  assert.equal(container.dataset.state, "selected");
  assert.equal(container.dataset.class, "warrior");
  assert.deepEqual(
    container.children.map((/** @type {any} */ child) => child.getAttribute("class")),
    [
      "champion-frame__chip",
      "champion-frame__portrait",
      "champion-frame__name",
      "champion-frame__statuses",
      "champion-frame__recent",
      "champion-frame__next",
    ],
  );
  assert.equal(
    container.children[0].textContent,
    "Full Record · Not Limited To Agent 0's View",
  );
  const portrait = container.children[1].children[0];
  assert.equal(portrait.getAttribute("viewBox"), "0 0 46 46");
  assert.equal(portrait.getAttribute("aria-hidden"), "true");
  assert.ok(portrait.children.length > 0, "the portrait holds a token");
  assert.equal(byClass(container, "champion-frame__status").length, 1);
  // The life is the one shared Life bar.
  const [bar] = byClass(container, "life-bar");
  assert.equal(bar.dataset.life, "good");
  assert.deepEqual(
    bar.children.map((/** @type {any} */ child) => child.getAttribute("class")),
    ["life-bar__label", "life-bar__track", "life-bar__value"],
  );
});
