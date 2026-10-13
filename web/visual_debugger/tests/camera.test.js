/**
 * @file Check the Camera look (src/looks/camera.js). Everyone in View frames
 * the START and END spots of every living, falling and returning body with a
 * 1.5-unit margin, keeps the map's aspect, zooms only in 0.25 steps up to 2x,
 * stays within 0.45 units of the map, sits near the map centre and never cuts a
 * drawn body; with no bodies it is the Whole Map. Battle Cam frames the fights
 * at the Whole Map or 1.5x, keeps the Whole Map with Auto-Frame Fights off, and
 * ignores fights that are not in the audience's plan camera. Kill Cam leans
 * only on a seen fall, with ordered windows inside the tick and keyframes that
 * end on the resting frame (fill "backwards"). Kill Slow Motion halves the
 * chosen speed (never below 0.25x) only in the Replay Viewer's Battle Cam in
 * play. Camera switches: framing off gives Whole Map; B does nothing outside
 * the Replay Viewer. Tip texts are quoted exactly. The remembered camera maps
 * the old "director" value to Battle Cam, and every storage failure is safe.
 * There is no Camera Inset any more (owner, 4 Oct 2026): records never give
 * a "camera-inset" record and the switch draws no picture.
 * Agent POV: over the browser fixture's POV frames, changing hidden champions'
 * researcher rows and match summary rows gives identical camera records in
 * every camera. The switch component shows the tips, remembers the camera,
 * handles V and B through the one command set and applies frames with the
 * right ease. The New Camera Tip switch is read live: Off hides a showing
 * first-open tip at once, On shows a tip not seen yet, and once seen it stays
 * seen. The first-open tip button has its own Battle Cam hover help.
 * switchCameras: the Replay Viewer's switch with [] (the owner's 4 Oct 2026
 * key stack: the arena's V and B keys are the camera's buttons) has no
 * buttons; V and B still change the camera and the slot's data-camera names
 * it. tipHost: the
 * Replay Viewer's change tip and first-open tip mount in the given host (its
 * old children replaced, its hidden flag cleared) and not in the slot; live
 * play ignores tipHost.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { authorizedPresentationSceneView } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { buildChoreographyPlan } from "../src/choreography-plan.js";
import { runLocalCommand } from "../src/command-set.js";
import { createViewportTransform } from "../src/layout.js";
import {
  BODY_REACH,
  BUDGET,
  battleCamFrame,
  CAMERA_NAMES,
  CHANGE_TIP_MS,
  cameraChangeTip,
  cameraEase,
  cameraFrameFor,
  camerasFor,
  createCameraSwitch,
  EDGE_INSET,
  effectiveCamera,
  everyoneInViewFrame,
  FIRST_OPEN_TIP,
  FIRST_OPEN_TIP_MS,
  FIRST_OPEN_TIP_SEEN_KEY,
  FOOTPRINT,
  firstOpenTipDue,
  killCamKeyframes,
  killCamLean,
  killSlowMotion,
  MAP_EDGE_ALLOWANCE,
  nextCamera,
  PAINT_PARTS,
  REMEMBERED_CAMERA_KEY,
  readRememberedCamera,
  records,
  rememberCamera,
  tooltip,
} from "../src/looks/camera.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  enableAllVisualFilters,
} from "../src/visual-filters.js";

const W = 20;
const H = 10;
const TOL = 1e-6;

/** @param {number} seed */
function seeded(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/**
 * @param {Array<[number, number] | null>} starts
 * @param {Array<[number, number]>} ends
 * @param {Array<"living" | "falling" | "returning">} [states]
 * @returns {any}
 */
function campCamera(starts, ends, states = []) {
  return {
    audience: "researcher",
    bodies: ends.map((end, index) => ({
      publicAgentId: `agent-${index}`,
      start: starts[index] ?? null,
      end,
      state: states[index] ?? "living",
    })),
    fights: [],
  };
}

/**
 * @param {{minX: number, minY: number, maxX: number, maxY: number}} frame
 * @param {readonly [number, number]} point
 */
function wholeInside(frame, point) {
  const reach = BODY_REACH + EDGE_INSET - TOL;
  return (
    point[0] - reach >= frame.minX - TOL &&
    point[0] + reach <= frame.maxX + TOL &&
    point[1] - reach >= frame.minY - TOL &&
    point[1] + reach <= frame.maxY + TOL
  );
}

/**
 * @param {{minX: number, minY: number, maxX: number, maxY: number}} frame
 * @param {readonly [number, number]} point
 */
function notCut(frame, point) {
  const outsideX =
    point[0] + BODY_REACH <= frame.minX + TOL ||
    point[0] - BODY_REACH >= frame.maxX - TOL;
  const outsideY =
    point[1] + BODY_REACH <= frame.minY + TOL ||
    point[1] - BODY_REACH >= frame.maxY - TOL;
  const insideX =
    point[0] - BODY_REACH >= frame.minX - TOL &&
    point[0] + BODY_REACH <= frame.maxX + TOL;
  const insideY =
    point[1] - BODY_REACH >= frame.minY - TOL &&
    point[1] + BODY_REACH <= frame.maxY + TOL;
  return outsideX || outsideY || (insideX && insideY);
}

test("cameras, keys, framing switch and tip texts follow the command set", () => {
  assert.deepEqual(camerasFor("replay_viewer"), [
    "whole_map",
    "everyone_in_view",
    "battle_cam",
  ]);
  assert.deepEqual(camerasFor("combat_debugger"), ["whole_map", "everyone_in_view"]);
  assert.equal(effectiveCamera("battle_cam", "combat_debugger"), "whole_map");
  assert.equal(
    effectiveCamera("everyone_in_view", "replay_viewer", { camera_framing: false }),
    "whole_map",
  );
  assert.equal(effectiveCamera("nonsense", "replay_viewer"), "whole_map");
  assert.equal(nextCamera("whole_map", "v", "replay_viewer"), "everyone_in_view");
  assert.equal(nextCamera("everyone_in_view", "V", "replay_viewer"), "whole_map");
  assert.equal(nextCamera("everyone_in_view", "b", "replay_viewer"), "battle_cam");
  assert.equal(nextCamera("battle_cam", "B", "replay_viewer"), "whole_map");
  assert.equal(nextCamera("battle_cam", "v", "replay_viewer"), "everyone_in_view");
  assert.equal(nextCamera("whole_map", "b", "combat_debugger"), "whole_map");
  assert.equal(nextCamera("whole_map", "x", "replay_viewer"), "whole_map");
  assert.equal(
    cameraChangeTip("battle_cam", "replay_viewer"),
    "Battle Cam on · B: back to Whole Map · V: Everyone in View",
  );
  assert.equal(
    cameraChangeTip("everyone_in_view", "replay_viewer"),
    "Everyone in View on · V: back to Whole Map · B: Battle Cam",
  );
  assert.equal(
    cameraChangeTip("everyone_in_view", "combat_debugger"),
    "Everyone in View on · V: back to Whole Map",
  );
  assert.equal(
    cameraChangeTip("whole_map", "combat_debugger"),
    "Whole Map on · V: Everyone in View",
  );
  assert.equal(
    FIRST_OPEN_TIP,
    "New: Battle Cam follows the fights for you. Click here or press B.",
  );
  assert.deepEqual(CAMERA_NAMES, {
    whole_map: "Whole Map",
    everyone_in_view: "Everyone in View",
    battle_cam: "Battle Cam",
  });
  assert.equal(cameraEase({ reason: "camera" }), "ease");
  assert.equal(cameraEase({ reason: "play" }), "ease");
  assert.equal(cameraEase({ reason: "view" }), "cut");
  assert.equal(cameraEase({ reason: "recipient" }), "cut");
  assert.equal(cameraEase({ reason: "camera", reducedMotion: true }), "cut");
  assert.deepEqual(PAINT_PARTS, []);
  assert.deepEqual(FOOTPRINT, {});
  assert.equal(BUDGET.nodesPerRecord, 0);
});

test("the remembered camera maps old names and survives every storage failure", () => {
  /** @param {string | null} value */
  const store = (value) => ({ getItem: () => value });
  assert.equal(readRememberedCamera(store("director")), "battle_cam");
  assert.equal(readRememberedCamera(store("Battle Director")), "battle_cam");
  assert.equal(readRememberedCamera(store("everyone_in_view")), "everyone_in_view");
  assert.equal(readRememberedCamera(store("Everyone in View")), "everyone_in_view");
  assert.equal(readRememberedCamera(store("zoom")), "whole_map");
  assert.equal(readRememberedCamera(store(null)), "whole_map");
  assert.equal(readRememberedCamera(null), "whole_map");
  const throwing = {
    getItem() {
      throw new Error("blocked");
    },
    setItem() {
      throw new Error("blocked");
    },
  };
  assert.equal(readRememberedCamera(throwing), "whole_map");
  assert.equal(rememberCamera(throwing, "battle_cam"), false);
  assert.equal(rememberCamera(null, "battle_cam"), false);
  assert.equal(firstOpenTipDue(throwing), false);
  assert.equal(firstOpenTipDue(null), false);
  /** @type {Map<string, string>} */
  const saved = new Map();
  const memory = {
    getItem: (/** @type {string} */ key) => saved.get(key) ?? null,
    setItem: (/** @type {string} */ key, /** @type {string} */ value) => {
      saved.set(key, value);
    },
  };
  assert.equal(firstOpenTipDue(memory), true);
  assert.equal(rememberCamera(memory, "battle_cam"), true);
  assert.equal(saved.get(REMEMBERED_CAMERA_KEY), "battle_cam");
});

test("Everyone in View keeps every spot, zooms in 0.25 steps and never cuts a body", () => {
  const random = seeded(7);
  let zoomed = 0;
  for (let trial = 0; trial < 400; trial += 1) {
    const count = 1 + Math.floor(random() * 6);
    const cx = 1 + random() * 18;
    const cy = 1 + random() * 8;
    const spread = random() * 6;
    /** @type {Array<[number, number]>} */
    const ends = [];
    /** @type {Array<[number, number] | null>} */
    const starts = [];
    /** @type {Array<"living" | "falling" | "returning">} */
    const states = [];
    for (let index = 0; index < count; index += 1) {
      const clampX = (/** @type {number} */ v) => Math.min(19.5, Math.max(0.5, v));
      const clampY = (/** @type {number} */ v) => Math.min(9.5, Math.max(0.5, v));
      const end = /** @type {[number, number]} */ ([
        clampX(cx + (random() - 0.5) * spread),
        clampY(cy + (random() - 0.5) * spread),
      ]);
      ends.push(end);
      starts.push(
        random() < 0.3
          ? null
          : [clampX(end[0] + random() - 0.5), clampY(end[1] + random() - 0.5)],
      );
      states.push(random() < 0.1 ? "falling" : random() < 0.1 ? "returning" : "living");
    }
    const corpses = [
      [clampAny(random() * 20, 0.5, 19.5), clampAny(random() * 10, 0.5, 9.5)],
    ];
    const camera = campCamera(starts, ends, states);
    const scene = { agents: corpses.map((position) => ({ position })) };
    const { frame, zoom } = everyoneInViewFrame({
      camera,
      worldWidth: W,
      worldHeight: H,
      scene,
    });
    const width = frame.maxX - frame.minX;
    const height = frame.maxY - frame.minY;
    assert.ok(Math.abs(width / height - W / H) < 1e-6, "keeps the map's aspect");
    assert.ok(Math.abs(W / width - zoom) < 1e-6);
    assert.ok(zoom >= 1 && zoom <= 2 + TOL);
    assert.ok(
      Math.abs(zoom * 4 - Math.round(zoom * 4)) < 1e-6,
      `zoom ${zoom} is a 0.25 step`,
    );
    assert.ok(
      frame.minX >= -MAP_EDGE_ALLOWANCE - TOL &&
        frame.maxX <= W + MAP_EDGE_ALLOWANCE + TOL,
    );
    assert.ok(
      frame.minY >= -MAP_EDGE_ALLOWANCE - TOL &&
        frame.maxY <= H + MAP_EDGE_ALLOWANCE + TOL,
    );
    if (zoom > 1) {
      zoomed += 1;
      for (const body of camera.bodies) {
        assert.ok(wholeInside(frame, body.end), "every END spot is in view");
        if (body.state !== "returning" && body.start) {
          assert.ok(wholeInside(frame, body.start), "every START spot is in view");
        }
      }
      for (const spot of [...ends, ...corpses]) {
        assert.ok(
          notCut(frame, /** @type {[number, number]} */ (spot)),
          "no drawn body is cut",
        );
      }
    }
  }
  assert.ok(zoomed > 100, `many trials zoom in (${zoomed})`);
});

/**
 * @param {number} value
 * @param {number} low
 * @param {number} high
 */
function clampAny(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

test("Everyone in View sits near the map centre, and is the Whole Map with no bodies", () => {
  const empty = everyoneInViewFrame({
    camera: campCamera([], []),
    worldWidth: W,
    worldHeight: H,
  });
  assert.deepEqual(empty, { frame: { minX: 0, minY: 0, maxX: W, maxY: H }, zoom: 1 });
  const camera = campCamera([[9, 5]], [[11, 5]]);
  const { frame, zoom } = everyoneInViewFrame({
    camera,
    worldWidth: W,
    worldHeight: H,
  });
  assert.equal(zoom, 2);
  assert.ok(Math.abs((frame.minX + frame.maxX) / 2 - W / 2) < TOL);
  assert.ok(Math.abs((frame.minY + frame.maxY) / 2 - H / 2) < TOL);
  // The frame stays put while the group moves a little inside it.
  const moved = everyoneInViewFrame({
    camera: campCamera([[9.5, 5]], [[10.5, 5.5]]),
    worldWidth: W,
    worldHeight: H,
  });
  assert.deepEqual(moved.frame, frame);
  assert.throws(
    () => everyoneInViewFrame({ camera, worldWidth: 0, worldHeight: H }),
    TypeError,
  );
});

test("Battle Cam frames the fight at 1.5x or the Whole Map, and only fights it is given", () => {
  const camera = {
    ...campCamera(
      [
        [3, 3],
        [4, 3],
        [16, 8],
      ],
      [
        [3, 3],
        [4, 3],
        [16, 8],
      ],
    ),
    fights: [
      {
        kind: "blow",
        points: [
          [3, 3],
          [4, 3],
        ],
      },
    ],
  };
  const fight = battleCamFrame({ camera, worldWidth: W, worldHeight: H });
  assert.equal(fight.zoom, 1.5);
  assert.ok(wholeInside(fight.frame, [3, 3]) && wholeInside(fight.frame, [4, 3]));
  assert.ok(notCut(fight.frame, [16, 8]));
  const off = battleCamFrame({
    camera,
    worldWidth: W,
    worldHeight: H,
    filters: { auto_frame_fights: false },
  });
  assert.equal(off.zoom, 1);
  const spread = {
    ...camera,
    fights: [
      {
        kind: "blow",
        points: [
          [1, 1],
          [19, 9],
        ],
      },
    ],
  };
  assert.equal(
    battleCamFrame({ camera: spread, worldWidth: W, worldHeight: H }).zoom,
    1,
  );
  // A fight the audience did not see is simply absent from its plan camera:
  // the frame is the one of a tick with no fight at all.
  const unseen = { ...camera, fights: [] };
  const withoutFight = battleCamFrame({
    camera: unseen,
    worldWidth: W,
    worldHeight: H,
  });
  assert.deepEqual(
    withoutFight,
    battleCamFrame({ camera: unseen, worldWidth: W, worldHeight: H }),
  );
  assert.notDeepEqual(withoutFight.frame, fight.frame);
});

/** @type {any} */
const BEATS = Object.freeze({
  windows: Object.freeze({ impact: Object.freeze([280, 900]) }),
  fall: Object.freeze([1500, 2400]),
  returnBeat: null,
  tickMs: 3200,
});

test("Kill Cam leans only on a seen fall, inside the tick, and ends on the resting frame", () => {
  const base = campCamera(
    [
      [5, 5],
      [6, 5],
      [15, 5],
    ],
    [
      [5, 5],
      [6, 5],
      [15, 5],
    ],
    ["falling", "living", "living"],
  );
  const camera = {
    ...base,
    fights: [
      {
        kind: "blow",
        points: [
          [6, 5],
          [5, 5],
        ],
      },
      { kind: "fall", points: [[5, 5]] },
    ],
  };
  const restFrame = { minX: 0, minY: 0, maxX: W, maxY: H };
  const lean = killCamLean({
    camera,
    worldWidth: W,
    worldHeight: H,
    restFrame,
    beats: BEATS,
  });
  assert.ok(lean);
  assert.ok(lean.zoom >= 1.05 - TOL && lean.zoom <= 1.5 + TOL);
  assert.ok(wholeInside(lean.frame, [5, 5]) && wholeInside(lean.frame, [6, 5]));
  const { push, hold, back } = lean.windows;
  assert.ok(push[0] < push[1] && push[1] === hold[0] && hold[1] === back[0]);
  assert.ok(back[1] <= BEATS.tickMs - 180 + TOL);
  assert.equal(
    killCamLean({
      camera: { ...camera, fights: [] },
      worldWidth: W,
      worldHeight: H,
      restFrame,
      beats: BEATS,
    }),
    null,
  );
  assert.equal(
    killCamLean({
      camera,
      worldWidth: W,
      worldHeight: H,
      restFrame,
      beats: { ...BEATS, fall: null },
    }),
    null,
  );
  const transform = (/** @type {any} */ frame) =>
    createViewportTransform({
      worldWidth: W,
      worldHeight: H,
      viewportWidth: 1000,
      viewportHeight: 520,
      padding: 10,
      cameraFrame: frame,
    });
  const motion = killCamLean({
    camera,
    worldWidth: W,
    worldHeight: H,
    restFrame,
    beats: BEATS,
  });
  const spec = killCamKeyframes(
    /** @type {any} */ (motion),
    transform(restFrame),
    transform(lean.frame),
    transform(lean.creepFrame),
    BEATS.tickMs,
  );
  assert.equal(spec.options.fill, "backwards");
  assert.equal(spec.options.duration, BEATS.tickMs);
  assert.equal(spec.keyframes.at(-1)?.transform, "matrix(1, 0, 0, 1, 0, 0)");
  assert.equal(spec.keyframes[0].transform, "matrix(1, 0, 0, 1, 0, 0)");
  const offsets = spec.keyframes.map((frame) => Number(frame.offset));
  assert.deepEqual(
    [...offsets].sort((a, b) => a - b),
    offsets,
  );
});

test("Kill Slow Motion halves the chosen speed only in the Replay Viewer's Battle Cam in play", () => {
  const camera = {
    ...campCamera([[5, 5]], [[5, 5]], ["falling"]),
    fights: [{ kind: "fall", points: [[5, 5]] }],
  };
  const input = {
    camera,
    beats: BEATS,
    camera_name: /** @type {const} */ ("battle_cam"),
    product: "replay_viewer",
    playing: true,
    chosenRate: 1,
  };
  const slow = killSlowMotion(input);
  assert.ok(slow);
  assert.equal(slow.rate, 0.5);
  assert.equal(slow.label, "Slow Motion · 0.5×");
  assert.deepEqual(slow.windows, [
    [280, 900],
    [1500, 2400],
  ]);
  assert.equal(killSlowMotion({ ...input, chosenRate: 0.4 })?.rate, 0.25);
  for (const change of [
    { chosenRate: 0.25 },
    { chosenRate: 1.5 },
    { product: "combat_debugger" },
    { playing: false },
    { reducedMotion: true },
    { camera_name: /** @type {const} */ ("everyone_in_view") },
    { filters: { kill_slow_motion: false } },
    { camera: { ...camera, fights: [] } },
  ]) {
    assert.equal(killSlowMotion({ ...input, ...change }), null, JSON.stringify(change));
  }
});

test("records give one frame record and play-only moments, and no inset", () => {
  const camera = {
    ...campCamera(
      [
        [5, 5],
        [6, 5],
      ],
      [
        [5, 5],
        [6, 5],
      ],
      ["falling", "living"],
    ),
    fights: [
      {
        kind: "blow",
        points: [
          [6, 5],
          [5, 5],
        ],
      },
      { kind: "fall", points: [[5, 5]] },
    ],
  };
  const scene = {
    agents: [{ public_agent_id: "agent-1", team_id: 2, position: [6, 5] }],
  };
  const input = {
    camera,
    beats: BEATS,
    scene,
    worldWidth: W,
    worldHeight: H,
    product: "replay_viewer",
    camera_name: "battle_cam",
    playing: true,
    chosenRate: 1,
  };
  const list = records(input);
  assert.deepEqual(
    list.map((record) => record.kind),
    ["camera-frame", "kill-cam", "slow-motion"],
  );
  const still = records({ ...input, playing: false });
  assert.deepEqual(
    still.map((record) => record.kind),
    ["camera-frame"],
  );
  assert.deepEqual(
    records({ ...input, filters: { kill_cam: false } }).map((r) => r.kind),
    ["camera-frame", "slow-motion"],
  );
  const live = records({ ...input, product: "combat_debugger" });
  assert.deepEqual(live, [
    { kind: "camera-frame", camera: "whole_map", frame: null, zoom: 1 },
  ]);
  const off = records({ ...input, filters: { camera_framing: false } });
  assert.deepEqual(off, [
    { kind: "camera-frame", camera: "whole_map", frame: null, zoom: 1 },
  ]);
  assert.deepEqual(records({ camera }), []);
  assert.equal(cameraFrameFor({ ...input, camera_name: "whole_map" }), null);
  const everyone = cameraFrameFor({ ...input, camera_name: "everyone_in_view" });
  assert.equal(everyone?.camera, "everyone_in_view");

  const help = tooltip({
    kind: "camera-choice",
    camera: "battle_cam",
    product: "replay_viewer",
  });
  assert.equal(help?.title, "Battle Cam");
  const firstTipHelp = tooltip({ kind: "camera-first-tip" });
  assert.equal(firstTipHelp?.title, "Battle Cam");
  assert.equal(
    firstTipHelp?.summary,
    "Turn on Battle Cam: the camera follows the fights for you.",
  );
  assert.equal(tooltip({ kind: "camera-frame" }), null);
});

/** @returns {Promise<Record<string, any>>} */
async function fixture() {
  return JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
}

const surface = Object.freeze({
  worldToScreen: (/** @type {any} */ point) => ({
    x: 24 + Number(point.x ?? point[0]) * 40,
    y: 664 - Number(point.y ?? point[1]) * 40,
  }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 40,
  viewportBounds: Object.freeze({
    left: 24,
    top: 24,
    right: 1064,
    bottom: 664,
    width: 1040,
    height: 640,
  }),
  protectedRects: Object.freeze([]),
});

/** @param {Record<string, any>} raw */
function hiddenVariants(raw) {
  const seen = new Set(
    raw.current_endpoint.parts.scene.agents.map(
      (/** @type {any} */ agent) => agent.public_agent_id,
    ),
  );
  const hidden = (/** @type {Record<string, any>} */ copy) =>
    copy.researcher_space.roster_agents.filter(
      (/** @type {any} */ agent) => !seen.has(agent.public_agent_id),
    );
  if (hidden(raw).length === 0) return [];
  const life = structuredClone(raw);
  for (const agent of hidden(life)) {
    if (agent.life_state === "alive") agent.current_health = 1;
    if (Array.isArray(agent.position)) agent.position = [1, 1];
  }
  const variants = [life];
  if (raw.match_summary) {
    const victim = hidden(raw)[0];
    const death = structuredClone(raw);
    death.match_summary.deaths = [
      {
        public_agent_id: victim.public_agent_id,
        team_id: victim.team_id,
        class_id: victim.class_id,
        killing_team_id: 3 - victim.team_id,
        contributors: raw.researcher_space.roster_agents
          .filter((/** @type {any} */ agent) => agent.team_id !== victim.team_id)
          .map((/** @type {any} */ agent) => ({
            public_agent_id: agent.public_agent_id,
            team_id: agent.team_id,
            class_id: agent.class_id,
          })),
      },
    ];
    variants.push(death);
    const returned = structuredClone(raw);
    returned.match_summary.returned_agent_ids = [victim.public_agent_id];
    variants.push(returned);
  }
  return variants;
}

/** @param {unknown} raw */
async function cameraView(raw) {
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const scene = authorizedPresentationSceneView(frame);
  /** @type {unknown[]} */
  const out = [];
  for (const filters of [
    DEFAULT_VISUAL_FILTER_STATE,
    enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE),
  ]) {
    const plan = buildChoreographyPlan(frame, surface, filters);
    for (const camera_name of ["whole_map", "everyone_in_view", "battle_cam"]) {
      out.push(
        records({
          camera: plan?.camera ?? null,
          beats: plan?.beats ?? null,
          scene,
          worldWidth: scene?.map?.width ?? W,
          worldHeight: scene?.map?.height ?? H,
          filters,
          product: "replay_viewer",
          camera_name,
          playing: true,
          chosenRate: 1,
        }),
      );
    }
  }
  return out;
}

test("Agent POV camera records ignore hidden champions' researcher and match rows", async () => {
  const data = await fixture();
  const raws = Object.values(data.presentations).filter(
    (raw) =>
      typeof raw?.presentation_kind === "string" &&
      raw.presentation_kind.endsWith("agent_pov") &&
      raw.researcher_space,
  );
  let checked = 0;
  for (const raw of raws) {
    const variants = hiddenVariants(raw);
    if (variants.length === 0) continue;
    const expected = await cameraView(raw);
    for (const variant of variants) {
      assert.deepEqual(await cameraView(variant), expected);
      checked += 1;
    }
  }
  assert.ok(checked > 0, "at least one POV frame with hidden champions was checked");
});

test("the switch remembers the camera, shows tips, handles V and B, and applies frames", () => {
  const doc = fakeDocument();
  /** @type {Map<string, string>} */
  const saved = new Map([[REMEMBERED_CAMERA_KEY, "director"]]);
  const storage = /** @type {any} */ ({
    getItem: (/** @type {string} */ key) => saved.get(key) ?? null,
    setItem: (/** @type {string} */ key, /** @type {string} */ value) =>
      saved.set(key, value),
  });
  /** @type {Array<{ms: number, callback: () => void, cleared: boolean}>} */
  const pending = [];
  const timers = {
    setTimeout: (/** @type {() => void} */ callback, /** @type {number} */ ms) => {
      const entry = { ms, callback, cleared: false };
      pending.push(entry);
      return entry;
    },
    clearTimeout: (/** @type {any} */ entry) => {
      if (entry) entry.cleared = true;
    },
  };
  /** @type {Array<[unknown, boolean]>} */
  const applied = [];
  const slot = doc.createElement("div");
  slot.hidden = true;
  const camera = createCameraSwitch({
    slot: /** @type {any} */ (slot),
    product: "replay_viewer",
    storage,
    timers,
    applyFrame: (frame, { ease }) => {
      applied.push([frame, ease]);
      // The page re-renders and calls update again; the switch must ignore it.
      camera.update({ worldWidth: W, worldHeight: H, reason: "play" });
    },
  });
  try {
    assert.equal(
      camera.camera(),
      "battle_cam",
      "an old 'director' value opens Battle Cam",
    );
    assert.equal(slot.hidden, false);
    const firstTip = find(slot, "camera-switch__first-tip");
    assert.equal(firstTip.hidden, false);
    assert.equal(firstTip.textContent, FIRST_OPEN_TIP);
    assert.equal(saved.get(FIRST_OPEN_TIP_SEEN_KEY), "1");
    assert.ok(pending.some((entry) => entry.ms === FIRST_OPEN_TIP_MS));
    const fightCamera = {
      ...campCamera(
        [
          [3, 3],
          [4, 3],
        ],
        [
          [3, 3],
          [4, 3],
        ],
      ),
      fights: [
        {
          kind: "blow",
          points: [
            [3, 3],
            [4, 3],
          ],
        },
      ],
    };
    camera.update({
      camera: fightCamera,
      worldWidth: W,
      worldHeight: H,
      reason: "seek",
    });
    assert.equal(applied.length, 1);
    assert.equal(applied[0][1], false, "a seek cuts");
    assert.equal(/** @type {any} */ (applied[0][0]).camera, "battle_cam");
    camera.update({
      camera: fightCamera,
      worldWidth: W,
      worldHeight: H,
      reason: "play",
    });
    assert.equal(applied.length, 1, "an unchanged frame is not applied again");
    const pressed = runLocalCommand("replay_viewer", /** @type {any} */ ({ key: "v" }));
    assert.equal(pressed, true);
    assert.equal(camera.camera(), "everyone_in_view");
    assert.equal(saved.get(REMEMBERED_CAMERA_KEY), "everyone_in_view");
    assert.equal(applied.at(-1)?.[1], true, "a camera change eases");
    const tip = find(slot, "camera-switch__tip");
    assert.equal(tip.hidden, false);
    assert.equal(
      tip.textContent,
      "Everyone in View on · V: back to Whole Map · B: Battle Cam",
    );
    assert.equal(firstTip.hidden, true, "a camera change ends the first-open tip");
    assert.ok(pending.some((entry) => entry.ms === CHANGE_TIP_MS && !entry.cleared));
    doc.dispatch("keydown", {});
    assert.equal(tip.hidden, true, "the next input hides the tip");
    assert.equal(
      runLocalCommand("replay_viewer", /** @type {any} */ ({ key: "v" })),
      true,
    );
    assert.equal(camera.camera(), "whole_map");
    assert.equal(applied.at(-1)?.[0], null, "Whole Map applies a null frame");
    const pressedB = find(slot, "camera-switch__choice", "battle_cam");
    pressedB.click();
    assert.equal(camera.camera(), "battle_cam");
    assert.equal(pressedB.getAttribute("aria-pressed"), "true");
  } finally {
    camera.destroy();
  }
  assert.equal(
    runLocalCommand("replay_viewer", /** @type {any} */ ({ key: "v" })),
    false,
  );

  const liveSlot = doc.createElement("div");
  const live = createCameraSwitch({
    slot: /** @type {any} */ (liveSlot),
    product: "combat_debugger",
    storage,
    timers,
    applyFrame: () => {},
    getFilters: () => ({ camera_framing: false }),
  });
  try {
    assert.equal(
      live.camera(),
      "whole_map",
      "the BattleClient always opens on Whole Map",
    );
    assert.equal(liveSlot.hidden, true, "framing off hides the switch");
    assert.equal(live.pressKey("v"), false, "V does nothing with framing off");
    assert.equal(live.pressKey("b"), false);
    assert.equal(
      liveSlot.children.some(
        (/** @type {any} */ child) => child.className === "camera-switch__first-tip",
      ),
      false,
    );
  } finally {
    live.destroy();
  }
});

test("the New Camera Tip switch is read live: Off hides a showing first-open tip", () => {
  const doc = fakeDocument();
  /** @type {Map<string, string>} */
  const saved = new Map();
  const storage = /** @type {any} */ ({
    getItem: (/** @type {string} */ key) => saved.get(key) ?? null,
    setItem: (/** @type {string} */ key, /** @type {string} */ value) =>
      saved.set(key, value),
  });
  const timers = { setTimeout: () => ({}), clearTimeout: () => {} };
  /** @type {Record<string, boolean>} */
  const filters = { camera_tip: true };
  const slot = doc.createElement("div");
  const camera = createCameraSwitch({
    slot: /** @type {any} */ (slot),
    product: "replay_viewer",
    storage,
    timers,
    applyFrame: () => {},
    getFilters: () => filters,
    registerKeys: false,
  });
  const repaint = () =>
    camera.update({ worldWidth: W, worldHeight: H, reason: "resize" });
  try {
    const firstTip = find(slot, "camera-switch__first-tip");
    assert.equal(firstTip.hidden, false, "the first open shows the tip");
    filters.camera_tip = false;
    repaint();
    assert.equal(firstTip.hidden, true, "Off hides the showing tip at once");
    filters.camera_tip = true;
    repaint();
    assert.equal(firstTip.hidden, true, "once seen, the tip stays seen");
    assert.equal(saved.get(FIRST_OPEN_TIP_SEEN_KEY), "1");
  } finally {
    camera.destroy();
  }

  // Opened with the switch Off: nothing shows and nothing is marked seen;
  // turning it On lets the unseen tip show.
  saved.clear();
  filters.camera_tip = false;
  const laterSlot = doc.createElement("div");
  const later = createCameraSwitch({
    slot: /** @type {any} */ (laterSlot),
    product: "replay_viewer",
    storage,
    timers,
    applyFrame: () => {},
    getFilters: () => filters,
    registerKeys: false,
  });
  try {
    const firstTip = find(laterSlot, "camera-switch__first-tip");
    assert.equal(firstTip.hidden, true);
    assert.equal(saved.has(FIRST_OPEN_TIP_SEEN_KEY), false);
    filters.camera_tip = true;
    later.update({ worldWidth: W, worldHeight: H, reason: "resize" });
    assert.equal(firstTip.hidden, false, "On shows a tip not seen yet");
    assert.equal(saved.get(FIRST_OPEN_TIP_SEEN_KEY), "1");
  } finally {
    later.destroy();
  }
});

test("the Replay Viewer's switch has no buttons, and V and B still change the camera", () => {
  const doc = fakeDocument();
  const slot = doc.createElement("div");
  const camera = createCameraSwitch({
    slot: /** @type {any} */ (slot),
    product: "replay_viewer",
    storage: null,
    timers: { setTimeout: () => ({}), clearTimeout: () => {} },
    applyFrame: () => {},
    registerKeys: false,
    switchCameras: [],
  });
  try {
    /** @param {any} node @returns {any[]} */
    const all = (node) => [node, ...node.children.flatMap(all)];
    assert.deepEqual(
      all(slot).filter(
        (node) =>
          node.className === "camera-switch__choice" ||
          String(node.className).includes("inset"),
      ),
      [],
    );
    assert.equal(camera.pressKey("v"), true);
    assert.equal(camera.camera(), "everyone_in_view");
    assert.equal(slot.dataset.camera, "everyone_in_view");
    assert.equal(camera.pressKey("b"), true);
    assert.equal(camera.camera(), "battle_cam");
    assert.equal(slot.dataset.camera, "battle_cam");
    assert.equal(camera.pressKey("b"), true);
    assert.equal(camera.camera(), "whole_map");
  } finally {
    camera.destroy();
  }
});

test("tipHost holds both Replay Viewer tips, and live play ignores it", () => {
  const doc = fakeDocument();
  /** @param {any} node @returns {any[]} */
  const all = (node) => [node, ...node.children.flatMap(all)];
  const has = (/** @type {any} */ root, /** @type {string} */ name) =>
    all(root).some((node) => node.className === name);
  const slot = doc.createElement("div");
  const host = doc.createElement("div");
  host.hidden = true;
  host.append(doc.createElement("span"));
  const camera = createCameraSwitch({
    slot: /** @type {any} */ (slot),
    tipHost: /** @type {any} */ (host),
    product: "replay_viewer",
    storage: /** @type {any} */ ({ getItem: () => null, setItem: () => {} }),
    timers: { setTimeout: () => ({}), clearTimeout: () => {} },
    applyFrame: () => {},
    registerKeys: false,
  });
  try {
    assert.equal(host.hidden, false);
    assert.deepEqual(
      host.children.map((/** @type {any} */ node) => node.className),
      ["camera-switch__tip", "camera-switch__first-tip"],
      "the host's old children are replaced by the two tips",
    );
    assert.equal(find(host, "camera-switch__first-tip").hidden, false);
    assert.equal(has(slot, "camera-switch__tip"), false);
    assert.equal(has(slot, "camera-switch__first-tip"), false);
    assert.equal(camera.pressKey("v"), true);
    assert.equal(
      find(host, "camera-switch__tip").hidden,
      false,
      "the change tip shows there",
    );
  } finally {
    camera.destroy();
  }
  const liveSlot = doc.createElement("div");
  const liveHost = doc.createElement("div");
  liveHost.hidden = true;
  const live = createCameraSwitch({
    slot: /** @type {any} */ (liveSlot),
    tipHost: /** @type {any} */ (liveHost),
    product: "combat_debugger",
    storage: null,
    timers: { setTimeout: () => ({}), clearTimeout: () => {} },
    applyFrame: () => {},
    registerKeys: false,
  });
  try {
    assert.equal(liveHost.hidden, true);
    assert.equal(liveHost.children.length, 0);
  } finally {
    live.destroy();
  }
});

/** @param {any} root @param {string} className @param {string} [camera] */
function find(root, className, camera) {
  /** @param {any} node @returns {any} */
  const walk = (node) => {
    if (
      node.className === className &&
      (camera === undefined || node.dataset.camera === camera)
    ) {
      return node;
    }
    for (const child of node.children) {
      const found = walk(child);
      if (found) return found;
    }
    return null;
  };
  const found = walk(root);
  assert.ok(found, `found ${className}`);
  return found;
}

function fakeDocument() {
  /** @type {Map<string, Array<(event: any) => void>>} */
  const windowListeners = new Map();
  /** @type {any} */
  const doc = {
    defaultView: {
      addEventListener: (/** @type {string} */ type, /** @type {any} */ listener) => {
        windowListeners.set(type, [...(windowListeners.get(type) ?? []), listener]);
      },
      removeEventListener: (
        /** @type {string} */ type,
        /** @type {any} */ listener,
      ) => {
        windowListeners.set(
          type,
          (windowListeners.get(type) ?? []).filter((entry) => entry !== listener),
        );
      },
    },
    /** @param {string} type @param {Record<string, unknown>} event */
    dispatch(type, event) {
      for (const listener of windowListeners.get(type) ?? [])
        listener({ type, ...event });
    },
    createElement: (/** @type {string} */ tag) => node(tag),
    createElementNS: (/** @type {string} */ _namespace, /** @type {string} */ tag) =>
      node(tag),
  };
  /** @param {string} tag */
  function node(tag) {
    /** @type {Map<string, string>} */
    const attributes = new Map();
    /** @type {Map<string, Array<(event: any) => void>>} */
    const listeners = new Map();
    /** @type {Set<string>} */
    const classes = new Set();
    /** @type {any} */
    const element = {
      ownerDocument: doc,
      tagName: tag,
      attributes,
      children: /** @type {any[]} */ ([]),
      dataset: /** @type {Record<string, string>} */ ({}),
      textContent: "",
      hidden: false,
      type: "",
      get className() {
        return attributes.get("class") ?? "";
      },
      set className(value) {
        attributes.set("class", String(value));
      },
      classList: {
        add: (/** @type {string} */ name) => {
          classes.add(name);
          attributes.set(
            "class",
            [
              ...new Set([
                ...(attributes.get("class") ?? "").split(" ").filter(Boolean),
                name,
              ]),
            ].join(" "),
          );
        },
        toggle: (/** @type {string} */ name, /** @type {boolean} */ on) => {
          if (on) classes.add(name);
          else classes.delete(name);
        },
      },
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) => {
        attributes.set(name, String(value));
      },
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      removeAttribute: (/** @type {string} */ name) => {
        attributes.delete(name);
      },
      addEventListener: (/** @type {string} */ type, /** @type {any} */ listener) => {
        listeners.set(type, [...(listeners.get(type) ?? []), listener]);
      },
      append: (/** @type {any[]} */ ...nodes) => {
        element.children.push(...nodes);
      },
      replaceChildren: (/** @type {any[]} */ ...nodes) => {
        element.children = [...nodes];
      },
      contains: (/** @type {any} */ other) =>
        other === element ||
        element.children.some((/** @type {any} */ child) => child.contains(other)),
      click() {
        for (const listener of listeners.get("click") ?? [])
          listener({ type: "click", target: element });
      },
    };
    return element;
  }
  return doc;
}
