/**
 * @file The Camera look: Whole Map, Everyone in View (key V) and Battle Cam
 * (key B, Replay Viewer only), with the camera switch and its tips. (The
 * Camera Inset picture was removed, owner 4 Oct 2026.)
 *
 * What it owns:
 * - The camera rules as pure functions of one tick (no memory of earlier
 *   ticks): everyoneInViewFrame, battleCamFrame, killCamLean and
 *   killSlowMotion, and records(input), which turns them into drawing records.
 *   Every frame is a world rectangle in world units with y counting up.
 * - The words: camera names, keycaps and the tip texts, quoted exactly as the
 *   plan (section 4.4) and the owner's row 12 give them.
 * - The camera switch (createCameraSwitch): a self-contained component the
 *   shell (Replay Viewer, slot #replay-camera-switch) and the BattleClient
 *   screen (slot #live-camera-switch) mount. In the Replay Viewer it shows no
 *   buttons since 4 Oct 2026 (the owner's key stack: the arena's V and B keys,
 *   src/arena-buttons.js, are the camera's buttons, and no lit key means
 *   Whole Map); it owns the change tip, the one-time first-open tip (both
 *   shown in the scorebug's empty score half, #camera-tips, so they cover
 *   nothing) and the Battle Cam words (for screen readers only), and
 *   remembers the last camera in localStorage. In the BattleClient and the DevClient combat
 *   screen no switch shows (owner, 1 October 2026): V alone toggles Everyone
 *   in View, and the change tip (#camera-tip, "Everyone in View on · V: back
 *   to Whole Map" or "Whole Map on · V: Everyone in View") shows under the
 *   arena, over the deck header line's message, and goes after about 4 s or at
 *   the next input. Either way it registers the V and B handlers with
 *   `command-set.js`.
 * - The live view tag's spot (viewTagPlace): the arena's view tag (#pov-tag,
 *   painted by `shell.js`) sits on the arena's frame under the floor, just
 *   past the left home strip, as the store's Agent POV caption does, so it
 *   never covers the left home end's brackets or the herald cards' top band.
 *   The spot follows the Whole Map framing, so the live switch writes it on
 *   #arena after each render and overlays.css draws the tag there.
 *
 * Who calls it: `main.js` mounts the switch and calls its update() after each
 * render with the tick's plan (`choreography-plan.js` plan.camera and
 * plan.beats). The switch hands each new frame to the page through the
 * applyFrame callback, which calls the page's one entry point setCameraFrame
 * and then the choreographer's easeCamera. `visual-filters.js` reads
 * PAINT_PARTS (none: the camera draws nothing on the arena).
 *
 * Boundary (Agent POV): every frame, lean and slow-down comes only
 * from the audience's own plan camera record (its bodies and fights), its own
 * beat schedule and, optionally, its own arena scene. In Agent POV those hold
 * only champions and fights the team saw, so the camera never leans, zooms or
 * slows toward anything hidden. Nothing here reads a researcher view
 * (researcher_space) or a match summary. The tips and button words name only
 * cameras and keys, never a champion, so they are the same in both views. The
 * view tag's spot reads only the map's Red Zone record and the projection,
 * public facts that are the same in both views.
 *
 * Timing: the camera ease (520 ms) and any Kill Cam lean run as Web
 * Animations on the camera wrapper through the choreographer, with fill
 * "backwards", so the DOM always holds the settled frame (END). The only
 * timers here hide tips (about 4 s and 6 s); tips are not beat-timed.
 */

import { registerLocalCommandHandler } from "../command-set.js";
import {
  EVERYONE_IN_VIEW_MARGIN,
  EVERYONE_IN_VIEW_MAX_ZOOM,
  EVERYONE_IN_VIEW_ZOOM_STEP,
} from "../layout.js";
import { createSemanticDescriptor, registerTooltipOwner } from "../tooltip.js";
import { EMPTY_BUDGET, isRecord } from "./look-support.js";

/**
 * @typedef {import("./look-support.js").LookInput} LookInput
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {"whole_map" | "everyone_in_view" | "battle_cam"} CameraName
 * @typedef {"combat_debugger" | "replay_viewer"} CameraProduct
 *   "combat_debugger" is the BattleClient and the DevClient combat screen;
 *   "replay_viewer" is the Replay Viewer.
 * @typedef {Readonly<{minX: number, minY: number, maxX: number, maxY: number}>} WorldFrame
 *   A world rectangle in world units, y counting up. It may reach up to
 *   MAP_EDGE_ALLOWANCE past the map edge.
 * @typedef {readonly [number, number]} WorldPoint
 * @typedef {Readonly<{
 *   presentationKey?: string,
 *   publicAgentId?: string,
 *   start: WorldPoint | null,
 *   end: WorldPoint,
 *   state: "living" | "falling" | "returning",
 * }>} CameraBody
 *   One body of plan.camera.bodies: its START spot (null when not disclosed or
 *   for a return), its END spot and its state this tick.
 * @typedef {Readonly<{eventId?: string, kind: "blow" | "fall", points: ReadonlyArray<WorldPoint>}>} CameraFight
 *   One fight of plan.camera.fights: a Basic or Ultimate ("blow", source and
 *   target spots) or a seen fall ("fall", the victim's END spot).
 * @typedef {Readonly<{
 *   audience?: "researcher" | "agent_pov" | null,
 *   bodies: ReadonlyArray<CameraBody>,
 *   fights: ReadonlyArray<CameraFight>,
 * }>} PlanCamera
 *   The audience's own camera input from `choreography-plan.js` (planCamera).
 * @typedef {Readonly<{
 *   windows: Readonly<{impact: readonly [number, number]} & Record<string, any>>,
 *   fall: readonly [number, number] | null,
 *   returnBeat?: readonly [number, number] | null,
 *   tickMs: number,
 * } & Record<string, any>>} CameraBeats
 *   The fields of the plan's BeatSchedule that the camera reads, in ms from the
 *   start of the tick.
 * @typedef {{x: number, y: number, w: number, h: number}} Box
 *   Internal frame form: left x, bottom y, width and height in world units.
 * @typedef {{worldToScreen: (point: readonly [number, number]) => {x: number, y: number}}} ScreenProjection
 *   Any viewport transform from `layout.js` (createViewportTransform).
 */

/** The three cameras, in switch order. */
export const CAMERAS = Object.freeze(
  /** @type {const} */ (["whole_map", "everyone_in_view", "battle_cam"]),
);

/** The name each camera shows on its button and in tips. */
export const CAMERA_NAMES = Object.freeze({
  whole_map: "Whole Map",
  everyone_in_view: "Everyone in View",
  battle_cam: "Battle Cam",
});

/** The keycap printed on each camera button (Whole Map has no key). */
export const CAMERA_KEYCAPS = Object.freeze({
  whole_map: null,
  everyone_in_view: "V",
  battle_cam: "B",
});

/** The Replay Viewer's one-time first-open tip (plan 4.4, quoted exactly). */
export const FIRST_OPEN_TIP =
  "New: Battle Cam follows the fights for you. Click here or press B.";

/** How long the first-open tip stays before it hides itself, in ms. */
export const FIRST_OPEN_TIP_MS = 6000;

/** How long a camera change tip stays before it hides itself, in ms. */
export const CHANGE_TIP_MS = 4000;

/**
 * localStorage key of the Replay Viewer's last camera (this browser only). The
 * stored value is a camera name; older or spelled-out names are mapped by
 * readRememberedCamera.
 */
export const REMEMBERED_CAMERA_KEY = "marl-bgs-replay-camera";

/** localStorage key that records that the first-open tip was shown once. */
export const FIRST_OPEN_TIP_SEEN_KEY = "marl-bgs-camera-tip-seen";

/**
 * A body's drawn reach in world units (the 0.5 circle and its team ring). A
 * frame keeps every drawn body wholly inside or wholly outside it.
 */
export const BODY_REACH = 0.56;

/** How far (world units) a body kept inside a frame stays from its edge. */
export const EDGE_INSET = 0.06;

/** How far (world units) a camera frame may reach past the map edge. */
export const MAP_EDGE_ALLOWANCE = 0.45;

/**
 * Room (world units) kept around a returning champion's pad, so its return
 * ring and Spawn Shield chip stay in frame.
 */
export const RETURN_ROOM = 0.9;

/**
 * Battle Cam rules (store "battle-director", unchanged): the engaged spots plus
 * every living body within nearReach world units, a margin in world units, and
 * two levels only: the whole map or zoom.
 */
export const BATTLE_CAM = Object.freeze({ margin: 2, zoom: 1.5, nearReach: 3.5 });

/**
 * Kill Cam rules: a lean of at least minLead times the resting zoom, up to
 * maxZoom, fitted with margin world units around the fall; living bodies
 * within nearFall world units of a fall stay in; it zooms less in zoomStep
 * steps down to minLean times the resting zoom, else no lean. While the fall
 * plays the frame creeps to creep times its zoom. It is back on the resting
 * frame backRest ms before the tick ends, and each move lasts at least
 * minMove ms.
 */
export const KILL_CAM = Object.freeze({
  margin: 1.5,
  maxZoom: 1.5,
  minLead: 1.1,
  minLean: 1.05,
  zoomStep: 0.05,
  nearFall: 2,
  creep: 1.03,
  backRest: 180,
  minMove: 120,
});

/**
 * Kill Slow Motion rules: the Impact and Fall beats of a fall tick play at
 * factor times the chosen speed, never below minRate; skipped when the chosen
 * speed is at or below minRate or at or above fastRate.
 */
export const KILL_SLOW_MOTION = Object.freeze({
  factor: 0.5,
  minRate: 0.25,
  fastRate: 1.5,
});

/** The camera draws nothing on the arena, so no allocator footprint. */
export const FOOTPRINT = Object.freeze({});

/** No arena nodes and no event animations. */
export const BUDGET = EMPTY_BUDGET;

/**
 * No arena paint parts: the camera switches (camera_framing and the rest) act on
 * the switch and the frame, not on arena drawings.
 *
 * @type {ReadonlyArray<import("./look-support.js").PaintPartRegistration>}
 */
export const PAINT_PARTS = Object.freeze([]);

const EPSILON = 1e-7;

/**
 * Old, spelled-out and earlier store names of a remembered camera, keyed by the
 * name in lower case with everything but letters removed.
 */
const CAMERA_ALIASES = Object.freeze({
  whole: "whole_map",
  wholemap: "whole_map",
  map: "whole_map",
  everyone: "everyone_in_view",
  everyoneinview: "everyone_in_view",
  director: "battle_cam",
  battledirector: "battle_cam",
  battlecam: "battle_cam",
  battlecamera: "battle_cam",
  cam: "battle_cam",
});

/**
 * Return the cameras a product offers, in switch order.
 *
 * product "replay_viewer" offers all three; "combat_debugger" (the BattleClient
 * and DevClient combat screen) offers Whole Map and Everyone in View only (B
 * does nothing there). Any other value is treated as "combat_debugger". Pure.
 *
 * @param {unknown} product
 * @returns {ReadonlyArray<CameraName>}
 */
export function camerasFor(product) {
  return product === "replay_viewer" ? CAMERAS : CAMERAS.slice(0, 2);
}

/**
 * Return the camera a product really shows for a chosen camera and switches.
 *
 * camera is the chosen camera; a camera the product does not offer, an unknown
 * value, or camera_framing switched off (filters.camera_framing === false) all
 * give "whole_map". filters may be null or omitted (every switch at its
 * default, camera_framing on). Pure.
 *
 * @param {unknown} camera
 * @param {unknown} product
 * @param {Readonly<Record<string, boolean>> | null} [filters]
 * @returns {CameraName}
 */
export function effectiveCamera(camera, product, filters = null) {
  if (filters?.camera_framing === false) return "whole_map";
  const offered = camerasFor(product);
  return offered.includes(/** @type {CameraName} */ (camera))
    ? /** @type {CameraName} */ (camera)
    : "whole_map";
}

/**
 * Return the camera a key press chooses.
 *
 * key "v" or "V" toggles Everyone in View; "b" or "B" toggles Battle Cam (in
 * the Replay Viewer only). The lit key returns to Whole Map; the other key
 * switches straight across. Any other key, or B outside the Replay Viewer,
 * returns current unchanged. Pure.
 *
 * @param {CameraName} current The camera on now.
 * @param {string} key The key pressed (KeyboardEvent.key).
 * @param {unknown} product
 * @returns {CameraName}
 */
export function nextCamera(current, key, product) {
  const lower = String(key).toLowerCase();
  /** @type {CameraName | null} */
  const target =
    lower === "v" ? "everyone_in_view" : lower === "b" ? "battle_cam" : null;
  if (target === null || !camerasFor(product).includes(target)) return current;
  return current === target ? "whole_map" : target;
}

/**
 * Return the change tip for a camera, quoted exactly (plan 4.4, row 12).
 *
 * In the Replay Viewer: "Battle Cam on · B: back to Whole Map · V: Everyone in
 * View", "Everyone in View on · V: back to Whole Map · B: Battle Cam" and, for
 * Whole Map, "Whole Map on · V: Everyone in View · B: Battle Cam". In the
 * BattleClient and DevClient (any other product): "Everyone in View on · V:
 * back to Whole Map" and "Whole Map on · V: Everyone in View". The words name
 * cameras and keys only, so they are the same in Oracle View and Agent POV.
 * Pure.
 *
 * @param {CameraName} camera
 * @param {unknown} product
 * @returns {string}
 */
export function cameraChangeTip(camera, product) {
  if (product === "replay_viewer") {
    if (camera === "battle_cam") {
      return "Battle Cam on · B: back to Whole Map · V: Everyone in View";
    }
    if (camera === "everyone_in_view") {
      return "Everyone in View on · V: back to Whole Map · B: Battle Cam";
    }
    return "Whole Map on · V: Everyone in View · B: Battle Cam";
  }
  return camera === "everyone_in_view"
    ? "Everyone in View on · V: back to Whole Map"
    : "Whole Map on · V: Everyone in View";
}

/**
 * Read the Replay Viewer's remembered camera from storage.
 *
 * storage is a Storage-like object (window.localStorage) or null. A stored
 * camera name is returned as is; an older or spelled-out name ("director",
 * "Battle Director", "battle-cam", "Everyone in View", "whole_map" ...) maps to
 * its camera, so a viewer who left the old Battle Director lands on Battle Cam.
 * No storage, an empty or unknown value, or a storage error give "whole_map".
 * Never throws; reads storage only.
 *
 * @param {Pick<Storage, "getItem"> | null | undefined} storage
 * @returns {CameraName}
 */
export function readRememberedCamera(storage) {
  let raw = null;
  try {
    raw = storage ? storage.getItem(REMEMBERED_CAMERA_KEY) : null;
  } catch {
    return "whole_map";
  }
  if (typeof raw !== "string") return "whole_map";
  if (CAMERAS.includes(/** @type {CameraName} */ (raw))) {
    return /** @type {CameraName} */ (raw);
  }
  const key = raw.toLowerCase().replace(/[^a-z]/g, "");
  return Object.hasOwn(CAMERA_ALIASES, key)
    ? CAMERA_ALIASES[/** @type {keyof typeof CAMERA_ALIASES} */ (key)]
    : "whole_map";
}

/**
 * Remember the Replay Viewer's camera in storage.
 *
 * Writes camera under REMEMBERED_CAMERA_KEY. Return true when written, false
 * when there is no storage or writing failed (private mode, blocked storage,
 * quota). Never throws.
 *
 * @param {Pick<Storage, "setItem"> | null | undefined} storage
 * @param {CameraName} camera
 * @returns {boolean}
 */
export function rememberCamera(storage, camera) {
  try {
    if (!storage) return false;
    storage.setItem(REMEMBERED_CAMERA_KEY, camera);
    return true;
  } catch {
    return false;
  }
}

/**
 * Return whether the first-open tip still has to be shown in this browser.
 *
 * True when storage holds no FIRST_OPEN_TIP_SEEN_KEY. With no storage, or when
 * reading fails, return false: without storage the tip could never be
 * remembered, so it is not shown on every open. Never throws.
 *
 * @param {Pick<Storage, "getItem"> | null | undefined} storage
 * @returns {boolean}
 */
export function firstOpenTipDue(storage) {
  try {
    return Boolean(storage) && storage?.getItem(FIRST_OPEN_TIP_SEEN_KEY) === null;
  } catch {
    return false;
  }
}

/**
 * Return the world x where the left home strip (the left team's Red Zone)
 * ends, or 0 (the floor's left edge) when the map has no usable Red Zone.
 *
 * redZone is the map record's `red_zone` (team_a_x_range and team_b_x_range,
 * each [low, high] in world units); worldWidth is the map width. The left
 * strip is the range whose middle lies left of the map's middle; when neither
 * or both do, 0 comes back. Pure.
 *
 * @param {unknown} redZone
 * @param {number} worldWidth
 * @returns {number}
 */
function leftHomeEndX(redZone, worldWidth) {
  if (!isRecord(redZone)) return 0;
  /** @type {number[]} */
  const ends = [];
  for (const range of [redZone.team_a_x_range, redZone.team_b_x_range]) {
    if (!Array.isArray(range) || range.length !== 2) continue;
    const [low, high] = range.map(Number);
    if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) continue;
    if ((low + high) / 2 < worldWidth / 2) ends.push(high);
  }
  return ends.length === 1 ? ends[0] : 0;
}

/**
 * Return the live view tag's spot (store Agent POV caption rule), or null.
 *
 * The arena's view tag (#pov-tag, "Agent POV · …" or "Oracle View · …") sits
 * on the arena's frame under the floor, starting just past the left home strip
 * and its corner bracket (the owner's store rule of 30 September 2026), so it
 * never covers the left home end's bracket lines, a floor mark or the herald
 * cards' top band (owner ruling M3, 3 October 2026). This returns the two
 * lengths that spot needs, as shares of the field's width (the battlefield
 * SVG's viewport width), so a resize keeps them true:
 * - homeEnd: where the left home strip ends (leftHomeEndX of the scene's map
 *   `red_zone`, or the floor's left edge without one), measured from the
 *   field's left edge;
 * - band: the height of the frame under the floor (viewport bottom less the
 *   floor's bottom edge).
 *
 * input.transform is this render's viewport transform (`layout.js`
 * createViewportTransform); input.scene is the audience's own arena scene
 * (only scene.map.red_zone is read: public map facts, the same in both
 * views). It returns null unless the transform shows the Whole Map (no camera
 * zoom), because a zoomed frame moves the strip; the caller then keeps the last
 * Whole Map spot (the map does not change during a game). Pure.
 *
 * @param {Readonly<{transform?: any, scene?: Record<string, any> | null}>} input
 * @returns {Readonly<{homeEnd: number, band: number}> | null}
 */
export function viewTagPlace(input) {
  const transform = input?.transform;
  const view = transform?.viewportBounds;
  const floor = transform?.mapBounds;
  if (!view || !floor || typeof transform.worldToScreen !== "function") return null;
  if (!(view.width > 0)) return null;
  const frame = transform.cameraFrame;
  if (
    frame &&
    (frame.minX > EPSILON ||
      frame.minY > EPSILON ||
      frame.maxX < transform.worldWidth - EPSILON ||
      frame.maxY < transform.worldHeight - EPSILON)
  ) {
    return null;
  }
  const end = leftHomeEndX(input?.scene?.map?.red_zone, transform.worldWidth);
  const x = transform.worldToScreen([end, 0]).x;
  return Object.freeze({
    homeEnd: (x - view.left) / view.width,
    band: Math.max(0, view.bottom - floor.bottom) / view.width,
  });
}

/**
 * Return whether a camera change eases or cuts.
 *
 * reason is why the frame changed: "camera" (the viewer picked a camera),
 * "play" (a new tick in play: Everyone in View and Battle Cam move with the
 * glide), "step" (one tick forward or back) ease; "seek", "view" (Oracle View
 * and Agent POV switch), "recipient" (a new POV champion), "resize" and any
 * other reason cut, so a POV change never eases. reducedMotion true always
 * cuts. The ease itself is the choreographer's (520 ms). Pure.
 *
 * @param {Readonly<{reason?: string, reducedMotion?: boolean}>} input
 * @returns {"ease" | "cut"}
 */
export function cameraEase(input) {
  if (input.reducedMotion === true) return "cut";
  return input.reason === "camera" || input.reason === "play" || input.reason === "step"
    ? "ease"
    : "cut";
}

/**
 * Choose the Everyone in View frame for one tick.
 *
 * input.camera is the audience's plan camera record; input.worldWidth and
 * input.worldHeight are the map size in world units; input.scene (optional) is
 * the audience's own arena scene, whose agents' positions are added to the
 * clean-edge check (earlier corpses are drawn too). The frame keeps every
 * living and falling body's START and END spots and each returning body's pad
 * with RETURN_ROOM around it, with a 1.5-unit margin (EVERYONE_IN_VIEW_MARGIN).
 * Zoom (against the Whole Map, keeping the map's aspect) is the highest step
 * of 0.25 up to 2x that fits; the frame sits as near the map centre as the
 * group allows (so it stays put while the group moves inside it), then slides
 * so no drawn body is cut by an edge; when no slide works it zooms out one
 * step, down to the Whole Map.
 *
 * When the plan camera has no bodies (no tick has played yet, such as the
 * first frame of a game or a restored start), the scene's living champions
 * stand in, each at its scene position (sceneCameraBodies), so V frames
 * everyone on the very first turn too.
 *
 * Return frozen {frame, zoom}: frame is a WorldFrame and zoom 1 means the Whole
 * Map frame {0, 0, worldWidth, worldHeight}. With no bodies at all, the Whole
 * Map. Agent POV: pass only the POV plan camera and POV scene; nothing else is
 * read, so only champions the team sees can be framed.
 * Throw TypeError when the world size is not positive and finite. Pure.
 *
 * @param {Readonly<{camera?: PlanCamera | null, worldWidth: number, worldHeight: number, scene?: Record<string, any> | null}>} input
 * @returns {Readonly<{frame: WorldFrame, zoom: number}>}
 */
export function everyoneInViewFrame(input) {
  const world = worldOf(input);
  const planned = cameraBodies(input.camera);
  const bodies = planned.length > 0 ? planned : sceneCameraBodies(input.scene);
  /** @type {WorldPoint[]} */
  const must = [];
  const drawn = drawnSpots(bodies, input.scene);
  for (const body of bodies) {
    if (body.state === "returning") {
      must.push(...returnRoom(body.end));
      continue;
    }
    must.push(body.end);
    if (body.start !== null) must.push(body.start);
  }
  if (must.length === 0) return wholeResult(world);
  const margin = EVERYONE_IN_VIEW_MARGIN;
  const step = EVERYONE_IN_VIEW_ZOOM_STEP;
  const need = zoomNeeded(world, must, margin);
  const span = extent(must);
  const top = Math.min(
    EVERYONE_IN_VIEW_MAX_ZOOM,
    Math.floor(need.zoom / step + 1e-9) * step,
  );
  for (let zoom = top; zoom > 1 + 1e-6; zoom -= step) {
    const width = world.width / zoom;
    const height = world.height / zoom;
    const centreX = calmCentre(span.minX, span.maxX, margin, width, world.width);
    const centreY = calmCentre(span.minY, span.maxY, margin, height, world.height);
    const clean = cleanBox(world, boxAt(world, centreX, centreY, zoom), must, drawn);
    if (clean !== null) return Object.freeze({ frame: frameOf(clean), zoom });
  }
  return wholeResult(world);
}

/**
 * Choose the Battle Cam frame for one tick (Auto-Frame Fights).
 *
 * input.camera, input.worldWidth, input.worldHeight and input.scene are as for
 * everyoneInViewFrame; input.filters is the visual filter state (null for
 * defaults). With auto_frame_fights off, the Whole Map. Otherwise the frame
 * holds every fight spot of the tick plus every living body whose START or END
 * spot is within 3.5 units of one (no fight: every living body), with a
 * 2-unit margin, at one of two levels: the Whole Map or 1.5x. A 1.5x frame is
 * slid until no drawn body is cut and each returning pad is wholly inside;
 * when that fails, the Whole Map. Return frozen {frame, zoom}.
 *
 * Agent POV: the fights are only those the team saw (plan.camera.fights comes
 * from visual_events), so Battle Cam never leans toward a hidden fight. Throw
 * TypeError for a bad world size. Pure.
 *
 * @param {Readonly<{camera?: PlanCamera | null, worldWidth: number, worldHeight: number, scene?: Record<string, any> | null, filters?: Readonly<Record<string, boolean>> | null}>} input
 * @returns {Readonly<{frame: WorldFrame, zoom: number}>}
 */
export function battleCamFrame(input) {
  const world = worldOf(input);
  if (input.filters?.auto_frame_fights === false) return wholeResult(world);
  const bodies = cameraBodies(input.camera);
  const engaged = cameraFights(input.camera).flatMap((fight) => fight.points);
  /** @type {WorldPoint[]} */
  const points = [...engaged];
  for (const body of bodies) {
    if (body.state !== "living") continue;
    const near = (/** @type {WorldPoint | null} */ spot) =>
      spot !== null &&
      engaged.some((point) => distance(spot, point) <= BATTLE_CAM.nearReach);
    if (near(body.end)) points.push(body.end);
    if (body.start !== null && near(body.start)) points.push(body.start);
  }
  if (points.length === 0) {
    for (const body of bodies) if (body.state === "living") points.push(body.end);
  }
  if (points.length === 0) return wholeResult(world);
  const need = zoomNeeded(world, points, BATTLE_CAM.margin);
  if (need.zoom < BATTLE_CAM.zoom) return wholeResult(world);
  const must = [...points];
  for (const body of bodies) {
    if (body.state === "returning") must.push(...returnRoom(body.end));
  }
  const clean = cleanBox(
    world,
    boxAt(world, need.centre[0], need.centre[1], BATTLE_CAM.zoom),
    must,
    drawnSpots(bodies, input.scene),
  );
  return clean === null
    ? wholeResult(world)
    : Object.freeze({ frame: frameOf(clean), zoom: BATTLE_CAM.zoom });
}

/**
 * Plan the Kill Cam lean for one tick of play, or return null.
 *
 * input.camera, worldWidth, worldHeight and scene are as for
 * everyoneInViewFrame; input.restFrame is the resting frame on screen (the
 * Battle Cam frame); input.beats is the tick's beat schedule. A lean exists
 * only when the tick has a fall the audience saw (a "fall" fight) and a fall
 * beat. It holds, wholly inside: where each falling body started and fell,
 * every spot of a blow that touched a falling body, every living body within 2
 * units of a fall and each returning pad with its room. Zoom is from 1.1x the
 * resting zoom (or what the fall needs, up to 1.5x) down to 1.05x in 0.05
 * steps, fitted with a 1.5-unit margin and slid clean; none clean gives null.
 *
 * Return frozen {frame, creepFrame, zoom, windows}: frame is the lean,
 * creepFrame the 3% closer frame reached by the end of the fall beat, zoom the
 * lean's zoom against the Whole Map, and windows {push, hold, back}, each
 * [startMs, endMs] from the tick start: push from the Impact beat to the fall
 * beat, hold through the fall beat (and the return beat when later), back to
 * the resting frame by 180 ms before the tick ends. Agent POV: only seen falls
 * and seen bodies are read. The caller decides whether a lean applies (Battle
 * Cam, kill_cam on, in play, motion on: see records). Pure.
 *
 * @param {Readonly<{camera?: PlanCamera | null, worldWidth: number, worldHeight: number, scene?: Record<string, any> | null, restFrame: WorldFrame, beats?: CameraBeats | null}>} input
 * @returns {Readonly<{frame: WorldFrame, creepFrame: WorldFrame, zoom: number, windows: Readonly<{push: readonly [number, number], hold: readonly [number, number], back: readonly [number, number]}>}> | null}
 */
export function killCamLean(input) {
  const world = worldOf(input);
  const beats = input.beats;
  if (!isRecord(beats) || !Array.isArray(beats.fall)) return null;
  const fights = cameraFights(input.camera);
  const fallSpots = fights
    .filter((fight) => fight.kind === "fall")
    .flatMap((fight) => fight.points);
  if (fallSpots.length === 0) return null;
  const bodies = cameraBodies(input.camera);
  const victims = bodies.filter((body) => body.state === "falling");
  /** @type {WorldPoint[]} */
  const victimSpots = victims.flatMap((body) =>
    body.start === null ? [body.end] : [body.start, body.end],
  );
  /** @type {WorldPoint[]} */
  const must = [...fallSpots, ...victimSpots];
  const touched = [...victimSpots, ...fallSpots];
  for (const fight of fights) {
    if (fight.kind !== "blow") continue;
    if (
      fight.points.some((point) =>
        touched.some((spot) => distance(point, spot) <= BODY_REACH),
      )
    ) {
      must.push(...fight.points);
    }
  }
  for (const body of bodies) {
    if (
      body.state === "living" &&
      fallSpots.some((spot) => distance(body.end, spot) <= KILL_CAM.nearFall)
    ) {
      must.push(body.end);
    }
    if (body.state === "returning") must.push(...returnRoom(body.end));
  }
  const rest = boxOf(input.restFrame);
  const restZoom = world.width / rest.w;
  const need = zoomNeeded(world, must, KILL_CAM.margin);
  const top = Math.min(
    Math.max(restZoom * KILL_CAM.minLead, Math.min(need.zoom, KILL_CAM.maxZoom)),
    need.zoom,
  );
  const drawn = drawnSpots(bodies, input.scene);
  for (
    let zoom = top;
    zoom >= restZoom * KILL_CAM.minLean - EPSILON;
    zoom -= KILL_CAM.zoomStep
  ) {
    const lean = cleanBox(
      world,
      boxAt(world, need.centre[0], need.centre[1], zoom),
      must,
      drawn,
    );
    if (lean === null) continue;
    const centreX = lean.x + lean.w / 2;
    const centreY = lean.y + lean.h / 2;
    const creep =
      cleanBox(
        world,
        boxAt(world, centreX, centreY, zoom * KILL_CAM.creep),
        must,
        drawn,
      ) ?? lean;
    return Object.freeze({
      frame: frameOf(lean),
      creepFrame: frameOf(creep),
      zoom,
      windows: leanWindows(beats),
    });
  }
  return null;
}

/**
 * Plan Kill Slow Motion for one tick, or return null.
 *
 * Applies only in the Replay Viewer (input.product "replay_viewer") with
 * Battle Cam on (input.camera_name "battle_cam"), kill_slow_motion on, in play
 * (input.playing true; never on a step, seek or still), without reduced
 * motion, on a tick with a fall the audience saw and a fall beat, at a chosen
 * speed input.chosenRate strictly between 0.25x and 1.5x. Then the Impact beat
 * and the fall beat play at half the chosen speed, never below 0.25x.
 *
 * Return frozen {rate, windows, label}: rate is the slowed playback rate,
 * windows the [startMs, endMs] spans (tick time) to slow, label the words
 * beside the switch, for example "Slow Motion · 0.5×". Agent POV: a hidden fall
 * is not in the plan camera, so it never slows the tick. Pure.
 *
 * @param {Readonly<{camera?: PlanCamera | null, beats?: CameraBeats | null, camera_name?: CameraName, product?: unknown, filters?: Readonly<Record<string, boolean>> | null, playing?: boolean, reducedMotion?: boolean, chosenRate?: number}>} input
 * @returns {Readonly<{rate: number, windows: ReadonlyArray<readonly [number, number]>, label: string}> | null}
 */
export function killSlowMotion(input) {
  const rate = input.chosenRate;
  if (
    input.product !== "replay_viewer" ||
    input.camera_name !== "battle_cam" ||
    input.filters?.kill_slow_motion === false ||
    input.filters?.camera_framing === false ||
    input.playing !== true ||
    input.reducedMotion === true ||
    typeof rate !== "number" ||
    !Number.isFinite(rate) ||
    rate <= KILL_SLOW_MOTION.minRate ||
    rate >= KILL_SLOW_MOTION.fastRate
  ) {
    return null;
  }
  const beats = input.beats;
  if (!isRecord(beats) || !Array.isArray(beats.fall)) return null;
  if (!cameraFights(input.camera).some((fight) => fight.kind === "fall")) return null;
  const slowed = Math.max(KILL_SLOW_MOTION.minRate, rate * KILL_SLOW_MOTION.factor);
  /** @type {Array<readonly [number, number]>} */
  const windows = [];
  if (Array.isArray(beats.windows?.impact)) {
    windows.push(Object.freeze([beats.windows.impact[0], beats.windows.impact[1]]));
  }
  windows.push(Object.freeze([beats.fall[0], beats.fall[1]]));
  return Object.freeze({
    rate: slowed,
    windows: Object.freeze(windows),
    label: `Slow Motion · ${formatRate(slowed)}×`,
  });
}

/**
 * Return the camera frame to hand to the page's setCameraFrame for one tick.
 *
 * input fields are those of records (camera_name, product, filters, camera,
 * worldWidth, worldHeight, scene). Return null for Whole Map (also when
 * camera_framing is off, or the product does not offer the camera, or the
 * frame works out to the whole map), else frozen {camera, minX, minY, maxX,
 * maxY}. Agent POV: reads only the POV plan camera and POV scene. Pure.
 *
 * @param {LookInput} input
 * @returns {Readonly<{camera: CameraName, minX: number, minY: number, maxX: number, maxY: number}> | null}
 */
export function cameraFrameFor(input) {
  const frame = /** @type {LookRecord | undefined} */ (
    records(input).find((record) => record.kind === "camera-frame")
  );
  if (!frame || frame.frame === null) return null;
  return Object.freeze({ camera: frame.camera, ...frame.frame });
}

/**
 * Return this tick's camera records (pure data; no DOM).
 *
 * input fields read:
 * - camera_name: the chosen camera (CameraName, default "whole_map");
 * - product: "replay_viewer" or "combat_debugger" (default);
 * - filters: the visual filter state (null for defaults);
 * - camera: the audience's plan camera record (plan.camera);
 * - beats: the tick's beat schedule (plan.beats), for Kill Cam and Kill Slow
 *   Motion;
 * - worldWidth and worldHeight, or transform.worldWidth and
 *   transform.worldHeight: the map size in world units;
 * - scene (optional): the audience's own arena scene (agents' position,
 *   public_agent_id and team_id are read for the clean-edge check);
 * - playing, reducedMotion, chosenRate: Battle Cam's play-only moments.
 *
 * Return a frozen list, in this order:
 * - always one {kind: "camera-frame", camera, frame, zoom}: camera is the
 *   effective camera, frame a WorldFrame or null for Whole Map, zoom 1 for
 *   Whole Map;
 * - {kind: "kill-cam", frame, creepFrame, zoom, windows} with Battle Cam,
 *   kill_cam on, in play and motion on, on a seen fall tick;
 * - {kind: "slow-motion", rate, windows, label} (see killSlowMotion).
 * With no world size, return an empty list. Agent POV: only the POV plan
 * camera, beats and scene are read; never a researcher view or match summary.
 *
 * @param {LookInput} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const width = input.worldWidth ?? input.transform?.worldWidth;
  const height = input.worldHeight ?? input.transform?.worldHeight;
  if (!positive(width) || !positive(height)) return Object.freeze([]);
  const filters = input.filters ?? null;
  const camera = effectiveCamera(
    input.camera_name ?? "whole_map",
    input.product,
    filters,
  );
  const base = {
    camera: input.camera ?? null,
    worldWidth: /** @type {number} */ (width),
    worldHeight: /** @type {number} */ (height),
    scene: input.scene ?? null,
    filters,
  };
  const chosen =
    camera === "everyone_in_view"
      ? everyoneInViewFrame(base)
      : camera === "battle_cam"
        ? battleCamFrame(base)
        : wholeResult(worldOf(base));
  /** @type {LookRecord[]} */
  const out = [
    Object.freeze({
      kind: "camera-frame",
      camera,
      frame: chosen.zoom === 1 ? null : chosen.frame,
      zoom: chosen.zoom,
    }),
  ];
  if (camera === "battle_cam") {
    const playMoment =
      input.playing === true &&
      input.reducedMotion !== true &&
      filters?.kill_cam !== false;
    const lean = playMoment
      ? killCamLean({ ...base, restFrame: chosen.frame, beats: input.beats ?? null })
      : null;
    if (lean !== null) out.push(Object.freeze({ kind: "kill-cam", ...lean }));
    const slow = killSlowMotion({
      camera: base.camera,
      beats: input.beats ?? null,
      camera_name: camera,
      product: input.product,
      filters,
      playing: input.playing === true,
      reducedMotion: input.reducedMotion === true,
      chosenRate: input.chosenRate,
    });
    if (slow !== null) out.push(Object.freeze({ kind: "slow-motion", ...slow }));
  }
  return Object.freeze(out);
}

/**
 * Build arena nodes for camera records: none. The camera draws nothing on the
 * arena (its picture inset was removed, owner 4 Oct 2026); this keeps the
 * shared look contract (records, build, update, tooltip). Returns a new empty
 * array whatever it is given.
 *
 * @param {ReadonlyArray<LookRecord>} _records
 * @param {Document} _doc
 * @returns {SVGElement[]}
 */
export function build(_records, _doc) {
  return [];
}

/**
 * Update arena nodes for new camera records: nothing to do, because build
 * makes none (see build). Returns undefined.
 *
 * @param {ReadonlyArray<Element>} _nodes
 * @param {ReadonlyArray<LookRecord>} _records
 */
export function update(_nodes, _records) {}

/**
 * Return the hover and focus explanation for a camera button record.
 *
 * record is {kind: "camera-choice", camera, product} for a camera button, or
 * {kind: "camera-first-tip"} for the Replay Viewer's one-time first-open tip
 * button (FIRST_OPEN_TIP; clicking it turns Battle Cam on). Return a semantic
 * "control" descriptor naming the camera, what it does and its key, or null for
 * any other record. The words never name a champion, so they are the same in
 * Oracle View and Agent POV.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  if (record.kind === "camera-first-tip") {
    return createSemanticDescriptor({
      kind: "control",
      id: "camera-first-tip",
      title: "Battle Cam",
      tone: "information",
      accent: "none",
      summary: "Turn on Battle Cam: the camera follows the fights for you.",
      rows: [{ label: "Key", value: "B", metadata: { compact: true, full: true } }],
      sections: [],
      metadata: { compact: true, full: true },
      anchor: "element",
    });
  }
  if (record.kind !== "camera-choice") return null;
  const camera = /** @type {CameraName} */ (record.camera);
  const live = record.product !== "replay_viewer";
  const summary =
    camera === "everyone_in_view"
      ? "Zooms in as far as it can while every champion stays in view, with room around them. Press V again for the Whole Map."
      : camera === "battle_cam"
        ? "Follows the fights for you: frames each fight, leans in on each fall and slows the fall in play. Press B again for the Whole Map."
        : live
          ? "Shows the whole arena, still. Press V for Everyone in View."
          : "Shows the whole arena, still. Press V for Everyone in View or B for Battle Cam.";
  const key = CAMERA_KEYCAPS[camera];
  return createSemanticDescriptor({
    kind: "control",
    id: `camera-choice:${camera}`,
    title: CAMERA_NAMES[camera],
    tone: "information",
    accent: "none",
    summary,
    rows: key
      ? [{ label: "Key", value: key, metadata: { compact: true, full: true } }]
      : [],
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "element",
  });
}

/**
 * @typedef {{
 *   setTimeout: (callback: () => void, ms: number) => unknown,
 *   clearTimeout: (handle: unknown) => void,
 * }} CameraTimers
 * @typedef {{
 *   slot: HTMLElement,
 *   product: CameraProduct,
 *   applyFrame: (frame: Readonly<{camera: CameraName, minX: number, minY: number, maxX: number, maxY: number}> | null, options: Readonly<{ease: boolean}>) => void,
 *   getFilters?: () => Readonly<Record<string, boolean>> | null,
 *   storage?: Storage | null,
 *   reducedMotion?: () => boolean,
 *   timers?: CameraTimers,
 *   registerKeys?: boolean,
 *   switchCameras?: ReadonlyArray<CameraName>,
 *   tipHost?: HTMLElement | null,
 * }} CameraSwitchOptions
 * @typedef {Readonly<{
 *   camera: () => CameraName,
 *   select: (camera: CameraName, options?: Readonly<{event?: Event | null}>) => boolean,
 *   pressKey: (key: string, event?: Event | null) => boolean,
 *   update: (input: LookInput & {reason?: string}) => ReadonlyArray<LookRecord>,
 *   destroy: () => void,
 * }>} CameraSwitch
 */

/**
 * Mount the camera switch in a slot and return its controller.
 *
 * options:
 * - slot: the empty container (#replay-camera-switch or #live-camera-switch).
 *   Its children are replaced and it gets class "camera-switch". In the
 *   Replay Viewer its hidden attribute is removed while camera_framing is on
 *   (set while off). In the BattleClient and the DevClient the slot stays
 *   hidden and empty: there is no switch on screen (owner, 1 October 2026).
 * - product: "replay_viewer" (three cameras, a button each unless
 *   switchCameras leaves one out, B, remembered camera, first-open
 *   tip and Battle Cam words) or "combat_debugger" (Whole Map and
 *   Everyone in View only, by V alone; always opens on Whole Map). For
 *   "combat_debugger" the change tip is one `span#camera-tip.camera-tip`
 *   (role status) appended to the deck's header line (#status-line) when the page
 *   has one, else to the slot's `.bc` page root (else the body, else the
 *   slot). Each time it shows in the deck's header line it takes the box of the
 *   deck header line's message (#msg: its offsetLeft, offsetTop, offsetWidth and
 *   offsetHeight in CSS px, written as the tip's inline left, top, width and
 *   height), and src/styles/battleclient/overlays.css draws it there. So it
 *   never covers the arena, the scorebug, the rail or any control (owner,
 *   3 Oct 2026); the message shows again when the tip goes. Without a status
 *   line, styles/camera.css places it at the bottom right of the window. A
 *   second mount replaces an older `#camera-tip`; destroy() removes it.
 * - applyFrame(frame, {ease}): called with the new camera frame (null for
 *   Whole Map) whenever it changes. The page calls setCameraFrame(frame) and,
 *   when ease is true, the choreographer's easeCamera after the re-render;
 *   ease false cuts.
 * - getFilters: returns the current visual filter state (default: null, every
 *   switch at its default).
 * - storage: Storage for the remembered camera and first-open tip (default
 *   window.localStorage when reachable, else null). Every access is guarded.
 * - reducedMotion: returns true under reduced motion (default false): camera
 *   changes cut and tips appear and go with no fade.
 * - timers: setTimeout and clearTimeout (default the window's).
 * - registerKeys: register the V and B handlers with command-set.js (default
 *   true). Only one switch per page may register them (the command set throws
 *   TypeError on a second registration).
 * - switchCameras: the offered cameras that get a button in the switch, in
 *   this order (default every offered camera). A camera left out still works
 *   by its key and its other on-screen button; the slot's data-camera names
 *   it while it is on, and no switch button is lit then. The Replay Viewer
 *   passes ["whole_map", "battle_cam"]: Everyone in View has its one button
 *   in the arena's corner (src/arena-buttons.js; owner, 3 Oct 2026).
 * - tipHost: Replay Viewer only, the element that holds the change tip and
 *   the first-open tip (default null: the slot holds them). The Replay
 *   Viewer passes the scorebug's #camera-tips (index.html), the empty inner
 *   half of the right-hand score column, so a tip covers nothing (owner,
 *   3 Oct 2026, Micro List M7). Its children are replaced at mount and its
 *   hidden attribute is removed; the tips keep their own hidden flags.
 *   Ignored for "combat_debugger" (its tip goes to the deck's header line).
 *
 * The controller: camera() is the chosen camera; select(camera) picks one
 * (returns false when it is already on, not offered, or camera_framing is
 * off), stores it (Replay Viewer), shows the change tip, and applies the frame
 * at once with an ease from the last update input; pressKey(key) does what V
 * or B does; update(input) is called after each render with the tick's plan
 * camera, beats, world size, scene and a reason ("play", "step", "seek",
 * "view", "recipient", "resize"; see cameraEase) and returns that tick's
 * records; destroy() removes listeners, timers and the key handlers. The
 * Replay Viewer's first-open tip follows the New Camera Tip switch
 * (camera_tip) live: mounting and every update or camera change repaint the
 * switch, which shows the tip once while this browser has not seen it and the
 * switch is on, and hides a showing tip as soon as the switch is off; once
 * seen it stays seen (storage). The tip is a button (clicking it turns
 * Battle Cam on) with its own Battle Cam hover help (tooltip, kind
 * "camera-first-tip").
 * Re-entrant update calls (applyFrame re-renders the page, which calls update
 * again) are ignored. DOM side effects stay inside slot (and tipHost when
 * given), except in live play:
 * the change tip in the deck's header line, and each update writes the view tag's
 * spot (viewTagPlace: --left-home-end and --floor-band, shares of the field's
 * width) as custom properties on #arena, which battleclient/overlays.css
 * reads to place #pov-tag; a zoomed camera leaves the last Whole Map spot.
 * Storage is written only by the Replay Viewer.
 *
 * Agent POV: update reads only what its input holds; pass the audience's own
 * plan camera, beats and scene.
 *
 * @param {CameraSwitchOptions} options
 * @returns {CameraSwitch}
 */
export function createCameraSwitch(options) {
  const { slot, product } = options;
  const doc = slot.ownerDocument;
  const storage = options.storage === undefined ? defaultStorage() : options.storage;
  const timers = options.timers ?? {
    setTimeout: (/** @type {() => void} */ callback, /** @type {number} */ ms) =>
      globalThis.setTimeout(callback, ms),
    clearTimeout: (/** @type {unknown} */ handle) =>
      globalThis.clearTimeout(/** @type {any} */ (handle)),
  };
  const reduced = () => options.reducedMotion?.() === true;
  const filters = () => options.getFilters?.() ?? null;
  const offered = camerasFor(product);
  const switchCameras = (options.switchCameras ?? offered).filter((camera) =>
    offered.includes(camera),
  );
  /** @type {CameraName} */
  let chosen =
    product === "replay_viewer" ? readRememberedCamera(storage) : "whole_map";
  /** @type {(LookInput & {reason?: string}) | null} */
  let lastInput = null;
  /** @type {string | null} */
  let appliedKey = null;
  let updating = false;
  /** @type {Event | null} */
  let tipEvent = null;
  /** @type {unknown} */
  let tipTimer = null;
  /** @type {unknown} */
  let firstTimer = null;

  slot.replaceChildren();
  slot.classList.add("camera-switch");
  slot.dataset.product = product;
  const label = element(doc, "span", "camera-switch__label", "Camera");
  const group = element(doc, "div", "camera-switch__choices");
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Camera");
  /** @type {Map<CameraName, HTMLButtonElement>} */
  const buttons = new Map();
  for (const camera of switchCameras) {
    const button = /** @type {HTMLButtonElement} */ (
      element(doc, "button", "camera-switch__choice")
    );
    button.type = "button";
    button.dataset.camera = camera;
    button.append(element(doc, "span", "camera-switch__name", CAMERA_NAMES[camera]));
    const key = CAMERA_KEYCAPS[camera];
    if (key) {
      const cap = element(doc, "kbd", "camera-switch__keycap", key);
      cap.setAttribute("aria-hidden", "true");
      button.append(cap);
      button.setAttribute("aria-keyshortcuts", key);
    }
    button.addEventListener("click", (event) => {
      select(camera === chosen ? "whole_map" : camera, { event });
    });
    const descriptor = tooltip(
      Object.freeze({ kind: "camera-choice", camera, product }),
    );
    if (descriptor) {
      try {
        registerTooltipOwner(button, descriptor);
      } catch {
        // A missing tooltip never blocks the switch; the button text says it all.
      }
    }
    buttons.set(camera, button);
    group.append(button);
  }
  const words = element(doc, "span", "camera-switch__words");
  // The BattleClient and DevClient show no switch (owner, 1 October 2026):
  // V alone changes the camera, and its tip (#camera-tip) sits over the
  // deck header line's message under the arena, never over the arena, the HUD,
  // the rail or a control.
  const live = product === "combat_debugger";
  // Where the Replay Viewer's two tips go: the slot, or the given host.
  const tipHost = !live && options.tipHost ? options.tipHost : slot;
  const tip = element(doc, "span", live ? "camera-tip" : "camera-switch__tip");
  tip.setAttribute("role", "status");
  tip.setAttribute("aria-live", "polite");
  tip.hidden = true;
  if (live) {
    tip.id = "camera-tip";
    doc.getElementById?.("camera-tip")?.remove();
    (
      doc.getElementById?.("status-line") ??
      slot.closest?.(".bc") ??
      doc.body ??
      slot
    ).append(tip);
  } else if (tipHost === slot) {
    slot.append(label, group, words, tip);
  } else {
    slot.append(label, group, words);
    tipHost.replaceChildren(tip);
    tipHost.hidden = false;
  }
  /** @type {HTMLButtonElement | null} */
  let firstTip = null;
  if (product === "replay_viewer") {
    firstTip = /** @type {HTMLButtonElement} */ (
      element(doc, "button", "camera-switch__first-tip", FIRST_OPEN_TIP)
    );
    firstTip.type = "button";
    firstTip.hidden = true;
    const firstTipHelp = tooltip(Object.freeze({ kind: "camera-first-tip" }));
    if (firstTipHelp) {
      try {
        registerTooltipOwner(firstTip, firstTipHelp);
      } catch {
        // A missing tooltip never blocks the tip; its own words say it all.
      }
    }
    firstTip.addEventListener("click", (event) => {
      hideFirstTip();
      select("battle_cam", { event });
    });
    tipHost.append(firstTip);
  }

  /** @param {Event} event */
  const onInput = (event) => {
    if (event === tipEvent) return;
    hideChangeTip();
    // A press on the first-open tip itself must reach its click (Battle Cam).
    const target = /** @type {Node | null} */ (event.target ?? null);
    if (!(firstTip && target && firstTip.contains(target))) hideFirstTip();
  };
  const view = doc.defaultView;
  for (const type of ["pointerdown", "keydown", "wheel"]) {
    view?.addEventListener(type, onInput, { capture: true, passive: true });
  }

  /** @type {Array<() => void>} */
  const unregister = [];
  if (options.registerKeys !== false) {
    const handler = (/** @type {any} */ call) =>
      pressKey(String(call?.key ?? ""), call?.event ?? null) ? undefined : false;
    unregister.push(registerLocalCommandHandler("camera_everyone_in_view", handler));
    unregister.push(registerLocalCommandHandler("camera_battle_cam", handler));
  }

  // Whether this page has shown the first-open tip already (at most once).
  let firstTipOffered = false;
  paintSwitch();

  /**
   * Show or hide the Replay Viewer's first-open tip from the New Camera Tip
   * switch (camera_tip), read live on every paint. Off hides a showing tip at
   * once. On shows the tip when this browser has not seen it yet
   * (firstOpenTipDue) and this page has not shown it: it marks the tip seen in
   * storage (FIRST_OPEN_TIP_SEEN_KEY), so once seen it stays seen, and hides
   * it after FIRST_OPEN_TIP_MS. No effect outside the Replay Viewer. Side
   * effects: the tip's hidden flag, one storage write, one timer.
   *
   * @returns {void}
   */
  function syncFirstTip() {
    if (firstTip === null) return;
    if (filters()?.camera_tip === false) {
      if (!firstTip.hidden) hideFirstTip();
      return;
    }
    if (firstTipOffered || !firstOpenTipDue(storage)) return;
    firstTipOffered = true;
    try {
      storage?.setItem(FIRST_OPEN_TIP_SEEN_KEY, "1");
    } catch {
      // Without storage the tip is simply not remembered.
    }
    firstTip.hidden = false;
    firstTimer = timers.setTimeout(hideFirstTip, FIRST_OPEN_TIP_MS);
  }

  function hideFirstTip() {
    if (firstTimer !== null) timers.clearTimeout(firstTimer);
    firstTimer = null;
    if (firstTip) firstTip.hidden = true;
  }

  function hideChangeTip() {
    if (tipTimer !== null) timers.clearTimeout(tipTimer);
    tipTimer = null;
    tip.hidden = true;
    tip.textContent = "";
    words.hidden = false;
  }

  /** @param {Event | null} event */
  function showChangeTip(event) {
    hideChangeTip();
    if (filters()?.camera_change_tip === false) return;
    tipEvent = event;
    tip.textContent = cameraChangeTip(
      effectiveCamera(chosen, product, filters()),
      product,
    );
    tip.hidden = false;
    tip.classList.toggle("camera-switch__tip--still", reduced());
    if (live) coverStatusMessage();
    words.hidden = true;
    tipTimer = timers.setTimeout(hideChangeTip, CHANGE_TIP_MS);
  }

  /**
   * Give the live tip the box of the deck header line's message (#msg), when the
   * tip sits in the deck's header line beside it: inline left, top, width and
   * height in CSS px from the message's offset box (both are placed against
   * the deck's header line). Reads layout once per shown tip. Does nothing when
   * the page has no message or the tip is elsewhere.
   */
  function coverStatusMessage() {
    const message = doc.getElementById?.("msg") ?? null;
    if (message === null || message.parentElement !== tip.parentElement) return;
    tip.style.left = `${message.offsetLeft}px`;
    tip.style.top = `${message.offsetTop}px`;
    tip.style.width = `${message.offsetWidth}px`;
    tip.style.height = `${message.offsetHeight}px`;
  }

  /**
   * Give the live arena box the view tag's spot (viewTagPlace) as two custom
   * properties on #arena: --left-home-end and --floor-band (shares of the
   * field's width), which battleclient/overlays.css reads to put the view
   * tag on the frame under the floor, past the left home strip. A zoomed
   * camera gives no spot, so the last Whole Map spot stays (the map is the
   * same all game). Does nothing on a page without #arena.
   *
   * @param {LookInput} input The tick's update input (transform and scene).
   */
  function placeViewTag(input) {
    const place = viewTagPlace(input);
    const arena = doc.getElementById?.("arena") ?? null;
    if (place === null || arena === null) return;
    // Five decimals: under 0.02 px of error on the widest field.
    const share = (/** @type {number} */ value) =>
      String(Math.round(value * 1e5) / 1e5);
    arena.style.setProperty("--left-home-end", share(place.homeEnd));
    arena.style.setProperty("--floor-band", share(place.band));
  }

  function paintSwitch() {
    const framing = filters()?.camera_framing !== false;
    slot.hidden = live || !framing;
    const shown = effectiveCamera(chosen, product, filters());
    for (const [camera, button] of buttons) {
      button.setAttribute("aria-pressed", String(camera === shown));
    }
    slot.dataset.camera = shown;
    syncFirstTip();
  }

  /**
   * @param {CameraName} camera
   * @param {Readonly<{event?: Event | null}>} [selectOptions]
   */
  function select(camera, selectOptions = {}) {
    if (filters()?.camera_framing === false) return false;
    if (!offered.includes(camera) || camera === chosen) return false;
    chosen = camera;
    if (product === "replay_viewer") rememberCamera(storage, camera);
    hideFirstTip();
    paintSwitch();
    showChangeTip(selectOptions.event ?? null);
    if (lastInput !== null) {
      apply({ ...lastInput, reason: "camera" });
    } else {
      // No tick yet: the next update applies the frame.
      appliedKey = null;
    }
    return true;
  }

  /**
   * @param {string} key
   * @param {Event | null} [event]
   */
  function pressKey(key, event = null) {
    if (filters()?.camera_framing === false) return false;
    const next = nextCamera(chosen, key, product);
    if (next === chosen) return false;
    return select(next, { event });
  }

  /** @param {LookInput & {reason?: string}} input */
  function apply(input) {
    const list = records({
      ...input,
      camera_name: chosen,
      product,
      filters: filters(),
      reducedMotion: input.reducedMotion ?? reduced(),
    });
    const frameRecord = list.find((record) => record.kind === "camera-frame");
    const frame =
      frameRecord && frameRecord.frame !== null
        ? Object.freeze({ camera: frameRecord.camera, ...frameRecord.frame })
        : null;
    const key = frame === null ? "whole" : JSON.stringify(frame);
    paintWords(list);
    paintSwitch();
    if (key !== appliedKey) {
      appliedKey = key;
      const ease =
        cameraEase({ reason: input.reason, reducedMotion: reduced() }) === "ease";
      updating = true;
      try {
        options.applyFrame(frame, Object.freeze({ ease }));
      } finally {
        updating = false;
      }
    }
    return list;
  }

  /** @param {ReadonlyArray<LookRecord>} list */
  function paintWords(list) {
    const slow = list.find((record) => record.kind === "slow-motion");
    const frame = list.find((record) => record.kind === "camera-frame");
    let text = "";
    if (slow) {
      text = slow.label;
    } else if (frame?.camera === "battle_cam") {
      text =
        frame.zoom > 1
          ? `Battle Cam · ${formatRate(frame.zoom)}×`
          : "Battle Cam · Whole Map";
    }
    words.textContent = text;
    words.classList.toggle("camera-switch__words--slow", Boolean(slow));
  }

  return Object.freeze({
    camera: () => chosen,
    select,
    pressKey,
    update(/** @type {LookInput & {reason?: string}} */ input) {
      if (updating) return Object.freeze([]);
      lastInput = input;
      const list = apply(input);
      if (live) placeViewTag(input);
      return list;
    },
    destroy() {
      hideChangeTip();
      hideFirstTip();
      for (const type of ["pointerdown", "keydown", "wheel"]) {
        view?.removeEventListener(type, onInput, { capture: true });
      }
      for (const remove of unregister.splice(0)) remove();
      if (live) tip.remove?.();
    },
  });
}

/**
 * Build the keyframes that play one Kill Cam lean on the camera wrapper.
 *
 * lean is a killCamLean result; restTransform, leanTransform and
 * creepTransform are viewport transforms (createViewportTransform) for the
 * resting frame (the one the DOM is drawn with), lean.frame and
 * lean.creepFrame; tickMs is the tick length in ms (plan.beats.tickMs). Return
 * frozen {keyframes, options}: the wrapper rests (identity) until the push,
 * eases in to the lean over the push window, creeps through the hold, eases
 * back over the back window and rests to the end. options are {duration:
 * tickMs, easing: "linear", fill: "backwards"}: no forward fill, so finishing
 * or cancelling the animation shows the resting frame (END). The caller starts
 * it through the choreographer's animation factory (so it is counted, scaled
 * with the playback rate and cancelled on seek). Throw RangeError for a
 * nonpositive tickMs. Pure.
 *
 * @param {NonNullable<ReturnType<typeof killCamLean>>} lean
 * @param {ScreenProjection} restTransform
 * @param {ScreenProjection} leanTransform
 * @param {ScreenProjection} creepTransform
 * @param {number} tickMs
 * @returns {Readonly<{keyframes: Keyframe[], options: Readonly<KeyframeAnimationOptions>}>}
 */
export function killCamKeyframes(
  lean,
  restTransform,
  leanTransform,
  creepTransform,
  tickMs,
) {
  if (!positive(tickMs)) throw new RangeError("tickMs must be positive.");
  const offset = (/** @type {number} */ ms) => Math.min(1, Math.max(0, ms / tickMs));
  const toLean = cameraMatrix(restTransform, leanTransform);
  const toCreep = cameraMatrix(restTransform, creepTransform);
  const identity = "matrix(1, 0, 0, 1, 0, 0)";
  const { push, hold, back } = lean.windows;
  return Object.freeze({
    keyframes: [
      { offset: 0, transform: identity },
      { offset: offset(push[0]), transform: identity, easing: "ease-in-out" },
      { offset: offset(push[1]), transform: toLean, easing: "ease-out" },
      { offset: offset(hold[1]), transform: toCreep, easing: "ease-in-out" },
      { offset: offset(back[1]), transform: identity },
      { offset: 1, transform: identity },
    ],
    options: Object.freeze({ duration: tickMs, easing: "linear", fill: "backwards" }),
  });
}

/**
 * Return the CSS matrix that moves a picture drawn with fromTransform onto the
 * same world drawn with toTransform (uniform scale and translate).
 *
 * Both are viewport transforms with worldToScreen. Pure.
 *
 * @param {ScreenProjection} fromTransform
 * @param {ScreenProjection} toTransform
 * @returns {string}
 */
export function cameraMatrix(fromTransform, toTransform) {
  const a0 = fromTransform.worldToScreen([0, 0]);
  const a1 = fromTransform.worldToScreen([1, 0]);
  const b0 = toTransform.worldToScreen([0, 0]);
  const b1 = toTransform.worldToScreen([1, 0]);
  const scale = (b1.x - b0.x) / (a1.x - a0.x || 1);
  const x = b0.x - scale * a0.x;
  const y = b0.y - scale * a0.y;
  return `matrix(${round(scale)}, 0, 0, ${round(scale)}, ${round(x)}, ${round(y)})`;
}

/* ---------- Private helpers: world geometry (world units, y up) ---------- */

/**
 * Read and check the map size.
 *
 * @param {Readonly<{worldWidth: number, worldHeight: number}>} input
 * @returns {{width: number, height: number}}
 * @throws {TypeError} When either size is not a positive finite number.
 */
function worldOf(input) {
  if (!positive(input.worldWidth) || !positive(input.worldHeight)) {
    throw new TypeError("worldWidth and worldHeight must be positive finite numbers.");
  }
  return { width: input.worldWidth, height: input.worldHeight };
}

/**
 * Return whether value is a finite number above zero.
 *
 * @param {unknown} value
 * @returns {value is number}
 */
function positive(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/**
 * The Whole Map result for a world.
 *
 * @param {{width: number, height: number}} world
 */
function wholeResult(world) {
  return Object.freeze({
    frame: Object.freeze({ minX: 0, minY: 0, maxX: world.width, maxY: world.height }),
    zoom: 1,
  });
}

/**
 * Read a world point [x, y]; null for anything else.
 *
 * @param {unknown} value
 * @returns {WorldPoint | null}
 */
function pointOf(value) {
  if (!Array.isArray(value) || value.length < 2) return null;
  const [x, y] = value;
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

/**
 * Read the plan camera's bodies, skipping malformed rows.
 *
 * @param {PlanCamera | null | undefined} camera
 * @returns {CameraBody[]}
 */
function cameraBodies(camera) {
  if (!isRecord(camera) || !Array.isArray(camera.bodies)) return [];
  /** @type {CameraBody[]} */
  const out = [];
  for (const body of camera.bodies) {
    if (!isRecord(body)) continue;
    const end = pointOf(body.end);
    if (end === null) continue;
    const state =
      body.state === "falling" || body.state === "returning" ? body.state : "living";
    out.push({
      publicAgentId:
        typeof body.publicAgentId === "string" ? body.publicAgentId : undefined,
      start: pointOf(body.start),
      end,
      state,
    });
  }
  return out;
}

/**
 * Stand-in camera bodies from the audience's own arena scene: every living
 * champion in scene.agents (alive true or life_state "alive") at its scene
 * position, as a "living" body that has not moved (start equals end). Used
 * only when the plan camera has no bodies. In Agent POV the scene holds only
 * champions the team sees, so nothing hidden is framed. Malformed rows are
 * skipped; no scene gives []. Pure.
 *
 * @param {Record<string, any> | null | undefined} scene The audience's arena scene.
 * @returns {CameraBody[]}
 */
function sceneCameraBodies(scene) {
  if (!isRecord(scene) || !Array.isArray(scene.agents)) return [];
  /** @type {CameraBody[]} */
  const out = [];
  for (const agent of scene.agents) {
    if (!isRecord(agent)) continue;
    if (agent.alive !== true && agent.life_state !== "alive") continue;
    const spot = pointOf(agent.position);
    if (spot === null) continue;
    out.push({
      publicAgentId:
        typeof agent.public_agent_id === "string" ? agent.public_agent_id : undefined,
      start: spot,
      end: spot,
      state: "living",
    });
  }
  return out;
}

/**
 * Read the plan camera's fights, skipping malformed rows and points.
 *
 * @param {PlanCamera | null | undefined} camera
 * @returns {CameraFight[]}
 */
function cameraFights(camera) {
  if (!isRecord(camera) || !Array.isArray(camera.fights)) return [];
  /** @type {CameraFight[]} */
  const out = [];
  for (const fight of camera.fights) {
    if (!isRecord(fight) || !Array.isArray(fight.points)) continue;
    const points = fight.points.map(pointOf).filter((point) => point !== null);
    if (points.length === 0) continue;
    out.push({ kind: fight.kind === "fall" ? "fall" : "blow", points });
  }
  return out;
}

/**
 * Every drawn body spot for the clean-edge check: the camera bodies' END spots
 * plus the audience scene's agent positions (earlier corpses).
 *
 * @param {CameraBody[]} bodies
 * @param {Record<string, any> | null | undefined} scene
 * @returns {WorldPoint[]}
 */
function drawnSpots(bodies, scene) {
  /** @type {WorldPoint[]} */
  const spots = bodies.map((body) => body.end);
  if (isRecord(scene) && Array.isArray(scene.agents)) {
    for (const agent of scene.agents) {
      const spot = isRecord(agent) ? pointOf(agent.position) : null;
      if (spot !== null) spots.push(spot);
    }
  }
  return spots;
}

/**
 * The four points a returning pad keeps inside a frame (room for its ring).
 *
 * @param {WorldPoint} pad
 * @returns {WorldPoint[]}
 */
function returnRoom(pad) {
  const room = RETURN_ROOM - (BODY_REACH + EDGE_INSET);
  return [
    pad,
    [pad[0] - room, pad[1]],
    [pad[0] + room, pad[1]],
    [pad[0], pad[1] - room],
    [pad[0], pad[1] + room],
  ];
}

/**
 * Bounding extent of points.
 *
 * @param {ReadonlyArray<WorldPoint>} points
 */
function extent(points) {
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

/**
 * The zoom (against the Whole Map, at least 1) needed to fit points plus a
 * margin, and the points' centre.
 *
 * @param {{width: number, height: number}} world
 * @param {ReadonlyArray<WorldPoint>} points
 * @param {number} margin
 * @returns {{zoom: number, centre: WorldPoint}}
 */
function zoomNeeded(world, points, margin) {
  const span = extent(points);
  const width = span.maxX - span.minX + 2 * margin;
  const height = span.maxY - span.minY + 2 * margin;
  return {
    zoom: Math.max(1, Math.min(world.width / width, world.height / height)),
    centre: [(span.minX + span.maxX) / 2, (span.minY + span.maxY) / 2],
  };
}

/**
 * Centre on one axis of a frame of size that holds [low, high] plus margin, as
 * near the map centre as possible (Everyone in View's calm placement).
 *
 * @param {number} low
 * @param {number} high
 * @param {number} margin
 * @param {number} size
 * @param {number} mapSize
 * @returns {number}
 */
function calmCentre(low, high, margin, size, mapSize) {
  const lowest = high + margin - size;
  const highest = low - margin;
  const start =
    lowest <= highest
      ? Math.min(Math.max(mapSize / 2 - size / 2, lowest), highest)
      : (lowest + highest) / 2;
  return start + size / 2;
}

/**
 * A frame box of zoom centred on (centreX, centreY), kept inside the map plus
 * MAP_EDGE_ALLOWANCE. Zoom 1 or less gives the Whole Map box.
 *
 * @param {{width: number, height: number}} world
 * @param {number} centreX
 * @param {number} centreY
 * @param {number} zoom
 * @returns {Box}
 */
function boxAt(world, centreX, centreY, zoom) {
  if (zoom <= 1 + 1e-4) return { x: 0, y: 0, w: world.width, h: world.height };
  const w = world.width / zoom;
  const h = world.height / zoom;
  const edge = MAP_EDGE_ALLOWANCE;
  const x =
    w >= world.width + 2 * edge
      ? world.width / 2 - w / 2
      : clamp(centreX - w / 2, -edge, world.width + edge - w);
  const y =
    h >= world.height + 2 * edge
      ? world.height / 2 - h / 2
      : clamp(centreY - h / 2, -edge, world.height + edge - h);
  return { x, y, w, h };
}

/**
 * @param {WorldFrame} frame
 * @returns {Box}
 */
function boxOf(frame) {
  return {
    x: frame.minX,
    y: frame.minY,
    w: frame.maxX - frame.minX,
    h: frame.maxY - frame.minY,
  };
}

/**
 * @param {Box} box
 * @returns {WorldFrame}
 */
function frameOf(box) {
  return Object.freeze({
    minX: round(box.x),
    minY: round(box.y),
    maxX: round(box.x + box.w),
    maxY: round(box.y + box.h),
  });
}

/**
 * Is a body centred at c (one axis) wholly outside or wholly inside the span
 * [start, start + size], keeping EDGE_INSET from the edge when inside?
 *
 * @param {number} c
 * @param {number} start
 * @param {number} size
 */
function cleanOn(c, start, size) {
  const low = c - BODY_REACH;
  const high = c + BODY_REACH;
  return (
    high <= start + EPSILON ||
    low >= start + size - EPSILON ||
    (low >= start + EDGE_INSET - EPSILON && high <= start + size - EDGE_INSET + EPSILON)
  );
}

/**
 * On one axis, the start in [lo, hi] nearest preferred where every centre in
 * centres is clean for a span of size, or null when none is.
 *
 * @param {number} preferred
 * @param {number} lo
 * @param {number} hi
 * @param {number} size
 * @param {number[]} centres
 * @returns {number | null}
 */
function slideAxis(preferred, lo, hi, size, centres) {
  const candidates = [preferred, lo, hi];
  for (const c of centres) {
    candidates.push(
      c - BODY_REACH - EDGE_INSET,
      c + BODY_REACH,
      c + BODY_REACH + EDGE_INSET - size,
      c - BODY_REACH - size,
    );
  }
  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of candidates) {
    if (candidate < lo - EPSILON || candidate > hi + EPSILON) continue;
    const start = clamp(candidate, lo, hi);
    if (!centres.every((c) => cleanOn(c, start, size))) continue;
    const gap = Math.abs(start - preferred);
    if (gap < bestDistance - EPSILON) {
      bestDistance = gap;
      best = start;
    }
  }
  return best;
}

/**
 * Slide box so every must point stays wholly inside (with a body's reach) and
 * every drawn body is wholly inside or wholly outside. Return the moved box, or
 * null when no position works at this zoom.
 *
 * @param {{width: number, height: number}} world
 * @param {Box} box
 * @param {ReadonlyArray<WorldPoint>} must
 * @param {ReadonlyArray<WorldPoint>} bodies
 * @returns {Box | null}
 */
function cleanBox(world, box, must, bodies) {
  const reach = BODY_REACH + EDGE_INSET;
  const edge = MAP_EDGE_ALLOWANCE;
  const fixX = box.w >= world.width + 2 * edge - EPSILON;
  const fixY = box.h >= world.height + 2 * edge - EPSILON;
  let xLow = fixX ? box.x : -edge;
  let xHigh = fixX ? box.x : world.width + edge - box.w;
  let yLow = fixY ? box.y : -edge;
  let yHigh = fixY ? box.y : world.height + edge - box.h;
  for (const point of must) {
    xLow = Math.max(xLow, point[0] + reach - box.w);
    xHigh = Math.min(xHigh, point[0] - reach);
    yLow = Math.max(yLow, point[1] + reach - box.h);
    yHigh = Math.min(yHigh, point[1] - reach);
  }
  if (xLow > xHigh + EPSILON || yLow > yHigh + EPSILON) return null;
  const overlaps = (
    /** @type {number} */ c,
    /** @type {number} */ start,
    /** @type {number} */ size,
  ) => c + BODY_REACH > start + EPSILON && c - BODY_REACH < start + size - EPSILON;
  let x = clamp(box.x, xLow, xHigh);
  let y = clamp(box.y, yLow, yHigh);
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const nextX = slideAxis(
      x,
      xLow,
      xHigh,
      box.w,
      bodies.filter((body) => overlaps(body[1], y, box.h)).map((body) => body[0]),
    );
    if (nextX === null) return null;
    x = nextX;
    const nextY = slideAxis(
      y,
      yLow,
      yHigh,
      box.h,
      bodies.filter((body) => overlaps(body[0], x, box.w)).map((body) => body[1]),
    );
    if (nextY === null) return null;
    y = nextY;
    const clean = bodies.every(
      (body) =>
        (!overlaps(body[1], y, box.h) || cleanOn(body[0], x, box.w)) &&
        (!overlaps(body[0], x, box.w) || cleanOn(body[1], y, box.h)),
    );
    if (clean) return { x, y, w: box.w, h: box.h };
  }
  return null;
}

/**
 * The Kill Cam windows from the beat schedule (ms from the tick start).
 *
 * @param {CameraBeats} beats
 */
function leanWindows(beats) {
  const fall = /** @type {readonly [number, number]} */ (beats.fall);
  const tick = positive(beats.tickMs) ? beats.tickMs : fall[1] + KILL_CAM.backRest;
  const impactStart = Array.isArray(beats.windows?.impact)
    ? beats.windows.impact[0]
    : fall[0] - KILL_CAM.minMove;
  const pushStart = Math.min(impactStart, fall[0] - KILL_CAM.minMove);
  const pushEnd = fall[0];
  const holdEnd = Math.max(
    fall[1],
    Array.isArray(beats.returnBeat) ? beats.returnBeat[1] : fall[1],
  );
  const backEnd = Math.min(
    tick,
    Math.max(holdEnd + KILL_CAM.minMove, tick - KILL_CAM.backRest),
  );
  return Object.freeze({
    push: Object.freeze(/** @type {const} */ ([Math.max(0, pushStart), pushEnd])),
    hold: Object.freeze(/** @type {const} */ ([pushEnd, holdEnd])),
    back: Object.freeze(/** @type {const} */ ([holdEnd, backEnd])),
  });
}

/**
 * Create an HTML element with a class and optional text.
 *
 * @param {Document} doc
 * @param {string} tag
 * @param {string} className
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function element(doc, tag, className, text) {
  const node = doc.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Return window.localStorage when it can be reached, else null (never throws).
 *
 * @returns {Storage | null}
 */
function defaultStorage() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

/**
 * Format a rate or zoom for words: at most two decimals, no trailing zeros.
 *
 * @param {number} value
 * @returns {string}
 */
function formatRate(value) {
  return String(Math.round(value * 100) / 100);
}

/**
 * @param {number} value
 * @param {number} low
 * @param {number} high
 */
function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

/**
 * Round to 1e-6 so frames compare equal across identical inputs.
 *
 * @param {number} value
 */
function round(value) {
  return Math.round(value * 1e6) / 1e6;
}

/**
 * @param {WorldPoint} a
 * @param {WorldPoint} b
 */
function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
