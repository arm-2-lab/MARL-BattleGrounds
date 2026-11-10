/**
 * @file The Flags and Bases look: Capture the Flag's two flags and two bases in
 * the arena, on two layers.
 *
 * The ground part (layer `flags-ground`, under every champion) draws each team's
 * base and every flag on the ground. A base is a rim of radius 0.66 world units
 * around its home (Team A one rim, Team B two, because Team B is always
 * double-lined) and a dashed capture ring at the capture distance, cut at the end
 * wall. A base shows its flag's state by strength: full while the flag is home,
 * 40% while it is away. A home or dropped flag is a pennant planted on a pole at
 * its position: Team A's a triangle, Team B's a swallowtail. A dropped flag also
 * gets the return countdown: a faint ring at the pickup and return reach and the
 * remaining-fraction arc in the flag's color.
 *
 * The above part (layer `flags-above`, over every champion) draws a carried flag
 * on its carrier's back, outlined in the light team color when this view sees the
 * carrier (that outline is the carrier's mark), the ticks left beside a dropped
 * flag's ring, and two one-tick cues: a pulse on a base when its flag goes home,
 * and the capture mark (eight short rays) around the capturer's base on the
 * capture tick.
 *
 * Sizes follow the zoom with a minimum: pixel sizes are set at 33.3 px per world
 * unit (the whole-arena camera) and grow above it, never shrink below it.
 *
 * The DOM always holds the END state. When a flag changes this tick, the look
 * also draws its start state as a "before" copy and marks the end state "after",
 * both with `data-beat-hold` attributes, so the choreographer shows the earlier
 * look until the change's beat (300 ms fades): "score" for a flag going home
 * (capture, return by touch, return by timer), so it lands when the scorebug
 * steps up; "fall", keyed by the carrier, for a drop; "impact" for a pickup and
 * for a grab and fall. A carried pennant carries `data-glide-follow`, so it rides
 * its carrier's glide; a carried pennant whose carrier this view cannot see
 * slides between its two public spots (animationSpecs), so it never steps
 * while bodies glide. A paused frame, a seek and a PNG export show END plus the
 * capture mark. With the Flags switch off, the look draws plain pennants and
 * plain bases, and no rings, arcs, numbers, outlines, pulses or marks.
 *
 * Who calls it: `scene.js`, once per part. `visual-filters.js` reads PAINT_PARTS.
 * The scorebug's flag chips (`match-summary.js`) draw the same pennant
 * (pennantGeometry) and wait for the same beat (flagChange).
 *
 * Boundary: records reads only this audience's arena scene (`capture_the_flag`),
 * this audience's flag trajectories (start and end of the tick) and the return
 * ticks, a public rule setting the caller passes in. In Agent POV the scene names
 * a carrier only when the view sees it, so an unseen carrier gets a plain
 * pennant. It never reads a researcher view or a match summary. No timers or CSS
 * animations.
 */
import {
  arcPath,
  asArray,
  circlePath,
  finiteNumber,
  isRecord,
  paintParts,
  pathNumber,
  polarPoint,
  setNodeAttributes,
  svgNode,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {import("./look-support.js").Rectangle} Rectangle
 * @typedef {readonly [number, number]} Pair
 * @typedef {{
 *   kind: "impact" | "fall" | "score",
 *   key: string | null,
 *   side: "before" | "after",
 *   from: number,
 *   fade: number,
 *   pulse?: number,
 * }} FlagHold
 *   One beat hold: the beat, the champion it is keyed by (only "fall"), which
 *   side of the change the node shows, the opacity before the beat, the fade in
 *   ms and, for a pulse, its peak opacity.
 */

/**
 * Store numbers for the Flags and Bases look. Names ending in `Units` are world
 * units; `Px` are CSS pixels at the reference scale (they grow with the zoom
 * above it); times are ms; opacities and shares are 0 to 1.
 */
export const FOOTPRINT = Object.freeze({
  referencePpu: 33.3,
  pennantUnits: 0.92,
  pennantWidth: 0.625,
  pennantLow: 0.62,
  notch: 0.62,
  carriedOffsetUnits: 0.12,
  baseRimUnits: 0.66,
  teamBRimGapPx: 1.5,
  away: 0.4,
  pulseUnits: 0.9,
  pulsePeak: 0.9,
  pulseFade: 600,
  markInnerUnits: 0.85,
  markOuterUnits: 1.1,
  beatFade: 300,
  ticksFontPx: 11,
});

/**
 * Node allowance per record: a Team B base is the largest (its group, the zone,
 * the ring and two rims: 5). No durable animations.
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 5,
  animationsPerRecord: 0,
  persistentNodesPerRecord: 0,
});

/**
 * The Flags switch (shown only in Capture the Flag games) switches the flag
 * marks: the rings, the countdown, the ticks left, the carrier's outline, the
 * base strength, the pulse and the capture mark. The pennants and the bases'
 * rims are game state and always drawn.
 */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "flag_marks" }, "flags"],
]);

/**
 * Read one switch from a visual filter state (null: the default). Pure.
 *
 * @param {Readonly<Record<string, boolean>> | null | undefined} filters
 * @param {string} id
 * @param {boolean} fallback
 */
function switchOn(filters, id, fallback) {
  const value = filters?.[id];
  return typeof value === "boolean" ? value : fallback;
}

/**
 * Return a finite world [x, y] pair, or null. Pure.
 *
 * @param {unknown} value
 * @returns {Pair | null}
 */
function worldPair(value) {
  const pair = asArray(value).map((part) => finiteNumber(part, Number.NaN));
  return pair.length === 2 && pair.every(Number.isFinite)
    ? /** @type {Pair} */ (/** @type {unknown} */ (pair))
    : null;
}

/**
 * Return one usable flag row (team 1 or 2, a known state, finite position and
 * home), or null. Pure.
 *
 * @param {unknown} flag
 * @returns {Record<string, any> | null}
 */
function usableFlag(flag) {
  return isRecord(flag) &&
    (flag.team_id === 1 || flag.team_id === 2) &&
    ["home", "carried", "dropped"].includes(flag.state) &&
    worldPair(flag.position) !== null &&
    worldPair(flag.home_position) !== null
    ? flag
    : null;
}

/**
 * Return the pole and pennant path text of one flag. Pure.
 *
 * Team A's pennant is a triangle and Team B's a swallowtail. foot is where the
 * pole stands (screen px); height is the pole's length in px. The pennant is
 * 0.625 of the height wide and fills the top 38% of the pole. side (default 1)
 * is the way it flies: 1 to the right, -1 to the left. The arena points every
 * flag toward its own home's end wall (flagDrawing), so a carried flag trails
 * behind its carrier and never flips at the middle, and it moves a pole in when
 * the cloth would pass the floor's edge. The scorebug's flag chip draws the same
 * shape at its own size.
 *
 * @param {number} teamId 1 or 2.
 * @param {Pair} foot
 * @param {number} height
 * @param {1 | -1} [side]
 * @returns {{pole: string, shape: string}}
 */
export function pennantGeometry(teamId, foot, height, side = 1) {
  const n = pathNumber;
  const [x, y] = foot;
  const top = y - height;
  const low = y - FOOTPRINT.pennantLow * height;
  const mid = (top + low) / 2;
  const width = side * FOOTPRINT.pennantWidth * height;
  const shape =
    teamId === 2
      ? `M${n(x)} ${n(top)}L${n(x + width)} ${n(top)}L${n(x + FOOTPRINT.notch * width)} ${n(mid)}L${n(x + width)} ${n(low)}L${n(x)} ${n(low)}Z`
      : `M${n(x)} ${n(top)}L${n(x + width)} ${n(mid)}L${n(x)} ${n(low)}Z`;
  return { pole: `M${n(x)} ${n(y)}L${n(x)} ${n(top)}`, shape };
}

/**
 * Say which flag a champion carries, in words, or null when it carries none:
 * "Carrying Team B's flag". captureTheFlag is the audience scene's
 * capture_the_flag (null outside Capture the Flag); presentationKey is the
 * champion's key in that scene. In Agent POV the scene names a carrier only
 * when the team sees it, so an unseen carrier gives null. The one owner of
 * these words, shared by the Champion Card, the agent inspector, the body's
 * screen-reader label and the roster's carrier chip. Pure.
 *
 * @param {unknown} captureTheFlag
 * @param {unknown} presentationKey
 * @returns {string | null}
 */
export function carryingWords(captureTheFlag, presentationKey) {
  const team = carriedFlagTeam(captureTheFlag, presentationKey);
  return team === null ? null : `Carrying ${team === 2 ? "Team B" : "Team A"}'s flag`;
}

/**
 * Return the team (1 or 2) whose flag a champion carries, or null when it
 * carries none. The same inputs and Agent POV rule as carryingWords. Pure.
 *
 * @param {unknown} captureTheFlag
 * @param {unknown} presentationKey
 * @returns {1 | 2 | null}
 */
export function carriedFlagTeam(captureTheFlag, presentationKey) {
  if (!isRecord(captureTheFlag) || typeof presentationKey !== "string") return null;
  const flag = asArray(captureTheFlag.flags).find(
    (row) =>
      isRecord(row) &&
      row.state === "carried" &&
      row.carrier_presentation_key === presentationKey,
  );
  return isRecord(flag) && (flag.team_id === 1 || flag.team_id === 2)
    ? flag.team_id
    : null;
}

/**
 * Return a circle's parts inside a screen rectangle (the floor) as path text:
 * `ring`, the arcs alone, and `zone`, the arcs joined along the wall and closed.
 * Both are empty when the circle lies outside. Pure.
 *
 * The SVG export refuses clip paths, so the cut is geometry: the circle is split
 * where it crosses an edge, and each piece whose middle is inside is kept.
 *
 * @param {Pair} center
 * @param {number} radius CSS pixels.
 * @param {Rectangle} bounds
 * @returns {{ring: string, zone: string}}
 */
function wallCut(center, radius, bounds) {
  const slack = 1e-6;
  /** @param {number} degrees */
  const inside = (degrees) => {
    const [x, y] = polarPoint(center, radius, degrees);
    return (
      x >= bounds.left - slack &&
      x <= bounds.right + slack &&
      y >= bounds.top - slack &&
      y <= bounds.bottom + slack
    );
  };
  /** @type {number[]} */
  const cuts = [];
  for (const edge of [bounds.left, bounds.right]) {
    const share = (edge - center[0]) / radius;
    if (Math.abs(share) < 1) {
      const angle = (Math.acos(share) * 180) / Math.PI;
      cuts.push(angle, 360 - angle);
    }
  }
  for (const edge of [bounds.top, bounds.bottom]) {
    const share = (edge - center[1]) / radius;
    if (Math.abs(share) < 1) {
      const angle = (Math.asin(share) * 180) / Math.PI;
      cuts.push((angle + 360) % 360, 180 - angle);
    }
  }
  if (cuts.length === 0) {
    const whole = inside(0) ? circlePath(center, radius) : "";
    return { ring: whole, zone: whole };
  }
  cuts.sort((left, right) => left - right);
  /** @type {Array<[number, number]>} */
  const arcs = [];
  for (const [index, start] of cuts.entries()) {
    const end = index + 1 < cuts.length ? cuts[index + 1] : cuts[0] + 360;
    if (end - start > slack && inside((start + end) / 2)) arcs.push([start, end]);
  }
  const pieces = arcs.map(([start, end]) => arcPath(center, radius, start, end));
  return {
    ring: pieces.join(""),
    zone: pieces.length
      ? `${pieces.map((piece, index) => (index ? `L${piece.slice(1)}` : piece)).join("")}Z`
      : "",
  };
}

/**
 * Return how a flag changed this tick and the beat its change waits for, or null
 * when nothing a viewer should see changed. Pure. The one owner of that rule:
 * the arena and the scorebug's flag chips both wait for the beat it names.
 *
 * start and end are the same flag (scene flag rows) at the start and end of the
 * tick; start is null when the tick has no start row (a fresh frame). Going home
 * (a capture from carried, a return from dropped) waits for "score"; a pickup
 * waits for "impact"; a drop waits for its carrier's "fall" (key: the start
 * carrier, null when this view did not name it, which shows END at once); a grab
 * and fall (from home, or from the ground to a new spot or a restarted countdown)
 * waits for "impact". A dropped flag counting down and a flag carried at both
 * ends are no change: Core never hands a carried flag to a new carrier within
 * one tick (a pickup takes only a flag nobody carries, and the death drop comes
 * after it), so the carrier is the same, or only enters or leaves this view's
 * sight.
 *
 * @param {Record<string, any> | null} start
 * @param {Record<string, any>} end
 * @returns {{beat: "impact" | "fall" | "score", key: string | null, capture: boolean, home: boolean, grab: boolean} | null}
 */
export function flagChange(start, end) {
  if (start === null) return null;
  const startKey =
    typeof start.carrier_presentation_key === "string"
      ? start.carrier_presentation_key
      : null;
  const change = (
    /** @type {"impact" | "fall" | "score"} */ beat,
    /** @type {string | null} */ key,
    flags = {},
  ) => ({ beat, key, capture: false, home: false, grab: false, ...flags });
  if (end.state === "home") {
    return start.state === "home"
      ? null
      : change("score", null, { home: true, capture: start.state === "carried" });
  }
  if (end.state === "carried") {
    return start.state === "carried" ? null : change("impact", null);
  }
  if (start.state === "carried") return change("fall", startKey);
  const a = /** @type {Pair} */ (worldPair(start.position));
  const b = /** @type {Pair} */ (worldPair(end.position));
  return start.state === "home" ||
    a[0] !== b[0] ||
    a[1] !== b[1] ||
    finiteNumber(end.countdown_steps) > finiteNumber(start.countdown_steps)
    ? change("impact", null, { grab: true })
    : null;
}

/**
 * Return the screen offset [dx, dy] in px from a carried flag's END spot back
 * to its START spot when the flag was carried at both ends, no carrier is named
 * at the END (Agent POV: a carrier the team cannot see, or one that just left
 * its sight) and the flag moved, else null. Both spots are public, so sliding
 * between them shows nothing hidden. A carrier named at the END needs none: the
 * pennant rides its body's glide (data-glide-follow). Pure.
 *
 * @param {Record<string, any> | null} start
 * @param {Record<string, any>} end
 * @param {LookTransform} transform
 * @returns {readonly [number, number] | null}
 */
function unseenCarryOffset(start, end, transform) {
  if (
    start === null ||
    start.state !== "carried" ||
    end.state !== "carried" ||
    typeof end.carrier_presentation_key === "string"
  ) {
    return null;
  }
  const a = transform.worldToScreen(/** @type {Pair} */ (worldPair(start.position)));
  const b = transform.worldToScreen(/** @type {Pair} */ (worldPair(end.position)));
  const dx = pathNumber(a.x - b.x);
  const dy = pathNumber(a.y - b.y);
  return dx === 0 && dy === 0 ? null : Object.freeze([dx, dy]);
}

/**
 * Return a frozen beat hold. Pure.
 *
 * @param {{beat: "impact" | "fall" | "score", key: string | null}} change
 * @param {"before" | "after"} side
 * @param {number} from
 * @param {{fade?: number, pulse?: number}} [extra]
 * @returns {Readonly<FlagHold>}
 */
function hold(change, side, from, extra = {}) {
  return Object.freeze({
    kind: change.beat,
    key: change.beat === "fall" ? change.key : null,
    side,
    from,
    fade: extra.fade ?? FOOTPRINT.beatFade,
    ...(extra.pulse === undefined ? {} : { pulse: extra.pulse }),
  });
}

/**
 * Return one flag's drawing records for one part, in paint order. Pure.
 *
 * view holds this render's scale and settings; hold (or null) marks every
 * record of this drawing. Ground: a dropped flag's countdown (marks on), then
 * the planted pennant of a home or dropped flag. Above: a carried flag's pennant
 * (outlined when the view names the carrier and marks are on; it follows the
 * carrier's glide when named), then a dropped flag's ticks left (marks on).
 *
 * glideFrom (default null) is, for a carried pennant with no carrier to ride,
 * the screen offset [dx, dy] in px from its END spot back to its START spot,
 * so animationSpecs can slide it.
 *
 * @param {Record<string, any>} flag
 * @param {"ground" | "above"} part
 * @param {Record<string, any>} view
 * @param {Readonly<FlagHold> | null} flagHold
 * @param {readonly [number, number] | null} [glideFrom]
 * @returns {LookRecord[]}
 */
function flagDrawing(flag, part, view, flagHold, glideFrom = null) {
  const teamId = flag.team_id;
  const point = view.transform.worldToScreen(worldPair(flag.position));
  const at = /** @type {Pair} */ ([point.x, point.y]);
  const carrier =
    typeof flag.carrier_presentation_key === "string"
      ? flag.carrier_presentation_key
      : null;
  const reach = finiteNumber(view.reach) * view.ppu;
  const countdown = Math.max(0, Math.trunc(finiteNumber(flag.countdown_steps)));
  /** @type {LookRecord[]} */
  const result = [];
  // Every pennant points toward its own home's end wall: only the enemy carries
  // a flag, always toward its own base, so a carried pennant trails behind its
  // carrier, and it never flips as it crosses the middle. The two ends mirror
  // each other whichever team spawns where.
  const side =
    finiteNumber(worldPair(flag.home_position)?.[0]) > view.transform.worldWidth / 2
      ? 1
      : -1;
  const height = FOOTPRINT.pennantUnits * view.scale;
  const cloth = FOOTPRINT.pennantWidth * height;
  const floorEdge = view.transform.worldToScreen([
    side > 0 ? view.transform.worldWidth : 0,
    0,
  ]).x;
  const pennant = (/** @type {Pair} */ foot, /** @type {boolean} */ outlined) =>
    Object.freeze({
      kind: "flag-pennant",
      teamId,
      state: flag.state,
      // At or near its home the cloth would pass the floor's edge, so the pole
      // moves in just far enough to keep it on the floor.
      ...pennantGeometry(
        teamId,
        [
          side > 0
            ? Math.min(foot[0], floorEdge - cloth)
            : Math.max(foot[0], floorEdge + cloth),
          foot[1],
        ],
        height,
        side,
      ),
      outlined,
      follow: flag.state === "carried" ? carrier : null,
      glideFrom: flag.state === "carried" && carrier === null ? glideFrom : null,
      u: view.u,
      hold: flagHold,
    });
  if (part === "ground") {
    if (flag.state === "dropped" && view.marks && reach > 0) {
      const share =
        view.returnTicks === null ? 0 : Math.min(1, countdown / view.returnTicks);
      result.push(
        Object.freeze({
          kind: "flag-countdown",
          teamId,
          track: circlePath(at, reach),
          arc: share > 0 ? arcPath(at, reach, -90, -90 + 359.9 * share) : "",
          u: view.u,
          hold: flagHold,
        }),
      );
    }
    if (flag.state !== "carried") result.push(pennant(at, false));
    return result;
  }
  if (flag.state === "carried") {
    // The pole stands just ahead of the carrier's center in its run (the side
    // away from where the pennant flies), so the cloth lies back over it.
    const offset = FOOTPRINT.carriedOffsetUnits * view.ppu;
    result.push(
      pennant([at[0] - side * offset, at[1] - offset], view.marks && carrier !== null),
    );
  } else if (flag.state === "dropped" && view.marks && reach > 0) {
    result.push(
      Object.freeze({
        kind: "flag-ticks",
        teamId,
        x: pathNumber(at[0] + reach + 4 * view.u),
        y: pathNumber(at[1] - reach + 6 * view.u),
        text: String(countdown),
        u: view.u,
        hold: flagHold,
      }),
    );
  }
  return result;
}

/**
 * Return the drawing records of this look for one part.
 *
 * input.scene is this audience's arena scene; only its `capture_the_flag` is
 * read (null outside Capture the Flag, which gives []): `flags` (Team A's flag,
 * then Team B's: team_id, state "home" | "carried" | "dropped", position and
 * home_position as world [x, y], carrier_presentation_key or null,
 * countdown_steps), `capture_distance` and `interaction_distance` (world units).
 * input.part is "ground" (default) or "above". input.transform projects; with
 * none the result is []. input.filters is the visual filter state (null: the
 * defaults, the flags switch on). input.flagTrajectories (optional, default
 * none) is this audience's rows for this tick: team_id, transition_start and
 * successor flag rows. input.returnTicks (optional) is the positive return-ticks
 * setting, public in every view; with none, a dropped flag shows its ring and
 * number but no arc.
 *
 * Records, each frozen with a `kind`. Ground: {kind: "flag-base", teamId, rim,
 * rimWidth, rimOpacity, ring, zone, u, strength, hold} per flag, then per flag
 * its "before" copy (when it changed) and its END drawing:
 * {kind: "flag-countdown", teamId, track, arc, u, hold} and {kind:
 * "flag-pennant", teamId, state, pole, shape, outlined, follow, glideFrom, u,
 * hold} (glideFrom: null, or the px offset back to the START spot of a carried
 * pennant with no carrier to ride).
 * Above: per flag the same two drawings of a carried pennant and {kind:
 * "flag-ticks", teamId, x, y, text, u, hold}, then {kind: "flag-pulse",
 * teamId, d, u, hold} for each flag that went home or was grabbed and dropped,
 * and {kind: "flag-mark", teamId, d, u, hold} at the capturer's base on a capture
 * tick. Path text is in screen px; `u` is the stroke scale (1 at 33.3 px per
 * unit); `hold` is null or a FlagHold. A flag without a usable team, state,
 * position or home is skipped.
 *
 * Pure data: no DOM. Agent POV: only the audience's own scene and trajectories
 * are read.
 *
 * @param {{
 *   part?: "ground" | "above" | null,
 *   scene?: Record<string, any> | null,
 *   transform?: LookTransform | null,
 *   audience?: "researcher" | "agent_pov" | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   flagTrajectories?: ReadonlyArray<unknown> | null,
 *   returnTicks?: number | null,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const ctf = isRecord(input.scene) ? input.scene.capture_the_flag : null;
  const transform = input.transform;
  if (!transform || !isRecord(ctf)) return Object.freeze([]);
  const part = input.part === "above" ? "above" : "ground";
  const ppu = transform.worldLengthToScreen(1);
  const scale = Math.max(ppu, FOOTPRINT.referencePpu);
  const returnTicks =
    Number.isInteger(input.returnTicks) && Number(input.returnTicks) > 0
      ? Number(input.returnTicks)
      : null;
  const view = {
    transform,
    ppu,
    scale,
    u: scale / FOOTPRINT.referencePpu,
    marks: switchOn(input.filters, "flags", true),
    reach: ctf.interaction_distance,
    returnTicks,
  };
  const flags = asArray(ctf.flags)
    .map(usableFlag)
    .filter((flag) => flag !== null);
  const starts = new Map(
    asArray(input.flagTrajectories)
      .filter(isRecord)
      .map((row) => [row.team_id, usableFlag(row.transition_start)]),
  );
  /** @type {LookRecord[]} */
  const result = [];
  if (part === "ground") {
    for (const flag of flags) {
      result.push(baseRecord(flag, starts.get(flag.team_id) ?? null, view, ctf));
    }
  }
  /** @type {LookRecord[]} */
  const cues = [];
  for (const flag of flags) {
    const start = starts.get(flag.team_id) ?? null;
    const change = flagChange(start, flag);
    if (change !== null && start !== null) {
      result.push(...flagDrawing(start, part, view, hold(change, "before", 1)));
    }
    result.push(
      ...flagDrawing(
        flag,
        part,
        view,
        change === null ? null : hold(change, "after", 0),
        change === null ? unseenCarryOffset(start, flag, transform) : null,
      ),
    );
    if (part !== "above" || change === null || !view.marks) continue;
    if (change.home || change.grab) {
      const center = transform.worldToScreen(
        /** @type {Pair} */ (
          worldPair(change.home ? flag.home_position : flag.position)
        ),
      );
      const radius = change.home
        ? FOOTPRINT.pulseUnits * ppu
        : finiteNumber(ctf.interaction_distance) * ppu;
      cues.push(
        Object.freeze({
          kind: "flag-pulse",
          teamId: flag.team_id,
          d: circlePath([center.x, center.y], radius),
          u: view.u,
          hold: hold(change, "after", 0, {
            fade: FOOTPRINT.pulseFade,
            pulse: FOOTPRINT.pulsePeak,
          }),
        }),
      );
    }
    const capturer = flags.find((other) => other.team_id !== flag.team_id);
    if (change.capture && capturer) {
      const home = transform.worldToScreen(
        /** @type {Pair} */ (worldPair(capturer.home_position)),
      );
      const c = /** @type {Pair} */ ([home.x, home.y]);
      let rays = "";
      for (let index = 0; index < 8; index += 1) {
        const [x0, y0] = polarPoint(c, FOOTPRINT.markInnerUnits * ppu, index * 45);
        const [x1, y1] = polarPoint(c, FOOTPRINT.markOuterUnits * ppu, index * 45);
        rays += `M${pathNumber(x0)} ${pathNumber(y0)}L${pathNumber(x1)} ${pathNumber(y1)}`;
      }
      cues.push(
        Object.freeze({
          kind: "flag-mark",
          teamId: capturer.team_id,
          d: rays,
          u: view.u,
          hold: hold(change, "after", 0),
        }),
      );
    }
  }
  return Object.freeze([...result, ...cues]);
}

/**
 * Return one team's base record: its rims, its wall-cut capture ring and zone,
 * and its strength (1 while its flag is home, 0.4 away; always 1 with marks off,
 * which also drops the ring and zone). When the flag left or reached home this
 * tick (marks on), the strength waits for the change's beat. Pure.
 *
 * @param {Record<string, any>} flag The END flag row.
 * @param {Record<string, any> | null} start The same flag at the tick's start.
 * @param {Record<string, any>} view
 * @param {Record<string, any>} ctf
 * @returns {LookRecord}
 */
function baseRecord(flag, start, view, ctf) {
  const point = view.transform.worldToScreen(worldPair(flag.home_position));
  const c = /** @type {Pair} */ ([point.x, point.y]);
  const rimRadius = FOOTPRINT.baseRimUnits * view.ppu;
  const gap = FOOTPRINT.teamBRimGapPx * view.u;
  const rim =
    flag.team_id === 2
      ? circlePath(c, rimRadius - gap) + circlePath(c, rimRadius + gap)
      : circlePath(c, rimRadius);
  const cut = view.marks
    ? wallCut(
        c,
        finiteNumber(ctf.capture_distance) * view.ppu,
        view.transform.mapBounds,
      )
    : { ring: "", zone: "" };
  const strengthOf = (/** @type {Record<string, any>} */ row) =>
    !view.marks || row.state === "home" ? 1 : FOOTPRINT.away;
  const change = flagChange(start, flag);
  const strength = strengthOf(flag);
  return Object.freeze({
    kind: "flag-base",
    teamId: flag.team_id,
    rim,
    rimWidth: (flag.team_id === 2 ? 1 : 2) * view.u,
    rimOpacity: flag.team_id === 2 ? 1 : 0.95,
    ring: cut.ring,
    zone: cut.zone,
    u: view.u,
    strength,
    hold:
      change !== null && start !== null && strengthOf(start) !== strength
        ? hold(change, "after", strengthOf(start))
        : null,
  });
}

/**
 * Write a beat hold on a node, with its END opacity. DOM change only.
 *
 * @param {Element} node
 * @param {Readonly<FlagHold> | null} flagHold
 * @param {number} end END opacity.
 */
function writeHold(node, flagHold, end) {
  if (flagHold === null) {
    setNodeAttributes(node, { opacity: end === 1 ? null : end });
    return;
  }
  setNodeAttributes(node, {
    opacity: flagHold.side === "before" ? 0 : flagHold.pulse === undefined ? end : 0,
    "data-beat-hold": flagHold.kind,
    "data-beat-side": flagHold.side,
    "data-beat-key": flagHold.key,
    "data-beat-from": flagHold.pulse === undefined ? flagHold.from : null,
    "data-beat-fade": flagHold.fade,
    "data-beat-pulse": flagHold.pulse ?? null,
  });
}

/**
 * Return the CSS team suffix of a record. Pure.
 *
 * @param {LookRecord} record
 */
function team(record) {
  return record.teamId === 2 ? "b" : "a";
}

/**
 * Build one record's node and its children, without geometry. DOM only.
 *
 * @param {LookRecord} record
 * @param {Document} doc
 * @returns {SVGElement | null}
 */
function nodeFor(record, doc) {
  const side = team(record);
  if (record.kind === "flag-base") {
    const group = svgNode(doc, "g", {
      class: `flag-base flag-base--team-${side}`,
      "data-team-id": record.teamId,
    });
    group.append(
      svgNode(doc, "path", { class: "flag-base__zone" }),
      svgNode(doc, "path", { class: "flag-base__ring" }),
      svgNode(doc, "path", { class: "flag-base__rim" }),
    );
    return group;
  }
  if (record.kind === "flag-countdown") {
    const group = svgNode(doc, "g", {
      class: `flag-countdown flag-countdown--team-${side}`,
      "data-team-id": record.teamId,
    });
    group.append(
      svgNode(doc, "path", { class: "flag-countdown__track" }),
      svgNode(doc, "path", { class: "flag-countdown__arc" }),
    );
    return group;
  }
  if (record.kind === "flag-pennant") {
    const group = svgNode(doc, "g", {
      class: `flag-pennant flag-pennant--team-${side}${record.outlined ? " flag-pennant--marked" : ""}`,
      "data-team-id": record.teamId,
      "data-state": record.state,
      "data-glide-follow": record.follow ?? null,
      "data-flag-glide": record.glideFrom ? record.glideFrom.join(" ") : null,
    });
    group.append(
      svgNode(doc, "path", { class: "flag-pennant__pole-shadow" }),
      svgNode(doc, "path", { class: "flag-pennant__pole" }),
      svgNode(doc, "path", { class: "flag-pennant__shape" }),
    );
    return group;
  }
  if (record.kind === "flag-ticks") {
    const text = svgNode(doc, "text", {
      class: `flag-ticks flag-ticks--team-${side}`,
      "data-team-id": record.teamId,
      "paint-order": "stroke",
    });
    text.textContent = record.text;
    return text;
  }
  if (record.kind === "flag-pulse" || record.kind === "flag-mark") {
    return svgNode(doc, "path", {
      class: `${record.kind} ${record.kind}--team-${side}`,
      "data-team-id": record.teamId,
    });
  }
  return null;
}

/**
 * Write a record's geometry, stroke widths and opacity onto its node (build and
 * update). DOM change only.
 *
 * @param {Element} node
 * @param {LookRecord} record
 */
function writeGeometry(node, record) {
  const u = record.u;
  const [first, second, third] = [...node.children];
  if (record.kind === "flag-base") {
    setNodeAttributes(first, { d: record.zone || null });
    setNodeAttributes(second, {
      d: record.ring || null,
      "stroke-width": 1.2 * u,
      "stroke-dasharray": `${pathNumber(5 * u)} ${pathNumber(4 * u)}`,
    });
    setNodeAttributes(third, {
      d: record.rim,
      "stroke-width": record.rimWidth,
      "stroke-opacity": record.rimOpacity,
    });
    writeHold(node, record.hold, record.strength);
  } else if (record.kind === "flag-countdown") {
    setNodeAttributes(first, { d: record.track, "stroke-width": 1.2 * u });
    setNodeAttributes(second, { d: record.arc || null, "stroke-width": 2.6 * u });
    writeHold(node, record.hold, 1);
  } else if (record.kind === "flag-pennant") {
    setNodeAttributes(first, { d: record.pole, "stroke-width": 3.2 * u });
    setNodeAttributes(second, { d: record.pole, "stroke-width": 1.6 * u });
    setNodeAttributes(third, {
      d: record.shape,
      "stroke-width": (record.outlined ? 1.6 : 1) * u,
    });
    writeHold(node, record.hold, 1);
  } else if (record.kind === "flag-ticks") {
    setNodeAttributes(node, {
      x: record.x,
      y: record.y,
      "font-size": pathNumber(FOOTPRINT.ticksFontPx * u),
      "stroke-width": 3 * u,
    });
    writeHold(node, record.hold, 1);
  } else if (record.kind === "flag-pulse" || record.kind === "flag-mark") {
    setNodeAttributes(node, {
      d: record.d,
      "stroke-width": (record.kind === "flag-mark" ? 2.4 : 2) * u,
    });
    writeHold(node, record.hold, 1);
  }
}

/** The one curve every body glide uses (choreography.js), so a flag keeps pace. */
const GLIDE_EASING = "cubic-bezier(0.45, 0, 0.55, 1)";

/**
 * Describe the slide of each carried pennant that has no carrier to ride.
 *
 * root is the battlefield (or any node holding this look's nodes); options are
 * the choreographer's: epochKey names the specs, mode "played" (a played tick)
 * or "step" (a -1 rewind), motionMode, and beats (the plan's beats, whose
 * windows.glide [start, end] in ms times the bodies' glides). In a played tick
 * with normal motion and Body Glide on (beats.glide not false), each node with
 * data-flag-glide ("dx dy", the px offset back to its START spot) slides from
 * there to its END spot along the bodies' glide window and curve (default 0 to
 * 260 ms). Reduced motion, motion off, Body Glide off and a step give none: the
 * pennant shows its END spot at once. Every spec uses fill "backwards", so
 * ending or cancelling it shows the END look the DOM holds. Return a new array
 * of {element, keyframes, options, id}; nothing is started. Reads the DOM only.
 * Agent POV: only this audience's nodes exist, and both spots are public.
 *
 * @param {ParentNode | null | undefined} root
 * @param {{
 *   epochKey: string,
 *   mode: "played" | "step",
 *   motionMode: "normal" | "reduced" | "off",
 *   beats?: Record<string, any> | null,
 * }} options
 * @returns {Array<{element: Element, keyframes: Keyframe[], options: KeyframeAnimationOptions, id: string}>}
 */
export function animationSpecs(root, options) {
  if (!root || typeof root.querySelectorAll !== "function") return [];
  const beats = isRecord(options.beats) ? options.beats : null;
  if (
    options.mode !== "played" ||
    options.motionMode !== "normal" ||
    beats?.glide === false
  ) {
    return [];
  }
  const window = asArray(beats?.windows?.glide);
  const start = finiteNumber(window[0], 0);
  const end = finiteNumber(window[1], 260);
  /** @type {Array<{element: Element, keyframes: Keyframe[], options: KeyframeAnimationOptions, id: string}>} */
  const specs = [];
  for (const element of root.querySelectorAll("[data-flag-glide]")) {
    const [dx, dy] = String(element.getAttribute("data-flag-glide"))
      .split(" ")
      .map((value) => finiteNumber(Number(value), 0));
    specs.push({
      element,
      keyframes: [
        { transform: `translate(${dx}px, ${dy}px)`, easing: GLIDE_EASING },
        { transform: "translate(0px, 0px)" },
      ],
      options: { delay: start, duration: Math.max(1, end - start), fill: "backwards" },
      id: `mbg:${options.epochKey}:flags:${specs.length}`,
    });
  }
  return specs;
}

/**
 * Build the nodes for records, in record order.
 *
 * records come from records(); doc creates the nodes. A "flag-base" record gives
 * a `g.flag-base` (zone, dashed ring and rim paths) whose opacity is its
 * strength; "flag-countdown" a `g.flag-countdown` (track and arc); "flag-pennant"
 * a `g.flag-pennant` (pole shadow, pole and pennant shape; `flag-pennant--marked`
 * when outlined; `data-glide-follow` naming the carrier when it follows one;
 * `data-flag-glide` with its slide offset when it has no carrier to ride);
 * "flag-ticks" a `text.flag-ticks`; "flag-pulse" and "flag-mark" one path each.
 * A record with a hold carries data-beat-hold, data-beat-side, data-beat-key (a
 * drop only), data-beat-from (or data-beat-pulse for a pulse) and
 * data-beat-fade; its opacity attribute is its END look (0 for a "before" copy
 * and a pulse). Side effects: DOM nodes only, no tooltips. Export-safe: flat
 * fills, strokes and opacity; no ids, clip paths or url() values. Return the
 * nodes.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @returns {SVGElement[]}
 */
export function build(records, doc) {
  /** @type {SVGElement[]} */
  const nodes = [];
  for (const record of records) {
    const node = nodeFor(record, doc);
    if (node === null) continue;
    writeGeometry(node, record);
    nodes.push(node);
  }
  return nodes;
}

/**
 * Move existing nodes to new records (reprojection after a resize or a camera
 * change). nodes are build()'s nodes and records the new records in the same
 * order and kinds. Return undefined; DOM change only.
 *
 * @param {ReadonlyArray<Element>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  for (const [index, node] of nodes.entries()) {
    const record = records[index];
    if (record !== undefined) writeGeometry(node, record);
  }
}

/**
 * Return the Symbol Tip of a record: always null, because flags and bases carry
 * no hover card (their layers are hidden from assistive technology, and the
 * scorebug's flag chips say each flag's state in words). Pure.
 *
 * @param {LookRecord} _record
 * @returns {null}
 */
export function tooltip(_record) {
  return null;
}
