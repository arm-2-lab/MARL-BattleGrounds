/**
 * @file The Rune Scrubber: the Replay Viewer's transport dock under the arena.
 *
 * What it owns: the one dock of two rows. Row 1 is the 36 px Event Track (an
 * SVG): the groove with its played part tinted Moon Silver, Team A Fallen marks
 * above the groove and Team B Fallen marks below it, Ultimate runes on the
 * groove in the caster's class color (filled for a damaging Ultimate, hollow
 * for Burst or Holy Word: Salvation, damaging drawn on top), return pips
 * outside the Fallen rows, the dim veil over the ticks ahead, the playhead and
 * the hollow 12 px thumb ring. It replaces the plain Tick slider: click or drag
 * to seek, hover a mark for its tip. Row 2 is one button row: Start, −10, −1,
 * Play/Pause, +1, +10, End, the readout ("Tick 118 of 204" with the beat label
 * under it: "Settled", "Playing" or "Paused · Strike"; a Replay Viewer notice
 * takes that line for 4 s, see showNotice), Speed, one Export menu button and
 * the camera switch. On the Replay Viewer page this module MOVES the
 * page's transport buttons, Speed select, artifact actions (Export PNG,
 * Download Metrics CSV, Episode Details, now inside the Export menu) and camera
 * switch into these rows. It never copies them: their owners keep their IDs and
 * handlers. The old slider, "Tick 110 / 245" readout and status line stay in
 * the DOM for keyboard and screen-reader users, visually hidden (sr-only). The
 * dock styles live in src/styles/rune-scrubber.css.
 *
 * Who calls it: main.js creates one view with createRuneScrubber(), passes every
 * ReplayPlaybackController snapshot to view.update(), and calls
 * view.presentTick() when a tick starts to play, so the current tick's marks
 * light at their beats (never before the arena shows them). main.js setNotice
 * hands every Replay Viewer notice to view.showNotice(). The pure helpers
 * (normalizeTimelineMarks, scrubberTrackRecords, scrubberReadout,
 * scrubberBeatLabel, scrubberBeatsFromSchedule and the rest) are exported for
 * tests and other callers.
 *
 * Keys: none. Every transport key (Space, the arrows, Shift + arrows and N,
 * which does the same as +1) belongs to replay-controls.js through the one
 * command set. The only key this module handles is Escape inside its own
 * Export menu, which closes that menu.
 *
 * Buttons stay live during play: a press on the track pauses first, then
 * seeks. At most one replay request is in flight; a press made while one is in
 * flight becomes the controller's one pending target, and a newer press
 * replaces it (ScrubberNavigator over ReplayPlaybackController's queue, shared
 * with the transport buttons and keys).
 *
 * Agent POV boundary: every mark comes from the host's per-audience timeline
 * marks (GET /api/replay/timeline-marks). In Agent POV that payload holds only
 * what the recipient's team saw. This module reads no frame, scene, researcher
 * space or match summary. When the audience changes it drops the old marks at
 * once and shows a plain groove until the new audience's marks arrive, and it
 * refuses a payload whose audience or recipient does not match the one it
 * asked for. Oracle -> Agent POV -> Oracle therefore gives exactly a fresh
 * render.
 *
 * Nothing here is ever part of the exported battlefield PNG.
 */
import { canonicalAgentIdentity, teamLabelFromId } from "./agent-identity.js";
import { classAbilityNameV1, classNamesV1 } from "./class-documentation.js";
import { registerTooltipOwner } from "./tooltip.js";
import { isVisualFilterEnabled } from "./visual-filters.js";

const SVG_NS = "http://www.w3.org/2000/svg";

/** Height of the Event Track in CSS pixels. */
export const SCRUBBER_TRACK_HEIGHT = 36;

/** Left and right gutter of the Event Track in CSS pixels. */
export const SCRUBBER_GUTTER = 16;

/** How long a notice stays under the readout before the beat label returns, in ms. */
export const SCRUBBER_NOTICE_MS = 4000;

/**
 * Vertical layout of the 36 px Event Track in CSS pixels, measured from its
 * top. Team A marks sit above the groove and Team B marks below it, so the team
 * reads from the row even in greyscale. Each entry is the top edge (y) and the
 * height of one row; rune is the center line of the Ultimate runes, which sit
 * on the groove. A second return on one tick grows a pip from 2 px to 4 px,
 * away from the groove (Team A upward, Team B downward).
 */
export const SCRUBBER_LANES = Object.freeze({
  pipA: Object.freeze({ y: 1, height: 2 }),
  fallA: Object.freeze({ y: 5, height: 9 }),
  groove: Object.freeze({ y: 15, height: 6 }),
  rune: 18,
  fallB: Object.freeze({ y: 22, height: 9 }),
  pipB: Object.freeze({ y: 33, height: 2 }),
});

const RUNE_RADIUS = 2.5;
const THUMB_RADIUS = 6;
const CLASS_TOKENS = Object.freeze({
  1: "mage",
  2: "warrior",
  3: "hunter",
  4: "rogue",
  5: "priest",
});

/**
 * @typedef {Readonly<{
 *   publicAgentId: string,
 *   teamId: 1 | 2,
 *   classId: 1 | 2 | 3 | 4 | 5,
 * }>} ScrubberAgentMark
 * @typedef {Readonly<ScrubberAgentMark & {homeFall: boolean}>} ScrubberFallMark
 * @typedef {Readonly<ScrubberAgentMark & {damaging: boolean}>} ScrubberUltimateMark
 * @typedef {Readonly<{
 *   frameIndex: number,
 *   falls: ReadonlyArray<ScrubberFallMark>,
 *   ultimates: ReadonlyArray<ScrubberUltimateMark>,
 *   returns: ReadonlyArray<ScrubberAgentMark>,
 * }>} ScrubberTickMarks
 * @typedef {Readonly<{
 *   audience: "oracle" | "agent_pov",
 *   recipientPublicAgentId: string | null,
 *   finalFrameIndex: number,
 *   ticks: ReadonlyArray<ScrubberTickMarks>,
 * }>} ScrubberMarks
 * @typedef {Readonly<{kind: "oracle"} | {kind: "agent_pov", recipientPublicAgentId: string | null}>} ScrubberAudience
 * @typedef {Readonly<{name: string, start: number, end: number}>} ScrubberBeatWindow
 * @typedef {Readonly<{
 *   impactMs: number | null,
 *   fallMs: number | null,
 *   returnMs: number | null,
 *   tickMs: number | null,
 *   windows: ReadonlyArray<ScrubberBeatWindow>,
 * }>} ScrubberBeats
 * @typedef {Readonly<{
 *   cursor: Readonly<{frame_index: number, final_frame_index: number}> | null,
 *   playing: boolean,
 *   requestPending: boolean,
 *   connected: boolean,
 *   hidden: boolean,
 *   transportState: string,
 *   pauseReason: string | null,
 *   playbackRate: number,
 *   atEnd?: boolean,
 *   targetFrameIndex?: number | null,
 * }>} ScrubberPlayback
 *   targetFrameIndex: the frame the controller's pending target or seek in
 *   flight goes to, or null (ReplayPlaybackController snapshot).
 * @typedef {{
 *   snapshot: () => ScrubberPlayback,
 *   seek: (frameIndex: number) => Promise<boolean>,
 *   pause: (reason?: string) => unknown,
 * }} ScrubberController
 */

/**
 * Return value when it is a plain object, otherwise throw TypeError naming label.
 *
 * @param {unknown} value
 * @param {string} label
 * @returns {Record<string, unknown>}
 */
function requireRecord(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`);
  }
  return /** @type {Record<string, unknown>} */ (value);
}

/**
 * Return value when it is an integer at least min, otherwise throw TypeError.
 *
 * @param {unknown} value
 * @param {number} min
 * @param {string} label
 * @returns {number}
 */
function requireInteger(value, min, label) {
  if (!Number.isSafeInteger(value) || /** @type {number} */ (value) < min) {
    throw new TypeError(`${label} must be an integer of at least ${min}.`);
  }
  return /** @type {number} */ (value);
}

/**
 * Return value when it is an array, otherwise throw TypeError.
 *
 * @param {unknown} value
 * @param {string} label
 * @returns {ReadonlyArray<unknown>}
 */
function requireArray(value, label) {
  if (!Array.isArray(value)) {
    throw new TypeError(`${label} must be a list.`);
  }
  return value;
}

/**
 * Check one champion named by a mark and return its frozen camelCase copy.
 * extra names the one extra boolean field ("home_fall", "damaging") or null.
 * Throws TypeError for a missing ID, a team other than 1 or 2, a class other
 * than 1 to 5, a non-boolean extra field or any unknown field.
 *
 * @param {unknown} raw
 * @param {"home_fall" | "damaging" | null} extra
 * @param {string} label
 * @returns {any}
 */
function normalizeAgentMark(raw, extra, label) {
  const record = requireRecord(raw, label);
  const allowed = new Set(["public_agent_id", "team_id", "class_id"]);
  if (extra !== null) allowed.add(extra);
  for (const key of Object.keys(record)) {
    if (!allowed.has(key)) throw new TypeError(`${label} has unknown field ${key}.`);
  }
  const publicAgentId = record.public_agent_id;
  if (typeof publicAgentId !== "string" || publicAgentId.length === 0) {
    throw new TypeError(`${label}.public_agent_id must be a nonempty string.`);
  }
  if (record.team_id !== 1 && record.team_id !== 2) {
    throw new TypeError(`${label}.team_id must be 1 or 2.`);
  }
  if (!Object.hasOwn(CLASS_TOKENS, String(record.class_id))) {
    throw new TypeError(`${label}.class_id must be 1 to 5.`);
  }
  /** @type {Record<string, unknown>} */
  const mark = {
    publicAgentId,
    teamId: record.team_id,
    classId: record.class_id,
  };
  if (extra !== null) {
    if (typeof record[extra] !== "boolean") {
      throw new TypeError(`${label}.${extra} must be true or false.`);
    }
    mark[extra === "home_fall" ? "homeFall" : "damaging"] = record[extra];
  }
  return Object.freeze(mark);
}

/**
 * Check the host's timeline marks payload and return a frozen camelCase copy.
 *
 * raw is the parsed JSON of GET /api/replay/timeline-marks
 * (ReplayTimelineMarksV1): schema_version 1, marks_kind "replay_timeline_marks",
 * audience "oracle" or "agent_pov", recipient_public_agent_id (a string in
 * Agent POV, null in Oracle View), final_frame_index (integer, 0 or more) and
 * ticks (only ticks with at least one mark; frame_index 1 or more, strictly
 * increasing; falls, ultimates and returns lists). Frame indices above
 * final_frame_index are refused, and so is any other field (the payload has
 * no fight list any more).
 *
 * @param {unknown} raw Parsed payload.
 * @returns {ScrubberMarks} A new deeply frozen record; raw is not changed.
 * @throws {TypeError} When any field is missing, unknown or out of range.
 */
export function normalizeTimelineMarks(raw) {
  const record = requireRecord(raw, "timeline marks");
  const expected = [
    "schema_version",
    "marks_kind",
    "audience",
    "recipient_public_agent_id",
    "final_frame_index",
    "ticks",
  ];
  for (const key of Object.keys(record)) {
    if (!expected.includes(key)) {
      throw new TypeError(`timeline marks has unknown field ${key}.`);
    }
  }
  if (record.schema_version !== 1) {
    throw new TypeError("timeline marks schema_version must be 1.");
  }
  if (record.marks_kind !== "replay_timeline_marks") {
    throw new TypeError("timeline marks kind must be replay_timeline_marks.");
  }
  if (record.audience !== "oracle" && record.audience !== "agent_pov") {
    throw new TypeError("timeline marks audience must be oracle or agent_pov.");
  }
  const recipient = record.recipient_public_agent_id;
  if (record.audience === "oracle" && recipient !== null) {
    throw new TypeError("Oracle timeline marks must have no recipient.");
  }
  if (
    record.audience === "agent_pov" &&
    (typeof recipient !== "string" || recipient.length === 0)
  ) {
    throw new TypeError("Agent POV timeline marks must name their recipient.");
  }
  const finalFrameIndex = requireInteger(
    record.final_frame_index,
    0,
    "final_frame_index",
  );
  let previousFrame = 0;
  const ticks = requireArray(record.ticks, "ticks").map((rawTick, index) => {
    const tick = requireRecord(rawTick, `ticks[${index}]`);
    for (const key of Object.keys(tick)) {
      if (!["frame_index", "falls", "ultimates", "returns"].includes(key)) {
        throw new TypeError(`ticks[${index}] has unknown field ${key}.`);
      }
    }
    const frameIndex = requireInteger(
      tick.frame_index,
      1,
      `ticks[${index}].frame_index`,
    );
    if (frameIndex <= previousFrame || frameIndex > finalFrameIndex) {
      throw new TypeError(`ticks[${index}].frame_index is out of order or range.`);
    }
    previousFrame = frameIndex;
    return Object.freeze({
      frameIndex,
      falls: Object.freeze(
        requireArray(tick.falls, `ticks[${index}].falls`).map((mark, j) =>
          normalizeAgentMark(mark, "home_fall", `ticks[${index}].falls[${j}]`),
        ),
      ),
      ultimates: Object.freeze(
        requireArray(tick.ultimates, `ticks[${index}].ultimates`).map((mark, j) =>
          normalizeAgentMark(mark, "damaging", `ticks[${index}].ultimates[${j}]`),
        ),
      ),
      returns: Object.freeze(
        requireArray(tick.returns, `ticks[${index}].returns`).map((mark, j) =>
          normalizeAgentMark(mark, null, `ticks[${index}].returns[${j}]`),
        ),
      ),
    });
  });
  return Object.freeze({
    audience: /** @type {"oracle" | "agent_pov"} */ (record.audience),
    recipientPublicAgentId: /** @type {string | null} */ (recipient),
    finalFrameIndex,
    ticks: Object.freeze(ticks),
  });
}

/**
 * Return the cache key of one audience: "oracle" or "agent_pov:<public ID>".
 * An Agent POV audience with no recipient yet gives "agent_pov:".
 *
 * @param {ScrubberAudience} audience
 * @returns {string}
 */
export function scrubberAudienceKey(audience) {
  return audience.kind === "oracle"
    ? "oracle"
    : `agent_pov:${audience.recipientPublicAgentId ?? ""}`;
}

/**
 * Return whether marks were built for audience: same kind and, in Agent POV,
 * the same recipient (a recipient of null in audience accepts any recipient).
 * The view uses this to refuse a payload that arrives after the view changed.
 *
 * @param {ScrubberMarks} marks
 * @param {ScrubberAudience} audience
 * @returns {boolean}
 */
export function marksMatchAudience(marks, audience) {
  if (marks.audience !== audience.kind) return false;
  if (audience.kind === "oracle") return marks.recipientPublicAgentId === null;
  return (
    audience.recipientPublicAgentId === null ||
    marks.recipientPublicAgentId === audience.recipientPublicAgentId
  );
}

/**
 * Return the short player label for a public agent ID: "Agent 3" for
 * "agent-3". Other IDs are shown as they are ("Agent scout-7").
 *
 * @param {string} publicAgentId
 * @returns {string}
 */
export function scrubberAgentLabel(publicAgentId) {
  const match = /^agent-([0-9])$/u.exec(publicAgentId);
  return `Agent ${match === null ? publicAgentId : match[1]}`;
}

/**
 * Return the readout of Row 2, for example "Tick 118 of 204". With no cursor
 * it returns "Tick —".
 *
 * @param {ScrubberPlayback} playback A playback snapshot.
 * @param {{tickForFrameIndex?: (frameIndex: number) => number}} [options]
 *   tickForFrameIndex turns a frame index into the shown simulator tick;
 *   default: the frame index itself.
 * @returns {string}
 */
export function scrubberReadout(playback, options = {}) {
  const cursor = playback.cursor;
  if (cursor === null) return "Tick —";
  const tickOf = options.tickForFrameIndex ?? ((frameIndex) => frameIndex);
  return `Tick ${tickOf(cursor.frame_index)} of ${tickOf(cursor.final_frame_index)}`;
}

/**
 * Return the beat label shown under the readout.
 *
 * "Playing" while playback runs; "Moving" while a seek is in flight; after the
 * viewer paused, "Paused · Strike" when pausedBeat names the beat the tick was
 * frozen in (Freeze On Pause), else "Paused"; otherwise "Settled" (a step, a
 * seek, the first load or the end of the replay).
 *
 * @param {ScrubberPlayback} playback A playback snapshot.
 * @param {string | null} [pausedBeat] The beat a mid-tick pause froze in, for
 *   example "Strike"; null or absent when the pause was not mid-tick.
 * @returns {string}
 */
export function scrubberBeatLabel(playback, pausedBeat = null) {
  if (playback.playing) return "Playing";
  if (playback.requestPending) return "Moving";
  if (playback.pauseReason === "user_pause") {
    return pausedBeat ? `Paused · ${pausedBeat}` : "Paused";
  }
  return "Settled";
}

/**
 * Round n to 2 decimals for compact SVG path text.
 *
 * @param {number} n
 * @returns {number}
 */
function round2(n) {
  return Math.round(n * 100) / 100;
}

/**
 * Return the SVG path text of one rune diamond (radius 2.5 px, so 5 px across)
 * centered on x, y.
 *
 * @param {number} x
 * @param {number} y
 * @returns {string}
 */
function diamondPath(x, y) {
  const r = RUNE_RADIUS;
  return `M${round2(x)} ${round2(y - r)}L${round2(x + r)} ${round2(y)}L${round2(x)} ${round2(y + r)}L${round2(x - r)} ${round2(y)}Z`;
}

/**
 * Return a rectangle as SVG path text.
 *
 * @param {number} x
 * @param {number} y
 * @param {number} width
 * @param {number} height
 * @returns {string}
 */
function rectPath(x, y, width, height) {
  return `M${round2(x)} ${round2(y)}h${round2(width)}v${round2(height)}h${round2(-width)}z`;
}

/**
 * @typedef {Readonly<{
 *   fallA: string, fallB: string, pipA: string, pipB: string,
 *   runes: ReadonlyArray<Readonly<{classToken: string, damaging: boolean, d: string}>>,
 * }>} ScrubberMarkPaths
 */

/**
 * Build the path text of the marks of the given ticks, one path per kind.
 *
 * Fallen marks are 2 x 9 px bars (a second and third fall of one team on one
 * tick add bars 3 px to the right); return pips are 3 x 2 px (4 px for two or
 * more returns, growing away from the groove); runes are 5 px diamonds on the
 * groove, grouped by class and by damaging or not. Damaging runes are listed
 * last, so they draw on top when several share a tick.
 *
 * @param {ReadonlyArray<ScrubberTickMarks>} ticks
 * @param {(frameIndex: number) => number} xOf Frame index to x in CSS pixels.
 * @returns {ScrubberMarkPaths}
 */
function markPaths(ticks, xOf) {
  const paths = { fallA: "", fallB: "", pipA: "", pipB: "" };
  /** @type {Map<string, {classToken: string, damaging: boolean, d: string}>} */
  const runes = new Map();
  for (const tick of ticks) {
    const x = xOf(tick.frameIndex);
    for (const teamId of /** @type {const} */ ([1, 2])) {
      const falls = tick.falls.filter((mark) => mark.teamId === teamId).length;
      const lane = teamId === 1 ? SCRUBBER_LANES.fallA : SCRUBBER_LANES.fallB;
      for (let j = 0; j < Math.min(falls, 3); j += 1) {
        const d = rectPath(x - 1 + j * 3, lane.y, 2, lane.height);
        if (teamId === 1) paths.fallA += d;
        else paths.fallB += d;
      }
      const returns = tick.returns.filter((mark) => mark.teamId === teamId).length;
      if (returns > 0) {
        const pip = teamId === 1 ? SCRUBBER_LANES.pipA : SCRUBBER_LANES.pipB;
        const height = returns >= 2 ? pip.height * 2 : pip.height;
        const y = teamId === 1 ? pip.y + pip.height - height : pip.y;
        const d = rectPath(x - 1.5, y, 3, height);
        if (teamId === 1) paths.pipA += d;
        else paths.pipB += d;
      }
    }
    for (const mark of tick.ultimates) {
      const classToken = CLASS_TOKENS[mark.classId];
      const key = `${classToken}|${mark.damaging}`;
      const entry = runes.get(key) ?? { classToken, damaging: mark.damaging, d: "" };
      entry.d += diamondPath(x, SCRUBBER_LANES.rune);
      runes.set(key, entry);
    }
  }
  const runeList = [...runes.values()]
    .sort((left, right) => Number(left.damaging) - Number(right.damaging))
    .map((entry) => Object.freeze(entry));
  return Object.freeze({ ...paths, runes: Object.freeze(runeList) });
}

/**
 * @typedef {Readonly<{
 *   width: number,
 *   height: number,
 *   x0: number,
 *   x1: number,
 *   perTick: number,
 *   groove: Readonly<{x: number, y: number, width: number, height: number}>,
 *   played: Readonly<{x: number, width: number}>,
 *   staticMarks: ScrubberMarkPaths,
 *   currentMarks: ScrubberMarkPaths & Readonly<{hasFalls: boolean, hasRunes: boolean, hasReturns: boolean}>,
 *   veil: Readonly<{x: number, width: number}> | null,
 *   playhead: Readonly<{x: number}>,
 *   hitTicks: ReadonlyArray<Readonly<{tick: ScrubberTickMarks, x: number, width: number}>>,
 * }>} ScrubberTrackRecords
 */

/**
 * Turn marks and the current frame into drawing records for the Event Track.
 * Pure: no DOM, no storage, no clock.
 *
 * Frame i sits at x = 16 + i * perTick, where perTick = (width - 32) /
 * finalFrameIndex (0 when finalFrameIndex is 0). The current frame's own marks
 * go to currentMarks (drawn above the veil and lit at their beats); every other
 * shown tick goes to staticMarks. The veil starts half a tick left of the
 * playhead and covers the ticks ahead, so the current tick's own marks under it
 * stay dim until the Current Tick layer lights them.
 *
 * showMarks false (switch event_timeline_marks off) leaves only the groove,
 * the played tint, the veil and the playhead. hideMarksAhead true drops the
 * marks past the playhead. marks null draws a plain groove.
 *
 * @param {{
 *   marks: ScrubberMarks | null,
 *   frameIndex: number,
 *   finalFrameIndex: number,
 *   width: number,
 *   showMarks?: boolean,
 *   hideMarksAhead?: boolean,
 * }} input width is the track width in CSS pixels (at least 64 is used).
 *   showMarks defaults to true and hideMarksAhead to false.
 * @returns {ScrubberTrackRecords}
 */
export function scrubberTrackRecords(input) {
  const width = Math.max(64, Number.isFinite(input.width) ? input.width : 64);
  const finalFrameIndex = Math.max(0, input.finalFrameIndex);
  const frameIndex = Math.max(0, Math.min(finalFrameIndex, input.frameIndex));
  const x0 = SCRUBBER_GUTTER;
  const x1 = width - SCRUBBER_GUTTER;
  const perTick = finalFrameIndex > 0 ? (x1 - x0) / finalFrameIndex : 0;
  /** @param {number} index */
  const xOf = (index) => x0 + index * perTick;
  const hideAhead = input.hideMarksAhead === true;
  const marks = input.marks;
  const showMarks = input.showMarks !== false && marks !== null;
  const shownTicks = showMarks
    ? marks.ticks.filter((tick) => !hideAhead || tick.frameIndex <= frameIndex)
    : [];
  const currentTick = shownTicks.find((tick) => tick.frameIndex === frameIndex) ?? null;
  const staticMarks = markPaths(
    shownTicks.filter((tick) => tick.frameIndex !== frameIndex),
    xOf,
  );
  const current = markPaths(currentTick === null ? [] : [currentTick], xOf);
  const hitWidth = Math.max(perTick, 4);
  const playheadX = xOf(frameIndex);
  const veilX = Math.max(x0 - 4, playheadX - perTick / 2);
  return Object.freeze({
    width,
    height: SCRUBBER_TRACK_HEIGHT,
    x0,
    x1,
    perTick,
    groove: Object.freeze({
      x: x0 - 4,
      y: SCRUBBER_LANES.groove.y,
      width: x1 - x0 + 8,
      height: SCRUBBER_LANES.groove.height,
    }),
    played: Object.freeze({ x: x0 - 4, width: round2(playheadX - (x0 - 4)) }),
    staticMarks,
    currentMarks: Object.freeze({
      ...current,
      hasFalls: currentTick !== null && currentTick.falls.length > 0,
      hasRunes: currentTick !== null && currentTick.ultimates.length > 0,
      hasReturns: currentTick !== null && currentTick.returns.length > 0,
    }),
    veil:
      finalFrameIndex > 0 && frameIndex < finalFrameIndex
        ? Object.freeze({ x: round2(veilX), width: round2(x1 + 4 - veilX) })
        : null,
    playhead: Object.freeze({ x: round2(playheadX) }),
    hitTicks: Object.freeze(
      shownTicks.map((tick) =>
        Object.freeze({
          tick,
          x: round2(xOf(tick.frameIndex) - hitWidth / 2),
          width: round2(hitWidth),
        }),
      ),
    ),
  });
}

/**
 * Write one champion as "Team A Mage · ID 3" from a mark. The display ID is
 * taken from a public ID of the form "agent-3"; other IDs are shown as they are.
 *
 * @param {ScrubberAgentMark} mark
 * @returns {string}
 */
function markAgentText(mark) {
  const displayMatch = /^agent-([0-9])$/u.exec(mark.publicAgentId);
  const identity = canonicalAgentIdentity({
    public_agent_id: mark.publicAgentId,
    class_id: mark.classId,
    team_id: mark.teamId,
    ...(displayMatch === null ? {} : { display_agent_id: displayMatch[1] }),
  });
  return `${teamLabelFromId(mark.teamId)} ${identity.title}`;
}

/**
 * Return the hover rows for one marked tick: who fell (with "Home Fall, +2"
 * when the fall scored double), whose Ultimate was cast (by its ability name)
 * and who returned. In Agent POV the marks are already limited to what the
 * recipient's team saw, so the rows name only seen champions.
 *
 * @param {ScrubberTickMarks} tick
 * @returns {Array<{label: string, value: string}>}
 */
export function tickMarkRows(tick) {
  /** @type {Array<{label: string, value: string}>} */
  const rows = [];
  for (const fall of tick.falls) {
    rows.push({
      label: "Fell",
      value: `${markAgentText(fall)}${fall.homeFall ? " · Home Fall, +2" : ""}`,
    });
  }
  for (const ultimate of tick.ultimates) {
    rows.push({
      label: "Ultimate",
      value: `${markAgentText(ultimate)} · ${
        classAbilityNameV1(ultimate.classId, "ultimate") ?? "Ultimate"
      }`,
    });
  }
  for (const back of tick.returns) {
    rows.push({ label: "Returned", value: markAgentText(back) });
  }
  return rows;
}

/**
 * Return the hover title for one marked tick: "Tick 58 · Team B Warrior Fell"
 * names the first fall, else the first Ultimate ("Tick 40 · Team A Rogue Used
 * Crippling Poison"), else the first return ("Tick 61 · Team B Warrior
 * Returned"). "· 2 More" follows when the tick holds further marks.
 *
 * @param {ScrubberTickMarks} tick
 * @param {number} shownTick The simulator tick shown for tick.frameIndex.
 * @returns {string}
 */
export function tickMarkTitle(tick, shownTick) {
  /**
   * Write "Team B Warrior" for one mark (team and class, no ID).
   *
   * @param {ScrubberAgentMark} mark
   * @returns {string}
   */
  const who = (mark) =>
    `${teamLabelFromId(mark.teamId)} ${classNamesV1(mark.classId)?.className ?? "Champion"}`;
  const first =
    tick.falls.length > 0
      ? `${who(tick.falls[0])} Fell`
      : tick.ultimates.length > 0
        ? `${who(tick.ultimates[0])} Used ${
            classAbilityNameV1(tick.ultimates[0].classId, "ultimate") ?? "Ultimate"
          }`
        : tick.returns.length > 0
          ? `${who(tick.returns[0])} Returned`
          : null;
  const more = tick.falls.length + tick.ultimates.length + tick.returns.length - 1;
  return [
    `Tick ${shownTick}`,
    ...(first === null ? [] : [first]),
    ...(more > 0 ? [`${more} More`] : []),
  ].join(" · ");
}

/** Beat names read from a BeatSchedule, in the order they start. */
const BEAT_WINDOW_NAMES = Object.freeze([
  ["stride", "Stride"],
  ["glide", "Glide"],
  ["chargeRush", "Charge"],
  ["strike", "Strike"],
  ["impact", "Impact"],
  ["settle", "Settle"],
]);

/**
 * Pick what the Event Track needs from a choreography BeatSchedule
 * (choreography-plan.js): impactMs from windows.impact[0] (runes light then),
 * fallMs from fall[0] (Fallen marks light at the Fall beat, with the arena and
 * the score), returnMs from returnBeat[0] (return pips), tickMs (the tick's
 * full length) and the named beat windows used by the "Paused · Strike" label
 * (Stride, Glide, Charge, Strike, Impact, Settle, Fall and Return; a missing
 * window is left out). Times are plan milliseconds at 1x. A missing schedule
 * gives null; a missing beat gives null for that beat (its marks show at once).
 *
 * @param {unknown} schedule
 * @returns {ScrubberBeats | null}
 */
export function scrubberBeatsFromSchedule(schedule) {
  if (schedule === null || typeof schedule !== "object") return null;
  const record = /** @type {Record<string, any>} */ (schedule);
  /**
   * Return the start of a [start, end] pair in milliseconds, or null when the
   * pair is missing or its start is not a finite number of at least 0.
   *
   * @param {unknown} pair
   * @returns {number | null}
   */
  const start = (pair) =>
    Array.isArray(pair) && Number.isFinite(pair[0]) && pair[0] >= 0
      ? Number(pair[0])
      : null;
  /**
   * Return [{name, start, end}] for a [start, end] pair of finite milliseconds
   * with end after start, or [] for a missing or empty window.
   *
   * @param {string} name The beat name shown in the label, for example "Strike".
   * @param {unknown} pair
   * @returns {ScrubberBeatWindow[]}
   */
  const windowOf = (name, pair) =>
    Array.isArray(pair) &&
    Number.isFinite(pair[0]) &&
    Number.isFinite(pair[1]) &&
    pair[1] > pair[0]
      ? [Object.freeze({ name, start: Number(pair[0]), end: Number(pair[1]) })]
      : [];
  const windows = [
    ...BEAT_WINDOW_NAMES.flatMap(([key, name]) =>
      windowOf(name, record.windows?.[key]),
    ),
    ...windowOf("Fall", record.fall),
    ...windowOf("Return", record.returnBeat),
  ];
  return Object.freeze({
    impactMs: start(record.windows?.impact),
    fallMs: start(record.fall),
    returnMs: start(record.returnBeat),
    tickMs:
      Number.isFinite(record.tickMs) && record.tickMs > 0
        ? Number(record.tickMs)
        : null,
    windows: Object.freeze(windows),
  });
}

/**
 * Return the name of the beat playing at ms plan milliseconds into a tick.
 *
 * Among the windows that hold ms, the one that started last wins (so Impact
 * beats Strike where they overlap). Before every window it is "Muster"; after
 * them but inside the tick it is "Hold". At or past the tick's end, or with no
 * windows, it is null (the tick has settled).
 *
 * @param {ScrubberBeats | null} beats From scrubberBeatsFromSchedule.
 * @param {number} ms Plan milliseconds since the tick started (at 1x).
 * @returns {string | null}
 */
export function scrubberBeatAt(beats, ms) {
  if (beats === null || !Number.isFinite(ms) || ms < 0) return null;
  if (beats.tickMs !== null && ms >= beats.tickMs) return null;
  let found = /** @type {ScrubberBeatWindow | null} */ (null);
  for (const window of beats.windows) {
    if (
      ms >= window.start &&
      ms < window.end &&
      (!found || window.start >= found.start)
    )
      found = window;
  }
  if (found !== null) return found.name;
  if (beats.windows.length === 0) return null;
  return ms < Math.min(...beats.windows.map((window) => window.start))
    ? "Muster"
    : "Hold";
}

/**
 * The Rune Scrubber's link to ReplayPlaybackController for track clicks and
 * drags. It keeps the track live during play and uses the controller's one
 * queue (ReplayPlaybackController: one request in flight, one pending target),
 * so a track seek and the transport buttons and keys share one target and the
 * last press wins.
 *
 * A press first pauses playback, then asks the controller to seek. If a request
 * is in flight the controller keeps the seek as its pending target (a newer
 * press replaces it) and sends it when that request lands.
 */
export class ScrubberNavigator {
  /**
   * @param {ScrubberController} controller The replay playback controller (or
   *   a test double with the same methods). It is not owned: dispose() leaves it.
   */
  constructor(controller) {
    this.controller = controller;
  }

  /**
   * Return the frame the thumb shows and the next relative press counts from:
   * the controller's targetFrameIndex (its pending target, else the seek in
   * flight), else the cursor's frame. Null when no cursor is installed.
   *
   * @returns {number | null}
   */
  baseFrame() {
    const state = this.controller.snapshot();
    if (state.cursor === null) return null;
    return state.targetFrameIndex ?? state.cursor.frame_index;
  }

  /**
   * Pause playback, then seek to frameIndex (clamped by the controller). Seek
   * failures are reported by the controller's own onError; they are not rethrown.
   *
   * @param {number} frameIndex Nonnegative integer frame index (a fraction is
   *   cut toward zero, a negative value becomes 0).
   * @returns {"sent" | "queued" | "refused"} sent: the seek went out at once;
   *   queued: it waits as the controller's pending target; refused: no cursor,
   *   offline or hidden.
   */
  seek(frameIndex) {
    let state = this.controller.snapshot();
    if (state.cursor === null || !state.connected || state.hidden) return "refused";
    if (state.playing) {
      this.controller.pause("user_pause");
      state = this.controller.snapshot();
    }
    const queued = state.requestPending;
    void this.controller.seek(Math.max(0, Math.trunc(frameIndex))).catch(() => {});
    return queued ? "queued" : "sent";
  }

  /** Nothing to release: the pending target lives in the controller. */
  dispose() {}
}

/**
 * Return whether the switch filterId is on in filters (null: the default
 * state). An unknown switch ID throws through isVisualFilterEnabled.
 *
 * @param {unknown} filters A visual filter state, or null.
 * @param {string} filterId
 * @param {boolean} fallback The registry default, used when filters is null.
 * @returns {boolean}
 */
function switchOn(filters, filterId, fallback) {
  return filters === null || filters === undefined
    ? fallback
    : isVisualFilterEnabled(filters, filterId);
}

/**
 * Create an element in doc's SVG namespace with the given attributes.
 *
 * @param {Document} doc
 * @param {string} tag
 * @param {Record<string, string | number>} attributes
 * @returns {SVGElement}
 */
function svgElement(doc, tag, attributes) {
  const node = /** @type {SVGElement} */ (doc.createElementNS(SVG_NS, tag));
  for (const [name, value] of Object.entries(attributes)) {
    node.setAttribute(name, String(value));
  }
  return node;
}

/**
 * Append one path per non-empty mark kind of paths to parent, with the
 * rune-scrubber__ CSS classes that give each kind its color. Return the groups
 * made: falls, runes and returns, so each can be lit at its own beat.
 *
 * @param {Document} doc
 * @param {SVGElement} parent
 * @param {ScrubberMarkPaths} paths
 * @returns {{falls: SVGElement, runes: SVGElement, returns: SVGElement}}
 */
function appendMarkPaths(doc, parent, paths) {
  const returns = svgElement(doc, "g", { class: "rune-scrubber__returns" });
  const falls = svgElement(doc, "g", { class: "rune-scrubber__falls" });
  const runes = svgElement(doc, "g", { class: "rune-scrubber__runes" });
  /**
   * Append one path with className and path text d to group; an empty d adds
   * nothing.
   *
   * @param {SVGElement} group
   * @param {string} className
   * @param {string} d
   */
  const add = (group, className, d) => {
    if (d) group.append(svgElement(doc, "path", { class: className, d }));
  };
  add(returns, "rune-scrubber__pip rune-scrubber__pip--a", paths.pipA);
  add(returns, "rune-scrubber__pip rune-scrubber__pip--b", paths.pipB);
  add(falls, "rune-scrubber__fall rune-scrubber__fall--a", paths.fallA);
  add(falls, "rune-scrubber__fall rune-scrubber__fall--b", paths.fallB);
  for (const rune of paths.runes) {
    add(
      runes,
      `rune-scrubber__rune rune-scrubber__rune--${rune.classToken} rune-scrubber__rune--${
        rune.damaging ? "damaging" : "support"
      }`,
      rune.d,
    );
  }
  parent.append(returns, falls, runes);
  return { falls, runes, returns };
}

/**
 * Build the Event Track's SVG from records (export-safe: flat fills, strokes
 * and opacity only; colors come from CSS classes in
 * src/styles/rune-scrubber.css; geometry comes from SVG attributes, never a
 * style attribute). Layers, bottom to top: groove, played tint, the marks of
 * every other tick, veil, playhead line, the Current Tick layer, the thumb
 * ring, then the invisible hover areas (one per marked tick).
 *
 * @param {ScrubberTrackRecords} records
 * @param {Document} doc
 * @returns {{svg: SVGSVGElement, current: {falls: SVGElement, runes: SVGElement, returns: SVGElement}, tickHits: SVGElement[]}}
 *   The new root and the parts the view wires up. Nothing is attached to the page.
 */
export function buildScrubberTrack(records, doc) {
  const svg = /** @type {SVGSVGElement} */ (
    /** @type {unknown} */ (
      svgElement(doc, "svg", {
        class: "rune-scrubber__svg",
        width: records.width,
        height: records.height,
        viewBox: `0 0 ${records.width} ${records.height}`,
        role: "img",
        "aria-label":
          "Event Track: falls, Ultimates and returns on the replay timeline",
      })
    )
  );
  const groove = records.groove;
  svg.append(
    svgElement(doc, "rect", {
      class: "rune-scrubber__groove",
      x: groove.x,
      y: groove.y,
      width: groove.width,
      height: groove.height,
      rx: 3,
    }),
    svgElement(doc, "rect", {
      class: "rune-scrubber__played",
      x: records.played.x,
      y: groove.y,
      width: Math.max(0, records.played.width),
      height: groove.height,
      rx: 3,
    }),
  );
  appendMarkPaths(doc, svg, records.staticMarks);
  if (records.veil !== null) {
    svg.append(
      svgElement(doc, "rect", {
        class: "rune-scrubber__veil",
        x: records.veil.x,
        y: 0,
        width: records.veil.width,
        height: records.height,
      }),
    );
  }
  svg.append(
    svgElement(doc, "path", {
      class: "rune-scrubber__playhead",
      d: `M${records.playhead.x} 0V${records.height}`,
    }),
  );
  const nowGroup = svgElement(doc, "g", { class: "rune-scrubber__now" });
  svg.append(nowGroup);
  const current = appendMarkPaths(doc, nowGroup, records.currentMarks);
  svg.append(
    svgElement(doc, "circle", {
      class: "rune-scrubber__thumb",
      cx: records.playhead.x,
      cy: SCRUBBER_LANES.rune,
      r: THUMB_RADIUS,
    }),
  );
  const hits = svgElement(doc, "g", { class: "rune-scrubber__hits" });
  /** @type {SVGElement[]} */
  const tickHits = [];
  for (const hit of records.hitTicks) {
    const rect = svgElement(doc, "rect", {
      class: "rune-scrubber__hit",
      "data-frame-index": hit.tick.frameIndex,
      x: hit.x,
      y: 0,
      width: hit.width,
      height: records.height,
    });
    tickHits.push(rect);
    hits.append(rect);
  }
  svg.append(hits);
  return { svg, current, tickHits };
}

/**
 * Return a "control" tooltip descriptor for tooltip.js.
 *
 * @param {string} id
 * @param {string} title
 * @param {string} summary
 * @param {Array<{label: string, value: string}>} [rows]
 * @returns {Record<string, unknown>}
 */
function controlTip(id, title, summary, rows = []) {
  return {
    kind: "control",
    id: `rune-scrubber:${id}`,
    title,
    tone: "information",
    accent: "none",
    summary,
    rows: rows.map((row) => ({ ...row, metadata: { compact: true, full: true } })),
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "element",
  };
}

/** IDs of the seven replay transport buttons, in the dock's order. */
const TRANSPORT_BUTTON_IDS = Object.freeze([
  "replay-first-button",
  "replay-back-ten-button",
  "replay-previous-button",
  "replay-play-pause-button",
  "replay-next-button",
  "replay-forward-ten-button",
  "replay-last-button",
]);

/**
 * Move the page's replay transport into the Rune Scrubber, so the Replay
 * Viewer has one dock: Row 1 the Event Track, Row 2 one button row.
 *
 * Row 2 reuses the page's transport fieldset (#replay-timeline
 * .replay-timeline__transport) and holds, in order: Start, −10, −1,
 * Play/Pause, +1, +10, End, the readout, Speed, the Export menu (the page's
 * #replay-artifact-actions fieldset is moved inside parts.exportMenu) and the
 * camera switch (#replay-camera-switch, filled by camera.js). Elements are
 * MOVED, never copied: their IDs, labels and handlers (owned by
 * replay-controls.js, camera.js and main.js) stay the same. The native Tick
 * slider, the old "Tick 110 / 245" readout and the status line stay in the DOM
 * at the row's end for keyboard and screen-reader users and for their owners'
 * writes, but get the sr-only class, so the Event Track is the one visible
 * scrubber and the Rune Scrubber readout the one visible readout.
 *
 * Side effects: moves page elements and adds the sr-only class to three of
 * them. Reads nothing about the match, so it has no Agent POV effect.
 *
 * @param {HTMLElement} container The scrubber's slot (#chronicle in index.html).
 * @param {{
 *   readout: HTMLElement,
 *   exportControl: HTMLElement,
 *   exportMenu: HTMLElement,
 * }} parts The scrubber's own readout box, Export control (button and menu)
 *   and the menu inside it.
 * @returns {{row: HTMLElement, slider: HTMLInputElement | null, exportDocked: boolean} | null}
 *   row: the docked button row. slider: the hidden native Tick slider, or
 *   null. exportDocked: whether the artifact actions were found and moved into
 *   the Export menu (otherwise no Export button is placed). Null (and nothing
 *   moved) when container is not inside #replay-timeline or the page lacks the
 *   transport fieldset or any of its seven buttons, as in unit tests with a
 *   fake DOM.
 */
function dockReplayTransport(container, parts) {
  const timeline = container.closest?.("#replay-timeline") ?? null;
  const row = /** @type {HTMLElement | null} */ (
    timeline?.querySelector(".replay-timeline__transport") ?? null
  );
  if (timeline === null || row === null) return null;
  /** @param {string} id @returns {HTMLElement | null} */
  const byId = (id) => timeline.querySelector(`#${id}`);
  const buttons = TRANSPORT_BUTTON_IDS.map(byId);
  if (buttons.some((node) => node === null)) return null;
  const sliderLabel = /** @type {HTMLElement | null} */ (
    row.querySelector(".replay-timeline__slider")
  );
  const quiet = [
    sliderLabel,
    byId("replay-frame-position"),
    byId("replay-transport-status"),
  ];
  for (const node of quiet) node?.classList.add("sr-only");
  const artifactActions = byId("replay-artifact-actions");
  if (artifactActions !== null) parts.exportMenu.append(artifactActions);
  row.classList.add("rune-scrubber__row");
  row.replaceChildren(
    .../** @type {HTMLElement[]} */ (
      [
        row.querySelector("legend"),
        ...buttons,
        parts.readout,
        row.querySelector(".replay-timeline__rate"),
        artifactActions === null ? null : parts.exportControl,
        byId("replay-camera-switch"),
        ...quiet,
      ].filter((node) => node !== null)
    ),
  );
  return {
    row,
    slider: /** @type {HTMLInputElement | null} */ (
      sliderLabel?.querySelector("input") ?? null
    ),
    exportDocked: artifactActions !== null,
  };
}

/**
 * @typedef {{
 *   playback: ScrubberPlayback,
 *   audience: ScrubberAudience,
 *   filters?: unknown,
 *   replayKey?: string | null,
 *   visible?: boolean,
 * }} ScrubberUpdate
 *   playback: the latest ReplayPlaybackController snapshot. audience: Oracle
 *   View, or Agent POV with its recipient's public agent ID (null while not yet
 *   known). filters: the visual filter state (null or absent: registry
 *   defaults). replayKey: the loaded replay's identity (artifact ID or digest;
 *   a new key drops every cached mark of the old replay). visible: false hides
 *   the scrubber (default true).
 * @typedef {{
 *   root: HTMLElement,
 *   docked: boolean,
 *   exportButton: HTMLButtonElement,
 *   navigator: ScrubberNavigator,
 *   update: (input: ScrubberUpdate) => void,
 *   presentTick: (frameIndex: number, beats: ScrubberBeats | null, playbackRate?: number) => void,
 *   showNotice: (text: string, level?: string) => void,
 *   dispose: () => void,
 * }} ScrubberView
 *   root: the container. docked: true when the page's replay transport was
 *   moved into the scrubber (one dock, see dockReplayTransport); false on a page
 *   without it, where only the track and the readout show. exportButton: the
 *   Export menu button (placed only when docked with the artifact actions).
 *   navigator: the track's link to the controller's one queue. update,
 *   presentTick, showNotice and dispose are documented where createRuneScrubber
 *   defines them.
 */

/**
 * Fill container (the page's empty scrubber slot) with the Rune Scrubber and
 * return its view.
 *
 * On the Replay Viewer page (container inside #replay-timeline with the
 * transport fieldset) the page's transport is moved in, so the dock is one
 * block: Row 1 the 36 px Event Track, which replaces the plain Tick slider
 * (click or drag to seek), and Row 2 one button row (see dockReplayTransport).
 * A docked view is shown at once and stays shown while no cursor exists, so
 * the transport never disappears. Without the transport (unit tests) the view
 * holds the track and the readout, starts hidden and update() shows it once a
 * cursor exists. visible: false always hides it.
 *
 * The thumb (with the playhead, played tint and veil) never jumps back or
 * skips: while the track is pressed it follows the pointer (a local preview,
 * while the seeks bring the arena along), and while a seek (from the track,
 * a button or a key) is queued or in flight it shows that seek's target.
 * Otherwise it shows the cursor's frame. The readout always says the cursor's tick, the one the arena shows.
 *
 * The Export button opens a small menu holding the page's artifact actions
 * (Export PNG, Download Metrics CSV and Episode Details, with their own IDs,
 * handlers and enabled states). Escape (focus returns to Export), a click
 * outside the menu, focus leaving it, or a press of one of its buttons closes
 * the menu. The Export button itself is never disabled.
 *
 * Notices (owner, 3 Oct 2026): showNotice() puts a Replay Viewer notice under
 * the readout, on the beat label's line and in the free space beside it up to
 * Speed, for SCRUBBER_NOTICE_MS (4 s); then the beat label comes back. The
 * whole sentence shows: a long one wraps upward over the tick line and the
 * track rather than being cut. The notice line is role="status" with
 * aria-live="polite", so screen readers hear it once.
 *
 * Call update() with every ReplayPlaybackController snapshot and whenever the
 * view (Oracle or Agent POV and its recipient), the visual filters or the
 * replay change. When the audience or replay changes, the old marks are
 * dropped at once and options.loadMarks() is called; its answer is used only if
 * the audience and replay are still the same and the payload says it was built
 * for that audience. Loaded marks are kept per (replay, audience) for the life
 * of the view, so Oracle -> Agent POV -> Oracle shows the Oracle marks again
 * with no new request and no leftover Agent POV marks. While marks load or
 * after they fail, the track host's data-marks attribute is "loading" or
 * "failed" and the groove is plain.
 *
 * Side effects: replaces container's children, adds the rune-scrubber class to
 * it, moves the page's transport elements (when present), adds click (Export
 * menu), pointer (track: a plain press seeks on pointerup, a drag scrubs),
 * focus and keydown listeners on its own elements and on the moved Tick slider
 * (focus only: the track shows the slider's focus ring), adds one pointerdown
 * listener on the document only while the Export menu is open, starts one 4 s
 * timer per notice, observes the
 * track's size with ResizeObserver when the browser has it, reads the clock
 * (performance.now) to name the beat a mid-tick pause froze in, and calls
 * controller seek, play and pause. Web Animations made by presentTick() hold the
 * current tick's marks dim until their beats; the DOM always holds the lit end
 * state, so cancelling them shows the truth. No storage is read or written.
 *
 * @param {HTMLElement} container The scrubber's slot (#chronicle in index.html).
 * @param {{
 *   controller: ScrubberController,
 *   loadMarks: () => Promise<unknown>,
 *   tickForFrameIndex?: (frameIndex: number) => unknown,
 * }} options controller: the ReplayPlaybackController. loadMarks: fetches GET
 *   /api/replay/timeline-marks for the current audience and resolves to the
 *   parsed JSON (rejects on failure). tickForFrameIndex: maps a frame to its
 *   simulator tick (a non-integer answer falls back to the frame index; default:
 *   the frame index).
 * @returns {ScrubberView}
 * @throws {TypeError} When container has no ownerDocument, or controller or
 *   loadMarks is missing.
 */
export function createRuneScrubber(container, options) {
  const doc = container?.ownerDocument;
  if (!doc) throw new TypeError("createRuneScrubber needs a page element.");
  if (!options?.controller || typeof options.loadMarks !== "function") {
    throw new TypeError("createRuneScrubber needs a controller and loadMarks.");
  }
  const controller = options.controller;
  const navigator = new ScrubberNavigator(controller);

  /**
   * Return the simulator tick shown for frameIndex (the frame index when the
   * caller's mapping gives no nonnegative integer).
   *
   * @param {number} frameIndex
   * @returns {number}
   */
  const tickOf = (frameIndex) => {
    const tick = options.tickForFrameIndex?.(frameIndex);
    return Number.isSafeInteger(tick) && Number(tick) >= 0 ? Number(tick) : frameIndex;
  };

  /**
   * Create one HTML element with a class and optional text.
   *
   * @param {string} tag
   * @param {string} className
   * @param {string} [text]
   * @returns {HTMLElement}
   */
  const html = (tag, className, text) => {
    const node = doc.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  /**
   * Return the clock in milliseconds (performance.now, else Date.now).
   *
   * @returns {number}
   */
  const now = () => globalThis.performance?.now?.() ?? Date.now();

  const trackHost = html("div", "rune-scrubber__track");
  trackHost.dataset.marks = "loading";
  const readoutBox = html("div", "rune-scrubber__readout");
  const readout = html("span", "rune-scrubber__tick");
  readout.id = "rune-scrubber-readout";
  const beatLabel = html("span", "rune-scrubber__beat");
  beatLabel.id = "rune-scrubber-beat";
  // The line under the tick holds the beat label, or a notice for 4 s.
  const lowerLine = html("span", "rune-scrubber__line");
  const noticeLine = html("span", "rune-scrubber__notice");
  noticeLine.id = "rune-scrubber-notice";
  noticeLine.setAttribute("role", "status");
  noticeLine.setAttribute("aria-live", "polite");
  noticeLine.hidden = true;
  lowerLine.append(beatLabel, noticeLine);
  readoutBox.append(readout, lowerLine);
  const exportControl = html("div", "rune-scrubber__export");
  const exportButton = /** @type {HTMLButtonElement} */ (
    html("button", "rune-scrubber__export-button", "Export")
  );
  exportButton.type = "button";
  exportButton.id = "rune-scrubber-export-button";
  exportButton.setAttribute("aria-expanded", "false");
  exportButton.setAttribute("aria-controls", "rune-scrubber-export-menu");
  registerTooltipOwner(
    exportButton,
    controlTip(
      "export",
      "Export",
      "Export PNG, Download Metrics CSV and Episode Details. Export PNG pauses and exports the settled still of this tick.",
    ),
  );
  const exportMenu = html("div", "rune-scrubber__export-menu");
  exportMenu.id = "rune-scrubber-export-menu";
  exportMenu.setAttribute("role", "group");
  exportMenu.setAttribute("aria-label", "Export");
  exportMenu.hidden = true;
  exportControl.append(exportButton, exportMenu);
  container.classList.add("rune-scrubber");
  const dock = dockReplayTransport(container, {
    readout: readoutBox,
    exportControl,
    exportMenu,
  });
  if (dock === null) {
    container.replaceChildren(trackHost, readoutBox);
  } else {
    container.replaceChildren(trackHost, dock.row);
    container.hidden = false;
  }

  /** @type {Map<string, ScrubberMarks>} */
  const marksCache = new Map();
  /** @type {ScrubberMarks | null} */
  let marks = null;
  let loadState = /** @type {"loading" | "failed" | "ready"} */ ("loading");
  let loadGeneration = 0;
  /** @type {string | null} */
  let loadedKey = null;
  /** @type {string | null | undefined} */
  let cachedReplayKey;
  /** @type {ScrubberUpdate | null} */
  let last = null;
  /** @type {{falls: SVGElement, runes: SVGElement, returns: SVGElement} | null} */
  let currentGroups = null;
  /** @type {ReadonlyArray<unknown> | null} */
  let trackInputs = null;
  /** @type {number | null} */
  let trackFrame = null;
  /** @type {Animation[]} */
  let animations = [];
  /**
   * The clock of the tick presentTick() last started, used to name the beat a
   * mid-tick pause froze in. pausedAtMs is set once, at the first paused render.
   * @type {{frameIndex: number, beats: ScrubberBeats, rate: number, startedAt: number, pausedAtMs: number | null} | null}
   */
  let tickClock = null;
  let disposed = false;
  /** The timer that ends the notice shown under the readout. @type {ReturnType<typeof setTimeout> | null} */
  let noticeTimer = null;
  /** The pointer pressing the track, or null. */
  let dragPointer = /** @type {number | null} */ (null);
  /** The frame under the pointer while the track is pressed, or null. */
  let dragFrame = /** @type {number | null} */ (null);

  /** Cancel every beat animation, so the track shows its lit end state. */
  const cancelAnimations = () => {
    for (const animation of animations) animation.cancel();
    animations = [];
  };

  /**
   * Return the playback snapshot of the last update, or the controller's own.
   *
   * @returns {ScrubberPlayback}
   */
  const playbackState = () => last?.playback ?? controller.snapshot();

  /**
   * Return whether a switch is on in the last update's filters (registry
   * default when no filters were given).
   *
   * @param {string} filterId
   * @param {boolean} fallback
   * @returns {boolean}
   */
  const on = (filterId, fallback) =>
    switchOn(last?.filters ?? null, filterId, fallback);

  /**
   * Return "Agent 3" for an Agent POV view (the recipient from the update,
   * else from the loaded marks), "The Chosen Agent" while neither names one,
   * or null in Oracle View.
   *
   * @returns {string | null}
   */
  const recipientText = () => {
    if (last?.audience.kind !== "agent_pov") return null;
    const id =
      last.audience.recipientPublicAgentId ?? marks?.recipientPublicAgentId ?? null;
    return id ? scrubberAgentLabel(id) : "The Chosen Agent";
  };

  /**
   * Start loading the current (replay, audience) marks unless they are already
   * shown. A cached copy is used at once. Otherwise the old marks are dropped
   * (so nothing from another view stays drawn) and loadMarks() runs; a late
   * answer is ignored, and a failed, malformed or mismatched one leaves a plain
   * groove (data-marks "failed"). Never throws.
   */
  const requestMarks = () => {
    if (last === null) return;
    const key = `${last.replayKey ?? ""}|${scrubberAudienceKey(last.audience)}`;
    if (key === loadedKey) return;
    loadedKey = key;
    loadGeneration += 1;
    const generation = loadGeneration;
    const audience = last.audience;
    const cached = marksCache.get(key);
    if (cached !== undefined) {
      marks = cached;
      loadState = "ready";
      return;
    }
    marks = null;
    loadState = "loading";
    void Promise.resolve()
      .then(() => options.loadMarks())
      .then((raw) => {
        if (disposed || generation !== loadGeneration) return;
        const normalized = normalizeTimelineMarks(raw);
        if (!marksMatchAudience(normalized, audience)) {
          throw new TypeError("Timeline marks were built for another view.");
        }
        marksCache.set(key, normalized);
        marks = normalized;
        loadState = "ready";
        render();
      })
      .catch(() => {
        if (disposed || generation !== loadGeneration) return;
        marks = null;
        loadState = "failed";
        render();
      });
  };

  /**
   * Rebuild the track SVG and its tips when anything it draws changed (frame,
   * length, width, marks, switches or view). Otherwise keep the nodes, so a
   * running beat animation is not lost to a state change in the same tick. A
   * rebuild cancels the beat animations.
   *
   * @param {number} frameIndex
   * @param {number} finalFrameIndex
   */
  const renderTrack = (frameIndex, finalFrameIndex) => {
    const width = Math.max(
      64,
      Math.round(trackHost.getBoundingClientRect?.().width || 640),
    );
    const recipient = recipientText();
    const inputs = [
      marks,
      frameIndex,
      finalFrameIndex,
      width,
      on("event_timeline_marks", true),
      on("hide_marks_ahead", false),
      recipient,
    ];
    if (
      trackInputs !== null &&
      inputs.every((value, index) => value === trackInputs?.[index])
    ) {
      return;
    }
    trackInputs = inputs;
    cancelAnimations();
    const records = scrubberTrackRecords({
      marks,
      frameIndex,
      finalFrameIndex,
      width,
      showMarks: on("event_timeline_marks", true),
      hideMarksAhead: on("hide_marks_ahead", false),
    });
    const built = buildScrubberTrack(records, doc);
    built.tickHits.forEach((hit, index) => {
      const tick = records.hitTicks[index].tick;
      registerTooltipOwner(
        hit,
        controlTip(
          `tick-${tick.frameIndex}`,
          tickMarkTitle(tick, tickOf(tick.frameIndex)),
          recipient === null
            ? "What happened on this tick."
            : `What ${recipient}'s team saw on this tick.`,
          tickMarkRows(tick),
        ),
        { inspectable: false },
      );
    });
    trackHost.replaceChildren(built.svg);
    currentGroups = built.current;
    trackFrame = frameIndex;
  };

  /**
   * Return the beat a mid-tick pause froze in ("Strike"), or null. It names a
   * beat only after a pause by the viewer with Freeze On Pause on, on the tick
   * presentTick() last started; the time is read once at the first paused
   * render and then kept, so the label holds as still as the arena does.
   *
   * @param {ScrubberPlayback} playback
   * @param {number} frameIndex
   * @returns {string | null}
   */
  const pausedBeat = (playback, frameIndex) => {
    if (
      playback.playing ||
      playback.requestPending ||
      playback.pauseReason !== "user_pause" ||
      !on("freeze_on_pause", true) ||
      tickClock === null ||
      tickClock.frameIndex !== frameIndex
    ) {
      return null;
    }
    if (tickClock.pausedAtMs === null) {
      tickClock.pausedAtMs = (now() - tickClock.startedAt) * tickClock.rate;
    }
    return scrubberBeatAt(tickClock.beats, tickClock.pausedAtMs);
  };

  /**
   * Return the frame the playhead, veil and thumb are drawn at (the A1 rule:
   * the thumb never jumps back or skips). While the track is pressed it is the
   * frame under the pointer; while a seek is queued or in flight it is that
   * seek's target (the controller's targetFrameIndex), so landing on an earlier
   * target does not pull
   * the thumb back; otherwise it is the cursor's frame. The readout always
   * says the cursor's tick, the one the arena shows.
   *
   * @param {number} cursorFrame The cursor's frame index (0 with no cursor).
   * @returns {number}
   */
  const trackFrameFor = (cursorFrame) =>
    dragFrame ?? navigator.baseFrame() ?? cursorFrame;

  /**
   * Draw everything from the last update: visibility, readout, beat label,
   * track and the data-marks state. A pause made by the viewer holds the beat
   * animations when Freeze On Pause is on (the arena holds its moment too); any
   * other stop cancels them, showing the settled end.
   */
  const render = () => {
    if (disposed || last === null) return;
    const playback = last.playback;
    const cursor = playback.cursor;
    container.hidden = last.visible === false || (cursor === null && dock === null);
    const frameIndex = cursor?.frame_index ?? 0;
    const finalFrameIndex = cursor?.final_frame_index ?? marks?.finalFrameIndex ?? 0;
    const live = playback.playing ? "off" : "polite";
    readout.textContent = scrubberReadout(playback, { tickForFrameIndex: tickOf });
    readout.setAttribute("aria-live", live);
    const showBeat = on("beat_label", true);
    beatLabel.hidden = !showBeat;
    beatLabel.textContent = showBeat
      ? scrubberBeatLabel(playback, pausedBeat(playback, frameIndex))
      : "";
    beatLabel.setAttribute("aria-live", live);
    trackHost.dataset.marks = loadState;
    renderTrack(trackFrameFor(frameIndex), finalFrameIndex);
    if (!playback.playing && !playback.requestPending) {
      if (playback.pauseReason === "user_pause" && on("freeze_on_pause", true)) {
        for (const animation of animations) animation.pause();
      } else {
        cancelAnimations();
      }
    }
  };

  /**
   * Return the frame under a pointer event on the track, or null when no
   * cursor is installed or the replay has a single frame.
   *
   * @param {Event} event
   * @returns {number | null}
   */
  const frameAtPointer = (event) => {
    const svg = trackHost.firstElementChild;
    const cursor = playbackState().cursor;
    if (!svg || cursor === null || cursor.final_frame_index === 0) return null;
    const box = svg.getBoundingClientRect();
    const perTick = (box.width - 2 * SCRUBBER_GUTTER) / cursor.final_frame_index;
    const x = /** @type {MouseEvent} */ (event).clientX - box.left;
    return Math.max(
      0,
      Math.min(cursor.final_frame_index, Math.round((x - SCRUBBER_GUTTER) / perTick)),
    );
  };

  // A press on the track seeks: a plain click goes to the tick under the
  // pointer, and dragging scrubs, as the Tick slider did (each new frame under
  // the pointer is one seek press; the navigator keeps one request in flight).
  // The thumb follows the pointer while the track is pressed (a local preview;
  // the arena follows through the seeks). The seek of a plain click is sent on
  // pointerup, not on "click": the press redraws the track under the pointer,
  // and a browser sends no click when the node under the press was replaced.
  trackHost.addEventListener("pointerdown", (event) => {
    const pointer = /** @type {PointerEvent} */ (event);
    if (pointer.button !== 0) return;
    dragPointer = pointer.pointerId;
    trackHost.setPointerCapture?.(pointer.pointerId);
    dragFrame = frameAtPointer(event);
    render();
  });
  trackHost.addEventListener("pointermove", (event) => {
    const pointer = /** @type {PointerEvent} */ (event);
    if (dragPointer === null || pointer.pointerId !== dragPointer) return;
    const frameIndex = frameAtPointer(event);
    if (frameIndex === null || frameIndex === navigator.baseFrame()) return;
    dragFrame = frameIndex;
    navigator.seek(frameIndex);
    render();
  });
  trackHost.addEventListener("pointerup", (event) => {
    const pointer = /** @type {PointerEvent} */ (event);
    if (dragPointer === null || pointer.pointerId !== dragPointer) return;
    const frameIndex = frameAtPointer(event) ?? dragFrame;
    dragPointer = null;
    dragFrame = null;
    // A plain click, or a release on a frame the last move did not send yet.
    if (frameIndex !== null && frameIndex !== navigator.baseFrame()) {
      navigator.seek(frameIndex);
    }
    // The thumb now shows the seek's target, so it never steps back to the
    // old cursor while the seek is in flight.
    render();
  });
  trackHost.addEventListener("pointercancel", () => {
    dragPointer = null;
    dragFrame = null;
    render();
  });
  // The native slider stays the keyboard and screen-reader control but is
  // visually hidden, so the track shows its focus ring.
  // rune-scrubber.css draws the ring from data-slider-focus.
  dock?.slider?.addEventListener("focus", () => {
    trackHost.dataset.sliderFocus = "true";
  });
  dock?.slider?.addEventListener("blur", () => {
    delete trackHost.dataset.sliderFocus;
  });

  /**
   * Close the Export menu when a press lands outside the Export control.
   *
   * @param {Event} event
   */
  const onOutsidePress = (event) => {
    const target = /** @type {Node | null} */ (event.target);
    if (target === null || !exportControl.contains(target)) closeExportMenu();
  };

  /**
   * Close the Export menu (when open) and stop listening for outside presses.
   *
   * @param {{returnFocus?: boolean}} [closeOptions] returnFocus true moves focus
   *   back to the Export button (Escape and a press of a menu button do this).
   */
  const closeExportMenu = (closeOptions = {}) => {
    if (exportMenu.hidden) return;
    exportMenu.hidden = true;
    exportButton.setAttribute("aria-expanded", "false");
    doc.removeEventListener("pointerdown", onOutsidePress, true);
    if (closeOptions.returnFocus === true) exportButton.focus?.();
  };

  /** Open the Export menu and start listening for presses outside it. */
  const openExportMenu = () => {
    exportMenu.hidden = false;
    exportButton.setAttribute("aria-expanded", "true");
    doc.addEventListener("pointerdown", onOutsidePress, true);
  };

  exportButton.addEventListener("click", () => {
    if (exportMenu.hidden) openExportMenu();
    else closeExportMenu();
  });
  exportControl.addEventListener("keydown", (event) => {
    if (/** @type {KeyboardEvent} */ (event).key !== "Escape" || exportMenu.hidden) {
      return;
    }
    event.stopPropagation();
    closeExportMenu({ returnFocus: true });
  });
  exportControl.addEventListener("focusout", (event) => {
    const next = /** @type {Node | null} */ (
      /** @type {FocusEvent} */ (event).relatedTarget
    );
    if (next !== null && !exportControl.contains(next)) closeExportMenu();
  });
  // A press of a menu button runs its own handler first (target phase), then
  // this closes the menu and returns focus to Export.
  exportMenu.addEventListener("click", (event) => {
    const target = /** @type {Element | null} */ (event.target);
    if (target?.closest?.("button")) closeExportMenu({ returnFocus: true });
  });

  /** @type {ResizeObserver | null} */
  let resizeObserver = null;
  const ResizeObserverClass = /** @type {any} */ (globalThis).ResizeObserver;
  if (typeof ResizeObserverClass === "function") {
    resizeObserver = new ResizeObserverClass(() => {
      if (!animations.some((animation) => animation.playState === "running")) render();
    });
    resizeObserver?.observe(trackHost);
  }

  return Object.freeze({
    root: container,
    docked: dock !== null,
    exportButton,
    navigator,
    /**
     * Take the latest playback snapshot, view, filters and replay, then redraw.
     * A new replay key drops
     * the cached marks of the old replay; a new replay or view starts loading
     * its marks. Does nothing after dispose().
     *
     * @param {ScrubberUpdate} input
     */
    update(input) {
      if (disposed) return;
      last = input;
      const replayKey = input.replayKey ?? null;
      if (replayKey !== cachedReplayKey) {
        cachedReplayKey = replayKey;
        for (const key of [...marksCache.keys()]) {
          if (!key.startsWith(`${replayKey ?? ""}|`)) marksCache.delete(key);
        }
      }
      requestMarks();
      render();
    },
    /**
     * Hold the current tick's marks dim until their beats while an animated
     * tick plays: runes until beats.impactMs, Fallen marks until beats.fallMs
     * (the Fall beat, when the arena and score show the fall), return pips until
     * beats.returnMs. With Hide Marks Ahead on they stay hidden instead of dim.
     * It also starts the tick clock that names the beat of a later mid-tick
     * pause. Call right after the choreographer starts presenting frameIndex.
     * Times are plan milliseconds at 1x; playbackRate (default 1) speeds the
     * animations the same way as the arena's. Does nothing when beats is null,
     * when the track is not showing frameIndex, or after dispose(). Earlier beat
     * animations are cancelled first.
     *
     * @param {number} frameIndex
     * @param {ScrubberBeats | null} beats
     * @param {number} [playbackRate]
     */
    presentTick(frameIndex, beats, playbackRate = 1) {
      cancelAnimations();
      tickClock = null;
      if (
        disposed ||
        beats === null ||
        currentGroups === null ||
        trackFrame !== frameIndex
      ) {
        return;
      }
      const dim = on("hide_marks_ahead", false) ? 0 : 0.3;
      const rate = Number.isFinite(playbackRate) && playbackRate > 0 ? playbackRate : 1;
      tickClock = { frameIndex, beats, rate, startedAt: now(), pausedAtMs: null };
      /**
       * Hold group at the dim opacity for `at` plan milliseconds (no forward fill).
       *
       * @param {SVGElement} group
       * @param {number | null} at
       */
      const dimUntil = (group, at) => {
        if (at === null || at <= 0 || group.childNodes.length === 0) return;
        if (typeof group.animate !== "function") return;
        const animation = group.animate([{ opacity: dim }, { opacity: dim }], {
          duration: at,
          fill: "backwards",
        });
        animation.playbackRate = rate;
        animations.push(animation);
      };
      dimUntil(currentGroups.runes, beats.impactMs);
      dimUntil(currentGroups.falls, beats.fallMs);
      dimUntil(currentGroups.returns, beats.returnMs);
    },
    /**
     * Show a notice under the readout for SCRUBBER_NOTICE_MS (4 s), then bring
     * the beat label back. It takes the beat label's line and the free space
     * beside it up to Speed; a long sentence wraps upward instead of being cut.
     * A newer notice replaces the shown one and starts the 4 s again; an empty
     * text ends the notice at once. Does nothing after dispose().
     *
     * Side effects: writes the notice line (role="status", aria-live polite, so
     * screen readers announce it), sets data-notice on the readout (which hides
     * the beat label's words but keeps its line) and starts one timer.
     *
     * @param {string} text The whole sentence, for example "Exported tick-58.png.".
     * @param {string} [level] "info" (default), "success", "warning" or
     *   "error"; warnings and errors show in the BattleClient alert color.
     *   Any other value shows as "info".
     */
    showNotice(text, level = "info") {
      if (disposed) return;
      if (noticeTimer !== null) clearTimeout(noticeTimer);
      noticeTimer = null;
      const sentence = typeof text === "string" ? text.trim() : "";
      if (sentence === "") {
        noticeLine.hidden = true;
        noticeLine.textContent = "";
        delete readoutBox.dataset.notice;
        return;
      }
      noticeLine.dataset.level = ["success", "warning", "error"].includes(level)
        ? level
        : "info";
      noticeLine.textContent = sentence;
      noticeLine.hidden = false;
      readoutBox.dataset.notice = "on";
      noticeTimer = setTimeout(() => {
        noticeTimer = null;
        noticeLine.hidden = true;
        noticeLine.textContent = "";
        delete readoutBox.dataset.notice;
      }, SCRUBBER_NOTICE_MS);
    },
    /**
     * Stop the view for good: cancel beat animations, drop a queued press,
     * close the Export menu, end a shown notice and its timer, stop observing
     * size and ignore any marks still loading. The DOM (with any moved
     * transport) stays.
     */
    dispose() {
      if (noticeTimer !== null) clearTimeout(noticeTimer);
      noticeTimer = null;
      disposed = true;
      cancelAnimations();
      closeExportMenu();
      navigator.dispose();
      resizeObserver?.disconnect();
      loadGeneration += 1;
    },
  });
}
