/**
 * @file The Saga Chronicle: the Replay Viewer's fight chapters and tick ribbon.
 *
 * What it owns: the Replay Viewer's one transport dock. Row 0 holds the chapter
 * caption ("Fight 9 Of 17 · Ticks 118–122", "Fight 9 · Starts In 2 Ticks",
 * "Between Fights" or a Fights Only Chapter Card "Skipped Ticks 100–102 ·
 * Fight 8"), the camera switch and the beat and tick readout. Row 1 is the
 * 56 px Chronicle Ribbon (mirrored Team A and Team B lanes with Fallen bars,
 * Ultimate runes and return pips, the groove with one lit chapter per fight,
 * bookmark pins, the dim veil over the ticks ahead and the playhead); it
 * replaces the plain Tick slider: click or drag to seek. Row 2 is one button
 * row. The Chronicle makes five buttons: Previous Fight, Step Play, Next Fight,
 * Fights Only (off by default) and Bookmark (with its list of pins). On the
 * Replay Viewer page it moves the page's transport buttons, Speed, artifact
 * actions and camera switch into these rows (moved, never copied; their
 * owners keep their IDs and handlers) and visually hides the old slider,
 * "Tick 110 / 245" readout and status line, which stay for keyboard and
 * screen-reader use. It also owns the dock styles in src/styles/chronicle.css.
 *
 * Who calls it: main.js creates one view with createChronicle(), passes every
 * ReplayPlaybackController snapshot to view.update(), and may call
 * view.presentTick() when an animated tick starts so the current tick's marks
 * light at their beats (never before the arena shows them). The pure helpers
 * (normalizeTimelineMarks, chapterCaption, chronicleRibbonRecords,
 * fightsOnlyCut and the bookmark helpers) are exported for tests and other
 * callers.
 *
 * Keys: none. N (Step Play) and every other transport key belong to
 * replay-controls.js through the one command set. Unapproved keys (H, [ ], { },
 * M, < >, Home, End and /) are deliberately not bound here.
 *
 * Buttons stay live during play: a press pauses first, then acts. At most one
 * replay request is in flight; a press made while one is in flight is kept as
 * the one pending press, and a newer press replaces it (ChronicleNavigator).
 *
 * Agent POV boundary: every mark and chapter comes from the host's per-audience
 * timeline marks (GET /api/replay/timeline-marks). In Agent POV that payload
 * holds only what the recipient's team saw, and the fights are built from those
 * marks alone ("Fights As Seen By Agent 3"). This module reads no frame, scene,
 * researcher space or match summary. When the audience changes it drops the old
 * marks at once and shows a plain groove until the new audience's marks arrive,
 * and it refuses a payload whose audience or recipient does not match the one it
 * asked for. Fights Only cuts use the same per-audience fights, so in Agent POV
 * they never reveal a fight the team did not see. Oracle -> Agent POV -> Oracle therefore gives exactly a fresh render.
 * Bookmarks store only frame indices and notes, per replay, in this browser.
 *
 * Nothing here is ever part of the exported battlefield PNG.
 */
import { canonicalAgentIdentity, teamLabelFromId } from "./agent-identity.js";
import { classAbilityNameV1 } from "./class-documentation.js";
import { registerTooltipOwner } from "./tooltip.js";
import { isVisualFilterEnabled } from "./visual-filters.js";

const SVG_NS = "http://www.w3.org/2000/svg";

/** Height of the Chronicle Ribbon in CSS pixels. */
export const CHRONICLE_RIBBON_HEIGHT = 56;

/** Left and right gutter of the ribbon in CSS pixels. */
export const CHRONICLE_GUTTER = 16;

/**
 * Vertical layout of the ribbon in CSS pixels, measured from its top (spec
 * "Chronicle Ribbon Lanes"). Team A sits above the groove and Team B below it,
 * so the team reads from the lane even in greyscale.
 */
export const CHRONICLE_LANES = Object.freeze({
  pipA: Object.freeze({ y: 1, height: 2 }),
  fallA: Object.freeze({ y: 5, height: 9 }),
  runeA: 19,
  groove: Object.freeze({ y: 24, height: 6 }),
  runeB: 35,
  fallB: Object.freeze({ y: 40, height: 9 }),
  pipB: Object.freeze({ y: 51, height: 2 }),
});

/** Ticks before a fight that the caption calls a lead-in ("Starts In 2 Ticks"). */
export const CHRONICLE_LEAD_IN_TICKS = 2;

/** Most bookmarks kept for one replay. */
export const CHRONICLE_BOOKMARK_LIMIT = 50;

/** Longest bookmark note, in characters. */
export const CHRONICLE_BOOKMARK_NOTE_LIMIT = 60;

/** localStorage key prefix; the replay key follows it. */
export const CHRONICLE_BOOKMARK_STORAGE_PREFIX = "marl-bgs.chronicle.bookmarks.v1:";

const RUNE_RADIUS = 2.5;
const RUNE_SPREAD = 5.5;
const RUNE_SPREAD_MIN_TICK_WIDTH = 8;
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
 * }>} ChronicleAgentMark
 * @typedef {Readonly<ChronicleAgentMark & {homeFall: boolean}>} ChronicleFallMark
 * @typedef {Readonly<ChronicleAgentMark & {damaging: boolean}>} ChronicleUltimateMark
 * @typedef {Readonly<{
 *   frameIndex: number,
 *   falls: ReadonlyArray<ChronicleFallMark>,
 *   ultimates: ReadonlyArray<ChronicleUltimateMark>,
 *   returns: ReadonlyArray<ChronicleAgentMark>,
 * }>} ChronicleTickMarks
 * @typedef {Readonly<{
 *   fightNumber: number,
 *   firstFrameIndex: number,
 *   lastFrameIndex: number,
 *   ultimateCount: number,
 *   fallenCount: number,
 * }>} ChronicleFight
 * @typedef {Readonly<{
 *   audience: "oracle" | "agent_pov",
 *   recipientPublicAgentId: string | null,
 *   finalFrameIndex: number,
 *   ticks: ReadonlyArray<ChronicleTickMarks>,
 *   fights: ReadonlyArray<ChronicleFight>,
 * }>} ChronicleMarks
 * @typedef {Readonly<{kind: "oracle"} | {kind: "agent_pov", recipientPublicAgentId: string | null}>} ChronicleAudience
 * @typedef {Readonly<{frameIndex: number, note: string}>} ChronicleBookmark
 * @typedef {Readonly<{impactMs: number | null, fallMs: number | null, returnMs: number | null}>} ChronicleBeats
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
 * }>} ChroniclePlayback
 * @typedef {{
 *   snapshot: () => ChroniclePlayback,
 *   seek: (frameIndex: number) => Promise<boolean>,
 *   play: () => unknown,
 *   pause: (reason?: string) => unknown,
 *   playOneTick: () => boolean,
 * }} ChronicleController
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
 * Agent POV, null in Oracle View), final_frame_index (integer, 0 or more), ticks
 * (only ticks with at least one mark; frame_index 1 or more, strictly
 * increasing; falls, ultimates and returns lists) and fights (fight_number from
 * 1 counting up, first_frame_index <= last_frame_index, ultimate_count and
 * fallen_count 0 or more). Frame indices above final_frame_index are refused.
 *
 * @param {unknown} raw Parsed payload.
 * @returns {ChronicleMarks} A new deeply frozen record; raw is not changed.
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
    "fights",
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
  let previousLast = 0;
  const fights = requireArray(record.fights, "fights").map((rawFight, index) => {
    const fight = requireRecord(rawFight, `fights[${index}]`);
    for (const key of Object.keys(fight)) {
      if (
        ![
          "fight_number",
          "first_frame_index",
          "last_frame_index",
          "ultimate_count",
          "fallen_count",
        ].includes(key)
      ) {
        throw new TypeError(`fights[${index}] has unknown field ${key}.`);
      }
    }
    const fightNumber = requireInteger(
      fight.fight_number,
      1,
      `fights[${index}].fight_number`,
    );
    const first = requireInteger(fight.first_frame_index, 1, `fights[${index}].first`);
    const last = requireInteger(fight.last_frame_index, 1, `fights[${index}].last`);
    if (
      fightNumber !== index + 1 ||
      first > last ||
      first <= previousLast ||
      last > finalFrameIndex
    ) {
      throw new TypeError(`fights[${index}] is out of order or range.`);
    }
    previousLast = last;
    return Object.freeze({
      fightNumber,
      firstFrameIndex: first,
      lastFrameIndex: last,
      ultimateCount: requireInteger(
        fight.ultimate_count,
        0,
        `fights[${index}].ultimates`,
      ),
      fallenCount: requireInteger(fight.fallen_count, 0, `fights[${index}].fallen`),
    });
  });
  return Object.freeze({
    audience: /** @type {"oracle" | "agent_pov"} */ (record.audience),
    recipientPublicAgentId: /** @type {string | null} */ (recipient),
    finalFrameIndex,
    ticks: Object.freeze(ticks),
    fights: Object.freeze(fights),
  });
}

/**
 * Return the cache key of one audience: "oracle" or "agent_pov:<public ID>".
 * An Agent POV audience with no recipient yet gives "agent_pov:".
 *
 * @param {ChronicleAudience} audience
 * @returns {string}
 */
export function chronicleAudienceKey(audience) {
  return audience.kind === "oracle"
    ? "oracle"
    : `agent_pov:${audience.recipientPublicAgentId ?? ""}`;
}

/**
 * Return whether marks were built for audience: same kind and, in Agent POV,
 * the same recipient (a recipient of null in audience accepts any recipient).
 * The view uses this to refuse a payload that arrives after the view changed.
 *
 * @param {ChronicleMarks} marks
 * @param {ChronicleAudience} audience
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
export function chronicleAgentLabel(publicAgentId) {
  const match = /^agent-([0-9])$/u.exec(publicAgentId);
  return `Agent ${match === null ? publicAgentId : match[1]}`;
}

/**
 * Return the fight whose first and last frames include frameIndex, or null.
 *
 * @param {ReadonlyArray<ChronicleFight>} fights In time order.
 * @param {number} frameIndex Current frame index.
 * @returns {ChronicleFight | null}
 */
export function fightAtFrame(fights, frameIndex) {
  return (
    fights.find(
      (fight) =>
        frameIndex >= fight.firstFrameIndex && frameIndex <= fight.lastFrameIndex,
    ) ?? null
  );
}

/**
 * Return the first fight that starts 1 to leadTicks frames after frameIndex,
 * with how many ticks remain, or null when no fight starts that soon.
 *
 * @param {ReadonlyArray<ChronicleFight>} fights In time order.
 * @param {number} frameIndex Current frame index.
 * @param {number} [leadTicks] Lead-in length in ticks; default 2.
 * @returns {{fight: ChronicleFight, ticksAway: number} | null}
 */
export function fightLeadIn(fights, frameIndex, leadTicks = CHRONICLE_LEAD_IN_TICKS) {
  const fight = fights.find(
    (candidate) =>
      candidate.firstFrameIndex > frameIndex &&
      candidate.firstFrameIndex - frameIndex <= leadTicks,
  );
  return fight === undefined
    ? null
    : { fight, ticksAway: fight.firstFrameIndex - frameIndex };
}

/**
 * Return the first frame of the latest fight that starts before frameIndex, or
 * null when there is none. Inside a fight (after its first tick) this is that
 * fight's own start. Previous Fight lands here, on the fight's first opening
 * tick, so its paused still shows the cause.
 *
 * @param {ReadonlyArray<ChronicleFight>} fights In time order.
 * @param {number} frameIndex
 * @returns {number | null}
 */
export function previousFightStart(fights, frameIndex) {
  let found = null;
  for (const fight of fights) {
    if (fight.firstFrameIndex < frameIndex) found = fight.firstFrameIndex;
  }
  return found;
}

/**
 * Return the first frame of the earliest fight that starts after frameIndex, or
 * null when there is none. Next Fight lands here.
 *
 * @param {ReadonlyArray<ChronicleFight>} fights In time order.
 * @param {number} frameIndex
 * @returns {number | null}
 */
export function nextFightStart(fights, frameIndex) {
  return (
    fights.find((fight) => fight.firstFrameIndex > frameIndex)?.firstFrameIndex ?? null
  );
}

/**
 * Return where Fights Only cuts to when autoplay reaches frameIndex, or null
 * when it keeps playing normally.
 *
 * Autoplay keeps playing inside a fight and in the leadTicks ticks before one.
 * Anywhere else (after a fight's last tick, or before the first lead-in) it
 * cuts to leadTicks ticks before the next fight's first tick. After the last
 * fight there is nothing to cut to, so the result is null and play runs on to
 * the end. The cut is a viewing aid: it never changes the recorded data.
 *
 * @param {ReadonlyArray<ChronicleFight>} fights In time order (from the
 *   current view's marks only, so Agent POV cuts follow what the team saw).
 * @param {number} frameIndex The frame autoplay just reached.
 * @param {number} [leadTicks] Lead-in length in ticks; default 2.
 * @returns {Readonly<{targetFrameIndex: number, skippedFirst: number,
 *   skippedLast: number, fight: ChronicleFight}> | null} targetFrameIndex: the
 *   frame to seek to before playing on. skippedFirst and skippedLast: the
 *   first and last skipped frame indices (both inclusive). fight: the fight
 *   the cut leads into.
 */
export function fightsOnlyCut(fights, frameIndex, leadTicks = CHRONICLE_LEAD_IN_TICKS) {
  if (fightAtFrame(fights, frameIndex) !== null) return null;
  if (fightLeadIn(fights, frameIndex, leadTicks) !== null) return null;
  const fight = fights.find((candidate) => candidate.firstFrameIndex > frameIndex);
  if (fight === undefined) return null;
  const targetFrameIndex = Math.max(0, fight.firstFrameIndex - leadTicks);
  if (targetFrameIndex <= frameIndex) return null;
  return Object.freeze({
    targetFrameIndex,
    skippedFirst: frameIndex,
    skippedLast: targetFrameIndex - 1,
    fight,
  });
}

/**
 * Write a tick span: "Tick 118" for one tick, "Ticks 118–122" for several.
 *
 * @param {number} firstTick
 * @param {number} lastTick
 * @returns {string}
 */
function tickSpanText(firstTick, lastTick) {
  return firstTick === lastTick
    ? `Tick ${firstTick}`
    : `Ticks ${firstTick}–${lastTick}`;
}

/**
 * Return the chapter caption for the current frame.
 *
 * Inside a fight: "Fight 9 Of 17 · Ticks 118–122". In a lead-in (a fight starts
 * 1 or 2 ticks later): "Fight 9 · Starts In 2 Ticks". Otherwise "Between
 * Fights". With hideMarksAhead true the caption gives nothing away: no "Of 17",
 * no end tick ("Fight 9 · From Tick 118") and no lead-in. With marks null
 * (not loaded yet) the kind is "none" and the text is empty. CSS shows the
 * caption in capitals; the DOM keeps Title Case for screen readers.
 *
 * @param {{
 *   marks: ChronicleMarks | null,
 *   frameIndex: number,
 *   hideMarksAhead?: boolean,
 *   tickForFrameIndex?: (frameIndex: number) => number,
 * }} input tickForFrameIndex turns a frame index into the shown simulator tick;
 *   default: the frame index itself.
 * @returns {Readonly<{kind: "fight" | "lead_in" | "between" | "none", text: string, fight: ChronicleFight | null}>}
 */
export function chapterCaption(input) {
  const marks = input.marks;
  if (marks === null) return Object.freeze({ kind: "none", text: "", fight: null });
  const tickOf = input.tickForFrameIndex ?? ((frameIndex) => frameIndex);
  const fight = fightAtFrame(marks.fights, input.frameIndex);
  if (fight !== null) {
    const text = input.hideMarksAhead
      ? `Fight ${fight.fightNumber} · From Tick ${tickOf(fight.firstFrameIndex)}`
      : `Fight ${fight.fightNumber} Of ${marks.fights.length} · ${tickSpanText(
          tickOf(fight.firstFrameIndex),
          tickOf(fight.lastFrameIndex),
        )}`;
    return Object.freeze({ kind: "fight", text, fight });
  }
  const lead = input.hideMarksAhead
    ? null
    : fightLeadIn(marks.fights, input.frameIndex);
  if (lead !== null) {
    return Object.freeze({
      kind: "lead_in",
      text: `Fight ${lead.fight.fightNumber} · Starts In ${lead.ticksAway} ${
        lead.ticksAway === 1 ? "Tick" : "Ticks"
      }`,
      fight: lead.fight,
    });
  }
  return Object.freeze({ kind: "between", text: "Between Fights", fight: null });
}

/**
 * Return the readout beside the caption, for example "Settled · Tick 118 of 204".
 *
 * The beat word is "Playing" while playback runs, "Moving" while a seek is in
 * flight, "Paused" after the viewer paused, otherwise "Settled". With
 * showBeat false only "Tick 118 of 204" is returned. With no cursor: "Tick —".
 *
 * @param {ChroniclePlayback} playback
 * @param {{showBeat?: boolean, tickForFrameIndex?: (frameIndex: number) => number}} [options]
 *   showBeat defaults to true; tickForFrameIndex defaults to the frame index.
 * @returns {string}
 */
export function chronicleReadout(playback, options = {}) {
  const cursor = playback.cursor;
  if (cursor === null) return "Tick —";
  const tickOf = options.tickForFrameIndex ?? ((frameIndex) => frameIndex);
  const tick = `Tick ${tickOf(cursor.frame_index)} of ${tickOf(cursor.final_frame_index)}`;
  if (options.showBeat === false) return tick;
  const beat = playback.playing
    ? "Playing"
    : playback.requestPending
      ? "Moving"
      : playback.pauseReason === "user_pause"
        ? "Paused"
        : "Settled";
  return `${beat} · ${tick}`;
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
 * Return the SVG path text of one rune diamond (radius 2.5 px) centred on x, y.
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
 *   runes: ReadonlyArray<Readonly<{lane: "a" | "b", classToken: string, damaging: boolean, d: string}>>,
 * }>} ChronicleMarkPaths
 */

/**
 * Build the path text of the marks of the given ticks, one path per kind.
 *
 * Fallen bars are 2 px wide (a second and third fall on one tick add bars 3 px
 * to the right); return pips are 3 x 2 px (4 px tall for two or more returns);
 * runes are 5 px diamonds on the caster's team lane, grouped by class and by
 * damaging or not (damaging runes are listed last so they draw on top). When
 * ticks are at least 8 px apart, several runes on one lane spread 5.5 px apart.
 *
 * @param {ReadonlyArray<ChronicleTickMarks>} ticks
 * @param {(frameIndex: number) => number} xOf Frame index to x in CSS pixels.
 * @param {number} perTick Pixels per tick.
 * @returns {ChronicleMarkPaths}
 */
function markPaths(ticks, xOf, perTick) {
  const paths = { fallA: "", fallB: "", pipA: "", pipB: "" };
  /** @type {Map<string, {lane: "a" | "b", classToken: string, damaging: boolean, d: string}>} */
  const runes = new Map();
  for (const tick of ticks) {
    const x = xOf(tick.frameIndex);
    for (const teamId of /** @type {const} */ ([1, 2])) {
      const falls = tick.falls.filter((mark) => mark.teamId === teamId).length;
      const lane = teamId === 1 ? CHRONICLE_LANES.fallA : CHRONICLE_LANES.fallB;
      for (let j = 0; j < Math.min(falls, 3); j += 1) {
        const d = rectPath(x - 1 + j * 3, lane.y, 2, lane.height);
        if (teamId === 1) paths.fallA += d;
        else paths.fallB += d;
      }
      const returns = tick.returns.filter((mark) => mark.teamId === teamId).length;
      if (returns > 0) {
        const height = returns >= 2 ? 4 : 2;
        const y =
          teamId === 1 ? CHRONICLE_LANES.pipA.y + 2 - height : CHRONICLE_LANES.pipB.y;
        const d = rectPath(x - 1.5, Math.max(0, y), 3, height);
        if (teamId === 1) paths.pipA += d;
        else paths.pipB += d;
      }
      const lanes = tick.ultimates.filter((mark) => mark.teamId === teamId);
      lanes.forEach((mark, index) => {
        const offset =
          lanes.length > 1 && perTick >= RUNE_SPREAD_MIN_TICK_WIDTH
            ? (index - (lanes.length - 1) / 2) * RUNE_SPREAD
            : 0;
        const laneKey = teamId === 1 ? "a" : "b";
        const classToken = CLASS_TOKENS[mark.classId];
        const key = `${laneKey}|${classToken}|${mark.damaging}`;
        const entry = runes.get(key) ?? {
          lane: /** @type {"a" | "b"} */ (laneKey),
          classToken,
          damaging: mark.damaging,
          d: "",
        };
        entry.d += diamondPath(
          x + offset,
          teamId === 1 ? CHRONICLE_LANES.runeA : CHRONICLE_LANES.runeB,
        );
        runes.set(key, entry);
      });
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
 *   chapters: ReadonlyArray<Readonly<{fight: ChronicleFight, x: number, width: number, current: boolean}>>,
 *   staticMarks: ChronicleMarkPaths,
 *   currentMarks: ChronicleMarkPaths & Readonly<{hasFalls: boolean, hasRunes: boolean, hasReturns: boolean}>,
 *   veil: Readonly<{x: number, width: number}> | null,
 *   playhead: Readonly<{x: number}>,
 *   pins: ReadonlyArray<Readonly<{frameIndex: number, x: number}>>,
 *   hitTicks: ReadonlyArray<Readonly<{tick: ChronicleTickMarks, x: number, width: number}>>,
 * }>} ChronicleRibbonRecords
 */

/**
 * Turn marks and the current frame into drawing records for the ribbon. Pure:
 * no DOM, no storage, no clock.
 *
 * Frame i sits at x = 16 + i * perTick, where perTick = (width - 32) /
 * finalFrameIndex (0 when finalFrameIndex is 0). The current frame's own marks
 * go to currentMarks (drawn above the veil and lit at their beats); every other
 * shown tick goes to staticMarks. The veil starts half a tick left of the
 * playhead and covers the ticks ahead. Chapters are padded by 0.35 of a tick on
 * each side (at least 1 px, at most 4 px) so a one-tick fight stays visible.
 *
 * showMarks false (switch event_timeline_marks off) leaves only the groove,
 * chapters, pins and playhead. showChapters false (fight_chapters off) drops
 * the chapters. hideMarksAhead true drops marks, pins and chapter parts past the
 * playhead (a chapter that is still running is cut at the playhead).
 * showBookmarks false drops the pins. marks null draws a plain groove.
 *
 * @param {{
 *   marks: ChronicleMarks | null,
 *   frameIndex: number,
 *   finalFrameIndex: number,
 *   width: number,
 *   showMarks?: boolean,
 *   showChapters?: boolean,
 *   hideMarksAhead?: boolean,
 *   bookmarks?: ReadonlyArray<ChronicleBookmark>,
 *   showBookmarks?: boolean,
 * }} input width is the ribbon width in CSS pixels (at least 64 is used).
 *   showMarks, showChapters and showBookmarks default to true; hideMarksAhead to
 *   false; bookmarks to none.
 * @returns {ChronicleRibbonRecords}
 */
export function chronicleRibbonRecords(input) {
  const width = Math.max(64, Number.isFinite(input.width) ? input.width : 64);
  const finalFrameIndex = Math.max(0, input.finalFrameIndex);
  const frameIndex = Math.max(0, Math.min(finalFrameIndex, input.frameIndex));
  const x0 = CHRONICLE_GUTTER;
  const x1 = width - CHRONICLE_GUTTER;
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
    perTick,
  );
  const current = markPaths(currentTick === null ? [] : [currentTick], xOf, perTick);
  const pad = Math.max(1, Math.min(4, perTick * 0.35));
  const chapters =
    input.showChapters === false || marks === null
      ? []
      : marks.fights
          .filter((fight) => !hideAhead || fight.firstFrameIndex <= frameIndex)
          .map((fight) => {
            const last = hideAhead
              ? Math.min(fight.lastFrameIndex, frameIndex)
              : fight.lastFrameIndex;
            const x = xOf(fight.firstFrameIndex) - pad;
            return Object.freeze({
              fight,
              x: round2(x),
              width: round2(xOf(last) + pad - x),
              current:
                frameIndex >= fight.firstFrameIndex &&
                frameIndex <= fight.lastFrameIndex,
            });
          });
  const pins =
    input.showBookmarks === false
      ? []
      : (input.bookmarks ?? [])
          .filter(
            (pin) =>
              pin.frameIndex <= finalFrameIndex &&
              (!hideAhead || pin.frameIndex <= frameIndex),
          )
          .map((pin) =>
            Object.freeze({
              frameIndex: pin.frameIndex,
              x: round2(xOf(pin.frameIndex)),
            }),
          );
  const hitWidth = Math.max(perTick, 4);
  const playheadX = xOf(frameIndex);
  const veilX = Math.max(x0 - 4, playheadX - perTick / 2);
  return Object.freeze({
    width,
    height: CHRONICLE_RIBBON_HEIGHT,
    x0,
    x1,
    perTick,
    groove: Object.freeze({
      x: x0 - 4,
      y: CHRONICLE_LANES.groove.y,
      width: x1 - x0 + 8,
      height: CHRONICLE_LANES.groove.height,
    }),
    played: Object.freeze({ x: x0 - 4, width: round2(playheadX - (x0 - 4)) }),
    chapters: Object.freeze(chapters),
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
    pins: Object.freeze(pins),
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
 * @param {ChronicleAgentMark} mark
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
 * @param {ChronicleTickMarks} tick
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
 * Read the saved bookmarks of one replay. Storage errors (private windows,
 * blocked site data), missing or malformed values all give an empty list;
 * nothing throws. Entries with a bad frame index are dropped; notes are cut to
 * 60 characters; at most 50 pins are kept, in frame order.
 *
 * @param {Pick<Storage, "getItem"> | null | undefined} storage Usually localStorage.
 * @param {string | null} replayKey The replay's identity; null gives [].
 * @returns {ReadonlyArray<ChronicleBookmark>}
 */
export function readBookmarks(storage, replayKey) {
  if (!storage || replayKey === null) return Object.freeze([]);
  try {
    const raw = storage.getItem(`${CHRONICLE_BOOKMARK_STORAGE_PREFIX}${replayKey}`);
    if (raw === null) return Object.freeze([]);
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return Object.freeze([]);
    /** @type {Map<number, ChronicleBookmark>} */
    const byFrame = new Map();
    for (const entry of parsed) {
      if (
        entry !== null &&
        typeof entry === "object" &&
        Number.isSafeInteger(entry.frameIndex) &&
        entry.frameIndex >= 0
      ) {
        byFrame.set(
          entry.frameIndex,
          Object.freeze({
            frameIndex: entry.frameIndex,
            note:
              typeof entry.note === "string"
                ? entry.note.slice(0, CHRONICLE_BOOKMARK_NOTE_LIMIT)
                : "",
          }),
        );
      }
    }
    return Object.freeze(
      [...byFrame.values()]
        .sort((left, right) => left.frameIndex - right.frameIndex)
        .slice(0, CHRONICLE_BOOKMARK_LIMIT),
    );
  } catch {
    return Object.freeze([]);
  }
}

/**
 * Save the bookmarks of one replay as JSON. Return true when saved; false when
 * there is no storage or replay key or the browser refused (nothing throws).
 *
 * @param {Pick<Storage, "setItem"> | null | undefined} storage Usually localStorage.
 * @param {string | null} replayKey
 * @param {ReadonlyArray<ChronicleBookmark>} pins
 * @returns {boolean}
 */
export function writeBookmarks(storage, replayKey, pins) {
  if (!storage || replayKey === null) return false;
  try {
    storage.setItem(
      `${CHRONICLE_BOOKMARK_STORAGE_PREFIX}${replayKey}`,
      JSON.stringify(pins.slice(0, CHRONICLE_BOOKMARK_LIMIT)),
    );
    return true;
  } catch {
    return false;
  }
}

/**
 * Add a pin at frameIndex, or remove it when one is there.
 *
 * @param {ReadonlyArray<ChronicleBookmark>} pins Current pins (not changed).
 * @param {number} frameIndex Frame to pin; a nonnegative integer.
 * @returns {Readonly<{pins: ReadonlyArray<ChronicleBookmark>, change: "added" | "removed" | "full"}>}
 *   "full" means 50 pins already exist and nothing changed.
 */
export function toggleBookmark(pins, frameIndex) {
  if (pins.some((pin) => pin.frameIndex === frameIndex)) {
    return Object.freeze({
      pins: Object.freeze(pins.filter((pin) => pin.frameIndex !== frameIndex)),
      change: "removed",
    });
  }
  if (pins.length >= CHRONICLE_BOOKMARK_LIMIT) {
    return Object.freeze({ pins, change: "full" });
  }
  return Object.freeze({
    pins: Object.freeze(
      [...pins, Object.freeze({ frameIndex, note: "" })].sort(
        (left, right) => left.frameIndex - right.frameIndex,
      ),
    ),
    change: "added",
  });
}

/**
 * Return pins with the note of frameIndex replaced (cut to 60 characters).
 * pins is not changed; an unknown frame returns the same list.
 *
 * @param {ReadonlyArray<ChronicleBookmark>} pins
 * @param {number} frameIndex
 * @param {string} note
 * @returns {ReadonlyArray<ChronicleBookmark>}
 */
export function setBookmarkNote(pins, frameIndex, note) {
  return Object.freeze(
    pins.map((pin) =>
      pin.frameIndex === frameIndex
        ? Object.freeze({
            frameIndex,
            note: note.slice(0, CHRONICLE_BOOKMARK_NOTE_LIMIT),
          })
        : pin,
    ),
  );
}

/**
 * Return the plain-text list Copy List puts on the clipboard, one line per pin:
 * "Tick 118 · Rogue poison opens the fight" (no note: "Tick 118").
 *
 * @param {ReadonlyArray<ChronicleBookmark>} pins
 * @param {(frameIndex: number) => number} [tickForFrameIndex] Default: the frame index.
 * @returns {string}
 */
export function bookmarkListText(pins, tickForFrameIndex = (frameIndex) => frameIndex) {
  return pins
    .map((pin) => {
      const tick = `Tick ${tickForFrameIndex(pin.frameIndex)}`;
      return pin.note.trim().length > 0 ? `${tick} · ${pin.note.trim()}` : tick;
    })
    .join("\n");
}

/**
 * Pick the three beat times the ribbon needs from a choreography BeatSchedule
 * (choreography-plan.js): impactMs from windows.impact[0] (runes light then),
 * fallMs from fall[0] (Fallen bars light at the Fall beat, with the arena and
 * the score) and returnMs from returnBeat[0] (return pips). Times are plan
 * milliseconds at 1x. A missing schedule gives null; a missing beat gives null
 * for that beat (its marks show at once).
 *
 * @param {unknown} schedule
 * @returns {ChronicleBeats | null}
 */
export function chronicleBeatsFromSchedule(schedule) {
  if (schedule === null || typeof schedule !== "object") return null;
  const record = /** @type {Record<string, any>} */ (schedule);
  /** @param {unknown} pair */
  const start = (pair) =>
    Array.isArray(pair) && Number.isFinite(pair[0]) && pair[0] >= 0
      ? Number(pair[0])
      : null;
  return Object.freeze({
    impactMs: start(record.windows?.impact),
    fallMs: start(record.fall),
    returnMs: start(record.returnBeat),
  });
}

/**
 * The Chronicle's link to ReplayPlaybackController: it keeps buttons live during
 * play while sending at most one replay request at a time.
 *
 * A press first pauses playback. If no request is in flight it is sent at once;
 * otherwise it becomes the one pending press (a newer press replaces an older
 * one) and flush() sends it when the controller next reports no request in
 * flight. A pending press is dropped if playback starts again before it runs.
 */
export class ChronicleNavigator {
  /**
   * @param {ChronicleController} controller The replay playback controller (or
   *   a test double with the same methods). It is not owned: dispose() leaves it.
   */
  constructor(controller) {
    this.controller = controller;
    /** @type {{kind: "seek", frameIndex: number, thenPlay: boolean} | {kind: "one_tick"} | null} */
    this.pending = null;
    /** The frame the last sent seek asked for, or null. @type {number | null} */
    this.sentTarget = null;
  }

  /**
   * Return the frame the next relative press counts from: the pending seek's
   * target, else the in-flight seek's target, else the cursor's frame. Null
   * when no cursor is installed. This is how quick presses add up.
   *
   * @returns {number | null}
   */
  baseFrame() {
    const state = this.controller.snapshot();
    if (state.cursor === null) return null;
    if (this.pending?.kind === "seek") return this.pending.frameIndex;
    if (state.requestPending && this.sentTarget !== null) return this.sentTarget;
    return state.cursor.frame_index;
  }

  /**
   * Seek to frameIndex (clamped by the controller). With thenPlay true playback
   * starts after the seek lands (double-click on a chapter).
   *
   * @param {number} frameIndex Nonnegative integer frame index.
   * @param {{thenPlay?: boolean}} [options] thenPlay defaults to false.
   * @returns {"sent" | "queued" | "refused"} refused: no cursor, offline or hidden.
   */
  seek(frameIndex, options = {}) {
    return this.#press({
      kind: "seek",
      frameIndex: Math.max(0, Math.trunc(frameIndex)),
      thenPlay: options.thenPlay === true,
    });
  }

  /**
   * Play exactly one tick, then settle (the Step Play button; the same job as N).
   *
   * @returns {"sent" | "queued" | "refused"}
   */
  stepPlay() {
    return this.#press({ kind: "one_tick" });
  }

  /**
   * Send the pending press when the controller has no request in flight. Call on
   * every controller state change. Return whether a press was sent.
   *
   * @returns {boolean}
   */
  flush() {
    if (this.pending === null) return false;
    const state = this.controller.snapshot();
    if (state.playing) {
      this.pending = null;
      return false;
    }
    if (state.requestPending || !this.#available(state)) return false;
    const action = this.pending;
    this.pending = null;
    this.#run(action);
    return true;
  }

  /** Drop any pending press. No request is cancelled. */
  dispose() {
    this.pending = null;
  }

  /**
   * Return whether the controller can take navigation at all.
   *
   * @param {ChroniclePlayback} state
   * @returns {boolean}
   */
  #available(state) {
    return state.cursor !== null && state.connected && !state.hidden;
  }

  /**
   * Pause, then send or queue one press.
   *
   * @param {{kind: "seek", frameIndex: number, thenPlay: boolean} | {kind: "one_tick"}} action
   * @returns {"sent" | "queued" | "refused"}
   */
  #press(action) {
    let state = this.controller.snapshot();
    if (!this.#available(state)) return "refused";
    if (state.playing) {
      this.controller.pause("user_pause");
      state = this.controller.snapshot();
    }
    if (state.requestPending) {
      this.pending = action;
      return "queued";
    }
    this.#run(action);
    return "sent";
  }

  /**
   * Send one press to the controller. Seek failures are reported by the
   * controller's own onError; they are not rethrown here.
   *
   * @param {{kind: "seek", frameIndex: number, thenPlay: boolean} | {kind: "one_tick"}} action
   */
  #run(action) {
    if (action.kind === "one_tick") {
      this.sentTarget = null;
      this.controller.playOneTick();
      return;
    }
    this.sentTarget = action.frameIndex;
    void this.controller
      .seek(action.frameIndex)
      .then((accepted) => {
        if (accepted && action.thenPlay) this.controller.play();
      })
      .catch(() => {});
  }
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
 * chronicle__ CSS classes that give each kind its colour. Return the groups
 * made: falls, runes and returns, so each can be lit at its own beat.
 *
 * @param {Document} doc
 * @param {SVGElement} parent
 * @param {ChronicleMarkPaths} paths
 * @returns {{falls: SVGElement, runes: SVGElement, returns: SVGElement}}
 */
function appendMarkPaths(doc, parent, paths) {
  const returns = svgElement(doc, "g", { class: "chronicle__returns" });
  const falls = svgElement(doc, "g", { class: "chronicle__falls" });
  const runes = svgElement(doc, "g", { class: "chronicle__runes" });
  if (paths.pipA)
    returns.append(
      svgElement(doc, "path", {
        class: "chronicle__pip chronicle__pip--a",
        d: paths.pipA,
      }),
    );
  if (paths.pipB)
    returns.append(
      svgElement(doc, "path", {
        class: "chronicle__pip chronicle__pip--b",
        d: paths.pipB,
      }),
    );
  if (paths.fallA)
    falls.append(
      svgElement(doc, "path", {
        class: "chronicle__fall chronicle__fall--a",
        d: paths.fallA,
      }),
    );
  if (paths.fallB)
    falls.append(
      svgElement(doc, "path", {
        class: "chronicle__fall chronicle__fall--b",
        d: paths.fallB,
      }),
    );
  for (const rune of paths.runes) {
    runes.append(
      svgElement(doc, "path", {
        class: `chronicle__rune chronicle__rune--${rune.classToken} chronicle__rune--${
          rune.damaging ? "damaging" : "support"
        }`,
        "data-lane": rune.lane,
        d: rune.d,
      }),
    );
  }
  parent.append(returns, falls, runes);
  return { falls, runes, returns };
}

/**
 * Build the ribbon's SVG from records (export-safe: flat fills, strokes and
 * opacity only; colours come from CSS classes in src/styles/chronicle.css).
 * Layers, bottom to top: groove, played tint, chapters, ahead-and-behind marks,
 * veil, bookmark pins, playhead line, current-tick marks, thumb ring, then the
 * invisible hover areas (one per marked tick, one per chapter on the groove).
 *
 * @param {ChronicleRibbonRecords} records
 * @param {Document} doc
 * @returns {{svg: SVGSVGElement, current: {falls: SVGElement, runes: SVGElement, returns: SVGElement}, tickHits: SVGElement[], chapterHits: SVGElement[]}}
 *   The new root and the parts the view wires up. Nothing is attached to the page.
 */
export function buildChronicleRibbon(records, doc) {
  const svg = /** @type {SVGSVGElement} */ (
    /** @type {unknown} */ (
      svgElement(doc, "svg", {
        class: "chronicle__ribbon",
        width: records.width,
        height: records.height,
        viewBox: `0 0 ${records.width} ${records.height}`,
        role: "img",
        "aria-label":
          "Chronicle: falls, Ultimates, returns and fights on the replay timeline",
      })
    )
  );
  const groove = records.groove;
  svg.append(
    svgElement(doc, "rect", {
      class: "chronicle__groove",
      x: groove.x,
      y: groove.y,
      width: groove.width,
      height: groove.height,
      rx: 3,
    }),
    svgElement(doc, "rect", {
      class: "chronicle__played",
      x: records.played.x,
      y: groove.y,
      width: Math.max(0, records.played.width),
      height: groove.height,
      rx: 3,
    }),
  );
  for (const chapter of records.chapters) {
    svg.append(
      svgElement(doc, "rect", {
        class: `chronicle__chapter${chapter.current ? " chronicle__chapter--current" : ""}`,
        "data-fight-number": chapter.fight.fightNumber,
        x: chapter.x,
        y: groove.y,
        width: chapter.width,
        height: groove.height,
        rx: Math.min(3, chapter.width / 2),
      }),
    );
  }
  appendMarkPaths(doc, svg, records.staticMarks);
  if (records.veil !== null) {
    svg.append(
      svgElement(doc, "rect", {
        class: "chronicle__veil",
        x: records.veil.x,
        y: 0,
        width: records.veil.width,
        height: records.height,
      }),
    );
  }
  if (records.pins.length > 0) {
    svg.append(
      svgElement(doc, "path", {
        class: "chronicle__pins",
        d: records.pins
          .map((pin) => `M${pin.x - 1.5} ${groove.y - 2}h3v10h-3z`)
          .join(""),
      }),
    );
  }
  svg.append(
    svgElement(doc, "path", {
      class: "chronicle__playhead",
      d: `M${records.playhead.x} 0V${records.height}`,
    }),
  );
  const nowGroup = svgElement(doc, "g", { class: "chronicle__now" });
  svg.append(nowGroup);
  const current = appendMarkPaths(doc, nowGroup, records.currentMarks);
  svg.append(
    svgElement(doc, "circle", {
      class: "chronicle__thumb",
      cx: records.playhead.x,
      cy: groove.y + groove.height / 2,
      r: 4,
    }),
  );
  const hits = svgElement(doc, "g", { class: "chronicle__hits" });
  /** @type {SVGElement[]} */
  const tickHits = [];
  for (const hit of records.hitTicks) {
    const rect = svgElement(doc, "rect", {
      class: "chronicle__hit",
      "data-frame-index": hit.tick.frameIndex,
      x: hit.x,
      y: 0,
      width: hit.width,
      height: records.height,
    });
    tickHits.push(rect);
    hits.append(rect);
  }
  /** @type {SVGElement[]} */
  const chapterHits = [];
  for (const chapter of records.chapters) {
    const rect = svgElement(doc, "rect", {
      class: "chronicle__hit chronicle__hit--chapter",
      "data-fight-number": chapter.fight.fightNumber,
      x: chapter.x,
      y: groove.y - 3,
      width: Math.max(chapter.width, 4),
      height: groove.height + 6,
    });
    chapterHits.push(rect);
    hits.append(rect);
  }
  svg.append(hits);
  return { svg, current, tickHits, chapterHits };
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
    id: `chronicle:${id}`,
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

/** IDs of the seven replay transport buttons, in the page's own order. */
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
 * Move the page's replay transport into the Chronicle, so the Replay Viewer
 * has one dock: Row 0 the caption, the camera switch and the readout; Row 1
 * the ribbon; Row 2 one button row.
 *
 * Row 2 reuses the page's transport fieldset (#replay-timeline
 * .replay-timeline__transport, so its 32 px button and fixed 48 px Play/Pause
 * styles still apply) and holds, in order: Start, Previous Fight, −10, −1,
 * Play/Pause, Step Play, +1, +10, Next Fight, End, Fights Only, Bookmark,
 * Speed, then the replay artifact actions (Export PNG, Download Metrics CSV,
 * Episode Details). Elements are MOVED, never copied: their IDs, labels and
 * handlers (owned by replay-controls.js, camera.js and main.js) stay the same.
 * The native Tick slider, the old "Tick 110 / 245" readout and the status
 * line stay in the DOM for keyboard and screen-reader users and for their
 * owners' writes, but get the sr-only class, so the ribbon is the one visible
 * scrubber and the Chronicle readout the one visible readout.
 *
 * Side effects: moves page elements and adds the sr-only class to three of
 * them. Reads nothing about the match, so it has no Agent POV effect.
 *
 * @param {HTMLElement} container The #chronicle element.
 * @param {{
 *   captionRow: HTMLElement,
 *   readout: HTMLElement,
 *   previousFight: HTMLButtonElement,
 *   stepPlay: HTMLButtonElement,
 *   nextFight: HTMLButtonElement,
 *   fightsOnly: HTMLButtonElement,
 *   bookmark: HTMLButtonElement,
 * }} parts The Chronicle's own caption row, readout and buttons.
 * @returns {{row: HTMLElement, slider: HTMLInputElement | null} | null} row:
 *   the docked button row. slider: the hidden native Tick slider, or null.
 *   Null (and nothing moved) when container is not inside #replay-timeline or
 *   the page lacks the transport fieldset or any of its seven buttons, as in
 *   unit tests with a fake DOM.
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
  const [first, backTen, previous, playPause, next, forwardTen, end] = buttons;
  const sliderLabel = /** @type {HTMLElement | null} */ (
    row.querySelector(".replay-timeline__slider")
  );
  const quiet = [
    sliderLabel,
    byId("replay-frame-position"),
    byId("replay-transport-status"),
  ];
  for (const node of quiet) node?.classList.add("sr-only");
  const camera = byId("replay-camera-switch");
  if (camera !== null) parts.captionRow.insertBefore(camera, parts.readout);
  row.replaceChildren(
    .../** @type {HTMLElement[]} */ (
      [
        row.querySelector("legend"),
        first,
        parts.previousFight,
        backTen,
        previous,
        playPause,
        parts.stepPlay,
        next,
        forwardTen,
        parts.nextFight,
        end,
        parts.fightsOnly,
        parts.bookmark,
        row.querySelector(".replay-timeline__rate"),
        byId("replay-artifact-actions"),
        ...quiet,
      ].filter((node) => node !== null)
    ),
  );
  return {
    row,
    slider: /** @type {HTMLInputElement | null} */ (
      sliderLabel?.querySelector("input") ?? null
    ),
  };
}

/**
 * @typedef {{
 *   playback: ChroniclePlayback,
 *   audience: ChronicleAudience,
 *   filters?: unknown,
 *   replayKey?: string | null,
 *   visible?: boolean,
 * }} ChronicleUpdate
 *   playback: the latest ReplayPlaybackController snapshot. audience: Oracle
 *   View, or Agent POV with its recipient's public agent ID (null while not yet
 *   known). filters: the visual filter state (null or absent: registry
 *   defaults). replayKey: the loaded replay's identity (artifact ID or digest;
 *   null: bookmarks off). visible: false hides the Chronicle (default true).
 * @typedef {{
 *   root: HTMLElement,
 *   buttons: Readonly<{previousFight: HTMLButtonElement, stepPlay: HTMLButtonElement, nextFight: HTMLButtonElement, fightsOnly: HTMLButtonElement, bookmark: HTMLButtonElement}>,
 *   docked: boolean,
 *   navigator: ChronicleNavigator,
 *   update: (input: ChronicleUpdate) => void,
 *   presentTick: (frameIndex: number, beats: ChronicleBeats | null, playbackRate?: number) => void,
 *   dispose: () => void,
 * }} ChronicleView
 *   root: the container. buttons: the five Chronicle buttons (Previous Fight,
 *   Step Play, Next Fight, Fights Only and Bookmark). docked: true when the
 *   page's replay transport was moved into the Chronicle (one dock, see
 *   dockReplayTransport); false on a page without it, where the buttons sit in
 *   the Chronicle's own row. navigator: the shared one-request-in-flight queue. update, presentTick and
 *   dispose are documented where createChronicle defines them.
 */

/**
 * Fill container (the page's empty #chronicle slot) with the Saga Chronicle and
 * return its view.
 *
 * On the Replay Viewer page (container inside #replay-timeline with the
 * transport fieldset) the page's transport is moved in, so the dock is one
 * block: Row 0 the chapter caption, the camera switch and the readout; Row 1
 * the 56 px ribbon, which replaces the plain Tick slider (click or drag to
 * seek); Row 2 one button row (see dockReplayTransport). A docked view is shown
 * at once and stays shown while no cursor exists, so the transport never
 * disappears. Without the transport (unit tests) the view starts hidden and
 * update() shows it once a cursor exists. visible: false always hides it.
 *
 * Fights Only (off by default, kept for the life of the view): while it is on
 * and autoplay reaches a tick outside a fight and its 2-tick lead-in, the view
 * seeks to 2 ticks before the next fight and plays on (fightsOnlyCut), and the
 * caption shows a Chapter Card such as "Skipped Ticks 100–102 · Fight 8" for
 * 840 ms. The fights come from the current view's marks only, so in Agent POV
 * the cuts follow what the team saw. Call update() with every ReplayPlaybackController
 * snapshot and whenever the view (Oracle or Agent POV and its recipient), the
 * visual filters or the replay change. When the audience or replay changes, the
 * old marks are dropped at once and options.loadMarks() is called; its answer is
 * used only if the audience and replay are still the same and the payload says
 * it was built for that audience. Loaded marks are kept per (replay, audience)
 * for the life of the view, so Oracle -> Agent POV -> Oracle shows the Oracle
 * marks again with no new request and no leftover Agent POV marks.
 *
 * Side effects: replaces container's children, moves the page's transport
 * elements (when present), adds click, double-click, pointer, focus, change
 * and keydown listeners on its own elements and on the moved Tick slider
 * (focus only: the ribbon shows the slider's focus ring), starts one 840 ms
 * timer per Chapter Card, observes the ribbon's size
 * with ResizeObserver when the browser has it, reads and writes localStorage
 * through options.storage (errors ignored), writes the clipboard on Copy List,
 * and calls controller seek, play, pause and playOneTick. Web Animations made
 * by presentTick() hold the current tick's marks dim until their beats; the DOM
 * always holds the lit end state, so cancelling them shows the truth.
 *
 * @param {HTMLElement} container The #chronicle element.
 * @param {{
 *   controller: ChronicleController,
 *   loadMarks: () => Promise<unknown>,
 *   tickForFrameIndex?: (frameIndex: number) => unknown,
 *   storage?: Pick<Storage, "getItem" | "setItem"> | null,
 *   clipboard?: {writeText: (text: string) => Promise<void>} | null,
 * }} options controller: the ReplayPlaybackController. loadMarks: fetches GET
 *   /api/replay/timeline-marks for the current audience and resolves to the
 *   parsed JSON (rejects on failure). tickForFrameIndex: maps a frame to its
 *   simulator tick (a non-integer answer falls back to the frame index; default:
 *   the frame index). storage: defaults to globalThis.localStorage when the
 *   browser allows it; null keeps pins for this page only. clipboard: defaults
 *   to navigator.clipboard; null disables Copy List.
 * @returns {ChronicleView}
 * @throws {TypeError} When container has no ownerDocument, or controller or
 *   loadMarks is missing.
 */
export function createChronicle(container, options) {
  const doc = container?.ownerDocument;
  if (!doc) throw new TypeError("createChronicle needs a page element.");
  if (!options?.controller || typeof options.loadMarks !== "function") {
    throw new TypeError("createChronicle needs a controller and loadMarks.");
  }
  const controller = options.controller;
  const navigator = new ChronicleNavigator(controller);
  const storage = options.storage === undefined ? defaultStorage() : options.storage;
  const clipboard =
    options.clipboard === undefined
      ? /** @type {any} */ (globalThis.navigator?.clipboard ?? null)
      : options.clipboard;

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
   * Create one Chronicle button (disabled until the first update) with its
   * hover and focus tip.
   *
   * @param {string} id Element ID.
   * @param {string} label Visible text.
   * @param {string} title Tip title.
   * @param {string} summary Tip sentence.
   * @returns {HTMLButtonElement}
   */
  const button = (id, label, title, summary) => {
    const node = /** @type {HTMLButtonElement} */ (
      html("button", "chronicle__button", label)
    );
    node.type = "button";
    node.id = id;
    node.disabled = true;
    registerTooltipOwner(node, controlTip(id, title, summary));
    return node;
  };

  /**
   * Create one plain button inside the bookmark list (no tip; its label says it all).
   *
   * @param {string} label
   * @param {string} [ariaLabel]
   * @returns {HTMLButtonElement}
   */
  const plainButton = (label, ariaLabel) => {
    const node = /** @type {HTMLButtonElement} */ (
      html("button", "chronicle__button", label)
    );
    node.type = "button";
    if (ariaLabel !== undefined) node.setAttribute("aria-label", ariaLabel);
    return node;
  };

  const captionRow = html("div", "chronicle__caption-row");
  const caption = html("span", "chronicle__caption");
  caption.id = "chronicle-caption";
  caption.tabIndex = 0;
  const readout = html("span", "chronicle__readout");
  readout.id = "chronicle-readout";
  captionRow.append(caption, readout);
  const ribbonHost = html("div", "chronicle__ribbon-host");
  const controls = html("div", "chronicle__controls");
  const previousFight = button(
    "chronicle-previous-fight-button",
    "Previous Fight",
    "Previous Fight",
    "Goes to the first tick of the fight before this point, so you see what started it.",
  );
  const stepPlay = button(
    "chronicle-step-play-button",
    "Step Play",
    "Step Play (N)",
    "Plays one tick with its animation, then stops. Key N.",
  );
  stepPlay.setAttribute("aria-keyshortcuts", "N");
  const nextFight = button(
    "chronicle-next-fight-button",
    "Next Fight",
    "Next Fight",
    "Goes to the first tick of the next fight, so you see what started it.",
  );
  const bookmark = button(
    "chronicle-bookmark-button",
    "Bookmark",
    "Bookmark",
    "Pin ticks you want to find again. Pins stay in this browser only, up to 50 per replay.",
  );
  bookmark.setAttribute("aria-expanded", "false");
  bookmark.setAttribute("aria-controls", "chronicle-bookmarks");
  const fightsOnly = button(
    "chronicle-fights-only-button",
    "Fights Only",
    "Fights Only",
    "Skips the ticks between fights, including pokes, positioning and turtling. 41% of hits in a typical game happen between fights. Off by default.",
  );
  fightsOnly.setAttribute("aria-pressed", "false");
  const popover = html("div", "chronicle__bookmarks");
  popover.id = "chronicle-bookmarks";
  popover.setAttribute("role", "dialog");
  popover.setAttribute("aria-label", "Bookmarks");
  popover.hidden = true;
  container.classList.add("chronicle");
  const dock = dockReplayTransport(container, {
    captionRow,
    readout,
    previousFight,
    stepPlay,
    nextFight,
    fightsOnly,
    bookmark,
  });
  if (dock === null) {
    controls.append(previousFight, stepPlay, nextFight, fightsOnly, bookmark);
    container.replaceChildren(captionRow, ribbonHost, controls, popover);
  } else {
    container.replaceChildren(captionRow, ribbonHost, dock.row, popover);
    container.hidden = false;
  }

  /** @type {Map<string, ChronicleMarks>} */
  const marksCache = new Map();
  /** @type {ChronicleMarks | null} */
  let marks = null;
  let loadState = /** @type {"loading" | "failed" | "ready"} */ ("loading");
  let loadGeneration = 0;
  /** @type {string | null} */
  let loadedKey = null;
  /** @type {ChronicleUpdate | null} */
  let last = null;
  /** @type {ReadonlyArray<ChronicleBookmark>} */
  let pins = Object.freeze([]);
  /** @type {string | null | undefined} */
  let pinsReplayKey;
  /** @type {{falls: SVGElement, runes: SVGElement, returns: SVGElement} | null} */
  let currentGroups = null;
  /** @type {ReadonlyArray<unknown> | null} */
  let ribbonInputs = null;
  /** @type {number | null} */
  let ribbonFrame = null;
  /** @type {Animation[]} */
  let animations = [];
  let copyStatus = "";
  let disposed = false;
  let fightsOnlyOn = false;
  /** The Chapter Card shown in the caption after a Fights Only cut. @type {string | null} */
  let chapterCard = null;
  /** @type {ReturnType<typeof setTimeout> | null} */
  let chapterCardTimer = null;
  /** Ribbon drag state: the pointer being tracked and whether it moved. */
  let dragPointer = /** @type {number | null} */ (null);
  let dragMoved = false;

  /** Cancel every beat animation, so the ribbon shows its lit end state. */
  const cancelAnimations = () => {
    for (const animation of animations) animation.cancel();
    animations = [];
  };

  /**
   * Return the playback snapshot of the last update, or the controller's own.
   *
   * @returns {ChroniclePlayback}
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
    return id ? chronicleAgentLabel(id) : "The Chosen Agent";
  };

  /**
   * Start loading the current (replay, audience) marks unless they are already
   * shown. A cached copy is used at once. Otherwise the old marks are dropped
   * (so nothing from another view stays drawn) and loadMarks() runs; a late,
   * failed, malformed or mismatched answer is ignored or shown as "Fights
   * Unavailable". Never throws.
   */
  const requestMarks = () => {
    if (last === null) return;
    const key = `${last.replayKey ?? ""}|${chronicleAudienceKey(last.audience)}`;
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

  /** Close the bookmark list and mark the Bookmark button collapsed. */
  const closePopover = () => {
    popover.hidden = true;
    bookmark.setAttribute("aria-expanded", "false");
  };

  /**
   * Save the pins of the current replay (failures ignored) and redraw.
   *
   * @param {ReadonlyArray<ChronicleBookmark>} next
   */
  const savePins = (next) => {
    pins = next;
    writeBookmarks(storage, pinsReplayKey ?? null, pins);
    render();
  };

  /**
   * Rebuild the bookmark list: header, "Pin Tick N" (or "Unpin"), one row per
   * pin (tick, note field of up to 60 characters, Jump, Remove), Copy List and
   * a status line. DOM only; storage is written by the row actions.
   */
  const renderBookmarks = () => {
    const frameIndex = playbackState().cursor?.frame_index ?? null;
    const pinned =
      frameIndex !== null && pins.some((pin) => pin.frameIndex === frameIndex);
    const pinHere = plainButton(
      frameIndex === null
        ? "Pin This Tick"
        : `${pinned ? "Unpin" : "Pin"} Tick ${tickOf(frameIndex)}`,
    );
    pinHere.id = "chronicle-pin-button";
    pinHere.disabled = frameIndex === null || (pinsReplayKey ?? null) === null;
    pinHere.addEventListener("click", () => {
      const at = playbackState().cursor?.frame_index;
      if (at === undefined) return;
      const result = toggleBookmark(pins, at);
      copyStatus = result.change === "full" ? "50 Pins Is The Most" : "";
      savePins(result.pins);
    });
    const list = html("ul", "chronicle__bookmark-list");
    for (const pin of pins) {
      const tick = tickOf(pin.frameIndex);
      const row = html("li", "chronicle__bookmark");
      const note = /** @type {HTMLInputElement} */ (
        html("input", "chronicle__bookmark-note")
      );
      note.type = "text";
      note.maxLength = CHRONICLE_BOOKMARK_NOTE_LIMIT;
      note.value = pin.note;
      note.placeholder = "Note (Optional)";
      note.setAttribute("aria-label", `Note For Tick ${tick}`);
      note.addEventListener("change", () => {
        pins = setBookmarkNote(pins, pin.frameIndex, note.value);
        writeBookmarks(storage, pinsReplayKey ?? null, pins);
      });
      const jump = plainButton("Jump", `Jump To Tick ${tick}`);
      jump.addEventListener("click", () => {
        navigator.seek(pin.frameIndex);
      });
      const remove = plainButton("Remove", `Remove The Pin At Tick ${tick}`);
      remove.addEventListener("click", () => {
        savePins(
          Object.freeze(pins.filter((entry) => entry.frameIndex !== pin.frameIndex)),
        );
      });
      row.append(
        html("span", "chronicle__bookmark-tick", `Tick ${tick}`),
        note,
        jump,
        remove,
      );
      list.append(row);
    }
    if (pins.length === 0) {
      list.append(
        html(
          "li",
          "chronicle__bookmark chronicle__bookmark--empty",
          "No Bookmarks Yet.",
        ),
      );
    }
    const copy = plainButton("Copy List");
    copy.disabled = pins.length === 0 || clipboard === null;
    copy.addEventListener("click", () => {
      if (clipboard === null) return;
      void clipboard
        .writeText(bookmarkListText(pins, tickOf))
        .then(() => {
          copyStatus = "List Copied";
        })
        .catch(() => {
          copyStatus = "Could Not Copy";
        })
        .finally(() => {
          if (!disposed && !popover.hidden) renderBookmarks();
        });
    });
    const status = html("p", "chronicle__bookmarks-status", copyStatus);
    status.setAttribute("role", "status");
    const actions = html("div", "chronicle__bookmarks-actions");
    actions.append(pinHere, copy);
    popover.replaceChildren(
      html("p", "chronicle__bookmarks-title", "Bookmarks"),
      html("p", "chronicle__bookmarks-scope", "This Browser Only · Max 50"),
      actions,
      list,
      status,
    );
  };

  /**
   * Rebuild the ribbon SVG and its tips when anything it draws changed (frame,
   * length, width, marks, pins, switches or view). Otherwise keep the nodes, so
   * a running beat animation is not lost to a state change in the same tick.
   * A rebuild cancels the beat animations.
   *
   * @param {number} frameIndex
   * @param {number} finalFrameIndex
   */
  const renderRibbon = (frameIndex, finalFrameIndex) => {
    const width = Math.max(
      64,
      Math.round(ribbonHost.getBoundingClientRect?.().width || 640),
    );
    const recipient = recipientText();
    const inputs = [
      marks,
      pins,
      frameIndex,
      finalFrameIndex,
      width,
      on("event_timeline_marks", true),
      on("fight_chapters", true),
      on("hide_marks_ahead", false),
      on("bookmarks", true),
      recipient,
    ];
    if (
      ribbonInputs !== null &&
      inputs.every((value, index) => value === ribbonInputs?.[index])
    ) {
      return;
    }
    ribbonInputs = inputs;
    cancelAnimations();
    const records = chronicleRibbonRecords({
      marks,
      frameIndex,
      finalFrameIndex,
      width,
      showMarks: on("event_timeline_marks", true),
      showChapters: on("fight_chapters", true),
      hideMarksAhead: on("hide_marks_ahead", false),
      bookmarks: pins,
      showBookmarks: on("bookmarks", true),
    });
    const built = buildChronicleRibbon(records, doc);
    const viewRows =
      recipient === null
        ? []
        : [{ label: "View", value: `Fights As Seen By ${recipient}` }];
    built.tickHits.forEach((hit, index) => {
      const tick = records.hitTicks[index].tick;
      registerTooltipOwner(
        hit,
        controlTip(
          `tick-${tick.frameIndex}`,
          `Tick ${tickOf(tick.frameIndex)}`,
          recipient === null
            ? "What happened on this tick."
            : `What ${recipient}'s team saw on this tick.`,
          tickMarkRows(tick),
        ),
        { inspectable: false },
      );
    });
    built.chapterHits.forEach((hit, index) => {
      const fight = records.chapters[index].fight;
      registerTooltipOwner(
        hit,
        controlTip(
          `fight-${fight.fightNumber}`,
          `Fight ${fight.fightNumber} · ${tickSpanText(
            tickOf(fight.firstFrameIndex),
            tickOf(fight.lastFrameIndex),
          )}`,
          `${fight.ultimateCount} ${fight.ultimateCount === 1 ? "Ultimate" : "Ultimates"} · ${
            fight.fallenCount
          } Fallen. Double-click to play it from 2 ticks before.`,
          viewRows,
        ),
        { inspectable: false },
      );
    });
    ribbonHost.replaceChildren(built.svg);
    currentGroups = built.current;
    ribbonFrame = frameIndex;
  };

  /**
   * Draw everything from the last update: visibility, caption, readout, ribbon
   * and button states, and the bookmark list when open. A pause made by the
   * viewer holds the beat animations when Freeze On Pause is on (the arena
   * holds its moment too); any other stop cancels them, showing the settled end.
   */
  const render = () => {
    if (disposed || last === null) return;
    const playback = last.playback;
    const cursor = playback.cursor;
    container.hidden = last.visible === false || (cursor === null && dock === null);
    const showChapters = on("fight_chapters", true);
    const hideAhead = on("hide_marks_ahead", false);
    const frameIndex = cursor?.frame_index ?? 0;
    const finalFrameIndex = cursor?.final_frame_index ?? marks?.finalFrameIndex ?? 0;
    const recipient = recipientText();

    const captionValue = chapterCaption({
      marks,
      frameIndex,
      hideMarksAhead: hideAhead,
      tickForFrameIndex: tickOf,
    });
    caption.textContent =
      chapterCard !== null
        ? chapterCard
        : loadState === "loading"
          ? recipient === null
            ? "Finding Fights"
            : `Finding Fights As Seen By ${recipient}`
          : loadState === "failed"
            ? "Fights Unavailable"
            : showChapters
              ? captionValue.text
              : "";
    caption.dataset.kind =
      chapterCard !== null
        ? "card"
        : loadState === "ready"
          ? captionValue.kind
          : loadState;
    caption.setAttribute("aria-live", playback.playing ? "off" : "polite");
    registerTooltipOwner(
      caption,
      controlTip(
        "caption",
        captionValue.kind === "between" ? "Between Fights" : "Fight Chapters",
        captionValue.kind === "between"
          ? "No one fell and no Charge, Freezing Trap or Crippling Poison landed. Pokes and positioning still happen here."
          : "A fight starts with a fall or a Charge, Freezing Trap or Crippling Poison, and ends after 2 quiet ticks. A viewing aid, not a metric.",
        recipient === null
          ? []
          : [{ label: "View", value: `Fights As Seen By ${recipient}` }],
      ),
    );
    readout.textContent = chronicleReadout(playback, {
      showBeat: on("beat_label", true),
      tickForFrameIndex: tickOf,
    });
    readout.setAttribute("aria-live", playback.playing ? "off" : "polite");

    renderRibbon(frameIndex, finalFrameIndex);
    if (!playback.playing && !playback.requestPending) {
      if (playback.pauseReason === "user_pause" && on("freeze_on_pause", true)) {
        for (const animation of animations) animation.pause();
      } else {
        cancelAnimations();
      }
    }

    const available =
      cursor !== null &&
      playback.connected &&
      !playback.hidden &&
      playback.transportState !== "OFFLINE";
    const base = navigator.baseFrame() ?? frameIndex;
    const fights = marks?.fights ?? [];
    previousFight.disabled = !available || previousFightStart(fights, base) === null;
    nextFight.disabled = !available || nextFightStart(fights, base) === null;
    stepPlay.disabled =
      !available || cursor === null || cursor.frame_index >= cursor.final_frame_index;
    fightsOnly.disabled = !available;
    fightsOnly.setAttribute("aria-pressed", String(fightsOnlyOn));
    const showPins = on("bookmarks", true);
    bookmark.hidden = !showPins;
    bookmark.disabled = !available || (last.replayKey ?? null) === null;
    if (!showPins || bookmark.disabled) closePopover();
    if (!popover.hidden) renderBookmarks();
  };

  /**
   * Return the frame under a pointer event on the ribbon, or null when no
   * cursor is installed or the replay has a single frame.
   *
   * @param {Event} event
   * @returns {number | null}
   */
  const frameAtPointer = (event) => {
    const svg = ribbonHost.firstElementChild;
    const cursor = playbackState().cursor;
    if (!svg || cursor === null || cursor.final_frame_index === 0) return null;
    const box = svg.getBoundingClientRect();
    const perTick = (box.width - 2 * CHRONICLE_GUTTER) / cursor.final_frame_index;
    const x = /** @type {MouseEvent} */ (event).clientX - box.left;
    return Math.max(
      0,
      Math.min(cursor.final_frame_index, Math.round((x - CHRONICLE_GUTTER) / perTick)),
    );
  };

  previousFight.addEventListener("click", () => {
    const base = navigator.baseFrame();
    const target = base === null ? null : previousFightStart(marks?.fights ?? [], base);
    if (target !== null) navigator.seek(target);
  });
  nextFight.addEventListener("click", () => {
    const base = navigator.baseFrame();
    const target = base === null ? null : nextFightStart(marks?.fights ?? [], base);
    if (target !== null) navigator.seek(target);
  });
  stepPlay.addEventListener("click", () => {
    navigator.stepPlay();
  });
  bookmark.addEventListener("click", () => {
    if (popover.hidden) {
      popover.hidden = false;
      bookmark.setAttribute("aria-expanded", "true");
      renderBookmarks();
    } else {
      closePopover();
    }
  });
  popover.addEventListener("keydown", (event) => {
    if (/** @type {KeyboardEvent} */ (event).key === "Escape") {
      event.stopPropagation();
      closePopover();
      bookmark.focus();
    }
  });
  fightsOnly.addEventListener("click", () => {
    fightsOnlyOn = !fightsOnlyOn;
    render();
  });
  ribbonHost.addEventListener("click", (event) => {
    if (dragMoved) {
      dragMoved = false;
      return;
    }
    const frameIndex = frameAtPointer(event);
    if (frameIndex !== null) navigator.seek(frameIndex);
  });
  // Dragging on the ribbon scrubs, as the Tick slider did: each new frame under
  // the pointer is one seek press, and the navigator keeps one request in flight.
  ribbonHost.addEventListener("pointerdown", (event) => {
    const pointer = /** @type {PointerEvent} */ (event);
    if (pointer.button !== 0) return;
    dragPointer = pointer.pointerId;
    dragMoved = false;
    ribbonHost.setPointerCapture?.(pointer.pointerId);
  });
  ribbonHost.addEventListener("pointermove", (event) => {
    const pointer = /** @type {PointerEvent} */ (event);
    if (dragPointer === null || pointer.pointerId !== dragPointer) return;
    const frameIndex = frameAtPointer(event);
    if (frameIndex === null || frameIndex === navigator.baseFrame()) return;
    dragMoved = true;
    navigator.seek(frameIndex);
  });
  /** End a ribbon drag (pointer up or cancelled). */
  const endDrag = () => {
    dragPointer = null;
  };
  ribbonHost.addEventListener("pointerup", endDrag);
  ribbonHost.addEventListener("pointercancel", endDrag);
  // The native slider stays the keyboard and screen-reader control but is
  // visually hidden, so the ribbon shows its focus ring.
  // chronicle.css draws the ring from data-slider-focus.
  dock?.slider?.addEventListener("focus", () => {
    ribbonHost.dataset.sliderFocus = "true";
  });
  dock?.slider?.addEventListener("blur", () => {
    delete ribbonHost.dataset.sliderFocus;
  });

  /**
   * Show the Chapter Card for one Fights Only cut in the caption for 840 ms
   * (120 ms in, 600 ms hold, 120 ms out in the store; here a plain text swap),
   * then redraw the normal caption. A newer card replaces the old one.
   *
   * @param {string} text For example "Skipped Ticks 100–102 · Fight 8".
   */
  const showChapterCard = (text) => {
    if (chapterCardTimer !== null) clearTimeout(chapterCardTimer);
    chapterCard = text;
    chapterCardTimer = setTimeout(() => {
      chapterCard = null;
      chapterCardTimer = null;
      render();
    }, 840);
  };

  /**
   * Make the Fights Only cut when it is on, autoplay is running with no request
   * in flight and the cursor sits outside a fight and its lead-in: seek to 2
   * ticks before the next fight, keep playing and show the Chapter Card. Reads
   * only the current view's marks.
   */
  const maybeCutToFight = () => {
    if (!fightsOnlyOn || marks === null || last === null) return;
    const playback = last.playback;
    if (!playback.playing || playback.requestPending || playback.cursor === null)
      return;
    const cut = fightsOnlyCut(marks.fights, playback.cursor.frame_index);
    if (cut === null) return;
    if (navigator.seek(cut.targetFrameIndex, { thenPlay: true }) === "refused") return;
    showChapterCard(
      `Skipped ${tickSpanText(tickOf(cut.skippedFirst), tickOf(cut.skippedLast))} · Fight ${
        cut.fight.fightNumber
      }`,
    );
    render();
  };
  ribbonHost.addEventListener("dblclick", (event) => {
    const target = /** @type {Element | null} */ (event.target);
    const number = Number(target?.getAttribute?.("data-fight-number"));
    const fight = marks?.fights.find((entry) => entry.fightNumber === number);
    if (fight !== undefined) {
      navigator.seek(Math.max(0, fight.firstFrameIndex - CHRONICLE_LEAD_IN_TICKS), {
        thenPlay: true,
      });
    }
  });

  /** @type {ResizeObserver | null} */
  let resizeObserver = null;
  const ResizeObserverClass = /** @type {any} */ (globalThis).ResizeObserver;
  if (typeof ResizeObserverClass === "function") {
    resizeObserver = new ResizeObserverClass(() => {
      if (!animations.some((animation) => animation.playState === "running")) render();
    });
    resizeObserver?.observe(ribbonHost);
  }

  return Object.freeze({
    root: container,
    buttons: Object.freeze({
      previousFight,
      stepPlay,
      nextFight,
      fightsOnly,
      bookmark,
    }),
    docked: dock !== null,
    navigator,
    /**
     * Take the latest playback snapshot, view, filters and replay, then redraw
     * and send a queued press if the controller is free. A new replay key loads
     * that replay's pins; a new replay or view starts loading its marks. With
     * Fights Only on, a playing snapshot outside every fight and lead-in makes
     * the cut (a seek, then play). Does nothing after dispose().
     *
     * @param {ChronicleUpdate} input
     */
    update(input) {
      if (disposed) return;
      last = input;
      const replayKey = input.replayKey ?? null;
      if (replayKey !== pinsReplayKey) {
        pinsReplayKey = replayKey;
        pins = readBookmarks(storage, replayKey);
        copyStatus = "";
        for (const key of [...marksCache.keys()]) {
          if (!key.startsWith(`${replayKey ?? ""}|`)) marksCache.delete(key);
        }
      }
      requestMarks();
      render();
      navigator.flush();
      maybeCutToFight();
    },
    /**
     * Hold the current tick's marks dim until their beats while an animated
     * tick plays: runes until beats.impactMs, Fallen bars until beats.fallMs
     * (the Fall beat, when the arena and score show the fall), return pips until
     * beats.returnMs. With Hide Marks Ahead on they stay hidden instead of dim.
     * Call right after the choreographer starts presenting frameIndex. Times are
     * plan milliseconds at 1x; playbackRate (default 1) speeds the animations
     * the same way as the arena's. Does nothing when beats is null, when the
     * ribbon is not showing frameIndex, or after dispose(). Earlier beat
     * animations are cancelled first.
     *
     * @param {number} frameIndex
     * @param {ChronicleBeats | null} beats
     * @param {number} [playbackRate]
     */
    presentTick(frameIndex, beats, playbackRate = 1) {
      cancelAnimations();
      if (
        disposed ||
        beats === null ||
        currentGroups === null ||
        ribbonFrame !== frameIndex
      ) {
        return;
      }
      const dim = on("hide_marks_ahead", false) ? 0 : 0.3;
      const rate = Number.isFinite(playbackRate) && playbackRate > 0 ? playbackRate : 1;
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
     * Stop the view for good: cancel beat animations, drop a queued press,
     * clear a Chapter Card timer, stop observing size and ignore any marks
     * still loading. The DOM (with any moved transport) stays.
     */
    dispose() {
      disposed = true;
      if (chapterCardTimer !== null) clearTimeout(chapterCardTimer);
      chapterCardTimer = null;
      cancelAnimations();
      navigator.dispose();
      resizeObserver?.disconnect();
      loadGeneration += 1;
    },
  });
}

/**
 * Return globalThis.localStorage, or null when the browser blocks it (the
 * accessor itself can throw) or it does not exist (Node tests).
 *
 * @returns {Storage | null}
 */
function defaultStorage() {
  try {
    return /** @type {any} */ (globalThis).localStorage ?? null;
  } catch {
    return null;
  }
}
