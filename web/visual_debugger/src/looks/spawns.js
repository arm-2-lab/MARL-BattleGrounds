/**
 * @file The Spawns And Bases look (Moon Waystones): one rune circle per claimed
 * spawn pad in the `spawns` layer, and the Call when a champion returns.
 *
 * The waystone (durable, drawn every render): a rim of radius 0.56 world units
 * with its gap facing the nearer floor edge, an inner ring at 0.34 (Team B adds a
 * second at 0.28), six rune ticks and a keystone diamond, in the team colour at
 * 22% (Dormant: its champion stands), 40% (Vigil: its champion is fallen) or 60%
 * (Wake: its champion came back this tick). While Vigil Pips (owned by the Class
 * Requiems look) count the wait on the fallen body, a waiting stone stays quiet
 * (it looks Dormant); with Vigil Pips off the stone counts the wait with silver
 * pips on a ring of 0.19 units (one arc instead of pips when the wave period is
 * over 12 ticks). The detail ladder follows the screen scale (0.5-unit body):
 * from 36 px per unit everything, from 28 the ticks and keystone, from 20 one
 * inner ring and pips, below that the rim only.
 * The DOM always holds the END state. A stone that changes this tick carries
 * `data-beat-hold` attributes so the choreographer shows the earlier look until
 * the beat: Wake from the Return beat, Vigil from the Fall beat (300 ms fades).
 * With Spawn Waystones off a plain thin ring marks each pad (plan P26).
 *
 * The Call (event part, `records({part: "call", event})`): when a champion this
 * audience saw comes back, its whole rune circle blazes in Team Light (up to 90%:
 * rise 120 ms, hold to 420 ms, gone by 720 ms), Moonpool light pools on the pad
 * (22%) and the Moonward Descent (nine dashes of the Spawn Ward's twelve) comes
 * down onto the footprint, 60 to 780 ms into the Return beat; arrivals are 30 ms
 * apart. Reduced motion: one 220 ms fade in and out. A paused frame shows none of
 * it. These are browser animations described for the choreographer's animation
 * factory (fill "backwards"; END is invisible).
 *
 * Who calls it: `scene.js` (durable waystones) and the event painter for an
 * agent_respawned pulse (the Call). `visual-filters.js` reads PAINT_PARTS.
 *
 * Boundary: records reads only this audience's arena scene (`spawn_pads`,
 * `respawn_waves`, `agents`) and the caller's beat lists, which come from this
 * audience's own events. In Agent POV a pad whose champion is unseen (no
 * assigned key in the scene, or no such body in the scene) is Dormant, with no
 * pips and no champion in its tip. It never reads a researcher view or a match
 * summary. No timers or CSS animations.
 */
import {
  canonicalAgentIdentity,
  teamLabelFromId,
  tickCountText,
} from "../agent-identity.js";
import { createSymbolDescriptor, registerTooltipOwner } from "../tooltip.js";
import {
  animationSpec,
  asArray,
  finiteNumber,
  isRecord,
  paintParts,
  setNodeAttributes,
  svgNode,
} from "./look-support.js";

/**
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {import("./look-support.js").EventMount} EventMount
 * @typedef {readonly [number, number]} Pair
 */

/**
 * Store numbers for the Moon Waystones (design store `skins/spawns-and-bases.js`).
 * Names ending in `Units` are world units; times are ms; opacities 0 to 1.
 */
export const FOOTPRINT = Object.freeze({
  rimUnits: 0.56,
  innerRingUnits: 0.34,
  teamBRingUnits: 0.28,
  tickInnerUnits: 0.38,
  tickOuterUnits: 0.47,
  keystoneUnits: 0.06,
  pipRingUnits: 0.19,
  moonpoolUnits: 0.48,
  bodyUnits: 0.5,
  dormant: 0.22,
  vigil: 0.4,
  wake: 0.6,
  keystoneOpacity: 0.22,
  beatFade: 300,
  callPeak: 0.9,
  moonpoolPeak: 0.22,
  arrivalStagger: 30,
  reducedFade: 220,
});

/**
 * Node allowance per pad record: one group with the rune body, the keystone, its
 * held copy and up to 12 pips (16). No durable animations.
 */
export const BUDGET = Object.freeze({
  nodesPerRecord: 16,
  animationsPerRecord: 0,
  persistentNodesPerRecord: 0,
});

/**
 * Allowance of one Call (event part): a group, the blaze, the Moonpool and the
 * descent group with its core and glow (6 nodes) and three animations. The plan
 * must add this to an agent_respawned pulse's allowance when the Call is drawn.
 */
export const CALL_BUDGET = Object.freeze({
  nodesPerRecord: 6,
  animationsPerRecord: 3,
  persistentNodesPerRecord: 0,
});

/**
 * Spawn Waystones switches the rune circles (off: a plain ring). The Call is
 * part of the agent_respawned pulse, whose tags the Class Requiems look owns;
 * records() reads respawn_wave, resurrection_effects and soft_glow directly.
 */
export const PAINT_PARTS = paintParts([
  [{ surface: "durable", kind: "spawn_waystone" }, "spawn_waystones"],
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
 * Round to 0.001 for stable path text. Pure.
 *
 * @param {number} value
 */
function fixed(value) {
  return Number(value.toFixed(3));
}

/**
 * Return the screen point at radius r and angle (degrees, 0 = right, 90 = down)
 * from centre c. Pure.
 *
 * @param {Pair} c
 * @param {number} r
 * @param {number} degrees
 * @returns {Pair}
 */
function polar(c, r, degrees) {
  const radians = (degrees * Math.PI) / 180;
  return [c[0] + r * Math.cos(radians), c[1] + r * Math.sin(radians)];
}

/**
 * Return a clockwise arc from angle a0 to a1 (degrees) as path text. Pure.
 *
 * @param {Pair} c
 * @param {number} r
 * @param {number} a0
 * @param {number} a1
 */
function arcPath(c, r, a0, a1) {
  const p0 = polar(c, r, a0);
  const p1 = polar(c, r, a1);
  return `M${fixed(p0[0])} ${fixed(p0[1])}A${fixed(r)} ${fixed(r)} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${fixed(p1[0])} ${fixed(p1[1])}`;
}

/**
 * Return a full circle as path text. Pure.
 *
 * @param {Pair} c
 * @param {number} r
 */
function circlePath(c, r) {
  return `M${fixed(c[0] - r)} ${fixed(c[1])}a${fixed(r)} ${fixed(r)} 0 1 0 ${fixed(2 * r)} 0a${fixed(r)} ${fixed(r)} 0 1 0 ${fixed(-2 * r)} 0`;
}

/**
 * Return the detail step for a screen scale: 3 everything, 2 ticks and keystone,
 * 1 one inner ring and pips, 0 the rim only. Pure.
 *
 * @param {number} ppu Screen px per world unit.
 */
function detailStep(ppu) {
  const radius = FOOTPRINT.bodyUnits * ppu;
  return radius >= 18 ? 3 : radius >= 14 ? 2 : radius >= 10 ? 1 : 0;
}

/**
 * Return the rune circle's path data for one pad: rim (gap toward the floor
 * edge), rings, ticks and keystone. Pure.
 *
 * @param {Pair} c Screen centre.
 * @param {number} ppu
 * @param {number} gap Gap direction in degrees (0 right edge, 180 left edge).
 * @param {number} teamId
 * @param {number} step Detail step.
 */
function runeGeometry(c, ppu, gap, teamId, step) {
  /** @param {number} units */
  const length = (units) => units * ppu;
  let rings = step >= 1 ? circlePath(c, length(FOOTPRINT.innerRingUnits)) : "";
  if (step >= 3 && teamId === 2)
    rings += circlePath(c, length(FOOTPRINT.teamBRingUnits));
  let ticks = "";
  if (step >= 2) {
    for (const angle of [30, 90, 150, 210, 270, 330]) {
      const a = polar(c, length(FOOTPRINT.tickInnerUnits), -angle);
      const b = polar(c, length(FOOTPRINT.tickOuterUnits), -angle);
      ticks += `M${fixed(a[0])} ${fixed(a[1])}L${fixed(b[0])} ${fixed(b[1])}`;
    }
  }
  const h = length(FOOTPRINT.keystoneUnits);
  const keystone =
    step >= 2
      ? `M${fixed(c[0])} ${fixed(c[1] - h)}L${fixed(c[0] + h)} ${fixed(c[1])}L${fixed(c[0])} ${fixed(c[1] + h)}L${fixed(c[0] - h)} ${fixed(c[1])}Z`
      : "";
  return {
    body: arcPath(c, length(FOOTPRINT.rimUnits), gap + 40, gap + 320) + rings + ticks,
    keystone,
  };
}

/**
 * Return the pip drawing for a waiting stone: lit pips for the ticks left
 * (countdown + 1) and hollow ones for the rest of the period, stepping from 12
 * o'clock toward the field; one remaining-fraction arc when the period is over
 * 12. Pure.
 *
 * @param {Pair} c
 * @param {number} ppu
 * @param {number} side +1 when the floor edge is to the right, else -1.
 * @param {number} period Wave period in ticks (positive).
 * @param {number} left Ticks until the return (1 or more).
 */
function pipGeometry(c, ppu, side, period, left) {
  const u = ppu / 46;
  const r0 = FOOTPRINT.pipRingUnits * ppu;
  if (period > 12) {
    const fraction = Math.min(1, Math.max(0, left / period));
    return {
      lit: "",
      hollow: "",
      ring: circlePath(c, r0),
      arc:
        fraction > 0
          ? side > 0
            ? arcPath(c, r0, -90 - 359.9 * fraction, -90)
            : arcPath(c, r0, -90, -90 + 359.9 * fraction)
          : "",
      u,
    };
  }
  const gap = (2 * Math.PI * r0) / period;
  const litRadius = Math.min(2.1 * u, 0.35 * gap);
  const spentRadius = Math.min(1.6 * u, litRadius * 0.78);
  let lit = "";
  let hollow = "";
  for (let index = 0; index < period; index += 1) {
    const point = polar(c, r0, -90 - side * index * (360 / period));
    if (index < left) lit += circlePath(point, litRadius);
    else hollow += circlePath(point, spentRadius);
  }
  return { lit, hollow, ring: "", arc: "", u };
}

/**
 * Normalize a beat list to a set of presentation keys. Accepts key strings or
 * objects with presentationKey (beatSchedule's falls and returns). Pure.
 *
 * @param {unknown} list
 * @returns {Set<string>}
 */
function keySet(list) {
  return new Set(
    asArray(list)
      .map((entry) =>
        typeof entry === "string"
          ? entry
          : isRecord(entry)
            ? entry.presentationKey
            : null,
      )
      .filter((key) => typeof key === "string"),
  );
}

/**
 * Return the Call's record for an agent_respawned pulse, or [] for anything else.
 *
 * @param {Record<string, any>} input
 * @returns {ReadonlyArray<LookRecord>}
 */
function callRecords(input) {
  const event = input.event;
  if (
    !isRecord(event) ||
    event.kind !== "semantic_pulse" ||
    event.cueSemantic !== "agent_respawned" ||
    !isRecord(event.anchor) ||
    !switchOn(input.filters, "respawn_wave", true)
  ) {
    return Object.freeze([]);
  }
  const transform = input.transform;
  const ppu = transform ? transform.worldLengthToScreen(1) : 46;
  const x = finiteNumber(event.anchor.x);
  const y = finiteNumber(event.anchor.y);
  const middle = transform
    ? transform.mapBounds.left + transform.mapBounds.width / 2
    : x;
  const side = x > middle ? 1 : -1;
  const teamId = Number(event.recipientIdentity?.team_id);
  return Object.freeze([
    Object.freeze({
      kind: "waystone-call",
      x,
      y,
      ppu,
      side,
      gap: side > 0 ? 0 : 180,
      teamId: teamId === 1 || teamId === 2 ? teamId : null,
      at: finiteNumber(event.phaseStart),
      arrival: Number.isInteger(event.arrivalIndex) ? event.arrivalIndex : 0,
      resurrection: switchOn(input.filters, "resurrection_effects", true),
      glow: switchOn(input.filters, "soft_glow", true),
      step: detailStep(ppu),
    }),
  ]);
}

/**
 * Return the drawing records of this look.
 *
 * Durable waystones (input.part absent): input.scene is this audience's arena
 * scene (`spawn_pads`: team_id, team_local_slot, position [x, y] in world units,
 * configured_active, currently_alive, assigned_presentation_key; `respawn_waves`:
 * team_id, period_steps, countdown_steps; `agents`); input.transform projects;
 * input.filters is the visual filter state (null: defaults, spawn_waystones and
 * vigil_pips on); input.falls and input.returns are this tick's seen falls and
 * returns from this audience's beat schedule (presentation keys, or entries with
 * presentationKey; default none). Return one frozen record per active pad:
 * Spawns off: {kind: "spawn-pad-plain", cx, cy, r, teamId}. Otherwise {kind:
 * "waystone", teamId, slot, cx, cy, state ("dormant" | "vigil" | "wake"), seen,
 * quiet, body, keystone, opacity, keystoneOpacity, strokeWidth, pips (null or
 * {lit, hollow, ring, arc, u}), hold (null, or {kind: "fall" | "return", key,
 * fromOpacity, keystoneHeld}), hit {cx, cy, r}, tip}. A pad without a usable
 * position, team or active flag is skipped.
 *
 * The Call (input.part "call"): input.event is one admitted plan event; only a
 * semantic_pulse whose cueSemantic is "agent_respawned" with a screen anchor
 * gives a record (and only while respawn_wave is on): {kind: "waystone-call",
 * x, y, ppu, side, gap, teamId, at, arrival, resurrection, glow, step};
 * input.transform (optional) gives the scale (default 46 px per unit) and the
 * floor middle.
 *
 * Pure data: no DOM. Agent POV: only the audience's own scene and beats are read.
 *
 * @param {{
 *   part?: "call" | null,
 *   event?: Record<string, any> | null,
 *   scene?: Record<string, any> | null,
 *   transform?: LookTransform | null,
 *   audience?: "researcher" | "agent_pov" | null,
 *   filters?: Readonly<Record<string, boolean>> | null,
 *   falls?: ReadonlyArray<unknown> | null,
 *   returns?: ReadonlyArray<unknown> | null,
 * }} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  if (input.part === "call") {
    return callRecords(input);
  }
  const transform = input.transform;
  const scene = input.scene;
  if (!transform || !isRecord(scene)) {
    return Object.freeze([]);
  }
  const ppu = transform.worldLengthToScreen(1);
  const step = detailStep(ppu);
  const width = transform.worldWidth;
  const drawn = switchOn(input.filters, "spawn_waystones", true);
  const quiet = switchOn(input.filters, "vigil_pips", true);
  const fallKeys = keySet(input.falls);
  const returnKeys = keySet(input.returns);
  const agents = new Map(
    asArray(scene.agents)
      .filter((agent) => isRecord(agent) && typeof agent.presentation_key === "string")
      .map((agent) => [String(agent.presentation_key), agent]),
  );
  const waves = new Map(
    asArray(scene.respawn_waves)
      .filter(isRecord)
      .map((wave) => [Number(wave.team_id), wave]),
  );
  /** @type {LookRecord[]} */
  const result = [];
  for (const pad of asArray(scene.spawn_pads)) {
    if (!isRecord(pad) || pad.configured_active !== true) continue;
    const teamId = Number(pad.team_id);
    const position = asArray(pad.position).map((value) =>
      finiteNumber(value, Number.NaN),
    );
    if (
      (teamId !== 1 && teamId !== 2) ||
      position.length !== 2 ||
      !position.every(Number.isFinite)
    ) {
      continue;
    }
    const point = transform.worldToScreen(
      /** @type {Pair} */ (/** @type {unknown} */ (position)),
    );
    const c = /** @type {Pair} */ ([point.x, point.y]);
    if (!drawn) {
      result.push(
        Object.freeze({
          kind: "spawn-pad-plain",
          teamId,
          cx: point.x,
          cy: point.y,
          r: FOOTPRINT.rimUnits * ppu,
        }),
      );
      continue;
    }
    const key =
      typeof pad.assigned_presentation_key === "string"
        ? pad.assigned_presentation_key
        : null;
    const agent = key === null ? undefined : agents.get(key);
    const seen = agent !== undefined;
    const state = !seen
      ? "dormant"
      : returnKeys.has(/** @type {string} */ (key))
        ? "wake"
        : pad.currently_alive === false
          ? "vigil"
          : "dormant";
    const side = position[0] > width / 2 ? 1 : -1;
    const gap = side > 0 ? 0 : 180;
    const geometry = runeGeometry(c, ppu, gap, teamId, step);
    const opacityOf = /** @param {string} name */ (name) =>
      name === "wake"
        ? FOOTPRINT.wake
        : name === "vigil" && !quiet
          ? FOOTPRINT.vigil
          : FOOTPRINT.dormant;
    const keyWeight = /** @param {string} name */ (name) =>
      name === "dormant" || (quiet && name === "vigil") ? 1 : 0;
    const wave = waves.get(teamId);
    const period = Math.trunc(finiteNumber(wave?.period_steps, 0));
    const left = Math.trunc(finiteNumber(wave?.countdown_steps, 0)) + 1;
    const pips =
      state === "vigil" && !quiet && step >= 1 && period > 0
        ? Object.freeze(pipGeometry(c, ppu, side, period, left))
        : null;
    /** @type {Readonly<Record<string, any>> | null} */
    let hold = null;
    if (state === "wake") {
      hold = Object.freeze({
        kind: "return",
        key,
        fromOpacity: opacityOf("vigil") / FOOTPRINT.wake,
        keystoneHeld: keyWeight("vigil") === 1,
      });
    } else if (
      state === "vigil" &&
      !quiet &&
      fallKeys.has(/** @type {string} */ (key))
    ) {
      hold = Object.freeze({
        kind: "fall",
        key,
        fromOpacity: FOOTPRINT.dormant / FOOTPRINT.vigil,
        keystoneHeld: true,
      });
    }
    /** @type {Array<{label: string, value: string}>} */
    const facts = [{ label: "Team", value: teamLabelFromId(teamId) ?? "Unavailable" }];
    if (seen) {
      facts.push({ label: "Returns Here", value: canonicalAgentIdentity(agent).title });
      facts.push({
        label: "State",
        value:
          state === "wake"
            ? "Came back this tick"
            : state === "vigil"
              ? `Waiting, back in ${tickCountText(left) ?? "a few ticks"}`
              : "Quiet, its champion is standing",
      });
    }
    if (period > 0)
      facts.push({ label: "Wave Period", value: `Every ${tickCountText(period)}` });
    result.push(
      Object.freeze({
        kind: "waystone",
        teamId,
        slot: Number(pad.team_local_slot),
        cx: point.x,
        cy: point.y,
        state,
        seen,
        quiet,
        body: geometry.body,
        keystone: geometry.keystone,
        opacity: opacityOf(state),
        keystoneOpacity: FOOTPRINT.keystoneOpacity * keyWeight(state),
        strokeWidth: 1.25 * (ppu / 46),
        pips,
        hold,
        hit: Object.freeze({ cx: point.x, cy: point.y, r: FOOTPRINT.rimUnits * ppu }),
        tip: Object.freeze({
          facts: Object.freeze(facts.map((fact) => Object.freeze(fact))),
          note: seen ? null : "Your team cannot see this pad's champion now.",
        }),
      }),
    );
  }
  return Object.freeze(result);
}

/**
 * Write the beat-hold attributes on one node. DOM change only.
 *
 * @param {Element} node
 * @param {string} kind "fall" or "return".
 * @param {string} key Presentation key.
 * @param {"before" | "after"} side
 * @param {number} from Opacity before the beat.
 * @param {number} end END opacity.
 */
function holdNode(node, kind, key, side, from, end) {
  setNodeAttributes(node, {
    opacity: end,
    "data-beat-hold": kind,
    "data-beat-side": side,
    "data-beat-key": key,
    "data-beat-from": from,
    "data-beat-fade": FOOTPRINT.beatFade,
  });
}

/**
 * Build one waystone group. DOM only; registers the pad's Symbol Tip.
 *
 * @param {LookRecord} record
 * @param {Document} doc
 */
function buildWaystone(record, doc) {
  const group = svgNode(doc, "g", {
    class: `waystone waystone--team-${record.teamId === 1 ? "a" : "b"} waystone--${record.state}`,
    "data-team-id": record.teamId,
    "data-state": record.state,
  });
  const body = svgNode(doc, "path", { class: "waystone__rune", "aria-hidden": "true" });
  group.append(body);
  const keystone = record.keystone
    ? svgNode(doc, "path", { class: "waystone__keystone", "aria-hidden": "true" })
    : null;
  if (keystone) group.append(keystone);
  if (record.pips) {
    const pips = svgNode(doc, "g", { class: "waystone__pips", "aria-hidden": "true" });
    pips.append(
      svgNode(doc, "path", { class: "waystone__pip-lit" }),
      svgNode(doc, "path", { class: "waystone__pip-hollow" }),
      svgNode(doc, "path", { class: "waystone__pip-ring" }),
      svgNode(doc, "path", { class: "waystone__pip-arc" }),
    );
    group.append(pips);
    if (record.hold?.kind === "fall")
      holdNode(pips, "fall", record.hold.key, "after", 0, 1);
  }
  const hit = svgNode(doc, "circle", {
    class: "waystone__hit",
    role: "img",
    "aria-label": `${teamLabelFromId(record.teamId) ?? "Team"} Spawn Pad`,
  });
  group.append(hit);
  registerTooltipOwner(hit, tooltip(record));
  if (record.hold) {
    holdNode(
      body,
      record.hold.kind,
      record.hold.key,
      "after",
      record.hold.fromOpacity,
      1,
    );
    if (keystone && record.hold.keystoneHeld && record.keystoneOpacity === 0) {
      holdNode(keystone, record.hold.kind, record.hold.key, "before", 1, 0);
    }
  }
  return group;
}

/**
 * Write a record's geometry onto its node (build and update). DOM only.
 *
 * @param {Element} node
 * @param {LookRecord} record
 */
function writeGeometry(node, record) {
  if (record.kind === "spawn-pad-plain") {
    setNodeAttributes(node, { cx: record.cx, cy: record.cy, r: record.r });
    return;
  }
  if (record.kind !== "waystone") return;
  const strokeKeystone =
    record.keystoneOpacity > 0 || record.hold?.keystoneHeld === true;
  for (const child of [...node.children]) {
    const name = child.getAttribute("class") ?? "";
    if (name === "waystone__rune") {
      setNodeAttributes(child, {
        d: record.body,
        "stroke-width": record.strokeWidth,
        "stroke-opacity": record.opacity,
      });
    } else if (name === "waystone__keystone") {
      setNodeAttributes(child, {
        d: record.keystone,
        "stroke-width": record.strokeWidth,
        "stroke-opacity": strokeKeystone ? FOOTPRINT.keystoneOpacity : 0,
      });
    } else if (name === "waystone__pips" && record.pips) {
      const [lit, hollow, ring, arc] = [...child.children];
      setNodeAttributes(lit, { d: record.pips.lit || null });
      setNodeAttributes(hollow, {
        d: record.pips.hollow || null,
        "stroke-width": record.pips.u,
      });
      setNodeAttributes(ring, {
        d: record.pips.ring || null,
        "stroke-width": record.pips.u,
      });
      setNodeAttributes(arc, {
        d: record.pips.arc || null,
        "stroke-width": 1.5 * record.pips.u,
      });
    } else if (name === "waystone__hit") {
      setNodeAttributes(child, record.hit);
    }
  }
}

/**
 * Build the Call into the painter's mount: the blaze, the Moonpool and the
 * Moonward Descent, each invisible at END, with their animation descriptions
 * pushed to mount.animationSpecs (normal or reduced motion, unsettled only).
 *
 * @param {LookRecord} record
 * @param {Document} doc
 * @param {EventMount | undefined} mount
 * @returns {SVGElement}
 */
function buildCall(record, doc, mount) {
  const c = /** @type {Pair} */ ([record.x, record.y]);
  const u = record.ppu / 46;
  const team = record.teamId === 2 ? "b" : record.teamId === 1 ? "a" : "neutral";
  const group = svgNode(doc, "g", {
    class: `waystone-call waystone-call--team-${team}`,
    "aria-hidden": "true",
  });
  const geometry = runeGeometry(
    c,
    record.ppu,
    record.gap,
    record.teamId ?? 1,
    record.step,
  );
  const blaze = svgNode(doc, "path", {
    class: "waystone-call__blaze",
    d: geometry.body,
    "stroke-width": 1.25 * u,
    opacity: 0,
  });
  group.append(blaze);
  /** @type {SVGElement | null} */
  let pool = null;
  /** @type {SVGElement | null} */
  let descent = null;
  if (record.resurrection) {
    pool = svgNode(doc, "path", {
      class: "waystone-call__moonpool",
      d: circlePath(c, FOOTPRINT.moonpoolUnits * record.ppu),
      opacity: 0,
    });
    group.append(pool);
    if (record.step > 0) {
      const radius = FOOTPRINT.bodyUnits * record.ppu;
      let dashes = "";
      for (let hour = 0; hour < 12; hour += 1) {
        const angle = -90 + 30 * hour;
        const away = Math.abs(((angle - record.gap + 540) % 360) - 180);
        if (away > 31) dashes += arcPath(c, radius, angle - 9, angle + 9);
      }
      descent = svgNode(doc, "g", { class: "waystone-call__descent", opacity: 0 });
      if (record.glow) {
        descent.append(
          svgNode(doc, "path", {
            class: "waystone-call__glow",
            d: dashes,
            "stroke-width": 4.5 * u,
          }),
        );
      }
      descent.append(
        svgNode(doc, "path", {
          class: "waystone-call__dash",
          d: dashes,
          "stroke-width": 1.5 * u,
        }),
      );
      group.append(descent);
    }
  }
  const options = mount?.options;
  if (mount && options && !options.settled && options.motionMode !== "off") {
    const reduced = options.motionMode === "reduced";
    const at = record.at;
    const stagger = record.arrival * FOOTPRINT.arrivalStagger;
    const event = { eventId: `call:${record.x}:${record.y}` };
    /**
     * @param {SVGElement} element
     * @param {Keyframe[]} keyframes
     * @param {number} delay
     * @param {number} duration
     * @param {string} part
     */
    const push = (element, keyframes, delay, duration, part) =>
      mount.animationSpecs.push(
        animationSpec(
          element,
          keyframes,
          { delay, duration, fill: "backwards" },
          mount.plan,
          event,
          part,
        ),
      );
    if (!reduced) {
      push(
        blaze,
        [
          { opacity: 0, easing: "ease-out" },
          { opacity: FOOTPRINT.callPeak, offset: 120 / 720 },
          { opacity: FOOTPRINT.callPeak, offset: 420 / 720 },
          { opacity: 0 },
        ],
        at,
        720,
        "waystone-call",
      );
    }
    if (pool) {
      push(
        pool,
        reduced
          ? [
              { opacity: 0 },
              { opacity: FOOTPRINT.moonpoolPeak, offset: 0.5 },
              { opacity: 0 },
            ]
          : [
              { opacity: 0, easing: "ease-out" },
              { opacity: FOOTPRINT.moonpoolPeak, offset: 180 / 640 },
              { opacity: FOOTPRINT.moonpoolPeak, offset: 440 / 640 },
              { opacity: 0 },
            ],
        reduced ? at : at + 60 + stagger,
        reduced ? 2 * FOOTPRINT.reducedFade : 640,
        "moonpool",
      );
    }
    if (descent) {
      const start = Math.min(1.36, (0.48 - 2.25 / record.ppu) / 0.315);
      /** @param {number} scale */
      const around = (scale) =>
        `translate(${fixed(c[0])}px, ${fixed(c[1])}px) scale(${fixed(scale)}) translate(${fixed(-c[0])}px, ${fixed(-c[1])}px)`;
      push(
        descent,
        reduced
          ? [{ opacity: 0 }, { opacity: 0.7, offset: 0.5 }, { opacity: 0 }]
          : [
              { opacity: 0, transform: around(start), easing: "ease-in" },
              { opacity: 0.85, offset: 120 / 720 },
              { opacity: 0.85, transform: around(1), offset: 360 / 720 },
              { opacity: 1, offset: 360.5 / 720 },
              { opacity: 1, offset: 440 / 720 },
              { opacity: 0, transform: around(1) },
            ],
        reduced ? at : at + 60 + stagger,
        reduced ? 2 * FOOTPRINT.reducedFade : 720,
        "moonward-descent",
      );
    }
  }
  if (mount) mount.group.append(group);
  return group;
}

/**
 * Build the nodes for records, in record order.
 *
 * records come from records(); doc creates the nodes; mount (optional) is the
 * painter's EventMount for the Call. A "waystone" record gives a
 * `g.waystone` (rune path, keystone path, optional pips group and a transparent
 * `circle.waystone__hit` that carries the "Spawn Pad" Symbol Tip); stones that
 * change this tick carry data-beat-hold, data-beat-side, data-beat-key,
 * data-beat-from and data-beat-fade (300 ms). A "spawn-pad-plain" record gives a
 * `circle.spawn-pad-plain`. A "waystone-call" record gives a `g.waystone-call`
 * appended to mount.group (when given) and pushes up to three animation
 * descriptions to mount.animationSpecs (none when settled or motion is off).
 * Side effects: DOM nodes and tooltip registration only. Export-safe: flat
 * fills, strokes, opacity and transforms; no ids. Return the nodes.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @param {EventMount} [mount]
 * @returns {SVGElement[]}
 */
export function build(records, doc, mount) {
  /** @type {SVGElement[]} */
  const nodes = [];
  for (const record of records) {
    if (record.kind === "waystone") {
      const node = buildWaystone(record, doc);
      writeGeometry(node, record);
      nodes.push(node);
    } else if (record.kind === "spawn-pad-plain") {
      const node = svgNode(doc, "circle", {
        class: `spawn-pad-plain spawn-pad-plain--team-${record.teamId === 1 ? "a" : "b"}`,
        "aria-hidden": "true",
      });
      writeGeometry(node, record);
      nodes.push(node);
    } else if (record.kind === "waystone-call") {
      nodes.push(buildCall(record, doc, mount));
    }
  }
  return nodes;
}

/**
 * Move existing waystone nodes to new records (reprojection after a resize or a
 * camera change). nodes are build()'s nodes and records the new records in the
 * same order and kinds; the Call is not reprojected (it lives one tick). Beat
 * holds and tooltips are unchanged. Return undefined; DOM change only.
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
 * Return the "Spawn Pad" Symbol Tip for a waystone record, or null for any other
 * record. Facts: the team, and only for a champion this audience sees, who
 * returns here and the stone's state; the wave period. An unseen pad in Agent
 * POV says "Your team cannot see this pad's champion now." Pure.
 *
 * @param {LookRecord} record
 */
export function tooltip(record) {
  if (record?.kind !== "waystone" || !isRecord(record.tip)) {
    return null;
  }
  return createSymbolDescriptor({
    symbol: "waystone",
    id: `waystone:${record.teamId}:${record.slot}`,
    facts: record.tip.facts,
    note: record.tip.note,
  });
}
