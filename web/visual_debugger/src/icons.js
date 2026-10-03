/**
 * @file The one glyph owner for all three clients (BattleClient, DevClient
 * combat screen and Replay Viewer). Every class emblem, team crest, status,
 * aura, life-change and ability icon on bodies, panels, cards and the HUD is
 * made here, so a glyph means the same thing everywhere.
 *
 * Two styles:
 * - "sigil" (the default): the Heraldic Sigils. Filled shapes with a dark Ink
 *   keyline. Class emblems: Mage star, Warrior shield with a boss, Hunter
 *   antlers (no bow, so no facing), Rogue crossed daggers, Priest sun cross
 *   (a Priest that stands on its Class Seal shows the cross alone, because the
 *   seal is its sun: createSvgIcon's onClassSeal option).
 *   Status icons are small copies of the body marks (the Freezing Trap ice
 *   hexagon, the stun stars, the slow beads, the Serrated Arrows thread, the
 *   Crippling Poison veins, the Blessing of Freedom wings, the Burst runes);
 *   In Combat is an hourglass, Rest a dew drop, Unseen a closed eye; life
 *   changes are pip rails (no plus sign). Colors come from the CSS color
 *   tokens through `sigil-paint--<paint>` classes (src/styles/icons.css), so
 *   PNG export resolves them like any other style.
 * - "plain": today's 24-unit monoline glyphs in `currentColor`, for callers
 *   that ask for them by name. The Hunter bow has no arrowhead, so no icon
 *   shows a facing. Rest and Moon Dew draw a plain drop (`status-rest`).
 * - "outline": the plain, truthful drawing used when the Lore Emblems or
 *   Status Icons From Marks switch is off (never today's old glyphs): a class
 *   emblem is its sigil's silhouette (star, shield, antlers, crossed daggers,
 *   sun cross) as one thin line; a status, aura or life icon is every shape
 *   of its sigil as thin lines. One color (currentColor), no fills.
 *
 * The page sets which style a call without an explicit style gets
 * (setIconStyleDefaults, from those two switches), so each switch changes its
 * marks wherever they are drawn.
 *
 * Glyph keys: every old key (`class-mage`, `status-stun`, `lifecycle-applied`,
 * `combat-in-progress` and the rest) still works and draws its sigil in the
 * sigil style; the private DevClient and other modules import them. New keys
 * name a sigil directly: `emblem-<class>`, `sigil-<name>` (for example
 * `sigil-frozen`, `sigil-dazed-rogue`, `sigil-unseen`, `sigil-life-applied`).
 * statusSigilKey and modifierSigilKey turn a status or aura token id into the
 * exact sigil key with the right class color.
 *
 * Boundary: pure geometry. No image download, HTML parsing, simulator lookup
 * or game fact; callers choose accessible names, size, paint and placement.
 * The sigil geometry is generated data from the design store
 * (artifacts/visual-overhaul-20260927/skins/icons-and-emblems.js GEO), written
 * by artifacts/runefield-implementation/b2-statuses-icons/gen_sigil_data.py.
 * Export-safe SVG only: flat fills, strokes, opacity and transforms.
 */
const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

/**
 * @typedef {"circle" | "line" | "path" | "polyline" | "rect"} IconPrimitiveTag
 */

/**
 * @typedef {{
 *   tag: IconPrimitiveTag,
 *   attributes: Readonly<Record<string, string | number>>,
 * }} IconPrimitive
 */

/**
 * @typedef {{
 *   glyphKey: string,
 *   primitives: readonly Readonly<IconPrimitive>[],
 * }} IconDefinition
 */

/**
 * @typedef {"sigil" | "plain" | "outline"} IconStyle
 *   "sigil" draws the Heraldic Sigil; "plain" draws today's monoline glyph;
 *   "outline" draws the sigil's own shape as thin lines in currentColor (the
 *   switch-off look of Lore Emblems and Status Icons From Marks).
 */

/**
 * @typedef {"class" | "ready" | "host" | "ash" | "flare"} EmblemPaint
 *   How a class emblem is painted: "class" its class color; "ready" the class
 *   color with a class-colored keyline at 45% (Ultimate ready); "host" its
 *   team color (teamId 1 or 2); "ash" gray at 55% (fallen); "flare" one bright
 *   Starcore silhouette (the Mage's Burst at Impact).
 */

/**
 * @typedef {{
 *   glyphKey: string,
 *   kind: "emblem" | "glyph",
 *   name: string,
 *   classPaint: string | null,
 * }} SigilDefinition
 *   kind "emblem" names a class (name is mage, warrior, hunter, rogue or
 *   priest); kind "glyph" names a store glyph (name such as "frozen"). When
 *   classPaint is a class name, that class color is drawn in the caller's
 *   currentColor instead (used by the generic old keys such as status-stun).
 */

/**
 * Return a frozen SVG primitive with tag and attributes. tag is an allowed
 * shape name; attributes holds its SVG values. Freeze the supplied attributes
 * object itself, without copying or validating it. This module owns those
 * objects, so callers must not expect to edit one after registration.
 *
 * @param {IconPrimitiveTag} tag
 * @param {Record<string, string | number>} attributes
 * @returns {Readonly<IconPrimitive>}
 */
function primitive(tag, attributes) {
  return Object.freeze({
    tag,
    attributes: Object.freeze(attributes),
  });
}

/**
 * Return a frozen icon definition for glyphKey and its primitives. Freeze
 * the supplied primitives array itself; existing primitive objects are reused.
 * The module owns and validates the registry through its authored constants.
 *
 * @param {string} glyphKey
 * @param {IconPrimitive[]} primitives
 * @returns {Readonly<IconDefinition>}
 */
function icon(glyphKey, primitives) {
  return Object.freeze({
    glyphKey,
    primitives: Object.freeze(primitives),
  });
}

/**
 * The plain monoline glyphs (24-unit view box, currentColor). They are the
 * "plain" style and the drawing iconDefinition returns.
 *
 * @type {Readonly<Record<string, Readonly<IconDefinition>>>}
 */
const ICONS = Object.freeze({
  "class-mage": icon("class-mage", [
    primitive("path", {
      d: "M12 2.5 14.3 9.7 21.5 12l-7.2 2.3L12 21.5l-2.3-7.2L2.5 12l7.2-2.3Z",
    }),
    primitive("circle", { cx: 12, cy: 12, r: 1.8, fill: "currentColor" }),
  ]),
  "class-warrior": icon("class-warrior", [
    primitive("path", {
      d: "M12 2.5 20 5.4v5.8c0 5-3.2 8.3-8 10.3-4.8-2-8-5.3-8-10.3V5.4Z",
    }),
    primitive("path", { d: "M12 5.8v11.7M8.3 9.4h7.4" }),
  ]),
  "class-hunter": icon("class-hunter", [
    primitive("path", { d: "M6.5 3.5c7 4.6 7 12.4 0 17" }),
    primitive("line", { x1: 6.5, y1: 3.5, x2: 6.5, y2: 20.5 }),
    primitive("line", { x1: 3, y1: 12, x2: 21, y2: 12 }),
  ]),
  "class-rogue": icon("class-rogue", [
    primitive("path", { d: "m5 4 6.5 6.5-2.7 2.7L2.3 6.7Z" }),
    primitive("path", { d: "m19 4-6.5 6.5 2.7 2.7 6.5-6.5Z" }),
    primitive("path", { d: "m8.8 13.2-3.1 3.1M15.2 13.2l3.1 3.1" }),
    primitive("circle", { cx: 12, cy: 17.5, r: 2.4 }),
  ]),
  "class-priest": icon("class-priest", [
    primitive("rect", {
      x: 9,
      y: 3,
      width: 6,
      height: 18,
      fill: "currentColor",
      stroke: "none",
    }),
    primitive("rect", {
      x: 3,
      y: 9,
      width: 18,
      height: 6,
      fill: "currentColor",
      stroke: "none",
    }),
  ]),
  "team-a": icon("team-a", [
    primitive("path", {
      d: "M12 2.8 20 6v5.3c0 4.8-3.1 8-8 9.9-4.9-1.9-8-5.1-8-9.9V6Z",
      fill: "currentColor",
    }),
  ]),
  "team-b": icon("team-b", [
    primitive("path", {
      d: "M12 2.8 20 6v5.3c0 4.8-3.1 8-8 9.9-4.9-1.9-8-5.1-8-9.9V6Z",
    }),
    primitive("path", { d: "M8 12h8" }),
  ]),
  "status-stun": icon("status-stun", [
    primitive("path", { d: "m12 2 8 5v10l-8 5-8-5V7Z" }),
    primitive("path", { d: "m8.5 8.5 7 7M15.5 8.5l-7 7" }),
  ]),
  "status-slow": icon("status-slow", [
    primitive("path", {
      d: "M19.5 8.2c-2-4-7.1-5.5-10.9-3.1-4.1 2.5-4.4 8.3-.9 11.4 3 2.7 7.8 2.1 9.7-1.1 1.6-2.7.3-6.3-2.7-7.2-2.4-.7-4.9.9-5 3.4-.1 2 1.8 3.6 3.7 3.2 1.4-.3 2.3-1.8 1.8-3.2",
    }),
  ]),
  "status-anti-heal": icon("status-anti-heal", [
    primitive("path", {
      d: "M12 20.2 4.2 12.8A5.1 5.1 0 0 1 11.4 5l.6.7.6-.7a5.1 5.1 0 0 1 7.2 7.8Z",
    }),
    primitive("line", { x1: 4, y1: 20, x2: 20, y2: 4 }),
  ]),
  "status-freedom": icon("status-freedom", [
    primitive("path", { d: "M9.5 7.5 7 5a3 3 0 0 0-4.2 4.2l3 3" }),
    primitive("path", { d: "m14.5 16.5 2.5 2.5a3 3 0 0 0 4.2-4.2l-3-3" }),
    primitive("path", { d: "m8 16 8-8M5 19l3-3M16 8l3-3" }),
  ]),
  "status-burst": icon("status-burst", [
    primitive("path", {
      d: "m12 2 2 6 5.5-3.5-2.3 6.2 6.3.3-5.6 3.2 5.1 3.7-6.3-.2 1.7 6.1-4.9-4-2.4 5.8-.2-6.3-5.6 3 4-4.9-6.2.9 5.2-3.7Z",
    }),
  ]),
  "status-spawn-shield": icon("status-spawn-shield", [
    primitive("path", {
      d: "M12 2.5 19.5 6v5.3c0 4.7-2.9 7.8-7.5 10.2-4.6-2.4-7.5-5.5-7.5-10.2V6Z",
    }),
    primitive("path", {
      d: "m12 6.5 1.4 4.1 4.1 1.4-4.1 1.4-1.4 4.1-1.4-4.1L6.5 12l4.1-1.4Z",
    }),
  ]),
  "status-unseen": icon("status-unseen", [
    primitive("path", { d: "M3 12c2.5 3.2 5.6 4.8 9 4.8s6.5-1.6 9-4.8" }),
    primitive("path", { d: "m6.5 15.2-1.6 2.3M12 16.8v2.6M17.5 15.2l1.6 2.3" }),
  ]),
  "status-rest": icon("status-rest", [
    primitive("path", {
      d: "M12 3.5C9.2 8 6.5 11 6.5 14.5a5.5 5.5 0 0 0 11 0C17.5 11 14.8 8 12 3.5Z",
    }),
  ]),
  "activation-basic-damage": icon("activation-basic-damage", [
    primitive("path", { d: "M3 12h14M12 6l6 6-6 6" }),
    primitive("circle", { cx: 5, cy: 12, r: 2, fill: "currentColor" }),
  ]),
  "activation-basic-heal": icon("activation-basic-heal", [
    primitive("path", { d: "M12 4v16M4 12h16" }),
    primitive("circle", { cx: 12, cy: 12, r: 8 }),
  ]),
  "activation-holy-word": icon("activation-holy-word", [
    primitive("circle", { cx: 12, cy: 12, r: 9 }),
    primitive("path", { d: "M12 5v14M5 12h14" }),
    primitive("path", { d: "M12 2v3M12 19v3M2 12h3M19 12h3" }),
  ]),
  "activation-burst": icon("activation-burst", [
    primitive("path", {
      d: "M12 2.5 14.5 9 21 6.5 17 12l4 5.5-6.5-2.5L12 21.5 9.5 15 3 17.5 7 12 3 6.5 9.5 9Z",
    }),
    primitive("circle", { cx: 12, cy: 12, r: 2, fill: "currentColor" }),
  ]),
  "activation-charge": icon("activation-charge", [
    primitive("path", { d: "M3 12h14M12 5l7 7-7 7" }),
    primitive("path", { d: "M3 7h5M3 17h5" }),
  ]),
  "activation-trap": icon("activation-trap", [
    primitive("path", { d: "M4 20 12 4l8 16Z" }),
    primitive("circle", { cx: 12, cy: 14, r: 3.5 }),
    primitive("path", { d: "M12 10.5V20M8.5 14h7" }),
  ]),
  "activation-poison": icon("activation-poison", [
    primitive("path", { d: "M4 20 9 15" }),
    primitive("path", { d: "M6 14 10 18" }),
    primitive("path", { d: "M9 15 20 4 17 12 12 17Z" }),
    primitive("path", {
      d: "M19 12.5c1.2 1.8 1.8 2.7 1.8 3.6a1.8 1.8 0 1 1-3.6 0c0-.9.6-1.8 1.8-3.6Z",
      fill: "currentColor",
      stroke: "none",
    }),
  ]),
  "modifier-amplification": icon("modifier-amplification", [
    primitive("path", { d: "M12 21V4M5 11l7-7 7 7" }),
    primitive("path", { d: "M6 17h12" }),
  ]),
  "modifier-mitigation": icon("modifier-mitigation", [
    primitive("path", {
      d: "M12 2.8 20 6v5.3c0 4.8-3.1 8-8 9.9-4.9-1.9-8-5.1-8-9.9V6Z",
    }),
    primitive("path", { d: "M7.5 12h9" }),
  ]),
  "lifecycle-applied": icon("lifecycle-applied", [
    primitive("circle", { cx: 12, cy: 12, r: 9 }),
    primitive("path", { d: "M12 7v10M7 12h10" }),
  ]),
  "lifecycle-refreshed": icon("lifecycle-refreshed", [
    primitive("path", { d: "M19 8V3l-2.2 2.2A8 8 0 1 0 20 12" }),
    primitive("polyline", { points: "15,3 19,3 19,7" }),
  ]),
  "lifecycle-decremented": icon("lifecycle-decremented", [
    primitive("circle", { cx: 12, cy: 12, r: 9 }),
    primitive("path", { d: "M7 12h10" }),
  ]),
  "lifecycle-expired": icon("lifecycle-expired", [
    primitive("path", {
      d: "M7 3h10M7 21h10M8 3c0 5 3 5.5 4 9-1 3.5-4 4-4 9M16 3c0 5-3 5.5-4 9 1 3.5 4 4 4 9",
    }),
  ]),
  "lifecycle-trap-broken": icon("lifecycle-trap-broken", [
    primitive("rect", { x: 4, y: 4, width: 16, height: 16, rx: 2 }),
    primitive("path", { d: "m5 19 5-7-2-2 5-6-1 7 3 2-5 7 1-6Z" }),
  ]),
  "lifecycle-death-clear": icon("lifecycle-death-clear", [
    primitive("path", { d: "M5 5 19 19M19 5 5 19" }),
    primitive("path", { d: "M3 12h18" }),
  ]),
  "lifecycle-ended": icon("lifecycle-ended", [
    primitive("circle", { cx: 12, cy: 12, r: 8 }),
    primitive("path", { d: "M8.5 8.5h.1M15.5 8.5h.1M9 16c1.8-1.3 4.2-1.3 6 0" }),
  ]),
  "lifecycle-trap-broken-reapplied": icon("lifecycle-trap-broken-reapplied", [
    primitive("rect", { x: 3, y: 4, width: 14, height: 16, rx: 2 }),
    primitive("path", { d: "m4 19 5-7-2-2 5-6-1 7 3 2-5 7 1-6Z" }),
    primitive("path", { d: "M19 12v8M15 16h8" }),
  ]),
  "combat-in-progress": icon("combat-in-progress", [
    primitive("path", { d: "M4 3 15 14M13 16l3-3 4 4-3 3Z" }),
    primitive("path", { d: "M20 3 9 14M11 16l-3-3-4 4 3 3Z" }),
  ]),
  unknown: icon("unknown", [
    primitive("path", {
      d: "M8.8 8.3a3.4 3.4 0 1 1 5.8 2.4c-1.7 1.4-2.6 2-2.6 4",
    }),
    primitive("circle", { cx: 12, cy: 19, r: 1, fill: "currentColor" }),
  ]),
});

/** The five class names in class-ID order (1 Mage to 5 Priest). */
const CLASS_NAMES = Object.freeze(["mage", "warrior", "hunter", "rogue", "priest"]);

/**
 * Old glyph key -> [sigil name, kind, class color drawn in currentColor].
 * A generic old key (one key for several classes, such as status-stun) draws
 * its sigil with the class color replaced by the caller's currentColor.
 *
 * @type {Readonly<Record<string, readonly [string, "emblem" | "glyph", string | null]>>}
 */
const OLD_KEY_SIGILS = Object.freeze({
  "class-mage": ["mage", "emblem", null],
  "class-warrior": ["warrior", "emblem", null],
  "class-hunter": ["hunter", "emblem", null],
  "class-rogue": ["rogue", "emblem", null],
  "class-priest": ["priest", "emblem", null],
  "team-a": ["crest_A", "glyph", null],
  "team-b": ["crest_B", "glyph", null],
  "status-stun": ["dazed_warrior", "glyph", "warrior"],
  "status-slow": ["hobbled_warrior", "glyph", "warrior"],
  "status-anti-heal": ["withering", "glyph", null],
  "status-freedom": ["unbound", "glyph", null],
  "status-burst": ["kindled", "glyph", null],
  "status-spawn-shield": ["moonward", "glyph", null],
  "status-unseen": ["unseen", "glyph", null],
  "activation-basic-damage": ["seal_basic", "glyph", null],
  "activation-basic-heal": ["ab_lesser_heal", "glyph", null],
  "activation-holy-word": ["ab_salvation", "glyph", null],
  "activation-burst": ["ab_burst", "glyph", null],
  "activation-charge": ["ab_charge", "glyph", null],
  "activation-trap": ["ab_freezing_trap", "glyph", null],
  "activation-poison": ["ab_crippling_poison", "glyph", null],
  "modifier-amplification": ["empowerment", "glyph", null],
  "modifier-mitigation": ["barrier", "glyph", null],
  "lifecycle-applied": ["life_applied", "glyph", null],
  "lifecycle-refreshed": ["life_refreshed", "glyph", null],
  "lifecycle-decremented": ["life_decremented", "glyph", null],
  "lifecycle-expired": ["life_expired", "glyph", null],
  "lifecycle-trap-broken": ["life_shattered", "glyph", null],
  "lifecycle-death-clear": ["life_death_clear", "glyph", null],
  "lifecycle-ended": ["life_ended", "glyph", null],
  "lifecycle-trap-broken-reapplied": ["life_broken_reapplied", "glyph", null],
  "combat-in-progress": ["in_combat", "glyph", null],
});

/**
 * Plain glyph shown for a new sigil key when the plain style is asked for.
 * Sigils with no plain twin fall back to the unknown glyph.
 *
 * @type {Readonly<Record<string, string>>}
 */
const SIGIL_PLAIN_TWINS = Object.freeze({
  dazed_warrior: "status-stun",
  dazed_rogue: "status-stun",
  frozen: "status-stun",
  snagged: "status-slow",
  hobbled_warrior: "status-slow",
  hobbled_rogue: "status-slow",
  withering: "status-anti-heal",
  unbound: "status-freedom",
  kindled: "status-burst",
  moonward: "status-spawn-shield",
  empowerment: "modifier-amplification",
  barrier: "modifier-mitigation",
  in_combat: "combat-in-progress",
  unseen: "status-unseen",
  life_applied: "lifecycle-applied",
  life_refreshed: "lifecycle-refreshed",
  life_decremented: "lifecycle-decremented",
  life_expired: "lifecycle-expired",
  life_ended: "lifecycle-ended",
  life_thawed: "lifecycle-expired",
  life_shattered: "lifecycle-trap-broken",
  life_death_clear: "lifecycle-death-clear",
  life_broken_reapplied: "lifecycle-trap-broken-reapplied",
  rest_vessel: "status-rest",
  moon_dew: "status-rest",
  crest_A: "team-a",
  crest_B: "team-b",
  ab_burst: "activation-burst",
  ab_charge: "activation-charge",
  ab_freezing_trap: "activation-trap",
  ab_crippling_poison: "activation-poison",
  ab_salvation: "activation-holy-word",
  ab_lesser_heal: "activation-basic-heal",
  seal_basic: "activation-basic-damage",
});

/**
 * Status token id (vocabulary.js) -> sigil glyph name, so a status icon is the
 * small copy of its body mark in the color of the class that laid it. Two
 * more IDs of `symbols.js` STATUS_NAMES have one: "rest" (Rest
 * (Regeneration): the hand-drawn rest_vessel of HAND_GLYPHS, made from its
 * arena mark, the pale pour into the life vessel) and "stun_broken" (Freezing
 * Trap (Stun Broken): the shattered chain of its status-change mark).
 *
 * @type {Readonly<Record<string, string>>}
 */
const STATUS_SIGILS = Object.freeze({
  stun_warrior_charge: "dazed_warrior",
  stun_rogue_poison: "dazed_rogue",
  stun_hunter_trap: "frozen",
  stun_broken: "life_shattered",
  slow_warrior_charge: "hobbled_warrior",
  slow_rogue_poison: "hobbled_rogue",
  slow_hunter_basic: "snagged",
  anti_heal_rogue_poison: "withering",
  priest_freedom: "unbound",
  mage_burst: "kindled",
  in_combat: "in_combat",
  spawn_shield: "moonward",
  rest: "rest_vessel",
});

/**
 * Aura modifier token id (vocabulary.js) -> sigil glyph name.
 *
 * @type {Readonly<Record<string, string>>}
 */
const MODIFIER_SIGILS = Object.freeze({
  mage_amplification: "empowerment",
  warrior_mitigation: "barrier",
});

/**
 * Freeze value and every object and array inside it, then return value.
 * Used once on the generated sigil data so no caller can change it.
 *
 * @template T
 * @param {T} value
 * @returns {T}
 */
function deepFreeze(value) {
  if (value !== null && typeof value === "object") {
    for (const child of Object.values(value)) {
      deepFreeze(child);
    }
    Object.freeze(value);
  }
  return value;
}

// GENERATED SIGIL DATA START (gen_sigil_data.py writes the data line; do not edit it by hand)
/**
 * The generated Heraldic Sigils geometry, all in a 24-unit box.
 * emblems.<class>.<micro|standard|display> = {outline, body, jewels: [[d,
 * paint]], filigree: [[d, paint, opacity]]}; glyphs.<name>.<std|mic> =
 * [[d, paint, fillOpacity, keyline (1 joins the Ink keyline)]]. Paint names
 * map to colour tokens in src/styles/icons.css; "cls" means currentColor.
 *
 * @type {{
 *   emblems: Readonly<Record<string, Readonly<Record<string, {outline: string, body: string, jewels: ReadonlyArray<readonly [string, string]>, filigree: ReadonlyArray<readonly [string, string, number]>}>>>>,
 *   glyphs: Readonly<Record<string, Readonly<Record<string, ReadonlyArray<readonly [string, string, number, number]>>>>>,
 * }}
 */
const SIGIL_DATA = deepFreeze(
  JSON.parse(
    '{"emblems":{"mage":{"micro":{"outline":"M12 2L13.4 8.7L16.7 7.3L15.3 10.6L22 12L15.3 13.4L16.7 16.7L13.4 15.3L12 22L10.6 15.3L7.3 16.7L8.7 13.4L2 12L8.7 10.6L7.3 7.3L10.6 8.7Z","body":"M12 2L13.4 8.7L16.7 7.3L15.3 10.6L22 12L15.3 13.4L16.7 16.7L13.4 15.3L12 22L10.6 15.3L7.3 16.7L8.7 13.4L2 12L8.7 10.6L7.3 7.3L10.6 8.7Z","jewels":[["M14 12L13.9 12.5L13.7 13L13.4 13.4L13 13.7L12.5 13.9L12 14L11.5 13.9L11 13.7L10.6 13.4L10.3 13L10.1 12.5L10 12L10.1 11.5L10.3 11L10.6 10.6L11 10.3L11.5 10.1L12 10L12.5 10.1L13 10.3L13.4 10.6L13.7 11L13.9 11.5Z","starcore_plain"]],"filigree":[]},"standard":{"outline":"M12 2L13.1 9.4L16.2 7.8L14.6 10.9L22 12L14.6 13.1L16.2 16.2L13.1 14.6L12 22L10.9 14.6L7.8 16.2L9.4 13.1L2 12L9.4 10.9L7.8 7.8L10.9 9.4Z","body":"M12 2L13.1 9.4L16.2 7.8L14.6 10.9L22 12L14.6 13.1L16.2 16.2L13.1 14.6L12 22L10.9 14.6L7.8 16.2L9.4 13.1L2 12L9.4 10.9L7.8 7.8L10.9 9.4Z","jewels":[["M13.6 12L13.6 12.4L13.4 12.8L13.1 13.1L12.8 13.4L12.4 13.6L12 13.6L11.6 13.6L11.2 13.4L10.9 13.1L10.6 12.8L10.4 12.4L10.4 12L10.4 11.6L10.6 11.2L10.9 10.9L11.2 10.6L11.6 10.4L12 10.4L12.4 10.4L12.8 10.6L13.1 10.9L13.4 11.2L13.6 11.6Z","starcore"]],"filigree":[]},"display":{"outline":"M12 2L13.1 9.4L16.2 7.8L14.6 10.9L22 12L14.6 13.1L16.2 16.2L13.1 14.6L12 22L10.9 14.6L7.8 16.2L9.4 13.1L2 12L9.4 10.9L7.8 7.8L10.9 9.4Z","body":"M12 2L13.1 9.4L16.2 7.8L14.6 10.9L22 12L14.6 13.1L16.2 16.2L13.1 14.6L12 22L10.9 14.6L7.8 16.2L9.4 13.1L2 12L9.4 10.9L7.8 7.8L10.9 9.4Z","jewels":[["M13.6 12L13.6 12.4L13.4 12.8L13.1 13.1L12.8 13.4L12.4 13.6L12 13.6L11.6 13.6L11.2 13.4L10.9 13.1L10.6 12.8L10.4 12.4L10.4 12L10.4 11.6L10.6 11.2L10.9 10.9L11.2 10.6L11.6 10.4L12 10.4L12.4 10.4L12.8 10.6L13.1 10.9L13.4 11.2L13.6 11.6Z","starcore"]],"filigree":[["M12 4.4L12.8 10L15.2 8.8L14 11.2L19.6 12L14 12.8L15.2 15.2L12.8 14L12 19.6L11.2 14L8.8 15.2L10 12.8L4.4 12L10 11.2L8.8 8.8L11.2 10Z","starcore",0.45]]}},"warrior":{"micro":{"outline":"M5.5 5.4L5.5 5.2L5.7 4.8L6.1 4.6L6.3 4.6L17.7 4.6L17.9 4.6L18.3 4.8L18.5 5.2L18.5 11.2L18.4 12.1L18.2 12.8L18.1 13.6L17.5 15.1L16.8 16.5L16 17.8L15.5 18.4L14.4 19.5L13.9 20.1L13.3 20.6L12 21.6L10.7 20.6L10.1 20.1L9.6 19.5L8.5 18.4L8 17.8L7.2 16.5L6.5 15.1L5.9 13.6L5.6 12.1L5.5 10.4Z","body":"M5.5 5.4L5.5 5.2L5.7 4.8L6.1 4.6L6.3 4.6L17.7 4.6L17.9 4.6L18.3 4.8L18.5 5.2L18.5 11.2L18.4 12.1L18.2 12.8L18.1 13.6L17.5 15.1L16.8 16.5L16 17.8L15.5 18.4L14.4 19.5L13.9 20.1L13.3 20.6L12 21.6L10.7 20.6L10.1 20.1L9.6 19.5L8.5 18.4L8 17.8L7.2 16.5L6.5 15.1L5.9 13.6L5.6 12.1L5.5 10.4Z","jewels":[],"filigree":[]},"standard":{"outline":"M5.5 5.4L5.5 5.2L5.7 4.8L6.1 4.6L6.3 4.6L17.7 4.6L17.9 4.6L18.3 4.8L18.5 5.2L18.5 11.2L18.4 12.1L18.2 12.8L18.1 13.6L17.5 15.1L16.8 16.5L16 17.8L15.5 18.4L14.4 19.5L13.9 20.1L13.3 20.6L12 21.6L10.7 20.6L10.1 20.1L9.6 19.5L8.5 18.4L8 17.8L7.2 16.5L6.5 15.1L5.9 13.6L5.6 12.1L5.5 10.4Z","body":"M5.5 5.4L5.5 5.2L5.7 4.8L6.1 4.6L6.3 4.6L17.7 4.6L17.9 4.6L18.3 4.8L18.5 5.2L18.5 11.2L18.4 12.1L18.2 12.8L18.1 13.6L17.5 15.1L16.8 16.5L16 17.8L15.5 18.4L14.4 19.5L13.9 20.1L13.3 20.6L12 21.6L10.7 20.6L10.1 20.1L9.6 19.5L8.5 18.4L8 17.8L7.2 16.5L6.5 15.1L5.9 13.6L5.6 12.1L5.5 10.4Z","jewels":[["M14.5 10.6L14.4 11.2L14.2 11.7L13.9 12.2L13.6 12.6L13.1 12.8L12.6 13L12 13.1L11.4 13L10.9 12.8L10.4 12.6L10.1 12.2L9.8 11.7L9.6 11.2L9.5 10.6L9.6 10L9.8 9.5L10.1 9L10.4 8.7L10.9 8.3L11.4 8.2L12 8.1L12.6 8.2L13.1 8.3L13.6 8.7L13.9 9L14.2 9.5L14.4 10Z","cls"]],"filigree":[]},"display":{"outline":"M5.5 5.4L5.5 5.2L5.7 4.8L6.1 4.6L6.3 4.6L17.7 4.6L17.9 4.6L18.3 4.8L18.5 5.2L18.5 11.2L18.4 12.1L18.2 12.8L18.1 13.6L17.5 15.1L16.8 16.5L16 17.8L15.5 18.4L14.4 19.5L13.9 20.1L13.3 20.6L12 21.6L10.7 20.6L10.1 20.1L9.6 19.5L8.5 18.4L8 17.8L7.2 16.5L6.5 15.1L5.9 13.6L5.6 12.1L5.5 10.4Z","body":"M5.5 5.4L5.5 5.2L5.7 4.8L6.1 4.6L6.3 4.6L17.7 4.6L17.9 4.6L18.3 4.8L18.5 5.2L18.5 11.2L18.4 12.1L18.2 12.8L18.1 13.6L17.5 15.1L16.8 16.5L16 17.8L15.5 18.4L14.4 19.5L13.9 20.1L13.3 20.6L12 21.6L10.7 20.6L10.1 20.1L9.6 19.5L8.5 18.4L8 17.8L7.2 16.5L6.5 15.1L5.9 13.6L5.6 12.1L5.5 10.4Z","jewels":[["M14.5 10.6L14.4 11.2L14.2 11.7L13.9 12.2L13.6 12.6L13.1 12.8L12.6 13L12 13.1L11.4 13L10.9 12.8L10.4 12.6L10.1 12.2L9.8 11.7L9.6 11.2L9.5 10.6L9.6 10L9.8 9.5L10.1 9L10.4 8.7L10.9 8.3L11.4 8.2L12 8.1L12.6 8.2L13.1 8.3L13.6 8.7L13.9 9L14.2 9.5L14.4 10Z","cls"]],"filigree":[["M6.8 6.5L6.8 6.3L7 6.1L7.1 6L7.4 5.9L16.6 5.9L16.9 6L17.1 6.2L17.2 6.5L17.2 10.4L17.1 12.1L16.9 13.6L16.6 15L16.1 16.3L15.5 17.4L15.1 17.9L14.3 18.8L13.8 19.1L13.2 19.4L12.6 19.7L12 19.9L11.4 19.7L10.8 19.4L10.2 19.1L9.7 18.8L8.9 17.9L8.5 17.4L7.9 16.3L7.4 15L7.2 14.3L6.9 12.8L6.9 12.1L6.8 10.4Z","forge_white",0.4]]}},"hunter":{"micro":{"outline":"M11.4 14.9L11.7 15.3L11.9 15.8L11.8 16.3L11.6 16.7L11.2 17L10.7 17.2L10.2 17.1L9.6 16.9L8.9 16.6L8.3 16.3L7.3 15.4L6.5 14.5L5.8 13.4L5.6 12.8L5.2 11.6L5.1 10.3L5.1 9L5.3 7.2L5.4 6.7L5.7 5.7L6.1 4.8L6.6 4L6.8 3.8L7 3.6L7.3 3.6L7.5 3.7L7.7 3.9L7.8 4.1L7.9 4.4L7.3 5.8L7.1 6.6L7 7.5L6.9 9L7 10.1L7 10.6L7.3 11.6L7.5 12L8 12.8L8.5 13.4L8.8 13.7L9.6 14.2L10 14.4L11 14.7ZM13 14.7L14 14.4L14.4 14.2L15.2 13.7L15.5 13.4L16.1 12.8L16.3 12.4L16.7 11.6L16.9 10.6L17 10.1L17.1 9L17 7.5L16.9 6.6L16.7 5.8L16.1 4.4L16.1 4.1L16.3 3.9L16.4 3.7L16.7 3.6L17 3.6L17.2 3.8L17.4 4L17.9 4.8L18.3 5.7L18.6 6.7L18.7 7.2L18.9 9L18.9 10.3L18.8 11.6L18.4 12.8L18.2 13.4L17.6 14.5L16.7 15.4L15.7 16.3L15.1 16.6L14.4 16.9L13.8 17.1L13.3 17.2L12.8 17L12.4 16.7L12.2 16.3L12.1 15.8L12.3 15.3L12.6 14.9ZM6.2 13.3L5.8 13.3L5.5 13.1L5.2 12.8L5.1 12.4L5.1 12L5.3 11.7L5.6 11.4L6.6 11L7.5 10.5L8.4 9.8L9.2 9L9.3 8.9L9.5 8.9L9.7 9L9.8 9.2L9.9 9.3L9.9 9.5L9.8 9.7L9.3 10.5L8.7 11.3L8.2 11.9L7.2 12.7L6.6 13.2ZM17.4 13.2L16.8 12.7L16.1 12.2L15.6 11.6L15 10.9L14.5 10.1L14.1 9.5L14.1 9.3L14.2 9.2L14.3 9L14.5 8.9L14.7 8.9L14.8 9L15.6 9.8L16.5 10.5L17.4 11L18.4 11.4L18.7 11.7L18.9 12L18.9 12.4L18.8 12.8L18.6 13.1L18.2 13.3L17.8 13.3ZM14.8 17.2L14.7 17.8L14.5 18.4L14.2 18.9L13.8 19.4L13.2 19.7L12.6 19.9L12 20L11.4 19.9L10.8 19.7L10.2 19.4L9.8 18.9L9.5 18.4L9.3 17.8L9.2 17.2L9.3 16.6L9.5 16L9.8 15.4L10.2 15L10.8 14.7L11.4 14.5L12 14.4L12.6 14.5L13.2 14.7L13.8 15L14.2 15.4L14.5 16L14.7 16.6Z","body":"M11.4 14.9L11.7 15.3L11.9 15.8L11.8 16.3L11.6 16.7L11.2 17L10.7 17.2L10.2 17.1L9.6 16.9L8.9 16.6L8.3 16.3L7.3 15.4L6.5 14.5L5.8 13.4L5.6 12.8L5.2 11.6L5.1 10.3L5.1 9L5.3 7.2L5.4 6.7L5.7 5.7L6.1 4.8L6.6 4L6.8 3.8L7 3.6L7.3 3.6L7.5 3.7L7.7 3.9L7.8 4.1L7.9 4.4L7.3 5.8L7.1 6.6L7 7.5L6.9 9L7 10.1L7 10.6L7.3 11.6L7.5 12L8 12.8L8.5 13.4L8.8 13.7L9.6 14.2L10 14.4L11 14.7ZM13 14.7L14 14.4L14.4 14.2L15.2 13.7L15.5 13.4L16.1 12.8L16.3 12.4L16.7 11.6L16.9 10.6L17 10.1L17.1 9L17 7.5L16.9 6.6L16.7 5.8L16.1 4.4L16.1 4.1L16.3 3.9L16.4 3.7L16.7 3.6L17 3.6L17.2 3.8L17.4 4L17.9 4.8L18.3 5.7L18.6 6.7L18.7 7.2L18.9 9L18.9 10.3L18.8 11.6L18.4 12.8L18.2 13.4L17.6 14.5L16.7 15.4L15.7 16.3L15.1 16.6L14.4 16.9L13.8 17.1L13.3 17.2L12.8 17L12.4 16.7L12.2 16.3L12.1 15.8L12.3 15.3L12.6 14.9ZM6.2 13.3L5.8 13.3L5.5 13.1L5.2 12.8L5.1 12.4L5.1 12L5.3 11.7L5.6 11.4L6.6 11L7.5 10.5L8.4 9.8L9.2 9L9.3 8.9L9.5 8.9L9.7 9L9.8 9.2L9.9 9.3L9.9 9.5L9.8 9.7L9.3 10.5L8.7 11.3L8.2 11.9L7.2 12.7L6.6 13.2ZM17.4 13.2L16.8 12.7L16.1 12.2L15.6 11.6L15 10.9L14.5 10.1L14.1 9.5L14.1 9.3L14.2 9.2L14.3 9L14.5 8.9L14.7 8.9L14.8 9L15.6 9.8L16.5 10.5L17.4 11L18.4 11.4L18.7 11.7L18.9 12L18.9 12.4L18.8 12.8L18.6 13.1L18.2 13.3L17.8 13.3ZM14.8 17.2L14.7 17.8L14.5 18.4L14.2 18.9L13.8 19.4L13.2 19.7L12.6 19.9L12 20L11.4 19.9L10.8 19.7L10.2 19.4L9.8 18.9L9.5 18.4L9.3 17.8L9.2 17.2L9.3 16.6L9.5 16L9.8 15.4L10.2 15L10.8 14.7L11.4 14.5L12 14.4L12.6 14.5L13.2 14.7L13.8 15L14.2 15.4L14.5 16L14.7 16.6Z","jewels":[],"filigree":[]},"standard":{"outline":"M11.2 15.2L11.4 15.5L11.6 15.8L11.5 16.2L11.3 16.5L11 16.7L10.7 16.9L10.3 16.8L9.1 16.3L8.5 16L8 15.6L7.5 15.2L7.1 14.8L6.3 13.8L5.8 12.7L5.6 12.1L5.5 11.5L5.3 10.2L5.3 9L5.4 7.8L5.6 6.7L5.9 5.8L6.1 5.3L6.3 4.9L6.8 4L6.9 3.9L7.1 3.8L7.3 3.8L7.5 3.9L7.6 4L7.7 4.2L7.7 4.4L7.1 5.7L6.9 6.5L6.8 7.4L6.7 9L6.7 10.1L6.8 10.7L7.1 11.6L7.5 12.5L7.7 12.9L8 13.3L8.6 13.9L9.4 14.4L10.3 14.8L10.9 15ZM13.1 15L14.1 14.7L14.6 14.4L15.4 13.9L15.7 13.6L16.3 12.9L16.5 12.5L16.9 11.6L17.1 11.2L17.3 10.1L17.3 9L17.2 7.4L17.1 6.5L16.9 5.7L16.3 4.4L16.3 4.2L16.4 4L16.5 3.9L16.7 3.8L16.9 3.8L17.1 3.9L17.2 4L17.7 4.9L17.9 5.3L18.1 5.8L18.4 6.7L18.6 7.8L18.7 9L18.7 10.2L18.5 11.5L18.4 12.1L18.2 12.7L17.7 13.8L16.9 14.8L16.5 15.2L16 15.6L15.5 16L14.9 16.3L13.7 16.8L13.3 16.9L13 16.7L12.7 16.5L12.5 16.2L12.4 15.8L12.6 15.5L12.8 15.2ZM6.2 13.1L5.9 13L5.6 12.9L5.4 12.7L5.3 12.4L5.4 12.1L5.5 11.8L5.8 11.6L6.7 11.2L7.6 10.7L8.5 9.9L9.2 9.1L9.5 9.1L9.6 9.1L9.7 9.2L9.7 9.3L9.7 9.6L9.2 10.4L8.6 11.1L8 11.8L7.4 12.3L6.5 13ZM17.6 13L16.3 12L15.4 11.1L14.8 10.4L14.3 9.6L14.3 9.3L14.3 9.2L14.4 9.1L14.5 9.1L14.8 9.1L15.5 9.9L16.4 10.7L17.3 11.2L18.2 11.6L18.5 11.8L18.6 12.1L18.6 12.4L18.6 12.7L18.4 12.9L18.1 13L17.8 13.1ZM6.1 8.9L5.8 8.8L5.6 8.7L5.4 8.5L5.3 8.3L5.4 8L5.5 7.8L5.7 7.6L6.5 7.2L7.2 6.8L7.9 6.1L8.5 5.4L8.6 5.4L8.9 5.4L9 5.5L9 5.6L9 5.9L8.6 6.6L8.1 7.2L7.4 8L6.9 8.4L6.3 8.8ZM17.7 8.8L17.1 8.4L16.6 8L16.1 7.5L15.6 6.9L15.2 6.2L15 5.8L15 5.5L15.1 5.4L15.2 5.4L15.5 5.4L16.1 6.1L16.8 6.8L17.5 7.2L18.3 7.6L18.5 7.8L18.6 8L18.7 8.3L18.6 8.5L18.4 8.7L18.2 8.8L17.9 8.9ZM14.3 17.2L14.2 17.7L14.1 18.2L13.8 18.6L13.4 19L13 19.3L12.5 19.4L12 19.5L11.5 19.4L11 19.3L10.6 19L10.2 18.6L9.9 18.2L9.8 17.7L9.7 17.2L9.8 16.7L9.9 16.2L10.2 15.8L10.6 15.4L11 15.1L11.5 15L12 14.9L12.5 15L13 15.1L13.4 15.4L13.8 15.8L14.1 16.2L14.2 16.7Z","body":"M11.2 15.2L11.4 15.5L11.6 15.8L11.5 16.2L11.3 16.5L11 16.7L10.7 16.9L10.3 16.8L9.1 16.3L8.5 16L8 15.6L7.5 15.2L7.1 14.8L6.3 13.8L5.8 12.7L5.6 12.1L5.5 11.5L5.3 10.2L5.3 9L5.4 7.8L5.6 6.7L5.9 5.8L6.1 5.3L6.3 4.9L6.8 4L6.9 3.9L7.1 3.8L7.3 3.8L7.5 3.9L7.6 4L7.7 4.2L7.7 4.4L7.1 5.7L6.9 6.5L6.8 7.4L6.7 9L6.7 10.1L6.8 10.7L7.1 11.6L7.5 12.5L7.7 12.9L8 13.3L8.6 13.9L9.4 14.4L10.3 14.8L10.9 15ZM13.1 15L14.1 14.7L14.6 14.4L15.4 13.9L15.7 13.6L16.3 12.9L16.5 12.5L16.9 11.6L17.1 11.2L17.3 10.1L17.3 9L17.2 7.4L17.1 6.5L16.9 5.7L16.3 4.4L16.3 4.2L16.4 4L16.5 3.9L16.7 3.8L16.9 3.8L17.1 3.9L17.2 4L17.7 4.9L17.9 5.3L18.1 5.8L18.4 6.7L18.6 7.8L18.7 9L18.7 10.2L18.5 11.5L18.4 12.1L18.2 12.7L17.7 13.8L16.9 14.8L16.5 15.2L16 15.6L15.5 16L14.9 16.3L13.7 16.8L13.3 16.9L13 16.7L12.7 16.5L12.5 16.2L12.4 15.8L12.6 15.5L12.8 15.2ZM6.2 13.1L5.9 13L5.6 12.9L5.4 12.7L5.3 12.4L5.4 12.1L5.5 11.8L5.8 11.6L6.7 11.2L7.6 10.7L8.5 9.9L9.2 9.1L9.5 9.1L9.6 9.1L9.7 9.2L9.7 9.3L9.7 9.6L9.2 10.4L8.6 11.1L8 11.8L7.4 12.3L6.5 13ZM17.6 13L16.3 12L15.4 11.1L14.8 10.4L14.3 9.6L14.3 9.3L14.3 9.2L14.4 9.1L14.5 9.1L14.8 9.1L15.5 9.9L16.4 10.7L17.3 11.2L18.2 11.6L18.5 11.8L18.6 12.1L18.6 12.4L18.6 12.7L18.4 12.9L18.1 13L17.8 13.1ZM6.1 8.9L5.8 8.8L5.6 8.7L5.4 8.5L5.3 8.3L5.4 8L5.5 7.8L5.7 7.6L6.5 7.2L7.2 6.8L7.9 6.1L8.5 5.4L8.6 5.4L8.9 5.4L9 5.5L9 5.6L9 5.9L8.6 6.6L8.1 7.2L7.4 8L6.9 8.4L6.3 8.8ZM17.7 8.8L17.1 8.4L16.6 8L16.1 7.5L15.6 6.9L15.2 6.2L15 5.8L15 5.5L15.1 5.4L15.2 5.4L15.5 5.4L16.1 6.1L16.8 6.8L17.5 7.2L18.3 7.6L18.5 7.8L18.6 8L18.7 8.3L18.6 8.5L18.4 8.7L18.2 8.8L17.9 8.9Z","jewels":[["M14.3 17.2L14.2 17.7L14.1 18.2L13.8 18.6L13.4 19L13 19.3L12.5 19.4L12 19.5L11.5 19.4L11 19.3L10.6 19L10.2 18.6L9.9 18.2L9.8 17.7L9.7 17.2L9.8 16.7L9.9 16.2L10.2 15.8L10.6 15.4L11 15.1L11.5 15L12 14.9L12.5 15L13 15.1L13.4 15.4L13.8 15.8L14.1 16.2L14.2 16.7Z","cls"]],"filigree":[]},"display":{"outline":"M11.2 15.2L11.4 15.5L11.6 15.8L11.5 16.2L11.3 16.5L11 16.7L10.7 16.9L10.3 16.8L9.1 16.3L8.5 16L8 15.6L7.5 15.2L7.1 14.8L6.3 13.8L5.8 12.7L5.6 12.1L5.5 11.5L5.3 10.2L5.3 9L5.4 7.8L5.6 6.7L5.9 5.8L6.1 5.3L6.3 4.9L6.8 4L6.9 3.9L7.1 3.8L7.3 3.8L7.5 3.9L7.6 4L7.7 4.2L7.7 4.4L7.1 5.7L6.9 6.5L6.8 7.4L6.7 9L6.7 10.1L6.8 10.7L7.1 11.6L7.5 12.5L7.7 12.9L8 13.3L8.6 13.9L9.4 14.4L10.3 14.8L10.9 15ZM13.1 15L14.1 14.7L14.6 14.4L15.4 13.9L15.7 13.6L16.3 12.9L16.5 12.5L16.9 11.6L17.1 11.2L17.3 10.1L17.3 9L17.2 7.4L17.1 6.5L16.9 5.7L16.3 4.4L16.3 4.2L16.4 4L16.5 3.9L16.7 3.8L16.9 3.8L17.1 3.9L17.2 4L17.7 4.9L17.9 5.3L18.1 5.8L18.4 6.7L18.6 7.8L18.7 9L18.7 10.2L18.5 11.5L18.4 12.1L18.2 12.7L17.7 13.8L16.9 14.8L16.5 15.2L16 15.6L15.5 16L14.9 16.3L13.7 16.8L13.3 16.9L13 16.7L12.7 16.5L12.5 16.2L12.4 15.8L12.6 15.5L12.8 15.2ZM6.2 13.1L5.9 13L5.6 12.9L5.4 12.7L5.3 12.4L5.4 12.1L5.5 11.8L5.8 11.6L6.7 11.2L7.6 10.7L8.5 9.9L9.2 9.1L9.5 9.1L9.6 9.1L9.7 9.2L9.7 9.3L9.7 9.6L9.2 10.4L8.6 11.1L8 11.8L7.4 12.3L6.5 13ZM17.6 13L16.3 12L15.4 11.1L14.8 10.4L14.3 9.6L14.3 9.3L14.3 9.2L14.4 9.1L14.5 9.1L14.8 9.1L15.5 9.9L16.4 10.7L17.3 11.2L18.2 11.6L18.5 11.8L18.6 12.1L18.6 12.4L18.6 12.7L18.4 12.9L18.1 13L17.8 13.1ZM6.1 8.9L5.8 8.8L5.6 8.7L5.4 8.5L5.3 8.3L5.4 8L5.5 7.8L5.7 7.6L6.5 7.2L7.2 6.8L7.9 6.1L8.5 5.4L8.6 5.4L8.9 5.4L9 5.5L9 5.6L9 5.9L8.6 6.6L8.1 7.2L7.4 8L6.9 8.4L6.3 8.8ZM17.7 8.8L17.1 8.4L16.6 8L16.1 7.5L15.6 6.9L15.2 6.2L15 5.8L15 5.5L15.1 5.4L15.2 5.4L15.5 5.4L16.1 6.1L16.8 6.8L17.5 7.2L18.3 7.6L18.5 7.8L18.6 8L18.7 8.3L18.6 8.5L18.4 8.7L18.2 8.8L17.9 8.9ZM7.9 15L7.7 15.1L7.5 15.1L7.3 15L7.1 14.8L7 14.6L7 14.4L7.1 14.2L7.4 13.8L7.7 13.5L8.2 13L8.9 12.6L9.2 12.4L9.9 12.1L10 12.1L10.2 12.2L10.3 12.4L10.2 12.6L9.6 13L9.1 13.4L8.7 13.9L8.1 14.8ZM15.9 14.8L15.3 13.9L14.9 13.4L14.4 13L13.8 12.6L13.7 12.4L13.8 12.2L14 12.1L14.1 12.1L14.5 12.2L15.1 12.6L15.8 13L16.4 13.5L16.6 13.8L16.9 14.2L17 14.4L17 14.6L16.9 14.8L16.7 15L16.5 15.1L16.3 15.1L16.1 15ZM14.3 17.2L14.2 17.7L14.1 18.2L13.8 18.6L13.4 19L13 19.3L12.5 19.4L12 19.5L11.5 19.4L11 19.3L10.6 19L10.2 18.6L9.9 18.2L9.8 17.7L9.7 17.2L9.8 16.7L9.9 16.2L10.2 15.8L10.6 15.4L11 15.1L11.5 15L12 14.9L12.5 15L13 15.1L13.4 15.4L13.8 15.8L14.1 16.2L14.2 16.7Z","body":"M11.2 15.2L11.4 15.5L11.6 15.8L11.5 16.2L11.3 16.5L11 16.7L10.7 16.9L10.3 16.8L9.1 16.3L8.5 16L8 15.6L7.5 15.2L7.1 14.8L6.3 13.8L5.8 12.7L5.6 12.1L5.5 11.5L5.3 10.2L5.3 9L5.4 7.8L5.6 6.7L5.9 5.8L6.1 5.3L6.3 4.9L6.8 4L6.9 3.9L7.1 3.8L7.3 3.8L7.5 3.9L7.6 4L7.7 4.2L7.7 4.4L7.1 5.7L6.9 6.5L6.8 7.4L6.7 9L6.7 10.1L6.8 10.7L7.1 11.6L7.5 12.5L7.7 12.9L8 13.3L8.6 13.9L9.4 14.4L10.3 14.8L10.9 15ZM13.1 15L14.1 14.7L14.6 14.4L15.4 13.9L15.7 13.6L16.3 12.9L16.5 12.5L16.9 11.6L17.1 11.2L17.3 10.1L17.3 9L17.2 7.4L17.1 6.5L16.9 5.7L16.3 4.4L16.3 4.2L16.4 4L16.5 3.9L16.7 3.8L16.9 3.8L17.1 3.9L17.2 4L17.7 4.9L17.9 5.3L18.1 5.8L18.4 6.7L18.6 7.8L18.7 9L18.7 10.2L18.5 11.5L18.4 12.1L18.2 12.7L17.7 13.8L16.9 14.8L16.5 15.2L16 15.6L15.5 16L14.9 16.3L13.7 16.8L13.3 16.9L13 16.7L12.7 16.5L12.5 16.2L12.4 15.8L12.6 15.5L12.8 15.2ZM6.2 13.1L5.9 13L5.6 12.9L5.4 12.7L5.3 12.4L5.4 12.1L5.5 11.8L5.8 11.6L6.7 11.2L7.6 10.7L8.5 9.9L9.2 9.1L9.5 9.1L9.6 9.1L9.7 9.2L9.7 9.3L9.7 9.6L9.2 10.4L8.6 11.1L8 11.8L7.4 12.3L6.5 13ZM17.6 13L16.3 12L15.4 11.1L14.8 10.4L14.3 9.6L14.3 9.3L14.3 9.2L14.4 9.1L14.5 9.1L14.8 9.1L15.5 9.9L16.4 10.7L17.3 11.2L18.2 11.6L18.5 11.8L18.6 12.1L18.6 12.4L18.6 12.7L18.4 12.9L18.1 13L17.8 13.1ZM6.1 8.9L5.8 8.8L5.6 8.7L5.4 8.5L5.3 8.3L5.4 8L5.5 7.8L5.7 7.6L6.5 7.2L7.2 6.8L7.9 6.1L8.5 5.4L8.6 5.4L8.9 5.4L9 5.5L9 5.6L9 5.9L8.6 6.6L8.1 7.2L7.4 8L6.9 8.4L6.3 8.8ZM17.7 8.8L17.1 8.4L16.6 8L16.1 7.5L15.6 6.9L15.2 6.2L15 5.8L15 5.5L15.1 5.4L15.2 5.4L15.5 5.4L16.1 6.1L16.8 6.8L17.5 7.2L18.3 7.6L18.5 7.8L18.6 8L18.7 8.3L18.6 8.5L18.4 8.7L18.2 8.8L17.9 8.9ZM7.9 15L7.7 15.1L7.5 15.1L7.3 15L7.1 14.8L7 14.6L7 14.4L7.1 14.2L7.4 13.8L7.7 13.5L8.2 13L8.9 12.6L9.2 12.4L9.9 12.1L10 12.1L10.2 12.2L10.3 12.4L10.2 12.6L9.6 13L9.1 13.4L8.7 13.9L8.1 14.8ZM15.9 14.8L15.3 13.9L14.9 13.4L14.4 13L13.8 12.6L13.7 12.4L13.8 12.2L14 12.1L14.1 12.1L14.5 12.2L15.1 12.6L15.8 13L16.4 13.5L16.6 13.8L16.9 14.2L17 14.4L17 14.6L16.9 14.8L16.7 15L16.5 15.1L16.3 15.1L16.1 15Z","jewels":[["M14.3 17.2L14.2 17.7L14.1 18.2L13.8 18.6L13.4 19L13 19.3L12.5 19.4L12 19.5L11.5 19.4L11 19.3L10.6 19L10.2 18.6L9.9 18.2L9.8 17.7L9.7 17.2L9.8 16.7L9.9 16.2L10.2 15.8L10.6 15.4L11 15.1L11.5 15L12 14.9L12.5 15L13 15.1L13.4 15.4L13.8 15.8L14.1 16.2L14.2 16.7Z","cls"]],"filigree":[]}},"rogue":{"micro":{"outline":"M12.6 14.1L10.2 11.4L8.7 7.8L7.9 4.2L10.9 6.3L13.7 9.1L15.3 12.3ZM10.3 15.7L17.7 10.7L18.2 12.1L11.3 16.7ZM13.9 15L15.6 13.8L17.8 16.9L15.9 18.1ZM16.7 17.2L18.8 17.4L18.1 19.4L16 19.3ZM8.7 12.3L10.3 9.1L13.1 6.3L16.1 4.2L15.3 7.8L13.8 11.4L11.4 14.1ZM6.3 10.7L13.7 15.7L12.7 16.7L5.8 12.1ZM8.4 13.8L10.1 15L8.1 18.1L6.2 16.9ZM7.3 17.2L8 19.3L5.8 19.4L5.2 17.4Z","body":"M12.6 14.1L10.2 11.4L8.7 7.8L7.9 4.2L10.9 6.3L13.7 9.1L15.3 12.3ZM10.3 15.7L17.7 10.7L18.2 12.1L11.3 16.7ZM13.9 15L15.6 13.8L17.8 16.9L15.9 18.1ZM16.7 17.2L18.8 17.4L18.1 19.4L16 19.3ZM8.7 12.3L10.3 9.1L13.1 6.3L16.1 4.2L15.3 7.8L13.8 11.4L11.4 14.1ZM6.3 10.7L13.7 15.7L12.7 16.7L5.8 12.1ZM8.4 13.8L10.1 15L8.1 18.1L6.2 16.9ZM7.3 17.2L8 19.3L5.8 19.4L5.2 17.4Z","jewels":[],"filigree":[]},"standard":{"outline":"M12.9 14L10.6 11.1L8.9 7.6L7.9 4.2L10.7 6.4L13.3 9.3L15.1 12.5ZM11 15.2L16.9 11.2L17.5 12.6L12 16.2ZM14.1 14.9L15.4 13.9L17.6 17L16.1 18ZM16.7 17.2L18.6 17.6L18.1 19.4L16.3 19.1ZM8.9 12.5L10.7 9.3L13.3 6.4L16.1 4.2L15.1 7.6L13.4 11.1L11.1 14ZM7.1 11.2L13 15.2L12 16.2L6.5 12.6ZM8.6 13.9L9.9 14.9L7.9 18L6.4 17ZM7.3 17.2L7.7 19.1L5.8 19.4L5.5 17.6Z","body":"M12.9 14L10.6 11.1L8.9 7.6L7.9 4.2L10.7 6.4L13.3 9.3L15.1 12.5ZM11 15.2L16.9 11.2L17.5 12.6L12 16.2ZM14.1 14.9L15.4 13.9L17.6 17L16.1 18ZM16.7 17.2L18.6 17.6L18.1 19.4L16.3 19.1ZM8.9 12.5L10.7 9.3L13.3 6.4L16.1 4.2L15.1 7.6L13.4 11.1L11.1 14ZM7.1 11.2L13 15.2L12 16.2L6.5 12.6ZM8.6 13.9L9.9 14.9L7.9 18L6.4 17ZM7.3 17.2L7.7 19.1L5.8 19.4L5.5 17.6Z","jewels":[["M12 8L12.6 8.9L13.1 9.6L13.4 10.3L13.4 10.7L13.3 11.2L13.1 11.6L12.8 11.9L12.4 12L12 12.1L11.6 12L11.2 11.9L10.9 11.6L10.7 11.2L10.6 10.7L10.6 10.3L10.9 9.6L11.4 8.9Z","cls"]],"filigree":[]},"display":{"outline":"M12.9 14L10.6 11.1L8.9 7.6L7.9 4.2L10.7 6.4L13.3 9.3L15.1 12.5ZM11 15.2L16.9 11.2L17.5 12.6L12 16.2ZM14.1 14.9L15.4 13.9L17.6 17L16.1 18ZM16.7 17.2L18.6 17.6L18.1 19.4L16.3 19.1ZM8.9 12.5L10.7 9.3L13.3 6.4L16.1 4.2L15.1 7.6L13.4 11.1L11.1 14ZM7.1 11.2L13 15.2L12 16.2L6.5 12.6ZM8.6 13.9L9.9 14.9L7.9 18L6.4 17ZM7.3 17.2L7.7 19.1L5.8 19.4L5.5 17.6Z","body":"M12.9 14L10.6 11.1L8.9 7.6L7.9 4.2L10.7 6.4L13.3 9.3L15.1 12.5ZM11 15.2L16.9 11.2L17.5 12.6L12 16.2ZM14.1 14.9L15.4 13.9L17.6 17L16.1 18ZM16.7 17.2L18.6 17.6L18.1 19.4L16.3 19.1ZM8.9 12.5L10.7 9.3L13.3 6.4L16.1 4.2L15.1 7.6L13.4 11.1L11.1 14ZM7.1 11.2L13 15.2L12 16.2L6.5 12.6ZM8.6 13.9L9.9 14.9L7.9 18L6.4 17ZM7.3 17.2L7.7 19.1L5.8 19.4L5.5 17.6Z","jewels":[["M12 8L12.6 8.9L13.1 9.6L13.4 10.3L13.4 10.7L13.3 11.2L13.1 11.6L12.8 11.9L12.4 12L12 12.1L11.6 12L11.2 11.9L10.9 11.6L10.7 11.2L10.6 10.7L10.6 10.3L10.9 9.6L11.4 8.9Z","cls"]],"filigree":[["M13.4 12.5L9.3 6.3","blade",0.5],["M10.6 12.5L14.7 6.3","blade",0.5]]}},"priest":{"micro":{"outline":"M21.6 12L21.5 13.2L21.3 14.5L20.9 15.7L20.3 16.8L19.6 17.8L18.8 18.8L17.8 19.6L16.8 20.3L15.7 20.9L14.5 21.3L13.2 21.5L12 21.6L10.8 21.5L9.5 21.3L8.3 20.9L7.2 20.3L6.2 19.6L5.2 18.8L4.4 17.8L3.7 16.8L3.1 15.7L2.7 14.5L2.5 13.2L2.4 12L2.5 10.8L2.7 9.5L3.1 8.3L3.7 7.2L4.4 6.2L5.2 5.2L6.2 4.4L7.2 3.7L8.3 3.1L9.5 2.7L10.8 2.5L12 2.4L13.2 2.5L14.5 2.7L15.7 3.1L16.8 3.7L17.8 4.4L18.8 5.2L19.6 6.2L20.3 7.2L20.9 8.3L21.3 9.5L21.5 10.8Z","body":"M21.6 12L21.5 13.2L21.3 14.5L20.9 15.7L20.3 16.8L19.6 17.8L18.8 18.8L17.8 19.6L16.8 20.3L15.7 20.9L14.5 21.3L13.2 21.5L12 21.6L10.8 21.5L9.5 21.3L8.3 20.9L7.2 20.3L6.2 19.6L5.2 18.8L4.4 17.8L3.7 16.8L3.1 15.7L2.7 14.5L2.5 13.2L2.4 12L2.5 10.8L2.7 9.5L3.1 8.3L3.7 7.2L4.4 6.2L5.2 5.2L6.2 4.4L7.2 3.7L8.3 3.1L9.5 2.7L10.8 2.5L12 2.4L13.2 2.5L14.5 2.7L15.7 3.1L16.8 3.7L17.8 4.4L18.8 5.2L19.6 6.2L20.3 7.2L20.9 8.3L21.3 9.5L21.5 10.8ZM10.5 7.4L10.9 6.9L11.4 6.5L12 6.4L12.6 6.5L13.1 6.9L13.5 7.4L13.6 8L13.6 16L13.5 16.6L13.1 17.1L12.6 17.5L12 17.6L11.4 17.5L10.9 17.1L10.5 16.6L10.4 16L10.4 8ZM7.4 13.5L6.9 13.1L6.5 12.6L6.4 12L6.5 11.4L6.9 10.9L7.4 10.5L8 10.4L16 10.4L16.6 10.5L17.1 10.9L17.5 11.4L17.6 12L17.5 12.6L17.1 13.1L16.6 13.5L16 13.6L8 13.6ZM19.1 11.1L18.9 10.1L18.6 9.2L18.2 8.4L17.7 7.6L17.1 6.9L16.4 6.3L15.6 5.8L14.8 5.3L13.9 5L12.9 4.9L12 4.8L11.1 4.9L10.1 5L9.2 5.3L8.4 5.8L7.6 6.3L6.9 6.9L6.3 7.6L5.8 8.4L5.3 9.2L5 10.1L4.9 11.1L4.8 12L4.9 12.9L5 13.9L5.3 14.8L5.8 15.6L6.3 16.4L6.9 17.1L7.6 17.7L8.4 18.2L9.2 18.6L10.1 18.9L11.1 19.1L12 19.2L12.9 19.1L13.9 18.9L14.8 18.6L15.6 18.2L16.4 17.7L17.1 17.1L17.7 16.4L18.2 15.6L18.6 14.8L18.9 13.9L19.1 12.9L19.2 12Z","jewels":[],"filigree":[]},"standard":{"outline":"M19 12L18.9 12.9L18.8 13.8L18.5 14.7L18.1 15.5L17.6 16.3L16.9 16.9L16.3 17.6L15.5 18.1L14.7 18.5L13.8 18.8L12.9 18.9L12 19L11.1 18.9L10.2 18.8L9.3 18.5L8.5 18.1L7.7 17.6L7 16.9L6.5 16.3L5.9 15.5L5.5 14.7L5.2 13.8L5.1 12.9L5 12L5.1 11.1L5.2 10.2L5.5 9.3L5.9 8.5L6.5 7.7L7 7L7.7 6.5L8.5 5.9L9.3 5.5L10.2 5.2L11.1 5.1L12 5L12.9 5.1L13.8 5.2L14.7 5.5L15.5 5.9L16.3 6.5L16.9 7L17.6 7.7L18.1 8.5L18.5 9.3L18.8 10.2L18.9 11.1ZM12.7 4.4L12.5 4.6L12.3 4.8L12 4.8L11.7 4.8L11.5 4.6L11.3 4.4L11.2 4.1L11.5 3.2L11.7 2.2L11.9 2L12.1 2L12.3 2.2L12.8 4.1ZM17.9 7.1L17.6 7.2L17.3 7.1L17.1 6.9L16.9 6.7L16.8 6.4L16.9 6.1L17.1 5.9L17.4 5.6L18.7 4.9L19 4.9L19.1 5L19.1 5.3L18.1 6.9ZM19.6 12.7L19.4 12.5L19.2 12.3L19.1 12L19.2 11.7L19.4 11.5L19.6 11.3L19.9 11.2L20.8 11.5L21.8 11.7L22 11.9L22 12.1L21.8 12.3L19.9 12.8ZM16.9 17.9L16.8 17.6L16.9 17.3L17.1 17.1L17.3 16.9L17.6 16.8L17.9 16.9L18.1 17.1L18.4 17.4L19.1 18.7L19.1 19L19 19.1L18.7 19.1L17.1 18.1ZM11.3 19.6L11.5 19.4L11.7 19.2L12 19.1L12.3 19.2L12.5 19.4L12.7 19.6L12.8 19.9L12.5 20.8L12.3 21.8L12.1 22L11.9 22L11.7 21.8L11.2 19.9ZM6.1 16.9L6.4 16.8L6.7 16.9L6.9 17.1L7.1 17.3L7.2 17.6L7.1 17.9L6.9 18.1L6.5 18.4L5.3 19.1L5 19.1L4.9 19L4.9 18.7L5.9 17.1ZM4.4 11.3L4.6 11.5L4.8 11.7L4.8 12L4.8 12.3L4.6 12.5L4.4 12.7L4.1 12.8L3.2 12.5L2.2 12.3L2 12.1L2 11.9L2.2 11.7L4.1 11.2ZM7.1 6.1L7.2 6.4L7.1 6.7L6.9 6.9L6.7 7.1L6.4 7.2L6.1 7.1L5.9 6.9L5.6 6.5L4.9 5.3L4.9 5L5 4.9L5.3 4.9L6.9 5.9Z","body":"M19 12L18.9 12.9L18.8 13.8L18.5 14.7L18.1 15.5L17.6 16.3L16.9 16.9L16.3 17.6L15.5 18.1L14.7 18.5L13.8 18.8L12.9 18.9L12 19L11.1 18.9L10.2 18.8L9.3 18.5L8.5 18.1L7.7 17.6L7 16.9L6.5 16.3L5.9 15.5L5.5 14.7L5.2 13.8L5.1 12.9L5 12L5.1 11.1L5.2 10.2L5.5 9.3L5.9 8.5L6.5 7.7L7 7L7.7 6.5L8.5 5.9L9.3 5.5L10.2 5.2L11.1 5.1L12 5L12.9 5.1L13.8 5.2L14.7 5.5L15.5 5.9L16.3 6.5L16.9 7L17.6 7.7L18.1 8.5L18.5 9.3L18.8 10.2L18.9 11.1ZM10.8 8.2L11.1 7.8L11.5 7.5L12 7.4L12.5 7.5L12.9 7.8L13.2 8.2L13.3 8.7L13.3 15.3L13.2 15.8L12.9 16.2L12.5 16.5L12 16.6L11.5 16.5L11.1 16.2L10.8 15.8L10.7 15.3L10.7 8.7ZM8.2 13.2L7.8 12.9L7.5 12.5L7.4 12L7.5 11.5L7.8 11.1L8.2 10.8L8.7 10.7L15.3 10.7L15.8 10.8L16.2 11.1L16.5 11.5L16.6 12L16.5 12.5L16.2 12.9L15.8 13.2L15.3 13.3L8.7 13.3ZM12.7 4.4L12.5 4.6L12.3 4.8L12 4.8L11.7 4.8L11.5 4.6L11.3 4.4L11.2 4.1L11.5 3.2L11.7 2.2L11.9 2L12.1 2L12.3 2.2L12.8 4.1ZM17.9 7.1L17.6 7.2L17.3 7.1L17.1 6.9L16.9 6.7L16.8 6.4L16.9 6.1L17.1 5.9L17.4 5.6L18.7 4.9L19 4.9L19.1 5L19.1 5.3L18.1 6.9ZM19.6 12.7L19.4 12.5L19.2 12.3L19.1 12L19.2 11.7L19.4 11.5L19.6 11.3L19.9 11.2L20.8 11.5L21.8 11.7L22 11.9L22 12.1L21.8 12.3L19.9 12.8ZM16.9 17.9L16.8 17.6L16.9 17.3L17.1 17.1L17.3 16.9L17.6 16.8L17.9 16.9L18.1 17.1L18.4 17.4L19.1 18.7L19.1 19L19 19.1L18.7 19.1L17.1 18.1ZM11.3 19.6L11.5 19.4L11.7 19.2L12 19.1L12.3 19.2L12.5 19.4L12.7 19.6L12.8 19.9L12.5 20.8L12.3 21.8L12.1 22L11.9 22L11.7 21.8L11.2 19.9ZM6.1 16.9L6.4 16.8L6.7 16.9L6.9 17.1L7.1 17.3L7.2 17.6L7.1 17.9L6.9 18.1L6.5 18.4L5.3 19.1L5 19.1L4.9 19L4.9 18.7L5.9 17.1ZM4.4 11.3L4.6 11.5L4.8 11.7L4.8 12L4.8 12.3L4.6 12.5L4.4 12.7L4.1 12.8L3.2 12.5L2.2 12.3L2 12.1L2 11.9L2.2 11.7L4.1 11.2ZM7.1 6.1L7.2 6.4L7.1 6.7L6.9 6.9L6.7 7.1L6.4 7.2L6.1 7.1L5.9 6.9L5.6 6.5L4.9 5.3L4.9 5L5 4.9L5.3 4.9L6.9 5.9ZM17.9 11.2L17.7 10.5L17.4 9.7L17.1 9.1L16.7 8.4L16.2 7.8L15.6 7.3L14.9 6.9L14.3 6.5L13.5 6.3L12.8 6.2L12 6.1L11.2 6.2L10.5 6.3L9.7 6.5L9.1 6.9L8.4 7.3L7.8 7.8L7.3 8.4L6.9 9.1L6.5 9.7L6.3 10.5L6.2 11.2L6.1 12L6.2 12.8L6.3 13.5L6.5 14.3L6.9 14.9L7.3 15.6L7.8 16.2L8.4 16.7L9.1 17.1L9.7 17.4L10.5 17.7L11.2 17.9L12 17.9L12.8 17.9L13.5 17.7L14.3 17.4L14.9 17.1L15.6 16.7L16.2 16.2L16.7 15.6L17.1 14.9L17.4 14.3L17.7 13.5L17.9 12.8L17.9 12Z","jewels":[],"filigree":[]},"display":{"outline":"M19 12L18.9 12.9L18.8 13.8L18.5 14.7L18.1 15.5L17.6 16.3L16.9 16.9L16.3 17.6L15.5 18.1L14.7 18.5L13.8 18.8L12.9 18.9L12 19L11.1 18.9L10.2 18.8L9.3 18.5L8.5 18.1L7.7 17.6L7 16.9L6.5 16.3L5.9 15.5L5.5 14.7L5.2 13.8L5.1 12.9L5 12L5.1 11.1L5.2 10.2L5.5 9.3L5.9 8.5L6.5 7.7L7 7L7.7 6.5L8.5 5.9L9.3 5.5L10.2 5.2L11.1 5.1L12 5L12.9 5.1L13.8 5.2L14.7 5.5L15.5 5.9L16.3 6.5L16.9 7L17.6 7.7L18.1 8.5L18.5 9.3L18.8 10.2L18.9 11.1ZM12.7 4.4L12.5 4.6L12.3 4.8L12 4.8L11.7 4.8L11.5 4.6L11.3 4.4L11.2 4.1L11.5 3.2L11.7 2.2L11.9 2L12.1 2L12.3 2.2L12.8 4.1ZM17.9 7.1L17.6 7.2L17.3 7.1L17.1 6.9L16.9 6.7L16.8 6.4L16.9 6.1L17.1 5.9L17.4 5.6L18.7 4.9L19 4.9L19.1 5L19.1 5.3L18.1 6.9ZM19.6 12.7L19.4 12.5L19.2 12.3L19.1 12L19.2 11.7L19.4 11.5L19.6 11.3L19.9 11.2L20.8 11.5L21.8 11.7L22 11.9L22 12.1L21.8 12.3L19.9 12.8ZM16.9 17.9L16.8 17.6L16.9 17.3L17.1 17.1L17.3 16.9L17.6 16.8L17.9 16.9L18.1 17.1L18.4 17.4L19.1 18.7L19.1 19L19 19.1L18.7 19.1L17.1 18.1ZM11.3 19.6L11.5 19.4L11.7 19.2L12 19.1L12.3 19.2L12.5 19.4L12.7 19.6L12.8 19.9L12.5 20.8L12.3 21.8L12.1 22L11.9 22L11.7 21.8L11.2 19.9ZM6.1 16.9L6.4 16.8L6.7 16.9L6.9 17.1L7.1 17.3L7.2 17.6L7.1 17.9L6.9 18.1L6.5 18.4L5.3 19.1L5 19.1L4.9 19L4.9 18.7L5.9 17.1ZM4.4 11.3L4.6 11.5L4.8 11.7L4.8 12L4.8 12.3L4.6 12.5L4.4 12.7L4.1 12.8L3.2 12.5L2.2 12.3L2 12.1L2 11.9L2.2 11.7L4.1 11.2ZM7.1 6.1L7.2 6.4L7.1 6.7L6.9 6.9L6.7 7.1L6.4 7.2L6.1 7.1L5.9 6.9L5.6 6.5L4.9 5.3L4.9 5L5 4.9L5.3 4.9L6.9 5.9Z","body":"M19 12L18.9 12.9L18.8 13.8L18.5 14.7L18.1 15.5L17.6 16.3L16.9 16.9L16.3 17.6L15.5 18.1L14.7 18.5L13.8 18.8L12.9 18.9L12 19L11.1 18.9L10.2 18.8L9.3 18.5L8.5 18.1L7.7 17.6L7 16.9L6.5 16.3L5.9 15.5L5.5 14.7L5.2 13.8L5.1 12.9L5 12L5.1 11.1L5.2 10.2L5.5 9.3L5.9 8.5L6.5 7.7L7 7L7.7 6.5L8.5 5.9L9.3 5.5L10.2 5.2L11.1 5.1L12 5L12.9 5.1L13.8 5.2L14.7 5.5L15.5 5.9L16.3 6.5L16.9 7L17.6 7.7L18.1 8.5L18.5 9.3L18.8 10.2L18.9 11.1ZM10.8 8.2L11.1 7.8L11.5 7.5L12 7.4L12.5 7.5L12.9 7.8L13.2 8.2L13.3 8.7L13.3 15.3L13.2 15.8L12.9 16.2L12.5 16.5L12 16.6L11.5 16.5L11.1 16.2L10.8 15.8L10.7 15.3L10.7 8.7ZM8.2 13.2L7.8 12.9L7.5 12.5L7.4 12L7.5 11.5L7.8 11.1L8.2 10.8L8.7 10.7L15.3 10.7L15.8 10.8L16.2 11.1L16.5 11.5L16.6 12L16.5 12.5L16.2 12.9L15.8 13.2L15.3 13.3L8.7 13.3ZM12.7 4.4L12.5 4.6L12.3 4.8L12 4.8L11.7 4.8L11.5 4.6L11.3 4.4L11.2 4.1L11.5 3.2L11.7 2.2L11.9 2L12.1 2L12.3 2.2L12.8 4.1ZM17.9 7.1L17.6 7.2L17.3 7.1L17.1 6.9L16.9 6.7L16.8 6.4L16.9 6.1L17.1 5.9L17.4 5.6L18.7 4.9L19 4.9L19.1 5L19.1 5.3L18.1 6.9ZM19.6 12.7L19.4 12.5L19.2 12.3L19.1 12L19.2 11.7L19.4 11.5L19.6 11.3L19.9 11.2L20.8 11.5L21.8 11.7L22 11.9L22 12.1L21.8 12.3L19.9 12.8ZM16.9 17.9L16.8 17.6L16.9 17.3L17.1 17.1L17.3 16.9L17.6 16.8L17.9 16.9L18.1 17.1L18.4 17.4L19.1 18.7L19.1 19L19 19.1L18.7 19.1L17.1 18.1ZM11.3 19.6L11.5 19.4L11.7 19.2L12 19.1L12.3 19.2L12.5 19.4L12.7 19.6L12.8 19.9L12.5 20.8L12.3 21.8L12.1 22L11.9 22L11.7 21.8L11.2 19.9ZM6.1 16.9L6.4 16.8L6.7 16.9L6.9 17.1L7.1 17.3L7.2 17.6L7.1 17.9L6.9 18.1L6.5 18.4L5.3 19.1L5 19.1L4.9 19L4.9 18.7L5.9 17.1ZM4.4 11.3L4.6 11.5L4.8 11.7L4.8 12L4.8 12.3L4.6 12.5L4.4 12.7L4.1 12.8L3.2 12.5L2.2 12.3L2 12.1L2 11.9L2.2 11.7L4.1 11.2ZM7.1 6.1L7.2 6.4L7.1 6.7L6.9 6.9L6.7 7.1L6.4 7.2L6.1 7.1L5.9 6.9L5.6 6.5L4.9 5.3L4.9 5L5 4.9L5.3 4.9L6.9 5.9ZM17.9 11.2L17.7 10.5L17.4 9.7L17.1 9.1L16.7 8.4L16.2 7.8L15.6 7.3L14.9 6.9L14.3 6.5L13.5 6.3L12.8 6.2L12 6.1L11.2 6.2L10.5 6.3L9.7 6.5L9.1 6.9L8.4 7.3L7.8 7.8L7.3 8.4L6.9 9.1L6.5 9.7L6.3 10.5L6.2 11.2L6.1 12L6.2 12.8L6.3 13.5L6.5 14.3L6.9 14.9L7.3 15.6L7.8 16.2L8.4 16.7L9.1 17.1L9.7 17.4L10.5 17.7L11.2 17.9L12 17.9L12.8 17.9L13.5 17.7L14.3 17.4L14.9 17.1L15.6 16.7L16.2 16.2L16.7 15.6L17.1 14.9L17.4 14.3L17.7 13.5L17.9 12.8L17.9 12Z","jewels":[],"filigree":[["M18.4 12L18.4 12.8L18.2 13.7L18 14.5L17.6 15.2L17.1 15.9L16.6 16.6L15.9 17.1L15.2 17.6L14.5 18L13.7 18.2L12.8 18.4L12 18.4L11.2 18.4L10.3 18.2L9.5 18L8.8 17.6L8.1 17.1L7.4 16.6L6.9 15.9L6.4 15.2L6 14.5L5.8 13.7L5.6 12.8L5.5 12L5.6 11.2L5.8 10.3L6 9.5L6.4 8.8L6.9 8.1L7.4 7.4L8.1 6.9L8.8 6.4L9.5 6L10.3 5.8L11.2 5.6L12 5.5L12.8 5.6L13.7 5.8L14.5 6L15.2 6.4L15.9 6.9L16.6 7.4L17.1 8.1L17.6 8.8L18 9.5L18.2 10.3L18.4 11.2Z","dawnlight",0.5]]}}},"glyphs":{"dazed_warrior":{"std":[["M12 2.2L13 5.8L16.6 6.8L13 7.8L12 11.4L11 7.8L7.4 6.8L11 5.8ZM6.2 5.7L6.9 8.3L9.5 9L6.9 9.7L6.2 12.3L5.5 9.7L2.9 9L5.5 8.3ZM17.8 5.7L18.5 8.3L21.1 9L18.5 9.7L17.8 12.3L17.1 9.7L14.5 9L17.1 8.3Z","warrior",1.0,1]],"mic":[["M12 1.2L13.7 7.7L20.2 9.4L13.7 11.1L12 17.6L10.3 11.1L3.8 9.4L10.3 7.7Z","warrior",1.0,1],["M6.3 14.6L6.2 15.2L5.8 15.8L5.2 16.2L4.6 16.3L4 16.2L3.4 15.8L3 15.2L2.9 14.6L3 13.9L3.4 13.4L4 13L4.6 12.9L5.2 13L5.8 13.4L6.2 13.9ZM21.1 14.6L21 15.2L20.6 15.8L20.1 16.2L19.4 16.3L18.8 16.2L18.2 15.8L17.8 15.2L17.7 14.6L17.8 13.9L18.2 13.4L18.8 13L19.4 12.9L20.1 13L20.6 13.4L21 13.9Z","warrior",1.0,1]]},"dazed_rogue":{"std":[["M12 2.2L13 5.8L16.6 6.8L13 7.8L12 11.4L11 7.8L7.4 6.8L11 5.8ZM6.2 5.7L6.9 8.3L9.5 9L6.9 9.7L6.2 12.3L5.5 9.7L2.9 9L5.5 8.3ZM17.8 5.7L18.5 8.3L21.1 9L18.5 9.7L17.8 12.3L17.1 9.7L14.5 9L17.1 8.3Z","rogue",1.0,1]],"mic":[["M12 1.2L13.7 7.7L20.2 9.4L13.7 11.1L12 17.6L10.3 11.1L3.8 9.4L10.3 7.7Z","rogue",1.0,1],["M6.3 14.6L6.2 15.2L5.8 15.8L5.2 16.2L4.6 16.3L4 16.2L3.4 15.8L3 15.2L2.9 14.6L3 13.9L3.4 13.4L4 13L4.6 12.9L5.2 13L5.8 13.4L6.2 13.9ZM21.1 14.6L21 15.2L20.6 15.8L20.1 16.2L19.4 16.3L18.8 16.2L18.2 15.8L17.8 15.2L17.7 14.6L17.8 13.9L18.2 13.4L18.8 13L19.4 12.9L20.1 13L20.6 13.4L21 13.9Z","rogue",1.0,1]]},"frozen":{"std":[["M12 6.4L17.7 9.7L17.7 16.3L12 19.6L6.3 16.3L6.3 9.7Z","frostglass",0.45,1],["M12 6.4L17.7 9.7L17.7 16.3L12 19.6L6.3 16.3L6.3 9.7ZM7.2 10.2L7.2 15.8L12 18.5L16.8 15.8L16.8 10.2L12 7.5Z","rime",1,0],["M11.6 7.7L11.6 7.5L11.8 7.2L12.1 7.2L12.3 7.4L17 15.5L17.1 15.8L16.9 16.1L16.7 16.1L16.6 16.1L16.4 16L16.3 15.9L11.7 7.8ZM11.8 7.3L11.9 7.2L12.2 7.2L12.4 7.5L12.3 7.8L7.7 15.9L7.6 16L7.4 16.1L7.3 16.1L7.1 16.1L6.9 15.8L7 15.5L11.7 7.4ZM7.2 10.7L7 10.4L6.9 10.3L7 10.2L7 10L7.2 9.9L7.3 9.9L16.7 9.9L16.8 9.9L17 10L17.1 10.2L17.1 10.3L17.1 10.4L17 10.6L16.8 10.7L16.7 10.7Z","rime",0.6,0],["M7.5 3.3L9 4.2L8.9 5.9L7.4 5ZM10.5 2.3L11.6 3.7L10.9 5.3L9.8 3.9ZM13.5 2.3L14.2 3.9L13.1 5.3L12.4 3.7ZM16.4 3.3L16.6 5L15.1 5.9L15 4.2Z","rime",1.0,1]],"mic":[["M12 2.6L20.1 7.3L20.1 16.7L12 21.4L3.9 16.7L3.9 7.3Z","frostglass",0.7,1],["M12 2.6L20.1 7.3L20.1 16.7L12 21.4L3.9 16.7L3.9 7.3ZM6.1 8.6L6.1 15.4L12 18.8L17.9 15.4L17.9 8.6L12 5.2Z","rime",1,0]]},"snagged":{"std":[["M17.5 20.1L17.1 20.4L16.2 20.9L15.7 21.1L14.7 21.4L13.3 21.7L12.2 21.8L10.7 21.7L9.3 21.4L8.3 21.1L7.8 20.9L7.4 20.6L6.5 20.1L8.2 17.6L9.1 18.2L10.1 18.5L10.8 18.7L11.8 18.8L12.9 18.7L13.9 18.5L14.9 18.2L15.8 17.6Z","hunter",1.0,1],["M13.6 14L13.5 14.6L13.1 15.1L12.6 15.5L12 15.6L11.4 15.5L10.9 15.1L10.5 14.6L10.4 14L10.5 13.4L10.9 12.9L11.4 12.5L12 12.4L12.6 12.5L13.1 12.9L13.5 13.4Z","hunter",1.0,1]],"mic":[["M17.2 20.3L16.4 20.8L15 21.3L13.7 21.7L12.2 21.8L10.8 21.7L9.4 21.4L8.1 21L6.8 20.3L8.4 17.8L9.3 18.2L10.2 18.6L11.2 18.8L11.8 18.8L12.8 18.8L13.8 18.6L14.7 18.2L15.6 17.8Z","hunter",1.0,1],["M14.2 8.6L14 9.4L13.6 10.2L12.8 10.6L12 10.8L11.2 10.6L10.4 10.2L10 9.4L9.8 8.6L10 7.8L10.4 7L11.2 6.6L12 6.4L12.8 6.6L13.6 7L14 7.8Z","hunter",1.0,1]]},"hobbled_warrior":{"std":[["M19.7 18L18.7 19.2L18.1 19.7L16.8 20.5L15.4 21.2L14.7 21.4L13.9 21.6L13.2 21.7L12.4 21.8L11.6 21.8L10.8 21.7L10.1 21.6L9.3 21.4L8.6 21.2L7.9 20.9L7.2 20.5L5.9 19.7L5.3 19.2L4.3 18L6.8 16.1L7.5 16.8L8.3 17.5L8.8 17.8L9.2 18L10.2 18.4L11.2 18.6L12.3 18.6L13.3 18.5L13.8 18.4L14.8 18L15.7 17.5L16.1 17.2L16.5 16.8L17.2 16.1Z","warrior",1.0,1],["M16.1 15.5L15.9 16.1L15.5 16.5L15 16.7L14.4 16.5L14 16.1L13.8 15.5L14 14.9L14.4 14.5L15 14.4L15.5 14.5L15.9 14.9ZM14.7 16.3L14.6 16.9L14.2 17.3L13.6 17.5L13 17.3L12.6 16.9L12.4 16.3L12.6 15.8L13 15.3L13.6 15.2L14.2 15.3L14.6 15.8ZM13.2 16.6L13 17.2L12.6 17.6L12 17.8L11.4 17.6L11 17.2L10.8 16.6L11 16L11.4 15.6L12 15.4L12.6 15.6L13 16ZM11.6 16.3L11.4 16.9L11 17.3L10.4 17.5L9.8 17.3L9.4 16.9L9.3 16.3L9.4 15.8L9.8 15.3L10.4 15.2L11 15.3L11.4 15.8ZM10.2 15.5L10 16.1L9.6 16.5L9 16.7L8.5 16.5L8.1 16.1L7.9 15.5L8.1 14.9L8.5 14.5L9 14.4L9.6 14.5L10 14.9Z","warrior",1.0,1]],"mic":[["M19.5 18.3L19 18.9L17.9 19.8L17.3 20.3L16 21L14.6 21.5L13.1 21.7L12.4 21.8L10.9 21.7L10.2 21.6L9.4 21.5L8 21L6.7 20.3L6.1 19.8L5.5 19.4L4.5 18.3L6.2 16.9L7 17.7L7.9 18.4L8.4 18.7L9.4 19.2L10.6 19.5L11.1 19.6L11.7 19.6L12.9 19.6L14 19.3L15.1 18.9L16.1 18.4L16.6 18.1L17 17.7L17.8 16.9ZM16.8 16L16.1 16.7L15.7 16.9L14.9 17.5L14.5 17.7L13.6 18L13.2 18.1L12.2 18.2L11.8 18.2L10.8 18.1L9.9 17.8L9.5 17.7L9.1 17.5L8.3 16.9L7.9 16.7L7.2 16L8.9 14.6L9.6 15.2L10.4 15.7L11.2 15.9L11.8 16L12.8 15.9L13.6 15.7L14.4 15.2L15.1 14.6Z","warrior",1.0,1]]},"hobbled_rogue":{"std":[["M19.7 18L18.7 19.2L18.1 19.7L16.8 20.5L15.4 21.2L14.7 21.4L13.9 21.6L13.2 21.7L12.4 21.8L11.6 21.8L10.8 21.7L10.1 21.6L9.3 21.4L8.6 21.2L7.9 20.9L7.2 20.5L5.9 19.7L5.3 19.2L4.3 18L6.8 16.1L7.5 16.8L8.3 17.5L8.8 17.8L9.2 18L10.2 18.4L11.2 18.6L12.3 18.6L13.3 18.5L13.8 18.4L14.8 18L15.7 17.5L16.1 17.2L16.5 16.8L17.2 16.1Z","rogue",1.0,1],["M16.1 15.5L15.9 16.1L15.5 16.5L15 16.7L14.4 16.5L14 16.1L13.8 15.5L14 14.9L14.4 14.5L15 14.4L15.5 14.5L15.9 14.9ZM14.7 16.3L14.6 16.9L14.2 17.3L13.6 17.5L13 17.3L12.6 16.9L12.4 16.3L12.6 15.8L13 15.3L13.6 15.2L14.2 15.3L14.6 15.8ZM13.2 16.6L13 17.2L12.6 17.6L12 17.8L11.4 17.6L11 17.2L10.8 16.6L11 16L11.4 15.6L12 15.4L12.6 15.6L13 16ZM11.6 16.3L11.4 16.9L11 17.3L10.4 17.5L9.8 17.3L9.4 16.9L9.3 16.3L9.4 15.8L9.8 15.3L10.4 15.2L11 15.3L11.4 15.8ZM10.2 15.5L10 16.1L9.6 16.5L9 16.7L8.5 16.5L8.1 16.1L7.9 15.5L8.1 14.9L8.5 14.5L9 14.4L9.6 14.5L10 14.9Z","rogue",1.0,1]],"mic":[["M19.5 18.3L19 18.9L17.9 19.8L17.3 20.3L16 21L14.6 21.5L13.1 21.7L12.4 21.8L10.9 21.7L10.2 21.6L9.4 21.5L8 21L6.7 20.3L6.1 19.8L5.5 19.4L4.5 18.3L6.2 16.9L7 17.7L7.9 18.4L8.4 18.7L9.4 19.2L10.6 19.5L11.1 19.6L11.7 19.6L12.9 19.6L14 19.3L15.1 18.9L16.1 18.4L16.6 18.1L17 17.7L17.8 16.9ZM16.8 16L16.1 16.7L15.7 16.9L14.9 17.5L14.5 17.7L13.6 18L13.2 18.1L12.2 18.2L11.8 18.2L10.8 18.1L9.9 17.8L9.5 17.7L9.1 17.5L8.3 16.9L7.9 16.7L7.2 16L8.9 14.6L9.6 15.2L10.4 15.7L11.2 15.9L11.8 16L12.8 15.9L13.6 15.7L14.4 15.2L15.1 14.6Z","rogue",1.0,1]]},"withering":{"std":[["M18.1 4.1L18.6 3.9L19.1 3.8L19.5 4L19.9 4.3L20.1 4.7L20.2 5.2L20 5.7L19.3 6.5L18.9 7L18 7.6L17.1 8.2L16.2 8.7L15.2 9.1L15.1 9.1L15 9.1L14.9 8.9L14.9 8.8L16 7.6L16.9 6.5L17.1 6.1L17.5 5.3L17.8 4.5ZM17.2 6.7L17.3 6.5L17.4 6.3L17.6 6.2L17.8 6.2L18 6.3L18.1 6.4L18.2 6.5L18.4 7.6L18.8 8.8L18.8 8.9L18.7 9L18.5 9L18 8.3L17.5 7.4L17.2 6.8ZM19.9 18.1L20.1 18.6L20.2 19.1L20 19.5L19.7 19.9L19.3 20.1L18.8 20.2L18.3 20L17.4 19.3L17.1 18.9L16.4 18L15.8 17.1L15.3 16.2L14.9 15.2L14.9 15.1L14.9 15L15.1 14.9L15.2 14.9L16.4 16L17.5 16.9L17.9 17.1L18.7 17.5L19.5 17.8ZM17.4 17.2L17.5 17.3L17.7 17.4L17.8 17.6L17.8 17.8L17.7 18L17.6 18.1L17.4 18.2L16.4 18.4L15.2 18.8L15.1 18.8L15 18.7L15 18.5L15.7 18L16.6 17.5L17.2 17.2ZM5.8 19.9L5.4 20.1L4.9 20.2L4.5 20L4.1 19.7L3.9 19.3L3.8 18.8L4 18.3L4.7 17.4L5.1 17.1L6 16.4L6.9 15.8L7.8 15.3L8.8 14.9L8.9 14.9L9 14.9L9.1 15.1L9.1 15.2L8 16.4L7.1 17.5L6.9 17.9L6.5 18.7L6.2 19.5ZM6.8 17.4L6.7 17.5L6.6 17.7L6.4 17.8L6.2 17.8L6 17.7L5.9 17.6L5.8 17.4L5.6 16.4L5.2 15.2L5.2 15.1L5.3 15L5.5 15L6 15.7L6.5 16.6L6.8 17.2ZM4.1 5.8L3.9 5.4L3.8 4.9L4 4.5L4.3 4.1L4.7 3.9L5.2 3.8L5.7 4L6.5 4.7L7 5.1L7.6 6L8.2 6.9L8.7 7.8L9.1 8.8L9.1 8.9L9.1 9L8.9 9.1L8.8 9.1L7.6 8L6.5 7.1L6.1 6.9L5.3 6.5L4.5 6.2ZM6.7 6.8L6.5 6.7L6.3 6.6L6.2 6.4L6.2 6.2L6.3 6L6.4 5.9L6.5 5.8L7.6 5.6L8.8 5.2L8.9 5.2L9 5.3L9 5.5L8.3 6L7.4 6.5L6.8 6.8Z","rogue",1.0,1],["M11.4 10.1L11.5 9.8L11.8 9.7L12 9.7L12.2 9.7L12.5 9.8L12.6 10.1L12.7 10.3L12.7 13.7L12.6 13.9L12.5 14.2L12.2 14.3L12 14.3L11.8 14.3L11.5 14.2L11.4 13.9L11.3 13.7L11.3 10.3ZM10.1 12.6L9.8 12.5L9.7 12.2L9.7 12L9.7 11.8L9.8 11.5L10.1 11.4L10.3 11.3L13.7 11.3L13.9 11.4L14.2 11.5L14.3 11.8L14.3 12L14.3 12.2L14.2 12.5L13.9 12.6L13.7 12.7L10.3 12.7Z","withered",1.0,1]],"mic":[["M18.1 3.6L18.8 3.5L19.4 3.6L20 4L20.4 4.6L20.5 5.2L20.4 5.9L20 6.4L15.8 8.7L15.5 8.7L15.4 8.6L15.3 8.5L15.3 8.2L17.6 4ZM20.4 18.1L20.5 18.8L20.4 19.4L20 20L19.4 20.4L18.8 20.5L18.1 20.4L17.6 20L15.3 15.8L15.3 15.5L15.4 15.4L15.5 15.3L15.8 15.3L20 17.6ZM5.9 20.4L5.2 20.5L4.6 20.4L4 20L3.6 19.4L3.5 18.8L3.6 18.1L4 17.6L8.2 15.3L8.5 15.3L8.6 15.4L8.7 15.5L8.7 15.8L6.4 20ZM3.6 5.9L3.5 5.2L3.6 4.6L4 4L4.6 3.6L5.2 3.5L5.9 3.6L6.4 4L8.7 8.2L8.7 8.3L8.7 8.5L8.6 8.6L8.5 8.7L8.3 8.7L8.2 8.7L4 6.4Z","rogue",1.0,1],["M14.2 12L14 12.8L13.6 13.6L12.8 14L12 14.2L11.2 14L10.4 13.6L10 12.8L9.8 12L10 11.2L10.4 10.4L11.2 10L12 9.8L12.8 10L13.6 10.4L14 11.2Z","withered",1.0,1]]},"unbound":{"std":[["M14.8 14.3L14.6 14.4L14.4 14.4L14.1 14.4L13.9 14.2L13.8 14L13.8 13.8L15.6 7.9L16.3 6L16.7 5.2L17.1 4.5L17.5 4.1L18 3.9L18.5 3.9L18.9 4.1L19.3 4.5L19.5 5L19.5 5.5L19.3 5.9L18.8 6.5L18 7.9L17.1 9.6L15.6 13L15.3 13.6ZM14.8 14.3L14.5 14.4L14.3 14.4L14 14.3L13.8 14.2L13.8 13.9L13.8 13.7L14.1 12.9L15 11.8L18 8.7L19 7.5L19.4 7L19.8 6.7L20.3 6.5L20.8 6.6L21.2 6.8L21.5 7.2L21.6 7.7L21.6 8.2L21.4 8.6L20.8 9.2L19.5 10.3L15.6 13.4ZM14.7 14.4L14.4 14.4L14.2 14.4L14 14.3L13.8 14.1L13.8 13.8L13.8 13.6L14.3 12.9L14.8 12.5L15.5 12.2L18.4 11.4L19.1 11.2L19.6 10.9L20 10.6L20.4 10.2L20.8 9.9L21.2 9.8L21.7 9.9L22 10.2L22.3 10.6L22.4 11L22.2 11.5L22 11.8L21.4 12.4L20.6 12.8L19.7 13.1L18.9 13.3L15.9 13.8L15.5 13.9L15.1 14ZM9 14.1L8.4 13L6.9 9.6L6 7.9L5.2 6.5L4.7 5.9L4.5 5.5L4.5 5L4.7 4.5L5.1 4.1L5.5 3.9L6 3.9L6.5 4.1L6.9 4.5L7.3 5.2L7.7 6L8.4 7.9L10.2 13.8L10.2 14L10.1 14.2L9.9 14.4L9.6 14.4L9.4 14.4L9.2 14.3ZM9.1 14.2L8.4 13.4L4.5 10.3L3.2 9.2L2.6 8.6L2.4 8.2L2.4 7.7L2.5 7.2L2.8 6.8L3.2 6.6L3.7 6.5L4.2 6.7L4.5 7L5 7.5L6 8.7L9 11.8L9.9 12.9L10.2 13.7L10.2 13.9L10.2 14.2L10 14.3L9.7 14.4L9.5 14.4L9.2 14.3ZM9.1 14.2L8.9 14L8.5 13.9L8.1 13.8L5.1 13.3L4.3 13.1L3.4 12.8L2.6 12.4L2 11.8L1.8 11.5L1.6 11L1.7 10.6L2 10.2L2.3 9.9L2.8 9.8L3.2 9.9L3.6 10.2L4 10.6L4.4 10.9L4.9 11.2L5.6 11.4L8.5 12.2L9.2 12.5L9.7 12.9L10.2 13.6L10.2 13.8L10.2 14.1L10 14.3L9.8 14.4L9.6 14.4L9.3 14.4Z","priest",1.0,1],["M15.9 13.8L15.8 14.3L15.5 14.7L15.1 15L14.6 15.1L14.1 15L13.7 14.7L13.4 14.3L13.3 13.8L13.4 13.3L13.7 12.9L14.1 12.6L14.6 12.5L15.1 12.6L15.5 12.9L15.8 13.3ZM10.7 13.8L10.6 14.3L10.3 14.7L9.9 15L9.4 15.1L8.9 15L8.5 14.7L8.2 14.3L8.1 13.8L8.2 13.3L8.5 12.9L8.9 12.6L9.4 12.5L9.9 12.6L10.3 12.9L10.6 13.3Z","amber",1.0,1]],"mic":[["M15.1 14.6L14.7 14.8L14.3 14.8L13.9 14.7L13.6 14.4L13.4 14.1L13.4 13.7L15 8.8L15.8 6.9L16.1 6L17 4.4L17.5 4L18.1 3.7L18.8 3.7L19.4 4L19.8 4.5L20.1 5.1L20.1 5.8L19.8 6.4L19.3 6.9L18.5 8.2L17.6 9.9L15.9 13.2L15.6 13.8ZM14.9 14.7L14.5 14.8L14.1 14.8L13.8 14.6L13.5 14.3L13.4 13.9L13.4 13.5L13.6 13.2L14 12.7L14.6 12.2L15.2 11.8L17.3 10.8L18.6 10.2L19.1 9.9L19.5 9.5L19.9 9.1L20.4 8.7L21 8.6L21.6 8.7L22.1 9.1L22.5 9.6L22.6 10.2L22.5 10.8L22.1 11.3L21.5 11.9L20.7 12.4L19.9 12.8L19 13.1L16.8 13.7L15.7 14.1L15.4 14.2ZM8.7 14.2L8.1 13.2L6.4 9.9L5.5 8.2L4.7 6.9L4.2 6.4L3.9 5.8L3.9 5.1L4.2 4.5L4.6 4L5.2 3.7L5.9 3.7L6.5 4L7 4.4L7.8 6L8.2 6.9L9.3 9.8L10.6 13.7L10.6 14.1L10.4 14.4L10.1 14.7L9.7 14.8L9.3 14.8L8.9 14.6ZM8.8 14.4L8.6 14.2L7.8 13.9L5.8 13.3L4.1 12.8L3.3 12.4L2.5 11.9L1.9 11.3L1.5 10.8L1.4 10.2L1.5 9.6L1.9 9.1L2.4 8.7L3 8.6L3.6 8.7L4.1 9.1L4.5 9.5L4.9 9.9L5.4 10.2L6.7 10.8L8.8 11.8L9.4 12.2L10 12.7L10.4 13.2L10.6 13.5L10.6 13.9L10.5 14.3L10.2 14.6L9.9 14.8L9.5 14.8L9.1 14.7Z","priest",1.0,1]]},"kindled":{"std":[["M15.4 12L15.3 12.9L14.9 13.7L14.4 14.4L13.7 14.9L12.9 15.3L12 15.4L11.1 15.3L10.3 14.9L9.6 14.4L9.1 13.7L8.7 12.9L8.6 12L8.7 11.1L9.1 10.3L9.6 9.6L10.3 9.1L11.1 8.7L12 8.6L12.9 8.7L13.7 9.1L14.4 9.6L14.9 10.3L15.3 11.1Z","mage",0.7,1],["M13.9 12L13.8 12.7L13.3 13.3L12.7 13.8L12 13.9L11.3 13.8L10.7 13.3L10.2 12.7L10.1 12L10.2 11.3L10.7 10.7L11.3 10.2L12 10.1L12.7 10.2L13.3 10.7L13.8 11.3Z","starcore",1,0],["M12 1.6L13.5 4.4L12 7.2L10.5 4.4ZM21.9 8.8L19.7 11.1L16.6 10.5L18.8 8.2ZM18.1 20.4L15.2 19L14.8 15.9L17.7 17.3ZM5.9 20.4L6.3 17.3L9.2 15.9L8.8 19ZM2.1 8.8L5.2 8.2L7.4 10.5L4.3 11.1Z","starcore",1.0,1]],"mic":[["M12 3L14.4 9.1L20.9 9.5L15.9 13.7L17.5 20L12 16.5L6.5 20L8.1 13.7L3.1 9.5L9.6 9.1Z","starcore",1.0,1],["M13.9 12.4L13.8 13.1L13.3 13.7L12.7 14.2L12 14.3L11.3 14.2L10.7 13.7L10.2 13.1L10.1 12.4L10.2 11.7L10.7 11.1L11.3 10.6L12 10.5L12.7 10.6L13.3 11.1L13.8 11.7Z","mage",1,0]]},"moonward":{"std":[["M20 12L19.9 13L19.7 14.1L19.4 15.1L18.9 16L18.4 16.9L17.7 17.7L16.9 18.4L16 18.9L15.1 19.4L14.1 19.7L13 19.9L12 20L11 19.9L9.9 19.7L8.9 19.4L8 18.9L7.1 18.4L6.3 17.7L5.7 16.9L5.1 16L4.6 15.1L4.3 14.1L4.1 13L4 12L4.1 11L4.3 9.9L4.6 8.9L5.1 8L5.7 7.1L6.3 6.3L7.1 5.7L8 5.1L8.9 4.6L9.9 4.3L11 4.1L12 4L13 4.1L14.1 4.3L15.1 4.6L16 5.1L16.9 5.7L17.7 6.3L18.4 7.1L18.9 8L19.4 8.9L19.7 9.9L19.9 11Z","veil",0.12,0],["M13.2 2.7L14.2 2.9L14.8 3L15.7 3.4L15.3 4.3L14.1 3.9L13 3.7ZM17.7 4.5L18.6 5.3L19.5 6.3L18.7 6.9L18 6.1L17.1 5.3ZM20.6 8.3L21 9.2L21.1 9.8L21.3 10.8L20.3 11L20.1 9.9L19.7 8.7ZM21.3 13.2L21.1 14.2L21 14.8L20.6 15.7L19.7 15.3L20.1 14.1L20.3 13ZM19.5 17.7L18.7 18.6L17.7 19.5L17.1 18.7L17.8 18.1L18.3 17.6L18.7 17.1ZM15.7 20.6L14.8 21L14.2 21.1L13.2 21.3L13 20.3L14.1 20.1L15.3 19.7ZM10.8 21.3L9.8 21.1L9.2 21L8.3 20.6L8.7 19.7L9.9 20.1L11 20.3ZM6.3 19.5L5.4 18.7L4.5 17.7L5.3 17.1L5.9 17.8L6.4 18.3L6.9 18.7ZM3.4 15.7L3 14.8L2.9 14.2L2.7 13.2L3.7 13L3.9 14.1L4.3 15.3ZM2.7 10.8L2.9 9.8L3 9.2L3.4 8.3L4.3 8.7L3.9 9.9L3.7 11ZM4.5 6.3L5.3 5.4L6.3 4.5L6.9 5.3L6.2 5.9L5.7 6.4L5.3 6.9ZM8.3 3.4L9.2 3L9.8 2.9L10.8 2.7L11 3.7L9.9 3.9L8.7 4.3Z","ward",1.0,1],["M12 4.8L13.3 6.6L12 8.4L10.7 6.6ZM18.2 15.6L16 15.8L15.1 13.8L17.3 13.6ZM5.8 15.6L6.7 13.6L8.9 13.8L8 15.8Z","ward",1.0,1]],"mic":[["M13.7 2.4L14.6 2.5L15.1 2.7L15.7 2.9L16.8 3.4L17.3 3.7L18 4.3L18.9 5L19.5 5.7L17.7 7.2L17.1 6.6L16.5 6.2L15.8 5.7L15.2 5.3L14.4 5L13.3 4.7ZM21.2 8.7L21.4 9.2L21.6 10.1L21.7 10.7L21.8 11.8L21.8 12.7L21.7 13.6L21.5 14.5L21.2 15.3L18.9 14.5L19.3 13.4L19.4 12.1L19.3 10.8L19.2 10.1L18.9 9.5ZM19.5 18.3L18.9 19L18 19.7L17.3 20.3L16.8 20.6L15.7 21.1L15.1 21.3L14.6 21.5L13.7 21.6L13.3 19.3L14.4 19L15.2 18.7L15.8 18.4L16.5 17.9L17.1 17.4L17.7 16.8ZM10.3 21.6L9.4 21.5L8.9 21.3L8.3 21.1L7.2 20.6L6.7 20.3L6 19.7L5.1 19L4.5 18.3L6.3 16.8L7 17.4L7.5 17.9L8.2 18.4L8.8 18.7L9.6 19L10.7 19.3ZM2.8 15.3L2.6 14.8L2.4 13.9L2.3 13.3L2.2 12.2L2.2 11.3L2.3 10.4L2.5 9.5L2.8 8.7L5 9.5L4.7 10.6L4.6 11.9L4.7 13.2L4.8 13.9L5 14.5ZM4.5 5.7L5.1 5L6 4.3L6.7 3.7L7.2 3.4L8.3 2.9L8.9 2.7L9.4 2.5L10.3 2.4L10.7 4.7L9.6 5L8.8 5.3L8.2 5.7L7.5 6.2L7 6.6L6.3 7.2Z","ward",1.0,1],["M12 9.4L13.8 12L12 14.6L10.2 12Z","ward",1.0,1]]},"empowerment":{"std":[["M21.9 12L21.8 12.4L21.4 12.8L21 12.9L20.6 12.8L20.2 12.4L20.1 12L20.2 11.6L20.6 11.2L21 11.1L21.4 11.2L21.8 11.6ZM21.2 15.4L21.1 15.9L20.8 16.2L20.3 16.3L19.9 16.2L19.5 15.9L19.4 15.4L19.5 15L19.9 14.7L20.3 14.5L20.8 14.7L21.1 15ZM19.3 18.4L19.1 18.8L18.8 19.1L18.4 19.3L17.9 19.1L17.6 18.8L17.5 18.4L17.6 17.9L17.9 17.6L18.4 17.5L18.8 17.6L19.1 17.9ZM16.3 20.3L16.2 20.8L15.9 21.1L15.4 21.2L15 21.1L14.7 20.8L14.5 20.3L14.7 19.9L15 19.5L15.4 19.4L15.9 19.5L16.2 19.9ZM12.9 21L12.8 21.4L12.4 21.8L12 21.9L11.6 21.8L11.2 21.4L11.1 21L11.2 20.6L11.6 20.2L12 20.1L12.4 20.2L12.8 20.6ZM9.5 20.3L9.3 20.8L9 21.1L8.6 21.2L8.1 21.1L7.8 20.8L7.7 20.3L7.8 19.9L8.1 19.5L8.6 19.4L9 19.5L9.3 19.9ZM6.5 18.4L6.4 18.8L6.1 19.1L5.6 19.3L5.2 19.1L4.9 18.8L4.7 18.4L4.9 17.9L5.2 17.6L5.6 17.5L6.1 17.6L6.4 17.9ZM4.6 15.4L4.5 15.9L4.1 16.2L3.7 16.3L3.2 16.2L2.9 15.9L2.8 15.4L2.9 15L3.2 14.7L3.7 14.5L4.1 14.7L4.5 15ZM3.9 12L3.8 12.4L3.5 12.8L3 12.9L2.5 12.8L2.2 12.4L2.1 12L2.2 11.6L2.5 11.2L3 11.1L3.5 11.2L3.8 11.6ZM4.6 8.6L4.5 9L4.1 9.3L3.7 9.5L3.2 9.3L2.9 9L2.8 8.6L2.9 8.1L3.2 7.8L3.7 7.7L4.1 7.8L4.5 8.1ZM6.5 5.6L6.4 6.1L6.1 6.4L5.6 6.5L5.2 6.4L4.9 6.1L4.7 5.6L4.9 5.2L5.2 4.9L5.6 4.7L6.1 4.9L6.4 5.2ZM9.5 3.7L9.3 4.1L9 4.5L8.6 4.6L8.1 4.5L7.8 4.1L7.7 3.7L7.8 3.2L8.1 2.9L8.6 2.8L9 2.9L9.3 3.2ZM12.9 3L12.8 3.5L12.4 3.8L12 3.9L11.6 3.8L11.2 3.5L11.1 3L11.2 2.5L11.6 2.2L12 2.1L12.4 2.2L12.8 2.5ZM16.3 3.7L16.2 4.1L15.9 4.5L15.4 4.6L15 4.5L14.7 4.1L14.5 3.7L14.7 3.2L15 2.9L15.4 2.8L15.9 2.9L16.2 3.2ZM19.3 5.6L19.1 6.1L18.8 6.4L18.4 6.5L17.9 6.4L17.6 6.1L17.5 5.6L17.6 5.2L17.9 4.9L18.4 4.7L18.8 4.9L19.1 5.2ZM21.2 8.6L21.1 9L20.8 9.3L20.3 9.5L19.9 9.3L19.5 9L19.4 8.6L19.5 8.1L19.9 7.8L20.3 7.7L20.8 7.8L21.1 8.1Z","mage",1.0,1],["M12 7L12.6 10.7L14.1 9.9L13.3 11.4L17 12L13.3 12.6L14.1 14.1L12.6 13.3L12 17L11.4 13.3L9.9 14.1L10.7 12.6L7 12L10.7 11.4L9.9 9.9L11.4 10.7Z","mage",1.0,1]],"mic":[["M21.8 12L21.6 12.8L21 13.4L20.2 13.6L19.4 13.4L18.8 12.8L18.6 12L18.8 11.2L19.4 10.6L20.2 10.4L21 10.6L21.6 11.2ZM19.4 17.8L19.2 18.6L18.6 19.2L17.8 19.4L17 19.2L16.4 18.6L16.2 17.8L16.4 17L17 16.4L17.8 16.2L18.6 16.4L19.2 17ZM13.6 20.2L13.4 21L12.8 21.6L12 21.8L11.2 21.6L10.6 21L10.4 20.2L10.6 19.4L11.2 18.8L12 18.6L12.8 18.8L13.4 19.4ZM7.8 17.8L7.6 18.6L7 19.2L6.2 19.4L5.4 19.2L4.8 18.6L4.6 17.8L4.8 17L5.4 16.4L6.2 16.2L7 16.4L7.6 17ZM5.4 12L5.2 12.8L4.6 13.4L3.8 13.6L3 13.4L2.4 12.8L2.2 12L2.4 11.2L3 10.6L3.8 10.4L4.6 10.6L5.2 11.2ZM7.8 6.2L7.6 7L7 7.6L6.2 7.8L5.4 7.6L4.8 7L4.6 6.2L4.8 5.4L5.4 4.8L6.2 4.6L7 4.8L7.6 5.4ZM13.6 3.8L13.4 4.6L12.8 5.2L12 5.4L11.2 5.2L10.6 4.6L10.4 3.8L10.6 3L11.2 2.4L12 2.2L12.8 2.4L13.4 3ZM19.4 6.2L19.2 7L18.6 7.6L17.8 7.8L17 7.6L16.4 7L16.2 6.2L16.4 5.4L17 4.8L17.8 4.6L18.6 4.8L19.2 5.4Z","mage",1.0,1]]},"barrier":{"std":[["M21.8 12L21.7 13.3L21.5 14.5L21.1 15.8L20.5 16.9L19.8 18L18.9 18.9L18 19.8L16.9 20.5L15.8 21.1L14.5 21.5L13.3 21.7L12 21.8L10.7 21.7L9.5 21.5L8.2 21.1L7.1 20.5L6 19.8L5.1 18.9L4.2 18L3.5 16.9L3 15.8L2.5 14.5L2.3 13.3L2.2 12L2.3 10.7L2.5 9.5L3 8.2L3.5 7.1L4.2 6L5.1 5.1L6 4.2L7.1 3.5L8.2 3L9.5 2.5L10.7 2.3L12 2.2L13.3 2.3L14.5 2.5L15.8 3L16.9 3.5L18 4.2L18.9 5.1L19.8 6L20.5 7.1L21.1 8.2L21.5 9.5L21.7 10.7ZM20.5 10.9L20.3 9.8L19.9 8.7L19.4 7.7L18.8 6.8L18.1 5.9L17.2 5.2L16.3 4.5L15.3 4L14.2 3.7L13.1 3.5L12 3.4L10.9 3.5L9.8 3.7L8.7 4L7.7 4.5L6.8 5.2L5.9 5.9L5.2 6.8L4.5 7.7L4 8.7L3.7 9.8L3.5 10.9L3.4 12L3.5 13.1L3.7 14.2L4 15.3L4.5 16.3L5.2 17.2L5.9 18.1L6.8 18.8L7.7 19.4L8.7 19.9L9.8 20.3L10.9 20.5L12 20.6L13.1 20.5L14.2 20.3L15.3 19.9L16.3 19.4L17.2 18.8L18.1 18.1L18.8 17.2L19.4 16.3L19.9 15.3L20.3 14.2L20.5 13.1L20.6 12Z","warrior",1,1],["M11.4 4.1L12.6 4.1L12.6 5.6L11.4 5.6ZM15.5 4.9L16.4 5.4L15.7 6.7L14.7 6.2ZM18.6 7.6L19.1 8.5L17.8 9.3L17.3 8.3ZM19.9 11.4L19.9 12.6L18.4 12.6L18.4 11.4ZM19.1 15.5L18.6 16.4L17.3 15.7L17.8 14.7ZM16.4 18.6L15.5 19.1L14.7 17.8L15.7 17.3ZM12.6 19.9L11.4 19.9L11.4 18.4L12.6 18.4ZM8.5 19.1L7.6 18.6L8.3 17.3L9.3 17.8ZM5.4 16.4L4.9 15.5L6.2 14.7L6.7 15.7ZM4.1 12.6L4.1 11.4L5.6 11.4L5.6 12.6ZM4.9 8.5L5.4 7.6L6.7 8.3L6.2 9.3ZM7.6 5.4L8.5 4.9L9.3 6.2L8.3 6.7Z","warrior",1.0,1],["M9.3 9.5L9.4 9.3L9.6 9.2L14.4 9.2L14.6 9.3L14.7 9.4L14.7 12L14.6 12.7L14.4 13.3L14 14.2L13.5 15L12.8 15.7L12 16.3L11.2 15.7L10.5 15L10.1 14.4L9.7 13.6L9.4 13L9.3 12Z","warrior",1.0,1]],"mic":[["M21.8 12L21.7 13.3L21.5 14.5L21.1 15.8L20.5 16.9L19.8 18L18.9 18.9L18 19.8L16.9 20.5L15.8 21.1L14.5 21.5L13.3 21.7L12 21.8L10.7 21.7L9.5 21.5L8.2 21.1L7.1 20.5L6 19.8L5.1 18.9L4.2 18L3.5 16.9L3 15.8L2.5 14.5L2.3 13.3L2.2 12L2.3 10.7L2.5 9.5L3 8.2L3.5 7.1L4.2 6L5.1 5.1L6 4.2L7.1 3.5L8.2 3L9.5 2.5L10.7 2.3L12 2.2L13.3 2.3L14.5 2.5L15.8 3L16.9 3.5L18 4.2L18.9 5.1L19.8 6L20.5 7.1L21.1 8.2L21.5 9.5L21.7 10.7ZM10.6 6.9L13.4 6.9L12 9.9ZM17.1 10.6L17.1 13.4L14.1 12ZM13.4 17.1L10.6 17.1L12 14.1ZM6.9 13.4L6.9 10.6L9.9 12ZM18.7 11.1L18.6 10.2L18.3 9.4L17.9 8.6L17.4 7.9L16.8 7.2L16.1 6.6L15.4 6.1L14.6 5.7L13.8 5.4L12.9 5.3L12 5.2L11.1 5.3L10.2 5.4L9.4 5.7L8.6 6.1L7.9 6.6L7.2 7.2L6.6 7.9L6.1 8.6L5.7 9.4L5.4 10.2L5.3 11.1L5.2 12L5.3 12.9L5.4 13.8L5.7 14.6L6.1 15.4L6.6 16.1L7.2 16.8L7.9 17.4L8.6 17.9L9.4 18.3L10.2 18.6L11.1 18.7L12 18.8L12.9 18.7L13.8 18.6L14.6 18.3L15.4 17.9L16.1 17.4L16.8 16.8L17.4 16.1L17.9 15.4L18.3 14.6L18.6 13.8L18.7 12.9L18.8 12Z","warrior",1,1]]},"moon_dew":{"std":[["M19.5 11L19.7 11.7L19.8 12.4L19.8 13.1L19.8 13.9L19.7 14.6L19.5 15.3L19.3 16L19 16.6L18.6 17.3L18.2 17.9L17.8 18.4L17.3 18.9L16.7 19.4L16.1 19.8L15.5 20.2L14.8 20.5L14.1 20.7L13.4 20.9L12.7 21L11.3 21L10.6 20.9L9.9 20.7L9.2 20.5L7.9 19.8L7.3 19.4L6.7 18.9L6.2 18.4L5.8 17.9L5.3 17.3L5 16.6L4.7 16L4.5 15.3L4.3 14.6L4.2 13.9L4.2 13.1L4.3 11.7L4.5 11L4.9 11.9L5.3 12.8L5.6 13.2L6.2 14L6.9 14.7L7.3 15L8.2 15.6L9.1 16L9.5 16.2L10.5 16.5L11 16.5L12 16.6L13 16.5L13.5 16.5L14.5 16.2L14.9 16L15.8 15.6L16.7 15L17.1 14.7L17.8 14L18.4 13.2L18.7 12.8L19.1 11.9Z","dew",1.0,1],["M12 8.1L13.2 9.7L13.5 10.2L13.7 10.7L13.8 11.1L13.9 11.5L13.8 12.5L13.5 13.1L13.1 13.4L12.6 13.6L12 13.7L11.4 13.6L10.9 13.4L10.5 13.1L10.2 12.5L10.1 11.5L10.2 11.1L10.3 10.7L10.5 10.2L10.8 9.7Z","dew",1.0,1]],"mic":[["M19.8 10.3L20 11L20.1 11.8L20.2 12.6L20.2 13.3L20.1 14.1L19.9 14.9L19.7 15.6L19.4 16.3L19.1 17L18.6 17.6L18.1 18.2L17.6 18.8L17 19.3L16.4 19.7L15.7 20.1L15 20.4L14.3 20.7L13.5 20.9L12.8 21L12 21L11.2 21L10.5 20.9L9.7 20.7L9 20.4L8.3 20.1L7.6 19.7L7 19.3L6.4 18.8L5.9 18.2L5.4 17.6L5 17L4.6 16.3L4.3 15.6L4.1 14.9L3.9 14.1L3.8 13.3L3.8 12.6L3.9 11.8L4 11L4.2 10.3L4.4 10.8L4.8 11.7L5.3 12.6L6 13.4L6.8 14.1L7.2 14.4L8 15L9 15.4L9.5 15.6L10.5 15.8L11.5 16L12 16L12.5 16L13.5 15.8L14.5 15.6L15 15.4L16 15L16.8 14.4L17.2 14.1L18 13.4L18.6 12.6L19.2 11.7L19.6 10.8Z","dew",1.0,1],["M12 6.6L13.5 8.5L13.9 9.1L14.2 9.6L14.3 10.1L14.4 10.6L14.3 11.7L13.9 12.3L13.4 12.8L12.7 13L12 13L11.3 13L10.6 12.8L10.1 12.3L9.7 11.7L9.6 10.6L9.7 10.1L9.8 9.6L10.1 9.1L10.5 8.5Z","dew",1.0,1]]},"in_combat":{"std":[["M7.3 4.5L7.1 4.3L7 4.1L6.9 3.8L7 3.5L7.1 3.3L7.3 3.1L7.6 3.1L16.4 3.1L16.7 3.1L16.9 3.3L17.1 3.5L17.1 3.8L17.1 4.1L16.9 4.3L16.7 4.5L16.4 4.5L7.6 4.5ZM7.3 20.9L7.1 20.7L7 20.5L6.9 20.2L7 19.9L7.1 19.7L7.3 19.6L7.6 19.5L16.4 19.5L16.7 19.6L16.9 19.7L17.1 19.9L17.1 20.2L17.1 20.5L16.9 20.7L16.7 20.9L16.4 20.9L7.6 20.9Z","muted",1.0,1],["M8 5.3L8 5L8.1 4.8L8.3 4.7L8.5 4.6L8.8 4.6L9 4.7L9.1 4.9L12 11L14.9 4.9L15 4.7L15.2 4.6L15.5 4.6L15.7 4.7L15.9 4.8L16 5L16 5.3L15.9 5.5L12 12.2L8.1 5.5ZM9 19.3L8.8 19.4L8.5 19.4L8.3 19.3L8.1 19.2L8 19L8 18.7L8.1 18.5L12 11.8L15.9 18.5L16 18.7L16 19L15.9 19.2L15.7 19.3L15.5 19.4L15.2 19.4L15 19.3L14.9 19.1L12 13L9.1 19.1Z","muted",1.0,1],["M12 14.6L14.3 18.6L9.7 18.6Z","dew",1.0,1]],"mic":[["M7.2 4.7L6.9 4.5L6.7 4.2L6.6 3.8L6.7 3.4L6.9 3.1L7.2 2.9L7.6 2.8L16.4 2.8L16.8 2.9L17.1 3.1L17.3 3.4L17.4 3.8L17.3 4.2L17.1 4.5L16.8 4.7L16.4 4.8L7.6 4.8ZM7.2 21.1L6.9 20.9L6.7 20.6L6.6 20.2L6.7 19.8L6.9 19.5L7.2 19.3L7.6 19.2L16.4 19.2L16.8 19.3L17.1 19.5L17.3 19.8L17.4 20.2L17.3 20.6L17.1 20.9L16.8 21.1L16.4 21.2L7.6 21.2Z","muted",1.0,1],["M8.4 5.4L15.6 5.4L12 12Z","muted",0.55,1],["M12 12L15.6 18.6L8.4 18.6Z","dew",1.0,1]]},"unseen":{"std":[["M2.7 9.8L2.7 9.4L2.9 9.1L3.1 8.8L3.4 8.7L3.8 8.7L4.1 8.9L4.3 9.1L4.9 10L5.5 10.7L6.2 11.3L7 11.8L7.8 12.3L8.7 12.6L9.6 12.9L10.6 13.1L11.5 13.2L12.5 13.2L13.4 13.1L14.4 12.9L15.3 12.6L16.2 12.3L17 11.8L17.8 11.3L18.5 10.7L19.1 10L19.6 9.1L19.9 8.9L20.2 8.7L20.6 8.7L20.9 8.8L21.1 9.1L21.3 9.4L21.3 9.8L21.1 10.1L20.5 11L19.8 11.9L18.9 12.7L18 13.4L17 13.9L15.9 14.3L14.8 14.7L13.7 14.9L12.6 15L11.4 15L10.3 14.9L9.2 14.7L8.1 14.3L7 13.9L6 13.4L5.1 12.7L4.2 11.9L3.5 11L2.9 10.1ZM6.1 12L6.2 11.8L6.4 11.7L6.6 11.7L6.8 11.7L7 11.8L7.1 12L7.2 12.2L7 15L6.8 15.3L6.6 15.4L6.3 15.3L6.2 15L6 12.2ZM9.4 13.6L9.5 13.4L9.7 13.3L9.9 13.3L10.1 13.3L10.3 13.4L10.4 13.6L10.4 13.8L10.2 16.6L10.2 16.9L9.9 17L9.7 16.9L9.6 16.6L9.3 13.8ZM13.6 13.6L13.7 13.4L13.9 13.3L14.1 13.3L14.3 13.3L14.5 13.4L14.6 13.6L14.7 13.8L14.4 16.6L14.3 16.9L14.1 17L13.8 16.9L13.8 16.6L13.6 13.8ZM16.9 12L17 11.8L17.2 11.7L17.4 11.7L17.6 11.7L17.8 11.8L17.9 12L17.9 12.2L17.8 15L17.6 15.3L17.4 15.4L17.1 15.3L17.1 15L16.9 12.2Z","muted",1.0,1]],"mic":[["M2.3 9.8L2.3 9.3L2.5 8.9L2.9 8.5L3.4 8.3L3.9 8.3L4.3 8.5L4.7 8.9L5.2 9.7L5.8 10.4L6.5 11L7.2 11.5L8 11.9L8.8 12.3L9.7 12.5L10.6 12.7L11.5 12.8L12.5 12.8L13.4 12.7L14.3 12.5L15.2 12.3L16 11.9L16.8 11.5L17.5 11L18.2 10.4L18.8 9.7L19.3 8.9L19.7 8.5L20.1 8.3L20.6 8.3L21.1 8.5L21.5 8.9L21.7 9.3L21.7 9.8L21.5 10.3L20.9 11.3L20.1 12.2L19.2 13L18.2 13.7L17.1 14.3L16.1 14.7L14.9 15.1L13.8 15.3L12.6 15.4L11.4 15.4L10.2 15.3L9.1 15.1L8 14.7L6.8 14.3L5.8 13.7L4.8 13L4 12.2L3.1 11.3L2.5 10.3ZM7.3 12.8L7.4 12.5L7.7 12.3L8 12.3L8.3 12.3L8.6 12.5L8.7 12.8L8.8 13.1L8.5 16.5L8.3 16.6L8.2 16.7L8 16.8L7.8 16.7L7.7 16.6L7.5 16.5L7.5 16.3L7.2 13.1ZM11.3 13.8L11.4 13.5L11.7 13.4L12 13.3L12.3 13.4L12.6 13.5L12.7 13.8L12.8 14.1L12.5 17.5L12.3 17.6L12.2 17.8L12 17.8L11.8 17.8L11.7 17.6L11.5 17.5L11.5 17.3L11.2 14.1ZM15.3 12.8L15.4 12.5L15.7 12.3L16 12.3L16.3 12.3L16.6 12.5L16.7 12.8L16.8 13.1L16.5 16.5L16.4 16.6L16.2 16.7L16 16.8L15.8 16.7L15.7 16.6L15.5 16.5L15.5 16.3L15.2 13.1Z","muted",1.0,1]]},"life_applied":{"std":[["M2.3 12.7L2.1 12.5L1.9 12.3L1.9 12L1.9 11.7L2.1 11.5L2.3 11.3L2.6 11.3L21.4 11.3L21.7 11.3L21.9 11.5L22.1 11.7L22.1 12L22.1 12.3L21.9 12.5L21.7 12.7L21.4 12.7L2.6 12.7Z","moon",0.8,1],["M8.8 12L8.7 12.9L8.3 13.7L7.7 14.3L6.9 14.7L6 14.8L5.1 14.7L4.3 14.3L3.7 13.7L3.3 12.9L3.2 12L3.3 11.1L3.7 10.3L4.3 9.7L5.1 9.3L6 9.2L6.9 9.3L7.7 9.7L8.3 10.3L8.7 11.1Z","moon",1.0,1],["M14.8 12L14.7 12.9L14.3 13.7L13.7 14.3L12.9 14.7L12 14.8L11.1 14.7L10.3 14.3L9.7 13.7L9.3 12.9L9.2 12L9.3 11.1L9.7 10.3L10.3 9.7L11.1 9.3L12 9.2L12.9 9.3L13.7 9.7L14.3 10.3L14.7 11.1Z","moon",1.0,1],["M20.8 12L20.7 12.9L20.3 13.7L19.6 14.3L18.9 14.7L18 14.8L17.1 14.7L16.4 14.3L15.7 13.7L15.3 12.9L15.2 12L15.3 11.1L15.7 10.3L16.4 9.7L17.1 9.3L18 9.2L18.9 9.3L19.6 9.7L20.3 10.3L20.7 11.1Z","moon",1.0,1]],"mic":[["M2.2 12.9L1.9 12.7L1.7 12.4L1.6 12L1.7 11.6L1.9 11.3L2.2 11.1L2.6 11L21.4 11L21.8 11.1L22.1 11.3L22.3 11.6L22.4 12L22.3 12.4L22.1 12.7L21.8 12.9L21.4 13L2.6 13Z","moon",0.8,1],["M9 12L8.8 12.9L8.4 13.8L7.8 14.4L6.9 14.8L6 15L5.1 14.8L4.2 14.4L3.6 13.8L3.1 12.9L3 12L3.1 11.1L3.6 10.2L4.2 9.6L5.1 9.2L6 9L6.9 9.2L7.8 9.6L8.4 10.2L8.8 11.1Z","moon",1.0,1],["M15 12L14.8 12.9L14.4 13.8L13.8 14.4L12.9 14.8L12 15L11.1 14.8L10.2 14.4L9.6 13.8L9.2 12.9L9 12L9.2 11.1L9.6 10.2L10.2 9.6L11.1 9.2L12 9L12.9 9.2L13.8 9.6L14.4 10.2L14.8 11.1Z","moon",1.0,1],["M21 12L20.9 12.9L20.4 13.8L19.8 14.4L18.9 14.8L18 15L17.1 14.8L16.2 14.4L15.6 13.8L15.2 12.9L15 12L15.2 11.1L15.6 10.2L16.2 9.6L17.1 9.2L18 9L18.9 9.2L19.8 9.6L20.4 10.2L20.9 11.1Z","moon",1.0,1]]},"life_refreshed":{"std":[["M2.3 12.7L2.1 12.5L1.9 12.3L1.9 12L1.9 11.7L2.1 11.5L2.3 11.3L2.6 11.3L21.4 11.3L21.7 11.3L21.9 11.5L22.1 11.7L22.1 12L22.1 12.3L21.9 12.5L21.7 12.7L21.4 12.7L2.6 12.7Z","moon",0.8,1],["M8.8 12L8.7 12.9L8.3 13.7L7.7 14.3L6.9 14.7L6 14.8L5.1 14.7L4.3 14.3L3.7 13.7L3.3 12.9L3.2 12L3.3 11.1L3.7 10.3L4.3 9.7L5.1 9.3L6 9.2L6.9 9.3L7.7 9.7L8.3 10.3L8.7 11.1Z","moon",1.0,1],["M14.8 12L14.7 12.9L14.3 13.7L13.7 14.3L12.9 14.7L12 14.8L11.1 14.7L10.3 14.3L9.7 13.7L9.3 12.9L9.2 12L9.3 11.1L9.7 10.3L10.3 9.7L11.1 9.3L12 9.2L12.9 9.3L13.7 9.7L14.3 10.3L14.7 11.1Z","moon",1.0,1],["M20.8 12L20.7 12.9L20.3 13.7L19.6 14.3L18.9 14.7L18 14.8L17.1 14.7L16.4 14.3L15.7 13.7L15.3 12.9L15.2 12L15.3 11.1L15.7 10.3L16.4 9.7L17.1 9.3L18 9.2L18.9 9.3L19.6 9.7L20.3 10.3L20.7 11.1Z","moon",1.0,1],["M16.4 12L16.2 13.1L15.8 14.2L15.1 15.1L14.2 15.8L13.1 16.2L12 16.4L10.9 16.2L9.8 15.8L8.9 15.1L8.2 14.2L7.8 13.1L7.6 12L7.8 10.9L8.2 9.8L8.9 8.9L9.8 8.2L10.9 7.8L12 7.6L13.1 7.8L14.2 8.2L15.1 8.9L15.8 9.8L16.2 10.9ZM15.3 11.1L14.9 10.3L14.4 9.6L13.7 9.1L12.9 8.7L12 8.6L11.1 8.7L10.3 9.1L9.6 9.6L9.1 10.3L8.7 11.1L8.6 12L8.7 12.9L9.1 13.7L9.6 14.4L10.3 14.9L11.1 15.3L12 15.4L12.9 15.3L13.7 14.9L14.4 14.4L14.9 13.7L15.3 12.9L15.4 12Z","moon",1,1]],"mic":[["M2.2 12.9L1.9 12.7L1.7 12.4L1.6 12L1.7 11.6L1.9 11.3L2.2 11.1L2.6 11L21.4 11L21.8 11.1L22.1 11.3L22.3 11.6L22.4 12L22.3 12.4L22.1 12.7L21.8 12.9L21.4 13L2.6 13Z","moon",0.8,1],["M9 12L8.8 12.9L8.4 13.8L7.8 14.4L6.9 14.8L6 15L5.1 14.8L4.2 14.4L3.6 13.8L3.1 12.9L3 12L3.1 11.1L3.6 10.2L4.2 9.6L5.1 9.2L6 9L6.9 9.2L7.8 9.6L8.4 10.2L8.8 11.1Z","moon",1.0,1],["M15 12L14.8 12.9L14.4 13.8L13.8 14.4L12.9 14.8L12 15L11.1 14.8L10.2 14.4L9.6 13.8L9.2 12.9L9 12L9.2 11.1L9.6 10.2L10.2 9.6L11.1 9.2L12 9L12.9 9.2L13.8 9.6L14.4 10.2L14.8 11.1Z","moon",1.0,1],["M21 12L20.9 12.9L20.4 13.8L19.8 14.4L18.9 14.8L18 15L17.1 14.8L16.2 14.4L15.6 13.8L15.2 12.9L15 12L15.2 11.1L15.6 10.2L16.2 9.6L17.1 9.2L18 9L18.9 9.2L19.8 9.6L20.4 10.2L20.9 11.1Z","moon",1.0,1],["M16.4 12L16.2 13.1L15.8 14.2L15.1 15.1L14.2 15.8L13.1 16.2L12 16.4L10.9 16.2L9.8 15.8L8.9 15.1L8.2 14.2L7.8 13.1L7.6 12L7.8 10.9L8.2 9.8L8.9 8.9L9.8 8.2L10.9 7.8L12 7.6L13.1 7.8L14.2 8.2L15.1 8.9L15.8 9.8L16.2 10.9ZM15.3 11.1L14.9 10.3L14.4 9.6L13.7 9.1L12.9 8.7L12 8.6L11.1 8.7L10.3 9.1L9.6 9.6L9.1 10.3L8.7 11.1L8.6 12L8.7 12.9L9.1 13.7L9.6 14.4L10.3 14.9L11.1 15.3L12 15.4L12.9 15.3L13.7 14.9L14.4 14.4L14.9 13.7L15.3 12.9L15.4 12Z","moon",1,1]]},"life_decremented":{"std":[["M2.3 12.7L2.1 12.5L1.9 12.3L1.9 12L1.9 11.7L2.1 11.5L2.3 11.3L2.6 11.3L21.4 11.3L21.7 11.3L21.9 11.5L22.1 11.7L22.1 12L22.1 12.3L21.9 12.5L21.7 12.7L21.4 12.7L2.6 12.7Z","moon",0.8,1],["M8.8 12L8.7 12.9L8.3 13.7L7.7 14.3L6.9 14.7L6 14.8L5.1 14.7L4.3 14.3L3.7 13.7L3.3 12.9L3.2 12L3.3 11.1L3.7 10.3L4.3 9.7L5.1 9.3L6 9.2L6.9 9.3L7.7 9.7L8.3 10.3L8.7 11.1Z","moon",1.0,1],["M14.8 12L14.7 12.9L14.3 13.7L13.7 14.3L12.9 14.7L12 14.8L11.1 14.7L10.3 14.3L9.7 13.7L9.3 12.9L9.2 12L9.3 11.1L9.7 10.3L10.3 9.7L11.1 9.3L12 9.2L12.9 9.3L13.7 9.7L14.3 10.3L14.7 11.1Z","moon",1.0,1],["M20.8 12L20.7 12.9L20.3 13.7L19.6 14.3L18.9 14.7L18 14.8L17.1 14.7L16.4 14.3L15.7 13.7L15.3 12.9L15.2 12L15.3 11.1L15.7 10.3L16.4 9.7L17.1 9.3L18 9.2L18.9 9.3L19.6 9.7L20.3 10.3L20.7 11.1ZM19.7 11.4L19.5 10.9L19.1 10.5L18.6 10.3L18 10.2L17.4 10.3L16.9 10.5L16.5 10.9L16.3 11.4L16.2 12L16.3 12.6L16.5 13.1L16.9 13.5L17.4 13.7L18 13.8L18.6 13.7L19.1 13.5L19.5 13.1L19.7 12.6L19.8 12Z","moon",1,1]],"mic":[["M2.2 12.9L1.9 12.7L1.7 12.4L1.6 12L1.7 11.6L1.9 11.3L2.2 11.1L2.6 11L21.4 11L21.8 11.1L22.1 11.3L22.3 11.6L22.4 12L22.3 12.4L22.1 12.7L21.8 12.9L21.4 13L2.6 13Z","moon",0.8,1],["M9 12L8.8 12.9L8.4 13.8L7.8 14.4L6.9 14.8L6 15L5.1 14.8L4.2 14.4L3.6 13.8L3.1 12.9L3 12L3.1 11.1L3.6 10.2L4.2 9.6L5.1 9.2L6 9L6.9 9.2L7.8 9.6L8.4 10.2L8.8 11.1Z","moon",1.0,1],["M15 12L14.8 12.9L14.4 13.8L13.8 14.4L12.9 14.8L12 15L11.1 14.8L10.2 14.4L9.6 13.8L9.2 12.9L9 12L9.2 11.1L9.6 10.2L10.2 9.6L11.1 9.2L12 9L12.9 9.2L13.8 9.6L14.4 10.2L14.8 11.1Z","moon",1.0,1],["M21 12L20.9 12.9L20.4 13.8L19.8 14.4L18.9 14.8L18 15L17.1 14.8L16.2 14.4L15.6 13.8L15.2 12.9L15 12L15.2 11.1L15.6 10.2L16.2 9.6L17.1 9.2L18 9L18.9 9.2L19.8 9.6L20.4 10.2L20.9 11.1ZM19.6 11.5L19.4 11L19 10.6L18.5 10.4L18 10.3L17.5 10.4L17 10.6L16.6 11L16.4 11.5L16.3 12L16.4 12.5L16.6 13L17 13.4L17.5 13.6L18 13.7L18.5 13.6L19 13.4L19.4 13L19.6 12.5L19.7 12Z","moon",1,1]]},"life_expired":{"std":[["M2.3 12.7L2.1 12.5L1.9 12.3L1.9 12L1.9 11.7L2.1 11.5L2.3 11.3L2.6 11.3L4 11.3L4.3 11.3L4.5 11.5L4.7 11.7L4.7 12L4.7 12.3L4.5 12.5L4.3 12.7L4 12.7L2.6 12.7ZM7.7 12.7L7.5 12.5L7.3 12.3L7.3 12L7.3 11.7L7.5 11.5L7.7 11.3L8 11.3L10 11.3L10.3 11.3L10.5 11.5L10.7 11.7L10.7 12L10.7 12.3L10.5 12.5L10.3 12.7L10 12.7L8 12.7ZM13.7 12.7L13.5 12.5L13.3 12.3L13.3 12L13.3 11.7L13.5 11.5L13.7 11.3L14 11.3L16 11.3L16.3 11.3L16.5 11.5L16.6 11.7L16.7 12L16.6 12.3L16.5 12.5L16.3 12.7L16 12.7L14 12.7ZM19.7 12.7L19.5 12.5L19.4 12.3L19.3 12L19.4 11.7L19.5 11.5L19.7 11.3L20 11.3L21.4 11.3L21.7 11.3L21.9 11.5L22.1 11.7L22.1 12L22.1 12.3L21.9 12.5L21.7 12.7L21.4 12.7L20 12.7Z","moon",0.8,1],["M8.8 12L8.7 12.9L8.3 13.7L7.7 14.3L6.9 14.7L6 14.8L5.1 14.7L4.3 14.3L3.7 13.7L3.3 12.9L3.2 12L3.3 11.1L3.7 10.3L4.3 9.7L5.1 9.3L6 9.2L6.9 9.3L7.7 9.7L8.3 10.3L8.7 11.1ZM7.7 11.4L7.5 10.9L7.1 10.5L6.6 10.3L6 10.2L5.4 10.3L4.9 10.5L4.5 10.9L4.3 11.4L4.2 12L4.3 12.6L4.5 13.1L4.9 13.5L5.4 13.7L6 13.8L6.6 13.7L7.1 13.5L7.5 13.1L7.7 12.6L7.8 12Z","moon",1,1],["M14.8 12L14.7 12.9L14.3 13.7L13.7 14.3L12.9 14.7L12 14.8L11.1 14.7L10.3 14.3L9.7 13.7L9.3 12.9L9.2 12L9.3 11.1L9.7 10.3L10.3 9.7L11.1 9.3L12 9.2L12.9 9.3L13.7 9.7L14.3 10.3L14.7 11.1ZM13.7 11.4L13.5 10.9L13.1 10.5L12.6 10.3L12 10.2L11.4 10.3L10.9 10.5L10.5 10.9L10.3 11.4L10.2 12L10.3 12.6L10.5 13.1L10.9 13.5L11.4 13.7L12 13.8L12.6 13.7L13.1 13.5L13.5 13.1L13.7 12.6L13.8 12Z","moon",1,1],["M20.8 12L20.7 12.9L20.3 13.7L19.6 14.3L18.9 14.7L18 14.8L17.1 14.7L16.4 14.3L15.7 13.7L15.3 12.9L15.2 12L15.3 11.1L15.7 10.3L16.4 9.7L17.1 9.3L18 9.2L18.9 9.3L19.6 9.7L20.3 10.3L20.7 11.1ZM19.7 11.4L19.5 10.9L19.1 10.5L18.6 10.3L18 10.2L17.4 10.3L16.9 10.5L16.5 10.9L16.3 11.4L16.2 12L16.3 12.6L16.5 13.1L16.9 13.5L17.4 13.7L18 13.8L18.6 13.7L19.1 13.5L19.5 13.1L19.7 12.6L19.8 12Z","moon",1,1]],"mic":[["M2.2 12.9L1.9 12.7L1.7 12.4L1.6 12L1.7 11.6L1.9 11.3L2.2 11.1L2.6 11L4 11L4.4 11.1L4.7 11.3L4.9 11.6L5 12L4.9 12.4L4.7 12.7L4.4 12.9L4 13L2.6 13ZM7.6 12.9L7.3 12.7L7.1 12.4L7 12L7.1 11.6L7.3 11.3L7.6 11.1L8 11L10 11L10.4 11.1L10.7 11.3L10.9 11.6L11 12L10.9 12.4L10.7 12.7L10.4 12.9L10 13L8 13ZM13.6 12.9L13.3 12.7L13.1 12.4L13 12L13.1 11.6L13.3 11.3L13.6 11.1L14 11L16 11L16.4 11.1L16.7 11.3L16.9 11.6L17 12L16.9 12.4L16.7 12.7L16.4 12.9L16 13L14 13ZM19.6 12.9L19.3 12.7L19.1 12.4L19 12L19.1 11.6L19.3 11.3L19.6 11.1L20 11L21.4 11L21.8 11.1L22.1 11.3L22.3 11.6L22.4 12L22.3 12.4L22.1 12.7L21.8 12.9L21.4 13L20 13Z","moon",0.8,1],["M9 12L8.8 12.9L8.4 13.8L7.8 14.4L6.9 14.8L6 15L5.1 14.8L4.2 14.4L3.6 13.8L3.1 12.9L3 12L3.1 11.1L3.6 10.2L4.2 9.6L5.1 9.2L6 9L6.9 9.2L7.8 9.6L8.4 10.2L8.8 11.1ZM7.6 11.5L7.4 11L7 10.6L6.5 10.4L6 10.3L5.5 10.4L5 10.6L4.6 11L4.4 11.5L4.3 12L4.4 12.5L4.6 13L5 13.4L5.5 13.6L6 13.7L6.5 13.6L7 13.4L7.4 13L7.6 12.5L7.7 12Z","moon",1,1],["M15 12L14.8 12.9L14.4 13.8L13.8 14.4L12.9 14.8L12 15L11.1 14.8L10.2 14.4L9.6 13.8L9.2 12.9L9 12L9.2 11.1L9.6 10.2L10.2 9.6L11.1 9.2L12 9L12.9 9.2L13.8 9.6L14.4 10.2L14.8 11.1ZM13.6 11.5L13.4 11L13 10.6L12.5 10.4L12 10.3L11.5 10.4L11 10.6L10.6 11L10.4 11.5L10.3 12L10.4 12.5L10.6 13L11 13.4L11.5 13.6L12 13.7L12.5 13.6L13 13.4L13.4 13L13.6 12.5L13.7 12Z","moon",1,1],["M21 12L20.9 12.9L20.4 13.8L19.8 14.4L18.9 14.8L18 15L17.1 14.8L16.2 14.4L15.6 13.8L15.2 12.9L15 12L15.2 11.1L15.6 10.2L16.2 9.6L17.1 9.2L18 9L18.9 9.2L19.8 9.6L20.4 10.2L20.9 11.1ZM19.6 11.5L19.4 11L19 10.6L18.5 10.4L18 10.3L17.5 10.4L17 10.6L16.6 11L16.4 11.5L16.3 12L16.4 12.5L16.6 13L17 13.4L17.5 13.6L18 13.7L18.5 13.6L19 13.4L19.4 13L19.6 12.5L19.7 12Z","moon",1,1]]},"life_ended":{"std":[["M2.3 12.7L2.1 12.5L1.9 12.3L1.9 12L1.9 11.7L2.1 11.5L2.3 11.3L2.6 11.3L21.4 11.3L21.7 11.3L21.9 11.5L22.1 11.7L22.1 12L22.1 12.3L21.9 12.5L21.7 12.7L21.4 12.7L2.6 12.7Z","muted",0.8,1],["M1.9 9.1L2 8.8L2.3 8.7L2.6 8.6L2.9 8.7L3.2 8.8L3.3 9.1L3.4 9.4L3.4 14.6L3.3 14.9L3.2 15.2L2.9 15.3L2.6 15.4L2.3 15.3L2 15.2L1.9 14.9L1.8 14.6L1.8 9.4ZM20.7 9.1L20.8 8.8L21.1 8.7L21.4 8.6L21.7 8.7L22 8.8L22.1 9.1L22.2 9.4L22.2 14.6L22.1 14.9L22 15.2L21.7 15.3L21.4 15.4L21.1 15.3L20.8 15.2L20.7 14.9L20.6 14.6L20.6 9.4Z","muted",1.0,1]],"mic":[["M2.2 12.9L1.9 12.7L1.7 12.4L1.6 12L1.7 11.6L1.9 11.3L2.2 11.1L2.6 11L21.4 11L21.8 11.1L22.1 11.3L22.3 11.6L22.4 12L22.3 12.4L22.1 12.7L21.8 12.9L21.4 13L2.6 13Z","muted",0.8,1],["M1.9 9.1L2 8.8L2.3 8.7L2.6 8.6L2.9 8.7L3.2 8.8L3.3 9.1L3.4 9.4L3.4 14.6L3.3 14.9L3.2 15.2L2.9 15.3L2.6 15.4L2.3 15.3L2 15.2L1.9 14.9L1.8 14.6L1.8 9.4ZM20.7 9.1L20.8 8.8L21.1 8.7L21.4 8.6L21.7 8.7L22 8.8L22.1 9.1L22.2 9.4L22.2 14.6L22.1 14.9L22 15.2L21.7 15.3L21.4 15.4L21.1 15.3L20.8 15.2L20.7 14.9L20.6 14.6L20.6 9.4Z","muted",1.0,1]]},"life_thawed":{"std":[["M2.3 12.7L2.1 12.5L1.9 12.3L1.9 12L1.9 11.7L2.1 11.5L2.3 11.3L2.6 11.3L21.4 11.3L21.7 11.3L21.9 11.5L22.1 11.7L22.1 12L22.1 12.3L21.9 12.5L21.7 12.7L21.4 12.7L2.6 12.7Z","frostglass",0.8,1],["M8.8 12L8.7 12.9L8.3 13.7L7.7 14.3L6.9 14.7L6 14.8L5.1 14.7L4.3 14.3L3.7 13.7L3.3 12.9L3.2 12L3.3 11.1L3.7 10.3L4.3 9.7L5.1 9.3L6 9.2L6.9 9.3L7.7 9.7L8.3 10.3L8.7 11.1ZM7.7 11.4L7.5 10.9L7.1 10.5L6.6 10.3L6 10.2L5.4 10.3L4.9 10.5L4.5 10.9L4.3 11.4L4.2 12L4.3 12.6L4.5 13.1L4.9 13.5L5.4 13.7L6 13.8L6.6 13.7L7.1 13.5L7.5 13.1L7.7 12.6L7.8 12Z","frostglass",1,1],["M14.8 12L14.7 12.9L14.3 13.7L13.7 14.3L12.9 14.7L12 14.8L11.1 14.7L10.3 14.3L9.7 13.7L9.3 12.9L9.2 12L9.3 11.1L9.7 10.3L10.3 9.7L11.1 9.3L12 9.2L12.9 9.3L13.7 9.7L14.3 10.3L14.7 11.1ZM13.7 11.4L13.5 10.9L13.1 10.5L12.6 10.3L12 10.2L11.4 10.3L10.9 10.5L10.5 10.9L10.3 11.4L10.2 12L10.3 12.6L10.5 13.1L10.9 13.5L11.4 13.7L12 13.8L12.6 13.7L13.1 13.5L13.5 13.1L13.7 12.6L13.8 12Z","frostglass",1,1],["M20.8 12L20.7 12.9L20.3 13.7L19.6 14.3L18.9 14.7L18 14.8L17.1 14.7L16.4 14.3L15.7 13.7L15.3 12.9L15.2 12L15.3 11.1L15.7 10.3L16.4 9.7L17.1 9.3L18 9.2L18.9 9.3L19.6 9.7L20.3 10.3L20.7 11.1ZM19.7 11.4L19.5 10.9L19.1 10.5L18.6 10.3L18 10.2L17.4 10.3L16.9 10.5L16.5 10.9L16.3 11.4L16.2 12L16.3 12.6L16.5 13.1L16.9 13.5L17.4 13.7L18 13.8L18.6 13.7L19.1 13.5L19.5 13.1L19.7 12.6L19.8 12Z","frostglass",1,1],["M12 15.7L12.9 17.1L13.2 17.4L13.3 17.8L13.5 18.1L13.5 18.5L13.4 19.2L13.2 19.7L12.8 20L12.4 20.1L12 20.1L11.6 20.1L11.2 20L10.8 19.7L10.6 19.2L10.5 18.5L10.5 18.1L10.8 17.4L11.1 17.1Z","frostglass",1.0,1]],"mic":[["M2.2 12.9L1.9 12.7L1.7 12.4L1.6 12L1.7 11.6L1.9 11.3L2.2 11.1L2.6 11L21.4 11L21.8 11.1L22.1 11.3L22.3 11.6L22.4 12L22.3 12.4L22.1 12.7L21.8 12.9L21.4 13L2.6 13Z","frostglass",0.8,1],["M9 12L8.8 12.9L8.4 13.8L7.8 14.4L6.9 14.8L6 15L5.1 14.8L4.2 14.4L3.6 13.8L3.1 12.9L3 12L3.1 11.1L3.6 10.2L4.2 9.6L5.1 9.2L6 9L6.9 9.2L7.8 9.6L8.4 10.2L8.8 11.1ZM7.6 11.5L7.4 11L7 10.6L6.5 10.4L6 10.3L5.5 10.4L5 10.6L4.6 11L4.4 11.5L4.3 12L4.4 12.5L4.6 13L5 13.4L5.5 13.6L6 13.7L6.5 13.6L7 13.4L7.4 13L7.6 12.5L7.7 12Z","frostglass",1,1],["M15 12L14.8 12.9L14.4 13.8L13.8 14.4L12.9 14.8L12 15L11.1 14.8L10.2 14.4L9.6 13.8L9.2 12.9L9 12L9.2 11.1L9.6 10.2L10.2 9.6L11.1 9.2L12 9L12.9 9.2L13.8 9.6L14.4 10.2L14.8 11.1ZM13.6 11.5L13.4 11L13 10.6L12.5 10.4L12 10.3L11.5 10.4L11 10.6L10.6 11L10.4 11.5L10.3 12L10.4 12.5L10.6 13L11 13.4L11.5 13.6L12 13.7L12.5 13.6L13 13.4L13.4 13L13.6 12.5L13.7 12Z","frostglass",1,1],["M21 12L20.9 12.9L20.4 13.8L19.8 14.4L18.9 14.8L18 15L17.1 14.8L16.2 14.4L15.6 13.8L15.2 12.9L15 12L15.2 11.1L15.6 10.2L16.2 9.6L17.1 9.2L18 9L18.9 9.2L19.8 9.6L20.4 10.2L20.9 11.1ZM19.6 11.5L19.4 11L19 10.6L18.5 10.4L18 10.3L17.5 10.4L17 10.6L16.6 11L16.4 11.5L16.3 12L16.4 12.5L16.6 13L17 13.4L17.5 13.6L18 13.7L18.5 13.6L19 13.4L19.4 13L19.6 12.5L19.7 12Z","frostglass",1,1],["M12 15.8L12.8 16.8L13.5 17.7L13.7 18.1L13.8 18.5L13.9 18.9L13.8 19.7L13.5 20.3L13.1 20.6L12.6 20.8L12 20.8L11.4 20.8L10.9 20.6L10.5 20.3L10.2 19.7L10.1 18.9L10.2 18.5L10.3 18.1L10.5 17.7L11.2 16.8Z","frostglass",1.0,1]]},"life_shattered":{"std":[["M2.3 12.7L2.1 12.5L1.9 12.3L1.9 12L1.9 11.7L2.1 11.5L2.3 11.3L2.6 11.3L8.2 11.3L10 10L10.8 12.6L10.3 11.2L10.5 10.9L10.7 10.8L10.9 10.7L11.2 10.7L11.5 10.9L11.6 11.1L11.7 11.3L11.7 11.6L10.1 13.8L9.2 11.2L8.6 12.7L2.6 12.7ZM13.5 13.1L13.3 13.2L13.1 13.3L12.8 13.3L12.5 13.1L12.4 12.9L12.3 12.7L12.3 12.4L13.9 10.2L14.8 12.8L15.4 11.3L21.4 11.3L21.7 11.3L21.9 11.5L22.1 11.7L22.1 12L22.1 12.3L21.9 12.5L21.7 12.7L21.4 12.7L15.8 12.7L14 14L13.2 11.4L13.7 12.8Z","rime",0.9,1],["M7.4 12L7.3 12.9L6.9 13.7L6.2 14.3L5.5 14.7L4.6 14.8L3.7 14.7L3 14.3L2.3 13.7L1.9 12.9L1.8 12L1.9 11.1L2.3 10.3L3 9.7L3.7 9.3L4.6 9.2L5.5 9.3L6.2 9.7L6.9 10.3L7.3 11.1ZM6.3 11.4L6.1 10.9L5.7 10.5L5.2 10.3L4.6 10.2L4 10.3L3.5 10.5L3.1 10.9L2.9 11.4L2.8 12L2.9 12.6L3.1 13.1L3.5 13.5L4 13.7L4.6 13.8L5.2 13.7L5.7 13.5L6.1 13.1L6.3 12.6L6.4 12Z","rime",1,1],["M22.2 12L22.1 12.9L21.7 13.7L21.1 14.3L20.3 14.7L19.4 14.8L18.5 14.7L17.8 14.3L17.1 13.7L16.7 12.9L16.6 12L16.7 11.1L17.1 10.3L17.8 9.7L18.5 9.3L19.4 9.2L20.3 9.3L21.1 9.7L21.7 10.3L22.1 11.1ZM21.1 11.4L20.9 10.9L20.5 10.5L20 10.3L19.4 10.2L18.8 10.3L18.3 10.5L17.9 10.9L17.7 11.4L17.6 12L17.7 12.6L17.9 13.1L18.3 13.5L18.8 13.7L19.4 13.8L20 13.7L20.5 13.5L20.9 13.1L21.1 12.6L21.2 12Z","rime",1,1],["M10.6 11.4L10.6 11.3L10.7 11.1L10.8 11L11 10.9L11.1 11L11.3 11.1L11.4 11.2L12.2 13L12.7 12.2L12.9 12.2L13.1 12.2L13.2 12.2L13.3 12.3L13.4 12.5L13.4 12.7L13.4 12.8L13.3 12.9L11.8 13.8L10.6 11.6Z","ice_crack",1,0],["M10.6 5.4L12.6 4.2L11.8 7.4ZM13.4 17L15.2 18.8L12.8 19.4Z","rime",1.0,1]],"mic":[["M2.2 12.9L1.9 12.7L1.7 12.4L1.6 12L1.7 11.6L1.9 11.3L2.2 11.1L2.6 11L8.1 11L10.1 9.7L10.9 12.3L10.1 11.1L10.2 10.7L10.6 10.5L10.9 10.4L11.3 10.4L11.7 10.6L11.9 10.9L12 11.3L11.9 11.7L9.9 14.1L9.1 11.5L8.7 13L2.6 13ZM13.8 13.3L13.4 13.5L13.1 13.6L12.7 13.6L12.3 13.4L12.1 13.1L12 12.7L12.1 12.3L14.1 9.9L14.9 12.5L15.3 11L21.4 11L21.8 11.1L22.1 11.3L22.3 11.6L22.4 12L22.3 12.4L22.1 12.7L21.8 12.9L21.4 13L15.9 13L13.9 14.3L13.1 11.7L13.9 12.9Z","rime",0.9,1],["M7.6 12L7.5 12.9L7 13.8L6.4 14.4L5.5 14.8L4.6 15L3.7 14.8L2.8 14.4L2.2 13.8L1.8 12.9L1.6 12L1.8 11.1L2.2 10.2L2.8 9.6L3.7 9.2L4.6 9L5.5 9.2L6.4 9.6L7 10.2L7.5 11.1ZM6.2 11.5L6 11L5.6 10.6L5.1 10.4L4.6 10.3L4.1 10.4L3.6 10.6L3.2 11L3 11.5L2.9 12L3 12.5L3.2 13L3.6 13.4L4.1 13.6L4.6 13.7L5.1 13.6L5.6 13.4L6 13L6.2 12.5L6.3 12Z","rime",1,1],["M22.4 12L22.2 12.9L21.8 13.8L21.2 14.4L20.3 14.8L19.4 15L18.5 14.8L17.6 14.4L17 13.8L16.6 12.9L16.4 12L16.6 11.1L17 10.2L17.6 9.6L18.5 9.2L19.4 9L20.3 9.2L21.2 9.6L21.8 10.2L22.2 11.1ZM21 11.5L20.8 11L20.4 10.6L19.9 10.4L19.4 10.3L18.9 10.4L18.4 10.6L18 11L17.8 11.5L17.7 12L17.8 12.5L18 13L18.4 13.4L18.9 13.6L19.4 13.7L19.9 13.6L20.4 13.4L20.8 13L21 12.5L21.1 12Z","rime",1,1],["M10.6 11.4L10.6 11.3L10.7 11.1L10.8 11L11 10.9L11.1 11L11.3 11.1L11.4 11.2L12.2 13L12.7 12.2L12.9 12.2L13.1 12.2L13.2 12.2L13.3 12.3L13.4 12.5L13.4 12.7L13.4 12.8L13.3 12.9L11.8 13.8L10.6 11.6Z","ice_crack",1,0]]},"life_death_clear":{"std":[["M2.3 12.7L2.1 12.5L1.9 12.3L1.9 12L1.9 11.7L2.1 11.5L2.3 11.3L2.6 11.3L21.4 11.3L21.7 11.3L21.9 11.5L22.1 11.7L22.1 12L22.1 12.3L21.9 12.5L21.7 12.7L21.4 12.7L2.6 12.7Z","ash",0.8,1],["M17 12L16.9 13L16.6 13.9L16.2 14.8L15.5 15.5L14.8 16.2L13.9 16.6L13 16.9L12 17L11 16.9L10.1 16.6L9.2 16.2L8.5 15.5L7.8 14.8L7.4 13.9L7.1 13L7 12L7.1 11L7.4 10.1L7.8 9.2L8.5 8.5L9.2 7.8L10.1 7.4L11 7.1L12 7L13 7.1L13.9 7.4L14.8 7.8L15.5 8.5L16.2 9.2L16.6 10.1L16.9 11Z","grave",1.0,1],["M17 12L16.9 13L16.6 13.9L16.2 14.8L15.5 15.5L14.8 16.2L13.9 16.6L13 16.9L12 17L11 16.9L10.1 16.6L9.2 16.2L8.5 15.5L7.8 14.8L7.4 13.9L7.1 13L7 12L7.1 11L7.4 10.1L7.8 9.2L8.5 8.5L9.2 7.8L10.1 7.4L11 7.1L12 7L13 7.1L13.9 7.4L14.8 7.8L15.5 8.5L16.2 9.2L16.6 10.1L16.9 11ZM15.9 11.2L15.7 10.5L15.3 9.8L14.8 9.2L14.2 8.7L13.5 8.3L12.8 8.1L12 8L11.2 8.1L10.5 8.3L9.8 8.7L9.2 9.2L8.7 9.8L8.3 10.5L8.1 11.2L8 12L8.1 12.8L8.3 13.5L8.7 14.2L9.2 14.8L9.8 15.3L10.5 15.7L11.2 15.9L12 16L12.8 15.9L13.5 15.7L14.2 15.3L14.8 14.8L15.3 14.2L15.7 13.5L15.9 12.8L16 12Z","ash",1,1]],"mic":[["M2.2 12.9L1.9 12.7L1.7 12.4L1.6 12L1.7 11.6L1.9 11.3L2.2 11.1L2.6 11L21.4 11L21.8 11.1L22.1 11.3L22.3 11.6L22.4 12L22.3 12.4L22.1 12.7L21.8 12.9L21.4 13L2.6 13Z","ash",0.8,1],["M17 12L16.9 13L16.6 13.9L16.2 14.8L15.5 15.5L14.8 16.2L13.9 16.6L13 16.9L12 17L11 16.9L10.1 16.6L9.2 16.2L8.5 15.5L7.8 14.8L7.4 13.9L7.1 13L7 12L7.1 11L7.4 10.1L7.8 9.2L8.5 8.5L9.2 7.8L10.1 7.4L11 7.1L12 7L13 7.1L13.9 7.4L14.8 7.8L15.5 8.5L16.2 9.2L16.6 10.1L16.9 11Z","grave",1.0,1],["M17 12L16.9 13L16.6 13.9L16.2 14.8L15.5 15.5L14.8 16.2L13.9 16.6L13 16.9L12 17L11 16.9L10.1 16.6L9.2 16.2L8.5 15.5L7.8 14.8L7.4 13.9L7.1 13L7 12L7.1 11L7.4 10.1L7.8 9.2L8.5 8.5L9.2 7.8L10.1 7.4L11 7.1L12 7L13 7.1L13.9 7.4L14.8 7.8L15.5 8.5L16.2 9.2L16.6 10.1L16.9 11ZM15.9 11.2L15.7 10.5L15.3 9.8L14.8 9.2L14.2 8.7L13.5 8.3L12.8 8.1L12 8L11.2 8.1L10.5 8.3L9.8 8.7L9.2 9.2L8.7 9.8L8.3 10.5L8.1 11.2L8 12L8.1 12.8L8.3 13.5L8.7 14.2L9.2 14.8L9.8 15.3L10.5 15.7L11.2 15.9L12 16L12.8 15.9L13.5 15.7L14.2 15.3L14.8 14.8L15.3 14.2L15.7 13.5L15.9 12.8L16 12Z","ash",1,1]]},"life_broken_reapplied":{"std":[["M2.3 12.7L2.1 12.5L1.9 12.3L1.9 12L1.9 11.7L2.1 11.5L2.3 11.3L2.6 11.3L5.2 11.3L7 10L7.7 12.6L7.6 11.1L7.7 10.9L8 10.8L8.2 10.7L8.5 10.8L8.7 10.9L8.8 11.2L8.9 11.4L8.8 11.7L7.1 13.8L6.2 11.2L5.6 12.7L2.6 12.7Z","rime",0.9,1],["M10.3 12.7L10.1 12.5L9.9 12.3L9.9 12L9.9 11.7L10.1 11.5L10.3 11.3L10.6 11.3L21.4 11.3L21.7 11.3L21.9 11.5L22.1 11.7L22.1 12L22.1 12.3L21.9 12.5L21.7 12.7L21.4 12.7L10.6 12.7Z","rime",0.8,1],["M16.4 12L16.3 12.9L15.9 13.7L15.2 14.3L14.5 14.7L13.6 14.8L12.7 14.7L11.9 14.3L11.3 13.7L10.9 12.9L10.8 12L10.9 11.1L11.3 10.3L11.9 9.7L12.7 9.3L13.6 9.2L14.5 9.3L15.2 9.7L15.9 10.3L16.3 11.1Z","rime",1.0,1],["M21.8 12L21.7 12.9L21.3 13.7L20.6 14.3L19.9 14.7L19 14.8L18.1 14.7L17.4 14.3L16.7 13.7L16.3 12.9L16.2 12L16.3 11.1L16.7 10.3L17.4 9.7L18.1 9.3L19 9.2L19.9 9.3L20.6 9.7L21.3 10.3L21.7 11.1Z","rime",1.0,1]],"mic":[["M2.2 12.9L1.9 12.7L1.7 12.4L1.6 12L1.7 11.6L1.9 11.3L2.2 11.1L2.6 11L5.1 11.1L7.1 9.7L7.8 12.3L7.3 11L7.5 10.7L7.8 10.5L8.2 10.4L8.6 10.5L8.9 10.7L9.1 11L9.2 11.4L9.1 11.8L7 14.1L6.1 11.5L5.7 12.9L2.6 13Z","rime",0.9,1],["M10.2 12.9L9.9 12.7L9.7 12.4L9.6 12L9.7 11.6L9.9 11.3L10.2 11.1L10.6 11L21.4 11L21.8 11.1L22.1 11.3L22.3 11.6L22.4 12L22.3 12.4L22.1 12.7L21.8 12.9L21.4 13L10.6 13Z","rime",0.8,1],["M16.6 12L16.4 12.9L16 13.8L15.4 14.4L14.5 14.8L13.6 15L12.7 14.8L11.8 14.4L11.2 13.8L10.8 12.9L10.6 12L10.8 11.1L11.2 10.2L11.8 9.6L12.7 9.2L13.6 9L14.5 9.2L15.4 9.6L16 10.2L16.4 11.1Z","rime",1.0,1],["M22 12L21.9 12.9L21.4 13.8L20.8 14.4L19.9 14.8L19 15L18.1 14.8L17.2 14.4L16.6 13.8L16.1 12.9L16 12L16.1 11.1L16.6 10.2L17.2 9.6L18.1 9.2L19 9L19.9 9.2L20.8 9.6L21.4 10.2L21.9 11.1Z","rime",1.0,1]]},"tier_basic":{"std":[["M15.2 12L15 13L14.6 13.9L13.9 14.6L13 15L12 15.2L11 15L10.1 14.6L9.4 13.9L9 13L8.8 12L9 11L9.4 10.1L10.1 9.4L11 9L12 8.8L13 9L13.9 9.4L14.6 10.1L15 11Z","text",1.0,1]],"mic":[["M16 12L15.8 13.2L15.2 14.3L14.3 15.2L13.2 15.8L12 16L10.8 15.8L9.7 15.2L8.8 14.3L8.2 13.2L8 12L8.2 10.8L8.8 9.7L9.7 8.8L10.8 8.2L12 8L13.2 8.2L14.3 8.8L15.2 9.7L15.8 10.8Z","text",1.0,1]]},"tier_ultimate":{"std":[["M12 6L16 12L12 18L8 12Z","starcore",1.0,1],["M11.5 8.4L11.7 8.2L11.8 8.1L12 8.1L12.2 8.1L12.3 8.2L12.5 8.4L12.5 8.6L12.5 15.4L12.5 15.6L12.3 15.8L12.2 15.9L12 15.9L11.8 15.9L11.7 15.8L11.5 15.6L11.5 15.4L11.5 8.6Z","ink",1,0]],"mic":[["M12 6L16 12L12 18L8 12Z","starcore",1.0,1]]},"seal_basic":{"std":[["M15.2 12L15 13L14.6 13.9L13.9 14.6L13 15L12 15.2L11 15L10.1 14.6L9.4 13.9L9 13L8.8 12L9 11L9.4 10.1L10.1 9.4L11 9L12 8.8L13 9L13.9 9.4L14.6 10.1L15 11Z","cls",1.0,1]],"mic":[["M16 12L15.8 13.2L15.2 14.3L14.3 15.2L13.2 15.8L12 16L10.8 15.8L9.7 15.2L8.8 14.3L8.2 13.2L8 12L8.2 10.8L8.8 9.7L9.7 8.8L10.8 8.2L12 8L13.2 8.2L14.3 8.8L15.2 9.7L15.8 10.8Z","cls",1.0,1]]},"seal_ultimate":{"std":[["M12 4.2L17.2 12L12 19.8L6.8 12Z","cls",1.0,1],["M12 6L16 12L12 18L8 12Z","starcore",1.0,1],["M11.5 8.4L11.7 8.2L11.8 8.1L12 8.1L12.2 8.1L12.3 8.2L12.5 8.4L12.5 8.6L12.5 15.4L12.5 15.6L12.3 15.8L12.2 15.9L12 15.9L11.8 15.9L11.7 15.8L11.5 15.6L11.5 15.4L11.5 8.6Z","ink",1,0]],"mic":[["M12 4.2L17.2 12L12 19.8L6.8 12Z","cls",1.0,1],["M12 6L16 12L12 18L8 12Z","starcore",1.0,1]]},"crest_A":{"std":[["M22 12L21.9 13.3L21.7 14.6L21.2 15.8L20.7 17L19.9 18.1L19.1 19.1L18.1 19.9L17 20.7L15.8 21.2L14.6 21.7L13.3 21.9L12 22L10.7 21.9L9.4 21.7L8.2 21.2L7 20.7L5.9 19.9L4.9 19.1L4.1 18.1L3.3 17L2.8 15.8L2.3 14.6L2.1 13.3L2 12L2.1 10.7L2.3 9.4L2.8 8.2L3.3 7L4.1 5.9L4.9 4.9L5.9 4.1L7 3.3L8.2 2.8L9.4 2.3L10.7 2.1L12 2L13.3 2.1L14.6 2.3L15.8 2.8L17 3.3L18.1 4.1L19.1 4.9L19.9 5.9L20.7 7L21.2 8.2L21.7 9.4L21.9 10.7ZM19.5 11L19.3 10L19 9.1L18.6 8.2L18 7.4L17.4 6.6L16.6 6L15.8 5.4L14.9 5L14 4.7L13 4.5L12 4.4L11 4.5L10 4.7L9.1 5L8.2 5.4L7.4 6L6.6 6.6L6 7.4L5.4 8.2L5 9.1L4.7 10L4.5 11L4.4 12L4.5 13L4.7 14L5 14.9L5.4 15.8L6 16.6L6.6 17.4L7.4 18L8.2 18.6L9.1 19L10 19.3L11 19.5L12 19.6L13 19.5L14 19.3L14.9 19L15.8 18.6L16.6 18L17.4 17.4L18 16.6L18.6 15.8L19 14.9L19.3 14L19.5 13L19.6 12Z","azure",1,1],["M19.6 12L19.5 13L19.3 14L19 14.9L18.6 15.8L18 16.6L17.4 17.4L16.6 18L15.8 18.6L14.9 19L14 19.3L13 19.5L12 19.6L11 19.5L10 19.3L9.1 19L8.2 18.6L7.4 18L6.6 17.4L6 16.6L5.4 15.8L5 14.9L4.7 14L4.5 13L4.4 12L4.5 11L4.7 10L5 9.1L5.4 8.2L6 7.4L6.6 6.6L7.4 6L8.2 5.4L9.1 5L10 4.7L11 4.5L12 4.4L13 4.5L14 4.7L14.9 5L15.8 5.4L16.6 6L17.4 6.6L18 7.4L18.6 8.2L19 9.1L19.3 10L19.5 11Z","azure_field",1,0]],"mic":[["M22 12L21.9 13.3L21.7 14.6L21.2 15.8L20.7 17L19.9 18.1L19.1 19.1L18.1 19.9L17 20.7L15.8 21.2L14.6 21.7L13.3 21.9L12 22L10.7 21.9L9.4 21.7L8.2 21.2L7 20.7L5.9 19.9L4.9 19.1L4.1 18.1L3.3 17L2.8 15.8L2.3 14.6L2.1 13.3L2 12L2.1 10.7L2.3 9.4L2.8 8.2L3.3 7L4.1 5.9L4.9 4.9L5.9 4.1L7 3.3L8.2 2.8L9.4 2.3L10.7 2.1L12 2L13.3 2.1L14.6 2.3L15.8 2.8L17 3.3L18.1 4.1L19.1 4.9L19.9 5.9L20.7 7L21.2 8.2L21.7 9.4L21.9 10.7ZM18.9 11.1L18.8 10.2L18.5 9.3L18.1 8.5L17.6 7.7L16.9 7L16.3 6.5L15.5 5.9L14.7 5.5L13.8 5.2L12.9 5.1L12 5L11.1 5.1L10.2 5.2L9.3 5.5L8.5 5.9L7.7 6.5L7 7L6.5 7.7L5.9 8.5L5.5 9.3L5.2 10.2L5.1 11.1L5 12L5.1 12.9L5.2 13.8L5.5 14.7L5.9 15.5L6.5 16.3L7 16.9L7.7 17.6L8.5 18.1L9.3 18.5L10.2 18.8L11.1 18.9L12 19L12.9 18.9L13.8 18.8L14.7 18.5L15.5 18.1L16.3 17.6L16.9 16.9L17.6 16.3L18.1 15.5L18.5 14.7L18.8 13.8L18.9 12.9L19 12Z","azure",1,1],["M19 12L18.9 12.9L18.8 13.8L18.5 14.7L18.1 15.5L17.6 16.3L16.9 16.9L16.3 17.6L15.5 18.1L14.7 18.5L13.8 18.8L12.9 18.9L12 19L11.1 18.9L10.2 18.8L9.3 18.5L8.5 18.1L7.7 17.6L7 16.9L6.5 16.3L5.9 15.5L5.5 14.7L5.2 13.8L5.1 12.9L5 12L5.1 11.1L5.2 10.2L5.5 9.3L5.9 8.5L6.5 7.7L7 7L7.7 6.5L8.5 5.9L9.3 5.5L10.2 5.2L11.1 5.1L12 5L12.9 5.1L13.8 5.2L14.7 5.5L15.5 5.9L16.3 6.5L16.9 7L17.6 7.7L18.1 8.5L18.5 9.3L18.8 10.2L18.9 11.1Z","azure_field",1,0]]},"crest_B":{"std":[["M22 12L21.9 13.3L21.7 14.6L21.2 15.8L20.7 17L19.9 18.1L19.1 19.1L18.1 19.9L17 20.7L15.8 21.2L14.6 21.7L13.3 21.9L12 22L10.7 21.9L9.4 21.7L8.2 21.2L7 20.7L5.9 19.9L4.9 19.1L4.1 18.1L3.3 17L2.8 15.8L2.3 14.6L2.1 13.3L2 12L2.1 10.7L2.3 9.4L2.8 8.2L3.3 7L4.1 5.9L4.9 4.9L5.9 4.1L7 3.3L8.2 2.8L9.4 2.3L10.7 2.1L12 2L13.3 2.1L14.6 2.3L15.8 2.8L17 3.3L18.1 4.1L19.1 4.9L19.9 5.9L20.7 7L21.2 8.2L21.7 9.4L21.9 10.7ZM20.5 10.9L20.3 9.8L19.9 8.7L19.4 7.7L18.8 6.8L18.1 5.9L17.2 5.2L16.3 4.5L15.3 4L14.2 3.7L13.1 3.5L12 3.4L10.9 3.5L9.8 3.7L8.7 4L7.7 4.5L6.8 5.2L5.9 5.9L5.2 6.8L4.5 7.7L4 8.7L3.7 9.8L3.5 10.9L3.4 12L3.5 13.1L3.7 14.2L4 15.3L4.5 16.3L5.2 17.2L5.9 18.1L6.8 18.8L7.7 19.4L8.7 19.9L9.8 20.3L10.9 20.5L12 20.6L13.1 20.5L14.2 20.3L15.3 19.9L16.3 19.4L17.2 18.8L18.1 18.1L18.8 17.2L19.4 16.3L19.9 15.3L20.3 14.2L20.5 13.1L20.6 12Z","crimson",1,1],["M20.6 12L20.5 13.1L20.3 14.2L19.9 15.3L19.4 16.3L18.8 17.2L18.1 18.1L17.2 18.8L16.3 19.4L15.3 19.9L14.2 20.3L13.1 20.5L12 20.6L10.9 20.5L9.8 20.3L8.7 19.9L7.7 19.4L6.8 18.8L5.9 18.1L5.2 17.2L4.5 16.3L4 15.3L3.7 14.2L3.5 13.1L3.4 12L3.5 10.9L3.7 9.8L4 8.7L4.5 7.7L5.2 6.8L5.9 5.9L6.8 5.2L7.7 4.5L8.7 4L9.8 3.7L10.9 3.5L12 3.4L13.1 3.5L14.2 3.7L15.3 4L16.3 4.5L17.2 5.2L18.1 5.9L18.8 6.8L19.4 7.7L19.9 8.7L20.3 9.8L20.5 10.9ZM19.3 11L19.1 10.1L18.8 9.2L18.4 8.3L17.9 7.5L17.2 6.8L16.5 6.1L15.7 5.6L14.8 5.2L13.9 4.8L13 4.7L12 4.6L11 4.7L10.1 4.8L9.2 5.2L8.3 5.6L7.5 6.1L6.8 6.8L6.1 7.5L5.6 8.3L5.2 9.2L4.8 10.1L4.7 11L4.6 12L4.7 13L4.8 13.9L5.2 14.8L5.6 15.7L6.1 16.5L6.8 17.2L7.5 17.9L8.3 18.4L9.2 18.8L10.1 19.1L11 19.3L12 19.4L13 19.3L13.9 19.1L14.8 18.8L15.7 18.4L16.5 17.9L17.2 17.2L17.9 16.5L18.4 15.7L18.8 14.8L19.1 13.9L19.3 13L19.4 12Z","ink",1,0],["M19.4 12L19.3 13L19.1 13.9L18.8 14.8L18.4 15.7L17.9 16.5L17.2 17.2L16.5 17.9L15.7 18.4L14.8 18.8L13.9 19.1L13 19.3L12 19.4L11 19.3L10.1 19.1L9.2 18.8L8.3 18.4L7.5 17.9L6.8 17.2L6.1 16.5L5.6 15.7L5.2 14.8L4.8 13.9L4.7 13L4.6 12L4.7 11L4.8 10.1L5.2 9.2L5.6 8.3L6.1 7.5L6.8 6.8L7.5 6.1L8.3 5.6L9.2 5.2L10.1 4.8L11 4.7L12 4.6L13 4.7L13.9 4.8L14.8 5.2L15.7 5.6L16.5 6.1L17.2 6.8L17.9 7.5L18.4 8.3L18.8 9.2L19.1 10.1L19.3 11ZM17.9 11.2L17.8 10.4L17.5 9.7L17.2 9L16.8 8.3L16.2 7.8L15.7 7.2L15 6.8L14.3 6.5L13.6 6.2L12.8 6L12 6L11.2 6L10.4 6.2L9.7 6.5L9 6.8L8.3 7.2L7.8 7.8L7.2 8.3L6.8 9L6.5 9.7L6.2 10.4L6 11.2L6 12L6 12.8L6.2 13.6L6.5 14.3L6.8 15L7.2 15.7L7.8 16.2L8.3 16.8L9 17.2L9.7 17.5L10.4 17.8L11.2 17.9L12 18L12.8 17.9L13.6 17.8L14.3 17.5L15 17.2L15.7 16.8L16.2 16.2L16.8 15.7L17.2 15L17.5 14.3L17.8 13.6L17.9 12.8L18 12Z","crimson",1,1],["M18 12L17.9 12.8L17.8 13.6L17.5 14.3L17.2 15L16.8 15.7L16.2 16.2L15.7 16.8L15 17.2L14.3 17.5L13.6 17.8L12.8 17.9L12 18L11.2 17.9L10.4 17.8L9.7 17.5L9 17.2L8.3 16.8L7.8 16.2L7.2 15.7L6.8 15L6.5 14.3L6.2 13.6L6 12.8L6 12L6 11.2L6.2 10.4L6.5 9.7L6.8 9L7.2 8.3L7.8 7.8L8.3 7.2L9 6.8L9.7 6.5L10.4 6.2L11.2 6L12 6L12.8 6L13.6 6.2L14.3 6.5L15 6.8L15.7 7.2L16.2 7.8L16.8 8.3L17.2 9L17.5 9.7L17.8 10.4L17.9 11.2Z","crimson_field",1,0]],"mic":[["M22 12L21.9 13.3L21.7 14.6L21.2 15.8L20.7 17L19.9 18.1L19.1 19.1L18.1 19.9L17 20.7L15.8 21.2L14.6 21.7L13.3 21.9L12 22L10.7 21.9L9.4 21.7L8.2 21.2L7 20.7L5.9 19.9L4.9 19.1L4.1 18.1L3.3 17L2.8 15.8L2.3 14.6L2.1 13.3L2 12L2.1 10.7L2.3 9.4L2.8 8.2L3.3 7L4.1 5.9L4.9 4.9L5.9 4.1L7 3.3L8.2 2.8L9.4 2.3L10.7 2.1L12 2L13.3 2.1L14.6 2.3L15.8 2.8L17 3.3L18.1 4.1L19.1 4.9L19.9 5.9L20.7 7L21.2 8.2L21.7 9.4L21.9 10.7ZM19.9 11L19.7 9.9L19.4 8.9L18.9 8L18.4 7.1L17.7 6.3L16.9 5.7L16 5.1L15.1 4.6L14.1 4.3L13 4.1L12 4L11 4.1L9.9 4.3L8.9 4.6L8 5.1L7.1 5.7L6.3 6.3L5.7 7.1L5.1 8L4.6 8.9L4.3 9.9L4.1 11L4 12L4.1 13L4.3 14.1L4.6 15.1L5.1 16L5.7 16.9L6.3 17.7L7.1 18.4L8 18.9L8.9 19.4L9.9 19.7L11 19.9L12 20L13 19.9L14.1 19.7L15.1 19.4L16 18.9L16.9 18.4L17.7 17.7L18.4 16.9L18.9 16L19.4 15.1L19.7 14.1L19.9 13L20 12Z","crimson",1,1],["M20 12L19.9 13L19.7 14.1L19.4 15.1L18.9 16L18.4 16.9L17.7 17.7L16.9 18.4L16 18.9L15.1 19.4L14.1 19.7L13 19.9L12 20L11 19.9L9.9 19.7L8.9 19.4L8 18.9L7.1 18.4L6.3 17.7L5.7 16.9L5.1 16L4.6 15.1L4.3 14.1L4.1 13L4 12L4.1 11L4.3 9.9L4.6 8.9L5.1 8L5.7 7.1L6.3 6.3L7.1 5.7L8 5.1L8.9 4.6L9.9 4.3L11 4.1L12 4L13 4.1L14.1 4.3L15.1 4.6L16 5.1L16.9 5.7L17.7 6.3L18.4 7.1L18.9 8L19.4 8.9L19.7 9.9L19.9 11ZM18.5 11.1L18.4 10.3L18.1 9.5L17.7 8.7L17.2 8L16.7 7.3L16 6.8L15.3 6.3L14.5 5.9L13.7 5.6L12.9 5.5L12 5.4L11.1 5.5L10.3 5.6L9.5 5.9L8.7 6.3L8 6.8L7.3 7.3L6.8 8L6.3 8.7L5.9 9.5L5.6 10.3L5.5 11.1L5.4 12L5.5 12.9L5.6 13.7L5.9 14.5L6.3 15.3L6.8 16L7.3 16.7L8 17.2L8.7 17.7L9.5 18.1L10.3 18.4L11.1 18.5L12 18.6L12.9 18.5L13.7 18.4L14.5 18.1L15.3 17.7L16 17.2L16.7 16.7L17.2 16L17.7 15.3L18.1 14.5L18.4 13.7L18.5 12.9L18.6 12Z","ink",1,0],["M18.6 12L18.5 12.9L18.4 13.7L18.1 14.5L17.7 15.3L17.2 16L16.7 16.7L16 17.2L15.3 17.7L14.5 18.1L13.7 18.4L12.9 18.5L12 18.6L11.1 18.5L10.3 18.4L9.5 18.1L8.7 17.7L8 17.2L7.3 16.7L6.8 16L6.3 15.3L5.9 14.5L5.6 13.7L5.5 12.9L5.4 12L5.5 11.1L5.6 10.3L5.9 9.5L6.3 8.7L6.8 8L7.3 7.3L8 6.8L8.7 6.3L9.5 5.9L10.3 5.6L11.1 5.5L12 5.4L12.9 5.5L13.7 5.6L14.5 5.9L15.3 6.3L16 6.8L16.7 7.3L17.2 8L17.7 8.7L18.1 9.5L18.4 10.3L18.5 11.1ZM16.8 11.4L16.6 10.8L16.4 10.2L16.2 9.6L15.8 9.1L15.4 8.6L14.9 8.2L14.4 7.8L13.8 7.6L13.2 7.4L12.6 7.2L12 7.2L11.4 7.2L10.8 7.4L10.2 7.6L9.6 7.8L9.1 8.2L8.6 8.6L8.2 9.1L7.8 9.6L7.6 10.2L7.4 10.8L7.2 11.4L7.2 12L7.2 12.6L7.4 13.2L7.6 13.8L7.8 14.4L8.2 14.9L8.6 15.4L9.1 15.8L9.6 16.2L10.2 16.4L10.8 16.6L11.4 16.8L12 16.8L12.6 16.8L13.2 16.6L13.8 16.4L14.4 16.2L14.9 15.8L15.4 15.4L15.8 14.9L16.2 14.4L16.4 13.8L16.6 13.2L16.8 12.6L16.8 12Z","crimson",1,1],["M16.8 12L16.8 12.6L16.6 13.2L16.4 13.8L16.2 14.4L15.8 14.9L15.4 15.4L14.9 15.8L14.4 16.2L13.8 16.4L13.2 16.6L12.6 16.8L12 16.8L11.4 16.8L10.8 16.6L10.2 16.4L9.6 16.2L9.1 15.8L8.6 15.4L8.2 14.9L7.8 14.4L7.6 13.8L7.4 13.2L7.2 12.6L7.2 12L7.2 11.4L7.4 10.8L7.6 10.2L7.8 9.6L8.2 9.1L8.6 8.6L9.1 8.2L9.6 7.8L10.2 7.6L10.8 7.4L11.4 7.2L12 7.2L12.6 7.2L13.2 7.4L13.8 7.6L14.4 7.8L14.9 8.2L15.4 8.6L15.8 9.1L16.2 9.6L16.4 10.2L16.6 10.8L16.8 11.4Z","crimson_field",1,0]]},"waystone_A":{"std":[["M21.4 12L21.4 12.8L21.3 13.6L21.1 14.4L20.8 15.2L20.5 16L20.1 16.7L19.7 17.4L19.2 18L18.6 18.6L18 19.2L17.4 19.7L16.7 20.1L16 20.5L15.2 20.8L14.4 21.1L13.6 21.3L12.8 21.4L12 21.4L11.2 21.4L10.4 21.3L9.6 21.1L8.8 20.8L8 20.5L7.3 20.1L6.6 19.7L6 19.2L5.3 18.6L4.8 18L4.3 17.4L3.9 16.7L3.5 16L3.2 15.2L2.9 14.4L2.7 13.6L2.6 12.8L2.6 12L2.6 11.2L2.7 10.4L2.9 9.6L3.2 8.8L3.5 8L3.9 7.3L4.3 6.6L4.8 6L5.3 5.3L6 4.8L6.6 4.3L7.3 3.9L8 3.5L8.8 3.2L9.6 2.9L10.4 2.7L11.2 2.6L12 2.6L12.8 2.6L13.6 2.7L14.4 2.9L15.2 3.2L16 3.5L16.7 3.9L17.4 4.3L18 4.8L18.6 5.3L19.2 6L19.7 6.6L20.1 7.3L20.5 8L20.8 8.8L21.1 9.6L21.3 10.4L21.4 11.2Z","azure",1.0,1],["M17.6 12L17.6 12.7L17.4 13.4L17.2 14.1L16.9 14.8L16.4 15.4L16 16L15.4 16.4L14.8 16.9L14.1 17.2L13.4 17.4L12.7 17.6L12 17.6L11.3 17.6L10.6 17.4L9.9 17.2L9.2 16.9L8.6 16.4L8 16L7.6 15.4L7.2 14.8L6.8 14.1L6.6 13.4L6.5 12.7L6.4 12L6.5 11.3L6.6 10.6L6.8 9.9L7.2 9.2L7.6 8.6L8 8L8.6 7.6L9.2 7.2L9.9 6.8L10.6 6.6L11.3 6.5L12 6.4L12.7 6.5L13.4 6.6L14.1 6.8L14.8 7.2L15.4 7.6L16 8L16.4 8.6L16.9 9.2L17.2 9.9L17.4 10.6L17.6 11.3ZM16.5 11.4L16.4 10.8L16.2 10.3L15.9 9.8L15.6 9.3L15.2 8.8L14.7 8.4L14.2 8.1L13.7 7.8L13.2 7.7L12.6 7.5L12 7.5L11.4 7.5L10.8 7.7L10.3 7.8L9.8 8.1L9.3 8.4L8.8 8.8L8.4 9.3L8.1 9.8L7.8 10.3L7.7 10.8L7.5 11.4L7.5 12L7.5 12.6L7.7 13.2L7.8 13.7L8.1 14.2L8.4 14.7L8.8 15.2L9.3 15.6L9.8 15.9L10.3 16.2L10.8 16.4L11.4 16.5L12 16.5L12.6 16.5L13.2 16.4L13.7 16.2L14.2 15.9L14.7 15.6L15.2 15.2L15.6 14.7L15.9 14.2L16.2 13.7L16.4 13.2L16.5 12.6L16.5 12Z","ink",1,0],["M12.5 5.7L12.3 5.8L12.2 6L12 6L11.8 6L11.7 5.8L11.5 5.7L11.5 5.5L11.5 3.7L11.5 3.5L11.7 3.4L11.8 3.2L12 3.2L12.2 3.2L12.3 3.4L12.5 3.5L12.5 3.7L12.5 5.5ZM17.7 9.2L17.5 9.2L17.3 9.2L17.2 9L17.1 8.8L17.1 8.6L17.2 8.4L17.4 8.3L18.9 7.4L19.1 7.3L19.3 7.4L19.5 7.5L19.6 7.6L19.7 7.8L19.7 8L19.6 8.2L19.4 8.3L17.9 9.2ZM17.2 15.6L17.1 15.4L17.1 15.2L17.2 15L17.3 14.8L17.5 14.8L17.7 14.8L17.9 14.8L19.4 15.7L19.6 15.8L19.7 16L19.7 16.2L19.6 16.4L19.5 16.6L19.3 16.6L19.1 16.6L17.4 15.7ZM11.5 18.3L11.7 18.1L11.8 18L12 18L12.2 18L12.3 18.1L12.5 18.3L12.5 18.5L12.5 20.3L12.5 20.5L12.3 20.6L12.2 20.8L12 20.8L11.8 20.8L11.7 20.6L11.5 20.5L11.5 20.3L11.5 18.5ZM6.3 14.8L6.5 14.8L6.7 14.8L6.8 15L6.9 15.2L6.8 15.4L6.8 15.6L6.6 15.7L5.1 16.6L4.9 16.6L4.7 16.6L4.5 16.6L4.4 16.4L4.3 16.2L4.3 16L4.4 15.8L4.6 15.7L6.1 14.8ZM6.8 8.4L6.8 8.6L6.9 8.8L6.8 9L6.7 9.2L6.5 9.2L6.3 9.2L6.1 9.2L4.6 8.3L4.4 8.2L4.3 8L4.3 7.8L4.4 7.6L4.5 7.5L4.7 7.4L4.9 7.3L6.6 8.3Z","ink",1,0]],"mic":[["M21.6 12L21.6 12.8L21.4 13.7L21.3 14.5L21 15.3L20.7 16.1L20.3 16.8L19.9 17.5L19.4 18.2L18.8 18.8L18.2 19.4L17.5 19.9L16.8 20.3L16.1 20.7L15.3 21L14.5 21.3L13.7 21.4L12.8 21.6L12 21.6L11.2 21.6L10.3 21.4L9.5 21.3L8.7 21L7.9 20.7L7.2 20.3L6.5 19.9L5.8 19.4L5.2 18.8L4.7 18.2L4.1 17.5L3.7 16.8L3.3 16.1L3 15.3L2.7 14.5L2.5 13.7L2.4 12.8L2.4 12L2.4 11.2L2.5 10.3L2.7 9.5L3 8.7L3.3 7.9L3.7 7.2L4.1 6.5L4.7 5.8L5.2 5.2L5.8 4.7L6.5 4.1L7.2 3.7L7.9 3.3L8.7 3L9.5 2.7L10.3 2.5L11.2 2.4L12 2.4L12.8 2.4L13.7 2.5L14.5 2.7L15.3 3L16.1 3.3L16.8 3.7L17.5 4.1L18.2 4.7L18.8 5.2L19.4 5.8L19.9 6.5L20.3 7.2L20.7 7.9L21 8.7L21.3 9.5L21.4 10.3L21.6 11.2ZM11.1 4.8L12.9 4.8L12.9 2.2L11.1 2.2ZM17.8 7.6L18.7 9.2L20.9 7.9L20 6.3ZM18.7 14.8L17.8 16.4L20 17.7L20.9 16.1ZM12.9 19.2L11.1 19.2L11.1 21.8L12.9 21.8ZM6.2 16.4L5.3 14.8L3.1 16.1L4 17.7ZM5.3 9.2L6.2 7.6L4 6.3L3.1 7.9Z","azure",1,1]]},"waystone_B":{"std":[["M21.4 12L21.4 12.8L21.3 13.6L21.1 14.4L20.8 15.2L20.5 16L20.1 16.7L19.7 17.4L19.2 18L18.6 18.6L18 19.2L17.4 19.7L16.7 20.1L16 20.5L15.2 20.8L14.4 21.1L13.6 21.3L12.8 21.4L12 21.4L11.2 21.4L10.4 21.3L9.6 21.1L8.8 20.8L8 20.5L7.3 20.1L6.6 19.7L6 19.2L5.3 18.6L4.8 18L4.3 17.4L3.9 16.7L3.5 16L3.2 15.2L2.9 14.4L2.7 13.6L2.6 12.8L2.6 12L2.6 11.2L2.7 10.4L2.9 9.6L3.2 8.8L3.5 8L3.9 7.3L4.3 6.6L4.8 6L5.3 5.3L6 4.8L6.6 4.3L7.3 3.9L8 3.5L8.8 3.2L9.6 2.9L10.4 2.7L11.2 2.6L12 2.6L12.8 2.6L13.6 2.7L14.4 2.9L15.2 3.2L16 3.5L16.7 3.9L17.4 4.3L18 4.8L18.6 5.3L19.2 6L19.7 6.6L20.1 7.3L20.5 8L20.8 8.8L21.1 9.6L21.3 10.4L21.4 11.2Z","crimson",1.0,1],["M17.6 12L17.6 12.7L17.4 13.4L17.2 14.1L16.9 14.8L16.4 15.4L16 16L15.4 16.4L14.8 16.9L14.1 17.2L13.4 17.4L12.7 17.6L12 17.6L11.3 17.6L10.6 17.4L9.9 17.2L9.2 16.9L8.6 16.4L8 16L7.6 15.4L7.2 14.8L6.8 14.1L6.6 13.4L6.5 12.7L6.4 12L6.5 11.3L6.6 10.6L6.8 9.9L7.2 9.2L7.6 8.6L8 8L8.6 7.6L9.2 7.2L9.9 6.8L10.6 6.6L11.3 6.5L12 6.4L12.7 6.5L13.4 6.6L14.1 6.8L14.8 7.2L15.4 7.6L16 8L16.4 8.6L16.9 9.2L17.2 9.9L17.4 10.6L17.6 11.3ZM16.5 11.4L16.4 10.8L16.2 10.3L15.9 9.8L15.6 9.3L15.2 8.8L14.7 8.4L14.2 8.1L13.7 7.8L13.2 7.7L12.6 7.5L12 7.5L11.4 7.5L10.8 7.7L10.3 7.8L9.8 8.1L9.3 8.4L8.8 8.8L8.4 9.3L8.1 9.8L7.8 10.3L7.7 10.8L7.5 11.4L7.5 12L7.5 12.6L7.7 13.2L7.8 13.7L8.1 14.2L8.4 14.7L8.8 15.2L9.3 15.6L9.8 15.9L10.3 16.2L10.8 16.4L11.4 16.5L12 16.5L12.6 16.5L13.2 16.4L13.7 16.2L14.2 15.9L14.7 15.6L15.2 15.2L15.6 14.7L15.9 14.2L16.2 13.7L16.4 13.2L16.5 12.6L16.5 12Z","ink",1,0],["M12.5 5.7L12.3 5.8L12.2 6L12 6L11.8 6L11.7 5.8L11.5 5.7L11.5 5.5L11.5 3.7L11.5 3.5L11.7 3.4L11.8 3.2L12 3.2L12.2 3.2L12.3 3.4L12.5 3.5L12.5 3.7L12.5 5.5ZM17.7 9.2L17.5 9.2L17.3 9.2L17.2 9L17.1 8.8L17.1 8.6L17.2 8.4L17.4 8.3L18.9 7.4L19.1 7.3L19.3 7.4L19.5 7.5L19.6 7.6L19.7 7.8L19.7 8L19.6 8.2L19.4 8.3L17.9 9.2ZM17.2 15.6L17.1 15.4L17.1 15.2L17.2 15L17.3 14.8L17.5 14.8L17.7 14.8L17.9 14.8L19.4 15.7L19.6 15.8L19.7 16L19.7 16.2L19.6 16.4L19.5 16.6L19.3 16.6L19.1 16.6L17.4 15.7ZM11.5 18.3L11.7 18.1L11.8 18L12 18L12.2 18L12.3 18.1L12.5 18.3L12.5 18.5L12.5 20.3L12.5 20.5L12.3 20.6L12.2 20.8L12 20.8L11.8 20.8L11.7 20.6L11.5 20.5L11.5 20.3L11.5 18.5ZM6.3 14.8L6.5 14.8L6.7 14.8L6.8 15L6.9 15.2L6.8 15.4L6.8 15.6L6.6 15.7L5.1 16.6L4.9 16.6L4.7 16.6L4.5 16.6L4.4 16.4L4.3 16.2L4.3 16L4.4 15.8L4.6 15.7L6.1 14.8ZM6.8 8.4L6.8 8.6L6.9 8.8L6.8 9L6.7 9.2L6.5 9.2L6.3 9.2L6.1 9.2L4.6 8.3L4.4 8.2L4.3 8L4.3 7.8L4.4 7.6L4.5 7.5L4.7 7.4L4.9 7.3L6.6 8.3Z","ink",1,0]],"mic":[["M21.6 12L21.6 12.8L21.4 13.7L21.3 14.5L21 15.3L20.7 16.1L20.3 16.8L19.9 17.5L19.4 18.2L18.8 18.8L18.2 19.4L17.5 19.9L16.8 20.3L16.1 20.7L15.3 21L14.5 21.3L13.7 21.4L12.8 21.6L12 21.6L11.2 21.6L10.3 21.4L9.5 21.3L8.7 21L7.9 20.7L7.2 20.3L6.5 19.9L5.8 19.4L5.2 18.8L4.7 18.2L4.1 17.5L3.7 16.8L3.3 16.1L3 15.3L2.7 14.5L2.5 13.7L2.4 12.8L2.4 12L2.4 11.2L2.5 10.3L2.7 9.5L3 8.7L3.3 7.9L3.7 7.2L4.1 6.5L4.7 5.8L5.2 5.2L5.8 4.7L6.5 4.1L7.2 3.7L7.9 3.3L8.7 3L9.5 2.7L10.3 2.5L11.2 2.4L12 2.4L12.8 2.4L13.7 2.5L14.5 2.7L15.3 3L16.1 3.3L16.8 3.7L17.5 4.1L18.2 4.7L18.8 5.2L19.4 5.8L19.9 6.5L20.3 7.2L20.7 7.9L21 8.7L21.3 9.5L21.4 10.3L21.6 11.2ZM11.1 4.8L12.9 4.8L12.9 2.2L11.1 2.2ZM17.8 7.6L18.7 9.2L20.9 7.9L20 6.3ZM18.7 14.8L17.8 16.4L20 17.7L20.9 16.1ZM12.9 19.2L11.1 19.2L11.1 21.8L12.9 21.8ZM6.2 16.4L5.3 14.8L3.1 16.1L4 17.7ZM5.3 9.2L6.2 7.6L4 6.3L3.1 7.9Z","crimson",1,1]]},"feed_arrow":{"std":[["M3.7 12.7L3.5 12.5L3.3 12.3L3.2 12L3.3 11.7L3.5 11.5L3.7 11.3L4 11.2L18 11.2L18.3 11.3L18.5 11.5L18.7 11.7L18.8 12L18.7 12.3L18.5 12.5L18.3 12.7L18 12.8L4 12.8ZM19.9 11.7L19.9 11.9L19.9 12.2L19.8 12.5L19.5 12.7L19.2 12.8L19 12.7L18.7 12.6L13.7 8.2L13.5 7.9L13.4 7.7L13.5 7.4L13.6 7.1L13.9 6.9L14.2 6.8L14.4 6.9L14.7 7L19.7 11.4ZM19 11.3L19.2 11.2L19.5 11.3L19.8 11.5L19.9 11.8L19.9 12.1L19.9 12.3L19.7 12.6L14.7 17L14.4 17.1L14.2 17.1L13.9 17.1L13.6 16.9L13.5 16.6L13.4 16.4L13.5 16.1L13.7 15.8L18.7 11.4Z","muted",1.0,1]]},"ab_arcane_blast":{"std":[["M16.5 3.1L16.7 3.1L16.9 3.1L17.1 3.2L17.2 3.4L17.2 3.6L17.2 3.8L16.7 4.3L15.8 5.1L15.3 5.4L14.3 5.9L13.2 6.2L12 6.3L10.8 6.2L10.3 6.1L9.7 5.9L8.7 5.4L8.2 5.1L7.3 4.3L6.9 3.9L6.8 3.8L6.8 3.6L6.8 3.4L6.9 3.2L7.1 3.1L7.3 3.1L7.5 3.1L8.3 4L9.2 4.5L9.6 4.8L10.5 5.1L11 5.2L11.5 5.3L12.5 5.3L13.5 5.1L14.4 4.8L14.8 4.5L15.7 4Z","muted",0.7,0],["M7.5 20.9L7.3 20.9L7.1 20.9L6.9 20.8L6.8 20.6L6.8 20.4L6.8 20.2L7.3 19.7L8.2 18.9L8.7 18.6L9.7 18.1L10.8 17.8L12 17.7L13.2 17.8L13.7 17.9L14.3 18.1L15.3 18.6L15.8 18.9L16.7 19.7L17.1 20.1L17.2 20.2L17.2 20.4L17.2 20.6L17.1 20.8L16.9 20.9L16.7 20.9L16.5 20.9L15.7 20L14.8 19.5L14.4 19.2L13.5 18.9L13 18.8L12.5 18.7L11.5 18.7L10.5 18.9L9.6 19.2L9.2 19.5L8.3 20Z","muted",0.7,0],["M13 17.8L12.8 18.2L12.4 18.4L12 18.5L11.6 18.4L11.2 18.2L11 17.8L10.9 17.4L10.9 8.6L11 8.2L11.2 7.8L11.6 7.6L12 7.5L12.4 7.6L12.8 7.8L13 8.2L13.1 8.6L13.1 17.4Z","mage",1.0,1],["M12.4 17.1L12.2 17.4L12 17.4L11.8 17.4L11.7 17.3L11.6 17.1L11.6 17L11.6 9L11.6 8.8L11.7 8.7L11.8 8.6L12 8.6L12.2 8.6L12.3 8.7L12.4 8.8L12.4 9Z","starcore",1,0],["M12 2.5L12.7 5.7L15.9 6.4L12.7 7.1L12 10.3L11.3 7.1L8.1 6.4L11.3 5.7Z","starcore",1.0,1]]},"ab_shield_bash":{"std":[["M17.2 8.6L17.2 9.3L17 9.9L16.8 10.6L16.5 11.2L16.1 11.8L15.7 12.3L15.2 12.7L14.6 13.1L14 13.4L13.3 13.6L12.7 13.8L12 13.8L11.3 13.8L10.7 13.6L10 13.4L9.4 13.1L8.8 12.7L8.3 12.3L7.9 11.8L7.5 11.2L7.2 10.6L7 9.9L6.8 9.3L6.8 8.6L6.8 7.9L7 7.2L7.2 6.6L7.5 6L7.9 5.4L8.3 4.9L8.8 4.5L9.4 4.1L10 3.8L10.7 3.6L11.3 3.4L12 3.4L12.7 3.4L13.3 3.6L14 3.8L14.6 4.1L15.2 4.5L15.7 4.9L16.1 5.4L16.5 6L16.8 6.6L17 7.2L17.2 7.9ZM16.3 8L16.1 7.5L16 7L15.7 6.5L15.4 6L15 5.6L14.6 5.2L14.2 4.9L13.7 4.6L13.1 4.5L12.6 4.3L12 4.3L11.4 4.3L10.9 4.5L10.3 4.6L9.8 4.9L9.4 5.2L9 5.6L8.6 6L8.3 6.5L8 7L7.8 7.5L7.7 8L7.7 8.6L7.7 9.2L7.8 9.7L8 10.2L8.3 10.8L8.6 11.2L9 11.6L9.4 12L9.8 12.3L10.3 12.6L10.9 12.8L11.4 12.9L12 12.9L12.6 12.9L13.1 12.8L13.7 12.6L14.2 12.3L14.6 12L15 11.6L15.4 11.2L15.7 10.8L16 10.2L16.1 9.7L16.3 9.2L16.3 8.6Z","muted",0.7,0],["M16.1 15.7L15.4 16L14.4 16.4L13.3 16.7L12.2 16.8L11.1 16.8L10 16.5L9.2 16.3L7.9 15.7L9 13.8L9.5 14.1L10.2 14.3L11.1 14.5L11.9 14.6L12.7 14.6L13.5 14.4L14.3 14.2L15 13.8Z","warrior",1.0,1],["M15.5 16.7L15.5 16.4L15.6 16.2L15.8 16.1L16 16L16.3 16.1L16.5 16.2L16.6 16.4L17.4 18.2L17.4 18.4L17.3 18.7L17.2 18.8L17 18.8L16.7 18.7L15.6 16.9ZM11.4 17.4L11.6 17.2L11.8 17.1L12 17L12.2 17.1L12.4 17.2L12.6 17.4L12.6 17.6L12.4 19.6L12.4 19.8L12.2 20L12 20L11.8 20L11.6 19.8L11.4 17.6ZM7.5 16.2L7.7 16.1L8 16L8.2 16.1L8.4 16.2L8.5 16.4L8.5 16.7L8.4 16.9L7.3 18.7L7 18.8L6.8 18.8L6.7 18.7L6.6 18.4L7.4 16.4Z","forge_white",1.0,1]]},"ab_serrated_arrows":{"std":[["M16.5 3.1L16.7 3.1L16.9 3.1L17.1 3.2L17.2 3.4L17.2 3.6L17.2 3.8L16.7 4.3L15.8 5.1L15.3 5.4L14.3 5.9L13.2 6.2L12 6.3L10.8 6.2L10.3 6.1L9.7 5.9L8.7 5.4L8.2 5.1L7.3 4.3L6.9 3.9L6.8 3.8L6.8 3.6L6.8 3.4L6.9 3.2L7.1 3.1L7.3 3.1L7.5 3.1L8.3 4L9.2 4.5L9.6 4.8L10.5 5.1L11 5.2L11.5 5.3L12.5 5.3L13.5 5.1L14.4 4.8L14.8 4.5L15.7 4Z","muted",0.7,0],["M7.5 20.9L7.3 20.9L7.1 20.9L6.9 20.8L6.8 20.6L6.8 20.4L6.8 20.2L7.3 19.7L8.2 18.9L8.7 18.6L9.7 18.1L10.8 17.8L12 17.7L13.2 17.8L13.7 17.9L14.3 18.1L15.3 18.6L15.8 18.9L16.7 19.7L17.1 20.1L17.2 20.2L17.2 20.4L17.2 20.6L17.1 20.8L16.9 20.9L16.7 20.9L16.5 20.9L15.7 20L14.8 19.5L14.4 19.2L13.5 18.9L13 18.8L12.5 18.7L11.5 18.7L10.5 18.9L9.6 19.2L9.2 19.5L8.3 20Z","muted",0.7,0],["M12.6 17.9L12.5 18.1L12.2 18.2L12 18.2L11.8 18.2L11.5 18.1L11.4 17.9L11.3 17.6L11.3 6.6L11.4 6.3L11.5 6.1L11.8 6L12 6L12.2 6L12.5 6.1L12.6 6.3L12.7 6.6L12.7 17.6Z","hunter",1.0,1],["M11.9 15.3L12.2 15.3L12.4 15.4L12.4 15.6L12.4 15.8L12.4 15.9L12.3 16.1L10.1 17.5L9.8 17.6L9.7 17.5L9.6 17.3L9.6 17.1L9.6 17L11.7 15.3ZM11.6 15.9L11.6 15.6L11.6 15.4L11.8 15.3L11.9 15.3L12.1 15.3L12.3 15.3L14.4 17L14.4 17.1L14.4 17.4L14.3 17.5L14.2 17.6L13.9 17.5L11.7 16.1ZM11.9 13.5L12.2 13.5L12.4 13.6L12.4 13.8L12.4 14L12.4 14.1L12.3 14.3L10.1 15.7L9.8 15.8L9.7 15.7L9.6 15.5L9.6 15.3L9.6 15.2L11.7 13.5ZM11.6 14.1L11.6 13.8L11.6 13.6L11.8 13.5L11.9 13.5L12.1 13.5L12.3 13.5L14.4 15.2L14.4 15.3L14.4 15.6L14.3 15.7L14.2 15.8L13.9 15.7L11.7 14.3Z","hunter",1.0,1],["M11.7 5.1L11.9 5L12.2 5L12.4 5.1L12.5 5.3L12.6 5.5L12.6 5.8L12.5 6L10.2 8.7L10.1 8.8L9.9 8.9L9.7 8.9L9.5 8.8L9.4 8.7L9.3 8.5L9.3 8.3L9.4 8.1L11.5 5.2ZM11.4 5.8L11.4 5.5L11.5 5.3L11.6 5.1L11.8 5L12.1 5L12.3 5.1L12.5 5.2L14.6 8.1L14.7 8.3L14.7 8.5L14.6 8.7L14.5 8.8L14.3 8.9L14.1 8.9L13.9 8.8L13.8 8.7L11.5 6Z","hunter",1.0,1]]},"ab_backstab":{"std":[["M17 8.2L17 8.8L16.8 9.5L16.6 10.1L16.3 10.7L16 11.2L15.5 11.7L15 12.2L14.5 12.5L13.9 12.8L13.3 13L12.7 13.2L12 13.2L11.3 13.2L10.7 13L10.1 12.8L9.5 12.5L9 12.2L8.5 11.7L8 11.2L7.7 10.7L7.4 10.1L7.2 9.5L7 8.8L7 8.2L7 7.5L7.2 6.9L7.4 6.3L7.7 5.7L8 5.2L8.5 4.7L9 4.2L9.5 3.9L10.1 3.6L10.7 3.4L11.3 3.2L12 3.2L12.7 3.2L13.3 3.4L13.9 3.6L14.5 3.9L15 4.2L15.5 4.7L16 5.2L16.3 5.7L16.6 6.3L16.8 6.9L17 7.5ZM16.1 7.7L16 7.1L15.8 6.6L15.6 6.2L15.2 5.7L14.9 5.3L14.5 5L14.1 4.7L13.6 4.4L13.1 4.2L12.5 4.1L12 4.1L11.5 4.1L10.9 4.2L10.4 4.4L9.9 4.7L9.5 5L9.1 5.3L8.8 5.7L8.4 6.2L8.2 6.6L8 7.1L7.9 7.7L7.9 8.2L7.9 8.7L8 9.3L8.2 9.8L8.4 10.2L8.8 10.7L9.1 11.1L9.5 11.4L9.9 11.8L10.4 12L10.9 12.2L11.5 12.3L12 12.3L12.5 12.3L13.1 12.2L13.6 12L14.1 11.8L14.5 11.4L14.9 11.1L15.2 10.7L15.6 10.2L15.8 9.8L16 9.3L16.1 8.7L16.1 8.2Z","muted",0.7,0],["M8.5 9.7L8.6 9.5L8.7 9.5L10.3 10.3L11 10.8L11.7 11.3L12.2 11.8L12.8 12.4L13.8 13.7L14.2 14.3L14.6 15.1L14.9 15.8L15.2 16.6L15.5 17.5L15.4 17.7L15.3 17.7L14.4 17.3L13 16.4L11.8 15.4L11.2 14.8L10.7 14.2L9.8 12.8L9.1 11.4ZM15.3 9.5L15.4 9.5L15.5 9.7L14.9 11.4L14.2 12.8L13.3 14.2L12.2 15.4L11 16.4L9.6 17.3L8.7 17.7L8.6 17.7L8.5 17.5L8.8 16.6L9.1 15.8L9.4 15.1L9.8 14.3L10.7 13L11.8 11.8L13 10.8L14.4 9.9Z","rogue",1.0,1],["M9.6 10.9L9.6 10.8L9.7 10.8L10.4 11.3L12.1 13L13.5 14.9L14.4 16.3L14.4 16.4L14.3 16.4L13.6 15.9L11.9 14.2L10.5 12.3ZM14.3 10.8L14.4 10.8L14.4 10.9L14 11.6L12.6 13.6L12.1 14.2L10.9 15.3L9.7 16.4L9.6 16.4L9.6 16.3L10 15.6L11.4 13.7L11.9 13L13.1 11.9Z","blade",1,0]]},"ab_lesser_heal":{"std":[["M16.5 3.1L16.7 3.1L16.9 3.1L17.1 3.2L17.2 3.4L17.2 3.6L17.2 3.8L16.7 4.3L15.8 5.1L15.3 5.4L14.3 5.9L13.2 6.2L12 6.3L10.8 6.2L10.3 6.1L9.7 5.9L8.7 5.4L8.2 5.1L7.3 4.3L6.9 3.9L6.8 3.8L6.8 3.6L6.8 3.4L6.9 3.2L7.1 3.1L7.3 3.1L7.5 3.1L8.3 4L9.2 4.5L9.6 4.8L10.5 5.1L11 5.2L11.5 5.3L12.5 5.3L13.5 5.1L14.4 4.8L14.8 4.5L15.7 4Z","muted",0.7,0],["M7.5 20.9L7.3 20.9L7.1 20.9L6.9 20.8L6.8 20.6L6.8 20.4L6.8 20.2L7.3 19.7L8.2 18.9L8.7 18.6L9.7 18.1L10.8 17.8L12 17.7L13.2 17.8L13.7 17.9L14.3 18.1L15.3 18.6L15.8 18.9L16.7 19.7L17.1 20.1L17.2 20.2L17.2 20.4L17.2 20.6L17.1 20.8L16.9 20.9L16.7 20.9L16.5 20.9L15.7 20L14.8 19.5L14.4 19.2L13.5 18.9L13 18.8L12.5 18.7L11.5 18.7L10.5 18.9L9.6 19.2L9.2 19.5L8.3 20Z","muted",0.7,0],["M13.3 17.5L13.3 18L13.2 18.4L12.9 18.7L12.5 18.9L12 18.9L11.6 18.8L11.3 18.5L10.6 17.4L10.1 16.2L9.8 14.9L9.6 13.6L9.6 12.2L9.8 10.9L9.9 10.3L10.3 9.1L10.9 8L11.2 7.7L11.6 7.5L12.1 7.5L12.5 7.6L12.8 7.9L13 8.3L13 8.8L12.9 9.2L12.7 9.6L12.3 10.4L12 11.4L11.9 12.4L11.9 13.4L12 14.5L12.3 15.5L12.6 16.4L13.1 17.1Z","priest",1.0,1],["M12.6 17.7L12.5 18L12.4 18.1L12.3 18.2L12.1 18.2L12 18.1L11.9 18L11.6 17.6L11.2 17.1L10.8 15.9L10.5 14.8L10.4 14.1L10.3 13.5L10.4 12.3L10.4 11.7L10.7 10.5L11 9.4L11.3 8.8L11.6 8.4L11.8 8.2L12 8.2L12.1 8.3L12.3 8.5L12.3 8.7L11.8 9.7L11.6 10.2L11.3 11.2L11.2 12.3L11.2 13.5L11.3 14.6L11.6 15.7L12 16.7Z","dawnlight",1,0],["M11.2 3.1L11.4 2.8L11.7 2.6L12 2.5L12.3 2.6L12.6 2.8L12.8 3.1L12.8 3.4L12.8 8.2L12.8 8.5L12.6 8.8L12.3 9L12 9.1L11.7 9L11.4 8.8L11.2 8.5L11.2 8.2L11.2 3.4ZM9.3 6.6L9 6.4L8.8 6.1L8.8 5.8L8.8 5.5L9 5.2L9.3 5L9.6 5L14.4 5L14.7 5L15 5.2L15.2 5.5L15.2 5.8L15.2 6.1L15 6.4L14.7 6.6L14.4 6.7L9.6 6.7Z","priest",1.0,1]]},"ab_burst":{"std":[["M21.3 15.2L21 15.9L20.5 16.9L19.8 17.9L19.1 18.8L18.2 19.6L17.3 20.3L16.6 20.6L15.9 21L15.2 21.3L14.8 20.1L15.4 19.9L16.6 19.2L17.2 18.9L18 18.2L18.6 17.4L19.2 16.6L19.9 15.4L20.1 14.8ZM8.8 21.3L8.1 21L7 20.5L6.1 19.8L5.2 19.1L4.4 18.2L3.7 17.3L3.4 16.6L3 15.9L2.7 15.2L3.9 14.8L4.1 15.4L4.8 16.6L5.1 17.2L5.8 18L6.5 18.6L7.4 19.2L8.6 19.9L9.2 20.1ZM2.7 8.8L3 8.1L3.5 7L4.2 6.1L4.9 5.2L5.8 4.4L6.7 3.7L7.4 3.4L8.1 3L8.8 2.7L9.2 3.9L8.6 4.1L7.4 4.8L6.8 5.1L6 5.8L5.3 6.5L4.8 7.4L4.1 8.6L3.9 9.2ZM15.2 2.7L15.9 3L16.9 3.5L17.9 4.2L18.8 4.9L19.6 5.8L20.3 6.7L20.6 7.4L21 8.1L21.3 8.8L20.1 9.2L19.9 8.6L19.2 7.4L18.9 6.8L18.2 6L17.4 5.3L16.6 4.8L15.4 4.1L14.8 3.9Z","mage",1.0,1],["M10.4 3.2L12 5.1L13.6 3.2L13.6 4.4L12 6.5L10.4 4.4ZM20.8 10.4L18.9 12L20.8 13.6L19.6 13.6L17.5 12L19.6 10.4ZM13.6 20.8L12 18.9L10.4 20.8L10.4 19.6L12 17.5L13.6 19.6ZM3.2 13.6L5.1 12L3.2 10.4L4.4 10.4L6.5 12L4.4 13.6Z","mage",1.0,1],["M16.4 12L16.3 12.9L16.1 13.7L15.7 14.4L15.1 15.1L14.4 15.7L13.7 16.1L12.9 16.3L12 16.4L11.1 16.3L10.3 16.1L9.6 15.7L8.9 15.1L8.3 14.4L7.9 13.7L7.7 12.9L7.6 12L7.7 11.1L7.9 10.3L8.3 9.6L8.9 8.9L9.6 8.3L10.3 7.9L11.1 7.7L12 7.6L12.9 7.7L13.7 7.9L14.4 8.3L15.1 8.9L15.7 9.6L16.1 10.3L16.3 11.1Z","mage",0.35,0],["M15 12L14.9 12.8L14.6 13.5L14.1 14.1L13.5 14.6L12.8 14.9L12 15L11.2 14.9L10.5 14.6L9.9 14.1L9.4 13.5L9.1 12.8L9 12L9.1 11.2L9.4 10.5L9.9 9.9L10.5 9.4L11.2 9.1L12 9L12.8 9.1L13.5 9.4L14.1 9.9L14.6 10.5L14.9 11.2Z","starcore",1.0,1]]},"ab_charge":{"std":[["M15 18.8L14.9 19.6L14.8 19.9L14.4 20.6L14.1 20.9L13.8 21.2L13.2 21.6L12.8 21.7L12 21.8L11.2 21.7L10.8 21.6L10.2 21.2L9.9 20.9L9.6 20.6L9.2 19.9L9.1 19.6L9 18.8L9.1 18L9.2 17.6L9.6 17L9.9 16.7L10.2 16.4L10.8 16L11.2 15.9L12 15.8L12.8 15.9L13.2 16L13.8 16.4L14.1 16.7L14.4 17L14.8 17.6L14.9 18ZM14.2 18.5L14.1 18.2L13.9 17.7L13.6 17.2L13.3 17.1L13.1 16.9L12.6 16.7L12 16.6L11.7 16.6L11.4 16.7L10.9 16.9L10.4 17.2L10.2 17.5L10.1 17.7L9.9 18.2L9.8 18.8L9.8 19.1L9.9 19.4L10.1 19.9L10.4 20.4L10.7 20.6L10.9 20.7L11.4 20.9L12 21L12.3 21L12.6 20.9L13.1 20.7L13.6 20.4L13.8 20.1L13.9 19.9L14.1 19.4L14.2 18.8Z","warrior",0.6,0],["M9.4 9.6L14.6 9.6L12.8 16.2L11.2 16.2Z","warrior",1.0,1],["M12.2 15.9L12.1 16L11.9 16L11.8 16L11.8 15.8L11.6 10.2L11.6 10.1L11.7 9.9L11.8 9.8L12 9.8L12.2 9.8L12.3 9.9L12.4 10.1L12.4 10.2Z","forge_white",1,0],["M12 2.3L12.8 4.7L15.3 4.2L13.6 6.1L15.3 8L12.8 7.5L12 9.9L11.2 7.5L8.7 8L10.4 6.1L8.7 4.2L11.2 4.7Z","forge_white",1.0,1]]},"ab_freezing_trap":{"std":[["M12 2.1L20.6 7L20.6 16.9L12 21.9L3.4 16.9L3.4 7ZM4.4 7.6L4.4 16.4L12 20.8L19.6 16.4L19.6 7.6L12 3.2Z","rime",1,1],["M11.5 2.9L11.6 2.7L11.8 2.6L12 2.5L12.2 2.6L12.4 2.7L12.5 2.9L12.6 3.1L12.3 5.1L12.3 5.2L12.1 5.3L11.9 5.3L11.7 5.2L11.4 3.1ZM19.6 7L19.9 7L20 7.1L20.2 7.3L20.2 7.5L20.2 7.7L20.1 7.9L20 8L18.2 8.8L18.1 8.8L17.8 8.8L17.7 8.6L17.8 8.3L19.4 7.1ZM20.1 16.1L20.2 16.3L20.2 16.5L20.2 16.7L20 16.9L19.9 17L19.6 17L19.4 16.9L17.9 15.7L17.8 15.7L17.7 15.4L17.8 15.2L18.1 15.2L20 16ZM12.5 21.1L12.4 21.3L12.2 21.4L12 21.4L11.8 21.4L11.6 21.3L11.5 21.1L11.4 20.9L11.7 18.9L11.7 18.8L11.9 18.7L12.1 18.7L12.3 18.8L12.6 20.9ZM4.4 17L4.2 17L4 16.9L3.8 16.7L3.8 16.5L3.8 16.3L3.9 16.1L4 16L5.8 15.2L6 15.2L6.2 15.2L6.3 15.4L6.2 15.7L4.6 16.9ZM3.9 7.9L3.8 7.7L3.8 7.5L3.8 7.3L4 7.1L4.2 7L4.4 7L4.6 7.1L6.1 8.3L6.2 8.3L6.3 8.6L6.2 8.8L6 8.8L4 8Z","rime",1.0,1],["M12 6.4L16.9 9.2L16.9 14.8L12 17.6L7.2 14.8L7.2 9.2Z","frostglass",0.45,1],["M12 6.4L16.9 9.2L16.9 14.8L12 17.6L7.2 14.8L7.2 9.2ZM8 9.7L8 14.3L12 16.6L16 14.3L16 9.7L12 7.4Z","rime",1,0],["M11.6 7.5L11.6 7.4L11.8 7.2L12.1 7.1L12.3 7.3L16.2 14.1L16.3 14.3L16.1 14.6L15.9 14.7L15.8 14.6L15.7 14.6L15.6 14.4L11.7 7.7ZM11.8 7.2L11.9 7.1L12.2 7.2L12.4 7.4L12.3 7.7L8.4 14.4L8.3 14.6L8.2 14.6L8.1 14.7L7.9 14.6L7.7 14.3L7.8 14.1L11.7 7.3ZM8 10.1L7.7 9.9L7.7 9.8L7.7 9.6L7.8 9.5L8 9.4L8.1 9.3L15.9 9.3L16.1 9.4L16.2 9.5L16.3 9.6L16.3 9.8L16.3 9.9L16.2 10L16.1 10.1L15.9 10.2Z","rime",0.6,0]]},"ab_crippling_poison":{"std":[["M18 8.6L17.9 9.4L17.8 10.2L17.5 10.9L17.2 11.6L16.8 12.2L16.2 12.8L15.7 13.4L15 13.8L14.3 14.1L13.6 14.4L12.8 14.6L12 14.6L11.2 14.6L10.4 14.4L9.7 14.1L9 13.8L8.3 13.4L7.8 12.8L7.2 12.2L6.8 11.6L6.5 10.9L6.2 10.2L6 9.4L6 8.6L6 7.8L6.2 7L6.5 6.3L6.8 5.6L7.2 5L7.8 4.4L8.3 3.8L9 3.4L9.7 3.1L10.4 2.8L11.2 2.6L12 2.6L12.8 2.6L13.6 2.8L14.3 3.1L15 3.4L15.7 3.8L16.2 4.4L16.8 5L17.2 5.6L17.5 6.3L17.8 7L17.9 7.8ZM17.1 7.9L16.9 7.3L16.7 6.7L16.4 6L16.1 5.5L15.6 5L15.1 4.5L14.6 4.2L13.9 3.9L13.3 3.7L12.7 3.5L12 3.5L11.3 3.5L10.7 3.7L10.1 3.9L9.4 4.2L8.9 4.5L8.4 5L8 5.5L7.6 6L7.3 6.7L7.1 7.3L6.9 7.9L6.9 8.6L6.9 9.3L7.1 9.9L7.3 10.6L7.6 11.2L8 11.7L8.4 12.2L8.9 12.7L9.4 13L10.1 13.3L10.7 13.5L11.3 13.7L12 13.7L12.7 13.7L13.3 13.5L13.9 13.3L14.6 13L15.1 12.7L15.6 12.2L16.1 11.7L16.4 11.2L16.7 10.6L16.9 9.9L17.1 9.3L17.1 8.6Z","muted",0.7,0],["M15.4 4.3L15.7 4.3L15.9 4.3L16.1 4.5L16.3 4.7L16.3 4.9L16.3 5.2L16.1 5.4L14.1 6.7L14 6.8L13.9 6.7L13.8 6.6L13.9 6.5L15.2 4.5ZM16.3 12L16.3 12.3L16.3 12.5L16.1 12.7L15.9 12.9L15.7 12.9L15.4 12.9L15.2 12.7L13.9 10.7L13.8 10.6L13.9 10.5L14 10.4L14.1 10.5L16.1 11.8ZM8.6 12.9L8.3 12.9L8.1 12.9L7.9 12.7L7.7 12.5L7.7 12.3L7.7 12L7.9 11.8L9.9 10.5L10 10.4L10.1 10.5L10.2 10.6L10.1 10.7L8.8 12.7ZM7.7 5.2L7.7 4.9L7.7 4.7L7.9 4.5L8.1 4.3L8.3 4.3L8.6 4.3L8.8 4.5L10.1 6.5L10.2 6.6L10.1 6.7L10 6.8L9.9 6.7L7.9 5.4Z","rogue",1.0,1],["M10.2 21.4L11.9 9L12 8.9L12.1 9L13.8 21.4Z","venom_deep",1.0,1],["M11.4 21.4L11.9 10.6L12.1 10.6L12.6 21.4Z","blade",1,0]]},"ab_salvation":{"std":[["M7.5 20.9L7.3 20.9L7.1 20.9L6.9 20.8L6.8 20.6L6.8 20.4L6.8 20.2L7.3 19.7L8.2 18.9L8.7 18.6L9.7 18.1L10.8 17.8L12 17.7L13.2 17.8L13.7 17.9L14.3 18.1L15.3 18.6L15.8 18.9L16.7 19.7L17.1 20.1L17.2 20.2L17.2 20.4L17.2 20.6L17.1 20.8L16.9 20.9L16.7 20.9L16.5 20.9L15.7 20L14.8 19.5L14.4 19.2L13.5 18.9L13 18.8L12.5 18.7L11.5 18.7L10.5 18.9L9.6 19.2L9.2 19.5L8.3 20Z","muted",0.7,0],["M13.2 20.7L12.9 21.1L12.5 21.4L12 21.5L11.5 21.4L11.1 21.1L10.8 20.7L10.7 20.2L10.7 14.6L10.8 14.1L11.1 13.7L11.5 13.4L12 13.3L12.5 13.4L12.9 13.7L13.2 14.1L13.3 14.6L13.3 20.2Z","priest",1.0,1],["M12.4 19.9L12.2 20.2L12 20.2L11.8 20.2L11.7 20.1L11.6 19.9L11.6 19.8L11.6 14.8L11.6 14.7L11.7 14.5L11.8 14.4L12 14.4L12.2 14.4L12.3 14.5L12.4 14.7L12.4 14.8Z","starcore",1,0],["M8.5 5.2L8.5 5.4L8.5 5.7L8.3 5.9L8.1 6.1L7.8 6.1L7.6 6.1L7.3 5.9L7.1 5.5L6.5 4.5L6.5 4.2L6.7 4.1L6.9 4.1L8.3 4.9ZM16.4 6.1L16.2 6.1L15.9 6.1L15.7 5.9L15.5 5.7L15.5 5.4L15.5 5.2L15.7 4.9L16.1 4.7L17.1 4.1L17.4 4.1L17.5 4.2L17.5 4.5L16.7 5.9ZM15.5 14L15.5 13.8L15.5 13.5L15.7 13.3L15.9 13.1L16.2 13.1L16.4 13.1L16.7 13.3L16.9 13.7L17.5 14.7L17.5 14.9L17.4 15.1L17.1 15.1L15.7 14.3ZM7.6 13.1L7.8 13.1L8.1 13.1L8.3 13.3L8.5 13.5L8.5 13.8L8.5 14L8.3 14.3L7.9 14.5L6.9 15.1L6.7 15.1L6.5 14.9L6.5 14.7L7.3 13.3Z","priest",0.8,1],["M16.8 9.6L16.8 10.2L16.6 10.8L16.4 11.4L16.2 12L15.8 12.5L15.4 13L14.9 13.4L14.4 13.8L13.8 14L13.2 14.2L12.6 14.4L12 14.4L11.4 14.4L10.8 14.2L10.2 14L9.6 13.8L9.1 13.4L8.6 13L8.2 12.5L7.8 12L7.6 11.4L7.4 10.8L7.2 10.2L7.2 9.6L7.2 9L7.4 8.4L7.6 7.8L7.8 7.2L8.2 6.7L8.6 6.2L9.1 5.8L9.6 5.4L10.2 5.2L10.8 5L11.4 4.8L12 4.8L12.6 4.8L13.2 5L13.8 5.2L14.4 5.4L14.9 5.8L15.4 6.2L15.8 6.7L16.2 7.2L16.4 7.8L16.6 8.4L16.8 9ZM15.8 9.1L15.7 8.6L15.5 8.2L15.3 7.7L14.7 6.9L14.3 6.6L13.9 6.3L13.4 6.1L13 5.9L12.5 5.8L12 5.8L11.5 5.8L11 5.9L10.6 6.1L10.1 6.3L9.3 6.9L9 7.3L8.7 7.7L8.5 8.2L8.3 8.6L8.2 9.1L8.2 9.6L8.2 10.1L8.3 10.6L8.5 11.1L8.7 11.5L9.3 12.3L9.7 12.6L10.1 12.9L10.6 13.1L11 13.3L11.5 13.4L12 13.4L12.5 13.4L13 13.3L13.4 13.1L13.9 12.9L14.7 12.3L15 11.9L15.3 11.5L15.5 11.1L15.7 10.6L15.8 10.1L15.8 9.6Z","priest",1,1],["M15.8 9.6L15.7 10.3L15.5 11.1L15.2 11.7L14.7 12.3L14.1 12.8L13.4 13.1L12.7 13.3L12 13.4L11.3 13.3L10.6 13.1L9.9 12.8L9.3 12.3L8.8 11.7L8.5 11.1L8.3 10.3L8.2 9.6L8.3 8.9L8.5 8.2L8.8 7.5L9.3 6.9L9.9 6.4L10.6 6.1L11.3 5.9L12 5.8L12.7 5.9L13.4 6.1L14.1 6.4L14.7 6.9L15.2 7.5L15.5 8.2L15.7 8.9Z","dawnlight",0.95,0]]}}}',
  ),
);
// GENERATED SIGIL DATA END

/**
 * Find the Priest's cross inside one master of its generated emblem body.
 *
 * body is the master's `body` path (absolute M, L and Z commands only, a
 * 24-unit box): the sun ring, two rounded cross bars, eight rays and the
 * ring's inner edge. A bar is a subpath whose bounding box is centered on
 * (12, 12) within 0.1 units and is at least twice as long as it is wide; the
 * ring, its inner edge and the rays are not. Return {bars, silhouette}: bars
 * joins the two bar subpaths exactly as generated (the sigil fill and its Ink
 * keyline), and silhouette is the cross's one outline, a 12-corner polygon
 * around both bars' boxes (the "outline" style). Throw Error when the body
 * does not hold exactly one upright and one level bar, so a change to the
 * generated data fails at load instead of drawing a wrong cross. Pure.
 *
 * @param {string} body
 * @returns {{bars: string, silhouette: string}}
 */
function priestCross(body) {
  /** @type {Array<{d: string, x0: number, y0: number, x1: number, y1: number}>} */
  const bars = [];
  for (const d of body.match(/M[^M]*/gu) ?? []) {
    const numbers = (d.match(/-?\d+(?:\.\d+)?/gu) ?? []).map(Number);
    const xs = numbers.filter((_value, index) => index % 2 === 0);
    const ys = numbers.filter((_value, index) => index % 2 === 1);
    const box = {
      d,
      x0: Math.min(...xs),
      y0: Math.min(...ys),
      x1: Math.max(...xs),
      y1: Math.max(...ys),
    };
    const width = box.x1 - box.x0;
    const height = box.y1 - box.y0;
    const centered =
      Math.abs((box.x0 + box.x1) / 2 - 12) <= 0.1 &&
      Math.abs((box.y0 + box.y1) / 2 - 12) <= 0.1;
    if (centered && Math.max(width, height) >= 2 * Math.min(width, height)) {
      bars.push(box);
    }
  }
  const upright = bars.find((bar) => bar.y1 - bar.y0 > bar.x1 - bar.x0);
  const level = bars.find((bar) => bar.x1 - bar.x0 > bar.y1 - bar.y0);
  if (bars.length !== 2 || upright === undefined || level === undefined) {
    throw new Error("The Priest emblem body must hold exactly two cross bars.");
  }
  const u = upright;
  const v = level;
  return {
    bars: `${u.d}${v.d}`,
    silhouette:
      `M${u.x0} ${u.y0}H${u.x1}V${v.y0}H${v.x1}V${v.y1}H${u.x1}V${u.y1}` +
      `H${u.x0}V${v.y1}H${v.x0}V${v.y0}H${u.x0}Z`,
  };
}

/**
 * The Priest's cross alone, by master (micro, standard, display), cut from
 * the generated emblem (priestCross). A Priest token that stands on its Class
 * Seal draws only this cross, because the seal is its sun (spec
 * `specs/champions.json`, Class Seals: "The Priest emblem drops its own ring
 * and rays under this seal and shows the cross only").
 *
 * @type {Readonly<Record<string, Readonly<{bars: string, silhouette: string}>>>}
 */
const PRIEST_CROSS = deepFreeze(
  Object.fromEntries(
    Object.entries(SIGIL_DATA.emblems.priest).map(([master, geometry]) => [
      master,
      priestCross(geometry.body),
    ]),
  ),
);

/**
 * Hand-drawn sigils the design store does not have, in the generated glyph
 * format (glyphs.<name>.std = [[d, paint, fillOpacity, keyline]], a 24-unit
 * box; no "mic" set, so every size draws "std"). One so far:
 * - rest_vessel: the icon of Rest (Regeneration), made from its arena mark
 *   (owner, 2 October 2026: "Rest needs a small readable icon made from its
 *   real mark"): the life vessel ring, its pale Rest level and the drop that
 *   pours in, all in the Rest color (paint "dew", the --moon-dew token).
 *
 * @type {Readonly<Record<string, Readonly<Record<string, ReadonlyArray<readonly [string, string, number, number]>>>>>}
 */
const HAND_GLYPHS = deepFreeze({
  rest_vessel: {
    std: [
      [
        "M3.4 14.6A8.6 8.6 0 0 0 20.6 14.6A8.6 8.6 0 0 0 3.4 14.6Z M5.6 14.6A6.4 6.4 0 0 1 18.4 14.6A6.4 6.4 0 0 1 5.6 14.6Z",
        "dew",
        0.8,
        1,
      ],
      ["M6.4 15.2A5.6 5.6 0 0 0 17.6 15.2Z", "dew", 1, 0],
      [
        "M12 0.6C12 0.6 9.5 3.9 9.5 5.3A2.5 2.5 0 0 0 14.5 5.3C14.5 3.9 12 0.6 12 0.6Z",
        "dew",
        1,
        1,
      ],
    ],
  },
});

/**
 * Turn a store glyph name into its public sigil key: "dazed_rogue" ->
 * "sigil-dazed-rogue", "crest_A" -> "sigil-crest-a".
 *
 * @param {string} name
 * @returns {string}
 */
function sigilKeyFor(name) {
  return `sigil-${name.toLowerCase().replaceAll("_", "-")}`;
}

/**
 * Every sigil key -> its definition (new keys, the HAND_GLYPHS keys and old
 * aliases together).
 *
 * @type {ReadonlyMap<string, Readonly<SigilDefinition>>}
 */
const SIGILS = (() => {
  /** @type {Map<string, Readonly<SigilDefinition>>} */
  const map = new Map();
  for (const name of CLASS_NAMES) {
    const glyphKey = `emblem-${name}`;
    map.set(
      glyphKey,
      Object.freeze({ glyphKey, kind: "emblem", name, classPaint: null }),
    );
  }
  for (const name of [...Object.keys(SIGIL_DATA.glyphs), ...Object.keys(HAND_GLYPHS)]) {
    const glyphKey = sigilKeyFor(name);
    map.set(
      glyphKey,
      Object.freeze({ glyphKey, kind: "glyph", name, classPaint: null }),
    );
  }
  for (const [glyphKey, [name, kind, classPaint]] of Object.entries(OLD_KEY_SIGILS)) {
    map.set(glyphKey, Object.freeze({ glyphKey, kind, name, classPaint }));
  }
  return map;
})();

/** Every key createSvgIcon draws: the plain keys, then the new sigil keys. */
export const KNOWN_GLYPH_KEYS = Object.freeze([
  ...Object.keys(ICONS),
  ...[...SIGILS.keys()].filter((key) => !Object.hasOwn(ICONS, key)),
]);

/**
 * Look up the plain (monoline) glyph for glyphKey after trimming string input.
 * A new sigil key returns its plain twin (for example sigil-frozen gives
 * status-stun); nonstring, empty or unlisted input gives the shared unknown
 * icon. Return the shared frozen definition. No DOM is created and the
 * registry is not changed.
 *
 * @param {unknown} glyphKey
 * @returns {Readonly<IconDefinition>}
 */
export function iconDefinition(glyphKey) {
  if (typeof glyphKey !== "string" || !glyphKey.trim()) {
    return ICONS.unknown;
  }
  const normalized = glyphKey.trim();
  if (Object.hasOwn(ICONS, normalized)) {
    return ICONS[normalized];
  }
  const sigil = SIGILS.get(normalized);
  if (sigil?.kind === "emblem") {
    return ICONS[`class-${sigil.name}`];
  }
  const twin = sigil ? SIGIL_PLAIN_TWINS[sigil.name] : undefined;
  return twin ? ICONS[twin] : ICONS.unknown;
}

/**
 * Look up the Heraldic Sigil for glyphKey (an old key or a new `emblem-*` /
 * `sigil-*` key) after trimming string input. Return the shared frozen
 * definition, or null when the key has no sigil (for example "unknown" or a
 * nonstring). No DOM is created.
 *
 * @param {unknown} glyphKey
 * @returns {Readonly<SigilDefinition> | null}
 */
export function sigilDefinition(glyphKey) {
  if (typeof glyphKey !== "string") {
    return null;
  }
  return SIGILS.get(glyphKey.trim()) ?? null;
}

/**
 * Return the sigil key for a status token id from vocabulary.js (for example
 * "stun_rogue_poison" -> "sigil-dazed-rogue", "stun_hunter_trap" ->
 * "sigil-frozen", "in_combat" -> "sigil-in-combat", "spawn_shield" ->
 * "sigil-moonward"; also "rest" -> "sigil-rest-vessel" and "stun_broken" ->
 * "sigil-life-shattered"), or null for an unknown token. The color comes from the
 * class named by the status itself, so it reveals nothing beyond the status.
 *
 * @param {unknown} statusTokenId
 * @returns {string | null}
 */
export function statusSigilKey(statusTokenId) {
  return typeof statusTokenId === "string" &&
    Object.hasOwn(STATUS_SIGILS, statusTokenId)
    ? sigilKeyFor(STATUS_SIGILS[statusTokenId])
    : null;
}

/**
 * Return the sigil key for an aura modifier token id ("mage_amplification" ->
 * "sigil-empowerment", "warrior_mitigation" -> "sigil-barrier"), or null for
 * an unknown token.
 *
 * @param {unknown} modifierTokenId
 * @returns {string | null}
 */
export function modifierSigilKey(modifierTokenId) {
  return typeof modifierTokenId === "string" &&
    Object.hasOwn(MODIFIER_SIGILS, modifierTokenId)
    ? sigilKeyFor(MODIFIER_SIGILS[modifierTokenId])
    : null;
}

/**
 * Return the emblem master the store uses at a drawn size: "micro" below 14
 * CSS pixels, "standard" from 14 to under 32, "display" from 32 up. A missing
 * or non-finite size means "standard".
 *
 * @param {number | undefined} size Drawn width in CSS pixels.
 * @returns {"micro" | "standard" | "display"}
 */
export function sigilMasterForSize(size) {
  if (typeof size !== "number" || !Number.isFinite(size)) return "standard";
  return size < 14 ? "micro" : size < 32 ? "standard" : "display";
}

/**
 * Append one path to svg with the given attributes (values turned to strings;
 * null or undefined values are skipped). Return the path.
 *
 * @param {Document} ownerDocument
 * @param {SVGSVGElement} svg
 * @param {Record<string, string | number | null | undefined>} attributes
 * @returns {SVGPathElement}
 */
function appendPath(ownerDocument, svg, attributes) {
  const path = /** @type {SVGPathElement} */ (
    ownerDocument.createElementNS(SVG_NAMESPACE, "path")
  );
  for (const [name, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined) {
      path.setAttribute(name, String(value));
    }
  }
  svg.append(path);
  return path;
}

/**
 * The CSS class that paints a sigil part. "cls" (and a classPaint swapped to
 * the caller's color) uses currentColor; every other paint is a color token.
 *
 * @param {string} paint Store paint name, such as "rime" or "mage".
 * @param {string | null} classPaint Class drawn in currentColor, or null.
 * @returns {string}
 */
function paintClass(paint, classPaint) {
  const name = paint === "cls" || paint === classPaint ? "current" : paint;
  return `sigil-paint sigil-paint--${name.replaceAll("_", "-")}`;
}

/**
 * Draw a class emblem's parts into svg (store `emblem`, sigil style).
 *
 * crossOnly true (a Priest on its Class Seal) draws the Priest's cross alone
 * (PRIEST_CROSS bars as both keyline and body, no sun ring, rays, jewels or
 * filigree); the caller passes it only for the Priest. DOM only.
 *
 * @param {Document} ownerDocument
 * @param {SVGSVGElement} svg
 * @param {string} className mage, warrior, hunter, rogue or priest.
 * @param {"micro" | "standard" | "display"} master
 * @param {EmblemPaint} paint
 * @param {1 | 2 | null} teamId Team for the "host" paint (1 Team A, 2 Team B).
 * @param {boolean} [crossOnly] Default false.
 */
function drawEmblem(ownerDocument, svg, className, master, paint, teamId, crossOnly) {
  const cross = crossOnly === true ? PRIEST_CROSS[master] : undefined;
  const geometry =
    cross === undefined
      ? SIGIL_DATA.emblems[className][master]
      : { outline: cross.bars, body: cross.bars, jewels: [], filigree: [] };
  if (paint === "flare") {
    appendPath(ownerDocument, svg, {
      d: geometry.body,
      class: paintClass("starcore", null),
    });
    return;
  }
  const keyWidth = master === "micro" ? 2 : 3;
  const team = teamId === 2 ? "team_b" : "team_a";
  const bodyPaint =
    paint === "host" ? team : paint === "ash" ? "ash" : `class_${className}`;
  const jewelPaint =
    paint === "host" ? `${team}_light` : paint === "ash" ? "ash" : bodyPaint;
  const alpha = paint === "ash" ? 0.55 : null;
  if (paint === "ready") {
    appendPath(ownerDocument, svg, {
      d: geometry.outline,
      class: "sigil-keyline",
      "stroke-width": keyWidth,
      "vector-effect": "non-scaling-stroke",
    });
  }
  appendPath(ownerDocument, svg, {
    d: geometry.outline,
    class:
      paint === "ready"
        ? `sigil-keyline sigil-keyline--ready ${paintClass(bodyPaint, null)}`
        : "sigil-keyline",
    "stroke-width": keyWidth,
    "stroke-opacity": paint === "ready" ? 0.45 : null,
    "vector-effect": "non-scaling-stroke",
    opacity: alpha,
  });
  appendPath(ownerDocument, svg, {
    d: geometry.body,
    class: `sigil-body ${paintClass(bodyPaint, null)}`,
    opacity: alpha,
  });
  for (const [d, jewelKind] of geometry.jewels) {
    const starcore =
      jewelKind.startsWith("starcore") && (paint === "class" || paint === "ready");
    appendPath(ownerDocument, svg, {
      d,
      class: `sigil-jewel ${paintClass(starcore ? "starcore" : jewelPaint, null)}${
        jewelKind === "starcore_plain" ? "" : " sigil-jewel--ringed"
      }`,
      "stroke-width": jewelKind === "starcore_plain" ? null : 1,
      "vector-effect": jewelKind === "starcore_plain" ? null : "non-scaling-stroke",
      opacity: alpha,
    });
  }
  if (master === "display" && paint === "class") {
    for (const [d, filigreePaint, strokeOpacity] of geometry.filigree) {
      appendPath(ownerDocument, svg, {
        d,
        class: `sigil-filigree ${paintClass(filigreePaint, null)}`,
        "stroke-opacity": strokeOpacity,
        "stroke-width": 1,
        "vector-effect": "non-scaling-stroke",
      });
    }
  }
}

/**
 * Draw a sigil in the "outline" style into svg: thin lines in the svg's
 * currentColor stroke, no fills, no class colors. A class emblem draws its
 * silhouette (the keyline outline) alone; a glyph draws every layer's shape.
 * ash true (the fallen "ash" paint) draws it at 55% opacity. crossOnly true
 * (a Priest on its Class Seal; the caller passes it only for the Priest)
 * draws the cross's own silhouette (PRIEST_CROSS) instead of the sun. DOM
 * only.
 *
 * @param {Document} ownerDocument
 * @param {SVGSVGElement} svg
 * @param {Readonly<SigilDefinition>} sigil
 * @param {"micro" | "standard" | "display"} master
 * @param {boolean} ash
 * @param {boolean} [crossOnly] Default false.
 */
function drawOutline(ownerDocument, svg, sigil, master, ash, crossOnly) {
  const opacity = ash ? 0.55 : null;
  if (sigil.kind === "emblem") {
    const cross = crossOnly === true ? PRIEST_CROSS[master] : undefined;
    appendPath(ownerDocument, svg, {
      d:
        cross === undefined
          ? SIGIL_DATA.emblems[sigil.name][master].outline
          : cross.silhouette,
      class: "sigil-outline",
      opacity,
    });
    return;
  }
  const sets = SIGIL_DATA.glyphs[sigil.name] ?? HAND_GLYPHS[sigil.name];
  const layers = master === "micro" && sets.mic ? sets.mic : sets.std;
  for (const [d] of layers) {
    appendPath(ownerDocument, svg, { d, class: "sigil-outline", opacity });
  }
}

/**
 * Draw a store glyph's layers into svg (store `glyph`, sigil style; a
 * HAND_GLYPHS sigil the same way): one Ink keyline under every layer marked
 * for it, then each colored layer.
 *
 * @param {Document} ownerDocument
 * @param {SVGSVGElement} svg
 * @param {string} name Store glyph name, such as "frozen".
 * @param {boolean} micro True below 14 CSS pixels (uses the "mic" set).
 * @param {string | null} classPaint Class color drawn in currentColor, or null.
 */
function drawGlyph(ownerDocument, svg, name, micro, classPaint) {
  const sets = SIGIL_DATA.glyphs[name] ?? HAND_GLYPHS[name];
  const layers = micro && sets.mic ? sets.mic : sets.std;
  const keyline = layers
    .filter((layer) => layer[3] === 1)
    .map((layer) => layer[0])
    .join("");
  if (keyline) {
    appendPath(ownerDocument, svg, {
      d: keyline,
      class: "sigil-keyline sigil-keyline--filled",
      "stroke-width": micro ? 2 : 2.5,
      "vector-effect": "non-scaling-stroke",
    });
  }
  for (const [d, paint, fillOpacity] of layers) {
    appendPath(ownerDocument, svg, {
      d,
      class: paintClass(paint, classPaint),
      "fill-opacity": fillOpacity < 1 ? fillOpacity : null,
    });
  }
}

/**
 * Sigil names that Status Icons From Marks governs: every status, aura and
 * life-change sigil (the small copies of body marks), plus In Combat, Rest,
 * Moon Dew and Unseen. Class emblems follow Lore Emblems instead; team crests,
 * ability icons, tier seals and waystones follow neither switch.
 */
const STATUS_FAMILY_NAMES = new Set([
  ...Object.values(STATUS_SIGILS),
  ...Object.values(MODIFIER_SIGILS),
  "unseen",
  "moon_dew",
  "life_applied",
  "life_refreshed",
  "life_decremented",
  "life_expired",
  "life_ended",
  "life_thawed",
  "life_shattered",
  "life_death_clear",
  "life_broken_reapplied",
]);

/**
 * Every IconStyle value.
 *
 * @type {ReadonlySet<unknown>}
 */
const ICON_STYLES = new Set(["sigil", "plain", "outline"]);

/**
 * The page-wide style a call without `options.style` gets.
 *
 * @type {Readonly<{classEmblems: IconStyle, statusIcons: IconStyle}>}
 */
let styleDefaults = Object.freeze({ classEmblems: "sigil", statusIcons: "sigil" });

/**
 * Set the style that createSvgIcon draws when its caller gives no style.
 *
 * The page calls this from its Visual Filters state before each repaint, so a
 * switch changes its mark everywhere it is drawn (bodies, deck, scorebug, rail,
 * cards, panels) in all three clients:
 * - classEmblems: "sigil" (Lore Emblems on, the default) or "outline" (Lore
 *   Emblems off: the outline class emblem). Applies to `class-*` and
 *   `emblem-*` keys.
 * - statusIcons: "sigil" (Status Icons From Marks on, the default) or
 *   "outline" (off: plain line icons with the same meaning). Applies to
 *   status, aura, life-change, In Combat, Rest and Unseen keys.
 * "plain" is accepted too. A missing or unknown value keeps "sigil". An explicit `options.style` always
 * wins. Module state only: it changes no existing node, so the caller repaints.
 * It reveals nothing: it only picks a drawing style.
 *
 * @param {{classEmblems?: unknown, statusIcons?: unknown}} [defaults]
 * @returns {void}
 */
export function setIconStyleDefaults(defaults = {}) {
  styleDefaults = Object.freeze({
    classEmblems: ICON_STYLES.has(defaults.classEmblems)
      ? /** @type {IconStyle} */ (defaults.classEmblems)
      : "sigil",
    statusIcons: ICON_STYLES.has(defaults.statusIcons)
      ? /** @type {IconStyle} */ (defaults.statusIcons)
      : "sigil",
  });
}

/**
 * Return the page-wide default style for glyphKey (see setIconStyleDefaults):
 * classEmblems for a class emblem, statusIcons for a status-family sigil,
 * otherwise "sigil". Pure apart from reading that module state.
 *
 * @param {unknown} glyphKey
 * @returns {IconStyle}
 */
export function defaultIconStyle(glyphKey) {
  const sigil = sigilDefinition(glyphKey);
  if (sigil === null) return "sigil";
  if (sigil.kind === "emblem") return styleDefaults.classEmblems;
  return STATUS_FAMILY_NAMES.has(sigil.name) ? styleDefaults.statusIcons : "sigil";
}

/**
 * Create a detached SVG element for a glyph key.
 *
 * ownerDocument creates every SVG node. glyphKey is any key in
 * KNOWN_GLYPH_KEYS (old keys and new `emblem-*` / `sigil-*` keys, trimmed);
 * anything else draws the unknown glyph. options (default {}):
 * - style: "sigil" draws the Heraldic Sigil when the key has one, else the
 *   plain glyph; "plain" always draws today's monoline glyph; "outline" draws
 *   the sigil's shapes as thin currentColor lines (the plain glyph for a key
 *   with no sigil). Default: the page-wide style for the key
 *   (defaultIconStyle), which is "outline" for a class emblem while Lore
 *   Emblems is off and for a status-family icon while Status Icons From Marks
 *   is off, and "sigil" otherwise.
 * - size: the drawn width in CSS pixels, used only to pick the master (below
 *   14 px the simpler micro shapes; from 32 px the display emblem with
 *   filigree). Default: standard. It does not set width or height.
 * - paint: class emblems only (see EmblemPaint); default "class".
 * - teamId: 1 or 2, for paint "host"; default Team A.
 * - onClassSeal: true when the emblem stands on its Class Seal (the Champions
 *   look passes it while it draws a seal or a seal's ash outline). The Priest
 *   emblem then draws its cross alone, because the seal is its sun; the
 *   "sigil" and "outline" styles follow it, every other key (and the "plain"
 *   style) ignores it, and the svg gets data-icon-variant "cross". Default
 *   false: the full sun cross.
 * - className is copied when truthy. A nonempty trimmed accessibleName makes
 *   a labeled image; otherwise the icon is decorative (aria-hidden).
 * Return the new SVGSVGElement (viewBox 0 0 24 24) with data-icon set to the
 * key drawn and data-icon-style to the style drawn. The caller attaches and
 * sizes it. Plain icons stroke in currentColor; sigils fill through
 * `sigil-paint--<paint>` classes (src/styles/icons.css), and their keylines
 * keep a fixed pixel width at any size (non-scaling strokes). No network
 * access or existing DOM mutation occurs.
 *
 * @param {Document} ownerDocument
 * @param {unknown} glyphKey
 * @param {{
 *   accessibleName?: string,
 *   className?: string,
 *   style?: IconStyle,
 *   size?: number,
 *   paint?: EmblemPaint,
 *   teamId?: 1 | 2 | null,
 *   onClassSeal?: boolean,
 * }} [options]
 * @returns {SVGSVGElement}
 */
export function createSvgIcon(ownerDocument, glyphKey, options = {}) {
  const style = options.style ?? defaultIconStyle(glyphKey);
  const sigil = style === "plain" ? null : sigilDefinition(glyphKey);
  const outline = style === "outline" && sigil !== null;
  // A Priest on its Class Seal shows its cross alone: the seal is its sun.
  const crossOnly =
    options.onClassSeal === true && sigil?.kind === "emblem" && sigil.name === "priest";
  const definition = iconDefinition(glyphKey);
  const svg = /** @type {SVGSVGElement} */ (
    ownerDocument.createElementNS(SVG_NAMESPACE, "svg")
  );
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("focusable", "false");
  if (outline) {
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "1.6");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.dataset.icon = sigil.glyphKey;
    svg.dataset.iconStyle = "outline";
  } else if (sigil) {
    svg.setAttribute("fill", "currentColor");
    svg.setAttribute("stroke", "none");
    svg.setAttribute("stroke-linejoin", "round");
    svg.dataset.icon = sigil.glyphKey;
    svg.dataset.iconStyle = "sigil";
  } else {
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "1.8");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.dataset.icon = definition.glyphKey;
    svg.dataset.iconStyle = "plain";
  }
  if (options.className) {
    svg.setAttribute("class", options.className);
  }
  if (crossOnly) {
    svg.dataset.iconVariant = "cross";
  }

  const accessibleName =
    typeof options.accessibleName === "string" &&
    options.accessibleName.trim().length > 0
      ? options.accessibleName.trim()
      : null;
  if (accessibleName) {
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", accessibleName);
  } else {
    svg.setAttribute("aria-hidden", "true");
  }

  if (outline) {
    drawOutline(
      ownerDocument,
      svg,
      sigil,
      sigilMasterForSize(options.size),
      options.paint === "ash",
      crossOnly,
    );
    return svg;
  }
  if (sigil?.kind === "emblem") {
    drawEmblem(
      ownerDocument,
      svg,
      sigil.name,
      sigilMasterForSize(options.size),
      options.paint ?? "class",
      options.teamId === 2 ? 2 : 1,
      crossOnly,
    );
    return svg;
  }
  if (sigil) {
    drawGlyph(
      ownerDocument,
      svg,
      sigil.name,
      sigilMasterForSize(options.size) === "micro",
      sigil.classPaint,
    );
    return svg;
  }
  for (const descriptor of definition.primitives) {
    const element = ownerDocument.createElementNS(SVG_NAMESPACE, descriptor.tag);
    for (const [name, value] of Object.entries(descriptor.attributes)) {
      element.setAttribute(name, String(value));
    }
    svg.append(element);
  }
  return svg;
}
