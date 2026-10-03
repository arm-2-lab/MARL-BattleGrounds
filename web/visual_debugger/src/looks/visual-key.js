/**
 * @file The Visual Key: a picture dictionary of every symbol a viewer can meet,
 * opened by the always-visible Visual Key (book) button beside Help.
 *
 * Owns:
 * - records(input): one entry per symbol of `symbols.js` (grouped as
 *   SYMBOL_GROUPS lists them) with its title, plain sentence and its numbers
 *   from the host mechanics catalog, plus the Camera And Keys rows from
 *   `command-set.js` for the client.
 * - drawSymbolPicture(symbolId, doc): a small standalone SVG of one symbol,
 *   drawn live by the real look module's records() and build() on fixed example
 *   data (P16: no demo recording). Every arena symbol has an example, events
 *   included: Basics, Ultimates, life numbers, Fall, Fallen, Returns In,
 *   Return, Start Spot, Last Seen and the auras are drawn on a fixed
 *   two-champion example. A look may instead export
 *   visualKeyExample(symbolId, {doc, transform}) to draw its own still example.
 *   Only the HUD Marks (HTML, not SVG) and the camera keys V, B and F get a
 *   small drawn sign; data-picture-source says which.
 * - build(records, doc): the dictionary's sections and cards (HTML).
 * - createVisualKey(options): the controller main.js mounts: the button, the
 *   modal <dialog> and its Close and Escape.
 *
 * Boundary: pictures use fixed example records and numbers come from the
 * static class mechanics catalog of the audience's own arena scene, never the
 * live match and never the researcher view, so the Key shows nothing an
 * audience could be denied. It is identical in Oracle View and Agent POV.
 * Export-safe SVG only: flat fills, strokes, opacity and transforms.
 */
import { CLASS_IDENTITIES } from "../agent-identity.js";
import { commandHelpRows } from "../command-set.js";
import { formatDisplayNumber } from "../display.js";
import { createViewportTransform } from "../layout.js";
import { SYMBOL_GROUPS, SYMBOLS, symbolById } from "../symbols.js";
import { createSvgIcon } from "../icons.js";
import { DEFAULT_VISUAL_FILTER_STATE } from "../visual-filters.js";
import { createSymbolDescriptor } from "../tooltip.js";
import * as basics from "./basics.js";
import * as champions from "./champions.js";
import * as fog from "./fog.js";
import * as healthText from "./health-text.js";
import * as hud from "./hud.js";
import { EMPTY_BUDGET, isRecord, SVG_NAMESPACE } from "./look-support.js";
import * as movement from "./movement.js";
import * as obstacles from "./obstacles.js";
import * as redZone from "./red-zone.js";
import * as requiems from "./requiems.js";
import * as spawns from "./spawns.js";
import * as statuses from "./statuses.js";
import * as ultimates from "./ultimates.js";

/**
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 * @typedef {import("../symbols.js").SymbolEntry} SymbolEntry
 */

/**
 * The static catalog the Key reads its numbers from: the audience scene's
 * class_mechanics rows, spawn_shield_mechanics, map.red_zone and
 * respawn_waves. Every field may be missing; a missing number is left out.
 *
 * @typedef {{
 *   classMechanics?: ReadonlyArray<Record<string, any>> | null,
 *   spawnShieldMechanics?: Record<string, any> | null,
 *   redZone?: Record<string, any> | null,
 *   respawnWaves?: ReadonlyArray<Record<string, any>> | null,
 * }} VisualKeyCatalogue
 */

/** Picture size in CSS pixels. */
export const VISUAL_KEY_PICTURE = Object.freeze({ width: 112, height: 72 });

/** The Key draws nothing on the arena: no allocator footprint. */
export const FOOTPRINT = Object.freeze({});

/** The Key draws nothing on the arena: no nodes or animations there. */
export const BUDGET = EMPTY_BUDGET;

/**
 * No switch: the Visual Key button is always shown, like Help (owner, 1
 * October 2026), and the Key draws only when opened.
 *
 * @type {ReadonlyArray<import("./look-support.js").PaintPartRegistration>}
 */
export const PAINT_PARTS = Object.freeze([]);

/**
 * The example world of a one-champion, stone, Red Zone, fog or map picture,
 * in world units. A 0.5-unit body is about 19 px across its radius here.
 */
const WORLD = Object.freeze({ width: 2.6, height: 1.7 });

/**
 * The wider example world of an event picture (a Basic, Ultimate, life number,
 * Fall, Return, Start Spot or Last Seen), in world units: room for two
 * champions about 2 units apart, with a 0.5-unit body still about 15 px in
 * radius, so the Basics look draws at its detail level 2.
 */
const EVENT_WORLD = Object.freeze({ width: 3.4, height: 2.2 });

/**
 * The example world of a status picture, in world units: closer than WORLD,
 * so the 0.5-unit body is about 33 px across its radius and its small Rune
 * Zones marks (stars, beads, veins, wings, runes) read clearly.
 */
const STATUS_WORLD = Object.freeze({ width: 1.42, height: 0.92 });

/**
 * The widest example world, for an aura picture, in world units: the whole
 * 2.0-unit reach of a Ward Circle fits, drawn at its true size around its
 * caster.
 */
const AURA_WORLD = Object.freeze({ width: 6.8, height: 4.4 });

/** The real reach of a Mage or Warrior aura, in world units (Ward Circle). */
const AURA_REACH = 2;

/**
 * The example Ultimate of each class: the host token ID the Ultimates look
 * reads, and whether its target is an ally (Salvation) or nobody (Burst).
 */
const ULTIMATE_EXAMPLES = Object.freeze({
  mage: Object.freeze({ tokenId: "mage_burst", target: "none" }),
  warrior: Object.freeze({ tokenId: "warrior_charge", target: "enemy" }),
  hunter: Object.freeze({ tokenId: "hunter_trap", target: "enemy" }),
  rogue: Object.freeze({ tokenId: "rogue_poison", target: "enemy" }),
  priest: Object.freeze({ tokenId: "holy_word", target: "ally" }),
});

/** The example aura field of each aura symbol (host aura and token IDs). */
const AURA_EXAMPLES = Object.freeze({
  aura_empowerment: Object.freeze({
    classId: 1,
    auraId: "mage_damage_amplification",
    tokenId: "mage_amplification",
  }),
  aura_barrier: Object.freeze({
    classId: 2,
    auraId: "warrior_damage_mitigation",
    tokenId: "warrior_mitigation",
  }),
});

/** The look module asked first for each symbol (its visualKeyExample hook). */
const EXAMPLE_OWNERS = Object.freeze({
  blow: "basics",
  basic_mage: "basics",
  basic_warrior: "basics",
  basic_hunter: "basics",
  basic_rogue: "basics",
  basic_priest: "basics",
  ult_mage: "ultimates",
  ult_warrior: "ultimates",
  ult_hunter: "ultimates",
  ult_rogue: "ultimates",
  ult_priest: "ultimates",
  damage_number: "health-text",
  heal_number: "health-text",
  fall: "requiems",
  fallen: "requiems",
  returns_in: "requiems",
  respawn: "requiems",
  imprint: "movement",
  last_seen: "fog",
  stun_sapped: "statuses",
  stun_frozen: "statuses",
  stun_broken: "statuses",
  slow_snagged: "statuses",
  slow_hamstrung: "statuses",
  anti_heal: "statuses",
  freedom: "statuses",
  damage_boost: "statuses",
  spawn_shield: "statuses",
  in_combat: "statuses",
  rest: "statuses",
  aura_empowerment: "statuses",
  aura_barrier: "statuses",
  more_statuses: "statuses",
  score: "hud",
  tick_clock: "hud",
  fighters_standing: "hud",
  chronicle_ribbon: "hud",
});

/**
 * One example status row of a status picture: [host status ID, ticks left].
 * Pure.
 *
 * @param {string} statusId A host status ID, for example "hunter_basic_slow".
 * @param {number} ticks Ticks left, a whole number from 1 to 5.
 * @returns {readonly [string, number]}
 */
function statusRow(statusId, ticks) {
  return Object.freeze(/** @type {const} */ ([statusId, ticks]));
}

/**
 * Example status rows on the example champion of each status picture, as
 * [host status ID, ticks left] pairs. Blessing of Freedom (Slow Resistance)
 * shows its wings only when it matters, so its champion also has Charge
 * (Slow), with 2 ticks left (2 beads, so it reads apart from the slow
 * picture's 5). Freezing Trap (Stun Broken) has no status at the end of the
 * tick: its picture breaks the Freezing Trap instead (see
 * STATUS_BREAK_EXAMPLES).
 *
 * @type {Readonly<Record<string, ReadonlyArray<readonly [string, number]>>>}
 */
const STATUS_EXAMPLES = Object.freeze({
  stun_sapped: Object.freeze([statusRow("warrior_charge_stun", 5)]),
  stun_frozen: Object.freeze([statusRow("hunter_trap_stun", 5)]),
  stun_broken: Object.freeze([]),
  slow_snagged: Object.freeze([statusRow("hunter_basic_slow", 5)]),
  slow_hamstrung: Object.freeze([statusRow("warrior_charge_slow", 5)]),
  anti_heal: Object.freeze([statusRow("rogue_poison_anti_heal", 5)]),
  freedom: Object.freeze([
    statusRow("warrior_charge_slow", 2),
    statusRow("priest_blessing_of_freedom_movement_floor", 5),
  ]),
  damage_boost: Object.freeze([statusRow("mage_burst_damage_amplification", 5)]),
});

/**
 * The status a picture shows breaking this tick (a "status_broken_by_damage"
 * event on the example champion), by symbol ID.
 *
 * @type {Readonly<Record<string, string>>}
 */
const STATUS_BREAK_EXAMPLES = Object.freeze({ stun_broken: "hunter_trap_stun" });

/**
 * Return the Visual Key's entries for one client.
 *
 * input.product is "combat_debugger" (BattleClient and DevClient) or
 * "replay_viewer"; without a known product the result is [] (nothing to show).
 * input.catalogue (default none) supplies the numbers (see
 * VisualKeyCatalogue). Records, in SYMBOL_GROUPS order:
 * - {kind: "visual_key_entry", symbolId, group, title, names, eli5, key,
 *   numbers}: one per symbol; names lists every status name a mark stands for
 *   (both "Charge (Stun)" and "Crippling Poison (Stun)" under the stun
 *   stars; just the title for most symbols); numbers is a list of {label,
 *   value} texts read from the
 *   catalog fields the symbol's numberSources name (live per-champion values
 *   such as life now are not shown here). Battle Cam (B) and Fighters
 *   Standing appear only in the Replay Viewer (the live scorebug has no
 *   standing count; its portraits show who is down).
 * - {kind: "visual_key_command", group: "Camera And Keys", id, keycaps, help}:
 *   one per key the client uses, from commandHelpRows.
 * Pure data: no DOM.
 *
 * @param {{product?: unknown, catalogue?: VisualKeyCatalogue | null}} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const product = input.product;
  if (product !== "combat_debugger" && product !== "replay_viewer") {
    return Object.freeze([]);
  }
  const catalogue = isRecord(input.catalogue) ? input.catalogue : {};
  /** @type {LookRecord[]} */
  const out = [];
  for (const group of SYMBOL_GROUPS) {
    for (const entry of SYMBOLS) {
      if (entry.group !== group) {
        continue;
      }
      if (
        (entry.key === "B" || entry.id === "fighters_standing") &&
        product !== "replay_viewer"
      ) {
        continue;
      }
      out.push(
        Object.freeze({
          kind: "visual_key_entry",
          symbolId: entry.id,
          group,
          title: entry.title,
          names: entry.names,
          eli5: entry.eli5,
          key: entry.key,
          numbers: symbolNumbers(entry, catalogue),
        }),
      );
    }
  }
  for (const row of commandHelpRows(product)) {
    out.push(
      Object.freeze({
        kind: "visual_key_command",
        group: "Camera And Keys",
        id: row.id,
        keycaps: row.keycaps,
        help: row.help,
      }),
    );
  }
  return Object.freeze(out);
}

/**
 * Return the numbers of one symbol from the catalogue, as frozen
 * {label, value} rows in numberSources order. Only static catalog records
 * are read: class mechanics (by class_id), class status and aura mechanics (by
 * status_id or aura_id, searched in every class row), spawn shield mechanics,
 * the Red Zone depth and the respawn period. A source without select is shown
 * only when every class row agrees. Missing values are left out; repeated
 * label and value pairs are shown once. Booleans read "Yes" or "No".
 *
 * @param {SymbolEntry} entry
 * @param {VisualKeyCatalogue} catalogue
 * @returns {ReadonlyArray<Readonly<{label: string, value: string}>>}
 */
export function symbolNumbers(entry, catalogue) {
  const classRows = Array.isArray(catalogue.classMechanics)
    ? catalogue.classMechanics.filter(isRecord)
    : [];
  /** @type {Array<Readonly<{label: string, value: string}>>} */
  const rows = [];
  const seen = new Set();
  for (const source of entry.numberSources) {
    const value = catalogueValue(source, classRows, catalogue);
    if (value === undefined) {
      continue;
    }
    const text = formatValue(value);
    if (text === null) {
      continue;
    }
    const id = `${source.meaning}\u0000${text}`;
    if (seen.has(id)) {
      continue;
    }
    seen.add(id);
    rows.push(Object.freeze({ label: source.meaning, value: text }));
  }
  return Object.freeze(rows);
}

/**
 * Build the dictionary for records: one `section.visual-key__group` per group
 * that has records (heading plus cards), in record order. An entry becomes an
 * `article.visual-key__card` (data-symbol-id) with its picture
 * (drawSymbolPicture), title (a two-cause mark's names one per line, each a
 * `span.visual-key__name`), sentence, key cap and numbers. Command rows
 * become one `table.visual-key__keys` at the end of Camera And Keys. Text is
 * written with textContent only.
 *
 * doc creates the nodes; null doc or no records returns []. options.looks
 * replaces the look modules (for tests); default the real ones.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document | null} doc
 * @param {{looks?: Readonly<Record<string, any>>}} [options]
 * @returns {HTMLElement[]}
 */
export function build(records, doc, options = {}) {
  if (doc === null || records.length === 0) {
    return [];
  }
  /** @type {HTMLElement[]} */
  const sections = [];
  for (const group of SYMBOL_GROUPS) {
    const entries = records.filter(
      (record) => record.kind === "visual_key_entry" && record.group === group,
    );
    const commands = records.filter(
      (record) => record.kind === "visual_key_command" && record.group === group,
    );
    if (entries.length === 0 && commands.length === 0) {
      continue;
    }
    const section = doc.createElement("section");
    section.className = "visual-key__group";
    section.setAttribute("data-group", group);
    const heading = doc.createElement("h3");
    heading.className = "visual-key__group-title";
    heading.textContent = group;
    const grid = doc.createElement("div");
    grid.className = "visual-key__cards";
    for (const record of entries) {
      grid.append(card(record, doc, options.looks));
    }
    section.append(heading, grid);
    if (commands.length > 0) {
      section.append(keyTable(commands, doc));
    }
    sections.push(section);
  }
  return sections;
}

/**
 * Nothing to move: the Key is not drawn on the arena. Returns undefined.
 *
 * @param {ReadonlyArray<Element>} _nodes
 * @param {ReadonlyArray<LookRecord>} _records
 */
export function update(_nodes, _records) {}

/**
 * Return the Symbol Tip of a Key entry (its symbol's plain sentence), or null
 * for any other record.
 *
 * @param {LookRecord} record
 * @returns {import("../tooltip.js").TooltipDescriptor | null}
 */
export function tooltip(record) {
  if (record.kind !== "visual_key_entry" || symbolById(record.symbolId) === null) {
    return null;
  }
  return createSymbolDescriptor({ symbol: record.symbolId });
}

/**
 * Draw one symbol as a small standalone SVG picture.
 *
 * symbolId is a symbol ID from `symbols.js`. doc creates the nodes. looks
 * (default DEFAULT_LOOKS, the real look modules by file name) is searched in
 * this order:
 * 1. the symbol's owning look's visualKeyExample(symbolId, {doc, transform})
 *    when that look exports it and returns at least one node;
 * 2. the built-in example (see exampleInputs): the real look's records()
 *    and build() on fixed example data, never a recording: a champion of the
 *    right class, team, life, Ultimate charge, status or Spawn Shield; a Red
 *    Zone strip; a stone; a waystone; the sight veil; a small whole map; or a
 *    fixed two-champion example event (a Basic, an Ultimate, a life number, a
 *    Fall, a fallen champion and its Returns In pips, a Return, a Start Spot,
 *    a Last Seen ring, an aura's Ward Circle). An event picture draws its
 *    example champions first, under the event, with the Champions look;
 *    those layers are best effort and never decide the source;
 * 3. a drawn sign: every arena symbol has an example, so only the HUD Marks
 *    (drawn as HTML by the HUD, which cannot go in an SVG picture) and the
 *    camera keys get one: a key cap for V, B and F, otherwise the symbol's
 *    first letter in a ring.
 * A look that throws or draws nothing falls through to the next step. The
 * returned <svg> has class "visual-key__picture", a fixed viewBox of
 * VISUAL_KEY_PICTURE, role="img", aria-label = the symbol's title and
 * data-picture-source = "example:<look>", "look:<look>" or "sign".
 *
 * @param {string} symbolId
 * @param {Document} doc
 * @param {Readonly<Record<string, any>>} [looks]
 * @returns {SVGSVGElement}
 */
export function drawSymbolPicture(symbolId, doc, looks = DEFAULT_LOOKS) {
  const entry = symbolById(symbolId);
  const svg = /** @type {SVGSVGElement} */ (doc.createElementNS(SVG_NAMESPACE, "svg"));
  svg.setAttribute("class", "visual-key__picture");
  svg.setAttribute(
    "viewBox",
    `0 0 ${VISUAL_KEY_PICTURE.width} ${VISUAL_KEY_PICTURE.height}`,
  );
  svg.setAttribute("width", String(VISUAL_KEY_PICTURE.width));
  svg.setAttribute("height", String(VISUAL_KEY_PICTURE.height));
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", entry?.title ?? symbolId);
  const transform = pictureTransform();
  const owner = /** @type {Record<string, string>} */ (EXAMPLE_OWNERS)[symbolId];
  const ownerLook = owner === undefined ? undefined : looks[owner];
  if (typeof ownerLook?.visualKeyExample === "function") {
    const nodes = attempt(() =>
      ownerLook.visualKeyExample(symbolId, { doc, transform }),
    );
    if (nodes.length > 0) {
      svg.append(...nodes);
      svg.setAttribute("data-picture-source", `example:${owner}`);
      return svg;
    }
  }
  for (const { look: lookName, input, under } of exampleInputs(symbolId, transform)) {
    const nodes = drawLook(looks[lookName], input, doc);
    if (nodes.length > 0) {
      const beneath = under.flatMap((layer) =>
        drawLook(looks[layer.look], layer.input, doc),
      );
      svg.append(...beneath, ...nodes);
      if (symbolId === "in_combat") svg.append(inCombatPlate(doc));
      svg.setAttribute("data-picture-source", `look:${lookName}`);
      return svg;
    }
  }
  svg.append(...drawSign(entry, doc));
  svg.setAttribute("data-picture-source", "sign");
  return svg;
}

/**
 * @typedef {{
 *   button: HTMLElement,
 *   product: "combat_debugger" | "replay_viewer",
 *   getCatalogue?: () => VisualKeyCatalogue | null,
 *   doc?: Document,
 *   looks?: Readonly<Record<string, any>>,
 * }} VisualKeyOptions
 */

/**
 * Create the Visual Key for one page.
 *
 * options.button is the Visual Key button beside Help (#visual-key-button); it
 * is shown (always, whatever the switches), gets a small book picture,
 * aria-haspopup="dialog" and opens the Key on click. options.product picks the
 * client's keys. options.getCatalogue() (default none) returns the numbers'
 * catalog for the shown audience, read from its arena scene only.
 * options.doc defaults to the button's document; options.looks replaces the
 * look modules (tests).
 *
 * Side effects: on first open a `dialog.visual-key-dialog` is appended to the
 * document body and opened with showModal (or the open attribute when
 * showModal is missing). Its content is rebuilt when the numbers change. Close
 * and Escape close it (native dialog behaviour) and focus returns to the
 * button. destroy removes the listener and the dialog.
 *
 * @param {VisualKeyOptions} options
 * @returns {Readonly<{open: () => void, close: () => void, isOpen: () => boolean, destroy: () => void}>}
 * @throws {TypeError} When options or button is missing.
 * @throws {RangeError} When product is not a known product.
 */
export function createVisualKey(options) {
  if (!isRecord(options) || !isRecord(options.button)) {
    throw new TypeError("Visual Key needs options with a button.");
  }
  const { button, product } = options;
  if (product !== "combat_debugger" && product !== "replay_viewer") {
    throw new RangeError(`Unknown product ${String(product)}.`);
  }
  const doc = options.doc ?? button.ownerDocument;
  /** @type {HTMLDialogElement | null} */
  let dialog = null;
  /** @type {HTMLElement | null} */
  let body = null;
  let contentKey = "";

  button.hidden = false;
  button.setAttribute("aria-haspopup", "dialog");
  if (!button.querySelector?.("svg.visual-key-book")) {
    button.prepend?.(bookIcon(doc));
  }

  /** Create the dialog once, with its header, Close button and body. */
  function ensureDialog() {
    if (dialog !== null) {
      return;
    }
    dialog = /** @type {HTMLDialogElement} */ (doc.createElement("dialog"));
    dialog.className = "visual-key-dialog";
    dialog.setAttribute("aria-labelledby", "visual-key-dialog-title");
    const header = doc.createElement("header");
    header.className = "visual-key-dialog__header";
    const title = doc.createElement("h2");
    title.id = "visual-key-dialog-title";
    title.textContent = "Visual Key";
    const lead = doc.createElement("p");
    lead.className = "visual-key-dialog__lead";
    lead.textContent =
      "Every mark you can meet, drawn by the arena itself, with one plain sentence and its numbers.";
    const close = doc.createElement("button");
    close.type = "button";
    close.className = "button button--quiet visual-key-dialog__close";
    close.textContent = "Close";
    close.addEventListener("click", () => closeKey());
    header.append(title, lead, close);
    body = doc.createElement("div");
    body.className = "visual-key-dialog__body";
    dialog.append(header, body);
    dialog.addEventListener("close", () => {
      if (typeof button.focus === "function") {
        button.focus({ preventScroll: true });
      }
    });
    doc.body.append(dialog);
  }

  /** Rebuild the content when the client or the numbers changed. */
  function fill() {
    const catalogue = options.getCatalogue?.() ?? null;
    const keyRecords = records({ product, catalogue });
    const key = JSON.stringify(
      keyRecords.map((record) =>
        record.kind === "visual_key_entry" ? record.numbers : 0,
      ),
    );
    if (key === contentKey && body !== null && body.childElementCount > 0) {
      return;
    }
    contentKey = key;
    body?.replaceChildren(...build(keyRecords, doc, { looks: options.looks }));
  }

  /** Open the Key (build or refresh it first). Does nothing when open. */
  function open() {
    ensureDialog();
    const current = /** @type {HTMLDialogElement} */ (dialog);
    if (current.open) {
      return;
    }
    fill();
    if (typeof current.showModal === "function") {
      current.showModal();
    } else {
      current.setAttribute("open", "");
    }
  }

  /** Close the Key when it is open. */
  function closeKey() {
    if (dialog === null || !dialog.open) {
      return;
    }
    if (typeof dialog.close === "function") {
      dialog.close();
    } else {
      dialog.removeAttribute("open");
    }
  }

  /** Open from the button. */
  function onButtonClick() {
    open();
  }
  button.addEventListener("click", onButtonClick);

  return Object.freeze({
    open,
    close: closeKey,
    isOpen: () => dialog?.open === true,
    destroy() {
      closeKey();
      button.removeEventListener("click", onButtonClick);
      dialog?.remove();
      dialog = null;
      body = null;
      contentKey = "";
    },
  });
}

/**
 * The real look modules by file name. The built-in examples use basics,
 * champions, fog, health-text, movement, obstacles, red-zone, requiems,
 * spawns, statuses and ultimates; hud is asked only for its visualKeyExample
 * hook (its own marks are HTML).
 */
const DEFAULT_LOOKS = Object.freeze({
  basics,
  champions,
  fog,
  "health-text": healthText,
  hud,
  movement,
  obstacles,
  "red-zone": redZone,
  requiems,
  spawns,
  statuses,
  ultimates,
});

/**
 * Return the fixed projection of a picture: world fitted into
 * VISUAL_KEY_PICTURE with 6 px padding (Whole Map camera).
 *
 * @param {Readonly<{width: number, height: number}>} [world] The example
 *   world in world units: WORLD (the default), EVENT_WORLD or AURA_WORLD.
 * @returns {LookTransform}
 */
function pictureTransform(world = WORLD) {
  return /** @type {LookTransform} */ (
    /** @type {unknown} */ (
      createViewportTransform({
        worldWidth: world.width,
        worldHeight: world.height,
        viewportWidth: VISUAL_KEY_PICTURE.width,
        viewportHeight: VISUAL_KEY_PICTURE.height,
        padding: 6,
      })
    )
  );
}

/**
 * One built-in picture try: look names the real look module that must draw
 * (its records(input), then build(list, doc) with no mount, so it draws a
 * settled still with no animation); under lists layers drawn first, beneath
 * it, each best effort (a layer that throws or draws nothing is left out).
 *
 * @typedef {{
 *   look: string,
 *   input: Record<string, any>,
 *   under: ReadonlyArray<{look: string, input: Record<string, any>}>,
 * }} PictureTry
 */

/**
 * Return the built-in picture tries for symbolId, in order (at most one per
 * symbol today). Every input is fixed example data made here, never a
 * recording, a live match, the researcher view or a match summary, so a
 * picture shows nothing an Agent POV audience could be denied.
 *
 * transform is the WORLD picture projection (pictureTransform()); event and
 * aura pictures make their own wider projection (EVENT_WORLD, AURA_WORLD).
 * The tries:
 * - champions, classes, statuses, Spawn Shield, In Combat and Rest: one
 *   champion (Champions look) with the right class, team, life, Ultimate
 *   charge, status row or Spawn Shield.
 * - Basics, Ultimates, life numbers, Deaths And Returns, Start Spot, Last
 *   Seen and auras: an example event or part (see eventExample).
 * - Red Zone (both marks), stone, waystone, sight veil and Whole Map: the
 *   Red Zone, Obstacles, Spawns or Fog look on a small example floor; the
 *   Whole Map draws stones over both Red Zone strips.
 * Unknown symbols, HUD Marks and the camera keys V, B and F return []
 * (the caller then draws a sign).
 *
 * @param {string} symbolId A symbol ID from `symbols.js`.
 * @param {LookTransform} transform The WORLD picture projection.
 * @returns {PictureTry[]}
 */
function exampleInputs(symbolId, transform) {
  const center = [WORLD.width / 2, WORLD.height / 2];
  const event = eventExample(symbolId);
  if (event !== null) {
    return [event];
  }
  const classMatch = /^class_(mage|warrior|hunter|rogue|priest)$/u.exec(symbolId);
  /** @type {AgentSpec} */
  const agent = { classId: 1, teamId: 1 };
  if (classMatch !== null) {
    agent.classId = classIdByName(classMatch[1]);
  }
  if (symbolId === "team_b") agent.teamId = 2;
  if (symbolId === "life_bar") agent.life = 0.55;
  if (symbolId === "rune_ring") agent.cooldown = 0.4;
  if (symbolId === "spawn_shield") agent.shield = 2;
  const statusRows =
    /** @type {Record<string, ReadonlyArray<readonly [string, number]>>} */ (
      STATUS_EXAMPLES
    )[symbolId];
  if (statusRows !== undefined) agent.statuses = statusRows;
  if (symbolId === "in_combat") agent.combatTicks = 3;
  if (symbolId === "rest") agent.life = 0.7;
  if (statusRows !== undefined) {
    // Status pictures look closer (STATUS_WORLD) so each small mark reads.
    const zoom = pictureTransform(STATUS_WORLD);
    const scene = exampleScene([
      exampleAgent(agent, [STATUS_WORLD.width / 2, STATUS_WORLD.height / 2]),
    ]);
    const broken = /** @type {Record<string, string>} */ (STATUS_BREAK_EXAMPLES)[
      symbolId
    ];
    const key = scene.agents[0].presentation_key;
    return [
      {
        look: "statuses",
        input: {
          part: "status-marks",
          scene,
          transform: zoom,
          events:
            broken === undefined
              ? []
              : [
                  {
                    event_kind: "status_broken_by_damage",
                    status_id: broken,
                    recipient_anchor: { presentation_key: key },
                  },
                ],
          switches: { cooldownEffects: false },
        },
        under: [{ look: "champions", input: championsInput(scene, zoom, false) }],
      },
    ];
  }
  const redZoneStrips = {
    transform,
    map: { width: WORLD.width },
    redZone: {
      depth: 0.7,
      team_a_x_range: [0, 0.7],
      team_b_x_range: [WORLD.width - 0.7, WORLD.width],
    },
  };
  switch (symbolId) {
    case "red_zone":
    case "red_zone_x2":
      return [{ look: "red-zone", input: redZoneStrips, under: [] }];
    case "stone":
      return [
        {
          look: "obstacles",
          input: {
            transform,
            map: {
              obstacles: [{ kind: "pillar", id: 1, center, radius: 0.55 }],
            },
          },
          under: [],
        },
      ];
    case "camera_whole_map":
      return [
        {
          look: "obstacles",
          input: {
            transform,
            map: {
              obstacles: [
                { kind: "pillar", id: 1, center: [1.05, 1.15], radius: 0.22 },
                { kind: "pillar", id: 2, center: [1.55, 0.55], radius: 0.22 },
              ],
            },
          },
          under: [{ look: "red-zone", input: redZoneStrips }],
        },
      ];
    case "waystone":
      return [
        {
          look: "spawns",
          input: {
            transform,
            audience: "researcher",
            scene: exampleScene(
              [],
              [
                {
                  position: center,
                  team_id: 1,
                  team_local_slot: 0,
                  configured_active: true,
                  currently_alive: true,
                },
              ],
            ),
          },
          under: [],
        },
      ];
    case "sight_veil":
      return [
        {
          look: "fog",
          input: {
            transform,
            audience: "agent_pov",
            scene: exampleScene([exampleAgent({ classId: 3, teamId: 1 }, [0.9, 0.85])]),
          },
          under: [],
        },
      ];
    default:
      return symbolId.startsWith("class_") ||
        [
          "champion",
          "team_a",
          "team_b",
          "life_bar",
          "rune_ring",
          "spawn_shield",
          "in_combat",
          "rest",
        ].includes(symbolId)
        ? [
            {
              look: "champions",
              input: championsInput(
                exampleScene([exampleAgent(agent, center)]),
                transform,
                symbolId === "spawn_shield",
                symbolId === "rest" ? restRows(exampleAgent(agent, center)) : null,
              ),
              under: [],
            },
          ]
        : [];
  }
}

/**
 * Return the example event (or event-like part) picture of symbolId, or null
 * when symbolId is not an event symbol. All positions are fixed points of
 * EVENT_WORLD (AURA_WORLD for an aura) projected into the picture; the champions
 * involved are drawn under the event by the Champions look.
 *
 * - blow and basic_<class>: a Basics "activation" (component "basic") from a
 *   Team A champion of that class (blow: a Mage) to a Team B enemy, or to a
 *   Team A ally for the Priest's heal. Ranged classes stand about 0.9 units
 *   apart rim to rim (beyond the 0.55-unit close range, so they draw their
 *   stroke); the Warrior and Rogue stand 0.1 units apart.
 * - ult_<class>: an Ultimates "activation" (component "ultimate") with the
 *   class's token (ULTIMATE_EXAMPLES): Burst around a lone Mage, Charge from
 *   its start spot to a landing at the target's rim, Trap and Poison on an
 *   enemy, Salvation on an ally.
 * - damage_number and heal_number: a Health Text "net_health" event of -12 or
 *   +9 life on one champion, in Oracle View terms (plan audience "researcher").
 * - fall and respawn: a Requiems "semantic_pulse" (agent_died over a fallen
 *   Warrior; agent_respawned over a Spawn Shielded Warrior on its waystone).
 * - fallen and returns_in: the Requiems corpse marks of one fallen Warrior;
 *   fallen hides the Returns In pips, returns_in shows 3 ticks to the wave
 *   (4 pips).
 * - imprint: a Movement trajectory of one Hunter from a start spot 1.4 units
 *   to the left of where it ended.
 * - last_seen: a Fog "last_seen" plan row of a Rogue (the Rogue has no
 *   body at END, as in Agent POV).
 * - aura_empowerment and aura_barrier: the Statuses "aura-fields" part: one
 *   Ward Circle of the true 2.0-unit reach around its Mage or Warrior.
 *
 * @param {string} symbolId A symbol ID from `symbols.js`.
 * @returns {PictureTry | null}
 */
function eventExample(symbolId) {
  const aura = /** @type {Record<string, any>} */ (AURA_EXAMPLES)[symbolId];
  if (aura !== undefined) {
    return auraExample(aura);
  }
  const transform = pictureTransform(EVENT_WORLD);
  const middle = [EVENT_WORLD.width / 2, EVENT_WORLD.height / 2];
  const at = (/** @type {ReadonlyArray<number>} */ point) => {
    const screen = transform.worldToScreen([point[0], point[1]]);
    return { x: screen.x, y: screen.y };
  };
  const radius = transform.worldLengthToScreen(0.5);
  const scale = {
    sourceRadius: radius,
    targetRadius: radius,
    pixelsPerUnit: transform.worldLengthToScreen(1),
    unitPx: transform.worldLengthToScreen(1),
  };
  /** @param {Array<Record<string, any>>} agents @param {Array<Record<string, any>>} [pads] */
  const bodies = (agents, pads = []) => ({
    look: "champions",
    input: championsInput(exampleScene(agents, pads), transform, true),
  });
  const basic = /^basic_(mage|warrior|hunter|rogue|priest)$/u.exec(symbolId);
  if (basic !== null || symbolId === "blow") {
    const classId = classIdByName(basic === null ? "mage" : basic[1]);
    const melee = classId === 2 || classId === 4;
    const ally = classId === 5;
    const [from, to] = melee
      ? [
          [1.15, middle[1]],
          [2.25, middle[1]],
        ]
      : [
          [0.75, middle[1]],
          [2.65, middle[1]],
        ];
    const source = exampleAgent({ classId, teamId: 1, key: "visual-key-source" }, from);
    const target = exampleAgent(
      {
        classId: classId === 2 ? 1 : 2,
        teamId: ally ? 1 : 2,
        key: "visual-key-target",
      },
      to,
    );
    return {
      look: "basics",
      input: {
        transform,
        event: {
          ...scale,
          kind: "activation",
          component: "basic",
          sourceClassId: classId,
          source: at(from),
          target: at(to),
          sourcePresentationKey: source.presentation_key,
          targetPresentationKey: target.presentation_key,
          sourceEndpointPhase: "successor",
          targetEndpointPhase: "successor",
          outputAmount: 10,
        },
      },
      under: [bodies([source, target])],
    };
  }
  const ultimate = /^ult_(mage|warrior|hunter|rogue|priest)$/u.exec(symbolId);
  if (ultimate !== null) {
    const classId = classIdByName(ultimate[1]);
    const example = /** @type {Record<string, {tokenId: string, target: string}>} */ (
      ULTIMATE_EXAMPLES
    )[ultimate[1]];
    const charge = example.tokenId === "warrior_charge";
    const from = example.target === "none" ? middle : [charge ? 0.6 : 0.75, middle[1]];
    const to = [2.65, middle[1]];
    const landing = [to[0] - 1.02, middle[1]];
    const source = exampleAgent(
      { classId, teamId: 1, key: "visual-key-source" },
      charge ? landing : from,
    );
    const target = exampleAgent(
      {
        classId: classId === 2 ? 1 : 2,
        teamId: example.target === "ally" ? 1 : 2,
        key: "visual-key-target",
      },
      to,
    );
    const lone = example.target === "none";
    return {
      look: "ultimates",
      input: {
        transform,
        event: {
          ...scale,
          kind: "activation",
          component: "ultimate",
          tokenId: example.tokenId,
          source: at(from),
          target: lone ? null : at(to),
          sourcePresentationKey: source.presentation_key,
          targetPresentationKey: lone ? null : target.presentation_key,
          sourceEndpointPhase: "successor",
          targetEndpointPhase: lone ? null : "successor",
          ...(charge
            ? {
                chargeLanding: at(landing),
                chargeTravel: landing[0] - from[0],
                sourceEnd: at(landing),
                targetEnd: at(to),
              }
            : {}),
        },
      },
      under: [bodies(lone ? [source] : [source, target])],
    };
  }
  const warrior = (/** @type {Partial<AgentSpec>} */ extra = {}) =>
    exampleAgent({ classId: 2, teamId: 2, key: "visual-key-target", ...extra }, middle);
  switch (symbolId) {
    case "damage_number":
    case "heal_number": {
      const damage = symbolId === "damage_number";
      const body = warrior({ life: damage ? 0.48 : 0.6 });
      return {
        look: "health-text",
        input: {
          plan: { audience: "researcher", events: [], bodies: [] },
          event: {
            kind: "net_health",
            netDelta: damage ? -12 : 9,
            healthAfter: damage ? 48 : 60,
            recipientPresentationKey: body.presentation_key,
            recipient: at(middle),
            cue: { x: at(middle).x, y: at(middle).y - radius - 9 },
          },
        },
        under: [bodies([body])],
      };
    }
    case "fall":
    case "respawn": {
      const fall = symbolId === "fall";
      const body = fall ? warrior({ alive: false }) : warrior({ teamId: 1, shield: 2 });
      const pad = {
        position: middle,
        team_id: 1,
        team_local_slot: 0,
        configured_active: true,
        currently_alive: true,
      };
      return {
        look: "requiems",
        input: {
          event: {
            kind: "semantic_pulse",
            cueSemantic: fall ? "agent_died" : "agent_respawned",
            anchor: at(middle),
            anchorRadius: radius,
            recipientPresentationKey: body.presentation_key,
            recipientIdentity: { class_id: 2, team_id: fall ? 2 : 1 },
          },
        },
        under: fall
          ? [bodies([body])]
          : [
              {
                look: "spawns",
                input: {
                  transform,
                  audience: "researcher",
                  scene: exampleScene([], [pad]),
                },
              },
              bodies([body]),
            ],
      };
    }
    case "fallen":
    case "returns_in": {
      const scene = exampleScene([warrior({ alive: false })]);
      scene.respawn_waves = [{ team_id: 2, period_steps: 10, countdown_steps: 3 }];
      return {
        look: "requiems",
        input: {
          part: "corpse-marks",
          scene,
          transform,
          audience: "researcher",
          filters: { death_effects: true, vigil_pips: symbolId === "returns_in" },
        },
        under: [{ look: "champions", input: championsInput(scene, transform, true) }],
      };
    }
    case "imprint": {
      const end = [2.3, middle[1]];
      const body = exampleAgent(
        { classId: 3, teamId: 1, key: "visual-key-source" },
        end,
      );
      const scene = exampleScene([body]);
      return {
        look: "movement",
        input: {
          scene,
          transform,
          audience: "researcher",
          trajectories: [
            {
              agent_presentation_key: body.presentation_key,
              transition_start: { position: [end[0] - 1.4, end[1]] },
              successor: { position: [...end] },
            },
          ],
          returned: [],
        },
        under: [bodies([body])],
      };
    }
    case "last_seen":
      return {
        look: "fog",
        input: {
          transform,
          event: {
            kind: "last_seen",
            eventId: "visual-key-last-seen",
            presentationKey: "visual-key-unseen",
            publicAgentId: "agent-slot-7",
            classId: 4,
            teamId: null,
            anchor: at(middle),
          },
        },
        under: [],
      };
    default:
      return null;
  }
}

/**
 * Return the aura picture try: the Statuses look's "aura-fields" part with
 * one Ward Circle of the true reach (AURA_REACH, 2.0 units) around its
 * caster, a Team A champion of aura.classId, in AURA_WORLD. The tooltip input
 * is the field in Oracle View terms with no source agent, so no live champion
 * is named.
 *
 * @param {{classId: number, auraId: string, tokenId: string}} aura
 * @returns {PictureTry}
 */
function auraExample(aura) {
  const transform = pictureTransform(AURA_WORLD);
  const middle = [AURA_WORLD.width / 2, AURA_WORLD.height / 2];
  const caster = exampleAgent(
    { classId: aura.classId, teamId: 1, key: "visual-key-source" },
    middle,
  );
  const field = {
    presentation_key: "visual-key-aura",
    global_slot: 5,
    source_presentation_key: caster.presentation_key,
    source_class_id: aura.classId,
    aura_id: aura.auraId,
    token_id: aura.tokenId,
    center: [...middle],
    radius: AURA_REACH,
  };
  return {
    look: "statuses",
    input: {
      part: "aura-fields",
      transform,
      fields: [field],
      agents: [caster],
      tooltipInput: (/** @type {Record<string, any>} */ row) => ({
        field: row,
        sourceAgent: null,
        audience: "researcher",
      }),
    },
    under: [
      {
        look: "champions",
        input: championsInput(exampleScene([caster]), transform, false),
      },
    ],
  };
}

/**
 * Return the Champions look input for an example scene: default switches and,
 * without rows, no beat clock (the settled END look).
 *
 * @param {Record<string, any>} scene From exampleScene.
 * @param {LookTransform} transform The picture projection.
 * @param {boolean} showSpawnShield Whether a Spawn Shield is drawn.
 * @param {ReadonlyArray<Record<string, any>> | null} [rows] This tick's
 *   example incoming rows (beats.rows, for example restRows), or null (the
 *   default) for no beat cue.
 * @returns {Record<string, any>}
 */
function championsInput(scene, transform, showSpawnShield, rows = null) {
  return {
    scene,
    transform,
    showSpawnShield,
    visualFilters: DEFAULT_VISUAL_FILTER_STATE,
    beats: rows === null ? null : { rows, view: null },
  };
}

/**
 * Return the incoming rows of the Rest (Regeneration) picture: one
 * "health_regenerated" event of 4 life on the example champion, so the
 * Champions look pours its pale Rest light (the --moon-dew color) into its
 * life vessel. Pure.
 *
 * @param {Record<string, any>} agent The example champion (exampleAgent).
 * @returns {ReadonlyArray<Record<string, any>>}
 */
function restRows(agent) {
  return Object.freeze([
    {
      kind: "health_regenerated",
      vocabulary: "event",
      payload: {
        agent_anchor: { presentation_key: agent.presentation_key },
        actual_health_regenerated: 4,
      },
    },
  ]);
}

/**
 * Run one look on one example input with no mount (a settled still) and
 * return its SVG nodes (see attempt). A missing look, an error or nothing
 * drawn gives [].
 *
 * @param {Readonly<Record<string, any>> | undefined} look A look module.
 * @param {Record<string, any>} input Its records() input.
 * @param {Document} doc Creates the nodes.
 * @returns {Element[]}
 */
function drawLook(look, input, doc) {
  if (!look) {
    return [];
  }
  return attempt(() => {
    const list = look.records(input);
    const built = look.build(list, doc);
    look.update?.(built, list);
    return built;
  });
}

/**
 * Return the class ID (1 to 5) of a lower-case class name, from the one class
 * table; 1 when not found.
 *
 * @param {string} name
 * @returns {number}
 */
function classIdByName(name) {
  for (const identity of Object.values(CLASS_IDENTITIES)) {
    if (identity.label.toLowerCase() === name) {
      return identity.classId;
    }
  }
  return 1;
}

/**
 * What one example champion looks like.
 *
 * classId is 1 to 5 and teamId 1 (Team A) or 2 (Team B). Optional: life is
 * the share of full life (default 1; a fallen champion has 0); cooldown the
 * share of the Ultimate recharge left (default 0); shield the Spawn Shield
 * ticks left (default 0); statuses its [host status ID, ticks left] pairs
 * (default none); combatTicks its In Combat countdown (default 0, out of combat);
 * key its presentation key (default "visual-key-<classId>-<teamId>"; two
 * bodies in one picture need different keys); alive false makes it a fallen
 * champion (life state "corpse"; default true).
 *
 * @typedef {{
 *   classId: number,
 *   teamId: number,
 *   life?: number,
 *   cooldown?: number,
 *   shield?: number,
 *   statuses?: ReadonlyArray<readonly [string, number]>,
 *   combatTicks?: number,
 *   key?: string,
 *   alive?: boolean,
 * }} AgentSpec
 */

/**
 * Make one fixed example champion row (never a recorded one), with the same
 * fields an adapted arena scene row has (alive, max_health, ultimate_cooldown,
 * display_agent_id and a status token_id equal to its status_id, as the
 * adapter writes it, beside the host fields).
 *
 * @param {AgentSpec} spec
 * @param {ReadonlyArray<number>} position World units.
 * @returns {Record<string, any>}
 */
function exampleAgent(spec, position) {
  const maximum = 100;
  const key = spec.key ?? `visual-key-${spec.classId}-${spec.teamId}`;
  const alive = spec.alive !== false;
  return {
    presentation_key: key,
    display_key: key,
    public_agent_id: "agent-slot-5",
    display_agent_id: "5",
    global_slot: 5,
    class_id: spec.classId,
    team_id: spec.teamId,
    relation: "self",
    position: [...position],
    radius: 0.5,
    life_state: alive ? "alive" : "corpse",
    alive,
    current_health: alive ? maximum * (spec.life ?? 1) : 0,
    maximum_health: maximum,
    max_health: maximum,
    ultimate_cooldown_remaining: Math.round(30 * (spec.cooldown ?? 0)),
    ultimate_cooldown: Math.round(30 * (spec.cooldown ?? 0)),
    spawn_shield_remaining: spec.shield ?? 0,
    steps_until_out_of_combat: spec.combatTicks ?? 0,
    statuses: (spec.statuses ?? []).map(([statusId, ticks]) => ({
      status_id: statusId,
      token_id: statusId,
      remaining_duration: ticks,
      configured_duration_steps: 5,
      direct_sources: [],
    })),
  };
}

/**
 * Make a fixed example scene around agents and spawn pads.
 *
 * @param {Array<Record<string, any>>} agents
 * @param {Array<Record<string, any>>} [pads]
 * @returns {Record<string, any>}
 */
function exampleScene(agents, pads = []) {
  return {
    audience: "researcher",
    map: { width: WORLD.width, height: WORLD.height, obstacles: [] },
    agents,
    aura_fields: [],
    spawn_pads: pads,
    respawn_waves: [],
    class_mechanics: [],
    selection: {},
  };
}

/**
 * Run draw and return the SVG nodes it made: each result that is a DOM node is
 * kept, and a bundle object gives its first node-valued field among wrapper,
 * root and group. Errors and empty results give [].
 *
 * @param {() => unknown} draw
 * @returns {Element[]}
 */
function attempt(draw) {
  try {
    const result = draw();
    if (!Array.isArray(result)) {
      return [];
    }
    return result.flatMap((item) => {
      if (isNode(item)) {
        return [item];
      }
      if (isRecord(item)) {
        const node = [item.wrapper, item.root, item.group].find(isNode);
        return node === undefined ? [] : [node];
      }
      return [];
    });
  } catch {
    return [];
  }
}

/**
 * Return whether value looks like a DOM element.
 *
 * @param {unknown} value
 * @returns {value is Element}
 */
function isNode(value) {
  return (
    isRecord(value) && typeof value.setAttribute === "function" && "tagName" in value
  );
}

/**
 * Draw the sign that stands in when no look draws a symbol: a key cap with
 * the key for V, B and F, otherwise a ring with the first letter of the name.
 * Flat fills and strokes only.
 *
 * @param {SymbolEntry | null} entry
 * @param {Document} doc
 * @returns {SVGElement[]}
 */
function drawSign(entry, doc) {
  const cx = VISUAL_KEY_PICTURE.width / 2;
  const cy = VISUAL_KEY_PICTURE.height / 2;
  const group = svg(doc, "g", { class: "visual-key__sign" });
  if (entry?.key) {
    group.append(
      svg(doc, "rect", {
        x: cx - 16,
        y: cy - 16,
        width: 32,
        height: 32,
        rx: 6,
        fill: "#1A1E2B",
        stroke: "#C9B98F",
        "stroke-width": 1.5,
      }),
      svgText(doc, entry.key, cx, cy, 15),
    );
  } else {
    group.append(
      svg(doc, "circle", {
        cx,
        cy,
        r: 18,
        fill: "#1A1E2B",
        stroke: "#C9B98F",
        "stroke-width": 1.5,
      }),
      svgText(doc, (entry?.name ?? "?").slice(0, 1), cx, cy, 15),
    );
  }
  return [group];
}

/**
 * Draw the In Combat picture's plate beside the example champion: a small dark
 * plate with the In Combat glyph (icons.js, the glyph the Status Plates use)
 * and its countdown "3" (ticks until it can rest). Export-safe SVG. Pure apart
 * from creating nodes.
 *
 * @param {Document} doc Creates the nodes.
 * @returns {SVGElement}
 */
function inCombatPlate(doc) {
  const x = VISUAL_KEY_PICTURE.width / 2 + 22;
  const y = VISUAL_KEY_PICTURE.height / 2 - 9;
  const group = svg(doc, "g", { class: "visual-key__plate" });
  const glyph = createSvgIcon(doc, "combat-in-progress", { size: 14 });
  for (const [name, value] of [
    ["x", String(x + 2)],
    ["y", String(y + 2)],
    ["width", "14"],
    ["height", "14"],
    ["color", "#E6EAF2"],
  ]) {
    glyph.setAttribute(name, value);
  }
  group.append(
    svg(doc, "rect", {
      x,
      y,
      width: 28,
      height: 18,
      rx: 4,
      fill: "#1A1E2B",
      stroke: "#C9B98F",
      "stroke-width": 1,
    }),
    glyph,
    svgText(doc, "3", x + 22, y + 9, 10),
  );
  return group;
}

/**
 * Create one SVG element with attributes (numbers written as text).
 *
 * @param {Document} doc
 * @param {string} tag
 * @param {Record<string, string | number>} attributes
 * @returns {SVGElement}
 */
function svg(doc, tag, attributes) {
  const node = /** @type {SVGElement} */ (doc.createElementNS(SVG_NAMESPACE, tag));
  for (const [name, value] of Object.entries(attributes)) {
    node.setAttribute(name, String(value));
  }
  return node;
}

/**
 * Create centred ivory SVG text.
 *
 * @param {Document} doc
 * @param {string} text
 * @param {number} x
 * @param {number} y
 * @param {number} size Font size in px.
 * @returns {SVGElement}
 */
function svgText(doc, text, x, y, size) {
  const node = svg(doc, "text", {
    x,
    y,
    "text-anchor": "middle",
    "dominant-baseline": "central",
    fill: "#F4EBD6",
    "font-size": size,
    "font-weight": 700,
  });
  node.textContent = text;
  return node;
}

/**
 * Make the small book picture of the Visual Key button (16 x 16 px, current
 * color strokes, aria-hidden).
 *
 * @param {Document} doc
 * @returns {SVGElement}
 */
function bookIcon(doc) {
  const icon = svg(doc, "svg", {
    class: "visual-key-book",
    viewBox: "0 0 16 16",
    width: 16,
    height: 16,
    "aria-hidden": "true",
    fill: "none",
    stroke: "currentColor",
    "stroke-width": 1.4,
  });
  icon.append(
    svg(doc, "path", { d: "M2 3.2c2-.9 4-.9 6 .4v9.6c-2-1.3-4-1.3-6-.4z" }),
    svg(doc, "path", { d: "M14 3.2c-2-.9-4-.9-6 .4v9.6c2-1.3 4-1.3 6-.4z" }),
  );
  return icon;
}

/**
 * Build one card for a Key entry: its picture, its title (or, for a mark
 * that serves two causes, each of record.names on its own line), its plain
 * sentence, its key cap and its numbers. Text is written with textContent
 * only.
 *
 * @param {LookRecord} record
 * @param {Document} doc
 * @param {Readonly<Record<string, any>> | undefined} looks
 * @returns {HTMLElement}
 */
function card(record, doc, looks) {
  const article = doc.createElement("article");
  article.className = "visual-key__card";
  article.setAttribute("data-symbol-id", record.symbolId);
  const text = doc.createElement("div");
  text.className = "visual-key__text";
  const title = doc.createElement("h4");
  title.className = "visual-key__title";
  if (Array.isArray(record.names) && record.names.length > 1) {
    // A mark that serves two causes lists both names, one per line.
    for (const name of record.names) {
      const line = doc.createElement("span");
      line.className = "visual-key__name";
      line.textContent = name;
      title.append(line);
    }
  } else {
    title.textContent = record.title;
  }
  if (record.key) {
    const key = doc.createElement("kbd");
    key.className = "visual-key__keycap";
    key.textContent = record.key;
    title.append(" ", key);
  }
  const sentence = doc.createElement("p");
  sentence.className = "visual-key__sentence";
  sentence.textContent = record.eli5;
  text.append(title, sentence);
  if (record.numbers.length > 0) {
    const list = doc.createElement("dl");
    list.className = "visual-key__numbers";
    for (const number of record.numbers) {
      const label = doc.createElement("dt");
      label.textContent = number.label;
      const value = doc.createElement("dd");
      value.textContent = number.value;
      list.append(label, value);
    }
    text.append(list);
  }
  article.append(drawSymbolPicture(record.symbolId, doc, looks), text);
  return article;
}

/**
 * Build the keys table of Camera And Keys.
 *
 * @param {ReadonlyArray<LookRecord>} commands
 * @param {Document} doc
 * @returns {HTMLElement}
 */
function keyTable(commands, doc) {
  const table = doc.createElement("table");
  table.className = "visual-key__keys";
  const caption = doc.createElement("caption");
  caption.textContent = "Keys";
  const tbody = doc.createElement("tbody");
  for (const command of commands) {
    const row = doc.createElement("tr");
    row.setAttribute("data-command-id", command.id);
    const keys = doc.createElement("th");
    keys.scope = "row";
    const cap = doc.createElement("kbd");
    cap.textContent = command.keycaps;
    keys.append(cap);
    const help = doc.createElement("td");
    help.textContent = command.help;
    row.append(keys, help);
    tbody.append(row);
  }
  table.append(caption, tbody);
  return table;
}

/**
 * Read one number source from the static catalogue, or undefined.
 *
 * @param {import("../symbols.js").SymbolNumberSource} source
 * @param {Array<Record<string, any>>} classRows
 * @param {VisualKeyCatalogue} catalogue
 * @returns {unknown}
 */
function catalogueValue(source, classRows, catalogue) {
  const select = source.select ?? {};
  switch (source.record) {
    case "AuthorizedClassMechanicsV2": {
      if (select.class_id !== undefined) {
        return classRows.find((row) => row.class_id === select.class_id)?.[
          source.field
        ];
      }
      const values = classRows.map((row) => row[source.field]);
      return values.length > 0 && values.every((value) => value === values[0])
        ? values[0]
        : undefined;
    }
    case "AuthorizedClassStatusMechanicV1":
      return nestedRow(classRows, "status_mechanics", "status_id", select.status_id)?.[
        source.field
      ];
    case "AuthorizedClassAuraMechanicV1":
      return nestedRow(classRows, "aura_mechanics", "aura_id", select.aura_id)?.[
        source.field
      ];
    case "AuthorizedSpawnShieldMechanicsAvailableV2":
      return isRecord(catalogue.spawnShieldMechanics)
        ? catalogue.spawnShieldMechanics[source.field]
        : undefined;
    case "AuthorizedRedZoneV1":
      return source.field === "depth" && isRecord(catalogue.redZone)
        ? catalogue.redZone.depth
        : undefined;
    case "AuthorizedRespawnWaveV1":
      return source.field === "period_steps" && Array.isArray(catalogue.respawnWaves)
        ? catalogue.respawnWaves.find(isRecord)?.period_steps
        : undefined;
    default:
      return undefined;
  }
}

/**
 * Find the first nested row (status or aura mechanics) with id in any class row.
 *
 * @param {Array<Record<string, any>>} classRows
 * @param {string} listField
 * @param {string} idField
 * @param {string | undefined} id
 * @returns {Record<string, any> | undefined}
 */
function nestedRow(classRows, listField, idField, id) {
  if (id === undefined) {
    return undefined;
  }
  for (const row of classRows) {
    const list = Array.isArray(row[listField]) ? row[listField] : [];
    const match = list.find(
      (/** @type {unknown} */ item) => isRecord(item) && item[idField] === id,
    );
    if (match !== undefined) {
      return match;
    }
  }
  return undefined;
}

/**
 * Format a catalog value: numbers with up to two decimals, booleans as Yes or
 * No; anything else gives null.
 *
 * @param {unknown} value
 * @returns {string | null}
 */
function formatValue(value) {
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return formatDisplayNumber(value);
  }
  return null;
}
