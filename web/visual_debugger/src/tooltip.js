/**
 * @file Own semantic help records, tooltip registration, rendering and placement.
 * Use one delegated controller per tooltip surface. Registered owners hold
 * frozen descriptions in weak maps; prose is rendered as text, never HTML.
 * This module validates display structure, not scientific authorization.
 * Callers supply facts permitted for their audience and destroy controllers
 * when their view is removed.
 *
 * This file is the one tooltip owner for every client (BattleClient, DevClient
 * combat screen and Replay Viewer). It also owns the Symbol Tip: the "symbol"
 * descriptor kind (createSymbolDescriptor reads names and plain sentences from
 * `symbols.js`), registerSymbolOwner (focusable owners with a stable
 * data-symbol-key), symbolHitRect (an 18 px minimum hit box for SVG marks) and
 * heldSymbolValue (the value a tip shows during the dramatic hold, from
 * `beatView`). The controller shows the card in the browser's top layer
 * (popover="manual") so it stays visible over modal dialogs, pins a card on
 * click or tap (a second tap or a tap elsewhere closes it), closes it on Escape,
 * keeps keyboard focus on the same symbol across redraws and turns symbol tips
 * off when the symbol_explanations switch is off (setSymbolTipsEnabled). It
 * opens no card over the live Command Deck (#command-deck,
 * insideLiveCommandDeck): the approved vision gives the deck's controls no
 * hover card.
 *
 * It also owns the Champion Card, the one card every client shows for a
 * hovered champion (design C "Clean List", owner, 2 October 2026): the name
 * and calling, the Agent ID with its team crest, the shared Life bar
 * (`life-bar.js`: "77 / 80"; green above half, yellow at half or less, red at
 * 30% or less), a thin bar named after the Ultimate ("Burst", filling as it
 * recharges, "30 ticks" or "Ready"), then one two-line row per status (its
 * icon, its approved name, its time left, what it does with the exact number
 * and where it comes from by class, and a line that drains as the ticks run
 * out; stuns first, then slows, then the rest; at most four, then "+N more"
 * with "Hold Shift For All"), and who plays it and "Your Target". A champion
 * the viewing team cannot see gets the reduced card (name, calling, crest, ID
 * and "Unseen"); a fallen one shows life 0 and "Fallen · Returns In N" from
 * its team's public wave clock. championCardFactsFromAgent reads a scene row into facts,
 * championCardModel turns facts into words and shares, renderChampionCard
 * draws it, and championCardText writes it as text.
 *
 * The plain card never leaves the arena box (owner, 2 October 2026):
 * placeChampionCard places it beside the pointer, fully inside the arena box.
 * The plain card is always one column ("Fewer rows", owner, 2 October 2026):
 * when the box is too short for all its rows, renderFittedChampionCard draws
 * it with as many rows as fit (at least one; championCardRowsThatFit decides
 * from measured heights), then "+N more · Hold Shift For All". Only the Hold
 * Shift card spreads its rows into two (or, in a small window, three or four)
 * columns when it would be taller than the box; only when even that is too
 * tall is a card scaled down to fit. When the Hold Shift card would need a
 * scale below 0.85 inside the box (CHAMPION_CARD_MIN_INSIDE_SCALE), it grows
 * past the box instead ("Grow past arena", owner, 2 October 2026): normal
 * size, the column count that makes it shortest, over the panels and fully
 * inside the window (championCardLayout decides, championCardPastArenaPosition
 * places it).
 * Hold Shift (createShiftHold): Shift held by itself for about 300 ms
 * (CHAMPION_CARD_HOLD_MS) while the pointer rests on the champion shows every
 * status in place ("All 7 statuses", "Release Shift To Shrink"); releasing
 * Shift, any other key, any click and any pointer movement shrink it at once,
 * so every Shift combination keeps its own job, and the gesture never sends
 * anything. Live play (BattleClient and DevClient combat screen) shows the
 * card in #hcard: createHoverCard binds it to the page (shell.js builds the
 * facts and installs it with #arena as the box, and main.js calls its
 * refresh after every installed frame, so a resting pointer never keeps an
 * old frame's card). The Replay Viewer shows the
 * same card through this module's tooltip controller:
 * createChampionCardDescriptor makes a "champion" descriptor that the
 * controller draws with renderChampionCard and places inside the box its
 * championCardBox option names.
 *
 * Agent POV: this module never reads a frame. Each caller builds its facts from
 * the audience's own scene and events, so a tip over the arena can only say what
 * the caller was allowed to draw. Status sources are named by class only, so
 * Agent POV and Oracle View give the same card for a seen champion.
 */
import { classIdentityFromId } from "./agent-identity.js";
import { classAbilityNameV1, classNamesV1 } from "./class-documentation.js";
import { createSvgIcon, modifierSigilKey, statusSigilKey } from "./icons.js";
import { createLifeBar, lifeBarModel } from "./life-bar.js";
import { statusCardSentence, statusTimeText } from "./semantic-vocabulary.js";
import { statusNameFor, statusOrderIndex, symbolById } from "./symbols.js";
import { statusTokenIdFromCatalogId } from "./vocabulary.js";

const TOOLTIP_OWNER_ATTRIBUTE = "data-tooltip-owner";
const TOOLTIP_KIND_ATTRIBUTE = "data-tooltip-kind";
const TOOLTIP_TONE_ATTRIBUTE = "data-tooltip-tone";
const TOOLTIP_ACCENT_ATTRIBUTE = "data-tooltip-accent";
const SYMBOL_KEY_ATTRIBUTE = "data-symbol-key";
const SYMBOL_ID_ATTRIBUTE = "data-symbol";
const VIEWPORT_GUTTER = 8;
const TOOLTIP_GAP = 12;
// An SVG smaller than this on either side (CSS pixels) is an icon, such as a
// scorebug portrait's 28 px picture, not a drawing surface like the arena: a
// card for an owner inside it is placed in the viewport instead, so the icon
// never squeezes the card to its own width (ownerSurfaceRectangle).
const MIN_INFERRED_SURFACE = 160;

/** The descriptor kind of every Symbol Tip (see createSymbolDescriptor). */
export const SYMBOL_TIP_KIND = "symbol";

/**
 * The element ID of the live Command Deck (the BattleClient's and the DevClient
 * combat screen's deck: steps, targets, actions and Lock In). The controller
 * never opens a card for an owner inside it, because the approved vision page
 * gives the deck's controls no hover card: they show their state with their
 * own tags ("Not Allowed", "Armed") and refusals go to the status line. The
 * owners keep their aria-description, so a screen reader still reads the
 * sentence. The Replay Viewer has no deck.
 */
export const LIVE_COMMAND_DECK_ID = "command-deck";

/**
 * Return whether element is the live Command Deck (#command-deck) or sits
 * inside it. It walks `parentElement`, so it also works for DOM stand-ins
 * that have `id` and `parentElement`. Pure.
 *
 * @param {Element | null | undefined} element Any element, or nothing.
 * @returns {boolean} True inside the deck; false for null, undefined or any
 *   element outside it.
 */
export function insideLiveCommandDeck(element) {
  for (let node = element ?? null; node !== null; node = node.parentElement ?? null) {
    if (node.id === LIVE_COMMAND_DECK_ID) return true;
  }
  return false;
}

/**
 * The smallest hit box of a symbol, in CSS pixels (width and height). HTML
 * owners get it from `styles/symbols-ui.css`; SVG marks add symbolHitRect.
 */
export const SYMBOL_HIT_BOX_PX = 18;

/** Accent of each class ID (1 Mage to 5 Priest) for symbol descriptors. */
const CLASS_ACCENTS = Object.freeze({
  1: "mage",
  2: "warrior",
  3: "hunter",
  4: "rogue",
  5: "priest",
});

const descriptorByOwner = new WeakMap();
const surfaceByOwner = new WeakMap();
const inspectableByOwner = new WeakMap();
const controllerByTooltip = new WeakMap();
const semanticDescriptors = new WeakSet();

export const SEMANTIC_TONES = Object.freeze([
  "neutral",
  "information",
  "positive",
  "warning",
  "danger",
]);
const SEMANTIC_TONE_SET = new Set(SEMANTIC_TONES);
export const SEMANTIC_ACCENTS = Object.freeze([
  "none",
  "mage",
  "warrior",
  "hunter",
  "rogue",
  "priest",
]);
const SEMANTIC_ACCENT_SET = new Set(SEMANTIC_ACCENTS);

// Lower wins when several owners are under the pointer; between equal ranks
// the one painted on top wins (chooseTooltipCandidate). A Champion Card
// ("champion") ranks with the Symbol Tips, so the owner on top wins: a
// champion's body beats the Spawn Pad drawn under it, and a status mark drawn
// on the body, or a HUD chrome tip ("hud": a life bar or status chip inside a
// Replay Viewer frame), beats the card of the body or frame it sits on, so
// the small mark under the pointer explains itself.
const KIND_PRIORITY = Object.freeze({
  hud: 0,
  symbol: 0,
  status: 0,
  modifier: 0,
  overflow: 0,
  "status-overflow": 0,
  "modifier-overflow": 0,
  legality: 0,
  cooldown: 0,
  visibility: 0,
  control: 0,
  champion: 0,
  agent: 1,
  activation: 2,
  impact: 2,
  event: 2,
  "accepted-route": 2,
  "combat-route": 2,
  "pending-route": 3,
  obstacle: 4,
  range: 5,
  "range-ultimate": 5,
  "range-basic": 5,
  "range-observation": 5,
  aura: 6,
  map: 7,
  composite: 8,
});

/** @typedef {"element" | "pointer"} TooltipAnchor */
/** @typedef {"compact" | "full"} SemanticSurface */
/** @typedef {{compact: boolean, full: boolean}} SemanticMetadata */
/**
 * @typedef {{
 *   label: string,
 *   value: string,
 *   metadata: Readonly<SemanticMetadata>,
 * }} SemanticRow
 */
/**
 * @typedef {{
 *   title: string,
 *   summary: string | null,
 *   rows: ReadonlyArray<SemanticRow>,
 *   metadata: Readonly<SemanticMetadata>,
 * }} SemanticSection
 */
/**
 * @typedef {{
 *   kind: string,
 *   id: string,
 *   title: string,
 *   tone: "neutral" | "information" | "positive" | "warning" | "danger",
 *   accent: "none" | "mage" | "warrior" | "hunter" | "rogue" | "priest",
 *   summary: string,
 *   rows: ReadonlyArray<SemanticRow>,
 *   sections: ReadonlyArray<SemanticSection>,
 *   metadata: Readonly<SemanticMetadata>,
 *   anchor: TooltipAnchor,
 * }} TooltipDescriptor
 */
/**
 * @typedef {{
 *   kind: string,
 *   id: string,
 *   title: string,
 *   tone: "neutral" | "information" | "positive" | "warning" | "danger",
 *   accent: "none" | "mage" | "warrior" | "hunter" | "rogue" | "priest",
 *   summary: null,
 *   rows: ReadonlyArray<SemanticRow>,
 *   sections: ReadonlyArray<SemanticSection>,
 *   metadata: Readonly<SemanticMetadata>,
 *   anchor: TooltipAnchor,
 * }} RowOnlyTooltipDescriptor
 */
/** @typedef {TooltipDescriptor | RowOnlyTooltipDescriptor} AnyTooltipDescriptor */
/**
 * @typedef {{
 *   descriptor: AnyTooltipDescriptor,
 *   paintOrder?: number,
 *   element?: Element,
 *   trigger?: Element,
 * }} TooltipCandidate
 */
/** @typedef {{x: number, y: number}} TooltipPoint */
/**
 * @typedef {{
 *   left: number,
 *   top: number,
 *   right: number,
 *   bottom: number,
 * }} TooltipRectangle
 */
/** @typedef {{width: number, height: number}} TooltipSize */
/** @typedef {"right-below" | "left-below" | "right-above" | "left-above"} TooltipPlacementName */
/**
 * @typedef {{
 *   anchorRect: TooltipRectangle,
 *   pointer: TooltipPoint,
 *   tooltipSize: TooltipSize,
 *   viewport: TooltipRectangle,
 *   protectedRects?: ReadonlyArray<TooltipRectangle>,
 * }} TooltipPlacementInput
 */
/**
 * @typedef {{
 *   left: number,
 *   top: number,
 *   placement: TooltipPlacementName,
 * }} TooltipPlacement
 */
/**
 * @typedef {{
 *   root: Document | ShadowRoot | HTMLElement,
 *   tooltip: HTMLElement,
 *   title: HTMLElement,
 *   details: HTMLElement,
 *   onInspect?: (descriptor: AnyTooltipDescriptor, context: Readonly<{owner: Element, trigger: Element | null}>) => void,
 *   championCardBox?: () => Element | null,
 * }} TooltipControllerOptions
 */
/**
 * @typedef {{
 *   refresh: () => void,
 *   hide: () => void,
 *   inspect: (target?: unknown) => boolean,
 *   destroy: () => void,
 *   setSymbolTipsEnabled: (enabled: boolean) => void,
 *   pinnedOwner: () => Element | null,
 * }} TooltipController
 */
/**
 * One current value shown in a Symbol Tip, for example
 * {label: "Life", value: "57 of 80"}.
 *
 * @typedef {{label: string, value: string}} SymbolFact
 */

/**
 * Construct and recursively freeze the only semantic explanation shape that
 * leaves the browser fact layer. Payload prose remains text; it cannot select
 * a tag name, class name, data attribute, or tone outside the fixed allowlist.
 *
 * @overload
 * @param {{summary: null} & Record<string, unknown>} rawDescriptor
 * @returns {RowOnlyTooltipDescriptor}
 */
/**
 * @overload
 * @param {unknown} rawDescriptor
 * @returns {TooltipDescriptor}
 */
/**
 * Validate and freeze a semantic help record, or reuse an existing branded
 * record unchanged. Require exactly kind, id, title, tone, accent, summary,
 * rows, sections, metadata and anchor. Text fields are trimmed; unknown tone/
 * accent become neutral/none, and anchor must be element or pointer. A null
 * summary requires at least one row visible on compact or full surfaces.
 * Normalize/freeze nested rows, sections and visibility metadata. Invalid
 * structure throws TypeError. This is a display contract: accessors may be
 * read, and the brand does not certify scientific information rights.
 *
 * @param {unknown} rawDescriptor
 * @returns {AnyTooltipDescriptor}
 */
export function createSemanticDescriptor(rawDescriptor) {
  if (!isRecord(rawDescriptor)) {
    throw new TypeError("semantic descriptor must be an object.");
  }
  if (semanticDescriptors.has(rawDescriptor)) {
    return /** @type {AnyTooltipDescriptor} */ (rawDescriptor);
  }
  assertExactKeys(
    rawDescriptor,
    [
      "kind",
      "id",
      "title",
      "tone",
      "accent",
      "summary",
      "rows",
      "sections",
      "metadata",
      "anchor",
    ],
    "semantic descriptor",
  );
  const kind = nonEmptyString(rawDescriptor.kind, "descriptor.kind");
  const id = nonEmptyString(rawDescriptor.id, "descriptor.id");
  const title = nonEmptyString(rawDescriptor.title, "descriptor.title");
  const summary =
    rawDescriptor.summary === null
      ? null
      : nonEmptyString(rawDescriptor.summary, "descriptor.summary");
  const tone =
    typeof rawDescriptor.tone === "string" && SEMANTIC_TONE_SET.has(rawDescriptor.tone)
      ? /** @type {TooltipDescriptor["tone"]} */ (rawDescriptor.tone)
      : "neutral";
  const accent =
    typeof rawDescriptor.accent === "string" &&
    SEMANTIC_ACCENT_SET.has(rawDescriptor.accent)
      ? /** @type {TooltipDescriptor["accent"]} */ (rawDescriptor.accent)
      : "none";
  const anchor = rawDescriptor.anchor;
  if (anchor !== "element" && anchor !== "pointer") {
    throw new TypeError('descriptor.anchor must be "element" or "pointer".');
  }
  const metadata = normalizeMetadata(rawDescriptor.metadata, "descriptor.metadata");
  const rows = Object.freeze(
    normalizeArray(rawDescriptor.rows, "descriptor.rows").map((row, index) =>
      normalizeSemanticRow(row, `descriptor.rows[${index}]`),
    ),
  );
  const sections = Object.freeze(
    normalizeArray(rawDescriptor.sections, "descriptor.sections").map(
      (section, index) =>
        normalizeSemanticSection(section, `descriptor.sections[${index}]`),
    ),
  );
  if (
    summary === null &&
    !["compact", "full"].some((surface) =>
      hasVisibleSemanticRow(rows, sections, /** @type {SemanticSurface} */ (surface)),
    )
  ) {
    throw new TypeError(
      "descriptor.summary may be null only when at least one semantic row is visible.",
    );
  }
  const descriptor = Object.freeze({
    kind,
    id,
    title,
    tone,
    accent,
    summary,
    rows,
    sections,
    metadata,
    anchor,
  });
  semanticDescriptors.add(descriptor);
  return descriptor;
}

/**
 * Return a frozen compact/full projection with rows and sections filtered
 * by their visibility metadata, preserving order. Normalize rawDescriptor
 * first; an existing branded descriptor is reused. Reject another surface
 * or a null-summary result with no visible row using TypeError. Share
 * already frozen row values and leave the original descriptor unchanged.
 *
 * @param {unknown} rawDescriptor
 * @param {SemanticSurface} surface
 */
export function projectSemanticDescriptor(rawDescriptor, surface) {
  const descriptor = normalizeDescriptor(rawDescriptor);
  if (surface !== "compact" && surface !== "full") {
    throw new TypeError('semantic surface must be "compact" or "full".');
  }
  const rows = Object.freeze(descriptor.rows.filter((row) => row.metadata[surface]));
  const sections = Object.freeze(
    descriptor.sections
      .filter((section) => section.metadata[surface])
      .map((section) =>
        Object.freeze({
          ...section,
          rows: Object.freeze(section.rows.filter((row) => row.metadata[surface])),
        }),
      ),
  );
  if (descriptor.summary === null && !hasVisibleSemanticRow(rows, sections, surface)) {
    throw new TypeError(
      `descriptor.summary may be null only when the ${surface} projection contains at least one semantic row.`,
    );
  }
  return Object.freeze({
    kind: descriptor.kind,
    id: descriptor.id,
    title: descriptor.title,
    tone: descriptor.tone,
    accent: descriptor.accent,
    summary: descriptor.summary,
    rows,
    sections,
    metadata: descriptor.metadata,
    anchor: descriptor.anchor,
    surface,
  });
}

/**
 * Render options.descriptor into title and details and return its
 * projection. surface defaults to compact and also accepts full. Require
 * compatible text elements or throw TypeError. Replace details with fixed
 * DOM structure and textContent only; minimal DOM stand-ins receive flattened
 * text instead. This changes supplied nodes but not the descriptor, and
 * performs no visibility, focus or tooltip-registration work.
 *
 * @param {{
 *   descriptor: unknown,
 *   title: HTMLElement,
 *   details: HTMLElement,
 *   surface?: SemanticSurface,
 * }} options
 */
export function renderSemanticDescriptor(options) {
  if (!isRecord(options)) {
    throw new TypeError("semantic render options must be an object.");
  }
  const surface = options.surface ?? "compact";
  const projection = projectSemanticDescriptor(options.descriptor, surface);
  const title = options.title;
  const details = options.details;
  if (!isTextSurface(title) || !isTextSurface(details)) {
    throw new TypeError("semantic title and details must be text elements.");
  }
  title.textContent = projection.title;
  const ownerDocument = details.ownerDocument;
  if (
    !ownerDocument ||
    typeof ownerDocument.createElement !== "function" ||
    typeof details.replaceChildren !== "function"
  ) {
    details.textContent = semanticDescriptorText(projection).join("\n");
    return projection;
  }

  const nodes = [];
  if (projection.summary !== null) {
    const summary = ownerDocument.createElement("p");
    summary.className = "semantic-explanation__summary";
    summary.textContent = projection.summary;
    nodes.push(summary);
  }
  if (projection.rows.length > 0) {
    nodes.push(renderRows(ownerDocument, projection.rows));
  }
  for (const section of projection.sections) {
    const sectionNode = ownerDocument.createElement("section");
    sectionNode.className = "semantic-explanation__section";
    const heading = ownerDocument.createElement("h3");
    heading.className = "semantic-explanation__heading";
    heading.textContent = section.title;
    sectionNode.append(heading);
    if (section.summary !== null) {
      const sectionSummary = ownerDocument.createElement("p");
      sectionSummary.className = "semantic-explanation__section-summary";
      sectionSummary.textContent = section.summary;
      sectionNode.append(sectionSummary);
    }
    if (section.rows.length > 0) {
      sectionNode.append(renderRows(ownerDocument, section.rows));
    }
    nodes.push(sectionNode);
  }
  details.replaceChildren(...nodes);
  return projection;
}

/**
 * Flatten an existing projection into ordered text strings for accessible
 * help or diagnostics. Start with summary, or title when summary is null;
 * then emit rows and sections in order. Return a new array without mutation
 * or another validation pass. It does not include title when summary exists.
 *
 * @param {ReturnType<typeof projectSemanticDescriptor>} projection
 * @returns {string[]}
 */
export function semanticDescriptorText(projection) {
  const text = projection.summary === null ? [projection.title] : [projection.summary];
  for (const row of projection.rows) {
    text.push(`${row.label}: ${row.value}`);
  }
  for (const section of projection.sections) {
    text.push(section.title);
    if (section.summary !== null) {
      text.push(section.summary);
    }
    for (const row of section.rows) {
      text.push(`${row.label}: ${row.value}`);
    }
  }
  return text;
}

/**
 * Associate element with a normalized descriptor in the shared weak map.
 * options defaults to {}; surface may override placement bounds and
 * inspectable=false disables full-card inspection (default true). Add the
 * owner attribute and accessible description for interactive owners. No
 * per-owner event listener is installed. Invalid inputs throw TypeError;
 * validation is not transactional, so call with valid options before showing
 * the view. Re-registering replaces its descriptor and placement settings.
 *
 * @param {Element} element
 * @param {unknown} descriptor
 * @param {{surface?: Element, inspectable?: boolean}} [options]
 * @returns {void}
 */
export function registerTooltipOwner(element, descriptor, options = {}) {
  if (!isTooltipOwner(element)) {
    throw new TypeError("tooltip owner must be an element.");
  }
  if (!isRecord(options)) {
    throw new TypeError("tooltip owner options must be an object.");
  }
  const normalized = normalizeDescriptor(descriptor);
  descriptorByOwner.set(element, normalized);
  const tagName =
    typeof element.tagName === "string" ? element.tagName.toLowerCase() : "";
  if (
    normalized.kind === "control" ||
    normalized.kind === "composite" ||
    ["a", "button", "input", "select", "textarea", "summary"].includes(tagName) ||
    (element.getAttribute("tabindex") !== null &&
      element.getAttribute("tabindex") !== "-1")
  ) {
    const ariaDescription =
      normalized.summary === null
        ? semanticDescriptorText(projectSemanticDescriptor(normalized, "compact")).join(
            " · ",
          )
        : normalized.summary;
    element.setAttribute("aria-description", ariaDescription);
  }
  if (options.surface !== undefined) {
    if (!isTooltipOwner(options.surface)) {
      throw new TypeError("tooltip owner surface must be an element.");
    }
    surfaceByOwner.set(element, options.surface);
  } else {
    surfaceByOwner.delete(element);
  }
  inspectableByOwner.set(element, options.inspectable !== false);
  element.setAttribute(TOOLTIP_OWNER_ATTRIBUTE, "");
}

/**
 * Build the frozen descriptor of a Symbol Tip from the one symbol list.
 *
 * The title is the symbol's title from `symbols.js` ("Serrated Arrows
 * (Slow)", "Champion"); the subtitle rule lives there. options.title, when
 * given, replaces it with a plainer name for this one mark: the stones use
 * the store's names "Pillar", "Wall" and "Thin Wall" instead of the symbol's
 * "Stone". The summary is
 * options.line when given (the one-line naming rule, for example "Serrated
 * Arrows (Slow) · x0.85 speed · 1 tick left · from Hunter ID 7's Serrated
 * Arrows"), otherwise the symbol's
 * plain sentence. Rows are options.facts in order; when a line is given the
 * plain sentence follows as a "What It Means" row; options.note adds a "Note"
 * row last (for example "Your team cannot see this champion now"). The accent
 * is the symbol's class color, or none. The anchor is the element.
 *
 * Values must come from the current frame of the caller's audience (Agent POV:
 * the POV scene and visual events only; use heldSymbolValue for life and score
 * during the dramatic hold). This function copies text only; it never reads a
 * frame and certifies no information right. Every word reaches the player, so
 * callers pass plain words only: never an internal ID such as
 * "base-sensor-obstacle-3".
 *
 * @param {{
 *   symbol: string | Readonly<{id: string}>,
 *   id?: string,
 *   title?: string | null,
 *   line?: string | null,
 *   facts?: ReadonlyArray<SymbolFact>,
 *   note?: string | null,
 * }} options symbol is a symbol ID or alias from `symbols.js` (or its entry);
 *   id is the descriptor ID (default "symbol:<symbol id>"); title is the card's
 *   title (default the symbol's own title); line, facts and note are optional
 *   (default none). Empty or blank strings count as absent.
 * @returns {TooltipDescriptor} A branded frozen descriptor of kind "symbol".
 * @throws {RangeError} When symbol names no known symbol.
 * @throws {TypeError} When options is not an object or a fact has an empty
 *   label or value.
 */
export function createSymbolDescriptor(options) {
  if (!isRecord(options)) {
    throw new TypeError("symbol descriptor options must be an object.");
  }
  const requested =
    isRecord(options.symbol) && typeof options.symbol.id === "string"
      ? options.symbol.id
      : options.symbol;
  const entry = symbolById(requested);
  if (entry === null) {
    throw new RangeError(`Unknown symbol ${String(requested)}.`);
  }
  const line =
    typeof options.line === "string" && options.line.trim()
      ? options.line.trim()
      : null;
  const note =
    typeof options.note === "string" && options.note.trim()
      ? options.note.trim()
      : null;
  const title =
    typeof options.title === "string" && options.title.trim()
      ? options.title.trim()
      : entry.title;
  const metadata = { compact: true, full: true };
  const rows = [
    ...normalizeArray(options.facts ?? [], "symbol facts").map((fact, index) => {
      if (!isRecord(fact)) {
        throw new TypeError(`symbol facts[${index}] must be an object.`);
      }
      return {
        label: nonEmptyString(fact.label, `symbol facts[${index}].label`),
        value: nonEmptyString(fact.value, `symbol facts[${index}].value`),
        metadata,
      };
    }),
    ...(line === null ? [] : [{ label: "What It Means", value: entry.eli5, metadata }]),
    ...(note === null ? [] : [{ label: "Note", value: note, metadata }]),
  ];
  const accent =
    entry.classId === null
      ? "none"
      : CLASS_ACCENTS[/** @type {1 | 2 | 3 | 4 | 5} */ (entry.classId)];
  return /** @type {TooltipDescriptor} */ (
    createSemanticDescriptor({
      kind: SYMBOL_TIP_KIND,
      id:
        typeof options.id === "string" && options.id.trim()
          ? options.id
          : `symbol:${entry.id}`,
      title,
      tone: "information",
      accent,
      summary: line ?? entry.eli5,
      rows,
      sections: [],
      metadata,
      anchor: "element",
    })
  );
}

/**
 * Register element as a symbol owner: the Symbol Tip shows on hover, tap and
 * keyboard focus, and a click or tap pins it.
 *
 * element is the HTML or SVG node that shows the symbol. descriptor should come
 * from createSymbolDescriptor (any valid descriptor is accepted). key is a
 * stable text that names the same symbol after a redraw, for example
 * "status:slow_snagged:pov_ab12"; the controller moves keyboard focus and a
 * pinned card to the new node with the same key. options.surface and
 * options.inspectable work as in registerTooltipOwner.
 *
 * Side effects on element: data-symbol-key = key, data-symbol = the symbol ID
 * (from descriptor.id after "symbol:", when present) and tabindex="0" when it
 * has no tabindex (so focus can reach it); then registerTooltipOwner adds the
 * owner mark and aria-description. Keep each key unique on the page, or focus
 * may move to the first node with that key.
 *
 * @param {Element} element
 * @param {unknown} descriptor
 * @param {string} key
 * @param {{surface?: Element, inspectable?: boolean}} [options] Default {}.
 * @returns {void}
 * @throws {TypeError} When element is not an element, key is empty or the
 *   descriptor or options are invalid (see registerTooltipOwner).
 */
export function registerSymbolOwner(element, descriptor, key, options = {}) {
  if (!isTooltipOwner(element)) {
    throw new TypeError("symbol owner must be an element.");
  }
  const stableKey = nonEmptyString(key, "symbol key");
  const normalized = normalizeDescriptor(descriptor);
  element.setAttribute(SYMBOL_KEY_ATTRIBUTE, stableKey);
  if (normalized.id.startsWith("symbol:")) {
    element.setAttribute(SYMBOL_ID_ATTRIBUTE, normalized.id.slice("symbol:".length));
  }
  if (element.getAttribute("tabindex") === null) {
    element.setAttribute("tabindex", "0");
  }
  registerTooltipOwner(element, normalized, options);
}

/**
 * Create an invisible SVG hit box at least SYMBOL_HIT_BOX_PX wide and tall,
 * centered on a small mark, so a pointer or finger can reach it.
 *
 * doc creates the node. centerX and centerY are the mark's center and width and
 * height its drawn size, all in CSS pixels of the SVG it goes into; each side
 * becomes max(size, 18). The rect is export-safe: a flat black fill at
 * fill-opacity 0 with pointer-events="all" and class "symbol-hit-area". The
 * caller appends it inside the symbol's owner node (before the mark).
 *
 * @param {Document} doc
 * @param {number} centerX
 * @param {number} centerY
 * @param {number} [width] Drawn width in px; default 0.
 * @param {number} [height] Drawn height in px; default 0.
 * @returns {SVGRectElement}
 * @throws {TypeError} When a number is not finite.
 * @throws {RangeError} When width or height is negative.
 */
export function symbolHitRect(doc, centerX, centerY, width = 0, height = 0) {
  const x = finiteNumber(centerX, "hit box centerX");
  const y = finiteNumber(centerY, "hit box centerY");
  const boxWidth = Math.max(
    nonNegativeFinite(width, "hit box width"),
    SYMBOL_HIT_BOX_PX,
  );
  const boxHeight = Math.max(
    nonNegativeFinite(height, "hit box height"),
    SYMBOL_HIT_BOX_PX,
  );
  const rect = /** @type {SVGRectElement} */ (
    doc.createElementNS("http://www.w3.org/2000/svg", "rect")
  );
  rect.setAttribute("class", "symbol-hit-area");
  rect.setAttribute("x", String(x - boxWidth / 2));
  rect.setAttribute("y", String(y - boxHeight / 2));
  rect.setAttribute("width", String(boxWidth));
  rect.setAttribute("height", String(boxHeight));
  rect.setAttribute("fill", "#000");
  rect.setAttribute("fill-opacity", "0");
  rect.setAttribute("pointer-events", "all");
  return rect;
}

/**
 * Return the number a Symbol Tip shows now, so a tip never tells a fall or a
 * point before the dramatic hold plays it.
 *
 * symbolId picks the rule: "life_bar" uses the beat entry of presentationKey
 * (its shown life while the Fall beat is held), "score" uses beat.score.shown
 * (Team A then Team B); any other symbol, or a missing entry, returns value
 * unchanged. beat is the result of `beatView(frame, stage)` in
 * `choreography-plan.js` for the same frame and audience, or null. value is the
 * caller's END value from the audience's scene (a number or a [Team A, Team B]
 * pair). Pure: no DOM.
 *
 * Agent POV: beatView lists only falls the team saw, so an unseen champion's
 * life is never changed or revealed here.
 *
 * @template T
 * @param {{
 *   symbolId: string,
 *   beat: Readonly<{
 *     life?: ReadonlyArray<Readonly<{presentationKey: string, shown: unknown}>>,
 *     score?: Readonly<{shown: unknown}>,
 *   }> | null,
 *   presentationKey?: string | null,
 *   value: T,
 * }} input
 * @returns {T | number | ReadonlyArray<number>}
 */
export function heldSymbolValue(input) {
  const beat = input.beat;
  if (!isRecord(beat)) {
    return input.value;
  }
  if (input.symbolId === "life_bar" && Array.isArray(beat.life)) {
    const entry = beat.life.find(
      (row) => isRecord(row) && row.presentationKey === input.presentationKey,
    );
    return entry !== undefined && typeof entry.shown === "number"
      ? entry.shown
      : input.value;
  }
  if (
    input.symbolId === "score" &&
    isRecord(beat.score) &&
    Array.isArray(beat.score.shown)
  ) {
    return /** @type {ReadonlyArray<number>} */ (beat.score.shown);
  }
  return input.value;
}

/**
 * Return the best candidate reference, or null for an empty array. Rank
 * by semantic kind, smaller paintOrder, descriptor ID and input index.
 * Omitted paintOrder uses that index. Invalid candidates/kind/ID throw
 * TypeError; negative paintOrder throws RangeError. Input is not sorted or
 * mutated. This checks ranking fields only, not the complete descriptor.
 *
 * @template {TooltipCandidate} Candidate
 * @param {ReadonlyArray<Candidate>} candidates
 * @returns {Candidate | null}
 */
export function chooseTooltipCandidate(candidates) {
  if (!Array.isArray(candidates)) {
    throw new TypeError("tooltip candidates must be an array.");
  }

  /** @type {Candidate | null} */
  let best = null;
  /** @type {[number, number, string, number] | null} */
  let bestRank = null;

  for (const [index, candidate] of candidates.entries()) {
    const uncheckedCandidate = /** @type {unknown} */ (candidate);
    if (!isRecord(uncheckedCandidate) || !isRecord(uncheckedCandidate.descriptor)) {
      throw new TypeError("each tooltip candidate must contain a descriptor.");
    }
    const kind = nonEmptyString(candidate.descriptor.kind, "descriptor.kind");
    const id = nonEmptyString(candidate.descriptor.id, "descriptor.id");
    const paintOrder =
      candidate.paintOrder === undefined
        ? index
        : nonNegativeFinite(candidate.paintOrder, "candidate.paintOrder");
    /** @type {[number, number, string, number]} */
    const rank = [kindPriority(kind), paintOrder, id, index];

    if (bestRank === null || compareRanks(rank, bestRank) < 0) {
      best = candidate;
      bestRank = rank;
    }
  }

  return best;
}

/**
 * Choose fixed-position coordinates in CSS pixels for one measured card.
 * input supplies anchorRect, pointer, tooltipSize and viewport; optional
 * protectedRects defaults to [] (non-arrays also become []). Validate finite
 * ordered rectangles and nonnegative sizes; a viewport must have positive
 * area. Type/range failures throw TypeError/RangeError.
 *
 * Try four corners with an 8px viewport gutter and 12px gap. Minimize
 * protected-area overlap first, then owner overlap, overflow, pointer
 * distance and clamping movement. Ties favor right-below. Return frozen
 * left/top/placement; oversized cards can still overflow or overlap when
 * no clear placement exists. No DOM reads or writes occur here.
 *
 * @param {TooltipPlacementInput} input
 * @returns {TooltipPlacement}
 */
export function placeTooltip(input) {
  if (!isRecord(input)) {
    throw new TypeError("tooltip placement input must be an object.");
  }
  const anchorRect = normalizeRectangle(input.anchorRect, "anchorRect");
  const pointer = normalizePoint(input.pointer, "pointer");
  const tooltipSize = normalizeSize(input.tooltipSize, "tooltipSize");
  const viewport = normalizeRectangle(input.viewport, "viewport");
  const protectedRects = Array.isArray(input.protectedRects)
    ? input.protectedRects.map((rectangle, index) =>
        normalizeRectangle(rectangle, `protectedRects[${index}]`),
      )
    : [];
  if (viewport.right <= viewport.left || viewport.bottom <= viewport.top) {
    throw new RangeError("viewport must have positive width and height.");
  }

  const horizontalGutter = Math.min(
    VIEWPORT_GUTTER,
    (viewport.right - viewport.left) / 2,
  );
  const verticalGutter = Math.min(
    VIEWPORT_GUTTER,
    (viewport.bottom - viewport.top) / 2,
  );
  const usableViewport = Object.freeze({
    left: viewport.left + horizontalGutter,
    top: viewport.top + verticalGutter,
    right: viewport.right - horizontalGutter,
    bottom: viewport.bottom - verticalGutter,
  });
  const placementEnvelope = protectedRects.reduce(
    (envelope, protectedRect) => ({
      left: Math.min(envelope.left, protectedRect.left),
      top: Math.min(envelope.top, protectedRect.top),
      right: Math.max(envelope.right, protectedRect.right),
      bottom: Math.max(envelope.bottom, protectedRect.bottom),
    }),
    anchorRect,
  );
  const right = Math.max(placementEnvelope.right, pointer.x) + TOOLTIP_GAP;
  const left =
    Math.min(placementEnvelope.left, pointer.x) - TOOLTIP_GAP - tooltipSize.width;
  const below = Math.max(placementEnvelope.bottom, pointer.y) + TOOLTIP_GAP;
  const above =
    Math.min(placementEnvelope.top, pointer.y) - TOOLTIP_GAP - tooltipSize.height;

  /** @type {ReadonlyArray<{placement: TooltipPlacementName, left: number, top: number}>} */
  const candidates = Object.freeze([
    Object.freeze({ placement: "right-below", left: right, top: below }),
    Object.freeze({ placement: "left-below", left, top: below }),
    Object.freeze({ placement: "right-above", left: right, top: above }),
    Object.freeze({ placement: "left-above", left, top: above }),
  ]);

  /** @type {TooltipPlacement | null} */
  let best = null;
  /** @type {number[] | null} */
  let bestScore = null;

  for (const [index, candidate] of candidates.entries()) {
    const rawBounds = rectangleFromPosition(candidate.left, candidate.top, tooltipSize);
    const overflow = rectangleOverflow(rawBounds, usableViewport);
    const clamped = clampPosition(
      candidate.left,
      candidate.top,
      tooltipSize,
      usableViewport,
    );
    const clampedBounds = rectangleFromPosition(clamped.left, clamped.top, tooltipSize);
    const ownerClearance = TOOLTIP_GAP / 2;
    const paddedAnchor = {
      left: anchorRect.left - ownerClearance,
      top: anchorRect.top - ownerClearance,
      right: anchorRect.right + ownerClearance,
      bottom: anchorRect.bottom + ownerClearance,
    };
    const inspectedOverlap = rectangleOverlapArea(clampedBounds, paddedAnchor);
    const protectedOverlap = protectedRects.reduce(
      (total, protectedRect) =>
        total +
        rectangleOverlapArea(clampedBounds, {
          left: protectedRect.left - ownerClearance,
          top: protectedRect.top - ownerClearance,
          right: protectedRect.right + ownerClearance,
          bottom: protectedRect.bottom + ownerClearance,
        }),
      0,
    );
    const pointerDistance = pointToRectangleDistance(pointer, clampedBounds);
    const displacement =
      Math.abs(clamped.left - candidate.left) + Math.abs(clamped.top - candidate.top);
    const score = [
      protectedOverlap,
      inspectedOverlap,
      overflow,
      pointerDistance,
      displacement,
      index,
    ];

    if (bestScore === null || compareNumericRanks(score, bestScore) < 0) {
      bestScore = score;
      best = Object.freeze({
        left: clamped.left,
        top: clamped.top,
        placement: candidate.placement,
      });
    }
  }

  if (best === null) {
    throw new Error("tooltip placement produced no candidates.");
  }
  return best;
}

/**
 * Set tooltip's maximum CSS width/height from the owning viewport and
 * measure it. If neither free axis fits, constrain the axis retaining the
 * larger card area, then measure again. Inputs are validated CSS-pixel
 * rectangles/point; return the resulting DOMRect. This mutates tooltip
 * styles and causes layout reads, but does not choose its final position.
 *
 * @param {{
 *   tooltip: HTMLElement,
 *   anchorRect: TooltipRectangle,
 *   pointer: TooltipPoint,
 *   viewport: TooltipRectangle,
 *   protectedRects: ReadonlyArray<TooltipRectangle>,
 * }} input
 * @returns {DOMRect}
 */
function measureTooltipForPlacement(input) {
  const { tooltip, anchorRect, pointer, viewport, protectedRects } = input;
  const viewportWidth = viewport.right - viewport.left;
  const viewportHeight = viewport.bottom - viewport.top;
  const horizontalGutter = Math.min(VIEWPORT_GUTTER, viewportWidth / 2);
  const verticalGutter = Math.min(VIEWPORT_GUTTER, viewportHeight / 2);
  const maxWidth = Math.max(viewportWidth - 2 * horizontalGutter, 0);
  const maxHeight = Math.max(viewportHeight - 2 * verticalGutter, 0);
  tooltip.style.maxWidth = `${maxWidth}px`;
  tooltip.style.maxHeight = `${maxHeight}px`;

  let measured = tooltip.getBoundingClientRect();
  const placementEnvelope = protectedRects.reduce(
    (envelope, protectedRect) => ({
      left: Math.min(envelope.left, protectedRect.left),
      top: Math.min(envelope.top, protectedRect.top),
      right: Math.max(envelope.right, protectedRect.right),
      bottom: Math.max(envelope.bottom, protectedRect.bottom),
    }),
    anchorRect,
  );
  const usableViewport = {
    left: viewport.left + horizontalGutter,
    top: viewport.top + verticalGutter,
    right: viewport.right - horizontalGutter,
    bottom: viewport.bottom - verticalGutter,
  };
  const horizontalClearance = Math.min(
    maxWidth,
    Math.max(
      Math.min(placementEnvelope.left, pointer.x) - TOOLTIP_GAP - usableViewport.left,
      usableViewport.right - Math.max(placementEnvelope.right, pointer.x) - TOOLTIP_GAP,
      0,
    ),
  );
  const verticalClearance = Math.min(
    maxHeight,
    Math.max(
      Math.min(placementEnvelope.top, pointer.y) - TOOLTIP_GAP - usableViewport.top,
      usableViewport.bottom -
        Math.max(placementEnvelope.bottom, pointer.y) -
        TOOLTIP_GAP,
      0,
    ),
  );

  if (measured.width > horizontalClearance && measured.height > verticalClearance) {
    const widthLimitedArea = horizontalClearance * measured.height;
    const heightLimitedArea = measured.width * verticalClearance;
    if (heightLimitedArea >= widthLimitedArea) {
      tooltip.style.maxHeight = `${verticalClearance}px`;
    } else {
      tooltip.style.maxWidth = `${horizontalClearance}px`;
    }
    measured = tooltip.getBoundingClientRect();
  }

  return measured;
}

/**
 * Create the sole delegated controller for options.tooltip. Supply root,
 * tooltip with a nonempty ID, title and details; optional onInspect receives
 * a descriptor plus owner/trigger context. Validate compatible DOM surfaces
 * and reject a second active controller with Error. Other invalid options
 * throw TypeError. Install pointer/focus/click/keyboard/scroll/resize handlers
 * and a removal observer when available, then hide the tooltip.
 *
 * Return frozen refresh, hide, inspect, destroy, setSymbolTipsEnabled and
 * pinnedOwner methods. hide clears the current card and any pin; later
 * interaction/refresh may show it again. inspect returns whether onInspect was
 * called. destroy removes listeners, observer and controller ownership and is
 * safe to repeat. Owner registrations remain weakly held. The caller owns
 * persistent inspector rendering and disposal.
 *
 * Symbol Tip behavior (owner rules, 1 October 2026):
 * - Top layer: when the browser has the Popover API, the card gets
 *   popover="manual" and is shown with showPopover, so it is drawn over a
 *   modal dialog. When a modal dialog is open, each show puts the card back on
 *   top. Without the API the card works as before.
 * - Pin: a click or tap on a "symbol" owner pins its card (it stays while the
 *   pointer moves on); a tap (touch or pen) pins any owner except native
 *   controls. A second click or tap on the pinned owner closes it, and a click
 *   or tap anywhere else unpins it. Enter on a focused non-native symbol owner
 *   acts as a click. Escape closes the card and the pin.
 * - Redraws: a pinned owner or focused symbol owner that a redraw removes is
 *   replaced by the new node with the same data-symbol-key (registerSymbolOwner);
 *   focus moves there with preventScroll when focus had fallen to the page. With
 *   no such node the card hides.
 * - setSymbolTipsEnabled(false) (the symbol_explanations switch) stops every
 *   "symbol" card from showing on hover, tap or focus and drops a symbol pin;
 *   other tooltip kinds keep working. Default: enabled.
 * - Live Command Deck: no card of any kind opens for an owner inside
 *   #command-deck (LIVE_COMMAND_DECK_ID), on hover, focus, click or refresh,
 *   and inspect refuses it; the vision gives the deck no hover card. The live
 *   hover card (#hcard, createHoverCard) is a separate surface and keeps
 *   working on the deck's Allowed Now buttons.
 * - Champion Card: a descriptor made by createChampionCardDescriptor (kind
 *   "champion") is drawn as the Champion Card (renderChampionCard into
 *   details, with the title element left empty), so the Replay Viewer's card
 *   is the live hover card's card. Its text stays in the descriptor's summary
 *   for screen readers and text-only surfaces. When
 *   options.championCardBox returns an element with an area (the Replay
 *   Viewer's arena box), the plain card is drawn one column wide with as
 *   many status rows as fit that box (renderFittedChampionCard, at least
 *   one, then "+N more · Hold Shift For All") and placed fully inside it by
 *   placeChampionCard, the live hover card's rules, whether the owner is an
 *   arena body or a docked frame outside the box; otherwise it shows up to
 *   four rows and is placed like any other card. The Hold Shift gesture
 *   (createShiftHold) works as on the live hover card: a card shown for a
 *   resting pointer that leaves rows out shows every status (in columns
 *   when needed) while Shift is held by itself, and shrinks back at once to
 *   the fitted plain card on release, any other key, any click or any
 *   pointer movement. When that card would need a scale below 0.85 inside
 *   the box, it grows past the box at its normal size, fully inside the
 *   window and off its owner when it can be (placeChampionCard; the card
 *   then has data-tooltip-placement "past-arena", else "arena-box"). It
 *   acts only while the card is actually drawn (a stylesheet may hide it).
 *
 * @param {TooltipControllerOptions} options root, tooltip, title and details
 *   as above; onInspect (default none) is called by inspect;
 *   championCardBox (default none) returns the arena box Champion Cards stay
 *   inside.
 * @returns {TooltipController}
 */
export function createTooltipController(options) {
  if (!isRecord(options)) {
    throw new TypeError("tooltip controller options must be an object.");
  }
  const { root, tooltip, title, details } = options;
  const onInspect = options.onInspect;
  if (!isEventRoot(root)) {
    throw new TypeError("tooltip root must support delegated events.");
  }
  if (!isTooltipSurface(tooltip)) {
    throw new TypeError("tooltip surface must be an HTML element.");
  }
  if (!isTextSurface(title) || !isTextSurface(details)) {
    throw new TypeError("tooltip title and details must be text elements.");
  }
  if (onInspect !== undefined && typeof onInspect !== "function") {
    throw new TypeError("tooltip onInspect must be a function when provided.");
  }
  const tooltipId = nonEmptyString(tooltip.id, "tooltip.id");
  if (controllerByTooltip.has(tooltip)) {
    throw new Error("tooltip surface already has an active controller.");
  }

  let destroyed = false;
  /** @type {Element | null} */
  let activeOwner = null;
  /** @type {Element | null} */
  let focusedOwner = null;
  /** @type {Element | null} */
  let focusedTrigger = null;
  /** @type {TooltipPoint | null} */
  let lastPointer = null;
  /** @type {Element | null} */
  let describedByTrigger = null;
  let describedByAdded = false;
  let dismissedUntilInteraction = false;
  /** @type {Element | null} */
  let pinned = null;
  let symbolTipsEnabled = true;
  /** The data-symbol-key of the last focused symbol owner, kept across a redraw. */
  /** @type {string | null} */
  let focusKey = null;
  /** @type {Element | null} */
  let focusKeyElement = null;
  /** Whether the Champion Card shows every status (the Hold Shift gesture). */
  let championCardExpanded = false;
  /**
   * How many status rows the last plain Champion Card drawn shows (fewer
   * than its statuses when the four-row limit or a short arena box left rows
   * out); the Hold Shift gesture starts only when rows were left out.
   */
  let championShownRows = 0;
  /** @type {ReturnType<typeof createShiftHold> | null} */
  let championHold = null;

  tooltip.style.pointerEvents = "none";
  const supportsTopLayer =
    "showPopover" in tooltip &&
    typeof tooltip.showPopover === "function" &&
    typeof tooltip.hidePopover === "function";
  if (supportsTopLayer) {
    tooltip.setAttribute("popover", "manual");
  }

  /**
   * Return whether the card is open in the top layer. False without the
   * Popover API or when the browser cannot answer.
   *
   * @returns {boolean}
   */
  function topLayerOpen() {
    try {
      return supportsTopLayer && tooltip.matches(":popover-open");
    } catch {
      return false;
    }
  }

  /**
   * Put the card in the browser's top layer (Popover API), so it shows over
   * modal dialogs. When it is already open and a modal dialog is open, close
   * and reopen it so it sits above that dialog. Does nothing without the API;
   * browser errors (for example a card outside the document) are ignored.
   */
  function openTopLayer() {
    if (!supportsTopLayer) {
      return;
    }
    try {
      if (topLayerOpen()) {
        const modalOpen =
          typeof tooltip.ownerDocument?.querySelector === "function" &&
          tooltip.ownerDocument.querySelector("dialog:modal") !== null;
        if (!modalOpen) {
          return;
        }
        tooltip.hidePopover();
      }
      tooltip.showPopover();
    } catch {
      // The card still works without the top layer.
    }
  }

  /** Take the card out of the top layer when it is there. Errors are ignored. */
  function closeTopLayer() {
    if (!topLayerOpen()) {
      return;
    }
    try {
      tooltip.hidePopover();
    } catch {
      // Already closed.
    }
  }

  /**
   * Return whether the card of owner must not open: a symbol card while the
   * symbol_explanations switch is off, or any card of an owner inside the live
   * Command Deck (see LIVE_COMMAND_DECK_ID). Pure.
   *
   * @param {AnyTooltipDescriptor} descriptor The owner's registered descriptor.
   * @param {Element} owner The registered owner element.
   * @returns {boolean}
   */
  function suppressed(descriptor, owner) {
    return (
      (!symbolTipsEnabled && descriptor.kind === SYMBOL_TIP_KIND) ||
      insideLiveCommandDeck(owner)
    );
  }

  /**
   * Find the node that replaced old after a redraw: a registered, connected
   * owner inside root with the same data-symbol-key. Return null when old has
   * no key, root cannot be searched or no such node exists.
   *
   * @param {Element | null} old
   * @param {string | null} [key] The key to look for; default old's key.
   * @returns {Element | null}
   */
  function replacementOwner(
    old,
    key = old?.getAttribute(SYMBOL_KEY_ATTRIBUTE) ?? null,
  ) {
    if (key === null || !("querySelectorAll" in root)) {
      return null;
    }
    for (const element of root.querySelectorAll(`[${SYMBOL_KEY_ATTRIBUTE}]`)) {
      if (
        element !== old &&
        element.getAttribute(SYMBOL_KEY_ATTRIBUTE) === key &&
        descriptorByOwner.has(element) &&
        ownerWithinRoot(root, element)
      ) {
        return element;
      }
    }
    return null;
  }

  /**
   * Attach this tooltip ID to trigger's aria-describedby token list.
   * First remove only the token previously added by this controller. null
   * clears the current link; an unchanged trigger is a no-op. Preserve other
   * ARIA tokens and remember whether this controller added the ID.
   *
   * @param {Element | null} trigger
   */
  function setDescribedByTrigger(trigger) {
    if (describedByTrigger === trigger) {
      return;
    }
    clearDescribedByTrigger();
    if (trigger === null) {
      return;
    }
    const tokens = attributeTokens(trigger, "aria-describedby");
    describedByAdded = !tokens.includes(tooltipId);
    if (describedByAdded) {
      tokens.push(tooltipId);
      trigger.setAttribute("aria-describedby", tokens.join(" "));
    }
    describedByTrigger = trigger;
  }

  /**
   * Remove this controller's added aria-describedby token and clear its
   * link state. Preserve a preexisting identical token and every unrelated
   * token. Return nothing; repeat calls are harmless.
   */
  function clearDescribedByTrigger() {
    if (describedByTrigger !== null && describedByAdded) {
      const remaining = attributeTokens(describedByTrigger, "aria-describedby").filter(
        (token) => token !== tooltipId,
      );
      if (remaining.length === 0) {
        describedByTrigger.removeAttribute("aria-describedby");
      } else {
        describedByTrigger.setAttribute("aria-describedby", remaining.join(" "));
      }
    }
    describedByTrigger = null;
    describedByAdded = false;
  }

  /**
   * Clear active-owner state, owned ARIA link, tooltip content and visual
   * attributes/size limits. Hide the card without dropping remembered pointer
   * or focused-owner state, so refresh can show it again. Does not destroy
   * listeners or unregister owners.
   */
  function hide() {
    activeOwner = null;
    championCardExpanded = false;
    championHold?.cancel();
    clearDescribedByTrigger();
    closeTopLayer();
    tooltip.hidden = true;
    tooltip.style.visibility = "";
    tooltip.removeAttribute(TOOLTIP_KIND_ATTRIBUTE);
    tooltip.removeAttribute(TOOLTIP_TONE_ATTRIBUTE);
    tooltip.removeAttribute(TOOLTIP_ACCENT_ATTRIBUTE);
    tooltip.removeAttribute("data-tooltip-placement");
    tooltip.style.maxWidth = "";
    tooltip.style.maxHeight = "";
    title.textContent = "";
    if (typeof details.replaceChildren === "function") {
      details.replaceChildren();
    } else {
      details.textContent = "";
    }
  }

  /**
   * Render candidate's compact descriptor (a Champion Card descriptor draws
   * the Champion Card instead) and place its card near pointer or the owner
   * center. Hide instead if owner left root. fromFocus controls
   * ARIA linkage; pointer anchoring uses a point rectangle. Read DOM geometry,
   * measure with protected regions and update tooltip styles/attributes.
   * Candidate must be registered and its descriptor valid; errors propagate.
   *
   * @param {{element: Element, descriptor: AnyTooltipDescriptor, trigger?: Element}} candidate
   * @param {TooltipPoint | null} pointer
   * @param {boolean} fromFocus
   */
  function show(candidate, pointer, fromFocus) {
    if (!ownerWithinRoot(root, candidate.element)) {
      hide();
      return;
    }
    const descriptor = candidate.descriptor;
    activeOwner = candidate.element;
    setDescribedByTrigger(fromFocus ? (candidate.trigger ?? candidate.element) : null);
    const championCard = championCardOf(descriptor);
    const championFacts = championFactsByDescriptor.get(descriptor) ?? null;
    const drawsChampion =
      championCard !== null && typeof details.replaceChildren === "function";
    const championBox = drawsChampion ? areaOf(options.championCardBox) : null;
    // A plain card inside the arena box is drawn once the tooltip shows, by
    // renderFittedChampionCard, with as many rows as fit the box.
    const fitsRows =
      championBox !== null && !championCardExpanded && championFacts !== null;
    /** @type {HTMLElement | null} */
    let championRoot = null;
    if (championCard !== null && drawsChampion) {
      // The Champion Card draws its own title; the card's text stays in the
      // descriptor for screen readers.
      title.textContent = "";
      if (fitsRows) {
        details.replaceChildren();
      } else {
        const model =
          championCardExpanded && championFacts !== null
            ? championCardModel(championFacts, { expanded: true })
            : championCard;
        championRoot = renderChampionCard(details, model);
        if (!model.expanded) championShownRows = model.rows.length;
      }
    } else {
      renderSemanticDescriptor({ descriptor, title, details, surface: "compact" });
    }
    tooltip.setAttribute(TOOLTIP_KIND_ATTRIBUTE, descriptor.kind);
    tooltip.setAttribute(TOOLTIP_TONE_ATTRIBUTE, descriptor.tone);
    tooltip.setAttribute(TOOLTIP_ACCENT_ATTRIBUTE, descriptor.accent);
    tooltip.style.visibility = "hidden";
    tooltip.hidden = false;
    openTopLayer();

    if (championBox !== null) {
      // The Champion Card is placed by the live hover card's own rule: fully
      // inside the arena box (owner, 2 October 2026), except a Hold Shift
      // card that grows past it ("Grow past arena").
      tooltip.style.maxWidth = "";
      tooltip.style.maxHeight = "";
      if (fitsRows && championFacts !== null) {
        const fitted = renderFittedChampionCard({
          host: details,
          surface: tooltip,
          facts: championFacts,
          box: championBox,
        });
        championRoot = fitted.root;
        championShownRows = fitted.model.rows.length;
      }
    }
    if (championRoot !== null && championBox !== null) {
      const owner = candidate.element.getBoundingClientRect();
      const place = placeChampionCard({
        surface: tooltip,
        root: championRoot,
        box: championBox,
        pointer: pointer ?? {
          x: (owner.left + owner.right) / 2,
          y: (owner.top + owner.bottom) / 2,
        },
        subject: owner,
      });
      tooltip.style.left = `${place.left}px`;
      tooltip.style.top = `${place.top}px`;
      tooltip.setAttribute(
        "data-tooltip-placement",
        place.pastArena ? "past-arena" : "arena-box",
      );
      tooltip.style.visibility = "";
      return;
    }

    const ownerRect = normalizeRectangle(
      candidate.element.getBoundingClientRect(),
      "tooltip owner rectangle",
    );
    const anchorPoint =
      pointer === null
        ? Object.freeze({
            x: (ownerRect.left + ownerRect.right) / 2,
            y: (ownerRect.top + ownerRect.bottom) / 2,
          })
        : pointer;
    const anchorRect =
      descriptor.anchor === "pointer"
        ? Object.freeze({
            left: anchorPoint.x,
            top: anchorPoint.y,
            right: anchorPoint.x,
            bottom: anchorPoint.y,
          })
        : ownerRect;
    const viewport = ownerSurfaceRectangle(candidate.element, tooltip.ownerDocument);
    const protectedRects = localProtectedRects(candidate.element);
    const measured = measureTooltipForPlacement({
      tooltip,
      anchorRect,
      pointer: anchorPoint,
      viewport,
      protectedRects,
    });
    const placement = placeTooltip({
      anchorRect,
      pointer: anchorPoint,
      tooltipSize: {
        width: measured.width,
        height: measured.height,
      },
      viewport,
      protectedRects,
    });
    tooltip.style.left = `${placement.left}px`;
    tooltip.style.top = `${placement.top}px`;
    tooltip.setAttribute("data-tooltip-placement", placement.placement);
    tooltip.style.visibility = "";
  }

  /**
   * Return the highest-ranked registered owner under pointer, or null.
   * Use elementsFromPoint paint order, ignore duplicate/disconnected/outside
   * owners and reuse stored descriptors. pointer uses viewport CSS pixels.
   * This reads DOM hit testing without changing registration or focus.
   *
   * @param {TooltipPoint} pointer
   * @returns {{element: Element, descriptor: AnyTooltipDescriptor} | null}
   */
  function candidateAtPointer(pointer) {
    const elements = elementsFromPoint(root, pointer.x, pointer.y);
    /** @type {Array<{element: Element, descriptor: AnyTooltipDescriptor, paintOrder: number}>} */
    const candidates = [];
    const visited = new Set();

    for (const [paintOrder, element] of elements.entries()) {
      const owner = closestRegisteredOwner(element);
      if (owner === null || visited.has(owner) || !ownerWithinRoot(root, owner)) {
        continue;
      }
      const descriptor = descriptorByOwner.get(owner);
      if (descriptor === undefined || suppressed(descriptor, owner)) {
        continue;
      }
      visited.add(owner);
      candidates.push({ element: owner, descriptor, paintOrder });
    }

    return chooseTooltipCandidate(candidates);
  }

  /**
   * Resolve target or its registered ancestor inside root. Return owner,
   * stored descriptor and original element trigger, or null for invalid,
   * unregistered or disconnected targets. No nearest-by-distance fallback occurs.
   *
   * @param {unknown} target
   * @returns {{element: Element, descriptor: AnyTooltipDescriptor, trigger: Element} | null}
   */
  function candidateForTarget(target) {
    if (!isElementNode(target)) {
      return null;
    }
    const owner = closestRegisteredOwner(target);
    if (owner === null || !ownerWithinRoot(root, owner)) {
      return null;
    }
    const descriptor = descriptorByOwner.get(owner);
    return descriptor === undefined || suppressed(descriptor, owner)
      ? null
      : { element: owner, descriptor, trigger: target };
  }

  /**
   * Recompute the visible card after data, layout or owner changes.
   * Do nothing after destroy. Respect Escape dismissal until interaction;
   * otherwise prefer a pinned owner that is still connected and allowed,
   * then the last pointer, then a still-connected focused owner. A pin that
   * is gone or turned off is dropped. Hide if no candidate remains. May
   * render, measure and reposition DOM.
   */
  function refresh() {
    if (destroyed) {
      return;
    }
    if (dismissedUntilInteraction) {
      hide();
      return;
    }
    if (pinned !== null) {
      const descriptor = descriptorByOwner.get(pinned);
      if (
        descriptor !== undefined &&
        !suppressed(descriptor, pinned) &&
        ownerWithinRoot(root, pinned)
      ) {
        show({ element: pinned, descriptor }, null, false);
        return;
      }
      pinned = null;
    }
    if (lastPointer !== null) {
      const candidate = candidateAtPointer(lastPointer);
      if (candidate === null) {
        hide();
      } else {
        show(candidate, lastPointer, false);
      }
      return;
    }
    if (focusedOwner !== null) {
      const descriptor = descriptorByOwner.get(focusedOwner);
      if (
        descriptor !== undefined &&
        focusedTrigger !== null &&
        ownerWithinRoot(root, focusedOwner) &&
        ownerWithinRoot(root, focusedTrigger)
      ) {
        show(
          { element: focusedOwner, descriptor, trigger: focusedTrigger },
          null,
          true,
        );
        return;
      }
    }
    hide();
  }

  /**
   * Handle a delegated pointer event using finite client coordinates.
   * Clear Escape dismissal, remember the pointer and show its best owner or
   * hide when none exists. Invalid coordinates hide the card. No event default
   * is prevented and no inspection callback runs.
   *
   * @param {Event} event
   */
  function onPointerMove(event) {
    const point = eventPoint(event);
    if (point === null) {
      hide();
      return;
    }
    dismissedUntilInteraction = false;
    lastPointer = point;
    if (pinned !== null) {
      return;
    }
    const candidate = candidateAtPointer(point);
    if (candidate === null) {
      hide();
    } else {
      show(candidate, point, false);
    }
  }

  /**
   * Forget the last pointer and refresh from any remaining focused owner.
   * This can hide or reposition the card; it does not clear focus state.
   */
  function onPointerLeave() {
    lastPointer = null;
    refresh();
  }

  /**
   * Resolve the focused target, clear Escape dismissal and show its card
   * with an ARIA link. Forget pointer state for a valid candidate. Invalid
   * targets clear remembered focus and hide. No native focus change is made.
   *
   * @param {Event} event
   */
  function onFocusIn(event) {
    const candidate = candidateForTarget(event.target);
    if (candidate === null) {
      focusedOwner = null;
      focusedTrigger = null;
      focusKey = null;
      focusKeyElement = null;
      if (pinned === null) {
        hide();
      }
      return;
    }
    dismissedUntilInteraction = false;
    focusedOwner = candidate.element;
    focusedTrigger = candidate.trigger;
    focusKey = candidate.element.getAttribute(SYMBOL_KEY_ATTRIBUTE);
    focusKeyElement = focusKey === null ? null : candidate.element;
    lastPointer = null;
    if (pinned !== null && pinned !== candidate.element) {
      pinned = null;
    }
    show(candidate, null, true);
  }

  /**
   * Use relatedTarget to follow focus to another registered owner.
   * Show that card with focus linkage, or clear focus and hide when no
   * candidate remains. This does not call focus() or submit an action.
   *
   * @param {Event} event
   */
  function onFocusOut(event) {
    const nextTarget = "relatedTarget" in event ? event.relatedTarget : null;
    const candidate = candidateForTarget(nextTarget);
    if (candidate === null) {
      const leaving = focusKeyElement;
      if (leaving !== null) {
        // Keep the key only when a redraw removed the node; a real blur clears it.
        queueMicrotask(() => {
          if (focusKeyElement === leaving && ownerWithinRoot(root, leaving)) {
            focusKey = null;
            focusKeyElement = null;
          }
        });
      }
      focusedOwner = null;
      focusedTrigger = null;
      if (pinned === null) {
        hide();
      } else {
        refresh();
      }
      return;
    }
    focusedOwner = candidate.element;
    focusedTrigger = candidate.trigger;
    show(candidate, null, true);
  }

  /**
   * Hide a visible tooltip on Escape until a new interaction. On Enter
   * or Space, request inspection and prevent the native default only when
   * inspection succeeds. Leave native interactive controls to their own
   * actions; do not stop event propagation.
   *
   * @param {Event} event
   */
  function onKeyDown(event) {
    if (
      "key" in event &&
      event.key === "Escape" &&
      (!tooltip.hidden || pinned !== null)
    ) {
      pinned = null;
      dismissedUntilInteraction = true;
      hide();
      return;
    }
    if ("key" in event && event.key === "Enter") {
      const candidate = candidateForTarget(event.target);
      if (
        candidate !== null &&
        candidate.descriptor.kind === SYMBOL_TIP_KIND &&
        !isNativeInteractiveTarget(candidate.trigger, candidate.element)
      ) {
        togglePin(candidate, null);
        event.preventDefault();
      }
    }
    if (
      "key" in event &&
      (event.key === "Enter" || event.key === " ") &&
      inspect(event.target)
    ) {
      event.preventDefault();
    }
  }

  /**
   * Pin candidate's card, or close it when it is already the pinned one (the
   * second tap). Closing also dismisses hover until the next pointer move.
   *
   * @param {{element: Element, descriptor: AnyTooltipDescriptor, trigger?: Element}} candidate
   * @param {TooltipPoint | null} pointer Where the click landed, or null.
   */
  function togglePin(candidate, pointer) {
    if (candidate.element === pinned) {
      pinned = null;
      dismissedUntilInteraction = true;
      hide();
      return;
    }
    pinned = candidate.element;
    dismissedUntilInteraction = false;
    show(candidate, pointer, false);
  }

  /**
   * Pin or unpin on click or tap, then try to inspect the event target through
   * the shared inspect method. A click on a "symbol" owner, or a touch or pen
   * tap on any owner that is not a native control, toggles its pin; a click
   * anywhere else drops the current pin. The inspect result is ignored and
   * event propagation/defaults are unchanged.
   *
   * @param {Event} event
   */
  function onClick(event) {
    const candidate = candidateForTarget(event.target);
    const tap =
      "pointerType" in event &&
      (event.pointerType === "touch" || event.pointerType === "pen");
    if (
      candidate !== null &&
      (candidate.descriptor.kind === SYMBOL_TIP_KIND ||
        (tap && !isNativeInteractiveTarget(candidate.trigger, candidate.element)))
    ) {
      togglePin(candidate, eventPoint(event));
    } else if (pinned !== null) {
      pinned = null;
      refresh();
    }
    inspect(event.target);
  }

  /**
   * Request full-card inspection for target, defaulting to activeOwner.
   * Return false after destroy, without onInspect, for invalid/unregistered
   * owners, explicitly noninspectable owners or native interactive targets.
   * Otherwise call onInspect once with frozen owner/trigger context and return
   * true. The callback owns rendering; callback exceptions propagate.
   *
   * @param {unknown} [target]
   * @returns {boolean}
   */
  function inspect(target = activeOwner) {
    if (destroyed || onInspect === undefined || target === null) {
      return false;
    }
    const candidate = candidateForTarget(target);
    if (
      candidate === null ||
      inspectableByOwner.get(candidate.element) === false ||
      isNativeInteractiveTarget(candidate.trigger, candidate.element)
    ) {
      return false;
    }
    onInspect(
      candidate.descriptor,
      Object.freeze({ owner: candidate.element, trigger: candidate.trigger ?? null }),
    );
    return true;
  }

  /**
   * Refresh card content/placement after a delegated scroll or window
   * resize. This uses current owner geometry; no event object is required.
   */
  function onViewportChange() {
    refresh();
  }

  const eventRoot = /** @type {EventTarget} */ (root);
  eventRoot.addEventListener("pointermove", onPointerMove, true);
  eventRoot.addEventListener("pointerleave", onPointerLeave);
  eventRoot.addEventListener("focusin", onFocusIn, true);
  eventRoot.addEventListener("focusout", onFocusOut, true);
  eventRoot.addEventListener("keydown", onKeyDown, true);
  eventRoot.addEventListener("click", onClick, true);
  eventRoot.addEventListener("scroll", onViewportChange, true);

  const ownerDocument =
    "ownerDocument" in root && root.ownerDocument !== null
      ? root.ownerDocument
      : /** @type {Document} */ (root);
  const ownerWindow = ownerDocument.defaultView;
  ownerWindow?.addEventListener?.("resize", onViewportChange);

  // The Champion Card's Hold Shift gesture: only for a card that is drawn
  // (live play hides the tooltip copy of the card), shown for a resting
  // pointer, not pinned, and leaving rows out (the four-row limit, or a short
  // arena box: the drawn card shows fewer rows than the champion's statuses).
  championHold = createShiftHold({
    target:
      typeof ownerWindow?.addEventListener === "function" ? ownerWindow : eventRoot,
    canStart: () => {
      if (
        destroyed ||
        activeOwner === null ||
        pinned !== null ||
        lastPointer === null ||
        tooltip.hidden ||
        (typeof tooltip.getClientRects === "function" &&
          tooltip.getClientRects().length === 0)
      ) {
        return false;
      }
      return (
        (championCardOf(descriptorByOwner.get(activeOwner))?.total ?? 0) >
        championShownRows
      );
    },
    onExpand: () => {
      championCardExpanded = true;
      refresh();
    },
    onShrink: () => {
      championCardExpanded = false;
      refresh();
    },
  });

  const mutationObserver = createRemovalObserver(root, () => {
    let pinMoved = false;
    if (pinned !== null && !ownerWithinRoot(root, pinned)) {
      pinned = replacementOwner(pinned);
      pinMoved = true;
    }
    const activeRemoved = activeOwner !== null && !ownerWithinRoot(root, activeOwner);
    const focusRemoved =
      (focusedOwner !== null && !ownerWithinRoot(root, focusedOwner)) ||
      (focusedTrigger !== null && !ownerWithinRoot(root, focusedTrigger));
    if (focusRemoved) {
      focusedOwner = null;
      focusedTrigger = null;
    }
    restoreSymbolFocus();
    if (activeRemoved || focusRemoved || pinMoved) {
      refresh();
    }
  });

  /**
   * After a redraw, move keyboard focus to the node that now shows the last
   * focused symbol (same data-symbol-key), but only when that node's old
   * element is gone and focus fell to the page (no element, or body). Focus
   * moves with preventScroll; the focusin handler then shows its card.
   */
  function restoreSymbolFocus() {
    if (focusKey === null || focusKeyElement === null) {
      return;
    }
    if (ownerWithinRoot(root, focusKeyElement)) {
      return;
    }
    const ownerDocument = tooltip.ownerDocument;
    const active = ownerDocument?.activeElement ?? null;
    if (active !== null && active !== ownerDocument.body) {
      return;
    }
    const replacement = replacementOwner(focusKeyElement, focusKey);
    if (replacement === null) {
      return;
    }
    focusKeyElement = replacement;
    const focusable = /** @type {{focus?: (options?: FocusOptions) => void}} */ (
      /** @type {unknown} */ (replacement)
    );
    focusable.focus?.({ preventScroll: true });
  }

  /**
   * Remove all controller listeners, disconnect the removal observer,
   * hide the card and release tooltip ownership. Repeat calls do nothing.
   * This does not delete registered owner descriptions or remove supplied DOM.
   */
  function destroy() {
    if (destroyed) {
      return;
    }
    destroyed = true;
    pinned = null;
    focusKey = null;
    focusKeyElement = null;
    eventRoot.removeEventListener("pointermove", onPointerMove, true);
    eventRoot.removeEventListener("pointerleave", onPointerLeave);
    eventRoot.removeEventListener("focusin", onFocusIn, true);
    eventRoot.removeEventListener("focusout", onFocusOut, true);
    eventRoot.removeEventListener("keydown", onKeyDown, true);
    eventRoot.removeEventListener("click", onClick, true);
    eventRoot.removeEventListener("scroll", onViewportChange, true);
    ownerWindow?.removeEventListener?.("resize", onViewportChange);
    mutationObserver?.disconnect();
    championHold?.destroy();
    hide();
    controllerByTooltip.delete(tooltip);
  }

  /**
   * Hide the card and drop any pin (the public hide). Later interaction or
   * refresh may show a card again.
   */
  function hideAndUnpin() {
    pinned = null;
    hide();
  }

  /**
   * Turn every "symbol" card on or off (the symbol_explanations switch). Off
   * drops a symbol pin and hides a symbol card now; on lets the next hover,
   * tap, focus or refresh show one. Other tooltip kinds are not affected.
   *
   * @param {boolean} enabled
   */
  function setSymbolTipsEnabled(enabled) {
    symbolTipsEnabled = Boolean(enabled);
    refresh();
  }

  /**
   * Return the owner whose card is pinned, or null when nothing is pinned.
   *
   * @returns {Element | null}
   */
  function pinnedOwner() {
    return pinned;
  }

  /** @type {TooltipController} */
  const controller = Object.freeze({
    refresh,
    hide: hideAndUnpin,
    inspect,
    destroy,
    setSymbolTipsEnabled,
    pinnedOwner,
  });
  controllerByTooltip.set(tooltip, controller);
  hide();
  return controller;
}

// ---------------------------------------------------------------------------
// The Champion Card (live play: #hcard; Replay Viewer: the tooltip card)
// ---------------------------------------------------------------------------

/**
 * The elements the live hover card answers to, as in the vision page: a
 * champion's body in the arena, a scorebug portrait, a Turn Checklist row, an
 * Allowed Now button, or any element that names a champion by public ID.
 */
export const HOVER_CARD_SELECTOR =
  "#battlefield .agent[data-presentation-key], .por[data-slot], .cl-row[data-slot], .pick[data-target], [data-public-agent-id]";

/** The most status rows a Champion Card shows before "+N more" (owner, 2 October 2026). */
export const CHAMPION_CARD_STATUS_LIMIT = 4;

/** The descriptor kind of a Champion Card shown by the shared tooltip controller. */
export const CHAMPION_CARD_KIND = "champion";

/** The Champion Card model of each Champion Card descriptor (see createChampionCardDescriptor). */
const championCardByDescriptor = new WeakMap();

/** The facts of each Champion Card descriptor, for its Hold Shift card. */
const championFactsByDescriptor = new WeakMap();

/**
 * How long Shift must be held by itself, with the pointer resting on the
 * hovered champion, before its Champion Card shows every status, in
 * milliseconds (owner, 2 October 2026: "about 300 ms").
 */
export const CHAMPION_CARD_HOLD_MS = 300;

/**
 * The smallest gap, in CSS pixels, between a Champion Card and the arena box's
 * edge (or, for a Hold Shift card that grows past the box, the window's edge).
 */
export const CHAMPION_CARD_BOX_MARGIN = 8;

/**
 * One status on the Champion Card, as the caller read it from the audience's
 * own scene.
 *
 * - id: a status name ID of `symbols.js` STATUS_NAMES ("slow_warrior_charge",
 *   "spawn_shield", "in_combat", "rest", "stun_broken", "mage_amplification",
 *   "warrior_mitigation"). An unknown ID shows as "Recorded Status".
 * - ticks: whole ticks left, or null for a status with no count (Rest, an
 *   aura, Freezing Trap (Stun Broken)).
 * - full: its full length in ticks, for the line that drains under the row, or
 *   null when unknown (no line is drawn).
 * - amount: the number its sentence needs (a multiplier, Blessing of Freedom's
 *   lowest speed multiplier or the life Rest mends per tick), or null.
 *
 * @typedef {Readonly<{
 *   id: string,
 *   ticks: number | null,
 *   full: number | null,
 *   amount: number | null,
 * }>} ChampionCardStatus
 */

/**
 * Everything the Champion Card says about one champion. The caller builds it
 * from the audience's own scene, so an Agent POV card never holds a hidden
 * fact. Agent POV and Oracle View build the same facts for a seen champion.
 *
 * - state: "seen", or "unseen" for a champion the viewing team cannot see
 *   now (Agent POV only): the reduced card, with no life, Ultimate or status.
 * - name and calling: "Mage" and "The Astral Walker", or null when unknown.
 * - classId: 1 (Mage) to 5 (Priest), or null; it colors the Ultimate bar.
 * - displayId: the shown Agent ID ("5"), or null.
 * - teamId: 1 (Team A) or 2 (Team B), or null; it draws the ID line's crest.
 * - life: current and maximum life in hit points, or null.
 * - ultimate: the Ultimate's name ("Burst"), ticks until it is ready (0 means
 *   Ready) and its full recharge in ticks (null when unknown), or null.
 * - statuses: every status, in any order (the card sorts them).
 * - note: one extra muted line, or null: "Fallen · Returns In 3" or "Fallen"
 *   for a fallen champion (championCardFactsFromAgent writes it), else null.
 * - you: true when the viewer plays this champion ("You play this champion").
 * - playedBy: who plays it ("tdm-beta"), or null.
 * - yourTarget: true when it is the target of the order being set.
 * - unseenBy: the team that cannot see it ("Team A"), or null for "your team".
 *
 * @typedef {Readonly<{
 *   state: "seen" | "unseen",
 *   name: string | null,
 *   calling: string | null,
 *   classId: 1 | 2 | 3 | 4 | 5 | null,
 *   displayId: string | null,
 *   teamId: 1 | 2 | null,
 *   life: Readonly<{current: number, maximum: number}> | null,
 *   ultimate: Readonly<{name: string | null, remaining: number, full: number | null}> | null,
 *   statuses: ReadonlyArray<ChampionCardStatus>,
 *   note: string | null,
 *   you: boolean,
 *   playedBy: string | null,
 *   yourTarget: boolean,
 *   unseenBy: string | null,
 * }>} ChampionCardFacts
 */

/**
 * One status row of a Champion Card model: icon (a sigil key of `icons.js`,
 * or null), name ("Charge") and effect ("Slow", or null), time ("3 ticks
 * left", "This tick", "Now", "While near" or ""), line (what it does with its
 * number, then where it comes from by class), drain (the share of its time
 * left, 0 to 1, or null for no drain line) and paint (the drain line's color:
 * a class token, "rime", "moon", "neutral" or "dew").
 *
 * @typedef {Readonly<{
 *   id: string,
 *   icon: string | null,
 *   name: string,
 *   effect: string | null,
 *   time: string,
 *   line: string,
 *   drain: number | null,
 *   paint: string,
 * }>} ChampionCardRow
 */

/**
 * The words and shares of one Champion Card, ready to draw
 * (championCardModel builds it; renderChampionCard draws it).
 *
 * - title: "Mage · The Astral Walker"; idText: "Agent ID 5" or null; teamId.
 * - unseenText: "none of your team can see it now." (after a bold "Unseen")
 *   for the reduced card, else null.
 * - life: the shared Life bar's model (`life-bar.js` lifeBarModel: text "77",
 *   maximumText "80", fraction 0 to 1, tone "good" above half, "warn" at half
 *   or less, "low" at 30% or less), or null.
 * - ultimate: {label (its name, or "Ultimate"), ready, fraction 0 to 1, text
 *   ("30 ticks" or "Ready"), classToken ("mage" ... or null)}, or null.
 * - rows: the status rows shown, stuns first, then slows, then the rest: at
 *   most CHAMPION_CARD_STATUS_LIMIT (fewer when the model was built with a
 *   rowLimit, for a short arena box), or every row when expanded.
 * - more: how many rows were left out ("+2 more"); 0 when expanded.
 * - total: how many status rows the champion has in all.
 * - expanded: true for the Hold Shift card that shows every status ("All 7
 *   statuses").
 * - note, played ("Played by tdm-beta" or null), you, yourTarget.
 *
 * @typedef {Readonly<{
 *   state: "seen" | "unseen",
 *   title: string,
 *   idText: string | null,
 *   teamId: 1 | 2 | null,
 *   unseenText: string | null,
 *   life: import("./life-bar.js").LifeBarModel | null,
 *   ultimate: Readonly<{label: string, ready: boolean, fraction: number, text: string, classToken: string | null}> | null,
 *   rows: ReadonlyArray<ChampionCardRow>,
 *   more: number,
 *   total: number,
 *   expanded: boolean,
 *   note: string | null,
 *   played: string | null,
 *   you: boolean,
 *   yourTarget: boolean,
 * }>} ChampionCardModel
 */

/**
 * The champion a hover-card element names, read from its data attributes:
 * presentationKey (an arena body's data-presentation-key), slot (a portrait's
 * or row's data-slot, or an Allowed Now button's data-target: the global slot,
 * which is also the shown Agent ID) and publicAgentId (data-public-agent-id).
 * classId comes from data-class-id when the element carries it. Every field
 * is null when absent.
 *
 * @typedef {Readonly<{
 *   presentationKey: string | null,
 *   slot: number | null,
 *   publicAgentId: string | null,
 *   classId: number | null,
 * }>} HoverCardSubject
 */

/**
 * Write a whole number of ticks the way the vision does: "1 tick", "5 ticks",
 * or with titleCase "1 Tick", "5 Ticks". Pure.
 *
 * @param {number} count Ticks, a whole number of at least 0.
 * @param {boolean} [titleCase] True for "Tick" and "Ticks". Default false.
 * @returns {string}
 */
function tickWords(count, titleCase = false) {
  const word = titleCase ? "Tick" : "tick";
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/**
 * Read one champion's Champion Card facts from its row in the audience's own
 * arena scene (an AuthorizedAgentV1 record). Shared by the live hover card
 * (`shell.js`), the Replay Viewer's arena bodies (`shell.js`, through
 * `main.js`) and the Replay Viewer's HUD frames (`looks/hud.js`), so every
 * client writes the same card. Pure.
 *
 * Reads: class_id (name, calling and the Ultimate's name from
 * `class-documentation.js`), display_agent_id, team_id, life_state or alive,
 * current_health, maximum_health (or max_health), ultimate_cooldown_remaining
 * (or ultimate_cooldown), spawn_shield_remaining, statuses (status_id,
 * remaining_duration, configured_duration_steps, magnitude),
 * steps_until_out_of_combat, out_of_combat_delay_steps,
 * out_of_combat_health_regeneration_fraction_per_step and aura_modifiers
 * (aura_id, multiplier). A fallen champion shows life 0, no Ultimate or
 * status, and the note "Fallen · Returns In N" (N = its team's
 * countdown_steps + 1 in context.respawnWaves, the public wave clock that the
 * HUD's fallen frame and the scorebug also count), or just "Fallen" when the
 * scene has no clock for its team; this is the one place the card's fallen
 * note is written, so arena bodies, portraits and frames in every client
 * show the same line. A living champion shows: Spawn Shield while it lasts; each status
 * row (a row naming Spawn Shield or In Combat is not shown twice);
 * Freezing Trap (Stun Broken) when context.trapBroke is true; In Combat
 * while steps_until_out_of_combat is above 0; Rest when it is 0 and life is
 * below full (the life it mends per tick: full life times the regeneration
 * share, times the healing multiplier of Crippling Poison (Healing
 * Reduction) while that lasts, as Core does); and each aura it stands in.
 * Sources are never read: the card names them by class.
 *
 * @param {Readonly<Record<string, any>>} agent The scene row.
 * @param {{
 *   classMechanics?: ReadonlyArray<Readonly<Record<string, any>>>,
 *   spawnShieldSteps?: number | null,
 *   trapBroke?: boolean,
 *   respawnWaves?: ReadonlyArray<Readonly<Record<string, any>>> | null,
 *   you?: boolean,
 *   playedBy?: string | null,
 *   yourTarget?: boolean,
 * }} [context] classMechanics: the scene's class_mechanics rows (the
 *   Ultimate's full recharge, ultimate_cooldown_steps; default none);
 *   spawnShieldSteps: the Spawn Shield's full length (default null); trapBroke:
 *   the audience saw this champion's Freezing Trap break this tick (default
 *   false); respawnWaves: the scene's respawn_waves rows (team_id 1 or 2 and
 *   countdown_steps, a whole number of at least 0), read only for a fallen
 *   champion (default none: the note is "Fallen"); you, playedBy and
 *   yourTarget are copied (defaults false, null, false).
 * @returns {ChampionCardFacts} Frozen facts with state "seen".
 */
export function championCardFactsFromAgent(agent, context = {}) {
  const classId = [1, 2, 3, 4, 5].includes(agent.class_id) ? agent.class_id : null;
  const names = classNamesV1(classId);
  const alive =
    agent.alive === true || (agent.alive !== false && agent.life_state === "alive");
  const maximum = agent.maximum_health ?? agent.max_health;
  const current = alive ? agent.current_health : 0;
  const remaining = agent.ultimate_cooldown_remaining ?? agent.ultimate_cooldown;
  const full = (context.classMechanics ?? []).find(
    (row) => row?.class_id === classId,
  )?.ultimate_cooldown_steps;
  /** @type {ChampionCardStatus[]} */
  const statuses = [];
  /** @param {string} id @param {unknown} ticks @param {unknown} length @param {unknown} amount */
  const add = (id, ticks, length, amount) =>
    statuses.push(
      Object.freeze({
        id,
        ticks: Number.isInteger(ticks) ? /** @type {number} */ (ticks) : null,
        full:
          Number.isInteger(length) && /** @type {number} */ (length) > 0
            ? /** @type {number} */ (length)
            : null,
        amount: typeof amount === "number" && Number.isFinite(amount) ? amount : null,
      }),
    );
  if (alive) {
    if (
      Number.isInteger(agent.spawn_shield_remaining) &&
      agent.spawn_shield_remaining > 0
    ) {
      add("spawn_shield", agent.spawn_shield_remaining, context.spawnShieldSteps, null);
    }
    let healingMultiplier = 1;
    for (const status of Array.isArray(agent.statuses) ? agent.statuses : []) {
      if (status === null || typeof status !== "object") continue;
      const id = statusTokenIdFromCatalogId(status.status_id);
      if (id === null) continue;
      if (id === "anti_heal_rogue_poison" && typeof status.magnitude === "number") {
        healingMultiplier = status.magnitude;
      }
      // The row's own Spawn Shield count is already shown above.
      if (id === "spawn_shield" && statuses.some((row) => row.id === id)) continue;
      add(
        id,
        status.remaining_duration,
        id === "in_combat"
          ? (status.configured_duration_steps ?? agent.out_of_combat_delay_steps)
          : status.configured_duration_steps,
        status.magnitude,
      );
    }
    if (context.trapBroke === true) add("stun_broken", null, null, null);
    const quiet = agent.steps_until_out_of_combat;
    // A scene may list In Combat among its statuses too; show it once.
    if (
      Number.isInteger(quiet) &&
      quiet > 0 &&
      !statuses.some((row) => row.id === "in_combat")
    ) {
      add("in_combat", quiet, agent.out_of_combat_delay_steps, null);
    }
    const share = agent.out_of_combat_health_regeneration_fraction_per_step;
    if (
      quiet === 0 &&
      typeof current === "number" &&
      typeof maximum === "number" &&
      current < maximum
    ) {
      add(
        "rest",
        null,
        null,
        typeof share === "number" ? maximum * share * healingMultiplier : null,
      );
    }
    for (const modifier of Array.isArray(agent.aura_modifiers)
      ? agent.aura_modifiers
      : []) {
      const id = statusNameFor(modifier?.aura_id)?.id;
      if (id === "mage_amplification" || id === "warrior_mitigation") {
        add(id, null, null, modifier.multiplier);
      }
    }
  }
  // A fallen champion returns with its team's next wave: countdown + 1 ticks.
  const countdown = alive
    ? undefined
    : (context.respawnWaves ?? []).find((wave) => wave?.team_id === agent.team_id)
        ?.countdown_steps;
  const note = alive
    ? null
    : Number.isInteger(countdown) && countdown >= 0
      ? `Fallen · Returns In ${countdown + 1}`
      : "Fallen";
  return Object.freeze({
    state: /** @type {const} */ ("seen"),
    name: names?.className ?? null,
    calling: names?.calling ?? null,
    classId,
    displayId:
      agent.display_agent_id === undefined || agent.display_agent_id === null
        ? null
        : String(agent.display_agent_id),
    teamId: agent.team_id === 1 || agent.team_id === 2 ? agent.team_id : null,
    life:
      typeof current === "number" && typeof maximum === "number"
        ? Object.freeze({ current, maximum })
        : null,
    ultimate:
      alive && Number.isInteger(remaining)
        ? Object.freeze({
            name: classAbilityNameV1(classId, "ultimate"),
            remaining,
            full: Number.isInteger(full) && full > 0 ? full : null,
          })
        : null,
    statuses: Object.freeze(statuses),
    note,
    you: context.you === true,
    playedBy: context.playedBy ?? null,
    yourTarget: context.yourTarget === true,
    unseenBy: null,
  });
}

/**
 * Turn a champion's facts into the Champion Card model (design C "Clean
 * List", owner, 2 October 2026). Pure.
 *
 * Seen champion: the title "Mage · The Astral Walker", "Agent ID 5", the life
 * bar (`life-bar.js` lifeBarModel: "77" of "80", colored green, yellow or
 * red), the Ultimate bar (labeled with the Ultimate's own name, filled by the
 * share recharged, "30 ticks" or "Ready"), then one row per status: the
 * status's own icon (statusSigilKey or modifierSigilKey of `icons.js`), its
 * approved name and effect (`symbols.js` statusNameFor), its time
 * (statusTimeText of `semantic-vocabulary.js`: "5 ticks left", "This tick",
 * "Now", "While near"), what it does with the exact number and where it comes
 * from by class (statusCardSentence, with the champion's own class, so its
 * own aura reads "From its own aura."), and the share of its time left for
 * the drain line. Rows go stuns first, then slows, then the rest
 * (statusOrderIndex), and more ticks left first within one status. The card
 * shows at most CHAMPION_CARD_STATUS_LIMIT rows (or options.rowLimit, the
 * fewer rows a short arena box has room for: renderFittedChampionCard picks
 * it), with `more` counting the rest; options.expanded (the Hold Shift card,
 * owner, 2 October 2026) shows every row and `more` is 0. Then the note,
 * "Played by <who>" (left out when you play it) and the you and yourTarget
 * flags. Unseen champion: the title, the ID line and unseenText only
 * (expanded and rowLimit change nothing). A missing fact leaves its part out.
 *
 * @param {ChampionCardFacts} facts
 * @param {{expanded?: boolean, rowLimit?: number}} [options] expanded: show
 *   every status row; default false. rowLimit: the most status rows a card
 *   that is not expanded shows, a whole number from 1 to
 *   CHAMPION_CARD_STATUS_LIMIT (a smaller number counts as 1, a larger one
 *   as the limit); default (or any value that is not a whole number)
 *   CHAMPION_CARD_STATUS_LIMIT. It is ignored when expanded is true.
 * @returns {ChampionCardModel} A new frozen model.
 * @example championCardModel(facts, {rowLimit: 2}).more
 * // 2 for a champion with four statuses: two rows, then "+2 more"
 */
export function championCardModel(facts, options = {}) {
  const title =
    facts.name === null
      ? "Champion"
      : facts.calling === null
        ? facts.name
        : `${facts.name} · ${facts.calling}`;
  const played =
    facts.you || facts.playedBy === null ? null : `Played by ${facts.playedBy}`;
  const base = {
    state: facts.state,
    title,
    idText: facts.displayId === null ? null : `Agent ID ${facts.displayId}`,
    teamId: facts.teamId,
    played,
    you: facts.you,
  };
  if (facts.state === "unseen") {
    return Object.freeze({
      ...base,
      unseenText:
        facts.unseenBy === null
          ? "none of your team can see it now."
          : `${facts.unseenBy} cannot see it now.`,
      life: null,
      ultimate: null,
      rows: Object.freeze([]),
      more: 0,
      total: 0,
      expanded: false,
      note: null,
      yourTarget: false,
    });
  }
  const life =
    facts.life === null ? null : lifeBarModel(facts.life.current, facts.life.maximum);
  const ultimate =
    facts.ultimate === null
      ? null
      : Object.freeze({
          label: facts.ultimate.name ?? "Ultimate",
          ready: facts.ultimate.remaining === 0,
          fraction:
            facts.ultimate.remaining === 0
              ? 1
              : facts.ultimate.full === null
                ? 0
                : Math.max(
                    0,
                    Math.min(1, 1 - facts.ultimate.remaining / facts.ultimate.full),
                  ),
          text:
            facts.ultimate.remaining === 0
              ? "Ready"
              : tickWords(facts.ultimate.remaining),
          classToken: classIdentityFromId(facts.classId)?.tokenId ?? null,
        });
  const sorted = [...facts.statuses].sort(
    (a, b) =>
      statusOrderIndex(a.id) - statusOrderIndex(b.id) ||
      (b.ticks ?? 0) - (a.ticks ?? 0),
  );
  const expanded = options.expanded === true;
  const limit = Number.isInteger(options.rowLimit)
    ? Math.max(
        1,
        Math.min(CHAMPION_CARD_STATUS_LIMIT, /** @type {number} */ (options.rowLimit)),
      )
    : CHAMPION_CARD_STATUS_LIMIT;
  const shown = expanded ? sorted : sorted.slice(0, limit);
  const rows = shown.map((status) => {
    const named = statusNameFor(status.id);
    const time = named?.time ?? "ticks";
    return Object.freeze({
      id: status.id,
      icon:
        statusSigilKey(named?.id ?? status.id) ??
        modifierSigilKey(named?.id ?? status.id),
      name: named?.name ?? "Recorded Status",
      effect: named?.effect ?? null,
      time: statusTimeText(status.id, status.ticks),
      line: statusCardSentence(status.id, status.amount, facts.classId).line,
      drain:
        time === "ticks" && status.ticks !== null && status.full !== null
          ? Math.max(0, Math.min(1, status.ticks / status.full))
          : null,
      paint: named?.paint ?? "neutral",
    });
  });
  return Object.freeze({
    ...base,
    unseenText: null,
    life,
    ultimate,
    rows: Object.freeze(rows),
    more: sorted.length - shown.length,
    total: sorted.length,
    expanded,
    note: facts.note,
    yourTarget: facts.yourTarget,
  });
}

/**
 * Return the words of a Champion Card's footer line under its rows, or null
 * when it has none: ["+3 more", "Hold Shift For All"] when rows were left out
 * (by the four-row limit, or by a short arena box's rowLimit), and ["All 7
 * statuses", "Release Shift To Shrink"] for the Hold Shift card when it shows
 * more than CHAMPION_CARD_STATUS_LIMIT rows (owner, 2 October 2026). A Hold
 * Shift card of four rows or fewer (a short arena box had cut the plain card)
 * has no footer, so it needs no extra height in that short box. Pure.
 *
 * @param {ChampionCardModel} model
 * @returns {readonly [string, string] | null}
 */
export function championCardFooter(model) {
  if (model.more > 0) return [`+${model.more} more`, "Hold Shift For All"];
  if (model.expanded && model.total > CHAMPION_CARD_STATUS_LIMIT) {
    return [`All ${model.total} statuses`, "Release Shift To Shrink"];
  }
  return null;
}

/**
 * Write a Champion Card model as plain text lines, for screen readers and for
 * DOM stand-ins without element creation: the title, the ID line, then
 * "Unseen · ..." or "Life 77 / 80", "Burst 30 ticks", one line per status
 * row ("Burst (Damage Boost) · 5 ticks left · Its attacks deal ×1.5 damage.
 * From its own Ultimate."), the footer ("+2 more · Hold Shift For All"), the
 * note, who plays it and "Your Target". Pure.
 *
 * @param {ChampionCardModel} model
 * @returns {string[]} A new array.
 */
export function championCardText(model) {
  const lines = [model.title];
  if (model.idText !== null) lines.push(model.idText);
  if (model.unseenText !== null) lines.push(`Unseen · ${model.unseenText}`);
  if (model.life !== null)
    lines.push(`Life ${model.life.text} / ${model.life.maximumText}`);
  if (model.ultimate !== null)
    lines.push(`${model.ultimate.label} ${model.ultimate.text}`);
  for (const row of model.rows) {
    const name = row.effect === null ? row.name : `${row.name} (${row.effect})`;
    lines.push([name, row.time, row.line].filter((part) => part !== "").join(" · "));
  }
  const footer = championCardFooter(model);
  if (footer !== null) lines.push(footer.join(" · "));
  if (model.note !== null) lines.push(model.note);
  if (model.you) lines.push("You play this champion");
  else if (model.played !== null) lines.push(model.played);
  if (model.yourTarget) lines.push("Your Target");
  return lines;
}

/**
 * Draw a Champion Card model into host: host's children are replaced by one
 * `div.champion-card` (data-state "seen" or "unseen", data-life "good",
 * "warn" or "low", data-expanded "true" on the Hold Shift card) holding the
 * title (`b`), the ID line with its 14 px team crest, the reduced card's
 * "Unseen" line, the bars (the shared Life bar of `life-bar.js`, then the
 * Ultimate bar: a label, a thin track whose fill width is the share, and the
 * value), the status rows (each `div.champion-card__row` with data-status and
 * data-paint: the status icon from `icons.js`, the name with its effect in a
 * dimmer `small`, the time on the right, the sentence line and the drain
 * line), the footer (`div.champion-card__more`, championCardFooter: "+3 more"
 * and "Hold Shift For All", or "All 7 statuses" and "Release Shift To
 * Shrink", with Shift drawn as a `kbd`), the note, "Played by ..."
 * (span.champion-card__played: one line, a long controller name cut with "…"
 * at the card's edge, the controller-name rule of controller-names.js) or
 * "You play this champion", and "Your Target". Text is set as text, never
 * HTML; widths are set through the CSS object model (the page's content
 * policy allows it). Every drawing is export-safe SVG. The look lives in
 * `styles/symbols-ui.css` and `styles/life-bar.css`, so the card looks the
 * same in the BattleClient, the DevClient combat screen and the Replay
 * Viewer. The wide look (data-columns "2" to "4" and "champion-card--wide")
 * is added by placeChampionCard when the Hold Shift card is taller than the
 * arena box; a plain card stays one column (renderFittedChampionCard draws it
 * with fewer rows instead). placeChampionCard also adds data-past-arena
 * "true" when the Hold Shift card grows past the arena box. Side effects:
 * replaces host's children only.
 *
 * @param {HTMLElement} host The element that holds the card (#hcard, or the
 *   tooltip card's details element).
 * @param {ChampionCardModel} model From championCardModel.
 * @returns {HTMLElement} The new card root.
 */
export function renderChampionCard(host, model) {
  const doc = host.ownerDocument;
  /**
   * @param {string} tag
   * @param {string} className
   * @param {string} [text]
   */
  const el = (tag, className, text) => {
    const node = doc.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  /** @param {string} className @param {number} share @param {string} [extra] */
  const track = (className, share, extra = "") => {
    const outer = el("span", `champion-card__track ${className}${extra}`);
    const fill = el("span", "champion-card__fill");
    fill.style.width = `${(share * 100).toFixed(1)}%`;
    outer.append(fill);
    return outer;
  };
  const root = el("div", "champion-card");
  root.dataset.state = model.state;
  if (model.life !== null) root.dataset.life = model.life.tone;
  if (model.expanded) root.dataset.expanded = "true";
  root.append(el("b", "champion-card__title", model.title));
  if (model.idText !== null) {
    const id = el("span", "champion-card__id");
    if (model.teamId !== null) id.append(hoverCardCrest(doc, model.teamId));
    id.append(model.idText);
    root.append(id);
  }
  if (model.unseenText !== null) {
    const unseen = el("span", "champion-card__unseen");
    unseen.append(el("b", "", "Unseen"), ` · ${model.unseenText}`);
    root.append(unseen);
  }
  if (model.life !== null || model.ultimate !== null) {
    const bars = el("div", "champion-card__bars");
    if (model.life !== null) bars.append(createLifeBar(doc, model.life));
    if (model.ultimate !== null) {
      const ready = model.ultimate.ready ? " is-ready" : "";
      const ultimateTrack = track(
        "champion-card__track--ultimate",
        model.ultimate.fraction,
        ready,
      );
      if (model.ultimate.classToken !== null) {
        ultimateTrack.dataset.class = model.ultimate.classToken;
      }
      bars.append(
        el("span", "champion-card__label", model.ultimate.label),
        ultimateTrack,
        el("span", `champion-card__value${ready}`, model.ultimate.text),
      );
    }
    root.append(bars);
  }
  if (model.rows.length > 0) {
    const rows = el("div", "champion-card__rows");
    for (const row of model.rows) {
      const node = el("div", "champion-card__row");
      node.dataset.status = row.id;
      node.dataset.paint = row.paint;
      const icon = el("span", "champion-card__icon");
      if (row.icon !== null) {
        icon.append(createSvgIcon(doc, row.icon, { size: 16 }));
      }
      const name = el("span", "champion-card__name", row.name);
      if (row.effect !== null) name.append(" ", el("small", "", `(${row.effect})`));
      node.append(
        icon,
        name,
        el("span", "champion-card__time", row.time),
        el("span", "champion-card__line", row.line),
      );
      if (row.drain !== null) {
        const drain = el("span", "champion-card__drain");
        const fill = el("span", "champion-card__fill");
        fill.style.width = `${(row.drain * 100).toFixed(1)}%`;
        drain.append(fill);
        node.append(drain);
      }
      rows.append(node);
    }
    root.append(rows);
  }
  const footer = championCardFooter(model);
  if (footer !== null) {
    const line = el("div", "champion-card__more");
    const [count, gesture] = footer;
    const hint = el("span", "champion-card__hint");
    // "Hold Shift For All" with Shift drawn as a key.
    const [before, after] = gesture.split("Shift");
    hint.append(before, el("kbd", "champion-card__key", "Shift"), after);
    line.append(el(model.more > 0 ? "b" : "span", "", count), hint);
    root.append(line);
  }
  if (model.note !== null) root.append(el("span", "champion-card__muted", model.note));
  if (model.you)
    root.append(el("span", "champion-card__you", "You play this champion"));
  else if (model.played !== null) {
    // champion-card__played: one line, a long controller name cut with "…"
    // at the card's edge (src/styles/controller-names.css).
    root.append(el("span", "champion-card__muted champion-card__played", model.played));
  }
  if (model.yourTarget) root.append(el("span", "champion-card__you", "Your Target"));
  host.replaceChildren(root);
  return root;
}

/**
 * Make the tooltip descriptor of a Champion Card, for a surface the shared
 * tooltip controller shows (the Replay Viewer's arena bodies and HUD frames).
 *
 * The descriptor has kind CHAMPION_CARD_KIND ("champion"), the card's title,
 * the class accent, the card's text lines (championCardText) as its summary
 * and rows, so a screen reader and a text-only surface read the same words.
 * The controller draws the card itself with renderChampionCard, so it looks
 * exactly like the live hover card. Creates a frozen descriptor and remembers
 * its model and facts in weak maps (the facts give the controller the Hold
 * Shift card); no DOM.
 *
 * @param {ChampionCardFacts} facts From championCardFactsFromAgent, or the
 *   reduced unseen facts.
 * @param {string} [id] The descriptor ID; default "champion:<Agent ID>".
 * @returns {TooltipDescriptor}
 */
export function createChampionCardDescriptor(facts, id) {
  const model = championCardModel(facts);
  const lines = championCardText(model);
  const metadata = { compact: true, full: true };
  const accent = classIdentityFromId(facts.classId)?.tokenId ?? "none";
  const descriptor = /** @type {TooltipDescriptor} */ (
    createSemanticDescriptor({
      kind: CHAMPION_CARD_KIND,
      id: id ?? `champion:${facts.displayId ?? model.title}`,
      title: model.title,
      tone: "information",
      accent,
      summary: lines.slice(1).join(" · ") || model.title,
      rows: [],
      sections: [],
      metadata,
      anchor: "element",
    })
  );
  championCardByDescriptor.set(descriptor, model);
  championFactsByDescriptor.set(descriptor, facts);
  return descriptor;
}

/**
 * Return the Champion Card model a descriptor carries (made by
 * createChampionCardDescriptor), or null for any other descriptor. Pure.
 *
 * @param {unknown} descriptor
 * @returns {ChampionCardModel | null}
 */
export function championCardOf(descriptor) {
  return isRecord(descriptor)
    ? (championCardByDescriptor.get(descriptor) ?? null)
    : null;
}

/**
 * Draw the 14 px team crest of the card's ID line: a 5 px ring (stroke 3) in
 * the team color with a faint 2.25 px dot, and for Team B a 1 px dark keyline
 * on the ring. Export-safe SVG. Pure apart from creating nodes.
 *
 * The team color comes from CSS, not from this file: the ring carries
 * `crest-ring crest-ring--a` (or `--b`) and the dot `crest-dot crest-dot--a`
 * (or `--b`), and styles/symbols-ui.css paints them with the palette tokens
 * `--team-a` and `--team-b`.
 *
 * @param {Document} doc Creates the nodes.
 * @param {1 | 2} teamId 1 for Team A, 2 for Team B.
 * @returns {SVGSVGElement}
 */
function hoverCardCrest(doc, teamId) {
  const ns = "http://www.w3.org/2000/svg";
  const team = teamId === 1 ? "a" : "b";
  const svg = /** @type {SVGSVGElement} */ (doc.createElementNS(ns, "svg"));
  for (const [name, value] of [
    ["class", "crest"],
    ["width", "14"],
    ["height", "14"],
    ["viewBox", "0 0 14 14"],
    ["aria-hidden", "true"],
  ]) {
    svg.setAttribute(name, value);
  }
  /** @param {Record<string, string>} attributes */
  const circle = (attributes) => {
    const node = doc.createElementNS(ns, "circle");
    for (const [name, value] of Object.entries({ cx: "7", cy: "7", ...attributes })) {
      node.setAttribute(name, value);
    }
    svg.append(node);
  };
  circle({
    r: "5",
    class: `crest-ring crest-ring--${team}`,
    fill: "none",
    "stroke-width": "3",
  });
  if (teamId === 2) {
    circle({ r: "5", fill: "none", stroke: "#0A0D14", "stroke-width": "1" });
  }
  circle({ r: "2.25", class: `crest-dot crest-dot--${team}`, opacity: "0.35" });
  return svg;
}

/**
 * Place a Champion Card beside the pointer and keep it inside the arena box
 * (owner, 2 October 2026: the plain card never leaves the arena box; a Hold
 * Shift card that grows past the box is placed by
 * championCardPastArenaPosition instead): 16 px right
 * of and below the pointer; flipped to the left (card width + 16 px back) when
 * it would pass the box's right edge (8 px margin); flipped above (card
 * height + 12 px) when it would pass the bottom edge (8 px margin); then kept
 * at least CHAMPION_CARD_BOX_MARGIN (8 px) inside every edge of the box. A
 * pointer outside the box (a scorebug portrait or a docked frame) works the
 * same way, so the card lands at the box edge nearest it. When the card would
 * cover a rectangle in avoid, the other three corners around the pointer are
 * tried in turn (left below, right above, left above) and the first that
 * covers none is used; when every corner covers one, the corner that covers
 * the least wins. A card larger than the box sits at the box's top left
 * margin. All numbers are CSS pixels relative to the box. Pure.
 *
 * @param {{x: number, y: number}} pointer The pointer, relative to the box.
 * @param {{width: number, height: number}} box The arena box.
 * @param {{width: number, height: number}} size The card's size.
 * @param {ReadonlyArray<{left: number, top: number, right: number, bottom: number}>} [avoid]
 *   Rectangles, relative to the box, the card must not cover. Default none.
 * @returns {Readonly<{left: number, top: number}>}
 */
export function hoverCardPosition(pointer, box, size, avoid = []) {
  const margin = CHAMPION_CARD_BOX_MARGIN;
  /** @param {number} x @param {number} y */
  const keep = (x, y) => ({
    left: Math.max(margin, Math.min(x, box.width - size.width - margin)),
    top: Math.max(margin, Math.min(y, box.height - size.height - margin)),
  });
  const right = pointer.x + 16;
  const left = pointer.x - size.width - 16;
  const below = pointer.y + 16;
  const above = pointer.y - size.height - 12;
  const first = keep(
    right + size.width + 8 > box.width ? left : right,
    below + size.height + 8 > box.height ? above : below,
  );
  /** @param {{left: number, top: number}} place */
  const covered = (place) =>
    avoid.reduce((total, rect) => {
      const width =
        Math.min(place.left + size.width, rect.right) - Math.max(place.left, rect.left);
      const height =
        Math.min(place.top + size.height, rect.bottom) - Math.max(place.top, rect.top);
      return total + (width > 0 && height > 0 ? width * height : 0);
    }, 0);
  let best = first;
  let bestCover = covered(first);
  for (const place of [keep(left, below), keep(right, above), keep(left, above)]) {
    if (bestCover === 0) break;
    const cover = covered(place);
    if (cover < bestCover) {
      best = place;
      bestCover = cover;
    }
  }
  return Object.freeze({ left: best.left, top: best.top });
}

/** The width of one Champion Card column, in CSS pixels (styles/symbols-ui.css). */
export const CHAMPION_CARD_COLUMN_WIDTH = 262;

/** The gap between two Champion Card columns, in CSS pixels. */
export const CHAMPION_CARD_COLUMN_GAP = 16;

/** The most columns a Champion Card's rows spread into. */
export const CHAMPION_CARD_MAX_COLUMNS = 4;

/**
 * Decide how many status rows a plain Champion Card (not the Hold Shift card)
 * shows, so the whole card fits the height the arena box gives it (owner, 2
 * October 2026: "Fewer rows"; a plain card is always one column). Pure: the
 * caller measures, this only decides.
 *
 * All the rows fit: they are all shown. Otherwise the card shows the most
 * leading rows (rows keep their order: stuns, then slows, then the rest) for
 * which frame, plus those rows, plus footer (the "+N more · Hold Shift For
 * All" line that appears once a row is left out) is no taller than
 * available. It always shows at least one row, even when one row does not
 * fit; placeChampionCard then scales that card down so it stays inside the
 * box.
 *
 * @param {{
 *   available: number,
 *   frame: number,
 *   footer: number,
 *   rows: ReadonlyArray<number>,
 *   more?: number,
 * }} input Every height is in CSS pixels. available: the height the card may
 *   take (the arena box's height less two CHAMPION_CARD_BOX_MARGIN). frame:
 *   the measured height of everything on the card but its status rows and
 *   its footer (the title, ID line, Life and Ultimate bars, the rule and
 *   padding above the rows, the note, "Played by" and "Your Target"). footer:
 *   the measured height the footer line adds when it shows. rows: the
 *   measured height each row adds, in card order (a row's own height plus the
 *   gap above it; the first row has no gap), one entry per row the card could
 *   show (at most CHAMPION_CARD_STATUS_LIMIT). more: how many rows are already
 *   left out beyond rows (the four-row limit's "+N more"), a whole number;
 *   default 0. When it is above 0 the footer always shows.
 * @returns {number} How many leading rows to show: rows.length when they all
 *   fit, else the most that fit with the footer, never less than 1; 0 when
 *   rows is empty (the reduced "Unseen" card, or a champion with no status).
 * @example championCardRowsThatFit({available: 274, frame: 120, footer: 29, rows: [50, 57, 57, 57]})
 * // 2: 120 + 50 + 57 + 29 = 256 fits, and a third row would need 313
 */
export function championCardRowsThatFit(input) {
  const { available, frame, footer, rows } = input;
  if (rows.length === 0) return 0;
  const all = rows.reduce((sum, height) => sum + height, 0);
  if (frame + all + ((input.more ?? 0) > 0 ? footer : 0) <= available) {
    return rows.length;
  }
  let used = frame + footer;
  let shown = 0;
  // At least one row is left out here, so at most rows.length - 1 show.
  for (const height of rows.slice(0, -1)) {
    if (used + height > available) break;
    used += height;
    shown += 1;
  }
  return Math.max(1, shown);
}

/**
 * Draw a plain Champion Card (not the Hold Shift card) into host with as many
 * status rows as fit the arena box, the one row-fitting rule of every client
 * (owner, 2 October 2026: "Fewer rows"). The live hover card (#hcard,
 * createHoverCard) and the Replay Viewer's tooltip card (the tooltip
 * controller) both call it, each time they draw or refresh the card (a new
 * frame, a view change, a resize), and then place the card with
 * placeChampionCard.
 *
 * Steps: draw championCardModel(facts) (at most CHAMPION_CARD_STATUS_LIMIT
 * rows, one column) and measure surface. When the card fits the box's height
 * less two CHAMPION_CARD_BOX_MARGIN, or it has one row or none, keep it.
 * Otherwise measure each row, the footer line ("+N more · Hold Shift For
 * All"; when the card has none yet, it is drawn once with one row fewer to
 * measure it) and the rest of the card, ask championCardRowsThatFit how many
 * rows fit, and draw the card again with that rowLimit. The header, Life and
 * Ultimate bars, note, "Played by" and "Your Target" always show; rows keep
 * their order. Every number is in client (viewport) CSS pixels.
 *
 * Side effects: replaces host's children (renderChampionCard, up to three
 * times) and reads layout, so surface must be shown (it may be invisible)
 * and the card must be one column (a fresh drawing is).
 *
 * @param {{
 *   host: HTMLElement,
 *   surface: HTMLElement,
 *   facts: ChampionCardFacts,
 *   box: {height: number},
 * }} input host holds the card (#hcard, or the tooltip card's details
 *   element); surface is the element whose height is the card's (#hcard, or
 *   the tooltip card itself); facts are the champion's facts; box is the
 *   arena box (only its height is read).
 * @returns {Readonly<{root: HTMLElement, model: ChampionCardModel}>} The drawn
 *   `.champion-card` root and the model it shows (model.rows.length rows,
 *   model.more left out).
 */
export function renderFittedChampionCard(input) {
  const { host, surface, facts } = input;
  const available = input.box.height - 2 * CHAMPION_CARD_BOX_MARGIN;
  /** @param {number} [rowLimit] */
  const draw = (rowLimit) => {
    const model = championCardModel(facts, rowLimit === undefined ? {} : { rowLimit });
    return Object.freeze({ root: renderChampionCard(host, model), model });
  };
  /**
   * The height the footer line adds below the rows: from the bottom of the
   * rows block to the bottom of the footer (the gap and margin above it
   * included). 0 when the card has no footer or no rows.
   *
   * @param {HTMLElement} root
   */
  const footerHeight = (root) => {
    const line = root.querySelector(".champion-card__more");
    const block = root.querySelector(".champion-card__rows");
    return line === null || block === null
      ? 0
      : line.getBoundingClientRect().bottom - block.getBoundingClientRect().bottom;
  };
  const first = draw();
  const count = first.model.rows.length;
  const height = surface.getBoundingClientRect().height;
  if (count <= 1 || height <= available) return first;
  const rects = [...first.root.querySelectorAll(".champion-card__row")].map((row) =>
    row.getBoundingClientRect(),
  );
  const rows = rects.map(
    (rect, index) => rect.bottom - (index === 0 ? rect.top : rects[index - 1].bottom),
  );
  const span = rects[rects.length - 1].bottom - rects[0].top;
  let frame = height - span;
  let footer;
  let drawn = first;
  if (first.model.more > 0) {
    footer = footerHeight(first.root);
    frame -= footer;
  } else {
    // No footer yet: draw one row fewer once, which adds it, to measure it.
    drawn = draw(count - 1);
    footer = footerHeight(drawn.root);
  }
  const shown = championCardRowsThatFit({
    available,
    frame,
    footer,
    rows,
    more: first.model.more,
  });
  return shown === drawn.model.rows.length ? drawn : draw(shown);
}

/**
 * The smallest scale the Hold Shift card may be drawn at inside the arena box
 * ("Grow past arena", owner, 2 October 2026). When its best layout inside the
 * box needs a smaller scale, it grows past the box instead, at its normal size
 * (championCardLayout).
 */
export const CHAMPION_CARD_MIN_INSIDE_SCALE = 0.85;

/**
 * Return how many Champion Card columns fit a width: as many columns of
 * CHAMPION_CARD_COLUMN_WIDTH, CHAMPION_CARD_COLUMN_GAP apart, as fit the width
 * less two CHAMPION_CARD_BOX_MARGIN; at least 1 and at most
 * CHAMPION_CARD_MAX_COLUMNS. Pure.
 *
 * @param {number} width The width in CSS pixels (the arena box's or the
 *   viewport's).
 * @returns {number} A whole number from 1 to CHAMPION_CARD_MAX_COLUMNS.
 */
function championCardColumnsWithin(width) {
  return Math.max(
    1,
    Math.min(
      CHAMPION_CARD_MAX_COLUMNS,
      Math.floor(
        (width - 2 * CHAMPION_CARD_BOX_MARGIN + CHAMPION_CARD_COLUMN_GAP) /
          (CHAMPION_CARD_COLUMN_WIDTH + CHAMPION_CARD_COLUMN_GAP),
      ),
    ),
  );
}

/**
 * Decide how a drawn Champion Card is laid out: how many columns its rows
 * spread into, the scale it is drawn at, and whether it grows past the arena
 * box. The one layout rule of every client (the BattleClient, the DevClient
 * combat screen and the Replay Viewer): placeChampionCard measures the card,
 * this only decides. Pure.
 *
 * Inside the box (the plain card always; the Hold Shift card unless it grows
 * past the box): "room" is the box's height less two CHAMPION_CARD_BOX_MARGIN.
 * A plain card (expanded false) stays one column. A Hold Shift card (expanded
 * true) with status rows that is taller than the room in one column tries
 * two, then three, then four columns, only as many as fit the box's width,
 * and stops at the first count no taller than the room. A count is kept only
 * when it makes the card more than 0.5 px shorter than the best count so far,
 * so the card never widens for nothing (four rows take two lines in two or
 * three columns, so two are kept). A card with rows that is still taller than
 * the room gets scale room / height; otherwise its scale is 1. A card with no
 * rows (the reduced "Unseen" card) is never scaled.
 *
 * Past the box ("Grow past arena", owner, 2 October 2026): only a Hold Shift
 * card with status rows, only when viewport is not null, and only when the
 * inside layout above needs a scale below CHAMPION_CARD_MIN_INSIDE_SCALE
 * (0.85). Then it takes the column count that makes it shortest among the
 * counts that fit the viewport's width (fewest columns on a tie, with the same
 * 0.5 px rule), at scale 1. When even the viewport less two
 * CHAMPION_CARD_BOX_MARGIN cannot hold that card, its scale is just small
 * enough to fit it (height and width). An inside scale of 0.85 or more
 * changes nothing: the card stays inside the box.
 *
 * @param {{
 *   box: {width: number, height: number},
 *   viewport: {width: number, height: number} | null,
 *   sizes: ReadonlyArray<{width: number, height: number}>,
 *   expanded: boolean,
 *   hasRows: boolean,
 * }} input Every size is in CSS pixels. box: the arena box. viewport: the
 *   window the card is shown in, or null when it is not known (the card then
 *   never grows past the box). sizes: the card's measured size at scale 1 in
 *   one column (sizes[0]), two columns (sizes[1]) and so on; at least one
 *   entry; counts past its end are never chosen. expanded: true for the Hold
 *   Shift card (every status). hasRows: whether the card shows any status row.
 * @returns {Readonly<{columns: number, scale: number, pastArena: boolean}>}
 *   columns: the column count to draw (1 for every plain card). scale: 1, or
 *   the factor the card is shrunk by (above 0 and below 1). pastArena: true
 *   when the card grows past the arena box and is placed in the viewport
 *   (championCardPastArenaPosition), false when it stays inside the box
 *   (hoverCardPosition).
 * @example championCardLayout({box: {width: 966, height: 227}, viewport: {width: 1280, height: 720}, sizes: [{width: 262, height: 600}, {width: 540, height: 400}, {width: 818, height: 330}, {width: 1096, height: 270}], expanded: true, hasRows: true})
 * // {columns: 4, scale: 1, pastArena: true}: inside, three columns would
 * // need scale 211 / 330 = 0.64
 */
export function championCardLayout(input) {
  const { box, viewport, sizes, expanded, hasRows } = input;
  const margin = CHAMPION_CARD_BOX_MARGIN;
  /**
   * The best column count among the first `limit` sizes: a count wins only
   * when it is more than 0.5 px shorter than the best so far, and the search
   * stops at the first count no taller than room.
   *
   * @param {number} limit
   * @param {number} room
   */
  const shortest = (limit, room) => {
    let columns = 1;
    for (let count = 2; count <= Math.min(limit, sizes.length); count += 1) {
      const height = sizes[count - 1].height;
      if (height < sizes[columns - 1].height - 0.5) columns = count;
      if (height <= room) break;
    }
    return columns;
  };
  const room = box.height - 2 * margin;
  const columns =
    expanded && hasRows && sizes[0].height > room
      ? shortest(championCardColumnsWithin(box.width), room)
      : 1;
  const height = sizes[columns - 1].height;
  const scale = hasRows && height > room && room > 0 ? room / height : 1;
  if (
    !expanded ||
    !hasRows ||
    viewport === null ||
    scale >= CHAMPION_CARD_MIN_INSIDE_SCALE
  ) {
    return Object.freeze({ columns, scale, pastArena: false });
  }
  // Grow past the arena: the shortest card the viewport's width allows.
  const wide = shortest(championCardColumnsWithin(viewport.width), -Infinity);
  const size = sizes[wide - 1];
  const fit = Math.min(
    1,
    (viewport.height - 2 * margin) / size.height,
    (viewport.width - 2 * margin) / size.width,
  );
  return Object.freeze({ columns: wide, scale: fit > 0 ? fit : 1, pastArena: true });
}

/**
 * Place a Champion Card that grows past the arena box ("Grow past arena",
 * owner, 2 October 2026): fully inside the viewport (CHAMPION_CARD_BOX_MARGIN,
 * 8 px, inside every edge), over the panels, as close to its champion as it
 * can be and never over the champion when that can be avoided. Pure.
 *
 * It tries four places around the champion's rectangle (subject): below it
 * (12 px gap), above it (12 px), right of it (16 px) and left of it (16 px).
 * Below and above line up the card's left edge with the champion's left
 * edge, or its right edge with the champion's right edge when the card would
 * pass the viewport's right edge; right and left line up the card's top edge
 * with the champion's top edge, or its bottom edge with the champion's bottom
 * edge when the card would pass the bottom edge. Each place is then kept
 * inside the viewport, so a place pushed back over the champion covers it.
 * The winner covers the least of the champion (a place that covers the
 * pointer counts as covering it), then the least of the avoid rectangles,
 * then sits nearest the champion (the gap between the two rectangles), then
 * covers the most of the arena box (it stays near the arena); differences of
 * 0.5 px (or 0.5 square px) or less count as ties, and a full tie keeps the
 * order above. Every number is in client (viewport) CSS pixels.
 *
 * @param {{
 *   pointer: {x: number, y: number},
 *   subject: {left: number, top: number, right: number, bottom: number},
 *   box: {left: number, top: number, width: number, height: number},
 *   viewport: {width: number, height: number},
 *   size: {width: number, height: number},
 *   avoid?: ReadonlyArray<{left: number, top: number, right: number, bottom: number}>,
 * }} input pointer: the pointer (or a keyboard focus's anchor point).
 *   subject: the hovered champion element's rectangle (its body, portrait,
 *   row, button or docked frame). box: the arena box. viewport: the window,
 *   with its top left corner at 0, 0. size: the card's drawn size (after its
 *   scale). avoid: rectangles the card should not cover (default none).
 * @returns {Readonly<{left: number, top: number}>} The card's top left corner.
 */
export function championCardPastArenaPosition(input) {
  const { pointer, subject, box, viewport, size } = input;
  const margin = CHAMPION_CARD_BOX_MARGIN;
  /** @param {number} start @param {number} length @param {number} limit */
  const keep = (start, length, limit) =>
    Math.max(margin, Math.min(start, limit - length - margin));
  // Lined up with the champion's edges, so a place kept inside the window by
  // moving it back over the champion covers it (and loses).
  const across = keep(
    subject.left + size.width + margin > viewport.width
      ? subject.right - size.width
      : subject.left,
    size.width,
    viewport.width,
  );
  const down = keep(
    subject.top + size.height + margin > viewport.height
      ? subject.bottom - size.height
      : subject.top,
    size.height,
    viewport.height,
  );
  const places = [
    { left: across, top: keep(subject.bottom + 12, size.height, viewport.height) },
    {
      left: across,
      top: keep(subject.top - 12 - size.height, size.height, viewport.height),
    },
    { left: keep(subject.right + 16, size.width, viewport.width), top: down },
    {
      left: keep(subject.left - 16 - size.width, size.width, viewport.width),
      top: down,
    },
  ];
  /**
   * @param {{left: number, top: number}} place
   * @param {{left: number, top: number, right: number, bottom: number}} rect
   */
  const overlap = (place, rect) =>
    Math.max(
      0,
      Math.min(place.left + size.width, rect.right) - Math.max(place.left, rect.left),
    ) *
    Math.max(
      0,
      Math.min(place.top + size.height, rect.bottom) - Math.max(place.top, rect.top),
    );
  const arena = {
    left: box.left,
    top: box.top,
    right: box.left + box.width,
    bottom: box.top + box.height,
  };
  /** @param {{left: number, top: number}} place @returns {number[]} */
  const score = (place) => {
    const onPointer =
      pointer.x >= place.left &&
      pointer.x <= place.left + size.width &&
      pointer.y >= place.top &&
      pointer.y <= place.top + size.height;
    const dx = Math.max(
      0,
      subject.left - (place.left + size.width),
      place.left - subject.right,
    );
    const dy = Math.max(
      0,
      subject.top - (place.top + size.height),
      place.top - subject.bottom,
    );
    return [
      overlap(place, subject) + (onPointer ? 1 : 0),
      (input.avoid ?? []).reduce((total, rect) => total + overlap(place, rect), 0),
      Math.hypot(dx, dy),
      -overlap(place, arena),
    ];
  };
  let best = places[0];
  let bestScore = score(best);
  for (const place of places.slice(1)) {
    const next = score(place);
    const index = next.findIndex((value, at) => Math.abs(value - bestScore[at]) > 0.5);
    if (index !== -1 && next[index] < bestScore[index]) {
      best = place;
      bestScore = next;
    }
  }
  return Object.freeze({ left: best.left, top: best.top });
}

/**
 * Return the size of the window an element is shown in, in CSS pixels: the
 * document element's client size (the window less its scroll bars), else the
 * window's inner size. Reads layout only.
 *
 * @param {HTMLElement} element Any element in the page.
 * @returns {{width: number, height: number} | null} null when neither has an
 *   area (a DOM stand-in with no window).
 */
function viewportOf(element) {
  const doc = element.ownerDocument;
  const width = doc?.documentElement?.clientWidth || doc?.defaultView?.innerWidth || 0;
  const height =
    doc?.documentElement?.clientHeight || doc?.defaultView?.innerHeight || 0;
  return width > 0 && height > 0 ? { width, height } : null;
}

/**
 * Fit a drawn Champion Card and return where it goes, the one placement rule
 * of every client: the live hover card (#hcard) and the Replay Viewer's
 * tooltip card both call it.
 *
 * The plain card (root without data-expanded "true") always stays fully
 * inside the arena box, one column wide (owner, 2 October 2026: "Fewer rows";
 * renderFittedChampionCard has already left out the rows that do not fit).
 * The Hold Shift card (data-expanded "true") stays inside the box too, spread
 * into more columns when that makes it fit, unless the best layout inside
 * would need a scale below CHAMPION_CARD_MIN_INSIDE_SCALE (0.85): then it
 * grows past the box ("Grow past arena", owner, 2 October 2026) at its normal
 * size, in the column count that makes it shortest, over the panels and fully
 * inside the viewport. championCardLayout owns these rules.
 *
 * Steps: draw the card in one column and measure surface. For a Hold Shift
 * card with status rows that is taller than the box less two
 * CHAMPION_CARD_BOX_MARGIN, also measure it in two, three and four columns
 * (as many as the box's or the viewport's width allows). Ask
 * championCardLayout for the layout, set it (the root gets data-columns "2",
 * "3" or "4" and the class "champion-card--wide" for more than one column; a
 * CSS scale from its top left corner when it is shrunk; data-past-arena
 * "true" when it grows past the box), then place it: inside the box with
 * hoverCardPosition, or past the box with championCardPastArenaPosition. A
 * card placed again first drops all of these, so a card that now fits goes
 * back to one column, no scale and inside the box. Every number is in client
 * (viewport) CSS pixels. Side effects: the root's data-columns,
 * data-past-arena, "champion-card--wide" class and transform; reads layout
 * (the surface must be shown, even if invisible).
 *
 * @param {{
 *   surface: HTMLElement,
 *   root: HTMLElement,
 *   box: {left: number, top: number, width: number, height: number},
 *   pointer: {x: number, y: number},
 *   avoid?: ReadonlyArray<{left: number, top: number, right: number, bottom: number}>,
 *   subject?: {left: number, top: number, right: number, bottom: number},
 *   viewport?: {width: number, height: number} | null,
 * }} input surface is the element that is moved (#hcard, or the tooltip
 *   card); root is the `.champion-card` inside it (renderChampionCard's
 *   result); box is the arena box; pointer is the pointer, or the hovered
 *   element's anchor point for a keyboard focus; avoid lists rectangles the
 *   card must not cover (default none); subject is the hovered champion
 *   element's rectangle, which a card past the box does not cover when it can
 *   avoid it (default: the pointer's point); viewport is the window's size
 *   (default: read from surface's document; null, or no window, means the
 *   card never grows past the box).
 * @returns {Readonly<{left: number, top: number, wide: boolean, columns: number, scale: number, pastArena: boolean}>}
 *   The card's top left corner in client pixels; wide is true for two
 *   columns or more (only the Hold Shift card); columns is the column count;
 *   scale is 1, or the factor it was shrunk by; pastArena is true when it
 *   grew past the arena box (only the Hold Shift card).
 */
export function placeChampionCard(input) {
  const { surface, root, box, pointer } = input;
  const expanded = root.dataset.expanded === "true";
  const hasRows = root.querySelector(".champion-card__row") !== null;
  const viewport = input.viewport === undefined ? viewportOf(surface) : input.viewport;
  /** @param {number} count */
  const setColumns = (count) => {
    if (count > 1) {
      root.dataset.columns = String(count);
      root.classList.add("champion-card--wide");
    } else {
      delete root.dataset.columns;
      root.classList.remove("champion-card--wide");
    }
  };
  root.style.transform = "";
  delete root.dataset.pastArena;
  setColumns(1);
  /** @type {Array<{width: number, height: number}>} */
  const sizes = [surface.getBoundingClientRect()];
  if (
    expanded &&
    hasRows &&
    sizes[0].height > box.height - 2 * CHAMPION_CARD_BOX_MARGIN
  ) {
    const most = Math.max(
      championCardColumnsWithin(box.width),
      viewport === null ? 1 : championCardColumnsWithin(viewport.width),
    );
    for (let count = 2; count <= most; count += 1) {
      setColumns(count);
      sizes.push(surface.getBoundingClientRect());
    }
  }
  const { columns, scale, pastArena } = championCardLayout({
    box,
    viewport,
    sizes,
    expanded,
    hasRows,
  });
  setColumns(columns);
  if (scale < 1) {
    root.style.transformOrigin = "0 0";
    root.style.transform = `scale(${scale.toFixed(4)})`;
  }
  const size = {
    width: sizes[columns - 1].width * scale,
    height: sizes[columns - 1].height * scale,
  };
  const avoid = input.avoid ?? [];
  if (pastArena && viewport !== null) {
    root.dataset.pastArena = "true";
    const place = championCardPastArenaPosition({
      pointer,
      subject: input.subject ?? {
        left: pointer.x,
        top: pointer.y,
        right: pointer.x,
        bottom: pointer.y,
      },
      box,
      viewport,
      size,
      avoid,
    });
    return Object.freeze({ ...place, wide: columns > 1, columns, scale, pastArena });
  }
  const place = hoverCardPosition(
    { x: pointer.x - box.left, y: pointer.y - box.top },
    { width: box.width, height: box.height },
    size,
    avoid.map((rect) => ({
      left: rect.left - box.left,
      top: rect.top - box.top,
      right: rect.right - box.left,
      bottom: rect.bottom - box.top,
    })),
  );
  return Object.freeze({
    left: place.left + box.left,
    top: place.top + box.top,
    wide: columns > 1,
    columns,
    scale,
    pastArena: false,
  });
}

/**
 * Return the client rectangle of the first element the getter gives that has
 * an area, or null when the getter is missing, gives nothing or gives a
 * hidden element. Reads layout only.
 *
 * @param {(() => Element | null) | undefined} getter
 * @returns {{left: number, top: number, width: number, height: number} | null}
 */
function areaOf(getter) {
  const element = getter?.() ?? null;
  if (element === null || typeof element.getBoundingClientRect !== "function") {
    return null;
  }
  const rect = element.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0
    ? { left: rect.left, top: rect.top, width: rect.width, height: rect.height }
    : null;
}

/**
 * Watch the window for the Champion Card's Hold Shift gesture (owner, 2
 * October 2026) and report when the card should show every status.
 *
 * The card expands when Shift is pressed by itself (no Ctrl, Alt or Meta,
 * not a key repeat) while canStart() is true (a card with left-out rows is
 * showing for a resting pointer), stays down for options.delayMs (default
 * CHAMPION_CARD_HOLD_MS, 300 ms) and canStart() is still true then: onExpand
 * runs. It shrinks back at once (onShrink, only when it had expanded, and
 * the wait is dropped either way) on: releasing Shift, any other key (so
 * Shift+Tab, Shift+arrows, "?" and every other Shift combination keep their
 * own jobs), any pointer press, any pointer movement (a pointermove at new
 * coordinates; a pointermove at the same spot, which browsers send after a
 * layout change, is not movement), the wheel, and the window losing focus.
 * Shift held before the pointer came to rest never expands the card, so an
 * aim with Shift held never opens it. It never cancels, stops or changes an
 * event, and it sends nothing anywhere.
 *
 * Listeners are added to target in the capture phase, so they run before the
 * page's own key and pointer handlers.
 *
 * @param {{
 *   target: EventTarget,
 *   canStart: () => boolean,
 *   onExpand: () => void,
 *   onShrink: () => void,
 *   delayMs?: number,
 *   setTimer?: (callback: () => void, ms: number) => unknown,
 *   clearTimer?: (timer: unknown) => void,
 * }} options target is the window; canStart says whether a card that can
 *   expand is showing for a resting pointer; onExpand and onShrink redraw the
 *   card; setTimer and clearTimer default to setTimeout and clearTimeout
 *   (tests pass their own clock).
 * @returns {Readonly<{expanded: () => boolean, cancel: () => void, destroy: () => void}>}
 *   expanded says whether the card should show every status now; cancel
 *   drops the wait and the expansion without calling onShrink (for a card
 *   that has just been hidden); destroy removes the listeners and cancels.
 */
export function createShiftHold(options) {
  const { target, canStart, onExpand, onShrink } = options;
  const delay = options.delayMs ?? CHAMPION_CARD_HOLD_MS;
  const setTimer =
    options.setTimer ??
    ((/** @type {() => void} */ callback, /** @type {number} */ ms) =>
      setTimeout(callback, ms));
  const clearTimer =
    options.clearTimer ??
    ((/** @type {unknown} */ timer) =>
      clearTimeout(/** @type {ReturnType<typeof setTimeout>} */ (timer)));
  /** @type {unknown} */
  let timer = null;
  let expanded = false;
  /** @type {{x: number, y: number} | null} */
  let lastPoint = null;

  /** Drop the wait and the expansion; quiet skips onShrink. @param {boolean} quiet */
  const stop = (quiet) => {
    if (timer !== null) {
      clearTimer(timer);
      timer = null;
    }
    if (expanded) {
      expanded = false;
      if (!quiet) onShrink();
    }
  };
  /** @param {Event} event */
  const onKeyDown = (event) => {
    const key = /** @type {KeyboardEvent} */ (event);
    if (key.key === "Shift" && !key.ctrlKey && !key.altKey && !key.metaKey) {
      if (key.repeat || timer !== null || expanded || !canStart()) return;
      timer = setTimer(() => {
        timer = null;
        if (canStart()) {
          expanded = true;
          onExpand();
        }
      }, delay);
      return;
    }
    stop(false);
  };
  /** @param {Event} event */
  const onKeyUp = (event) => {
    if (/** @type {KeyboardEvent} */ (event).key === "Shift") stop(false);
  };
  /** @param {Event} event */
  const onPointerMove = (event) => {
    const pointer = /** @type {PointerEvent} */ (event);
    const point = { x: pointer.clientX, y: pointer.clientY };
    const moved =
      lastPoint === null || lastPoint.x !== point.x || lastPoint.y !== point.y;
    lastPoint = point;
    if (moved) stop(false);
  };
  const onOther = () => stop(false);
  // Only the window's own blur: an element's blur reaches a capture listener
  // on the window too, and a focus change alone is not a reason to shrink.
  /** @param {Event} event */
  const onBlur = (event) => {
    if (event.target === target) stop(false);
  };
  /** @type {ReadonlyArray<[string, (event: Event) => void]>} */
  const listeners = [
    ["keydown", onKeyDown],
    ["keyup", onKeyUp],
    ["pointermove", onPointerMove],
    ["pointerdown", onOther],
    ["wheel", onOther],
    ["blur", onBlur],
  ];
  for (const [type, listener] of listeners) {
    target.addEventListener(type, listener, { capture: true });
  }
  return Object.freeze({
    expanded: () => expanded,
    cancel: () => stop(true),
    destroy: () => {
      for (const [type, listener] of listeners) {
        target.removeEventListener(type, listener, { capture: true });
      }
      stop(true);
    },
  });
}

/**
 * Read the champion an element names, for the hover card. Pure (reads
 * attributes only).
 *
 * @param {Element} element An element that matches HOVER_CARD_SELECTOR.
 * @returns {HoverCardSubject}
 */
export function hoverCardSubject(element) {
  /** @param {string} name @returns {string | null} */
  const read = (name) => {
    const value = element.getAttribute(name);
    return value === null || value === "" ? null : value;
  };
  const slotText = read("data-slot") ?? read("data-target");
  const slot = slotText !== null && /^\d$/u.test(slotText) ? Number(slotText) : null;
  const classText = read("data-class-id");
  return Object.freeze({
    presentationKey: read("data-presentation-key"),
    slot,
    publicAgentId: read("data-public-agent-id"),
    classId:
      classText !== null && /^[1-5]$/u.test(classText) ? Number(classText) : null,
  });
}

/**
 * Bind the live hover card to one page: while the pointer is over (or the
 * keyboard focus is on) an element matching HOVER_CARD_SELECTOR inside root,
 * the card shows that champion's Champion Card (championCardModel, drawn by
 * renderChampionCard) and follows the pointer; it hides when the pointer
 * leaves those elements or root, when the element's champion has no facts
 * (factsFor returns null, for example an unseen arena body that is no longer
 * drawn), and on Escape.
 *
 * Placement (owner, 2 October 2026): the plain card is always placed fully
 * inside the arena box that box() returns (#arena), never over the deck,
 * rail, Lock In, scorebug or top bar. The plain card is always one column:
 * renderFittedChampionCard draws it with as many status rows as fit the box
 * (at least one, then "+N more · Hold Shift For All"), every time it is drawn
 * or refreshed. placeChampionCard places it beside the pointer (or, for a
 * keyboard focus, beside the focused element's lower left corner), spreads
 * only the Hold Shift card into more columns, and keeps it 8 px inside the
 * box. Without a box (box missing, or it returns a hidden element) the card's
 * offset parent is the box. It also never covers the elements avoid()
 * returns. One exception ("Grow past arena", owner, 2 October 2026): a Hold
 * Shift card that would need a scale below 0.85 inside the box grows past
 * it, at its normal size, over the panels and fully inside the window, and
 * off the hovered element and the avoid() elements when it can be
 * (placeChampionCard; its card root then has data-past-arena "true").
 *
 * Hold Shift (createShiftHold): when the card leaves rows out ("+3 more",
 * "Hold Shift For All", by the four-row limit or because the box is short)
 * and Shift is held by itself for CHAMPION_CARD_HOLD_MS while the pointer
 * rests on the champion, the card shows every status in place ("All 7
 * statuses", "Release Shift To Shrink"); releasing Shift, any other key, any
 * click and any pointer movement shrink it back at once to the fitted plain
 * card. A card shown for a keyboard focus never expands. Expanding and
 * shrinking read the facts again, as refresh does.
 *
 * Fresh facts: a card never keeps old facts. The page calls the returned
 * refresh after every installed frame (a new tick, or Tab giving the view to
 * the other team): while the pointer rests, refresh finds the champion
 * element under the same pointer point again (or the focused element, and
 * places a focus card beside it again), asks factsFor again and redraws the
 * card, or hides it when nothing is there or factsFor gives null. So a
 * champion that just left the view's sight shows
 * its reduced "Unseen" card (portraits) or no card (arena bodies) at once,
 * not the full card of the earlier frame.
 *
 * Touch pointers are ignored, as in the vision. Side effects: event
 * listeners on root, capture listeners on root's window (the Hold Shift
 * gesture; they never cancel or change an event), the card's children,
 * `hidden`, `style.left` and `style.top` (set through the CSS object model,
 * which the page's content policy allows). Agent POV: factsFor must build
 * facts from the audience's own scene; this controller adds nothing, and the
 * Hold Shift card shows only the rows those facts hold.
 *
 * @param {{
 *   root: HTMLElement,
 *   card: HTMLElement,
 *   factsFor: (subject: HoverCardSubject) => ChampionCardFacts | null,
 *   enabled?: () => boolean,
 *   avoid?: () => ReadonlyArray<Element>,
 *   box?: () => Element | null,
 * }} options root is the area to watch (#bc); card is #hcard; factsFor turns
 *   the hovered element's subject into facts or null; enabled (default always
 *   true) turns the card off, for example outside live play or with the
 *   target_selection_visuals switch off; avoid (default none) returns the
 *   elements the card must never cover (hidden elements, with no area, are
 *   skipped); box (default none: the offset parent) returns the arena box the
 *   plain card must stay inside.
 * @returns {Readonly<{hide: () => void, refresh: () => void, destroy: () => void}>}
 *   hide hides the card now; refresh draws a shown card again from fresh
 *   facts or hides it (nothing happens while no card shows; call it after
 *   every installed frame); destroy removes the listeners and hides it.
 */
export function createHoverCard(options) {
  const { root, card, factsFor } = options;
  const enabled = options.enabled ?? (() => true);
  const avoid = options.avoid ?? (() => []);
  /**
   * What the card shows now: the element, its facts, the client point it was
   * placed at, whether the pointer (not a keyboard focus) showed it, and how
   * many status rows the drawn card shows (shown; fewer than the facts'
   * statuses when rows were left out).
   *
   * @type {{element: Element, facts: ChampionCardFacts, point: {x: number, y: number}, byPointer: boolean, shown: number} | null}
   */
  let current = null;
  const hide = () => {
    current = null;
    hold.cancel();
    card.hidden = true;
  };
  /**
   * Draw and place the card for element at point (client pixels): the Hold
   * Shift card with every status when the gesture says so, else the plain
   * card with as many rows as fit the box (renderFittedChampionCard).
   *
   * @param {Element} element
   * @param {ChampionCardFacts} facts
   * @param {{x: number, y: number}} point
   * @param {boolean} byPointer
   */
  const draw = (element, facts, point, byPointer) => {
    // Shown first: the card is measured while it is drawn (no paint happens
    // before this function returns).
    card.hidden = false;
    const parent = card.offsetParent instanceof HTMLElement ? card.offsetParent : root;
    const parentRect = parent.getBoundingClientRect();
    const box = areaOf(options.box) ?? {
      left: parentRect.left,
      top: parentRect.top,
      width: parentRect.width,
      height: parentRect.height,
    };
    const drawn = hold.expanded()
      ? (() => {
          const model = championCardModel(facts, { expanded: true });
          return { root: renderChampionCard(card, model), model };
        })()
      : renderFittedChampionCard({ host: card, surface: card, facts, box });
    const place = placeChampionCard({
      surface: card,
      root: drawn.root,
      box,
      pointer: point,
      avoid: avoid()
        .map((node) => node.getBoundingClientRect())
        .filter((rect) => rect.width > 0 && rect.height > 0),
      subject: element.getBoundingClientRect(),
    });
    card.style.left = `${place.left - parentRect.left}px`;
    card.style.top = `${place.top - parentRect.top}px`;
    current = { element, facts, point, byPointer, shown: drawn.model.rows.length };
  };
  /**
   * The client point a keyboard-focus card is placed at: the focused
   * element's lower left corner, 8 px up. Reads layout only.
   *
   * @param {Element} element
   * @returns {{x: number, y: number}}
   */
  const focusPoint = (element) => {
    const anchor = element.getBoundingClientRect();
    return { x: anchor.left, y: anchor.bottom - 8 };
  };
  /**
   * Show the card for element: at clientPoint for a pointer, or beside the
   * element (focusPoint) when clientPoint is null (a keyboard focus). Hides
   * the card when it is off or factsFor gives null.
   *
   * @param {Element} element
   * @param {{x: number, y: number} | null} clientPoint
   */
  const show = (element, clientPoint) => {
    if (!enabled()) return hide();
    const facts = factsFor(hoverCardSubject(element));
    if (facts === null) return hide();
    draw(element, facts, clientPoint ?? focusPoint(element), clientPoint !== null);
  };
  /** @param {EventTarget | null} target */
  const subjectOf = (target) =>
    target instanceof Element ? target.closest(HOVER_CARD_SELECTOR) : null;
  const doc = root.ownerDocument ?? null;
  /**
   * Find, without drawing, what the shown card describes now. For a card the
   * pointer showed: the champion element under the remembered pointer point,
   * hit-tested again (frames redraw arena bodies and portraits, so the old
   * element may be gone or moved); the topmost element there counts, as for
   * a pointer move. For a card a keyboard focus showed: the focused element,
   * when it is still a champion element inside root. Then that element's
   * fresh facts from factsFor. Without hit testing or a focus record (a DOM
   * stand-in), the remembered element is kept while it is still on the page.
   * Reads layout only.
   *
   * @returns {{element: Element, facts: ChampionCardFacts} | null} null when
   *   no card shows, the card is off (enabled() is false), no champion
   *   element is there now, or factsFor gives null (for example a champion
   *   that just left the view's sight).
   */
  const resolve = () => {
    if (current === null || card.hidden || !enabled()) return null;
    /** @type {Element | null} */
    let element;
    if (current.byPointer && typeof doc?.elementsFromPoint === "function") {
      const hit =
        doc
          .elementsFromPoint(current.point.x, current.point.y)
          .find((node) => !card.contains(node)) ?? null;
      element = subjectOf(hit);
    } else if (!current.byPointer && doc?.activeElement !== undefined) {
      const active = doc.activeElement;
      element = subjectOf(active) === active ? active : null;
    } else {
      element = current.element.isConnected === false ? null : current.element;
    }
    if (element === null || !root.contains(element)) return null;
    const facts = factsFor(hoverCardSubject(element));
    return facts === null ? null : { element, facts };
  };
  /**
   * Draw the shown card again from fresh facts (resolve), in place, or hide
   * it when nothing is left to describe. Runs on every installed frame (the
   * returned refresh) and when the Hold Shift gesture changes.
   */
  const refresh = () => {
    if (current === null) return;
    const now = resolve();
    if (now === null) return hide();
    // A pointer card stays at the resting pointer; a focus card follows its
    // element, which a new frame may have moved.
    draw(
      now.element,
      now.facts,
      current.byPointer ? current.point : focusPoint(now.element),
      current.byPointer,
    );
  };
  const view = doc?.defaultView ?? null;
  const hold = createShiftHold({
    target: typeof view?.addEventListener === "function" ? view : root,
    // Only when the fresh facts hold more statuses than the drawn plain card
    // shows (the four-row limit, or a short box, left rows out).
    canStart: () => {
      if (current === null || !current.byPointer) return false;
      const now = resolve();
      return now !== null && championCardModel(now.facts).total > current.shown;
    },
    onExpand: refresh,
    onShrink: refresh,
  });
  /** @param {PointerEvent} event */
  const onMove = (event) => {
    if (event.pointerType === "touch") return;
    const element = subjectOf(event.target);
    if (element === null || !root.contains(element)) return hide();
    show(element, { x: event.clientX, y: event.clientY });
  };
  /** @param {FocusEvent} event */
  const onFocus = (event) => {
    const element = subjectOf(event.target);
    if (element === null || element !== event.target) return;
    show(element, null);
  };
  /** @param {KeyboardEvent} event */
  const onKey = (event) => {
    if (event.key === "Escape") hide();
  };
  root.addEventListener("pointermove", onMove);
  root.addEventListener("pointerleave", hide);
  root.addEventListener("focusin", onFocus);
  root.addEventListener("focusout", hide);
  root.addEventListener("keydown", onKey);
  return Object.freeze({
    hide,
    refresh,
    destroy: () => {
      root.removeEventListener("pointermove", onMove);
      root.removeEventListener("pointerleave", hide);
      root.removeEventListener("focusin", onFocus);
      root.removeEventListener("focusout", hide);
      root.removeEventListener("keydown", onKey);
      hold.destroy();
      hide();
    },
  });
}

/**
 * Return a known frozen descriptor unchanged, otherwise validate/create
 * one through createSemanticDescriptor. Nonobjects throw TypeError. This
 * normalizes a display record, not its scientific provenance.
 *
 * @param {unknown} descriptor
 * @returns {AnyTooltipDescriptor}
 */
function normalizeDescriptor(descriptor) {
  if (!isRecord(descriptor)) {
    throw new TypeError("tooltip descriptor must be an object.");
  }
  if (semanticDescriptors.has(descriptor)) {
    return /** @type {AnyTooltipDescriptor} */ (descriptor);
  }
  return /** @type {AnyTooltipDescriptor} */ (createSemanticDescriptor(descriptor));
}

/**
 * Return whether rows or a visible section contain a row enabled for
 * surface. Call with normalized metadata and compact/full. Section headings
 * or summaries alone do not satisfy this row-presence check.
 *
 * @param {ReadonlyArray<SemanticRow>} rows
 * @param {ReadonlyArray<SemanticSection>} sections
 * @param {SemanticSurface} surface
 */
function hasVisibleSemanticRow(rows, sections, surface) {
  return (
    rows.some((row) => row.metadata[surface]) ||
    sections.some(
      (section) =>
        section.metadata[surface] && section.rows.some((row) => row.metadata[surface]),
    )
  );
}

/**
 * Require exactly compact/full boolean fields and return a frozen copy.
 * Invalid fields throw TypeError naming name. This uses ordinary property
 * reads and enumerable-key checks, not an accessor/prototype security check.
 *
 * @param {unknown} raw
 * @param {string} name
 * @returns {Readonly<SemanticMetadata>}
 */
function normalizeMetadata(raw, name) {
  if (
    !isRecord(raw) ||
    typeof raw.compact !== "boolean" ||
    typeof raw.full !== "boolean"
  ) {
    throw new TypeError(`${name} must contain compact and full booleans.`);
  }
  assertExactKeys(raw, ["compact", "full"], name);
  return Object.freeze({ compact: raw.compact, full: raw.full });
}

/**
 * Require exactly nonempty label/value strings plus valid metadata and
 * return a frozen normalized row. Trim strings; invalid input throws
 * TypeError naming name. No DOM creation or scientific validation occurs.
 *
 * @param {unknown} raw
 * @param {string} name
 * @returns {SemanticRow}
 */
function normalizeSemanticRow(raw, name) {
  if (!isRecord(raw)) {
    throw new TypeError(`${name} must be an object.`);
  }
  assertExactKeys(raw, ["label", "value", "metadata"], name);
  return Object.freeze({
    label: nonEmptyString(raw.label, `${name}.label`),
    value: nonEmptyString(raw.value, `${name}.value`),
    metadata: normalizeMetadata(raw.metadata, `${name}.metadata`),
  });
}

/**
 * Require exactly title, summary, rows and metadata, then return a frozen
 * section with normalized/frozen rows. title must be nonempty; summary may
 * be null or undefined, otherwise must be nonempty text. Invalid fields
 * throw TypeError naming name. Preserve row order and do not mutate raw.
 *
 * @param {unknown} raw
 * @param {string} name
 * @returns {SemanticSection}
 */
function normalizeSemanticSection(raw, name) {
  if (!isRecord(raw)) {
    throw new TypeError(`${name} must be an object.`);
  }
  assertExactKeys(raw, ["title", "summary", "rows", "metadata"], name);
  const summary =
    raw.summary === undefined || raw.summary === null
      ? null
      : nonEmptyString(raw.summary, `${name}.summary`);
  return Object.freeze({
    title: nonEmptyString(raw.title, `${name}.title`),
    summary,
    rows: Object.freeze(
      normalizeArray(raw.rows, `${name}.rows`).map((row, index) =>
        normalizeSemanticRow(row, `${name}.rows[${index}]`),
      ),
    ),
    metadata: normalizeMetadata(raw.metadata, `${name}.metadata`),
  });
}

/**
 * Return value as-is when it is an array; otherwise throw TypeError
 * naming name. This checks neither dense indices nor element validity.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {unknown[]}
 */
function normalizeArray(value, name) {
  if (!Array.isArray(value)) {
    throw new TypeError(`${name} must be an array.`);
  }
  return value;
}

/**
 * Require enumerable string keys to match expected, ignoring order.
 * Throw TypeError naming name for extra/missing keys. Do not mutate either
 * input. Symbols, inherited fields and accessors are not rejected here.
 *
 * @param {Record<string, unknown>} value
 * @param {ReadonlyArray<string>} expected
 * @param {string} name
 */
function assertExactKeys(value, expected, name) {
  const actual = Object.keys(value).sort();
  const required = [...expected].sort();
  if (
    actual.length !== required.length ||
    actual.some((key, index) => key !== required[index])
  ) {
    throw new TypeError(`${name} must contain exactly: ${expected.join(", ")}.`);
  }
}

/**
 * Create an unattached definition list for normalized rows in order.
 * Use fixed class names and textContent for label/value pairs. Return the
 * list; no markup is interpreted and inputs are unchanged.
 *
 * @param {Document} ownerDocument
 * @param {ReadonlyArray<SemanticRow>} rows
 */
function renderRows(ownerDocument, rows) {
  const list = ownerDocument.createElement("dl");
  list.className = "semantic-explanation__rows";
  for (const row of rows) {
    const term = ownerDocument.createElement("dt");
    term.className = "semantic-explanation__label";
    term.textContent = row.label;
    const value = ownerDocument.createElement("dd");
    value.className = "semantic-explanation__value";
    value.textContent = row.value;
    list.append(term, value);
  }
  return list;
}

/**
 * Return the declared numeric priority for a known semantic kind.
 * Unknown and inherited-property names receive Number.MAX_SAFE_INTEGER.
 * Smaller numbers win; no descriptor validation occurs.
 *
 * @param {string} kind
 * @returns {number}
 */
function kindPriority(kind) {
  return Object.hasOwn(KIND_PRIORITY, kind)
    ? KIND_PRIORITY[/** @type {keyof typeof KIND_PRIORITY} */ (kind)]
    : Number.MAX_SAFE_INTEGER;
}

/**
 * Compare [kind priority, paint order, stable ID, input index] tuples.
 * Return negative/zero/positive for first before/equal/after second. IDs use
 * JavaScript lexical order; no locale sorting or mutation occurs.
 *
 * @param {[number, number, string, number]} first
 * @param {[number, number, string, number]} second
 * @returns {number}
 */
function compareRanks(first, second) {
  const priorityComparison = first[0] - second[0];
  if (priorityComparison !== 0) {
    return priorityComparison;
  }
  const paintOrderComparison = first[1] - second[1];
  if (paintOrderComparison !== 0) {
    return paintOrderComparison;
  }
  const stableIdComparison = first[2] === second[2] ? 0 : first[2] < second[2] ? -1 : 1;
  return stableIdComparison === 0 ? first[3] - second[3] : stableIdComparison;
}

/**
 * Compare numeric arrays lexicographically, with shorter length winning
 * when their common prefix matches. Return a signed difference without
 * mutation. Caller supplies finite scores; this does not handle NaN specially.
 *
 * @param {number[]} first
 * @param {number[]} second
 * @returns {number}
 */
function compareNumericRanks(first, second) {
  for (let index = 0; index < Math.min(first.length, second.length); index += 1) {
    const comparison = first[index] - second[index];
    if (comparison !== 0) {
      return comparison;
    }
  }
  return first.length - second.length;
}

/**
 * Sum how far rectangle extends past each bounds edge in CSS pixels.
 * Return zero when contained. This is edge-distance cost, not overflow area;
 * inputs must already have finite ordered edges.
 *
 * @param {TooltipRectangle} rectangle
 * @param {TooltipRectangle} bounds
 * @returns {number}
 */
function rectangleOverflow(rectangle, bounds) {
  return (
    Math.max(bounds.left - rectangle.left, 0) +
    Math.max(rectangle.right - bounds.right, 0) +
    Math.max(bounds.top - rectangle.top, 0) +
    Math.max(rectangle.bottom - bounds.bottom, 0)
  );
}

/**
 * Return intersection area of two ordered rectangles in square CSS
 * pixels. Disjoint or edge-touching rectangles return zero. No validation
 * or mutation occurs.
 *
 * @param {TooltipRectangle} first
 * @param {TooltipRectangle} second
 * @returns {number}
 */
function rectangleOverlapArea(first, second) {
  const width = Math.min(first.right, second.right) - Math.max(first.left, second.left);
  const height =
    Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top);
  return Math.max(width, 0) * Math.max(height, 0);
}

/**
 * Return the shortest Euclidean CSS-pixel distance from point to an
 * ordered rectangle. Points inside or on the edge return zero. Caller
 * supplies finite validated geometry.
 *
 * @param {TooltipPoint} point
 * @param {TooltipRectangle} rectangle
 * @returns {number}
 */
function pointToRectangleDistance(point, rectangle) {
  const horizontal =
    point.x < rectangle.left
      ? rectangle.left - point.x
      : point.x > rectangle.right
        ? point.x - rectangle.right
        : 0;
  const vertical =
    point.y < rectangle.top
      ? rectangle.top - point.y
      : point.y > rectangle.bottom
        ? point.y - rectangle.bottom
        : 0;
  return Math.hypot(horizontal, vertical);
}

/**
 * Return a new rectangle with left/top and right/bottom derived from
 * size.width/height. Values are CSS pixels; caller validates finite
 * coordinates and nonnegative size. Inputs remain unchanged.
 *
 * @param {number} left
 * @param {number} top
 * @param {TooltipSize} size
 * @returns {TooltipRectangle}
 */
function rectangleFromPosition(left, top, size) {
  return {
    left,
    top,
    right: left + size.width,
    bottom: top + size.height,
  };
}

/**
 * Return left/top clamped so size fits viewport where possible. When a
 * card is larger than an axis, anchor it at that viewport's starting edge.
 * Inputs are validated CSS pixels; no resizing or DOM mutation occurs.
 *
 * @param {number} left
 * @param {number} top
 * @param {TooltipSize} size
 * @param {TooltipRectangle} viewport
 * @returns {{left: number, top: number}}
 */
function clampPosition(left, top, size, viewport) {
  const maxLeft = viewport.right - size.width;
  const maxTop = viewport.bottom - size.height;
  return {
    left: maxLeft < viewport.left ? viewport.left : clamp(left, viewport.left, maxLeft),
    top: maxTop < viewport.top ? viewport.top : clamp(top, viewport.top, maxTop),
  };
}

/**
 * Clamp numeric value to inclusive lower/upper. Caller provides ordered
 * finite bounds; this helper performs no validation or rounding.
 *
 * @param {number} value
 * @param {number} lower
 * @param {number} upper
 * @returns {number}
 */
function clamp(value, lower, upper) {
  return Math.min(Math.max(value, lower), upper);
}

/**
 * Require finite numeric left/top/right/bottom and return a frozen copy.
 * Zero width/height is allowed; reversed edges throw RangeError and invalid
 * fields throw TypeError naming name. Extra fields are ignored.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {TooltipRectangle}
 */
function normalizeRectangle(value, name) {
  if (!isRecord(value)) {
    throw new TypeError(`${name} must be an object.`);
  }
  const left = finiteNumber(value.left, `${name}.left`);
  const top = finiteNumber(value.top, `${name}.top`);
  const right = finiteNumber(value.right, `${name}.right`);
  const bottom = finiteNumber(value.bottom, `${name}.bottom`);
  if (right < left || bottom < top) {
    throw new RangeError(`${name} edges must be ordered.`);
  }
  return Object.freeze({ left, top, right, bottom });
}

/**
 * Require finite numeric x/y and return a frozen copy. Invalid input
 * throws TypeError naming name; extra fields are ignored. Units remain
 * those supplied by the caller, normally viewport CSS pixels.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {TooltipPoint}
 */
function normalizePoint(value, name) {
  if (!isRecord(value)) {
    throw new TypeError(`${name} must be an object.`);
  }
  return Object.freeze({
    x: finiteNumber(value.x, `${name}.x`),
    y: finiteNumber(value.y, `${name}.y`),
  });
}

/**
 * Require nonnegative finite width/height and return a frozen copy.
 * Zero dimensions are allowed. Invalid types throw TypeError and negative
 * values RangeError naming name. Extra fields are ignored.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {TooltipSize}
 */
function normalizeSize(value, name) {
  if (!isRecord(value)) {
    throw new TypeError(`${name} must be an object.`);
  }
  const width = nonNegativeFinite(value.width, `${name}.width`);
  const height = nonNegativeFinite(value.height, `${name}.height`);
  return Object.freeze({ width, height });
}

/**
 * Return the document's viewport bounds at origin in CSS pixels.
 * Prefer window innerWidth/innerHeight, falling back to documentElement
 * client sizes. Nonpositive dimensions throw RangeError. This reads layout
 * without changing the document.
 *
 * @param {Document} ownerDocument
 * @returns {TooltipRectangle}
 */
function viewportRectangle(ownerDocument) {
  const view = ownerDocument.defaultView;
  const width = view?.innerWidth ?? ownerDocument.documentElement.clientWidth;
  const height = view?.innerHeight ?? ownerDocument.documentElement.clientHeight;
  if (width <= 0 || height <= 0) {
    throw new RangeError("tooltip viewport must have positive dimensions.");
  }
  return Object.freeze({ left: 0, top: 0, right: width, bottom: height });
}

/**
 * Return the visible intersection of the owner's registered surface
 * (or closest SVG) with the document viewport. If no surface or no positive
 * intersection exists, use the viewport. A closest SVG (not a registered
 * surface) smaller than MIN_INFERRED_SURFACE (160 CSS pixels) on either side
 * is an icon, not a drawing surface, so the viewport is used then too: an
 * owner inside a scorebug portrait's 28 px picture gets a card of normal
 * width. Read/validate DOM rectangles; malformed bounds can throw. No
 * registration or layout mutation occurs.
 *
 * @param {Element} owner
 * @param {Document} ownerDocument
 * @returns {TooltipRectangle}
 */
function ownerSurfaceRectangle(owner, ownerDocument) {
  const viewport = viewportRectangle(ownerDocument);
  const explicit = surfaceByOwner.get(owner);
  const inferred =
    explicit ?? (typeof owner.closest === "function" ? owner.closest("svg") : null);
  if (inferred === null || inferred === undefined) {
    return viewport;
  }
  const raw = normalizeRectangle(
    inferred.getBoundingClientRect(),
    "tooltip owner surface rectangle",
  );
  if (
    explicit === undefined &&
    (raw.right - raw.left < MIN_INFERRED_SURFACE ||
      raw.bottom - raw.top < MIN_INFERRED_SURFACE)
  ) {
    return viewport;
  }
  const intersection = Object.freeze({
    left: Math.max(raw.left, viewport.left),
    top: Math.max(raw.top, viewport.top),
    right: Math.min(raw.right, viewport.right),
    bottom: Math.min(raw.bottom, viewport.bottom),
  });
  return intersection.right > intersection.left &&
    intersection.bottom > intersection.top
    ? intersection
    : viewport;
}

/**
 * Hit-test viewport CSS coordinates through root when supported, otherwise
 * its ownerDocument. Return elements in the browser's paint order, or []
 * when neither supplies hit testing. No containment filtering occurs here.
 *
 * @param {Document | ShadowRoot | HTMLElement} root
 * @param {number} x
 * @param {number} y
 * @returns {Element[]}
 */
function elementsFromPoint(root, x, y) {
  const pointSource = isPointSource(root)
    ? root
    : isPointSource(root.ownerDocument)
      ? root.ownerDocument
      : null;
  return pointSource === null ? [] : pointSource.elementsFromPoint(x, y);
}

/**
 * Walk element and its parentElement chain to the first owner carrying
 * both the marker attribute and a registered descriptor. Return that element
 * or null. A forged marker alone is insufficient; Shadow DOM host traversal
 * is not added by this helper.
 *
 * @param {Element} element
 * @returns {Element | null}
 */
function closestRegisteredOwner(element) {
  /** @type {Element | null} */
  let current = element;
  while (current !== null) {
    if (
      current.hasAttribute(TOOLTIP_OWNER_ATTRIBUTE) &&
      descriptorByOwner.has(current)
    ) {
      return current;
    }
    current = current.parentElement;
  }
  return null;
}

/**
 * Return whether trigger-to-owner ancestor traversal encounters a native
 * interactive tag or contenteditable=true. Include owner itself. This
 * preserves native actions during inspection. Caller ensures owner is the
 * relevant ancestor; no event or DOM change occurs.
 *
 * @param {Element} trigger
 * @param {Element} owner
 * @returns {boolean}
 */
function isNativeInteractiveTarget(trigger, owner) {
  /** @type {Element | null} */
  let current = trigger;
  while (current !== null) {
    const tagName = current.tagName.toLowerCase();
    if (
      ["a", "button", "input", "select", "textarea", "summary"].includes(tagName) ||
      current.getAttribute("contenteditable") === "true"
    ) {
      return true;
    }
    if (current === owner) {
      break;
    }
    current = current.parentElement;
  }
  return false;
}

/**
 * Collect positive-area rectangles for an inspected battlefield agent
 * and its durable docks sharing data-slot. Return [] outside a usable SVG
 * context. Only the declared agent/dock classes are included. This reads
 * layout for tooltip avoidance; it does not interpret global slot identity
 * or reveal additional scientific information.
 *
 * @param {Element} owner
 * @returns {TooltipRectangle[]}
 */
function localProtectedRects(owner) {
  if (typeof owner.closest !== "function") {
    return [];
  }
  const battlefield = owner.closest("svg");
  const slotOwner = owner.closest("[data-slot]");
  const slot = slotOwner?.getAttribute("data-slot");
  if (
    battlefield === null ||
    battlefield.tagName.toLowerCase() !== "svg" ||
    slot === null
  ) {
    return [];
  }
  const protectedClasses = new Set([
    "agent",
    "status-dock",
    "cooldown-dock",
    "required-dock-fallback-dock",
    "modifier-dock",
  ]);
  const rectangles = [];
  for (const element of battlefield.querySelectorAll("[data-slot]")) {
    if (
      element.getAttribute("data-slot") !== slot ||
      ![...protectedClasses].some((className) => element.classList.contains(className))
    ) {
      continue;
    }
    const bounds = element.getBoundingClientRect();
    if (bounds.width > 0 && bounds.height > 0) {
      rectangles.push(normalizeRectangle(bounds, "local protected rectangle"));
    }
  }
  return rectangles;
}

/**
 * Return whether owner is connected inside root. Document roots check
 * documentElement containment when available; other roots use contains.
 * Minimal document stand-ins without contains are accepted. This is a DOM
 * ownership check, not a scientific authorization check.
 *
 * @param {Document | ShadowRoot | HTMLElement} root
 * @param {Element} owner
 * @returns {boolean}
 */
function ownerWithinRoot(root, owner) {
  if ("isConnected" in owner && owner.isConnected === false) {
    return false;
  }
  if (root === owner.ownerDocument) {
    const documentElement = root.documentElement;
    return typeof documentElement.contains === "function"
      ? documentElement.contains(owner)
      : true;
  }
  return typeof root.contains === "function" && root.contains(owner);
}

/**
 * Return a frozen x/y point from finite numeric clientX/clientY, or null
 * when unavailable. Coordinates remain viewport CSS pixels; no page-offset
 * or device-pixel conversion is applied.
 *
 * @param {Event} event
 * @returns {TooltipPoint | null}
 */
function eventPoint(event) {
  if (
    !("clientX" in event) ||
    !("clientY" in event) ||
    typeof event.clientX !== "number" ||
    typeof event.clientY !== "number" ||
    !Number.isFinite(event.clientX) ||
    !Number.isFinite(event.clientY)
  ) {
    return null;
  }
  return Object.freeze({ x: event.clientX, y: event.clientY });
}

/**
 * Split element's named attribute on whitespace and remove empty tokens.
 * Return a new string array; a missing attribute yields []. Preserve token
 * order and duplicates and leave the attribute unchanged.
 *
 * @param {Element} element
 * @param {string} name
 * @returns {string[]}
 */
function attributeTokens(element, name) {
  return (element.getAttribute(name) ?? "").split(/\s+/u).filter(Boolean);
}

/**
 * Observe child-list changes throughout root and call onRemoval.
 * Document roots observe documentElement; other roots observe themselves.
 * Return the MutationObserver, or null when the owning window lacks it.
 * The callback runs for all child-list mutations, not removals alone; the
 * caller checks ownership and disconnects the observer on disposal.
 *
 * @param {Document | ShadowRoot | HTMLElement} root
 * @param {() => void} onRemoval
 * @returns {MutationObserver | null}
 */
function createRemovalObserver(root, onRemoval) {
  const ownerDocument =
    "ownerDocument" in root && root.ownerDocument !== null
      ? root.ownerDocument
      : /** @type {Document} */ (root);
  const Observer = ownerDocument.defaultView?.MutationObserver;
  if (Observer === undefined) {
    return null;
  }
  const observer = new Observer(onRemoval);
  const target =
    "documentElement" in root && root.documentElement !== null
      ? root.documentElement
      : root;
  observer.observe(target, { childList: true, subtree: true });
  return observer;
}

/**
 * Return whether value is any non-null object, including arrays. This
 * shallow guard does not establish a plain-data record or validate fields.
 *
 * @param {unknown} value
 * @returns {value is Record<string, unknown>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null;
}

/**
 * Check for the attribute/geometry methods and parentElement field used
 * by tooltip owners. Accept compatible DOM stand-ins. This structural check
 * is not an instanceof check or proof of a connected element.
 *
 * @param {unknown} value
 * @returns {value is Element}
 */
function isElementNode(value) {
  return (
    isRecord(value) &&
    typeof value.hasAttribute === "function" &&
    typeof value.getAttribute === "function" &&
    typeof value.setAttribute === "function" &&
    typeof value.removeAttribute === "function" &&
    typeof value.getBoundingClientRect === "function" &&
    "parentElement" in value
  );
}

/**
 * Apply the shared element-shape check for tooltip ownership. This does
 * not mean the element has a registered descriptor or belongs to a root.
 *
 * @param {unknown} value
 * @returns {value is Element}
 */
function isTooltipOwner(value) {
  return isElementNode(value);
}

/**
 * Check that value supports event listener add/remove and exposes an
 * ownerDocument or documentElement. Accept compatible stand-ins; membership
 * and connectedness are checked separately.
 *
 * @param {unknown} value
 * @returns {value is Document | ShadowRoot | HTMLElement}
 */
function isEventRoot(value) {
  return (
    isRecord(value) &&
    typeof value.addEventListener === "function" &&
    typeof value.removeEventListener === "function" &&
    ("ownerDocument" in value || "documentElement" in value)
  );
}

/**
 * Check the element-like shape plus id, hidden, style.setProperty and
 * ownerDocument needed by a tooltip. This does not validate a nonempty ID
 * or whether another controller owns the surface.
 *
 * @param {unknown} value
 * @returns {value is HTMLElement}
 */
function isTooltipSurface(value) {
  return (
    isElementNode(value) &&
    "id" in value &&
    "hidden" in value &&
    "style" in value &&
    isRecord(value.style) &&
    typeof value.style.setProperty === "function" &&
    "ownerDocument" in value
  );
}

/**
 * Check element-like shape and a textContent property. Minimal compatible
 * DOM stand-ins are allowed; this does not check a particular HTML tag.
 *
 * @param {unknown} value
 * @returns {value is HTMLElement}
 */
function isTextSurface(value) {
  return isElementNode(value) && "textContent" in value;
}

/**
 * Return whether value exposes an elementsFromPoint function. This does
 * not call it, validate its results or establish a DOM root type.
 *
 * @param {unknown} value
 * @returns {value is {elementsFromPoint: (x: number, y: number) => Element[]}}
 */
function isPointSource(value) {
  return isRecord(value) && typeof value.elementsFromPoint === "function";
}

/**
 * Return trimmed string value when nonempty; otherwise throw TypeError
 * naming name. No markup parsing, maximum length or identity check occurs.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {string}
 */
function nonEmptyString(value, name) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${name} must be a non-empty string.`);
  }
  return value.trim();
}

/**
 * Return a finite numeric value unchanged, or throw TypeError naming
 * name. Do not coerce strings/booleans or round values.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {number}
 */
function finiteNumber(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be finite.`);
  }
  return value;
}

/**
 * Return a finite numeric value at least zero. Invalid types/nonfinite
 * values throw TypeError; negative numbers throw RangeError naming name.
 * Zero and fractions are accepted without rounding.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {number}
 */
function nonNegativeFinite(value, name) {
  const finite = finiteNumber(value, name);
  if (finite < 0) {
    throw new RangeError(`${name} must be non-negative.`);
  }
  return finite;
}
