/**
 * @file Draw the page shell for all three clients: the BattleClient, the
 * DevClient combat screen and the Replay Viewer.
 *
 * Who calls it: main.js calls applyProductHeader once while it installs the
 * launcher's product identity, renderCommandHelpLists once, and createShell
 * once at startup. main.js then calls the returned renderers on every repaint.
 *
 * What it owns: the brand lockup (the owner's emblem, the rune-cut "MARL-BGs"
 * wordmark, whose flat hyphen keeps the one spelling "MARL-BGs", and the
 * product name) and the tab title; the place line (the map or a scenario's
 * tale title in bold, then "Map 47" or "Scenario 1" and "Team Deathmatch",
 * in two spans so that only "Team Deathmatch" gives way when the top bar
 * runs short and the map name stays whole in live play); the live terminal
 * pill's "Game Over"; the live Visual Filters drawer's top (below the
 * scorebug); the View switch (Agent POV first, its hover help kept true) and
 * the hidden View select it drives; the Replay Viewer's POV agent chip and its
 * picker; the notice pill and the status pills (the audience pill is for
 * screen readers; the View switch shows the view); the Details popover (Step,
 * Transition with Copy, and the replay badges); the Help and Exit
 * labels; the Title Card shown while no frame has come for 300 ms; in every
 * client, the page palette that follows the High Contrast switch; in live
 * play, the Brand Signature (emblem and "MARL-BGs" in the margin by the floor's
 * bottom-right corner, page chrome only, off by default); the arena's
 * view tag in live play (#pov-tag: "Agent POV · Your Team's Sight · N Unseen"
 * or the amber "Oracle View · You See Everything"; the old #oracle-view-tag
 * stays hidden); the How to Play overlay's four rule cards (#htp-body); the
 * live hover card (#hcard): its facts, built from the audience's own scene
 * by championCardFacts, drawn by tooltip.js as the Champion Card and drawn
 * again from every new frame (refreshHoverCard, which main.js calls after
 * each paint), so a resting pointer never keeps an old frame's card (the
 * Replay Viewer's arena bodies get the same card through main.js); the notice
 * line; the Help
 * dialog's key lists, written from the one command set (command-set.js);
 * which page sections show for each product; Reconnect, Exit and Reset
 * availability; the live Ranges toggle state; and the battlefield's accessible
 * role, label and instructions. Its styles live in src/styles/shell.css and,
 * for live play, src/styles/battleclient/topbar.css and overlays.css.
 *
 * Boundary: it reads the page state but never assigns it, and it never sends a
 * request itself. The View switch sets the View select and fires its change
 * event, so main.js sends set_view as before; the POV agent picker calls the
 * activateReplayPovAgent callback main.js passes. The scoreboard, recording
 * panel, replay metadata, command deck and control help keep their own
 * owners; renderSessionToolbar calls them in the same order as before.
 * main.js opens and closes the How to Play overlay; this file only fills it.
 *
 * DevClient: the private DevClient sets its own tab title, #app-title and
 * #help-heading after main.js has loaded. Keep those IDs, keep applyProductHeader
 * running while main.js loads, and never rewrite #app-title later.
 *
 * Agent POV: on or over the arena this file draws only the view tag and the
 * hover card. The view tag's Unseen count and the hover card read only the
 * Agent POV scene (its bodies and spawn pads), the transport frame's combat
 * configuration (who is Manual, controller labels; in the Replay Viewer the
 * match summary's team names instead), the team's own visual events and, for
 * an unseen champion's class only, the public identity fields
 * (publicRosterIdentities: public ID, team, team slot, class; the same facts
 * the scorebug portraits show), never other researcher_space facts; status
 * sources are named by class only, so the card is
 * the same in Agent POV and Oracle View; an unseen champion's card says only
 * its identity and who plays it. The top bar shows the
 * audience name, the simulator step, an opaque transition ID, the public map or
 * tale title and, in the Replay Viewer only, the POV agent and a picker of
 * every agent's identity (the Replay Viewer's researcher chrome lists
 * everyone). Agent POV and Oracle View change only what the human sees.
 */

import { exactAuthorizedAgentIdentityV1 } from "./agent-identity.js";
import {
  authorizedPresentationAudience,
  authorizedPresentationIdentityRows,
  authorizedPresentationIncomingRows,
  authorizedPresentationInspectionState,
  authorizedPresentationLatestTransitionId,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import { classNamesV1 } from "./class-documentation.js";
import { commandHelpRows } from "./command-set.js";
import { createSvgIcon } from "./icons.js";
import { publicRosterIdentities } from "./match-summary.js";
import { applyPalette } from "./palette.js";
import {
  championCardFactsFromAgent,
  createHoverCard,
  createSemanticDescriptor,
  registerTooltipOwner,
} from "./tooltip.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualFilterEnabled,
} from "./visual-filters.js";
import { classTokenFromId } from "./vocabulary.js";

/**
 * The installed transport and presentation pair, or null while authority is
 * pending. main.js resolves it with resolveInstalledPresentationAuthorityV1.
 *
 * @typedef {Readonly<{
 *   transport: Readonly<Record<string, any>>,
 *   presentation: Readonly<Record<string, any>>,
 * }> | null} InstalledAuthority
 */

/**
 * Tab title for each launcher product: the official title "MARL-BattleGrounds"
 * plus the product name. Both live products share "combat_debugger"; the
 * private DevClient replaces its title after load.
 *
 * @type {Readonly<{combat_debugger: string, replay_viewer: string}>}
 */
export const PRODUCT_TITLES = Object.freeze({
  combat_debugger: "MARL-BattleGrounds BattleClient",
  replay_viewer: "MARL-BattleGrounds Replay Viewer",
});

/**
 * Product name shown in gray after the header's brand lockup.
 *
 * @type {Readonly<{combat_debugger: string, replay_viewer: string}>}
 */
export const PRODUCT_NAMES = Object.freeze({
  combat_debugger: "BattleClient",
  replay_viewer: "Replay Viewer",
});

/** The everyday name, carved in the header lockup. */
export const BRAND_NAME = "MARL-BGs";

/** The official title, written in mixed case on the Title Card and the tab. */
export const OFFICIAL_TITLE = "MARL-BattleGrounds";

/** How long the arena must wait for its first frame before the Title Card shows, in ms. */
export const TITLE_CARD_DELAY_MS = 300;

/** Brand Signature emblem size, in CSS px (the Rune-Cut Crest pick's 18 px emblem). */
export const BRAND_SIGNATURE_EMBLEM_PX = 18;

/** Brand Signature wordmark cap height, in CSS px (the pick's cap-10 rune letters). */
export const BRAND_SIGNATURE_CAP_PX = 10;

// Brand Signature spacing, in CSS px: the gap between emblem and letters in a
// row (the store's 8 px) and in a stack; the clear space kept from the floor's
// edge (it clears the Home End Marks bracket, its badge and the Home Gates);
// and the inset from the arena box's inner edge.
const SIGNATURE_ROW_GAP = 8;
const SIGNATURE_STACK_GAP = 4;
const SIGNATURE_FLOOR_CLEARANCE = 16;
const SIGNATURE_BOX_INSET = 4;

/**
 * Decide where the live Brand Signature goes, from measured page boxes.
 *
 * It only ever uses clear space: inside the arena box, outside the
 * battlefield picture (the SVG, where every mark is drawn) and at least 16 px
 * from the floor's edge. The store's rule comes first: when the band under
 * the floor is at least 40 px tall and the floor at least 400 px wide, the
 * signature is one row (emblem, 8 px gap, letters) whose right edge sits on
 * the floor's right edge, centered in the clear part of that band. The live
 * page fits on one screen, so that band is often thinner; then the signature
 * stacks (emblem above letters, 4 px gap) in the clear margin right of the
 * floor, centered across it, with its bottom on the floor's bottom edge.
 * When neither fits (a narrow window, or a camera that fills the box) it
 * returns null: the signature never covers the floor or another mark.
 * Pure: reads plain numbers only.
 *
 * @param {{
 *   box: {left: number, top: number, right: number, bottom: number},
 *   field: {left: number, top: number, right: number, bottom: number},
 *   floor: {left: number, top: number, right: number, bottom: number},
 *   word: {width: number, height: number},
 * }} input box is the arena box (`#arena`, inside its border), field the
 *   battlefield SVG (`#battlefield`) and floor the floor edge
 *   (`rect.map-boundary`), all in page CSS px as getBoundingClientRect gives
 *   them; word is the carved wordmark's drawn size in CSS px.
 * @returns {{layout: "row" | "stack", left: number, top: number} | null} The
 *   layout and the signature's top-left corner in CSS px relative to the box,
 *   or null when it must not show. Nonfinite input also gives null.
 */
export function brandSignaturePlacement(input) {
  const { box, field, floor, word } = input;
  const values = [box, field, floor].flatMap((rect) => [
    rect.left,
    rect.top,
    rect.right,
    rect.bottom,
  ]);
  values.push(word.width, word.height);
  if (!values.every((value) => Number.isFinite(value))) return null;
  const emblem = BRAND_SIGNATURE_EMBLEM_PX;
  const clearTop = Math.max(floor.bottom + SIGNATURE_FLOOR_CLEARANCE, field.bottom);
  const clearLeft = Math.max(floor.right + SIGNATURE_FLOOR_CLEARANCE, field.right);
  const freeBottom = box.bottom - SIGNATURE_BOX_INSET;
  const freeRight = box.right - SIGNATURE_BOX_INSET;
  const rowWidth = emblem + SIGNATURE_ROW_GAP + word.width;
  const rowHeight = Math.max(emblem, word.height);
  if (
    box.bottom - floor.bottom >= 40 &&
    floor.right - floor.left >= 400 &&
    freeBottom - clearTop >= rowHeight &&
    floor.right - rowWidth >= box.left
  ) {
    return {
      layout: "row",
      left: floor.right - rowWidth - box.left,
      top: (clearTop + freeBottom - rowHeight) / 2 - box.top,
    };
  }
  const stackWidth = Math.max(emblem, word.width);
  const stackHeight = emblem + SIGNATURE_STACK_GAP + word.height;
  const bottom = Math.min(floor.bottom, freeBottom);
  if (freeRight - clearLeft >= stackWidth && bottom - stackHeight >= box.top) {
    return {
      layout: "stack",
      left: (clearLeft + freeRight - stackWidth) / 2 - box.left,
      top: bottom - stackHeight - box.top,
    };
  }
  return null;
}

// Rune letter geometry, ported from the design store's Rune-Cut Crest
// (skins/brand-and-typography.js). Units: cap height 10. Each entry is
// [advance, strokes, closed flags or null, lozenges or null]. A stroke is a
// list of [u, v] points; a lozenge is [u, v, half width, half height].
// T is the top of the cap height and B its bottom, in cap-10 units.
const T = 0.9;
const B = 9.1;

/**
 * The rune-cut letters of the brand wordmark: straight strokes and 45-degree
 * chamfers only; the A's crossbar and the hyphen are small lozenges. The
 * hyphen's lozenge is flat (3.8 units wide, 1.9 tall, about one stroke), so
 * the carved name reads "MARL-BGs", the one spelling every client shows
 * (owner, 3 Oct 2026). The store's upright lozenge (2.5 wide, 4.2 tall) read
 * as a middle dot: "MARL·BGs".
 *
 * @type {Readonly<Record<string, readonly [number, ReadonlyArray<ReadonlyArray<readonly [number, number]>>, ReadonlyArray<boolean> | null, ReadonlyArray<readonly [number, number, number, number]> | null]>>}
 */
// biome-ignore format: one letter per line keeps the ported geometry readable.
export const RUNE_LETTERS = Object.freeze({
  M: [9, [[[T, B], [T, T], [4.5, 5.6], [8.1, T], [8.1, B]]], null, null],
  A: [8.2, [[[T, B], [4.1, T], [7.3, B]]], null, [[4.1, 6.35, 1.05, 1.5]]],
  R: [8, [[[T, B], [T, T], [5.3, T], [7.1, 2.7], [7.1, 3.7], [5.3, 5.5], [T, 5.5]], [[4.3, 5.5], [7.1, B]]], null, null],
  L: [7.3, [[[T, T], [T, B], [6.6, B]]], null, null],
  B: [8, [[[T, B], [T, T], [5.1, T], [6.6, 2.4], [6.6, 3.5], [5.3, 4.8], [T, 4.8]], [[5.3, 4.8], [7.1, 6.5], [7.1, 7.5], [5.5, B], [T, B]]], null, null],
  T: [8, [[[0.3, T], [7.7, T]], [[4, T], [4, B]]], null, null],
  E: [7.5, [[[6.8, T], [T, T], [T, B], [6.8, B]], [[T, 5], [5.6, 5]]], null, null],
  G: [8, [[[7.1, 2.6], [5.4, T], [2.6, T], [T, 2.6], [T, 7.4], [2.6, B], [5.4, B], [7.1, 7.4], [7.1, 5.3], [4.6, 5.3]]], null, null],
  O: [8, [[[2.6, T], [5.4, T], [7.1, 2.6], [7.1, 7.4], [5.4, B], [2.6, B], [T, 7.4], [T, 2.6]]], [true], null],
  U: [8, [[[T, T], [T, 7.4], [2.6, B], [5.4, B], [7.1, 7.4], [7.1, T]]], null, null],
  N: [8, [[[T, B], [T, T], [7.1, B], [7.1, T]]], null, null],
  D: [8, [[[T, T], [4.8, T], [7.1, 3.2], [7.1, 6.8], [4.8, B], [T, B]]], [true], null],
  S: [8, [[[7.1, 2.4], [5.6, T], [2.4, T], [T, 2.4], [T, 3.6], [2.3, 5], [5.7, 5], [7.1, 6.4], [7.1, 7.6], [5.6, B], [2.4, B], [T, 7.6]]], null, null],
  s: [6.2, [[[5.3, 4.6], [4.3, 3.6], [1.9, 3.6], [T, 4.6], [T, 5.3], [1.9, 6.3], [4.3, 6.3], [5.3, 7.3], [5.3, 8.1], [4.3, B], [1.9, B], [T, 8.1]]], null, null],
  "-": [4.4, [], null, [[2.2, 5, 1.9, 0.95]]],
  " ": [4, [], null, null],
});

/** @param {number} value @returns {string} The value rounded to 2 decimals, without trailing zeros. */
function svgNumber(value) {
  return String(Math.round(value * 100) / 100);
}

/**
 * Compute the SVG paths that carve text in rune letters.
 *
 * The letters start at (0, 0), the top-left of the cap height. Stroke width is
 * 1.8 units per 10 units of cap height; tracking is the gap between letters in
 * the same units. A character missing from RUNE_LETTERS is a space. Pure: the
 * same input always gives the same strings. Use it for chrome only (header,
 * Title Card); it is never drawn on the battlefield.
 *
 * @param {string} text Letters to carve, for example "MARL-BGs".
 * @param {number} [cap=11] Cap height in CSS pixels; must be positive.
 * @param {number} [tracking=2] Letter gap in cap-10 units.
 * @returns {Readonly<{
 *   strokePath: string,
 *   lozengePath: string,
 *   width: number,
 *   height: number,
 *   strokeWidth: number,
 *   keylineWidth: number,
 * }>} strokePath holds every chisel stroke (draw with no fill); lozengePath
 *   holds the filled lozenges (empty string when none); width and height are
 *   the letters' box in px (height equals cap); strokeWidth is the cut width
 *   and keylineWidth the dark keyline drawn under it.
 * @throws {RangeError} When cap is not a positive finite number.
 */
export function runeWordmark(text, cap = 11, tracking = 2) {
  if (!Number.isFinite(cap) || cap <= 0) {
    throw new RangeError("Rune cap height must be a positive number.");
  }
  const k = cap / 10;
  let cursor = 0;
  /** @type {string[]} */
  const strokes = [];
  /** @type {string[]} */
  const lozenges = [];
  for (const ch of String(text)) {
    const [advance, lines, closed, extras] = RUNE_LETTERS[ch] ?? RUNE_LETTERS[" "];
    lines.forEach((line, index) => {
      const points = line.map(
        ([u, v]) => `${svgNumber(cursor + u * k)} ${svgNumber(v * k)}`,
      );
      strokes.push(`M${points.join("L")}${closed?.[index] ? "Z" : ""}`);
    });
    for (const [u, v, halfWidth, halfHeight] of extras ?? []) {
      const cx = cursor + u * k;
      const cy = v * k;
      const corners = [
        [cx, cy - halfHeight * k],
        [cx + halfWidth * k, cy],
        [cx, cy + halfHeight * k],
        [cx - halfWidth * k, cy],
      ].map(([x, y]) => `${svgNumber(x)} ${svgNumber(y)}`);
      lozenges.push(`M${corners.join("L")}Z`);
    }
    cursor += (advance + tracking) * k;
  }
  const strokeWidth = 1.8 * k;
  return Object.freeze({
    strokePath: strokes.join(""),
    lozengePath: lozenges.join(""),
    width: Math.max(0, cursor - tracking * k),
    height: cap,
    strokeWidth,
    keylineWidth: strokeWidth + Math.max(1.5, k * 0.5),
  });
}

/**
 * Carve text into an empty SVG element as the rune-cut wordmark.
 *
 * Replaces svg's children with a dark keyline under Moon Silver cuts (colors
 * come from src/styles/shell.css classes rune-wordmark__*), and sets its
 * width, height and viewBox so the letters draw at 1 CSS px per unit with room
 * for the keyline. Export-safe SVG: flat strokes and fills only. Side effects:
 * DOM only. It reads no frame, so it is the same in Oracle View and Agent POV.
 *
 * @param {SVGSVGElement} svg Target element; its children are replaced.
 * @param {string} [text=BRAND_NAME] Letters to carve.
 * @param {number} [cap=11] Cap height in CSS pixels.
 * @returns {void}
 */
export function renderRuneWordmark(svg, text = BRAND_NAME, cap = 11) {
  const mark = runeWordmark(text, cap);
  const margin = Math.ceil(mark.keylineWidth / 2);
  const width = Math.ceil(mark.width) + margin * 2;
  const height = Math.ceil(mark.height) + margin * 2;
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `${-margin} ${-margin} ${width} ${height}`);
  const namespace = "http://www.w3.org/2000/svg";
  const doc = svg.ownerDocument;
  /** @param {string} className @param {string} d @param {Record<string, string>} attributes */
  const path = (className, d, attributes) => {
    const node = doc.createElementNS(namespace, "path");
    node.setAttribute("class", className);
    node.setAttribute("d", d);
    for (const [name, value] of Object.entries(attributes))
      node.setAttribute(name, value);
    return node;
  };
  const cutStyle = {
    "stroke-linejoin": "miter",
    "stroke-miterlimit": "4",
    "stroke-linecap": "butt",
  };
  const nodes = [
    path("rune-wordmark__keyline", mark.strokePath, {
      ...cutStyle,
      "stroke-width": svgNumber(mark.keylineWidth),
    }),
  ];
  if (mark.lozengePath) {
    nodes.push(
      path("rune-wordmark__keyline-lozenge", mark.lozengePath, {
        "stroke-width": svgNumber(mark.keylineWidth - mark.strokeWidth),
      }),
    );
  }
  nodes.push(
    path("rune-wordmark__cut", mark.strokePath, {
      ...cutStyle,
      "stroke-width": svgNumber(mark.strokeWidth),
    }),
  );
  if (mark.lozengePath)
    nodes.push(path("rune-wordmark__lozenge", mark.lozengePath, {}));
  svg.replaceChildren(...nodes);
}

/**
 * Write the product's names into the browser tab, the header and the Help
 * dialog, and carve the header wordmark.
 *
 * main.js calls this once per validated launcher identity. "replay_viewer"
 * gets the tab title "MARL-BattleGrounds Replay Viewer", the header name
 * "Replay Viewer", the Help heading "Replay Viewer Help" and a "Help" button;
 * "combat_debugger" gets "MARL-BattleGrounds BattleClient", "BattleClient",
 * the heading "How to Play" and a "How to Play" button. Both buttons show the
 * ? keycap. The Exit label is painted later by renderSessionToolbar. Side
 * effects: sets `document.title`, the text of both elements, the Help button
 * (when `elements.helpButton` is given) and, when the page has an empty
 * `#brand-wordmark` SVG, carves "MARL-BGs" into it. For "combat_debugger" it
 * also removes the WebP choice from the top bar's emblem `<picture>`, so the
 * emblem shows its 64 px PNG, the same picture the approved vision page draws
 * (the lossy WebP copy differs from it by up to 42 color levels at the edges;
 * the browser may already have fetched the 4 KB WebP once). The Replay Viewer
 * keeps the WebP. It reads no frame, so it is the same in Oracle View and
 * Agent POV.
 *
 * @param {{appTitle: HTMLElement, helpHeading: HTMLElement,
 *   helpButton?: HTMLElement}} elements The header product name (#app-title),
 *   the Help dialog heading and, optionally, the Help button.
 * @param {"combat_debugger" | "replay_viewer"} productKind The validated
 *   launcher product.
 * @returns {void}
 */
export function applyProductHeader(elements, productKind) {
  const replay = productKind === "replay_viewer";
  document.title = PRODUCT_TITLES[productKind];
  elements.appTitle.textContent = PRODUCT_NAMES[productKind];
  elements.helpHeading.textContent = replay ? "Replay Viewer Help" : "How to Play";
  const helpButton = elements.helpButton;
  if (helpButton && typeof helpButton.ownerDocument?.createElement === "function") {
    const keycap = helpButton.ownerDocument.createElement("kbd");
    keycap.textContent = "?";
    helpButton.replaceChildren(replay ? "Help " : "How to Play ", keycap);
    helpButton.dataset.product = productKind;
  }
  const wordmark =
    typeof document.getElementById === "function"
      ? document.getElementById("brand-wordmark")
      : null;
  if (
    wordmark !== null &&
    typeof SVGSVGElement !== "undefined" &&
    wordmark instanceof SVGSVGElement &&
    wordmark.childElementCount === 0
  ) {
    renderRuneWordmark(wordmark);
  }
  if (!replay && typeof document.querySelectorAll === "function") {
    for (const source of document.querySelectorAll(
      ".bc-top .brand picture > source[type='image/webp']",
    )) {
      source.remove();
    }
  }
}

/**
 * Tale titles of the eight built-in scenarios, keyed by scenario name (the
 * names in the task manifest). The browser owns this table; the host sends no
 * tale title.
 *
 * @type {Readonly<Record<string, string>>}
 */
export const SCENARIO_TALE_TITLES = Object.freeze({
  scenario_1_HRP_vs_MRP: "The Last Mending",
  scenario_2_WP_vs_MR: "The Physic Garden",
  scenario_3_WH_vs_W: "The Living Wall",
  scenario_4_WR_vs_MWP: "The Gambit",
  scenario_5_MWP_vs_MRP: "Turning the Tide",
  scenario_6_MHP_vs_MWR: "The Lure",
  scenario_7_MWP_vs_MHR: "The Turning Moon",
  scenario_8_WRP_vs_RP: "The Waiting Hour",
});

/**
 * Return the tale title and number of a built-in scenario name.
 *
 * Accepts the manifest names ("scenario_1_HRP_vs_MRP"), "scenario_N", the
 * live scenario names "tdm_scenario_N" and the scenario map layout names the
 * BattleClient records ("tdm-scenario-N-map", the layout identifier a packaged
 * scenario's live game and its replay carry as the map's technical name), for
 * N from 1 to 8. Pure.
 *
 * @param {unknown} name A scenario or layout name.
 * @returns {Readonly<{title: string, number: number}> | null} The tale title
 *   and scenario number, or null for anything else.
 */
export function scenarioTale(name) {
  if (typeof name !== "string") return null;
  const match = /^(?:tdm[_-])?scenario[_-]([1-8])(?:[_-]|$)/u.exec(name);
  if (match === null) return null;
  const number = Number(match[1]);
  const key = Object.keys(SCENARIO_TALE_TITLES)[number - 1];
  return Object.freeze({ title: SCENARIO_TALE_TITLES[key], number });
}

/**
 * Build the place title for the top bar from a frame's match summary.
 *
 * A built-in scenario reads "The Last Mending · Scenario 1"; a registered map
 * reads "Teutoburg Forest · Map 47"; a map with no registered ID reads its
 * display name alone ("Custom Map"). The map is public (both teams play on
 * it), so Agent POV shows the same title. Pure.
 *
 * @param {unknown} matchSummary `presentation.match_summary`, or anything else.
 * @returns {string | null} The title, or null when the summary has no usable map.
 */
export function matchPlaceTitle(matchSummary) {
  const parts = matchPlaceParts(matchSummary);
  if (parts === null) return null;
  return [parts.name, ...placeLineSplit(parts).lead].join(" · ");
}

/** What the top bar's place line says for task mode 1, Team Deathmatch. */
const TEAM_DEATHMATCH_LABEL = "Team Deathmatch";

/**
 * Split the top bar's place line into its bold name and the plain parts that
 * follow it, as the vision writes "**Teutoburg Forest** · Map 47 · Team
 * Deathmatch".
 *
 * name is the scenario's tale title or the map's display name. details lists,
 * in order, "Scenario N" (a built-in scenario) or "Map N" (a map with a
 * registered ID; none for a map without one, such as "Custom Map"), then
 * "Team Deathmatch" when the summary's task_mode is 1. The map is public (both
 * teams play on it), so Agent POV shows the same line. Pure.
 *
 * @param {unknown} matchSummary `presentation.match_summary`, or anything else.
 * @returns {Readonly<{name: string, details: ReadonlyArray<string>}> | null} The
 *   parts, or null when the summary has no usable map.
 */
export function matchPlaceParts(matchSummary) {
  if (matchSummary === null || typeof matchSummary !== "object") return null;
  const summary = /** @type {Record<string, unknown>} */ (matchSummary);
  const map = summary.map;
  if (map === null || typeof map !== "object") return null;
  const record = /** @type {Record<string, unknown>} */ (map);
  const tale = scenarioTale(record.technical_name) ?? scenarioTale(record.display_name);
  /** @type {string[]} */
  const details = [];
  let name = "";
  if (tale !== null) {
    name = tale.title;
    details.push(`Scenario ${tale.number}`);
  } else {
    if (typeof record.display_name !== "string" || record.display_name === "")
      return null;
    name = record.display_name;
    if (Number.isInteger(record.map_id)) details.push(`Map ${record.map_id}`);
  }
  if (summary.task_mode === 1) details.push(TEAM_DEATHMATCH_LABEL);
  return Object.freeze({ name, details: Object.freeze(details) });
}

/**
 * Split the place line's plain details into the part that names the place
 * and the part that may give way.
 *
 * lead keeps the details that say where the match is ("Map 47" or
 * "Scenario 1"; empty for a map without a registered ID). The top bar always
 * shows the bold name and lead whole, for example "Teutoburg Forest · Map
 * 47" (owner, 3 Oct 2026: the full map name at every supported size and in
 * every phase). more keeps the rest ("Team Deathmatch"), which the top-bar
 * styles cut with "…" first when the bar runs short. Both keep their order
 * in parts.details. Pure.
 *
 * @param {Readonly<{name: string, details: ReadonlyArray<string>}>} parts
 *   matchPlaceParts' result.
 * @returns {Readonly<{lead: ReadonlyArray<string>, more: ReadonlyArray<string>}>}
 *   New frozen lists.
 */
export function placeLineSplit(parts) {
  return Object.freeze({
    lead: Object.freeze(
      parts.details.filter((detail) => detail !== TEAM_DEATHMATCH_LABEL),
    ),
    more: Object.freeze(
      parts.details.filter((detail) => detail === TEAM_DEATHMATCH_LABEL),
    ),
  });
}

/**
 * Whether the top bar cuts the place line's second part (span.where__more,
 * " · Team Deathmatch"): its text is wider than its box. A cut part is then
 * hidden whole rather than shown ending in "…" (owner, 3 Oct 2026).
 *
 * Pure.
 *
 * @param {Readonly<{scrollWidth: number, clientWidth: number}>} box The
 *   span's text width (scrollWidth) and box width (clientWidth), in CSS
 *   pixels. Half a pixel of rounding is not a cut.
 * @returns {boolean} True when the text does not fit its box.
 */
export function placeRestIsCut(box) {
  return box.scrollWidth > box.clientWidth + 0.5;
}

/**
 * Whether any champion of this game is played by hand ("Manual").
 *
 * Reads the transport frame's combat configuration: a nonempty manual_slots
 * list, or (older configurations without per-slot controllers) a team
 * controller named "manual". A game with no Manual champion is Policies Only:
 * the human watches. Pure.
 *
 * @param {unknown} combatConfiguration `transport.combat_configuration`.
 * @returns {boolean} True when at least one champion is Manual; false for
 *   Policies Only and for a missing configuration.
 */
export function hasManualChampion(combatConfiguration) {
  if (!isPlainRecord(combatConfiguration)) return false;
  if (Array.isArray(combatConfiguration.manual_slots)) {
    return combatConfiguration.manual_slots.length > 0;
  }
  return (
    combatConfiguration.team_a_controller === "manual" ||
    combatConfiguration.team_b_controller === "manual"
  );
}

/**
 * Build the arena's view tag (#pov-tag) for live play, as the vision writes it.
 *
 * Oracle View: lead "Oracle View", detail "· You See Everything" (the page
 * paints it amber). Agent POV: lead "Agent POV", detail "· Your Team's Sight ·
 * N Unseen", or "· Team A's Sight · N Unseen" when nobody is played by hand
 * (Policies Only; the team is the Agent POV recipient's team). N is, for each
 * team, the number of its configured champions that are alive (spawn pads
 * with configured_active and currently_alive; the Agent POV scene gives enemy
 * pads no identity) less the number of its living bodies in the scene. It
 * reads only the audience's own arena scene (authorizedPresentationSceneView):
 * never researcher_space or the match summary. Pure.
 *
 * @param {{
 *   product: unknown,
 *   audience: unknown,
 *   scene: Readonly<Record<string, any>> | null,
 *   recipientPublicAgentId: unknown,
 *   watchOnly: boolean,
 * }} input product is the launcher product ("combat_debugger" shows a tag);
 *   audience is "researcher" or "agent_pov"; scene is the audience scene view;
 *   recipientPublicAgentId is the Agent POV recipient's public ID; watchOnly
 *   is true in a game with no Manual champion.
 * @returns {Readonly<{view: "pov" | "oracle", lead: string, detail: string}> | null}
 *   The tag, or null where no tag shows (the Replay Viewer, no frame, or an
 *   unknown audience).
 */
export function liveViewTag(input) {
  if (input.product !== "combat_debugger") return null;
  if (input.audience === "researcher") {
    return Object.freeze({
      view: /** @type {const} */ ("oracle"),
      lead: "Oracle View",
      detail: "· You See Everything",
    });
  }
  if (input.audience !== "agent_pov" || input.scene === null) return null;
  const agents = Array.isArray(input.scene.agents) ? input.scene.agents : [];
  const pads = Array.isArray(input.scene.spawn_pads) ? input.scene.spawn_pads : [];
  let unseen = 0;
  for (const team of [1, 2]) {
    const alive = pads.filter(
      (pad) =>
        isPlainRecord(pad) &&
        pad.team_id === team &&
        pad.configured_active === true &&
        pad.currently_alive === true,
    ).length;
    const drawn = agents.filter(
      (agent) =>
        isPlainRecord(agent) && agent.team_id === team && agent.life_state !== "corpse",
    ).length;
    unseen += Math.max(0, alive - drawn);
  }
  const recipient = agents.find(
    (agent) =>
      isPlainRecord(agent) && agent.public_agent_id === input.recipientPublicAgentId,
  );
  const teamLetter =
    recipient?.team_id === 1 ? "A" : recipient?.team_id === 2 ? "B" : null;
  const whose =
    input.watchOnly && teamLetter !== null ? `Team ${teamLetter}'s` : "Your Team's";
  return Object.freeze({
    view: /** @type {const} */ ("pov"),
    lead: "Agent POV",
    detail: `· ${whose} Sight · ${unseen} Unseen`,
  });
}

/**
 * Build the Champion Card facts of one champion, from the audience's own
 * arena scene. One builder for live play (the BattleClient and the DevClient
 * combat screen show it in #hcard) and the Replay Viewer (its arena bodies
 * show it through the tooltip card; `main.js` asks for it).
 *
 * The champion is found by the hovered element's subject: an arena body's
 * presentation key, a public ID, or a portrait's or row's global slot (the
 * shown Agent ID, matched against each body's display_agent_id). A champion
 * with a body in the scene gets the full card (tooltip.js
 * championCardFactsFromAgent: life, Ultimate and every status, named by class
 * only, the same in Agent POV and Oracle View), with Freezing Trap (Stun
 * Broken) when the audience's own events this tick (its
 * "status_broken_by_damage" rows) show its Freezing Trap broke. In live play
 * only, a champion with no body is "unseen" in Agent POV when the subject's
 * slot names a configured spawn pad (team and team slot): the reduced card
 * with its Agent ID, its name and calling (the class comes from the identity
 * directory's identity fields, publicRosterIdentities in `match-summary.js`,
 * or else from the element's data-class-id; class is public), and never
 * life, place, Ultimate or status. In Oracle View, or in the Replay Viewer, a
 * champion with no body gets no card. A fallen champion's card ends with
 * "Fallen · Returns in N" from the scene's public respawn_waves clock (the
 * same note the Replay Viewer's frames show).
 *
 * Who plays it: live play reads the transport frame's combat configuration
 * ("You play this champion" for a manual_slots slot, else its slot_labels
 * label); the Replay Viewer reads the champion's recorded controller name
 * (the match summary's slot_controller_names, at the slot the identity
 * directory gives the champion), or, in an old replay without it, the team's
 * display name. "Your Target" marks the
 * target of the order being set (live play's draft_target). Of
 * researcher_space it reads only the identity fields that
 * publicRosterIdentities copies (public ID, team, team slot and class: the
 * same public facts the scorebug portraits show), and only for an unseen
 * champion's class. Pure.
 *
 * @param {{
 *   presentation: Readonly<Record<string, any>> | null,
 *   combatConfiguration: unknown,
 *   subject: import("./tooltip.js").HoverCardSubject,
 * }} input presentation is the installed presentation (live or replay);
 *   combatConfiguration is `transport.combat_configuration` in live play (any
 *   value, for example null, in the Replay Viewer); subject comes from
 *   hoverCardSubject (or names a presentation key).
 * @returns {import("./tooltip.js").ChampionCardFacts | null} The facts, or null
 *   when there is no presentation or no such champion to describe.
 */
export function championCardFacts(input) {
  const { presentation, subject } = input;
  if (presentation === null) return null;
  const live = presentation.viewer_mode === "live";
  if (!live && presentation.viewer_mode !== "replay") return null;
  const audience = authorizedPresentationAudience(presentation);
  if (audience !== "researcher" && audience !== "agent_pov") return null;
  const scene = authorizedPresentationSceneView(presentation);
  if (scene === null) return null;
  const agents = (Array.isArray(scene.agents) ? scene.agents : []).filter(
    isPlainRecord,
  );
  const pads = (Array.isArray(scene.spawn_pads) ? scene.spawn_pads : []).filter(
    isPlainRecord,
  );
  /** @param {Record<string, any>} pad */
  const padSlot = (pad) =>
    (pad.team_id === 1 || pad.team_id === 2) && Number.isInteger(pad.team_local_slot)
      ? (pad.team_id - 1) * 5 + pad.team_local_slot
      : null;
  /** @param {Record<string, any>} row @returns {number | null} */
  const shownSlot = (row) =>
    /^\d$/u.test(String(row.display_agent_id ?? ""))
      ? Number(row.display_agent_id)
      : null;
  const agent =
    agents.find(
      (row) =>
        (subject.presentationKey !== null &&
          row.presentation_key === subject.presentationKey) ||
        (subject.publicAgentId !== null &&
          row.public_agent_id === subject.publicAgentId) ||
        (subject.slot !== null && shownSlot(row) === subject.slot),
    ) ?? null;
  const pad =
    live && agent === null && subject.slot !== null
      ? (pads.find((row) => padSlot(row) === subject.slot) ?? null)
      : null;
  if (agent === null && (audience !== "agent_pov" || pad === null)) return null;
  const publicId = agent?.public_agent_id ?? null;
  const slot = agent !== null ? shownSlot(agent) : padSlot(/** @type {any} */ (pad));
  const teamId = agent?.team_id ?? pad?.team_id;
  const config = isPlainRecord(input.combatConfiguration)
    ? input.combatConfiguration
    : {};
  const you =
    live && slot !== null && Array.isArray(config.manual_slots)
      ? config.manual_slots.includes(slot)
      : false;
  const teams = Array.isArray(presentation.match_summary?.teams)
    ? presentation.match_summary.teams
    : [];
  // A replay's recorded per-champion names, ten in global slot order; the
  // endpoint's identity directory (identity fields only) gives the slot.
  const recorded = presentation.match_summary?.slot_controller_names;
  const directory = presentation.current_endpoint?.identity_directory;
  const identity = Array.isArray(directory?.identities)
    ? directory.identities.find(
        (/** @type {Record<string, any>} */ row) =>
          publicId !== null && row?.public_agent_id === publicId,
      )
    : undefined;
  const globalSlot =
    (identity?.team_id === 1 || identity?.team_id === 2) &&
    Number.isInteger(identity.team_local_slot)
      ? (identity.team_id - 1) * 5 + identity.team_local_slot
      : null;
  const label = live
    ? slot !== null && Array.isArray(config.slot_labels)
      ? config.slot_labels[slot]
      : null
    : Array.isArray(recorded) &&
        globalSlot !== null &&
        typeof recorded[globalSlot] === "string"
      ? recorded[globalSlot]
      : teamId === 1 || teamId === 2
        ? teams[teamId - 1]?.display_name
        : null;
  const playedBy = typeof label === "string" && label !== "" ? label : null;
  if (agent === null) {
    // The class is public: the identity directory gives it for every
    // champion, so every hover owner (a portrait, a rail row, an Allowed Now
    // button) names it, whether or not the element carries data-class-id.
    const listed = publicRosterIdentities(presentation).find(
      (row) => (row.teamId - 1) * 5 + row.teamLocalSlot === slot,
    )?.classId;
    const classId = [1, 2, 3, 4, 5].includes(listed ?? 0)
      ? /** @type {number} */ (listed)
      : subject.classId;
    const names = classNamesV1(classId);
    const recipientId = isPlainRecord(presentation.authority)
      ? presentation.authority.recipient_public_agent_id
      : null;
    const viewerTeam = agents.find(
      (row) => row.public_agent_id === recipientId,
    )?.team_id;
    return Object.freeze({
      state: /** @type {const} */ ("unseen"),
      name: names?.className ?? null,
      calling: names?.calling ?? null,
      classId: /** @type {1 | 2 | 3 | 4 | 5 | null} */ (
        [1, 2, 3, 4, 5].includes(classId ?? 0) ? classId : null
      ),
      displayId: slot === null ? null : String(slot),
      teamId: teamId === 1 || teamId === 2 ? teamId : null,
      life: null,
      ultimate: null,
      statuses: Object.freeze([]),
      note: null,
      you,
      playedBy,
      yourTarget: false,
      unseenBy:
        !hasManualChampion(config) && (viewerTeam === 1 || viewerTeam === 2)
          ? `Team ${viewerTeam === 1 ? "A" : "B"}`
          : null,
    });
  }
  const trapBroke = authorizedPresentationIncomingRows(presentation).some(
    (row) =>
      row.vocabulary === "event" &&
      row.kind === "status_broken_by_damage" &&
      row.payload?.status_id === "hunter_trap_stun" &&
      row.payload?.recipient_anchor?.public_agent_id === publicId,
  );
  // A live editable draft's inspection is the draft itself; older shapes
  // nest it under "draft".
  const inspection = live
    ? authorizedPresentationInspectionState(presentation).inspection
    : null;
  const target = (inspection?.draft ?? inspection)?.draft_target;
  const facts = championCardFactsFromAgent(agent, {
    classMechanics: Array.isArray(scene.class_mechanics) ? scene.class_mechanics : [],
    spawnShieldSteps: scene.spawn_shield_mechanics?.configured_duration_steps ?? null,
    trapBroke,
    respawnWaves: Array.isArray(scene.respawn_waves) ? scene.respawn_waves : [],
    you,
    playedBy,
    yourTarget: isPlainRecord(target) && target.target_public_agent_id === publicId,
  });
  // Live play shows the slot as the Agent ID even when the row has no
  // display ID of its own.
  return facts.displayId === null && slot !== null
    ? Object.freeze({ ...facts, displayId: String(slot) })
    : facts;
}

/**
 * Whether value is a non-null, non-array object. Pure.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isPlainRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * The Oracle View button's hover and focus help, in every client.
 */
export const ORACLE_VIEW_BUTTON_HELP =
  "Shows every champion, even those your team cannot see.";

/**
 * Return the Agent POV button's hover and focus help for a product. Only the
 * BattleClient opens in Agent POV (the vision's View row: "Agent POV is the
 * default in the BattleClient"), so only its help adds "and the default here";
 * the DevClient combat screen and the Replay Viewer open in Oracle View. The
 * words depend on the product only, never on the view the page loaded in. Pure.
 *
 * @param {{replay: boolean, devClient: boolean}} product replay is true in the
 *   Replay Viewer; devClient is true in the private DevClient (its Developer
 *   drawer is shown).
 * @returns {string}
 */
export function agentPovButtonTitle(product) {
  return !product.replay && !product.devClient
    ? "Shows only what your team can see. The fair view for play, and the default here."
    : "Shows only what your team can see. The fair view for play.";
}

/**
 * Register one View switch button's hover and focus help: a registered
 * "control" tip (the shared tooltip controller), never a native title. A
 * second call replaces the first. Side effects: the button's tooltip
 * registration, aria-description and data-tooltip-owner.
 *
 * @param {Element} button A `#view-switch` button.
 * @param {string} title The tip's heading, the button's own words ("Agent POV").
 * @param {string} summary The one plain sentence.
 * @returns {void}
 */
function registerViewButtonHelp(button, title, summary) {
  registerTooltipOwner(
    button,
    createSemanticDescriptor({
      kind: "control",
      id: `control:view-switch:${title}`,
      title,
      tone: "information",
      accent: "none",
      summary,
      rows: [],
      sections: [],
      metadata: { compact: true, full: false },
      anchor: "element",
    }),
    { inspectable: false },
  );
}

/**
 * Choose the connection pill's words and state.
 *
 * In this order of priority: "Closed" once Exit was accepted (the server is
 * closing and the page stops reading, so the pill never waits on a later
 * answer; the Replay Viewer a BattleClient kept open has its own pill); while
 * a request is in flight "Submit Queued", "Input Queued" or "Command in
 * Flight", except autoplay's own next-frame fetch (status.quietBusy), which
 * keeps the pill as it was, so it does not blink before every played tick;
 * then
 * "Disconnected" (the page lost the local server, so it must reload its
 * frame too), "Resync Required" (the server is there but the page's frame is
 * stale), "Connecting" (no frame yet) and otherwise "Connected". Pure.
 *
 * @param {{busy: boolean, quietBusy?: boolean, offline: boolean,
 *   resyncRequired: boolean, shuttingDown: boolean, hasFrame: boolean}} status
 *   Page flags. quietBusy (default false): the request in flight is
 *   autoplay's own fetch; a press the user made always shows "Command in
 *   Flight".
 * @param {Readonly<{deferredDrafts: ReadonlyArray<unknown>, deferredSubmit: unknown}> | null}
 *   transaction The live command in flight, or null (its queued order keys,
 *   oldest first, and its queued Lock In).
 * @returns {Readonly<{label: string, status: "online" | "busy" | "offline" | "loading"}>}
 *   The pill text and its `data-state` value.
 */
export function connectionPill(status, transaction) {
  if (status.shuttingDown) return Object.freeze({ label: "Closed", status: "offline" });
  if (status.busy && status.quietBusy !== true) {
    return Object.freeze({
      label: transaction?.deferredSubmit
        ? "Submit Queued"
        : (transaction?.deferredDrafts.length ?? 0) > 0
          ? "Input Queued"
          : "Command in Flight",
      status: "busy",
    });
  }
  if (status.offline) {
    return Object.freeze({ label: "Disconnected", status: "offline" });
  }
  if (status.resyncRequired) {
    return Object.freeze({ label: "Resync Required", status: "offline" });
  }
  if (!status.hasFrame)
    return Object.freeze({ label: "Connecting", status: "loading" });
  return Object.freeze({ label: "Connected", status: "online" });
}

/**
 * One agent in the Replay Viewer's POV agent picker.
 *
 * @typedef {Readonly<{
 *   publicAgentId: string,
 *   label: string,
 *   classLetter: string,
 *   accent: string,
 *   teamId: 1 | 2,
 *   current: boolean,
 *   command: Readonly<{command_type: "activate_replay_pov_agent",
 *     global_slot: number, public_agent_id: string}>,
 * }>} PovAgentChoice
 */

/**
 * Build the Replay Viewer's POV agent chip and picker from identity rows.
 *
 * The chip names the agent whose view the arena shows, for example "Hunter ·
 * Agent 2". The picker lists every agent that can become the POV agent (rows
 * with activation kind "replay_pov_global" and a command slot), Team A first,
 * then Team B, each in row order. Its command is the same one a roster row
 * click sends. Identity only (class, ID, team): no position, life or status.
 * Use it only in the Replay Viewer, whose researcher chrome lists every
 * agent; the BattleClient and DevClient never show it. Pure.
 *
 * @param {unknown} recipientPublicAgentId The POV agent's public ID
 *   (`presentation.authority.recipient_public_agent_id`).
 * @param {ReadonlyArray<Readonly<Record<string, any>>>} rows Rows from
 *   authorizedPresentationIdentityRows for the same presentation.
 * @returns {Readonly<{
 *   label: string,
 *   classLetter: string,
 *   accent: string,
 *   teamId: 1 | 2,
 *   teams: ReadonlyArray<ReadonlyArray<PovAgentChoice>>,
 * }> | null} The chip and its two picker rows (Team A, Team B), or null when
 *   the recipient is missing or has no valid identity.
 */
export function povAgentChipView(recipientPublicAgentId, rows) {
  if (typeof recipientPublicAgentId !== "string") return null;
  /** @type {PovAgentChoice[][]} */
  const teams = [[], []];
  let chip = null;
  for (const row of Array.isArray(rows) ? rows : []) {
    const identity = exactAuthorizedAgentIdentityV1(row?.agent);
    if (identity === null) continue;
    const classLetter = identity.classLabel.slice(0, 1);
    const label = `${identity.classLabel} · ${identity.chipText}`;
    const current = identity.publicAgentId === recipientPublicAgentId;
    if (current) {
      chip = { label, classLetter, accent: identity.accent, teamId: identity.teamId };
    }
    if (
      row.activation_kind !== "replay_pov_global" ||
      !Number.isInteger(row.command_global_slot)
    ) {
      continue;
    }
    teams[identity.teamId - 1].push(
      Object.freeze({
        publicAgentId: identity.publicAgentId,
        label,
        classLetter,
        accent: identity.accent,
        teamId: identity.teamId,
        current,
        command: Object.freeze({
          command_type: "activate_replay_pov_agent",
          global_slot: /** @type {number} */ (row.command_global_slot),
          public_agent_id: identity.publicAgentId,
        }),
      }),
    );
  }
  if (chip === null) return null;
  return Object.freeze({
    ...chip,
    teams: Object.freeze(teams.map((team) => Object.freeze(team))),
  });
}

/**
 * Return where the live Visual Filters drawer starts: the top of the rail's
 * first card, in CSS px from the top of the window.
 *
 * railTop is the rail's (#rail) top edge from getBoundingClientRect, which
 * is the scorebug's bottom edge; railPaddingTop is the rail's top padding in
 * CSS px (12 in live play). So the drawer lines up with the rail's cards and
 * never covers the top bar or the scorebug. Pure.
 *
 * @param {number} railTop The rail's top edge in CSS px.
 * @param {number} railPaddingTop The rail's top padding in CSS px; a missing
 *   or bad value counts as 0.
 * @returns {number | null} The drawer's top in whole CSS px (0 or more), or
 *   null when railTop is not a finite number (the CSS fallback then holds).
 */
export function visualFiltersDrawerTop(railTop, railPaddingTop) {
  if (!Number.isFinite(railTop)) return null;
  const padding = Number.isFinite(railPaddingTop) ? Math.max(0, railPaddingTop) : 0;
  return Math.max(0, Math.round(railTop + padding));
}

/**
 * Place a popover just under the button that opened it, kept inside the window.
 *
 * Aligns the popover's right edge with the anchor's right edge when the anchor
 * sits in the right half of the window, else its left edge with the anchor's
 * left edge, then keeps a 16 px gap from both window sides. Pure.
 *
 * @param {{left: number, right: number, bottom: number}} anchor The opening
 *   button's box in CSS px (from getBoundingClientRect).
 * @param {number} popoverWidth The popover's width in CSS px.
 * @param {number} viewportWidth The window's width in CSS px.
 * @returns {Readonly<{top: number, left: number}>} Fixed-position offsets in CSS px.
 */
export function popoverPlacement(anchor, popoverWidth, viewportWidth) {
  const gap = 16;
  const preferred =
    anchor.left + (anchor.right - anchor.left) / 2 > viewportWidth / 2
      ? anchor.right - popoverWidth
      : anchor.left;
  const left = Math.max(gap, Math.min(preferred, viewportWidth - gap - popoverWidth));
  return Object.freeze({ top: Math.round(anchor.bottom + 6), left: Math.round(left) });
}

/**
 * Return the key rows Help shows for one product, from the one command set.
 *
 * One `[keycaps, help]` pair per command-set row that has a job in product, in
 * table order, for example `["W A S D Q E Z C", "Step (staged)"]`. A key with no
 * job in that product is left out (B in the BattleClient, the arrows in live
 * play). Pure: reads only the command table.
 *
 * @param {"combat_debugger" | "replay_viewer"} product "combat_debugger" for the
 *   BattleClient and the DevClient combat screen, "replay_viewer" for the
 *   Replay Viewer.
 * @returns {ReadonlyArray<readonly [string, string]>} A new frozen list of
 *   frozen pairs.
 * @throws {RangeError} When product is not a known product.
 */
export function commandHelpEntries(product) {
  return Object.freeze(
    commandHelpRows(product).map(({ keycaps, help }) =>
      Object.freeze(/** @type {const} */ ([keycaps, help])),
    ),
  );
}

/**
 * Write the key lists of the Help dialog from the one command set.
 *
 * Every `<dl data-command-help="combat_debugger">` or
 * `<dl data-command-help="replay_viewer">` under root gets one `<dt>` (the
 * keycaps) and `<dd>` (the job) per commandHelpEntries row of that product; its
 * old children are replaced. Lists with any other value are left alone. Call
 * once at startup; the table never changes while the page runs. Side effects:
 * DOM only. It reads no frame, so it is the same in Oracle View and Agent POV.
 *
 * @param {ParentNode} root Where to look, normally `document`.
 * @returns {number} How many lists were written.
 */
export function renderCommandHelpLists(root) {
  let written = 0;
  for (const list of root.querySelectorAll("dl[data-command-help]")) {
    if (!(list instanceof HTMLElement)) continue;
    const product = list.dataset.commandHelp;
    if (product !== "combat_debugger" && product !== "replay_viewer") continue;
    const doc = list.ownerDocument;
    const fragment = doc.createDocumentFragment();
    for (const [keycaps, help] of commandHelpEntries(product)) {
      const term = doc.createElement("dt");
      term.textContent = keycaps;
      const detail = doc.createElement("dd");
      detail.textContent = help;
      fragment.append(term, detail);
    }
    list.replaceChildren(fragment);
    written += 1;
  }
  return written;
}

/**
 * The four rule cards of the How to Play overlay, in order, word for word from
 * the approved vision page: [title, sentence].
 *
 * @type {ReadonlyArray<readonly [string, string]>}
 */
export const HELP_OVERLAY_RULES = Object.freeze([
  Object.freeze(
    /** @type {const} */ ([
      "1 · Everyone Decides at Once",
      "The game moves in ticks. In each tick every champion chooses at the same moment, then the whole tick plays at once. Nobody sees anyone else's choice first. Nothing moves until you lock in, and there is no timer.",
    ]),
  ),
  Object.freeze(
    /** @type {const} */ ([
      "2 · One Step, One Action",
      "Each tick your champion takes one step, or stays. It also does at most one action: its Basic or its Ultimate, on one target. Or it picks No Combat and only moves.",
    ]),
  ),
  Object.freeze(
    /** @type {const} */ ([
      "3 · No Facing",
      "Champions never face a way. Each one sees all around itself, 6 units (the Hunter 6.5). Stones block sight and bodies. Other champions do not block sight. Your team shares what it sees.",
    ]),
  ),
  Object.freeze(
    /** @type {const} */ ([
      "4 · Points and Winning",
      "A knocked-out champion gives the other team 1 point, or 2 if it fell inside its own Red Zone. First team to 20 wins. At tick 300 the game is a draw, whatever the score. The fallen come back on the next respawn wave (every 5 ticks), shielded for 3 ticks.",
    ]),
  ),
]);

/**
 * Class name of each class ID, for the rule cards' small tokens. The studs get
 * the class `tok-studs--<name>`; overlays.css paints them with the palette's
 * class token (`--class-<name>`), so this file holds no team or class color.
 */
const RULE_CLASS_NAMES = Object.freeze({
  1: "mage",
  2: "warrior",
  3: "hunter",
  4: "rogue",
  5: "priest",
});

/**
 * Draw a class emblem for a rule card picture, as the vision's emblem symbols
 * draw it.
 *
 * icons.js draws the emblem's dark keyline 3 px wide and the jewel rings 1 px
 * wide at any size (non-scaling strokes). The vision's symbols are cut for a
 * 26 px emblem and scale with it instead: 2.77 and 0.92 units of the 24-unit
 * drawing. So at the rule cards' 12.32 px and 16 px the lines grow thin with
 * the emblem, as in the vision. This function makes that change on the
 * emblem's own paths. Export-safe SVG. Pure apart from creating nodes.
 *
 * @param {Document} doc Creates the nodes.
 * @param {number} classId 1 Mage, 2 Warrior, 3 Hunter, 4 Rogue, 5 Priest.
 * @param {number} x Left edge in the parent SVG's units.
 * @param {number} y Top edge in the parent SVG's units.
 * @param {number} size Width and height in the parent SVG's units.
 * @returns {SVGSVGElement} The emblem, ready to append.
 */
function ruleEmblem(doc, classId, x, y, size) {
  // size 24 picks the standard emblem master, as the vision's symbols use.
  const emblem = createSvgIcon(doc, classTokenFromId(classId).glyphKey, {
    size: 24,
    paint: "class",
  });
  emblem.setAttribute("x", String(x));
  emblem.setAttribute("y", String(y));
  emblem.setAttribute("width", String(size));
  emblem.setAttribute("height", String(size));
  for (const path of emblem.querySelectorAll("[vector-effect='non-scaling-stroke']")) {
    const width = Number(path.getAttribute("stroke-width") ?? "1");
    path.removeAttribute("vector-effect");
    path.setAttribute(
      "stroke-width",
      String(Math.round(((width * 24) / 26) * 100) / 100),
    );
  }
  return emblem;
}

/**
 * Create one SVG element with attributes. Pure apart from creating the node.
 *
 * @param {Document} doc Creates the node.
 * @param {string} tag The SVG tag name.
 * @param {Record<string, string | number>} attributes Attribute values.
 * @returns {SVGElement}
 */
function svgElement(doc, tag, attributes) {
  const node = /** @type {SVGElement} */ (
    doc.createElementNS("http://www.w3.org/2000/svg", tag)
  );
  for (const [name, value] of Object.entries(attributes)) {
    node.setAttribute(name, String(value));
  }
  return node;
}

/**
 * Draw one small champion token of the rule cards (30 px), as the vision draws
 * it: the team's dark body, the class emblem (ruleEmblem), four class-colored
 * studs and the team ring (Team B adds a dark keyline). A fallen token is the
 * vision's ash disc with two faint team rings, a dimmed emblem and three gray
 * vigil pips. Export-safe SVG. Pure apart from creating nodes.
 *
 * Team and class colors come from CSS, not from this file: the rings carry
 * `tok-ring tok-ring--a` or `tok-ring--b` and the studs `tok-studs--<class>`,
 * and overlays.css paints them with the palette tokens (`--team-a`,
 * `--team-b`, `--class-<name>`). So a palette change (for example Full Moon
 * Contrast) reaches the rule cards too.
 *
 * @param {Document} doc Creates the nodes.
 * @param {number} classId 1 Mage, 2 Warrior, 3 Hunter, 4 Rogue, 5 Priest.
 * @param {1 | 2} teamId 1 Team A, 2 Team B.
 * @param {boolean} [fallen] True for the fallen token. Default false.
 * @returns {SVGSVGElement}
 */
function ruleToken(doc, classId, teamId, fallen = false) {
  const svg = /** @type {SVGSVGElement} */ (
    svgElement(doc, "svg", {
      class: "tok",
      width: 30,
      height: 30,
      viewBox: "0 0 30 30",
      "aria-hidden": "true",
    })
  );
  const ring = teamId === 1 ? "tok-ring tok-ring--a" : "tok-ring tok-ring--b";
  const body = teamId === 1 ? "#11264A" : "#3B141B";
  const emblem = ruleEmblem(doc, classId, 8.84, 8.84, 12.32);
  if (fallen) {
    emblem.setAttribute("opacity", "0.55");
    svg.append(
      svgElement(doc, "circle", {
        cx: 15,
        cy: 15,
        r: 13,
        fill: "#1A1F29",
        opacity: 0.75,
      }),
      svgElement(doc, "circle", {
        cx: 15,
        cy: 15,
        r: 13.5,
        class: ring,
        fill: "none",
        "stroke-width": 1,
        opacity: 0.45,
      }),
      svgElement(doc, "circle", {
        cx: 15,
        cy: 15,
        r: 11,
        class: ring,
        fill: "none",
        "stroke-width": 1,
        opacity: 0.45,
      }),
      emblem,
      ...[
        [18.58, 26.03],
        [15, 26.6],
        [11.42, 26.03],
      ].map(([cx, cy]) => svgElement(doc, "circle", { cx, cy, r: 1, fill: "#9AA6BC" })),
    );
    return svg;
  }
  const group = svgElement(doc, "g", { opacity: 1 });
  group.append(
    svgElement(doc, "circle", { cx: 15, cy: 15, r: 12.25, fill: body }),
    svgElement(doc, "circle", { cx: 15, cy: 15, r: 10.5, fill: "#0D1119" }),
    svgElement(doc, "circle", { cx: 15, cy: 15, r: 10.5, fill: body }),
    emblem,
    svgElement(doc, "path", {
      d: "M14.1 7.4a0.9 0.9 0 1 0 1.8 0a0.9 0.9 0 1 0-1.8 0M21.7 15a0.9 0.9 0 1 0 1.8 0a0.9 0.9 0 1 0-1.8 0M14.1 22.6a0.9 0.9 0 1 0 1.8 0a0.9 0.9 0 1 0-1.8 0M6.5 15a0.9 0.9 0 1 0 1.8 0a0.9 0.9 0 1 0-1.8 0",
      class: `tok-studs tok-studs--${
        /** @type {Record<number, string>} */ (RULE_CLASS_NAMES)[classId] ?? "unknown"
      }`,
    }),
    svgElement(doc, "circle", {
      cx: 15,
      cy: 15,
      r: 12.25,
      class: ring,
      fill: "none",
      "stroke-width": 3.5,
    }),
  );
  if (teamId === 2) {
    group.append(
      svgElement(doc, "circle", {
        cx: 15,
        cy: 15,
        r: 12.25,
        fill: "none",
        stroke: "#0A0D14",
        "stroke-width": 1,
      }),
    );
  }
  svg.append(group);
  return svg;
}

/**
 * Build the picture (.ill) of one rule card, as in the vision page.
 *
 * Rule 1: Team A Warrior and Team B Mage, a dot, Team A Hunter and Team B
 * Rogue (everyone decides at once). Rule 2: a 3 by 3 step pad with North lit,
 * "+", the "1 Basic" slot, "or", the "2 Ultimate" slot. Rule 3: a Hunter's
 * dashed sight circle with a stone casting a shadow (no facing). Rule 4: a
 * fallen Team B Rogue "+1", then inside a Red Zone strip marked ×2, another
 * "+2". Pure apart from creating nodes.
 *
 * @param {Document} doc Creates the nodes.
 * @param {number} index The rule, 0 to 3.
 * @returns {HTMLDivElement}
 */
function rulePicture(doc, index) {
  const ill = doc.createElement("div");
  ill.className = "ill";
  /** @param {string} className @param {string} text */
  const span = (className, text) => {
    const node = doc.createElement("span");
    node.className = className;
    node.textContent = text;
    return node;
  };
  /** @param {string} number @param {string} word */
  const slot = (number, word) => {
    const node = doc.createElement("span");
    node.className = "mini-slot";
    node.append(number, doc.createElement("br"), word);
    return node;
  };
  if (index === 0) {
    ill.append(
      ruleToken(doc, 2, 1),
      ruleToken(doc, 1, 2),
      span("plus", "·"),
      ruleToken(doc, 3, 1),
      ruleToken(doc, 4, 2),
    );
  } else if (index === 1) {
    const pad = doc.createElement("span");
    pad.className = "mini-pad";
    for (let key = 0; key < 9; key += 1) {
      const cell = doc.createElement("i");
      if (key === 1) cell.className = "on";
      pad.append(cell);
    }
    ill.append(
      pad,
      span("plus", "+"),
      slot("1", "Basic"),
      span("plus", "or"),
      slot("2", "Ultimate"),
    );
  } else if (index === 2) {
    const svg = svgElement(doc, "svg", {
      width: 150,
      height: 80,
      viewBox: "0 0 150 80",
      "aria-hidden": "true",
    });
    const hunter = ruleEmblem(doc, 3, 52, 32, 16);
    svg.append(
      svgElement(doc, "circle", {
        cx: 60,
        cy: 40,
        r: 34,
        fill: "#C9D4E8",
        opacity: 0.05,
      }),
      svgElement(doc, "circle", {
        cx: 60,
        cy: 40,
        r: 34,
        fill: "none",
        stroke: "#C9D4E8",
        "stroke-width": 1,
        "stroke-dasharray": "3 3",
        opacity: 0.6,
      }),
      svgElement(doc, "path", {
        d: "M92 26 L150 8 L150 70 L92 54 Z",
        fill: "#0A0D14",
        opacity: 0.55,
      }),
      svgElement(doc, "rect", {
        x: 86,
        y: 28,
        width: 10,
        height: 24,
        rx: 2,
        fill: "#333C4F",
        stroke: "#454F66",
      }),
      svgElement(doc, "circle", {
        cx: 60,
        cy: 40,
        r: 13,
        class: "tok-ring tok-ring--a",
        fill: "#11264A",
        "stroke-width": 3,
      }),
      hunter,
    );
    ill.append(svg);
  } else {
    const zone = doc.createElement("span");
    zone.className = "zone";
    const times = doc.createElement("em");
    times.textContent = "×2";
    zone.append(times, ruleToken(doc, 4, 2, true), span("pts", "+2"));
    ill.append(ruleToken(doc, 4, 2, true), span("pts", "+1"), zone);
  }
  return ill;
}

/**
 * Fill the How to Play overlay's body (#htp-body) with the four rule cards,
 * as the vision page does when the overlay first opens.
 *
 * Writes one `.rules` grid with four `.rule` cards (picture, title h3 and
 * sentence p, from HELP_OVERLAY_RULES). Does nothing when root has no
 * #htp-body or the body already has children, so calling it again is safe.
 * The overlay's key row and Close button are page markup. It reads no frame,
 * so it is the same in Oracle View and Agent POV. Side effects: DOM only.
 *
 * @param {ParentNode} root Where to look, normally `document`.
 * @returns {boolean} True when the cards were written now.
 */
export function renderHelpOverlayRules(root) {
  const body = root.querySelector("#htp-body");
  if (body === null || body.childElementCount > 0) return false;
  const doc = body.ownerDocument;
  const rules = doc.createElement("div");
  rules.className = "rules";
  for (const [index, [title, sentence]] of HELP_OVERLAY_RULES.entries()) {
    const card = doc.createElement("div");
    card.className = "rule";
    const heading = doc.createElement("h3");
    heading.textContent = title;
    const text = doc.createElement("p");
    text.textContent = sentence;
    card.append(rulePicture(doc, index), heading, text);
    rules.append(card);
  }
  body.append(rules);
  return true;
}

/**
 * Map internal audience names to Oracle View or Agent POV.
 * Unknown values return "View Unavailable"; protocol discriminators remain unchanged.
 * Pure: reads only `value`.
 *
 * @param {unknown} value An audience name from a frame: "researcher",
 *   "agent_pov", "actor_pov" or "pov"; anything else is unknown.
 * @returns {"Oracle View" | "Agent POV" | "View Unavailable"} The plain label.
 */
export function publicAudienceLabel(value) {
  if (value === "researcher") {
    return "Oracle View";
  }
  if (value === "actor_pov" || value === "agent_pov" || value === "pov") {
    return "Agent POV";
  }
  return "View Unavailable";
}

// Battlefield labels and instructions for each interaction state. Only
// applyBattlefieldBoundaryCopy reads them.
const SCRIPTED_BATTLEFIELD_LABEL =
  "Inspection-only scripted battlefield. Authorized bodies can be inspected; use Advance scripted frame for the next authorized step.";
const SCRIPTED_BATTLEFIELD_INSTRUCTIONS =
  "Scripted live view is inspection-only. Activate an authorized body to inspect current facts; use Advance scripted frame for the next authorized step.";
const FENCED_LIVE_BATTLEFIELD_LABEL =
  "Read-only live battlefield. Simulator and actor activation controls are unavailable.";
const FENCED_LIVE_BATTLEFIELD_INSTRUCTIONS =
  "Live battlefield interaction is unavailable while authority is pending, offline, resynchronizing, shutting down, or terminal.";
const READ_ONLY_LIVE_SCIENTIFIC_LABEL =
  "Read-only live battlefield. Select an agent to inspect it; new actions are unavailable.";
const READ_ONLY_LIVE_SCIENTIFIC_INSTRUCTIONS =
  "Inspect the accepted game. Select an agent to see its facts. This view does not submit actions; Restart starts another match when available.";
const AGENT_CLOSEOUT_BATTLEFIELD_LABEL =
  "Agent POV recording closeout battlefield. Authorized bodies can be inspected; simulator controls are unavailable.";
const AGENT_CLOSEOUT_BATTLEFIELD_INSTRUCTIONS =
  "Inspect visible bodies here, or click a champion's portrait in the scorebug to inspect its view. This recording cannot accept another turn.";
const TERMINAL_REPLAY_BATTLEFIELD_LABEL =
  "Replay arena at its last tick. It never changes.";
const TERMINAL_REPLAY_BATTLEFIELD_INSTRUCTIONS =
  "This is the replay's last tick. Click a champion to look at it, or use the timeline to go to another tick.";
const TERMINAL_REPLAY_AGENT_BATTLEFIELD_INSTRUCTIONS =
  "This is the replay's last tick. Click a champion you can see, or pick any champion in the roster, to see what its team saw; use the timeline to go to another tick.";

/**
 * Page elements the shell paints. main.js finds them once at startup by ID.
 *
 * @typedef {object} ShellElements
 * @property {any} connectionStatus `#connection-status` pill.
 * @property {any} notice `#notice` one-line status message.
 * @property {any} battlefieldShell `#battlefield-shell`; its `aria-busy` follows
 *   page requests.
 * @property {any} battlefield `#battlefield` SVG; the shell sets its role,
 *   `tabIndex` and `aria-label`.
 * @property {any} battlefieldInstructions `#battlefield-instructions`.
 * @property {any} audienceBadge `#audience-badge` pill.
 * @property {any} terminalBadge `#terminal-badge` pill.
 * @property {any} stepValue `#step-value`.
 * @property {any} transitionValue `#transition-value`.
 * @property {any} scenarioDescription `#scenario-description`.
 * @property {any} viewSelect `#view-select`.
 * @property {any} reconnectButton `#reconnect-button`.
 * @property {any} exitButton `#exit-button`.
 * @property {any} resetButton `#reset-button` (availability only; its click
 *   belongs to the command deck).
 * @property {any} liveRangesButton `#live-ranges-button`.
 * @property {any} replayTimeline `#replay-timeline`.
 * @property {NodeListOf<HTMLElement>} liveOnly Every `[data-live-only]` element.
 * @property {NodeListOf<HTMLElement>} replayOnly Every `[data-replay-only]` element.
 * @property {NodeListOf<HTMLElement>} liveHelpModes Every
 *   `[data-live-help-mode]` block in the Help dialog.
 */

/**
 * The live page state that main.js owns. The shell reads it and never assigns it.
 *
 * @typedef {object} ShellState
 * @property {Record<string, any> | null} frame Installed transport frame, or
 *   null before the first frame.
 * @property {Readonly<Record<string, any>> | null} presentation Installed
 *   authorized presentation, or null while authority is pending.
 * @property {boolean} busy True while one page request is in flight.
 * @property {boolean} [quietBusy] True while that request is autoplay's own
 *   next-frame fetch (Replay Viewer play): the connection pill stays
 *   "Connected" then. Absent means false.
 * @property {boolean} offline True after a connection failure.
 * @property {boolean} resyncRequired True until Reconnect installs fresh authority.
 * @property {boolean} shuttingDown True after Exit was accepted.
 * @property {string | null} notice Message for the notice line, or null for none.
 * @property {string} noticeLevel Notice level: "info", "success", "warning" or "error".
 */

/**
 * Everything createShell needs from main.js. Each function keeps the contract
 * documented beside its definition in main.js (or tactics-turn.js).
 *
 * @typedef {object} ShellContext
 * @property {ShellElements} elements Header and toolbar elements (see above).
 * @property {ShellState} state Live page state, read only.
 * @property {() => Readonly<{schema_version: 1,
 *   product_kind: "combat_debugger" | "replay_viewer"}> | null} getProductIdentity
 *   The validated launcher identity, or null when startup was refused.
 * @property {() => Readonly<{deferredDrafts: ReadonlyArray<unknown>, deferredSubmit: unknown}>
 *   | null} getActiveLiveCommandTransaction The live command now in flight, or
 *   null. Only its queued order keys and queued Enter are read.
 * @property {() => boolean} isReplayMode True when the product is the Replay Viewer.
 * @property {(frame: unknown) => boolean} isTerminal True at a terminal or final frame.
 * @property {(value: unknown) => value is Record<string, any>} isRecord Shape guard.
 * @property {(value: unknown) => any[]} asArray The array itself, or a new empty array.
 * @property {() => InstalledAuthority} installedPresentationAuthority The trusted
 *   installed pair, or null.
 * @property {(presentation: unknown) => boolean} installedPresentationRangesVisible
 *   Whether ranges show for this exact installed presentation.
 * @property {() => boolean} liveScriptedInspectionOnly True for scripted live playback.
 * @property {() => boolean} liveBattlefieldCommandsInteractive True when the live
 *   battlefield may take Manual commands now.
 * @property {() => boolean} scriptedBattlefieldLocallyActionable True when scripted
 *   live playback may take local inspection now.
 * @property {() => boolean} liveAgentCloseoutInspectionActionable True when live
 *   Agent POV inspection stays available during recording closeout.
 * @property {() => boolean} installedLiveScientificInspectionAvailable True when
 *   any trusted live presentation is installed.
 * @property {() => boolean} recordingScientificControlsFenced True when recording
 *   closeout fences game controls.
 * @property {() => boolean} recordingRestartControlsBlocked True when recording
 *   blocks Restart with no discard choice.
 * @property {() => void} applyAuthorizedInspectorChrome Repaint the Agent Details
 *   heading and accent.
 * @property {(presentation: unknown) => void} renderInstalledMatch Repaint the
 *   match scoreboard.
 * @property {() => void} renderPendingPresentationChrome Paint the unavailable
 *   state while no authority is installed.
 * @property {(installed: InstalledAuthority) => void} renderRecordingControls
 *   Repaint the live recording panel.
 * @property {(installed: InstalledAuthority) => void} renderReplayMetadata
 *   Repaint the replay identity, badges and timeline controls.
 * @property {() => void} renderCommandAvailability Repaint the command deck
 *   (from tactics-turn.js).
 * @property {() => void} registerAuthorityAwareUtilityHelp Refresh help for
 *   Ranges, Clear Selection and actor cycling.
 * @property {() => Readonly<Record<string, boolean>>} [getVisualFilterState]
 *   The current Visual Filters state. Optional; without it every switch reads
 *   its shipped default. The shell reads brand_lockup, title_card and
 *   target_selection_visuals, and in live play
 *   high_contrast and brand_signature.
 * @property {(command: PovAgentChoice["command"]) => void} [activateReplayPovAgent]
 *   Make another agent the Replay Viewer's POV agent, by the same code a
 *   roster row click runs. Optional; without it the POV agent chip shows but
 *   its picker buttons stay disabled.
 */

/**
 * Create the header and toolbar renderers for one page.
 *
 * Call once at page startup, after the page state exists and before anything
 * can repaint. The returned functions read `context.state` and the getters each
 * time they run. Creating the shell finds the shell's own page parts by ID
 * (#view-switch, #pov-agent-chip, #pov-agent-chip-button, #pov-agent-picker,
 * #session-details, #copy-transition-button, #recording-panel,
 * #recording-badge, #match-place, #oracle-view-tag, #pov-tag, #title-card,
 * #arena, #battlefield, #hcard, #bc and the
 * .app-header); a missing part is
 * skipped. It then binds, once: View switch clicks (set the View select and
 * fire its change event), POV agent picker clicks (call
 * activateReplayPovAgent), the Copy button (write the transition ID to the
 * clipboard), placement of the three popovers under their buttons when they
 * open, and the live Visual Filters drawer's top below the scorebug when it
 * opens and when the window resizes (bindVisualFiltersDrawer). It also fills the How to Play overlay's rule cards once
 * (renderHelpOverlayRules) and installs the live hover card on #bc
 * (createHoverCard from tooltip.js, with championCardFacts, the arena box
 * #arena as the box it always stays inside, and the Lock In button
 * (#submit-turn-button) as a place it never covers; it shows only in the
 * BattleClient and DevClient combat screen, with the
 * target_selection_visuals switch on; holding Shift by itself shows every
 * status; the returned refreshHoverCard draws it again from each new frame).
 * Nothing else touches the DOM until a returned function runs or the
 * pointer moves over the page.
 *
 * @param {ShellContext} context Page elements, state and helpers from main.js.
 *   Functions are captured once; pass stable functions or wrappers.
 * @returns {Readonly<{
 *   renderConnection: () => void,
 *   applyBattlefieldBoundaryCopy: () => void,
 *   renderViewerBoundary: () => void,
 *   renderSessionToolbar: (installed?: InstalledAuthority) => void,
 *   refreshHoverCard: () => void,
 * }>} The shell API. See each function's own documentation.
 */
export function createShell(context) {
  const {
    elements,
    state,
    getProductIdentity,
    getActiveLiveCommandTransaction,
    isReplayMode,
    isTerminal,
    isRecord,
    asArray,
    installedPresentationAuthority,
    installedPresentationRangesVisible,
    liveScriptedInspectionOnly,
    liveBattlefieldCommandsInteractive,
    scriptedBattlefieldLocallyActionable,
    liveAgentCloseoutInspectionActionable,
    installedLiveScientificInspectionAvailable,
    recordingScientificControlsFenced,
    recordingRestartControlsBlocked,
    applyAuthorizedInspectorChrome,
    renderInstalledMatch,
    renderPendingPresentationChrome,
    renderRecordingControls,
    renderReplayMetadata,
    renderCommandAvailability,
    registerAuthorityAwareUtilityHelp,
    getVisualFilterState,
    activateReplayPovAgent,
  } = context;

  /** @param {string} id @returns {HTMLElement | null} */
  const byId = (id) =>
    typeof document.getElementById === "function" ? document.getElementById(id) : null;
  const shellParts = Object.freeze({
    header:
      typeof document.querySelector === "function"
        ? /** @type {HTMLElement | null} */ (document.querySelector(".app-header"))
        : null,
    viewSwitch: byId("view-switch"),
    povAgentChip: byId("pov-agent-chip"),
    povAgentChipButton: byId("pov-agent-chip-button"),
    povAgentPicker: byId("pov-agent-picker"),
    copyTransitionButton: byId("copy-transition-button"),
    matchPlace: byId("match-place"),
    oracleViewTag: byId("oracle-view-tag"),
    povTag: byId("pov-tag"),
    titleCard: byId("title-card"),
    arena: byId("arena"),
    battlefield: byId("battlefield"),
    visualFiltersButton: byId("visual-filters-button"),
    hoverCard: byId("hcard"),
    grid: byId("bc"),
  });
  // Whether the Agent POV button's live help is set (settleAgentPovTitle).
  let agentPovTitleSettled = false;
  // The view tag is rebuilt only when its words change.
  let viewTagKey = "";
  // Title Card timing: "waiting" until the first frame, then "done" for good.
  /** @type {"waiting" | "shown" | "done"} */
  let titleCardStage = "waiting";
  /** @type {ReturnType<typeof setTimeout> | null} */
  let titleCardTimer = null;
  // The picker is rebuilt only when its agents, current agent or availability change.
  let povAgentPickerKey = "";
  // The palette the High Contrast switch last applied (null before the first
  // repaint).
  /** @type {"moonlit" | "high_contrast" | null} */
  let livePalette = null;
  // Live play: the Brand Signature element (built on first use), its last
  // placement ("" while hidden) and its pending animation frame (0: none).
  /** @type {HTMLElement | null} */
  let brandSignature = null;
  let brandSignatureKey = "";
  let brandSignatureFrame = 0;
  /** One observer on #match-place that rechecks a cut place line (markCutPlaceRest). */
  /** @type {ResizeObserver | null} */
  let placeRestObserver = null;

  /**
   * Read one Visual Filters switch, falling back to its shipped default.
   *
   * @param {string} filterId A switch ID from visual-filters.js.
   * @returns {boolean} Whether the switch is on; true for an unknown ID.
   */
  function filterOn(filterId) {
    try {
      return isVisualFilterEnabled(
        getVisualFilterState?.() ?? DEFAULT_VISUAL_FILTER_STATE,
        filterId,
      );
    } catch {
      return true;
    }
  }

  /**
   * Show the Title Card after TITLE_CARD_DELAY_MS with no frame, and hide it
   * for good once any frame arrives.
   *
   * The card holds the owner's emblem, the rune-cut "MARL-BGs", the official
   * title "MARL-BattleGrounds" in mixed case and a ward rule with a lozenge.
   * It is decorative (aria-hidden); the waiting sentence under it stays the
   * message. Its lozenge breathes only while the page is still trying to
   * connect (`data-waiting="true"`), not while offline, resynchronizing or
   * shutting down. With the title_card switch off it never shows. Side
   * effects: one setTimeout at most, DOM `hidden` and `data-waiting`, and the
   * card's content built once. It shows no game fact, so it is the same in
   * every view.
   *
   * @param {boolean} hasFrame True once any frame or installed authority exists.
   * @returns {void}
   */
  function syncTitleCard(hasFrame) {
    const card = shellParts.titleCard;
    if (card === null) return;
    if (hasFrame || titleCardStage === "done") {
      titleCardStage = "done";
      if (titleCardTimer !== null) clearTimeout(titleCardTimer);
      titleCardTimer = null;
      card.hidden = true;
      return;
    }
    card.dataset.waiting = String(
      !state.offline && !state.resyncRequired && !state.shuttingDown,
    );
    if (!filterOn("title_card")) {
      card.hidden = true;
      return;
    }
    if (titleCardStage === "shown") {
      card.hidden = false;
      return;
    }
    if (titleCardTimer !== null) return;
    titleCardTimer = setTimeout(() => {
      titleCardTimer = null;
      if (titleCardStage !== "waiting" || !filterOn("title_card")) return;
      buildTitleCard(card);
      titleCardStage = "shown";
      card.hidden = false;
    }, TITLE_CARD_DELAY_MS);
  }

  /**
   * Fill the empty Title Card once: emblem, rune-cut wordmark, official title
   * and ward rule. Side effects: DOM only.
   *
   * @param {HTMLElement} card The #title-card section.
   * @returns {void}
   */
  function buildTitleCard(card) {
    if (card.childElementCount > 0) return;
    const doc = card.ownerDocument;
    const namespace = "http://www.w3.org/2000/svg";
    card.classList.add("title-card");
    card.setAttribute("aria-hidden", "true");
    const picture = doc.createElement("picture");
    const source = doc.createElement("source");
    source.type = "image/webp";
    source.srcset = "/assets/brand/marl-bgs-emblem-128.webp";
    const emblem = doc.createElement("img");
    emblem.className = "title-card__emblem";
    emblem.src = "/assets/brand/marl-bgs-emblem-64.png";
    emblem.width = 64;
    emblem.height = 64;
    emblem.alt = "";
    picture.append(source, emblem);
    const wordmark = /** @type {SVGSVGElement} */ (
      doc.createElementNS(namespace, "svg")
    );
    wordmark.setAttribute("class", "title-card__wordmark");
    renderRuneWordmark(wordmark, BRAND_NAME, 16);
    const title = doc.createElement("p");
    title.className = "title-card__title";
    title.textContent = OFFICIAL_TITLE;
    const rule = doc.createElementNS(namespace, "svg");
    rule.setAttribute("class", "title-card__rule");
    rule.setAttribute("width", "240");
    rule.setAttribute("height", "12");
    rule.setAttribute("viewBox", "0 0 240 12");
    const line = doc.createElementNS(namespace, "line");
    for (const [name, value] of [
      ["x1", "0"],
      ["y1", "6.5"],
      ["x2", "240"],
      ["y2", "6.5"],
    ]) {
      line.setAttribute(name, value);
    }
    const lozenge = doc.createElementNS(namespace, "polygon");
    lozenge.setAttribute("class", "title-card__lozenge");
    lozenge.setAttribute("points", "120,0.5 124,6.5 120,12.5 116,6.5");
    rule.append(line, lozenge);
    card.replaceChildren(picture, wordmark, title, rule);
  }

  /**
   * Make the page palette follow the High Contrast switch (high_contrast,
   * Arena group, on by default), in all three clients (owner, 4 Oct 2026:
   * one switch everywhere; the Replay Viewer's Palette select is gone).
   *
   * On: palette.js's High Contrast palette (`data-palette="high-contrast"` on
   * the root element, which styles/palette.css reads). Off: Moonlit (no
   * attribute). It writes only when the switch's value differs from the last
   * value it applied, so ordinary repaints change nothing. Colors only: it
   * shows no game fact, so Agent POV and Oracle View are the same, and an
   * exported PNG keeps its locked backdrop.
   *
   * @returns {void}
   */
  function syncPalette() {
    const wanted = filterOn("high_contrast") ? "high_contrast" : "moonlit";
    if (wanted === livePalette || typeof document.documentElement !== "object") return;
    livePalette = applyPalette(document.documentElement, wanted);
  }

  /**
   * Live play only: show, place or hide the Brand Signature (brand_signature,
   * HUD group, off by default).
   *
   * The signature is page chrome inside the arena box (#arena), never part of the
   * battlefield SVG, so it can never reach an exported PNG: the owner's
   * emblem (18 px) and the rune-cut "MARL-BGs" (cap 10 px, UI Muted strokes,
   * no keyline), placed by brandSignaturePlacement in the margin by the
   * floor's bottom-right corner. Off: hidden at once. On: one placement in the
   * next animation frame, after the arena has drawn its floor; a resize of
   * the arena box places it again. It writes style only when the placement
   * changes. It is decorative (aria-hidden, no pointer events) and shows no
   * game fact, so Agent POV and Oracle View are the same.
   *
   * @returns {void}
   */
  function syncBrandSignature() {
    if (shellParts.arena === null) return;
    if (!filterOn("brand_signature")) {
      if (brandSignature !== null && !brandSignature.hidden) {
        brandSignature.hidden = true;
        brandSignatureKey = "";
      }
      return;
    }
    scheduleBrandSignature();
  }

  /**
   * Ask for one Brand Signature placement in the next animation frame. A
   * request while one is pending does nothing. Without requestAnimationFrame
   * (a test page) it does nothing.
   *
   * @returns {void}
   */
  function scheduleBrandSignature() {
    if (brandSignatureFrame !== 0 || typeof requestAnimationFrame !== "function") {
      return;
    }
    brandSignatureFrame = requestAnimationFrame(() => {
      brandSignatureFrame = 0;
      placeBrandSignature();
    });
  }

  /**
   * Build the Brand Signature on first use, then place it from the arena box
   * (inside its border), the battlefield SVG and the floor edge
   * (`#battlefield rect.map-boundary`), or hide it when the
   * switch is off, the floor is missing or brandSignaturePlacement finds no
   * room. Side effects: DOM nodes once (and one ResizeObserver on the arena
   * box when the browser has it), then `hidden`, `data-layout`, `left` and
   * `top`.
   *
   * @returns {void}
   */
  function placeBrandSignature() {
    const arena = shellParts.arena;
    if (arena === null || !filterOn("brand_signature")) return;
    const signature = brandSignature ?? buildBrandSignature(arena);
    const field = shellParts.battlefield;
    const floor = field?.querySelector("rect.map-boundary") ?? null;
    const word = signature.querySelector("svg");
    // The box inside the arena's border: absolute left and top count from there.
    const outer = arena.getBoundingClientRect();
    const left = outer.left + arena.clientLeft;
    const top = outer.top + arena.clientTop;
    const box = {
      left,
      top,
      right: left + arena.clientWidth,
      bottom: top + arena.clientHeight,
    };
    const place =
      field === null || floor === null || word === null
        ? null
        : brandSignaturePlacement({
            box,
            field: field.getBoundingClientRect(),
            floor: floor.getBoundingClientRect(),
            word: {
              width: Number(word.getAttribute("width")),
              height: Number(word.getAttribute("height")),
            },
          });
    const key =
      place === null
        ? ""
        : `${place.layout} ${place.left.toFixed(1)} ${place.top.toFixed(1)}`;
    if (key === brandSignatureKey && signature.hidden === (place === null)) return;
    brandSignatureKey = key;
    if (place === null) {
      signature.hidden = true;
      return;
    }
    signature.dataset.layout = place.layout;
    signature.style.left = `${place.left.toFixed(1)}px`;
    signature.style.top = `${place.top.toFixed(1)}px`;
    signature.hidden = false;
  }

  /**
   * Create the hidden Brand Signature element in the arena box: a
   * `div.brand-signature` (aria-hidden) holding the emblem image and the
   * carved wordmark, cut to the letters alone (the signature draws no keyline,
   * so the keyline's margin is dropped from its size and viewBox). Styles live in src/styles/shell.css. Side effects: DOM
   * nodes and, when the browser has ResizeObserver, one observer on the box
   * that asks for a new placement after the box changes size.
   *
   * @param {HTMLElement} arena The arena box, #arena.
   * @returns {HTMLElement} The new element, also kept in brandSignature.
   */
  function buildBrandSignature(arena) {
    const doc = arena.ownerDocument;
    const signature = doc.createElement("div");
    signature.className = "brand-signature";
    signature.setAttribute("aria-hidden", "true");
    signature.hidden = true;
    const emblem = doc.createElement("img");
    emblem.className = "brand-signature__emblem";
    emblem.src = "/assets/brand/marl-bgs-emblem-64.png";
    emblem.width = BRAND_SIGNATURE_EMBLEM_PX;
    emblem.height = BRAND_SIGNATURE_EMBLEM_PX;
    emblem.alt = "";
    const wordmark = /** @type {SVGSVGElement} */ (
      doc.createElementNS("http://www.w3.org/2000/svg", "svg")
    );
    wordmark.setAttribute("class", "brand-signature__wordmark");
    renderRuneWordmark(wordmark, BRAND_NAME, BRAND_SIGNATURE_CAP_PX);
    // The signature hides the keyline, so drop the keyline's margin too: the
    // box is then the letters alone.
    const [margin = 0] = (wordmark.getAttribute("viewBox") ?? "")
      .split(" ")
      .map((value) => -Number(value));
    if (Number.isFinite(margin) && margin > 0) {
      const width = Number(wordmark.getAttribute("width")) - margin * 2;
      const height = Number(wordmark.getAttribute("height")) - margin * 2;
      wordmark.setAttribute("width", String(width));
      wordmark.setAttribute("height", String(height));
      wordmark.setAttribute("viewBox", `0 0 ${width} ${height}`);
    }
    signature.append(emblem, wordmark);
    arena.append(signature);
    brandSignature = signature;
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => {
        if (filterOn("brand_signature")) scheduleBrandSignature();
      }).observe(arena);
    }
    return signature;
  }

  /**
   * Paint the View switch buttons from the View select's value and state.
   * Side effects: `aria-pressed` and `disabled` on each button.
   *
   * @returns {void}
   */
  function syncViewSwitch() {
    for (const button of shellParts.viewSwitch?.querySelectorAll("button") ?? []) {
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.viewMode === elements.viewSelect.value),
      );
      button.disabled = elements.viewSelect.disabled;
    }
  }

  /**
   * Show the Replay Viewer's POV agent chip and rebuild its picker when needed.
   *
   * Shows always in the Replay Viewer's Agent POV, and never elsewhere: no
   * switch hides it, because it is how you choose whose eyes you watch
   * through. Reads identity rows from the installed presentation (the
   * Replay Viewer's researcher chrome lists every agent). Side effects: the
   * chip's `hidden`, the chip button's content and `aria-label`, the
   * picker's buttons. When the chip hides (any view but Replay Agent POV)
   * it closes the picker and empties the
   * button, its `aria-label` and the picker, so Oracle -> POV -> Oracle
   * leaves the same chip DOM as a fresh page.
   *
   * @param {Readonly<Record<string, any>> | null} presentation The installed presentation.
   * @param {boolean} replay True in the Replay Viewer.
   * @param {boolean} disabled True while picking must wait (busy, offline, no frame).
   * @returns {void}
   */
  function renderPovAgentChip(presentation, replay, disabled) {
    const chip = shellParts.povAgentChip;
    const button = shellParts.povAgentChipButton;
    const picker = shellParts.povAgentPicker;
    if (chip === null || button === null || picker === null) return;
    const view =
      replay &&
      presentation !== null &&
      authorizedPresentationAudience(presentation) === "agent_pov"
        ? povAgentChipView(
            isRecord(presentation.authority)
              ? presentation.authority.recipient_public_agent_id
              : null,
            authorizedPresentationIdentityRows(presentation),
          )
        : null;
    chip.hidden = view === null;
    if (view === null) {
      if (picker.matches?.(":popover-open")) picker.hidePopover();
      povAgentPickerKey = "";
      // Match a fresh page: no old agent's label, token or picker rows stay
      // behind in the hidden chip after leaving Agent POV.
      button.replaceChildren();
      button.removeAttribute("aria-label");
      picker.replaceChildren();
      return;
    }
    const doc = chip.ownerDocument;
    /** @param {string} letter @param {string} accent @param {number} team @param {boolean} large */
    const token = (letter, accent, team, large) => {
      const node = doc.createElement("span");
      node.className = large ? "mini-token mini-token--large" : "mini-token";
      node.dataset.class = accent;
      node.dataset.team = String(team);
      node.setAttribute("aria-hidden", "true");
      node.textContent = letter;
      return node;
    };
    const canPick = typeof activateReplayPovAgent === "function" && !disabled;
    const key = JSON.stringify([view.label, canPick, view.teams]);
    if (key === povAgentPickerKey) return;
    povAgentPickerKey = key;
    const label = doc.createElement("span");
    label.textContent = view.label;
    button.replaceChildren(
      token(view.classLetter, view.accent, view.teamId, false),
      label,
    );
    button.setAttribute(
      "aria-label",
      `Agent POV: ${view.label}. Choose another agent.`,
    );
    const heading = doc.createElement("p");
    heading.className = "eyebrow";
    heading.textContent = "Watch Through the Eyes Of";
    const rows = view.teams.map((team, index) => {
      const row = doc.createElement("div");
      row.className = "pov-agent-picker__row";
      row.setAttribute("role", "group");
      row.setAttribute("aria-label", index === 0 ? "Team A" : "Team B");
      for (const choice of team) {
        const option = doc.createElement("button");
        option.type = "button";
        option.dataset.publicAgentId = choice.publicAgentId;
        option.disabled = !canPick;
        option.setAttribute("aria-current", String(choice.current));
        const text = doc.createElement("span");
        text.textContent = choice.label;
        option.append(
          token(choice.classLetter, choice.accent, choice.teamId, true),
          text,
        );
        row.append(option);
      }
      return row;
    });
    picker.replaceChildren(heading, ...rows);
  }

  /**
   * Bind the shell's own controls once. See createShell for the list.
   * Side effects: event listeners on the shell's page parts, and the View
   * buttons' registered hover help (registerViewButtonHelp).
   *
   * @returns {void}
   */
  function bindShellControls() {
    // Both View buttons get registered help at once; live play's Agent POV
    // words settle with the first frame (settleAgentPovTitle).
    for (const button of shellParts.viewSwitch?.querySelectorAll?.("button") ?? []) {
      const pov = button.dataset.viewMode === "pov";
      registerViewButtonHelp(
        button,
        pov ? "Agent POV" : "Oracle View",
        pov
          ? agentPovButtonTitle({ replay: true, devClient: true })
          : ORACLE_VIEW_BUTTON_HELP,
      );
    }
    shellParts.viewSwitch?.addEventListener("click", (event) => {
      const target =
        event.target instanceof Element ? event.target.closest("button") : null;
      const mode = target?.dataset.viewMode;
      if (!mode || target.disabled || elements.viewSelect.disabled) return;
      if (elements.viewSelect.value === mode) return;
      elements.viewSelect.value = mode;
      syncViewSwitch();
      elements.viewSelect.dispatchEvent(new Event("change"));
    });
    shellParts.povAgentPicker?.addEventListener("click", (event) => {
      const target =
        event.target instanceof Element ? event.target.closest("button") : null;
      const publicAgentId = target?.dataset.publicAgentId;
      if (
        !publicAgentId ||
        target.disabled ||
        typeof activateReplayPovAgent !== "function"
      ) {
        return;
      }
      const installed = installedPresentationAuthority();
      const presentation = installed?.presentation ?? null;
      const view =
        presentation === null
          ? null
          : povAgentChipView(
              isRecord(presentation.authority)
                ? presentation.authority.recipient_public_agent_id
                : null,
              authorizedPresentationIdentityRows(presentation),
            );
      const choice = view?.teams
        .flat()
        .find((row) => row.publicAgentId === publicAgentId);
      shellParts.povAgentPicker?.hidePopover();
      if (choice && !choice.current) activateReplayPovAgent(choice.command);
    });
    const copy = shellParts.copyTransitionButton;
    copy?.addEventListener("click", () => {
      const text = elements.transitionValue.textContent ?? "";
      if (text === "" || text === "—" || !navigator.clipboard) return;
      navigator.clipboard.writeText(text).then(
        () => {
          copy.textContent = "Copied";
          setTimeout(() => {
            copy.textContent = "Copy";
          }, 1500);
        },
        () => {
          copy.textContent = "Copy Failed";
          setTimeout(() => {
            copy.textContent = "Copy";
          }, 1500);
        },
      );
    });
    for (const [popoverId, anchorId] of [
      ["session-details", "session-details-button"],
      ["recording-panel", "recording-badge"],
      ["pov-agent-picker", "pov-agent-chip-button"],
    ]) {
      const popover = byId(popoverId);
      const anchor = byId(anchorId);
      if (popover === null || anchor === null) continue;
      popover.addEventListener("toggle", (event) => {
        if (/** @type {ToggleEvent} */ (event).newState !== "open") return;
        const place = popoverPlacement(
          anchor.getBoundingClientRect(),
          popover.getBoundingClientRect().width,
          document.documentElement.clientWidth,
        );
        popover.style.top = `${place.top}px`;
        popover.style.left = `${place.left}px`;
        popover.style.right = "auto";
      });
    }
    bindVisualFiltersDrawer();
  }

  /**
   * Keep the live Visual Filters drawer (#visual-filters-drawer) below the
   * scorebug: just before it opens, and on each window resize while it is
   * open, write its --vf-drawer-top (visualFiltersDrawerTop of the rail
   * #rail: the rail's top plus its top padding, the top of its first card)
   * as an inline custom property; overlays.css reads it. Only in live play
   * (product "combat_debugger"); the Replay Viewer moves the drawer's card
   * into its rail. Side effects: two event listeners (the drawer's
   * beforetoggle, the window's resize) and the custom property. Does nothing
   * on a page without the drawer or the rail.
   *
   * @returns {void}
   */
  function bindVisualFiltersDrawer() {
    const drawer = byId("visual-filters-drawer");
    const rail = byId("rail");
    if (
      drawer === null ||
      rail === null ||
      typeof drawer.addEventListener !== "function" ||
      typeof window === "undefined"
    ) {
      return;
    }
    const place = () => {
      if (getProductIdentity()?.product_kind !== "combat_debugger") return;
      const top = visualFiltersDrawerTop(
        rail.getBoundingClientRect().top,
        Number.parseFloat(getComputedStyle(rail).paddingTop),
      );
      if (top === null) drawer.style.removeProperty("--vf-drawer-top");
      else drawer.style.setProperty("--vf-drawer-top", `${top}px`);
    };
    drawer.addEventListener("beforetoggle", (event) => {
      if (/** @type {ToggleEvent} */ (event).newState === "open") place();
    });
    window.addEventListener("resize", () => {
      let open = false;
      try {
        open = drawer.matches(":popover-open");
      } catch {
        // A browser without popovers never opens the drawer.
      }
      if (open) place();
    });
  }

  bindShellControls();
  if (typeof document.querySelector === "function") renderHelpOverlayRules(document);
  // The live hover card's handle: refreshHoverCard redraws it on every frame.
  const hoverCard =
    shellParts.hoverCard !== null &&
    shellParts.grid !== null &&
    typeof shellParts.grid.addEventListener === "function"
      ? createHoverCard({
          root: shellParts.grid,
          card: shellParts.hoverCard,
          enabled: () =>
            getProductIdentity()?.product_kind === "combat_debugger" &&
            !isReplayMode() &&
            filterOn("target_selection_visuals"),
          factsFor: (subject) => {
            const installed = installedPresentationAuthority();
            return championCardFacts({
              presentation: installed?.presentation ?? null,
              combatConfiguration: installed?.transport?.combat_configuration,
              subject,
            });
          },
          avoid: () => {
            const lockIn = byId("submit-turn-button");
            return lockIn === null ? [] : [lockIn];
          },
          // The card stays inside the arena box; only a Hold Shift card that
          // would shrink below 85% there grows past it (owner, 2 October 2026).
          box: () => byId("arena"),
        })
      : null;

  /**
   * Draw the live hover card (#hcard) again from the installed frame's facts
   * while the pointer rests, or hide it when nothing is left to describe
   * (createHoverCard's refresh). main.js calls it after every paint, next to
   * the tooltip controller's refresh, so a new tick, or Tab giving Agent POV
   * to the other team, never leaves an old frame's card on screen. Does
   * nothing while no card shows or when the page has no hover card.
   *
   * @returns {void}
   */
  function refreshHoverCard() {
    hoverCard?.refresh();
  }

  /**
   * Paint the arena's view tag (#pov-tag) in live play and keep the old
   * Oracle View tag (#oracle-view-tag) hidden: the tag now says both views.
   *
   * Words come from liveViewTag. The tag holds an eye picture, the lead and a
   * `<span>` with the detail; `data-view` ("pov" or "oracle") lets the page
   * paint Oracle View amber. Hidden in the Replay Viewer and with no frame.
   * Side effects: the tag's `hidden`, `data-view` and children (rebuilt only
   * when the words change).
   *
   * @param {Readonly<Record<string, any>> | null} presentation The installed presentation.
   * @param {Readonly<Record<string, any>> | null} transport The installed transport frame.
   * @returns {void}
   */
  function renderViewTag(presentation, transport) {
    if (shellParts.oracleViewTag !== null) shellParts.oracleViewTag.hidden = true;
    const tag = shellParts.povTag;
    if (tag === null) return;
    const audience =
      presentation === null ? null : authorizedPresentationAudience(presentation);
    const words = liveViewTag({
      product: isReplayMode() ? "replay_viewer" : getProductIdentity()?.product_kind,
      audience,
      scene:
        presentation === null ? null : authorizedPresentationSceneView(presentation),
      recipientPublicAgentId: isRecord(presentation?.authority)
        ? presentation.authority.recipient_public_agent_id
        : null,
      watchOnly: !hasManualChampion(transport?.combat_configuration),
    });
    tag.hidden = words === null;
    const key = words === null ? "" : `${words.view}|${words.lead}|${words.detail}`;
    if (key === viewTagKey) return;
    viewTagKey = key;
    if (words === null) {
      tag.replaceChildren();
      delete tag.dataset.view;
      return;
    }
    const doc = tag.ownerDocument;
    const eye = svgElement(doc, "svg", { viewBox: "0 0 16 16", "aria-hidden": "true" });
    eye.append(
      svgElement(doc, "path", {
        d: "M1.5 8 C3.6 4.6 5.7 3.2 8 3.2 S12.4 4.6 14.5 8 C12.4 11.4 10.3 12.8 8 12.8 S3.6 11.4 1.5 8 Z",
        fill: "none",
        stroke: "currentColor",
        "stroke-width": 1.3,
      }),
      svgElement(doc, "circle", { cx: 8, cy: 8, r: 2.2, fill: "currentColor" }),
    );
    const detail = doc.createElement("span");
    detail.textContent = words.detail;
    tag.dataset.view = words.view;
    tag.replaceChildren(eye, `${words.lead} `, detail);
  }

  /**
   * Write the top bar's place line: the bold name, then " · " and the plain
   * details (matchPlaceParts), for example "**Teutoburg Forest** · Map 47 ·
   * Team Deathmatch". The line is two spans (placeLineSplit):
   * span.where__name holds the bold name and the details that name the
   * place ("Teutoburg Forest · Map 47"), and span.where__more the rest
   * (" · Team Deathmatch"), absent when there is no rest. The top-bar styles
   * let only the second span give way when the bar runs short, so the map
   * name always shows whole in live play; a cut second span is marked
   * `data-cut` (markCutPlaceRest) and hidden whole. The text is the same as
   * one line. Hidden when the summary has no usable map. Side effects:
   * #match-place's `hidden` and children, the second span's `data-cut`, and
   * one ResizeObserver on #match-place created on the first write.
   *
   * @param {Readonly<Record<string, any>> | null} presentation The installed presentation.
   * @returns {void}
   */
  function renderPlaceLine(presentation) {
    const place = shellParts.matchPlace;
    if (place === null) return;
    const parts = matchPlaceParts(presentation?.match_summary);
    place.hidden = parts === null;
    if (parts === null) {
      place.replaceChildren();
      return;
    }
    const doc = place.ownerDocument;
    const { lead, more } = placeLineSplit(parts);
    const name = doc.createElement("b");
    name.textContent = parts.name;
    const where = doc.createElement("span");
    where.className = "where__name";
    where.append(name, ...(lead.length > 0 ? [` · ${lead.join(" · ")}`] : []));
    if (more.length === 0) {
      place.replaceChildren(where);
      return;
    }
    const rest = doc.createElement("span");
    rest.className = "where__more";
    rest.textContent = ` · ${more.join(" · ")}`;
    place.replaceChildren(where, rest);
    markCutPlaceRest();
    if (placeRestObserver === null && typeof ResizeObserver === "function") {
      placeRestObserver = new ResizeObserver(markCutPlaceRest);
      placeRestObserver.observe(place);
      const product = place.ownerDocument.getElementById?.("app-title");
      if (product) placeRestObserver.observe(product);
    }
  }

  /**
   * Mark the parts of the top bar that may give way with `data-cut` while the
   * bar cuts them (placeRestIsCut), so the top-bar styles hide them whole
   * instead of showing them cut with "…" (owner, 3 Oct 2026: a part that does
   * not fit hides whole, never cut halfway): the place line's
   * span.where__more (" · Team Deathmatch") and the brand's product name
   * (h1#app-title, "Replay Viewer" or "BattleClient"). Runs after each place
   * line write and whenever #match-place or the product name changes size.
   * Side effects: those elements' `data-cut` attribute only.
   *
   * @returns {void}
   */
  function markCutPlaceRest() {
    const rest = shellParts.matchPlace?.querySelector(".where__more");
    if (rest instanceof HTMLElement)
      rest.toggleAttribute("data-cut", placeRestIsCut(rest));
    const product = shellParts.matchPlace?.ownerDocument?.getElementById?.("app-title");
    if (product instanceof HTMLElement)
      product.toggleAttribute("data-cut", placeRestIsCut(product));
  }

  /**
   * Set the Agent POV button's hover help once a live frame is installed, from
   * the product's own default view (agentPovButtonTitle), never from the
   * view the page happened to load in: the BattleClient opens in Agent POV,
   * the DevClient (its Developer drawer, #developer-drawer, is shown before
   * this module loads) and the Replay Viewer open in Oracle View. So a view
   * round trip and a reload in either view give the same help. The Replay
   * Viewer keeps the help bindShellControls gave it.
   *
   * Side effects: the button's registered tip (registerViewButtonHelp), set
   * once per page.
   *
   * @param {Readonly<Record<string, any>> | null} transport The installed
   *   live transport frame, or null (the Replay Viewer, or no frame yet):
   *   then nothing changes.
   * @returns {void}
   */
  function settleAgentPovTitle(transport) {
    if (agentPovTitleSettled || transport === null) return;
    const button = shellParts.viewSwitch?.querySelector?.(
      "button[data-view-mode='pov']",
    );
    if (!button) return;
    agentPovTitleSettled = true;
    registerViewButtonHelp(
      button,
      "Agent POV",
      agentPovButtonTitle({
        replay: isReplayMode(),
        devClient:
          button.ownerDocument?.getElementById?.("developer-drawer")?.hidden === false,
      }),
    );
  }

  /**
   * Paint connection status, pending-input status, notice text, and battlefield busy state.
   * Read page state only; rendering a status does not reconnect or retry a command.
   *
   * The pill words and `data-state` come from connectionPill ("Connected",
   * "Connecting", "Command in Flight" and so on). Side effects: pill text and
   * `data-state`, the notice pill's text, `title`, `hidden` and
   * `data-level`, Reconnect's `hidden` (it shows only while offline or when
   * a resync is required), `aria-busy` on the battlefield shell, and the
   * Title Card (see syncTitleCard).
   *
   * @returns {void}
   */
  function renderConnection() {
    const pill = connectionPill(
      {
        busy: state.busy,
        quietBusy: state.quietBusy === true,
        offline: state.offline,
        resyncRequired: state.resyncRequired,
        shuttingDown: state.shuttingDown,
        hasFrame: Boolean(state.frame),
      },
      getActiveLiveCommandTransaction(),
    );
    elements.connectionStatus.textContent = pill.label;
    elements.connectionStatus.dataset.state = pill.status;

    elements.notice.hidden = !state.notice;
    elements.notice.textContent = state.notice ?? "";
    // The pill may cut a long notice short; its tooltip keeps the whole text.
    elements.notice.title = state.notice ?? "";
    elements.notice.dataset.level = state.noticeLevel;
    elements.reconnectButton.hidden = !(state.offline || state.resyncRequired);
    elements.battlefieldShell.setAttribute("aria-busy", String(state.busy));
    syncTitleCard(Boolean(state.frame));
  }

  /**
   * Set battlefield accessibility role, focusability, and instructions for current authority.
   * Distinguish replay, editable live, scripted inspection, Agent closeout inspection,
   * and unavailable surfaces using the same interaction gates as the controls.
   *
   * Only an editable live battlefield gets `role="application"` and
   * `tabIndex = 0`, so live keys reach it only then. Side effects: the SVG's
   * `role`, `tabIndex` and `aria-label`, and the instructions text. Agent POV:
   * the sentences describe what the user can do, never a champion.
   *
   * @returns {void}
   */
  function applyBattlefieldBoundaryCopy() {
    const installed = installedPresentationAuthority();
    const replay = isReplayMode();
    const terminal = installed !== null && isTerminal(installed.transport);
    const scientificFenced = recordingScientificControlsFenced();
    const audience = authorizedPresentationAudience(installed?.presentation);
    const scriptedInspection = scriptedBattlefieldLocallyActionable();
    const agentCloseoutInspection = liveAgentCloseoutInspectionActionable();
    const liveInteractive = liveBattlefieldCommandsInteractive();
    if (installed === null) {
      elements.battlefield.setAttribute("role", "img");
      elements.battlefield.tabIndex = -1;
      elements.battlefield.setAttribute("aria-label", FENCED_LIVE_BATTLEFIELD_LABEL);
      elements.battlefieldInstructions.textContent =
        FENCED_LIVE_BATTLEFIELD_INSTRUCTIONS;
    } else if (scriptedInspection && !scientificFenced) {
      elements.battlefield.setAttribute("role", "group");
      elements.battlefield.tabIndex = -1;
      elements.battlefield.setAttribute("aria-label", SCRIPTED_BATTLEFIELD_LABEL);
      elements.battlefieldInstructions.textContent = SCRIPTED_BATTLEFIELD_INSTRUCTIONS;
    } else if (agentCloseoutInspection) {
      elements.battlefield.setAttribute("role", "group");
      elements.battlefield.tabIndex = -1;
      elements.battlefield.setAttribute("aria-label", AGENT_CLOSEOUT_BATTLEFIELD_LABEL);
      elements.battlefieldInstructions.textContent =
        AGENT_CLOSEOUT_BATTLEFIELD_INSTRUCTIONS;
    } else if (replay) {
      elements.battlefield.setAttribute("role", "group");
      elements.battlefield.tabIndex = -1;
      elements.battlefield.setAttribute(
        "aria-label",
        terminal
          ? TERMINAL_REPLAY_BATTLEFIELD_LABEL
          : "Replay arena. It never changes; click a champion to look at it.",
      );
      elements.battlefieldInstructions.textContent = terminal
        ? audience === "agent_pov"
          ? TERMINAL_REPLAY_AGENT_BATTLEFIELD_INSTRUCTIONS
          : TERMINAL_REPLAY_BATTLEFIELD_INSTRUCTIONS
        : audience === "agent_pov"
          ? "Replay Agent POV shows only what one team saw. Click a champion you can see, or pick any champion in the roster, to see what its team saw at this tick."
          : "A replay never changes. Next Tick in the Actions tab shows the orders given on this tick. Click a champion to look at it, or use the timeline to go to another tick.";
    } else if (liveInteractive) {
      elements.battlefield.setAttribute("role", "application");
      elements.battlefield.tabIndex = 0;
      elements.battlefield.setAttribute(
        "aria-label",
        "Interactive battlefield. Press Help for keyboard controls.",
      );
      elements.battlefieldInstructions.textContent =
        audience === "agent_pov"
          ? "Live Agent POV is interactive and shows only what your champion may see. Click a champion you can see, or pick one in Roster, to inspect it; picking a champion you play also gives it your orders, and the view follows it. With only policies playing, the view follows the champion you pick. Shift-click selects a target you can see; Escape clears the target and leaves battlefield focus."
          : "Live Oracle View is interactive. Activate an authorized actor to control it; Shift-click selects an authorized target; Escape clears the target and leaves battlefield focus. Battlefield keyboard commands apply only while this surface has focus.";
    } else if (installedLiveScientificInspectionAvailable()) {
      elements.battlefield.setAttribute("role", "group");
      elements.battlefield.tabIndex = -1;
      elements.battlefield.setAttribute("aria-label", READ_ONLY_LIVE_SCIENTIFIC_LABEL);
      elements.battlefieldInstructions.textContent =
        READ_ONLY_LIVE_SCIENTIFIC_INSTRUCTIONS;
    } else {
      elements.battlefield.setAttribute("role", "img");
      elements.battlefield.tabIndex = -1;
      elements.battlefield.setAttribute(
        "aria-label",
        scientificFenced
          ? "Read-only recording closeout battlefield snapshot."
          : FENCED_LIVE_BATTLEFIELD_LABEL,
      );
      elements.battlefieldInstructions.textContent = scientificFenced
        ? "Recording closeout has fenced simulator and pending-action controls. Presentation, recovery, review, and Exit controls remain available."
        : FENCED_LIVE_BATTLEFIELD_INSTRUCTIONS;
    }
  }

  /**
   * Show product-specific page sections and audience-specific live help.
   * Update the root viewer-mode marker, timeline visibility, battlefield instructions,
   * and inspector heading from installed product and presentation state.
   *
   * Side effects: sets `<html data-viewer-mode>` to "replay" or "live"; toggles
   * `hidden` on every `[data-live-only]` and `[data-replay-only]` element, on
   * the Help dialog's live blocks (one of oracle, agent, scripted or
   * unavailable shows in live play) and on the replay timeline; then calls
   * applyBattlefieldBoundaryCopy and applyAuthorizedInspectorChrome. New slots
   * that must stay hidden until their module fills them must not carry
   * `data-live-only` or `data-replay-only`, because this function un-hides those.
   *
   * @returns {void}
   */
  function renderViewerBoundary() {
    const replay = isReplayMode();
    document.documentElement.dataset.viewerMode = replay ? "replay" : "live";
    for (const element of elements.liveOnly) {
      element.toggleAttribute("hidden", replay);
    }
    for (const element of elements.replayOnly) {
      element.toggleAttribute("hidden", !replay);
    }
    const inspectionState = authorizedPresentationInspectionState(state.presentation);
    const audience = authorizedPresentationAudience(state.presentation);
    const liveHelpMode = !isAuthorizedPresentationFrame(state.presentation)
      ? "unavailable"
      : inspectionState.state_kind === "live_scripted"
        ? "scripted"
        : audience === "researcher"
          ? "oracle"
          : audience === "agent_pov"
            ? "agent"
            : "unavailable";
    for (const element of elements.liveHelpModes) {
      element.toggleAttribute(
        "hidden",
        replay || element.dataset.liveHelpMode !== liveHelpMode,
      );
    }
    elements.replayTimeline.toggleAttribute("hidden", !replay);
    applyBattlefieldBoundaryCopy();
    applyAuthorizedInspectorChrome();
  }

  /**
   * Paint match, episode, view, recording, and command toolbar controls.
   * The optional installed pair defaults to the currently resolved authority. Missing
   * authority clears scientific chrome; all controls respect busy, connection,
   * terminal, recording, and audience gates.
   *
   * Calls, in order: renderInstalledMatch, renderViewerBoundary,
   * renderPendingPresentationChrome (only without authority), then after its own
   * toolbar work renderRecordingControls, renderReplayMetadata,
   * renderCommandAvailability and registerAuthorityAwareUtilityHelp. It reads
   * the transport and presentation only through `installed`, never through
   * `state.frame` or `state.presentation`. Its own work: Step and Transition
   * (and whether Copy works), the audience and terminal pills (the terminal
   * pill reads "Game Over" when the game has ended; a replay may instead
   * read "Time Limit", "Recording Ends" or "Replay End"), the scenario
   * line, the View select and switch, Reconnect, Exit ("Exit" in every
   * client, so the top bar has room for the full map name), Reset, live Ranges, the place line (renderPlaceLine), the view
   * tag (renderViewTag; the old Oracle View tag stays hidden), the Agent POV
   * button's hover help (settleAgentPovTitle, once), the
   * POV agent chip (Replay Viewer Agent POV only), the header's `data-brand-lockup` ("on" or "off" from the
   * brand_lockup switch), the always-visible Visual Filters
   * button once the product is known (main.js gives the Replay Viewer's
   * Visual Filters button its own job: it opens the rail's Visuals tab), and
   * the Title Card. Side effects: DOM text, `hidden`,
   * `disabled`, `aria-pressed` and the root `data-audience` and `data-preset`
   * attributes; no request is sent. Agent POV: the Transition fact may show the
   * researcher branch's transition ID, an opaque ID that names no champion;
   * the place title is the public map; nothing new is drawn over the arena.
   *
   * @param {InstalledAuthority} [installed] The installed pair to paint; defaults
   *   to installedPresentationAuthority(). Null paints the unavailable state.
   * @returns {void}
   */
  function renderSessionToolbar(installed = installedPresentationAuthority()) {
    const productIdentity = getProductIdentity();
    const frame = installed?.transport ?? null;
    const presentation = installed?.presentation ?? null;
    renderInstalledMatch(presentation);
    renderViewerBoundary();
    if (installed === null) {
      renderPendingPresentationChrome();
    }
    const replay = isReplayMode();
    const disabled =
      state.busy ||
      !frame ||
      state.shuttingDown ||
      state.resyncRequired ||
      state.offline;
    const restartControlsBlocked = recordingRestartControlsBlocked();

    elements.stepValue.textContent = presentation
      ? String(presentation.simulator_step_count ?? "—")
      : "—";
    const incomingTransition = authorizedPresentationLatestTransitionId(presentation);
    elements.transitionValue.textContent = incomingTransition
      ? String(incomingTransition)
      : "—";

    const audience = authorizedPresentationAudience(presentation) ?? "unavailable";
    elements.audienceBadge.textContent = publicAudienceLabel(audience);
    elements.audienceBadge.dataset.audience = audience;
    document.documentElement.dataset.audience = elements.audienceBadge.dataset.audience;
    document.documentElement.dataset.preset = "analysis";

    const terminal = frame !== null && isTerminal(frame);
    elements.terminalBadge.hidden = !terminal;
    if (terminal && replay) {
      const completion = isRecord(frame?.completion) ? frame.completion : {};
      const bases = asArray(completion.completion_bases);
      // Short plain words, as in live play, so the top bar keeps the whole
      // map name: the game ended ("Game Over", also when it ended on the
      // episode limit), the replay stops at the episode limit ("Time
      // Limit"), the recording stops before the game ended ("Recording
      // Ends"), or the replay is complete for another reason ("Replay End").
      elements.terminalBadge.textContent =
        completion.completion_state !== "complete"
          ? "Recording Ends"
          : bases.includes("task_terminal")
            ? "Game Over"
            : bases.includes("declared_horizon")
              ? "Time Limit"
              : "Replay End";
    } else {
      // Short, so the top bar keeps the whole map name at game over too.
      elements.terminalBadge.textContent = terminal ? "Game Over" : "Terminal";
    }

    const retainedAuthority =
      document.documentElement.dataset.presentationAuthority === "retained";
    const retainedState = state.shuttingDown
      ? "session shutting down"
      : state.busy
        ? "update pending"
        : "reconnect required";
    elements.scenarioDescription.textContent = presentation
      ? retainedAuthority
        ? replay
          ? `Last confirmed ${publicAudienceLabel(authorizedPresentationAudience(presentation))} recorded frame ${frame?.cursor?.frame_index ?? "—"} of ${frame?.cursor?.final_frame_index ?? "—"} · ${retainedState}`
          : `Last confirmed ${publicAudienceLabel(authorizedPresentationAudience(presentation))} battlefield · ${retainedState}`
        : replay
          ? `${publicAudienceLabel(authorizedPresentationAudience(presentation))} · recorded frame ${frame?.cursor?.frame_index ?? "—"} of ${frame?.cursor?.final_frame_index ?? "—"}`
          : `${publicAudienceLabel(authorizedPresentationAudience(presentation))} · authorized live presentation`
      : "Waiting for an authorized presentation.";

    const viewMode = frame?.view_mode === "agent_pov" ? "pov" : frame?.view_mode;
    if (typeof viewMode === "string") {
      elements.viewSelect.value = viewMode;
    } else {
      elements.viewSelect.value = "";
    }
    elements.viewSelect.disabled = disabled;
    syncViewSwitch();
    elements.reconnectButton.disabled =
      state.busy || state.shuttingDown || productIdentity === null;
    elements.exitButton.disabled = disabled;
    elements.exitButton.textContent = "Exit";
    elements.resetButton.disabled =
      disabled ||
      Boolean(frame?.fresh_launch_required) ||
      restartControlsBlocked ||
      liveScriptedInspectionOnly();
    const authorizedLive =
      !replay &&
      (authorizedPresentationAudience(presentation) === "researcher" ||
        authorizedPresentationAudience(presentation) === "agent_pov");
    elements.liveRangesButton.hidden = !authorizedLive;
    elements.liveRangesButton.setAttribute(
      "aria-pressed",
      String(authorizedLive && installedPresentationRangesVisible(presentation)),
    );
    elements.liveRangesButton.disabled = disabled || !authorizedLive;
    if (shellParts.copyTransitionButton !== null) {
      /** @type {HTMLButtonElement} */ (shellParts.copyTransitionButton).disabled =
        !incomingTransition;
    }
    renderPlaceLine(presentation);
    renderViewTag(presentation, frame);
    settleAgentPovTitle(replay ? null : frame);
    renderPovAgentChip(presentation, replay, disabled);
    if (shellParts.header !== null) {
      shellParts.header.dataset.brandLockup = filterOn("brand_lockup") ? "on" : "off";
    }
    if (productIdentity !== null) syncPalette();
    if (productIdentity?.product_kind === "combat_debugger") {
      syncBrandSignature();
    }
    if (shellParts.visualFiltersButton !== null && productIdentity !== null) {
      shellParts.visualFiltersButton.hidden = false;
    }
    syncTitleCard(installed !== null);
    renderRecordingControls(installed);
    renderReplayMetadata(installed);
    renderCommandAvailability();
    registerAuthorityAwareUtilityHelp();
  }

  return Object.freeze({
    renderConnection,
    applyBattlefieldBoundaryCopy,
    renderViewerBoundary,
    renderSessionToolbar,
    refreshHoverCard,
  });
}
