/**
 * @file The Palette choice: Moonlit or High Contrast (the default since 4 Oct
 * 2026, through the High Contrast switch).
 *
 * It owns one page setting, the `data-palette` attribute on the root element.
 * `styles/palette.css` reads it: under `data-palette="high-contrast"` it
 * redefines the colour tokens of `styles.css` (chrome, arena, teams, classes,
 * life) with brighter, clearer values for projectors and low vision. It never
 * changes `--export-backdrop` or `--export-backdrop-grid`, so Export PNG
 * accepts the page's backdrop as before. The arena's own surround
 * (`--battlefield`) does follow the palette, so a High Contrast PNG shows
 * #0f1522 around the floor where a Moonlit one shows #111827; the PNG's
 * provenance records the High Contrast switch.
 *
 * Who calls it: shell.js applies the palette the High Contrast switch
 * (high_contrast, a visual filter switch in all three clients, on by default
 * since 4 Oct 2026) picks, on every repaint where it changed. The Replay
 * Viewer's Palette select is gone: the switch is the one control. The choice
 * is not stored beyond the switch state.
 *
 * Boundary: colours only. It reads no frame, scene or researcher data, so it
 * cannot reveal anything in Agent POV, and switching palettes does not
 * re-render the arena or change its content.
 */

/** The palette IDs, default first. */
export const PALETTE_IDS = Object.freeze(
  /** @type {const} */ (["moonlit", "high_contrast"]),
);

/** @typedef {typeof PALETTE_IDS[number]} PaletteId */

/**
 * The palette a page has before its first repaint (no `data-palette`); the
 * High Contrast switch, on by default, then picks the palette.
 */
export const DEFAULT_PALETTE = /** @type {PaletteId} */ ("moonlit");

/** The root attribute value that palette.css matches, by palette ID (null: none). */
const ROOT_VALUE = Object.freeze({ moonlit: null, high_contrast: "high-contrast" });

/**
 * Return whether value is one of the PALETTE_IDS.
 *
 * @param {unknown} value
 * @returns {value is PaletteId}
 */
export function isPaletteId(value) {
  return typeof value === "string" && PALETTE_IDS.includes(/** @type {any} */ (value));
}

/**
 * Switch the page to a palette.
 *
 * root is the element that carries the palette (normally
 * `document.documentElement`); paletteId is "moonlit" (removes
 * `data-palette`) or "high_contrast" (sets `data-palette="high-contrast"`).
 * Return the palette ID now in force. Side effect: one DOM attribute; no
 * storage, timer or re-render.
 *
 * @param {Element} root
 * @param {unknown} paletteId
 * @returns {PaletteId}
 * @throws {RangeError} When paletteId is not one of PALETTE_IDS.
 */
export function applyPalette(root, paletteId) {
  if (!isPaletteId(paletteId)) {
    throw new RangeError(`Unknown palette: ${String(paletteId)}.`);
  }
  const value = ROOT_VALUE[paletteId];
  if (value === null) {
    root.removeAttribute("data-palette");
  } else {
    root.setAttribute("data-palette", value);
  }
  return paletteId;
}

/**
 * Return the palette now on root: "high_contrast" when its `data-palette` is
 * "high-contrast", otherwise "moonlit" (including no attribute or an unknown
 * value). Pure read of one attribute. Export provenance can record it.
 *
 * @param {Element} root
 * @returns {PaletteId}
 */
export function currentPalette(root) {
  return root.getAttribute("data-palette") === ROOT_VALUE.high_contrast
    ? "high_contrast"
    : "moonlit";
}
