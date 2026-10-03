/**
 * @file The Palette choice: Moonlit (the default look) or High Contrast.
 *
 * It owns one page setting, the `data-palette` attribute on the root element.
 * `styles/palette.css` reads it: under `data-palette="high-contrast"` it
 * redefines the colour tokens of `styles.css` (chrome, arena, teams, classes,
 * life) with brighter, clearer values for projectors and low vision. It never
 * changes `--export-backdrop` or `--export-backdrop-grid`, so Export PNG keeps
 * its #111827 backdrop and grid; an exported still shows the High Contrast
 * marks over that same backdrop.
 *
 * Who calls it: main.js mounts createPaletteControl in the Visuals controls
 * and may call applyPalette directly. Enable All, Disable All and the presets
 * never touch the palette (it is not a visual filter switch). The choice is not
 * stored: every page opens in Moonlit.
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

/** The palette every page opens with. */
export const DEFAULT_PALETTE = /** @type {PaletteId} */ ("moonlit");

/** The root attribute value that palette.css matches, by palette ID (null: none). */
const ROOT_VALUE = Object.freeze({ moonlit: null, high_contrast: "high-contrast" });

/**
 * The choices offered in the Palette select, with plain labels and help text.
 *
 * @type {ReadonlyArray<Readonly<{id: PaletteId, label: string, help: string}>>}
 */
export const PALETTE_OPTIONS = Object.freeze([
  Object.freeze({
    id: "moonlit",
    label: "Moonlit",
    help: "The soft moonlit night: the default look.",
  }),
  Object.freeze({
    id: "high_contrast",
    label: "High Contrast",
    help: "Raises contrast for projectors and low vision: brighter rings, clearer stones, crisper text. A still shows the same marks with stronger shapes. The export backdrop stays the same.",
  }),
]);

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

/**
 * Build the Palette select and append it to container.
 *
 * options.container receives one `label.palette-control` holding the visible
 * text "Palette" and a `select#palette-select` with one option per
 * PALETTE_OPTIONS entry (title attribute = its help text). options.root is the
 * element applyPalette changes (default: the container's
 * `ownerDocument.documentElement`). The select starts at the root's current
 * palette. Changing it calls applyPalette, then options.onChange (default
 * none) with the new ID. Return {element, select, setValue(id)}; setValue
 * changes both the select and the page and throws RangeError for an unknown
 * ID. Side effects: DOM nodes and one change listener; no storage or timer.
 *
 * @param {{
 *   container: HTMLElement,
 *   root?: Element,
 *   onChange?: (paletteId: PaletteId) => void,
 * }} options
 * @returns {{
 *   element: HTMLLabelElement,
 *   select: HTMLSelectElement,
 *   setValue: (paletteId: PaletteId) => void,
 * }}
 */
export function createPaletteControl(options) {
  const doc = options.container.ownerDocument;
  const root = options.root ?? doc.documentElement;
  const element = doc.createElement("label");
  element.className = "palette-control";
  const text = doc.createElement("span");
  text.className = "palette-control__label";
  text.textContent = "Palette";
  const select = doc.createElement("select");
  select.id = "palette-select";
  select.className = "palette-control__select";
  for (const option of PALETTE_OPTIONS) {
    const node = doc.createElement("option");
    node.value = option.id;
    node.textContent = option.label;
    node.title = option.help;
    select.append(node);
  }
  select.value = currentPalette(root);
  select.addEventListener("change", () => {
    const applied = applyPalette(root, select.value);
    options.onChange?.(applied);
  });
  element.append(text, select);
  options.container.append(element);
  return {
    element,
    select,
    setValue(paletteId) {
      applyPalette(root, paletteId);
      select.value = paletteId;
    },
  };
}
