/**
 * @file Build and run the Visual Filters panel (the Replay Viewer's Visuals
 * tab and the BattleClient's Visual Filters drawer): one checkbox per display
 * switch that the current client shows (visualFiltersForProduct: in live play
 * only built switches, under their V5 labels and groups; less any switch the
 * page omits, such as Panel Tokens in the BattleClient), grouped under those
 * group names, each with its one-line help visible under it and its shipped default
 * ("On By Default" or "Off By Default"); the "N Of M On" count for the panel
 * and for each group; the five preset buttons (Default Configuration first and
 * lit while in use); the preset status line; Reset To Default Configuration;
 * and Enable All and Disable All.
 *
 * Who calls it: main.js calls installVisualFilterControls once, after it has
 * read the launcher's product identity and before the first frame. It spreads
 * VISUAL_FILTER_CONTROL_HELP into its control-help table, calls
 * createFilterControls once at startup, then calls renderVisualFilterControls
 * on every repaint and bindFilterControls once.
 *
 * What it owns: the checkbox markup inside #visual-filter-options (only the
 * switches of visualFiltersForProduct for this client, grouped in
 * VISUAL_FILTER_GROUPS order and in registry order inside a group), the
 * preset buttons inside #visual-filter-presets, the hover and screen-reader help
 * of each switch (its registry help line, which always says what Off shows),
 * the "N Of M On" counts, the preset status line ("Broadcast · 2 switches
 * changed by hand"), Reset To Default Configuration (grayed while nothing
 * differs from Default Configuration), the Enable All and Disable All buttons
 * and the panel's focus return when it closes.
 *
 * Boundary: the switch state and the preset in use stay in main.js. This module
 * reads a snapshot and asks main.js to change it through applyVisualFilterAction
 * and setActiveRangesVisible; it never sends a request itself and never
 * changes any game fact. The one switch it stores is the Title Card, kept per
 * browser tab in sessionStorage (TITLE_CARD_STORAGE_KEY) as the store spec
 * asks, so turning it off holds after a reload; no other switch is stored. Presets and Reset set switches only:
 * they never touch the View (Oracle View or Agent POV), Ranges, layout or
 * motion. Enable All and Disable All also ask for the same Ranges choice, as
 * before.
 *
 * Agent POV: the panel shows the same switches in Oracle View and Agent POV.
 * Turning a switch on never reveals anything the audience may not see; the
 * battlefield still draws only its own audience's scene.
 */
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualFilterEnabled,
  LIVE_SWITCH_NAMES,
  VISUAL_FILTER_GROUPS,
  VISUAL_FILTER_PRESETS,
  VISUAL_FILTER_REGISTRY,
  visualFilterHandChanges,
  visualFiltersForProduct,
} from "./visual-filters.js";

/**
 * Help for every display switch, keyed by switch ID, built from the shared
 * registry so the panel never keeps a second copy of the wording.
 *
 * Each entry is {id, title, text}: `title` is the registry label, `text` the
 * registry help line (it always says what Off shows) and `id` the ID of the
 * hidden help element, `visual-filter-<switch id with hyphens>-help`. text fills
 * that hidden element, which the checkbox lists after visual-filters-help in its
 * aria-describedby, and the option's hover and focus tooltip (through
 * VISUAL_FILTER_CONTROL_HELP). Each ID is unique in the page. Only the switches
 * of the current client are drawn, so only their help elements exist.
 *
 * @type {Readonly<Record<string, Readonly<{id: string, title: string, text: string}>>>}
 */
export const VISUAL_FILTER_OPTION_HELP = Object.freeze(
  Object.fromEntries(
    VISUAL_FILTER_REGISTRY.map(({ id, label, help }) => [
      id,
      Object.freeze({
        id: `visual-filter-${id.replaceAll("_", "-")}-help`,
        title: label,
        text: help,
      }),
    ]),
  ),
);

/**
 * The sessionStorage key that keeps this browser tab's Title Card choice.
 *
 * The store spec makes the Title Card a page preference kept per tab: its
 * switch can only be reached after the moment it controls (the wait for the
 * first frame), so turning it off must hold across reloads, reconnects and
 * resyncs in that tab. The value is "on" or "off". A new tab, or a browser
 * that refuses storage, uses the default (on).
 */
export const TITLE_CARD_STORAGE_KEY = "marl-battlegrounds.title-card";

/**
 * Return the page's sessionStorage, or null when the browser has none or
 * refuses it (for example storage blocked for the site). Never throws.
 *
 * @returns {Pick<Storage, "getItem" | "setItem"> | null}
 */
function tabStorage() {
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

/**
 * Read the Title Card choice saved in this browser tab.
 *
 * @param {Pick<Storage, "getItem"> | null | undefined} storage Usually
 *   sessionStorage; null or undefined reads nothing.
 * @returns {boolean | null} false when "off" is saved, true when "on" is
 *   saved, null when nothing valid is saved or storage fails (the caller then
 *   keeps the default). Never throws.
 */
export function readStoredTitleCard(storage) {
  try {
    const value = storage?.getItem(TITLE_CARD_STORAGE_KEY);
    return value === "off" ? false : value === "on" ? true : null;
  } catch {
    return null;
  }
}

/**
 * Save the Title Card choice for this browser tab as "on" or "off".
 *
 * @param {Pick<Storage, "setItem"> | null | undefined} storage Usually
 *   sessionStorage; null or undefined saves nothing.
 * @param {boolean} enabled Whether the Title Card switch is on.
 * @returns {boolean} Whether the value was saved; false when storage is
 *   missing, full or refused. Never throws.
 */
export function writeStoredTitleCard(storage, enabled) {
  try {
    if (!storage) return false;
    storage.setItem(TITLE_CARD_STORAGE_KEY, enabled ? "on" : "off");
    return true;
  } catch {
    return false;
  }
}

// One sentence added to every preset and Reset tip: they set switches only.
const PRESET_KEEPS_NOTE = "The view, Ranges and motion stay as they are.";

/**
 * Control-help rows for the Visual Filters panel, in page order: the panel
 * summary, the five preset buttons (Default Configuration first), Reset To
 * Default Configuration, Enable All, Disable All, then one row per entry of
 * VISUAL_FILTER_OPTION_HELP in registry order. A switch named in
 * LIVE_SWITCH_NAMES with a live label gets two rows instead, each scoped to
 * one client by `html[data-product-kind=...]`, so the tip title always
 * matches the label that client shows (for example "Death Announcer" in the
 * BattleClient and "Fallen Feed" in the Replay Viewer). Each row is
 * `[selector, title, summary]`. main.js spreads these rows into its
 * control-help table; a row whose element is not in the page (a switch this
 * client does not show) registers nothing.
 *
 * @type {ReadonlyArray<string[]>}
 */
export const VISUAL_FILTER_CONTROL_HELP = Object.freeze([
  [
    "#visual-filters > summary",
    "Visual Filters",
    "Show or hide parts of the battlefield picture. The game and its data never change.",
  ],
  ...VISUAL_FILTER_PRESETS.map(
    ({ id, label, purpose }) =>
      /** @type {[string, string, string]} */ ([
        presetButtonSelector(id),
        label,
        `${purpose} ${PRESET_KEEPS_NOTE}`,
      ]),
  ),
  [
    "#reset-visual-filters-button",
    "Reset To Default Configuration",
    `Set every switch back to how it ships and use Default Configuration. ${PRESET_KEEPS_NOTE}`,
  ],
  [
    "#enable-all-visual-filters-button",
    "Enable All",
    "Turn on every switch and the active Ranges overlay.",
  ],
  [
    "#disable-all-visual-filters-button",
    "Disable All",
    "Turn off every switch and the active Ranges overlay.",
  ],
  ...Object.entries(VISUAL_FILTER_OPTION_HELP).flatMap(([id, help]) => {
    const liveLabel = LIVE_SWITCH_NAMES[id]?.label;
    if (liveLabel === undefined) {
      return [
        /** @type {[string, string, string]} */ ([
          `[data-visual-filter-option='${id}']`,
          help.title,
          help.text,
        ]),
      ];
    }
    // A renamed switch: each client's tip title matches the label it shows.
    return [
      /** @type {[string, string, string]} */ ([
        `html[data-product-kind='replay_viewer'] [data-visual-filter-option='${id}']`,
        help.title,
        help.text,
      ]),
      /** @type {[string, string, string]} */ ([
        `html[data-product-kind='combat_debugger'] [data-visual-filter-option='${id}']`,
        liveLabel,
        help.text,
      ]),
    ];
  }),
]);

/**
 * Return the CSS selector of one preset button. Default Configuration keeps its
 * page ID, `#default-visual-filters-button`; every other preset button is found
 * by its `data-visual-filter-preset` value. Pure.
 *
 * @param {string} presetId A preset ID from VISUAL_FILTER_PRESETS.
 * @returns {string} The selector.
 */
function presetButtonSelector(presetId) {
  return presetId === "default"
    ? "#default-visual-filters-button"
    : `[data-visual-filter-preset='${presetId}']`;
}

/**
 * Describe the preset in use for the status line and Reset button. Pure.
 *
 * Hand changes are the switches whose value in `state` differs from what the
 * preset gives (visualFilterHandChanges), counted only among `switchIds`, the
 * switches this client shows, so a switch the viewer cannot see never counts.
 * The text follows the approved store rule: with hand changes it reads
 * "<Preset> · 1 switch changed by hand" or "<Preset> · N switches changed by
 * hand"; with none it reads "<Preset> is on", or "Every switch as it ships"
 * for Default Configuration.
 *
 * @param {unknown} state A complete switch state.
 * @param {unknown} presetId The preset in use: "default", "broadcast",
 *   "researcher", "manuscript" or "minimal".
 * @param {ReadonlyArray<string>} switchIds The switch IDs this client shows
 *   (for example visualFiltersForProduct(product).map(({id}) => id)).
 * @returns {Readonly<{
 *   presetId: string,
 *   label: string,
 *   handChanges: number,
 *   text: string,
 *   atDefault: boolean,
 * }>} `label` is the preset's button label; `handChanges` the count above;
 *   `atDefault` is true only for Default Configuration with no hand change,
 *   which is when Reset To Default Configuration is grayed out.
 * @throws {TypeError} When state is not a complete switch state.
 * @throws {RangeError} When presetId is not a known preset.
 */
export function visualFilterPresetStatus(state, presetId, switchIds) {
  const shown = new Set(switchIds);
  const handChanges = visualFilterHandChanges(state, presetId).filter((id) =>
    shown.has(id),
  ).length;
  const preset = VISUAL_FILTER_PRESETS.find(({ id }) => id === presetId);
  const label = preset?.label ?? "Default Configuration";
  const text =
    handChanges > 0
      ? `${label} · ${handChanges} ${handChanges === 1 ? "switch" : "switches"} changed by hand`
      : presetId === "default"
        ? "Every switch as it ships"
        : `${label} is on`;
  return Object.freeze({
    presetId: String(presetId),
    label,
    handChanges,
    text,
    atDefault: presetId === "default" && handChanges === 0,
  });
}

/**
 * Page elements installVisualFilterControls builds into.
 *
 * @typedef {object} FilterInstallElements
 * @property {HTMLElement} visualFilterOptions `#visual-filter-options`, the
 *   checkbox list; its children are replaced.
 * @property {HTMLElement} visualFilterPresets `#visual-filter-presets`, the preset
 *   row. It holds the static Default Configuration button; the other preset
 *   buttons are added after it.
 * @property {HTMLButtonElement} defaultVisualFiltersButton
 *   `#default-visual-filters-button`, the Default Configuration preset button.
 */

/**
 * Build the page checkboxes and preset buttons for one client.
 *
 * Call once while the page loads, after the product identity is known. The
 * checkboxes are the switches of visualFiltersForProduct(product) (product
 * null, when startup was refused, shows none). They are grouped: one open
 * `<details class="visual-filters__group" data-visual-filter-group="<group>">`
 * per registry group that has a switch, in VISUAL_FILTER_GROUPS order, whose
 * summary holds the group name and an "N Of M On" count
 * (`data-visual-filter-group-count`); inside a group the switches keep
 * registry order. Replace the container's children so browser form
 * restoration cannot override a fresh page load. Each option label carries
 * `data-visual-filter-option` with its switch ID and ends with its shipped
 * default ("On By Default" or "Off By Default"); its help element (the
 * registry help line, shown under the switch) follows the label and is listed
 * in the checkbox's `aria-describedby` after `visual-filters-help`. Every
 * checkbox starts at its registry default.
 *
 * The preset row keeps the static Default Configuration button (marked with
 * `data-visual-filter-preset="default"`) and gets one more button per other
 * preset, in VISUAL_FILTER_PRESETS order, each with ID
 * `visual-filter-preset-<preset id>`, `data-visual-filter-preset` and
 * `aria-pressed`. Calling twice replaces the added buttons; it never adds a
 * second copy.
 *
 * @param {FilterInstallElements} elements The checkbox list, the preset row and
 *   the Default Configuration button.
 * @param {"combat_debugger" | "replay_viewer" | null} product The client:
 *   "combat_debugger" (BattleClient and DevClient combat screen),
 *   "replay_viewer", or null to show no switches.
 * @param {{omit?: ReadonlyArray<string>}} [options] omit (default none): switch
 *   IDs this page leaves out of the list although the product shows them, for
 *   example "panel_tokens" in the BattleClient, which has no docked panels for
 *   it to change (the DevClient keeps it). An omitted switch keeps its state
 *   and default, and presets still set it; it only gets no checkbox. Unknown
 *   IDs are ignored.
 * @returns {ReadonlyArray<import("./visual-filters.js").VisualFilterRegistration>}
 *   The frozen list of switch records now shown (empty for null).
 * @throws {TypeError} When the registry repeats an ID, or a registry default
 *   disagrees with DEFAULT_VISUAL_FILTER_STATE. Nothing is built then.
 * @throws {RangeError} When product is not one of the accepted values.
 */
export function installVisualFilterControls(elements, product, options = {}) {
  const registeredIds = VISUAL_FILTER_REGISTRY.map(({ id }) => id);
  if (new Set(registeredIds).size !== registeredIds.length) {
    throw new TypeError("Visual Filters requires unique registry entries.");
  }
  for (const { id, defaultEnabled } of VISUAL_FILTER_REGISTRY) {
    if (isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, id) !== defaultEnabled) {
      throw new TypeError(`Visual filter ${id} disagrees with the default state.`);
    }
  }
  const omitted = new Set(options.omit ?? []);
  const switches =
    product === null
      ? Object.freeze([])
      : Object.freeze(
          visualFiltersForProduct(product).filter(({ id }) => !omitted.has(id)),
        );
  const fragment = document.createDocumentFragment();
  for (const group of VISUAL_FILTER_GROUPS) {
    const members = switches.filter((entry) => entry.group === group);
    if (members.length === 0) continue;
    const section = document.createElement("details");
    section.className = "visual-filters__group";
    section.dataset.visualFilterGroup = group;
    section.open = true;
    const summary = document.createElement("summary");
    const title = document.createElement("span");
    title.textContent = group;
    const count = document.createElement("span");
    count.className = "visual-filters__group-count";
    count.dataset.visualFilterGroupCount = group;
    count.textContent = `${members.filter((entry) => entry.defaultEnabled).length} Of ${members.length} On`;
    summary.append(title, count);
    const body = document.createElement("div");
    body.className = "visual-filters__group-body";
    for (const { id, label, defaultEnabled } of members) {
      const help = VISUAL_FILTER_OPTION_HELP[id];
      const option = document.createElement("label");
      option.className = "visual-filters__option";
      option.dataset.visualFilterOption = id;
      const input = document.createElement("input");
      input.type = "checkbox";
      input.id = `visual-filter-${id.replaceAll("_", "-")}`;
      input.value = id;
      input.dataset.visualFilterId = id;
      input.setAttribute("autocomplete", "off");
      input.setAttribute("aria-describedby", `visual-filters-help ${help.id}`);
      input.defaultChecked = defaultEnabled;
      input.checked = defaultEnabled;
      const text = document.createElement("span");
      text.textContent = label;
      const shipped = document.createElement("span");
      shipped.className = "visual-filters__option-default";
      shipped.textContent = defaultEnabled ? "On By Default" : "Off By Default";
      option.append(input, text, shipped);
      const description = document.createElement("span");
      description.id = help.id;
      description.className = "visual-filters__option-help";
      description.textContent = help.text;
      body.append(option, description);
    }
    section.append(summary, body);
    fragment.append(section);
  }
  elements.visualFilterOptions.replaceChildren(fragment);

  const defaultButton = elements.defaultVisualFiltersButton;
  defaultButton.dataset.visualFilterPreset = "default";
  defaultButton.setAttribute("aria-pressed", "true");
  for (const stale of elements.visualFilterPresets.querySelectorAll(
    "button[data-visual-filter-preset]",
  )) {
    if (stale !== defaultButton) stale.remove();
  }
  for (const { id, label } of VISUAL_FILTER_PRESETS) {
    if (id === "default") continue;
    const button = document.createElement("button");
    button.type = "button";
    button.id = `visual-filter-preset-${id}`;
    button.dataset.visualFilterPreset = id;
    button.setAttribute("aria-pressed", "false");
    button.textContent = label;
    elements.visualFilterPresets.append(button);
  }
  return switches;
}

/**
 * Page elements the Visual Filters panel uses. main.js finds them once by ID.
 *
 * @typedef {object} FilterControlElements
 * @property {any} visualFilters `#visual-filters`, the panel's `<details>`.
 * @property {any} visualFilterOptions `#visual-filter-options`, the checkbox list.
 * @property {any} visualFilterCount `#visual-filter-count`, the "N Of M On" badge.
 * @property {any} visualFilterPresets `#visual-filter-presets`, the preset row.
 * @property {any} visualFilterPresetStatus `#visual-filter-preset-status`, the
 *   preset status line.
 * @property {any} resetVisualFiltersButton `#reset-visual-filters-button`.
 * @property {any} enableAllVisualFiltersButton `#enable-all-visual-filters-button`.
 * @property {any} disableAllVisualFiltersButton `#disable-all-visual-filters-button`.
 */

/**
 * Everything createFilterControls needs from main.js. Each function keeps the
 * contract documented beside its definition in main.js.
 *
 * @typedef {object} FilterControlsContext
 * @property {FilterControlElements} elements Panel elements (see above).
 * @property {{presentation: Readonly<Record<string, any>> | null}} state Live
 *   page state; only the installed presentation is read, never assigned.
 * @property {ReadonlyArray<import("./visual-filters.js").VisualFilterRegistration>}
 *   switches The switch records installVisualFilterControls returned: the
 *   checkboxes in the page, in order.
 * @property {(presentation: unknown) => boolean} installedPresentationRangesVisible
 *   Whether ranges show for this exact installed presentation.
 * @property {(action: unknown) => boolean} applyVisualFilterAction Apply one
 *   switch action (set, enable_all, disable_all, apply_preset or
 *   restore_defaults) to main.js's state; returns whether the state or the
 *   preset in use changed. A state change cancels a pending download, pauses
 *   replay play and repaints the page.
 * @property {(visible: boolean) => boolean} setActiveRangesVisible Ask for range
 *   visibility through the current audience's own route; returns whether a local
 *   change or one request started.
 * @property {Pick<Storage, "getItem" | "setItem"> | null} [storage] Where this
 *   tab's Title Card choice is kept (TITLE_CARD_STORAGE_KEY). Default: the
 *   page's sessionStorage, or none when the browser refuses it; null keeps
 *   nothing. Tests pass their own.
 */

/**
 * Create the Visual Filters panel renderers for one page.
 *
 * Call once at page startup, after installVisualFilterControls. Creating the
 * panel reads this tab's saved Title Card choice (readStoredTitleCard) and has
 * no other effect; only the returned functions touch the DOM or the storage.
 *
 * The Title Card choice is kept per browser tab, as the store spec asks:
 * bindFilterControls puts a saved "off" back into main.js's state, and from
 * then on every repaint that shows a new Title Card value saves it
 * (writeStoredTitleCard), whichever control changed it (its checkbox, a
 * preset, Reset, Enable All or Disable All). Only clients that list the Title
 * Card switch keep it.
 *
 * @param {FilterControlsContext} context Panel elements, state, the shown
 *   switches and helpers from main.js. Functions are captured once; pass stable
 *   functions or wrappers.
 * @returns {Readonly<{
 *   renderVisualFilterControls: (
 *     snapshot: typeof DEFAULT_VISUAL_FILTER_STATE,
 *     presetId: string,
 *   ) => void,
 *   applyAllVisualControls: (enabled: boolean) => void,
 *   bindFilterControls: () => void,
 * }>} The panel API. See each function's own documentation.
 */
export function createFilterControls(context) {
  const {
    elements,
    state,
    switches,
    installedPresentationRangesVisible,
    applyVisualFilterAction,
    setActiveRangesVisible,
  } = context;
  const switchIds = Object.freeze(switches.map(({ id }) => id));
  const storage = context.storage === undefined ? tabStorage() : context.storage;
  const keepsTitleCard = switchIds.includes("title_card");
  const storedTitleCard = keepsTitleCard ? readStoredTitleCard(storage) : null;
  // The Title Card value this tab has saved (its default when nothing is
  // saved), and whether the saved value is back in main.js's state. Until it
  // is, repaints save nothing, so the default never overwrites a saved "off".
  let savedTitleCard =
    storedTitleCard ?? isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, "title_card");
  let titleCardRestored = false;

  /**
   * Paint checkbox, preset and bulk-button state from one local snapshot.
   *
   * Sets each checkbox to its value in `snapshot`. The panel count reads
   * "N Of M On": it covers the switches this client shows plus the Ranges
   * toggle (on when ranges show for the installed presentation), which counts
   * as one more control. Each group's count reads "N Of M On" for its own
   * switches only. Enable All is disabled when every control is on; Disable All when
   * every control is off. The preset button of `presetId` is pressed
   * (`aria-pressed="true"`) and the others are not; the status line and Reset To
   * Default Configuration follow visualFilterPresetStatus (Reset is disabled
   * while Default Configuration is in use with no hand change). Changes the
   * DOM and, once the saved Title Card choice is restored, saves a changed
   * Title Card value for this tab; the snapshot is not changed.
   *
   * @param {typeof DEFAULT_VISUAL_FILTER_STATE} snapshot The switch state to show.
   * @param {string} presetId The preset in use (from main.js).
   * @returns {void}
   * @throws {TypeError} When the page has a different number of switch
   *   checkboxes than this client's switch list (markup and registry disagree).
   * @throws {RangeError} When presetId is not a known preset.
   */
  function renderVisualFilterControls(snapshot, presetId) {
    const inputs = /** @type {NodeListOf<HTMLInputElement>} */ (
      elements.visualFilterOptions.querySelectorAll(
        'input[type="checkbox"][data-visual-filter-id]',
      )
    );
    if (inputs.length !== switches.length) {
      throw new TypeError(
        `Visual Filters requires exactly ${switches.length} checkboxes.`,
      );
    }
    let enabledCount = 0;
    for (const input of inputs) {
      const enabled = isVisualFilterEnabled(snapshot, input.dataset.visualFilterId);
      input.checked = enabled;
      enabledCount += enabled ? 1 : 0;
    }
    const rangesEnabled = installedPresentationRangesVisible(state.presentation);
    const visibleControlCount = switches.length + 1;
    const enabledControlCount = enabledCount + (rangesEnabled ? 1 : 0);
    elements.visualFilterCount.textContent = `${enabledControlCount} Of ${visibleControlCount} On`;
    for (const count of elements.visualFilterOptions.querySelectorAll(
      "[data-visual-filter-group-count]",
    )) {
      const group = count.dataset.visualFilterGroupCount;
      const members = switches.filter((entry) => entry.group === group);
      const on = members.filter(({ id }) => isVisualFilterEnabled(snapshot, id)).length;
      const text = `${on} Of ${members.length} On`;
      if (count.textContent !== text) count.textContent = text;
    }
    elements.enableAllVisualFiltersButton.disabled =
      enabledControlCount === visibleControlCount;
    elements.disableAllVisualFiltersButton.disabled = enabledControlCount === 0;

    const status = visualFilterPresetStatus(snapshot, presetId, switchIds);
    for (const button of elements.visualFilterPresets.querySelectorAll(
      "button[data-visual-filter-preset]",
    )) {
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.visualFilterPreset === status.presetId),
      );
    }
    if (elements.visualFilterPresetStatus.textContent !== status.text) {
      elements.visualFilterPresetStatus.textContent = status.text;
    }
    elements.resetVisualFiltersButton.disabled = status.atDefault;
    if (keepsTitleCard && titleCardRestored) {
      const titleCard = isVisualFilterEnabled(snapshot, "title_card");
      if (titleCard !== savedTitleCard && writeStoredTitleCard(storage, titleCard)) {
        savedTitleCard = titleCard;
      }
    }
  }

  /**
   * Enable or disable every local switch, then request the same range visibility.
   * The range change keeps its normal local or server-confirmed authority path;
   * the two calls may repaint or send a request through their existing handlers.
   * In Agent POV the range change stays local; in Oracle View it asks Python.
   * The preset in use does not change; the switches now count as hand changes.
   *
   * @param {boolean} enabled True for Enable All, false for Disable All.
   * @returns {void}
   */
  function applyAllVisualControls(enabled) {
    applyVisualFilterAction({ type: enabled ? "enable_all" : "disable_all" });
    setActiveRangesVisible(enabled);
  }

  /**
   * Connect the panel's controls. Call exactly once per page; a second call
   * would add every listener again.
   *
   * A checkbox change applies one `set` action for its switch. A preset button
   * applies `apply_preset` for its preset (Default Configuration included);
   * Reset To Default Configuration applies `restore_defaults`. Neither touches
   * Ranges, the View, layout or motion. Enable All and Disable All change every
   * switch and ask for the same range visibility. Closing the panel while focus
   * is inside it moves focus to its summary so keyboard users are not lost. No
   * listener sends a request itself; range changes go through
   * setActiveRangesVisible.
   *
   * It also restores this tab's saved Title Card choice: in a microtask (after
   * the page's startup code has finished), a saved "off" becomes one `set`
   * action that turns the Title Card off, which counts as a hand change. After
   * that, repaints keep the saved value current.
   *
   * @returns {void}
   */
  function bindFilterControls() {
    queueMicrotask(() => {
      try {
        if (storedTitleCard === false) {
          applyVisualFilterAction({
            type: "set",
            filterId: "title_card",
            enabled: false,
          });
        }
      } finally {
        titleCardRestored = true;
      }
    });

    /**
     * Apply a checkbox change from the switch container to local switch state.
     * Ignore other event targets or inputs without a switch ID. The shared action
     * handler validates the ID, pauses replay on change, and repaints.
     *
     * @param {Event} event
     */
    const handleVisualFilterChange = (event) => {
      const input = event.target;
      if (
        !(input instanceof HTMLInputElement) ||
        input.type !== "checkbox" ||
        !input.dataset.visualFilterId
      ) {
        return;
      }
      applyVisualFilterAction({
        type: "set",
        filterId: input.dataset.visualFilterId,
        enabled: input.checked,
      });
    };

    elements.visualFilterOptions.addEventListener("change", handleVisualFilterChange);

    elements.visualFilterPresets.addEventListener(
      "click",
      (/** @type {MouseEvent} */ event) => {
        const button =
          event.target instanceof Element
            ? event.target.closest("button[data-visual-filter-preset]")
            : null;
        if (
          !(button instanceof HTMLButtonElement) ||
          !elements.visualFilterPresets.contains(button) ||
          button.disabled
        ) {
          return;
        }
        applyVisualFilterAction({
          type: "apply_preset",
          presetId: button.dataset.visualFilterPreset,
        });
      },
    );

    elements.resetVisualFiltersButton.addEventListener("click", () => {
      applyVisualFilterAction({ type: "restore_defaults" });
    });

    elements.enableAllVisualFiltersButton.addEventListener("click", () => {
      applyAllVisualControls(true);
    });

    elements.disableAllVisualFiltersButton.addEventListener("click", () => {
      applyAllVisualControls(false);
    });

    elements.visualFilters.addEventListener("toggle", () => {
      if (elements.visualFilters.open) {
        return;
      }
      const active = document.activeElement;
      const summary = elements.visualFilters.querySelector(":scope > summary");
      if (
        active instanceof Element &&
        active !== summary &&
        elements.visualFilters.contains(active) &&
        summary instanceof HTMLElement
      ) {
        summary.focus({ preventScroll: true });
      }
    });
  }

  return Object.freeze({
    renderVisualFilterControls,
    applyAllVisualControls,
    bindFilterControls,
  });
}
