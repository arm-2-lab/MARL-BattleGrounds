/**
 * @file Check the Visual Filters panel contract (src/filter-controls.js): each
 * client shows only its own switches, grouped in VISUAL_FILTER_GROUPS order,
 * each switch exactly once with its visible one-line help and its shipped
 * default ("On by Default" or "Off by Default"), every group with an
 * "N of M On" count; the five preset buttons follow Default Configuration in
 * order and Default Configuration starts pressed; the preset status line
 * counts hand changes only among the shown switches, reads "Every switch as it
 * ships" at the default, and greys Reset only then; and the `omit` option
 * leaves a switch out of the list and its group count (the BattleClient omits
 * Panel Tokens, which changes only the DevClient's docked panels). A switch
 * listed only in some game modes (Flags, Capture the Flag only) is built hidden
 * with its help line, shown and counted only while a game of its mode is
 * installed, and hidden and uncounted again after a game of another mode.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  createFilterControls,
  installVisualFilterControls,
  visualFilterPresetStatus,
} from "../src/filter-controls.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  setVisualFilterEnabled,
  VISUAL_FILTER_GROUPS,
  VISUAL_FILTER_PRESETS,
  visualFilterShownInMode,
  visualFiltersForProduct,
} from "../src/visual-filters.js";

function fakeDom() {
  /** @param {string} tagName */
  const create = (tagName) => {
    /** @type {any[]} */
    const children = [];
    const attributes = new Map();
    /** @type {any} */
    const node = {
      tagName: tagName.toUpperCase(),
      className: "",
      textContent: "",
      dataset: {},
      children,
      /** @param {...any} nodes */
      append(...nodes) {
        for (const child of nodes) {
          if (child.isFragment) children.push(...child.children);
          else children.push(child);
        }
      },
      /** @param {...any} nodes */
      replaceChildren(...nodes) {
        children.splice(0);
        this.append(...nodes);
      },
      /** @param {string} name @param {unknown} value */
      setAttribute(name, value) {
        attributes.set(name, String(value));
      },
      /** @param {string} name */
      getAttribute(name) {
        return attributes.get(name) ?? null;
      },
      querySelectorAll() {
        return [];
      },
    };
    return node;
  };
  const document = {
    createElement: create,
    createDocumentFragment() {
      const fragment = create("#fragment");
      fragment.isFragment = true;
      return fragment;
    },
  };
  return { create, document };
}

/** @param {any} root @returns {any[]} */
const descendants = (root) => [
  root,
  ...(root.children ?? []).flatMap((/** @type {any} */ child) => descendants(child)),
];

for (const product of /** @type {const} */ (["replay_viewer", "combat_debugger"])) {
  test(`${product} switches are grouped with help and defaults`, () => {
    const { create, document } = fakeDom();
    const saved = Object.getOwnPropertyDescriptor(globalThis, "document");
    Object.defineProperty(globalThis, "document", {
      configurable: true,
      value: document,
    });
    try {
      const options = create("div");
      const presets = create("div");
      const defaultButton = create("button");
      const shown = installVisualFilterControls(
        /** @type {any} */ ({
          visualFilterOptions: options,
          visualFilterPresets: presets,
          defaultVisualFiltersButton: defaultButton,
        }),
        product,
      );
      const expected = visualFiltersForProduct(product);
      assert.deepEqual(
        shown.map(({ id }) => id),
        expected.map(({ id }) => id),
      );
      const groups = options.children;
      assert.deepEqual(
        groups.map((/** @type {any} */ group) => group.dataset.visualFilterGroup),
        VISUAL_FILTER_GROUPS.filter((name) =>
          expected.some((entry) => entry.group === name),
        ),
      );
      /** @type {string[]} */
      const seen = [];
      for (const group of groups) {
        assert.equal(group.tagName, "DETAILS");
        assert.equal(group.open, true);
        const members = expected.filter(
          (entry) => entry.group === group.dataset.visualFilterGroup,
        );
        // Before any game, a mode-only switch (Flags) is built hidden and not counted.
        const listed = members.filter(({ id }) => visualFilterShownInMode(id, null));
        const count = descendants(group).find(
          (node) => node.dataset?.visualFilterGroupCount !== undefined,
        );
        assert.equal(
          count.textContent,
          `${listed.filter((entry) => entry.defaultEnabled).length} of ${listed.length} On`,
        );
        for (const label of descendants(group).filter(
          (node) => node.dataset?.visualFilterOption !== undefined,
        )) {
          const id = label.dataset.visualFilterOption;
          const entry = members.find((member) => member.id === id);
          assert.ok(entry, `${id} sits in its own group`);
          seen.push(id);
          const texts = label.children.map(
            (/** @type {any} */ child) => child.textContent,
          );
          assert.ok(texts.includes(entry.label));
          assert.ok(
            texts.includes(entry.defaultEnabled ? "On by Default" : "Off by Default"),
          );
          const input = label.children[0];
          assert.equal(input.checked, entry.defaultEnabled);
          assert.equal(label.hidden === true, id === "flags");
        }
        const helps = descendants(group).filter(
          (node) => node.className === "visual-filters__option-help",
        );
        assert.equal(helps.length, members.length);
        assert.equal(
          helps.every((/** @type {any} */ help) => help.textContent.length > 0),
          true,
        );
      }
      assert.deepEqual(seen.sort(), expected.map(({ id }) => id).sort());
      assert.equal(defaultButton.getAttribute("aria-pressed"), "true");
      assert.deepEqual(
        presets.children.map(
          (/** @type {any} */ button) => button.dataset.visualFilterPreset,
        ),
        VISUAL_FILTER_PRESETS.filter(({ id }) => id !== "default").map(({ id }) => id),
      );
    } finally {
      if (saved) Object.defineProperty(globalThis, "document", saved);
      else Reflect.deleteProperty(globalThis, "document");
    }
  });
}

test("Flags is listed and counted only while a Capture the Flag game is installed", () => {
  const switches = visualFiltersForProduct("replay_viewer");
  const inputs = switches.map(({ id }) => ({
    dataset: { visualFilterId: id },
    checked: false,
  }));
  const flagsRow = { hidden: true };
  const flagsHelp = { hidden: true };
  const node = () => ({ textContent: "", disabled: false, querySelectorAll: () => [] });
  const count = node();
  const state = { presentation: /** @type {any} */ (null) };
  const controls = createFilterControls(
    /** @type {any} */ ({
      elements: {
        visualFilters: node(),
        visualFilterOptions: {
          querySelectorAll: (/** @type {string} */ selector) =>
            selector.includes("data-visual-filter-id")
              ? inputs
              : selector.includes('data-visual-filter-option="flags"')
                ? [flagsRow, flagsHelp]
                : [],
        },
        visualFilterCount: count,
        visualFilterPresets: node(),
        visualFilterPresetStatus: node(),
        resetVisualFiltersButton: node(),
        enableAllVisualFiltersButton: node(),
        disableAllVisualFiltersButton: node(),
      },
      state,
      switches,
      installedPresentationRangesVisible: () => false,
      applyVisualFilterAction: () => true,
      setActiveRangesVisible: () => false,
      storage: null,
    }),
  );
  /** @param {number | null} mode */
  const expected = (mode) => {
    const listed = switches.filter(({ id }) => visualFilterShownInMode(id, mode));
    return `${listed.filter((entry) => entry.defaultEnabled).length} of ${listed.length + 1} On`;
  };
  /** @param {number | null} taskMode */
  const show = (taskMode) => {
    state.presentation =
      taskMode === null ? null : { match_summary: { task_mode: taskMode } };
    controls.renderVisualFilterControls(DEFAULT_VISUAL_FILTER_STATE, "default");
    return [count.textContent, flagsRow.hidden, flagsHelp.hidden];
  };
  // Team Deathmatch lists what it listed before Flags existed, so its count
  // leaves Flags out; Capture the Flag lists Flags and leaves out the two Red
  // Zone switches.
  assert.equal(expected(1), expected(null));
  assert.notEqual(expected(3), expected(1));
  assert.deepEqual(show(null), [expected(null), true, true]);
  assert.deepEqual(show(1), [expected(1), true, true]);
  assert.deepEqual(show(3), [expected(3), false, false]);
  // The DevClient can install a Team Deathmatch game after a Capture the Flag one.
  assert.deepEqual(show(1), [expected(1), true, true]);
  // A hand change to the hidden switch is not counted in another mode.
  const flagsOff = setVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, "flags", false);
  assert.equal(
    visualFilterPresetStatus(flagsOff, "default", ["moonlight"]).atDefault,
    true,
  );
});

test("omit leaves a switch out of the list and its group count", () => {
  const { create, document } = fakeDom();
  const saved = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: document,
  });
  try {
    const options = create("div");
    const shown = installVisualFilterControls(
      /** @type {any} */ ({
        visualFilterOptions: options,
        visualFilterPresets: create("div"),
        defaultVisualFiltersButton: create("button"),
      }),
      "combat_debugger",
      { omit: ["panel_tokens", "not_a_switch"] },
    );
    const live = visualFiltersForProduct("combat_debugger");
    assert.ok(live.some(({ id }) => id === "panel_tokens"));
    assert.deepEqual(
      shown.map(({ id }) => id),
      live.filter(({ id }) => id !== "panel_tokens").map(({ id }) => id),
    );
    const labels = descendants(options).filter(
      (node) => node.dataset?.visualFilterOption !== undefined,
    );
    assert.equal(
      labels.some((node) => node.dataset.visualFilterOption === "panel_tokens"),
      false,
    );
    assert.equal(labels.length, shown.length);
  } finally {
    if (saved) Object.defineProperty(globalThis, "document", saved);
    else Reflect.deleteProperty(globalThis, "document");
  }
});

test("preset status counts only shown switches and greys Reset at the default", () => {
  const replayIds = visualFiltersForProduct("replay_viewer").map(({ id }) => id);
  const atDefault = visualFilterPresetStatus(
    DEFAULT_VISUAL_FILTER_STATE,
    "default",
    replayIds,
  );
  assert.equal(atDefault.text, "Every switch as it ships");
  assert.equal(atDefault.atDefault, true);
  const changed = setVisualFilterEnabled(
    DEFAULT_VISUAL_FILTER_STATE,
    replayIds[0],
    !DEFAULT_VISUAL_FILTER_STATE[
      /** @type {keyof typeof DEFAULT_VISUAL_FILTER_STATE} */ (replayIds[0])
    ],
  );
  const handChanged = visualFilterPresetStatus(changed, "default", replayIds);
  assert.equal(handChanged.handChanges, 1);
  assert.equal(handChanged.text, "Default Configuration · 1 switch changed by hand");
  assert.equal(handChanged.atDefault, false);
  assert.equal(visualFilterPresetStatus(changed, "default", []).handChanges, 0);
  assert.throws(
    () => visualFilterPresetStatus(changed, "nonsense", replayIds),
    RangeError,
  );
});
