/**
 * @file Check the Visual Filters panel contract (src/filter-controls.js): each
 * client shows only its own switches, grouped in VISUAL_FILTER_GROUPS order,
 * each switch exactly once with its visible one-line help and its shipped
 * default ("On By Default" or "Off By Default"), every group with an
 * "N Of M On" count; the five preset buttons follow Default Configuration in
 * order and Default Configuration starts pressed; the preset status line
 * counts hand changes only among the shown switches, reads "Every switch as it
 * ships" at the default, and greys Reset only then; and the `omit` option
 * leaves a switch out of the list and its group count (the BattleClient omits
 * Panel Tokens, which changes only the DevClient's docked panels).
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  installVisualFilterControls,
  visualFilterPresetStatus,
} from "../src/filter-controls.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  setVisualFilterEnabled,
  VISUAL_FILTER_GROUPS,
  VISUAL_FILTER_PRESETS,
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
        const count = descendants(group).find(
          (node) => node.dataset?.visualFilterGroupCount !== undefined,
        );
        assert.equal(
          count.textContent,
          `${members.filter((entry) => entry.defaultEnabled).length} Of ${members.length} On`,
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
            texts.includes(entry.defaultEnabled ? "On By Default" : "Off By Default"),
          );
          const input = label.children[0];
          assert.equal(input.checked, entry.defaultEnabled);
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
