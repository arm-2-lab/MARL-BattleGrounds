/**
 * @file Check three live Visual Filters switches that must do a real job.
 *
 * Contracts: Brand Signature and Palette: High Contrast are listed only in
 * live play (the BattleClient and DevClient), off by default, in the HUD and
 * Arena groups. brandSignaturePlacement (src/shell.js) puts the signature
 * only in clear space: a stack in the margin right of the battlefield picture
 * with its bottom on the floor's bottom edge at the 1440 x 900 Vision Moment,
 * the store's row under the floor when that band is 40 px or taller, and
 * nowhere (null) when no margin is big enough or a number is not finite. The
 * Title Card choice is kept per browser tab (src/filter-controls.js):
 * readStoredTitleCard and writeStoredTitleCard read and write "on" or "off"
 * and never throw; bindFilterControls puts a saved "off" back as one set
 * action; repaints save a changed value, but never before the saved value is
 * restored; a client without the Title Card switch keeps nothing.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  createFilterControls,
  readStoredTitleCard,
  TITLE_CARD_STORAGE_KEY,
  writeStoredTitleCard,
} from "../src/filter-controls.js";
import { brandSignaturePlacement } from "../src/shell.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  setVisualFilterEnabled,
  VISUAL_FILTER_REGISTRY,
  visualFiltersForProduct,
} from "../src/visual-filters.js";

/** @param {Record<string, string>} [initial] */
function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    values,
    /** @param {string} key */
    getItem: (key) => values.get(key) ?? null,
    /** @param {string} key @param {string} value */
    setItem: (key, value) => {
      values.set(key, value);
    },
  };
}

const throwingStorage = {
  getItem() {
    throw new Error("blocked");
  },
  setItem() {
    throw new Error("blocked");
  },
};

/** @param {ReadonlyArray<string>} ids @param {any} storage */
function panel(ids, storage) {
  /** @type {unknown[]} */
  const actions = [];
  const inputs = ids.map((id) => ({ dataset: { visualFilterId: id }, checked: false }));
  const node = () => ({
    textContent: "",
    disabled: false,
    addEventListener() {},
    querySelectorAll: () => [],
  });
  const options = {
    ...node(),
    querySelectorAll: (/** @type {string} */ selector) =>
      selector.includes("data-visual-filter-id") ? inputs : [],
  };
  const controls = createFilterControls(
    /** @type {any} */ ({
      elements: {
        visualFilters: node(),
        visualFilterOptions: options,
        visualFilterCount: node(),
        visualFilterPresets: node(),
        visualFilterPresetStatus: node(),
        resetVisualFiltersButton: node(),
        enableAllVisualFiltersButton: node(),
        disableAllVisualFiltersButton: node(),
      },
      state: { presentation: null },
      switches: VISUAL_FILTER_REGISTRY.filter(({ id }) => ids.includes(id)),
      installedPresentationRangesVisible: () => false,
      applyVisualFilterAction: (/** @type {unknown} */ action) => {
        actions.push(action);
        return true;
      },
      setActiveRangesVisible: () => false,
      storage,
    }),
  );
  return { controls, actions };
}

const titleCardOff = setVisualFilterEnabled(
  DEFAULT_VISUAL_FILTER_STATE,
  "title_card",
  false,
);

test("Brand Signature and Palette: High Contrast are live-only switches, off by default", () => {
  const live = visualFiltersForProduct("combat_debugger");
  const replay = visualFiltersForProduct("replay_viewer");
  for (const [id, label, group] of [
    ["brand_signature", "Brand Signature", "HUD"],
    ["high_contrast", "Palette: High Contrast", "Arena"],
  ]) {
    const entry = live.find((row) => row.id === id);
    assert.ok(entry, id);
    assert.equal(entry.label, label);
    assert.equal(entry.group, group);
    assert.equal(entry.defaultEnabled, false);
    assert.match(entry.help, /Off: /);
    assert.equal(
      replay.some((row) => row.id === id),
      false,
      id,
    );
  }
});

test("the Brand Signature stacks in the clear margin at the 1440 x 900 Vision Moment", () => {
  const place = brandSignaturePlacement({
    box: { left: 13, top: 145, right: 1087, bottom: 613 },
    field: { left: 96, top: 145, right: 1004, bottom: 613 },
    floor: { left: 110, top: 159, right: 990, bottom: 599 },
    word: { width: 74, height: 10 },
  });
  assert.ok(place);
  assert.equal(place.layout, "stack");
  // Centered between 16 px past the floor (1006) and 4 px inside the box (1083).
  assert.equal(place.left + 13, (1006 + 1083 - 74) / 2);
  assert.ok(place.left + 13 >= 1004, "starts right of the battlefield picture");
  assert.equal(place.top + 145 + 18 + 4 + 10, 599, "bottom on the floor's bottom edge");
});

test("the Brand Signature uses the store's row under the floor when that band is 40 px tall", () => {
  const place = brandSignaturePlacement({
    box: { left: 0, top: 0, right: 1000, bottom: 600 },
    field: { left: 0, top: 0, right: 1000, bottom: 540 },
    floor: { left: 20, top: 20, right: 980, bottom: 520 },
    word: { width: 74, height: 10 },
  });
  assert.deepEqual(place, {
    layout: "row",
    left: 980 - (18 + 8 + 74),
    top: (540 + 596 - 18) / 2,
  });
});

test("the Brand Signature never shows without clear room or with a bad number", () => {
  const tight = {
    box: { left: 0, top: 0, right: 1000, bottom: 520 },
    field: { left: 0, top: 0, right: 1000, bottom: 520 },
    floor: { left: 20, top: 20, right: 980, bottom: 500 },
    word: { width: 74, height: 10 },
  };
  assert.equal(brandSignaturePlacement(tight), null);
  assert.equal(
    brandSignaturePlacement({ ...tight, word: { width: Number.NaN, height: 10 } }),
    null,
  );
});

test("the saved Title Card choice reads and writes on or off and never throws", () => {
  const storage = memoryStorage();
  assert.equal(readStoredTitleCard(storage), null);
  assert.equal(writeStoredTitleCard(storage, false), true);
  assert.equal(storage.values.get(TITLE_CARD_STORAGE_KEY), "off");
  assert.equal(readStoredTitleCard(storage), false);
  assert.equal(writeStoredTitleCard(storage, true), true);
  assert.equal(readStoredTitleCard(storage), true);
  assert.equal(
    readStoredTitleCard(memoryStorage({ [TITLE_CARD_STORAGE_KEY]: "maybe" })),
    null,
  );
  assert.equal(readStoredTitleCard(throwingStorage), null);
  assert.equal(writeStoredTitleCard(throwingStorage, false), false);
  assert.equal(readStoredTitleCard(null), null);
  assert.equal(writeStoredTitleCard(undefined, true), false);
});

test("a saved Off comes back after a reload, and repaints keep the saved value current", async () => {
  const storage = memoryStorage({ [TITLE_CARD_STORAGE_KEY]: "off" });
  const { controls, actions } = panel(["title_card"], storage);
  // A repaint before the restore must not overwrite the saved Off.
  controls.renderVisualFilterControls(DEFAULT_VISUAL_FILTER_STATE, "default");
  assert.equal(storage.values.get(TITLE_CARD_STORAGE_KEY), "off");
  controls.bindFilterControls();
  assert.deepEqual(actions, []);
  await Promise.resolve();
  assert.deepEqual(actions, [{ type: "set", filterId: "title_card", enabled: false }]);
  controls.renderVisualFilterControls(titleCardOff, "default");
  assert.equal(storage.values.get(TITLE_CARD_STORAGE_KEY), "off");
  controls.renderVisualFilterControls(DEFAULT_VISUAL_FILTER_STATE, "default");
  assert.equal(storage.values.get(TITLE_CARD_STORAGE_KEY), "on");
});

test("with nothing saved the Title Card keeps its default until someone changes it", async () => {
  const storage = memoryStorage();
  const { controls, actions } = panel(["title_card"], storage);
  controls.bindFilterControls();
  await Promise.resolve();
  assert.deepEqual(actions, []);
  controls.renderVisualFilterControls(DEFAULT_VISUAL_FILTER_STATE, "default");
  assert.equal(storage.values.has(TITLE_CARD_STORAGE_KEY), false);
  controls.renderVisualFilterControls(titleCardOff, "default");
  assert.equal(storage.values.get(TITLE_CARD_STORAGE_KEY), "off");
});

test("a client without the Title Card switch, or a refused storage, keeps nothing", async () => {
  const storage = memoryStorage({ [TITLE_CARD_STORAGE_KEY]: "off" });
  const without = panel(["brand_lockup"], storage);
  without.controls.bindFilterControls();
  await Promise.resolve();
  without.controls.renderVisualFilterControls(DEFAULT_VISUAL_FILTER_STATE, "default");
  assert.deepEqual(without.actions, []);
  assert.equal(storage.values.get(TITLE_CARD_STORAGE_KEY), "off");
  const refused = panel(["title_card"], throwingStorage);
  refused.controls.bindFilterControls();
  await Promise.resolve();
  refused.controls.renderVisualFilterControls(titleCardOff, "default");
  assert.deepEqual(refused.actions, []);
});
