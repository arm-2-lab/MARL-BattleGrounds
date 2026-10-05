/**
 * @file Check the Visual Filter presets and the owner's naming rules for every
 * switch: four presets with Default Configuration first (the owner removed
 * Broadcast on 4 Oct 2026; its spectacle switches, Moon Motes and Death
 * Dimming among them, are now on by default); each preset's exact switch
 * choices (the approved store presets kept to the built switches); preset
 * states are full switch states that never carry View, Ranges, Layout, Motion
 * or a "palette" setting, and every preset keeps the High Contrast switch at
 * its default, on; applying a preset drops hand changes and the "changed
 * by hand" list counts the switches that differ from the preset in use; Reset
 * To Default Configuration returns the shipped defaults; and a banned-word
 * scan of every switch label, help line, group, preset text and command help
 * line (no Host, Azure, Crimson, today, Mock, Blue, Red as a team name, or the
 * retired names), plus Title Case labels and a help line that says what Off
 * shows.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { COMMAND_SET } from "../src/command-set.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  reduceVisualFilterState,
  setVisualFilterEnabled,
  VISUAL_FILTER_GROUPS,
  VISUAL_FILTER_IDS,
  VISUAL_FILTER_PRESET_IDS,
  VISUAL_FILTER_PRESETS,
  VISUAL_FILTER_REGISTRY,
  visualFilterHandChanges,
  visualFilterPresetState,
} from "../src/visual-filters.js";

/** @param {string[]} on @param {string[]} off */
function choices(on, off) {
  return Object.fromEntries([
    ...on.map((id) => [id, true]),
    ...off.map((id) => [id, false]),
  ]);
}

const EXPECTED_PRESETS = [
  ["default", "Default Configuration", {}],
  [
    "researcher",
    "Researcher",
    choices(
      [
        "aura_fields",
        "duration_status_badges",
        "aura_modifier_badges",
        "target_selection_visuals",
        "survey_frame",
        "class_letters",
        "hud_momentum",
        "spawn_shield_expiry",
        "status_clear_on_death",
        "focus_fire",
        "battle_text_detail",
        "last_breath",
      ],
      [],
    ),
  ],
  [
    "manuscript",
    "Manuscript",
    choices(
      [
        "class_letters",
        "survey_frame",
        "floor_grid",
        "spawn_waystones",
        "obstacle_relief",
        "red_zone_floors",
        "last_move_ghosts",
        "status_marks",
        "cooldown_effects",
        "recent_health_change",
        "scrolling_battle_text",
        "basic_ability_effects",
        "ultimate_ability_effects",
        "death_effects",
      ],
      [
        "map_skins",
        "moonlight",
        "obstacle_shadows",
        "class_seals",
        "soft_glow",
        "hit_reactions",
        "pov_sight_veil",
        "selected_agent_frame",
        "scorebug",
        "team_rosters",
        "fighters_standing",
        "match_moments",
        "moment_full_cards",
        "time_of_night",
        "moon_motes",
        "death_dimming",
        "camera_framing",
        "kill_cam",
        "kill_slow_motion",
      ],
    ),
  ],
  [
    "minimal",
    "Minimal",
    choices(
      [
        "floor_grid",
        "spawn_waystones",
        "red_zone_floors",
        "status_marks",
        "cooldown_effects",
        "spawn_shield",
        "stride_glide",
        "last_move_ghosts",
        "basic_ability_effects",
        "ultimate_ability_effects",
        "scrolling_battle_text",
        "recent_health_change",
        "regeneration_effects",
        "death_effects",
        "resurrection_effects",
        "respawn_wave",
        "scorebug",
        "death_announcer",
      ],
      [
        "duration_status_badges",
        "aura_modifier_badges",
        "target_selection_visuals",
        "focus_fire",
        "map_skins",
        "survey_frame",
        "obstacle_relief",
        "obstacle_shadows",
        "class_seals",
        "class_letters",
        "soft_glow",
        "hit_reactions",
        "status_application",
        "natural_status_expiry",
        "freezing_trap_break",
        "status_clear_on_death",
        "spawn_shield_expiry",
        "focus_trail",
        "aura_fields",
        "red_zone_home_fall",
        "team_rosters",
        "hud_momentum",
        "moonlight",
        "moon_motes",
        "time_of_night",
        "death_dimming",
        "match_moments",
        "moment_full_cards",
        "selected_agent_frame",
        "reach_rings",
        "next_action_thread",
        "selection_moonlight",
        "kill_cam",
        "kill_slow_motion",
        "low_life_warning",
        "vigil_pips",
        "kindled_blast",
      ],
    ),
  ],
];

// Words the owner banned from viewer-facing text, plus retired names.
const BANNED = [
  /\bHosts?\b/u,
  /\bAzure\b/u,
  /\bCrimson\b/u,
  /\btoday\b/iu,
  /\bMock\b/iu,
  /\bBlue\b/iu,
  /\bRed\b(?! Zone)/iu,
  /\bheroe?s?\b/iu,
  /\bshowroom\b/iu,
  /\bstore\b/iu,
  /Battle Director/u,
  /Explain This Frame/u,
  /Arcane Missiles/u,
  /Kindled Lance/u,
  /Stride Glide/u,
];

test("four presets, Default Configuration first, with the approved switch choices", () => {
  assert.deepEqual(VISUAL_FILTER_PRESET_IDS, [
    "default",
    "researcher",
    "manuscript",
    "minimal",
  ]);
  assert.equal(Object.isFrozen(VISUAL_FILTER_PRESETS), true);
  assert.deepEqual(
    VISUAL_FILTER_PRESETS.map(({ id, label, overrides }) => [
      id,
      label,
      { ...overrides },
    ]),
    EXPECTED_PRESETS,
  );
  for (const entry of VISUAL_FILTER_PRESETS) {
    assert.equal(Object.isFrozen(entry), true);
    assert.equal(Object.isFrozen(entry.overrides), true);
    assert.equal(typeof entry.purpose, "string");
    assert.ok(entry.purpose.length > 10);
  }
  // Broadcast is gone; the owner's defaults keep its motes and dimming on and
  // Time Of Night off (4 Oct 2026).
  assert.throws(() => visualFilterPresetState("broadcast"), RangeError);
  for (const [id, on] of /** @type {const} */ ([
    ["moon_motes", true],
    ["death_dimming", true],
    ["time_of_night", false],
  ])) {
    assert.equal(DEFAULT_VISUAL_FILTER_STATE[id], on, id);
  }
});

test("preset states are full switch states and never touch View, Ranges, Layout or Motion", () => {
  for (const [id, , overrides] of EXPECTED_PRESETS) {
    const state = visualFilterPresetState(id);
    assert.equal(Object.isFrozen(state), true);
    assert.equal(visualFilterPresetState(id), state);
    assert.deepEqual(Object.keys(state), [...VISUAL_FILTER_IDS]);
    for (const filterId of VISUAL_FILTER_IDS) {
      const expected = Object.hasOwn(overrides, filterId)
        ? overrides[filterId]
        : DEFAULT_VISUAL_FILTER_STATE[filterId];
      assert.equal(state[filterId], expected, `${id}.${filterId}`);
    }
  }
  assert.equal(visualFilterPresetState("default"), DEFAULT_VISUAL_FILTER_STATE);
  for (const forbidden of [
    "view",
    "view_mode",
    "ranges",
    "show_ranges",
    "layout",
    "motion",
    "animation",
    "reduced_motion",
    "palette",
    "theme",
  ]) {
    assert.equal(
      VISUAL_FILTER_IDS.includes(/** @type {any} */ (forbidden)),
      false,
      forbidden,
    );
  }
  for (const [id] of EXPECTED_PRESETS) {
    assert.equal(visualFilterPresetState(id).high_contrast, true, id);
  }
  assert.throws(() => visualFilterPresetState("analysis"), RangeError);
  assert.throws(() => visualFilterPresetState("Researcher"), RangeError);
  assert.throws(() => visualFilterPresetState(undefined), RangeError);
});

test("applying a preset drops hand changes; the hand-change list counts differences", () => {
  const researcher = reduceVisualFilterState(DEFAULT_VISUAL_FILTER_STATE, {
    type: "apply_preset",
    presetId: "researcher",
  });
  assert.equal(researcher, visualFilterPresetState("researcher"));
  assert.deepEqual(visualFilterHandChanges(researcher, "researcher"), []);
  const byHand = setVisualFilterEnabled(
    setVisualFilterEnabled(researcher, "moon_motes", false),
    "time_of_night",
    true,
  );
  const changed = visualFilterHandChanges(byHand, "researcher");
  assert.equal(Object.isFrozen(changed), true);
  assert.deepEqual(changed, ["moon_motes", "time_of_night"]);
  assert.equal(changed.length, 2);
  assert.equal(
    reduceVisualFilterState(byHand, { type: "apply_preset", presetId: "researcher" }),
    researcher,
  );
  // Relative to Default Configuration, the Researcher state differs in exactly
  // the switches Researcher changes from their defaults.
  const researcherOverrides = /** @type {Record<string, boolean>} */ (
    EXPECTED_PRESETS[1][2]
  );
  assert.deepEqual(
    visualFilterHandChanges(researcher, "default"),
    VISUAL_FILTER_IDS.filter(
      (id) =>
        Object.hasOwn(researcherOverrides, id) &&
        researcherOverrides[id] !== DEFAULT_VISUAL_FILTER_STATE[id],
    ),
  );
  assert.throws(
    () => visualFilterHandChanges({ aura_fields: true }, "default"),
    TypeError,
  );
  assert.throws(() => visualFilterHandChanges(researcher, "loud"), RangeError);
  assert.throws(
    () =>
      reduceVisualFilterState(researcher, {
        type: "apply_preset",
        presetId: "researcher",
        extra: true,
      }),
    /invalid shape/u,
  );
  assert.throws(
    () => reduceVisualFilterState(researcher, { type: "apply_preset", presetId: "x" }),
    RangeError,
  );
});

test("Reset To Default Configuration returns the shipped defaults from any state", () => {
  for (const id of VISUAL_FILTER_PRESET_IDS) {
    const state = setVisualFilterEnabled(
      visualFilterPresetState(id),
      "floor_grid",
      !visualFilterPresetState(id).floor_grid,
    );
    const reset = reduceVisualFilterState(state, { type: "restore_defaults" });
    assert.equal(reset, DEFAULT_VISUAL_FILTER_STATE);
    assert.deepEqual(visualFilterHandChanges(reset, "default"), []);
  }
  assert.equal(
    reduceVisualFilterState(DEFAULT_VISUAL_FILTER_STATE, {
      type: "apply_preset",
      presetId: "default",
    }),
    DEFAULT_VISUAL_FILTER_STATE,
  );
});

test("labels and help follow the owner's naming rules; help always says what Off shows", () => {
  /** @type {Array<[string, string]>} */
  const texts = [];
  for (const { id, label, help, group } of VISUAL_FILTER_REGISTRY) {
    texts.push([`${id}.label`, label], [`${id}.help`, help], [`${id}.group`, group]);
    for (const word of label.split(" ")) {
      assert.match(word, /^[A-Z0-9]/u, `${id} label "${label}" is not Title Case`);
    }
    assert.match(help, /\bOff\b/u, `${id} help must say what Off shows`);
    assert.ok(VISUAL_FILTER_GROUPS.includes(group), id);
  }
  for (const { id, label, purpose } of VISUAL_FILTER_PRESETS) {
    texts.push([`preset.${id}.label`, label], [`preset.${id}.purpose`, purpose]);
  }
  for (const { id, keycaps, jobs } of COMMAND_SET) {
    texts.push([`command.${id}.keycaps`, keycaps]);
    for (const entry of Object.values(jobs)) {
      if (entry !== null) texts.push([`command.${id}.help`, entry.help]);
    }
  }
  for (const [where, text] of texts) {
    for (const banned of BANNED) {
      assert.doesNotMatch(text, banned, `${where}: ${text}`);
    }
  }
  const labelOf = (/** @type {string} */ id) =>
    VISUAL_FILTER_REGISTRY.find((entry) => entry.id === id)?.label;
  assert.equal(labelOf("stride_glide"), "Body Glide");
  assert.equal(labelOf("kindled_blast"), "Arcane Blast During Burst");
  assert.equal(labelOf("symbol_explanations"), "Symbol Explanations");
  assert.equal(DEFAULT_VISUAL_FILTER_STATE.symbol_explanations, true);
  assert.equal(
    new Set(VISUAL_FILTER_REGISTRY.map(({ label }) => label)).size,
    VISUAL_FILTER_IDS.length,
  );
});
