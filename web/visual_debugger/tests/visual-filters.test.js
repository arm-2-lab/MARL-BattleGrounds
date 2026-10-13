/**
 * @file Check the Visual Filter registry, defaults, immutable updates and the
 * paint-part table: 99 switches in fixed order (the 20 earlier switches first,
 * then the switches of the V5-picked looks, including the live-only Brand
 * Signature, High Contrast in all three clients, and Momentum, an interface
 * switch, then the Replay Viewer's Next Step Ghost and All Staged Orders,
 * both off), with no switch for an unbuilt part such as instant replay, none
 * of the removed Chronicle and Scribe's Ledger switches (Fight Chapters,
 * Bookmarks, Ledger Heraldry, Actions in Words, Seen and Not Seen, Reference
 * Chip) and no Camera Inset; the owner's 4 Oct 2026 defaults (High Contrast,
 * Class Seals, Moon Motes, Death Dimming, Spawn Shield Expiry, Saved Beat,
 * Champion Streaks, Momentum, Kill Cam and Kill Slow Motion on; Moonlight,
 * Floor Grid and Time of Night off); each
 * with its exact label, group, default, surface (arena or interface) and
 * products; the twelve V5 switches the shared looks draw (Class Seals, Kill
 * Credit Glint, Saved Beat, Last Breath, Battle Text Detail, Aura Glints,
 * Barrier Held, Spawn Shield Expiry, Status Clear on Death, Freezing Trap
 * Break, Champion Streaks and Momentum) listed in both live play and the
 * Replay Viewer; 79 switches on by
 * default; the v4 paint key; one switch list per product (95 in the Replay
 * Viewer); strict reducer and
 * state validation; and paint parts read from each look module's PAINT_PARTS:
 * one owner per part, a repeated part or a malformed list throws, every part
 * drawn before V5 still classifies to its original switch (so every earlier
 * switch still owns a drawing part), and unknown or malformed parts fail closed.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  buildPaintPartTable,
  classifyVisualPaintPart,
  DEFAULT_VISUAL_FILTER_STATE,
  disableAllVisualFilters,
  enableAllVisualFilters,
  isVisualFilterEnabled,
  isVisualPaintPartEnabled,
  reduceVisualFilterState,
  setVisualFilterEnabled,
  VISUAL_FILTER_GROUPS,
  VISUAL_FILTER_IDS,
  VISUAL_FILTER_REGISTRY,
  visualFilterPaintKey,
  visualFiltersForProduct,
  visualPaintPartRegistry,
} from "../src/visual-filters.js";

// id, label, group, default, surface, products ("both" means both products).
const EXPECTED_FILTERS = Object.freeze([
  ["aura_fields", "Aura Fields", "Bodies", false, "arena", "both"],
  ["aura_modifier_badges", "Aura Multipliers", "Diagnostics", false, "arena", "both"],
  ["duration_status_badges", "Status Plates", "Diagnostics", false, "arena", "both"],
  ["spawn_shield", "Spawn Shield", "Bodies", true, "arena", "both"],
  [
    "target_selection_visuals",
    "Target Selection Visuals",
    "Selection",
    true,
    "arena",
    "both",
  ],
  ["basic_ability_effects", "Basic Ability Effects", "Combat", true, "arena", "both"],
  [
    "ultimate_ability_effects",
    "Ultimate Ability Effects",
    "Combat",
    true,
    "arena",
    "both",
  ],
  ["regeneration_effects", "Regeneration Effects", "Combat", true, "arena", "both"],
  ["cooldown_effects", "Cooldown Effects", "Bodies", true, "arena", "both"],
  ["status_application", "Status Application", "Bodies", true, "arena", "both"],
  ["natural_status_expiry", "Natural Status Expiry", "Bodies", true, "arena", "both"],
  ["freezing_trap_break", "Freezing Trap Break", "Bodies", true, "arena", "both"],
  ["status_clear_on_death", "Status Clear on Death", "Bodies", false, "arena", "both"],
  ["death_effects", "Death Effects", "Life and Death", true, "arena", "both"],
  ["respawn_wave", "Respawn Wave", "Life and Death", true, "arena", "both"],
  [
    "resurrection_effects",
    "Resurrection Effects",
    "Life and Death",
    true,
    "arena",
    "both",
  ],
  [
    "spawn_shield_expiry",
    "Spawn Shield Expiry",
    "Life and Death",
    true,
    "arena",
    "both",
  ],
  ["scrolling_battle_text", "Scrolling Battle Text", "Combat", true, "arena", "both"],
  ["death_announcer", "Fallen Feed", "Life and Death", true, "arena", "both"],
  ["red_zone_floors", "Red Zone Floors", "Arena", true, "arena", "both"],
  ["map_skins", "Map Skins", "Arena", true, "arena", "both"],
  ["moonlight", "Moonlight", "Arena", false, "arena", "both"],
  ["floor_grid", "Floor Grid", "Arena", false, "arena", "both"],
  ["spawn_waystones", "Spawn Waystones", "Arena", true, "arena", "both"],
  ["survey_frame", "Survey Frame", "Arena", false, "arena", "both"],
  ["arena_rim", "Arena Rim", "Arena", false, "arena", "both"],
  ["obstacle_skins", "Obstacle Skins", "Arena", true, "arena", "both"],
  ["obstacle_relief", "Obstacle Relief", "Arena", true, "arena", "both"],
  ["obstacle_shadows", "Obstacle Shadows", "Arena", true, "arena", "both"],
  ["red_zone_home_fall", "Red Zone Home Fall", "Moments", true, "arena", "both"],
  ["soft_glow", "Soft Glow", "Combat", true, "arena", "both"],
  ["moon_motes", "Moon Motes", "Arena", true, "arena", "both"],
  ["time_of_night", "Time of Night", "Arena", false, "arena", "both"],
  ["death_dimming", "Death Dimming", "Moments", true, "arena", "both"],
  ["pov_sight_veil", "Sight Reach Veil", "Vision", true, "arena", "both"],
  ["pov_recipient_rim", "Agent POV Rim", "Vision", true, "arena", "replay_viewer"],
  ["sight_change_marks", "Sight Change Marks", "Vision", true, "arena", "both"],
  ["life_vessels", "Life Vessels", "Bodies", true, "arena", "both"],
  ["class_seals", "Class Seals", "Bodies", true, "arena", "both"],
  ["recent_health_change", "Recent Health Change", "Combat", true, "arena", "both"],
  ["low_life_warning", "Low Life Warning", "Bodies", true, "arena", "both"],
  ["hit_reactions", "Hit Reactions", "Combat", true, "arena", "both"],
  ["fallen_step_aside", "Fallen Step Aside", "Bodies", true, "arena", "both"],
  ["class_letters", "Class Letters", "Bodies", false, "arena", "both"],
  ["life_segments", "Life Segments", "Bodies", false, "arena", "both"],
  ["stride_glide", "Body Glide", "Motion", true, "arena", "both"],
  ["scholar_long_tick", "Longer Tick", "Motion", true, "arena", "both"],
  ["step_glide", "Step Glide", "Motion", true, "arena", "replay_viewer"],
  ["last_move_ghosts", "Start Spots", "Motion", true, "arena", "both"],
  ["phantom_afterimage", "Phantom Afterimage", "Motion", true, "arena", "both"],
  ["focus_trail", "Focus Trail", "Motion", true, "arena", "both"],
  ["home_end_marks", "Home End Marks", "Arena", true, "arena", "both"],
  ["lore_emblems", "Lore Emblems", "Bodies", true, "arena", "both"],
  ["status_marks_icons", "Status Icons From Marks", "HUD", true, "interface", "both"],
  ["status_marks", "Status Marks", "Bodies", true, "arena", "both"],
  ["aura_glints", "Aura Glints", "Combat", false, "arena", "both"],
  ["barrier_held", "Barrier Held", "Moments", true, "arena", "both"],
  ["kindled_blast", "Arcane Blast During Burst", "Combat", true, "arena", "both"],
  ["close_range_strokes", "Close-Range Strokes", "Combat", true, "arena", "both"],
  ["saved_beat", "Saved Beat", "Moments", true, "arena", "both"],
  ["last_breath", "Last Breath", "Combat", false, "arena", "both"],
  ["battle_text_detail", "Battle Text Detail", "Combat", false, "arena", "both"],
  ["vigil_pips", "Vigil Pips", "Life and Death", true, "arena", "both"],
  ["kill_credit_glint", "Kill Credit Glint", "Life and Death", false, "arena", "both"],
  ["match_moments", "Match Moments", "Moments", true, "arena", "both"],
  ["moment_full_cards", "Full Lineup and Verdict", "Moments", true, "arena", "both"],
  ["champion_streaks", "Champion Streaks", "Moments", true, "arena", "both"],
  ["camera_framing", "Camera Framing", "Camera", true, "interface", "both"],
  ["camera_change_tip", "Camera Change Tip", "Camera", true, "interface", "both"],
  ["camera_tip", "New Camera Tip", "Camera", true, "interface", "replay_viewer"],
  [
    "auto_frame_fights",
    "Auto-Frame Fights",
    "Camera",
    true,
    "interface",
    "replay_viewer",
  ],
  ["kill_cam", "Kill Cam", "Camera", true, "interface", "replay_viewer"],
  [
    "kill_slow_motion",
    "Kill Slow Motion",
    "Camera",
    true,
    "interface",
    "replay_viewer",
  ],
  ["scorebug", "Scorebug", "HUD", true, "interface", "both"],
  ["team_rosters", "Team Rosters", "HUD", true, "interface", "both"],
  ["fighters_standing", "Fighters Standing", "HUD", true, "interface", "replay_viewer"],
  ["hud_momentum", "Momentum", "HUD", true, "interface", "both"],
  ["symbol_explanations", "Symbol Explanations", "HUD", true, "interface", "both"],
  ["reach_rings", "Reach Rings", "Selection", true, "arena", "both"],
  ["next_action_thread", "Next Action Thread", "Selection", true, "arena", "both"],
  ["selection_moonlight", "Selection Moonlight", "Selection", true, "arena", "both"],
  [
    "selected_agent_frame",
    "Selected Agent Frame",
    "Panels",
    true,
    "interface",
    "replay_viewer",
  ],
  [
    "event_timeline_marks",
    "Event Timeline Marks",
    "HUD",
    true,
    "interface",
    "replay_viewer",
  ],
  ["beat_label", "Beat Label", "HUD", true, "interface", "replay_viewer"],
  ["freeze_on_pause", "Freeze on Pause", "HUD", true, "interface", "replay_viewer"],
  ["hide_marks_ahead", "Hide Marks Ahead", "HUD", false, "interface", "replay_viewer"],
  ["class_guide", "Class Guide", "Panels", true, "interface", "both"],
  ["ledger_summary", "Match Summary", "Panels", true, "interface", "replay_viewer"],
  ["panel_tokens", "Panel Tokens", "Panels", true, "interface", "both"],
  ["brand_lockup", "Brand Lockup", "HUD", true, "interface", "both"],
  ["brand_signature", "Brand Signature", "HUD", false, "interface", "combat_debugger"],
  ["title_card", "Title Card", "HUD", true, "interface", "both"],
  ["command_marks", "Command Marks", "Bodies", true, "arena", "combat_debugger"],
  ["reach_marks", "Reach Marks", "Combat", true, "arena", "combat_debugger"],
  ["step_ghost", "Step Ghost", "Motion", true, "arena", "combat_debugger"],
  ["focus_fire", "Focus Tally", "Diagnostics", false, "arena", "both"],
  ["high_contrast", "High Contrast", "Arena", true, "arena", "both"],
  [
    "next_step_ghost",
    "Next Step Ghost",
    "Replay Tools",
    false,
    "arena",
    "replay_viewer",
  ],
  [
    "all_staged_orders",
    "All Staged Orders",
    "Replay Tools",
    false,
    "arena",
    "replay_viewer",
  ],
]);

// The 20 switches the clients had before V5, in their original order.
const EARLIER_FILTER_IDS = Object.freeze(
  EXPECTED_FILTERS.slice(0, 20).map(([id]) => /** @type {string} */ (id)),
);
const INITIAL_FILTER_IDS = EXPECTED_FILTERS.filter(([, , , enabled]) => enabled).map(
  ([id]) => /** @type {string} */ (id),
);
const ALL_ENABLED = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);

// The 35 drawable parts the renderer drew before V5, each with its switch.
// They now come from the look modules and must keep their switches.
const EARLIER_PAINT_PARTS = Object.freeze([
  [{ surface: "durable", kind: "aura_field" }, "aura_fields"],
  [{ surface: "durable", kind: "aura_modifier_badge" }, "aura_modifier_badges"],
  [{ surface: "durable", kind: "duration_status_badge" }, "duration_status_badges"],
  [{ surface: "durable", kind: "pov_duration_status_badge" }, "duration_status_badges"],
  [{ surface: "durable", kind: "spawn_shield" }, "spawn_shield"],
  [{ surface: "durable", kind: "cooldown_badge" }, "cooldown_effects"],
  [{ surface: "durable", kind: "selection_reticle" }, "target_selection_visuals"],
  [{ surface: "durable", kind: "selected_pair_legality" }, "target_selection_visuals"],
  [{ surface: "durable", kind: "red_zone_floor" }, "red_zone_floors"],
  ...["ability", "semantic"].flatMap((part) => [
    [
      { surface: "transient", kind: "activation", component: "basic", part },
      "basic_ability_effects",
    ],
    [
      { surface: "transient", kind: "activation", component: "ultimate", part },
      "ultimate_ability_effects",
    ],
  ]),
  ...["damage", "healing", "unchanged"].flatMap((outcome) =>
    ["effect", "battle_text", "recipient_text"].map((part) => [
      { surface: "transient", kind: "net_health", outcome, part },
      "scrolling_battle_text",
    ]),
  ),
  [
    { surface: "transient", kind: "regeneration", part: "effect" },
    "regeneration_effects",
  ],
  [
    { surface: "transient", kind: "regeneration", part: "battle_text" },
    "scrolling_battle_text",
  ],
  [{ surface: "transient", kind: "cooldown", semantic: "started" }, "cooldown_effects"],
  [{ surface: "transient", kind: "cooldown", semantic: "ready" }, "cooldown_effects"],
  ...[
    ["applied", "status_application"],
    ["expired", "natural_status_expiry"],
    ["trap_broken", "freezing_trap_break"],
    ["cleared_by_death", "status_clear_on_death"],
  ].map(([lifecycle, filter]) => [
    { surface: "transient", kind: "status_lifecycle", lifecycle, part: "effect" },
    filter,
  ]),
  [{ surface: "transient", kind: "death_effect" }, "death_effects"],
  [{ surface: "transient", kind: "death_announcement" }, "death_announcer"],
  [{ surface: "transient", kind: "respawn_wave" }, "respawn_wave"],
  [{ surface: "transient", kind: "resurrection_effect" }, "resurrection_effects"],
  [{ surface: "transient", kind: "spawn_shield_expiry" }, "spawn_shield_expiry"],
]);

test("locked registry exposes 99 switches with exact labels, groups and defaults", () => {
  assert.deepEqual(
    VISUAL_FILTER_REGISTRY.map(
      ({ id, label, group, defaultEnabled, surface, products }) => [
        id,
        label,
        group,
        defaultEnabled,
        surface,
        products.length === 2 ? "both" : products[0],
      ],
    ),
    EXPECTED_FILTERS,
  );
  assert.deepEqual(
    VISUAL_FILTER_IDS,
    EXPECTED_FILTERS.map(([id]) => id),
  );
  assert.equal(new Set(VISUAL_FILTER_IDS).size, 99);
  assert.equal(INITIAL_FILTER_IDS.length, 79);
  assert.deepEqual(EARLIER_FILTER_IDS, [
    "aura_fields",
    "aura_modifier_badges",
    "duration_status_badges",
    "spawn_shield",
    "target_selection_visuals",
    "basic_ability_effects",
    "ultimate_ability_effects",
    "regeneration_effects",
    "cooldown_effects",
    "status_application",
    "natural_status_expiry",
    "freezing_trap_break",
    "status_clear_on_death",
    "death_effects",
    "respawn_wave",
    "resurrection_effects",
    "spawn_shield_expiry",
    "scrolling_battle_text",
    "death_announcer",
    "red_zone_floors",
  ]);
  // V5 turns these earlier switches on by default (plan P9 and the Rune Zones pick).
  for (const id of [
    "target_selection_visuals",
    "status_application",
    "natural_status_expiry",
    "freezing_trap_break",
  ]) {
    assert.equal(DEFAULT_VISUAL_FILTER_STATE[id], true, id);
  }
  for (const removed of [
    "kindled_lance",
    "arcane_missiles",
    "red_zone_home_fall_preview",
    "barrier_held_preview",
    "pov_preview",
    "pov_shared_obs",
    "pov_hide_unseen",
    "camera_inset",
    "moment_examples",
    "devclient_turn_deck",
    "director_camera",
    "layout",
    "fight_chapters",
    "bookmarks",
    "ledger_heraldry",
    "plain_actions",
    "seen_grouping",
    "reference_chip",
  ]) {
    assert.equal(
      VISUAL_FILTER_IDS.includes(/** @type {any} */ (removed)),
      false,
      removed,
    );
  }
  assert.equal(Object.isFrozen(VISUAL_FILTER_REGISTRY), true);
  assert.equal(VISUAL_FILTER_REGISTRY.every(Object.isFrozen), true);
  assert.equal(
    VISUAL_FILTER_REGISTRY.every(({ products }) => Object.isFrozen(products)),
    true,
  );
  assert.deepEqual(
    [...new Set(VISUAL_FILTER_REGISTRY.map(({ group }) => group))].sort(),
    [...VISUAL_FILTER_GROUPS].sort(),
  );
});

test("each product lists only the switches that matter there", () => {
  const live = visualFiltersForProduct("combat_debugger");
  const replay = visualFiltersForProduct("replay_viewer");
  assert.equal(Object.isFrozen(live), true);
  assert.deepEqual(
    live.map(({ id }) => id),
    EXPECTED_FILTERS.filter(([, , , , , products]) => products !== "replay_viewer").map(
      ([id]) => id,
    ),
  );
  assert.deepEqual(
    replay.map(({ id }) => id),
    EXPECTED_FILTERS.filter(
      ([, , , , , products]) => products !== "combat_debugger",
    ).map(([id]) => id),
  );
  assert.equal(replay.length, 95);
  for (const id of [
    "camera_tip",
    "kill_cam",
    "step_glide",
    "next_step_ghost",
    "all_staged_orders",
  ]) {
    assert.equal(
      live.some((entry) => entry.id === id),
      false,
      id,
    );
  }
  for (const id of ["command_marks", "reach_marks"]) {
    assert.equal(
      replay.some((entry) => entry.id === id),
      false,
      id,
    );
  }
  for (const id of [
    "class_seals",
    "kill_credit_glint",
    "saved_beat",
    "last_breath",
    "battle_text_detail",
    "aura_glints",
    "barrier_held",
    "spawn_shield_expiry",
    "status_clear_on_death",
    "freezing_trap_break",
    "champion_streaks",
    "hud_momentum",
  ]) {
    assert.equal(
      live.some((entry) => entry.id === id),
      true,
      id,
    );
    assert.equal(
      replay.some((entry) => entry.id === id),
      true,
      id,
    );
  }
  assert.throws(() => visualFiltersForProduct("battle_client"), RangeError);
});

test("default state is immutable, exact, and enabled through the public helper", () => {
  assert.equal(Object.isFrozen(DEFAULT_VISUAL_FILTER_STATE), true);
  assert.deepEqual(Object.keys(DEFAULT_VISUAL_FILTER_STATE), VISUAL_FILTER_IDS);
  for (const id of VISUAL_FILTER_IDS) {
    assert.equal(DEFAULT_VISUAL_FILTER_STATE[id], INITIAL_FILTER_IDS.includes(id));
    assert.equal(
      isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, id),
      INITIAL_FILTER_IDS.includes(id),
    );
  }
  assert.equal(Reflect.set(DEFAULT_VISUAL_FILTER_STATE, "aura_fields", false), false);
});

test("set and bulk actions return frozen states without mutating their input", () => {
  const before = JSON.stringify(DEFAULT_VISUAL_FILTER_STATE);
  const disabled = setVisualFilterEnabled(
    DEFAULT_VISUAL_FILTER_STATE,
    "spawn_shield",
    false,
  );
  assert.notEqual(disabled, DEFAULT_VISUAL_FILTER_STATE);
  assert.equal(Object.isFrozen(disabled), true);
  assert.equal(disabled.aura_fields, false);
  assert.equal(disabled.spawn_shield, false);
  assert.equal(JSON.stringify(DEFAULT_VISUAL_FILTER_STATE), before);
  assert.equal(setVisualFilterEnabled(disabled, "spawn_shield", false), disabled);
  assert.equal(enableAllVisualFilters(disabled), ALL_ENABLED);
  assert.equal(enableAllVisualFilters(ALL_ENABLED), ALL_ENABLED);
  assert.equal(enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE), ALL_ENABLED);
  const allDisabled = disableAllVisualFilters(disabled);
  assert.equal(Object.isFrozen(allDisabled), true);
  assert.equal(
    VISUAL_FILTER_IDS.every((id) => allDisabled[id] === false),
    true,
  );
  assert.equal(disableAllVisualFilters(allDisabled), allDisabled);
  assert.throws(
    () => setVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, "spawn_shield", 1),
    /must be a boolean/u,
  );
});

test("strict reducer accepts only exact set, bulk, default and preset actions", () => {
  const disabled = reduceVisualFilterState(DEFAULT_VISUAL_FILTER_STATE, {
    type: "set",
    filterId: "scrolling_battle_text",
    enabled: false,
  });
  assert.equal(disabled.scrolling_battle_text, false);
  assert.equal(reduceVisualFilterState(disabled, { type: "enable_all" }), ALL_ENABLED);
  const allDisabled = reduceVisualFilterState(disabled, { type: "disable_all" });
  assert.equal(
    VISUAL_FILTER_IDS.every((id) => allDisabled[id] === false),
    true,
  );
  for (const changed of [disabled, allDisabled, ALL_ENABLED]) {
    assert.equal(
      reduceVisualFilterState(changed, { type: "restore_defaults" }),
      DEFAULT_VISUAL_FILTER_STATE,
    );
  }
  assert.equal(DEFAULT_VISUAL_FILTER_STATE.death_announcer, true);
  assert.equal(DEFAULT_VISUAL_FILTER_STATE.cooldown_effects, true);
  assert.equal(DEFAULT_VISUAL_FILTER_STATE.red_zone_floors, true);
  assert.equal(DEFAULT_VISUAL_FILTER_STATE.symbol_explanations, true);
  assert.throws(
    () =>
      reduceVisualFilterState(DEFAULT_VISUAL_FILTER_STATE, {
        type: "set",
        filterId: "aura_fields",
        enabled: false,
        unexpected: true,
      }),
    /invalid shape/u,
  );
  assert.throws(
    () => reduceVisualFilterState(DEFAULT_VISUAL_FILTER_STATE, { type: "unknown" }),
    /Unknown visual filter action/u,
  );
  assert.throws(
    () => reduceVisualFilterState(DEFAULT_VISUAL_FILTER_STATE, null),
    /tagged object/u,
  );
});

test("state validation and paint-key serialization are strict and deterministic", () => {
  const disabled = ["aura_fields", "scrolling_battle_text"].reduce(
    (state, filterId) => setVisualFilterEnabled(state, filterId, false),
    DEFAULT_VISUAL_FILTER_STATE,
  );
  assert.equal(
    visualFilterPaintKey(DEFAULT_VISUAL_FILTER_STATE),
    "visual-filters-v4:000111111111011111111001001111110111111111100111111111101111001011111111111111111111101111011110100",
  );
  assert.equal(
    visualFilterPaintKey(disabled),
    "visual-filters-v4:000111111111011110111001001111110111111111100111111111101111001011111111111111111111101111011110100",
  );
  assert.equal(
    visualFilterPaintKey(Object.fromEntries([...Object.entries(disabled)].reverse())),
    visualFilterPaintKey(disabled),
  );
  assert.equal(
    visualFilterPaintKey(ALL_ENABLED),
    `visual-filters-v4:${"1".repeat(VISUAL_FILTER_IDS.length)}`,
  );
  assert.throws(
    () => visualFilterPaintKey({ ...DEFAULT_VISUAL_FILTER_STATE, extra: true }),
    /invalid shape/u,
  );
  const missing = { ...DEFAULT_VISUAL_FILTER_STATE };
  delete missing.focus_fire;
  assert.throws(() => visualFilterPaintKey(missing), /invalid shape/u);
  assert.throws(
    () => visualFilterPaintKey({ ...DEFAULT_VISUAL_FILTER_STATE, map_skins: "yes" }),
    /must be a boolean/u,
  );
  assert.throws(
    () =>
      isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, "future_unregistered_filter"),
    /Unknown visual filter/u,
  );
});

test("every registered paint part has one owner and every earlier switch owns a part", () => {
  const registry = visualPaintPartRegistry();
  assert.equal(visualPaintPartRegistry(), registry);
  assert.equal(Object.isFrozen(registry), true);
  const serializedParts = new Set();
  const usedFilters = new Set();
  for (const registration of registry) {
    assert.equal(Object.isFrozen(registration), true);
    assert.equal(Object.isFrozen(registration.tag), true);
    assert.equal(typeof registration.owner, "string");
    assert.ok(registration.owner.length > 0);
    const serialized = JSON.stringify(
      Object.entries(registration.tag).sort(([left], [right]) =>
        left.localeCompare(right),
      ),
    );
    assert.equal(serializedParts.has(serialized), false, serialized);
    serializedParts.add(serialized);
    assert.equal(classifyVisualPaintPart(registration.tag), registration.filterId);
    assert.equal(
      isVisualPaintPartEnabled(DEFAULT_VISUAL_FILTER_STATE, registration.tag),
      INITIAL_FILTER_IDS.includes(registration.filterId),
    );
    assert.equal(VISUAL_FILTER_IDS.includes(registration.filterId), true);
    usedFilters.add(registration.filterId);
  }
  for (const id of EARLIER_FILTER_IDS) {
    assert.equal(usedFilters.has(id), true, id);
  }
  // Every part drawn before V5 is still classified, with its original switch.
  assert.equal(EARLIER_PAINT_PARTS.length, 35);
  for (const [part, filter] of EARLIER_PAINT_PARTS) {
    assert.equal(classifyVisualPaintPart(part), filter, JSON.stringify(part));
  }
});

test("the paint-part builder keeps one owner per part and fails on conflicts", () => {
  const red = {
    part: { surface: "durable", kind: "red_zone_floor" },
    filter: "red_zone_floors",
  };
  const veil = {
    part: { surface: "durable", kind: "sight_veil" },
    filter: "pov_sight_veil",
  };
  const table = buildPaintPartTable([
    ["red-zone", [red]],
    ["hud", []],
    ["fog", [veil]],
  ]);
  assert.equal(Object.isFrozen(table), true);
  assert.deepEqual(
    table.registrations.map(({ owner, filterId, tag }) => [
      owner,
      filterId,
      { ...tag },
    ]),
    [
      ["red-zone", "red_zone_floors", red.part],
      ["fog", "pov_sight_veil", veil.part],
    ],
  );
  assert.notEqual(table.registrations[0].tag, red.part);
  assert.equal(Object.isFrozen(table.registrations[0].tag), true);
  assert.deepEqual(buildPaintPartTable([]).registrations, []);
  assert.throws(
    () =>
      buildPaintPartTable([
        ["fog", [veil]],
        ["hud", [veil]],
      ]),
    /Duplicate visual paint-part registration .*\(fog and hud\)/u,
  );
  assert.throws(
    () => buildPaintPartTable([["fog", [veil, veil]]]),
    /Duplicate visual paint-part registration/u,
  );
  assert.throws(
    () =>
      buildPaintPartTable([
        ["red-zone", [red]],
        [
          "map-skins",
          [
            {
              part: { kind: "red_zone_floor", surface: "durable" },
              filter: "map_skins",
            },
          ],
        ],
      ]),
    /Duplicate visual paint-part registration .*\(red-zone and map-skins\)/u,
  );
  assert.throws(() => buildPaintPartTable([["fog", "parts"]]), /PAINT_PARTS array/u);
  assert.throws(() => buildPaintPartTable([["fog", undefined]]), /PAINT_PARTS array/u);
  assert.throws(() => buildPaintPartTable([["fog", [null]]]), /not an object/u);
  assert.throws(
    () => buildPaintPartTable([["fog", [{ ...veil, owner: "fog" }]]]),
    /invalid shape/u,
  );
  assert.throws(
    () => buildPaintPartTable([["fog", [{ part: veil.part, filter: "fog_of_war" }]]]),
    RangeError,
  );
  assert.throws(
    () =>
      buildPaintPartTable([
        ["fog", [{ part: { kind: "veil" }, filter: "pov_sight_veil" }]],
      ]),
    TypeError,
  );
});

test("multipart effects retain coherent filter ownership", () => {
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "applied",
      part: "effect",
    }),
    "status_application",
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "trap_broken",
      part: "effect",
    }),
    "freezing_trap_break",
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "net_health",
      outcome: "unchanged",
      part: "battle_text",
    }),
    "scrolling_battle_text",
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "activation",
      component: "basic",
      part: "semantic",
    }),
    "basic_ability_effects",
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "activation",
      component: "ultimate",
      part: "semantic",
    }),
    "ultimate_ability_effects",
  );
  for (const outcome of ["damage", "healing", "unchanged"]) {
    assert.equal(
      classifyVisualPaintPart({
        surface: "transient",
        kind: "net_health",
        outcome,
        part: "effect",
      }),
      "scrolling_battle_text",
    );
  }
  for (const removedId of ["damage_effects", "healing_effects"]) {
    assert.throws(
      () => isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, removedId),
      /Unknown visual filter/u,
    );
  }
  assert.equal(
    VISUAL_FILTER_IDS.includes(/** @type {any} */ ("damage_effects")) ||
      VISUAL_FILTER_IDS.includes(/** @type {any} */ ("healing_effects")),
    false,
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "durable",
      kind: "cooldown_badge",
    }),
    "cooldown_effects",
  );
  assert.equal(
    classifyVisualPaintPart({ surface: "durable", kind: "red_zone_floor" }),
    "red_zone_floors",
  );
  for (const kind of ["selection_reticle", "selected_pair_legality"]) {
    assert.equal(
      classifyVisualPaintPart({ surface: "durable", kind }),
      "target_selection_visuals",
    );
  }
  assert.equal(
    VISUAL_FILTER_IDS.includes(/** @type {any} */ ("rejected_action_feedback")),
    false,
  );
});

test("unknown and malformed future paint parts fail closed", () => {
  for (const malformed of [
    null,
    [],
    {},
    { surface: "transient" },
    { kind: "death_effect" },
    { surface: "transient", kind: 1 },
    { surface: "", kind: "death_effect" },
  ]) {
    assert.throws(() => classifyVisualPaintPart(malformed), TypeError);
  }
  assert.throws(
    () =>
      classifyVisualPaintPart({
        surface: "transient",
        kind: "future_effect",
      }),
    /Unregistered visual paint part/u,
  );
  assert.throws(
    () =>
      classifyVisualPaintPart({
        surface: "transient",
        kind: "death_effect",
        unexpected: "field",
      }),
    /Unregistered visual paint part/u,
  );
});
