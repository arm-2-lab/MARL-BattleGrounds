/**
 * @file Own the Visual Filters: every display switch of the chosen V5 look, its
 * plain label and help line, its default, the five presets and the paint-part
 * table that tells each drawing which switch controls it.
 *
 * Which client lists which switch: each row names its products. A switch whose
 * drawing belongs to the match itself is listed in live play (the BattleClient
 * and the DevClient combat screen) and in the Replay Viewer, with the default
 * of the owner's V5 pick, and the shared look modules under `src/looks/` draw
 * it the same way in every client. That includes the V5 add-ons Class Seals,
 * Aura Glints, Barrier Held (on by default), Saved Beat, Last Breath, Battle
 * Text Detail, Kill Credit Glint, Champion Streaks and Momentum. Replay tools
 * (Battle Cam, timeline marks, and the Replay Tools group's Next Step Ghost
 * and All Staged Orders, which show the recorded next tick) are listed only
 * in the Replay Viewer, and the live play marks (Command Marks, Reach Marks,
 * Step Ghost) only in live play. Lore Emblems is
 * listed in both clients: its V5 pick (Heraldic Sigils) draws every token's
 * class emblem, so the approved parity plan keeps it (P25). Off draws a plain
 * outline emblem (P26, never today's old glyph) everywhere an emblem shows:
 * main.js passes the choice to icons.js (setIconStyleDefaults), so bodies,
 * frames, rail and cards all change. Live
 * play shows the owner's V5 label and group (LIVE_SWITCH_NAMES) where the
 * Replay Viewer still uses its older name. Two V5 switches are live only:
 * Palette: High Contrast (the Full Moon Contrast pick; shell.js applies it as
 * the page palette from palette.js, while the Replay Viewer keeps its Palette
 * select) and Brand Signature (the Rune-Cut Crest pick; shell.js draws it as
 * page chrome beside the floor, never in the battlefield SVG or a PNG). The
 * Title Card choice is kept per browser tab by filter-controls.js.
 *
 * Who calls it: the renderer (`scene.js`), the plan builder
 * (`choreography-plan.js`), the PNG export (`replay-export.js`), the page
 * (`main.js`) and the Visuals tab and preset buttons. Each look module under
 * `src/looks/` lists its own drawable parts in a `PAINT_PARTS` export; this file
 * reads those lists and builds one table from them on first use.
 *
 * Boundary: a switch only changes what is drawn. It never grants a fact the
 * audience may not see, never changes a simulation and never enters a result
 * fingerprint. A skin switch turned off falls back to a plain, truthful drawing
 * (for example a flat stone with its true footprint); it never hides a fact of
 * the game. Champions are always drawn as heraldic tokens: that look is part of
 * the game, not a switch (owner, 3 Oct 2026). Every help line says
 * what Off shows. In Agent POV, no switch can show a champion the team cannot
 * see. Presets set switches only: they never touch View, Ranges, Layout or
 * Motion (the animation setting), because those are not switches.
 */
import { PAINT_PARTS as ATMOSPHERE_PAINT_PARTS } from "./looks/atmosphere.js";
import { PAINT_PARTS as BASICS_PAINT_PARTS } from "./looks/basics.js";
import { PAINT_PARTS as CAMERA_PAINT_PARTS } from "./looks/camera.js";
import { PAINT_PARTS as CHAMPION_FRAME_PAINT_PARTS } from "./looks/champion-frame.js";
import { PAINT_PARTS as CHAMPIONS_PAINT_PARTS } from "./looks/champions.js";
import { PAINT_PARTS as FOCUS_TALLY_PAINT_PARTS } from "./looks/focus-tally.js";
import { PAINT_PARTS as FOG_PAINT_PARTS } from "./looks/fog.js";
import { PAINT_PARTS as FRAME_EXPLANATION_PAINT_PARTS } from "./looks/frame-explanation.js";
import { PAINT_PARTS as HEALTH_TEXT_PAINT_PARTS } from "./looks/health-text.js";
import { PAINT_PARTS as HERALDS_PAINT_PARTS } from "./looks/heralds.js";
import { PAINT_PARTS as HOTBAR_PAINT_PARTS } from "./looks/hotbar.js";
import { PAINT_PARTS as HUD_PAINT_PARTS } from "./looks/hud.js";
import { PAINT_PARTS as MAP_SKINS_PAINT_PARTS } from "./looks/map-skins.js";
import { PAINT_PARTS as MOVEMENT_PAINT_PARTS } from "./looks/movement.js";
import { PAINT_PARTS as OBSTACLES_PAINT_PARTS } from "./looks/obstacles.js";
import { PAINT_PARTS as RED_ZONE_PAINT_PARTS } from "./looks/red-zone.js";
import { PAINT_PARTS as REQUIEMS_PAINT_PARTS } from "./looks/requiems.js";
import { PAINT_PARTS as SPAWNS_PAINT_PARTS } from "./looks/spawns.js";
import { PAINT_PARTS as STATUSES_PAINT_PARTS } from "./looks/statuses.js";
import { PAINT_PARTS as TEAM_IDENTITY_PAINT_PARTS } from "./looks/team-identity.js";
import { PAINT_PARTS as ULTIMATES_PAINT_PARTS } from "./looks/ultimates.js";

/**
 * @typedef {
 *   | "aura_fields"
 *   | "aura_modifier_badges"
 *   | "duration_status_badges"
 *   | "spawn_shield"
 *   | "target_selection_visuals"
 *   | "basic_ability_effects"
 *   | "ultimate_ability_effects"
 *   | "regeneration_effects"
 *   | "cooldown_effects"
 *   | "status_application"
 *   | "natural_status_expiry"
 *   | "freezing_trap_break"
 *   | "status_clear_on_death"
 *   | "death_effects"
 *   | "respawn_wave"
 *   | "resurrection_effects"
 *   | "spawn_shield_expiry"
 *   | "scrolling_battle_text"
 *   | "death_announcer"
 *   | "red_zone_floors"
 *   | "map_skins"
 *   | "moonlight"
 *   | "floor_grid"
 *   | "spawn_waystones"
 *   | "survey_frame"
 *   | "arena_rim"
 *   | "obstacle_skins"
 *   | "obstacle_relief"
 *   | "obstacle_shadows"
 *   | "red_zone_home_fall"
 *   | "soft_glow"
 *   | "moon_motes"
 *   | "time_of_night"
 *   | "death_dimming"
 *   | "pov_sight_veil"
 *   | "pov_recipient_rim"
 *   | "sight_change_marks"
 *   | "life_vessels"
 *   | "class_seals"
 *   | "recent_health_change"
 *   | "low_life_warning"
 *   | "hit_reactions"
 *   | "fallen_step_aside"
 *   | "class_letters"
 *   | "life_segments"
 *   | "stride_glide"
 *   | "scholar_long_tick"
 *   | "step_glide"
 *   | "last_move_ghosts"
 *   | "phantom_afterimage"
 *   | "focus_trail"
 *   | "home_end_marks"
 *   | "lore_emblems"
 *   | "status_marks_icons"
 *   | "status_marks"
 *   | "aura_glints"
 *   | "barrier_held"
 *   | "kindled_blast"
 *   | "close_range_strokes"
 *   | "saved_beat"
 *   | "last_breath"
 *   | "battle_text_detail"
 *   | "vigil_pips"
 *   | "kill_credit_glint"
 *   | "match_moments"
 *   | "moment_full_cards"
 *   | "champion_streaks"
 *   | "camera_framing"
 *   | "camera_change_tip"
 *   | "camera_tip"
 *   | "auto_frame_fights"
 *   | "kill_cam"
 *   | "kill_slow_motion"
 *   | "scorebug"
 *   | "team_rosters"
 *   | "fighters_standing"
 *   | "hud_momentum"
 *   | "symbol_explanations"
 *   | "reach_rings"
 *   | "next_action_thread"
 *   | "selection_moonlight"
 *   | "selected_agent_frame"
 *   | "event_timeline_marks"
 *   | "beat_label"
 *   | "freeze_on_pause"
 *   | "hide_marks_ahead"
 *   | "class_guide"
 *   | "ledger_summary"
 *   | "panel_tokens"
 *   | "brand_lockup"
 *   | "brand_signature"
 *   | "title_card"
 *   | "command_marks"
 *   | "reach_marks"
 *   | "step_ghost"
 *   | "focus_fire"
 *   | "high_contrast"
 *   | "next_step_ghost"
 *   | "all_staged_orders"
 * } VisualFilterId
 * @typedef {"combat_debugger" | "replay_viewer"} VisualFilterProduct
 * @typedef {"arena" | "interface"} VisualFilterSurface
 * @typedef {Readonly<{
 *   id: VisualFilterId,
 *   label: string,
 *   help: string,
 *   group: string,
 *   defaultEnabled: boolean,
 *   surface: VisualFilterSurface,
 *   products: ReadonlyArray<VisualFilterProduct>,
 * }>} VisualFilterRegistration
 * @typedef {Readonly<Record<VisualFilterId, boolean>>} VisualFilterState
 * @typedef {Readonly<Record<string, string>>} VisualPaintPart
 * @typedef {Readonly<{
 *   tag: VisualPaintPart,
 *   filterId: VisualFilterId,
 *   owner: string,
 * }>} VisualPaintPartRegistration
 * @typedef {"default" | "researcher" | "manuscript" | "minimal"} VisualFilterPresetId
 * @typedef {Readonly<{
 *   id: VisualFilterPresetId,
 *   label: string,
 *   purpose: string,
 *   overrides: Readonly<Partial<Record<VisualFilterId, boolean>>>,
 * }>} VisualFilterPreset
 */

/**
 * The groups the Visuals tab shows, in display order. Every switch belongs to
 * exactly one of them.
 */
export const VISUAL_FILTER_GROUPS = Object.freeze([
  "Arena",
  "Bodies",
  "Combat",
  "Life And Death",
  "Moments",
  "Motion",
  "Vision",
  "Selection",
  "Camera",
  "HUD",
  "Panels",
  "Replay Tools",
  "Diagnostics",
]);

const BOTH = Object.freeze(
  /** @type {VisualFilterProduct[]} */ (["combat_debugger", "replay_viewer"]),
);
const REPLAY = Object.freeze(/** @type {VisualFilterProduct[]} */ (["replay_viewer"]));
const LIVE = Object.freeze(/** @type {VisualFilterProduct[]} */ (["combat_debugger"]));

// One row per switch: id, label, group, default, surface, products, help.
// The first twenty rows are the switches the clients had before V5 (their ids
// and order never change, so old settings and PNGs still read). The rest are
// the switches of the V5-picked looks. Help lines follow the owner's naming
// rules: Team A and Team B only, plain words, and each says what Off shows.
/** @type {ReadonlyArray<readonly [string, string, string, boolean, VisualFilterSurface, ReadonlyArray<VisualFilterProduct>, string]>} */
const REGISTRY_ROWS = [
  [
    "aura_fields",
    "Aura Fields",
    "Bodies",
    false,
    "arena",
    BOTH,
    "A faint ring shows how far each Warrior's and Mage's aura reaches: 2 units, through walls. Off: no ring; the aura still works.",
  ],
  [
    "aura_modifier_badges",
    "Aura Multipliers",
    "Diagnostics",
    false,
    "arena",
    BOTH,
    "Small cells beside each champion show the damage and guard multipliers that allied auras give it right now. Off: no cells.",
  ],
  [
    "duration_status_badges",
    "Status Plates",
    "Diagnostics",
    false,
    "arena",
    BOTH,
    "Small plates beside each champion list its statuses and the ticks each one has left, including the In Combat countdown. Off: no plates.",
  ],
  [
    "spawn_shield",
    "Spawn Shield",
    "Bodies",
    true,
    "arena",
    BOTH,
    "A champion that just came back looks shielded: a broken band, a pale veil and silver diamonds that count its shield ticks. Off: a plain token; the shield still protects it.",
  ],
  [
    "target_selection_visuals",
    "Target Selection Visuals",
    "Selection",
    true,
    "arena",
    BOTH,
    "Silver brackets frame the selected champion, and a faint ring and, in live play, a hover card show the one you point at. In Agent POV the brackets hide while that champion is unseen. Off: no brackets, hover ring or hover card.",
  ],
  [
    "basic_ability_effects",
    "Basic Ability Effects",
    "Combat",
    true,
    "arena",
    BOTH,
    "Draws every Basic: a stroke, a bolt or a cleave between the two champions and a landing on the one it hit. In a still, strokes stay at 50% and landings at 80%. Off: no Basic strokes or landings; life, numbers and statuses still show.",
  ],
  [
    "ultimate_ability_effects",
    "Ultimate Ability Effects",
    "Combat",
    true,
    "arena",
    BOTH,
    "Draws every Ultimate (the Charge's path, Freezing Trap, Burst, Crippling Poison and Holy Word: Salvation) and a small gesture by the caster. Off: no Ultimate drawings; life, numbers and statuses still show.",
  ],
  [
    "regeneration_effects",
    "Regeneration Effects",
    "Combat",
    true,
    "arena",
    BOTH,
    "Rest (Regeneration), the life won back while out of combat, pours a pale blue light into the champion's life. Off: the life level still rises, with no pour.",
  ],
  [
    "cooldown_effects",
    "Cooldown Effects",
    "Bodies",
    true,
    "arena",
    BOTH,
    "A ring on each champion shows how many ticks are left before its Ultimate is ready, and lights up when it is ready. Off: no ring on the body, and every Class Seal stays faint.",
  ],
  [
    "status_application",
    "Status Application",
    "Bodies",
    true,
    "arena",
    BOTH,
    "A new status fades in on its champion when it lands. Off: the status shows at once.",
  ],
  [
    "natural_status_expiry",
    "Natural Status Expiry",
    "Bodies",
    true,
    "arena",
    BOTH,
    "A status that runs out fades away piece by piece. Off: it disappears at once.",
  ],
  [
    "freezing_trap_break",
    "Freezing Trap Break",
    "Bodies",
    true,
    "arena",
    BOTH,
    "The Freezing Trap's ice shatters when damage breaks the stun: a crack, then six ice shards around the body. In Agent POV only on champions your team saw at the start of the tick. Paused: the crack and shards stay. Off: the ice disappears at once.",
  ],
  [
    "status_clear_on_death",
    "Status Clear On Death",
    "Bodies",
    false,
    "arena",
    BOTH,
    "When a champion falls, its status marks stay until its fall and then turn to ash. In Agent POV only for a fall your team watched. Paused: not shown. Off: the marks disappear at the fall.",
  ],
  [
    "death_effects",
    "Death Effects",
    "Life And Death",
    true,
    "arena",
    BOTH,
    "Each class falls in its own way where it came to rest (star, shield, leaves, smoke or candle), and the field holds still for it. Paused: a thin ring marks this tick's falls and a small gray relic shows how each fell. Off: no fall beat and no hold; the fallen simply turn to ash.",
  ],
  [
    "respawn_wave",
    "Respawn Wave",
    "Life And Death",
    true,
    "arena",
    BOTH,
    "When champions come back, their Arena Frames mark the return, their waystones light (with Resurrection Effects on) and the field holds still for the return. A wave that brings nobody back shows nothing. Off: no mark, no light and no hold.",
  ],
  [
    "resurrection_effects",
    "Resurrection Effects",
    "Life And Death",
    true,
    "arena",
    BOTH,
    "As a champion comes home, moonlight pools on its waystone, a ring of moonlight comes down onto it and its old ash lifts away. Paused: not shown. Off: champions simply appear on their pads.",
  ],
  [
    "spawn_shield_expiry",
    "Spawn Shield Expiry",
    "Life And Death",
    true,
    "arena",
    BOTH,
    "When a spawn shield ends, the ward's dashed band closes into the solid team ring and a dashed silver ring lifts off the champion. In Agent POV only on champions your team sees. Paused: not shown. Off: the ward simply ends as the tick settles.",
  ],
  [
    "scrolling_battle_text",
    "Scrolling Battle Text",
    "Combat",
    true,
    "arena",
    BOTH,
    "One number above each champion whose life changed this tick: ivory for a loss, gold for a gain and pale blue for Rest. Off: no numbers; the life level still shows the change.",
  ],
  [
    "death_announcer",
    "Fallen Feed",
    "Life And Death",
    true,
    "arena",
    BOTH,
    "The Fallen panel at the top of the side rail lists the falls of the last 8 ticks, newest first, at most 4: everyone who shared the fall, a dot for a Basic or a diamond for an Ultimate, who fell and the points. In Agent POV only the falls your team saw, naming only the attackers it saw. Off: no panel.",
  ],
  [
    "red_zone_floors",
    "Red Zone Floors",
    "Arena",
    true,
    "arena",
    BOTH,
    "Each team's home zone looks the same for both teams: an ash floor, a border of small gravestones and a skull beside x2. Fall in your own home zone and the enemy scores 2 points. Off: a plain thin line marks the edge of each zone.",
  ],
  [
    "map_skins",
    "Map Skins",
    "Arena",
    true,
    "arena",
    BOTH,
    "Each named map gets its own floor and stone look. It is only a look: no floor has any game effect. Off: the plain moonstone floor on every map.",
  ],
  [
    "moonlight",
    "Moonlight",
    "Arena",
    false,
    "arena",
    BOTH,
    "A soft pool of moonlight over the middle of the floor, darker toward the walls. It means nothing in the game. Off: a flat floor.",
  ],
  [
    "floor_grid",
    "Floor Grid",
    "Arena",
    false,
    "arena",
    BOTH,
    "Faint stone joints on whole world units (one stone course is one unit) and a dashed seam down the mirror line. Off: a plain floor with no lines.",
  ],
  [
    "spawn_waystones",
    "Spawn Waystones",
    "Arena",
    true,
    "arena",
    BOTH,
    "One rune circle per champion at its team's spawn edge. It lights when its champion comes home. In Agent POV a pad whose champion is unseen stays quiet. Off: a plain thin ring marks each spawn pad.",
  ],
  [
    "survey_frame",
    "Survey Frame",
    "Arena",
    false,
    "arena",
    BOTH,
    "World coordinates around the floor, with y counting up as in the simulator, so a paper figure carries its own axes. While it is on, the Arena Rim is hidden. Off: a plain margin.",
  ],
  [
    "arena_rim",
    "Arena Rim",
    "Arena",
    false,
    "arena",
    BOTH,
    "A stone wall around the floor, because the map edge stops movement like a wall. Off: the floor edge is a plain line.",
  ],
  [
    "obstacle_skins",
    "Obstacle Skins",
    "Arena",
    true,
    "arena",
    BOTH,
    "Standing stones for pillars, walls and thin walls, drawn on their true footprint. Off: flat stone shapes with the same true footprint.",
  ],
  [
    "obstacle_relief",
    "Obstacle Relief",
    "Arena",
    true,
    "arena",
    BOTH,
    "Stone caps, wall tops and lit ends on thin walls. Off: flat stones with the same footprint.",
  ],
  [
    "obstacle_shadows",
    "Obstacle Shadows",
    "Arena",
    true,
    "arena",
    BOTH,
    "A short shadow under every obstacle. It lies just outside the true edge, so turn it off for exact-footprint figures. Off: no shadows.",
  ],
  [
    "red_zone_home_fall",
    "Red Zone Home Fall",
    "Moments",
    true,
    "arena",
    BOTH,
    "A fall inside the fallen champion's own home zone (worth 2 points) lights that zone's gravestones and border once, at the fall beat. In Agent POV only when your team saw that fall. Needs Red Zone Floors. Off: no light; the score still counts 2.",
  ],
  [
    "soft_glow",
    "Soft Glow",
    "Combat",
    true,
    "arena",
    BOTH,
    "A faint wide light under bright lines, such as bolts and the ring of moonlight, while playing. Never in a still. Off: the lines only; nothing changes meaning.",
  ],
  [
    "moon_motes",
    "Moon Motes",
    "Arena",
    true,
    "arena",
    BOTH,
    "Twelve faint silver motes drift slowly during play; in a still they rest in fixed spots. They mean nothing in the game. Off: a clear floor.",
  ],
  [
    "time_of_night",
    "Time Of Night",
    "Arena",
    false,
    "arena",
    BOTH,
    "Over the last fifth of the game the floor grays in four small steps, to show time running out: a game with no winner ends in a draw at the last tick. Off: the night color stays the same all game.",
  ],
  [
    "death_dimming",
    "Death Dimming",
    "Moments",
    true,
    "arena",
    BOTH,
    "When a champion falls during play, the floor, home zones and stones dim for the fall beat while champions and marks stay bright. Never in a still. In Agent POV only for falls your team saw. Off: the light stays steady.",
  ],
  [
    "pov_sight_veil",
    "Sight Reach Veil",
    "Vision",
    true,
    "arena",
    BOTH,
    "In Agent POV, darkens every spot your team cannot see: beyond its living champions' sight reach and in the shadow behind each stone. Enemies just back on their pads stay hidden even in the light. Off: the floor at full light; unseen champions stay hidden either way.",
  ],
  [
    "pov_recipient_rim",
    "Agent POV Rim",
    "Vision",
    true,
    "arena",
    // Replay Viewer only: live team sight has no single recipient to ring.
    REPLAY,
    "In Agent POV, a thin silver rim marks the champion whose eyes you see through. Off: no rim.",
  ],
  [
    "sight_change_marks",
    "Sight Change Marks",
    "Vision",
    true,
    "arena",
    BOTH,
    "In Agent POV, a dotted Last Seen ring marks where an enemy was when it left sight this tick, and a glint marks an enemy that came into sight. Off: no rings or glints.",
  ],
  [
    "life_vessels",
    "Life Vessels",
    "Bodies",
    true,
    "arena",
    BOTH,
    "Life fills the token from the bottom like a vessel; the height is the champion's life. Off: life shows as an ivory ring around the token.",
  ],
  [
    "class_seals",
    "Class Seals",
    "Bodies",
    true,
    "arena",
    BOTH,
    "On larger tokens, each champion stands on its class seal, which lights up when its Ultimate is ready (with Cooldown Effects on) and fades back under a status mark. The Priest then shows only its cross, because the seal is its sun. Off: the plain class halo.",
  ],
  [
    "recent_health_change",
    "Recent Health Change",
    "Combat",
    true,
    "arena",
    BOTH,
    "Life lost this tick stays as a faint ivory layer under a dashed line at the old level; heals pour in gold. Off: the token simply shows the new level.",
  ],
  [
    "low_life_warning",
    "Low Life Warning",
    "Bodies",
    true,
    "arena",
    BOTH,
    "At 30% life or less, the champion's life turns a hot orange. Off: life keeps its usual ivory color.",
  ],
  [
    "hit_reactions",
    "Hit Reactions",
    "Combat",
    true,
    "arena",
    BOTH,
    "A struck token flashes lighter for 80 ms as the blow lands; the target of a damaging Ultimate or a killing blow also shivers (the Warrior never flinches). Play only; never in a still. Off: no flash or shiver.",
  ],
  [
    "fallen_step_aside",
    "Fallen Step Aside",
    "Bodies",
    true,
    "arena",
    BOTH,
    "When a standing champion is on top of a fallen one, the fallen token becomes a small ash coin at its side, so its team ring is never mistaken for the standing champion's. Off: the full-size ash token stays under the standing champion.",
  ],
  [
    "class_letters",
    "Class Letters",
    "Bodies",
    false,
    "arena",
    BOTH,
    "A class letter (M, W, H, R, P) under a smaller emblem, for printed figures. Off: the emblem alone.",
  ],
  [
    "life_segments",
    "Life Segments",
    "Bodies",
    false,
    "arena",
    BOTH,
    "Thin notches across a champion's life ring every 20 points, so a full Warrior reads as tough in a still. Shows only with Life Vessels off, because a vessel has no ring. Off: smooth life with no notches.",
  ],
  [
    "stride_glide",
    "Body Glide",
    "Motion",
    true,
    "arena",
    BOTH,
    "In play, champions glide to where they ended the tick, then every blow plays between those spots. Paused: champions and blows at the end spots. Off: champions stand at their end spots for the whole tick.",
  ],
  [
    "scholar_long_tick",
    "Longer Tick",
    "Motion",
    true,
    "arena",
    BOTH,
    "Each tick plays 0.2 seconds longer (1.7 seconds, or 3.2 seconds with a fall) so every attack keeps its own speed after the glide. Paused frames do not change. Off: attacks squeeze into the shorter tick.",
  ],
  [
    "step_glide",
    "Step Glide",
    "Motion",
    true,
    "arena",
    REPLAY,
    "Stepping exactly one tick: +1 (Right arrow, N) plays that tick as Play does, then pauses on it; -1 (Left arrow) rewinds it: a champion that fell stands up first, then everyone glides back and the earlier tick's blows appear. Bigger jumps snap. Off: every step snaps.",
  ],
  [
    "last_move_ghosts",
    "Start Spots",
    "Motion",
    true,
    "arena",
    BOTH,
    "A faint ring and hairline mark where each champion began this tick: the spot the game judged its blows from. In Agent POV only for champions your team saw. Off: no rings or hairlines.",
  ],
  [
    "phantom_afterimage",
    "Phantom Afterimage",
    "Motion",
    true,
    "arena",
    BOTH,
    "When a Rogue darts away, a smoke copy of its token stays where it stood for a moment, then thins into its start ring. Paused: not shown. Off: no smoke copy.",
  ],
  [
    "focus_trail",
    "Focus Trail",
    "Motion",
    true,
    "arena",
    BOTH,
    "Dots trace the selected champion's path from where it began this tick to where it ended. It shows this tick only, not earlier ticks. Off: no dots.",
  ],
  [
    "home_end_marks",
    "Home End Marks",
    "Arena",
    true,
    "arena",
    BOTH,
    "A thin bracket in the team's color outside each end of the floor, as deep as that team's Red Zone, shows whose home it is (one line for Team A, two for Team B). It follows the recorded side, not the screen side. Off: no brackets.",
  ],
  [
    "lore_emblems",
    "Lore Emblems",
    "Bodies",
    true,
    "arena",
    BOTH,
    "Every champion shows its class emblem as a filled shape in its own color: star, shield, antlers, crossed daggers or sun cross. Off: a simple outline emblem in the same place, so the class still shows.",
  ],
  [
    "status_marks_icons",
    "Status Icons From Marks",
    "HUD",
    true,
    "interface",
    BOTH,
    "Status, aura and life icons in panels and cards are small copies of the marks on the body. Off: simple plain icons with the same meaning.",
  ],
  [
    "status_marks",
    "Status Marks",
    "Bodies",
    true,
    "arena",
    BOTH,
    "Stuns, slows, poison and the other statuses show as marks inside each champion's token, at most three per body. Off: the token stays plain; the agent details still list every status.",
  ],
  [
    "aura_glints",
    "Aura Glints",
    "Combat",
    false,
    "arena",
    BOTH,
    "During play, a stud on a Warrior's or Mage's aura ring glints on a tick when that aura helped an ally inside it. Needs Aura Fields. In Agent POV only when your team saw the aura's champion, the ally and the blow. Paused: not shown. Off: no glints.",
  ],
  [
    "barrier_held",
    "Barrier Held",
    "Moments",
    true,
    "arena",
    BOTH,
    "A bronze collar closes around an ally that lived only because Guardian's Barrier, a Warrior's aura, cut the damage it took. It is rare. Paused: a faint collar. In Agent POV only on champions your team sees, worked out from the life change your team saw, so it can miss a save but never shows a false one. Off: no collar.",
  ],
  [
    "kindled_blast",
    "Arcane Blast During Burst",
    "Combat",
    true,
    "arena",
    BOTH,
    "While a Mage's Burst (Damage Boost) is active (x1.5 damage), its Arcane Blast flies as a Comet Bolt: a brighter bolt whose comet head rides out to the target, with a larger star and a ring that flashes open around it. From right beside the target its stroke is thicker and its star larger. Shown in stills too. Off: every Arcane Blast looks the same.",
  ],
  [
    "close_range_strokes",
    "Close-Range Strokes",
    "Combat",
    true,
    "arena",
    BOTH,
    "When a Mage, Hunter or Priest strikes from right beside its target (token edges under 0.55 units apart), its Basic is drawn as a short weapon stroke in its class color, so the hit and its side stay clear. Shown in stills too. Off: only the landing shows.",
  ],
  [
    "saved_beat",
    "Saved Beat",
    "Moments",
    true,
    "arena",
    BOTH,
    "When a heal keeps a champion alive through blows that would have killed it, 'Saved' shows above its number and its rim flashes gold twice. Needs Scrolling Battle Text. Oracle View only: Agent POV does not see the blows and heals behind a life change. Paused: the word stays, the flash does not. Off: no 'Saved' word and no flash.",
  ],
  [
    "last_breath",
    "Last Breath",
    "Combat",
    false,
    "arena",
    BOTH,
    "When a champion lost at least as much life this tick as it has left, the empty part of its life smolders: one more tick like this ends it. Needs Scrolling Battle Text. In Agent POV only on champions your team saw at both ends of the tick. Off: no smolder.",
  ],
  [
    "battle_text_detail",
    "Battle Text Detail",
    "Combat",
    false,
    "arena",
    BOTH,
    "Bigger losses get bigger numbers that drift into place, with a small underline in each attacker's class color and a 'Beyond Death' tag for overkill when there is room. Needs Scrolling Battle Text. In Agent POV only the size and the drift change. Off: plain numbers in two sizes.",
  ],
  [
    "vigil_pips",
    "Vigil Pips",
    "Life And Death",
    true,
    "arena",
    BOTH,
    "Silver dots on each fallen champion count the ticks until its team's wave brings it home. Paused: shown. Off: no dots.",
  ],
  [
    "kill_credit_glint",
    "Kill Credit Glint",
    "Life And Death",
    false,
    "arena",
    BOTH,
    "When a champion falls, every living champion that shared the kill gets the same brief ring in its class color. Needs Death Effects. In Agent POV only on champions your team sees and whose blow it saw land. Paused: not shown. Off: no rings.",
  ],
  [
    "match_moments",
    "Match Moments",
    "Moments",
    true,
    "arena",
    BOTH,
    "Herald cards above the arena name the big moments: First Blood, takedowns, runs, trades, Ultimate kills and combos, rescues, Match Point, a level score and the last ticks. In Agent POV a card names only what your team saw. Off: no cards; the score still shows.",
  ],
  [
    "moment_full_cards",
    "Full Lineup And Verdict",
    "Moments",
    true,
    "arena",
    BOTH,
    "The opening lineup and the closing verdict show as full cards. Off: they show as slim bands.",
  ],
  [
    "champion_streaks",
    "Champion Streaks",
    "Moments",
    true,
    "arena",
    BOTH,
    "The herald cards also name a champion each time it reaches 5, 10, 15 or 20 takedowns without falling, such as 'Team A Mage' with '10 Takedowns Without Falling'. In Agent POV only the takedowns your team saw count. Needs Match Moments. Off: no streak cards.",
  ],
  [
    "camera_framing",
    "Camera Framing",
    "Camera",
    true,
    "interface",
    BOTH,
    "The camera switch: Whole Map, Everyone In View (V) and, in the Replay Viewer, Battle Cam (B). Off: the camera stays on Whole Map and V and B do nothing.",
  ],
  [
    "camera_change_tip",
    "Camera Change Tip",
    "Camera",
    true,
    "interface",
    BOTH,
    "After each camera change, a short tip beside the switch says which camera is on and how to leave it, for example 'Everyone In View on · V: back to Whole Map'. It fades after about 4 seconds or at the next click or key. Off: no tip.",
  ],
  [
    "camera_tip",
    "New Camera Tip",
    "Camera",
    true,
    "interface",
    REPLAY,
    "The one-time tip under the camera switch the first time the Replay Viewer opens in this browser: 'New: Battle Cam follows the fights for you. Click here or press B.' Off: no tip.",
  ],
  [
    "auto_frame_fights",
    "Auto-Frame Fights",
    "Camera",
    true,
    "interface",
    REPLAY,
    "Battle Cam only: frames each tick's fight. In Agent POV only fights your team saw. Off: Battle Cam keeps the whole map.",
  ],
  [
    "kill_cam",
    "Kill Cam",
    "Camera",
    true,
    "interface",
    REPLAY,
    "Battle Cam only: leans in on each fall during play, then eases back before the tick ends. In Agent POV only falls your team saw. Off: no lean.",
  ],
  [
    "kill_slow_motion",
    "Kill Slow Motion",
    "Camera",
    true,
    "interface",
    REPLAY,
    "Battle Cam only: names the held fall beat in the words beside the camera switch. Off: no words.",
  ],
  [
    "scorebug",
    "Scorebug",
    "HUD",
    true,
    "interface",
    BOTH,
    "The score for each team, the tick and a bar for each team's race to the winning score. Off: a plain text score line.",
  ],
  [
    "team_rosters",
    "Team Rosters",
    "HUD",
    true,
    "interface",
    BOTH,
    "Each team's champions with their life and state. In the BattleClient's Agent POV an unseen enemy reads Unseen, with no life. Off: the rosters close; the score stays.",
  ],
  [
    "fighters_standing",
    "Fighters Standing",
    "HUD",
    true,
    "interface",
    // Replay Viewer only: the live scorebug shows who is down on its portraits.
    REPLAY,
    "How many of each team's champions are alive, as 'Standing 4 of 5'. Off: no count.",
  ],
  [
    "hud_momentum",
    "Momentum",
    "HUD",
    true,
    "interface",
    BOTH,
    "On each team's score bar, the points it scored in the last 10 ticks glow brighter over a thin silver line. Scores are public, so Agent POV shows the same. Needs Scorebug. Off: every scored point looks the same.",
  ],
  [
    "symbol_explanations",
    "Symbol Explanations",
    "HUD",
    true,
    "interface",
    BOTH,
    "Hover, tap or focus a symbol (a status icon, a panel row, the selected champion's frame, or a floor mark when paused) to read what it means, with its numbers. Off: no tips. F (Frame Explanation) still works.",
  ],
  [
    "reach_rings",
    "Reach Rings",
    "Selection",
    true,
    "arena",
    BOTH,
    "The selected champion's sight, Basic and Ultimate reach, drawn as rings from its center under the stones. Off: no rings for the selected champion; Ranges (G) is a separate control.",
  ],
  [
    "next_action_thread",
    "Next Action Thread",
    "Selection",
    true,
    "arena",
    BOTH,
    "A dashed silver line to whoever the selected champion acts on next, with a dot for a Basic or a diamond for an Ultimate under that body. In Agent POV a teammate's next action is not shown: champions cannot see their teammates' plans. Off: no line.",
  ],
  [
    "selection_moonlight",
    "Selection Moonlight",
    "Selection",
    true,
    "arena",
    BOTH,
    "A soft pool of moonlight on the floor around the selected champion. Off: no pool.",
  ],
  [
    "selected_agent_frame",
    "Selected Agent Frame",
    "Panels",
    true,
    "interface",
    REPLAY,
    "After a click, a frame under the arena shows the selected agent's token, life, Ultimate, statuses, recent ticks and next target. Off: no frame; the Agent tab still shows the facts.",
  ],
  [
    "event_timeline_marks",
    "Event Timeline Marks",
    "HUD",
    true,
    "interface",
    REPLAY,
    "Falls, Ultimates and returns on the timeline, with a dim veil over the ticks ahead. Off: a plain groove with the playhead; every button and key still works.",
  ],
  [
    "beat_label",
    "Beat Label",
    "HUD",
    true,
    "interface",
    REPLAY,
    "The beat under the tick readout: 'Settled', 'Playing' or, after a pause in the middle of a tick, 'Paused · Strike'. Off: the tick alone.",
  ],
  [
    "freeze_on_pause",
    "Freeze On Pause",
    "HUD",
    true,
    "interface",
    REPLAY,
    "Pausing in the middle of a tick holds the picture exactly where it is. Off: pausing jumps to the settled end of the tick.",
  ],
  [
    "hide_marks_ahead",
    "Hide Marks Ahead",
    "HUD",
    false,
    "interface",
    REPLAY,
    "Hides every timeline mark past the playhead, so the timeline gives nothing away. Off: marks ahead show under a dim veil.",
  ],
  [
    "class_guide",
    "Class Guide",
    "Panels",
    true,
    "interface",
    BOTH,
    "The class pages: overview, strength and weakness, the exact numbers, then the lore and how each ability looks. Off: the plain champion facts only.",
  ],
  [
    "ledger_summary",
    "Match Summary",
    "Panels",
    true,
    "interface",
    REPLAY,
    "One line of result and a short row per team at the head of the metrics. Off: the metrics alone.",
  ],
  [
    "panel_tokens",
    "Panel Tokens",
    "Panels",
    true,
    "interface",
    BOTH,
    "Mini tokens, life bars and status icons in the roster, agent and action panels. Off: plain text rows.",
  ],
  [
    "brand_lockup",
    "Brand Lockup",
    "HUD",
    true,
    "interface",
    BOTH,
    "The emblem, 'MARL-BGs' and the client name in the header. Off: the client name in plain text.",
  ],
  [
    "brand_signature",
    "Brand Signature",
    "HUD",
    false,
    "interface",
    LIVE,
    "A small emblem and 'MARL-BGs' in the dark margin by the floor's bottom-right corner, for screenshots and slides. It never covers the floor, never shows a game fact and is never part of an exported PNG. When the margin is too small it does not show. Off: no mark.",
  ],
  [
    "title_card",
    "Title Card",
    "HUD",
    true,
    "interface",
    BOTH,
    "While no frame has arrived yet, shows the emblem and 'MARL-BattleGrounds' above the waiting message. Never in a still. This browser tab remembers your choice, so Off still holds after a reload. Off: the plain waiting message.",
  ],
  [
    "command_marks",
    "Command Marks",
    "Bodies",
    true,
    "arena",
    LIVE,
    "Live play: small marks on your champion's command ring show the armed ability, and a refused order fades to gray. Off: the plain command ring and order line only.",
  ],
  [
    "reach_marks",
    "Reach Marks",
    "Combat",
    true,
    "arena",
    LIVE,
    "Live play: a small mark on each body your armed ability can reach right now, from the game's own list of allowed targets. Off: no marks; the Allowed Now buttons still list the targets.",
  ],
  [
    "step_ghost",
    "Step Ghost",
    "Motion",
    true,
    "arena",
    LIVE,
    "Live play: a faint circle shows where your champion will stand after the step you chose. Off: no circle; the step still happens.",
  ],
  [
    "focus_fire",
    "Focus Tally",
    "Diagnostics",
    false,
    "arena",
    BOTH,
    "In a still, a champion struck by two or more enemies this tick carries one short bead per attacker just below it, in the attacking team's color. It is a plain count of hits, not the Focus Fire Concentration metric. Off: no beads.",
  ],
  [
    "high_contrast",
    "High Contrast",
    "Arena",
    true,
    "arena",
    BOTH,
    "Brighter rings, clearer stones and crisper text for projectors, bright rooms and low vision. It changes colors only: every mark stays where it is and means the same, and an exported PNG shows the same colors. Off: the soft Moonlit night colors.",
  ],
  [
    "next_step_ghost",
    "Next Step Ghost",
    "Replay Tools",
    false,
    "arena",
    REPLAY,
    "A dotted circle marks where each living champion stands at the end of the next tick, taken from the recorded next tick. In Agent POV only champions your team sees now and still sees at the end of the next tick. The last tick has none. Off: no circles.",
  ],
  [
    "all_staged_orders",
    "All Staged Orders",
    "Replay Tools",
    false,
    "arena",
    REPLAY,
    "A dashed silver line from every champion to whoever it acts on next, with a dot for a Basic or a diamond for an Ultimate under that body, taken from the recorded next tick. In Agent POV only the order of the champion whose eyes you see through: champions cannot see their teammates' plans, and the other team's orders never show. Off: only the selected champion's Next Action Thread.",
  ],
];

/**
 * Every switch, in fixed order, as frozen records: `id`, Title Case `label`,
 * one plain `help` line (it always says what Off shows), `group` (one of
 * VISUAL_FILTER_GROUPS), `defaultEnabled`, `surface` ("arena" when it changes
 * the battlefield picture, "interface" when it changes the page around it) and
 * `products` (the clients that use it: "combat_debugger" for the BattleClient
 * and DevClient combat screen, "replay_viewer" for the Replay Viewer).
 *
 * @type {ReadonlyArray<VisualFilterRegistration>}
 */
export const VISUAL_FILTER_REGISTRY = Object.freeze(
  REGISTRY_ROWS.map(([id, label, group, defaultEnabled, surface, products, help]) =>
    Object.freeze({
      id: /** @type {VisualFilterId} */ (id),
      label,
      help,
      group,
      defaultEnabled,
      surface,
      products,
    }),
  ),
);

/** Every switch ID in the fixed registry order. */
export const VISUAL_FILTER_IDS = Object.freeze(
  VISUAL_FILTER_REGISTRY.map(({ id }) => id),
);

const VISUAL_FILTER_ID_SET = new Set(VISUAL_FILTER_IDS);

/** The state a new viewer sees: each switch at its registry default. */
export const DEFAULT_VISUAL_FILTER_STATE = freezeVisualFilterState(
  Object.fromEntries(
    VISUAL_FILTER_REGISTRY.map(({ id, defaultEnabled }) => [id, defaultEnabled]),
  ),
);

const ENABLED_VISUAL_FILTER_STATE = freezeVisualFilterState(
  Object.fromEntries(VISUAL_FILTER_IDS.map((id) => [id, true])),
);

const DISABLED_VISUAL_FILTER_STATE = freezeVisualFilterState(
  Object.fromEntries(VISUAL_FILTER_IDS.map((id) => [id, false])),
);

/**
 * The V5 names of the switches whose live label or group differs from the
 * registry row, keyed by switch ID.
 *
 * The BattleClient and the DevClient combat screen (the Visual Filters drawer)
 * show the label and group the owner picked in V5. The registry row keeps the
 * name the Replay Viewer shows today, because the Replay Viewer's own list is
 * reviewed in a later pass. Only the shown words differ: the switch ID, its
 * default, its help line and what it draws are the same in every client.
 *
 * @type {Readonly<Record<string, Readonly<{label?: string, group?: string}>>>}
 */
export const LIVE_SWITCH_NAMES = Object.freeze({
  target_selection_visuals: Object.freeze({ group: "HUD" }),
  death_announcer: Object.freeze({ label: "Death Announcer" }),
  pov_sight_veil: Object.freeze({ label: "POV Sight Veil", group: "HUD" }),
  sight_change_marks: Object.freeze({ group: "HUD" }),
  last_move_ghosts: Object.freeze({ label: "Last-Move Ghosts" }),
  symbol_explanations: Object.freeze({ group: "Panels" }),
  reach_rings: Object.freeze({ group: "HUD" }),
  next_action_thread: Object.freeze({ group: "HUD" }),
  selection_moonlight: Object.freeze({ group: "HUD" }),
  focus_fire: Object.freeze({ label: "Focus Fire" }),
});

/**
 * Return the registry records a client shows, in registry order.
 *
 * The Visual Filters list shows only the switches that matter in the current
 * client: for example Battle Cam switches only in the Replay Viewer and Command
 * Marks only in live play. A switch that draws the match itself (for example
 * Class Seals or Momentum) is in both lists. In live play ("combat_debugger") a
 * switch named in LIVE_SWITCH_NAMES comes back as a copy that carries its V5
 * label and group. The Replay Viewer gets the registry records unchanged.
 *
 * @param {unknown} product "combat_debugger" (BattleClient and DevClient combat
 *   screen) or "replay_viewer".
 * @returns {ReadonlyArray<VisualFilterRegistration>} A new frozen array of
 *   frozen records (the shared registry records, or frozen live copies for the
 *   renamed switches); empty never happens for a valid product.
 * @throws {RangeError} When product is not one of the two names.
 */
export function visualFiltersForProduct(product) {
  if (product !== "combat_debugger" && product !== "replay_viewer") {
    throw new RangeError(`Unknown visual filter product ${String(product)}.`);
  }
  const shown = VISUAL_FILTER_REGISTRY.filter(({ products }) =>
    products.includes(product),
  );
  if (product === "replay_viewer") return Object.freeze(shown);
  return Object.freeze(
    shown.map((entry) => {
      const live = LIVE_SWITCH_NAMES[entry.id];
      return live === undefined ? entry : Object.freeze({ ...entry, ...live });
    }),
  );
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

// Preset contents come from the approved store presets, kept to the switches
// built here. The owner removed Broadcast on 4 Oct 2026: its spectacle switches
// are now the Default Configuration.
/** @type {ReadonlyArray<VisualFilterPreset>} */
export const VISUAL_FILTER_PRESETS = Object.freeze([
  preset(
    "default",
    "Default Configuration",
    "Every switch as it ships: what a new viewer sees.",
    {},
  ),
  preset(
    "researcher",
    "Researcher",
    "Study each tick with exact numbers and every diagnostic shown.",
    {
      aura_fields: true,
      duration_status_badges: true,
      aura_modifier_badges: true,
      target_selection_visuals: true,
      survey_frame: true,
      class_letters: true,
      hud_momentum: true,
      spawn_shield_expiry: true,
      status_clear_on_death: true,
      focus_fire: true,
      battle_text_detail: true,
      last_breath: true,
    },
  ),
  preset(
    "manuscript",
    "Manuscript",
    "Clean, flat paper figures that still read in grayscale.",
    {
      map_skins: false,
      moonlight: false,
      obstacle_shadows: false,
      class_seals: false,
      soft_glow: false,
      hit_reactions: false,
      pov_sight_veil: false,
      class_letters: true,
      survey_frame: true,
      floor_grid: true,
      spawn_waystones: true,
      obstacle_relief: true,
      red_zone_floors: true,
      last_move_ghosts: true,
      status_marks: true,
      cooldown_effects: true,
      recent_health_change: true,
      scrolling_battle_text: true,
      basic_ability_effects: true,
      ultimate_ability_effects: true,
      death_effects: true,
      selected_agent_frame: false,
      scorebug: false,
      team_rosters: false,
      fighters_standing: false,
      match_moments: false,
      moment_full_cards: false,
      time_of_night: false,
      moon_motes: false,
      death_dimming: false,
      camera_framing: false,
      kill_cam: false,
      kill_slow_motion: false,
    },
  ),
  preset(
    "minimal",
    "Minimal",
    "Just the fight: bodies, blows, life and falls, with no decoration.",
    {
      duration_status_badges: false,
      aura_modifier_badges: false,
      target_selection_visuals: false,
      focus_fire: false,
      map_skins: false,
      survey_frame: false,
      obstacle_relief: false,
      obstacle_shadows: false,
      class_seals: false,
      class_letters: false,
      soft_glow: false,
      hit_reactions: false,
      status_application: false,
      natural_status_expiry: false,
      freezing_trap_break: false,
      status_clear_on_death: false,
      spawn_shield_expiry: false,
      focus_trail: false,
      aura_fields: false,
      red_zone_home_fall: false,
      team_rosters: false,
      hud_momentum: false,
      floor_grid: true,
      spawn_waystones: true,
      red_zone_floors: true,
      status_marks: true,
      cooldown_effects: true,
      spawn_shield: true,
      stride_glide: true,
      last_move_ghosts: true,
      basic_ability_effects: true,
      ultimate_ability_effects: true,
      scrolling_battle_text: true,
      recent_health_change: true,
      regeneration_effects: true,
      death_effects: true,
      resurrection_effects: true,
      respawn_wave: true,
      scorebug: true,
      death_announcer: true,
      moonlight: false,
      moon_motes: false,
      time_of_night: false,
      death_dimming: false,
      match_moments: false,
      moment_full_cards: false,
      selected_agent_frame: false,
      reach_rings: false,
      next_action_thread: false,
      selection_moonlight: false,
      kill_cam: false,
      kill_slow_motion: false,
      low_life_warning: false,
      vigil_pips: false,
      kindled_blast: false,
    },
  ),
]);

/** Every preset ID, Default Configuration first. */
export const VISUAL_FILTER_PRESET_IDS = Object.freeze(
  VISUAL_FILTER_PRESETS.map(({ id }) => id),
);

const PRESET_BY_ID = new Map(VISUAL_FILTER_PRESETS.map((entry) => [entry.id, entry]));

const PRESET_STATE_BY_ID = new Map(
  VISUAL_FILTER_PRESETS.map((entry) => [
    entry.id,
    entry.id === "default"
      ? DEFAULT_VISUAL_FILTER_STATE
      : freezeVisualFilterState(
          /** @type {Record<string, boolean>} */ ({
            ...DEFAULT_VISUAL_FILTER_STATE,
            ...entry.overrides,
          }),
        ),
  ]),
);

/**
 * Return the full switch state a preset gives: every switch at its default,
 * then the preset's own choices on top. Pure: nothing is stored.
 *
 * A preset sets switches only. View (Oracle View or Agent POV), Ranges, Layout
 * and Motion (the animation setting) are not switches, so no preset can change
 * them.
 *
 * @param {unknown} presetId One of VISUAL_FILTER_PRESET_IDS: "default"
 *   (Default Configuration), "researcher", "manuscript" or
 *   "minimal".
 * @returns {VisualFilterState} A shared frozen state; the same object for the
 *   same preset every time. "default" returns DEFAULT_VISUAL_FILTER_STATE.
 * @throws {RangeError} When presetId is not a known preset.
 */
export function visualFilterPresetState(presetId) {
  const state = PRESET_STATE_BY_ID.get(
    /** @type {VisualFilterPresetId} */ (assertPresetId(presetId)),
  );
  return /** @type {VisualFilterState} */ (state);
}

/**
 * List the switches changed by hand: those whose value in state differs from
 * what presetId gives. The preset row shows the count, for example
 * "Researcher · 2 switches changed by hand", and Reset To Default Configuration
 * is grayed out when this list is empty for "default".
 *
 * @param {unknown} state A complete switch state (validated like every helper).
 * @param {unknown} presetId The preset in use; "default" for Default
 *   Configuration.
 * @returns {ReadonlyArray<VisualFilterId>} A new frozen list in registry order;
 *   its length is the "changed by hand" count. Empty means the state is exactly
 *   the preset.
 * @throws {TypeError} When state is not a complete switch state.
 * @throws {RangeError} When presetId is not a known preset.
 */
export function visualFilterHandChanges(state, presetId) {
  const normalized = assertVisualFilterState(state);
  const presetState = visualFilterPresetState(presetId);
  return Object.freeze(
    VISUAL_FILTER_IDS.filter((id) => normalized[id] !== presetState[id]),
  );
}

// ---------------------------------------------------------------------------
// Paint parts: which switch controls each drawing
// ---------------------------------------------------------------------------

// Each look module lists the drawable parts it owns in its PAINT_PARTS export,
// as {part, filter} records. The table is built on first use, not at load, so a
// look module may itself import this file without an import-order trap.
/** @type {ReadonlyArray<readonly [string, () => unknown]>} */
const LOOK_PAINT_PART_SOURCES = Object.freeze([
  ["map-skins", () => MAP_SKINS_PAINT_PARTS],
  ["obstacles", () => OBSTACLES_PAINT_PARTS],
  ["red-zone", () => RED_ZONE_PAINT_PARTS],
  ["spawns", () => SPAWNS_PAINT_PARTS],
  ["team-identity", () => TEAM_IDENTITY_PAINT_PARTS],
  ["atmosphere", () => ATMOSPHERE_PAINT_PARTS],
  ["champions", () => CHAMPIONS_PAINT_PARTS],
  ["health-text", () => HEALTH_TEXT_PAINT_PARTS],
  ["statuses", () => STATUSES_PAINT_PARTS],
  ["movement", () => MOVEMENT_PAINT_PARTS],
  ["basics", () => BASICS_PAINT_PARTS],
  ["ultimates", () => ULTIMATES_PAINT_PARTS],
  ["requiems", () => REQUIEMS_PAINT_PARTS],
  ["heralds", () => HERALDS_PAINT_PARTS],
  ["fog", () => FOG_PAINT_PARTS],
  ["camera", () => CAMERA_PAINT_PARTS],
  ["hud", () => HUD_PAINT_PARTS],
  ["frame-explanation", () => FRAME_EXPLANATION_PAINT_PARTS],
  ["hotbar", () => HOTBAR_PAINT_PARTS],
  ["champion-frame", () => CHAMPION_FRAME_PAINT_PARTS],
  ["focus-tally", () => FOCUS_TALLY_PAINT_PARTS],
]);

/**
 * @typedef {Readonly<{
 *   registrations: ReadonlyArray<VisualPaintPartRegistration>,
 *   filterByKey: ReadonlyMap<string, VisualFilterId>,
 * }>} PaintPartTable
 */

/** @type {PaintPartTable | null} */
let paintPartTable = null;

/**
 * Build one paint-part table from look-module lists.
 *
 * Each source is [owner, parts]: owner is the look module name and parts is
 * its PAINT_PARTS array of exact {part, filter} records, where part is the
 * drawable-part tag record (string surface and kind plus any other string
 * tags) and filter is the switch ID that controls it. A part listed by two
 * sources, or twice by one, throws: every part has exactly one owner.
 *
 * Exported for the paint-part tests; the renderer uses
 * visualPaintPartRegistry() and classifyVisualPaintPart().
 *
 * @param {ReadonlyArray<readonly [string, unknown]>} sources Look-module lists,
 *   in a fixed order.
 * @returns {PaintPartTable} Frozen registrations in source order and a
 *   key-to-switch map. Each registration copies and freezes its tag.
 * @throws {TypeError} When a list is not an array, an entry is not exactly
 *   {part, filter}, a part is malformed, or a part is listed twice.
 * @throws {RangeError} When an entry names an unknown switch.
 */
export function buildPaintPartTable(sources) {
  /** @type {VisualPaintPartRegistration[]} */
  const registrations = [];
  /** @type {Map<string, VisualFilterId>} */
  const filterByKey = new Map();
  /** @type {Map<string, string>} */
  const ownerByKey = new Map();
  for (const [owner, parts] of sources) {
    if (!Array.isArray(parts)) {
      throw new TypeError(`Look module ${owner} must export a PAINT_PARTS array.`);
    }
    for (const entry of parts) {
      if (!isRecord(entry)) {
        throw new TypeError(
          `Look module ${owner} has a paint part that is not an object.`,
        );
      }
      assertExactKeys(entry, ["filter", "part"], `Look module ${owner} paint part`);
      const filterId = assertVisualFilterId(entry.filter);
      const key = paintPartKey(entry.part);
      const earlier = ownerByKey.get(key);
      if (earlier !== undefined) {
        throw new TypeError(
          `Duplicate visual paint-part registration ${key} (${earlier} and ${owner}).`,
        );
      }
      ownerByKey.set(key, owner);
      filterByKey.set(key, filterId);
      registrations.push(
        Object.freeze({
          tag: Object.freeze({ .../** @type {Record<string, string>} */ (entry.part) }),
          filterId,
          owner,
        }),
      );
    }
  }
  return Object.freeze({
    registrations: Object.freeze(registrations),
    filterByKey,
  });
}

/**
 * Return the shared paint-part table, building it on first use from every
 * look module's PAINT_PARTS. Throws the same errors as buildPaintPartTable,
 * on every call, until the lists are fixed.
 *
 * @returns {PaintPartTable}
 */
function sharedPaintPartTable() {
  if (paintPartTable === null) {
    paintPartTable = buildPaintPartTable(
      LOOK_PAINT_PART_SOURCES.map(([owner, read]) => [owner, read()]),
    );
  }
  return paintPartTable;
}

/**
 * Return every registered paint part: each one names its exact tag record,
 * the switch that controls it and the look module that owns it ("owner").
 *
 * @returns {ReadonlyArray<VisualPaintPartRegistration>} The shared frozen list.
 * @throws {TypeError | RangeError} When a look module's PAINT_PARTS list is
 *   invalid or repeats a part (see buildPaintPartTable).
 */
export function visualPaintPartRegistry() {
  return sharedPaintPartTable().registrations;
}

/**
 * Return the boolean for filterId in state. Require the registered filter
 * name and exactly the expected enumerable state keys with boolean values.
 * Invalid state throws TypeError; unknown filter IDs throw RangeError. No
 * state copy, freeze or mutation occurs.
 *
 * @param {unknown} state
 * @param {unknown} filterId
 */
export function isVisualFilterEnabled(state, filterId) {
  const normalized = assertVisualFilterState(state);
  const id = assertVisualFilterId(filterId);
  return normalized[id];
}

/**
 * Set one filter without mutating the supplied state. Validate state and
 * filterId, and require enabled to be boolean. Return the original object for
 * a no-op; otherwise return a new frozen state. A valid input need not itself
 * be frozen, so the no-op return is not promised to be frozen. Invalid input
 * throws TypeError or RangeError through the shared validators.
 *
 * @param {unknown} state
 * @param {unknown} filterId
 * @param {unknown} enabled
 * @returns {VisualFilterState}
 */
export function setVisualFilterEnabled(state, filterId, enabled) {
  const normalized = assertVisualFilterState(state);
  const id = assertVisualFilterId(filterId);
  if (typeof enabled !== "boolean") {
    throw new TypeError("Visual filter enabled must be a boolean.");
  }
  if (normalized[id] === enabled) {
    return normalized;
  }
  return freezeVisualFilterState(
    Object.fromEntries(
      VISUAL_FILTER_IDS.map((candidate) => [
        candidate,
        candidate === id ? enabled : normalized[candidate],
      ]),
    ),
  );
}

/**
 * Validate state and return a state with every filter enabled. If all are
 * already true, reuse the input unchanged; otherwise return the shared frozen
 * all-enabled state. Invalid state throws TypeError. A reused input is not
 * automatically frozen.
 *
 * @param {unknown} state
 * @returns {VisualFilterState}
 */
export function enableAllVisualFilters(state) {
  const normalized = assertVisualFilterState(state);
  return VISUAL_FILTER_IDS.every((id) => normalized[id])
    ? normalized
    : ENABLED_VISUAL_FILTER_STATE;
}

/**
 * Validate state and return a state with every filter disabled. If all are
 * already false, reuse the input unchanged; otherwise return the shared frozen
 * all-disabled state. Invalid state throws TypeError. No input is mutated or
 * automatically frozen.
 *
 * @param {unknown} state
 * @returns {VisualFilterState}
 */
export function disableAllVisualFilters(state) {
  const normalized = assertVisualFilterState(state);
  return VISUAL_FILTER_IDS.every((id) => !normalized[id])
    ? normalized
    : DISABLED_VISUAL_FILTER_STATE;
}

/**
 * Apply one validated page-local filter action and return the resulting state.
 *
 * state uses the shared state validator. action.type is one of:
 * - "set": exactly {type, filterId, enabled}; one switch changes.
 * - "enable_all" / "disable_all": exactly {type}; every switch on or off.
 * - "restore_defaults": exactly {type}; Reset To Default Configuration.
 * - "apply_preset": exactly {type, presetId}; the preset's full state
 *   (visualFilterPresetState), which drops every change made by hand.
 *
 * Invalid shape/value throws TypeError; unknown action, filter or preset
 * names throw RangeError. A result equal to the input may reuse it; changed,
 * default and preset states are shared frozen objects or new frozen states.
 * This function changes no DOM, no storage and no scientific data, and never
 * touches View, Ranges, Layout or Motion.
 *
 * @param {unknown} state
 * @param {unknown} action
 * @returns {VisualFilterState}
 */
export function reduceVisualFilterState(state, action) {
  const normalized = assertVisualFilterState(state);
  if (!isRecord(action) || typeof action.type !== "string") {
    throw new TypeError("Visual filter action must be a tagged object.");
  }
  if (action.type === "set") {
    assertExactKeys(action, ["enabled", "filterId", "type"], "set action");
    return setVisualFilterEnabled(normalized, action.filterId, action.enabled);
  }
  if (action.type === "enable_all") {
    assertExactKeys(action, ["type"], "enable-all action");
    return enableAllVisualFilters(normalized);
  }
  if (action.type === "disable_all") {
    assertExactKeys(action, ["type"], "disable-all action");
    return disableAllVisualFilters(normalized);
  }
  if (action.type === "restore_defaults") {
    assertExactKeys(action, ["type"], "restore-defaults action");
    return DEFAULT_VISUAL_FILTER_STATE;
  }
  if (action.type === "apply_preset") {
    assertExactKeys(action, ["presetId", "type"], "apply-preset action");
    return visualFilterPresetState(action.presetId);
  }
  throw new RangeError(`Unknown visual filter action ${action.type}.`);
}

/**
 * Validate state and encode its booleans in the fixed registry order. Return
 * a "visual-filters-v4:" string with one flag per switch, for local redraw and
 * cache decisions. Invalid state throws TypeError. The key describes display
 * preferences, not scientific conditions or information rights, and must not
 * enter result fingerprints. Version 4 started with the V5 switch list.
 *
 * @param {unknown} state
 */
export function visualFilterPaintKey(state) {
  const normalized = assertVisualFilterState(state);
  return `visual-filters-v4:${VISUAL_FILTER_IDS.map((id) =>
    normalized[id] ? "1" : "0",
  ).join("")}`;
}

/**
 * Return the registered filter ID for an exact drawable-part tag record.
 * part must contain nonempty string tags including surface and kind. The
 * canonical sorted key must be listed by a look module's PAINT_PARTS;
 * malformed input throws TypeError and an unknown combination
 * throws RangeError. No future/unrecognized part is silently enabled. The first
 * call builds the shared table and can throw its errors.
 *
 * @param {unknown} part
 * @returns {VisualFilterId}
 */
export function classifyVisualPaintPart(part) {
  const key = paintPartKey(part);
  const filterId = sharedPaintPartTable().filterByKey.get(key);
  if (filterId === undefined) {
    throw new RangeError(`Unregistered visual paint part ${key}.`);
  }
  return filterId;
}

/**
 * Classify part, validate state and return that part's enabled flag.
 * The classification runs first. Propagate TypeError/RangeError for malformed
 * or unregistered inputs. This controls drawing only and changes nothing.
 *
 * @param {unknown} state
 * @param {unknown} part
 */
export function isVisualPaintPartEnabled(state, part) {
  return isVisualFilterEnabled(state, classifyVisualPaintPart(part));
}

/**
 * Create a frozen preset record. The authored caller owns validity; the
 * module-level checks below verify every override names a registered switch.
 *
 * @param {VisualFilterPresetId} id
 * @param {string} label Title Case button label.
 * @param {string} purpose One plain sentence for the button's tip.
 * @param {Partial<Record<VisualFilterId, boolean>>} overrides Switches the
 *   preset sets; every other switch keeps its default.
 * @returns {VisualFilterPreset}
 */
function preset(id, label, purpose, overrides) {
  return Object.freeze({
    id,
    label,
    purpose,
    overrides: Object.freeze({ ...overrides }),
  });
}

/**
 * Return presetId when it names a known preset; otherwise throw RangeError.
 * No trimming or coercion is used.
 *
 * @param {unknown} presetId
 * @returns {string}
 */
function assertPresetId(presetId) {
  if (
    typeof presetId !== "string" ||
    !PRESET_BY_ID.has(/** @type {any} */ (presetId))
  ) {
    throw new RangeError(`Unknown visual filter preset ${String(presetId)}.`);
  }
  return presetId;
}

/**
 * Sort value's enumerable string entries and serialize a drawable-part key.
 * Require a non-array object, at least two entries, string surface/kind and
 * nonempty string keys/values. Throw TypeError otherwise. This is not a safe
 * parser for hostile accessors; callers pass normalized/local data. It does
 * not itself check registry membership.
 *
 * @param {unknown} value
 */
function paintPartKey(value) {
  if (!isRecord(value)) {
    throw new TypeError("Visual paint part must be a tagged object.");
  }
  const entries = Object.entries(value).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  if (
    entries.length < 2 ||
    typeof value.surface !== "string" ||
    typeof value.kind !== "string"
  ) {
    throw new TypeError("Visual paint part requires string surface and kind tags.");
  }
  for (const [key, field] of entries) {
    if (typeof field !== "string" || key.length === 0 || field.length === 0) {
      throw new TypeError("Visual paint-part tags must be non-empty strings.");
    }
  }
  return JSON.stringify(entries);
}

/**
 * Validate value's enumerable string keys and registered boolean fields.
 * Return the same object; do not copy or freeze it. Missing/extra enumerable
 * keys or nonboolean fields throw TypeError. This local-state check does not
 * reject inherited/symbol fields or inspect property descriptors.
 *
 * @param {unknown} value @returns {VisualFilterState}
 */
function assertVisualFilterState(value) {
  if (!isRecord(value)) {
    throw new TypeError("Visual filter state must be an object.");
  }
  assertExactKeys(value, [...VISUAL_FILTER_IDS], "visual filter state");
  for (const id of VISUAL_FILTER_IDS) {
    if (typeof value[id] !== "boolean") {
      throw new TypeError(`Visual filter ${id} must be a boolean.`);
    }
  }
  return /** @type {VisualFilterState} */ (value);
}

/**
 * Return value when it is an exact registered filter-name string. Otherwise
 * throw RangeError. No trimming or coercion is used for membership.
 *
 * @param {unknown} value @returns {VisualFilterId}
 */
function assertVisualFilterId(value) {
  if (
    typeof value !== "string" ||
    !VISUAL_FILTER_ID_SET.has(/** @type {VisualFilterId} */ (value))
  ) {
    throw new RangeError(`Unknown visual filter ${String(value)}.`);
  }
  return /** @type {VisualFilterId} */ (value);
}

/**
 * Require value's enumerable string keys to equal expected, ignoring order.
 * Throw TypeError using label when they differ; return undefined on success.
 * This does not validate values or reject symbols/non-enumerable fields.
 *
 * @param {Record<string, unknown>} value
 * @param {ReadonlyArray<string>} expected
 * @param {string} label
 */
function assertExactKeys(value, expected, label) {
  const actual = Object.keys(value).sort();
  const canonical = [...expected].sort();
  if (
    actual.length !== canonical.length ||
    actual.some((key, index) => key !== canonical[index])
  ) {
    throw new TypeError(`${label} has an invalid shape.`);
  }
}

/**
 * Return a frozen shallow copy of value. The caller supplies the complete
 * boolean filter record; this helper does not validate its keys or values.
 *
 * @param {Record<string, boolean>} value @returns {VisualFilterState}
 */
function freezeVisualFilterState(value) {
  return /** @type {VisualFilterState} */ (Object.freeze({ ...value }));
}

/**
 * Return true when value is a non-null object that is not an array.
 * This is a shape check only; no prototype, descriptor or authority validation.
 *
 * @param {unknown} value @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Load-time checks of the authored tables: unique switch IDs, known groups,
// and presets that name only registered switches. A mistake here fails at load.
if (VISUAL_FILTER_ID_SET.size !== VISUAL_FILTER_IDS.length) {
  throw new TypeError("Visual filter registry has a repeated switch ID.");
}
for (const { id, group } of VISUAL_FILTER_REGISTRY) {
  if (!VISUAL_FILTER_GROUPS.includes(group)) {
    throw new TypeError(`Visual filter ${id} has unknown group ${group}.`);
  }
}
for (const { id, overrides } of VISUAL_FILTER_PRESETS) {
  for (const filterId of Object.keys(overrides)) {
    if (!VISUAL_FILTER_ID_SET.has(/** @type {VisualFilterId} */ (filterId))) {
      throw new TypeError(
        `Visual filter preset ${id} names unknown switch ${filterId}.`,
      );
    }
  }
}
