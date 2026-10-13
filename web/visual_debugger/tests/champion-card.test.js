/**
 * @file Check the Champion Card (src/tooltip.js), the one card every client
 * shows for a hovered champion (design C "Clean List", owner, 2 October 2026).
 *
 * Contracts: the fifteen approved status names (`symbols.js` STATUS_NAMES)
 * name the card's rows, one per status, each with its own drawn icon (Rest
 * with its hand-drawn vessel), with the exact number in the
 * sentence (×1.5 damage, ×0.5 speed, 85% speed) and the source by class only;
 * rows go stuns first, then slows, then the rest, at most four, then "+N
 * more"; the Life bar is green above half, yellow at half or less and red at
 * 30% or less (51, 50, 31 and 30 percent); the Ultimate bar is named after the
 * Ultimate, fills as it recharges and reads "Ready" when ready; a scene row
 * gives the same facts whether or not it carries Oracle View's direct sources,
 * so Agent POV and Oracle View show the same card; Spawn Shield, In Combat
 * (each shown once, even when the scene also lists it as a status), Rest (with
 * the life it mends, cut by Crippling Poison), the two auras and a broken
 * Freezing Trap come from the row (its sentence, "Damage broke the ice
 * early.", never says the champion is free, since another stun may hold it);
 * a fallen champion's note, "Fallen · Returns in N", comes from its team's
 * public wave clock (countdown + 1), or reads "Fallen" without one; a
 * champion the team cannot see gets
 * the reduced card; renderChampionCard draws the rows, the drain lines, the
 * shared Life bar and the footer ("+N more" with "Hold Shift for All"); and a
 * Champion Card descriptor carries its model for the Replay Viewer's tooltip
 * controller.
 *
 * Owner rulings of 2 October 2026: the one shared Life bar (`life-bar.js`:
 * its tones at the same lines, its numbers, its fill width and its three
 * parts); a champion's own aura reads "From its own aura."; the card is placed
 * fully inside the arena box (8 px margin), also for a pointer outside the box
 * (a portrait or a docked frame); a Hold Shift card taller than the box
 * spreads into two, three or four columns (as many as the box's width allows)
 * and, only when even that is too tall, is scaled down to fit; a card that
 * fits again drops the wide look and the scale.
 * "Fewer rows" (owner, 2 October 2026): a plain card never widens, it stays
 * one column (scaled down only when even one row cannot fit);
 * championCardRowsThatFit shows every row that fits, else the most leading
 * rows that fit with the "+N more" footer, never fewer than one, counting
 * the footer whenever the four-row limit already left rows out and never
 * skipping a tall first row for shorter ones; a model's rowLimit keeps the
 * row order, shows 1 to 4 rows with "+N more · Hold Shift for All", and the
 * Hold Shift card ignores it; a Hold Shift card spreads into more columns
 * only when they make it shorter (four rows take two lines in two or three
 * columns, so it stays two columns and is scaled down instead).
 * Hold Shift (createShiftHold): Shift held by itself for 300 ms expands the
 * card to every status ("All 7 statuses", "Release Shift to Shrink"), a key
 * repeat or a same-spot pointer move changes nothing, Shift with Ctrl, Alt or
 * Meta never expands, and releasing Shift, the window's blur, any other key,
 * any pointer press, any pointer movement and the wheel shrink it at once;
 * every Shift combination (Shift+click, Shift+Tab, Shift+arrows, "?", Shift+R)
 * still reaches the page's own handler uncancelled, never expands the card
 * when it comes within 300 ms, and shrinks an expanded card at once.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { sigilDefinition } from "../src/icons.js";
import { createLifeBar, lifeBarModel, lifeTone } from "../src/life-bar.js";
import { STATUS_NAMES } from "../src/symbols.js";
import {
  CHAMPION_CARD_BOX_MARGIN,
  CHAMPION_CARD_HOLD_MS,
  CHAMPION_CARD_KIND,
  CHAMPION_CARD_MIN_INSIDE_SCALE,
  CHAMPION_CARD_STATUS_LIMIT,
  championCardFactsFromAgent,
  championCardFooter,
  championCardLayout,
  championCardModel,
  championCardOf,
  championCardPastArenaPosition,
  championCardRowsThatFit,
  championCardText,
  createChampionCardDescriptor,
  createShiftHold,
  hoverCardPosition,
  placeChampionCard,
  renderChampionCard,
} from "../src/tooltip.js";

/** @typedef {import("../src/tooltip.js").ChampionCardFacts} Facts */

/**
 * @param {Partial<Facts>} [overrides]
 * @returns {Facts}
 */
function facts(overrides = {}) {
  return {
    state: "seen",
    name: "Mage",
    calling: "The Astral Walker",
    classId: 1,
    displayId: "5",
    teamId: 2,
    life: { current: 77, maximum: 80 },
    ultimate: { name: "Burst", remaining: 30, full: 30 },
    statuses: [],
    note: null,
    you: false,
    playedBy: "tdm-beta",
    yourTarget: false,
    unseenBy: null,
    ...overrides,
  };
}

/**
 * @param {string} id
 * @param {number | null} [ticks]
 * @param {number | null} [amount]
 */
const status = (id, ticks = 3, amount = null) => ({ id, ticks, full: 5, amount });

// A Mage scene row as the audience's arena scene gives it.
const MAGE_ROW = Object.freeze({
  public_agent_id: "5",
  display_agent_id: "5",
  presentation_key: "pk-5",
  team_id: 2,
  class_id: 1,
  life_state: "alive",
  current_health: 40,
  maximum_health: 80,
  ultimate_cooldown_remaining: 12,
  spawn_shield_remaining: 0,
  steps_until_out_of_combat: 0,
  out_of_combat_delay_steps: 5,
  out_of_combat_health_regeneration_fraction_per_step: 0.02,
  statuses: [
    {
      status_id: "warrior_charge_slow",
      remaining_duration: 3,
      configured_duration_steps: 5,
      magnitude_kind: "movement_multiplier",
      magnitude: 0.5,
      source_class_id: 2,
      source_action_component: "ultimate",
    },
    {
      status_id: "rogue_poison_anti_heal",
      remaining_duration: 4,
      configured_duration_steps: 5,
      magnitude_kind: "healing_multiplier",
      magnitude: 0.5,
      source_class_id: 4,
      source_action_component: "ultimate",
    },
  ],
  aura_modifiers: [{ aura_id: "warrior_damage_mitigation", multiplier: 0.85 }],
});

test("the fifteen approved names name the card's rows", () => {
  const titles = STATUS_NAMES.map((entry) => {
    const [row] = championCardModel(
      facts({ statuses: [status(entry.id, 2, 1.5)] }),
    ).rows;
    return row.effect === null ? row.name : `${row.name} (${row.effect})`;
  });
  assert.deepEqual(titles, [
    "Freezing Trap (Stun)",
    "Charge (Stun)",
    "Crippling Poison (Stun)",
    "Freezing Trap (Stun Broken)",
    "Charge (Slow)",
    "Crippling Poison (Slow)",
    "Serrated Arrows (Slow)",
    "Crippling Poison (Healing Reduction)",
    "Blessing of Freedom (Slow Resistance)",
    "Burst (Damage Boost)",
    "Spawn Shield (Invulnerable)",
    "In Combat",
    "Rest (Regeneration)",
    "Sorcerer's Empowerment (Damage Boost)",
    "Guardian's Barrier (Damage Mitigation)",
  ]);
  const line = (/** @type {string} */ id, /** @type {number | null} */ amount) =>
    championCardModel(facts({ statuses: [status(id, 2, amount)] })).rows[0].line;
  assert.equal(
    line("mage_burst", 1.5),
    "Its attacks deal ×1.5 damage. From its own Ultimate.",
  );
  assert.equal(
    line("slow_warrior_charge", 0.5),
    "It moves at ×0.5 speed. From a Warrior.",
  );
  assert.equal(
    line("priest_freedom", 0.8500000238418579),
    "Slows can't drop it below 85% speed. From a Priest.",
  );
  assert.equal(
    line("slow_hunter_basic", null),
    "It moves a little slower. From a Hunter.",
  );
  assert.equal(line("stun_rogue_poison", null), "It can't move or act. From a Rogue.");
  assert.equal(line("in_combat", null), "It fought recently, so it can't rest yet.");
  // A broken Freezing Trap never says the champion is free: another stun (a
  // Charge that broke the ice) may still hold it on the same tick.
  assert.equal(line("stun_broken", null), "Damage broke the ice early. From a Hunter.");
  // Every row names its source by class only and has its own drawn icon.
  for (const entry of STATUS_NAMES) {
    assert.doesNotMatch(line(entry.id, 1.5), /\bID\b|hidden/u, entry.id);
    const [row] = championCardModel(facts({ statuses: [status(entry.id)] })).rows;
    assert.notEqual(sigilDefinition(row.icon), null, entry.id);
  }
  const icon = (/** @type {string} */ id) =>
    championCardModel(facts({ statuses: [status(id)] })).rows[0].icon;
  assert.equal(icon("rest"), "sigil-rest-vessel");
  assert.equal(icon("stun_broken"), "sigil-life-shattered");
  assert.equal(icon("stun_warrior_charge"), "sigil-dazed-warrior");
  assert.equal(icon("mage_amplification"), "sigil-empowerment");
  assert.equal(
    championCardModel(facts({ statuses: [status("future_status")] })).rows[0].name,
    "Recorded Status",
  );
});

test("rows go stuns first, then slows, then the rest, four at most, then +N more", () => {
  const model = championCardModel(
    facts({
      statuses: [
        status("in_combat", 5),
        status("slow_hunter_basic", 2, 0.85),
        status("stun_rogue_poison", 1),
        status("mage_burst", 5, 1.5),
        status("stun_hunter_trap", 4),
        status("priest_freedom", 1, 0.85),
      ],
    }),
  );
  assert.equal(CHAMPION_CARD_STATUS_LIMIT, 4);
  assert.deepEqual(
    model.rows.map((row) => [row.id, row.time]),
    [
      ["stun_hunter_trap", "4 ticks left"],
      ["stun_rogue_poison", "1 tick left"],
      ["slow_hunter_basic", "2 ticks left"],
      ["priest_freedom", "1 tick left"],
    ],
  );
  assert.equal(model.more, 2);
  assert.deepEqual(
    model.rows.map((row) => row.drain),
    [0.8, 0.2, 0.4, 0.2],
  );
  assert.equal(championCardText(model).at(-2), "+2 more · Hold Shift for All");
});

test("life is green above half, yellow at half or less, red at 30% or less", () => {
  assert.deepEqual(
    [51, 50, 31, 30, 0].map((life) => lifeTone(life, 100)),
    ["good", "warn", "warn", "low", "low"],
  );
  assert.equal(lifeTone(24, 80), "low");
  assert.equal(lifeTone(5, null), "good");
  const low = championCardModel(facts({ life: { current: 24, maximum: 80 } })).life;
  assert.deepEqual(low, { text: "24", maximumText: "80", fraction: 0.3, tone: "low" });
  assert.equal(
    championCardModel(facts({ life: { current: 199.6, maximum: 200 } })).life?.text,
    "199",
  );
});

test("the Ultimate bar is named after the Ultimate and reads Ready when ready", () => {
  assert.deepEqual(
    championCardModel(facts({ ultimate: { name: "Burst", remaining: 0, full: 30 } }))
      .ultimate,
    { label: "Burst", ready: true, fraction: 1, text: "Ready", classToken: "mage" },
  );
  const charging = championCardModel(
    facts({ ultimate: { name: "Burst", remaining: 15, full: 30 } }),
  ).ultimate;
  assert.deepEqual([charging?.fraction, charging?.text], [0.5, "15 ticks"]);
  assert.equal(
    championCardModel(facts({ ultimate: { name: null, remaining: 1, full: null } }))
      .ultimate?.text,
    "1 tick",
  );
  assert.equal(
    championCardModel(facts({ ultimate: { name: null, remaining: 1, full: null } }))
      .ultimate?.label,
    "Ultimate",
  );
});

test("a scene row gives the same card with or without Oracle View's direct sources", () => {
  const oracleRow = {
    ...MAGE_ROW,
    statuses: MAGE_ROW.statuses.map((row) => ({
      ...row,
      direct_sources: [
        { source_public_agent_id: "1", source_presentation_key: "pk-1" },
      ],
    })),
  };
  const context = {
    classMechanics: [{ class_id: 1, ultimate_cooldown_steps: 30 }],
    playedBy: "tdm-beta",
  };
  const oracle = championCardFactsFromAgent(oracleRow, context);
  assert.deepEqual(championCardFactsFromAgent(MAGE_ROW, context), oracle);
  assert.deepEqual(championCardText(championCardModel(oracle)), [
    "Mage · The Astral Walker",
    "Agent ID 5",
    "Life 40 / 80",
    "Burst 12 ticks",
    "Charge (Slow) · 3 ticks left · It moves at ×0.5 speed. From a Warrior.",
    "Crippling Poison (Healing Reduction) · 4 ticks left · Heals and rest give it ×0.5 healing. From a Rogue.",
    "Rest (Regeneration) · Now · It mends 0.8 life every tick.",
    "Guardian's Barrier (Damage Mitigation) · While near · It takes ×0.85 damage near a Warrior. From a Warrior's aura.",
    "Played by tdm-beta",
  ]);
  assert.equal(championCardModel(oracle).ultimate?.fraction, 0.6);
});

test("Spawn Shield, In Combat, a broken Freezing Trap and a fall come from the row", () => {
  const shielded = championCardFactsFromAgent(
    {
      ...MAGE_ROW,
      statuses: [],
      aura_modifiers: [],
      spawn_shield_remaining: 2,
      steps_until_out_of_combat: 4,
    },
    { spawnShieldSteps: 3, trapBroke: true, you: true, yourTarget: true },
  );
  assert.deepEqual(
    shielded.statuses.map((row) => [row.id, row.ticks, row.full]),
    [
      ["spawn_shield", 2, 3],
      ["stun_broken", null, null],
      ["in_combat", 4, 5],
    ],
  );
  const model = championCardModel(shielded);
  assert.deepEqual(
    model.rows.map((row) => [row.id, row.time]),
    [
      ["stun_broken", "This tick"],
      ["spawn_shield", "2 ticks left"],
      ["in_combat", "4 ticks left"],
    ],
  );
  assert.equal(model.played, null);
  assert.deepEqual(championCardText(model).slice(-2), [
    "You play this champion",
    "Your Target",
  ]);
  // A scene that also lists In Combat and Spawn Shield among its statuses
  // shows each once.
  const listed = championCardFactsFromAgent({
    ...MAGE_ROW,
    statuses: [
      { status_id: "in_combat", remaining_duration: 4 },
      { status_id: "spawn_shield", remaining_duration: 2 },
    ],
    aura_modifiers: [],
    spawn_shield_remaining: 2,
    steps_until_out_of_combat: 4,
  });
  assert.deepEqual(
    listed.statuses.map((row) => [row.id, row.ticks, row.full]),
    [
      ["spawn_shield", 2, null],
      ["in_combat", 4, 5],
    ],
  );
  // A fallen champion's note comes from its own team's public wave clock
  // (countdown + 1, as the HUD's fallen frame counts), so arena bodies,
  // portraits and frames all show the same line.
  const waves = [
    { team_id: 1, countdown_steps: 0 },
    { team_id: 2, countdown_steps: 2 },
  ];
  const fallen = championCardFactsFromAgent(
    { ...MAGE_ROW, life_state: "corpse", current_health: 12 },
    { respawnWaves: waves },
  );
  assert.deepEqual(fallen.life, { current: 0, maximum: 80 });
  assert.equal(fallen.ultimate, null);
  assert.deepEqual(fallen.statuses, []);
  assert.equal(fallen.note, "Fallen · Returns in 3");
  assert.equal(
    championCardFactsFromAgent(
      { ...MAGE_ROW, team_id: 1, alive: false },
      { respawnWaves: waves },
    ).note,
    "Fallen · Returns in 1",
  );
  // No clock for its team: just "Fallen". A living champion has no note.
  assert.equal(
    championCardFactsFromAgent({ ...MAGE_ROW, life_state: "corpse" }).note,
    "Fallen",
  );
  assert.equal(
    championCardFactsFromAgent(MAGE_ROW, { respawnWaves: waves }).note,
    null,
  );
});

test("a champion the team cannot see gets the reduced card", () => {
  const unseen = championCardModel(
    facts({
      state: "unseen",
      life: null,
      ultimate: null,
      statuses: [status("mage_burst")],
      unseenBy: "Team A",
    }),
  );
  assert.deepEqual(championCardText(unseen), [
    "Mage · The Astral Walker",
    "Agent ID 5",
    "Unseen · Team A cannot see it now.",
    "Played by tdm-beta",
  ]);
  assert.deepEqual(
    [unseen.life, unseen.ultimate, unseen.rows, unseen.more],
    [null, null, [], 0],
  );
});

// A small document stand-in: elements with children, attributes and style.
function fakeDocument() {
  /** @param {string} tagName */
  const create = (tagName) => {
    /** @type {any[]} */
    let children = [];
    const attributes = new Map();
    /** @type {any} */
    const node = {
      tagName: tagName.toUpperCase(),
      className: "",
      dataset: {},
      style: {},
      get children() {
        return children;
      },
      get textContent() {
        return children
          .map((child) => (typeof child === "string" ? child : child.textContent))
          .join("");
      },
      set textContent(value) {
        children = [String(value)];
      },
      /** @param {...any} nodes */
      append(...nodes) {
        children.push(...nodes);
      },
      /** @param {...any} nodes */
      replaceChildren(...nodes) {
        children = [...nodes];
      },
      /** @param {string} name @param {unknown} value */
      setAttribute(name, value) {
        attributes.set(name, String(value));
      },
      /** @param {string} name */
      getAttribute(name) {
        return attributes.get(name) ?? null;
      },
    };
    node.ownerDocument = doc;
    return node;
  };
  const doc = {
    createElement: create,
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      create(tag),
  };
  return doc;
}

/** @param {any} root @param {string} className @returns {any[]} */
const byClass = (root, className) => [
  ...(typeof root !== "string" && root.className.split(" ").includes(className)
    ? [root]
    : []),
  ...(typeof root === "string" ? [] : root.children).flatMap(
    (/** @type {any} */ child) => byClass(child, className),
  ),
];

test("renderChampionCard draws the bars, rows, drain lines and +N more", () => {
  const doc = fakeDocument();
  const host = doc.createElement("div");
  const root = renderChampionCard(
    /** @type {any} */ (host),
    championCardModel(
      facts({
        life: { current: 33, maximum: 80 },
        ultimate: { name: "Burst", remaining: 0, full: 30 },
        statuses: [
          status("stun_hunter_trap", 4),
          status("stun_warrior_charge", 1),
          status("slow_warrior_charge", 3, 0.5),
          status("anti_heal_rogue_poison", 4, 0.5),
          status("mage_burst", 5, 1.5),
        ],
        yourTarget: true,
      }),
    ),
  );
  assert.deepEqual(host.children, [root]);
  assert.equal(root.className, "champion-card");
  assert.equal(root.dataset.life, "warn");
  const rows = byClass(root, "champion-card__row");
  assert.deepEqual(
    rows.map((row) => [row.dataset.status, row.dataset.paint]),
    [
      ["stun_hunter_trap", "rime"],
      ["stun_warrior_charge", "warrior"],
      ["slow_warrior_charge", "warrior"],
      ["anti_heal_rogue_poison", "rogue"],
    ],
  );
  assert.equal(
    byClass(rows[0], "champion-card__name")[0].textContent,
    "Freezing Trap (Stun)",
  );
  assert.equal(byClass(rows[0], "champion-card__time")[0].textContent, "4 ticks left");
  assert.equal(
    byClass(rows[0], "champion-card__drain")[0].children[0].style.width,
    "80.0%",
  );
  // The footer: "+1 more" on the left, "Hold Shift for All" on the right, with
  // Shift drawn as a key.
  const [footer] = byClass(root, "champion-card__more");
  assert.equal(footer.children[0].tagName, "B");
  assert.equal(footer.children[0].textContent, "+1 more");
  assert.equal(
    byClass(footer, "champion-card__hint")[0].textContent,
    "Hold Shift for All",
  );
  assert.equal(byClass(footer, "champion-card__key")[0].tagName, "KBD");
  // The Life bar is the shared one; the Ultimate keeps its own bar.
  const [lifeBar] = byClass(root, "life-bar");
  assert.equal(lifeBar.dataset.life, "warn");
  assert.equal(byClass(lifeBar, "life-bar__label")[0].textContent, "Life");
  assert.equal(byClass(lifeBar, "life-bar__fill")[0].style.width, "41.3%");
  assert.equal(byClass(lifeBar, "life-bar__value")[0].textContent, "33 / 80");
  const values = byClass(root, "champion-card__value").map((node) => node.textContent);
  assert.deepEqual(values, ["Ready"]);
  assert.equal(byClass(root, "champion-card__you")[0].textContent, "Your Target");
});

test("a Champion Card descriptor carries its model for the tooltip controller", () => {
  const descriptor = createChampionCardDescriptor(
    facts({ statuses: [status("mage_burst", 5, 1.5)] }),
    "champion:body:5",
  );
  assert.equal(descriptor.kind, CHAMPION_CARD_KIND);
  assert.equal(descriptor.id, "champion:body:5");
  assert.equal(descriptor.title, "Mage · The Astral Walker");
  assert.equal(descriptor.accent, "mage");
  assert.match(
    descriptor.summary,
    /Burst \(Damage Boost\) · 5 ticks left · Its attacks deal ×1\.5 damage\./u,
  );
  assert.equal(championCardOf(descriptor)?.rows[0].id, "mage_burst");
  assert.equal(championCardOf({ kind: "agent" }), null);
});

test("the shared Life bar: tones, numbers, fill and parts", () => {
  assert.deepEqual(lifeBarModel(77, 80), {
    text: "77",
    maximumText: "80",
    fraction: 0.9625,
    tone: "good",
  });
  assert.deepEqual(
    [
      [200, 200],
      [100, 200],
      [61, 200],
      [60, 200],
      [46, 200],
      [-3, 80],
    ].map(([current, maximum]) => lifeBarModel(current, maximum)?.tone),
    ["good", "warn", "warn", "low", "low", "low"],
  );
  assert.equal(lifeBarModel(-3, 80)?.text, "0");
  assert.equal(lifeBarModel(9.41, 80)?.text, "9.5");
  assert.equal(lifeBarModel(null, 80), null);
  assert.equal(lifeBarModel(40, Number.NaN), null);
  const doc = fakeDocument();
  const bar = createLifeBar(
    /** @type {any} */ (doc),
    /** @type {any} */ (lifeBarModel(14, 80)),
  );
  assert.equal(bar.className, "life-bar");
  assert.equal(bar.dataset.life, "low");
  assert.deepEqual(
    /** @type {any} */ (bar).children.map(
      (/** @type {any} */ child) => child.className,
    ),
    ["life-bar__label", "life-bar__track", "life-bar__value"],
  );
  assert.equal(byClass(bar, "life-bar__fill")[0].style.width, "17.5%");
  assert.equal(bar.textContent, "Life14 / 80");
});

test("Hold Shift: the card shows every status, the footer says how to shrink it", () => {
  const seven = facts({
    statuses: [
      status("in_combat", 5),
      status("slow_hunter_basic", 2, 0.85),
      status("stun_rogue_poison", 1),
      status("mage_burst", 5, 1.5),
      status("stun_hunter_trap", 4),
      status("priest_freedom", 1, 0.85),
      status("mage_amplification", null, 1.15),
    ],
  });
  const collapsed = championCardModel(seven);
  assert.deepEqual(
    [collapsed.rows.length, collapsed.more, collapsed.total, collapsed.expanded],
    [4, 3, 7, false],
  );
  assert.deepEqual(championCardFooter(collapsed), ["+3 more", "Hold Shift for All"]);
  const expanded = championCardModel(seven, { expanded: true });
  assert.deepEqual(
    [expanded.rows.length, expanded.more, expanded.total, expanded.expanded],
    [7, 0, 7, true],
  );
  // The same order, then the rest.
  assert.deepEqual(
    expanded.rows.slice(0, 4).map((row) => row.id),
    collapsed.rows.map((row) => row.id),
  );
  assert.deepEqual(championCardFooter(expanded), [
    "All 7 statuses",
    "Release Shift to Shrink",
  ]);
  assert.ok(
    championCardText(expanded).includes("All 7 statuses · Release Shift to Shrink"),
  );
  // A Mage's own aura reads "From its own aura." (owner, 2 October 2026).
  assert.equal(
    expanded.rows.find((row) => row.id === "mage_amplification")?.line,
    "It deals ×1.15 damage near a Mage. From its own aura.",
  );
  // A card with four rows or fewer has no footer, expanded or not (a short
  // arena box's Hold Shift card of four rows needs no extra height).
  const four = facts({ statuses: seven.statuses.slice(0, 4) });
  assert.equal(championCardFooter(championCardModel(four)), null);
  assert.equal(championCardFooter(championCardModel(four, { expanded: true })), null);
  // Expanding never adds facts to the reduced unseen card.
  const unseen = championCardModel(
    facts({ state: "unseen", life: null, ultimate: null, statuses: [] }),
    { expanded: true },
  );
  assert.deepEqual([unseen.rows, unseen.more, unseen.expanded], [[], 0, false]);
  // The drawn Hold Shift card carries its mark and its footer words.
  const doc = fakeDocument();
  const root = renderChampionCard(
    /** @type {any} */ (doc.createElement("div")),
    expanded,
  );
  assert.equal(root.dataset.expanded, "true");
  assert.equal(byClass(root, "champion-card__row").length, 7);
  const [footer] = byClass(root, "champion-card__more");
  assert.equal(footer.children[0].textContent, "All 7 statuses");
  assert.equal(
    byClass(footer, "champion-card__hint")[0].textContent,
    "Release Shift to Shrink",
  );
});

test("own aura: a Warrior's Guardian's Barrier on itself reads From its own aura", () => {
  const warrior = championCardFactsFromAgent({
    ...MAGE_ROW,
    class_id: 2,
    statuses: [],
    aura_modifiers: [{ aura_id: "warrior_damage_mitigation", multiplier: 0.85 }],
  });
  assert.equal(
    championCardModel(warrior).rows.find((row) => row.id === "warrior_mitigation")
      ?.line,
    "It takes ×0.85 damage near a Warrior. From its own aura.",
  );
  // A Mage in a Warrior's aura still reads the class form.
  assert.equal(
    championCardModel(championCardFactsFromAgent(MAGE_ROW)).rows.find(
      (row) => row.id === "warrior_mitigation",
    )?.line,
    "It takes ×0.85 damage near a Warrior. From a Warrior's aura.",
  );
});

test("the card is placed fully inside the arena box", () => {
  const arena = { width: 1076, height: 470 };
  const card = { width: 262, height: 330 };
  // Beside the pointer when there is room.
  assert.deepEqual(hoverCardPosition({ x: 100, y: 60 }, arena, card), {
    left: 116,
    top: 76,
  });
  // Near the right and bottom edges it flips, and stays 8 px inside.
  assert.deepEqual(hoverCardPosition({ x: 1000, y: 440 }, arena, card), {
    left: 722,
    top: 98,
  });
  // A pointer outside the box (a scorebug portrait above it, a docked frame
  // below it) puts the card at the nearest edge, still inside.
  for (const pointer of [
    { x: 300, y: -60 },
    { x: 300, y: 560 },
    { x: -40, y: 200 },
    { x: 1200, y: 200 },
  ]) {
    const place = hoverCardPosition(pointer, arena, card);
    assert.ok(place.left >= CHAMPION_CARD_BOX_MARGIN, JSON.stringify(pointer));
    assert.ok(place.top >= CHAMPION_CARD_BOX_MARGIN, JSON.stringify(pointer));
    assert.ok(place.left + card.width <= arena.width - CHAMPION_CARD_BOX_MARGIN);
    assert.ok(place.top + card.height <= arena.height - CHAMPION_CARD_BOX_MARGIN);
  }
  assert.equal(CHAMPION_CARD_BOX_MARGIN, 8);
});

// A card stand-in for placeChampionCard: its size depends on how many columns
// its rows spread into (heights[columns - 1]), and it keeps its look marks;
// expanded marks the Hold Shift card (data-expanded "true").
/** @param {number[]} heights @param {boolean} [rows] @param {boolean} [expanded] */
function measuredCard(heights, rows = true, expanded = true) {
  const classes = new Set(["champion-card"]);
  /** @type {Record<string, string>} */
  const dataset = expanded ? { expanded: "true" } : {};
  const root = {
    dataset,
    style: { transform: "x", transformOrigin: "" },
    classList: {
      add: (/** @type {string} */ name) => classes.add(name),
      remove: (/** @type {string} */ name) => classes.delete(name),
    },
    querySelector: () => (rows ? {} : null),
  };
  const surface = {
    getBoundingClientRect: () => {
      const columns = Number(dataset.columns ?? 1);
      return {
        width: 262 * columns + 16 * (columns - 1),
        height: heights[columns - 1],
      };
    },
  };
  return { root, surface, classes, dataset };
}

test("a Hold Shift card taller than the arena box spreads into more columns and stays inside", () => {
  const box = { left: 12, top: 144, width: 1076, height: 470 };
  /** @param {ReturnType<typeof measuredCard>} card */
  const place = (card) =>
    placeChampionCard({
      surface: /** @type {any} */ (card.surface),
      root: /** @type {any} */ (card.root),
      box,
      pointer: { x: 600, y: 300 },
    });
  // Fits: one column, beside the pointer, in page pixels. No room below, so
  // it flips above and is held 8 px inside the top edge.
  const short = measuredCard([330, 200, 150, 120]);
  assert.deepEqual(place(short), {
    left: 616,
    top: 152,
    wide: false,
    columns: 1,
    scale: 1,
    pastArena: false,
  });
  assert.equal(short.classes.has("champion-card--wide"), false);
  assert.equal(short.dataset.columns, undefined);
  // Taller than the box less its two 8 px margins: two columns, then inside.
  const tall = measuredCard([470 - 15, 320, 250, 200]);
  const two = place(tall);
  assert.deepEqual([two.wide, two.columns, two.scale], [true, 2, 1]);
  assert.equal(tall.dataset.columns, "2");
  assert.equal(tall.classes.has("champion-card--wide"), true);
  assert.ok(two.left >= box.left + 8 && two.left + 540 <= box.left + box.width - 8);
  assert.ok(two.top >= box.top + 8 && two.top + 320 <= box.top + box.height - 8);
  // Still too tall in two columns: three.
  const taller = measuredCard([900, 600, 400, 300]);
  assert.equal(place(taller).columns, 3);
  // A box too narrow for more columns and too short for the card: the
  // widest card that fits is scaled down to the box's height.
  const narrow = { left: 0, top: 0, width: 600, height: 300 };
  const huge = measuredCard([900, 600, 400, 300]);
  const fitted = placeChampionCard({
    surface: /** @type {any} */ (huge.surface),
    root: /** @type {any} */ (huge.root),
    box: narrow,
    pointer: { x: 100, y: 100 },
  });
  assert.equal(fitted.columns, 2);
  assert.equal(fitted.scale, 284 / 600);
  assert.equal(huge.root.style.transform, "scale(0.4733)");
  assert.equal(huge.root.style.transformOrigin, "0 0");
  assert.ok(fitted.top >= 8 && fitted.top + 600 * fitted.scale <= 300 - 8 + 1e-9);
  // Placing it again first drops the wide look and the scale, so a card that
  // now fits goes back to one column.
  const again = measuredCard([330, 200, 150, 120]);
  again.root.dataset.columns = "3";
  again.classes.add("champion-card--wide");
  place(again);
  assert.deepEqual(
    [
      again.dataset.columns,
      again.classes.has("champion-card--wide"),
      again.root.style.transform,
    ],
    [undefined, false, ""],
  );
  // The reduced card has no rows and never goes wide.
  assert.equal(place(measuredCard([600, 600, 600, 600], false)).wide, false);
  // A third column that makes the card no shorter (four rows take two lines
  // in two or three columns) is not kept: two columns, scaled to the box.
  const shortBox = { left: 12, top: 144, width: 1076, height: 290 };
  const same = measuredCard([400, 300, 300, 200]);
  const kept = placeChampionCard({
    surface: /** @type {any} */ (same.surface),
    root: /** @type {any} */ (same.root),
    box: shortBox,
    pointer: { x: 600, y: 300 },
  });
  assert.deepEqual([kept.columns, kept.scale], [2, 274 / 300]);
  assert.equal(same.dataset.columns, "2");
});

test("a plain card never widens: it stays one column inside the arena box", () => {
  const box = { left: 12, top: 144, width: 1076, height: 290 };
  // Too tall for the box even in one column (its rows were already fitted):
  // still one column, scaled down to the box, never three columns.
  const plain = measuredCard([400, 200, 150, 120], true, false);
  const place = placeChampionCard({
    surface: /** @type {any} */ (plain.surface),
    root: /** @type {any} */ (plain.root),
    box,
    pointer: { x: 600, y: 300 },
  });
  assert.deepEqual([place.wide, place.columns, place.scale], [false, 1, 274 / 400]);
  assert.equal(plain.dataset.columns, undefined);
  assert.equal(plain.classes.has("champion-card--wide"), false);
  assert.ok(place.top >= box.top + 8 && place.top + 274 <= box.top + box.height - 8);
  // A fitted plain card that fits: one column, no scale.
  const fits = measuredCard([260, 200, 150, 120], true, false);
  assert.deepEqual(
    (({ wide, columns, scale }) => [wide, columns, scale])(
      placeChampionCard({
        surface: /** @type {any} */ (fits.surface),
        root: /** @type {any} */ (fits.root),
        box,
        pointer: { x: 600, y: 300 },
      }),
    ),
    [false, 1, 1],
  );
});

// The sizes of a card in one to four columns (262, 540, 818 and 1096 px
// wide) for the given heights.
/** @param {number[]} heights */
const sizesOf = (heights) =>
  heights.map((height, index) => ({ width: 278 * (index + 1) - 16, height }));

test("Grow past arena: championCardLayout grows a Hold Shift card past the box only below 0.85", () => {
  const viewport = { width: 1280, height: 720 };
  assert.equal(CHAMPION_CARD_MIN_INSIDE_SCALE, 0.85);
  // The DevClient at 1280x720: an arena box 227 px tall (211 inside the
  // margins). Inside, three columns (330 px) would need scale 0.64, so the
  // card takes four columns, its shortest, at scale 1, past the box.
  const dc = { width: 966, height: 227 };
  const eight = sizesOf([600, 400, 330, 270]);
  assert.deepEqual(
    championCardLayout({
      box: dc,
      viewport,
      sizes: eight,
      expanded: true,
      hasRows: true,
    }),
    { columns: 4, scale: 1, pastArena: true },
  );
  // 0.85 or more inside: nothing changes, the card stays inside the box
  // (the BattleClient's trap_break start at 1280x720 needs 0.8693).
  const bc = { width: 960, height: 290 };
  const trap = championCardLayout({
    box: bc,
    viewport,
    sizes: sizesOf([500, 360, 315.2, 315.2]),
    expanded: true,
    hasRows: true,
  });
  assert.deepEqual([trap.columns, trap.pastArena], [3, false]);
  assert.equal(trap.scale, 274 / 315.2);
  // Exactly 0.85 stays inside; just below it grows past the box, in the one
  // column this card was measured in.
  const thin = { width: 300, height: 186 };
  /** @param {number} height */
  const one = (height) =>
    championCardLayout({
      box: thin,
      viewport,
      sizes: sizesOf([height]),
      expanded: true,
      hasRows: true,
    });
  assert.deepEqual(one(200), { columns: 1, scale: 0.85, pastArena: false });
  assert.deepEqual(one(201), { columns: 1, scale: 1, pastArena: true });
  // The plain card, a card with no rows and a card with no known window
  // never grow past the box.
  assert.deepEqual(
    championCardLayout({
      box: dc,
      viewport,
      sizes: [eight[0]],
      expanded: false,
      hasRows: true,
    }),
    { columns: 1, scale: 211 / 600, pastArena: false },
  );
  assert.deepEqual(
    championCardLayout({
      box: dc,
      viewport,
      sizes: eight,
      expanded: true,
      hasRows: false,
    }),
    { columns: 1, scale: 1, pastArena: false },
  );
  assert.deepEqual(
    championCardLayout({
      box: dc,
      viewport: null,
      sizes: eight,
      expanded: true,
      hasRows: true,
    }),
    { columns: 3, scale: 211 / 330, pastArena: false },
  );
  // Past the box, a column that does not make the card shorter is not added
  // (six rows take two lines in three or four columns): three columns.
  assert.equal(
    championCardLayout({
      box: dc,
      viewport,
      sizes: sizesOf([500, 380, 300, 300]),
      expanded: true,
      hasRows: true,
    }).columns,
    3,
  );
  // A window too narrow for four columns: three, the shortest that fits it.
  assert.equal(
    championCardLayout({
      box: dc,
      viewport: { width: 1000, height: 720 },
      sizes: eight,
      expanded: true,
      hasRows: true,
    }).columns,
    3,
  );
  // A window too short even for the shortest card: scaled just enough to
  // fit the window less its two 8 px margins.
  assert.deepEqual(
    championCardLayout({
      box: dc,
      viewport: { width: 1280, height: 300 },
      sizes: sizesOf([900, 600, 500, 400]),
      expanded: true,
      hasRows: true,
    }),
    { columns: 4, scale: 284 / 400, pastArena: true },
  );
});

test("Grow past arena: the card sits next to its champion, inside the window, off the champion", () => {
  const viewport = { width: 1280, height: 720 };
  const box = { left: 12, top: 206, width: 966, height: 227 };
  const body = { left: 400, top: 300, right: 430, bottom: 330 };
  const pointer = { x: 415, y: 315 };
  /** @param {{left: number, top: number}} place @param {{width: number, height: number}} size */
  const rectOf = (place, size) => ({
    left: place.left,
    top: place.top,
    right: place.left + size.width,
    bottom: place.top + size.height,
  });
  /** @param {ReturnType<typeof rectOf>} a @param {{left: number, top: number, right: number, bottom: number}} b */
  const covers = (a, b) =>
    a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
  /** @param {ReturnType<typeof rectOf>} rect */
  const onScreen = (rect) =>
    rect.left >= 8 && rect.top >= 8 && rect.right <= 1272 && rect.bottom <= 712;
  // A wide card: 12 px below the champion, held 8 px inside the window's
  // left edge. Below and above are both 12 px away (right and left, pushed
  // back over the champion, cover it); below covers more of the arena, so
  // it wins.
  const wide = { width: 1096, height: 270 };
  const below = championCardPastArenaPosition({
    pointer,
    subject: body,
    box,
    viewport,
    size: wide,
  });
  assert.deepEqual(below, { left: 8, top: 342 });
  assert.ok(onScreen(rectOf(below, wide)) && !covers(rectOf(below, wide), body));
  // Lock In right under the champion: the card goes above it instead.
  const lockIn = { left: 300, top: 450, right: 500, bottom: 500 };
  const above = championCardPastArenaPosition({
    pointer,
    subject: body,
    box,
    viewport,
    size: wide,
    avoid: [lockIn],
  });
  assert.deepEqual(above, { left: 8, top: 18 });
  // A champion low in the window with a tall card: above and below would
  // both cover it, so the card goes beside it (16 px right, its bottom edge
  // level with the champion's), on screen.
  const low = { left: 400, top: 400, right: 430, bottom: 430 };
  const tall = { width: 540, height: 400 };
  const beside = championCardPastArenaPosition({
    pointer: { x: 415, y: 415 },
    subject: low,
    box,
    viewport,
    size: tall,
  });
  assert.deepEqual(beside, { left: 446, top: 30 });
  assert.ok(onScreen(rectOf(beside, tall)) && !covers(rectOf(beside, tall), low));
  // Without a champion rectangle (only the pointer's point) the card never
  // covers the pointer.
  const point = { left: 415, top: 415, right: 415, bottom: 415 };
  const off = rectOf(
    championCardPastArenaPosition({
      pointer: { x: 415, y: 415 },
      subject: point,
      box,
      viewport,
      size: tall,
    }),
    tall,
  );
  assert.ok(
    !(415 >= off.left && 415 <= off.right && 415 >= off.top && 415 <= off.bottom),
  );
});

test("Grow past arena: placeChampionCard places a Hold Shift card past the box and back", () => {
  const viewport = { width: 1280, height: 720 };
  const box = { left: 12, top: 206, width: 966, height: 227 };
  const body = { left: 400, top: 300, right: 430, bottom: 330 };
  /** @param {ReturnType<typeof measuredCard>} card */
  const place = (card) =>
    placeChampionCard({
      surface: /** @type {any} */ (card.surface),
      root: /** @type {any} */ (card.root),
      box,
      pointer: { x: 415, y: 315 },
      subject: body,
      viewport,
    });
  // Eight statuses in the DevClient at 1280x720: four columns, scale 1,
  // past the box, below the champion.
  const eight = measuredCard([600, 400, 330, 270]);
  assert.deepEqual(place(eight), {
    left: 8,
    top: 342,
    wide: true,
    columns: 4,
    scale: 1,
    pastArena: true,
  });
  assert.deepEqual(
    [eight.dataset.columns, eight.dataset.pastArena, eight.root.style.transform],
    ["4", "true", ""],
  );
  // Placed again when it fits the box at 0.85 or more: inside, no mark.
  const fits = measuredCard([600, 400, 240, 240]);
  fits.dataset.pastArena = "true";
  const inside = place(fits);
  assert.deepEqual([inside.columns, inside.pastArena], [3, false]);
  assert.equal(inside.scale, 211 / 240);
  assert.equal(fits.dataset.pastArena, undefined);
  assert.ok(inside.top >= box.top + 8 && inside.top + 211 <= box.top + box.height - 8);
  // The plain card never leaves the box, however small it must be drawn.
  const plain = measuredCard([600, 400, 330, 270], true, false);
  const kept = place(plain);
  assert.deepEqual([kept.columns, kept.pastArena], [1, false]);
  assert.equal(kept.scale, 211 / 600);
});

test("championCardRowsThatFit: every row that fits, else the most with the footer, at least one", () => {
  const rows = [50, 57, 57, 57];
  /** @param {number} available @param {number} [more] */
  const fit = (available, more = 0) =>
    championCardRowsThatFit({ available, frame: 120, footer: 29, rows, more });
  // A tall box: all four rows, no footer needed (120 + 221 = 341).
  assert.equal(fit(600), 4);
  assert.equal(fit(341), 4);
  // One pixel short: a row goes, the footer comes (120 + 29 + 164 = 313).
  assert.equal(fit(340), 3);
  // The 1280x720 BattleClient arena (about 290 px, less two 8 px margins).
  assert.equal(fit(274), 2);
  // Too short for even one row with the footer: still one row.
  assert.equal(fit(150), 1);
  assert.equal(fit(0), 1);
  // Rows already left out by the four-row limit: the footer always counts.
  assert.equal(fit(341, 3), 3);
  assert.equal(fit(370, 3), 4);
  // Rows keep their order: a tall first row is never skipped for short ones.
  assert.equal(
    championCardRowsThatFit({
      available: 215,
      frame: 100,
      footer: 20,
      rows: [100, 10, 10, 10],
    }),
    1,
  );
  // No rows (the reduced card, or no status): none.
  assert.equal(
    championCardRowsThatFit({ available: 100, frame: 300, footer: 29, rows: [] }),
    0,
  );
});

test("rowLimit: a short box's card keeps the order, shows fewer rows, then +N more", () => {
  const four = facts({
    statuses: [
      status("in_combat", 5),
      status("mage_burst", 5, 1.5),
      status("stun_hunter_trap", 4),
      status("priest_freedom", 1, 0.85),
    ],
  });
  const full = championCardModel(four);
  const two = championCardModel(four, { rowLimit: 2 });
  assert.deepEqual(
    two.rows.map((row) => row.id),
    full.rows.slice(0, 2).map((row) => row.id),
  );
  assert.deepEqual([two.rows.length, two.more, two.total], [2, 2, 4]);
  assert.deepEqual(championCardFooter(two), ["+2 more", "Hold Shift for All"]);
  assert.ok(championCardText(two).includes("+2 more · Hold Shift for All"));
  // At least one row, at most the four-row limit; not a whole number: the limit.
  assert.equal(championCardModel(four, { rowLimit: 0 }).rows.length, 1);
  assert.equal(championCardModel(four, { rowLimit: 9 }).rows.length, 4);
  assert.equal(championCardModel(four, { rowLimit: 2.5 }).rows.length, 4);
  // The Hold Shift card ignores it and shows every row.
  assert.equal(championCardModel(four, { rowLimit: 1, expanded: true }).rows.length, 4);
});

/** @param {{canStart?: () => boolean}} [options] */
function shiftHarness(options = {}) {
  const target = new EventTarget();
  /** @type {Array<{callback: () => void, at: number} | null>} */
  const timers = [];
  let now = 0;
  const log = /** @type {string[]} */ ([]);
  const hold = createShiftHold({
    target,
    canStart: options.canStart ?? (() => true),
    onExpand: () => log.push("expand"),
    onShrink: () => log.push("shrink"),
    setTimer: (callback, ms) => timers.push({ callback, at: now + ms }) - 1,
    clearTimer: (id) => {
      timers[/** @type {number} */ (id)] = null;
    },
  });
  /** @param {number} ms */
  const wait = (ms) => {
    now += ms;
    timers.forEach((timer, index) => {
      if (timer !== null && timer.at <= now) {
        timers[index] = null;
        timer.callback();
      }
    });
  };
  // Returns the event, so a test can check that nothing canceled or stopped it.
  /** @param {string} type @param {Record<string, unknown>} [fields] */
  const fire = (type, fields = {}) => {
    const event = Object.assign(new Event(type, { cancelable: true }), {
      key: "",
      repeat: false,
      ctrlKey: false,
      altKey: false,
      metaKey: false,
      shiftKey: false,
      clientX: 100,
      clientY: 100,
      ...fields,
    });
    target.dispatchEvent(event);
    return event;
  };
  return { target, hold, log, wait, fire };
}

test("Hold Shift: Shift held alone for 300 ms expands; release shrinks at once", () => {
  assert.equal(CHAMPION_CARD_HOLD_MS, 300);
  const { hold, log, wait, fire } = shiftHarness();
  fire("pointermove", { clientX: 100, clientY: 100 });
  fire("keydown", { key: "Shift", shiftKey: true });
  wait(299);
  assert.deepEqual([hold.expanded(), log], [false, []]);
  // A key repeat neither restarts nor ends the wait.
  fire("keydown", { key: "Shift", shiftKey: true, repeat: true });
  wait(1);
  assert.deepEqual([hold.expanded(), log], [true, ["expand"]]);
  // The browser's layout-change move at the same spot is not movement.
  fire("pointermove", { clientX: 100, clientY: 100 });
  assert.equal(hold.expanded(), true);
  fire("keyup", { key: "Shift" });
  assert.deepEqual([hold.expanded(), log], [false, ["expand", "shrink"]]);
  // Shift with Ctrl, Alt or Meta never starts the wait.
  for (const modifier of ["ctrlKey", "altKey", "metaKey"]) {
    fire("keydown", { key: "Shift", [modifier]: true });
    wait(400);
    assert.equal(hold.expanded(), false, modifier);
  }
  // Nothing expands while no card that can expand is showing.
  const idle = shiftHarness({ canStart: () => false });
  idle.fire("keydown", { key: "Shift", shiftKey: true });
  idle.wait(400);
  assert.deepEqual([idle.hold.expanded(), idle.log], [false, []]);
  // The card going away while Shift is down cancels quietly.
  const gone = shiftHarness();
  gone.fire("keydown", { key: "Shift", shiftKey: true });
  gone.hold.cancel();
  gone.wait(400);
  assert.deepEqual([gone.hold.expanded(), gone.log], [false, []]);
});

test("every Shift combination keeps its job and never leaves the card expanded", () => {
  // Each combination: the second event after Shift, as the browser sends it.
  /** @type {Array<[string, string, Record<string, unknown>]>} */
  const combinations = [
    ["Shift+click (target a body)", "pointerdown", { shiftKey: true, button: 0 }],
    ["Shift+Tab (previous champion)", "keydown", { key: "Tab", shiftKey: true }],
    [
      "Shift+Arrow Left (10 ticks back)",
      "keydown",
      { key: "ArrowLeft", shiftKey: true },
    ],
    [
      "Shift+Arrow Right (10 ticks on)",
      "keydown",
      { key: "ArrowRight", shiftKey: true },
    ],
    ["? (How to Play)", "keydown", { key: "?", shiftKey: true }],
    ["Shift+R", "keydown", { key: "R", shiftKey: true }],
    ["any pointer movement", "pointermove", { shiftKey: true, clientX: 101 }],
    ["the wheel", "wheel", { shiftKey: true }],
  ];
  for (const [name, type, fields] of combinations) {
    // Quick: the second event comes before 300 ms, so the card never expands.
    const quick = shiftHarness();
    /** @type {Event[]} */
    const seen = [];
    quick.target.addEventListener(type, (event) => seen.push(event));
    quick.fire("pointermove");
    quick.fire("keydown", { key: "Shift", shiftKey: true });
    quick.wait(120);
    const event = quick.fire(type, fields);
    quick.wait(400);
    assert.deepEqual([quick.hold.expanded(), quick.log], [false, []], name);
    // The page's own handler still gets the event, uncancelled.
    assert.ok(seen.includes(event), name);
    assert.equal(event.defaultPrevented, false, name);
    // Slow: the card has expanded, and the second event shrinks it at once.
    const slow = shiftHarness();
    slow.fire("pointermove");
    slow.fire("keydown", { key: "Shift", shiftKey: true });
    slow.wait(300);
    assert.equal(slow.hold.expanded(), true, name);
    const late = slow.fire(type, fields);
    assert.deepEqual(
      [slow.hold.expanded(), slow.log],
      [false, ["expand", "shrink"]],
      name,
    );
    assert.equal(late.defaultPrevented, false, name);
  }
  // Leaving the window shrinks it; an element's own blur does not.
  const left = shiftHarness();
  left.fire("keydown", { key: "Shift", shiftKey: true });
  left.wait(300);
  const field = new EventTarget();
  left.target.addEventListener("blur", () => {}, true);
  field.dispatchEvent(new Event("blur"));
  assert.equal(left.hold.expanded(), true);
  left.fire("blur");
  assert.equal(left.hold.expanded(), false);
  // destroy removes every listener.
  const done = shiftHarness();
  done.hold.destroy();
  done.fire("keydown", { key: "Shift", shiftKey: true });
  done.wait(400);
  assert.deepEqual(done.log, []);
});
