/**
 * @file Check the BattleClient deck's words and drawings against the approved
 * vision page (src/tactics-turn.js): the refusal sentences written from the
 * host's masks before anything is sent (needs a target first, not allowed on
 * this target, Burst not allowed with its ready time, a policy's champion,
 * nobody is yours, a fresh start is needed, a fallen champion, and the
 * silent refusal while a turn is in flight; Lock In is never refused while
 * the policies are not choosing); the status-line announcements of accepted
 * inputs (Step, No Combat, armed lanes, a target pick with Auto, Clear,
 * Orders Sent, Playing One Turn); the This Turn titles and the Lock In
 * labels for every phase; the Class Guide written from the host's real
 * class_mechanics, and its card markup (classGuideCardHtml, the one owner of
 * the card's words for the live rail and the Replay Viewer's Agent tab:
 * escaped words, the vision's icon or the glyph fallback, Ultimate
 * readiness, the caller's extra lines, and the short card for an unknown
 * class); reach formatting; and the export-safe token, crest and
 * Rune Ring SVG (flat fills, strokes, opacity only), painted through the
 * .tc-… palette classes of src/styles/battleclient/base.css.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  classGuideCardHtml,
  classGuideView,
  crestSvg,
  deckAnnouncement,
  deckRefusal,
  reachText,
  ticksText,
  tokenSvg,
  turnTitles,
} from "../src/tactics-turn.js";

const PLAN = Object.freeze({
  intent: /** @type {{kind: "lane", lane: 0 | 1 | 2}} */ ({ kind: "lane", lane: 1 }),
  stopped: false,
  terminal: false,
  phase: /** @type {const} */ ("plan"),
  watchMode: false,
  focusedManual: true,
  fallen: false,
  moveAllowed: true,
  classId: 2,
  ultimateNeedsNoTarget: false,
  hasTarget: true,
  basicOk: false,
  ultimateOk: true,
  cooldown: 0,
});

const WARRIOR = Object.freeze({
  class_id: 2,
  maximum_health: 200,
  base_movement_speed: 1,
  observation_radius: 6,
  basic_target_mode: "enemy",
  basic_interaction_radius: 1.5,
  basic_raw_damage: 8,
  basic_raw_healing: 0,
  ultimate_target_mode: "enemy",
  ultimate_interaction_radius: 5.75,
  ultimate_cooldown_steps: 30,
  ultimate_raw_damage: 20,
  ultimate_raw_healing: 0,
  status_mechanics: [
    {
      family: "slow",
      source_action_component: "ultimate",
      duration_steps: 5,
      magnitude: 0.5,
    },
    {
      family: "stun",
      source_action_component: "ultimate",
      duration_steps: 1,
      magnitude: null,
    },
  ],
});

test("refusals use the vision's sentences and are decided before sending", () => {
  assert.deepEqual(deckRefusal(PLAN), {
    refused: true,
    text: "Shield Bash is not allowed on this target right now. Your order is unchanged.",
  });
  assert.equal(
    deckRefusal({ ...PLAN, hasTarget: false }).text,
    "Shield Bash needs a target first. Your order is unchanged.",
  );
  assert.equal(
    deckRefusal({ ...PLAN, intent: { kind: "lane", lane: 2 }, hasTarget: false }).text,
    "Charge needs a target first. Your order is unchanged.",
  );
  assert.deepEqual(deckRefusal({ ...PLAN, intent: { kind: "lane", lane: 2 } }), {
    refused: false,
    text: null,
  });
  assert.equal(
    deckRefusal({
      ...PLAN,
      intent: { kind: "lane", lane: 2 },
      ultimateOk: false,
      cooldown: 4,
    }).text,
    "Charge is not allowed on this target right now. Your order is unchanged. It is ready in 4 ticks.",
  );
  assert.equal(
    deckRefusal({
      ...PLAN,
      classId: 1,
      intent: { kind: "lane", lane: 2 },
      ultimateNeedsNoTarget: true,
      ultimateOk: false,
      cooldown: 30,
    }).text,
    "Burst is not allowed right now. Your order is unchanged. It is ready in 30 ticks.",
  );
  assert.equal(
    deckRefusal({ ...PLAN, focusedManual: false }).text,
    "This champion is played by a policy. You can look, not command. Pick one of yours to give orders.",
  );
  assert.equal(
    deckRefusal({ ...PLAN, focusedManual: false, watchMode: true }).text,
    "Nobody is yours this game. You can look, not command.",
  );
  assert.equal(
    deckRefusal({ ...PLAN, stopped: true }).text,
    "A fresh start is needed, so orders cannot change. You can look around, save or exit.",
  );
  assert.equal(
    deckRefusal({ ...PLAN, fallen: true }).text,
    "A fallen champion cannot act until it returns.",
  );
  assert.deepEqual(deckRefusal({ ...PLAN, phase: "choosing" }), {
    refused: true,
    text: null,
  });
  assert.equal(
    deckRefusal({
      ...PLAN,
      intent: { kind: "step", moveAction: 1 },
      moveAllowed: false,
    }).text,
    "Step North is not allowed right now. Your order is unchanged.",
  );
  assert.equal(
    deckRefusal({ ...PLAN, intent: { kind: "lock" }, phase: "act" }).refused,
    false,
  );
  assert.equal(
    deckRefusal({ ...PLAN, intent: { kind: "lock" }, focusedManual: false }).refused,
    false,
  );
});

test("accepted inputs announce themselves in the vision's words", () => {
  const base = {
    classId: 2,
    ultimateNeedsNoTarget: false,
    targetName: "Mage (ID 5)",
    watchMode: false,
  };
  /** @param {any} intent */
  const say = (intent, extra = {}) => deckAnnouncement({ ...base, ...extra, intent });
  assert.equal(say({ kind: "step", moveAction: 6 }), "Step: North-West.");
  assert.equal(say({ kind: "step", moveAction: 0 }), "Step: Stay.");
  assert.equal(
    say({ kind: "lane", lane: 0 }),
    "No Combat: your champion only moves this tick. Your target stays picked.",
  );
  assert.equal(say({ kind: "lane", lane: 2 }), "Charge is armed on the Mage (ID 5).");
  assert.equal(
    say({ kind: "lane", lane: 2 }, { classId: 1, ultimateNeedsNoTarget: true }),
    "Burst is armed. It has no target, so your target was cleared.",
  );
  assert.equal(
    say({
      kind: "target",
      basicOk: false,
      ultimateOk: true,
      targetName: "Mage (ID 5)",
    }),
    "Target: Mage (ID 5). Shield Bash is not allowed on it, so your action is No Combat. Press 2 for Charge.",
  );
  assert.equal(
    say({
      kind: "target",
      basicOk: false,
      ultimateOk: false,
      targetName: "Hunter (ID 7)",
    }),
    "Target: Hunter (ID 7). Nothing is allowed on it right now, so your action is No Combat.",
  );
  assert.equal(
    say(
      { kind: "target", basicOk: true, ultimateOk: false, targetName: "Mage (ID 0)" },
      { classId: 1, ultimateNeedsNoTarget: true },
    ),
    "Target: Mage (ID 0). Arcane Blast armed itself, because it is allowed (Auto).",
  );
  assert.equal(
    say({ kind: "clear", viaKey: true, editable: true }),
    "Target cleared. Keys are off until you click the arena again.",
  );
  assert.equal(
    say({ kind: "clear", viaKey: false, editable: true }),
    "Target cleared.",
  );
  assert.equal(
    say({ kind: "lock" }),
    "Waiting for all players to submit their actions…",
  );
  assert.equal(
    say({ kind: "lock" }, { watchMode: true }),
    "Waiting for all players to submit their actions…",
  );
});

test("titles and Lock In labels follow the phase", () => {
  const titles = (/** @type {any} */ phase, extra = {}) =>
    turnTitles({
      phase,
      stopped: false,
      terminal: false,
      watchMode: false,
      running: false,
      ...extra,
    });
  assert.deepEqual(titles("plan"), { title: "Your Turn", lockLabel: "Lock In Turn" });
  assert.deepEqual(titles("sent"), { title: "Orders Sent", lockLabel: "Orders Sent" });
  assert.deepEqual(titles("choosing"), {
    title: "Policies Choosing",
    lockLabel: "Policies Choosing…",
  });
  assert.deepEqual(titles("act"), {
    title: "Everyone Acts",
    lockLabel: "Everyone Acts",
  });
  assert.equal(titles("plan", { stopped: true }).title, "Game Stopped");
  assert.equal(titles("plan", { watchMode: true }).title, "Watching · Paused");
  assert.equal(
    titles("plan", { watchMode: true, running: true }).title,
    "Watching · Running",
  );
});

test("the Class Guide is written from the host's class mechanics", () => {
  const guide = classGuideView(2, WARRIOR);
  assert.equal(guide?.name, "Warrior");
  assert.equal(guide?.calling, "The Undying");
  assert.equal(
    guide?.stats,
    "Life 200 · Speed 1.0 · Sight 6.0 · Ultimate every 30 ticks",
  );
  assert.equal(guide?.basicText, "8 damage to one enemy within 1.5.");
  assert.equal(
    guide?.ultimateText,
    "Rush in one straight line at a visible enemy within 5.75: 20 damage, a 1-tick stun and half speed for 5 ticks. No knockback. Your own step still runs after you land.",
  );
  const changed = classGuideView(2, { ...WARRIOR, basic_raw_damage: 9 });
  assert.equal(changed?.basicText, "9 damage to one enemy within 1.5.");
  assert.equal(classGuideView(2, null)?.basicText, null);
  assert.equal(classGuideView(9, WARRIOR), null);
  assert.equal(reachText(5.75), "5.75");
  assert.equal(reachText(3), "3.0");
  assert.equal(reachText(1.2999999523162842), "1.3");
  assert.equal(reachText("x"), null);
  assert.equal(ticksText(1), "1 tick");
  assert.equal(ticksText(30, true), "30 Ticks");
});

test("the Class Guide card markup is one owner for both clients", () => {
  const page = {
    /** @param {string} id */
    getElementById: (id) => (id === "ab-bash" || id === "ab-charge" ? {} : null),
  };
  const card = classGuideCardHtml(/** @type {any} */ (page), 2, WARRIOR, {
    cooldown: 3,
    basicNow: '<span class="now ok">Allowed on: Mage 5</span>',
    note: '<p class="stats cg-note">Played by tdm-beta.</p>',
  });
  assert.match(
    card,
    /^<p class="eyebrow">Class Guide<\/p><h3>Warrior<span>The Undying<\/span><\/h3>/u,
  );
  assert.match(card, /<use href="#ab-bash"\/>/u);
  assert.match(card, /<b>Charge<\/b><em>Ultimate · Ready In 3 Ticks<\/em>/u);
  assert.match(
    card,
    /8 damage to one enemy within 1\.5\.<\/p><span class="now ok">Allowed on: Mage 5<\/span>/u,
  );
  assert.ok(card.endsWith('<p class="stats cg-note">Played by tdm-beta.</p>'));
  const ready = classGuideCardHtml(/** @type {any} */ (page), 2, WARRIOR, {
    cooldown: 0,
  });
  assert.match(ready, /<em>Ultimate · Ready<\/em>/u);
  assert.match(
    classGuideCardHtml(/** @type {any} */ (page), 2, WARRIOR),
    /<em>Ultimate<\/em>/u,
  );
  assert.equal(
    classGuideCardHtml(/** @type {any} */ (page), 9, WARRIOR),
    '<p class="eyebrow">Class Guide</p><p class="stats">Pick a champion to read its guide.</p>',
  );
});

test("tokens and crests are export-safe SVG", async () => {
  const drawings = [
    tokenSvg({
      classKey: "warrior",
      team: "A",
      size: "M",
      lit: 12,
      ready: true,
      glyph:
        '<use href="#sig-warrior" x="13.32" y="13.32" width="19.36" height="19.36"/>',
    }),
    tokenSvg({ classKey: "mage", team: "B", size: "XS", glyph: "" }),
    crestSvg("A", 14),
    crestSvg("B", 16),
  ];
  for (const svg of drawings) {
    assert.doesNotMatch(svg, /Gradient|<mask|<filter|<image|url\(|style=/u);
  }
  assert.equal((drawings[0].match(/ L/gu) ?? []).length >= 12, true);
  assert.match(drawings[1], /class="tc-b" stroke="currentColor"/u);
  assert.match(drawings[2], /class="tc-a" stroke="currentColor"/u);
  const base = await readFile(
    new URL("../src/styles/battleclient/base.css", import.meta.url),
    "utf8",
  );
  for (const [paint, token] of [
    ["tc-a", "--team-a"],
    ["tc-b", "--team-b"],
    ["tc-mage", "--class-mage"],
    ["tc-warrior", "--class-warrior"],
    ["tc-hunter", "--class-hunter"],
    ["tc-rogue", "--class-rogue"],
    ["tc-priest", "--class-priest"],
  ]) {
    assert.match(base, new RegExp(`\\.${paint} \\{\\s*color: var\\(${token}\\);`, "u"));
  }
  assert.match(drawings[3], /stroke="#0A0D14"/u);
});
