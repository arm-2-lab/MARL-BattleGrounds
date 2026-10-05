/**
 * @file Check the BattleClient's top bar, arena view tag, hover card, How To
 * Play rule cards and Visual Filters drawer list against the approved vision
 * page and the owner's 1 October 2026 decisions.
 *
 * Contracts: the place line splits into a bold map name and "Map 47 · Team
 * Deathmatch" (a scenario shows its tale title; matchPlaceTitle keeps its old
 * words); the view tag reads "Oracle View · You See Everything" or "Agent POV ·
 * Your Team's Sight · N Unseen" ("Team A's" when nobody plays by hand), with N
 * counted from the Agent POV scene's own spawn pads and bodies, never from the
 * researcher view; the hover card is the Champion Card (design C, owner, 2
 * October 2026: name and calling, Agent ID, Life and Ultimate bars, one
 * two-line row per status, who plays it, "Your Target"), an unseen champion's
 * card names no life, place, Ultimate or status but always its name and
 * calling (from the identity directory, so a Turn Checklist row without
 * data-class-id still reads "Mage · The Astral Walker"), and the card flips left or up
 * near the box edges, stays 8 px inside the box and never covers Lock In; the facts
 * come from the audience scene (an enemy portrait in Agent POV gives the
 * reduced card, a seen champion the full card) and a seen champion's card is
 * the same in Agent POV and Oracle View, also in the Replay Viewer, where who
 * plays it is the team's name; the How To Play overlay has the vision's
 * four rule titles; the live Visual Filters list holds the V5 add-ons too
 * (Class Seals to Momentum, as in the Replay Viewer), lists Lore Emblems
 * (Bodies) in both clients, shows the V5
 * names while the Replay Viewer keeps its own, and lists Step Ghost in live
 * play only.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { authorizedPresentationSceneView } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  championCardFacts,
  HELP_OVERLAY_RULES,
  hasManualChampion,
  liveViewTag,
  matchPlaceParts,
  matchPlaceTitle,
} from "../src/shell.js";
import {
  championCardModel,
  championCardText,
  hoverCardPosition,
} from "../src/tooltip.js";
import { visualFiltersForProduct } from "../src/visual-filters.js";

const FIXTURE = new URL("./fixtures/authorized-presentations-v1.json", import.meta.url);

/** @param {string} name */
async function pairOf(name) {
  const fixture = JSON.parse(await readFile(FIXTURE, "utf8"));
  const pair = fixture.pairs[name];
  return {
    presentation: await normalizeAuthorizedPresentationFrameV1(
      structuredClone(pair.presentation),
    ),
    combatConfiguration: pair.transport.combat_configuration,
  };
}

/** @param {Partial<import("../src/tooltip.js").HoverCardSubject>} subject */
function subjectOf(subject) {
  return {
    presentationKey: null,
    slot: null,
    publicAgentId: null,
    classId: null,
    ...subject,
  };
}

test("the place line is a bold name, then the map number and the task", () => {
  const map = { map_id: 47, display_name: "Teutoburg Forest", technical_name: "m47" };
  assert.deepEqual(matchPlaceParts({ map, task_mode: 1 }), {
    name: "Teutoburg Forest",
    details: ["Map 47", "Team Deathmatch"],
  });
  assert.equal(matchPlaceTitle({ map, task_mode: 1 }), "Teutoburg Forest · Map 47");
  assert.deepEqual(
    matchPlaceParts({
      map: { map_id: null, display_name: "scenario_1_HRP_vs_MRP" },
      task_mode: 1,
    }),
    { name: "The Last Mending", details: ["Scenario 1", "Team Deathmatch"] },
  );
  assert.deepEqual(
    matchPlaceParts({ map: { map_id: null, display_name: "Custom Map" } }),
    {
      name: "Custom Map",
      details: [],
    },
  );
  assert.equal(matchPlaceParts({ map: null }), null);
});

test("the view tag says what each view shows, from the audience scene only", async () => {
  const pov = await pairOf("live_shared_obs_agent_pov");
  const scene = authorizedPresentationSceneView(pov.presentation);
  const input = {
    product: "combat_debugger",
    audience: "agent_pov",
    scene,
    recipientPublicAgentId: "0",
    watchOnly: false,
  };
  assert.deepEqual(liveViewTag(input), {
    view: "pov",
    lead: "Agent POV",
    detail: "· Your Team's Sight · 5 Unseen",
  });
  assert.equal(
    liveViewTag({ ...input, watchOnly: true })?.detail,
    "· Team A's Sight · 5 Unseen",
  );
  assert.deepEqual(liveViewTag({ ...input, audience: "researcher" }), {
    view: "oracle",
    lead: "Oracle View",
    detail: "· You See Everything",
  });
  assert.equal(liveViewTag({ ...input, product: "replay_viewer" }), null);
  assert.equal(liveViewTag({ ...input, audience: "unavailable" }), null);
  assert.equal(hasManualChampion(pov.combatConfiguration), true);
  assert.equal(hasManualChampion({ manual_slots: [] }), false);
  assert.equal(hasManualChampion({ team_a_controller: "manual" }), true);
  assert.equal(hasManualChampion(null), false);
});

test("the hover card is the Champion Card, in the vision's order", () => {
  const seen = championCardModel({
    state: "seen",
    name: "Mage",
    calling: "The Astral Walker",
    classId: 1,
    displayId: "5",
    teamId: 2,
    life: { current: 77.04, maximum: 80 },
    ultimate: { name: "Burst", remaining: 30, full: 30 },
    statuses: [
      { id: "mage_burst", ticks: 5, full: 5, amount: 1.5 },
      { id: "priest_freedom", ticks: 1, full: 5, amount: 0.85 },
    ],
    note: null,
    you: false,
    playedBy: "tdm-beta",
    yourTarget: true,
    unseenBy: null,
  });
  assert.deepEqual(championCardText(seen), [
    "Mage · The Astral Walker",
    "Agent ID 5",
    "Life 77 / 80",
    "Burst 30 ticks",
    "Blessing of Freedom (Slow Resistance) · 1 tick left · Slows can't drop it below 85% speed. From a Priest.",
    "Burst (Damage Boost) · 5 ticks left · Its attacks deal ×1.5 damage. From its own Ultimate.",
    "Played by tdm-beta",
    "Your Target",
  ]);
  assert.equal(seen.teamId, 2);
  const unseen = championCardModel({
    state: "unseen",
    name: "Warrior",
    calling: "The Undying",
    classId: 2,
    displayId: "6",
    teamId: 2,
    life: null,
    ultimate: null,
    statuses: [],
    note: null,
    you: false,
    playedBy: "tdm-beta",
    yourTarget: false,
    unseenBy: null,
  });
  assert.deepEqual(championCardText(unseen), [
    "Warrior · The Undying",
    "Agent ID 6",
    "Unseen · none of your team can see it now.",
    "Played by tdm-beta",
  ]);
  const size = { width: 262, height: 120 };
  assert.deepEqual(
    hoverCardPosition({ x: 100, y: 100 }, { width: 1440, height: 900 }, size),
    { left: 116, top: 116 },
  );
  assert.deepEqual(
    hoverCardPosition({ x: 1300, y: 850 }, { width: 1440, height: 900 }, size),
    { left: 1022, top: 718 },
  );
  // A card taller than the room below and above stays inside the box: it
  // flips up and is held 8 px below the top edge.
  assert.deepEqual(
    hoverCardPosition(
      { x: 100, y: 450 },
      { width: 1440, height: 900 },
      {
        width: 262,
        height: 600,
      },
    ),
    { left: 116, top: 8 },
  );
  // The card never covers Lock In: a card that would sit on it moves to
  // another corner around the pointer.
  const lockIn = { left: 1100, top: 760, right: 1380, bottom: 840 };
  const place = hoverCardPosition(
    { x: 1000, y: 700 },
    { width: 1440, height: 900 },
    size,
    [lockIn],
  );
  assert.deepEqual(place, { left: 722, top: 716 });
  assert.ok(
    place.left + size.width <= lockIn.left ||
      place.top + size.height <= lockIn.top ||
      place.left >= lockIn.right ||
      place.top >= lockIn.bottom,
  );
});

test("hover card facts come from the audience scene and hide what Agent POV cannot see", async () => {
  const pov = await pairOf("live_shared_obs_agent_pov");
  const own = championCardFacts({ ...pov, subject: subjectOf({ slot: 1 }) });
  assert.equal(own?.state, "seen");
  assert.equal(own?.name, "Warrior");
  assert.equal(own?.displayId, "1");
  assert.deepEqual(own?.life, { current: 200, maximum: 200 });
  assert.equal(own?.you, true);
  const enemy = championCardFacts({
    ...pov,
    subject: subjectOf({ slot: 5, classId: 1 }),
  });
  assert.equal(enemy?.state, "unseen");
  assert.equal(enemy?.displayId, "5");
  assert.equal(enemy?.name, "Mage");
  assert.equal(enemy?.life, null);
  assert.equal(enemy?.ultimate, null);
  assert.deepEqual(enemy?.statuses, []);
  // A Turn Checklist row carries only its slot (no data-class-id): the class
  // is public, so the identity directory still names the champion.
  const row = championCardFacts({ ...pov, subject: subjectOf({ slot: 5 }) });
  assert.equal(row?.state, "unseen");
  assert.equal(row?.name, "Mage");
  assert.equal(row?.calling, "The Astral Walker");
  assert.equal(
    championCardModel(/** @type {any} */ (row)).title,
    "Mage · The Astral Walker",
  );
  // An enemy named only by a public ID that has no body gives no card at all.
  assert.equal(
    championCardFacts({ ...pov, subject: subjectOf({ publicAgentId: "7" }) }),
    null,
  );
  const oracle = await pairOf("live_oracle");
  assert.equal(
    championCardFacts({ ...oracle, subject: subjectOf({ slot: 7 }) })?.state,
    "seen",
  );
  assert.equal(
    championCardFacts({
      presentation: null,
      combatConfiguration: null,
      subject: subjectOf({ slot: 1 }),
    }),
    null,
  );
});

test("a seen champion's card is the same in Agent POV and Oracle View, live and in replays", async () => {
  for (const [povName, oracleName] of [
    ["live_shared_obs_agent_pov", "live_oracle"],
    ["replay_shared_obs_agent_pov", "replay_oracle"],
  ]) {
    const pov = await pairOf(povName);
    const oracle = await pairOf(oracleName);
    for (const slot of [0, 1]) {
      const subject = subjectOf({ slot });
      const povFacts = championCardFacts({ ...pov, subject });
      assert.equal(povFacts?.state, "seen", `${povName} ${slot}`);
      assert.deepEqual(
        championCardFacts({ ...oracle, subject }),
        povFacts,
        `${povName} ${slot}`,
      );
      // Sources are named by class only, never by an Agent ID.
      for (const line of championCardText(championCardModel(povFacts))) {
        assert.doesNotMatch(line, /hidden source|ID \d's/u);
      }
    }
  }
  const replay = await pairOf("replay_oracle");
  // The Replay Viewer names the team's player and has no Manual champion.
  const facts = championCardFacts({ ...replay, subject: subjectOf({ slot: 0 }) });
  assert.equal(facts?.you, false);
  assert.equal(
    facts?.playedBy,
    replay.presentation.match_summary.teams[0].display_name,
  );
  // An unseen champion has no body in a replay's Agent POV scene: no card.
  const replayPov = await pairOf("replay_shared_obs_agent_pov");
  assert.equal(
    championCardFacts({ ...replayPov, subject: subjectOf({ slot: 7 }) }),
    null,
  );
});

test("How To Play shows the vision's four rule cards", () => {
  assert.deepEqual(
    HELP_OVERLAY_RULES.map(([title]) => title),
    [
      "1 · Everyone Decides At Once",
      "2 · One Step, One Action",
      "3 · No Facing",
      "4 · Points And Winning",
    ],
  );
});

test("the live Visual Filters list shows the V5 add-ons, under their V5 names", () => {
  const live = visualFiltersForProduct("combat_debugger");
  const replay = visualFiltersForProduct("replay_viewer");
  const liveIds = new Set(live.map(({ id }) => id));
  for (const addOn of [
    "class_seals",
    "aura_glints",
    "barrier_held",
    "saved_beat",
    "last_breath",
    "battle_text_detail",
    "kill_credit_glint",
    "champion_streaks",
    "hud_momentum",
  ]) {
    assert.equal(liveIds.has(addOn), true, addOn);
    assert.equal(
      replay.some(({ id }) => id === addOn),
      true,
      addOn,
    );
  }
  /** @param {ReadonlyArray<Record<string, any>>} list @param {string} id */
  const named = (list, id) => {
    const row = list.find((entry) => entry.id === id);
    return row === undefined ? null : [row.label, row.group];
  };
  assert.deepEqual(named(live, "death_announcer"), [
    "Death Announcer",
    "Life And Death",
  ]);
  assert.deepEqual(named(replay, "death_announcer"), ["Fallen Feed", "Life And Death"]);
  assert.deepEqual(named(live, "lore_emblems"), ["Lore Emblems", "Bodies"]);
  assert.deepEqual(named(replay, "lore_emblems"), ["Lore Emblems", "Bodies"]);
  assert.deepEqual(named(live, "pov_sight_veil"), ["POV Sight Veil", "HUD"]);
  assert.deepEqual(named(live, "last_move_ghosts"), ["Last-Move Ghosts", "Motion"]);
  assert.deepEqual(named(live, "focus_fire"), ["Focus Fire", "Diagnostics"]);
  assert.deepEqual(named(live, "symbol_explanations"), [
    "Symbol Explanations",
    "Panels",
  ]);
  assert.deepEqual(named(live, "step_ghost"), ["Step Ghost", "Motion"]);
  assert.equal(named(replay, "step_ghost"), null);
  assert.equal(live.every(Object.isFrozen), true);
});
