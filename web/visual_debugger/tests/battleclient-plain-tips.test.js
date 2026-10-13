/**
 * @file Checks that stone tooltips use the design store's plain words in every
 * client (BattleClient, DevClient combat screen and Replay Viewer): the Symbol
 * Tip title is "Pillar", "Wall" or "Thin Wall" (createSymbolDescriptor's title
 * option; a blank title keeps the symbol's own), Obstacle ID is a plain number
 * taken from the recorded obstacle slot, so the Agent POV ID
 * "base-sensor-obstacle-5" and the Oracle View ID "obstacle-5" give the same
 * card, the screen-reader label is "Pillar 5" (a stone whose ID ends in no
 * number takes its place in the list), lengths read "1 unit" or "0.5 units",
 * the rule ends "Bodies never block sight.", a thin wall adds "True width 0.01
 * units. Drawn thicker so you can see it.", and no internal ID or "Type" row
 * reaches the card. Also checks the other arena tips that `explanations.js`
 * writes use plain words, with no "transition", "activation", "NET",
 * "Authorized", "Legality" or "True"/"False": the fall card's tip
 * (explainDeathAnnouncement: its credit sentence and "This champion fell."),
 * the planned order's line ("Planned Basic", PLANNED_ROUTE_SENTENCE), a net
 * life change ("Net Damage", NET_LIFE_SENTENCE, "Life Change"), a blow without
 * its ability guide ("The Priest used Lesser Heal this tick.", and no class
 * for a redacted source) and the Basic or Ultimate availability ("Allowed Now":
 * "Yes" or "No").
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import {
  DEATH_CREDIT_SENTENCE,
  explainActivation,
  explainDeathAnnouncement,
  explainLegality,
  explainNetHealth,
  explainPendingRoute,
  NET_LIFE_SENTENCE,
  PLANNED_ROUTE_SENTENCE,
} from "../src/explanations.js";
import * as obstacles from "../src/looks/obstacles.js";
import { createSymbolDescriptor } from "../src/tooltip.js";

test("a symbol tip can carry a plainer title; a blank one keeps the symbol's", () => {
  assert.equal(
    createSymbolDescriptor({ symbol: "stone", title: "Pillar" }).title,
    "Pillar",
  );
  assert.equal(createSymbolDescriptor({ symbol: "stone", title: "  " }).title, "Stone");
  assert.equal(createSymbolDescriptor({ symbol: "stone" }).title, "Stone");
});

/** @param {string} prefix */
function map(prefix) {
  return {
    width: 20,
    height: 10,
    obstacles: [
      {
        obstacle_id: `${prefix}5`,
        kind: "pillar",
        center: [10, 0.5],
        radius: 0.5,
        theta: 0,
      },
      {
        obstacle_id: `${prefix}14`,
        kind: "wall",
        center: [8, 5],
        width: 1,
        height: 1.5,
        theta: 0,
      },
      {
        obstacle_id: `${prefix}20`,
        kind: "wall",
        center: [2, 7.9],
        width: 0.01,
        height: 2,
        theta: 0,
      },
      { obstacle_id: "", kind: "pillar", center: [4, 2.5], radius: 0.75, theta: 0 },
    ],
  };
}

const transform = createViewportTransform({
  worldWidth: 20,
  worldHeight: 10,
  viewportWidth: 1440,
  viewportHeight: 900,
  padding: 24,
});

/** @param {string} prefix */
function cards(prefix) {
  return obstacles
    .records({ map: map(prefix), transform })
    .map((record) => ({ record, tip: /** @type {any} */ (obstacles.tooltip(record)) }))
    .filter((entry) => entry.tip !== null)
    .map(({ record, tip }) => ({
      label: record.label,
      title: tip.title,
      summary: tip.summary,
      rows: tip.rows.map((/** @type {any} */ row) => `${row.label}: ${row.value}`),
    }));
}

test("stones use the store's plain names and the same card in every view", () => {
  const pov = cards("base-sensor-obstacle-");
  assert.deepEqual(
    pov,
    cards("obstacle-"),
    "Agent POV and Oracle View IDs give one card",
  );
  const text = JSON.stringify(pov);
  assert.doesNotMatch(text, /obstacle-|sensor|Type/u);
  const [thin, pillar, wall, unnumbered] = pov;
  assert.equal(pillar.title, "Pillar");
  assert.equal(pillar.label, "Pillar 5");
  assert.deepEqual(pillar.rows.slice(0, 3), [
    "Obstacle ID: 5",
    "Radius: 0.5 units",
    "Center: (10, 0.5)",
  ]);
  assert.equal(
    pillar.summary,
    "Blocks movement and sight. A Warrior's Charge can pass over it. Bodies never block sight.",
  );
  assert.equal(wall.title, "Wall");
  assert.deepEqual(wall.rows.slice(0, 5), [
    "Obstacle ID: 14",
    "Width: 1 unit",
    "Height: 1.5 units",
    "Turned: 0°",
    "Center: (8, 5)",
  ]);
  assert.equal(thin.title, "Thin Wall");
  assert.equal(thin.label, "Thin Wall 20");
  assert.ok(thin.rows.includes("Length: 2 units"));
  assert.equal(
    thin.rows.at(-1),
    "Note: True width 0.01 units. Drawn thicker so you can see it.",
  );
  assert.equal(
    unnumbered.label,
    "Pillar 3",
    "no number in the ID: its place in the list",
  );
});

test("the fall card's tip uses plain words", () => {
  const tip = explainDeathAnnouncement({
    eventId: "fall-98",
    label: "Fallen on Tick 98",
    members: [
      {
        title: "Priest · ID 4",
        killingTeamId: 1,
        contributors: [{ title: "Rogue · ID 8" }],
      },
    ],
  });
  assert.equal(tip.title, "Fallen on Tick 98");
  assert.equal(tip.summary, DEATH_CREDIT_SENTENCE);
  assert.equal(
    DEATH_CREDIT_SENTENCE,
    "Credit is shared by every champion that hit it on that tick, and by a Priest whose healing on that tick helped one of them. No one is named as the last hit.",
  );
  assert.equal(tip.sections[0].summary, "This champion fell.");
  assert.doesNotMatch(JSON.stringify(tip), /transition|agent|hitter/iu);
});

const INTERNAL =
  /[Tt]ransition|[Aa]ctivation|\bNET\b|[Aa]uthori[sz]ed|[Ll]egality|\bTrue\b|\bFalse\b/u;

test("route, net life, blow and availability tips use plain words", () => {
  const route = explainPendingRoute({ lane: 0 });
  assert.equal(route.title, "Planned Basic");
  assert.equal(route.summary, PLANNED_ROUTE_SENTENCE);
  assert.equal(explainPendingRoute({ lane: 1 }).title, "Planned Ultimate");
  assert.equal(explainPendingRoute({}).title, "Planned Action");

  const net = explainNetHealth({ outcome: "damage", netDelta: -9.81 });
  assert.equal(net.title, "Net Damage");
  assert.equal(net.summary, NET_LIFE_SENTENCE);
  assert.deepEqual(
    net.rows.map((row) => `${row.label}: ${row.value}`),
    ["Recipient: Unavailable in this view", "Life Change: -9.81"],
  );
  assert.equal(
    explainNetHealth({ outcome: "healing", netDelta: 8 }).title,
    "Net Healing",
  );
  assert.equal(
    explainNetHealth({ outcome: "unchanged", netDelta: 0 }).title,
    "No Net Change",
  );

  const heal = explainActivation({
    component: "basic",
    tokenId: "basic_heal",
    sourceClassId: 5,
  });
  assert.equal(heal.summary, "The Priest used Lesser Heal this tick.");
  const hidden = explainActivation({
    component: "basic",
    tokenId: "basic_heal",
    sourceClassId: 5,
    sourceDisclosure: "redacted",
  });
  assert.equal(hidden.summary, "Lesser Heal was used this tick.");
  const burst = explainActivation({
    component: "ultimate",
    tokenId: "mage_burst",
    sourceClassId: 1,
  });
  assert.equal(burst.summary, "The Mage used Burst this tick.");

  const owner = {
    presentation_key: "k:1",
    public_agent_id: "1",
    class_id: 2,
    team_id: 1,
  };
  const allowed = explainLegality(
    {
      owner_presentation_key: "k:1",
      owner_public_agent_id: "1",
      basic_available: false,
    },
    0,
    owner,
  );
  assert.ok(allowed);
  assert.deepEqual(
    allowed.rows.map((row) => `${row.label}: ${row.value}`),
    ["Allowed Now: No"],
  );
  for (const tip of [route, net, heal, hidden, burst, allowed]) {
    assert.doesNotMatch(
      `${tip.title} ${tip.summary ?? ""} ${tip.rows.map((row) => `${row.label} ${row.value}`).join(" ")}`,
      INTERNAL,
    );
  }
});
