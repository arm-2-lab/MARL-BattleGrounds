/**
 * @file Check the V5 Last Breath add-on of `src/looks/health-text.js` (switch
 * last_breath, off by default). Contracts: lastBreathHeld is true only for a
 * living champion that lost at least as much life this tick as it has left;
 * the switch comes from the page's filters, else the plan's paint part
 * lastBreath, else the registry default (off); switched on, build draws one
 * export-safe Cinder smolder at 25% over the empty part of the vessel (Life
 * Vessels) or of the life ring (Life Vessels off), sized from the Champions
 * token geometry, and a stale saved Heraldic Tokens Off keeps the vessel; switched off, or without the plan row's maximumHealth and
 * recipientRadius, nothing is drawn; in Agent POV it shows only for a champion
 * seen at both ends of the tick (never on an arrival tick, a return or an
 * unseen body); in play it fades in at the plan's Settle window start (fill
 * "backwards"), with no animation when settled or in reduced motion; Oracle View
 * then Agent POV then Oracle View draws exactly a fresh Oracle drawing; and the
 * event stays inside the shared per-event budget.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { tokenGeometry } from "../src/looks/champions.js";
import * as healthText from "../src/looks/health-text.js";
import {
  draw,
  EXPORT_UNSAFE,
  filtersWith,
  healthRow,
  planFor,
} from "./v5-health-text-support.js";

const ON = filtersWith({ last_breath: true });
const flagged = () => healthRow({ healthBefore: 40, healthAfter: 18 });

test("lastBreathHeld follows the store rule exactly", () => {
  assert.equal(healthText.lastBreathHeld(flagged()), true);
  assert.equal(
    healthText.lastBreathHeld(healthRow({ healthBefore: 36, healthAfter: 18 })),
    true,
  );
  assert.equal(
    healthText.lastBreathHeld(healthRow({ healthBefore: 50, healthAfter: 30 })),
    false,
  );
  assert.equal(
    healthText.lastBreathHeld(healthRow({ healthBefore: 20, healthAfter: 0 })),
    false,
  );
  assert.equal(
    healthText.lastBreathHeld(healthRow({ healthBefore: 18, healthAfter: 40 })),
    false,
  );
  assert.equal(healthText.lastBreathHeld({ kind: "regeneration", value: 3 }), false);
});

test("the switch is off by default and reads filters before the plan's paint part", () => {
  const event = flagged();
  const plan = planFor("researcher");
  const defaults = draw(event, plan, null);
  assert.equal(defaults.records[0].marks.lastBreath, false);
  assert.equal(defaults.group.querySelector(".combat-last-breath"), null);
  const fromPlan = draw(
    { ...event, paintParts: { ...event.paintParts, lastBreath: true } },
    plan,
    null,
  );
  assert.equal(fromPlan.records[0].marks.lastBreath, true);
  const filtersWin = draw(
    { ...event, paintParts: { ...event.paintParts, lastBreath: true } },
    plan,
    filtersWith({ last_breath: false }),
  );
  assert.equal(filtersWin.records[0].marks.lastBreath, false);
});

test("on a vessel the empty top smolders in Cinder at 25%; on the ring the empty arc", () => {
  const event = flagged();
  const vessel = draw(event, planFor("researcher"), ON);
  const smolder = vessel.group.querySelector(".combat-last-breath");
  assert.ok(smolder);
  assert.equal(smolder.getAttribute("data-style"), "vessel");
  assert.equal(smolder.getAttribute("opacity"), "0.25");
  assert.equal(smolder.getAttribute("visibility"), "visible");
  const geometry = tokenGeometry(20);
  const level = 200 + geometry.vesselRadius * (1 - 2 * 0.18);
  assert.match(
    smolder.getAttribute("d") ?? "",
    new RegExp(` ${Math.round(level * 1000) / 1000} A`, "u"),
  );
  assert.equal(smolder.getAttribute("stroke-width"), null);
  const ring = draw(
    event,
    planFor("researcher"),
    filtersWith({ last_breath: true, life_vessels: false }),
  );
  const arc = ring.group.querySelector(".combat-last-breath");
  assert.ok(arc);
  assert.equal(arc.getAttribute("data-style"), "arc");
  assert.equal(Number(arc.getAttribute("stroke-width")), geometry.lifeWidth);
  // Heraldic Tokens is no longer a switch: a stale saved Off keeps the vessel.
  const staleOff = draw(
    event,
    planFor("researcher"),
    filtersWith({ last_breath: true, heraldic_tokens: false }),
  );
  assert.equal(
    staleOff.group.querySelector(".combat-last-breath")?.getAttribute("data-style"),
    "vessel",
  );
  for (const drawn of [vessel, ring]) {
    assert.doesNotMatch(drawn.group.markup(), EXPORT_UNSAFE);
  }
});

test("without the row's geometry or with the switch off the plain number stays", () => {
  const event = { ...flagged(), recipientRadius: undefined, maximumHealth: undefined };
  const drawn = draw(event, planFor("researcher"), ON);
  assert.equal(drawn.records[0].marks.lastBreath, true);
  assert.equal(drawn.group.querySelector(".combat-last-breath"), null);
  assert.equal(drawn.group.querySelector(".combat-number")?.textContent, "−22");
  const off = draw(
    flagged(),
    planFor("researcher"),
    filtersWith({ last_breath: false }),
  );
  assert.equal(off.group.querySelector(".combat-last-breath"), null);
  assert.equal(off.group.querySelector(".combat-number")?.textContent, "−22");
});

test("Agent POV shows it only for a champion seen at both ends of the tick", () => {
  const event = flagged();
  assert.ok(
    draw(event, planFor("agent_pov"), ON).group.querySelector(".combat-last-breath"),
  );
  const arrived = draw(event, planFor("agent_pov", { start: null }), ON);
  assert.equal(arrived.group.querySelector(".combat-last-breath"), null);
  assert.equal(arrived.group.querySelector(".combat-number"), null);
  const returned = draw(event, planFor("agent_pov", { returned: true }), ON);
  assert.equal(returned.group.querySelector(".combat-last-breath"), null);
  const unseen = draw(event, { ...planFor("agent_pov"), bodies: [] }, ON);
  assert.equal(unseen.group.querySelector(".combat-last-breath"), null);
  const unknown = healthText.records({ event, filters: ON })[0];
  assert.equal(unknown.marks.lastBreath, false);
});

test("in play it fades in at Settle; settled and reduced motion add no animation", () => {
  const event = flagged();
  const playing = draw(event, planFor("researcher"), ON);
  const fade = playing.animationSpecs.find((spec) => spec.id.endsWith(":last-breath"));
  assert.ok(fade);
  assert.equal(fade.options.delay, 620);
  assert.equal(fade.options.fill, "backwards");
  assert.deepEqual(fade.keyframes, [{ opacity: 0 }, { opacity: 0.25 }]);
  const settled = draw(event, planFor("researcher"), ON, { settled: true });
  assert.equal(settled.animationSpecs.length, 0);
  assert.ok(settled.group.querySelector(".combat-last-breath"));
  const reduced = draw(event, planFor("researcher"), ON, { motionMode: "reduced" });
  assert.equal(
    reduced.animationSpecs.some((spec) => spec.id.endsWith(":last-breath")),
    false,
  );
});

test("Oracle View then Agent POV then Oracle View equals a fresh Oracle drawing", () => {
  const event = flagged();
  const fresh = draw(event, planFor("researcher"), ON).group.markup();
  draw(event, planFor("agent_pov", { start: null }), ON);
  const again = draw(event, planFor("researcher"), ON).group.markup();
  assert.equal(again, fresh);
});

test("the event stays inside the shared per-event budget", () => {
  const drawn = draw(flagged(), planFor("researcher"), ON);
  assert.ok(drawn.nodeCount <= healthText.BUDGET.nodesPerRecord);
  assert.ok(drawn.animationSpecs.length <= healthText.BUDGET.animationsPerRecord);
});
