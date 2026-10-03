/**
 * @file Check carryFromEarlierTicks in `src/choreography-painter.js`, the
 * painter's memory of the herald row of the tick it last painted for one
 * audience. A Herald Card holds through the next tick only: a tick with no
 * herald row of its own gets a held herald_card row with no falls, which
 * looks/heralds.js draws as the previous tick's card with held true, and a
 * tick with its own row (say a plain single fall) gets the previous row as
 * `previous`. A repaint of the same tick carries the same row; a seek, a skip
 * back, a view or recipient change (Oracle -> POV -> Oracle) or a plan without
 * audienceKey or an integer simulatorStep carries nothing, like a fresh
 * render. A herald row that was never drawn (not spatial) is never held.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { carryFromEarlierTicks } from "../src/choreography-painter.js";
import * as heralds from "../src/looks/heralds.js";

const ORACLE = JSON.stringify(["s", "researcher", null]);
const POV = JSON.stringify(["s", "agent_pov", "agent-slot-1"]);
const bounds = { left: 0, top: 0, right: 1200, bottom: 700, width: 1200, height: 700 };

/** @param {string} id @param {number} victimTeam */
function fall(id, victimTeam) {
  return {
    victim: { title: `Warrior · ID ${id}`, classId: 2, teamId: victimTeam },
    killers: [],
    killingTeamId: 3 - victimTeam,
    points: 1,
  };
}

/** @param {number} step @param {any[]} falls @param {Record<string, any>} [extra] */
function heraldRow(step, falls, extra = {}) {
  return {
    kind: "herald_card",
    eventId: `${step}:herald`,
    count: falls.length,
    falls,
    paintParts: { effect: true },
    spatial: true,
    persistent: true,
    phaseStart: 960,
    phaseEnd: 2400,
    ...extra,
  };
}

/**
 * @param {number} step
 * @param {{herald?: any, audienceKey?: string | null}} [rows]
 */
function plan(step, rows = {}) {
  return {
    audienceKey: rows.audienceKey === undefined ? ORACLE : rows.audienceKey,
    simulatorStep: step,
    transitionId: `t${step}`,
    beats: { explanationEnd: 1500 },
    events: [],
    derivedEvents: rows.herald ? [rows.herald] : [],
  };
}

/** Paint plans in order and return the carry of each. @param {any[]} plans */
function paintAll(plans) {
  /** @type {any} */
  let memory = null;
  return plans.map((each) => {
    const result = carryFromEarlierTicks(memory, each);
    memory = result.memory;
    return result.carry;
  });
}

test("a Herald Card holds through the next tick and no longer", () => {
  const double = heraldRow(10, [fall("6", 2), fall("7", 2)]);
  const [first, held, after] = paintAll([
    plan(10, { herald: double }),
    plan(11),
    plan(12),
  ]);
  assert.equal(first.heldRows.length, 0);
  assert.equal(first.previousHerald, null);
  assert.equal(held.previousHerald, double);
  assert.equal(held.heldRows.length, 1);
  const carrier = held.heldRows[0];
  assert.equal(carrier.kind, "herald_card");
  assert.equal(carrier.eventId, "t11:herald:held");
  assert.equal(carrier.held, true);
  assert.equal(carrier.persistent, true);
  assert.deepEqual(carrier.falls, []);
  const records = heralds.records({
    event: carrier,
    previous: held.previousHerald,
    bounds,
  });
  assert.equal(records.length, 1);
  assert.equal(records[0].held, true);
  assert.equal(records[0].title, "Double Takedown");
  assert.equal(heralds.records({ event: carrier, previous: null, bounds }).length, 0);
  assert.equal(after.previousHerald, null);
  assert.equal(after.heldRows.length, 0);
});

test("a tick with its own herald row gets the previous row, not a held carrier", () => {
  const double = heraldRow(10, [fall("6", 2), fall("7", 2)]);
  const single = heraldRow(11, [fall("8", 1)]);
  const [, next] = paintAll([
    plan(10, { herald: double }),
    plan(11, { herald: single }),
  ]);
  assert.equal(next.heldRows.length, 0);
  assert.equal(next.previousHerald, double);
  const records = heralds.records({
    event: single,
    previous: next.previousHerald,
    bounds,
  });
  assert.equal(records.length, 1);
  assert.equal(records[0].held, true);
  const quiet = heralds.records({
    event: single,
    previous: next.previousHerald,
    bounds,
    filters: { match_moments: false },
  });
  assert.equal(quiet.length, 0);
});

test("a seek, a skip back, a view change or a missing step carries nothing", () => {
  const double = heraldRow(10, [fall("6", 2), fall("7", 2)]);
  const start = plan(10, { herald: double });
  for (const next of [
    plan(12),
    plan(9),
    plan(11, { audienceKey: POV }),
    plan(11, { audienceKey: null }),
    { ...plan(11), simulatorStep: 11.5 },
  ]) {
    const [, carry] = paintAll([start, next]);
    assert.equal(carry.previousHerald, null);
    assert.equal(carry.heldRows.length, 0);
  }
  const [, pov, oracle] = paintAll([start, plan(11, { audienceKey: POV }), plan(11)]);
  assert.equal(pov.heldRows.length, 0);
  assert.equal(oracle.heldRows.length, 0);
  const [, held, repaint] = paintAll([start, plan(11), plan(11)]);
  assert.equal(held.heldRows.length, 1);
  assert.equal(repaint.previousHerald, double);
  assert.deepEqual(
    repaint.heldRows.map((row) => row.eventId),
    held.heldRows.map((row) => row.eventId),
  );
});

test("a herald row that was never drawn is never held", () => {
  const hidden = heraldRow(40, [fall("6", 2), fall("7", 2)], { spatial: false });
  const [, afterHidden] = paintAll([plan(40, { herald: hidden }), plan(41)]);
  assert.equal(afterHidden.previousHerald, null);
  assert.equal(afterHidden.heldRows.length, 0);
});
