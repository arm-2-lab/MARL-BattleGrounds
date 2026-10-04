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
 * The card queue (timing.queue true, a played tick one tick later) keeps one
 * card on screen at a time for at least MINIMUM_CARD_MS of replay time
 * ((now - shownAt) x rate), in continuous time: a card that arrives early
 * enters later (a copy of its row with the new phaseStart) while the card on
 * screen is held and fades out CARD_FADE_OUT_MS before it; a waiting card
 * starts mid-tick as soon as its turn comes, before the tick's own card; a
 * card that cannot start this tick keeps waiting; at most one card waits and
 * the bigger moment keeps that spot; on a long tick a card may enter and the
 * next follow in the same tick; with no new card the card on screen holds
 * through the next tick, then leaves once its minimum is up. A repaint of a
 * queued tick carries exactly the same; without
 * the queue (a seek, step, pause or still) a tick's own card shows at once.
 * A new episode in the same audience (a BattleClient Restart keeps the
 * session; plans tell episodes apart by the transitionId before its last
 * `:transition:<index>`) carries nothing, at the same step or one step on,
 * and a view switch while a card waits carries neither the card on screen
 * nor the waiting one, in either direction (Oracle -> POV -> Oracle).
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
 * @param {{herald?: any, audienceKey?: string | null, episode?: string}} [rows]
 */
function plan(step, rows = {}) {
  return {
    audienceKey: rows.audienceKey === undefined ? ORACLE : rows.audienceKey,
    simulatorStep: step,
    transitionId: `${rows.episode ?? "ep1"}:transition:${step}`,
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
  assert.equal(carrier.eventId, "ep1:transition:11:herald:held");
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

/** @param {number} step @param {number} count @param {Record<string, any>} [extra] */
function takedown(step, count, extra = {}) {
  return heraldRow(
    step,
    Array.from({ length: count }, (_, index) => fall(`${step}${index}`, 2)),
    { phaseStart: 1100, ...extra },
  );
}

test("the card queue keeps each card up for its minimum and makes the next one wait its turn", () => {
  const timing = (/** @type {number} */ now) => ({
    queue: true,
    now,
    rate: 1,
    filters: null,
  });
  const a = takedown(10, 2);
  const first = carryFromEarlierTicks(null, plan(10, { herald: a }), timing(0));
  assert.equal(first.memory?.card?.row, a);
  assert.equal(first.memory?.card?.shownAt, 1100);
  // Tick 11 starts 1700 ms later: A has been up 600 ms, so B enters at
  // 1400 + CARD_FADE_OUT_MS and A fades out at 1400.
  const b = takedown(11, 2);
  const second = carryFromEarlierTicks(
    first.memory,
    plan(11, { herald: b }),
    timing(1700),
  );
  const [heldA] = second.carry.heldRows;
  assert.equal(heldA.heldRow, a);
  assert.equal(heldA.exitAt, heralds.MINIMUM_CARD_MS - 600);
  assert.equal(heldA.persistent, false);
  assert.equal(
    second.carry.ownRow?.replacement?.phaseStart,
    1400 + heralds.CARD_FADE_OUT_MS,
  );
  assert.equal(second.carry.ownRow?.replacement?.queuedFromEventId, b.eventId);
  assert.equal(second.carry.ownRowsHoldPrevious, false);
  // Tick 12: B has been up only 150 ms, so C cannot enter this tick and waits.
  const c = takedown(12, 2);
  const third = carryFromEarlierTicks(
    second.memory,
    plan(12, { herald: c }),
    timing(3400),
  );
  assert.equal(third.carry.ownRow?.replacement, null);
  assert.equal(third.carry.heldRows.length, 1);
  assert.equal(third.carry.heldRows[0].exitAt, null);
  assert.equal(third.memory?.waiting, c);
  // A repaint of tick 12 carries exactly the same.
  const repaint = carryFromEarlierTicks(
    third.memory,
    plan(12, { herald: c }),
    timing(3500),
  );
  assert.equal(repaint.carry, third.carry);
  // Tick 13 has no card of its own: C enters once B's minimum is up.
  const fourth = carryFromEarlierTicks(third.memory, plan(13), timing(5100));
  const [heldB, queued] = fourth.carry.heldRows;
  assert.equal(heldB.exitAt, 150);
  assert.equal(queued.eventId, "ep1:transition:13:herald:queued");
  assert.equal(queued.phaseStart, 150 + heralds.CARD_FADE_OUT_MS);
  assert.equal(queued.queuedFromEventId, c.eventId);
  assert.equal(fourth.memory?.card?.row, queued);
  // Never two cards on screen: every hand-over fades the old card out first.
  for (const step of [second, fourth]) {
    const entering = step.carry.ownRow?.replacement ?? step.carry.heldRows.at(-1);
    const leaving = step.carry.heldRows[0];
    assert.ok(
      Number(leaving.exitAt) + heralds.CARD_FADE_OUT_MS <= Number(entering?.phaseStart),
    );
  }
});

test("a waiting card starts mid-tick when its turn comes, and only one card waits", () => {
  const timing = (/** @type {number} */ now) => ({
    queue: true,
    now,
    rate: 1,
    filters: null,
  });
  const first = carryFromEarlierTicks(
    null,
    plan(20, { herald: takedown(20, 2) }),
    timing(0),
  );
  const second = carryFromEarlierTicks(
    first.memory,
    plan(21, { herald: takedown(21, 2) }),
    timing(1700),
  );
  const double = takedown(22, 2);
  const third = carryFromEarlierTicks(
    second.memory,
    plan(22, { herald: double }),
    timing(3400),
  );
  assert.equal(third.memory?.waiting, double);
  // Tick 23: the waiting Double's turn comes at 300 ms, before tick 23's own
  // Triple arrives at 1100 ms, so the Double shows first and the Triple waits.
  const triple = takedown(23, 3);
  const fourth = carryFromEarlierTicks(
    third.memory,
    plan(23, { herald: triple }),
    timing(5100),
  );
  const queued = fourth.carry.heldRows.at(-1);
  assert.equal(queued?.queuedFromEventId, double.eventId);
  assert.equal(queued?.phaseStart, 150 + heralds.CARD_FADE_OUT_MS);
  assert.equal(fourth.carry.ownRow?.replacement, null);
  assert.equal(fourth.memory?.waiting, triple);
  // A card that arrives while another still waits: the bigger moment keeps the
  // waiting spot (a Triple outranks a Double) and the other is dropped.
  const early = takedown(23, 3, { phaseStart: 100 });
  const fifth = carryFromEarlierTicks(
    third.memory,
    plan(23, { herald: early }),
    timing(5100),
  );
  assert.equal(fifth.carry.ownRow?.replacement?.queuedFromEventId, early.eventId);
  assert.equal(fifth.carry.ownRow?.replacement?.phaseStart, 300);
  assert.ok(
    fifth.carry.heldRows.every((row) => row.queuedFromEventId !== double.eventId),
  );
  // A smaller own card loses to a waiting bigger one.
  const small = takedown(23, 2, { phaseStart: 100 });
  const sixth = carryFromEarlierTicks(
    { ...third.memory, waiting: triple, waitingSince: 3400 + 1100 },
    plan(23, { herald: small }),
    timing(5100),
  );
  assert.equal(sixth.carry.ownRow?.replacement, null);
  assert.equal(sixth.carry.heldRows.at(-1)?.queuedFromEventId, triple.eventId);
});

test("on a long tick a card can enter and leave in the same tick", () => {
  const timing = (/** @type {number} */ now) => ({
    queue: true,
    now,
    rate: 1,
    filters: null,
  });
  const longPlan = (/** @type {number} */ step, /** @type {any} */ herald) => ({
    ...plan(step, { herald }),
    beats: { explanationEnd: 2600, tickMs: 3200 },
  });
  const first = carryFromEarlierTicks(null, longPlan(50, takedown(50, 2)), timing(0));
  const waits = takedown(51, 2, { phaseStart: 3150 });
  const second = carryFromEarlierTicks(first.memory, longPlan(51, waits), timing(3200));
  // The card shown at 1100 ms of tick 50 is up 2100 ms at tick 51's start; the
  // own card arrives late (3150 ms) and shows then, the old one leaving first.
  assert.equal(second.carry.heldRows[0].exitAt, 3000);
  assert.equal(second.carry.ownRow?.replacement, waits);
  const own = takedown(52, 3, { phaseStart: 200 });
  const third = carryFromEarlierTicks(second.memory, longPlan(52, own), timing(6400));
  // Card 51 showed 50 ms before tick 52: tick 52's own card waits until
  // 1950 + CARD_FADE_OUT_MS, inside this long tick.
  assert.equal(third.carry.heldRows[0].exitAt, heralds.MINIMUM_CARD_MS - 50);
  assert.equal(
    third.carry.ownRow?.replacement?.phaseStart,
    heralds.MINIMUM_CARD_MS - 50 + heralds.CARD_FADE_OUT_MS,
  );
  assert.equal(third.carry.ownRow?.replacement?.exitAt, null);
});

test("with no new card the card on screen holds one more tick, then leaves after its minimum", () => {
  const timing = (/** @type {number} */ now, rate = 1) => ({
    queue: true,
    now,
    rate,
    filters: null,
  });
  const a = takedown(30, 2);
  const first = carryFromEarlierTicks(null, plan(30, { herald: a }), timing(0));
  const second = carryFromEarlierTicks(first.memory, plan(31), timing(1700));
  assert.equal(second.carry.heldRows[0].heldRow, a);
  assert.equal(second.carry.heldRows[0].exitAt, null);
  const third = carryFromEarlierTicks(second.memory, plan(32), timing(3400));
  assert.deepEqual(third.carry.heldRows, []);
  assert.equal(third.memory?.card, null);
  // At 2x everything runs in replay time (page time x rate): a tick lasts
  // 850 ms of page time and the minimum is 1 s of page time.
  const quick = carryFromEarlierTicks(null, plan(30, { herald: a }), timing(0, 2));
  const fast = carryFromEarlierTicks(
    quick.memory,
    plan(31, { herald: takedown(31, 2) }),
    timing(850, 2),
  );
  assert.equal(fast.carry.heldRows[0].exitAt, heralds.MINIMUM_CARD_MS - 600);
});

test("without the queue a seek, step or still shows the tick's own card at once", () => {
  const first = carryFromEarlierTicks(null, plan(40, { herald: takedown(40, 2) }), {
    queue: true,
    now: 0,
  });
  const own = takedown(41, 2);
  const still = carryFromEarlierTicks(first.memory, plan(41, { herald: own }), {
    queue: false,
    now: 100,
  });
  assert.equal(still.carry.ownRow, undefined);
  assert.equal(still.carry.ownRowsHoldPrevious, true);
  assert.deepEqual(still.carry.heldRows, []);
  assert.equal(still.memory?.card?.row, own);
  // A view change mid-queue forgets everything, like a fresh render.
  const pov = carryFromEarlierTicks(
    first.memory,
    plan(41, { herald: own, audienceKey: POV }),
    {
      queue: true,
      now: 100,
    },
  );
  assert.deepEqual(pov.carry.heldRows, []);
  assert.equal(pov.carry.ownRow, undefined);
});

test("a new episode at the same or the next step carries nothing from the old one", () => {
  const timing = (/** @type {number} */ now) => ({
    queue: true,
    now,
    rate: 1,
    filters: null,
  });
  const first = carryFromEarlierTicks(
    null,
    plan(50, { herald: takedown(50, 2) }),
    timing(0),
  );
  const second = carryFromEarlierTicks(
    first.memory,
    plan(51, { herald: takedown(51, 3) }),
    timing(1700),
  );
  assert.ok(second.memory?.card !== null || second.memory?.waiting !== null);
  for (const step of [51, 52]) {
    for (const queue of [true, false]) {
      const restarted = carryFromEarlierTicks(
        second.memory,
        plan(step, { episode: "ep2" }),
        {
          ...timing(3400),
          queue,
        },
      );
      assert.equal(restarted.carry.previousHerald, null);
      assert.deepEqual(restarted.carry.heldRows, []);
      assert.equal(restarted.carry.ownRow, undefined);
      assert.equal(restarted.memory?.card, null);
      assert.equal(restarted.memory?.waiting, null);
      assert.equal(restarted.memory?.episode, "ep2");
    }
  }
});

test("a view switch while a card waits carries neither card, in either direction", () => {
  const timing = (/** @type {number} */ now) => ({
    queue: true,
    now,
    rate: 1,
    filters: null,
  });
  /** @param {string} audienceKey @param {string} episode */
  const waitingIn = (audienceKey, episode) => {
    const first = carryFromEarlierTicks(
      null,
      // A late first card, so the next tick's card still waits when it ends.
      plan(60, { herald: takedown(60, 2, { phaseStart: 1600 }), audienceKey, episode }),
      timing(0),
    );
    const second = carryFromEarlierTicks(
      first.memory,
      plan(61, { herald: takedown(61, 3), audienceKey, episode }),
      timing(1700),
    );
    assert.notEqual(second.memory?.card, null);
    assert.notEqual(second.memory?.waiting, null);
    return second.memory;
  };
  const views = [
    { audienceKey: ORACLE, episode: "ep1" },
    { audienceKey: POV, episode: "ep1:actor-pov:1" },
  ];
  for (const [from, to] of [views, [...views].reverse()]) {
    const memory = waitingIn(from.audienceKey, from.episode);
    for (const step of [61, 62]) {
      const switched = carryFromEarlierTicks(
        memory,
        plan(step, { audienceKey: to.audienceKey, episode: to.episode }),
        timing(3400),
      );
      assert.equal(switched.carry.previousHerald, null);
      assert.deepEqual(switched.carry.heldRows, []);
      assert.equal(switched.memory?.card, null);
      assert.equal(switched.memory?.waiting, null);
      // Back to the first view: still nothing from before the switch.
      const back = carryFromEarlierTicks(
        switched.memory,
        plan(step + 1, { audienceKey: from.audienceKey, episode: from.episode }),
        timing(5100),
      );
      assert.deepEqual(back.carry.heldRows, []);
      assert.equal(back.carry.previousHerald, null);
      assert.equal(back.memory?.waiting, null);
    }
  }
});
