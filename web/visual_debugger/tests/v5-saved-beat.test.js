/**
 * @file Check the V5 Saved Beat add-on of `src/looks/health-text.js` (switch
 * saved_beat, off by default). Contracts: savedBeatHeld is true only when the
 * recorded Oracle totals say a heal kept a champion alive through blows that
 * would have killed it (healing above 0, start health minus damage at most 0,
 * alive after) and is false without totals; switched on in Oracle View, build
 * draws "Saved" in Dawn Gold stacked 3 px above the number inside the number's
 * set (or alone on the cue when the number is 0) and, while playing, a Dawn
 * Gold rim at 0.98 of the body radius whose DOM opacity is 0; in normal motion
 * the set pops, the word rises and the rim flashes exactly twice inside the
 * window where the painter's ease-out group fade is at full strength
 * (GROUP_FULL_STRENGTH); reduced motion gives the rim one steady flash;
 * settled draws no rim; switched off nothing extra is drawn; Agent POV never
 * shows it, even if totals were present; lifeNumberCueBox grows the number's
 * box upward only while the switch is on and the beat happened; the markup is
 * export-safe and inside the shared per-event budget.
 */
import assert from "node:assert/strict";
import test from "node:test";

import * as healthText from "../src/looks/health-text.js";
import {
  draw,
  EXPORT_UNSAFE,
  filtersWith,
  healthRow,
  planFor,
} from "./v5-health-text-support.js";

const ON = filtersWith({ saved_beat: true });
const saved = (overrides = {}) =>
  healthRow({
    healthBefore: 10,
    healthAfter: 6,
    totalDamage: 12,
    totalHealing: 8,
    ...overrides,
  });

test("savedBeatHeld compares only the recorded totals", () => {
  assert.equal(healthText.savedBeatHeld(saved()), true);
  assert.equal(
    healthText.savedBeatHeld(saved({ totalDamage: 10, healthAfter: 8, netDelta: -2 })),
    true,
  );
  assert.equal(healthText.savedBeatHeld(saved({ totalHealing: 0 })), false);
  assert.equal(healthText.savedBeatHeld(saved({ totalDamage: 9 })), false);
  assert.equal(healthText.savedBeatHeld(saved({ healthAfter: 0 })), false);
  assert.equal(
    healthText.savedBeatHeld(saved({ totalDamage: null, totalHealing: null })),
    false,
  );
  assert.equal(healthText.savedBeatHeld({ kind: "regeneration", value: 3 }), false);
});

test("switched on in Oracle View: Saved above the number and a rim that flashes twice", () => {
  const drawn = draw(saved(), planFor("researcher"), ON);
  const set = drawn.group.querySelector(".combat-number-set");
  assert.ok(set);
  const number = set.querySelector(".combat-number");
  const word = set.querySelector(".combat-saved");
  assert.ok(number && word);
  assert.equal(word.textContent, "Saved");
  assert.equal(word.getAttribute("font-size"), "13");
  assert.ok(Number(word.getAttribute("y")) < Number(number.getAttribute("y")) - 9);
  assert.equal(word.getAttribute("x"), number.getAttribute("x"));
  const rim = drawn.group.querySelector(".combat-saved-rim");
  assert.ok(rim);
  assert.equal(rim.getAttribute("opacity"), "0");
  assert.equal(Number(rim.getAttribute("r")), 20 * 0.98 - 0.75);
  assert.equal(rim.getAttribute("cx"), "300");
  const ids = drawn.animationSpecs.map((spec) => spec.id.split(":").at(-1));
  assert.deepEqual(ids.sort(), ["number-pop", "saved-rim", "saved-rise"]);
  const pop = drawn.animationSpecs.find((spec) => spec.id.endsWith(":number-pop"));
  assert.equal(pop.element, set);
  const flashes = drawn.animationSpecs.find((spec) => spec.id.endsWith(":saved-rim"));
  const peaks = flashes.keyframes.filter(
    (/** @type {Record<string, unknown>} */ frame) => frame.opacity === 1,
  );
  assert.equal(peaks.length, 2);
  // The painter's ease-out group fade is at full strength from about 11.3% to
  // 53.3% of the 640 ms window (keyframes 0.18 and 0.72 under ease-out).
  const { from, to } = healthText.GROUP_FULL_STRENGTH;
  assert.ok(Math.abs(from - 0.1127) < 0.002 && Math.abs(to - 0.533) < 0.002);
  assert.ok(Math.abs(flashes.options.delay - (460 + from * 640)) < 1e-9);
  assert.ok(Math.abs(flashes.options.duration - (to - from) * 640) < 1e-9);
  assert.doesNotMatch(drawn.group.markup(), EXPORT_UNSAFE);
});

test("a beat with no net change shows Saved alone on the cue", () => {
  const drawn = draw(
    saved({ healthBefore: 10, healthAfter: 10, totalDamage: 12, totalHealing: 12 }),
    planFor("researcher"),
    ON,
  );
  assert.equal(drawn.group.querySelector(".combat-number"), null);
  const word = drawn.group.querySelector(".combat-saved");
  assert.ok(word);
  assert.equal(word.getAttribute("y"), "168");
});

test("settled shows the word but no rim; reduced motion flashes the rim once, steadily", () => {
  const settled = draw(saved(), planFor("researcher"), ON, { settled: true });
  assert.ok(settled.group.querySelector(".combat-saved"));
  assert.equal(settled.group.querySelector(".combat-saved-rim"), null);
  assert.equal(settled.animationSpecs.length, 0);
  const reduced = draw(saved(), planFor("researcher"), ON, { motionMode: "reduced" });
  assert.equal(reduced.animationSpecs.length, 1);
  const [steady] = reduced.animationSpecs;
  assert.ok(steady.id.endsWith(":saved-rim"));
  assert.deepEqual(steady.keyframes, [{ opacity: 1 }, { opacity: 1 }]);
  const full = healthText.GROUP_FULL_STRENGTH;
  assert.ok(
    Math.abs(steady.options.delay - (460 + full.from * 640) * (220 / 1100)) < 1e-9,
  );
});

test("switched off, nothing extra is drawn and the number is plain", () => {
  const off = draw(saved(), planFor("researcher"), null);
  assert.equal(off.group.querySelector(".combat-saved"), null);
  assert.equal(off.group.querySelector(".combat-saved-rim"), null);
  assert.equal(off.group.querySelector(".combat-number-set"), null);
  assert.equal(off.group.querySelector(".combat-number")?.textContent, "−4");
  assert.equal(off.animationSpecs.length, 1);
});

test("Agent POV never shows Saved Beat, even if a row carried totals", () => {
  const pov = draw(saved(), planFor("agent_pov"), ON);
  assert.equal(pov.records[0].marks.saved, false);
  assert.equal(pov.group.querySelector(".combat-saved"), null);
  assert.equal(pov.group.querySelector(".combat-saved-rim"), null);
  assert.equal(
    pov.group.querySelector(".combat-number")?.getAttribute("data-tone"),
    "damage",
  );
});

test("lifeNumberCueBox grows upward only while the switch is on and the beat happened", () => {
  const event = saved({ paintParts: { battleText: true, savedBeat: true } });
  const box = healthText.lifeNumberCueBox(event);
  assert.ok(box);
  assert.ok(box.top < -healthText.FOOTPRINT.netText.height);
  assert.equal(
    healthText.lifeNumberCueBox({ ...event, paintParts: { battleText: true } }),
    null,
  );
  assert.equal(healthText.lifeNumberCueBox({ ...event, totalHealing: 0 }), null);
});

test("the event with every add-on stays inside the shared per-event budget", () => {
  const all = filtersWith({
    saved_beat: true,
    last_breath: true,
    battle_text_detail: true,
  });
  const drawn = draw(
    saved({
      healthBefore: 10,
      healthAfter: 2,
      totalDamage: 16,
      totalHealing: 8,
      damageSources: [{ sourcePresentationKey: "a", sourceClassId: 1, amount: 16 }],
    }),
    planFor("researcher"),
    all,
  );
  assert.ok(drawn.group.querySelector(".combat-last-breath"));
  assert.ok(drawn.group.querySelector(".combat-underline__bar"));
  assert.ok(drawn.nodeCount <= healthText.BUDGET.nodesPerRecord);
  assert.equal(drawn.animationSpecs.length, healthText.BUDGET.animationsPerRecord);
});
