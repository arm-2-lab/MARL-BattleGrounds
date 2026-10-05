/**
 * @file Check physical Manual ownership validation in live transport metadata.
 * Labels preserve empty roster rows, and malformed drafts cannot change ownership.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { normalizeCombatConfigurationV1 } from "../src/frame-normalizer.js";

const setup = () => ({
  team_a_controller: "system:team_a",
  team_b_controller: "system:team_b",
  execution_information_mode: "shared_obs",
  manual_slots: [0, 2, 8],
  slot_labels: [
    "Manual",
    "Actor A",
    "Manual",
    null,
    null,
    "Actor B",
    "Actor B",
    "Actor C",
    "Manual",
    null,
  ],
});

test("resolved physical ownership remains immutable", () => {
  const source = setup();
  const result = normalizeCombatConfigurationV1(source);
  assert.deepEqual(result, source);
  assert.ok(Object.isFrozen(result.manual_slots));
  assert.ok(Object.isFrozen(result.slot_labels));
  source.manual_slots.push(9);
  assert.deepEqual(result.manual_slots, [0, 2, 8]);
});

for (const change of [
  { manual_slots: [2, 0] },
  { manual_slots: [0, 0] },
  { manual_slots: [true] },
  { manual_slots: [10] },
  { manual_slots: [-1] },
  { manual_slots: [3] },
  { manual_slots: [1] },
  { manual_slots: null },
  { slot_labels: null },
  { slot_labels: Array(9).fill("Manual") },
  { slot_labels: Array(10).fill(" ") },
]) {
  test(`reject invalid ownership ${JSON.stringify(change)}`, () => {
    assert.throws(
      () => normalizeCombatConfigurationV1({ ...setup(), ...change }),
      TypeError,
    );
  });
}

test("legacy frame omits both fields; incomplete ownership is rejected", () => {
  const { manual_slots, slot_labels, ...legacy } = setup();
  assert.deepEqual(normalizeCombatConfigurationV1(legacy), legacy);
  assert.throws(
    () => normalizeCombatConfigurationV1({ ...legacy, manual_slots }),
    TypeError,
  );
  assert.throws(
    () => normalizeCombatConfigurationV1({ ...legacy, slot_labels }),
    TypeError,
  );
});
