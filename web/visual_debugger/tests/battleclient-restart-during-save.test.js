/**
 * @file Check the restart rule while the game-over replay saves in the
 * background (recordingCommandDecision in controls.js). While the recording
 * lifecycle is "sealed" or "finalized_unsaved" (the page says "Saving
 * Replay…"), R or a plain reset is allowed as a new frozen save_and_replace
 * command that wraps the reset, so no Save, Discard or Cancel dialog opens and
 * the server restarts once the save ends. Other keys stay fenced, and the
 * ordinary rules are unchanged: an unsaved recording still asks to confirm,
 * and a saved one restarts directly.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { keyboardCommand, recordingCommandDecision } from "../src/controls.js";

/** @param {string} lifecycle @param {boolean} discard */
function frameWith(lifecycle, discard) {
  return {
    scenario: { name: "alpha", ordinary_movement_distance_scale: 0.5 },
    available_scenarios: [{ name: "alpha" }],
    recording: {
      lifecycle,
      captured_transition_count: 300,
      expected_transition_count: 300,
      restart_fenced: !["saved", "reviewing"].includes(lifecycle),
      discard_available: discard,
    },
  };
}

test("a restart during the background save goes out as Save And Replace", () => {
  for (const lifecycle of ["sealed", "finalized_unsaved"]) {
    for (const command of [keyboardCommand("r"), { command_type: "reset" }]) {
      const decision = recordingCommandDecision(frameWith(lifecycle, true), command);
      assert.equal(decision.action, "allow");
      assert.deepEqual(decision.command, {
        command_type: "save_and_replace",
        replacement: { command_type: "reset" },
      });
      assert.notEqual(decision.command, command);
      assert.equal(Object.isFrozen(decision.command), true);
    }
  }
});

test("other keys stay fenced during the save, and the ordinary rules are kept", () => {
  assert.equal(
    recordingCommandDecision(frameWith("sealed", true), keyboardCommand("w")).action,
    "block",
  );
  const unsaved = recordingCommandDecision(
    frameWith("recording", true),
    keyboardCommand("r"),
  );
  assert.equal(unsaved.action, "confirm");
  const saved = recordingCommandDecision(
    frameWith("saved", false),
    keyboardCommand("r"),
  );
  assert.equal(saved.action, "allow");
  assert.equal(saved.command?.command_type, "keyboard");
});
