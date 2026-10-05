/**
 * @file Check actual Python live frames through browser normalization and display.
 * Python supplies a temporary JSON file. Manual rows, policy read-only masks,
 * accepted actions, run state and forged ownership are checked without fixtures.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  authorizedPresentationController,
  authorizedPresentationInspectionState,
  authorizedPresentationPendingJointActionRows,
  authorizedPresentationResearcherInspectionState,
  authorizedPresentationSceneView,
} from "../src/authorized-presentation-adapter.js";
import { joinTransportAndAuthorizedPresentationV1 } from "../src/authorized-presentation-normalizer.js";

const pairs = JSON.parse(readFileSync(process.argv[2], "utf8"));
for (const pair of pairs) {
  const joined = await joinTransportAndAuthorizedPresentationV1(
    pair.raw,
    pair.presentation,
  );
  const presentation = joined.presentation;
  const inspection = authorizedPresentationInspectionState(presentation);
  const researcher = authorizedPresentationResearcherInspectionState(presentation);
  assert.equal(inspection.state_kind, researcher.state_kind);
  assert.ok(inspection.inspection?.decision_mask);
  const actor = inspection.inspection.actor_public_agent_id;
  const controller = authorizedPresentationController(
    presentation,
    joined.transport.combat_configuration,
    actor,
  );
  assert.ok(controller);
  assert.equal(controller.manual, inspection.state_kind === "live_editable");
  const rows = authorizedPresentationPendingJointActionRows(presentation);
  assert.equal(rows.length, pair.manual_slots.length);
  assert.deepEqual(
    rows.map((row) => row.pending_action.move_action),
    pair.manual_moves,
  );
  const sourceRows = (
    presentation.researcher_space?.pending_joint_action ??
    presentation.pending_joint_action
  ).action_rows;
  assert.deepEqual(
    rows.map((row) => row.pending_action),
    sourceRows.map((/** @type {any} */ row) => row.pending_action),
  );
  const display = authorizedPresentationSceneView(presentation);
  assert.ok(display);
  if (!controller.manual) {
    assert.equal(inspection.state_kind, "live_policy");
    assert.equal("draft_action" in inspection.inspection, false);
    assert.equal(display.pending_route, null);
    const falseDraft = structuredClone(pair);
    falseDraft.presentation.live_inspection.inspection.draft_action = {
      move_action: 0,
    };
    await assert.rejects(
      joinTransportAndAuthorizedPresentationV1(falseDraft.raw, falseDraft.presentation),
      /unknown field/u,
    );
    const wrongMask = structuredClone(pair);
    const mask = wrongMask.presentation.live_inspection.inspection.decision_mask;
    mask.owner_public_agent_id = "99999";
    await assert.rejects(
      joinTransportAndAuthorizedPresentationV1(wrongMask.raw, wrongMask.presentation),
    );
  }
  const badRun = structuredClone(pair);
  badRun.raw.run_state = "running";
  badRun.presentation.run_state = "paused";
  await assert.rejects(
    joinTransportAndAuthorizedPresentationV1(badRun.raw, badRun.presentation),
    /run state/u,
  );
  const badFresh = structuredClone(pair);
  badFresh.raw.fresh_launch_required = true;
  badFresh.presentation.fresh_launch_required = false;
  await assert.rejects(
    joinTransportAndAuthorizedPresentationV1(badFresh.raw, badFresh.presentation),
    /fresh launch state/u,
  );
  const invalidFresh = structuredClone(pair);
  invalidFresh.raw.fresh_launch_required = "false";
  await assert.rejects(
    joinTransportAndAuthorizedPresentationV1(
      invalidFresh.raw,
      invalidFresh.presentation,
    ),
    /boolean/u,
  );
  const badLabel = structuredClone(pair);
  const manual = badLabel.raw.combat_configuration.manual_slots[0];
  if (manual !== undefined) {
    badLabel.raw.combat_configuration.slot_labels[manual] = "Policy";
    await assert.rejects(
      joinTransportAndAuthorizedPresentationV1(badLabel.raw, badLabel.presentation),
      /ownership/u,
    );
  }
  if (inspection.state_kind === "live_policy" && sourceRows.length) {
    const missing = structuredClone(pair);
    const root = missing.presentation.researcher_space ?? missing.presentation;
    root.pending_joint_action.action_rows.pop();
    await assert.rejects(
      joinTransportAndAuthorizedPresentationV1(missing.raw, missing.presentation),
      /Manual ownership/u,
    );
  }
}
console.log(`Validated ${pairs.length} actual live presentation pairs`);
