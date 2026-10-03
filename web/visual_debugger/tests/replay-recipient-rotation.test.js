/**
 * @file Check that an agent-recipient change preserves every required replay scope
 * field and accepts only joined frames, and that Replay Viewer Tab / Shift+Tab
 * stepping (replayAgentStepCommand, bindReplayAgentStepping) moves the Oracle
 * selection or the Agent POV agent through the recorded roster with the same
 * command the roster row sends, wraps at the ends, ignores live frames and keys
 * from outside the battlefield field, and registers exactly one handler.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  joinReplayTransportAndTimelineV1,
  joinTransportAndAuthorizedPresentationV1,
} from "../src/authorized-presentation-normalizer.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { authorizedPresentationIdentityRows } from "../src/authorized-presentation-adapter.js";
import { runLocalCommand } from "../src/command-set.js";
import {
  bindReplayAgentStepping,
  isReplayAgentRecipientIdentityRotation,
  isReplayAgentRecipientRotation,
  replayAgentRecipientRotationIdentity,
  replayAgentStepCommand,
} from "../src/replay-recipient-rotation.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @type {readonly ("replay_no_shared_obs_agent_pov" | "replay_shared_obs_agent_pov")[]} */
const REPLAY_AGENT_KINDS = Object.freeze([
  "replay_no_shared_obs_agent_pov",
  "replay_shared_obs_agent_pov",
]);

/** @param {"replay_no_shared_obs_agent_pov" | "replay_shared_obs_agent_pov"} kind */
async function joinedReplayAgent(kind) {
  const source = fixture.pairs[kind];
  const joined = await joinTransportAndAuthorizedPresentationV1(
    source.transport,
    source.presentation,
  );
  return joinReplayTransportAndTimelineV1(joined, source.timeline);
}

/** @param {NonNullable<ReturnType<typeof replayAgentRecipientRotationIdentity>>} source
 * @param {Readonly<Record<string, unknown>>} changes */
function changedIdentity(source, changes) {
  return Object.freeze({ ...source, ...changes });
}

/** @param {unknown} joined */
function requiredRotationIdentity(joined) {
  const identity = replayAgentRecipientRotationIdentity(joined);
  if (identity === null) {
    throw new TypeError("Expected a Replay Agent rotation identity.");
  }
  return identity;
}

test("Replay Agent recipient identity is extracted only from a branded pair", async () => {
  for (const kind of REPLAY_AGENT_KINDS) {
    const joined = await joinedReplayAgent(kind);
    const identity = requiredRotationIdentity(joined);
    assert.equal(Object.isFrozen(identity), true);
    assert.equal(
      identity.recipientPublicAgentId,
      joined.presentation.authority.recipient_public_agent_id,
    );
    assert.equal(
      identity.recipientPresentationKey,
      joined.presentation.authority.recipient_presentation_key,
    );
    assert.equal(JSON.parse(identity.scope).length, 23);
    assert.equal(replayAgentRecipientRotationIdentity({ ...joined }), null);
  }
});

test("Replay Agent recipient rotation requires one unchanged replay scope", async () => {
  for (const kind of REPLAY_AGENT_KINDS) {
    const identity = requiredRotationIdentity(await joinedReplayAgent(kind));
    const next = changedIdentity(identity, {
      recipientPublicAgentId: "agent-slot-1",
      recipientPresentationKey:
        "pov_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    });
    assert.equal(isReplayAgentRecipientIdentityRotation(identity, next), true);

    const scope = JSON.parse(identity.scope);
    for (let index = 0; index < scope.length; index += 1) {
      const changedScope = [...scope];
      changedScope[index] =
        typeof changedScope[index] === "boolean"
          ? !changedScope[index]
          : typeof changedScope[index] === "number"
            ? changedScope[index] + 1
            : `${changedScope[index]}-changed`;
      assert.equal(
        isReplayAgentRecipientIdentityRotation(
          identity,
          changedIdentity(next, { scope: JSON.stringify(changedScope) }),
        ),
        false,
        `scope slot ${index} must be a reset boundary`,
      );
    }
    assert.equal(
      isReplayAgentRecipientIdentityRotation(
        identity,
        changedIdentity(next, {
          recipientPublicAgentId: identity.recipientPublicAgentId,
        }),
      ),
      false,
    );
    assert.equal(
      isReplayAgentRecipientIdentityRotation(
        identity,
        changedIdentity(next, {
          recipientPresentationKey: identity.recipientPresentationKey,
        }),
      ),
      false,
    );
  }
});

test("Replay Agent preference rotation rejects same-recipient and leaf crossings", async () => {
  const noShared = await joinedReplayAgent("replay_no_shared_obs_agent_pov");
  const shared = await joinedReplayAgent("replay_shared_obs_agent_pov");
  assert.equal(isReplayAgentRecipientRotation(noShared, noShared), false);
  assert.equal(isReplayAgentRecipientRotation(shared, shared), false);
  assert.equal(isReplayAgentRecipientRotation(noShared, shared), false);
  assert.equal(isReplayAgentRecipientRotation(shared, noShared), false);
  assert.equal(isReplayAgentRecipientRotation(null, shared), false);
});

test("Tab steps the Oracle selection and the Agent POV agent through the roster", async () => {
  const oracle = await normalizeAuthorizedPresentationFrameV1(
    fixture.presentations.replay_oracle,
  );
  const oracleRows = authorizedPresentationIdentityRows(oracle);
  assert.ok(oracleRows.length > 1);
  const first = oracleRows[0];
  const last = oracleRows.at(-1);
  assert.ok(first && last);
  assert.deepEqual(replayAgentStepCommand(oracle, 1, first.public_agent_id), {
    command_type: "activate_authorized_agent",
    presentation_key: oracleRows[1].presentation_key,
  });
  assert.deepEqual(replayAgentStepCommand(oracle, -1, first.public_agent_id), {
    command_type: "activate_authorized_agent",
    presentation_key: last.presentation_key,
  });
  assert.deepEqual(replayAgentStepCommand(oracle, 1, last.public_agent_id), {
    command_type: "activate_authorized_agent",
    presentation_key: first.presentation_key,
  });

  const pov = await normalizeAuthorizedPresentationFrameV1(
    fixture.presentations.replay_no_shared_obs_agent_pov,
  );
  const povRows = authorizedPresentationIdentityRows(pov);
  const recipientIndex = povRows.findIndex(
    (row) => row.public_agent_id === pov.authority.recipient_public_agent_id,
  );
  assert.ok(recipientIndex >= 0);
  const next = povRows[(recipientIndex + 1) % povRows.length];
  assert.deepEqual(replayAgentStepCommand(pov, 1), {
    command_type: "activate_replay_pov_agent",
    global_slot: next.command_global_slot,
    public_agent_id: next.public_agent_id,
  });
  assert.ok(Object.isFrozen(replayAgentStepCommand(pov, -1)));

  const live = await normalizeAuthorizedPresentationFrameV1(
    fixture.presentations.live_oracle,
  );
  assert.equal(replayAgentStepCommand(live, 1), null);
  assert.equal(replayAgentStepCommand({ raw: true }, 1), null);
});

test("Tab stepping acts only from the field and registers one handler", async () => {
  const pov = await normalizeAuthorizedPresentationFrameV1(
    fixture.presentations.replay_shared_obs_agent_pov,
  );
  const field = { closest: () => null };
  const elsewhere = { closest: () => null };
  /** @type {unknown[]} */
  const sent = [];
  const unbind = bindReplayAgentStepping({
    getPresentation: () => pov,
    isFieldTarget: (target) => /** @type {unknown} */ (target) === field,
    onCommand: (command) => {
      sent.push(command);
    },
  });
  try {
    assert.throws(
      () =>
        bindReplayAgentStepping({
          getPresentation: () => pov,
          isFieldTarget: () => true,
          onCommand: () => undefined,
        }),
      TypeError,
    );
    assert.equal(
      runLocalCommand("replay_viewer", {
        key: "Tab",
        shiftKey: false,
        target: elsewhere,
      }),
      false,
    );
    assert.equal(sent.length, 0);
    assert.equal(
      runLocalCommand("replay_viewer", { key: "Tab", shiftKey: true, target: field }),
      true,
    );
    assert.deepEqual(sent, [replayAgentStepCommand(pov, -1)]);
  } finally {
    unbind();
  }
  assert.equal(
    runLocalCommand("replay_viewer", { key: "Tab", shiftKey: false, target: field }),
    false,
  );
});
