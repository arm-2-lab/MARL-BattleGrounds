/**
 * @file Recognize and drive a Replay Viewer change of agent: the recipient
 * rotation checks and the Tab / Shift+Tab agent stepping.
 *
 * Rotation checks (isReplayAgentRecipientRotation and helpers) compare already
 * joined transport/presentation identities: both recipient identifiers must
 * change while artifact, cursor and display conditions remain equal. main.js
 * uses them to keep panel preferences when only the Agent POV agent changes.
 *
 * Agent stepping (replayAgentStepCommand, bindReplayAgentStepping): in the
 * Replay Viewer, Tab and Shift+Tab step through the agents while the
 * battlefield field has focus (the one command set, row "next_champion",
 * handler "replay_next_agent"). In Oracle View they move the selection; in
 * Agent POV they move the POV agent, through every agent in the recorded
 * roster, because the docked Replay Viewer may watch any agent's view. The
 * command is the same one the roster row sends, so the host still checks it.
 *
 * Boundary: nothing here loads a replay, grants access, or reads facts beyond
 * identity (public ID, presentation key, command slot). No DOM is touched
 * except by the key handler, which only reads the event target.
 */
import {
  authorizedPresentationAudience,
  authorizedPresentationIdentityRows,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import { isJoinedTransportAndAuthorizedPresentationV1 } from "./authorized-presentation-normalizer.js";
import { registerLocalCommandHandler } from "./command-set.js";

/**
 * @typedef {Readonly<{
 *   scope: string,
 *   recipientPublicAgentId: string,
 *   recipientPresentationKey: string,
 * }>} ReplayAgentRecipientRotationIdentity
 */

/**
 * Return true for a non-null, non-array object. value is otherwise unchecked;
 * this helper alone does not establish a valid replay identity.
 *
 * @param {unknown} value @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Extract the continuity key and recipient IDs from a joined replay Agent view.
 *
 * value must be a registered transport/presentation pair for a SharedObs or
 * NoSharedObs agent-POV replay. Return null for another kind or a missing/wrongly
 * typed required field. Success returns a frozen record: scope serializes the
 * artifact, source, cursor and display conditions in a fixed order; the two
 * recipient strings are separate. Inputs are unchanged and nothing is fetched.
 * Nested scientific validation belongs to the normalizer that made the join.
 *
 * @param {unknown} value
 * @returns {ReplayAgentRecipientRotationIdentity | null}
 */
export function replayAgentRecipientRotationIdentity(value) {
  if (!isJoinedTransportAndAuthorizedPresentationV1(value)) {
    return null;
  }
  const joined = /** @type {Readonly<Record<string, any>>} */ (value);
  const transport = joined.transport;
  const presentation = joined.presentation;
  if (
    !isRecord(transport) ||
    !isRecord(presentation) ||
    transport.viewer_mode !== "replay" ||
    presentation.product_kind !== "replay_viewer" ||
    presentation.viewer_mode !== "replay" ||
    presentation.authority?.authority_kind !== "agent_pov" ||
    !["replay_no_shared_obs_agent_pov", "replay_shared_obs_agent_pov"].includes(
      presentation.presentation_kind,
    )
  ) {
    return null;
  }
  const artifactFacts = transport.artifact_facts;
  const artifactReference = artifactFacts?.artifact_summary?.replay_reference;
  const cursor = transport.cursor;
  const source = presentation.source;
  const authority = presentation.authority;
  if (
    !isRecord(artifactReference) ||
    !isRecord(cursor) ||
    !isRecord(source) ||
    !isRecord(authority) ||
    typeof transport.session_id !== "string" ||
    typeof source.source_session_id !== "string" ||
    typeof source.episode_id !== "string" ||
    typeof artifactReference.episode_id !== "string" ||
    typeof artifactReference.artifact_id !== "string" ||
    typeof artifactReference.context_digest_sha256 !== "string" ||
    typeof artifactReference.trajectory_content_digest_sha256 !== "string" ||
    typeof artifactReference.canonical_digest_sha256 !== "string" ||
    !Number.isSafeInteger(artifactReference.replay_schema_version) ||
    !Number.isSafeInteger(cursor.schema_version) ||
    !Number.isSafeInteger(cursor.frame_index) ||
    !Number.isSafeInteger(cursor.final_frame_index) ||
    !Number.isSafeInteger(cursor.cursor_generation) ||
    !Number.isSafeInteger(cursor.choreography_generation) ||
    typeof transport.preset !== "string" ||
    typeof transport.verbose !== "boolean" ||
    typeof authority.observation_mode !== "string" ||
    typeof authority.projection_basis !== "string" ||
    typeof authority.exact_actor_input_export_available !== "boolean" ||
    typeof authority.recipient_public_agent_id !== "string" ||
    typeof authority.recipient_presentation_key !== "string"
  ) {
    return null;
  }
  const scope = JSON.stringify([
    presentation.product_kind,
    presentation.viewer_mode,
    transport.frame_kind,
    transport.session_id,
    source.source_session_id,
    source.episode_id,
    artifactReference.episode_id,
    artifactReference.artifact_id,
    artifactReference.replay_schema_version,
    artifactReference.context_digest_sha256,
    artifactReference.trajectory_content_digest_sha256,
    artifactReference.canonical_digest_sha256,
    cursor.schema_version,
    cursor.frame_index,
    cursor.final_frame_index,
    cursor.cursor_generation,
    cursor.choreography_generation,
    authority.observation_mode,
    authority.projection_basis,
    authority.exact_actor_input_export_available,
    presentation.presentation_kind,
    transport.preset,
    transport.verbose,
  ]);
  return Object.freeze({
    scope,
    recipientPublicAgentId: authority.recipient_public_agent_id,
    recipientPresentationKey: authority.recipient_presentation_key,
  });
}

/**
 * Compare two extracted identities for a recipient-only change.
 *
 * previous and next may be any inputs. Return true only when their scope
 * strings match and both recipientPublicAgentId and recipientPresentationKey
 * are strings that differ. This checks the supplied identity records, not
 * their provenance; use isReplayAgentRecipientRotation for joined frames.
 * No values are changed.
 *
 * @param {unknown} previous
 * @param {unknown} next
 */
export function isReplayAgentRecipientIdentityRotation(previous, next) {
  return (
    isRecord(previous) &&
    isRecord(next) &&
    typeof previous.scope === "string" &&
    typeof next.scope === "string" &&
    previous.scope === next.scope &&
    typeof previous.recipientPublicAgentId === "string" &&
    typeof next.recipientPublicAgentId === "string" &&
    previous.recipientPublicAgentId !== next.recipientPublicAgentId &&
    typeof previous.recipientPresentationKey === "string" &&
    typeof next.recipientPresentationKey === "string" &&
    previous.recipientPresentationKey !== next.recipientPresentationKey
  );
}

/**
 * Return whether two joined replay frames differ only by agent recipient.
 *
 * previous and next are each checked by replayAgentRecipientRotationIdentity.
 * An invalid/non-replay input returns false. Both recipient IDs must change
 * while every extracted continuity field remains equal. This pure check does
 * not install the new view or change playback.
 *
 * @param {unknown} previous
 * @param {unknown} next
 */
export function isReplayAgentRecipientRotation(previous, next) {
  return isReplayAgentRecipientIdentityRotation(
    replayAgentRecipientRotationIdentity(previous),
    replayAgentRecipientRotationIdentity(next),
  );
}

/**
 * Work out the command that Tab (direction 1) or Shift+Tab (direction -1)
 * sends in the Replay Viewer. Pure.
 *
 * Oracle View: the agents are the recorded roster in its own order; the
 * current agent is currentPublicAgentId, else the scene's inspected agent;
 * the result selects the next one with {command_type:
 * "activate_authorized_agent", presentation_key}. Agent POV: the agents are the
 * whole recorded roster; the current agent is the POV agent
 * (authority.recipient_public_agent_id); the result moves the POV agent with
 * {command_type: "activate_replay_pov_agent", global_slot, public_agent_id}.
 * With no current agent, Tab picks the first agent and Shift+Tab the last.
 * Steps wrap around the ends.
 *
 * @param {unknown} presentation The installed presentation.
 * @param {1 | -1} direction 1 for Tab (next), -1 for Shift+Tab (previous).
 * @param {string | null} [currentPublicAgentId] Oracle View only: the agent
 *   selected on the page (default null: use the scene's inspected agent).
 * @returns {Readonly<Record<string, unknown>> | null} A frozen command, or null
 *   for a live frame, an unknown audience, a roster with no usable agent or a
 *   roster of one agent that is already current.
 */
export function replayAgentStepCommand(
  presentation,
  direction,
  currentPublicAgentId = null,
) {
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    presentation.viewer_mode !== "replay"
  ) {
    return null;
  }
  const audience = authorizedPresentationAudience(presentation);
  const pov = audience === "agent_pov";
  if (!pov && audience !== "researcher") return null;
  const rows = authorizedPresentationIdentityRows(presentation).filter((row) =>
    pov
      ? row.activation_kind === "replay_pov_global" &&
        Number.isInteger(row.command_global_slot)
      : row.activation_kind === "scene_agent",
  );
  if (rows.length === 0) return null;
  let current = pov ? presentation.authority?.recipient_public_agent_id : null;
  if (!pov) {
    const ownerKey =
      authorizedPresentationSceneView(presentation)?.selection
        ?.inspection_owner_presentation_key ?? null;
    current =
      typeof currentPublicAgentId === "string"
        ? currentPublicAgentId
        : (rows.find((row) => row.presentation_key === ownerKey)?.public_agent_id ??
          null);
  }
  const index = rows.findIndex((row) => row.public_agent_id === current);
  const step = direction === -1 ? -1 : 1;
  const nextIndex =
    index < 0
      ? step === 1
        ? 0
        : rows.length - 1
      : (index + step + rows.length) % rows.length;
  if (index >= 0 && nextIndex === index) return null;
  const next = rows[nextIndex];
  return pov
    ? Object.freeze({
        command_type: "activate_replay_pov_agent",
        global_slot: next.command_global_slot,
        public_agent_id: next.public_agent_id,
      })
    : Object.freeze({
        command_type: "activate_authorized_agent",
        presentation_key: next.presentation_key,
      });
}

/**
 * Register the Replay Viewer's Tab / Shift+Tab handler ("replay_next_agent"
 * in command-set.js). Call once at page start in the Replay Viewer only.
 *
 * The handler acts only when isFieldTarget(event.target) is true (the
 * battlefield field has focus); otherwise it returns false so Tab moves focus
 * as usual. It then builds replayAgentStepCommand(getPresentation(), Shift
 * ? -1 : 1, getCurrentPublicAgentId()) and, when there is one, calls
 * onCommand(command, {pointerOriginated: false}) and returns true (the
 * replay key dispatcher then stops the key's default focus move).
 *
 * Side effects: one registration in the command set until the returned
 * function runs. Sends nothing itself; onCommand owns the request.
 *
 * @param {{
 *   getPresentation: () => unknown,
 *   getCurrentPublicAgentId?: () => string | null,
 *   isFieldTarget: (target: EventTarget | null) => boolean,
 *   onCommand: (
 *     command: Readonly<Record<string, unknown>>,
 *     context: Readonly<{pointerOriginated: boolean}>,
 *   ) => unknown,
 * }} options getPresentation returns the installed presentation;
 *   getCurrentPublicAgentId (default: always null) returns the page's selected
 *   agent in Oracle View; isFieldTarget says whether a key event came from the
 *   battlefield field; onCommand sends the activation command.
 * @returns {() => void} Removes the registration.
 * @throws {TypeError} When the handler name already has a handler.
 */
export function bindReplayAgentStepping(options) {
  return registerLocalCommandHandler(
    "replay_next_agent",
    /** @param {import("./command-set.js").LocalCommandCall} call */ (call) => {
      const event = /** @type {KeyboardEvent} */ (call.event);
      if (!options.isFieldTarget(event?.target ?? null)) return false;
      const command = replayAgentStepCommand(
        options.getPresentation(),
        call.shiftKey ? -1 : 1,
        options.getCurrentPublicAgentId?.() ?? null,
      );
      if (command === null) return false;
      void options.onCommand(command, Object.freeze({ pointerOriginated: false }));
      return true;
    },
  );
}
