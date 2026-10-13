/**
 * @file Render the docked panels: roster, agent details, the Actions view
 * (This Tick and Next Tick), and the Technical Frame facts.
 *
 * Who calls it: main.js builds one DebuggerPanels with the panel elements and
 * calls render(frame, interactionState) for every installed frame. Only
 * installed, branded presentations supply facts. This module updates DOM nodes
 * and sends activation commands through the caller's onCommand; it never steps
 * the simulator or infers a fact the authorized view does not hold.
 *
 * A champion's life shows as the one shared Life bar (`life-bar.js`: "Life",
 * a thin bar and "87 / 100"), and its statuses in the Champion Card row's
 * words, in card order, with the source named by class only (statusLineV1),
 * the same in Agent POV and Oracle View.
 *
 * Champions are named with token chips (token-chip.js): the class emblem in the
 * team ring with "Agent N". The Panel Tokens switch (and, in the Replay Viewer,
 * Ledger Heraldry) turns chips back into plain "Agent ID N" text. The Status
 * Icons From Marks switch turns roster status icons into plain icons. Actions read
 * in words with the recorded codes underneath ("Moves Northwest · Arcane Blast
 * → Agent 2" over "Move 6 · Target 8 · Ultimate 0"); in the Replay Viewer the
 * Actions in Words switch turns that off and shows today's Submitted and
 * Accepted boxes.
 *
 * Agent POV boundary (plan P5 and P23):
 * - Replay Viewer: the docked panels are the researcher's full record. In Agent
 *   POV they list every champion and carry the chip "Full Record · Not Limited
 *   to Agent N's View". Nothing here is drawn on or over the arena.
 * - BattleClient and DevClient (viewer_mode "live"): in Agent POV every panel
 *   reads only the POV arena scene, its own latest transition and its visual
 *   events. A champion the team cannot see reads "Unseen" with no life, status
 *   or Ultimate; a teammate's action reads "Not Observed"; the global
 *   researcher_space is used only for the list of who is in the match (ID,
 *   class, team), which champions are Manual, and command plumbing. The
 *   Pending card shows the POV agent's own staged order from its own POV
 *   draft; every other staged order (a teammate's or another team's) reads
 *   "Not Observed".
 *
 * Beat clock: a champion's life reaching 0 waits for its Fall beat and a
 * returning champion reappears at its Return beat. The panels mark those nodes
 * with the choreography's data-beat-hold attributes (beatView in
 * choreography-plan.js is the one owner of held values); the choreographer
 * holds them when the panel roots are in its beatHoldRoots.
 */
import { canonicalAgentIdentity, fullRecordChipWords } from "./agent-identity.js";
import {
  authorizedPresentationAudience,
  authorizedPresentationController,
  authorizedPresentationHasResearcherSpace,
  authorizedPresentationIdentityRows,
  authorizedPresentationInspectionState,
  authorizedPresentationLatestTransitionId,
  authorizedPresentationPendingJointActionRows,
  authorizedPresentationResearcherInspectionState,
  authorizedPresentationResearcherSceneView,
  authorizedPresentationSceneView,
  authorizedPresentationTechnicalFacts,
  authorizedPresentationTransitionRows,
  authorizedPresentationUpcomingTransitionRows,
  authorizedPresentationNextActionOwnership,
  isAuthorizedPresentationFrame,
  NOT_OBSERVED_LABEL,
} from "./authorized-presentation-adapter.js";
import { beatView } from "./choreography-plan.js";
import {
  controllerNameNode,
  controllerOfChampion,
  publishedControllerRoster,
} from "./controller-names.js";
import {
  classAbilityNameV1,
  classLoreV1,
  classNamesV1,
} from "./class-documentation.js";
import { formatCompactDisplayNumber, formatDisplayNumber } from "./display.js";
import {
  explainAgent,
  explainClassDocumentation,
  explainModifier,
  explainPovAgent,
  explainPovStatus,
  explainStatus,
} from "./explanations.js";
import { createSvgIcon } from "./icons.js";
import { renderTechnicalFrame } from "./ledger.js";
import { createLifeBar, lifeBarModel } from "./life-bar.js";
import { statusLineV1 } from "./semantic-vocabulary.js";
import { statusOrderIndex } from "./symbols.js";
import { createTokenChip } from "./token-chip.js";
import { registerTooltipOwner, renderSemanticDescriptor } from "./tooltip.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualFilterEnabled,
} from "./visual-filters.js";
import {
  classTokenFromId,
  resolveVisualToken,
  statusTokenIdFromCatalogId,
  teamTokenFromId,
} from "./vocabulary.js";

/**
 * @typedef {{
 *   roster: HTMLElement,
 *   rosterCount: HTMLElement,
 *   selectionCard: HTMLElement,
 *   pendingHeading: HTMLElement,
 *   pendingCount: HTMLElement,
 *   pendingScope: HTMLElement,
 *   pendingCard: HTMLElement,
 *   acceptedCard: HTMLElement,
 *   acceptedAnnouncement: HTMLElement,
 *   diagnosticsCard: HTMLElement,
 *   actionsPanel?: HTMLElement | null,
 *   onCommand: (
 *     command: Record<string, unknown>,
 *     context?: Readonly<{pointerOriginated: boolean}>,
 *   ) => void | Promise<void>,
 * }} DebuggerPanelBindings
 * The panel elements main.js finds by ID. actionsPanel (optional, default
 * null) is the Replay Viewer's Actions tab (#replay-rail-actions): when given,
 * a replay frame draws This Tick and Next Tick together there as well. The
 * older cards (pendingCard for Next Tick, acceptedCard for This Tick) are
 * always kept up to date.
 */

/**
 * @typedef {{
 *   busy?: boolean,
 *   shuttingDown?: boolean,
 *   resyncRequired?: boolean,
 *   offline?: boolean,
 *   activationDisabled?: boolean,
 *   localInspectedPresentationKey?: string | null,
 *   activatedAgentPublicId?: string | null,
 *   combatConfiguration?: Record<string, any> | null,
 *   visualFilterState?: unknown,
 * }} PanelInteractionState
 * visualFilterState (optional) is the page's complete switch state; it decides
 * Panel Tokens, Ledger Heraldry and Actions in Words, and the beat clock's
 * Fall and Return beats. Absent or invalid means the shipped defaults.
 */

/**
 * The movement words by move code, used only when the frame carries no action
 * axis with the host's own names. The host owner is
 * core/axis_mappings.py MOVEMENT_ACTION_NAME_BY_ID; a unit test pins this copy
 * to it.
 */
export const FALLBACK_MOVEMENT_NAMES = Object.freeze([
  "Stay",
  "North",
  "South",
  "East",
  "West",
  "Northeast",
  "Northwest",
  "Southeast",
  "Southwest",
]);

/**
 * How the docked panels draw names and actions for one frame.
 *
 * tokens: champions show as token chips, with each team's crest, while Panel
 * Tokens is on (every client); false gives plain "Agent ID N" text. Actions
 * always read in words with the recorded codes under them, so there is no
 * style for them. Pure.
 *
 * @param {unknown} _presentation The installed presentation. Not read: every
 *   client uses the same rule. Kept so the callers stay unchanged.
 * @param {unknown} [visualFilterState] The page's complete switch state.
 *   Absent or invalid means DEFAULT_VISUAL_FILTER_STATE.
 * @returns {Readonly<{tokens: boolean}>}
 */
export function panelDisplayStyle(
  _presentation,
  visualFilterState = DEFAULT_VISUAL_FILTER_STATE,
) {
  /** @param {string} id */
  const on = (id) => {
    try {
      return isVisualFilterEnabled(visualFilterState, id);
    } catch {
      return isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, id);
    }
  };
  return Object.freeze({
    tokens: on("panel_tokens"),
  });
}

/**
 * Return whether roster status icons are small copies of the marks on the body
 * (the Status Icons From Marks switch, status_marks_icons; on by default in
 * every client). false means the simple plain icons with the same meaning:
 * the icons.js "outline" style (the mark's shapes as thin lines). Pure.
 *
 * @param {unknown} [visualFilterState] The page's complete switch state.
 *   Absent or invalid means DEFAULT_VISUAL_FILTER_STATE.
 * @returns {boolean}
 */
export function statusIconsFromMarks(visualFilterState = DEFAULT_VISUAL_FILTER_STATE) {
  try {
    return isVisualFilterEnabled(visualFilterState, "status_marks_icons");
  } catch {
    return isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, "status_marks_icons");
  }
}

/**
 * Return whether a frame is a BattleClient or DevClient frame in Agent POV,
 * where every panel reads only the POV scene (plan P23). Pure.
 *
 * @param {unknown} presentation
 * @returns {boolean}
 */
export function isLiveAgentPovPanel(presentation) {
  return (
    isAuthorizedPresentationFrame(presentation) &&
    presentation.viewer_mode === "live" &&
    authorizedPresentationAudience(presentation) === "agent_pov"
  );
}

/**
 * Return the Full Record chip words for a Replay Viewer frame in Agent POV,
 * for example "Full Record · Not Limited to Agent 3's View", or null for any
 * other frame. The POV agent comes from the frame's own authority; its label
 * and the words from agent-identity.js (fullRecordChipWords; "This Agent" when
 * the recipient has no identity row), so they match ledger.js and the HUD. Pure.
 *
 * @param {unknown} presentation
 * @returns {string | null}
 */
export function fullRecordLabel(presentation) {
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    presentation.viewer_mode !== "replay" ||
    authorizedPresentationAudience(presentation) !== "agent_pov"
  ) {
    return null;
  }
  const recipientId = presentation.authority?.recipient_public_agent_id;
  const recipient = authorizedPresentationIdentityRows(presentation).find(
    (identity) => identity.public_agent_id === recipientId,
  );
  return fullRecordChipWords(
    recipient === undefined ? null : canonicalAgentIdentity(recipient.agent).chipText,
  );
}

/**
 * Turn one recorded action into plain words and codes. Pure.
 *
 * Words: the move ("Moves Northwest", or "Stays" for move 0), then the
 * ability: the class's Ultimate name when use_ultimate_action is 1, the class's
 * Basic name when a target is set, or "No Combat" when neither. The target is
 * the public ID that targetMap gives for target_action (null for code 0 or a
 * missing map); when the map is missing but the code is not 0, targetCode
 * keeps the code so the caller can print "Target N". Itself is true when the
 * target is the actor.
 *
 * @param {Readonly<Record<string, any>>} action {move_action, target_action,
 *   use_ultimate_action} integer codes.
 * @param {number | null | undefined} classId The actor's class ID (1 to 5).
 * @param {ReadonlyArray<string | null> | null} targetMap Index = target code,
 *   value = the target's public ID or null; from the recorded action row's
 *   target_action_recipient_public_agent_id_by_id. null when not recorded.
 * @param {string | null} actorPublicId The actor's public ID, for "Itself".
 * @param {ReadonlyArray<string> | null} [movementNames] Names by move code;
 *   default FALLBACK_MOVEMENT_NAMES.
 * @returns {Readonly<{
 *   move: string,
 *   ability: string | null,
 *   noCombat: boolean,
 *   targetPublicId: string | null,
 *   targetCode: number | null,
 *   itself: boolean,
 *   codes: string,
 * }>}
 */
export function describeRecordedAction(
  action,
  classId,
  targetMap,
  actorPublicId,
  movementNames = FALLBACK_MOVEMENT_NAMES,
) {
  const names = movementNames ?? FALLBACK_MOVEMENT_NAMES;
  const moveCode = action.move_action;
  const moveName = Number.isInteger(moveCode) ? names[moveCode] : undefined;
  const move =
    moveCode === 0
      ? "Stays"
      : typeof moveName === "string"
        ? `Moves ${moveName}`
        : `Move ${moveCode}`;
  const ultimate = action.use_ultimate_action === 1;
  const targetCode = Number.isInteger(action.target_action) ? action.target_action : 0;
  const mapped =
    Array.isArray(targetMap) && typeof targetMap[targetCode] === "string"
      ? /** @type {string} */ (targetMap[targetCode])
      : null;
  const unmappedTarget =
    mapped === null && targetCode !== 0 && !Array.isArray(targetMap);
  const hasTarget = mapped !== null || unmappedTarget;
  const ability = ultimate
    ? classAbilityNameV1(classId, "ultimate")
    : hasTarget
      ? classAbilityNameV1(classId, "basic")
      : null;
  return Object.freeze({
    move,
    ability,
    noCombat: !ultimate && !hasTarget,
    targetPublicId: mapped,
    targetCode: unmappedTarget ? targetCode : null,
    itself: mapped !== null && mapped === actorPublicId,
    codes: `Move ${action.move_action} · Target ${action.target_action} · Ultimate ${action.use_ultimate_action}`,
  });
}

/**
 * @typedef {{
 *   element: HTMLElement,
 *   primaryButton: HTMLButtonElement,
 *   identityId: HTMLElement,
 *   identityClass: HTMLElement,
 *   controller: HTMLElement,
 *   health: HTMLElement,
 *   statuses: HTMLElement,
 *   modifiers: HTMLElement,
 * }} AuthorizedRosterRow
 */

/**
 * @typedef {{
 *   element: HTMLElement,
 *   count: HTMLElement,
 *   rows: HTMLElement,
 *   empty: HTMLElement,
 * }} RosterTeamGroup
 */

/**
 * Return whether value is a non-null object other than an array. This
 * shallow check accepts class instances and does not validate their fields.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return an agent view with exact neutral multiplier rows removed. Prefer
 * aura_modifiers when it is an array, otherwise modifiers. Copy that array
 * and the outer agent; preserve other references. With neither array, return
 * the original agent. This display filter changes no recorded metric or state.
 *
 * @param {Record<string, any>} agent
 */
function agentForPresentation(agent) {
  const key = Array.isArray(agent.aura_modifiers)
    ? "aura_modifiers"
    : Array.isArray(agent.modifiers)
      ? "modifiers"
      : null;
  if (key === null) {
    return agent;
  }
  return {
    ...agent,
    [key]: agent[key].filter(
      (/** @type {unknown} */ modifier) =>
        !isRecord(modifier) || modifier.multiplier !== 1,
    ),
  };
}

/**
 * Return value unchanged when it is an array, otherwise a new empty array.
 * Elements are neither validated nor copied.
 *
 * @param {unknown} value
 * @returns {any[]}
 */
function asArray(value) {
  return Array.isArray(value) ? value : [];
}

export const DISCLOSURE_PANEL_IDS = Object.freeze([
  "command-deck",
  "roster-details",
  "agent-details",
  "pending-turn-details",
  "latest-transition-details",
  "technical-frame-details",
]);

/**
 * Return the initial open state for a known DISCLOSURE_PANEL_IDS member.
 * Roster Details is open in both modes; Command Deck is also open when replay
 * is false. Every other known panel starts closed. Throw RangeError for an
 * unknown panelId. This does not read or update a user's saved panel state.
 *
 * @param {string} panelId
 * @param {boolean} replay
 */
export function disclosurePanelInitiallyOpen(panelId, replay) {
  if (!DISCLOSURE_PANEL_IDS.includes(panelId)) {
    throw new RangeError(`Unknown disclosure panel ${panelId}.`);
  }
  return panelId === "roster-details" || (!replay && panelId === "command-deck");
}

/**
 * Replace container's children with a hidden title and full details view.
 * Pass descriptor to the shared semantic renderer and return its result.
 * The caller owns visibility, closing and focus return. This replaces DOM
 * content but does not change descriptor or register a new data authority.
 *
 * @param {HTMLElement} container
 * @param {unknown} descriptor
 */
export function renderSemanticInspector(container, descriptor) {
  const title = htmlElement("span", "sr-only");
  const details = htmlElement("div", "semantic-inspector__details");
  container.replaceChildren(title, details);
  return renderSemanticDescriptor({
    descriptor,
    title,
    details,
    surface: "full",
  });
}

/**
 * Return the sole class-mechanics row matching owner's exact numeric ID
 * and class name. Return null for a non-array bank, invalid owner identity,
 * no match or multiple matches. Do not coerce values or copy the matched row.
 * Both inputs must already belong to the same authorized presentation.
 *
 * @param {Record<string, any>} owner
 * @param {unknown} rawClassMechanics
 */
function exactOwnerClassMechanics(owner, rawClassMechanics) {
  if (
    !Array.isArray(rawClassMechanics) ||
    !Number.isSafeInteger(owner.class_id) ||
    typeof owner.class_name !== "string"
  ) {
    return null;
  }
  const matches = rawClassMechanics.filter(
    (candidate) =>
      isRecord(candidate) &&
      candidate.class_id === owner.class_id &&
      candidate.class_name === owner.class_name,
  );
  return matches.length === 1 ? matches[0] : null;
}

/**
 * Build the selected agent's class-details view from a branded presentation.
 * Return null when presentation/audience/scene is unavailable. Replay frames
 * (the docked full record) and live Oracle frames with a researcher branch use
 * the researcher scene. Live Agent POV frames (BattleClient and DevClient) use
 * only the POV arena scene (plan P23): a selected champion the team cannot see
 * gives owner null, so its details read "Unseen" until it is seen again; the
 * selection itself is kept by the caller. Other frames use the scene selected
 * by localInspectedPresentationKey (undefined uses its default).
 * A nonempty activatedAgentPublicId selects that exact public identity;
 * otherwise use the scene's inspection owner. Missing or ambiguous class
 * mechanics leaves owner_descriptor null. The owner never carries a position
 * field (an owner taken from an arena scene is copied without it). Return a
 * frozen outer record with referenced owner data; this does not grant
 * battlefield or command access.
 *
 * @param {unknown} presentation
 * @param {string | null | undefined} [localInspectedPresentationKey]
 * @param {string | null | undefined} [activatedAgentPublicId]
 * @returns {Readonly<Record<string, any>> | null}
 */
export function authorizedInspectorView(
  presentation,
  localInspectedPresentationKey = undefined,
  activatedAgentPublicId = undefined,
) {
  if (!isAuthorizedPresentationFrame(presentation)) {
    return null;
  }
  const scene =
    !isLiveAgentPovPanel(presentation) &&
    (presentation.viewer_mode === "replay" ||
      authorizedPresentationHasResearcherSpace(presentation))
      ? authorizedPresentationResearcherSceneView(presentation)
      : authorizedPresentationSceneView(presentation, localInspectedPresentationKey);
  const audience = authorizedPresentationAudience(presentation);
  if (scene === null || (audience !== "researcher" && audience !== "agent_pov")) {
    return null;
  }

  const agents = asArray(scene.agents).filter(isRecord);
  const selection = isRecord(scene.selection) ? scene.selection : {};
  const ownerKey =
    typeof selection.inspection_owner_presentation_key === "string"
      ? selection.inspection_owner_presentation_key
      : null;
  const sceneOwner =
    typeof activatedAgentPublicId === "string" && activatedAgentPublicId.length > 0
      ? (agents.find((agent) => agent.public_agent_id === activatedAgentPublicId) ??
        null)
      : ownerKey === null
        ? null
        : (agents.find((agent) => agent.presentation_key === ownerKey) ?? null);
  // The panels never carry battlefield geometry, even for a seen champion.
  const owner =
    sceneOwner === null || !Object.hasOwn(sceneOwner, "position")
      ? sceneOwner
      : Object.freeze(
          Object.fromEntries(
            Object.entries(sceneOwner).filter(([field]) => field !== "position"),
          ),
        );
  const ownerMechanics =
    owner === null ? null : exactOwnerClassMechanics(owner, scene.class_mechanics);
  const ownerDescriptor =
    owner === null || ownerMechanics === null
      ? null
      : explainClassDocumentation(owner, ownerMechanics);
  const ownerClassAccent =
    owner === null ? null : classTokenFromId(owner.class_id).cssKey;

  return Object.freeze({
    title: "Comprehensive Agent Class Details",
    owner,
    owner_descriptor: ownerDescriptor,
    owner_class_accent: ownerClassAccent,
  });
}

/**
 * Create an unattached element of tagName. Set className only when truthy
 * and textContent when text is not null; both default to null. Return the
 * element. Text is inserted as text, not parsed as HTML.
 *
 * @template {keyof HTMLElementTagNameMap} K
 * @param {K} tagName
 * @param {string | null} className
 * @param {string | null} text
 * @returns {HTMLElementTagNameMap[K]}
 */
function htmlElement(tagName, className = null, text = null) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== null) {
    element.textContent = text;
  }
  return element;
}

/**
 * Append one labelled fact row to container and return its element.
 * Convert value with String and insert both fields as text. This performs no
 * scientific validation or tooltip registration.
 *
 * @param {HTMLElement} container
 * @param {string} label
 * @param {unknown} value
 * @returns {HTMLElement}
 */
function addFact(container, label, value) {
  const fact = htmlElement("div", "fact");
  fact.append(
    htmlElement("span", null, label),
    htmlElement("strong", null, String(value)),
  );
  container.append(fact);
  return fact;
}

/**
 * Create the unattached "Pending" card of a staged Manual order: its recorded
 * codes, read from move_action, target_action and use_ultimate_action without
 * recomputing acceptance. Return the section; inputs must be authorized rows.
 * (Submitted and Accepted rows read in words instead; see
 * authorizedTransitionList.)
 *
 * @param {"Pending"} label
 * @param {Record<string, any>} action
 */
function authorizedTransitionTuple(label, action) {
  const tuple = htmlElement("section", "accepted-action-tuple");
  tuple.dataset.kind = label.toLowerCase();
  tuple.append(
    htmlElement("h4", null, label),
    htmlElement(
      "p",
      "accepted-action-tuple__value",
      `Move ${action.move_action} · Target ${action.target_action} · Ultimate ${action.use_ultimate_action}`,
    ),
  );
  return tuple;
}

/**
 * One action row ready to draw: the adapter's authorized row plus the actor's
 * identity record and the recorded target map, when they could be joined.
 *
 * @typedef {{
 *   row: Readonly<Record<string, any>>,
 *   actor: Readonly<Record<string, any>> | null,
 *   targetMap: ReadonlyArray<string | null> | null,
 *   notObserved?: boolean,
 * }} ActionRowView
 */

/**
 * How one action list draws its rows.
 *
 * @typedef {{
 *   pending?: boolean,
 *   tokens: boolean,
 *   identityByPublicId: ReadonlyMap<string, Readonly<Record<string, any>>>,
 *   movementNames: ReadonlyArray<string> | null,
 *   onPick: ((publicAgentId: string) => void) | null,
 * }} ActionListOptions
 */

/**
 * Create a champion name node for a panel: a token chip when tokens is true,
 * otherwise plain text (the agent-identity.js label). Never reads live facts.
 *
 * @param {Readonly<Record<string, any>> | null} agent Identity record.
 * @param {boolean} tokens Draw a token chip (true) or plain text (false).
 * @param {((publicAgentId: string) => void) | null} onPick Click handler that
 *   picks the champion, or null for a chip that is not clickable.
 * @param {string | null} [label] Words that replace "Agent N" (for "Itself").
 * @returns {HTMLElement}
 */
function championName(agent, tokens, onPick, label = null) {
  return createTokenChip(document, agent, {
    plain: !tokens,
    label,
    onPick: onPick === null ? null : (publicAgentId) => onPick(publicAgentId),
  });
}

/**
 * Create an unattached ordered list of action rows.
 *
 * Each row shows the actor (chip or plain name, with its class), the action
 * in words with the target as a chip, the recorded codes underneath, and
 * "Changed by the Game" with the submitted codes when the game changed the
 * order. With pending true the codes are the staged Manual order and the row
 * keeps one "Pending" box. A row marked notObserved reads "Not Observed" (in
 * BattleClient and DevClient Agent POV: a teammate's last action, or another
 * team's staged Manual order) with no codes. Input order and labels are kept;
 * rows are not changed.
 *
 * @param {ReadonlyArray<ActionRowView>} rows
 * @param {ActionListOptions} options
 * @returns {HTMLOListElement}
 */
function authorizedTransitionList(rows, options) {
  const pending = options.pending === true;
  const list = htmlElement("ol", "accepted-action-list");
  for (const view of rows) {
    const transition = view.row;
    const item = htmlElement("li", "accepted-action-row");
    item.dataset.team = transition.actor_team;
    const title = htmlElement("h3", "accepted-action-row__title");
    title.dataset.class = transition.actor_accent;
    if (options.tokens && view.actor !== null) {
      title.append(
        championName(view.actor, true, options.onPick),
        htmlElement(
          "span",
          "accepted-action-row__class",
          canonicalAgentIdentity(view.actor).classLabel,
        ),
      );
    } else {
      title.textContent = transition.actor_title;
    }
    item.append(title);
    if (view.notObserved === true) {
      item.dataset.observed = "false";
      item.append(htmlElement("p", "accepted-action-row__words", NOT_OBSERVED_LABEL));
      list.append(item);
      continue;
    }
    const action = pending ? transition.pending_action : transition.accepted_action;
    const described = describeRecordedAction(
      action,
      view.actor?.class_id ?? null,
      view.targetMap,
      view.actor?.public_agent_id ?? null,
      options.movementNames,
    );
    const words = htmlElement("p", "accepted-action-row__words");
    const lead = described.noCombat
      ? `${described.move} · No Combat`
      : `${described.move} · ${described.ability ?? "Ability"}`;
    words.append(htmlElement("span", "accepted-action-row__lead", lead));
    const target =
      described.targetPublicId === null
        ? null
        : (options.identityByPublicId.get(described.targetPublicId) ?? null);
    if (target !== null) {
      words.append(
        htmlElement("span", "accepted-action-row__arrow", " → "),
        championName(
          target,
          options.tokens,
          options.onPick,
          described.itself ? "Itself" : null,
        ),
      );
    } else if (described.targetCode !== null) {
      words.append(
        htmlElement(
          "span",
          "accepted-action-row__arrow",
          ` → Target ${described.targetCode}`,
        ),
      );
    }
    const codes = htmlElement("p", "accepted-action-row__codes", described.codes);
    item.append(words, codes);
    if (pending) {
      const comparison = htmlElement("div", "accepted-action-row__comparison");
      comparison.dataset.layout = "single";
      comparison.append(authorizedTransitionTuple("Pending", action));
      item.append(comparison);
    } else {
      const submitted = transition.submitted_action;
      const changed =
        submitted.move_action !== action.move_action ||
        submitted.target_action !== action.target_action ||
        submitted.use_ultimate_action !== action.use_ultimate_action;
      if (changed) {
        item.dataset.changed = "true";
        item.append(
          htmlElement("span", "accepted-action-row__changed", "Changed by the Game"),
          htmlElement(
            "p",
            "accepted-action-row__submitted",
            `Submitted: Move ${submitted.move_action} · Target ${submitted.target_action} · Ultimate ${submitted.use_ultimate_action}`,
          ),
        );
      }
    }
    list.append(item);
  }
  return list;
}

/**
 * Create the "Fallen · No Action" line: one name per champion that takes no
 * action this tick because it is fallen. Return null for an empty list.
 *
 * @param {ReadonlyArray<Readonly<Record<string, any>>>} fallen Identity records.
 * @param {boolean} tokens Chips (true) or plain names (false).
 * @param {((publicAgentId: string) => void) | null} onPick Chip click handler.
 * @returns {HTMLElement | null}
 */
function fallenNoActionLine(fallen, tokens, onPick) {
  if (fallen.length === 0) return null;
  const line = htmlElement("p", "accepted-action-fallen");
  line.append(
    htmlElement("span", "accepted-action-fallen__label", "Fallen · No Action"),
  );
  for (const agent of fallen) {
    line.append(championName(agent, tokens, onPick));
  }
  return line;
}

/**
 * Return the recorded action rows of one transition branch, with the same
 * branch choice as the adapter: the attached researcher branch first (Replay
 * Viewer Agent POV and live Oracle with researcher space), otherwise the
 * frame's own branch. Used only to join each adapter row to its actor public ID
 * and target map, in the same order. Returns [] when absent.
 *
 * @param {Readonly<Record<string, any>>} presentation Approved presentation.
 * @param {"latest_transition" | "upcoming_transition" | "pending_joint_action"} field
 *   Which branch.
 * @param {boolean} ownBranchOnly true reads only the frame's own branch (the
 *   POV audience's), never researcher_space.
 * @returns {ReadonlyArray<Readonly<Record<string, any>>>}
 */
function recordedActionRows(presentation, field, ownBranchOnly) {
  const researcher =
    !ownBranchOnly && authorizedPresentationHasResearcherSpace(presentation)
      ? presentation.researcher_space
      : null;
  const transition = researcher?.[field] ?? presentation[field];
  return isRecord(transition) && Array.isArray(transition.action_rows)
    ? transition.action_rows.filter(isRecord)
    : [];
}

/**
 * Return the move names a frame carries (its action axis), or null when it
 * has none, so describeRecordedAction falls back to FALLBACK_MOVEMENT_NAMES.
 *
 * @param {Readonly<Record<string, any>>} presentation
 * @returns {ReadonlyArray<string> | null}
 */
function frameMovementNames(presentation) {
  const rows = presentation.action_axis?.movement_actions;
  if (!Array.isArray(rows) || rows.length === 0) return null;
  const names = [...FALLBACK_MOVEMENT_NAMES];
  for (const row of rows) {
    if (
      isRecord(row) &&
      Number.isInteger(row.move_action) &&
      typeof row.display_name === "string"
    ) {
      names[row.move_action] = row.display_name;
    }
  }
  return Object.freeze(names);
}

/**
 * Read the beat view for a frame and never throw: the panels must keep
 * working even if the beat clock cannot read a frame. Returns null on failure.
 *
 * @param {unknown} presentation
 * @param {unknown} visualFilterState
 * @returns {ReturnType<typeof beatView>}
 */
function safeBeatView(presentation, visualFilterState) {
  try {
    return beatView(
      presentation,
      "end",
      visualFilterState === undefined ? {} : { visualFilters: visualFilterState },
    );
  } catch {
    return null;
  }
}

/**
 * Return the compact display form of an integer duration, otherwise ?.
 * This accepts any JavaScript integer, including negative values; validation
 * of legal durations belongs upstream. Exact duration remains in chip data
 * and accessible text when renderFactTokens builds the chip.
 *
 * @param {unknown} duration
 */
export function rosterStatusDurationLabel(duration) {
  return Number.isInteger(duration) ? formatCompactDisplayNumber(duration) : "?";
}

/**
 * Replace container's children with status or modifier chips in input order.
 * Hide exact multiplier=1 modifier rows; show emptyText if no chips remain.
 * Use recipient and authorized sourceAgents only for explanation text.
 * agent_pov status chips use the restricted explanation route. Register
 * tooltips and keyboard focus on each chip; do not derive combat mechanics
 * or mutate the input records. kind and audience must be the declared values.
 *
 * @param {HTMLElement} container
 * @param {unknown[]} items
 * @param {"status" | "modifier"} kind
 * @param {string} emptyText
 * @param {Record<string, any>} recipient
 * @param {ReadonlyArray<unknown>} sourceAgents
 * @param {"researcher" | "agent_pov"} audience
 * @param {boolean} [markIcons] Status icons are small copies of the marks on
 *   the body (true, the default) or simple plain icons (false, the Status
 *   Icons From Marks switch off). Modifier chips have no icon and ignore it.
 */
function renderFactTokens(
  container,
  items,
  kind,
  emptyText,
  recipient,
  sourceAgents,
  audience,
  markIcons = true,
) {
  const nodes = [];
  const authorizedItems =
    kind === "modifier"
      ? items.filter((item) => !isRecord(item) || item.multiplier !== 1)
      : items;
  for (const rawItem of authorizedItems) {
    const item = isRecord(rawItem) ? rawItem : {};
    const token = resolveVisualToken(
      kind,
      item.token_id,
      audience === "agent_pov" && kind === "status" ? undefined : item,
    );
    const value =
      kind === "status"
        ? `duration ${Number.isInteger(item.duration) ? item.duration : "unknown"}`
        : `multiplier ${formatDisplayNumber(item.multiplier)}`;
    const chip = htmlElement("span", `roster-fact-token roster-fact-token--${kind}`);
    chip.dataset.tokenId = token.tokenId;
    if (kind === "status") {
      const displayDuration = rosterStatusDurationLabel(item.duration);
      const sourceClass = classTokenFromId(item.source_class_id);
      const icon = createSvgIcon(container.ownerDocument, token.glyphKey, {
        className: "roster-fact-token__icon",
        style: markIcons ? "sigil" : "outline",
      });
      const durationValue = htmlElement(
        "span",
        "roster-fact-token__duration",
        String(displayDuration),
      );
      chip.dataset.icon = token.glyphKey;
      chip.dataset.sourceClass = sourceClass.cssKey;
      if (Number.isInteger(item.duration)) {
        chip.dataset.duration = String(item.duration);
        chip.dataset.visibleValueAbbreviated = String(
          String(displayDuration) !== String(item.duration),
        );
      }
      chip.append(icon, durationValue);
    } else {
      chip.textContent = `${token.shortLabel} ×${formatDisplayNumber(item.multiplier)}`;
    }
    if (kind === "modifier" && Number.isFinite(item.multiplier)) {
      chip.dataset.multiplier = String(item.multiplier);
    }
    chip.setAttribute("aria-label", `${token.accessibleName}, ${value}`);
    chip.tabIndex = 0;
    registerTooltipOwner(
      chip,
      kind === "status"
        ? audience === "agent_pov"
          ? explainPovStatus(item, recipient)
          : explainStatus(item, recipient, sourceAgents)
        : explainModifier(item, recipient),
    );
    nodes.push(chip);
  }
  if (nodes.length === 0) {
    nodes.push(htmlElement("span", "roster-fact-empty", emptyText));
  }
  container.replaceChildren(...nodes);
}

/**
 * Return a frozen copy of an identity record with only its identity fields:
 * presentation_key, public_agent_id, class_id, class_name, team_id and
 * display_agent_id. Used for a champion the BattleClient's POV team cannot
 * see, so no life, status or position can reach the panel. Pure.
 *
 * @param {Readonly<Record<string, any>>} agent
 * @returns {Readonly<Record<string, any>>}
 */
function identityOnlyAgent(agent) {
  return Object.freeze({
    presentation_key: agent.presentation_key,
    public_agent_id: agent.public_agent_id,
    class_id: agent.class_id,
    class_name: agent.class_name,
    team_id: agent.team_id,
    display_agent_id: agent.display_agent_id ?? null,
  });
}

/**
 * Describe a champion the team cannot see, for its roster tip: its label and
 * one sentence, nothing live. Pure.
 *
 * @param {Readonly<Record<string, any>>} agent Identity-only record.
 * @returns {unknown} A semantic descriptor for registerTooltipOwner.
 */
function explainUnseenChampion(agent) {
  const identity = canonicalAgentIdentity(agent);
  return {
    kind: "agent",
    id: `unseen-champion:${String(agent.public_agent_id)}`,
    title: `${identity.title} · Unseen`,
    tone: "information",
    accent: identity.accent,
    summary:
      "Your team cannot see this champion now. Its life, statuses and Ultimate stay hidden until your team sees it again.",
    rows: [],
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "element",
  };
}

/**
 * Map each public ID to its presentation key in the frame's own arena scene
 * (the audience's scene), the key the beat clock uses for data-beat-key. In
 * Agent POV only seen champions are listed. Pure; reads keys only.
 *
 * @param {Readonly<Record<string, any>>} presentation
 * @returns {Map<string, string>}
 */
function audienceSceneKeys(presentation) {
  const scene = authorizedPresentationSceneView(presentation);
  /** @type {Map<string, string>} */
  const keys = new Map();
  for (const agent of asArray(scene?.agents)) {
    if (
      isRecord(agent) &&
      typeof agent.public_agent_id === "string" &&
      typeof agent.presentation_key === "string"
    ) {
      keys.set(agent.public_agent_id, agent.presentation_key);
    }
  }
  return keys;
}

/**
 * Set or clear one node's beat hold attributes (the choreography DOM
 * contract): data-beat-hold, data-beat-key and data-beat-side. kind null or
 * key null removes all three, so a reused node never keeps an old hold.
 *
 * @param {Element} element
 * @param {"fall" | "return" | null} kind
 * @param {string | null} key Presentation key in the audience scene.
 * @param {"before" | "after"} side
 * @returns {void}
 */
function setBeatHold(element, kind, key, side) {
  if (kind === null || key === null) {
    element.removeAttribute("data-beat-hold");
    element.removeAttribute("data-beat-key");
    element.removeAttribute("data-beat-side");
    return;
  }
  element.setAttribute("data-beat-hold", kind);
  element.setAttribute("data-beat-key", key);
  element.setAttribute("data-beat-side", side);
}

/**
 * Write a champion's life (and, with withUltimate, its Ultimate) into
 * element: the shared Life bar of `life-bar.js` ("Life", a thin bar and
 * "87 / 100", owner, 2 October 2026), then, with withUltimate, the words
 * "Ultimate Ready in 3 Ticks" in a `span.panel-life__ultimate`; "Fallen" for
 * a fallen champion. A record without life numbers falls back to the words
 * "Life — / —". When fall is a beat-view entry (the champion fell this tick
 * and the audience saw it), two parts share one grid cell: the Life bar at
 * the life before the tick, marked data-beat-side "before", and "Fallen",
 * marked "after", both with data-beat-hold "fall", so "Fallen" waits for the
 * Fall beat. Without the choreographer only the END value shows. Side
 * effects: replaces element's children only.
 *
 * @param {HTMLElement} element The node to fill; its children are replaced.
 * @param {Readonly<Record<string, any>>} agent Authorized agent facts.
 * @param {Readonly<Record<string, any>> | null} fall beatView life entry
 *   ({presentationKey, before, after}) or null.
 * @param {boolean} withUltimate Add the Ultimate part.
 * @returns {void}
 */
function renderLifeText(element, agent, fall, withUltimate) {
  const maximum = agent.max_health ?? agent.maximum_health;
  const fallen = agent.alive === false || agent.life_state === "corpse";
  const ultimate = Number.isInteger(agent.ultimate_cooldown_remaining)
    ? agent.ultimate_cooldown_remaining === 0
      ? "Ultimate Ready"
      : `Ultimate Ready in ${agent.ultimate_cooldown_remaining} ${agent.ultimate_cooldown_remaining === 1 ? "Tick" : "Ticks"}`
    : "Ultimate —";
  /**
   * The Life bar for one life value, or its words when a number is missing.
   *
   * @param {unknown} current
   * @returns {Node}
   */
  const life = (current) => {
    const model = lifeBarModel(current, maximum);
    return model === null
      ? element.ownerDocument.createTextNode(
          `Life ${formatDisplayNumber(current)} / ${formatDisplayNumber(maximum)}`,
        )
      : createLifeBar(element.ownerDocument, model);
  };
  element.replaceChildren();
  if (fall !== null && typeof fall.presentationKey === "string") {
    const pair = htmlElement("span", "panel-beat");
    const before = htmlElement("span", "panel-beat__before");
    before.append(life(fall.before));
    const after = htmlElement("span", "panel-beat__after", "Fallen");
    setBeatHold(before, "fall", fall.presentationKey, "before");
    setBeatHold(after, "fall", fall.presentationKey, "after");
    pair.append(before, after);
    element.append(pair);
    return;
  }
  if (fallen) {
    element.textContent = "Fallen";
    return;
  }
  element.append(life(agent.current_health));
  if (withUltimate) {
    element.append(htmlElement("span", "panel-life__ultimate", ultimate));
  }
}

/**
 * Create an unattached agent details section with an h4 heading. Pure DOM.
 *
 * @param {string} title Title Case heading, for example "Right Now".
 * @returns {HTMLElement}
 */
function detailSection(title) {
  const part = htmlElement("section", "agent-details__section");
  part.append(htmlElement("h4", "agent-details__heading", title));
  return part;
}

/**
 * Append one labelled fact row whose value is a node (a chip or a beat pair)
 * and return the row. Like addFact, but the value is not text.
 *
 * @param {HTMLElement} container
 * @param {string} label
 * @param {HTMLElement} node
 * @returns {HTMLElement}
 */
function addNodeFact(container, label, node) {
  const fact = htmlElement("div", "fact");
  fact.append(htmlElement("span", null, label), node);
  container.append(fact);
  return fact;
}

/**
 * Join the adapter's authorized action rows to each actor's identity record
 * and recorded target map, by position in the same recorded branch. When the
 * branch and the adapter rows do not line up one to one, every row keeps
 * actor null and targetMap null, so only codes and the adapter's labels show.
 * Pure; no row is changed.
 *
 * @param {Readonly<Record<string, any>>} presentation Approved presentation.
 * @param {ReadonlyArray<Readonly<Record<string, any>>>} adapterRows Rows from
 *   authorizedPresentationTransitionRows, ...UpcomingTransitionRows or
 *   ...PendingJointActionRows for the same frame.
 * @param {"latest_transition" | "upcoming_transition" | "pending_joint_action"} field
 *   The branch those rows came from.
 * @param {ReadonlyMap<string, Readonly<Record<string, any>>>} identityByPublicId
 * @returns {ActionRowView[]}
 */
function joinActionRows(presentation, adapterRows, field, identityByPublicId) {
  const raw = recordedActionRows(presentation, field, false);
  const aligned = raw.length === adapterRows.length;
  return adapterRows.map((row, index) => {
    const recorded = aligned ? raw[index] : null;
    const targetMap = recorded?.target_action_recipient_public_agent_id_by_id;
    return {
      row,
      actor:
        recorded === null
          ? null
          : (identityByPublicId.get(recorded.actor_public_agent_id) ?? null),
      targetMap: Array.isArray(targetMap) ? targetMap : null,
    };
  });
}

/**
 * Map public IDs to the researcher scene's agents (the docked full record in
 * the Replay Viewer; the ordinary scene when the frame has no researcher
 * branch). Never used for anything drawn over the arena. Pure.
 *
 * @param {Readonly<Record<string, any>>} presentation
 * @returns {Map<string, Readonly<Record<string, any>>>}
 */
function researcherSceneAgentsByPublicId(presentation) {
  const scene = authorizedPresentationResearcherSceneView(presentation);
  return new Map(
    asArray(scene?.agents)
      .filter(isRecord)
      .map((agent) => [String(agent.public_agent_id), agent]),
  );
}

/**
 * Return whether a recorded action row belongs on the "Fallen · No Action"
 * line instead of a row of its own. The recorded order must be all zeros.
 * For Next Tick ("next"): the actor is fallen in this frame. For This Tick
 * ("this"): the actor was fallen when the tick began, which is true when it
 * is fallen now and did not fall during this tick, or when it returned during
 * this tick (beats from beatView). Pure.
 *
 * @param {ActionRowView} view
 * @param {ReadonlyMap<string, Readonly<Record<string, any>>>} sceneAgents
 * @param {ReturnType<typeof beatView>} beats null for "next".
 * @param {"this" | "next"} which
 * @returns {boolean}
 */
function isFallenNoAction(view, sceneAgents, beats, which) {
  const publicId = view.actor?.public_agent_id;
  const action = view.row.accepted_action;
  if (
    typeof publicId !== "string" ||
    !isRecord(action) ||
    action.move_action !== 0 ||
    action.target_action !== 0 ||
    action.use_ultimate_action !== 0
  ) {
    return false;
  }
  const agent = sceneAgents.get(publicId);
  if (agent === undefined) return false;
  const fallenNow = agent.alive === false || agent.life_state === "corpse";
  if (which === "next") return fallenNow;
  const fellThisTick = (beats?.life ?? []).some(
    (entry) => entry.publicAgentId === publicId,
  );
  const returnedThisTick = (beats?.returns ?? []).some(
    (entry) => entry.publicAgentId === publicId,
  );
  return (fallenNow && !fellThisTick) || returnedThisTick;
}

/**
 * Build This Tick for a BattleClient or DevClient frame in Agent POV (plan
 * P23, s11 row 8): the POV agent's own recorded action from the frame's own
 * latest transition, and one "Not Observed" row for each teammate in the POV
 * scene. Enemies are not listed. Reads only the POV scene, the frame's own
 * transition branch and its authority; never researcher_space. Pure.
 *
 * @param {Readonly<Record<string, any>>} presentation Approved live Agent POV
 *   presentation.
 * @returns {ActionRowView[]} In POV scene order.
 */
function livePovTransitionRows(presentation) {
  const recipientId = presentation.authority?.recipient_public_agent_id;
  const own = recordedActionRows(presentation, "latest_transition", true).find(
    (row) => row.actor_public_agent_id === recipientId,
  );
  const scene = authorizedPresentationSceneView(presentation);
  const sceneAgents = new Map(
    asArray(scene?.agents)
      .filter(isRecord)
      .map((agent) => [String(agent.public_agent_id), agent]),
  );
  /** @type {ActionRowView[]} */
  const views = [];
  for (const owner of authorizedPresentationNextActionOwnership(presentation)) {
    const sceneAgent = sceneAgents.get(owner.public_agent_id);
    if (sceneAgent === undefined) continue;
    const actor = identityOnlyAgent(sceneAgent);
    const base = {
      actor_title: owner.actor_title,
      actor_accent: owner.actor_accent,
      actor_team: owner.actor_team,
    };
    if (!owner.observed) {
      views.push({ row: base, actor, targetMap: null, notObserved: true });
      continue;
    }
    const submitted = own?.submitted_action;
    const accepted = own?.accepted_action;
    if (
      !isRecord(submitted) ||
      !isRecord(accepted) ||
      ![...Object.values(submitted), ...Object.values(accepted)].every(Number.isInteger)
    ) {
      continue;
    }
    const targetMap = own?.target_action_recipient_public_agent_id_by_id;
    views.push({
      row: { ...base, submitted_action: submitted, accepted_action: accepted },
      actor,
      targetMap: Array.isArray(targetMap) ? targetMap : null,
    });
  }
  return views;
}

/**
 * Build the Pending Manual Actions rows for a BattleClient or DevClient frame
 * in Agent POV (owner's Agent POV rule: a panel in Agent POV never reads the
 * researcher branch for what it shows). The POV agent's own row takes its
 * staged order from the frame's own POV branch: the audience's live draft
 * (authorizedPresentationInspectionState, the same draft the deck and the
 * arena's Order Studs use), written as the triple the host would send
 * (move_action; target_action, or 0 when no action is armed;
 * use_ultimate_action 1 for an armed Ultimate, else 0). Every other row (a
 * teammate or another team's champion) keeps only the actor's identity
 * (title, class accent, team) and is marked notObserved, so it reads "Not
 * Observed" with no move, target or code: the POV branch carries no staged
 * order but the POV agent's own, exactly as This Tick lists a teammate's
 * last action. The POV agent's row is also Not Observed when the frame has
 * no editable draft for it.
 *
 * `views` decide only which rows exist and their identity (who is Manual:
 * a setup fact, the same in every view); their staged values are never
 * copied, so changing the researcher branch's staged orders changes nothing
 * here. Rows are returned in input order and their count is unchanged; the
 * inputs are not changed. Pure.
 *
 * @param {Readonly<Record<string, any>>} presentation Approved live Agent POV
 *   presentation.
 * @param {ReadonlyArray<ActionRowView>} views Pending rows from joinActionRows
 *   with field "pending_joint_action".
 * @returns {ActionRowView[]}
 */
function livePovPendingRows(presentation, views) {
  const recipientId = presentation.authority?.recipient_public_agent_id;
  const inspectionState = authorizedPresentationInspectionState(presentation);
  const draft =
    inspectionState.state_kind === "live_editable" &&
    inspectionState.inspection?.actor_public_agent_id === recipientId &&
    isRecord(inspectionState.inspection?.draft_action)
      ? inspectionState.inspection.draft_action
      : null;
  return views.map((view) => {
    const identity = {
      actor_title: view.row.actor_title,
      actor_accent: view.row.actor_accent,
      actor_team: view.row.actor_team,
    };
    if (
      draft !== null &&
      typeof recipientId === "string" &&
      view.actor?.public_agent_id === recipientId
    ) {
      const targetAction = Number.isInteger(draft.target_action)
        ? Number(draft.target_action)
        : 0;
      const lane =
        draft.armed_lane === "ultimate"
          ? "ultimate"
          : draft.armed_lane === "basic" && targetAction > 0
            ? "basic"
            : "none";
      return {
        row: {
          ...identity,
          pending_action: {
            move_action: Number.isInteger(draft.move_action)
              ? Number(draft.move_action)
              : 0,
            target_action: lane === "none" ? 0 : targetAction,
            use_ultimate_action: lane === "ultimate" ? 1 : 0,
          },
        },
        actor: view.actor,
        targetMap: null,
      };
    }
    return { row: identity, actor: view.actor, targetMap: null, notObserved: true };
  });
}

/**
 * Write one status line for the agent details by the owner's status rule
 * (statusLineV1): the Champion Card row's words, "Charge (Slow) · 5 ticks
 * left · It moves at ×0.5 speed. From a Warrior." The source is named by
 * class only (owner ruling, 2 October 2026), so the line is the same in
 * Agent POV and Oracle View and never names an Agent ID or "a hidden
 * source". Pure.
 *
 * @param {Readonly<Record<string, any>>} owner The agent with the status (its
 *   class gives "From its own aura.").
 * @param {Readonly<Record<string, any>>} status One status record.
 * @returns {string}
 */
function statusLine(owner, status) {
  return statusLineV1({
    // Scene statuses carry the catalogue status ID; the status line needs the
    // vocabulary token (for example "Blessing of Freedom (Slow Resistance)",
    // not "Recorded Status").
    tokenId: statusTokenIdFromCatalogId(status.status_id ?? status.token_id),
    magnitude: status.magnitude,
    remainingTicks: status.remaining_duration ?? status.duration,
    recipientClassId: owner.class_id,
  });
}

/**
 * Write a champion count for the roster: "1 Champion", "5 Champions". Pure.
 *
 * @param {number} count Whole number of champions, 0 or more.
 * @returns {string}
 */
function championsText(count) {
  return `${count} ${count === 1 ? "Champion" : "Champions"}`;
}

/**
 * Return who played one champion in a saved game, for the Replay Viewer
 * roster's "Played by" line: the recorded controller name from the page's
 * one controller model (controller-names.js publishedControllerRoster, which
 * the HUD publishes from every installed replay frame), shown verbatim.
 * Live play keeps its launch configuration (authorizedPresentationController)
 * and never reaches this. Reads the document's published roster only.
 *
 * @param {string} publicAgentId The champion's public agent ID.
 * @returns {Readonly<{manual: boolean, label: string}> | null} The recorded
 *   name and whether a person played it, or null when no roster lists the
 *   champion (old replays without names, unit tests without a document).
 */
function recordedController(publicAgentId) {
  if (typeof document === "undefined") return null;
  const found = controllerOfChampion(publishedControllerRoster(document), {
    publicAgentId,
  });
  return found === null
    ? null
    : Object.freeze({ manual: found.controller.manual, label: found.controller.name });
}

/**
 * Keep debugger panel DOM nodes in sync with installed presentations.
 * Reuse roster rows by display identity to preserve focus where possible.
 * The caller supplies all DOM bindings and an activation command handler.
 * This class owns panel updates, not transport, simulation or authorization.
 * Call render for each installed frame; invalid frames clear scientific views.
 */
export class DebuggerPanels {
  /**
   * Store required DOM bindings, clear roster and create its team/visibility
   * groups. Initialize row caches and transition-announcement state. onCommand
   * receives activation records and pointer-origin context; returned promises
   * are not awaited here. Bindings must be real, compatible DOM elements; the
   * optional actionsPanel (default null) also receives the combined Replay
   * Viewer Actions view. Construction changes the supplied roster immediately
   * and returns the instance.
   *
   * @param {DebuggerPanelBindings} bindings
   */
  constructor({
    roster,
    rosterCount,
    selectionCard,
    pendingHeading,
    pendingCount,
    pendingScope,
    pendingCard,
    acceptedCard,
    acceptedAnnouncement,
    diagnosticsCard,
    actionsPanel = null,
    onCommand,
  }) {
    this.actionsPanel = actionsPanel;
    /** @type {Map<string, Readonly<Record<string, unknown>>>} */
    this.activationByPublicId = new Map();
    this.roster = roster;
    this.rosterCount = rosterCount;
    this.selectionCard = selectionCard;
    this.pendingHeading = pendingHeading;
    this.pendingCount = pendingCount;
    this.pendingScope = pendingScope;
    this.pendingCard = pendingCard;
    this.acceptedCard = acceptedCard;
    this.acceptedAnnouncement = acceptedAnnouncement;
    this.diagnosticsCard = diagnosticsCard;
    this.onCommand = onCommand;
    /** @type {Map<string, AuthorizedRosterRow>} */
    this.rosterRows = new Map();
    /** @type {Map<number, RosterTeamGroup>} */
    this.rosterTeamGroups = new Map();
    /** @type {Map<string, RosterTeamGroup>} */
    this.rosterVisibilityGroups = new Map();
    /** @type {WeakMap<HTMLButtonElement, Readonly<Record<string, unknown>>>} */
    this.rosterActivationByButton = new WeakMap();
    /** @type {string | null} */
    this.lastAnnouncedTransitionKey = null;
    // The Full Record chip sits above the roster groups; it shows only in a
    // Replay Viewer frame in Agent POV.
    this.rosterFullRecord = htmlElement("p", "panel-full-record");
    this.rosterFullRecord.hidden = true;
    this.roster.replaceChildren(this.rosterFullRecord);
    this.ensureRosterTeamGroup(1);
    this.ensureRosterTeamGroup(2);
    this.ensureRosterVisibilityGroup("visible");
    this.ensureRosterVisibilityGroup("not-visible");
  }

  /**
   * Return the cached visible or not-visible roster group, creating it if
   * needed. A new group starts hidden with an empty-state row and is appended
   * to this.roster. visibility must be one of the two declared strings. The
   * returned object and its DOM nodes are shared mutable view state.
   *
   * @param {"visible" | "not-visible"} visibility
   * @returns {RosterTeamGroup}
   */
  ensureRosterVisibilityGroup(visibility) {
    const existing = this.rosterVisibilityGroups.get(visibility);
    if (existing) {
      return existing;
    }
    const visible = visibility === "visible";
    const element = htmlElement("section", "roster-team roster-visibility");
    element.dataset.visibility = visibility;
    element.hidden = true;
    const heading = htmlElement("div", "roster-team__heading");
    const title = htmlElement("h3", null, visible ? "VISIBLE" : "NOT VISIBLE");
    const count = htmlElement("span", "count-badge", "0 agents");
    heading.append(title, count);
    const rows = htmlElement("div", "roster-team__rows");
    const empty = htmlElement(
      "p",
      "empty-copy",
      visible ? "No agents are visible." : "No agents are outside this snapshot.",
    );
    rows.append(empty);
    element.append(heading, rows);
    this.roster.append(element);
    const group = { element, count, rows, empty };
    this.rosterVisibilityGroups.set(visibility, group);
    return group;
  }

  /**
   * Return a cached group for numeric teamId or create and append it.
   * Use the display vocabulary for labels, including unknown IDs; this does not
   * validate team membership. The new group starts with zero authorized agents.
   * Return shared DOM references for later count and row updates.
   *
   * @param {number} teamId
   * @returns {RosterTeamGroup}
   */
  ensureRosterTeamGroup(teamId) {
    const existing = this.rosterTeamGroups.get(teamId);
    if (existing) {
      return existing;
    }
    const team = teamTokenFromId(teamId);
    const element = htmlElement("section", "roster-team");
    element.dataset.teamId = String(teamId);
    element.dataset.team = team.cssKey;
    element.setAttribute("aria-label", team.accessibleName);
    const heading = htmlElement("div", "roster-team__heading");
    const title = htmlElement("h3", null, team.label);
    const count = htmlElement("span", "count-badge", "0 Champions");
    heading.append(title, count);
    const rows = htmlElement("div", "roster-team__rows");
    const empty = htmlElement("p", "empty-copy", "No champions.");
    rows.append(empty);
    element.append(heading, rows);
    this.roster.append(element);
    const group = { element, count, rows, empty };
    this.rosterTeamGroups.set(teamId, group);
    return group;
  }

  /**
   * Create an unattached roster row for one authorized identity. Its button
   * looks up the current activation record at click time and calls onCommand;
   * status/modifier chips are siblings so their clicks do not activate the agent.
   * Return mutable DOM references. The caller fills values, stores activation
   * and inserts the row. This installs a listener but performs no network call.
   *
   * @param {ReturnType<typeof authorizedPresentationIdentityRows>[number]} identity
   * @returns {AuthorizedRosterRow}
   */
  createAuthorizedRosterRow(identity) {
    const element = htmlElement("article", "roster-row roster-row--authorized");
    element.dataset.presentationKey = identity.presentation_key;
    const primaryButton = htmlElement("button", "roster-primary-action");
    primaryButton.type = "button";
    primaryButton.dataset.action = "activate-agent";
    primaryButton.dataset.presentationKey = identity.presentation_key;
    const summary = htmlElement("span", "roster-primary-action__summary");
    const identityContainer = htmlElement("span", "roster-identity");
    const identityId = htmlElement("span", "roster-id");
    const identityClass = htmlElement("span", "roster-class");
    const controller = htmlElement("span", "roster-controller");
    identityContainer.append(identityId, identityClass, controller);
    const health = htmlElement("span", "roster-health");
    summary.append(identityContainer, health);
    primaryButton.append(summary);
    primaryButton.addEventListener("click", (event) => {
      const command = this.rosterActivationByButton.get(primaryButton);
      if (command !== undefined) {
        void this.onCommand(command, {
          pointerOriginated: Number(event.detail) > 0,
        });
      }
    });

    const facts = htmlElement("div", "roster-row__facts");
    const statuses = htmlElement("div", "roster-fact-list roster-statuses");
    statuses.setAttribute("aria-label", "Persistent statuses");
    const modifiers = htmlElement("div", "roster-fact-list roster-modifiers");
    modifiers.setAttribute("aria-label", "Exact effective modifiers");
    facts.append(statuses, modifiers);
    element.append(primaryButton, facts);
    return {
      element,
      primaryButton,
      identityId,
      identityClass,
      controller,
      health,
      statuses,
      modifiers,
    };
  }

  /**
   * Update roster rows, labels, activation state and groups for presentation.
   * disabled blocks all row activation; localInspectedPresentationKey defaults
   * to the scene's inspection choice. Researcher-space Agent POV groups the
   * complete roster by visibility; other views group authorized rows by team.
   * Reuse rows by display key, remove stale ones and register tooltips.
   *
   * Names are token chips (or plain "Agent ID N" when panelDisplayStyle says
   * so). Replay Viewer Agent POV shows the full record under the Full Record
   * chip. BattleClient and DevClient Agent POV (plan P23) list every champion
   * of the match by identity only, read life, Ultimate, statuses and modifiers
   * from the POV arena scene alone, and show "Unseen" with no facts for a
   * champion the team cannot see. A life that drops to 0 this tick is marked
   * to wait for its Fall beat, and a returning champion's facts for its Return
   * beat (data-beat-hold; see the file description).
   *
   * Each row's "Played by <name>" names who played the champion: in live
   * play the slot's controller from the combat configuration, in the Replay
   * Viewer the recorded name from the page's one controller model
   * (recordedController), shown verbatim; no line when neither knows it. The
   * name is a controller-names.js name node (cut with "…" past the fixed name
   * width, its team's controller list on hover while cut). Counts read "5
   * Champions" (Agent POV in the Replay Viewer: "10 Champions · 4 Seen · 6
   * Not Seen").
   *
   * Also refreshes activationByPublicId, which chip clicks use to pick a
   * champion. This changes DOM/cache state only and assumes a branded
   * presentation.
   *
   * @param {Record<string, any>} presentation
   * @param {boolean} disabled
   * @param {string | null | undefined} [localInspectedPresentationKey]
   * @param {Record<string, any> | null} [combatConfiguration]
   * @param {unknown} [visualFilterState] The page's switch state (default:
   *   shipped defaults). Panel Tokens and Ledger Heraldry pick chips or plain
   *   names; Status Icons From Marks picks mark icons or plain status icons.
   */
  renderAuthorizedRoster(
    presentation,
    disabled,
    localInspectedPresentationKey = undefined,
    combatConfiguration = null,
    visualFilterState = undefined,
  ) {
    const audience = authorizedPresentationAudience(presentation);
    const identities = authorizedPresentationIdentityRows(presentation);
    const researcherInspectionState =
      authorizedPresentationResearcherInspectionState(presentation);
    const livePov = isLiveAgentPovPanel(presentation);
    const style = panelDisplayStyle(presentation, visualFilterState);
    const markIcons = statusIconsFromMarks(visualFilterState);
    const globalAgentRoster =
      audience === "agent_pov" &&
      authorizedPresentationHasResearcherSpace(presentation);
    // Both Agent POV rosters list every champion grouped by sight. The Replay
    // Viewer shows the full record; the BattleClient and DevClient read facts
    // from the POV scene only.
    const visibilityGroupedRoster = globalAgentRoster;
    const scene = livePov
      ? authorizedPresentationSceneView(presentation)
      : globalAgentRoster
        ? authorizedPresentationResearcherSceneView(presentation)
        : authorizedPresentationSceneView(presentation, localInspectedPresentationKey);
    const rosterAudience = livePov ? "agent_pov" : (scene?.audience ?? audience);
    const selection = isRecord(scene?.selection) ? scene.selection : {};
    const sceneAgents = asArray(scene?.agents).filter(isRecord);
    const sourceAgents =
      rosterAudience === "researcher" ? identities.map(({ agent }) => agent) : [];
    const activeKeys = new Set(identities.map(({ display_key }) => display_key));
    for (const [key, row] of this.rosterRows) {
      if (typeof key !== "string" || !activeKeys.has(key)) {
        row.element.remove();
        this.rosterRows.delete(key);
      }
    }
    const fullRecord = fullRecordLabel(presentation);
    this.rosterFullRecord.hidden = fullRecord === null;
    this.rosterFullRecord.textContent = fullRecord ?? "";
    const beats = safeBeatView(presentation, visualFilterState);
    const audienceKeyByPublicId = audienceSceneKeys(presentation);
    this.activationByPublicId.clear();

    const visibleCount = identities.filter(
      (identity) => identity.visible_in_snapshot,
    ).length;
    this.rosterCount.textContent = visibilityGroupedRoster
      ? `${championsText(identities.length)} · ${visibleCount} Seen · ${identities.length - visibleCount} Not Seen`
      : championsText(identities.length);
    /** @type {Map<number, HTMLElement[]>} */
    const desiredByTeam = new Map();
    /** @type {Map<string, HTMLElement[]>} */
    const desiredByVisibility = new Map();
    for (const identity of identities) {
      const sceneAgent = livePov
        ? (sceneAgents.find(
            (candidate) => candidate.public_agent_id === identity.public_agent_id,
          ) ?? null)
        : (sceneAgents.find(
            (candidate) => candidate.presentation_key === identity.presentation_key,
          ) ?? null);
      // BattleClient Agent POV: an unseen champion keeps only its identity.
      const unseen = livePov && sceneAgent === null;
      const agent = unseen
        ? identityOnlyAgent(identity.agent)
        : agentForPresentation(sceneAgent ?? identity.agent);
      const candidateRow = this.rosterRows.get(identity.display_key);
      let row =
        candidateRow && "primaryButton" in candidateRow ? candidateRow : undefined;
      if (!row) {
        candidateRow?.element.remove();
        row = this.createAuthorizedRosterRow(identity);
        this.rosterRows.set(identity.display_key, row);
      }
      const publicId = String(agent.public_agent_id);
      const controller =
        authorizedPresentationController(presentation, combatConfiguration, publicId) ??
        recordedController(publicId);
      const displayIdentity = canonicalAgentIdentity(agent).publicIdentity;
      const classToken = classTokenFromId(agent.class_id);
      const teamToken = teamTokenFromId(agent.team_id);
      row.element.dataset.presentationKey = String(agent.presentation_key);
      row.element.dataset.teamId = String(agent.team_id);
      row.element.dataset.classId = String(agent.class_id);
      row.element.dataset.team = teamToken.cssKey;
      if (visibilityGroupedRoster) {
        row.element.dataset.visibleInSnapshot = String(identity.visible_in_snapshot);
      } else {
        delete row.element.dataset.visibleInSnapshot;
      }
      if (unseen) {
        row.element.dataset.seen = "false";
      } else {
        delete row.element.dataset.seen;
      }
      row.element.removeAttribute("data-class");
      row.identityId.dataset.class = classToken.cssKey;
      row.element.setAttribute(
        "aria-label",
        `${displayIdentity}, ${classToken.label}, ${teamToken.label}${unseen ? ", Unseen" : ""}`,
      );
      registerTooltipOwner(
        row.primaryButton,
        unseen
          ? explainUnseenChampion(agent)
          : rosterAudience === "researcher"
            ? explainAgent(agent)
            : explainPovAgent(agent, {
                controlled:
                  agent.presentation_key === presentation.recipient_presentation_key,
                selected: false,
                inspected:
                  selection.inspection_owner_presentation_key ===
                  agent.presentation_key,
              }),
        { inspectable: false },
      );
      row.primaryButton.dataset.presentationKey = String(agent.presentation_key);
      const activation = Number.isInteger(identity.command_global_slot)
        ? identity.activation_kind === "replay_pov_global"
          ? Object.freeze({
              command_type: "activate_replay_pov_agent",
              global_slot: identity.command_global_slot,
              public_agent_id: publicId,
            })
          : identity.activation_kind === "live_pov_global" &&
              ["live_editable", "live_policy"].includes(
                researcherInspectionState.state_kind,
              )
            ? Object.freeze({
                command_type: "activate_live_pov_agent",
                global_slot: identity.command_global_slot,
                public_agent_id: publicId,
              })
            : identity.activation_kind === "scene_agent"
              ? Object.freeze({
                  command_type: "activate_authorized_agent",
                  presentation_key: identity.presentation_key,
                })
              : null
        : identity.activation_kind === "scene_agent"
          ? Object.freeze({
              command_type: "activate_authorized_agent",
              presentation_key: identity.presentation_key,
            })
          : null;
      if (activation === null) {
        this.rosterActivationByButton.delete(row.primaryButton);
      } else {
        this.rosterActivationByButton.set(row.primaryButton, activation);
        if (!disabled) this.activationByPublicId.set(publicId, activation);
      }
      row.primaryButton.disabled = disabled || activation === null;
      const inspected =
        selection.inspection_owner_presentation_key === agent.presentation_key;
      row.primaryButton.setAttribute("aria-pressed", String(inspected));
      row.element.dataset.selected = String(inspected);
      row.primaryButton.setAttribute(
        "aria-label",
        rosterAudience === "researcher" &&
          presentation.viewer_mode === "live" &&
          (controller?.manual ??
            researcherInspectionState.state_kind === "live_editable")
          ? `Control and inspect ${displayIdentity}`
          : `Inspect ${displayIdentity}`,
      );
      if (style.tokens) {
        row.identityId.replaceChildren(
          createTokenChip(document, agent, { tooltip: false }),
        );
      } else {
        row.identityId.replaceChildren();
        row.identityId.textContent = displayIdentity;
      }
      row.identityClass.textContent = `${classToken.label} · ${teamToken.label}`;
      // The name follows the shared controller-name rule: cut with "…" past
      // the fixed name width, the team's list on hover while cut.
      row.controller.replaceChildren(
        ...(controller === null
          ? []
          : [
              "Played by ",
              controllerNameNode(
                document,
                controller.label,
                agent.team_id === 1 || agent.team_id === 2 ? agent.team_id : null,
              ),
            ]),
      );
      row.controller.hidden = controller === null;
      const audienceKey = audienceKeyByPublicId.get(publicId) ?? null;
      const returned =
        !unseen &&
        audienceKey !== null &&
        (beats?.returns ?? []).some(
          (entry) => entry.publicAgentId === publicId && entry.seen === true,
        );
      for (const node of [row.health, row.statuses, row.modifiers]) {
        setBeatHold(node, returned ? "return" : null, audienceKey, "after");
      }
      if (unseen) {
        row.health.replaceChildren();
        row.health.textContent = "Unseen";
        row.statuses.replaceChildren();
        row.modifiers.replaceChildren();
      } else {
        const fall = (beats?.life ?? []).find(
          (entry) => entry.publicAgentId === publicId && entry.before !== null,
        );
        renderLifeText(row.health, agent, fall ?? null, true);
        renderFactTokens(
          row.statuses,
          asArray(agent.statuses),
          "status",
          "No persistent statuses",
          agent,
          sourceAgents,
          rosterAudience ?? "agent_pov",
          markIcons,
        );
        renderFactTokens(
          row.modifiers,
          asArray(agent.modifiers ?? agent.aura_modifiers),
          "modifier",
          "No effective modifiers",
          agent,
          sourceAgents,
          rosterAudience ?? "agent_pov",
        );
      }
      if (visibilityGroupedRoster) {
        const visibility = identity.visible_in_snapshot ? "visible" : "not-visible";
        const desired = desiredByVisibility.get(visibility) ?? [];
        desired.push(row.element);
        desiredByVisibility.set(visibility, desired);
      } else {
        const desired = desiredByTeam.get(Number(agent.team_id)) ?? [];
        desired.push(row.element);
        desiredByTeam.set(Number(agent.team_id), desired);
        this.ensureRosterTeamGroup(Number(agent.team_id));
      }
    }
    for (const [teamId, group] of this.rosterTeamGroups) {
      group.element.hidden = visibilityGroupedRoster;
      const desired = desiredByTeam.get(teamId) ?? [];
      group.count.textContent = championsText(desired.length);
      this.reconcileChildren(group.rows, desired.length > 0 ? desired : [group.empty]);
    }
    for (const [visibility, group] of this.rosterVisibilityGroups) {
      group.element.hidden = !visibilityGroupedRoster;
      const desired = desiredByVisibility.get(visibility) ?? [];
      group.count.textContent = `${desired.length} agents`;
      this.reconcileChildren(group.rows, desired.length > 0 ? desired : [group.empty]);
    }
  }

  /**
   * Pick a champion from a token chip: send the same activation command its
   * roster row would send, through onCommand, marked as pointer-originated.
   * Does nothing when the champion has no enabled roster activation (for
   * example while the page is busy). Replay Viewer chips only.
   *
   * @param {string} publicAgentId The champion's public ID.
   * @returns {void}
   */
  pickAgent(publicAgentId) {
    const activation = this.activationByPublicId.get(publicAgentId);
    if (activation !== undefined) {
      void this.onCommand(activation, { pointerOriginated: true });
    }
  }

  /**
   * Refresh agent details, the Actions view (This Tick and Next Tick) and the
   * technical facts from presentation.
   *
   * Agent details: activatedAgentPublicId=null shows the activation prompt,
   * while an omitted value allows scene-owner selection (see
   * authorizedInspectorView). See renderAgentDetails for the sections.
   *
   * Actions: in the Replay Viewer, acceptedCard holds This Tick (What
   * Happened) and pendingCard holds Next Tick (What Happens Next); the
   * optional actionsPanel holds both together. On the final frame Next Tick
   * reads "No Next Tick (Final Frame)". Fallen champions that take no action
   * are listed on one "Fallen · No Action" line. In Agent POV the docked
   * Replay Viewer views are the full record, under the Full Record chip. In
   * the BattleClient and DevClient, pendingCard keeps the player's staged
   * Manual orders and acceptedCard the last tick; in Agent POV the pending
   * card shows the POV agent's own staged order from its POV draft and reads
   * "Not Observed" for every other row (livePovPendingRows), and the last
   * tick shows only the POV agent's own
   * action and "Not Observed" for each teammate the POV scene shows (enemies
   * are not listed).
   *
   * Suppress repeated announcements while the key (session, transition,
   * audience and Agent POV recipient) stays unchanged; a new tick, a view
   * switch or a new POV agent announces the current row count, so the hidden
   * live text always matches the rows shown; returning to an earlier key can
   * announce it again. Replace
   * panel contents without changing recorded facts or submitting actions;
   * input must already be branded.
   *
   * @param {Record<string, any>} presentation
   * @param {string | null | undefined} [localInspectedPresentationKey]
   * @param {string | null | undefined} [activatedAgentPublicId]
   * @param {unknown} [visualFilterState] The page's switch state (default:
   *   shipped defaults).
   */
  renderAuthorizedInspector(
    presentation,
    localInspectedPresentationKey = undefined,
    activatedAgentPublicId = undefined,
    visualFilterState = undefined,
  ) {
    const inspector = authorizedInspectorView(
      presentation,
      localInspectedPresentationKey,
      activatedAgentPublicId,
    );
    const inspectionState =
      authorizedPresentationResearcherInspectionState(presentation);
    const replay = presentation.viewer_mode === "replay";
    const livePov = isLiveAgentPovPanel(presentation);
    const style = panelDisplayStyle(presentation, visualFilterState);
    const beats = safeBeatView(presentation, visualFilterState);
    const identities = authorizedPresentationIdentityRows(presentation);
    /** @type {Map<string, Readonly<Record<string, any>>>} */
    const identityByPublicId = new Map(
      identities.map((identity) => [
        identity.public_agent_id,
        livePov ? identityOnlyAgent(identity.agent) : identity.agent,
      ]),
    );
    const onPick = replay ? (/** @type {string} */ id) => this.pickAgent(id) : null;
    /** @type {ActionListOptions} */
    const listOptions = {
      tokens: style.tokens,
      identityByPublicId,
      movementNames: frameMovementNames(presentation),
      onPick,
    };

    this.renderAgentDetails(
      presentation,
      inspector,
      activatedAgentPublicId,
      style,
      beats,
      identityByPublicId,
    );

    const upcomingRows = replay
      ? joinActionRows(
          presentation,
          authorizedPresentationUpcomingTransitionRows(presentation),
          "upcoming_transition",
          identityByPublicId,
        )
      : [];
    const livePending = replay
      ? []
      : joinActionRows(
          presentation,
          authorizedPresentationPendingJointActionRows(presentation),
          "pending_joint_action",
          identityByPublicId,
        );
    const pendingRows = livePov
      ? livePovPendingRows(presentation, livePending)
      : livePending;
    if (inspectionState.submission_scope === null) {
      this.pendingCard.removeAttribute("data-submission-scope");
    } else {
      this.pendingCard.dataset.submissionScope = inspectionState.submission_scope;
    }
    this.pendingCard.dataset.inspectionState = inspectionState.state_kind;
    this.pendingHeading.textContent = {
      live_editable: "Pending Manual Actions",
      live_policy: "Pending Manual Actions",
      live_scripted: "Scripted playback inspection",
      replay_outgoing: "Next Tick (What Happens Next)",
      replay_none: "Next Tick (What Happens Next)",
      unavailable: "Inspection unavailable",
    }[inspectionState.state_kind];
    this.pendingCount.hidden = replay;
    this.pendingScope.hidden = replay;
    this.pendingCount.textContent = `${pendingRows.length} ${pendingRows.length === 1 ? "actor" : "actors"}`;
    this.pendingScope.textContent =
      inspectionState.state_kind === "live_scripted"
        ? "This live frame advances registered scripted actions and has no editable draft."
        : ["live_editable", "live_policy"].includes(inspectionState.state_kind)
          ? "These Manual drafts join the policies' actions on Submit. Policies choose their own actions when the turn starts."
          : inspectionState.state_kind === "replay_outgoing"
            ? "This panel shows the authorized recorded actions out of the current frame."
            : inspectionState.state_kind === "replay_none"
              ? "No upcoming transition is available at this replay frame."
              : "Inspection is unavailable for this frame.";

    /** @type {() => HTMLElement[]} */
    let buildNextTick = () => [];
    if (replay) {
      const sceneAgents = researcherSceneAgentsByPublicId(presentation);
      const fallen = upcomingRows.filter((view) =>
        isFallenNoAction(view, sceneAgents, null, "next"),
      );
      const acting = upcomingRows.filter((view) => !fallen.includes(view));
      buildNextTick = () => {
        const fallenLine = fallenNoActionLine(
          fallen.flatMap((view) => (view.actor === null ? [] : [view.actor])),
          style.tokens,
          onPick,
        );
        return upcomingRows.length > 0
          ? [
              authorizedTransitionList(acting, listOptions),
              ...(fallenLine === null ? [] : [fallenLine]),
            ]
          : [htmlElement("p", "empty-copy", "No Next Tick (Final Frame)")];
      };
      this.pendingCard.replaceChildren(...buildNextTick());
    } else if (["live_editable", "live_policy"].includes(inspectionState.state_kind)) {
      this.pendingCard.replaceChildren(
        pendingRows.length > 0
          ? authorizedTransitionList(pendingRows, { ...listOptions, pending: true })
          : htmlElement(
              "p",
              "empty-copy",
              "All agents are policy controlled. Their next actions have not been chosen yet.",
            ),
      );
    } else {
      this.pendingCard.replaceChildren(
        htmlElement(
          "p",
          "empty-copy",
          inspectionState.state_kind === "live_scripted"
            ? "No editable joint action is available during scripted playback."
            : "No pending joint action is available.",
        ),
      );
    }

    const transitionRows = livePov
      ? livePovTransitionRows(presentation)
      : joinActionRows(
          presentation,
          authorizedPresentationTransitionRows(presentation),
          "latest_transition",
          identityByPublicId,
        );
    this.acceptedCard.replaceChildren();
    /** @type {() => HTMLElement[]} */
    let buildThisTick = () => [htmlElement("p", "empty-copy", "No earlier tick yet.")];
    if (transitionRows.length === 0) {
      this.acceptedAnnouncement.textContent = "";
      this.lastAnnouncedTransitionKey = null;
    } else {
      const sceneAgents = replay
        ? researcherSceneAgentsByPublicId(presentation)
        : new Map();
      const fallen = replay
        ? transitionRows.filter((view) =>
            isFallenNoAction(view, sceneAgents, beats, "this"),
          )
        : [];
      const acting = transitionRows.filter((view) => !fallen.includes(view));
      buildThisTick = () => {
        const fallenLine = fallenNoActionLine(
          fallen.flatMap((view) => (view.actor === null ? [] : [view.actor])),
          style.tokens,
          onPick,
        );
        return [
          authorizedTransitionList(acting, listOptions),
          ...(fallenLine === null ? [] : [fallenLine]),
        ];
      };
      this.acceptedCard.append(...buildThisTick());
      const transitionId = authorizedPresentationLatestTransitionId(presentation);
      const sessionId = presentation.source?.source_session_id;
      // The rows depend on the view and, in Agent POV, on the POV agent (its
      // teammates), so a view or focus switch announces the new count, as a
      // freshly loaded page does.
      const transitionKey =
        typeof sessionId === "string" && typeof transitionId === "string"
          ? `${sessionId}:${transitionId}:${authorizedPresentationAudience(presentation)}:${String(presentation.authority?.recipient_public_agent_id ?? "")}`
          : null;
      if (transitionKey !== this.lastAnnouncedTransitionKey) {
        this.acceptedAnnouncement.textContent = `${transitionRows.length} Submitted / Accepted action ${transitionRows.length === 1 ? "row" : "rows"}.`;
        this.lastAnnouncedTransitionKey = transitionKey;
      }
    }

    if (this.actionsPanel !== null) {
      this.renderActionsPanel(
        replay,
        fullRecordLabel(presentation),
        buildThisTick(),
        buildNextTick(),
      );
    }

    // The Technical Frame in its three groups with Copy (ledger.js owns the
    // drawing; each row keeps data-technical-fact, focus and its help tip).
    renderTechnicalFrame(
      this.diagnosticsCard,
      authorizedPresentationTechnicalFacts(presentation),
    );
  }

  /**
   * Fill the optional Actions panel (the Replay Viewer's Actions tab) with
   * This Tick and Next Tick together, each under its heading with the older
   * name as a subtitle, after the Full Record chip in Agent POV. The nodes
   * passed in must be fresh (not the ones already placed in the older cards).
   * Outside the Replay Viewer the panel is emptied.
   *
   * @param {boolean} replay Whether the frame is a Replay Viewer frame.
   * @param {string | null} fullRecord Full Record chip words, or null.
   * @param {ReadonlyArray<HTMLElement>} thisTickNodes This Tick content.
   * @param {ReadonlyArray<HTMLElement>} nextTickNodes Next Tick content.
   * @returns {void}
   */
  renderActionsPanel(replay, fullRecord, thisTickNodes, nextTickNodes) {
    const panel = /** @type {HTMLElement} */ (this.actionsPanel);
    if (!replay) {
      panel.replaceChildren();
      return;
    }
    /**
     * @param {string} title
     * @param {string} subtitle
     * @param {ReadonlyArray<HTMLElement>} nodes
     */
    const section = (title, subtitle, nodes) => {
      const part = htmlElement("section", "panel-actions__part");
      part.append(
        htmlElement("h3", "panel-actions__title", title),
        htmlElement("p", "panel-actions__subtitle", subtitle),
        ...nodes,
      );
      return part;
    };
    panel.replaceChildren(
      ...(fullRecord === null
        ? []
        : [htmlElement("p", "panel-full-record", fullRecord)]),
      section("This Tick (What Happened)", "Latest Transition", thisTickNodes),
      section("Next Tick (What Happens Next)", "Upcoming Transition", nextTickNodes),
    );
  }

  /**
   * Fill the agent details card.
   *
   * Sections, in order: the Full Record chip (Replay Viewer Agent POV), Who
   * (the champion's chip, class and calling, team, order and home), Right Now
   * (life, Ultimate, combat, effective speed; life waits for the Fall beat),
   * Statuses (one line each by the owner's status rule: "Serrated Arrows
   * (Slow) · x0.85 speed · 1 tick left · from Hunter ID 7's Serrated
   * Arrows"), then the
   * class card from explanations.js (overview; role, strength and weakness;
   * Basic, Ultimate and passive with the host's numbers), then Lore (the
   * fantasy line). BattleClient and DevClient Agent POV: a selected champion
   * the team cannot see reads "Unseen" with its Out of Sight chip and no facts;
   * status sources follow the Agent POV rule ("from a hidden source" unless the
   * team saw the status applied this tick).
   *
   * @param {Record<string, any>} presentation
   * @param {Readonly<Record<string, any>> | null} inspector authorizedInspectorView result.
   * @param {string | null | undefined} activatedAgentPublicId
   * @param {Readonly<{tokens: boolean}>} style panelDisplayStyle result.
   * @param {ReturnType<typeof beatView>} beats
   * @param {ReadonlyMap<string, Readonly<Record<string, any>>>} identityByPublicId
   * @returns {void}
   */
  renderAgentDetails(
    presentation,
    inspector,
    activatedAgentPublicId,
    style,
    beats,
    identityByPublicId,
  ) {
    const card = this.selectionCard;
    const fullRecord = fullRecordLabel(presentation);
    const head =
      fullRecord === null ? [] : [htmlElement("p", "panel-full-record", fullRecord)];
    if (activatedAgentPublicId === null) {
      card.replaceChildren(
        ...head,
        htmlElement(
          "p",
          "empty-copy",
          "Activate an agent to inspect its comprehensive class details.",
        ),
      );
      return;
    }
    const livePov = isLiveAgentPovPanel(presentation);
    if (
      livePov &&
      typeof activatedAgentPublicId === "string" &&
      inspector?.owner === null &&
      identityByPublicId.has(activatedAgentPublicId)
    ) {
      const agent = /** @type {Readonly<Record<string, any>>} */ (
        identityByPublicId.get(activatedAgentPublicId)
      );
      const unseen = htmlElement("section", "agent-details__unseen");
      unseen.append(
        createTokenChip(document, agent, { outOfSight: true }),
        htmlElement(
          "p",
          "empty-copy",
          "Unseen. Your team cannot see this champion now. Its details come back when your team sees it again.",
        ),
      );
      card.replaceChildren(unseen);
      return;
    }
    if (inspector === null || inspector.owner_descriptor === null) {
      card.replaceChildren(
        ...head,
        htmlElement("p", "empty-copy", "No authorized agent details are available."),
      );
      return;
    }
    const owner = inspector.owner;
    const identity = canonicalAgentIdentity(owner);
    const names = classNamesV1(owner.class_id);
    const lore = classLoreV1(owner.class_id);

    const who = detailSection("Who");
    const whoFacts = htmlElement("div", "agent-details__facts");
    addNodeFact(
      whoFacts,
      "Agent",
      createTokenChip(document, owner, { plain: !style.tokens }),
    );
    addFact(
      whoFacts,
      "Class",
      names === null
        ? identity.classLabel
        : `${identity.classLabel} (${names.calling})`,
    );
    addFact(whoFacts, "Team", identity.teamLabel);
    if (lore !== null) addFact(whoFacts, "Order", lore.order);
    who.append(whoFacts);

    const now = detailSection("Right Now");
    const nowFacts = htmlElement("div", "agent-details__facts");
    const lifeValue = htmlElement("div", "agent-details__life");
    const fall = (beats?.life ?? []).find(
      (entry) => entry.publicAgentId === owner.public_agent_id && entry.before !== null,
    );
    renderLifeText(lifeValue, owner, fall ?? null, false);
    addNodeFact(nowFacts, "Life", lifeValue);
    const cooldown = owner.ultimate_cooldown_remaining;
    addFact(
      nowFacts,
      "Ultimate",
      Number.isInteger(cooldown)
        ? cooldown === 0
          ? "Ready"
          : `Ready in ${cooldown} ${cooldown === 1 ? "Tick" : "Ticks"}`
        : "—",
    );
    const combatTicks = owner.steps_until_out_of_combat;
    addFact(
      nowFacts,
      "Combat",
      Number.isInteger(combatTicks) && combatTicks > 0
        ? `In Combat · ${combatTicks} ${combatTicks === 1 ? "Tick" : "Ticks"} Left`
        : "Out of Combat",
    );
    if (Number.isFinite(owner.effective_movement_speed)) {
      addFact(
        nowFacts,
        "Effective Speed",
        formatDisplayNumber(owner.effective_movement_speed),
      );
    }
    now.append(nowFacts);

    const statuses = detailSection("Statuses");
    const statusList = htmlElement("ul", "agent-details__statuses");
    // Champion Card order: stuns, then slows, then the rest.
    const ordered = asArray(owner.statuses)
      .filter(isRecord)
      .sort(
        (a, b) =>
          statusOrderIndex(statusTokenIdFromCatalogId(a.status_id ?? a.token_id)) -
          statusOrderIndex(statusTokenIdFromCatalogId(b.status_id ?? b.token_id)),
      );
    for (const status of ordered) {
      statusList.append(htmlElement("li", null, statusLine(owner, status)));
    }
    if (statusList.children.length === 0) {
      statusList.append(htmlElement("li", "empty-copy", "No statuses."));
    }
    statuses.append(statusList);

    const classCard = htmlElement("div", "agent-details__class");
    renderSemanticInspector(classCard, inspector.owner_descriptor);

    const nodes = [...head, who, now, statuses, classCard];
    if (lore !== null) {
      const loreSection = detailSection("Lore");
      loreSection.append(htmlElement("p", "agent-details__lore", lore.fantasyLine));
      nodes.push(loreSection);
    }
    card.replaceChildren(...nodes);
  }

  /**
   * Make container's element children match desired by identity and order.
   * Remove unlisted elements and move/insert existing desired nodes in place.
   * The caller supplies a duplicate-free array of elements. Text nodes are not
   * part of this reconciliation. Return nothing; existing element state survives.
   *
   * @param {HTMLElement} container
   * @param {HTMLElement[]} desired
   */
  reconcileChildren(container, desired) {
    const desiredSet = new Set(desired);
    for (const child of [...container.children]) {
      if (!desiredSet.has(/** @type {HTMLElement} */ (child))) {
        child.remove();
      }
    }

    for (let index = 0; index < desired.length; index += 1) {
      const child = container.children.item(index);
      const desiredChild = desired[index];
      if (child !== desiredChild) {
        container.insertBefore(desiredChild, child);
      }
    }
  }

  /**
   * Clear roster caches, chip pick routes, the Full Record chip, the optional
   * Actions panel and scientific panel content and show unavailable
   * messages. Reset the last announced transition and action-scope attributes.
   * Keep reusable group DOM and bindings. Return nothing; raw transport data
   * is never used as a fallback when an installed presentation is absent.
   */
  renderUnavailable() {
    for (const row of this.rosterRows.values()) {
      row.element.remove();
    }
    this.rosterRows.clear();
    for (const group of this.rosterTeamGroups.values()) {
      group.element.hidden = false;
      group.count.textContent = championsText(0);
      this.reconcileChildren(group.rows, [group.empty]);
    }
    for (const group of this.rosterVisibilityGroups.values()) {
      group.element.hidden = true;
      group.count.textContent = "0 agents";
      this.reconcileChildren(group.rows, [group.empty]);
    }
    this.rosterCount.textContent = "0 actors";
    this.rosterFullRecord.hidden = true;
    this.rosterFullRecord.textContent = "";
    this.activationByPublicId.clear();

    this.selectionCard.replaceChildren(
      htmlElement("p", "empty-copy", "No authorized agent details are available."),
    );
    this.pendingHeading.textContent = "Inspection unavailable";
    this.pendingCount.textContent = "0 actors";
    this.pendingScope.textContent = "Waiting for authorized action details.";
    this.pendingCard.removeAttribute("data-submission-scope");
    this.pendingCard.removeAttribute("data-inspection-state");
    this.pendingCard.removeAttribute("data-pending-count");
    this.pendingCard.replaceChildren(
      htmlElement("p", "empty-copy", "No authorized action details."),
    );

    this.acceptedCard.replaceChildren();
    this.acceptedAnnouncement.textContent = "";
    this.lastAnnouncedTransitionKey = null;
    this.actionsPanel?.replaceChildren();

    this.diagnosticsCard.replaceChildren(
      htmlElement("p", "empty-copy", "Technical Frame unavailable."),
    );
  }

  /**
   * Render a branded frame, or clear scientific panels for any other value.
   * interactionState defaults to {}; busy, shutdown, resync, offline or explicit
   * activationDisabled flags disable roster actions and chip picks. Pass
   * optional inspection and activated public identity to their owned views;
   * visualFilterState (default: shipped switches) picks chips or plain names,
   * words or codes, and the beat holds. This mutates panel DOM
   * and caches, returns nothing, and neither advances simulation nor sends a
   * command until a user later activates a row.
   *
   * @param {Record<string, any> | null} frame
   * @param {PanelInteractionState} interactionState
   */
  render(frame, interactionState = {}) {
    const disabled = Boolean(
      interactionState.busy ||
        interactionState.shuttingDown ||
        interactionState.resyncRequired ||
        interactionState.offline ||
        interactionState.activationDisabled,
    );
    if (isAuthorizedPresentationFrame(frame)) {
      this.renderAuthorizedRoster(
        frame,
        disabled,
        interactionState.localInspectedPresentationKey,
        interactionState.combatConfiguration ?? null,
        interactionState.visualFilterState,
      );
      this.renderAuthorizedInspector(
        frame,
        interactionState.localInspectedPresentationKey,
        interactionState.activatedAgentPublicId ?? null,
        interactionState.visualFilterState,
      );
      return;
    }
    this.renderUnavailable();
  }
}
