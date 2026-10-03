/**
 * @file Token chips: the one way the docked panels name a champion.
 *
 * What it owns: the chip that stands for a champion in the roster, the agent
 * details, the Actions view and every target, source and "Agent ID N" the
 * panels print. A chip is the champion's class emblem inside its team ring
 * (Team A one ring, Team B a double ring) followed by "Agent N", the same token
 * the champion wears on the arena. Two other forms exist:
 *
 * - Out Of Sight: the class emblem with no team ring and the words
 *   "Warrior · ID 6 · Out Of Sight". It never shows life, position or a ring.
 * - Plain: today's text, "Agent ID 6", used when the Panel Tokens switch is off
 *   or, in the Replay Viewer, when Ledger Heraldry is off.
 *
 * Who calls it: panels.js (roster rows, agent details, Actions rows) and any
 * other docked panel that names a champion. tokenChipModel is pure data for
 * tests and for callers that draw their own markup; createTokenChip builds the
 * DOM.
 *
 * Boundary: a chip carries identity only (public ID, class, team), which every
 * view may know from the match setup. It never reads life, statuses, position
 * or any live fact. Agent POV: the caller decides whether the champion is seen;
 * for an unseen champion over the arena or in a BattleClient panel it passes
 * outOfSight: true, which drops the team ring. Labels come from
 * agent-identity.js, the one label owner (plan P7).
 */
import { canonicalAgentIdentity } from "./agent-identity.js";
import { classNamesV1 } from "./class-documentation.js";
import { createSvgIcon } from "./icons.js";
import { createSemanticDescriptor, registerTooltipOwner } from "./tooltip.js";
import { classTokenFromId, teamTokenFromId } from "./vocabulary.js";

/**
 * @typedef {"token" | "out_of_sight" | "plain"} TokenChipMode
 */

/**
 * @typedef {Readonly<{
 *   mode: TokenChipMode,
 *   text: string,
 *   ariaLabel: string,
 *   publicAgentId: string | null,
 *   classKey: string,
 *   glyphKey: string,
 *   teamKey: string,
 *   ring: "single" | "double" | "none",
 *   tipTitle: string,
 *   tipSummary: string,
 * }>} TokenChipModel
 */

/**
 * Options for tokenChipModel and createTokenChip.
 *
 * @typedef {{
 *   outOfSight?: boolean,
 *   plain?: boolean,
 *   label?: string | null,
 *   pickable?: boolean,
 * }} TokenChipOptions
 */

/**
 * Return whether value is a non-null, non-array object. Pure.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Describe one chip as plain data, with no DOM.
 *
 * The words follow agent-identity.js: "Agent 5" for a token chip, "Agent ID 5"
 * for a plain chip and "Mage · ID 5 · Out Of Sight" for an Out Of Sight chip.
 * Out Of Sight wins over plain, because the words "Out Of Sight" are a fact the
 * reader needs in either style.
 *
 * @param {unknown} agent A record with the champion's identity fields:
 *   public_agent_id, class_id (1 Mage to 5 Priest), team_id (1 Team A, 2 Team
 *   B) and optional display_agent_id (a one-digit global label that wins over
 *   the public ID). Only these fields are read. Anything else gives a chip with
 *   "Unknown" class words, as agent-identity.js does.
 * @param {TokenChipOptions} [options] outOfSight (default false): the
 *   champion is out of its audience's sight, so show the Out Of Sight form.
 *   plain (default false): show today's plain text instead of the token. label
 *   (default null): words that replace "Agent N" in a token chip, for example
 *   "Itself"; ignored for the other forms. pickable (default false): the chip
 *   picks its champion on a click, so the tip says so.
 * @returns {TokenChipModel} A frozen record. mode is the form; text the visible
 *   words; ariaLabel the full spoken name ("Agent ID 5, Mage, Team A"; Out Of
 *   Sight leaves the team out because the ring is not drawn); ring "single"
 *   (Team A), "double" (Team B) or "none" (Out Of Sight, plain, or an unknown
 *   team); classKey, glyphKey and teamKey are the vocabulary CSS and icon keys;
 *   tipTitle and tipSummary are the hover words.
 */
export function tokenChipModel(agent, options = {}) {
  const identity = canonicalAgentIdentity(agent);
  const source = isRecord(agent) ? agent : {};
  const classToken = classTokenFromId(source.class_id);
  const teamToken = teamTokenFromId(source.team_id);
  const names = classNamesV1(source.class_id);
  const publicAgentId =
    typeof source.public_agent_id === "string" ? source.public_agent_id : null;
  const outOfSight = options.outOfSight === true;
  const plain = !outOfSight && options.plain === true;
  /** @type {TokenChipMode} */
  const mode = outOfSight ? "out_of_sight" : plain ? "plain" : "token";
  const label =
    typeof options.label === "string" && options.label.trim().length > 0
      ? options.label.trim()
      : null;
  const text =
    mode === "out_of_sight"
      ? identity.outOfSightLabel
      : mode === "plain"
        ? identity.publicIdentity
        : (label ?? identity.chipText);
  const ring =
    mode !== "token"
      ? "none"
      : teamToken.cssKey === "team-a"
        ? "single"
        : teamToken.cssKey === "team-b"
          ? "double"
          : "none";
  const pickNote = options.pickable === true ? " Click to pick this champion." : "";
  const tipSummary = outOfSight
    ? "Your team cannot see this champion now. Its life and place stay hidden until your team sees it again."
    : `${names === null ? identity.classLabel : `${identity.classLabel} (${names.calling})`}.${pickNote}`;
  return Object.freeze({
    mode,
    text,
    ariaLabel: outOfSight
      ? identity.outOfSightLabel
      : `${identity.publicIdentity}, ${identity.classLabel}, ${identity.teamLabel}`,
    publicAgentId,
    classKey: classToken.cssKey,
    glyphKey: classToken.glyphKey,
    teamKey: teamToken.cssKey,
    ring,
    tipTitle: outOfSight
      ? identity.outOfSightLabel
      : `${identity.publicIdentity} · ${identity.classLabel} · ${identity.teamLabel}`,
    tipSummary,
  });
}

/**
 * Build one chip element.
 *
 * The element is a span (a button when onPick is given) with class
 * "token-chip", data-chip-mode ("token", "out_of_sight" or "plain"),
 * data-token-class (class CSS key), data-token-team (team CSS key; absent for
 * Out Of Sight) and data-token-ring ("single", "double" or "none"). A token or
 * Out Of Sight chip holds a span.token-chip__token with the class emblem SVG
 * (decorative); the ring is drawn by panels.css from data-token-ring. The
 * visible words sit in span.token-chip__text. The chip registers a hover and
 * focus tip through tooltip.js (identity only) unless tooltip is false. No
 * data attribute is named "class", so row owners that mark their own class
 * colour keep that attribute to themselves.
 *
 * Side effects: creates DOM nodes in doc; registers one tooltip owner; with
 * onPick, adds one click listener that calls onPick(publicAgentId, event) and
 * stops the click from reaching parent buttons. No network, storage or timers.
 *
 * @param {Document} doc The document that will own the chip.
 * @param {unknown} agent Identity record; see tokenChipModel.
 * @param {TokenChipOptions & {
 *   onPick?: ((publicAgentId: string, event: Event) => void) | null,
 *   tooltip?: boolean,
 * }} [options] The tokenChipModel options plus onPick (default null: not
 *   clickable; a function makes the chip a button that picks its champion) and
 *   tooltip (default true; false skips the tip, for chips inside an element
 *   that already explains the champion).
 * @returns {HTMLElement} The unattached chip.
 */
export function createTokenChip(doc, agent, options = {}) {
  const onPick = typeof options.onPick === "function" ? options.onPick : null;
  const model = tokenChipModel(agent, { ...options, pickable: onPick !== null });
  const chip = doc.createElement(onPick === null ? "span" : "button");
  chip.className = "token-chip";
  chip.dataset.chipMode = model.mode;
  chip.dataset.tokenClass = model.classKey;
  chip.dataset.tokenRing = model.ring;
  if (model.mode !== "out_of_sight") {
    chip.dataset.tokenTeam = model.teamKey;
  }
  if (model.mode !== "plain") {
    const token = doc.createElement("span");
    token.className = "token-chip__token";
    token.append(
      createSvgIcon(doc, model.glyphKey, { className: "token-chip__emblem" }),
    );
    chip.append(token);
  }
  const words = doc.createElement("span");
  words.className = "token-chip__text";
  words.textContent = model.text;
  chip.append(words);
  chip.setAttribute("aria-label", model.ariaLabel);
  if (onPick !== null && model.publicAgentId !== null) {
    const button = /** @type {HTMLButtonElement} */ (chip);
    button.type = "button";
    const publicAgentId = model.publicAgentId;
    chip.addEventListener("click", (event) => {
      event.stopPropagation?.();
      onPick(publicAgentId, event);
    });
  }
  if (options.tooltip !== false) {
    registerTooltipOwner(
      chip,
      createSemanticDescriptor({
        kind: "agent",
        id: `token-chip:${model.publicAgentId ?? "unknown"}:${model.mode}`,
        title: model.tipTitle,
        tone: "information",
        accent: model.classKey,
        summary: model.tipSummary,
        rows: [],
        sections: [],
        metadata: { compact: true, full: true },
        anchor: "element",
      }),
      { inspectable: false },
    );
  }
  return chip;
}
