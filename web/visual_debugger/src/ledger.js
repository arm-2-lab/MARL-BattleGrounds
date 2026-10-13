/**
 * @file The Replay Viewer's right-hand rail (the Arena Stage panels, owner,
 * 3 Oct 2026): its six tabs and the parts of their content this module draws.
 * The content follows Scribe's Ledger's (full Metrics, Actions as This Tick
 * and Next Tick, the Visuals organization); the look is the BattleClient
 * rail's (src/styles/battleclient/rail.css and src/styles/ledger.css).
 *
 * What it owns:
 * - The rail's six tabs (Roster, Agent, Metrics, Actions, Visuals, Frame) in
 *   the empty #replay-rail slot of index.html. Each tab panel scrolls on its
 *   own. mountLedgerRail moves today's cards into the tab panels by element
 *   ID (LEDGER_TAB_CONTENT: the roster into Roster, the Champion Frame into
 *   Agent, where main.js adds the Class Guide under it, the Metrics card,
 *   the two action cards, the Visual Filters card, and the Technical Frame
 *   with the Replay File facts), so every existing ID, listener and renderer
 *   keeps working (panels.js and filter-controls.js still draw the Roster,
 *   Actions and Visuals bodies).
 * - The Ledger Header of the Metrics tab: a sticky header with the Scope
 *   switch ("Up to Tick N | Whole Game", with "Shows the ending too."), the
 *   Topic select with its section eyebrow, the View switch ("Totals | By
 *   Recipient"), and Find a Measurement. The switches mirror today's native
 *   selects, so main.js keeps owning what a choice does.
 * - The Match Summary at the head of the Metrics tab (ledgerMatchSummary).
 * - The Technical Frame in three groups (This Frame, This Game, This File)
 *   with Copy buttons (renderTechnicalFrame).
 * - The "Full Record" chip that the Roster, Agent, Metrics, Actions and
 *   Frame tabs show in Agent POV.
 *
 * Who calls it: main.js, only in the Replay Viewer. It calls mountLedgerRail
 * once when the replay page starts, then rail.update for every installed
 * frame and every new metric summary. metrics-panel.js draws the rows.
 *
 * Agent POV boundary: the rail is a docked researcher panel. In Agent POV it
 * shows the full recorded game for every agent, under the Full Record chip.
 * Nothing in this module draws on or over the arena, and nothing here feeds
 * the arena, its hover cards or the HUD.
 */

import {
  canonicalAgentIdentity,
  fullRecordChipWords,
  teamLabelFromId,
} from "./agent-identity.js";
import {
  authorizedPresentationAudience,
  authorizedPresentationIdentityRows,
  authorizedPresentationTechnicalFacts,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import { explainTechnicalFact } from "./explanations.js";
import {
  createMetricSubjectChips,
  metricValueText,
  renderChoiceSwitch,
} from "./metrics-panel.js";
import { registerTooltipOwner } from "./tooltip.js";

/**
 * The six rail tabs in order. id names the tab; label is its button text and
 * its hover help's heading; help is the one plain sentence the tab button's
 * hover and focus help reads; panelId is the index.html slot it fills;
 * fullRecord says whether the tab shows the Full Record chip in Agent POV
 * (every researcher panel does; the Visuals tab only holds view switches).
 */
export const LEDGER_TABS = Object.freeze([
  Object.freeze({
    id: "roster",
    label: "Roster",
    help: "Every champion in this game, team by team. Pick one to look at it.",
    panelId: "replay-rail-roster",
    fullRecord: true,
  }),
  Object.freeze({
    id: "agent",
    label: "Agent",
    help: "The champion you picked: its life, statuses, Ultimate and class guide.",
    panelId: "replay-rail-agent",
    fullRecord: true,
  }),
  Object.freeze({
    id: "metrics",
    label: "Metrics",
    help: "Numbers about this game for researchers. The agents never see them.",
    panelId: "replay-rail-metrics",
    fullRecord: true,
  }),
  Object.freeze({
    id: "actions",
    label: "Actions",
    help: "What every champion did on this tick, and what each one does next.",
    panelId: "replay-rail-actions",
    fullRecord: true,
  }),
  Object.freeze({
    id: "visuals",
    label: "Visuals",
    help: "Choose which marks and effects the battlefield shows. It changes only your screen, never the game.",
    panelId: "replay-rail-visuals",
    fullRecord: false,
  }),
  Object.freeze({
    id: "frame",
    label: "Frame",
    help: "The raw facts of this tick, this game and this file, for developers.",
    panelId: "replay-rail-frame",
    fullRecord: true,
  }),
]);

/**
 * The cards each tab holds, by element ID, in order. mountLedgerRail moves
 * these existing index.html cards into the tab panels. A card that is missing
 * from the page is skipped.
 */
export const LEDGER_TAB_CONTENT = Object.freeze({
  roster: Object.freeze(["roster-details"]),
  agent: Object.freeze(["champion-frame"]),
  metrics: Object.freeze(["evaluation-metrics"]),
  actions: Object.freeze(["pending-turn-details", "latest-transition-details"]),
  visuals: Object.freeze(["visual-filters"]),
  frame: Object.freeze(["technical-frame-details", "replay-details"]),
});

/** The tab the rail opens on. Agent keeps metric work lazy until asked. */
const DEFAULT_TAB = "agent";

/** How long a Copy button reads "Copied", in milliseconds. */
const COPIED_MS = 1500;

/** The longest id shown in full; longer ids are cut in the middle. */
const ID_SHOWN_CHARACTERS = 32;

/**
 * The three Technical Frame groups, in order, with the fact IDs each holds.
 * Fact IDs come from authorizedPresentationTechnicalFacts. A fact ID this
 * table does not know goes to This Frame, so no fact is ever dropped.
 */
export const TECHNICAL_FRAME_GROUPS = Object.freeze([
  Object.freeze({
    id: "this-frame",
    title: "This Frame",
    factIds: Object.freeze(["frame", "simulator_step", "incoming_transition"]),
  }),
  Object.freeze({
    id: "this-game",
    title: "This Game",
    factIds: Object.freeze([
      "episode",
      "task_mode",
      "map",
      "observation_mode",
      "episode_limit",
      "seeds",
    ]),
  }),
  Object.freeze({
    id: "this-file",
    title: "This File",
    factIds: Object.freeze([
      "artifact_digest_prefix",
      "ordinary_movement_distance_scale",
    ]),
  }),
]);

/** Fact IDs whose value is an id worth copying. */
const COPYABLE_FACTS = new Set([
  "episode",
  "incoming_transition",
  "artifact_digest_prefix",
]);

/**
 * Cut a long id in the middle so its start and end stay readable.
 *
 * text is any string. A text of at most max characters returns unchanged;
 * a longer one keeps its first and last parts around a "…" so the result is
 * exactly max characters. max defaults to 32 and must be at least 3. Pure.
 *
 * @param {string} text The full id.
 * @param {number} [max] The longest shown length, in characters.
 * @returns {string}
 */
export function middleTruncate(text, max = ID_SHOWN_CHARACTERS) {
  if (text.length <= max) return text;
  const keep = max - 1;
  const head = Math.ceil(keep / 2);
  return `${text.slice(0, head)}…${text.slice(text.length - (keep - head))}`;
}

/**
 * Sort Technical Frame facts into This Frame, This Game and This File.
 *
 * facts is the list from authorizedPresentationTechnicalFacts (each {id,
 * label, value}); it may be empty. Each group keeps the facts' own order.
 * Each fact gains text (the value as a string), shown (text cut to 32
 * characters with middleTruncate) and copy (true for episode, incoming
 * transition and artifact digest ids). A group with no facts is left out.
 * Unknown fact IDs join This Frame. Pure: returns new frozen records.
 *
 * @param {ReadonlyArray<Readonly<{id: string, label: string, value: unknown}>>} facts
 * @returns {ReadonlyArray<Readonly<{id: string, title: string,
 *   facts: ReadonlyArray<Readonly<{id: string, label: string, text: string,
 *   shown: string, copy: boolean}>>}>>}
 */
export function technicalFrameGroups(facts) {
  const known = new Set(TECHNICAL_FRAME_GROUPS.flatMap((group) => group.factIds));
  return Object.freeze(
    TECHNICAL_FRAME_GROUPS.map((group, index) =>
      Object.freeze({
        id: group.id,
        title: group.title,
        facts: Object.freeze(
          facts
            .filter((fact) =>
              group.factIds.includes(fact.id)
                ? true
                : index === 0 && !known.has(fact.id),
            )
            .map((fact) => {
              const text = String(fact.value);
              return Object.freeze({
                id: fact.id,
                label: fact.label,
                text,
                shown: middleTruncate(text),
                copy: COPYABLE_FACTS.has(fact.id),
              });
            }),
        ),
      }),
    ).filter((group) => group.facts.length > 0),
  );
}

/**
 * Draw the Technical Frame in three groups into container.
 *
 * container is emptied and receives one section per group from
 * technicalFrameGroups(facts): a small rune divider title, then one row per
 * fact (label, value as a code chip, and a Copy button for ids). The value
 * chip carries the full value in its title when it was cut. Each row keeps
 * data-technical-fact = the fact ID, is focusable, and registers the fact's
 * help tooltip from explainTechnicalFact (unknown IDs get no tooltip). An
 * empty facts list shows "Technical Frame unavailable.". Copy writes the full
 * value with navigator.clipboard.writeText and reads "Copied" for 1.5 s (one
 * timer per press); if the clipboard is unavailable or refuses, it reads
 * "Copy Failed" for 1.5 s. Side effects: container's DOM, timers, clipboard.
 * Facts come from the replay's own technical frame, a docked researcher panel;
 * nothing here is drawn over the arena.
 *
 * @param {HTMLElement} container The Technical Frame card body.
 * @param {ReadonlyArray<Readonly<{id: string, label: string, value: unknown}>>} facts
 */
export function renderTechnicalFrame(container, facts) {
  const doc = container.ownerDocument;
  const groups = technicalFrameGroups(facts);
  if (groups.length === 0) {
    const empty = doc.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = "Technical Frame unavailable.";
    container.replaceChildren(empty);
    return;
  }
  container.replaceChildren(
    ...groups.map((group) => {
      const section = doc.createElement("section");
      section.className = "ledger-frame__group";
      section.dataset.group = group.id;
      const title = doc.createElement("h3");
      title.className = "ledger-eyebrow ledger-rune";
      title.textContent = group.title;
      section.append(title);
      for (const fact of group.facts) {
        const row = doc.createElement("div");
        row.className = "ledger-frame__fact";
        row.dataset.technicalFact = fact.id;
        row.tabIndex = 0;
        const label = doc.createElement("span");
        label.className = "ledger-frame__label";
        label.textContent = fact.label;
        const value = doc.createElement("code");
        value.className = "ledger-frame__value";
        value.textContent = fact.shown;
        if (fact.shown !== fact.text) value.title = fact.text;
        row.append(label, value);
        if (fact.copy) row.append(copyButton(doc, fact.text, fact.label));
        try {
          registerTooltipOwner(row, explainTechnicalFact(fact.id), {
            inspectable: false,
          });
        } catch (error) {
          // A fact without help text keeps its row; only the tooltip is missing.
          if (!(error instanceof RangeError)) throw error;
        }
        section.append(row);
      }
      return section;
    }),
  );
}

/**
 * Create one Copy button for an id.
 *
 * doc creates the button. text is the full value to copy; label names it for
 * screen readers ("Copy Episode"). Pressing it writes text to the clipboard
 * and shows "Copied" (or "Copy Failed") for 1.5 s, then "Copy" again.
 * Side effects on press: clipboard write and one timer.
 *
 * @param {Document} doc
 * @param {string} text
 * @param {string} label
 * @returns {HTMLButtonElement}
 */
function copyButton(doc, text, label) {
  const button = doc.createElement("button");
  button.type = "button";
  button.className = "ledger-copy";
  button.textContent = "Copy";
  button.setAttribute("aria-label", `Copy ${label}`);
  /** @type {ReturnType<typeof setTimeout> | null} */
  let timer = null;
  /** @param {string} words */
  const show = (words) => {
    button.textContent = words;
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => {
      button.textContent = "Copy";
      timer = null;
    }, COPIED_MS);
  };
  button.addEventListener("click", () => {
    const clipboard = globalThis.navigator?.clipboard;
    if (!clipboard) {
      show("Copy Failed");
      return;
    }
    clipboard.writeText(text).then(
      () => show("Copied"),
      () => show("Copy Failed"),
    );
  });
  return button;
}

/**
 * Return the Full Record chip words for a frame, or null in Oracle View.
 *
 * presentation is an installed authorized presentation (any other value
 * returns null). In Agent POV the chip names the POV agent from the frame's
 * own recipient ID, looked up in the frame's display-identity rows
 * (authorizedPresentationIdentityRows) and labelled by canonicalAgentIdentity,
 * the same way panels.js fullRecordLabel names it: "Full Record · Not Limited
 * to Agent 3's View", never the raw public ID ("agent-3"). When the recipient
 * has no identity row, it reads "...Not Limited to This Agent's View". The
 * words come from agent-identity.js fullRecordChipWords.
 * tooltip explains that the docked cards show every agent while the arena
 * shows only what that agent saw. Agent POV limit: it names only the POV agent
 * itself, never another champion. Pure: reads presentation only.
 *
 * @param {unknown} presentation
 * @returns {Readonly<{text: string, tooltip: string}> | null}
 */
export function fullRecordChip(presentation) {
  if (authorizedPresentationAudience(presentation) !== "agent_pov") return null;
  const frame = /** @type {Record<string, any>} */ (presentation);
  const recipientId = frame.authority?.recipient_public_agent_id;
  const recipient =
    typeof recipientId === "string"
      ? authorizedPresentationIdentityRows(presentation).find(
          (identity) => identity.public_agent_id === recipientId,
        )
      : undefined;
  const name =
    recipient === undefined ? null : canonicalAgentIdentity(recipient.agent).chipText;
  return Object.freeze({
    text: fullRecordChipWords(name),
    tooltip: `These cards show the recorded game for every agent. Only the battlefield and what is drawn over it show what ${recipient === undefined ? "the POV agent" : name} saw.`,
  });
}

/**
 * The Episode Results rows the Match Summary shows for each team, by the
 * stem after "team_a_" or "team_b_", in order.
 */
const MATCH_SUMMARY_STEMS = Object.freeze(["score", "kills", "deaths", "return"]);

/**
 * Build the Match Summary that heads the Metrics tab, or null.
 *
 * presentation is the installed replay frame (its match_summary gives the
 * scores, the threshold, the episode limit and the outcome at this tick; its
 * technical frame gives the simulator tick). metricSummary is the last Python
 * metric summary for this frame, or null while it loads.
 *
 * heading: on the completion frame a verdict, for example "Team B Wins · 17 –
 * 20 · Score Limit Reached · Tick 204" or "Draw · 19 – 16 · No Team Reached
 * 20 · A Lead at Tick 300 Wins Nothing"; before it "Match in Progress · Up to
 * Tick 118". teams: Team A first, then Team B; each lists Score, Total Kills,
 * Total Deaths and Return with the catalog's own labels, values and CSV names
 * from the Episode Results rows. The summary never shows the ending early:
 * when the Scope is Whole Game and the game is not over at this tick, only
 * the scores from match_summary are listed and note says how to see the rest.
 * Missing values are "—". Returns null for a non-TDM game or an unrecognized
 * frame. Pure. A docked researcher panel: it is the full record in every view.
 *
 * @param {unknown} presentation
 * @param {Record<string, any> | null} metricSummary
 * @returns {Readonly<{heading: string, note: string | null,
 *   teams: ReadonlyArray<Readonly<{teamId: 1 | 2, label: string,
 *   values: ReadonlyArray<Readonly<{label: string, text: string,
 *   csv: string | null}>>}>>}> | null}
 */
export function ledgerMatchSummary(presentation, metricSummary) {
  if (!isAuthorizedPresentationFrame(presentation)) return null;
  const match = /** @type {Record<string, any>} */ (presentation).match_summary;
  if (match?.task_mode !== 1) return null;
  const tickFact = authorizedPresentationTechnicalFacts(presentation).find(
    (fact) => fact.id === "simulator_step",
  );
  const tick = typeof tickFact?.value === "number" ? tickFact.value : null;
  const tickText = tick === null ? "" : ` · Tick ${tick}`;
  const [scoreA, scoreB] = match.scores;
  const threshold = match.score_threshold;
  const complete = match.outcome !== "in_progress";
  let heading;
  if (match.outcome === "team_a_win" || match.outcome === "team_b_win") {
    const winner = match.outcome === "team_a_win" ? 1 : 2;
    const reached =
      Math.max(scoreA, scoreB) >= threshold ? " · Score Limit Reached" : "";
    heading = `${teamLabelFromId(winner)} Wins · ${scoreA} – ${scoreB}${reached}${tickText}`;
  } else if (match.outcome === "draw") {
    const both = scoreA >= threshold && scoreB >= threshold;
    const lead =
      !both && scoreA !== scoreB && match.episode_limit
        ? ` · A Lead at Tick ${match.episode_limit} Wins Nothing`
        : "";
    heading = `Draw · ${scoreA} – ${scoreB} · ${both ? "Both Teams Reached" : "No Team Reached"} ${threshold}${lead}`;
  } else {
    heading =
      tick === null ? "Match in Progress" : `Match in Progress · Up to Tick ${tick}`;
  }
  const useRows =
    metricSummary !== null &&
    Array.isArray(metricSummary.statistics) &&
    (metricSummary.scope === "cursor" || complete);
  /** @type {Map<string, Record<string, any>>} */
  const byName = new Map();
  if (useRows) {
    for (const row of metricSummary.statistics) byName.set(row.name, row);
  }
  const teams = /** @type {const} */ ([1, 2]).map((teamId) => {
    const letter = teamId === 1 ? "a" : "b";
    const values = useRows
      ? MATCH_SUMMARY_STEMS.map((stem) => byName.get(`team_${letter}_${stem}`))
          .filter((row) => row !== undefined)
          .map((row) =>
            Object.freeze({
              label: String(row.label),
              text: metricValueText(
                /** @type {{valid: boolean, value: number | null, unit: string}} */ (
                  row
                ),
              ),
              csv: String(row.name),
            }),
          )
      : [
          Object.freeze({
            label: "Score",
            text: String(match.scores[teamId - 1]),
            csv: null,
          }),
        ];
    return Object.freeze({
      teamId,
      label: /** @type {string} */ (teamLabelFromId(teamId)),
      values: Object.freeze(values),
    });
  });
  const note =
    metricSummary !== null && !useRows
      ? "Choose Up to Tick to see kills and deaths here. This summary never shows the ending early."
      : null;
  return Object.freeze({ heading, note, teams: Object.freeze(teams) });
}

/**
 * Draw a Match Summary view into container, or hide container for null.
 *
 * view comes from ledgerMatchSummary. The heading line comes first, then one
 * row per team with its crest chip and each value as "label value" (the CSV
 * name is the value's title), then the note when present. options.subjectChip
 * is passed to createMetricSubjectChips for the crest. Side effects:
 * container's DOM and hidden flag.
 *
 * @param {HTMLElement} container
 * @param {ReturnType<typeof ledgerMatchSummary>} view
 * @param {{subjectChip?: import("./metrics-panel.js").SubjectChipFactory}} [options]
 */
export function renderLedgerMatchSummary(container, view, options = {}) {
  container.hidden = view === null;
  if (view === null) {
    container.replaceChildren();
    return;
  }
  const doc = container.ownerDocument;
  const heading = doc.createElement("p");
  heading.className = "ledger-match__heading";
  heading.textContent = view.heading;
  container.replaceChildren(heading);
  for (const team of view.teams) {
    const row = doc.createElement("div");
    row.className = "ledger-match__team";
    row.dataset.team = team.teamId === 1 ? "a" : "b";
    const chips = createMetricSubjectChips(
      doc,
      [Object.freeze({ kind: /** @type {const} */ ("team"), teamId: team.teamId })],
      { subjectChip: options.subjectChip },
    );
    const name = doc.createElement("span");
    name.className = "sr-only";
    name.textContent = team.label;
    if (chips) row.append(chips);
    row.append(name);
    const list = doc.createElement("dl");
    list.className = "ledger-match__values";
    for (const value of team.values) {
      const term = doc.createElement("dt");
      term.textContent = value.label;
      const detail = doc.createElement("dd");
      detail.textContent = value.text;
      if (value.csv) detail.title = value.csv;
      list.append(term, detail);
    }
    row.append(list);
    container.append(row);
  }
  if (view.note) {
    const note = doc.createElement("p");
    note.className = "ledger-match__note";
    note.textContent = view.note;
    container.append(note);
  }
}

/**
 * @typedef {{
 *   element: HTMLElement,
 *   activeTab: () => string,
 *   selectTab: (id: string, options?: {focus?: boolean}) => void,
 *   update: (input?: {presentation?: unknown,
 *     metricSummary?: Record<string, any> | null,
 *     summaryEnabled?: boolean}) => void,
 * }} LedgerRail
 */

/** One rail controller per rail element, so a second mount reuses it. */
const railByElement = new WeakMap();

/**
 * Return whether node is a folding <details> card. Works without the browser's
 * element classes (tests use plain objects with tagName).
 *
 * @param {Element | null} node
 * @returns {node is HTMLDetailsElement}
 */
function isDetails(node) {
  return node?.tagName === "DETAILS";
}

/**
 * Build the Replay Viewer rail and the Ledger Header, once per page.
 *
 * doc is the page document. It must hold index.html's #replay-rail slot with
 * #replay-rail-tabs and the six #replay-rail-<tab> panels; without them this
 * returns null and changes nothing. options defaults to {}:
 * options.initialTab is the first open tab ("agent" by default; an unknown
 * name falls back to it); options.subjectChip builds the crest chips (see
 * createMetricSubjectChips; the built-in chip by default).
 *
 * What it does: shows the rail; fills #replay-rail-tabs with a tab list (one
 * role="tab" button per LEDGER_TABS entry, each with its hover and focus help
 * from the entry's label and help; Left, Right, Home and End move between tabs
 * and open them); moves each card in LEDGER_TAB_CONTENT into
 * its panel, keeping its ID; moves every visually hidden live region
 * (.sr-only with aria-live) out of the panels into one always-shown holder,
 * so an announcement is never lost in a hidden tab; adds a hidden Full Record
 * chip to the top of each full-record panel; and dresses the Metrics card as
 * the ledger (mountLedgerHeader). Opening a tab opens its folding cards. The
 * Metrics card is open only while its tab is shown, so metric work waits
 * until the researcher asks for it, as today. Calling it again for the same
 * document returns the same controller and changes nothing.
 *
 * Returns the LedgerRail controller: element (the rail), activeTab(),
 * selectTab(id, {focus}) and update({presentation, metricSummary,
 * summaryEnabled}), which refreshes the Full Record chips, the Scope switch's
 * tick and the Match Summary (drawn only when the input has a metricSummary
 * key, which may be null while it loads; summaryEnabled false, the Match
 * Summary switch off, hides it instead). In Agent POV update shows each chip and registers its help tip;
 * outside Agent POV it hides each chip, empties its text and removes its
 * aria-description and data-tooltip-owner, so Oracle -> POV -> Oracle leaves
 * the chips exactly as a fresh render. Side effects: DOM moves and listeners in doc; no storage, timers
 * or network. The rail is a docked researcher panel: in Agent POV it shows
 * the full record under the chip and draws nothing over the arena.
 *
 * @param {Document} doc
 * @param {{initialTab?: string,
 *   subjectChip?: import("./metrics-panel.js").SubjectChipFactory}} [options]
 * @returns {LedgerRail | null}
 */
export function mountLedgerRail(doc, options = {}) {
  const railElement = doc.getElementById("replay-rail");
  const tabsHost = doc.getElementById("replay-rail-tabs");
  if (!railElement || !tabsHost) return null;
  const rail = railElement;
  const existing = railByElement.get(rail);
  if (existing) return existing;
  /** @type {Map<string, HTMLElement>} */
  const panels = new Map();
  for (const tab of LEDGER_TABS) {
    const panel = doc.getElementById(tab.panelId);
    if (!panel) return null;
    panels.set(tab.id, panel);
  }
  let active = "";
  rail.classList.add("ledger-rail");
  tabsHost.setAttribute("role", "tablist");
  tabsHost.setAttribute("aria-label", "Inspector Tabs");
  tabsHost.className = "ledger-rail__tabs";
  const live = doc.createElement("div");
  live.className = "ledger-rail__live";
  /** @type {Map<string, HTMLButtonElement>} */
  const buttons = new Map();
  /** @type {Map<string, HTMLElement>} */
  const chips = new Map();
  for (const tab of LEDGER_TABS) {
    const panel = /** @type {HTMLElement} */ (panels.get(tab.id));
    const button = doc.createElement("button");
    button.type = "button";
    button.id = `replay-rail-tab-${tab.id}`;
    button.className = "ledger-rail__tab";
    button.dataset.tab = tab.id;
    button.textContent = tab.label;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", tab.panelId);
    button.addEventListener("click", () => selectTab(tab.id));
    registerTooltipOwner(
      button,
      {
        kind: "control",
        tone: "information",
        accent: "none",
        id: `ledger-tab:${tab.id}`,
        title: tab.label,
        summary: tab.help,
        rows: [],
        sections: [],
        metadata: { compact: true, full: false },
        anchor: "element",
      },
      { inspectable: false },
    );
    buttons.set(tab.id, button);
    tabsHost.append(button);
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", button.id);
    panel.classList.add("ledger-rail__panel");
    panel.tabIndex = 0;
    if (tab.fullRecord) {
      const chip = doc.createElement("p");
      chip.className = "ledger-full-record";
      chip.tabIndex = 0;
      chip.hidden = true;
      panel.append(chip);
      chips.set(tab.id, chip);
    }
    for (const id of LEDGER_TAB_CONTENT[
      /** @type {keyof typeof LEDGER_TAB_CONTENT} */ (tab.id)
    ]) {
      const card = doc.getElementById(id);
      if (!card) continue;
      card.classList.add("ledger-card");
      panel.append(card);
    }
    for (const region of panel.querySelectorAll(".sr-only[aria-live]")) {
      live.append(region);
    }
  }
  rail.append(live);
  tabsHost.addEventListener("keydown", (/** @type {KeyboardEvent} */ event) => {
    const order = LEDGER_TABS.map((tab) => tab.id);
    const index = order.indexOf(/** @type {typeof order[number]} */ (active));
    const next =
      event.key === "ArrowRight"
        ? order[(index + 1) % order.length]
        : event.key === "ArrowLeft"
          ? order[(index - 1 + order.length) % order.length]
          : event.key === "Home"
            ? order[0]
            : event.key === "End"
              ? order[order.length - 1]
              : null;
    if (next === null) return;
    event.preventDefault();
    selectTab(next, { focus: true });
  });
  const header = mountLedgerHeader(doc);
  const matchSummary = doc.getElementById("ledger-match-summary");

  /**
   * Show tab id and hide the others; open its folding cards. The Metrics card
   * opens only while shown and closes when left. focus moves keyboard focus
   * to the tab button.
   *
   * @param {string} id @param {{focus?: boolean}} [selectOptions]
   */
  function selectTab(id, selectOptions = {}) {
    const chosen = panels.has(id) ? id : DEFAULT_TAB;
    for (const tab of LEDGER_TABS) {
      const shown = tab.id === chosen;
      const button = /** @type {HTMLButtonElement} */ (buttons.get(tab.id));
      button.setAttribute("aria-selected", String(shown));
      button.tabIndex = shown ? 0 : -1;
      /** @type {HTMLElement} */ (panels.get(tab.id)).hidden = !shown;
    }
    rail.dataset.activeTab = chosen;
    if (chosen !== active) {
      const metrics = doc.getElementById("evaluation-metrics");
      if (isDetails(metrics)) metrics.open = chosen === "metrics";
      for (const id of LEDGER_TAB_CONTENT[
        /** @type {keyof typeof LEDGER_TAB_CONTENT} */ (chosen)
      ]) {
        const card = doc.getElementById(id);
        if (isDetails(card) && !card.open) card.open = true;
      }
    }
    active = chosen;
    if (selectOptions.focus) buttons.get(chosen)?.focus();
  }

  /** @type {LedgerRail} */
  const controller = Object.freeze({
    element: rail,
    activeTab: () => active,
    selectTab,
    update(input = {}) {
      const presentation = input.presentation ?? null;
      const chip = fullRecordChip(presentation);
      for (const node of chips.values()) {
        node.hidden = chip === null;
        node.textContent = chip?.text ?? "";
        if (chip === null) {
          // Oracle View: drop the old POV agent's help so the chip matches a
          // fresh render. The tooltip controller only finds owners that carry
          // data-tooltip-owner, so removing it retires the registration.
          node.removeAttribute("aria-description");
          node.removeAttribute("data-tooltip-owner");
        } else {
          registerTooltipOwner(node, {
            kind: "ledger-help",
            tone: "information",
            accent: "none",
            id: "ledger-help:full-record",
            title: "Full Record",
            summary: chip.tooltip,
            rows: [],
            sections: [],
            metadata: { compact: true, full: true },
            anchor: "element",
          });
        }
      }
      header?.refresh(presentation);
      if (matchSummary && input.summaryEnabled === false) {
        // The Match Summary switch (ledger_summary) is off: the metrics alone.
        renderLedgerMatchSummary(matchSummary, null);
      } else if (matchSummary && "metricSummary" in input) {
        renderLedgerMatchSummary(
          matchSummary,
          ledgerMatchSummary(presentation, input.metricSummary ?? null),
          { subjectChip: options.subjectChip },
        );
      }
    },
  });
  railByElement.set(rail, controller);
  rail.hidden = false;
  tabsHost.hidden = false;
  selectTab(
    LEDGER_TABS.some((tab) => tab.id === options.initialTab)
      ? /** @type {string} */ (options.initialTab)
      : DEFAULT_TAB,
  );
  return controller;
}

/**
 * Dress the Metrics card as the ledger, once.
 *
 * doc must hold today's #evaluation-metrics card and its controls
 * (#metric-scope, #metric-selection, #metric-view-field, #metric-view,
 * #metric-search-area, #metric-rows); without the card this returns null.
 * It adds the ledger class, inserts the #ledger-match-summary section at the
 * top of the card, gathers the Scope, Topic, View and search controls into
 * one div.ledger-header placed just before the card (its own block, so in the
 * Replay Viewer's Metrics tab only the card under it scrolls; ledger.css),
 * adds the Topic eyebrow #metric-topic-section and two
 * switches (#metric-scope-switch, #metric-view-switch) that mirror the native
 * selects, and folds the two reading notes under the table into an "About
 * These Numbers" disclosure. The native Scope and View selects stay in the
 * page with their IDs, values and change events (main.js reads them), but are
 * visually hidden and taken out of the tab order and the accessibility tree,
 * because the switches replace them. No row, value, topic or note text
 * changes. Returns {refresh(presentation)} which rewrites the Scope switch's
 * first label as "Up to Tick N" from the frame's simulator tick ("Up to
 * Current Tick" when unknown). Side effects: DOM moves and listeners.
 *
 * @param {Document} doc
 * @returns {{refresh: (presentation: unknown) => void} | null}
 */
export function mountLedgerHeader(doc) {
  const card = doc.getElementById("evaluation-metrics");
  const scope = /** @type {HTMLSelectElement | null} */ (
    doc.getElementById("metric-scope")
  );
  const topic = /** @type {HTMLSelectElement | null} */ (
    doc.getElementById("metric-selection")
  );
  const viewField = doc.getElementById("metric-view-field");
  const view = /** @type {HTMLSelectElement | null} */ (
    doc.getElementById("metric-view")
  );
  const search = doc.getElementById("metric-search-area");
  const rows = doc.getElementById("metric-rows");
  if (!card || !scope || !topic || !viewField || !view || !search || !rows) return null;
  const body = /** @type {HTMLElement} */ (scope.parentElement);
  card.classList.add("ledger");
  const summary = doc.createElement("section");
  summary.id = "ledger-match-summary";
  summary.className = "ledger-match";
  summary.setAttribute("aria-label", "Match Summary");
  summary.hidden = true;
  const header = doc.createElement("div");
  header.className = "ledger-header";
  // The header is its own block before the card (owner, 4 Oct 2026): in the
  // Replay Viewer's Metrics tab only the card under it scrolls.
  body.prepend(summary);
  card.before(header);
  /** @param {HTMLSelectElement} select */
  const hideNative = (select) => {
    select.classList.add("ledger-native");
    select.tabIndex = -1;
    select.setAttribute("aria-hidden", "true");
    const label = doc.querySelector(`label[for="${select.id}"]`);
    if (label) /** @type {HTMLElement} */ (label).hidden = true;
  };
  const scopeLabel = doc.querySelector('label[for="metric-scope"]');
  const topicLabel = doc.querySelector('label[for="metric-selection"]');
  const scopeSwitch = doc.createElement("div");
  scopeSwitch.id = "metric-scope-switch";
  scopeSwitch.className = "ledger-switch";
  scopeSwitch.setAttribute("role", "radiogroup");
  scopeSwitch.setAttribute("aria-label", "Scope");
  const eyebrow = doc.createElement("span");
  eyebrow.id = "metric-topic-section";
  eyebrow.className = "ledger-eyebrow";
  eyebrow.setAttribute("aria-hidden", "true");
  const viewSwitch = doc.createElement("div");
  viewSwitch.id = "metric-view-switch";
  viewSwitch.className = "ledger-switch";
  viewSwitch.setAttribute("role", "radiogroup");
  viewSwitch.setAttribute("aria-label", "View");
  viewField.append(viewSwitch);
  const topicRow = doc.createElement("div");
  topicRow.className = "ledger-header__topic";
  if (topicLabel) {
    topicLabel.classList.add("sr-only");
    topicRow.append(topicLabel);
  }
  topicRow.append(eyebrow, topic);
  if (scopeLabel) header.append(scopeLabel);
  header.append(scope, scopeSwitch, topicRow, viewField, search);
  hideNative(scope);
  hideNative(view);
  const notes = [];
  for (let node = rows.nextElementSibling; node; node = node.nextElementSibling) {
    if (node.classList.contains("section-note")) notes.push(node);
  }
  if (notes.length > 0) {
    const fold = doc.createElement("details");
    fold.className = "ledger-notes";
    const title = doc.createElement("summary");
    title.textContent = "About These Numbers";
    fold.append(title, ...notes);
    rows.after(fold);
  }
  let tickLabel = "Up to Current Tick";
  const syncScope = () =>
    renderChoiceSwitch(scopeSwitch, scope, [
      { name: "cursor", label: tickLabel },
      { name: "final", label: "Whole Game", note: "Shows the ending too." },
    ]);
  const syncTopic = () => {
    const option = topic.selectedOptions?.[0];
    const group = option?.parentElement;
    if (group?.tagName === "OPTGROUP") {
      eyebrow.textContent = /** @type {HTMLOptGroupElement} */ (group).label;
    }
  };
  scope.addEventListener("change", syncScope);
  topic.addEventListener("change", syncTopic);
  view.addEventListener("change", () => {
    if (viewSwitch.children.length > 0) {
      const views = [...view.options].map((option) => ({
        name: option.value,
        label: option.text,
      }));
      renderChoiceSwitch(viewSwitch, view, views);
    }
  });
  syncScope();
  syncTopic();
  return Object.freeze({
    /** @param {unknown} presentation */
    refresh(presentation) {
      const tick = authorizedPresentationTechnicalFacts(presentation).find(
        (fact) => fact.id === "simulator_step",
      )?.value;
      tickLabel =
        typeof tick === "number" ? `Up to Tick ${tick}` : "Up to Current Tick";
      syncScope();
    },
  });
}
