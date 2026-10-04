/**
 * @file Display the metric catalog and values supplied by Python. This module
 * orders rows using catalog keys, builds navigation/search displays and
 * registers metric tooltips. It does not compute measurements, change missing
 * values to zero or decide which scientific measurements apply.
 *
 * The Replay Viewer's Metrics tab (the rail tab owned by ledger.js) calls
 * these renderers. The ledger look adds small subject chips (team crest or
 * class emblem plus "Agent 3") beside each row's subject. The chips only show
 * identity from the catalog's own subject IDs and agent list; the exact host
 * subject text stays in the row for screen readers and text search. Every
 * topic, row, section, definition, unit, value and CSV name is unchanged.
 * Metrics are a docked researcher panel: they show the full record in every
 * view and are never drawn on or over the arena.
 */
import { createSvgIcon } from "./icons.js";
import { registerTooltipOwner } from "./tooltip.js";

const TOOLTIP_UNITS = Object.freeze({
  health: "Health points (HP)",
  steps: "Ticks",
  agent_steps: "Ticks added across agents",
  pair_steps: "Pair measurements",
  fraction: "Share: 1 means 100%",
  indicator: "1 means yes; 0 means no",
  ratio: "One number divided by another",
  distance: "Map distance units",
  score: "Points",
});

/**
 * Turn string value into a title by replacing underscores with spaces and
 * capitalizing word initials. Return text only; exact CSV names stay unchanged
 * in records and exports. The caller supplies a string.
 *
 * @param {string} value
 */
export function metricLabel(value) {
  return value.replaceAll("_", " ").replace(/\b\w/gu, (letter) => letter.toUpperCase());
}

/**
 * The five ledger section titles, by the index metricSection returns: 0
 * episode and team rows, 1 agent rows, 2 recipient totals, 3 source-to-
 * recipient rows, 4 ally pairs. The words are today's exact section names;
 * the ledger's CSS shows them as small capital eyebrows.
 */
export const METRIC_SECTION_TITLES = Object.freeze([
  "Episode and Team Totals",
  "Agent Details",
  "Recipient Totals",
  "Source-to-Recipient Details",
  "Ally Pair Details",
]);

/**
 * Format one statistics row's value exactly as the metric table shows it.
 *
 * A valid row's number uses en-GB grouping with up to four decimals for the
 * fraction and ratio units and up to two for every other unit. An invalid row
 * (row.valid is false, so value is null) returns an em dash "—": a blank is
 * never shown as zero. Pure: no DOM, no input change.
 *
 * @param {{valid: boolean, value: number | null, unit: string}} row A row from
 *   a Python metric summary.
 * @returns {string} The display text.
 */
export function metricValueText(row) {
  return row.valid && typeof row.value === "number"
    ? row.value.toLocaleString("en-GB", {
        maximumFractionDigits: row.unit === "fraction" || row.unit === "ratio" ? 4 : 2,
      })
    : "—";
}

/**
 * @typedef {Readonly<{kind: "team", teamId: 1 | 2}>
 *   | Readonly<{kind: "agent", slot: number, classId: number, className: string,
 *     teamId: 1 | 2}>} MetricSubjectPart
 */

/**
 * Builds one subject chip for a metric part. The integrator passes the shared
 * token chip here so every panel draws the same chip; it receives the
 * document and one MetricSubjectPart and returns a detached element.
 *
 * @typedef {(doc: Document, part: MetricSubjectPart) => Element} SubjectChipFactory
 */

/**
 * Split a metric row's subject into chip parts from the catalog's own IDs.
 *
 * row.scope and row.subjects say who the row is about: "episode" has no
 * subject; "team" is [teamId]; "agent" is [slot]; "team_recipient" is
 * [teamId, recipient slot]; "source_recipient" is [source slot, recipient
 * slot]; "ally_pair" is [slot, slot]. agents is the catalog's agent list
 * (slot 0-9, class_id, class_name, team_id 1 or 2); it defaults to empty.
 * Return a frozen list of parts in reading order. Return an empty list when
 * the row has no subject or any ID cannot be matched (an unknown slot, a team
 * ID other than 1 or 2): the caller then shows the plain host text, so a chip
 * is never guessed. Pure: reads only row and agents, changes neither. Metrics
 * are the full researcher record, so no Agent POV filter applies here.
 *
 * @param {Record<string, any>} row A catalog or summary statistics row.
 * @param {ReadonlyArray<Record<string, any>>} [agents] The catalog agent list.
 * @returns {ReadonlyArray<MetricSubjectPart>}
 */
export function metricSubjectParts(row, agents = []) {
  const subjects = Array.isArray(row.subjects) ? row.subjects : [];
  /** @param {unknown} teamId */
  const team = (teamId) =>
    teamId === 1 || teamId === 2
      ? Object.freeze({ kind: /** @type {const} */ ("team"), teamId })
      : null;
  /** @param {unknown} slot */
  const agent = (slot) => {
    const found = agents.find((item) => item.slot === slot);
    if (!found || (found.team_id !== 1 && found.team_id !== 2)) return null;
    return Object.freeze({
      kind: /** @type {const} */ ("agent"),
      slot: /** @type {number} */ (slot),
      classId: found.class_id,
      className: String(found.class_name),
      teamId: /** @type {1 | 2} */ (found.team_id),
    });
  };
  /** @type {Array<MetricSubjectPart | null>} */
  let parts;
  if (row.scope === "episode") parts = [];
  else if (row.scope === "team") parts = [team(subjects[0])];
  else if (row.scope === "team_recipient")
    parts = [team(subjects[0]), agent(subjects[1])];
  else parts = subjects.map(agent);
  if (parts.length === 0 || parts.includes(null)) return Object.freeze([]);
  return Object.freeze(/** @type {MetricSubjectPart[]} */ (parts));
}

/**
 * Build the ledger's small subject chips for one metric row.
 *
 * doc creates the nodes. parts comes from metricSubjectParts. options.join is
 * the mark between two parts: "→" (default) for a source and its recipient,
 * "↔" for an ally pair. options.subjectChip, when given, builds each chip
 * instead of the built-in one (the integrator passes the shared token chip so
 * every panel draws one chip look); it receives (doc, part) and returns an
 * element. The built-in chip is a span.ledger-chip with data-team "a" or "b",
 * the team crest or class emblem from icons.js, and the words "Team A" or
 * "Agent 3"; options.withClass adds the class name under a single agent chip.
 * The whole group is aria-hidden: the caller keeps the exact host subject
 * text beside it for screen readers. Return a detached span.ledger-subject__
 * chips, or null for an empty parts list. No other DOM is touched.
 *
 * @param {Document} doc The document that creates the nodes.
 * @param {ReadonlyArray<MetricSubjectPart>} parts The chip parts.
 * @param {{join?: string, withClass?: boolean,
 *   subjectChip?: SubjectChipFactory}} [options]
 * @returns {HTMLElement | null}
 */
export function createMetricSubjectChips(doc, parts, options = {}) {
  if (parts.length === 0) return null;
  const group = doc.createElement("span");
  group.className = "ledger-subject__chips";
  group.setAttribute("aria-hidden", "true");
  parts.forEach((part, index) => {
    if (index > 0) {
      const join = doc.createElement("span");
      join.className = "ledger-subject__join";
      join.textContent = options.join ?? "→";
      group.append(join);
    }
    if (options.subjectChip) {
      group.append(options.subjectChip(doc, part));
      return;
    }
    const chip = doc.createElement("span");
    chip.className = "ledger-chip";
    chip.dataset.team = part.teamId === 1 ? "a" : "b";
    const glyph =
      part.kind === "team"
        ? part.teamId === 1
          ? "team-a"
          : "team-b"
        : `class-${part.className.toLowerCase()}`;
    chip.append(createSvgIcon(doc, glyph, { className: "ledger-chip__emblem" }));
    const words = doc.createElement("span");
    words.className = "ledger-chip__name";
    words.textContent =
      part.kind === "team"
        ? part.teamId === 1
          ? "Team A"
          : "Team B"
        : `Agent ${part.slot}`;
    chip.append(words);
    if (options.withClass && part.kind === "agent" && parts.length === 1) {
      const role = doc.createElement("small");
      role.className = "ledger-chip__class";
      role.textContent = part.className;
      chip.append(role);
    }
    group.append(chip);
  });
  return group;
}

/**
 * Mirror a native select as a two-part (or longer) switch of buttons.
 *
 * host receives one button per choice, in order (role "radio", aria-checked
 * on the chosen one, roving tabIndex). choices are {name, label, note?}; name
 * must be one of select's option values; note adds a small line under the
 * label (for example "Shows the ending too."). Buttons are rebuilt only when
 * the choice names change (cached in host.dataset.choices), so a focused
 * button keeps focus; labels, notes and the checked state are refreshed in
 * place on every call. Clicking a button sets
 * select.value and dispatches one bubbling "change" event, so the page's
 * existing select listeners stay the one owner of what happens next. Left and
 * Right arrows move the choice. A disabled select disables every button.
 * Return undefined. Side effects: host's children and listeners.
 *
 * @param {HTMLElement} host The switch container (role radiogroup).
 * @param {HTMLSelectElement} select The native select that keeps the value.
 * @param {ReadonlyArray<{name: string, label: string, note?: string}>} choices
 */
export function renderChoiceSwitch(host, select, choices) {
  const doc = host.ownerDocument;
  const key = JSON.stringify(choices.map((choice) => choice.name));
  if (host.dataset.choices !== key) {
    host.replaceChildren(
      ...choices.map((choice) => {
        const button = doc.createElement("button");
        button.type = "button";
        button.className = "ledger-switch__part";
        button.dataset.value = choice.name;
        button.setAttribute("role", "radio");
        button.append(doc.createElement("span"), doc.createElement("small"));
        button.children[1].className = "ledger-switch__note";
        button.addEventListener("click", () => {
          if (select.disabled || select.value === choice.name) return;
          select.value = choice.name;
          select.dispatchEvent(new Event("change", { bubbles: true }));
        });
        return button;
      }),
    );
    host.dataset.choices = key;
  }
  if (host.dataset.keys !== "on") {
    host.dataset.keys = "on";
    host.addEventListener("keydown", (/** @type {KeyboardEvent} */ event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const buttons = /** @type {HTMLButtonElement[]} */ ([...host.children]);
      const index = buttons.findIndex(
        (button) => button.dataset.value === select.value,
      );
      const next = buttons[index + (event.key === "ArrowRight" ? 1 : -1)];
      if (!next) return;
      event.preventDefault();
      next.click();
      next.focus();
    });
  }
  const buttons = /** @type {HTMLButtonElement[]} */ ([...host.children]);
  buttons.forEach((button, index) => {
    button.children[0].textContent = choices[index].label;
    button.children[1].textContent = choices[index].note ?? "";
    /** @type {HTMLElement} */ (button.children[1]).hidden = !choices[index].note;
    const checked = button.dataset.value === select.value;
    button.setAttribute("aria-checked", String(checked));
    button.tabIndex = checked ? 0 : -1;
    button.disabled = select.disabled;
  });
}

/**
 * Return row's display section index: 0 for episode/team, 2 for recipient,
 * 3 for source-to-recipient, 4 for ally pairs, and 1 otherwise. Scope/role
 * fields come from the catalog; no record is changed or revalidated.
 *
 * @param {Record<string, any>} row
 */
function metricSection(row) {
  if (row.scope === "team" || row.scope === "episode") return 0;
  if (row.scope === "team_recipient" || row.subject_role === "recipient") return 2;
  if (row.scope === "source_recipient") return 3;
  if (row.scope === "ally_pair") return 4;
  return 1;
}

/**
 * Apply row.topic_text[topic] to a shallow row copy when present. All topics
 * may override description/subtitle; only the five named Ultimate topics may
 * also override label, numerator/denominator, guidance and missing_when.
 * Without topic text, return the original row. Inputs and metric values are
 * unchanged; nested data remains shared.
 *
 * @param {Record<string, any>} row @param {string} topic
 */
function metricText(row, topic) {
  const text = row.topic_text?.[topic];
  if (!text) return row;
  const fields = ["description", "subtitle"];
  // Only the five named ability topics can change their row and tooltip words.
  if (
    [
      "ultimate_mage",
      "ultimate_warrior",
      "ultimate_hunter",
      "ultimate_rogue",
      "ultimate_priest",
    ].includes(topic)
  ) {
    fields.push("label", "numerator", "denominator", "guidance", "missing_when");
  }
  const result = { ...row };
  for (const field of fields) {
    if (text[field] !== undefined) result[field] = text[field];
  }
  return result;
}

/**
 * Select summary.statistics for the exact topic/view location. Omit rows
 * whose applicable field is false, keep missing values, and sort by Python's
 * topic_order tuple then row.order. Return a new array with topic wording
 * applied. The caller supplies a valid catalog summary; no independent
 * metric calculation, shape validation or mutation occurs.
 *
 * @param {Record<string, any>} summary @param {string} topic @param {string} view
 */
export function selectedMetricRows(summary, topic, view) {
  const rows = summary.statistics.filter(
    (/** @type {Record<string, any>} */ row) =>
      row.applicable !== false &&
      row.locations.some(
        (/** @type {{topic: string, view: string}} */ location) =>
          location.topic === topic && location.view === view,
      ),
  );
  // Python owns topic order, including related rows and their section order.
  return rows
    .sort(
      (
        /** @type {Record<string, any>} */ first,
        /** @type {Record<string, any>} */ second,
      ) => {
        const a = first.topic_order[topic];
        const b = second.topic_order[topic];
        for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
          const difference = (a[index] ?? 0) - (b[index] ?? 0);
          if (difference) return difference;
        }
        return first.order - second.order;
      },
    )
    .map((/** @type {Record<string, any>} */ row) => metricText(row, topic));
}

/**
 * Update topic selection and view selects from the supplied topics catalog.
 *
 * selection groups topics by section; view shows the selected topic's views.
 * Retain existing valid choices, otherwise select each list's first choice.
 * Cache JSON inventories in element datasets to avoid rebuilding equal lists.
 * Hide viewField for a single view. topics and each views list must be
 * nonempty/valid; an unresolved topic throws TypeError. Return undefined and
 * change only the DOM, not the catalog.
 *
 * Ledger look: when the page has the ledger header (mounted by ledger.js),
 * also write the chosen topic's section into #metric-topic-section (the small
 * eyebrow above Topic) and mirror the view choices into #metric-view-switch
 * (a "Totals | By Recipient" switch) with renderChoiceSwitch. Without those
 * elements nothing else changes.
 *
 * @param {HTMLSelectElement} selection
 * @param {HTMLSelectElement} view
 * @param {HTMLElement} viewField
 * @param {Record<string, any>[]} topics
 */
export function renderMetricNavigation(selection, view, viewField, topics) {
  const inventory = JSON.stringify(topics);
  if (selection.dataset.inventory !== inventory) {
    const selected = selection.value;
    /** @type {Map<string, HTMLOptGroupElement>} */
    const sections = new Map();
    for (const topic of topics) {
      let section = sections.get(topic.section);
      if (!section) {
        section = document.createElement("optgroup");
        section.label = topic.section;
        sections.set(topic.section, section);
      }
      section.append(new Option(topic.label, topic.name));
    }
    selection.replaceChildren(...sections.values());
    selection.value = topics.some((topic) => topic.name === selected)
      ? selected
      : topics[0].name;
    selection.dataset.inventory = inventory;
  }
  const topic = topics.find((item) => item.name === selection.value);
  if (!topic) throw new TypeError("Unknown measurement topic.");
  const views = JSON.stringify(topic.views);
  if (view.dataset.inventory !== views) {
    const selected = view.value;
    view.replaceChildren(
      ...topic.views.map(
        (/** @type {{name: string, label: string}} */ item) =>
          new Option(item.label, item.name),
      ),
    );
    view.value = topic.views.some(
      (/** @type {{name: string}} */ item) => item.name === selected,
    )
      ? selected
      : topic.views[0].name;
    view.dataset.inventory = views;
  }
  viewField.hidden = topic.views.length === 1;
  const doc = selection.ownerDocument;
  const eyebrow = doc?.getElementById?.("metric-topic-section");
  if (eyebrow) eyebrow.textContent = topic.section;
  const viewSwitch = doc?.getElementById?.("metric-view-switch");
  if (viewSwitch) renderChoiceSwitch(viewSwitch, view, topic.views);
}

export {
  buildMetricSearchIndex,
  findMeasurements,
  searchMeasurements,
} from "./metric-search.js";

/**
 * Return row's primary topic label, adding its view label when needed.
 * topics must contain primary_topic and primary_view. An unknown topic
 * throws TypeError; a missing required view may also fail. No lookup fallback
 * or input mutation occurs.
 *
 * @param {Record<string, any>} row @param {Record<string, any>[]} topics
 */
function primaryLocationLabel(row, topics) {
  const topic = topics.find((item) => item.name === row.primary_topic);
  if (!topic) throw new TypeError("Unknown primary measurement topic.");
  const view = topic.views.find(
    (/** @type {{name: string}} */ item) => item.name === row.primary_view,
  );
  return topic.views.length === 1 ? topic.label : `${topic.label}: ${view.label}`;
}

/**
 * Replace container with the first limit matches as accessible result buttons.
 *
 * matches is already ranked; topics resolves each primary location. limit is
 * a caller-supplied nonnegative slice bound. Clicking a button calls select
 * with the original row. Show applicability notices without dropping results.
 * Return undefined; create DOM/listeners but do not compute or alter values.
 *
 * Each result reads as three lines (the ledger look): line 1 the subject's
 * chips (when options.agents is given and the subject resolves) and the
 * measure's label; line 2 who it is about and where it lives ("Agent ID 4 ·
 * Hunter · Team B → Agent ID 0 · Mage · Team A · Ability Activations: By
 * Recipient"); line 3 the exact CSV name. A row Python marks inapplicable adds
 * the tag "Not Applicable To This Roster". options defaults to {}:
 * options.agents is the catalog agent list (default none, so no chips);
 * options.subjectChip is passed to createMetricSubjectChips. Results are the full researcher record and never
 * follow Agent POV.
 *
 * @param {HTMLElement} container
 * @param {Record<string, any>[]} matches
 * @param {Record<string, any>[]} topics
 * @param {number} limit
 * @param {(row: Record<string, any>) => void} select
 * @param {{agents?: ReadonlyArray<Record<string, any>>,
 *   subjectChip?: SubjectChipFactory}} [options]
 */
export function renderMetricSearchResults(
  container,
  matches,
  topics,
  limit,
  select,
  options = {},
) {
  const doc = container.ownerDocument;
  const list = doc.createElement("ol");
  list.className = "metric-search-results";
  list.setAttribute("role", "presentation");
  for (const row of matches.slice(0, limit)) {
    const item = doc.createElement("li");
    item.setAttribute("role", "presentation");
    const button = doc.createElement("button");
    button.type = "button";
    button.id = `metric-search-option-${row.order}`;
    button.tabIndex = -1;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", "false");
    button.dataset.measurement = row.name;
    const title = doc.createElement("span");
    title.className = "metric-search-results__title";
    const chips = createMetricSubjectChips(
      doc,
      metricSubjectParts(row, options.agents),
      {
        join: row.scope === "ally_pair" ? "↔" : "→",
        subjectChip: options.subjectChip,
      },
    );
    if (chips) title.append(chips);
    title.append(element("strong", row.label));
    button.append(
      title,
      element("small", `${row.subject} · ${primaryLocationLabel(row, topics)}`),
      element("code", row.name),
    );
    if (!row.applicable) {
      const tag = element("small", "Not Applicable To This Roster");
      tag.className = "metric-search-results__tag";
      button.append(tag);
    }
    button.addEventListener("click", () => select(row));
    item.append(button);
    list.append(item);
  }
  container.replaceChildren(list);
}

/**
 * Show and focus a metric definition in container. row uses primary-topic
 * wording; topics supplies its location label. Include guidance, units and
 * not-applicable explanation. Return undefined and replace DOM children.
 * The owning search flow decides when this definition view is appropriate;
 * invalid catalog references may throw.
 *
 * @param {HTMLElement} container @param {Record<string, any>} row @param {Record<string, any>[]} topics
 */
export function renderMetricDefinition(container, row, topics) {
  row = metricText(row, row.primary_topic);
  const title = element("strong", row.label);
  const fields = document.createElement("dl");
  for (const { label, value } of metricTooltipRows(row)) {
    fields.append(element("dt", label), element("dd", value));
  }
  container.replaceChildren(
    title,
    element("p", row.subject),
    element("p", row.description),
    element("p", row.not_applicable_reason ?? "Not applicable to this roster."),
    element("p", `Primary location: ${primaryLocationLabel(row, topics)}.`),
    fields,
  );
  container.hidden = false;
  container.focus();
}

/**
 * Build new label/value/metadata records for row's metric tooltip. Include
 * exact CSV name, units, directed From/To identities, a fraction's numerator
 * and denominator, guidance and missing-value rule. Team IDs and global
 * agent IDs come from the catalog subjects; no identities or values are
 * inferred from the current scene. Return a mutable array without editing row.
 *
 * @param {Record<string, any>} row
 */
export function metricTooltipRows(row) {
  const directed = row.scope === "source_recipient" || row.scope === "team_recipient";
  const rows = [
    ["CSV Column", row.name],
    [
      "Unit",
      TOOLTIP_UNITS[/** @type {keyof typeof TOOLTIP_UNITS} */ (row.unit)] ??
        metricLabel(row.unit),
    ],
  ];
  if (directed) {
    rows.push(
      [
        "From",
        row.scope === "team_recipient"
          ? `Team ${row.subjects[0] === 1 ? "A" : "B"}`
          : `Agent ID ${row.subjects[0]}`,
      ],
      ["To", `Agent ID ${row.subjects[1]}`],
    );
  }
  if (row.denominator) {
    rows.push(["Numerator", row.numerator], ["Denominator", row.denominator]);
  }
  rows.push(["How to Read It", row.guidance], ["Blank When", row.missing_when]);
  return rows.map(([label, value]) => ({
    label,
    value,
    metadata: { compact: true, full: true },
  }));
}

/**
 * Create a detached page-document element with tag and literal textContent.
 * Return the node; no HTML parsing or attachment occurs. Callers supply the
 * trusted local tag and display text.
 *
 * @param {string} tag @param {string} text
 */
function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

/**
 * Replace container with the selected metric table and update description.
 *
 * topicName/view select rows from summary; the catalog controls row order and
 * section placement. Format valid values with metricValueText. Preserve exact
 * values in data attributes. Invalid measurements show an em dash and
 * missing-value help, never zero. Register tooltips on new measure labels.
 * Return undefined; mutate only DOM. The caller supplies a valid
 * summary/topic and owns fetching/exporting data.
 *
 * Ledger look: options defaults to {}. With options.agents (the catalog's
 * agent list) each subject cell shows chips first (team crest or class emblem
 * and "Agent 3"; a directed row shows two chips joined by "→", an ally pair
 * by "↔") and keeps the exact host subject text in a span.ledger-subject__text
 * that screen readers read; the CSS hides that text visually only while chips
 * are shown. Without options.agents every subject is today's plain host text.
 * options.subjectChip replaces the built-in chip (see
 * createMetricSubjectChips). Section rows keep their exact titles from
 * METRIC_SECTION_TITLES. No option changes which rows, labels, units,
 * definitions, values or CSV names appear. The table is the full researcher
 * record in every view; Agent POV never filters it.
 *
 * @param {HTMLElement} container
 * @param {string} topicName
 * @param {string} view
 * @param {Record<string, any>} summary
 * @param {HTMLElement} description
 * @param {{agents?: ReadonlyArray<Record<string, any>>,
 *   subjectChip?: SubjectChipFactory}} [options]
 */
export function renderMetricRows(
  container,
  topicName,
  view,
  summary,
  description,
  options = {},
) {
  const doc = container.ownerDocument;
  const rows = selectedMetricRows(summary, topicName, view);
  const available = rows.filter(
    (/** @type {Record<string, any>} */ row) => row.valid,
  ).length;
  const topic = summary.topics.find(
    (/** @type {Record<string, string>} */ item) => item.name === topicName,
  );
  description.textContent = `${topic.description} ${available} of ${rows.length} measurements available.`;
  const table = doc.createElement("table");
  table.className = "metric-table";
  const head = doc.createElement("thead");
  const headings = doc.createElement("tr");
  for (const label of ["Subject", "Measure", "Value"]) {
    const heading = element("th", label);
    heading.setAttribute("scope", "col");
    headings.append(heading);
  }
  head.append(headings);
  table.append(head);
  const body = doc.createElement("tbody");
  body.tabIndex = 0;
  body.setAttribute("aria-label", "Scrollable metric values");
  let previousSection = -1;
  for (const row of rows) {
    const section = metricSection(row);
    if (section !== previousSection) {
      const headingRow = doc.createElement("tr");
      headingRow.className = "metric-section";
      const heading = element("th", METRIC_SECTION_TITLES[section]);
      heading.setAttribute("colspan", "3");
      headingRow.append(heading);
      body.append(headingRow);
      previousSection = section;
    }
    const tr = doc.createElement("tr");
    tr.dataset.metric = row.name;
    const chips = !options.agents
      ? null
      : createMetricSubjectChips(doc, metricSubjectParts(row, options.agents), {
          join: row.scope === "ally_pair" ? "↔" : "→",
          withClass: true,
          subjectChip: options.subjectChip,
        });
    const subject = element("td", chips ? "" : row.subject);
    if (chips) {
      subject.className = "ledger-subject";
      const text = element("span", row.subject);
      text.className = "ledger-subject__text";
      subject.append(chips, text);
    }
    const measure = element("td", "");
    const label = element("span", row.label);
    label.tabIndex = 0;
    label.className = "metric-measure";
    const subtitle = row.subtitle ?? row.status;
    registerTooltipOwner(label, {
      kind: "metric",
      tone: "neutral",
      accent: "none",
      id: `metric:${row.name}`,
      title: row.label,
      summary: subtitle ? `${subtitle}. ${row.description}` : row.description,
      rows: metricTooltipRows(row),
      sections: [],
      metadata: { compact: true, full: true },
      anchor: "element",
    });
    measure.append(label);
    if (subtitle) {
      const status = element("div", subtitle);
      status.className = "metric-condition";
      measure.append(status);
    }
    measure.append(
      element("small", row.unit === "health" ? "HP" : metricLabel(row.unit)),
    );
    const value = element("td", metricValueText(row));
    value.className = "metric-value";
    value.dataset.value = row.valid ? String(row.value) : "";
    if (!row.valid)
      value.setAttribute(
        "aria-label",
        `Unavailable. Possible reasons: ${row.missing_when}`,
      );
    tr.append(subject, measure, value);
    body.append(tr);
  }
  if (rows.length === 0) {
    const empty = doc.createElement("tr");
    const message = element("td", "No applicable measurements for this roster.");
    message.setAttribute("colspan", "3");
    empty.append(message);
    body.append(empty);
  }
  table.append(body);
  container.replaceChildren(table);
}
