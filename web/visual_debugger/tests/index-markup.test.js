/**
 * @file Check unique public page IDs, supported controls and required keyboard/help text.
 * Help key lists are empty `data-command-help` lists that shell.js fills from
 * the one command set (replay arrows step ticks; live play has no arrows and
 * no B; N has a job in both products). The Visual Filters preset row is a
 * fieldset that starts with Default Configuration pressed and Reset to
 * Default Configuration grayed. The Replay Viewer speed list selects 1.00× by
 * default. The top bar holds the brand lockup (emblem, an empty
 * #brand-wordmark SVG, divider, #app-title), the place title, the View switch
 * (Agent POV first) driving a hidden #view-select, the POV agent chip and
 * picker, the status pills (the Recording pill opens the #rec-pop popover with
 * the save buttons), a Details popover with Step, Transition, Copy and the
 * replay badges, and the toolbar actions; the dock (#replay-timeline) sits under the
 * arena with the transport and export actions; the amber Oracle View tag sits
 * on the field. There is no Visual Key (owner, 4 Oct 2026: no button, no
 * sidebar key) and the Replay Viewer's #champion-frame slot sits before
 * #team-rosters (the Arena Frames). The #roster-details card is marked for
 * no product: the live DevClient shows it in its Developer drawer and the
 * Replay Viewer moves it into its rail's Roster tab. Exit reads "Exit" in
 * every client. The live deck keeps its controls in
 * #command-deck-body with the vision's Lock In Turn button and Prev and Next
 * buttons and its header line #status-line (the message and Keys On or
 * Off); Restart and Finish and Save sit in the top bar's toolbar actions. Retired copy stays out of main.js and the page parts it
 * starts (shell.js, tactics-turn.js, filter-controls.js); shell.js owns the
 * audience labels.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { commandHelpEntries } from "../src/shell.js";

const indexUrl = new URL("../index.html", import.meta.url);
const stylesUrl = new URL("../styles.css", import.meta.url);
const productionSourceUrls = [
  new URL("../src/controls.js", import.meta.url),
  new URL("../src/main.js", import.meta.url),
  new URL("../src/scene.js", import.meta.url),
  new URL("../src/explanations.js", import.meta.url),
  new URL("../src/panels.js", import.meta.url),
  new URL("../src/shell.js", import.meta.url),
  new URL("../src/tactics-turn.js", import.meta.url),
  new URL("../src/filter-controls.js", import.meta.url),
];

/** @param {string} markup
 * @param {string} id
 * @param {string} tagName */
function elementBody(markup, id, tagName) {
  const opening = new RegExp(`<${tagName}\\b[^>]*\\bid="${id}"[^>]*>`, "u").exec(
    markup,
  );
  assert.ok(opening, `#${id} must be a <${tagName}> element.`);
  const contentStart = opening.index + opening[0].length;
  const tags = new RegExp(`<\\/?${tagName}\\b[^>]*>`, "gu");
  tags.lastIndex = contentStart;
  let depth = 1;
  for (let match = tags.exec(markup); match !== null; match = tags.exec(markup)) {
    depth += match[0].startsWith("</") ? -1 : 1;
    if (depth === 0) return markup.slice(contentStart, match.index);
  }
  throw new Error(`#${id} has no matching closing tag.`);
}

/** @param {string} markup
 * @param {string} panelId */
function disclosureBody(markup, panelId) {
  const details = elementBody(markup, panelId, "details");
  const bodyId = `${panelId}-body`;
  const directWrappers = [
    ...details.matchAll(
      new RegExp(
        `</summary>\\s*<div\\b(?=[^>]*\\bid="${bodyId}")(?=[^>]*\\bdata-disclosure-body="${panelId}")[^>]*>`,
        "gu",
      ),
    ),
  ];
  assert.match(details, /^\s*<summary\b/u, `#${panelId} summary must remain direct.`);
  assert.equal(
    directWrappers.length,
    1,
    `#${panelId} must have exactly one direct stable disclosure body.`,
  );

  const wrapper = directWrappers[0];
  const wrapperStart = wrapper.index + wrapper[0].lastIndexOf("<div");
  const divTag = /<\/?div\b[^>]*>/giu;
  divTag.lastIndex = wrapperStart;
  const opening = divTag.exec(details);
  assert.ok(opening, `#${bodyId} must be a div.`);

  let depth = 1;
  for (let tag = divTag.exec(details); tag; tag = divTag.exec(details)) {
    depth += tag[0].startsWith("</") ? -1 : 1;
    if (depth === 0) {
      return details.slice(opening.index + opening[0].length, tag.index);
    }
  }
  assert.fail(`#${bodyId} must close.`);
}

test("static debugger IDs are unique", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const ids = [...markup.matchAll(/\bid="([^"]+)"/gu)].map((match) => match[1]);

  assert.equal(new Set(ids).size, ids.length);
});

test("Help key lists come from the one command set, with the replay arrows and Escape", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const help = elementBody(markup, "help-dialog", "dialog");
  assert.equal(
    [
      ...help.matchAll(
        /<dl class="help-grid" data-command-help="replay_viewer"><\/dl>/gu,
      ),
    ].length,
    1,
  );
  assert.equal(
    [
      ...help.matchAll(
        /<dl class="help-grid" data-command-help="combat_debugger"><\/dl>/gu,
      ),
    ].length,
    2,
  );
  assert.doesNotMatch(
    help,
    /Arrow keys|Cardinal movement|LEFT ARROW|<dt>Left \/ Right<\/dt>/u,
  );
  assert.doesNotMatch(help, /<dt>(?:Escape|Space|Tab \/ Shift\+Tab|G|R|X)<\/dt>/u);
  const replay = commandHelpEntries("replay_viewer");
  assert.deepEqual(
    replay.find(([keycaps]) => keycaps === "Arrow Left / Right"),
    ["Arrow Left / Right", "Back / forward one tick (glides)"],
  );
  assert.deepEqual(
    replay.find(([keycaps]) => keycaps === "Shift + Arrow Left / Right"),
    ["Shift + Arrow Left / Right", "Back / forward ten ticks (snaps)"],
  );
  assert.deepEqual(
    replay.find(([keycaps]) => keycaps === "Escape"),
    ["Escape", "Close, clear selection, leave the field"],
  );
  assert.deepEqual(
    replay.find(([keycaps]) => keycaps === "B"),
    ["B", "Battle Cam"],
  );
  const live = commandHelpEntries("combat_debugger");
  assert.equal(
    live.some(([keycaps]) => /Arrow|^B$/u.test(keycaps)),
    false,
  );
  assert.deepEqual(
    live.find(([keycaps]) => keycaps === "W A S D Q E Z C"),
    ["W A S D Q E Z C", "Step (staged)"],
  );
  assert.deepEqual(
    live.find(([keycaps]) => keycaps === "V"),
    ["V", "Everyone in View"],
  );
});

test("every native disclosure has one named phrasing-content summary", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const detailsCount = [...markup.matchAll(/<details\b/gu)].length;
  const summaries = [...markup.matchAll(/<summary\b[^>]*>([\s\S]*?)<\/summary>/gu)];

  assert.equal(summaries.length, detailsCount);
  for (const [, content] of summaries) {
    assert.doesNotMatch(
      content,
      /<(?:address|article|aside|div|dl|fieldset|footer|form|h[1-6]|header|main|nav|ol|p|section|table|ul)\b/iu,
    );
    assert.notEqual(content.replace(/<[^>]+>/gu, "").trim(), "");
  }
});

test("every CP3 disclosure keeps its scientific content in one stable direct body", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const panels = {
    "command-deck": [
      "command-controlled-actor",
      "stay-button",
      "command-target-select",
      "no-combat-button",
      "basic-button",
      "ultimate-button",
      "command-commit-title",
      "command-commit-summary",
      "submit-turn-button",
    ],
    "roster-details": ["roster"],
    "agent-details": ["selection-card"],
    "pending-turn-details": ["pending-scope", "pending-card"],
    "latest-transition-details": ["accepted-card", "accepted-announcement"],
    "technical-frame-details": ["diagnostics-card"],
  };

  for (const [panelId, contentIds] of Object.entries(panels)) {
    const body = disclosureBody(markup, panelId);
    for (const contentId of contentIds) {
      assert.match(
        body,
        new RegExp(`\\bid="${contentId}"`, "u"),
        `#${contentId} must stay inside #${panelId}-body.`,
      );
      assert.equal(
        [...markup.matchAll(new RegExp(`\\bid="${contentId}"`, "gu"))].length,
        1,
        `#${contentId} must remain unique.`,
      );
    }
  }
  // Restart sits in the top bar beside Exit (owner, 4 Oct 2026); the deck's
  // header line (the message and Keys On or Off) is the deck body's first row.
  const toolbar = /<div class="toolbar-actions">([\s\S]*?)\n {6}<\/div>/u.exec(markup);
  assert.ok(toolbar);
  assert.match(toolbar[1], /\bid="reset-button"/u);
  assert.equal([...markup.matchAll(/\bid="reset-button"/gu)].length, 1);
  assert.match(
    disclosureBody(markup, "command-deck"),
    /^\s*(?:<!--[\s\S]*?-->\s*)?<div class="dstatus" id="status-line"/u,
  );
});

test("shared shell uses the requested details and visual-filter controls without an event feed", async () => {
  const markup = await readFile(indexUrl, "utf8");

  assert.match(markup, />Comprehensive Agent Class Details</u);
  assert.match(markup, /id="accepted-heading">Latest Transition</u);
  assert.match(markup, /id="command-commit-title">Submit the staged joint turn</u);
  assert.match(
    markup,
    /id="submit-turn-button"[^>]*><b id="lock-label">Lock In Turn<\/b><span>Space or Enter<\/span><\/button>/u,
  );
  assert.match(
    markup,
    /<button type="button" class="sbtn" id="prev-actor-button" data-key="Tab" data-shift="true">‹ Prev<\/button><button type="button" class="sbtn" id="next-actor-button" data-key="Tab">Next › <kbd>Tab<\/kbd><\/button>/u,
  );
  // No Visual Key in any client (owner, 4 Oct 2026): no button, no sidebar key.
  assert.doesNotMatch(markup, /<summary>Visual Key<\/summary>/u);
  assert.doesNotMatch(markup, /id="(?:live|replay)-visual-key"/u);
  assert.doesNotMatch(markup, /id="visual-key-button"/u);
  // The Replay Viewer's Champion Frame slot sits before the Team Rosters strip.
  assert.match(
    markup,
    /<section\s+id="champion-frame"[^>]*data-replay-only[^>]*hidden[^>]*>[\s\S]*id="team-rosters"/u,
  );
  // The Roster card serves both: the DevClient's Developer drawer and the
  // Replay Viewer's Roster tab (src/ledger.js moves it there).
  assert.match(markup, /<details\s+id="roster-details"[^>]*>/u);
  assert.doesNotMatch(
    /<details\s+id="roster-details"[^>]*>/u.exec(markup)?.[0] ?? "",
    /data-(?:live|replay)-only/u,
  );
  // The count is written by filter-controls.js for the client's own switches
  // (they differ per client); the page starts at the empty count.
  assert.match(markup, /id="visual-filter-count"[^>]*>0 enabled</u);
  const presets = elementBody(markup, "visual-filter-presets", "fieldset");
  assert.match(
    presets,
    /^\s*<button id="default-visual-filters-button" type="button" aria-pressed="true">Default Configuration<\/button>\s*$/u,
  );
  assert.match(
    markup,
    /<button id="reset-visual-filters-button" type="button" disabled>Reset to Default Configuration<\/button>/u,
  );
  assert.match(
    markup,
    /id="enable-all-visual-filters-button"[^>]*>\s*Enable All\s*<\/button>/u,
  );
  assert.match(
    markup,
    /id="disable-all-visual-filters-button"[^>]*>\s*Disable All\s*<\/button>/u,
  );
  assert.doesNotMatch(markup, /Restore All|Latest events/u);
  assert.doesNotMatch(markup, /<dt>Right click<\/dt>/u);
  for (const removedId of ["events-details", "event-count", "event-feed"]) {
    assert.doesNotMatch(markup, new RegExp(`id="${removedId}"`, "u"));
  }
});

test("current, upcoming, and latest-transition surfaces remain available in both products", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const pendingOpeningTag = markup.match(
    /<details\b[^>]*\bid="pending-turn-details"[^>]*>/u,
  );
  assert.ok(pendingOpeningTag);
  assert.doesNotMatch(pendingOpeningTag[0], /\bdata-live-only\b/u);
  const latestTransitionOpeningTag = markup.match(
    /<details\b[^>]*\bid="latest-transition-details"[^>]*>/u,
  );
  assert.ok(latestTransitionOpeningTag);
  assert.doesNotMatch(latestTransitionOpeningTag[0], /\bdata-live-only\b/u);
  assert.doesNotMatch(latestTransitionOpeningTag[0], /\bdata-replay-only\b/u);
});

test("product shell loads strict identity before the module and omits retired controls", async () => {
  const markup = await readFile(indexUrl, "utf8");

  const bootstrapIndex = markup.indexOf('<script src="/bootstrap.js"></script>');
  const moduleIndex = markup.indexOf(
    '<script type="module" src="/src/main.js"></script>',
  );
  assert.notEqual(bootstrapIndex, -1);
  assert.ok(bootstrapIndex < moduleIndex);
  assert.match(markup, /<title>MARL-BattleGrounds<\/title>/u);
  assert.match(markup, /<h1 id="app-title" class="bprod">MARL-BattleGrounds<\/h1>/u);
  assert.match(markup, /<option value="researcher">Oracle View<\/option>/u);

  for (const retiredId of [
    "scenario-select",
    "preset-select",
    "motion-pause-button",
    "motion-off-button",
    "motion-skip-button",
    "motion-status",
    "advance-script-button",
    "revision-value",
    "replay-incoming-value",
  ]) {
    assert.doesNotMatch(markup, new RegExp(`id="${retiredId}"`, "u"));
  }
  assert.doesNotMatch(markup, /class="replay-shortcuts"/u);
  assert.doesNotMatch(markup, /command-deck__shortcut/u);
  // Help lists start empty: shell.js writes every key, N included, from the
  // command set. P and [ / ] still have no job anywhere.
  const helpLists = [
    ...markup.matchAll(/<dl\b[^>]*\bdata-command-help="[^"]+"[^>]*>([\s\S]*?)<\/dl>/gu),
  ];
  assert.equal(helpLists.length, 3);
  for (const [, body] of helpLists) assert.equal(body.trim(), "");
  assert.doesNotMatch(markup, /<dt>(?:P|\[ \/ \])<\/dt>/u);
  for (const product of /** @type {const} */ (["combat_debugger", "replay_viewer"])) {
    const keys = commandHelpEntries(product).map(([keycaps]) => keycaps);
    assert.ok(keys.includes("N"), `${product} help must list N.`);
    assert.ok(!keys.includes("P") && !keys.includes("[ / ]"), product);
  }
  assert.doesNotMatch(markup, /Previous scenario|Next scenario|Motion Off/u);
  for (const id of ["replay-completion-badge", "replay-processing-badge"]) {
    assert.match(
      markup,
      new RegExp(`<dd\\b(?=[^>]*\\bid="${id}")(?=[^>]*\\btabindex="0")[^>]*>`, "u"),
    );
  }
  assert.doesNotMatch(markup, /<dt>Incoming<\/dt>|>Revision\s*</u);
});

test("replay transport exposes the exact CP8 controls, rates, and truthful help", async () => {
  const [markup, styles] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(stylesUrl, "utf8"),
  ]);
  assert.match(
    markup,
    /id="replay-first-button"[^>]*aria-label="Start replay tick"[^>]*>Start<\/button>/u,
  );
  assert.match(
    markup,
    /id="replay-last-button"[^>]*aria-label="End replay tick"[^>]*>End<\/button>/u,
  );
  const rateOptions = elementBody(markup, "replay-playback-rate", "select");
  // The Replay Viewer plays at 1x by default: with the beat clock a quiet tick
  // already lasts 1.7 s at 1x.
  assert.match(rateOptions, /<option value="1" selected>1\.00×<\/option>/u);
  assert.equal([...rateOptions.matchAll(/ selected>/gu)].length, 1);
  assert.deepEqual(
    [
      ...rateOptions.matchAll(
        /<option value="([^"]+)"(?: selected)?>([^<]+)<\/option>/gu,
      ),
    ].map(([, value, label]) => [value, label]),
    [
      ["0.25", "0.25×"],
      ["0.5", "0.50×"],
      ["0.75", "0.75×"],
      ["1", "1.00×"],
      ["1.25", "1.25×"],
      ["1.5", "1.50×"],
      ["1.75", "1.75×"],
      ["2", "2.00×"],
    ],
  );
  assert.match(markup, /id="replay-transport-status"[^>]*aria-live="polite"/u);
  assert.match(
    markup,
    /Frame slider<\/dt><dd>Drag to pick a tick; the replay jumps there when you let go<\/dd>/u,
  );
  assert.match(
    markup,
    /Playback speed<\/dt><dd>Change how fast the replay plays<\/dd>/u,
  );
  assert.doesNotMatch(markup, /Home \/ End|Shift\+Left|Shift\+Right|short debounce/u);
  assert.match(
    styles,
    /@media \(max-width: 70rem\)[\s\S]*\.replay-timeline__transport\s*\{\s*grid-template-columns: repeat\(7, minmax\(0, 1fr\)\);/u,
  );
});

test("replay artifact actions are accessible, fail closed, and CSP compatible", async () => {
  const [markup, styles] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(stylesUrl, "utf8"),
  ]);
  const actions = elementBody(markup, "replay-artifact-actions", "fieldset");
  const opening = markup.match(
    /<fieldset\b(?=[^>]*\bid="replay-artifact-actions")(?=[^>]*\bdata-replay-only)(?=[^>]*\bhidden)[^>]*>/u,
  );
  assert.ok(opening);
  for (const [id, label, helpId] of [
    ["replay-export-png-button", "Export PNG", "replay-export-png-help"],
    [
      "replay-download-metrics-button",
      "Download Metrics CSV",
      "replay-download-metrics-help",
    ],
  ]) {
    assert.match(
      actions,
      new RegExp(
        `<button\\b(?=[^>]*\\bid="${id}")(?=[^>]*\\bdisabled)(?=[^>]*\\baria-describedby="${helpId}")[^>]*>${label}</button>`,
        "u",
      ),
    );
    assert.equal([...markup.matchAll(new RegExp(`\\bid="${id}"`, "gu"))].length, 1);
    assert.match(actions, new RegExp(`\\bid="${helpId}"`, "u"));
  }
  assert.match(
    markup,
    /Content-Security-Policy[\s\S]*img-src 'self' data:;[\s\S]*font-src 'self'/u,
  );
  assert.doesNotMatch(markup, /img-src[^;]*blob:/u);
  assert.match(
    styles,
    /\.replay-artifact-actions\s*\{[\s\S]*justify-content: flex-end;[\s\S]*border: 0;/u,
  );
  assert.match(
    styles,
    /@media \(max-width: 42rem\)[\s\S]*\.replay-artifact-actions\s*\{\s*justify-content: flex-start;/u,
  );
});

test("browser production paths omit retired navigation and privileged display copy", async () => {
  const [
    controls,
    main,
    scene,
    explanations,
    panels,
    shell,
    tacticsTurn,
    filterControls,
  ] = await Promise.all(productionSourceUrls.map((url) => readFile(url, "utf8")));
  const pageStart = [main, shell, tacticsTurn, filterControls];

  for (const source of pageStart) {
    assert.doesNotMatch(source, /liveDebuggerFrameIsScripted|REMOVED_LIVE_KEYS/u);
    assert.doesNotMatch(source, /onPresentationKey|Manual submit unavailable/u);
    assert.doesNotMatch(source, /SharedObs source material/u);
    assert.doesNotMatch(
      source,
      /fixed recipient|passive inspection|controlled_actor/iu,
    );
  }
  assert.match(
    shell,
    /publicAudienceLabel\(authorizedPresentationAudience\(presentation\)\)/u,
  );
  assert.match(shell, /return "Oracle View"/u);
  assert.match(shell, /return "Agent POV"/u);
  assert.doesNotMatch(panels, /fixed recipient|controlled_actor/iu);
  assert.doesNotMatch(scene, /scene\.audience_badge/u);
  assert.match(scene, /"Oracle View" : "Agent POV"/u);

  for (const source of [controls, scene, explanations, panels, ...pageStart]) {
    assert.doesNotMatch(source, /Privileged researcher|PLAYBACK \/ INSPECTION ONLY/u);
  }
});

test("battlefield support owns visible instructions, product controls, and minimal keys", async () => {
  const [markup, styles] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(stylesUrl, "utf8"),
  ]);
  const battlefieldShellStart = markup.indexOf('id="battlefield-shell"');
  const supportStart = markup.indexOf('<div class="battlefield-support">');
  const commandDeckStart = markup.indexOf('<details id="command-deck"');
  assert.notEqual(battlefieldShellStart, -1);
  assert.ok(battlefieldShellStart < supportStart);
  assert.ok(supportStart < commandDeckStart);

  assert.match(
    markup,
    /id="battlefield"[\s\S]*?aria-describedby="battlefield-instructions"/u,
  );
  assert.match(
    markup,
    /id="battlefield"[\s\S]*?role="img"[\s\S]*?tabindex="-1"[\s\S]*?aria-label="Battlefield unavailable until product identity is validated\."/u,
  );
  assert.doesNotMatch(
    markup,
    /id="battlefield"[\s\S]*?aria-label="Interactive battlefield/u,
  );
  assert.match(
    markup,
    /id="battlefield-instructions" class="battlefield-instructions"/u,
  );
  assert.match(markup, /Product-specific Battlefield instructions are loading\./u);
  assert.doesNotMatch(markup, /id="battlefield-instructions"[^>]*class="[^"]*sr-only/u);

  const utilities = elementBody(markup, "battlefield-utilities", "fieldset");
  const visualFilters = elementBody(markup, "visual-filters", "details");
  for (const id of ["live-ranges-button", "replay-ranges-button"]) {
    assert.doesNotMatch(utilities, new RegExp(`id="${id}"`, "u"));
    assert.match(visualFilters, new RegExp(`id="${id}"`, "u"));
    assert.equal([...markup.matchAll(new RegExp(`id="${id}"`, "gu"))].length, 1);
  }
  assert.match(utilities, /id="replay-clear-reference-button"/u);
  assert.equal([...markup.matchAll(/id="replay-clear-reference-button"/gu)].length, 1);
  assert.match(utilities, />Clear Selection<\/button>/u);
  assert.doesNotMatch(markup, /Clear Reference/u);

  assert.match(
    styles,
    /html:not\(\[data-product-kind\]\) \[data-live-only\],[\s\S]*html:not\(\[data-product-kind\]\) \[data-replay-only\],[\s\S]*html:not\(\[data-product-kind\]\) #battlefield-instructions,[\s\S]*html\[data-product-kind="combat_debugger"\] \[data-replay-only\],[\s\S]*html\[data-product-kind="replay_viewer"\] \[data-live-only\]/u,
  );
});

test("the top bar, popovers, dock and Oracle View tag follow the V5 shell", async () => {
  const [markup, shell] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(new URL("../src/shell.js", import.meta.url), "utf8"),
  ]);
  const header = elementBody(markup, "app-title", "h1");
  assert.equal(header.trim(), "MARL-BattleGrounds");
  assert.match(
    markup,
    /<svg id="brand-wordmark" class="brand__wordmark" role="img" aria-label="MARL-BGs"><\/svg>/u,
  );
  assert.match(
    markup,
    /<link rel="icon" type="image\/png"[^>]*marl-bgs-emblem-64\.png">/u,
  );
  assert.match(markup, /<span class="brand__divider" aria-hidden="true"><\/span>/u);

  const viewSwitch = elementBody(markup, "view-switch", "fieldset");
  assert.deepEqual(
    [...viewSwitch.matchAll(/data-view-mode="([^"]+)"[^>]*>([^<]+)</gu)].map(
      (match) => [match[1], match[2]],
    ),
    [
      ["pov", "Agent POV"],
      ["researcher", "Oracle View"],
    ],
  );
  const select = /<select\b[^>]*\bid="view-select"[^>]*>/u.exec(markup)?.[0] ?? "";
  assert.match(select, /tabindex="-1"/u);
  assert.match(select, /aria-hidden="true"/u);

  assert.match(
    markup,
    /<button\s+id="recording-badge"\s+class="status-pill status-pill--recording pill"\s+type="button"\s+popovertarget="rec-pop"\s+aria-haspopup="dialog"\s+hidden\s*><i><\/i><span id="rec-pill-text">Recording<\/span><\/button>/u,
  );
  assert.match(markup, /id="recording-panel"[^>]*\bpopover\b[^>]*data-live-only/u);
  const recPop = elementBody(markup, "rec-pop", "div");
  for (const id of [
    "pop-finish",
    "recording-review-button",
    "recording-retry-button",
    "recording-save-as-control",
  ]) {
    assert.match(recPop, new RegExp(`id="${id}"`, "u"), id);
  }
  assert.match(markup, /<div class="pop" id="rec-pop" popover\b/u);
  const details = elementBody(markup, "session-details", "div");
  for (const id of [
    "step-value",
    "transition-value",
    "copy-transition-button",
    "replay-artifact-reference",
    "replay-completion-badge",
    "replay-processing-badge",
    "replay-end-reason",
  ]) {
    assert.match(details, new RegExp(`id="${id}"`, "u"), id);
  }
  assert.match(
    markup,
    /id="session-details-button"[^>]*popovertarget="session-details"/u,
  );
  assert.match(
    markup,
    /id="pov-agent-chip-button"[^>]*popovertarget="pov-agent-picker"/u,
  );
  assert.match(markup, /<div id="pov-agent-chip" class="pov-agent-chip" hidden>/u);

  const supportStart = markup.indexOf('<div class="battlefield-support">');
  const dockStart = markup.indexOf('id="replay-timeline"');
  const rostersStart = markup.indexOf('id="team-rosters"');
  assert.ok(supportStart < dockStart && dockStart < rostersStart);
  const dock = elementBody(markup, "replay-timeline", "section");
  assert.match(dock, /id="replay-artifact-actions"/u);
  assert.match(dock, /class="replay-timeline__transport"/u);
  assert.doesNotMatch(dock, /replay-badges|Read-only artifact/u);

  const shellBody = elementBody(markup, "battlefield-shell", "div");
  assert.match(
    shellBody,
    /<p id="oracle-view-tag" class="oracle-view-tag" hidden>[\s\S]*Oracle View <span>· You See Everything<\/span>/u,
  );
  assert.doesNotMatch(
    /<button id="exit-button"[^>]*>/u.exec(markup)?.[0] ?? "",
    /button--danger/u,
  );
  assert.doesNotMatch(markup, /\b(?:Sound|Mute|Volume)\b/u);
  assert.doesNotMatch(shell, /Exit combat debugger|Exit replay viewer/u);
  assert.match(shell, /elements\.exitButton\.textContent = "Exit";/u);
});
