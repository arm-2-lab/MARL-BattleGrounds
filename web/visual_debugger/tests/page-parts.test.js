/**
 * @file Check the page parts that main.js starts and the page slots they will fill.
 * shell.js, tactics-turn.js and filter-controls.js load in Node with no page.
 * The deck's key helpers sort Step, action, submit, Tab and Escape keys; the
 * shell names each product and audience ("View Unavailable" for anything else)
 * and writes the product header (tab title, product name, How To Play or
 * Replay Viewer Help); the
 * Visual Filters help rows keep their page order (the five presets with
 * Default Configuration first, Reset To Default Configuration, Enable All,
 * Disable All, then each switch; a switch renamed for live play gets one row
 * per client, each with the label that client shows), and every preset and
 * Reset tip says the
 * view, Ranges and motion stay as they are. index.html keeps the five
 * exact strings private tools insert before, the IDs and classes those tools
 * hide or retitle, and one hidden, empty slot per planned interface piece, none
 * marked for a product (the shell would un-hide it). The side rail (#rail)
 * starts with the live Your Turn, Class Guide and How To Play cards, then the
 * This Turn and How To Play slots, the Fallen panel and the Replay Viewer
 * rail. The toolbar holds Visual Filters (live play only) after Visual Key,
 * and the button opens and closes the #visual-filters-drawer popover.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  VISUAL_FILTER_CONTROL_HELP,
  VISUAL_FILTER_OPTION_HELP,
} from "../src/filter-controls.js";
import { LIVE_SWITCH_NAMES } from "../src/visual-filters.js";
import {
  applyProductHeader,
  PRODUCT_TITLES,
  publicAudienceLabel,
} from "../src/shell.js";
import {
  commandPreparesDeferredSubmit,
  commandRole,
  DRAFT_KEYS,
  isDeferredDraftKey,
  isFreshDeferredDraft,
} from "../src/tactics-turn.js";

const indexUrl = new URL("../index.html", import.meta.url);
const mainUrl = new URL("../src/main.js", import.meta.url);

/** @param {string} key
 * @param {Record<string, unknown>} [flags] */
function keyCommand(key, flags = {}) {
  return {
    command_type: "keyboard",
    key,
    shift_key: false,
    ctrl_key: false,
    alt_key: false,
    meta_key: false,
    repeat: false,
    ...flags,
  };
}

/** @param {string} markup
 * @param {string} id */
function openingTag(markup, id) {
  const tags = [
    ...markup.matchAll(new RegExp(`<[a-z][a-z0-9]*\\b[^>]*\\bid="${id}"[^>]*>`, "gu")),
  ];
  assert.equal(tags.length, 1, `#${id} must appear exactly once.`);
  return tags[0];
}

test("page parts load in Node without a page", async () => {
  for (const path of [
    "../src/shell.js",
    "../src/tactics-turn.js",
    "../src/filter-controls.js",
  ]) {
    const module = await import(path);
    assert.equal(typeof module, "object", path);
  }
});

test("deck key helpers sort Step, action, submit, Tab and Escape keys", () => {
  for (const key of ["w", "a", "s", "d", "q", "e", "z", "c", "x", "0", "1", "2"]) {
    assert.ok(DRAFT_KEYS.has(key), key);
    assert.equal(commandRole(keyCommand(key)), "draft", key);
    assert.equal(commandRole(keyCommand(key.toUpperCase())), "draft", key);
    assert.equal(commandPreparesDeferredSubmit(keyCommand(key)), true, key);
  }
  for (const key of ["Enter", "Return", " ", "Spacebar"]) {
    assert.equal(commandRole(keyCommand(key)), "interactive-submit", key);
    assert.equal(commandPreparesDeferredSubmit(keyCommand(key)), false, key);
  }
  for (const key of ["Tab", "Escape", "g", "r", "?"]) {
    assert.equal(commandRole(keyCommand(key)), null, key);
  }
  assert.equal(commandRole({ command_type: "reset" }), null);
  assert.equal(commandPreparesDeferredSubmit(keyCommand("Tab")), true);
  assert.equal(commandPreparesDeferredSubmit(keyCommand("Escape")), true);
  assert.equal(commandPreparesDeferredSubmit(keyCommand("g")), false);
  for (const commandType of ["battlefield_pointer", "roster_selection"]) {
    assert.equal(commandPreparesDeferredSubmit({ command_type: commandType }), true);
    assert.equal(isDeferredDraftKey({ command_type: commandType }), false);
  }
  assert.equal(isDeferredDraftKey(keyCommand("w")), true);
  assert.equal(isFreshDeferredDraft(keyCommand("w")), true);
  assert.equal(isFreshDeferredDraft(keyCommand("Tab", { shift_key: true })), true);
  assert.equal(isFreshDeferredDraft(keyCommand("w", { shift_key: true })), false);
  assert.equal(isFreshDeferredDraft(keyCommand("w", { repeat: true })), false);
  assert.equal(isFreshDeferredDraft(keyCommand("w", { ctrl_key: true })), false);
  assert.equal(isFreshDeferredDraft({ command_type: "keyboard", key: "w" }), false);
  assert.equal(isFreshDeferredDraft(keyCommand("Enter")), false);
});

test("shell names each product and audience and writes the product header", () => {
  assert.deepEqual(
    { ...PRODUCT_TITLES },
    {
      combat_debugger: "MARL-BattleGrounds BattleClient",
      replay_viewer: "MARL-BattleGrounds Replay Viewer",
    },
  );
  assert.ok(Object.isFrozen(PRODUCT_TITLES));
  assert.equal(publicAudienceLabel("researcher"), "Oracle View");
  for (const value of ["agent_pov", "actor_pov", "pov"]) {
    assert.equal(publicAudienceLabel(value), "Agent POV");
  }
  for (const value of ["unavailable", null, undefined, "Researcher"]) {
    assert.equal(publicAudienceLabel(value), "View Unavailable");
  }

  const previousDocument = Reflect.get(globalThis, "document");
  const fakeDocument = { title: "MARL-BattleGrounds" };
  Reflect.set(globalThis, "document", fakeDocument);
  try {
    // The tab title keeps the official name; the header shows the product name
    // after the MARL-BGs lockup; Help is How To Play in the BattleClient.
    for (const [productKind, title, name, help] of /** @type {const} */ ([
      [
        "combat_debugger",
        "MARL-BattleGrounds BattleClient",
        "BattleClient",
        "How To Play",
      ],
      [
        "replay_viewer",
        "MARL-BattleGrounds Replay Viewer",
        "Replay Viewer",
        "Replay Viewer Help",
      ],
    ])) {
      const appTitle = { textContent: "" };
      const helpHeading = { textContent: "" };
      applyProductHeader(/** @type {any} */ ({ appTitle, helpHeading }), productKind);
      assert.equal(fakeDocument.title, title);
      assert.equal(appTitle.textContent, name);
      assert.equal(helpHeading.textContent, help);
    }
  } finally {
    Reflect.set(globalThis, "document", previousDocument);
  }
});

test("Visual Filters help rows keep their page order", () => {
  assert.ok(Object.isFrozen(VISUAL_FILTER_CONTROL_HELP));
  assert.deepEqual(
    VISUAL_FILTER_CONTROL_HELP.slice(0, 9).map(([selector, title]) => [
      selector,
      title,
    ]),
    [
      ["#visual-filters > summary", "Visual Filters"],
      ["#default-visual-filters-button", "Default Configuration"],
      ["[data-visual-filter-preset='broadcast']", "Broadcast"],
      ["[data-visual-filter-preset='researcher']", "Researcher"],
      ["[data-visual-filter-preset='manuscript']", "Manuscript"],
      ["[data-visual-filter-preset='minimal']", "Minimal"],
      ["#reset-visual-filters-button", "Reset To Default Configuration"],
      ["#enable-all-visual-filters-button", "Enable All"],
      ["#disable-all-visual-filters-button", "Disable All"],
    ],
  );
  for (const [, , summary] of VISUAL_FILTER_CONTROL_HELP.slice(1, 7)) {
    assert.match(summary, /The view, Ranges and motion stay as they are\.$/u);
  }
  assert.deepEqual(
    VISUAL_FILTER_CONTROL_HELP.slice(9),
    Object.entries(VISUAL_FILTER_OPTION_HELP).flatMap(([id, help]) => {
      const liveLabel = LIVE_SWITCH_NAMES[id]?.label;
      return liveLabel === undefined
        ? [[`[data-visual-filter-option='${id}']`, help.title, help.text]]
        : [
            [
              `html[data-product-kind='replay_viewer'] [data-visual-filter-option='${id}']`,
              help.title,
              help.text,
            ],
            [
              `html[data-product-kind='combat_debugger'] [data-visual-filter-option='${id}']`,
              liveLabel,
              help.text,
            ],
          ];
    }),
  );
  const deathAnnouncerTitles = VISUAL_FILTER_CONTROL_HELP.filter(([selector]) =>
    selector.endsWith("[data-visual-filter-option='death_announcer']"),
  ).map(([, title]) => title);
  assert.deepEqual(deathAnnouncerTitles, ["Fallen Feed", "Death Announcer"]);
  for (const row of VISUAL_FILTER_CONTROL_HELP) {
    assert.equal(row.length, 3);
    assert.doesNotMatch(row[0], /#battlefield/u);
  }
  const helpIds = Object.values(VISUAL_FILTER_OPTION_HELP).map(({ id }) => id);
  assert.equal(new Set(helpIds).size, helpIds.length);
});

test("the page keeps the strings and names private tools rely on", async () => {
  const [markup, main] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(mainUrl, "utf8"),
  ]);
  for (const anchor of [
    '<div class="header-status"',
    '<div class="toolbar-actions"',
    '<aside\n      id="visual-tooltip"',
    "</head>",
    '<script type="module" src="/src/main.js"></script>',
  ]) {
    assert.equal(markup.split(anchor).length - 1, 1, JSON.stringify(anchor));
  }
  for (const id of ["app-title", "help-heading", "notice", "workspace"]) {
    openingTag(markup, id);
  }
  for (const className of ["session-toolbar", "recording-panel", "replay-timeline"]) {
    assert.match(markup, new RegExp(`\\bclass="${className}"`, "u"), className);
  }
  assert.match(main, /new CustomEvent\("marl-combat-configuration-installed", \{/u);
  assert.match(main, /keys\.length !== 2 \|\| keys\[0\] !== "product_kind"/u);
});

test("each planned page slot starts hidden, empty and unmarked by product", async () => {
  const markup = await readFile(indexUrl, "utf8");
  /** @type {Array<[string, string, RegExp]>} */
  const slots = [
    [
      "party-frames",
      "section",
      /id="match-scoreboard"[\s\S]*id="party-frames"[\s\S]*id="battlefield-shell"/u,
    ],
    [
      "title-card",
      "section",
      /id="battlefield-shell"[^>]*>[\s\S]*id="title-card"[\s\S]*<\/div>\s*<div class="battlefield-support">/u,
    ],
    [
      "live-camera-switch",
      "div",
      /id="battlefield-shell"[^>]*>[\s\S]*id="live-camera-switch"[\s\S]*<\/div>\s*<div class="battlefield-support">/u,
    ],
    [
      "team-rosters",
      "section",
      /class="battlefield-support"[\s\S]*id="team-rosters"[\s\S]*id="command-deck"/u,
    ],
    [
      "this-turn",
      "section",
      /<aside id="rail" class="hud-panel bc-rail"[^>]*>\s*<!--[^>]*-->\s*<section class="rc" id="turn-card"[\s\S]*?id="howto-card"[\s\S]*?<\/section>\s*<!--[^>]*-->\s*<section id="this-turn"/u,
    ],
    [
      "how-to-play",
      "section",
      /id="this-turn"[^>]*><\/section>\s*<!--[^>]*-->\s*<section id="how-to-play"/u,
    ],
    [
      "fallen-feed",
      "section",
      /id="how-to-play"[^>]*><\/section>\s*<!--[^>]*-->\s*<section id="fallen-feed"/u,
    ],
    [
      "chronicle",
      "div",
      /id="replay-transport-status"[\s\S]*<\/fieldset>\s*<!--[^>]*-->\s*<div id="chronicle"/u,
    ],
    [
      "replay-camera-switch",
      "div",
      /id="replay-frame-position"[\s\S]*id="replay-camera-switch"[\s\S]*id="replay-transport-status"/u,
    ],
    [
      "replay-rail",
      "div",
      /id="fallen-feed"[^>]*><\/section>\s*<!--[^>]*-->\s*<div id="replay-rail"/u,
    ],
    ["replay-rail-tabs", "div", /id="replay-rail"[\s\S]*id="replay-rail-tabs"/u],
    [
      "replay-rail-agent",
      "section",
      /id="replay-rail-tabs"[\s\S]*id="replay-rail-agent"/u,
    ],
    [
      "replay-rail-metrics",
      "section",
      /id="replay-rail-agent"[\s\S]*id="replay-rail-metrics"/u,
    ],
    [
      "replay-rail-actions",
      "section",
      /id="replay-rail-metrics"[\s\S]*id="replay-rail-actions"/u,
    ],
    [
      "replay-rail-visuals",
      "section",
      /id="replay-rail-actions"[\s\S]*id="replay-rail-visuals"/u,
    ],
    [
      "replay-rail-frame",
      "section",
      /id="replay-rail-visuals"[\s\S]*id="replay-rail-frame"/u,
    ],
  ];
  for (const [id, tagName, placement] of slots) {
    const tag = openingTag(markup, id);
    assert.match(tag[0], new RegExp(`^<${tagName}\\b`, "u"), id);
    assert.match(tag[0], /\shidden[\s>]/u, `#${id} must start hidden.`);
    assert.doesNotMatch(tag[0], /data-(?:live|replay)-only/u, id);
    assert.match(markup, placement, `#${id} is in its planned place.`);
    if (id !== "replay-rail") {
      const rest = markup.slice((tag.index ?? 0) + tag[0].length);
      assert.match(rest, new RegExp(`^</${tagName}>`, "u"), `#${id} must start empty.`);
    }
  }
  const rail = /<div id="replay-rail" hidden>([\s\S]*?)\n {8}<\/div>/u.exec(markup);
  assert.ok(rail);
  assert.deepEqual(
    [...rail[1].matchAll(/\bid="([^"]+)"/gu)].map((match) => match[1]),
    [
      "replay-rail-tabs",
      "replay-rail-agent",
      "replay-rail-metrics",
      "replay-rail-actions",
      "replay-rail-visuals",
      "replay-rail-frame",
    ],
  );

  const toolbar = /<div class="toolbar-actions">([\s\S]*?)\n {6}<\/div>/u.exec(markup);
  assert.ok(toolbar);
  const order = [...toolbar[1].matchAll(/<button\b[^>]*\bid="([^"]+)"/gu)].map(
    (match) => match[1],
  );
  assert.deepEqual(order, [
    "reconnect-button",
    "help-button",
    "visual-key-button",
    "visual-filters-button",
    "frame-explanation-button",
    "exit-button",
  ]);
  const filtersButton = openingTag(markup, "visual-filters-button");
  assert.match(filtersButton[0], /^<button\b/u);
  assert.match(filtersButton[0], /\btype="button"/u);
  assert.match(filtersButton[0], /\bpopovertarget="visual-filters-drawer"/u);
  assert.match(filtersButton[0], /\sdata-live-only[\s>]/u);
  assert.match(markup, /id="visual-filters-button"[^>]*>Visual Filters<\/button>/u);
  assert.match(
    openingTag(markup, "visual-filters-drawer")[0],
    /^<div\b(?=[^>]*\bpopover="manual")/u,
  );
  for (const [id, label] of [
    ["visual-key-button", "Visual Key"],
    ["frame-explanation-button", "Frame Explanation"],
  ]) {
    const tag = openingTag(markup, id);
    assert.match(tag[0], /^<button\b/u);
    assert.match(tag[0], /\btype="button"/u);
    assert.match(tag[0], /\shidden[\s>]/u, `#${id} must start hidden.`);
    assert.doesNotMatch(tag[0], /data-(?:live|replay)-only/u, id);
    assert.match(markup, new RegExp(`id="${id}"[^>]*>${label}</button>`, "u"));
  }
});
