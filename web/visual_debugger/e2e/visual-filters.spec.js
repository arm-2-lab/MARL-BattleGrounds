/**
 * @file Check page-local visual preferences across live/replay audiences and dense
 * real scenario displays. A fresh page starts with every listed switch at its
 * shipped default (the owner's defaults of 4 Oct 2026): 67 of the 84 live
 * (DevClient combat screen) switches, 66 of the BattleClient's 83 (it leaves
 * Panel Tokens out: no docked panels) and 76 of the 95 Replay Viewer switches,
 * including Cooldown Effects and Red Zone Floors; the Replay Viewer starts with
 * Ranges off ("76 Of 96 On"). There is no Broadcast preset. Default
 * Configuration, Enable All and Disable All change only local paint and the
 * Ranges setting (the side panels are compared with every champion name read
 * the same way: Panel Tokens, which Enable All and Disable All also flip, only
 * chooses token chips or plain names there). A scenario smoke page shows no error notice in the top bar or
 * under the Replay Viewer's readout. Every filter has a short help line: under
 * its switch, and in the Replay Viewer also as a hover card on hover and
 * keyboard focus (live play never shows control hover cards, so there the card
 * stays hidden while the hover and focus still reach the switch), and
 * its checkbox's name is the switch name and its default tag ("Aura Fields Off
 * By Default"), and its description is the shared help and its own line. Turning Red
 * Zone Floors off and on in Oracle, or Cooldown Effects off and on in Oracle
 * and Agent POV, sends no request and leaves the scientific data unchanged; the
 * Replay Viewer's This Tick and Next Tick are read from its Actions tab, live
 * play's from the Pending and Latest Transition cards.
 */
import { expect, test } from "@playwright/test";

import { REPLAY_AUTOPLAY_CADENCE_MS } from "../src/replay-controls.js";
import {
  VISUAL_FILTER_IDS,
  VISUAL_FILTER_REGISTRY,
  visualFiltersForProduct,
} from "../src/visual-filters.js";
import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import {
  expectReplayFrameIndex,
  exportReplayArtifacts,
  removeReplayArtifacts,
  startReplayViewer,
} from "./support/replay-viewer.js";

test.describe.configure({ mode: "serial" });
test.use({ screenshot: "off" });

// The V5 switch registry is the one owner of switch IDs, labels, help and
// defaults. Each client shows only its own switches; the paint key covers
// every registered switch in registry order.
const FILTER_IDS = [...VISUAL_FILTER_IDS];
const INITIAL_FILTERS = new Set(
  VISUAL_FILTER_REGISTRY.filter(({ defaultEnabled }) => defaultEnabled).map(
    ({ id }) => id,
  ),
);
const INITIAL_DISABLED_FILTERS = FILTER_IDS.filter((id) => !INITIAL_FILTERS.has(id));
const FILTER_INPUT = 'input[type="checkbox"][data-visual-filter-id]';
const CHOREOGRAPHY_ROOTS =
  "#battlefield .combat-choreography, #battlefield .combat-choreography-connectors, #battlefield .combat-choreography-routes";
// Red Zone Floors paints the ash floor, its gravestone border and the marks.
const RED_ZONE_FLOOR_PAINT =
  "#battlefield .red-zone-floor, #battlefield .grave-border, #battlefield .grave-mark";
// Cooldown Effects paints the Rune Ring on each body (a dim track with lit
// arcs, or the ready ring with its gems).
const COOLDOWN_RING_PAINT = '#battlefield g.status-marks [data-rune-piece^="ring."]';
const DEFAULT_CONFIGURATION_HELP =
  "Every switch as it ships: what a new viewer sees. The view, Ranges and motion stay as they are.";

/** @param {import("@playwright/test").Page} page */
async function shownFilters(page) {
  const product = await page.locator("html").getAttribute("data-product-kind");
  // The BattleClient has no docked panels for Panel Tokens to change, so its
  // drawer leaves that switch out; the DevClient (its Developer drawer shown)
  // keeps it (main.js installVisualFilterControls omit).
  const developerDrawer = await page
    .locator("#developer-drawer")
    .evaluate((drawer) => !(drawer instanceof HTMLElement) || !drawer.hidden)
    .catch(() => false);
  return visualFiltersForProduct(product).filter(
    ({ id }) =>
      product !== "combat_debugger" || developerDrawer || id !== "panel_tokens",
  );
}

/** @type {Awaited<ReturnType<typeof exportReplayArtifacts>> | null} */
let artifacts = null;
/** @type {Awaited<ReturnType<typeof startDebugger>> | null} */
let liveDebugger = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let sharedReplay = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let noSharedReplay = null;

test.beforeAll(async () => {
  /** @type {import("node:child_process").ChildProcess[]} */
  const startedProcesses = [];
  try {
    artifacts = await exportReplayArtifacts();
    liveDebugger = await startDebugger();
    startedProcesses.push(liveDebugger.process);
    sharedReplay = await startReplayViewer({
      replayPath: artifacts.shared,
      frameIndex: 1,
    });
    startedProcesses.push(sharedReplay.process);
    noSharedReplay = await startReplayViewer({
      replayPath: artifacts.complete,
      frameIndex: 1,
    });
    startedProcesses.push(noSharedReplay.process);
  } catch (error) {
    const cleanupResults = await Promise.allSettled(
      startedProcesses.map((process) => stopDebugger(process)),
    );
    const cleanupErrors = cleanupResults.flatMap((result) =>
      result.status === "rejected" ? [result.reason] : [],
    );
    liveDebugger = null;
    sharedReplay = null;
    noSharedReplay = null;
    try {
      await removeReplayArtifacts(artifacts?.outputDirectory);
    } catch (cleanupError) {
      cleanupErrors.push(cleanupError);
    }
    artifacts = null;
    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        [error, ...cleanupErrors],
        "Visual-filter E2E startup and cleanup both failed.",
      );
    }
    throw error;
  }
});

test.afterAll(async () => {
  const processes = [
    liveDebugger?.process ?? null,
    sharedReplay?.process ?? null,
    noSharedReplay?.process ?? null,
  ];
  liveDebugger = null;
  sharedReplay = null;
  noSharedReplay = null;
  const cleanupResults = await Promise.allSettled(
    processes.map((process) => stopDebugger(process)),
  );
  const cleanupErrors = cleanupResults.flatMap((result) =>
    result.status === "rejected" ? [result.reason] : [],
  );
  try {
    await removeReplayArtifacts(artifacts?.outputDirectory);
  } catch (error) {
    cleanupErrors.push(error);
  }
  artifacts = null;
  if (cleanupErrors.length > 0) {
    throw new AggregateError(cleanupErrors, "Visual-filter E2E cleanup failed.");
  }
});

/** @template T
 * @param {T | null} value
 * @param {string} label
 * @returns {T} */
function requiredService(value, label) {
  if (value === null) {
    throw new Error(`${label} was not started.`);
  }
  return value;
}

/** @param {import("@playwright/test").Page} page */
function captureBrowserErrors(page) {
  /** @type {string[]} */
  const errors = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
  return errors;
}

/** @param {import("@playwright/test").Page} page */
function captureApiRequests(page) {
  /** @type {{method: string, path: string, conditionalSnapshot?: boolean}[]} */
  const requests = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path.startsWith("/api/")) {
      requests.push({
        method: request.method(),
        path,
        ...(request.method() === "GET" &&
        path === "/api/live/snapshot" &&
        request.headers()["if-none-match"] !== undefined
          ? { conditionalSnapshot: true }
          : {}),
      });
    }
  });
  return requests;
}

/** @param {import("@playwright/test").Page} page
 * @param {string} path */
async function authenticatedText(page, path) {
  return page.evaluate(async (requestPath) => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) {
      throw new Error("Debugger capability token is unavailable.");
    }
    const response = await fetch(requestPath, {
      cache: "no-store",
      credentials: "omit",
      headers: { "X-MARL-Debugger-Token": token },
      redirect: "error",
    });
    if (!response.ok) {
      throw new Error(`${requestPath} failed with HTTP ${response.status}.`);
    }
    return response.text();
  }, path);
}

/** @param {import("@playwright/test").Page} page
 * @param {string} url
 * @param {{viewerMode: "live" | "replay", audience: string, presentationKind: string}} expected */
async function openInstalled(page, url, expected) {
  await page.goto(url);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const root = page.locator("html");
  await expect(root).toHaveAttribute("data-viewer-mode", expected.viewerMode);
  await expect(root).toHaveAttribute("data-audience", expected.audience);
  await expect(root).toHaveAttribute("data-presentation-authority", "installed");
  await expect(root).toHaveAttribute("data-submission-blocked", "false");
  const presentation = JSON.parse(
    await authenticatedText(page, "/api/presentation/frame"),
  );
  expect(presentation.presentation_kind).toBe(expected.presentationKind);
  expect(page.url()).not.toContain("token=");
}

/** @param {import("@playwright/test").Page} page */
async function settleLocalRender(page) {
  await page.evaluate(
    () =>
      new Promise((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      }),
  );
}

/** @param {import("@playwright/test").Page} page
 * @param {{method: string, path: string, conditionalSnapshot?: boolean}[]} apiRequests
 * @param {() => Promise<void>} action
 * @param {{label: string, delayMs?: number}} options */
async function expectLocalOnly(page, apiRequests, action, options) {
  const mark = apiRequests.length;
  await action();
  await settleLocalRender(page);
  const delayMs = options.delayMs;
  if (delayMs !== undefined && delayMs > 0) {
    await page.waitForTimeout(delayMs);
  }
  const localRequests = apiRequests
    .slice(mark)
    .filter(({ conditionalSnapshot }) => !conditionalSnapshot);
  expect(localRequests, `${options.label} caused an API request`).toEqual([]);
}

/** @param {import("@playwright/test").Page} page
 * @param {"/api/command" | "/api/replay/command"} path
 * @param {() => Promise<unknown>} action */
async function performCommand(page, path, action) {
  // The deck and the arena sit under the live Visual Filters drawer.
  await closeLiveVisualFilters(page);
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === path,
  );
  await action();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-submission-blocked",
    "false",
  );
}

/**
 * Seek with the replay slider the way a let-go does. The Rune Scrubber keeps
 * the native range input (#replay-frame-slider) for screen readers only (its
 * label is visually hidden) and draws its own track, so the input has no box
 * to fill: set its value, then send input and change.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} frameIndex The replay frame to seek to.
 */
async function seekWithSlider(page, frameIndex) {
  await page.locator("#replay-frame-slider").evaluate((slider, value) => {
    if (!(slider instanceof HTMLInputElement)) {
      throw new TypeError("Replay slider is unavailable.");
    }
    slider.value = String(value);
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    slider.dispatchEvent(new Event("change", { bubbles: true }));
  }, frameIndex);
}

/**
 * Show the Replay Viewer's Roster tab (the roster rows sit in it); live play
 * shows its roster without tabs, so nothing changes there.
 *
 * @param {import("@playwright/test").Page} page
 */
async function showReplayRoster(page) {
  const rosterTab = page.locator("#replay-rail-tab-roster");
  if (await rosterTab.isVisible()) await rosterTab.click();
}

// The live drawer is a manual popover: it stays open until its top bar button
// closes it, as a player does before using the deck. The Replay Viewer keeps
// its filters in the rail, so nothing happens there.
/** @param {import("@playwright/test").Page} page */
async function closeLiveVisualFilters(page) {
  const drawer = page.locator("#visual-filters-drawer");
  if (await drawer.evaluate((node) => node.matches(":popover-open"))) {
    await page.locator("#visual-filters-button").click();
    await expect(drawer).toBeHidden();
  }
}

// Enable All, Disable All, Default Configuration and Ranges sit with the
// switches, so open them first (the live drawer may have closed for a command).
/** @param {import("@playwright/test").Page} page
 * @param {string} selector */
async function clickFilterControl(page, selector) {
  await openVisualFilters(page);
  await page.locator(selector).click();
}

/** @param {import("@playwright/test").Page} page */
async function openVisualFilters(page) {
  // The Replay Viewer keeps Visual Filters in the rail's Visuals tab; live play
  // keeps them in the Visual Filters drawer, which its button opens.
  const visualsTab = page.locator("#replay-rail-tab-visuals");
  if (await visualsTab.isVisible()) {
    await visualsTab.click();
  } else if (
    !(await page
      .locator("#visual-filters-drawer")
      .evaluate((drawer) => drawer.matches(":popover-open")))
  ) {
    await page.locator("#visual-filters-button").click();
  }
  await page.locator("#visual-filters").evaluate((details) => {
    if (!(details instanceof HTMLDetailsElement)) {
      throw new TypeError("Visual Filters disclosure is unavailable.");
    }
    details.open = true;
  });
  await expect(page.locator("#visual-filter-options")).toBeVisible();
}

/** @param {import("@playwright/test").Page} page
 * @param {string[]} disabledIds
 * @param {boolean} rangesEnabled */
async function expectFilterSurface(page, disabledIds = [], rangesEnabled = true) {
  const expectedDisabled = new Set(disabledIds);
  const shown = await shownFilters(page);
  const rows = await page
    .locator(`#visual-filter-options ${FILTER_INPUT}`)
    .evaluateAll((inputs) =>
      inputs.map((input) => {
        if (!(input instanceof HTMLInputElement)) {
          throw new TypeError("Visual filter is not an input.");
        }
        return {
          id: input.dataset.visualFilterId,
          option: input.parentElement?.dataset.visualFilterOption,
          // The first span holds the label; the second says "On By Default".
          label: input.parentElement?.querySelector("span")?.textContent?.trim(),
          // Focus adds visual-tooltip for a moment; expectVisualFilterHelp checks it.
          describedBy: (input.getAttribute("aria-describedby") ?? "")
            .split(/\s+/u)
            .filter((token) => token !== "visual-tooltip")
            .join(" "),
          value: input.value,
          type: input.type,
          autocomplete: input.autocomplete,
          defaultChecked: input.defaultChecked,
          checked: input.checked,
        };
      }),
    );
  // Rows follow the panel's groups; compare them as one record per switch.
  const byId = (/** @type {Array<{id?: string}>} */ list) =>
    [...list].sort((left, right) => String(left.id).localeCompare(String(right.id)));
  expect(byId(rows)).toEqual(
    byId(
      shown.map(({ id, label, defaultEnabled }) => ({
        id,
        option: id,
        label,
        describedBy: `visual-filters-help visual-filter-${id.replaceAll("_", "-")}-help`,
        value: id,
        type: "checkbox",
        autocomplete: "off",
        defaultChecked: defaultEnabled,
        checked: !expectedDisabled.has(id),
      })),
    ),
  );
  const enabledCount =
    shown.filter(({ id }) => !expectedDisabled.has(id)).length +
    (rangesEnabled ? 1 : 0);
  await expect(page.locator("#visual-filter-count")).toHaveText(
    `${enabledCount} Of ${shown.length + 1} On`,
  );
  const enableAll = page.locator("#enable-all-visual-filters-button");
  const disableAll = page.locator("#disable-all-visual-filters-button");
  if (enabledCount === shown.length + 1) {
    await expect(enableAll).toBeDisabled();
  } else {
    await expect(enableAll).toBeEnabled();
  }
  if (enabledCount === 0) {
    await expect(disableAll).toBeDisabled();
  } else {
    await expect(disableAll).toBeEnabled();
  }
  expect(
    await page.locator("#visual-filters button").evaluateAll((buttons) =>
      buttons.map((button) => ({
        id: button.id,
        text: button.textContent?.trim(),
      })),
    ),
  ).toEqual([
    { id: "default-visual-filters-button", text: "Default Configuration" },
    { id: "visual-filter-preset-researcher", text: "Researcher" },
    { id: "visual-filter-preset-manuscript", text: "Manuscript" },
    { id: "visual-filter-preset-minimal", text: "Minimal" },
    { id: "reset-visual-filters-button", text: "Reset To Default Configuration" },
    { id: "live-ranges-button", text: "Ranges" },
    { id: "replay-ranges-button", text: "Ranges" },
    { id: "enable-all-visual-filters-button", text: "Enable All" },
    { id: "disable-all-visual-filters-button", text: "Disable All" },
  ]);
}

/** @param {import("@playwright/test").Page} page */
async function expectVisualFilterHelp(page) {
  const tooltip = page.locator("#visual-tooltip");
  // Live play (BattleClient and DevClient) shows each switch's help line under
  // it and never a control hover card (symbols-ui.css); the Replay Viewer
  // shows the card on hover and focus.
  const live =
    (await page.locator("html").getAttribute("data-product-kind")) ===
    "combat_debugger";
  const sectionHelp = (await page.locator("#visual-filters-help").textContent()) ?? "";
  const descriptions = new Set();
  const shown = await shownFilters(page);
  for (const { id, label, help: registryHelp, defaultEnabled } of shown) {
    const helpId = `visual-filter-${id.replaceAll("_", "-")}-help`;
    const option = page.locator(
      `#visual-filter-options [data-visual-filter-option="${id}"]`,
    );
    const input = option.locator(FILTER_INPUT);
    const hiddenHelp = page.locator(`#visual-filter-options #${helpId}`);
    const ownDescribedBy = `visual-filters-help ${helpId}`;
    await expect(hiddenHelp).toHaveCount(1);
    // The help line shows under its switch (no longer screen-reader only).
    await expect(hiddenHelp).toHaveClass("visual-filters__option-help");
    const help = (await hiddenHelp.textContent()) ?? "";
    // The registry owns the help line, and every line says what Off shows.
    expect(help).toBe(registryHelp);
    expect(help).toContain("Off");
    descriptions.add(help);
    await expect(input).toHaveAttribute("aria-describedby", ownDescribedBy);
    // The label wraps the name and its default tag, so both are in the name.
    await expect(input).toHaveAccessibleName(
      `${label} ${defaultEnabled ? "On" : "Off"} By Default`,
    );
    await expect(input).toHaveAccessibleDescription(
      `${sectionHelp.replace(/\s+/gu, " ").trim()} ${help}`,
    );

    // Scroll the switch into view first and let that scroll land: a scroll
    // closes an open tip, and the live drawer scrolls under a hover.
    await option.scrollIntoViewIfNeeded();
    await settleLocalRender(page);
    await expect(hiddenHelp).toBeVisible();
    await option.hover();
    // Live: the hover reaches this switch (the card holds its title) but the
    // card stays hidden.
    await expect(page.locator("#visual-tooltip-title")).toHaveText(label);
    if (live) await expect(tooltip).toBeHidden();
    else await expect(tooltip).toBeVisible();
    await expect(page.locator("#visual-tooltip-details")).toHaveText(help);
    await page.mouse.move(1, 1);
    // The card's controller closes it (in live play too, where CSS already
    // keeps it out of sight): hidden attribute set and title cleared.
    await expect(tooltip).toHaveAttribute("hidden", "");
    await expect(page.locator("#visual-tooltip-title")).toHaveText("");

    await input.focus();
    await expect(page.locator("#visual-tooltip-title")).toHaveText(label);
    if (live) await expect(tooltip).toBeHidden();
    else await expect(tooltip).toBeVisible();
    await expect(page.locator("#visual-tooltip-details")).toHaveText(help);
    await expect(input).toHaveAttribute(
      "aria-describedby",
      `${ownDescribedBy} visual-tooltip`,
    );
    await input.blur();
    await expect(tooltip).toHaveAttribute("hidden", "");
    await expect(page.locator("#visual-tooltip-title")).toHaveText("");
    await expect(input).toHaveAttribute("aria-describedby", ownDescribedBy);
  }
  expect(descriptions.size).toBe(shown.length);
}

/** @param {import("@playwright/test").Page} page
 * @param {"#live-ranges-button" | "#replay-ranges-button"} rangesSelector */
async function expectFreshDefaults(page, rangesSelector) {
  await expectFilterSurface(page, INITIAL_DISABLED_FILTERS, false);
  await expect(page.locator(rangesSelector)).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator("#default-visual-filters-button")).toHaveAttribute(
    "aria-description",
    DEFAULT_CONFIGURATION_HELP,
  );
  await expectVisualFilterHelp(page);
}

/** @param {string[]} disabledIds */
function expectedPaintKey(disabledIds) {
  const disabled = new Set(disabledIds);
  return `visual-filters-v4:${FILTER_IDS.map((id) => (disabled.has(id) ? "0" : "1")).join("")}`;
}

/** @param {import("@playwright/test").Page} page */
function choreographyPaintKeys(page) {
  return page
    .locator(CHOREOGRAPHY_ROOTS)
    .evaluateAll((roots) => roots.map((root) => root.getAttribute("data-paint-key")));
}

/** @param {import("@playwright/test").Page} page */
function cooldownInventory(page) {
  return page.locator("#battlefield g.status-marks").evaluateAll((groups) =>
    groups
      .map((group) => ({
        key: group.getAttribute("data-marks-for"),
        ring: [...group.querySelectorAll('[data-rune-piece^="ring."]')].map((piece) =>
          piece.getAttribute("data-rune-piece"),
        ),
      }))
      .filter(({ ring }) => ring.length > 0),
  );
}

/** @param {import("@playwright/test").Page} page
 * @param {{method: string, path: string, conditionalSnapshot?: boolean}[]} apiRequests
 * @param {boolean} replay
 * @param {string} label */
async function expectCooldownToggle(page, apiRequests, replay, label) {
  const science = await scientificSignature(page, replay);
  const cooldowns = await cooldownInventory(page);
  expect(cooldowns.length, `${label} cooldown rings`).toBeGreaterThan(0);
  await setFilter(
    page,
    apiRequests,
    "cooldown_effects",
    false,
    `${label} cooldowns off`,
  );
  await expect(
    page.locator(
      `${COOLDOWN_RING_PAINT}, #battlefield .cooldown-cell, #battlefield .required-dock-fallback[data-kind="cooldown"]`,
    ),
  ).toHaveCount(0);
  expect(await scientificSignature(page, replay)).toEqual(science);
  await setFilter(page, apiRequests, "cooldown_effects", true, `${label} cooldowns on`);
  expect(await cooldownInventory(page)).toEqual(cooldowns);
  expect(await scientificSignature(page, replay)).toEqual(science);
}

/** @param {import("@playwright/test").Page} page
 * @param {{method: string, path: string, conditionalSnapshot?: boolean}[]} apiRequests
 * @param {string} filterId
 * @param {boolean} enabled
 * @param {string} label */
async function setFilter(page, apiRequests, filterId, enabled, label) {
  const input = page.locator(`${FILTER_INPUT}[data-visual-filter-id="${filterId}"]`);
  if (!(await input.isVisible())) await openVisualFilters(page);
  await expectLocalOnly(
    page,
    apiRequests,
    async () => {
      if (enabled) {
        await input.check();
      } else {
        await input.uncheck();
      }
    },
    { label },
  );
  await expect(input).toBeChecked({ checked: enabled });
}

/** @param {import("@playwright/test").Page} page
 * @param {boolean} replay */
async function scientificSignature(page, replay) {
  const apiPaths = replay
    ? ["/api/frame", "/api/presentation/frame", "/api/replay/timeline"]
    : ["/api/frame", "/api/presentation/frame"];
  const api = Object.fromEntries(
    await Promise.all(
      apiPaths.map(async (path) => [path, await authenticatedText(page, path)]),
    ),
  );
  const dom = await page.evaluate((replayViewer) => {
    /** @param {string} selector */
    const exactHtml = (selector) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing scientific surface ${selector}.`);
      }
      return element.innerHTML;
    };
    // The Replay Viewer shows This Tick and Next Tick in its Actions tab, each
    // part found by its heading words (owner's Actions tab structure, 3 October
    // 2026); live play keeps #pending-card and #accepted-card.
    /** @param {string} title */
    const actionsPartHtml = (title) => {
      const part = Array.from(
        document.querySelectorAll("#replay-rail-actions .panel-actions__part"),
      ).find(
        (candidate) =>
          candidate.querySelector(".panel-actions__title")?.textContent?.trim() ===
          title,
      );
      if (!(part instanceof HTMLElement)) {
        throw new Error(`Missing scientific surface Actions part ${title}.`);
      }
      return part.innerHTML;
    };
    const controls = Array.from(
      document.querySelectorAll(
        "#view-select, #command-deck button[data-key], #command-target-select, #reset-button",
      ),
    ).map((control) => ({
      id: control.id,
      tag: control.tagName,
      key: control.getAttribute("data-key"),
      value:
        control instanceof HTMLInputElement || control instanceof HTMLSelectElement
          ? control.value
          : null,
      disabled:
        control instanceof HTMLButtonElement ||
        control instanceof HTMLInputElement ||
        control instanceof HTMLSelectElement
          ? control.disabled
          : null,
      pressed: control.getAttribute("aria-pressed"),
      description: control.getAttribute("aria-description"),
      text: control.textContent?.trim() ?? "",
    }));
    return {
      authority: {
        productKind: document.documentElement.dataset.productKind,
        viewerMode: document.documentElement.dataset.viewerMode,
        audience: document.documentElement.dataset.audience,
        preset: document.documentElement.dataset.preset,
        presentationAuthority: document.documentElement.dataset.presentationAuthority,
      },
      step: document.querySelector("#step-value")?.textContent,
      transition: document.querySelector("#transition-value")?.textContent,
      selection: exactHtml("#selection-card"),
      pending: replayViewer
        ? actionsPartHtml("Next Tick (What Happens Next)")
        : exactHtml("#pending-card"),
      accepted: replayViewer
        ? actionsPartHtml("This Tick (What Happened)")
        : exactHtml("#accepted-card"),
      diagnostics: exactHtml("#diagnostics-card"),
      disclosures: [
        "roster-details",
        "agent-details",
        "pending-turn-details",
        "latest-transition-details",
        "technical-frame-details",
      ].map((id) => ({
        id,
        open: /** @type {HTMLDetailsElement | null} */ (document.getElementById(id))
          ?.open,
      })),
      controls,
      rosterAuthority: Array.from(
        document.querySelectorAll("#roster [data-presentation-key]"),
      ).map((element) => ({
        key: element.getAttribute("data-presentation-key"),
        controlled: element.getAttribute("data-controlled"),
        selected: element.getAttribute("data-selected"),
        pressed: element.getAttribute("aria-pressed"),
        disabled: element.getAttribute("aria-disabled"),
      })),
      battlefieldAuthority: Array.from(
        document.querySelectorAll("#battlefield .agent[data-presentation-key]"),
      ).map((element) => ({
        key: element.getAttribute("data-presentation-key"),
        controlled: element.getAttribute("data-controlled"),
        selected: element.getAttribute("data-selected"),
      })),
    };
  }, replay);
  return { api, dom };
}

/** @param {unknown} value */
function omitPresentationRevision(value) {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    Reflect.deleteProperty(value, "source_revision");
    Reflect.deleteProperty(value, "source_authority_epoch");
  }
}

/** @param {string} path
 * @param {string} serialized
 * @returns {unknown} */
function rangeIndependentApiPayload(path, serialized) {
  /** @type {Record<string, any>} */
  const payload = JSON.parse(serialized);
  if (path === "/api/frame") {
    delete payload.revision;
    delete payload.show_ranges;
    if (payload.projection?.scene) {
      delete payload.projection.scene.ranges;
    }
    if (Array.isArray(payload.hud?.diagnostics)) {
      for (const fact of payload.hud.diagnostics) {
        if (fact?.fact_id === "revision") {
          fact.value = "<range-presentation revision>";
        }
      }
    }
    return payload;
  }
  if (path === "/api/presentation/frame") {
    omitPresentationRevision(payload.source);
    omitPresentationRevision(payload.live_inspection);
    omitPresentationRevision(payload.researcher_space);
  }
  return payload;
}

/** @param {import("@playwright/test").Page} page
 * @param {boolean} replay */
async function rangeIndependentScientificSignature(page, replay) {
  const signature = await scientificSignature(page, replay);
  return {
    api: Object.fromEntries(
      Object.entries(signature.api).map(([path, payload]) => [
        path,
        rangeIndependentApiPayload(path, payload),
      ]),
    ),
    dom: signature.dom,
  };
}

// Panel Tokens (panel_tokens) only chooses how the side panels name a
// champion: a token chip or plain words. Enable All and Disable All flip it,
// even in a client that leaves the switch out of its drawer, so a comparison
// across them reads every name the same way: a chip becomes the plain name its
// label gives ("Agent ID 0, Mage, Team A" becomes "Mage · ID 0") and a row
// title's extra class word goes. Everything else in the panels is compared
// exactly.
/** @param {import("@playwright/test").Page} page
 * @param {boolean} replay */
async function bulkScientificSignature(page, replay) {
  const signature = await rangeIndependentScientificSignature(page, replay);
  const panels = /** @type {const} */ ([
    "selection",
    "pending",
    "accepted",
    "diagnostics",
  ]);
  const named = await page.evaluate(
    (htmlByPanel) =>
      Object.fromEntries(
        Object.entries(htmlByPanel).map(([key, html]) => {
          const template = document.createElement("template");
          // The page's CSP (style-src 'self') logs an error for every style
          // attribute parsed here, so the copy keeps each one under another
          // name: both sides of a comparison are renamed alike.
          template.innerHTML = String(html).replace(
            /(\s)style="/gu,
            '$1data-test-style="',
          );
          for (const chip of template.content.querySelectorAll(".token-chip")) {
            const label = chip.getAttribute("aria-label") ?? "";
            const match = /^Agent ID (\S+), ([^,]+), Team [AB]$/u.exec(label);
            chip.replaceWith(match ? `${match[2]} · ID ${match[1]}` : label);
          }
          for (const word of template.content.querySelectorAll(
            ".accepted-action-row__class",
          )) {
            word.remove();
          }
          template.content.normalize();
          return [key, template.innerHTML];
        }),
      ),
    Object.fromEntries(panels.map((key) => [key, signature.dom[key]])),
  );
  return { api: signature.api, dom: { ...signature.dom, ...named } };
}

/** @param {import("@playwright/test").Page} page
 * @param {"#live-ranges-button" | "#replay-ranges-button"} buttonSelector */
async function rangeSignature(page, buttonSelector) {
  return page.evaluate((selector) => {
    const button = document.querySelector(selector);
    const layer = document.querySelector('#battlefield [data-layer="debug-range"]');
    if (!(button instanceof HTMLButtonElement) || !(layer instanceof Element)) {
      throw new Error("Range presentation surface is unavailable.");
    }
    return {
      id: button.id,
      pressed: button.getAttribute("aria-pressed"),
      disabled: button.disabled,
      description: button.getAttribute("aria-description"),
      markup: layer.innerHTML,
    };
  }, buttonSelector);
}

/** @param {import("@playwright/test").Page} page */
async function targetSelectionPaintSignature(page) {
  return page.locator("#battlefield").evaluate((battlefield) => ({
    visibleReticles: battlefield.querySelectorAll(".selected-reticle:not([hidden])")
      .length,
    legality: Array.from(
      battlefield.querySelectorAll(".legality-pill"),
      (node) => node.outerHTML,
    ),
    controlled: battlefield.querySelector(".controlled-halo:not([hidden])")?.outerHTML,
    // Live play draws the staged order as the Order Thread in the plan marks;
    // the Replay Viewer keeps its line in the pending-route layer.
    pendingRoute:
      (battlefield.querySelector('[data-layer="pending-route"]')?.innerHTML ?? "") +
      Array.from(
        battlefield.querySelectorAll(
          '[data-layer="plan-marks"] [data-hotbar^="thread"]',
        ),
        (line) => line.outerHTML,
      ).join(""),
    actionRoutes: battlefield.querySelector(".combat-choreography-routes")?.innerHTML,
    ranges: battlefield.querySelector('[data-layer="debug-range"]')?.innerHTML,
    // The selection brackets live in the body layer's first group, so read
    // only the body groups.
    bodies: Array.from(
      battlefield.querySelectorAll('[data-layer="body"] > [data-body-state]'),
      (group) => group.outerHTML,
    ).join(""),
  }));
}

/** @param {import("@playwright/test").Page} page
 * @param {{method: string, path: string, conditionalSnapshot?: boolean}[]} apiRequests
 * @param {{url: string}} replay
 * @param {string} expectedPresentationKind
 * @param {string} label */
async function expectReplayAgentTargetFilter(
  page,
  apiRequests,
  replay,
  expectedPresentationKind,
  label,
) {
  await openInstalled(page, replay.url, {
    viewerMode: "replay",
    audience: "researcher",
    presentationKind: "replay_oracle",
  });
  await expectReplayFrameIndex(page, 1);
  await showReplayRoster(page);
  const candidate = page
    .locator('#roster .roster-primary-action:not([aria-pressed="true"])')
    .first();
  await expect(candidate).toBeEnabled();
  await performCommand(page, "/api/replay/command", () => candidate.click());
  await openVisualFilters(page);
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    `${label} enable replay-default target visuals`,
  );
  const oracleScience = await scientificSignature(page, true);
  const oraclePaint = await targetSelectionPaintSignature(page);
  expect(oraclePaint.visibleReticles, `${label} Oracle enabled reticle`).toBe(1);
  expect(oraclePaint.legality, `${label} Oracle enabled legality`).toEqual([]);
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    false,
    `${label} Oracle target visuals off`,
  );
  const filteredOracle = await targetSelectionPaintSignature(page);
  expect(filteredOracle.visibleReticles).toBe(0);
  expect(filteredOracle.legality).toEqual([]);
  expect(filteredOracle.controlled).toBe(oraclePaint.controlled);
  expect(filteredOracle.pendingRoute).toBe(oraclePaint.pendingRoute);
  expect(filteredOracle.actionRoutes).toBe(oraclePaint.actionRoutes);
  expect(filteredOracle.ranges).toBe(oraclePaint.ranges);
  expect(filteredOracle.bodies).toBe(oraclePaint.bodies);
  expect(await scientificSignature(page, true)).toEqual(oracleScience);
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    `${label} Oracle target visuals on`,
  );
  expect(await targetSelectionPaintSignature(page)).toEqual(oraclePaint);
  await performCommand(page, "/api/replay/command", () =>
    page.locator('#view-switch button[data-view-mode="pov"]').click(),
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe(expectedPresentationKind);
  const input = page.locator(
    `${FILTER_INPUT}[data-visual-filter-id="target_selection_visuals"]`,
  );
  if (!(await input.isChecked())) {
    await setFilter(
      page,
      apiRequests,
      "target_selection_visuals",
      true,
      `${label} target visuals initial restore`,
    );
  }
  const science = await scientificSignature(page, true);
  const paint = await targetSelectionPaintSignature(page);
  expect(paint.visibleReticles, `${label} enabled reticle`).toBe(1);
  expect(paint.legality, `${label} enabled legality`).toEqual([]);

  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    false,
    `${label} target visuals off`,
  );
  const filtered = await targetSelectionPaintSignature(page);
  expect(filtered.visibleReticles).toBe(0);
  expect(filtered.legality).toEqual([]);
  expect(filtered.controlled).toBe(paint.controlled);
  expect(filtered.pendingRoute).toBe(paint.pendingRoute);
  expect(filtered.actionRoutes).toBe(paint.actionRoutes);
  expect(filtered.ranges).toBe(paint.ranges);
  expect(filtered.bodies).toBe(paint.bodies);
  expect(await scientificSignature(page, true)).toEqual(science);

  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    `${label} target visuals on`,
  );
  expect(await targetSelectionPaintSignature(page)).toEqual(paint);
  expect(await scientificSignature(page, true)).toEqual(science);
  await performCommand(page, "/api/replay/command", () =>
    page.locator('#view-switch button[data-view-mode="researcher"]').click(),
  );
  await page.goto("about:blank");
}

/** @param {import("@playwright/test").Page} page
 * @param {"live" | "replay"} mode */
async function ensureRangesOn(page, mode) {
  const selector = mode === "live" ? "#live-ranges-button" : "#replay-ranges-button";
  const path = mode === "live" ? "/api/command" : "/api/replay/command";
  const button = page.locator(selector);
  if ((await button.getAttribute("aria-pressed")) !== "true") {
    await performCommand(page, path, () => clickFilterControl(page, selector));
  }
  await expect(button).toHaveAttribute("aria-pressed", "true");
}

/** @param {import("@playwright/test").Page} page */
function auraMarkup(page) {
  return page.locator('#battlefield [data-layer="aura"]').innerHTML();
}

/** @param {import("@playwright/test").Page} page */
async function expectTooltipCleared(page) {
  await page.evaluate(() => {
    const active = document.activeElement;
    if (active instanceof HTMLElement || active instanceof SVGElement) {
      active.blur();
    }
  });
  await page.mouse.move(1, 1);
  await settleLocalRender(page);
  await expect(page.locator("#visual-tooltip")).toBeHidden();
  await expect(page.locator("#visual-tooltip-title")).toHaveText("");
  await expect(page.locator("#visual-tooltip-details")).toHaveText("");
  await expect(
    page.locator('#battlefield [aria-describedby~="visual-tooltip"]'),
  ).toHaveCount(0);
}

/** @param {import("@playwright/test").Page} page */
async function expectAuraAbsent(page) {
  await expect(page.locator('#battlefield [data-layer="aura"]')).toBeEmpty();
  await expect(
    page.locator(
      "#battlefield .aura-field, #battlefield .aura-field-owner, #battlefield .aura-field-hit",
    ),
  ).toHaveCount(0);
  await expectTooltipCleared(page);
}

/** @param {import("@playwright/test").Page} page */
async function ultimateInventory(page) {
  return page
    .locator('#battlefield .combat-effect--activation[data-component="ultimate"]')
    .evaluateAll((elements) =>
      elements.map((element) => ({
        eventId: element.getAttribute("data-event-id"),
        eventType: element.getAttribute("data-event-type"),
        component: element.getAttribute("data-component"),
        tokenId: element.getAttribute("data-token-id"),
        sourceClass: element.getAttribute("data-source-class"),
        ariaLabel: element.getAttribute("aria-label"),
        ariaDescription: element.getAttribute("aria-description"),
        icons: Array.from(element.querySelectorAll("[data-icon]"), (icon) =>
          icon.getAttribute("data-icon"),
        ),
      })),
    );
}

/** @param {import("@playwright/test").Page} page
 * @param {string[]} eventIds */
async function expectUltimateAbsent(page, eventIds) {
  const residue = await page.locator("#battlefield").evaluate((battlefield, ids) => {
    const markup = battlefield.innerHTML;
    return {
      ownedEventIds: Array.from(
        battlefield.querySelectorAll("[data-event-id]"),
        (element) => element.getAttribute("data-event-id"),
      ).filter((id) => id !== null && ids.includes(id)),
      serializedIds: ids.filter((id) => markup.includes(id)),
    };
  }, eventIds);
  expect(residue).toEqual({ ownedEventIds: [], serializedIds: [] });
  await expectTooltipCleared(page);
}

/** @param {import("@playwright/test").Page} page */
async function disableAllFilters(page) {
  const replay =
    (await page.locator("html").getAttribute("data-viewer-mode")) === "replay";
  await performCommand(page, replay ? "/api/replay/command" : "/api/command", () =>
    clickFilterControl(page, "#disable-all-visual-filters-button"),
  );
}

/** @param {import("@playwright/test").Page} page */
async function expectAllPaintAbsentWithoutDwell(page) {
  await expect(
    page.locator(
      [
        "#battlefield .aura-field",
        "#battlefield .modifier-cell",
        "#battlefield .modifier-overflow",
        "#battlefield .status-cell",
        "#battlefield .status-overflow",
        '#battlefield .required-dock-fallback[data-kind="status"]',
        "#battlefield .pov-observed-status",
        "#battlefield .agent-spawn-shield",
        COOLDOWN_RING_PAINT,
        RED_ZONE_FLOOR_PAINT,
        "#battlefield .combat-effect",
        "#battlefield .combat-connector-effect",
        "#battlefield .combat-route-effect",
      ].join(", "),
    ),
  ).toHaveCount(0);
  const state = await page.evaluate(() => ({
    blocked: document.documentElement.dataset.submissionBlocked,
    roots: Array.from(
      document.querySelectorAll(
        "#battlefield .combat-choreography, #battlefield .combat-choreography-connectors, #battlefield .combat-choreography-routes",
      ),
    ).map((root) => ({
      state: root.getAttribute("data-state"),
      paintKey: root.getAttribute("data-paint-key"),
      childCount: root.childElementCount,
    })),
    animations: document
      .getAnimations()
      .map((animation) => animation.id)
      .filter((id) => id.startsWith("mbg:")),
    agentAria: Array.from(document.querySelectorAll("#battlefield .agent"), (agent) =>
      agent.getAttribute("aria-label"),
    ),
  }));
  expect(state.blocked).toBe("false");
  expect(state.roots).toEqual([
    {
      state: "settled",
      paintKey: `visual-filters-v4:${"0".repeat(FILTER_IDS.length)}`,
      childCount: 0,
    },
    {
      state: "settled",
      paintKey: `visual-filters-v4:${"0".repeat(FILTER_IDS.length)}`,
      childCount: 0,
    },
    {
      state: "settled",
      paintKey: `visual-filters-v4:${"0".repeat(FILTER_IDS.length)}`,
      childCount: 0,
    },
  ]);
  expect(state.animations).toEqual([]);
  for (const ariaLabel of state.agentAria) {
    expect(ariaLabel).not.toMatch(/combat status|spawn shield/iu);
  }
  await expectTooltipCleared(page);
}

test("visual filters remain page-local across live Oracle/NoShared and replay Oracle/Shared authority", async ({
  page,
}) => {
  const browserErrors = captureBrowserErrors(page);
  const apiRequests = captureApiRequests(page);

  const live = requiredService(liveDebugger, "Live debugger");
  await openInstalled(page, live.url, {
    viewerMode: "live",
    audience: "researcher",
    presentationKind: "live_oracle",
  });
  await openVisualFilters(page);
  await expectFreshDefaults(page, "#live-ranges-button");
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  await expect(page.locator("#visual-filters")).toHaveAttribute("open", "");
  await ensureRangesOn(page, "live");
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator("#enable-all-visual-filters-button").evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Enable All is unavailable.");
        }
        button.click();
      }),
    { label: "initial Enable All restores every paint family" },
  );
  await expectFilterSurface(page);

  const liveBulkScience = await bulkScientificSignature(page, false);
  const liveBulkRanges = await rangeSignature(page, "#live-ranges-button");
  const liveBulkStep = await page.locator("#step-value").textContent();
  const liveDisableMark = apiRequests.length;
  await disableAllFilters(page);
  expect(
    apiRequests.slice(liveDisableMark).filter(({ method }) => method === "POST"),
    "Live Disable All range request count",
  ).toEqual([{ method: "POST", path: "/api/command" }]);
  await expectFilterSurface(page, FILTER_IDS, false);
  expect(await bulkScientificSignature(page, false)).toEqual(liveBulkScience);
  await expect(page.locator("#step-value")).toHaveText(liveBulkStep ?? "0");
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator("#disable-all-visual-filters-button").evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Disable All is unavailable.");
        }
        button.click();
      }),
    { label: "live disabled Disable All idempotency" },
  );

  const liveEnableMark = apiRequests.length;
  await performCommand(page, "/api/command", () =>
    clickFilterControl(page, "#enable-all-visual-filters-button"),
  );
  expect(
    apiRequests.slice(liveEnableMark).filter(({ method }) => method === "POST"),
    "Live Enable All range request count",
  ).toEqual([{ method: "POST", path: "/api/command" }]);
  await expectFilterSurface(page);
  expect(await bulkScientificSignature(page, false)).toEqual(liveBulkScience);
  expect(await rangeSignature(page, "#live-ranges-button")).toEqual(liveBulkRanges);
  await expect(page.locator("#step-value")).toHaveText(liveBulkStep ?? "0");
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator("#enable-all-visual-filters-button").evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Enable All is unavailable.");
        }
        button.click();
      }),
    { label: "live disabled Enable All idempotency" },
  );

  // Default Configuration restores every shipped default and leaves Ranges as
  // they are (DEFAULT_CONFIGURATION_HELP), so it sends nothing; a second press
  // changes nothing either.
  await expectLocalOnly(
    page,
    apiRequests,
    () => clickFilterControl(page, "#default-visual-filters-button"),
    { label: "live Default Configuration" },
  );
  await expectFilterSurface(page, INITIAL_DISABLED_FILTERS, true);
  await expect(page.locator("#live-ranges-button")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(await bulkScientificSignature(page, false)).toEqual(liveBulkScience);
  await expect(page.locator("#step-value")).toHaveText(liveBulkStep ?? "0");
  await expectLocalOnly(
    page,
    apiRequests,
    () => clickFilterControl(page, "#default-visual-filters-button"),
    { label: "live Default Configuration idempotency" },
  );
  await expectFilterSurface(page, INITIAL_DISABLED_FILTERS, true);
  // Ranges are already on, so Enable All changes only switches: no request.
  await expectLocalOnly(
    page,
    apiRequests,
    () => clickFilterControl(page, "#enable-all-visual-filters-button"),
    { label: "live Enable All with Ranges on" },
  );
  await expectFilterSurface(page);
  expect(await rangeSignature(page, "#live-ranges-button")).toEqual(liveBulkRanges);

  // Agent ID 4's row in the deck's checklist gives it the orders (a roster
  // selection command, as the old Control button sent).
  await performCommand(page, "/api/command", () =>
    page.locator('#checklist .cl-row[data-slot="4"]').click(),
  );
  await performCommand(page, "/api/command", () =>
    page.locator('#command-deck button[data-key="d"]').click(),
  );
  // Allowed Now (#command-target-select) holds one pick button per allowed
  // target; picking Agent ID 3 stages it as the target.
  const targetPick = page.locator(
    '#command-target-select .pick[aria-label*="Agent ID 3,"]',
  );
  await expect(targetPick).toHaveCount(1);
  await performCommand(page, "/api/command", () => targetPick.click());
  await performCommand(page, "/api/command", () =>
    page.locator('#command-deck button[data-key="1"]').click(),
  );
  await expect(targetPick).toHaveAttribute("aria-pressed", "true");
  const targetScience = await scientificSignature(page, false);
  const targetPaint = await targetSelectionPaintSignature(page);
  expect(targetPaint.visibleReticles).toBe(1);
  expect(targetPaint.legality).toEqual([]);
  expect(targetPaint.controlled).toBeDefined();
  expect(targetPaint.pendingRoute).not.toBe("");

  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    false,
    "live target-selection visuals off",
  );
  const targetPaintHidden = await targetSelectionPaintSignature(page);
  expect(targetPaintHidden.visibleReticles).toBe(0);
  expect(targetPaintHidden.legality).toEqual([]);
  expect(targetPaintHidden.controlled).toBe(targetPaint.controlled);
  expect(targetPaintHidden.pendingRoute).toBe(targetPaint.pendingRoute);
  expect(targetPaintHidden.actionRoutes).toBe(targetPaint.actionRoutes);
  expect(targetPaintHidden.ranges).toBe(targetPaint.ranges);
  expect(targetPaintHidden.bodies).toBe(targetPaint.bodies);
  expect(await scientificSignature(page, false)).toEqual(targetScience);
  await expect(targetPick).toHaveAttribute("aria-pressed", "true");
  await expectTooltipCleared(page);

  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    "live target-selection visuals on",
  );
  expect(await targetSelectionPaintSignature(page)).toEqual(targetPaint);
  expect(await scientificSignature(page, false)).toEqual(targetScience);
  await performCommand(page, "/api/command", () =>
    page.locator("#clear-target-button").click(),
  );
  await expect(
    page.locator('#command-target-select .pick[aria-pressed="true"]'),
  ).toHaveCount(0);

  const liveScience = await scientificSignature(page, false);
  const liveRanges = await rangeSignature(page, "#live-ranges-button");
  const liveAuras = await auraMarkup(page);
  expect(liveAuras).not.toBe("");
  await expect(
    page.locator(
      '#battlefield [data-layer="aura"] .aura-field[role="img"][data-tooltip-owner]',
    ),
  ).not.toHaveCount(0);

  await setFilter(page, apiRequests, "aura_fields", false, "live aura off");
  await expectFilterSurface(page, ["aura_fields"]);
  await expectAuraAbsent(page);
  expect(await scientificSignature(page, false)).toEqual(liveScience);
  expect(await rangeSignature(page, "#live-ranges-button")).toEqual(liveRanges);

  await setFilter(page, apiRequests, "aura_fields", true, "live aura on");
  expect(await auraMarkup(page)).toBe(liveAuras);
  expect(await scientificSignature(page, false)).toEqual(liveScience);
  expect(await rangeSignature(page, "#live-ranges-button")).toEqual(liveRanges);

  await setFilter(page, apiRequests, "aura_fields", false, "live durable filter");
  await setFilter(
    page,
    apiRequests,
    "ultimate_ability_effects",
    false,
    "live transient filter",
  );
  const disabledPair = ["aura_fields", "ultimate_ability_effects"];
  await expectFilterSurface(page, disabledPair);

  const initialStep = Number(await page.locator("#step-value").textContent());
  await performCommand(page, "/api/command", () =>
    page.locator("#submit-turn-button").click(),
  );
  await expect(page.locator("#step-value")).toHaveText(String(initialStep + 1));
  await expectFilterSurface(page, disabledPair);

  await performCommand(page, "/api/command", () =>
    page.locator('#view-switch button[data-view-mode="pov"]').click(),
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  const livePov = JSON.parse(await authenticatedText(page, "/api/presentation/frame"));
  expect(livePov.presentation_kind).toBe("live_shared_obs_agent_pov");
  await expectLocalOnly(
    page,
    apiRequests,
    () => clickFilterControl(page, "#live-ranges-button"),
    {
      label: "Enable Agent-local ranges for the visibility proof",
    },
  );
  await expectFilterSurface(page, disabledPair);

  const visiblePovPick = page.locator(
    '#command-target-select .pick[aria-label*="Agent ID 3,"]',
  );
  await expect(visiblePovPick).toHaveCount(1);
  await performCommand(page, "/api/command", () => visiblePovPick.click());
  await performCommand(page, "/api/command", () =>
    page.locator('#command-deck button[data-key="1"]').click(),
  );
  const visiblePovScience = await scientificSignature(page, false);
  const visiblePovPaint = await targetSelectionPaintSignature(page);
  expect(visiblePovPaint.visibleReticles).toBe(1);
  expect(visiblePovPaint.legality).toEqual([]);
  expect(visiblePovPaint.pendingRoute).not.toBe("");
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    false,
    "live Agent visible target visuals off",
  );
  const hiddenVisiblePovPaint = await targetSelectionPaintSignature(page);
  expect(hiddenVisiblePovPaint.visibleReticles).toBe(0);
  expect(hiddenVisiblePovPaint.legality).toEqual([]);
  expect(hiddenVisiblePovPaint.controlled).toBe(visiblePovPaint.controlled);
  expect(hiddenVisiblePovPaint.pendingRoute).toBe(visiblePovPaint.pendingRoute);
  expect(hiddenVisiblePovPaint.actionRoutes).toBe(visiblePovPaint.actionRoutes);
  expect(hiddenVisiblePovPaint.ranges).toBe(visiblePovPaint.ranges);
  expect(hiddenVisiblePovPaint.bodies).toBe(visiblePovPaint.bodies);
  expect(await scientificSignature(page, false)).toEqual(visiblePovScience);
  await expect(visiblePovPick).toHaveAttribute("aria-pressed", "true");
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    "live Agent visible target visuals on",
  );
  expect(await targetSelectionPaintSignature(page)).toEqual(visiblePovPaint);

  // Agent POV: Allowed Now lists only bodies the team's scene draws
  // (allowedNowTargets), so the fog-hidden enemy Agent ID 9 gets no pick and
  // the frame's scene does not hold it.
  const fogHiddenPick = page.locator(
    '#command-target-select .pick[aria-label*="Agent ID 9,"]',
  );
  await expect(fogHiddenPick).toHaveCount(0);
  const fogHiddenPresentation = JSON.parse(
    await authenticatedText(page, "/api/presentation/frame"),
  );
  expect(
    fogHiddenPresentation.current_endpoint.parts.scene.agents.some(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "9",
    ),
  ).toBe(false);
  await performCommand(page, "/api/command", () =>
    page.locator("#clear-target-button").click(),
  );

  await performCommand(page, "/api/command", () =>
    page.locator("#reset-button").click(),
  );
  await expect(page.locator("#step-value")).toHaveText("0");
  await expectFilterSurface(page, disabledPair);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await openVisualFilters(page);
  await expectFilterSurface(page, INITIAL_DISABLED_FILTERS, false);
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("live_shared_obs_agent_pov");

  await expectReplayAgentTargetFilter(
    page,
    apiRequests,
    requiredService(noSharedReplay, "NoShared replay viewer"),
    "replay_no_shared_obs_agent_pov",
    "Replay NoShared",
  );
  await expectReplayAgentTargetFilter(
    page,
    apiRequests,
    requiredService(sharedReplay, "Shared replay viewer"),
    "replay_shared_obs_agent_pov",
    "Replay Shared",
  );

  const replay = requiredService(sharedReplay, "Shared replay viewer");
  await openInstalled(page, replay.url, {
    viewerMode: "replay",
    audience: "researcher",
    presentationKind: "replay_oracle",
  });
  await expectReplayFrameIndex(page, 1);
  await openVisualFilters(page);
  await expectFreshDefaults(page, "#replay-ranges-button");
  // Cooldown Effects now starts on, so a fresh page already paints cooldowns.
  await expect(page.locator(COOLDOWN_RING_PAINT)).not.toHaveCount(0);
  await ensureRangesOn(page, "replay");
  await clickFilterControl(page, "#enable-all-visual-filters-button");
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    false,
    "replay target overlay stays optional",
  );
  await expectFilterSurface(page, ["target_selection_visuals"]);

  const replayScience = await scientificSignature(page, true);
  const replayRanges = await rangeSignature(page, "#replay-ranges-button");
  const replayAuras = await auraMarkup(page);
  expect(replayAuras).not.toBe("");

  const playPause = page.locator("#replay-play-pause-button");
  const replayAuraInput = page.locator(
    `${FILTER_INPUT}[data-visual-filter-id="aura_fields"]`,
  );
  await expectLocalOnly(
    page,
    apiRequests,
    async () => {
      await playPause.click();
      await expect(playPause).toHaveText("Pause");
      await expect(playPause).toHaveAttribute("aria-pressed", "true");
      await replayAuraInput.evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) {
          throw new TypeError("Aura filter is unavailable.");
        }
        input.click();
      });
    },
    {
      label: "replay auto-pause aura toggle",
      delayMs: REPLAY_AUTOPLAY_CADENCE_MS.normal + 125,
    },
  );
  await expect(playPause).toHaveText("Play");
  await expect(playPause).toHaveAttribute("aria-pressed", "false");
  await expectReplayFrameIndex(page, 1);
  await expectAuraAbsent(page);
  expect(await scientificSignature(page, true)).toEqual(replayScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(replayRanges);

  await setFilter(page, apiRequests, "aura_fields", true, "replay aura on");
  expect(await auraMarkup(page)).toBe(replayAuras);
  expect(await scientificSignature(page, true)).toEqual(replayScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(replayRanges);

  const ultimateBefore = await ultimateInventory(page);
  expect(ultimateBefore.length).toBeGreaterThan(0);
  expect(ultimateBefore.every((row) => row.component === "ultimate")).toBe(true);
  const ultimateIds = ultimateBefore.map(({ eventId }) => {
    if (typeof eventId !== "string" || eventId.length === 0) {
      throw new Error("Real replay ultimate has no event ID.");
    }
    return eventId;
  });
  const ultimateOwner = page
    .locator(
      '#battlefield .combat-effect--activation[data-component="ultimate"][data-tooltip-owner]',
    )
    .first();
  await ultimateOwner.dispatchEvent("focusin");
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "activation",
  );
  await expect(page.locator("#visual-tooltip-title")).not.toHaveText("");

  const ultimateInput = page.locator(
    `${FILTER_INPUT}[data-visual-filter-id="ultimate_ability_effects"]`,
  );
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      ultimateInput.evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) {
          throw new TypeError("Ultimate filter is unavailable.");
        }
        input.checked = false;
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }),
    { label: "real replay ultimate off" },
  );
  await expect(ultimateInput).not.toBeChecked();
  await expect(page.locator("#visual-tooltip")).toBeHidden();
  await expectUltimateAbsent(page, ultimateIds);
  expect(await scientificSignature(page, true)).toEqual(replayScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(replayRanges);

  await setFilter(
    page,
    apiRequests,
    "ultimate_ability_effects",
    true,
    "real replay ultimate on",
  );
  expect(await ultimateInventory(page)).toEqual(ultimateBefore);
  expect(await scientificSignature(page, true)).toEqual(replayScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(replayRanges);

  await expectCooldownToggle(page, apiRequests, true, "Replay Oracle");

  // Red Zone Floors off removes only the tint windows from the map layer and
  // changes only its own paint-key flag; scientific data and ranges stay the same.
  const replayMapLayer = page.locator('#battlefield [data-layer="map"]');
  const mapWithoutTint = () =>
    replayMapLayer.evaluate((layer) => {
      const copy = layer.cloneNode(true);
      if (!(copy instanceof Element)) {
        throw new TypeError("Map layer is unavailable.");
      }
      for (const tint of copy.querySelectorAll(".red-zone-floor-clip")) {
        tint.remove();
      }
      return copy.innerHTML;
    });
  const replayMap = await replayMapLayer.innerHTML();
  const replayMapWithoutTint = await mapWithoutTint();
  await setFilter(
    page,
    apiRequests,
    "red_zone_floors",
    false,
    "replay Red Zone Floors off",
  );
  await expectFilterSurface(page, ["target_selection_visuals", "red_zone_floors"]);
  await expect(page.locator(RED_ZONE_FLOOR_PAINT)).toHaveCount(0);
  expect(await replayMapLayer.innerHTML()).toBe(replayMapWithoutTint);
  expect(await choreographyPaintKeys(page)).toEqual(
    Array(3).fill(expectedPaintKey(["target_selection_visuals", "red_zone_floors"])),
  );
  expect(await scientificSignature(page, true)).toEqual(replayScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(replayRanges);
  await setFilter(
    page,
    apiRequests,
    "red_zone_floors",
    true,
    "replay Red Zone Floors on",
  );
  expect(await replayMapLayer.innerHTML()).toBe(replayMap);
  expect(await choreographyPaintKeys(page)).toEqual(
    Array(3).fill(expectedPaintKey(["target_selection_visuals"])),
  );
  expect(await scientificSignature(page, true)).toEqual(replayScience);
  await expectTooltipCleared(page);

  await setFilter(
    page,
    apiRequests,
    "aura_fields",
    false,
    "replay durable persistence",
  );
  await setFilter(
    page,
    apiRequests,
    "ultimate_ability_effects",
    false,
    "replay transient persistence",
  );
  const replayDisabled = [...disabledPair, "target_selection_visuals"];
  await expectFilterSurface(page, replayDisabled);

  await performCommand(page, "/api/replay/command", () =>
    page.locator('#view-switch button[data-view-mode="pov"]').click(),
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("replay_shared_obs_agent_pov");
  await expectFilterSurface(page, replayDisabled, false);
  await expectCooldownToggle(page, apiRequests, true, "Replay Shared Agent POV");
  await expectLocalOnly(
    page,
    apiRequests,
    () => clickFilterControl(page, "#replay-ranges-button"),
    {
      label: "Enable Agent-local ranges for the recipient-switch proof",
    },
  );

  const replayAgentRangesButton = page.locator("#replay-ranges-button");
  await expectLocalOnly(page, apiRequests, () => replayAgentRangesButton.click(), {
    label: "Replay Agent ranges off before recipient switch",
  });
  await expect(replayAgentRangesButton).toHaveAttribute("aria-pressed", "false");
  const recipientSwitchCursor = structuredClone(
    JSON.parse(await authenticatedText(page, "/api/frame")).cursor,
  );
  await showReplayRoster(page);
  const nextReplayRecipient = page
    .locator('#roster .roster-primary-action:not([aria-pressed="true"])')
    .first();
  await expect(nextReplayRecipient).toBeEnabled();
  await performCommand(page, "/api/replay/command", () => nextReplayRecipient.click());
  expect(JSON.parse(await authenticatedText(page, "/api/frame")).cursor).toEqual(
    recipientSwitchCursor,
  );
  await expect(replayAgentRangesButton).toHaveAttribute("aria-pressed", "false");
  // Ranges off: only the Reach Rings switch's rings remain, all of them the
  // one champion in focus (the page's own inspection; the renderer spec checks
  // that it is the selected champion).
  const ringKeys = await page
    .locator('[data-layer="debug-range"] .range-ring')
    .evaluateAll((nodes) =>
      nodes.map((node) => node.getAttribute("data-presentation-key")),
    );
  expect(ringKeys.every((key) => key !== null)).toBe(true);
  expect(new Set(ringKeys).size).toBeLessThanOrEqual(1);
  await expectFilterSurface(page, replayDisabled, false);

  await performCommand(page, "/api/replay/command", () => seekWithSlider(page, 2));
  await expectReplayFrameIndex(page, 2);
  await expectFilterSurface(page, replayDisabled, false);
  await performCommand(page, "/api/replay/command", () => seekWithSlider(page, 1));
  await expectReplayFrameIndex(page, 1);
  await expectFilterSurface(page, replayDisabled, false);

  await performCommand(page, "/api/replay/command", () =>
    page.locator('#view-switch button[data-view-mode="researcher"]').click(),
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("replay_oracle");
  await expectFilterSurface(page, replayDisabled);

  await performCommand(page, "/api/replay/command", () =>
    page.locator('#view-switch button[data-view-mode="pov"]').click(),
  );
  await expect(replayAgentRangesButton).toHaveAttribute("aria-pressed", "false");
  await expectFilterSurface(page, replayDisabled, false);
  await performCommand(page, "/api/replay/command", () =>
    page.locator('#view-switch button[data-view-mode="researcher"]').click(),
  );
  await expect(replayAgentRangesButton).toHaveAttribute("aria-pressed", "true");
  await expectFilterSurface(page, replayDisabled);

  const bulkCursor = structuredClone(
    JSON.parse(await authenticatedText(page, "/api/frame")).cursor,
  );
  const bulkScience = await bulkScientificSignature(page, true);
  const allOffRanges = await rangeSignature(page, "#replay-ranges-button");
  const replayDisableMark = apiRequests.length;
  await disableAllFilters(page);
  expect(
    apiRequests.slice(replayDisableMark).filter(({ method }) => method === "POST"),
    "Replay Disable All range request count",
  ).toEqual([{ method: "POST", path: "/api/replay/command" }]);
  await expectFilterSurface(page, FILTER_IDS, false);
  await expectAllPaintAbsentWithoutDwell(page);
  expect(JSON.parse(await authenticatedText(page, "/api/frame")).cursor).toEqual(
    bulkCursor,
  );
  expect(await bulkScientificSignature(page, true)).toEqual(bulkScience);
  await expect(page.locator("#replay-ranges-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator("#disable-all-visual-filters-button").evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Disable All is unavailable.");
        }
        button.click();
      }),
    { label: "disabled Disable All idempotency" },
  );
  await expectFilterSurface(page, FILTER_IDS, false);

  const replayEnableMark = apiRequests.length;
  await performCommand(page, "/api/replay/command", () =>
    clickFilterControl(page, "#enable-all-visual-filters-button"),
  );
  expect(
    apiRequests.slice(replayEnableMark).filter(({ method }) => method === "POST"),
    "Replay Enable All range request count",
  ).toEqual([{ method: "POST", path: "/api/replay/command" }]);
  await expectFilterSurface(page);
  expect(JSON.parse(await authenticatedText(page, "/api/frame")).cursor).toEqual(
    bulkCursor,
  );
  expect(await bulkScientificSignature(page, true)).toEqual(bulkScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(allOffRanges);
  expect(await auraMarkup(page)).toBe(replayAuras);
  expect(await ultimateInventory(page)).toEqual(ultimateBefore);
  const enabledChoreographyRoots = page.locator(CHOREOGRAPHY_ROOTS);
  await expect(enabledChoreographyRoots).toHaveCount(3);
  expect(
    await enabledChoreographyRoots.evaluateAll((roots) =>
      roots.map((root) => root.getAttribute("data-paint-key")),
    ),
  ).toEqual(Array(3).fill(`visual-filters-v4:${"1".repeat(FILTER_IDS.length)}`));
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator("#enable-all-visual-filters-button").evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Enable All is unavailable.");
        }
        button.click();
      }),
    { label: "post-enable Enable All idempotency" },
  );
  await expectFilterSurface(page);

  await setFilter(page, apiRequests, "aura_fields", false, "pre-reload replay aura");
  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expectReplayFrameIndex(page, 1);
  await openVisualFilters(page);
  await expectFilterSurface(page, INITIAL_DISABLED_FILTERS);
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("replay_oracle");

  expect(browserErrors).toEqual([]);
});

test("real maximum-status replay renders +2 without losing owner semantics", async ({
  page,
}) => {
  const replay = await startReplayViewer({
    scenario: "max_status_stack",
    includeStress: true,
    frameIndex: 1,
  });

  try {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(replay.url);
    await expect(page.locator("#connection-status")).toHaveText("Connected");
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );

    await clickFilterControl(page, "#enable-all-visual-filters-button");
    await expectFilterSurface(page);
    await expect(page.locator("#battlefield .modifier-overflow")).toHaveCount(0);
    await expect(page.locator("#battlefield .modifier-cell__value")).toHaveText([
      "×1.15",
      "×0.85",
    ]);

    const presentation = JSON.parse(
      await authenticatedText(page, "/api/presentation/frame"),
    );
    const ownerAgent = presentation.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "0",
    );
    if (!ownerAgent) {
      throw new Error("The maximum-status replay has no Agent ID 0 body.");
    }
    const ownerBody = page.locator(
      `#battlefield .agent[data-presentation-key="${ownerAgent.presentation_key}"]`,
    );
    if ((await ownerBody.getAttribute("data-selected")) !== "true") {
      await showReplayRoster(page);
      await performCommand(page, "/api/replay/command", () =>
        page
          .locator(
            `#roster .roster-primary-action[data-presentation-key="${ownerAgent.presentation_key}"]`,
          )
          .click(),
      );
    }
    await expect(ownerBody).toHaveAttribute("data-selected", "true");

    const overflow = page.locator("#battlefield .status-overflow").filter({
      has: page.locator(".status-overflow__label", { hasText: /^\+2$/u }),
    });
    await expect(overflow).toHaveCount(1);
    await expect(overflow.locator(".status-overflow__label")).toHaveText("+2");
    expect(await overflow.locator(".status-overflow__label").innerHTML()).toBe("+2");

    const owner = "Agent ID 0";
    await expect(overflow).toHaveAttribute("data-zone", "status-overflow");
    await expect(overflow).toHaveAttribute("data-hidden-count", "2");
    await expect(overflow).toHaveAttribute("data-owner-label", owner);
    await expect(overflow).toHaveAttribute("aria-label", new RegExp(owner, "u"));
    await expect(overflow).toHaveAttribute("data-presentation-key", /^oracle_/u);

    await overflow.dispatchEvent("focusin");
    await expect(page.locator("#visual-tooltip")).toBeVisible();
    // The overflow's name comes from the one symbol list ("More Statuses").
    await expect(page.locator("#visual-tooltip-title")).toHaveText("2 More Statuses");
    // The tip lists the statuses the row left out, one row each; the owner is
    // named on the overflow itself (data-owner-label and aria-label above).
    await expect(
      page.locator("#visual-tooltip .semantic-explanation__label"),
    ).toHaveText(["Status 1", "Status 2"]);

    await page.setViewportSize({ width: 480, height: 360 });
    await page.evaluate(async () => {
      await document.fonts.ready;
      await new Promise((resolve) => {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            requestAnimationFrame(resolve);
          });
        });
      });
    });
    const compactStatusOverflows = page.locator(
      [
        "#battlefield .status-overflow",
        '#battlefield .required-dock-fallback[data-kind="status"]',
      ].join(", "),
    );
    await expect(compactStatusOverflows).not.toHaveCount(0);
    const compactRows = await compactStatusOverflows.evaluateAll((overflows) =>
      overflows.map((overflow) => ({
        visibleText: overflow.textContent,
        value: overflow.querySelector(".status-overflow__label")?.textContent,
        ownerNodes: overflow.querySelectorAll(
          ".status-overflow__owner, .required-dock-fallback__owner",
        ).length,
        ownerLabel: overflow.getAttribute("data-owner-label"),
        ariaLabel: overflow.getAttribute("aria-label"),
        presentationKey: overflow.getAttribute("data-presentation-key"),
      })),
    );
    for (const row of compactRows) {
      expect(row.value).toMatch(/^\+(?:[2-9]|\d{2,})$/u);
      expect(row.visibleText).toBe(row.value);
      expect(row.ownerNodes).toBe(0);
      expect(row.ownerLabel).toMatch(/^Agent ID /u);
      expect(row.ariaLabel).toContain(row.ownerLabel);
      expect(row.presentationKey).toMatch(/^oracle_/u);
    }

    await expect(
      page.locator('#battlefield .required-dock-fallback[data-kind="cooldown"]'),
    ).toHaveCount(0);
  } finally {
    await stopDebugger(replay.process);
  }
});

test("amended regular and stress scenarios retain Oracle-Agent presentation continuity", async ({
  page,
}) => {
  const browserErrors = captureBrowserErrors(page);
  const scenarios = [
    { name: "ultimate_showcase", includeStress: false },
    { name: "max_status_stack", includeStress: true },
    { name: "lifecycle_density", includeStress: true },
  ];
  const playingChoreography = [
    "#battlefield .combat-choreography[data-state=playing]",
    "#battlefield .combat-choreography-connectors[data-state=playing]",
    "#battlefield .combat-choreography-routes[data-state=playing]",
  ].join(", ");

  /** @param {any} value
   * @returns {any} */
  const atAuthorizedWirePrecision = (value) => {
    if (typeof value === "number" && Number.isFinite(value)) {
      return Math.fround(value);
    }
    if (Array.isArray(value)) {
      return value.map(atAuthorizedWirePrecision);
    }
    if (value !== null && typeof value === "object") {
      return Object.fromEntries(
        Object.entries(value).map(([key, item]) => [
          key,
          atAuthorizedWirePrecision(item),
        ]),
      );
    }
    return value;
  };

  /** @param {Record<string, any>} agent */
  const publicAgentFacts = (agent) => {
    const {
      presentation_key: _presentationKey,
      relation: _relation,
      statuses,
      ...facts
    } = agent;
    return atAuthorizedWirePrecision({
      ...facts,
      statuses: statuses.map((/** @type {Record<string, any>} */ status) => {
        const { direct_sources: _directSources, ...publicStatus } = status;
        return publicStatus;
      }),
    });
  };

  /** @param {"researcher" | "agent_pov"} audience
   * @param {number} frameIndex
   * @param {Record<string, any> | null} oracle */
  const inspectInstalledFrame = async (audience, frameIndex, oracle) => {
    await expect(page.locator("#connection-status")).toHaveText("Connected");
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
    await expect(page.locator("html")).toHaveAttribute(
      "data-submission-blocked",
      "false",
    );
    await expect(page.locator("#battlefield-empty")).toBeHidden();
    await expect(page.locator(playingChoreography)).toHaveCount(0);
    const presentation = JSON.parse(
      await authenticatedText(page, "/api/presentation/frame"),
    );
    expect(presentation.source.source_frame_index).toBe(frameIndex);
    expect(presentation.presentation_kind).toBe(
      audience === "researcher" ? "replay_oracle" : "replay_no_shared_obs_agent_pov",
    );
    const scene =
      audience === "researcher"
        ? presentation.current_endpoint.scene
        : presentation.current_endpoint.parts.scene;
    const authorizedKeys = scene.agents
      .map((/** @type {Record<string, any>} */ agent) => agent.presentation_key)
      .sort();
    const bodyKeys = await page
      .locator("#battlefield .agent[data-presentation-key]")
      .evaluateAll((agents) =>
        agents.map((agent) => agent.getAttribute("data-presentation-key")).sort(),
      );
    expect(bodyKeys).toEqual(authorizedKeys);
    await expect(
      page.locator(
        "#battlefield [data-global-slot], #battlefield [data-team-local-slot]",
      ),
    ).toHaveCount(0);
    // Replay Viewer notices show under the readout (owner, 3 Oct 2026, B9).
    const notice = page.locator(
      "#notice:not([hidden]), #rune-scrubber-notice:not([hidden])",
    );
    for (const text of await notice.allInnerTexts()) {
      expect(text).not.toMatch(
        /could not process|safe fault|reconnect|no authorized battlefield scene/iu,
      );
    }

    if (audience === "agent_pov") {
      if (oracle === null) {
        throw new Error("Agent scenario smoke requires its Oracle frame.");
      }
      const oracleByPublicId = new Map(
        oracle.current_endpoint.scene.agents.map(
          (/** @type {Record<string, any>} */ agent) => [agent.public_agent_id, agent],
        ),
      );
      for (const agent of scene.agents) {
        const oracleAgent = oracleByPublicId.get(agent.public_agent_id);
        expect(oracleAgent).toBeDefined();
        expect(publicAgentFacts(agent)).toEqual(publicAgentFacts(oracleAgent));
      }
      const battlefieldMarkup = await page.locator("#battlefield").innerHTML();
      for (const oracleAgent of oracle.current_endpoint.scene.agents) {
        expect(battlefieldMarkup).not.toContain(oracleAgent.presentation_key);
      }
    }
    return presentation;
  };

  for (const scenario of scenarios) {
    const replay = await startReplayViewer({
      scenario: scenario.name,
      includeStress: scenario.includeStress,
    });
    try {
      await openInstalled(page, replay.url, {
        viewerMode: "replay",
        audience: "researcher",
        presentationKind: "replay_oracle",
      });
      await page.locator("#replay-playback-rate").selectOption("2");
      const timeline = JSON.parse(
        await authenticatedText(page, "/api/replay/timeline"),
      );
      const oracleFrames = [];
      oracleFrames.push(await inspectInstalledFrame("researcher", 0, null));
      for (
        let frameIndex = 1;
        frameIndex <= timeline.final_frame_index;
        frameIndex += 1
      ) {
        await performCommand(page, "/api/replay/command", () =>
          page.locator("#replay-next-button").click(),
        );
        await expectReplayFrameIndex(page, frameIndex);
        oracleFrames.push(await inspectInstalledFrame("researcher", frameIndex, null));
      }

      await performCommand(page, "/api/replay/command", () => seekWithSlider(page, 0));
      await expectReplayFrameIndex(page, 0);
      await performCommand(page, "/api/replay/command", () =>
        page.locator('#view-switch button[data-view-mode="pov"]').click(),
      );
      await inspectInstalledFrame("agent_pov", 0, oracleFrames[0]);
      for (
        let frameIndex = 1;
        frameIndex <= timeline.final_frame_index;
        frameIndex += 1
      ) {
        await performCommand(page, "/api/replay/command", () =>
          page.locator("#replay-next-button").click(),
        );
        await expectReplayFrameIndex(page, frameIndex);
        await inspectInstalledFrame("agent_pov", frameIndex, oracleFrames[frameIndex]);
      }
      await page.goto("about:blank");
    } finally {
      await stopDebugger(replay.process);
    }
  }
  expect(browserErrors).toEqual([]);
});
