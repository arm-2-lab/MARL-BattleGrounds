/**
 * @file Check API-configured Team B GAMMA through one accepted step, recording,
 * and separate Replay Viewer inspection. Preserve exact policy and replay identities:
 * the shared scorebug names Team B "tdm-gamma" and Team A "You" in live play or the
 * recorded "Manual" in the Replay Viewer, and the match summary keeps every policy
 * reference. In live play, a click on GAMMA's Priest sends one control request
 * and the deck shows Watch Only: the Priest's class, Agent ID and "Played by
 * tdm-gamma", with every Allowed Now pick disabled. Lock In then accepts one
 * step, and the Recording pill's popover shows the kept tick and saves the
 * replay with its Finish and Save.
 * Private controller-menu behavior is checked in the local editor test suite.
 */
import { expect, test } from "@playwright/test";

import { closeRecordingPanel, openRecordingPanel } from "./support/live-debugger.js";
import {
  readJsonArtifact,
  startRecordingDebugger,
  stopRecordingDebugger,
} from "./support/recording-handoff.js";
import { startReplayViewer, stopDebugger } from "./support/replay-viewer.js";

/** @type {Awaited<ReturnType<typeof startRecordingDebugger>> | null} */
let recording = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let replayViewer = null;

test.afterEach(async () => {
  const viewer = replayViewer;
  const started = recording;
  replayViewer = null;
  recording = null;
  /** @type {unknown[]} */
  const cleanupErrors = [];
  try {
    // The Replay Viewer reads the saved file, so it stops before the recording
    // debugger removes that file's folder.
    await stopDebugger(viewer?.process ?? null);
  } catch (error) {
    cleanupErrors.push(error);
  }
  try {
    await stopRecordingDebugger(started);
  } catch (error) {
    cleanupErrors.push(error);
  }
  if (cleanupErrors.length > 0) {
    throw new AggregateError(cleanupErrors, "TDM-GAMMA E2E cleanup failed.");
  }
});

// GAMMA's public name: its live slot label and its recorded controller name.
const GAMMA_LABEL = "tdm-gamma";
const GAMMA_CONFIGURATION = Object.freeze({
  team_a_controller: "manual",
  team_b_controller: "tdm_gamma",
  execution_information_mode: "shared_obs",
});

/** @param {import("@playwright/test").Page} page */
function collectBrowserErrors(page) {
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

/** @param {import("@playwright/test").Page} page
 * @param {"/api/frame" | "/api/presentation/frame"} path
 * @returns {Promise<Record<string, any>>} */
function authenticatedGet(page, path) {
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
    return response.json();
  }, path);
}

/** @param {import("@playwright/test").Page} page
 * @param {() => Promise<unknown>} activate */
async function applyLiveCommand(page, activate) {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await activate();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  return response;
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} teamAName The Manual team's name: "You" in live play, the
 *   recorded "Manual" in the Replay Viewer.
 */
async function expectGammaScoreboard(page, teamAName) {
  // The shared scorebug (#scorebug) is the one score display in every client;
  // its controller lines name each team's controllers (src/controller-names.js).
  // The policy references stay in the match summary, checked below. Live play
  // writes "You" as plain text (never cut); a name is a span.controller-name.
  await expect(page.locator("#ctl-b .controller-name")).toHaveText(GAMMA_LABEL);
  await expect(page.locator("#ctl-a")).toHaveText(teamAName);
  const presentation = await authenticatedGet(page, "/api/presentation/frame");
  const [teamASummary, teamBSummary] = presentation.match_summary.teams;
  expect(teamASummary).toMatchObject({ team_id: 1, display_name: "Manual" });
  expect(teamBSummary).toMatchObject({ team_id: 2, display_name: GAMMA_LABEL });
  expect(teamBSummary.policy_ids.length).toBeGreaterThan(0);
  for (const policyId of teamBSummary.policy_ids) {
    expect(policyId).toMatch(/^debugger-action-source:tdm_gamma:slot:\d+$/u);
  }
}

/** @param {Record<string, any>} context */
function expectGammaContext(context) {
  const keys = Object.fromEntries(
    context.aggregation_keys.map((/** @type {{name: string, value: string}} */ row) => [
      row.name,
      row.value,
    ]),
  );
  expect(keys).toMatchObject({
    team_a_controller: "manual",
    team_b_controller: "tdm_gamma",
    information_regime: "shared_obs",
    pressure_protocol: "reactive-team-deathmatch-gamma-controller@2",
  });
  expect(keys.pressure_protocol_digest).toMatch(/^[0-9a-f]{64}$/u);
  const teamBySlot = new Map(
    context.roster.map(
      (/** @type {{global_slot: number, configured_team_id: number}} */ row) => [
        row.global_slot,
        row.configured_team_id,
      ],
    ),
  );
  const assigned = context.policy_assignments.filter(
    (/** @type {Record<string, any>} */ row) => typeof row.policy_kind === "string",
  );
  const teamBRows = assigned.filter(
    (/** @type {Record<string, any>} */ row) => teamBySlot.get(row.global_slot) === 2,
  );
  const teamARows = assigned.filter(
    (/** @type {Record<string, any>} */ row) => teamBySlot.get(row.global_slot) === 1,
  );
  expect(teamBRows.length).toBeGreaterThan(0);
  expect(teamARows.length).toBeGreaterThan(0);
  for (const row of teamBRows) {
    expect(row).toMatchObject({
      policy_kind: "tdm_gamma",
      policy_id: `debugger-action-source:tdm_gamma:slot:${row.global_slot}`,
      algorithm_id: "reactive-team-deathmatch-gamma-controller",
      execution_mode: "deterministic",
      policy_content_digest: keys.pressure_protocol_digest,
    });
  }
  for (const row of teamARows) {
    expect(row).toMatchObject({ policy_kind: "manual" });
  }
}

test("Team B GAMMA records one real step and reopens with its identity", async ({
  page,
  context,
}) => {
  const started = await startRecordingDebugger({
    stem: "tdm-gamma",
    extraArgs: ["--team-b-controller", "tdm_gamma"],
  });
  recording = started;
  const errors = collectBrowserErrors(page);
  await page.goto(started.url);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-product-kind",
    "combat_debugger",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "recording",
  );
  const installed = await authenticatedGet(page, "/api/frame");
  expect(installed.combat_configuration).toMatchObject(GAMMA_CONFIGURATION);
  expect(installed.recording).toMatchObject({
    lifecycle: "recording",
    captured_transition_count: 0,
  });
  await expect(page.locator("#step-value")).toHaveText("0");
  await expectGammaScoreboard(page, "You");

  // The BattleClient has no Roster panel: a click on a champion's body sends
  // the same control request the old Roster row sent.
  const priestB = page.locator(
    '#battlefield .agent[data-team="team-b"][data-class="priest"]',
  );
  const focusResponse = await applyLiveCommand(page, () => priestB.click());
  expect(focusResponse.request().postDataJSON().command).toMatchObject({
    command_type: "roster_selection",
    role: "control",
    global_slot: expect.any(Number),
  });
  await expect(page.locator("#command-deck")).toHaveAttribute(
    "data-policy-controller-read-only",
    "true",
  );
  // GAMMA's champion is in view but not yours: the deck shows Watch Only with
  // its class, ID and the tdm-gamma label, and no target can be picked.
  const watchOnly = page.locator("#watch-only");
  await expect(watchOnly).toBeVisible();
  await expect(watchOnly.locator(".eyebrow")).toHaveText("Watch Only");
  await expect(page.locator("#wo-name")).toHaveText(/^Priest · /u);
  await expect(page.locator("#wo-meta")).toHaveText(
    /^Agent ID \d+ · Played by tdm-gamma$/u,
  );
  await expect(page.locator("#wo-meta .controller-name")).toHaveText(GAMMA_LABEL);
  await expect(watchOnly).toContainText(
    "You can look at this champion, but not command it. Policies never see your orders.",
  );
  await expect(page.locator("#command-target-select")).toBeDisabled();
  const picks = page.locator("#command-target-select button.pick");
  await expect(picks.first()).toBeVisible();
  expect(
    await picks.evaluateAll((buttons) =>
      buttons.every((button) => button instanceof HTMLButtonElement && button.disabled),
    ),
  ).toBe(true);

  await expect(page.locator("#submit-turn-button")).toBeEnabled();
  await applyLiveCommand(page, () => page.locator("#submit-turn-button").click());
  await expect(page.locator("#step-value")).toHaveText("1");
  await expect(page.locator("#recording-progress")).toHaveText(
    /^1 \/ [1-9]\d* transitions$/u,
  );
  const stepped = await authenticatedGet(page, "/api/frame");
  expect(stepped.combat_configuration).toMatchObject(GAMMA_CONFIGURATION);
  expect(stepped.recording).toMatchObject({
    lifecycle: "recording",
    captured_transition_count: 1,
  });

  // The Recording pill opens the Replay Recording popover; its Finish and Save
  // saves the replay.
  await openRecordingPanel(page);
  await expect(page.locator("#pop-state")).toHaveText("1 Tick Kept Since Tick 0");
  const finishResponse = await applyLiveCommand(page, () =>
    page.locator("#pop-finish").click(),
  );
  await closeRecordingPanel(page);
  expect(finishResponse.request().postDataJSON().command).toMatchObject({
    command_type: "finish_and_review",
  });
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "saved",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-product-kind",
    "combat_debugger",
  );
  await expectGammaScoreboard(page, "You");

  const saved = await readJsonArtifact(started.replayPath);
  expect(saved.value).toMatchObject({
    schema_id: "marl_battlegrounds.evaluation.replay_artifact",
    schema_version: 4,
    header: { recorded_transition_count: 1 },
  });
  expectGammaContext(saved.value.header.context);

  const viewer = await startReplayViewer({ replayPath: started.replayPath });
  replayViewer = viewer;
  const reviewPage = await context.newPage();
  const reviewErrors = collectBrowserErrors(reviewPage);
  await reviewPage.goto(viewer.url);
  await expect(reviewPage).toHaveTitle("MARL-BattleGrounds Replay Viewer");
  await expect(reviewPage.locator("#connection-status")).toHaveText("Connected");
  await expect(reviewPage.locator("html")).toHaveAttribute(
    "data-product-kind",
    "replay_viewer",
  );
  await expectGammaScoreboard(reviewPage, "Manual");
  expect(reviewErrors).toEqual([]);
  await reviewPage.close();
  expect(errors).toEqual([]);
});
