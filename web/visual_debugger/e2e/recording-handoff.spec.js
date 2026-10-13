/**
 * @file Check live recording finish/discard/exit, saved interrupted prefixes and
 * multi-tab save/restart and separate replay viewing. Saved live recordings are replay V4 files (they record
 * the Red Zone rule) without a metric sidecar.
 *
 * The BattleClient shows the recording in its top bar. The Recording pill
 * (#rec-pill-text) and its popover's state line (#pop-state) name the tick and
 * the kept ticks, then Saved or Save Failed. The top bar's save button reads
 * Finish and Save, Saving Replay… while a save waits, then Open Replay or Retry
 * Save. The Recording popover (#rec-pop) holds Open Replay (disabled until a
 * save exists), Retry Save, Save As and the fallback link. Restart's Save This
 * Game First? dialog focuses Save. Live notices show on the deck's message line
 * (#msg); the replay-link race reads the page's notice (#notice), because the
 * new match's first line replaces that warning on #msg. A lost Finish answer
 * reads Disconnected; Reconnect then shows the saved replay without a second
 * save. A new Replay Viewer opens in Oracle View at tick 0 whatever the live
 * view, moves no focus, and its Agent POV shows the live Agent POV champion
 * with that team's fog. After a failed save the Recording popover says why in
 * plain words ("Reason: The save folder or file name could not be used, ..."),
 * never a path, and the line is gone once Save As saves; the page stays Connected: the View
 * buttons, Visual Filters (with Ranges) and Exit stay in the Tab order, while
 * the arena, the deck's keys, the Allowed Now picks and Lock In do not. Agent
 * POV closeout lists every champion by team, with unseen enemies Out of Sight.
 *
 * Exit must end the server with exit code 0 and no
 * signal, checked strictly by expectServerShutdown. A server that dies before its
 * shutdown response must reject that response promptly through the browser's real
 * requestfailed event, be reported as a failure naming its SIGKILL outcome (never as
 * a shutdown), and leave no process holding its owner token. That test pauses the
 * uv launcher with SIGSTOP while it kills the Python server, so the browser's
 * requestfailed always arrives before Node can see the launcher exit. It waits for
 * the browser's own requestfailed event, so a helper that ignored that event would
 * fail the test instead of leaving it waiting. Each signal rechecks the server
 * owner token and process start time through the shared lifecycle helper.
 */
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

import { expect, test } from "@playwright/test";
import { finishControllerClock } from "./support/choreography.js";
import {
  captureServerFailure,
  closeRecordingPanel,
  expectServerShutdown,
  findOwnedProcessIds,
  killOwnedProcesses,
  openRecordingPanel,
  ServerFailureError,
  serverOwnerToken,
  signalOwnedProcess,
} from "./support/live-debugger.js";
import {
  createReplayTargetRace,
  metricReportPathForReplay,
  readJsonArtifact,
  startRecordingDebugger,
  stopRecordingDebugger,
} from "./support/recording-handoff.js";
import { expectVisibleInteractiveHelpInventory } from "./support/visual-regression.js";

test.describe.configure({ mode: "serial" });

/** @type {Awaited<ReturnType<typeof startRecordingDebugger>> | null} */
let recording = null;

/** @type {WeakMap<import("@playwright/test").Page, string[]>} */
const browserErrors = new WeakMap();

const RECORDING_STATUS_KEYS = Object.freeze([
  "captured_transition_count",
  "completion_reason",
  "completion_state",
  "discard_available",
  "expected_transition_count",
  "finish_available",
  "lifecycle",
  "persistence_error_code",
  "restart_fenced",
  "retry_available",
  "review_available",
  "save_as_available",
  "schema_version",
]);

test.beforeEach(async () => {
  recording = await startRecordingDebugger();
});

test.afterEach(async () => {
  const started = recording;
  recording = null;
  await stopRecordingDebugger(started);
});

function requiredRecording() {
  if (!recording) {
    throw new Error("The recording debugger did not start.");
  }
  return recording;
}

/** @param {import("@playwright/test").Page} page */
function collectBrowserErrors(page) {
  if (browserErrors.has(page)) {
    return;
  }
  /** @type {string[]} */
  const errors = [];
  browserErrors.set(page, errors);
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
}

/** @param {import("@playwright/test").Page} page
 * @param {string} url */
async function openRecording(page, url) {
  collectBrowserErrors(page);
  await page.goto(url);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page).toHaveTitle("MARL-BattleGrounds BattleClient");
  await expect(page.locator("html")).toHaveAttribute(
    "data-product-kind",
    "combat_debugger",
  );
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "recording",
  );
  await expect(page.locator("#recording-badge")).toBeVisible();
  await expect(page.locator("#rec-pill-text")).toHaveText("Recording · Tick 0");
  await expect(page.locator("#pop-state")).toHaveText("0 Ticks Kept Since Tick 0");
  expect(page.url()).not.toContain("token=");
}

/** @param {import("@playwright/test").Page} page
 * @param {string} text */
async function expectLiveMessage(page, text) {
  // Live play shows notices on the deck's message line, never in the top bar.
  await expect(page.locator("#msg")).toBeVisible();
  await expect(page.locator("#msg")).toContainText(text);
}

/** @param {import("@playwright/test").Page} page
 * @param {"pov" | "researcher"} mode */
async function chooseView(page, mode) {
  // The top bar's View buttons drive the hidden View select.
  const button = page.locator(`#view-switch button[data-view-mode="${mode}"]`);
  await button.click();
  await expect(button).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#view-select")).toHaveValue(mode);
}

/** @param {import("@playwright/test").Page} page */
function expectNoBrowserErrors(page) {
  expect(browserErrors.get(page) ?? []).toEqual([]);
}

/** @param {import("@playwright/test").Request} request */
function isLiveCommandPost(request) {
  return (
    request.method() === "POST" && new URL(request.url()).pathname === "/api/command"
  );
}

/** @param {string} token @param {number} launcherPid
 * @returns {Promise<number[]>} */
async function killServerWithLauncherPaused(token, launcherPid) {
  // Pause the uv launcher so Node cannot see it exit yet, then kill every other
  // process that holds the token, which includes the Python server.
  if (!signalOwnedProcess(launcherPid, [token], "SIGSTOP")) {
    throw new Error("The recording launcher is no longer owned by this test.");
  }
  /** @type {number[]} */
  const killed = [];
  for (;;) {
    const cleanup = killOwnedProcesses([token], undefined, [launcherPid]);
    killed.push(...cleanup.killedProcessIds);
    if (cleanup.errors.length > 0) {
      throw new AggregateError(cleanup.errors, "The recording server cleanup failed.");
    }
    if (cleanup.stillPresentProcessIds.length === 0) {
      return killed;
    }
    // The launcher stays paused while its signalled server children finish dying.
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

/** @param {import("@playwright/test").Page} page */
async function expectReplaySaving(page) {
  // While the save request waits, the top bar's save button says so and does nothing.
  const save = page.locator("#recording-finish-button");
  await expect(save).toBeVisible();
  await expect(save).toHaveText("Saving Replay…");
  await expect(save).toBeDisabled();
  await expect(save).not.toHaveAttribute("data-action");
  await expect(save).toHaveAttribute(
    "aria-description",
    "The replay is being saved. This button turns into Open Replay when it is done.",
  );
}

/** @param {import("@playwright/test").Page} page */
async function expectFinishAndSave(page) {
  const save = page.locator("#recording-finish-button");
  await expect(save).toBeVisible();
  await expect(save).toHaveText("Finish and Save");
  await expect(save).toHaveAttribute("data-action", "finish");
}

/** @param {import("@playwright/test").Page} page
 * @param {string} path */
async function authenticatedGet(page, path) {
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

/** @param {import("@playwright/test").Page} page */
function currentFrame(page) {
  return authenticatedGet(page, "/api/frame");
}

/** @param {import("@playwright/test").Page} page */
function currentTimeline(page) {
  return authenticatedGet(page, "/api/replay/timeline");
}

/** @param {Record<string, any>} frame
 * @param {number} count */
function expectExactWireRecording(frame, count) {
  expect(frame.recording).not.toBeNull();
  expect(Object.keys(frame.recording).sort()).toEqual(RECORDING_STATUS_KEYS);
  expect(frame.recording).toMatchObject({
    schema_version: 1,
    captured_transition_count: count,
    expected_transition_count: expect.any(Number),
  });
  expect(frame.recording.expected_transition_count).toBeGreaterThan(0);
  expect(frame.recording.captured_transition_count).toBe(frame.frame_index);
  for (const forbidden of [
    "path",
    "replay_path",
    "metric_report_path",
    "destination",
    "detail",
    "last_io_error_code",
  ]) {
    expect(frame.recording).not.toHaveProperty(forbidden);
    expect(frame).not.toHaveProperty(forbidden);
  }
}

/** @param {import("@playwright/test").Page} page
 * @param {number} count */
async function expectRecordingCount(page, count) {
  const kept = `${count} Tick${count === 1 ? "" : "s"} Kept`;
  await expect(page.locator("#pop-state")).toHaveText(
    new RegExp(`^${kept} Since Tick \\d+$`),
  );
  const frame = await currentFrame(page);
  expectExactWireRecording(frame, count);
  expect(frame.frame_index).toBe(count);
  // The pill names the current tick; the popover counts the kept ticks.
  const tick = frame.simulator_step_count;
  expect(Number.isInteger(tick)).toBe(true);
  await expect(page.locator("#rec-pill-text")).toHaveText(`Recording · Tick ${tick}`);
  await expect(page.locator("#pop-state")).toHaveText(
    `${kept} Since Tick ${tick - count}`,
  );
  return frame;
}

/** @param {import("@playwright/test").Page} page
 * @param {number} expectedCount */
async function captureNextTransition(page, expectedCount) {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await page.locator("#submit-turn-button").click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const frame = await expectRecordingCount(page, expectedCount);
  const activeChoreography = page.locator(
    '[data-layer="transient-events"] > .combat-choreography[data-state="active"]',
  );
  if ((await activeChoreography.count()) > 0) {
    await finishControllerClock(page, "cleanup");
    await expect(activeChoreography).toHaveCount(0);
  }
  return frame;
}

/** @param {import("@playwright/test").Page} page */
function captureOneTransition(page) {
  return captureNextTransition(page, 1);
}

/** @param {import("@playwright/test").Page} livePage */
async function openSavedReplay(livePage) {
  await expect(livePage.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "saved",
  );
  await expect(livePage.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  // Saved: the top bar's save button turns into Open Replay. The test opens the
  // Replay Viewer from the Recording popover's Open Replay.
  const save = livePage.locator("#recording-finish-button");
  await expect(save).toHaveText("Open Replay");
  await expect(save).toHaveAttribute("data-action", "open");
  await expect(save).toBeEnabled();
  await expect(livePage.locator("#rec-pill-text")).toHaveText("Saved");
  await openRecordingPanel(livePage);
  await expect(livePage.locator("#pop-state")).toHaveText(/^Saved · \d+ Ticks?$/u);
  const popup = livePage.waitForEvent("popup");
  await livePage.locator("#recording-review-button").click();
  const replayPage = await popup;
  await closeRecordingPanel(livePage);
  collectBrowserErrors(replayPage);
  await expect(replayPage.locator("#connection-status")).toHaveText("Connected");
  expect(new URL(replayPage.url()).origin).not.toBe(new URL(livePage.url()).origin);
  await expect(livePage.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  await expect(livePage.locator("#submit-turn-button")).toBeDisabled();
  await expect(livePage.locator("#reset-button")).toBeEnabled();
  return replayPage;
}

/** @param {Record<string, any>} frame
 * @param {Record<string, any>} timeline
 * @param {"researcher" | "pov"} audience
 * @param {"shared_obs" | "no_shared_obs"} informationMode */
function expectReplayAudienceFrame(
  frame,
  timeline,
  audience,
  informationMode = "shared_obs",
) {
  const researcher = audience === "researcher";
  const sharedPov = !researcher && informationMode === "shared_obs";
  expect(frame).toMatchObject({
    schema_version: 1,
    frame_kind: researcher
      ? "researcher_replay_viewer"
      : sharedPov
        ? "shared_obs_agent_pov_replay_viewer"
        : "actor_pov_replay_viewer",
    view_mode: researcher ? "researcher" : "pov",
    cursor: { frame_index: 0 },
    timeline_id: timeline.timeline_id,
  });
  if (researcher) {
    expect(frame).toMatchObject({
      incoming_transition_index: null,
      incoming_transition_id: null,
    });
    expect(timeline).toMatchObject({ timeline_kind: "researcher" });
    expect(timeline.rows[0].frame_id).toBe(frame.frame_id);
  } else if (sharedPov) {
    expect(frame).toMatchObject({ incoming_recipient_transition_id: null });
    expect(timeline).toMatchObject({
      timeline_kind: "shared_obs_agent_pov",
      artifact_summary: { public_agent_id: frame.public_agent_id },
    });
    expect(timeline.rows[0].recipient_frame_id).toBe(frame.recipient_frame_id);
  } else {
    expect(frame).toMatchObject({ incoming_pov_transition_id: null });
    expect(timeline).toMatchObject({
      timeline_kind: "actor_pov",
      pov_global_slot: frame.pov_global_slot,
      public_agent_id: frame.public_agent_id,
    });
    expect(timeline.rows[0].pov_frame_id).toBe(frame.pov_frame_id);
  }
}

/** @param {import("@playwright/test").Page} replayPage */
async function switchReplayToAgentPov(replayPage) {
  const response = replayPage.waitForResponse(
    (candidate) =>
      candidate.request().method() === "POST" &&
      new URL(candidate.url()).pathname === "/api/replay/command",
  );
  await chooseView(replayPage, "pov");
  expect((await response).status()).toBe(200);
  await expect(replayPage.locator("html")).toHaveAttribute(
    "data-audience",
    "agent_pov",
  );
  const [frame, timeline] = await Promise.all([
    currentFrame(replayPage),
    currentTimeline(replayPage),
  ]);
  expectReplayAudienceFrame(frame, timeline, "pov");
  return { frame, timeline };
}

/** @param {import("@playwright/test").Page} page */
async function expectSettledReplayHandoff(page) {
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay");
  // A new Viewer opens in Oracle View, whatever the live view was.
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");
  await expect(page).toHaveTitle("MARL-BattleGrounds Replay Viewer");
  await expect(page.locator("html")).toHaveAttribute(
    "data-product-kind",
    "replay_viewer",
  );
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#replay-timeline")).toBeVisible();
  // The first replay load moves no focus: replay keys listen on the whole page,
  // and a focused timeline would open its help card at rest.
  expect(await page.evaluate(() => document.activeElement === document.body)).toBe(
    true,
  );
  await expect(page.locator("#visual-tooltip")).toBeHidden();
  await expect(page.locator("#replay-frame-slider")).toHaveValue("0");
  await expect(page.locator("#replay-frame-position")).toContainText("Tick 0 /");
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("[data-live-only]:not([hidden])")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Submit joint turn" })).toHaveCount(0);
  await expect(
    page.locator(
      "#motion-pause-button, #motion-off-button, #motion-skip-button, #motion-status",
    ),
  ).toHaveCount(0);

  const [frame, timeline, boundary] = await Promise.all([
    currentFrame(page),
    currentTimeline(page),
    page.evaluate(() => {
      const liveRoots = [...document.querySelectorAll("[data-live-only]")];
      const focusable = [
        ...document.querySelectorAll(
          'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ),
      ].filter((element) => {
        if (!(element instanceof HTMLElement)) {
          return false;
        }
        const style = getComputedStyle(element);
        return (
          !element.hidden &&
          !element.hasAttribute("disabled") &&
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          element.getClientRects().length > 0
        );
      });
      return {
        everyLiveRootHidden: liveRoots.every(
          (element) => element instanceof HTMLElement && element.hidden,
        ),
        focusableInsideLiveRoot: focusable
          .filter((element) => element.closest("[data-live-only]") !== null)
          .map((element) => element.id || element.textContent?.trim()),
      };
    }),
  ]);
  expectReplayAudienceFrame(frame, timeline, "researcher");
  expect(boundary).toEqual({
    everyLiveRootHidden: true,
    focusableInsideLiveRoot: [],
  });
  const token = await page.evaluate(() =>
    window.sessionStorage.getItem("marl-battlegrounds.debugger-token"),
  );
  if (!token) {
    throw new Error("Debugger capability token is unavailable.");
  }
  const authoringResponse = await page.request.post(
    new URL("/api/dev/authoring/command", page.url()).href,
    {
      data: { command_type: "list", asset_kind: "all" },
      failOnStatusCode: false,
      headers: { "X-MARL-Debugger-Token": token },
      maxRedirects: 0,
    },
  );
  expect(authoringResponse.status()).toBe(404);
  const transientPresentation = await page
    .locator("#battlefield")
    .evaluate((battlefield) => ({
      activeRootCount: battlefield.querySelectorAll(
        '[data-layer="transient-events"] > .combat-choreography[data-state="active"]',
      ).length,
      ownedAnimationIds: battlefield
        .getAnimations({ subtree: true })
        .map(({ id }) => id)
        .filter((id) => id.startsWith("mbg:")),
    }));
  expect(transientPresentation).toEqual({
    activeRootCount: 0,
    ownedAnimationIds: [],
  });
  return { frame, timeline };
}

/** @param {string} replayPath
 * @param {string} metricPath
 * @param {number} transitionCount */
async function expectSavedArtifacts(replayPath, metricPath, transitionCount) {
  const replay = await readJsonArtifact(replayPath);
  await expect(readFile(metricPath)).rejects.toMatchObject({ code: "ENOENT" });
  expect(replay.bytes.byteLength).toBeGreaterThan(0);
  expect(replay.value).toMatchObject({
    schema_id: "marl_battlegrounds.evaluation.replay_artifact",
    schema_version: 4,
    header: {
      recorded_transition_count: transitionCount,
      recorded_frame_count: transitionCount + 1,
    },
    completion: { validated_transition_count: transitionCount },
  });
  expect(replay.value.frames).toHaveLength(transitionCount + 1);
  expect(replay.value.transitions).toHaveLength(transitionCount);
  expect(replay.value).not.toHaveProperty("metric_report_reference");
  expect(replay.value).not.toHaveProperty("processing_status");
  return { replay };
}

test("confirmed prefix discard and save keep live inspection, separate replay and fresh restart", async ({
  page,
}, testInfo) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  const stableUrl = page.url();
  const stableToken = await page.evaluate(() =>
    window.sessionStorage.getItem("marl-battlegrounds.debugger-token"),
  );
  expect(stableToken).toMatch(/^[A-Za-z0-9_-]+$/u);
  const initial = await currentFrame(page);
  expectExactWireRecording(initial, 0);
  expect(initial.recording).toMatchObject({
    schema_version: 1,
    lifecycle: "recording",
    captured_transition_count: 0,
    finish_available: true,
    restart_fenced: false,
    discard_available: false,
  });
  expect(initial.recording).not.toHaveProperty("path");
  await expectFinishAndSave(page);
  await expect(page.locator("#recording-finish-button")).toBeEnabled();
  // The Recording popover: Finish and Save works; Open Replay stays in view but
  // waits for a save; Retry Save and Save As appear only after a failed save.
  await openRecordingPanel(page);
  await expect(page.locator("#pop-finish")).toBeVisible();
  await expect(page.locator("#pop-finish")).toBeEnabled();
  await expect(page.locator("#recording-review-button")).toBeVisible();
  await expect(page.locator("#recording-review-button")).toBeDisabled();
  await expect(page.locator("#recording-retry-button")).toBeHidden();
  await expect(page.locator("#recording-save-as-control")).toBeHidden();
  await expect(page.locator("#recording-replay-link")).toBeHidden();
  await closeRecordingPanel(page);
  await expect(page.locator("body")).not.toContainText(started.outputDirectory);
  await expect(page.locator("body")).not.toContainText(started.replayPath);

  await chooseView(page, "pov");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  const pov = await currentFrame(page);
  expect(pov).toMatchObject({
    schema_version: 2,
    frame_kind: "shared_obs_agent_pov_live_debugger",
    view_mode: "pov",
  });
  expectExactWireRecording(pov, 0);
  expect(pov.recording).toEqual(initial.recording);
  await expect(page.locator("#recording-badge")).toBeVisible();
  await expect(page.locator("#rec-pill-text")).toHaveText("Recording · Tick 0");
  await expect(page.locator("body")).not.toContainText(started.outputDirectory);
  await chooseView(page, "researcher");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");

  const captured = await captureOneTransition(page);
  expect(captured.recording).toMatchObject({
    lifecycle: "recording",
    captured_transition_count: 1,
    restart_fenced: true,
    discard_available: true,
  });
  await expect(page.locator("#reset-button")).toBeEnabled();
  await expect(page.locator("#scenario-select")).toHaveCount(0);
  await expect(page.locator("#movement-scale-input")).toHaveCount(0);

  await page.locator("#reset-button").click();
  await expect(page.locator("#recording-discard-dialog")).toBeVisible();
  await expect(page.locator("#recording-discard-intent")).toHaveText(
    "Reset the current episode",
  );
  // Save is the suggested choice, so it takes focus.
  await expect(page.locator("#recording-save-replace-button")).toBeFocused();
  await expectRecordingCount(page, 1);
  const openDiscardHelp = await expectVisibleInteractiveHelpInventory(page);
  expect(openDiscardHelp.registered).toContain("#recording-discard-cancel-button");
  expect(openDiscardHelp.registered).toContain("#recording-discard-confirm-button");
  await page.locator("#recording-discard-cancel-button").click();
  await expect(page.locator("#recording-discard-dialog")).toBeHidden();
  const closedDiscardHelp = await expectVisibleInteractiveHelpInventory(page);
  expect(closedDiscardHelp.registered).not.toContain(
    "#recording-discard-cancel-button",
  );
  expect(closedDiscardHelp.registered).not.toContain(
    "#recording-discard-confirm-button",
  );
  await expectRecordingCount(page, 1);

  await page.locator("#reset-button").click();
  const resetResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await page.locator("#recording-discard-confirm-button").click();
  expect((await resetResponse).status()).toBe(200);
  const restarted = await expectRecordingCount(page, 0);
  expect(restarted.episode_id).not.toBe(captured.episode_id);
  expect(restarted.recording).toMatchObject({
    restart_fenced: false,
    discard_available: false,
  });

  await captureOneTransition(page);
  await expectFinishAndSave(page);
  let releaseFinish = () => {};
  const heldFinish = new Promise((resolve) => {
    releaseFinish = () => resolve(null);
  });
  await page.route(
    "**/api/command",
    async (route) => {
      await heldFinish;
      await route.continue();
    },
    { times: 1 },
  );
  const finishResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  try {
    await page.locator("#recording-finish-button").click();
    await expectReplaySaving(page);
  } finally {
    releaseFinish();
  }
  const response = await finishResponse;
  expect(response.status()).toBe(200);
  const replayPage = await openSavedReplay(page);
  await expectSettledReplayHandoff(replayPage);
  await expect(page.locator("#recording-finish-button")).toHaveText("Open Replay");
  expect(page.url()).toBe(stableUrl);
  expect(
    await page.evaluate(() =>
      window.sessionStorage.getItem("marl-battlegrounds.debugger-token"),
    ),
  ).toBe(stableToken);
  const savedNames = (await readdir(started.outputDirectory)).filter((name) =>
    name.endsWith(".marlbg-replay.json"),
  );
  expect(savedNames).toHaveLength(1);
  const firstPath = join(started.outputDirectory, savedNames[0]);
  const firstSaved = await expectSavedArtifacts(
    firstPath,
    metricReportPathForReplay(firstPath),
    1,
  );
  // A saved game stays open to look at: the Your Turn card's champion rows
  // still change which champion the deck shows.
  const secondChampion = page.locator("#checklist .cl-row").nth(1);
  await expect(secondChampion).not.toHaveClass(/\bfocus\b/u);
  await secondChampion.click();
  await expect(secondChampion).toHaveClass(/\bfocus\b/u);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#submit-turn-button")).toBeDisabled();
  await page.locator("#reset-button").click();
  const fresh = await expectRecordingCount(page, 0);
  expect(fresh.episode_id).not.toBe(restarted.episode_id);
  await captureOneTransition(page);
  await page.locator("#reset-button").click();
  await expect(page.locator("#recording-discard-dialog")).toBeVisible();
  await page.locator("#recording-save-replace-button").click();
  const afterSaveAndRestart = await expectRecordingCount(page, 0);
  expect(afterSaveAndRestart.episode_id).not.toBe(fresh.episode_id);
  const allNames = (await readdir(started.outputDirectory)).filter((name) =>
    name.endsWith(".marlbg-replay.json"),
  );
  expect(allNames).toHaveLength(2);
  expect(await readFile(firstPath)).toEqual(firstSaved.replay.bytes);
  const evidencePath = testInfo.outputPath("saved-first.marlbg-replay.json");
  await mkdir(dirname(evidencePath), { recursive: true });
  await writeFile(evidencePath, firstSaved.replay.bytes);
  const secondName = allNames.find((name) => !savedNames.includes(name));
  if (secondName === undefined) throw new Error("The second recording was not saved.");
  await writeFile(
    testInfo.outputPath("saved-second.marlbg-replay.json"),
    await readFile(join(started.outputDirectory, secondName)),
  );
  await writeFile(
    testInfo.outputPath("accepted-restarts.json"),
    JSON.stringify(
      { first: restarted, second: fresh, third: afterSaveAndRestart },
      null,
      2,
    ),
  );
  expectNoBrowserErrors(replayPage);
  expectNoBrowserErrors(page);
});

test("Exit persists an interrupted prefix before clean process shutdown", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await captureOneTransition(page);
  expectNoBrowserErrors(page);
  const shutdown = expectServerShutdown(page, started.process, isLiveCommandPost);
  const responsePromise = shutdown.response.then(async (response) => ({
    payload: await response.json(),
    status: response.status(),
  }));
  await page.locator("#exit-button").click();
  const response = await responsePromise;
  expect(response.status).toBe(200);
  expect(response.payload).toMatchObject({
    schema_version: 2,
    result: "shutdown_scheduled",
    frame: {
      recording: {
        lifecycle: "saved",
        captured_transition_count: 1,
        completion_state: "interrupted",
        completion_reason: "user_exit",
      },
    },
  });
  const { replay } = await expectSavedArtifacts(
    started.replayPath,
    started.metricReportPath,
    1,
  );
  expect(replay.value.completion).toMatchObject({
    completion_state: "interrupted",
    end_or_failure_reason: "user_exit",
  });
  expect(await shutdown.exit).toEqual({ exitCode: 0, signalCode: null });
});

test("Exit reports a server that dies before its shutdown response", async ({
  page,
}) => {
  const started = requiredRecording();
  const launcher = started.process;
  const launcherPid = Number(launcher.pid);
  await openRecording(page, started.url);
  const token = serverOwnerToken(launcher);
  const failure = captureServerFailure(launcher);
  const shutdown = expectServerShutdown(page, launcher, isLiveCommandPost);
  // The registry settles these inside the launcher's exit event, before the
  // listener below runs, so the test awaits only the real exit and then reads them.
  const observed = {
    /** @type {any} */ failure: null,
    /** @type {any} */ exit: null,
  };
  failure.then((error) => {
    observed.failure = error;
  });
  shutdown.exit.then((outcome) => {
    observed.exit = outcome;
  });
  const launcherExit = new Promise((resolveExit) => {
    launcher.once("exit", resolveExit);
  });
  /** @type {number[]} */
  let killedServerProcesses = [];
  let launcherPaused = false;
  await page.route("**/api/command", async (route) => {
    if (route.request().postDataJSON()?.command?.command_type !== "exit") {
      await route.continue();
      return;
    }
    launcherPaused = true;
    killedServerProcesses = await killServerWithLauncherPaused(token, launcherPid);
    await route.continue();
  });

  const response = {
    /** @type {"pending" | "fulfilled" | "rejected"} */ status: "pending",
    /** @type {any} */ value: undefined,
    settledAt: 0,
  };
  shutdown.response.then(
    (value) => {
      Object.assign(response, {
        status: "fulfilled",
        value,
        settledAt: performance.now(),
      });
    },
    (error) => {
      Object.assign(response, {
        status: "rejected",
        value: error,
        settledAt: performance.now(),
      });
    },
  );
  // Wait for the browser's own requestfailed, not for the helper's promise: the
  // launcher is paused, so a helper that ignored requestfailed would otherwise leave
  // this test waiting forever. waitForEvent also rejects on a page crash or close.
  const exitCommandFailed = page.waitForEvent("requestfailed", isLiveCommandPost);
  exitCommandFailed.catch(() => {});

  try {
    const clickedAt = performance.now();
    await page.locator("#exit-button").click();
    const failedRequest = await exitCommandFailed;
    await new Promise((resolveTurn) => setImmediate(resolveTurn));
    test.info().annotations.push(
      {
        type: "shutdown-response-rejection-ms",
        description: String(Math.round(response.settledAt - clickedAt)),
      },
      {
        type: "shutdown-response-rejection",
        description: String(response.value?.message),
      },
      {
        type: "killed-server-processes",
        description: killedServerProcesses.join(", "),
      },
    );
    expect(killedServerProcesses.length).toBeGreaterThan(0);
    expect(killedServerProcesses).not.toContain(launcherPid);
    // The launcher is still paused, so Node has not reported its exit, and the
    // helper's requestfailed handler is what rejected the response.
    expect(launcher.exitCode).toBeNull();
    expect(launcher.signalCode).toBeNull();
    expect(observed.failure).toBeNull();
    expect(response.status).toBe("rejected");
    expect(response.value).not.toBeInstanceOf(ServerFailureError);
    expect(response.value.message).toMatch(
      /^The shutdown request POST http:\/\/127\.0\.0\.1:\d+\/api\/command failed before its response arrived \(net::ERR_[A-Z_]+\)\. Recording debugger \(.+\) exit: not reported yet\.$/u,
    );
    expect(response.value.message).toContain(
      `(${failedRequest.failure()?.errorText}).`,
    );
  } finally {
    if (launcherPaused && launcher.exitCode === null && launcher.signalCode === null) {
      // SIGKILL also ends a paused process. The registry then runs the failure path.
      signalOwnedProcess(launcherPid, [token], "SIGKILL");
    }
  }
  await launcherExit;
  await new Promise((resolveTurn) => setImmediate(resolveTurn));

  expect(observed.failure).toBeInstanceOf(ServerFailureError);
  expect(observed.failure.outcome).toEqual({ exitCode: null, signalCode: "SIGKILL" });
  expect(observed.failure.errors[0].message).toContain(
    "exited during the test with exit code none and signal SIGKILL",
  );
  expect(observed.failure.errors[0].message).toContain(
    "Allowed exit: exit code 0 and no signal, after a product shutdown request.",
  );
  expect(observed.failure.message.startsWith(observed.failure.errors[0].message)).toBe(
    true,
  );
  expect(observed.exit).toEqual({ exitCode: null, signalCode: "SIGKILL" });
  expect(findOwnedProcessIds([token])).toEqual([]);
  await page.unroute("**/api/command");
});

test("a second live tab stays live after saving and rejects its old turn", async ({
  context,
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  const stalePage = await context.newPage();
  await openRecording(stalePage, started.url);
  const before = await currentFrame(stalePage);
  await captureOneTransition(page);
  await page.locator("#recording-finish-button").click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "saved",
  );
  const token = await stalePage.evaluate(() =>
    sessionStorage.getItem("marl-battlegrounds.debugger-token"),
  );
  const rejected = await stalePage.request.post(
    new URL("/api/command", stalePage.url()).href,
    {
      headers: { "X-MARL-Debugger-Token": token ?? "" },
      data: {
        schema_version: 1,
        client_id: "stale-recording-tab",
        command_id: "old-turn",
        base_revision: before.revision,
        command: {
          command_type: "keyboard",
          key: "Enter",
          shift_key: false,
          ctrl_key: false,
          alt_key: false,
          meta_key: false,
          repeat: false,
        },
      },
    },
  );
  expect(rejected.status()).toBe(409);
  expect((await rejected.json()).latest_frame.recording.lifecycle).toBe("saved");
  await stalePage.reload();
  await expect(stalePage.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  await expect(stalePage.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "saved",
  );
  await expect(stalePage.locator("#submit-turn-button")).toBeDisabled();
  const replayPage = await openSavedReplay(stalePage);
  const { frame } = await expectSettledReplayHandoff(replayPage);
  expect(frame.cursor).toMatchObject({ frame_index: 0, final_frame_index: 1 });
  const saved = await readJsonArtifact(started.replayPath);
  expect(saved.value.transitions).toHaveLength(1);
  expectNoBrowserErrors(page);
  expectNoBrowserErrors(stalePage);
  expectNoBrowserErrors(replayPage);
  await stalePage.close();
});

test("actor POV separate Viewer retains battlefield fog and artifact-wide facts", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await chooseView(page, "pov");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  const initialPov = await currentFrame(page);
  expect(initialPov).toMatchObject({
    schema_version: 2,
    frame_kind: "shared_obs_agent_pov_live_debugger",
    view_mode: "pov",
  });
  expectExactWireRecording(initialPov, 0);
  for (const forbidden of [
    "scenario",
    "available_scenarios",
    "incoming_transition_id",
    "incoming_transition_index",
  ]) {
    expect(initialPov).not.toHaveProperty(forbidden);
  }
  await expect(page.locator("#recording-badge")).toBeVisible();
  await expect(page.locator("#rec-pill-text")).toHaveText("Recording · Tick 0");

  await captureOneTransition(page);
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await page.locator("#recording-finish-button").click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const livePage = page;
  page = await openSavedReplay(livePage);
  // A new Viewer opens in Oracle View, whatever the live view. Its Agent POV
  // shows the champion the live Agent POV showed, with that team's fog.
  await expectSettledReplayHandoff(page);
  const { frame, timeline } = await switchReplayToAgentPov(page);
  expect(initialPov.recipient_public_agent_id).toEqual(expect.any(String));
  expect(frame.public_agent_id).toBe(initialPov.recipient_public_agent_id);
  expect(frame).toMatchObject({
    artifact_summary: {
      public_agent_id: frame.public_agent_id,
      captured_transition_count: 1,
      captured_frame_count: 2,
    },
    artifact_facts: {
      schema_version: 1,
      artifact_summary: {
        metric_report_availability: "not_recorded",
        recorded_transition_count: 1,
        recorded_frame_count: 2,
      },
      completion: { validated_transition_count: 1 },
      processing: {
        status: "not_requested",
        processed_transition_count: 0,
      },
    },
  });
  expect(frame.artifact_summary).not.toHaveProperty("replay_reference");
  await expect(page.locator("#replay-artifact-reference")).toHaveText(
    frame.artifact_facts.artifact_summary.replay_reference.artifact_id,
  );
  await expect(page.locator("#replay-processing-badge")).toHaveText(
    "Authorized replay",
  );
  await expect(page.locator("#replay-download-metrics-button")).toBeEnabled();
  const forbiddenKeys = new Set([
    "processing",
    "processing_status",
    "processed_transition_count",
    "failure_stage",
    "failure_code",
    "attempted_transition_index",
    "metric_report_reference",
  ]);
  /** @type {Set<string>} */
  const leaked = new Set();
  /** @param {unknown} value */
  const visit = (value) => {
    if (Array.isArray(value)) {
      for (const child of value) visit(child);
      return;
    }
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      if (forbiddenKeys.has(key)) leaked.add(key);
      visit(child);
    }
  };
  const battlefieldTransport = { ...frame };
  delete battlefieldTransport.artifact_facts;
  visit(battlefieldTransport);
  visit(timeline);
  expect([...leaked].sort()).toEqual([]);
  await expectSavedArtifacts(started.replayPath, started.metricReportPath, 1);
  expectNoBrowserErrors(livePage);
  expectNoBrowserErrors(page);
});

test("target race remains Connected and fenced until Save As recovers cached artifacts", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await captureOneTransition(page);
  const sentinel = await createReplayTargetRace(started);

  await page.locator("#recording-finish-button").click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "persistence_failed",
  );
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#recording-persistence-error")).toHaveText(
    "Destination unavailable",
  );
  // A failed save shows in the pill, the deck's message line and the top
  // bar's save button, which turns into Retry Save.
  await expect(page.locator("#rec-pill-text")).toHaveText("Save Failed · Retry");
  await expect(page.locator("#pop-state")).toHaveText("Save Failed · 1 Tick Kept");
  await expectLiveMessage(page, "Replay save failed; choose Retry Save or Save As.");
  const saveButton = page.locator("#recording-finish-button");
  await expect(saveButton).toHaveText("Retry Save");
  await expect(saveButton).toHaveAttribute("data-action", "retry");
  await expect(saveButton).toBeEnabled();
  await openRecordingPanel(page);
  await expect(page.locator("#pop-finish")).toBeHidden();
  await expect(page.locator("#recording-retry-button")).toBeVisible();
  await expect(page.locator("#recording-retry-button")).toBeEnabled();
  await expect(page.locator("#recording-save-as-input")).toBeVisible();
  await expect(page.locator("#recording-save-as-input")).toBeEnabled();
  await expect(page.locator("#recording-save-as-button")).toBeEnabled();
  await closeRecordingPanel(page);
  await expect(page.locator("#reset-button")).toBeEnabled();
  await expect(page.locator("#scenario-select")).toHaveCount(0);
  await expect(page.locator("#movement-scale-input")).toHaveCount(0);
  await expect(page.locator("#advance-script-button")).toHaveCount(0);
  for (const mode of ["pov", "researcher"]) {
    await expect(
      page.locator(`#view-switch button[data-view-mode="${mode}"]`),
    ).toBeEnabled();
  }
  await expect(page.locator("#preset-select")).toHaveCount(0);
  await expect(page.locator("#exit-button")).toBeEnabled();
  const oracleCloseoutLabel =
    "Read-only live battlefield. Select an agent to inspect it; new actions are unavailable.";
  await expect(page.getByRole("group", { name: oracleCloseoutLabel })).toBeVisible();
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Inspect the accepted game. Select an agent to see its facts. This view does not submit actions; Restart starts another match when available.",
  );
  await expect(page.locator('[data-key="g"]')).toBeEnabled();
  await expect(page.locator('[data-key="v"]')).toHaveCount(0);
  await expect(page.locator('[data-key="n"]')).toHaveCount(0);
  await expect(page.locator('[data-key="w"]')).toBeDisabled();
  // Allowed Now is a group of pick buttons: the group is disabled, and so is
  // every pick in it.
  await expect(page.locator("#command-target-select")).toBeDisabled();
  await expect(page.locator("#command-target-select")).toHaveAttribute(
    "aria-disabled",
    "true",
  );
  await expect(page.locator("#command-target-select .pick:enabled")).toHaveCount(0);
  // The Ranges switch (G) sits in the Visual Filters drawer and stays usable.
  await expect(page.locator("#visual-filters-button")).toBeEnabled();
  await page.locator("#visual-filters-button").click();
  await expect(page.locator("#live-ranges-button")).toBeVisible();
  await expect(page.locator("#live-ranges-button")).toBeEnabled();
  await page.locator("#visual-filters-button").click();
  await expect(page.locator("#visual-filters-drawer")).toBeHidden();
  // The Your Turn card's champion rows stay usable for looking at champions.
  const rosterButtons = page.locator("#checklist .cl-row");
  await expect.poll(() => rosterButtons.count()).toBeGreaterThan(0);
  expect(
    await rosterButtons.evaluateAll((buttons) =>
      buttons.every(
        (button) => button instanceof HTMLButtonElement && !button.disabled,
      ),
    ),
  ).toBe(true);
  const scientificOwner = page
    .locator('#battlefield [data-tooltip-owner][tabindex="0"]')
    .first();
  await expect(scientificOwner).toBeVisible();
  /** @type {unknown[]} */
  const scientificOwnerRequests = [];
  /** @param {import("@playwright/test").Request} request */
  const recordScientificOwnerRequest = (request) => {
    if (
      request.method() === "POST" &&
      ["/api/command", "/api/replay/command"].includes(new URL(request.url()).pathname)
    ) {
      scientificOwnerRequests.push(request.postDataJSON());
    }
  };
  page.on("request", recordScientificOwnerRequest);
  await scientificOwner.focus();
  await scientificOwner.press("Enter");
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  page.off("request", recordScientificOwnerRequest);
  expect(scientificOwnerRequests).toEqual([]);
  const focusFence = await page.evaluate(() => {
    const blockedSelectors = [
      "#battlefield",
      "#command-target-select",
      "#command-target-select .pick",
      '[data-key="w"]',
      "#submit-turn-button",
    ];
    /** @param {Element | null} element */
    const tabFocusable = (element) =>
      element instanceof HTMLElement &&
      !element.hidden &&
      !element.hasAttribute("disabled") &&
      element.tabIndex >= 0 &&
      getComputedStyle(element).display !== "none" &&
      getComputedStyle(element).visibility !== "hidden" &&
      element.getClientRects().length > 0;
    const blockedTabLeaks = [];
    for (const selector of blockedSelectors) {
      for (const element of document.querySelectorAll(selector)) {
        if (tabFocusable(element)) {
          blockedTabLeaks.push(selector);
        }
      }
    }
    const allowedSelectors = [
      '#view-switch button[data-view-mode="pov"]',
      '#view-switch button[data-view-mode="researcher"]',
      "#visual-filters-button",
      "#exit-button",
    ];
    const allowedFocus = [];
    for (const selector of allowedSelectors) {
      const element = document.querySelector(selector);
      if (tabFocusable(element)) {
        allowedFocus.push(selector);
      }
    }
    return { allowedFocus, blockedTabLeaks };
  });
  expect(focusFence).toEqual({
    allowedFocus: [
      '#view-switch button[data-view-mode="pov"]',
      '#view-switch button[data-view-mode="researcher"]',
      "#visual-filters-button",
      "#exit-button",
    ],
    blockedTabLeaks: [],
  });

  await chooseView(page, "pov");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  const agentCloseoutLabel =
    "Agent POV recording closeout battlefield. Authorized bodies can be inspected; simulator controls are unavailable.";
  await expect(page.getByRole("group", { name: agentCloseoutLabel })).toBeVisible();
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Inspect visible bodies here, or click a champion's portrait in the scorebug to inspect its view. This recording cannot accept another turn.",
  );
  const agentCloseoutBodies = page.locator("#battlefield .agent[role='button']");
  await expect.poll(() => agentCloseoutBodies.count()).toBeGreaterThan(0);
  // Agent POV still lists every champion under its team. Enemies the team
  // cannot see read Out of Sight, in the list and on the scorebug.
  const agentCloseoutRows = page.locator("#checklist .cl-row");
  await expect.poll(() => agentCloseoutRows.count()).toBeGreaterThan(1);
  await expect(page.locator("#checklist > .cl-head")).toHaveText(["Team A", "Team B"]);
  const unseenRows = page.locator('#checklist .cl-row[data-sight="out"]');
  await expect(unseenRows.first()).toBeVisible();
  await expect(unseenRows.first()).toContainText("Out of Sight");
  await expect(
    page.locator('#checklist .cl-row:not([data-sight="out"])').first(),
  ).toBeVisible();
  await expect(page.locator("#scorebug .por.unseen").first()).toBeVisible();
  await expect(page.locator("#scorebug .por:not(.unseen)").first()).toBeVisible();
  expect(
    await agentCloseoutRows.evaluateAll((buttons) =>
      buttons.every(
        (button) => button instanceof HTMLButtonElement && !button.disabled,
      ),
    ),
  ).toBe(true);
  /** @type {unknown[]} */
  const localActivationRequests = [];
  /** @param {import("@playwright/test").Request} request */
  const recordLocalActivation = (request) => {
    if (
      request.method() === "POST" &&
      ["/api/command", "/api/replay/command"].includes(new URL(request.url()).pathname)
    ) {
      localActivationRequests.push(request.postDataJSON());
    }
  };
  page.on("request", recordLocalActivation);
  const inspectedCloseoutBody = agentCloseoutBodies.nth(1);
  const inspectedCloseoutLabel = await inspectedCloseoutBody.getAttribute("aria-label");
  const inspectedCloseoutPublicId =
    inspectedCloseoutLabel?.match(/Agent ID ([^.]+)/u)?.[1];
  if (inspectedCloseoutPublicId === undefined) {
    throw new Error("Agent closeout body lacks a public identity.");
  }
  await inspectedCloseoutBody.click();
  // The page's agent inspector follows the inspected body. The BattleClient
  // keeps that inspector in its closed Developer drawer and shows the champion
  // in its hover card instead.
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  await expect(page.locator("#agent-details")).toContainText(
    new RegExp(`Agent ${inspectedCloseoutPublicId}(?!\\d)`, "u"),
  );
  await expect(page.locator("#hcard")).toBeVisible();
  await expect(page.locator("#hcard")).toContainText(
    `Agent ID ${inspectedCloseoutPublicId}`,
  );
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  page.off("request", recordLocalActivation);
  expect(localActivationRequests).toEqual([]);

  const failedFrame = await currentFrame(page);
  expect(failedFrame.recording).toMatchObject({
    lifecycle: "persistence_failed",
    persistence_error_code: "target_unavailable",
    retry_available: true,
    save_as_available: true,
  });
  await expect(page.locator("body")).not.toContainText(started.outputDirectory);
  await expect(page.locator("body")).not.toContainText(started.replayPath);
  // Check the help inventory with the recovery controls in view.
  await openRecordingPanel(page);
  // The popover says why the save failed, in plain words.
  await expect(page.locator("#pop-reason")).toHaveText(
    "Reason: The save folder or file name could not be used, for example a missing folder or a file that already exists.",
  );
  const recoveryHelp = await expectVisibleInteractiveHelpInventory(page);
  expect(recoveryHelp.disabled.length).toBeGreaterThan(0);
  const helped = [
    ...recoveryHelp.registered,
    ...recoveryHelp.describedBy.map(({ label }) => label),
  ];
  for (const control of [
    "#recording-retry-button",
    "#recording-save-as-input",
    "#recording-save-as-button",
  ]) {
    expect(helped).toContain(control);
  }

  await page.locator("#recording-retry-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "persistence_failed",
  );

  await closeRecordingPanel(page);
  await page.locator("#exit-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#connection-status")).not.toHaveText("Closed");
  await expectLiveMessage(page, "remains open for recovery");

  await openRecordingPanel(page);
  await page.locator("#recording-save-as-input").fill("../escape.marlbg-replay.json");
  await page.locator("#recording-save-as-button").click();
  await expectLiveMessage(
    page,
    "Save As requires a basename ending in .marlbg-replay.json",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "persistence_failed",
  );

  const recoveredName = "recovered-episode.marlbg-replay.json";
  const recoveredReplayPath = join(started.outputDirectory, recoveredName);
  const recoveredMetricPath = metricReportPathForReplay(recoveredReplayPath);
  await page.locator("#recording-save-as-input").fill(recoveredName);
  await page.locator("#recording-save-as-button").click();
  const replayPage = await openSavedReplay(page);
  // Saved now: the reason line is gone.
  await expect(page.locator("#pop-reason")).toBeHidden();
  await expect(page.locator("#pop-reason")).toHaveText("");
  await expectSettledReplayHandoff(replayPage);
  const { frame: recoveredPov } = await switchReplayToAgentPov(replayPage);
  expect(recoveredPov.public_agent_id).toBe(failedFrame.recipient_public_agent_id);
  await expectSavedArtifacts(recoveredReplayPath, recoveredMetricPath, 1);
  expect(await readFile(started.replayPath)).toEqual(Buffer.from(sentinel));
  expectNoBrowserErrors(page);
});

test("Reconnect completes a lost Finish response without retrying publication", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await captureOneTransition(page);
  let finishRequests = 0;
  let releaseResponse = () => {};
  const delayed = new Promise((resolve) => {
    releaseResponse = () => resolve(null);
  });
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    if (payload?.command?.command_type !== "finish_and_review") {
      await route.continue();
      return;
    }
    finishRequests += 1;
    const response = await route.fetch();
    expect(response.status()).toBe(200);
    await delayed;
    await route.abort("failed");
  });

  try {
    await page.locator("#recording-finish-button").click();
    await expectReplaySaving(page);
  } finally {
    releaseResponse();
  }
  // A failed connection reads Disconnected, which comes before Resync Required.
  await expect(page.locator("#connection-status")).toHaveText("Disconnected");
  await expect(page.locator("#reconnect-button")).toBeVisible();
  await expectLiveMessage(
    page,
    "Command outcome is unknown because the connection failed",
  );
  // The lost answer ends the saving state: the button is Finish and Save
  // again, disabled until Reconnect.
  await expectFinishAndSave(page);
  await expect(page.locator("#recording-finish-button")).toBeDisabled();
  expect(finishRequests).toBe(1);

  await page.unroute("**/api/command");
  await page.locator("#reconnect-button").click();
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  const replayPage = await openSavedReplay(page);
  await expectSettledReplayHandoff(replayPage);
  await expect(page.locator("#recording-finish-button")).toHaveText("Open Replay");
  expect(finishRequests).toBe(1);
  await expectSavedArtifacts(started.replayPath, started.metricReportPath, 1);

  const errors = browserErrors.get(page) ?? [];
  expect(errors).toEqual(["console: Failed to load resource: net::ERR_FAILED"]);
});

test("blocked popups use a normal link and Viewer Exit can be reopened", async ({
  page,
}) => {
  const started = requiredRecording();
  await page.addInitScript(() => {
    window.open = () => null;
  });
  await openRecording(page, started.url);
  await page.locator("#recording-finish-button").click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "saved",
  );
  await openRecordingPanel(page);
  await page.locator("#recording-review-button").click();
  const link = page.locator("#recording-replay-link");
  await expect(link).toBeVisible();
  const firstUrl = await link.getAttribute("href");
  const popup = page.waitForEvent("popup");
  await link.click();
  const viewer = await popup;
  await expectSettledReplayHandoff(viewer);
  await viewer.locator("#exit-button").click();
  await expect(viewer.locator("#connection-status")).toHaveText("Closed");
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  await openRecordingPanel(page);
  await page.locator("#recording-review-button").click();
  await expect(link).not.toHaveAttribute("href", firstUrl ?? "");
  const reopened = page.waitForEvent("popup");
  await link.click();
  await expectSettledReplayHandoff(await reopened);
  await expectSavedArtifacts(started.replayPath, started.metricReportPath, 0);
  expectNoBrowserErrors(page);
});

test("an old replacement dialog cannot save or discard a newer match", async ({
  page,
  context,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await captureOneTransition(page);
  await page.locator("#reset-button").click();
  await expect(page.locator("#recording-discard-dialog")).toBeVisible();
  const second = await context.newPage();
  await second.goto(started.url);
  await expect(second.locator("#connection-status")).toHaveText("Connected");
  await captureNextTransition(second, 2);
  await expect
    .poll(async () => (await currentFrame(page)).recording.captured_transition_count)
    .toBe(2);
  // The normal background read installs the newer revision while the dialog stays open.
  await expect(page.locator("#pop-state")).toHaveText(/^2 Ticks Kept Since Tick \d+$/u);
  /** @type {import("@playwright/test").Request[]} */
  const commands = [];
  page.on("request", (request) => {
    if (isLiveCommandPost(request)) commands.push(request);
  });
  await page.locator("#recording-save-replace-button").click();
  await expectLiveMessage(page, "The match changed");
  expect(commands).toHaveLength(0);
  expect((await currentFrame(page)).recording.captured_transition_count).toBe(2);
  expect(
    (await readdir(started.outputDirectory)).filter((name) =>
      name.endsWith(".marlbg-replay.json"),
    ),
  ).toHaveLength(0);
});

test("an old replay response cannot attach a link to a newer match", async ({
  page,
  context,
}) => {
  const started = requiredRecording();
  await page.addInitScript(() => {
    window.open = () => null;
  });
  await openRecording(page, started.url);
  await page.locator("#recording-finish-button").click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "saved",
  );
  const previous = await currentFrame(page);
  let release = () => {};
  let held = () => {};
  const heldPromise = new Promise((resolve) => {
    held = () => resolve(null);
  });
  let armed = true;
  await page.route("**/api/presentation/frame", async (route) => {
    if (armed) {
      armed = false;
      await new Promise((resolve) => {
        release = () => resolve(null);
        held();
      });
    }
    await route.continue();
  });
  await openRecordingPanel(page);
  await page.locator("#recording-review-button").click();
  await heldPromise;
  try {
    const origin = new URL(started.url).origin;
    const token = new URL(started.url).hash.slice("#token=".length);
    const replacement = await context.request.post(`${origin}/api/command`, {
      headers: { "X-MARL-Debugger-Token": token, Origin: origin },
      data: {
        client_id: "replay-response-race",
        command_id: "restart",
        base_revision: previous.revision,
        command: { command_type: "reset" },
      },
    });
    expect(replacement.status()).toBe(200);
    expect((await replacement.json()).frame.episode_id).not.toBe(previous.episode_id);
  } finally {
    release();
  }
  await expect(page.locator("#notice")).toContainText(
    "The match changed while opening the replay",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "recording",
  );
  // The new match has no saved replay: no link, and Open Replay waits again.
  await openRecordingPanel(page);
  await expect(page.locator("#recording-replay-link")).toBeHidden();
  await expect(page.locator("#recording-review-button")).toBeVisible();
  await expect(page.locator("#recording-review-button")).toBeDisabled();
  await expectFinishAndSave(page);
  expect((await currentFrame(page)).episode_id).not.toBe(previous.episode_id);
  await expectSavedArtifacts(started.replayPath, started.metricReportPath, 0);
  expectNoBrowserErrors(page);
});
