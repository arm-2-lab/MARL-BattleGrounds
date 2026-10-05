/**
 * @file Prove the BattleClient's controls do their real job against a real
 * server, in Chromium. ? on the focused arena opens and closes How To Play
 * (#htp) and sends nothing; How To Play's button and Close do the same. Esc
 * says the target is cleared and keys are off. A 1 with no target is refused
 * by the page in the vision's words and never reaches Python. A right click
 * on the arena sends the secondary pointer. G in Oracle View sends keyboard g
 * and the deck's message line says the reach rings changed. N on the focused arena
 * is one tick: in interactive play it goes to Python as keyboard n and locks
 * in, so the tick advances by exactly one. The Recording pill opens its popover; Finish And Save there closes
 * it and saves; the pill reads Saved, the top bar's save button turns into an
 * enabled Open Replay, and the popover's Open Replay stays visible and becomes
 * enabled. After the
 * save, R restarts at once (a new episode, no dialog). Exit closes the server
 * and the page says the match is closed. In Policies Only (the harness's
 * autonomous fixture), Run is paced by the page: it sends only Play One Turn
 * (keyboard Enter) commands, one turn at a time, never Python's run or pause,
 * so the server's run_state stays paused; the button reads Pause (pressed),
 * Play One Turn is disabled, Space and N on the focused arena are ignored,
 * and the deck's message line keeps "Running." from turn to turn. Pause lets at most
 * the turn already out land, then says "Paused after the current turn." and
 * sends nothing more; Play One Turn then plays exactly
 * one tick and says "The turn played. The match is paused again.".
 */
import { expect, test } from "@playwright/test";
import {
  expectServerShutdown,
  startDebugger,
  stopDebugger,
} from "./support/live-debugger.js";
import {
  startRecordingDebugger,
  stopRecordingDebugger,
} from "./support/recording-handoff.js";

/** @type {Awaited<ReturnType<typeof startRecordingDebugger>> | null} */
let started = null;
/** @type {import("node:child_process").ChildProcess | null} */
let plain = null;
test.afterEach(async () => {
  const owned = started;
  started = null;
  await stopRecordingDebugger(owned);
  const child = plain;
  plain = null;
  await stopDebugger(child);
});

/** @param {import("@playwright/test").Page} page */
async function liveFrame(page) {
  return page.evaluate(async () => {
    const token = sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) throw new Error("No live capability token.");
    const response = await fetch("/api/live/snapshot", {
      headers: { "X-MARL-Debugger-Token": token },
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`Live snapshot failed: ${response.status}`);
    const body = await response.json();
    return {
      tick: Number(body.frame.simulator_step_count),
      runState: String(body.frame.run_state),
      working: Boolean(body.frame.system_operation),
    };
  });
}

/** @param {import("@playwright/test").Page} page */
async function liveEpisode(page) {
  return page.evaluate(async () => {
    const token = sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) throw new Error("No live capability token.");
    const response = await fetch("/api/live/snapshot", {
      headers: { "X-MARL-Debugger-Token": token },
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`Live snapshot failed: ${response.status}`);
    const body = await response.json();
    return String(body.frame.episode_id);
  });
}

test("every BattleClient control sends its command and shows its effect", async ({
  page,
}) => {
  started = await startRecordingDebugger({ stem: "battle-client-controls" });
  const server = started.process;
  /** @type {Array<Record<string, unknown>>} */
  const commands = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/command")) {
      const body = JSON.parse(request.postData() ?? "{}");
      commands.push(body.command ?? body);
    }
  });
  await page.goto(started.url);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const arena = page.locator("#battlefield");
  const message = page.locator("#msg");

  // Keys on: a click on the arena focuses it.
  await arena.click({ position: { x: 24, y: 24 } });
  await expect(page.locator("#keys-text")).toHaveText("Keys On");

  // ? opens and closes How To Play and sends nothing.
  const sentBeforeHelp = commands.length;
  await page.keyboard.press("?");
  await expect(page.locator("#htp")).toBeVisible();
  await page.keyboard.press("?");
  await expect(page.locator("#htp")).toBeHidden();
  await page.locator("#help-button").click();
  await expect(page.locator("#htp")).toBeVisible();
  await page.locator("#help-close-button").click();
  await expect(page.locator("#htp")).toBeHidden();
  expect(commands.length).toBe(sentBeforeHelp);

  // 1 with no target is refused by the page in the vision's words.
  await arena.click({ position: { x: 24, y: 24 } });
  await page.keyboard.press("Escape");
  await expect(message).toHaveText(
    "Target cleared. Keys are off until you click the arena again.",
  );
  await expect(page.locator("#keys-text")).toHaveText("Keys Off · Click The Arena");
  await arena.click({ position: { x: 24, y: 24 } });
  const sentBeforeRefusal = commands.length;
  await page.keyboard.press("1");
  await expect(message).toHaveText(
    / needs a target first\. Your order is unchanged\.$/u,
  );
  expect(commands.length).toBe(sentBeforeRefusal);

  // A right click sends the secondary pointer. A wait meant to expire first:
  // the deck keeps a refusal it wrote on the line for 1.5 s (REFUSAL_HOLD_MS
  // in tactics-turn.js), so "Target cleared." is said after that hold.
  await page.waitForTimeout(1600);
  await arena.click({ position: { x: 40, y: 40 }, button: "right" });
  await expect
    .poll(() => commands.some((command) => command.button === "secondary"))
    .toBe(true);
  await expect(message).toHaveText("Target cleared.");

  // G in Oracle View goes to Python and the status line says what changed.
  await arena.click({ position: { x: 24, y: 24 } });
  await page.keyboard.press("g");
  await expect(message).toHaveText(/^Reach rings (shown|hidden)\.$/u);
  expect(commands.at(-1)).toMatchObject({ command_type: "keyboard", key: "g" });

  // N on the focused arena is one tick: in interactive play it goes to
  // Python as keyboard n, the same Lock In as Space, and the tick advances
  // by exactly one.
  await arena.click({ position: { x: 24, y: 24 } });
  const sentBeforeN = commands.length;
  const tickBeforeN = (await liveFrame(page)).tick;
  await page.keyboard.press("n");
  await expect.poll(async () => (await liveFrame(page)).tick).toBe(tickBeforeN + 1);
  expect(
    commands
      .slice(sentBeforeN)
      .filter((command) => command.command_type === "keyboard"),
  ).toEqual([expect.objectContaining({ key: "n" })]);

  // The Recording pill opens its popover; Finish And Save there saves.
  await page.locator("#recording-badge").click();
  await expect(page.locator("#rec-pop")).toBeVisible();
  await expect(page.locator("#pop-state")).toHaveText(/Ticks? Kept Since Tick \d+$/u);
  await expect(page.locator("#recording-review-button")).toBeVisible();
  await expect(page.locator("#recording-review-button")).toBeDisabled();
  await page.locator("#pop-finish").click();
  await expect(page.locator("#rec-pop")).toBeHidden();
  await expect(page.locator("#rec-pill-text")).toHaveText("Saved");
  // The top bar's save button turns into Open Replay (owner, 4 Oct 2026).
  await expect(page.locator("#recording-finish-button")).toHaveText("Open Replay");
  await expect(page.locator("#recording-finish-button")).toBeEnabled();
  expect(commands.at(-1)).toMatchObject({ command_type: "finish_and_review" });
  await page.locator("#recording-badge").click();
  await expect(page.locator("#recording-review-button")).toBeEnabled();
  await page.locator("#recording-badge").click();

  // After the save, R restarts at once: a new episode and no dialog.
  const savedEpisode = await liveEpisode(page);
  await arena.click({ position: { x: 24, y: 24 } });
  await page.keyboard.press("r");
  await expect.poll(() => liveEpisode(page)).not.toBe(savedEpisode);
  await expect(page.locator("#recording-discard-dialog")).toBeHidden();

  // Exit closes the server and the page says the match is closed.
  const shutdown = expectServerShutdown(
    page,
    server,
    (request) =>
      request.method() === "POST" && new URL(request.url()).pathname === "/api/command",
  );
  await page.locator("#exit-button").click();
  expect((await shutdown.response).status()).toBe(200);
  await expect(message).toHaveText(/The match is closed\./u);
  await shutdown.exit;
});

test("Policies Only Run plays one turn at a time and Pause stops after the current turn", async ({
  page,
}) => {
  const launched = await startDebugger({
    extraArgs: ["--controller-fixture", "autonomous"],
  });
  plain = launched.process;
  /** @type {Array<Record<string, unknown>>} */
  const commands = [];
  let commandsOut = 0;
  /** @param {import("@playwright/test").Request} request */
  const isCommand = (request) =>
    request.method() === "POST" && request.url().endsWith("/api/command");
  page.on("request", (request) => {
    if (!isCommand(request)) return;
    const body = JSON.parse(request.postData() ?? "{}");
    commands.push(body.command ?? body);
    commandsOut += 1;
  });
  page.on("requestfinished", (request) => {
    if (isCommand(request)) commandsOut -= 1;
  });
  page.on("requestfailed", (request) => {
    if (isCommand(request)) commandsOut -= 1;
  });
  await page.goto(launched.url);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const run = page.locator("#live-run-button");
  const playOne = page.locator("#watch-submit");
  const message = page.locator("#msg");
  await expect(run).toHaveText("Run");
  const start = (await liveFrame(page)).tick;
  const sentBeforeRun = commands.length;

  await run.click();
  await expect(run).toHaveText("Pause");
  await expect(run).toHaveAttribute("aria-pressed", "true");
  await expect(playOne).toBeDisabled();
  // Space and N on the focused arena while Run plays are ignored, as on the
  // vision page: only Run sends turns, so no key " " or "n" command may go out
  // (checked below: every command during Run is Run's own Enter).
  await page.locator("#battlefield").focus();
  await page.keyboard.press("Space");
  await page.keyboard.press("n");
  await page.keyboard.press("Space");
  await expect
    .poll(async () => (await liveFrame(page)).tick)
    .toBeGreaterThanOrEqual(start + 2);
  await expect(message).toHaveText("Running.");
  expect((await liveFrame(page)).runState).toBe("paused");
  const runTurns = commands.slice(sentBeforeRun);
  expect(runTurns.length).toBeGreaterThanOrEqual(2);
  for (const command of runTurns) {
    expect(command).toMatchObject({ command_type: "keyboard", key: "Enter" });
  }

  await run.click();
  const pausedAt = (await liveFrame(page)).tick;
  await expect(run).toHaveText("Run");
  await expect(run).toHaveAttribute("aria-pressed", "false");
  await expect(message).toHaveText("Paused after the current turn.");
  // The turn already out (if any) lands; then nothing is out any more.
  await expect
    .poll(async () => {
      const frame = await liveFrame(page);
      return commandsOut === 0 && !frame.working;
    })
    .toBe(true);
  await expect(playOne).toBeEnabled();
  const settled = (await liveFrame(page)).tick;
  expect(settled - pausedAt).toBeLessThanOrEqual(1);
  const sentAfterPause = commands.length;
  // A wait meant to expire: nothing more may be sent or played after Pause.
  await page.waitForTimeout(1500);
  expect(commands.length).toBe(sentAfterPause);
  expect((await liveFrame(page)).tick).toBe(settled);

  await playOne.click();
  await expect.poll(async () => (await liveFrame(page)).tick).toBe(settled + 1);
  await expect(message).toHaveText("The turn played. The match is paused again.");
  expect(commands.slice(sentAfterPause)).toEqual([
    expect.objectContaining({ command_type: "keyboard", key: "Enter" }),
  ]);
  expect((await liveFrame(page)).tick).toBe(settled + 1);
});
