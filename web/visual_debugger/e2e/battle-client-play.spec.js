/**
 * @file Prove mixed-controller ownership and page-owned Run/Pause in Chromium.
 * The slow host fixture chooses neutral actions. This checks interaction and
 * session lifetime, not learned team behavior.
 *
 * Mixed match (two Manual champions and three policy champions on Team A, an
 * 8-tick limit): the scorebug reads "Tick 0 of 8"; its portraits mark the
 * champions you play "yours", and a click on one focuses that champion in the
 * deck (class, Agent ID). Each Manual champion keeps its own draft when the
 * focus moves. A policy champion's deck is Watch Only: read only, its Step and
 * Action buttons unpressed with the policy's own masks, its Allowed Now picks
 * exactly the targets the policy's mask allows and all disabled, its No Combat
 * slot saying the policy has not chosen its action, Lock In still enabled and
 * every draft unchanged. The Watch Only card names the policy and offers Back
 * To Your <Class>, the last Manual champion focused. Agent POV keeps that
 * Manual champion's view, never the policy champion's. A key typed while a
 * deck button has focus sends nothing; only the focused arena takes order
 * keys. Two quick Lock In clicks play one turn, and a refresh keeps the tick.
 *
 * Policies Only: Run belongs to the page that pressed it. The server stays
 * paused, another tab never shows Run as running, a refresh ends that Run once
 * the turn already out lands, and closing the tab leaves the other tab free to
 * Run. Pause stops after the current turn, Play One Turn plays exactly one,
 * and Run to the tick limit ends at Game Over with Run and Play One Turn
 * disabled.
 *
 * A fixed scenario keeps all five Manual slots and its packaged clock ("Tick
 * 295 of 300"); Your Turn keeps a row for each fallen champion you play,
 * marked Fallen, counts only the living ones in its set pill, and Lock In
 * stays enabled. Finish And Save, in the top bar and in the Recording popover,
 * waits while the accepted turn's policy evidence is written, then saves from
 * the popover.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import {
  openRecordingPanel,
  startDebugger,
  stopDebugger,
} from "./support/live-debugger.js";

/** @type {import("node:child_process").ChildProcess | null} */
let server = null;
test.afterEach(async () => {
  const child = server;
  server = null;
  await stopDebugger(child);
});

/** @param {import("@playwright/test").Page} page */
async function snapshot(page) {
  return page.evaluate(async () => {
    const token = sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) throw new Error("No live capability token.");
    const response = await fetch("/api/live/snapshot", {
      headers: { "X-MARL-Debugger-Token": token },
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`Live snapshot failed: ${response.status}`);
    return response.json();
  });
}

/** @param {import("@playwright/test").Page} page
 * @param {string} fixture */
async function launch(page, fixture) {
  const started = await startDebugger({ extraArgs: ["--controller-fixture", fixture] });
  server = started.process;
  await page.goto(started.url);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  return started.url;
}

/** @param {import("@playwright/test").Page} page */
function tickLine(page) {
  return page.locator("#scorebug .sb-mid b");
}

/** @param {import("@playwright/test").Page} page
 * @param {{public_agent_id: string, class_name: string, team_id: number}} row
 * @param {boolean} manual */
async function inspect(page, row, manual) {
  const id = row.public_agent_id;
  const team = row.team_id === 1 ? "A" : "B";
  // The portrait's name marks the champions you play "yours".
  await page
    .locator("#scorebug")
    .getByRole("button", {
      name: `${row.class_name}, Agent ID ${id}, Team ${team}${manual ? ", yours" : ""}`,
      exact: true,
    })
    .click();
  await expect(page.locator("#ag-name")).toHaveText(row.class_name);
  await expect(page.locator("#ag-meta")).toHaveText(`Agent ID ${id}`);
  await expect(page.locator("#watch-only")).toBeVisible({ visible: !manual });
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
}

test("Manual drafts survive focus changes and policy inspection stays read-only", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 720, height: 900 });
  /** @type {Array<Record<string, any>>} */
  const commands = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/command")) {
      const body = JSON.parse(request.postData() ?? "{}");
      commands.push(body.command ?? body);
    }
  });
  await launch(page, "mixed");
  const initial = await snapshot(page);
  /** @type {Array<any>} */
  const rows = initial.presentation.current_endpoint.identity_directory.identities;
  const first = rows.find((row) => row.team_id === 1 && row.team_local_slot === 0);
  const second = rows.find((row) => row.team_id === 1 && row.team_local_slot === 1);
  // The Priest: its mask allows healing targets at tick 0, so Allowed Now has rows.
  const policy = rows.find((row) => row.team_id === 1 && row.team_local_slot === 4);
  const policyLabel =
    initial.frame.combat_configuration.slot_labels[rows.indexOf(policy)];
  expect(initial.frame.combat_configuration.manual_slots).toEqual([0, 1]);
  await expect(page.locator("#live-run-button")).toBeHidden();
  await expect(tickLine(page)).toHaveText("Tick 0 of 8");
  await expect(page.locator("#scorebug button.por")).toHaveCount(10);
  await expect(page.locator('#scorebug button.por[aria-label$=", yours"]')).toHaveCount(
    2,
  );
  await inspect(page, first, true);
  await expect(page.locator("#ag-chip")).toHaveText("You Control This Agent");
  const step = page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"]:not([aria-pressed="true"])',
    )
    .first();
  const firstMove = Number(await step.getAttribute("data-move-action"));
  await step.click();
  await expect(
    page.locator(`#command-deck button[data-move-action="${firstMove}"]`),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const firstDrafts = (await snapshot(page)).frame.hud.pending_actions;
  expect(firstDrafts[0].move_action).toBe(firstMove);
  await inspect(page, second, true);
  await page.locator("#battlefield").focus();
  await page.keyboard.press("d");
  await expect(page.locator('#command-deck button[data-key="d"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const secondDrafts = (await snapshot(page)).frame.hud.pending_actions;
  expect(secondDrafts[0]).toEqual(firstDrafts[0]);
  expect(secondDrafts[1]).not.toEqual(firstDrafts[1]);

  await inspect(page, policy, false);
  await expect(page.locator("#command-deck")).toHaveAttribute(
    "data-policy-controller-read-only",
    "true",
  );
  await expect(page.locator("#ag-chip")).toHaveText(`Played By ${policyLabel}`);
  await expect(page.locator("#wo-name")).toHaveText(
    new RegExp(`^${policy.class_name} · `, "u"),
  );
  await expect(page.locator("#wo-meta")).toHaveText(
    `Agent ID ${policy.public_agent_id} · Played By ${policyLabel}`,
  );
  await expect(page.locator("#wo-back")).toHaveText(
    `Back To Your ${second.class_name}`,
  );
  await expect(page.locator("#command-target-select")).toBeDisabled();
  await expect(page.locator("#submit-turn-button")).toBeEnabled();
  const policyFrame = await snapshot(page);
  expect(policyFrame.frame.hud.pending_actions).toEqual(secondDrafts);
  const mask = policyFrame.presentation.live_inspection.inspection.decision_mask;
  const movement = await page
    .locator("#command-deck button[data-move-action]")
    .evaluateAll((buttons) =>
      buttons.map((button) => ({
        action: Number(button.getAttribute("data-move-action")),
        selected: button.getAttribute("aria-pressed"),
        available: button.getAttribute("data-authoritative-available"),
      })),
    );
  for (const row of movement) {
    expect(row.selected).toBe("false");
    expect(row.available).toBe(String(mask.movement_action_mask[row.action]));
  }
  for (const id of ["no-combat-button", "basic-button", "ultimate-button"])
    await expect(page.locator(`#${id}`)).toHaveAttribute("aria-pressed", "false");
  // Allowed Now lists exactly the targets the policy's mask allows for an
  // ability (row 0 is No Target), each naming the allowed abilities.
  /** @type {boolean[][]} */
  const pairs = mask.target_use_ultimate_joint_mask;
  const basicName = (await page.locator("#basic-name").textContent()) ?? "";
  const ultimateName = (await page.locator("#ult-name").textContent()) ?? "";
  const allowed = mask.target_actions
    .filter((/** @type {any} */ row) => row.target_action > 0)
    .filter((/** @type {any} */ row) => pairs[row.target_action].some(Boolean))
    .map((/** @type {any} */ row) => ({
      target: row.target_public_agent_id,
      what: [
        pairs[row.target_action][0] ? basicName : null,
        pairs[row.target_action][1] ? ultimateName : null,
      ]
        .filter(Boolean)
        .join(" · "),
    }));
  expect(allowed.length).toBeGreaterThan(0);
  const picks = page.locator("#command-target-select button.pick");
  await expect(picks).toHaveCount(allowed.length);
  for (const [index, row] of allowed.entries()) {
    await expect(picks.nth(index)).toHaveAttribute("data-target", row.target);
    await expect(picks.nth(index).locator("em")).toHaveText(row.what);
    await expect(picks.nth(index)).toBeDisabled();
  }
  // No Combat (No Target, Basic lane) is allowed; the policy has not chosen.
  expect(pairs[0][0]).toBe(true);
  await expect(page.locator("#no-combat-button")).toHaveAttribute(
    "data-authoritative-available",
    "true",
  );
  await expect(page.locator("#no-combat-button")).toHaveAccessibleDescription(
    "The policy may choose No Combat. It has not chosen its action yet.",
  );
  await expect(page.locator("#basic-button")).toHaveAttribute(
    "data-authoritative-available",
    String(pairs.some((pair, index) => index > 0 && pair[0])),
  );
  await expect(page.locator("#ultimate-button")).toHaveAttribute(
    "data-authoritative-available",
    String(pairs.some((pair) => pair[1])),
  );

  // Agent POV keeps the view of the last Manual champion focused (second),
  // never the focused policy champion's. The deck still shows the policy
  // champion as Watch Only, with the way back to that Manual champion.
  await page.locator('#view-switch button[data-view-mode="pov"]').click();
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const pov = await snapshot(page);
  expect(pov.presentation.authority.recipient_public_agent_id).toBe(
    second.public_agent_id,
  );
  expect(pov.presentation.live_inspection.inspection.draft.actor_public_agent_id).toBe(
    second.public_agent_id,
  );
  await expect(page.locator("#wo-back")).toHaveText(
    `Back To Your ${second.class_name}`,
  );
  await page.locator('#view-switch button[data-view-mode="researcher"]').click();
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");
  await inspect(page, second, true);

  // A key typed while a deck button has focus sends nothing; only the focused
  // arena takes order keys. "a" (West) would change the second champion's
  // step if it reached the draft. The "s" on the arena after it is a marker:
  // once its South step shows, every command sent since is on the list.
  await page.locator("#no-combat-button").focus();
  const sentBeforeKey = commands.length;
  await page.keyboard.press("a");
  await page.locator("#battlefield").focus();
  await page.keyboard.press("s");
  const south = page.locator('#command-deck button[data-key="s"]');
  await expect(south).toHaveAttribute("aria-pressed", "true");
  expect(commands.slice(sentBeforeKey)).toEqual([
    expect.objectContaining({ command_type: "keyboard", key: "s" }),
  ]);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const southDrafts = (await snapshot(page)).frame.hud.pending_actions;
  expect(southDrafts[0]).toEqual(secondDrafts[0]);
  expect(southDrafts[1].move_action).toBe(
    Number(await south.getAttribute("data-move-action")),
  );
  expect(southDrafts[1].move_action).not.toBe(secondDrafts[1].move_action);
  await page.keyboard.press("Escape");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );

  await page.locator("#submit-turn-button").evaluate((element) => {
    const button = /** @type {HTMLButtonElement} */ (element);
    button.click();
    button.click();
  });
  await expect(page.locator("#submit-turn-button")).toBeDisabled();
  await expect(tickLine(page)).toHaveText("Tick 1 of 8");
  await expect(page.locator("#submit-turn-button")).toBeEnabled();
  expect((await snapshot(page)).frame.simulator_step_count).toBe(1);
  await page.reload();
  await expect(tickLine(page)).toHaveText("Tick 1 of 8");
  await expect(page.locator("#live-run-button")).toBeHidden();
});

test("Run belongs to the page that pressed it; Pause fences the next turn", async ({
  page,
  context,
}) => {
  /** @type {Array<Record<string, any>>} */
  const sent = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/command")) {
      const body = JSON.parse(request.postData() ?? "{}");
      sent.push(body.command ?? body);
    }
  });
  const url = await launch(page, "autonomous");
  const run = page.locator("#live-run-button");
  await expect(run).toHaveText("Run");
  await expect(page.locator("#wb-state")).toHaveText("Paused");
  const other = await context.newPage();
  await other.goto(url);
  await expect(other.locator("#connection-status")).toHaveText("Connected");
  const sentBeforeRun = sent.length;
  await run.click();
  await expect(run).toHaveText("Pause");
  await expect(run).toHaveAttribute("aria-pressed", "true");
  // The page paces Run with Play One Turn commands; the server stays paused,
  // so the other tab never shows Run as running.
  await expect.poll(() => sent.length).toBeGreaterThan(sentBeforeRun);
  for (const command of sent.slice(sentBeforeRun))
    expect(command).toMatchObject({ command_type: "keyboard", key: "Enter" });
  expect((await snapshot(page)).frame.run_state).toBe("paused");
  await expect(other.locator("#live-run-button")).toHaveText("Run");
  await expect(other.locator("#live-run-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );

  // A refresh ends this page's Run: the turn already out lands, then nothing
  // more is sent or played.
  await page.reload();
  const sentAfterReload = sent.length;
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(run).toHaveText("Run");
  await expect(run).toHaveAttribute("aria-pressed", "false");
  await expect
    .poll(async () => Boolean((await snapshot(page)).frame.system_operation))
    .toBe(false);
  const settled = (await snapshot(page)).frame.simulator_step_count;
  // A wait meant to expire: no further turn may start after the refresh.
  await page.waitForTimeout(1500);
  expect(sent.length).toBe(sentAfterReload);
  expect((await snapshot(page)).frame.simulator_step_count).toBe(settled);

  await page.close();
  const otherRun = other.locator("#live-run-button");
  await expect(otherRun).toHaveText("Run");
  await expect(otherRun).toBeEnabled();
  await otherRun.click();
  await expect(otherRun).toHaveText("Pause");
  await expect
    .poll(async () => (await snapshot(other)).frame.simulator_step_count)
    .toBeGreaterThan(settled);
  await otherRun.click();
  await expect(otherRun).toHaveText("Run");
  await expect(other.locator("#msg")).toHaveText("Paused after the current turn.");
  await expect
    .poll(async () => Boolean((await snapshot(other)).frame.system_operation))
    .toBe(false);
  await expect(other.locator("#wb-state")).toHaveText("Paused");
  const stopped = await snapshot(other);
  expect(stopped.frame.run_state).toBe("paused");
  expect(stopped.frame.system_operation).toBeUndefined();
  const tick = stopped.frame.simulator_step_count;
  expect(tick).toBeGreaterThan(0);
  expect(tick).toBeLessThan(8);
  await expect(tickLine(other)).toHaveText(`Tick ${tick} of 8`);
  // Lock In is hidden in Policies Only; the Watching bar's Play One Turn
  // plays one turn instead.
  await other.locator("#watch-submit").click();
  await expect(tickLine(other)).toHaveText(`Tick ${tick + 1} of 8`);
  await expect(other.locator("#watch-submit")).toBeEnabled();
  await otherRun.click();
  await expect(tickLine(other)).toHaveText("Tick 8 of 8");
  await expect(other.locator("#wb-state")).toHaveText("Game Over");
  await expect(otherRun).toBeDisabled();
  await expect(other.locator("#watch-submit")).toBeDisabled();
});

test("Fixed scenarios keep dead Manual slots and the packaged clock", async ({
  page,
}) => {
  await launch(page, "fixed_manual");
  const current = await snapshot(page);
  expect(current.frame.combat_configuration.manual_slots).toEqual([0, 1, 2, 3, 4]);
  await expect(tickLine(page)).toHaveText("Tick 295 of 300");
  await expect(page.locator("#live-run-button")).toBeHidden();
  // Your Turn keeps a row for every champion you play; a fallen one keeps its
  // row, marked Fallen, and is not counted as needing an order.
  /** @type {Array<any>} */
  const ours = current.presentation.current_endpoint.scene.agents.filter(
    (/** @type {any} */ agent) => agent.team_id === 1,
  );
  const fallen = ours.filter((agent) => agent.life_state !== "alive");
  expect(fallen.length).toBeGreaterThan(0);
  const rows = page.locator("#checklist .cl-row");
  await expect(rows).toHaveCount(ours.length);
  for (const [index, agent] of ours.entries()) {
    const row = rows.nth(index);
    await expect(row).toHaveAttribute("data-slot", agent.public_agent_id);
    if (agent.life_state === "alive") {
      await expect(row).not.toHaveAttribute("data-fallen", "true");
    } else {
      await expect(row).toHaveAttribute("data-fallen", "true");
      await expect(row.locator(".cl-state")).toHaveText("Fallen");
    }
  }
  await expect(page.locator("#set-pill")).toHaveText(
    `0 Of ${ours.length - fallen.length} Set`,
  );
  await expect(page.locator("#submit-turn-button")).toBeEnabled();
});

test("Finish And Save waits for the accepted turn's evidence to settle", async ({
  page,
}, testInfo) => {
  const directory = testInfo.outputPath("finishing");
  await mkdir(directory, { recursive: true });
  const release = `${directory}/release`;
  try {
    const started = await startDebugger({
      extraArgs: [
        "--controller-fixture",
        "finishing",
        "--record-replay",
        `${directory}/recording.marlbg-replay.json`,
        "--finish-release",
        release,
      ],
    });
    server = started.process;
    await page.goto(started.url);
    await expect(page.locator("#connection-status")).toHaveText("Connected");
    // Policies Only: the Watching bar's Play One Turn stands in for Lock In.
    await page.locator("#watch-submit").click();
    await expect(tickLine(page)).toHaveText("Tick 1 of 8");
    const accepted = await snapshot(page);
    expect(accepted.frame.system_operation.state).toBe("finishing");
    expect(accepted.frame.recording.captured_transition_count).toBe(1);
    await expect(page.locator("#recording-finish-button")).toBeDisabled();
    await expect(page.locator("#pop-finish")).toBeDisabled();
    await writeFile(release, "Release accepted evidence\n");
    await expect(page.locator("#recording-finish-button")).toBeEnabled();
    await expect(page.locator("#pop-finish")).toBeEnabled();
    const response = page.waitForResponse(
      (reply) =>
        reply.request().method() === "POST" &&
        new URL(reply.url()).pathname === "/api/command",
    );
    await openRecordingPanel(page);
    await page.locator("#pop-finish").click();
    // Finish And Save closes the popover itself.
    await expect(page.locator("#rec-pop")).toBeHidden();
    expect((await response).status()).toBe(200);
    await expect(page.locator("html")).toHaveAttribute(
      "data-recording-lifecycle",
      "saved",
    );
  } finally {
    await writeFile(release, "Release on test cleanup\n");
  }
});
