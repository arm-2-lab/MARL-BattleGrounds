/**
 * @file Prove mixed-controller ownership and server-owned Run/Pause in Chromium.
 * The slow host fixture chooses neutral actions. This checks interaction and
 * session lifetime, not learned team behavior. The deck names a champion
 * "<Class> · ID N", adds " · Watch Only · <policy>" for a policy champion, and in
 * Agent POV names the last Manual champion focused. Finish And Save is reached
 * through the Recording popover.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import {
  closeRecordingPanel,
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

/** @param {import("@playwright/test").Page} page
 * @param {{public_agent_id: string, class_name: string}} row
 * @param {boolean} manual */
async function inspect(page, row, manual) {
  const id = row.public_agent_id;
  await page
    .getByRole("button", {
      name: `${manual ? "Control and inspect" : "Inspect"} Agent ID ${id}`,
      exact: true,
    })
    .click();
  await expect(page.locator("#command-controlled-actor")).toContainText(
    new RegExp(`^${row.class_name} · ID ${id}(?: ·|$)`, "u"),
  );
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
  await launch(page, "mixed");
  const initial = await snapshot(page);
  const rows = initial.presentation.current_endpoint.identity_directory.identities;
  const first = rows.find(
    (/** @type {any} */ row) => row.team_id === 1 && row.team_local_slot === 0,
  );
  const second = rows.find(
    (/** @type {any} */ row) => row.team_id === 1 && row.team_local_slot === 1,
  );
  const policy = rows.find(
    (/** @type {any} */ row) => row.team_id === 1 && row.team_local_slot === 2,
  );
  expect(initial.frame.combat_configuration.manual_slots).toEqual([0, 1]);
  await expect(page.locator("#live-run-button")).toBeHidden();
  await expect(page.locator("#live-match-rules")).toContainText("Step Limit 8");
  await expect(page.locator("#roster .roster-controller")).toHaveCount(10);
  await inspect(page, first, true);
  await page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"]:not([aria-pressed="true"])',
    )
    .first()
    .click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const firstDrafts = (await snapshot(page)).frame.hud.pending_actions;
  await inspect(page, second, true);
  await page.locator("#battlefield").focus();
  await page.keyboard.press("d");
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
  await expect(page.locator("#command-target-select")).toHaveValue("policy-pending");
  await page.locator("#policy-target-masks > summary").click();
  await expect(page.locator("#policy-target-mask-rows li")).toHaveCount(
    mask.target_actions.length,
  );
  await expect(page.locator("#policy-target-masks")).toContainText(
    "has not chosen its action",
  );
  await expect(page.locator("#policy-target-mask-rows li").first()).toContainText(
    "No Combat Allowed",
  );
  await expect(page.locator("#basic-button")).toHaveAttribute(
    "data-authoritative-available",
    String(
      mask.target_use_ultimate_joint_mask.some(
        (/** @type {boolean[]} */ pair, /** @type {number} */ index) =>
          index > 0 && pair[0],
      ),
    ),
  );
  await expect(page.locator("#ultimate-button")).toHaveAttribute(
    "data-authoritative-available",
    String(
      mask.target_use_ultimate_joint_mask.some(
        (/** @type {boolean[]} */ pair) => pair[1],
      ),
    ),
  );
  await page.locator("#view-select").selectOption("pov");
  // Agent POV shows the last Manual champion focused (second), never the
  // focused policy champion, so its deck names that Manual champion.
  await expect(page.locator("#command-controlled-actor")).toHaveText(
    `${second.class_name} · ID ${second.public_agent_id}`,
  );
  await page.locator("#view-select").selectOption("researcher");
  await inspect(page, second, true);
  await page.locator("#command-target-select").focus();
  const beforeInput = (await snapshot(page)).frame.revision;
  // "d" is a movement key that starts no option (a native select jumps to the
  // first option starting with the typed letter, such as "Warrior · ID 6").
  await page.keyboard.press("d");
  expect((await snapshot(page)).frame.revision).toBe(beforeInput);
  await page.locator("#battlefield").focus();
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
  await expect(page.locator("#step-value")).toHaveText("1");
  await expect(page.locator("#submit-turn-button")).toBeEnabled();
  await page.reload();
  await expect(page.locator("#step-value")).toHaveText("1");
  await expect(page.locator("#live-run-button")).toBeHidden();
});

test("Run survives refresh and closing a tab; Pause fences the next turn", async ({
  page,
  context,
}) => {
  const url = await launch(page, "autonomous");
  await expect(page.locator("#live-run-button")).toHaveText("Run");
  await expect(page.locator("#live-run-status")).toHaveText("Paused");
  const other = await context.newPage();
  await other.goto(url);
  await expect(other.locator("#connection-status")).toHaveText("Connected");
  await page.locator("#live-run-button").click();
  await expect(page.locator("#live-run-button")).toHaveText("Pause");
  await page.reload();
  await expect(page.locator("#live-run-button")).toHaveText("Pause");
  await page.close();
  await expect(other.locator("#live-run-button")).toHaveText("Pause");
  await other.locator("#live-run-button").click();
  await expect(other.locator("#live-run-status")).toHaveText("Paused");
  const stopped = await snapshot(other);
  expect(stopped.frame.run_state).toBe("paused");
  expect(stopped.frame.system_operation).toBeUndefined();
  const step = Number(await other.locator("#step-value").textContent());
  expect(step).toBeGreaterThan(0);
  expect(step).toBeLessThan(8);
  await other.locator("#submit-turn-button").click();
  await expect(other.locator("#step-value")).toHaveText(String(step + 1));
  await expect(other.locator("#submit-turn-button")).toBeEnabled();
  await other.locator("#live-run-button").click();
  await expect(other.locator("#step-value")).toHaveText("8");
  await expect(other.locator("#live-run-status")).toHaveText("Paused");
  await expect(other.locator("#live-run-button")).toBeDisabled();
  await expect(other.locator("#submit-turn-button")).toBeDisabled();
});

test("Fixed scenarios keep dead Manual slots and the packaged clock", async ({
  page,
}) => {
  await launch(page, "fixed_manual");
  const current = await snapshot(page);
  expect(current.frame.combat_configuration.manual_slots).toEqual([0, 1, 2, 3, 4]);
  await expect(page.locator("#step-value")).toHaveText("295");
  await expect(page.locator("#live-run-button")).toBeHidden();
  await expect(page.locator("#live-control-help")).toContainText(
    "A fallen champion you play still waits for Lock In.",
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
    await page.locator("#submit-turn-button").click();
    await expect(page.locator("#step-value")).toHaveText("1");
    const accepted = await snapshot(page);
    expect(accepted.frame.system_operation.state).toBe("finishing");
    expect(accepted.frame.recording.captured_transition_count).toBe(1);
    await expect(page.locator("#recording-finish-button")).toBeDisabled();
    await writeFile(release, "Release accepted evidence\n");
    await expect(page.locator("#recording-finish-button")).toBeEnabled();
    const response = page.waitForResponse(
      (reply) =>
        reply.request().method() === "POST" &&
        new URL(reply.url()).pathname === "/api/command",
    );
    await openRecordingPanel(page);
    await page.locator("#recording-finish-button").click();
    await closeRecordingPanel(page);
    expect((await response).status()).toBe(200);
    await expect(page.locator("html")).toHaveAttribute(
      "data-recording-lifecycle",
      "saved",
    );
  } finally {
    await writeFile(release, "Release on test cleanup\n");
  }
});
