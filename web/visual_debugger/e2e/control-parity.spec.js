/**
 * @file Check that pointer and keyboard orders in the BattleClient's live play
 * install exactly one real successor transition each, however fast they come.
 * Contracts:
 * - Order keys and a Lock In pressed while a draft is out wait in order
 *   ("Input Queued", "Submit Queued"). While a Lock In is out, the deck refuses
 *   order keys and a second Lock In: it shakes a refused step key and sends or
 *   queues nothing. After a tick, Everyone Acts shows for a moment; the deck
 *   takes orders again once Lock In reads "Lock In Turn".
 * - The deck's Your Champion shows the class and "Agent ID N". Allowed Now has
 *   one pick per body that allows an ability now, naming those abilities; a
 *   pick, by pointer or keyboard, sets the target and gives the arena the
 *   keys, and × Esc clears it. Shift + click on a body also picks it.
 * - A Your Turn row gives that Manual champion the orders and the arena the
 *   keys, also for a champion the Agent POV team cannot see. Enter on an arena
 *   body keeps focus on that body's successor.
 * - A Warrior's Charge and the reply Charge stay installable in Agent POV.
 * - Escape clears the target and turns the keys off. A right click on the
 *   arena clears it too, takes the keys and keeps the browser's context menu
 *   away while orders are open.
 * - The Rune Ring on the body shows the cooldown; the ready ring from before a
 *   Burst stays only as the tick's faded "before" piece.
 * - A lost answer or a refused capability shows "Disconnected"; Reconnect
 *   reads the frame again and nothing is sent twice. A stale tab adopts the
 *   latest Ranges state from the Visual Filters drawer.
 * - The Class Guide changes only when a champion is activated.
 * - Space on How To Play opens the How To Play card and sends nothing.
 * - The developer cards (Roster, Agent Details, Technical Frame) sit in the
 *   Developer drawer, which only the DevClient shows; the page still keeps
 *   their open state, which the tests set directly.
 * - A declared System shows Policies Choosing; a cancelled reply stops the
 *   game, keeps it and shows the deck's Stopped panel. The hidden Reconnect
 *   button is clicked through its handler after a configuration change.
 */
import { expect, test } from "@playwright/test";

import { startDebugger, stopDebugger } from "./support/live-debugger.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";

// The deck's Your Champion shows the class and "Agent ID N".
const CLASS_NAMES = "(?:Mage|Warrior|Hunter|Rogue|Priest)";
// The arena_5v5 roster is Mage, Warrior, Hunter, Rogue and Priest on each
// team, so Agent ID 4 is Team A's Priest.
const PRIEST_ID = "4";
const SCIENTIFIC_DISCLOSURES = Object.freeze([
  "#command-deck",
  "#roster-details",
  "#agent-details",
  "#pending-turn-details",
  "#latest-transition-details",
  "#technical-frame-details",
]);

/** @typedef {{target_public_agent_id?: string, target_action: number}} TargetActionRow */
/** @typedef {{move_action: number, target_action: number, use_ultimate_action: number}} ActionTuple */
/** @typedef {{actor_public_agent_id: string, submitted_action: ActionTuple, accepted_action: ActionTuple}} TransitionActionRow */
/** @typedef {{status_id: string}} IncomingStatus */
/** @typedef {{delta_kind: string, agent_public_agent_id: string, changed_dynamic_fields: string[], start_observation: {statuses: IncomingStatus[]}, successor_observation: {statuses: IncomingStatus[]}}} IncomingStatusDelta */

test.beforeAll(async () => {
  const started = await startDebugger();
  serverProcess = started.process;
  debuggerUrl = started.url;
});

test.afterAll(async () => {
  const child = serverProcess;
  serverProcess = null;
  await stopDebugger(child);
});

/** @param {import("@playwright/test").Page} page
 * @returns {Promise<number>} */
async function currentStep(page) {
  return Number(await page.locator("#step-value").textContent());
}

/** @param {import("@playwright/test").Page} page */
async function waitForPlanPhase(page) {
  // After a played tick the deck shows Everyone Acts for a moment and takes
  // no orders. You may plan again once Lock In reads Lock In Turn.
  await expect(page.locator("#lock-label")).toHaveText("Lock In Turn");
}

/** @param {import("@playwright/test").Page} page
 * @returns {Promise<Record<string, any>>} */
async function currentAuthorizedPresentation(page) {
  return page.evaluate(async () => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) {
      throw new Error("Debugger capability token is unavailable.");
    }
    const response = await window.fetch("/api/presentation/frame", {
      cache: "no-store",
      credentials: "omit",
      headers: { "X-MARL-Debugger-Token": token },
      redirect: "error",
    });
    if (!response.ok) {
      throw new Error(
        `Could not obtain the authorized presentation: HTTP ${response.status}.`,
      );
    }
    return response.json();
  });
}

/** @param {import("@playwright/test").Page} page
 * @param {() => Promise<unknown>} activate */
async function activateLiveCommand(page, activate) {
  const response = page.waitForResponse(
    (candidate) =>
      candidate.request().method() === "POST" &&
      new URL(candidate.url()).pathname === "/api/command",
  );
  await activate();
  expect((await response).status()).toBe(200);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#battlefield-empty")).toBeHidden();
}

/** @param {import("@playwright/test").Page} page
 * @returns {Promise<string>} */
async function deckChampionId(page) {
  const meta = await page.locator("#ag-meta").textContent();
  const id = meta?.match(/^Agent ID (\d+)$/u)?.[1];
  if (id === undefined) {
    throw new Error(`The deck names no champion: ${meta}.`);
  }
  return id;
}

/** @param {import("@playwright/test").Page} page
 * @param {string | number} publicAgentId */
async function focusOwnChampion(page, publicAgentId) {
  // A Your Turn row gives the orders to that Manual champion.
  await activateLiveCommand(page, () =>
    page.locator(`#checklist .cl-row[data-slot="${publicAgentId}"]`).click(),
  );
  await expect(page.locator("#ag-meta")).toHaveText(`Agent ID ${publicAgentId}`);
  await expect(page.locator("#ag-name")).toHaveText(
    new RegExp(`^${CLASS_NAMES}$`, "u"),
  );
  await expect(page.locator("#ag-chip")).toHaveText("You Control This Agent");
}

/** @param {import("@playwright/test").Page} page
 * @param {string | number} publicAgentId */
function allowedNowPick(page, publicAgentId) {
  return page.locator(`#command-target-select .pick[data-target="${publicAgentId}"]`);
}

/** @param {import("@playwright/test").Page} page
 * @param {string | number} publicAgentId
 * @returns {Promise<boolean>} */
async function ultimateAllowedOn(page, publicAgentId) {
  // A pick names the abilities allowed on that body now ("Shield Bash ·
  // Charge"); a body with nothing allowed has no pick.
  const ultimateName = await page.locator("#ult-name").textContent();
  const abilities = await allowedNowPick(page, publicAgentId)
    .locator("em")
    .allTextContents();
  return (
    ultimateName !== null &&
    abilities.some((what) => what.split(" · ").includes(ultimateName))
  );
}

/** @param {import("@playwright/test").Page} page */
async function ownRowOutsideBattlefield(page) {
  // In Agent POV, Your Turn keeps a row for every Manual champion, also for
  // one the viewing team cannot see. Find such a row other than the focused
  // champion's.
  const drawnIds = new Set(
    await page
      .locator("#battlefield .agent[data-presentation-key]")
      .evaluateAll((agents) =>
        agents.map(
          (agent) => agent.getAttribute("aria-label")?.match(/^Agent ID (\d+)/u)?.[1],
        ),
      ),
  );
  const rowIds = await page
    .locator("#checklist .cl-row:not(.focus)")
    .evaluateAll((rows) => rows.map((row) => row.getAttribute("data-slot")));
  const id = rowIds.find((candidate) => candidate !== null && !drawnIds.has(candidate));
  if (id === undefined) {
    throw new Error("Agent POV has no Manual champion outside its sight.");
  }
  return page.locator(`#checklist .cl-row[data-slot="${id}"]`);
}

/** @param {import("@playwright/test").Page} page
 * @param {"pov" | "researcher"} mode */
function viewButton(page, mode) {
  // The top bar's Agent POV and Oracle View buttons set the hidden View
  // select (#view-select), which holds the page's view.
  return page.locator(`#view-switch button[data-view-mode="${mode}"]`);
}

/** @param {import("@playwright/test").Page} page
 * @param {"pov" | "researcher"} mode */
async function chooseView(page, mode) {
  await viewButton(page, mode).click();
  await expect(page.locator("#view-select")).toHaveValue(mode);
  await expect(viewButton(page, mode)).toHaveAttribute("aria-pressed", "true");
}

/** @param {import("@playwright/test").Page} page */
async function openVisualFilters(page) {
  // Live play keeps Visual Filters in a drawer that its top bar button opens.
  const drawer = page.locator("#visual-filters-drawer");
  if (!(await drawer.evaluate((node) => node.matches(":popover-open")))) {
    await page.locator("#visual-filters-button").click();
  }
  await expect(drawer).toBeVisible();
  await expect(page.locator("#visual-filters")).toHaveAttribute("open", "");
}

/** @param {import("@playwright/test").Page} page */
async function closeVisualFilters(page) {
  const drawer = page.locator("#visual-filters-drawer");
  if (await drawer.evaluate((node) => node.matches(":popover-open"))) {
    await page.locator("#visual-filters-button").click();
  }
  await expect(drawer).toBeHidden();
}

/** @param {import("@playwright/test").Page} page
 * @param {string} [path] */
async function holdNextPresentation(page, path = "/api/presentation/frame") {
  const routePattern = `**${path}`;
  let captured = false;
  let markHeld = () => {};
  let releaseResponse = () => {};
  let released = false;
  const held = new Promise((resolve) => {
    markHeld = () => resolve(undefined);
  });
  const release = new Promise((resolve) => {
    releaseResponse = () => resolve(undefined);
  });
  const handler = async (/** @type {import("@playwright/test").Route} */ route) => {
    // A background poll must not consume the full Reconnect snapshot hold.
    if (captured || route.request().headers()["if-none-match"] !== undefined) {
      await route.continue();
      return;
    }
    captured = true;
    const response = await route.fetch();
    markHeld();
    await release;
    await route.fulfill({ response });
  };
  await page.route(routePattern, handler);
  return {
    held,
    release() {
      if (!released) {
        released = true;
        releaseResponse();
      }
    },
    async dispose() {
      if (!released) {
        released = true;
        releaseResponse();
      }
      await page.unroute(routePattern, handler);
    },
  };
}

/** @param {import("@playwright/test").Page} page
 * @param {string} selector
 * @param {boolean} open */
async function setDisclosureOpen(page, selector, open) {
  const disclosure = page.locator(selector);
  const summary = disclosure.locator(":scope > summary");
  if ((await disclosure.getAttribute("open")) !== (open ? "" : null)) {
    if (await summary.isVisible()) {
      await summary.click();
    } else {
      // The BattleClient keeps the developer cards (Roster, Agent Details,
      // Technical Frame) in its closed Developer drawer, which only the
      // DevClient shows; the page still keeps their open state.
      await disclosure.evaluate((details, value) => {
        if (!(details instanceof HTMLDetailsElement)) {
          throw new TypeError("Disclosure is unavailable.");
        }
        details.open = value;
      }, open);
    }
  }
  if (open) {
    await expect(disclosure).toHaveAttribute("open", "");
  } else {
    await expect(disclosure).not.toHaveAttribute("open", "");
  }
}

/** @param {import("@playwright/test").Page} page */
async function expectExactDisclosureDefaults(page) {
  await expect(page.locator("#command-deck")).toHaveAttribute("open", "");
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  for (const selector of [
    "#agent-details",
    "#pending-turn-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("open", "");
  }
}

/** @param {import("@playwright/test").Page} page
 * @returns {Promise<Readonly<Record<string, boolean>>>} */
async function disclosureOpenState(page) {
  /** @type {Record<string, boolean>} */
  const result = {};
  for (const selector of SCIENTIFIC_DISCLOSURES) {
    result[selector] = (await page.locator(selector).getAttribute("open")) !== null;
  }
  return Object.freeze(result);
}

/** @param {import("@playwright/test").Page} page
 * @param {string} previousPresentationKey
 * @param {Readonly<Record<string, boolean>>} expectedDisclosureOpen */
async function expectRetainedDisclosureAuthority(
  page,
  previousPresentationKey,
  expectedDisclosureOpen,
) {
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "retained",
  );
  await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
  await expect(page.locator("#battlefield-shell")).toHaveAttribute("aria-busy", "true");
  await expect(page.locator("#scenario-description")).toContainText("Last confirmed");
  await expect(page.locator("#scenario-description")).toContainText("update pending");
  await expect(page.locator("#battlefield-empty")).toBeHidden();
  await expect(
    page.locator(
      `#battlefield .agent[data-presentation-key="${previousPresentationKey}"]`,
    ),
  ).toHaveCount(1);
  await expect(page.locator("#submit-turn-button")).toBeDisabled();
  await expect(page.locator("#command-target-select")).toBeDisabled();
  await expect(page.locator("#view-select")).toBeDisabled();
  for (const selector of SCIENTIFIC_DISCLOSURES) {
    const disclosure = page.locator(selector);
    await expect(disclosure).toHaveAttribute("inert", "");
    await expect(disclosure.locator(":scope > [data-disclosure-body]")).toHaveAttribute(
      "inert",
      "",
    );
    await expect(disclosure.locator(":scope > summary")).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    await expect(disclosure.locator(":scope > summary")).toHaveAttribute(
      "tabindex",
      "-1",
    );
    expect((await disclosure.getAttribute("open")) !== null).toBe(
      expectedDisclosureOpen[selector],
    );
  }
}

test("authorized draft edit and rapid Submit install exactly one successor", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const commands = [];
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    commands.push(payload?.command ?? {});
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-inspection-state",
    "live_editable",
  );
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-submission-scope",
    "joint_turn",
  );
  await expect(page.locator("#ag-name")).toHaveText(
    new RegExp(`^${CLASS_NAMES}$`, "u"),
  );
  await expect(page.locator("#ag-meta")).toHaveText(/^Agent ID \d+$/u);
  await expect(
    page.locator("#battlefield .agent[data-presentation-key]"),
  ).not.toHaveCount(0);

  const controlledPublicId = await deckChampionId(page);
  // Your Turn lists each Manual champion's staged order in words.
  const draftLabel = page.locator(
    `#checklist .cl-row[data-slot="${controlledPublicId}"] > span > span`,
  );
  await expect(draftLabel).toHaveCount(1);
  const draftBefore = await draftLabel.textContent();
  if (draftBefore === null || draftBefore.length === 0) {
    throw new Error("Authorized draft label is unavailable.");
  }
  const movement = page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"]:not([aria-pressed="true"])',
    )
    .first();
  await expect(movement).toBeVisible();
  const movementAction = await movement.getAttribute("data-move-action");
  if (movementAction === null) {
    throw new Error("Authorized movement control did not publish its action identity.");
  }
  const selectedMovement = page.locator(
    `#command-deck button[data-move-action=${JSON.stringify(movementAction)}]`,
  );
  await movement.click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(selectedMovement).toHaveAttribute("aria-pressed", "true");
  await expect(draftLabel).not.toHaveText(draftBefore);
  await expect.poll(() => commands.length).toBe(1);

  const successorStep = await currentStep(page);
  const transitionBefore = await page.locator("#transition-value").textContent();
  const retainedPresentationKey = await page
    .locator("#battlefield .agent[data-presentation-key]")
    .first()
    .getAttribute("data-presentation-key");
  if (retainedPresentationKey === null) {
    throw new Error("Installed battlefield has no presentation key.");
  }
  const retainedDisclosureOpen = await disclosureOpenState(page);
  const northKey = page.locator('#pad .key[data-key="w"]');
  const heldSubmitPresentation = await holdNextPresentation(page);
  try {
    await page.locator("#submit-turn-button").evaluate((button) => {
      if (!(button instanceof HTMLButtonElement)) {
        throw new TypeError("Submit button is unavailable.");
      }
      button.click();
      button.click();
    });
    await heldSubmitPresentation.held;
    await expectRetainedDisclosureAuthority(
      page,
      retainedPresentationKey,
      retainedDisclosureOpen,
    );
    await page.locator("#submit-turn-button").evaluate((button) => {
      if (!(button instanceof HTMLButtonElement)) {
        throw new TypeError("Submit button is unavailable.");
      }
      button.click();
    });
    await page
      .locator(
        `#roster .roster-primary-action[data-presentation-key="${retainedPresentationKey}"]`,
      )
      .evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Retained roster action is unavailable.");
        }
        button.click();
      });
    await page
      .locator(
        `#battlefield .agent[data-presentation-key="${retainedPresentationKey}"]`,
      )
      .dispatchEvent("pointerdown", { button: 0 });
    // While a Lock In is out, the deck refuses order keys without a word: it
    // shakes the key and neither sends nor queues it.
    await expect(northKey).not.toHaveClass(/\bshake\b/u);
    await page.locator("#battlefield").focus();
    await page.keyboard.press("w");
    await expect(northKey).toHaveClass(/\bshake\b/u);
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    expect(commands).toHaveLength(2);
    heldSubmitPresentation.release();
  } finally {
    heldSubmitPresentation.release();
    await heldSubmitPresentation.dispose();
  }
  await expect(page.locator("#step-value")).toHaveText(String(successorStep + 1));
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  // The refused W never lands: the new turn's step is the reset Stay.
  await expect(page.locator('#pad .key[data-key="x"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(northKey).toHaveAttribute("aria-pressed", "false");
  expect(commands).toHaveLength(2);
  expect(
    commands.filter(
      (command) =>
        command.command_type === "keyboard" &&
        String(command.key).toLowerCase() === "enter",
    ),
  ).toHaveLength(1);

  const transitionId = await page.locator("#transition-value").textContent();
  expect(transitionId).not.toBeNull();
  expect(transitionId).not.toBe("—");
  expect(transitionId).not.toBe(transitionBefore);
  await expect(page.locator("#accepted-card")).not.toHaveAttribute(
    "data-transition-id",
    /.+/u,
  );
  await expect(page.locator("#accepted-card .accepted-action-row")).not.toHaveCount(0);
  // Actions In Words is on by default: every observed row shows its words and
  // its recorded codes.
  await expect(
    page.locator("#accepted-card .accepted-action-row__words"),
  ).not.toHaveCount(0);
  await expect(
    page.locator("#accepted-card .accepted-action-row__codes").first(),
  ).toHaveText(/^Move \d+ · Target \d+ · Ultimate [01]$/u);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await page.unroute("**/api/command");
});

test("pointer draft keeps battlefield focus and queues one Enter exactly once", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  const movement = page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"]:not([aria-pressed="true"])',
    )
    .first();
  await expect(movement).toBeVisible();

  const heldDraftPresentation = await holdNextPresentation(page);
  let heldSubmitPresentation = null;
  try {
    await movement.click();
    await heldDraftPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect(battlefield).toBeFocused();
    await expect.poll(() => requests.length).toBe(1);

    await page.keyboard.press("Enter");
    await battlefield.dispatchEvent("keydown", {
      key: "Enter",
      repeat: true,
    });
    await expect.poll(() => requests.length).toBe(1);
    await expect(page.locator("#connection-status")).toHaveText("Submit Queued");

    heldSubmitPresentation = await holdNextPresentation(page);
    heldDraftPresentation.release();
    await heldSubmitPresentation.held;
    await expect.poll(() => requests.length).toBe(2);
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await battlefield.dispatchEvent("keydown", {
      key: "Enter",
      repeat: true,
    });
    await expect.poll(() => requests.length).toBe(2);

    expect(requests[0].command).toMatchObject({ command_type: "keyboard" });
    expect(String(requests[0].command.key).toLowerCase()).not.toBe("enter");
    expect(requests[1].command).toEqual({
      command_type: "keyboard",
      key: "Enter",
      shift_key: false,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    });
    expect(requests[1].command_id).not.toBe(requests[0].command_id);
    expect(requests[1].base_revision).toBe(requests[0].base_revision + 1);
    heldSubmitPresentation.release();
  } finally {
    heldDraftPresentation.release();
    await heldDraftPresentation.dispose();
    heldSubmitPresentation?.release();
    await heldSubmitPresentation?.dispose();
  }

  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await expect.poll(() => requests.length).toBe(2);
  await expect(battlefield).toBeFocused();
  await waitForPlanPhase(page);

  const keyboardMovementCandidate = page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"]:not([aria-pressed="true"])',
    )
    .first();
  const keyboardMovementAction =
    await keyboardMovementCandidate.getAttribute("data-move-action");
  if (keyboardMovementAction === null) {
    throw new Error("Keyboard movement control has no stable action identity.");
  }
  const keyboardMovement = page.locator(
    `#command-deck button[data-move-action=${JSON.stringify(keyboardMovementAction)}]`,
  );
  const keyboardDraftResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  const heldKeyboardPresentation = await holdNextPresentation(page);
  let heldKeyboardSubmitPresentation = null;
  try {
    await keyboardMovement.press("Enter");
    await heldKeyboardPresentation.held;
    expect((await keyboardDraftResponse).status()).toBe(200);
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(3);
    await expect(page.locator("#connection-status")).toHaveText("Submit Queued");
    heldKeyboardSubmitPresentation = await holdNextPresentation(page);
    heldKeyboardPresentation.release();
    await heldKeyboardSubmitPresentation.held;
    await expect.poll(() => requests.length).toBe(4);
    expect(String(requests[2].command.key).toLowerCase()).not.toBe("enter");
    expect(requests[3].command).toEqual({
      command_type: "keyboard",
      key: "Enter",
      shift_key: false,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    });
    expect(requests[3].command_id).not.toBe(requests[2].command_id);
    expect(requests[3].base_revision).toBe(requests[2].base_revision + 1);
    heldKeyboardSubmitPresentation.release();
  } finally {
    heldKeyboardPresentation.release();
    await heldKeyboardPresentation.dispose();
    heldKeyboardSubmitPresentation?.release();
    await heldKeyboardSubmitPresentation?.dispose();
  }
  await expect.poll(() => requests.length).toBe(4);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(keyboardMovement).toBeFocused();
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 2));
  await page.unroute("**/api/command");
});

test("Enter follows a coherent idempotent draft exactly once", async ({ page }) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  const selectedMovement = page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"][aria-pressed="true"]',
    )
    .first();
  await expect(selectedMovement).toBeVisible();
  const heldNoOpPresentation = await holdNextPresentation(page);
  try {
    await selectedMovement.click();
    await heldNoOpPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(1);
    heldNoOpPresentation.release();
  } finally {
    heldNoOpPresentation.release();
    await heldNoOpPresentation.dispose();
  }

  await expect.poll(() => requests.length).toBe(2);
  expect(String(requests[0].command.key).toLowerCase()).not.toBe("enter");
  expect(String(requests[1].command.key).toLowerCase()).toBe("enter");
  expect(requests[1].command_id).not.toBe(requests[0].command_id);
  expect(requests[1].base_revision).toBe(requests[0].base_revision);
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(battlefield).toBeFocused();
  await page.unroute("**/api/command");
});

test("Allowed Now picks queue one Lock In behind pointer and keyboard picks", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await waitForPlanPhase(page);
  const initialChampionId = await deckChampionId(page);
  // The Priest may heal its teammates and itself from its spawn, so Allowed
  // Now has picks for it at the start of a match.
  await focusOwnChampion(page, PRIEST_ID);
  await expect(page.locator("#ag-name")).toHaveText("Priest");
  requests.length = 0;

  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  const picks = page.locator("#command-target-select .pick");
  const targetChip = page.locator("#tchip");
  const basicButton = page.locator("#basic-button");
  const basicTag = basicButton.locator(".tag");
  const basicName = await page.locator("#basic-name").textContent();
  if (basicName === null || basicName.length === 0) {
    throw new Error("The deck does not name the Basic.");
  }
  await expect(targetChip).toHaveClass(/\bnone\b/u);
  await expect(targetChip).toContainText("No Target Yet");
  await expect(
    page.locator('#command-target-select .pick[aria-pressed="true"]'),
  ).toHaveCount(0);
  await expect(basicButton).toHaveAttribute("data-authoritative-available", "false");
  await expect(basicTag).toHaveText("Not Allowed");
  expect(await picks.count()).toBeGreaterThanOrEqual(2);

  const firstPick = picks.first();
  const pointerTarget = await firstPick.getAttribute("data-target");
  const pointerAbilities = await firstPick.locator("em").textContent();
  if (pointerTarget === null || pointerAbilities === null) {
    throw new Error("The first Allowed Now pick does not name its target.");
  }
  // A pick names the abilities allowed on that body now; the Basic arms
  // itself (Auto) when the Basic is one of them.
  const basicAvailable = pointerAbilities.includes(basicName);
  await firstPick.click();
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0].command).toMatchObject({
    command_type: "roster_selection",
    role: "target",
  });
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(allowedNowPick(page, pointerTarget)).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(targetChip).not.toHaveClass(/\bnone\b/u);
  await expect(basicButton).toHaveAttribute(
    "data-authoritative-available",
    String(basicAvailable),
  );
  await expect(basicTag).toHaveText(basicAvailable ? "Auto" : "Not Allowed");

  await page.locator("#clear-target-button").click();
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[1].command).toMatchObject({
    command_type: "keyboard",
    key: "Escape",
  });
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(targetChip).toHaveClass(/\bnone\b/u);
  await expect(
    page.locator('#command-target-select .pick[aria-pressed="true"]'),
  ).toHaveCount(0);
  await expect(basicButton).toHaveAttribute("data-authoritative-available", "false");
  await expect(basicTag).toHaveText("Not Allowed");
  requests.length = 0;

  const heldPointerPresentation = await holdNextPresentation(page);
  try {
    await allowedNowPick(page, pointerTarget).click();
    await heldPointerPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page.locator("#connection-status")).toHaveText("Submit Queued");
    await expect.poll(() => requests.length).toBe(1);
    heldPointerPresentation.release();
  } finally {
    heldPointerPresentation.release();
    await heldPointerPresentation.dispose();
  }

  await expect.poll(() => requests.length).toBe(2);
  expect(requests[0].command).toMatchObject({
    command_type: "roster_selection",
    role: "target",
  });
  expect(String(requests[1].command.key).toLowerCase()).toBe("enter");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(battlefield).toBeFocused();
  await waitForPlanPhase(page);

  // A keyboard pick hands focus to the arena, so the keys stay on for the
  // Lock In that follows it.
  const keyboardPick = page
    .locator('#command-target-select .pick[aria-pressed="false"]')
    .first();
  const keyboardTarget = await keyboardPick.getAttribute("data-target");
  if (keyboardTarget === null) {
    throw new Error("Allowed Now has no second pick for the keyboard.");
  }
  const heldKeyboardPresentation = await holdNextPresentation(page);
  let heldKeyboardSubmitPresentation = null;
  try {
    await keyboardPick.focus();
    await page.keyboard.press("Enter");
    await heldKeyboardPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(3);
    await expect(page.locator("#connection-status")).toHaveText("Submit Queued");
    heldKeyboardSubmitPresentation = await holdNextPresentation(page);
    heldKeyboardPresentation.release();
    await heldKeyboardSubmitPresentation.held;
    await expect.poll(() => requests.length).toBe(4);
    expect(requests[2].command).toMatchObject({
      command_type: "roster_selection",
      role: "target",
    });
    expect(String(requests[3].command.key).toLowerCase()).toBe("enter");
    expect(requests[3].command_id).not.toBe(requests[2].command_id);
    expect(requests[3].base_revision).toBe(requests[2].base_revision + 1);
    heldKeyboardSubmitPresentation.release();
  } finally {
    heldKeyboardPresentation.release();
    await heldKeyboardPresentation.dispose();
    heldKeyboardSubmitPresentation?.release();
    await heldKeyboardSubmitPresentation?.dispose();
  }
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect.poll(() => requests.length).toBe(4);
  await expect(battlefield).toBeFocused();
  await expect(allowedNowPick(page, keyboardTarget)).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 2));
  await page.unroute("**/api/command");

  await waitForPlanPhase(page);
  await activateLiveCommand(page, () => page.locator("#clear-target-button").click());
  await focusOwnChampion(page, initialChampionId);
});

test("Agent pointer Your Turn selection queues movement before one Enter", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await chooseView(page, "pov");
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const initialChampionId = await deckChampionId(page);
  requests.length = 0;

  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  const hiddenChampionRow = await ownRowOutsideBattlefield(page);
  await expect(hiddenChampionRow).toBeVisible();
  const selectedId = await hiddenChampionRow.getAttribute("data-slot");

  const heldRosterPresentation = await holdNextPresentation(page);
  try {
    await hiddenChampionRow.click();
    await heldRosterPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect(battlefield).toBeFocused();
    await expect.poll(() => requests.length).toBe(1);

    await page.keyboard.press("w");
    await expect(page.locator("#connection-status")).toHaveText("Input Queued");
    await page.keyboard.press("Enter");
    await expect(page.locator("#connection-status")).toHaveText("Submit Queued");
    await expect.poll(() => requests.length).toBe(1);

    heldRosterPresentation.release();
  } finally {
    heldRosterPresentation.release();
    await heldRosterPresentation.dispose();
  }

  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await expect.poll(() => requests.length).toBe(3);
  expect(requests[0].command).toMatchObject({
    command_type: "roster_selection",
    role: "control",
  });
  expect(requests[1].command).toMatchObject({
    command_type: "keyboard",
    key: "w",
    repeat: false,
  });
  expect(requests[2].command).toEqual({
    command_type: "keyboard",
    key: "Enter",
    shift_key: false,
    ctrl_key: false,
    alt_key: false,
    meta_key: false,
    repeat: false,
  });
  expect(requests[1].command_id).not.toBe(requests[0].command_id);
  expect(requests[2].command_id).not.toBe(requests[1].command_id);
  expect(requests[1].base_revision).toBe(requests[0].base_revision + 1);
  expect(requests[2].base_revision).toBe(requests[1].base_revision + 1);
  await expect(page.locator("#ag-meta")).toHaveText(`Agent ID ${selectedId}`);
  await expect(battlefield).toBeFocused();

  await page.unroute("**/api/command");
  await focusOwnChampion(page, initialChampionId);
  await chooseView(page, "researcher");
  await expect(page.locator("#connection-status")).toHaveText("Connected");
});

test("Agent keyboard Your Turn selection gives the arena the keys without submitting", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await chooseView(page, "pov");
  await expect(page.locator("#connection-status")).toHaveText("Connected");

  const initialChampionId = await deckChampionId(page);
  const hiddenChampionRow = await ownRowOutsideBattlefield(page);
  await expect(hiddenChampionRow).toBeVisible();
  const selectedId = await hiddenChampionRow.getAttribute("data-slot");
  const baseStep = await currentStep(page);
  requests.length = 0;

  const heldRosterPresentation = await holdNextPresentation(page);
  try {
    await hiddenChampionRow.press("Enter");
    await heldRosterPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect.poll(() => requests.length).toBe(1);
    heldRosterPresentation.release();
  } finally {
    heldRosterPresentation.release();
    await heldRosterPresentation.dispose();
  }

  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep));
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0].command).toMatchObject({
    command_type: "roster_selection",
    role: "control",
  });
  // The row gives that champion the orders and the arena the keys, so the
  // next order key goes to it.
  await expect(page.locator("#ag-meta")).toHaveText(`Agent ID ${selectedId}`);
  await expect(
    page.locator(`#checklist .cl-row[data-slot="${selectedId}"]`),
  ).toHaveClass(/\bfocus\b/u);
  await expect(page.locator("#battlefield")).toBeFocused();

  await page.unroute("**/api/command");
  // With the keys on, the Oracle View button leaves the arena focused.
  await chooseView(page, "researcher");
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expectExactDisclosureDefaults(page);
  await expect(page.locator("#ag-meta")).toHaveText(`Agent ID ${selectedId}`);
  await expect(page.locator("#battlefield")).toBeFocused();

  await focusOwnChampion(page, initialChampionId);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
});

test("Agent keyboard battlefield activation rebinds focus to the authorized successor", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const viewSelect = page.locator("#view-select");
  await chooseView(page, "pov");
  await expect(page.locator("#connection-status")).toHaveText("Connected");

  const initialPublicId = await deckChampionId(page);
  const battlefieldActors = page.locator("#battlefield .agent[data-presentation-key]");
  const labels = await battlefieldActors.evaluateAll((actors) =>
    actors.map((actor) => actor.getAttribute("aria-label")),
  );
  const targetIndex = labels.findIndex(
    (label) =>
      typeof label === "string" && !label.startsWith(`Agent ID ${initialPublicId}.`),
  );
  if (targetIndex < 0) {
    throw new Error("Agent POV has no visible non-controlled battlefield actor.");
  }
  const target = battlefieldActors.nth(targetIndex);
  const targetLabel = await target.getAttribute("aria-label");
  const oldPresentationKey = await target.getAttribute("data-presentation-key");
  const targetPublicId = targetLabel?.match(/Agent ID ([^.]+)/u)?.[1];
  if (
    targetLabel === null ||
    oldPresentationKey === null ||
    targetPublicId === undefined
  ) {
    throw new Error("Visible Agent battlefield actor lacks its public identity.");
  }
  await setDisclosureOpen(page, "#technical-frame-details", true);
  await openVisualFilters(page);
  const ranges = page.locator("#live-ranges-button");
  if ((await ranges.getAttribute("aria-pressed")) !== "false") {
    await ranges.click();
  }
  await expect(ranges).toHaveAttribute("aria-pressed", "false");
  await closeVisualFilters(page);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  const baseStep = await currentStep(page);
  requests.length = 0;

  const heldPresentation = await holdNextPresentation(page);
  try {
    await target.press("Enter");
    await heldPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect.poll(() => requests.length).toBe(1);
    heldPresentation.release();
  } finally {
    heldPresentation.release();
    await heldPresentation.dispose();
  }

  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep));
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0].command).toMatchObject({
    command_type: "roster_selection",
    role: "control",
  });
  await expect(page.locator("#ag-meta")).toHaveText(`Agent ID ${targetPublicId}`);
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  await expect(page.locator("#technical-frame-details")).toHaveAttribute("open", "");
  await expect(ranges).toHaveAttribute("aria-pressed", "false");
  const restoredFocus = await page.evaluate(() => ({
    inBattlefield: Boolean(document.activeElement?.closest("#battlefield")),
    label: document.activeElement?.getAttribute("aria-label") ?? null,
    presentationKey:
      document.activeElement?.getAttribute("data-presentation-key") ?? null,
  }));
  expect(restoredFocus).toMatchObject({
    inBattlefield: true,
    label: targetLabel,
  });
  expect(restoredFocus.presentationKey).not.toBe(oldPresentationKey);

  await page.unroute("**/api/command");
  await viewSelect.evaluate((select) => {
    if (!(select instanceof HTMLSelectElement)) {
      throw new TypeError("View mode control is unavailable.");
    }
    select.value = "researcher";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(viewSelect).toHaveValue("researcher");
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#ag-meta")).toHaveText(`Agent ID ${targetPublicId}`);
  await expectExactDisclosureDefaults(page);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.activeElement?.matches("#battlefield .agent") ?? false,
      ),
    )
    .toBe(false);

  await focusOwnChampion(page, initialPublicId);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
});

test("post-Charge Agent history stays installable through a reciprocal Charge", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const commandRequests = [];
  /** @type {{path: string, status: number}[]} */
  const endpointFailures = [];
  /** @type {string[]} */
  const pageErrors = [];
  const recordResponse = (
    /** @type {import("@playwright/test").Response} */ response,
  ) => {
    const path = new URL(response.url()).pathname;
    if (
      (path === "/api/command" || path === "/api/presentation/frame") &&
      response.status() >= 400
    ) {
      endpointFailures.push({ path, status: response.status() });
    }
  };
  const recordPageError = (/** @type {Error} */ error) => {
    pageErrors.push(error.message);
  };
  page.on("response", recordResponse);
  page.on("pageerror", recordPageError);
  await page.route("**/api/command", async (route) => {
    commandRequests.push(route.request().postDataJSON());
    await route.continue();
  });

  const battlefield = page.locator("#battlefield");
  const viewSelect = page.locator("#view-select");
  const targetChip = page.locator("#tchip");
  /** @type {string | null} */
  let initialControlledPublicId = null;
  let primaryError = null;
  const expectHealthyInstallation = async () => {
    await expect(page.locator("#connection-status")).toHaveText("Connected");
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
    await expect(page.locator("#battlefield-empty")).toBeHidden();
    await expect(page.locator("#notice")).not.toContainText(
      /could not process|safe fault|reconnect before sending|no authorized battlefield scene/iu,
    );
  };
  /** @param {string} key */
  const selectMovement = async (key) => {
    const button = page.locator(`#command-deck button[data-key="${key}"]`);
    await expect(button).toBeEnabled();
    if ((await button.getAttribute("aria-pressed")) !== "true") {
      await activateLiveCommand(page, () => button.click());
    }
    await expect(button).toHaveAttribute("aria-pressed", "true");
  };

  try {
    await page.goto(debuggerUrl);
    await expectHealthyInstallation();
    initialControlledPublicId = await deckChampionId(page);
    await activateLiveCommand(page, () => page.locator("#reset-button").click());
    await expect(page.locator("#step-value")).toHaveText("0");

    // Use only ordinary browser drafts and joint submissions to converge the
    // two Warriors. Stop as soon as Agent 1's exact mask exposes Charge, which
    // Allowed Now then offers on Agent 6.
    let stagedTurns = 0;
    let chargeAllowed = false;
    while (stagedTurns < 9) {
      await focusOwnChampion(page, 1);
      chargeAllowed = await ultimateAllowedOn(page, 6);
      if (chargeAllowed) {
        break;
      }
      await selectMovement("d");
      await focusOwnChampion(page, 6);
      await selectMovement("z");
      const stepBeforeMovement = await currentStep(page);
      await activateLiveCommand(page, () =>
        page.locator("#submit-turn-button").click(),
      );
      await expect(page.locator("#step-value")).toHaveText(
        String(stepBeforeMovement + 1),
      );
      await waitForPlanPhase(page);
      stagedTurns += 1;
    }
    expect(stagedTurns).toBeGreaterThan(0);
    expect(chargeAllowed).toBe(true);

    // Freeze both movement drafts at the converged geometry before combat.
    await selectMovement("x");
    await focusOwnChampion(page, 6);
    await selectMovement("x");
    await focusOwnChampion(page, 1);
    expect(await ultimateAllowedOn(page, 6)).toBe(true);
    await activateLiveCommand(page, () => allowedNowPick(page, 6).click());
    await expect(allowedNowPick(page, 6)).toHaveAttribute("aria-pressed", "true");
    await expect(targetChip).toContainText("Warrior · ID 6");
    await expect(page.locator("#ultimate-button")).toHaveAttribute(
      "data-authoritative-available",
      "true",
    );
    await expect(battlefield.locator('.agent[data-selected="true"]')).toHaveCount(1);
    await expect(battlefield.locator('.agent[data-selected="true"]')).toHaveAttribute(
      "aria-label",
      /^Agent ID 6[,.]/u,
    );
    const firstChargeDraftPresentation = await currentAuthorizedPresentation(page);
    const firstChargeDraft =
      firstChargeDraftPresentation.live_inspection.inspection.draft;
    const firstChargeTargetAction = /** @type {TargetActionRow[]} */ (
      firstChargeDraft.decision_mask.target_actions
    ).find((target) => target.target_public_agent_id === "6")?.target_action;
    expect(firstChargeTargetAction).toEqual(expect.any(Number));
    if (typeof firstChargeTargetAction !== "number") {
      throw new Error("Agent 6 is absent from Agent 1's authorized target axis.");
    }
    expect(firstChargeDraft.draft_action.target_action).toBe(firstChargeTargetAction);

    await activateLiveCommand(page, () => page.locator("#ultimate-button").click());
    const firstChargeStep = await currentStep(page);
    await activateLiveCommand(page, () => page.locator("#submit-turn-button").click());
    await expect(page.locator("#step-value")).toHaveText(String(firstChargeStep + 1));
    await waitForPlanPhase(page);
    const afterFirstCharge = await currentAuthorizedPresentation(page);
    const firstChargeRow = /** @type {TransitionActionRow[]} */ (
      afterFirstCharge.latest_transition.action_rows
    ).find((row) => row.actor_public_agent_id === "1");
    expect(firstChargeRow).toBeTruthy();
    if (firstChargeRow === undefined) {
      throw new Error("Agent 1 is absent from the first Charge transition.");
    }
    expect(firstChargeRow.submitted_action).toMatchObject({
      target_action: firstChargeTargetAction,
      use_ultimate_action: 1,
    });
    expect(firstChargeRow.accepted_action).toEqual(firstChargeRow.submitted_action);

    // Age the one-tick stun while retaining Charge's five-tick slow. This is
    // the exact transition whose historical status ordering used to make the
    // subsequent Agent presentation unbuildable.
    await activateLiveCommand(page, () => page.locator("#no-combat-button").click());
    const settlingStep = await currentStep(page);
    await activateLiveCommand(page, () => page.locator("#submit-turn-button").click());
    await expect(page.locator("#step-value")).toHaveText(String(settlingStep + 1));
    await waitForPlanPhase(page);

    await focusOwnChampion(page, 6);
    const stepBeforeAgentInstall = await currentStep(page);
    await activateLiveCommand(page, () => viewButton(page, "pov").click());
    await expect(viewSelect).toHaveValue("pov");
    await expect(page.locator("#step-value")).toHaveText(
      String(stepBeforeAgentInstall),
    );
    await expectHealthyInstallation();

    const agentAfterSettling = await currentAuthorizedPresentation(page);
    expect(agentAfterSettling.presentation_kind).toBe("live_shared_obs_agent_pov");
    expect(agentAfterSettling.authority.recipient_public_agent_id).toBe("6");
    const statusHistory = /** @type {IncomingStatusDelta[]} */ (
      agentAfterSettling.latest_events.deltas
    ).find(
      (delta) =>
        delta.delta_kind === "observed_values_change" &&
        delta.agent_public_agent_id === "6" &&
        delta.changed_dynamic_fields.includes("statuses"),
    );
    expect(statusHistory).toBeTruthy();
    if (statusHistory === undefined) {
      throw new Error("Agent 6 has no post-Charge status history delta.");
    }
    expect(
      statusHistory.start_observation.statuses.map((status) => status.status_id),
    ).toEqual(["warrior_charge_stun", "warrior_charge_slow"]);
    expect(
      statusHistory.successor_observation.statuses.map((status) => status.status_id),
    ).toEqual(["warrior_charge_slow"]);
    for (const status of [
      ...statusHistory.start_observation.statuses,
      ...statusHistory.successor_observation.statuses,
    ]) {
      expect("direct_sources" in status).toBe(false);
      expect("source_public_agent_id" in status).toBe(false);
      expect("source_presentation_key" in status).toBe(false);
    }

    expect(await ultimateAllowedOn(page, 1)).toBe(true);
    await activateLiveCommand(page, () => allowedNowPick(page, 1).click());
    await expect(allowedNowPick(page, 1)).toHaveAttribute("aria-pressed", "true");
    await expect(targetChip).toContainText("Warrior · ID 1");
    await expect(battlefield.locator('.agent[data-selected="true"]')).toHaveAttribute(
      "aria-label",
      /^Agent ID 1[,.]/u,
    );
    const reverseDraftPresentation = await currentAuthorizedPresentation(page);
    const reverseDraft =
      reverseDraftPresentation.researcher_space.pending_inspection.draft;
    const reverseTargetAction = /** @type {TargetActionRow[]} */ (
      reverseDraft.decision_mask.target_actions
    ).find((target) => target.target_public_agent_id === "1")?.target_action;
    expect(reverseDraft.actor_public_agent_id).toBe("6");
    expect(reverseTargetAction).toEqual(expect.any(Number));
    if (typeof reverseTargetAction !== "number") {
      throw new Error("Agent 1 is absent from Agent 6's authorized target axis.");
    }
    expect(reverseDraft.draft_action.target_action).toBe(reverseTargetAction);

    // Reproduce the user's rapid path: arm Ultimate and press Enter before the
    // Agent successor has finished installing. The queue must serialize both
    // commands and advance exactly once.
    commandRequests.length = 0;
    const reverseChargeStep = await currentStep(page);
    await page.locator("#ultimate-button").click();
    await page.keyboard.press("Enter");
    await expect.poll(() => commandRequests.length).toBe(2);
    expect(commandRequests.map((request) => request.command.key)).toEqual([
      "2",
      "Enter",
    ]);
    expect(commandRequests[1].base_revision).toBe(commandRequests[0].base_revision + 1);
    await expect(page.locator("#step-value")).toHaveText(String(reverseChargeStep + 1));
    await expectHealthyInstallation();
    await expect.poll(() => commandRequests.length).toBe(2);

    const afterReverseCharge = await currentAuthorizedPresentation(page);
    const reverseChargeRow = /** @type {TransitionActionRow[]} */ (
      afterReverseCharge.researcher_space.latest_transition.action_rows
    ).find((row) => row.actor_public_agent_id === "6");
    expect(reverseChargeRow).toBeTruthy();
    if (reverseChargeRow === undefined) {
      throw new Error("Agent 6 is absent from the reciprocal Charge transition.");
    }
    expect(reverseChargeRow.submitted_action).toMatchObject({
      target_action: reverseTargetAction,
      use_ultimate_action: 1,
    });
    expect(reverseChargeRow.accepted_action).toEqual(reverseChargeRow.submitted_action);

    const fixedStep = await currentStep(page);
    for (let cycle = 0; cycle < 3; cycle += 1) {
      for (const mode of /** @type {const} */ (["researcher", "pov"])) {
        await activateLiveCommand(page, () => viewButton(page, mode).click());
        await expect(viewSelect).toHaveValue(mode);
        await expect(page.locator("#step-value")).toHaveText(String(fixedStep));
        // The picked target stays Agent 1 in both views.
        await expect(targetChip).toContainText("Warrior · ID 1");
        await expect(
          battlefield.locator('.agent[data-controlled="true"]'),
        ).toHaveAttribute("aria-label", /^Agent ID 6[,.]/u);
        await expect(battlefield.locator('.agent[data-selected="true"]')).toHaveCount(
          1,
        );
        await expect(
          battlefield.locator('.agent[data-selected="true"]'),
        ).toHaveAttribute("aria-label", /^Agent ID 1[,.]/u);
        await expectHealthyInstallation();
      }
    }

    expect(endpointFailures).toEqual([]);
    expect(pageErrors).toEqual([]);
  } catch (error) {
    primaryError = error;
  }

  let cleanupError = null;
  try {
    page.off("response", recordResponse);
    page.off("pageerror", recordPageError);
    await page.unroute("**/api/command");
    if (!page.isClosed()) {
      if ((await viewSelect.inputValue()) !== "researcher") {
        await activateLiveCommand(page, () => viewButton(page, "researcher").click());
      }
      await activateLiveCommand(page, () => page.locator("#reset-button").click());
      if (initialControlledPublicId !== null) {
        await focusOwnChampion(page, initialControlledPublicId);
      }
    }
  } catch (error) {
    cleanupError = error;
  }
  if (primaryError !== null) {
    throw primaryError;
  }
  if (cleanupError !== null) {
    throw cleanupError;
  }
});

test("Escape clears the target and frees the keys; a right click clears it and keeps them", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const commands = [];
  await page.route("**/api/command", async (route) => {
    commands.push(route.request().postDataJSON()?.command ?? {});
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await waitForPlanPhase(page);
  const initialChampionId = await deckChampionId(page);
  // The Priest has Allowed Now picks from its spawn.
  await focusOwnChampion(page, PRIEST_ID);
  const battlefield = page.locator("#battlefield");
  const targetChip = page.locator("#tchip");
  const pressedPicks = page.locator(
    '#command-target-select .pick[aria-pressed="true"]',
  );
  const pickFirstTarget = async () => {
    const pick = page.locator("#command-target-select .pick").first();
    const target = await pick.getAttribute("data-target");
    if (target === null) {
      throw new Error("Allowed Now has no authorized target.");
    }
    await activateLiveCommand(page, () => pick.click());
    await expect(allowedNowPick(page, target)).toHaveAttribute("aria-pressed", "true");
    await expect(targetChip).not.toHaveClass(/\bnone\b/u);
  };

  await pickFirstTarget();
  commands.length = 0;
  await battlefield.focus();
  await page.keyboard.press("Escape");
  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({
    command_type: "keyboard",
    key: "Escape",
  });
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(targetChip).toHaveClass(/\bnone\b/u);
  await expect(pressedPicks).toHaveCount(0);
  await expect(battlefield).not.toBeFocused();

  await pickFirstTarget();
  commands.length = 0;
  // A right click on the arena clears the target as well. It takes the
  // keys, even from another control, and keeps the browser's context menu
  // away while orders are open.
  await page.locator("#help-button").focus();
  const rightClick = await battlefield.evaluate((surface) => {
    const box = surface.getBoundingClientRect();
    const point = {
      clientX: box.left + box.width / 2,
      clientY: box.top + box.height / 2,
    };
    const pointerdown = new PointerEvent("pointerdown", {
      bubbles: true,
      cancelable: true,
      button: 2,
      pointerType: "mouse",
      ...point,
    });
    const contextmenu = new MouseEvent("contextmenu", {
      bubbles: true,
      cancelable: true,
      button: 2,
      ...point,
    });
    surface.dispatchEvent(pointerdown);
    surface.dispatchEvent(contextmenu);
    return {
      pointerDefaultPrevented: pointerdown.defaultPrevented,
      contextmenuDefaultPrevented: contextmenu.defaultPrevented,
      activeElementId: document.activeElement?.id ?? null,
    };
  });
  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({
    command_type: "battlefield_pointer",
    button: "secondary",
  });
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(targetChip).toHaveClass(/\bnone\b/u);
  await expect(pressedPicks).toHaveCount(0);
  await expect(battlefield).toBeFocused();
  expect(rightClick).toEqual({
    pointerDefaultPrevented: true,
    contextmenuDefaultPrevented: true,
    activeElementId: "battlefield",
  });
  await page.unroute("**/api/command");
  await focusOwnChampion(page, initialChampionId);
});

test("Live Mage Burst retains every authorized Mage and Warrior aura modifier", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await openVisualFilters(page);
  await page.locator("#enable-all-visual-filters-button").click();
  await closeVisualFilters(page);
  await page.locator("#reset-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#step-value")).toHaveText("0");
  const stepBefore = await currentStep(page);
  await page.locator("#ultimate-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#ultimate-button")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.locator("#submit-turn-button").click();
  await expect(page.locator("#step-value")).toHaveText(String(stepBefore + 1));
  await expect(page.locator("#connection-status")).toHaveText("Connected");

  const inventories = await page.evaluate(() => {
    const identity = (/** @type {Element} */ element) =>
      element
        .closest("[data-presentation-key]")
        ?.getAttribute("data-presentation-key") ?? null;
    const factKey = (/** @type {Element} */ element) =>
      `${identity(element)}|${element.getAttribute("data-token-id")}`;
    return {
      roster: Array.from(
        document.querySelectorAll(".roster-fact-token--modifier"),
        factKey,
      ).sort(),
      battlefield: Array.from(
        document.querySelectorAll("#battlefield .modifier-cell"),
        factKey,
      ).sort(),
      controlledKey: document
        .querySelector('#battlefield .agent[data-controlled="true"]')
        ?.getAttribute("data-presentation-key"),
      suppressed: document
        .querySelector('#battlefield [data-layer="durable-status-modifier"]')
        ?.getAttribute("data-suppressed-modifier-presentation-keys"),
    };
  });
  expect(inventories.roster).toHaveLength(10);
  expect(inventories.battlefield).toEqual(inventories.roster);
  expect(inventories.suppressed).toBe("");
  expect(inventories.controlledKey).toBeTruthy();
  await expect(
    page.locator(
      `#battlefield .status-cell[data-presentation-key="${inventories.controlledKey}"]`,
    ),
  ).not.toHaveCount(0);
  // The Rune Ring on the body shows the cooldown: a dim track with lit arcs.
  const cooldownRing = page.locator(
    `#battlefield g.status-marks[data-marks-for="${inventories.controlledKey}"] [data-rune-piece="ring.track"]`,
  );
  await expect(cooldownRing).toHaveCount(1);

  await page.locator("#battlefield-shell").evaluate((shell) => {
    if (!(shell instanceof HTMLElement)) {
      throw new TypeError("Battlefield shell is unavailable.");
    }
    shell.style.width = "50px";
    shell.style.height = "70px";
    shell.style.minHeight = "0";
    const battlefield = shell.querySelector("#battlefield");
    if (!(battlefield instanceof SVGSVGElement)) {
      throw new TypeError("Battlefield is unavailable.");
    }
    battlefield.style.width = "50px";
    battlefield.style.height = "70px";
    battlefield.style.minWidth = "0";
    battlefield.style.minHeight = "0";
  });
  // A tiny arena keeps the same Rune Ring on the body; there is no separate
  // cooldown dock or fallback badge.
  await expect(cooldownRing).toHaveCount(1);
  await expect(
    page.locator('#battlefield .required-dock-fallback[data-kind="cooldown"]'),
  ).toHaveCount(0);
  await expect(page.locator("#battlefield .cooldown-cell")).toHaveCount(0);
  // The ready ring from before the Burst stays only as the tick's faded
  // "before" piece, drawn at opacity 0; no ready ring shows at rest.
  await expect(
    page.locator(
      `#battlefield g.status-marks[data-marks-for="${inventories.controlledKey}"] [data-rune-piece="ring.ready"]:not([data-beat-side="before"][opacity="0"])`,
    ),
  ).toHaveCount(0);
});

test("WASD followed immediately by Enter submits the installed draft once", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  await battlefield.focus();
  const heldDraftPresentation = await holdNextPresentation(page);
  try {
    await page.keyboard.press("w");
    await heldDraftPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(1);
    heldDraftPresentation.release();
  } finally {
    heldDraftPresentation.release();
    await heldDraftPresentation.dispose();
  }

  await expect.poll(() => requests.length).toBe(2);
  expect(String(requests[0].command.key).toLowerCase()).toBe("w");
  expect(String(requests[1].command.key).toLowerCase()).toBe("enter");
  expect(requests[1].command_id).not.toBe(requests[0].command_id);
  expect(requests[1].base_revision).toBe(requests[0].base_revision + 1);
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(battlefield).toBeFocused();
  await page.unroute("**/api/command");
});

for (const [viewLabel, viewValue] of /** @type {const} */ ([
  ["Oracle", "researcher"],
  ["Agent", "pov"],
])) {
  test(`${viewLabel} refuses next-turn WASD and Enter while a Submit is in flight`, async ({
    page,
  }) => {
    /** @type {Record<string, any>[]} */
    const requests = [];
    await page.route("**/api/command", async (route) => {
      requests.push(route.request().postDataJSON());
      await route.continue();
    });

    await page.goto(debuggerUrl);
    await expect(page.locator("#connection-status")).toHaveText("Connected");
    if (viewValue === "pov") {
      await chooseView(page, viewValue);
      await expect(page.locator("#connection-status")).toHaveText("Connected");
    }
    requests.length = 0;

    const baseStep = await currentStep(page);
    const battlefield = page.locator("#battlefield");
    const northKey = page.locator('#pad .key[data-key="w"]');
    const westKey = page.locator('#pad .key[data-key="a"]');
    await expect(northKey).not.toHaveClass(/\bshake\b/u);
    await expect(westKey).not.toHaveClass(/\bshake\b/u);
    await battlefield.focus();
    const heldSubmitPresentation = await holdNextPresentation(page);
    try {
      await page.keyboard.press("Enter");
      await heldSubmitPresentation.held;
      await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
      await expect.poll(() => requests.length).toBe(1);
      await expect(battlefield).toBeFocused();

      // Until the turn has played, the deck refuses order keys and a second
      // Lock In without a word: it shakes a refused step key and neither
      // sends nor queues anything.
      await page.keyboard.press("w");
      await expect(northKey).toHaveClass(/\bshake\b/u);
      await page.keyboard.press("Enter");
      await page.keyboard.press("a");
      await expect(westKey).toHaveClass(/\bshake\b/u);
      await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
      expect(requests).toHaveLength(1);
      heldSubmitPresentation.release();
    } finally {
      heldSubmitPresentation.release();
      await heldSubmitPresentation.dispose();
    }

    await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
    await expect(page.locator("#connection-status")).toHaveText("Connected");
    expect(requests).toHaveLength(1);
    // Nothing waited behind the Lock In: the new turn starts at Stay.
    await expect(page.locator('#pad .key[data-key="x"]')).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await expect(battlefield).toBeFocused();

    // Once the turn has played, the same keys plan and lock in the next one.
    await waitForPlanPhase(page);
    await page.keyboard.press("w");
    await expect.poll(() => requests.length).toBe(2);
    await expect(northKey).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator("#connection-status")).toHaveText("Connected");
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(3);
    expect(
      requests.map((request) => String(request.command.key).toLowerCase()),
    ).toEqual(["enter", "w", "enter"]);
    expect(requests[1].command_id).not.toBe(requests[0].command_id);
    expect(requests[2].command_id).not.toBe(requests[1].command_id);
    expect(requests[1].base_revision).toBe(requests[0].base_revision + 1);
    expect(requests[2].base_revision).toBe(requests[1].base_revision + 1);
    await expect(page.locator("#step-value")).toHaveText(String(baseStep + 2));
    await expect(page.locator("#connection-status")).toHaveText("Connected");
    await expect(battlefield).toBeFocused();
    await page.unroute("**/api/command");
    if (viewValue === "pov") {
      await chooseView(page, "researcher");
      await expect(page.locator("#connection-status")).toHaveText("Connected");
    }
    await page.locator("#reset-button").click();
    await expect(page.locator("#connection-status")).toHaveText("Connected");
    await expect(page.locator("#step-value")).toHaveText("0");
  });
}

test("queued Escape preserves its direct battlefield focus release", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  await battlefield.focus();
  const heldSubmitPresentation = await holdNextPresentation(page);
  try {
    await page.keyboard.press("Enter");
    await heldSubmitPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect.poll(() => requests.length).toBe(1);

    await page.keyboard.press("Escape");
    await expect(page.locator("#connection-status")).toHaveText("Input Queued");
    await expect.poll(() => requests.length).toBe(1);
    heldSubmitPresentation.release();
  } finally {
    heldSubmitPresentation.release();
    await heldSubmitPresentation.dispose();
  }

  await expect.poll(() => requests.length).toBe(2);
  expect(requests.map((request) => String(request.command.key).toLowerCase())).toEqual([
    "enter",
    "escape",
  ]);
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(battlefield).not.toBeFocused();
  await page.unroute("**/api/command");
  await page.locator("#reset-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#step-value")).toHaveText("0");
});

test("battlefield Space submits once and consumes busy repeats without scrolling", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const commands = [];
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    commands.push(payload?.command ?? {});
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const stepBefore = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  await battlefield.focus();
  const scrollBeforeSpace = await page.evaluate(() => {
    window.scrollTo(0, 0);
    return window.scrollY;
  });
  const heldPresentation = await holdNextPresentation(page);
  try {
    await page.keyboard.press("Space");
    await heldPresentation.held;
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "retained",
    );
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await expect(battlefield).toBeFocused();
    await expect.poll(() => commands.length).toBe(1);
    expect(commands[0]).toEqual({
      command_type: "keyboard",
      key: " ",
      shift_key: false,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    });

    await page.keyboard.press("Space");
    await page.evaluate(
      () =>
        new Promise((resolve) => {
          window.requestAnimationFrame(() => {
            window.requestAnimationFrame(resolve);
          });
        }),
    );
    expect(await page.evaluate(() => window.scrollY)).toBe(scrollBeforeSpace);
    await expect(battlefield).toBeFocused();
    await expect.poll(() => commands.length).toBe(1);
    heldPresentation.release();
  } finally {
    heldPresentation.release();
    await heldPresentation.dispose();
  }

  await expect(page.locator("#step-value")).toHaveText(String(stepBefore + 1));
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect.poll(() => commands.length).toBe(1);

  // Space on the focused How To Play button stays native: it opens the How
  // To Play card over the arena and sends nothing.
  const helpButton = page.locator("#help-button");
  await helpButton.focus();
  await page.keyboard.press("Space");
  await expect(page.locator("#htp")).toBeVisible();
  await expect.poll(() => commands.length).toBe(1);
  await page.locator("#help-close-button").click();
  await expect(page.locator("#htp")).toBeHidden();
  await page.unroute("**/api/command");
});

test("a stale tab adopts the latest ranges state without replaying its command", async ({
  context,
  page,
}) => {
  await page.goto(debuggerUrl);
  const stalePage = await context.newPage();
  // Keep this tab stale while allowing a full read after its command is rejected.
  await stalePage.route("**/api/live/snapshot", async (route) => {
    const knownIdentity = route.request().headers()["if-none-match"];
    if (knownIdentity !== undefined) {
      await route.fulfill({ status: 304, headers: { ETag: knownIdentity } });
      return;
    }
    await route.continue();
  });
  await stalePage.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(stalePage.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(stalePage.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await openVisualFilters(page);
  await openVisualFilters(stalePage);
  const ranges = page.locator("#live-ranges-button");
  const staleRanges = stalePage.locator("#live-ranges-button");
  const initialRangesPressed = await ranges.getAttribute("aria-pressed");
  expect(["true", "false"]).toContain(initialRangesPressed);
  if (initialRangesPressed === null) {
    throw new Error("Live range control did not publish pressed state.");
  }

  await ranges.click();
  await expect(ranges).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed === "true" ? "false" : "true",
  );
  await expect(staleRanges).toHaveAttribute("aria-pressed", initialRangesPressed);

  await staleRanges.click();
  await expect(stalePage.locator("#notice")).toContainText("stale");
  await expect(staleRanges).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed === "true" ? "false" : "true",
  );
  await staleRanges.click();
  await expect(staleRanges).toHaveAttribute("aria-pressed", initialRangesPressed);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("#live-ranges-button")).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed,
  );
  await stalePage.close();
});

test("a lost applied response requires GET resync and never replays submit", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-inspection-state",
    "live_editable",
  );
  const baseStep = await currentStep(page);
  let markAppliedResponseHeld = () => {};
  let releaseAppliedResponse = () => {};
  const appliedResponseHeld = new Promise((resolve) => {
    markAppliedResponseHeld = () => resolve(undefined);
  });
  const appliedResponseRelease = new Promise((resolve) => {
    releaseAppliedResponse = () => resolve(undefined);
  });
  let interceptedCommands = 0;
  let appliedStatus = 0;
  await page.route("**/api/command", async (route) => {
    interceptedCommands += 1;
    const response = await route.fetch();
    appliedStatus = response.status();
    markAppliedResponseHeld();
    await appliedResponseRelease;
    await route.abort("failed");
  });

  const battlefield = page.locator("#battlefield");
  try {
    await battlefield.focus();
    await page.keyboard.press("Enter");
    await appliedResponseHeld;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await page.keyboard.press("Escape");
    await expect(page.locator("#connection-status")).toHaveText("Input Queued");
    await expect.poll(() => interceptedCommands).toBe(1);
  } finally {
    releaseAppliedResponse();
  }
  // A command lost on the way back leaves the page without its server's
  // answer, so the pill reads Disconnected until Reconnect reads the frame.
  await expect(page.locator("#connection-status")).toHaveText("Disconnected");
  await expect(page.locator("#connection-status")).toHaveAttribute(
    "data-state",
    "offline",
  );
  await expect(battlefield).not.toBeFocused();
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "retained",
  );
  await expect(page.locator("#battlefield-empty")).toBeHidden();
  await expect(
    page.locator("#battlefield .agent[data-presentation-key]"),
  ).not.toHaveCount(0);
  await expect(page.locator("#scenario-description")).toContainText("Last confirmed");
  await expect(page.locator("#scenario-description")).toContainText(
    "reconnect required",
  );
  await expect.poll(() => appliedStatus).toBe(200);
  await battlefield.focus();
  await page.keyboard.press("Enter");
  await expect.poll(() => interceptedCommands).toBe(1);

  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await expect.poll(() => interceptedCommands).toBe(1);
  await page.unroute("**/api/command");

  await openVisualFilters(page);
  const ranges = page.locator("#live-ranges-button");
  const rangesBefore = await ranges.getAttribute("aria-pressed");
  await ranges.click();
  await expect(ranges).toHaveAttribute(
    "aria-pressed",
    rangesBefore === "true" ? "false" : "true",
  );
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
});

test("a command authorization failure clears the retained battlefield", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const retainedPresentationKey = await page
    .locator("#battlefield .agent[data-presentation-key]")
    .first()
    .getAttribute("data-presentation-key");
  if (retainedPresentationKey === null) {
    throw new Error("Installed battlefield has no presentation key.");
  }
  let markCommandHeld = () => {};
  let releaseCommand = () => {};
  const commandHeld = new Promise((resolve) => {
    markCommandHeld = () => resolve(undefined);
  });
  const commandRelease = new Promise((resolve) => {
    releaseCommand = () => resolve(undefined);
  });
  let interceptedCommands = 0;
  await page.route("**/api/command", async (route) => {
    interceptedCommands += 1;
    markCommandHeld();
    await commandRelease;
    await route.fulfill({
      status: 403,
      contentType: "application/json",
      body: JSON.stringify({ message: "Denied by browser regression proof." }),
    });
  });

  try {
    await page.locator("#battlefield").focus();
    await page.keyboard.press("w");
    await commandHeld;
    await expect(page.locator("#connection-status")).toHaveText("Command In Flight");
    await page.keyboard.press("Enter");
  } finally {
    releaseCommand();
  }
  // A refused capability takes the page offline: the pill reads Disconnected.
  await expect(page.locator("#connection-status")).toHaveText("Disconnected");
  await expect(page.locator("#connection-status")).toHaveAttribute(
    "data-state",
    "offline",
  );
  await expect.poll(() => interceptedCommands).toBe(1);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  await expect(page.locator("#battlefield .agent")).toHaveCount(0);
  await expect(
    page.locator(`[data-presentation-key="${retainedPresentationKey}"]`),
  ).toHaveCount(0);
  await expect(page.locator("#battlefield-empty")).toBeVisible();
  await page.unroute("**/api/command");
});

test("an accepted Exit cannot retain a scene after sidecar authorization fails", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  const currentFrame = await page.evaluate(async () => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    const response = await window.fetch("/api/frame", {
      headers: { "X-MARL-Debugger-Token": token ?? "" },
    });
    if (!response.ok) {
      throw new Error(`Could not obtain the live frame: HTTP ${response.status}.`);
    }
    return response.json();
  });
  await page.route(
    "**/api/command",
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          schema_version: 2,
          result: "shutdown_scheduled",
          frame: currentFrame,
        }),
      });
    },
    { times: 1 },
  );
  await page.route(
    "**/api/presentation/frame",
    async (route) => {
      await route.fulfill({
        status: 403,
        contentType: "application/json",
        body: JSON.stringify({ message: "Presentation capability denied." }),
      });
    },
    { times: 1 },
  );

  await page.locator("#exit-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Closed");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  await expect(page.locator("#battlefield .agent")).toHaveCount(0);
  await expect(page.locator("#battlefield-empty")).toBeVisible();
  await page.unroute("**/api/command");
  await page.unroute("**/api/presentation/frame");
});

test("the Class Guide changes only through explicit champion activation", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const commands = [];
  await page.route("**/api/command", async (route) => {
    commands.push(route.request().postDataJSON().command ?? {});
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await waitForPlanPhase(page);
  // Live play shows the champion in view's class details in the rail's
  // Class Guide.
  const classGuide = page.locator("#class-card");
  const guideText = () =>
    classGuide.evaluate((card) => card.textContent?.replace(/\s+/gu, " ").trim() ?? "");
  /** @param {string} publicAgentId */
  const expectGuideFor = async (publicAgentId) => {
    await expect(page.locator("#ag-meta")).toHaveText(`Agent ID ${publicAgentId}`);
    const className = await page.locator("#ag-name").textContent();
    if (className === null || className.length === 0) {
      throw new Error("The deck does not name the champion in view.");
    }
    await expect(classGuide.locator("h3")).toContainText(className);
  };
  const initialChampionId = await deckChampionId(page);
  await expect(classGuide).toBeVisible();
  await expectGuideFor(initialChampionId);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("data-accent");
  const initialGuide = await guideText();
  const baseStep = await currentStep(page);

  const obstacle = page.locator("#battlefield .obstacle[data-tooltip-owner]").first();
  await expect(obstacle).toBeVisible();
  await obstacle.hover();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await obstacle.focus();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await obstacle.click();
  await expect.poll(guideText).toBe(initialGuide);
  expect(commands).toHaveLength(0);

  const otherIds = await page
    .locator("#checklist .cl-row")
    .evaluateAll(
      (rows, initial) =>
        rows
          .map((row) => row.getAttribute("data-slot") ?? "")
          .filter((id) => id !== "" && id !== initial),
      initialChampionId,
    );
  if (otherIds.length < 5) {
    throw new Error("The live match has fewer than five other Manual champions.");
  }
  const [battlefieldId, rosterPointerId, rosterKeyboardId] = otherIds;
  const [firstTargetId, secondTargetId] = otherIds.slice(-2);
  const targetChip = page.locator("#tchip");
  /** @param {string} publicAgentId */
  const pickTargetBody = async (publicAgentId) => {
    // Shift + click on a body picks it as the target.
    const commandsBefore = commands.length;
    await activateLiveCommand(page, () =>
      page
        .locator(`#battlefield .agent[aria-label^="Agent ID ${publicAgentId}."]`)
        .click({ modifiers: ["Shift"] }),
    );
    expect(commands).toHaveLength(commandsBefore + 1);
    expect(commands.at(-1)).toMatchObject({
      command_type: "battlefield_pointer",
      button: "primary",
      shift_key: true,
    });
    await expect(targetChip).toContainText(`· ID ${publicAgentId}`);
  };

  await pickTargetBody(firstTargetId);
  await expect.poll(guideText).toBe(initialGuide);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");

  const battlefieldActor = page.locator(
    `#battlefield .agent[aria-label^="Agent ID ${battlefieldId}."]`,
  );
  await expect(battlefieldActor).toHaveCount(1);
  await activateLiveCommand(page, () => battlefieldActor.click());
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  await expectGuideFor(battlefieldId);
  const battlefieldGuide = await guideText();
  expect(battlefieldGuide).not.toBe(initialGuide);

  const retainedObstacle = page
    .locator("#battlefield .obstacle[data-tooltip-owner]")
    .first();
  await retainedObstacle.hover();
  await retainedObstacle.focus();
  await retainedObstacle.click();
  await expect.poll(guideText).toBe(battlefieldGuide);
  await pickTargetBody(secondTargetId);
  await expect.poll(guideText).toBe(battlefieldGuide);

  await activateLiveCommand(page, () =>
    page.locator(`#checklist .cl-row[data-slot="${rosterPointerId}"]`).click(),
  );
  await expectGuideFor(rosterPointerId);
  const rosterPointerGuide = await guideText();
  expect(rosterPointerGuide).not.toBe(battlefieldGuide);

  await activateLiveCommand(page, () =>
    page.locator(`#checklist .cl-row[data-slot="${rosterKeyboardId}"]`).press("Enter"),
  );
  await expectGuideFor(rosterKeyboardId);
  const rosterKeyboardGuide = await guideText();
  expect(rosterKeyboardGuide).not.toBe(rosterPointerGuide);

  const finalObstacle = page
    .locator("#battlefield .obstacle[data-tooltip-owner]")
    .first();
  await finalObstacle.focus();
  await finalObstacle.click();
  await expect.poll(guideText).toBe(rosterKeyboardGuide);
  await expect(page.locator("#step-value")).toHaveText(String(baseStep));

  await focusOwnChampion(page, initialChampionId);
  if (!(await targetChip.evaluate((chip) => chip.classList.contains("none")))) {
    await activateLiveCommand(page, () => page.locator("#clear-target-button").click());
  }
  await expect(targetChip).toHaveClass(/\bnone\b/u);
  await expect(page.locator("#step-value")).toHaveText(String(baseStep));
  await page.unroute("**/api/command");
});

test("declared System loads, accepts one turn and abandons a cancelled late reply", async ({
  page,
}) => {
  const started = await startDebugger({
    extraArgs: [
      "--offer-system",
      "counter=tests.visual_debugger_system_browser_harness:make_system",
    ],
  });
  /** @param {string} controller */
  async function configureTeamA(controller) {
    const result = await page.evaluate(async (choice) => {
      const token = sessionStorage.getItem("marl-battlegrounds.debugger-token");
      if (!token) throw new Error("Missing client token.");
      const headers = { "X-MARL-Debugger-Token": token };
      const frameResponse = await fetch("/api/frame", { headers });
      if (!frameResponse.ok) throw new Error("Could not read current frame.");
      const frame = await frameResponse.json();
      const response = await fetch("/api/command", {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({
          client_id: "browser-system-test",
          command_id: crypto.randomUUID(),
          base_revision: frame.revision,
          command: {
            command_type: "set_combat_configuration",
            team_a_controller: choice,
            team_b_controller: "manual",
            execution_information_mode: "shared_obs",
          },
        }),
      });
      if (!response.ok) throw new Error(`Configuration failed: ${response.status}`);
      return response.json();
    }, controller);
    // The Reconnect button shows only when offline or a resync is required;
    // its click handler still fetches the new configuration while hidden.
    await expect(page.locator("#reconnect-button")).toBeHidden();
    await page.locator("#reconnect-button").evaluate((button) => {
      if (!(button instanceof HTMLButtonElement)) {
        throw new TypeError("Reconnect button is unavailable.");
      }
      button.click();
    });
    return result;
  }
  page.on("pageerror", (error) => {
    throw error;
  });
  try {
    await page.goto(started.url);
    await configureTeamA("system:counter");
    await expect(page.locator("#command-deck")).toHaveAttribute(
      "data-policy-controller-read-only",
      "true",
    );
    await expect(page.locator("#system-operation-status")).toHaveText("");
    await setDisclosureOpen(page, "#command-deck", true);
    await page.locator("#submit-turn-button").click();
    await expect(page.locator("#system-operation-status")).toHaveText(
      "Policies Choosing",
    );
    await expect(page.locator("#lock-label")).toHaveText("Policies Choosing…");
    await expect(page.locator("#step-value")).toHaveText("1");
    await expect(page.locator("#system-operation-status")).toHaveText("");
    await page.locator("#submit-turn-button").click();
    await expect(page.locator("#cancel-system-button")).toBeEnabled();
    await page.locator("#cancel-system-button").click();
    await expect(page.locator("#system-operation-status")).toHaveText("");
    await expect(page.locator("#step-value")).toHaveText("1");
    // The deck's Stopped panel says a fresh launch is needed, that the
    // accepted game is kept, and that it can still be inspected and saved.
    const stopped = page.locator("#stopped");
    await expect(page.locator("#command-deck")).toHaveAttribute(
      "data-state",
      "stopped",
    );
    await expect(stopped).toBeVisible();
    await expect(stopped).toContainText("A Fresh Start Is Needed");
    await expect(stopped).toContainText("The game so far is kept.");
    await expect(page.locator("#submit-turn-button")).toBeDisabled();
    await expect(page.locator("#reset-button")).toBeDisabled();
    await expect(stopped).toContainText("You can still look around, save it, or exit.");
    await expect(page.locator("#stopped-save")).toBeVisible();
    const rejected = await configureTeamA("manual");
    expect(rejected.notice).toContain("fresh client launch");
    await expect(page.locator("#command-deck")).toHaveAttribute(
      "data-policy-controller-read-only",
      "true",
    );
    await expect(page.locator("#system-operation-status")).toHaveText("");
    await expect(page.locator("#step-value")).toHaveText("1");
  } finally {
    await stopDebugger(started.process);
  }
});
