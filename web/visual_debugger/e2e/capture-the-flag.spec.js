/**
 * @file Check Capture the Flag in the real Replay Viewer on the hand-built story
 * replay V5 (map 42, Team A's Mage and Warrior against Team B's Rogue, a 2-tick
 * return, every flag state once). Oracle View at frame 1: the two flag layers sit
 * under and over the bodies; Team A's dropped flag lies on the ground layer with
 * its countdown ring and "2" left; Team B's flag rides outlined on its carrier,
 * following its glide; the scorebug reads "/ 1" and "First to 1" with one flag
 * chip per team ("Dropped, 2 ticks left", "Carried by a Mage"); the place line
 * names Capture the Flag; the Visual Filters list Flags and leave out the two Red
 * Zone switches. At frame 6, the winning capture: the capture mark at Team A's
 * base and the Verdict with Show Arena alone, and no Match Summary key. Agent POV
 * of Team B's Rogue at frame 1: its flag, carried by a champion it cannot see,
 * is a plain pennant that follows no body, and its chip names no carrier. This
 * spec sets no time limits; the Playwright configuration waits without limit.
 */
import { expect, test } from "@playwright/test";
import {
  exportCaptureTheFlagReplayArtifacts,
  removeReplayArtifacts,
  startReplayViewer,
  stopDebugger,
} from "./support/replay-viewer.js";

/** @type {Awaited<ReturnType<typeof exportCaptureTheFlagReplayArtifacts>> | null} */
let artifacts = null;

test.beforeAll(async () => {
  artifacts = await exportCaptureTheFlagReplayArtifacts();
});

test.afterAll(async () => {
  await removeReplayArtifacts(artifacts?.outputDirectory);
});

/**
 * Open the story in a fresh Replay Viewer at one frame and run check on it.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, unknown>} options startReplayViewer's options, less the replay.
 * @param {() => Promise<void>} check
 */
async function onFrame(page, options, check) {
  if (artifacts === null) throw new TypeError("The story replays were not exported.");
  /** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
  let viewer = null;
  try {
    viewer = await startReplayViewer({ replayPath: artifacts.noSharedObs, ...options });
    await page.goto(viewer.url);
    await expect(page.locator("#connection-status")).toHaveText("Connected");
    await expect(page.locator("g[data-layer='flags-ground'] .flag-base")).toHaveCount(
      2,
    );
    await check();
  } finally {
    await stopDebugger(viewer?.process ?? null);
  }
}

test("Oracle View draws both flags, their bases, chips and the Flags switch", async ({
  page,
}) => {
  await onFrame(page, { frameIndex: 1 }, async () => {
    const layers = await page
      .locator("g.battlefield-camera > g[data-layer]")
      .evaluateAll((groups) => groups.map((group) => group.getAttribute("data-layer")));
    const at = (/** @type {string} */ name) => layers.indexOf(name);
    expect(at("strike-band")).toBeLessThan(at("flags-ground"));
    expect(at("flags-ground")).toBeLessThan(at("body"));
    expect(at("over")).toBeLessThan(at("flags-above"));

    const ground = page.locator("g[data-layer='flags-ground']");
    await expect(ground.locator(".flag-pennant[data-state='dropped']")).toHaveCount(1);
    await expect(ground.locator(".flag-countdown")).toHaveCount(1);
    const carried = page.locator(
      "g[data-layer='flags-above'] .flag-pennant[data-state='carried']",
    );
    await expect(carried).toHaveCount(1);
    await expect(carried).toHaveClass(/flag-pennant--marked/u);
    await expect(carried).toHaveAttribute("data-glide-follow", /.+/u);
    await expect(page.locator("g[data-layer='flags-above'] .flag-ticks")).toHaveText(
      "2",
    );

    await expect(page.locator("#scorebug .sb-score.a .of")).toHaveText("/ 1");
    await expect(page.locator("#scorebug .sb-mid span:not(.wave)")).toHaveText(
      "First to 1",
    );
    await expect(
      page.locator("#scorebug .sb-score.a .sb-flag").first(),
    ).toHaveAttribute("aria-label", "Team A's Flag: Dropped, 2 ticks left");
    await expect(
      page.locator("#scorebug .sb-score.b .sb-flag").first(),
    ).toHaveAttribute("aria-label", "Team B's Flag: Carried by a Mage");
    await expect(page.locator(".bc-top")).toContainText("Capture the Flag");

    const listed = (/** @type {string} */ id) =>
      page
        .locator(`[data-visual-filter-option="${id}"]`)
        .evaluate((row) => !(/** @type {HTMLElement} */ (row).hidden));
    expect(await listed("flags")).toBe(true);
    expect(await listed("red_zone_floors")).toBe(false);
    expect(await listed("red_zone_home_fall")).toBe(false);
  });
});

test("the winning capture shows its mark and a one-button Verdict", async ({
  page,
}) => {
  await onFrame(page, { frameIndex: 6 }, async () => {
    await expect(
      page.locator("g[data-layer='flags-above'] .flag-mark--team-a"),
    ).toHaveCount(1);
    await expect(
      page.locator("g[data-layer='flags-ground'] .flag-pennant[data-state='home']"),
    ).toHaveCount(2);
    await expect(page.locator("button[data-verdict-action]")).toHaveCount(1);
    await expect(page.locator("button[data-verdict-action]")).toHaveAttribute(
      "data-verdict-action",
      "show_arena",
    );
    await expect(page.locator("#arena-match-summary-button")).toBeHidden();
  });
});

test("Agent POV never names a carrier its team cannot see", async ({ page }) => {
  await onFrame(page, { frameIndex: 1, view: "pov", povSlot: 5 }, async () => {
    const carried = page.locator(
      "g[data-layer='flags-above'] .flag-pennant[data-state='carried']",
    );
    await expect(carried).toHaveCount(1);
    await expect(carried).not.toHaveClass(/flag-pennant--marked/u);
    await expect(carried).not.toHaveAttribute("data-glide-follow", /.*/u);
    const chip = page.locator("#scorebug .sb-score.b .sb-flag").first();
    await expect(chip).toHaveAttribute("aria-label", "Team B's Flag: Carried");
    await expect(chip.locator(".sb-flag__seal")).toHaveCount(0);
  });
});
