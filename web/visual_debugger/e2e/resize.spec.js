/**
 * @file Check real browser resizing, installed authority, the exact debugger grid
 * and independently scrolling panels. The BattleClient's supported windows are
 * 1280 x 720 and larger (owner, 2 Oct 2026), so its checks run at 1440 x 900
 * and 1280 x 720. Floor Grid is off by default; once its Visual Filters switch
 * is on, the BattleClient draws a one-unit grid that is exact, the lowest
 * layer and click-transparent. Resizing keeps the installed authority, the
 * champions, the tick and keyboard focus, keeps the stage and the rail side
 * by side, and sends no command. At both sizes the rail scrolls by itself and
 * a wheel there moves nothing else; the command deck grows to its content;
 * the arena, the deck and Lock In fit one screen with no page scroll;
 * focusing the rail's last control scrolls it into view; a Guide level change
 * moves the rail's cards without scrolling the page; and the rail's cards
 * never overlap or run sideways. The Replay Viewer's dock stays whole on
 * screen, with no sideways page scroll, at 1440 x 900 and at its smallest
 * supported screen, 1366 x 768.
 */
import { expect, test } from "@playwright/test";

import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import { startReplayViewer } from "./support/replay-viewer.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";

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

/** @param {import("@playwright/test").Page} page */
async function settleResponsiveLayout(page) {
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
}

// The floor (src/looks/map-skins.js) writes each coordinate rounded to
// 0.01 px. A grid line, the floor edge's corner and the edge's size each carry
// one rounding of up to 0.005 px, so a line may sit up to 0.015 px from its
// exact whole-unit place.
const GRID_ROUNDING_PX = 0.015 + 1e-9;

// The BattleClient's two supported sizes: a common desktop window and the
// smallest supported laptop screen (owner, 2 Oct 2026; layout.css).
const BATTLE_CLIENT_VIEWPORTS = Object.freeze([
  { width: 1440, height: 900 },
  { width: 1280, height: 720 },
]);

/** @param {import("@playwright/test").Page} page */
async function responsiveSnapshot(page) {
  return page.evaluate(() => {
    // #workspace is display: contents; the page grid is #bc.
    const pageGrid = document.querySelector("#bc");
    const battlefieldPanel = document.querySelector(".battlefield-panel");
    const battlefield = document.querySelector("#battlefield");
    const hud = document.querySelector(".hud-panel");
    if (
      !(pageGrid instanceof HTMLElement) ||
      !(battlefieldPanel instanceof HTMLElement) ||
      !(battlefield instanceof SVGSVGElement) ||
      !(hud instanceof HTMLElement)
    ) {
      throw new Error("Supported resize measurement targets are unavailable.");
    }
    const battlefieldPanelBounds = battlefieldPanel.getBoundingClientRect();
    const hudBounds = hud.getBoundingClientRect();
    const viewBox = battlefield.viewBox.baseVal;
    return {
      battlefieldClientHeight: battlefield.clientHeight,
      battlefieldClientWidth: battlefield.clientWidth,
      battlefieldPanelRight: battlefieldPanelBounds.right,
      battlefieldPanelTop: battlefieldPanelBounds.top,
      documentClientWidth: document.documentElement.clientWidth,
      documentScrollWidth: document.documentElement.scrollWidth,
      gridColumns: getComputedStyle(pageGrid).gridTemplateColumns,
      hudLeft: hudBounds.left,
      hudTop: hudBounds.top,
      viewBoxHeight: viewBox.height,
      viewBoxWidth: viewBox.width,
    };
  });
}

/** @param {import("@playwright/test").Page} page */
async function debuggerGridSnapshot(page) {
  return page.locator("#battlefield").evaluate((battlefield) => {
    if (!(battlefield instanceof SVGSVGElement)) {
      throw new Error("Debugger battlefield is unavailable.");
    }
    const mapLayer = battlefield.querySelector('[data-layer="map"]');
    const obstacleLayer = battlefield.querySelector('[data-layer="obstacle"]');
    const bodyLayer = battlefield.querySelector('[data-layer="body"]');
    const boundary = battlefield.querySelector(".map-boundary");
    const vertical = [...battlefield.querySelectorAll(".map-grid-line--vertical")];
    const horizontal = [...battlefield.querySelectorAll(".map-grid-line--horizontal")];
    if (
      !(mapLayer instanceof SVGGElement) ||
      !(obstacleLayer instanceof SVGGElement) ||
      !(bodyLayer instanceof SVGGElement) ||
      !(boundary instanceof SVGRectElement) ||
      !(vertical[0] instanceof SVGLineElement)
    ) {
      throw new Error("Debugger grid layers are unavailable.");
    }
    /** @param {Element} element @param {string} name */
    const numberAttribute = (element, name) => {
      const value = Number(element.getAttribute(name));
      if (!Number.isFinite(value)) {
        throw new Error(`Grid ${name} is not finite.`);
      }
      return value;
    };
    const boundaryBox = {
      x: numberAttribute(boundary, "x"),
      y: numberAttribute(boundary, "y"),
      width: numberAttribute(boundary, "width"),
      height: numberAttribute(boundary, "height"),
    };
    const first = vertical[0];
    const midpoint = battlefield.createSVGPoint();
    midpoint.x = numberAttribute(first, "x1");
    midpoint.y = (numberAttribute(first, "y1") + numberAttribute(first, "y2")) / 2;
    const matrix = first.getScreenCTM();
    if (matrix === null) {
      throw new Error("Debugger grid has no screen transform.");
    }
    const screenMidpoint = midpoint.matrixTransform(matrix);
    const hit = document.elementFromPoint(screenMidpoint.x, screenMidpoint.y);
    /** @param {Element} layer */
    const followsMap = (layer) =>
      Boolean(
        mapLayer.compareDocumentPosition(layer) & Node.DOCUMENT_POSITION_FOLLOWING,
      );
    return {
      vertical: vertical.map((line) => ({
        x: numberAttribute(line, "x1"),
        y1: numberAttribute(line, "y1"),
        y2: numberAttribute(line, "y2"),
      })),
      horizontal: horizontal.map((line) => ({
        y: numberAttribute(line, "y1"),
        x1: numberAttribute(line, "x1"),
        x2: numberAttribute(line, "x2"),
      })),
      boundary: boundaryBox,
      mapAriaHidden: mapLayer.getAttribute("aria-hidden"),
      mapIsFirstLayer: mapLayer === battlefield.querySelector("[data-layer]"),
      mapPrecedesObstacleAndBody: followsMap(obstacleLayer) && followsMap(bodyLayer),
      lineAccessibility: [...vertical, ...horizontal].map((line) => ({
        role: line.getAttribute("role"),
        label: line.getAttribute("aria-label"),
        tabindex: line.getAttribute("tabindex"),
        pointerEvents: getComputedStyle(line).pointerEvents,
      })),
      hitPresent: hit !== null,
      hitIsGrid: Boolean(hit?.closest(".map-grid-line")),
    };
  });
}

/** @param {import("@playwright/test").Page} page
 * @param {boolean} enabled */
async function setFloorGrid(page, enabled) {
  // Live play keeps Visual Filters in a drawer that its top-bar button opens
  // and closes. A Visual Filters switch changes this page only.
  const drawer = page.locator("#visual-filters-drawer");
  const floorGrid = drawer.locator(
    'input[type="checkbox"][data-visual-filter-id="floor_grid"]',
  );
  await page.locator("#visual-filters-button").click();
  await expect(drawer).toBeVisible();
  await expect(floorGrid).toBeChecked({ checked: !enabled });
  await floorGrid.setChecked(enabled);
  await expect(floorGrid).toBeChecked({ checked: enabled });
  // Close the drawer so the grid's hit test reaches the arena.
  await page.locator("#visual-filters-button").click();
  await expect(drawer).toBeHidden();
}

/** @param {import("@playwright/test").Page} page */
async function railCardTops(page) {
  return page.evaluate(() => ({
    classCardTop:
      document.querySelector("#class-card")?.getBoundingClientRect().top ?? -1,
    documentScrollTop: document.scrollingElement?.scrollTop ?? -1,
  }));
}

/** @param {import("@playwright/test").Page} page
 * @param {{width: number, height: number}} viewport */
async function expectIndependentRailLayout(page, viewport) {
  await page.setViewportSize(viewport);
  await settleResponsiveLayout(page);
  await expect(page.locator(".bc-top")).toBeVisible();

  const initial = await page.evaluate(() => {
    const topBar = document.querySelector(".bc-top");
    const stage = document.querySelector(".battlefield-panel");
    const command = document.querySelector("#command-deck-body");
    const deck = document.querySelector("#command-deck");
    const lockIn = document.querySelector("#submit-turn-button");
    const rail = document.querySelector(".hud-panel");
    if (
      !(topBar instanceof HTMLElement) ||
      !(stage instanceof HTMLElement) ||
      !(command instanceof HTMLElement) ||
      !(deck instanceof HTMLElement) ||
      !(lockIn instanceof HTMLElement) ||
      !(rail instanceof HTMLElement)
    ) {
      throw new Error("Independent rail layout targets are unavailable.");
    }
    const topBarBounds = topBar.getBoundingClientRect();
    return {
      commandClientHeight: command.clientHeight,
      commandOverflowY: getComputedStyle(command).overflowY,
      commandScrollHeight: command.scrollHeight,
      commandScrollTop: command.scrollTop,
      deckBottom: deck.getBoundingClientRect().bottom,
      documentClientHeight: document.documentElement.clientHeight,
      documentClientWidth: document.documentElement.clientWidth,
      documentScrollHeight: document.documentElement.scrollHeight,
      documentScrollTop: document.scrollingElement?.scrollTop ?? -1,
      documentScrollWidth: document.documentElement.scrollWidth,
      lockInBottom: lockIn.getBoundingClientRect().bottom,
      railClientHeight: rail.clientHeight,
      railOverflowY: getComputedStyle(rail).overflowY,
      railScrollHeight: rail.scrollHeight,
      stageClientHeight: stage.clientHeight,
      stageOverflowY: getComputedStyle(stage).overflowY,
      stageScrollHeight: stage.scrollHeight,
      stageScrollTop: stage.scrollTop,
      topBarBottom: topBarBounds.bottom,
      topBarLeft: topBarBounds.left,
      topBarRight: topBarBounds.right,
      topBarTop: topBarBounds.top,
    };
  });
  expect(initial.topBarTop).toBeGreaterThanOrEqual(0);
  expect(initial.topBarLeft).toBeGreaterThanOrEqual(0);
  expect(initial.topBarRight).toBeLessThanOrEqual(viewport.width);
  expect(initial.topBarBottom).toBeLessThanOrEqual(viewport.height);
  // The rail (Your Turn, Class Guide, How To Play) scrolls by itself.
  expect(initial.railOverflowY).toBe("auto");
  expect(initial.railScrollHeight).toBeGreaterThan(initial.railClientHeight);
  // The command deck grows to its content and never clips it.
  expect(initial.commandOverflowY).toBe("visible");
  expect(initial.commandScrollHeight).toBeLessThanOrEqual(
    initial.commandClientHeight + 1,
  );
  // The arena, the deck and Lock In fit one screen: only the arena gets
  // shorter, the stage never scrolls and the page scrolls in no direction.
  expect(initial.stageOverflowY).toBe("visible");
  expect(initial.stageScrollHeight).toBeLessThanOrEqual(initial.stageClientHeight + 1);
  expect(initial.deckBottom).toBeLessThanOrEqual(viewport.height);
  expect(initial.lockInBottom).toBeLessThanOrEqual(viewport.height);
  expect(initial.documentScrollHeight).toBeLessThanOrEqual(
    initial.documentClientHeight,
  );
  expect(initial.documentScrollWidth).toBeLessThanOrEqual(initial.documentClientWidth);

  const rail = page.locator(".hud-panel");
  await rail.evaluate((panel) => {
    panel.scrollTop = Math.min(48, panel.scrollHeight - panel.clientHeight);
  });
  const independentlyScrolled = await page.evaluate(() => {
    const command = document.querySelector("#command-deck-body");
    const stage = document.querySelector(".battlefield-panel");
    const panel = document.querySelector(".hud-panel");
    if (
      !(command instanceof HTMLElement) ||
      !(stage instanceof HTMLElement) ||
      !(panel instanceof HTMLElement)
    ) {
      throw new Error("Rail scroll targets disappeared.");
    }
    return {
      commandScrollTop: command.scrollTop,
      documentScrollTop: document.scrollingElement?.scrollTop ?? -1,
      railScrollTop: panel.scrollTop,
      stageScrollTop: stage.scrollTop,
    };
  });
  expect(independentlyScrolled.railScrollTop).toBeGreaterThan(0);
  expect(independentlyScrolled.commandScrollTop).toBe(initial.commandScrollTop);
  expect(independentlyScrolled.stageScrollTop).toBe(initial.stageScrollTop);
  expect(independentlyScrolled.documentScrollTop).toBe(initial.documentScrollTop);

  // A wheel over the rail at its end moves nothing else.
  await rail.evaluate((panel) => {
    panel.scrollTop = panel.scrollHeight;
  });
  const railBoundary = await rail.evaluate((panel) => panel.scrollTop);
  await rail.hover();
  const beforeWheel = await page.evaluate(() => ({
    documentScrollTop: document.scrollingElement?.scrollTop ?? -1,
    stageScrollTop: document.querySelector(".battlefield-panel")?.scrollTop ?? -1,
  }));
  await page.mouse.wheel(0, 600);
  await settleResponsiveLayout(page);
  expect(await rail.evaluate((panel) => panel.scrollTop)).toBe(railBoundary);
  expect(
    await page.locator("#command-deck-body").evaluate((body) => body.scrollTop),
  ).toBe(initial.commandScrollTop);
  expect(
    await page.locator(".battlefield-panel").evaluate((stage) => stage.scrollTop),
  ).toBe(beforeWheel.stageScrollTop);
  expect(await page.evaluate(() => document.scrollingElement?.scrollTop ?? -1)).toBe(
    beforeWheel.documentScrollTop,
  );

  // From the top of the rail, focusing its last control scrolls the rail so
  // the control shows inside it.
  await rail.evaluate((panel) => {
    panel.scrollTop = 0;
  });
  const lastRailControl = page.locator(".hud-panel button:visible").last();
  await lastRailControl.focus();
  await expect(lastRailControl).toBeFocused();
  const focusedBounds = await page.evaluate(() => {
    const panel = document.querySelector(".hud-panel");
    const active = document.activeElement;
    if (!(panel instanceof HTMLElement) || !(active instanceof HTMLElement)) {
      throw new Error("Focused rail control is unavailable.");
    }
    const panelBounds = panel.getBoundingClientRect();
    const activeBounds = active.getBoundingClientRect();
    return {
      activeBottom: activeBounds.bottom,
      activeTop: activeBounds.top,
      panelBottom: panelBounds.bottom,
      panelScrollTop: panel.scrollTop,
      panelTop: panelBounds.top,
    };
  });
  expect(focusedBounds.panelScrollTop).toBeGreaterThan(0);
  expect(focusedBounds.activeTop).toBeGreaterThanOrEqual(focusedBounds.panelTop - 1);
  expect(focusedBounds.activeBottom).toBeLessThanOrEqual(focusedBounds.panelBottom + 1);

  // The rail has no disclosures to close. Its Guide level is the closest
  // change of a card's height: Off hides Do This Now, so the Class Guide
  // below moves up, and a keyboard press never scrolls the page.
  await rail.evaluate((panel) => {
    panel.scrollTop = 0;
  });
  const guide = page.locator('#guide-level button[data-level="guide"]');
  const off = page.locator('#guide-level button[data-level="off"]');
  await expect(guide).toHaveAttribute("aria-pressed", "true");
  const withGuide = await railCardTops(page);
  await off.focus();
  await off.press("Space");
  await expect(off).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#do-this")).toBeHidden();
  await settleResponsiveLayout(page);
  const withoutGuide = await railCardTops(page);
  expect(withoutGuide.classCardTop).toBeLessThan(withGuide.classCardTop);
  expect(withoutGuide.documentScrollTop).toBe(withGuide.documentScrollTop);
  await guide.focus();
  await guide.press("Space");
  await expect(guide).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#do-this")).toBeVisible();
  await settleResponsiveLayout(page);
  expect(await railCardTops(page)).toEqual(withGuide);

  const containment = await page.evaluate(() => {
    const panel = document.querySelector(".hud-panel");
    if (!(panel instanceof HTMLElement)) {
      throw new Error("Rail is unavailable.");
    }
    const panelBounds = panel.getBoundingClientRect();
    return {
      cards: [...panel.children]
        .filter((card) => card.getClientRects().length > 0)
        .map((card) => {
          const bounds = card.getBoundingClientRect();
          return {
            bottom: bounds.bottom,
            clientWidth: card.clientWidth,
            left: bounds.left,
            right: bounds.right,
            scrollWidth: card.scrollWidth,
            top: bounds.top,
          };
        })
        .sort((upper, lower) => upper.top - lower.top),
      panelLeft: panelBounds.left,
      panelRight: panelBounds.right,
    };
  });
  expect(containment.cards.length).toBeGreaterThan(1);
  for (let index = 1; index < containment.cards.length; index += 1) {
    expect(containment.cards[index].top).toBeGreaterThanOrEqual(
      containment.cards[index - 1].bottom - 1,
    );
  }
  for (const card of containment.cards) {
    expect(card.left).toBeGreaterThanOrEqual(containment.panelLeft - 1);
    expect(card.right).toBeLessThanOrEqual(containment.panelRight + 1);
    expect(card.scrollWidth).toBeLessThanOrEqual(card.clientWidth + 1);
  }
}

test("real Debugger grid is exact, lowest, and click-transparent", async ({ page }) => {
  let commandRequests = 0;
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/command"
    ) {
      commandRequests += 1;
    }
  });
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const stepBefore = await page.locator("#step-value").textContent();
  const transitionBefore = await page.locator("#transition-value").textContent();

  // Floor Grid is off by default (owner, 4 Oct 2026): no grid line until its
  // switch is on.
  const gridLines = page.locator("#battlefield .map-grid-line");
  await expect(gridLines).toHaveCount(0);
  await setFloorGrid(page, true);
  await expect(gridLines).toHaveCount(19 + 9);

  for (const viewport of BATTLE_CLIENT_VIEWPORTS) {
    await page.setViewportSize(viewport);
    await settleResponsiveLayout(page);
    await page.locator("#battlefield").scrollIntoViewIfNeeded();
    const snapshot = await debuggerGridSnapshot(page);
    expect(snapshot.vertical).toHaveLength(19);
    expect(snapshot.horizontal).toHaveLength(9);
    expect(snapshot.mapAriaHidden).toBe("true");
    expect(snapshot.mapIsFirstLayer).toBe(true);
    expect(snapshot.mapPrecedesObstacleAndBody).toBe(true);
    expect(snapshot.hitPresent).toBe(true);
    expect(snapshot.hitIsGrid).toBe(false);
    expect(
      snapshot.lineAccessibility.every(
        ({ role, label, tabindex, pointerEvents }) =>
          role === null &&
          label === null &&
          tabindex === null &&
          pointerEvents === "none",
      ),
    ).toBe(true);
    // Each line sits on a whole world unit and spans the floor edge to edge.
    /** @param {number} actual @param {number} expected */
    const expectOnGrid = (actual, expected) =>
      expect(Math.abs(actual - expected)).toBeLessThanOrEqual(GRID_ROUNDING_PX);
    const { x, y, width, height } = snapshot.boundary;
    for (const [index, line] of snapshot.vertical.entries()) {
      expectOnGrid(line.x, x + (width * (index + 1)) / 20);
      const [low, high] = [line.y1, line.y2].sort((left, right) => left - right);
      expectOnGrid(low, y);
      expectOnGrid(high, y + height);
    }
    for (const [index, line] of snapshot.horizontal
      .sort((left, right) => left.y - right.y)
      .entries()) {
      expectOnGrid(line.y, y + (height * (index + 1)) / 10);
      const [low, high] = [line.x1, line.x2].sort((left, right) => left - right);
      expectOnGrid(low, x);
      expectOnGrid(high, x + width);
    }
  }
  await expect(page.locator("#step-value")).toHaveText(stepBefore ?? "");
  await expect(page.locator("#transition-value")).toHaveText(transitionBefore ?? "");
  expect(commandRequests).toBe(0);
});

test("supported resize preserves real installed authority at 1280 x 720", async ({
  page,
}) => {
  let commandRequests = 0;
  const authorityGets = new Set();
  page.on("request", (request) => {
    const pathname = new URL(request.url()).pathname;
    if (
      request.method() === "GET" &&
      ["/api/frame", "/api/presentation/frame", "/api/live/snapshot"].includes(pathname)
    ) {
      authorityGets.add(pathname);
    }
    if (request.method() === "POST" && pathname === "/api/command") {
      commandRequests += 1;
    }
  });

  const [largerViewport, smallestViewport] = BATTLE_CLIENT_VIEWPORTS;
  await page.setViewportSize(largerViewport);
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await settleResponsiveLayout(page);
  expect([...authorityGets].sort()).toEqual(["/api/live/snapshot"]);

  const agentKeysBefore = await page
    .locator("#battlefield .agent")
    .evaluateAll((agents) =>
      agents.map((agent) => agent.getAttribute("data-presentation-key")),
    );
  expect(agentKeysBefore.length).toBeGreaterThan(0);
  expect(
    agentKeysBefore.every((key) => typeof key === "string" && key.length > 0),
  ).toBe(true);
  const stepBefore = await page.locator("#step-value").textContent();
  const transitionBefore = await page.locator("#transition-value").textContent();
  const initial = await responsiveSnapshot(page);
  expect(Math.abs(initial.viewBoxWidth - initial.battlefieldClientWidth)).toBeLessThan(
    1,
  );
  expect(
    Math.abs(initial.viewBoxHeight - initial.battlefieldClientHeight),
  ).toBeLessThan(1);

  // The BattleClient has no Roster panel; the deck's Next champion button is
  // its control for choosing which champion to command.
  const focusedControl = page.locator("#command-deck #next-actor-button");
  await expect(focusedControl).toBeVisible();
  await expect(focusedControl).toBeEnabled();
  await focusedControl.focus();
  await expect(focusedControl).toBeFocused();

  await page.setViewportSize(smallestViewport);
  await settleResponsiveLayout(page);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const resized = await responsiveSnapshot(page);
  expect(resized.documentScrollWidth).toBeLessThanOrEqual(resized.documentClientWidth);
  expect(resized.gridColumns.split(" ")).toHaveLength(2);
  expect(resized.hudLeft).toBeGreaterThanOrEqual(resized.battlefieldPanelRight);
  expect(Math.abs(resized.hudTop - resized.battlefieldPanelTop)).toBeLessThan(2);
  expect(Math.abs(resized.viewBoxWidth - resized.battlefieldClientWidth)).toBeLessThan(
    1,
  );
  expect(
    Math.abs(resized.viewBoxHeight - resized.battlefieldClientHeight),
  ).toBeLessThan(1);
  expect(
    Math.abs(resized.battlefieldClientWidth - initial.battlefieldClientWidth) > 1 ||
      Math.abs(resized.battlefieldClientHeight - initial.battlefieldClientHeight) > 1,
  ).toBe(true);

  expect(
    await page
      .locator("#battlefield .agent")
      .evaluateAll((agents) =>
        agents.map((agent) => agent.getAttribute("data-presentation-key")),
      ),
  ).toEqual(agentKeysBefore);
  await expect(page.locator("#step-value")).toHaveText(stepBefore ?? "");
  await expect(page.locator("#transition-value")).toHaveText(transitionBefore ?? "");
  await expect(focusedControl).toBeFocused();
  expect(commandRequests).toBe(0);
});

test("panels scroll independently at both supported viewports", async ({ page }) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Connected");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  for (const viewport of BATTLE_CLIENT_VIEWPORTS) {
    await expectIndependentRailLayout(page, viewport);
  }

  const replay = await startReplayViewer({
    sampleReplay: "death-respawn-shield",
  });
  try {
    // Owner, 2 Oct 2026: the Replay Viewer's smallest supported screen is
    // 1366 x 768; below it the page may run off the screen.
    for (const viewport of [
      { width: 1440, height: 900 },
      { width: 1366, height: 768 },
    ]) {
      await page.setViewportSize(viewport);
      await page.goto(replay.url);
      await expect(page.locator("#connection-status")).toHaveText("Connected");
      await expect(page.locator("html")).toHaveAttribute(
        "data-presentation-authority",
        "installed",
      );
      await settleResponsiveLayout(page);
      await expect(page.locator(".replay-timeline__transport")).toBeVisible();
      const transport = await page
        .locator(".replay-timeline__transport")
        .evaluate((element) => {
          const bounds = element.getBoundingClientRect();
          return {
            bottom: bounds.bottom,
            clientWidth: element.clientWidth,
            left: bounds.left,
            right: bounds.right,
            scrollWidth: element.scrollWidth,
            top: bounds.top,
          };
        });
      expect(transport.top).toBeGreaterThanOrEqual(0);
      expect(transport.left).toBeGreaterThanOrEqual(0);
      expect(transport.right).toBeLessThanOrEqual(viewport.width);
      expect(transport.bottom).toBeLessThanOrEqual(viewport.height);
      expect(transport.scrollWidth).toBeLessThanOrEqual(transport.clientWidth + 1);
      expect(
        await page.evaluate(
          () =>
            document.documentElement.scrollWidth <=
            document.documentElement.clientWidth,
        ),
      ).toBe(true);
    }
  } finally {
    await stopDebugger(replay.process);
  }
});
