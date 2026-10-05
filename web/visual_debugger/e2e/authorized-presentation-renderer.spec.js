/**
 * @file Check the shared battlefield renderer and event painter on real
 * authorized fixture frames, in a bare page without the client shell.
 *
 * - Authority: only authorized ranges are drawn, and raw options cannot add
 *   one. The Replay Viewer's Ranges button shows every authorized ring; with it
 *   off, the Reach Rings switch keeps only the selected champion's rings. Bodies
 *   carry opaque presentation keys and no slot attributes. Raw or forged scenes
 *   paint nothing.
 * - Floor: live play's one-unit grid is the lowest paint, meets the floor edge
 *   within rounding, ignores the pointer and is absent from replays.
 * - Red Zone: one Ash floor path holds one subpath per recorded team strip,
 *   clipped to the floor. expectedRedZoneStrips is the independent copy of that
 *   rule (a strip no wider than 1e-6 units is left out). Each rule edge gets a
 *   border. The Red Zone lies over the floor and grid and under the stones and
 *   bodies; off, one plain line marks each rule edge. Stones keep their look and
 *   hit testing, and the Ash corners follow the floor's own corner radius. The
 *   test attaches a screenshot of a depth-6 Oracle with a wall and a pillar on
 *   the strip edges for a person to inspect.
 * - Switches: Target Selection Visuals hides only the selection brackets, and
 *   no lane pill is painted. Each durable switch (aura fields, modifier and
 *   status badges, Spawn Shield, the Rune Ring cooldown, Red Zone Floors)
 *   removes only its own paint and keeps every body, its identity and label.
 *   The old cooldown docks never return.
 * - Tips: an aura tip names its source with the champion label ("Mage · ID 0"),
 *   and no hidden source key reaches the DOM. In all three Agent leaves a status
 *   tip reads the Champion Card's words (source by class only, then one row,
 *   "Time Left") and never names the recipient or a source by ID.
 * - Events: a Basic stroke runs from 2 px outside the striker's rim to just
 *   outside the target's rim and ends in its class landing (a Priest's heal
 *   blooms, a Mage's blow lands in a star), outside the body and without
 *   catching the pointer at the body's center. Charge stays feed-only.
 *   Regeneration numbers sit on their planned cues. A cooldown start paints
 *   nothing. Crippling Poison draws its stiletto and one venom seep. A broken
 *   Freezing Trap shows on the frozen token (crack at Impact, six shards at
 *   Settle). Falls and returns center on their bodies and end on their still
 *   picture in every motion mode. Team waves and the death card stay off the
 *   arena. A merged status application paints no off-body badge.
 * - Layout: a champion's body, with the Replay Viewer's Champion Card, wins
 *   pointer arbitration over an accepted route and its Spawn Pad. The Rune Ring
 *   shows the Ultimate cooldown on the body at every arena size. Living bodies
 *   paint over corpses. The plan and the score strip keep each team's recorded
 *   side and color.
 */
import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";
import { MINIMUM_VIEWPORT } from "./support/visual-regression.js";

const WEB_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SOURCE_ROOT = resolve(WEB_ROOT, "src");
const fixture = JSON.parse(
  readFileSync(
    new URL("../tests/fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @type {import("node:http").Server | null} */
let server = null;
let origin = "";

test.beforeAll(async () => {
  server = createServer(async (request, response) => {
    try {
      const pathname = new URL(request.url ?? "/", "http://127.0.0.1").pathname;
      if (pathname === "/") {
        response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        response.end(`<!doctype html>
          <html><head><link rel="stylesheet" href="/styles.css"></head><body>
            <svg id="battlefield" class="battlefield" style="width: 800px; height: 600px"></svg>
            <p id="empty"></p>
            <aside id="visual-tooltip" class="visual-tooltip" hidden>
              <strong id="visual-tooltip-title"></strong>
              <div id="visual-tooltip-details"></div>
            </aside>
          </body></html>`);
        return;
      }
      if (pathname === "/styles.css") {
        response.writeHead(200, { "content-type": "text/css; charset=utf-8" });
        response.end(await readFile(resolve(WEB_ROOT, "styles.css")));
        return;
      }
      const path = resolve(WEB_ROOT, `.${pathname}`);
      // styles.css imports the part styles under src/styles, so serve them too.
      if (path.startsWith(`${SOURCE_ROOT}${sep}`) && extname(path) === ".css") {
        response.writeHead(200, { "content-type": "text/css; charset=utf-8" });
        response.end(await readFile(path));
        return;
      }
      if (!path.startsWith(`${SOURCE_ROOT}${sep}`) || extname(path) !== ".js") {
        response.writeHead(404).end();
        return;
      }
      response.writeHead(200, { "content-type": "text/javascript; charset=utf-8" });
      response.end(await readFile(path));
    } catch (error) {
      response.writeHead(500).end(String(error));
    }
  });
  await new Promise((resolveListen, rejectListen) => {
    server?.once("error", rejectListen);
    server?.listen(0, "127.0.0.1", () => resolveListen(undefined));
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("Authorized presentation test server has no TCP address.");
  }
  origin = `http://127.0.0.1:${address.port}`;
});

test.afterAll(async () => {
  if (server === null) {
    return;
  }
  await new Promise((resolveClose, rejectClose) => {
    server?.close((error) => (error ? rejectClose(error) : resolveClose(undefined)));
  });
});

/** @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} rawPresentation
 * @param {boolean} showRanges
 * @param {boolean} reachRings */
async function renderPresentation(page, rawPresentation, showRanges, reachRings) {
  await page.goto(origin);
  return await page.evaluate(
    async ({ rawPresentation: raw, showRanges: visible, reachRings: ringsOn }) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const {
        DEFAULT_VISUAL_FILTER_STATE,
        enableAllVisualFilters,
        setVisualFilterEnabled,
      } = await import(`${moduleRoot}/visual-filters.js`);
      const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
      const { authorizedPresentationSceneView } = await import(
        `${moduleRoot}/authorized-presentation-adapter.js`
      );
      const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
      const authorizedScene = authorizedPresentationSceneView(presentation);
      const authorizedRanges = Array.isArray(authorizedScene?.ranges)
        ? authorizedScene.ranges.filter(
            (/** @type {Record<string, any>} */ range) =>
              typeof range.radius === "number" && range.radius > 0,
          )
        : [];
      const selectedKey = authorizedScene?.selection?.selected_presentation_key ?? null;
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Renderer test surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      const painted = renderer.render(presentation, {
        showRanges: visible,
        visualFilterState: setVisualFilterEnabled(
          allVisualFilters,
          "reach_rings",
          ringsOn,
        ),
        // Deliberately ignored: transport may control visibility, never geometry.
        ranges: [{ center: [999, 999], radius: 999, global_slot: 999 }],
      });
      const rangeNodes = Array.from(
        battlefield.querySelectorAll('[data-layer="debug-range"] .range-ring'),
      );
      const bodyNodes = Array.from(battlefield.querySelectorAll(".agent"));
      const slotAttributes = Array.from(battlefield.querySelectorAll("*"))
        .flatMap((element) => Array.from(element.attributes))
        .map((attribute) => attribute.name)
        .filter(
          (name) =>
            name === "data-slot" ||
            name === "data-source-slot" ||
            name === "data-target-slot" ||
            name === "data-recipient-slot" ||
            name === "data-actor-slot" ||
            name.endsWith("-slots"),
        );
      return {
        painted,
        audience: presentation.presentation_kind,
        authorizedRangeCount: authorizedRanges.length,
        authorizedRangeKeys: authorizedRanges.map(
          (/** @type {Record<string, any>} */ range) => range.presentation_key,
        ),
        selectedRangeKeys: authorizedRanges
          .filter(
            (/** @type {Record<string, any>} */ range) =>
              range.presentation_key === selectedKey,
          )
          .map((/** @type {Record<string, any>} */ range) => range.presentation_key),
        inspectionOwner: presentation.inspection?.actor_presentation_key ?? null,
        rangeCount: rangeNodes.length,
        rangeKeys: rangeNodes.map((node) => node.getAttribute("data-presentation-key")),
        bodyCount: bodyNodes.length,
        bodyKeys: bodyNodes.map((node) => node.getAttribute("data-presentation-key")),
        bodySlots: bodyNodes.map((node) => node.getAttribute("data-slot")),
        slotAttributes,
      };
    },
    { rawPresentation, showRanges, reachRings },
  );
}

// The Replay Viewer's Ranges button (showRanges) draws every authorized ring.
// With it off, the Reach Rings switch keeps only the selected champion's rings.
test("replay Oracle range preference changes visibility but not authority", async ({
  page,
}) => {
  const raw = fixture.presentations.replay_oracle;
  const hidden = await renderPresentation(page, raw, false, false);
  const selectedOnly = await renderPresentation(page, raw, false, true);
  const visible = await renderPresentation(page, raw, true, true);
  const visibleWithoutSwitch = await renderPresentation(page, raw, true, false);
  expect(hidden.painted).toBe(true);
  expect(hidden.rangeCount).toBe(0);
  expect(selectedOnly.selectedRangeKeys.length).toBeGreaterThan(0);
  expect(selectedOnly.rangeKeys).toEqual(selectedOnly.selectedRangeKeys);
  expect(visible.rangeCount).toBe(visible.authorizedRangeCount);
  expect(visible.rangeKeys).toEqual(visible.authorizedRangeKeys);
  expect(visibleWithoutSwitch.rangeKeys).toEqual(
    visibleWithoutSwitch.authorizedRangeKeys,
  );
});

test("Agent DOM identity remains opaque and raw options cannot mint ranges", async ({
  page,
}) => {
  const raw = fixture.presentations.replay_shared_obs_agent_pov;
  const hidden = await renderPresentation(page, raw, false, false);
  const selectedOnly = await renderPresentation(page, raw, false, true);
  const visible = await renderPresentation(page, raw, true, true);
  expect(hidden.rangeCount).toBe(0);
  expect(hidden.slotAttributes).toEqual([]);
  expect(selectedOnly.selectedRangeKeys.length).toBeGreaterThan(0);
  expect(selectedOnly.rangeKeys).toEqual(selectedOnly.selectedRangeKeys);
  expect(selectedOnly.slotAttributes).toEqual([]);
  expect(visible.rangeCount).toBe(visible.authorizedRangeCount);
  expect(visible.rangeKeys).toEqual(visible.authorizedRangeKeys);
  expect(visible.bodyCount).toBeGreaterThan(0);
  expect(
    visible.bodyKeys.every((key) => typeof key === "string" && key.length > 0),
  ).toBe(true);
  expect(visible.bodySlots).toEqual(Array(visible.bodyCount).fill(null));
  expect(visible.slotAttributes).toEqual([]);
});

test("Debugger unit grid is lowest, exact, click-transparent, and absent from replay", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(
    async ({ liveRaw, replayRaw }) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
        `${moduleRoot}/visual-filters.js`
      );
      const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Grid test surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      const live = await normalizeAuthorizedPresentationFrameV1(liveRaw);
      const replay = await normalizeAuthorizedPresentationFrameV1(replayRaw);
      const liveRows = [];
      for (const viewport of [
        { width: 960, height: 600 },
        { width: 1440, height: 900 },
      ]) {
        battlefield.style.width = `${viewport.width}px`;
        battlefield.style.height = `${viewport.height}px`;
        renderer.render(live, {
          showRanges: true,
          visualFilterState: allVisualFilters,
        });
        const mapLayer = battlefield.querySelector('[data-layer="map"]');
        const boundary = battlefield.querySelector(".map-boundary");
        const firstVertical = battlefield.querySelector(".map-grid-line--vertical");
        if (
          !(mapLayer instanceof SVGGElement) ||
          !(boundary instanceof SVGRectElement)
        ) {
          throw new Error("Debugger map layer is unavailable.");
        }
        liveRows.push({
          viewport,
          vertical: battlefield.querySelectorAll(".map-grid-line--vertical").length,
          horizontal: battlefield.querySelectorAll(".map-grid-line--horizontal").length,
          mapAriaHidden: mapLayer.getAttribute("aria-hidden"),
          mapFirst: mapLayer === battlefield.querySelector("[data-layer]"),
          pointerEvents:
            firstVertical instanceof SVGLineElement
              ? getComputedStyle(firstVertical).pointerEvents
              : null,
          boundary: {
            x: boundary.getAttribute("x"),
            y: boundary.getAttribute("y"),
            width: boundary.getAttribute("width"),
            height: boundary.getAttribute("height"),
          },
          firstVertical: firstVertical
            ? {
                x1: firstVertical.getAttribute("x1"),
                y1: firstVertical.getAttribute("y1"),
                x2: firstVertical.getAttribute("x2"),
                y2: firstVertical.getAttribute("y2"),
              }
            : null,
        });
      }
      renderer.render(replay, {
        showRanges: true,
        visualFilterState: allVisualFilters,
      });
      return {
        liveRows,
        replayGridCount: battlefield.querySelectorAll(".map-grid-line").length,
      };
    },
    {
      liveRaw: fixture.pairs.live_oracle.presentation,
      replayRaw: fixture.pairs.replay_oracle.presentation,
    },
  );
  // Each SVG attribute is rounded to 0.01 px on its own. The floor's bottom edge
  // is y plus height, so a line end may differ from it by three such roundings.
  const ROUNDING_PX = 0.015 + 1e-9;
  expect(result.liveRows).toHaveLength(2);
  for (const row of result.liveRows) {
    expect(row.vertical).toBe(19);
    expect(row.horizontal).toBe(9);
    expect(row.mapAriaHidden).toBe("true");
    expect(row.mapFirst).toBe(true);
    expect(row.pointerEvents).toBe("none");
    expect(row.firstVertical).not.toBeNull();
    const [lineTop, lineBottom] = [
      Number(row.firstVertical?.y1),
      Number(row.firstVertical?.y2),
    ].sort((left, right) => left - right);
    const floorTop = Number(row.boundary.y);
    const floorBottom = floorTop + Number(row.boundary.height);
    expect(Math.abs(lineTop - floorTop)).toBeLessThanOrEqual(ROUNDING_PX);
    expect(Math.abs(lineBottom - floorBottom)).toBeLessThanOrEqual(ROUNDING_PX);
    expect(Number(row.firstVertical?.x1)).toBeGreaterThan(Number(row.boundary.x));
    expect(row.firstVertical?.x1).toBe(row.firstVertical?.x2);
  }
  expect(result.replayGridCount).toBe(0);
});

/**
 * @param {number} width
 * @param {Record<string, any> | null} redZone
 * @returns {Array<{teamId: number, low: number, high: number, atLeft: boolean, atRight: boolean, inner: number | null}>}
 */
function expectedRedZoneStrips(width, redZone) {
  if (redZone === null) {
    return [];
  }
  // A strip no wider than this many world units is left out.
  const collapsed = 1e-6;
  /** @type {Array<{teamId: number, range: number[]}>} */
  const recorded = [
    { teamId: 1, range: redZone.team_a_x_range },
    { teamId: 2, range: redZone.team_b_x_range },
  ];
  return recorded
    .map(({ teamId, range: [from, to] }) => ({
      teamId,
      low: Math.min(Math.max(Math.min(from, to), 0), width),
      high: Math.min(Math.max(Math.max(from, to), 0), width),
    }))
    .filter(({ low, high }) => high - low > collapsed)
    .map((strip) => {
      const atLeft = strip.low <= collapsed;
      const atRight = strip.high >= width - collapsed;
      return {
        ...strip,
        atLeft,
        atRight,
        inner: atLeft && !atRight ? strip.high : atRight && !atLeft ? strip.low : null,
      };
    });
}

test("Red Zone Floors paint exact recorded strips beneath obstacles", async ({
  page,
}, testInfo) => {
  const caseNames = [
    "live_oracle_depth_6",
    "live_oracle_depth_5_5_swapped_banks",
    "live_oracle_overlapping_strips",
    "live_oracle_full-width_strips",
    "live_oracle_both_banks_on_the_left",
    "live_oracle_depth_0_1_on_width_17_3",
    "live_oracle_12_x_12_map",
    "live_oracle_depth_0",
    "live_oracle_depth_1e-8_on_width_20",
    "live_oracle_depth_12_1_on_width_12_1",
    "live_oracle_depth_1e-8_on_width_12_1",
    "live_oracle_depth_6_obstacles_on_strip_edges",
    "replay_oracle_depth_5",
  ];
  await page.goto(origin);
  const result = await page.evaluate(
    async ({ cases, legacyRaw }) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const {
        DEFAULT_VISUAL_FILTER_STATE,
        enableAllVisualFilters,
        setVisualFilterEnabled,
      } = await import(`${moduleRoot}/visual-filters.js`);
      const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
      const tintOff = setVisualFilterEnabled(
        allVisualFilters,
        "red_zone_floors",
        false,
      );
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Red Zone test surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      // Measure each subpath of the Ash floor with a scratch path's real bounds.
      /** @param {Element} ash */
      const stripBoxes = (ash) =>
        (ash.getAttribute("d")?.match(/M[^M]*/gu) ?? []).map((part) => {
          const scratch = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "path",
          );
          scratch.setAttribute("d", part);
          ash.parentElement?.append(scratch);
          const box = scratch.getBBox();
          scratch.remove();
          return {
            left: box.x,
            right: box.x + box.width,
            top: box.y,
            bottom: box.y + box.height,
            arcRadii: Array.from(part.matchAll(/A([\d.]+) ([\d.]+)/gu), (match) =>
              Number(match[1]),
            ),
          };
        });
      /** @param {string | null | undefined} path */
      const firstX = (path) =>
        Number(/^M([-\d.]+) /u.exec(path ?? "")?.[1] ?? Number.NaN);
      /** @param {Record<string, any>} presentation @param {any} filters */
      const paint = (presentation, filters) => {
        renderer.render(presentation, {
          showRanges: false,
          visualFilterState: filters,
        });
        const camera = battlefield.querySelector(".battlefield-camera");
        const layer = battlefield.querySelector('[data-layer="red-zone"]');
        const boundary = battlefield.querySelector(".map-boundary");
        if (
          !(camera instanceof SVGGElement) ||
          !(layer instanceof SVGGElement) ||
          !(boundary instanceof SVGRectElement)
        ) {
          throw new Error("Red Zone layer is unavailable.");
        }
        const ash = Array.from(layer.querySelectorAll(".red-zone-floor"));
        const ashStyle = ash[0] ? getComputedStyle(ash[0]) : null;
        const borders = Array.from(layer.querySelectorAll(".grave-border"));
        const marks = layer.querySelectorAll(".grave-mark").length;
        const lines = Array.from(layer.querySelectorAll(".red-zone-line"));
        return {
          layerOrder: Array.from(camera.children, (child) =>
            child.getAttribute("data-layer"),
          ),
          floorRadius: Number(boundary.getAttribute("rx")),
          ashCount: ash.length,
          strips: ash[0] ? stripBoxes(ash[0]) : [],
          ashLook: ashStyle
            ? {
                fill: ashStyle.fill,
                fillOpacity: ashStyle.fillOpacity,
                fillRule: ashStyle.fillRule,
                pointerEvents: ashStyle.pointerEvents,
              }
            : null,
          ashPaintOnly: ash.every(
            (node) =>
              node.getAttribute("aria-hidden") === "true" &&
              !node.hasAttribute("role") &&
              !node.hasAttribute("aria-label") &&
              !node.hasAttribute("data-tooltip-owner"),
          ),
          borderX: borders.map((border) =>
            firstX(border.querySelector(".grave-border__bone")?.getAttribute("d")),
          ),
          borderPointerEvents: borders.map(
            (border) => getComputedStyle(border).pointerEvents,
          ),
          marks,
          lines: lines.map((line) => ({
            x1: Number(line.getAttribute("x1")),
            x2: Number(line.getAttribute("x2")),
            y: [Number(line.getAttribute("y1")), Number(line.getAttribute("y2"))].sort(
              (left, right) => left - right,
            ),
          })),
          otherCount:
            layer.children.length - ash.length - borders.length - marks - lines.length,
          gridCount: battlefield.querySelectorAll('[data-layer="map"] .map-grid-line')
            .length,
        };
      };
      /** @param {Element} obstacle */
      const obstacleLook = (obstacle) => {
        const style = getComputedStyle(obstacle);
        return `${style.fill}|${style.stroke}`;
      };
      const rows = [];
      for (const [name, raw] of Object.entries(cases)) {
        const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
        const map = presentation.current_endpoint.scene.map;
        const on = paint(presentation, allVisualFilters);
        const surface = renderer.choreographySurface();
        if (surface === null) {
          throw new Error("Red Zone projection is unavailable.");
        }
        const obstacleRows = Array.from(battlefield.querySelectorAll(".obstacle")).map(
          (obstacle) => {
            const box = obstacle.getBoundingClientRect();
            const hit = document.elementFromPoint(
              box.left + box.width / 2,
              box.top + box.height / 2,
            );
            return {
              hitsObstacle: hit?.closest(".obstacle") === obstacle,
              look: obstacleLook(obstacle),
            };
          },
        );
        const off = paint(presentation, tintOff);
        const offLooks = Array.from(battlefield.querySelectorAll(".obstacle")).map(
          obstacleLook,
        );
        rows.push({
          name,
          width: map.width,
          height: map.height,
          redZone: map.red_zone ?? null,
          origin: surface.worldToScreen([0, 0]),
          unit: surface.worldToScreen([1, 1]),
          on,
          off,
          obstacleRows,
          offLooks,
        });
      }
      const legacy = await normalizeAuthorizedPresentationFrameV1(legacyRaw);
      renderer.render(legacy, {
        showRanges: false,
        visualFilterState: allVisualFilters,
      });
      return {
        rows,
        legacyNodes:
          battlefield.querySelector('[data-layer="red-zone"]')?.children.length ?? null,
      };
    },
    {
      cases: Object.fromEntries(
        caseNames.map((name) => [name, fixture.red_zone_cases[name]]),
      ),
      legacyRaw: fixture.pairs.replay_oracle.presentation,
    },
  );

  // Each team's strip is drawn on its own (overlaps are not merged); a strip
  // with exactly one map edge has an inner edge, the rule line.
  const expectedStripCounts = {
    live_oracle_depth_6: 2,
    live_oracle_depth_5_5_swapped_banks: 2,
    live_oracle_overlapping_strips: 2,
    "live_oracle_full-width_strips": 2,
    live_oracle_both_banks_on_the_left: 2,
    live_oracle_depth_0_1_on_width_17_3: 2,
    live_oracle_12_x_12_map: 2,
    live_oracle_depth_0: 0,
    "live_oracle_depth_1e-8_on_width_20": 0,
    live_oracle_depth_12_1_on_width_12_1: 2,
    "live_oracle_depth_1e-8_on_width_12_1": 0,
    live_oracle_depth_6_obstacles_on_strip_edges: 2,
    replay_oracle_depth_5: 2,
  };
  const expectedEdgeCounts = {
    ...expectedStripCounts,
    "live_oracle_full-width_strips": 0,
    live_oracle_depth_12_1_on_width_12_1: 0,
  };
  expect(result.rows.map((row) => row.name)).toEqual(caseNames);
  expect(result.legacyNodes).toBe(0);
  for (const row of result.rows) {
    const name = /** @type {keyof typeof expectedStripCounts} */ (row.name);
    const expected = expectedRedZoneStrips(row.width, row.redZone);
    const scaleX = row.unit.x - row.origin.x;
    /** @param {number} x */
    const screenX = (x) => row.origin.x + x * scaleX;
    const floorTop = row.origin.y + row.height * (row.unit.y - row.origin.y);
    const floorBottom = row.origin.y;
    const edges = expected.flatMap(({ inner }) => (inner === null ? [] : [inner]));
    expect(expected, row.name).toHaveLength(expectedStripCounts[name]);
    expect(edges, row.name).toHaveLength(expectedEdgeCounts[name]);
    expect(row.on.ashCount, row.name).toBe(row.redZone === null ? 0 : 1);
    expect(row.on.strips, row.name).toHaveLength(expected.length);
    for (const [index, strip] of expected.entries()) {
      const box = row.on.strips[index];
      expect(box.left, `${row.name} left`).toBeCloseTo(screenX(strip.low), 2);
      expect(box.right, `${row.name} right`).toBeCloseTo(screenX(strip.high), 2);
      expect(box.top, `${row.name} top`).toBeCloseTo(floorTop, 2);
      expect(box.bottom, `${row.name} bottom`).toBeCloseTo(floorBottom, 2);
      // Only the corners on the map edge are rounded, as the floor's are.
      expect(box.arcRadii, row.name).toHaveLength(
        (strip.atLeft ? 2 : 0) + (strip.atRight ? 2 : 0),
      );
    }
    expect(row.on.borderX, row.name).toHaveLength(edges.length);
    for (const [index, inner] of edges.entries()) {
      expect(row.on.borderX[index], `${row.name} rule edge`).toBeCloseTo(
        screenX(inner),
        2,
      );
    }
    expect(row.on.borderPointerEvents.every((value) => value === "none")).toBe(true);
    expect(row.on.otherCount, row.name).toBe(0);
    // Off: no floor, border or mark, but one plain line on each rule edge.
    expect(row.off.ashCount, row.name).toBe(0);
    expect(row.off.borderX, row.name).toEqual([]);
    expect(row.off.marks, row.name).toBe(0);
    expect(row.off.otherCount, row.name).toBe(0);
    expect(row.off.lines, row.name).toHaveLength(edges.length);
    for (const [index, inner] of edges.entries()) {
      const line = row.off.lines[index];
      expect(line.x1).toBeCloseTo(screenX(inner), 2);
      expect(line.x2).toBe(line.x1);
      expect(line.y[0]).toBeCloseTo(floorTop, 2);
      expect(line.y[1]).toBeCloseTo(floorBottom, 2);
    }
    // The Red Zone sits over the floor (and its grid) and under the stones
    // and bodies.
    const order = row.on.layerOrder;
    expect(order.indexOf("map")).toBeGreaterThanOrEqual(0);
    expect(order.indexOf("red-zone")).toBeGreaterThan(order.indexOf("map"));
    expect(order.indexOf("obstacle")).toBeGreaterThan(order.indexOf("red-zone"));
    expect(order.indexOf("body")).toBeGreaterThan(order.indexOf("red-zone"));
    if (expected.length > 0) {
      expect(row.on.ashLook, row.name).toEqual({
        fill: "rgb(107, 115, 133)",
        fillOpacity: "0.17",
        // Two strips that overlap are one path, so the overlap is not darker.
        fillRule: "nonzero",
        pointerEvents: "none",
      });
    }
    expect(row.on.ashPaintOnly, row.name).toBe(true);
    expect(
      row.obstacleRows.map((obstacle) => obstacle.look),
      row.name,
    ).toEqual(row.offLooks);
    for (const obstacle of row.obstacleRows) {
      expect(obstacle.hitsObstacle, row.name).toBe(true);
    }
  }
  const byName = Object.fromEntries(result.rows.map((row) => [row.name, row]));
  // A wall across x = 6 and a pillar across x = 14 sit on the strip edges; both stay
  // on top of the Red Zone and keep their fill and stroke.
  expect(byName.live_oracle_depth_6_obstacles_on_strip_edges.obstacleRows).toHaveLength(
    2,
  );
  // Live play draws its unit grid under the Red Zone; the Replay Viewer has no
  // grid lines, and the Red Zone still shows.
  expect(byName.live_oracle_depth_6.on.gridCount).toBeGreaterThan(0);
  expect(byName.replay_oracle_depth_5.on.gridCount).toBe(0);
  expect(byName.replay_oracle_depth_5.on.strips).toHaveLength(2);
  // A strip no wider than 1e-6 units paints nothing and has no rule line.
  for (const name of [
    "live_oracle_depth_1e-8_on_width_20",
    "live_oracle_depth_1e-8_on_width_12_1",
  ]) {
    expect(byName[name].on.strips).toEqual([]);
    expect(byName[name].off.lines).toEqual([]);
  }
  // Depth equal to the float32 width covers the whole drawn floor.
  const fullNarrow = byName.live_oracle_depth_12_1_on_width_12_1;
  for (const strip of fullNarrow.on.strips) {
    expect(strip.left).toBeCloseTo(fullNarrow.origin.x, 2);
    expect(strip.right).toBeCloseTo(
      fullNarrow.origin.x +
        fullNarrow.width * (fullNarrow.unit.x - fullNarrow.origin.x),
      2,
    );
  }

  await page.evaluate(
    async ({ raw }) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { DEFAULT_VISUAL_FILTER_STATE } = await import(
        `${moduleRoot}/visual-filters.js`
      );
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Red Zone screenshot surface is unavailable.");
      }
      new BattlefieldRenderer({ battlefield, empty }).render(
        await normalizeAuthorizedPresentationFrameV1(raw),
        { showRanges: false, visualFilterState: DEFAULT_VISUAL_FILTER_STATE },
      );
    },
    { raw: fixture.red_zone_cases.live_oracle_depth_6_obstacles_on_strip_edges },
  );
  const screenshotPath = testInfo.outputPath("red-zone-floors-obstacles.png");
  await page.locator("#battlefield").screenshot({ path: screenshotPath });
  await testInfo.attach("red-zone-floors", {
    path: screenshotPath,
    contentType: "image/png",
  });

  // The Ash floor's rounded corners follow the floor's own corners: the same
  // radius, or less where the strip is too narrow for it.
  for (const row of result.rows) {
    for (const strip of row.on.strips) {
      const fits = Math.min(
        row.on.floorRadius,
        (strip.right - strip.left) / 2,
        (strip.bottom - strip.top) / 2,
      );
      for (const radius of strip.arcRadii) {
        expect(radius, `${row.name} Ash corner radius`).toBeCloseTo(fits, 1);
      }
    }
  }
});

// The arena's lane pills are retired: the deck's Action slots and the Command
// Ring carry legality, so no legality paint or legality region reaches the arena.
test("Target Selection Visuals hides only the selection brackets and no lane pill is painted", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (raw) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const {
      DEFAULT_VISUAL_FILTER_STATE,
      enableAllVisualFilters,
      setVisualFilterEnabled,
    } = await import(`${moduleRoot}/visual-filters.js`);
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Target-filter test surface is unavailable.");
    }
    const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
    const presentationBytes = JSON.stringify(presentation);
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    /** @param {Record<string, boolean>} visualFilterState */
    const paint = (visualFilterState) => {
      renderer.render(presentation, {
        showRanges: true,
        visualFilterState,
      });
      return {
        visibleReticles: battlefield.querySelectorAll(".selected-reticle:not([hidden])")
          .length,
        legality: battlefield.querySelectorAll(".legality-pill").length,
        legalityTooltipOwners: battlefield.querySelectorAll(
          ".legality-pill[data-tooltip-owner]",
        ).length,
        controlled: battlefield.querySelector(".controlled-halo:not([hidden])")
          ?.outerHTML,
        pendingRoute: battlefield.querySelector('[data-layer="pending-route"]')
          ?.innerHTML,
        ranges: battlefield.querySelector('[data-layer="debug-range"]')?.innerHTML,
        // The brackets live in the body layer's first group, so read only the
        // body groups here.
        agents: Array.from(
          battlefield.querySelectorAll('[data-layer="body"] > [data-body-state]'),
          (group) => group.outerHTML,
        ).join(""),
        legalityPaint: battlefield.querySelectorAll(
          '[data-layer="selection-legality"] [data-layer="legality-cues"] *',
        ).length,
        legalityProtected: renderer
          .choreographySurface()
          .protectedRects.filter(
            (/** @type {{protectedKind?: string}} */ region) =>
              region.protectedKind === "legality",
          ).length,
        presentationBytes: JSON.stringify(presentation),
      };
    };
    const enabled = paint(allVisualFilters);
    const disabled = paint(
      setVisualFilterEnabled(allVisualFilters, "target_selection_visuals", false),
    );
    const restored = paint(allVisualFilters);
    return { enabled, disabled, restored, presentationBytes };
  }, fixture.presentations.replay_oracle);
  expect(result.enabled.visibleReticles).toBe(1);
  expect(result.disabled.visibleReticles).toBe(0);
  for (const state of [result.enabled, result.disabled, result.restored]) {
    expect(state.legality).toBe(0);
    expect(state.legalityTooltipOwners).toBe(0);
    expect(state.legalityPaint).toBe(0);
    expect(state.legalityProtected).toBe(0);
  }
  expect(result.enabled.agents.length).toBeGreaterThan(0);
  expect(result.disabled.controlled).toBe(result.enabled.controlled);
  expect(result.restored.controlled).toBe(result.enabled.controlled);
  expect(result.disabled.pendingRoute).toBe(result.enabled.pendingRoute);
  expect(result.restored.pendingRoute).toBe(result.enabled.pendingRoute);
  expect(result.disabled.ranges).toBe(result.enabled.ranges);
  expect(result.restored.ranges).toBe(result.enabled.ranges);
  expect(result.disabled.agents).toBe(result.enabled.agents);
  expect(result.restored.agents).toBe(result.enabled.agents);
  expect(result.restored.visibleReticles).toBe(result.enabled.visibleReticles);
  expect(result.restored.legality).toBe(result.enabled.legality);
  expect(result.restored.legalityTooltipOwners).toBe(
    result.enabled.legalityTooltipOwners,
  );
  expect(result.enabled.presentationBytes).toBe(result.presentationBytes);
  expect(result.disabled.presentationBytes).toBe(result.presentationBytes);
  expect(result.restored.presentationBytes).toBe(result.presentationBytes);
});

test("raw and forged researcher-looking scenes render unavailable and clear accepted paint", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const { authorizedPresentationSceneView } = await import(
      `${moduleRoot}/authorized-presentation-adapter.js`
    );
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const authorizedScene = authorizedPresentationSceneView(presentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (
      authorizedScene === null ||
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement)
    ) {
      throw new Error("Raw authority-fence test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    if (
      !renderer.render(presentation, {
        showRanges: true,
        visualFilterState: allVisualFilters,
      })
    ) {
      throw new Error("The accepted control frame did not paint.");
    }
    const acceptedCounts = {
      agents: battlefield.querySelectorAll(".agent").length,
      auras: battlefield.querySelectorAll(".aura-field").length,
    };
    const candidates = [
      { candidate_kind: "raw_scene_envelope", scene: structuredClone(authorizedScene) },
      structuredClone(presentation),
    ];
    const rejected = candidates.map((candidate) => {
      const painted = renderer.render(candidate, {
        showRanges: true,
        visualFilterState: allVisualFilters,
      });
      return {
        painted,
        agents: battlefield.querySelectorAll(".agent").length,
        auras: battlefield.querySelectorAll(".aura-field").length,
        audience: battlefield.getAttribute("data-audience"),
        preset: battlefield.getAttribute("data-preset"),
        viewBox: battlefield.getAttribute("viewBox"),
        emptyHidden: empty.hidden,
        emptyText: empty.textContent,
      };
    });
    return { acceptedCounts, rejected };
  }, fixture.presentations.replay_oracle);

  expect(result.acceptedCounts.agents).toBeGreaterThan(0);
  expect(result.acceptedCounts.auras).toBeGreaterThan(0);
  expect(result.rejected).toEqual([
    {
      painted: false,
      agents: 0,
      auras: 0,
      audience: null,
      preset: null,
      viewBox: null,
      emptyHidden: false,
      emptyText: "No authorized battlefield scene was returned.",
    },
    {
      painted: false,
      agents: 0,
      auras: 0,
      audience: null,
      preset: null,
      viewBox: null,
      emptyHidden: false,
      emptyText: "No authorized battlefield scene was returned.",
    },
  ]);
});

test("durable visual filters remove owned paint and restore stable battlefield identity", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(
    async ({ oracleRaw, povRaw, redZoneRaw }) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const {
        DEFAULT_VISUAL_FILTER_STATE,
        enableAllVisualFilters,
        setVisualFilterEnabled,
      } = await import(`${moduleRoot}/visual-filters.js`);
      const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Durable visual-filter test surface is unavailable.");
      }
      const presentation = await normalizeAuthorizedPresentationFrameV1(oracleRaw);
      const povPresentation = await normalizeAuthorizedPresentationFrameV1(povRaw);
      // The Oracle fixture above has a version 1 map with no strips, so Red Zone
      // Floors runs on a version 2 Oracle with two strips and obstacles on their edges.
      const redZonePresentation =
        await normalizeAuthorizedPresentationFrameV1(redZoneRaw);
      const presentationBytes = JSON.stringify(presentation);
      const povPresentationBytes = JSON.stringify(povPresentation);
      const redZonePresentationBytes = JSON.stringify(redZonePresentation);
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      /** @type {Array<{filterId: string, ownerSelector: string, ownsTooltip: boolean, reservesLayout: boolean, layoutAttribute: string | null, accessible?: boolean, absentFromFrame?: boolean}>} */
      const filterCases = [
        {
          filterId: "aura_fields",
          ownerSelector: ".aura-field",
          ownsTooltip: true,
          reservesLayout: false,
          layoutAttribute: null,
        },
        {
          filterId: "aura_modifier_badges",
          ownerSelector: ".modifier-cell, .modifier-overflow",
          ownsTooltip: true,
          reservesLayout: true,
          layoutAttribute: "data-suppressed-modifier-presentation-keys",
        },
        {
          filterId: "duration_status_badges",
          ownerSelector:
            '.status-cell, .status-overflow, .required-dock-fallback[data-kind="status"], .pov-observed-status',
          ownsTooltip: true,
          reservesLayout: true,
          layoutAttribute: "data-suppressed-status-presentation-keys",
        },
        {
          // The Spawn Ward is drawn only on a shielded champion, and no fixture
          // frame has one, so this switch must leave the drawing unchanged.
          filterId: "spawn_shield",
          ownerSelector: ".agent-spawn-shield, .agent-ward-lift",
          ownsTooltip: false,
          reservesLayout: false,
          layoutAttribute: null,
          absentFromFrame: true,
        },
        {
          // The Rune Ring on each body shows the Ultimate cooldown; it is paint
          // inside the body's hidden status marks.
          filterId: "cooldown_effects",
          ownerSelector: '.status-marks > [data-rune-piece^="ring."]',
          ownsTooltip: false,
          reservesLayout: false,
          layoutAttribute: null,
          accessible: false,
        },
      ];
      /** @type {typeof filterCases} */
      const redZoneFilterCases = [
        {
          // The Ash floor, the rule-edge borders and the "×2" marks (each
          // mark has its own label and tip); off leaves a plain edge line.
          filterId: "red_zone_floors",
          ownerSelector: ".red-zone-floor, .grave-border, .grave-mark",
          ownsTooltip: true,
          reservesLayout: false,
          layoutAttribute: null,
        },
      ];
      const viewports = [
        { width: 720, height: 480 },
        { width: 1080, height: 720 },
      ];
      // The Oracle runs keep their original order; the Red Zone runs follow them.
      const renderRuns = [
        ...viewports.map((viewport) => ({
          viewport,
          runPresentation: presentation,
          runCases: filterCases,
        })),
        ...viewports.map((viewport) => ({
          viewport,
          runPresentation: redZonePresentation,
          runCases: redZoneFilterCases,
        })),
      ];
      const rows = [];
      const coreSelectors = [
        ".map-boundary",
        ".obstacle",
        ".agent",
        ".agent-body",
        ".champion-life",
        ".agent-team-ring",
        ".agent-class-icon",
        ".champion-fallen",
        ".agent-selection",
      ];
      /** @param {{left: number, top: number, right: number, bottom: number}} bounds */
      const protectedRectangleKey = ({ left, top, right, bottom }) =>
        [left, top, right, bottom].map((value) => value.toFixed(3)).join(",");
      const protectedKey = () =>
        (renderer.choreographySurface()?.protectedRects ?? [])
          .map(protectedRectangleKey)
          .join(";");
      const coreCounts = () =>
        Object.fromEntries(
          coreSelectors.map((selector) => [
            selector,
            battlefield.querySelectorAll(selector).length,
          ]),
        );
      /** @param {string} ownerSelector */
      const ownedSnapshot = (ownerSelector) => {
        const owners = Array.from(battlefield.querySelectorAll(ownerSelector));
        return {
          count: owners.length,
          tooltipOwnerCount: owners.filter((owner) =>
            owner.hasAttribute("data-tooltip-owner"),
          ).length,
          ariaOwnerCount: owners.filter(
            (owner) =>
              owner.hasAttribute("aria-label") ||
              owner.hasAttribute("aria-hidden") ||
              owner.hasAttribute("role"),
          ).length,
          hiddenFromAssistiveTech: owners.every(
            (owner) => owner.closest('[aria-hidden="true"]') !== null,
          ),
        };
      };
      // Retired docks and the Red Zone's plain edge lines, counted in every state.
      const sideCounts = () => ({
        cooldownDocks: battlefield.querySelectorAll(
          '.cooldown-cell, .required-dock-fallback[data-kind="cooldown"]',
        ).length,
        redZoneLines: battlefield.querySelectorAll(".red-zone-line").length,
        ashPaintOnly: Array.from(battlefield.querySelectorAll(".red-zone-floor")).every(
          (ash) =>
            ash.getAttribute("aria-hidden") === "true" &&
            !ash.hasAttribute("aria-label") &&
            !ash.hasAttribute("role") &&
            !ash.hasAttribute("data-tooltip-owner"),
        ),
      });

      for (const { viewport, runPresentation, runCases } of renderRuns) {
        battlefield.style.width = `${viewport.width}px`;
        battlefield.style.height = `${viewport.height}px`;
        renderer.render(runPresentation, { showRanges: true });
        const defaultMarkup = battlefield.outerHTML;
        renderer.render(runPresentation, {
          showRanges: true,
          visualFilterState: DEFAULT_VISUAL_FILTER_STATE,
        });
        const explicitDefaultMatchesImplicit = battlefield.outerHTML === defaultMarkup;
        renderer.render(runPresentation, {
          showRanges: true,
          visualFilterState: allVisualFilters,
        });
        const originalRoot = battlefield;
        const originalAgent = battlefield.querySelector(".agent");
        const originalBodyHit = originalAgent?.querySelector(".agent-body") ?? null;
        if (!(originalAgent instanceof SVGElement) || !originalBodyHit) {
          throw new Error("Stable agent identity surface is unavailable.");
        }
        const baselineCoreCounts = coreCounts();

        for (const filterCase of runCases) {
          const baselineOwned = ownedSnapshot(filterCase.ownerSelector);
          const baselineSides = sideCounts();
          const baselineMarkup = battlefield.outerHTML;
          const baselineProtectedKey = protectedKey();
          const baselineAgentAria = originalAgent.getAttribute("aria-label") ?? "";
          const durableLayer = battlefield.querySelector(
            '[data-layer="durable-status-modifier"]',
          );
          const baselineLayoutAttribute =
            filterCase.layoutAttribute === null
              ? null
              : durableLayer?.hasAttribute(filterCase.layoutAttribute) === true;
          const disabledState = setVisualFilterEnabled(
            allVisualFilters,
            filterCase.filterId,
            false,
          );
          renderer.render(runPresentation, {
            showRanges: true,
            visualFilterState: disabledState,
          });
          const disabledOwned = ownedSnapshot(filterCase.ownerSelector);
          const disabledSides = sideCounts();
          const disabledMarkupUnchanged = battlefield.outerHTML === baselineMarkup;
          const disabledProtectedKey = protectedKey();
          const disabledAgentAria = originalAgent.getAttribute("aria-label") ?? "";
          const disabledLayoutAttribute =
            filterCase.layoutAttribute === null
              ? null
              : durableLayer?.hasAttribute(filterCase.layoutAttribute) === true;
          const disabledCoreCounts = coreCounts();
          const disabledIdentityStable =
            battlefield === originalRoot &&
            originalAgent.isConnected &&
            originalBodyHit.isConnected &&
            battlefield.contains(originalAgent) &&
            originalAgent.querySelector(".agent-body") === originalBodyHit;
          const disabledAgentAccessible =
            originalAgent.getAttribute("role") === "img" &&
            originalAgent.hasAttribute("data-tooltip-owner") &&
            disabledAgentAria.length > 0;

          renderer.render(runPresentation, {
            showRanges: true,
            visualFilterState: allVisualFilters,
          });
          const restoredOwned = ownedSnapshot(filterCase.ownerSelector);
          const restoredProtectedKey = protectedKey();
          const restoredAgentAria = originalAgent.getAttribute("aria-label") ?? "";
          rows.push({
            viewport: `${viewport.width}x${viewport.height}`,
            ...filterCase,
            explicitDefaultMatchesImplicit,
            baselineOwned,
            baselineSides,
            baselineProtectedKey,
            baselineAgentAria,
            baselineLayoutAttribute,
            baselineCoreCounts,
            disabledOwned,
            disabledSides,
            disabledMarkupUnchanged,
            disabledProtectedKey,
            disabledAgentAria,
            disabledLayoutAttribute,
            disabledCoreCounts,
            disabledIdentityStable,
            disabledAgentAccessible,
            restoredOwned,
            restoredProtectedKey,
            restoredAgentAria,
            restoredIdentityStable:
              battlefield === originalRoot &&
              originalAgent.isConnected &&
              originalAgent.querySelector(".agent-body") === originalBodyHit,
          });
        }
      }
      const povStatusRows = [];
      const statusOwnerSelector =
        '.status-cell, .status-overflow, .required-dock-fallback[data-kind="status"], .pov-observed-status';
      for (const viewport of viewports) {
        battlefield.style.width = `${viewport.width}px`;
        battlefield.style.height = `${viewport.height}px`;
        renderer.render(povPresentation, {
          showRanges: true,
          visualFilterState: allVisualFilters,
        });
        const originalAgent = battlefield.querySelector(".agent");
        const originalBodyHit = originalAgent?.querySelector(".agent-body") ?? null;
        if (!(originalAgent instanceof SVGElement) || !originalBodyHit) {
          throw new Error("Agent POV status identity surface is unavailable.");
        }
        const baselineOwned = ownedSnapshot(statusOwnerSelector);
        renderer.render(povPresentation, {
          showRanges: true,
          visualFilterState: setVisualFilterEnabled(
            allVisualFilters,
            "duration_status_badges",
            false,
          ),
        });
        const disabledOwned = ownedSnapshot(statusOwnerSelector);
        const disabledIdentityStable =
          originalAgent.isConnected &&
          originalAgent.querySelector(".agent-body") === originalBodyHit &&
          originalAgent.getAttribute("role") === "img" &&
          originalAgent.hasAttribute("data-tooltip-owner") &&
          (originalAgent.getAttribute("aria-label") ?? "").length > 0;
        renderer.render(povPresentation, {
          showRanges: true,
          visualFilterState: allVisualFilters,
        });
        povStatusRows.push({
          viewport: `${viewport.width}x${viewport.height}`,
          audience: battlefield.getAttribute("data-audience"),
          baselineOwned,
          disabledOwned,
          restoredOwned: ownedSnapshot(statusOwnerSelector),
          disabledIdentityStable,
        });
      }
      return {
        presentationUnchanged:
          JSON.stringify(presentation) === presentationBytes &&
          JSON.stringify(povPresentation) === povPresentationBytes &&
          JSON.stringify(redZonePresentation) === redZonePresentationBytes,
        stateFrozen: Object.isFrozen(DEFAULT_VISUAL_FILTER_STATE),
        rows,
        povStatusRows,
      };
    },
    {
      oracleRaw: fixture.state_cases.replay_oracle_final_selected,
      povRaw: fixture.state_cases.replay_shared_final,
      redZoneRaw: fixture.red_zone_cases.live_oracle_depth_6_obstacles_on_strip_edges,
    },
  );

  expect(result.presentationUnchanged).toBe(true);
  expect(result.stateFrozen).toBe(true);
  expect(result.rows).toHaveLength(12);
  const nothingOwned = {
    count: 0,
    tooltipOwnerCount: 0,
    ariaOwnerCount: 0,
    hiddenFromAssistiveTech: true,
  };
  // Two strips with a rule edge each: one Ash floor path, two borders and four
  // "×2" marks, each mark with its own label and tip. Off, a plain line marks
  // each rule edge instead, and the Ash floor is always paint only.
  const redZoneRows = result.rows.filter((row) => row.filterId === "red_zone_floors");
  expect(redZoneRows).toHaveLength(2);
  for (const row of redZoneRows) {
    expect(row.baselineOwned).toEqual(
      expect.objectContaining({ count: 7, tooltipOwnerCount: 4, ariaOwnerCount: 7 }),
    );
    expect(row.baselineSides.redZoneLines, row.viewport).toBe(0);
    expect(row.disabledSides.redZoneLines, row.viewport).toBe(2);
    expect(row.baselineSides.ashPaintOnly, row.viewport).toBe(true);
  }
  for (const row of result.rows) {
    expect(row.explicitDefaultMatchesImplicit, row.viewport).toBe(true);
    // The old Ultimate cooldown docks are retired in every state.
    expect(row.baselineSides.cooldownDocks, row.filterId).toBe(0);
    expect(row.disabledSides.cooldownDocks, row.filterId).toBe(0);
    if (row.absentFromFrame) {
      expect(row.baselineOwned, `${row.viewport} ${row.filterId}`).toEqual(
        nothingOwned,
      );
      expect(row.disabledMarkupUnchanged, row.filterId).toBe(true);
    } else {
      expect(
        row.baselineOwned.count,
        `${row.viewport} ${row.filterId}`,
      ).toBeGreaterThan(0);
    }
    if (row.accessible === false) {
      expect(row.baselineOwned.ariaOwnerCount, row.filterId).toBe(0);
      expect(row.baselineOwned.hiddenFromAssistiveTech, row.filterId).toBe(true);
    } else if (!row.absentFromFrame) {
      expect(row.baselineOwned.ariaOwnerCount, row.filterId).toBeGreaterThan(0);
    }
    if (row.ownsTooltip) {
      expect(row.baselineOwned.tooltipOwnerCount, row.filterId).toBeGreaterThan(0);
    }
    expect(row.disabledOwned).toEqual(nothingOwned);
    expect(row.disabledCoreCounts).toEqual(row.baselineCoreCounts);
    expect(row.disabledIdentityStable).toBe(true);
    expect(row.disabledAgentAccessible).toBe(true);
    expect(row.restoredOwned).toEqual(row.baselineOwned);
    expect(row.restoredProtectedKey).toBe(row.baselineProtectedKey);
    expect(row.restoredIdentityStable).toBe(true);
    if (row.reservesLayout) {
      expect(row.disabledProtectedKey, row.filterId).not.toBe(row.baselineProtectedKey);
      expect(row.baselineLayoutAttribute, row.filterId).toBe(true);
      expect(row.disabledLayoutAttribute, row.filterId).toBe(false);
    }
  }
  expect(result.povStatusRows).toHaveLength(2);
  for (const row of result.povStatusRows) {
    expect(row.audience).toBe("agent_pov");
    expect(row.baselineOwned.count, row.viewport).toBeGreaterThan(0);
    expect(row.baselineOwned.tooltipOwnerCount, row.viewport).toBeGreaterThan(0);
    expect(row.baselineOwned.ariaOwnerCount, row.viewport).toBeGreaterThan(0);
    expect(row.disabledOwned).toEqual(nothingOwned);
    expect(row.restoredOwned).toEqual(row.baselineOwned);
    expect(row.disabledIdentityStable).toBe(true);
  }
});

test("selected Replay aura modifiers remain exact beside Mage Burst duration and cooldown", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (raw) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { authorizedPresentationSceneView } = await import(
      `${moduleRoot}/authorized-presentation-adapter.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const { resolveVisualToken } = await import(`${moduleRoot}/vocabulary.js`);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Aura-coexistence test surface is unavailable.");
    }
    const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
    const scene = authorizedPresentationSceneView(presentation);
    if (scene === null) {
      throw new Error("Authorized Replay scene is unavailable.");
    }
    const expected = scene.agents
      .flatMap((/** @type {Record<string, any>} */ agent) =>
        (Array.isArray(agent.modifiers) ? agent.modifiers : [])
          .filter(
            (/** @type {Record<string, any>} */ modifier) =>
              typeof modifier.multiplier !== "number" ||
              !Number.isFinite(modifier.multiplier) ||
              modifier.multiplier !== 1,
          )
          .map((/** @type {Record<string, any>} */ modifier) => {
            const token = resolveVisualToken("modifier", modifier.token_id, modifier);
            return `${agent.presentation_key}|${token.tokenId}`;
          }),
      )
      .sort();
    const selectedKey = scene.selection?.selected_presentation_key ?? null;
    const selectedCooldown =
      scene.agents.find(
        (/** @type {Record<string, any>} */ agent) =>
          agent.presentation_key === selectedKey,
      )?.ultimate_cooldown_remaining ?? null;
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    const rows = [];
    for (const viewport of [
      { width: 960, height: 600 },
      { width: 1440, height: 900 },
    ]) {
      battlefield.style.width = `${viewport.width}px`;
      battlefield.style.height = `${viewport.height}px`;
      renderer.render(presentation, {
        showRanges: true,
        visualFilterState: allVisualFilters,
      });
      const actual = Array.from(
        battlefield.querySelectorAll(".modifier-cell"),
        (cell) =>
          `${cell.getAttribute("data-presentation-key")}|${cell.getAttribute("data-token-id")}`,
      ).sort();
      const durableLayer = battlefield.querySelector(
        '[data-layer="durable-status-modifier"]',
      );
      rows.push({
        viewport: `${viewport.width}x${viewport.height}`,
        actual,
        suppressed:
          durableLayer?.getAttribute("data-suppressed-modifier-presentation-keys") ??
          null,
        collisionFree: Array.from(
          battlefield.querySelectorAll(".modifier-dock"),
          (dock) => dock.getAttribute("data-collision-free"),
        ),
        selectedStatusCount:
          selectedKey === null
            ? 0
            : battlefield.querySelectorAll(
                `.status-cell[data-presentation-key="${selectedKey}"]`,
              ).length,
        // The Ultimate cooldown is the Rune Ring on the body (the old cooldown
        // docks are retired).
        selectedRingPieces:
          selectedKey === null
            ? []
            : Array.from(
                battlefield.querySelectorAll(
                  `.status-marks[data-marks-for="${CSS.escape(selectedKey)}"] > [data-rune-piece^="ring."]`,
                ),
                (piece) => piece.getAttribute("data-rune-piece"),
              ),
        cooldownDocks: battlefield.querySelectorAll(
          '.cooldown-cell, .required-dock-fallback[data-kind="cooldown"]',
        ).length,
      });
    }
    return { expected, selectedKey, selectedCooldown, rows };
  }, fixture.state_cases.replay_oracle_final_selected);

  expect(result.selectedKey).not.toBeNull();
  // The selected Mage's Burst is cooling down: the Rune Ring's six arcs of 5
  // ticks light as the 30-tick cooldown runs out.
  expect(result.selectedCooldown).toBeGreaterThan(0);
  const litArcs = Math.min(
    5,
    Math.max(0, Math.floor((30 - result.selectedCooldown) / 5)),
  );
  const expectedRing = [
    "ring.track",
    ...Array.from({ length: litArcs }, (_, index) => `ring.arc.${index}`),
  ];
  expect(result.expected).toHaveLength(2);
  expect(
    result.expected.some((/** @type {string} */ key) =>
      key.endsWith("|mage_amplification"),
    ),
  ).toBe(true);
  expect(
    result.expected.some((/** @type {string} */ key) =>
      key.endsWith("|warrior_mitigation"),
    ),
  ).toBe(true);
  for (const row of result.rows) {
    expect(row.actual, row.viewport).toEqual(result.expected);
    expect(row.suppressed, row.viewport).toBe("");
    expect(row.collisionFree, row.viewport).not.toContain("false");
    expect(row.selectedStatusCount, row.viewport).toBeGreaterThan(0);
    expect(row.selectedRingPieces, row.viewport).toEqual(expectedRing);
    expect(row.cooldownDocks, row.viewport).toBe(0);
  }
});

test("all three normalized Agent POV leaves use researcher-space aura attribution without geometry", async ({
  page,
}) => {
  const leafNames = [
    "live_no_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ];
  const renderings = [];
  for (const leafName of leafNames) {
    const raw = fixture.presentations[leafName];
    await page.goto(origin);
    const installed = await page.evaluate(async (rawPresentation) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
        `${moduleRoot}/visual-filters.js`
      );
      const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
      const { createTooltipController } = await import(`${moduleRoot}/tooltip.js`);
      const { resolveVisualToken } = await import(`${moduleRoot}/vocabulary.js`);
      const presentation =
        await normalizeAuthorizedPresentationFrameV1(rawPresentation);
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      const tooltip = document.querySelector("#visual-tooltip");
      const title = document.querySelector("#visual-tooltip-title");
      const details = document.querySelector("#visual-tooltip-details");
      if (
        !(battlefield instanceof SVGSVGElement) ||
        !(empty instanceof HTMLElement) ||
        !(tooltip instanceof HTMLElement) ||
        !(title instanceof HTMLElement) ||
        !(details instanceof HTMLElement)
      ) {
        throw new Error("Agent POV aura tooltip surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      const painted = renderer.render(presentation, {
        showRanges: true,
        visualFilterState: allVisualFilters,
      });
      const controller = createTooltipController({
        root: document.body,
        tooltip,
        title,
        details,
      });
      const rawScene = rawPresentation.current_endpoint?.parts?.scene;
      const rawAuraFields = Array.isArray(rawScene?.aura_fields)
        ? rawScene.aura_fields
        : [];
      const expectedModifierRows = (
        Array.isArray(rawScene?.agents) ? rawScene.agents : []
      )
        .flatMap((/** @type {Record<string, any>} */ agent) =>
          (Array.isArray(agent.aura_modifiers) ? agent.aura_modifiers : []).map(
            (/** @type {Record<string, any>} */ modifier) => {
              const token = resolveVisualToken("modifier", modifier.aura_id, modifier);
              return `${agent.presentation_key}|${token.tokenId}|${modifier.multiplier}`;
            },
          ),
        )
        .sort();
      const auraOwners = Array.from(
        battlefield.querySelectorAll(
          '[data-layer="aura"] .aura-field[data-tooltip-owner]',
        ),
      );
      const tooltipBytes = auraOwners.map((owner) => {
        if (!(owner instanceof SVGElement)) {
          throw new Error("Agent POV aura owner is not an SVG element.");
        }
        owner.focus();
        return JSON.stringify({
          title: title.textContent,
          details: details.innerHTML,
          text: details.textContent,
        });
      });
      // Live play and the Replay Viewer place the arena differently, so compare
      // the aura owners without their circle geometry.
      const auraMarkup = auraOwners.map((owner) => {
        const copy = owner.cloneNode(true);
        if (!(copy instanceof Element)) {
          throw new Error("Agent POV aura owner did not copy.");
        }
        for (const name of ["cx", "cy", "r"]) {
          copy.removeAttribute(name);
        }
        return copy.outerHTML;
      });
      const modifierOwners = Array.from(
        battlefield.querySelectorAll(
          '[data-layer="durable-status-modifier"] .modifier-cell[data-tooltip-owner]',
        ),
      );
      const modifierRows = modifierOwners
        .map(
          (owner) =>
            `${owner.getAttribute("data-presentation-key")}|${owner.getAttribute("data-token-id")}|${owner.getAttribute("data-multiplier")}`,
        )
        .sort();
      const modifierTooltipBytes = modifierOwners.map((owner) => {
        if (!(owner instanceof SVGElement)) {
          throw new Error("Agent POV modifier owner is not an SVG element.");
        }
        owner.focus();
        return JSON.stringify({
          title: title.textContent,
          details: details.innerHTML,
          text: details.textContent,
        });
      });
      const modifierMarkup = modifierOwners.map((owner) => owner.outerHTML);
      controller.destroy();
      return {
        auraCount: auraOwners.length,
        audience: battlefield.dataset.audience,
        painted,
        hiddenSourceKeys: rawAuraFields.map(
          (/** @type {Record<string, any>} */ field) => field.source_presentation_key,
        ),
        sourceAttributeCount: battlefield.querySelectorAll(
          '[data-layer="aura"] [data-source-presentation-key]',
        ).length,
        auraMarkup,
        tooltipBytes,
        expectedModifierRows,
        modifierRows,
        modifierMarkup,
        modifierTooltipBytes,
        modifierSourceAttributeCount: battlefield.querySelectorAll(
          '[data-layer="durable-status-modifier"] .modifier-dock [data-source-presentation-key], [data-layer="durable-status-modifier"] .modifier-dock [data-source-public-agent-id], [data-layer="durable-status-modifier"] .modifier-dock [data-source-class-id]',
        ).length,
      };
    }, raw);
    expect(installed.auraCount).toBe(2);
    expect(installed.audience).toBe("agent_pov");
    expect(installed.painted).toBe(true);
    expect(installed.sourceAttributeCount).toBe(0);
    expect(installed.hiddenSourceKeys).toHaveLength(2);
    expect(new Set(installed.hiddenSourceKeys).size).toBe(2);
    for (const hiddenSourceKey of installed.hiddenSourceKeys) {
      expect(typeof hiddenSourceKey).toBe("string");
      expect(installed.auraMarkup.join("\n")).not.toContain(hiddenSourceKey);
      expect(installed.tooltipBytes.join("\n")).not.toContain(hiddenSourceKey);
    }
    expect(installed.tooltipBytes).toHaveLength(2);
    for (const tooltipBytes of installed.tooltipBytes) {
      // The Source row names the emitter with the champion label ("Mage · ID 0").
      expect(tooltipBytes).toMatch(/Source(?:Mage|Warrior) · ID \d+"/u);
      expect(tooltipBytes).not.toContain("Not disclosed");
    }
    expect(installed.expectedModifierRows).toHaveLength(2);
    expect(installed.modifierRows).toEqual(installed.expectedModifierRows);
    expect(installed.modifierSourceAttributeCount).toBe(0);
    expect(installed.modifierTooltipBytes).toHaveLength(2);
    expect(installed.modifierMarkup.join("\n")).not.toContain("data-source-");
    for (const tooltipBytes of installed.modifierTooltipBytes) {
      expect(tooltipBytes).not.toContain('semantic-explanation__label">Source');
      expect(tooltipBytes).not.toContain("Emitter");
    }
    renderings.push(installed);
  }

  const allHiddenSourceKeys = renderings.flatMap(
    (rendering) => rendering.hiddenSourceKeys,
  );
  expect(new Set(allHiddenSourceKeys).size).toBe(allHiddenSourceKeys.length);
  expect(renderings[1].hiddenSourceKeys).not.toEqual(renderings[0].hiddenSourceKeys);
  expect(renderings[2].hiddenSourceKeys).not.toEqual(renderings[0].hiddenSourceKeys);
  expect(renderings[1].auraMarkup).toEqual(renderings[0].auraMarkup);
  expect(renderings[2].auraMarkup).toEqual(renderings[0].auraMarkup);
  expect(renderings[1].tooltipBytes).toEqual(renderings[0].tooltipBytes);
  expect(renderings[2].tooltipBytes).toEqual(renderings[0].tooltipBytes);
  expect(renderings[1].modifierTooltipBytes).toEqual(
    renderings[0].modifierTooltipBytes,
  );
  expect(renderings[2].modifierTooltipBytes).toEqual(
    renderings[0].modifierTooltipBytes,
  );
});

test("all three Agent leaves show status tips in the card's words, with no researcher keys", async ({
  page,
}) => {
  for (const leafName of [
    "live_no_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    const raw = fixture.presentations[leafName];
    await page.goto(origin);
    const result = await page.evaluate(async (rawPresentation) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
        `${moduleRoot}/visual-filters.js`
      );
      const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
      const { createTooltipController } = await import(`${moduleRoot}/tooltip.js`);
      const presentation =
        await normalizeAuthorizedPresentationFrameV1(rawPresentation);
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      const tooltip = document.querySelector("#visual-tooltip");
      const title = document.querySelector("#visual-tooltip-title");
      const details = document.querySelector("#visual-tooltip-details");
      if (
        !(battlefield instanceof SVGSVGElement) ||
        !(empty instanceof HTMLElement) ||
        !(tooltip instanceof HTMLElement) ||
        !(title instanceof HTMLElement) ||
        !(details instanceof HTMLElement)
      ) {
        throw new TypeError("Agent status tooltip surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      const controller = createTooltipController({
        root: document.body,
        tooltip,
        title,
        details,
      });
      const localAgents = /** @type {Record<string, any>[]} */ (
        rawPresentation.current_endpoint.parts.scene.agents
      );
      const roster = /** @type {Record<string, any>[]} */ (
        rawPresentation.researcher_space.roster_agents
      );
      const localStatusRecipient = localAgents.find(
        (/** @type {Record<string, any>} */ agent) => agent.statuses.length > 0,
      );
      if (!localStatusRecipient) {
        throw new TypeError("Agent status recipient is unavailable.");
      }
      const canonical = (/** @type {Record<string, any>} */ agent) => {
        const identity =
          rawPresentation.researcher_space.identity_directory.identities.find(
            (/** @type {Record<string, any>} */ row) =>
              row.public_agent_id === agent.public_agent_id,
          );
        if (!identity)
          throw new Error("Tooltip identity is absent from the recorded directory.");
        const slot = (identity.team_id - 1) * 5 + identity.team_local_slot;
        return `${agent.class_name} · ID ${slot}`;
      };
      const producerSelectors = {
        durable: ".status-cell[data-tooltip-owner]",
        overflow: ".status-overflow[data-tooltip-owner]",
        fallback: '.required-dock-fallback[data-kind="status"][data-tooltip-owner]',
        povObserved: ".pov-observed-status[data-tooltip-owner]",
      };
      const capture = () => {
        /** @type {Record<string, any>[]} */
        const rows = [];
        /** @type {Record<string, number>} */
        const counts = {};
        for (const [producer, selector] of Object.entries(producerSelectors)) {
          const owners = Array.from(battlefield.querySelectorAll(selector));
          counts[producer] = owners.length;
          for (const owner of owners) {
            if (!(owner instanceof SVGElement)) {
              throw new TypeError("Agent status owner is not SVG.");
            }
            owner.dispatchEvent(new FocusEvent("focusin", { bubbles: true }));
            const ownerKey = owner.getAttribute("data-presentation-key");
            const observedBody = owner.closest(".pov-observed-body");
            const ownerPublicId =
              localAgents.find(
                (/** @type {Record<string, any>} */ agent) =>
                  agent.presentation_key === ownerKey,
              )?.public_agent_id ??
              observedBody?.getAttribute("data-public-agent-id") ??
              null;
            const recipient = roster.find(
              (/** @type {Record<string, any>} */ agent) =>
                agent.public_agent_id === ownerPublicId,
            );
            if (!recipient) {
              throw new TypeError("Status owner does not join the researcher roster.");
            }
            const tokenId = owner.getAttribute("data-token-id");
            const duration = Number(owner.getAttribute("data-duration"));
            const statusIndex = Number(owner.getAttribute("data-index"));
            const candidateStatuses =
              producer === "durable" &&
              Number.isInteger(statusIndex) &&
              recipient.statuses[statusIndex]
                ? [recipient.statuses[statusIndex]]
                : tokenId === null
                  ? recipient.statuses
                  : recipient.statuses.filter(
                      (/** @type {Record<string, any>} */ status) =>
                        status.status_id === tokenId &&
                        status.remaining_duration === duration,
                    );
            const sourceReferences = candidateStatuses.flatMap(
              (/** @type {Record<string, any>} */ status) => status.direct_sources,
            );
            const uniqueSourcePublicIds = [
              ...new Set(
                sourceReferences.map(
                  (/** @type {Record<string, any>} */ source) =>
                    source.source_public_agent_id,
                ),
              ),
            ];
            const expectedSources = uniqueSourcePublicIds.map((publicId) => {
              const source = roster.find(
                (/** @type {Record<string, any>} */ agent) =>
                  agent.public_agent_id === publicId,
              );
              if (!source) {
                throw new TypeError(
                  "Status source does not join the researcher roster.",
                );
              }
              return canonical(source);
            });
            rows.push({
              producer,
              labels: Array.from(
                details.querySelectorAll(".semantic-explanation__label"),
                (node) => node.textContent,
              ),
              detailsHtml: details.innerHTML,
              detailsText: details.textContent ?? "",
              ownerMarkup: owner.outerHTML,
              expectedRecipient: canonical(recipient),
              expectedSources,
              researcherKeys: roster.map(
                (/** @type {Record<string, any>} */ agent) => agent.presentation_key,
              ),
            });
          }
        }
        return { counts, rows };
      };

      const render = () =>
        renderer.render(presentation, {
          showRanges: true,
          visualFilterState: allVisualFilters,
          localInspectedPresentationKey: localStatusRecipient.presentation_key,
        });
      render();
      const standard = capture();
      battlefield.style.width = "320px";
      battlefield.style.height = "240px";
      render();
      const compact = capture();
      controller.destroy();
      return { standard, compact };
    }, raw);
    expect(result.standard.counts.durable, leafName).toBeGreaterThan(0);
    expect(result.compact.counts.durable, leafName).toBeGreaterThan(0);
    expect(result.compact.counts.overflow, leafName).toBe(0);
    // The branded five-leaf adapter currently has no observed-body collection.
    // Each fixture's singleton keeps its real status badge in the compact dock;
    // neither a hidden-count badge nor an impossible-placement fallback is needed.
    expect(result.standard.counts.fallback, leafName).toBe(0);
    expect(result.compact.counts.fallback, leafName).toBe(0);
    expect(result.standard.counts.povObserved, leafName).toBe(0);
    expect(result.compact.counts.povObserved, leafName).toBe(0);
    expect(result.standard.counts.overflow, leafName).toBe(0);
    for (const row of [...result.standard.rows, ...result.compact.rows]) {
      // A status tip is the Champion Card's row: the card's sentence with the
      // source by class only, then one row, Time Left. The recipient and the
      // other rows stay on the full surface. Every leaf's one status is a
      // Priest's Blessing of Freedom on a Warrior, 1 tick left.
      if (row.producer === "durable" || row.producer === "povObserved") {
        expect(row.labels, `${leafName}/${row.producer}`).toEqual(["Time Left"]);
      }
      expect(row.detailsText, `${leafName}/${row.producer}`).toContain(
        "Slows can't drop it below 85% speed. From a Priest.",
      );
      expect(row.detailsText, `${leafName}/${row.producer}`).not.toContain("Recipient");
      expect(row.detailsText, `${leafName}/${row.producer}`).not.toContain(
        row.expectedRecipient,
      );
      expect(row.detailsText, `${leafName}/${row.producer}`).not.toContain(
        "Not disclosed",
      );
      expect(row.expectedSources.length, `${leafName}/${row.producer}`).toBe(1);
      for (const expectedSource of row.expectedSources) {
        expect(row.detailsText, `${leafName}/${row.producer}`).not.toContain(
          expectedSource,
        );
      }
      for (const researcherKey of row.researcherKeys) {
        expect(row.detailsHtml, `${leafName}/${row.producer}`).not.toContain(
          researcherKey,
        );
        expect(row.ownerMarkup, `${leafName}/${row.producer}`).not.toContain(
          researcherKey,
        );
      }
    }
  }
});

test("Oracle aura attribution remains exact in the tooltip and absent from aura DOM keys", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const { createTooltipController } = await import(`${moduleRoot}/tooltip.js`);
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    const tooltip = document.querySelector("#visual-tooltip");
    const title = document.querySelector("#visual-tooltip-title");
    const details = document.querySelector("#visual-tooltip-details");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement) ||
      !(tooltip instanceof HTMLElement) ||
      !(title instanceof HTMLElement) ||
      !(details instanceof HTMLElement)
    ) {
      throw new Error("Oracle aura tooltip surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    const painted = renderer.render(presentation, {
      showRanges: true,
      visualFilterState: allVisualFilters,
    });
    const controller = createTooltipController({
      root: document.body,
      tooltip,
      title,
      details,
    });
    const auraOwners = Array.from(
      battlefield.querySelectorAll(
        '[data-layer="aura"] .aura-field[data-tooltip-owner]',
      ),
    );
    const firstAura = auraOwners[0];
    if (!(firstAura instanceof SVGElement)) {
      throw new Error("Oracle aura owner is unavailable.");
    }
    firstAura.focus();
    const labels = Array.from(
      details.querySelectorAll(".semantic-explanation__label"),
      (node) => node.textContent,
    );
    const values = Array.from(
      details.querySelectorAll(".semantic-explanation__value"),
      (node) => node.textContent,
    );
    const auraMarkup = auraOwners.map((owner) => owner.outerHTML);
    controller.destroy();
    return {
      painted,
      audience: battlefield.dataset.audience,
      auraCount: auraOwners.length,
      sourceAttributeCount: battlefield.querySelectorAll(
        '[data-layer="aura"] [data-source-presentation-key]',
      ).length,
      auraMarkup,
      labels,
      values,
    };
  }, fixture.presentations.replay_oracle);

  expect(result.painted).toBe(true);
  expect(result.audience).toBe("researcher");
  expect(result.auraCount).toBe(2);
  expect(result.sourceAttributeCount).toBe(0);
  expect(result.auraMarkup.join("\n")).not.toContain("data-source-presentation-key");
  const sourceIndex = result.labels.indexOf("Source");
  expect(sourceIndex).toBeGreaterThanOrEqual(0);
  expect(result.values[sourceIndex]).toBe("Mage · ID 0");
});

test("incoming choreography paints presentation-key metadata and no slots", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (raw) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Renderer test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, {
      showRanges: true,
      visualFilterState: allVisualFilters,
    });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface, allVisualFilters);
    if (surface === null || plan === null) {
      throw new Error("Authorized choreography surface or plan is unavailable.");
    }
    new SvgChoreographyPainter().install(plan, surface, {
      motionMode: "off",
      settled: false,
      persistentOnly: false,
    });
    const eventNodes = Array.from(battlefield.querySelectorAll(".combat-effect"));
    return {
      eventCount: eventNodes.length,
      keyAttributeCount: eventNodes.reduce(
        (count, node) =>
          count +
          Array.from(node.attributes).filter((attribute) =>
            attribute.name.endsWith("-presentation-key"),
          ).length,
        0,
      ),
      slotAttributeCount: Array.from(
        battlefield.querySelectorAll(
          ".combat-choreography *, .combat-choreography-routes *",
        ),
      ).reduce(
        (count, node) =>
          count +
          Array.from(node.attributes).filter(
            (attribute) =>
              attribute.name === "data-slot" || attribute.name.endsWith("-slot"),
          ).length,
        0,
      ),
    };
  }, fixture.presentations.replay_oracle);
  expect(result.eventCount).toBeGreaterThan(0);
  expect(result.keyAttributeCount).toBeGreaterThan(0);
  expect(result.slotAttributeCount).toBe(0);
});

/**
 * @param {{
 *   source: {x: number, y: number},
 *   target: {x: number, y: number},
 *   sourceRadius: number,
 *   targetRadius: number,
 *   start: {x: number, y: number},
 *   end: {x: number, y: number},
 *   landingCenter: {x: number, y: number},
 *   endInsideLanding: boolean,
 * }} blow
 * @param {string} label
 */
function expectBasicBlowGeometry(blow, label) {
  const along = { x: blow.target.x - blow.source.x, y: blow.target.y - blow.source.y };
  const length = Math.hypot(along.x, along.y);
  const unit = { x: along.x / length, y: along.y / length };
  /** @param {{x: number, y: number}} point */
  const offLine = (point) =>
    Math.abs((point.x - blow.source.x) * unit.y - (point.y - blow.source.y) * unit.x);
  /** @param {{x: number, y: number}} point */
  const fromTarget = (point) =>
    Math.hypot(point.x - blow.target.x, point.y - blow.target.y);
  // The stroke leaves 2 px outside the striker's rim and ends just outside the
  // target's rim (1.04 radii), on the line between the two champions.
  expect(blow.start.x, `${label} stroke start`).toBeCloseTo(
    blow.source.x + unit.x * (blow.sourceRadius + 2),
    1,
  );
  expect(blow.start.y, `${label} stroke start`).toBeCloseTo(
    blow.source.y + unit.y * (blow.sourceRadius + 2),
    1,
  );
  expect(fromTarget(blow.end), `${label} stroke end`).toBeCloseTo(
    1.04 * blow.targetRadius,
    1,
  );
  expect(offLine(blow.end), `${label} stroke end`).toBeLessThan(0.05);
  // The landing sits on the same line, outside the target body and on the
  // striker's side, and it covers the stroke's end.
  expect(offLine(blow.landingCenter), `${label} landing`).toBeLessThan(0.05);
  expect(fromTarget(blow.landingCenter), `${label} landing`).toBeGreaterThan(
    blow.targetRadius,
  );
  expect(
    (blow.landingCenter.x - blow.target.x) * unit.x +
      (blow.landingCenter.y - blow.target.y) * unit.y,
    `${label} landing side`,
  ).toBeLessThan(0);
  expect(blow.endInsideLanding, `${label} landing covers the stroke end`).toBe(true);
}

test("live and replay Basic blows run rim to rim and land where their stroke ends", async ({
  page,
}) => {
  const presentations = [
    {
      sourcePresentation: fixture.presentations.live_oracle,
      expectedSemantic: "healing",
    },
    {
      sourcePresentation: fixture.presentations.live_oracle,
      expectedSemantic: "damage",
    },
    {
      sourcePresentation: fixture.presentations.replay_oracle,
      expectedSemantic: "healing",
    },
    {
      sourcePresentation: fixture.presentations.replay_oracle,
      expectedSemantic: "damage",
    },
  ];
  for (const { sourcePresentation, expectedSemantic } of presentations) {
    const raw = structuredClone(sourcePresentation);
    const agentsByKey = new Map(
      raw.current_endpoint.scene.agents.map(
        (/** @type {Record<string, any>} */ agent) => [agent.presentation_key, agent],
      ),
    );
    /** @param {number} classId */
    const anchorForClass = (classId) => {
      const trajectory = raw.latest_events.agent_phase_trajectories.find(
        (/** @type {Record<string, any>} */ candidate) =>
          agentsByKey.get(candidate.agent_presentation_key)?.class_id === classId,
      );
      if (!trajectory) {
        throw new Error(`Oracle class ${classId} trajectory is unavailable.`);
      }
      return structuredClone(trajectory.transition_start);
    };
    const activation = raw.latest_events.events.find(
      (/** @type {Record<string, any>} */ event) =>
        event.event_kind === "ability_activated",
    );
    if (!activation) {
      throw new Error("Oracle activation event is unavailable.");
    }
    if (expectedSemantic === "damage") {
      activation.source_anchor = anchorForClass(1);
      activation.recipient_anchor = anchorForClass(3);
    }
    await page.goto(origin);
    const result = await page.evaluate(async (rawPresentation) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
        `${moduleRoot}/visual-filters.js`
      );
      const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
      const { buildChoreographyPlan } = await import(
        `${moduleRoot}/choreography-plan.js`
      );
      const { SvgChoreographyPainter } = await import(
        `${moduleRoot}/choreography-painter.js`
      );
      const presentation =
        await normalizeAuthorizedPresentationFrameV1(rawPresentation);
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Activation-impact test surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      renderer.render(presentation, {
        showRanges: false,
        visualFilterState: allVisualFilters,
      });
      const surface = renderer.choreographySurface();
      const plan = buildChoreographyPlan(presentation, surface, allVisualFilters);
      if (surface === null || plan === null) {
        throw new Error("Authorized activation choreography is unavailable.");
      }
      new SvgChoreographyPainter().install(plan, surface, {
        motionMode: "off",
        settled: false,
        persistentOnly: false,
      });
      const activations = /** @type {Record<string, any>[]} */ (plan.events).filter(
        (event) =>
          event.kind === "activation" &&
          (event.impactSemantic === "damage" || event.impactSemantic === "healing") &&
          event.route,
      );
      const camera = battlefield.querySelector(".battlefield-camera");
      const cameraMatrix =
        camera instanceof SVGGElement ? camera.getScreenCTM()?.inverse() : null;
      if (!cameraMatrix) {
        throw new Error("Activation camera frame is unavailable.");
      }
      // Map a point in a node's own frame to the plan's CSS pixel frame.
      /** @param {SVGGraphicsElement} node @param {DOMPointInit} point */
      const toField = (node, point) => {
        const matrix = node.getScreenCTM();
        if (matrix === null) {
          throw new Error("Activation node has no screen transform.");
        }
        const field = new DOMPoint(point.x, point.y)
          .matrixTransform(matrix)
          .matrixTransform(cameraMatrix);
        return { x: field.x, y: field.y };
      };
      const netCues = Array.from(
        battlefield.querySelectorAll(".combat-effect--net-health"),
      );
      return {
        presentationKind: presentation.presentation_kind,
        netCueLabels: netCues.map((cue) => cue.getAttribute("aria-label")),
        netNumberCount: battlefield.querySelectorAll(
          ".combat-effect--net-health .combat-number, .combat-net__label",
        ).length,
        activations: activations.map((event) => {
          const selector = `[data-event-id="${CSS.escape(event.eventId)}"]`;
          const effect = battlefield.querySelector(
            `.combat-effect--activation${selector}`,
          );
          const routeEffect = battlefield.querySelector(
            `.combat-route-effect--activation${selector}`,
          );
          const landing = effect?.querySelector(".combat-impact.basic-landing");
          const stroke = routeEffect?.querySelector(".basic-blow__core");
          if (
            !(effect instanceof SVGElement) ||
            !(landing instanceof SVGGraphicsElement) ||
            !(stroke instanceof SVGPathElement)
          ) {
            throw new Error(`Activation landing ${event.eventId} is unavailable.`);
          }
          const strokeEnd = stroke.getPointAtLength(stroke.getTotalLength());
          const strokeEndOnScreen = new DOMPoint(
            strokeEnd.x,
            strokeEnd.y,
          ).matrixTransform(stroke.getScreenCTM() ?? undefined);
          // The stroke stops on the landing's edge; allow 0.01 px of rounding.
          const landingBox = landing.getBoundingClientRect();
          const edge = 0.01;
          return {
            semantic: event.impactSemantic,
            sourceClass: effect.getAttribute("data-source-class"),
            blow: {
              source: event.source,
              target: event.target,
              sourceRadius: event.sourceRadius,
              targetRadius: event.targetRadius,
              start: toField(stroke, stroke.getPointAtLength(0)),
              end: toField(stroke, strokeEnd),
              landingCenter: toField(landing, { x: 0, y: 0 }),
              endInsideLanding:
                strokeEndOnScreen.x >= landingBox.left - edge &&
                strokeEndOnScreen.x <= landingBox.right + edge &&
                strokeEndOnScreen.y >= landingBox.top - edge &&
                strokeEndOnScreen.y <= landingBox.bottom + edge,
            },
            landingClass: landing.getAttribute("class"),
            landingBasicClass: landing.getAttribute("data-basic-class"),
            semanticMarkCount: effect.querySelectorAll(".combat-impact__semantic")
              .length,
            impactLayoutAttributes: landing
              .getAttributeNames()
              .filter((name) => name.startsWith("data-layout-")),
            impactLeaderCount: battlefield.querySelectorAll(
              `${selector} .combat-cue__leader--impact`,
            ).length,
            hasPlannedImpactCue: event.impactCue !== undefined,
            hasPlannedImpactBounds: event.impactBounds !== undefined,
            hasPlannedImpactLeader: event.impactLeader !== undefined,
            hasPlannedImpactLayoutKey: event.impactLayoutKey !== undefined,
            hasPlannedImpactDisposition: event.impactDisposition !== undefined,
            hasPlannedImpactCollisionFlag: event.impactCueCollisionFree !== undefined,
          };
        }),
      };
    }, raw);

    expect(result.presentationKind).toBe(raw.presentation_kind);
    expect(result.activations.length).toBeGreaterThan(0);
    expect(new Set(result.activations.map(({ semantic }) => semantic))).toEqual(
      new Set([expectedSemantic]),
    );
    // The recipient's whole-tick life change is unchanged in this frame, so its
    // battle text cue says so and draws no number.
    expect(result.netCueLabels).toEqual(["No Net Change"]);
    expect(result.netNumberCount).toBe(0);
    for (const activation of result.activations) {
      expectBasicBlowGeometry(
        activation.blow,
        `${raw.presentation_kind} ${expectedSemantic}`,
      );
      // The landing's shape carries the meaning: a Priest's heal blooms, a
      // Mage's blow lands in a star; no plus or minus mark is drawn.
      expect(activation.sourceClass).toBe(
        expectedSemantic === "healing" ? "priest" : "mage",
      );
      expect(activation.landingBasicClass).toBe(activation.sourceClass);
      expect(activation.landingClass).toContain(
        expectedSemantic === "healing" ? "basic-landing--bloom" : "basic-landing--star",
      );
      expect(activation.semanticMarkCount).toBe(0);
      expect(activation.impactLayoutAttributes).toEqual([]);
      expect(activation.impactLeaderCount).toBe(0);
      expect(activation.hasPlannedImpactCue).toBe(false);
      expect(activation.hasPlannedImpactBounds).toBe(false);
      expect(activation.hasPlannedImpactLeader).toBe(false);
      expect(activation.hasPlannedImpactLayoutKey).toBe(false);
      expect(activation.hasPlannedImpactDisposition).toBe(false);
      expect(activation.hasPlannedImpactCollisionFlag).toBe(false);
    }
  }
});

test("Agent POV damage and healing landings stop outside the recipient body without intercepting its center", async ({
  page,
}) => {
  await page.setViewportSize(MINIMUM_VIEWPORT);
  const cases = [
    {
      leafName: "live_no_shared_obs_agent_pov",
      expectedSemantic: "healing",
    },
    {
      leafName: "live_no_shared_obs_agent_pov",
      expectedSemantic: "damage",
    },
    {
      leafName: "replay_no_shared_obs_agent_pov",
      expectedSemantic: "healing",
    },
    {
      leafName: "replay_no_shared_obs_agent_pov",
      expectedSemantic: "damage",
    },
  ];

  for (const { leafName, expectedSemantic } of cases) {
    const raw = structuredClone(fixture.presentations[leafName]);
    const activation = raw.visual_events.events.find(
      (/** @type {Record<string, any>} */ event) =>
        event.event_kind === "ability_activated",
    );
    if (!activation) {
      throw new Error(`${leafName} Agent activation is unavailable.`);
    }
    if (expectedSemantic === "damage") {
      const sourceTrajectory = raw.visual_events.agent_phase_trajectories.find(
        (/** @type {Record<string, any>} */ trajectory) =>
          trajectory.agent_class_id === 1 && trajectory.successor !== null,
      );
      const targetTrajectory = raw.visual_events.agent_phase_trajectories.find(
        (/** @type {Record<string, any>} */ trajectory) =>
          trajectory.agent_class_id === 2 && trajectory.successor !== null,
      );
      if (!sourceTrajectory?.transition_start || !targetTrajectory?.transition_start) {
        throw new Error(`${leafName} Agent damage trajectories are unavailable.`);
      }
      activation.source_anchor = structuredClone(sourceTrajectory.transition_start);
      activation.recipient_anchor = structuredClone(targetTrajectory.transition_start);
    }

    await page.goto(origin);
    const geometry = await page.evaluate(
      async ({ rawPresentation, expectedImpactSemantic }) => {
        const moduleRoot = "/src";
        const { normalizeAuthorizedPresentationFrameV1 } = await import(
          `${moduleRoot}/authorized-presentation-normalizer.js`
        );
        const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
        const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
          `${moduleRoot}/visual-filters.js`
        );
        const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
        const { buildChoreographyPlan } = await import(
          `${moduleRoot}/choreography-plan.js`
        );
        const { SvgChoreographyPainter } = await import(
          `${moduleRoot}/choreography-painter.js`
        );
        const presentation =
          await normalizeAuthorizedPresentationFrameV1(rawPresentation);
        const battlefield = document.querySelector("#battlefield");
        const empty = document.querySelector("#empty");
        if (
          !(battlefield instanceof SVGSVGElement) ||
          !(empty instanceof HTMLElement)
        ) {
          throw new Error("Agent impact test surface is unavailable.");
        }
        document.body.style.margin = "0";
        battlefield.style.width = `${window.innerWidth}px`;
        battlefield.style.height = `${window.innerHeight}px`;
        battlefield.style.display = "block";
        const renderer = new BattlefieldRenderer({ battlefield, empty });
        renderer.render(presentation, {
          showRanges: false,
          visualFilterState: allVisualFilters,
        });
        const surface = renderer.choreographySurface();
        const plan = buildChoreographyPlan(presentation, surface, allVisualFilters);
        if (surface === null || plan === null) {
          throw new Error("Agent impact choreography is unavailable.");
        }
        new SvgChoreographyPainter().install(plan, surface, {
          motionMode: "off",
          settled: false,
          persistentOnly: false,
        });
        const event = /** @type {Record<string, any> | undefined} */ (
          plan.events.find(
            (/** @type {Record<string, any>} */ candidate) =>
              candidate.kind === "activation" &&
              candidate.component === "basic" &&
              candidate.impactSemantic === expectedImpactSemantic &&
              candidate.route,
          )
        );
        if (!event || typeof event.targetPresentationKey !== "string") {
          throw new Error(
            `Agent ${expectedImpactSemantic} activation route is unavailable.`,
          );
        }
        const escapedEventId = CSS.escape(event.eventId);
        const effect = battlefield.querySelector(
          `.combat-effect--activation[data-event-id="${escapedEventId}"]`,
        );
        const landing = effect?.querySelector(".combat-impact.basic-landing");
        const stroke = battlefield.querySelector(
          `.combat-route-effect--activation[data-event-id="${escapedEventId}"] .basic-blow__core`,
        );
        const targetAgent = battlefield.querySelector(
          `.agent[data-presentation-key="${CSS.escape(event.targetPresentationKey)}"]`,
        );
        const body = targetAgent?.querySelector(".agent-body");
        const camera = battlefield.querySelector(".battlefield-camera");
        if (
          !(effect instanceof SVGElement) ||
          !(landing instanceof SVGGraphicsElement) ||
          !(stroke instanceof SVGPathElement) ||
          !(targetAgent instanceof SVGElement) ||
          !(body instanceof SVGCircleElement) ||
          !(camera instanceof SVGGElement)
        ) {
          throw new Error("Agent impact or recipient body is unavailable.");
        }
        const bodyMatrix = body.getScreenCTM();
        const cameraMatrix = camera.getScreenCTM()?.inverse();
        if (bodyMatrix === null || !cameraMatrix) {
          throw new Error("Agent impact or recipient screen geometry is unavailable.");
        }
        // Map a point in a node's own frame to the plan's CSS pixel frame.
        /** @param {SVGGraphicsElement} node @param {DOMPointInit} point */
        const toField = (node, point) => {
          const matrix = node.getScreenCTM();
          if (matrix === null) {
            throw new Error("Agent impact node has no screen transform.");
          }
          const field = new DOMPoint(point.x, point.y)
            .matrixTransform(matrix)
            .matrixTransform(cameraMatrix);
          return { x: field.x, y: field.y };
        };
        const strokeEnd = stroke.getPointAtLength(stroke.getTotalLength());
        const strokeEndOnScreen = new DOMPoint(
          strokeEnd.x,
          strokeEnd.y,
        ).matrixTransform(stroke.getScreenCTM() ?? undefined);
        // The stroke stops on the landing's edge; allow 0.01 px of rounding.
        const landingBox = landing.getBoundingClientRect();
        const edge = 0.01;
        const centre = { x: body.cx.baseVal.value, y: body.cy.baseVal.value };
        const bodyRadius = body.r.baseVal.value;
        const routeEndDistance = Math.hypot(
          event.route.end.x - centre.x,
          event.route.end.y - centre.y,
        );
        const landingCenter = toField(landing, { x: 0, y: 0 });
        const impactDistance = Math.hypot(
          landingCenter.x - centre.x,
          landingCenter.y - centre.y,
        );
        const targetCentreDistance = Math.hypot(
          event.target.x - centre.x,
          event.target.y - centre.y,
        );
        const screenCentre = new DOMPoint(centre.x, centre.y).matrixTransform(
          bodyMatrix,
        );
        battlefield.addEventListener(
          "pointerdown",
          (pointerEvent) => {
            const pointerTarget =
              pointerEvent.target instanceof Element ? pointerEvent.target : null;
            battlefield.dataset.centrePointerAgentKey =
              pointerTarget
                ?.closest(".agent[data-presentation-key]")
                ?.getAttribute("data-presentation-key") ?? "";
            battlefield.dataset.centrePointerInterceptedByImpact = String(
              pointerTarget?.closest(".combat-impact, .basic-blow") !== null,
            );
          },
          { capture: true, once: true },
        );
        return {
          audience: battlefield.dataset.audience,
          component: event.component,
          impactSemantic: event.impactSemantic,
          sourceEndpointPhase: event.sourceEndpointPhase,
          targetEndpointPhase: event.targetEndpointPhase,
          targetPresentationKey: event.targetPresentationKey,
          bodyRadius,
          routeEndDistance,
          impactDistance,
          targetCentreDistance,
          sourceClass: effect.getAttribute("data-source-class"),
          landingClass: landing.getAttribute("class"),
          semanticMarkCount: effect.querySelectorAll(".combat-impact__semantic").length,
          blow: {
            source: event.source,
            target: event.target,
            sourceRadius: event.sourceRadius,
            targetRadius: bodyRadius,
            start: toField(stroke, stroke.getPointAtLength(0)),
            end: toField(stroke, strokeEnd),
            landingCenter,
            endInsideLanding:
              strokeEndOnScreen.x >= landingBox.left - edge &&
              strokeEndOnScreen.x <= landingBox.right + edge &&
              strokeEndOnScreen.y >= landingBox.top - edge &&
              strokeEndOnScreen.y <= landingBox.bottom + edge,
          },
          screenCentre: { x: screenCentre.x, y: screenCentre.y },
          viewport: { width: window.innerWidth, height: window.innerHeight },
        };
      },
      { rawPresentation: raw, expectedImpactSemantic: expectedSemantic },
    );

    await page.mouse.click(geometry.screenCentre.x, geometry.screenCentre.y);
    const pointerTarget = await page
      .locator("#battlefield")
      .evaluate((battlefield) => ({
        agentKey: battlefield.dataset.centrePointerAgentKey ?? null,
        interceptedByImpact:
          battlefield.dataset.centrePointerInterceptedByImpact ?? null,
      }));

    expect(geometry.audience, `${leafName} ${expectedSemantic}`).toBe("agent_pov");
    expect(geometry.viewport).toEqual(MINIMUM_VIEWPORT);
    expect(geometry.component).toBe("basic");
    expect(geometry.impactSemantic).toBe(expectedSemantic);
    expect(geometry.sourceEndpointPhase).toBe("successor");
    expect(geometry.targetEndpointPhase).toBe("successor");
    expect(geometry.targetCentreDistance).toBeCloseTo(0, 3);
    expect(geometry.routeEndDistance).toBeCloseTo(geometry.bodyRadius + 3, 3);
    // The landing stops outside the recipient body, where the stroke ends.
    expect(geometry.impactDistance).toBeGreaterThan(geometry.bodyRadius);
    expectBasicBlowGeometry(geometry.blow, `${leafName} ${expectedSemantic}`);
    expect(geometry.sourceClass).toBe(
      expectedSemantic === "healing" ? "priest" : "mage",
    );
    expect(geometry.landingClass).toContain(
      expectedSemantic === "healing" ? "basic-landing--bloom" : "basic-landing--star",
    );
    expect(geometry.semanticMarkCount).toBe(0);
    expect(pointerTarget.agentKey).toBe(geometry.targetPresentationKey);
    expect(pointerTarget.interceptedByImpact).toBe("false");
  }
});

test("live and replay keep Charge displacement authorized without painting an overlay", async ({
  page,
}) => {
  for (const leafName of ["live_oracle", "replay_oracle"]) {
    const raw = structuredClone(fixture.presentations[leafName]);
    const warrior = raw.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.class_id === 2,
    );
    const trajectory = raw.latest_events.agent_phase_trajectories.find(
      (/** @type {Record<string, any>} */ candidate) =>
        candidate.agent_presentation_key === warrior?.presentation_key,
    );
    if (!warrior || !trajectory) {
      throw new Error(`${leafName} Warrior trajectory is unavailable.`);
    }
    trajectory.transition_start.position = [3, 4];
    trajectory.post_charge.position = [7, 5];
    const event = {
      event_id: `${raw.latest_events.incoming_transition_id}:event:0000`,
      event_kind: "charge_phase_displacement",
      ordinal: 0,
      phase_rank: 70,
      realized_displacement: [4, 1],
      start_anchor: structuredClone(trajectory.transition_start),
      end_anchor: structuredClone(trajectory.post_charge),
    };
    raw.latest_events.events = [event];
    raw.latest_events.event_count = 1;
    raw.latest_events.ordered_event_ids = [event.event_id];
    raw.latest_events.ordered_event_kinds = [event.event_kind];

    await page.goto(origin);
    const result = await page.evaluate(async (rawPresentation) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
        `${moduleRoot}/visual-filters.js`
      );
      const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
      const { buildChoreographyPlan } = await import(
        `${moduleRoot}/choreography-plan.js`
      );
      const { SvgChoreographyPainter } = await import(
        `${moduleRoot}/choreography-painter.js`
      );
      const presentation =
        await normalizeAuthorizedPresentationFrameV1(rawPresentation);
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Charge displacement test surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      renderer.render(presentation, {
        showRanges: false,
        visualFilterState: allVisualFilters,
      });
      const surface = renderer.choreographySurface();
      const plan = buildChoreographyPlan(presentation, surface, allVisualFilters);
      if (surface === null || plan === null) {
        throw new Error("Authorized Charge event is unavailable.");
      }
      new SvgChoreographyPainter().install(plan, surface, {
        motionMode: "off",
        settled: false,
        persistentOnly: false,
      });
      const charge = /** @type {Record<string, any>} */ (plan.events[0]);
      return {
        presentationKind: presentation.presentation_kind,
        eventType: charge.eventType,
        kind: charge.kind,
        spatial: charge.spatial,
        geometryFields: [
          "start",
          "end",
          "route",
          "sourcePresentationKey",
          "sourcePublicAgentId",
          "paintParts",
          "persistent",
        ].filter((field) => Object.hasOwn(charge, field)),
        pathCount: battlefield.querySelectorAll(".combat-charge__path").length,
        endpointCount: battlefield.querySelectorAll(".combat-charge__endpoint").length,
        directionCount: battlefield.querySelectorAll(".combat-charge__direction")
          .length,
        leaderCount: battlefield.querySelectorAll(
          ".combat-cue__leader--charge-start, .combat-cue__leader--charge-end",
        ).length,
      };
    }, raw);

    expect(result.presentationKind).toBe(leafName);
    expect(result.eventType).toBe("charge_phase_displacement");
    expect(result.kind).toBe("feed_only");
    expect(result.spatial).toBe(false);
    expect(result.geometryFields).toEqual([]);
    expect(result.pathCount).toBe(0);
    expect(result.endpointCount).toBe(0);
    expect(result.directionCount).toBe(0);
    expect(result.leaderCount).toBe(0);
  }
});

test("digest-valid OOC state keeps all five class glyphs centered across projected radii", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Renderer test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const viewports = [
      { width: 360, height: 240 },
      { width: 720, height: 480 },
      { width: 1080, height: 720 },
    ];
    const states = [];
    for (const viewport of viewports) {
      battlefield.style.width = `${viewport.width}px`;
      battlefield.style.height = `${viewport.height}px`;
      renderer.render(presentation, {
        showRanges: false,
        visualFilterState: allVisualFilters,
      });
      states.push(
        ...Array.from(battlefield.querySelectorAll(".agent")).map((agent) => {
          const classIcon = agent.querySelector(".agent-class-icon");
          const body = agent.querySelector(".agent-body");
          if (
            !(classIcon instanceof SVGSVGElement) ||
            !(body instanceof SVGCircleElement)
          ) {
            throw new Error("Agent identity glyphs are unavailable.");
          }
          const classX = Number(classIcon.getAttribute("x"));
          const classWidth = Number(classIcon.getAttribute("width"));
          const bodyCenter = Number(body.getAttribute("cx"));
          return {
            classToken: agent.getAttribute("data-class"),
            viewport: `${battlefield.clientWidth}x${battlefield.clientHeight}`,
            projectedRadius: Number(body.getAttribute("r")),
            status: agent.getAttribute("data-combat-status"),
            countdown: agent.getAttribute("data-steps-until-out-of-combat"),
            ariaLabel: agent.getAttribute("aria-label"),
            dedicatedCombatIconCount: agent.querySelectorAll(".agent-combat-state-icon")
              .length,
            inCombatStatusCount: battlefield.querySelectorAll(
              '.status-cell[data-token-id="in_combat"]',
            ).length,
            classCentered: Math.abs(classX + classWidth / 2 - bodyCenter) < 0.001,
          };
        }),
      );
    }
    return states;
  }, fixture.presentations.replay_oracle);

  expect(new Set(result.map(({ classToken }) => classToken))).toEqual(
    new Set(["mage", "warrior", "hunter", "rogue", "priest"]),
  );
  expect(new Set(result.map(({ viewport }) => viewport)).size).toBe(3);
  expect(
    new Set(result.map(({ projectedRadius }) => projectedRadius.toFixed(3))).size,
  ).toBe(3);
  expect(result).toHaveLength(15);
  for (const state of result) {
    expect(state.dedicatedCombatIconCount).toBe(0);
    expect(state.inCombatStatusCount).toBe(0);
    expect(state.status).toBe("OOC");
    expect(state.countdown).toBe("0");
    expect(state.ariaLabel).not.toMatch(/combat|steps until out/iu);
    expect(state.classCentered).toBe(true);
  }
});

test("authorized regeneration paints packed successor heal numbers while reset stays feed-only", async ({
  page,
}) => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const transitionId = raw.latest_events.incoming_transition_id;
  const trajectories = raw.latest_events.agent_phase_trajectories;
  const specifications = [
    {
      event_kind: "combat_countdown_reset",
      agent_anchor: structuredClone(trajectories[0].transition_start),
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: structuredClone(trajectories[0].transition_start),
      actual_health_regenerated: 4,
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: structuredClone(trajectories[1].transition_start),
      actual_health_regenerated: 2,
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: structuredClone(trajectories[2].transition_start),
      actual_health_regenerated: 1,
    },
  ];
  raw.latest_events.events = specifications.map((event, ordinal) => ({
    ...event,
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
    phase_rank: 50,
  }));
  raw.latest_events.event_count = raw.latest_events.events.length;
  raw.latest_events.ordered_event_ids = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_id,
  );
  raw.latest_events.ordered_event_kinds = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_kind,
  );

  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Renderer test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, {
      showRanges: false,
      visualFilterState: allVisualFilters,
    });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface, allVisualFilters);
    if (surface === null || plan === null) {
      throw new Error("Authorized regeneration choreography is unavailable.");
    }
    new SvgChoreographyPainter().install(plan, surface, {
      motionMode: "off",
      settled: false,
      persistentOnly: false,
    });
    const plannedEvents = /** @type {Record<string, any>[]} */ (plan.events);
    const regenerations = plannedEvents.filter(
      (event) => event.kind === "regeneration",
    );
    const reset = plannedEvents.find(
      (event) => event.eventType === "combat_countdown_reset",
    );
    const effects = Array.from(
      battlefield.querySelectorAll(".combat-effect--regeneration"),
    );
    return {
      planEventIds: plannedEvents.map((event) => event.eventId),
      reset: reset
        ? {
            kind: reset.kind,
            spatial: reset.spatial,
            hasPhase: Object.hasOwn(reset, "phaseStart"),
          }
        : null,
      regenerationValues: regenerations.map((event) => event.value),
      regenerationAnchors: regenerations.map((event) => event.recipient),
      expectedRegenerationAnchors: /** @type {Record<string, any>[]} */ (
        presentation.latest_events.agent_phase_trajectories
      )
        .slice(0, 3)
        .map(({ successor }) => surface.worldToScreen(successor.position)),
      regenerationCues: regenerations.map((event) => event.cue),
      regenerationBounds: regenerations.map((event) => event.cueBounds),
      regenerationCollisionFree: regenerations.map((event) => event.cueCollisionFree),
      regenerationPhases: regenerations.map((event) => [
        event.phaseStart,
        event.phaseEnd,
      ]),
      effectCount: effects.length,
      // Each heal is one rest-toned battle-text number with its hover box.
      values: effects.map(
        (effect) => effect.querySelector(".combat-number")?.textContent ?? null,
      ),
      tones: effects.map(
        (effect) =>
          effect.querySelector(".combat-number")?.getAttribute("data-tone") ?? null,
      ),
      numberAt: effects.map((effect) => {
        const number = effect.querySelector(".combat-number");
        return number
          ? { x: Number(number.getAttribute("x")), y: Number(number.getAttribute("y")) }
          : null;
      }),
      hitBoxes: effects.map((effect) => {
        const hit = effect.querySelector(".combat-regeneration__hit");
        return hit
          ? {
              left: Number(hit.getAttribute("x")),
              top: Number(hit.getAttribute("y")),
              width: Number(hit.getAttribute("width")),
              height: Number(hit.getAttribute("height")),
            }
          : null;
      }),
      tooltipOwnerCounts: effects.map((effect) =>
        effect.hasAttribute("data-tooltip-owner") ? 1 : 0,
      ),
      sourceAttributeCount: effects.reduce(
        (count, effect) =>
          count +
          Array.from(effect.attributes).filter((attribute) =>
            attribute.name.includes("source"),
          ).length,
        0,
      ),
      onionCount: battlefield.querySelectorAll(
        ".combat-effect--regeneration .combat-semantic-pulse__ring, .combat-effect--regeneration .combat-semantic-pulse__core",
      ).length,
      resetEffectCount: battlefield.querySelectorAll(
        '.combat-effect[data-event-type="combat_countdown_reset"]',
      ).length,
      routeChildCount: battlefield.querySelectorAll(".combat-choreography-routes > *")
        .length,
    };
  }, raw);

  expect(result.planEventIds).toEqual(raw.latest_events.ordered_event_ids);
  expect(result.reset).toEqual({ kind: "feed_only", spatial: false, hasPhase: false });
  expect(result.regenerationValues).toEqual([4, 2, 1]);
  expect(result.regenerationAnchors).toEqual(result.expectedRegenerationAnchors);
  expect(
    result.regenerationCues.every(
      (/** @type {Record<string, number> | null} */ cue) => cue !== null,
    ),
  ).toBe(true);
  expect(result.regenerationCollisionFree).toEqual([true, true, true]);
  expect(
    new Set(
      result.regenerationPhases.map((/** @type {number[]} */ phase) =>
        JSON.stringify(phase),
      ),
    ),
  ).toEqual(new Set([JSON.stringify(result.regenerationPhases[0])]));
  for (const [index, bounds] of result.regenerationBounds.entries()) {
    for (const prior of result.regenerationBounds.slice(0, index)) {
      const overlap =
        Math.max(
          0,
          Math.min(bounds.right, prior.right) - Math.max(bounds.left, prior.left),
        ) *
        Math.max(
          0,
          Math.min(bounds.bottom, prior.bottom) - Math.max(bounds.top, prior.top),
        );
      expect(overlap).toBe(0);
    }
  }
  expect(result.effectCount).toBe(3);
  expect(result.values).toEqual(["+4", "+2", "+1"]);
  expect(result.tones).toEqual(["rest", "rest", "rest"]);
  // Each number sits on its planned cue and its hover box is the planned box.
  for (const [index, cue] of result.regenerationCues.entries()) {
    expect(result.numberAt[index]?.x).toBeCloseTo(cue.x, 6);
    expect(result.numberAt[index]?.y).toBeCloseTo(cue.y, 6);
    const bounds = result.regenerationBounds[index];
    expect(result.hitBoxes[index]).toEqual({
      left: expect.closeTo(bounds.left, 6),
      top: expect.closeTo(bounds.top, 6),
      width: expect.closeTo(bounds.width, 6),
      height: expect.closeTo(bounds.height, 6),
    });
  }
  expect(result.tooltipOwnerCounts).toEqual([1, 1, 1]);
  expect(result.sourceAttributeCount).toBe(0);
  expect(result.onionCount).toBe(0);
  expect(result.resetEffectCount).toBe(0);
  expect(result.routeChildCount).toBe(0);
});

test("shared painter removes cooldown start clutter and keeps clear trap and poison semantics", async ({
  page,
}) => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const transitionId = raw.latest_events.incoming_transition_id;
  const agentsByKey = new Map(
    raw.current_endpoint.scene.agents.map(
      (/** @type {Record<string, any>} */ agent) => [agent.presentation_key, agent],
    ),
  );
  /** @param {number} classId @param {"transition_start" | "successor"} phase */
  const anchor = (classId, phase) => {
    const trajectory = raw.latest_events.agent_phase_trajectories.find(
      (/** @type {Record<string, any>} */ candidate) =>
        agentsByKey.get(candidate.agent_presentation_key)?.class_id === classId,
    );
    if (!trajectory) {
      throw new Error(`Oracle class ${classId} trajectory is unavailable.`);
    }
    return structuredClone(trajectory[phase]);
  };
  const specifications = [
    {
      event_kind: "ability_activated",
      ability_component: "ultimate",
      source_anchor: anchor(4, "transition_start"),
      recipient_anchor: anchor(3, "transition_start"),
      phase_rank: 20,
    },
    {
      event_kind: "cooldown_started",
      agent_anchor: anchor(4, "transition_start"),
      phase_rank: 60,
    },
    {
      event_kind: "cooldown_ready",
      agent_anchor: anchor(4, "transition_start"),
      phase_rank: 60,
    },
    {
      event_kind: "status_broken_by_damage",
      recipient_anchor: anchor(3, "successor"),
      status_channel: 4,
      status_id: "hunter_trap_stun",
      phase_rank: 100,
    },
  ];
  raw.latest_events.events = specifications.map((event, ordinal) => ({
    ...event,
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
  }));
  raw.latest_events.event_count = raw.latest_events.events.length;
  raw.latest_events.ordered_event_ids = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_id,
  );
  raw.latest_events.ordered_event_kinds = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_kind,
  );

  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Shared cue test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, {
      showRanges: false,
      visualFilterState: allVisualFilters,
    });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface, allVisualFilters);
    if (surface === null || plan === null) {
      throw new Error("Authorized shared cue choreography is unavailable.");
    }
    new SvgChoreographyPainter().install(plan, surface, {
      motionMode: "off",
      settled: false,
      persistentOnly: false,
    });
    const planEvents = /** @type {Record<string, any>[]} */ (plan.events);
    const cooldownStarted = planEvents.find(
      (event) => event.cueSemantic === "cooldown_started",
    );
    const cooldownReady = planEvents.find(
      (event) => event.cueSemantic === "cooldown_ready",
    );
    if (!cooldownStarted || !cooldownReady) {
      throw new Error("Cooldown plan events are unavailable.");
    }
    const poisonEvent = planEvents.find((event) => event.tokenId === "rogue_poison");
    const trapEvent = planEvents.find((event) => event.lifecycle === "trap_broken");
    const poisonEffect = battlefield.querySelector(
      '.combat-effect--activation[data-token-id="rogue_poison"]',
    );
    const poisonRoute = battlefield.querySelector(
      '.combat-route-effect--activation[data-token-id="rogue_poison"]',
    );
    const poison = poisonEffect?.querySelector(
      '.ult--poison[data-ult-variant="poison"]',
    );
    const strikeFrame = poison?.querySelector('[data-ult-key="poison.strikeFrame"]');
    // A Freezing Trap break shows on the frozen champion's own token.
    const trapMarks =
      typeof trapEvent?.recipientPresentationKey === "string"
        ? battlefield.querySelector(
            `.status-marks[data-marks-for="${CSS.escape(trapEvent.recipientPresentationKey)}"]`,
          )
        : null;
    if (
      !poisonEvent ||
      !(poisonEffect instanceof SVGElement) ||
      !(poisonRoute instanceof SVGElement) ||
      !(poison instanceof SVGElement) ||
      !(strikeFrame instanceof SVGElement) ||
      !(trapMarks instanceof SVGElement)
    ) {
      throw new Error("Trap or poisoned-dagger cue is unavailable.");
    }
    /** @param {Element} parent @param {string} key */
    const ultOpacity = (parent, key) =>
      Array.from(parent.querySelectorAll(`[data-ult-key="${key}"]`), (node) =>
        node.getAttribute("opacity"),
      );
    /** @param {string} piece */
    const rune = (piece) => {
      const node = trapMarks.querySelector(`[data-rune-piece="${piece}"]`);
      return node
        ? {
            hold: node.getAttribute("data-beat-hold"),
            side: node.getAttribute("data-beat-side"),
            subpaths: (node.getAttribute("d")?.match(/M/gu) ?? []).length,
          }
        : null;
    };

    const forgedStartPlan = {
      ...plan,
      events: [
        {
          ...cooldownReady,
          eventId: `${cooldownReady.eventId}:defense`,
          eventType: "cooldown_started",
          cueSemantic: "cooldown_started",
          paintParts: { effect: true },
          presentationSuppressed: false,
          spatial: true,
        },
      ],
    };
    const defensiveInstallation = new SvgChoreographyPainter().install(
      forgedStartPlan,
      surface,
      {
        motionMode: "off",
        settled: false,
        persistentOnly: false,
      },
    );
    return {
      cooldownStartedPlan: {
        paintParts: cooldownStarted.paintParts,
        presentationSuppressed: cooldownStarted.presentationSuppressed,
        spatial: cooldownStarted.spatial,
        anchor: cooldownStarted.anchor,
      },
      cooldownReadyPlan: {
        paintParts: cooldownReady.paintParts,
        presentationSuppressed: cooldownReady.presentationSuppressed,
        spatial: cooldownReady.spatial,
        hasAnchor: cooldownReady.anchor !== null,
      },
      cooldownStartedDomCount: battlefield.querySelectorAll(
        '[data-cue-semantic="cooldown_started"]',
      ).length,
      cooldownReadyDomCount: battlefield.querySelectorAll(
        '[data-cue-semantic="cooldown_ready"] .combat-semantic-pulse__mark',
      ).length,
      defensiveEventNodeCount: defensiveInstallation.eventNodes.size,
      poison: {
        target: poisonEvent.target,
        source: poisonEvent.source,
        strikeFrame: strikeFrame.getAttribute("transform"),
        strike: ultOpacity(poison, "poison.strike"),
        flash: ultOpacity(poison, "poison.flash"),
        seep: ultOpacity(poison, "poison.seep"),
        rest: ultOpacity(poisonRoute, "poison.rest"),
        tether: ultOpacity(poisonRoute, "poison.tether"),
        semanticMarkCount: poisonEffect.querySelectorAll(".combat-impact__semantic")
          .length,
      },
      trap: {
        offBodyCount: battlefield.querySelectorAll(
          ".combat-effect--status-lifecycle, .combat-route-effect--status-lifecycle",
        ).length,
        shell: rune("frozen.shell"),
        crack: rune("shatter.crack"),
        shards: rune("shatter.shards"),
      },
    };
  }, raw);

  expect(result.cooldownStartedPlan).toEqual({
    paintParts: { effect: false },
    presentationSuppressed: true,
    spatial: false,
    anchor: null,
  });
  expect(result.cooldownReadyPlan).toEqual({
    paintParts: { effect: true },
    presentationSuppressed: false,
    spatial: true,
    hasAnchor: true,
  });
  expect(result.cooldownStartedDomCount).toBe(0);
  expect(result.cooldownReadyDomCount).toBe(1);
  expect(result.defensiveEventNodeCount).toBe(0);
  // Crippling Poison is the Rogue's stiletto on the target's rim, pointing at
  // the Rogue, with one venom seep; no separate damage mark competes with it.
  // A still shows only the resting blade (70%) and its tether (30%): the
  // strike, the tip flash and the seep play and are hidden afterwards.
  const { source, target } = result.poison;
  const facing = (Math.atan2(source.y - target.y, source.x - target.x) * 180) / Math.PI;
  const frame = /^translate\(([-\d.]+) ([-\d.]+)\) rotate\(([-\d.]+)\)$/u.exec(
    result.poison.strikeFrame ?? "",
  );
  expect(frame).not.toBeNull();
  expect(Number(frame?.[1])).toBeCloseTo(target.x, 1);
  expect(Number(frame?.[2])).toBeCloseTo(target.y, 1);
  expect(Number(frame?.[3])).toBeCloseTo(facing, 1);
  expect(result.poison.strike).toEqual(["0"]);
  expect(result.poison.flash).toEqual(["0"]);
  expect(result.poison.seep).toEqual(["0"]);
  expect(result.poison.rest).toEqual(["0.7"]);
  expect(result.poison.tether).toEqual(["0.3"]);
  expect(result.poison.semanticMarkCount).toBe(0);
  // A broken Freezing Trap draws no off-body badge: the ice shell holds until
  // Settle, the crack comes at Impact and six shards fall at Settle.
  expect(result.trap).toEqual({
    offBodyCount: 0,
    shell: { hold: "settle", side: "before", subpaths: 1 },
    crack: { hold: "impact", side: "after", subpaths: expect.any(Number) },
    shards: { hold: "settle", side: "after", subpaths: 6 },
  });
  expect(result.trap.crack?.subpaths).toBeGreaterThan(0);
});

test("authorized falls and returns settle on their bodies while team waves stay off the arena", async ({
  page,
}) => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const transitionId = raw.latest_events.incoming_transition_id;
  const trajectories = raw.latest_events.agent_phase_trajectories;
  const deathAnchor = structuredClone(trajectories[4].successor);
  const shieldAnchor = structuredClone(trajectories[0].successor);
  const respawnAnchor = structuredClone(trajectories[3].successor);
  const specifications = [
    {
      event_kind: "agent_died",
      recipient_anchor: deathAnchor,
      phase_rank: 90,
    },
    {
      event_kind: "spawn_shield_expired",
      agent_anchor: shieldAnchor,
      phase_rank: 110,
    },
    {
      event_kind: "respawn_wave_occurred",
      team_anchor: { phase: "successor", team_index: 0, team_id: 1 },
      phase_rank: 120,
    },
    {
      event_kind: "respawn_wave_occurred",
      team_anchor: { phase: "successor", team_index: 1, team_id: 2 },
      phase_rank: 120,
    },
    {
      event_kind: "agent_respawned",
      agent_anchor: respawnAnchor,
      team_id: 2,
      realized_successor_position: structuredClone(respawnAnchor.position),
      phase_rank: 120,
    },
  ];
  raw.latest_events.events = specifications.map((event, ordinal) => ({
    ...event,
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
  }));
  raw.latest_events.event_count = raw.latest_events.events.length;
  raw.latest_events.ordered_event_ids = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_id,
  );
  raw.latest_events.ordered_event_kinds = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_kind,
  );

  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const { createTooltipController } = await import(`${moduleRoot}/tooltip.js`);
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const serializedBefore = JSON.stringify(presentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    const tooltip = document.querySelector("#visual-tooltip");
    const title = document.querySelector("#visual-tooltip-title");
    const details = document.querySelector("#visual-tooltip-details");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement) ||
      !(tooltip instanceof HTMLElement) ||
      !(title instanceof HTMLElement) ||
      !(details instanceof HTMLElement)
    ) {
      throw new Error("Renderer test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, {
      showRanges: false,
      visualFilterState: allVisualFilters,
    });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface, allVisualFilters);
    if (surface === null || plan === null) {
      throw new Error("Authorized lifecycle choreography is unavailable.");
    }
    const ringSelector = ".requiem__fall-ring, .homecoming__ring";
    /** @param {Element} ring */
    const stillOpacity = (ring) => {
      // A ring shows in a still at its own opacity times its groups' opacity.
      let opacity = 1;
      for (
        let node = /** @type {Element | null} */ (ring);
        node !== null && !node.classList.contains("combat-lifecycle-ring");
        node = node.parentElement
      ) {
        const value = node.getAttribute("opacity");
        opacity *= value === null ? 1 : Number(value);
      }
      return opacity;
    };
    /** @param {Element} ring */
    const ringRadius = (ring) =>
      ring instanceof SVGGraphicsElement ? ring.getBBox().width / 2 : Number.NaN;
    /** @param {Record<string, any>} installation */
    const ringAnimations = (installation) =>
      installation.animationSpecs
        .filter((/** @type {Record<string, any>} */ spec) =>
          /:(?:fall-ring|homecoming)$/u.test(spec.id),
        )
        .map((/** @type {Record<string, any>} */ spec) => ({
          id: spec.id.split(":").at(-1),
          keyframes: spec.keyframes.map((/** @type {Record<string, any>} */ frame) => ({
            opacity: Number(frame.opacity),
            transform: frame.transform ?? null,
          })),
        }));
    const painter = new SvgChoreographyPainter();
    const normal = painter.install(plan, surface, {
      motionMode: "normal",
      settled: false,
      persistentOnly: false,
    });
    const controller = createTooltipController({
      root: document.body,
      tooltip,
      title,
      details,
    });
    const lifecycleGeometry = Array.from(
      battlefield.querySelectorAll(".combat-lifecycle-ring"),
    ).map((node) => {
      const effect = node.closest(".combat-effect");
      const event = plan.events.find(
        (/** @type {Record<string, any>} */ candidate) =>
          candidate.eventId === effect?.getAttribute("data-event-id"),
      );
      const ring = node.querySelector(ringSelector);
      const hit = node.querySelector(".combat-lifecycle-ring__hit");
      const agent =
        typeof event?.agentPresentationKey === "string"
          ? battlefield.querySelector(
              `.agent[data-presentation-key="${CSS.escape(event.agentPresentationKey)}"]`,
            )
          : null;
      const body = agent?.querySelector(".agent-body");
      if (
        !(node instanceof SVGGraphicsElement) ||
        !(effect instanceof SVGElement) ||
        !event?.anchor ||
        !(ring instanceof SVGGraphicsElement) ||
        !(hit instanceof SVGCircleElement) ||
        !(agent instanceof SVGElement) ||
        !(body instanceof SVGCircleElement)
      ) {
        throw new Error("Authorized lifecycle geometry is unavailable.");
      }
      const ringMatrix = node.getScreenCTM();
      const bodyMatrix = body.getScreenCTM();
      if (ringMatrix === null || bodyMatrix === null) {
        throw new Error("Authorized lifecycle transform is unavailable.");
      }
      const center = new DOMPoint(0, 0).matrixTransform(ringMatrix);
      const bodyCenter = new DOMPoint(
        body.cx.baseVal.value,
        body.cy.baseVal.value,
      ).matrixTransform(bodyMatrix);
      const centerElements = document.elementsFromPoint(bodyCenter.x, bodyCenter.y);
      effect.focus();
      return {
        eventType: event.eventType,
        lifecycle: node.getAttribute("data-lifecycle-ring"),
        ring: ring.getAttribute("class"),
        stroke: getComputedStyle(ring).stroke,
        radiusInBodies: ringRadius(ring) / body.r.baseVal.value,
        stillOpacity: stillOpacity(ring),
        center: { x: center.x, y: center.y },
        bodyCenter: { x: bodyCenter.x, y: bodyCenter.y },
        planLayoutFields: [
          "cue",
          "cueBounds",
          "cueLeader",
          "cueDisposition",
          "cueCollisionFree",
          "cueLayoutKey",
        ].filter((field) => Object.hasOwn(event, field)),
        domLayoutAttributes: Array.from(node.attributes, ({ name }) => name).filter(
          (name) => name.startsWith("data-layout-"),
        ),
        semanticLeaderCount: effect.querySelectorAll(".combat-cue__leader--semantic")
          .length,
        hitOwnsBodyCenter: centerElements.includes(hit),
        agentOwnsBodyCenter: centerElements.some(
          (element) => element.closest(".agent") === agent,
        ),
        hitPointerEvents: getComputedStyle(hit).pointerEvents,
        hitFill: getComputedStyle(hit).fill,
        ariaLabel: effect.getAttribute("aria-label"),
        tooltipTitle: title.textContent,
        tooltipSummary:
          tooltip.querySelector(".semantic-explanation__summary")?.textContent ?? null,
      };
    });
    controller.destroy();
    const beforeSettle = {
      eventCount: normal.eventNodes.size,
      nodeCount: normal.nodeCount,
      persistentNodeCount: normal.persistentNodeCount,
      persistentNodeBound: plan.bounds.persistentNodes,
      lifecycleGeometry,
      ringAnimations: ringAnimations(normal),
      waveRows: plan.events
        .filter(
          (/** @type {Record<string, any>} */ event) =>
            event.eventType === "respawn_wave_occurred",
        )
        .map((/** @type {Record<string, any>} */ event) => ({
          teamIndex: event.teamIndex,
          teamId: event.teamId,
          side: event.teamSide,
          anchor: event.anchor ?? null,
        })),
      arenaWaveCount: battlefield.querySelectorAll(
        '.combat-effect[data-event-type="respawn_wave_occurred"], .combat-respawn-wave',
      ).length,
      shieldCount: battlefield.querySelectorAll(
        '.combat-effect[data-event-type="spawn_shield_expired"]',
      ).length,
      eventOwnerCount: battlefield.querySelectorAll(
        ".combat-effect[data-tooltip-owner]",
      ).length,
      childOwnerCount: battlefield.querySelectorAll(
        ".combat-lifecycle-ring [data-tooltip-owner]",
      ).length,
    };
    painter.settle(normal);
    const afterNormalSettle = {
      eventCount: normal.eventNodes.size,
      eventTypes: Array.from(
        battlefield.querySelectorAll(".combat-effect[data-event-type]"),
      ).map((node) => node.getAttribute("data-event-type")),
      settledCount: battlefield.querySelectorAll('.combat-effect[data-settled="true"]')
        .length,
    };
    painter.clear(normal);

    /** @type {Record<string, any>} */
    const endpointModes = {};
    /** @type {Array<[string, {motionMode: "normal" | "reduced" | "off", settled: boolean, persistentOnly: boolean}]>} */
    const modeCases = [
      ["normal", { motionMode: "normal", settled: false, persistentOnly: false }],
      ["reduced", { motionMode: "reduced", settled: false, persistentOnly: false }],
      ["off", { motionMode: "off", settled: false, persistentOnly: false }],
      ["settled", { motionMode: "normal", settled: true, persistentOnly: false }],
    ];
    for (const [name, options] of modeCases) {
      const installation = painter.install(plan, surface, options);
      const rings = Array.from(battlefield.querySelectorAll(ringSelector));
      endpointModes[name] = {
        ringRadii: rings.map((ring) => ringRadius(ring).toFixed(2)),
        ringOpacities: rings.map(stillOpacity),
        ringAnimations: ringAnimations(installation),
        eventTypes: Array.from(
          battlefield.querySelectorAll(".combat-effect[data-event-type]"),
        ).map((node) => node.getAttribute("data-event-type")),
      };
      painter.clear(installation);
    }
    return {
      serializedUnchanged: JSON.stringify(presentation) === serializedBefore,
      planEventIds: plan.events.map(
        (/** @type {Record<string, any>} */ event) => event.eventId,
      ),
      beforeSettle,
      afterNormalSettle,
      endpointModes,
      remainingEffectCount: battlefield.querySelectorAll(".combat-effect").length,
    };
  }, raw);

  expect(result.serializedUnchanged).toBe(true);
  expect(result.planEventIds).toEqual(raw.latest_events.ordered_event_ids);
  // Three of the five rows paint on the arena: the fall, the Ward Lift and the
  // return. A respawn wave draws nothing on the arena (the scorebug counts it
  // down).
  expect(result.beforeSettle.eventCount).toBe(3);
  expect(result.beforeSettle.nodeCount).toBeLessThanOrEqual(
    raw.latest_events.event_count * 30 + 3,
  );
  expect(result.beforeSettle.persistentNodeCount).toBeLessThanOrEqual(
    result.beforeSettle.persistentNodeBound,
  );
  // The fall keeps a thin Team B Fall Ring at 1.3 body radii (35% in a still);
  // the return's moonlight ring narrows to 0.9 radii and is gone in a still.
  expect(result.beforeSettle.lifecycleGeometry).toEqual([
    expect.objectContaining({
      eventType: "agent_died",
      lifecycle: "death",
      ring: "requiem__fall-ring",
      stroke: "rgb(239, 75, 75)",
      radiusInBodies: expect.closeTo(1.3, 2),
      stillOpacity: 0.35,
      planLayoutFields: [],
      domLayoutAttributes: [],
      semanticLeaderCount: 0,
      hitOwnsBodyCenter: false,
      agentOwnsBodyCenter: true,
      hitPointerEvents: "stroke",
      hitFill: "none",
      ariaLabel: "Fallen",
      tooltipTitle: "Fallen",
      tooltipSummary: "This champion fell this tick.",
    }),
    expect.objectContaining({
      eventType: "agent_respawned",
      lifecycle: "resurrection",
      ring: "homecoming__ring",
      stroke: "rgb(220, 230, 255)",
      radiusInBodies: expect.closeTo(0.9, 2),
      stillOpacity: 0,
      planLayoutFields: [],
      domLayoutAttributes: [],
      semanticLeaderCount: 0,
      hitOwnsBodyCenter: false,
      agentOwnsBodyCenter: true,
      hitPointerEvents: "stroke",
      hitFill: "none",
      ariaLabel: "Returned",
      tooltipTitle: "Returned",
      tooltipSummary: "This champion came back at its waystone this tick.",
    }),
  ]);
  for (const lifecycle of result.beforeSettle.lifecycleGeometry) {
    expect(Math.abs(lifecycle.center.x - lifecycle.bodyCenter.x)).toBeLessThanOrEqual(
      0.001,
    );
    expect(Math.abs(lifecycle.center.y - lifecycle.bodyCenter.y)).toBeLessThanOrEqual(
      0.001,
    );
    const copy = [
      lifecycle.ariaLabel,
      lifecycle.tooltipTitle,
      lifecycle.tooltipSummary,
    ].join(" ");
    expect(copy).not.toMatch(/_/u);
    expect(copy).not.toMatch(/semantic pulse/iu);
  }
  // In play the Fall Ring grows outward onto its still; the Homecoming ring
  // narrows from 1.6 radii and fades out, which is its still.
  const [fallRing, homecoming] = result.beforeSettle.ringAnimations;
  expect(
    result.beforeSettle.ringAnimations.map((/** @type {any} */ row) => row.id),
  ).toEqual(["fall-ring", "homecoming"]);
  const scales = (/** @type {Array<{transform: string | null}>} */ frames) =>
    frames.map((frame) =>
      Number(/scale\(([\d.]+)\)/u.exec(frame.transform ?? "")?.[1]),
    );
  expect(fallRing.keyframes.at(0)).toEqual({ opacity: 0, transform: "scale(0.77)" });
  expect(fallRing.keyframes.at(-1)).toEqual({ opacity: 0.35, transform: "scale(1)" });
  const fallScales = scales(fallRing.keyframes);
  expect(fallScales).toEqual([...fallScales].sort((left, right) => left - right));
  expect(scales(homecoming.keyframes)[0]).toBeCloseTo(1.6 / 0.9, 2);
  expect(homecoming.keyframes.at(-1)).toEqual({ opacity: 0, transform: "scale(1)" });
  expect(result.beforeSettle.waveRows).toEqual([
    { teamIndex: 0, teamId: 1, side: "left", anchor: null },
    { teamIndex: 1, teamId: 2, side: "right", anchor: null },
  ]);
  expect(result.beforeSettle.arenaWaveCount).toBe(0);
  expect(result.beforeSettle.shieldCount).toBe(1);
  expect(result.beforeSettle.eventOwnerCount).toBe(3);
  expect(result.beforeSettle.childOwnerCount).toBe(0);
  expect(result.afterNormalSettle.eventCount).toBe(2);
  expect(result.afterNormalSettle.eventTypes).toEqual([
    "agent_died",
    "agent_respawned",
  ]);
  expect(result.afterNormalSettle.settledCount).toBe(2);
  // Every motion mode ends on the same still rings; only play moves them, and
  // reduced motion only fades.
  for (const mode of ["reduced", "off", "settled"]) {
    expect(result.endpointModes[mode].ringRadii).toEqual(
      result.endpointModes.normal.ringRadii,
    );
    expect(result.endpointModes[mode].ringOpacities).toEqual([0.35, 0]);
  }
  expect(result.endpointModes.off.ringAnimations).toEqual([]);
  expect(result.endpointModes.settled.ringAnimations).toEqual([]);
  expect(result.endpointModes.reduced.ringAnimations.length).toBeGreaterThan(0);
  for (const animation of result.endpointModes.reduced.ringAnimations) {
    for (const frame of animation.keyframes) {
      expect(frame.transform, animation.id).toBeNull();
    }
  }
  expect(result.endpointModes.settled.eventTypes).toEqual([
    "agent_died",
    "agent_respawned",
  ]);
  expect(result.remainingEffectCount).toBe(0);
});

test("authorized multi-application status merges into one plan row and paints no off-body badge", async ({
  page,
}) => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const template = raw.latest_events.events.find(
    (/** @type {Record<string, any>} */ event) => event.event_kind === "status_applied",
  );
  if (!template) {
    throw new Error("Authorized status application fixture is unavailable.");
  }
  const sources = [
    raw.latest_events.agent_phase_trajectories[0].successor,
    raw.latest_events.agent_phase_trajectories[2].successor,
  ];
  const applicationIds = sources.map(
    (/** @type {Record<string, any>} */ _source, ordinal) =>
      `${raw.latest_events.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
  );
  raw.latest_events.events = sources.map(
    (/** @type {Record<string, any>} */ sourceAnchor, ordinal) => ({
      ...structuredClone(template),
      event_id: applicationIds[ordinal],
      ordinal,
      source_anchor: structuredClone(sourceAnchor),
    }),
  );
  raw.latest_events.event_count = raw.latest_events.events.length;
  raw.latest_events.ordered_event_ids = [...applicationIds];
  raw.latest_events.ordered_event_kinds = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_kind,
  );

  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const { createTooltipController } = await import(`${moduleRoot}/tooltip.js`);
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    const tooltip = document.querySelector("#visual-tooltip");
    const title = document.querySelector("#visual-tooltip-title");
    const details = document.querySelector("#visual-tooltip-details");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement) ||
      !(tooltip instanceof HTMLElement) ||
      !(title instanceof HTMLElement) ||
      !(details instanceof HTMLElement)
    ) {
      throw new Error("Authorized status component surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, {
      showRanges: true,
      visualFilterState: allVisualFilters,
    });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface, allVisualFilters);
    if (surface === null || plan === null) {
      throw new Error("Authorized status choreography plan is unavailable.");
    }
    new SvgChoreographyPainter().install(plan, surface, {
      motionMode: "off",
      settled: false,
      persistentOnly: false,
    });
    const controller = createTooltipController({
      root: document.body,
      tooltip,
      title,
      details,
    });
    const planned = plan.events[0];
    // No badge is painted off the body: the recipient's status card shows the
    // status, and it names its source by class only.
    const recipientKey = planned?.recipientPresentationKey;
    const card =
      typeof recipientKey === "string"
        ? battlefield.querySelector(
            `.status-cell[data-presentation-key="${CSS.escape(recipientKey)}"][data-token-id="${CSS.escape(String(planned.tokenId))}"]`,
          )
        : null;
    if (!(card instanceof SVGElement)) {
      throw new Error("Recipient status card is unavailable.");
    }
    card.dispatchEvent(new FocusEvent("focusin", { bubbles: true }));
    const response = {
      planEventCount: plan.events.length,
      plannedKind: planned?.kind ?? null,
      plannedLifecycle: planned?.lifecycle ?? null,
      effectCount: battlefield.querySelectorAll(".combat-effect--status-lifecycle")
        .length,
      lifecycleCount: battlefield.querySelectorAll(".combat-lifecycle").length,
      routeChildCount: battlefield.querySelectorAll(".combat-choreography-routes > *")
        .length,
      statusUnderlayCount: battlefield.querySelectorAll(
        ".combat-route-effect--status-lifecycle",
      ).length,
      plannedAtomicIds: planned?.atomicEventIds ?? null,
      plannedApplicationIds: planned?.applicationEventIds ?? null,
      sourcePublicAgentIds:
        planned?.applicationSources?.map(
          (/** @type {Record<string, any>} */ source) => source.sourcePublicAgentId,
        ) ?? null,
      sourceRecordKeys:
        planned?.applicationSources?.map((/** @type {Record<string, any>} */ source) =>
          Object.keys(source).sort(),
        ) ?? null,
      sourceIdentities:
        planned?.applicationSources?.map(
          (/** @type {Record<string, any>} */ source) => ({
            classId: source.sourceIdentity?.class_id,
            displayId: source.sourceIdentity?.display_agent_id,
            teamId: source.sourceIdentity?.team_id,
          }),
        ) ?? null,
      cardTitle: title.textContent,
      cardText: details.textContent ?? "",
      cardLabels: Array.from(
        details.querySelectorAll(".semantic-explanation__label"),
        (node) => node.textContent,
      ),
    };
    controller.destroy();
    return response;
  }, raw);

  expect(result.planEventCount).toBe(1);
  expect(result.plannedKind).toBe("status_lifecycle");
  expect(result.plannedLifecycle).toBe("applied");
  expect(result.effectCount).toBe(0);
  expect(result.lifecycleCount).toBe(0);
  expect(result.routeChildCount).toBe(0);
  expect(result.statusUnderlayCount).toBe(0);
  expect(result.plannedAtomicIds).toEqual(applicationIds);
  expect(result.plannedApplicationIds).toEqual(applicationIds);
  expect(result.sourcePublicAgentIds).toEqual(["agent-slot-0", "agent-slot-2"]);
  expect(result.sourceRecordKeys).toEqual([
    ["eventId", "sourceIdentity", "sourcePresentationKey", "sourcePublicAgentId"],
    ["eventId", "sourceIdentity", "sourcePresentationKey", "sourcePublicAgentId"],
  ]);
  // The merged row keeps both sources' exact identities: a Mage (ID 0) and a
  // Priest (ID 2), both of Team A.
  expect(result.sourceIdentities).toEqual([
    { classId: 1, displayId: "0", teamId: 1 },
    { classId: 5, displayId: "2", teamId: 1 },
  ]);
  expect(result.cardTitle).toBe("Blessing of Freedom (Slow Resistance)");
  expect(result.cardText).toContain("From a Priest.");
  expect(result.cardLabels).toEqual(["Time Left"]);
  expect(result.cardText).not.toContain("Mage · ID 0");
  expect(result.cardText).not.toContain("Priest · ID 2");
});

test("a champion's body wins real SVG hit arbitration over an overlapping accepted route", async ({
  page,
}) => {
  await page.goto(origin);
  const overlap = await page.evaluate(async (raw) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const { createRouteGeometry } = await import(`${moduleRoot}/routes.js`);
    const {
      createChampionCardDescriptor,
      createSemanticDescriptor,
      createTooltipController,
      registerTooltipOwner,
    } = await import(`${moduleRoot}/tooltip.js`);
    const { championCardFacts } = await import(`${moduleRoot}/shell.js`);
    const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    const tooltip = document.querySelector("#visual-tooltip");
    const title = document.querySelector("#visual-tooltip-title");
    const details = document.querySelector("#visual-tooltip-details");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement) ||
      !(tooltip instanceof HTMLElement) ||
      !(title instanceof HTMLElement) ||
      !(details instanceof HTMLElement)
    ) {
      throw new Error("Renderer tooltip test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, {
      showRanges: true,
      visualFilterState: allVisualFilters,
    });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface, allVisualFilters);
    if (surface === null || plan === null) {
      throw new Error("Authorized choreography surface or plan is unavailable.");
    }
    new SvgChoreographyPainter().install(plan, surface, {
      motionMode: "off",
      settled: false,
      persistentOnly: false,
    });

    // A Basic stroke's hit path is the accepted route's pointer surface.
    const routeHit = battlefield.querySelector(
      ".combat-route-effect--activation .basic-blow__hit",
    );
    const routeOwner = routeHit?.closest(".combat-route-effect");
    const targetKey = routeOwner?.getAttribute("data-target-presentation-key");
    const agent =
      typeof targetKey === "string" && targetKey
        ? battlefield.querySelector(
            `.agent[data-presentation-key="${CSS.escape(targetKey)}"]`,
          )
        : battlefield.querySelector(".agent");
    const body = agent?.querySelector(".agent-body");
    const hitMatrix =
      routeHit instanceof SVGGraphicsElement ? routeHit.getScreenCTM() : null;
    const bodyMatrix = body instanceof SVGGraphicsElement ? body.getScreenCTM() : null;
    if (
      !(routeHit instanceof SVGPathElement) ||
      !(routeOwner instanceof SVGElement) ||
      !(agent instanceof SVGElement) ||
      !(body instanceof SVGGraphicsElement) ||
      hitMatrix === null ||
      bodyMatrix === null
    ) {
      throw new Error("Authorized route and agent geometry are unavailable.");
    }
    const bodyBounds = body.getBBox();
    const screenCenter = new DOMPoint(
      bodyBounds.x + bodyBounds.width / 2,
      bodyBounds.y + bodyBounds.height / 2,
    ).matrixTransform(bodyMatrix);
    // The stroke group is moved and turned, so build the route in the hit
    // path's own frame: 80 px either side of the body center on screen.
    const toHitFrame = (/** @type {number} */ dx) => {
      const local = new DOMPoint(screenCenter.x + dx, screenCenter.y).matrixTransform(
        hitMatrix.inverse(),
      );
      return { x: local.x, y: local.y };
    };
    const geometry = createRouteGeometry(
      {
        eventId: "component-route-agent-overlap",
        source: toHitFrame(-80),
        target: toHitFrame(80),
        sourceRadius: 0,
        targetRadius: 0,
        offset: 0,
      },
      { endpointGap: 0 },
    );
    routeHit.setAttribute("d", geometry.path);
    routeOwner.setAttribute("opacity", "1");
    registerTooltipOwner(
      routeOwner,
      createSemanticDescriptor({
        kind: "accepted-route",
        id: "component-route-agent-overlap",
        title: "Accepted Route",
        tone: "information",
        accent: "none",
        summary: "Accepted route component overlap probe.",
        rows: [],
        sections: [],
        metadata: { compact: true, full: false },
        anchor: "pointer",
      }),
    );
    // The Replay Viewer page gives each body its Champion Card, as main.js
    // installAuthorizedAgentActivation does; every body here also stands on
    // its Spawn Pad, whose Symbol Tip lies under the body.
    const facts = championCardFacts({
      presentation,
      combatConfiguration: null,
      subject: {
        presentationKey: agent.getAttribute("data-presentation-key"),
        slot: null,
        publicAgentId: null,
        classId: null,
      },
    });
    if (facts === null) {
      throw new Error("The body's Champion Card facts are unavailable.");
    }
    registerTooltipOwner(agent, createChampionCardDescriptor(facts, "champion:body"), {
      inspectable: false,
    });
    const controller = createTooltipController({
      root: document.body,
      tooltip,
      title,
      details,
    });
    // Retain the controller for the pointer assertion and clean it up on navigation.
    window.addEventListener("pagehide", () => controller.destroy(), { once: true });

    const owners = [
      ...new Set(
        document
          .elementsFromPoint(screenCenter.x, screenCenter.y)
          .map((element) => element.closest("[data-tooltip-owner]"))
          .filter((element) => element instanceof Element),
      ),
    ];
    const routeRoot = routeOwner.parentElement;
    const routeLayer = routeRoot?.parentElement;
    const camera = battlefield.querySelector(".battlefield-camera");
    return {
      agentPaintOrder: owners.indexOf(agent),
      routePaintOrder: owners.indexOf(routeOwner),
      padPaintOrder: owners.findIndex((owner) => owner.closest(".waystone") !== null),
      activationRoute: routeOwner.classList.contains("combat-route-effect--activation"),
      routeRootOwned:
        routeRoot?.classList.contains("combat-choreography-routes") === true,
      routeLayer: routeLayer?.getAttribute("data-layer") ?? null,
      layerOrder: Array.from(camera?.children ?? [], (layer) =>
        layer.getAttribute("data-layer"),
      ),
      x: screenCenter.x,
      y: screenCenter.y,
    };
  }, fixture.presentations.replay_oracle);

  expect(overlap.agentPaintOrder).toBeGreaterThanOrEqual(0);
  expect(overlap.routePaintOrder).toBeGreaterThanOrEqual(0);
  expect(overlap.agentPaintOrder).toBeLessThan(overlap.routePaintOrder);
  expect(overlap.agentPaintOrder).toBeLessThan(overlap.padPaintOrder);
  expect(overlap.activationRoute).toBe(true);
  expect(overlap.routeRootOwned).toBe(true);
  expect(overlap.routeLayer).toBe("strike-band");
  const routeLayerIndex = overlap.layerOrder.indexOf("strike-band");
  expect(routeLayerIndex).toBeGreaterThanOrEqual(0);
  // Routes lie above the stones and under the bodies and everything on them.
  expect(overlap.layerOrder.indexOf("obstacle")).toBeGreaterThanOrEqual(0);
  expect(overlap.layerOrder.indexOf("obstacle")).toBeLessThan(routeLayerIndex);
  for (const foregroundLayer of [
    "body",
    "selection-legality",
    "transient-events",
    "durable-status-modifier",
    "accessible-labels",
  ]) {
    expect(overlap.layerOrder.indexOf(foregroundLayer)).toBeGreaterThan(
      routeLayerIndex,
    );
  }
  await page.mouse.move(overlap.x, overlap.y);
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "champion",
  );
});

// The old Ultimate cooldown docks (and their far-away fallback badge) are
// retired: the Rune Ring on the champion's own token shows the cooldown.
test("the Ultimate cooldown stays on the champion's Rune Ring at every arena size", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (raw) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { authorizedPresentationSceneView } = await import(
      `${moduleRoot}/authorized-presentation-adapter.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const allVisualFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
    const { createTooltipController } = await import(`${moduleRoot}/tooltip.js`);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    const tooltip = document.querySelector("#visual-tooltip");
    const title = document.querySelector("#visual-tooltip-title");
    const details = document.querySelector("#visual-tooltip-details");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement) ||
      !(tooltip instanceof HTMLElement) ||
      !(title instanceof HTMLElement) ||
      !(details instanceof HTMLElement)
    ) {
      throw new Error("Cooldown test surface is unavailable.");
    }
    const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
    const scene = authorizedPresentationSceneView(presentation);
    const selectedKey = scene?.selection?.selected_presentation_key;
    if (typeof selectedKey !== "string") {
      throw new Error("The selected champion is unavailable.");
    }
    const cooldowns = Object.fromEntries(
      scene.agents.map((/** @type {Record<string, any>} */ agent) => [
        agent.presentation_key,
        agent.ultimate_cooldown_remaining,
      ]),
    );
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    const controller = createTooltipController({
      root: document.body,
      tooltip,
      title,
      details,
    });
    /** @param {number} width @param {number} height */
    const renderAt = (width, height) => {
      battlefield.style.minWidth = "0";
      battlefield.style.minHeight = "0";
      battlefield.style.width = `${width}px`;
      battlefield.style.height = `${height}px`;
      renderer.render(presentation, {
        showRanges: true,
        visualFilterState: allVisualFilters,
      });
      return {
        cooldownDocks: battlefield.querySelectorAll(
          '.cooldown-cell, .required-dock-fallback[data-kind="cooldown"]',
        ).length,
        rings: Array.from(battlefield.querySelectorAll(".agent"), (agent) => {
          const key = agent.getAttribute("data-presentation-key") ?? "";
          const body = agent.querySelector(".agent-body");
          const marks = battlefield.querySelector(
            `.status-marks[data-marks-for="${CSS.escape(key)}"]`,
          );
          const ready = marks?.querySelector('[data-rune-piece="ring.ready"]');
          return {
            key,
            cooldown: cooldowns[key],
            bodyRadius: Number(body?.getAttribute("r")),
            bodyCenter: [
              Number(body?.getAttribute("cx")),
              Number(body?.getAttribute("cy")),
            ],
            marksAt: (
              /^translate\(([-\d.]+) ([-\d.]+)\)$/u.exec(
                marks?.getAttribute("transform") ?? "",
              ) ?? []
            )
              .slice(1)
              .map(Number),
            marksRadius: Number(marks?.getAttribute("data-radius")),
            marksHidden: marks?.getAttribute("aria-hidden") ?? null,
            pieces: Array.from(
              marks?.querySelectorAll(':scope > [data-rune-piece^="ring."]') ?? [],
              (piece) => piece.getAttribute("data-rune-piece"),
            ),
            readyRadius: ready ? Number(ready.getAttribute("r")) : null,
          };
        }),
      };
    };
    const ordinary = renderAt(720, 480);
    const remote = renderAt(50, 70);
    const selected = battlefield.querySelector(
      `.agent[data-presentation-key="${CSS.escape(selectedKey)}"]`,
    );
    if (!(selected instanceof SVGElement)) {
      throw new Error("The selected champion's token is unavailable.");
    }
    selected.dispatchEvent(new FocusEvent("focusin", { bubbles: true }));
    const remoteTip = {
      hidden: tooltip.hidden,
      title: title.textContent,
      details: details.textContent,
    };
    const restored = renderAt(720, 480);
    controller.destroy();
    return { selectedKey, ordinary, remote, remoteTip, restored };
  }, fixture.state_cases.replay_oracle_final_selected);

  /** @param {number} ticks */
  const expectedRing = (ticks) =>
    ticks <= 0
      ? ["ring.ready", "ring.gem.0", "ring.gem.1", "ring.gem.2", "ring.gem.3"]
      : [
          "ring.track",
          ...Array.from(
            { length: Math.min(5, Math.max(0, Math.floor((30 - ticks) / 5))) },
            (_, index) => `ring.arc.${index}`,
          ),
        ];
  for (const { size, state } of [
    { size: "720x480", state: result.ordinary },
    { size: "50x70", state: result.remote },
  ]) {
    expect(state.cooldownDocks, size).toBe(0);
    expect(state.rings).toHaveLength(5);
    for (const ring of state.rings) {
      // The ring rides the body at its own size and is hidden paint; the
      // champion's card says the cooldown in words.
      expect(ring.marksAt, `${size} ${ring.key}`).toHaveLength(2);
      expect(ring.marksAt[0], `${size} ${ring.key}`).toBeCloseTo(ring.bodyCenter[0], 1);
      expect(ring.marksAt[1], `${size} ${ring.key}`).toBeCloseTo(ring.bodyCenter[1], 1);
      expect(ring.marksRadius, `${size} ${ring.key}`).toBeCloseTo(ring.bodyRadius, 2);
      expect(ring.marksHidden).toBe("true");
      expect(ring.pieces, `${size} ${ring.key}`).toEqual(expectedRing(ring.cooldown));
      if (ring.readyRadius !== null) {
        expect(ring.readyRadius).toBeCloseTo(0.68 * ring.marksRadius, 1);
      }
    }
  }
  const selectedOrdinary = result.ordinary.rings.find(
    (/** @type {Record<string, any>} */ ring) => ring.key === result.selectedKey,
  );
  const selectedRemote = result.remote.rings.find(
    (/** @type {Record<string, any>} */ ring) => ring.key === result.selectedKey,
  );
  // The selected Mage's Burst has 29 ticks left: an unlit track at any size.
  expect(selectedOrdinary?.cooldown).toBe(29);
  expect(selectedOrdinary?.pieces).toEqual(["ring.track"]);
  expect(selectedRemote?.pieces).toEqual(["ring.track"]);
  expect(selectedRemote?.bodyRadius).toBeLessThan(selectedOrdinary?.bodyRadius ?? 0);
  expect(result.remoteTip.hidden).toBe(false);
  expect(result.remoteTip.title).toBe("Mage · ID 0");
  expect(result.remoteTip.details).toContain("Ultimate Status");
  expect(result.remoteTip.details).toContain("On Cooldown (29 ticks)");
  expect(result.restored).toEqual(result.ordinary);
});

for (const caseName of [
  "replay_no_shared_corpse_overlay",
  "live_no_shared_editable_corpse_overlay",
]) {
  test(`living bodies win overlapping corpse paint and mouse targeting: ${caseName}`, async ({
    page,
  }) => {
    await page.goto(origin);
    const result = await page.evaluate(async (raw) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement))
        throw new Error("Missing renderer surface");
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      renderer.render(presentation, { showRanges: false });
      const dead = battlefield.querySelector('.agent[data-alive="false"]');
      const living = battlefield.querySelector('.agent[data-alive="true"]');
      if (
        !(dead instanceof SVGGraphicsElement) ||
        !(living instanceof SVGGraphicsElement)
      )
        throw new Error("Fixture needs a corpse and a living body");
      const corpseBody = dead.querySelector(".agent-body");
      const livingBody = living.querySelector(".agent-body");
      if (
        !(corpseBody instanceof SVGCircleElement) ||
        !(livingBody instanceof SVGCircleElement)
      )
        throw new Error("Missing body circles");
      // Align only the rendered test geometry. The authorized records stay intact.
      dead.setAttribute(
        "transform",
        `translate(${livingBody.cx.baseVal.value - corpseBody.cx.baseVal.value} ${livingBody.cy.baseVal.value - corpseBody.cy.baseVal.value})`,
      );
      const matrix = livingBody.getScreenCTM();
      if (matrix === null) throw new Error("Missing screen transform");
      const center = new DOMPoint(
        livingBody.cx.baseVal.value,
        livingBody.cy.baseVal.value,
      ).matrixTransform(matrix);
      const top = document.elementFromPoint(center.x, center.y)?.closest(".agent");
      // Each body sits in its glide wrapper inside the dead or living group.
      const deadGroup = dead.closest("[data-body-state]");
      const livingGroup = living.closest("[data-body-state]");
      const ordered = Boolean(
        deadGroup &&
          livingGroup &&
          deadGroup.compareDocumentPosition(livingGroup) &
            Node.DOCUMENT_POSITION_FOLLOWING,
      );
      dead.removeAttribute("transform");
      renderer.render(presentation, { showRanges: false });
      const retained = dead.isConnected && living.isConnected;
      renderer.render(null);
      renderer.render(presentation, { showRanges: false });
      return {
        ordered,
        topAlive: top === living,
        retained,
        afterClear:
          battlefield.querySelector(
            '[data-body-state="living"] .agent[data-alive="true"]',
          ) !== null,
        corpseGroup: deadGroup?.getAttribute("data-body-state"),
        livingGroup: livingGroup?.getAttribute("data-body-state"),
      };
    }, fixture.state_cases[caseName]);
    expect(result).toEqual({
      ordered: true,
      topAlive: true,
      retained: true,
      afterClear: true,
      corpseGroup: "dead",
      livingGroup: "living",
    });
  });
}

// Team waves and the death card are page chrome now (the scorebug's wave
// countdown and the Fallen panel), so the arena draws neither; their plan rows keep each
// team's recorded side. The Replay Viewer strip keeps Team A in its left box
// whatever its spawn side, and records each team's side on its box.
test("team displays follow recorded spawn sides and keep team colors", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (raw) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { renderMatchSummary } = await import(`${moduleRoot}/match-summary.js`);
    raw.match_summary.teams[0].display_side = "right";
    raw.match_summary.teams[1].display_side = "left";
    raw.match_summary.deaths = [
      {
        public_agent_id: "agent-slot-5",
        team_id: 2,
        class_id: 3,
        killing_team_id: 1,
        contributors: [{ public_agent_id: "agent-slot-2", team_id: 1, class_id: 5 }],
      },
    ];
    const batch = raw.latest_events;
    batch.events = [1, 2].map((teamId, ordinal) => ({
      event_kind: "respawn_wave_occurred",
      phase_rank: 120,
      ordinal,
      event_id: `${batch.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
      team_anchor: { phase: "successor", team_index: teamId - 1, team_id: teamId },
    }));
    batch.event_count = 2;
    batch.ordered_event_ids = batch.events.map(
      (/** @type {Record<string, any>} */ event) => event.event_id,
    );
    batch.ordered_event_kinds = ["respawn_wave_occurred", "respawn_wave_occurred"];
    const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters } = await import(
      `${moduleRoot}/visual-filters.js`
    );
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement))
      throw new Error("Missing renderer surface");
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, { showRanges: false });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(
      presentation,
      surface,
      enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE),
    );
    if (surface === null || plan === null) throw new Error("Missing display plan");
    new SvgChoreographyPainter().install(plan, surface, {
      motionMode: "off",
      settled: false,
      persistentOnly: false,
    });
    const planEvents = /** @type {Record<string, any>[]} */ (plan.events);
    const deathCard = planEvents.find(
      (event) => event.cueSemantic === "death_announcement",
    );

    const root = document.createElement("div");
    const task = document.createElement("span");
    const teams = [document.createElement("div"), document.createElement("div")];
    const taskSelect = document.createElement("select");
    root.append(task, ...teams, taskSelect);
    document.body.append(root);
    renderMatchSummary({ root, task, teams, taskSelect }, presentation);
    return {
      waves: planEvents
        .filter((event) => event.eventType === "respawn_wave_occurred")
        .map((event) => ({ team: event.teamId, side: event.teamSide })),
      deathCard: deathCard
        ? { team: deathCard.teamId, side: deathCard.teamSide, label: deathCard.label }
        : null,
      arenaWaveCount: battlefield.querySelectorAll(
        '.combat-respawn-wave, .combat-effect[data-event-type="respawn_wave_occurred"]',
      ).length,
      arenaDeathCardCount: battlefield.querySelectorAll(
        '.combat-death-announcement, [data-cue-semantic="death_announcement"]',
      ).length,
      teams: teams.map((node) => ({
        team: node.dataset.team,
        side: node.dataset.side,
        label: node.querySelector("strong")?.textContent,
        blue: node.classList.contains("match-scoreboard__team--a"),
        red: node.classList.contains("match-scoreboard__team--b"),
      })),
    };
  }, fixture.presentations.replay_oracle);
  expect(result.waves).toEqual([
    { team: 1, side: "right" },
    { team: 2, side: "left" },
  ]);
  // Team A (on the right) killed a Team B champion.
  expect(result.deathCard).toEqual({ team: 1, side: "right", label: "Team A Kills" });
  expect(result.arenaWaveCount).toBe(0);
  expect(result.arenaDeathCardCount).toBe(0);
  expect(result.teams).toEqual([
    { team: "1", side: "right", label: "Team A", blue: true, red: false },
    { team: "2", side: "left", label: "Team B", blue: false, red: true },
  ]);
});
