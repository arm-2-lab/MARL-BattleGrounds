/**
 * @file Check where live play puts the arena's view tag and the Replay
 * Recording popover (owner ruling M3, 3 Oct 2026).
 *
 * Contracts checked:
 * - viewTagPlace (src/looks/camera.js): on the Whole Map it gives where the
 *   left home strip ends (the left team's Red Zone, whichever team that is;
 *   the floor's left edge without a Red Zone) and the frame's height under
 *   the floor, both as shares of the field's width, so they hold at every
 *   window size; a zoomed camera gives null.
 * - createCameraSwitch, live play: each update writes those shares on #arena
 *   as --left-home-end and --floor-band; a zoomed tick keeps the last Whole
 *   Map spot.
 * - battleclient/overlays.css: the view tag sits on the frame under the
 *   floor, 8 px past the left home strip, its font filling that frame but
 *   never below 10 px (10 to 11 px); the Recording popover opens over the
 *   rail's cards below the scorebug (no longer at 52 px under the top bar).
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { createViewportTransform } from "../src/layout.js";
import { createCameraSwitch, viewTagPlace } from "../src/looks/camera.js";

/** @param {number} [width] @param {any} [cameraFrame] */
function liveTransform(width = 1032, cameraFrame = null) {
  return createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: width,
    viewportHeight: (width * 532) / 1032,
    padding: (width * 16) / 1032,
    cameraFrame,
  });
}

const RED_ZONE = { team_a_x_range: [0, 5], team_b_x_range: [15, 20] };

test("the tag spot is the left strip's end and the frame under the floor", () => {
  for (const width of [652, 908, 1257]) {
    const place = viewTagPlace({
      transform: liveTransform(width),
      scene: { map: { red_zone: RED_ZONE } },
    });
    assert.ok(place);
    // 16 px pad plus a quarter of the 1000 px floor, out of 1032.
    assert.ok(Math.abs(place.homeEnd - (16 + 250) / 1032) < 1e-9, String(width));
    assert.ok(Math.abs(place.band - 16 / 1032) < 1e-9, String(width));
  }
  // Team B's home on the left: its strip decides.
  const swapped = viewTagPlace({
    transform: liveTransform(),
    scene: { map: { red_zone: { team_a_x_range: [15, 20], team_b_x_range: [0, 4] } } },
  });
  assert.ok(Math.abs((swapped?.homeEnd ?? 0) - (16 + 200) / 1032) < 1e-9);
  // No Red Zone: the floor's left edge.
  const bare = viewTagPlace({ transform: liveTransform(), scene: { map: {} } });
  assert.ok(Math.abs((bare?.homeEnd ?? 0) - 16 / 1032) < 1e-9);
  // A zoomed camera moves the strip: no spot.
  assert.equal(
    viewTagPlace({
      transform: liveTransform(1032, { minX: 4, minY: 2, maxX: 14, maxY: 7 }),
      scene: { map: { red_zone: RED_ZONE } },
    }),
    null,
  );
  assert.equal(viewTagPlace({ transform: null, scene: null }), null);
});

function livePage() {
  /** @type {Map<string, string>} */
  const arenaStyle = new Map();
  const arena = {
    style: {
      setProperty: (/** @type {string} */ name, /** @type {string} */ value) =>
        arenaStyle.set(name, value),
    },
  };
  /** @type {any} */
  const doc = {
    defaultView: { addEventListener() {}, removeEventListener() {} },
    getElementById: (/** @type {string} */ id) => (id === "arena" ? arena : null),
    createElement: () => node(),
    createElementNS: () => node(),
    body: null,
  };
  function node() {
    /** @type {any} */
    const element = {
      ownerDocument: doc,
      children: /** @type {any[]} */ ([]),
      dataset: {},
      style: {},
      hidden: false,
      textContent: "",
      classList: { add() {}, toggle() {} },
      setAttribute() {},
      getAttribute: () => null,
      removeAttribute() {},
      addEventListener() {},
      closest: () => null,
      remove() {},
      append: (/** @type {any[]} */ ...nodes) => element.children.push(...nodes),
      replaceChildren: (/** @type {any[]} */ ...nodes) => {
        element.children = [...nodes];
      },
    };
    return element;
  }
  return { slot: node(), arenaStyle };
}

test("the live switch writes the tag spot on the arena box", () => {
  const { slot, arenaStyle } = livePage();
  const camera = createCameraSwitch({
    slot,
    product: "combat_debugger",
    storage: null,
    timers: { setTimeout: () => 0, clearTimeout: () => {} },
    applyFrame: () => {},
    registerKeys: false,
  });
  const tick = (/** @type {any} */ transform) =>
    camera.update({
      camera: { audience: "agent_pov", bodies: [], fights: [] },
      beats: null,
      scene: { map: { red_zone: RED_ZONE } },
      worldWidth: 20,
      worldHeight: 10,
      transform,
      reason: "play",
    });
  try {
    tick(liveTransform(908));
    assert.equal(arenaStyle.get("--left-home-end"), "0.25775");
    assert.equal(arenaStyle.get("--floor-band"), "0.0155");
    arenaStyle.clear();
    tick(liveTransform(908, { minX: 4, minY: 2, maxX: 14, maxY: 7 }));
    assert.equal(arenaStyle.size, 0, "a zoomed tick keeps the last spot");
  } finally {
    camera.destroy();
  }
});

test("overlays.css puts the tag under the floor and the popover over the rail", () => {
  // Whitespace-insensitive: the formatter may break a long selector over lines.
  const overlays = readFileSync(
    new URL("../src/styles/battleclient/overlays.css", import.meta.url),
    "utf8",
  ).replace(/\s+/gu, " ");
  const tag = overlays.slice(
    overlays.indexOf(".bc .povtag { --tag-field-w"),
    overlays.indexOf("}", overlays.indexOf(".bc .povtag { --tag-field-w")),
  );
  assert.match(tag, /--tag-home-end: var\(--left-home-end, 0\.0155\);/u);
  assert.match(tag, /--tag-band: calc\(var\(--floor-band, 0\.0155\) \*/u);
  assert.match(tag, /top: auto;/u);
  assert.match(tag, /var\(--tag-home-end\) \* var\(--tag-field-w\) \+ 8px\);/u);
  assert.match(
    tag,
    /font-size: clamp\(10px, \(var\(--tag-band\) - 2px\) \/ 1\.1, 11px\);/u,
  );
  assert.match(
    overlays,
    /\.bc \.pop \{ position: fixed; inset: 144px 12px auto auto;/u,
  );
  assert.match(
    overlays,
    /width: calc\(var\(--rail-w\) - 22px\); max-height: calc\(100vh - 156px\);/u,
  );
  assert.doesNotMatch(overlays, /inset: 52px 130px auto auto;/u);
});
