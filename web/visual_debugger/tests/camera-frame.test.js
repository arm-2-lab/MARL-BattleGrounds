/**
 * @file Check the camera frame math in layout.js. Whole Map is the default: a
 * transform with no frame (or null) is exactly today's fitted map, and equals an
 * explicit Whole Map frame. A smaller frame fits inside the same padded
 * viewport, zooms world positions and lengths only (the viewport and padding,
 * which set every fixed pixel size, stay the same), clips visibleMapBounds to
 * the screen and round-trips points. Bad frames throw. layout.js keeps only the
 * Everyone in View margin (1.5 units) and zoom steps (0.25 up to 2x); the rule
 * itself is owned by looks/camera.js and checked in camera.test.js.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  createViewportTransform,
  EVERYONE_IN_VIEW_MARGIN,
  EVERYONE_IN_VIEW_MAX_ZOOM,
  EVERYONE_IN_VIEW_ZOOM_STEP,
  wholeMapCameraFrame,
} from "../src/layout.js";

const TOLERANCE = 1e-9;

/** @param {number} actual @param {number} expected @param {string} [label] */
function assertClose(actual, expected, label = "") {
  assert.ok(
    Math.abs(actual - expected) <= TOLERANCE * Math.max(1, Math.abs(expected)),
    `${label} expected ${expected}, got ${actual}`,
  );
}

test("Whole Map is the default camera and matches the fitted map exactly", () => {
  for (const [worldWidth, worldHeight, viewportWidth, viewportHeight] of [
    [20, 12, 1440, 600],
    [17.3, 12, 960, 600],
    [12, 12, 640, 900],
    [30, 8, 800, 480],
  ]) {
    const input = {
      worldWidth,
      worldHeight,
      viewportWidth,
      viewportHeight,
      padding: 24,
    };
    const plain = createViewportTransform(input);
    const withNull = createViewportTransform({ ...input, cameraFrame: null });
    const explicit = createViewportTransform({
      ...input,
      cameraFrame: wholeMapCameraFrame(worldWidth, worldHeight),
    });
    const availableWidth = viewportWidth - 48;
    const availableHeight = viewportHeight - 48;
    const scale = Math.min(availableWidth / worldWidth, availableHeight / worldHeight);
    const left = 24 + (availableWidth - worldWidth * scale) / 2;
    const top = 24 + (availableHeight - worldHeight * scale) / 2;
    for (const transform of [plain, withNull, explicit]) {
      assert.equal(transform.scale, scale);
      assert.deepEqual(transform.cameraFrame, {
        minX: 0,
        minY: 0,
        maxX: worldWidth,
        maxY: worldHeight,
      });
      const right = left + worldWidth * scale;
      const bottom = top + worldHeight * scale;
      assert.deepEqual(transform.mapBounds, {
        left,
        top,
        right,
        bottom,
        width: right - left,
        height: bottom - top,
      });
      assert.equal(transform.visibleMapBounds, transform.mapBounds);
      assert.deepEqual(transform.viewportBounds, plain.viewportBounds);
      /** @type {Array<readonly [number, number]>} */
      const points = [
        [0, 0],
        [worldWidth, worldHeight],
        [worldWidth / 3, worldHeight / 7],
      ];
      for (const point of points) {
        assert.deepEqual(transform.worldToScreen(point), plain.worldToScreen(point));
      }
      assert.equal(transform.worldLengthToScreen(0.5), plain.worldLengthToScreen(0.5));
    }
    assert.ok(Object.isFrozen(plain.cameraFrame));
  }
});

test("a camera frame zooms world positions and lengths but keeps the viewport and padding", () => {
  const wholeMap = createViewportTransform({
    worldWidth: 20,
    worldHeight: 12,
    viewportWidth: 1040,
    viewportHeight: 648,
    padding: 24,
  });
  const frame = { minX: 5, minY: 3, maxX: 15, maxY: 9 };
  const zoomed = createViewportTransform({
    worldWidth: 20,
    worldHeight: 12,
    viewportWidth: 1040,
    viewportHeight: 648,
    padding: 24,
    cameraFrame: frame,
  });
  assert.equal(zoomed.scale, wholeMap.scale * 2);
  assert.deepEqual(zoomed.cameraFrame, frame);
  assert.notEqual(zoomed.cameraFrame, frame);
  assert.ok(Object.isFrozen(zoomed.cameraFrame));
  assert.deepEqual(zoomed.viewportBounds, wholeMap.viewportBounds);
  // The frame's corners land on the padded, centred fit (the same box the
  // Whole Map filled, because the frame has the map's aspect ratio).
  const topLeft = zoomed.worldToScreen([frame.minX, frame.maxY]);
  const bottomRight = zoomed.worldToScreen([frame.maxX, frame.minY]);
  assertClose(topLeft.x, wholeMap.mapBounds.left, "frame left");
  assertClose(topLeft.y, wholeMap.mapBounds.top, "frame top");
  assertClose(bottomRight.x, wholeMap.mapBounds.right, "frame right");
  assertClose(bottomRight.y, wholeMap.mapBounds.bottom, "frame bottom");
  // The whole map now reaches past the screen; only its visible part is kept.
  assert.ok(zoomed.mapBounds.left < 0 && zoomed.mapBounds.right > 1040);
  assert.deepEqual(zoomed.visibleMapBounds, {
    left: 0,
    top: 0,
    right: 1040,
    bottom: 648,
    width: 1040,
    height: 648,
  });
  assert.equal(zoomed.worldLengthToScreen(1), wholeMap.worldLengthToScreen(1) * 2);
  for (const world of [
    { x: 5, y: 3 },
    { x: 10.25, y: 6.5 },
    { x: -1, y: 13 },
  ]) {
    const back = zoomed.screenToWorld(zoomed.worldToScreen(world));
    assertClose(back.x, world.x, "round trip x");
    assertClose(back.y, world.y, "round trip y");
  }
});

test("a frame of another aspect ratio is centred inside the padded viewport", () => {
  const transform = createViewportTransform({
    worldWidth: 20,
    worldHeight: 12,
    viewportWidth: 1000,
    viewportHeight: 600,
    padding: { top: 10, right: 20, bottom: 30, left: 40 },
    cameraFrame: { minX: 2, minY: 2, maxX: 6, maxY: 10 },
  });
  // Available 940 x 560; the 4 x 8 frame is height-limited: 70 px per unit.
  assert.equal(transform.scale, 70);
  const topLeft = transform.worldToScreen([2, 10]);
  const bottomRight = transform.worldToScreen([6, 2]);
  assertClose(topLeft.y, 10, "top");
  assertClose(bottomRight.y, 570, "bottom");
  assertClose((topLeft.x + bottomRight.x) / 2, 40 + 940 / 2, "centre x");
  assertClose(bottomRight.x - topLeft.x, 280, "frame width");
});

test("bad camera frames are refused", () => {
  const base = {
    worldWidth: 20,
    worldHeight: 12,
    viewportWidth: 800,
    viewportHeight: 600,
  };
  assert.throws(
    () =>
      createViewportTransform({
        ...base,
        cameraFrame: /** @type {any} */ ("whole map"),
      }),
    TypeError,
  );
  assert.throws(
    () =>
      createViewportTransform({
        ...base,
        cameraFrame: { minX: 0, minY: 0, maxX: Number.NaN, maxY: 12 },
      }),
    /cameraFrame\.maxX must be finite/,
  );
  assert.throws(
    () =>
      createViewportTransform({
        ...base,
        cameraFrame: { minX: 5, minY: 0, maxX: 5, maxY: 12 },
      }),
    RangeError,
  );
  assert.throws(
    () =>
      createViewportTransform({
        ...base,
        cameraFrame: { minX: 0, minY: 9, maxX: 20, maxY: 3 },
      }),
    RangeError,
  );
  assert.throws(() => wholeMapCameraFrame(0, 12), RangeError);
});

test("Everyone in View's margin and zoom steps are kept here for the camera look", () => {
  // The Everyone in View rule itself has one owner, looks/camera.js (checked in
  // camera.test.js); layout.js keeps only its numbers.
  assert.equal(EVERYONE_IN_VIEW_MARGIN, 1.5);
  assert.equal(EVERYONE_IN_VIEW_MAX_ZOOM, 2);
  assert.equal(EVERYONE_IN_VIEW_ZOOM_STEP, 0.25);
});
