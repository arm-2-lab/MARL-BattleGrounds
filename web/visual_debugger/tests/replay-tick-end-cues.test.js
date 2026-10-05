/**
 * @file Check how a played replay tick ends (src/choreography.js and
 * src/choreography-painter.js): the live install policies (live_once, and its
 * safe rebuild) are exactly as before, with no retained transients; a played
 * replay tick (replay_animated) asks the painter to retain its transients, so
 * when its cleanup clock settles it nothing is removed or redrawn (a held cue
 * stays, a number that already faded stays faded, as the tick drew them), and
 * its cues leave only when the next tick's installation replaces it; the
 * Settled Still (replay_static) still shows every transient fully opaque; a
 * live settle still removes transients; when the next played tick starts, the
 * kept cues are handed to a fade-out (their roots hold END opacity 0, a finished
 * fade removes its root) instead of a clear, while a seek clears them at once
 * and reduced motion never fades; and the painter hands over only the kept
 * cues (the herald root and persistent marks are removed at once, a root no
 * longer in the page is not returned). A cue fades out once: in a played tick
 * (replay or live) a look animation that ends at opacity 0 with no forward
 * fill leaves its transient node at opacity 0 when it finishes, so the still
 * picture under it never comes back that tick; a node in a persistent event,
 * an animation that ends visible or has a forward fill, and a node of a tick
 * already replaced are left alone.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { CombatChoreographer, ConsumedTransitionLedger } from "../src/choreography.js";
import { SvgChoreographyPainter } from "../src/choreography-painter.js";

class FakeAnimation {
  /** @param {string} id @param {number} [duration] */
  constructor(id, duration = 0) {
    this.id = id;
    this.currentTime = 0;
    this.playbackRate = 1;
    this.playState = "running";
    this.duration = duration;
    this.finished = new Promise((resolve, reject) => {
      this.resolveFinished = resolve;
      this.rejectFinished = reject;
    });
  }
  pause() {
    this.playState = "paused";
  }
  play() {
    this.playState = "running";
  }
  /** @param {number} rate */
  updatePlaybackRate(rate) {
    this.playbackRate = rate;
  }
  finish() {
    this.playState = "finished";
    this.resolveFinished(this);
  }
  cancel() {
    this.playState = "idle";
    this.rejectFinished(new Error("cancelled"));
  }
}

class FakeAnimationFactory {
  constructor() {
    /** @type {FakeAnimation[]} */
    this.created = [];
  }
  /** @param {{id: string}} spec */
  create(spec) {
    const animation = new FakeAnimation(spec.id);
    this.created.push(animation);
    return animation;
  }
  /** @param {unknown} _element @param {number} duration @param {string} id */
  createClock(_element, duration, id) {
    const animation = new FakeAnimation(id, duration);
    this.created.push(animation);
    return animation;
  }
}

class FakePainter {
  constructor() {
    /** @type {any[][]} */
    this.calls = [];
  }
  /** @param {Record<string, any>} plan @param {unknown} _surface @param {Record<string, any>} options */
  install(plan, _surface, options) {
    this.calls.push(["install", plan.epochKey, options]);
    return {
      root: /** @type {any} */ ({}),
      nodeCount: 6,
      persistentNodeCount: 2,
      animationSpecs:
        options.settled || options.persistentOnly
          ? []
          : [
              {
                element: /** @type {any} */ ({}),
                keyframes: [],
                options: {},
                id: `mbg:${plan.epochKey}:e`,
              },
            ],
    };
  }
  /** @param {unknown} _installation @param {string} reason */
  clear(_installation, reason) {
    this.calls.push(["clear", reason]);
  }
  settle() {
    this.calls.push(["settle"]);
  }
  reproject() {
    this.calls.push(["reproject"]);
  }
}

/** @param {string} epoch */
function plan(epoch) {
  return {
    epochKey: epoch,
    authorizationKey: "researcher",
    fingerprint: `events-${epoch}`,
    paintKey: "visual-filters-v4:test",
    phases: { submissionRelease: 450, reducedTotal: 220, total: 900 },
    bounds: { nodes: 16, animations: 8, persistentNodes: 2 },
    events: [],
  };
}

class FakeRoot {
  constructor() {
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    this.removed = false;
    this.isConnected = true;
  }
  /** @param {string} name @param {string} value */
  setAttribute(name, value) {
    this.attributes.set(name, value);
  }
  remove() {
    this.removed = true;
    this.isConnected = false;
  }
}

/** A painter that can hand a settled tick's cues over to a fade. */
class FadingPainter extends FakePainter {
  constructor() {
    super();
    /** @type {FakeRoot[]} */
    this.roots = [];
  }
  /** @param {unknown} _installation */
  releaseForFade(_installation) {
    this.calls.push(["release"]);
    const roots = [new FakeRoot(), new FakeRoot()];
    this.roots.push(...roots);
    return roots;
  }
}

/** @param {{painter?: FakePainter, motionMode?: "normal" | "reduced"}} [options] */
function harness(options = {}) {
  const painter = options.painter ?? new FakePainter();
  const animationFactory = new FakeAnimationFactory();
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (painter),
    animationFactory,
    ledger: new ConsumedTransitionLedger({ storage: null, storageKey: "t", limit: 4 }),
    motionMode: options.motionMode ?? "normal",
    planBuilder: (frame) => /** @type {any} */ (frame)?.plan ?? null,
  });
  return { painter, animationFactory, controller };
}

const surface = /** @type {any} */ ({ viewportKey: "16:12:800:600" });

/** @param {any[][]} calls */
// The page clock (the card queue's) changes every call; checks compare the rest.
const installOptions = (calls) =>
  calls
    .filter(([kind]) => kind === "install")
    .map((call) => {
      const { clock: _clock, ...rest } = call[2];
      return rest;
    });

test("live install policies are unchanged and retain no transients", () => {
  const live = harness();
  live.controller.presentFrame({ plan: plan("live-1") }, surface, {
    renderPolicy: "live_once",
  });
  // The same tick again with other switches: an already consumed live tick
  // is rebuilt safely (settled, persistent nodes only).
  live.controller.presentFrame(
    { plan: { ...plan("live-1"), paintKey: "visual-filters-v4:other" } },
    surface,
    { renderPolicy: "live_once" },
  );
  assert.deepEqual(installOptions(live.painter.calls), [
    {
      motionMode: "normal",
      renderPolicy: "live_once",
      settled: false,
      persistentOnly: false,
      playbackRate: 1,
    },
    {
      motionMode: "normal",
      renderPolicy: "live_once",
      settled: true,
      persistentOnly: true,
      playbackRate: 1,
    },
  ]);
});

test("a played replay tick keeps its cues at settle and loses them when the next tick starts", async () => {
  const { painter, animationFactory, controller } = harness();
  controller.presentFrame({ plan: plan("tick-60") }, surface, {
    renderPolicy: "replay_animated",
  });
  assert.deepEqual(installOptions(painter.calls)[0], {
    motionMode: "normal",
    renderPolicy: "replay_animated",
    settled: false,
    persistentOnly: false,
    playbackRate: 1,
    retainTransientOnSettle: true,
  });
  for (const animation of animationFactory.created) animation.finish();
  for (let index = 0; index < 6; index += 1) await Promise.resolve();
  assert.deepEqual(
    painter.calls.map(([kind]) => kind),
    ["install", "settle"],
    "the cleanup clock settles the tick and nothing is cleared yet",
  );
  controller.presentFrame({ plan: plan("tick-61") }, surface, {
    renderPolicy: "replay_animated",
  });
  assert.deepEqual(
    painter.calls.slice(2).map(([kind]) => kind),
    ["clear", "install"],
    "the next tick replaces the settled tick's installation",
  );
  controller.presentFrame({ plan: plan("tick-62") }, surface, {
    renderPolicy: "replay_static",
  });
  assert.deepEqual(installOptions(painter.calls).at(-1), {
    motionMode: "normal",
    renderPolicy: "replay_static",
    settled: true,
    persistentOnly: false,
    playbackRate: 1,
    retainTransientOnSettle: true,
  });
});

/**
 * @param {"live_once" | "replay_animated" | "replay_static"} renderPolicy
 * @param {boolean} retain
 */
function fakeInstallation(renderPolicy, retain) {
  /** @param {string} persistent @param {string} opacity */
  const group = (persistent, opacity) => {
    const attributes = new Map([["opacity", opacity]]);
    return {
      dataset: /** @type {Record<string, string>} */ ({ persistent }),
      removed: false,
      /** @param {string} name @param {string} value */
      setAttribute(name, value) {
        attributes.set(name, value);
      },
      /** @param {string} name */
      getAttribute(name) {
        return attributes.get(name) ?? null;
      },
      remove() {
        this.removed = true;
      },
    };
  };
  const held = group("false", "1");
  const faded = group("false", "0");
  const persistent = group("true", "0");
  const root = () => ({ dataset: /** @type {Record<string, string>} */ ({}) });
  const installation = /** @type {any} */ ({
    root: root(),
    connectorRoot: root(),
    routeRoot: root(),
    eventNodes: new Map([
      ["held", { group: held }],
      ["faded", { group: faded }],
      ["persistent", { group: persistent }],
    ]),
    renderPolicy,
    retainTransientOnSettle: retain,
  });
  return { installation, held, faded, persistent };
}

test("the painter leaves a played tick's cues as drawn, keeps the Settled Still opaque and clears live transients", () => {
  const painter = new SvgChoreographyPainter();
  const played = fakeInstallation("replay_animated", true);
  painter.settle(played.installation);
  assert.equal(played.held.removed, false);
  assert.equal(played.faded.removed, false);
  assert.equal(played.held.getAttribute("opacity"), "1", "a held cue stays");
  assert.equal(played.faded.getAttribute("opacity"), "0", "a faded number stays faded");
  assert.equal(played.persistent.getAttribute("opacity"), "1");
  assert.equal(played.installation.eventNodes.size, 3);

  const still = fakeInstallation("replay_static", true);
  painter.settle(still.installation);
  assert.equal(still.faded.getAttribute("opacity"), "1", "the Settled Still shows it");

  const live = fakeInstallation("live_once", false);
  painter.settle(live.installation);
  assert.equal(live.held.removed, true);
  assert.equal(live.faded.removed, true);
  assert.equal(live.persistent.removed, false);
  assert.deepEqual([...live.installation.eventNodes.keys()], ["persistent"]);
});

/**
 * @param {ReturnType<typeof harness>} h
 * @param {string} epoch
 */
async function playTick(h, epoch) {
  h.controller.presentFrame({ plan: plan(epoch) }, surface, {
    renderPolicy: "replay_animated",
  });
  for (const animation of h.animationFactory.created) {
    if (!animation.id.includes("cue-fade-out")) animation.finish();
  }
  for (let index = 0; index < 6; index += 1) await Promise.resolve();
}

test("the next played tick fades the kept cues out, and a seek or reduced motion clears them at once", async () => {
  const painter = new FadingPainter();
  const h = harness({ painter });
  await playTick(h, "tick-60");
  h.controller.presentFrame({ plan: plan("tick-61") }, surface, {
    renderPolicy: "replay_animated",
  });
  assert.deepEqual(
    painter.calls.slice(2).map(([kind]) => kind),
    ["install", "release"],
    "the settled tick is handed to a fade after the new install, never cleared",
  );
  const fades = h.animationFactory.created.filter((a) => a.id.includes("cue-fade-out"));
  assert.equal(fades.length, 2);
  for (const root of painter.roots) {
    assert.equal(root.attributes.get("opacity"), "0", "the DOM holds END");
    assert.equal(root.removed, false);
  }
  fades[0].finish();
  for (let index = 0; index < 4; index += 1) await Promise.resolve();
  assert.equal(painter.roots[0].removed, true, "a finished fade removes its root");
  assert.equal(painter.roots[1].removed, false);
  // A seek (the Settled Still) clears the rest at once.
  h.controller.presentFrame({ plan: plan("tick-62") }, surface, {
    renderPolicy: "replay_static",
  });
  assert.equal(painter.roots[1].removed, true);
  assert.equal(fades[1].playState, "idle", "its fade is cancelled");

  const reduced = harness({ painter: new FadingPainter(), motionMode: "reduced" });
  await playTick(reduced, "tick-60");
  reduced.controller.presentFrame({ plan: plan("tick-61") }, surface, {
    renderPolicy: "replay_animated",
  });
  assert.deepEqual(
    reduced.painter.calls.slice(2).map(([kind]) => kind),
    ["clear", "install"],
    "reduced motion clears at once",
  );
});

test("the painter hands only a played tick's kept cues to the fade", () => {
  const painter = new SvgChoreographyPainter();
  const played = fakeInstallation("replay_animated", true);
  /** @param {boolean} connected */
  const root = (connected) => ({
    isConnected: connected,
    removed: false,
    remove() {
      this.removed = true;
    },
  });
  const heraldRoot = root(true);
  Object.assign(played.installation, {
    root: { ...played.installation.root, ...root(true) },
    connectorRoot: { ...played.installation.connectorRoot, ...root(true) },
    routeRoot: { ...played.installation.routeRoot, ...root(false) },
    heraldRoot,
  });
  const roots = painter.releaseForFade(played.installation);
  assert.equal(heraldRoot.removed, true, "the next tick draws its own herald card");
  assert.equal(played.persistent.removed, true, "persistent marks are drawn again");
  assert.equal(played.held.removed, false);
  assert.deepEqual(roots, [
    played.installation.root,
    played.installation.connectorRoot,
  ]);
});

class FakeCue {
  /** @param {boolean} persistent */
  constructor(persistent) {
    this.persistent = persistent;
    /** @type {Map<string, string>} */
    this.attributes = new Map([["opacity", "0.8"]]);
  }
  /** @param {string} name @param {string} value */
  setAttribute(name, value) {
    this.attributes.set(name, value);
  }
  /** @param {string} selector */
  closest(selector) {
    return selector === '[data-persistent="true"]' && this.persistent ? this : null;
  }
}

/** A painter whose one tick has four look animations over four cues. */
class CuePainter extends FakePainter {
  constructor() {
    super();
    /** @type {Record<string, FakeCue>[]} */
    this.ticks = [];
  }
  /** @param {Record<string, any>} plan @param {unknown} _surface @param {Record<string, any>} options @returns {any} */
  install(plan, _surface, options) {
    this.calls.push(["install", plan.epochKey, options]);
    const cues = {
      fades: new FakeCue(false),
      persistent: new FakeCue(true),
      staysLit: new FakeCue(false),
      forwardFill: new FakeCue(false),
    };
    this.ticks.push(cues);
    const out = [{ opacity: 0 }, { opacity: 1, offset: 0.3 }, { opacity: 0 }];
    /** @param {FakeCue} element @param {Keyframe[]} keyframes @param {FillMode} fill @param {string} name */
    const spec = (element, keyframes, fill, name) => ({
      element: /** @type {any} */ (element),
      keyframes,
      options: { delay: 0, duration: 600, fill },
      id: `mbg:${plan.epochKey}:${name}`,
    });
    return {
      root: /** @type {any} */ ({}),
      nodeCount: 6,
      persistentNodeCount: 2,
      animationSpecs: options.settled
        ? []
        : [
            spec(cues.fades, out, "backwards", "fades"),
            spec(cues.persistent, out, "backwards", "persistent"),
            spec(cues.staysLit, [{ opacity: 0 }, { opacity: 1 }], "backwards", "lit"),
            spec(cues.forwardFill, out, "both", "forward"),
          ],
    };
  }
}

test("a cue that has faded out stays out for the rest of its played tick", async () => {
  for (const renderPolicy of /** @type {const} */ (["replay_animated", "live_once"])) {
    const painter = new CuePainter();
    const h = harness({ painter });
    h.controller.presentFrame({ plan: plan(`${renderPolicy}-60`) }, surface, {
      renderPolicy,
    });
    const cues = painter.ticks[0];
    for (const animation of h.animationFactory.created) {
      if (/:(fades|persistent|lit|forward)$/u.test(animation.id)) animation.finish();
    }
    for (let index = 0; index < 4; index += 1) await Promise.resolve();
    assert.equal(
      cues.fades.attributes.get("opacity"),
      "0",
      `${renderPolicy}: faded stays out`,
    );
    assert.equal(
      cues.persistent.attributes.get("opacity"),
      "0.8",
      "a persistent still returns",
    );
    assert.equal(
      cues.staysLit.attributes.get("opacity"),
      "0.8",
      "ends visible: untouched",
    );
    assert.equal(
      cues.forwardFill.attributes.get("opacity"),
      "0.8",
      "forward fill: untouched",
    );
  }
  // A fade that finishes after its tick was replaced changes nothing.
  const painter = new CuePainter();
  const h = harness({ painter });
  h.controller.presentFrame({ plan: plan("tick-60") }, surface, {
    renderPolicy: "replay_animated",
  });
  const late = h.animationFactory.created.find((a) => a.id.endsWith(":fades"));
  h.controller.presentFrame({ plan: plan("tick-61") }, surface, {
    renderPolicy: "replay_animated",
  });
  late?.finish();
  for (let index = 0; index < 4; index += 1) await Promise.resolve();
  assert.equal(painter.ticks[0].fades.attributes.get("opacity"), "0.8");
});
