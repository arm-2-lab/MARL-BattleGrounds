/**
 * @file Check that live ticks flow into each other without a cut or a still.
 *
 * Contracts:
 * (1) CombatChoreographer.presentFrame keeps a running live tick when a new
 * server revision of the same tick arrives for the same audience (same
 * epochKey, audienceKey, fingerprint, paintKey and render policy; only the
 * authorizationKey's authority epoch moved): no animation is cancelled, the
 * painter neither clears nor installs again, and the plan and the
 * installation roots carry the new authorizationKey.
 * (2) A change of audience at the same tick (Agent POV to Oracle View) still
 * clears the old nodes and installs again, and plans without an audienceKey
 * keep the old rule: any authorizationKey change clears.
 * (3) main.js's own Run (Policies Only) sends the next Play One Turn once the
 * tick on screen has released submission, without cutting that tick short
 * (keepPresentation), holds the next tick back (liveRunHoldsNextTick) until
 * the tick on screen settles, and the live poll installs a snapshot read
 * during the hold only after that settle. dispatchCommand skips its
 * settle-before-submit step only when keepPresentation is set.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { CombatChoreographer, ConsumedTransitionLedger } from "../src/choreography.js";

const MAIN = readFileSync(new URL("../src/main.js", import.meta.url), "utf8");

class FakeAnimation {
  /** @param {string} id
   * @param {KeyframeAnimationOptions} [timing] */
  constructor(id, timing = {}) {
    this.id = id;
    this.currentTime = 0;
    this.playbackRate = 1;
    this.playState = "running";
    this.duration = Number(timing.duration ?? 0);
    this.finished = new Promise((resolve, reject) => {
      this.resolveFinished = resolve;
      this.rejectFinished = reject;
    });
    this.finished.catch(() => {});
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

class FakeElement {
  constructor() {
    /** @type {Map<string, string>} */
    this.attributes = new Map();
  }

  /** @param {string} name
   * @param {string} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
}

function harness() {
  /** @type {FakeAnimation[]} */
  const created = [];
  /** @type {any[][]} */
  const calls = [];
  const painter = {
    /** @param {Record<string, any>} plan
     * @param {unknown} _surface
     * @param {Record<string, any>} options */
    install(plan, _surface, options) {
      calls.push(["install", plan.authorizationKey]);
      const root = new FakeElement();
      root.setAttribute("data-authorization-key", plan.authorizationKey);
      return {
        root,
        routeRoot: new FakeElement(),
        connectorRoot: new FakeElement(),
        nodeCount: 4,
        persistentNodeCount: 1,
        animationSpecs: options.settled
          ? []
          : [
              {
                element: {},
                keyframes: [],
                options: {},
                id: `mbg:${plan.epochKey}:effect-0`,
              },
            ],
      };
    },
    /** @param {unknown} _installation
     * @param {string} reason */
    clear(_installation, reason) {
      calls.push(["clear", reason]);
    },
    settle() {
      calls.push(["settle"]);
    },
    reproject() {
      calls.push(["reproject"]);
    },
  };
  const animationFactory = {
    /** @param {{id: string, options: KeyframeAnimationOptions}} spec */
    create(spec) {
      const animation = new FakeAnimation(spec.id, spec.options);
      created.push(animation);
      return animation;
    },
    /** @param {unknown} _element
     * @param {number} duration
     * @param {string} id */
    createClock(_element, duration, id) {
      const animation = new FakeAnimation(id, { duration });
      created.push(animation);
      return animation;
    },
  };
  const controller = new CombatChoreographer({
    painter: /** @type {any} */ (painter),
    animationFactory: /** @type {any} */ (animationFactory),
    ledger: new ConsumedTransitionLedger({
      storage: null,
      storageKey: "smooth-run-test",
      limit: 4,
    }),
    motionMode: "normal",
    planBuilder: (frame) => /** @type {any} */ (frame)?.plan ?? null,
  });
  return { calls, controller, created };
}

/** @param {{authority: number, audience?: string | null}} options */
function plan({ authority, audience = '["session","agent_pov","pov-a"]' }) {
  return {
    epochKey: '["session","transition-7"]',
    authorizationKey: JSON.stringify(["session", authority, "agent_pov", "pov-a"]),
    audienceKey: audience,
    fingerprint: "3:abc",
    paintKey: "visual-filters-v2:all-on",
    simulatorStep: 7,
    phases: { submissionRelease: 450, reducedTotal: 220, total: 1100 },
    bounds: { nodes: 16, animations: 8, persistentNodes: 4, overrideAnimations: 8 },
    events: [],
  };
}

const surface = /** @type {any} */ ({ viewportKey: "0:0:1074:552" });

test("a new revision of the same tick for the same audience keeps the glide", () => {
  const { calls, controller, created } = harness();
  controller.presentFrame({ plan: plan({ authority: 3 }) }, surface);
  const running = created.filter((animation) => animation.playState === "running");
  assert.ok(running.length > 0);
  const installsBefore = calls.filter(([kind]) => kind === "install").length;

  controller.presentFrame({ plan: plan({ authority: 4 }) }, surface);

  assert.equal(calls.filter(([kind]) => kind === "install").length, installsBefore);
  assert.deepEqual(
    calls.filter(([kind]) => kind === "clear"),
    [],
  );
  assert.ok(running.every((animation) => animation.playState === "running"));
  const key = JSON.stringify(["session", 4, "agent_pov", "pov-a"]);
  assert.equal(controller.snapshot().authorizationKey, key);
  assert.equal(controller.snapshot().submissionBlocked, true);
});

test("a new audience at the same tick still clears and installs again", () => {
  const { calls, controller, created } = harness();
  controller.presentFrame({ plan: plan({ authority: 3 }) }, surface);
  const first = created.filter((animation) => animation.playState === "running");

  controller.presentFrame(
    { plan: plan({ authority: 4, audience: '["session","oracle",null]' }) },
    surface,
  );

  assert.deepEqual(
    calls.filter(([kind]) => kind === "clear").map(([, reason]) => reason),
    ["authorization_or_disclosure_changed"],
  );
  assert.equal(calls.filter(([kind]) => kind === "install").length, 2);
  assert.ok(first.every((animation) => animation.playState === "idle"));
});

test("plans without an audience key keep the old clear rule", () => {
  const { calls, controller } = harness();
  controller.presentFrame({ plan: plan({ authority: 3, audience: null }) }, surface);
  controller.presentFrame({ plan: plan({ authority: 4, audience: null }) }, surface);
  assert.deepEqual(
    calls.filter(([kind]) => kind === "clear").map(([, reason]) => reason),
    ["authorization_or_disclosure_changed"],
  );
});

test("main.js Run overlaps the policies' work and holds the next tick", () => {
  const loop = /async function runLiveTurns\(\) \{[\s\S]*?\n\}/u.exec(MAIN)?.[0] ?? "";
  assert.ok(loop, "main.js defines runLiveTurns");
  const released = loop.indexOf("await whenRunTickReleased();");
  const hold = loop.indexOf("liveRunHoldsNextTick = true;");
  const send = loop.search(
    /dispatchCommand\(keyboardCommand\("Enter"\), \{\s*source: "control",\s*quiet: true,\s*keepPresentation: true,\s*\}\)/u,
  );
  const settle = loop.indexOf("await choreographer.whenSettled();", send);
  const release = loop.indexOf("liveRunHoldsNextTick = false;", settle);
  assert.ok(released >= 0 && hold > released && send > hold, "send after release");
  assert.ok(settle > send && release > settle, "hold until the tick settles");
  assert.match(
    MAIN,
    /if \(joined !== null && liveRunHoldsNextTick\) \{[\s\S]{0,320}?await choreographer\.whenSettled\(\);\s*\}\s*if \(\s*joined === null/u,
  );
  assert.match(
    MAIN,
    /isSubmissionCommand\(command\) &&\s*!keepPresentation &&\s*presentationRequiresSubmissionSettle\(choreographer\.snapshot\(\)\)/u,
  );
});
