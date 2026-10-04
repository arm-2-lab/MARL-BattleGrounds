/**
 * @file The arena's two on-screen keys, top right: Everyone In View (V) and
 * Frame Explanation (F) (owner, 3 Oct 2026: one button for each job, in the
 * place that leaves the battlefield the most room; the Replay Viewer first,
 * the BattleClient reuses the same markup and this module).
 *
 * What it owns: showing index.html's #arena-buttons group, and the click
 * wiring and lit state of its Everyone In View button. It owns no camera or
 * explanation rule. A click on Everyone In View presses "v" through the one
 * command set (command-set.js runLocalCommand), so it does exactly what V
 * does, with the camera's own limits (the Camera Framing switch), and its
 * lit state (aria-pressed) mirrors the camera slot's data-camera
 * ("everyone_in_view"). The Frame Explanation button in the same group is
 * Frame Explanation's own button: the page passes it to
 * createFrameExplanation (looks/frame-explanation.js), which shows it,
 * answers its click exactly as F and sets its aria-pressed. No timers,
 * storage or network.
 */
import { runLocalCommand } from "./command-set.js";

/**
 * @typedef {{
 *   group: HTMLElement,
 *   everyoneInView: HTMLButtonElement,
 *   product: "combat_debugger" | "replay_viewer",
 *   cameraSlot?: HTMLElement | null,
 *   context?: () => Readonly<Record<string, unknown>>,
 * }} ArenaButtonsOptions
 *   group: the #arena-buttons group (shown by wireArenaButtons);
 *   everyoneInView: its Everyone In View button; product: whose command
 *   table the V key uses; cameraSlot: the camera switch slot whose
 *   data-camera names the camera shown (and, in the Replay Viewer, whose
 *   hidden flag hides the button, because the camera switch is off),
 *   default null (never lit); context: extra facts passed to the key's
 *   handler, as the keyboard path passes them (live: {interactive}),
 *   default none.
 */

/**
 * Show the arena's on-screen keys and wire Everyone In View.
 *
 * A click on Everyone In View presses "v" through runLocalCommand(product,
 * key record, context): a plain key record {key "v", target (the button), no
 * modifier keys, not a repeat}, with context() plus {source:
 * "arena_buttons"}. A press the handler refuses (V with Camera Framing off)
 * changes nothing. The button's aria-pressed is true while the camera slot's
 * data-camera is "everyone_in_view", read now and on every change of that
 * attribute or the slot's hidden flag (one MutationObserver when the page
 * has MutationObserver). In the Replay Viewer the button is hidden while the
 * camera slot is hidden.
 *
 * Side effects: shows group, adds one click listener and at most one
 * observer, and sets aria-pressed and hidden on the button.
 *
 * @param {ArenaButtonsOptions} options
 * @returns {Readonly<{sync: () => void, destroy: () => void}>} sync reads the
 *   camera slot again; destroy removes the listener and observer and hides
 *   the group.
 */
export function wireArenaButtons(options) {
  const { group, everyoneInView, product } = options;
  const cameraSlot = options.cameraSlot ?? null;
  const context = options.context ?? (() => ({}));

  const onEveryoneInView = () => {
    runLocalCommand(
      product,
      /** @type {any} */ ({
        key: "v",
        target: everyoneInView,
        shiftKey: false,
        ctrlKey: false,
        altKey: false,
        metaKey: false,
        repeat: false,
      }),
      { ...context(), source: "arena_buttons" },
    );
    sync();
  };

  function sync() {
    const camera = cameraSlot?.getAttribute("data-camera") ?? null;
    everyoneInView.setAttribute("aria-pressed", String(camera === "everyone_in_view"));
    everyoneInView.hidden = product === "replay_viewer" && cameraSlot?.hidden === true;
  }

  everyoneInView.addEventListener("click", onEveryoneInView);
  /** @type {MutationObserver | null} */
  let observer = null;
  const Observer = globalThis.MutationObserver;
  if (cameraSlot !== null && typeof Observer === "function") {
    observer = new Observer(sync);
    observer.observe(cameraSlot, {
      attributes: true,
      attributeFilter: ["data-camera", "hidden"],
    });
  }
  group.hidden = false;
  sync();

  return Object.freeze({
    sync,
    destroy() {
      everyoneInView.removeEventListener("click", onEveryoneInView);
      observer?.disconnect();
      group.hidden = true;
    },
  });
}
