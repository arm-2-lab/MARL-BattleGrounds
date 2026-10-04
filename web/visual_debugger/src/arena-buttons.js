/**
 * @file The arena's on-screen keys, stacked at its top right, key only (owner,
 * 4 Oct 2026; earlier, 3 Oct: one button for each job, in the place that
 * leaves the battlefield the most room): V Everyone In View, B Battle Cam
 * (Replay Viewer only), F Frame Explanation and Esc Clear Selection (Replay
 * Viewer only). Each button's name is its aria-label and its hover card
 * (main.js CONTROL_HELP); only its keycap shows. The Replay Viewer first, the
 * BattleClient and DevClient reuse the same markup and this module.
 *
 * What it owns: showing index.html's #arena-buttons group, and the click
 * wiring and lit state of its V, B and Esc buttons. It owns no camera,
 * selection or explanation rule. A click presses the button's key through
 * the one command set (command-set.js runLocalCommand), so it does exactly
 * what the key does, with the key's own limits (the Camera Framing switch for
 * V and B). V's and B's lit state (aria-pressed) mirrors the camera slot's
 * data-camera ("everyone_in_view", "battle_cam"). The Frame Explanation
 * button in the same group is Frame Explanation's own button: the page passes
 * it to createFrameExplanation (looks/frame-explanation.js), which shows it,
 * answers its click exactly as F and sets its aria-pressed. No timers,
 * storage or network.
 */
import { runLocalCommand } from "./command-set.js";

/**
 * @typedef {{
 *   group: HTMLElement,
 *   everyoneInView: HTMLButtonElement,
 *   battleCam?: HTMLButtonElement | null,
 *   clearSelection?: HTMLButtonElement | null,
 *   product: "combat_debugger" | "replay_viewer",
 *   cameraSlot?: HTMLElement | null,
 *   context?: () => Readonly<Record<string, unknown>>,
 * }} ArenaButtonsOptions
 *   group: the #arena-buttons group (shown by wireArenaButtons);
 *   everyoneInView: its Everyone In View button (V); battleCam and
 *   clearSelection: its Battle Cam (B) and Clear Selection (Esc) buttons,
 *   used only in the Replay Viewer (default null; in live play they stay
 *   hidden); product: whose command table the keys use; cameraSlot: the
 *   camera switch slot whose data-camera names the camera shown (and, in the
 *   Replay Viewer, whose hidden flag hides V and B, because the camera switch
 *   is off), default null (never lit); context: extra facts passed to the
 *   key's handler, as the keyboard path passes them (live: {interactive}),
 *   default none.
 */

/**
 * Show the arena's on-screen keys and wire V, B and Esc.
 *
 * A click on a key button presses its key ("v", "b" or "Escape") through
 * runLocalCommand(product, key record, context): a plain key record {key,
 * target (the button), no modifier keys, not a repeat}, with context() plus
 * {source: "arena_buttons"}. A press the handler refuses (V or B with Camera
 * Framing off) changes nothing. V's aria-pressed is true while the camera
 * slot's data-camera is "everyone_in_view" and B's while it is "battle_cam",
 * read now and on every change of that attribute or the slot's hidden flag
 * (one MutationObserver when the page has MutationObserver). In the Replay
 * Viewer, B and Esc are shown, and V and B are hidden while the camera slot
 * is hidden; in live play B and Esc stay hidden.
 *
 * Side effects: shows group, adds one click listener per wired button and at
 * most one observer, and sets aria-pressed and hidden on the buttons.
 *
 * @param {ArenaButtonsOptions} options
 * @returns {Readonly<{sync: () => void, destroy: () => void}>} sync reads the
 *   camera slot again; destroy removes the listeners and observer and hides
 *   the group.
 */
export function wireArenaButtons(options) {
  const { group, everyoneInView, product } = options;
  const replay = product === "replay_viewer";
  const battleCam = replay ? (options.battleCam ?? null) : null;
  const clearSelection = replay ? (options.clearSelection ?? null) : null;
  const cameraSlot = options.cameraSlot ?? null;
  const context = options.context ?? (() => ({}));

  /** @param {string} key @param {HTMLButtonElement} target */
  const press = (key, target) => {
    runLocalCommand(
      product,
      /** @type {any} */ ({
        key,
        target,
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
  /** @type {Array<[HTMLButtonElement, () => void]>} */
  const wired = [[everyoneInView, () => press("v", everyoneInView)]];
  if (battleCam !== null) wired.push([battleCam, () => press("b", battleCam)]);
  if (clearSelection !== null) {
    wired.push([clearSelection, () => press("Escape", clearSelection)]);
  }

  function sync() {
    const camera = cameraSlot?.getAttribute("data-camera") ?? null;
    const cameraOff = replay && cameraSlot?.hidden === true;
    everyoneInView.setAttribute("aria-pressed", String(camera === "everyone_in_view"));
    everyoneInView.hidden = cameraOff;
    if (battleCam !== null) {
      battleCam.setAttribute("aria-pressed", String(camera === "battle_cam"));
      battleCam.hidden = cameraOff;
    }
    if (clearSelection !== null) clearSelection.hidden = false;
  }

  for (const [button, onClick] of wired) button.addEventListener("click", onClick);
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
      for (const [button, onClick] of wired) {
        button.removeEventListener("click", onClick);
      }
      observer?.disconnect();
      group.hidden = true;
    },
  });
}
