/**
 * @file Own the one command set: every key the BattleClient, the DevClient
 * combat screen and the Replay Viewer know, and the job each key does in each.
 *
 * The rule (owner, 30 Sep and 1 Oct 2026): a key does the same job in every
 * client, or nothing. COMMAND_SET encodes the plan's command table row by row.
 * Each row names its keys, how they print on a keycap, and for each product
 * ("combat_debugger" is the BattleClient and the DevClient combat screen,
 * "replay_viewer" is the Replay Viewer) either null (the key does nothing
 * there) or an entry with the shared job ID, a plain help line, the route
 * ("forwarded" to Python as a keyboard command, or "local" in the browser) and
 * the owning handler.
 *
 * Who calls it: `controls.js` (which live keys are forwarded, and V and F on the
 * battlefield), `replay-controls.js` (Replay Viewer keys), the camera, Frame
 * Explanation, How to Play and Help (help lines). Local jobs run through a tiny
 * registry: the module that owns a job registers its handler once with
 * registerLocalCommandHandler, and key listeners call runLocalCommand.
 *
 * Boundary: this file decides what a key means. It sends no command to Python,
 * changes no simulation and reads no frame, so it cannot leak anything hidden
 * in Agent POV; each registered handler keeps its own Agent POV limits. Python
 * stays the authority for every forwarded key.
 */

/**
 * @typedef {"combat_debugger" | "replay_viewer"} CommandProduct
 * @typedef {"forwarded" | "local"} CommandRoute
 * @typedef {"up" | "down" | "any"} CommandShift
 * @typedef {Readonly<{
 *   job: string,
 *   help: string,
 *   route: CommandRoute,
 *   handler: string,
 *   onField: boolean,
 * }>} CommandEntry
 * @typedef {Readonly<{
 *   id: string,
 *   keys: ReadonlyArray<string>,
 *   keycaps: string,
 *   shift: CommandShift,
 *   skipsTextAndDialogs: boolean,
 *   jobs: Readonly<Record<CommandProduct, CommandEntry | null>>,
 * }>} CommandRow
 *   skipsTextAndDialogs true marks a key that keeps its job while a button has
 *   focus (V, B, F, G and N: a focused transport button does not swallow them)
 *   but never fires from a text field or an open dialog. Space and Enter are
 *   not marked, so a focused button still takes them. In live play G and N are
 *   forwarded to Python, so the mark changes only the Replay Viewer for them.
 * @typedef {Readonly<{
 *   row: CommandRow,
 *   entry: CommandEntry,
 *   key: string,
 * }>} CommandMatch
 * @typedef {{
 *   key: string,
 *   shiftKey?: boolean,
 *   ctrlKey?: boolean,
 *   altKey?: boolean,
 *   metaKey?: boolean,
 *   repeat?: boolean,
 *   target?: unknown,
 * }} CommandKeyEvent
 * @typedef {Readonly<{
 *   product: CommandProduct,
 *   job: string,
 *   key: string,
 *   shiftKey: boolean,
 *   repeat: boolean,
 *   event: CommandKeyEvent,
 *   context: Readonly<Record<string, unknown>>,
 * }>} LocalCommandCall
 * @typedef {(call: LocalCommandCall) => boolean | void} LocalCommandHandler
 */

/** The two products that share this command set. */
export const COMMAND_PRODUCTS = Object.freeze(
  /** @type {CommandProduct[]} */ (["combat_debugger", "replay_viewer"]),
);

/**
 * The handler name for every key Python handles. A forwarded key is sent as a
 * keyboard command; Python decides whether it is allowed.
 */
export const PYTHON_HANDLER = "python";

/**
 * Keys the owner has not approved (plan P29). No row may bind them; they are
 * listed so the owner can decide later: H (Clean Screen), [ and ], { and },
 * M, < and >, Home, End and /.
 */
export const UNAPPROVED_KEYS = Object.freeze([
  "h",
  "[",
  "]",
  "{",
  "}",
  "m",
  "<",
  ">",
  "Home",
  "End",
  "/",
]);

/**
 * The command table, in the plan's order. Keys use KeyboardEvent.key values;
 * single letters are lowercase and match either case.
 *
 * @type {ReadonlyArray<CommandRow>}
 */
export const COMMAND_SET = Object.freeze([
  commandRow({
    id: "step",
    keys: ["w", "a", "s", "d", "q", "e", "z", "c"],
    keycaps: "W A S D Q E Z C",
    combat: forwarded("step", "Step (staged)"),
    replay: null,
  }),
  commandRow({
    id: "stay",
    keys: ["x"],
    keycaps: "X",
    combat: forwarded("stay", "Stay"),
    replay: null,
  }),
  commandRow({
    id: "combat_choice",
    keys: ["0", "1", "2"],
    keycaps: "0 / 1 / 2",
    combat: forwarded("combat_choice", "No Combat / Basic / Ultimate"),
    replay: null,
  }),
  commandRow({
    id: "go",
    keys: [" "],
    keycaps: "Space",
    combat: forwarded("go", "Lock In (Policies Only: play one turn)"),
    replay: local("go", "Play or pause", "replay_play_pause"),
  }),
  commandRow({
    id: "lock_in",
    keys: ["Enter"],
    keycaps: "Enter",
    combat: forwarded("lock_in", "Lock In"),
    replay: null,
  }),
  commandRow({
    id: "one_tick",
    keys: ["n"],
    keycaps: "N",
    skipsTextAndDialogs: true,
    // N is one tick in every client: forwarded from the focused battlefield.
    // In interactive play Python treats it as Space (Manual: Lock In; Policies
    // Only: play one turn), and the page queues and refuses it as it does
    // Space; in scripted play it shows the next frame, as before. In the
    // Replay Viewer it does exactly what +1 and Arrow Right do (owner, 3 Oct
    // 2026).
    combat: forwarded("one_tick", "One tick (Lock In; scripted play: next frame)"),
    replay: local("one_tick", "Forward one tick, the same as +1", "replay_one_tick"),
  }),
  commandRow({
    id: "tick_step",
    keys: ["ArrowLeft", "ArrowRight"],
    keycaps: "Arrow Left / Right",
    shift: "up",
    combat: null,
    replay: local("tick_step", "Back / forward one tick (glides)", "replay_tick_step"),
  }),
  commandRow({
    id: "tick_jump",
    keys: ["ArrowLeft", "ArrowRight"],
    keycaps: "Shift + Arrow Left / Right",
    shift: "down",
    combat: null,
    replay: local("tick_jump", "Back / forward ten ticks (snaps)", "replay_tick_jump"),
  }),
  commandRow({
    id: "arrow_up_down",
    keys: ["ArrowUp", "ArrowDown"],
    keycaps: "Arrow Up / Down",
    combat: null,
    replay: null,
  }),
  commandRow({
    id: "restart",
    keys: ["r"],
    keycaps: "R",
    combat: forwarded("restart", "Restart (Save, Discard, Cancel)"),
    replay: null,
  }),
  commandRow({
    id: "ranges",
    keys: ["g"],
    keycaps: "G",
    skipsTextAndDialogs: true,
    combat: forwarded("ranges", "Ranges"),
    replay: local("ranges", "Ranges", "ranges"),
  }),
  commandRow({
    id: "help",
    keys: ["?"],
    keycaps: "?",
    combat: local("help", "How to Play", "help"),
    replay: local("help", "Help", "help"),
  }),
  commandRow({
    id: "next_champion",
    keys: ["Tab"],
    keycaps: "Tab / Shift+Tab",
    shift: "any",
    // Live play with Manual champions: main.js turns Tab into a focus change
    // to your next champion that can take orders (never a policy's champion).
    // Policies Only: Python walks every champion in sight.
    combat: forwarded(
      "next_champion",
      "Your next / previous champion (Policies Only: every champion in sight)",
    ),
    replay: local(
      "next_champion",
      "Next / previous agent while the field has focus",
      "replay_next_agent",
    ),
  }),
  commandRow({
    id: "escape",
    keys: ["Escape"],
    keycaps: "Escape",
    combat: forwarded("escape", "Clear target, then keys off"),
    replay: local("escape", "Close, clear selection, leave the field", "replay_escape"),
  }),
  commandRow({
    id: "everyone_in_view",
    keys: ["v"],
    keycaps: "V",
    shift: "up",
    skipsTextAndDialogs: true,
    combat: local("everyone_in_view", "Everyone in View", "camera_everyone_in_view"),
    replay: local("everyone_in_view", "Everyone in View", "camera_everyone_in_view"),
  }),
  commandRow({
    id: "battle_cam",
    keys: ["b"],
    keycaps: "B",
    shift: "up",
    skipsTextAndDialogs: true,
    combat: null,
    replay: local("battle_cam", "Battle Cam", "camera_battle_cam"),
  }),
  commandRow({
    id: "frame_explanation",
    keys: ["f"],
    keycaps: "F",
    shift: "up",
    skipsTextAndDialogs: true,
    combat: local(
      "frame_explanation",
      "Frame Explanation (while waiting for orders)",
      "frame_explanation",
    ),
    replay: local(
      "frame_explanation",
      "Frame Explanation (paused; in play it pauses first)",
      "frame_explanation",
    ),
  }),
]);

/**
 * Every local handler name the table uses. registerLocalCommandHandler accepts
 * only these names.
 */
export const LOCAL_COMMAND_HANDLERS = Object.freeze(
  [
    ...new Set(
      COMMAND_SET.flatMap(({ jobs }) =>
        COMMAND_PRODUCTS.map((product) => jobs[product]).filter(
          (entry) => entry !== null && entry.route === "local",
        ),
      ).map((entry) => /** @type {CommandEntry} */ (entry).handler),
    ),
  ].sort(),
);

/** @type {Map<string, LocalCommandHandler>} */
const localHandlers = new Map();

/**
 * Find the row and entry a key event selects in product, or null.
 *
 * Ctrl, Alt and Meta always return null, so browser shortcuts such as Ctrl+F
 * and Ctrl+V stay native. Shift must match the row: "up" rows need Shift up,
 * "down" rows need Shift down, "any" rows (Tab, and keys whose meaning Python
 * reads from the Shift flag) accept both. A single character matches either
 * case. A row whose product entry is null (the key does nothing there) returns
 * null. Pure: no event is cancelled and no handler runs.
 *
 * @param {unknown} product "combat_debugger" or "replay_viewer".
 * @param {CommandKeyEvent} event A KeyboardEvent or a record with the same
 *   fields; missing modifiers count as up.
 * @returns {CommandMatch | null} A new frozen match, or null when the key has no
 *   job in this product.
 * @throws {RangeError} When product is not a known product.
 */
export function resolveCommandKey(product, event) {
  const name = assertProduct(product);
  if (event.ctrlKey || event.altKey || event.metaKey || typeof event.key !== "string") {
    return null;
  }
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  const shift = Boolean(event.shiftKey);
  for (const row of COMMAND_SET) {
    if (!row.keys.includes(key)) continue;
    if ((row.shift === "up" && shift) || (row.shift === "down" && !shift)) continue;
    const entry = row.jobs[name];
    return entry === null ? null : Object.freeze({ row, entry, key });
  }
  return null;
}

/**
 * Return the keys a product forwards to Python from the focused battlefield,
 * in table order. For "combat_debugger" these are the live game keys (W A S D
 * Q E Z C, X, 0 1 2, Space, Enter, N, R, G, Tab and Escape); the arrows, V, B
 * and F are never forwarded.
 *
 * @param {unknown} product "combat_debugger" or "replay_viewer" (which
 *   forwards nothing, so its list is empty).
 * @returns {ReadonlyArray<string>} A new frozen list of KeyboardEvent.key
 *   values, single letters in lowercase.
 * @throws {RangeError} When product is not a known product.
 */
export function forwardedFieldKeys(product) {
  const name = assertProduct(product);
  return Object.freeze(
    COMMAND_SET.flatMap(({ keys, jobs }) => {
      const entry = jobs[name];
      return entry !== null && entry.route === "forwarded" && entry.onField ? keys : [];
    }),
  );
}

/**
 * Return the help rows for one product, in table order: one row per table row
 * that has a job there. How to Play (BattleClient and DevClient) and Help
 * (Replay Viewer) read these, so the help always matches what the keys do.
 *
 * @param {unknown} product "combat_debugger" or "replay_viewer".
 * @returns {ReadonlyArray<Readonly<{id: string, keycaps: string, help: string, job: string}>>}
 *   A new frozen list of new frozen rows.
 * @throws {RangeError} When product is not a known product.
 */
export function commandHelpRows(product) {
  const name = assertProduct(product);
  return Object.freeze(
    COMMAND_SET.flatMap(({ id, keycaps, jobs }) => {
      const entry = jobs[name];
      return entry === null
        ? []
        : [Object.freeze({ id, keycaps, help: entry.help, job: entry.job })];
    }),
  );
}

/**
 * Return the help text for one product: one line per help row, as
 * "Keycaps: help", joined with newlines. For plain-text places such as a
 * title attribute or a log.
 *
 * @param {unknown} product "combat_debugger" or "replay_viewer".
 * @returns {string} The lines; never empty for a known product.
 * @throws {RangeError} When product is not a known product.
 */
export function commandHelpText(product) {
  return commandHelpRows(product)
    .map(({ keycaps, help }) => `${keycaps}: ${help}`)
    .join("\n");
}

/**
 * Register the one handler for a local job. Each handler name has exactly one
 * owner: the camera registers camera_everyone_in_view and camera_battle_cam,
 * Frame Explanation registers frame_explanation, and the Replay Viewer
 * transport registers its replay_* handlers.
 *
 * The handler is called as handler(call) with call = {product, job, key,
 * shiftKey, repeat, event, context}. It may return false to say "not now"
 * (for example F while the field is not waiting for orders); any other result
 * counts as handled. It owns its own Agent POV limits.
 *
 * @param {unknown} handlerName One of LOCAL_COMMAND_HANDLERS.
 * @param {unknown} handler The function to call.
 * @returns {() => void} A function that removes this registration (only if
 *   it is still the registered one); safe to call twice.
 * @throws {RangeError} When handlerName is not a local handler of the table.
 * @throws {TypeError} When handler is not a function, or the name already has
 *   a handler.
 */
export function registerLocalCommandHandler(handlerName, handler) {
  if (
    typeof handlerName !== "string" ||
    !LOCAL_COMMAND_HANDLERS.includes(handlerName)
  ) {
    throw new RangeError(`Unknown local command handler ${String(handlerName)}.`);
  }
  if (typeof handler !== "function") {
    throw new TypeError("A local command handler must be a function.");
  }
  if (localHandlers.has(handlerName)) {
    throw new TypeError(`Local command handler ${handlerName} is already registered.`);
  }
  const owned = /** @type {LocalCommandHandler} */ (handler);
  localHandlers.set(handlerName, owned);
  return () => {
    if (localHandlers.get(handlerName) === owned) {
      localHandlers.delete(handlerName);
    }
  };
}

/**
 * Run the local job a key event selects in product, if one is registered.
 *
 * Steps: resolve the key (resolveCommandKey); stop unless its route is "local";
 * for the rows marked skipsTextAndDialogs (V, B, F, G and N) stop when the
 * event's target is a text field or sits in an
 * open dialog, or a modal dialog is open; stop when no handler is registered;
 * otherwise call the handler. The caller decides whether to cancel the event:
 * when this returns true it should call preventDefault and stopPropagation, so
 * no second listener runs the same job.
 *
 * @param {unknown} product "combat_debugger" or "replay_viewer".
 * @param {CommandKeyEvent} event The keydown event (its target is read for the
 *   text-field and dialog check).
 * @param {Readonly<Record<string, unknown>>} [context] Extra facts the caller
 *   passes to the handler, for example {interactive: false}; default empty.
 * @returns {boolean} True when a handler ran and did not return false.
 * @throws {RangeError} When product is not a known product.
 * Errors thrown by the handler propagate to the caller.
 */
export function runLocalCommand(product, event, context = {}) {
  const name = /** @type {CommandProduct} */ (assertProduct(product));
  const match = resolveCommandKey(name, event);
  if (match === null || match.entry.route !== "local") {
    return false;
  }
  if (match.row.skipsTextAndDialogs && isTextOrDialogTarget(event.target)) {
    return false;
  }
  const handler = localHandlers.get(match.entry.handler);
  if (handler === undefined) {
    return false;
  }
  const result = handler(
    Object.freeze({
      product: name,
      job: match.entry.job,
      key: match.key,
      shiftKey: Boolean(event.shiftKey),
      repeat: Boolean(event.repeat),
      event,
      context: Object.freeze({ ...context }),
    }),
  );
  return result !== false;
}

/**
 * Return true when target is a text field (input, textarea, select, an
 * editable element, or a textbox, combobox or spinbutton role), sits inside an
 * open dialog, or its document has an open modal dialog. Non-element targets
 * (no closest method) return false. Reads the DOM only.
 *
 * @param {unknown} target
 * @returns {boolean}
 */
export function isTextOrDialogTarget(target) {
  const element = /** @type {any} */ (target);
  if (
    element === null ||
    typeof element !== "object" ||
    typeof element.closest !== "function"
  ) {
    return false;
  }
  if (
    element.closest(
      'input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="textbox"], [role="combobox"], [role="spinbutton"], dialog[open]',
    )
  ) {
    return true;
  }
  const doc = element.ownerDocument;
  if (doc && typeof doc.querySelector === "function") {
    try {
      return doc.querySelector("dialog:modal") !== null;
    } catch {
      return false;
    }
  }
  return false;
}

/**
 * Build one frozen table row. Keys are copied and frozen; shift defaults to
 * "any" for rows whose meaning does not depend on Shift being up or down, and
 * skipsTextAndDialogs defaults to false.
 *
 * @param {{
 *   id: string,
 *   keys: string[],
 *   keycaps: string,
 *   shift?: CommandShift,
 *   skipsTextAndDialogs?: boolean,
 *   combat: CommandEntry | null,
 *   replay: CommandEntry | null,
 * }} spec
 * @returns {CommandRow}
 */
function commandRow({
  id,
  keys,
  keycaps,
  shift = "any",
  skipsTextAndDialogs = false,
  combat,
  replay,
}) {
  return Object.freeze({
    id,
    keys: Object.freeze([...keys]),
    keycaps,
    shift,
    skipsTextAndDialogs,
    jobs: Object.freeze({ combat_debugger: combat, replay_viewer: replay }),
  });
}

/**
 * Build a frozen entry for a key Python handles.
 *
 * @param {string} job Shared job ID.
 * @param {string} help Plain help line.
 * @param {boolean} [onField] True (default) when the key is forwarded from the
 *   focused battlefield; false when only a button sends it.
 * @returns {CommandEntry}
 */
function forwarded(job, help, onField = true) {
  return Object.freeze({
    job,
    help,
    route: "forwarded",
    handler: PYTHON_HANDLER,
    onField,
  });
}

/**
 * Build a frozen entry for a key the browser handles itself.
 *
 * @param {string} job Shared job ID.
 * @param {string} help Plain help line.
 * @param {string} handler The local handler name its owner registers.
 * @returns {CommandEntry}
 */
function local(job, help, handler) {
  return Object.freeze({ job, help, route: "local", handler, onField: true });
}

/**
 * Return product when it is a known product name; otherwise throw RangeError.
 *
 * @param {unknown} product
 * @returns {CommandProduct}
 */
function assertProduct(product) {
  if (product !== "combat_debugger" && product !== "replay_viewer") {
    throw new RangeError(`Unknown command product ${String(product)}.`);
  }
  return product;
}
