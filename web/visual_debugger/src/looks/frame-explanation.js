/**
 * @file Frame Explanation (key F): numbered marks on the paused tick, in the
 * arena's `explain` layer, and a matching numbered list of what happened.
 *
 * Owns three things:
 * - explainTick(input): the items of one tick, in tick order: blows and heals
 *   (one item per recipient, with its recorded total and life), each new status
 *   as its own item (in the Champion Card row's words, statusLineV1 from
 *   `semantic-vocabulary.js`: the source named by class only, in both views),
 *   broken ice, falls with their score change (x2 only from the event's
 *   home_fall flag), the score after the falls, returns and Ultimates ready
 *   again.
 * - The look contract (records, build, update, tooltip, FOOTPRINT, BUDGET,
 *   PAINT_PARTS): numbered discs beside the champions the items name.
 * - createFrameExplanation(options): the controller the page mounts. It
 *   registers the F key (`frame_explanation` in `command-set.js`), the
 *   Frame Explanation button, the callouts and the list.
 *
 * Callers: `main.js` creates the controller once and calls refresh() after each
 * render (frame, camera, view or switch change); `scene.js` owns the `explain`
 * layer the callouts go in.
 *
 * Agent POV boundary: this look reads no frame. Like every look it may not
 * import the presentation adapter (tests/pov-boundary.test.js), so the caller
 * hands it a FrameExplanationInput built from the shown audience's data only:
 * its own events (Oracle View: latest_events; Agent POV: visual_events, from
 * `authorizedPresentationIncomingRows`), its own arena scene agents
 * (`authorizedPresentationSceneView`) and the public scoreboard
 * (`authorizedPresentationMatchBeatFacts`), never the researcher view. A
 * champion missing from that scene is never named, placed or described: in
 * an event line a recipient reads "a champion out of sight" and a source
 * reads "a hidden source", with no class, ID, life or position, and gets no
 * callout. A status item names its source by class only ("From a Warrior."),
 * which the status itself already tells. Nothing is
 * cached, so Oracle -> Agent POV -> Oracle gives exactly a fresh answer.
 */
import { canonicalAgentIdentity, teamLabelFromId } from "../agent-identity.js";
import { classAbilityNameV1 } from "../class-documentation.js";
import { registerLocalCommandHandler } from "../command-set.js";
import { formatDisplayNumber } from "../display.js";
import { statusLineV1 } from "../semantic-vocabulary.js";
import { symbolById } from "../symbols.js";
import { createSymbolDescriptor } from "../tooltip.js";
import { statusTokenIdFromCatalogId } from "../vocabulary.js";
import { EMPTY_BUDGET, finiteNumber, isRecord, svgNode } from "./look-support.js";

/**
 * @typedef {import("./look-support.js").LookTransform} LookTransform
 * @typedef {import("./look-support.js").LookRecord} LookRecord
 */

/**
 * One numbered item of a Frame Explanation.
 *
 * - n: its number, 1 upward, in tick order.
 * - kind: "blow" (damage, or damage and healing), "heal", "touch" (a Basic or
 *   Ultimate reached a champion and changed no life), "ability" (an ability used
 *   with no recipient this tick), "status", "broken_ice", "fall", "score",
 *   "return" or "ultimate_ready".
 * - text: the plain sentence shown in the list.
 * - anchorKey: the presentation key of the champion the item is about, when that
 *   champion is in the audience's scene (it gets a callout), else null.
 *
 * @typedef {Readonly<{
 *   n: number,
 *   kind: string,
 *   text: string,
 *   anchorKey: string | null,
 * }>} FrameExplanationItem
 */

/**
 * One tick's explanation: audience "researcher" (Oracle View) or "agent_pov";
 * tick, the simulator step the tick ends on, or null when the frame has none;
 * items in tick order.
 *
 * @typedef {Readonly<{
 *   audience: "researcher" | "agent_pov",
 *   tick: number | null,
 *   items: ReadonlyArray<FrameExplanationItem>,
 * }>} FrameExplanation
 */

/**
 * The plain data of one tick, built by the caller from the shown audience only.
 *
 * - audience: "researcher" (Oracle View) or "agent_pov".
 * - agents: that audience's arena scene agents (rows with presentation_key,
 *   public_agent_id, display_agent_id, class_id, team_id, position in world
 *   units, radius, maximum_health and statuses).
 * - events: that audience's own event payloads (Oracle View: latest_events;
 *   Agent POV: visual_events), in any order (sorted here by ordinal).
 * - score: the public scoreboard {changed, after: [Team A, Team B]}, or null.
 * - tick: the simulator step the tick ends on, or null.
 *
 * @typedef {Readonly<{
 *   audience: "researcher" | "agent_pov",
 *   agents: ReadonlyArray<unknown>,
 *   events: ReadonlyArray<unknown>,
 *   score: Readonly<{changed: boolean, after: ReadonlyArray<number>}> | null,
 *   tick: number | null,
 * }>} FrameExplanationInput
 */

/** Disc radius of a callout, in CSS pixels (an 18 px hit box). */
const CALLOUT_RADIUS = 9;
/** Space between two discs on one champion, in CSS pixels. */
const CALLOUT_STEP = 20;

/** One callout disc: an 18 x 18 px box beside the champion. */
export const FOOTPRINT = Object.freeze({
  callout: Object.freeze({ width: CALLOUT_RADIUS * 2, height: CALLOUT_RADIUS * 2 }),
});

/** Three nodes per callout (group, disc, number). No animations. */
export const BUDGET = Object.freeze({ ...EMPTY_BUDGET, nodesPerRecord: 3 });

/**
 * No switch: Frame Explanation shows only when asked (F or its button), so it
 * has no visual filter (owner, 1 October 2026).
 *
 * @type {ReadonlyArray<import("./look-support.js").PaintPartRegistration>}
 */
export const PAINT_PARTS = Object.freeze([]);

/** Words for a recipient the audience cannot see. */
const UNSEEN_RECIPIENT = "a champion out of sight";
/** Words for a source the audience cannot see (the adapter's words). */
const HIDDEN_SOURCE = "a hidden source";

/**
 * Explain one tick from the shown audience's plain data.
 *
 * input is a FrameExplanationInput; null, or an audience other than
 * "researcher" and "agent_pov", returns null. The caller must pass only data
 * its audience may see: anything named here is shown.
 *
 * Items follow the events' ordinal: each recipient's blows and heals at its
 * health resolution, then the rest as they come. An ability whose recipient had
 * no health resolution this tick gets its own "touch" (it reached someone) or
 * "ability" (no recipient) item. A recipient whose recorded life did not change
 * and that no ability reached is left out. Numbers are the recorded totals:
 * Oracle View uses total_effective_damage and total_effective_healing; Agent POV
 * has only the life change the team saw (realized_net_health_change). Life is
 * "start → end of full" from the event and the scene. No per-blow damage is
 * worked out in the browser. A fall gives 2 points when its home_fall flag is
 * true, else 1, to the other team. The score line (when score.changed) follows
 * the last fall listed, or ends the list when no fall was seen. Pure: no DOM,
 * storage or timers.
 *
 * @param {FrameExplanationInput | null | undefined} input
 * @returns {FrameExplanation | null}
 */
export function explainTick(input) {
  if (
    !isRecord(input) ||
    (input.audience !== "researcher" && input.audience !== "agent_pov")
  ) {
    return null;
  }
  const audience = input.audience;
  /** @type {Map<string, Record<string, any>>} */
  const agents = new Map();
  for (const agent of Array.isArray(input.agents) ? input.agents : []) {
    if (isRecord(agent) && typeof agent.presentation_key === "string") {
      agents.set(agent.presentation_key, agent);
    }
  }
  const rows = (Array.isArray(input.events) ? input.events : [])
    .filter(isRecord)
    .filter((payload) => typeof payload.event_kind === "string")
    .map((payload) => ({ kind: String(payload.event_kind), payload }))
    .sort((left, right) => ordinal(left.payload) - ordinal(right.payload));

  /** @type {Map<string, Array<Record<string, any>>>} */
  const activationsByRecipient = new Map();
  /** @type {Set<string>} */
  const resolvedRecipients = new Set();
  for (const { kind, payload } of rows) {
    const key = anchorKey(payload.recipient_anchor);
    if (kind === "ability_activated" && key !== null) {
      activationsByRecipient.set(key, [
        ...(activationsByRecipient.get(key) ?? []),
        payload,
      ]);
    } else if (kind === "recipient_health_resolution" && key !== null) {
      resolvedRecipients.add(key);
    }
  }

  /** @type {Array<{kind: string, text: string, anchorKey: string | null}>} */
  const drafts = [];
  let fallSeen = false;
  for (const { kind, payload } of rows) {
    if (kind === "ability_activated") {
      const key = anchorKey(payload.recipient_anchor);
      if (key !== null && resolvedRecipients.has(key)) {
        continue;
      }
      const user = sourcePhrase(
        payload.source_anchor,
        payload.ability_component,
        agents,
      );
      drafts.push(
        key === null
          ? { kind: "ability", text: `${capitalize(user)} was used.`, anchorKey: null }
          : {
              kind: "touch",
              text: `${capitalize(user)} reached ${nameOf(payload.recipient_anchor, agents, UNSEEN_RECIPIENT)}.`,
              anchorKey: seenKey(payload.recipient_anchor, agents),
            },
      );
    } else if (kind === "recipient_health_resolution") {
      const draft = healthDraft(payload, audience, agents, activationsByRecipient);
      if (draft !== null) {
        drafts.push(draft);
      }
    } else if (kind === "status_applied") {
      drafts.push(statusDraft(payload, agents));
    } else if (kind === "status_broken_by_damage") {
      drafts.push({
        kind: "broken_ice",
        text: `The ice around ${nameOf(payload.recipient_anchor, agents, UNSEEN_RECIPIENT)} broke: ${symbolTitle("stun_broken")}.`,
        anchorKey: seenKey(payload.recipient_anchor, agents),
      });
    } else if (kind === "agent_died") {
      fallSeen = true;
      drafts.push(fallDraft(payload, agents));
    } else if (kind === "agent_respawned") {
      drafts.push({
        kind: "return",
        text: `${capitalize(nameOf(payload.agent_anchor, agents, UNSEEN_RECIPIENT))} came back on a spawn pad under ${symbolTitle("spawn_shield")}.`,
        anchorKey: seenKey(payload.agent_anchor, agents),
      });
    } else if (kind === "cooldown_ready") {
      const owner = agents.get(anchorKey(payload.agent_anchor) ?? "");
      const ultimate = owner ? classAbilityNameV1(owner.class_id, "ultimate") : null;
      drafts.push({
        kind: "ultimate_ready",
        text:
          owner && ultimate
            ? `${nameOf(payload.agent_anchor, agents, UNSEEN_RECIPIENT)}'s ${ultimate} is ready again.`
            : `${capitalize(nameOf(payload.agent_anchor, agents, UNSEEN_RECIPIENT))}'s Ultimate is ready again.`,
        anchorKey: seenKey(payload.agent_anchor, agents),
      });
    }
  }

  if (input.score?.changed) {
    const scoreDraft = {
      kind: "score",
      text: `Score: Team A ${input.score.after[0]}, Team B ${input.score.after[1]}.`,
      anchorKey: null,
    };
    // The score line follows the last fall; with no seen fall it comes last.
    let lastFall = -1;
    if (fallSeen) {
      drafts.forEach((draft, index) => {
        if (draft.kind === "fall") lastFall = index;
      });
    }
    drafts.splice(lastFall === -1 ? drafts.length : lastFall + 1, 0, scoreDraft);
  }

  return Object.freeze({
    audience,
    tick: Number.isInteger(input.tick) ? input.tick : null,
    items: Object.freeze(
      drafts.map((draft, index) => Object.freeze({ n: index + 1, ...draft })),
    ),
  });
}

/**
 * Return one callout record per item whose champion is in the audience's scene.
 *
 * input.tick is the FrameExplanationInput of the shown tick and
 * input.transform this render's projection (it includes the camera frame). Items about the same champion sit
 * in a row of discs above its right side, CALLOUT_STEP px apart; the row
 * flips to the left side or below the body when it would leave
 * transform.viewportBounds. Each frozen
 * record is {kind: "frame_explanation_callout", n, itemKind, text, anchorKey,
 * x, y} with x and y the disc centre in CSS pixels. No transform, no tick or no
 * items gives []. Pure data: no DOM.
 *
 * Agent POV: positions come only from input.tick.agents (the POV scene), so an
 * unseen champion gets no callout.
 *
 * @param {{tick?: FrameExplanationInput | null, transform?: LookTransform | null}} input
 * @returns {ReadonlyArray<LookRecord>}
 */
export function records(input) {
  const explanation = explainTick(input.tick);
  const transform = input.transform;
  if (explanation === null || !transform || !input.tick) {
    return Object.freeze([]);
  }
  /** @type {Map<string, Record<string, any>>} */
  const agents = new Map();
  for (const agent of Array.isArray(input.tick.agents) ? input.tick.agents : []) {
    if (isRecord(agent) && typeof agent.presentation_key === "string") {
      agents.set(agent.presentation_key, agent);
    }
  }
  /** @type {Map<string, number>} */
  const perChampion = new Map();
  /** @type {LookRecord[]} */
  const out = [];
  for (const item of explanation.items) {
    const agent = item.anchorKey === null ? undefined : agents.get(item.anchorKey);
    if (agent === undefined || !Array.isArray(agent.position)) {
      continue;
    }
    const index = perChampion.get(item.anchorKey ?? "") ?? 0;
    perChampion.set(item.anchorKey ?? "", index + 1);
    const center = transform.worldToScreen([
      finiteNumber(agent.position[0]),
      finiteNumber(agent.position[1]),
    ]);
    const radius = transform.worldLengthToScreen(finiteNumber(agent.radius, 0.5));
    out.push(
      Object.freeze({
        kind: "frame_explanation_callout",
        n: item.n,
        itemKind: item.kind,
        text: item.text,
        anchorKey: item.anchorKey,
        ...calloutPoint(center, radius, index, transform.viewportBounds),
      }),
    );
  }
  return Object.freeze(out);
}

/**
 * Build one export-safe group per callout record: a dark disc with a gold edge
 * and its number in ivory, all flat fills.
 *
 * Each group is `g.frame-explanation-callout` with data-item = n and
 * data-item-kind; nodes get their place from update. Return the groups in
 * record order. No tooltip, storage or timer; the controller registers tips.
 *
 * @param {ReadonlyArray<LookRecord>} records
 * @param {Document} doc
 * @returns {SVGElement[]}
 */
export function build(records, doc) {
  const nodes = records.map((record) => {
    const group = svgNode(doc, "g", {
      class: "frame-explanation-callout",
      "data-item": String(record.n),
      "data-item-kind": record.itemKind,
    });
    group.append(
      svgNode(doc, "circle", {
        class: "frame-explanation-callout__disc",
        r: CALLOUT_RADIUS,
        fill: "#14161F",
        stroke: "#E9C46A",
        "stroke-width": 1.5,
      }),
      svgNode(doc, "text", {
        class: "frame-explanation-callout__number",
        "text-anchor": "middle",
        "dominant-baseline": "central",
        fill: "#F4EBD6",
        "font-size": 11,
        "font-weight": 700,
      }),
    );
    const number = /** @type {SVGElement} */ (group.lastChild);
    number.textContent = String(record.n);
    return group;
  });
  update(nodes, records);
  return nodes;
}

/**
 * Move callout groups to their records (after a resize or camera change).
 * nodes and records pair up by index; extra entries on either side are left
 * alone. Writes a translate transform on each group. Returns nothing.
 *
 * @param {ReadonlyArray<Element>} nodes
 * @param {ReadonlyArray<LookRecord>} records
 */
export function update(nodes, records) {
  const count = Math.min(nodes.length, records.length);
  for (let index = 0; index < count; index += 1) {
    const record = records[index];
    nodes[index].setAttribute(
      "transform",
      `translate(${formatPixel(record.x)} ${formatPixel(record.y)})`,
    );
  }
}

/**
 * Return the Symbol Tip of a callout: the Frame Explanation symbol with the
 * item's sentence as its line. Any other record returns null.
 *
 * @param {LookRecord} record
 * @returns {import("../tooltip.js").TooltipDescriptor | null}
 */
export function tooltip(record) {
  if (record.kind !== "frame_explanation_callout") {
    return null;
  }
  return createSymbolDescriptor({
    symbol: "frame_explanation",
    id: `frame-explanation:${record.n}`,
    line: `${record.n}. ${record.text}`,
  });
}

/**
 * Fill host with the numbered list of explanation.
 *
 * host's children are replaced by a caption ("Tick N · Oracle View" or "Tick N ·
 * Agent POV · Only what your team saw") and an `ol.frame-explanation__list`
 * whose items carry data-item = n, a number badge and the sentence; with no
 * items, a short note says nothing could be named. explanation null clears host.
 * Text is set with textContent only. Returns nothing.
 *
 * @param {HTMLElement} host
 * @param {FrameExplanation | null} explanation
 */
export function renderFrameExplanationList(host, explanation) {
  const doc = host.ownerDocument;
  if (explanation === null) {
    host.replaceChildren();
    return;
  }
  const caption = doc.createElement("p");
  caption.className = "frame-explanation__caption";
  caption.textContent = [
    explanation.tick === null ? null : `Tick ${explanation.tick}`,
    explanation.audience === "agent_pov"
      ? "Agent POV · Only what your team saw"
      : "Oracle View",
  ]
    .filter(Boolean)
    .join(" · ");
  if (explanation.items.length === 0) {
    const empty = doc.createElement("p");
    empty.className = "frame-explanation__empty";
    empty.textContent =
      "Nothing happened on this tick that Frame Explanation can name.";
    host.replaceChildren(caption, empty);
    return;
  }
  const list = doc.createElement("ol");
  list.className = "frame-explanation__list";
  for (const item of explanation.items) {
    const row = doc.createElement("li");
    row.className = "frame-explanation__item";
    row.setAttribute("data-item", String(item.n));
    row.setAttribute("data-item-kind", item.kind);
    const number = doc.createElement("span");
    number.className = "frame-explanation__number";
    number.setAttribute("aria-hidden", "true");
    number.textContent = String(item.n);
    const text = doc.createElement("span");
    text.className = "frame-explanation__text";
    text.textContent = item.text;
    row.append(number, text);
    list.append(row);
  }
  host.replaceChildren(caption, list);
}

/**
 * @typedef {{
 *   product: "combat_debugger" | "replay_viewer",
 *   getTickInput: () => FrameExplanationInput | null,
 *   getTransform: () => LookTransform | null,
 *   getLayer: () => Element | null,
 *   listHost?: HTMLElement | null,
 *   button?: HTMLElement | null,
 *   isPaused?: () => boolean,
 *   pause?: () => void,
 *   canExplain?: () => boolean,
 *   onOpenChange?: (open: boolean) => void,
 *   registerSymbolTip?: (element: Element, record: LookRecord) => void,
 * }} FrameExplanationOptions
 */

/**
 * @typedef {Readonly<{
 *   open: () => boolean,
 *   close: () => void,
 *   toggle: () => boolean,
 *   isOpen: () => boolean,
 *   refresh: () => void,
 *   destroy: () => void,
 * }>} FrameExplanationController
 */

/**
 * Create the Frame Explanation controller for one page and register the F key.
 *
 * options:
 * - product: "combat_debugger" (BattleClient and DevClient) or "replay_viewer".
 * - getTickInput(): the FrameExplanationInput of the shown tick, built from
 *   the shown audience's data only (see the file header), or null.
 * - getTransform(): this render's projection (with the camera), or null.
 * - getLayer(): the arena's `explain` layer (`scene.js`), or null.
 * - listHost: where the numbered list goes (for example the Replay Viewer's
 *   Frame tab); default none. Its hidden flag follows open and closed.
 * - button: the Frame Explanation button; it is shown, toggles on click and
 *   carries aria-pressed. Default none.
 * - isPaused() and pause() (Replay Viewer): F during play pauses first; a
 *   refresh while playing closes it ("Play hides it"). Defaults: always paused,
 *   no pause.
 * - canExplain() (BattleClient and DevClient): whether the field is waiting for
 *   orders (no command in flight). F or the button pressed while it is false
 *   does not open at once: the request is kept for the tick shown when it was
 *   made (getTickInput().tick), and the first refresh() after canExplain()
 *   turns true opens it, if that tick is still shown (a new tick drops it).
 *   Pressing again while it waits cancels it; close() drops it. So F right
 *   after a click on the arena (whose command is briefly in flight) opens
 *   Frame Explanation, just as the button does. Default always true.
 * - onOpenChange(open): called after each open or close, for example to pick
 *   the Frame tab. Default none.
 * - registerSymbolTip(element, record): registers a callout's tip; default
 *   none (callouts then have no tip).
 *
 * Side effects: registers the `frame_explanation` local command handler (one
 * per page; a second controller throws TypeError until the first is
 * destroyed), adds click and pointerover listeners to button and listHost,
 * and draws into the layer only while open. Hovering a list item or a callout
 * lights both (data-lit="true"). Call refresh() after every render so the
 * marks follow each step, seek, camera and view change; it rebuilds from the
 * current frame with no memory, so Oracle -> Agent POV -> Oracle is a fresh
 * answer.
 *
 * @param {FrameExplanationOptions} options
 * @returns {FrameExplanationController}
 * @throws {TypeError} When options or a required getter is missing, or F
 *   already has a handler.
 * @throws {RangeError} When product is not a known product.
 */
export function createFrameExplanation(options) {
  if (!isRecord(options)) {
    throw new TypeError("Frame Explanation options must be an object.");
  }
  const { product, getTickInput, getTransform, getLayer } = options;
  if (product !== "combat_debugger" && product !== "replay_viewer") {
    throw new RangeError(`Unknown product ${String(product)}.`);
  }
  if (
    typeof getTickInput !== "function" ||
    typeof getTransform !== "function" ||
    typeof getLayer !== "function"
  ) {
    throw new TypeError(
      "Frame Explanation needs getTickInput, getTransform and getLayer.",
    );
  }
  const listHost = options.listHost ?? null;
  const button = options.button ?? null;
  const isPaused = options.isPaused ?? (() => true);
  const canExplain = options.canExplain ?? (() => true);
  let opened = false;
  // A combat request made while canExplain() was false: the tick it was made
  // on (null when the input has no tick), or undefined for no request.
  /** @type {number | null | undefined} */
  let requestedTick;
  /** @type {Element[]} */
  let drawn = [];

  /**
   * Return the shown tick from getTickInput(), or null when there is none.
   *
   * @returns {number | null}
   */
  function shownTick() {
    const tick = getTickInput()?.tick;
    return typeof tick === "number" ? tick : null;
  }

  /** Remove the callouts this controller drew. */
  function clearCallouts() {
    for (const node of drawn) {
      node.remove();
    }
    drawn = [];
  }

  /**
   * Light (or unlight) the list row and callouts numbered n.
   *
   * @param {string | null} n The data-item value, or null to unlight all.
   */
  function light(n) {
    for (const node of [
      ...drawn,
      ...(listHost ? [...listHost.querySelectorAll("[data-item]")] : []),
    ]) {
      if (n !== null && node.getAttribute("data-item") === n) {
        node.setAttribute("data-lit", "true");
      } else {
        node.removeAttribute("data-lit");
      }
    }
  }

  /** Draw the callouts and the list for the current frame. */
  function draw() {
    clearCallouts();
    const tick = getTickInput();
    const explanation = explainTick(tick);
    const layer = getLayer();
    const transform = getTransform();
    if (layer !== null && transform !== null) {
      const callouts = records({ tick, transform });
      drawn = build(callouts, layer.ownerDocument);
      drawn.forEach((node, index) => {
        options.registerSymbolTip?.(node, callouts[index]);
        layer.append(node);
      });
    }
    if (listHost !== null) {
      renderFrameExplanationList(listHost, explanation);
    }
  }

  /**
   * Set the open state and tell the button, list host and onOpenChange.
   *
   * @param {boolean} next
   */
  function setOpen(next) {
    opened = next;
    button?.setAttribute("aria-pressed", next ? "true" : "false");
    if (listHost !== null) {
      listHost.hidden = !next;
    }
    options.onOpenChange?.(next);
  }

  /**
   * Open Frame Explanation. Replay Viewer: pauses first when playing.
   * BattleClient and DevClient: refused while the field is not waiting for
   * orders. Return whether it is open afterwards.
   *
   * @returns {boolean}
   */
  function open() {
    if (product === "combat_debugger" && !canExplain()) {
      return opened;
    }
    if (product === "replay_viewer" && !isPaused()) {
      options.pause?.();
    }
    draw();
    if (!opened) {
      setOpen(true);
    }
    return true;
  }

  /**
   * Close Frame Explanation: remove the callouts, empty the list and drop a
   * waiting request.
   */
  function close() {
    requestedTick = undefined;
    clearCallouts();
    if (listHost !== null) {
      listHost.replaceChildren();
    }
    if (opened) {
      setOpen(false);
    }
  }

  /**
   * Toggle: close when open, otherwise open. Return whether it is open.
   *
   * @returns {boolean}
   */
  function toggle() {
    if (opened) {
      close();
      return false;
    }
    return open();
  }

  /**
   * Redraw from the current frame when open. Replay Viewer: closes when the
   * replay is playing (Play hides it). When closed, opens a waiting combat
   * request (press) once canExplain() is true and the tick it was made on is
   * still shown, and drops it when the tick has changed; otherwise does
   * nothing.
   */
  function refresh() {
    if (!opened) {
      if (
        requestedTick !== undefined &&
        product === "combat_debugger" &&
        canExplain()
      ) {
        const sameTick = shownTick() === requestedTick;
        requestedTick = undefined;
        if (sameTick) open();
      }
      return;
    }
    if (product === "replay_viewer" && !isPaused()) {
      close();
      return;
    }
    if (product === "combat_debugger" && !canExplain()) {
      close();
      return;
    }
    draw();
  }

  /**
   * Answer F or the button: close when open; open when the field waits for
   * orders; otherwise (combat, a command in flight) keep a request for the
   * shown tick, or cancel the one already waiting (see canExplain in
   * createFrameExplanation). Returns true: the press is always handled.
   *
   * @returns {boolean}
   */
  function press() {
    if (opened) {
      close();
      return true;
    }
    if (product === "combat_debugger" && !canExplain()) {
      requestedTick = requestedTick === undefined ? shownTick() : undefined;
      return true;
    }
    requestedTick = undefined;
    open();
    return true;
  }

  const unregister = registerLocalCommandHandler("frame_explanation", press);

  /** Answer the button exactly as F (press). */
  function onButtonClick() {
    press();
  }

  /**
   * Light the row or callout under the pointer.
   *
   * @param {Event} event
   */
  function onPointerOver(event) {
    const target = /** @type {Element | null} */ (event.target);
    const owner =
      target && typeof target.closest === "function"
        ? target.closest("[data-item]")
        : null;
    light(owner === null ? null : owner.getAttribute("data-item"));
  }

  if (button !== null) {
    button.hidden = false;
    button.setAttribute("aria-pressed", "false");
    button.setAttribute("aria-keyshortcuts", "F");
    button.addEventListener("click", onButtonClick);
  }
  listHost?.addEventListener("pointerover", onPointerOver);
  const layerAtStart = getLayer();
  layerAtStart?.addEventListener("pointerover", onPointerOver);

  /**
   * Close, unregister F and remove the listeners. Safe to call twice.
   */
  function destroy() {
    close();
    unregister();
    button?.removeEventListener("click", onButtonClick);
    listHost?.removeEventListener("pointerover", onPointerOver);
    layerAtStart?.removeEventListener("pointerover", onPointerOver);
  }

  return Object.freeze({
    open,
    close,
    toggle,
    isOpen: () => opened,
    refresh,
    destroy,
  });
}

/**
 * Return a row's ordinal for sorting; a missing ordinal sorts last.
 *
 * @param {Readonly<Record<string, any>>} payload
 * @returns {number}
 */
function ordinal(payload) {
  return Number.isInteger(payload.ordinal) ? payload.ordinal : Number.MAX_SAFE_INTEGER;
}

/**
 * Return an anchor's presentation key, or null.
 *
 * @param {unknown} anchor
 * @returns {string | null}
 */
function anchorKey(anchor) {
  return isRecord(anchor) && typeof anchor.presentation_key === "string"
    ? anchor.presentation_key
    : null;
}

/**
 * Return the anchor's key when that champion is in the scene, else null.
 *
 * @param {unknown} anchor
 * @param {Map<string, Record<string, any>>} agents
 * @returns {string | null}
 */
function seenKey(anchor, agents) {
  const key = anchorKey(anchor);
  return key !== null && agents.has(key) ? key : null;
}

/**
 * Name the champion at anchor ("Mage ID 5") when it is in the audience's scene;
 * otherwise return unseen.
 *
 * @param {unknown} anchor
 * @param {Map<string, Record<string, any>>} agents
 * @param {string} unseen
 * @returns {string}
 */
function nameOf(anchor, agents, unseen) {
  const agent = agents.get(anchorKey(anchor) ?? "");
  return agent === undefined ? unseen : canonicalAgentIdentity(agent).sourceName;
}

/**
 * Say who used which ability: "Hunter ID 7's Serrated Arrows", or "a hidden
 * source" when the source is not in the scene.
 *
 * @param {unknown} anchor
 * @param {unknown} component "basic" or "ultimate".
 * @param {Map<string, Record<string, any>>} agents
 * @returns {string}
 */
function sourcePhrase(anchor, component, agents) {
  const agent = agents.get(anchorKey(anchor) ?? "");
  if (agent === undefined) {
    return HIDDEN_SOURCE;
  }
  const ability =
    component === "basic" || component === "ultimate"
      ? classAbilityNameV1(agent.class_id, component)
      : null;
  const name = canonicalAgentIdentity(agent).sourceName;
  return ability === null ? name : `${name}'s ${ability}`;
}

/**
 * Build the item of one recipient's health resolution, or null when its life
 * did not change and no ability reached it.
 *
 * @param {Readonly<Record<string, any>>} payload
 * @param {"researcher" | "agent_pov"} audience
 * @param {Map<string, Record<string, any>>} agents
 * @param {Map<string, Array<Record<string, any>>>} activationsByRecipient
 * @returns {{kind: string, text: string, anchorKey: string | null} | null}
 */
function healthDraft(payload, audience, agents, activationsByRecipient) {
  const key = anchorKey(payload.recipient_anchor);
  const activations = activationsByRecipient.get(key ?? "") ?? [];
  let damage = 0;
  let healing = 0;
  if (audience === "researcher") {
    damage = numberOr(payload.total_effective_damage, 0);
    healing = numberOr(payload.total_effective_healing, 0);
  } else {
    const net = numberOr(payload.realized_net_health_change, 0);
    damage = net < 0 ? -net : 0;
    healing = net > 0 ? net : 0;
  }
  if (damage === 0 && healing === 0 && activations.length === 0) {
    return null;
  }
  const recipient = nameOf(payload.recipient_anchor, agents, UNSEEN_RECIPIENT);
  const sources = [
    ...new Set(
      activations.map((row) =>
        sourcePhrase(row.source_anchor, row.ability_component, agents),
      ),
    ),
  ];
  const kind = damage > 0 ? "blow" : healing > 0 ? "heal" : "touch";
  const verb = damage > 0 ? "hit" : healing > 0 ? "healed" : "reached";
  const subject = sources.length === 0 ? null : capitalize(joinWords(sources));
  const amounts = [
    damage > 0 ? `${formatDisplayNumber(damage)} damage` : null,
    healing > 0 ? `${formatDisplayNumber(healing)} healing` : null,
  ].filter(Boolean);
  const amountText =
    amounts.length === 0 ? "no change in life" : `${amounts.join(" and ")} this tick`;
  const lead =
    subject === null
      ? `${capitalize(recipient)} took ${amountText}.`
      : `${subject} ${verb} ${recipient}: ${amountText}.`;
  const agent = agents.get(key ?? "");
  const before = payload.transition_start_health;
  const after = payload.health_after_combat_resolution;
  const lifeText =
    agent !== undefined && typeof before === "number" && typeof after === "number"
      ? ` Life ${formatDisplayNumber(before)} → ${formatDisplayNumber(after)}${
          typeof agent.maximum_health === "number"
            ? ` of ${formatDisplayNumber(agent.maximum_health)}`
            : ""
        }.`
      : "";
  return {
    kind,
    text: `${lead}${lifeText}`,
    anchorKey: seenKey(payload.recipient_anchor, agents),
  };
}

/**
 * Build the item of one new status in the Champion Card row's words
 * (statusLineV1, owner ruling, 2 October 2026): "Mage ID 5: Charge (Slow) ·
 * 5 ticks left · It moves at ×0.5 speed. From a Warrior." The source is named
 * by class only, in Agent POV and Oracle View alike; the event lines keep
 * their own sight rule for naming who did what.
 *
 * @param {Readonly<Record<string, any>>} payload
 * @param {Map<string, Record<string, any>>} agents
 * @returns {{kind: string, text: string, anchorKey: string | null}}
 */
function statusDraft(payload, agents) {
  const tokenId = statusTokenIdFromCatalogId(payload.status_id);
  const recipient = agents.get(anchorKey(payload.recipient_anchor) ?? "");
  const status = Array.isArray(recipient?.statuses)
    ? recipient.statuses.find(
        (/** @type {unknown} */ row) =>
          isRecord(row) && row.status_id === payload.status_id,
      )
    : undefined;
  const line = statusLineV1({
    tokenId,
    magnitude: status?.magnitude,
    remainingTicks: status?.remaining_duration,
    recipientClassId: recipient?.class_id,
  });
  return {
    kind: "status",
    text: `${capitalize(nameOf(payload.recipient_anchor, agents, UNSEEN_RECIPIENT))}: ${line}`,
    anchorKey: seenKey(payload.recipient_anchor, agents),
  };
}

/**
 * Build the item of one fall, with its points: 2 when the event's home_fall
 * flag says it scored double, else 1, to the other team.
 *
 * @param {Readonly<Record<string, any>>} payload
 * @param {Map<string, Record<string, any>>} agents
 * @returns {{kind: string, text: string, anchorKey: string | null}}
 */
function fallDraft(payload, agents) {
  const fallen = agents.get(anchorKey(payload.recipient_anchor) ?? "");
  const points = payload.home_fall === true ? 2 : 1;
  const scorer =
    fallen?.team_id === 1
      ? teamLabelFromId(2)
      : fallen?.team_id === 2
        ? teamLabelFromId(1)
        : null;
  const pointsText = `${points} ${points === 1 ? "point" : "points"}`;
  const scoreText =
    scorer === null
      ? `The other team scores ${pointsText}`
      : `${scorer} scores ${pointsText}`;
  const double = payload.home_fall === true ? " (x2: it fell in its own Red Zone)" : "";
  return {
    kind: "fall",
    text: `${capitalize(nameOf(payload.recipient_anchor, agents, UNSEEN_RECIPIENT))} fell. ${scoreText}${double}.`,
    anchorKey: seenKey(payload.recipient_anchor, agents),
  };
}

/**
 * Place the index-th disc of one champion: a row above its right side, flipped
 * to the left side when it would leave bounds on the right, and below the body
 * when it would leave bounds at the top. All values are CSS pixels.
 *
 * @param {{x: number, y: number}} center The body centre.
 * @param {number} radius The body radius.
 * @param {number} index 0 for the first disc of this champion.
 * @param {{left: number, top: number, right: number, bottom: number} | undefined} bounds
 *   The visible viewport, or undefined (no flipping).
 * @returns {{x: number, y: number}}
 */
function calloutPoint(center, radius, index, bounds) {
  const offset = radius * 0.7 + CALLOUT_RADIUS + index * CALLOUT_STEP;
  let x = center.x + offset;
  let y = center.y - radius * 0.7 - CALLOUT_RADIUS;
  if (bounds !== undefined) {
    if (x + CALLOUT_RADIUS > bounds.right) {
      x = center.x - offset;
    }
    if (y - CALLOUT_RADIUS < bounds.top) {
      y = center.y + radius * 0.7 + CALLOUT_RADIUS;
    }
  }
  return { x, y };
}

/**
 * Return a symbol's title from `symbols.js`, or its ID when unknown.
 *
 * @param {string} id
 * @returns {string}
 */
function symbolTitle(id) {
  return symbolById(id)?.title ?? id;
}

/**
 * Return value when it is a finite number, else fallback.
 *
 * @param {unknown} value
 * @param {number} fallback
 * @returns {number}
 */
function numberOr(value, fallback) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Join words as "A", "A and B" or "A, B and C".
 *
 * @param {ReadonlyArray<string>} words
 * @returns {string}
 */
function joinWords(words) {
  return words.length <= 1
    ? (words[0] ?? "")
    : `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`;
}

/**
 * Return text with its first letter in upper case.
 *
 * @param {string} text
 * @returns {string}
 */
function capitalize(text) {
  return text.length === 0 ? text : `${text[0].toUpperCase()}${text.slice(1)}`;
}

/**
 * Format a pixel value with at most two decimals for an SVG attribute.
 *
 * @param {unknown} value
 * @returns {string}
 */
function formatPixel(value) {
  return String(Math.round(finiteNumber(value) * 100) / 100);
}
