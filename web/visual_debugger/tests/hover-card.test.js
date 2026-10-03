/**
 * @file Check the live hover card's binding to the page (src/tooltip.js
 * createHoverCard, the #hcard of the BattleClient and DevClient) on a small
 * stand-in page.
 *
 * Contracts (owner, 2 October 2026): a card never keeps an earlier frame's
 * facts. While the pointer rests on a champion, the returned refresh (main.js
 * calls it through shell.js after every installed frame: a new tick, or Tab
 * giving Agent POV to the other team) finds the champion element under the
 * same pointer point again, even when the frame redrew that element, asks
 * factsFor again and redraws the card in place: new life and statuses after a
 * tick, the reduced "Unseen" card when the new view cannot see that champion,
 * and no card when factsFor gives null or no champion element is under the
 * pointer any more. A card shown for a keyboard focus follows its element to
 * where the new frame put it. refresh does nothing while no card shows. Hold
 * Shift starts and expands only from fresh facts: it never expands a card
 * that already shows all of its fresh statuses, and the expanded card shows
 * the fresh rows, not the rows captured when the pointer arrived.
 *
 * "Fewer rows" (owner, 2 October 2026): in a short arena box (290 px, as at
 * 1280x720) the plain card stays one column and shows only the leading rows
 * that fit, in order, then "+N more" with "Hold Shift For All", inside the
 * box; Hold Shift shows every row in two columns, inside the box (four rows
 * have no footer, so no extra line); releasing shows the fitted card
 * again; a card whose rows all fit after a new frame shows no footer and
 * Shift does not expand it; and a taller box (a resize, then refresh) fits
 * all four rows again.
 *
 * "Grow past arena" (owner, 2 October 2026): in the DevClient's 227 px arena
 * box in a 1280x720 window, an eight-status Hold Shift card that would need a
 * scale below 0.85 inside the box takes four columns at scale 1, marks itself
 * data-past-arena, and sits 12 px below the champion, past the box and inside
 * the window; releasing Shift brings back the fitted plain card inside the box,
 * in the same place as before.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { CHAMPION_CARD_HOLD_MS, createHoverCard } from "../src/tooltip.js";

/** @typedef {import("../src/tooltip.js").ChampionCardFacts} Facts */

// A stand-in DOM element: just what createHoverCard, renderChampionCard and
// placeChampionCard touch. `subject` marks a champion element (what
// HOVER_CARD_SELECTOR matches on the real page).
class StandInElement {
  /** @param {any} doc @param {string} tagName */
  constructor(doc, tagName) {
    this.ownerDocument = doc;
    this.tagName = tagName.toUpperCase();
    this.className = "";
    /** @type {Record<string, string>} */
    this.dataset = {};
    /** @type {Record<string, string>} */
    this.style = {};
    this.hidden = false;
    this.subject = false;
    /** @type {StandInElement | null} */
    this.parent = null;
    /** @type {Array<StandInElement | string>} */
    this.kids = [];
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {Map<string, Array<(event: any) => void>>} */
    this.listeners = new Map();
    this.rect = { left: 0, top: 0, width: 0, height: 0 };
  }
  get children() {
    return this.kids;
  }
  /** @returns {string} */
  get textContent() {
    return this.kids
      .map((child) => (typeof child === "string" ? child : child.textContent))
      .join("");
  }
  set textContent(value) {
    this.kids = [String(value)];
  }
  get classList() {
    const names = () => this.className.split(" ").filter(Boolean);
    return {
      add: (/** @type {string} */ name) => {
        if (!names().includes(name)) this.className = [...names(), name].join(" ");
      },
      remove: (/** @type {string} */ name) => {
        this.className = names()
          .filter((each) => each !== name)
          .join(" ");
      },
    };
  }
  /** @param {...(StandInElement | string)} nodes */
  append(...nodes) {
    for (const node of nodes) {
      if (typeof node !== "string") node.parent = this;
      this.kids.push(node);
    }
  }
  /** @param {...(StandInElement | string)} nodes */
  replaceChildren(...nodes) {
    this.kids = [];
    this.append(...nodes);
  }
  /** @param {string} name @param {unknown} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }
  /** @param {string} name */
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  // Only ".class" selectors, which is all placeChampionCard and
  // renderFittedChampionCard ask.
  /** @param {string} selector @returns {StandInElement | null} */
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }
  /** @param {string} selector @returns {StandInElement[]} */
  querySelectorAll(selector) {
    const name = selector.replace(/^\./u, "");
    return this.kids.flatMap((child) =>
      typeof child === "string"
        ? []
        : [
            ...(child.className.split(" ").includes(name) ? [child] : []),
            ...child.querySelectorAll(selector),
          ],
    );
  }
  // The nearest champion element, as closest(HOVER_CARD_SELECTOR) gives.
  /** @returns {StandInElement | null} */
  closest() {
    /** @type {StandInElement | null} */
    let node = this;
    while (node !== null && !node.subject) node = node.parent;
    return node;
  }
  /** @param {any} node */
  contains(node) {
    for (let each = node; each !== null && each !== undefined; each = each.parent) {
      if (each === this) return true;
    }
    return false;
  }
  get isConnected() {
    return this.ownerDocument.body.contains(this);
  }
  // The page's layout (doc.layout, set by a test) may size an element; else
  // its fixed rect.
  getBoundingClientRect() {
    const { left, top, width, height } = this.ownerDocument.layout?.(this) ?? this.rect;
    return { left, top, width, height, right: left + width, bottom: top + height };
  }
  /** @param {string} type @param {(event: any) => void} listener */
  addEventListener(type, listener) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }
  /** @param {string} type @param {(event: any) => void} listener */
  removeEventListener(type, listener) {
    this.listeners.set(
      type,
      (this.listeners.get(type) ?? []).filter((each) => each !== listener),
    );
  }
  /** @param {string} type @param {Record<string, unknown>} [event] */
  fire(type, event = {}) {
    for (const listener of this.listeners.get(type) ?? []) listener({ type, ...event });
  }
}

// A stand-in page: #bc (root) holding the arena box, #hcard and the champion
// elements, a window for the Hold Shift listeners, and hit testing that
// returns whatever `doc.under` holds (the stack under the pointer).
function standInPage() {
  /** @type {any} */
  const doc = {
    /** @type {StandInElement[]} */
    under: [],
    /** @type {StandInElement | null} */
    activeElement: null,
    createElement: (/** @type {string} */ tag) => new StandInElement(doc, tag),
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      new StandInElement(doc, tag),
    elementsFromPoint: () => doc.under,
  };
  doc.defaultView = new StandInElement(doc, "window");
  doc.body = new StandInElement(doc, "body");
  const root = new StandInElement(doc, "div");
  root.rect = { left: 0, top: 0, width: 1440, height: 900 };
  doc.body.append(root);
  const arena = new StandInElement(doc, "div");
  arena.rect = { left: 12, top: 144, width: 1076, height: 470 };
  const card = new StandInElement(doc, "div");
  card.hidden = true;
  // The drawn card's size: one column, 262 by 200 pixels.
  card.rect = { left: 0, top: 0, width: 262, height: 200 };
  root.append(arena, card);
  // Draws a champion element (a portrait or an arena body) for slot, at rect.
  /**
   * @param {number} slot
   * @param {{left: number, top: number, width: number, height: number}} rect
   */
  const champion = (slot, rect) => {
    const element = new StandInElement(doc, "div");
    element.subject = true;
    element.setAttribute("data-slot", String(slot));
    element.rect = rect;
    // A child inside it, as the real portrait's picture is.
    const inner = new StandInElement(doc, "span");
    inner.rect = rect;
    element.append(inner);
    root.append(element);
    return { element, inner };
  };
  return { doc, root, arena, card, champion };
}

// Card facts for the Mage, Agent ID 5.
/** @param {Partial<Facts>} [overrides] @returns {Facts} */
function mage(overrides = {}) {
  return {
    state: "seen",
    name: "Mage",
    calling: "The Astral Walker",
    classId: 1,
    displayId: "5",
    teamId: 2,
    life: { current: 77, maximum: 80 },
    ultimate: { name: "Burst", remaining: 30, full: 30 },
    statuses: [],
    note: null,
    you: false,
    playedBy: "tdm-beta",
    yourTarget: false,
    unseenBy: null,
    ...overrides,
  };
}

/** @param {string} id @param {number} [ticks] */
const status = (id, ticks = 3) => ({ id, ticks, full: 5, amount: null });

const UNSEEN = mage({
  state: "unseen",
  life: null,
  ultimate: null,
  playedBy: "tdm-beta",
});

/** @param {any} root @param {string} name @returns {any[]} */
const byClass = (root, name) => [
  ...(typeof root !== "string" && root.className.split(" ").includes(name)
    ? [root]
    : []),
  ...(typeof root === "string" ? [] : root.children).flatMap(
    (/** @type {any} */ child) => byClass(child, name),
  ),
];

// What the shown card says (null while hidden): its state, its life number
// and its status rows.
/** @param {StandInElement} card */
function shown(card) {
  if (card.hidden) return null;
  const [root] = byClass(card, "champion-card");
  return {
    state: root.dataset.state,
    expanded: root.dataset.expanded === "true",
    life: byClass(root, "life-bar__value")[0]?.textContent ?? null,
    unseen: byClass(root, "champion-card__unseen").length > 0,
    rows: byClass(root, "champion-card__row").map((row) => row.dataset.status),
  };
}

// Runs body with StandInElement as the page's Element and HTMLElement
// (createHoverCard checks `instanceof Element`), then puts the originals back.
/** @param {() => void} body */
function withStandInClasses(body) {
  const saved = ["Element", "HTMLElement"].map((name) => [
    name,
    Object.getOwnPropertyDescriptor(globalThis, name),
  ]);
  for (const name of ["Element", "HTMLElement"]) {
    Object.defineProperty(globalThis, name, {
      configurable: true,
      writable: true,
      value: StandInElement,
    });
  }
  try {
    body();
  } finally {
    for (const [name, descriptor] of saved) {
      if (descriptor === undefined) Reflect.deleteProperty(globalThis, String(name));
      else Object.defineProperty(globalThis, String(name), descriptor);
    }
  }
}

test("a resting pointer's card follows each new frame: new facts, Unseen, or none", () => {
  withStandInClasses(() => {
    const page = standInPage();
    /** @type {Map<number, Facts | null>} */
    const frame = new Map([[5, mage({ statuses: [status("stun_hunter_trap", 3)] })]]);
    const hover = createHoverCard({
      root: /** @type {any} */ (page.root),
      card: /** @type {any} */ (page.card),
      factsFor: (subject) => frame.get(subject.slot ?? -1) ?? null,
      box: () => /** @type {any} */ (page.arena),
    });
    const portrait = page.champion(5, { left: 1170, top: 60, width: 40, height: 40 });
    page.doc.under = [portrait.inner, portrait.element, page.root];
    page.root.fire("pointermove", {
      pointerType: "mouse",
      target: portrait.inner,
      clientX: 1190,
      clientY: 80,
    });
    assert.deepEqual(shown(page.card), {
      state: "seen",
      expanded: false,
      life: "77 / 80",
      unseen: false,
      rows: ["stun_hunter_trap"],
    });
    const placed = { left: page.card.style.left, top: page.card.style.top };

    // A new tick: the trap broke and the Mage lost life. The pointer has not
    // moved; refresh redraws the card from the new frame, in the same place.
    frame.set(
      5,
      mage({ life: { current: 56, maximum: 80 }, statuses: [status("stun_broken")] }),
    );
    hover.refresh();
    assert.equal(shown(page.card)?.life, "56 / 80");
    assert.deepEqual(shown(page.card)?.rows, ["stun_broken"]);
    assert.deepEqual({ left: page.card.style.left, top: page.card.style.top }, placed);

    // Tab gives Agent POV to a team that cannot see the Mage, and the frame
    // redraws the portrait: refresh finds the new portrait under the same
    // point and shows the reduced card, with no life or statuses.
    portrait.element.parent = null;
    const redrawn = page.champion(5, { left: 1170, top: 60, width: 40, height: 40 });
    page.doc.under = [redrawn.inner, redrawn.element, page.root];
    frame.set(5, UNSEEN);
    hover.refresh();
    assert.deepEqual(shown(page.card), {
      state: "unseen",
      expanded: false,
      life: null,
      unseen: true,
      rows: [],
    });

    // An arena body the new view cannot see is no longer drawn: no card.
    frame.set(5, null);
    hover.refresh();
    assert.equal(shown(page.card), null);

    // While no card shows, refresh does nothing, even when facts return.
    frame.set(5, mage());
    hover.refresh();
    assert.equal(shown(page.card), null);
    hover.destroy();
  });
});

test("refresh hides the card when no champion element is under the pointer any more", () => {
  withStandInClasses(() => {
    const page = standInPage();
    const hover = createHoverCard({
      root: /** @type {any} */ (page.root),
      card: /** @type {any} */ (page.card),
      factsFor: () => mage(),
      box: () => /** @type {any} */ (page.arena),
    });
    const body = page.champion(5, { left: 400, top: 300, width: 30, height: 30 });
    page.doc.under = [body.inner, body.element, page.root];
    page.root.fire("pointermove", {
      pointerType: "mouse",
      target: body.inner,
      clientX: 415,
      clientY: 315,
    });
    assert.equal(shown(page.card)?.state, "seen");
    // The body moved away on the new tick; only the arena floor is there.
    page.doc.under = [page.arena, page.root];
    hover.refresh();
    assert.equal(shown(page.card), null);
    hover.destroy();
  });
});

test("a keyboard focus card follows its element to where the new frame put it", () => {
  withStandInClasses(() => {
    const page = standInPage();
    let life = 80;
    const hover = createHoverCard({
      root: /** @type {any} */ (page.root),
      card: /** @type {any} */ (page.card),
      factsFor: () => mage({ life: { current: life, maximum: 80 } }),
      box: () => /** @type {any} */ (page.arena),
    });
    const row = page.champion(5, { left: 1100, top: 400, width: 200, height: 30 });
    page.doc.activeElement = row.element;
    page.root.fire("focusin", { target: row.element });
    assert.equal(shown(page.card)?.life, "80 / 80");
    const before = page.card.style.top;
    // The new frame moved the focused row down and the Mage lost life.
    row.element.rect = { left: 1100, top: 560, width: 200, height: 30 };
    life = 40;
    hover.refresh();
    assert.equal(shown(page.card)?.life, "40 / 80");
    assert.notEqual(page.card.style.top, before);
    // Focus left the champion: the next refresh hides the card.
    page.doc.activeElement = null;
    hover.refresh();
    assert.equal(shown(page.card), null);
    hover.destroy();
  });
});

test("Hold Shift starts and expands from fresh facts only", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  withStandInClasses(() => {
    const page = standInPage();
    const five = [
      status("stun_hunter_trap"),
      status("slow_warrior_charge"),
      status("mage_burst"),
      status("in_combat"),
      status("priest_freedom"),
    ];
    /** @type {Facts} */
    let now = mage({ statuses: five });
    const hover = createHoverCard({
      root: /** @type {any} */ (page.root),
      card: /** @type {any} */ (page.card),
      factsFor: () => now,
      box: () => /** @type {any} */ (page.arena),
    });
    const body = page.champion(5, { left: 400, top: 300, width: 30, height: 30 });
    page.doc.under = [body.inner, body.element, page.root];
    page.root.fire("pointermove", {
      pointerType: "mouse",
      target: body.inner,
      clientX: 415,
      clientY: 315,
    });
    assert.equal(shown(page.card)?.rows.length, 4);
    const shift = { key: "Shift", ctrlKey: false, altKey: false, metaKey: false };
    // The new frame leaves only three statuses: the card fits, so Shift
    // never expands it, although five rows were there on arrival.
    now = mage({ statuses: five.slice(0, 3) });
    hover.refresh();
    page.doc.defaultView.fire("keydown", { ...shift, repeat: false });
    t.mock.timers.tick(CHAMPION_CARD_HOLD_MS);
    assert.equal(shown(page.card)?.expanded, false);
    assert.equal(shown(page.card)?.rows.length, 3);
    page.doc.defaultView.fire("keyup", shift);

    // Five statuses again, and one more arrives during the hold: the
    // expanded card shows the fresh six rows.
    now = mage({ statuses: five });
    hover.refresh();
    page.doc.defaultView.fire("keydown", { ...shift, repeat: false });
    now = mage({ statuses: [...five, status("slow_hunter_basic")] });
    t.mock.timers.tick(CHAMPION_CARD_HOLD_MS);
    assert.equal(shown(page.card)?.expanded, true);
    assert.equal(shown(page.card)?.rows.length, 6);
    // Releasing Shift shrinks it back to four rows, again from fresh facts.
    page.doc.defaultView.fire("keyup", shift);
    assert.equal(shown(page.card)?.expanded, false);
    assert.equal(shown(page.card)?.rows.length, 4);
    hover.destroy();
  });
});

// A layout for the drawn Champion Card (owner, 2 October 2026: "Fewer
// rows"): 110 px above the rows, 50 px rows 7 px apart, a 29 px footer and
// 24 px below; the Hold Shift card's columns split the rows evenly.
/** @param {StandInElement} card @returns {(node: StandInElement) => any} */
function cardLayout(card) {
  const [top, row, gap, footer, below] = [110, 50, 7, 29, 24];
  /** @param {number} count */
  const rowsHeight = (count) => count * row + Math.max(0, count - 1) * gap;
  return (node) => {
    const names = node.className.split(" ");
    if (names.includes("champion-card__row")) {
      const index = /** @type {StandInElement} */ (node.parent).kids.indexOf(node);
      return { left: 0, top: top + index * (row + gap), width: 262, height: row };
    }
    if (names.includes("champion-card__rows")) {
      return { left: 0, top, width: 262, height: rowsHeight(node.kids.length) };
    }
    if (names.includes("champion-card__more")) {
      const block = /** @type {StandInElement} */ (node.parent).querySelector(
        ".champion-card__rows",
      );
      const bottom = top + rowsHeight(block?.kids.length ?? 0) + footer;
      return { left: 0, top: bottom - 20, width: 262, height: 20 };
    }
    if (node !== card || card.hidden) return null;
    const [root] = byClass(card, "champion-card");
    const rows = byClass(root, "champion-card__row").length;
    const columns = Number(root.dataset.columns ?? 1);
    const tall = rowsHeight(Math.ceil(rows / columns));
    const more = byClass(root, "champion-card__more").length > 0 ? footer : 0;
    return {
      left: 0,
      top: 0,
      width: 262 * columns + 16 * (columns - 1),
      height: top + tall + more + below,
    };
  };
}

test("a short arena box: the plain card shows fewer rows, Shift shows all, release fits again", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  withStandInClasses(() => {
    const page = standInPage();
    // The 1280x720 BattleClient arena: 290 px tall, 274 px inside the margins.
    page.arena.rect = { left: 12, top: 144, width: 960, height: 290 };
    page.doc.layout = cardLayout(page.card);
    const four = [
      status("in_combat"),
      status("mage_burst"),
      status("stun_hunter_trap"),
      status("priest_freedom"),
    ];
    /** @type {Facts} */
    let now = mage({ statuses: four });
    const hover = createHoverCard({
      root: /** @type {any} */ (page.root),
      card: /** @type {any} */ (page.card),
      factsFor: () => now,
      box: () => /** @type {any} */ (page.arena),
    });
    const body = page.champion(5, { left: 400, top: 300, width: 30, height: 30 });
    page.doc.under = [body.inner, body.element, page.root];
    page.root.fire("pointermove", {
      pointerType: "mouse",
      target: body.inner,
      clientX: 415,
      clientY: 315,
    });
    /** @returns {string | null} */
    const footer = () =>
      byClass(page.card, "champion-card__more")[0]?.textContent ?? null;
    /** @returns {any} */
    const root = () => byClass(page.card, "champion-card")[0];
    // Four rows would need 355 px: the card shows the first two, in order,
    // then "+2 more", one column, inside the box.
    assert.deepEqual(shown(page.card)?.rows, ["stun_hunter_trap", "priest_freedom"]);
    assert.equal(footer(), "+2 moreHold Shift For All");
    assert.equal(root().dataset.columns, undefined);
    const top = Number.parseFloat(page.card.style.top);
    const height = page.card.getBoundingClientRect().height;
    assert.ok(top >= 144 + 8 && top + height <= 144 + 290 - 8, `${top} ${height}`);

    // Hold Shift: every status, in two columns, still inside the box.
    const shift = { key: "Shift", ctrlKey: false, altKey: false, metaKey: false };
    page.doc.defaultView.fire("keydown", { ...shift, repeat: false });
    t.mock.timers.tick(CHAMPION_CARD_HOLD_MS);
    assert.equal(shown(page.card)?.expanded, true);
    assert.equal(shown(page.card)?.rows.length, 4);
    assert.equal(root().dataset.columns, "2");
    // Four rows or fewer: no footer, so the short box needs no extra line.
    assert.equal(footer(), null);
    const tall = page.card.getBoundingClientRect().height;
    const shiftTop = Number.parseFloat(page.card.style.top);
    assert.ok(shiftTop >= 152 && shiftTop + tall <= 426, `${shiftTop} ${tall}`);

    // Release: the fitted plain card again.
    page.doc.defaultView.fire("keyup", shift);
    assert.equal(shown(page.card)?.expanded, false);
    assert.equal(shown(page.card)?.rows.length, 2);

    // A new frame with one status left: the card fits, so nothing is left
    // out and Shift does not expand it.
    now = mage({ statuses: four.slice(2, 3) });
    hover.refresh();
    assert.deepEqual(shown(page.card)?.rows, ["stun_hunter_trap"]);
    assert.equal(footer(), null);
    page.doc.defaultView.fire("keydown", { ...shift, repeat: false });
    t.mock.timers.tick(CHAMPION_CARD_HOLD_MS);
    assert.equal(shown(page.card)?.expanded, false);
    page.doc.defaultView.fire("keyup", shift);

    // A taller box (a resize, then the next refresh): all four rows fit.
    now = mage({ statuses: four });
    page.arena.rect = { left: 12, top: 144, width: 1076, height: 470 };
    hover.refresh();
    assert.equal(shown(page.card)?.rows.length, 4);
    assert.equal(footer(), null);
    hover.destroy();
  });
});

test("Grow past arena: a Hold Shift card too small inside the box grows past it, release fits again", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  withStandInClasses(() => {
    const page = standInPage();
    // The DevClient at 1280x720: an arena box 227 px tall, in a 1280x720
    // window.
    page.arena.rect = { left: 12, top: 206, width: 966, height: 227 };
    page.doc.documentElement = { clientWidth: 1280, clientHeight: 720 };
    page.doc.layout = cardLayout(page.card);
    const eight = [
      "stun_rogue_poison",
      "slow_rogue_poison",
      "slow_hunter_basic",
      "anti_heal_rogue_poison",
      "mage_burst",
      "in_combat",
      "mage_amplification",
      "warrior_mitigation",
    ].map((id) => status(id));
    const hover = createHoverCard({
      root: /** @type {any} */ (page.root),
      card: /** @type {any} */ (page.card),
      factsFor: () => mage({ statuses: eight }),
      box: () => /** @type {any} */ (page.arena),
    });
    const body = page.champion(0, { left: 400, top: 300, width: 30, height: 30 });
    page.doc.under = [body.inner, body.element, page.root];
    page.root.fire("pointermove", {
      pointerType: "mouse",
      target: body.inner,
      clientX: 415,
      clientY: 315,
    });
    /** @returns {any} */
    const root = () => byClass(page.card, "champion-card")[0];
    // The card's drawn rectangle: the stand-in does not apply the CSS
    // scale, so it is applied here.
    const where = () => {
      const left = Number.parseFloat(page.card.style.left);
      const top = Number.parseFloat(page.card.style.top);
      const scale = Number(/scale\(([\d.]+)\)/u.exec(root().style.transform)?.[1] ?? 1);
      const { width, height } = page.card.getBoundingClientRect();
      return { left, top, right: left + width * scale, bottom: top + height * scale };
    };
    // The plain card: one column, inside the box, never past it.
    assert.equal(root().dataset.pastArena, undefined);
    const plain = where();
    assert.ok(plain.top >= 214 && plain.bottom <= 425 + 0.01, JSON.stringify(plain));

    // Hold Shift: inside, three columns (327 px) would need scale 0.65, so
    // the card takes four columns (270 px), its shortest, at scale 1, past
    // the box, 12 px below the champion and inside the window.
    const shift = { key: "Shift", ctrlKey: false, altKey: false, metaKey: false };
    page.doc.defaultView.fire("keydown", { ...shift, repeat: false });
    t.mock.timers.tick(CHAMPION_CARD_HOLD_MS);
    assert.equal(shown(page.card)?.rows.length, 8);
    assert.deepEqual(
      [root().dataset.columns, root().dataset.pastArena, root().style.transform],
      ["4", "true", ""],
    );
    const grown = where();
    assert.deepEqual(grown, { left: 8, top: 342, right: 1104, bottom: 612 });
    assert.ok(grown.right > 978, "wider than the arena box");
    assert.ok(grown.top >= 330, "off the champion");

    // Release: the fitted plain card, inside the box again.
    page.doc.defaultView.fire("keyup", shift);
    assert.equal(shown(page.card)?.expanded, false);
    assert.deepEqual(
      [root().dataset.columns, root().dataset.pastArena],
      [undefined, undefined],
    );
    assert.deepEqual(where(), plain);
    hover.destroy();
  });
});
