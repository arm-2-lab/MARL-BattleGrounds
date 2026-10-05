/**
 * @file Check that the old semantic tooltip never opens over the live Command
 * Deck, as on the approved vision page, whose deck controls have no hover card.
 *
 * Contracts: insideLiveCommandDeck is true for #command-deck and anything
 * inside it, and false for null and for elements outside it; the tooltip
 * controller opens no card for an owner inside the deck on hover, keyboard
 * focus or refresh (so a frame install can no longer hide and re-show a deck
 * card, the one-frame blink the play-through found), while an owner outside
 * the deck still opens its card; the deck owner keeps its aria-description for
 * screen readers.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  createSemanticDescriptor,
  createTooltipController,
  insideLiveCommandDeck,
  LIVE_COMMAND_DECK_ID,
  registerTooltipOwner,
} from "../src/tooltip.js";

/**
 * @param {any} ownerDocument
 * @param {string} [id]
 * @param {any} [parentElement]
 * @param {string} [tagName]
 */
function fakeElement(ownerDocument, id = "", parentElement = null, tagName = "DIV") {
  /** @type {Map<string, string>} */
  const attributes = new Map();
  return /** @type {any} */ ({
    id,
    hidden: false,
    ownerDocument,
    parentElement,
    tagName,
    textContent: "",
    style: {
      left: "",
      top: "",
      maxHeight: "",
      maxWidth: "",
      pointerEvents: "",
      visibility: "",
      setProperty() {},
    },
    getBoundingClientRect() {
      return { left: 100, top: 100, right: 140, bottom: 120, width: 40, height: 20 };
    },
    /** @param {string} name */
    hasAttribute: (name) => attributes.has(name),
    /** @param {string} name */
    getAttribute: (name) => attributes.get(name) ?? null,
    /** @param {string} name @param {string} value */
    setAttribute: (name, value) => attributes.set(name, value),
    /** @param {string} name */
    removeAttribute: (name) => attributes.delete(name),
  });
}

/** @param {string} id */
function legality(id) {
  return createSemanticDescriptor({
    kind: "legality",
    id,
    title: id,
    tone: "positive",
    accent: "none",
    summary: `${id} is allowed now.`,
    rows: [],
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "element",
  });
}

function harness() {
  const ownerDocument = {
    defaultView: { innerWidth: 1440, innerHeight: 900 },
    documentElement: { clientWidth: 1440, clientHeight: 900 },
  };
  /** @type {any[]} */
  let hits = [];
  const root = Object.assign(new EventTarget(), {
    ownerDocument,
    elementsFromPoint: () => hits,
    /** @param {any} element */
    contains: (element) => element.ownerDocument === ownerDocument,
  });
  const tooltip = fakeElement(ownerDocument, "visual-tooltip");
  const controller = createTooltipController({
    root: /** @type {any} */ (root),
    tooltip,
    title: fakeElement(ownerDocument),
    details: fakeElement(ownerDocument),
  });
  const deck = fakeElement(ownerDocument, LIVE_COMMAND_DECK_ID);
  const deckBody = fakeElement(ownerDocument, "command-deck-body", deck);
  const stepButton = fakeElement(ownerDocument, "", deckBody, "BUTTON");
  const arenaMark = fakeElement(ownerDocument, "arena-mark");
  return {
    root,
    tooltip,
    controller,
    deck,
    stepButton,
    arenaMark,
    /** @param {any[]} elements */
    hit(elements) {
      hits = elements;
    },
  };
}

/** @param {EventTarget} root @param {number} x @param {number} y */
function pointerMove(root, x, y) {
  const event = new Event("pointermove");
  Object.defineProperties(event, { clientX: { value: x }, clientY: { value: y } });
  root.dispatchEvent(event);
}

/** @param {EventTarget} root @param {any} target */
function focusIn(root, target) {
  const event = new Event("focusin");
  Object.defineProperties(event, {
    target: { value: target },
    relatedTarget: { value: null },
  });
  root.dispatchEvent(event);
}

test("insideLiveCommandDeck finds the deck and anything inside it", () => {
  const page = harness();
  assert.equal(LIVE_COMMAND_DECK_ID, "command-deck");
  assert.equal(insideLiveCommandDeck(page.deck), true);
  assert.equal(insideLiveCommandDeck(page.stepButton), true);
  assert.equal(insideLiveCommandDeck(page.arenaMark), false);
  assert.equal(insideLiveCommandDeck(null), false);
  assert.equal(insideLiveCommandDeck(undefined), false);
  page.controller.destroy();
});

test("no card opens over the live Command Deck; cards elsewhere still open", () => {
  const page = harness();
  registerTooltipOwner(page.stepButton, legality("North, key W"));
  registerTooltipOwner(page.arenaMark, legality("Step Ghost"));
  assert.equal(
    page.stepButton.getAttribute("aria-description"),
    "North, key W is allowed now.",
    "the deck owner keeps its sentence for screen readers",
  );

  page.hit([page.stepButton]);
  pointerMove(page.root, 120, 110);
  assert.equal(page.tooltip.hidden, true, "no hover card on a deck control");
  focusIn(page.root, page.stepButton);
  assert.equal(page.tooltip.hidden, true, "no focus card on a deck control");
  page.controller.refresh();
  assert.equal(page.tooltip.hidden, true, "a refresh opens no deck card");

  page.hit([page.arenaMark]);
  pointerMove(page.root, 120, 110);
  assert.equal(page.tooltip.hidden, false, "an owner outside the deck shows its card");
  page.hit([page.stepButton]);
  page.controller.refresh();
  assert.equal(page.tooltip.hidden, true, "moving onto the deck closes the card");
  page.controller.destroy();
});
