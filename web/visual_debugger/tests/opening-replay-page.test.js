/**
 * @file Check Open Replay's dark new-tab page (src/opening-replay-page.js
 * writeOpeningReplayPage): it titles the tab "Opening Replay…", sets the dark
 * background, text color and font through the style object only (never a
 * style attribute, which the page's CSP blocks), shows the heading "Opening
 * Replay" and one line saying the Replay Viewer opens by itself, uses the
 * fallback colors for empty values, and does nothing (returns false, throws
 * nothing) for a blocked tab (null) or a tab that can no longer be written;
 * main.js calls it on the window Open Replay opens.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  OPENING_REPLAY_FALLBACK_COLORS,
  writeOpeningReplayPage,
} from "../src/opening-replay-page.js";

class FakeStyle {
  constructor() {
    /** @type {Map<string, string>} */
    this.values = new Map();
  }
  /** @param {string} name @param {string} value */
  setProperty(name, value) {
    this.values.set(name, value);
  }
}

class FakeElement {
  /** @param {string} tag */
  constructor(tag) {
    this.tag = tag;
    this.style = new FakeStyle();
    this.textContent = "";
    /** @type {FakeElement[]} */
    this.children = [];
  }
  /** @param {FakeElement} child */
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  /** @param {...FakeElement} children */
  replaceChildren(...children) {
    this.children = children;
  }
  setAttribute() {
    throw new Error("a style attribute is blocked by the page's CSP");
  }
}

function fakeTab() {
  const root = new FakeElement("html");
  const doc = {
    title: "",
    documentElement: root,
    /** @type {FakeElement | null} */
    body: null,
    /** @param {string} tag */
    createElement: (tag) => new FakeElement(tag),
  };
  return /** @type {any} */ ({ document: doc });
}

test("the empty tab becomes the dark Opening Replay page", () => {
  const tab = fakeTab();
  const written = writeOpeningReplayPage(tab, {
    background: " #101010 ",
    text: "#fafafa",
    muted: "",
    font: "Inter, sans-serif",
  });
  assert.equal(written, true);
  const doc = tab.document;
  assert.equal(doc.title, "Opening Replay…");
  assert.equal(doc.documentElement.style.values.get("background"), "#101010");
  assert.equal(doc.documentElement.style.values.get("color-scheme"), "dark");
  const body = doc.documentElement.children[0];
  assert.equal(body.tag, "body");
  assert.equal(body.style.values.get("color"), "#fafafa");
  assert.equal(body.style.values.get("font-family"), "Inter, sans-serif");
  const [heading, line] = body.children;
  assert.equal(heading.textContent, "Opening Replay");
  assert.match(line.textContent, /The Replay Viewer opens here by itself\.$/u);
  assert.equal(line.style.values.get("color"), OPENING_REPLAY_FALLBACK_COLORS.muted);
});

test("a blocked or unwritable tab is left alone without an error", () => {
  assert.equal(writeOpeningReplayPage(null), false);
  const closed = /** @type {any} */ ({
    get document() {
      throw new Error("cross-origin");
    },
  });
  assert.equal(writeOpeningReplayPage(closed), false);
});

test("main.js writes the page into the window Open Replay opens", () => {
  const main = readFileSync(new URL("../src/main.js", import.meta.url), "utf8");
  const opened = main.indexOf('window.open("about:blank", "_blank")');
  const written = main.indexOf("writeOpeningReplayPage(replayWindow,");
  assert.ok(opened > 0 && written > opened, "written right after the tab opens");
});
