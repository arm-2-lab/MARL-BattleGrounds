/**
 * @file Check that every short Lock In row (three or more Manual champions)
 * stays whole on one line (owner, 3 Oct 2026; fitOrderRow and fitOrderRows
 * in src/tactics-turn.js, the row rules in deck.css):
 * - a row that fits at 12 px keeps its step and no inline style;
 * - a row too long drops its step first (`.order-tail` hidden), and only
 *   then steps its text down by half a pixel, stopping at the first size that
 *   fits and never going below 10 px;
 * - a row still too long at 10 px closes up its letters (-0.01em steps)
 *   while keeping its gaps, so the longest real row ("Priest: Holy Word:
 *   Salvation → Hunter 2", about 196 px in the 190 px column) fits; only a
 *   row still too long after -0.05em also closes its gaps (0.1em);
 * - fitting again starts over, so a row that became shorter grows back and
 *   shows its step;
 * - only the step is ever hidden; the ability and the target are never cut;
 *   fitOrderRows fits only `.order-row` paragraphs;
 * - deck.css never shrinks or cuts a row part ("flex: none", no ellipsis).
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { fitOrderRow, fitOrderRows } from "../src/tactics-turn.js";

const DECK_CSS = readFileSync(
  new URL("../src/styles/battleclient/deck.css", import.meta.url),
  "utf8",
);

/**
 * A fake `.order-row`: its width at 12 px is `base` without the step and
 * `tail` for the step, scaled by the font size; its two gaps are 0.3em and
 * it has `letters` letters for letter spacing.
 *
 * @param {{base: number, tail?: number | null, letters?: number, client?: number}} setup
 */
function fakeRow({ base, tail = null, letters = 40, client = 190 }) {
  /** @type {Map<string, string>} */
  const props = new Map();
  /** @type {string[]} */
  const asked = [];
  const tailPart = tail === null ? null : { hidden: false };
  const row = {
    asked,
    tailPart,
    props,
    clientWidth: client,
    style: {
      /** @param {string} name @param {string} value */
      setProperty(name, value) {
        props.set(name, value);
      },
      /** @param {string} name */
      removeProperty(name) {
        props.delete(name);
      },
    },
    /** @param {string} selector */
    querySelector(selector) {
      asked.push(selector);
      return selector === ".order-tail" ? tailPart : null;
    },
    get scrollWidth() {
      const size = Number.parseFloat(props.get("font-size") ?? "12");
      const gap = Number.parseFloat(props.get("column-gap") ?? "0.3");
      const spacing = Number.parseFloat(props.get("letter-spacing") ?? "0");
      const tailWidth = tailPart !== null && !tailPart.hidden ? (tail ?? 0) : 0;
      return Math.round(
        ((base + tailWidth) * size) / 12 +
          2 * (gap - 0.3) * size +
          letters * spacing * size,
      );
    },
  };
  return row;
}

/** @param {ReturnType<typeof fakeRow>} row */
function fit(row) {
  fitOrderRow(/** @type {any} */ (row));
}

test("a row that fits keeps its step and no inline style", () => {
  const row = fakeRow({ base: 150, tail: 30 });
  fit(row);
  assert.equal(row.tailPart?.hidden, false);
  assert.equal(row.props.size, 0);
});

test("the step gives way first, before the text gets smaller", () => {
  const row = fakeRow({ base: 185, tail: 40 });
  fit(row);
  assert.equal(row.tailPart?.hidden, true);
  assert.equal(row.props.has("font-size"), false);
});

test("then the text steps down by half a pixel and stops at the first fit", () => {
  // 210 px at 12 px: 201 at 11.5 (too long), 192.5 at 11 (too long), 184 at 10.5.
  const row = fakeRow({ base: 210, tail: 40 });
  fit(row);
  assert.equal(row.tailPart?.hidden, true);
  assert.equal(row.props.get("font-size"), "10.5px");
  assert.ok(row.scrollWidth <= row.clientWidth);
});

test("at 10 px the letters close up first, keeping the gaps", () => {
  // Like "Priest: Holy Word: Salvation → Hunter 2": about 196 px at 10 px.
  const row = fakeRow({ base: 235, tail: 50, letters: 40 });
  fit(row);
  assert.equal(row.props.get("font-size"), "10px");
  assert.equal(row.props.get("letter-spacing"), "-0.02em");
  assert.equal(row.props.has("column-gap"), false);
  assert.ok(row.scrollWidth <= row.clientWidth);
  // A row too long even at -0.05em also closes its gaps.
  const longer = fakeRow({ base: 270, tail: null, letters: 20 });
  fit(longer);
  assert.equal(longer.props.get("letter-spacing"), "-0.05em");
  assert.equal(longer.props.get("column-gap"), "0.1em");
});

test("fitting again starts over, so a shorter row grows back with its step", () => {
  const row = fakeRow({ base: 210, tail: 40 });
  fit(row);
  assert.equal(row.tailPart?.hidden, true);
  const shorter = fakeRow({ base: 120, tail: 40 });
  // The same element now holds a shorter order: copy the old inline state in.
  for (const [name, value] of row.props) shorter.props.set(name, value);
  if (shorter.tailPart) shorter.tailPart.hidden = true;
  fit(shorter);
  assert.equal(shorter.tailPart?.hidden, false);
  assert.equal(shorter.props.size, 0);
});

test("only the step is ever looked up and hidden; rows without one still fit", () => {
  const row = fakeRow({ base: 200, tail: null });
  fit(row);
  assert.deepEqual(row.asked, [".order-tail"]);
  assert.ok(row.scrollWidth <= row.clientWidth);
});

test("fitOrderRows fits every short row of the list and nothing else", () => {
  const rows = [fakeRow({ base: 210, tail: 40 }), fakeRow({ base: 100, tail: 20 })];
  /** @type {string[]} */
  const selectors = [];
  fitOrderRows(
    /** @type {any} */ ({
      /** @param {string} selector */
      querySelectorAll(selector) {
        selectors.push(selector);
        return rows;
      },
    }),
  );
  assert.deepEqual(selectors, [".order-row"]);
  assert.equal(rows[0].props.get("font-size"), "10.5px");
  assert.equal(rows[1].props.size, 0);
});

test("deck.css never shrinks or cuts a part of a short row", () => {
  const rules = DECK_CSS.slice(DECK_CSS.indexOf("/* One short row:"));
  const block = rules.slice(0, rules.indexOf(".order p + p"));
  assert.match(
    block,
    /\.order-row > b,\n[^{]*\.order-main,\n[^{]*\.order-target,\n[^{]*\.order-tail \{\n {2}flex: none;\n\}/u,
  );
  assert.doesNotMatch(block, /text-overflow: ellipsis/u);
  assert.match(block, /\.order-row \{[^}]*white-space: nowrap;/u);
});
