/**
 * @file Check the token chip contract (src/token-chip.js): a token chip reads
 * "Agent N" with the class emblem in a single (Team A) or double (Team B) ring;
 * the Out Of Sight chip reads "Warrior · ID 6 · Out Of Sight" with no ring and
 * no team; the plain chip reads "Agent ID N"; "Itself" replaces only a token
 * chip's words; the chip carries identity only (no life or position field is
 * read or written); no chip node sets a data-class attribute; a pickable chip
 * is a button that calls onPick with its public ID and stops the click there.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { createTokenChip, tokenChipModel } from "../src/token-chip.js";

const WARRIOR_B = Object.freeze({
  public_agent_id: "agent-slot-6",
  display_agent_id: "6",
  class_id: 2,
  team_id: 2,
  presentation_key: "key-6",
});
const MAGE_A = Object.freeze({
  public_agent_id: "agent-slot-0",
  display_agent_id: "0",
  class_id: 1,
  team_id: 1,
});

function fakeDocument() {
  /** @param {string} tagName */
  const create = (tagName) => {
    /** @type {any[]} */
    const children = [];
    const attributes = new Map();
    const listeners = new Map();
    /** @type {any} */
    const node = {
      tagName: tagName.toUpperCase(),
      className: "",
      textContent: "",
      type: "",
      dataset: {},
      children,
      /** @param {...any} nodes */
      append(...nodes) {
        children.push(...nodes);
      },
      /** @param {string} name @param {unknown} value */
      setAttribute(name, value) {
        attributes.set(name, String(value));
      },
      /** @param {string} name */
      getAttribute(name) {
        return attributes.get(name) ?? null;
      },
      /** @param {string} name */
      hasAttribute(name) {
        return attributes.has(name);
      },
      /** @param {string} name */
      removeAttribute(name) {
        attributes.delete(name);
      },
      /** @param {string} type @param {(event: any) => void} listener */
      addEventListener(type, listener) {
        listeners.set(type, listener);
      },
      /** @param {any} event */
      dispatch(event) {
        listeners.get(event.type)?.(event);
      },
    };
    return node;
  };
  return {
    createElement: create,
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      create(tag),
  };
}

/** @param {any} root @returns {any[]} */
const descendants = (root) => [
  root,
  ...(root.children ?? []).flatMap((/** @type {any} */ child) => descendants(child)),
];

test("chip words and rings follow the label owner", () => {
  const warrior = tokenChipModel(WARRIOR_B);
  assert.equal(warrior.mode, "token");
  assert.equal(warrior.text, "Agent 6");
  assert.equal(warrior.ring, "double");
  assert.equal(warrior.teamKey, "team-b");
  assert.equal(warrior.classKey, "warrior");
  assert.equal(warrior.ariaLabel, "Agent ID 6, Warrior, Team B");
  assert.equal(tokenChipModel(MAGE_A).ring, "single");
  assert.equal(tokenChipModel(MAGE_A, { label: "Itself" }).text, "Itself");

  const plain = tokenChipModel(WARRIOR_B, { plain: true, label: "Itself" });
  assert.equal(plain.mode, "plain");
  assert.equal(plain.text, "Agent ID 6");
  assert.equal(plain.ring, "none");

  const unseen = tokenChipModel(WARRIOR_B, { outOfSight: true, plain: true });
  assert.equal(unseen.mode, "out_of_sight");
  assert.equal(unseen.text, "Warrior · ID 6 · Out Of Sight");
  assert.equal(unseen.ring, "none");
  assert.doesNotMatch(unseen.ariaLabel, /Team/u);
  assert.doesNotMatch(unseen.tipTitle, /Team/u);
  assert.equal(Object.isFrozen(unseen), true);
});

test("chips read identity only", () => {
  const guarded = new Proxy(
    { ...WARRIOR_B },
    {
      get(target, field) {
        if (["current_health", "position", "statuses"].includes(String(field))) {
          throw new Error(`chip read ${String(field)}`);
        }
        return Reflect.get(target, field);
      },
    },
  );
  assert.equal(tokenChipModel(guarded).text, "Agent 6");
});

test("chip DOM carries its form, ring and pick route", () => {
  const doc = fakeDocument();
  const chip = createTokenChip(/** @type {any} */ (doc), WARRIOR_B, { tooltip: false });
  const anyChip = /** @type {any} */ (chip);
  assert.equal(anyChip.tagName, "SPAN");
  assert.equal(anyChip.dataset.chipMode, "token");
  assert.equal(anyChip.dataset.tokenRing, "double");
  assert.equal(anyChip.dataset.tokenTeam, "team-b");
  assert.equal(
    descendants(anyChip).some((node) => Object.hasOwn(node.dataset ?? {}, "class")),
    false,
  );
  assert.equal(
    descendants(anyChip).some((node) => node.textContent === "Agent 6"),
    true,
  );

  const unseen = /** @type {any} */ (
    createTokenChip(/** @type {any} */ (doc), WARRIOR_B, {
      outOfSight: true,
      tooltip: false,
    })
  );
  assert.equal(Object.hasOwn(unseen.dataset, "tokenTeam"), false);
  assert.equal(unseen.dataset.tokenRing, "none");

  /** @type {string[]} */
  const picked = [];
  const button = /** @type {any} */ (
    createTokenChip(/** @type {any} */ (doc), WARRIOR_B, {
      tooltip: false,
      onPick: (publicAgentId) => picked.push(publicAgentId),
    })
  );
  assert.equal(button.tagName, "BUTTON");
  assert.equal(button.type, "button");
  let stopped = false;
  button.dispatch({
    type: "click",
    stopPropagation() {
      stopped = true;
    },
  });
  assert.deepEqual(picked, ["agent-slot-6"]);
  assert.equal(stopped, true);
});
