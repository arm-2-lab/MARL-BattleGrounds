/**
 * @file Check the V5 docked panel rules in src/panels.js: actions in words
 * (describeRecordedAction: "Moves Northwest", "Stays", the class's Basic or
 * Ultimate name, "No Combat", "Itself", the recorded codes) with the movement
 * fallback pinned to core/axis_mappings.py; panelDisplayStyle (Panel Tokens
 * alone decides chips in every client, shipped defaults on bad input; actions
 * always read in words, so it has no action style); the
 * Full Record chip only in Replay Viewer Agent POV; and the BattleClient Agent
 * POV boundary: an unseen champion's roster row reads "Unseen" with no life,
 * status or modifier (a seen row shows the shared Life bar, "Life" and
 * "200 / 200"), and its selected details read Out Of Sight with no facts;
 * its Pending Manual Actions card shows only the POV agent's own staged order,
 * taken from its own POV draft, and reads "Not Observed" (no move, target or
 * code) for every other staged order (a teammate's or another team's); the
 * researcher branch's staged orders never change that card.
 * statusIconsFromMarks and the roster status icons follow the Status Icons From
 * Marks switch (mark icons on, plain icons off).
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { authorizedPresentationIdentityRows } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  DebuggerPanels,
  describeRecordedAction,
  FALLBACK_MOVEMENT_NAMES,
  fullRecordLabel,
  panelDisplayStyle,
  statusIconsFromMarks,
} from "../src/panels.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  setVisualFilterEnabled,
} from "../src/visual-filters.js";

const fixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @param {string} kind */
const presentation = (kind) =>
  normalizeAuthorizedPresentationFrameV1(fixture.presentations[kind]);

function domHarness() {
  /** @type {any} */
  let ownerDocument = null;
  /** @param {string} tagName */
  const create = (tagName) => {
    /** @type {any[]} */
    const children = [];
    Object.defineProperty(children, "item", {
      value: (/** @type {number} */ index) => children[index] ?? null,
    });
    const attributes = new Map();
    /** @type {any} */
    const node = {
      ownerDocument,
      tagName: tagName.toUpperCase(),
      parentElement: null,
      className: "",
      textContent: "",
      type: "",
      hidden: false,
      disabled: false,
      tabIndex: -1,
      dataset: {},
      style: {},
      children,
      /** @param {...any} nodes */
      append(...nodes) {
        for (const child of nodes) {
          child.remove?.();
          child.parentElement = this;
          children.push(child);
        }
      },
      /** @param {...any} nodes */
      replaceChildren(...nodes) {
        for (const child of children) child.parentElement = null;
        children.splice(0);
        this.append(...nodes);
      },
      /** @param {any} child @param {any} reference */
      insertBefore(child, reference) {
        child.remove?.();
        const index =
          reference === null ? children.length : children.indexOf(reference);
        child.parentElement = this;
        children.splice(index < 0 ? children.length : index, 0, child);
      },
      remove() {
        if (this.parentElement === null) return;
        const siblings = this.parentElement.children;
        siblings.splice(siblings.indexOf(this), 1);
        this.parentElement = null;
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
      addEventListener() {},
      getBoundingClientRect() {
        return { left: 0, top: 0, right: 1, bottom: 1, width: 1, height: 1 };
      },
    };
    return node;
  };
  const document = {
    createElement: create,
    /** @param {string} _namespace @param {string} tagName */
    createElementNS: (_namespace, tagName) => create(tagName),
  };
  ownerDocument = document;
  /** @param {any} root @returns {any[]} */
  const descendants = (root) => [
    root,
    ...root.children.flatMap((/** @type {any} */ child) => descendants(child)),
  ];
  /** @param {any} root */
  const text = (root) =>
    descendants(root)
      .map((node) => node.textContent)
      .filter(Boolean)
      .join(" ");
  return { create, document, descendants, text };
}

/**
 * @template T
 * @param {ReturnType<typeof domHarness>} dom
 * @param {(panels: DebuggerPanels, nodes: Record<string, any>) => Promise<T>} run
 */
async function withPanels(dom, run) {
  const saved = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: dom.document,
  });
  try {
    /** @type {Record<string, any>} */
    const nodes = {};
    for (const name of [
      "roster",
      "rosterCount",
      "selectionCard",
      "pendingHeading",
      "pendingCount",
      "pendingScope",
      "pendingCard",
      "acceptedCard",
      "acceptedAnnouncement",
      "diagnosticsCard",
      "actionsPanel",
    ]) {
      nodes[name] = dom.create("div");
    }
    const panels = new DebuggerPanels(
      /** @type {any} */ ({ ...nodes, onCommand: () => undefined }),
    );
    return await run(panels, nodes);
  } finally {
    if (saved) Object.defineProperty(globalThis, "document", saved);
    else Reflect.deleteProperty(globalThis, "document");
  }
}

test("the movement fallback matches the host's movement names", async () => {
  const core = await readFile(
    new URL("../../../src/marl_battlegrounds/core/axis_mappings.py", import.meta.url),
    "utf8",
  );
  const block = core.match(/MOVEMENT_ACTION_NAME_BY_ID[^=]*=\s*\(([^)]*)\)/u);
  assert.ok(block);
  const names = [...block[1].matchAll(/"([^"]+)"/gu)].map((match) => match[1]);
  assert.deepEqual([...FALLBACK_MOVEMENT_NAMES], names);
});

test("recorded actions read in words over their codes", () => {
  const map = [null, "a0", "a1", "a2"];
  const blast = describeRecordedAction(
    { move_action: 6, target_action: 2, use_ultimate_action: 0 },
    1,
    map,
    "a0",
  );
  assert.equal(blast.move, "Moves Northwest");
  assert.equal(blast.ability, "Arcane Blast");
  assert.equal(blast.targetPublicId, "a1");
  assert.equal(blast.itself, false);
  assert.equal(blast.codes, "Move 6 · Target 2 · Ultimate 0");

  const burst = describeRecordedAction(
    { move_action: 0, target_action: 0, use_ultimate_action: 1 },
    1,
    map,
    "a0",
  );
  assert.equal(burst.move, "Stays");
  assert.equal(burst.ability, "Burst");
  assert.equal(burst.noCombat, false);

  const idle = describeRecordedAction(
    { move_action: 3, target_action: 0, use_ultimate_action: 0 },
    5,
    map,
    "a0",
  );
  assert.equal(idle.move, "Moves East");
  assert.equal(idle.noCombat, true);
  assert.equal(idle.ability, null);

  const heal = describeRecordedAction(
    { move_action: 0, target_action: 1, use_ultimate_action: 0 },
    5,
    map,
    "a0",
  );
  assert.equal(heal.ability, "Lesser Heal");
  assert.equal(heal.itself, true);

  const unmapped = describeRecordedAction(
    { move_action: 1, target_action: 8, use_ultimate_action: 0 },
    3,
    null,
    "a0",
  );
  assert.equal(unmapped.targetPublicId, null);
  assert.equal(unmapped.targetCode, 8);
  assert.equal(unmapped.ability, "Serrated Arrows");
});

test("panel style follows Panel Tokens and falls back to the defaults", async () => {
  const replay = await presentation("replay_oracle");
  const live = await presentation("live_oracle");
  assert.deepEqual(panelDisplayStyle(replay), { tokens: true });
  assert.deepEqual(panelDisplayStyle(live), { tokens: true });
  const plain = setVisualFilterEnabled(
    DEFAULT_VISUAL_FILTER_STATE,
    "panel_tokens",
    false,
  );
  assert.deepEqual(panelDisplayStyle(replay, plain), { tokens: false });
  assert.deepEqual(panelDisplayStyle(live, plain), { tokens: false });
  assert.deepEqual(panelDisplayStyle(replay, { broken: true }), { tokens: true });
});

test("only the Replay Viewer's Agent POV panels carry the Full Record chip", async () => {
  const replayPov = await presentation("replay_no_shared_obs_agent_pov");
  const recipient = authorizedPresentationIdentityRows(replayPov).find(
    (row) => row.public_agent_id === replayPov.authority.recipient_public_agent_id,
  );
  assert.ok(recipient);
  assert.equal(
    fullRecordLabel(replayPov),
    `Full Record · Not Limited To Agent ${recipient.agent.display_agent_id}'s View`,
  );
  assert.equal(fullRecordLabel(await presentation("replay_oracle")), null);
  assert.equal(
    fullRecordLabel(await presentation("live_no_shared_obs_agent_pov")),
    null,
  );

  const dom = domHarness();
  await withPanels(dom, async (panels, nodes) => {
    panels.render(replayPov, {});
    const chips = dom
      .descendants(nodes.roster)
      .filter((node) => node.className === "panel-full-record");
    assert.equal(chips.length, 1);
    assert.equal(chips[0].hidden, false);
    assert.match(dom.text(nodes.actionsPanel), /^Full Record · /u);
    assert.match(
      dom.text(nodes.actionsPanel),
      /This Tick \(What Happened\).*Next Tick \(What Happens Next\)/u,
    );
    panels.render(await presentation("replay_oracle"), {});
    assert.equal(chips[0].hidden, true);
  });
});

test("BattleClient Agent POV panels show unseen champions as Unseen only", async () => {
  const live = await presentation("live_no_shared_obs_agent_pov");
  const seenIds = new Set(
    live.scene.agents.map(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id,
    ),
  );
  const unseen = authorizedPresentationIdentityRows(live).find(
    (row) => !seenIds.has(row.public_agent_id),
  );
  assert.ok(unseen);
  const dom = domHarness();
  await withPanels(dom, async (panels, nodes) => {
    panels.render(live, { activatedAgentPublicId: unseen.public_agent_id });
    const rows = [...panels.rosterRows.values()];
    const unseenRows = rows.filter((row) => row.element.dataset.seen === "false");
    assert.equal(
      unseenRows.length,
      authorizedPresentationIdentityRows(live).length - seenIds.size,
    );
    for (const row of unseenRows) {
      assert.equal(row.health.textContent, "Unseen");
      assert.equal(row.statuses.children.length, 0);
      assert.equal(row.modifiers.children.length, 0);
      assert.doesNotMatch(dom.text(row.element), /Life|Ultimate|\d+ \/ \d+/u);
    }
    for (const row of rows.filter(
      (candidate) => candidate.element.dataset.seen !== "false",
    )) {
      // A seen row's life is the shared Life bar ("Life", the bar, "200 / 200").
      assert.match(dom.text(row.health), /^(?:Life \d+ +\/ \d+|Fallen)/u);
    }
    const details = dom.text(nodes.selectionCard);
    assert.match(details, /Out Of Sight/u);
    assert.match(details, /Unseen/u);
    assert.doesNotMatch(details, /Life|Right Now|Statuses|Class Overview/u);
  });
});

test("BattleClient Agent POV pending card shows only the POV agent's own draft", async () => {
  const raw = structuredClone(fixture.presentations.live_shared_obs_agent_pov);
  const recipientId = raw.authority.recipient_public_agent_id;
  const step = { armed_lane: "none", move_action: 2, target_action: 0 };
  // The normalizer requires the researcher copy of the POV draft to match.
  raw.live_inspection.inspection.draft.draft_action = { ...step };
  raw.researcher_space.pending_inspection.draft.draft_action = { ...step };
  for (const row of raw.researcher_space.pending_joint_action.action_rows) {
    if (row.actor_public_agent_id === recipientId) row.pending_action.move_action = 2;
  }
  // The researcher branch says something else for every other actor: none of
  // it may reach the card.
  const scrambled = structuredClone(raw);
  for (const row of scrambled.researcher_space.pending_joint_action.action_rows) {
    if (row.actor_public_agent_id === recipientId) continue;
    row.pending_action = { move_action: 3, target_action: 4, use_ultimate_action: 1 };
  }
  const live = await normalizeAuthorizedPresentationFrameV1(raw);
  const changed = await normalizeAuthorizedPresentationFrameV1(scrambled);
  const dom = domHarness();
  /** @type {string[]} */
  const texts = [];
  await withPanels(dom, async (panels, nodes) => {
    for (const frame of [live, changed]) {
      panels.renderAuthorizedInspector(frame);
      const rows = dom
        .descendants(nodes.pendingCard)
        .filter((node) => node.className === "accepted-action-row");
      assert.equal(rows.length, 10);
      assert.equal(nodes.pendingCount.textContent, "10 actors");
      const observed = rows.filter((row) => row.dataset.observed !== "false");
      assert.equal(observed.length, 1);
      assert.match(dom.text(observed[0]), /Pending/u);
      assert.match(dom.text(observed[0]), /Move 2/u);
      const hidden = rows.filter((row) => row.dataset.observed === "false");
      assert.equal(hidden.length, 9);
      assert.equal(hidden.filter((row) => row.dataset.team === "team-a").length, 4);
      for (const row of hidden) {
        assert.match(dom.text(row), /Not Observed/u);
        assert.doesNotMatch(
          dom.text(row),
          /Pending|East|Move \d|Target \d|Ultimate \d/u,
        );
      }
      assert.doesNotMatch(dom.text(nodes.pendingCard), /Move 3|Target 4|Ultimate 1/u);
      texts.push(dom.text(nodes.pendingCard));
    }
  });
  assert.equal(texts[0], texts[1]);
  assert.equal(typeof recipientId, "string");
});

test("roster status icons follow the Status Icons From Marks switch", async () => {
  const off = setVisualFilterEnabled(
    DEFAULT_VISUAL_FILTER_STATE,
    "status_marks_icons",
    false,
  );
  assert.equal(statusIconsFromMarks(), true);
  assert.equal(statusIconsFromMarks(off), false);
  assert.equal(statusIconsFromMarks({ broken: true }), true);
  const live = await presentation("live_oracle");
  const dom = domHarness();
  await withPanels(dom, async (panels, nodes) => {
    /** @param {unknown} state */
    const iconStyles = (state) => {
      panels.renderAuthorizedRoster(live, false, undefined, null, state);
      return dom
        .descendants(nodes.roster)
        .filter(
          (node) =>
            node.tagName === "SVG" &&
            String(node.parentElement?.className).includes("roster-fact-token--status"),
        )
        .map((node) => node.dataset.iconStyle ?? "plain");
    };
    const marks = iconStyles(DEFAULT_VISUAL_FILTER_STATE);
    assert.ok(marks.length > 0);
    assert.ok(marks.includes("sigil"));
    const plain = iconStyles(off);
    assert.equal(plain.length, marks.length);
    assert.ok(plain.every((style) => style !== "sigil"));
  });
});
