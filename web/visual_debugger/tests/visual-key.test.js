/**
 * @file Check the Visual Key (src/looks/visual-key.js): one entry per symbol in
 * the symbol list's group order, Battle Cam (B) and Fighters Standing only in
 * the Replay Viewer (the live scorebug has no standing count), the
 * Camera And Keys rows from command-set.js, the approved status names (a mark
 * that serves two causes, such as the stun stars, lists both: Charge (Stun)
 * and Crippling Poison (Stun)), numbers read only from the static
 * mechanics catalogue (class, status, aura, Spawn Shield, Red Zone depth,
 * respawn period; missing numbers left out, never invented, no live values);
 * pictures drawn by a look's visualKeyExample hook first, then by the real look
 * modules' records() and build() on fixed example data (a status picture is
 * the Statuses look's marks over the Champions look's token), then a drawn
 * sign;
 * every arena symbol (events too: Basics, Ultimates, life numbers, Deaths And
 * Returns, Start Spot, Last Seen, auras) and the Whole Map drawn by a real look,
 * with signs left only for the HTML HUD Marks and the V, B and F key caps;
 * export-safe SVG only; the same Key in Oracle View and Agent POV; and the
 * controller (always-visible book button, modal dialog, Close, rebuild when the
 * numbers change, teardown).
 */
import assert from "node:assert/strict";
import test from "node:test";

import { commandHelpRows } from "../src/command-set.js";
import {
  build,
  createVisualKey,
  drawSymbolPicture,
  records,
  symbolNumbers,
  tooltip,
} from "../src/looks/visual-key.js";
import * as DEFAULT_BASICS from "../src/looks/basics.js";
import { SYMBOL_GROUPS, SYMBOLS, symbolById } from "../src/symbols.js";

class FakeNode {
  /** @param {string} tagName @param {string | null} namespace @param {any} ownerDocument */
  constructor(tagName, namespace, ownerDocument) {
    this.tagName = tagName.toUpperCase();
    this.localName = tagName;
    this.namespaceURI = namespace;
    this.ownerDocument = ownerDocument;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeNode[]} */
    this.children = [];
    /** @type {FakeNode | null} */
    this.parentNode = null;
    this.ownText = "";
    this.hidden = false;
    this.open = false;
    this.id = "";
    this.type = "";
    this.scope = "";
    /** @type {Record<string, string>} */
    this.dataset = {};
    /** @type {Record<string, string>} */
    this.style = {};
    /** @type {Map<string, Set<Function>>} */
    this.listeners = new Map();
  }
  get className() {
    return this.getAttribute("class") ?? "";
  }
  set className(value) {
    this.setAttribute("class", value);
  }
  get parentElement() {
    return this.parentNode;
  }
  getBoundingClientRect() {
    return { left: 0, top: 0, right: 10, bottom: 10, width: 10, height: 10 };
  }
  /** @returns {FakeNode | null} */
  get lastElementChild() {
    return this.children.filter((child) => child.tagName !== "#TEXT").at(-1) ?? null;
  }
  get childElementCount() {
    return this.children.length;
  }
  get firstChild() {
    return this.children[0] ?? null;
  }
  /** @returns {FakeNode | null} */
  get lastChild() {
    return this.children.at(-1) ?? null;
  }
  /** @returns {string} */
  get textContent() {
    return this.ownText + this.children.map((child) => child.textContent).join("");
  }
  set textContent(value) {
    this.children = [];
    this.ownText = String(value);
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
  hasAttribute(name) {
    return this.attributes.has(name);
  }
  /** @param {string} name */
  removeAttribute(name) {
    this.attributes.delete(name);
  }
  /** @param {...(FakeNode | string)} nodes */
  append(...nodes) {
    for (const item of nodes) {
      const node =
        typeof item === "string"
          ? Object.assign(new FakeNode("#text", null, this.ownerDocument), {
              ownText: item,
            })
          : item;
      node.remove();
      node.parentNode = this;
      this.children.push(node);
    }
  }
  /** @param {FakeNode} node */
  appendChild(node) {
    this.append(node);
    return node;
  }
  /** @param {...FakeNode} nodes */
  prepend(...nodes) {
    for (const node of nodes.reverse()) {
      node.remove();
      node.parentNode = this;
      this.children.unshift(node);
    }
  }
  /** @param {...FakeNode} nodes */
  replaceChildren(...nodes) {
    for (const child of this.children) child.parentNode = null;
    this.children = [];
    this.ownText = "";
    this.append(...nodes);
  }
  /** @param {FakeNode} node */
  replaceWith(node) {
    const parent = this.parentNode;
    if (parent === null) return;
    node.remove();
    parent.children.splice(parent.children.indexOf(this), 1, node);
    node.parentNode = parent;
    this.parentNode = null;
  }
  /** @param {FakeNode} node @param {FakeNode | null} before */
  insertBefore(node, before) {
    node.remove();
    const index = before === null ? -1 : this.children.indexOf(before);
    node.parentNode = this;
    if (index < 0) this.children.push(node);
    else this.children.splice(index, 0, node);
    return node;
  }
  remove() {
    if (this.parentNode !== null) {
      this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1);
      this.parentNode = null;
    }
  }
  /** @param {string} selector */
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }
  /** @param {string} selector */
  querySelectorAll(selector) {
    const [tag, className] = selector.split(".");
    return this.descendants().filter(
      (node) =>
        (tag === "" || node.localName === tag) &&
        (className === undefined || node.className.split(" ").includes(className)),
    );
  }
  /** @returns {FakeNode[]} */
  descendants() {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }
  /** @param {string} type @param {Function} listener */
  addEventListener(type, listener) {
    this.listeners.set(type, (this.listeners.get(type) ?? new Set()).add(listener));
  }
  /** @param {string} type @param {Function} listener */
  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }
  /** @param {string} type */
  fire(type) {
    for (const listener of [...(this.listeners.get(type) ?? [])])
      listener({ target: this });
  }
  showModal() {
    this.open = true;
  }
  close() {
    this.open = false;
    this.fire("close");
  }
  focus() {
    this.ownerDocument.focused = this;
  }
}

function fakeDocument() {
  /** @type {any} */
  const doc = { focused: null };
  doc.createElement = (/** @type {string} */ tag) => new FakeNode(tag, null, doc);
  doc.createElementNS = (/** @type {string} */ namespace, /** @type {string} */ tag) =>
    new FakeNode(tag, namespace, doc);
  doc.body = new FakeNode("body", null, doc);
  return doc;
}

const CATALOGUE = Object.freeze({
  classMechanics: [
    {
      class_id: 1,
      body_radius: 0.5,
      observation_radius: 6,
      maximum_health: 80,
      basic_raw_damage: 13,
      status_mechanics: [
        {
          status_id: "mage_burst_damage_amplification",
          magnitude: 1.5,
          duration_steps: 5,
        },
      ],
      aura_mechanics: [
        {
          aura_id: "mage_damage_amplification",
          radius: 2,
          per_emitter_multiplier: 1.15,
          clamp_value: 1.3225,
        },
      ],
    },
    {
      class_id: 3,
      body_radius: 0.5,
      observation_radius: 7.5,
      maximum_health: 100,
      status_mechanics: [
        {
          status_id: "hunter_trap_stun",
          duration_steps: 2,
          breaks_on_positive_damage: true,
        },
      ],
    },
  ],
  spawnShieldMechanics: { configured_duration_steps: 3, movement_speed: 2 },
  redZone: { depth: 2.5, team_a_x_range: [0, 2.5], team_b_x_range: [17.5, 20] },
  respawnWaves: [{ period_steps: 10, countdown_steps: 4 }],
});

/** @param {ReadonlyArray<Record<string, any>>} list @param {string} id */
function entry(list, id) {
  const found = list.find((record) => record.symbolId === id);
  assert.ok(found, id);
  return found;
}

test("the Key lists every symbol in group order and the client's own keys", () => {
  assert.deepEqual(records({}), []);
  const replay = records({ product: "replay_viewer" });
  const combat = records({ product: "combat_debugger" });
  const entries = replay.filter((record) => record.kind === "visual_key_entry");
  assert.equal(entries.length, SYMBOLS.length);
  const groupOrder = [...new Set(entries.map((record) => record.group))];
  assert.deepEqual(groupOrder, [...SYMBOL_GROUPS]);
  assert.equal(entry(replay, "slow_snagged").title, "Serrated Arrows (Slow)");
  // One mark that serves two causes lists both approved names under it.
  assert.deepEqual(entry(replay, "stun_sapped").names, [
    "Charge (Stun)",
    "Crippling Poison (Stun)",
  ]);
  assert.deepEqual(entry(replay, "slow_hamstrung").names, [
    "Charge (Slow)",
    "Crippling Poison (Slow)",
  ]);
  assert.equal(entry(replay, "red_zone").title, "Red Zone");
  assert.equal(entry(replay, "camera_battle_cam").key, "B");
  assert.equal(
    combat.some((record) => record.symbolId === "camera_battle_cam"),
    false,
    "Battle Cam is Replay Viewer only",
  );
  assert.ok(replay.some((record) => record.symbolId === "fighters_standing"));
  assert.equal(
    combat.some((record) => record.symbolId === "fighters_standing"),
    false,
    "Fighters Standing is Replay Viewer only",
  );
  assert.ok(combat.some((record) => record.symbolId === "frame_explanation"));
  assert.match(
    entry(replay, "sight_veil").eli5,
    /Stone also blocks sight inside the circle/u,
  );
  for (const product of /** @type {const} */ (["replay_viewer", "combat_debugger"])) {
    const keys = records({ product }).filter(
      (record) => record.kind === "visual_key_command",
    );
    assert.deepEqual(
      keys.map((record) => [record.keycaps, record.help]),
      commandHelpRows(product).map((row) => [row.keycaps, row.help]),
    );
    assert.ok(keys.every((record) => record.group === "Camera And Keys"));
  }
});

test("numbers come only from the static catalogue and are never invented", () => {
  const list = records({ product: "replay_viewer", catalogue: CATALOGUE });
  /** @param {string} id */
  const numbers = (id) =>
    Object.fromEntries(
      entry(list, id).numbers.map((/** @type {any} */ row) => [row.label, row.value]),
    );
  assert.deepEqual(numbers("class_hunter"), {
    "Full life": "100",
    "How far it sees": "7.5",
  });
  assert.equal(numbers("stun_frozen")["Whether damage breaks it early"], "Yes");
  assert.equal(numbers("stun_frozen")["Ticks the stun lasts"], "2");
  assert.equal(
    numbers("aura_empowerment")["Highest multiplier when auras overlap"],
    "1.32",
  );
  assert.equal(numbers("spawn_shield")["Ticks the shield lasts"], "3");
  assert.equal(numbers("red_zone")["How deep each Red Zone reaches"], "2.5");
  assert.equal(numbers("respawn")["Ticks between waves"], "10");
  assert.deepEqual(
    numbers("champion"),
    { "Body size (radius)": "0.5" },
    "only values every class shares",
  );
  assert.deepEqual(
    numbers("class_priest"),
    {},
    "a class missing from the catalogue shows no numbers",
  );
  assert.deepEqual(numbers("life_bar"), {}, "live values never appear in the Key");
  assert.deepEqual(numbers("score"), {});
  assert.deepEqual(
    symbolNumbers(/** @type {any} */ (symbolById("class_mage")), {}),
    [],
  );
  const oracle = records({ product: "replay_viewer", catalogue: CATALOGUE });
  assert.deepEqual(
    records({ product: "replay_viewer", catalogue: CATALOGUE }),
    oracle,
    "the Key reads no view, so Oracle View and Agent POV give the same Key",
  );
});

test("pictures: a look's own example first, then the real look, then a sign", () => {
  const doc = fakeDocument();
  /** @type {Array<Record<string, any>>} */
  const inputs = [];
  const fakeChampions = {
    /** @param {Record<string, any>} input */
    records(input) {
      inputs.push(input);
      return [{ kind: "champion", agent: input.scene.agents[0] }];
    },
    /** @param {any[]} list @param {any} document */
    build(list, document) {
      return list.map(() => ({
        wrapper: document.createElementNS("http://www.w3.org/2000/svg", "g"),
      }));
    },
    update() {},
  };
  const example = {
    visualKeyExample: (/** @type {string} */ id, /** @type {any} */ context) =>
      id === "basic_mage"
        ? [context.doc.createElementNS("http://www.w3.org/2000/svg", "path")]
        : null,
  };
  /** @type {Array<Record<string, any>>} */
  const statusInputs = [];
  const fakeStatuses = {
    /** @param {Record<string, any>} input */
    records(input) {
      statusInputs.push(input);
      return [{ kind: "status-marks" }];
    },
    /** @param {any[]} list @param {any} document */
    build(list, document) {
      return list.map(() =>
        document.createElementNS("http://www.w3.org/2000/svg", "g"),
      );
    },
    update() {},
  };
  const looks = { champions: fakeChampions, basics: example, statuses: fakeStatuses };
  const hunter = /** @type {any} */ (drawSymbolPicture("class_hunter", doc, looks));
  assert.equal(hunter.getAttribute("data-picture-source"), "look:champions");
  assert.equal(hunter.getAttribute("aria-label"), "Hunter (The Sentinel)");
  assert.equal(inputs[0].scene.agents[0].class_id, 3);
  drawSymbolPicture("team_b", doc, looks);
  assert.equal(inputs[1].scene.agents[0].team_id, 2);
  const snagged = /** @type {any} */ (drawSymbolPicture("slow_snagged", doc, looks));
  assert.equal(snagged.getAttribute("data-picture-source"), "look:statuses");
  assert.equal(statusInputs[0].part, "status-marks");
  assert.equal(
    statusInputs[0].scene.agents[0].statuses[0].status_id,
    "hunter_basic_slow",
  );
  // The champion under the status marks is drawn by the Champions look.
  assert.equal(inputs[2].scene.agents[0].statuses[0].status_id, "hunter_basic_slow");
  const basic = /** @type {any} */ (drawSymbolPicture("basic_mage", doc, looks));
  assert.equal(basic.getAttribute("data-picture-source"), "example:basics");
  const broken = /** @type {any} */ (
    drawSymbolPicture("champion", doc, {
      champions: {
        records() {
          throw new Error("not ready");
        },
        build() {
          return [];
        },
      },
    })
  );
  assert.equal(broken.getAttribute("data-picture-source"), "sign");
  const key = /** @type {any} */ (
    drawSymbolPicture("camera_everyone_in_view", doc, looks)
  );
  assert.equal(key.getAttribute("data-picture-source"), "sign");
  assert.match(key.textContent, /V/u);
});

test("every picture drawn by the real looks is export-safe SVG", () => {
  const doc = fakeDocument();
  /** @type {Record<string, number>} */
  const sources = {};
  for (const symbol of SYMBOLS) {
    const picture = /** @type {any} */ (drawSymbolPicture(symbol.id, doc));
    const source = String(picture.getAttribute("data-picture-source"));
    sources[source] = (sources[source] ?? 0) + 1;
    for (const node of [picture, ...picture.descendants()]) {
      if (node.tagName === "#TEXT") continue;
      assert.equal(node.namespaceURI, "http://www.w3.org/2000/svg", symbol.id);
      assert.ok(
        ![
          "LINEARGRADIENT",
          "RADIALGRADIENT",
          "MASK",
          "FILTER",
          "IMAGE",
          "PATTERN",
        ].includes(node.tagName),
        `${symbol.id} ${node.tagName}`,
      );
      for (const [name, value] of node.attributes) {
        assert.doesNotMatch(`${name}=${value}`, /url\(|^filter=|^mask=/iu, symbol.id);
      }
    }
  }
  assert.ok((sources["look:champions"] ?? 0) >= 5, JSON.stringify(sources));
  for (const [id, source] of [
    ["class_rogue", "look:champions"],
    ["stone", "look:obstacles"],
    ["red_zone", "look:red-zone"],
  ]) {
    assert.equal(
      /** @type {any} */ (drawSymbolPicture(id, doc)).getAttribute(
        "data-picture-source",
      ),
      source,
      id,
    );
  }
});

test("every arena symbol is drawn by a real look; only HUD Marks and V, B, F are signs", () => {
  const doc = fakeDocument();
  /** @type {Record<string, string>} */
  const expected = {
    blow: "look:basics",
    basic_mage: "look:basics",
    basic_warrior: "look:basics",
    basic_hunter: "look:basics",
    basic_rogue: "look:basics",
    basic_priest: "look:basics",
    ult_mage: "look:ultimates",
    ult_warrior: "look:ultimates",
    ult_hunter: "look:ultimates",
    ult_rogue: "look:ultimates",
    ult_priest: "look:ultimates",
    damage_number: "look:health-text",
    heal_number: "look:health-text",
    fall: "look:requiems",
    fallen: "look:requiems",
    returns_in: "look:requiems",
    respawn: "look:requiems",
    imprint: "look:movement",
    last_seen: "look:fog",
    waystone: "look:spawns",
    aura_empowerment: "look:statuses",
    aura_barrier: "look:statuses",
    camera_whole_map: "look:obstacles",
  };
  /** @type {string[]} */
  const signs = [];
  for (const symbol of SYMBOLS) {
    const picture = /** @type {any} */ (drawSymbolPicture(symbol.id, doc));
    const source = String(picture.getAttribute("data-picture-source"));
    if (expected[symbol.id] !== undefined) {
      assert.equal(source, expected[symbol.id], symbol.id);
    }
    if (source === "sign") {
      signs.push(symbol.id);
      continue;
    }
    // An event picture draws its example champions under the event.
    assert.ok(picture.descendants().length > 1, symbol.id);
  }
  assert.deepEqual(
    signs,
    SYMBOLS.filter(
      (symbol) =>
        symbol.group === "HUD Marks" || ["V", "B", "F"].includes(String(symbol.key)),
    ).map((symbol) => symbol.id),
  );
  const basic = /** @type {any} */ (drawSymbolPicture("basic_hunter", doc));
  const eventOnly = /** @type {any} */ (
    drawSymbolPicture("basic_hunter", doc, { basics: DEFAULT_BASICS })
  );
  assert.equal(eventOnly.getAttribute("data-picture-source"), "look:basics");
  assert.equal(
    basic.children.length - eventOnly.children.length,
    2,
    "two example champions are drawn under the Basic",
  );
});

test("cards hold the picture, title, sentence, key cap and numbers; tips name the symbol", () => {
  const doc = fakeDocument();
  const list = records({ product: "replay_viewer", catalogue: CATALOGUE });
  assert.deepEqual(build(list, null), []);
  const sections = /** @type {any[]} */ (build(list, doc));
  assert.deepEqual(
    sections.map((section) => section.getAttribute("data-group")),
    [...SYMBOL_GROUPS],
  );
  const camera = sections.at(-1);
  const card = camera
    .querySelectorAll("article")
    .find(
      (/** @type {any} */ node) =>
        node.getAttribute("data-symbol-id") === "camera_battle_cam",
    );
  assert.ok(card);
  assert.equal(card.children[0].localName, "svg");
  assert.match(card.textContent, /Battle Cam/u);
  assert.equal(card.querySelector("kbd").textContent, "B");
  assert.equal(camera.querySelectorAll("table").length, 1);
  const hunter = sections[1]
    .querySelectorAll("article")
    .find(
      (/** @type {any} */ node) =>
        node.getAttribute("data-symbol-id") === "class_hunter",
    );
  assert.match(hunter.querySelector("dl").textContent, /How far it sees7\.5/u);
  const descriptor = tooltip(entry(list, "red_zone"));
  assert.equal(descriptor?.kind, "symbol");
  assert.equal(descriptor?.title, "Red Zone");
  assert.equal(tooltip({ kind: "anything" }), null);
});

test("the controller shows the book button always and opens a modal dictionary", () => {
  const doc = fakeDocument();
  const button = /** @type {any} */ (doc.createElement("button"));
  button.hidden = true;
  let catalogue = /** @type {any} */ (CATALOGUE);
  let reads = 0;
  const key = createVisualKey({
    button,
    product: "combat_debugger",
    getCatalogue: () => {
      reads += 1;
      return catalogue;
    },
    doc,
  });
  assert.equal(button.hidden, false);
  assert.equal(button.getAttribute("aria-haspopup"), "dialog");
  assert.equal(button.querySelectorAll("svg.visual-key-book").length, 1);
  button.fire("click");
  assert.equal(key.isOpen(), true);
  const dialog = doc.body.children[0];
  assert.equal(dialog.localName, "dialog");
  assert.equal(dialog.className, "visual-key-dialog");
  const body = dialog.children[1];
  const firstCards = body.querySelectorAll("article").length;
  assert.equal(
    firstCards,
    SYMBOLS.length - 2,
    "no Battle Cam and no Fighters Standing in the BattleClient",
  );
  const close = dialog.children[0].querySelector("button");
  close.fire("click");
  assert.equal(key.isOpen(), false);
  assert.equal(doc.focused, button, "focus returns to the button");
  const firstSection = body.children[0];
  key.open();
  assert.equal(body.children[0], firstSection, "same numbers: the content is kept");
  key.close();
  catalogue = { ...CATALOGUE, redZone: { depth: 3 } };
  key.open();
  assert.notEqual(
    body.children[0],
    firstSection,
    "new numbers: the content is rebuilt",
  );
  assert.equal(reads, 3);
  key.destroy();
  assert.equal(doc.body.children.length, 0);
  button.fire("click");
  assert.equal(doc.body.children.length, 0, "destroy removes the listener");
  assert.throws(() => createVisualKey(/** @type {any} */ ({})), TypeError);
  assert.throws(
    () => createVisualKey(/** @type {any} */ ({ button, product: "nope", doc })),
    RangeError,
  );
});
