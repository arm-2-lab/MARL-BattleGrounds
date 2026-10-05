/**
 * @file Check the corpse lift of a return (src/looks/requiems.js). records()
 * gives a return's "homecoming" record an oldCorpse only from the audience's
 * own plan: the returning body's START spot (so an Agent POV corpse the team
 * did not see when the tick began has none), and none with Resurrection
 * Effects off or without a plan. The copy repeats the Champions look's Ashen
 * Token exactly (disc, rims, emblem box, and the Fallen Step Aside coin when a
 * standing champion covered the corpse at the start of the tick), with the
 * ash relic and the last Vigil Pip. build() draws it in the Strike Band
 * (connector) group at END opacity 0, shown from the start of the tick and
 * faded over 200 ms from the Return beat (160 ms in reduced motion), plus an
 * Ash lift ring that appears at the beat at 50% and fades over 300 ms while it
 * grows to 1.2 times (no growth in reduced motion); a settled mount draws
 * neither. update() moves both to the new corpse spot. Export-safe SVG.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createViewportTransform } from "../src/layout.js";
import * as champions from "../src/looks/champions.js";
import * as requiems from "../src/looks/requiems.js";

class FakeElement {
  /** @param {FakeDocument} doc @param {string} tagName */
  constructor(doc, tagName) {
    this.ownerDocument = doc;
    this.tagName = tagName;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    /** @type {FakeElement | null} */
    this.parentElement = null;
    /** @type {Record<string, string>} */
    this.dataset = {};
    this.textContent = "";
    this.classList = { add: () => {} };
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
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    for (const node of nodes) {
      node.remove();
      node.parentElement = this;
      this.children.push(node);
    }
  }
  /** @param {...FakeElement} nodes */
  replaceChildren(...nodes) {
    for (const child of this.children) child.parentElement = null;
    this.children = [];
    this.append(...nodes);
  }
  /** @param {FakeElement} node */
  replaceWith(node) {
    const parent = this.parentElement;
    if (!parent) return;
    const index = parent.children.indexOf(this);
    node.remove();
    parent.children[index] = node;
    node.parentElement = parent;
    this.parentElement = null;
  }
  remove() {
    if (this.parentElement) {
      this.parentElement.children = this.parentElement.children.filter(
        (child) => child !== this,
      );
      this.parentElement = null;
    }
  }
  /** @returns {FakeElement[]} */
  descendants() {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }
  /** @param {string} selector */
  querySelectorAll(selector) {
    const name = selector.replace(/^\./u, "");
    return this.descendants().filter((node) =>
      (node.getAttribute("class") ?? "").split(" ").includes(name),
    );
  }
  /** @param {string} selector */
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }
  addEventListener() {}
  blur() {}
  getBoundingClientRect() {
    return { x: 0, y: 0, width: 0, height: 0, left: 0, top: 0, right: 0, bottom: 0 };
  }
}

class FakeDocument {
  constructor() {
    this.activeElement = null;
  }
  /** @param {string} _namespace @param {string} tagName */
  createElementNS(_namespace, tagName) {
    return new FakeElement(this, tagName);
  }
}

const transform = createViewportTransform({
  worldWidth: 20,
  worldHeight: 10,
  viewportWidth: 1000,
  viewportHeight: 500,
  padding: 0,
});

/** @param {FakeElement} node @returns {string} */
function serialize(node) {
  const attributes = [...node.attributes.entries()]
    .map(([name, value]) => ` ${name}="${value}"`)
    .join("");
  return `<${node.tagName}${attributes}>${node.children.map(serialize).join("")}</${node.tagName}>`;
}

/**
 * @param {Record<string, any>} [extra]
 */
function returnPulse(extra = {}) {
  return {
    kind: "semantic_pulse",
    cueSemantic: "agent_respawned",
    eventId: "back-w",
    anchor: { x: 950, y: 250 },
    anchorRadius: 25,
    phaseStart: 900,
    phaseEnd: 2400,
    recipientPresentationKey: "pk-w",
    recipientPublicAgentId: "7",
    recipientIdentity: { class_id: 2, team_id: 2, public_agent_id: "7" },
    ...extra,
  };
}

/**
 * @param {{start?: number[] | null, mate?: number[] | null}} [options]
 */
function planWith(options = {}) {
  const bodies = [
    {
      presentationKey: "pk-w",
      publicAgentId: "7",
      teamId: 2,
      alive: true,
      start: options.start === undefined ? [6, 5] : options.start,
      end: [19, 5],
      died: false,
      returned: true,
    },
  ];
  if (options.mate) {
    bodies.push({
      presentationKey: "pk-m",
      publicAgentId: "8",
      teamId: 2,
      alive: true,
      start: options.mate,
      end: options.mate,
      died: false,
      returned: false,
    });
  }
  return { bodies, derivedEvents: [] };
}

/** @param {{settled?: boolean, motionMode?: "normal" | "reduced" | "off"}} [options] */
function mountFor(options = {}) {
  const doc = new FakeDocument();
  return {
    doc,
    mount: {
      group: doc.createElementNS("", "g"),
      connector: doc.createElementNS("", "g"),
      underlay: null,
      plan: {
        epochKey: "epoch",
        phases: { total: 1500, reducedTotal: 375 },
        beats: { reducedScale: 0.25 },
      },
      options: {
        motionMode: options.motionMode ?? "normal",
        settled: options.settled ?? false,
      },
      animationSpecs: /** @type {any[]} */ ([]),
    },
  };
}

/**
 * @param {number[] | null} mate
 */
function championCorpse(mate) {
  const agents = [
    {
      display_key: "w",
      class_id: 2,
      team_id: 2,
      alive: false,
      life_state: "corpse",
      position: [6, 5],
      radius: 0.5,
    },
  ];
  if (mate) {
    agents.push({
      display_key: "m",
      class_id: 1,
      team_id: 2,
      alive: true,
      life_state: "alive",
      position: mate,
      radius: 0.5,
    });
  }
  const rows = champions.records({ scene: { agents }, transform });
  const doc = /** @type {any} */ (new FakeDocument());
  const nodes = champions.build(rows, doc);
  champions.update(nodes, rows);
  return { record: rows[0], fallen: nodes[0].fallen, body: nodes[0].body };
}

test("only a seen corpse of the audience's own plan gets an old corpse", () => {
  const plan = planWith();
  const [record] = requiems.records({ event: returnPulse(), plan, transform });
  assert.equal(record.kind, "homecoming");
  const corpse = record.oldCorpse;
  assert.ok(corpse);
  assert.deepEqual(corpse.center, transform.worldToScreen([6, 5]));
  assert.equal(corpse.radius, 25);
  assert.equal(corpse.teamKey, "b");
  assert.equal(corpse.classKey, "warrior");
  assert.equal(corpse.aside, null);
  assert.equal(corpse.relic, true);
  assert.equal(corpse.pips, 1);
  // Agent POV gives no START for a corpse the team did not see at tick start.
  const unseen = requiems.records({
    event: returnPulse(),
    plan: planWith({ start: null }),
    transform,
  });
  assert.equal(unseen[0].oldCorpse, null);
  // Resurrection Effects off, or no plan: no copy.
  const off = requiems.records({
    event: returnPulse(),
    plan,
    transform,
    filters: { resurrection_effects: false },
  });
  assert.equal(off[0].oldCorpse, null);
  assert.equal(requiems.records({ event: returnPulse() })[0].oldCorpse, null);
  // Switches drop the relic and the pip.
  const bare = requiems.records({
    event: returnPulse(),
    plan,
    transform,
    filters: { death_effects: false, vigil_pips: false },
  });
  assert.equal(bare[0].oldCorpse.relic, false);
  assert.equal(bare[0].oldCorpse.pips, 0);
});

test("the copy repeats the Champions look's Ashen Token and its coin", () => {
  for (const mate of [null, [6.2, 5]]) {
    const { record: own, fallen, body } = championCorpse(mate);
    const [record] = requiems.records({
      event: returnPulse(),
      plan: planWith({ mate }),
      transform,
    });
    const corpse = record.oldCorpse;
    assert.deepEqual(corpse.aside, own.aside);
    assert.equal(corpse.aside === null, mate === null);
    const { doc, mount } = mountFor();
    requiems.build([record], /** @type {any} */ (doc), /** @type {any} */ (mount));
    const [copy] = mount.connector.querySelectorAll(".requiem-old-corpse");
    const disc = copy.querySelector(".requiem-old-corpse__disc");
    assert.equal(Number(disc.getAttribute("r")), Number(body.getAttribute("r")));
    assert.equal(disc.getAttribute("fill-opacity"), body.getAttribute("fill-opacity"));
    assert.equal(
      Number(disc.getAttribute("cx")) + corpse.center.x,
      Number(body.getAttribute("cx")),
    );
    const rims = copy.querySelectorAll(".requiem-old-corpse__rim");
    const ownRims = fallen.querySelectorAll(".champion-ashen__rim");
    assert.equal(rims.length, ownRims.length);
    rims.forEach((rim, index) => {
      for (const name of ["r", "stroke-width", "opacity"]) {
        assert.equal(rim.getAttribute(name), ownRims[index].getAttribute(name), name);
      }
    });
    const emblem = copy.querySelector(".requiem-old-corpse__emblem");
    const ownEmblem = /** @type {any} */ (
      fallen.querySelector(".champion-ashen__emblem")
    );
    for (const name of ["width", "height"]) {
      assert.equal(emblem.getAttribute(name), ownEmblem.getAttribute(name), name);
    }
    assert.equal(
      Number(emblem.getAttribute("x")) + corpse.center.x,
      Number(ownEmblem.getAttribute("x")),
    );
    // The relic and the last pip sit on the corpse spot itself.
    assert.equal(copy.querySelectorAll(".requiem-relic").length, 1);
    assert.equal(copy.querySelector(".vigil-pips").getAttribute("data-pips"), "1");
  }
});

test("the old corpse holds from the tick's start and lifts at the Return beat", () => {
  const record = requiems.records({
    event: returnPulse(),
    plan: planWith(),
    transform,
  });
  const { doc, mount } = mountFor();
  requiems.build(record, /** @type {any} */ (doc), /** @type {any} */ (mount));
  const [copy] = mount.connector.querySelectorAll(".requiem-old-corpse");
  const [lift] = mount.connector.querySelectorAll(".requiem-old-corpse__lift-group");
  assert.equal(copy.getAttribute("opacity"), "0");
  assert.equal(lift.getAttribute("opacity"), "0");
  assert.equal(copy.getAttribute("aria-hidden"), "true");
  const center = transform.worldToScreen([6, 5]);
  assert.equal(copy.getAttribute("transform"), `translate(${center.x} ${center.y})`);
  const specs = new Map(
    mount.animationSpecs.map((spec) => [spec.id.split(":").at(-1), spec]),
  );
  const fade = specs.get("old-corpse");
  assert.equal(fade.element, copy);
  assert.deepEqual(fade.keyframes, [{ opacity: 1 }, { opacity: 0 }]);
  assert.equal(fade.options.delay, 900);
  assert.equal(fade.options.duration, 200);
  assert.equal(fade.options.fill, "backwards");
  const fadeLift = specs.get("old-corpse-lift");
  assert.equal(fadeLift.element, lift);
  assert.equal(fadeLift.keyframes[0].opacity, 0);
  assert.equal(fadeLift.keyframes[1].opacity, 0.5);
  assert.equal(fadeLift.options.duration, 300);
  const grow = specs.get("old-corpse-lift-grow");
  assert.equal(grow.keyframes.at(-1).transform, "scale(1.2)");
  assert.doesNotMatch(
    serialize(mount.connector),
    /url\(|gradient|<mask|<filter|<image|<foreignObject| id="/iu,
  );
  // Reduced motion: a 160 ms fade from the scaled beat and no growth.
  const reduced = mountFor({ motionMode: "reduced" });
  requiems.build(
    record,
    /** @type {any} */ (reduced.doc),
    /** @type {any} */ (reduced.mount),
  );
  const parts = reduced.mount.animationSpecs.map((spec) => spec.id.split(":").at(-1));
  assert.ok(!parts.includes("old-corpse-lift-grow"));
  const quick = reduced.mount.animationSpecs.find((spec) =>
    spec.id.endsWith(":old-corpse"),
  );
  assert.equal(quick.options.duration, 160);
  assert.equal(quick.options.delay, 225);
  // A paused still shows no old corpse.
  const still = mountFor({ settled: true });
  requiems.build(
    record,
    /** @type {any} */ (still.doc),
    /** @type {any} */ (still.mount),
  );
  assert.equal(still.mount.connector.children.length, 0);
});

test("update moves the old corpse and its lift to the new corpse spot", () => {
  const record = requiems.records({
    event: returnPulse(),
    plan: planWith(),
    transform,
  });
  const { doc, mount } = mountFor();
  const nodes = requiems.build(
    record,
    /** @type {any} */ (doc),
    /** @type {any} */ (mount),
  );
  const wider = createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: 1200,
    viewportHeight: 600,
    padding: 0,
  });
  const moved = requiems.records({
    event: returnPulse({ anchorRadius: 30 }),
    plan: planWith(),
    transform: wider,
  });
  requiems.update(nodes, moved);
  const center = wider.worldToScreen([6, 5]);
  const place = `translate(${center.x} ${center.y})`;
  const [copy] = mount.connector.querySelectorAll(".requiem-old-corpse");
  const [lift] = mount.connector.querySelectorAll(".requiem-old-corpse__lift-group");
  assert.equal(copy.getAttribute("transform"), place);
  assert.equal(lift.getAttribute("transform"), place);
  assert.equal(copy.querySelector(".requiem-old-corpse__disc").getAttribute("r"), "29");
});
