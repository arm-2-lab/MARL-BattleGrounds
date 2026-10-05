/**
 * @file Check the Rune Zones look (src/looks/statuses.js): the marks a body's
 * statuses give (Frozen, Sapped, Hamstrung, Snagged, Withering, Unbound only
 * when Freedom matters, Kindled), at most three per body in the store's
 * priority; the Rune Ring's lit arcs (six 5-tick arcs, ready at 0); the
 * "status-marks" records (living bodies only, Agent POV gives no beats unless
 * the body was seen at tick start, switches turn the ring and marks off);
 * Settle beats rebuilt from the current frame only (one crystal goes out per
 * tick, an applied status comes in, an aged-out status goes out, a broken trap
 * leaves shards); Ward Circles (class colour, Team B double line); the 2 chips
 * + "+N" status tray; and export safety of the built SVG (no url(),
 * gradients, masks, filters, images or ids, and no raw hex in statuses.css
 * outside tokens).
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import * as statuses from "../src/looks/statuses.js";

class FakeElement {
  /** @param {string} tag */
  constructor(tag) {
    this.tagName = tag;
    this.localName = tag;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    /** @type {Record<string, string>} */
    this.dataset = {};
    this.ownerDocument = fakeDocument;
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
  /** @param {...FakeElement} nodes */
  append(...nodes) {
    this.children.push(...nodes);
  }
  /** @param {...FakeElement} nodes */
  replaceChildren(...nodes) {
    this.children = [...nodes];
  }
  /** @param {string} _type @param {unknown} _listener */
  addEventListener(_type, _listener) {}
  /** @returns {string} */
  serialize() {
    const attrs = [...this.attributes].map(([key, value]) => ` ${key}="${value}"`);
    return `<${this.tagName}${attrs.join("")}>${this.children.map((child) => child.serialize()).join("")}</${this.tagName}>`;
  }
}

const fakeDocument = {
  /** @param {string} _namespace @param {string} tag */
  createElementNS(_namespace, tag) {
    return new FakeElement(tag);
  },
};

const transform = /** @type {any} */ ({
  worldToScreen: (/** @type {number[]} */ point) => ({
    x: point[0] * 46,
    y: point[1] * 46,
  }),
  worldLengthToScreen: (/** @type {number} */ length) => length * 46,
});

/** @param {Record<string, any>} [overrides] */
function agent(overrides = {}) {
  return {
    presentation_key: "oracle_a",
    class_id: 2,
    team_id: 1,
    alive: true,
    position: [2, 3],
    radius: 0.5,
    ultimate_cooldown: 12,
    spawn_shield_remaining: 0,
    statuses: [],
    ...overrides,
  };
}

/** @param {string} id @param {number} duration @param {number} [magnitude] */
function status(id, duration, magnitude) {
  return { token_id: id, duration, ...(magnitude === undefined ? {} : { magnitude }) };
}

/** @param {Record<string, any>} input */
function marks(input) {
  return /** @type {any[]} */ (
    statuses.records({ part: "status-marks", transform, ...input })
  );
}

/** @param {any} record @param {string} id */
function pieceById(record, id) {
  return record.pieces.find((/** @type {any} */ entry) => entry.id === id);
}

test("marks follow the store's rules and keep at most three per body", () => {
  /** @param {string} token @param {number} ticks @param {number | null} [magnitude] */
  const row = (token, ticks, magnitude = null) => ({ token, ticks, magnitude });
  const frozen = statuses.runeMarksFromStatuses([row("stun_hunter_trap", 3)], false);
  assert.equal(frozen.frozen, 3);
  assert.equal(frozen.sapped, null);
  const sapped = statuses.runeMarksFromStatuses([row("stun_rogue_poison", 1)], false);
  assert.equal(sapped.sapped, "poison");
  const hamstrung = statuses.runeMarksFromStatuses(
    [row("slow_warrior_charge", 2), row("slow_hunter_basic", 4)],
    false,
  );
  assert.deepEqual(hamstrung.roots, { kind: "charge", hamstrung: true, beads: 4 });
  const snagged = statuses.runeMarksFromStatuses([row("slow_hunter_basic", 1)], false);
  assert.deepEqual(snagged.roots, { kind: "snag", hamstrung: false, beads: 1 });
  // Freedom matters only when slowed below x0.85, not stunned and not shielded.
  const freeSnag = statuses.runeMarksFromStatuses(
    [row("priest_freedom", 1), row("slow_hunter_basic", 1)],
    false,
  );
  assert.equal(freeSnag.unbound, false);
  const freeChain = statuses.runeMarksFromStatuses(
    [row("priest_freedom", 1), row("slow_warrior_charge", 1)],
    false,
  );
  assert.equal(freeChain.unbound, true);
  assert.equal(
    statuses.runeMarksFromStatuses(
      [row("priest_freedom", 1), row("slow_warrior_charge", 1)],
      true,
    ).unbound,
    false,
  );
  const busy = statuses.runeZones(
    statuses.runeMarksFromStatuses(
      [
        row("stun_hunter_trap", 2),
        row("anti_heal_rogue_poison", 3),
        row("slow_rogue_poison", 2),
        row("mage_burst", 5),
      ],
      false,
    ),
  );
  assert.equal(busy.frozen, 2);
  assert.equal(busy.withering, 3);
  assert.equal(busy.roots?.kind, "poison");
  assert.equal(busy.kindled, 0, "a fourth mark is dropped");
});

test("the Rune Ring lights one arc per five cooldown ticks and is ready at zero", () => {
  assert.equal(statuses.runeRingLitArcs(30), 0);
  assert.equal(statuses.runeRingLitArcs(26), 0);
  assert.equal(statuses.runeRingLitArcs(25), 1);
  assert.equal(statuses.runeRingLitArcs(1), 5);
  assert.equal(statuses.runeRingLitArcs(0), 6);
  assert.equal(statuses.runeRingLitArcs(null), null);
  const [record] = marks({ scene: { agents: [agent({ ultimate_cooldown: 0 })] } });
  assert.ok(pieceById(record, "ring.ready"));
  assert.equal(
    record.pieces.filter((/** @type {any} */ p) => p.id.startsWith("ring.gem")).length,
    4,
  );
  const [charging] = marks({ scene: { agents: [agent({ ultimate_cooldown: 12 })] } });
  assert.equal(pieceById(charging, "ring.track").opacity, 0.12);
  assert.equal(
    charging.pieces.filter((/** @type {any} */ p) => p.id.startsWith("ring.arc"))
      .length,
    3,
  );
  assert.equal(pieceById(charging, "ring.arc.0").opacity, 0.55);
});

test("status-marks records cover living bodies only and obey the switches", () => {
  const scene = {
    agents: [
      agent({ statuses: [status("hunter_trap_stun", 2)] }),
      agent({
        presentation_key: "oracle_dead",
        alive: false,
        statuses: [status("hunter_trap_stun", 2)],
      }),
    ],
  };
  const all = marks({ scene });
  assert.equal(all.length, 1);
  assert.equal(all[0].presentationKey, "oracle_a");
  assert.equal(all[0].classPaint, "warrior");
  assert.deepEqual(all[0].center, { x: 92, y: 138 });
  assert.ok(pieceById(all[0], "frozen.shell"));
  const ringOnly = marks({ scene, switches: { statusMarks: false } });
  assert.equal(
    ringOnly[0].pieces.some((/** @type {any} */ p) => p.id.startsWith("frozen")),
    false,
  );
  assert.equal(
    ringOnly[0].pieces.some((/** @type {any} */ p) => p.id.startsWith("ring")),
    true,
  );
  assert.deepEqual(
    marks({ scene, switches: { statusMarks: false, cooldownEffects: false } }),
    [],
  );
  assert.deepEqual(marks({ scene: null }), []);
});

test("Settle beats come from this frame: a crystal melts, a new status installs", () => {
  const events = [
    {
      event_kind: "status_applied",
      recipient_anchor: { presentation_key: "oracle_a" },
      status_id: "rogue_poison_anti_heal",
    },
  ];
  const scene = {
    agents: [
      agent({
        statuses: [status("hunter_trap_stun", 2), status("rogue_poison_anti_heal", 3)],
      }),
    ],
  };
  const [record] = marks({ scene, events });
  // Trap went 3 -> 2: crystal 2 is lit before Settle and a ghost after.
  const melt = pieceById(record, "frozen.crystal.2.before");
  assert.equal(melt.side, "before");
  assert.equal(melt.fromOpacity, 1);
  assert.equal(melt.fadeMs, 300);
  assert.equal(pieceById(record, "frozen.crystal.2").side, "after");
  assert.equal(pieceById(record, "frozen.shell").side, "steady");
  // Withering was applied this tick: every vein comes in at Settle.
  assert.equal(pieceById(record, "withering.0").side, "after");
  assert.equal(pieceById(record, "withering.0").fadeMs, 200);
  const cut = marks({ scene, events, switches: { statusApplication: false } })[0];
  assert.equal(pieceById(cut, "withering.0").fadeMs, 1);
});

test("an aged-out trap thaws and a broken trap shatters", () => {
  const anchor = { presentation_key: "oracle_a" };
  const thawed = marks({
    scene: { agents: [agent()] },
    events: [
      {
        event_kind: "status_aged_to_zero",
        recipient_anchor: anchor,
        status_id: "hunter_trap_stun",
      },
    ],
  })[0];
  assert.equal(pieceById(thawed, "frozen.shell").side, "before");
  assert.equal(pieceById(thawed, "frozen.shell").fadeMs, 480);
  assert.equal(pieceById(thawed, "thaw.ring").opacity, 0.28);
  const broken = marks({
    scene: { agents: [agent()] },
    events: [
      {
        event_kind: "status_broken_by_damage",
        recipient_anchor: anchor,
        status_id: "hunter_trap_stun",
      },
    ],
  })[0];
  assert.equal(pieceById(broken, "frozen.shell").fadeMs, 60);
  assert.equal(pieceById(broken, "shatter.shards").opacity, 0.45);
  const noBreak = marks({
    scene: { agents: [agent()] },
    switches: { freezingTrapBreak: false },
    events: [
      {
        event_kind: "status_broken_by_damage",
        recipient_anchor: anchor,
        status_id: "hunter_trap_stun",
      },
    ],
  })[0];
  assert.equal(pieceById(noBreak, "shatter.shards"), undefined);
  assert.equal(pieceById(noBreak, "frozen.shell").fadeMs, 1);
});

test("Agent POV draws only scene bodies and no beats from the dark", () => {
  const scene = {
    audience: "agent_pov",
    agents: [agent({ statuses: [status("hunter_trap_stun", 2)] })],
  };
  const [record] = marks({ scene });
  assert.ok(record.pieces.every((/** @type {any} */ p) => p.side === "steady"));
  const [seen] = marks({ scene, startSeen: () => true });
  assert.ok(seen.pieces.some((/** @type {any} */ p) => p.side === "before"));
  // A champion that is not in the POV scene has no record at all.
  assert.deepEqual(
    marks({ scene: { audience: "agent_pov", agents: [] } }).map(
      (r) => r.presentationKey,
    ),
    [],
  );
});

test("built marks and Ward Circles are export-safe SVG with beat-hold attributes", () => {
  const events = [
    { event_kind: "cooldown_ready", agent_anchor: { presentation_key: "oracle_a" } },
  ];
  const scene = {
    agents: [
      agent({
        ultimate_cooldown: 0,
        statuses: [status("warrior_charge_stun", 1), status("warrior_charge_slow", 2)],
      }),
    ],
  };
  const nodes = statuses.build(
    statuses.records({ part: "status-marks", transform, scene, events }),
    /** @type {any} */ (fakeDocument),
  );
  assert.equal(nodes.length, 1);
  const markup = /** @type {any} */ (nodes[0]).serialize();
  assert.match(markup, /data-beat-hold="settle"/u);
  assert.match(markup, /data-beat-side="after"/u);
  assert.match(markup, /class="status-marks"/u);
  const wardRecords = statuses.records({
    part: "aura-fields",
    transform,
    fields: [
      {
        aura_id: "warrior_damage_mitigation",
        token_id: "warrior_damage_mitigation",
        center: [2, 3],
        radius: 2,
        source_class_id: 2,
        source_presentation_key: "oracle_b",
      },
    ],
    agents: [{ presentation_key: "oracle_b", team_id: 2 }],
    tooltipInput: () => null,
  });
  assert.equal(/** @type {any} */ (wardRecords[0]).radius, 92);
  assert.equal(/** @type {any} */ (wardRecords[0]).teamB, true);
  assert.equal(/** @type {any} */ (wardRecords[0]).classKey, "warrior");
  for (const text of [markup]) {
    assert.doesNotMatch(text, /url\(|Gradient|<mask|<filter|<image|\sid=/u);
  }
});

test("the status tray shows two chips and counts the rest", () => {
  const tray = statuses.statusTray([
    status("priest_blessing_of_freedom_movement_floor", 1),
    status("hunter_basic_slow", 1),
    status("in_combat", 4),
    status("hunter_trap_stun", 3),
    status("rogue_poison_anti_heal", 2),
  ]);
  assert.deepEqual(
    tray.chips.map((chip) => [chip.tokenId, chip.sigilKey, chip.ticks]),
    [
      ["stun_hunter_trap", "sigil-frozen", 3],
      ["slow_hunter_basic", "sigil-snagged", 1],
    ],
  );
  assert.equal(tray.hiddenCount, 2);
});

test("statuses.css uses only colour tokens", async () => {
  const css = await readFile(
    new URL("../src/styles/statuses.css", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(css, /#[0-9a-f]{3,8}\b/iu);
  assert.doesNotMatch(css, /animation\s*:|transition\s*:|url\(/u);
});
