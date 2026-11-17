/**
 * @file Checks the Flags and Bases look (`src/looks/flags.js`): no records
 * outside Capture the Flag or without a transform; the ground part (both bases,
 * Team B double-rimmed, a base full while its flag is home and 40% while it is
 * away, the capture ring and zone cut at the end wall, a dropped flag's countdown
 * arc sized by the public return ticks and no arc without them, planted
 * pennants: Team A a triangle, Team B a swallowtail) and the above part (a
 * carried pennant outlined and following its carrier only when the view names
 * the carrier, a dropped flag's ticks left); this tick's changes (a capture goes
 * home at "score" with a "before" copy on the carrier, a pulse on its base and
 * the capture mark at the capturer's base; a carried flag stays on its carrier
 * for a carry's whole life (pickup, carried move, death, capture: every carried
 * pennant stands at the carrier's END spot, so its glide keeps it on the body;
 * with no body in the scene, which is the Agent POV case of a carrier seen at
 * the tick's start and not at its end, a changing copy stays at its start spot);
 * a drop waits for the start carrier's "fall", or shows at once when the view did
 * not name it; a pickup and a grab and fall wait for "impact", the base dimming
 * with the pickup; a countdown alone is no change); the Flags switch off (plain pennants and bases, no rings, numbers,
 * outlines, pulses or marks); and build() nodes, beat-hold attributes, export
 * safety and update() reprojection; a pennant whose carrier the view cannot
 * see slides between its public spots in a played tick with normal motion and
 * Body Glide on, along the bodies' glide window (animationSpecs), and snaps
 * otherwise, also when its carrier has just left sight (a flag carried at both
 * ends is no change); every pennant points toward its own home's wall, so a
 * carried one trails behind its carrier and never flips at the middle, and at
 * home its cloth ends on the floor's edge; and the carrier words ("Carrying Team B's
 * flag", carryingWords and carriedFlagTeam) only for a carrier the scene names,
 * with the agent inspector's Carrying row from them; and the hover cards: a
 * base's (team, its flag, capture reach, the scoring rule) on a ring band that
 * leaves the spawn pad's card in the middle and is never thinner than 14 px at
 * the reference scale, and a flag's (team; at home, carried by the champion its
 * roster row names, or dropped with the ticks until it goes home; touch reach;
 * a note per state, and "out of sight" for an unnamed carrier), with no card on
 * a copy that fades out at its beat.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { explainAgent } from "../src/explanations.js";
import { createViewportTransform } from "../src/layout.js";
import * as flags from "../src/looks/flags.js";

class FakeElement {
  /** @param {string} tagName */
  constructor(tagName) {
    this.tagName = tagName;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    this.textContent = "";
    /** @type {FakeElement | null} */
    this.parentElement = null;
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
    for (const node of nodes) node.parentElement = this;
    this.children.push(...nodes);
  }
  getBoundingClientRect() {
    return { left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0 };
  }
}

const doc = /** @type {Document} */ (
  /** @type {unknown} */ ({
    createElementNS: (/** @type {string} */ _namespace, /** @type {string} */ tag) =>
      new FakeElement(tag),
  })
);

/** @param {FakeElement} node @returns {FakeElement[]} */
function walk(node) {
  return [node, ...node.children.flatMap(walk)];
}

const transform = createViewportTransform({
  worldWidth: 20,
  worldHeight: 10,
  viewportWidth: 1440,
  viewportHeight: 900,
  padding: 24,
});

/** @type {Readonly<Record<1 | 2, [number, number]>>} */
const HOMES = Object.freeze({ 1: [0.5, 5], 2: [19.5, 5] });

/**
 * @param {1 | 2} team
 * @param {"home" | "carried" | "dropped"} state
 * @param {Record<string, any>} [extra]
 */
function flag(team, state, extra = {}) {
  return {
    team_id: team,
    state,
    position: state === "home" ? HOMES[team] : [7.7, 3.8],
    home_position: HOMES[team],
    carrier_presentation_key: null,
    carrier_public_agent_id: null,
    countdown_steps: state === "dropped" ? 28 : 0,
    ...extra,
  };
}

const CARRIED_BY_KEY = Object.freeze({
  position: [4.7, 6],
  carrier_presentation_key: "key-hunter",
  carrier_public_agent_id: "agent-2",
});

/** @param {Record<string, any>} teamA @param {Record<string, any>} teamB */
function scene(teamA, teamB) {
  return {
    capture_the_flag: {
      flags: [teamA, teamB],
      capture_distance: 1.25,
      interaction_distance: 0.75,
      return_steps: 30,
    },
  };
}

/**
 * @param {"ground" | "above"} part
 * @param {Record<string, any>} input
 */
function draw(part, input) {
  return flags.records({ transform, returnTicks: 30, ...input, part });
}

/** @param {ReadonlyArray<Record<string, any>>} records @param {string} kind */
const ofKind = (records, kind) => records.filter((record) => record.kind === kind);

test("no flags outside Capture the Flag or without a transform", () => {
  for (const part of /** @type {const} */ (["ground", "above"])) {
    assert.deepEqual(draw(part, { scene: { capture_the_flag: null } }), []);
    assert.deepEqual(draw(part, { scene: {} }), []);
    assert.deepEqual(
      flags.records({ part, scene: scene(flag(1, "home"), flag(2, "home")) }),
      [],
    );
  }
  assert.deepEqual(flags.build([], doc), []);
});

test("flags and bases explain themselves on hover, in every state", () => {
  /** @param {Record<string, any> | null} descriptor */
  const card = (descriptor) => {
    if (!descriptor) return null;
    const rows = descriptor.rows.map((/** @type {any} */ row) => [
      row.label,
      row.value,
    ]);
    return {
      title: descriptor.title,
      rows: rows.filter((/** @type {string[]} */ row) => row[0] !== "Note"),
      note: rows.find((/** @type {string[]} */ row) => row[0] === "Note")?.[1],
    };
  };
  const base = {
    ...scene(flag(1, "home"), flag(2, "carried", CARRIED_BY_KEY)),
    agents: [
      {
        presentation_key: "key-hunter",
        public_agent_id: "agent-2",
        display_agent_id: "2",
        team_id: 1,
        class_id: 3,
        position: [4.7, 6],
      },
    ],
  };
  const ground = draw("ground", { scene: base });
  const above = draw("above", { scene: base });
  const [baseA, baseB] = ofKind(ground, "flag-base");
  assert.deepEqual(card(flags.tooltip(baseA)), {
    title: "Flag Base",
    rows: [
      ["Team", "Team A"],
      ["Its Flag", "At Home"],
      ["Capture Reach", "1.25 units"],
    ],
    note: "Team A scores by bringing Team B's flag here while Team A's own flag is home.",
  });
  assert.equal(card(flags.tooltip(baseB))?.rows[1][1], "Away");
  const [homeFlag] = ofKind(ground, "flag-pennant");
  assert.deepEqual(card(flags.tooltip(homeFlag))?.rows, [
    ["Team", "Team A"],
    ["State", "At Home"],
    ["Touch Reach", "0.75 units"],
  ]);
  assert.match(
    String(card(flags.tooltip(homeFlag))?.note),
    /^Team B takes it by touching it/u,
  );
  const [carried] = ofKind(above, "flag-pennant");
  assert.equal(
    card(flags.tooltip(carried))?.rows[1][1],
    "Carried by Agent 2 Hunter, Team A",
  );
  // An unnamed carrier (Agent POV: one the team cannot see) stays unnamed.
  const unseen = ofKind(
    draw("above", { scene: scene(flag(1, "home"), flag(2, "carried")) }),
    "flag-pennant",
  )[0];
  assert.deepEqual(
    [card(flags.tooltip(unseen))?.rows[1][1], card(flags.tooltip(unseen))?.note],
    ["Carried", "Its carrier is out of this champion's sight."],
  );
  const dropped = ofKind(
    draw("ground", { scene: scene(flag(1, "home"), flag(2, "dropped")) }),
    "flag-pennant",
  )[1];
  assert.equal(
    card(flags.tooltip(dropped))?.rows[1][1],
    "Dropped, goes home in 28 ticks",
  );
  assert.match(
    String(card(flags.tooltip(dropped))?.note),
    /when the count runs out\.$/u,
  );
  // Only what the page shows at the END of the tick has a card.
  const start = flag(2, "carried", { ...CARRIED_BY_KEY, position: [1, 5.2] });
  const capture = draw("above", {
    scene: { ...scene(flag(1, "home"), flag(2, "home")), agents: base.agents },
    flagTrajectories: [
      { team_id: 2, transition_start: start, successor: flag(2, "home") },
    ],
  });
  assert.equal(flags.tooltip(ofKind(capture, "flag-pennant")[0]), null);
  for (const record of [
    ...ofKind(ground, "flag-countdown"),
    ...ofKind(above, "flag-ticks"),
  ]) {
    assert.equal(flags.tooltip(record), null);
  }
  // The built nodes carry a labeled hover shape: the base a ring band whose
  // middle stays the spawn pad's, the flag a circle on its cloth.
  const nodes = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (flags.build([baseA, homeFlag], doc))
  );
  const hits = nodes.map((node) => node.children.at(-1));
  assert.deepEqual(
    hits.map((hit) => [
      hit?.tagName,
      hit?.getAttribute("class"),
      hit?.getAttribute("aria-label"),
    ]),
    [
      ["path", "flag-base__hit", "Team A's Flag Base"],
      ["circle", "flag-pennant__hit", "Team A's Flag"],
    ],
  );
  assert.equal(hits[0]?.getAttribute("fill-rule"), "evenodd");
  assert.equal(hits[0]?.getAttribute("d")?.match(/M/gu)?.length, 2);
  assert.ok(Number(hits[1]?.getAttribute("r")) > 0);
  // At a 0.75 capture reach the band would be only about 0.3 units wide, so it
  // keeps its minimum width (14 px at the reference scale) and stays easy to find.
  const narrow = ofKind(
    draw("ground", {
      scene: {
        ...scene(flag(1, "home"), flag(2, "home")),
        capture_the_flag: {
          ...scene(flag(1, "home"), flag(2, "home")).capture_the_flag,
          capture_distance: 0.75,
        },
      },
    }),
    "flag-base",
  )[0];
  const [inner, outer] = [...String(narrow.hit).matchAll(/a(-?[\d.]+) /gu)]
    .map((match) => Number(match[1]))
    .filter((_, index) => index % 2 === 0);
  assert.ok(outer - inner >= 14 * narrow.u - 0.01, `band ${outer - inner} px`);
});

test("bases show their flag's state by strength, and Team B is double-lined", () => {
  const input = { scene: scene(flag(1, "home"), flag(2, "carried", CARRIED_BY_KEY)) };
  const [baseA, baseB, ...rest] = draw("ground", input);
  assert.deepEqual(
    [
      baseA.kind,
      baseA.teamId,
      baseA.strength,
      baseB.kind,
      baseB.teamId,
      baseB.strength,
    ],
    ["flag-base", 1, 1, "flag-base", 2, 0.4],
  );
  assert.equal(baseA.rim.match(/M/gu)?.length, 1);
  assert.equal(baseB.rim.match(/M/gu)?.length, 2);
  // Only Team A's flag is on the ground; Team B's rides above the champions.
  assert.deepEqual(
    rest.map((record) => [record.kind, record.teamId, record.state]),
    [["flag-pennant", 1, "home"]],
  );
  const triangle = rest[0].shape.match(/[ML]/gu)?.length;
  const [carried] = draw("above", input);
  assert.equal(triangle, 3);
  assert.equal(carried.shape.match(/[ML]/gu)?.length, 5);
  assert.deepEqual(
    [carried.kind, carried.state, carried.outlined, carried.follow],
    ["flag-pennant", "carried", true, "key-hunter"],
  );
});

test("at home each pennant points at its own wall, its cloth ending on the floor's edge", () => {
  const xs = (/** @type {string} */ path) =>
    [...path.matchAll(/[ML](-?\d+(?:\.\d+)?) /gu)].map((match) => Number(match[1]));
  const pennants = ofKind(
    draw("ground", { scene: scene(flag(1, "home"), flag(2, "home")) }),
    "flag-pennant",
  );
  const [poleA, poleB] = pennants.map((record) => xs(record.pole)[0]);
  const left = transform.worldToScreen([0, 0]).x;
  const right = transform.worldToScreen([transform.worldWidth, 0]).x;
  const [clothA, clothB] = pennants.map((record) => xs(record.shape));
  assert.ok(
    clothA.every((x) => x <= poleA + 0.001),
    "Team A's points left",
  );
  assert.ok(
    clothB.every((x) => x >= poleB - 0.001),
    "Team B's points right",
  );
  // The homes sit 0.5 units from the walls, closer than the cloth is wide, so
  // each pole moves in until its cloth ends exactly on the floor's edge.
  assert.ok(Math.abs(Math.min(...clothA) - left) < 0.01, String(Math.min(...clothA)));
  assert.ok(Math.abs(Math.max(...clothB) - right) < 0.01, String(Math.max(...clothB)));
});

test("the capture ring is cut at the end wall, never past the floor", () => {
  const [baseA] = draw("ground", { scene: scene(flag(1, "home"), flag(2, "home")) });
  assert.equal(baseA.ring.match(/M/gu)?.length, 1);
  assert.match(baseA.zone, /Z$/u);
  const xs = [
    ...baseA.ring.matchAll(
      /[MA][^MA]*?(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)(?=[MA]|$)/gu,
    ),
  ].map((match) => Number(match[1]));
  for (const x of xs) assert.ok(x >= transform.mapBounds.left - 0.001, String(x));
  const left = transform.worldToScreen([0, 0]).x;
  assert.ok(Math.abs(Math.min(...xs) - left) < 0.01, "the cut ends on the wall");
});

test("a carrier the view cannot name gets a plain pennant that follows nobody", () => {
  const unseen = flag(2, "carried", { position: [4.7, 6] });
  const [carried] = draw("above", { scene: scene(flag(1, "home"), unseen) });
  assert.deepEqual([carried.outlined, carried.follow], [false, null]);
});

test("a dropped flag shows its countdown arc and its ticks left", () => {
  const input = { scene: scene(flag(1, "home"), flag(2, "dropped")) };
  const countdown = ofKind(draw("ground", input), "flag-countdown");
  const ticks = ofKind(draw("above", input), "flag-ticks");
  assert.equal(countdown.length, 1);
  assert.notEqual(countdown[0].arc, "");
  assert.deepEqual(
    ticks.map((record) => [record.teamId, record.text]),
    [[2, "28"]],
  );
  // With no public return ticks there is a ring but no arc: never a guess.
  const [unsized] = ofKind(
    draw("ground", { ...input, returnTicks: null }),
    "flag-countdown",
  );
  assert.equal(unsized.arc, "");
  assert.notEqual(unsized.track, "");
});

test("a capture goes home at the score beat, with a pulse and the capture mark", () => {
  const start = flag(2, "carried", { ...CARRIED_BY_KEY, position: [1, 5.2] });
  const input = {
    scene: {
      ...scene(flag(1, "home"), flag(2, "home")),
      agents: [{ presentation_key: "key-hunter", position: [0.9, 5.2] }],
    },
    flagTrajectories: [
      { team_id: 1, transition_start: flag(1, "home"), successor: flag(1, "home") },
      { team_id: 2, transition_start: start, successor: flag(2, "home") },
    ],
  };
  const ground = draw("ground", input);
  const above = draw("above", input);
  const hold = (/** @type {Record<string, any>} */ record) =>
    record.hold && [
      record.hold.kind,
      record.hold.side,
      record.hold.from,
      record.hold.key,
    ];
  assert.deepEqual(hold(ground[0]), null, "Team A's base did not change");
  assert.deepEqual(hold(ground[1]), ["score", "after", 0.4, null]);
  assert.deepEqual(
    ofKind(ground, "flag-pennant").map((record) => [
      record.teamId,
      ...(hold(record) || []),
    ]),
    [[1], [2, "score", "after", 0, null]],
  );
  const [copy] = ofKind(above, "flag-pennant");
  assert.deepEqual(
    [copy.state, copy.follow, copy.outlined, ...hold(copy)],
    ["carried", "key-hunter", true, "score", "before", 1, null],
  );
  const [pulse] = ofKind(above, "flag-pulse");
  const [mark] = ofKind(above, "flag-mark");
  assert.deepEqual(
    [pulse.teamId, pulse.hold.kind, pulse.hold.pulse],
    [2, "score", 0.9],
  );
  assert.deepEqual(
    [mark.teamId, mark.hold.kind, mark.hold.side],
    [1, "score", "after"],
  );
  assert.equal(mark.d.match(/M/gu)?.length, 8);
  const home = transform.worldToScreen(HOMES[1]);
  const [x0, y0] = /** @type {RegExpMatchArray} */ (
    mark.d.match(/-?\d+(?:\.\d+)?/gu)
  ).map(Number);
  assert.ok(
    Math.abs(
      Math.hypot(x0 - home.x, y0 - home.y) - 0.85 * transform.worldLengthToScreen(1),
    ) < 0.01,
  );
});

test("a carried flag stays on its carrier for a carry's whole life", () => {
  // Every carried pennant drawn in a tick rides its carrier's glide, and a glide
  // carries a rider from the body's START spot to its END spot, so each one must
  // stand at the carrier's END spot: the pickup, a carried move, a death and a
  // capture. With no body in the scene a changing copy stays at its start spot.
  /** @param {[number, number]} position */
  const carriedAt = (position) =>
    ofKind(
      draw("above", {
        scene: scene(
          flag(1, "home"),
          flag(2, "carried", { ...CARRIED_BY_KEY, position }),
        ),
      }),
      "flag-pennant",
    )[0];
  const carried = (/** @type {[number, number]} */ position) =>
    flag(2, "carried", { ...CARRIED_BY_KEY, position });
  /** @type {Array<[string, Record<string, any>, Record<string, any>, [number, number]]>} */
  const ticks = [
    ["pickup", flag(2, "home"), carried([18.6, 4.7]), [18.6, 4.7]],
    ["move", carried([10, 5]), carried([9.4, 5.3]), [9.4, 5.3]],
    [
      "death",
      carried([5.5, 6.5]),
      flag(2, "dropped", { position: [4.7, 6] }),
      [4.7, 6],
    ],
    ["capture", carried([1, 6.4]), flag(2, "home"), [0.9, 5.4]],
  ];
  for (const [name, start, successor, endAt] of ticks) {
    const flagTrajectories = [{ team_id: 2, transition_start: start, successor }];
    const base = scene(flag(1, "home"), successor);
    const riding = ofKind(
      draw("above", {
        scene: {
          ...base,
          agents: [{ presentation_key: "key-hunter", position: endAt }],
        },
        flagTrajectories,
      }),
      "flag-pennant",
    );
    assert.equal(riding.length, 1, name);
    assert.deepEqual(
      [riding[0].state, riding[0].follow],
      ["carried", "key-hunter"],
      name,
    );
    assert.equal(riding[0].pole, carriedAt(endAt).pole, name);
    if (name === "death" || name === "capture") {
      assert.equal(riding[0].hold.side, "before", name);
      const [alone] = ofKind(
        draw("above", { scene: base, flagTrajectories }),
        "flag-pennant",
      );
      assert.deepEqual(
        [alone.follow, alone.pole],
        [null, carriedAt(/** @type {[number, number]} */ (start.position)).pole],
        name,
      );
    }
  }
});

test("a drop waits for its carrier's fall, or shows at once when unnamed", () => {
  const dropped = flag(2, "dropped", { position: [4.7, 6] });
  for (const [key, expected] of /** @type {const} */ ([
    ["key-hunter", "key-hunter"],
    [null, null],
  ])) {
    const start = flag(2, "carried", {
      position: [4.7, 6],
      carrier_presentation_key: key,
      carrier_public_agent_id: key === null ? null : "agent-2",
    });
    const input = {
      scene: scene(flag(1, "home"), dropped),
      flagTrajectories: [{ team_id: 2, transition_start: start, successor: dropped }],
    };
    const above = draw("above", input);
    const ground = draw("ground", input);
    const [copy] = ofKind(above, "flag-pennant");
    assert.deepEqual(
      [copy.hold.kind, copy.hold.side, copy.hold.key],
      ["fall", "before", expected],
    );
    for (const record of [
      ...ofKind(ground, "flag-pennant").slice(1),
      ...ofKind(ground, "flag-countdown"),
    ]) {
      assert.deepEqual([record.hold.kind, record.hold.key], ["fall", expected]);
    }
    assert.equal(ground[1].hold, null, "the base stays dim: its flag is still away");
    assert.equal(ofKind(above, "flag-pulse").length, 0);
  }
});

test("a pickup and a grab and fall wait for impact; a countdown alone is no change", () => {
  const carried = flag(1, "carried", {
    ...CARRIED_BY_KEY,
    carrier_presentation_key: "key-rogue",
  });
  const pickup = {
    scene: scene(carried, flag(2, "home")),
    flagTrajectories: [
      { team_id: 1, transition_start: flag(1, "home"), successor: carried },
    ],
  };
  const [baseA] = draw("ground", pickup);
  assert.deepEqual(
    [baseA.strength, baseA.hold.kind, baseA.hold.from],
    [0.4, "impact", 1],
  );
  assert.deepEqual(
    ofKind(draw("ground", pickup), "flag-pennant").map((record) => [
      record.teamId,
      record.hold?.side,
    ]),
    [
      [1, "before"],
      [2, undefined],
    ],
  );
  const grabbed = flag(2, "dropped", { position: [8, 4], countdown_steps: 30 });
  const grab = {
    scene: scene(flag(1, "home"), grabbed),
    flagTrajectories: [
      { team_id: 2, transition_start: flag(2, "dropped"), successor: grabbed },
    ],
  };
  const [pulse] = ofKind(draw("above", grab), "flag-pulse");
  assert.deepEqual([pulse.hold.kind, pulse.hold.pulse], ["impact", 0.9]);
  const later = flag(2, "dropped", { countdown_steps: 27 });
  const tick = {
    scene: scene(flag(1, "home"), later),
    flagTrajectories: [
      { team_id: 2, transition_start: flag(2, "dropped"), successor: later },
    ],
  };
  for (const part of /** @type {const} */ (["ground", "above"])) {
    assert.ok(draw(part, tick).every((record) => record.hold === null));
  }
  // A carrier only leaving this view's sight changes nothing about the flag;
  // its pennant slides from the seen spot to the public END spot.
  const leaving = flag(2, "carried", { position: [5, 6] });
  const sight = {
    scene: scene(flag(1, "home"), leaving),
    flagTrajectories: [
      {
        team_id: 2,
        transition_start: flag(2, "carried", CARRIED_BY_KEY),
        successor: leaving,
      },
    ],
  };
  for (const part of /** @type {const} */ (["ground", "above"])) {
    assert.ok(draw(part, sight).every((record) => record.hold === null));
  }
  const step = transform.worldLengthToScreen(0.3);
  assert.deepEqual(draw("above", sight)[0].glideFrom, [-Number(step.toFixed(3)), 0]);
});

test("Flags off leaves plain pennants and plain bases, with no rings or marks", () => {
  const start = flag(2, "carried", { ...CARRIED_BY_KEY, position: [1, 5.2] });
  const input = {
    filters: { flags: false },
    scene: scene(flag(1, "carried", CARRIED_BY_KEY), flag(2, "home")),
    flagTrajectories: [
      { team_id: 2, transition_start: start, successor: flag(2, "home") },
    ],
  };
  const ground = draw("ground", input);
  const above = draw("above", input);
  for (const base of ofKind(ground, "flag-base")) {
    assert.deepEqual(
      [base.ring, base.zone, base.strength, base.hold],
      ["", "", 1, null],
    );
  }
  assert.deepEqual(
    [...ground, ...above]
      .map((record) => record.kind)
      .filter((kind) => kind !== "flag-base" && kind !== "flag-pennant"),
    [],
  );
  assert.ok(ofKind(above, "flag-pennant").every((record) => record.outlined === false));
  const dropped = {
    filters: { flags: false },
    scene: scene(flag(1, "home"), flag(2, "dropped")),
  };
  assert.deepEqual(ofKind(draw("ground", dropped), "flag-countdown"), []);
  assert.deepEqual(draw("above", dropped), []);
});

test("a pennant whose carrier the view cannot see slides with the bodies", () => {
  const before = flag(2, "carried", { position: [4.7, 6] });
  const after = flag(2, "carried", { position: [5.2, 6] });
  const input = {
    scene: scene(flag(1, "home"), after),
    flagTrajectories: [{ team_id: 2, transition_start: before, successor: after }],
  };
  const [pennant] = draw("above", input);
  const step = transform.worldLengthToScreen(0.5);
  assert.deepEqual(pennant.glideFrom, [-Number(step.toFixed(3)), 0]);
  assert.equal(pennant.hold, null);
  const root = new FakeElement("g");
  root.append(
    .../** @type {FakeElement[]} */ (
      /** @type {unknown} */ (flags.build([pennant], doc))
    ),
  );
  /** @type {any} */ (root).querySelectorAll = (/** @type {string} */ selector) =>
    selector === "[data-flag-glide]"
      ? walk(root).filter((node) => node.attributes.has("data-flag-glide"))
      : [];
  const beats = { glide: true, windows: { glide: [20, 280] } };
  const parent = /** @type {ParentNode} */ (/** @type {unknown} */ (root));
  const played = flags.animationSpecs(parent, {
    epochKey: "e",
    mode: "played",
    motionMode: "normal",
    beats,
  });
  assert.equal(played.length, 1);
  assert.deepEqual(
    [played[0].options.delay, played[0].options.duration, played[0].options.fill],
    [20, 260, "backwards"],
  );
  assert.equal(played[0].keyframes.at(-1)?.transform, "translate(0px, 0px)");
  for (const options of [
    { mode: "played", motionMode: "reduced", beats },
    { mode: "step", motionMode: "normal", beats },
    { mode: "played", motionMode: "normal", beats: { ...beats, glide: false } },
  ]) {
    assert.deepEqual(
      flags.animationSpecs(parent, /** @type {any} */ ({ epochKey: "e", ...options })),
      [],
    );
  }
  // A seen carrier's pennant rides its glide instead (data-glide-follow).
  const seen = draw("above", {
    scene: scene(flag(1, "home"), flag(2, "carried", CARRIED_BY_KEY)),
  });
  assert.equal(seen[0].glideFrom, null);
});

test("a carried pennant trails toward its own home on both halves of the map", () => {
  for (const x of [4.7, 15.3]) {
    for (const team of /** @type {const} */ ([1, 2])) {
      const carried = flag(team, "carried", { ...CARRIED_BY_KEY, position: [x, 6] });
      const home = flag(team === 1 ? 2 : 1, "home");
      const flagScene = team === 1 ? scene(carried, home) : scene(home, carried);
      const pennant = ofKind(draw("above", { scene: flagScene }), "flag-pennant")[0];
      // The shape starts at the pole top, then goes to the cloth's far edge.
      const [pole, tip] = [...pennant.shape.matchAll(/[ML](-?[\d.]+) /gu)].map(
        (match) => Number(match[1]),
      );
      assert.equal(
        Math.sign(tip - pole),
        team === 1 ? -1 : 1,
        `team ${team} at x ${x}`,
      );
    }
  }
});

test("carrier words name the carried flag only when the scene names the carrier", () => {
  const ctf = scene(
    flag(1, "home"),
    flag(2, "carried", CARRIED_BY_KEY),
  ).capture_the_flag;
  assert.equal(flags.carryingWords(ctf, "key-hunter"), "Carrying Team B's flag");
  assert.equal(flags.carriedFlagTeam(ctf, "key-hunter"), 2);
  for (const key of ["key-other", null, undefined]) {
    assert.equal(flags.carryingWords(ctf, key), null);
    assert.equal(flags.carriedFlagTeam(ctf, key), null);
  }
  // Agent POV: an unseen carrier is not named, so nobody is "carrying".
  const unseen = scene(flag(1, "home"), flag(2, "carried", { position: [4.7, 6] }));
  assert.equal(flags.carryingWords(unseen.capture_the_flag, "key-hunter"), null);
  assert.equal(flags.carryingWords(null, "key-hunter"), null);
  const explained = explainAgent(
    { class_id: 3, team_id: 1 },
    { carrying: "Carrying Team B's flag" },
  );
  assert.ok(JSON.stringify(explained).includes("Team B's flag"));
  assert.equal(
    JSON.stringify(explainAgent({ class_id: 3, team_id: 1 })).includes("Carrying"),
    false,
  );
});

test("build gives export-safe nodes with beat holds, and update reprojects them", () => {
  const start = flag(2, "carried", { ...CARRIED_BY_KEY, position: [1, 5.2] });
  const input = {
    scene: {
      ...scene(flag(1, "dropped"), flag(2, "home")),
      agents: [{ presentation_key: "key-hunter", position: [0.9, 5.2] }],
    },
    flagTrajectories: [
      { team_id: 2, transition_start: start, successor: flag(2, "home") },
    ],
  };
  const records = [...draw("ground", input), ...draw("above", input)];
  const nodes = /** @type {FakeElement[]} */ (
    /** @type {unknown} */ (flags.build(records, doc))
  );
  assert.deepEqual(
    nodes.map((node) => node.getAttribute("class")),
    [
      "flag-base flag-base--team-a",
      "flag-base flag-base--team-b",
      "flag-countdown flag-countdown--team-a",
      "flag-pennant flag-pennant--team-a",
      "flag-pennant flag-pennant--team-b",
      "flag-ticks flag-ticks--team-a",
      "flag-pennant flag-pennant--team-b flag-pennant--marked",
      "flag-pulse flag-pulse--team-b",
      "flag-mark flag-mark--team-a",
    ],
  );
  const attributes = (/** @type {FakeElement} */ node) =>
    [
      "opacity",
      "data-beat-hold",
      "data-beat-side",
      "data-beat-from",
      "data-beat-pulse",
      "data-glide-follow",
    ].map((name) => node.getAttribute(name));
  assert.deepEqual(attributes(nodes[0]), ["0.4", null, null, null, null, null]);
  assert.deepEqual(attributes(nodes[1]), ["1", "score", "after", "0.4", null, null]);
  assert.deepEqual(attributes(nodes[4]), ["1", "score", "after", "0", null, null]);
  assert.deepEqual(attributes(nodes[6]), [
    "0",
    "score",
    "before",
    "1",
    null,
    "key-hunter",
  ]);
  assert.deepEqual(attributes(nodes[7]), ["0", "score", "after", null, "0.9", null]);
  assert.equal(nodes[5].textContent, "28");
  for (const node of nodes.flatMap(walk)) {
    assert.ok(
      !["clipPath", "mask", "use", "image", "pattern"].includes(node.tagName),
      node.tagName,
    );
    assert.equal(node.attributes.has("id"), false);
    for (const value of node.attributes.values())
      assert.equal(value.includes("url("), false, value);
  }
  const wider = createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: 2880,
    viewportHeight: 1800,
    padding: 24,
  });
  const moved = [
    ...flags.records({ ...input, transform: wider, returnTicks: 30, part: "ground" }),
    ...flags.records({ ...input, transform: wider, returnTicks: 30, part: "above" }),
  ];
  flags.update(/** @type {Element[]} */ (/** @type {unknown} */ (nodes)), moved);
  assert.equal(nodes[3].children[2].getAttribute("d"), moved[3].shape);
  assert.notEqual(moved[3].shape, records[3].shape);
});
