/**
 * @file Check the look-module contract under `src/looks/`. Every agreed look
 * (and the shared helpers they use) exists, and every look exports records,
 * build, update and tooltip functions and frozen FOOTPRINT, BUDGET and
 * PAINT_PARTS values of the right shape. Every paint part the visual filters
 * know has exactly one look owner. records() is pure data: run in Node with no
 * DOM, on deep-frozen inputs from every fixture frame (Oracle View and Agent
 * POV), it returns frozen records with a kind, holds no function or DOM node,
 * and gives the same output twice. Each painted plan event is claimed by
 * exactly one event look; a status change ("status_lifecycle") is claimed by
 * none, because V5 Rune Zones shows it on the body, never as an off-body badge,
 * and the death announcement and team respawn wave are claimed by none and are
 * not spatial, because they are page chrome. Movement and heralds draw real
 * records: Movement alone draws a tick's trajectories (Start Spot and Focus
 * Trail records whose kinds are its own paint parts; no other look changes when
 * trajectories are added), heralds alone draws a "herald_card" plan row (a part
 * it owns), and each built node carries its record's kind. The looks' footprints
 * match the plan's allocator table while it keeps one, and neither keeps a
 * respawn-wave box. With nothing to draw from, every look returns no records and
 * no nodes, and every look that draws nothing from a scene alone, plus movement
 * and heralds, gives no tooltip for an unknown kind.
 */
import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

import { authorizedPresentationSceneView } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import * as choreographyPlan from "../src/choreography-plan.js";
import { createViewportTransform, layoutStatusDocks } from "../src/layout.js";
import * as visualFilters from "../src/visual-filters.js";

const { DEFAULT_VISUAL_FILTER_STATE, enableAllVisualFilters, VISUAL_FILTER_IDS } =
  visualFilters;

// The filters build their registry from the looks' PAINT_PARTS
// (visualPaintPartRegistry); an older build kept VISUAL_PAINT_PART_REGISTRY.
/** @returns {Array<[Record<string, string>, string]>} */
function registeredPaintParts() {
  const exports = /** @type {Record<string, any>} */ (visualFilters);
  const registrations =
    typeof exports.visualPaintPartRegistry === "function"
      ? exports.visualPaintPartRegistry()
      : exports.VISUAL_PAINT_PART_REGISTRY;
  assert.ok(Array.isArray(registrations), "the visual filters list their paint parts");
  return registrations.map((/** @type {Record<string, any>} */ registration) => [
    registration.tag ?? registration.part,
    registration.filterId ?? registration.filter,
  ]);
}

const LOOK_NAMES = Object.freeze([
  "map-skins",
  "obstacles",
  "red-zone",
  "spawns",
  "team-identity",
  "atmosphere",
  "champions",
  "health-text",
  "statuses",
  "movement",
  "basics",
  "ultimates",
  "requiems",
  "heralds",
  "fog",
  "camera",
  "hud",
  "frame-explanation",
  "hotbar",
  "champion-frame",
  "focus-tally",
]);
const SUPPORT_FILES = Object.freeze([
  "look-support.js",
  "map-skin-data.js",
  "strike-routes.js",
]);
const EMPTY_LOOKS = Object.freeze([
  "spawns",
  "team-identity",
  "atmosphere",
  "fog",
  "camera",
  "hud",
  "frame-explanation",
  "hotbar",
  "focus-tally",
]);
const EVENT_LOOK_NAMES = Object.freeze([
  "basics",
  "ultimates",
  "health-text",
  "statuses",
  "requiems",
]);

/** @type {Record<string, Record<string, any>>} */
const looks = Object.fromEntries(
  await Promise.all(
    LOOK_NAMES.map(async (name) => [name, await import(`../src/looks/${name}.js`)]),
  ),
);

/** @template T @param {T} value @returns {T} */
function deepFreeze(value) {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) {
      deepFreeze(child);
    }
  }
  return value;
}

/** @param {unknown} value @param {string} path */
function assertPlainData(value, path) {
  assert.notEqual(typeof value, "function", `${path} holds a function`);
  if (value !== null && typeof value === "object") {
    assert.equal(
      /** @type {Record<string, unknown>} */ (value).nodeType,
      undefined,
      `${path} holds a DOM node`,
    );
    for (const [key, child] of Object.entries(value)) {
      assertPlainData(child, `${path}.${key}`);
    }
  }
}

/** @param {ReadonlyArray<Record<string, any>>} records @param {string} label */
function assertRecordShape(records, label) {
  assert.ok(Array.isArray(records), `${label} returns an array`);
  assert.ok(Object.isFrozen(records), `${label} returns a frozen array`);
  for (const [index, record] of records.entries()) {
    assert.ok(Object.isFrozen(record), `${label}[${index}] is frozen`);
    assert.equal(typeof record.kind, "string", `${label}[${index}] has a kind`);
    assertPlainData(record, `${label}[${index}]`);
  }
}

/** @param {Record<string, any>} tag */
function tagKey(tag) {
  return JSON.stringify(
    Object.entries(tag).sort(([left], [right]) => (left < right ? -1 : 1)),
  );
}

/**
 * @returns {Promise<Array<{name: string, frame: Record<string, any>}>>}
 */
async function fixtureFrames() {
  const fixture = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  const raws = [
    ...Object.entries(fixture.presentations).map(([name, raw]) => [
      `presentations.${name}`,
      raw,
    ]),
    ...Object.entries(fixture.pairs).map(([name, pair]) => [
      `pairs.${name}`,
      /** @type {Record<string, any>} */ (pair).presentation,
    ]),
    ...Object.entries(fixture.red_zone_cases).map(([name, raw]) => [
      `red_zone.${name}`,
      raw,
    ]),
  ];
  return Promise.all(
    raws.map(async ([name, raw]) => ({
      name: String(name),
      frame: await normalizeAuthorizedPresentationFrameV1(structuredClone(raw)),
    })),
  );
}

test("the looks folder holds every agreed look and the shared helpers they use", async () => {
  const files = new Set(await readdir(new URL("../src/looks/", import.meta.url)));
  for (const name of [...LOOK_NAMES.map((look) => `${look}.js`), ...SUPPORT_FILES]) {
    assert.ok(files.has(name), `src/looks/${name} exists`);
  }
});

test("every look module exports the look contract with the right types", () => {
  for (const name of LOOK_NAMES) {
    const look = looks[name];
    for (const exported of ["records", "build", "update", "tooltip"]) {
      assert.equal(typeof look[exported], "function", `${name}.${exported}`);
    }
    assert.equal(look.records.length, 1, `${name}.records(input)`);
    assert.ok(look.build.length >= 2, `${name}.build(records, doc)`);
    assert.equal(look.update.length, 2, `${name}.update(nodes, records)`);
    assert.equal(look.tooltip.length, 1, `${name}.tooltip(record)`);
    assert.ok(
      look.FOOTPRINT !== null &&
        typeof look.FOOTPRINT === "object" &&
        !Array.isArray(look.FOOTPRINT),
      `${name}.FOOTPRINT is an object`,
    );
    assert.ok(Object.isFrozen(look.FOOTPRINT), `${name}.FOOTPRINT is frozen`);
    assert.ok(Object.isFrozen(look.BUDGET), `${name}.BUDGET is frozen`);
    assert.deepEqual(
      Object.keys(look.BUDGET).sort(),
      ["animationsPerRecord", "nodesPerRecord", "persistentNodesPerRecord"],
      `${name}.BUDGET fields`,
    );
    for (const value of Object.values(look.BUDGET)) {
      assert.ok(Number.isInteger(value) && value >= 0, `${name}.BUDGET values`);
    }
    assert.ok(Array.isArray(look.PAINT_PARTS), `${name}.PAINT_PARTS is an array`);
    assert.ok(Object.isFrozen(look.PAINT_PARTS), `${name}.PAINT_PARTS is frozen`);
    for (const registration of look.PAINT_PARTS) {
      assert.deepEqual(Object.keys(registration).sort(), ["filter", "part"]);
      assert.ok(Object.isFrozen(registration) && Object.isFrozen(registration.part));
      assert.ok(["durable", "transient"].includes(registration.part.surface));
      for (const value of Object.values(registration.part)) {
        assert.equal(typeof value, "string", `${name} paint-part tag values`);
      }
      assert.ok(VISUAL_FILTER_IDS.includes(registration.filter), `${name} filter id`);
    }
  }
});

test("every registered paint part has exactly one look owner", () => {
  /** @type {Map<string, {look: string, filter: string}>} */
  const owners = new Map();
  for (const name of LOOK_NAMES) {
    for (const { part, filter } of looks[name].PAINT_PARTS) {
      const key = tagKey(part);
      assert.equal(owners.has(key), false, `${key} is claimed twice`);
      owners.set(key, { look: name, filter });
    }
  }
  const registry = new Map(
    registeredPaintParts().map(([tag, filter]) => [tagKey(tag), filter]),
  );
  assert.deepEqual([...owners.keys()].sort(), [...registry.keys()].sort());
  for (const [key, owner] of owners) {
    assert.equal(owner.filter, registry.get(key), `${key} filter`);
  }
});

test("records() is pure data in every view and gives the same output twice", async () => {
  assert.equal(typeof globalThis.document, "undefined");
  const frames = await fixtureFrames();
  const allFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
  let eventCount = 0;
  for (const { name, frame } of frames) {
    const scene = authorizedPresentationSceneView(frame);
    assert.ok(scene, `${name} has an arena scene`);
    const transform = createViewportTransform({
      worldWidth: scene.map.width,
      worldHeight: scene.map.height,
      viewportWidth: 960,
      viewportHeight: 600,
      padding: 24,
    });
    const frozenScene = deepFreeze(structuredClone(scene));
    const champions = looks.champions.records({
      scene: frozenScene,
      transform,
      showSpawnShield: true,
    });
    const dockLayout = layoutStatusDocks(
      {
        agents: champions.map((/** @type {Record<string, any>} */ champion) => ({
          globalSlot: champion.layoutSlot,
          center: champion.center,
          radius: champion.radius,
          statuses: champion.statuses,
          controlled: champion.controlled,
          selected: champion.selected,
        })),
        viewport: transform.viewportBounds,
      },
      { ...looks.statuses.FOOTPRINT.statusDock, dockGap: 5 },
    );
    const owners = new Map(
      champions.map((/** @type {Record<string, any>} */ champion) => [
        champion.layoutSlot,
        champion.agent,
      ]),
    );
    /** @type {Array<[string, Record<string, any>]>} */
    const inputs = [
      ["map-skins", { transform, map: frozenScene.map, showUnitGrid: true }],
      [
        "red-zone",
        { transform, map: frozenScene.map, redZone: frozenScene.map.red_zone ?? null },
      ],
      ["obstacles", { transform, map: frozenScene.map }],
      ["champions", { scene: frozenScene, transform, showSpawnShield: true }],
      [
        "statuses",
        {
          part: "aura-fields",
          transform,
          fields: frozenScene.aura_fields ?? [],
          tooltipInput: (/** @type {Record<string, any>} */ field) => ({
            field,
            sourceAgent: null,
            audience: frozenScene.audience,
          }),
        },
      ],
      [
        "statuses",
        {
          part: "fact-docks",
          docks: dockLayout.docks.map((placement) => ({
            placement,
            dockKind: "status",
            dimensions: looks.statuses.FOOTPRINT.statusDock,
          })),
          ownerByLayoutSlot: owners,
          audience: frozenScene.audience,
          tooltipInputs: {
            cell: (
              /** @type {unknown} */ status,
              /** @type {unknown} */ recipient,
            ) => ({
              kind: "pov-status",
              status,
              recipient,
            }),
            overflow: (
              /** @type {unknown} */ items,
              /** @type {unknown} */ recipient,
            ) => ({
              kind: "pov-overflow",
              items,
              recipient,
            }),
          },
        },
      ],
      [
        "champion-frame",
        { part: "reach-rings", scene: frozenScene, transform, showRanges: true },
      ],
      ["champion-frame", { part: "next-action-route", scene: frozenScene, transform }],
      [
        "champion-frame",
        { part: "selection-marks", champions, showSelectionReticle: true },
      ],
      [
        "movement",
        {
          scene: frozenScene,
          transform,
          audience: frozenScene.audience,
          ...looks.movement.movementSource(
            frozenScene.audience === "agent_pov"
              ? frame.visual_events
              : frame.latest_events,
            frozenScene.audience,
          ),
        },
      ],
      ["heralds", { part: "lineup", scene: frozenScene, mapTitle: "Fixture Map" }],
      ...EMPTY_LOOKS.map(
        (look) =>
          /** @type {[string, Record<string, any>]} */ ([
            look,
            { scene: frozenScene, transform, audience: frozenScene.audience },
          ]),
      ),
    ];
    const surface = {
      worldToScreen: transform.worldToScreen,
      worldLengthToScreen: transform.worldLengthToScreen,
      viewportBounds: transform.mapBounds,
      protectedRects: [],
    };
    const plan = choreographyPlan.buildChoreographyPlan(frame, surface, allFilters);
    for (const event of plan?.events ?? []) {
      eventCount += 1;
      const frozenEvent = deepFreeze(structuredClone(event));
      for (const look of [...EVENT_LOOK_NAMES, "heralds", "focus-tally", "movement"]) {
        inputs.push([look, { event: frozenEvent }]);
      }
    }
    for (const [lookName, input] of inputs) {
      const frozenInput = deepFreeze({ ...input });
      const first = looks[lookName].records(frozenInput);
      const second = looks[lookName].records(frozenInput);
      assertRecordShape(first, `${name} ${lookName}`);
      assert.equal(
        JSON.stringify(first),
        JSON.stringify(second),
        `${name} ${lookName}`,
      );
    }
  }
  assert.ok(eventCount > 0, "the fixtures exercise plan events");
});

test("each painted plan event belongs to exactly one event look", async () => {
  const frames = await fixtureFrames();
  const allFilters = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);
  /** @type {Record<string, string>} */
  const expectedOwnerByKind = {
    net_health: "health-text",
    regeneration: "health-text",
  };
  const requiemPulses = new Set(["agent_died", "agent_respawned"]);
  // The death announcement and team wave are page chrome: no arena look owns them.
  const chromePulses = new Set(["respawn_wave_occurred", "death_announcement"]);
  /** @type {Set<string>} */
  const claimedKinds = new Set();
  // The fixtures hold no pulses or Ultimates, so every painted kind and pulse
  // meaning is also checked with a minimal event.
  /** @type {Array<Record<string, any>>} */
  const events = [
    { kind: "activation", component: "basic" },
    { kind: "activation", component: "ultimate" },
    { kind: "net_health" },
    { kind: "regeneration" },
    { kind: "status_lifecycle" },
    ...[
      "cooldown_started",
      "cooldown_ready",
      "agent_died",
      "agent_respawned",
      "respawn_wave_occurred",
      "death_announcement",
      "health_regenerated",
    ].map((cueSemantic) => ({ kind: "semantic_pulse", cueSemantic })),
    { kind: "feed_only" },
    { kind: "movement_displacement" },
    { kind: "unknown" },
  ];
  for (const { name, frame } of frames) {
    const scene = authorizedPresentationSceneView(frame);
    assert.ok(scene, `${name} has an arena scene`);
    const transform = createViewportTransform({
      worldWidth: scene.map.width,
      worldHeight: scene.map.height,
      viewportWidth: 960,
      viewportHeight: 600,
      padding: 24,
    });
    const plan = choreographyPlan.buildChoreographyPlan(
      frame,
      {
        worldToScreen: transform.worldToScreen,
        worldLengthToScreen: transform.worldLengthToScreen,
        viewportBounds: transform.mapBounds,
        protectedRects: [],
      },
      allFilters,
    );
    events.push(...(plan?.events ?? []));
  }
  for (const event of events) {
    const owners = EVENT_LOOK_NAMES.filter(
      (name) => looks[name].records({ event }).length > 0,
    );
    if (event.kind === "semantic_pulse" && chromePulses.has(event.cueSemantic)) {
      assert.deepEqual(owners, [], `${event.cueSemantic} is page chrome`);
      assert.equal(
        event.spatial === true,
        false,
        `${event.cueSemantic} is not spatial`,
      );
    } else if (
      ["activation", "net_health", "regeneration", "semantic_pulse"].includes(
        event.kind,
      )
    ) {
      const expected =
        event.kind === "activation"
          ? event.component === "ultimate"
            ? "ultimates"
            : "basics"
          : event.kind === "semantic_pulse"
            ? requiemPulses.has(event.cueSemantic)
              ? "requiems"
              : "statuses"
            : expectedOwnerByKind[event.kind];
      assert.deepEqual(owners, [expected], `${event.kind} ${event.cueSemantic ?? ""}`);
      claimedKinds.add(event.kind);
    } else {
      assert.deepEqual(owners, [], `${event.kind} has no look`);
      if (event.kind === "status_lifecycle") {
        claimedKinds.add("status_lifecycle (no look)");
      }
    }
  }
  assert.deepEqual([...claimedKinds].sort(), [
    "activation",
    "net_health",
    "regeneration",
    "semantic_pulse",
    "status_lifecycle (no look)",
  ]);
});

/** @param {any} element @returns {any[]} */
function descendants(element) {
  return element.children.flatMap((/** @type {any} */ child) => [
    child,
    ...descendants(child),
  ]);
}

// Enough selector support for the looks' own lookups: "[attr]", ".class",
// "tag" and "tag.class".
/** @param {any} element @param {string} selector @returns {boolean} */
function matches(element, selector) {
  const attribute = selector.match(/^\[([a-z-]+)\]$/u);
  if (attribute) return element.hasAttribute(attribute[1]);
  const [tag, className] = selector.startsWith(".")
    ? ["", selector.slice(1)]
    : selector.split(".");
  return (
    (tag === "" || element.tagName === tag) &&
    (className === undefined ||
      (element.getAttribute("class") ?? "").split(" ").includes(className))
  );
}

/** @returns {any} */
function fakeDocument() {
  /** @param {string} tag @returns {any} */
  const node = (tag) => {
    /** @type {Map<string, string>} */
    const attributes = new Map();
    /** @type {any} */
    const element = {
      tagName: tag,
      children: [],
      parentElement: null,
      dataset: {},
      textContent: "",
      setAttribute: (/** @type {string} */ name, /** @type {unknown} */ value) =>
        attributes.set(name, String(value)),
      getAttribute: (/** @type {string} */ name) => attributes.get(name) ?? null,
      hasAttribute: (/** @type {string} */ name) => attributes.has(name),
      removeAttribute: (/** @type {string} */ name) => attributes.delete(name),
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
      get lastElementChild() {
        return element.children.at(-1) ?? null;
      },
      querySelectorAll: (/** @type {string} */ selectors) =>
        descendants(element).filter((/** @type {any} */ child) =>
          selectors.split(",").some((part) => matches(child, part.trim())),
        ),
      querySelector: (/** @type {string} */ selectors) =>
        element.querySelectorAll(selectors)[0] ?? null,
      append: (/** @type {any[]} */ ...nodes) => {
        for (const child of nodes) {
          child.parentElement = element;
          element.children.push(child);
        }
      },
    };
    return element;
  };
  return {
    createElementNS: (/** @type {string} */ _ns, /** @type {string} */ tag) =>
      node(tag),
  };
}

test("movement and heralds draw real records that only they own", () => {
  const transform = createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: 848,
    viewportHeight: 448,
    padding: 24,
  });
  const scene = {
    audience: "researcher",
    map: { width: 20, height: 10, obstacles: [] },
    agents: [
      {
        presentation_key: "walker",
        public_agent_id: "agent-slot-walker",
        class_id: 1,
        team_id: 1,
        position: [5, 5],
        radius: 0.5,
        life_state: "alive",
      },
    ],
    selection: { selected_presentation_key: "walker" },
  };
  const trajectories = [
    {
      agent_presentation_key: "walker",
      agent_public_agent_id: "agent-slot-walker",
      transition_start: { phase: "transition_start", position: [4, 5] },
      successor: { phase: "successor", position: [5, 5] },
    },
  ];
  const still = deepFreeze({ scene, transform, audience: "researcher", returned: [] });
  const moving = deepFreeze({ ...still, trajectories });
  const fall = (/** @type {number} */ points, /** @type {string} */ id) => ({
    victim: { title: `Warrior · ID ${id}`, classId: 2, teamId: 2 },
    killers: [],
    killingTeamId: 1,
    points,
  });
  const heraldInput = deepFreeze({
    event: {
      kind: "herald_card",
      eventId: "t:herald",
      count: 2,
      falls: [fall(1, "6"), fall(2, "7")],
      paintParts: { effect: true },
      phaseStart: 960,
      phaseEnd: 2400,
    },
  });
  const ownKinds = (/** @type {string} */ name) =>
    looks[name].PAINT_PARTS.map(
      (/** @type {Record<string, any>} */ registration) => registration.part.kind,
    );

  assert.deepEqual(looks.movement.records(still), [], "no trajectories, no marks");
  const marks = looks.movement.records(moving);
  assertRecordShape(marks, "movement");
  assert.deepEqual(
    marks.map((/** @type {Record<string, any>} */ mark) => mark.kind).sort(),
    ["focus_trail", "start_spot"],
  );
  for (const mark of marks) {
    assert.ok(ownKinds("movement").includes(mark.kind), `movement owns ${mark.kind}`);
  }
  for (const name of LOOK_NAMES.filter((look) => look !== "movement")) {
    assert.equal(
      JSON.stringify(looks[name].records(moving)),
      JSON.stringify(looks[name].records(still)),
      `${name} ignores trajectories`,
    );
  }

  const plates = looks.heralds.records(heraldInput);
  assertRecordShape(plates, "heralds");
  assert.equal(plates.length, 1);
  assert.equal(plates[0].kind, "herald-plate");
  assert.equal(plates[0].title, "Double Takedown");
  assert.ok(ownKinds("heralds").includes("herald_card"), "heralds owns herald_card");
  assert.deepEqual(
    LOOK_NAMES.filter((name) => looks[name].records(heraldInput).length > 0),
    ["heralds"],
  );

  const doc = /** @type {Document} */ (/** @type {unknown} */ (fakeDocument()));
  for (const [name, records, attribute] of /** @type {const} */ ([
    ["movement", marks, "data-movement-kind"],
    ["heralds", plates, "data-herald-kind"],
  ])) {
    const nodes = /** @type {any[]} */ (looks[name].build(records, doc));
    assert.deepEqual(
      nodes.map((node) => node.getAttribute(attribute)),
      records.map((/** @type {Record<string, any>} */ record) => record.kind),
      `${name} builds one node per record`,
    );
  }
});

test("look footprints match the plan's allocator table while it keeps one", () => {
  const table = /** @type {Record<string, any> | undefined} */ (
    /** @type {Record<string, unknown>} */ (choreographyPlan)
      .CHOREOGRAPHY_PAINT_FOOTPRINTS
  );
  const basics = looks.basics.FOOTPRINT;
  const ultimates = looks.ultimates.FOOTPRINT;
  assert.deepEqual(basics.activation.basic_damage, { width: 44, height: 44 });
  assert.deepEqual(ultimates.chargeOwnership, {
    width: 72,
    height: 22,
    labelLength: 60,
  });
  assert.equal(Object.hasOwn(looks.requiems.FOOTPRINT, "respawnWave"), false);
  if (table === undefined) {
    return;
  }
  for (const [token, footprint] of Object.entries({
    ...basics.activation,
    ...ultimates.activation,
  })) {
    assert.deepEqual(footprint, table.activation[token], token);
  }
  assert.deepEqual(ultimates.chargeOwnership, table.chargeOwnership);
  assert.equal(Object.hasOwn(table, "respawnWave"), false);
  assert.equal(
    basics.route.activationPathPadding,
    table.route.activationPathPadding.default,
  );
  assert.equal(
    ultimates.route.holyWordPathPadding,
    table.route.activationPathPadding.holy_word,
  );
  assert.equal(ultimates.route.chargePathPadding, table.route.chargePathPadding);
  assert.equal(
    basics.route.activationMarkerPadding,
    table.route.activationMarkerPadding,
  );
  assert.equal(
    basics.route.activationCompactMarkerPadding,
    table.route.activationCompactMarkerPadding,
  );
  assert.equal(ultimates.route.chargeMarkerPadding, table.route.chargeMarkerPadding);
});

test("with nothing to draw from, every look returns no records and no nodes", () => {
  for (const name of LOOK_NAMES) {
    const look = looks[name];
    const records = look.records({});
    assert.deepEqual(records, [], name);
    assert.ok(Object.isFrozen(records), name);
    assert.deepEqual(look.build(records, /** @type {any} */ (null)), [], name);
    assert.equal(look.update([], records), undefined, name);
  }
  for (const name of [...EMPTY_LOOKS, "movement", "heralds"]) {
    assert.equal(looks[name].tooltip({ kind: "anything" }), null, name);
  }
});
