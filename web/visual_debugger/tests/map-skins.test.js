/**
 * @file Check the Map Skins look (`src/looks/map-skins.js`) and its generated
 * data (`src/looks/map-skin-data.js`). Every catalogue map 0-51 and scenario
 * 1-8 resolves its tale skin by technical name and by its exact layout, with a
 * realm kit from the nine; a changed layout falls back to its kit floor (and
 * Map Skins off to plain Moonstone); a name-less empty floor is never matched
 * by geometry. records() is frozen plain data, cached for equal inputs and new
 * for a new camera frame; build() reuses its nodes for the same records and
 * update() moves them. The switches add or remove only their own marks:
 * moonlight, floor_grid (coursed joints, or the one-unit grid when asked),
 * spawn_waystones (seals and gates on active pads only), red_zone_floors (the
 * tale's homeland ground, only at the depth the tale was drawn for),
 * arena_rim and survey_frame (which hides the rim). Oracle View and Agent POV
 * get the same floor, except Map 47's Legion Mist, which is left out in Agent
 * POV while the Sight Reach veil is on. Map 50's Heart of Petals is never in
 * the data. Tale stones come only for an exact layout with obstacle skins on.
 * Every built node is export-safe: flat paint, no url(), gradient, mask,
 * filter, clip path, image, use, id or style attribute.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { createViewportTransform } from "../src/layout.js";
import {
  MAP_SKIN_ENTRIES,
  MAP_SKIN_KITS,
  taleFloorTree,
  taleStoneTree,
} from "../src/looks/map-skin-data.js";
import {
  build,
  mapGeometryKey,
  records,
  resolveMapSkin,
  taleStoneRecords,
  update,
} from "../src/looks/map-skins.js";

const DATA = new URL("../../../src/marl_battlegrounds/data/tdm/", import.meta.url);
const MANIFEST = JSON.parse(readFileSync(new URL("manifest.json", DATA), "utf8"));
const MAP_47 = "tdm_map_id_47_teutoburg_forest_test";
const MAP_50 = "tdm_map_id_50_jamylas_grove_test";
const KITS = [
  "moonstone",
  "deepwood",
  "sunfield",
  "dunes",
  "tidewrack",
  "starfall",
  "forge",
  "wonderland",
  "bastion",
];
const SAFE_TAGS = new Set(["g", "path", "ellipse", "circle", "line", "rect", "text"]);

class FakeElement {
  /** @param {string} tagName @param {FakeDocument} doc */
  constructor(tagName, doc) {
    this.tagName = tagName;
    this.ownerDocument = doc;
    /** @type {Map<string, string>} */
    this.attributes = new Map();
    /** @type {FakeElement[]} */
    this.children = [];
    this.textContent = "";
  }
  /** @param {string} name @param {string} value */
  setAttribute(name, value) {
    this.attributes.set(name, value);
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
}

class FakeDocument {
  /** @param {string} _namespace @param {string} tagName */
  createElementNS(_namespace, tagName) {
    return new FakeElement(tagName, this);
  }
}

/** @param {FakeElement[]} nodes @returns {FakeElement[]} */
function flatten(nodes) {
  return nodes.flatMap((node) => [node, ...flatten(node.children)]);
}

/** @param {number} [width] @param {number} [height] */
function transformFor(width = 1100, height = 600) {
  return createViewportTransform({
    worldWidth: 20,
    worldHeight: 10,
    viewportWidth: width,
    viewportHeight: height,
    padding: 24,
  });
}

/**
 * @param {{width: number, height: number, rows: any[], depth?: number}} source
 */
function sceneMap({ width, height, rows, depth = 5 }) {
  const obstacles = rows
    .map((row) =>
      Array.isArray(row)
        ? {
            on: row[7] > 0,
            type: row[0],
            x: row[1],
            y: row[2],
            r: row[3],
            w: row[4],
            h: row[5],
            t: row[6],
          }
        : {
            on: row.is_active === true,
            type: row.obstacle_type_id,
            x: row.x,
            y: row.y,
            r: row.radius,
            w: row.width,
            h: row.height,
            t: row.theta,
          },
    )
    .filter((row) => row.on)
    .map((row, index) => ({
      obstacle_id: String(index),
      kind: Math.round(row.type) === 1 ? "pillar" : "wall",
      center: [row.x, row.y],
      radius: Math.round(row.type) === 1 ? row.r : null,
      width: Math.round(row.type) === 1 ? null : row.w,
      height: Math.round(row.type) === 1 ? null : row.h,
      theta: row.t,
    }));
  return {
    width,
    height,
    obstacles,
    red_zone:
      depth > 0
        ? { depth, team_a_x_range: [0, depth], team_b_x_range: [width - depth, width] }
        : null,
  };
}

/** @param {number} mapId */
function catalogueMap(mapId) {
  const raw = JSON.parse(readFileSync(new URL(`maps/${mapId}.json`, DATA), "utf8"));
  return sceneMap({
    width: raw.map_width,
    height: raw.map_height,
    rows: raw.obstacles,
  });
}

/** @param {number} scenarioId */
function scenarioMap(scenarioId) {
  const raw = JSON.parse(
    readFileSync(new URL(`scenarios/${scenarioId}.json`, DATA), "utf8"),
  ).configuration;
  return sceneMap({
    width: raw.map_width,
    height: raw.map_height,
    rows: raw.obstacle_slots,
    depth: raw.team_deathmatch_red_zone_depth,
  });
}

const PADS = [
  { team_id: 1, position: [0.5, 1], configured_active: true },
  { team_id: 1, position: [0.5, 3], configured_active: false },
  { team_id: 2, position: [19.5, 1], configured_active: true },
];

/**
 * @param {Record<string, any>} [extra]
 * @returns {any}
 */
function input47(extra = {}) {
  return {
    transform: transformFor(),
    map: catalogueMap(47),
    mapKey: MAP_47,
    spawnPads: PADS,
    audience: "researcher",
    ...extra,
  };
}

/** @param {ReadonlyArray<Record<string, any>>} list */
const kinds = (list) => list.map((record) => record.kind);

test("every catalogue map and scenario resolves its tale by name and by layout", () => {
  assert.equal(MANIFEST.maps.length, 52);
  assert.equal(MANIFEST.scenarios.length, 8);
  const sources = [
    ...MANIFEST.maps.map((/** @type {any} */ map) => [
      map.name,
      catalogueMap(map.map_id),
    ]),
    ...MANIFEST.scenarios.map((/** @type {any} */ scenario) => [
      scenario.name,
      scenarioMap(scenario.scenario_id),
    ]),
  ];
  for (const [name, map] of sources) {
    const entry = MAP_SKIN_ENTRIES[name];
    assert.ok(entry, `${name} has an entry`);
    assert.ok(KITS.includes(entry.kit), `${name} kit ${entry.kit}`);
    assert.ok(MAP_SKIN_KITS[entry.kit], `${name} kit colours`);
    assert.equal(entry.tale, true, `${name} has a tale`);
    assert.equal(entry.geometry, mapGeometryKey(map), `${name} geometry`);
    const byName = resolveMapSkin(name, map);
    assert.equal(byName.key, name);
    assert.equal(byName.layoutMatches, true, `${name} layout matches`);
    assert.ok(taleFloorTree(name)?.length, `${name} tale floor`);
    assert.ok(taleStoneTree(name), `${name} tale stones`);
    const tale = records({ transform: transformFor(), map, mapKey: name }).find(
      (record) => record.kind === "tale",
    );
    assert.equal(tale?.mapKey, name, `${name} draws its tale`);
  }
  assert.equal(Object.keys(MAP_SKIN_ENTRIES).length, 60);
  assert.equal(MAP_SKIN_ENTRIES[MAP_47].kit, "deepwood");
  assert.equal(MAP_SKIN_ENTRIES.scenario_5_MWP_vs_MRP.kit, "tidewrack");
});

test("an unnamed map is found by its exact layout, never an empty floor", () => {
  const found = resolveMapSkin(null, catalogueMap(47));
  assert.equal(found.key, MAP_47);
  assert.equal(found.layoutMatches, true);
  const empty = resolveMapSkin(null, { width: 20, height: 10, obstacles: [] });
  assert.equal(empty.key, null);
  const plain = records({ transform: transformFor(), map: { width: 20, height: 10 } });
  assert.ok(!kinds(plain).includes("tale"));
  assert.equal(
    plain.find((record) => record.kind === "floor")?.attributes.fill,
    MAP_SKIN_KITS.moonstone.floor,
  );
  assert.equal(resolveMapSkin("not_a_map", { width: 20, height: 10 }).entry, null);
});

test("a changed layout falls back to its kit; Map Skins off gives plain Moonstone", () => {
  const map = catalogueMap(47);
  const moved = {
    ...map,
    obstacles: map.obstacles.map((obstacle, index) =>
      index === 0
        ? { ...obstacle, center: [obstacle.center[0] + 0.5, obstacle.center[1]] }
        : obstacle,
    ),
  };
  const kit = records(input47({ map: moved }));
  assert.ok(!kinds(kit).includes("tale"));
  assert.equal(
    kit.find((record) => record.kind === "floor")?.attributes.fill,
    MAP_SKIN_KITS.deepwood.floor,
  );
  assert.ok(kinds(kit).includes("inlay"), "the kit's sparse inlay");
  assert.deepEqual(taleStoneRecords({ ...input47({ map: moved }) }), []);
  const off = records(input47({ filters: { map_skins: false } }));
  assert.ok(!kinds(off).includes("tale"));
  assert.ok(!kinds(off).includes("inlay"));
  assert.equal(
    off.find((record) => record.kind === "floor")?.attributes.fill,
    MAP_SKIN_KITS.moonstone.floor,
  );
  assert.ok(kinds(off).includes("edge"), "the edge always draws");
});

test("records are frozen plain data, cached for equal inputs and new for a new camera", () => {
  const first = records(input47());
  assert.ok(Object.isFrozen(first));
  for (const record of first) {
    assert.ok(Object.isFrozen(record));
    assert.equal(typeof record.kind, "string");
    assert.doesNotThrow(() => structuredClone(record));
  }
  assert.equal(records(input47()), first, "equal inputs reuse the result");
  const wider = records(input47({ transform: transformFor(1440, 900) }));
  assert.notEqual(wider, first);
  assert.notEqual(
    wider.find((record) => record.kind === "tale")?.attributes.transform,
    first.find((record) => record.kind === "tale")?.attributes.transform,
  );
  assert.deepEqual(records({}), []);
  assert.deepEqual(build([], /** @type {any} */ (null)), []);
});

test("build reuses its nodes for the same records and update moves them", () => {
  const doc = new FakeDocument();
  const list = records(input47());
  const nodes = build(list, /** @type {any} */ (doc));
  assert.equal(nodes.length, list.length);
  const again = build(list, /** @type {any} */ (doc));
  assert.notEqual(again, nodes);
  assert.deepEqual(again, nodes, "the same node objects");
  assert.ok(nodes.every((node, index) => node === again[index]));
  const moved = records(input47({ transform: transformFor(1440, 900) }));
  update(/** @type {any} */ (nodes), moved);
  const taleIndex = moved.findIndex((record) => record.kind === "tale");
  assert.equal(
    /** @type {any} */ (nodes[taleIndex]).getAttribute("transform"),
    moved[taleIndex].attributes.transform,
  );
  const rebuilt = build(list, /** @type {any} */ (doc));
  assert.ok(
    rebuilt.every((node, index) => node !== nodes[index]),
    "old records rebuild",
  );
});

test("each switch adds or removes only its own marks", () => {
  const base = kinds(records(input47()));
  for (const kind of ["tale", "moonlight", "joints", "seals", "gate", "edge"]) {
    assert.ok(base.includes(kind), `default has ${kind}`);
  }
  assert.ok(!base.includes("parapet") && !base.includes("ruler-label"));
  const noMoon = kinds(records(input47({ filters: { moonlight: false } })));
  assert.ok(!noMoon.includes("moonlight"));
  const noGrid = kinds(records(input47({ filters: { floor_grid: false } })));
  assert.ok(!noGrid.includes("joints"));
  const unitGrid = records(input47({ showUnitGrid: true }));
  assert.equal(unitGrid.filter((record) => record.kind === "grid-line").length, 19 + 9);
  assert.ok(!kinds(unitGrid).includes("joints"));
  const noWay = kinds(records(input47({ filters: { spawn_waystones: false } })));
  assert.ok(!noWay.includes("seals") && !noWay.includes("gate"));
  const rim = kinds(records(input47({ filters: { arena_rim: true } })));
  assert.ok(rim.includes("parapet") && rim.includes("north-shade"));
  const ruler = kinds(
    records(input47({ filters: { arena_rim: true, survey_frame: true } })),
  );
  assert.ok(ruler.includes("ruler-label") && !ruler.includes("parapet"));
  const taleMix = (/** @type {Record<string, boolean>} */ filters) =>
    records(input47({ filters })).find((record) => record.kind === "tale")?.mix;
  assert.notEqual(taleMix({ moonlight: false }), taleMix({}));
  assert.notEqual(taleMix({ red_zone_floors: false }), taleMix({}));
});

test("seals and gates mark active pads only, in their own team's colour", () => {
  const list = records(input47());
  const seals = list.find((record) => record.kind === "seals");
  assert.equal(seals?.attributes.d.split("M").length - 1, 2, "two active pads");
  const gates = list.filter((record) => record.kind === "gate");
  assert.deepEqual(
    gates.map((record) => record.className),
    ["map-skin-gate map-skin-gate--a", "map-skin-gate map-skin-gate--b"],
  );
  assert.equal(list.filter((record) => record.kind === "gate-split").length, 1);
});

test("the homeland ground follows the recorded Red Zone depth", () => {
  const mixOf = (/** @type {Record<string, any>} */ map) =>
    records(input47({ map })).find((record) => record.kind === "tale")?.mix ?? -1;
  const red = 1 << 3;
  assert.notEqual(mixOf(catalogueMap(47)) & red, 0);
  assert.equal(mixOf({ ...catalogueMap(47), red_zone: null }) & red, 0);
  const shallow = {
    ...catalogueMap(47),
    red_zone: { depth: 3, team_a_x_range: [0, 3], team_b_x_range: [17, 20] },
  };
  assert.equal(mixOf(shallow) & red, 0);
});

test("Oracle View and Agent POV share the floor except Map 47's Legion Mist", () => {
  const map12 = catalogueMap(12);
  const name12 = "tdm-map-id-12-foxhole-training";
  const oracle = records({
    transform: transformFor(),
    map: map12,
    mapKey: name12,
    audience: "researcher",
  });
  const pov = records({
    transform: transformFor(),
    map: map12,
    mapKey: name12,
    audience: "agent_pov",
    filters: { pov_sight_veil: true },
  });
  assert.deepEqual(pov, oracle);
  const doc = new FakeDocument();
  /** @param {Record<string, any>} extra */
  const mist = (extra) =>
    flatten(
      /** @type {any} */ (build(records(input47(extra)), /** @type {any} */ (doc))),
    ).filter((node) => node.getAttribute("fill") === "#C9D3CD").length;
  assert.ok(mist({}) > 0, "Oracle View keeps the mist");
  assert.ok(mist({ audience: "agent_pov", filters: { pov_sight_veil: false } }) > 0);
  assert.equal(mist({ audience: "agent_pov", filters: { pov_sight_veil: true } }), 0);
});

test("Map 50's Heart of Petals is never in the data", () => {
  const text = JSON.stringify(taleFloorTree(MAP_50));
  assert.ok(text.length > 1000);
  assert.ok(!text.includes("#2A2724"), "no heart colour");
});

test("tale stones come only for an exact layout with obstacle skins on", () => {
  const stones = taleStoneRecords(input47());
  assert.equal(stones.length, 1);
  assert.equal(stones[0].layer, "stones");
  assert.deepEqual(
    taleStoneRecords(input47({ filters: { obstacle_skins: false } })),
    [],
  );
  assert.deepEqual(taleStoneRecords(input47({ filters: { map_skins: false } })), []);
  const doc = new FakeDocument();
  const [group] = build(stones, /** @type {any} */ (doc));
  assert.equal(group.getAttribute("class"), "map-skin-tale-stones");
  assert.ok(flatten([/** @type {any} */ (group)]).length > 10);
  const flat = taleStoneRecords(
    input47({ filters: { obstacle_relief: false, obstacle_shadows: false } }),
  );
  assert.notEqual(flat[0].mix, stones[0].mix);
});

test("every built node is export-safe", () => {
  const doc = new FakeDocument();
  /** @type {Array<Record<string, boolean>>} */
  const filterSets = [
    {},
    { arena_rim: true },
    { survey_frame: true },
    { map_skins: false },
  ];
  for (const name of Object.keys(MAP_SKIN_ENTRIES)) {
    taleFloorTree(name);
  }
  /** @type {Array<[string, Record<string, any>]>} */
  const maps = [
    [MAP_47, catalogueMap(47)],
    [MAP_50, catalogueMap(50)],
    ["tdm_map_id_27_stairway_to_heaven_training", catalogueMap(27)],
    ["scenario_8_WRP_vs_RP", scenarioMap(8)],
  ];
  for (const [name, map] of maps) {
    for (const filters of filterSets) {
      const list = [
        ...records({
          transform: transformFor(),
          map,
          mapKey: name,
          spawnPads: PADS,
          filters,
        }),
        ...taleStoneRecords({ transform: transformFor(), map, mapKey: name, filters }),
      ];
      for (const node of flatten(
        /** @type {any} */ (build(list, /** @type {any} */ (doc))),
      )) {
        assert.ok(SAFE_TAGS.has(node.tagName), `${name} tag ${node.tagName}`);
        for (const [attribute, value] of node.attributes) {
          assert.ok(
            !["id", "style", "mask", "filter", "clip-path"].includes(attribute),
            attribute,
          );
          assert.doesNotMatch(value, /url\(|gradient/iu, `${name} ${attribute}`);
        }
      }
    }
  }
});
