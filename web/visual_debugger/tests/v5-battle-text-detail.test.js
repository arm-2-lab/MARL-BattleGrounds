/**
 * @file Check the V5 Battle Text Detail add-on of `src/looks/health-text.js`
 * (switch battle_text_detail, off by default). Contracts: lifeWeight sorts a
 * change by its share of the champion's maximum life (under 10% small, under
 * 30% medium, else large) and is null without a maximum; switched on, an Oracle
 * number is 13, 15 or 18 px but never below its plain size (the Ultimate and
 * killing-blow floor stays), carries data-weight, pops 6 px lower and drifts
 * into place over 600 ms, and has a share underline (Ink backing, one bar per
 * recorded attacker sized by share, largest first, 3 px for an Ultimate, Rime
 * for a Freezing Trap, a Dawn Gold healing bar last) only when the recorded
 * sources add up to the recorded damage, plus "Beyond Death N" 4 px right of
 * the number on an overkill, drawn only when it covers no other body or number
 * box and stays in view (and never without a projection);
 * Agent POV gets only the size tier and the drift (no Ultimate floor, no
 * underline, no tag); a Rest number only drifts; switched off, numbers are the
 * plain 13 or 15 px with the plain 4 px pop and no data-weight; lifeNumberCueBox
 * grows the box for an 18 px number and its underline and is null while off;
 * the markup is export-safe.
 */
import assert from "node:assert/strict";
import test from "node:test";

import * as healthText from "../src/looks/health-text.js";
import {
  draw,
  EXPORT_UNSAFE,
  filtersWith,
  HUNTER_KEY,
  healthRow,
  MAGE_KEY,
  planFor,
  RECIPIENT_KEY,
  TRANSFORM,
} from "./v5-health-text-support.js";

const ON = filtersWith({ battle_text_detail: true });

test("lifeWeight sorts a change by its share of the champion's maximum life", () => {
  assert.equal(
    healthText.lifeWeight(healthRow({ healthBefore: 80, healthAfter: 75 })),
    "small",
  );
  assert.equal(
    healthText.lifeWeight(healthRow({ healthBefore: 80, healthAfter: 70 })),
    "medium",
  );
  assert.equal(
    healthText.lifeWeight(healthRow({ healthBefore: 80, healthAfter: 50 })),
    "large",
  );
  assert.equal(healthText.lifeWeight(healthRow({ maximumHealth: undefined })), null);
});

test("Oracle sizes follow the tier but never drop below the plain size", () => {
  const plan = planFor("researcher");
  const sized = (/** @type {Record<string, any>} */ event, p = plan) =>
    draw(event, p, ON).group.querySelector(".combat-number");
  const small = sized(healthRow({ healthBefore: 80, healthAfter: 75 }));
  assert.equal(small?.getAttribute("font-size"), "13");
  assert.equal(small?.getAttribute("data-weight"), "small");
  assert.equal(
    sized(healthRow({ healthBefore: 80, healthAfter: 70 }))?.getAttribute("font-size"),
    "15",
  );
  assert.equal(
    sized(healthRow({ healthBefore: 80, healthAfter: 50 }))?.getAttribute("font-size"),
    "18",
  );
  const ultimatePlan = planFor("researcher", {
    events: [
      {
        kind: "activation",
        component: "ultimate",
        sourcePresentationKey: MAGE_KEY,
        targetPresentationKey: RECIPIENT_KEY,
      },
    ],
  });
  const floor = sized(healthRow({ healthBefore: 80, healthAfter: 75 }), ultimatePlan);
  assert.equal(floor?.getAttribute("font-size"), "15");
  assert.equal(floor?.getAttribute("data-weight"), "small");
});

test("the share underline is drawn only from recorded sources that add up", () => {
  const plan = planFor("researcher", {
    events: [
      {
        kind: "activation",
        component: "ultimate",
        sourcePresentationKey: HUNTER_KEY,
        targetPresentationKey: RECIPIENT_KEY,
      },
    ],
  });
  const event = healthRow({
    healthBefore: 80,
    healthAfter: 52,
    totalDamage: 30,
    totalHealing: 2,
    damageSources: [
      { sourcePresentationKey: HUNTER_KEY, sourceClassId: 3, amount: 10 },
      { sourcePresentationKey: MAGE_KEY, sourceClassId: 1, amount: 20 },
    ],
  });
  const drawn = draw(event, plan, ON);
  const backing = drawn.group.querySelector(".combat-underline__backing");
  assert.ok(backing);
  const bars = drawn.group.querySelectorAll(".combat-underline__bar");
  assert.deepEqual(
    bars.map((bar) => [
      bar.getAttribute("data-paint"),
      bar.getAttribute("stroke-width"),
    ]),
    [
      ["mage", "2"],
      ["trap", "3"],
      ["heal", "2"],
    ],
  );
  const lengths = bars.map(
    (bar) => Number(bar.getAttribute("x2")) - Number(bar.getAttribute("x1")),
  );
  assert.ok(lengths[0] > lengths[1] && lengths[1] > lengths[2]);
  assert.ok(lengths.every((length) => length >= 3.999));
  assert.ok(
    Number(backing.getAttribute("y1")) >
      Number(drawn.group.querySelector(".combat-number")?.getAttribute("y")),
  );
  assert.doesNotMatch(drawn.group.markup(), EXPORT_UNSAFE);
  const mismatch = draw({ ...event, totalDamage: 31 }, plan, ON);
  assert.equal(mismatch.group.querySelector(".combat-underline__bar"), null);
});

test("an overkill gets Beyond Death N beside the number only where it fits", () => {
  const event = healthRow({
    healthBefore: 14,
    healthAfter: 0,
    totalDamage: 20,
    totalHealing: 0,
  });
  /** @param {any[]} bodies @param {any[]} [events] */
  const planWith = (bodies, events = []) => ({
    ...planFor("researcher"),
    bodies: [
      { presentationKey: RECIPIENT_KEY, start: [1, 1], end: [7.5, 5] },
      ...bodies,
    ],
    events,
  });
  const open = draw(
    event,
    planWith([{ presentationKey: MAGE_KEY, end: [20, 12] }]),
    ON,
    {
      transform: TRANSFORM,
    },
  );
  assert.equal(open.records[0].tagClear, true);
  const tag = open.group.querySelector(".combat-number__tag");
  assert.equal(tag?.textContent, "Beyond Death 6");
  assert.equal(tag?.getAttribute("text-anchor"), "start");
  assert.ok(Number(tag?.getAttribute("x")) > 300 + 4);
  assert.equal(tag?.getAttribute("fill-opacity"), "0.75");
  // A body (8.5, 4.2 in world units = 340, 168 px) where the tag would go: the tag is shed.
  const body = draw(
    event,
    planWith([{ presentationKey: MAGE_KEY, end: [8.5, 4.2] }]),
    ON,
    {
      transform: TRANSFORM,
    },
  );
  assert.equal(body.group.querySelector(".combat-number__tag"), null);
  assert.equal(body.group.querySelector(".combat-number")?.textContent, "−14");
  const number = draw(
    event,
    planWith(
      [],
      [
        {
          eventId: "other",
          cueBounds: { left: 330, top: 160, right: 360, bottom: 176 },
        },
      ],
    ),
    ON,
    { transform: TRANSFORM },
  );
  assert.equal(number.group.querySelector(".combat-number__tag"), null);
  const edge = draw({ ...event, cue: { x: 960, y: 168 } }, planWith([]), ON, {
    transform: TRANSFORM,
  });
  assert.equal(edge.group.querySelector(".combat-number__tag"), null);
  const unknown = draw(event, planWith([]), ON);
  assert.equal(unknown.group.querySelector(".combat-number__tag"), null);
  const plain = draw(event, planWith([]), null, { transform: TRANSFORM });
  assert.equal(plain.group.querySelector(".combat-number__tag"), null);
});

test("Agent POV gets only the size tier and the drift", () => {
  const event = healthRow({
    healthBefore: 80,
    healthAfter: 50,
    totalDamage: 30,
    damageSources: [{ sourcePresentationKey: MAGE_KEY, sourceClassId: 1, amount: 30 }],
  });
  const plan = planFor("agent_pov", {
    events: [
      {
        kind: "activation",
        component: "ultimate",
        sourcePresentationKey: MAGE_KEY,
        targetPresentationKey: RECIPIENT_KEY,
      },
    ],
  });
  const drawn = draw(event, plan, ON);
  const number = drawn.group.querySelector(".combat-number");
  assert.equal(number?.getAttribute("font-size"), "18");
  assert.equal(number?.getAttribute("data-tone"), "damage");
  assert.equal(drawn.group.querySelector(".combat-number-set"), null);
  assert.equal(drawn.group.querySelector(".combat-underline__bar"), null);
  const small = draw(healthRow({ healthBefore: 80, healthAfter: 75 }), plan, ON);
  assert.equal(
    small.group.querySelector(".combat-number")?.getAttribute("font-size"),
    "13",
  );
  const pop = drawn.animationSpecs.find((spec) => spec.id.endsWith(":number-pop"));
  assert.equal(pop.element, number);
  assert.equal(pop.options.duration, 600);
});

test("the drift pops 10 px low and rests in place; off keeps the plain pop", () => {
  const drawn = draw(
    healthRow({ healthBefore: 80, healthAfter: 50 }),
    planFor("researcher"),
    ON,
  );
  const pop = drawn.animationSpecs.find((spec) => spec.id.endsWith(":number-pop"));
  assert.equal(pop.options.duration, 600);
  assert.equal(pop.options.fill, "backwards");
  assert.equal(pop.options.delay, 460);
  assert.equal(pop.keyframes[0].opacity, 0);
  assert.match(pop.keyframes[0].transform, /translate\(0px, 10px\) scale\(1\.2\)/u);
  assert.equal(pop.keyframes.at(-1).transform, "translate(0px, 0px) scale(1)");
  const rest = draw(
    {
      kind: "regeneration",
      value: 3,
      recipient: { x: 10, y: 10 },
      phaseStart: 460,
      paintParts: { battleText: true, effect: true },
    },
    planFor("researcher"),
    ON,
  );
  const restPop = rest.animationSpecs.find((spec) => spec.id.endsWith(":number-pop"));
  assert.equal(restPop.options.duration, 600);
  assert.equal(
    rest.group.querySelector(".combat-number")?.getAttribute("data-weight"),
    null,
  );
  const off = draw(
    healthRow({ healthBefore: 80, healthAfter: 50 }),
    planFor("researcher"),
    null,
  );
  const number = off.group.querySelector(".combat-number");
  assert.equal(number?.getAttribute("font-size"), "13");
  assert.equal(number?.getAttribute("data-weight"), null);
  const plainPop = off.animationSpecs.find((spec) => spec.id.endsWith(":number-pop"));
  assert.equal(plainPop.options.duration, 80);
});

test("lifeNumberCueBox grows for an 18 px number and its underline, and is null while off", () => {
  const event = healthRow({
    healthBefore: 200,
    healthAfter: 0,
    maximumHealth: 200,
    totalDamage: 200,
    damageSources: [{ sourcePresentationKey: MAGE_KEY, sourceClassId: 1, amount: 200 }],
    paintParts: { battleText: true, battleTextDetail: true },
  });
  const box = healthText.lifeNumberCueBox(event);
  assert.ok(box);
  assert.ok(box.right > healthText.FOOTPRINT.netText.width / 2);
  assert.ok(box.bottom > healthText.FOOTPRINT.netText.height / 2);
  assert.equal(
    healthText.lifeNumberCueBox({ ...event, paintParts: { battleText: true } }),
    null,
  );
});
