/**
 * @file The one Life bar every client draws for a champion's life (owner, 2
 * October 2026): the word "Life", a thin 6 px track whose fill is the share of
 * life left, and the value "77 / 80" beside it, all in one row. The fill is
 * green above half, yellow at half or less and red at 30% or less
 * (LOW_LIFE_FRACTION of `display.js`, the client's low-life warning line).
 * No color legend goes with it: the colors explain themselves.
 *
 * Who uses it: the Champion Card (`tooltip.js`, where the bar shares a grid
 * with the Ultimate bar), the BattleClient and DevClient deck's Your Champion
 * box, Target chip and Watch Only overlay (`tactics-turn.js`), and the Replay
 * Viewer's Champion Frame (`looks/champion-frame.js`), Team Rosters frames
 * (`looks/hud.js`) and Agent tab (`panels.js`). The arena tokens keep their
 * own Life Vessel (`looks/champions.js`) and the scorebug portraits their own
 * small bar. Its look lives in `src/styles/life-bar.css`.
 *
 * Agent POV: this module draws only the numbers its caller passes. A caller
 * never passes the life of a champion its audience cannot see; it shows its
 * own words instead (for example the Target chip's "Out Of Sight").
 */
import { formatLifeNumber, LOW_LIFE_FRACTION } from "./display.js";

/**
 * The words and share of one Life bar, ready to draw (lifeBarModel builds it;
 * createLifeBar draws it).
 *
 * - text: life now as printed by formatLifeNumber ("77", "9.4", "0").
 * - maximumText: full life as printed ("80").
 * - fraction: the share of life left, 0 to 1; the fill's width.
 * - tone: "good" (green, above half), "warn" (yellow, half or less) or "low"
 *   (red, 30% or less).
 *
 * @typedef {Readonly<{
 *   text: string,
 *   maximumText: string,
 *   fraction: number,
 *   tone: "good" | "warn" | "low",
 * }>} LifeBarModel
 */

/**
 * Return the Life bar's color for a life: "good" (green) above half, "warn"
 * (yellow) at half or less, "low" (red) at LOW_LIFE_FRACTION (30%) or less. A
 * missing or zero maximum counts as full life when current is above 0. Pure.
 *
 * @param {number} current Life now, in hit points.
 * @param {number | null} maximum Full life, in hit points, or null.
 * @returns {"good" | "warn" | "low"}
 * @example lifeTone(40, 80) // "warn"; lifeTone(24, 80) // "low"
 */
export function lifeTone(current, maximum) {
  const share =
    typeof maximum === "number" && maximum > 0
      ? current / maximum
      : current > 0
        ? 1
        : 0;
  return share > 0.5 ? "good" : share > LOW_LIFE_FRACTION ? "warn" : "low";
}

/**
 * Turn a champion's life into a Life bar model. A negative current life
 * reads 0. Pure.
 *
 * @param {unknown} current Life now, in hit points.
 * @param {unknown} maximum Full life, in hit points.
 * @returns {LifeBarModel | null} A new frozen model, or null when current or
 *   maximum is missing or not a finite number (the caller then shows its own
 *   words, or nothing).
 * @example lifeBarModel(77, 80)
 * // {text: "77", maximumText: "80", fraction: 0.9625, tone: "good"}
 */
export function lifeBarModel(current, maximum) {
  if (
    typeof current !== "number" ||
    !Number.isFinite(current) ||
    typeof maximum !== "number" ||
    !Number.isFinite(maximum)
  ) {
    return null;
  }
  return Object.freeze({
    text: formatLifeNumber(Math.max(0, current)) ?? "0",
    maximumText: formatLifeNumber(Math.max(0, maximum)) ?? "0",
    fraction: maximum > 0 ? Math.max(0, Math.min(1, current / maximum)) : 0,
    tone: lifeTone(current, maximum),
  });
}

/**
 * Draw one Life bar: a new `div.life-bar` (data-life "good", "warn" or "low")
 * holding `span.life-bar__label` ("Life"), `span.life-bar__track` with its
 * `span.life-bar__fill` (width: the share, set through the CSS object model,
 * which the page's content policy allows) and `span.life-bar__value` ("77"
 * then a dimmer `small` " / 80"). Text is set as text, never HTML; nothing is
 * attached to the page. A screen reader reads "Life 77 / 80".
 *
 * Inside the Champion Card the bar's three parts join the card's bars grid
 * (life-bar.css), so its track lines up with the Ultimate bar's track.
 *
 * @param {Document} doc Creates the nodes.
 * @param {LifeBarModel} model From lifeBarModel.
 * @returns {HTMLDivElement} The new, unattached bar.
 */
export function createLifeBar(doc, model) {
  /**
   * @param {string} tag
   * @param {string} className
   * @param {string} [text]
   */
  const el = (tag, className, text) => {
    const node = doc.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const bar = /** @type {HTMLDivElement} */ (el("div", "life-bar"));
  bar.dataset.life = model.tone;
  const track = el("span", "life-bar__track");
  const fill = el("span", "life-bar__fill");
  fill.style.width = `${(model.fraction * 100).toFixed(1)}%`;
  track.append(fill);
  const value = el("span", "life-bar__value", model.text);
  value.append(el("small", "", ` / ${model.maximumText}`));
  bar.append(el("span", "life-bar__label", "Life"), track, value);
  return bar;
}
