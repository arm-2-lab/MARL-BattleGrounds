/**
 * @file The dark "Opening Replay" page that Open Replay writes into its new
 * tab (BattleClient and DevClient; owner, 3 Oct 2026, Micro List M21).
 *
 * Open Replay opens an empty tab at the click (so the browser allows it),
 * then points it at the Replay Viewer's address when the game answers. The
 * Replay Viewer holds its page until every agent's Agent POV is built (a few
 * seconds after a full game), and the browser keeps showing the tab's
 * current page until then, so without this the tab stays blank and white.
 * writeOpeningReplayPage fills that empty tab with one short dark page in the
 * client's own colors instead.
 *
 * What it owns: only the empty tab's title and body. The tab inherits the
 * page's Content Security Policy (style-src 'self'), so every style is set
 * through the element's style object (allowed), never a style attribute or
 * a style element. No timers, storage or network.
 */

/**
 * @typedef {{
 *   background: string,
 *   text: string,
 *   muted: string,
 *   font: string,
 * }} OpeningReplayColors
 *   CSS values for the tab: background (the client's --ink), text (--text),
 *   muted (--muted, the second line) and font (the page's font-family).
 */

/** Fallback colors: the BattleClient tokens in base.css (--ink, --text, --muted). */
export const OPENING_REPLAY_FALLBACK_COLORS = Object.freeze({
  background: "#0A0D14",
  text: "#E6EAF2",
  muted: "#8C97AD",
  font: "system-ui, sans-serif",
});

/**
 * Fill Open Replay's empty new tab with the dark "Opening Replay" page.
 *
 * The tab gets the title "Opening Replay…", a dark background and, centered,
 * the heading "Opening Replay" with one line under it: "Getting every
 * agent's view ready. The Replay Viewer opens here by itself." Any empty
 * color value falls back to OPENING_REPLAY_FALLBACK_COLORS. A tab that cannot
 * be written (closed, or not same-origin any more) keeps its page, and
 * nothing is thrown.
 *
 * Side effects: changes only the tab's document (title, html and body
 * styles, body children).
 *
 * @param {Window | null} tab The window from window.open("about:blank"), or
 *   null when the browser blocked it (then nothing happens).
 * @param {Partial<OpeningReplayColors>} [colors] The client's colors; default
 *   the fallbacks.
 * @returns {boolean} True when the page was written.
 */
export function writeOpeningReplayPage(tab, colors = {}) {
  if (tab === null) return false;
  const pick = (/** @type {keyof OpeningReplayColors} */ name) =>
    (colors[name] ?? "").trim() || OPENING_REPLAY_FALLBACK_COLORS[name];
  try {
    const doc = tab.document;
    const root = doc.documentElement;
    const body = doc.body ?? root.appendChild(doc.createElement("body"));
    doc.title = "Opening Replay…";
    root.style.setProperty("color-scheme", "dark");
    root.style.setProperty("background", pick("background"));
    for (const [property, value] of /** @type {const} */ ([
      ["margin", "0"],
      ["min-height", "100vh"],
      ["display", "grid"],
      ["place-content", "center"],
      ["gap", "8px"],
      ["padding", "16px"],
      ["box-sizing", "border-box"],
      ["text-align", "center"],
      ["background", pick("background")],
      ["color", pick("text")],
      ["font-family", pick("font")],
    ])) {
      body.style.setProperty(property, value);
    }
    const heading = doc.createElement("h1");
    heading.textContent = "Opening Replay";
    heading.style.setProperty("margin", "0");
    heading.style.setProperty("font-size", "22px");
    heading.style.setProperty("font-weight", "600");
    const line = doc.createElement("p");
    line.textContent =
      "Getting every agent's view ready. The Replay Viewer opens here by itself.";
    line.style.setProperty("margin", "0");
    line.style.setProperty("font-size", "14px");
    line.style.setProperty("color", pick("muted"));
    body.replaceChildren(heading, line);
    return true;
  } catch {
    return false;
  }
}
