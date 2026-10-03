/**
 * @file Check the Agent POV button's title (src/shell.js
 * agentPovButtonTitle). It depends on the product only, never on the view the
 * page loaded in: the BattleClient, which opens in Agent POV, says "...and the
 * default here"; the DevClient combat screen and the Replay Viewer, which open
 * in Oracle View, say only what Agent POV shows. So an Oracle -> POV -> Oracle
 * round trip and a reload in either view give the same title.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { agentPovButtonTitle } from "../src/shell.js";

test("only the BattleClient calls Agent POV its default", () => {
  assert.equal(
    agentPovButtonTitle({ replay: false, devClient: false }),
    "Shows only what your team can see. The fair view for play, and the default here.",
  );
  for (const product of [
    { replay: false, devClient: true },
    { replay: true, devClient: false },
    { replay: true, devClient: true },
  ]) {
    assert.equal(
      agentPovButtonTitle(product),
      "Shows only what your team can see. The fair view for play.",
    );
  }
});
