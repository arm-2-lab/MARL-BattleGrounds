# Browser Tools

Use **BattleClient** for local live play and **Replay Viewer** for recorded games.
Both use the shared combat drawing and inspection tools. Python owns simulation,
action legality, validation and information rights; the browser displays those
facts and accepts input. Display changes do not run a simulator turn.

| Task | Public Route |
| --- | --- |
| Play or watch a live match | `marl_bgs.battle_client(map_id=0, ...)` |
| Watch a fixed scenario | `marl_bgs.battle_client(scenario_id=1, team_a=...)` |
| Open an installed replay | `python -m marl_battlegrounds replay PATH` |
| Open a repository sample | `./scripts/dev/run_replay_viewer.sh --sample-replay NAME` |
| List repository demonstrations | `./scripts/dev/run_replay_viewer.sh --list-scenarios` |
| Draw one exact replay frame | `./scripts/dev/run_replay_viewer.sh --replay PATH --static --frame-index N` |

See the [BattleClient guide](../battle_client.md) and [Replay Viewer guide](replay_viewer.md).
Ordinary users need no Node installation. Contributor checks use Node separately.
Local authoring tools are private and are not part of the public package or gate.

## Built-In Controller Notes

The earlier wall-stall results for ALPHA version 2 and BETA version 4 are
historical; later study found longer stalls. The shared `scenario_5` BETA
behavior serves Scenarios 3 and 5; the old
standalone Scenario 3, Scripted TDM, and separate Reactive MRP interfaces were
removed. Historical recording identities remain readable.

GAMMA v2 is BETA plus three rules for Warriors, Mages, Hunters and Rogues: with
no enemy in view they walk toward the middle of the enemy spawn pads; they never
damage an enemy with 2 or more Hunter Trap ticks left; and Hunters start a new
Trap only on an untrapped enemy they can reach now, choosing Priest, Mage,
Rogue, Warrior, Hunter in that order. Its Priests keep BETA's rules
([A43](../design/specification_amendments.md#a43-reactive-tdm-gamma)).

These are diagnostic/scenario-pressure tools, not official baselines or Big 12
entrants. Future official evaluation definitions must bind the same versioned
controller to every compared treatment; saved physical scenarios stay
independent of controllers. See the [BattleClient workflow](../battle_client.md),
[A26](../design/specification_amendments.md#a26-scenario-pressure-controllers-and-behavioral-ablations),
and
[A32](../design/specification_amendments.md#a32-scenario-5-shoulder-bypass-and-fallback-prey).

## Display And Evidence Boundaries

Both clients draw the Runefield look from one page. The BattleClient opens in
Agent POV, which shows only what the viewing team sees. The Replay Viewer and
the private DevClient combat screen open in Oracle View, which shows
everything. The DevClient combat screen is the BattleClient screen plus its
private setup bar, with the same Agent POV rules. One command table owns every
key in every client; press **?** in the page for the current list.
[A45](../design/specification_amendments.md#a45-runefield-client-presentation)
records these rules, the beat timing and the Sight Reach veil.

The visual filters and Ranges change display, not simulation or recorded data.
SharedObs display combines permitted same-tick sensor views, as described in
[A17](../design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation).
It creates no extra policy input. Human Oracle inspection is diagnostic play,
not an autonomous evaluation. A working client or replay establishes no learned
competence.

Follow the [Documentation Standard](documentation_standard.md), [Quality Gates](quality_gates.md)
and [Dependency Policy](dependency_policy.md) when changing these tools.
