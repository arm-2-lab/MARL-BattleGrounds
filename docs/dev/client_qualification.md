# Client Qualification

The September 2026 client checks cover local BattleClient use, the public Replay
Viewer and the separately kept private DevClient. The October 2026 Runefield
look has its own checks, described below. These are software checks. They do not show trained skill, lower sample complexity
or learned coordination.

## Installed Use

The candidate wheel and a wheel rebuilt from its source distribution have equal
bytes. Fresh non-editable base, training-extra and source-distribution installs
ran with every checkout, private toolkit and build archive hidden. Python modules
came from `site-packages`; browser files came from installed package resources.
Node was outside the application environments and served only browser automation.

The actual browser checks covered Manual and native policy turns, exact fixed
scenario starts, Finish And Save, a separate Viewer, reopening, restart and Exit.
A copied tutorial also completed its 2v3 flow. Removing an installed required
browser asset made both launchers fail with that file's name. Neither found a
checkout copy. The archives contain 52 maps and eight scenarios, with no private
authoring code or assets. These checks qualify local installation; no package
index release or hosted service is implied.

## Methods And Scenarios

Real actor exports and complete learner checkpoints were loaded for recurrent
MAPPO, recurrent IPPO, feedforward MAPPO, feedforward IPPO, QMIX and recurrent
PQN-VDN. The matrix includes shared, class-shared and separate actor parameters,
saved input scales of 1 and 0.01, and world/left spawn frames. Accepted actions
and carried memory agreed with the existing System execution owner. Restart
created fresh memory without loading the same methods again.

These artifacts contain seeded initial weights and zero training updates. Each
full checkpoint and paired actor has its real saved identity. They establish
format, settings and execution compatibility, not competence. Malformed metadata,
repeated sources, separate objects, custom JAX methods, opaque host methods and
Manual ownership have separate focused checks.

The three main tutorial matchups use 5v5: one human alongside a policy, two
policies per team, and ten different controllers. A 2v3 variation uses repeated
classes. Exact scenario 5 examples keep all ten configured slots, including dead
agents, and use its living Team A slots 0, 1 and 4. They never change its opponent,
clock, rules or initial state. The autonomous partner example uses the existing
evaluator and records declared partner exposure. Human-assisted play and unknown
training exposure do not establish zero-shot coordination.

A real local Qwen3.5-4B FP8 service answered four actor requests across the hard
matchup and fixed scenario, including one restart each. All four replies became
legal accepted actions in saved replays, with no retry or fallback. Replies used
12 or 13 tokens, within the 64-token limit. The complete check took 76.45 seconds,
including 32.90 seconds of server startup. Model files were hashed against the
cached pinned revision. The owned server exited successfully and its internal-GPU
allocation returned to baseline. Its log retains a semaphore-cleanup warning at
shutdown; this small run is not a model-server lifetime qualification.

## Lifecycle And Cost

A fixed CPU workload mixed Manual, custom JAX and opaque host controllers. For
each capture setting it used two warm-up launches and six measured launches,
with two episodes and two turns per episode. All 64 accepted turns agreed with
64 direct reference turns. Focus, drawing and saving caused no extra inference
or method loading. Queues, futures, file handles and owned resources returned to
their expected state after each launch. Compiled cache counts stayed stable as
weights, keys and episode memory changed. This bounded check cannot prove that
no lifetime leak is possible.

One recorded turn had built eight speculative error displays as well as its
normal display. Reusing equal display validation within each failure group
reduced the total from nine to three. Every raw response is still validated
before publication. No renderer or simulator rule changed.

| Warmed CPU Work | Measured Time |
| --- | ---: |
| Direct turn through the existing control owner | 23–25 ms |
| Client turn, Oracle, capture off | 41 ms |
| Client turn, SharedObs POV, capture off | 241 ms |
| Client turn, Oracle, capture on | 115 ms |
| Client turn, SharedObs POV, capture on | 693 ms |
| Save a two-transition replay | 1.09 seconds |
| Open its first Viewer | 508 ms |

Each turn row uses 12 paired samples. Later capture samples shared CPU capacity
with installed-package checks. The recorded POV turn previously took 1,884 ms
in the same workload. Remaining time mainly comes from existing authorization
and display checks. These are visible interactive costs, not a high-frame-rate
or optimal-rendering claim. Whole-process RSS ended near 1.30 GiB and grew about
2.5 MiB across the final six recorded launches, without forced garbage collection.
Each two-transition replay used about 247 kB. Disabled capture created no recorder
or replay output. CPU simulation numbers do not establish GPU throughput or VRAM.

The workload above used a small custom JAX method, so it hid a large cost: the
client ran every loaded `System` uncompiled. JAX then traced and compiled the
whole model again on every turn. With the recurrent 7b+pool checkpoint against
TDM-Beta a turn took 3 to 5 seconds on CPU and 5 to 7 seconds on an RTX 5090
(15 to 23 seconds for the first). Cache-size counts did not show it, because
uncompiled calls never enter a compiled program's cache. Two fixes followed. A
pair of JAX methods now runs through one compiled program, the same shared pair
function evaluation compiles; host methods keep the uncompiled route. Every
compiled call also commits its inputs to their device: before, the first turn's
uncommitted arrays and the later committed ones made each program compile twice.

| 7b+pool against TDM-Beta | CPU | RTX 5090 |
| --- | ---: | ---: |
| First turn, compiled once (now done before the page opens) | 4–7 s | 12.5 s |
| Every later turn | 0.02 s | 0.02 s |
| Same turn, old uncompiled route | 3–5 s | 5–7 s |

The compiled route chose bit-identical actions, memory and state over four turns
on the Vision Moment, a normal start, Policies Only, a fixed scenario and a host
plus JAX pair. `tests/test_battle_client_compiled_systems.py` counts real XLA
compiles with `jax.monitoring` for Systems the project has never seen: the first
turn compiles, later turns compile nothing, and the launch warm-up leaves the first
Lock In nothing to compile. With input committing turned off, all six
compile-count tests fail; the seventh checks only that preparing never calls a
host method.

Agent POV was several times slower than Oracle View in the BattleClient, the
DevClient and the Replay Viewer. No JAX compile was involved; the cost was host
work repeated per request. Three causes were removed. First, every one of the ten
SharedObs source rows re-validated the whole context/frame join (about 5 ms of
each 5.3 ms row); the join is now validated once per pair of frozen objects.
Second, four pydantic `TypeAdapter`s were rebuilt on every call (7 to 18 ms each,
many times per request); each is now built once per module. Third, the live
service built the same presentation at commit, again for the page's request and
again for the live snapshot; a build is now reused when every input is the same,
including the live worker status inside the Agent POV researcher panels. The
Replay Viewer also keeps each frame's source rows for the replay's lifetime and
shares them across viewers.

| Warm CPU work, HTTP, under shared load | Before | After | Oracle View after |
| --- | ---: | ---: | ---: |
| BattleClient Agent POV switch (command + page read) | 418 ms | 72 ms | 38 ms |
| BattleClient presentation read, unchanged state | 178 ms | 11 ms | 13 ms |
| BattleClient Lock In presentation read | 255 ms | 61 ms | — |
| Replay Viewer Agent POV step | 250 ms | 36–39 ms | 22 ms |
| Replay Viewer Agent POV agent switch | 256–288 ms | 37–38 ms | 20 ms |
| Replay Viewer timeline marks, first Agent POV agent | 8.96 s | 1.79 s | under 1 ms |
| Replay Viewer timeline marks, each later new agent | 9.0–9.8 s | 0.42–0.44 s | — |

The page draws a switched agent as soon as its presentation arrives; the timeline
marks load separately. The shared source rows hold about 59 MiB for a 246-frame
replay (2,460 rows), the price of the later-agent speed-up. Every BattleClient
and Replay Viewer response in a fixed command script (74 and 66 responses: picks of
all ten agents, Tab, view switches, a Lock In, steps, seeks and agent switches)
was identical before and after, apart from per-session random keys, so neither
view shows anything different. `tests/test_battle_client_presentation_reuse.py`
checks that one switch builds once, that every reused presentation equals a
fresh build, and that the join is validated once and a failing join always fails.

A separate browser regression holds the accepted turn's evidence callback open.
Save remains disabled until it finishes, then succeeds using the settled revision.
A failed method still permits inspection and saving of the last accepted game.
The private client repeats its save/view/restart and authoring replacement flows
through the same revised runtime, using temporary authoring stores.

## Runefield Look

The Runefield look ([A45](../design/specification_amendments.md#a45-runefield-client-presentation))
changes drawing only, so its checks guard the presentation contract, not the
simulator. Permanent browser unit tests check these rules:

| Rule | What The Tests Check |
| --- | --- |
| Agent POV leak | Over recorded Agent POV frames, changing the hidden champions' rows in the global researcher data changes no arena, overlay, camera, beat time or BattleClient panel record. |
| Agent POV boundary | Arena looks, the camera, the BattleClient panels and Frame Explanation do not import researcher helpers or read the global researcher data. |
| Drawing budget | The number of drawn nodes and animations stays under fixed caps for every fixture and sample frame, in both views. |
| Beat clock and glide | Tick and beat lengths, the Longer Tick setting and the step glide. |
| One command set | No key has another job elsewhere; B does nothing outside the Replay Viewer. |
| Switches and presets | Every switch label and help line uses the approved names, and presets never change the view. |
| Export | Each look's tests check export-safe SVG (flat fills, strokes, opacity and transforms only); the export tests check that PNG provenance version 3 records the camera frame. |

Python tests check the host additions: the `home_fall` flag, the per-audience
timeline marks, the viewing-team rule and the checkout page fallback. Two
independent reviewers also checked the Agent POV rules in the running
BattleClient and Replay Viewer with recorded games. These checks cover the
declared fixtures and sample replays. They do not prove that no hidden fact can
ever leak through a path the tests do not reach, and Agent POV is a fairness
aid, not a security boundary: browser developer tools can read the global
researcher data the Replay Viewer's panels use.

See the [BattleClient guide](../battle_client.md) and
[copyable tutorials](../battle_client_tutorial.md) for the supported workflow.
Raw qualification records, source hashes and failed diagnostic attempts are kept
in the local M12b evidence directory. The final commit gate separately checks the
complete public test inventory; these focused results do not substitute for it.
