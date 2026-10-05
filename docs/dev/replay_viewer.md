# Replay Viewer

Use Replay Viewer to inspect a recorded game without changing it. Open a local
replay, a checked sample, or a scripted demonstration. The Viewer draws recorded
frames, plays their recorded events, exports a battlefield image, and computes
measurements from the captured facts. It cannot stage actions, run another
simulator turn, reset the game, or record to a chosen destination.

A normal workflow is:

1. Open a replay with the launcher below and use the printed URL.
2. Pause at a frame, choose Oracle View or an agent's POV, and read the
   **Actions** tab: Latest Transition and Upcoming Transition.
3. Use the **Visuals** tab to choose which recorded effects to draw.
4. Export PNG for that view, or open the **Metrics** tab and download CSV
   for the current tick or the full captured game.
5. Use Exit Replay Viewer or stop its terminal process when finished.

The Viewer opens in Oracle View at 1x speed. The right-hand rail has six
tabs: **Roster**, **Agent**, **Metrics**, **Actions**, **Visuals** and
**Frame**. Four key buttons are stacked at the arena's top right, showing
only their key so the arena keeps its size on every screen: **V** Everyone
In View, **B** Battle Cam, **F** Frame Explanation and **Esc** Clear
Selection. Each does exactly what its key does, and V, B and F light while
they are on. Hover or focus one for its name and what it does. There is no
Visual Key: hover tips, the Champion Card and Frame Explanation explain each
mark where it shows. Hover or focus a tab, a top-bar button (**Agent POV**,
**Oracle View**, **Visual Filters**), a key button or the one-time Battle Cam
tip for one plain sentence on
what it holds or does; the same tips show in the BattleClient. The **Arena
Frames** under the arena list every champion, one thin row each; each team's
"Standing 4 of 5" count is under its name in the scoreboard (in Agent POV,
the viewing team's only). The **Agent** tab shows the selected champion's
Champion Frame (life, Ultimate, Effective Speed, statuses, Recent Ticks and
Next Target) and its Class Guide; under it, the closed **Full Class Details**
opens to the class's order, overview, role, strength and weakness, every
class mechanic and the passive's rules. Press **?** in the page for Help and
the full key list. The look and its rules are recorded in
[A45](../design/specification_amendments.md#a45-runefield-client-presentation).

The top bar shows the brand (the carved **MARL-BGs**, the same spelling as
every client), **Replay Viewer**, the place line ("**Teutoburg Forest** · Map
47 · Team Deathmatch"), the view switch and the status pills and buttons. When
the bar runs short, the place line gives way first, "Team Deathmatch" before
the map name, because the scoreboard above the arena names the map too; only
then is "Replay Viewer" cut. At 1440 px wide the map name shows whole, except
in Agent POV on a frame with an end-of-game pill (such as Time Limit),
where the agent picker and the pill take its room. At 1366 px it shows whole
in Oracle View before the end. The scoreboard always shows it.

Use [BattleClient](../battle_client.md) for live play. New recordings use Replay V4, which records the Team Deathmatch Red
Zone depth in its configuration and in context column 19. Historical V1, V2 and
V3 files retain their original meanings (V3 and older scored one point per
death). Current policy rows store ally/enemy flags and a local self
index; the Viewer gets Team A/Team B labels from the recorded roster. Changing
the view or frame never changes observations, policy assignments, or saved
bytes. Current NoSharedObs actor exports use POV V3 (from Replay V4);
historical POV V1 and V2 are still readable.

## Open An Installed Replay

Follow the [base wheel installation commands](../battle_client.md#install-and-open),
then open a saved replay from any directory:

```bash
python -m marl_battlegrounds replay episode.marlbg-replay.json
python -m marl_battlegrounds replay "runs/my game.marlbg-replay.json" \
  --frame-index 12 --view pov --pov-slot 5 --no-open
```

The installed package includes the existing browser HTML, JavaScript, CSS and
fonts. It needs no source checkout, Node.js, frontend build or asset download.
The server stays on loopback and prints its actual URL. The replay is checked
before serving or opening the browser. Then the launcher prints "getting Agent
POV ready" and builds every agent's Agent POV view first, so entering Agent POV
and switching agents are quick from the first click. That takes a few seconds
for a full game (about 7 s for a 246-tick five-against-five game on CPU) and
well under a second for a short scenario. `--no-open` leaves browser opening to you.
The viewer opens your computer's default browser. As for the
[BattleClient](../battle_client.md), we recommend an up-to-date desktop
Firefox; Chrome and Edge work too (the full automated browser tests run on
Chromium), and Safari should work but is not tested.
The process reads the replay; it does not change that file or run simulator turns.

For a static frame, install the optional plotting extra and choose a frame:

```bash
python -m pip install "./marl_battlegrounds-0.0.0-py3-none-any.whl[viz]"
python -m marl_battlegrounds replay episode.marlbg-replay.json \
  --static --frame-index 0
```

Static display uses Matplotlib. Browser viewing does not import Matplotlib.
Requested metrics use the same numerical analysis as the repository Viewer;
viewing is not a promise of zero numerical-backend work. Help and stored-result
reading remain independent of that analysis.

The package command accepts one replay path. Sample catalogs and scripted
scenario creation remain repository tools, described below. Both launch routes
use the same replay service, authorization, playback and browser assets.

## Select Exactly One Repository Input

Run commands from the checkout with `uv` available. Each invocation must choose
exactly one replay, sample, scripted scenario, or list operation:

| Selector | Purpose |
| --- | --- |
| `--replay PATH` | Validate and open a local canonical-format replay bundle. |
| `--sample-replay NAME` | Verify and open one checked-in sample by stable launch name. |
| `--scenario NAME` | Run one registered demonstration in a separate process, then open its validated replay. |
| `--list-scenarios` | List default scripted demonstrations and exit. |
| `--list-sample-replays` | List checked-in sample names and descriptions and exit. |

Open a local artifact:

```bash
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json
```

Choose the initial frame and audience without opening the browser automatically:

```bash
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json \
  --frame-index 12 --view pov --pov-slot 5 --no-open
```

List and open a checked sample:

```bash
./scripts/dev/run_replay_viewer.sh --list-sample-replays
./scripts/dev/run_replay_viewer.sh \
  --sample-replay death-respawn-shield
```

List and record a scripted demonstration for viewing:

```bash
./scripts/dev/run_replay_viewer.sh --list-scenarios
./scripts/dev/run_replay_viewer.sh --scenario stacked_team_auras
```

The public browser options are `--frame-index`, `--pov-slot`, `--view
oracle|pov`, `--ranges`/`--no-ranges`, `--port`, and `--no-open`. Ranges start
hidden; `--ranges` opts in without changing the default visual switches. `--seed`
applies only when recording a scripted scenario; its default is `0`. Browser
frame index defaults to `0`, view to `oracle`, and port to `0` (a free port
chosen by the operating system). `--pov-slot` selects an active global slot for
Agent POV. By default, it uses the lowest recorded focal-policy slot. If the
replay has no focal slot, supply an active `--pov-slot` to open Agent POV. List
operations reject unrelated options. `--static` has its own narrow matrix
described below. Write complete option names; abbreviations are rejected.

## Open A Saved BattleClient Game

In [BattleClient](../battle_client.md), use **Finish And Save** (a finished game
saves by itself: wait until the Recording pill says **Saved**), then **Open Replay**.
The Viewer opens in a separate tab, in Oracle View. After a full game the new tab first shows
a dark "Opening Replay" page for a few seconds while every agent's Agent POV view is built, so
switching is quick from the start; the game keeps answering meanwhile. The live page stays available for inspection
and Restart. Each product has its own local URL and access token. Closing a tab
leaves its server running; the Viewer's Exit stops only that Viewer. BattleClient
can reopen it. Exiting BattleClient leaves the Viewer it opened running: the
terminal prints the Viewer's address, and the BattleClient call returns once you
press the Viewer's Exit or Ctrl-C. Ctrl-C while the game is still open stops both.

The Viewer reads the saved recording. It never adds turns to that file. Restart
in BattleClient starts a fresh recording with a new file name and episode memory.

## Scripted-Scenario Isolation

`--list-scenarios` shows researcher scripted demonstrations and omits the manual
`arena_5v5` laboratory. `stacked_team_auras` follows `aura_crossfire` in the
default catalog and demonstrates simultaneous Basics under two same-team Mage
and two same-team Warrior emitters per team.

Developer visual-stress demonstrations are excluded by default:

```bash
./scripts/dev/run_replay_viewer.sh --list-scenarios --include-stress
./scripts/dev/run_replay_viewer.sh \
  --scenario max_status_stack --include-stress
```

Use `--include-stress` only with `--list-scenarios` or `--scenario`. It lets you
discover or run developer stress demonstrations. It is not a browser view option
and cannot be used with local replays or checked samples.

The launcher records the scripted demonstration in a temporary child process
with `JAX_PLATFORMS=cpu`. The child executes the registered commands and
publishes one V4 replay containing the captured frames, facts and episode
provenance. No metric sidecar or full metric computation is required. The parent
opens those bytes through the public loader before starting the Replay Viewer.
The read-only viewer process does not import or run simulator control. Only the
separate demonstration process performs those scripted turns.

### Authoritative-Battlefield Visual Coverage Rule

Every battlefield visual mechanic needs executable checks in both a default
researcher demonstration and an opt-in stress demonstration. A label or
description is not enough. Tests must find the actual fact in validated
transition events or Scene V2 rows. Status mechanics must additionally prove
their stable catalog token through natural expiry.

Any change that adds a battlefield effect, durable badge, status, aura,
ability-family presentation, or lifecycle presentation must extend the regular
and stress trajectories—and the paired semantic coverage contract—in the same
change. Prefer extending an existing coherent trajectory; add a new scenario
only when doing so keeps the demonstration readable. Research-space panels and
controls outside the authoritative battlefield snapshot are not governed by this
scenario-pair rule.

## Checked Sample Replays

The checked V1 bundle under `examples/replays/v1/` contains exactly three
replay/metric pairs plus `manifest.json`:

| Launch name | Source scenario | Coverage focus |
| --- | --- | --- |
| `death-respawn-shield` | `death_respawn_cycle` | Lethal damage through the first post-shield interaction. |
| `recovery-status-lifecycle` | `recovery_refresh_cycle` | Recovery, rejection, refresh, break, reapply, and expiry. |
| `mirrored-five-class-ultimates` | `mirrored_ultimates` | Reciprocal demonstrations of every class Ultimate family. |

Every sample uses an 18×12 map. The manifest records stable names, source
scenarios, transition/frame counts, event-kind coverage, byte lengths, hashes,
and actual source/runtime provenance. These are deterministic, unofficial
display demonstrations. They do not qualify benchmark performance, evaluate a
policy, or certify the current source tree or host machine.

Verify the complete seven-file set through the public loaders:

```bash
JAX_PLATFORMS=cpu uv run python \
  scripts/dev/generate_visual_debugger_sample_replays.py \
  --check --output-directory examples/replays/v1
```

New generation writes manifest schema 3 and three self-contained Replay V4
files. Manifest schema 2 sets (Replay V3) still verify. It
does not write metric sidecars or run metric reducers. The Viewer computes
metrics later from the recorded facts. The checked V1 pairs stay unchanged and
keep their original strict verification. Both versions bound file reads, hold
one directory snapshot, verify hashes and source facts, and validate events and
captured facts. Current source provenance adds the explicit CodeRevision V2
marker; all recorded source facts must still match.

```bash
JAX_PLATFORMS=cpu uv run python \
  scripts/dev/generate_visual_debugger_sample_replays.py --generate
JAX_PLATFORMS=cpu uv run python \
  scripts/dev/generate_visual_debugger_sample_replays.py \
  --check --output-directory artifacts/visual-debugger-samples/v4
```

Generation defaults to `artifacts/visual-debugger-samples/v4`. Verification and
sample launch defaults still use `examples/replays/v1`. Current generation
writes four files instead of seven. Each replay stores ten local self indices
per frame and keeps the existing 58-column unit rows. Its semantic verification
is one pass over the recorded transitions, as for the historical samples. This
storage change and the skipped metric work do not establish a runtime speedup.

Generation is a maintainer task and refuses to overwrite a destination. Use a
new, absent directory only after scenario source, tests, and public
documentation are frozen; verify the complete set before an explicit reviewed
publication. Never hand-edit one member or delete the checked directory merely
to bypass the no-overwrite boundary.

## Transport and Exact-Frame Summaries

The **Rune Scrubber** dock under the arena holds the transport. Its Event
Track shows each team's falls (Team A above the groove, Team B below it),
Ultimate runes on the groove in the caster's class color (filled for a
damaging Ultimate, hollow for Burst or Holy Word: Salvation) and return pips.
Hover a mark for who fell, cast or returned; click or drag the track to seek.
Its one button row holds, in order: Start, −10, −1, Play/Pause, +1, +10, End,
the readout ("Tick 118 of 204" with the beat under it: Settled, Playing or
Paused · Strike), Speed, an Export menu (Export PNG, Download Metrics CSV and
Episode Details) and the camera switch. In Agent POV the marks come only from
what the viewing team saw, so an event it never saw is never marked.

A Replay Viewer notice (for example "Exported …" or a failed request) shows
under the readout for about 4 seconds, whole, then the beat label comes back;
there is no notice in the top bar.

**Stepping one tick.** +1 (and the Right arrow, N, or a slider move of one
tick) plays that tick exactly as Play does: the champions move, the blows
land, the numbers pop and a champion that falls falls after its move. Then the
replay pauses on the tick's still, the same picture a seek to it shows. A
further +1 while the tick plays skips it to its end and plays the next one.
−1 (and the Left arrow) rewinds the tick you are leaving: a champion that fell
in it stands up where it fell, and one that came back to its pad goes back to
its corpse; then every champion glides back to where the earlier tick ended
(240 ms), and the earlier tick's strokes, landings, numbers and marks fade in
once they have landed. In Agent POV the dark veil fades back with them. During
a played tick or a rewind nothing shows ahead of a champion: the selected
champion's moonlight, brackets and reach rings move with it, its Focus Trail's
dots appear as it passes them, and its next action line, All Staged Orders and
Next Step Ghost appear once it has landed.
Every speed (0.25× to 2×) scales all of this together. With the **Step
Glide** switch off, +1 and −1 show the paused tick at once.

Playback sends at most one replay request at a time, with one presentation in
flight. A press made during a request (a button, a key or the track) is not
lost: it waits as the one pending target. Relative presses add up, so five
quick +1 presses become one move of five ticks; Start, End and the track
replace the target. The buttons never grey out while a request runs. The
native Tick slider and readout stay in the page for keyboard and screen-reader
users. V, B, F, G and N also work while a transport button has focus; Space
and Enter still press the focused button. In a text box, such as Find A
Measurement, every key types as usual: N types an "n" and moves nothing.

Buttons, committed seeks, view/range changes, and reconnects show a
static summary of the exact selected frame. It shows enabled durable facts and
the events that produced the frame together, with no animation in progress.
Pressing Play restarts the displayed incoming transition at logical time zero
when one exists, waits for its scaled presentation to settle, and then requests
the next frame. Each next-frame response animates only the recorded transition
that produced that successor frame. A tick plays as a sequence of beats; see
[How A Tick Is Drawn](../battle_client.md#how-a-tick-is-drawn).

**Latest Transition** is the recorded incoming transition `T_(n-1)` that
produced the displayed frame `s_n`; it is absent at frame zero. **Upcoming
Transition** is the recorded `T_n` out of `s_n` in the same Submitted / Accepted
row grammar. Both panels remain global researcher evidence in Oracle View and
Agent POV: they show every configured-active actor in canonical roster order and
never stage, predict, or execute an action. In Agent POV they carry a **Full
Record** chip. Agent POV fog of war applies only to the arena and everything
drawn over it. The Arena Frames likewise remain a global researcher control.
Each team lists its champions in ID order in both views, so a frame never moves
when sight changes, and any agent's POV can be selected at the same replay
tick. Upcoming is absent only when the displayed frame has no recorded
successor.

The eight supported rates are exactly **0.25×, 0.50×, 0.75×, 1.00×, 1.25×,
1.50×, 1.75×, and 2.00×**. A rate scales the complete presentation clock,
including animation phases, waits, and the replay terminal hold. It never
changes simulator ticks or artifact contents. A fresh Viewer starts at **1.00×**
(it was 0.50× before the beat clock; a quiet tick at 1.00× now lasts 1.7
seconds). Your choice stays in place through pause,
seeking and loading another replay in the same page.

### Keys and Cameras

One command table owns every key of the BattleClient, the DevClient and the
Replay Viewer. A key does the same job in every client, or nothing. Press **?**
for Help: its key list is written from that table, so it always matches what
the keys do. This guide does not copy the list.

Some rules stay fixed. With Control, Alt or Meta held, the Viewer issues no
replay command, so browser shortcuts keep working. Keys typed into a button,
input, select, text area, link, disclosure summary, dialog, editable region or
ARIA widget keep their normal behavior; V, B and F also work from a focused
button but never from a text field or dialog. A held Space does not repeat
play/pause. Without installed, connected, visible replay authority, the keys
do nothing. While a frame is still loading, Space and the arrow keys are
consumed without a new command, so the page does not scroll.

The camera choices are **Whole Map** (the default), **Everyone In View** (V)
and **Battle Cam** (B, Replay Viewer only). V and B, or their key buttons,
turn their camera on and off: pressing the lit one again returns to Whole
Map, and no lit key means Whole Map. There are no camera buttons in the Rune
Scrubber row and no small map picture. Everyone In View zooms in as far as it
can while every champion stays on screen. Battle Cam frames the fights and,
while the replay plays, leans in on each fall and slows it down; its parts
are Visuals switches. The Viewer remembers the
last camera in this browser. A camera change shows a short tip, and the first
visit shows one Battle Cam tip; both sit in the scoreboard's empty space beside
the tick, so they never cover the arena or a panel. In Agent POV the camera only frames champions and
fights the viewing team saw.

## Audiences and Recorded Authority

Oracle View exposes the full authorized battlefield presentation. Agent POV
applies the selected recipient's fog of war to that battlefield: NoSharedObs
shows the recipient's local view, while SharedObs shows the same-tick visual
union authorized by its team observations. Activating a visible agent in Replay
Agent POV switches the recipient at the current tick; it never advances or
mutates the artifact. Artifact identity, completion/processing evidence, PNG
export, and metric download remain capabilities of the researcher tool in every
visual POV.

In Agent POV nothing on or over the arena shows a champion the viewing team
cannot see, or anything tied to it: position, life, statuses, an unseen
attacker's class, or the timing of its events. Hover tips, Frame Explanation
(F), the camera and the Rune Scrubber marks follow the same rule. The docked
Agent, Metrics, Actions and Frame tabs are researcher panels: they show the
full recorded game under a **Full Record** chip. The **Sight Reach Veil**
darkens every blind spot of the viewing team: the floor beyond the sight reach
of the frame's authorized living sensors and the shadow behind every stone.
The server builds the shape from the recorded positions with the game's own
line-of-sight rule, the same shape the BattleClient draws; while a tick plays
it keeps the start-of-tick shape until the glide and changes at each Fall and
Return beat. Enemies just back on their pads stay hidden even in the light.
Where a
champion left sight this tick, a dotted **Last Seen** ring with a hollow
outline of its class emblem marks where it started the tick. A returning
champion's waystone stays plain until its return beat; its token, ward and
status marks arrive with it, and its old corpse stays where it fell until that
beat, then fades out over 0.2 seconds while a thin ash ring lifts off it (only
during play; a paused or stepped frame shows the settled end). The home end marks, the Grave Ground and these
marks are the same looks the BattleClient draws (see
[the BattleClient guide](../battle_client.md)). Switching Oracle View to Agent POV and back gives exactly a
fresh render.

Hover or focus a champion's body in the arena, or its frame in the docked
Arena Frames, for its **Champion Card**: the same card the BattleClient and the
DevClient show (see [the BattleClient guide](../battle_client.md)). It has the
name and calling, crest and Agent ID, thin Life and Ultimate bars, one row per
status (its icon, approved name, time left, what it does with its exact number
and where it comes from by class) and who plays it, here the name of that
champion's controller from the recording (a long name is cut with "…" at the
card's edge). Each team's line in the score strip follows the BattleClient's
controller-name rule: the names when they fit ("Manual + Recurrent MAPPO"),
cut with "…" past a fixed width, a short count such as "3 Policies" (or
"Manual + 3 Policies" when a person played one champion) when even the cut
names do not fit, and hovering it drops down **Team A Controllers** (or
Team B): each
controller's name in full, how many champions it plays and each of those
champions. Each champion's tip in the Arena Frames names its controller from
the same list. These are the names the
live client showed, one per champion, exactly as recorded: "tdm-beta" stays
"tdm-beta", and a champion a person played reads **Manual** (only the live
client calls it **You**). Like **You** in live play, **Manual** comes first
in the line and the list, and the short count never counts it as a policy.
A team made of several Systems (`marl_bgs.team`)
names each member's champions. The card is the same in Oracle View and Agent POV; in Agent POV
a champion the viewing team cannot see gets the reduced card ("Unseen · Team A
cannot see it now."), even though its docked frame keeps the Full Record. The
frames' status chips and their hover text use the same status names and the
same second line as the card's rows, and the chips follow the card's order:
stuns, then slows, then the rest. A fallen champion's card shows life 0 and
"Fallen · Returns In 3" (its team's respawn wave countdown + 1), the same line
on its arena body and its docked frame. While the pointer rests, the card is
drawn again from every new frame, so a tick step or a new POV agent never
leaves an earlier frame's card on screen.

Hover or focus a stone for its tip, in the same plain words as the
BattleClient: **Pillar**, **Wall** or **Thin Wall**, then **Obstacle ID** (a
plain number, the same in Oracle View and Agent POV), its size in world units
(Radius, Length, or Width and Height), **Turned** for walls and **Center**,
and the rule "Blocks movement and sight. A Warrior's Charge can pass over it.
Bodies never block sight." A thin wall adds "True width 0.01 units. Drawn
thicker so you can see it." No tip shows an internal ID. A click on a stone
pins its tip; a second click, a click anywhere else or **Esc** closes it.

The other arena tips use the same plain words. A Basic's tip says who used
it, for example "The Priest used Lesser Heal this tick." (only the ability
when the viewing team did not see who used it), and an Ultimate's tip says
what the Ultimate does; both then name the **Source** and **Recipient**. A life number's tip is **Net Damage**, **Net Healing** or **No
Net Change**: the champion's whole life change that tick, every blow and heal
added together, with its **Life Change** ("-9.81"); it does not say who caused
it. A planned order's line is **Planned Basic** or **Planned Ultimate**: who
will act on whom, not a path anyone walks. A fall's tip in **Fallen Feed**
("Fallen On Tick 92") says that credit is shared by every champion that hit
the fallen one on that tick and by a Priest whose healing on that tick helped
one of them, then "This champion fell.", the **Killing Team** and the **Kill
Contributors**.

Without Shift, the card always sits fully inside the arena box, also for a
docked frame below it, and it is always one column wide, so it never spreads
across the map. It shows at most four status rows; when the arena box is too
short for the card with all its rows (about 270 pixels in a 1280 x 720 window),
it shows as many rows as fit, at least one, in the same order. Then comes
"+3 more" with "Hold Shift For All". The name, the bars and who plays it always
show. The fit is worked out again each time the card is drawn: every frame,
every view change and every window resize. Rest the pointer and hold **Shift**
by itself for about a third of a second to see every status (with more than
four, "All 7 statuses" and "Release Shift To Shrink"); when that is taller than
the arena box, the rows spread into two columns, or more only when two do not
fit (a column that would not make the card shorter is not added). Inside the
arena box it may be drawn down to 85% of its size. When it would need to be
smaller than that (for example seven statuses in a 1280 x 720 window), it grows
past the arena box instead: normal size, the number of columns that makes it
shortest, over the panels next to its champion, fully on screen and off the
champion when there is room. Only a window too small for even that draws it
smaller. Releasing Shift, any other key, any click and any pointer movement
shrink it at once back to the one-column card inside the arena box, so
**Shift + Arrow Left / Right** still moves ten ticks and **Shift + Tab** still
steps the Agent POV agent back. Shift by itself sends nothing.

Every status line in the docked panels (the Champion Frame, the Agent tab's
**Statuses**, the status chip tips and the Spawn Shield chip's tip) and in
Frame Explanation (F) uses the card's words and names the source by class only ("Charge (Slow) · 5 ticks
left · It moves at ×0.5 speed. From a Warrior."), so Oracle View and Agent POV
read the same. The Spawn Shield ring on an arena body has a small tip in the
same words: "Spawn Shield (Invulnerable)", "It can't be seen or hit, and can't
strike yet." and **Time Left**. Each panel that shows a champion's life (the Champion Frame,
the Agent tab and the Arena Frames) uses the BattleClient's one **Life**
bar: green above half life, yellow at half or less, red at 30% or less. Each
Arena Frames row shows the bar with its number (for example 114 / 200).

Blows are drawn between the end spots of their source and target, so a stroke
can look longer than the ability's real reach; a faint imprint ring and
hairline keep the true spot. A Warrior's Charge is drawn from its start to its
landing. See
[A45](../design/specification_amendments.md#a45-runefield-client-presentation).

Under [amendment
A25](../design/specification_amendments.md#a25-sharedobs-only-canonical-benchmark-execution),
support for both views keeps old and custom evidence readable. It does not make
both information modes eligible for official benchmark claims. NoSharedObs
material is historical, diagnostic, or custom evidence and is noncanonical for
new official claims. Drawing SharedObs data does not give a run official status.
Official evidence must separately prove the canonical actor projection and the
exact configured-active, same-team, off-diagonal availability matrix on every
recorded frame. The Replay Viewer continues to render both regimes and does not
mutate, migrate, or relabel their artifacts.

The SharedObs visual union follows [specification amendment
A17](../design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation).
It combines only recorded rows from the same decision tick and permitted
same-team sensors. It does not recompute geometry, visibility, line of sight,
action masks, mechanics, or simulator state. It does not add teammate masks or
history, rewards, policy/critic state, transition facts, or hidden Oracle facts.
This display union is not a ready-made SharedObs learning tensor.

A separate server-authorized corpse overlay keeps locally visible dead bodies
consistent with Oracle View. Python admits a corpse only from the same decision
tick and only when an authorized living sensor has it within recorded
observation radius and static line of sight. The overlay is used for corpse
painting and inspection and may additionally admit only a death/respawn
presentation cue and that cue's owned endpoint. It never admits another event,
moves an ability route, or changes policy input, masks, targeting, actions,
simulator or recorded transition semantics, or the replay artifact. Every allowed
corpse is drawn below every living body, so a corpse cannot cover a living actor
or take its mouse target when the bodies overlap.

### Technical Frame

Technical Frame shows the recorded Episode, Task Mode, technical Map name,
Observation Mode, Episode Limit, and root/episode-stream Seeds in both live and
replay views. These are shared researcher metadata; they do not enter policy
observations. Unknown historical seeds remain explicitly unknown.

The task header also shows the friendly map name and its recorded split, for
example `Three Body Problem (Test Map)`. Approved identity requires a declared
map ID or unchanged authored source plus matching current or retained historical
packaged geometry. A custom layout stays `Custom Map`; historical records retain
their recorded name, or show `Map name unavailable` when no name was recorded. Historical free-form
names never claim an approved map ID or split. Geometry alone never assigns an
old replay to a split.

The [approved 2026-09-16 map update](../design/specification_amendments.md#approved-map-publication-on-2026-09-16)
changes the map selection for new games. Publication and focused replay
compatibility checks passed. It keeps earlier packaged identities in one
immutable history resource: opening an old replay shows that game's
recorded layout and version, not the newest map with the same ID. Old replay
files and their recorded results are not rewritten. A36 lists the approved
clearance limits and exact map exceptions.

After installing the map update, restart BattleClient, Replay Viewer and any
long-running Python process to refresh the cached map catalog. Opening an old
replay after restart still shows the old game. Earlier replay and speed results
remain evidence for the map versions they actually used.

Replay Oracle retains its five existing technical fields:

1. Artifact digest prefix
2. Frame
3. Simulator step
4. Incoming transition, omitted at frame zero
5. Ordinary movement distance scale

Alongside shared episode metadata, Agent POV receives Frame, Simulator step, and
its conditional authorized Incoming transition. It never receives the canonical
artifact digest or movement scale through this panel. Completion describes how
the captured game ended. Processing describes whether host output preparation
succeeded. They remain separate facts.

Replay files generated by `RunWriter` use
`<technical-map>__episode-<id>__seed-<root>__stream-<episode-seed>__a-<policy>__b-<policy>__<full-sha256>.marlbg-replay.json`.
Policy display labels are sanitized and shortened to keep the ASCII basename at
most 255 bytes; the technical map name and full replay digest are retained.
Unknown seeds are written as `unknown`, and unnamed layouts use a digest-based
custom/recorded-map token. Explicit researcher-selected save paths and stored
resume references are unchanged.

## Visual Filters

The **Visuals** tab lists every switch of the Runefield look, in groups
(Bodies, Combat, Life And Death, Arena, Moments, Vision, Motion, HUD, Camera,
Panels and others). Each switch has a plain label and a help line that says
what it draws and what Off shows. One file, `visual-filters.js`, owns every
switch, its label, help and default; the tab, the renderer and the PNG export
all read it, so this guide does not copy the list. Ranges is a separate
control and starts off.

The shipped defaults turn on **High Contrast**, **Class Seals**, **Moon
Motes**, **Death Dimming**, **Spawn Shield Expiry**, **Saved Beat**,
**Champion Streaks**, **Momentum**, **Kill Cam** and **Kill Slow Motion**, and
turn off **Moonlight**, **Floor Grid** and **Time Of Night**. **High
Contrast** is one switch in all three clients (brighter rings, clearer stones
and crisper text; an exported PNG shows the same colors).

Four presets set many switches at once: **Default Configuration** (every
switch as it ships), **Researcher** (exact numbers and every diagnostic),
**Manuscript** (flat paper
figures that still read in grayscale) and **Minimal** (bodies, blows, life and
falls only). A preset never changes the view, Ranges, layout or motion.

A switch only changes drawing. It never shows a fact the current view may not
see. A switch turned off falls back to a plain, truthful drawing (for example a
flat stone with its true footprint, or a plain token with its team ring and
life); it never hides a fact of the game.

**Class Seals** (Bodies, on by default) puts each champion on its class seal
instead of the plain class glow: a starglass octagon (Mage), a riveted bronze
boss (Warrior), a greenwood four-leaf shape (Hunter), a steel diamond with
curved sides (Rogue) or a gold sun (Priest, whose emblem then shows only its
cross, because the seal is its sun; `icons.js` createSvgIcon's onClassSeal
option). The seal is faint while the
Ultimate recharges and lights up in its own material when the Ultimate is
ready. During play it goes dark at the blow when the Ultimate is used, and it
lights up at the end of the tick when the Ultimate becomes ready (the Mage's
facets flash, the Warrior's rivets light pair by pair from the bottom). It
fades back at the end of the tick while a status mark is on the champion, and
a fallen champion keeps a faint ash outline of it. A paused frame shows only
the end look. Seals need a body at least 18 pixels in radius (champions are
always heraldic tokens; that look is not a switch); with **Cooldown Effects** off every seal stays faint. In Agent POV,
seals show only on champions the viewing team sees, and a champion whose
Ultimate timer that team cannot see keeps a faint seal. The switch is recorded
in the PNG provenance with the other battlefield switches.

Red Zone Floors tints each team's Red Zone a subtle deep red. The Red Zone is a
full-height strip on the team's own spawn side. When an agent dies inside its
own team's Red Zone, the enemy team gets 2 points instead of 1
([A44](../design/specification_amendments.md#a44-team-deathmatch-red-zone-scoring)).
The tint sits on the floor, under obstacles, bodies and effects, and shows with
or without grid lines. The strips come from the recorded depth and spawn sides;
the browser does not work out the rule itself. Replays recorded before the rule,
or with depth 0, show no tint. Turning the filter off only hides the tint.

**Fallen Feed** lists the falls of the last 8 ticks at the top of the rail, and
**Match Moments** herald cards name the match's big moments (see Herald cards
below). The earlier ticks' rows come from the frame's `recent_falls` field,
which the server works out per view from the recorded frames
(`viewer/champion_streaks.py`, the one owner of kill credit), so a seek, a
reload or a view switch shows the same rows as playing up to the tick; the
shown tick's own row comes from the beat plan and appears at its Fall beat.
An old recording without lethal damage contribution events names no killers
on the earlier rows. Hover
or focus a fall to see its complete **Kill Contributors** list. Useful same-tick
Priest support shares credit; healing that is all excess does not. In Agent POV
both show only falls the viewing team saw, naming only attackers it saw. Team
A's kills remain blue and Team B's remain red. Their cards, respawn banners and scoreboard entries follow
their recorded spawn sides. Missing or non-left-right spawn banks use the older
Team A left, Team B right layout. Contributor details use the existing
accessible event tooltip; historical missing attribution is explicitly
unavailable.

**Kill Credit Glint** (Life And Death, off by default) draws the shared credit
on the arena. At a fall's Fall beat every living champion that took part in
the kill gets the same thin ring in its class color, 1.06 times the body
radius and 1.25 pixels wide, under the bodies. It fades from 70% to nothing
over 240 ms while it grows from 95% to full size (reduced motion: the same
fade in place over 200 ms). At most four rings show per fall, it needs
**Death Effects**, and a paused frame or a PNG export never shows one. Oracle
View rings every recorded contributor. Agent POV rings a champion only when
the viewing team sees it standing at the end of the tick and saw its blow
land on the fallen champion; a contributor the team cannot see gets no ring.
The look is `src/looks/requiems.js`; it reads the plan's own Fallen feed row
for the killers and the plan's end-of-tick bodies for the spots, never the
match summary.

**Champion Streaks** (Moments, on by default) lets a herald card name a
champion each time it has taken part in 5, 10, 15 or 20 takedowns without
falling: its living medallion, the eyebrow "Team A Mage" and the title "10
Takedowns Without Falling" (no dots). Up to two other moments
of the tick join the eyebrow, such as "· Double Takedown" or "· ◆ Charge
Takedown". Off, the cards are the plain ones. The look
(`src/looks/heralds.js`) draws only the streak facts a herald row carries and
never counts history itself, because the count spans the whole match and, in
Agent POV, must use only the takedowns the viewing team saw. The server counts
(`viewer/champion_streaks.py`), once per replay and view, and sends the
champions that reach a multiple of 5 on each frame as the frame's
`champion_streaks`.
Oracle View counts every recorded fall with the champions that dealt damage on
it (the killers the Oracle Fallen feed names). Agent POV counts only the
recipient's own visual events: a fall the team saw, struck by a champion it saw
at the start of the tick; only a fall it saw resets a count. A seek, a view
switch or a page reload shows the same card, and the plan names a streak only
for a killer of one of the tick's named falls.

**Herald cards** name every moment of the store's Herald Cards option that
can happen in a game, with the store's words, medallions and headline order:
Penta Takedown (all five champions of one team on one tick; it heads the card
with "Team B Wiped" in the eyebrow), Team Wiped, First Blood (only in a game
that began at tick 0 with 0-0), Match Point (a score that rose this tick to
exactly 1 point short of First To), Double, Triple or Quadruple Takedown, runs
("5 Unanswered", then 10, 15 and 20; no dots), Champion Streaks, Rescue,
Ultimate Takedown, Trade, Scores Level (level after a lead of 3 or more),
Ultimate Combo (two or more damaging Ultimates on one champion) and 30 Ticks
Left. There is no countdown chip, no 10 or 5-to-1 Ticks Left card and no
"Match Point · Needs 1" chip. The first
heads the card and up to two more join its eyebrow; a plain single takedown
gets no card. `heraldMoments` in `src/looks/heralds.js` is the one moment
finder: it reads only the plan's `herald_card` row. The plan
(`heraldTickFacts` in `src/choreography-plan.js`) puts on that row the public
scoreboard (scores before and after the tick, First To, the tick and the draw
tick), the view's own Ultimate combos and wiped teams, and the frame's own
`match_moments` field. The server owns that field (`viewer/match_moments.py`),
counted per view once per replay and kept for the service's life, because it
needs history or blows a frame does not carry: whether the game began fresh,
the largest earlier lead, each team's unanswered run and each tick's rescues.
Oracle View reads every recorded fact; Agent POV counts runs from the falls the
team saw and names a rescue only when the team saw the rescued champion at
both ends of the tick, the Priest's heal and every blow on it. A card enters at
the Fall beat (a Rescue, Ticks Left or a combo whose target lived waits for the
tick to settle); Penta Takedown, First Blood, Team Wiped and Match Point draw the Herald Rule,
a 1 px line across the top band behind the card. With **Death Announcer** off
the moments born from a fall are left out, as in the store. Each card names its
moments in `data-herald-moments`. The live BattleClient and DevClient use the
same look and the same server facts.

**Herald cards** (`src/looks/heralds.js`) are always the same plate, with its
swallowtails, rule, glyphs, eyebrow and title: there is no one-line band or
chip form. A plate hugs the top edge of the arena and slides along it to the
spot that touches the fewest champions the view draws, then covers the least
of them, then sits nearest the middle. It never covers the V, B, F and Esc
keys at the arena's top right while a clear spot exists (that matters when
the camera zooms in). Only when every full-size spot touches
a champion does it shrink, never under 9 px words. When no spot fits at all
it still draws at its smallest readable size, so a card never silently
vanishes, and a card held through the next tick keeps the spot it was born
with.

**One card at a time.** While the replay plays (and in live play), one card
shows at a time and each stays at least 2 seconds at 1x (1 second at 2x).
A card that comes sooner waits its turn, then replaces the one on screen; the
old card fades out before the new one fades in, so there is no overlap and no
empty gap between back-to-back cards. At most one card waits: when another
arrives, the bigger moment keeps the waiting spot. A paused, stepped or sought
tick shows its own card at once. The painter (`src/choreography-painter.js`,
`carryFromEarlierTicks`) owns this queue; a view or recipient change starts it
fresh, so Agent POV and Oracle View never share a queue. A plate draws at
most seven medallions (the eyebrow keeps the real counts), so even a ten-fall
tick stays inside the look's node budget (48 nodes: seven fallen Mages, whose
sigil has one more path, with the Herald Rule). The plan charges a herald row
only its own card's allowance (`rowBudget`: a run's crest and dots need 12
nodes), so a busy tick, such as a return wave of eight champions, keeps its
card instead of dropping it for the largest card's allowance. Scores Level
shows both teams' crests and Ultimate Combo one rune diamond per caster, on
the same plate.

The **Verdict** card on the completion frame is the store's full card: the
basis words, both teams' controller names, scores and results, the lead line
of a horizon draw, **Honors** and the **Match Summary** and **Show Arena**
buttons. Its left column is the team whose home end (recorded spawn side) is
on the left. Honors come from the frame's own `match_honors` field: the server
adds up every transition's totals once per replay (`viewer/match_summary.py`:
Kill Contributions from the shared kill credit, Damage Done and Effective
Healing Done with the full metrics' formulas, Deaths, Red Zone Deaths and
Ultimate Ability Activations) and picks each honor's leaders. Both views count
every champion of both teams: the replay's game is over, so Agent POV has
nothing left to hide. On the
owner's 300-tick game of 3 October 2026 all sixty per-champion totals equal
the replay's full metrics; adding them up takes about 15 ms once JAX's kill
credit calculation is warm (about 0.4 s on the first replay of a process).
Match Summary opens the same totals as a table under the arena (one row per
metric, one column per team total and champion; in the BattleClient it takes
the deck's place), and a view switch closes it; Show Arena tucks the card into
its band plate for the session, and a **Match Summary** key then shows in the
arena's top-right corner beside the other two, which opens or closes the
table again.

Five status and aura switches belong to the Rune Zones look
(`src/looks/statuses.js`); each draws only from the viewed audience's own
scene and events, and Off keeps the plain, truthful drawing:

- **Aura Glints** (Combat, off by default): on a tick when a Warrior's or
  Mage's aura helped an ally inside it, the stud of that aura's Ward Circle that
  faces the ally flashes once at Impact (120 ms, play only; it needs **Aura
  Fields**). Oracle View reads the covering emitters on the recorded damage
  rows. Agent POV has no such rows, so it works the glint out from the team's
  own ability rows and tick-start spots: the aura's champion and the ally must
  have been seen at tick start and within 2.0 units.
- **Barrier Held** (Moments, on by default): a bronze collar at 1.06 times the
  body radius closes at Impact around a champion that lived only because
  another champion's Guardian's Barrier cut the damage it took, flashes once,
  and rests at 35% for the tick, so a paused frame shows a faint collar. With
  Aura Fields on, the Warrior's studs flash too. Oracle View uses the recorded
  damage before the barrier and the covering Warriors. Agent POV uses only
  champions the team saw (tick-start spots, 0.85 per covering Warrior) and the
  net life change its rows carry, taken as damage with no healing, so it can
  miss a save but never invents one.
- **Spawn Shield Expiry** (Life And Death, on by default): the Ward Lift, a
  dashed silver ring at 0.95 times the body radius, lifts off a champion as
  its Spawn Shield ends, in the Settle window (play only). The plan row gives
  the look the body's radius in pixels. The band closes with it: the
  Champions look (`src/looks/champions.js`) keeps the ward (veil, dashed ring
  and one diamond) until Settle, then cross-fades it into the solid team ring
  over 200 ms while the token comes up from 80% to full; off, the ward cuts at
  Settle. In Agent POV both show only on a champion the team saw at tick
  start.
- **Status Clear On Death** (Bodies, off by default): a champion that falls
  this tick keeps its tick-start marks and Rune Ring until its own Fall beat.
  On, they give way in 53 ms while an ash copy flashes and fades by 160 ms;
  off, they cut at the Fall. Tick-start marks are rebuilt from the frame:
  statuses cleared by the death have unknown ticks (their pieces show as
  ghosts), statuses that aged out had 1. A paused frame shows the fallen token
  with no marks. In Agent POV it plays only for a fall the team watched from
  tick start; an unseen fall has no Fall beat.
- **Freezing Trap Break** (Bodies, on by default): when damage breaks a seen
  Freezing Trap (Stun), the ice pieces go out in 60 ms at Settle, a crack
  (80%) comes in at Impact and six shards (45%) at Settle; the crack and
  shards stay for the frame, so a paused frame shows them. Off, the ice cuts
  out at Settle with no crack or shards.

**Momentum** (HUD, on by default) marks each team's recent form on its First
To bar in the score strip: the points it scored in the last 10 ticks glow in
the team's lighter color over a thin silver line, and older points keep the
team color. The bar's hover tip adds **Last 10 Ticks** ("3 points, the bright
segments", or "No points"). The baseline is a public match fact the server
sends with every frame (`match_summary.scores_ten_ticks_ago`: each team's score
10 frames back, or at frame 0 in a younger game), so seeking to any tick shows
the right marks and the page keeps no history. Scores are public, so Agent POV
shows exactly what Oracle View shows. Off, every scored point looks the same.

Two **Replay Tools** switches (both off by default, Replay Viewer only) show
the recorded next tick:

- **Next Step Ghost** draws a dotted circle, the BattleClient's Step Ghost
  ring, at the spot where each living champion stands at the end of the next
  tick. The spots come from the frame's `next_end_spots` field, which the
  server reads from the recorded next frame. In Agent POV it lists only
  champions the viewing team sees now and still sees at the end of the next
  tick, so a ghost never tells where an unseen champion went. A champion that
  does not move gets no ghost, and the last tick has none.
- **All Staged Orders** draws the Next Action Thread for every champion at
  once: a dashed silver line to whoever it acts on next, with a dot for a
  Basic or a diamond for an Ultimate under that body (a self Ultimate, such as
  the Mage's Burst, puts the diamond under the champion itself). It reads the
  frame's own recorded next actions (`upcoming_transition`). In Agent POV
  those hold only the recipient's own order: champions cannot see their
  teammates' plans, and the other team's orders never show. The selected
  champion's line stays the Next Action Thread's, so no line is drawn twice.

Both are recorded in the PNG provenance (version 4 and later) with the other
battlefield switches.

The map label reads either the recording's map metadata or its verified authored
layout name, revision and digest. Older authored replays such as Three Body
Problem r3 therefore retain their recorded map identity. Geometry alone does not
assign an unnamed historical recording to a train, validation or test split.

Notices last 1.5 seconds at normal playback speed and remain readable on their
paused/static transition. Seeking, restarting or replacing the replay clears the
previous notices. These researcher HUD details are available in both POVs and do
not change policy observations or battlefield visibility.

Duration Status Badges includes the white crossed-swords **In Combat**
countdown. Basic Ability Effects and Ultimate Ability Effects each own their
corresponding activation presentation and damage/healing impact glyphs.
Scrolling Battle Text owns the complete net-health unit: outcome glyph, signed
value, recipient label, and connector. Those parts are enabled or disabled
together; damage and healing remain distinguished by their outcome sign and
color rather than by separate filters.

Saved Beat (on by default), Last Breath and Battle Text Detail (both off by
default) add to the
life numbers, so they need Scrolling Battle Text. Saved Beat writes "Saved"
above the number of a champion that a heal kept alive through blows that would
have killed it, and flashes its rim gold twice in play (Oracle View only: Agent
POV rows carry no damage or healing totals). Last Breath tints the empty part
of a living champion's life in a dim ember when it lost at least as much as it
has left this tick (in Agent POV only for a champion seen at both ends of the
tick). Battle Text Detail sizes each number by the share of life it took (13,
15 or 18 px) and lets it drift into place; in Oracle View it also adds the
attackers' share underline and a "Beyond Death N" tag, which is dropped when it
would cover another champion or number. A paused frame shows the word, the
ember and the detail, never the flash.

A filter change pauses playback and reinstalls the current settled summary after
filtering, so disabled paint never consumes layout space. Filters and Ranges do
not change authorized data or authorized event data used by battlefield
choreography. **Reset To Default Configuration** restores every switch (it is grayed
while nothing differs). **Enable All** and **Disable All** turn every switch of
this client on or off; neither changes the view.

## PNG Export and Metrics

**Export PNG** is enabled only when a matched replay frame and presentation are
connected, visible, settled, and free of pending replay/presentation work. It
exports only the battlefield at twice its displayed pixel dimensions. The
toolbar, timeline, and inspectors are outside the image. The result uses the
bundled fonts and locked battlefield background, reflects the current audience,
camera, selection, Ranges, and visual-filter states, and embeds one standard
`MARL-BattleGrounds Replay Provenance` iTXt record. It shows what is on screen,
so a zoomed camera exports the zoomed view, but it never includes the match
moment cards (moment plates, the Lineup and the Verdict): a figure shows the
game. New exports write provenance version 5, which records the battlefield
switches (version 3's list plus High Contrast, Next Step Ghost and All Staged
Orders) and the camera frame as a world rectangle; PNGs saved with versions 1
to 4 still read. Exported PNGs carry no brand mark. Export does not navigate the replay or request another
replay frame.

The scoreboard derives the task, participant identities, current scores and
configured target from the captured episode. Its tick line reads "Tick 40 of
156": the second number is the replay's last recorded tick (where the game
ended, or where it was saved from the BattleClient), the same number the
transport readout shows, not the 300-tick draw limit. Recordings made from 2 October
2026, by `evaluate` or saved from the BattleClient or DevClient, keep each
champion's controller name, and the scoreboard shows exactly those names. An
older recording keeps its fixed scenario title for Team B of a fixed scenario
("TDM Scenario 5 BETA Controller"), even beside a recorded team name, as
before. Otherwise it falls back to its recorded team name, shown exactly as
recorded (`tdm-beta`, where earlier Viewers showed `Tdm beta`); with no
recorded name it keeps the older labels (a file-style policy ID made
readable). Exact IDs and checkpoint digests remain in the name tooltip. An old
recording with only a hash shows **Recorded System** instead of using the hash
as a player name. Victory, draw and defeat appear
only when the current frame contains the task completion event. Seeking backward
removes the later result. Legacy task-zero replays say **Combat diagnostic**.
Changing between Oracle and Agent POV preserves the cursor and playing intent.

Open the **Metrics** tab to prepare the offline analysis once. Its Scope switch
reads **Up To Tick N** (Up to Current Tick) or **Whole Game** (Entire Episode).
**Up to Current Tick** is the default: local
frame k uses exactly k captured transitions. The displayed tick comes from the
captured simulator state, which may start above zero. **Entire Episode** selects
the complete captured prefix; interrupted recordings do not become completed
results. An indeterminate progress indicator appears while metrics are prepared.
Longer replays can take longer; the indicator makes no fixed-duration promise.
Playback, seeking and POV changes remain available while analysis prepares. CSV
preparation also shows an indicator when the metrics panel is closed. The
**Topic** selector starts with **Episode Results**. The Scope switch, Topic,
View and **Find A Measurement** stay fixed at the top of the tab; only the
results and the table under them scroll. The table always keeps a few rows:
when the top part and those rows do not fit (open search results on a short
screen such as 1366 x 768, or 200% zoom), the whole tab scrolls, so nothing
is cut off and **Show More** is always reachable. Team rows show their crest and
agent and recipient rows their class emblem beside the words, in the table
and in search results; Episode rows stay plain. The topic list has 28 topics under six
headings, including five named Ultimates and **Team Formation**. Sixteen topics
have separate **Totals** and **By Recipient** views; twelve have one table.
**Respawning** shows each team's wave count, mean agents returned per wave, and
mean waiting ticks seen in the recording. **Red Zone**, right after it, shows
each team's kills and deaths inside a team's own Red Zone, then the agents who
helped with those kills and their shares, then the agents who died there and
their shares. Each such death counts once but gave the enemy team 2 points, so
compare it with the kills and scores in **Episode Results**. A recording made
before the Red Zone rule shows a dash in every Red Zone row, and its **Blank
When** help says the rule was not recorded. **Deaths and Time Dead** keeps death
counts and dead time. Totals hold team-wide and acting-agent measurements. By
Recipient holds affected-agent totals and team-to-agent or agent-to-agent
details. Related views share no rows. Different topics can reuse an existing
measurement to help explain it; this does not add a CSV column or calculation.

Inside the five named Ultimate topics, rows and tooltips use **Mage Burst**,
**Warrior Charge**, **Hunter Trap**, **Rogue Poison** and **Priest Salvation**.
Other topics keep their shared measurement names. When a share uses a combined
team total, its tooltip still names all abilities included in that total. These
local wording changes do not change the numbers.

Schema 13 adds **Warrior Charge Applications**, **Rogue Poison Applications**
and **Priest Salvation Applications** for each team: six new columns. Their CSV
names are `team_{a,b}_warrior_charge_applications`,
`team_{a,b}_rogue_poison_applications` and
`team_{a,b}_priest_holy_word_salvation_applications`. These team rows appear
only in their named Ultimate topics. All existing CSV names remain unchanged,
including Slow, Stun and Anti-Heal application columns. Status Applications
keeps those separate effect rows. Mage Burst and Hunter Trap retain their
existing team application columns. Earlier CSV files stay untouched. See the
[schema-13 column
contract](../evaluation/metric_specification.md#schema-13-team-ability-application-columns).

**Find a Measurement** searches all 11,192 numerical columns. An exact CSV name
is checked first and returns that column, including one that does not apply to
the recorded roster. Other searches use measurement names, topics, related words
and recorded agent identities. The start of a word works too: `regen` finds
regeneration; `crippl pois` finds Crippling Poison. `Overhealing` finds excess
healing, and `spread out` finds teammate distances.

Search keeps who acts and who receives the effect separate. For example,
`healing from Priest to Mage` differs from `healing from Mage to Priest`.
`Damage received by Priest from Warrior` asks for the Warrior's damage to the
Priest. Add a team or an agent ID to make the identity more specific. A class
named inside an effect does not identify that effect's recorded caster: `damage
received while slowed by Rogue Poison` describes the damaged agent's status. It
does not say which Rogue caused the slow.

Search checks the kind of measurement and these roles before looking at related
words. A description cannot turn regeneration into damage or reverse the giver
and receiver. Applicable results come before inapplicable results; direct names
come before related detail. Search explains why an inapplicable column cannot
apply. Searching an ability name puts its own applications and effects before
shared context such as Total Kills. `Team A Charge` means Team A's Charge
actions; `Charge Slow Count` asks for the separate Slow application count. It
supports common measurement questions, not arbitrary English. Conditions such as
`damage without poison` or `damage after poison` show a clear explanation
instead of guessing a different measurement. Applicable matches open their table
and focus the named measure. A Moon Silver bar on its left and a soft silver
tint mark the row for three seconds. Choosing another result moves the cue; choosing the same
result starts its three seconds again. The cue disappears without moving
keyboard focus. It does not flash or move, including when reduced motion is
enabled. The search list is loaded once per replay analysis. Seeking, scope
changes and POV switches preserve it; replacing the replay clears it. Identity
fields remain in Episode Details and the column guide. Click outside the search
area to hide its results. Click or focus the search box to reopen the same list.
Your text and results loaded with Show More stay. Escape also hides the list
without clearing your text. Use Up and Down to highlight a result, then Enter to
open it. The cursor stays in the search box while you move through results, so
you can keep typing.

Scope, Topic, View, search, the definition and column headings remain outside
the scrolling rows. Rows show **Subject**, **Measure** and **Value**, with teams
first and numeric agent IDs using their recorded classes. Unavailable values
appear as a dash; hover or focus a measure for its definition, units and
missing-value rules. Formation summaries include mean distance and the number of
measurements taken while both allies were alive. Coordination and class
associations describe measured patterns. They do not prove that a policy learned
strategic reasoning.

**Download Metrics CSV** exports one wide row for the selected boundary, using
the same scalar names, order and values as the run tables; unavailable cells are
empty. Schema 15 exports 11,241 columns: 49 identity fields and 11,192 numerical
measurements. The export identifies its scope, local frame index, actual
simulator tick, and captured roster/policy identities. **Episode Details**
downloads the recorded episode, policy, completion and runtime metadata as JSON
without copying the trajectory or requiring a metrics sidecar. Older V1 replays
can be analyzed from captured facts without rewriting their artifacts. Derived
analysis records its own source fingerprint so it can be traced separately from
the original replay. Metrics remain researcher analysis in every visual POV and
never enter policy inputs, recipient tooltips or battlefield visibility.
Preparing analysis does not hold the replay command lock; subsequent seeks reuse
cached prefix components. Metric computation uses the captured game directly and
retains inexpensive consistency checks. Imported artifacts still receive the
input checks described below; computing metrics does not repeatedly revalidate
that same data.

## Static Matplotlib Frame

Render one exact frame without opening the browser or starting an HTTP server:

```bash
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json --static --frame-index 12

./scripts/dev/run_replay_viewer.sh \
  --sample-replay recovery-status-lifecycle --static --frame-index 3

./scripts/dev/run_replay_viewer.sh \
  --scenario stacked_team_auras --static --frame-index 1
```

`--frame-index` is required with `--static`. Only the selected input, range
state, and—for a scripted scenario—seed/stress authorization are accepted in
this mode; browser audience, POV, port, and opening options are rejected. The
shell activates the optional `viz` dependency. Static mode validates the
complete bundle and paints the exact researcher frame through the stateless
scene-native Matplotlib adapter.

## Loopback and Artifact Safety

Local replays, checked samples, and newly recorded demonstrations are validated
as complete artifacts before the server opens a port or launches a browser.
Historical V1 companions are checked when present or required by the
checked-sample contract; V2, V3 and current V4 replays are self-contained. The loader
rejects invalid schemas, noncanonical bytes, hash mismatches, broken event/frame
links, invalid frame indices or POV recipients, and unsupported paths. Its file
checks also reject prohibited symlink paths.

The browser server binds only to `127.0.0.1`, uses a random fragment-delivered
capability token, validates request headers and origins, serves an explicit
asset allowlist under restrictive Content Security Policy and `no-store`, and
handles replay commands one at a time with revision and command-ID checks.
Replay commands change only which recorded frame/view is selected; none can call
`step` or modify the source files.

Refresh and **Reconnect** fetch current authority without repeating a seek.
Closing the tab does not stop Python; use **Exit Replay Viewer** or `Ctrl-C`.
Once Exit is accepted the connection pill reads **Closed** (in the Viewer and
in BattleClient alike), because the page reads no more frames.

## Troubleshooting

- **No selector:** choose exactly one replay, sample, scenario, or list
  operation.
- **Artifact rejected before a URL appears:** inspect the reported path,
  canonicality, companion, frame, or POV error; the launcher intentionally did
  not bind a server.
- **Manual arena selected as a scenario:** open it with the
  [BattleClient](../battle_client.md).
- **Stress scenario rejected:** add `--include-stress` to the scenario command.
- **Transport disabled:** reconnect if offline; Start/negative moves stop at the
  lower bound, positive moves/End stop at the captured endpoint, and artifact
  actions wait for a settled frame.
- **Metrics unavailable:** inspect the analysis panel's error. A missing V1
  metrics sidecar does not prevent analysis from recorded facts or the Episode
  Details download; V2, V3 and current V4 recordings do not need a sidecar.
- **PNG export disabled:** pause playback and wait for the exact-frame summary
  to settle in a visible connected tab.
- **Static Matplotlib import failed:** run `uv sync --extra viz`.
- **Server remains after tab closure:** use the in-page Exit action or `Ctrl-C`.

Contributor prose follows the [Documentation
Standard](documentation_standard.md). The replay format itself is documented in
[replay_format.md](../evaluation/replay_format.md). Return to the [browser-tools
migration page](visual_debugger.md) or the [project README](../../README.md).

Current BattleClient and scripted-scenario recording captures the replay without
computing the full metric suite. Opening Evaluation Metrics or requesting CSV
opts into full replay analysis once; the result is cached across cursor and POV
changes. Library callers likewise opt in with `analyze_replay(bundle,
full=True)`. Research training and evaluation independently select priority/full
metrics and replay saving through the [evaluation
workflow](../evaluation/workflows.md).
