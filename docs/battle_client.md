# BattleClient

BattleClient runs a match on your computer and shows it in your browser. Python
loads your policies; the browser displays the game, accepts Manual actions and
opens saved replays. It uses the same simulator and System interface as evaluation.

## Install And Open

Use Python 3.14 and the supplied MARL-BattleGrounds wheel. From the folder
containing the current `marl_battlegrounds-0.0.0-py3-none-any.whl` file, create an
environment and install it:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install ./marl_battlegrounds-0.0.0-py3-none-any.whl
```

The base package includes the local server, browser files, maps, scenarios and
built-in policies. Manual play, built-in opponents and browser Replay Viewer work
with this installation. It needs no checkout, Node, frontend build, Docker or
hosted website. These commands use a local wheel; they do not assume this
candidate has been published to a package index. For another release, use its
actual wheel file name.

Working from a source checkout instead (for example after `pip install -e .`)
also works. A checkout has no packaged browser files, so the BattleClient and the
Replay Viewer serve the same checkout's `web/visual_debugger` page.

Native learned checkpoints also need the `training` extra:

```bash
python -m pip install "./marl_battlegrounds-0.0.0-py3-none-any.whl[training]"
```

Your custom method may need its own Python libraries. An LLM uses its separately
configured service. We do not bundle or download policy/model weights. Save this
small program as `play.py` and run `python play.py` in the installed environment:

```python
import marl_battlegrounds as marl_bgs

marl_bgs.battle_client(map_id=0, team_a="manual", team_b="tdm-alpha")
```

The call first prepares the first turn (it prints "preparing the first turn
(compiling once)"; this takes a few seconds, see
[Startup And Turn Speed](#startup-and-turn-speed)). Then it opens a browser and
prints its local URL. If the browser cannot open, copy that URL into your browser. The call waits until you use Exit or press
Ctrl-C in its terminal. Closing a tab leaves Python running; reopen the printed
URL to return. `open_browser=False` prints the URL without opening it. If you
opened the saved replay with **Open Replay**, Exit closes only the game: the call
then prints the Replay Viewer's address and waits until you press Exit in the
Replay Viewer or Ctrl-C (see [Save And Continue](#save-and-continue)).

**Which browser.** We recommend an up-to-date desktop Firefox: the
maintainers play and review the BattleClient in it, and it gets a smaller
automated check of play, steps and Agent POV. Chrome and Edge work too: the
full automated browser tests run on Chromium, their engine. The client uses
only standard web features, so Safari should work as well, but it is not
tested. The client does not choose a browser: it opens your computer's default
browser (Python's `webbrowser` module). To use a different one, pass
`open_browser=False` and paste the printed URL into it. The client is made for
desktop and laptop screens, not phones or tablets.

Choose exactly one `map_id` or `scenario_id`. `list_tdm_maps()` and
`list_tdm_scenarios()` list the packaged choices. Ordinary maps default to the
canonical five classes per team, both teams Manual, seed 0, 300 steps, score
threshold 20 and Red Zone depth 5.0. Policy execution uses SharedObs.

For complete saved-policy matchups, follow [Play Your Saved Methods](battle_client_tutorial.md).
It includes a downloadable script, optional untrained actors, six native families,
a real LLM setup, custom Systems and fixed-scenario partner comparisons.

## Choose Controllers

A controller can be:

- `"manual"` for your actions.
- An existing built-in name: `"random"`, `"tdm-alpha"`, `"tdm-beta"` or `"tdm-gamma"`.
- A supported native actor export or full checkpoint folder.
- Your existing `Policy` or `System` object.
- A trusted `"module:function"` factory that returns a Policy or System.

Built-in names and bare factory strings keep their meaning even beside a folder
with the same name. Use `Path("my_actor")`, `"./my_actor"` or an absolute path to
name a saved model explicitly. Checkpoint paths must not contain symbolic links.
Ordinary paths that name the same folder share one load. Browser checkpoint fields
accept only folder paths and never run Python factories.

Use one controller for a whole team, or a list with one entry for each configured
roster slot. Repeated entries share loading and grouped inference where supported.
Different object instances remain different methods. Team memory stays separate.
A System can contain more internal components than there are agent slots. The
client does not inspect or rewrite its internal design. Your System still owns
its allowed inputs, action rows and memory under the existing System contract.

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds.training import load_system

my_policy = load_system("checkpoints/my_actor")
opponent = load_system("checkpoints/opponent_actor")
marl_bgs.battle_client(
    map_id=0,
    team_a=["manual", my_policy, my_policy, my_policy, my_policy],
    team_b=opponent,
)
```

Those paths must contain real supported saved artifacts. They are not bundled
examples. A native checkpoint need not live in a special directory. For another
format or framework, load it with your own code and pass a valid System. Its
permitted inputs, actions, memory and resource contract stay the same as in the
[System workflow](evaluation/workflows.md#policy-inputs).

Ordinary teams independently support one to five roster members, including
repeated classes. `team_a_roster` and `team_b_roster` set their ordered classes.
Controller lists follow that order. Loading weights never silently changes a
roster or makes a slot-specific model compatible with a different roster.

## Play And Watch

The screen shows the map, step and score limits, Red Zone depth, classes and
each configured agent's controller. Under the battlefield, the deck plans one
turn for the champion you order, in four steps. Its top line is the message
line: what each of your orders did, refusals and game messages, with **Keys
On** or **Keys Off · Click The Arena** at its right (a sentence too long for the
line ends in "…"; point at it to read it whole). **Your Champion** shows its
token, class, calling, Agent ID, "You Control This Agent", its **Life** bar,
and **‹ Prev** and **Next ›** (Tab). **1 Step** is a pad of nine keys, Q W E A
X D Z S C, where X is Stay; the name of the chosen step shows in its heading
("1 Step North-West"). The arrow keys do nothing. **2 Target** shows the picked target ("Mage · ID 5" with
its Life bar under the name, "No Target Yet", or, in Agent POV, "Warrior · ID 6
· Out Of Sight" with no life for a kept target your team cannot see now), a
**× Esc** button that clears it, and
**Allowed Now**: one button per body you may use an ability on right now
with its abilities ("Mage · ID 0 Arcane Blast"); when three or more would not
fit, they show as a two-column grid of short buttons, and pointing at one shows
its abilities), or "Nothing is allowed right now." **3
Action** has three slots, **None**, the class's Basic and its Ultimate, with
reach and the Ultimate's Rune Ring and readiness ("Ready" or "In 3 Ticks").
A slot is tagged **Armed**, **Auto** (the game armed your Basic itself when you
picked a target it allows) or **Not Allowed**. **4 Lock In** is the **Lock In
Turn** button (Space, Enter or N, the one-tick key); each champion's order is
listed once, in the rail's **Your Turn** card. The deck keeps one height from
turn to turn, so the arena never changes size when Allowed Now grows. It is
short, so the arena gets the room: about 176 px tall in a 1920 x 1080 window,
where Your Champion takes two short columns and Allowed Now sits beside the
target, and about 194 px on laptops. While the turn
plays the button reads **Orders Sent**, **Policies Choosing…** and **Everyone
Acts**. You may plan several Manual champions before Lock In; changing focus
keeps the other plans. A joint turn uses all your Manual rows and the policies'
actions once. Everyone acts at the same moment.

Every champion's life in the panels is the same **Life** bar: the word "Life",
a thin bar and the number beside it ("200 / 200"). The bar is green above half
life, yellow at half or less and red at 30% or less. You see it in Your
Champion, under the target's name, in the Watch Only box and on the Champion
Card; the Replay Viewer's panels use it too. A fallen champion reads "Fallen ·
Returns In 3 Ticks" instead. The deck and the rail say a champion has fallen at
its Fall beat, the moment the arena shows the fall and the score changes, never
earlier. The tokens on the arena keep their own Life Vessel.

The rail beside the arena has three cards and the Fallen panel. **Your Turn** shows the tick, a set
pill ("1 Of 1 Set"), **Guide**, **Hints** and **Off** (how much help; the
choice is kept in this browser), **Do This Now** (one next step; at the Guide
level a dashed outline marks the control it names), the four phases **You
Plan · Lock In · Policies Choose · All Act**, one row per champion you play
with its step, action and **Set** or **Default** (click a row to give it
orders; a fallen champion's row reads "Fallen · Returns In 3 Ticks" with a
**Fallen** tag, because it takes no orders until it returns; the set pill and Do
This Now count and ask only champions that can take orders). **Set** means the
game holds an order for that champion that is not the default one, Stay with
No Combat. So the pill and the tags are the same after a reload and in both
views, and Stay with No Combat reads **Default** even when you chose it on
purpose. With one champion on each team, Agent POV shows only the viewing
team's orders: your champion on the other team keeps its row (click it to give
it orders), which reads **Out Of Sight** when your team cannot see it, with
nothing else about it, or **In Sight · Order Hidden** when it can. Then comes
**Policies**: every policy of both teams with its team crest, its
full name and its number of champions ("Recurrent MAPPO · 4 Champions"), and
**Waiting**, **Choosing…** or **Done**. It lists the same controllers, names
and counts as the scorebug's hover lists, without "You". Right under Your
Turn, the **Fallen** panel lists every fall of the last 8 ticks, so it shows
without scrolling. The **Class Guide** shows the champion
in view: its calling, lore line, life, speed, sight, and its Basic and
Ultimate written from the game's own numbers, with where each is allowed now
for a champion you play. **How To Play** repeats the steps for your roster:
"The silver ring is your champion." for one, "The silver rings are your
champions." and how to order another for several, and Run, Play One Turn and
Pause when nobody is yours. It shows at the Guide level, and **Full Guide**
opens the full help. Do This Now never asks for a target when none is allowed:
it says so and points to Step and Lock In.

The top bar shows the brand, the place ("**Teutoburg Forest** · Map 47 · Team
Deathmatch", or a scenario's tale title), the **Agent POV** and **Oracle View**
switch (in Agent POV the floor is dark wherever your team cannot see: beyond
its champions' sight reach and in the shadow behind every stone), a
**Connected** pill (it reads **Disconnected** when the page loses the
local server), the **Recording** pill, then **Visual Filters** (a drawer of
display switches, closed until you open it), **How To Play ?**, **Finish And
Save**, **Restart R** and **Exit**. Finish And Save reads **Saving Replay…**
while the replay is saved and then turns into **Open Replay**, which opens the
saved game in a Replay Viewer tab (and opens it again when pressed again); after
a failed save it reads **Retry Save**. There is no Visual Key: hover tips, the
Champion Card and Frame Explanation explain each mark where it shows. The drawer
lists the live switches under their
V5 names, including the twelve V5 add-ons described below, with the four
presets (**Default Configuration**, **Researcher**, **Manuscript** and
**Minimal**), **Enable All**, **Disable All** and **Reset To Default
Configuration**. The shipped defaults turn on **High Contrast** (one switch in
all three clients: brighter rings, clearer stones and crisper text), **Class
Seals**, **Moon Motes**, **Death Dimming**, **Spawn Shield Expiry**, **Saved
Beat**, **Champion Streaks** and **Momentum**, and turn off **Moonlight**,
**Floor Grid** and **Time Of Night**. The arena's top-right corner holds two
key buttons stacked one over the other, showing only their key: **V**
(Everyone In View) and **F** (Frame Explanation). Each does what its key does,
lights while it is on, and names itself on hover. The full map
name ("Teutoburg Forest · Map 47")
always shows, at every window size from 1366 x 768 up and in every phase: when
the bar runs short, "Team Deathmatch" disappears first (whole, never cut
off halfway), then the "BattleClient" name is cut. At game over a **Game Over** pill appears after
**Connected**. The **Visual Filters** drawer opens over the rail's cards, just
below the scorebug, so it never covers the scorebug, the deck or **Lock In**.
Every client spells the brand **MARL-BGs**.

**Class Seals** (Visual Filters, Bodies, on by default) puts each champion on
its class seal instead of the plain class glow: the Mage on a starglass
octagon, the Warrior on a bronze boss with eight rivets, the Hunter on a
greenwood four-leaf shape, the Rogue on a steel diamond with curved sides and
the Priest on a gold sun (its emblem then shows only its cross, because the
seal is its sun). A seal is faint while the Ultimate recharges. When
the Ultimate is ready, the seal lights up in its own material: the Mage's
facets, the Warrior's white-hot rivets, the Hunter's leaf veins, the Rogue's
jade line and the Priest's warm sun. It goes dark when the champion uses its
Ultimate. While a status mark is on the champion, the seal fades back so the
mark reads first. A fallen champion keeps a faint ash outline of its seal.
Seals need a body at least 18 pixels in radius, so a very small arena shows
the plain glow. With **Cooldown Effects** off, every
seal stays faint. The **Your Champion** token in the deck is the same token, so
it shows its seal too. In Agent POV, seals show only on champions your team
sees; a champion whose Ultimate timer your team cannot see keeps a faint seal.

**Kill Credit Glint** (Visual Filters, Life And Death, off by default) shows
that kill credit is shared. When a champion falls, every living champion that
took part in the kill gets the same thin ring in its class color at the Fall
beat. The ring sits just outside the body, under the bodies, and fades from
70% to nothing in about a quarter of a second; at most four rings show for one
fall. A paused tick shows no ring, and it needs **Death Effects**. In Agent
POV a champion gets a ring only when your team sees it standing and saw its
blow land on the fallen champion, so a ring never tells you about a hidden
champion or a blow your team did not see. Off: no rings; the Fallen panel
still lists everyone who shared the kill.

**Champion Streaks** (Visual Filters, Moments, on by default) lets a herald
card name a champion each time it has taken part in 5, 10, 15 or 20
takedowns without falling: its medallion, "Team A Mage" and "10 Takedowns
Without Falling", with any other moment of the tick added after a dot, such
as "· Double Takedown". The words say "took part", because kill
credit is shared. The server keeps every view's count on the game (Oracle
View and each champion's Agent POV), adds each turn's takedowns and sends your
view's streak with the turn, so switching views or reloading the page never
loses a count, and **Restart** starts it again. In Agent POV only takedowns
your team saw count: a fall it saw, struck by a champion it saw at the start
of the turn. Off: the herald cards are the plain ones.

**Herald cards** (Visual Filters, Moments, **Match Moments**) name a turn's
big moments, with the store's words: **Penta Takedown** (all five champions
of one team fell on one turn; it heads the card with "Team B Wiped" in its top
line), **Team Wiped** (every champion of a team is down), **First Blood** (the
first fall of a game that began at tick 0
with 0-0), **Match Point** (a team's score rose to exactly 1 point short of
First To), **Double**, **Triple** or **Quadruple Takedown**, a run such as **5
Unanswered** (a team's 5th, 10th, 15th or 20th takedown in a row without a
loss of its own), **Rescue** (a Priest's heal kept a champion alive through blows that
would have killed it), **Ultimate Takedown**, **Trade**, **Scores Level** (the
scores became level after one team led by 3 or more), **Ultimate Combo** (two
or more damaging Ultimates struck one champion) and **30 Ticks Left** (while
no team has reached First To; there is no countdown after it). When several
happen on one
turn the most important heads the card and up to two more join its small top
line, such as "Team A +2 · Trade". A plain single takedown gets no card; the
Fallen panel tells it. A card enters with the Fall beat (a Rescue, the ticks
left or a combo on a champion that lived wait until the turn settles) and
holds through the next turn. Only one card shows at a time, and each stays at
least 2 seconds; a card that comes sooner waits its turn, and the old card
fades out before the new one fades in. Penta Takedown, First Blood, Team Wiped
and Match Point also draw a thin line across the top band behind the card.
With **Death Announcer** off, the moments born from a fall are left out. Every
card is the same plate and always hugs the top edge of the arena: it slides
sideways to cover the fewest champions, keeps clear of the V and F keys at
the arena's top right, shrinks a little only when it must,
and if no spot is free it still shows; it never moves to the bottom, never
turns into a thin band and never silently disappears. It keeps its spot while
it holds through the next turn.

The server keeps the facts a card needs that one turn cannot show, for every
view (Oracle View and each champion's Agent POV): whether the game began at
tick 0 with 0-0, the largest lead so far, each team's run of takedowns and
each turn's rescues. It sends your view's facts with the turn, so switching
views or reloading the page never changes a card, and **Restart** starts them
again. In Agent POV a card names only what your team saw: a run counts only
falls your team saw, a Rescue shows only when your team saw the champion, the
Priest's heal and every blow on it, a combo only blows your team saw land, and
Team Wiped only when your team sees every champion of that team fallen. The
scores, First To and the tick are the public scoreboard, the same in both
views.

At game over the **Verdict** card shows why the game ended ("Tick 300 · No
Team Reached 20"), each team's controllers under its name ("You", "Recurrent
MAPPO"; a long single policy name shrinks and is cut with "…", and "You + 4
Policies" stands in only for a team of several policies), both scores and
Victory, Defeat or Draw. The team whose home end is on the left of the arena
has the left column. A draw at the time limit with a lead adds "A Lead At Tick
300 Wins Nothing". **Honors** names the champions with the most Kill
Contributions, Damage Done and Effective Healing Done (tied champions share
the honor, up to six shown and then "+N"; an honor nobody earned is left
out). The server counts these over the whole game with the same rules as the
metric catalog. Both views count every champion of both teams: the game is
over, so Agent POV has nothing left to hide. **Match Summary** opens every
champion's totals (Kill Contributions, Deaths, Damage Done, Effective Healing
Done, Red Zone Deaths and Ultimate Ability Activations) in the deck's place
below the arena; **Close** or a view switch closes it. **Show Arena** tucks the
Verdict into its plate at the top of the arena for the rest of the session; a
**Match Summary** button then shows in the arena's top-right corner, under V
and F, and opens or closes the sheet again.

**Aura Glints** (Visual Filters, Combat, off by default) shows when an aura
helped. On a tick when a Warrior's or Mage's aura helped an ally inside it, one
stud of that aura's ring glints at the moment the blows land: the stud that
faces the ally. A Warrior's aura helps an ally that takes a blow; a Mage's aura
helps an ally that deals one. It needs **Aura Fields**, and a paused tick shows
no glint. In Agent POV a glint shows only when your team saw the aura's
champion and the helped ally at the start of the tick, and saw the blow. Off:
no glints.

**Barrier Held** (Visual Filters, Moments, on by default) marks a rare save. A
bronze collar closes just outside a champion that lived only because another
champion's Guardian's Barrier (a Warrior's aura) cut the damage it took: the
same blows without the barrier would have taken all its life. The collar
flashes when the blows land and then rests faint for the tick, so a paused
tick shows a faint collar. With **Aura Fields** on, the Warrior's ring flashes
its studs at the same moment. In Agent POV the client works the moment out
only from champions your team sees, and from the life change your team saw, so
it can miss a save but never shows one that did not happen. Off: no collar.

**Spawn Shield Expiry** (Visual Filters, Life And Death, on by default) marks
the moment a Spawn Shield ends. The ward stays until the tick settles; then its
dashed band closes into the solid team ring over a fifth of a second, the token
comes up to full strength, and a dashed silver ring lifts off the champion. A
paused tick does not show it. In Agent POV it shows only on a champion your
team saw at the start of the tick. Off: the ward simply ends as the tick
settles.

**Status Clear On Death** (Visual Filters, Bodies, off by default) changes how
a falling champion's status marks end. In every case a champion that falls
keeps the marks and Rune Ring it had at the start of the tick until its own
Fall beat. On, the marks then turn to ash and fade in about a sixth of a
second; off, they simply go at the Fall. A mark whose count the tick cannot
tell (a status the death cleared) shows its pieces as faint ghosts. In Agent
POV this plays only for a fall your team watched from the start of the tick.

**Freezing Trap Break** (Visual Filters, Bodies, on by default) shows the ice
shattering when damage breaks a Freezing Trap (Stun): a crack runs across the
token when the blow lands, then the ice goes out in a quick fade and six ice
shards ring the body for the rest of the tick, so a paused tick still shows the
crack and the shards. Off: the ice simply goes out when the tick settles. In
Agent POV it shows only on a champion your team saw at the start of the tick.

**Saved Beat** (Visual Filters, Moments, on by default) marks a heal that
kept a champion alive. When the blows of a tick would have taken all of a
champion's life but a heal on the same tick kept it standing, "Saved" in gold
sits just above its life number, and in play the champion's rim flashes gold
twice. A paused tick keeps the word but not the flash. It shows in Oracle View
only: Agent POV knows each champion's life only as your team sees it, not the
blows and heals behind it. Off: the plain number.

**Last Breath** (Visual Filters, Combat, off by default) is a warning, not a
rule. When a living champion lost at least as much life this tick as it has
left, the empty part of its life smolders in a dim ember: the empty top of its
vessel, or the empty part of its life ring when **Life Vessels** is off. One
more tick like this would end it. The ember fades in as the tick settles and
stays in a paused tick. In Agent POV it shows only on a champion your team saw
at both ends of the tick. Off: the plain life.

**Battle Text Detail** (Visual Filters, Combat, off by default) makes the life
numbers say more. A number's size shows how much of the champion's life the
change took: 13 px under a tenth, 15 px up to three tenths and 18 px from
three tenths, never smaller than the plain size. Numbers pop a little lower and
drift up into place. In Oracle View a thin bar under a damage number shows who
dealt it: one piece per attacker in its class color, sized by its share
(thicker for an Ultimate, ice blue for a Freezing Trap, then gold for any
healing on the same tick). A killing blow that went past zero adds "Beyond
Death 6" beside its number when there is room; in a crowd the tag is dropped
rather than cover another champion or number. In Agent POV only the size and
the drift change, because your team does not see who dealt a blow. Off: plain
13 or 15 px numbers.

All three ride on the life numbers, so they need **Scrolling Battle Text**.

Each team's home is marked on the field. Outside each end of the floor a
bracket in the team's color (Team A one line, Team B two) with an **A** or
**B** badge shows whose home end it is; it reaches along the sides as far as
that team's Red Zone and always fits whole inside the arena box. Inside, a
gravestone border marks the Red Zone's edge, and a skull with "×2" near its top
and bottom says that a champion who falls there gives the enemy 2 points.

The arena marks your plan under the bodies. A silver **Command Ring** circles
the champion you order. Four **Order Studs** on it show the armed action:
silver dots for No Combat, class-color dots for the Basic, class-color
diamonds for the Ultimate and hollow diamonds for the Mage's Burst. A dashed
ring circles every body the armed action may target now, from the game's own
masks, and a dashed **Order Thread** joins your champion to its target while
that target is allowed; the target also gets silver corner brackets. A dotted
**Step Ghost** the size of your champion shows where your step ends if nothing
blocks it (it uses your champion's current speed, so a slow shortens it and a
stun removes it); it is hidden while a Warrior Charge is armed, because the
Charge decides where the Warrior lands. A policy's champion you only look at
gets a soft silver ring and no plan marks. While a turn plays, the rings,
studs, brackets and reach rings move with their champions, and the Order
Thread and Step Ghost for the next turn appear once the champions have landed,
so no mark shows where a champion is going before it gets there. The reach rings of the focused
champion (Basic dashed, Ultimate dotted) are shown from the start; **G** hides
or shows them. **V** (or the arena's **Everyone In View** key, lit while it is
on) zooms to keep every champion your view shows in frame and back to the
**Whole Map**, and a short tip says which camera is on ("Everyone In View on
· V: back to Whole Map"). It sits in the deck's message line, in place of the
message, for about 4 seconds or until your next key or click, so it never
covers the arena, the scorebug, the rail or a control. When the
champions are spread over the whole field, Everyone In View is the Whole Map.

Hover a champion's body, portrait, Turn Checklist row or Allowed Now button for
its **Champion Card**, the same card the DevClient and the Replay Viewer show:

- The name and calling, the team crest and the Agent ID.
- The **Life** bar with the number beside it ("77 / 80").
- A thin bar named after the champion's Ultimate ("Burst", "Charge" and so on).
  It fills as the Ultimate recharges and reads "30 ticks", or **Ready** when the
  Ultimate can be used.
- One row per status. The first line has the status's own icon, its name and
  the time left ("5 ticks left"). The second, dimmer line says what it does
  with its exact number, then where it comes from by class, for example "It
  moves at ×0.5 speed. From a Warrior." A champion's own aura reads "From its
  own aura." A thin line under the row drains as the ticks run out. Stuns come
  first, then slows, then the rest. The card shows at most four rows, then a
  last line "+2 more" with "Hold Shift For All".
- Who plays it, and **Your Target** when it is the target of the order you are
  setting.

A fallen champion's card shows life 0 and when it comes back with its team's
next respawn wave, for example "Fallen · Returns In 2" while the scorebug reads
**Respawn Wave In 2 Ticks**. Its body, portrait and docked frame all show that
line, in every client.

The card is always one column wide, so it never spreads across the map. When
the arena is too short for the card with all its rows (in a 1280 x 720 window
the arena is only about 290 pixels tall), it shows as many rows as fit, at
least one, in the same order, then "+3 more" with "Hold Shift For All". The
name, the Life and Ultimate bars, "Played by" and **Your Target** always
show. The card works this out again each time it is drawn: on every new tick,
every view change and every window resize.

To see every status, rest the pointer on the champion and hold **Shift** by
itself for about a third of a second. The card grows in place to show them all;
with more than four it says "All 7 statuses" and "Release Shift To Shrink" at
the bottom. When that would be taller than the arena, the rows spread into two
columns, or more only when two do not fit (a column that would not make the
card shorter is not added), and the card grows wider instead. Inside the arena
the card may be drawn a little smaller, down to 85% of its size. When it would
need to be smaller than that (a short arena and many statuses), it grows past
the arena instead: it keeps its normal size, takes the number of columns that
makes it shortest, and sits over the panels next to its champion, fully on
screen and off the champion when there is room. Only a window too small for
even that draws it smaller. The card without Shift never leaves the arena.
Releasing Shift, any other key, any click and any pointer movement shrink it
back at once to the one-column card inside the arena, so every Shift key still
does its own job: **Shift + Click** still picks a target, **Shift + Tab** still
steps back to your previous champion, and **?** still opens How To Play. Shift
by itself never sends anything to the game.

A status is named by what causes it, then its plain effect: **Charge (Stun)**,
**Crippling Poison (Stun)**, **Freezing Trap (Stun)**, **Freezing Trap (Stun
Broken)** (damage broke the ice early, shown on that tick), **Serrated Arrows
(Slow)**, **Charge (Slow)**, **Crippling Poison (Slow)**, **Crippling Poison
(Healing Reduction)**, **Blessing of Freedom (Slow Resistance)**, **Burst
(Damage Boost)**, **Spawn Shield (Invulnerable)**, **In Combat**, **Rest
(Regeneration)**, **Sorcerer's Empowerment (Damage Boost)** and **Guardian's
Barrier (Damage Mitigation)**. Where one mark serves two causes (the stun stars
serve Charge and Crippling Poison), its tip lists both names.

The card is the same in Agent POV and Oracle View. In Agent POV a champion your
team cannot see gets a short card: its name, calling, crest and Agent ID,
"Unseen" and who plays it, with no life, Ultimate or status; holding Shift adds
nothing to it. Every hover place gives the name and calling, a Turn Checklist
row too, because a champion's class is public. Without Shift, the card always
sits fully inside the arena, never over the deck, the rail, **Lock In Turn**,
the scorebug or the top bar. A portrait's card appears at the arena's top edge,
just under the scorebug. Only the Hold Shift card that grows past the arena
(see above) may cover them, while Shift is held.

The card always shows the tick on screen. While the pointer rests on a
champion, the card is drawn again from every new tick and every view change.
For example, with one Manual champion on each team, rest on Rogue ID 3's
portrait and press **Tab** so the view moves to the other team: if that team
cannot see the Rogue, the card turns into
the short "Unseen" card at once; a body that the new view no longer draws loses
its card. While the card shows, no other tip shows at the same spot (the Spawn
Pad under a body, the Spawn Shield ring on a portrait): one card at a time.
With the Champion Card switched off in **Visual Filters**, the Spawn Shield
ring's own tip shows instead, on the body and on the portrait. It is a small
tip in the card's words: "Spawn Shield (Invulnerable)", "It can't be seen or
hit, and can't strike yet." and **Time Left** ("2 ticks left"). Every other
status tip reads the same way, for example a **Status Plates** cell's tip
(Visual Filters, off by default): the status's name, the card's sentence with
its source by class, and **Time Left**.

The scorebug across the top shows Team A and Team B with their crests, a
**Yours** tag on your side and one line naming who plays each team, for example
"You + Recurrent MAPPO" or "tdm-beta". Each name is the controller's own name:
a built-in's fixed name, a saved checkpoint's saved name (else its method
name), or the name you gave your System. A `marl_bgs.team` of several methods
is named per champion by the member that plays it. When the names do not fit on the line,
it says how many policies play instead ("You + 4 Policies", "5 Policies", "1
Policy"), and a very long name is cut with "…". Hover the line to drop down the
team's full list, **Team A Controllers**: each controller's full name, how many
champions it plays, and each of those champions with its class and Agent ID.
A cut name elsewhere (Watch Only, **Played By**) opens the same list when you
hover it. Each team sits on the side of its own
spawn end, so the scorebug matches the field; when the ends swap, the sides swap
too. Each score reads out of the winning score ("2 / 20") over a bar of 20
marks. The middle shows the tick ("Tick 91 of 300"), "First To 20" and the next
respawn wave: "Respawn Wave In 4 Ticks" means fallen champions come back after 4
more turns. Each team has five portraits: the champion's token, a small life
bar (ivory, or cinder at 30% life or less) and "ID 5", or a silver **You** on every
champion you play. Click a portrait to focus that champion, or hover it for its
card. A score and a falling champion's life bar change when the fall is shown,
not before, and a returning champion reappears when it is set down. Until then
the old score and the old portrait stay on screen, so the score never goes
blank.

With **Momentum** on (Visual Filters, HUD group, on by default), the points a
team scored in the last 10 ticks glow in its lighter color on its bar, over a
thin silver line; older points keep the team color. For example, a team at 3
that scored its third point on this turn shows two plain marks and one bright
one. The server keeps each team's score from 10 ticks back and sends it with
every turn, so the page keeps no history, and **Restart** starts the count again.
Scores are public, so Agent POV and Oracle View show the same marks. Off, every
scored point looks the same.

A Manual plan only changes its configured agent's row. The other controllers
choose from their permitted observations and masks; they do not receive any
Manual plan. The client combines the rows, advances one turn and records the
actual submitted actions. Configured dead agents keep their controllers and
memory. Unused roster slots stay empty.

Looking at a policy-controlled champion does not make it Manual. Click its
portrait or its body to look at it. The deck then shows **Watch Only** with the
champion's name, calling, Agent ID and who plays it, and **Back To Your
Warrior** (your champion's class) returns the orders to you. Watch Only covers
the deck's other columns but never **4 Lock In** or the message line: **Lock
In Turn** stays visible, and Lock In still sends every order you staged. No action
is shown before that policy chooses, and policies never see your orders. Human
Oracle View inspection is diagnostic play, not an autonomous evaluation.

**Tab** and **Shift + Tab**, like **‹ Prev** and **Next ›**, move the orders
only between the champions you command that can take orders: your living
Manual champions, in Agent ID order, wrapping around. With one Manual champion
on each team they move between those two. They skip a fallen champion and never
land on a policy's champion, so with five Manual champions five presses visit
all five. With nobody Manual (Policies Only), Tab looks at every champion in
sight instead.

Matches start paused. With no configured Manual slot, the deck becomes the
**Watching · Nobody Is Yours** bar: **Run** advances one turn at a time,
**Pause** prevents the next turn and **Play One Turn** (Space or N) plays one.
While
Run plays, the policies choose the next turn as soon as the champions of the
current turn have landed, and the next turn starts the moment the current one
ends, so play flows without a pause between turns. After Pause, the turn on
screen and a turn already requested still finish; Play One Turn and Lock In
come back once they have.
A dead Manual agent still prevents automatic Run because it can respawn. With
Manual slots, the client always waits for Lock In. Lock In also advances one
turn when every agent is policy controlled. A turn plays only once every
player's actions are in, however long a method (an LLM, for example) takes.
While it waits, the message line says **Waiting for all players to submit
their actions…**, the Lock In button says **Policies Choosing…** and the deck
offers **Cancel Policy Work**; Lock In stays disabled until that turn
finishes. After a cancel the deck shows **Stopped · A
Fresh Start Is Needed**: the game so far is kept, you may look around, **Finish
And Save** or exit, and a fresh match starts from Python. Refreshes and other
tabs read the same game.
They do not create another running loop. Run belongs to the page that pressed
it: a refresh ends it after the turn already out, so press **Run** again to
carry on.

Click the arena to turn keys on; the deck's message line then says **Keys On**
(it says **Keys Off · Click The Arena** otherwise), and pressing a deck button
keeps them on. So does pressing a top-bar button (Agent POV, Oracle View, the
Recording pill, Visual Filters, How To Play, Finish And Save, Restart), the
arena's corner keys, and the game-over Verdict's **Match Summary** and **Show
Arena** and the Match Summary's **Close**: a view switch keeps keys on, and the
Restart dialog gives them back when it closes. After Exit no key works, so the
chip says **Keys Off**. A click on the arena never pins a card: a stone's or a
mark's card shows only while the pointer rests on it. **Tab** never leaves the
arena while keys are on: while the policies choose or after the game ends it
does nothing, instead of walking onto the stones and marks. Press
**?** for How To Play over the arena, at any time, even after the game ends;
**Read The Full Guide** opens the full key list. That list is written from the
client's one command table, so it always matches what each key does. **Esc**
clears the target and turns keys off; the **×** button beside the target clears
it and keeps keys on. A right click on the arena also clears the target. **R**
(Restart) and **G** (reach rings) still work after the game ends. Policies never
receive another actor's staged same-turn action.

Hover a stone for its tip: **Pillar**, **Wall** or **Thin Wall**, then its
**Obstacle ID** (a plain number, the same in Agent POV and Oracle View), its
size in world units, **Turned** for walls, **Center** and the rule "Blocks
movement and sight. A Warrior's Charge can pass over it. Bodies never block
sight." A thin wall adds "True width 0.01 units. Drawn thicker so you can see
it." A click on the arena only turns keys on and never pins a tip: when the
click lands on a stone or another mark, its tip shows only while the pointer
rests there and closes when the pointer moves away. (The Replay Viewer, which
has no keys, pins a mark's tip on a click.) Every tip uses plain words: a Basic
says "The Priest used Lesser Heal this tick.", a life number says **Net
Damage** or **Net Healing** with its **Life Change**, your planned order's
line says **Planned Basic** or **Planned Ultimate**, and a row of the Fallen
panel ("Fallen On Tick 92") says who shares the credit for the fall.

The deck's message line says what each of your orders did, for example
"Step: North." or "Charge is armed on the Mage (ID 5).". Its first sentence
fits your roster ("You play five champions on Team A. Each keeps its own order
until you lock in."). At game over it names only what works: "Finish And Save"
while the game is unsaved, "Saving the replay…" while it saves, then "The game
is over and saved. Press Restart for a new match, or Open Replay to watch it."
A page opened after the game ended starts with that game-over sentence.
Notices written in developer words are shown in plain words,
for example "That step is not allowed right now. Your order is unchanged." The page refuses an
order the game does not allow before sending it, from the game's own masks, and
the control shakes: "Charge needs a target first. Your order is unchanged." when
no target is picked, "Shield Bash is not allowed on this target right now. Your
order is unchanged." when the picked target does not allow it, "This champion is
played by a policy. You can look, not command. Pick one of yours to give
orders." for a policy's champion, and "A fallen champion cannot act until it
returns." while yours is fallen (the deck then says when it returns).

### Startup And Turn Speed

Every method that runs in JAX runs compiled: the built-in names, saved
checkpoints and your own JAX `System`. JAX compiles the two teams' methods into
one program and the game step into another. Compiling takes a few seconds, so the
BattleClient does it once, before the page opens, by preparing the first turn
and throwing the result away. Nothing about the game changes.

After that, no Lock In compiles anything. Weights, memory, observations and
random keys are ordinary inputs to the compiled programs, so each later turn
takes about 0.02 seconds plus your method's own time. Restart reuses the same
programs. Example, measured with 7b+pool against TDM-Beta: preparing took about
4 to 7 seconds on CPU and about 12 seconds on an RTX 5090; every turn after it
took 0.02 seconds on both.

A method that runs on the host (`execution="host"`, for example an LLM) is
never called while preparing, because its cost and outside effects belong to
real turns only. It runs each turn at its own speed. Its JAX opponent is still
compiled once.

### Agent POV And Oracle View

The BattleClient opens in **Agent POV**. The arena and the BattleClient panels
show only what the viewing team sees. An enemy your team cannot see reads
**Unseen**, with no life, and a teammate's next action is not shown:
champions cannot see their teammates' plans.
An **Unseen** portrait cannot be focused; clicking it says why in the status
line.
Your own team's staged orders always show, in both views: the rows in **Your
Turn**, the set pill, Do This Now and the **Lock In** list read the game's own
staged orders, including targets kept from earlier ticks, so they match after
every tick and after a view switch.
When you play one champion on each team, Agent POV also keeps the other
team's order private: that champion's row in **Your Turn** reads **Out Of
Sight** when your team cannot see it, with nothing else about it, or **In
Sight · Order Hidden** when it can (click it to give that champion orders,
which switches the view to its team), and the set pill and the **Lock In**
list count only the viewing team's orders. **Oracle View** shows both teams' orders.
Every status display names where a status comes from by class only ("From
a Warrior."), never by Agent ID and never "a hidden source", so the words are
the same in both views: the status itself already tells you its cause's class.
That holds for the Champion Card, the status items of Frame Explanation (**F**),
the status lines of the panels and the status tips. Frame Explanation's other
lines, which say who did what this tick, name an attacker only when your team
saw it, otherwise "a hidden source". In Agent POV the floor your
living champions cannot see is darkened,
including the shadows that stones cast: stone blocks sight, champions do not.
The game computes that shape from your team's own champions and the stones,
so it reveals no enemy. Thin silver lines mark where sight reach ends: solid
for the champion you order, dashed for its teammates. Where an enemy left your
team's sight this tick, a faint dotted ring with a hollow outline of its class
emblem marks where your team last saw it; it never shows where it went.

The viewing team is the team of the Manual champion you focused last. Looking
at a policy champion or an enemy, with a click or a roster pick, never
changes it. When no champion is Manual, the view follows the champion you look
at; then Tab and Shift+Tab, in Agent POV, skip champions your team cannot see.

**Oracle View** shows everything. It is one switch away. During play a tag on
the arena's frame under the floor, just past the left home end's red strip and
its corner bracket, names the view: "Agent POV · Your Team's Sight · 2 Unseen"
(2 living champions your team cannot see; "Team A's Sight" when nobody plays by
hand) or the amber "Oracle View · You See Everything". It never covers the home
end marks or the cards that appear along the top. Its words fill that frame
but never get smaller than 10 px, so at 1366 x 768, where the frame is thinner
than that, the tag overlaps the floor's bottom edge by about 3 px. Changing
the view changes what you inspect, not what any policy receives. **Open Replay**
always opens the Replay Viewer in Oracle View.

Agent POV is a fairness aid for play, not a security boundary. The browser
still receives the global researcher data that the Replay Viewer's panels use,
and anyone with browser developer tools can read it.

### How A Tick Is Drawn

Each tick plays as a short sequence of beats: champions glide to where they
ended the tick, blows play between those spots, then fallen champions fall and
returning champions come back. Life, the score and the fall notices change at
the fall beat, not before. A returning champion's waystone stays plain until its
return beat: its token, its spawn ward, its status marks, its Command Ring and
its plan marks all arrive with it, never before. Its old corpse stays where it
fell until that beat, then fades out over 0.2 seconds while a thin ash ring
lifts off it (with Resurrection Effects on; in Agent POV only if your team saw
the corpse when the tick began). With **Longer Tick** on (the default), an ordinary
tick lasts 1.7 seconds and a tick with a death 3.2 seconds. Paused frames always
show the settled end of the tick.

A blow is drawn between the end spots of its source and target. So a stroke can
look longer than the ability's real reach. A faint imprint ring and hairline
mark the true spot where the blow was decided. A Warrior's Charge is drawn from
its start to its landing. The exact rules are in
[A45](design/specification_amendments.md#a45-runefield-client-presentation).

## Fixed Scenarios

```python
import marl_battlegrounds as marl_bgs

marl_bgs.battle_client(scenario_id=1, team_a="tdm-alpha")
```

Only Team A controllers can change. The scenario fixes its opponent, all agent
slots, living/dead state, positions, rules, effects, timers, scores and horizon.
Do not supply `team_b`, map, roster, seed or rule overrides, even if the value
matches the fixed setting. Invalid setup is rejected before loading policies.

A dead agent still occupies its original slot. Do not shorten a controller list
to the number of living agents. Methods receive the configured layout and its
masks; the LLM adapter calls only owned, valid, living actors. Policy memory starts
fresh and survives later death/respawn. Restart restores the same exact start.

The interactive run uses the first existing scenario qualification seed case.
Watching one case is not a statistical evaluation or proof of learned skill.
Use the existing exact-start evaluation API for repeated autonomous partner
comparisons; do not train or adapt on the evaluation scenarios.

## Save And Continue

Capture starts at frame zero by default. `output_dir="./recordings"` resolves
against the directory where you launched Python. It never writes into the
installed package. Use `capture_replay=False` to skip recording and its cost.

Completed games save automatically, in the background. At game over the arena
shows the Verdict card at once: the result, both scores and why the game ended.
While the file is written, the Recording pill and the top bar's save button
read **Saving Replay…**; then the pill reads **Saved** and the button **Open
Replay**. On a busy CPU (3 October 2026) a
177-tick game (14.8 MB replay) said Saved about 8 s after game over, and a
300-tick game (23.7 MB) about 13 s after, because the client writes the file,
reads it back and loads it again before it says Saved. View
switches and the help keep working meanwhile; Exit, Restart and Open Replay wait
for the save to finish. **Finish And Save** (in the top bar, and in the popover
that the **Recording** pill in the top bar opens; the popover opens over the
side rail, below the scorebug, so it never covers the scorebug or the arena)
ends an unfinished game and saves the accepted prefix. The pill then reads
Saved and the top bar's button turns into **Open Replay**. The live page stays
open for inspection. **Open Replay** (that button, or the same button in the
popover) is available once a save exists; it opens the saved game in a
separate [Replay Viewer](dev/replay_viewer.md) tab, which shows "Opening
Replay" until the viewer is ready. If your
browser blocks the new tab, use the link shown beside that button.

**Restart** starts the same matchup with fresh episode memory, frame-zero capture
and a new output file. It reuses loaded methods. A sealed replay never receives
another turn. Fixed scenarios restart from their exact original state. The old
saved file stays unchanged.

If you Restart while a recording is unsaved, choose **Save**, **Discard** or
**Cancel**. Save keeps that recording before replacing the match. Discard drops
only the unsaved recording. Cancel keeps the current match. Another tab cannot
use an old confirmation to replace a newer game. When the game is already saved,
Restart starts the fresh match at once. **R** or **Restart** pressed while a
finished game is still saving (**Saving Replay…**) asks nothing either: the
client waits for the save to end, then starts the fresh match.

If saving fails, the Recording popover says why in one line (for example
"Reason: The replay file could not be written."); use **Retry Save** or
**Save As**. Save As accepts a file name
ending in `.marlbg-replay.json`, inside the same output directory. It rejects paths
and existing files. A failed save keeps the recorded data in memory. If saving
succeeds but the next match cannot start, the saved file and accepted game remain
available; retry the replacement explicitly.

Closing a Viewer tab leaves the live game running. The Viewer's Exit button stops
only that Viewer; Open Replay can reopen it. The client keeps one separate Viewer
at a time. Opening another saved game replaces the previous Viewer. Exiting the
live client leaves an open Viewer working: the terminal prints "The live game has
closed. Its Replay Viewer is still open:" with its address, and the Python call
returns once you press the Viewer's Exit or Ctrl-C ("Replay Viewer stopped.").
Ctrl-C while the game is still open stops both. Exit and Ctrl-C try to save
unsaved accepted play before stopping; Exit or Ctrl-C before the first tick
saves nothing, because there is nothing to watch ("No tick was played, so no
replay was saved."). A browser tab closing is not Exit. After Exit the page says
the match is closed and what happened to the replay: "The match is closed. The
game so far is saved. You can close this tab.", "The match is closed. No tick
was played, so no replay was saved. You can close this tab.", or, when nothing
was recorded, only that the match is closed. The connection pill then reads
**Closed**, the Recording pill and the seal read **Saved** (or **No Replay**),
the Your Turn card and its pill read **Closed** with no phase lit, and the Keys
chip reads **Keys Off**. Close the tab, or start a fresh launch from Python. An
Exit pressed just as a finished game's save ends is sent once more by the page,
so one press is always enough.

The saved replay keeps every controller's name, one per champion and one per
team, exactly as the scorebug showed them. The Replay Viewer shows that same
list in its scorebug, its **Team A Controllers** list and its Team Rosters
strip; a champion you played reads **Manual** there, since a replay has no
"You" (the short line reads "Manual + 4 Policies"). A `marl_bgs.team` System
passed as one controller is listed by member,
here and in the replay.

Method call records are separate from the verified saved replay. The saved-prefix
record identifies which accepted turns reached the saved file. Cancelled and
stale method answers never become recorded turns. A played answer alone does not
prove that a replay was written successfully.

## External Methods And Failures

Direct Python objects run on the launch caller's thread and context. The runner
uses their declared resource scopes and does not copy, serialize or close your
supplied clients. Keep their parameters and external behavior fixed while the
blocking launch is running. Restart creates fresh episode memory and reuses the
loaded methods. An LLM uses the [existing LLM API and local recipe](llm/README.md#play-in-battleclient).
No model server is started automatically.

The browser remains responsive during a method call. A persistent Fresh Launch
Required message disables further play after an uncertain method failure. You
can still inspect the accepted game and save it. Cancellation rejects a late
answer but cannot forcibly stop arbitrary Python. Existing provider deadlines
still apply. Cancelled System work requires a fresh Python launch before further
play; the last accepted game remains available for inspection and saving. The
client uses the same rule if a host method changes its own state and then fails.
The client does not pretend it can undo arbitrary external state.

BattleClient's setup is supplied by Python. To choose a different map or method,
change the launch arguments. Local authoring tools are private; the public package
contains no editor routes, editor files or fallback to a developer checkout.

The [client qualification](dev/client_qualification.md) records tested workflows,
installation isolation, measured costs and the limits of the evidence.
