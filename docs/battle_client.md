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

The call opens a browser and prints its local URL. If the browser cannot open,
copy that URL into your browser. The call waits until you use Exit or press
Ctrl-C in its terminal. Closing a tab leaves Python running; reopen the printed
URL to return. `open_browser=False` prints the URL without opening it.

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
each configured agent's controller. Choose a Manual agent,
stage its movement and combat choice, then Submit. You may stage several Manual
agents before Submit; changing focus preserves the other drafts. A joint turn
uses all submitted Manual rows and the policies' actions once.

A Manual draft only changes its configured agent's row. The other controllers
choose from their permitted observations and masks; they do not receive any
Manual draft. The client combines the rows, advances one turn and records the
actual submitted actions. Configured dead agents keep their controllers and
memory. Unused roster slots stay empty.

Inspecting a policy-controlled agent does not make it Manual. No action is
highlighted before that policy chooses. Current Policy Target Masks lists its
allowed targets without changing its action. Oracle and Agent
POV change what you inspect, not what a policy receives. Human Oracle inspection
is diagnostic play, not an autonomous evaluation.

Matches start paused. With no configured Manual slot, Run advances one turn at
a time and Pause prevents the next turn. A turn already in progress may finish.
A dead Manual agent still prevents automatic Run because it can respawn. With
Manual slots, the client always waits for Submit. Submit also advances one turn
when every agent is policy controlled. A slow method shows Choosing Actions;
Submit stays disabled until that turn finishes. Refreshes and other tabs read
the same game. They do not create another running loop.

Use the on-screen help for the existing keyboard and pointer controls. Escape
clears the target; right-click retains normal browser behavior. Policies never
receive another actor's staged same-turn action.

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

Completed games save automatically. **Finish And Save** ends an unfinished
game and saves the accepted prefix. The live page stays open for inspection.
**Open Replay** opens the saved game in a separate [Replay Viewer](dev/replay_viewer.md)
tab. If your browser blocks the new tab, use the link shown beside that button.

**Restart** starts the same matchup with fresh episode memory, frame-zero capture
and a new output file. It reuses loaded methods. A sealed replay never receives
another turn. Fixed scenarios restart from their exact original state. The old
saved file stays unchanged.

If you Restart while a recording is unsaved, choose **Save**, **Discard** or
**Cancel**. Save keeps that recording before replacing the match. Discard drops
only the unsaved recording. Cancel keeps the current match. Another tab cannot
use an old confirmation to replace a newer game.

If saving fails, use **Retry Save** or **Save As**. Save As accepts a file name
ending in `.marlbg-replay.json`, inside the same output directory. It rejects paths
and existing files. A failed save keeps the recorded data in memory. If saving
succeeds but the next match cannot start, the saved file and accepted game remain
available; retry the replacement explicitly.

Closing a Viewer tab leaves the live game running. The Viewer's Exit button stops
only that Viewer; Open Replay can reopen it. The client keeps one separate Viewer
at a time. Opening another saved game replaces the previous Viewer. Exiting the
live client also closes its Viewer. Exit and Ctrl-C try to save unsaved accepted
play before stopping; a browser tab closing is not Exit.

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
