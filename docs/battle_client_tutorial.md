# Play Your Saved Methods

Use one small Python script to play saved methods, mix controllers and compare
partners from a fixed scenario start. Download
[`battle_client.py`](../examples/battle_client.py) into a working folder of your
choice. It works outside the checkout with an installed package. Pip installs
the package, not this example file.

The script has seven named modes. Each mode has a short, direct
`marl_bgs.battle_client(...)` call, except `scenario-partners`, which calls the
existing evaluator. You can edit those calls to use your own live `System`
objects. No private launcher or development files are needed.

## Install And Prepare Actors

Use Python 3.14. Put the current supplied wheel in your working folder, then run:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install "./marl_battlegrounds-0.0.0-py3-none-any.whl[training]"
python battle_client.py --help
```

This installs the base package and its native-checkpoint dependencies. These
commands use a local wheel; they do not assume a package-index release exists.
For another release, use its actual wheel name. Browser files, maps and scenarios
are included. You need no Node, frontend build or hosted site. The separate
[base installation](battle_client.md#install-and-open) is enough for Manual and
built-in policies, but every mode in this script loads native saved actors.

For trained methods, follow the existing
[train, save and load workflow](training/README.md#train-resume-load-and-analyze).
Use the real path returned as `result.selected_actor or result.final_actor`, or
use `result.final_checkpoint` for the complete learner save. An actor export is
smaller and contains the deployment weights and their identity. A full checkpoint
also holds training state; `load_system` validates it and restores only the actor
for play. Keep each folder intact. Do not pass a run folder, make up a checkpoint
ID, rename another format to look native or infer a method from its file name.
For your own learner, use the existing
[`export_system`](../src/marl_battlegrounds/training/checkpoints.py) contract or
load your own format and pass a `System` in Python.

If you only want to try the software, also download
[`prepare_battle_client.py`](../examples/prepare_battle_client.py). This optional
helper creates six real actor exports with random initial weights:

```bash
JAX_PLATFORMS=cpu python prepare_battle_client.py --output actors --seed 17
```

Choose a new output directory. The command writes `actors/mappo`, `actors/ippo`,
`actors/ff_mappo`, `actors/ff_ippo`, `actors/qmix` and `actors/pqn_vdn`.
`actors/preparation.json` records their identities, zero trained steps and setup
costs. These actors have never learned. They check loading and composition;
they do not show playing skill, sample efficiency or useful coordination.
The helper downloads no weights and launches no training.

The commands below use those six folders. Replace paths with your own real
artifacts to inspect learned behavior. Paths may be anywhere. Quote paths that
contain spaces. The loader keeps the saved input scale, spawn frame and parameter
sharing; loading does not make a model compatible with a different roster.

## Simple: One Human And Two Saved Policies

```bash
JAX_PLATFORMS=cpu python battle_client.py simple \
  --policy-a actors/mappo --policy-b actors/qmix
```

Both teams use the canonical order: Mage, Warrior, Hunter, Rogue, Priest.
Slots are numbered from zero.

| Team | Slots | Controller |
| --- | --- | --- |
| A | 0 | Manual |
| A | 1–4 | The same saved MAPPO actor |
| B | 0–4 | A different saved QMIX actor |

Each source loads once. Repeated slots reuse it; actor and team memories stay
separate. The main call is easy to change:

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds.training import load_system

policy_a = load_system("actors/mappo")
policy_b = load_system("actors/qmix")
marl_bgs.battle_client(
    map_id=0,
    team_a=["manual", policy_a, policy_a, policy_a, policy_a],
    team_b=policy_b,
)
```

The script starts on ordinary map 0 with the public defaults: seed 0, 300 steps,
score threshold 20 and Red Zone depth 5.0. `--map-id` chooses another ordinary map.
An explicit `--max-steps N` changes its horizon; there is no hidden short-game
default. `JAX_PLATFORMS=cpu` keeps these small live examples on CPU.

## Medium: Two Saved Policies Per Team

```bash
JAX_PLATFORMS=cpu python battle_client.py medium \
  --a-first actors/mappo --a-second actors/ippo \
  --b-first actors/ff_mappo --b-second actors/ff_ippo
```

| Team | Slots 0–2 | Slots 3–4 |
| --- | --- | --- |
| A | MAPPO | IPPO |
| B | FF-MAPPO | FF-IPPO |

Supply four different saved sources. All slots are automatic. Start with Lock In
Turn to inspect one joint turn, or use Run and Pause. The game still starts paused.

## Hard: Ten Different Controllers

First prepare a separate real model service using the
[pinned local Qwen recipe](llm/qwen_recipe.md#start-the-model-server). Verify that
the service loaded the stated revision. A model name or `--model-revision` value
is a declaration, not an inspection of its weights. Wait until that service is
ready, then run:

```bash
JAX_PLATFORMS=cpu python battle_client.py hard \
  --mappo actors/mappo --ippo actors/ippo \
  --ff-mappo actors/ff_mappo --ff-ippo actors/ff_ippo \
  --qmix actors/qmix --pqn-vdn actors/pqn_vdn \
  --model qwen3.5-4b-fp8 \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --server-url http://127.0.0.1:8000/v1
```

| Slot | Class | Team A | Team B |
| --- | --- | --- | --- |
| 0 | Mage | Manual | QMIX |
| 1 | Warrior | Recurrent MAPPO | Recurrent PQN-VDN |
| 2 | Hunter | Recurrent IPPO | Our LLM adapter |
| 3 | Rogue | Feedforward MAPPO | Custom JAX System |
| 4 | Priest | Feedforward IPPO | Custom Host System |

The custom JAX method samples legal actions from each actor's own input. The
host method chooses the first legal move and compatible target/Ultimate pair.
These small teaching methods claim no skill. Both return three `int32` action
heads of shape `(B, 5)`, where B is the number of games. They respect active slots,
owned slots and valid games, keep rows separate and return their empty memory
unchanged. Read their functions in the downloaded file, then use the
[System input contract](evaluation/workflows.md#policy-inputs) for your method.

The LLM uses our existing `llm.make_system` and a caller-owned `llm.Client`.
It gets only its owned actor's permitted input. History is off, the reply limit
is 64 tokens, the client allows one request at a time, each request has a
60-second deadline and there is no retry. Failed replies stop play rather than
silently substituting another action. Records default to `light`; add
`--llm-records full` to retain exact requests as well. Every accepted turn may
make a real model request and incur its normal service cost. Nothing in this
script starts a server or downloads model weights. Exit closes its HTTP client;
stop your separate model server yourself when finished.

## Smaller Teams: Two Against Three

```bash
JAX_PLATFORMS=cpu python battle_client.py small \
  --policy-a actors/mappo --policy-b actors/qmix
```

Team A is Mage/Priest with Manual/MAPPO. Team B is Rogue/Rogue/Priest with QMIX
in all three slots. The script passes both rosters explicitly. Controller lists
follow each roster's order; unused slots stay empty. Repeated classes are allowed.
Use compatible artifacts, such as the shared actors from the optional helper.

## Fixed Scenario: Keep All Five Slots

Scenario 5 begins with Team A slots **0, 1 and 4 alive**. Slots 2 and 3 are dead,
but remain configured. The two interactive examples put different methods on the
three living slots without moving or reviving anyone.

For human-assisted play:

```bash
JAX_PLATFORMS=cpu python battle_client.py scenario-manual \
  --qmix actors/qmix --mappo actors/mappo
```

For a separate autonomous run with the same real model service:

```bash
JAX_PLATFORMS=cpu python battle_client.py scenario-llm \
  --qmix actors/qmix --mappo actors/mappo \
  --model qwen3.5-4b-fp8 \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --server-url http://127.0.0.1:8000/v1
```

| Team A Slot | Start | Human-Assisted Run | Autonomous Run |
| --- | --- | --- | --- |
| 0 | Alive | Manual | Our LLM |
| 1 | Alive | QMIX | QMIX |
| 2 | Dead | MAPPO | MAPPO |
| 3 | Dead | MAPPO | MAPPO |
| 4 | Alive | MAPPO | MAPPO |

Both calls pass `scenario_id=5` and only Team A controllers. The packaged scenario
owns Team B, its roster, all positions, scores, effects, timers, seed and rules.
It starts at step 295 with five steps left. The script accepts no scenario map,
roster, seed or horizon override. Restart restores that complete start. Dead
actors keep their original slots and memory; the LLM adapter skips dead actors.
A dead configured Manual slot still prevents Run because it can respawn.

Human-assisted play, including Oracle inspection, is useful for trying an idea.
It is not an autonomous zero-shot coordination result. The separate LLM run has
no Manual controller and can use Run, but one short case still proves no skill.

## Compare Autonomous Partners At The Same Start

```bash
JAX_PLATFORMS=cpu python battle_client.py scenario-partners \
  --focal actors/qmix \
  --partner-a actors/mappo --partner-b actors/ippo \
  --exposure-a unknown --exposure-b unknown \
  --output-dir runs/scenario-partners
```

This opens no browser and makes no LLM request. The focal owns Team A slots 0–3;
each partner owns slot 4. Thus the living focal slots are 0 and 1, and the living
partner slot is 4. `team()` combines them. `evaluate_episodes()` runs two games
with `EpisodeSpec(initial_state=...)`, retaining the complete scenario state,
config, fixed opponent and first shared qualification seed case. Both use the
same episode number and random coordinates. The scenario horizon is unchanged.

The command prints episode tables, run directories and replay paths. Each
partner gets a separate normal evaluator run. Its saved episode metadata retains
the full scenario specification, physical slots and declared exposure; the normal
method records identify the actual loaded sources. Repeating the command creates
new runs, so it does not overwrite earlier results.

For a real held-out comparison, declare the partner's exposure to the focal
before evaluation. Use `--exposure-a familiar --exposure-b held_out` only when
training records support those statements. Keep `unknown` when they do not.
The zero-training helper gives no learned focal to assess. A label cannot prove
held-out exposure or coordination, and two short games cannot establish a useful
win rate. Keep evaluation scenarios out of training and method selection. See
[the broader cross-play workflow](training/README.md#cross-play-and-zero-shot-coordination)
for a declared study with more partners, games and training seeds.

## Lock In, Save, View And Restart

All interactive modes first prepare the first turn: they print "preparing the
first turn (compiling once)" and compile your JAX methods and the game step,
which takes a few seconds (about 12 seconds on an RTX 5090 for 7b+pool). Then
they print a local URL and normally open it, and no Lock In waits for compiling
again: later turns take about 0.02 seconds plus your method's own time. An LLM
or other host method is never called while preparing. See
[Startup And Turn Speed](battle_client.md#startup-and-turn-speed). If a browser
does not open, paste that URL into one. Add `--no-open` to print it only;
`--port 0` chooses a free port. Python waits until Exit or Ctrl-C. Closing a
browser tab leaves the game running, so you can reopen its URL.

1. Select a Manual champion. In the Tactics Turn deck, pick its step, target and
   action, then press Lock In Turn (Space or Enter). You may plan several Manual
   champions first. The policies then choose once from their own permitted
   inputs; they do not see your plans. Selecting a policy champion only inspects
   it. Press ? for How To Play and the full key list.
2. With no configured Manual controller, Lock In Turn advances one turn and
   Run/Pause controls repeated turns. While a method is still working, the deck
   says the policies are choosing and offers Cancel Policy Work. Pause stops the
   next turn; an in-progress turn can finish.
3. The page opens in Agent POV: you see only what the viewing team sees, and
   unseen enemies read Unseen. Switch to Oracle View to see everything; the
   screen marks it with an amber tag during play. Policies receive the same
   inputs in either view.
4. Use Finish And Save to keep an unfinished game's accepted turns. A completed
   game saves automatically. Capture includes the initial frame. The script writes
   under `recordings/MODE` by default; `--output-dir /chosen/folder` changes it.
5. Use Open Replay to view the saved file in a separate Viewer. It opens in
   Oracle View. If the new tab is blocked, use the displayed link. The Viewer reads saved actions without calling
   the policies or LLM. Its Exit closes only that Viewer.
6. Use Restart for fresh episode memory and a new recording with the same loaded
   methods. If the current recording is unsaved, choose Save, Discard or Cancel.
   Already saved files stay unchanged. Exit or Ctrl-C tries to save accepted play
   before stopping the live client and its Viewer.

You can later open any printed saved replay path from the installed environment:

```bash
python -m marl_battlegrounds replay /path/to/game.marlbg-replay.json
```

If saving fails, use Retry Save or Save As. Save As accepts a new file name in
the same output folder, ending in `.marlbg-replay.json`. If a method fails or is
cancelled and the page says Fresh Launch Required, inspect and save the accepted
game, then exit and start Python again. A rejected or late answer is not a played
turn. The client cannot undo arbitrary state changes inside an external method.
The [BattleClient guide](battle_client.md) gives the full control, recording and
failure contracts; the [Replay Viewer guide](dev/replay_viewer.md) covers playback.
