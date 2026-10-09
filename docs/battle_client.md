# BattleClient

Play a live match in your browser against, or beside, any method: built-in bots, saved models, your own Systems or LLMs.

Website: https://marl-battlegrounds.com/#battleclient

## Launch

```python
import marl_battlegrounds as marl_bgs

marl_bgs.battle_client(map_id=0, team_a="manual", team_b="tdm-alpha")
```

Python runs the match; your browser shows it. The call compiles the first turn once
("preparing the first turn (compiling once)"), opens your default browser at a local
address and waits until you press Exit or Ctrl-C. The page ships in the package: no Node or Docker.

**Watch Out:** closing the tab leaves the game running; open the printed address again
(`open_browser=False` only prints it). Use a desktop browser: Firefox is recommended.

| Setting | Default | What It Does |
| --- | --- | --- |
| `map_id` or `scenario_id` | Required | Exactly one: a map from 0 to 51, or a scenario from 1 to 8 |
| `team_a`, `team_b` | `"manual"` | One controller per team, or a list with one per roster slot |
| `team_a_roster`, `team_b_roster` | Mage, Warrior, Hunter, Rogue, Priest | One to five classes per team, repeats allowed |
| `seed`, `max_steps`, `score_threshold`, `red_zone_depth` | `0`, `300`, `20`, `5.0` | The ordinary game rules |
| `spawn_mode` | `"default"` | `"swapped"` puts Team A on the right |
| `port`, `replay_port` | `0` | Local ports for the game and for the Replay Viewer that Open Replay starts; `0` picks a free one |
| `output_dir`, `capture_replay` | `"./recordings"`, `True` | Where replays go, relative to your launch folder; `False` skips recording and its cost |

## Controllers Per Slot

```python
from marl_battlegrounds.training import load_system

mine = load_system("checkpoints/my_actor")  # a saved actor export or full checkpoint folder
marl_bgs.battle_client(map_id=0, team_a=["manual", mine, mine, mine, mine], team_b="tdm-beta")
```

A controller is `"manual"`, a built-in name (`"random"`, `"tdm-alpha"`, `"tdm-beta"`,
`"tdm-gamma"`), a saved model folder (needs the `training` extra), your own `Policy` or
`System`, or a trusted `"module:function"` factory. Repeated entries load once; each team
keeps its own memory. Policies choose from their own inputs and never see your orders.

**Watch Out:** name a folder that looks like a built-in as `Path("my_actor")` or
`"./my_actor"`, without symbolic links. Loading never changes a roster.

## Play A Turn

```python
marl_bgs.battle_client(map_id=47, team_a="tdm-beta", team_b="tdm-alpha")  # no Manual slot: watch
```

Click your champion, choose a step, a target and an action, then press Lock In Turn. Plan
several Manual champions first if you like; then the policies choose and everyone acts at
once. With no Manual slot, Run plays turn after turn. Press ? in the page for every key.

| Key | What It Does |
| --- | --- |
| W A S D, Q E Z C, X | Step north, west, south, east, northwest, northeast, southwest, southeast; X stays |
| 0, 1, 2 | No combat, Basic, Ultimate |
| Shift + click, Esc | Pick a target; Esc clears it and turns keys off (click the arena to turn them on) |
| Space or Enter (N: one tick) | Lock in the turn; with no Manual slot, play one turn |
| Tab, Shift + Tab | Your next or previous living Manual champion |
| V, F, G, R | Everyone in view, frame explanation, ranges, restart |

**Watch Out:** the page opens in Agent POV (only what your team sees); Oracle View shows all.
Views never change what a policy receives, and Agent POV is a fairness aid, not a security boundary.

## Play From Either End

```python
marl_bgs.battle_client(map_id=47, team_b="tdm-alpha")                        # Team A on the left
marl_bgs.battle_client(map_id=47, team_b="tdm-alpha", spawn_mode="swapped")  # Team A on the right
```

Each call waits for Exit, so these play one game after the other. You stay Team A; the
ends, pads and Red Zones swap. To compare ends fairly, keep the map, rosters, seed and
rules the same. The replay records which ends you played.

**Watch Out:** `spawn_mode="paired"` is rejected: a live game has one pair of ends.

## Fixed Scenarios

```python
marl_bgs.battle_client(scenario_id=5, team_a="tdm-alpha")
```

Only Team A's controllers change. The scenario fixes the opponent, positions, health,
scores, timers and rules; Restart returns to that exact start. A fallen champion keeps
its slot, so a list needs all five entries.

**Watch Out:** passing `team_b`, a map, rosters, a seed, a rule or `spawn_mode` is rejected,
even with the fixed value. One game proves nothing; use `evaluate_scenarios` ([evaluation.md](evaluation.md)).

## Save And Continue

```bash
python -m marl_battlegrounds replay recordings/GAME.marlbg-replay.json
```

A finished game saves itself in the background; Finish And Save keeps an unfinished
game's turns. Open Replay shows it in the Replay Viewer, and the command above reopens any
saved file. After Open Replay, Exit closes the game only: the call keeps serving the Replay
Viewer until you exit it too or press Ctrl+C. Restart starts the same match with fresh memory and a new file. To measure a
saved game, call `marl_bgs.replay_metrics(path, metrics="full")` ([metrics.md](metrics.md)).

**Watch Out:** if saving fails, use Retry Save, or Save As with a new `.marlbg-replay.json`
name in the same folder. A slow method shows Cancel Policy Work; after a cancel or a failed
method, save the game, exit and launch again (the client cannot undo your Python's effects).

## The Example Scripts

From A Checkout (examples are not in the installed package):

```bash
python examples/prepare_battle_client.py --output actors --seed 17
python examples/battle_client.py simple --policy-a actors/mappo --policy-b actors/qmix
```

[prepare_battle_client.py](../examples/prepare_battle_client.py) writes six untrained actor
folders, one per built-in learner: they test loading, not skill. [battle_client.py](../examples/battle_client.py)
has seven modes, from `simple` to `scenario-partners`; `--help` lists them all. The `hard`
and `scenario-llm` modes need a model server you start yourself ([llm.md](llm.md)).

## Next

- [replays.md](replays.md): watch saved games in the Replay Viewer.
- [your_method.md](your_method.md): build a System to play against.
- [performance.md](performance.md): how long startup and each turn take.
