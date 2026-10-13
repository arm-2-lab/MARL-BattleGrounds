# Replays

Record chosen games, watch them tick by tick in the analytical Replay Viewer and read the saved files.

Website: https://marl-battlegrounds.com/#record

## Record a Replay

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation import load_replay

result = marl_bgs.evaluate(
    "tdm-alpha", "tdm-beta", num_episodes=2, maps=[47], num_envs=2, max_steps=40,
    save_replays=1, output_dir="runs/replays",
)
loaded = load_replay(result.replay_paths[0])
print(result.replay_paths[0], len(loaded.replay.frames), len(loaded.replay.transitions))
```

A replay is one saved game: every tick, every agent's view and every event.
`save_replays=1` records the first game into the run's `replays/` folder.
`load_replay` checks the file and reads it without starting a game, policy or JAX.

**Watch Out:** without `output_dir` a replay stays in memory and has no file to open.

## Choose Which Games to Record

```python
result = marl_bgs.evaluate(
    "tdm-alpha", "tdm-beta", num_episodes=32, replay_episodes=[1, 7], output_dir="runs/replays"
)
env = marl_bgs.make("tdm", map_id=0, num_envs=128, replay_episodes=range(49_951, 50_001))
```

`replay_episodes` picks exact game IDs; `save_replays=N` takes the first N games.
`evaluate`, `run_tournament` and `run_canonical_tournament` take both; `make` takes
`replay_episodes`. Replays and metrics are separate choices: recording never
needs full metrics, and full metrics never need a replay.

**Watch Out:** selections are game IDs, not a frequency. Record no replays while
training; record the checkpoints you care about later, in a separate evaluation.
Recording never changes which games are played ([Same Seed, Same Game](evaluation.md#same-seed-same-game)).

## Open the Replay Viewer

```bash
python -m marl_battlegrounds replay PATH/TO/game.marlbg-replay.json
```

The Replay Viewer checks the file, prints a local address and opens your browser.
It runs only on your computer and needs no source checkout or Node.js. It only
reads: it never changes the file, runs a policy or plays new ticks. Press Ctrl-C
in the terminal to stop it.

**Watch Out:** Agent POV hides what one agent could not see, but the file itself
holds the whole battle (Oracle view). Watching a game is a diagnostic, not a result.

| Setting | Default | What It Does |
| --- | --- | --- |
| `path` | Required | The saved `.marlbg-replay.json` file |
| `--view oracle` or `--view pov` | `oracle` | Start in Oracle view or in one agent's POV |
| `--pov-slot` | The evaluated team's lowest slot | Whose view Agent POV shows: slot 0-4 Team A, 5-9 Team B |
| `--frame-index` | 0 | The recorded frame to start on |
| `--ranges` or `--no-ranges` | Off | Show ability ranges |
| `--port` | 0 | Local port; 0 picks a free one |
| `--no-open` | Off | Print the address without opening a browser |
| `--static` | Off | Draw one frame with Matplotlib instead (see below) |

## Still Frames

```bash
python -m marl_battlegrounds replay game.marlbg-replay.json --static --frame-index 0
```

`--static` draws one recorded frame in a Matplotlib window; it needs the `viz`
extra. For a paper figure, pause in the Replay Viewer and choose **Export**, then
**Export PNG**: it saves the settled still of that tick.

**Watch Out:** `--static` needs `--frame-index`. Only `--ranges` may join them;
browser options such as `--port` or `--no-open` are refused.

## The Replay File

```python
replay = loaded.replay  # from load_replay above
print(replay.schema_version, replay.completion.completion_state)
```

The current format is Replay V4 (`ReplayArtifactV4`): one strict JSON file per game.
A game of T ticks has a header, T + 1 frames, T transitions and a completion record.
Frame `k` is the game just before tick `k`; transition `k` is what that tick did.
V4 records the Red Zone depth; V1, V2 and V3 files load with their own rules.

**Watch Out:** a replay is evidence of a game, not a training checkpoint: it has no
optimizer state, network outputs, recurrent memory or random keys. Loading refuses
symlinks, duplicate keys, NaN, unknown fields, wrong digests and files over 1 GiB.
Saving never overwrites a file.

| Part | What It Holds |
| --- | --- |
| Header | Game settings, roster, information rules, who controlled each team and agent, seeds and machine details |
| Frames | All ten agents' observations, legal-action masks and a world snapshot |
| Transitions | Submitted and accepted actions, rewards, typed events and rule facts |
| Completion | Complete, or a partial, interrupted or failed recording, and why it ended |
| File name | Map, game ID, seed, random stream, Team A, Team B and the content digest: `tdm_map_id_47_teutoburg_forest_test__episode_1__seed_0__stream_1__a_tdm_alpha__b_tdm_beta__<digest>.marlbg-replay.json`. Labels shorten only to fit; the digest stays |

## Sample Replays

```bash
python -m marl_battlegrounds replay examples/replays/v1/mirrored-five-class-ultimates.marlbg-replay.json
```

**From a Checkout:** [`examples/replays/v1/`](../examples/replays/v1/) holds three short scripted V1 games, each with
a `.marlbg-metrics.json` report. They show mechanics; they are not benchmark results.

| File | What It Shows | Frames |
| --- | --- | --- |
| `death-respawn-shield.marlbg-replay.json` | A fatal hit, the respawn wave and the spawn shield | 8 |
| `recovery-status-lifecycle.marlbg-replay.json` | Health recovery, a refused action, and statuses refreshed, broken and expiring | 8 |
| `mirrored-five-class-ultimates.marlbg-replay.json` | Every class's Ultimate, used by both teams | 6 |

## Next

- [Metrics](metrics.md): measure any saved game with `replay_metrics`.
- [Evaluation](evaluation.md): play and save the games worth watching.
- [BattleClient](battle_client.md): play a game yourself and open its replay.
