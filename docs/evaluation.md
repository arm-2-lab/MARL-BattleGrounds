# Evaluation

Play your frozen team against any opponent and read who won, by how much and why.

Website: https://marl-battlegrounds.com/#evaluate

## Evaluate A System

```python
import marl_battlegrounds as marl_bgs

result = marl_bgs.evaluate(
    "tdm-alpha", "tdm-beta", num_episodes=4, maps=[47], num_envs=4, max_steps=60
)
print(result.head_to_head())
```

`evaluate` plays the first method (your team) against the second (the opponent).
`num_episodes` is the total number of games. `head_to_head` gives games, wins,
draws, losses and the mean point margin (your points minus theirs), both ways.
[`evaluation.py`](../examples/evaluation.py) and [`evaluation_results.py`](../examples/evaluation_results.py) run these calls end to end.

**Watch Out:** no `output_dir`, no files. The results then live only in memory.

| Setting | Default | What It Does |
| --- | --- | --- |
| `maps` | Test maps 47-51 | Map IDs in order; repeats allowed |
| `spawn_mode` | `"paired"` | Games in pairs with the spawn ends exchanged; needs an even total |
| `system_roster`, `opponent_roster` | Mage, Warrior, Hunter, Rogue, Priest | Classes in slot order, 1 to 5 agents, repeats allowed |
| `num_envs` | 128 | Games at once; never changes how many games are played |
| `metrics`, `save_replays` | `"priority"`, 0 | Measurement level ([Metrics](metrics.md)); record the first N games ([Replays](replays.md)) |
| `score_threshold`, `max_steps`, `red_zone_depth` | 20, 300, 5.0 | Points to win; ticks before a draw; Red Zone depth, `0.0` for off ([Game Rules](game_rules.md)) |
| `seed` | 0 | Root of every random number in the run |

## Team A, Spawn Pairs And Map Splits

```python
result = marl_bgs.evaluate(
    "tdm-alpha", "tdm-beta", num_episodes=10, phase="validation", pass_id="checkpoint-1"
)
print(result.table("episodes")["system_game_score"].mean())
```

Your method is always Team A; only the spawn ends move. A pair's two games share
the map and seed and exchange the spawn ends, so no team keeps the easier end.
With no `maps`, `phase="validation"` plays maps 42-46 and the default
`phase="evaluation"` plays test maps 47-51. Use one `pass_id` per checkpoint.

**Watch Out:** pick checkpoints on validation maps only; test maps give the final
answer once. An odd total fails with paired spawns (`spawn_mode="default"` plays
one end only). Other phase names need explicit `maps`.

## Evaluate A Saved Model

```bash
python examples/evaluate_recurrent_mappo.py checkpoint=RUN/checkpoints/CHECKPOINT_ID \
  evaluation.output_dir=runs/evaluation
```

```python
from marl_battlegrounds import training

run = training.load_run("runs/training/RUN")
actor = training.load_system(run.selected_actor)
result = marl_bgs.evaluate(actor, "tdm-alpha", num_episodes=32, output_dir="runs/evaluation")
```

The Hydra script plays 32 paired games against `tdm-alpha` on the test maps with
priority metrics; `evaluation=full` adds full metrics and replays of games 1 and 2.
`run.selected_actor` is the checkpoint the trainer chose on validation maps. The
command line (`--system`, `--opponent`), `team`, `pool` and `run_tournament` take every reference below.

**From A Checkout:** the six `evaluate_<method>.py` scripts live in `examples/` ([Hydra](hydra.md)).

**Watch Out:** `checkpoint=` needs an actor export or checkpoint folder, not a run folder.
`run.selected_actor` is None without validation; then pick `run.final_actor` on purpose.
In Python, `evaluate` reads only built-in names from text; use `load_method` for the rest.

| Reference | Example | What `load_method` Returns |
| --- | --- | --- |
| A built-in name | `"random"`, `"tdm-alpha"`, `"tdm-beta"`, `"tdm-gamma"` | Random legal actions or a scripted opponent |
| A saved actor | `"runs/my-run/checkpoints/CHECKPOINT_ID"` | The trained actor (needs the `training` extra) |
| A factory | `"my_module:load_my_system"` | What your trusted installed function returns, called once with no arguments |

## Mixed Teams And Opponent Pools

```python
researcher = marl_bgs.team("random", "tdm-alpha", slots=[[0, 1], [2, 3, 4]])
opponents = marl_bgs.pool({"random": 0.25, "tdm-beta": 0.75})
result = marl_bgs.evaluate(researcher, opponents, num_episodes=2, output_dir="runs/mixtures")
members = result.table("policy_assignments")
print(members["episode_id"], members["global_slot"], members["member"])
```

`team` gives each Team A slot one owner; slots are numbers 0-4 or class names such
as `"priest"`. `pool` draws one member when each game starts and keeps it until
that game ends. Both return an ordinary System. Owning a slot grants no extra
information: each actor still sees only its own permitted view.

**Watch Out:** members are recorded only in saved runs. A pool can field different
members in a spawn pair's two games; evaluate each member alone for paired results.

## Cross-Play With New Partners

```python
focal = marl_bgs.load_method("RUN/checkpoints/CHECKPOINT_ID")
for partner in ("tdm-alpha", "random"):
    mixed = marl_bgs.team(focal, partner, slots=[[0, 1, 2, 3], [4]])
    result = marl_bgs.evaluate(mixed, "tdm-beta", num_episodes=10, maps=[47])
    print(partner, result.point_margin_summary()["mean_point_margin"])
```

Your trained (focal) team plays slots 0-3 and a partner it may never have met plays
slot 4. [`cross_play_and_zsc.py`](../examples/cross_play_and_zsc.py) plays every
focal, partner and opponent combination with paired spawns and saves `games.csv`.

**Watch Out:** its "familiar" and "held out" labels are what you declare, not a
check of training history. A few short games do not prove zero-shot coordination.

## Scenario Tests

```python
(result,) = marl_bgs.evaluate_scenarios("tdm-alpha", scenario_ids=[1], num_envs=2)
print(result.table("episodes"))
```

Eight hand-made end-game puzzles start from exact states: positions, health,
scores, cooldowns and the tick. Your method is Team A against the scenario's own
opponent unless you pass `opponent=`. Each trial starts fresh from that state.
`load_tdm_scenario(1).notes` explains one; `evaluate_episodes` plays your own starts.

**Watch Out:** never train or tune on scenarios. Keep dead agents in their slots:
every scenario has five slots per team and several agents start dead.

## Results And Resume

```python
saved = marl_bgs.load_results("runs/evaluation/RUN_FOLDER")
print(saved.status, saved.head_to_head(by="map"))
print(saved.point_margin_summary())
resumed = marl_bgs.evaluate("tdm-alpha", "tdm-beta", num_episodes=32, resume_from=saved.run_dir)
```

Each `output_dir` call makes a new run folder inside it; `result.run_dir` names it.
`load_results` opens that exact folder without JAX and never plays or repairs
anything. Its tables: `episodes`, `priority_metrics`, `full_metrics`, `policy_assignments`.
`point_margin_summary` adds a 95% interval that keeps spawn pairs whole.

**Watch Out:** resume skips finished games and reuses omitted settings, but a
repeated setting must match: `seed=0` fails against a saved seed of 42. Blank
cells mean "not measured", never zero.

## Same Seed, Same Game

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true python -m marl_battlegrounds evaluate \
  --system tdm-alpha --opponent tdm-beta --episodes 32 --output-dir runs/exact
```

A GPU may add numbers in another order, which can flip a near-tie between actions.
In a 7,200-game test on one RTX 5090, 4.8% to 9.3% of games differed between
normal runs and median Elo moved about 2 points, less than a new seed moves it.
Recording replays or metrics never changes which games are played.

**Watch Out:** set the flag before JAX starts; `load_results(...).execution_settings`
shows whether it was set. To watch a run's exact games, record them in that run.

| Where It Runs | Same Seed And Settings Give |
| --- | --- |
| CPU | The same games every time |
| GPU in XLA's deterministic mode | The same games every time, on the same GPU model, JAX and CUDA versions |
| GPU in normal mode | Nearly the same: a few games with neural networks can differ |

## Next

- [Metrics](metrics.md): choose a level and read the numbers.
- [Tournaments](tournaments.md): rank many methods with one Elo fit.
- [`scenario_evaluation.py`](../examples/scenario_evaluation.py): all eight scenarios, with saving and resume.
- [`baseline_system.py`](../examples/baseline_system.py): an untrained recurrent MAPPO actor in your own compiled loop.
