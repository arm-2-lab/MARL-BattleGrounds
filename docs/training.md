# Training

Train any of the six built-in learners, keep its best checkpoint and continue it later.

Website: https://marl-battlegrounds.com/#training

## Train A Team

```bash
python -m marl_battlegrounds.experiments.train method=mappo recipe=software_check output_dir=runs/first-hydra
```

```python
from marl_battlegrounds import training
from marl_battlegrounds.baselines.config import PPOConfig
config = training.TrainConfig(
    method="mappo", num_envs=4, total_env_steps=32, keep_past=0,  # a tiny CPU check
    ppo=PPOConfig(rollout_length=4, epochs=1),
)
result = training.train(config, output_dir="runs/first")
actor = training.load_system(result.selected_actor or result.final_actor)  # an ordinary System
```

`train` plays games, updates the learner, saves checkpoints and writes a report. It returns
`run_dir`, `final_actor`, `selected_actor` (`None` without validation) and `final_checkpoint`.
The same fields work from JSON (plus `"schema_version": 1`): `python -m marl_battlegrounds train --config CONFIG.json --output-dir RUN`.
From a checkout, [train.py](../examples/train.py) is the Hydra launcher and [mappo_training.py](../examples/mappo_training.py) a full tutorial.

**Watch Out:** the output folder must be new or empty. Short runs need
`keep_past=0`: past copies must be a full game apart, or the run stops at setup.
A transition is one game step: one tick of one game. `TrainConfig()` means 32 games
and 10,000,000 transitions; Hydra's `recipe=default` sets each method's tuned learner,
games, reward and 250,675,200 transitions, but no opponent pool and no validation: add
`pool=scripted` and `'validation_opponents=[tdm-alpha,tdm-beta,tdm-gamma]'` for a best
checkpoint, and `retain_all_actors=false` unless you want gigabytes of actor exports.

| Setting | Default | What It Does |
| --- | --- | --- |
| `num_envs` | 32 | Games played side by side. Even, and it must fit the budget and the method's minibatch rules; the tuned recipes use 512 (PPO) and 768 (QMIX, PQN-VDN). |
| `total_env_steps` | 10,000,000 | Exact real transitions; must divide by `num_envs`. |

## The Six Learners

```bash
python -m marl_battlegrounds.experiments.train method=qmix qmix.q_lr=1e-4 recipe=software_check
```

```python
from marl_battlegrounds import training
from marl_battlegrounds.baselines.config import QMIXConfig
config = training.TrainConfig(method="qmix", qmix=QMIXConfig(q_lr=1e-4))
```

| Method | Learner | Actor | Critic Or Team Value |
| --- | --- | --- | --- |
| `mappo` | Recurrent MAPPO | 128-wide recurrent network | Critic reads the full game state, which actors never see |
| `ippo` | Recurrent IPPO | Same | Critic reads only its actor's own view |
| `ff_mappo` | Feedforward MAPPO | Two 128-wide layers, no memory | Full game state |
| `ff_ippo` | Feedforward IPPO | Two 128-wide layers, no memory | Its actor's own view |
| `qmix` | Recurrent QMIX | 256-wide recurrent Q-network, replay | A mixer reads the full game state |
| `pqn_vdn` | Recurrent PQN-VDN | 512-wide recurrent Q-network, no replay | Sum of the agents' values |

PPO methods take `ppo=PPOConfig(...)`, QMIX `qmix=QMIXConfig(...)` and PQN-VDN `pqn=PQNConfig(...)`.
Loaded PPO actors sample; QMIX and PQN-VDN play greedily. Each block's `parameter_sharing`
(for example `ppo=PPOConfig(parameter_sharing="class")`) picks one shared actor
(`"all"`, default), one per class (`"class"`) or one per slot (`"none"`); each agent keeps its own memory.
Tutorials, from a checkout: [qmix_training.py](../examples/qmix_training.py), [pqn_training.py](../examples/pqn_training.py).

**Watch Out:** another method's settings block is an error. Recurrent PPO needs
`num_envs` divisible by `groups` × `minibatches` (4); PQN-VDN by `num_minibatches` (16).
PQN-VDN first plays 132 random rounds (one round is one tick of every game in the batch)
that count in the budget.
QMIX saves its replay in every checkpoint; 512 games with the default 1,000 rows
do not fit a 32 GB card.

## Opponents And Partners

```bash
python -m marl_battlegrounds.experiments.train 'opponents={self:0.6,past:0.2,tdm-alpha:0.1,tdm-beta:0.1}' keep_past=20 past_capture_interval=5000000 output_dir=runs/league
```

```python
from marl_battlegrounds import training
config = training.TrainConfig(
    opponents={"self": 0.6, "past": 0.2, "guest": 0.2}, keep_past=20,
    learner_slots=(0, 2), partners={"alpha": 1.0},  # optional frozen teammates
)
result = training.train(config, output_dir="runs/league", opponents={"guest": "tdm-beta"},
                        partners={"alpha": "tdm-alpha"})
```

Each new game draws Team B from these shares: `self` is the current learner,
`past` one of the latest `keep_past` frozen copies. Names can be built-ins, absolute
saved actor paths, `module:function` factories or live Systems. With no `opponents`:
80% self, 20% past. Partners play the slots the learner does not own and never update.

**Watch Out:** a running game keeps its opponent; new shares affect new games.
Pass live objects again on resume. `train(..., matchmaking=pfsp)` (from
`marl_battlegrounds.experiments.callbacks`) favors opponents you beat least. See [competitive_training.py](../examples/competitive_training.py) and [partner_training.py](../examples/partner_training.py).

## Use A Released Season 0 Model

```bash
python -m marl_battlegrounds models download MARL-BattleGrounds/tdm-season-0-qmix@final
```

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training
qmix = marl_bgs.load_method("MARL-BattleGrounds/tdm-season-0-qmix@100M")  # an ordinary System
config = training.TrainConfig(opponents={"self": 0.5, "MARL-BattleGrounds/tdm-season-0-qmix": 0.5})
```

The six Season 0 models are on Hugging Face with every save from 0 to 250M steps.
Type a name anywhere a model goes: opponents, partners, `initial_actor`, evaluation,
tournaments and the BattleClient. No `@` gives the save the model entered Season 0 with,
`@final` the last save, `@100M` the last save at or before 100,000,000 steps, and
`@99_950_592` exactly that save. Each name downloads one file (3 to 19 MB) once into the
Hugging Face cache. Saved runs record its exact `hf://...@<commit>/...` address, so resume
uses the same file. `marl_bgs.export_model(actor_folder, "mine.zip")` saves your own model in
the same portable format. See [train_against_released_models.py](../examples/train_against_released_models.py).

**Watch Out:** a step that does not exist fails and lists the nearest saves. On machines
without internet, run `models download` first, then set `HF_HUB_OFFLINE=1`.

## Curriculum And Rewards

```bash
python -m marl_battlegrounds.experiments.train curriculum_preset=staged shaping=false reward=marl_battlegrounds.training.reward_recipes:combat_support_reward recipe=software_check
```

```python
from marl_battlegrounds import training
config = training.TrainConfig(curriculum=True, shaping=True)  # 17 stages, potential shaping
config = training.TrainConfig(score_threshold_curriculum=True, shaping=True, shaping_mode="score_delta")
config = training.TrainConfig(curriculum=[{"share": 1.0, "maps": [0, 1], "team_size": 5}],
                              reward="my_rewards:reward", red_zone_depth=6.0)
```

`curriculum=True` grows team size on map 0, then the map pool, then plays all
42 training maps; `score_threshold_curriculum` raises points to win from 1 to 20.
Shaping adds training reward: `"potential"` keeps the goal; `"score_delta"` pays for points and
changes it. A custom `reward(before, facts, after, progress)` returns float32 `(10,)`, one per agent slot.

**Watch Out:** stages change only when a game resets; `exposure.json` shows what
was played. QMIX and PQN-VDN learn one team-average reward: give them
`combat_support_team_reward`. Actors must never read `before`, `facts` or `after`.
See [curriculum.py](../examples/curriculum.py) and [training_distributions.py](../examples/training_distributions.py).

## Validation And The Best Checkpoint

```bash
python -m marl_battlegrounds.experiments.train 'validation_opponents=[tdm-alpha,tdm-beta]' routine_seed_pairs=4 confirmation_seed_pairs=10 recipe=software_check
```

```python
from marl_battlegrounds import training
config = training.TrainConfig(validation_opponents=("tdm-alpha", "tdm-beta"),
                              routine_seed_pairs=4, confirmation_seed_pairs=10)
result = training.train(config, output_dir="runs/validated")
print(result.selected_actor, result.final_actor)  # these can differ
```

At each validation point the checkpoint plays every opponent on maps 42-46 from
both spawn ends. The two best plus the final one play fresh confirmation games;
the highest mean point margin (our points minus theirs) wins.

| Setting | Default | What It Does |
| --- | --- | --- |
| `validation_fractions` | 0.1, 0.2, ..., 1.0 | When to validate. |
| `routine_seed_pairs` | 10 | One pair is 10 games per opponent (two ends, five maps). |
| `confirmation_seed_pairs` | 50 | Fresh pairs for each finalist. |

**Watch Out:** no validation opponents means no selected actor. Keep test maps
47-51 for final evaluation. To choose again from saved results, use
`python -m marl_battlegrounds reselect-checkpoint RUN --declaration choice.json --output-dir NEW`
([training_selection.py](../examples/training_selection.py), [population_selection.py](../examples/population_selection.py)).

## Save, Resume And Extend

```bash
python -m marl_battlegrounds extend-training runs/first/checkpoints/SAVED_CHECKPOINT --additional-env-steps 1048576 --output-dir runs/child
python -m marl_battlegrounds.experiments.train --config-name resume resume_from=runs/first
```

```python
from marl_battlegrounds import training
resumed = training.train(resume_from="runs/first")
child = training.extend_training("runs/first/checkpoints/SAVED_CHECKPOINT", additional_env_steps=1_048_576,
                                 output_dir="runs/child", changes={"ppo": {"entropy_coefficient": 0.02}})
```

Resume finishes a stopped run in its own folder with its saved settings.
`extend_training` continues a full checkpoint into a new folder; the parent never
changes. `changes` may set future seeds, rates, exploration, opponents, rewards, curriculum
and validation. `TrainConfig(initial_actor=PATH)` starts a new run from saved actor weights.

**Watch Out:** resume and extend need the exact Python and library versions the
run recorded (`dependencies` in `run_details.json`), the same device kind and
`XLA_FLAGS`. Install the recorded Python with uv, for example `uv python install 3.14.5` for the released models. Extending needs a full `checkpoints/` folder, not an actor export.
A child continues the same training history; it is not a new seed. See [training_continuation.py](../examples/training_continuation.py).

## Logs, Reports And Models

```bash
python -m marl_battlegrounds analyze-training runs/first --output-dir runs/report
python -m marl_battlegrounds.experiments.training_curves 'lineages=[{name:Mine,runs:[runs/first,runs/child]}]' output_dir=runs/curves
```

```python
from marl_battlegrounds import training
result = training.train(training.TrainConfig(), output_dir="runs/logged",
                        on_update=lambda row: print(row["env_steps"], row.get("policy_loss")))
library = training.collect_actors([result.run_dir], output_dir="models")
```

`on_update` gets a copy of each saved update row; `training.make_wandb_callback(run)` sends
them to Weights & Biases (W&B). `collect_actors` gathers models with a `catalog.json`.
`marl_bgs.export_bundle("models", output_dir="share", metadata=facts)` zips them
with `source`, `environment`, `license` and `citation` facts; `prepare_bundle` unpacks.

A run folder holds `actors/`, `checkpoints/`, `run_summary.md`, `learning_curves.png`,
`validation_cells.csv` and `training_updates.jsonl`; `python -m marl_battlegrounds runs ROOT`
lists every run under a folder with its status, method, seed and steps.

**Watch Out:** a recovery can send a row twice: use `run_id` and `env_steps`. If
your callback raises, training stops with the row saved. Move a model library as one folder.

## Compare Methods Across Seeds

```bash
python -m marl_battlegrounds analyze-training runs/a-1 runs/a-2 runs/b-1 runs/b-2 --grouping grouping.json --output-dir runs/compare
```

The grouping file gives each run a `group` (a method or recipe) and a `unit` (one independent
training run, usually one seed), plus the maps, opponents, Red Zone depth and `panel_digest`
(copied from a run's `run_details.json`) the runs were validated with, and optional
`comparisons` between groups; `help(training.analyze)` shows the full format. The report
gives each group's spread across units, and each listed comparison. `training.analyze(run_dirs, output_dir=..., grouping=...)`
does the same from Python.

**Watch Out:** three seeds with 100 games each are three independent results, not 300. Game
counts shrink intervals over games, not over training seeds; compare methods across seeds.

## Next

- [hydra.md](hydra.md): recipes, sweeps, studies and devices.
- [your_method.md](your_method.md): write your own learner instead.
- [evaluation.md](evaluation.md): test the trained actor on the test maps.
