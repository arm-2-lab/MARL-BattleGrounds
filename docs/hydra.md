# Hydra Configs

Compose experiment settings from YAML and command-line overrides, preview them and sweep them.

Website: https://marl-battlegrounds.com/#configs

## Install And Preview

```bash
pip install "marl-battlegrounds[training,hydra]"
python -m marl_battlegrounds.experiments.train --cfg job --resolve
python -m marl_battlegrounds.experiments.train preview=true method=qmix recipe=software_check
python -m marl_battlegrounds.experiments.train method=mappo ppo.actor_lr=3e-4 recipe=software_check output_dir=runs/demo
```

```python
from pathlib import Path
from omegaconf import OmegaConf
from marl_battlegrounds.experiments.hydra import resolve_training_config
settings = OmegaConf.create({"method": "qmix", "seed": 3, "qmix": {"q_lr": 1e-4}})
config = resolve_training_config(settings, base_dir=Path.cwd())  # an ordinary TrainConfig
```

Hydra only composes settings; the same parser builds the same `TrainConfig` that
Python uses, and the ordinary trainer does the work. `--cfg job` prints the YAML
and writes nothing. `preview=true` checks the settings and prints the budget and
output folders; it starts no workers and loads no models.

**Watch Out:** preview does not check input files, factories or devices; launch
does. `output_dir` is the exact run folder; Hydra's records go beside it in
`runs/demo.launches/<launch-id>/` (or under `artifacts/experiments/` without it).
Relative paths start from the folder you launch in.

## Groups And Overrides

```bash
python -m marl_battlegrounds.experiments.train preview=true pool=scripted curriculum_preset=staged resources=cpu 'ppo.critic_lr=${ppo.actor_lr}'
python -m marl_battlegrounds.experiments.train preview=true 'opponents={self:0.6,past:0.2,guest:0.2}' '+runtime.opponents={guest:./models/opponent}'
python -m marl_battlegrounds.experiments.train preview=true recipe=software_check '+runtime.on_update={factory:marl_battlegrounds.experiments.callbacks:wandb_logging,kwargs:{project:demo,directory:/tmp/demo-logs,mode:offline}}'
```

| Group | Choices | What It Sets |
| --- | --- | --- |
| `method=` | `mappo` (default), `ippo`, `ff_mappo`, `ff_ippo`, `qmix`, `pqn_vdn` | The learner and its block: `ppo.*`, `qmix.*` or `pqn.*`. |
| `recipe=` | `default`, `software_check` | `default`: each method's tuned learner, games and reward, 250,675,200 transitions, validation maps 42, 43, 44 and 46; no opponent pool or validation opponents (add `pool=scripted` and `validation_opponents`). `software_check`: 32 games, 20,480 transitions. |
| `pool=` | `none`, `scripted` | `scripted`: 60% self, 20% past, 20% over tdm-alpha, tdm-beta and tdm-gamma. |
| `curriculum_preset=` | `none`, `staged` | `staged` turns on the 17-stage curriculum. |
| `resources=` | `inherit`, `cpu` | Keep the calling process's device, or use the CPU. |

Fields keep their Python names from `TrainConfig`. `runtime.opponents`, `runtime.partners`
and `runtime.validation_partners` map names to saved actor folders, model files, released
names such as `MARL-BattleGrounds/tdm-season-0-qmix@final`, built-in names or
`module:function` factories. `runtime.on_update` and `runtime.matchmaking` take a
`factory` and its `kwargs` (W&B needs the `wandb` extra). Add `+` for a key no preset defines.

**Watch Out:** quote lists and mappings for the shell; write `true` and `false` without
quotes. A `pool` plus inline `opponents` is an error. A bare `module:function` always names
a factory: write `./model:version` for a folder. Credentials never go in YAML.

## Sweeps Become Studies

```bash
python -m marl_battlegrounds.experiments.train -m preview=true method=ff_ippo,qmix seed=1,2 recipe=software_check
python -m marl_battlegrounds.experiments.train -m method=ff_ippo,qmix seed=1,2 recipe=software_check resources=cpu output_dir=runs/demo-study
python -m marl_battlegrounds study status runs/demo-study --progress --follow
python -m marl_battlegrounds study run --resume-from runs/demo-study
```

`-m` expands finite choices into jobs (here four), and one MARL-BGs study runs
them with one set of recovery rules and reports. `output_dir` is the study folder.
`+study_options.failure_policy=continue` keeps other cases going after a failure;
`+study_options.duration_seconds=600` sets a time limit that survives restarts.

**Watch Out:** only Hydra's basic sweeper with finite lists works. An override
must suit every method in the sweep. Running a sweep again into its old folder
is not a resume: use `study run --resume-from`.

## Run A Declared Study

```bash
python -m marl_battlegrounds.experiments.study declaration=my_study.json output_dir=runs/my-study
python -m marl_battlegrounds study start --config my_study.json --output-dir runs/my-study-2
```

```python
from marl_battlegrounds import training
report = training.run_study("my_study.json", output_dir="runs/my-study-3")  # same study, from Python
```

A declaration holds `"schema_version": 1`, a `name` and ordered `cases`; each case
has an `id`, a training `config` and a `seeds` list, or continues a full
`checkpoint` with `additional_env_steps`. `study start` runs in the background;
`study stop` stops only that study. See [training_study.py](../examples/training/training_study.py).

**Watch Out:** a study never adds seeds, picks recipes or changes budgets. Its workers refuse
an installed source or input file that changed after launch: start long studies from a frozen copy.
Inside a Git checkout, write outputs to an ignored folder (`runs/`, `outputs/`,
`artifacts/`) or outside it. Tiny cases need `"keep_past": 0`.

## Devices And Workers

```bash
python -m marl_battlegrounds.experiments.train recipe=software_check 'resources={devices:[{device:GPU-00000000-0000-0000-0000-000000000000,max_workers:1,memory_fraction:0.5}]}'
```

Use the full UUID from `nvidia-smi --query-gpu=uuid --format=csv,noheader`. A run
uses one device with `max_workers: 1`. A study puts the same `resources` mapping
at the top of its declaration and can list several devices and workers.

**Watch Out:** a missing declared GPU never falls back to the CPU. A single run fails. A
study pauses that GPU (its report records "missing at start"), runs on the others, and
fails only when every declared GPU is missing. Workers sharing a GPU need fractions adding
up to at most one; a fraction is not a hard memory cap.

## Resume Without Changing Settings

```bash
python -m marl_battlegrounds.experiments.train --config-name resume resume_from=runs/demo preview=true
python -m marl_battlegrounds.experiments.train --config-name resume resume_from=runs/demo +ppo.actor_lr=3e-4
```

Omitted settings come from the saved run. A supplied setting is only checked: the second
command fails unless the run used `3e-4`. To change future settings, use `extend_training` ([training.md](training.md)).

## Evaluate A Saved Model

```bash
python examples/evaluate/evaluate_qmix.py checkpoint=MODEL evaluation.output_dir=runs/evaluation
python examples/evaluate/evaluate_qmix.py checkpoint=MODEL opponent=tdm-beta evaluation=full '+evaluation.maps=[47]' evaluation.output_dir=runs/evaluation
python examples/evaluate/evaluate_qmix.py --config-name evaluate_resume checkpoint=MODEL opponent=tdm-beta evaluation.resume_from=runs/evaluation/EXACT_RUN
```

```python
import marl_battlegrounds as marl_bgs
games = marl_bgs.evaluate(marl_bgs.load_method("MODEL"), "tdm-beta", num_episodes=32, maps=[47], output_dir="runs/evaluation")
```

From a checkout, one script per method: [recurrent MAPPO](../examples/evaluate/evaluate_recurrent_mappo.py), [feedforward MAPPO](../examples/evaluate/evaluate_feedforward_mappo.py),
[recurrent IPPO](../examples/evaluate/evaluate_recurrent_ippo.py), [feedforward IPPO](../examples/evaluate/evaluate_feedforward_ippo.py), [QMIX](../examples/evaluate/evaluate_qmix.py), [PQN-VDN](../examples/evaluate/evaluate_pqn_vdn.py).
`MODEL` is an actor export, a full checkpoint folder, a model file or a released name such
as `MARL-BattleGrounds/tdm-season-0-qmix@final`. Default: 32 paired games against
`tdm-alpha`; `evaluation=full` adds full metrics for every game and replays of games 1 and 2.

**Watch Out:** `evaluation.output_dir` is a parent; the script prints the real
result folder. When resuming, give the same model and opponent and no output folder.

## Every Experiment Stage

Every stage runs like `python -m marl_battlegrounds.experiments.speed output_dir=runs/speed`,
with its own name in place of `speed`, and takes `--help` and `--cfg job`.

| Stage | What It Does |
| --- | --- |
| `prepare` | Checks a frozen opponent bundle and writes a study protocol. |
| `train` | Trains one run from a recipe, or a sweep as one study. |
| `study` | Runs or previews a prepared study. |
| `select` | Picks checkpoints from finished runs by a named rule. |
| `analyze` | Builds reports and tables from saved studies and evaluations. |
| `scenarios` | Plays a field of Systems in the eight endgame scenarios. |
| `scenario_analysis` | Explains saved scenario results: who died, how fast games ended. |
| `llm_challenge` | Plays a language-model System against a field and in the scenarios. |
| `llm_analysis` | Explains how the language model played and what its replies chose. |
| `llm_prompt` | Shows exactly what a language-model team reads on one tick. |
| `zero_shot_coordination` | Plays a team with one agent swapped for another method's agent. |
| `zero_shot_coordination_analysis` | Compares each mixed team with the original team, game by game. |
| `human_games` | Walks a person through recorded BattleClient games against or beside a System. |
| `human_games_analysis` | Compares a person's recorded games with the Systems' own games. |
| `tournament_analysis` | Finds which team measures follow strength in a saved tournament. |
| `speed` | Times the simulator alone, then one training run, on your machine. |
| `training_curves` | Draws point margin, kills and deaths from saved validation checks. |

## Next

- [training.md](training.md): what each training setting does.
- [evaluation.md](evaluation.md): evaluation settings and results.
