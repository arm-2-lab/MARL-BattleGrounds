# Reproducible Experiments

These scripts use Hydra/OmegaConf to read settings and the public MARL-BGs API
to do the work. They prepare, train, select, evaluate and analyze. The shared
study runner owns workers and recovery. There is no second scheduler here.

The paper's search settings are in `hyperparameter_search/`, and
[Run The Six-Method Hyperparameter Search](#run-the-six-method-hyperparameter-search)
says how to run them. The trained models are released with the paper. The
`software_check` settings only exercise the tools. They are not recommended
hyperparameters and do not reproduce scientific results.

[Workflow checks and measured costs](../../docs/training/workflow_qualification.md)
describe the bounded six-method run, separate full evaluation and saved analysis.

## Install And Preview

Use Python 3.14 and the repository's locked environment. See
[installation](../../docs/dev/setup_ubuntu.md) for CPU and CUDA setup. The extra
for these launchers is `hydra`; training also needs `training`. The notebook
adds `notebook` and `viz`.

```bash
python -m pip install -e '.[training,hydra,notebook,viz]'
python scripts/paper_experiments/train.py --help
python scripts/paper_experiments/evaluate_qmix.py --cfg job
```

Help and configuration display do not load numerical libraries or models.
Direct Python, JSON declarations and custom training loops remain supported.
[The Hydra guide](../../docs/training/hydra.md) covers overrides and sweeps.

## Three Ways To Reproduce Work

| Task | Input | Work Performed |
| --- | --- | --- |
| Rebuild reports | Saved studies/results and declared grouping | Read saved rows; no games or training |
| Evaluate saved models | Checked actor bundle and evaluation settings | Play new games; no training |
| Repeat a declared study | Source/environment, opponents and complete study settings | Train, validate, select and report |

For each released experiment, record the exact source/environment, model and
result hashes, settings, hardware, disk space and measured costs. Matching seeds
do not give identical learned weights, even on the same machine: training runs
in XLA's normal GPU mode, where the order of floating-point sums can change
from run to run ([Repeatability On GPUs](../../docs/evaluation/repeatability.md)).
In XLA's deterministic mode, evaluating saved models repeats game for game.
Report game uncertainty separately from variation across independent training
seeds. Related snapshots from one run are not independent training seeds.

## Rebuild A Small Saved Report

This complete example uses packaged, clearly labelled synthetic saved rows:

```bash
python scripts/paper_experiments/prepare.py \
  protocol=analysis_software_check output_dir=/tmp/saved-study-demo
python scripts/paper_experiments/analyze.py \
  study_dirs=[/tmp/saved-study-demo/calibration/study] \
  grouping=/tmp/saved-study-demo/protocol/grouping.yaml \
  output_dir=/tmp/saved-study-demo/reports
```

Read the printed report paths and `data_index.json`. The synthetic rows test
calculations and file handling; their rankings are not learned-model results.
The [study-data guide](../../docs/training/study_data.md) explains the tables,
missing data, failed attempts and independent training units.

## Prepare The Six-Method Software Check

Supply a local frozen-opponent bundle with `pool.json` and its relative actor
folders. Preparation verifies all eight saved model identities, the three
scripted policies and the exact validation model. No download happens implicitly.
The packaged `combat_support_opponents.json` records the expected membership;
it does not contain model weights or claim a public download exists.

```bash
JAX_PLATFORMS=cpu python scripts/paper_experiments/prepare.py \
  protocol=software_check opponent_bundle=/data/frozen-opponents \
  output_dir=/data/software-check
python scripts/paper_experiments/train.py \
  declaration=/data/software-check/protocol/study.yaml \
  output_dir=/data/software-check/tuning/study preview=true
```

Preparation loads the validator once to obtain its panel identity, but plays no
games. It prints `protocol/study.yaml` and also writes
`checkpoint-selection.yaml`, `grouping.yaml`, `design.csv` and `source.json`.
Preview lists the exact cases and output paths without starting workers. It lists
input checks that execution still must perform.

The six methods are recurrent MAPPO, feedforward MAPPO, recurrent IPPO,
feedforward IPPO, QMIX and recurrent PQN-VDN. The software declaration uses the
shared combat/support reward (its team form `combat_support_team_reward` for
QMIX and PQN-VDN, which learn from one team reward), 60% current self-play, 20%
own history and 20%
uniform sampling across eight selected models plus `tdm-alpha`, `tdm-beta` and
`tdm-gamma`. Empty history falls back to current self-play. These shares describe
new game starts; saved exposure records show actual collected transitions.
Routine validation uses the exact pooled recurrent MAPPO actor at 250,675,200
transitions. Membership and identities stay fixed during the study.

The software settings use 32 environments and 20,480 real transitions per method,
with 900-second per-case limits. Choose explicit study resource settings for your
machine before running. Do not copy these limits into a scientific search without
planning that experiment. [Study resources](../../docs/training/hydra.md) explains
GPU assignment, memory and concurrency.

## Train And Resume

After checking the preview and resource declaration:

```bash
python scripts/paper_experiments/train.py \
  declaration=/data/software-check/protocol/study.yaml \
  output_dir=/data/software-check/tuning/study
python scripts/paper_experiments/train.py \
  resume_from=/data/software-check/tuning/study
```

The command prints the study's `report.json`. Resume inherits the saved settings;
omit `declaration` and `output_dir`. Failures print the durable study location.
Inspect its attempts and status before deciding what to do next.

Training keeps compact records and routine validation. Full behavioral metrics
and Viewer replays are collected in separate evaluation games afterward. QMIX's
learning replay buffer remains part of its learner state.

## Select And Collect Actors

The trainer already writes its selected actor and final actor. Use its actual
selection record; the final model is not automatically the best model.

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training

run = training.load_run(
    "/data/software-check/tuning/study/cases/qmix-software-check--seed-710/run"
)
if run.selected_actor is None:
    raise ValueError("No selected actor; inspect this run's selection record")
actor = training.export_actor(
    run.selected_actor, output_dir="/data/software-check/actors/qmix-selected"
)
catalog = training.collect_actors(
    [run.run_dir], output_dir="/data/software-check/actor-library"
)
print(actor, catalog)
```

For a **new named selection rule**, use the separate selection stage:

```bash
python scripts/paper_experiments/select.py \
  run_dirs=[/data/software-check/tuning/study/cases/qmix-software-check--seed-710/run] \
  declaration=/data/software-check/protocol/checkpoint-selection.yaml \
  output_dir=/data/software-check/selection confirm=true
python scripts/paper_experiments/select.py \
  resume_from=/data/software-check/selection confirm=true
```

Without `confirm=true`, the script only reports missing confirmation requests.
With it, the script calls the public validator with those exact saved requests.
It writes a linked decision in `selection/confirmed/` and keeps the first decision
unchanged. A complete resumed decision is checked against its saved evidence.
Live Python partner bindings still use the Python API.

The returned decision prints each selected actor path. Export that path when
using this alternative rule. `load_run().selected_actor` still names the trainer's
original selection. This stage chooses checkpoints, not a mean recipe across
training seeds. See [selection and grouping](../../docs/training/study_data.md).

## Evaluate And Analyze

There is one small evaluation entry for each method:

- `evaluate_recurrent_mappo.py`
- `evaluate_feedforward_mappo.py`
- `evaluate_recurrent_ippo.py`
- `evaluate_feedforward_ippo.py`
- `evaluate_qmix.py`
- `evaluate_pqn_vdn.py`

All six call the same evaluator, check the saved method and preserve saved input
scale, spawn frame, parameter sharing and statistics. PPO samples legal actions;
QMIX and PQN-VDN play greedily. No unsupported deterministic switch is implied.

```bash
python scripts/paper_experiments/evaluate_qmix.py \
  checkpoint=/data/software-check/actors/qmix-selected \
  evaluation=full evaluation.output_dir=/data/software-check/evaluation
```

The evaluator prints the generated result folder. Pass **that exact folder** to
analysis; the supplied evaluation output is a parent, not the result itself.
[Checkpoint evaluation](../../docs/evaluation/checkpoints.md) covers opponents,
maps, seeds, batch size, replays and sparse recovery commands.

```bash
python scripts/paper_experiments/analyze.py \
  study_dirs=[/data/software-check/tuning/study] \
  grouping=/data/software-check/protocol/grouping.yaml \
  evaluation_dirs=[/data/software-check/evaluation/ACTUAL-PRINTED-RUN] \
  output_dir=/data/software-check/reports
```

Replace `ACTUAL-PRINTED-RUN` with the printed folder. Add
`selection=/data/software-check/selection/confirmed/selection_decision.json`
when reporting that alternative decision. Without `evaluation_dirs`, analysis
has no post-training behavioral game tables. The software grouping has one
training unit per method and makes no across-seed uncertainty claim.

Open [the pandas notebook](../../examples/evaluation_analysis.ipynb) with the
[analysis tutorial](../../docs/evaluation/analysis.md) for outcomes, healing,
rescue, focus fire, control, agent and source-to-recipient views. It reads saved
results, explains missing values and units, and links exact selected replays.
It starts no experiment when opened or executed.

## Run The Six-Method Hyperparameter Search

This walkthrough reproduces the search reported in
[the hyperparameter search record](../../docs/training/hyperparameter_search.md)
in two parts. First, the search at 40,108,032 transitions per run: two tuning
studies and feedforward MAPPO's team-reward run (steps 1 to 3). Second, one run
per method continued from its saved 40M state to 250,675,200 transitions (study
3, step 4), followed by the tournament on the test maps (step 5). Every step
uses the public command line and API. The record explains the design, every
decision and the results; this section says what to run.

### What You Need

- MARL-BGs installed with the `training` and `hydra` extras, and a GPU. We used
  two RTX 5090s, one run per card; one card works, at about twice the time.
- The opponent bundle: the six fixed recurrent MAPPO models and the pooled 250M
  model, released with the trained models. `opponents.yaml` lists each
  member's expected identity; preparation refuses a bundle that differs.
- About 102 GPU-hours for the training runs and minutes for the tournament.
- A frozen copy of the source for each long study (see
  [the Hydra guide](../../docs/training/hydra.md)): a study's workers refuse
  to run if the source changes under them.

Retraining follows the same recipes but does not give the same weights:
training runs in XLA's normal GPU mode, where sums can round differently from
run to run (the record's
[Training Protocol](../../docs/training/hyperparameter_search.md#training-protocol)
explains this). The tournament runs in deterministic mode, so the released
models repeat the reported tournament game for game.

The committed inputs are in `hyperparameter_search/`:

| File | Step | What It Holds |
|---|---|---|
| `opponents.yaml` | 1-3 | The expected identity and role of each fixed opponent |
| `shapes_and_ppo.yaml` | 1 | Study 1: 18 cases, three search seeds each, 54 runs |
| `shapes_and_ppo_picks.yaml` | 1 | Study 1's recipe picks, declared before launch |
| `q_settings.yaml` | 2 | Study 2: 10 cases of one Q-method change each, 30 runs |
| `q_settings_picks.yaml` | 2 | Study 2's recipe picks, including the engagement floor |
| `q_settings_grouping.yaml` | 2 | Study 2's analysis grouping: its runs plus its bases' study 1 runs |
| `ff_mappo_team_reward.yaml` | 3 | Feedforward MAPPO's 40M run with the team form of the reward, seed 12259871, 1 run |
| `continue_to_250m.yaml` | 4 | Study 3: each method's tournament run continued from 40,108,032 to 250,675,200 transitions, 6 runs |
| `tournament_250m.json` | 5 | The tournament field: the six study 3 policies and three scripted bots on the test maps |

All steps write under one search root, here `/data/hyperparameter-search`:

```text
/data/hyperparameter-search/
  opponents/                 # The opponent bundle
  shapes-and-ppo/            # Step 1: protocol/, study/, analysis/
  q-settings/                # Step 2: protocol/, study/, analysis/
  ff-mappo-levers/           # Step 3: protocol/, study/
  250m/                      # Step 4: protocol/, study/
  tournament-250m/           # Step 5: field.json, runs/
```

### Step 1: Shapes And PPO Recipes (54 Runs, About 45 GPU-Hours)

Preparation checks every member of the opponent bundle against
`opponents.yaml`, binds the study's opponent names to their checked folders and
writes `protocol/`. It starts no game:

```bash
JAX_PLATFORMS=cpu python scripts/paper_experiments/prepare.py \
  protocol=declared_study \
  declaration=scripts/paper_experiments/hyperparameter_search/shapes_and_ppo.yaml \
  recipe_selection=scripts/paper_experiments/hyperparameter_search/shapes_and_ppo_picks.yaml \
  opponent_bundle=/data/hyperparameter-search/opponents \
  output_dir=/data/hyperparameter-search/shapes-and-ppo
```

Start the study in the background from your frozen copy, then watch it. Train
into `study/` beside `protocol/`, so the grouping's run folders match:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds study start \
  --config /data/hyperparameter-search/shapes-and-ppo/protocol/study.yaml \
  --output-dir /data/hyperparameter-search/shapes-and-ppo/study
python -m marl_battlegrounds study status \
  /data/hyperparameter-search/shapes-and-ppo/study --progress --summary --follow
```

The status view shows one row per run: progress, speed, the latest check's
results against each panel opponent, and the best check so far. A stopped or
crashed study resumes from its last saved state with
`study start --resume-from <study folder>`.

When the study is complete, analyze it:

```bash
python scripts/paper_experiments/analyze.py \
  study_dirs=[/data/hyperparameter-search/shapes-and-ppo/study] \
  grouping=/data/hyperparameter-search/shapes-and-ppo/protocol/grouping.yaml \
  output_dir=/data/hyperparameter-search/shapes-and-ppo/analysis
```

`analysis/recipe_selection.json` holds each declared pick: every candidate's
mean margin, the gain over its base, the seeds it won and pass or fail.
`recipe_curves.png` shows each recipe's margin at every check. Each run's own
records are in `study/cases/<case>/run/`: `selection.json` (its selected
checkpoint and confirmation games), `validation_results.json` (every check)
and `training_updates.jsonl` (every learner update).

### Step 2: Q-Method Settings (30 Runs, About 33 GPU-Hours)

Prepare and run it like step 1, with `q_settings.yaml`,
`q_settings_picks.yaml` and the folder `/data/hyperparameter-search/q-settings`.
Study 2 compares each change with its base's study 1 runs on the same seeds, so
its analysis reads both studies' run folders. Copy the combined grouping into
its protocol folder and name both study folders; the analysis refuses a
grouping run that lies outside the listed studies:

```bash
cp scripts/paper_experiments/hyperparameter_search/q_settings_grouping.yaml \
  /data/hyperparameter-search/q-settings/protocol/
python scripts/paper_experiments/analyze.py \
  study_dirs=[/data/hyperparameter-search/shapes-and-ppo/study,/data/hyperparameter-search/q-settings/study] \
  grouping=/data/hyperparameter-search/q-settings/protocol/q_settings_grouping.yaml \
  output_dir=/data/hyperparameter-search/q-settings/analysis
```

The engagement floor (at least 2 kills per game against each scripted bot) is
applied to the picks when they are reported: read each challenger's kills
against `tdm-alpha`, `tdm-beta` and `tdm-gamma` from its runs'
`selection.json` cells.

### Step 3: Feedforward MAPPO's Team-Reward Run (1 Run, About 37 Minutes)

Feedforward MAPPO trains on the team form of the reward (the record's decision
11). Prepare and run it like step 1, with `ff_mappo_team_reward.yaml` and the
folder `/data/hyperparameter-search/ff-mappo-levers`; step 4's checkpoint path
starts there. It has one run and no recipe picks, so it needs no analysis:

```bash
JAX_PLATFORMS=cpu python scripts/paper_experiments/prepare.py \
  protocol=declared_study \
  declaration=scripts/paper_experiments/hyperparameter_search/ff_mappo_team_reward.yaml \
  opponent_bundle=/data/hyperparameter-search/opponents \
  output_dir=/data/hyperparameter-search/ff-mappo-levers
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds study start \
  --config /data/hyperparameter-search/ff-mappo-levers/protocol/study.yaml \
  --output-dir /data/hyperparameter-search/ff-mappo-levers/study
```

### Step 4: Continue Each Method's Tournament Run To 250M (6 Runs, About 23 GPU-Hours)

Study 3 continues saved runs instead of training new ones, so it needs no
preparation. Each case names one run's full learner checkpoint at 40,108,032
transitions and inherits that run's settings, opponents and validation panel;
only the added budget, the 20 checks and PQN-VDN's new learning-rate schedule
are declared. The checkpoint folder names end in our runs' checkpoint ids;
replace each with your own run's 40,108,032 checkpoint from its
`checkpoints/` folder. Its paths are relative to
`<search root>/250m/protocol/`:

```bash
mkdir -p /data/hyperparameter-search/250m/protocol
cp scripts/paper_experiments/hyperparameter_search/continue_to_250m.yaml \
  /data/hyperparameter-search/250m/protocol/study.yaml
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds study start \
  --config /data/hyperparameter-search/250m/protocol/study.yaml \
  --output-dir /data/hyperparameter-search/250m/study
python -m marl_battlegrounds study status \
  /data/hyperparameter-search/250m/study --progress --summary --follow
```

In the status view, the "Done" and "Spent" columns count the whole lineage from
step 0, including the 40M run. Each continuation's records continue its
parent's step count, so the parent's `validation_results.json` followed by the
child's gives one 0-250M curve. A continuation's `selection.json` names its
selected checkpoint, which can be a parent checkpoint if nothing later beat it.

### Step 5: Tournament On The Test Maps (7,200 Games, Minutes)

The field file lists each method's selected study 3 checkpoint (the actor
folder its `selection.json` names) and the three scripted bots, with paths
relative to `<search root>/tournament-250m/`. Check that the six actor folders
match your own selections, then run the round robin. The field names our runs'
actors, including feedforward MAPPO's continuation in `ff-mappo-250m/`, where
it ran in its own study folder; step 4 puts yours in `250m/study/cases/` with
the others, so point that entrant there:

```bash
mkdir -p /data/hyperparameter-search/tournament-250m
cp scripts/paper_experiments/hyperparameter_search/tournament_250m.json \
  /data/hyperparameter-search/tournament-250m/field.json
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  python -m marl_battlegrounds tournament \
  --config /data/hyperparameter-search/tournament-250m/field.json \
  --output-dir /data/hyperparameter-search/tournament-250m/runs
```

`XLA_FLAGS=--xla_gpu_deterministic_ops=true` turns on XLA's deterministic GPU
mode, so two runs of the same field play the same games, game for game, on the
same hardware and software. Without it, a few percent of games that involve a
neural network can end differently between runs, because GPU floating-point
rounding can flip a near-tied action choice. In our tests the flag cost no
measurable time. The reported run used the command's default of 128 games at
a time; keep it to repeat our games exactly. See
[Repeatability On GPUs](../../docs/evaluation/repeatability.md) and XLA's
[determinism page](https://openxla.org/xla/determinism).

The same field runs from Python with
`marl_bgs.run_tournament(config=".../field.json", output_dir=".../runs")`; set
`os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"` before importing
`marl_battlegrounds` or JAX. The saved run folder holds:

| File | What It Holds |
|---|---|
| `tournament_results.csv` | Each entrant's Elo with its 95% interval, W-D-L and expected score |
| `matchup_results.csv` | Each pair's W-D-L, expected score and determinism flag |
| `map_results.csv` | Each entrant's results on each test map |
| `match_results.csv` | Every game: map, sides, scores, kills, deaths, length |
| `tournament_headline_metrics.csv` | Per-entrant game length, score difference, kills and deaths |

Tiers and divisions come from the rounded Elo, using the table in
[the protocol](../../docs/evaluation/protocol.md#leaderboard-tiers). Pairs that
the tool marks deterministic (greedy QMIX or PQN-VDN against each other or a
scripted bot, and the bots against each other) repeat one game per map and
spawn end within a run; read their results as 10 distinct games.

## Paths And Portable Files

Every stage exposes its destination. Command-line paths use the launch directory;
paths inside a JSON/YAML declaration use that file's directory. Live Python
mappings keep the API's working-directory rule. Hydra records sit beside the
trainer/study folder. They do not replace the existing run layout.

```text
software-check/
  protocol/                 # Declared cases, grouping and source facts
  tuning/study/
    cases/<method>-software-check--seed-710/run/
  selection/                # Optional separate checkpoint decision
  actors/                   # Explicit standalone exports
  actor-library/            # Collected models and evidence-backed labels
  evaluation/<actual-run>/  # Separate games, metrics and replays
  reports/                  # Readable analysis and figures
    analysis-data/          # Combined tables and data_index.json
```

`analysis_data_dir` can place the combined tables on a different disk.
`export_actor` and `collect_actors` take separate destinations. Full recovery
retention stays bounded while `retain_all_actors` keeps the promised play models.
Existing conflicting immutable outputs are refused; inputs are never overwritten.

[Portable bundles](../../docs/training/reproduction.md) explains local ZIP files,
explicit downloads, hashes, source/environment records and anonymous-copy limits.
Large models/results stay outside Git. Later public hosting is separate from
anonymous-review preparation. No named account, publication or venue compliance
is implied by these local tools.
