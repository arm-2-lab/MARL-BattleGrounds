# Reproducible Experiments

These scripts use Hydra/OmegaConf to read settings and the public MARL-BGs API
to do the work. They prepare, train, select, evaluate and analyze. The shared
study runner owns workers and recovery. There is no second scheduler here.

**The paper's search settings and released downloads are not available yet.**
The `software_check` settings exercise the tools. They are not recommended
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
alone do not promise identical learned weights across software or hardware.
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

The search's design, rules and results are in
[the hyperparameter search record](../../docs/training/hyperparameter_search.md).
Its committed inputs are in `hyperparameter_search/`:

| File | What It Holds |
|---|---|
| `opponents.yaml` | The expected identity and role of each fixed opponent |
| `shapes_and_ppo.yaml` | Study 1: 18 cases, three search seeds each, 54 runs |
| `shapes_and_ppo_picks.yaml` | Study 1's recipe picks, declared before launch |
| `q_settings.yaml` | Study 2: 10 cases of one Q-method change each, 30 runs |
| `q_settings_picks.yaml` | Study 2's recipe picks, including the engagement floor |
| `continue_to_250m.yaml` | Study 3: each method's strongest run (point margin, kills and deaths together), continued from 40,108,032 to 250,675,200 transitions, 6 runs |

Preparation checks every member of the local opponent bundle against
`opponents.yaml` and binds the study's opponent names to their checked
folders. It writes `protocol/` and starts no game:

```bash
JAX_PLATFORMS=cpu python scripts/paper_experiments/prepare.py \
  protocol=declared_study \
  declaration=scripts/paper_experiments/hyperparameter_search/shapes_and_ppo.yaml \
  recipe_selection=scripts/paper_experiments/hyperparameter_search/shapes_and_ppo_picks.yaml \
  opponent_bundle=/data/hyperparameter-search/opponents \
  output_dir=/data/hyperparameter-search/shapes-and-ppo
```

Start the study in the background, watch it, and analyze it when it finishes.
Train into `study/` beside `protocol/`, so the grouping's run folders match.
Launch a long study from a frozen worktree, as
[the Hydra guide](../../docs/training/hydra.md) explains:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds study start \
  --config /data/hyperparameter-search/shapes-and-ppo/protocol/study.yaml \
  --output-dir /data/hyperparameter-search/shapes-and-ppo/study
python -m marl_battlegrounds study status \
  /data/hyperparameter-search/shapes-and-ppo/study --progress --follow
python scripts/paper_experiments/analyze.py \
  study_dirs=[/data/hyperparameter-search/shapes-and-ppo/study] \
  grouping=/data/hyperparameter-search/shapes-and-ppo/protocol/grouping.yaml \
  output_dir=/data/hyperparameter-search/shapes-and-ppo/analysis
```

The analysis writes each recipe pick to `recipe_selection.json` and the
recipe learning curves to `recipe_curves.png`, `.svg` and `.csv`.

Study 2 compares each change with its base's study 1 runs on the same seeds,
so its analysis needs one grouping over both studies' run folders: study 2's
`grouping.yaml` runs plus the base groups' study 1 runs, with the picks from
`q_settings_picks.yaml` under `recipe_selection`.

Study 3 continues saved runs instead of training new ones, so it needs no
preparation step. Each case names a parent's full learner checkpoint at
40,108,032 transitions and inherits that run's settings, opponents and
validation panel. Its paths are relative to `<search root>/250m/protocol/`:
copy the file there as `study.yaml`, then start and watch it as above with
`--output-dir /data/hyperparameter-search/250m/study`.

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
