# Reproduce The Paper

This folder holds everything needed to repeat the paper's experiments with the
public MARL-BGs API: the frozen search declarations, four launch scripts, the
result tables and the write-ups. The [Hydra guide](../../docs/hydra.md) teaches
the settings, overrides and studies these commands use.

| Path | What It Holds |
|---|---|
| [`hyperparameter_search.md`](hyperparameter_search.md) | The six-method search: design, every decision and the results |
| [`findings.md`](findings.md) | The five experiments that run as Hydra stages: method, results, limits |
| [`results/`](results/) | The result tables those write-ups cite |
| [`hyperparameter_search/`](hyperparameter_search/) | The search's frozen declarations (never edited) |
| `prepare.py`, `train.py`, `select.py`, `analyze.py` | Launchers for the search's protocol, training, selection and analysis |

Install the `training` and `hydra` extras. Every launcher prints its settings
without starting work: `python scripts/paper_experiments/train.py --help`.

## The Six-Method Hyperparameter Search

The search trains each method at 40,108,032 transitions per run (steps 1 to 3),
continues one run per method to 250,675,200 transitions (step 4) and plays a
tournament on the test maps (step 5). You need a GPU (we used two RTX 5090s,
one run per card), about 102 GPU-hours, and the opponent bundle released with
the trained models; `hyperparameter_search/opponents.yaml` lists each member's
expected identity, and preparation refuses a bundle that differs. Run each long
study from a frozen copy of the source: a study's workers refuse to run if the
source changes under them.

Retraining follows the same recipes but does not give the same weights:
training runs in XLA's normal GPU mode, where sums can round differently from
run to run. The tournament runs in deterministic mode, so the released models
repeat the reported tournament game for game.

| File In `hyperparameter_search/` | Step | What It Holds |
|---|---|---|
| `opponents.yaml` | 1-3 | The expected identity and role of each fixed opponent |
| `shapes_and_ppo.yaml`, `shapes_and_ppo_picks.yaml` | 1 | Study 1: 18 cases, three seeds each (54 runs), and its declared picks |
| `q_settings.yaml`, `q_settings_picks.yaml`, `q_settings_grouping.yaml` | 2 | Study 2: 10 cases (30 runs), its picks and its analysis grouping |
| `ff_mappo_team_reward.yaml` | 3 | Feedforward MAPPO's team-reward run (1 run) |
| `continue_to_250m.yaml` | 4 | Study 3: each method's tournament run continued to 250,675,200 transitions (6 runs) |
| `tournament_250m.json` | 5 | The tournament field: six study 3 policies and three scripted bots |

All steps write under one search root, here `/data/hyperparameter-search`, with
one folder per step: `shapes-and-ppo/`, `q-settings/`, `ff-mappo-levers/`,
`250m/` and `tournament-250m/` (and `opponents/` for the bundle). Leave out
`opponent_bundle` and `prepare.py` downloads the released training opponents from
Hugging Face into `OUTPUT_DIR/opponents/` and checks them the same way.

### Step 1: Shapes And PPO Recipes (54 Runs, About 45 GPU-Hours)

```bash
JAX_PLATFORMS=cpu python scripts/paper_experiments/prepare.py \
  protocol=declared_study \
  declaration=scripts/paper_experiments/hyperparameter_search/shapes_and_ppo.yaml \
  recipe_selection=scripts/paper_experiments/hyperparameter_search/shapes_and_ppo_picks.yaml \
  opponent_bundle=/data/hyperparameter-search/opponents \
  output_dir=/data/hyperparameter-search/shapes-and-ppo
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds study start \
  --config /data/hyperparameter-search/shapes-and-ppo/protocol/study.yaml \
  --output-dir /data/hyperparameter-search/shapes-and-ppo/study
python -m marl_battlegrounds study status \
  /data/hyperparameter-search/shapes-and-ppo/study --progress --summary --follow
python scripts/paper_experiments/analyze.py \
  'study_dirs=[/data/hyperparameter-search/shapes-and-ppo/study]' \
  grouping=/data/hyperparameter-search/shapes-and-ppo/protocol/grouping.yaml \
  output_dir=/data/hyperparameter-search/shapes-and-ppo/analysis
```

Preparation checks the opponent bundle and writes `protocol/`; it starts no
game. A stopped study resumes with `study start --resume-from <study folder>`.
`analysis/recipe_selection.json` holds each declared pick (mean margin, gain
over its base, seeds won, pass or fail). Each run's records are in
`study/cases/<case>/run/`.

### Step 2: Q-Method Settings (30 Runs, About 33 GPU-Hours)

Prepare and run it like step 1, with `q_settings.yaml`, `q_settings_picks.yaml`
and `/data/hyperparameter-search/q-settings`. Its analysis compares each change
with its base's study 1 runs on the same seeds, so it reads both studies:

```bash
cp scripts/paper_experiments/hyperparameter_search/q_settings_grouping.yaml \
  /data/hyperparameter-search/q-settings/protocol/
python scripts/paper_experiments/analyze.py \
  'study_dirs=[/data/hyperparameter-search/shapes-and-ppo/study,/data/hyperparameter-search/q-settings/study]' \
  grouping=/data/hyperparameter-search/q-settings/protocol/q_settings_grouping.yaml \
  output_dir=/data/hyperparameter-search/q-settings/analysis
```

The engagement floor (at least 2 kills per game against each scripted bot) is
applied when the picks are reported, from each run's `selection.json` cells.

### Step 3: Feedforward MAPPO's Team-Reward Run (1 Run, About 37 Minutes)

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

### Step 4: Continue Each Method's Run To 250M (6 Runs, About 23 GPU-Hours)

Study 3 continues saved runs, so it needs no preparation. Each case names one
run's full checkpoint at 40,108,032 transitions; replace each checkpoint folder
with your own run's (paths are relative to `<search root>/250m/protocol/`):

```bash
mkdir -p /data/hyperparameter-search/250m/protocol
cp scripts/paper_experiments/hyperparameter_search/continue_to_250m.yaml \
  /data/hyperparameter-search/250m/protocol/study.yaml
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds study start \
  --config /data/hyperparameter-search/250m/protocol/study.yaml \
  --output-dir /data/hyperparameter-search/250m/study
```

A continuation's records continue its parent's step count, so the parent's
`validation_results.json` followed by the child's gives one 0-250M curve.

### Step 5: Tournament On The Test Maps (7,200 Games, Minutes)

The field lists each method's selected study 3 actor and the three scripted
bots, with paths relative to the field file. Point the six actors at your own
selections, then:

```bash
mkdir -p /data/hyperparameter-search/tournament-250m
cp scripts/paper_experiments/hyperparameter_search/tournament_250m.json \
  /data/hyperparameter-search/tournament-250m/field.json
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  python -m marl_battlegrounds tournament \
  --config /data/hyperparameter-search/tournament-250m/field.json \
  --output-dir /data/hyperparameter-search/tournament-250m/runs
```

The deterministic flag makes two runs of the same field play the same games on
the same hardware and software; keep the default of 128 games at a time to
repeat ours exactly (see "Same Seed, Same Game" in the
[evaluation guide](../../docs/evaluation.md)). The run folder holds
`tournament_results.csv` (Elo with its 95% interval, wins, draws and losses),
`matchup_results.csv`, `map_results.csv`, `match_results.csv` and
`tournament_headline_metrics.csv`. Tiers come from the rounded Elo, as the
[tournaments guide](../../docs/tournaments.md) explains.

## The Experiments That Run As Hydra Stages

Each of these is one or two `python -m marl_battlegrounds.experiments.<stage>`
commands, given in full in its section of [`findings.md`](findings.md), with
its result tables in [`results/`](results/).

| Experiment | Stages | Section Of `findings.md` |
|---|---|---|
| The Season 0 field in the eight scenarios | `scenarios`, `scenario_analysis` | Scenario Evaluation Of The Season 0 Field |
| A language model against the Season 0 field | `llm_challenge`, `llm_analysis`, `llm_prompt` | A Language Model Against The Season 0 Field |
| Zero-shot coordination with recurrent IPPO | `zero_shot_coordination`, `zero_shot_coordination_analysis` | Zero-Shot Coordination With Recurrent IPPO |
| Human games against and alongside recurrent IPPO | `human_games`, `human_games_analysis` | Human Games Against And Alongside Recurrent IPPO |

The Season 0 teamwork tables (`results/season_0_teamwork*.csv`) come from the
`tournament_analysis` stage, and the training-curve tables
(`results/season_0_training_curves*.csv`) from `training_curves`; the search
record's matching sections give their commands.

## Paths

Every stage takes its destination as a setting. Paths on the command line are
relative to where you run the command; paths inside a YAML or JSON declaration
are relative to that file. Existing outputs are never overwritten. Large
models and results stay outside Git.
