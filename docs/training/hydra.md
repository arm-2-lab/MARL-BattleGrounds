# Hydra And OmegaConf

Hydra is the standard launcher workflow for project experiments. It composes
YAML settings and command-line overrides. MARL-BGs' existing Python API owns
training, validation, checkpoints, recovery and analysis. Python calls, JSON
files and custom loops remain supported without Hydra.

## Installation

From a checkout:

```bash
uv sync --locked --extra training --extra hydra
```

For an installed package, use `pip install 'marl-battlegrounds[training,hydra]'`.
The `training` extra includes the plotting dependency needed for training
reports. Training checks these imports before learner setup, output creation
or recovery changes, so a missing report dependency fails early with the install
command. Environment-only use does not need either extra.

For saved-results notebooks, also install `notebook` and `viz`:

```bash
uv sync --locked --extra training --extra hydra --extra notebook --extra viz
```

The optional `wandb` extra installs the logging SDK. Installation alone does not
start a logging session, open a network connection or change training.

## Compatibility

The qualified pair is Hydra 1.3.6 and OmegaConf 2.3.0 on Python 3.14. Import
`marl_battlegrounds.experiments.hydra` before using a Hydra entry point. This
module applies the narrow parser fix described in the
[source-reuse record](source_reuse.md#hydra-parser-compatibility).
It keeps ordinary argparse checks and restores temporary parser state even if
parser creation fails. It does not edit installed dependencies. Prepare entry
points on the main thread before workers start.

Use `version_base="1.3"` and explicitly set `hydra.job.chdir=false` in project
entry points. This keeps relative researcher paths tied to the launching
working directory. Configuration parsing does not start a numerical backend.

## Compose A Training Run

Hydra reads YAML and command-line overrides. The shared training parser checks
those values and builds the same `TrainConfig` used by Python callers. Training,
validation, checkpoints and reports still belong to the ordinary training API.

From a checkout:

```bash
python examples/train.py --help
python examples/train.py --cfg job --resolve
python examples/train.py preview=true method=qmix recipe=software_check
python examples/train.py method=mappo ppo.actor_lr=3e-4 output_dir=/scratch/demo/run
```

The installed entry is `python -m marl_battlegrounds.experiments.train`, with
the same options. Install both the `training` and `hydra` extras to execute.
Help, config display and preview need only the `hydra` extra. `--cfg job --resolve`
prints resolved YAML without writing outputs. `preview=true` checks API settings
and prints effective settings, transition budget, declared inputs and resources,
and the actual output locations. It starts no workers, loads no models, and does
not verify input payloads or available devices. Those checks remain pending.

Choose one method group: `mappo`, `ff_mappo`, `ippo`, `ff_ippo`, `qmix` or
`pqn_vdn`. The four PPO methods use `ppo.*`; QMIX uses `qmix.*`; PQN-VDN uses
`pqn.*`. Switching the method replaces the learner settings. Supplying a block
for another method is an error. All training fields keep their Python API names.
The method YAML follows the API defaults, including method-dependent save
cadence. See `TrainConfig` and the learner config docstrings for field meanings.

`recipe=default`, used when no recipe is named, applies the settings of each
method's tournament policy run in the
[six-method hyperparameter search](hyperparameter_search.md#final-policies):
its seed, learner settings, number of games and reward, the whole lineage's
budget of 250,675,200 transitions, and the search's training and validation
settings. Everything else keeps the API default.

| Method | Seed | Games | Reward | Learner settings that differ from the API defaults |
| --- | ---: | ---: | --- | --- |
| `mappo` | 20261002 | 512 | `combat_support_reward` | `actor_lr=0.0001`, `rollout_length=32`, `gamma=0.999`, `input_scale=0.01` |
| `ippo`, `ff_ippo` | 20261001 | 512 | `combat_support_reward` | As for `mappo` |
| `ff_mappo` | 12259871 | 512 | `combat_support_team_reward` | As for `mappo`, with `actor_lr=0.0003` |
| `qmix` | 12259871 | 768 | `combat_support_team_reward` | `rollout_length=32`, `buffer_size=172`, `epochs=48`, `q_lr=0.0001`, `update_period=128`, `eps_decay=2000000`, `input_scale=0.01` |
| `pqn_vdn` | 12259871 | 768 | `combat_support_team_reward` | `num_minibatches=12`, `eps_decay_fraction=0.00798352295017654`, `reward_scale=10` |

Every method also gets the search's settings: `total_env_steps=250675200`,
`keep_past=12`, `past_capture_interval=1376256`,
`validation_maps=[42,43,44,46]`, `routine_seed_pairs=4`,
`confirmation_seed_pairs=8`, `retain_all_actors=true`, and 30
`validation_fractions` that place the checks where the tournament runs had
them: every 1.6% of the budget (about 4M transitions) up to 40,108,032, then
every 4.2% (about 10.5M). The tournament runs reached 250M as a 40M run
continued from its saved state; a default run trains the same length in one
go. For PPO and QMIX, whose learning rates are constant, that is the same
schedule. PQN-VDN's exploration and learning rate follow shares of the
planned learning blocks. Its `eps_decay_fraction` is set so exploration falls
over the same 20.35 blocks (about 2M transitions) as in the tournament run,
5% of that run's 40M budget. Its learning rate falls linearly over the whole
run, which from 40M on matches its continued run's schedule; only its first
40M differ.
With `retain_all_actors=true` a 250M run keeps every exported actor, several
gigabytes per run; set it to `false` to keep only the actors validation needs.

The search also trained against a pool of six fixed recurrent MAPPO models and
validated against the pooled 250M model. Those models are not packaged, so the
default recipe keeps the API's opponent and validation-panel defaults; set
`opponents` and `validation_opponents` to the declarations in
[`ff_mappo_team_reward.yaml`](../../scripts/paper_experiments/hyperparameter_search/ff_mappo_team_reward.yaml)
(or any study declaration) to train with the search's opponents and panel. A
rerun is not bit-identical: training used XLA's normal GPU mode, and a default
run trains 250M in one go rather than as a 40M run plus a continuation. Direct Python
calls (`TrainConfig`, `PPOConfig`, `QMIXConfig`, `PQNConfig`) keep the API
defaults.

`recipe=software_check` provides small settings for checking the software: 32
environments and 20,480 transitions. It is **not a paper experiment or a claim
about learning quality**. Its task wiring is extended as the shared experiment
tools become available.

Use ordinary Hydra syntax for scalars, lists, mappings, nulls and interpolation:

```bash
python examples/train.py preview=true ppo.actor_lr=3e-4 'ppo.critic_lr=${ppo.actor_lr}'
python examples/train.py preview=true 'opponents={self:0.6,past:0.2,tdm-alpha:0.2}'
python examples/train.py preview=true 'curriculum=[{share:1.0,maps:[0,1],team_size:5}]'
python examples/train.py preview=true 'validation_panel="inputs/frozen panel.json"'
```

`pool=scripted` supplies a complete opponent declaration: 60% current self,
20% history and 20% split equally across the three scripted policies. It is an
example pool. It does not stand in for a separately declared learned-model pool.
Use either a named pool or inline `opponents`; supplying both is an error.
`pool=none` leaves the API's normal opponent defaults. `curriculum_preset=staged`
selects the existing staged curriculum; `curriculum_preset=none` adds no override.

Default outputs are under `artifacts/experiments/training/<launch-id>/`:

```text
<launch-id>/
  .hydra/                 # Composed YAML, overrides and effective API settings
  train.log               # Hydra's own log
  run/                    # The trainer's unchanged run layout
```

The ID includes a UTC timestamp and a random suffix. Set `name=demo` to change
the default parent folder. Set `output_dir=/scratch/demo/run` to choose the exact
trainer destination. Hydra's records then go beside it, under
`/scratch/demo/run.launches/<launch-id>/`. Printed result paths are authoritative.
Relative inputs use the original launch directory, not the YAML file's location.
Paths may contain spaces. Registered policy names and `module:function` references
keep their meaning. A bare `module:function` always names a factory, even when
a same-named directory exists. For a folder containing a colon, write an explicit
path such as `./model:version` or an absolute path. This also applies to study
declarations; older declarations using bare colon folder names need that prefix.

Python users may convert an existing composed mapping directly:

```python
from pathlib import Path
from omegaconf import OmegaConf
from marl_battlegrounds.experiments.hydra import resolve_training_config

settings = OmegaConf.create({"method": "mappo", "seed": 3})
config = resolve_training_config(settings, base_dir=Path.cwd())
```

Live Python objects and callbacks still belong in public API calls. YAML can
name a trusted callback factory; it cannot encode an arbitrary object graph.
Python and JSON workflows do not require Hydra.

The single-run entry accepts runtime model bindings and one declared device.
The study API owns concurrent workers and finite Hydra sweeps.

## Run A Finite Sweep

Hydra expands finite choices; one MARL-BGs study owns every case and worker.
Preview the complete method/seed grid before execution:

```bash
python examples/train.py -m preview=true method=ff_ippo,qmix seed=1,2 recipe=software_check
python examples/train.py -m method=ff_ippo,qmix seed=1,2 recipe=software_check resources=cpu output_dir=/scratch/demo/study
python examples/train.py -m preview=true method=qmix qmix.q_lr=1e-4,3e-4 seed=1,2
```

These are software examples, not selected paper hyperparameters. The first grid
has four jobs in the displayed order. Each job becomes one study case and keeps
its seed. A method-specific override must be valid for every method in that
sweep. Use separate method-specific grids when settings differ between methods.
Duplicate effective jobs are rejected, including copied actor references with
the same saved identity. Preview reads existing actor descriptions but never
loads arrays or imports user factories. Remaining input/factory identity checks
happen before the study launches, so a preview is not an input-integrity receipt.

For `-m`, `output_dir` is the exact **study** folder. The trainer's case layout
stays unchanged:

```text
/scratch/demo/
  study.launches/<launch-id>/
    multirun.yaml
    .hydra/ordered_jobs.json
    .hydra/study_declaration.json
    .hydra/study_result.json
  study/
    declaration.json
    study.json
    report.json
    cases/ff_ippo-job-0000--seed-1/
      request.json
      process.json
      resources.json
      run/
```

Without `output_dir`, Hydra records and `study/` share the printed default launch
folder. Hydra may save `multirun.yaml` during preview; no study or worker starts.
All jobs must share the same output, name, resource choices and study options.
These control the study as a whole, not a learning recipe. For example,
`+study_options.failure_policy=continue` keeps unaffected cases running after a
failure. `+study_options.duration_seconds=600` limits the study from its original
start. The saved report includes failed and never-started cases before Hydra
reports an error.

Only Hydra's BasicSweeper, finite discrete overrides and `max_batch_size=null`
are supported here. Adaptive sweepers, partial batches, nested launchers and
Hydra job directory/environment changes are refused. Use inherited environment
settings and `resources` for workers. Hydra job callbacks are not run; use the
public training callback route below. There is no second scheduler.

Resume the saved study instead of expanding a changed grid:

```bash
python -m marl_battlegrounds study run --resume-from /scratch/demo/study
python -m marl_battlegrounds study status /scratch/demo/study
python -m marl_battlegrounds study stop /scratch/demo/study
```

For detached operation or explicit cases with different per-case deadlines, use
`training.start_study(...)` and the existing study declaration. A retry keeps the
same case and seed. Repeating a sweep command into its old folder is not resume.

### Watch A Running Study

`study status --progress` prints a plain-text view of a running or finished
study. It only reads saved files and `/proc`, starts no numerical backend and
never signals a process.

```bash
python -m marl_battlegrounds study status /scratch/demo/study --progress
python -m marl_battlegrounds study status /scratch/demo/study --progress --follow
python -m marl_battlegrounds study status /scratch/demo/study --progress --summary --shaping
```

- **Table 1, Validation:** one row per run with its progress, speed, GPU, time
  spent and left, and the latest routine check (points, point margin, kills
  and deaths, Red Zone kills, home deaths and the best check so far).
  `--summary` adds the point margin at every routine check, the run's
  learning curve.
- **Table 2, Training Games:** Team A's per-game results with each reward
  part, in the first and the latest windows of 25 update rows, so the effect
  of the reward from the start to the end of training stays visible.
  `--shaping` shows every window. Kills in training games are not saved, so
  K/D shows only deaths there.
- **Table 3, Opponent Groups:** training results grouped as Current, Past
  copies, fixed models and bots, with each group's share of training ticks in
  finished games.

`--follow` refreshes the view until Ctrl+C, which stops only the viewer. Missing
values print as "-". The Exit Code column shows each case's latest attempt.

### Launch A Long Study From A Frozen Copy

A study's workers refuse a source tree that changed after launch. To keep
editing the main checkout during a long study, launch it from a git worktree of
the exact commit with its own environment:

```bash
git worktree add /scratch/frozen-study <commit>
cd /scratch/frozen-study
uv sync --locked --extra cuda13 --extra training --extra hydra --extra viz
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m marl_battlegrounds \
  study start --config /path/study.yaml --output-dir /path/study
```

Resume and status commands run from the same worktree. Remove the worktree only
after the study and its analysis are finished.

## Measure Speed On Your Machine

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds.experiments.speed output_dir=/runs/speed
```

The stage first times the simulator alone, as JaxMARL times its environments:
canonical 5v5 games on test map 47 play 500-tick calls of random legal actions
inside one `jax.jit` and `lax.scan`, at 32, 64, 128, 512 and 1024 parallel
games, with compilation timed separately. It then trains recurrent MAPPO in pure
self-play with `train` for 40 updates of 1024 games by 32 ticks and takes the
median of the warm updates from the run's `training_updates.jsonl`. It writes
`simulator.csv`, `training.json`, `speed.md` and the run folder `training/`.
Override `batch_sizes=`, `map_id=`, `train_method=` (any PPO baseline) or
`train_blocks=0` to skip training. Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false`
so memory readings show real use; pin a card with `CUDA_VISIBLE_DEVICES`.

## Plot Training Curves From Saved Runs

```bash
python -m marl_battlegrounds.experiments.training_curves \
  'lineages=[{name:Mine,runs:[/runs/first/run,/runs/more/run]}]' output_dir=/runs/curves
```

The stage reads every routine validation check of the listed runs with
`training.list_runs(progress=True)` and writes `curves.csv` (point margin,
kills and deaths per game at each check), `curves.png` and `summary.md`. List a
run, then the runs that continued it, in training order, to draw one curve from
step 0. Add `+training_runs=season_0 runs_dir=PATH` instead for the six Season 0
baselines from step 0 to 250M; [the search record](hyperparameter_search.md#curves-from-0-to-250m)
shows their curves. The stage plays no games and needs no GPU; the plot needs
the viz extra.

## Play A Language Model Against A Field And In The Scenarios

Start a model server first, as [the Qwen recipe](../llm/qwen_recipe.md) shows,
then play the Season 0 field, the eight endgame scenarios, or both:

```bash
JAX_PLATFORMS=cpu python -m marl_battlegrounds.experiments.llm_challenge \
  +field=season_0 models_dir=/runs/hyperparameter-search output_dir=/runs/llm-field
JAX_PLATFORMS=cpu python -m marl_battlegrounds.experiments.llm_challenge \
  'maps=[]' 'scenario_ids=[1,2,3,4,5,6,7,8]' output_dir=/runs/llm-scenarios
```

The stage builds one System with `llm.make_system` (the recipe's Qwen3.5-9B by
default; set `model`, `model_revision` and `server_url` for another) and plays
it as Team A.

- **Field games** go through `marl_bgs.evaluate`. On each test map the System
  plays one pair of games, one from each spawn end, against every opponent
  before the next map starts. `time_limit_hours` (8 by default) stops new
  pairs, so a run that is cut off still has the same maps for every opponent.
  `results.md` and `results.csv` give one row per opponent. Without `+field`,
  the opponents are the three scripted systems.
- **Scenarios** go through `marl_bgs.evaluate_scenarios`, one scenario at a
  time, with the scenarios stage's settings: 256 games per scenario, 128 side
  by side. `scenarios/results.md` uses that stage's table format, and
  `scenarios/costs.csv` gives the model calls, tokens and fallbacks per
  scenario.

Results are saved after every pair of games and every scenario. Running the
same command again skips what is already saved. If a different System (another
prompt version, model or setting) made the saved games, the stage stops with an
error naming the folder before it loads anything; use a fresh `output_dir`.

Then explain how the model played, from the saved games and calls alone:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds.experiments.llm_analysis \
  field_dir=/runs/llm-field scenarios_dir=/runs/llm-scenarios/scenarios \
  output_dir=/runs/llm-analysis
```

`behavior.csv` puts the model's team beside its opponents in the same games,
per game: Basic uses, damage, ticks in which two or more teammates dealt damage,
kills shared by two or more agents, distance between teammates, deaths and
deaths inside the team's own Red Zone. `answers.csv` gives how many replies were
legal, what failed, tokens per call, the median call time, seconds per tick and
which moves and attacks the model chose. Either folder can be left out. The
field games must be played with `metrics=full`, the stage's default. No model
or GPU is needed.

To see the exact text the model reads on one tick of a real game, without a
model server:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds.experiments.llm_prompt \
  output_dir=/runs/llm-prompt
```

It writes `prompt.txt`, rendered by `llm.format_actor_view` in the model's own
frame, and `menu.json`, the tick, agent and legal choices. Run it with
`JAX_PLATFORMS=cpu` to reproduce the committed example's exact bytes; on a GPU
unit positions can differ in their last digits.

## Choose Devices And Worker Limits

Leave `resources=inherit` to keep the calling process's device and allocator
settings. For a single CPU training run:

```bash
python examples/train.py preview=true resources=cpu recipe=software_check
python examples/train.py resources=cpu recipe=software_check output_dir=/scratch/demo/run
```

A single run accepts one device with `max_workers: 1`. To choose a GPU, use its
full UUID from `nvidia-smi --query-gpu=uuid --format=csv,noheader`. For example,
place this mapping in your settings, replacing the UUID with your own:

```yaml
resources:
  devices:
    - device: GPU-00000000-0000-0000-0000-000000000000
      max_workers: 1
      memory_fraction: 0.5
  compilation_cache: /scratch/jax-cache
```

The same mapping belongs at the top of an ordinary study declaration. A study
can set more workers or list multiple devices. Each worker runs one independent
case. Cases enter in declared order and use free slots in declared device order.
Multiple workers on one GPU require an explicit fraction whose worker total is
at most one. A fraction controls JAX's allocator; it is **not a hard cap on total
VRAM**. Model libraries, compiler work and other processes also need memory.
Measure peak memory and useful throughput before choosing search concurrency.
The `platform` allocator cannot be combined with a declared memory fraction.

Resource controls reach workers before JAX and user factories load. CPU means
CPU, and a declared GPU never falls back to CPU silently. A single run refuses
a missing declared GPU. A study pauses a GPU that is missing when it starts or
resumes (reason "missing at start"), keeps working on the other declared
devices, and refuses only when no declared device is left. A failed GPU query
still refuses the study.
A GPU worker also gets `XLA_PYTHON_CLIENT_PREALLOCATE=false` unless you set that
flag yourself, so it takes memory as it needs it instead of JAX's default 75% up
front, and its recorded memory is its real use. A declared fraction still caps
its allocator.
Python import paths and unrelated environment settings are preserved. A cache
path is optional and resolves from the launch directory (or study file's folder).
A declared cache keeps every compiled program, including the many small setup
programs that compile in under a second, so a later worker with the same
settings reuses them. The cache folder therefore grows with each new shape.
Preview shows applied settings without opening a backend or checking GPU access.
An already imported JAX backend cannot be moved: use a fresh launcher process.
Direct Python `train()` keeps its existing environment behavior.

Single-run `resources.json` sits with its Hydra records. Each study attempt keeps
its resource request and the environment reported at worker startup. Training's
existing execution identity records the actual backend and device kind. Failed
workers remain in the study report; settings are not measured memory claims.
With `failure_policy: stop`, a failure stops admission and asks active siblings
to stop within the existing grace period. `continue` lets other cases continue.
When a case fails on a GPU before saving any checkpoint of its own, that GPU
gets no new cases for the rest of the command, because a lost or broken card
would otherwise fail every remaining case within seconds. The other GPUs keep
working. `study status` and `report.json` show each paused GPU under
`paused_devices` with the case and error that paused it. If every GPU is paused,
the study stops with the remaining cases pending, not failed. Resume starts
with no paused GPU and retries the failed case under the usual rules; check or
fix the card first. CPU slots are never paused.
Stopping and resuming preserve original deadlines, seeds and attempt records.
Completed cases do not run again. Explicit recovery creates another attempt of
the same case; it does not create another independent training seed.

To check generated presets after a config change, run
`python scripts/dev/generate_experiment_configs.py --check`. Regenerate without
`--check`, review the YAML diff, and run the configuration tests. Default settings
belong to the API dataclasses; do not maintain a separate handwritten default list.

## Resume Without Replacing Saved Settings

Use the separate resume config. It contains no fresh method, seed or learner
defaults. Point to the run folder or an exact complete learner checkpoint:

```bash
python examples/train.py --config-name resume resume_from=/scratch/demo/run preview=true
python examples/train.py --config-name resume resume_from=/scratch/demo/run
python examples/train.py --config-name resume resume_from=/scratch/demo/run +ppo.actor_lr=3e-4
```

The last command asserts that the saved learning rate is already `3e-4`. It does
not change it. Omitted settings come from the checkpoint. Supplied learner leaves
are compared against the saved block; population mappings, lists and curriculum
declarations replace the whole asserted value before comparison. Explicit null
is a value, not omission. Any different effective setting fails before recovery
writes. Use the existing continuation API to declare supported future changes.

A run folder follows only `latest_checkpoint.json`; no file scan chooses a
replacement when that pointer is missing or damaged. Preview checks the small
saved description and continuation ancestry without reading model payloads or
starting a backend. Actual recovery verifies the complete saved state through
the trainer. Keep continuation parent descriptions at their recorded locations.

Resume uses the original run directory; omit `output_dir`. New launch records
go to its sibling `run.launches/<launch-id>/`, leaving earlier declarations
intact. A missing or null resume path fails before launch records are written.
Use the fresh configuration to start a new run. Relative `resume_from` uses the
original launch directory. Symbolic links in saved checkpoint paths are rejected.

Output directories inside a source checkout must be ignored by Git, for example
under `artifacts/`, or placed outside the checkout. Outputs inside the installed
package are refused. This prevents saved outputs from changing their own source
identity. Trainer, study and Hydra record destinations are checked before writes.
Both direct adapter calls and Hydra require a new or empty launch directory.
Saved effective settings are never overwritten. `hydra.output_subdir` stays inside
`hydra.run.dir`. The
effective API JSON remains recorded even if Hydra's own YAML subdirectory is
disabled. Hydra's generated records do not authorize overwriting a trainer run.

Training and study `output_dir` name the exact run/study folder. Evaluation
`output_dir` names the parent of its returned generated result folder. Analysis
`output_dir` names its replaceable derived-report folder. Always use returned or
printed paths instead of guessing generated names. Evaluation's existing Python
resume API likewise inherits omitted scientific options and checks explicit ones;
the public evaluation launchers use that same rule.

## Bind Opponents And Partners

Use ordinary training names under `runtime`. The API still owns loading, frozen
identity checks and client lifetimes. For example, preview a pool containing a
saved actor:

```bash
python examples/train.py preview=true \
  'opponents={self:0.6,past:0.2,guest:0.2}' \
  '+runtime.opponents={guest:./models/opponent}'
```

`runtime.opponents`, `runtime.partners` and `runtime.validation_partners` map
names to saved actor folders, built-in names or trusted `module:function`
factories. `runtime.validation_opponents` is an ordered list of those references.
Aliases keep their names even if a same-named folder exists. Relative input paths
use the launch directory; a study JSON file uses its containing directory.
The same `runtime` mapping belongs beside `config` in a study case. It reaches
fresh, resumed and continued training through the ordinary trainer keywords.

Pool shares are relative weights, normalized together by the API. For an
experiment with 60% self-play, 20% history and 20% fixed opponents, give each of
N fixed members a weight of `0.2 / N`. Include every selected model and script
once under a descriptive alias. If history is empty, its share goes to current
self-play. A live game keeps its selected opponent until reset.

Validation is a separate ordered binding. The same exact saved actor may be both
a fixed-pool member and the validator:

```bash
python examples/train.py preview=true \
  'opponents={self:0.6,past:0.2,trained:0.1,tdm-alpha:0.1}' \
  '+runtime.opponents={trained:./models/trained-opponent}' \
  '+runtime.validation_opponents=[./models/trained-opponent]'
```

This small example has two fixed members; it is not a paper search recipe.
A saved pool record can keep these ordinary config/runtime mappings alongside
model IDs, exact training steps and selection evidence. Keep that metadata out of
`TrainConfig`. A sidecar file is not automatically an execution input: preparing
a study must preserve and hash the declaration as well as bind its actual actors.
Do not select an actor by an unrecorded directory scan or infer “best” from its
filename. Existing actor identities and historical provenance remain unchanged.

A reference is loaded once during each trainer setup. Its roles keep independent
game memory and keys. Reusing one factory reference also shares its returned
client; use distinct live Python bindings when separate clients are needed.
Resource scopes must support nested use, as the `System` contract requires.
Setup caches are released before training. Preview calls no factory and loads no
model. Python callers can still pass live `System` and `Policy` objects directly.

Studies pin input files, actor descriptions, scripted-policy identities and
available factory code. They check those facts again before each worker attempt.
A factory's remote service, mutable globals and hidden client state remain
unknown; saving its reference does not make them reproducible. Credentials belong
in the environment, never YAML or saved launch records.

To resume after moving an input actor, supply its new path under the same runtime
alias. The saved artifact identity, weights, registration and member order must
match. Only the reload location changes; earlier checkpoints and exports keep
their original records. Historical exposure is preserved only with a matching
artifact receipt. Older records without that receipt keep their stricter original
path checks. A fixed study declaration remains immutable; portable study copies
use the distribution workflow.


## Continue Runs And Use Callbacks

Studies accept the same 17 future changes as `extend_training`: validation,
learning rate, exploration, history capture points and spacing, history size,
opponents, reward, partners, shaping settings, seed, curriculum, and each
method's learner settings. The direct API still checks which values are valid
for that method. Continuation keeps the parent's learning clocks and lineage;
it does not create an independent training seed. Ordinary resume preserves the
saved settings. A child's future validation bindings remain available on resume.

The optional `runtime.on_update` and `runtime.matchmaking` keys accept this form:

```yaml
runtime:
  on_update:
    factory: marl_battlegrounds.experiments.callbacks:wandb_logging
    kwargs:
      project: my-experiment
      directory: /tmp/my-experiment-logs
      mode: offline
  matchmaking: null
```

Install `marl-battlegrounds[training,hydra,wandb]` to use that logger. Use an
absolute `directory` to keep the SDK's local records where you chose. Keep
credentials in the SDK login store or environment, never in YAML. This factory
accepts no credential setting. The default has no logger and imports no SDK.

A factory is trusted installed Python code. It receives the explicit `kwargs`
once per attempt and returns a callable, or a context manager yielding one.
The launcher closes entered contexts on success and failure. A bare callable
has no cleanup owned by MARL-BGs. Runtime declarations preserve code and kwargs
receipts; ordinary resume rejects changed factory receipts before run recovery
writes. Supply the same callbacks again on resume. Live Python callbacks retain
their existing caller-owned contract. Neither route restores hidden client,
closure or remote-service state automatically. An explicit continuation may
choose a new callback and records its new receipt.

W&B starts only after the first update row has been saved locally. Each training
attempt gets a separate SDK run, grouped by the durable training run ID. Plots
use `env_steps`; records retain `attempt_id`. Recovery may repeat steps in a new
attempt and does not overwrite an earlier SDK run. A logging error stops training
with the local row intact; the latest complete checkpoint remains the recovery
point. The SDK is finished as the launch scope exits. A failure before the first
update starts no SDK run. See the upstream [W&B run API](https://docs.wandb.ai/models/ref/python/experiments/run)
for remote logging behavior and costs.

For optional prioritized opponent sampling, use
`runtime.matchmaking.factory=marl_battlegrounds.experiments.callbacks:make_pfsp`
in a YAML declaration. Its pure example rule reads saved population statistics,
gives unseen members equal eligibility with a stated 50% win prior, and changes
only future games. It keeps the existing population identity. This is a research
choice; the project's fixed 60/20/20 experiment does not enable it. Python users
can call the same `pfsp` function or supply their own callback directly.

For a complete preview without opening a logger or starting training:

```bash
python -m marl_battlegrounds.experiments.train recipe=software_check preview=true \
  '+runtime.on_update={factory:marl_battlegrounds.experiments.callbacks:wandb_logging,kwargs:{project:example,directory:/tmp/example-logs,mode:offline}}'
```


## Score A Field On The Endgame Scenarios

`python -m marl_battlegrounds.experiments.scenarios output_dir=PATH` plays every
listed system as Team A in all eight endgame scenarios, against each scenario's
own opponent, and writes `results.csv` and `results.md` (wins, draws and losses
per system and scenario). It calls `marl_bgs.evaluate_scenarios` and reads the
games back with `marl_bgs.load_results`. Override `systems=` to play your own
methods, or add `+field=season_0 models_dir=PATH` for the Season 0 field and
Random.
Play with `metrics=full`, then run
`python -m marl_battlegrounds.experiments.scenario_analysis scenarios_dir=PATH output_dir=PATH`
to count, from the saved files, which agents died, which used their Ultimate,
how long games lasted and how each system's wins compare with Random's.
[The experiments README](../../src/marl_battlegrounds/experiments/README.md)
has the Season 0 results, the exact commands and their measured cost.

## Run The Public Tutorials

The experiment tutorials use the same optional Hydra adapter. Each script keeps
its ordinary Python lesson function, usually `run(...)`. Hydra composes plain
values, applies requested process resources, then calls that function once. The
lesson uses our public API. There is no second trainer, evaluator or scheduler.

```bash
python examples/mappo_training.py --help
python examples/mappo_training.py --cfg job
python examples/mappo_training.py preview=true output_dir=/scratch/lesson/run
python examples/mappo_training.py output_dir=/scratch/lesson/run method=ff_ippo
```

Tutorial `preview=true` shows resolved lesson arguments, launch paths and resource
settings without calling the lesson. Input identity and numerical checks remain
pending until the public API runs. Use `examples/train.py` for full TrainConfig
composition, checked training previews and study-owned multirun sweeps. Tutorial
sweeps fail clearly because a complete lesson may include several distinct runs.

| Tutorial | Main Task |
| --- | --- |
| `mappo_training.py` | Train any of the four PPO variants, load its actor and inspect reports. |
| `qmix_training.py` | Train and evaluate recurrent QMIX. |
| `pqn_training.py` | Train and evaluate recurrent PQN-VDN. |
| `curriculum.py` | Use explicit map and roster stages. |
| `competitive_training.py` | Self-play, a league, continuation and response populations. |
| `partner_training.py` | Train selected slots beside frozen partners. |
| `own_learner.py` | Write a learner with a privileged training critic and permitted actor inputs. |
| `training_collection.py` | Collect rows with optional curriculum and reward shaping. |
| `training_distributions.py` | Sample episode content through the public collector. |
| `training_continuation.py` | Continue a full checkpoint into a named child run. |
| `training_selection.py` | Read saved evidence and declare a separate checkpoint choice. |
| `training_study.py` | Run or resume a declared study, including detached execution. |
| `evaluation.py` | Evaluate, validate or run a tournament. |
| `evaluation_results.py` | Authored starts, saved results, recovery and reading. |
| `cross_play_and_zsc.py` | Fix focal, partner and opponent combinations. |
| `population_selection.py` | Select and reuse a frozen population. |
| `llm.py` | Connect an existing model provider and inspect calls and games. |
| `canonical_tournament.py` | Run a fixture, use a prepared comparison, or read saved tables. |

Use the same field names in YAML and overrides. Lists and mappings should be
shell-quoted, for example `'learner_slots=[0,2]'` and
`'partners={Alpha:tdm-alpha,Partner:/path/actor}'`. Python callers may instead
supply live Systems, Policies, callbacks and method objects where supported.
`marl_bgs.load_method("package:factory")` resolves a trusted factory once when
several team compositions should share that same live System. It is the same
loader used by package evaluation; use ordinary built-in names or saved actor
paths too. A factory reference executes trusted local Python code.

Explicit tutorial outputs retain the called API's layout. Training names an
exact run folder; generated evaluation usually names a parent for a new run.
Hydra records go to a unique sibling `OUTPUT.launches/<launch-id>/`. When a
lesson has no output folder, its Hydra records go under
`artifacts/experiments/<example>/<launch-id>/`. This does not enable game
recording. The command prints the launch location, and the lesson prints its
actual result paths. Relative paths use the original launch directory, even
when `hydra.job.chdir=true` is explicitly chosen. Failed lessons keep their
launch records and report that location.

Eight focused Python/maintenance examples keep their current entry points:
`environment.py`, `baseline_system.py`, `episode_tracking.py`,
`recorded_rollout.py`, `systems.py`, `research_methods.py`,
`canonical_admission.py` and `canonical_fixture.py`. They teach direct stepping,
method construction, recording or fixture maintenance, rather than introducing
another experiment launcher. Their supported flags remain documented in each
file. The package CLI also keeps its existing flag syntax.

The interactive [BattleClient tutorial](../battle_client_tutorial.md) adds two
direct Python examples: `battle_client.py` opens a match, and
`prepare_battle_client.py` creates optional untrained actors for the tutorial.
They keep their documented flags and use the same public loading and client APIs.

Boolean lesson settings use `true` or `false`, without quotes inside the value.
For example, `evaluate=false` skips optional games. `"evaluate='false'"` passes text through the shell
and is rejected before the lesson starts. The same checks apply to Python
callers; optional inherited controls such as `rerun_existing` also accept None.
For canonical fixture work, `directory` holds artificial source records and
optional saved results. Its default launcher records sit beside that folder,
and explicit Hydra folders must stay outside both `directory` and `output_dir`.

For a small custom launcher, `training.config.validate_red_zone_depth(value)`
checks one Python float without loading numerical libraries or creating files.
It returns that float unchanged. It rejects Boolean, nonfinite, negative and
unrepresentable depths. It does not know map widths; task construction still
checks those. Integer CLI values can be converted to float before this call.
This is the same scalar rule used by the training parser and evaluation lesson.
