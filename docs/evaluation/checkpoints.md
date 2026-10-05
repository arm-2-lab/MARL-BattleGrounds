# Evaluate Saved Models

Train with compact records, then evaluate saved actors separately. These launchers
use Hydra and OmegaConf to compose settings and call `marl_bgs.load_method` and
`marl_bgs.evaluate`. They do not train, select a model, download opponents or retry
failed games. Python callers can keep using those same API functions directly.

From a checkout, install the `training` and `hydra` extras. Add `viz` for plots and
`notebook` for the saved-results notebook. See the [Hydra guide](../training/hydra.md)
for installation, overrides and process resource settings.

Choose the command that matches the saved model. `MODEL` is an exact actor export
or full learner checkpoint folder. A training run folder or a model nickname is
not an actor. The command checks saved model metadata, even if its folder has a
misleading name.

```bash
python scripts/paper_experiments/evaluate_recurrent_mappo.py checkpoint=MODEL evaluation.output_dir=runs/evaluation
python scripts/paper_experiments/evaluate_feedforward_mappo.py checkpoint=MODEL evaluation.output_dir=runs/evaluation
python scripts/paper_experiments/evaluate_recurrent_ippo.py checkpoint=MODEL evaluation.output_dir=runs/evaluation
python scripts/paper_experiments/evaluate_feedforward_ippo.py checkpoint=MODEL evaluation.output_dir=runs/evaluation
python scripts/paper_experiments/evaluate_qmix.py checkpoint=MODEL evaluation.output_dir=runs/evaluation
python scripts/paper_experiments/evaluate_pqn_vdn.py checkpoint=MODEL evaluation.output_dir=runs/evaluation
```

To choose a training run's selected actor, resolve it once through the API:

```python
from marl_battlegrounds import training

run = training.load_run("/scratch/training/EXACT_RUN")
actor = run.selected_actor
if actor is None:
    raise ValueError("This run has no selected actor; choose a final actor explicitly")
print(actor)  # Use this exact path as checkpoint=MODEL and keep it for resume.
```

Choose `run.final_actor` explicitly if you want the final saved model instead.
A selected label records a selection result; it does not prove general strength.
The launcher receipt keeps the resolved saved identity.

Each command defaults to 32 paired games against `tdm-alpha`, canonical rosters,
the API's evaluation maps, priority metrics and no replays. The batch has 32 lanes;
`evaluation.num_envs` changes execution capacity, not the number of requested
games. These defaults are examples, not a paper's evaluation protocol.

Select an opponent by built-in name, exact saved artifact path, or trusted
`module:function` factory. A factory returns a valid `System` or `Policy` and runs
once during setup. Use an explicit path such as `./model:version` for a folder
whose name could otherwise mean a factory. PQN-VDN actors keep their saved running
statistics. Full learner folders restore the actor only, while their stored
payloads still receive the loader's integrity checks.

Inspect settings without loading model arrays:

```bash
python scripts/paper_experiments/evaluate_qmix.py --help
python scripts/paper_experiments/evaluate_qmix.py --cfg job
python scripts/paper_experiments/evaluate_qmix.py checkpoint=MODEL preview=true
```

Preview reads model descriptions and reports pending checks. It does not verify
payloads, execute factories, check device access or play games. Hydra can save
launch settings even for preview; `--help` and `--cfg job` create no launch output.

## Choose Metrics, Replays And Conditions

```bash
python scripts/paper_experiments/evaluate_qmix.py checkpoint=MODEL \
  opponent=OPPONENT_MODEL evaluation=full evaluation.num_episodes=32 \
  evaluation.output_dir=/scratch/results +evaluation.maps=[47] \
  +evaluation.seed=42
```

`evaluation=full` records full metrics for every requested game and replays for
game IDs 1 and 2. Change `evaluation.replay_episodes` explicitly to record other
games. `evaluation.metrics=none` disables optional metrics; outcomes still exist.
Replay capture is independent of metric capture. Full metrics and replays add
work only to this separate evaluation.

All serializable `evaluate` settings are supported. Fields absent from a preset
use Hydra's `+` syntax, for example `+evaluation.max_steps=300`,
`+evaluation.red_zone_depth=0`, `+evaluation.system_roster=[mage,priest]`,
`+evaluation.opponent_roster=[rogue]` and `+evaluation.full_metrics_episodes=[1]`.
Present fields use ordinary overrides, such as `evaluation.metrics=priority`.
Integer Red Zone depths become floats; invalid values fail rather than changing
the requested condition. The evaluator remains responsible for map widths,
paired schedules, roster validity and saved condition equality.

Use Python `evaluate` for live methods and writers, and `evaluate_episodes` for
exact authored starts. See [evaluation workflows](workflows.md). Those inputs do
not need a YAML encoding or a built-in learner identity.

## Find Outputs And Resume

`evaluation.output_dir` is a parent folder. The evaluator creates one named run
inside it and prints the **actual run folder**. Hydra records sit in a sibling
`evaluation.launches/` folder when the chosen parent is `evaluation/`. A request
receipt records resolved actor identities and explicitly supplied settings; a
result receipt links the saved run. With no output parent, games return in memory
and the command clearly says no game results were saved. Hydra launch records
still exist. Input model folders, result folders and launcher records stay
separate. Linked input/cache/receipt paths are refused.

Resume using the original actor and opponent references and the exact saved run:

```bash
python scripts/paper_experiments/evaluate_qmix.py --config-name evaluate_resume \
  checkpoint=ORIGINAL_MODEL opponent=ORIGINAL_OPPONENT \
  evaluation.resume_from=/scratch/results/EXACT_RUN
```

The sparse resume preset inherits omitted scientific settings and the selected
pass's original game budget. It does not restore fresh evaluation defaults over
the saved run. `evaluation.phase` and `evaluation.pass_id` choose the exact pass;
the defaults are `evaluation` and `1`. If you supply a scientific setting, such as
`+evaluation.seed=42`, it is an assertion that must match the saved conditions.
Changed actors or incompatible conditions fail before result recovery writes.
Use a new launch folder for each attempt. Do not supply `evaluation.output_dir`
when resuming.

The returned `result.episodes` contains newly played games;
`result.completed_episode_ids` includes all durable completed games. Reload
`marl_bgs.load_results(result.run_dir)` to inspect the complete saved result after
resume. An earlier `SavedResults` object is a snapshot. Errors preserve their
original type and saved output location; no hidden retry changes the experiment.

Read and plot saved output with the [analysis notebook](analysis.md).
