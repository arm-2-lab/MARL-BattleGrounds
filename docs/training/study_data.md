# Saved Study Data

Use saved records to compare recipes, inspect failures and build tables. These
commands do not train or play games. Install `marl-battlegrounds[hydra,viz]` for
the commands and plots. Direct Python calls do not require Hydra.

## Try A Complete Offline Example

```bash
python scripts/paper_experiments/prepare.py protocol=analysis_software_check output_dir=/tmp/recipe-check
python scripts/paper_experiments/analyze.py 'study_dirs=[/tmp/recipe-check/calibration/study]' grouping=/tmp/recipe-check/protocol/grouping.yaml output_dir=/tmp/recipe-check/reports
```

The installed equivalents are `python -m marl_battlegrounds.experiments.prepare`
and `python -m marl_battlegrounds.experiments.analyze`, with the same settings.
Use `--help` or `--cfg job --resolve` to inspect settings. Relative command paths
start at the original working directory. Paths inside YAML declarations start
at the directory containing that file. Hydra saves its settings under `.hydra`
in the chosen output folder and does not change the working directory.

These fixture rows are **invented software evidence**. No model was trained and
no game was played. The lucky recipe has margins 9 and −9; its mean is 0. The
steady recipe has margins 2 and 2; its mean is 2. The saved decision selects
steady. Its seed labels, map labels, tie rule and tiny budgets are not paper
settings. Prepare refuses to replace existing calibration or protocol folders.

For real saved studies, supply their exact paths. Optional `selection` names a
completed checked checkpoint decision. `evaluation_dirs` names exact result
folders printed by evaluation. `protocol` names a YAML file declaring each
case's stage/recipe and factor roles. `analysis_data_dir` overrides the default
`output_dir/analysis-data`. Missing optional inputs stay unavailable.

## Use The Same Python API

```python
from marl_battlegrounds import training

studies = ["saved-study"]
runs = [row["run_dir"] for row in training.list_runs(studies[0])
        if row["run_id"] is not None]
report = training.analyze(runs, output_dir="reports")
paths = training.export_study_data(
    studies, output_dir="reports/analysis-data", reports=[report],
    evaluation_dirs=["saved-evaluation"],
)
print(paths["data_index.json"])

# Later, check the exported files before opening their CSV or JSON data.
paths = training.load_study_data("reports/analysis-data")
```

Only include readable run folders in the plotting call. The export still lists
all declared cases, including jobs that never started. With an explicit grouping,
an absent or empty run folder (one holding only the trainer's `.training.lock`
counts as empty) remains an unavailable member of the comparison. Other damaged
run folders still fail clearly. Neither API guesses
which external decisions or evaluation folders belong to your study.

## Read The Tables

| File | Meaning |
| --- | --- |
| `run_index.csv` | One row per declared case and study attempt. A never-started case has a row with no attempt. |
| `learning/training_updates.csv` | Saved updates, with accepted, uncommitted, superseded or unknown boundaries. Method-specific values remain in `records_json`. |
| `learning/validation_cells.csv` | Recorded checkpoint/task/map/opponent cells. Native points and normalized game scores have separate fields. |
| `learning/exposure_history.csv` | Checkpoint-bound population exposure, including requested and actual boundaries in `records_json`. |
| `hyperparameters/configurations.jsonl` | Declared jobs, base settings, input/runtime facts and their content identity. |
| `hyperparameters/design.csv` | Explicit factor levels, coding, scale and fixed/searched roles; no guessed experimental design. |
| `costs/attempt_costs.csv` | Observed clocks and available costs, with their measurement scope. |
| `selection/checkpoints.csv` | Original selection, an explicitly supplied later decision and the final actor as separate roles. Final does not mean best. |
| `selection/recipes.csv` | Checked recipe means, complete decision rules, weights, ties, eligibility and source references. |
| `evaluation/games.csv` | Durable game outcomes, exact identities and pairing/block references. |
| `evaluation/conditions.csv` | Pass declarations and the saved configurations/system registrations needed to interpret them. |
| `behavior/metrics.csv` | Long-form saved metrics, their categories, units and source/recipient meanings. Full rows and other episodes' priority rows remain available. |
| `behavior-evidence/replays.csv` | Links to saved replay files and their recorded selection reason. |

An empty CSV field or JSON `null` means unknown or unavailable. It is not zero.
A measured zero is written as zero. An excluded or uncommitted row remains
visible with its reason. Empty tables still have headers. Missing full metrics
or replays are not rebuilt from summaries.

`attempt_id` names the study attempt; `run_attempt_id` names the runner's
separate attempt. Old records may lack this join. `attempt_number` keeps the
known study ordinal without inventing a UUID. A validation task can span
attempts: its recording attempt and observed execution attempts are separate.
Those fields do not allocate all its games to the last attempt. A completed
legacy case can supply its last attempt's status even when exact runner costs
are unknown. Active costs, progress and update joins require the same saved case and request
identity. They remain usable while the worker is running, before its controller
writes a completed attempt.

Steps count real environment transitions. `start_env_steps` is the saved
parent boundary, or zero for a fresh run. `requested_additional_env_steps` is
the declared new work. `planned_env_steps` is the cumulative requested end.
Observed steps may include work later abandoned during recovery. The checkpoint
cursor determines accepted update rows; the largest row does not.

For a child that never started, configurations retain the checked parent base
settings and declared future changes. They do not pretend that the child was
admitted. `runtime_identity_scope` distinguishes declared worker source/dependency
identities from admitted run facts. A checked grouping supplies each run's recipe;
a protocol cannot silently replace it with a different recipe. A branch seed is separate from the inherited seed; continuation rate
schedules and curriculum stay in their existing declarations. Parent/child
runs are one training family, not independent seeds.

All clocks are seconds. Training time measures the trainer's recorded learning
work. Active elapsed time excludes downtime. Wall time measures age since the
run was created. Child training/active counters can include parent history.
Use the separate within-attempt differences when present; never add cumulative
snapshots across retries. Missing compile time, output-byte measurements or
memory peaks stay null. Saved memory scopes explain what was measured; export
does not query a device to invent past measurements.

## Select Means And Describe Uncertainty

Grouping schema 1 keeps its original defaults and results. Schema 2 adds:

- `measure: game_score` or `point_margin` (native Team A points minus Team B
  points per completed game).
- `cell_weights`: one nonnegative finite weight per declared map/opponent cell.
  They are normalized once. Zero-weight cells still need complete evidence.
- `recipe_selection`: named comparisons, candidate groups, maximize/minimize,
  and an exact complete `tie_order`. Without more keys the winner is the group
  with the best mean.
- An optional base rule on one `recipe_selection` entry, set with all three of
  `base` (one of its groups), `min_margin_gain` (a number of at least 0, in the
  measure's units) and `min_units_won` (an integer of at least 0). Every other
  group is a challenger. A challenger passes only when its mean beats the base
  mean by **more than** `min_margin_gain` and it beats the base on at least
  `min_units_won` paired units. Units pair by their `unit` name, so declare the
  same unit names (seeds) in both groups. "Beats" follows `direction`.

For example, this entry keeps the base unless a change gains more than 2
points and wins on at least 2 of 3 seeds:

```yaml
recipe_selection:
  - name: qmix-learning-rate
    groups: [qmix-b768-t32, qmix-lr-1e-4]
    direction: maximize
    tie_order: [qmix-b768-t32, qmix-lr-1e-4]
    base: qmix-b768-t32
    min_margin_gain: 2.0
    min_units_won: 2
```

A gain of exactly 2.0 does not pass. `recipe_selection.json` then records, for
each challenger, `mean_gain`, `units_won`, `paired_units`, each unit's two
values and `passed` (true or false). `passed_challengers` lists every passing
group in tie order. The `winner` is the base when nothing passes; otherwise it
is the passing challenger with the best mean. Declare one entry per searched
setting to test each change on its own; to build a combined recipe, take every
group in `passed_challengers`. Missing evidence leaves `passed` and the winner
null. Entries without a base rule keep their earlier record unchanged.

A unit is one independent training family. Average complete vectors equally
across units, then apply cell weights. More games or a lucky checkpoint do not
make one training seed count more. Missing candidates, incompatible conditions
or unknown seed/lineage block a recipe decision. Available descriptive values
remain visible. One unit can demonstrate software selection but cannot provide
a training-seed confidence interval. Checked old games can supply omitted
native-point means; present conflicting values are rejected.

### Recipe Learning Curves

Any grouping also makes `analyze` write `recipe_curves.png`,
`recipe_curves.svg` and `recipe_curves.csv` in the report folder. The figure
shows each group's (recipe's) point margin at every saved routine validation
step, step 0 included. Each run (seed) is a thin line. Each recipe's mean is a
thick line in the same color. The x-axis is environment steps. A grouping that
mixes methods gets one panel per method, read from each run's saved config.

The CSV has one row per run and validation step, with the columns `group`,
`unit`, `run_dir`, `method`, `env_steps`, `point_margin` and `cells_used`.
`point_margin` is our points minus the opponent's per game. It uses only the
grouping's maps and opponents, with the grouping's cell weights (equal by
default, the same rule checkpoint selection uses). Results from another panel
or Red Zone depth are left out. If a step lacks points in any declared cell,
`point_margin` is empty and `cells_used` says how many cells had points. The
mean line is drawn only at steps where every seed of that recipe with saved
margins has one, so an unfinished seed shortens the line instead of moving it. When no saved
result has a margin, the three files are skipped (old copies are removed) and
`run_summary.md` says why.

For fixed actors, `results.point_margin_summary()` reads native game points.
`by="map"` gives map rows. Its 95% percentile intervals draw complete exchanged
spawn pairs or declared larger blocks, preserving linked cells and original
game-count proportions. It requires at least two independent complete units
per condition stratum. Repeated random coordinates do not become independent
just because a pass name or supported RNG protocol label changes. Missing
games or unsupported independence withholds an interval; unknown sampling
alone keeps it, labeled "Conditional: Sampling not verified".
`bootstrap_draws=2000` and `bootstrap_seed=19044001` are API defaults, not paper
choices. Means remain descriptive when uncertainty is unavailable.

These game-pair intervals describe repeated games for fixed systems. They do
not describe variation across training seeds, prove sample efficiency or show
that a team learned a useful strategy. Use training-family comparisons and
later behavioral inspection for those separate questions.

## Refresh And Move Outputs

`data_index.json` is published last. It records every table's rows, bytes and
hash, the declared sources and unavailable evidence. Read with
`training.load_study_data` to reject missing, changed or mixed table versions.
An interruption during publication may leave mixed derived files; export again
from the same raw inputs to repair them. Raw runs are never changed.

Path fields use named roots such as `study_1/cases/...`; resolve the first
component through the index's `source_roots`, whose values are relative to the
analysis folder. Replay paths are relative to that folder. Move the complete
sibling bundle to preserve these links. A table snapshot can be read without
its raw sources, but omitted external models or replays do not become available
through relocation. Preserve immutable raw records and their original hashes.

Completed final and selected actors must match the run's checked learner
boundaries. Inherited selections keep their admitted parent identity. Moving
local actors preserves their binding; external and inherited descriptions are
included in the source receipts. All actor roles use the same inference identity,
which includes input scaling and other execution settings; the original weight
digest stays in the raw description. A valid actor from another run is rejected
before old tables are replaced. Incomplete runs do not claim final actors.

Export streams evaluation/metric rows in small batches. It retains each run's
small event and description records for joins. Checking selected evidence may
hash actor files and read saved games; it never restores a model or plays a new
game. Use a settled source folder so another writer does not change records
while they are being checked.
