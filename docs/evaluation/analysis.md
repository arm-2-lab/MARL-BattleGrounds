# Analyze Saved Results

Use [evaluation_analysis.ipynb](../../examples/evaluation_analysis.ipynb) for a
worked investigation with pandas, Matplotlib and the public results API. It
covers outcomes, uncertainty, healing, rescues, coordination, control, agents
and recipients, then gives a complete gallery of the 28 catalog topics. Each
section explains what the numbers mean and what they cannot establish.

Opening it starts no experiment. Running its cells reads saved data and writes
reports. The packaged demonstration has two two-tick games with custom counter
Systems using Random actions, all ten agents active, one full record and two
replays. It teaches inspection; it is not evidence about trained methods.

## Run The Notebook

Install `marl-battlegrounds[viz,notebook]` in your chosen environment. Pandas is
part of the optional notebook extra; training does not need it. Open the notebook
in your existing notebook editor with that environment's kernel, or execute it
from a checkout with nbconvert. The extra includes the execution tools, not a
separate notebook web server.

```bash
export MARL_BGS_RESULTS=/scratch/results/EXACT_RUN
export MARL_BGS_REPORT_DIR=/scratch/reports/my-evaluation
mkdir -p /scratch/reports
python -m jupyter nbconvert --to notebook --execute \
  /absolute/checkout/examples/evaluation_analysis.ipynb \
  --output /scratch/reports/my-evaluation.executed.ipynb
```

`MARL_BGS_RESULTS` names the exact folder containing `run_details.json`. A wrong
path fails clearly. Omit it to use the small demo. Omit `MARL_BGS_REPORT_DIR` to
create and print a new temporary report folder. A supplied report folder must
be new or empty. Existing reports are preserved; choose a fresh destination when
executing the whole notebook again. Reports cannot go inside the
raw input folder. Jupyter and Matplotlib keep their own environment caches.

Optional `MARL_BGS_PHASE` and `MARL_BGS_PASS_ID` select a saved pass. Otherwise
all saved passes stay visible. Optional `MARL_BGS_STUDY_DATA` reads an existing
[study export](../training/study_data.md); it does not choose new winners.

## Follow The Investigation

1. **Check scope and coverage.** Identify actual Systems, rosters, maps, passes,
   completion and missing tables. Inspect whole-scope outcomes and API-provided
   point-margin uncertainty. Game sampling is not independent training seeds.
2. **Ask a behavior question.** Compare useful and excess healing with rescue
   opportunities. Read joint attacks and trap breaks with their support counts
   and timing. More events do not automatically mean better decisions.
3. **Check other games.** The notebook reads up to `MAX_METRIC_GAMES = 200` full
   rows, one at a time, retaining a small team-measurement table. Plots keep exact
   config/pass groups separate and show the first two groups. This bounded sample
   may not represent the whole run. The exported CSV keeps every read identity.
4. **Inspect one game deeply.** `DETAIL_ROW = 0` chooses a full row within that
   sample. Agent bars and source-recipient heatmaps preserve ownership. All 28
   topics get a panel and the coverage tables include all six metric scopes and
   the catalog's totals/recipients groups. Use the pandas `metrics` table to
   explore other columns; units, definitions and cautions come from the catalog.
5. **Read the matching replay.** The recorded game mapping links the selected
   game to its replay. No replay means unavailable, not a substitute game. The
   notebook prints a Viewer command without opening it. Inspect failures too.
6. **Form a testable claim.** The closing discussion explains independent units,
   held-out conditions and ablations. These plots can suggest a cause; they do
   not establish one.

Missing values stay missing, including ratios with no qualifying events. Zero
is shown only when recorded. Different units get separate axes. Shared topic
views reuse values and never become extra observations. Priority and full tables
are not concatenated into duplicate samples. Team A stays Team A when spawn ends
exchange. The reader parses each full CSV row's width, but keeps only one wide
record plus the chosen per-game columns. Historical records retain their limits.

## Find The Report

The report folder contains PNG and SVG charts with their plotted CSV data,
`behavior_by_game.csv`, `selected_metrics.csv` when full data exists,
`topic_coverage.csv`, `metric_groups.csv`, an optional `replay_index.csv`, and
`analysis_preview.json` with identities, coverage, interpretation and chart paths.
Figures appear directly in the executed notebook, so the discussion and evidence
can be read together. Inputs stay unchanged. Reload `marl_bgs.load_results` after
a later resume; an earlier reader is a snapshot.

For new full metrics and replays, run the separate
[checkpoint evaluation workflow](checkpoints.md), then use its printed result
folder. Keep training lean.
