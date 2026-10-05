# Experiment Workflow Checks

These checks exercise the researcher tools. They do not establish learned skill,
sample efficiency, a winning method or the paper's results. The actual search
settings still need a separate experiment plan.

Use the [complete experiment guide](../../scripts/paper_experiments/README.md)
for commands and the [Hydra guide](hydra.md) for overrides, paths and recovery.
Python calls and those launchers use the same training and evaluation owners.

## Six-Method Check

The bounded check on 29–30 September 2026 trained recurrent and feedforward
MAPPO, recurrent and feedforward IPPO, QMIX and recurrent PQN-VDN. Each used
32 environments, seed 710 and 20,480 real environment transitions. The tiny
settings in `protocol=software_check` test the workflow; they are not suggested
hyperparameters.

All six runs used the shared combat/support reward and the same game-start
sampling rule: 60% current self-play, 20% own history and 20% uniform sampling
from eight frozen actors plus ALPHA, BETA and GAMMA. Empty history fell back to
current self-play. The saved exposure records distinguish requested chances,
actual game starts, collected transitions and learner-specific sample use.
BETA had no sampled starts in these short runs. This is not evidence that every
fixed member was played during training.

Every run completed initial validation, final validation and confirmation against
the exact pooled recurrent MAPPO validator. Its inference identity was
`c072dd9959e42a07249e5d4bcacdc6b8f25e6056ccd2dd6210a1fb95aecb275e`.
Each task used one seed pair on each of maps 42–46. Training wrote compact
records and priority validation metrics, with no full metric tables or Viewer
replays. QMIX still used its own learning replay buffer.

Five methods completed 160 optimizer updates. PQN completed 318 updates across
161 collection blocks because its initial collection and minibatch rules differ.
These counts must not be compared as equal amounts of experience. History
requests at 9,600 and 19,200 transitions were saved at those boundaries for five
methods, and at 9,664 and 19,264 for PQN's next completed collection boundaries.

## Measured Costs

The serial six-case study took 1,684.09 seconds on the internal RTX 5090. Each
case had a 900-second limit. Python was 3.14.5 and JAX was 0.10.1. The table
includes setup, compilation, training, validation and output work; it is not a
steady-state simulation benchmark.

| Method | Case Wall Time (Seconds) | Peak JAX Bytes In Use (MiB) | Peak Process RAM (GiB) |
| --- | ---: | ---: | ---: |
| Recurrent MAPPO | 256.25 | 308.0 | 7.64 |
| Feedforward MAPPO | 260.52 | 284.9 | 7.54 |
| Recurrent IPPO | 231.14 | 339.0 | 7.63 |
| Feedforward IPPO | 250.65 | 318.4 | 7.54 |
| QMIX | 301.68 | 538.3 | 8.03 |
| Recurrent PQN-VDN | 379.48 | 669.2 | 9.33 |

JAX reserved a roughly 23.5 GiB pool. Reserved memory, live allocator bytes and
driver-reported GPU memory are different measurements. The live-byte peaks do
not include all driver or compiler memory. The first collection/update segment
took 112–140 seconds, including compilation. This small job is dominated by
startup and compilation; dividing its total time by transitions would misstate
normal training speed. These measurements do not establish safe multi-worker
concurrency or an optimal batch size.

These times describe the source checked on 29–30 September. The later opponent
pool and past-copy speed fix cut setup, compilation and warmed block time for
this pool; its measurements are in
[Opponent Pool And Past-Copy Speed](baseline_methods.md#opponent-pool-and-past-copy-speed).

A later checkpoint-memory repair was checked separately on QMIX with 512
environments and 32-step rollouts. The run reached 6,209,536 transitions before
its planned diagnostic stop. Stable live JAX memory was about 3.94 GiB; the
observed live-memory peak was 10.41 GiB. The shared runner spent 12.80 seconds
collecting unused Python objects during the 1,109.94-second check. This is
observed time spent on cleanup, not a controlled estimate of slowdown. All 311
training blocks shared with the earlier run had identical learning and reward
values. Retained full checkpoints still held their replay state. This separate
check covers QMIX at that batch and rollout size; it does not establish memory
costs for every method or configuration. The six-method measurements above
predate this repair and retain their original scope.

The 24 retained actors were collected, bundled and extracted into a different
folder. Collection took 2.98 seconds, ZIP export 1.99 seconds and checked
extraction 0.99 seconds. The library occupied 138.25 MiB; its ZIP occupied
134.37 MiB. An installed package loaded all 24 actors in 6.90 seconds after
imports while a file-open guard refused their original study/library paths.
This check covers these saved actors, not arbitrary storage hardware.

A separate B32 check exercised the compact accounting and four evaluations of
32 four-step games. The whole process took 48.03 seconds. Changing actor weights
reused the same lean evaluation program; requesting full metrics and a replay
needed a second program. Python and Hydra calls agreed on game results, as did
lean and full recording. The warm lean calls took 0.10–0.14 seconds; the first
lean call took 10.72 seconds, including compilation. These different timings
are not a direct measurement of Hydra overhead.

The accounting-only checks used authored B32/T4 arrays. Warm calls took about
51 microseconds for PPO, 22 for QMIX and 22 for PQN, including Python dispatch.
The cheaper reference kept only the earlier output fields and took about 40,
15 and 24 microseconds respectively. These small samples do not establish a
speedup or whole-learner throughput. The added counts preserved the earlier
outputs and ordinary value changes did not cause new compilation. Peak live
JAX memory for the whole probe was 134.25 MiB; its reserved pool was 15.67 GiB
and peak process RAM was 9.43 GiB. Full recording has a different output shape;
its cold run is not an isolated estimate of recording overhead.

## Evaluation And Analysis

All six method-specific evaluation scripts played 32 paired games against the
same validator on development map 0, with B32 and the native 300-step horizon.
Normal early termination remained enabled. Each result contains 32 full metric
rows with 11,222 columns and two selected replays. Separate processes took
about 31–45 seconds each, including loading, compilation, games and output.
The six final actors also completed 30 cross-play games across all 15 pairs.

The public selection and analysis stages read the actual completed study and
evaluations. The notebook ran from a fresh installed-package kernel outside the
checkout against these real saved records. It produced pandas tables and
15 charts, saved as PNG and SVG, covering all 28 metric topics and the six record
scopes. Its tables keep units, source/recipient roles, missing values and the selected game
identity. Zero rescue opportunities remain distinct from a failed rescue and
an unavailable rescue rate.

A portable report bundle preserved the 192 rich evaluation games, 30 cross-play
games, 12 selected replays and derived study tables. Installed readers checked
the moved files with access to the original run and model folders blocked.
The whole bundle operation took 3.34 seconds; saved readback took 1.79 seconds.
The 695.40 MiB of files compressed to 28.37 MiB on this machine. This report-only
bundle supports saved analysis; it does not contain the models and raw study
inputs needed to train or play new games.

One training seed per method cannot estimate variation between independent
training runs. Related history snapshots are not extra seeds. The notebook's
game comparisons describe these saved games; they do not select a best learning
algorithm or prove coordination from a single behavioral count.

## Other Checked Routes

- The current loader read all 104 archived actors, representing 97 distinct
  inference identities, and all three scripted policies.
- A local tournament with archived actors and all three scripts played 20 short
  games. Completed resume loaded no actors, played no games and left saved bytes
  unchanged. This local field was not presented as a released official snapshot.
- All eight packaged scenarios preserved their exact starting states, including
  dead slots. Each of the six newly trained actors also completed scenario 1.
  Host, mixed, recurrent and controlled-provider Systems used the same evaluator.
- Scenario interruption, copied-result recovery and changed-identity rejection
  preserved completed games. A finished resumed result did no additional work.
- The custom CTDE lesson trained, selected, loaded and evaluated its own actor.
  A fresh process then loaded that actor and produced legal actions without
  importing training code or calling the critic.
- The recurrent IPPO and QMIX partner lessons trained physical slots 0 and 2
  with repeated classes and class-shared weights. Both selected an actor through
  20 partner-validation games, loaded it and evaluated it beside both partners.
  Each tiny CPU run used four transitions and one update. They took 366.92 and
  377.82 seconds, including compilation and native-horizon validation; these are
  software checks, not CPU training-speed benchmarks.
- A two-case CPU study stopped after the first case's initial checkpoint.
  Resume completed both cases with the original deadlines and two/one attempts.
  Saved analysis kept the interrupted attempt beside both completed attempts.
- The curriculum lesson played 1,200 transitions and started four games in each
  of its two stages. A 32-transition demonstration alone had not reached a new
  episode in stage 2, so it was not counted as proof of a played curriculum change.
- A real local Qwen3.5-4B provider supplied 40 valid actions in two four-step
  games, with no fallback or transport failure. Saved request/action records
  matched the replays. This is an integration check, not an LLM skill result.
- The custom two-name reply lesson also completed two four-step games with the
  LLM controlling only physical Team A slot 4. All eight calls produced valid
  actions; six requests contained one earlier turn from that same actor. The
  other slots used Random. Completed resume made no HTTP calls and preserved
  saved bytes. Provider startup, games and cleanup took 35.80 seconds; requests
  used 34,705 input tokens and 43 output tokens. This lesson did not save replays.
  An earlier attempt returned labelled text and correctly stopped. Its saved
  failure led to a clearer plain-name instruction, with the strict parser kept
  unchanged. The successful check does not guarantee future model replies.

## Recovery And Evidence Limits

Saved input identities and source checks remain strict. An interrupted older
partner task correctly rejected newer source bytes. Using its original source
finished the exact saved confirmation; its source guard was not weakened.
A separate named selection then completed its own 20-game partner confirmation.
The selection stage reused those completed games, wrote the linked decision,
and passed that exact decision to analysis. The original selection, training
counters and game files stayed unchanged.
Saved actor references now remain reloadable as tournament challengers. Older
records that stored an opaque live System still need that matching System on
resume. A null original selection means no winner has yet been chosen.

Focused failure tests cover publication, corruption, retention, worker startup,
callbacks, missing data and condition mismatches. Actual workflows supplement
those tests; neither proves that every possible future method or provider is
bug-free. Full quality-gate and final review status belong in the delivery
receipt. These software checks do not publish model bundles, approve an official
tournament release or launch the scientific hyperparameter search.
