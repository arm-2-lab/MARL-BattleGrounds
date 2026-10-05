# Same Seed, Same Game: Repeatability On GPUs

**In short.** The MARL-BattleGrounds simulator is deterministic. With the same
seed, the same policies and the same settings, it plays the same game. Recording
replays and collecting metrics do not change which game is played. On a GPU,
two separate runs can still disagree on a small share of games that involve
neural-network policies. That comes from how GPUs round floating-point
arithmetic, not from the simulator or its seeds. The effect is small and
behaves like ordinary sampling noise: in our test it moved ratings by about 2
Elo (median), several times less than choosing another seed, and the six
trained policies kept their order in every run. If you need two runs to match
game for game, turn on XLA's deterministic GPU mode
([below](#turn-on-deterministic-mode)). In our tests it removed every
difference and cost no measurable time.

This page explains what you may see, why it happens, how we tested it and what
to do about it. Measured on 4 October 2026.

## Contents

- [What You May Notice](#what-you-may-notice)
- [Why It Happens](#why-it-happens)
- [How We Tested It](#how-we-tested-it)
- [What We Found](#what-we-found)
- [What This Means For You](#what-this-means-for-you)
- [Turn On Deterministic Mode](#turn-on-deterministic-mode)
- [Limits Of These Tests](#limits-of-these-tests)
- [Check Two Runs Yourself](#check-two-runs-yourself)
- [All Ratings](#all-ratings)
- [Sources](#sources)

## What You May Notice

Run the same evaluation or tournament twice on a GPU, or run it again to record
replays of chosen games, and a few games may end differently the second time: a
different score, a different length, sometimes a different winner. In our
7,200-game test, a second run with identical settings matched 92.2% of games
exactly. The other games were different, but equally valid, games between the
same two policies. The tournament's ratings barely moved.

A saved replay never changes. Loading it reads the recorded game; it does not
run the simulator or the policies again.

## Why It Happens

Computers store decimal numbers with limited precision, so the order in which
numbers are added can change the last digit of the answer: `(a + b) + c` and
`a + (b + c)` can differ by a tiny amount. A neural network adds up many
thousands of numbers for every decision.

On a GPU, XLA, the compiler JAX uses, can run the same math in several equally
correct ways that add numbers in different orders. By default it may time the
candidates when a program is compiled and keep the fastest. Timings are noisy,
so two runs can keep different ones. Some GPU operations also add numbers in
whatever order their threads finish. XLA's own
[GPU determinism documentation](https://openxla.org/xla/determinism) describes
both sources. It explains that tiny timing fluctuations during this tuning "can
cause different kernel binaries to be selected", and that "even when the
compiled binary is 100% identical across runs, running the exact same executable
multiple times can naturally yield different numerical outputs." Inside one run
the choices stay fixed, so the run is consistent with itself.

Most of the time a last-digit difference changes nothing. Sometimes two actions
score almost exactly the same, or a policy's random draw lands right on the
boundary between two actions. Then the tiny difference picks the other action,
and from that tick on the game follows a different path, like a ball rolling
off one side of a ridge instead of the other.

Seeds work as intended. A seed fixes every random number a game draws: each
champion's action draws on each tick, for each game. It cannot fix how a GPU
rounds arithmetic.

## How We Tested It

The workload was a full round robin on the five test maps (47 to 51): nine
entrants, 36 pairings, 200 games per pairing, 7,200 games in all, first to 20
points or 300 ticks, Red Zone depth 5.0, 128 games in parallel, seed 20261004.
The entrants were six trained policies (recurrent IPPO, recurrent MAPPO,
feedforward IPPO, feedforward MAPPO, QMIX and PQN-VDN) and the three scripted
bots tdm-alpha, tdm-beta and tdm-gamma. Every run used one RTX 5090, JAX and
jaxlib 0.10.1 with CUDA 13, and the public `tournament` command.

The four PPO-family policies sample each action from their action
probabilities, using the game's seed. In this test QMIX and PQN-VDN played
greedily, without exploration, and like the bots always took their
best-scoring action, using no random numbers. When neither side samples,
the seed changes nothing: such a pairing's 200 games are 10 distinct games (5
maps × 2 spawn ends), each played 20 times. This page calls them no-sampling
games. Tournament reports already label these pairings `Deterministic` in their
`determinism` column; that label is about the policies, not the GPU mode below.

| Run | GPU Mode | Metrics | Replays Recorded | Purpose |
| --- | --- | --- | --- | --- |
| Original | Normal | Priority | None | The tournament as first run |
| Normal 1 | Normal | Priority | None | Identical settings, run again |
| Recording 89 | Normal | Priority | 89 games | Re-run to record chosen games |
| Recording 23 | Normal | Priority | 23 games | A second recording re-run |
| Deterministic A | Deterministic | Priority | None | Same settings in deterministic mode |
| Deterministic B | Deterministic | Priority | None | Deterministic A, run again |
| Deterministic, Recording | Deterministic | Priority | 23 games | Does recording change games? |
| Deterministic, Metrics Off | Deterministic | None | None | Do metrics change games? |
| Deterministic, Full Metrics | Deterministic | Full | None | Do full metrics change games? |
| Seed 20261005 | Normal | Priority | None | Yardstick: an ordinary change of seed |

Two games count as identical when every per-game column of `match_results.csv`
except the run identifiers matches, including length, scores, wins, draws,
kills, deaths and all priority measurements. The metrics-off run leaves kills,
deaths, returns and the win/draw/loss flags blank, so it was compared on the
columns it fills: each game's result, length and both scores.

## What We Found

### 1. Each Run Agrees With Itself

In every run, all 20 copies of each no-sampling game came out exactly the same:
100 distinct games, no disagreements, in all ten runs. Within a run the
computation is fully consistent.

### 2. The Simulator Never Varied

Games between two scripted bots use no neural network. They were identical in
every run: 600 of 600 games each time, in both normal and deterministic mode.
The simulator's rules, its scripted bots and its seeds behave identically from
run to run.

### 3. Separate Normal GPU Runs Differ In A Few Percent Of Games

| Normal-Mode Runs Compared | Games That Differ |
| --- | --- |
| Original vs Normal 1 (identical settings) | 563 of 7,200 (7.8%) |
| Original vs Recording 89 | 670 (9.3%) |
| Original vs Recording 23 | 347 (4.8%) |
| Normal 1 vs Recording 89 | 383 (5.3%) |
| Normal 1 vs Recording 23 | 447 (6.2%) |
| Recording 89 vs Recording 23 | 448 (6.2%) |

Every pair of normal runs differs, whether or not it recorded anything. The
differences sit in games with a neural network. Between the original and Normal
1, 406 of 3,000 games between two trained policies changed and 157 of 3,600
games between a trained policy and a bot, while 0 of 600 bot-vs-bot games
changed.

### 4. Deterministic Mode Removes Every Difference

Deterministic A and Deterministic B matched in all 7,200 games, every column.
They took 482 and 479 seconds; the three normal runs without recording took 491
to 508 seconds (timed from the run's saved settings to its saved results). On
this workload, exact repeats cost nothing measurable.

### 5. Recording And Metrics Do Not Change Games

With the GPU noise removed, recording 23 replays changed nothing: the recording
run matched Deterministic A in all 7,200 games, every column. It took 738
seconds instead of 482, because writing replay files takes time.

Metrics changed nothing either. The metrics-off run and the full-metrics run
both matched Deterministic A in all 7,200 games, on every column they fill, and
gave identical ratings. The metrics-off run fills only each game's result,
length and both scores; the full-metrics run took 589 seconds instead of 482.

So the differences in finding 3 come from the GPU alone. Recording replays or
collecting metrics in a normal-mode run is not what changes the games. A normal
re-run would differ just as much without them.

### 6. The Overall Results Barely Move

| Run Compared With The Original | Median Elo Shift | Mean Elo Shift | Largest Shift | Order Of The Six Trained Policies |
| --- | --- | --- | --- | --- |
| Normal 1 (identical settings) | 2.4 | 4.7 | 14.8 | Unchanged |
| Recording 89 | 2.7 | 4.1 | 13.1 | Unchanged |
| Recording 23 | 1.5 | 3.9 | 13.3 | Unchanged |
| Deterministic A | 1.1 | 3.4 | 13.1 | Unchanged |
| Seed 20261005 (another seed, for scale) | 8.9 | 11.2 | 27.7 | Unchanged |

Shifts are taken over all nine entrants. For comparison, the original's 95% Elo
intervals were 20 to 35 points wide. Simply choosing another seed, which
researchers do all the time, moved the ratings several times more (median 8.9,
mean 11.2, largest 27.7 points) and also left the six trained policies in the
same order. The GPU noise is a small part of the noise every tournament already
carries.

The few larger shifts (13 to 15 points) always involve QMIX, PQN-VDN or a
scripted bot. Their games against each other are no-sampling games, so each
distinct game is played 20 times; when one of them changes, 20 games change
together. In Recording 89 and the deterministic runs, tdm-alpha and tdm-gamma
traded places. Those two bots were 14 Elo apart, with overlapping 95%
intervals, in the original.

## What This Means For You

- **Results from a normal GPU run are valid.** A game that differs between runs
  is another legitimate game between the same two policies. Treat the
  difference as noise, like a different seed, not as an error.
- **Keep each run's own files as its record.** Report the settings you used:
  seed, hardware, JAX version, number of parallel games and GPU mode. Do not
  expect a second normal-mode GPU run to reproduce every game.
- **To watch exact games, record them in the same run** with `replay_episodes`,
  or run in deterministic mode so that a later recording run reproduces them.
- **Recording and metrics are safe to leave on.** They do not change games.
  They do cost time: on this workload, recording 23 replays added about 4
  minutes and full metrics about 2 minutes to an 8-minute tournament.
- **For game-for-game repeats** (debugging, audits, or published numbers that
  others should match exactly), use deterministic mode with the same hardware,
  software versions and number of parallel games.
- **No-sampling pairings add less evidence than their game count.** Two
  policies that never sample play the same game for every seed, so 200 games
  are 10 distinct games. Tournament reports already label these pairings.

## Turn On Deterministic Mode

Set the flag before Python starts:

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true python -m marl_battlegrounds tournament --config field.json
```

In a script or notebook, set it before importing JAX:

```python
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import marl_battlegrounds as marl_bgs  # imports JAX after the flag is set
```

XLA's own flag description says: "Guarantees run-to-run determinism
on GPU." XLA's [determinism documentation](https://openxla.org/xla/determinism)
describes the related controls, such as `--xla_gpu_exclude_nondeterministic_ops`
and the autotuning settings, and their trade-offs. We tested
`--xla_gpu_deterministic_ops=true` only.

If you already pass other XLA flags, add this one to the same space-separated
`XLA_FLAGS` value. Deterministic mode makes repeated runs on the same hardware
and software match. It does not promise the same results on a different GPU
model or a different JAX or CUDA version.

## Limits Of These Tests

- One GPU model (RTX 5090), one software stack (JAX 0.10.1, CUDA 13) and one
  workload: tournament play with 128 games in parallel.
- Training was not tested. Training performs much more floating-point
  arithmetic, so expect the same kind of run-to-run differences in normal mode.
  The cost of deterministic mode during training was not measured.
- CPU runs were not tested here.
- The speed result covers this workload only. XLA notes that deterministic
  implementations can be slower for some operations.

## Check Two Runs Yourself

Run the same tournament twice into two folders, then compare their
`match_results.csv` files:

```python
import csv
from pathlib import Path

def games(folder):
    path = next(Path(folder).glob("*/match_results.csv"))
    with path.open(newline="") as stream:
        return {row["episode_id"]: {k: v for k, v in row.items() if k != "run_id"}
                for row in csv.DictReader(stream)}

first, second = games("runs/first"), games("runs/second")
same = sum(first[game] == second[game] for game in first)
print(f"{same} of {len(first)} games identical")
```

With `XLA_FLAGS=--xla_gpu_deterministic_ops=true` set for both runs, every game
should match.

## All Ratings

Elo per entrant in each run. Deterministic B, Deterministic, Recording and both
deterministic metrics runs match Deterministic A exactly. Seed 20261005 used
another seed on purpose, so most of its games are different games.

| Entrant | Original | Normal 1 | Recording 89 | Recording 23 | Deterministic A | Seed 20261005 |
| --- | --- | --- | --- | --- | --- | --- |
| Recurrent IPPO | 1465.6 | 1468.3 | 1462.7 | 1464.7 | 1466.1 | 1458.1 |
| PQN-VDN | 1373.5 | 1374.9 | 1372.8 | 1360.2 | 1372.6 | 1382.0 |
| QMIX | 1306.1 | 1320.9 | 1309.6 | 1305.1 | 1307.9 | 1333.8 |
| Recurrent MAPPO | 1294.9 | 1296.0 | 1294.5 | 1293.4 | 1294.9 | 1285.9 |
| Feedforward IPPO | 1230.3 | 1231.2 | 1232.9 | 1232.6 | 1231.4 | 1244.5 |
| tdm-beta | 1124.6 | 1122.9 | 1123.4 | 1124.0 | 1123.4 | 1124.1 |
| tdm-alpha | 1070.7 | 1056.6 | 1057.6 | 1071.0 | 1057.6 | 1051.4 |
| tdm-gamma | 1056.2 | 1053.8 | 1067.2 | 1068.2 | 1067.2 | 1043.1 |
| Feedforward MAPPO | 878.2 | 875.4 | 879.4 | 880.9 | 878.8 | 877.1 |

## Sources

- OpenXLA, [Determinism (GPU)](https://openxla.org/xla/determinism): the two
  sources of GPU run-to-run differences (compile-time autotuning and
  non-deterministic operations) and the flags that control them. Accessed
  4 October 2026.
- XLA's flag descriptions, as compiled into jaxlib 0.10.1:
  `--xla_gpu_deterministic_ops`,
  "Guarantees run-to-run determinism on GPU", and
  `--xla_gpu_exclude_nondeterministic_ops`, "Excludes non-deterministic ops from
  compiled executables."
