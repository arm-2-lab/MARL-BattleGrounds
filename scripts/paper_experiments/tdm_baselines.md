# Nine Baselines on Team Deathmatch

This experiment trains each of the nine built-in baselines ten times on 5v5 Team Deathmatch under one
protocol, and scores them on held-out test maps: recurrent MAPPO, recurrent IPPO, feedforward MAPPO,
feedforward IPPO, recurrent QMIX, recurrent PQN-VDN, HAPPO, TarMAC and PSRO. Settings come from a
declared search; each method's ten final runs are fresh, on seeds no earlier study used. This record
was written and committed before any run started. Results join it when they are public. Any choice
made after launch is listed, with its reason, in Decisions During the Experiment.

Stage A's run files are in [`tdm_baselines/`](tdm_baselines/); stage B's and the final runs' files join
it before each starts. The commands join [the paper experiment scripts' README](README.md) once each
has run as written. Every run uses a frozen copy of the code taken at launch, and its records name the
code's source digest and the package versions.

## Question

How do the nine baselines rank on 5v5 Team Deathmatch when each is trained ten times with a searched
setting under one protocol, and how do they fare against each other?

What this experiment shows, and what it does not:

| Claim | Evidence | Enough? |
| --- | --- | --- |
| The nine are ranked on Team Deathmatch under the field's evaluation protocol (Gorsane and others, NeurIPS 2022) | 90 picked teams in a round robin on the test maps, against opponents none of them trained against or was picked against; the statistics in Scoring, with 95% intervals; wins, draws, losses, kills and deaths beside every score | Yes, for Team Deathmatch at 250,675,200 steps. Two departures from the protocol are stated: the number of settings searched differs by method, and this record covers one game. |
| Each setting comes from a declared search, with the tuning each method received stated in numbers | The grids, seeds and pick rule below, all fixed before launch; every search run reported with its compute | Yes for "declared and reported". Not for "equal tuning": the number of settings differs by the rule in Search Design. |
| Head-to-head results and Elo between the nine | Every pair of teams in the round robin | Yes, with intervals over training runs. |
| Every table can be repeated with MARL-BattleGrounds | One command per table on the pinned release, from the released teams and results | Shown when each command has run from a clean install. Retraining is not claimed to give the same weights. |

Not claimed: a best possible setting for any method; that a ranking holds in other games; why a
method ranks where it does.

## Game, Maps and Splits

As in [the six-method search](hyperparameter_search.md#game-maps-and-splits): canonical 5v5 Team
Deathmatch, first to 20 points, 300 ticks, a kill worth 1 point and 2 inside the victim's own Red Zone
(depth 5.0). Training maps 0 to 41; validation maps 42 to 46; test maps 47 to 51. Map 45 is back in
validation: the scripted opponents, whose pathing fails on it, now only train and never validate.

## Methods

| Method | Code | The Question It Answers | Its Setting |
| --- | --- | --- | --- |
| Recurrent MAPPO | [`baselines/ppo.py`](../../src/marl_battlegrounds/baselines/ppo.py) | What a central critic with memory gives | The six-method search, then Stage A |
| Recurrent IPPO | `baselines/ppo.py` | What independent learners with memory give, without a central critic | The same |
| Feedforward MAPPO | `baselines/ppo.py` | What memory adds, against recurrent MAPPO | The same |
| Feedforward IPPO | `baselines/ppo.py` | The PPO family's lower bound: no memory, no central critic | The same |
| QMIX | [`baselines/qmix.py`](../../src/marl_battlegrounds/baselines/qmix.py) | Value-based learning with a monotonic team value and replay | The six-method search |
| PQN-VDN | [`baselines/pqn.py`](../../src/marl_battlegrounds/baselines/pqn.py) | Value-based learning with an additive team value and no replay | The six-method search |
| HAPPO | `baselines/ppo.py`, one actor per class | What one actor per kind of agent, updated in turn, gives a team of different agents. On the canonical roster each class plays once, so one actor per class is one actor per agent. | Stage B |
| TarMAC | `baselines/ppo.py`, one message round a tick | What learned messages between teammates give. Messages are read a tick later, so each agent still decides alone. | Stage B |
| PSRO | [`training/psro.py`](../../src/marl_battlegrounds/training/psro.py) | What training against a growing population gives: a recurrent IPPO learner trained against the Nash mixture of its population (the mix of its members that is hardest to beat) | Stage B |

The reward is the six-method search's combat and support reward. HAPPO, QMIX, PQN-VDN and feedforward
MAPPO learn from its team form; the others from its per-agent form.

## Opponents in Training

| Share | Opponent |
| ---: | --- |
| 60% | The learner's current weights |
| 20% | One of its 12 newest saved copies (one saved every 1,376,256 steps) |
| 15% | Fixed learned opponents: the six learned teams of the six-method search on a ramp, 2.5% each |
| 5% | The scripted `tdm-alpha`, `tdm-beta` and `tdm-gamma`, about 1.7% each |

**The ramp.** The fixed learned opponents grow with the learner. A run meets each team's save at
40,108,032 steps for its first 40,108,032 steps; then its later saves, switching at 40,108,032,
60,162,048, 80,216,064 and 100,270,080 steps; and from 100,270,080 steps to the end, its save near
120M. Each of the first four stages uses the team's last save at or before the step where that stage
ends (the stage from 40.1M to 60.2M, for example, uses each team's last save at or before 60,162,048);
the last stage, from 100.3M to the end, uses each team's save near 120M. All are named in the table. A
game already under way when a stage switches finishes against its old save.

| Team | First Stage | From 40.1M | From 60.2M | From 80.2M | From 100.3M |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recurrent IPPO | 40,108,032 | 59,801,600 | 79,872,000 | 99,942,400 | 120,012,800 |
| Recurrent MAPPO | 40,108,032 | 59,801,600 | 79,872,000 | 99,942,400 | 120,012,800 |
| Feedforward IPPO | 40,108,032 | 59,801,600 | 79,872,000 | 99,942,400 | 120,012,800 |
| Feedforward MAPPO | 40,108,032 | 59,801,600 | 79,872,000 | 99,942,400 | 120,012,800 |
| QMIX | 40,108,032 | 59,817,984 | 79,478,784 | 99,950,592 | 119,611,392 |
| PQN-VDN | 40,108,032 | 59,080,704 | 78,741,504 | 98,402,304 | 118,063,104 |

The ramp stops at 120M, so the teams that pick checkpoints in the final runs, saved at 208M to 250M,
are never met in training. A 40M search run meets only the first stage. PSRO's population starts from
the first stage's teams and the three scripted bots, and its own growing population is its ramp.
[`tdm_baselines/opponents.yaml`](tdm_baselines/opponents.yaml) lists the search's fixed opponents and
validation teams (the first-stage saves, the three bots and the saves near 80M) by exact step and actor
digest; [`tdm_baselines/final/opponents.yaml`](tdm_baselines/final/opponents.yaml) lists the final runs'
(the five stages' saves, the three bots and the leaderboard saves) the same way. Each is the six-method search's released model at
that exact step (for example `models/step-040_108_032.zip`, not the `@40M` name, which loads an earlier
save). Compare them by actor digest: a save can carry another checkpoint ID when a continuation exported
the same policy again.

## Validation Set and Picks

| Setting | Search Runs (40M) | Final Runs (250M) |
| --- | --- | --- |
| Validation set | The six learned teams' saves near 80M: 79,872,000 (the four PPO teams), 79,478,784 (QMIX), 78,741,504 (PQN-VDN) | The six learned teams' leaderboard saves: recurrent IPPO 250,675,200; recurrent MAPPO 219,103,232; feedforward IPPO 219,103,232; feedforward MAPPO 208,568,320; QMIX 219,095,040; PQN-VDN 240,156,672 |
| Maps | 42 to 46 | 42 to 46 |
| Routine checks | Step 0, then every 4.0M steps to 40.1M: 11 checks | Step 0; every 4.0M steps to 40.1M; every 10.5M steps from 50.6M to 250.7M; and at 60.2M, 80.2M and 100.3M: 34 checks |
| Games | 8 a map against each team: 240 a check | The same |
| Confirmation | The best two routine checks (step 0 left out) and the last one play 16 more games a map against each team: 480 a check | The same |
| Pick | The confirmed check with the highest point margin among those averaging at least 2 kills a game against each team of the validation set; kills and deaths shown | The same |

**Point margin** is our points minus the opponent's, averaged within each map and opponent, and the 30
map-and-opponent cells are averaged with equal weight. Kills and deaths are shown beside it. A check
averaging under 2 kills a game against any team of the validation set cannot be picked; this kill floor
is applied to the confirmed checks after each run, and a run with no passing check is listed in
Decisions During the Experiment. The scripted
bots never validate: they are useful early in training, and their pathing fails on one validation map.
PSRO picks among its own candidates, one a round, with its own games (10 seed pairs a check, 50 to
confirm).

## Search Design

Every search run lasts 40,108,032 steps on seeds 20261211, 20261212 and 20261213. Every grid includes a
fresh reference row on the same seeds, because the training opponents and the validation set are new.

**Stage A: the four PPO methods.** Each method's reference is its final recipe from the six-method
search (`reference` for recurrent MAPPO, recurrent IPPO and feedforward IPPO; team reward with actor
rate 0.0003 for feedforward MAPPO).

| Row | Change From the Reference |
| --- | --- |
| `reference` | None |
| `epochs-10` | `epochs = 10` (reference 4) |
| `clip-1e-1` | `clip_epsilon = 0.1` (reference 0.2) |
| `epochs-10-clip-1e-1` | Both |
| `minibatches-1`, `minibatches-4` | `minibatches` (reference 2) |
| `entropy-3e-3`, `entropy-3e-2` | `entropy_coefficient` (reference 0.01) |
| `grad-limit-10` | `max_grad_norm = 10` (reference 0.5) |

**Stage B: HAPPO, TarMAC and PSRO**, after stage A. HAPPO's and TarMAC's reference is recurrent MAPPO's
recipe after stage A (HAPPO on the team form of the reward); PSRO's learner is recurrent IPPO's. A stage A
change carries into them only after it passes its rerun (below). If it changed epochs or clip, the new
method's own row for that setting tests the other value.

| Method | Rows |
| --- | --- |
| HAPPO | Reference; actor rate 0.0003; critic rate 0.001; 10 epochs; clip 0.1; HAPPO's published settings as one row (both rates 0.0005, 5 epochs, 1 minibatch, gradient limit 10) |
| TarMAC | Reference (message sizes 16 and 32); actor rate 0.0003; critic rate 0.001; actor rate 0.0007 (its paper's value, with RMSProp there and Adam here); 10 epochs; clip 0.1 |
| PSRO | Reference (rounds of 20,054,016 steps); actor rate 0.0003; uniform share 0.1; uniform share 0.4; rounds of 10,027,008; rounds of 5,013,504 |

The uniform share mixes g/n of each of the n members into PSRO's training opponents, with (1 - g) times
the Nash weights; its two rows run on the pinned release plus that one setting, shown first to leave the
default unchanged. PSRO's budget of steps also pays for its payoff games (the games between population
members that set the Nash weights) and its validation games. Counting every game at its longest, 300
ticks, its learner trains on up to about 3.5%, 7% and 18% fewer steps at 250M for rounds of 20.05M,
10.03M and 5.01M (payoff games about 1.6%, 4.3% and 13.1%; validation about 1.9%, 2.9% and 4.7%). The
other methods' validation games are outside their 250M.

**Pick rule.** A change replaces the reference only if its mean point margin over the three seeds beats
the reference's by more than 2 points, it wins on at least 2 of the 3 seeds, and it averages at least 2
kills a game against each team of the validation set. With three seeds, differences under about 3 points
cannot be told from noise. When several changes pass, the one with the highest mean is rerun; if it fails
its rerun, the reference stands.

**Rerun of a passing change.** In a simulation of this rule, with eight changed rows and noise of about
1.5 points in a three-seed mean, noise alone passes at least one row about half the time (17% to 74%
for noise of 1.0 to 2.0 points); with stage B's five changed rows, about 38%. So the passing change is
run again with its reference on three fresh seeds (20261214, 20261215 and 20261216) and stands only if
it passes again.

**Tuning in numbers.** Each PPO method: 3 settings in the six-method search (recurrent MAPPO 2), then 9
rows here. QMIX: 7
settings; PQN-VDN: 10; both in the six-method search, under its opponents and validation set. HAPPO,
TarMAC and PSRO: 6 each. More settings went where no good setting was known for this game. Every
method's search compute is reported in card hours.

## Final Runs

Ten runs per method on seeds 20261201 to 20261210, 250,675,200 steps each, with the opponents and
validation set above. Each method's recipe is its packaged default (its six-method tournament run's
settings, except PQN-VDN's learning rate: see Decisions During the Experiment) with any change that
passed here; the new three take their stage B picks. Each method's ten runs are declared in
`tdm_baselines/final/<method>.yaml` (for example [`final/qmix.yaml`](tdm_baselines/final/qmix.yaml));
each run starts as its own one-seed study on whichever card is free, from a file that differs only in
the seed, the card and the name. A final run is never paused. A crashed run of one of the four PPO methods resumes once from its last save, once their resume with
the ramp is checked exact on this game and listed below; until then, and for the other five methods
always, a crashed run restarts from step 0 on its seed. Each run keeps an actor at every save (at most 409,600 steps apart for the PPO methods) and its
learner state at its picked step and its last step. Every crash is listed below and counted in the
reported compute. All ten runs are reported, whatever they score.

## Scoring

The round robin runs once, after all 90 picks are recorded. No setting, checkpoint or check in this
experiment is chosen from its numbers. Two points are set before the first test game and listed in
Decisions During the Experiment: the games per pairing, and whether a team's score counts its games
against its own method's other teams.

- **The round robin.** All 90 picked teams play each other on test maps 47 to 51, spawn ends exchanged
  between paired games. Each method's score is taken over its ten teams. Elo (`centered-davidson-v1`,
  with draws) and the head-to-head table come from the same games.
- **The canonical tournament.** Each method's best team, picked on validation, plays the six-method
  tournament's field the leaderboard's way. Whether this table is reported, and the games per pairing,
  are decided after the runs and listed below.
- **Score.** Point margin, with wins, draws, losses, kills and deaths beside it. How it is put on the
  0-to-1 scale the protocol's statistics use is fixed before the first test game.
- **Statistics**, with the estimators of rliable (Agarwal and others, NeurIPS 2021): the interquartile
  mean (the average of the middle half of the runs' scores), the optimality gap (how far the runs fall
  short of the top of the scale), the probability of improvement (the chance that a random run of one method beats a random
  run of another) and performance profiles (the share of runs above each score), each with a 95%
  stratified bootstrap interval over runs. A task is one test map.
- **Written before any result.** Expected bands: top, recurrent MAPPO, recurrent IPPO, TarMAC and HAPPO;
  middle, PSRO and feedforward MAPPO; bottom, feedforward IPPO, PQN-VDN and QMIX. A method two or more
  bands from its expected band gets one bug check before it is reported, and then the result stands.

## Compute

| Block | Runs | Estimated Card Hours |
| --- | ---: | ---: |
| Stage A | 108 | 59 to 91 |
| Reruns of passing changes (two to four expected) | 12 to 24 | 7 to 27 |
| Stage B | 54 | 41 to 47 |
| HAPPO's, TarMAC's and PSRO's final runs | 30 | 143 |
| The other six methods' final runs | 60 | 253 |
| Scoring | | 21 to 24 |

Estimates from the six-method search's measured times, on two RTX 5090s, one run per card, in XLA's
normal GPU mode; replaced by measured times as runs finish.

## Decisions During the Experiment

- **PQN-VDN's learning rate**, before its first final run. PQN-VDN's learning rate falls linearly from
  0.00025 to 0 over the whole run, its packaged default. The six-method run followed the same line from
  40.1M on, after falling to about 0 over its first 40.1M, because its search run was planned as 40M.
- **Recurrent IPPO's setting**, after its rerun. In stage A two changes passed: 10 epochs with clip 0.1
  (mean point margin +4.82 against the reference's -1.30, better on all 3 seeds) and 10 epochs alone
  (+4.80). The joined row was rerun because its mean was higher, by 0.02 points, far less than the
  noise; it also had the highest mean of all rows for recurrent and feedforward MAPPO. On the fresh
  seeds it passed again: +5.36 against -1.91, better on all 3 seeds, and at least 10.5 kills a game
  against every team of the validation set. Recurrent IPPO's final runs and PSRO's learner use 10 epochs
  and clip 0.1. The rerun shows that the joined change beats the reference, not that clip 0.1 adds to 10
  epochs. One stage A reference run (seed 20261213) had no confirmed check at 2 kills a game against
  every team; counted at its best margin or left out, the same change passes.
- **Feedforward IPPO's setting.** No change passed stage A; the closest, 10 epochs with clip 0.1, beat
  the reference by 1.77 points. Its final runs use the reference.

Fixed before the first test game and listed here: the score's 0-to-1 scale; the games per
pairing; whether a team's score counts its games against its own method's other teams; the release of
the results data; and whether scoring runs in the GPU's deterministic mode, so it repeats game for
game.

## Limits

- Settings are chosen at 40M steps for runs of 250M, the same for every method.
- Three seeds per setting.
- One network size (128 units) for every method.
- TarMAC uses one message round.
- Each row changes one setting, so the search cannot see two changes working together, except in the
  joined epochs-and-clip row.
- The six older methods meet their own method's earlier saves in training; the new three meet none.
- A ramp matched to steps assumes every method learns at about the six-method search's pace.
- Every learned opponent here trained against the same six recurrent MAPPO teams, the fixed opponents
  of the six-method search; only the scripted bots come from outside that family.
- QMIX's and PQN-VDN's settings were chosen under the six-method search's opponents and its validation
  set, which held scripted bots; every other method's are chosen under the ramp and a validation set of
  learned teams only.
- The final runs' validation set holds later saves of the teams a learner trains against, so it is unseen
  only in its later form; the round robin's opponents were never met at all.
- A round robin of independent runs cannot show what PSRO's paper claims, teams that are hard to exploit.
