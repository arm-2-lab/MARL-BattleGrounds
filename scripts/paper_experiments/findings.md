# Paper Findings

The five experiments of the paper that run as Hydra stages, with their commands,
results and limits. The result tables sit in [`results/`](results/).

## Scenario Evaluation Of The Season 0 Field

**Question.** How often does each member of the Season 0 field win the eight
endgame scenarios? The field is the six tuned baselines, trained to about 250
million game steps, and the three scripted systems. Random, which plays random
legal actions, also plays as a reference: it is not a member of the field. Every
scenario is a nearly lost endgame position with at least one verified winning
line.

**Method.** Each system plays Team A, 256 games per scenario, against the
scenario's own scripted opponent: TDM-Beta in scenarios 3, 5 and 8 and
TDM-Alpha in the others. Every game starts from the exact authored position
with fresh memory. The stage calls `marl_bgs.evaluate_scenarios` for each
system and reads the saved games back with `marl_bgs.load_results`. Wins,
draws and losses come from each game's `outcome` column, which is Team A's
result.

**Command.** From the repository root. The six released models download once
from Hugging Face (`+field=season_0_local models_dir=PATH` plays local copies of
the actor folders instead):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true \
python -m marl_battlegrounds.experiments.scenarios +field=season_0 \
    output_dir=artifacts/paper-experiments/scenario-evaluation-season-0
```

Without `+field=season_0` the stage plays only the built-in systems (TDM-Beta,
TDM-Gamma, TDM-Alpha and Random), which need no downloads. Override `systems=`
to play your own methods, or `num_episodes=32` for a quick check.

**Settings.** 256 games per system and scenario, 128 games side by side, seed
20261006, priority metrics, and XLA's deterministic GPU mode, so a repeat run
plays the same games.

**Hardware and cost.** One NVIDIA RTX 5090, JAX 0.10.1. All 20,480 games
(10 systems × 8 scenarios × 256) took 297 seconds of wall time, including
loading and compiling every system, with a peak of 5.3 GiB of RAM. Each system
took 6 to 44 seconds. An earlier run of the nine field members alone played the
same games: every count in the table below came out identical.

**Results.** Wins · draws · losses for Team A out of 256 games per cell. Rows
follow the Season 0 ranking, then Random.

| Team A | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | Wins of 2,048 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| Recurrent IPPO | 7 · 160 · 89 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 3 · 253 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 7 |
| PQN-VDN | 0 · 256 · 0 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 |
| Recurrent MAPPO | 3 · 200 · 53 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 2 · 254 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 3 |
| QMIX | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 |
| Feedforward IPPO | 8 · 148 · 100 | 1 · 3 · 252 | 2 · 1 · 253 | 0 · 21 · 235 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 11 |
| Feedforward MAPPO | 1 · 208 · 47 | 0 · 0 · 256 | 1 · 3 · 252 | 0 · 1 · 255 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 2 |
| TDM-Beta | 0 · 256 · 0 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 |
| TDM-Gamma | 0 · 256 · 0 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 |
| TDM-Alpha | 0 · 256 · 0 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 |
| Random | 2 · 92 · 162 | 0 · 3 · 253 | 0 · 0 · 256 | 0 · 17 · 239 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 0 · 0 · 256 | 2 |

The same counts are in [`results/season_0_scenarios.csv`](results/season_0_scenarios.csv).

**What it shows.**

- No member of the field solves any scenario. The best cell is 8 wins in 256
  games (feedforward IPPO in Scenario 1), and the field wins 23 of its 18,432
  games.
- In the field, only the four PPO baselines win at all. They sample their
  actions, so their games differ. PQN-VDN and QMIX play greedily and the
  scripted systems have no randomness, so all 256 games in each of their cells
  end the same way.
- Random wins 2 games of Scenario 1, more than feedforward MAPPO, so a win
  there can be luck rather than skill.
- Scenario 1 mostly ends in draws: the three scripted systems and PQN-VDN draw
  every game. In scenarios 5 to 8, every system, Random included, loses every
  game.

**Limits.** These are the Season 0 policies as released; scenario results never
feed their training, selection or Elo. 256 games measure a sampling policy's
win rate to within a few points; a cell with 0 wins does not prove the policy
can never win.

## Why The Field Loses The Scenarios

**Question.** What goes wrong in the games the field loses, and does any
system win Scenario 1 more often than Random?

**Method.** Play the same games again with full metrics, then analyze them.
The `scenarios` stage with `metrics=full` saves 11,192 measurements per game.
The `scenario_analysis` stage reads them back with `marl_bgs.load_results` and
reads each scenario's starting position with `marl_bgs.load_tdm_scenario`. For
every system, scenario and outcome, it counts how often each starting agent
died and used its Ultimate, and how many ticks the games lasted. It compares
each system's wins with Random's using Fisher's exact test.

**Command.** The first command plays; the second only reads saved files:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true \
python -m marl_battlegrounds.experiments.scenarios +field=season_0 metrics=full \
    output_dir=artifacts/paper-experiments/scenario-evaluation-season-0-full-metrics
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds.experiments.scenario_analysis \
    scenarios_dir=artifacts/paper-experiments/scenario-evaluation-season-0-full-metrics \
    output_dir=artifacts/paper-experiments/scenario-evaluation-season-0-full-metrics/analysis
```

**Hardware and cost.** Playing all 20,480 games with full metrics took 367
seconds on one NVIDIA RTX 5090, with a peak of 6.9 GiB of RAM, and saved 524 MB
of results. Every win, draw and loss matched the run above. The analysis took
45 seconds, with a peak of 1.2 GiB of RAM.

**Results.** Counts pool all ten systems, 2,560 games per scenario. HP is the
agent's health at the start.

| Scenario | Games lost | Lost within 2 ticks | Team A deaths in lost games |
| --- | ---: | ---: | --- |
| 1 | 707 (27.6%) | 0 (0.0%) | Hunter (41 HP), Rogue (12 HP) and Priest (1 HP) each 707 (100%) |
| 2 | 2,553 (99.7%) | 2,538 (99.4%) | Priest (8 HP) 2,289 (89.7%), Warrior (35 HP) 556 (21.8%) |
| 3 | 2,553 (99.7%) | 25 (1.0%) | Hunter (1 HP) 2,553 (100%) |
| 4 | 2,516 (98.3%) | 0 (0.0%) | Warrior (47 HP) 2,508 (99.7%), Rogue (36 HP) 8 (0.3%) |
| 5 | 2,560 (100%) | 2,560 (100%) | Priest (1 HP) 2,484 (97.0%), Mage (15 HP) 196 (7.7%) |
| 6 | 2,560 (100%) | 2,552 (99.7%) | Hunter (7 HP) 2,516 (98.3%), Priest (1 HP) 115 (4.5%) |
| 7 | 2,560 (100%) | 2,558 (99.9%) | Warrior (1 HP) 1,479 (57.8%), Priest (5 HP) 1,074 (42.0%), Mage (21 HP) 1,069 (41.8%) |
| 8 | 2,560 (100%) | 2,559 (99.96%) | Priest (1 HP) 2,558 (99.9%), Rogue (21 HP) 2 (0.1%) |

Fisher's exact test finds no system that wins Scenario 1 significantly more
often than Random's 2 of 256: feedforward IPPO's 8 wins give p = 0.106 and
recurrent IPPO's 7 give p = 0.176. The full tables are in
[`results/season_0_scenario_agents.csv`](results/season_0_scenario_agents.csv),
[`results/season_0_scenario_games.csv`](results/season_0_scenario_games.csv) and
[`results/season_0_scenario_reference.csv`](results/season_0_scenario_reference.csv).

**What it shows.**

- In Scenarios 2 and 5 to 8, Team B starts at 19 of the 20 points it needs, so
  one Team A death loses the game. Across all ten systems, Random included,
  that death comes at once: 12,767 of the 12,793 losses in these scenarios
  (99.8%) end within two ticks.
- Team B takes the kill from Team A's weakest agent: the 1 HP Priest in 97.0%
  of Scenario 5 losses and 99.9% of Scenario 8 losses, and the 7 HP Hunter in
  98.3% of Scenario 6 losses. Protecting a weak teammate in the first two ticks
  is the skill the field lacks most.
- The Priest's Ultimate, a full heal, is still on cooldown when these games
  end: it is ready on tick 3 in Scenarios 2 and 5 and on tick 4 in Scenario 8.
  So its rare use (10 of the 2,560 Scenario 2 games, and none in Scenarios 5
  and 8) says that games end early, not that the field avoids the heal.
- In Scenario 1, the three scripted systems and PQN-VDN draw every game the
  same way: Team A loses its Rogue and Priest and never kills the enemy Mage,
  which starts with 4 HP, so neither team reaches 20 points before time runs
  out.
- The field's wins are consistent with luck: no system wins Scenario 1 more
  often than Random by more than chance would explain.

**Limits.** Deaths and Ultimate uses are counted per game, not per tick, so
the tables say whether something happened, not when. The test compares each
system with Random separately and does not correct for making several
comparisons; correcting would only raise the p-values.

## A Language Model Against The Season 0 Field

**Question.** Can a language-model team be evaluated for team coordination
with the same tools as any other System, and how does a small one do?

**Prompt version.** The results below were made with prompt version 1
(`actor-text-v1`). The stage now sends version 2 (see What The Model Reads),
so the commands below play version 2 games, and the stage refuses to add them
to a folder of version 1 games.

**Method.** Serve Qwen3.5-9B with [the Qwen recipe](../../docs/llm.md)'s
vLLM settings (dynamic FP8 weights, thinking off), on one GPU. The
`llm_challenge` stage turns it into a System with `llm.make_system`
(temperature 0, no earlier turns, a bad reply becomes standing still) and
plays it as Team A: against each member of the Season 0 Big Nine, one pair of
games per test map with spawn ends swapped (10 games per member), and 256
games in each scenario. Full metrics are saved. The `llm_analysis` stage then
reads the saved games and model calls back.

**Command.** Start the server first (the recipe's command with the 9B weights,
served as `qwen3.5-9b-fp8`). The simulator runs on the CPU because the model
sets the pace; the last command only reads saved files:

```bash
JAX_PLATFORMS=cpu python -m marl_battlegrounds.experiments.llm_challenge \
    +field=season_0 output_dir=artifacts/paper-experiments/llm-challenge-season-0/games
JAX_PLATFORMS=cpu python -m marl_battlegrounds.experiments.llm_challenge \
    'maps=[]' 'scenario_ids=[1,2,3,4,5,6,7,8]' \
    output_dir=artifacts/paper-experiments/llm-challenge-season-0/scenario-games
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds.experiments.llm_analysis \
    field_dir=artifacts/paper-experiments/llm-challenge-season-0/games \
    scenarios_dir=artifacts/paper-experiments/llm-challenge-season-0/scenario-games/scenarios \
    output_dir=artifacts/paper-experiments/llm-challenge-season-0/analysis
```

**Hardware and cost.** Each experiment had its own NVIDIA RTX 5090 for the
model server, which used at most half its memory. The 90 field games took 3.5
hours (12,503 seconds) and saved 1.2 GB, mostly replays. The 2,048 scenario
games took 22 minutes, with a peak of 2.4 GiB of RAM, and saved 60 MB. The
analysis took 3 seconds, with a peak of 1.1 GiB of RAM.

**Results.** The team won none of its 90 field games (13 draws, 77 losses)
and none of its 2,048 scenario games (36 draws). Per game against each member
([season_0_llm_field.csv](results/season_0_llm_field.csv)):

| Opponent | W · D · L | Margin | Kills | Deaths |
| --- | :---: | ---: | ---: | ---: |
| Recurrent IPPO | 0 · 0 · 10 | -20.7 | 0.1 | 11.2 |
| PQN-VDN | 0 · 2 · 8 | -16.4 | 0.0 | 9.3 |
| Recurrent MAPPO | 0 · 0 · 10 | -20.5 | 0.0 | 11.2 |
| QMIX | 0 · 2 · 8 | -16.0 | 0.4 | 9.4 |
| Feedforward IPPO | 0 · 2 · 8 | -18.0 | 0.0 | 9.5 |
| Feedforward MAPPO | 0 · 0 · 10 | -20.8 | 0.0 | 12.0 |
| TDM-Beta | 0 · 3 · 7 | -15.1 | 0.0 | 10.9 |
| TDM-Gamma | 0 · 0 · 10 | -20.5 | 0.0 | 12.1 |
| TDM-Alpha | 0 · 4 · 6 | -14.2 | 0.0 | 11.7 |

In the scenarios it drew 3 games of Scenario 1 and 33 of Scenario 4 and lost
all others ([season_0_llm_scenarios.csv](results/season_0_llm_scenarios.csv)).

The evaluation itself ran smoothly
([season_0_llm_answers.csv](results/season_0_llm_answers.csv)). Of 62,629
field replies, 99.97% were legal actions: 16 were cut off at the 64-token reply
limit while padding the JSON with blank space, and 2 chose a masked action.
9,609 of 9,610 scenario replies were legal; the one failure was the same cut-off. A field reply read 4,038 tokens
and wrote 12.1 on average; a call took 1.67 seconds at the median, and a tick
of two games side by side 1.70 seconds.

The metrics show how its teamwork differs from its opponents' in the same 90
games ([season_0_llm_behavior.csv](results/season_0_llm_behavior.csv)):

| Per game | Language model | Its opponents |
| --- | ---: | ---: |
| Basic uses | 31.2 | 271.1 |
| Damage done | 152 | 1,526 |
| Ticks with two or more teammates dealing damage | 1.1 | 45.7 |
| Kills shared by two or more agents | 0.0 | 8.5 |
| Mean distance between teammates (units) | 6.8 | 2.9 |
| Deaths | 10.8 | 0.06 |
| Deaths inside its own Red Zone | 7.3 | 0.0 |

In 94.7% of its field answers the model chose no attack, and 98.8% of its
moves went east, toward the enemy's side in its own frame (61.8%), or north
(37.0%). It answers legally but rarely attacks, almost never attacks together,
spreads out and dies mostly at its own spawn. These results come from one
small model with 8-bit weights, thinking off and no memory of earlier turns;
they do not show how well larger reasoning models coordinate.

### What The Model Reads

The `llm_prompt` stage shows the exact text a language-model team reads. It
lets TDM-Alpha play TDM-Beta on test map 47 and stops at tick 12, the first
tick at which a Team A agent may hit an enemy. It puts that agent's permitted
view in the model's frame, with its own team on the left, using `team_on_right`
and `mirror_team_view`, as `llm.make_system` does. Then it renders the view with
`llm.format_actor_view`. No model is called:

```bash
JAX_PLATFORMS=cpu python -m marl_battlegrounds.experiments.llm_prompt \
    output_dir=artifacts/paper-experiments/llm-challenge-season-0/prompt
```

On the CPU it took 74 seconds while other work shared the machine (about 60
seconds when idle), mostly compiling the two scripted Systems, with a peak of
3.0 GB of RAM. The text ([llm_prompt.txt](results/llm_prompt.txt))
uses text version `actor-text-v2` and has 92 lines and 14,611 characters. It
covers the rules, the map's obstacles, who the agent is ("You are ally_0, the
Mage."), every visible unit under its roster name and the legal menu
([llm_prompt.json](results/llm_prompt.json)): Team A's Mage may stay or move in
eight directions, and may skip combat, use its Ultimate or hit enemy 2 with its
Basic ability. On a GPU the unit positions can differ in their last digits,
because floating-point arithmetic rounds differently there; the committed
example is the CPU's output.

## Zero-Shot Coordination With Recurrent IPPO

**Question.** How well does the Season 0 winner, recurrent IPPO, play with a
teammate it never trained with, and what makes a stranger a good fit?

**Method.** The `zero_shot_coordination` stage builds 31 teams with
`marl_bgs.team`: recurrent IPPO alone, and recurrent IPPO with one class at a
time (Mage, Warrior, Hunter, Rogue or Priest) played by the same class's agent
from another Season 0 method (PQN-VDN, recurrent MAPPO, QMIX, feedforward IPPO
or feedforward MAPPO) or by Random, which plays random legal actions and marks
the floor. Every method only ever had copies of itself as teammates in
training, so every newcomer is a true stranger. Each
team plays every member of the Season 0 field with `marl_bgs.evaluate`: 200
games per opponent on maps 47 to 51, spawn ends exchanged within each pair,
full metrics, and the same seed for every team, so game N of every team starts
from the same opponent, map, spawn end and seed. The games themselves differ,
because a different agent plays. The `zero_shot_coordination_analysis` stage
compares each team with recurrent IPPO alone game start by game start, and
resamples whole seed pairs within each opponent for 95% intervals. Slot value
kept places each stranger between recurrent IPPO's own agent (1) and Random
(0) by point margin, with an interval from the same draws. Fit drivers asks
which measure's change follows the cost within each class: the rank
correlation over the five strangers in each class, averaged over the classes,
with an exact permutation test. Home spacing reads the Season 0 tournament,
replayed with full metrics by the `tournament_analysis` stage, through the
same `results.system_metrics` call: how far each class stands from its own
teammates in its own team's games. Distance beyond home is how much farther a
stranger stands from recurrent IPPO's agents than that.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true \
python -m marl_battlegrounds.experiments.zero_shot_coordination \
    output_dir=artifacts/paper-experiments/zero-shot-coordination-season-0
python -m marl_battlegrounds.experiments.zero_shot_coordination_analysis \
    zsc_dir=artifacts/paper-experiments/zero-shot-coordination-season-0 \
    home_games_dir=artifacts/paper-experiments/tournament-analysis-season-0/full-metrics-timed/out/replay/tournament_20261007_095226_96f9aa8c5e85 \
    output_dir=artifacts/paper-experiments/zero-shot-coordination-analysis
```

`home_games_dir` is the replay folder the `tournament_analysis` stage made
and printed.

**Cost.** One RTX 5090, 128 games side by side, XLA's deterministic mode: 279
runs and 55,800 games took 39 minutes of `evaluate` time (8.4 seconds per run
of 200 games with full metrics on average, loading and compiling included),
and 3.0 GB on disk. The analysis takes about 37 seconds on the CPU, home
spacing included.

**Results.** Slot value kept, by the stranger's method and class, with 95%
intervals of about ±0.02 (Mage) to ±0.06 (Rogue); recurrent IPPO alone scores
+4.42 points per game against the nine members. The last column is each
method's mean change in point margin per game over its five classes.

| Stranger (Season 0 Elo) | Mage | Warrior | Hunter | Rogue | Priest | Mean change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recurrent MAPPO (1,304) | 0.91 | 0.89 | 0.90 | 0.96 | 0.83 | −1.17 |
| QMIX (1,300) | 0.81 | 0.83 | 0.71 | 0.61 | 0.90 | −2.07 |
| Feedforward IPPO (1,240) | 0.69 | 0.60 | 0.73 | 0.67 | 0.73 | −3.23 |
| Feedforward MAPPO (1,140) | 0.57 | 0.59 | 0.64 | 0.55 | 0.67 | −4.08 |
| PQN-VDN (1,358) | 0.60 | 0.54 | 0.55 | 0.59 | 0.69 | −4.11 |
| Random: margin change | −16.85 | −8.17 | −7.91 | −5.03 | −14.52 | −10.49 |

**Findings.**

1. **Strangers cost points, and some break the team.** 24 of the 25 learned
   strangers lower the margin, with 95% intervals entirely below zero (−0.8 to
   −7.2 points per game), and keep 54-91% of the slot's value; recurrent
   MAPPO's Rogue is within noise of no cost (0.96). A Mage from PQN-VDN or
   either feedforward method, or feedforward MAPPO's Priest, turns the Season 0
   winner into a team that loses more games than it wins (499-602-699,
   468-518-814, 500-603-697, 682-342-776). The cost holds across
   opponents: 44 of the 45 learned strangers' mean changes against single
   opponents are below zero.
2. **The Mage and Priest slots matter most.** A random Mage or Priest costs
   16.9 and 14.5 points per game, a random Rogue 5.0, and the learned
   strangers cost most there too: 4.8 points per game on average in the Mage
   slot and 3.4 in the Priest slot, against 1.6 in the Rogue slot.
3. **Strength alone does not decide fit.** Recurrent MAPPO, recurrent IPPO's closest
   relative, fits best in every class but the Priest: its actor has the same
   network, inputs, per-agent reward, memory and discount, and only its
   training critic and its seed differ. Without PQN-VDN, the other four line up
   exactly by Season 0 Elo; PQN-VDN, second in Season 0, ties with feedforward
   MAPPO for the largest cost (−4.11 and −4.08 points per game).
4. **Where a stranger stands follows its cost.** Within each class, the farther
   a stranger stands from its teammates, the more points the team loses (mean
   rank correlation −0.84; exact p = 0.00002 from shuffling strangers within
   every class), and the worst fits also deal less damage (+0.82), receive less
   useful healing (+0.78), help fewer kills (+0.76) and take less damage
   (+0.76). The link is closest for the classes whose aura or heals reach only
   nearby teammates (Mage and Priest −1.0, Warrior −0.9) and loosest for the
   Rogue (−0.5), which stands farthest from its teammates (3.98 map units on
   average, against 3.04 to 3.57 for the other classes) yet costs the least.
5. **Strangers stray from their own team's spacing, in the exact order of
   their cost.** Against the distance the same class keeps from its own
   teammates in its own Season 0 games, recurrent MAPPO's strangers stand 0.08
   map units closer to their new teammates, QMIX's 0.02 farther, feedforward
   IPPO's 0.30, feedforward MAPPO's 0.58 and PQN-VDN's 1.37: the order of their
   costs (−1.17, −2.07, −3.23, −4.08, −4.11). Recurrent IPPO's own agents match
   their Season 0 spacing within 0.03. PQN-VDN is the tightest team in Season 0
   (2.45 map units apart, against recurrent IPPO's 2.66), yet its agents keep
   close only to agents like themselves.
6. **The Mage's aura shows the cost.** A stranger Mage's aura covers its
   teammates less, in exactly the order of its cost (recurrent IPPO's own Mage
   0.56, recurrent MAPPO's 0.46, QMIX's 0.43, feedforward IPPO's 0.36,
   PQN-VDN's 0.34, feedforward MAPPO's 0.27).
7. **The host does not adapt.** The four kept agents keep their spacing
   within 0.12 map units, and
   recurrent IPPO's Priest gives 19 of the 20 non-Priest strangers less useful
   healing per point of damage taken than it gave its own agent (Mage: 0.71
   for its own, 0.55 to 0.63 for strangers).

The full results are in `results/zero_shot_coordination_teams.csv` (every
team and class, with each measure, its change and intervals, the slot value
kept with its interval, the Priest's useful healing per point of damage taken,
the home spacing and the distance beyond it),
`results/zero_shot_coordination_classes.csv` (finding 2),
`results/zero_shot_coordination_newcomers.csv` (findings 3 and 5),
`results/zero_shot_coordination_fit_drivers.csv` (finding 4),
`results/zero_shot_coordination_by_opponent.csv` (each method's change against
each opponent) and `results/zero_shot_coordination_map_ends.csv` (every team's
margin, results and each class's distance to its teammates against each
opponent on each map and spawn end).

**Scope.** One host, one checkpoint per method, maps 47 to 51 and one seed per
team: the intervals describe repeated games of these fixed teams.

## Human Games Against And Alongside Recurrent IPPO

**Question.** How does the Season 0 winner, recurrent IPPO, do against an
expert human (the Expert Human below), and how does it do with one as a teammate it never trained with?

**Method.** A person plays Team A in the BattleClient, in Agent POV: it shows
only what Team A sees, the same shared team view recurrent IPPO's agents get.
One series of six games has two parts:

- **Against (games 1 to 3).** The person controls all five Team A agents.
  Recurrent IPPO plays Team B.
- **Alongside (games 4 to 6).** The person controls the Mage, and recurrent IPPO
  controls the other four Team A agents. PQN-VDN, second in Season 0, plays
  Team B. Recurrent IPPO never trained with this teammate.

Each part plays one game per entry in its `maps`, here maps 47, 48 and 49, with
Team A starting on the left, the right and the left. A part's game N uses the
seed `seed + N - 1`. Before each game the stage prints the game and waits for
Enter. It then opens the BattleClient (`marl_bgs.battle_client`); play the game
to its end and press Exit. The stage checks that the replay is the scheduled
game (map, spawn end, seed and the person's slots) and measures it with
`marl_bgs.replay_metrics`, which gives the same full metrics `evaluate` saves.

**Every game counts.** The first replay in a game's folder that was played to
its end is the game, and a later replay never replaces it, so do not use
Restart. Run the same command again to continue the series: recorded games are
skipped, a finished game that was not measured yet is measured, and a game
exited before its end is opened again.

**Command.** From the repository root. The recorded series used local copies of
the six actor folders, so continuing it names them with `field=season_0_local`
and `models_dir`; a new series can leave out both and use the released models:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true \
python -m marl_battlegrounds.experiments.human_games field=season_0_local \
    models_dir=artifacts/paper-experiments/hyperparameter-search \
    output_dir=artifacts/paper-experiments/human-games
```

`parts` holds each part's settings: `kind`, `opponent` and `partner` (names in
`systems`, the Season 0 field by default), `player_class`, `maps` and `seed`.
Override them to play your own series, for example
`parts.1.player_class=warrior`; to play against or beside your own method, add
it to `systems`. A series keeps its first launch's settings in `settings.json`
and refuses other settings, so use a new `output_dir` for another series and,
when the same person plays again, seeds no earlier game used.

**Settings.** First to 20 points, 300 ticks and Red Zone depth 5.0, the same
rules as the tournament. The against games use seeds 20261007 to 20261009; the
alongside games use 20261013 to 20261015. Both were declared before the games
were played.

**Saved files.** Each game has its own folder, for example
`game-04-alongside-map-47-left/`, holding every replay attempt,
`full_metrics.csv` (the game, its replay and the replay's digest, then all
11,192 metric columns; an empty cell is a metric that does not apply) and
`game.json`: the game, its seed and Systems, the controller and class of every
slot (`"Manual"` is the person), every attempt and whether it was played to
its end, how long the BattleClient was open, and Team A's result, points, point
margin (`score_difference`), kills, deaths and game length. The series folder
holds `settings.json`, `results.csv` and `results.md`.

**Hardware and cost.** One NVIDIA RTX 5090 and JAX 0.10.1. Measuring the three against games with
`marl_bgs.replay_metrics` on the CPU took 16 seconds, including loading, with a
peak of 1.8 GB of RAM. Each replay takes 15 to 29 MB.

**Results.** Team A is the person's team. Point margin, kills and deaths are
per game.

| Game | Kind | Opponent | Map | Team A Starts | Result | Points | Kills · Deaths | Ticks |
| ---: | --- | --- | ---: | --- | --- | :---: | :---: | ---: |
| 1 | Against | Recurrent IPPO | 47 | Left | Win | 21 to 4 | 19 · 4 | 233 |
| 2 | Against | Recurrent IPPO | 48 | Right | Win | 20 to 8 | 16 · 8 | 161 |
| 3 | Against | Recurrent IPPO | 49 | Left | Win | 21 to 1 | 12 · 1 | 157 |
| 4 | Alongside | PQN-VDN | 47 | Left | Win | 20 to 3 | 10 · 3 | 94 |
| 5 | Alongside | PQN-VDN | 48 | Right | Loss | 15 to 20 | 15 · 20 | 296 |
| 6 | Alongside | PQN-VDN | 49 | Left | Win | 20 to 13 | 20 · 13 | 289 |

The same rows are in [`results/human_games.csv`](results/human_games.csv).

**Ratings.** After the series, the Expert Human rated each part. Workload uses
the NASA Task Load Index (NASA-TLX; Hart and Staveland, 1988): six scales from
0 to 100, where higher means more demanding and, for performance, 0 is perfect
and 100 is failure. The raw score is the mean of the six: 22.5 for the games
against recurrent IPPO and 57.5 for the games alongside it. For the alongside
games they also rated their AI teammates from 1 (strongly disagree) to 7
(strongly agree) on items adapted from Hoffman's fluency measures for
human-robot teams (Hoffman, 2013 and 2019), plus three items on predictability
and adaptation. Fluency, trust, being understood and predictability scored 1,
the teammates' intelligence 4, and carrying the team 7. In their words,
reworded: against recurrent IPPO, its agents could not anticipate or adapt to
their advances and played poorly as a team once their formation broke, and the
Expert Human made gos by first drawing out recurrent IPPO's Ultimates, then
baiting one agent into attacking, collapsing on it, and using the numbers
advantage to overrun the rest. Alongside it, the agents played all right as
individuals but poorly as a unit, above all in how they moved, which often
left the Mage stuck and unable to join the fight; the Expert Human played the
Mage aggressively and away from the group, almost like a Rogue, to draw the
team into fights, and the policy turned passive instead. Every item and answer
is in [`results/human_games_ratings.csv`](results/human_games_ratings.csv).

### Analysis Of The Human Games

**Question.** Where does the Expert Human's play differ from the policy's, and how
does a team of recurrent IPPO and the Expert Human compare with recurrent IPPO alone?

**Method.** The `human_games_analysis` stage reads the series and the Season 0
tournament's full-metrics replay (see `tournament_analysis`) with the public
API. For the games against recurrent IPPO, it compares every team-level metric
of the person's team with recurrent IPPO's own wins on the same map (240, 214
and 152 games on maps 47, 48 and 49), read from recurrent IPPO's side, and
again with only its wins by 12 points or more (`lead_margin`), a team as far
ahead as the person was (the person won by 17, 12 and 20). Per-game totals
are compared per 100 ticks, because these games ended early; fractions,
averages and distances as they are. A difference is kept when it points the
same way in all three games; with three games, about one in four metrics
that differ only by chance also pass this filter, so it is a screen, not a
test. For the games beside recurrent IPPO, it compares each game with
recurrent IPPO's own 20 games against PQN-VDN on the same map and spawn end.
Each measure also gets the person's lowest and highest game, the middle half
(25th to 75th percentile) of all the reference games, and how many of them
reach the person's level. From the replays it times every Ultimate: the ticks
from the first tick it could be used again, the kills by its user's team in
the next 10 ticks beside an average 10 ticks, and the ticks to the nearest
teammate's Ultimate. To tell coordination from luck, it slides each agent's
Ultimate times by a random number of ticks within its game, 2,000 times from
seed 20261016 (`chance_shuffles`, `chance_seed`), and counts how often
Ultimates still land within 4 ticks of a teammate's. Last, it puts the two
teams of the against games side by side, choice by choice, against what the
game allowed: the replay's action mask says on every tick which targets each
agent could attack or heal and whether it could use its Ultimate (alive, not
stunned, a target in range and in sight). A winning team can look better on
any measure just for winning, so each of these measures is a test against
luck, a comparison of a side with its own normal play, or a choice judged
against what was possible at that moment.

```bash
python -m marl_battlegrounds.experiments.human_games_analysis \
    series_dir=artifacts/paper-experiments/human-games \
    field_dir=artifacts/paper-experiments/tournament-analysis-season-0/full-metrics-timed/out/replay/tournament_20261007_095226_96f9aa8c5e85 \
    output_dir=artifacts/paper-experiments/human-games-analysis
```

It runs on the CPU in about 30 seconds, most of it loading the six replays.

**Findings against recurrent IPPO, head to head** (`head_to_head.csv`: the
two teams in the same three games; every finding points the same way in all
three games).

1. **The Expert Human spends its Ultimates at the right moment.** Recurrent IPPO
   used 51% of its Ultimates (34 of 67) on the first tick it could, the Expert Human
   25% (16 of 63). Once it could, recurrent IPPO's Warrior charged after a
   median of 0 ticks, the Expert Human's after 3.
2. **The Expert Human's Bursts pay; recurrent IPPO's do not.** The Expert Human pressed
   14 of its 15 Bursts with an enemy the Mage could attack, recurrent IPPO none
   of its 16. In a Burst's window, the tick it is pressed and its boosted
   ticks, the Expert Human's Mage dealt 2.9 times its own normal damage per tick,
   recurrent IPPO's 0.8 times. The opening Bursts, at 0-0, dealt 76, 80 and
   95 damage against 22, 0 and 0.
3. **The Expert Human makes gos.** A go, in player-versus-player games such as
   World of Warcraft, is a coordinated push in which a team spends its major
   abilities together to force a kill. 56 of the Expert Human's 63 Ultimates (89%)
   came within 4 ticks of a teammate's. Luck alone puts 60% together (95%
   range 43-75%), and none of the 2,000 slides reached 89%; recurrent IPPO's
   38 of 67 (57%) is what luck gives (61%; 77% of the slides reached it). The
   expert's Mage pressed Burst within 4 ticks of the Warrior's Charge 15 of 15
   times, recurrent IPPO's 3 of 16, and when one of the Expert Human's agents used its
   Ultimate, 1.2 teammates on average could also have used theirs, against 0.6.
4. **The Expert Human's Priest heals with care.** When it could heal a hurt teammate
   (missing at least the 8 health one Basic heal restores), 94% of its Basic
   heals went to one, against 81%. A teammate falling to 35% health or less
   was healed within 2 ticks 74% of the time (32 of 43), against 49% (23 of
   47), partly because the Expert Human's Priest could heal that teammate at once
   more often (60% against 43%; the same in game 1).
5. **Recurrent IPPO's targets.** When the Expert Human's Priest was one of the
   enemies recurrent IPPO could attack, 5% of its attacks went to it, where an
   even pick among those enemies gives 32%; it piled onto the Expert Human's Rogue,
   88% against 37%. The Expert Human spread its attacks about as an even pick would
   (40% and 41%, against 37% and 39%).

**Against recurrent IPPO's own wins** (the screen, per 100 ticks unless
stated; the reference is its wins on the same maps, and its 141 wins by 12
points or more, with the middle half of those games in brackets).

1. **More out of each Burst.** 29-45% of the Expert Human's Mage damage came during
   Burst (13-21%; only 5 of the 141 wins reach 29%), and the Expert Human's Bursts
   dealt 70-86 damage each (40-63).
2. **Recurrent IPPO's fire went to the Expert Human's Rogue.** It took 456 damage
   per 100 ticks, against 272 for recurrent IPPO's Rogue in its lopsided wins,
   and 9 of the team's 13 deaths; the Expert Human's Warrior took less in every game
   (323, 249 and 134, against 367, 589 and 520 on the same maps). Finding 5
   above shows why: recurrent IPPO attacks the Rogue whenever it can.
3. **Differences that are mostly the size of the lead.** The Expert Human's Priest
   took 42 damage per 100 ticks, against 141 in all of recurrent IPPO's wins
   but 88 in its lopsided ones. 15 of the Expert Human's kills, worth 30 of its 62
   points, came inside recurrent IPPO's Red Zone; recurrent IPPO's own
   lopsided wins put more kills there on map 47 (6.4 per 100 ticks, against
   the Expert Human's 0.9) and fewer on maps 48 and 49 (0.6 and 1.0, against 2.5
   and 5.7).

**How unusual the wins are.** In Season 0, recurrent IPPO played 960 games on
maps 47 to 49 and won 606, drew 255 and lost 99, never by more than 14 points
(`human_games_field_record.csv`). The Expert Human won by 17, 12 and 20.

**Findings beside recurrent IPPO.**

1. **Within the range of recurrent IPPO alone.** Margins were +17, −5 and +7,
   with 45 kills and 36 deaths in all. On the same map and spawn end,
   recurrent IPPO's own 20 Season 0 games against PQN-VDN ranged from +17 to
   +20, −13 to +6 and −10 to +18, so each margin lies within that range.
2. **The Expert Human's Mage was at least as productive, in a different style.**
   Per 100 ticks over the three games, it helped make 4.86 kills against 4.62
   for recurrent IPPO's own Mage, dealt 718 damage against 683, and made 1.03
   kills alone against 0.47. It also died more (1.33 against 1.03), and its
   aura covered teammates less in all three games (39% against 52% of their
   ticks).
3. **Formation.** Over the three games, the Expert Human's Mage stood farther from
   the Hunter, Rogue and Priest (2.8, 3.6 and 2.8 map units, against 2.5, 3.0
   and 2.4 for recurrent IPPO's own Mage in Season 0) and as far from the
   Warrior (2.98 for both).
4. **Ultimates came together only by luck.** With the Expert Human controlling only
   the Mage, 56 of the mixed team's 86 Ultimates (65%) came within 4 ticks of
   a teammate's, as luck alone gives (64%). The coordination in the games
   against recurrent IPPO came with controlling all five agents.

The full tables are in `results/human_games_against_screen.csv`,
`human_games_against_screen_lead.csv`, `human_games_classes.csv`,
`human_games_ultimates.csv`, `human_games_ultimates_together.csv`,
`human_games_head_to_head.csv`, `human_games_alongside.csv`,
`human_games_alongside_screen.csv` and `human_games_field_record.csv`.

**Limits.** The games are played by one expert, who designed the game, so they
show what the policy does against and beside that player, not how people in
general would do. The game
is turn-based with no time limit, so the person can think as long as they like
before each move. In the games against the policy, one person plans all five
agents, while the policy's five agents share one team view but each decides on
its own. A person's moves cannot be repeated, so the saved replays are the
record of the games. The analysis screens about 145 metrics in three games
per kind: it describes these games and shows where the policy falls short
against this player. The person is the developer; they rated their own adaptation to the
policy's agents 5 of 7 and the policy's adaptation to them 1 of 7 (see
Ratings). The policy is frozen, so it can
adapt only through its memory within a game.
