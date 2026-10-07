# Six-Method Hyperparameter Search

This study gives each of the six built-in baselines a tested recipe under one
shared training protocol: recurrent MAPPO, feedforward MAPPO, recurrent IPPO,
feedforward IPPO, recurrent QMIX and recurrent PQN-VDN. Studies 1 and 2 measure
complete recipes on a fixed budget of 40,108,032 transitions per run. Study 3
continues one run per method to 250,675,200 transitions, and the six continued
policies play a round robin on the held-out test maps. The search does not
claim to find a global optimum. Choices made from saved results are listed
with their reasons in Decisions During The Search. Every number comes from a
saved record file, and [Where Everything Is](#where-everything-is) gives the
location of each one.

The commands that prepare, run, watch and analyze the study are in
[the paper experiment scripts](../../scripts/paper_experiments/README.md).

## Question

Which settings give each method a strong start on 5v5 Team Deathmatch, and
which settings deserve more tuning? QMIX and PQN-VDN receive most of the
search, because no good batch size or rollout length was known for them on
this game. The four PPO methods test two learning rates around a known good
recurrent MAPPO learner.

## Game, Maps And Splits

Every game is canonical 5v5 Team Deathmatch with the environment defaults: the
first team to 20 points wins. A game is a draw when 300 ticks pass with neither
team at 20, or when both teams reach 20 on the same tick; a team that reaches
20 on tick 300 wins. A kill is worth 1 point. A kill inside the victim's own Red Zone (depth 5.0) is
worth 2. Our learning team is always Team A; spawn ends alternate between
paired games.

| Split | Maps | Use |
| --- | --- | --- |
| Training | 0-41 | Training games, drawn uniformly |
| Validation | 42, 43, 44, 46 | Choosing checkpoints and recipes |
| Test | 47-51 | Final assessment only; nothing is chosen here |

Map 45 (Split Push) is left out of validation. On it the scripted opponents'
pathing fails, and wins there come from that failure rather than from play.
Whether a policy learns to exploit it depends on the recipe, so it would bias
the comparison.

## Methods

| Method | Source | Code |
| --- | --- | --- |
| Recurrent MAPPO | Mava recurrent MAPPO | [`baselines/ppo.py`](../../src/marl_battlegrounds/baselines/ppo.py) |
| Feedforward MAPPO | The same learner without memory | [`baselines/ppo.py`](../../src/marl_battlegrounds/baselines/ppo.py) |
| Recurrent IPPO | The same learner with a local critic per agent | [`baselines/ppo.py`](../../src/marl_battlegrounds/baselines/ppo.py) |
| Feedforward IPPO | Local critics, no memory | [`baselines/ppo.py`](../../src/marl_battlegrounds/baselines/ppo.py) |
| Recurrent QMIX | Mava `rec_qmix` | [`baselines/qmix.py`](../../src/marl_battlegrounds/baselines/qmix.py) |
| Recurrent PQN-VDN | JaxMARL `pqn_vdn_rnn` | [`baselines/pqn.py`](../../src/marl_battlegrounds/baselines/pqn.py) |

Every change from the donors is listed in the [source ledger](source_reuse.md).
The four PPO methods share one learner,
[`training/learner.py`](../../src/marl_battlegrounds/training/learner.py).

## Reward

Every run adds the combat and support reward to the game's own reward. The
game pays +1 for a win, -1 for a loss and 0 for a draw, to every configured
Team A slot. Built-in score shaping is off.

| Part | Amount | Who receives it in the per-agent form |
| --- | ---: | --- |
| `enemy_death_credit` | +0.05 per enemy death | Each credited agent: every agent whose attack landed on the enemy on the tick it died, and any Priest whose useful healing reached one of those agents on that tick |
| `enemy_red_zone_death_credit` | +0.10 extra per enemy death in its own Red Zone | Each credited agent |
| `enemy_damage_hp` | +0.001 per enemy health point | The agent that dealt the damage (capped so overkill does not pay) |
| `priest_effective_healing_hp` | +0.001 per health point healed | The Priest, effective healing only |
| `own_team_deaths` | -0.05 per teammate death | Every configured agent |
| `own_red_zone_deaths` | -0.10 extra per teammate death in our own Red Zone | Every configured agent |
| `priest_heal_range_pairs` | +0.001 per living Priest and ally pair within Basic range, both outside our Red Zone, per tick | Every configured agent |
| `own_zone_ticks` | -0.002 per tick | Each living agent standing in our own Red Zone |

The first four parts and `own_zone_ticks` belong to the agent who earned them.
The other three are shared: every agent receives the same amount.

**Recurrent MAPPO, recurrent IPPO and feedforward IPPO:**
[`combat_support_reward`](../../src/marl_battlegrounds/training/reward_recipes.py).
Each agent learns from its own reward: the game reward, the shared parts and
its own parts.

**Feedforward MAPPO, QMIX and PQN-VDN:** [`combat_support_team_reward`](../../src/marl_battlegrounds/training/reward_recipes.py).
QMIX and PQN-VDN learn one team value from one team reward, and feedforward
MAPPO's critic predicts one value per game for all five agents (decision 11).
Our learners form a team reward as the average over Team A slots
([`team_task_reward`](../../src/marl_battlegrounds/baselines/qmix.py)). With
the per-agent form, that average would divide each agent's own parts by the
team size. The team form gives every slot the same amount instead: the shared
parts once, plus the team total of every agent's own parts. After the average,
each event pays the team exactly what it pays the per-agent learner it happens to:

| Event In 5v5 | Each PPO Agent | Team Average, Per-Agent Form | Team Average, Team Form |
| --- | --- | --- | --- |
| A teammate dies | -0.05 to every agent | -0.05 | -0.05 |
| Our Mage gets kill credit | +0.05 to the Mage | +0.01 | +0.05 |
| Our Mage damages 100 enemy health | +0.10 to the Mage | +0.02 | +0.10 |
| One agent stands in our Red Zone for one tick | -0.002 to that agent | -0.0004 | -0.002 |
| One Priest and ally pair is in heal range for one tick | +0.001 to every agent | +0.001 | +0.001 |

The game reward counts once in both. One difference remains by design: with
one team reward, every agent's value also gains from a teammate's own events.
For the same reason, a team's total shaped reward per game is larger than one
PPO agent's, because it adds five agents' events. The totals of every part in
every learner update are in the training records (see Curves From 0 To 250M).

**Reward scale (QMIX and PQN-VDN only).** `reward_scale` multiplies the whole
team reward, the game reward and every part above together, before the
learner computes its targets. With `reward_scale = 10`, a win is worth +10,
each kill credit +0.5 and a teammate's death −0.5. The balance between the parts does not
change, and a constant positive scale does not change which actions are best.
It changes only the size of the learning targets: QMIX and PQN-VDN learn raw
Q-values against raw reward targets, so the scale sets how large the reward
differences are compared with the network's starting noise. Logged rewards, game results, validation and selection stay in true
game units. The PPO family has no reward-scale setting because a constant
scale cancels out in it: each update normalizes the advantages by their mean
and spread, and value normalization trains the critic on targets normalized
by a running mean and spread. All PPO runs here use value normalization. Only
the first few updates differ, while the running statistics settle.

## Opponents

Each new training game draws its opponent:

| Share | Opponent |
| ---: | --- |
| 60% | The learner's current weights |
| 20% | One of its 12 newest saved copies, uniformly; one copy is saved every 1,376,256 steps, so the 12 copies span the last 16.5M steps (41% of a 40M run; the 250M continuations keep the same settings, so there they cover the last 7%). Until the first copy exists, this share plays the current weights. |
| 15% | Six fixed recurrent MAPPO models, 2.5% each (below) |
| 5% | The scripted `tdm-alpha` and `tdm-beta`, 2.5% each |

| Fixed Model | Training Steps |
| --- | ---: |
| `recurrent-mappo-combat-support-low-amounts-step-30081024` | 30,081,024 |
| `recurrent-mappo-combat-support-step-35094528` | 35,094,528 |
| `recurrent-mappo-tactical-support-step-40108032` | 40,108,032 |
| `recurrent-mappo-tactical-support-high-amounts-step-25067520` | 25,067,520 |
| `recurrent-mappo-tactical-support-zero-sum-step-35094528` | 35,094,528 |
| `recurrent-mappo-tactical-support-high-amounts-zero-sum-step-40108032` | 40,108,032 |

PPO opponents sample their actions from their policy. QMIX and PQN-VDN
opponents (current weights and saved copies) choose greedily except for the
exploration rate saved with them, which is each Q method's ordinary way of
acting during training. The fixed models and scripted bots are identical for
every method. Identity hashes of every fixed model are recorded in
[`opponents.yaml`](../../scripts/paper_experiments/hyperparameter_search/opponents.yaml).

## Training Protocol

| Setting | Value |
| --- | --- |
| Steps per run | 40,108,032 environment transitions in studies 1 and 2 (divisible by every batch size times rollout length used); study 3 continues six runs to 250,675,200 |
| Training state | Recovery saves every 25 PPO updates or 1,600 Q optimizer steps; a crash or Ctrl+C resumes from the last save |
| Recording | Off. No full metrics or Viewer replays during training; rich evaluation runs afterwards from frozen checkpoints. |
| Hardware | Two RTX 5090s, one run per card |
| GPU mode | XLA's normal mode, the default (see below) |

**Repeating the training.** On a GPU, XLA, the compiler JAX uses, can add the
same numbers in a different order from one run to the next. It may choose
between equally correct versions of an operation by timing them, and some
operations add numbers in whatever order the GPU's threads finish. Because of
rounding, the last digit of a sum can depend on that order
([XLA's GPU determinism page](https://openxla.org/xla/determinism)). In XLA's
normal mode, the default, two runs with the same seed can therefore differ in
the last digit. Deterministic mode (`XLA_FLAGS=--xla_gpu_deterministic_ops=true`)
uses only fixed-order versions, so the same inputs give the same outputs on the
same GPU model and software. All training here ran in normal mode; the
tournament ran in deterministic mode (decision 18). In training, a last-digit
difference now and then changes an action, which changes the game, the data
the learner sees and every later update. Retraining with the same declarations
and seeds therefore does not give the same weights, checkpoints or selected
steps as the runs reported here. We did not measure how far such a retrain
drifts; we expect differences of the kind seen between seeds of one recipe
(see the seed tables of studies 1 and 2). The saved checkpoints do repeat the
tournament game for game ([Repeatability On GPUs](../evaluation/repeatability.md)).

## Validation And Selection

| Setting | Value |
| --- | --- |
| Panel | `tdm-alpha`, `tdm-beta`, `tdm-gamma` and `recurrent-mappo-combat-support-pooled-step-250675200`, a recurrent MAPPO model trained for 250,675,200 steps with this opponent mix |
| Maps | 42, 43, 44, 46 |
| Routine checks | At step 0 and after every 10% of training: 8 games per map per opponent (128 games) |
| Confirmation | The two best routine checks (step 0 excluded) and the final checkpoint, if it is not one of them, each play 16 new games per map per opponent (256 games) |
| Measure | Point margin |
| Selected checkpoint | The confirmed check with the highest point margin represents the run |

`tdm-gamma` and the 250M model are never trained against. The 250M model keeps
strong runs apart after they beat the scripted bots; the bots keep a signal
for weak runs.

**Point margin.** For each game, our points minus the opponent's points. Games
are averaged within each map and opponent pair, and the 16 pairs are averaged
equally. For example, winning 20-14 on one map and losing 12-20 on another
gives -1. Wins are not counted separately, and kill difference is not used,
because Red Zone scoring makes it misleading.

## Search Design

Runs of 40,108,032 transitions (studies 1 and 2, and feedforward MAPPO's
team-reward run) choose the recipes. Study 3 continues one run per method from
its saved 40M state to 250,675,200 transitions, and the six continued policies
play the tournament.

| Study | Runs | What it decides |
| --- | ---: | --- |
| 1, `shapes_and_ppo` | 54 | QMIX and PQN-VDN batch size and rollout length; the PPO methods' learning rates |
| 2, `q_settings` | 30 | One-at-a-time changes around each Q method's chosen shape |
| `ff_mappo_team_reward` | 1 | Feedforward MAPPO's 40M run on the team form of the reward (decision 11) |
| 3, `continue_to_250m` | 6 | Each method's tournament run, continued from its saved state to 250,675,200 transitions; these policies enter the tournament |

**Seeds.** Every recipe in studies 1 and 2 runs on the same three seeds:
12259871, 20261001 and 20261002. Seed 12259871 trained the pooled 250M model.
The search feeds a tournament of the 250M policies, not a comparison of recipe
means, so it has no separate study on fresh seeds. A seed fixes the starting
weights and the random draws; it does not carry results across code or
settings.

**Fair rules for batch size (B) and rollout length (T).**

| Method | Held fixed across shapes | How |
| --- | --- | --- |
| QMIX | 1 optimizer step per 512 new steps, 4.75 training pairs per new step, a target copy every 131,072 new steps, 132,096 replay rows in total | `epochs = B x T / 512`, `update_period = 256`, `buffer_size = 132,096 / B`, `sample_batch_size = 128`, `sample_sequence_length = 20` |
| PQN-VDN | Optimizer steps per new step and a minibatch of 8,192 new steps | `num_minibatches = B x T / 8,192`, `epochs = 4` |
| PPO family | Not searched | B512, T32 |

**Base settings.**

- **QMIX:** `q_lr = 0.00003` (the donor's default), `hard_update = True`,
  `update_period = 256`, `gamma = 0.99`, `eps_min = 0.05` reached after
  2,000,000 steps, `min_buffer_size = 32`, `input_scale = 0.01`,
  `reward_scale = 1.0`, shared parameters, left spawn frame.
- **PQN-VDN:** `q_lr = 0.00025` falling linearly to 1e-10 over the run,
  `max_grad_norm = 1.0`, `epochs = 4`, `memory_window = 4`, `gamma = 0.99`,
  `td_lambda = 0.85`, exploration from 1.0 to 0.01 over the first 5% of the
  run, BatchNorm, `reward_scale = 1.0`, `input_scale = 1.0`, shared
  parameters, left spawn frame.
- **PPO family (`reference`):** `actor_lr = 0.0001`, `critic_lr = 0.00025`,
  `epochs = 4`, `minibatches = 2`, `groups = 2`, `gamma = 0.999`,
  `gae_lambda = 0.95`, `clip_epsilon = 0.2`, `entropy_coefficient = 0.01`,
  `value_coefficient = 0.5`, `max_grad_norm = 0.5`, `adam_epsilon = 0.00001`,
  `input_scale = 0.01`, value normalization on, 512 games, rollout length 32.
  This is the learner of the 250M panel model.

**Study 1 recipes** (3 runs each):

| Method | Recipe | Change from the base | Basis |
| --- | --- | --- | --- |
| QMIX | `b768-t32` | B768, T32 | Middle of the earlier range |
| QMIX | `b768-t128` | B768, T128 | Long rollouts |
| QMIX | `b768-t32-lr-1e-4` | B768, T32, `q_lr = 0.0001` | QMIX was still improving slowly at 20M steps at 3e-5 |
| PQN-VDN | `b768-t32` | B768, T32 | Short rollouts get a fair test with a fixed minibatch |
| PQN-VDN | `b768-t64` | B768, T64 | Middle |
| PQN-VDN | `b768-t128` | B768, T128 | Best rollout length in an earlier one-seed run |
| PQN-VDN | `b512-t128` | B512, T128 | Batch-size contrast |
| Each PPO method | `reference` | None | The known good learner |
| Recurrent IPPO, feedforward MAPPO, feedforward IPPO | `actor-lr-3e-4` | `actor_lr = 0.0003` | Updates were very small; local critics and no memory may want faster learning. Recurrent MAPPO already tested this at 40M steps, within noise. |
| Each PPO method | `critic-lr-1e-3` | `critic_lr = 0.001` | Never tested before |

**Study 2 recipes** (3 runs each, declared after study 1).

Study 1 changed the bases. For QMIX, study 2 builds on B768 T32 with
`q_lr = 0.0001`, not the declared B768 T32 pick: it learned to fight fastest
(7.2 kills per game at 40M, against 5.4) and had the best margin (−10.2),
missing the 2-point rule by 1 point. For PQN-VDN it builds on B768 T128, not
the declared B768 T32 pick, which won by avoiding fights. B768 T128 is the
only shape whose selected checkpoints all came late (36-40M), from runs that
fight. Both bases' study 1 runs serve as study 2's comparison runs on the same
seeds.

| Method | Recipe | Change | Basis |
| --- | --- | --- | --- |
| QMIX | `reuse-x2` | Twice the epochs and `update_period = 512`: 9.5 training pairs per new step, target spacing unchanged | The donor makes 1 optimizer step per 32 new steps; this base makes 1 per 512 |
| QMIX | `gamma-0995` | `gamma = 0.995` | A win 150 ticks away keeps 22% of its value at 0.99 and 47% at 0.995 |
| QMIX | `reward-x10` | `reward_scale = 10.0` | Early Q-value errors are much larger than per-step rewards; JaxMARL scales by 10 |
| QMIX | `target-faster` | `update_period = 128`: a target copy every 65,536 new steps | Faster value propagation |
| PQN-VDN | `explore-20pct` | `eps_decay_fraction = 0.2` | Exploration ended after 2M steps, possibly before the team learned that fighting pays |
| PQN-VDN | `reward-x10` | `reward_scale = 10.0` | The donor's SMAX setting |
| PQN-VDN | `layer-norm` | `normalization = "layer_norm"` | The PQN paper's strongest stability result |
| PQN-VDN | `lr-1e-4` | `q_lr = 0.0001` | Early loss blow-ups in an earlier run came at the peak learning rate |
| PQN-VDN | `gamma-0995` | `gamma = 0.995` | As for QMIX |
| PQN-VDN | `lambda-095` | `td_lambda = 0.95` | Payoffs here arrive 3-25 ticks after the action |

The declared rows QMIX `lr-1e-4` (now in the base) and PQN-VDN `epochs-2` were
dropped. Study 2 has 30 runs.

**Engagement floor (study 2 onward).** A change replaces its base only if it
also averages at least 2 kills per game against each of `tdm-alpha`,
`tdm-beta` and `tdm-gamma` in its confirmation games. Study 1 showed that point
margin alone can reward a recipe that avoids fights.

**Pick rules, declared before launch.** A shape is picked by the highest mean
point margin of its three selected checkpoints. A searched change replaces the
base only if its mean beats the base by more than 2 points and it wins on at
least 2 of the 3 seeds. A method's final recipe is its base, or, when changes pass,
the passing change with the highest mean. Only one change is kept, so the
final recipe always has trained runs that can be continued to 250M. With
three seeds, differences smaller than about 3 points cannot be told apart from
noise.

**Study 3: the 250M continuations.**

Each method's tournament run is continued with `extend_training`. For five
methods it is the method's strongest single run of studies 1 and 2, judged on
point margin, kills and deaths together (decision 10): with three seeds,
recipe means are noisy, and the tournament receives one run per method, not a
recipe. For four of those methods it is the best run of the final recipe; for
QMIX it comes from a recipe that was not chosen as the final recipe (see
Final Recipes). For feedforward MAPPO it is a 40M run of its team-reward
recipe on seed 12259871, trained with study 1's protocol and opponent mix
([`ff_mappo_team_reward.yaml`](../../scripts/paper_experiments/hyperparameter_search/ff_mappo_team_reward.yaml),
decision 11). Each run continues from its full learner checkpoint at
40,108,032 transitions by 210,567,168 transitions to 250,675,200, the length
of the pooled 250M model. Every run also keeps a full learner state at step 0
and at every recovery save (every 25 PPO updates or 1,600 Q optimizer steps),
so a run can be continued from an earlier checkpoint too. All six parents were
selected at 40M. The
continuation keeps the learner, optimizer, replay or recent rows, ongoing
games, memory and random streams. It is one training lineage with its parent,
not a new seed, and its steps continue from the parent's.

| Method | Parent Run: Margin / Kills / Deaths | Parent's Selected Checkpoint |
| --- | --- | ---: |
| PQN-VDN | B768 T128, `reward_scale = 10`, seed 12259871 ⭐: −5.1 / 13.1 / 17.1 | 40,108,032 |
| QMIX | B768 T32, `q_lr = 0.0001`, `update_period = 128` (`target-faster`), seed 12259871 ⭐: −2.8 / 12.8 / 15.1 | 40,108,032 |
| Recurrent IPPO | `reference`, seed 20261001: −8.5 / 11.1 / 15.4 | 40,108,032 |
| Recurrent MAPPO | `reference`, seed 20261002: −7.9 / 8.4 / 14.9 | 40,108,032 |
| Feedforward IPPO | `reference`, seed 20261001: −10.7 / 5.7 / 16.4 | 40,108,032 |
| Feedforward MAPPO | Team reward, `actor_lr = 0.0003`, seed 12259871 ⭐: −8.7 / 9.2 / 15.2 | 40,108,032 |

| Setting | Value |
| --- | --- |
| Inherited | Reward, opponent mix and fixed models, exploration, validation panel and maps, seed pairs, actor retention |
| Routine checks | 20, one every 5% of the added budget (about every 10.5M transitions); the last at 250,675,200 |
| Selection | As before. The parent's routine checks compete for the two confirmation places, and a parent checkpoint keeps its saved confirmation games, so the selected checkpoint can come from before 40M if nothing later beats it |
| PPO and QMIX learning rates | Constant in the parent runs; unchanged |
| PQN-VDN learning rate | Restarts at 0.00021 and falls linearly to 1e-10 at 250,675,200 (below) |
| Code | The same library code as studies 1 and 2 |
| Cost | Measured 23.4 GPU-hours for the six continuations |

**Learning rates in the continuations.** The four PPO methods and QMIX use
constant learning rates, so their continuations keep them. PQN-VDN's rate falls
linearly over a run: its 40M parent went from 0.00025 to about 0 (1.3e-8 after
19,536 optimizer steps), and a continuation does not stretch that decay.
Continuing at the parent's final rate would leave PQN-VDN almost unable to
learn for the remaining 210M transitions. The continuation therefore declares
a new linear decay: it starts at 0.00021 and falls to 1e-10 over its 102,816
optimizer steps (2,142 learning blocks of 4 epochs and 12 minibatches), which
end at 250,675,200 transitions. The starting value is the rate a
PQN-VDN run planned for 250M from the start would have at 40,108,032
transitions, 16% of the way: 0.00025 x (1 - 0.16) = 0.00021. The continuation
thus follows that run's schedule from 40M to 250M; only the first 40M differ,
where the parent's rate fell faster.

**Validation in the continuations.** 20 routine checks, one every 5% of the
added budget (about every 10.5M transitions), the last at 250,675,200, each
with the same 128 games as before. Every 4M, as in studies 1 and 2, would mean
53 checks. Selection is unchanged: the confirmed check with the highest point
margin represents the run. The parent's routine checks compete for the two
confirmation places, and a parent checkpoint keeps its saved confirmation
games, so a run whose continuation never beats its 40M-or-earlier best is
represented by that earlier checkpoint, and the report says so.

**Tournament on the test maps** (after study 3). The six study 3 policies and
the three scripted bots play one round robin on the held-out test maps, which
no training, tuning or checkpoint choice ever used. The field is declared in
[`tournament_250m.json`](../../scripts/paper_experiments/hyperparameter_search/tournament_250m.json)
and runs through the public `run_tournament` API.

| Setting | Value |
| --- | --- |
| Entrants | Each method's selected study 3 checkpoint, `tdm-alpha`, `tdm-beta`, `tdm-gamma` |
| Maps | 47, 48, 49, 50, 51 (test maps) |
| Games | 200 per pair: 40 per map, spawn ends exchanged between paired games; 36 pairs, 7,200 games |
| Rules | First to 20 points, 300 ticks, Red Zone depth 5.0 (as in training and validation) |
| Seed root | 20261004 |
| Play | PPO actors sample their actions; QMIX and PQN-VDN actors act greedily (no exploration), as the library loads them; the scripted bots are deterministic |
| Ratings | The tool's joint Elo fit with draws (`centered-davidson-v1`), field average 1,200, 95% intervals from 5,000 bootstrap resamples; tiers and divisions from [the protocol](../evaluation/protocol.md#leaderboard-tiers) |
| GPU mode | Deterministic (`XLA_FLAGS=--xla_gpu_deterministic_ops=true`), so a repeat run plays the same games, game for game ([Repeatability On GPUs](../evaluation/repeatability.md)) |
| Games at a time | 128, the command's default |
| Cost | 8.2 minutes on one RTX 5090 |

## How The Tournament Works

**Schedule.** Every entrant plays every other entrant once: 36 pairings of 200
games, 7,200 games in all. A pairing plays 40 games on each of the five test
maps. One entrant of the pairing is Team A in all 200 games and the other is
Team B; the games are mirrored 5v5, and Team A and Team B are only labels. The
games come in pairs: the same map twice, once from each spawn end.

**Seeds.** Each pair of games gets its own seed, so a pairing uses 100 seeds and
the tournament 3,600, all derived from the root seed 20261004. The two games of
a pair share their seed. Each tick, a game's random key is built from the root
seed, the game's seed, the tick and the team
([`episode_keys`](../../src/marl_battlegrounds/evaluation/evaluate.py) and
`_action_keys` in
[`policy_execution.py`](../../src/marl_battlegrounds/evaluation/policy_execution.py)).
Both games of a pair therefore give each team the same random numbers tick by
tick; because the spawn ends are swapped, the teams face different situations
and the games still play out differently.

**What a seed changes.** Only the actions of entrants that use randomness:

| Part Of A Game | Uses The Seed? | Why |
| --- | --- | --- |
| Map, roster, spawn positions | No | The canonical game uses fixed spawn pads and the fixed mirrored roster; the environment's reset ignores its random key |
| Combat and movement | No | The simulator draws no random numbers; the same actions always give the same next state |
| `tdm-alpha`, `tdm-beta`, `tdm-gamma` | No | The scripted bots ignore their random key |
| QMIX and PQN-VDN | No | They play greedily with exploration off, always taking their highest-valued legal action |
| The four PPO methods | Yes | They sample every action from their action probabilities, and the seed decides those draws. If a policy gives "attack" 70% and "pull back" 30% at some moment, the seed decides which happens |

**What this means for the results.**

1. Within a run, a game between two entrants that do not sample (QMIX,
   PQN-VDN and the three bots) is the same every time on a given map and spawn
   end. Those ten pairings' 200 games are copies of at most 10 distinct games,
   and the tournament tool marks them as deterministic and gives them no
   sampling interval (see Repeated games under Tournament).
2. Any pairing with a PPO entrant varies from game to game, because the PPO
   side samples. In QMIX against recurrent IPPO, QMIX reacts the same way to
   the same situation, but IPPO does not act the same way twice.
3. A PPO entrant's result is an average over its own random choices. A Q
   method's result reflects one fixed behavior per situation, so a strength or
   a weakness repeats every time it is met. Against a bot, each map and spawn
   end is one game played 20 times. PQN-VDN won all 40 games against every bot
   on maps 48, 49 and 50. On map 47 it drew all 40 against `tdm-alpha` and
   `tdm-beta` and lost all 40 to `tdm-gamma`. On map 51 it won all 20 games
   against `tdm-beta` from one spawn end and lost all 20 from the other;
   against `tdm-gamma` it was the other way round.
4. The tournament is reproducible: the same field file and root seed give the
   same schedule and the same random draws, and in deterministic GPU mode the
   same games, game for game, on the same hardware and software. In XLA's
   normal GPU mode a few percent of games that involve a neural network can
   play out differently between two runs, because GPU floating-point rounding
   can flip a near-tied action choice. This is ordinary floating-point noise,
   not a simulator defect; see
   [Repeatability On GPUs](../evaluation/repeatability.md).

## Decisions During The Search

Study 1's design was fixed before it started. These decisions were made from
saved results, and each is applied in the sections that follow.

| # | Decision | Reason |
| ---: | --- | --- |
| 1 | Every final policy must fight; a recipe or run that wins margin by avoiding fights is not chosen | PQN-VDN's highest-margin shape in study 1 (B768 T32, −9.7) had 0 kills and 0 deaths in every game against `tdm-alpha` and `tdm-beta`: it kept away and the bots never reached it. Point margin cannot tell "lost 0-10 while hiding" from "lost 10-20 while fighting" |
| 2 | Kills and deaths are reported beside every point margin | The same case: margin alone made avoidance look like the best shape |
| 3 | Study 2 picks add an engagement floor: at least 2 kills per game against each scripted bot | Blocks recipes that avoid fights. Every PPO and QMIX recipe of study 1 clears it easily (5-12 kills per game against each bot) |
| 4 | Study 2's QMIX base is B768 T32 with `q_lr = 0.0001`, not the declared B768 T32 pick | The best QMIX recipe of study 1 on margin, kills and kill/death ratio (−10.2, 6.2 kills, 0.38, still climbing at 40M), although B768 T128 won more games (33 against 21); it missed the 2-point rule by 1 point. Its study 1 runs serve as the comparison runs |
| 5 | Study 2's PQN-VDN base is B768 T128, not the declared B768 T32 pick | B768 T32 won by avoiding fights (decision 1). B768 T128 made the most kills of the PQN-VDN shapes and its selected checkpoints all came late (36-40M) |
| 6 | At most one passing change is kept per method, the one with the highest mean | A combination of changes would have no trained run to continue to 250M |
| 7 | No separate study on fresh seeds | It would estimate unbiased recipe means for a comparison of methods. The search instead feeds a tournament, which ranks the final policies directly |
| 8 | Each tournament policy comes from one run continued from its saved 40M state to 250,675,200 transitions, not retrained | The tournament needs each method's strongest policy at the length of the pooled 250M model. Continuing keeps the run's full history and saves 40M transitions per method |
| 9 | Reward scale is searched for QMIX and PQN-VDN only | The PPO family normalizes its advantages and value targets, so a constant reward scale cancels out there (see Reward) |
| 10 | Each continued run is the method's strongest single run, judged by point margin, kills and deaths together, from any recipe of studies 1 and 2 | Three-seed recipe means are noisy, and the tournament receives one run per method, not a recipe. This changes QMIX: `target-faster` seed 12259871 (−2.8, 12.8 kills, 15.1 deaths) replaces `reward-x10` seed 12259871 (−3.9, 12.9, 16.4) |
| 11 | Feedforward MAPPO trains on the team form of the reward with `actor_lr = 0.0003`; its tournament run is seed 12259871 | Feedforward MAPPO was the one method whose study 1 runs peaked early under the per-agent form: 7 of its 9 runs selected a check at 8-24M (study 1 finding 4). Its critic predicts one value per game for all five agents, and the team form pays every agent the same team amount (see Reward). `actor_lr = 0.0003` was its best study 1 change: +1.1 and better on all three seeds, though below the 2-point pick rule |
| 12 | PQN-VDN's continuation restarts its learning rate at 0.00021 with a new linear decay to 250M | Its rate had decayed to about 0 by 40M, and a continuation does not stretch that decay. 0.00021 is where a run planned for 250M would be at 40M (see Study 3) |
| 13 | The continuations check every 5% of the added budget (20 checks) and keep the parent's checkpoints eligible | Regular checks over the 210M added transitions find each run's best checkpoint at a bounded cost (checking every 4M, as at 40M, would mean 53 checks); a continuation must beat its earlier best to replace it |
| 14 | Every saved checkpoint of every run is kept for release; past self-play copies are not | The copies sit between saved checkpoints, so they add no distinct strength level to a released population. They stay in the run folders |
| 15 | Each method enters the tournament with its selected study 3 checkpoint | Runs are compared by their checkpoints chosen under the declared rule, never by their final checkpoints by default; each final checkpoint is recorded beside it |
| 16 | The tournament plays 200 games per pair on the test maps | Twice the API default of 100, for narrower rating intervals at a cost of minutes |
| 17 | QMIX and PQN-VDN are evaluated greedily, as the library loads them, and pairs whose games repeat are reported as such | Greedy play is the usual way to evaluate Q-learners (in PyMARL, the QMIX authors' code, [test mode sets epsilon to 0](https://github.com/oxwhirl/pymarl/blob/master/src/components/action_selectors.py)). The tournament tool flags every pair between two deterministic entrants and gives it no sampling interval (see Tournament) |
| 18 | Paper tournament and evaluation runs use XLA's deterministic GPU mode (`XLA_FLAGS=--xla_gpu_deterministic_ops=true`); training keeps the normal mode | Repeat runs then match game for game at no measured cost in evaluation ([Repeatability On GPUs](../evaluation/repeatability.md)) |

## Inputs

The actor features reach about 300 (99% of nonzero values are at most 200).
QMIX's Q-network has no input normalization, so at the donor's input scale of
1.0 its first layer starts with activations about 12 times the intended size
(measured under random play on six maps), which pushes its memory gates to
their limits. With `input_scale = 0.01`, the value the PPO baselines use, they
start near 0.12. PQN-VDN normalizes its
inputs with BatchNorm, so its input scale nearly cancels. QMIX's mixer
normalizes the game state with its own LayerNorm.

## Compute

Measured GPU time of the runs in this document, on RTX 5090s: 44.8 GPU-hours
for study 1, 32.8 for study 2, 0.6 for feedforward MAPPO's 40M team-reward run
and 23.4 for study 3's six continuations (101.6 together), plus 8.2 minutes for
the tournament. Training speeds on one card, in transitions per second of
training, were 12,700-14,000 for QMIX, 10,300-10,600 for PQN-VDN and
17,800-22,100 for the PPO methods; the continuations' wall-clock rates, which
include validation, are in the Study 3 cost table. Measured costs and peak memory per method are reported
with each study's results.

## Results

Every number below comes from saved run records. Each run is represented by
its selected checkpoint. Its two best routine checks by point margin (step 0
excluded) and its final checkpoint each played 256 new confirmation games: 16
per map and opponent, on maps 42, 43, 44 and 46, against the pooled 250M model,
`tdm-alpha`, `tdm-beta` and `tdm-gamma`. The one with the highest confirmation
margin is the selected checkpoint, and the tables report its 256 games. In 16 of
the 84 runs of studies 1 and 2 this is not the best routine check. Margin is our points minus
theirs per game, with every map and opponent weighted equally. Kills are our
kills per game and deaths are the opponent's kills per game. Red Zone kills are
points minus kills. W-D-L counts wins, draws and losses over all confirmation
games (768 per recipe, 256 per run); draws follow the game rule above. ⭐ marks seed 12259871, the seed that trained the pooled 250M model.
Margin alone can hide fight avoidance, so kills and deaths are reported beside
it everywhere.

### Where Everything Is

Every path below is relative to the search root,
`artifacts/paper-experiments/hyperparameter-search/` in this checkout. That
folder is kept locally, outside Git, and will be released with the trained
models. Every declaration is in
[`scripts/paper_experiments/hyperparameter_search/`](../../scripts/paper_experiments/hyperparameter_search/).
A run folder is `<study folder>/study/cases/<case>/run/`, where `<case>` is
`<method>-<recipe>--seed-<seed>`, for example
`shapes-and-ppo/study/cases/mappo-reference--seed-20261002/run/`.

| Stage | Declaration | Run Folders | Result Tables |
| --- | --- | --- | --- |
| Study 1 | `shapes_and_ppo.yaml`, picks `shapes_and_ppo_picks.yaml` | `shapes-and-ppo/study/cases/` (54 runs) | `results/shapes-and-ppo/` |
| Study 2 | `q_settings.yaml`, picks `q_settings_picks.yaml`, analysis grouping `q_settings_grouping.yaml` | `q-settings/study/cases/` (30 runs) | `results/q-settings/` |
| Feedforward MAPPO's 40M run | `ff_mappo_team_reward.yaml` | `ff-mappo-levers/study/cases/ff_mappo-team-reward--seed-12259871/run/` | `results/250m/` (its parent segment) |
| Study 3 | `continue_to_250m.yaml` | `250m/study/cases/` (five methods); feedforward MAPPO's continuation ran in its own folder, `ff-mappo-250m/study/cases/ff_mappo-team-reward--seed-12259871/run/` | `results/250m/` |
| Tournament | `tournament_250m.json` | `tournament-250m/runs/tournament_20261005_082220_ae54a3f293be/` (field `tournament-250m/field.json`, log `tournament-250m/tournament.log`) | `results/tournament-250m/` |
| Teamwork in the tournament | the `tournament_analysis` stage's packaged settings | `../tournament-analysis-season-0/full-metrics-timed/` (replay and analysis) and `../tournament-analysis-season-0/priority-only-timed/` (the timing reference) | [`season_0_teamwork.csv`](../../src/marl_battlegrounds/experiments/results/season_0_teamwork.csv), [`season_0_teamwork_by_class.csv`](../../src/marl_battlegrounds/experiments/results/season_0_teamwork_by_class.csv), [`season_0_teamwork_screen.csv`](../../src/marl_battlegrounds/experiments/results/season_0_teamwork_screen.csv) |
| Fixed opponents | `opponents.yaml` | `opponents/models/` (the six training models and the pooled 250M model) | – |

**The six tournament policies.** Each lineage is its 40M run followed by its
continuation. The selected actor folder is the tournament entrant; it loads
with `marl_bgs.load_method(path)`.

| Method | 40M Run | Continuation | Selected Actor |
| --- | --- | --- | --- |
| Recurrent IPPO | `shapes-and-ppo/study/cases/ippo-reference--seed-20261001/run/` | `250m/study/cases/ippo-reference--seed-20261001/run/` | `250m/study/cases/ippo-reference--seed-20261001/run/actors/ippo_actor_step_000250675200_76b55aa1d72003c22020d13872730d9cf6d96b50a6a936026287b7a35f0f3328/` |
| PQN-VDN | `q-settings/study/cases/pqn_vdn-b768-t128-reward-x10--seed-12259871/run/` | `250m/study/cases/pqn_vdn-b768-t128-reward-x10--seed-12259871/run/` | `250m/study/cases/pqn_vdn-b768-t128-reward-x10--seed-12259871/run/actors/pqn_vdn_actor_step_000240156672_8c1e52fbccfb238801b31041eb4df87cf6df38c62cc458e379c00d6224036f30/` |
| Recurrent MAPPO | `shapes-and-ppo/study/cases/mappo-reference--seed-20261002/run/` | `250m/study/cases/mappo-reference--seed-20261002/run/` | `250m/study/cases/mappo-reference--seed-20261002/run/actors/mappo_actor_step_000219103232_01f7e47402a9ba3e6a17a302ad9f21e680647cbb75e0186c7f1ef12f67303174/` |
| QMIX | `q-settings/study/cases/qmix-b768-t32-lr-1e-4-target-faster--seed-12259871/run/` | `250m/study/cases/qmix-b768-t32-lr-1e-4-target-faster--seed-12259871/run/` | `250m/study/cases/qmix-b768-t32-lr-1e-4-target-faster--seed-12259871/run/actors/qmix_actor_step_000219095040_c9a2ada94944055f57ebfafdc87245954bd270c603a48bbe1a634bf2761cf499/` |
| Feedforward IPPO | `shapes-and-ppo/study/cases/ff_ippo-reference--seed-20261001/run/` | `250m/study/cases/ff_ippo-reference--seed-20261001/run/` | `250m/study/cases/ff_ippo-reference--seed-20261001/run/actors/ff_ippo_actor_step_000219103232_83cc8f75bd015ea02ec977a4e446afbe3a4f247fddb604c42b60f8edcebc1952/` |
| Feedforward MAPPO | `ff-mappo-levers/study/cases/ff_mappo-team-reward--seed-12259871/run/` | `ff-mappo-250m/study/cases/ff_mappo-team-reward--seed-12259871/run/` | `ff-mappo-250m/study/cases/ff_mappo-team-reward--seed-12259871/run/actors/ff_mappo_actor_step_000208568320_f2b2650e9f35e96296d8b70bb28043dd551cc11abd382c75b366e9e24528a6f4/` |

**Files in every run folder.**

| File Or Folder | Contents |
| --- | --- |
| `run_details.json` | Every setting of the run (`config`), its source identity, and for a continuation the `continuation` record naming the parent checkpoint and the declared changes |
| `status.json` | Final state, training and wall seconds, transitions per second, peak JAX memory and process RAM |
| `training_updates.jsonl` | One row per learner update: losses, learning rate, gradient norms, the eight reward parts and the training games that finished in that update (`opponent_results`) |
| `validation_results.json` | Every routine and confirmation check: step, checkpoint id, and one cell per map and opponent (wins, draws, losses, points, kills, game length) |
| `selection.json` | The selected checkpoint and its 256 confirmation games |
| `validation/<purpose>_<method>_actor_step_<steps>_<id>/opponent_0N_<name>/validation_*/` | The raw games of each check against each opponent: `episodes.csv` and `priority_metrics.csv`, one row per game |
| `actors/<method>_actor_step_<steps>_<checkpoint id>/` | Every exported actor; each loads with `marl_bgs.load_method` |
| `checkpoints/` | Full learner states (learner, optimizer, games, memory and random streams) for continuing training |
| `past_copies/` | The saved self-play copies the run trained against |
| `learning_curve.csv`, `learning_curves.png`, `run_summary.md` | The run's own curve and summary |

**Model library.** Every saved checkpoint of every run is also filed by
method, study, recipe, seed and step under
`artifacts/checkpoint-pool/library/hyperparameter-search/`. Its `catalog.csv`
(and `catalog.json`) has one row per checkpoint with its scores, identities,
library path (`actor_path`) and run path (`source_actor`); `tournament_policy`
is `True` for the six tournament policies.

**Result tables.** Each `results/` folder is rebuilt from the saved records by
`results/make_results.py`; no table value is typed in by hand. The written
findings of studies 1 and 2 are in their `FINDINGS.md`. The folder's
`README.md` explains every column and shows how to load the files in pandas.

| File | One Row Per | Contents |
| --- | --- | --- |
| `runs.csv` | Run | Settings, run folder (`run_dir`, and `parent_run_dir` for study 3), selected checkpoint and actor folder (`selected_actor`), confirmation results overall and per opponent, speed, time, memory, library path |
| `recipes.csv` | Recipe | Means over the three seeds, per-seed margins, per-opponent results (studies 1 and 2) |
| `confirmation_cells.csv` | Run, map and opponent | The selected checkpoint's 16 confirmation games per cell |
| `routine_checks.csv` | Run, check, map and opponent | Every check's cells, step 0 included |
| `recipe_check_curves.csv` | Recipe and check | Margin, kills and deaths per game at each check, mean over seeds (studies 1 and 2) |
| `lineage_curves.csv` | Lineage and check | Margin, kills and deaths per game at every check from step 0 to 250,675,200 (study 3) |
| `training_updates.csv.gz` | Run and learner update | Every scalar of `training_updates.jsonl` |
| `training_games.csv.gz` | Run, update and training opponent group | Training games finished in that update |
| `picks.csv` | Declared pick | Candidates' margins, kills and deaths; gain, seeds won, pass or fail (study 2 adds the engagement floor) |
| `RESULTS.md`, `FINDINGS.md` | – | The tables of this section, per study; the written findings (studies 1 and 2) |
| `figures/` | – | Learning curves per recipe and run; QMIX and PQN-VDN diagnostics; `results/250m/figures/lineage_curves.png` plots margin and kills from 0 to 250M for all six policies |

`results/q-settings/` also holds study 2's six base runs from study 1 (column
`comparison_base`), so it is complete on its own. `results/250m/` holds the
six tournament lineages: each run's rows from its 40M run come first, then its
continuation's, with a `segment` column (`parent` or `continuation`), so one
filter gives a 0-250M record. Its `runs.csv` adds the parent's selection, the
selected checkpoint's segment and the final check, and `tournament_policy`.

| Tournament File | One Row Per | Contents |
| --- | --- | --- |
| `ratings.csv` | Entrant | Rank, tier and division, Elo with interval, W-D-L, score, margin, kills, deaths, mean game length |
| `head_to_head.csv` | Ordered pair | W-D-L, margin, kills, deaths, mean game length, distinct games, the tool's determinism flag |
| `map_results.csv` | Entrant and map | Expected score and W-D-L on each test map |
| `final_policies.csv` | Learned entrant | Rank and rating beside every saved setting of its run, its exploration rate in play (`evaluation_epsilon`: 0 for QMIX and PQN-VDN, empty for PPO) and its file paths |
| `RESULTS.md` | – | The tables of the Tournament section |

The tournament tool's own records stay in the run folder:
`match_results.csv` (every game, with both teams' points, kills, deaths and
game length), `matchup_results.csv`, `map_results.csv`,
`tournament_results.csv`, `tournament_headline_metrics.csv`,
`tournament_games.jsonl` (each game's map, spawn end and teams),
`tournament_config.json` and `run_details.json`.

### Curves From 0 To 250M

For each of the six tournament policies, these measures can be plotted at
every point from step 0 to 250,675,200 transitions. All are in
`results/250m/`; filter one lineage with `case_id` and order by `env_steps`.

**At every validation check** (31 per lineage: step 0, 10 checks to 40M about
every 4M, then 20 checks about every 10.5M; 128 games each, 8 per map and
opponent). Use the rows of `routine_checks.csv` whose `purpose` is
`routine` (step 0 included); `lineage_curves.csv` holds the first three
measures already averaged.

| Measure | How |
| --- | --- |
| Point margin per game | `mean_team_a_score − mean_team_b_score`, averaged over the 16 map and opponent cells |
| Kills per game | `mean_team_a_kills` |
| Deaths per game | `mean_team_b_kills` |
| Points per game, ours and theirs | `mean_team_a_score`, `mean_team_b_score` |
| Red Zone kills per game, ours and theirs | Points minus kills, for each team |
| Wins, draws and losses | `wins`, `draws`, `losses` out of `games` |
| Score (win 1, draw ½) | `score` |
| Game length in ticks | `mean_episode_length` |
| Kill difference | `mean_kill_difference` |
| Any of the above against one opponent or on one map | Filter `opponent` or `map_id` |
| Game returns and per-game distributions | Each check's `priority_metrics.csv` in the run folder (see Files in every run folder): every game's result, return, points, kills and deaths for both teams |

**At every learner update** (15,300 per PPO lineage, 10,200 for QMIX, 2,551 for
PQN-VDN). Use `training_updates.csv.gz`.

| Measure | Column |
| --- | --- |
| Transitions so far | `env_steps`; agent decisions so far, `actor_decisions` |
| Time | `elapsed_seconds`, `wall_seconds`, `training_seconds`, `collection_update_seconds` |
| Learning rate | `learning_rate` |
| Game reward and combat and support reward per agent transition | `task_reward_mean`, `custom_reward_mean` (built-in shaping, `shaping_mean`, is 0 in every run) |
| The eight reward parts | `reward_<part>_recipient_sum` and `reward_<part>_team_mean_sum` for each part in the Reward table, summed over the update's `real_transitions` |
| PPO methods | `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `policy_clip_fraction`, `value_clip_fraction`, `actor_grad_norm`, `critic_grad_norm`, `value_mean`, `value_rmse`, `target_mean`, `target_std`, `normalization_mean`, `normalization_std` |
| QMIX and PQN-VDN | `loss`, `mean_q`, `mean_target`, `epsilon`, `grad_norm_mean`, `grad_norm_max`, `grad_norm_nonfinite`, `used_td_pairs`; QMIX also `replay_rows_per_lane`, `sampled_sequences`, `td_pairs_per_transition`; PQN-VDN also `used_sequences`, `used_prefix_td_pairs` |

The reward parts also give event counts. Divide a part's sum by its amount in
the Reward table: kill credits (`enemy_death_credit`, 0.05), Red Zone kill
credits (0.10), enemy damage in health points (0.001), Priest healing in health
points (0.001), Priest and ally pairs in heal range per tick (0.001), teammate
deaths (−0.05), deaths in our own Red Zone (−0.10) and agent ticks in our own
Red Zone (−0.002). Use `team_mean_sum` for the three shared parts in every run
and for every part in the team-reward runs (feedforward MAPPO, QMIX, PQN-VDN);
use `recipient_sum` for the five own-agent parts in the per-agent runs
(recurrent MAPPO, recurrent IPPO, feedforward IPPO). Divide by
`real_transitions` for a rate per transition.

**Training games** (one row per update and training opponent group). Use
`training_games.csv.gz`: games finished, `wins`, `draws`, `losses`,
`points_for`, `points_against`, `mean_point_margin` and `mean_length`, for
each group: the current weights (`opponent_kind` `self`), each saved past copy,
each fixed recurrent MAPPO model and each scripted bot. For example, the share
of drawn games against its own current weights over time is `draws / games`
for `opponent_kind == "self"`. Kills are not recorded for training games;
deaths come from the `own_team_deaths` reward part.

### Study 1: Shapes And PPO Recipes

Study 1 ran all 54 runs to 40,108,032 transitions with no failed run. It took
22 hours 43 minutes on two RTX 5090s, or 44.8 GPU-hours (planned: 49).

**Recipes** (mean over three seeds):

| Method | Recipe | Mean Margin | SD | Seed 12259871 | Seed 20261001 | Seed 20261002 | Kills | Deaths | K/D | Red Zone Kills (Ours / Theirs) | W-D-L | vs 250M | vs Alpha | vs Beta | vs Gamma |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recurrent MAPPO | `critic-lr-1e-3` | −9.8 | 1.4 | −9.5 | −11.3 | −8.6 | 6.6 | 15.3 | 0.43 | 0.44 / 1.47 | 8-173-587 | −6.2 | −10.2 | −11.7 | −11.0 |
| Recurrent MAPPO | `reference` | −8.8 | 0.8 | −9.5 | −9.0 | −7.9 | 8.3 | 15.2 | 0.55 | 0.27 / 2.24 | 13-143-612 | −9.0 | −8.0 | −9.5 | −8.8 |
| Recurrent IPPO | `actor-lr-3e-4` | −10.9 | 0.5 | −10.4 | −11.0 | −11.2 | 5.6 | 16.8 | 0.33 | 0.77 / 0.42 | 1-178-589 | −6.9 | −11.4 | −12.5 | −12.7 |
| Recurrent IPPO | `critic-lr-1e-3` | −11.2 | 0.1 | −11.3 | −11.1 | −11.2 | 5.5 | 17.0 | 0.32 | 1.10 / 0.77 | 13-164-591 | −8.7 | −10.9 | −12.3 | −13.0 |
| Recurrent IPPO | `reference` | −10.4 | 1.6 | −11.6 | −8.5 | −11.1 | 8.2 | 17.0 | 0.49 | 0.59 / 2.26 | 24-83-661 | −13.7 | −8.2 | −9.9 | −9.9 |
| Feedforward MAPPO | `actor-lr-3e-4` | −9.4 | 1.3 | −9.6 | −10.6 | −8.1 | 6.6 | 15.0 | 0.44 | 0.36 / 1.42 | 3-183-582 | −5.0 | −10.4 | −11.3 | −11.0 |
| Feedforward MAPPO | `critic-lr-1e-3` | −11.3 | 0.4 | −11.1 | −10.9 | −11.7 | 5.0 | 17.0 | 0.29 | 1.08 / 0.36 | 6-178-584 | −7.3 | −11.8 | −12.8 | −13.1 |
| Feedforward MAPPO | `reference` | −10.6 | 0.8 | −10.4 | −11.4 | −9.9 | 5.7 | 14.1 | 0.41 | 0.22 / 2.43 | 0-179-589 | −4.9 | −11.9 | −12.9 | −12.5 |
| Feedforward IPPO | `actor-lr-3e-4` | −11.3 | 0.1 | −11.1 | −11.4 | −11.4 | 4.4 | 16.7 | 0.26 | 1.23 / 0.15 | 3-187-578 | −5.9 | −12.4 | −13.5 | −13.4 |
| Feedforward IPPO | `critic-lr-1e-3` | −11.4 | 0.4 | −11.8 | −10.9 | −11.4 | 5.4 | 16.9 | 0.32 | 0.68 / 0.63 | 0-166-602 | −8.0 | −11.2 | −13.4 | −12.8 |
| Feedforward IPPO | `reference` | −10.9 | 0.3 | −10.8 | −10.7 | −11.2 | 4.8 | 16.0 | 0.30 | 0.58 / 0.28 | 0-194-574 | −4.2 | −12.8 | −13.5 | −13.1 |
| QMIX | `b768-t128` | −11.6 | 0.8 | −10.8 | −11.6 | −12.4 | 5.6 | 16.2 | 0.35 | 1.50 / 2.49 | 33-105-630 | −10.8 | −9.9 | −12.5 | −13.2 |
| QMIX | `b768-t32` | −11.2 | 0.5 | −11.3 | −11.6 | −10.6 | 5.5 | 16.7 | 0.33 | 1.49 / 1.43 | 13-153-602 | −7.3 | −10.8 | −12.7 | −13.8 |
| QMIX | `b768-t32-lr-1e-4` | −10.2 | 1.1 | −11.5 | −9.6 | −9.5 | 6.2 | 16.4 | 0.38 | 1.17 / 1.16 | 21-150-597 | −6.3 | −10.7 | −11.7 | −12.1 |
| PQN-VDN | `b512-t128` | −14.2 | 1.6 | −15.2 | −15.0 | −12.4 | 1.0 | 11.7 | 0.09 | 0.16 / 3.72 | 0-248-520 | −11.7 | −12.2 | −14.0 | −18.8 |
| PQN-VDN | `b768-t128` | −15.3 | 0.7 | −14.5 | −15.7 | −15.7 | 1.8 | 15.4 | 0.12 | 0.40 / 2.13 | 0-197-571 | −9.5 | −15.9 | −17.7 | −18.2 |
| PQN-VDN | `b768-t32` | −9.7 | 0.5 | −10.1 | −9.1 | −9.9 | 0.3 | 5.4 | 0.05 | 0.01 / 4.61 | 0-404-364 | −19.3 | 0.0 | 0.0 | −19.5 |
| PQN-VDN | `b768-t64` | −15.0 | 1.3 | −13.5 | −15.5 | −15.9 | 1.5 | 10.8 | 0.14 | 0.02 / 5.69 | 0-213-555 | −17.0 | −11.7 | −13.0 | −18.1 |

**Kills / deaths per game by opponent:**

| Method | Recipe | vs 250M | vs Alpha | vs Beta | vs Gamma |
| --- | --- | ---: | ---: | ---: | ---: |
| Recurrent MAPPO | `critic-lr-1e-3` | 0.7 / 5.3 | 9.1 / 18.7 | 7.8 / 18.6 | 8.5 / 18.8 |
| Recurrent MAPPO | `reference` | 0.6 / 5.8 | 11.6 / 18.2 | 10.3 / 18.3 | 10.8 / 18.4 |
| Recurrent IPPO | `actor-lr-3e-4` | 1.8 / 8.5 | 7.5 / 19.8 | 6.6 / 19.6 | 6.5 / 19.4 |
| Recurrent IPPO | `critic-lr-1e-3` | 2.3 / 10.9 | 7.3 / 19.2 | 6.5 / 19.3 | 5.8 / 18.8 |
| Recurrent IPPO | `reference` | 3.4 / 14.3 | 10.8 / 18.1 | 9.3 / 17.8 | 9.4 / 17.6 |
| Feedforward MAPPO | `actor-lr-3e-4` | 0.4 / 3.8 | 9.1 / 18.6 | 8.3 / 18.8 | 8.6 / 18.7 |
| Feedforward MAPPO | `critic-lr-1e-3` | 1.8 / 9.0 | 6.6 / 19.7 | 5.8 / 19.7 | 5.7 / 19.4 |
| Feedforward MAPPO | `reference` | 0.5 / 3.7 | 7.9 / 17.5 | 7.0 / 17.7 | 7.4 / 17.4 |
| Feedforward IPPO | `actor-lr-3e-4` | 1.4 / 7.4 | 5.9 / 19.9 | 5.1 / 19.9 | 5.1 / 19.7 |
| Feedforward IPPO | `critic-lr-1e-3` | 1.7 / 9.5 | 7.7 / 19.5 | 6.0 / 19.4 | 6.3 / 19.2 |
| Feedforward IPPO | `reference` | 1.0 / 4.9 | 6.4 / 19.8 | 5.9 / 19.8 | 6.0 / 19.5 |
| QMIX | `b768-t128` | 3.5 / 11.6 | 7.7 / 18.5 | 5.9 / 17.8 | 5.3 / 17.0 |
| QMIX | `b768-t32` | 3.9 / 10.0 | 7.0 / 19.2 | 5.9 / 19.0 | 5.2 / 18.8 |
| QMIX | `b768-t32-lr-1e-4` | 3.3 / 8.2 | 7.7 / 19.2 | 6.9 / 19.3 | 7.0 / 19.0 |
| PQN-VDN | `b512-t128` | 0.5 / 7.6 | 1.3 / 11.5 | 1.1 / 12.7 | 1.2 / 14.8 |
| PQN-VDN | `b768-t128` | 0.6 / 8.4 | 2.6 / 17.5 | 2.0 / 17.9 | 1.9 / 17.6 |
| PQN-VDN | `b768-t32` | 0.3 / 10.1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.8 / 11.4 |
| PQN-VDN | `b768-t64` | 0.5 / 9.8 | 2.0 / 10.2 | 1.2 / 9.9 | 2.2 / 13.1 |

**Picks under the declared rules.** No searched change passed. The rule needs
a mean gain above 2 points and a win on at least 2 of 3 seeds.

Each candidate shows its mean margin / kills / deaths per game. Margin alone
can hide avoidance: see the PQN-VDN shape pick below.

| Pick | Candidates (Margin / Kills / Deaths) | Result | Mean Gain | Seeds Won |
| --- | --- | --- | ---: | ---: |
| PQN-VDN shape | B768 T32 −9.7 / 0.3 / 5.4; B768 T64 −15.0 / 1.5 / 10.8; B768 T128 −15.3 / 1.8 / 15.3; B512 T128 −14.2 / 1.0 / 11.7 | B768 T32 (highest mean) | – | – |
| QMIX shape | B768 T32 −11.2 / 5.5 / 16.7; B768 T128 −11.6 / 5.6 / 16.2 | B768 T32 (highest mean) | – | – |
| QMIX `q_lr = 0.0001` | Base −11.2 / 5.5 / 16.7; change −10.2 / 6.2 / 16.4 | Base kept | +1.0 | 2 |
| Recurrent MAPPO `critic_lr = 0.001` | Base −8.8 / 8.3 / 15.2; change −9.8 / 6.5 / 15.3 | Base kept | −1.0 | 1 |
| Recurrent IPPO `actor_lr = 0.0003` | Base −10.4 / 8.2 / 17.0; change −10.9 / 5.6 / 16.8 | Base kept | −0.5 | 1 |
| Recurrent IPPO `critic_lr = 0.001` | Base −10.4 / 8.2 / 17.0; change −11.2 / 5.5 / 17.0 | Base kept | −0.8 | 1 |
| Feedforward MAPPO `actor_lr = 0.0003` | Base −10.6 / 5.7 / 14.1; change −9.4 / 6.6 / 15.0 | Base kept | +1.1 | 3 |
| Feedforward MAPPO `critic_lr = 0.001` | Base −10.6 / 5.7 / 14.1; change −11.3 / 5.0 / 17.0 | Base kept | −0.7 | 1 |
| Feedforward IPPO `actor_lr = 0.0003` | Base −10.9 / 4.8 / 16.0; change −11.3 / 4.4 / 16.7 | Base kept | −0.4 | 0 |
| Feedforward IPPO `critic_lr = 0.001` | Base −10.9 / 4.8 / 16.0; change −11.4 / 5.4 / 16.9 | Base kept | −0.4 | 0 |

So each PPO method's recipe after study 1 is `reference`; feedforward MAPPO's
recipe adds the team reward and `actor_lr = 0.0003` (decision 11). For the Q methods, study 2
changed the bases (see Study 2 recipes): QMIX B768 T32 with `q_lr = 0.0001`,
PQN-VDN B768 T128.

**Findings.**

1. **No recipe beats any panel opponent at 40M transitions.** No margin is
   positive against any of the four opponents (PQN-VDN B768 T32 scores exactly
   0 against `tdm-alpha` and `tdm-beta` because neither side ever scores), and
   wins are 0-4% of games. The best
   recipe is recurrent MAPPO `reference` (−8.8). It also kills the most bots
   (11.6 per game against `tdm-alpha`) and has the best kill/death ratio
   (0.55).
2. **PQN-VDN avoided fights at every shape.** It makes 0.3-1.8 kills per game,
   while the other methods make 4-8. Its highest-margin shape, B768 T32, had
   0 kills and 0 deaths in every game against `tdm-alpha` and `tdm-beta`: it
   kept away and the bots never reached it. The point-margin pick therefore
   rewarded avoidance, not fighting. Point margin alone cannot tell "lost 0-10
   while hiding" from "lost 10-20 while fighting".
3. **QMIX fights under the team reward.** It makes 5.5-6.2 kills per game, and
   the most Red Zone kills of any method (about 1.5 per game).
4. **Recurrent methods were still improving at 40M; feedforward MAPPO peaked
   early.** Recurrent MAPPO `reference` rose from 5.0 kills per game at 4M to
   8.6 at 40M in its routine checks, with its margin going from −11.8 to −8.0. Recurrent MAPPO `reference` and recurrent IPPO `reference` selected
   their last check on every seed. Feedforward MAPPO selected a check at 8-24M
   in 7 of 9 runs. Feedforward IPPO was mixed: 4 of 9 runs at the last check.
5. **Close scores against the 250M model come from not engaging it.** Across
   the 33 PPO runs, the margin against the 250M model correlates −0.86 with
   kills plus deaths against it. It correlates −0.64 with the mean margin
   against the three bots. The 10 runs that lose to it by only 3-5 points
   barely fight it (0.2-0.9 kills, 3.0-5.5 deaths) and lose worse to the bots
   (−12.9 against −11.1 for the other PPO runs).
6. **Seed spread is large.** Recurrent IPPO `reference` ranges from −8.5 to
   −11.6 across seeds. With three seeds, differences under about 3 points
   cannot be told apart from noise.
7. **The strongest fighter of study 1 is recurrent IPPO `reference` on seed
   20261001:** 11.1 kills per game, kill/death ratio 0.72 and 19 wins in 256
   games, with a margin of −8.5.
8. **On seed 12259871**, recurrent MAPPO `reference`, the same recipe that
   produced the pooled 250M model at 250M transitions, scores −9.5 at 40M
   (7.4 kills, 15.0 deaths) and −8.0 against the 250M model.
9. **Checkpoint selection by point margin can choose a phase of fight
   avoidance.** It did so for PQN-VDN B768 T32 (finding 2) and again in
   study 2 (study 2 finding 2). Kills and deaths are therefore reported beside
   every margin, and study 2 adds an engagement floor to its picks.

**Learning curves.** Margin / kills per game at each routine check, mean over
the three seeds. Checks come every 10% of training (about 4M transitions);
0% is the untrained policy. Exact check steps differ slightly between batch
shapes.

| Method | Recipe | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recurrent MAPPO | `critic-lr-1e-3` | −19.9 / 0.6 | −12.7 / 5.1 | −11.8 / 5.3 | −13.7 / 4.3 | −11.5 / 5.5 | −11.7 / 6.1 | −11.9 / 5.6 | −11.6 / 5.8 | −10.8 / 5.8 | −11.5 / 5.2 | −10.0 / 6.2 |
| Recurrent MAPPO | `reference` | −19.9 / 0.6 | −11.8 / 5.0 | −11.1 / 5.5 | −11.5 / 6.1 | −11.0 / 6.9 | −12.0 / 7.3 | −10.8 / 7.7 | −10.4 / 8.1 | −9.7 / 8.3 | −9.7 / 8.3 | −8.0 / 8.6 |
| Recurrent IPPO | `actor-lr-3e-4` | −19.9 / 0.6 | −11.9 / 3.6 | −11.6 / 3.8 | −11.6 / 4.0 | −11.2 / 4.2 | −11.6 / 4.3 | −11.8 / 4.3 | −11.6 / 5.0 | −11.6 / 5.3 | −11.0 / 5.9 | −11.1 / 6.1 |
| Recurrent IPPO | `critic-lr-1e-3` | −19.9 / 0.6 | −12.7 / 2.9 | −12.7 / 3.3 | −12.1 / 3.7 | −12.0 / 4.1 | −11.8 / 4.1 | −12.4 / 4.3 | −12.1 / 4.8 | −11.6 / 5.3 | −11.3 / 6.0 | −11.4 / 6.2 |
| Recurrent IPPO | `reference` | −19.9 / 0.6 | −12.4 / 3.2 | −11.9 / 3.5 | −11.4 / 4.2 | −11.9 / 4.0 | −11.7 / 4.7 | −11.5 / 5.1 | −11.5 / 5.6 | −10.4 / 6.1 | −10.7 / 7.8 | −10.3 / 8.2 |
| Feedforward MAPPO | `actor-lr-3e-4` | −19.9 / 0.6 | −10.8 / 5.3 | −11.1 / 6.1 | −10.7 / 6.3 | −10.8 / 5.8 | −10.4 / 5.9 | −10.2 / 5.7 | −10.8 / 4.8 | −10.9 / 5.3 | −10.8 / 5.4 | −10.7 / 5.4 |
| Feedforward MAPPO | `critic-lr-1e-3` | −19.9 / 0.6 | −12.9 / 5.9 | −11.9 / 5.6 | −12.0 / 4.5 | −12.8 / 4.2 | −12.3 / 4.4 | −12.4 / 4.7 | −11.7 / 4.7 | −11.7 / 4.8 | −11.9 / 4.6 | −11.6 / 4.7 |
| Feedforward MAPPO | `reference` | −19.9 / 0.6 | −11.9 / 5.4 | −11.9 / 5.6 | −11.2 / 6.0 | −10.7 / 5.5 | −11.0 / 4.7 | −11.8 / 3.9 | −11.8 / 3.8 | −12.0 / 4.0 | −11.7 / 4.2 | −12.2 / 4.1 |
| Feedforward IPPO | `actor-lr-3e-4` | −19.9 / 0.6 | −11.5 / 3.7 | −11.5 / 3.3 | −11.6 / 3.7 | −11.8 / 3.3 | −11.9 / 3.4 | −12.0 / 3.7 | −11.8 / 4.0 | −11.8 / 4.3 | −11.3 / 4.8 | −11.5 / 4.9 |
| Feedforward IPPO | `critic-lr-1e-3` | −19.9 / 0.6 | −12.3 / 3.2 | −12.1 / 3.5 | −11.9 / 4.2 | −12.1 / 4.4 | −11.8 / 4.5 | −11.7 / 4.4 | −11.9 / 5.3 | −11.7 / 5.1 | −11.6 / 5.5 | −11.3 / 5.8 |
| Feedforward IPPO | `reference` | −19.9 / 0.6 | −12.5 / 3.6 | −11.8 / 3.8 | −12.1 / 4.2 | −12.2 / 4.2 | −11.8 / 4.6 | −11.8 / 4.4 | −11.9 / 4.7 | −11.8 / 4.7 | −11.8 / 5.1 | −11.0 / 5.5 |
| QMIX | `b768-t128` | −14.8 / 0.6 | −18.9 / 1.0 | −16.6 / 1.8 | −16.4 / 1.6 | −15.8 / 2.1 | −14.9 / 2.7 | −13.5 / 3.4 | −14.1 / 4.0 | −13.6 / 4.4 | −12.8 / 5.1 | −11.5 / 5.6 |
| QMIX | `b768-t32` | −14.8 / 0.6 | −18.7 / 0.5 | −16.6 / 1.2 | −16.4 / 1.7 | −15.3 / 1.9 | −15.8 / 1.7 | −15.5 / 2.2 | −13.0 / 3.9 | −13.2 / 3.7 | −11.6 / 5.4 | −12.0 / 5.4 |
| QMIX | `b768-t32-lr-1e-4` | −14.8 / 0.6 | −18.2 / 1.1 | −17.2 / 1.2 | −14.9 / 1.7 | −15.2 / 2.1 | −14.6 / 2.3 | −14.4 / 3.2 | −14.0 / 3.6 | −12.2 / 5.0 | −11.4 / 5.8 | −10.0 / 7.2 |
| PQN-VDN | `b512-t128` | −17.3 / 0.6 | −16.3 / 1.0 | −17.6 / 1.5 | −18.1 / 0.7 | −17.0 / 1.0 | −17.6 / 1.0 | −17.7 / 1.2 | −16.6 / 1.4 | −16.5 / 1.2 | −16.0 / 1.9 | −15.3 / 2.7 |
| PQN-VDN | `b768-t128` | −17.3 / 0.6 | −18.4 / 1.3 | −17.0 / 1.3 | −18.3 / 0.5 | −18.3 / 0.8 | −18.3 / 0.6 | −17.8 / 1.4 | −16.8 / 1.1 | −16.9 / 1.2 | −15.4 / 1.7 | −15.6 / 1.8 |
| PQN-VDN | `b768-t32` | −17.3 / 0.6 | −15.0 / 1.1 | −17.3 / 0.3 | −17.0 / 1.3 | −15.4 / 0.3 | −14.5 / 0.7 | −17.4 / 0.4 | −15.6 / 0.5 | −18.1 / 1.1 | −17.3 / 1.1 | −16.2 / 1.4 |
| PQN-VDN | `b768-t64` | −17.3 / 0.6 | −18.4 / 0.3 | −17.1 / 0.3 | −19.3 / 0.7 | −18.4 / 1.6 | −16.4 / 1.8 | −17.7 / 1.4 | −17.9 / 1.5 | −16.6 / 1.0 | −16.6 / 1.5 | −16.1 / 1.9 |

**Every run:**

| Method | Recipe | Seed | Selected Step | Margin | Kills | Deaths | W-D-L | vs 250M | vs Alpha | vs Beta | vs Gamma | Minutes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recurrent MAPPO | `critic-lr-1e-3` | 12259871 ⭐ | 12,042,240 | −9.5 | 6.5 | 14.6 | 0-61-195 | −3.9 | −10.4 | −12.4 | −11.2 | 41 |
| Recurrent MAPPO | `critic-lr-1e-3` | 20261001 | 32,096,256 | −11.3 | 4.6 | 16.5 | 0-64-192 | −5.2 | −12.7 | −13.8 | −13.5 | 41 |
| Recurrent MAPPO | `critic-lr-1e-3` | 20261002 | 40,108,032 | −8.6 | 8.6 | 15.0 | 8-48-200 | −9.6 | −7.6 | −9.0 | −8.3 | 41 |
| Recurrent MAPPO | `reference` | 12259871 ⭐ | 40,108,032 | −9.5 | 7.4 | 15.0 | 2-57-197 | −8.0 | −9.5 | −10.6 | −10.0 | 41 |
| Recurrent MAPPO | `reference` | 20261001 | 40,108,032 | −9.0 | 9.1 | 15.6 | 8-33-215 | −12.7 | −6.2 | −8.9 | −8.4 | 41 |
| Recurrent MAPPO | `reference` | 20261002 | 40,108,032 | −7.9 | 8.4 | 14.9 | 3-53-200 | −6.2 | −8.4 | −8.9 | −8.1 | 41 |
| Recurrent IPPO | `actor-lr-3e-4` | 12259871 ⭐ | 40,108,032 | −10.4 | 6.3 | 17.2 | 0-59-197 | −6.6 | −11.2 | −12.0 | −11.7 | 44 |
| Recurrent IPPO | `actor-lr-3e-4` | 20261001 | 16,056,320 | −11.0 | 4.6 | 15.8 | 0-63-193 | −4.2 | −12.3 | −13.4 | −14.1 | 45 |
| Recurrent IPPO | `actor-lr-3e-4` | 20261002 | 40,108,032 | −11.2 | 5.9 | 17.5 | 1-56-199 | −10.0 | −10.7 | −12.1 | −12.2 | 44 |
| Recurrent IPPO | `critic-lr-1e-3` | 12259871 ⭐ | 36,110,336 | −11.3 | 4.8 | 16.9 | 4-66-186 | −6.8 | −11.8 | −13.3 | −13.4 | 45 |
| Recurrent IPPO | `critic-lr-1e-3` | 20261001 | 40,108,032 | −11.1 | 7.2 | 17.4 | 6-36-214 | −12.8 | −9.1 | −10.7 | −11.8 | 44 |
| Recurrent IPPO | `critic-lr-1e-3` | 20261002 | 32,096,256 | −11.2 | 4.4 | 16.8 | 3-62-191 | −6.6 | −11.8 | −12.8 | −13.7 | 45 |
| Recurrent IPPO | `reference` | 12259871 ⭐ | 40,108,032 | −11.6 | 7.2 | 17.3 | 4-24-228 | −14.7 | −8.1 | −11.7 | −11.9 | 45 |
| Recurrent IPPO | `reference` | 20261001 | 40,108,032 | −8.5 | 11.1 | 15.4 | 19-12-225 | −14.7 | −6.1 | −7.1 | −6.2 | 44 |
| Recurrent IPPO | `reference` | 20261002 | 40,108,032 | −11.1 | 6.4 | 18.2 | 1-47-208 | −11.7 | −10.3 | −10.8 | −11.4 | 44 |
| Feedforward MAPPO | `actor-lr-3e-4` | 12259871 ⭐ | 12,042,240 | −9.6 | 7.0 | 14.1 | 1-61-194 | −5.9 | −10.2 | −11.2 | −11.2 | 38 |
| Feedforward MAPPO | `actor-lr-3e-4` | 20261001 | 8,028,160 | −10.6 | 4.8 | 15.6 | 0-63-193 | −3.2 | −13.1 | −13.1 | −13.1 | 38 |
| Feedforward MAPPO | `actor-lr-3e-4` | 20261002 | 24,068,096 | −8.1 | 8.0 | 15.2 | 2-59-195 | −5.9 | −7.8 | −9.6 | −8.9 | 37 |
| Feedforward MAPPO | `critic-lr-1e-3` | 12259871 ⭐ | 12,042,240 | −11.1 | 4.2 | 15.6 | 0-64-192 | −3.3 | −13.8 | −13.8 | −13.6 | 37 |
| Feedforward MAPPO | `critic-lr-1e-3` | 20261001 | 40,108,032 | −10.9 | 5.9 | 18.0 | 5-51-200 | −10.0 | −10.1 | −11.7 | −12.0 | 39 |
| Feedforward MAPPO | `critic-lr-1e-3` | 20261002 | 40,108,032 | −11.7 | 4.9 | 17.3 | 1-63-192 | −8.7 | −11.5 | −13.0 | −13.8 | 38 |
| Feedforward MAPPO | `reference` | 12259871 ⭐ | 16,056,320 | −10.4 | 5.3 | 14.7 | 0-64-192 | −3.6 | −11.8 | −13.2 | −13.1 | 38 |
| Feedforward MAPPO | `reference` | 20261001 | 16,056,320 | −11.4 | 5.3 | 14.9 | 0-57-199 | −6.7 | −12.5 | −13.2 | −13.1 | 39 |
| Feedforward MAPPO | `reference` | 20261002 | 12,042,240 | −9.9 | 6.6 | 12.6 | 0-58-198 | −4.3 | −11.5 | −12.3 | −11.4 | 37 |
| Feedforward IPPO | `actor-lr-3e-4` | 12259871 ⭐ | 40,108,032 | −11.1 | 4.6 | 16.2 | 0-64-192 | −4.7 | −12.5 | −13.6 | −13.8 | 41 |
| Feedforward IPPO | `actor-lr-3e-4` | 20261001 | 8,028,160 | −11.4 | 3.8 | 16.6 | 3-63-190 | −5.1 | −12.4 | −14.1 | −13.8 | 41 |
| Feedforward IPPO | `actor-lr-3e-4` | 20261002 | 28,082,176 | −11.4 | 4.7 | 17.3 | 0-60-196 | −7.8 | −12.2 | −12.8 | −12.7 | 41 |
| Feedforward IPPO | `critic-lr-1e-3` | 12259871 ⭐ | 36,110,336 | −11.8 | 5.9 | 17.9 | 0-47-209 | −11.4 | −10.9 | −13.2 | −11.6 | 40 |
| Feedforward IPPO | `critic-lr-1e-3` | 20261001 | 16,056,320 | −10.9 | 4.6 | 15.5 | 0-64-192 | −3.6 | −12.0 | −14.1 | −13.9 | 41 |
| Feedforward IPPO | `critic-lr-1e-3` | 20261002 | 40,108,032 | −11.4 | 5.9 | 17.2 | 0-55-201 | −9.1 | −10.8 | −12.8 | −13.0 | 40 |
| Feedforward IPPO | `reference` | 12259871 ⭐ | 40,108,032 | −10.8 | 5.0 | 16.1 | 0-65-191 | −4.7 | −12.4 | −13.5 | −12.7 | 41 |
| Feedforward IPPO | `reference` | 20261001 | 40,108,032 | −10.7 | 5.7 | 16.4 | 0-64-192 | −5.1 | −12.0 | −12.6 | −13.0 | 41 |
| Feedforward IPPO | `reference` | 20261002 | 4,014,080 | −11.2 | 3.8 | 15.5 | 0-65-191 | −3.0 | −13.9 | −14.5 | −13.7 | 41 |
| QMIX | `b768-t128` | 12259871 ⭐ | 40,108,032 | −10.8 | 7.0 | 16.1 | 8-16-232 | −9.6 | −11.1 | −11.6 | −10.8 | 53 |
| QMIX | `b768-t128` | 20261001 | 40,108,032 | −11.6 | 5.4 | 16.4 | 25-37-194 | −12.8 | −8.1 | −10.9 | −14.5 | 55 |
| QMIX | `b768-t128` | 20261002 | 40,108,032 | −12.4 | 4.4 | 16.1 | 0-52-204 | −10.0 | −10.4 | −15.0 | −14.4 | 55 |
| QMIX | `b768-t32` | 12259871 ⭐ | 40,108,032 | −11.3 | 5.4 | 16.1 | 0-53-203 | −7.6 | −11.1 | −13.5 | −13.1 | 55 |
| QMIX | `b768-t32` | 20261001 | 40,108,032 | −11.6 | 5.7 | 17.2 | 0-55-201 | −10.2 | −10.2 | −12.2 | −13.6 | 53 |
| QMIX | `b768-t32` | 20261002 | 36,102,144 | −10.6 | 5.3 | 16.9 | 13-45-198 | −4.3 | −11.0 | −12.4 | −14.8 | 55 |
| QMIX | `b768-t32-lr-1e-4` | 12259871 ⭐ | 20,054,016 | −11.5 | 4.0 | 15.0 | 0-64-192 | −3.5 | −14.5 | −13.9 | −14.0 | 54 |
| QMIX | `b768-t32-lr-1e-4` | 20261001 | 40,108,032 | −9.6 | 7.8 | 16.7 | 4-50-202 | −7.4 | −10.1 | −11.0 | −10.0 | 55 |
| QMIX | `b768-t32-lr-1e-4` | 20261002 | 40,108,032 | −9.5 | 6.8 | 17.5 | 17-36-203 | −8.1 | −7.4 | −10.2 | −12.4 | 55 |
| PQN-VDN | `b512-t128` | 12259871 ⭐ | 16,058,368 | −15.2 | 0.4 | 12.0 | 0-87-169 | −5.8 | −15.8 | −20.0 | −19.2 | 72 |
| PQN-VDN | `b512-t128` | 20261001 | 40,108,032 | −15.0 | 2.3 | 16.1 | 0-65-191 | −9.7 | −15.8 | −17.0 | −17.5 | 72 |
| PQN-VDN | `b512-t128` | 20261002 | 4,065,280 | −12.4 | 0.4 | 6.9 | 0-96-160 | −19.5 | −5.2 | −5.0 | −19.6 | 72 |
| PQN-VDN | `b768-t128` | 12259871 ⭐ | 36,178,944 | −14.5 | 2.5 | 16.1 | 0-72-184 | −9.5 | −13.9 | −16.9 | −17.6 | 69 |
| PQN-VDN | `b768-t128` | 20261001 | 40,108,032 | −15.7 | 0.8 | 14.7 | 0-68-188 | −6.6 | −17.2 | −19.6 | −19.5 | 68 |
| PQN-VDN | `b768-t128` | 20261002 | 36,178,944 | −15.7 | 2.0 | 15.3 | 0-57-199 | −12.4 | −16.5 | −16.6 | −17.4 | 69 |
| PQN-VDN | `b768-t32` | 12259871 ⭐ | 16,051,200 | −10.1 | 0.0 | 5.1 | 0-128-128 | −20.3 | 0.0 | 0.0 | −20.0 | 73 |
| PQN-VDN | `b768-t32` | 20261001 | 28,093,440 | −9.1 | 0.5 | 5.7 | 0-148-108 | −17.9 | 0.0 | 0.0 | −18.6 | 71 |
| PQN-VDN | `b768-t32` | 20261002 | 20,057,088 | −9.9 | 0.3 | 5.4 | 0-128-128 | −19.8 | 0.0 | 0.0 | −19.8 | 71 |
| PQN-VDN | `b768-t64` | 12259871 ⭐ | 20,057,088 | −13.5 | 1.9 | 8.6 | 0-64-192 | −19.8 | −8.5 | −8.8 | −16.9 | 71 |
| PQN-VDN | `b768-t64` | 20261001 | 40,108,032 | −15.5 | 1.6 | 13.3 | 0-82-174 | −11.0 | −14.5 | −17.9 | −18.6 | 69 |
| PQN-VDN | `b768-t64` | 20261002 | 32,099,328 | −15.9 | 0.9 | 10.4 | 0-67-189 | −20.3 | −12.1 | −12.5 | −18.9 | 69 |

**Cost per run, measured.** Minutes include setup and validation. Memory is
the process's peak with preallocation off.

| Method | Runs | Transitions Per Second | Minutes Per Run | JAX Memory In Use | Host RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recurrent MAPPO | 6 | 19,485 | 41 | 3.1 GB | 8.4 GB |
| Recurrent IPPO | 9 | 17,834 | 44-45 | 3.0 GB | 7.8 GB |
| Feedforward MAPPO | 9 | 21,621 | 37-39 | 2.7 GB | 8.7 GB |
| Feedforward IPPO | 9 | 19,639 | 40-41 | 2.7 GB | 8.6 GB |
| QMIX | 9 | 13,973 | 53-55 | 13.5 GB | 14.1 GB |
| PQN-VDN | 12 | 10,300 | 68-73 | 11.1 GB | 9.9 GB |

### Study 2: Q-Method Settings

Study 2 ran all 30 runs to 40,108,032 transitions with no failed run. It took
16 hours 39 minutes on two RTX 5090s, or 32.8 GPU-hours (planned: 29-32).
Each change is compared with its base's study 1 runs on the same three seeds;
those base runs appear below as `base`. QMIX's base is B768 T32 with
`q_lr = 0.0001`; PQN-VDN's base is B768 T128.

**Recipes** (mean over three seeds):

| Method | Recipe | Mean Margin | SD | Seed 12259871 | Seed 20261001 | Seed 20261002 | Kills | Deaths | K/D | Red Zone Kills (Ours / Theirs) | W-D-L | vs 250M | vs Alpha | vs Beta | vs Gamma |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QMIX | `base` | −10.2 | 1.1 | −11.5 | −9.6 | −9.5 | 6.2 | 16.4 | 0.38 | 1.17 / 1.16 | 21-150-597 | −6.3 | −10.7 | −11.7 | −12.1 |
| QMIX | `gamma-0995` | −12.1 | 1.2 | −12.0 | −13.3 | −10.9 | 4.8 | 16.2 | 0.29 | 0.98 / 1.65 | 13-163-592 | −7.7 | −12.7 | −14.3 | −13.7 |
| QMIX | `reuse-x2` | −7.6 | 1.9 | −9.3 | −5.5 | −8.1 | 11.4 | 18.1 | 0.63 | 0.43 / 1.44 | 75-47-646 | −11.0 | −5.0 | −6.3 | −8.2 |
| QMIX | `reward-x10` | −4.3 | 0.3 | −3.9 | −4.4 | −4.4 | 12.4 | 16.6 | 0.75 | 1.19 / 1.25 | 205-75-488 | −9.6 | −0.3 | −2.3 | −4.8 |
| QMIX | `target-faster` | −5.2 | 2.1 | −2.8 | −6.6 | −6.1 | 11.5 | 15.9 | 0.72 | 1.64 / 2.41 | 168-81-519 | −9.8 | −2.9 | −3.1 | −4.9 |
| PQN-VDN | `base` | −15.3 | 0.7 | −14.5 | −15.7 | −15.7 | 1.8 | 15.4 | 0.12 | 0.40 / 2.13 | 0-197-571 | −9.5 | −15.9 | −17.7 | −18.2 |
| PQN-VDN | `explore-20pct` | −15.2 | 1.1 | −14.9 | −14.3 | −16.4 | 0.9 | 11.3 | 0.08 | 0.07 / 4.86 | 0-215-553 | −17.7 | −12.0 | −12.4 | −18.8 |
| PQN-VDN | `gamma-0995` | −13.9 | 1.0 | −14.5 | −12.7 | −14.5 | 0.7 | 9.8 | 0.08 | 0.13 / 5.00 | 0-285-483 | −18.7 | −8.2 | −9.5 | −19.2 |
| PQN-VDN | `lambda-095` | −15.2 | 0.6 | −14.8 | −16.0 | −14.9 | 0.7 | 13.3 | 0.05 | 0.10 / 2.76 | 0-341-427 | −8.1 | −16.5 | −16.3 | −20.0 |
| PQN-VDN | `layer-norm` | −13.7 | 3.9 | −15.9 | −15.9 | −9.2 | 1.5 | 10.7 | 0.14 | 0.11 / 4.60 | 0-225-543 | −16.3 | −10.7 | −10.3 | −17.4 |
| PQN-VDN | `lr-1e-4` | −15.0 | 0.7 | −15.3 | −14.1 | −15.4 | 2.7 | 15.8 | 0.17 | 0.45 / 2.31 | 7-179-582 | −9.1 | −16.6 | −17.1 | −17.0 |
| PQN-VDN | `reward-x10` | −8.7 | 3.3 | −5.1 | −11.4 | −9.7 | 10.5 | 17.8 | 0.59 | 0.15 / 1.58 | 88-47-633 | −12.4 | −6.1 | −8.5 | −7.9 |

**Kills / deaths per game by opponent:**

| Method | Recipe | vs 250M | vs Alpha | vs Beta | vs Gamma |
| --- | --- | ---: | ---: | ---: | ---: |
| QMIX | `base` | 3.3 / 8.2 | 7.7 / 19.2 | 6.9 / 19.3 | 7.0 / 19.0 |
| QMIX | `gamma-0995` | 2.7 / 8.7 | 5.8 / 18.8 | 5.2 / 18.6 | 5.4 / 18.5 |
| QMIX | `reuse-x2` | 7.7 / 16.8 | 14.1 / 18.8 | 12.7 / 18.6 | 11.1 / 18.0 |
| QMIX | `reward-x10` | 6.3 / 14.2 | 15.7 / 16.3 | 14.2 / 17.6 | 13.6 / 18.4 |
| QMIX | `target-faster` | 5.3 / 13.4 | 14.2 / 16.9 | 13.5 / 16.4 | 12.8 / 16.8 |
| PQN-VDN | `base` | 0.6 / 8.4 | 2.6 / 17.5 | 2.0 / 17.9 | 1.9 / 17.6 |
| PQN-VDN | `explore-20pct` | 0.3 / 10.3 | 0.8 / 10.6 | 1.0 / 10.5 | 1.5 / 14.0 |
| PQN-VDN | `gamma-0995` | 0.3 / 11.2 | 1.0 / 7.1 | 0.7 / 7.8 | 1.0 / 13.1 |
| PQN-VDN | `lambda-095` | 0.6 / 7.2 | 1.0 / 15.2 | 0.7 / 14.7 | 0.5 / 15.9 |
| PQN-VDN | `layer-norm` | 0.9 / 10.4 | 1.0 / 9.3 | 1.2 / 9.2 | 2.9 / 13.9 |
| PQN-VDN | `lr-1e-4` | 2.7 / 10.8 | 3.0 / 17.7 | 2.7 / 17.6 | 2.7 / 17.2 |
| PQN-VDN | `reward-x10` | 7.0 / 18.5 | 12.3 / 17.2 | 11.1 / 18.0 | 11.5 / 17.3 |

**Picks under the declared rules.** A change replaces its base only if its
mean margin beats the base by more than 2 points, it wins on at least 2 of 3
seeds, and it makes at least 2 kills per game against each scripted bot (the
engagement floor). Each candidate shows its mean margin / kills / deaths per
game.

| Pick | Candidates (Margin / Kills / Deaths) | Mean Gain | Seeds Won | Fewest Kills Against A Bot | Result |
| --- | --- | ---: | ---: | ---: | --- |
| QMIX `reward_scale = 10` | Base −10.2 / 6.2 / 16.4; change −4.3 / 12.4 / 16.6 | +5.9 | 3 | 13.6 | Passes |
| QMIX `update_period = 128` (`target-faster`) | Base −10.2 / 6.2 / 16.4; change −5.2 / 11.5 / 15.9 | +5.0 | 3 | 12.8 | Passes |
| QMIX twice the training per step (`reuse-x2`) | Base −10.2 / 6.2 / 16.4; change −7.6 / 11.4 / 18.1 | +2.6 | 3 | 11.1 | Passes |
| QMIX `gamma = 0.995` | Base −10.2 / 6.2 / 16.4; change −12.1 / 4.8 / 16.2 | −1.9 | 0 | 5.2 | Fails gain |
| PQN-VDN `reward_scale = 10` | Base −15.3 / 1.8 / 15.4; change −8.7 / 10.5 / 17.8 | +6.6 | 3 | 11.1 | Passes |
| PQN-VDN `normalization = "layer_norm"` | Base −15.3 / 1.8 / 15.4; change −13.7 / 1.5 / 10.7 | +1.6 | 1 | 1.0 | Fails gain and floor |
| PQN-VDN `gamma = 0.995` | Base −15.3 / 1.8 / 15.4; change −13.9 / 0.7 / 9.8 | +1.4 | 2 | 0.7 | Fails gain and floor |
| PQN-VDN `q_lr = 0.0001` | Base −15.3 / 1.8 / 15.4; change −15.0 / 2.7 / 15.8 | +0.4 | 2 | 2.7 | Fails gain |
| PQN-VDN `eps_decay_fraction = 0.2` (`explore-20pct`) | Base −15.3 / 1.8 / 15.4; change −15.2 / 0.9 / 11.3 | +0.1 | 1 | 0.8 | Fails gain and floor |
| PQN-VDN `td_lambda = 0.95` | Base −15.3 / 1.8 / 15.4; change −15.2 / 0.7 / 13.3 | +0.1 | 1 | 0.5 | Fails gain and floor |

Three QMIX changes pass; the highest mean, `reward_scale = 10`, becomes the
final recipe. For PQN-VDN only `reward_scale = 10` passes.

**Findings.**

1. **Reward scale 10 is the largest gain for both Q methods.** QMIX goes from
   −10.2 to −4.3 and from 6.2 to 12.4 kills per game, better on all three
   seeds. PQN-VDN goes from −15.3 to −8.7 and from 1.8 to 10.5 kills, better
   on all three seeds. The scale changes only the size of the learning
   targets, not which actions are best (see Reward), so the gain is an effect
   on learning, not a new incentive to fight.
2. **Reward scale 10 is the only change that stopped PQN-VDN avoiding
   fights.** The other five changes make 0.7-2.7 kills per game. Three of
   them (exploration over 20%, gamma 0.995 and LayerNorm) improve the margin
   only by dying less: 9.8-11.3 deaths per game against 15.4 for the base.
   Two selected checkpoints avoided fights completely. LayerNorm on seed
   20261002 (selected at 12.1M) had 0 kills and 0 deaths against `tdm-alpha`
   and `tdm-beta`. Gamma 0.995 on seed 20261001 (selected at 20.1M) had 0
   kills against every opponent.
3. **QMIX: three of four changes pass every rule.** Reward scale 10 (+5.9),
   faster target copies (+5.0) and twice the training per step (+2.6). Twice
   the training costs 79 minutes per run instead of 55. Gamma 0.995 is worse
   on every seed (−1.9).
4. **The engagement floor blocked no change that passed the gain rules.** The
   four PQN-VDN changes under 2 kills per game against a bot also failed the
   gain rules.
5. **The Q methods now lead every PPO recipe against the scripted bots.**
   QMIX with reward scale 10 scores −0.3 against `tdm-alpha` and wins 205 of
   768 games; the best PPO recipe, recurrent MAPPO `reference`, wins 13.
   Against the 250M model, QMIX (−9.6) and PQN-VDN (−12.4) fall inside the
   range of the 12 PPO `reference` runs (−3.0 to −14.7), but they fight it
   far more: 6.3 and 7.0 kills per game against 0.3-4.7 for those PPO runs
   (compare study 1 finding 5).
6. **Both final recipes were still improving at 40M.** PQN-VDN with reward
   scale 10 selected its last check on all three seeds; its mean margin rose
   from −10.5 at 80% of training to −8.6 at 100%. QMIX with reward scale 10
   selected 36.1M, 36.1M and 40.1M.
7. **On seed 12259871 both final recipes have their best run:** QMIX −3.9
   (12.9 kills, 16.4 deaths) and PQN-VDN −5.1 (13.1 kills, 17.1 deaths).
8. **The best single QMIX run is not in the final recipe.** Faster target
   copies on seed 12259871 scored −2.8 (12.8 kills, 15.1 deaths, 74 wins),
   better than any reward-scale-10 run. Recipes are chosen by their
   three-seed mean, where reward scale 10 leads (−4.3 against −5.2).

**Learning curves.** Margin / kills per game at each routine check, mean over
the three seeds; checks as in study 1.

| Method | Recipe | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QMIX | `base` | −14.8 / 0.6 | −18.2 / 1.1 | −17.2 / 1.2 | −14.9 / 1.7 | −15.2 / 2.1 | −14.6 / 2.3 | −14.4 / 3.2 | −14.0 / 3.6 | −12.2 / 5.0 | −11.4 / 5.8 | −10.0 / 7.2 |
| QMIX | `gamma-0995` | −14.8 / 0.6 | −18.7 / 0.4 | −17.2 / 0.8 | −16.6 / 1.8 | −16.7 / 2.0 | −15.2 / 2.2 | −14.6 / 2.0 | −14.8 / 2.8 | −14.0 / 3.3 | −13.9 / 3.5 | −12.0 / 4.8 |
| QMIX | `reuse-x2` | −14.8 / 0.6 | −17.9 / 0.7 | −17.0 / 1.3 | −17.0 / 1.3 | −15.9 / 2.2 | −14.7 / 2.5 | −14.8 / 2.6 | −13.9 / 3.2 | −12.1 / 4.8 | −10.7 / 7.9 | −7.9 / 11.2 |
| QMIX | `reward-x10` | −14.8 / 0.6 | −15.0 / 3.1 | −13.5 / 4.0 | −12.4 / 4.9 | −11.0 / 6.5 | −10.4 / 7.7 | −7.4 / 10.2 | −6.2 / 10.5 | −6.1 / 10.6 | −4.9 / 11.6 | −4.6 / 12.4 |
| QMIX | `target-faster` | −14.8 / 0.6 | −19.7 / 0.5 | −16.8 / 1.8 | −14.2 / 2.8 | −13.2 / 4.1 | −10.9 / 6.3 | −10.1 / 7.6 | −9.0 / 9.2 | −7.2 / 11.0 | −6.0 / 11.4 | −5.7 / 11.3 |
| PQN-VDN | `base` | −17.3 / 0.6 | −18.4 / 1.3 | −17.0 / 1.3 | −18.3 / 0.5 | −18.3 / 0.8 | −18.3 / 0.6 | −17.8 / 1.4 | −16.8 / 1.1 | −16.9 / 1.2 | −15.4 / 1.7 | −15.6 / 1.8 |
| PQN-VDN | `explore-20pct` | −17.3 / 0.6 | −19.4 / 0.6 | −17.9 / 1.3 | −18.1 / 1.2 | −17.2 / 1.1 | −18.6 / 1.4 | −17.8 / 1.0 | −18.5 / 1.6 | −17.4 / 2.3 | −17.8 / 1.7 | −16.4 / 2.3 |
| PQN-VDN | `gamma-0995` | −17.3 / 0.6 | −17.0 / 1.6 | −16.8 / 1.5 | −17.7 / 1.8 | −17.3 / 0.6 | −15.6 / 0.6 | −19.2 / 0.6 | −17.7 / 0.7 | −16.4 / 1.1 | −15.7 / 1.5 | −16.9 / 1.6 |
| PQN-VDN | `lambda-095` | −17.3 / 0.6 | −18.4 / 0.3 | −18.8 / 0.9 | −17.1 / 1.1 | −17.7 / 0.8 | −17.6 / 1.1 | −17.2 / 1.2 | −16.4 / 0.6 | −15.8 / 0.5 | −15.2 / 0.6 | −16.0 / 0.5 |
| PQN-VDN | `layer-norm` | −13.9 / 0.9 | −17.3 / 1.6 | −15.5 / 2.4 | −15.4 / 1.5 | −17.4 / 0.8 | −18.1 / 1.5 | −18.2 / 1.1 | −16.5 / 2.1 | −15.7 / 2.2 | −17.6 / 1.5 | −15.6 / 2.1 |
| PQN-VDN | `lr-1e-4` | −17.3 / 0.6 | −18.6 / 1.1 | −17.8 / 1.3 | −16.4 / 1.7 | −17.3 / 1.2 | −18.3 / 1.0 | −16.9 / 2.3 | −15.6 / 1.9 | −15.0 / 2.5 | −15.3 / 2.4 | −16.0 / 2.1 |
| PQN-VDN | `reward-x10` | −17.3 / 0.6 | −14.7 / 2.9 | −16.2 / 2.6 | −15.7 / 2.9 | −15.3 / 3.1 | −14.7 / 3.8 | −12.7 / 5.7 | −12.5 / 6.8 | −10.5 / 8.7 | −10.5 / 9.1 | −8.6 / 10.5 |

**Every run:**

| Method | Recipe | Seed | Selected Step | Margin | Kills | Deaths | W-D-L | vs 250M | vs Alpha | vs Beta | vs Gamma | Minutes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QMIX | `base` | 12259871 ⭐ | 20,054,016 | −11.5 | 4.0 | 15.0 | 0-64-192 | −3.5 | −14.5 | −13.9 | −14.0 | 54 |
| QMIX | `base` | 20261001 | 40,108,032 | −9.6 | 7.8 | 16.7 | 4-50-202 | −7.4 | −10.1 | −11.0 | −10.0 | 55 |
| QMIX | `base` | 20261002 | 40,108,032 | −9.5 | 6.8 | 17.5 | 17-36-203 | −8.1 | −7.4 | −10.2 | −12.4 | 55 |
| QMIX | `gamma-0995` | 12259871 ⭐ | 40,108,032 | −12.0 | 5.0 | 16.0 | 0-57-199 | −7.2 | −12.4 | −14.9 | −13.5 | 56 |
| QMIX | `gamma-0995` | 20261001 | 40,108,032 | −13.3 | 3.6 | 15.2 | 0-68-188 | −8.1 | −14.4 | −14.9 | −15.9 | 55 |
| QMIX | `gamma-0995` | 20261002 | 40,108,032 | −10.9 | 5.7 | 17.2 | 13-38-205 | −7.8 | −11.2 | −13.1 | −11.6 | 55 |
| QMIX | `reuse-x2` | 12259871 ⭐ | 40,108,032 | −9.3 | 10.2 | 17.5 | 24-3-229 | −13.5 | −4.9 | −7.5 | −11.4 | 78 |
| QMIX | `reuse-x2` | 20261001 | 40,108,032 | −5.5 | 13.1 | 18.3 | 35-19-202 | −9.0 | −3.5 | −5.1 | −4.4 | 78 |
| QMIX | `reuse-x2` | 20261002 | 40,108,032 | −8.1 | 11.0 | 18.3 | 16-25-215 | −10.5 | −6.8 | −6.4 | −8.9 | 80 |
| QMIX | `reward-x10` | 12259871 ⭐ | 36,102,144 | −3.9 | 12.9 | 16.4 | 56-36-164 | −8.4 | −0.2 | −3.4 | −3.8 | 55 |
| QMIX | `reward-x10` | 20261001 | 36,102,144 | −4.4 | 11.4 | 16.3 | 81-2-173 | −12.8 | −1.0 | 0.5 | −4.2 | 56 |
| QMIX | `reward-x10` | 20261002 | 40,108,032 | −4.4 | 13.0 | 17.1 | 68-37-151 | −7.7 | 0.4 | −4.0 | −6.5 | 55 |
| QMIX | `target-faster` | 12259871 ⭐ | 40,108,032 | −2.8 | 12.8 | 15.1 | 74-54-128 | −6.2 | −1.2 | −1.0 | −2.8 | 58 |
| QMIX | `target-faster` | 20261001 | 36,102,144 | −6.6 | 9.9 | 16.8 | 36-18-202 | −11.7 | −5.0 | −4.5 | −5.1 | 55 |
| QMIX | `target-faster` | 20261002 | 36,102,144 | −6.1 | 11.7 | 15.7 | 58-9-189 | −11.5 | −2.5 | −3.8 | −6.9 | 56 |
| PQN-VDN | `base` | 12259871 ⭐ | 36,178,944 | −14.5 | 2.5 | 16.1 | 0-72-184 | −9.5 | −13.9 | −16.9 | −17.6 | 69 |
| PQN-VDN | `base` | 20261001 | 40,108,032 | −15.7 | 0.8 | 14.7 | 0-68-188 | −6.6 | −17.2 | −19.6 | −19.5 | 68 |
| PQN-VDN | `base` | 20261002 | 36,178,944 | −15.7 | 2.0 | 15.3 | 0-57-199 | −12.4 | −16.5 | −16.6 | −17.4 | 69 |
| PQN-VDN | `explore-20pct` | 12259871 ⭐ | 24,087,552 | −14.9 | 0.3 | 8.2 | 0-80-176 | −20.0 | −10.1 | −9.4 | −20.1 | 68 |
| PQN-VDN | `explore-20pct` | 20261001 | 16,124,928 | −14.3 | 0.8 | 9.4 | 0-88-168 | −20.1 | −8.5 | −10.0 | −18.5 | 69 |
| PQN-VDN | `explore-20pct` | 20261002 | 40,108,032 | −16.4 | 1.7 | 16.4 | 0-47-209 | −12.9 | −17.4 | −17.8 | −17.8 | 68 |
| PQN-VDN | `gamma-0995` | 12259871 ⭐ | 32,148,480 | −14.5 | 0.2 | 9.1 | 0-106-150 | −20.1 | −7.9 | −9.8 | −20.4 | 68 |
| PQN-VDN | `gamma-0995` | 20261001 | 20,057,088 | −12.7 | 0.0 | 6.4 | 0-96-160 | −20.1 | −5.2 | −5.2 | −20.4 | 69 |
| PQN-VDN | `gamma-0995` | 20261002 | 40,108,032 | −14.5 | 2.0 | 13.8 | 0-83-173 | −15.9 | −11.5 | −13.6 | −17.0 | 70 |
| PQN-VDN | `lambda-095` | 12259871 ⭐ | 36,178,944 | −14.8 | 0.4 | 12.4 | 0-104-152 | −5.1 | −16.2 | −17.9 | −20.0 | 69 |
| PQN-VDN | `lambda-095` | 20261001 | 28,118,016 | −16.0 | 0.6 | 13.8 | 0-118-138 | −12.0 | −17.8 | −14.4 | −19.8 | 70 |
| PQN-VDN | `lambda-095` | 20261002 | 36,178,944 | −14.9 | 1.1 | 13.6 | 0-119-137 | −7.4 | −15.5 | −16.6 | −20.1 | 69 |
| PQN-VDN | `layer-norm` | 12259871 ⭐ | 40,108,032 | −15.9 | 1.4 | 14.9 | 0-65-191 | −10.3 | −19.0 | −16.6 | −17.5 | 68 |
| PQN-VDN | `layer-norm` | 20261001 | 4,033,536 | −15.9 | 2.1 | 11.7 | 0-32-224 | −19.2 | −13.1 | −14.2 | −17.2 | 69 |
| PQN-VDN | `layer-norm` | 20261002 | 12,094,464 | −9.2 | 1.1 | 5.5 | 0-128-128 | −19.4 | 0.0 | 0.0 | −17.5 | 68 |
| PQN-VDN | `lr-1e-4` | 12259871 ⭐ | 32,148,480 | −15.3 | 3.2 | 17.3 | 1-47-208 | −13.0 | −15.1 | −16.6 | −16.5 | 69 |
| PQN-VDN | `lr-1e-4` | 20261001 | 36,178,944 | −14.1 | 3.5 | 15.5 | 6-45-205 | −8.3 | −16.4 | −16.6 | −15.2 | 68 |
| PQN-VDN | `lr-1e-4` | 20261002 | 40,108,032 | −15.4 | 1.5 | 14.7 | 0-87-169 | −6.0 | −18.2 | −18.0 | −19.4 | 68 |
| PQN-VDN | `reward-x10` | 12259871 ⭐ | 40,108,032 | −5.1 | 13.1 | 17.1 | 72-24-160 | −9.7 | −3.1 | −4.2 | −3.2 | 69 |
| PQN-VDN | `reward-x10` | 20261001 | 40,108,032 | −11.4 | 8.5 | 18.6 | 0-17-239 | −12.6 | −10.5 | −11.8 | −10.8 | 68 |
| PQN-VDN | `reward-x10` | 20261002 | 40,108,032 | −9.7 | 9.8 | 17.6 | 16-6-234 | −14.9 | −4.8 | −9.4 | −9.8 | 68 |

**Cost per run, measured** (study 2 runs only):

| Method | Runs | Transitions Per Second | Minutes Per Run | JAX Memory In Use | Host RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| QMIX | 12 | 12,700 | 55-80 | 12.1 GB | 14.1 GB |
| PQN-VDN | 18 | 10,537 | 68-70 | 15.0 GB | 11.0 GB |

### Final Recipes

Each method's recipe after studies 1 and 2, and the run each method continues
to 250M in study 3. For four methods the continued run is the best run of the
final recipe. For QMIX it comes from another recipe (decision 10), and
feedforward MAPPO continues its team-reward run (decision 11). ⭐ marks seed
12259871.

| Method | Final Recipe | Recipe Mean Margin / Kills | Continued Run | Its Selected Step | Its Margin / Kills / Deaths / Wins |
| --- | --- | --- | --- | ---: | --- |
| Recurrent MAPPO | `reference` | −8.8 / 8.3 | `reference`, seed 20261002 | 40,108,032 | −7.9 / 8.4 / 14.9 / 3 |
| Recurrent IPPO | `reference` | −10.4 / 8.2 | `reference`, seed 20261001 | 40,108,032 | −8.5 / 11.1 / 15.4 / 19 |
| Feedforward MAPPO | Team reward, `actor_lr = 0.0003` | – | Team reward, seed 12259871 ⭐ | 40,108,032 | −8.7 / 9.2 / 15.2 / 17 |
| Feedforward IPPO | `reference` | −10.9 / 4.8 | `reference`, seed 20261001 | 40,108,032 | −10.7 / 5.7 / 16.4 / 0 |
| QMIX | B768 T32, `q_lr = 0.0001`, `reward_scale = 10` | −4.3 / 12.4 | `target-faster`, seed 12259871 ⭐ | 40,108,032 | −2.8 / 12.8 / 15.1 / 74 |
| PQN-VDN | B768 T128, `reward_scale = 10` | −8.7 / 10.5 | `reward-x10`, seed 12259871 ⭐ | 40,108,032 | −5.1 / 13.1 / 17.1 / 72 |

QMIX `target-faster` on seed 12259871 scores −2.8 with 12.8 kills and 15.1
deaths, against −3.9, 12.9 and 16.4 for the best `reward-x10` run, so it is
the run continued.

### Study 3: 250M Continuations

All six continuations ran to 250,675,200 transitions with no failure: 23.4
GPU-hours. Each lineage is represented by its selected checkpoint, chosen and
confirmed exactly as in studies 1 and 2, with the parent's checkpoints eligible
too. ⭐ marks seed 12259871.

| Method | Run | Seed | Selected Checkpoint | Margin | Kills | Deaths | W-D-L | vs 250M | vs Alpha | vs Beta | vs Gamma | 40M Run's Selection: Margin / Kills / Deaths | Final Check At 250.7M: Margin / Kills / Deaths |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| PQN-VDN | `reward-x10` | 12259871 ⭐ | 240,156,672 | +7.3 | 19.4 | 12.4 | 234-16-6 | +3.2 | +8.5 | +8.9 | +8.6 | −5.1 / 13.1 / 17.1 | +5.2 / 19.2 / 14.4 |
| Recurrent IPPO | `reference` | 20261001 | 250,675,200 | +5.9 | 19.1 | 12.6 | 205-26-25 | +0.9 | +8.3 | +7.7 | +6.8 | −8.5 / 11.1 / 15.4 | +5.9 / 19.1 / 12.6 |
| Feedforward IPPO | `reference` | 20261001 | 219,103,232 | +4.0 | 18.5 | 14.1 | 177-35-44 | +1.2 | +7.0 | +4.5 | +3.4 | −10.7 / 5.7 / 16.4 | +3.1 / 18.4 / 14.8 |
| QMIX | `target-faster` | 12259871 ⭐ | 219,095,040 | +4.0 | 18.2 | 15.2 | 218-9-29 | −0.8 | +6.5 | +5.5 | +4.8 | −2.8 / 12.8 / 15.1 | +1.5 / 17.8 / 16.2 |
| Recurrent MAPPO | `reference` | 20261002 | 219,103,232 | +3.1 | 18.5 | 14.8 | 186-21-49 | +1.7 | +4.7 | +3.8 | +2.2 | −7.9 / 8.4 / 14.9 | +2.6 / 18.4 / 15.2 |
| Feedforward MAPPO | Team reward | 12259871 ⭐ | 208,568,320 | +1.9 | 16.0 | 12.7 | 169-38-49 | −10.9 | +6.2 | +6.5 | +5.6 | −8.7 / 9.2 / 15.2 | +0.6 / 15.3 / 13.0 |

**Learning curves, 40M to 250M.** Margin / kills per game at every second
routine check (128 games each); the 40M column is the 40M run's last check.
Every check is in `lineage_curves.csv` and the figure.

| Method | 40M | 61M | 82M | 103M | 124M | 145M | 166M | 188M | 209M | 230M | 251M |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PQN-VDN | −5.0 / 13.1 | −10.2 / 9.9 | −1.2 / 15.9 | +1.4 / 17.9 | +1.9 / 18.0 | −2.8 / 15.3 | +3.4 / 18.6 | +4.5 / 19.3 | +5.2 / 18.5 | +5.7 / 19.5 | +5.0 / 19.1 |
| Recurrent IPPO | −8.8 / 10.9 | −3.6 / 12.6 | −0.9 / 14.4 | −2.1 / 15.3 | −0.1 / 16.9 | +1.6 / 17.7 | +2.5 / 18.0 | +3.3 / 18.4 | +5.0 / 18.9 | +4.4 / 18.8 | +5.8 / 18.9 |
| Feedforward IPPO | −10.8 / 5.7 | −12.1 / 6.7 | −10.3 / 9.5 | −5.4 / 13.6 | −3.6 / 15.0 | −3.0 / 15.2 | +2.0 / 17.6 | +1.8 / 17.8 | +1.8 / 17.9 | +4.1 / 18.2 | +2.5 / 18.0 |
| QMIX | −1.8 / 13.0 | −3.5 / 14.6 | −1.7 / 14.9 | −2.1 / 15.7 | 0.0 / 16.3 | +1.9 / 17.7 | +1.7 / 17.6 | −0.4 / 16.9 | +1.6 / 18.2 | +2.3 / 17.9 | +1.7 / 18.1 |
| Recurrent MAPPO | −7.7 / 8.7 | −10.4 / 6.3 | −8.4 / 9.9 | −6.2 / 12.9 | −4.5 / 14.6 | −0.7 / 16.8 | −0.5 / 16.8 | +1.3 / 17.9 | +2.3 / 18.2 | +2.1 / 18.1 | +3.4 / 18.5 |
| Feedforward MAPPO | −8.4 / 9.2 | −5.4 / 10.5 | −3.6 / 12.0 | −1.0 / 13.8 | 0.0 / 15.0 | +0.5 / 15.2 | +0.2 / 15.1 | −0.3 / 14.9 | +2.4 / 16.9 | +2.1 / 16.7 | +0.3 / 15.5 |

**Findings.**

1. **All six methods went from losing to winning against the validation
   panel.** At 40M every method had a negative margin; at their selected
   checkpoints all six are between +1.9 and +7.3. Four beat the pooled 250M
   model (+0.9 to +3.2). QMIX is just below it (−0.8), and feedforward MAPPO
   loses to it clearly (−10.9) while beating all three bots (+5.6 to +6.5).
2. **Every method learned to win fights.** From the 40M selection to the
   selected checkpoint, kills rose from 6-13 per game to 16-19, while deaths
   fell or stayed level (14.9-17.1 per game before, 12.4-15.2 after; QMIX and
   recurrent MAPPO about unchanged). The gains do not come from avoiding
   fights.
3. **PQN-VDN dipped, then became the strongest on the validation panel.** Its margin fell from −5.0 at
   40M to −10.2 at 61M after its learning rate was restarted (see Our
   Reading), recovered by 82M, and climbed to the study's best checkpoint:
   +7.3 with 19.4 kills and 12.4 deaths at 240.2M, beating every panel
   opponent including the 250M model (+3.2).
4. **QMIX improved slowly, with no dip larger than its usual check-to-check
   swings** (−1.8 at 40M, −4.8 at its lowest at 93M, between +0.1 and +3.5
   from 198M on; its 40M parent's last three checks ranged from −6.0 to
   −1.8). Its rate is constant, so it had no restart.
5. **Only recurrent IPPO was clearly still improving at 250M.** Its last check
   is its best, and over its last six checks (198M to 251M) its margin rose
   from +3.4 to +5.8. Over the same checks recurrent MAPPO (+2.1 to +3.4),
   QMIX (+0.1 to +3.5), feedforward IPPO (+1.7 to +4.1), feedforward MAPPO
   (−0.9 to +2.4) and PQN-VDN (+5.0 to +7.3) moved up and down without a clear
   trend; PQN-VDN's last check is 2.4 points below its best.
6. **Feedforward MAPPO improved in two steps.** Its margin rose from −8.4 at
   40M to about 0 by 124M, stayed near 0 to 198M, then held between +1.7 and
   +2.4 from 209M to 240M. Its selected checkpoint at 208.6M scores +1.9 with
   16.0 kills and 12.7 deaths, the third-fewest deaths of the six.

**Cost per continuation, measured.**

| Method | Hours | Transitions Per Wall Second | JAX Memory In Use | Host RAM |
| --- | ---: | ---: | ---: | ---: |
| PQN-VDN | 5.79 | 10,103 | 15.2 GB | 12.7 GB |
| QMIX | 4.50 | 13,008 | 12.1 GB | 18.6 GB |
| Recurrent IPPO | 3.55 | 16,458 | 3.0 GB | 14.2 GB |
| Feedforward IPPO | 3.31 | 17,664 | 2.8 GB | 14.1 GB |
| Recurrent MAPPO | 3.22 | 18,168 | 3.1 GB | 14.5 GB |
| Feedforward MAPPO | 2.98 | 19,628 | 2.8 GB | 13.5 GB |

Hours include setup and the 20 checks with their confirmations. Feedforward
MAPPO's 40M run took 37 minutes (18,241 transitions per wall second, 2.7 GB JAX
memory in use, 8.5 GB host RAM).

### Tournament

The six study 3 policies and the three scripted bots, 200 games per pair on
the test maps (see Search Design), with QMIX and PQN-VDN greedy, in
deterministic GPU mode. Each entrant played 1,600 games. Score counts a win as
1 and a draw as ½. Tier and division come from the rounded Elo. ⭐ marks seed
12259871.

| Rank | Entrant | Seed | Checkpoint | Tier | Elo (95% Interval) | W-D-L | Score | Margin | Kills | Deaths |
| ---: | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Recurrent IPPO | 20261001 | 250,675,200 | Platinum 4 | 1,441 (1,426–1,458) | 1038-336-226 | 0.75 | +5.2 | 18.0 | 12.7 |
| 2 | PQN-VDN | 12259871 ⭐ | 240,156,672 | Gold 2 | 1,358 (1,349–1,368) | 867-419-314 | 0.67 | +1.3 | 14.7 | 12.5 |
| 3 | Recurrent MAPPO | 20261002 | 219,103,232 | Gold 3 | 1,304 (1,288–1,322) | 825-319-456 | 0.62 | +2.3 | 16.9 | 14.8 |
| 4 | QMIX | 12259871 ⭐ | 219,095,040 | Gold 3 | 1,300 (1,288–1,313) | 792-371-437 | 0.61 | +0.6 | 15.8 | 15.5 |
| 5 | Feedforward IPPO | 20261001 | 219,103,232 | Gold 4 | 1,240 (1,225–1,255) | 639-463-498 | 0.54 | +1.8 | 15.9 | 14.1 |
| 6 | Feedforward MAPPO | 12259871 ⭐ | 208,568,320 | Silver 2 | 1,140 (1,126–1,155) | 458-466-676 | 0.43 | −2.7 | 13.1 | 14.2 |
| 7 | tdm-beta | – | – | Silver 4 | 1,043 (1,030–1,055) | 485-77-1038 | 0.33 | −3.0 | 14.7 | 18.1 |
| 8 | tdm-gamma | – | – | Silver 5 | 1,000 (986–1,014) | 435-40-1125 | 0.28 | −2.4 | 15.1 | 18.4 |
| 9 | tdm-alpha | – | – | Bronze 1 | 974 (959–988) | 394-43-1163 | 0.26 | −3.1 | 14.5 | 18.3 |

**Head to head: the row's margin against the column** (row's kills / deaths
per game).

| Row against column | Rec. IPPO | PQN-VDN | Rec. MAPPO | QMIX | FF IPPO | FF MAPPO | Beta | Gamma | Alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Rec. IPPO** | — | +3.2 (14.9/13.6) | +1.0 (17.4/15.9) | +2.2 (17.9/14.9) | +3.4 (17.1/13.8) | +7.5 (17.0/10.6) | +8.7 (19.9/10.6) | +7.6 (19.9/11.6) | +8.1 (19.8/10.9) |
| **PQN-VDN** | −3.2 (13.6/14.9) | — | −3.4 (13.7/15.2) | +3.2 (15.1/11.7) | −1.6 (13.4/13.6) | +5.4 (14.1/8.9) | +4.5 (15.5/10.7) | +0.9 (16.6/13.5) | +4.6 (16.0/11.2) |
| **Rec. MAPPO** | −1.0 (15.9/17.4) | +3.4 (15.2/13.7) | — | +1.1 (17.4/16.0) | +4.4 (17.3/13.6) | +5.9 (14.3/10.4) | +1.7 (18.5/15.8) | 0.0 (17.8/16.5) | +2.8 (18.9/15.2) |
| **QMIX** | −2.2 (14.9/17.9) | −3.2 (11.7/15.1) | −1.1 (16.0/17.4) | — | −2.0 (14.3/16.7) | +3.6 (13.4/12.1) | +3.3 (18.2/14.4) | +3.2 (18.7/15.2) | +3.5 (19.1/15.1) |
| **FF IPPO** | −3.4 (13.8/17.1) | +1.6 (13.6/13.4) | −4.4 (13.6/17.3) | +2.0 (16.7/14.3) | — | +8.4 (12.8/7.1) | +3.2 (19.0/14.9) | +2.4 (18.6/15.1) | +4.2 (19.3/14.0) |
| **FF MAPPO** | −7.5 (10.6/17.0) | −5.4 (8.9/14.1) | −5.9 (10.4/14.3) | −3.6 (12.1/13.4) | −8.4 (7.1/12.8) | — | +3.5 (18.8/14.0) | +2.1 (18.3/14.5) | +3.8 (18.8/13.5) |
| **Beta** | −8.7 (10.6/19.9) | −4.5 (10.7/15.5) | −1.7 (15.8/18.5) | −3.3 (14.4/18.2) | −3.2 (14.9/19.0) | −3.5 (14.0/18.8) | — | +1.6 (19.1/17.2) | −0.7 (17.9/17.6) |
| **Gamma** | −7.6 (11.6/19.9) | −0.9 (13.5/16.6) | 0.0 (16.5/17.8) | −3.2 (15.2/18.7) | −2.4 (15.1/18.6) | −2.1 (14.5/18.3) | −1.6 (17.2/19.1) | — | −1.4 (16.8/18.6) |
| **Alpha** | −8.1 (10.9/19.8) | −4.6 (11.2/16.0) | −2.8 (15.2/18.9) | −3.5 (15.1/19.1) | −4.2 (14.0/19.3) | −3.8 (13.5/18.8) | +0.7 (17.6/17.9) | +1.4 (18.6/16.8) | — |

**Head to head: the row's wins-draws-losses against the column** (200 games
per pair).

| Row against column | Rec. IPPO | PQN-VDN | Rec. MAPPO | QMIX | FF IPPO | FF MAPPO | Beta | Gamma | Alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Rec. IPPO** | — | 73-51-76 | 95-44-61 | 92-55-53 | 86-97-17 | 113-87-0 | 192-1-7 | 195-1-4 | 192-0-8 |
| **PQN-VDN** | 76-51-73 | — | 61-70-69 | 100-80-20 | 68-95-37 | 122-43-35 | 140-40-20 | 140-0-60 | 160-40-0 |
| **Rec. MAPPO** | 61-44-95 | 69-70-61 | — | 104-40-56 | 111-77-12 | 107-82-11 | 123-4-73 | 105-1-94 | 145-1-54 |
| **QMIX** | 53-55-92 | 20-80-100 | 56-40-104 | — | 64-62-74 | 99-94-7 | 180-20-0 | 140-20-40 | 180-0-20 |
| **FF IPPO** | 17-97-86 | 37-95-68 | 12-77-111 | 74-62-64 | — | 66-130-4 | 145-1-54 | 131-0-69 | 157-1-42 |
| **FF MAPPO** | 0-87-113 | 35-43-122 | 11-82-107 | 7-94-99 | 4-130-66 | — | 138-11-51 | 114-18-68 | 149-1-50 |
| **Beta** | 7-1-192 | 20-40-140 | 73-4-123 | 0-20-180 | 54-1-145 | 51-11-138 | — | 180-0-20 | 100-0-100 |
| **Gamma** | 4-1-195 | 60-0-140 | 94-1-105 | 40-20-140 | 69-0-131 | 68-18-114 | 20-0-180 | — | 80-0-120 |
| **Alpha** | 8-0-192 | 0-40-160 | 54-1-145 | 20-0-180 | 42-1-157 | 50-1-149 | 100-0-100 | 120-0-80 | — |

**Expected score on each test map.**

| Entrant | Map 47 | Map 48 | Map 49 | Map 50 | Map 51 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recurrent IPPO | 0.85 | 0.75 | 0.70 | 0.71 | 0.77 |
| PQN-VDN | 0.22 | 0.85 | 0.70 | 0.90 | 0.69 |
| Recurrent MAPPO | 0.48 | 0.60 | 0.68 | 0.75 | 0.56 |
| QMIX | 0.44 | 0.65 | 0.67 | 0.66 | 0.62 |
| Feedforward IPPO | 0.45 | 0.66 | 0.60 | 0.51 | 0.49 |
| Feedforward MAPPO | 0.45 | 0.24 | 0.55 | 0.40 | 0.51 |
| tdm-beta | 0.54 | 0.30 | 0.13 | 0.30 | 0.37 |
| tdm-gamma | 0.51 | 0.27 | 0.15 | 0.17 | 0.32 |
| tdm-alpha | 0.55 | 0.17 | 0.32 | 0.09 | 0.17 |

**Repeated games** (see How The Tournament Works). QMIX and PQN-VDN act
greedily and the scripted bots are deterministic, so within a run a game
between two of these five repeats exactly for a given map and spawn end. Each
such pair's 200 games are copies of at most 10 distinct games (5 maps x 2
spawn ends). Games with a PPO method vary, because PPO actors sample their
actions. The tool marks those ten pairs as deterministic and gives them no
sampling interval, so the Elo intervals of PQN-VDN, QMIX and the bots come from
their games against the PPO methods. Their results against each other rest on
at most 10 distinct games per pair.

**Findings.**

1. **Recurrent IPPO is first** (1,441, Platinum 4) and the only Platinum
   entrant. Its margin is positive against every other entrant. Against
   PQN-VDN its margin is +3.2 but its wins and losses are about even
   (73-51-76, score 0.49): it leads on points, including in drawn games,
   without winning more games.
2. **Every learned policy ranks above every scripted bot.** PQN-VDN (Gold 2),
   recurrent MAPPO and QMIX (both Gold 3), feedforward IPPO (Gold 4) and
   feedforward MAPPO (Silver 2) sit above `tdm-beta` (Silver 4), `tdm-gamma`
   (Silver 5) and `tdm-alpha` (Bronze 1).
3. **The order is not a simple ladder.** PQN-VDN is second overall but loses
   on margin to recurrent MAPPO (−3.4) and feedforward IPPO (−1.6); recurrent
   MAPPO beats PQN-VDN and QMIX but only draws level with `tdm-gamma` on
   margin (0.0).
4. **PQN-VDN depends on the map.** It scores 0.85-0.90 on maps 48 and 50 and
   0.22 on map 47, where all three bots score above 0.5. Because its games
   against the bots repeat, a single game per map and side decides such a
   cell.
5. **Feedforward MAPPO is sixth** (1,140, Silver 2). It beats every scripted
   bot (138-11-51, 114-18-68 and 149-1-50; margins +2.1 to +3.8, with
   18.3-18.8 kills per game) and loses on margin to every other learned
   entrant (−3.6 to −8.4). 466 of its 1,600 games are draws, 436 of them
   against the five learned entrants, most against feedforward IPPO (130 of
   200).

### Teamwork In The Tournament

**Question.** Which teamwork separates the strongest Season 0 entrants from
the weakest?

**Method.** The tournament saved priority metrics for every game. The public
`tournament_analysis` stage played its schedule again on the same RTX 5090 in
XLA's deterministic mode, recording all full metrics for every game. All
7,200 games repeated exactly. One call to
[`system_metrics`](../evaluation/workflows.md#compare-every-systems-metrics)
then reads every entrant's games from the side it played, as Team A or Team B,
over all 1,600 of its games, and its point margin against each opponent equals
the saved head-to-head result. Games combine by each column's catalog rule:
counts per game, fractions with a saved denominator as total numerator over
total denominator, averages such as teammate distance weighted by their
observations. 95%
intervals resample whole spawn-swapped game pairs within each opponent and
map, as Elo does; matchups that repeat exactly (see Repeated games above) add
no width. A rank correlation of +1 means the measure rises with every step up
in Elo and −1 that it falls; the stage also ranks all 145 team-level metrics
in the catalog this way (`season_0_teamwork_screen.csv`), an exploratory
screen, and tests each one. Learned teams draw 20-29% of their games and the
scripted bots 2.5-4.8%, and a draw runs the full 300 ticks, so learned teams'
games last longer (238-252 ticks on average, against 205-212). The screen
therefore compares per-game totals (counts, health, ticks) per 100 ticks;
fractions, distances and averages keep their scale (`scale`). The tests:

- **Exact p-value.** The share of all 362,880 orderings of the nine ratings
  whose rank correlation is at least as far from zero (`p_value_all`; the
  smallest possible is 2 / 9!, about 0.0000055), and the same over the six
  learned entrants (`p_value_learned`; 720 orderings, smallest 0.0028).
- **Corrections for the whole screen.** `p_value_all` after Holm's correction
  (`holm_p_value_all`) and after the Westfall-Young max-statistic correction
  (`max_t_p_value_all`), over the 138 metrics whose values differ between
  entrants. The second allows for metrics that move together.
- **Winner check.** Over the 5,933 decided games, the share in which the
  winner had more of the metric than the loser (`winner_higher_share`).
- **Inside a matchup.** Within each cell of one entrant, opponent, map and
  spawn end with at least two wins and two losses, the chance that one of
  its wins shows more of the metric than one of its losses, averaged over
  cells and weighted by wins times losses (`wins_over_losses`). Holding the
  opponent fixed matters: a team's wins come mostly against weak opponents.
  0.5 means the metric does not change with the result inside a matchup.

Matchups between methods that never sample (the bots, greedy QMIX and
PQN-VDN) repeat the same 10 games 20 times, and the two game checks count
each copy.

The class tables (`season_0_teamwork_by_class.csv`) also give the share of
each ally class's ticks inside a Mage or Warrior aura, the share of the
Priest's Freedom uses each class received, and the distance between each pair
of classes.

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m marl_battlegrounds.experiments.tournament_analysis \
  tournament_dir=artifacts/paper-experiments/hyperparameter-search/tournament-250m/runs/tournament_20261005_082220_ae54a3f293be \
  field=artifacts/paper-experiments/hyperparameter-search/tournament-250m/field.json \
  output_dir=artifacts/paper-experiments/tournament-analysis-season-0/full-metrics-timed/out
```

To redo only the analysis from that saved replay, as for the results below
(`field` is required but unused when `replay_dir` is given):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m marl_battlegrounds.experiments.tournament_analysis \
  tournament_dir=artifacts/paper-experiments/hyperparameter-search/tournament-250m/runs/tournament_20261005_082220_ae54a3f293be \
  field=artifacts/paper-experiments/hyperparameter-search/tournament-250m/field.json \
  replay_dir=artifacts/paper-experiments/tournament-analysis-season-0/full-metrics-timed/out/replay/tournament_20261007_095226_96f9aa8c5e85 \
  output_dir=artifacts/paper-experiments/tournament-analysis-season-0/full-metrics-timed/analysis-system-metrics
```

**Cost.** Both runs used the same RTX 5090, 128 parallel games, deterministic
mode and the same code, on the same day. Time runs from the creation of the
run folder to the saved results; peak GPU memory was sampled every 0.5 s with
preallocation off.

| Metrics recorded | Time | Peak RAM | Peak GPU memory | Disk |
| --- | ---: | ---: | ---: | ---: |
| Priority, as in the tournament | 488 s (8.1 minutes) | 6.5 GB | 924 MiB | 40 MB |
| Full, for every game | 834 s (13.9 minutes) | 9.5 GB, with an earlier analysis | 980 MiB | 392 MB |

The analysis alone, from the saved replay, took 15.6 seconds on the CPU and
0.61 GB of peak RAM for the whole command, screen tests included; the two
`system_metrics` calls took 8.6 seconds.

**Results.** Entrants from the highest Elo down; percentages are shares.
Values and rank correlations in this table are per game; the findings below
use the screen's per-tick values and length-free shares where game length
matters.

| Measure | Rank correlation with Elo | Rec. IPPO | PQN-VDN | Rec. MAPPO | QMIX | FF IPPO | FF MAPPO | Beta | Gamma | Alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Elo | | 1,441 | 1,358 | 1,304 | 1,300 | 1,240 | 1,140 | 1,043 | 1,000 | 974 |
| Margin per game | | +5.2 | +1.3 | +2.3 | +0.6 | +1.8 | −2.7 | −3.0 | −2.4 | −3.1 |
| Kills / deaths per game | | 18.0 / 12.7 | 14.7 / 12.5 | 16.9 / 14.8 | 15.8 / 15.5 | 15.9 / 14.1 | 13.1 / 14.2 | 14.7 / 18.1 | 15.1 / 18.4 | 14.5 / 18.3 |
| Ticks with two or more attackers, per game | +0.42 | 151 | 121 | 142 | 139 | 134 | 121 | 123 | 126 | 123 |
| Kills with two or more helpers | +0.85 | 83.8% | 89.1% | 80.3% | 85.7% | 77.3% | 78.7% | 70.5% | 69.0% | 71.3% |
| Kills on enemies a Hunter had slowed | +0.85 | 52.1% | 57.3% | 47.6% | 53.1% | 42.2% | 45.6% | 31.8% | 35.3% | 32.6% |
| Kills helped by an Ultimate | −0.82 | 11.3% | 10.3% | 10.8% | 9.5% | 13.0% | 12.3% | 20.1% | 21.0% | 20.9% |
| Distance between teammates (map units) | −0.90 | 2.66 | 2.44 | 3.16 | 3.15 | 2.96 | 3.45 | 3.90 | 4.11 | 3.96 |
| Teammate time inside a Mage aura | +0.88 | 64.9% | 65.4% | 55.9% | 58.1% | 55.9% | 54.2% | 46.1% | 44.0% | 46.3% |
| Teammate time inside a Warrior aura | +0.87 | 53.6% | 63.0% | 49.0% | 49.6% | 50.7% | 46.5% | 40.2% | 37.8% | 40.5% |
| Damage Mage auras added, per game | +0.97 | 434 | 356 | 384 | 350 | 350 | 339 | 261 | 265 | 261 |
| Damage Warrior auras prevented, per game | +0.97 | 470 | 433 | 405 | 427 | 397 | 380 | 341 | 339 | 340 |
| Useful share of Priest healing | +0.83 | 76.8% | 62.1% | 65.0% | 70.4% | 68.2% | 60.2% | 50.7% | 51.3% | 49.7% |
| Chances to save a dying ally that were taken | −0.80 | 66.1% | 77.3% | 60.9% | 81.3% | 78.2% | 75.8% | 94.4% | 93.3% | 94.6% |
| Deaths in the team's own Red Zone, per game | +0.62 | 0.59 | 1.35 | 0.58 | 0.36 | 0.70 | 1.94 | 0.30 | 0.22 | 0.17 |
| Basic abilities used, per game | +0.80 | 652 | 570 | 614 | 607 | 613 | 575 | 548 | 552 | 538 |
| Ultimate abilities used, per game | +0.45 | 31.0 | 26.9 | 29.6 | 28.3 | 31.6 | 30.5 | 25.1 | 29.2 | 24.9 |

**Who the Priest healed, and who took the damage** (shares of the team's
useful Priest healing and of the damage it received).

| Entrant | Healing: Mage | Warrior | Hunter | Rogue | Priest | Damage taken: Mage | Warrior | Hunter | Rogue | Priest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recurrent IPPO | 34.6% | 32.6% | 13.8% | 8.5% | 10.5% | 25.9% | 34.2% | 13.2% | 19.1% | 7.6% |
| Feedforward IPPO | 25.3% | 26.3% | 12.4% | 20.2% | 15.9% | 19.9% | 29.0% | 15.0% | 25.8% | 10.4% |

**Findings.** Per-game values are in the table above; the screen's per-tick
values and tests are in `season_0_teamwork_screen.csv`.

1. **Activity does not separate strong teams from weak ones.** Per tick,
   kills (+0.02), Basic uses (−0.17), Ultimate uses (−0.13) and ticks with
   two or more attackers (−0.02) show no trend across the nine. Every
   Ultimate has a 30-tick cooldown, so a team can use at most about 0.17 per
   tick; the nine use 0.11 to 0.14.
2. **One trend survives the whole screen, narrowly.** The share of kills a
   Warrior's Ultimate helped with falls from 13.9-14.4% for the bots to
   3.1-3.7% for the learned teams (−0.95; Holm p = 0.049, max-statistic p =
   0.029; −0.89 among the learned six). Recurrent MAPPO and QMIX are 4 Elo
   apart; if they swapped places, this metric would fall below Holm's line
   and the useful share of Basic healing would rise above it.
3. **Strong teams hold together.** Teammates stand 3.90-4.11 map units apart
   in the bots and 2.45-3.45 in the learned teams (−0.90). The auras pay off
   accordingly: Warrior auras prevent 10.0-11.8 points per 100 points of
   damage learned teams take, against 8.7-9.0 for the bots
   (`warrior_aura_protection_share`, +0.92), and Mage auras add 8.6-9.8% of
   learned teams' damage, against 7.1-7.2% (`mage_aura_damage_share`,
   +0.90). Per game, both payoffs reach +0.97, partly because learned teams'
   games last longer.
4. **Freedom reaches slowed allies.** Every Priest heal gives its target
   Freedom, a short floor on its speed. In learned teams Freedom held up a
   slowed ally's speed on 4.9-6.3% of the ticks it was active, against
   3.5-3.8% for the bots (+0.93, max-statistic p = 0.055), and this orders
   the six learned teams almost perfectly (+0.94).
5. **A few measures order the six learned teams.** Damage to enemies a Hunter
   had slowed and kills a Warrior's Basic helped, both per tick, order them
   perfectly (+1.00, p = 0.0028); Freedom protection, Mage aura damage and
   Warrior aura protection per tick reach +0.94. Warrior aura protection per
   tick is only +0.68 over all nine: the bots, hit more often, prevent as
   much per tick (160-165 per 100 ticks) as feedforward IPPO and MAPPO
   (159 and 151).
6. **Most strong trends separate the bots from the learned teams.** 70 of the
   138 metrics put all three bots on one side of all six learned teams, and
   49 of the 54 with p < 0.05 over the nine are of this kind. The three bots
   are near copies, so the field acts smaller than nine. Among the learned
   six, 9 metrics reach p < 0.05, where chance alone gives about 5.
7. **Inside a matchup, teamwork also follows the result.** With the opponent,
   map and spawn end fixed, kills (0.94) and deaths (0.06) move with the
   result; a team also stands closer (0.43), gains more Mage aura damage per
   tick (0.77) and more from Freedom (0.58) in the games it wins, and
   prevents less per tick with Warrior auras (0.41), because a winning team
   is hit less. These measures are partly the state of a game, not only a
   fixed style.
8. **Learned teams bring the back line into the auras** (class tables, not
   tested). The bots keep their Priest inside the Mage's aura 23-24% of its
   ticks and their Hunter inside the Warrior's 16-18%; learned teams 46-59%
   and 34-61%. The bots' Priests give the Warrior 2-3% of their Freedom uses
   and learned Priests 8.5-13.5%, which separates the bots from learned teams
   but does not order the learned ones.
9. **The metrics recover a known rule.** TDM-Gamma never damages an enemy
   with two or more Trap ticks left. Its Hunters trapped enemies 10,355 times
   in its 1,600 games, and its team broke none of those Traps with damage.
10. **Limits.** These are correlations across one field of nine at Season
    0's level of play, an exploratory screen of 145 metrics. A trend may
    weaken or reverse among stronger teams. A measure that follows strength is
    a diagnostic, not a recipe: rewarding it directly, for example through
    reward shaping, may teach the measure without the skill, such as a team
    that stays close but stops fighting.

### Final Policies

Every setting of each method's tournament policy, in tournament order. Values
come from the runs' saved settings; ratings from the tournament.

| | Recurrent IPPO | PQN-VDN | Recurrent MAPPO | QMIX | Feedforward IPPO | Feedforward MAPPO |
| --- | --- | --- | --- | --- | --- | --- |
| **Rank (of 9)** | **1** | **2** | **3** | **4** | **5** | **6** |
| **Tier** | Platinum 4 | Gold 2 | Gold 3 | Gold 3 | Gold 4 | Silver 2 |
| **Elo (95% interval)** | 1,441 (1,426–1,458) | 1,358 (1,349–1,368) | 1,304 (1,288–1,322) | 1,300 (1,288–1,313) | 1,240 (1,225–1,255) | 1,140 (1,126–1,155) |
| **W-D-L (1,600 games)** | 1038-336-226 | 867-419-314 | 825-319-456 | 792-371-437 | 639-463-498 | 458-466-676 |
| **Margin / kills / deaths per game** | +5.2 / 18.0 / 12.7 | +1.3 / 14.7 / 12.5 | +2.3 / 16.9 / 14.8 | +0.6 / 15.8 / 15.5 | +1.8 / 15.9 / 14.1 | −2.7 / 13.1 / 14.2 |
| Run (recipe) | `reference` | B768 T128, reward x10 | `reference` | B768 T32, `q_lr = 0.0001`, `update_period = 128` | `reference` | Team reward, `actor_lr = 0.0003` |
| Seed | 20261001 | 12259871 ⭐ | 20261002 | 12259871 ⭐ | 20261001 | 12259871 ⭐ |
| Checkpoint (transitions) | 250,675,200 | 240,156,672 | 219,103,232 | 219,095,040 | 219,103,232 | 208,568,320 |
| Memory | Recurrent | Recurrent | Recurrent | Recurrent | None | None |
| Critic or value | Local critic per agent | VDN sum of agent Q-values | Centralized critic | QMIX mixer of agent Q-values | Local critic per agent | Centralized critic |
| Reward | Per-agent combat and support | Team combat and support | Per-agent combat and support | Team combat and support | Per-agent combat and support | Team combat and support |
| Games in parallel | 512 | 768 | 512 | 768 | 512 | 512 |
| Rollout length | 32 | 128 | 32 | 32 | 32 | 32 |
| Epochs | 4 | 4 | 4 | 48 | 4 | 4 |
| Minibatches | 2 | 12 | 2 | 128 sampled sequences of 20 steps | 2 | 2 |
| Learning rate | Actor 1e-4, critic 2.5e-4 | 2.5e-4 to 0 by 40M; restarted at 2.1e-4 to 0 at 250M | Actor 1e-4, critic 2.5e-4 | 1e-4, constant | Actor 1e-4, critic 2.5e-4 | Actor 3e-4, critic 2.5e-4 |
| Gamma | 0.999 | 0.99 | 0.999 | 0.99 | 0.999 | 0.999 |
| GAE or TD lambda | 0.95 | 0.85 | 0.95 | – | 0.95 | 0.95 |
| Clip / entropy / value coefficient | 0.2 / 0.01 / 0.5 | – | 0.2 / 0.01 / 0.5 | – | 0.2 / 0.01 / 0.5 | 0.2 / 0.01 / 0.5 |
| Max gradient norm | 0.5 | 1.0 | 0.5 | – | 0.5 | 0.5 |
| Exploration | Samples its policy | Epsilon 1.0 to 0.01 over the first 2.2M transitions (5% of the 40M run) | Samples its policy | Epsilon 1.0 to 0.05 over 2M transitions | Samples its policy | Samples its policy |
| Target network | – | – | – | Hard copy every 128 updates | – | – |
| Replay | – | – | – | 172 time steps per game x 768 games (132,096 rows); learning starts after 32 per game | – | – |
| Reward scale | – (normalized) | 10 | – (normalized) | 1 | – (normalized) | – (normalized) |
| Normalization | Value normalization | BatchNorm | Value normalization | Mixer LayerNorm on the game state | Value normalization | Value normalization |
| Input scale | 0.01 | 1.0 | 0.01 | 0.01 | 0.01 | 0.01 |
| Played in evaluation | Sampled | Greedy | Sampled | Greedy | Sampled | Sampled |

Shared by all six: 40,108,032 transitions in their 40M runs, continued to
250,675,200 in study 3; the opponent mix of Opponents; Red Zone depth 5.0 with
built-in shaping off; weights shared across all five agents; the left spawn
frame; validation on maps 42, 43, 44 and 46 with checkpoints chosen by point
margin. The PPO methods also use Adam epsilon 1e-5 and 2 groups. The same
values, with every remaining saved setting and each policy's file paths, are in
`final_policies.csv`. The training scripts' default recipe (`recipe=default`
in [Hydra training](hydra.md)) applies each method's settings and seed from
this table.

## Our Reading

This section gives our interpretation of the results. Each point says what the
data shows and what is still a hypothesis.

**Reward scale 10 changes learning, not incentives.** Multiplying every reward
by 10 keeps the balance between the reward parts and does not change which
actions are best. It changes the size of the Q-learning targets. At scale 1 a
kill credit is worth 0.05, so the value gap between fighting and hiding is
small next to the noise of early random weights, exploration and replay
sampling, and hiding avoids the −0.05 death penalty almost for free. At scale
10 that gap is ten times larger. The gain is not simply larger steps. QMIX's
Adam and PQN-VDN's gradient clipping followed by RAdam take steps whose size
does not depend on the gradient's size; PQN-VDN's gradients were above the
clipping limit of 1.0 in nearly every update at both scales, and QMIX, which
has no clipping, gained too. Our hypothesis is that each step changes the
Q-values by a similar amount at either scale, and only at scale 10 is that
amount small next to the reward differences. We did not test this directly.

**Faster target copies.** QMIX computes its learning targets with a frozen copy
of its network and copies the current weights into it every `update_period`
optimizer updates. Halving the period (every 65,536 new transitions instead of
131,072) lets what the network has just learned reach the targets sooner. It
gave +5.0 in study 2 on all three seeds without visible instability, and its
seed 12259871 run became QMIX's strongest.

**PQN-VDN's dip after the restart.** Over its last 8M transitions the 40M run
gained 3.6 points (−8.6 at 32M to −5.0 at 40M) while its learning rate fell
from 5e-5 to about 0. Restarting at 0.00021, about the rate it had around 6M
transitions and about 8 times its average rate over those last 8M, was
followed by a fall below that late gain: −10.2 at 61M. The restart is the
likely cause. Loss and gradient sizes were no larger than in the parent run
(loss 1.5-3.0 and mean gradient norm 6-15 from 40M to 61M, against 1.7-3.5
and 7-21 after 4M in the parent), which suggests the dip was not
instability. The policy recovered before the new
rate had fallen much: −4.2 at 72M, when the rate was still 85% of its restart
value, and −1.2 at 82M. It went on to the study's best checkpoint. QMIX, whose
rate is constant, shows no comparable dip.

**Feedforward MAPPO and the team reward.** Under the per-agent form, 7 of
feedforward MAPPO's 9 study 1 runs selected a checkpoint at 8-24M, and its
routine margins stopped improving after about 16M (study 1 learning curves).
Its team-reward run selected its last 40M check and kept improving in study 3,
from −8.4 at 40M to about 0 by 124M and +2.4 at 209M, with kills rising from 9
to 15-17 per game. Our reading, which this search did not test directly:
feedforward MAPPO's critic predicts one value per game for all five agents.
Under the per-agent form the five agents' returns differ (kill credit goes
only to the credited agents), so one shared value cannot match each agent's
return, while the team form gives every agent the same return to predict. A
feedforward actor also cannot remember events it no longer sees, which may
make a mismatched learning signal harder to use. Recurrent MAPPO shares the
critic design and improved under the per-agent form, so the reward form alone
may not explain the difference; we did not train recurrent MAPPO with the team
reward.

**Greedy evaluation of the Q methods.** Greedy play is the usual way to
evaluate Q-learners, but against deterministic opponents it reduces a pair to
at most 10 distinct games on five maps, and one changed game moves 20 results.
The same structure applied during training: validation played the Q methods
greedily, so their validation games against the bots repeated, a
confirmation's 64 games per bot were 8 distinct games, and checkpoint choice
for QMIX and PQN-VDN rested on fewer distinct games than for the PPO methods.
Their games against the pooled 250M model, a PPO policy, did vary.

**Fight avoidance was the most common failure, and margin alone hides it.**
PQN-VDN's weakest results shared one signature: few kills, few deaths and many
draws (the B768 T32 shape, and the exploration, gamma 0.995 and LayerNorm
changes of study 2). Point margin rewarded some of these phases. Reporting
kills and deaths beside every margin, the engagement floor and choosing runs
on margin, kills and deaths together kept such policies out of the final
field.

**Local critics ranked higher in the PPO family after long training.** Recurrent
IPPO beat recurrent MAPPO in study 3 and the tournament (Platinum 4 against
Gold 3), although recurrent MAPPO is the learner of the pooled 250M model, and
feedforward IPPO ranked above feedforward MAPPO (Gold 4 against Silver 2). At
40M the recurrent order was reversed (recurrent MAPPO `reference` −8.8 against
−10.4). With one run per method, and feedforward MAPPO on a different reward
form, these are single observations, not a method comparison.

## Limits

- Three seeds per recipe find only large differences.
- Study 2 changes one setting at a time, so it cannot show how two settings
  interact.
- The PPO family uses gamma 0.999; the final Q recipes use 0.99 (0.995 was
  tested in study 2 and did not pass the pick rules). This difference is part of each method's
  recipe, not a fair test of gamma.
- The team reward matches each event's price, not the total shaped reward per
  game (see Reward).
- Feedforward MAPPO trains on the team form of the reward and the other three
  PPO methods on the per-agent form, so comparisons within the PPO family mix
  two reward forms.
- The panel's learned model shares its learner with recurrent MAPPO.
- Each run is represented by its confirmed checkpoint with the best point
  margin. That rule can select a phase in which the policy avoids fights
  (study 1 finding 2, study 2 finding 2); kills and deaths are reported beside
  every margin so this stays visible.
- Reward scale was searched only for QMIX and PQN-VDN. The PPO family
  normalizes its advantages and value targets, so a constant reward scale
  cancels out there (see Reward).
- Study 3 and the tournament have one policy per method, so differences
  between methods there are single observations.
- The tournament covers the five test maps only; pairs between deterministic
  entrants rest on at most 10 distinct games (see Tournament).
- QMIX and PQN-VDN are rated as greedy players; the ratings do not show how
  they play with exploration.
- Training ran in XLA's normal GPU mode, so retraining with the same settings
  and seeds does not give the same weights (see Training Protocol). How far
  such a retrain drifts was not measured.
