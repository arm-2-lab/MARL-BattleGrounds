# Six-Method Hyperparameter Search

This study gives each of the six built-in baselines a tested starting recipe
under one shared training protocol: recurrent MAPPO, feedforward MAPPO,
recurrent IPPO, feedforward IPPO, recurrent QMIX and recurrent PQN-VDN. It
measures complete recipes on a fixed budget. Each method's strongest run is
then continued to 250M transitions and enters the tournament. It does not
claim to find a global optimum. The design was fixed before study 1 started.
Decisions made between studies, from saved results, are listed with their
reasons in Decisions During The Search. Results are added as each study
finishes, and every number comes from a saved
report file.

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
first team to 20 points wins, and a game that reaches 300 ticks is a draw. A
kill is worth 1 point. A kill inside the victim's own Red Zone (depth 5.0) is
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
| `enemy_death_credit` | +0.05 per enemy death | Each credited agent (the killer and any Priest whose healing helped) |
| `enemy_red_zone_death_credit` | +0.10 extra per enemy death in its own Red Zone | Each credited agent |
| `enemy_damage_hp` | +0.001 per enemy health point | The agent that dealt the damage (capped so overkill does not pay) |
| `priest_effective_healing_hp` | +0.001 per health point healed | The Priest, effective healing only |
| `own_team_deaths` | -0.05 per teammate death | Every configured agent |
| `own_red_zone_deaths` | -0.10 extra per teammate death in our own Red Zone | Every configured agent |
| `priest_heal_range_pairs` | +0.001 per living Priest and ally pair within Basic range, both outside our Red Zone, per tick | Every configured agent |
| `own_zone_ticks` | -0.002 per tick | Each living agent standing in our own Red Zone |

The first four parts and `own_zone_ticks` belong to the agent who earned them.
The other three are shared: every agent receives the same amount.

**PPO family:** [`combat_support_reward`](../../src/marl_battlegrounds/training/reward_recipes.py).
Each agent learns from its own reward: the game reward, the shared parts and
its own parts.

**QMIX and PQN-VDN:** [`combat_support_team_reward`](../../src/marl_battlegrounds/training/reward_recipes.py).
These methods learn one team value from one team reward. Our learners form
that reward as the average over Team A slots
([`team_task_reward`](../../src/marl_battlegrounds/baselines/qmix.py)). With
the per-agent form, that average would divide each agent's own parts by the
team size. The team form gives every slot the same amount instead: the shared
parts once, plus the team total of every agent's own parts. After the average,
each event pays the team exactly what it pays the PPO agent it happens to:

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
PPO agent's, because it adds five agents' events. Measured totals per game are
reported with the results.

**Reward scale (QMIX and PQN-VDN only).** `reward_scale` multiplies the whole
team reward, the game reward and every part above together, before the
learner computes its targets. With `reward_scale = 10`, a win is worth +10, a
kill +0.5 and a teammate's death −0.5. The balance between the parts does not
change, and a constant positive scale does not change which actions are best.
It changes only the size of the learning targets: QMIX and PQN-VDN learn raw
Q-values against raw reward targets, so the scale sets how large the
training errors and gradients are compared with initial noise and gradient
clipping. Logged rewards, game results, validation and selection stay in true
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
| 20% | One of its 12 newest saved copies, uniformly; one copy is saved every 1,376,256 steps, so the 12 copies span the last 41% of training. Until the first copy exists, this share plays the current weights. |
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
| Steps per run | 40,108,032 environment transitions (divisible by every batch size times rollout length used) |
| Training state | Recovery saves every 25 PPO updates or 1,600 Q optimizer steps; a crash or Ctrl+C resumes from the last save |
| Recording | Off. No full metrics or Viewer replays during training; rich evaluation runs afterwards from frozen checkpoints. |
| Hardware | Two RTX 5090s, one run per card |

## Validation And Selection

| Setting | Value |
| --- | --- |
| Panel | `tdm-alpha`, `tdm-beta`, `tdm-gamma` and `recurrent-mappo-combat-support-pooled-step-250675200`, a recurrent MAPPO model trained for 250,675,200 steps with this opponent mix |
| Maps | 42, 43, 44, 46 |
| Routine checks | At step 0 and after every 10% of training: 8 games per map per opponent (128 games) |
| Confirmation | The two best routine checks play 16 more games per map per opponent |
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

The search runs as two studies, one after the other, each on both cards. The
search then ends with study 3: one 250M run per method.

| Study | Runs | What it decides |
| --- | ---: | --- |
| 1, `shapes_and_ppo` | 54 | QMIX and PQN-VDN batch size and rollout length; each PPO method's final recipe |
| 2, `q_settings` | 30 | One-at-a-time changes around each Q method's chosen shape |
| 3, `continue_to_250m` | 6 | Each method's strongest single run, judged by point margin, kills and deaths together, continued from its saved state to 250,675,200 transitions; these policies enter the tournament |

**Seeds.** Every recipe in studies 1 and 2 runs on the same three seeds:
12259871, 20261001 and 20261002. Seed 12259871 trained the pooled 250M model.
A planned third study on fresh seeds was dropped (2 October 2026): the search
feeds a tournament of the 250M policies, not a comparison of recipe means.
A seed fixes the starting weights and the
random draws; it does not carry results across code or settings.

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

**Study 2 recipes** (3 runs each, declared on 2 October 2026 after study 1).

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

**Study 3: the 250M continuations** (declared 3 October 2026 after study 2,
to be confirmed before launch).

Each method's strongest single run of studies 1 and 2 is continued with
`extend_training`. Strongest means best on point margin, kills and deaths
together, not the best seed of the recipe with the best three-seed mean: with
three seeds, recipe means are noisy, and the tournament receives one run per
method, not a recipe. For four methods the two choices agree. For QMIX and
feedforward MAPPO the strongest run comes from a recipe whose mean lost (see
Final Recipes). Each run continues from its full learner checkpoint at 40,108,032 transitions,
the only full learner state a run keeps, by 210,567,168 transitions to
250,675,200, the length of the pooled 250M model. The continuation keeps the
learner, optimizer, replay or recent rows, ongoing games, memory and random
streams. It is one training lineage with its parent, not a new seed, and its
steps continue from the parent's.

| Method | Parent Run: Margin / Kills / Deaths | Parent's Selected Checkpoint |
| --- | --- | ---: |
| PQN-VDN | B768 T128, `reward_scale = 10`, seed 12259871 ⭐: −5.1 / 13.1 / 17.1 | 40,108,032 |
| QMIX | B768 T32, `q_lr = 0.0001`, `update_period = 128` (`target-faster`), seed 12259871 ⭐: −2.8 / 12.8 / 15.1 | 40,108,032 |
| Recurrent IPPO | `reference`, seed 20261001: −8.5 / 11.1 / 15.4 | 40,108,032 |
| Recurrent MAPPO | `reference`, seed 20261002: −7.9 / 8.4 / 14.9 | 40,108,032 |
| Feedforward IPPO | `reference`, seed 20261001: −10.7 / 5.7 / 16.4 | 40,108,032 |
| Feedforward MAPPO | `actor-lr-3e-4`, seed 20261002: −8.1 / 8.0 / 15.2 | 24,068,096 |

| Setting | Value |
| --- | --- |
| Inherited | Reward, opponent mix and fixed models, exploration, validation panel and maps, seed pairs, actor retention |
| Routine checks | 20, one every 5% of the added budget (about every 10.5M transitions); the last at 250,675,200 |
| Selection | As before. Parent checkpoints stay eligible, so a run's selected checkpoint can come from before 40M if nothing later beats it |
| PPO and QMIX learning rates | Constant in the parent runs; unchanged |
| PQN-VDN learning rate | Restarts at 0.00021 and falls linearly to 1e-10 at 250,675,200 (below) |
| Order | PQN-VDN, QMIX, then the PPO methods (longest first) |
| Code | The same source as the study 2 runs |

**Learning rates in the continuations.** The four PPO methods and QMIX use
constant learning rates, so their continuations keep them. PQN-VDN's rate falls
linearly over a run: its 40M parent went from 0.00025 to about 0 (1.3e-8 after
19,536 optimizer steps), and a continuation does not stretch that decay.
Continuing at the parent's final rate would leave PQN-VDN almost unable to
learn for the remaining 210M transitions. The continuation therefore declares
a new linear decay: it starts at 0.00021 and reaches 1e-10 at the last of its
102,816 optimizer steps (2,142 learning blocks of 4 epochs and 12
minibatches), at 250,675,200 transitions. The starting value is the rate a
PQN-VDN run planned for 250M from the start would have at 40,108,032
transitions, 16% of the way: 0.00025 x (1 - 0.16) = 0.00021. The continuation
thus follows that run's schedule from 40M to 250M; only the first 40M differ,
where the parent's rate fell faster.

**Validation in the continuations.** 20 routine checks, one every 5% of the
added budget (about every 10.5M transitions), the last at 250,675,200, each
with the same 128 games as before. Every 4M, as in studies 1 and 2, would mean
53 checks. Selection is unchanged: the confirmed check with the highest point
margin represents the run. The parent's checkpoints stay eligible, so a run
whose continuation never beats its 40M-or-earlier best is represented by that
earlier checkpoint, and the report says so.
| Planned cost | About 24 GPU-hours, about 12 hours on two cards, from study 1 and 2 speeds |

## Decisions During The Search

The design above was fixed before study 1. These decisions were made between
studies, from the saved results, and each is applied in the sections that
follow. They are listed in order.

| # | Decision | When | Reason |
| ---: | --- | --- | --- |
| 1 | Every final policy must fight; a recipe or run that wins margin by avoiding fights is not chosen | After study 1 | PQN-VDN's highest-margin shape (B768 T32, −9.7) had 0 kills and 0 deaths in every game against `tdm-alpha` and `tdm-beta`: it kept away and the bots never reached it. Point margin cannot tell "lost 0-10 while hiding" from "lost 10-20 while fighting" |
| 2 | Kills and deaths are reported beside every point margin | After study 1 | The same case: margin alone made avoidance look like the best shape |
| 3 | Study 2 picks add an engagement floor: at least 2 kills per game against each scripted bot | Before study 2 | Blocks recipes that avoid fights. Every PPO and QMIX recipe of study 1 clears it easily (4-11 kills per game) |
| 4 | Study 2's QMIX base is B768 T32 with `q_lr = 0.0001`, not the declared B768 T32 pick | Before study 2 | Best on every measure in study 1 (−10.2, 6.2 kills, still climbing at 40M); it missed the 2-point rule by 1 point. Its study 1 runs serve as the comparison runs |
| 5 | Study 2's PQN-VDN base is B768 T128, not the declared B768 T32 pick | Before study 2 | B768 T32 won by avoiding fights (decision 1). B768 T128 made the most kills of the PQN-VDN shapes and its selected checkpoints all came late (36-40M) |
| 6 | At most one passing change is kept per method, the one with the highest mean | Before study 2 | A combination of changes would have no trained run to continue to 250M |
| 7 | The planned third study on fresh seeds is dropped | Before study 2 | It would have estimated unbiased recipe means for a comparison of methods. The search instead feeds a tournament, which ranks the final policies directly |
| 8 | One run per method is continued to 250,675,200 transitions from its saved 40M state, not retrained | Before study 2 | The tournament needs each method's strongest policy at the length of the pooled 250M model. Continuing keeps the run's full history; 40M is the only full learner state a run keeps |
| 9 | Reward scale is searched for QMIX and PQN-VDN only | Study 2 design | The PPO family normalizes its advantages and value targets, so a constant reward scale cancels out there (see Reward) |
| 10 | Each method's continued run is its strongest single run, judged by point margin, kills and deaths together, from any recipe of studies 1 and 2 | After study 2 | Three-seed recipe means are noisy, and the tournament receives one run per method, not a recipe. This changes two methods: QMIX `target-faster` seed 12259871 (−2.8, 12.8 kills, 15.1 deaths) replaces `reward-x10` seed 12259871 (−3.9, 12.9, 16.4), and feedforward MAPPO `actor-lr-3e-4` seed 20261002 (−8.1, 8.0 kills) replaces `reference` seed 20261002 (−9.9, 6.6 kills) |
| 11 | PQN-VDN's continuation restarts its learning rate at 0.00021 with a new linear decay to 250M | After study 2 | Its rate had decayed to about 0 by 40M, and a continuation does not stretch that decay. 0.00021 is where a run planned for 250M would be at 40M (see Study 3) |
| 12 | The continuations check every 5% of the added budget (20 checks) and keep the parent's checkpoints eligible | After study 2 | Finer checks find the best checkpoint over 210M added transitions; a continuation must beat its earlier best to replace it |
| 13 | Every saved checkpoint of every run is kept for release; past self-play copies are not | After study 1 | The copies sit between saved checkpoints, so they add no distinct strength level to a released population. They stay in the run folders |

## Inputs

The actor features reach about 300 (99% of nonzero values are at most 200).
QMIX's Q-network has no input normalization, so at the donor's input scale of
1.0 its first layer starts with activations about 14 times the intended size,
which pushes its memory gates to their limits. With `input_scale = 0.01`, the
value the PPO baselines use, they start near 0.14. PQN-VDN normalizes its
inputs with BatchNorm, so its input scale nearly cancels. QMIX's mixer
normalizes the game state with its own LayerNorm.

## Compute

Planned cost, from measured training speeds (QMIX 13,169, PQN-VDN 9,641 and
PPO 17,917-21,634 transitions per second on one RTX 5090) plus setup and
validation: about 49 GPU-hours for study 1 and 29-32 for study 2. The six
250M continuations need about 25 more. Measured: 44.8 GPU-hours for study 1
and 32.8 for study 2 (77.6 together). Measured costs and peak memory per
method are reported with each study's results.

## Results

Every number below comes from saved run records. Each run is represented by
its selected checkpoint: its best routine check by point margin. That
checkpoint was then scored in 256 confirmation games: 16 per map and opponent,
on maps 42, 43, 44 and 46, against the pooled 250M model, `tdm-alpha`,
`tdm-beta` and `tdm-gamma`. Margin is our points minus theirs per game, with
every map and opponent weighted equally. Kills are our kills per game and
deaths are the opponent's kills per game. Red Zone kills are points minus
kills. W-D-L counts wins, draws and losses over all confirmation games (768
per recipe, 256 per run); a draw is any game that reaches 300 ticks, whatever
the score. ⭐ marks seed 12259871, the seed that trained the pooled 250M model.
Margin alone can hide fight avoidance, so kills and deaths are reported beside
it everywhere.

### Data Files

All result data is kept locally with the experiment outputs, one folder per
study, under `artifacts/paper-experiments/hyperparameter-search/results/`
(`shapes-and-ppo/` for study 1, `q-settings/` for study 2, `250m/` for study 3). The folder's
`README.md` explains every column and shows how to load the files in pandas.
The data will be released with the trained models.

| File | One Row Per | Contents |
| --- | --- | --- |
| `runs.csv` | Run | Settings, selected checkpoint, confirmation results overall and per opponent, speed, time, memory, model library path |
| `recipes.csv` | Recipe | Means over the three seeds, per-seed margins, per-opponent results |
| `confirmation_cells.csv` | Run, map and opponent | The selected checkpoint's 16 confirmation games per cell |
| `routine_checks.csv` | Run, check, map and opponent | The learning curves, step 0 included |
| `recipe_check_curves.csv` | Recipe and check | Margin, kills and deaths per game at each check, mean over seeds |
| `training_updates.csv.gz` | Run and learner update | Losses, learning rates, gradient norms, entropy, exploration rate, reward parts |
| `training_games.csv.gz` | Run, update and training opponent group | Training games finished in that update: wins, draws, losses, points |
| `picks.csv` | Declared pick | Candidates' margins, kills and deaths; gain, seeds won, pass or fail (study 2 adds the engagement floor) |
| `RESULTS.md`, `FINDINGS.md` | – | The tables of this section and the findings, per study |
| `figures/` | – | Learning curves per recipe and run; QMIX and PQN-VDN diagnostics |

Study 2's files also hold its six base runs from study 1 (column
`comparison_base`), so each study 2 folder is complete on its own. Every saved
checkpoint of both studies is filed by method, study, recipe, seed and step in
the local model library, with a catalog of identities and scores.

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

So each PPO method's final recipe is `reference`. For the Q methods, study 2
changed the bases (see Study 2 recipes): QMIX B768 T32 with `q_lr = 0.0001`,
PQN-VDN B768 T128.

**Findings.**

1. **No recipe beats any panel opponent at 40M transitions.** Every margin is
   negative against all four opponents, and wins are 0-4% of games. The best
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
   against the three bots. Runs that lose to it by only 3-5 points barely
   fight it (0.5-1 kills, 4-6 deaths) and lose worse to the bots.
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

Each method's final recipe after studies 1 and 2, and the best run of that
recipe by confirmation margin. Most kills in the recipe is shown for
comparison.

| Method | Final Recipe | Mean Margin / Kills | Best Run (Seed) | Its Selected Step | Its Margin / Kills / Deaths / Wins | Most Kills In The Recipe |
| --- | --- | --- | --- | ---: | --- | --- |
| Recurrent MAPPO | `reference` | −8.8 / 8.3 | 20261002 | 40,108,032 | −7.9 / 8.4 / 14.9 / 3 | 20261001: 9.1 kills, −9.0 |
| Recurrent IPPO | `reference` | −10.4 / 8.2 | 20261001 | 40,108,032 | −8.5 / 11.1 / 15.4 / 19 | The best run |
| Feedforward MAPPO | `reference` | −10.6 / 5.7 | 20261002 | 12,042,240 | −9.9 / 6.6 / 12.6 / 0 | The best run |
| Feedforward IPPO | `reference` | −10.9 / 4.8 | 20261001 | 40,108,032 | −10.7 / 5.7 / 16.4 / 0 | The best run |
| QMIX | B768 T32, `q_lr = 0.0001`, `reward_scale = 10` | −4.3 / 12.4 | 12259871 ⭐ | 36,102,144 | −3.9 / 12.9 / 16.4 / 56 | 20261002: 13.0 kills, −4.4 |
| PQN-VDN | B768 T128, `reward_scale = 10` | −8.7 / 10.5 | 12259871 ⭐ | 40,108,032 | −5.1 / 13.1 / 17.1 / 72 | The best run |

Two runs outside the final recipes are stronger than the final recipe's best
run of their method, on margin and on kills or deaths. Feedforward MAPPO
`actor-lr-3e-4` on seed 20261002 scores −8.1 with 8.0 kills, against −9.9 and
6.6 for `reference`; its selected checkpoint is at 24.1M, against 12.0M. QMIX
`target-faster` on seed 12259871 scores −2.8 with 12.8 kills and 15.1 deaths,
against −3.9, 12.9 and 16.4 for `reward-x10`. These two runs, not the final
recipes' best runs, are the ones continued to 250M (study 3).

### Study 3: 250M Continuations

Not run yet. Its declaration is
[`continue_to_250m.yaml`](../../scripts/paper_experiments/hyperparameter_search/continue_to_250m.yaml);
results will be reported here in the same form as studies 1 and 2, with each
continuation joined to its parent run into one 0-250M record.

## Limits

- Three seeds per recipe find only large differences.
- Study 2 changes one setting at a time, so it cannot show how two settings
  interact.
- The PPO family uses gamma 0.999; the Q methods use 0.99 or 0.995. This
  difference is part of each method's recipe, not a fair test of gamma.
- The team reward matches each event's price, not the total shaped reward per
  game (see Reward).
- The panel's learned model shares its learner with recurrent MAPPO.
- Each run is represented by the checkpoint with the best routine point
  margin. That rule can select a phase in which the policy avoids fights
  (study 1 finding 2, study 2 finding 2); kills and deaths are reported beside
  every margin so this stays visible.
- Reward scale was searched only for QMIX and PQN-VDN. The PPO family
  normalizes its advantages and value targets, so a constant reward scale
  cancels out there (see Reward).
- A continuation starts from a run's final saved training state (40.1M), even
  when its selected checkpoint came earlier.
