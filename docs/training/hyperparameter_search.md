# Six-Method Hyperparameter Search

This study gives each of the six built-in baselines a tested starting recipe
under one shared training protocol: recurrent MAPPO, feedforward MAPPO,
recurrent IPPO, feedforward IPPO, recurrent QMIX and recurrent PQN-VDN. It
measures complete recipes on a fixed budget. It does not claim to find a
global optimum. The design below was fixed before any study run started;
results are added as each study finishes, and every number comes from a saved
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

The search runs as three studies, one after the other, each on both cards.

| Study | Runs | What it decides |
| --- | ---: | --- |
| 1, `shapes_and_ppo` | 54 | QMIX and PQN-VDN batch size and rollout length; each PPO method's final recipe |
| 2, `q_settings` | 24-27 | One-at-a-time changes around each Q method's chosen shape |
| 3, `confirmation` | 18 | Each method's final recipe on three fresh seeds; these are the reported numbers |

**Seeds.** Every recipe in studies 1 and 2 runs on the same three seeds:
12259871, 20261001 and 20261002. Study 3 uses 20261011, 20261012 and 20261013,
which played no part in any choice. A seed fixes the starting weights and the
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

**Study 2 recipes** (3 runs each, around each Q method's chosen shape):

| Method | Recipe | Change | Basis |
| --- | --- | --- | --- |
| QMIX | `lr-1e-4` | `q_lr = 0.0001` | Only if `b768-t128` is chosen; study 1 already covers it otherwise |
| QMIX | `reuse-x2` | Twice the epochs and `update_period = 512`: 9.5 training pairs per new step, target spacing unchanged | The donor makes 1 optimizer step per 32 new steps; this base makes 1 per 512 |
| QMIX | `gamma-0.995` | `gamma = 0.995` | A win 150 ticks away keeps 22% of its value at 0.99 and 47% at 0.995 |
| QMIX | `reward-x10` | `reward_scale = 10.0` | Early Q-value errors are much larger than per-step rewards; JaxMARL scales by 10 |
| PQN-VDN | `layer-norm` | `normalization = "layer_norm"` | The PQN paper's strongest stability result |
| PQN-VDN | `reward-x10` | `reward_scale = 10.0` | The donor's SMAX setting |
| PQN-VDN | `lr-1e-4` | `q_lr = 0.0001` | Early loss blow-ups in an earlier run came at the peak learning rate |
| PQN-VDN | `lambda-0.95` | `td_lambda = 0.95` | Payoffs here arrive 3-25 ticks after the action |
| PQN-VDN | `gamma-0.995` | `gamma = 0.995` | As for QMIX |

Two further rows return only if study 1's measured cost leaves room in the
96 GPU-hour budget: QMIX `target-faster` (`update_period = 128`) and PQN-VDN
`epochs-2`.

**Pick rules, declared before launch.** A shape is picked by the highest mean
point margin of its three selected checkpoints. A searched change replaces the
base only if its mean beats the base by more than 2 points and it wins on at
least 2 of the 3 seeds. A method's final recipe is its base plus every change
that passes; a combination of changes is first measured in study 3. With
three seeds, differences smaller than about 3 points cannot be told apart from
noise.

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
validation: about 49 GPU-hours for study 1, 29-32 for study 2 and 16 for
study 3, about 96 in total. Measured costs, peak memory and disk are reported
with each study's results.

## Results

Added when each study finishes.

## Limits

- Three seeds per recipe find only large differences.
- Study 2 changes one setting at a time, so it cannot show how two settings
  interact.
- The PPO family uses gamma 0.999; the Q methods use 0.99 or 0.995. This
  difference is part of each method's recipe, not a fair test of gamma.
- The team reward matches each event's price, not the total shaped reward per
  game (see Reward).
- The panel's learned model shares its learner with recurrent MAPPO.
