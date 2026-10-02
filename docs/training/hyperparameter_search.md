# Six-Method Hyperparameter Search

This study gives each of the six built-in baselines a tested starting recipe
under one shared training protocol: recurrent MAPPO, feedforward MAPPO,
recurrent IPPO, feedforward IPPO, recurrent QMIX and recurrent PQN-VDN. It
measures complete recipes on a fixed budget. Each method's best run of its
final recipe is then continued to 250M transitions and enters the
tournament. It does not claim to find a
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

The search runs as two studies, one after the other, each on both cards. The
search then ends with one 250M run per method.

| Study | Runs | What it decides |
| --- | ---: | --- |
| 1, `shapes_and_ppo` | 54 | QMIX and PQN-VDN batch size and rollout length; each PPO method's final recipe |
| 2, `q_settings` | 24-27 | One-at-a-time changes around each Q method's chosen shape |
| 250M runs | 6 | Each method's best run of its final recipe (best recipe and seed by confirmation margin), continued from its saved state to 250,675,200 transitions; these policies enter the tournament |

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
250M continuations need about 25 more. Measured costs, peak memory and disk are reported
with each study's results.

## Results

### Study 1: Shapes And PPO Recipes

Study 1 ran all 54 runs to 40,108,032 transitions with no failed run. It took
22 hours 43 minutes on two RTX 5090s, or 44.8 GPU-hours (planned: 49). Each
run is represented by its selected checkpoint: its best routine check by point
margin. That checkpoint was then scored in 256 confirmation games: 16 per map
and opponent, on maps 42, 43, 44 and 46, against the pooled 250M model,
`tdm-alpha`, `tdm-beta` and `tdm-gamma`. Margin is our points minus theirs per
game, with every map and opponent weighted equally. Kills and deaths are per
game. Seed 12259871 is the seed that trained the pooled 250M model.

| Method | Recipe | Mean Margin | Seed 12259871 | Seed 20261001 | Seed 20261002 | Kills | Deaths | vs 250M | vs Alpha | vs Beta | vs Gamma |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recurrent MAPPO | `critic-lr-1e-3` | −9.8 | −9.5 | −11.3 | −8.6 | 6.6 | 15.3 | −6.2 | −10.2 | −11.7 | −11.0 |
| Recurrent MAPPO | `reference` | −8.8 | −9.5 | −9.0 | −7.9 | 8.3 | 15.2 | −9.0 | −8.0 | −9.5 | −8.8 |
| Recurrent IPPO | `actor-lr-3e-4` | −10.9 | −10.4 | −11.0 | −11.2 | 5.6 | 16.8 | −6.9 | −11.4 | −12.5 | −12.7 |
| Recurrent IPPO | `critic-lr-1e-3` | −11.2 | −11.3 | −11.1 | −11.2 | 5.5 | 17.0 | −8.7 | −10.9 | −12.3 | −13.0 |
| Recurrent IPPO | `reference` | −10.4 | −11.6 | −8.5 | −11.1 | 8.2 | 17.0 | −13.7 | −8.2 | −9.9 | −9.9 |
| Feedforward MAPPO | `actor-lr-3e-4` | −9.4 | −9.6 | −10.6 | −8.1 | 6.6 | 15.0 | −5.0 | −10.4 | −11.3 | −11.0 |
| Feedforward MAPPO | `critic-lr-1e-3` | −11.3 | −11.1 | −10.9 | −11.7 | 5.0 | 17.0 | −7.3 | −11.8 | −12.8 | −13.1 |
| Feedforward MAPPO | `reference` | −10.6 | −10.4 | −11.4 | −9.9 | 5.7 | 14.1 | −4.9 | −11.9 | −12.9 | −12.5 |
| Feedforward IPPO | `actor-lr-3e-4` | −11.3 | −11.1 | −11.4 | −11.4 | 4.4 | 16.7 | −5.9 | −12.4 | −13.5 | −13.4 |
| Feedforward IPPO | `critic-lr-1e-3` | −11.4 | −11.8 | −10.9 | −11.4 | 5.4 | 16.9 | −8.0 | −11.2 | −13.4 | −12.8 |
| Feedforward IPPO | `reference` | −10.9 | −10.8 | −10.7 | −11.2 | 4.8 | 16.0 | −4.2 | −12.8 | −13.5 | −13.1 |
| QMIX | `b768-t128` | −11.6 | −10.8 | −11.6 | −12.4 | 5.6 | 16.2 | −10.8 | −9.9 | −12.5 | −13.2 |
| QMIX | `b768-t32` | −11.2 | −11.3 | −11.6 | −10.6 | 5.5 | 16.7 | −7.3 | −10.8 | −12.7 | −13.8 |
| QMIX | `b768-t32-lr-1e-4` | −10.2 | −11.5 | −9.6 | −9.5 | 6.2 | 16.4 | −6.3 | −10.7 | −11.7 | −12.1 |
| PQN-VDN | `b512-t128` | −14.2 | −15.2 | −15.0 | −12.4 | 1.0 | 11.7 | −11.7 | −12.2 | −14.0 | −18.8 |
| PQN-VDN | `b768-t128` | −15.3 | −14.5 | −15.7 | −15.7 | 1.8 | 15.4 | −9.5 | −15.9 | −17.7 | −18.2 |
| PQN-VDN | `b768-t32` | −9.7 | −10.1 | −9.1 | −9.9 | 0.3 | 5.4 | −19.3 | 0.0 | 0.0 | −19.5 |
| PQN-VDN | `b768-t64` | −15.0 | −13.5 | −15.5 | −15.9 | 1.5 | 10.8 | −17.0 | −11.7 | −13.0 | −18.1 |

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

So each PPO method's final recipe is `reference`, and QMIX continues from
B768, T32.

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

**Cost per run, measured.**

| Method | Transitions Per Second | Minutes | JAX Memory In Use | Host RAM |
| --- | ---: | ---: | ---: | ---: |
| Recurrent MAPPO | 19,485 | 41 | 3.1 GB | 8.4 GB |
| Recurrent IPPO | 17,834 | 44 | 3.0 GB | 7.8 GB |
| Feedforward MAPPO | 21,621 | 38 | 2.7 GB | 8.7 GB |
| Feedforward IPPO | 19,639 | 41 | 2.7 GB | 8.6 GB |
| QMIX | 13,973 | 54 | 13.5 GB | 14.1 GB |
| PQN-VDN | 10,300 | 70 | 11.1 GB | 9.9 GB |

Minutes include setup and validation. Memory is the process's peak with
preallocation off.

## Limits

- Three seeds per recipe find only large differences.
- Study 2 changes one setting at a time, so it cannot show how two settings
  interact.
- The PPO family uses gamma 0.999; the Q methods use 0.99 or 0.995. This
  difference is part of each method's recipe, not a fair test of gamma.
- The team reward matches each event's price, not the total shaped reward per
  game (see Reward).
- The panel's learned model shares its learner with recurrent MAPPO.
