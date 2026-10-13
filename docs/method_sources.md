# Method Sources

Where the six built-in learners' code comes from, its licenses, and what we changed on purpose.

Website: https://marl-battlegrounds.com/#method-sources

## Donor Projects

```python
import inspect
from marl_battlegrounds.baselines import pqn, ppo, qmix  # needs the training extra
for module in (ppo, qmix, pqn):
    print(inspect.getsourcefile(module))  # each file starts with its donor's license notice
```

| Donor | Exact Commit | License | Our Files | Used For |
| --- | --- | --- | --- | --- |
| Mava | `9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93` | Apache 2.0 | `baselines/ppo.py`, `baselines/qmix.py`, `baselines/config.py` | The four PPO learners and recurrent QMIX |
| JaxMARL | `976aeb152cb184a5095021968bba94da96eb6394` | Apache 2.0 | `baselines/pqn.py`, `baselines/config.py` | Recurrent PQN-VDN |
| on-policy (official MAPPO) | `de66d7a4b23fac2513f56f96f73b3f5cb96695ac` | MIT | `baselines/ppo.py`, `baselines/config.py` | ValueNorm for the PPO critics |

The donors supply the learning math only. MARL-BattleGrounds keeps its own
environment, actor inputs, action rules, evaluation, recording, checkpoints,
validation and studies. MARL-BattleGrounds itself is Apache 2.0.

**Watch Out:** keep each file's license notice when you copy or change it.

## What We Reused

```bash
ls tests/fixtures/baseline_donor/source/mava tests/fixtures/baseline_donor/source/jaxmarl tests/fixtures/value_norm_donor
```

From a checkout, these folders hold the unchanged donor files we compare against.

| Donor File | What It Gave Us |
| --- | --- |
| Mava `networks/torsos.py`, `networks/base.py`, `networks/heads.py` | Dense torso, recurrent cell, actor, critic, QMIX Q-network and mixer, masked action head. |
| Mava `utils/multistep.py` | `calculate_gae`, with its rule for game endings. |
| Mava `systems/ppo/anakin/{rec_mappo,rec_ippo,ff_mappo,ff_ippo}.py`, `configs/` | PPO losses, group averaging, minibatch and epoch order, optimizer setup, starting settings. |
| Mava `systems/q_learning/anakin/rec_qmix.py`, `configs/` | Epsilon-greedy choice, sample preparation, Double-Q loss and update, starting settings. |
| JaxMARL `baselines/QLearning/pqn_vdn_rnn.py` | Q-network, recurrent cell, RAdam with its schedule, exploration, lambda returns, epochs. |
| JaxMARL `baselines/QLearning/config/alg/pqn_vdn_rnn_smax.yaml` | Starting settings for PQN-VDN. |
| on-policy `onpolicy/utils/valuenorm.py`, `onpolicy/algorithms/r_mappo/r_mappo.py` | Running normalization of critic targets. |

Nothing from the donors' environment wrappers, training and evaluation loops, loggers,
checkpoint loaders or distributed harness is copied. Flashbax (QMIX's replay buffer),
Hydra and OmegaConf are installed dependencies; no code of theirs is copied either.

## Deliberate Changes

```python
from marl_battlegrounds.baselines.config import PPOConfig, QMIXConfig
print(PPOConfig().spawn_frame, PPOConfig().input_scale, PPOConfig().value_normalization)  # left 1.0 True
print(QMIXConfig().reward_scale, QMIXConfig().spawn_frame)                                 # 1.0 left
```

| Change | Learners | Why |
| --- | --- | --- |
| Our permitted 5,165-value actor input; no one-hot agent index | All | Each actor's input already holds its own class and self features. |
| Spawn frame `"left"` by default; moves mirrored back | All | Every game looks like a start from the left side. `"world"` keeps raw coordinates. |
| `input_scale` option, default 1.0 | All | 1.0 keeps the donor math; a smaller value such as 0.01 shrinks large raw inputs. |
| 198-way action head with the joint legality mask | All | One output per native move, target and Ultimate choice. |
| Illegal actions masked with negative infinity | QMIX, PQN-VDN | The greedy choice is always legal; ties go to the lowest legal index. |
| Memory resets only when a new game starts | Recurrent | Death and respawn are temporary here, so memory carries through them. |
| A real ending stops bootstrapping; a rollout cutoff does not | All | A time-limit draw is a real ending; a block edge is not. |
| `parameter_sharing` `"class"` and `"none"`; learner slots beside frozen partners | All | Per-class or per-slot actors and partial teams. |
| Training loop, checkpoints (Orbax), validation and recording are ours | All | One owner for each job across all six learners. |
| GPU autotuning off for built-in training compilations | All | Fixed kernel choices remove one cause of run-to-run differences. |
| Inactive slots and padding left out; dead agents train only the critic; empty groups add nothing | PPO | A dead agent's forced "stay" carries no policy signal; an optimizer with no samples is skipped. |
| ValueNorm added, on by default | PPO | The official MAPPO critic-target normalization. |
| Small JAX categorical helper | PPO | Avoids adding TensorFlow Probability to the runtime. |
| Team reward is the average over active learner-owned slots | QMIX, PQN-VDN | Works with small rosters and frozen partners (Mava: all agents; JaxMARL: first agent). |
| `reward_scale` option, default 1.0 | QMIX, PQN-VDN | New for QMIX; JaxMARL's `REW_SCALE` is 10, a conversion for its own game. |
| No gradient clipping | QMIX | Mava's settings file names `max_grad_norm: 10`, but its optimizer never clips. |
| Exploration clock counts real transitions and stops at the end of decay | QMIX | Never overflows; the final rate is exactly `eps_min`. |
| Compact Team A replay rows, explicit readiness check, warmup blocks | QMIX | Smaller replay; no silent zero samples before the buffer is ready. |
| First random rounds pick legal actions and count in the budget | PQN-VDN | The donor drew from all actions, legal or not. |
| Padding left out of the loss and BatchNorm statistics | PQN-VDN | Rosters can be smaller than five; budgets can end mid-block. |
| A bad learning block is rejected and the last state kept | PQN-VDN | The donor has no rollback. |
| 32 games by default, one learner per run | PQN-VDN | The donor used 128 games and several learners per process. |

## How We Check the Match

```bash
JAX_PLATFORMS=cpu python -m pytest tests/baselines/test_baseline_ppo_reference.py tests/baselines/test_baseline_qmix_reference.py tests/baselines/test_baseline_pqn_reference.py tests/baselines/test_baseline_value_norm.py
```

From a checkout, the PPO, QMIX and PQN-VDN tests run the original donor functions on
fixed synthetic inputs and compare every array with ours (tolerance 0.000002, absolute
and relative); the ValueNorm test uses an independent NumPy version. The unchanged donor
files, their licenses and SHA-256 manifests are kept in `tests/fixtures/`.

**Watch Out:** a match on fixed inputs shows the math is the same. It says nothing
about how well a learner plays.

## Next

- [training.md](training.md): train the six learners.
- [api.md](api.md): every public function.
