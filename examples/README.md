# Examples

Short, runnable lessons that use only the public API. Each one prints where
its output went. Run them from a checkout of this repository; most take a
`key=value` override and `--help` (they use Hydra settings). Tiny defaults
check the workflow on a CPU in seconds to minutes; they do not train skilled
agents.

```bash
JAX_PLATFORMS=cpu python examples/environment.py
```

## Start

| Example | What It Teaches | Guide |
|---|---|---|
| `quickstart.ipynb` | Play games, evaluate a trained team and play the Season 0 teams on Colab or Kaggle | [Install](../docs/install.md) |
| `environment.py` | Make a batch of games, take legal random actions, reset finished games | [Getting Started](../docs/getting_started.md) |
| `episode_tracking.py` | Curricula, automatic resets and optional recording in your own loop | [Getting Started](../docs/getting_started.md) |

## Your Own Method

| Example | What It Teaches | Guide |
|---|---|---|
| `systems.py` | Run JAX, per-agent and host methods through the System interface | [Your Method](../docs/your_method.md) |
| `research_methods.py` | Editable fixed methods you can load by `module:function` | [Your Method](../docs/your_method.md) |
| `own_learner.py` | Train your own centralized-critic learner, then validate and evaluate it | [Your Method](../docs/your_method.md) |
| `training_collection.py` | Collect learner-ready rows with curriculum and shaping settings | [Your Method](../docs/your_method.md) |
| `recorded_rollout.py` | Record training games and restore them with a checkpoint | [Your Method](../docs/your_method.md) |

## Train

| Example | What It Teaches | Guide |
|---|---|---|
| `train.py` | Launch any built-in learner with packaged settings | [Training](../docs/training.md) |
| `mappo_training.py` | Train, load and inspect a small PPO run (MAPPO or IPPO) | [Training](../docs/training.md) |
| `qmix_training.py` | Train, load and inspect a small QMIX run | [Training](../docs/training.md) |
| `pqn_training.py` | Train, load and inspect a small PQN-VDN run | [Training](../docs/training.md) |
| `curriculum.py` | Train on a map and roster curriculum | [Training](../docs/training.md) |
| `competitive_training.py` | Leagues, population-based training and response populations | [Training](../docs/training.md) |
| `partner_training.py` | Train part of a team beside frozen partners | [Training](../docs/training.md) |
| `training_distributions.py` | Sample training games through the collection API | [Training](../docs/training.md) |
| `training_continuation.py` | Extend a finished run into a new child run | [Training](../docs/training.md) |
| `train_against_released_models.py` | Train against three saves of a released Season 0 model | [Training](../docs/training.md) |
| `training_selection.py` | Choose a checkpoint by your own rule | [Training](../docs/training.md) |
| `population_selection.py` | Declare a population choice before games, then test it | [Training](../docs/training.md) |
| `training_study.py` | Run a small declared study of several runs | [Hydra Configs](../docs/hydra.md) |

## Evaluate

| Example | What It Teaches | Guide |
|---|---|---|
| `evaluation.py` | Evaluate fixed methods, repeat validation and run a small tournament | [Evaluation](../docs/evaluation.md) |
| `evaluation_results.py` | Paired and fixed-spawn games, authored starts and saved results | [Evaluation](../docs/evaluation.md) |
| `baseline_system.py` | Run an untrained recurrent MAPPO actor through your own loop | [Evaluation](../docs/evaluation.md) |
| `scenario_evaluation.py` | Evaluate any method at the eight packaged scenario starts | [Evaluation](../docs/evaluation.md) |
| `cross_play_and_zsc.py` | Fixed partners and opponents, including a local host partner | [Evaluation](../docs/evaluation.md) |
| `evaluate_recurrent_ippo.py` | Evaluate a saved recurrent IPPO model | [Hydra Configs](../docs/hydra.md) |
| `evaluate_recurrent_mappo.py` | Evaluate a saved recurrent MAPPO model | [Hydra Configs](../docs/hydra.md) |
| `evaluate_feedforward_ippo.py` | Evaluate a saved feedforward IPPO model | [Hydra Configs](../docs/hydra.md) |
| `evaluate_feedforward_mappo.py` | Evaluate a saved feedforward MAPPO model | [Hydra Configs](../docs/hydra.md) |
| `evaluate_pqn_vdn.py` | Evaluate a saved PQN-VDN model | [Hydra Configs](../docs/hydra.md) |
| `evaluate_qmix.py` | Evaluate a saved QMIX model | [Hydra Configs](../docs/hydra.md) |
| `evaluation_analysis.ipynb` | Explore saved results: outcomes, metrics, roles and replays | [Metrics](../docs/metrics.md) |

## Tournaments

| Example | What It Teaches | Guide |
|---|---|---|
| `canonical_tournament.py` | Compare saved tournament records | [Tournaments](../docs/tournaments.md) |
| `canonical_admission.py` | A practice admission with artificial records | [Tournaments](../docs/tournaments.md) |
| `canonical_fixture.py` | Build artificial tournament records without playing games | [Tournaments](../docs/tournaments.md) |

## Play, Watch And Language Models

| Example | What It Teaches | Guide |
|---|---|---|
| `battle_client.py` | Play your saved methods in the BattleClient | [BattleClient](../docs/battle_client.md) |
| `prepare_battle_client.py` | Write six untrained actor folders for the BattleClient examples | [BattleClient](../docs/battle_client.md) |
| `llm.py` | Evaluate a served language model with standard or custom prompts | [LLM Agents](../docs/llm.md) |
| `replays/` | Saved replays to open in the Replay Viewer | [Replays](../docs/replays.md) |
