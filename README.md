# MARL-BattleGrounds

**The JAX-native Benchmark for Heterogeneous and Competitive Multi-agent Reinforcement Learning.**

[![PyPI](https://img.shields.io/pypi/v/marl-battlegrounds)](https://pypi.org/project/marl-battlegrounds/)
[![Python](https://img.shields.io/pypi/pyversions/marl-battlegrounds)](https://pypi.org/project/marl-battlegrounds/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/LICENSE)
[![Website](https://img.shields.io/badge/website-marl--battlegrounds.com-2b6cb0)](https://marl-battlegrounds.com)

![Two teams of five fight on a MARL-BattleGrounds map](https://marl-battlegrounds.com/assets/art/key-art-preview.jpg)

Two teams of up to five agents fight Team Deathmatch on 52 maps. Each agent
is a Mage, Warrior, Hunter, Rogue or Priest, so a team wins by combining
movement, attacks, healing and support. The whole game is written in JAX, so
thousands of games run at once on one GPU. You bring the method; MARL-BGs
gives you the game, training helpers, evaluation, tournaments and replays.

## Install

```bash
pip install "marl-battlegrounds[all]"
```

On Linux with an NVIDIA GPU, add CUDA: `pip install "marl-battlegrounds[all,cuda13]"`.
Python 3.12, 3.13 and 3.14 are supported; 3.14 is recommended.

## Quick Start

```python
import jax
import marl_battlegrounds as marl_bgs

env = marl_bgs.make("tdm", map_id=0, num_envs=4)  # four games at once
key = jax.random.key(0)
observation, state = env.reset(key)
for _ in range(50):
    key, act_key, step_key, reset_key = jax.random.split(key, 4)
    actions = env.sample_actions(act_key, state)  # legal random actions
    observation, state, reward, done, info = env.step(step_key, state, actions)
    observation, state = env.reset_done(reset_key, state)  # restart finished games
```

On one RTX 5090 with 1,024 parallel 5v5 games, the simulator alone runs 50,416 game steps per second (random legal actions, compilation excluded), and recurrent MAPPO trains at 29,878 game steps per second in pure self-play.

## Train

One command trains recurrent IPPO with the tested Season 0 recipe (250 million
game steps, a few hours on one RTX 5090):

```bash
python -m marl_battlegrounds.experiments.train method=ippo output_dir=runs/my-ippo
```

## Evaluate

```python
import marl_battlegrounds as marl_bgs

results = marl_bgs.evaluate("random", "tdm-alpha", num_episodes=32, num_envs=32)
print(results.head_to_head())  # games, wins, draws, losses, point margin
```

Your method is always Team A, and paired games swap the spawn ends, so a map's
layout cannot favor either side.

## Watch And Play

```bash
python -m marl_battlegrounds replay path/to/game.marlbg-replay.json
```

```python
import marl_battlegrounds as marl_bgs

marl_bgs.battle_client(team_b="tdm-alpha")  # play in your browser against a scripted team
```

## The Six Built-In Learners

| Method | Name | Memory | Critic |
|---|---|---|---|
| Recurrent MAPPO | `mappo` | Recurrent | Centralized |
| Feedforward MAPPO | `ff_mappo` | None | Centralized |
| Recurrent IPPO | `ippo` | Recurrent | One per agent |
| Feedforward IPPO | `ff_ippo` | None | One per agent |
| QMIX | `qmix` | Recurrent | QMIX mixer |
| PQN-VDN | `pqn_vdn` | Recurrent | Sum of agent values |

Your own method can be anything that turns permitted observations into legal
actions: a network, a planner, a language model or a mix.

## Guides

| Guide | What It Covers | Website |
|---|---|---|
| [Getting Started](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/getting_started.md) | Install, your first games, maps, rosters, what agents see | [start](https://marl-battlegrounds.com/#start) |
| [Game Rules](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/game_rules.md) | Classes, abilities, scoring, Red Zone, scenarios | [game](https://marl-battlegrounds.com/#game) |
| [Your Method](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/your_method.md) | Plug in your own model or learner, with JAX's jit, vmap and scan | [your-method](https://marl-battlegrounds.com/#your-method) |
| [Training](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/training.md) | The six learners, opponents, curriculum, rewards, checkpoints | [training](https://marl-battlegrounds.com/#training) |
| [Hydra](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/hydra.md) | Settings files, overrides, sweeps, studies and the experiment stages | [configs](https://marl-battlegrounds.com/#configs) |
| [Evaluation](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/evaluation.md) | Fair games, validation and test maps, scenarios, saved results | [evaluate](https://marl-battlegrounds.com/#evaluate) |
| [Metrics](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/metrics.md) | What is measured and how to read it | [metrics](https://marl-battlegrounds.com/#metrics) |
| [Tournaments](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/tournaments.md) | Round robins, Elo ratings and tiers | [round-robin](https://marl-battlegrounds.com/#round-robin) |
| [Tournament Rules](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/tournament_rules.md) | The leaderboard rulebook | [rules](https://marl-battlegrounds.com/#rules) |
| [Replays](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/replays.md) | Save games and watch them in the Replay Viewer | [replay-format](https://marl-battlegrounds.com/#replay-format) |
| [BattleClient](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/battle_client.md) | Play against or beside your own agents in the browser | [battleclient](https://marl-battlegrounds.com/#battleclient) |
| [Language Models](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/llm.md) | Run a language model as a team | [llm](https://marl-battlegrounds.com/#llm) |
| [Performance](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/performance.md) | Speed and memory, and how to measure them on your machine | [performance](https://marl-battlegrounds.com/#performance) |
| [Method Sources](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/method_sources.md) | Where the built-in learners come from | [method-sources](https://marl-battlegrounds.com/#method-sources) |
| [API Reference](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/docs/api.md) | Every public name on one page | [api](https://marl-battlegrounds.com/#api) |

## Reproduce The Paper

The paper's commands, frozen settings and result tables are in
[`scripts/paper_experiments/`](https://github.com/oceansystemslab/MARL-BattleGrounds/tree/main/scripts/paper_experiments).

## Cite

Use the "Cite this repository" button on GitHub, or:

```bibtex
@software{marl_battlegrounds,
  author = {Hawili, Ulixes Tariq},
  title = {{MARL-BattleGrounds}: The JAX-native Benchmark for Heterogeneous and Competitive Multi-agent Reinforcement Learning},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/oceansystemslab/MARL-BattleGrounds}
}
```

## Contribute

Pull requests are welcome. Start with
[CONTRIBUTING.md](https://github.com/oceansystemslab/MARL-BattleGrounds/blob/main/CONTRIBUTING.md),
and open an [issue](https://github.com/oceansystemslab/MARL-BattleGrounds/issues)
for bugs, ideas, questions and leaderboard submissions.

## Author

MARL-BattleGrounds is developed by Ulixes Tariq Hawili as part of the SPADS CDT,
University of Edinburgh School of Engineering, and the Ocean Systems Lab at HWU.

## License

Licensed under the Apache License, Version 2.0.
