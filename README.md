# MARL-BattleGrounds

**The JAX-native Benchmark for Heterogeneous and Competitive Multi-agent Reinforcement Learning: built from scratch by researchers, for researchers.**

[![arXiv](https://img.shields.io/badge/arXiv-2610.14883-b31b1b.svg)](https://arxiv.org/abs/2610.14883)
[![PyPI](https://img.shields.io/badge/pypi-v1.0.3-blue)](https://pypi.org/project/marl-battlegrounds/)
[![Python](https://img.shields.io/pypi/pyversions/marl-battlegrounds)](https://pypi.org/project/marl-battlegrounds/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arm-2-lab/MARL-BattleGrounds/blob/main/examples/quickstart.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MARL--BattleGrounds-ffcc4d)](https://huggingface.co/MARL-BattleGrounds)
[![Website](https://img.shields.io/badge/website-marl--battlegrounds.com-2b6cb0)](https://marl-battlegrounds.com)

[Installation](#installation) | [Quick Start](#quick-start) | [Baselines](#baselines) | [Guides](#guides) | [See Also](#see-also) | [Citation](#citation)

![Two teams fight a Team Deathmatch game in the analytical Replay Viewer](https://github.com/arm-2-lab/MARL-BattleGrounds/raw/main/docs/images/battle.gif)

**MARL-BattleGrounds (MARL-BGs)** is an easy-to-use, JAX-native benchmark
for heterogeneous and competitive team-vs-team multi-agent reinforcement
learning (MARL). MARL environments tend to be fast and simple, like SMAX,
or rich and costly, like Dota 2. MARL-BGs fills the missing middle, giving
you rich team play with fast, cheap experiments on one GPU.

Two teams of up to five agents fight on 52 hand-designed maps. Each agent is
a Mage, Warrior, Hunter, Rogue or Priest, classes inspired by World of
Warcraft®.

- **Bring any method.** Use plain JAX `reset` and `step`, or start from six
  tuned baselines whose trained models load by name from Hugging Face.
- **Test it fairly.** One call plays any mix of policies against each other.
  Eight endgame scenarios, each with a verified winning plan, test skills such
  as agent modeling and multi-step planning.
- **Enter the leaderboard.** Each season retrains every entry on the same GPU
  budget and adds it to a round-robin tournament on held-out test maps, which
  gives it an Elo. Can your method break into the Big Nine?
- **Watch and play.** Step through any game in the Replay Viewer, or play with
  or against your agents in the BattleClient.

Try it in your browser: the [walkthrough](https://colab.research.google.com/github/arm-2-lab/MARL-BattleGrounds/blob/main/examples/quickstart.ipynb) covers the API, the speed, training a team and the trained Season 0 teams, and the [analysis notebook](https://colab.research.google.com/github/arm-2-lab/MARL-BattleGrounds/blob/main/examples/evaluate/evaluation_analysis.ipynb) rebuilds the paper's analysis of all 7,200 Season 0 games.

## Installation

You do not need to install JAX first: each route installs the right JAX for
your machine. An NVIDIA GPU is optional but recommended: without one,
everything runs on the CPU, only slower.

To play the game on Linux, an Apple Silicon Mac or [Windows with WSL2](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/install.md#windows), one command installs everything and opens the BattleClient:

```bash
curl -L https://github.com/arm-2-lab/MARL-BattleGrounds/archive/refs/tags/v1.0.3.tar.gz | tar xz
cd MARL-BattleGrounds-1.0.3 && sh start.sh
```

In your own Python project (Python 3.12 to 3.14; 3.14 recommended):

```bash
pip install "marl-battlegrounds[all]"           # CPU
pip install "marl-battlegrounds[all,cuda13]"    # NVIDIA GPU, driver 580 or newer
```

With Docker or Podman, on any system (for an NVIDIA GPU, add `--gpus all` and use the `1.0.3-cuda12` tag):

```bash
docker run --rm -it -p 127.0.0.1:8765:8766 -p 127.0.0.1:8767:8768 -v "${PWD}:/work" ghcr.io/arm-2-lab/marl-battlegrounds:1.0.3
```

[The install guide](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/install.md) says which systems we tested each route on, and covers the GPU builds, Docker's limits, clusters and offline use.

## Quick Start

`make`, `reset` and `step` use JaxMARL's call order (`reset(key)`, `step(key, state,
actions)`), and compile under `jit`, `vmap` and `lax.scan`.

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

One command trains recurrent IPPO with the Season 0 learner settings (250 million game
steps) against itself, its past copies and the three scripted opponents, and keeps its
best checkpoint on the validation maps. [What Training Costs](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/performance.md#what-training-costs)
gives the time and memory on one GPU.

```bash
python -m marl_battlegrounds.experiments.train method=ippo pool=scripted \
  'validation_opponents=[tdm-alpha,tdm-beta,tdm-gamma]' output_dir=runs/my-ippo
```

## Evaluate

```python
import marl_battlegrounds as marl_bgs

results = marl_bgs.evaluate("random", "tdm-alpha", num_episodes=32, num_envs=32)
print(results.head_to_head())  # games, wins, draws, losses, point margin
```

Your method is always Team A, and paired games swap the spawn ends, so a map's
layout cannot favor either side. To analyze your own tournament, point the
[analysis notebook](https://colab.research.google.com/github/arm-2-lab/MARL-BattleGrounds/blob/main/examples/evaluate/evaluation_analysis.ipynb) at it with `MARL_BGS_RESULTS`.

## Watch and Play

```bash
python -m marl_battlegrounds replay path/to/game.marlbg-replay.json
```

```python
import marl_battlegrounds as marl_bgs

marl_bgs.battle_client(team_b="tdm-alpha")  # play in your browser against a scripted team
```

## Baselines

| Method | Name | Memory | Critic | Reference | Source |
|---|---|---|---|---|---|
| Recurrent MAPPO | `mappo` | Recurrent | Centralized | [Yu et al., 2022](https://arxiv.org/abs/2103.01955) | [`ppo.py`](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/src/marl_battlegrounds/baselines/ppo.py) |
| Feedforward MAPPO | `ff_mappo` | None | Centralized | [Yu et al., 2022](https://arxiv.org/abs/2103.01955) | [`ppo.py`](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/src/marl_battlegrounds/baselines/ppo.py) |
| Recurrent IPPO | `ippo` | Recurrent | One per agent | [de Witt et al., 2020](https://arxiv.org/abs/2011.09533) | [`ppo.py`](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/src/marl_battlegrounds/baselines/ppo.py) |
| Feedforward IPPO | `ff_ippo` | None | One per agent | [de Witt et al., 2020](https://arxiv.org/abs/2011.09533) | [`ppo.py`](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/src/marl_battlegrounds/baselines/ppo.py) |
| QMIX | `qmix` | Recurrent | QMIX mixer | [Rashid et al., 2018](https://arxiv.org/abs/1803.11485) | [`qmix.py`](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/src/marl_battlegrounds/baselines/qmix.py) |
| PQN-VDN | `pqn_vdn` | Recurrent | Sum of agent values | [Gallici et al., 2024](https://arxiv.org/abs/2407.04811); [Sunehag et al., 2017](https://arxiv.org/abs/1706.05296) | [`pqn.py`](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/src/marl_battlegrounds/baselines/pqn.py) |

Every save of the six Season 0 models, from 0 to 250 million steps, loads by name from
[Hugging Face](https://huggingface.co/MARL-BattleGrounds), for example
`marl_bgs.load_method("MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final")`
([Released Models](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/released_models.md)).

## Leaderboard

**Season 1:** submissions are open until 00:00 UK time on December 29, 2026, and its table is published at 00:00 UK time on January 1, 2027, on the [leaderboard](https://marl-battlegrounds.com/#leaderboard). Season 2 takes entries until 00:00 UK time on January 29, 2027, and its table is published on February 1, 2027; after that, there is one season a month.

Any method that runs as pure JAX may enter, not only MARL: learned, scripted, search-based, hybrid or built on a language model. We run your training script once on our RTX 5090: at most 255 million game steps and 8 hours. Every entry gets the same budget on the same machine, so the leaderboard rewards MARL methods that are efficient in both samples and compute. Your entry then plays the current Big Nine (at the start of Season 1, Season 0's six baselines and three scripted bots) on the test maps and is rated with draw-aware Elo. The table also shows what each entry cost to train and to play. Every entrant is released as a downloadable opponent.

```bash
python -m marl_battlegrounds entry new my-entry   # a starter entry
python -m marl_battlegrounds entry check          # checks it the way we will
python -m marl_battlegrounds entry submit         # opens the GitHub issue form
```

The [Tournament Rules](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/tournament_rules.md) say who may enter and set the limits. [Enter the Leaderboard](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/tournaments.md#enter-the-leaderboard) walks through the three commands.

## Guides

| Guide | What It Covers | Website |
|---|---|---|
| [Getting Started](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/getting_started.md) | Install, your first games, maps, rosters, what agents see | [start](https://marl-battlegrounds.com/#start) |
| [Game Rules](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/game_rules.md) | Classes, abilities, scoring, Red Zone, scenarios | [game](https://marl-battlegrounds.com/#game) |
| [Your Method](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/your_method.md) | Plug in your own model or learner, with JAX's jit, vmap and scan | [your-method](https://marl-battlegrounds.com/#your-method) |
| [Training](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/training.md) | The six learners, opponents, curriculum, rewards, checkpoints | [training](https://marl-battlegrounds.com/#training) |
| [Released Models](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/released_models.md) | Load, play, train against or start from the six trained Season 0 models | [models](https://marl-battlegrounds.com/#models) |
| [Hydra Configs](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/hydra.md) | Settings files, overrides, sweeps, studies and the experiment stages | [configs](https://marl-battlegrounds.com/#configs) |
| [Evaluation](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/evaluation.md) | Fair games, validation and test maps, scenarios, saved results | [evaluate](https://marl-battlegrounds.com/#evaluate) |
| [Metrics](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/metrics.md) | What is measured and how to read it | [metrics](https://marl-battlegrounds.com/#metrics) |
| [Tournaments](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/tournaments.md) | Round robins, Elo ratings and tiers | [round-robin](https://marl-battlegrounds.com/#round-robin) |
| [Tournament Rules](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/tournament_rules.md) | The leaderboard rulebook | [leaderboard-rules](https://marl-battlegrounds.com/#leaderboard-rules) |
| [Replays](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/replays.md) | Save games and watch them in the analytical Replay Viewer | [record](https://marl-battlegrounds.com/#record) |
| [BattleClient](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/battle_client.md) | Play against or beside your own agents in the browser | [battleclient](https://marl-battlegrounds.com/#battleclient) |
| [LLM Agents](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/llm.md) | Run a language model as a team | [llm](https://marl-battlegrounds.com/#llm) |
| [Performance](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/performance.md) | Speed and memory, and how to measure them on your machine | [performance](https://marl-battlegrounds.com/#performance) |
| [Method Sources](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/method_sources.md) | Where the built-in learners come from | [method-sources](https://marl-battlegrounds.com/#method-sources) |
| [API Reference](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/docs/api.md) | Every public name on one page | [api](https://marl-battlegrounds.com/#api) |

## Reproduce the Paper

The paper's commands, frozen settings and result tables are in
[`scripts/paper_experiments/`](https://github.com/arm-2-lab/MARL-BattleGrounds/tree/main/scripts/paper_experiments).
The Season 0 ladder and teamwork figures also rebuild in the
[analysis notebook](https://colab.research.google.com/github/arm-2-lab/MARL-BattleGrounds/blob/main/examples/evaluate/evaluation_analysis.ipynb), with nothing to install.

## See Also

JAX-native environments:

- [JaxMARL](https://github.com/bold-lab-ai/JaxMARL): Multi-Agent RL Environments and Algorithms in JAX.
- [JaxAHT](https://github.com/LARG/jax-aht): a benchmark for ad hoc teamwork (working with teammates never met in training) in JAX.
- [Gymnax](https://github.com/RobertTLange/gymnax): classic RL environments in JAX.
- [Jumanji](https://github.com/instadeepai/jumanji): combinatorial and planning environments in JAX.
- [Pgx](https://github.com/sotetsuk/pgx): board games in JAX.
- [Brax](https://github.com/google/brax): physics simulation for robot learning in JAX.
- [XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid): open-ended grid worlds for meta-RL in JAX.
- [Craftax](https://github.com/MichaelTMatthews/Craftax): an open-ended survival game in JAX.

JAX-native algorithms:

- [Mava](https://github.com/instadeepai/Mava): JAX implementations of popular MARL algorithms.
- [PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.

## Citation

If you use MARL-BattleGrounds in your work, please cite the paper:

```bibtex
@article{hawili2026marlbattlegrounds,
  title   = {{MARL-BattleGrounds}: A Novel {JAX}-native Benchmark for Heterogeneous and Competitive Multi-agent {RL}},
  author  = {Hawili, Ulixes Tariq and Ramamoorthy, Subramanian and Altmann, Yoann and Carlucho, Ignacio},
  journal = {arXiv preprint arXiv:2610.14883},
  month   = oct,
  year    = {2026}
}
```

## Contribute

Pull requests are welcome. Start with
[CONTRIBUTING.md](https://github.com/arm-2-lab/MARL-BattleGrounds/blob/main/CONTRIBUTING.md),
and open an [issue](https://github.com/arm-2-lab/MARL-BattleGrounds/issues)
for bugs, ideas, questions and leaderboard submissions.

## License

Licensed under the Apache License, Version 2.0.
