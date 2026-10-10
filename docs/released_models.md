# Released Models On Hugging Face

The six trained Season 0 models are on Hugging Face, with a file for every save from
the start of training to 250 million steps. Load any save by its name, play against
it, train against it, or start your own training from it.

Website: https://marl-battlegrounds.com/#models

## What You Need

| Install | Gives |
|---|---|
| `pip install marl-battlegrounds` | Load and evaluate released models, read the results dataset |
| `pip install "marl-battlegrounds[training]"` | Also train against them and start training from them |
| `pip install "marl-battlegrounds[training,hydra]"` | Also the evaluate scripts and the Hydra training commands |
| `pip install "marl-battlegrounds[hydra,viz]"` | Load and evaluate, plus draw training curves from their records (no training) |

On a GPU, add its build as the [install guide](install.md) shows, for example
`[training,cuda13]`. No Hugging Face account or token is needed.

## Names And Saves

The six models, in Season 0 rank order:
[`tdm-season-0-recurrent-ippo`](https://huggingface.co/MARL-BattleGrounds/tdm-season-0-recurrent-ippo),
[`tdm-season-0-pqn-vdn`](https://huggingface.co/MARL-BattleGrounds/tdm-season-0-pqn-vdn),
[`tdm-season-0-recurrent-mappo`](https://huggingface.co/MARL-BattleGrounds/tdm-season-0-recurrent-mappo),
[`tdm-season-0-qmix`](https://huggingface.co/MARL-BattleGrounds/tdm-season-0-qmix),
[`tdm-season-0-feedforward-ippo`](https://huggingface.co/MARL-BattleGrounds/tdm-season-0-feedforward-ippo) and
[`tdm-season-0-feedforward-mappo`](https://huggingface.co/MARL-BattleGrounds/tdm-season-0-feedforward-mappo),
each under `MARL-BattleGrounds/`.

| Name | Gives |
|---|---|
| `MARL-BattleGrounds/tdm-season-0-recurrent-ippo` | The save that played Season 0 |
| `MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final` | The last save |
| `MARL-BattleGrounds/tdm-season-0-recurrent-ippo@100M` | The nearest save at or before 100,000,000 steps |
| `MARL-BattleGrounds/tdm-season-0-recurrent-ippo@99_942_400` | Exactly the save at step 99,942,400 |

A name works anywhere MARL-BGs takes a model: `load_method`, `evaluate`, training
opponents and partners, `initial_actor`, tournaments and the BattleClient. Each
file downloads once and is checked against the repository's `steps.csv` before
use. A saved run records the file's exact address, `hf://<repo>@<commit>/<file>`,
so resuming it loads the same file.

To choose a save by its validation result, read `steps.csv`:

```python
import csv

import marl_battlegrounds as marl_bgs
from huggingface_hub import hf_hub_download

name = "MARL-BattleGrounds/tdm-season-0-recurrent-ippo"
path = hf_hub_download(name, "steps.csv")
with open(path, newline="") as file:
    rows = [row for row in csv.DictReader(file) if row["validation_games"]]
best = max(rows, key=lambda row: float(row["validation_point_margin"]))
system = marl_bgs.load_method(f"{name}@{best['step']}")
```

## Watch The Winner Learn

The Season 0 winner, Recurrent IPPO, at 10 million steps, at 100 million steps and
at its last save, each playing the same 32 games against the built-in TDM-Alpha
team on the five test maps:

```python
import marl_battlegrounds as marl_bgs

name = "MARL-BattleGrounds/tdm-season-0-recurrent-ippo"
for save in ("@10M", "@100M", "@final"):
    system = marl_bgs.load_method(name + save)
    games = marl_bgs.evaluate(system, "tdm-alpha", num_episodes=32, seed=0, num_envs=32)
    table = games.head_to_head()
    mine = list(table["system_id"]).index(games.metadata["system_ids"]["team_a"])
    kills = games.priority_metrics["team_a_kills"].mean()
    deaths = games.priority_metrics["team_a_deaths"].mean()
    print(
        f"{save:7} won {table['wins'][mine]}, drew {table['draws'][mine]}, "
        f"lost {table['losses'][mine]}; margin {table['mean_point_margin'][mine]:+.1f}, "
        f"kills {kills:.1f}, deaths {deaths:.1f} per game"
    )
```

One run on a CPU (AMD Ryzen 9 9950X3D, JAX 0.10.1) printed:

```text
@10M    won 0, drew 0, lost 32; margin -11.5, kills 7.9, deaths 19.7 per game
@100M   won 12, drew 0, lost 20; margin -0.9, kills 17.6, deaths 16.6 per game
@final  won 31, drew 0, lost 1; margin +8.0, kills 19.9, deaths 10.9 per game
```

Another CPU or a GPU can turn a game or two the other way.

## Train Against Released Models

In Python, put names in `opponents` beside self-play:

```python
from marl_battlegrounds import training

config = training.TrainConfig(
    method="mappo",
    opponents={
        "self": 0.5,
        "MARL-BattleGrounds/tdm-season-0-recurrent-ippo@10M": 0.25,
        "MARL-BattleGrounds/tdm-season-0-recurrent-ippo@100M": 0.25,
    },
)
training.train(config, output_dir="runs/ladder")
```

With Hydra, give each name a short alias, because a name holds `/` and `@`:

```bash
python -m marl_battlegrounds.experiments.train method=mappo \
  'opponents={self:0.5,ippo:0.5}' \
  '+runtime.opponents={ippo:"MARL-BattleGrounds/tdm-season-0-recurrent-ippo@100M"}' \
  output_dir=runs/ladder
```

[train_against_released_models.py](../examples/training/train_against_released_models.py)
is a runnable version. As a frozen teammate, a name goes in `partners` with
`learner_slots`, for example
`partners={"MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final": 1.0}`; see
[Training](training.md#opponents-and-partners).

## Start Training From One

```python
from marl_battlegrounds import training
from marl_battlegrounds.baselines.config import PPOConfig

config = training.TrainConfig(
    method="ippo",
    ppo=PPOConfig(input_scale=0.01),
    initial_actor="MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final",
)
```

The new run starts from that save's weights; the critic, mixer and optimizer
start fresh. Copy the method and the input scale from the model card: PQN-VDN
uses `input_scale=1.0` and the other five use `0.01`. A wrong value fails with
"Initial actor input_scale differs from the new learner".

## Evaluate, Tournaments And The BattleClient

- [Evaluation](evaluation.md): `marl_bgs.evaluate(name, "tdm-alpha", num_episodes=20)`,
  or, from a clone of the repository, a script such as
  `python examples/evaluate/evaluate_recurrent_ippo.py checkpoint=MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final evaluation.output_dir=runs/eval`
  (needs `[training,hydra]`).
- [Tournaments](tournaments.md): a name is one entrant.
- [BattleClient](battle_client.md): a name is one controller, for example
  `team_b="MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final"`.

## Downloads, Cache And Offline Use

A model file is 3 to 19 MB; a whole repository, with every save and the training
records, is about 2 to 3 GB. Files go into the Hugging Face cache (`HF_HOME`,
by default `~/.cache/huggingface`). On a machine without internet, for example a
cluster's compute nodes that share the cache, download first:

```bash
python -m marl_battlegrounds models download MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final
```

Then run your script with `HF_HUB_OFFLINE=1` set, for example
`HF_HUB_OFFLINE=1 python my_experiment.py`; it never uses the network.

## The Results Dataset

[`MARL-BattleGrounds/tdm-season-0-results`](https://huggingface.co/datasets/MARL-BattleGrounds/tdm-season-0-results)
holds every Season 0 game, the ratings, the original save folders and the
training opponents:

```python
import csv

from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "MARL-BattleGrounds/tdm-season-0-results",
    "season-0-ratings/ratings.csv",
    repo_type="dataset",
)
with open(path, newline="") as file:
    ratings = list(csv.DictReader(file))
print(ratings[0]["label"], ratings[0]["elo"])
```

## Training Curves

Every repository holds its training records. This draws Recurrent MAPPO's
validation curve from 0 to 250 million steps:

```bash
python -m marl_battlegrounds.experiments.training_curves \
  'lineages=[{name: Recurrent MAPPO, repo: MARL-BattleGrounds/tdm-season-0-recurrent-mappo}]' \
  output_dir=runs/curves hydra/job_logging=disabled
```

**Watch Out:**
- Hugging Face prints "You are sending unauthenticated requests to the HF Hub".
  It is harmless: no account or token is needed.
- A step with no save fails and lists the nearest saves.
- Released records name each training opponent by its released name or its
  address in the results dataset. Folders that were never published read
  "not published".
- To cite this work, cite the paper: see the README's
  [Citation](../README.md#citation).
