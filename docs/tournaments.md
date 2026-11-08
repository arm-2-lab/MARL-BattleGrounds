# Tournaments

Rank any set of methods with one Elo fit, or add your method to a released field.

Website: https://marl-battlegrounds.com/#round-robin

## Run a Round Robin

```python
import marl_battlegrounds as marl_bgs

result = marl_bgs.run_tournament(
    ["random", "tdm-alpha"], maps=[47], games_per_opponent=4, max_steps=16, num_envs=4, seed=7
)
print(result.table("tournament_rankings"))
```

```bash
python -m marl_battlegrounds tournament --entrants random,tdm-alpha,tdm-beta \
  --games-per-opponent 100 --output-dir runs/tournament
```

Everyone plays everyone. Each pair plays `games_per_opponent` games in total,
spread evenly over the maps and both spawn ends. Entrants can be built-in names,
saved actor folders, model files, released names such as
`MARL-BattleGrounds/tdm-season-0-qmix@final`, `module:function` factories or (in Python)
live Systems.
The tiny example above is a wiring check: four short games prove no skill.

**Watch Out:** the budget must divide by twice the number of maps (ten for five
maps), or the call fails before any game. Every entrant needs its own name.

| Setting | Default | What It Does |
| --- | --- | --- |
| `games_per_opponent` | 100 | Total games per pair over all maps and both spawn ends |
| `maps` | Test maps 47-51 | Map IDs |
| `num_envs` | 128 | Games at once inside one matchup; matchups run one after another |
| `metrics` | `"priority"` | `"none"` still gives outcomes and ratings ([Metrics](metrics.md)) |
| `seed` | 0 | Root of the games' random numbers and of the rating resamples |
| `output_dir`, `resume_from` | None | Save a new run, or continue a saved one |

## Save a Field and Reuse It

```json
{"entrants": ["random", "tdm-alpha", "tdm-beta"], "maps": [47, 48, 49, 50, 51],
 "games_per_opponent": 100, "seed": 7}
```

Save it as `field.json`, then run `run_tournament(config="field.json")` or
`python -m marl_battlegrounds tournament --config field.json`. MARL-BGs builds the
identities and a balanced schedule. A field also takes `rosters`, `seeds`, `metrics`,
`score_threshold`, `max_steps`, `red_zone_depth`, `opponent_weights` and `output_dir`.

**Watch Out:** model folders and output paths in a field file are relative to
that file, not to where you run the command (a Python mapping uses the current
folder). Unknown fields fail. A call argument may fill an empty field, but if
both give a value they must agree.

## Read the Ratings

```python
ranks = result.table("tournament_rankings")
print(ranks["policy"], ranks["elo"], ranks["elo_ci_low"], ranks["elo_ci_high"])
print(ranks["rank"], ranks["elo_interval_status"])
```

One joint fit rates every entrant from all its wins, draws and losses. The field
average is always 1,200. Ranks follow the unrounded Elo, and exact ties share a
rank (1, 1, 3). `matchup_results` holds every pair, `map_results` every entrant on
every map, and `opponent_matrix()` turns one pair column into a grid.

**Watch Out:** an interval needs evidence. When games cannot support one (all
draws, too few spawn pairs, or methods that never sample), the status says why
and the interval stays blank instead of a fake zero width.

| Part | What MARL-BGs Does |
| --- | --- |
| Model | Win, draw and loss for every pair in one fit; draws get their own parameter, not half a win |
| Scale | Elo = 1,200 + 400 x strength / ln(10); a small penalty keeps ratings finite |
| Intervals | 5,000 bootstrap resamples that keep whole spawn pairs together; about 95% |
| Expected score | Win 1, draw 0.5, loss 0, with equal map weights |
| Robustness | `worst_20_percent_expected_score`: the mean over the worst 20% of opponent and map cells |
| Your own System | Intervals are kept but marked "Conditional: Sampling not verified" |

## Tiers

```python
print(ranks["policy"], ranks["elo"].round())  # tiers use this rounded Elo
```

Leaderboard tiers are 200-Elo bands. A tier uses the Elo rounded to a whole number,
and each lower edge belongs to its tier: 1,200 is Gold. Tiers are labels only: they
never change rank, rating or admission, and the library does not print them.

| Tier | Elo | Divisions |
| --- | --- | --- |
| Grandmaster | 2,000 and above | None |
| Master, Diamond, Platinum | 1,800-1,999; 1,600-1,799; 1,400-1,599 | Five of 40 Elo each: 5 (lowest) to 1 (highest) |
| Gold, Silver, Bronze | 1,200-1,399; 1,000-1,199; 800-999 | Five of 40 Elo each: Silver 5 is 1,000-1,039, Silver 1 is 1,160-1,199 |
| Iron | Below 800 | Counts down from the top: Iron 1 is 760-799 and Iron 5 is everything below 640 |

## Add a Challenger

```python
challenged = marl_bgs.run_tournament(config="field.json", challenger="tdm-gamma")
print(challenged.table("matchup_results"))
```

The challenger plays Team A against every field member, and the spawn ends swap
between paired games. One joint fit then rates all N + 1 entrants; old Elo is
never carried over. A short field config plays the field's own games too, so a
field of N has N x (N - 1) / 2 matchups and the challenger adds N more.

**Watch Out:** for a weighted field, list the challenger among the entrants with
its weight; a separate `challenger=` leaves its weight undeclared and fails.

## Canonical Tournaments

```bash
python -m marl_battlegrounds canonical --system runs/my-run/checkpoints/CHECKPOINT_ID \
  --metrics priority --output-dir runs/challenge
```

`run_canonical_tournament(system)` adds your method to the released official field.
It reuses the field's checked recorded games, so only your N matchups are played,
then fits everyone together. Play is offline; `models download` fetches files on request.

**Watch Out:** until a release is installed, the call stops with "No released
official tournament snapshot is installed; prepare an approved bundle or use
run_tournament with a custom config". A changed `games_per_opponent` makes a
research run (`protocol_compliant` is False). Local runs never admit or publish.

**From a Checkout:** `python examples/tournaments/canonical_tournament.py command=fixture directory=runs/fixture save=true` runs the route
on artificial records built by [`canonical_fixture.py`](../examples/tournaments/canonical_fixture.py); [`canonical_admission.py`](../examples/tournaments/canonical_admission.py) shows a maintainer admission.

## Select a Field

```python
field = {
    "entrants": ["random", "tdm-alpha", "tdm-beta"], "maps": [42, 43, 44, 45, 46],
    "games_per_opponent": 10, "max_steps": 2,
    "selection": {"size": 2, "failure_policy": "require-complete-field"},
}
games = marl_bgs.run_tournament(config=field, output_dir="selection-games")
decision = marl_bgs.select_initial_population(games, output_dir="population")
```

Declare the rule before any game, play on validation maps, then keep the top
`size` entrants. Ties break by expected score, then by `entrant_order`. The tool
writes `population.json` (with a content hash, `population_id`) and a separate
`refit.json`; a later field with `"selection": {"stage": "test", ...}` tests them on maps 47-51.

**Watch Out:** an unfinished field writes `status="incomplete"` and selects nobody.
The command needs the exact run folder, as [`population_selection.py`](../examples/training/population_selection.py) shows:
`python -m marl_battlegrounds select-population selection-games/RUN_ID --output-dir population`.

## Enter the Leaderboard

We train every entry once on our machine, under the same budget for everyone: 255
million game steps and 8 hours on one RTX 5090 (see the [Tournament Rules](tournament_rules.md)).
So the leaderboard rewards methods that are efficient in both samples and compute.

```bash
python -m marl_battlegrounds entry new my-method
cd my-method
# put your method into the starter (entry.yaml and my_entry/)
git add . && git commit -m "First entry"
uv run python -m marl_battlegrounds entry check
git remote add origin <your GitHub repository URL>
git push -u origin HEAD
uv run python -m marl_battlegrounds entry submit
```

Inside the entry folder, `uv run` uses the entry's own locked environment, the
one our run uses.

| Command | What It Does |
| --- | --- |
| `entry new NAME` | Writes a starter entry in the folder `NAME`: `pyproject.toml`, `entry.yaml`, a `load()` function, a README and, when uv is installed, `uv.lock`, and makes the folder a Git repository. The starter trains Season 0's Recurrent MAPPO recipe against itself, its past saves and the three scripted bots. The folder's name is your method's name unless `--name` gives another; it becomes the entry's name on the leaderboard and its Hugging Face repo, which the command prints. |
| `entry check` | Checks your current commit the way we will: a fresh clone, the install from `uv.lock` (which must install a released marl-battlegrounds from PyPI, any version), the models `entry.yaml` lists, a 15-minute training test, loading, export to a portable model file and the speed test. Saves the result in `.git/marl-bgs-entry-check/COMMIT.json`. |
| `entry submit` | Opens the GitHub issue form with the method name, repository and commit filled in, and prints the same link (`--no-browser` only prints). Tick the license box, and the private box if your repository is private, then submit. It refuses uncommitted changes, a commit on no remote branch, a commit without a passing check and the unchanged starter. |

Put your own method into the starter before you submit, because an unchanged
starter is a duplicate of Season 0's Recurrent MAPPO. You do not need to train
at home; `entry check` trains for 15 minutes, and our run does the real
training.

Loading, export and the speed test wait until the run has saved something to
enter (for the built-in trainer, its first choice). Loading and export must each
finish within 10 minutes. If the 15 minutes end first, the check skips
them instead of failing them, and says so, because our run trains for up to 8
hours. A run that
ends by itself without saving anything fails. The speed test is decided on our
RTX 5090; on other machines it says "not measured". Your load function must load
what your run last saved as its choice; for the built-in trainer,
[`training.current_choice`](training.md#validation-and-the-best-checkpoint)
names it. For a built-in run, we export that choice itself, not what your load
function returns, so a wrapper added in `load()` is not what plays.
A custom training loop must save only models in its output folder (logs go
elsewhere), because the check treats any file there as a saved model.

**Models Your Run Uses:** commit small outside weights to your repository
(ordinary Git files of up to 100 MB each; our machine does not fetch Git LFS).
List everything else in `entry.yaml`: every released model your run starts from,
trains with or plays against, for example `MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final`,
and any public Hugging Face model by its address and commit,
`hf://<owner>/<repo>@<40-character commit>/<path>`, where the path is a file, a
folder or `*` for the whole repository. Our run is offline, so we download them
all before the clock, without a token.

**Watch Out:** an NVIDIA GPU is recommended. Without a GPU, the check takes
about 17 minutes on our 16-core CPU, plus the first download of your packages,
and cannot measure the speed test. Passing the check does not test the full 8
hours, RAM or disk. A method name is taken once an entry under it has run this
season or waits in the queue; you may reuse your own only to resend a run that
saved nothing. So a new version needs a new name, for example `hydra-mappo-2`.
An entry made with 1.0.3 still runs 1.0.3's tools, because its `uv.lock` pins
1.0.3. Its `entry check` names the old rule numbers. Its `entry submit` opens the
new form with your repository and commit filled in, but also prints notes about
team members, an entry type and agreements, which no longer apply. To get the
new tools, change `==1.0.3` to `==1.0.4` on both marl-battlegrounds lines in
`pyproject.toml`, then run `uv lock` and commit both files.

### How We Run and Score

**The Season.** We publish a new leaderboard at 00:00 UK time (Europe/London) on
January 1, April 1, July 1 and October 1. Entries close 72 elapsed hours before
each. Season 1's entries close at 00:00 UK time on December 29, 2026, and its
table is published on January 1, 2027. Its games are played on the last 1.1.0
pre-release, due in late December. Season 2's entries close at 00:00 UK time on
March 29, 2027, and its table is published on April 1, 2027. Each table goes on
the website's [leaderboard](https://marl-battlegrounds.com/#leaderboard), and
its games and every entrant go on Hugging Face. Season 0 is our six baselines
and three scripted bots, rated on the test maps. The bots are reference
opponents built as part of the benchmark, so the training rules do not apply to
them.

**Our Machine.** One RTX 5090 (32 GB), an AMD Ryzen 9 9950X3D (16 cores, 32
threads), 48 GB of RAM and 50 GB of disk for what your run writes (temporary
files count; your installed environment and downloaded models do not). Each run
is sealed: it can write only to its own folder, and once the clock starts it has
no network. Before the clock starts, we clone your commit, then install from
`uv.lock` and download every model `entry.yaml` lists, without the GPU. The
install and the downloads must finish within 1 hour together. Nothing else is
downloaded.

**Our Run.**

- We first run `entry check` on your commit for 15 minutes. If it fails, you may
  fix your entry once without losing your place in the queue.
- We read your code before our run; that is where we judge duplicates and what
  your run brings in.
- Settings you choose by hand, even ones tuned at home, are fine: they are not
  trained or recorded.
- Our run is the same check on our RTX 5090, training for up to 8 hours. Nothing
  else runs on the card. At 8 hours we send SIGINT, wait 60 seconds, then kill
  it. Its speed test is the one that counts.
- Our games play the portable model file that `marl_bgs.export_model` writes
  from your entry: a JAX `Policy` that acts per agent, or a built-in actor
  trained with `parameter_sharing="all"`. A system that runs on the host, such
  as a language-model team, cannot export.
- A game step is one tick of one game, and every game the run plays counts:
  training, validation, self-play and games against opponents. 512 games side by
  side make 512 game steps per tick. A custom training loop must log its total
  game steps, validation included.
- JAX takes 75% of the card unless your train command sets
  `XLA_PYTHON_CLIENT_MEM_FRACTION`. Weights & Biases logs in offline mode.
- On a GPU, our run does not repeat your home run exactly. Setting
  `XLA_FLAGS=--xla_gpu_deterministic_ops=true` in your train command makes it
  repeat on an RTX 5090, but it trains 3.6 times slower in the same 8 hours.
- If your run crashes, we enter its last choice, or else its last save. You may
  then send a version with the bug fixed; one that changes only its settings is
  a duplicate. A different algorithm or model design is a different method. If
  our side fails, we rerun it at no cost to you.
- If your repository is private, tick the box on the form: we reply with a
  read-only key for that one entry, which you add as a deploy key with write
  access off. If you add it within 7 days of our reply, your arrival time is the
  issue's time; otherwise your entry arrives when you add the key.

**Arrival Order.** We test complete entries one at a time, in the order their
issues arrive, against the leaderboard as it stands after the previous entry.
The public table stays the same until the next quarter's table. Unfinished work
rolls forward to the next season and keeps its place in the queue.

**The Games.**

- Your entry plays as Team A, and the spawn ends swap between paired games: 200
  games against each of the nine systems, 40 on each test map 47 to 51.
- Both teams use the same five-class roster in the same order (Mage, Warrior,
  Hunter, Rogue and Priest). First to 20 points wins, and a game still running
  at tick 300 is a draw. The Red Zone depth is 5.0, and 128 games run at once.
- The games run in XLA's deterministic mode. We draw the seed at random after
  your training run and publish it with your results. We repeated these games
  exactly on our own card; a second RTX 5090 is not tested yet.
- Each agent sees what lies in its own sight plus what its living teammates see,
  and no system ever reads hidden game data. Memory starts fresh every game, and
  weights never change during the games.
- Full metrics are recorded from the first tick, and a resource or write failure
  never becomes a made-up win, draw or loss. Scenario results never feed Elo.

**The Ratings.**

- We fit one joint Elo for the nine systems plus your entry, as in [Read the
  Ratings](#read-the-ratings), and refit it before each entry. Elo is shown and
  compared as whole numbers.
- Your entry joins if its Elo is higher than the lowest system's; equal is not
  enough. The lowest system leaves. If several tie for last, the one with the
  lower expected score against the others leaves; if that ties too, the one that
  joined earliest leaves (among Season 0 systems, the later one in Season 0's
  order).
- Rank follows the whole-number Elo. [Tiers](#tiers) are labels beside it. Elo
  from different seasons cannot be compared, because each season has its own
  nine systems.
- Rank 1 of each season joins the Hall of Champions. A system that stays Rank 1
  gets one banner showing its reign.
- Each table shows every entry's type (Learned, Scripted, Hybrid or Language
  Model) and five costs from our run: Steps to Entry (the game steps of training
  behind the entered save), Run Steps (every game step the run played), Training
  Time, Play Speed and Play Memory. They never change rank, rating or admission.
- If a rule break is found later, the system the entry pushed out returns.

**What We Publish.** For every entry, whether it joins the leaderboard or not,
we publish its code, results, one saved model per 5 million training steps and
the entered save under Apache-2.0, with the season's table; we do not release
training state. Every entrant goes into one general pool you can download, and
each season's nine systems also get their own folder. A system that leaves the
leaderboard stays in the pool and in past tables.

**Why These Numbers.**

- **255 million game steps and 8 hours** sit just above our own slowest
  baseline. Our six baselines took about 3.6 to 7.0 hours to reach 250 million
  game steps on one RTX 5090, counting each first run and its continuation (see the
  [hyperparameter search record](../scripts/paper_experiments/hyperparameter_search.md)), so
  every Season 0 learned entrant meets the limits an entry faces.
- **One budget for everyone** means an entry wins by learning more from each
  game step and doing more with each hour on the card.
- **The play limits** exist because every entrant becomes a downloadable
  training opponent, so a slow or memory-hungry entrant would slow down
  everyone who trains against it. Every Season 0 entrant plays well above the
  speed floor and well under the memory limit.
- **One official run by us, on one fixed and published machine,** follows the
  MineRL 2020 and Procgen 2020 competitions, which also retrained every
  submission themselves under fixed limits on experience, time and hardware.
- **Teammate messages arrive one tick late,** as in classic learned
  communication research such as DIAL, where each agent acts on the messages
  sent on the previous step.

## Next

- [Tournament Rules](tournament_rules.md): the ten rules of the Official Leaderboard.
- [Metrics](metrics.md): compare every entrant's play, each from its own side.
- [`canonical_tournament.py`](../examples/tournaments/canonical_tournament.py): a runnable tour of released fields.
