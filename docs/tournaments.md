# Tournaments

Rank any set of methods with one Elo fit, or add your method to a released field.

Website: https://marl-battlegrounds.com/#round-robin

## Run A Round Robin

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

## Save A Field And Reuse It

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

## Read The Ratings

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

## Add A Challenger

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

**From A Checkout:** `python examples/canonical_tournament.py command=fixture directory=runs/fixture save=true` runs the route
on artificial records built by [`canonical_fixture.py`](../examples/canonical_fixture.py); [`canonical_admission.py`](../examples/canonical_admission.py) shows a maintainer admission.

## Select A Field

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
The command needs the exact run folder, as [`population_selection.py`](../examples/population_selection.py) shows:
`python -m marl_battlegrounds select-population selection-games/RUN_ID --output-dir population`.

## Next

- [Tournament Rules](tournament_rules.md): who may enter the official ladder.
- [Metrics](metrics.md): compare every entrant's play, each from its own side.
- [`canonical_tournament.py`](../examples/canonical_tournament.py): a runnable tour of released fields.
