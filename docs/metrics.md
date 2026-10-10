# Metrics

Choose how much each game measures, then read the numbers without fooling yourself.

Website: https://marl-battlegrounds.com/#metrics

## Choose A Level

```python
import marl_battlegrounds as marl_bgs

result = marl_bgs.evaluate(
    "tdm-alpha", "tdm-beta", num_episodes=2, maps=[47], num_envs=2, max_steps=40, metrics="full"
)
print(result.table("full_metrics")["team_a_kills"])
```

Every run picks one level with `metrics=`; `make`, `evaluate` and the tournament
calls all take it. Each table row is one game. Outcomes, lengths and scores are
always kept, even with `"none"`. `full_metrics_episodes=[1, 50]` adds full detail
for chosen game IDs only, and `replay_episodes` records games the same way.

**Watch Out:** keep training lean: train and validate with `"priority"` and no
replays, then measure saved checkpoints in full in a separate evaluation.
Selections are game IDs, not a frequency; an ID the run never reaches records nothing.

| Level | What It Records | Typical Use |
| --- | --- | --- |
| `"none"` | Only each game's outcome, length and scores | Rankings and win rates |
| `"priority"` (default) | 16 scoreboard measures: length; win, draw and loss; return, score, kills and deaths per team; `score_difference` | Training, validation, tournaments |
| `"full"` | 11,192 measures in 28 topics, including the 16 | Studying the behavior of frozen checkpoints |

## What Full Metrics Measure

```python
from marl_battlegrounds.evaluation.metric_catalog import METRIC_COLUMNS_BY_NAME

column = METRIC_COLUMNS_BY_NAME["team_a_rescues"]
print(column.label, column.unit, column.description)
```

The catalog owns every name and meaning; [`metric_columns.csv`](metric_columns.csv)
lists each column with its unit, numerator, denominator and when it is blank.
Names read left to right: `agent_5_to_agent_3_...` is Agent 5 acting on Agent 3,
and `agent_0_and_agent_1_...` is a pair of teammates. Slots 0-4 are Team A, 5-9 Team B.

<!-- metric-family-summary:start -->
| Measurement family | Numerical columns | What the family describes |
| --- | ---: | --- |
| Episode results | 16 | Episode length, outcomes, returns, scores, kills, and deaths. |
| Ability use and action acceptance | 754 | How often abilities were used, who they targeted, and which actions the game rejected. |
| Deaths and respawning | 48 | Who died, how long agents were dead, and when they returned. |
| Kills and coordination | 746 | Who helped kill each enemy, who killed alone, and whether attackers chose the same target. |
| Red Zone kills and deaths | 44 | Kills and deaths inside each team's own Red Zone near its spawn, who helped with those kills, and each agent's share. Each such death counts once but gives the enemy team 2 points. |
| Damage | 538 | Who dealt damage, who took it, which ability dealt it, and each agent's share. |
| Healing and excess | 2,028 | Who healed whom, which ability was used, how much healing was useful or excess, and automatic health recovery. |
| Effects on controlled recipients | 3,592 | Damage, healing, and kill credit when the affected agent already had a named harmful status. |
| Status applications, duration, and Freedom | 1,532 | Who applied each status, who had it and for how long, and when Freedom protected movement. |
| Trap breaks | 218 | Who was trapped, who broke each Trap with damage, and how much Trap time remained. |
| Burst | 520 | Damage dealt during Burst, who took it, and who helped with kills while Burst was active. |
| Auras | 556 | Who gave and received aura coverage, extra damage from Mage auras, and damage blocked by Warrior auras. |
| Poison healing prevention | 12 | Priest healing prevented on affected agents and teams. |
| Lethal-damage rescues | 544 | Who could be saved from lethal damage, who survived, and which healers and abilities helped. |
| Formation | 44 | Distances between living teammates and how many times each pair was measured. |
| **Total** | **11,192** | **Unique numerical columns; shared viewer appearances are counted once.** |
<!-- metric-family-summary:end -->

## Topics In The Replay Viewer

```bash
python -m marl_battlegrounds replay PATH/TO/game.marlbg-replay.json
```

The Replay Viewer's **Metrics** tab sorts the full measures into 28 topics in six
groups, many with **Totals** and **By Recipient** tables. **Find A Measurement** takes
plain words or exact CSV names, and hovering a row shows its CSV column, numerator,
denominator and blank rule. Agent POV still shows the full record.

<!-- metric-navigation:start -->
| Topic | Primary columns in Totals | Primary columns By Recipient | Primary columns in single view |
| --- | ---: | ---: | ---: |
| Episode Results | — | — | 16 |
| Ability Activations | 24 | 640 | — |
| Accepted and Rejected Actions | — | — | 84 |
| Damage Done | 46 | 450 | — |
| Damage Received | — | — | 42 |
| Damage to Enemies With Harmful Effects | 84 | 1120 | — |
| Healing Done | 118 | 1050 | — |
| Healing Received | — | — | 148 |
| Excess Healing | 72 | 640 | — |
| Healing to Allies With Harmful Effects | 84 | 1030 | — |
| Priest Healing Saves | 74 | 470 | — |
| Kill Contributions | 142 | 600 | — |
| Kills of Enemies With Harmful Effects | 154 | 1120 | — |
| Deaths and Time Dead | — | — | 42 |
| Respawning | — | — | 6 |
| Red Zone | — | — | 44 |
| Team Coordination | — | — | 4 |
| Team Formation | — | — | 44 |
| Aura Coverage | 72 | 480 | — |
| Damage Added or Blocked by Auras | — | — | 4 |
| Status Applications | 108 | 1280 | — |
| Time With Status Effects | — | — | 108 |
| Freedom Against Slows | — | — | 36 |
| Burst (Mage Ultimate) | 50 | 470 | — |
| Charge (Warrior Ultimate) | 2 | 0 | — |
| Freezing Trap (Hunter Ultimate) | 28 | 190 | — |
| Crippling Poison (Rogue Ultimate) | 4 | 10 | — |
| Holy Word: Salvation (Priest Ultimate) | 2 | 0 | — |
| **Total** | | | **11,192 unique measurements** |
<!-- metric-navigation:end -->

## Reading Rules

```python
import pandas as pd

games = pd.DataFrame(result.table("full_metrics"))  # result.iter_table(...) reads a big table in parts
print(games["agent_4_effective_healing_done"].describe())  # blanks stay NaN
```

| Rule | What It Means |
| --- | --- |
| Blank is not zero | Blank means "does not apply" or "nothing to divide by". An active agent that could deal damage and dealt none has a real 0 |
| Allocation versus contribution | Agent 3 deals 100 damage, 40 to Agent 8; Team A deals 80 to Agent 8. Allocation: 40% of Agent 3's damage went to Agent 8. Contribution: Agent 3 dealt 50% of its team's damage to Agent 8 |
| Shared kills add past 100% | Damage, or useful Priest healing of an attacker, on the death tick earns kill credit. Two helpers on one kill both have 100% participation |
| Two ways to average | 1 of 1 and 1 of 9 average to 55.6%, but 2 of 10 pooled is 20%. Pick your question first |
| Points are not kills | A death in your own Red Zone gives the enemy 2 points but is still one kill and one death |
| Team return counts once | `team_a_return` adds the team's reward once per tick, not once per teammate |
| "Context dependent" | The number alone cannot say who played better: damage a Warrior takes may protect teammates |

**Watch Out:** pandas needs the `notebook` extra. With thousands of columns, some
differences look real by chance. Treat measures as diagnostics, not reward recipes.

## Compare Every System's Metrics

```python
saved = marl_bgs.load_results("runs/tournament/RUN_FOLDER")
field = saved.system_metrics("priority_metrics")  # one row per System
print(field["system_name"], field["elo"], field["score_difference_ci_low"])
mine = saved.system_metrics("full_metrics", by="opponent", systems=["My Method"])
```

Every System is shown as Team A: its own numbers sit in the `team_a_` and `agent_0`
to `agent_4` columns, even in games it played as Team B. `by` is `"system"`,
`"opponent"`, `"map"` or `"game"`. Fractions combine as total over total, and
`ratios={"kd": ("team_a_kills", "team_a_deaths")}` builds your own.

**Watch Out:** the 95% intervals (`_ci_low`, `_ci_high`) cover repeat games of fixed
methods, not training seeds; a System of your own gets "Conditional: Sampling not
verified". A row is blank when full metrics cover only some of its games.

## Measure A Saved Game

```python
values = marl_bgs.replay_metrics("runs/eval/RUN/replays/GAME.marlbg-replay.json", metrics="full")
print(values["team_a_kills"], values["team_a_deaths"])
```

`replay_metrics` measures any saved replay after the game: from `evaluate`, a
tournament or the BattleClient. It returns each value at the last tick, under the
table's column names (priority by default). It is the Replay Viewer's own
calculation, and it also takes a replay already read with `load_replay`.

**Watch Out:** a game that never finished is measured up to its last tick, so
end-of-game columns such as `team_a_win` stay NaN.

| Check | Games | Columns That Differed From Live `evaluate` |
| --- | --- | --- |
| CPU | Two 40-tick games of TDM-Alpha against TDM-Beta on map 47 | None of 11,192 |
| RTX 5090, XLA's deterministic mode | Four 188- to 213-tick games of recurrent IPPO against TDM-Alpha on maps 47 and 48 | None of 11,192 |
| RTX 5090, normal mode | The same four games | 0 to 24 per game, in the last digits of damage totals only |

## The Analysis Notebook

```bash
export MARL_BGS_RESULTS=/scratch/results/EXACT_RUN MARL_BGS_REPORT_DIR=/scratch/reports/eval
python -m jupyter nbconvert --to notebook --execute examples/evaluate/evaluation_analysis.ipynb \
  --output /scratch/reports/eval.executed.ipynb
```

[`evaluation_analysis.ipynb`](../examples/evaluate/evaluation_analysis.ipynb) turns one saved
run into charts, CSV tables and a testable claim. `MARL_BGS_RESULTS` is the exact
folder holding `run_details.json`; leave it unset for a tiny demo. Optional
`MARL_BGS_PHASE` and `MARL_BGS_PASS_ID` pick one saved pass.

**From A Checkout:** the notebook lives in `examples/` and needs the `viz` and `notebook` extras.

**Watch Out:** the report folder must be new or empty and outside the run folder.
Detailed plots read at most 200 games. The demo's tiny Random games are not evidence.

## Next

- [Evaluation](evaluation.md): play the games that fill these tables.
- [Replays](replays.md): record games and watch them in the Replay Viewer.
- [Tournaments](tournaments.md): compare a whole field, each System from its own side.
- [Performance](performance.md): what each metric level and each replay costs.
