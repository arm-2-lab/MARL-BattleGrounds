# Evaluate Exact Scenarios

Use the same evaluator for scripted policies, saved MAPPO/IPPO/QMIX/PQN actors,
LLMs, planners and custom Systems. A System must produce valid actions from its
permitted inputs. The scenario does not require a particular learning method.

The [runnable tutorial](../../examples/scenario_evaluation.py) uses
`evaluate_scenarios`, which uses the existing scenario loader and evaluator. It needs no DevClient or BattleClient.
Install the optional [Hydra launcher extra](../training/hydra.md) for commands.
Python calls remain available without Hydra.

## Start With Scenario 1

Scenario 1 asks a team to turn a nearly lost fight into a coordinated finish.
Hunter, Rogue and a Priest with 1 HP face Mage, Rogue and Priest. Team A leads
18–17, five ticks remain, and the score target is 20. Disabling the enemy healer,
using cover and arranging finishing attacks can matter more than attacking the
weakest enemy. Read `marl_bgs.load_tdm_scenario(1).notes` for the exact setup and
its verified authored solution. That solution does not prove what a learned
policy understands.

```bash
JAX_PLATFORMS=cpu python examples/scenario_evaluation.py \
  system=tdm-alpha scenario_ids=[1] seed=710 \
  metrics=full replay_episodes=[1] output_dir=/data/scenario-checks
```

This runs the whole remaining scenario. It prints the actual saved run folder
under `/data/scenario-checks/scenario-1/`. Replace `system=tdm-alpha` with a saved
actor directory or an installed `module:function` factory. The factory returns
a System or Policy. Keep the trained actor's required input schema compatible
with the environment; the loader checks its saved format.

The default opponent is the scenario's prescribed pressure. Setting
`opponent=tdm-gamma`, for example, makes a clearly different comparison. It keeps
the authored starting state but is not the original pressure condition.

## Keep Dead Agents In Their Slots

Every packaged scenario configures five slots per team, in Mage, Warrior,
Hunter, Rogue, Priest order. Several agents start dead. **Keep those slots.** Do
not shrink the model to the number of living agents, move survivors into new
slots, reset the scores or call an ordinary fresh-game reset. Dead agents can
respawn if play continues long enough.

The loader preserves positions, health, scores, effects, cooldowns, respawn
counts and the starting tick. The evaluator starts fresh method memory at that
first authored decision. A model receives no made-up history for the earlier
295 ticks. Death and respawn do not reset recurrent memory as a new episode.

## Use The Public Python API

This complete call works from an installed package:

```python
import marl_battlegrounds as marl_bgs

(result,) = marl_bgs.evaluate_scenarios(
    "tdm-alpha", scenario_ids=[1], seed=710, num_envs=2,
    metrics="full", replay_episodes=[1], output_dir="/data/scenario-checks",
)
print(result.run_dir)
```

For custom starts and schedules, use `load_tdm_scenario` and `evaluate_episodes`
with both `scenario.config` and `scenario.initial_state` in each `EpisodeSpec`.
For your own System, pass the live object as `system`. For a whole team of
independent Policies, use `marl_bgs.independent_policies` with all five slots.
For mixed methods, use `marl_bgs.team`. The [System examples](../../examples/systems.py)
and [LLM guide](../llm/README.md) show those existing tools. Returning actions for
a whole team does not give each actor other agents' private information.
Keep caller-owned LLM/provider sessions open for evaluation, then close them.

## Run Several Scenarios Or Trials

```bash
JAX_PLATFORMS=cpu python examples/scenario_evaluation.py \
  system=/data/models/selected-actor scenario_ids=[1,2,3,4,5,6,7,8] \
  num_episodes=10 seed=710 metrics=full replay_episodes=[1] \
  output_dir=/data/scenario-study
```

Each scenario gets a separate result folder, because prescribed opponents can
differ. Each trial repeats its exact starting state with a fresh method memory
and its own scheduled random keys. Batching changes how many trials run together;
it does not change their starting states or add games. The default metric mode
is priority; full metrics and replays are explicit choices. Omit `output_dir`
to keep results in memory.

## Resume And Read Results

Resume one printed result path, with the same scenario, methods and trial count:

```bash
JAX_PLATFORMS=cpu python examples/scenario_evaluation.py \
  system=/data/models/selected-actor scenario_ids=[1] num_episodes=10 \
  resume_from=/data/scenario-study/scenario-1/ACTUAL-RUN-ID
```

Omit `output_dir`. Omitted seed, metrics and replay choices inherit their saved
values. Explicit choices must match. Completed games are not repeated. An
unfinished game starts again from the authored state with fresh method memory.
A completed resume plays no games. The shared evaluator checks method and start
identities before changing saved results. An error reports saved locations;
earlier completed scenario folders remain available.

```python
import marl_battlegrounds as marl_bgs

saved = marl_bgs.load_results("/data/scenario-study/scenario-1/ACTUAL-RUN-ID")
print(saved.status)
print(saved.table("episodes"))
print(saved.replay_paths)
```

Use the [analysis notebook](analysis.md) for pandas tables, metric groups,
uncertainty and selected replay inspection. An early failure is incomplete data,
not a loss or draw. Compare methods under the same start, pressure, information
rules and trial schedule. Repeated trials from one trained actor do not replace
independent training seeds.

Public evaluation scenarios must stay out of training and tuning. A normal
saved result records behavior; it does not alone certify protected evaluation
or prove every skill named in the scenario. Keep training exposure records and
use matched behavioral ablations for the claims you intend to make.
