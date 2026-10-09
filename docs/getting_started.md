# Getting Started

Install MARL-BattleGrounds, play your first batch of games and learn what each agent sees and does.

Website: https://marl-battlegrounds.com/#start

## Install

```bash
pip install "marl-battlegrounds[all]"
pip install "marl-battlegrounds[all,cuda13]"   # Linux with an NVIDIA GPU
```

`all` adds the six built-in learners, Hydra launchers, notebook tools and still
frames. Python 3.12, 3.13 and 3.14 work; 3.14 is recommended. For every other
route (uv, Docker, Windows, Colab, Kaggle, clusters), see [install.md](install.md).

| Your Machine | What Works |
| --- | --- |
| Linux with an NVIDIA GPU | Everything, on the GPU. CUDA 13 needs driver 580 or newer and a card with compute capability 7.5 or newer. |
| Linux without a GPU | Everything, on the CPU, only slower. |
| Mac with Apple Silicon | Everything, on the CPU. Every release is install-tested on macOS. |
| Intel Mac | Not supported (JAX does not support it). |
| Windows | Through WSL2, which runs Ubuntu inside Windows. |

**Watch Out:** canonical tournaments and adding games to a saved tournament need
Python 3.14. Extending a saved training run needs the exact Python it recorded.

## Play Your First Games

```python
import jax
import marl_battlegrounds as marl_bgs

reset_key, action_key, step_key, next_reset_key = jax.random.split(jax.random.key(42), 4)
env = marl_bgs.make("tdm", map_id=0, num_envs=128)
observation, state = env.reset(reset_key)
actions = env.sample_actions(action_key, state)
observation, state, reward, done, info = env.step(step_key, state, actions)
observation, state = env.reset_done(next_reset_key, state)
print(reward.rewards.shape, done.done.shape)  # (128, 10) (128,)
```

`make` builds 128 Team Deathmatch ("tdm") games once. `reset` starts them, `step`
moves every game one tick and `reset_done` restarts only finished games. You keep
the state and pass it back in; [environment.py](../examples/environment.py) runs it.

**Watch Out:** `step` never restarts a game. Keep what you need from the last
step before `reset_done`; afterwards the observation belongs to the new game.
For restarts inside `step`, wrap the environment in `marl_bgs.AutoReset`, as
[episode_tracking.py](../examples/episode_tracking.py) does. Keys are explicit:
reusing a key repeats the same random choices.

## Batches And Spawn Ends

```python
scalar = marl_bgs.make("tdm", map_id=0, balance_spawn_locations=False)  # one game, no batch axis
print(state.episode_id[:4])  # [1 2 3 4]
```

Arrays put the games first: rewards are `(games, 10 slots)`. `make` keeps the
map's spawn ends in the first half of the batch and swaps them in the second
half. Every reset takes a fresh block of episode IDs, so gaps are normal.

**Watch Out:** an odd `num_envs` raises `ValueError: num_envs must be a positive
even integer`; pass `balance_spawn_locations=False` for one game or an odd batch.
A new `num_envs` compiles a new program; new keys and weights do not.

## Maps And Splits

```python
maps = marl_bgs.list_tdm_maps()                       # 52 maps, in ID order
print(maps[47].map_id, maps[47].split, maps[47].curriculum)  # 47 test False
```

| Map IDs | Split | Use |
| --- | --- | --- |
| 0-11 | Training (curriculum) | Introduce every kind of obstacle, step by step |
| 12-41 | Training | The other training maps |
| 42-46 | Validation | Choose your checkpoint |
| 47-51 | Test | Final results and tournaments |

Every map is 20 units wide and 10 high. `marl_bgs.load_tdm_scenario(n)` loads
one of eight hand-made end-game starts ([game_rules.md](game_rules.md)).

**Watch Out:** never train or tune on test maps 47-51 or on the eight scenarios.

## Rosters And Settings

```python
small = marl_bgs.make("tdm", map_id=12, num_envs=128,
                      team_a_roster=["mage", "priest", "priest"],
                      team_b_roster=["warrior", "rogue"])
```

Each team has one to five agents, repeats allowed, filling its first slots in
order (Team A 0-4, Team B 5-9). Unused slots can only stay still and get 0 reward.

| Setting | Default | What It Does |
| --- | --- | --- |
| `num_envs` | `None` | Games per batch; `None` means one game with no batch axis |
| `team_a_roster`, `team_b_roster` | Mage, Warrior, Hunter, Rogue, Priest | One to five of `"mage"`, `"warrior"`, `"hunter"`, `"rogue"`, `"priest"` |
| `score_threshold` | `20` | Points needed to win |
| `max_steps` | `300` | Game length in ticks; no winner by then means a draw |
| `red_zone_depth` | `5.0` | Red Zone depth in map units; `0.0` turns it off |
| `metrics` | `"priority"` | `"none"`, `"priority"` or `"full"`; see [metrics.md](metrics.md) |

## What Agents See And Do

```python
print(env.action_space("agent_0"))  # move 0-8, select_target 0-10, use_ultimate 0-1
mask = env.get_action_mask(state)   # what is legal right now
inputs = env.policy_inputs(observation, state, team=0)  # Team A's five permitted views
```

Each tick every agent sends three numbers. Move: 0 stay, then 1-8 north, south,
east, west, northeast, northwest, southeast, southwest. Target: 0 none, 1-5 its
own team's rows (itself included), 6-10 the enemy's. Ultimate: 0 Basic, 1 Ultimate.
It sees its own sight plus what living teammates see now; no team ID or slot number.

| Field | Shape (One Agent) | What It Holds |
| --- | --- | --- |
| `self_features` | `(58,)` | Its position, class, health, cooldown, statuses and ability strengths |
| `ally_unit_features`, `enemy_unit_features` | `(5, 58)` each | One row per roster slot; a row it cannot see is all zeros |
| `ally_visibility_mask`, `enemy_visibility_mask` | `(5,)` each | Which rows it sees now |
| `map_obstacle_features`, `context_features` | `(32, 8)`, `(20,)` | Every wall and pillar; clock, map size, team sizes, scores and rules |

**Watch Out:** check the joint mask `select_target_use_ultimate_joint_mask`: a
target and an Ultimate choice can each be legal alone but illegal together. A
masked move becomes stay; a masked pair becomes no combat. `info` and
`env.training_state(state)` hold the whole game: never feed them to an acting
agent ([your_method.md](your_method.md) explains the limits).

## Coming From JaxMARL Or PettingZoo

| Idea | JaxMARL | PettingZoo | MARL-BGs |
| --- | --- | --- | --- |
| Reset | `obs, state = env.reset(key)` | `obs, infos = env.reset(seed=42)` | `obs, state = env.reset(key)` |
| Step | `env.step(key, state, actions)` | `env.step(actions)` | `env.step(key, state, actions)`, same order as JaxMARL |
| Agents | Dicts keyed by name | Dicts; done agents leave `env.agents` | Arrays with ten fixed slots; dead agents stay and respawn |
| Many games | `jax.vmap` | A vector wrapper | `num_envs` in `make` (`jax.vmap` also works) |
| Finished games | `step` resets them | You call `reset` | `reset_done`, or wrap in `AutoReset` |
| Legal actions | `get_avail_actions` in some environments | Often an `action_mask` in the observation | `env.get_action_mask(state)` |
| Adapters | — | — | None: MARL-BGs has no Gymnasium or PettingZoo adapter; use these calls |

## Next

- [game_rules.md](game_rules.md): how a match works, class by class.
- [your_method.md](your_method.md): plug in your own model and compile the loop.
- [training.md](training.md): train one of the six built-in learners.
