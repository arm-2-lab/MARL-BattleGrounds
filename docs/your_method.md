# Your Method

Plug any model into MARL-BattleGrounds, run it in your own compiled loop and record it.

Website: https://marl-battlegrounds.com/#your-method

## Write A System

```python
import jax
import jax.numpy as jnp
import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines import actions as codec
from marl_battlegrounds.baselines.inputs import ACTOR_FEATURE_SIZE, encode_actor_inputs

def act(weights, memory, inputs, keys):
    features = encode_actor_inputs(inputs.actors)             # (B, 5, 5165): each agent's own view
    logits = features @ weights                                # (B, 5, 198)
    mask = codec.categorical_action_mask(inputs.action_mask)   # legal choices only
    actor_keys = jax.vmap(lambda key: jax.random.split(key, 5))(keys)
    index = codec.sample_actions(logits, mask, actor_keys)     # (B, 5)
    return marl_bgs.SystemOutput(codec.decode_actions(index), memory)
weights = jnp.zeros((ACTOR_FEATURE_SIZE, codec.NUM_ACTIONS), jnp.float32)  # untrained
mine = marl_bgs.System("Linear Team", act, variables=weights)
games = marl_bgs.evaluate(mine, "random", num_episodes=2, maps=[42], num_envs=2, phase="validation")
print(games.head_to_head())
```

A System decides for a whole team, once per tick, for a batch of `B` games. `inputs` holds
five separate permitted views (`actors`), legal choices (`action_mask`), slots in play
(`active_mask`, `(B, 5)`) and live games (`valid`, `(B,)`). Return `(actions, next_memory)` or a
`SystemOutput` whose `learning_outputs` carry values you already computed; nothing is recomputed.

**Watch Out:** keep output structure, shapes and dtypes fixed. Memory resets only when a new game
starts, not at death or respawn; memory without a leading game axis needs a `reset_memory` hook.

## Policies, Host Methods And Mixed Teams

```python
import numpy as np
import marl_battlegrounds as marl_bgs
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.types import ActorAction

def choose(variables, carry, actor_input, action_mask, key):  # one agent at a time
    return random_policy(actor_input.observation, action_mask, key), carry
def host_act(variables, memory, inputs, keys):  # NumPy inputs; any Python inside
    stay = np.zeros(inputs.active_mask.shape, np.int32)  # (B, 5): stay, no target, no Ultimate
    return ActorAction(stay, stay, stay), memory
shared = marl_bgs.shared_policy(marl_bgs.Policy("My Policy", choose))
python_team = marl_bgs.System("Stand Still", host_act, execution="host")
mixed = marl_bgs.team(shared, "tdm-beta", slots=[[0, 1, 2], [3, 4]])
```

A `Policy` decides for one agent; `shared_policy` runs it for every agent (shared weights,
separate memory) and `independent_policies([...])` uses one per slot. `execution="host"` runs
plain Python, such as a served language model. `team` mixes methods by slot; `load_method` loads one by name.

**Watch Out:** a host method gets one full NumPy batch: check `inputs.valid` and leave
other games alone. A `module:function` reference runs local code; load only code you trust.

## Your Own Compiled Loop

```python
from functools import partial
import jax
import marl_battlegrounds as marl_bgs

env = marl_bgs.make("tdm", map_id=0, num_envs=32)
foe = marl_bgs.shared_policy(marl_bgs.policy("tdm-alpha"))
key, reset_key, memory_key = jax.random.split(jax.random.key(0), 3)
obs, state = env.reset(reset_key)
memory = marl_bgs.init_systems(mine, foe, obs, state, memory_key)
def step(weights, carry, _):
    key, obs, state, memory = carry
    key, act_key, step_key, reset_key = jax.random.split(key, 4)
    actions, memory, learning = marl_bgs.apply_systems(
        mine, foe, memory, obs, state, act_key, variables_a=weights)
    result = env.step(step_key, state, actions)
    data = marl_bgs.system_step_data(state, actions, result)  # Team A rows, before any reset
    obs, state = env.reset_done(reset_key, result[1])
    return (key, obs, state, memory), (data, learning)
rollout = jax.jit(lambda weights, carry: jax.lax.scan(partial(step, weights), carry, None, length=16))
carry, (data, learning) = rollout(weights, (key, obs, state, memory))  # data.rewards: (16, 32, 5)
```

This reuses `mine` and `weights` from the first block. `apply_systems` picks both teams' actions,
`env.step` plays them and `env.reset_done` restarts finished games. Without Systems, use
`env.policy_inputs(obs, state, team=0)` for a team's permitted views and `env.join_actions(a, b)`.

**Watch Out:** keep what your learner needs before `reset_done`; after it, the observation belongs
to the new game. Pass changing weights as `variables_a`, never in a closure, or each update recompiles.

## Information Limits And A Training-Only Critic

```python
from marl_battlegrounds.baselines.inputs import encode_training_state
critic_input = encode_training_state(env.training_state(state), state.config)  # (32, 920); env, state from above
```

Each agent may act only on its own permitted view, which already includes what its teammates
share; returning a whole team's actions grants no extra rights. `info`, `env.training_state` and
training facts reveal hidden facts: they may feed a critic, a mixer or a reward, never an actor.

**Watch Out:** predicting hidden facts from allowed inputs is fine; reading them is not. See
[own_learner.py](../examples/your_method/own_learner.py), a complete small CTDE learner (centralized training, decentralized execution).

## Collection Helpers

```python
import jax
from marl_battlegrounds import training
from marl_battlegrounds.baselines import ppo
schedule = training.make_training_schedule(total_env_steps=4096, num_envs=32, curriculum=True)
variables = ppo.initialize_ppo(jax.random.key(0)).actor_params
actor = ppo.make_recurrent_mappo_system(variables)
collection, carry = training.init_training_collection(actor, variables, schedule=schedule, shaping=True)
carry, rollout = training.collect_training_rollout(collection, carry, length=128)
print(training.training_summary(collection, carry)["env_steps"])
```

These calls give exact experience budgets, the built-in curriculum, reward shaping and self-play
history, and leave the loss, critic and optimizer to you. `rollout.transitions` holds compact rows,
time first; `scan_training_rollout` fits a larger compiled loop ([training_collection.py](../examples/your_method/training_collection.py)).

**Watch Out:** `total_env_steps` must divide by an even `num_envs`, and a curriculum needs one
round per stage. The actor must be a JAX System whose memory has a leading game axis.

## Record Your Own Loop

```python
import marl_battlegrounds as marl_bgs
# step_fn(carry, None) returns (next_carry, (transition, info, policy_trace)).
with marl_bgs.RunWriter("runs/my-learner", phase="training", policies={"team_a": mine, "team_b": foe}) as writer:
    carry, transitions = marl_bgs.collect_rollout(step_fn, carry, num_steps=128, writer=writer)
    token = writer.checkpoint_recording()  # save it with your learner checkpoint
```

`collect_rollout` runs your scan step in compiled chunks and saves game records in a new run inside
`runs/my-learner` (`writer.run_dir`). To restore, reopen `RunWriter(resume_from=writer.run_dir,
recording_checkpoint=token, phase="training", policies=...)`; the records rewind to your restored
learner. See [recorded_rollout.py](../examples/your_method/recorded_rollout.py) and [systems.py](../examples/your_method/systems.py).

**Watch Out:** use one training writer per learner, pass every `info` (not only the last) and keep each token's files.

## Next

- [training.md](training.md): the built-in trainer and its six learners.
- [evaluation.md](evaluation.md): test your System on the test maps.
- [research_methods.py](../examples/your_method/research_methods.py): name your method as `module:function`.
