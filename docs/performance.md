# Performance

See how fast MARL-BattleGrounds runs, what each optional feature costs, and how to measure and tune it on your own machine.

Website: https://marl-battlegrounds.com/#performance

On one RTX 5090 with 1,024 parallel 5v5 games, the simulator alone runs 50,416 game steps per second (random legal actions, compilation excluded), and recurrent MAPPO trains at 29,878 game steps per second in pure self-play.

## How Fast

```python
import time
import jax
import marl_battlegrounds as marl_bgs

env = marl_bgs.make("tdm", map_id=47, num_envs=4)

def one_step(state, key):
    action_key, step_key, reset_key = jax.random.split(key, 3)
    _, state, reward, _, _ = env.step(step_key, state, env.sample_actions(action_key, state))
    return env.reset_done(reset_key, state)[1], reward.rewards

@jax.jit
def rollout(key):
    return jax.lax.scan(one_step, env.reset(key)[1], jax.random.split(key, 100))[1]

jax.block_until_ready(rollout(jax.random.key(0)))  # the first call compiles
start = time.perf_counter()
jax.block_until_ready(rollout(jax.random.key(1)))
print(f"{4 * 100 / (time.perf_counter() - start):,.0f} game steps per second")
```

A game step is one tick of one 5v5 game, or ten agent steps. The whole loop, restarts
included, is one compiled program. Timed this way, speed still rises at 1,024 games:

| Parallel Games | Game Steps Per Second | Agent Steps Per Second | First Call, Compile Included |
| --- | --- | --- | --- |
| 32 | 4,919 | 49,194 | 10.61 s |
| 64 | 10,122 | 101,221 | 10.31 s |
| 128 | 17,760 | 177,605 | 11.20 s |
| 512 | 39,921 | 399,205 | 14.86 s |
| 1,024 | 50,416 | 504,165 | 17.65 s |

One RTX 5090 alone, JAX 0.10.1, test map 47, 500-tick rollouts, median of five calls ([raw rows](../scripts/paper_experiments/results/speed_rtx5090_simulator.csv)).

## Measure It On Your Machine

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m marl_battlegrounds.experiments.speed output_dir=runs/speed
JAX_PLATFORMS=cpu python -m marl_battlegrounds.experiments.speed 'batch_sizes=[32]' train_blocks=0 output_dir=runs/speed-cpu
```

The `speed` stage times the simulator at 32, 64, 128, 512 and 1,024 games as above, then
trains recurrent MAPPO in pure self-play and times 40 updates. It writes `simulator.csv`,
`training.json` and `speed.md`. `train_blocks=0` skips training; `batch_sizes`, `steps`
and `repeats` change the workload. Pin a card with `CUDA_VISIBLE_DEVICES`.

**Watch Out:** without `XLA_PYTHON_CLIENT_PREALLOCATE=false`, JAX reserves most of the card
when it starts, so memory readings show that reservation, not real use.

## What Training Costs

```bash
python -m marl_battlegrounds.experiments.train method=mappo output_dir=runs/mappo
```

Recurrent MAPPO in pure self-play, 1,024 games, 32-tick rollouts: 29,878 game steps per
second (median of the 37 updates after warm-up), 5.0 GB peak JAX memory, 4.8 GB peak RAM.
The six baselines each trained from 40 million to 250,675,200 transitions against an
opponent pool, one RTX 5090 per run; hours include setup and 20 validation checks.

| Method | Parallel Games | Hours | Transitions Per Second | JAX Memory In Use | Host RAM |
| --- | --- | --- | --- | --- | --- |
| Feedforward MAPPO | 512 | 2.98 | 19,628 | 2.8 GB | 13.5 GB |
| Recurrent MAPPO | 512 | 3.22 | 18,168 | 3.1 GB | 14.5 GB |
| Feedforward IPPO | 512 | 3.31 | 17,664 | 2.8 GB | 14.1 GB |
| Recurrent IPPO | 512 | 3.55 | 16,458 | 3.0 GB | 14.2 GB |
| QMIX | 768 | 4.50 | 13,008 | 12.1 GB | 18.6 GB |
| PQN-VDN | 768 | 5.79 | 10,103 | 15.2 GB | 12.7 GB |

**Watch Out:** small runs are mostly setup: with that pool, a fresh run spends 25-31 s on setup
and 37-78 s compiling before its first block. A short run's time per transition misleads.

## What Choosing Actions Costs

```python
actions, memory, learning = marl_bgs.apply_systems(team_a, team_b, memory, observation, state, key)
```

Choosing time per tick for one team of five through `apply_systems` (one RTX 5090, self-play
on map 0, median of seven 100-tick rounds). A simulator step took 7.7-7.8 ms at 128 games.

| System | 128 Games | 1,024 Games | Share Of The 19.7-21.7 ms Simulator Step At 1,024 |
| --- | --- | --- | --- |
| Random | 0.011 ms | 0.012 ms | 0.06% |
| Recurrent MAPPO | 0.082 ms | 0.30 ms | 1.5% |
| Recurrent QMIX | 0.104 ms | 0.41 ms | 2.1% |
| Recurrent PQN-VDN | 0.149 ms | 0.73 ms | 3.7% |
| TDM-Alpha, TDM-Beta, TDM-Gamma | 1.85-1.87 ms | 17.5-19.1 ms | 86-88% |

**Watch Out:** a scripted bot costs nearly a whole simulator step, so scripted self-play runs
at about a third of the learned rate at 1,024 games. Pools with bots train slower.

## What Metrics, Recording And Exact Repeats Cost

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true python -m marl_battlegrounds tournament --config field.json
```

A 7,200-game round robin (nine entrants, test maps 47-51, 128 games at once, one RTX 5090)
took 491-508 s in normal GPU mode with priority metrics. Deterministic mode, which repeats
runs game for game, took 479-482 s: no measurable cost. Full metrics made it 589 s, and
recording 23 replays 738 s. Neither metrics nor recording changes which games are played.

| Item | Size |
| --- | --- |
| Priority or full metrics in each step's `info` | 80 or 55,960 bytes per game |
| A BattleClient replay of a 177-tick or 300-tick game | 14.8 MB or 23.7 MB, saved in about 8 s or 13 s |
| A recorded-loop checkpoint holding one unfinished 112 kB replay (32 games) | 249 kB, saved in 34.5 ms, restored in 433.2 ms |

**Watch Out:** `metrics="none"` and no replay choices skip that work entirely. Train with the
default priority level and no replays; collect full metrics and replays afterwards.

## Clients

```python
marl_bgs.battle_client(map_id=47, team_a="manual", team_b="tdm-beta")
```

Before the page opens, the BattleClient compiles the first turn: 4-7 s on CPU and 12.5 s on
one RTX 5090 for a recurrent learned policy against TDM-Beta; later turns took 0.02 s plus
your method's own time. The Replay Viewer prepares every agent's view first: 12.8 s for a
253-frame replay on CPU, then 38-41 ms per agent switch, with about 60 MiB more memory.

## The Knobs

```python
env = marl_bgs.make("tdm", map_id=0, num_envs=512, metrics="none")
```

| Knob | What It Does |
| --- | --- |
| `num_envs` | More games at once, more steps per second (GPU runs are timed at 32, 64, 128, 512 and 1,024). A new value compiles a new program; new weights, keys and maps of the same shape do not |
| `JAX_PLATFORMS=cuda,cpu` | Keeps host setup checks on the CPU while recording sampled training games |
| QMIX `buffer_size` | Replay memory grows with `num_envs` × `buffer_size`; the default recipe uses 768 games and 172 rows |

## Next

- [training.md](training.md): train with these settings.
- [hydra.md](hydra.md): every experiment stage, including `speed`.
