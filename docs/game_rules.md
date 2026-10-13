# Game Rules

Learn how a Team Deathmatch match works: classes, ticks, damage, scoring, respawns, rewards and scenarios.

Website: https://marl-battlegrounds.com/#game

## A Match at a Glance

```python
import jax
import marl_battlegrounds as marl_bgs

env = marl_bgs.make("tdm", map_id=12, num_envs=2, score_threshold=30, max_steps=400,
                    red_zone_depth=0.0, team_a_roster=["warrior", "mage", "priest"])
observation, state = env.reset(jax.random.key(0))
print(env.agent_info(state)["class_ids"][0])  # [2 1 5 0 0 1 2 3 4 5]
```

Two teams of one to five champions fight on a 20 by 10 arena, each starting on five
spawn pads at its own edge. Everyone acts at the same moment, once per tick. The first
team to 20 points wins; after 300 ticks with nobody at 20 the game is a draw. The
defaults are the official rules; `make` lets you change them for your own study.

**Watch Out:** when you report results, name every rule you changed.

## The Five Classes

```python
from marl_battlegrounds.core import combat
print(combat.MAX_HEALTH_BY_CLASS)  # [0. 80. 200. 100. 100. 100.]: padding, then IDs 1-5
```

| Class (ID) | Health | Basic, Every Tick | Ultimate, 30-Tick Cooldown | Aura or Passive |
| --- | --- | --- | --- | --- |
| Mage (1) | 80 | Arcane Blast: 13 damage, reach 3.0 | Burst: its own damage ×1.5 for 5 ticks, no target | Sorcerer's Empowerment: allies within 2.0 deal ×1.15 |
| Warrior (2) | 200 | Shield Bash: 8 damage, reach 1.5 | Charge: rushes to an enemy within 5.75; 20 damage, stun 1 tick, half speed for 5 | Guardian's Barrier: allies within 2.0 take ×0.85 |
| Hunter (3) | 100 | Serrated Arrows: 6 damage, reach 3.5, speed ×0.85 for 1 tick | Freezing Trap: enemy within 3.0; 10 damage, stun up to 4 ticks that damage on a later tick breaks | Far Sight: sees 6.5 units (others 6.0) |
| Rogue (4) | 100 | Backstab: 12 damage, reach 1.5 | Crippling Poison: enemy within 1.5; 36 damage, stun 1 tick, half speed for 5, half healing for 4 | Phantom's Quickness: speed 1.3 (others 1.0); recovers health after 3 ticks out of combat (others 5) |
| Priest (5) | 100 | Lesser Heal: 8 health to an ally or itself, reach 3.0 | Holy Word: Salvation: 200 health (a full heal), reach 5.75 | Blessing of Freedom: for 1 tick after Lesser Heal, slows cannot push the ally below 85% speed |

Each tick a champion uses its Basic or its Ultimate, never both, on one champion. The
Ultimate timer keeps running while a champion is fallen. Bodies are circles of radius
0.5, and reach is measured center to center. Auras include the caster, pass through
walls and stack to at most ×1.3225 (Mage) or ×0.7225 (Warrior).

## One Tick in Order

```python
mask = env.get_action_mask(state).select_target_use_ultimate_joint_mask  # (2, 10, 11, 2)
```

Everyone chooses at once, from the same picture. Every accepted ability then lands
together, from where everyone stood when the tick began. Next come out-of-combat
recovery and cooldowns, then Charges, then everyone's moves. Last, champions at 0
health fall, statuses update, a due respawn wave returns earlier falls, and points are added.

**Watch Out:** a masked move becomes stay, and a masked target and Ultimate pair becomes
no combat. A champion that falls this tick still delivers the action it chose.

## Targets, Damage and Healing

```python
print(combat.BASIC_DAMAGE_BY_CLASS, combat.ULTIMATE_DAMAGE_BY_CLASS)
```

To act, a champion must be alive, not stunned, not shielded, and see the target itself:
walls and pillars block sight, bodies do not. Damage reaches only enemies, heals only allies
(the healer included). Every hit lands: base × Burst × Mage auras × Warrior auras. Out of
combat, a champion recovers 4% of its maximum health per tick; Crippling Poison halves that.

**Watch Out:** damage and healing in one tick add up first, so a heal can cancel a killing
blow. A new status counts from the target's next decision: a 1-tick stun costs one decision.

## Respawns and the Red Zone

```python
print(env.training_state(state).team_respawn_wave_countdowns)  # ticks to each team's next wave
```

A fallen champion returns at its team's next wave, every 5 ticks, on its own pad at full
health with a 3-tick spawn shield: it cannot attack, heal, be seen by enemies or be targeted
by allies, and it moves 2.0 per tick through bodies. Each fall gives the other team 1 point,
or 2 inside the fallen champion's own Red Zone: still one kill and one death.

**Watch Out:** the Red Zone is the strip at a team's own spawn edge, 5.0 units deep by default
(`red_zone_depth=0.0` turns it off). Spawn pads sit inside it, so hiding at home pays double.

## Winning, Drawing and Rewards

```python
actions = env.sample_actions(jax.random.key(1), state)
observation, state, reward, done, info = env.step(jax.random.key(2), state, actions)
print(reward.rewards[0], done.terminated[0], done.truncated[0], info.team_scores[0])
```

The first team to 20 wins, even on tick 300; reaching 20 together, the higher score wins
and equal scores draw. After 300 ticks with nobody at 20 it is a draw, whatever the score.
This is deliberate: a lead at the time limit is worth nothing, so a team cannot take a
lead and then hide. The reward is +1 for each winner and -1 for each loser, once, at the end.

**Watch Out:** draws, running games and unused slots get 0; fallen members share their team's
result. `done.terminated` and `done.truncated` can both be true; the scores name the winner.

## The Eight Scenarios

```python
for scenario_id in range(1, 9):
    print(marl_bgs.load_tdm_scenario(scenario_id).notes.splitlines()[0])
```

Eight hand-made end-games, such as "Be your teammate's moving cover". Each starts a few
points from 20 and 5 ticks from the limit (Scenario 3: 10), some champions already fallen.
Score a method on them with `evaluate_scenarios` ([evaluation.md](evaluation.md)).

**Watch Out:** scenarios are tests only. Never train, tune or choose checkpoints on them.

## Next

- [getting_started.md](getting_started.md): make, reset and step.
- [battle_client.md](battle_client.md): play a match yourself.
