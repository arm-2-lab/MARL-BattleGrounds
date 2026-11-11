"""Pin how the published Team Deathmatch bots play whole games.

TDM-Alpha, TDM-Beta and TDM-Gamma are Season 0's published scripted
baselines: paper numbers, Elo ratings and saved results all rest on how they
play. ``tests/environment/test_tdm_goldens.py`` already pins Alpha against
Beta, every action and every state, on eight maps. This file adds Gamma. Each
test plays Gamma as Team A against Alpha or Beta, on every map in MAPS from
both spawn ends at once, with the canonical five-agent rosters and the
default 300-step limit. Core and the bots ignore their random keys, so a game
is fixed by its pairing, map and spawn end. The test hashes every action the
bots submit until each game ends, with each game's length, and compares the
SHA-256 digest with the one pinned on 11 November 2026. The released 1.0.4
package gives the same two digests, so they pin the released play.

MAPS holds the maps where these games get stuck, because they lean hardest
on the code that steers around walls. On maps 36 and 39 agents walk into
walls again and again, and on map 12 Beta's Warrior stands still for 125
ticks because it finds no way around a wall.

A change that alters even one action in these games fails here. The most
likely place is the movement and target code in policies/reactive_common.py,
which the Capture the Flag bots will share; a Core change to Team Deathmatch
also fails test_tdm_goldens.py. A failure is investigated, never re-recorded
to make it pass. The digests change only with a deliberate new bot version.

CPU float results can depend on the processor and the JAX version, so, like
the goldens, the test skips, saying why, when it is not on the CPU or when the
machine type or JAX version differs from the recording. The digests also held
with XLA limited to AVX2, which every hosted runner has. To re-record for
another machine or JAX version, set RECORDED and take the digests from this
file run on the released 1.0.4 source, never on the current tree.
"""

import hashlib
import platform
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.environment import EnvironmentState
from marl_battlegrounds.evaluation.policy_execution import apply_policies
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import (
    balanced_spawn_configs,
    make_standard_team_deathmatch_config,
)

Carry = tuple[Observations, EnvironmentState, Array]

RECORDED = ("x86_64", "0.10.1")
MAPS = (12, 36, 39)
EXPECTED = {
    ("tdm-gamma", "tdm-alpha"): (
        "cb651c8ba06f218ef76eb0d3e5f7fb18e685a9d42ecc3530dab0a820dffdbe0b"
    ),
    ("tdm-gamma", "tdm-beta"): (
        "0255bf361b190afc01d588344ee29b732e320dbd95cd289893b308931495a944"
    ),
}


def play(team_a: str, team_b: str) -> tuple[np.ndarray, np.ndarray]:
    roster_a, roster_b = marl_bgs.canonical_tournament_rosters()
    sources = [
        make_standard_team_deathmatch_config(
            map_id=map_id, team_a_roster=roster_a, team_b_roster=roster_b
        )
        for map_id in MAPS
    ] * 2
    stacked: EnvConfig = jax.tree.map(lambda *x: jnp.stack(x), *sources)
    games = len(sources)
    env = marl_bgs.make(
        "tdm",
        env_config=balanced_spawn_configs(stacked, num_envs=games),
        num_envs=games,
        metrics="none",
    )
    apply_a = marl_bgs.policy(team_a).apply
    apply_b = marl_bgs.policy(team_b).apply
    # The bots keep no weights and no memory, hence the four empty trees.
    bots = partial(apply_policies, apply_a, apply_b, (), (), (), ())
    obs, state = env.reset(jax.random.key(0))

    def tick(carry: Carry, key: Array) -> tuple[Carry, tuple[Array, Array]]:
        obs, state, ended = carry
        action_key, step_key = jax.random.split(key)
        actor_keys = jax.random.split(action_key, (games, 10))
        mask = env.get_action_mask(state)
        action, _, _ = jax.vmap(bots)(obs, mask, actor_keys)
        obs, state, _, done, _ = env.step(step_key, state, action)
        heads = jnp.stack(tuple(action), axis=-1)
        return (obs, state, ended | done.done), (heads, ended)

    keys = jax.random.split(jax.random.key(1), 300)
    start = (obs, state, jnp.zeros(games, bool))
    _, (heads, ended) = jax.lax.scan(tick, start, keys)
    return np.asarray(heads), np.asarray(ended)


@pytest.mark.parametrize(("team_a", "team_b"), list(EXPECTED))
def test_published_bots_play_unchanged(team_a: str, team_b: str) -> None:
    here = (platform.machine(), jax.__version__)
    if jax.default_backend() != "cpu" or here != RECORDED:
        pytest.skip(
            f"digests were recorded on the CPU, {RECORDED[0]}, JAX {RECORDED[1]}; "
            f"this run is {jax.default_backend()}, {here[0]}, JAX {here[1]}"
        )
    heads, ended = play(team_a, team_b)
    digest = hashlib.sha256()
    lengths = []
    for game in range(heads.shape[1]):
        played = heads[~ended[:, game], game]
        lengths.append(len(played))
        digest.update(np.int64(len(played)).tobytes() + played.astype("<i4").tobytes())
    assert digest.hexdigest() == EXPECTED[(team_a, team_b)], (
        f"{team_a} vs {team_b} on maps {MAPS}, both ends: game lengths {lengths}, "
        f"digest {digest.hexdigest()}. Find the change; do not re-record."
    )
