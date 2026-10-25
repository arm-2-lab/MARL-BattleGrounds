"""Record and check the Team Deathmatch golden games.

The goldens prove that adding Capture the Flag and King of the Hills leaves
Team Deathmatch exactly as it was. One golden game is one installed map
(0-51) from one spawn end (end 0 keeps the map's spawn banks, end 1 exchanges
them), with the canonical rosters, ``tdm-alpha`` as Team A and ``tdm-beta`` as
Team B, on the CPU, for at most 300 steps. Each game feeds every leaf of every
step into one running SHA-256 until the game ends. A game's episode ID is
1 + 2 * map + end, and its bots' random keys come from that ID, so a game plays
the same in any batch. The batch's episode ID allocator
(``last_reserved_episode_id``) holds the batch size, so it is left out.

Two paths play each game. ``core`` calls ``core.reset`` and ``core.step`` under
``vmap``, so the task mode is a batched value. ``handle`` plays through
``make("tdm")`` with training facts on. The leaf list was fixed when the
goldens were recorded at commit b4a5b39, before any Capture the Flag code. A
check hashes only those leaves, and requires every leaf added since to hold
its neutral value on every step: -1 for a leaf named ``*carrier_slots``, 0 for
any other.

Run from the repository root:

    JAX_PLATFORMS=cpu uv run --no-sync python -m tests.helpers.tdm_goldens \
        record OUT.json
    JAX_PLATFORMS=cpu uv run --no-sync python -m tests.helpers.tdm_goldens \
        check tests/fixtures/tdm_goldens.json [--maps 0 7 ...]

``record`` plays all 52 maps and writes the fixture. ``check`` plays the chosen
maps (all 52 by default) from both ends, prints each difference and exits 1
if there is any. A difference is investigated, never re-recorded.

The model goldens prove the same for the six released Season 0 models, which
may take actions the bots never take, and whose inputs pass through the model
loader. Each model, loaded by its typed name at its pinned commit, plays Team A
against ``tdm-beta`` on maps 36-51 from both ends: one batch of 32 games per
model through ``make("tdm")``, ``init_systems`` and ``apply_systems``. Its
submitted actions join the hash. These goldens run on the GPU, so they need
XLA's deterministic mode: without it, about 5-9% of games with a network
differ between two runs. A check runs only on the recorded device and JAX
version, and on all 32 games, because the GPU's arithmetic depends on both:

    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_FLAGS=--xla_gpu_deterministic_ops=true \
    uv run --no-sync python -m tests.helpers.tdm_goldens \
        check-models tests/fixtures/tdm_model_goldens.json

``record-models OUT.json`` writes that fixture, from a frozen ``git archive``
of b4a5b39 like the bot goldens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.environment import Environment, EnvironmentState, EpisodeInfo
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemState,
    apply_policies,
    apply_systems,
    init_systems,
    policy,
    shared_policy,
)
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.policies.shared_obs import (
    build_default_shared_obs_information_availability,
)
from marl_battlegrounds.released_models import PINNED_COMMITS
from marl_battlegrounds.tasks import (
    balanced_spawn_configs,
    canonical_tournament_rosters,
    make_standard_team_deathmatch_config,
)

type Leaf = NDArray[np.generic]
type Spec = dict[str, list[list[object]]]
type CoreReset = Callable[[EnvConfig], tuple[EnvState, Observation, ActionMask, Info]]
type CoreStep = Callable[
    [EnvConfig, Array, EnvState, Observation, ActionMask, int],
    tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
]
type HandleStep = Callable[
    [Environment, EnvironmentState, int],
    tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo],
]
type ModelStep = Callable[
    [Observations, EnvironmentState, SystemState, int],
    tuple[Observations, EnvironmentState, SystemState, dict[str, object]],
]

BASE_COMMIT = "b4a5b3946ef2397eee394cf6172202f598498139"
MAX_STEPS = 300
ALL_MAPS = tuple(range(52))
MODELS = tuple(
    f"MARL-BattleGrounds/tdm-season-0-{method}"
    for method in (
        "recurrent-ippo",
        "recurrent-mappo",
        "feedforward-ippo",
        "feedforward-mappo",
        "pqn-vdn",
        "qmix",
    )
)
MODEL_MAPS = tuple(range(36, 52))  # 32 games, an allowed GPU batch size.
DETERMINISTIC = "--xla_gpu_deterministic_ops=true"
_ROOT = jax.random.PRNGKey(0)
_ALPHA = policy("tdm-alpha").apply
_BETA = policy("tdm-beta").apply


def golden_games(map_ids: Sequence[int]) -> list[tuple[int, int]]:
    return [(m, 0) for m in map_ids] + [(m, 1) for m in map_ids]


def _episode_ids(map_ids: Sequence[int]) -> Array:
    # A game's identity comes from its map and end, never its batch position,
    # so any subset replays exactly as it was recorded in the full batch.
    return jnp.asarray([1 + 2 * m + e for m, e in golden_games(map_ids)], jnp.int32)


def _stack(*leaves: object) -> Array:
    return jnp.stack([jnp.asarray(leaf) for leaf in leaves])


def _configs(map_ids: Sequence[int]) -> EnvConfig:
    team_a, team_b = canonical_tournament_rosters()
    sources = [
        make_standard_team_deathmatch_config(
            map_id=m, team_a_roster=team_a, team_b_roster=team_b
        )
        for m in map_ids
    ] * 2
    stacked = cast(EnvConfig, jax.tree.map(_stack, *sources))
    return balanced_spawn_configs(stacked, num_envs=len(sources))


def _joint_action(
    observation: Observation, availability: Array, mask: ActionMask, keys: Array
) -> Action:
    action, _, _ = apply_policies(
        _ALPHA,
        _BETA,
        (),
        (),
        (),
        (),
        Observations(observation, availability),
        mask,
        keys,
    )
    return action


def _actor_keys(step: int | Array, episode_ids: Array) -> Array:
    def one(game: Array) -> Array:
        return jax.random.split(
            jax.random.fold_in(jax.random.fold_in(_ROOT, game), step), 10
        )

    return jax.vmap(one)(episode_ids)


def _core_reset_batch(
    configs: EnvConfig,
) -> tuple[EnvState, Observation, ActionMask, Info]:
    return jax.vmap(core.reset, in_axes=(0, None))(configs, _ROOT)


def _core_step_batch(
    configs: EnvConfig,
    ids: Array,
    state: EnvState,
    observation: Observation,
    mask: ActionMask,
    step: int,
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    availability = jax.vmap(build_default_shared_obs_information_availability)(
        configs.agent_profile.active_mask, configs.agent_profile.team_ids
    )
    actions = jax.vmap(_joint_action)(
        observation, availability, mask, _actor_keys(step, ids)
    )
    return jax.vmap(core.step, in_axes=(0, 0, 0, 0, None))(
        configs, state, mask, actions, _ROOT
    )


def _handle_step_batch(
    env: Environment, state: EnvironmentState, step: int
) -> tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]:
    actions = jax.vmap(_joint_action)(
        state.observation,
        state.source_availability,
        state.action_mask,
        _actor_keys(step, state.episode_id),
    )
    return env.step(_ROOT, state, actions)


_core_reset = cast(CoreReset, jax.jit(_core_reset_batch))
_core_step = cast(CoreStep, jax.jit(_core_step_batch))
_handle_step = cast(HandleStep, jax.jit(_handle_step_batch))


def _name(path: Sequence[object]) -> str:
    parts: list[str] = []
    for key in path:
        for attribute in ("name", "key", "idx"):
            if hasattr(key, attribute):
                parts.append(str(getattr(key, attribute)))
                break
    return ".".join(parts)


def _leaves(tree: object) -> list[tuple[str, Leaf]]:
    pairs: list[tuple[str, Leaf]] = []
    flat = cast(
        list[tuple[tuple[object, ...], Array]],
        jax.tree_util.tree_flatten_with_path(tree)[0],
    )
    for path, leaf in flat:
        name = _name(path)
        if name == "state.last_reserved_episode_id":
            continue  # The batch's ID allocator holds the batch size, not game play.
        if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        pairs.append((name, np.asarray(leaf)))
    return pairs


class _Trace:
    def __init__(self, count: int, base: Spec | None) -> None:
        self.count = count
        self.base = base
        self.hashes = [hashlib.sha256() for _ in range(count)]
        self.steps = [0] * count
        self.live = np.ones(count, dtype=bool)
        self.leaves: Spec = {}
        self.problems: list[str] = []

    def add(self, kind: str, tree: object, done: Leaf | None) -> None:
        pairs = _leaves(tree)
        spec: list[list[object]] = [
            [name, str(x.dtype), list(x.shape[1:])] for name, x in pairs
        ]
        self.leaves.setdefault(kind, spec)
        kept = pairs
        if self.base is not None:
            expected = {str(row[0]): row[1:] for row in self.base[kind]}
            actual = {str(row[0]): row[1:] for row in spec}
            for name in expected.keys() - actual.keys():
                self._problem(f"{kind} leaf {name} is missing")
            for name in expected.keys() & actual.keys():
                if expected[name] != actual[name]:
                    self._problem(
                        f"{kind} leaf {name}: {expected[name]} became {actual[name]}"
                    )
            order = {str(row[0]): i for i, row in enumerate(self.base[kind])}
            kept = sorted(
                ((name, x) for name, x in pairs if name in expected),
                key=lambda pair: order[pair[0]],
            )
            for name, x in pairs:
                neutral = -1 if name.endswith("carrier_slots") else 0
                if name not in expected and not np.all(x[self.live] == neutral):
                    self._problem(f"{kind} leaf {name} is not neutral ({neutral})")
        rows = np.concatenate(
            [
                np.ascontiguousarray(x)
                .reshape(self.count, x.size // self.count)
                .view(np.uint8)
                for _, x in kept
            ],
            axis=1,
        )
        for game in np.flatnonzero(self.live):
            self.hashes[game].update(rows[game].tobytes())
            self.steps[game] += kind == "step"
        if done is not None:
            self.live &= ~done.astype(bool)

    def _problem(self, text: str) -> None:
        if text not in self.problems:
            self.problems.append(text)


def play(map_ids: Sequence[int], path: str, base: Spec | None = None) -> _Trace:
    configs = _configs(map_ids)
    ids = _episode_ids(map_ids)
    count = 2 * len(map_ids)
    trace = _Trace(count, base)
    if path == "core":
        state, observation, mask, info = _core_reset(configs)
        trace.add(
            "reset",
            {"state": state, "observation": observation, "mask": mask, "info": info},
            None,
        )
        for step in range(MAX_STEPS):
            state, observation, reward, done, mask, info = _core_step(
                configs, ids, state, observation, mask, step
            )
            tree = {
                "state": state,
                "observation": observation,
                "reward": reward,
                "done": done,
                "mask": mask,
                "info": info,
            }
            trace.add("step", tree, np.asarray(done.done))
            if not trace.live.any():
                break
    else:
        env = marl_bgs.make(
            "tdm", env_config=configs, num_envs=count, training_facts=True
        )
        _, handle_state = env.reset(_ROOT, episode_id=ids)
        trace.add("reset", {"state": handle_state}, None)
        for step in range(MAX_STEPS):
            _, handle_state, reward, done, episode = _handle_step(
                env, handle_state, step
            )
            tree = {
                "state": handle_state,
                "reward": reward,
                "done": done,
                "info": episode,
            }
            trace.add("step", tree, np.asarray(done.done))
            if not trace.live.any():
                break
    return trace


def play_model(name: str, base: Spec | None = None) -> _Trace:
    model = marl_bgs.load_method(name)
    if not isinstance(model, System):
        raise TypeError(f"{name} loaded as {type(model).__name__}, not a System")
    beta = shared_policy(policy("tdm-beta"))
    count = 2 * len(MODEL_MAPS)
    env = marl_bgs.make(
        "tdm", env_config=_configs(MODEL_MAPS), num_envs=count, training_facts=True
    )
    observations, state = env.reset(_ROOT, episode_id=_episode_ids(MODEL_MAPS))
    memory = init_systems(model, beta, observations, state, _ROOT)

    def advance(
        observations: Observations,
        state: EnvironmentState,
        memory: SystemState,
        step: int,
    ) -> tuple[Observations, EnvironmentState, SystemState, dict[str, object]]:
        key = jax.random.fold_in(_ROOT, step)
        action, memory, _ = apply_systems(model, beta, memory, observations, state, key)
        observations, state, reward, done, episode = env.step(_ROOT, state, action)
        tree: dict[str, object] = {
            "state": state,
            "action": action,
            "reward": reward,
            "done": done,
            "info": episode,
        }
        return observations, state, memory, tree

    advance_jit = cast(ModelStep, jax.jit(advance))
    trace = _Trace(count, base)
    trace.add("reset", {"state": state}, None)
    for step in range(MAX_STEPS):
        observations, state, memory, tree = advance_jit(
            observations, state, memory, step
        )
        trace.add("step", tree, np.asarray(cast(DoneFlags, tree["done"]).done))
        if not trace.live.any():
            break
    return trace


def _differences(
    label: str,
    trace: _Trace,
    base: dict[str, list[object]],
    games: Sequence[tuple[int, int]],
    index: dict[tuple[int, int], int],
) -> list[str]:
    problems = [f"{label}: {text}" for text in trace.problems]
    for game, (map_id, end) in enumerate(games):
        i = index[(map_id, end)]
        if trace.steps[game] != base["steps"][i]:
            problems.append(
                f"{label}: map {map_id} end {end} ended after "
                f"{trace.steps[game]} steps, not {base['steps'][i]}"
            )
        elif trace.hashes[game].hexdigest() != base["digests"][i]:
            problems.append(f"{label}: map {map_id} end {end} played differently")
    return problems


def _device() -> str:
    return cast(str, jax.devices()[0].device_kind)


def record(out: Path) -> None:
    paths: dict[str, dict[str, object]] = {}
    for path in ("core", "handle"):
        trace = play(ALL_MAPS, path)
        paths[path] = {
            "leaves": trace.leaves,
            "steps": trace.steps,
            "digests": [h.hexdigest() for h in trace.hashes],
        }
    fixture = {
        "base_commit": BASE_COMMIT,
        "jax": jax.__version__,
        "machine": platform.machine(),
        "max_steps": MAX_STEPS,
        "games": golden_games(ALL_MAPS),
        "paths": paths,
    }
    out.write_text(json.dumps(fixture, indent=1) + "\n")


def check(fixture_path: Path, map_ids: Sequence[int] = ALL_MAPS) -> list[str]:
    fixture = json.loads(fixture_path.read_text())
    index = {(game[0], game[1]): i for i, game in enumerate(fixture["games"])}
    problems: list[str] = []
    for path, base in fixture["paths"].items():
        trace = play(map_ids, path, base["leaves"])
        problems += _differences(path, trace, base, golden_games(map_ids), index)
    return problems


def record_models(out: Path) -> None:
    traces = {name: play_model(name) for name in MODELS}
    fixture = {
        "base_commit": BASE_COMMIT,
        "jax": jax.__version__,
        "device": _device(),
        "max_steps": MAX_STEPS,
        "games": golden_games(MODEL_MAPS),
        "leaves": traces[MODELS[0]].leaves,  # The same for every model.
        "models": {
            name: {
                "commit": PINNED_COMMITS[name],
                "steps": trace.steps,
                "digests": [h.hexdigest() for h in trace.hashes],
            }
            for name, trace in traces.items()
        },
    }
    out.write_text(json.dumps(fixture, indent=1) + "\n")


def check_models(fixture_path: Path) -> list[str]:
    fixture = json.loads(fixture_path.read_text())
    here = (_device(), jax.__version__)
    if here != (fixture["device"], fixture["jax"]):
        return [
            f"model goldens were recorded on {fixture['device']} with JAX "
            f"{fixture['jax']}; this run is {here[0]} with JAX {here[1]}"
        ]
    games = golden_games(MODEL_MAPS)
    index = {game: i for i, game in enumerate(games)}
    problems: list[str] = []
    for name, base in fixture["models"].items():
        if base["commit"] != PINNED_COMMITS[name]:
            problems.append(
                f"{name}: pinned at {PINNED_COMMITS[name]}, not {base['commit']}"
            )
        trace = play_model(name, fixture["leaves"])
        problems += _differences(name, trace, base, games, index)
    return problems


def _main() -> int:
    parser = argparse.ArgumentParser(description="Team Deathmatch golden games.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("record").add_argument("out", type=Path)
    checking = commands.add_parser("check")
    checking.add_argument("fixture", type=Path)
    checking.add_argument("--maps", type=int, nargs="+", default=list(ALL_MAPS))
    commands.add_parser("record-models").add_argument("out", type=Path)
    commands.add_parser("check-models").add_argument("fixture", type=Path)
    args = parser.parse_args()
    if args.command.endswith("models") and DETERMINISTIC not in os.environ.get(
        "XLA_FLAGS", ""
    ):
        parser.error(f"model goldens need XLA_FLAGS={DETERMINISTIC}")
    if args.command == "record":
        record(args.out)
        return 0
    if args.command == "record-models":
        record_models(args.out)
        return 0
    if args.command == "check":
        problems = check(args.fixture, args.maps)
    else:
        problems = check_models(args.fixture)
    print("\n".join(problems) or "Team Deathmatch matches the goldens.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(_main())
