"""Check that every valid JAX System runs compiled in the BattleClient, once.

Loaded and custom Systems once ran uncompiled in the live client, so JAX
rebuilt and recompiled the whole model on every turn (seconds per Lock In).
These tests use Systems this project has never seen (defined here), beside
built-in names, on ordinary maps and fixed scenarios:

- every turn of a pure JAX pair goes through the one compiled pair program;
- the pair program and the Core step compile at most once, on the first turn;
  after it JAX compiles nothing at all, counted with JAX's own compile events
  (inputs are committed to their device, so the second turn does not compile a
  second copy);
- the compiled route chooses exactly the actions, memory and state of the
  uncompiled shared pair authority;
- a pair with a host method keeps the uncompiled host route, receives its own
  objects, and compiles nothing after its first turn;
- the launch warm-up (control.warm_compiled_turn) compiles the first turn, so
  no Lock In compiles anything, and never calls a host method (an LLM's cost
  and outside effects belong to real turns only).
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client import control
from marl_battlegrounds._battle_client.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.setup import BattleSetup, prepare_battle_setup
from marl_battlegrounds.evaluation import policy_execution
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.policies.actor import ActorAction

TURNS = 3
COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
# The real compiled programs, read before any test replaces them with a counter.
REAL_PAIR = control._COMPILED_SYSTEM_PAIR  # pyright: ignore[reportPrivateUsage]
REAL_STEP = control._COMPILED_STEP  # pyright: ignore[reportPrivateUsage]


def _session(setup: BattleSetup) -> DebuggerSession:
    resolved = resolve_controllers(setup)
    fixture = debugger_test_launch_specification(setup.seed)
    return control.create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=build_debugger_evaluation_launch_specification_v1(
            root_seed=setup.seed,
            code_revision=fixture.code_revision,
            capture_profile="debug",
        ),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller=resolved.controllers[0],
        team_b_controller=resolved.controllers[1],
        execution_information_mode="shared_obs",
        systems=resolved.systems,
        borrow_systems=True,
        manual_slots=resolved.manual_slots,
        slot_labels=resolved.slot_labels,
    )


def _noise(key: jax.Array) -> jax.Array:
    return jax.random.randint(key, (9,), 0, 3)


def _new_jax_system(name: str, shift: int) -> System:
    # A recurrent method with weights, built fresh so no earlier test shares it.
    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: jax.Array
    ) -> jax.Array:
        return jnp.zeros(inputs.valid.shape, jnp.int32)

    def choose(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: jax.Array
    ) -> SystemOutput:
        noise = jax.vmap(_noise)(keys)
        priority = (jnp.arange(9)[None, :] + memory[:, None] + shift) % 9 + noise
        weighted = (
            inputs.action_mask.move_mask * (priority * variables["scale"])[:, None, :]
        )
        moves = jnp.argmax(weighted, axis=-1).astype(jnp.int32)
        zeros = jnp.zeros_like(moves)
        return SystemOutput(
            ActorAction(moves, zeros, zeros),
            memory + inputs.valid.astype(jnp.int32),
            policy_ids=jnp.zeros(inputs.active_mask.shape, jnp.int32),
        )

    # Weights committed to a device, as loading a saved checkpoint leaves them.
    scale = jnp.asarray(1.5, jnp.float32)
    # Committed weights make a compiled call's outputs committed, which is what
    # used to compile a second copy of each program on the second turn.
    return System(
        name,
        choose,
        init=initialize,
        variables={"scale": jax.device_put(scale, scale.sharding)},
        components=({"name": name},),
    )


def _host_system(calls: list[int] | None = None) -> System:
    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: jax.Array
    ) -> list[int]:
        return [0 for _ in inputs.valid]

    def choose(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: jax.Array
    ) -> SystemOutput:
        if calls is not None:
            calls.append(1)
        moves = np.argmax(np.asarray(inputs.action_mask.move_mask), axis=-1).astype(
            np.int32
        )
        zeros = np.zeros_like(moves)
        return SystemOutput(ActorAction(moves, zeros, zeros), memory)

    return System("Host", choose, init=initialize, execution="host")


def _same_tree(first: object, second: object) -> None:
    def plain(leaf: object) -> np.ndarray:
        if isinstance(leaf, jax.Array) and jnp.issubdtype(
            leaf.dtype, jax.dtypes.prng_key
        ):
            return np.asarray(jax.random.key_data(leaf))
        return np.asarray(leaf)

    assert jax.tree.structure(first) == jax.tree.structure(second)
    for actual, expected in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        np.testing.assert_array_equal(plain(actual), plain(expected))


@contextmanager
def _counting_compiles() -> Generator[list[float]]:
    # Every XLA compile JAX finishes while the block runs, eager or jitted.
    compiles: list[float] = []

    def listen(event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event == COMPILE_EVENT:
            compiles.append(duration_secs)

    jax.monitoring.register_event_duration_secs_listener(listen)
    try:
        yield compiles
    finally:
        jax.monitoring.unregister_event_duration_listener(listen)


def _cache_sizes() -> tuple[int, int]:
    return (
        REAL_PAIR._cache_size(),  # pyright: ignore[reportFunctionMemberAccess]
        REAL_STEP._cache_size(),  # pyright: ignore[reportFunctionMemberAccess]
    )


def _play(session: DebuggerSession, turns: int) -> list[DebuggerSession]:
    played: list[DebuggerSession] = []
    for _ in range(turns):
        session = control.submit_interactive(session)
        played.append(session)
    return played


SETUPS: dict[str, Callable[[], BattleSetup]] = {
    "new JAX method against tdm-beta": lambda: prepare_battle_setup(
        map_id=0,
        team_a=_new_jax_system("New A", 1),
        team_b="tdm-beta",
        max_steps=8,
    ),
    "Manual beside a new JAX method against another": lambda: prepare_battle_setup(
        map_id=47,
        team_a=["manual", *(_new_jax_system("New B", 2),) * 4],
        team_b=_new_jax_system("New C", 5),
        seed=3,
        max_steps=8,
    ),
    "new JAX method in a fixed scenario": lambda: prepare_battle_setup(
        scenario_id=1, team_a=_new_jax_system("New D", 4)
    ),
}


def _play_compiled(
    monkeypatch: pytest.MonkeyPatch, session: DebuggerSession
) -> list[DebuggerSession]:
    calls = {"compiled": 0}

    def counted(*args: object, **kwargs: object) -> object:
        calls["compiled"] += 1
        return REAL_PAIR(*args, **kwargs)

    monkeypatch.setattr(control, "_COMPILED_SYSTEM_PAIR", counted)
    before = _cache_sizes()
    with _counting_compiles() as first_compiles:
        first = control.submit_interactive(session)
    after_first = _cache_sizes()
    with _counting_compiles() as later_compiles:
        played = [first, *_play(first, TURNS - 1)]
    # Every turn used the compiled pair program.
    assert calls["compiled"] == TURNS
    # A never-seen method must compile on its first turn; this also proves the
    # compile counter works. Afterwards nothing compiles at all: no second copy
    # of a program (uncommitted inputs) and no uncompiled model work.
    assert first_compiles
    assert later_compiles == []
    assert after_first[0] - before[0] <= 1
    assert after_first[1] - before[1] <= 1
    assert _cache_sizes() == after_first
    monkeypatch.setattr(control, "_COMPILED_SYSTEM_PAIR", REAL_PAIR)
    return played


@pytest.mark.parametrize("case", list(SETUPS))
def test_jax_systems_run_compiled_and_compile_only_on_the_first_turn(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    _play_compiled(monkeypatch, _session(SETUPS[case]()))


def test_compiled_turns_match_the_uncompiled_pair_authority_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The light case: built-in opponents are slow to run uncompiled.
    session = _session(SETUPS["Manual beside a new JAX method against another"]())
    compiled = _play_compiled(monkeypatch, session)
    monkeypatch.setattr(
        control,
        "_COMPILED_SYSTEM_PAIR",
        policy_execution._apply_system_pair,  # pyright: ignore[reportPrivateUsage]
    )
    reference = _play(session, TURNS)
    for fast, slow in zip(compiled, reference, strict=True):
        _same_tree(fast.environment_state, slow.environment_state)
        _same_tree(fast.system_memory, slow.system_memory)


def test_host_pairs_stay_uncompiled_and_add_no_program_after_the_first_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(
        prepare_battle_setup(
            map_id=0,
            team_a=_host_system(),
            team_b=_new_jax_system("New E", 3),
            max_steps=8,
        )
    )
    calls = {"compiled": 0}

    def counted(*args: object, **kwargs: object) -> object:
        calls["compiled"] += 1
        raise AssertionError("a host pair must not use the compiled pair program")

    monkeypatch.setattr(control, "_COMPILED_SYSTEM_PAIR", counted)
    first = control.submit_interactive(session)
    after_first = _cache_sizes()
    with _counting_compiles() as later_compiles:
        played = _play(first, TURNS)
    assert calls["compiled"] == 0
    # The host route keeps its JAX opponent compiled: nothing compiles again.
    assert later_compiles == []
    assert _cache_sizes() == after_first
    assert int(played[-1].state.step_count) == TURNS + 1


def test_warm_up_compiles_the_first_turn_so_no_lock_in_compiles() -> None:
    session = _session(
        prepare_battle_setup(
            map_id=0,
            team_a=_new_jax_system("New F", 6),
            team_b=_new_jax_system("New G", 7),
            max_steps=8,
        )
    )
    with _counting_compiles() as warm:
        assert control.warm_compiled_turn(session)
    assert warm
    with _counting_compiles() as played:
        _play(session, TURNS)
    assert played == []


def test_warm_up_never_calls_a_host_method() -> None:
    calls: list[int] = []
    session = _session(
        prepare_battle_setup(
            map_id=0,
            team_a=_host_system(calls),
            team_b=_new_jax_system("New H", 8),
            max_steps=8,
        )
    )
    assert not control.warm_compiled_turn(session)
    assert calls == []
