"""Check live method loading, object identity, grouping and checkpoint-only input.

Repeated references load once, distinct objects stay distinct, and compositions
keep physical slot ownership without reading a System's internal structure.
Browser checkpoint paths cannot become trusted Python factory expressions.
"""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds._battle_client import methods
from marl_battlegrounds._battle_client.setup import BattleSetup, prepare_battle_setup
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    apply_systems,
    init_systems,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput


@pytest.fixture(scope="module")
def prepared() -> BattleSetup:
    return prepare_battle_setup(map_id=0)


def _scalar(
    variables: PolicyTree,
    memory: PolicyTree,
    actor: ActorInput,
    mask: ActionMask,
    key: jax.Array,
) -> tuple[ActorAction, PolicyTree]:
    zero = jnp.asarray(0, jnp.int32)
    return ActorAction(zero, zero, zero), memory


def test_repeated_sources_load_once_across_teams_and_keep_variables(
    prepared: BattleSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    variables = np.ones(3, np.float32)
    original = Policy("Original", _scalar, variables=variables)
    calls: list[str] = []

    def load(source: str) -> Policy:
        calls.append(source)
        return original

    monkeypatch.setattr(methods, "load_method", load)
    result = methods.resolve_controllers(
        replace(prepared, team_a="custom:factory", team_b=("custom:factory",) * 5)
    )
    assert calls == ["custom:factory"]
    assert len(result.original_methods) == 1
    assert result.original_methods[0] is original
    assert len(result.methods) == 1
    assert result.methods[0].variables is variables
    assert result.methods[0]._policies[0] is original  # pyright: ignore[reportPrivateUsage]
    assert result.systems is not None
    assert result.systems[0] is result.methods[0]
    assert result.manual_slots == ()
    assert result.slot_labels == ("Original",) * 10


def test_path_spellings_share_loading_and_relative_paths_are_resolved(
    prepared: BattleSetup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "actor"
    folder.mkdir()
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []

    def load(source: str) -> Policy:
        calls.append(source)
        return Policy("Actor", _scalar)

    monkeypatch.setattr(methods, "load_method", load)
    result = methods.resolve_controllers(
        replace(
            prepared,
            team_a=(
                Path("actor"),
                "./actor",
                folder / ".." / "actor",
                str(folder),
                folder,
            ),
            team_b=str(folder),
        )
    )
    assert calls == [str(folder)]
    assert len(result.methods) == 1
    assert result.systems is not None and result.systems[0] is result.systems[1]


def test_factory_wins_over_folder_unless_path_is_explicit(
    prepared: BattleSetup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "custom:factory"
    folder.mkdir()
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []

    def load(source: str) -> Policy:
        calls.append(source)
        return Policy(source, _scalar)

    monkeypatch.setattr(methods, "load_method", load)
    result = methods.resolve_controllers(
        replace(
            prepared,
            team_a="custom:factory",
            team_b=(Path("custom:factory"), "./custom:factory", folder, folder, folder),
        )
    )
    assert calls == ["custom:factory", str(folder)]
    assert len(result.methods) == 2
    assert result.systems is not None and result.systems[0] is not result.systems[1]


@pytest.mark.parametrize("linked_part", ["leaf", "parent", "before_parent_step"])
@pytest.mark.parametrize("cached", [False, True])
def test_linked_checkpoint_is_rejected_even_after_target_was_loaded(
    prepared: BattleSetup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    linked_part: str,
    cached: bool,
) -> None:
    folder = tmp_path / "models" / "actor"
    folder.mkdir(parents=True)
    alias = tmp_path / "alias"
    if linked_part == "parent":
        alias.symlink_to(folder.parent, target_is_directory=True)
        source = alias / "actor"
    else:
        alias.symlink_to(folder, target_is_directory=True)
        source = alias if linked_part == "leaf" else alias / ".." / "actor"
    calls: list[str] = []

    def load(reference: str) -> Policy:
        calls.append(reference)
        return Policy("Saved", _scalar)

    monkeypatch.setattr(methods, "load_method", load)
    with pytest.raises(ValueError, match="symbolic links"):
        methods.resolve_controllers(
            replace(prepared, team_a=folder if cached else "manual", team_b=str(source))
        )
    assert calls == ([str(folder)] if cached else [])


@pytest.mark.parametrize("linked_parent", [False, True])
def test_browser_rejects_linked_paths_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked_parent: bool
) -> None:
    folder = tmp_path / "models" / "actor"
    folder.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(
        folder.parent if linked_parent else folder, target_is_directory=True
    )

    def forbidden(_source: str) -> Policy:
        pytest.fail("Linked input must fail before loading")

    monkeypatch.setattr(methods, "load_method", forbidden)
    with pytest.raises(ValueError, match="symbolic links"):
        methods.load_checkpoint_path(alias / "actor" if linked_parent else alias)


def test_browser_colon_folder_stays_a_checkpoint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "custom:factory"
    folder.mkdir()
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    saved = Policy("Saved", _scalar)

    def load(source: str) -> Policy:
        calls.append(source)
        return saved

    monkeypatch.setattr(methods, "load_method", load)
    assert methods.load_checkpoint_path("custom:factory") is saved
    assert calls == [str(folder)]


def test_distinct_equal_looking_objects_remain_separate(prepared: BattleSetup) -> None:
    first = Policy("Same", _scalar)
    second = Policy("Same", _scalar)
    result = methods.resolve_controllers(replace(prepared, team_a=first, team_b=second))
    assert len(result.methods) == 2
    assert result.original_methods[0] is first
    assert result.original_methods[1] is second
    assert result.systems is not None and result.systems[0] is not result.systems[1]
    assert result.methods[0]._policies[0] is first  # pyright: ignore[reportPrivateUsage]
    assert result.methods[1]._policies[0] is second  # pyright: ignore[reportPrivateUsage]


def test_whole_team_opaque_system_is_retained(prepared: BattleSetup) -> None:
    def apply(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> SystemOutput:
        zeros = jnp.zeros(inputs.active_mask.shape, jnp.int32)
        return SystemOutput(ActorAction(zeros, zeros, zeros), memory)

    original = System(
        "Opaque", apply, components=tuple({"name": f"Part {i}"} for i in range(8))
    )
    result = methods.resolve_controllers(replace(prepared, team_a=original))
    assert result.systems is not None and result.systems[0] is original
    assert result.methods == (original,)
    assert result.original_methods[0] is original
    assert result.manual_slots == tuple(range(5, 10))


def test_grouped_host_calls_keep_owned_slots_and_manual_drafts_private(
    prepared: BattleSetup,
) -> None:
    calls: Counter[str] = Counter()
    owned: dict[str, list[list[list[bool]]]] = {}

    def method(name: str) -> System:
        def apply(
            variables: PolicyTree,
            memory: PolicyTree,
            inputs: SystemInput,
            key: jax.Array,
        ) -> SystemOutput:
            calls[name] += 1
            owned.setdefault(name, []).append(
                np.asarray(inputs.controlled_mask).tolist()
            )
            assert variables == ()
            zeros = cast(jax.Array, np.zeros(inputs.active_mask.shape, np.int32))
            return SystemOutput(ActorAction(zeros, zeros, zeros), memory)

        return System(name, apply, execution="host")

    first, second = method("First"), method("Second")
    result = methods.resolve_controllers(
        replace(
            prepared,
            team_a=(first, first, first, second, "manual"),
            team_b=first,
        )
    )
    assert result.systems is not None
    a, b = result.systems
    assert a is not None and b is not None
    config, _ = prepared.scenario.build_scenario()
    env = make("tdm", metrics="none", balance_spawn_locations=False)
    observation, wrapped = env.reset(jax.random.key(0), env_config=config)
    memory = init_systems(a, b, observation, wrapped, jax.random.key(1))
    calls.clear()
    apply_systems(a, b, memory, observation, wrapped, jax.random.key(2))
    assert calls == {"First": 2, "Second": 1}
    assert owned["Second"] == [[[False, False, False, True, False]]]
    assert owned["First"] == [[[True, True, True, False, False]], [[True] * 5]]
    assert result.manual_slots == (4,)
    assert result.slot_labels[:5] == ("First", "First", "First", "Second", "Manual")


@pytest.mark.parametrize("scenario_id", range(1, 9))
def test_fixed_opponent_is_not_loaded_or_replaced(
    scenario_id: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = prepare_battle_setup(scenario_id=scenario_id)

    def forbidden(_source: str) -> Policy:
        raise AssertionError("Default Manual and fixed opponents need no method load")

    monkeypatch.setattr(methods, "load_method", forbidden)
    result = methods.resolve_controllers(setup)
    config, _ = setup.scenario.build_scenario()
    expected = tuple(np.flatnonzero(np.asarray(config.agent_profile.active_mask)[:5]))
    assert result.controllers == ("manual", setup.team_b)
    assert result.manual_slots == expected
    assert result.systems is None


@pytest.mark.parametrize("source", ["malicious:factory", "missing", "~/_missing_actor"])
def test_browser_checkpoint_text_never_runs_factory(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_source: str) -> Policy:
        raise AssertionError("A non-directory must not reach the method loader")

    monkeypatch.setattr(methods, "load_method", forbidden)
    with pytest.raises(ValueError, match="existing checkpoint folder"):
        methods.load_checkpoint_path(source)


def test_browser_checkpoint_folder_uses_shared_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Policy("Saved", _scalar)
    calls: list[str] = []

    def load(source: str) -> Policy:
        calls.append(source)
        return original

    monkeypatch.setattr(methods, "load_method", load)
    assert methods.load_checkpoint_path(tmp_path) is original
    assert calls == [str(tmp_path)]


def test_missing_explicit_path_never_becomes_factory(
    prepared: BattleSetup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_source: str) -> Policy:
        pytest.fail("Must reject path")

    monkeypatch.setattr(methods, "load_method", forbidden)
    with pytest.raises(ValueError, match="Checkpoint folder does not exist"):
        methods.resolve_controllers(replace(prepared, team_a=tmp_path / "a:b"))


def test_loader_failures_propagate_without_substitution(
    prepared: BattleSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ImportError("Training dependencies unavailable")

    def fail(_source: str) -> Policy:
        raise error

    monkeypatch.setattr(methods, "load_method", fail)
    with pytest.raises(ImportError) as caught:
        methods.resolve_controllers(replace(prepared, team_a="factory:load"))
    assert caught.value is error


@pytest.mark.parametrize("name", ["random", "tdm-alpha", "tdm-beta", "tdm-gamma"])
def test_real_builtin_resolver_reuses_existing_policy(
    prepared: BattleSetup, name: str
) -> None:
    resolved = methods.resolve_controllers(replace(prepared, team_a=name, team_b=name))
    assert resolved.systems is not None
    assert resolved.systems[0] is resolved.systems[1]
    assert resolved.methods[0].name == name
    assert resolved.manual_slots == ()


def test_real_factory_wrong_return_is_rejected(
    prepared: BattleSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    module = types.ModuleType("battle_invalid_factory")
    module.load = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(TypeError, match="System or Policy"):
        methods.resolve_controllers(
            replace(prepared, team_a="battle_invalid_factory:load")
        )


def test_base_methods_do_not_import_training_dependencies(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    source = """
import importlib.abc
import sys
class BaseOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        optional = fullname.split('.')[0] in {'flax', 'optax', 'orbax'}
        if optional or fullname.startswith('marl_battlegrounds.training'):
            raise AssertionError('Base match imported training dependency: ' + fullname)
sys.meta_path.insert(0, BaseOnly())
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds._battle_client.methods import resolve_controllers
for name in ('manual', 'random', 'tdm-alpha', 'tdm-beta', 'tdm-gamma'):
    resolved = resolve_controllers(prepare_battle_setup(map_id=0, team_a=name))
    assert len(resolved.slot_labels) == 10
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["random", "tdm-alpha"])
def test_builtin_names_win_over_same_named_working_directories(
    prepared: BattleSetup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    folder = tmp_path / name
    folder.mkdir()
    monkeypatch.chdir(tmp_path)
    actual = methods.resolve_controllers(replace(prepared, team_a=name))
    assert actual.methods[0].name == name
    calls: list[str] = []

    def load(source: str) -> Policy:
        calls.append(source)
        return Policy("Checkpoint", _scalar)

    monkeypatch.setattr(methods, "load_method", load)
    methods.resolve_controllers(replace(prepared, team_a=Path(name)))
    methods.resolve_controllers(replace(prepared, team_a="./" + name))
    assert calls == [str(folder), str(folder)]


@pytest.mark.parametrize("source", ["random", "random_valid", ("random",) * 5])
def test_noshared_legacy_random_manual_uses_existing_control_route(
    prepared: BattleSetup,
    source: str | tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_reference: str) -> Policy:
        pytest.fail("Legacy NoSharedObs must not load or wrap a method")

    monkeypatch.setattr(methods, "load_method", forbidden)
    loaded = methods.resolve_controllers(
        replace(prepared, team_a=source, team_b="manual"),
        execution_information_mode="no_shared_obs",
    )
    assert loaded.controllers == ("random_valid", "manual")
    assert loaded.systems is None
    assert loaded.methods == loaded.original_methods == ()
    assert loaded.manual_slots == (5, 6, 7, 8, 9)
    assert loaded.slot_labels == ("Random",) * 5 + ("Manual",) * 5


@pytest.mark.parametrize(
    "source",
    ["custom:factory", "tdm-alpha", ("manual", "random", "random", "random", "random")],
)
def test_noshared_incompatible_sources_fail_before_loading(
    prepared: BattleSetup,
    source: str | tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_reference: str) -> Policy:
        pytest.fail("Information contract must be checked before loading")

    monkeypatch.setattr(methods, "load_method", forbidden)
    setup = replace(prepared, team_a=source)
    with pytest.raises(ValueError, match="Choose SharedObs"):
        methods.validate_controller_information(setup, "no_shared_obs")
    with pytest.raises(ValueError, match="Choose SharedObs"):
        methods.resolve_controllers(setup, execution_information_mode="no_shared_obs")
