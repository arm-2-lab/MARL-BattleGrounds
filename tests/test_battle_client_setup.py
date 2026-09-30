"""Check ordinary roster setup and fixed scenario ownership before live execution.

All team sizes use the existing map builder. Scenario config/state stays exact,
including configured dead agents. Explicit overrides fail before content loading,
and shared HTTP/setup replacement paths cannot change a fixed opponent or start.
"""

from __future__ import annotations

from itertools import product
from typing import Any
from unittest.mock import Mock

import jax
import numpy as np
import pytest
from tests.visual_debugger_fixtures import debugger_test_launch_specification

import marl_battlegrounds._battle_client.setup as battle_setup
from marl_battlegrounds._battle_client.control import create_session
from marl_battlegrounds._battle_client.protocol import (
    CommandRequestV1,
    CommandResponseV2,
    ConfirmDiscardAndReplaceCommandV1,
    ScenarioSwitchCommandV1,
    SetCombatConfigurationCommandV1,
)
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds._battle_client.system_worker import SystemWorker
from marl_battlegrounds.tasks import (
    canonical_tournament_rosters,
    load_tdm_scenario,
    make_standard_team_deathmatch_config,
)


def _same_tree(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert np.asarray(left).dtype == np.asarray(right).dtype
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("size_a,size_b", list(product(range(1, 6), repeat=2)))
def test_ordinary_rosters_keep_each_physical_team_size(
    size_a: int, size_b: int
) -> None:
    roster_a = ("hunter",) * size_a
    roster_b = ("priest",) * size_b
    prepared = prepare_battle_setup(
        map_id=0,
        team_a_roster=roster_a,
        team_b_roster=roster_b,
        team_a=["manual"] * size_a,
        team_b=["random"] * size_b,
    )
    config, state = prepared.scenario.build_scenario()
    expected = make_standard_team_deathmatch_config(
        map_id=0,
        team_a_roster=roster_a,
        team_b_roster=roster_b,
    )
    _same_tree(config, expected)
    np.testing.assert_array_equal(
        config.agent_profile.active_mask,
        [True] * size_a
        + [False] * (5 - size_a)
        + [True] * size_b
        + [False] * (5 - size_b),
    )
    assert state.alive_mask.shape == (10,)
    assert prepared.team_a == tuple(["manual"] * size_a)
    assert prepared.team_b == tuple(["random"] * size_b)
    assert prepared.seed == 0
    assert prepared.scenario.packaged_map is not None
    assert prepared.scenario.packaged_map.map_id == 0


def test_ordinary_defaults_use_shared_rosters_and_rules() -> None:
    prepared = prepare_battle_setup(map_id=4)
    config, _ = prepared.scenario.build_scenario()
    first, second = canonical_tournament_rosters()
    _same_tree(
        config,
        make_standard_team_deathmatch_config(
            map_id=4,
            team_a_roster=first,
            team_b_roster=second,
        ),
    )
    assert prepared.team_a == prepared.team_b == "manual"
    assert config.max_steps == 300
    assert config.team_deathmatch_score_threshold == 20
    assert config.team_deathmatch_red_zone_depth == 5.0


@pytest.mark.parametrize("scenario_id", range(1, 9))
def test_fixed_scenario_keeps_every_config_and_state_leaf(scenario_id: int) -> None:
    prepared = prepare_battle_setup(scenario_id=scenario_id)
    loaded = load_tdm_scenario(scenario_id)
    config, state = prepared.scenario.build_scenario()
    _same_tree(config, loaded.config)
    _same_tree(state, loaded.initial_state)
    specification = prepared.scenario.scenario_specification
    assert specification is not None
    assert specification.horizon == loaded.info.horizon
    assert state.alive_mask.shape == (10,)
    assert int(np.asarray(config.agent_profile.active_mask).sum()) == sum(
        loaded.info.team_sizes
    )
    assert prepared.seed == 0
    assert prepared.team_a == "manual"
    assert prepared.team_b == (
        "scenario_5" if scenario_id in (3, 5, 8) else "reactive_tdm"
    )
    assert prepared.scenario.build_scenario()[1] is state


@pytest.mark.parametrize(
    "override",
    [
        {"team_b": "reactive_tdm"},
        {"team_b": None},
        {"team_b": "checkpoints/mappo"},
        {"map_id": 0},
        {"map_id": None},
        {"team_a_roster": canonical_tournament_rosters()[0]},
        {"team_b_roster": canonical_tournament_rosters()[1]},
        {"seed": 0},
        {"max_steps": 300},
        {"score_threshold": 20},
        {"red_zone_depth": 5.0},
    ],
)
def test_fixed_overrides_fail_before_loading_content(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
) -> None:
    loader = Mock(side_effect=AssertionError("Content must not load"))
    monkeypatch.setattr(battle_setup, "list_tdm_scenarios", loader)
    with pytest.raises(ValueError) as caught:
        prepare_battle_setup(scenario_id=1, **override)
    loader.assert_not_called()
    if "team_b" in override:
        assert str(caught.value) == (
            "Scenario opponents are fixed. Remove team_b. "
            "Only Team A controllers can be chosen."
        )
    else:
        assert f"Remove {next(iter(override))}" in str(caught.value)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"map_id": -1},
        {"map_id": 52},
        {"map_id": True},
        {"scenario_id": 0},
        {"scenario_id": 9},
        {"scenario_id": True},
        {"map_id": 0, "seed": True},
        {"map_id": 0, "seed": -1},
        {"map_id": 0, "team_a_roster": []},
        {"map_id": 0, "team_b_roster": ["mage"] * 6},
        {"map_id": 0, "team_a": ["manual"]},
        {"scenario_id": 1, "team_a": ["manual"] * 3},
        {"map_id": 0, "team_a": object()},
        {"map_id": 0, "team_a": [object()] * 5},
        {"map_id": 0, "team_b": ""},
    ],
)
def test_invalid_setup_never_produces_a_session(options: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        prepare_battle_setup(**options)


def _fixed_service() -> DebuggerService:
    prepared = prepare_battle_setup(scenario_id=1)
    session = create_session(
        prepared.scenario,
        seed=prepared.seed,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="manual",
        team_b_controller="reactive_tdm",
        execution_information_mode="shared_obs",
    )
    return DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        offered_systems={"poison": "must_never_load:factory"},
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("controller", ["reactive_tdm", "system:poison"])
def test_fixed_http_configuration_rejects_before_worker_or_discard(
    monkeypatch: pytest.MonkeyPatch,
    wrapped: bool,
    controller: str,
) -> None:
    service = _fixed_service()
    original = service.session
    start = Mock(side_effect=AssertionError("Worker must not load"))
    monkeypatch.setattr(SystemWorker, "prepare", start)
    configuration = SetCombatConfigurationCommandV1(
        team_a_controller="manual",
        team_b_controller=controller,
        execution_information_mode="shared_obs",
    )
    command = (
        ConfirmDiscardAndReplaceCommandV1(replacement=configuration)
        if wrapped
        else configuration
    )
    frame = service.current_frame()
    result = service.apply_command(
        CommandRequestV1(
            client_id="fixed-client",
            command_id="override",
            base_revision=frame.revision,
            command=command,
        )
    )
    assert isinstance(result.payload, CommandResponseV2)
    assert result.payload.notice == battle_setup.SCENARIO_OPPONENT_ERROR
    assert service.session is original
    start.assert_not_called()
    service.close()


def test_fixed_source_replacement_and_equal_switch_do_not_change_match() -> None:
    service = _fixed_service()
    original = service.session
    with pytest.raises(ValueError, match="Scenarios are fixed"):
        service.load_scenario(original.scenario)
    frame = service.current_frame()
    result = service.apply_command(
        CommandRequestV1(
            client_id="fixed-client",
            command_id="switch",
            base_revision=frame.revision,
            command=ScenarioSwitchCommandV1(scenario_name=original.scenario.name),
        )
    )
    assert isinstance(result.payload, CommandResponseV2)
    assert "Scenarios are fixed" in str(result.payload.notice)
    assert service.session is original
    service.close()
