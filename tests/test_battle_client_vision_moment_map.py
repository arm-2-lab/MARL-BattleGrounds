"""Check that a packaged map with exchanged spawn banks keeps its map name.

The BattleClient's Vision Moment plays map 47 with its two spawn banks
exchanged. These tests pin the evaluation bridge contract for such a scenario
(``DebuggerScenario.spawn_banks_exchanged``):

- A packaged map declared with exchanged banks is checked with the banks put
  back. The recorded keys name the registered map (map 47, Teutoburg Forest)
  and record the exchange (``marl_bgs.spawn_locations`` "1" and a SHA-256
  ``marl_bgs.source_config_id``), and the recorded map that the live match
  summary shows reads "Teutoburg Forest", map 47.
- A declaration that fails the check (the banks were not exchanged) records a
  custom map and never claims map 47.
- Exchanged banks without the declaration still fail the exact bank-order
  check, as before.
- Only a packaged map may declare exchanged banks, and the flag must be a bool.
"""

from __future__ import annotations

import re

import jax
import pytest

from marl_battlegrounds._battle_client.evaluation_bridge import (
    build_debugger_evaluation_context_v1,
    build_debugger_evaluation_launch_specification_v1,
)
from marl_battlegrounds._battle_client.model import DebuggerScenario
from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.evaluation.map_identity import recorded_map
from marl_battlegrounds.evaluation.models import (
    CodeRevisionV1,
    EvaluationEpisodeContextV4,
)
from marl_battlegrounds.tasks import (
    _swap_spawn_banks,  # pyright: ignore[reportPrivateUsage]
    canonical_tournament_rosters,
    list_tdm_maps,
    make_standard_team_deathmatch_config,
)

MAP_ID = 47


def _map_47_config(*, exchanged: bool) -> EnvConfig:
    team_a, team_b = canonical_tournament_rosters()
    config = make_standard_team_deathmatch_config(
        map_id=MAP_ID, team_a_roster=team_a, team_b_roster=team_b
    )
    return _swap_spawn_banks(config) if exchanged else config


def _scenario(config: EnvConfig, *, declared: bool) -> DebuggerScenario:
    state, _, _, _ = reset(config, jax.random.key(0))
    return DebuggerScenario(
        name="map_47_exchanged",
        title="Teutoburg Forest",
        description="Map 47 with its spawn banks exchanged.",
        mode="interactive",
        build_scenario=lambda: (config, state),
        frames=(),
        default_controlled_slot=1,
        packaged_map=list_tdm_maps()[MAP_ID],
        spawn_banks_exchanged=declared,
    )


def _context(config: EnvConfig, *, declared: bool) -> EvaluationEpisodeContextV4:
    launch = build_debugger_evaluation_launch_specification_v1(
        root_seed=7,
        code_revision=CodeRevisionV1(
            package_version="0.0.0",
            commit_sha="a" * 40,
            source_tree_digest="b" * 64,
            is_dirty=False,
            dirty_patch_digest=None,
        ),
        capture_profile="debug",
    )
    return build_debugger_evaluation_context_v1(
        launch,
        scenario=_scenario(config, declared=declared),
        config=config,
        run_generation=0,
        action_source_kind="manual",
        team_a_controller="manual",
        team_b_controller="manual",
        execution_information_mode="shared_obs",
    )


def test_declared_exchanged_map_47_is_named_and_records_the_exchange() -> None:
    context = _context(_map_47_config(exchanged=True), declared=True)
    keys = {row.name: row.value for row in context.aggregation_keys}
    assert keys["map_id"] == "47"
    assert keys["map_origin"] == "registered"
    assert keys["marl_bgs.spawn_locations"] == "1"
    assert re.fullmatch(r"[0-9a-f]{64}", keys["marl_bgs.source_config_id"])
    shown = recorded_map(context)
    assert shown.display_name == "Teutoburg Forest"
    assert shown.map_id == MAP_ID
    assert context.identity.layout.identifier == list_tdm_maps()[MAP_ID].source.asset_id


def test_failed_exchange_check_records_a_custom_map() -> None:
    context = _context(_map_47_config(exchanged=False), declared=True)
    keys = {row.name: row.value for row in context.aggregation_keys}
    assert keys["map_origin"] == "custom"
    assert "map_id" not in keys
    assert "marl_bgs.spawn_locations" not in keys
    shown = recorded_map(context)
    assert shown.display_name == "Custom Map"
    assert shown.map_id is None
    assert context.identity.layout.identifier == "resolved-debugger-environment"


def test_undeclared_exchanged_banks_still_fail_the_exact_check() -> None:
    with pytest.raises(ValueError, match="does not match the episode geometry"):
        _context(_map_47_config(exchanged=True), declared=False)


def test_only_a_packaged_map_declares_exchanged_banks() -> None:
    config = _map_47_config(exchanged=True)
    state, _, _, _ = reset(config, jax.random.key(0))
    with pytest.raises(ValueError, match="Only a packaged map"):
        DebuggerScenario(
            name="no_map",
            title="No Map",
            description="No packaged map.",
            mode="interactive",
            build_scenario=lambda: (config, state),
            frames=(),
            default_controlled_slot=1,
            spawn_banks_exchanged=True,
        )
    with pytest.raises(TypeError, match="True or False"):
        DebuggerScenario(
            name="bad_flag",
            title="Bad Flag",
            description="Not a bool.",
            mode="interactive",
            build_scenario=lambda: (config, state),
            frames=(),
            default_controlled_slot=1,
            packaged_map=list_tdm_maps()[MAP_ID],
            spawn_banks_exchanged=1,  # pyright: ignore[reportArgumentType]
        )
