"""Check exact packaged starts, evaluator keys and immutable scenario ownership.

The live controller is compared with evaluate_episodes using the same saved
scenario, seed coordinate and numeric episode ID. A custom host initializer makes
its random key visible in memory, so action-only agreement cannot hide a wrong
initialization stream. Old debugger identity tests cover the unchanged route.
"""

# These checks compare the private live boundary with its existing public owner.
# pyright: reportPrivateUsage=false

from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any, NoReturn, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.helpers.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client import control
from marl_battlegrounds._battle_client.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from marl_battlegrounds._battle_client.model import DebuggerScenario, DebuggerSession
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV2,
    CodeRevisionV2,
    EvaluationFrameV3,
)
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.evaluation.replay_io import load_replay
from marl_battlegrounds.evaluation.tdm_scenarios import (
    TDM_BETA_SCENARIO_IDS,
    build_tdm_qualification_seed_schedule,
    build_tdm_scenario_specification,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.tasks import load_tdm_scenario


@cache
def _scenario(identifier: int) -> DebuggerScenario:
    loaded = load_tdm_scenario(identifier)
    return DebuggerScenario(
        name=f"tdm-scenario-{identifier}",
        title=loaded.info.name,
        description=loaded.info.description,
        mode="interactive",
        build_scenario=lambda: (loaded.config, loaded.initial_state),
        frames=(),
        default_controlled_slot=0,
        scenario_specification=build_tdm_scenario_specification(
            identifier, build_tdm_qualification_seed_schedule()
        ),
    )


def _session(identifier: int, method: System | None = None) -> DebuggerSession:
    return control.create_session(
        _scenario(identifier),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="random_valid" if method is None else "system:custom",
        team_b_controller="scenario_5"
        if identifier in TDM_BETA_SCENARIO_IDS
        else "reactive_tdm",
        execution_information_mode="shared_obs",
        systems=None if method is None else (method, None),
    )


def _same_tree(first: object, second: object) -> None:
    assert jax.tree.structure(first) == jax.tree.structure(second)
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(a, b)


def _same_frame(first: EvaluationFrameV3, second: EvaluationFrameV3) -> None:
    assert first.frame_index == second.frame_index
    assert first.simulator_step_count == second.simulator_step_count
    assert first.snapshot == second.snapshot
    assert first.base_observation == second.base_observation
    assert first.action_mask == second.action_mask
    assert (
        first.shared_obs_information_availability_by_recipient_and_sensor_source
        == second.shared_obs_information_availability_by_recipient_and_sensor_source
    )


@pytest.mark.parametrize("identifier", range(1, 9))
def test_fixed_start_keeps_exact_content_roles_dead_slots_and_horizon(
    identifier: int,
) -> None:
    session = _session(identifier)
    loaded = load_tdm_scenario(identifier)
    specification = session.scenario.scenario_specification
    assert specification is not None
    assert specification.pressure_protocol is not None
    _same_tree(session.config, loaded.config)
    _same_tree(session.state, loaded.initial_state)
    assert session.numeric_episode_id == (identifier - 1) * 2 + 1
    assert (
        session.evaluation_context.seed_protocol
        == specification.seed_schedule.realized_seed_protocols[0]
    )
    assert session.evaluation_context.identity.layout == specification.layout
    metadata = {
        row.name: row.value for row in session.evaluation_context.aggregation_keys
    }
    assert (
        metadata["pressure_protocol_digest"]
        == specification.pressure_protocol.canonical_digest
    )
    assert session.evaluation_context.expected_horizon == loaded.info.horizon
    assert tuple(
        row.public_agent_id for row in session.evaluation_context.roster
    ) == tuple(row.public_agent_id for row in specification.roster_template)
    assert (
        tuple(
            row.evaluation_role
            if isinstance(row, AssignedPolicySlotV2)
            else "not_applicable"
            for row in session.evaluation_context.policy_assignments
        )
        == specification.role_template
    )
    assert all(row.configured_active for row in session.evaluation_context.roster)
    assert session.current_evaluation_frame.frame_index == 0
    assert session.current_evaluation_frame.simulator_step_count == int(
        loaded.initial_state.step_count
    )
    assert np.all(np.asarray(session.config.agent_profile.active_mask))
    np.testing.assert_array_equal(
        session.state.alive_mask, loaded.initial_state.alive_mask
    )


@pytest.mark.parametrize("identifier", range(1, 9))
def test_built_in_scenario_trajectory_matches_public_evaluator(
    identifier: int, tmp_path: Path
) -> None:
    session = _session(identifier)
    result = evaluate_episodes(
        "random",
        "tdm-beta" if identifier in TDM_BETA_SCENARIO_IDS else "tdm-alpha",
        [
            EpisodeSpec(
                session.numeric_episode_id,
                session.config,
                seed_id=0,
                initial_state=session.state,
            )
        ],
        seed=0,
        num_envs=1,
        metrics="none",
        chunk_size=1,
        replay_episodes=[session.numeric_episode_id],
        output_dir=tmp_path / "evaluation",
    )
    assert result.status == "complete"
    replay = load_replay(result.replay_paths[0]).replay
    _same_frame(
        session.current_evaluation_frame, cast(EvaluationFrameV3, replay.frames[0])
    )
    for transition, frame in zip(replay.transitions, replay.frames[1:], strict=True):
        session = control.submit_interactive(session)
        assert session.incoming_evaluation_view is not None
        assert session.incoming_evaluation_view.transition.facts == transition.facts
        _same_frame(session.current_evaluation_frame, cast(EvaluationFrameV3, frame))
    assert session.episode_sealed


def _key_sensitive_system(records: list[np.ndarray[Any, Any]]) -> System:
    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: jax.Array
    ) -> list[np.ndarray[Any, Any]]:
        memory = np.asarray(jax.random.key_data(keys)).copy()
        assert memory.shape == (len(inputs.valid), 2)
        records.append(memory.copy())
        return [row.copy() for row in memory]

    def choose(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: jax.Array
    ) -> SystemOutput:
        next_memory = np.asarray(memory) ^ np.asarray(jax.random.key_data(keys))
        moves = np.zeros(inputs.active_mask.shape, np.int32)
        for lane in range(moves.shape[0]):
            order = (np.arange(9) + int(next_memory[lane, 0] % 9)) % 9
            for actor in range(5):
                moves[lane, actor] = next(
                    int(move)
                    for move in order
                    if inputs.action_mask.move_mask[lane, actor, move]
                )
        zeros = np.zeros_like(moves)
        records.append(next_memory.copy())
        return SystemOutput(
            ActorAction(jnp.asarray(moves), jnp.asarray(zeros), jnp.asarray(zeros)),
            [row.copy() for row in next_memory],
        )

    return System(
        "Key-sensitive custom method", choose, init=initialize, execution="host"
    )


def test_custom_initializer_actions_and_memory_match_evaluator_and_restart(
    tmp_path: Path,
) -> None:
    live_records: list[np.ndarray[Any, Any]] = []
    eval_records: list[np.ndarray[Any, Any]] = []
    session = _session(6, _key_sensitive_system(live_records))
    initial = session
    result = evaluate_episodes(
        _key_sensitive_system(eval_records),
        "tdm-alpha",
        [
            EpisodeSpec(
                session.numeric_episode_id,
                session.config,
                seed_id=0,
                initial_state=session.state,
            )
        ],
        seed=0,
        num_envs=1,
        metrics="none",
        chunk_size=1,
        replay_episodes=[session.numeric_episode_id],
        output_dir=tmp_path / "evaluation",
    )
    replay = load_replay(result.replay_paths[0]).replay
    np.testing.assert_array_equal(live_records[0], eval_records[0])
    for index, (transition, frame) in enumerate(
        zip(replay.transitions, replay.frames[1:], strict=True), start=1
    ):
        session = control.submit_interactive(session)
        assert session.system_memory is not None
        assert session.incoming_evaluation_view is not None
        assert session.incoming_evaluation_view.transition.facts == transition.facts
        _same_frame(session.current_evaluation_frame, cast(EvaluationFrameV3, frame))
        np.testing.assert_array_equal(session.system_memory.team_a, eval_records[index])
        np.testing.assert_array_equal(live_records[index], eval_records[index])
    assert len(live_records) == len(eval_records) == len(replay.transitions) + 1
    restarted = control.reset_session(session)
    _same_tree(restarted.state, initial.state)
    np.testing.assert_array_equal(
        jax.random.key_data(restarted.key), jax.random.key_data(initial.key)
    )
    assert restarted.numeric_episode_id == initial.numeric_episode_id
    assert restarted.run_generation == initial.run_generation + 1
    assert restarted.system_memory is not initial.system_memory
    assert restarted.system_memory is not None and initial.system_memory is not None
    np.testing.assert_array_equal(
        restarted.system_memory.team_a, initial.system_memory.team_a
    )


def test_fixed_overrides_fail_before_start_callback_and_equal_setup_is_rejected() -> (
    None
):
    session = _session(6)
    calls: list[str] = []

    def unexpected() -> NoReturn:
        calls.append("called")
        raise AssertionError("A forbidden replacement must not build a start")

    replacement = replace(session.scenario, build_scenario=unexpected)
    with pytest.raises(control.CombatConfigurationRejectedError):
        control.switch_scenario(session, replacement)
    with pytest.raises(control.CombatConfigurationRejectedError):
        control.set_combat_configuration(
            session,
            team_a_controller=session.team_a_controller,
            team_b_controller=session.team_b_controller,
            execution_information_mode="shared_obs",
        )
    with pytest.raises(control.CombatConfigurationRejectedError):
        control.create_session(
            replacement,
            seed=0,
            evaluation_launch_specification=debugger_test_launch_specification(0),
            controlled_global_slot=None,
            show_ranges=False,
            verbose_logging=False,
            team_a_controller="random_valid",
            team_b_controller="manual",
            execution_information_mode="shared_obs",
        )
    assert calls == []


def test_fixed_continuation_binds_numeric_id_setup_context_and_local_step() -> None:
    session = _session(6)
    with pytest.raises(ValueError, match="raw continuation"):
        replace(session, numeric_episode_id=1)
    with pytest.raises(ValueError, match="raw continuation"):
        replace(session, scenario=replace(session.scenario, title="Other title"))
    with pytest.raises(ValueError, match="raw continuation"):
        replace(session, evaluation_context=session.evaluation_context.model_copy())
    with pytest.raises(ValueError, match="raw continuation"):
        replace(
            session,
            current_evaluation_frame=session.current_evaluation_frame.model_copy(
                update={"frame_index": 1}
            ),
        )


def test_v2_source_identity_survives_fixed_restart() -> None:
    source = CodeRevisionV2(package_version="9.8.7", source_tree_digest="a" * 64)
    session = control.create_session(
        _scenario(6),
        seed=0,
        evaluation_launch_specification=build_debugger_evaluation_launch_specification_v1(
            root_seed=0, code_revision=source
        ),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="manual",
        team_b_controller="reactive_tdm",
        execution_information_mode="shared_obs",
    )
    restarted = control.reset_session(session)
    assert restarted.evaluation_context.code_revision == source
    assert type(restarted.evaluation_context.code_revision) is CodeRevisionV2


def test_fixed_restart_and_turn_do_not_reload_packaged_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(6)
    from marl_battlegrounds import tasks
    from marl_battlegrounds.evaluation import tdm_scenarios

    def unexpected(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(
            "Already prepared scenario content must not be loaded again"
        )

    monkeypatch.setattr(tasks, "load_tdm_scenario", unexpected)
    monkeypatch.setattr(tdm_scenarios, "load_tdm_scenario", unexpected)
    restarted = control.reset_session(session)
    assert (
        control.submit_interactive(restarted).current_evaluation_frame.frame_index == 1
    )


@pytest.mark.parametrize("changed", ("config", "state"))
def test_fixed_content_mismatch_rejects_before_custom_initialization(
    changed: str,
) -> None:
    scenario = _scenario(6)
    config, state = scenario.build_scenario()
    if changed == "config":
        config = config._replace(
            spawn_shield_duration_steps=config.spawn_shield_duration_steps + 1
        )
    else:
        state = state._replace(
            team_deathmatch_scores=state.team_deathmatch_scores.at[0].add(1)
        )
    altered = replace(scenario, build_scenario=lambda: (config, state))
    records: list[np.ndarray[Any, Any]] = []
    with pytest.raises(ValueError, match="Fixed scenario"):
        control.create_session(
            altered,
            seed=0,
            evaluation_launch_specification=debugger_test_launch_specification(0),
            controlled_global_slot=None,
            show_ranges=False,
            verbose_logging=False,
            team_a_controller="system:custom",
            team_b_controller="reactive_tdm",
            execution_information_mode="shared_obs",
            systems=(_key_sensitive_system(records), None),
        )
    assert records == []
