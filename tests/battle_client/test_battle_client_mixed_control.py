"""Check mixed controllers through real turns, exact starts and saved replays.

Manual drafts stay outside method inputs and replace only owned physical rows.
Grouped host and JAX methods keep their permitted observations, keys and memory.
Dead slots remain configured, inactive slots remain empty, and saved actions match
what Core received. Fixed autonomous mixtures agree with the public evaluator.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.helpers.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client import control
from marl_battlegrounds._battle_client.evaluation_bridge import (
    DebuggerActionSourceKindV1,
    build_debugger_evaluation_launch_specification_v1,
)
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.recording import (
    build_debugger_recording_specification_v1,
    recording_action_source,
    recording_policy_execution_included,
)
from marl_battlegrounds._battle_client.replay_recorder import DebuggerReplayRecorder
from marl_battlegrounds._battle_client.setup import BattleSetup, prepare_battle_setup
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV2,
    EvaluationFrameV3,
    EvaluationTransitionV1,
)
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    system_inputs,
)
from marl_battlegrounds.evaluation.replay_io import (
    load_replay,
    preflight_replay_destination,
)
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import load_tdm_scenario


@pytest.fixture(scope="module")
def ordinary() -> BattleSetup:
    return prepare_battle_setup(map_id=0, max_steps=6)


def _session(setup: BattleSetup, *, capture: bool = False) -> DebuggerSession:
    resolved = resolve_controllers(setup)
    fixture = debugger_test_launch_specification(setup.seed)
    launch = build_debugger_evaluation_launch_specification_v1(
        root_seed=setup.seed,
        code_revision=fixture.code_revision,
        capture_profile="evaluation_metric_complete" if capture else "debug",
    )
    return control.create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=launch,
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


def _host(
    records: list[dict[str, Any]], name: str = "Host", *, parts: int = 1
) -> System:
    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: jax.Array
    ) -> list[int]:
        records.append(
            {"phase": "init", "keys": np.asarray(jax.random.key_data(keys)).copy()}
        )
        return [0 for _ in inputs.valid]

    def choose(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: jax.Array
    ) -> SystemOutput:
        assert variables == ()
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(inputs))
        offset = np.asarray(jax.random.key_data(keys))[:, 0] % 9
        priority = (np.arange(9)[None, :] + offset[:, None]) % 9 + 1
        moves = np.argmax(
            np.asarray(inputs.action_mask.move_mask) * priority[:, None, :], axis=-1
        ).astype(np.int32)
        zeros = np.zeros_like(moves)
        action = ActorAction(
            *(cast(jax.Array, value) for value in (moves, zeros, zeros))
        )
        records.append(
            {
                "phase": "apply",
                "inputs": jax.tree.map(np.copy, inputs),
                "keys": np.asarray(jax.random.key_data(keys)).copy(),
                "memory": list(memory),
                "memory_object": memory,
                "action": action,
            }
        )
        return SystemOutput(
            action,
            [
                int(count) + int(valid)
                for count, valid in zip(memory, inputs.valid, strict=True)
            ],
            policy_ids=cast(jax.Array, np.full(moves.shape, parts - 1, np.int32)),
        )

    return System(
        name,
        choose,
        init=initialize,
        execution="host",
        components=tuple({"name": f"{name} part {part}"} for part in range(parts)),
    )


def _jax_initialize(
    variables: PolicyTree, inputs: SystemInput, keys: jax.Array
) -> jax.Array:
    return jnp.zeros(inputs.valid.shape, jnp.int32)


def _jax_choose(
    variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: jax.Array
) -> SystemOutput:
    priority = (jnp.arange(9)[None, :] + memory[:, None]) % 9 + 1
    moves = jnp.argmax(inputs.action_mask.move_mask * priority[:, None, :], axis=-1)
    zeros = jnp.zeros_like(moves)
    return SystemOutput(
        ActorAction(moves, zeros, zeros),
        memory + inputs.valid.astype(jnp.int32),
        policy_ids=jnp.zeros(inputs.active_mask.shape, jnp.int32),
    )


def _jax(name: str = "JAX") -> System:
    return System(name, _jax_choose, init=_jax_initialize, components=({"name": name},))


def _same_tree(first: object, second: object) -> None:
    assert jax.tree.structure(first) == jax.tree.structure(second)
    for actual, expected in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def _draft(session: DebuggerSession, slot: int, move: int) -> DebuggerSession:
    return control.set_pending_movement(
        control.select_controlled_actor(session, slot), move
    )


def _apply_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["phase"] == "apply"]


def _assert_permitted(
    record: dict[str, Any], session: DebuggerSession, team: int
) -> None:
    wrapped = session.environment_state
    assert wrapped is not None
    expected = system_inputs(
        Observations(wrapped.observation, wrapped.source_availability),
        wrapped,
        team=team,
    )
    actual = cast(SystemInput, record["inputs"])
    _same_tree(actual.actors, expected.actors)
    _same_tree(actual.action_mask, expected.action_mask)
    _same_tree(actual.active_mask, expected.active_mask)
    _same_tree(actual.episode_start, expected.episode_start)
    _same_tree(actual.valid, expected.valid)


def test_drafts_stay_private_and_repeated_sources_keep_team_memory_and_keys(
    ordinary: BattleSetup, record_property: Callable[[str, object], None]
) -> None:
    records: list[dict[str, Any]] = []
    method = _host(records)
    compiled = _jax()
    session = _session(
        replace(
            ordinary,
            team_a=("manual", method, method, compiled, method),
            team_b=("manual", method, method, compiled, method),
        )
    )
    assert session.manual_slots == (0, 5)
    assert len(records) == 2
    assert not np.array_equal(records[0]["keys"], records[1]["keys"])
    first = _draft(_draft(session, 0, 1), 5, 2)
    first = control.submit_interactive(first)
    calls = _apply_records(records)
    assert len(calls) == 2
    _assert_permitted(calls[0], session, 0)
    _assert_permitted(calls[1], session, 1)
    np.testing.assert_array_equal(
        calls[0]["inputs"].controlled_mask, [[False, True, True, False, True]]
    )
    np.testing.assert_array_equal(
        calls[1]["inputs"].controlled_mask, [[False, True, True, False, True]]
    )
    assert calls[0]["memory"] == calls[1]["memory"] == [0]
    assert calls[0]["memory_object"] is not calls[1]["memory_object"]
    assert not np.array_equal(calls[0]["keys"], calls[1]["keys"])
    changed_drafts = control.submit_interactive(_draft(_draft(session, 0, 3), 5, 4))
    calls = _apply_records(records)
    assert len(calls) == 4
    for before, after in zip(calls[:2], calls[2:], strict=True):
        _same_tree(before["inputs"], after["inputs"])
        _same_tree(before["action"], after["action"])
        _same_tree(before["keys"], after["keys"])
    assert first.incoming_evaluation_view is not None
    assert changed_drafts.incoming_evaluation_view is not None
    before_facts = (
        first.incoming_evaluation_view.transition.facts.action_acceptance_facts
    )
    after_facts = (
        changed_drafts.incoming_evaluation_view.transition.facts.action_acceptance_facts
    )
    before = before_facts.submitted_joint_action
    after = after_facts.submitted_joint_action
    assert (before.move[0], before.move[5]) == (1, 2)
    assert (after.move[0], after.move[5]) == (3, 4)
    assert tuple(before.move[i] for i in range(10) if i not in (0, 5)) == tuple(
        after.move[i] for i in range(10) if i not in (0, 5)
    )
    successor = control.submit_interactive(first)
    calls = _apply_records(records)
    assert len(calls) == 6
    for record, team in zip(calls[4:], (0, 1), strict=True):
        _assert_permitted(record, first, team)
        assert record["memory"] == [1]
    assert successor.current_evaluation_frame.frame_index == 2
    assert successor.environment_state is not None
    assert int(successor.environment_state.cumulative_transition_count) == 2
    assert successor.system_memory is not None
    assert first.system_memory is not None
    assert session.system_memory is not None
    assert first.system_memory.team_a is not first.system_memory.team_b
    restarted = control.reset_session(successor)
    assert len([row for row in records if row["phase"] == "init"]) == 4
    assert restarted.systems is session.systems
    assert restarted.manual_slots == session.manual_slots
    assert restarted.slot_labels == session.slot_labels
    assert restarted.system_memory is not None
    _same_tree(restarted.system_memory.team_a, session.system_memory.team_a)
    _same_tree(restarted.system_memory.team_b, session.system_memory.team_b)
    record_property("host_apply_calls", len(calls))
    record_property("host_init_calls_including_restart", 4)
    record_property("manual_slots", session.manual_slots)


@pytest.mark.parametrize(
    "edit", ["movement", "basic", "ultimate", "target", "clear", "none"]
)
def test_policy_inspection_never_takes_manual_ownership_or_loses_drafts(
    ordinary: BattleSetup, edit: str
) -> None:
    method = _jax()
    initial = _session(
        replace(
            ordinary,
            team_a=("manual", method, method, method, method),
            team_b=("manual", method, method, method, method),
        )
    )
    session = _draft(_draft(initial, 0, 1), 5, 3)
    saved = session.pending_actions
    inspected = control.select_controlled_actor(session, 1)
    if edit == "movement":
        inspected = control.set_pending_movement(inspected, 4)
    elif edit == "basic":
        inspected = control.arm_basic(inspected)
    elif edit == "ultimate":
        inspected = control.arm_ultimate(inspected)
    elif edit == "target":
        inspected = control.select_clicked_target(inspected, 5)
    elif edit == "clear":
        inspected = control.clear_pending_target(inspected)
    else:
        inspected = control.select_no_combat(inspected)
    assert inspected.controlled_global_slot == 1
    assert inspected.pending_actions == saved
    assert inspected.manual_slots == (0, 5)
    restored = control.select_controlled_actor(inspected, 0)
    assert restored.pending_action.move_action == 1
    assert control.select_controlled_actor(restored, 5).pending_action.move_action == 3


def test_small_rosters_leave_inactive_rows_empty_and_merge_only_manual_rows() -> None:
    records: list[dict[str, Any]] = []
    method = _host(records)
    setup = prepare_battle_setup(
        map_id=0,
        team_a=("manual", method),
        team_b=(method, "manual", _jax()),
        team_a_roster=("warrior", "warrior"),
        team_b_roster=("mage", "priest", "hunter"),
        max_steps=4,
    )
    initial = _session(setup)
    assert initial.manual_slots == (0, 6)
    assert initial.slot_labels == (
        "Manual",
        "Host",
        None,
        None,
        None,
        "Host",
        "Manual",
        "JAX",
        None,
        None,
    )
    staged = _draft(_draft(initial, 0, 2), 6, 4)
    staged = control.select_controlled_actor(staged, 6)
    staged = control.arm_ultimate(control.select_clicked_target(staged, 0))
    session = control.submit_interactive(staged)
    assert session.incoming_evaluation_view is not None
    facts = session.incoming_evaluation_view.transition.facts.action_acceptance_facts
    assert facts.submitted_joint_action.move[0] == 2
    assert facts.submitted_joint_action.move[6] == 4
    assert facts.submitted_joint_action.select_target[6] == 6
    assert facts.submitted_joint_action.use_ultimate[6] == 1
    for head in ("move", "select_target", "use_ultimate"):
        assert (
            tuple(
                getattr(facts.submitted_joint_action, head)[i] for i in (2, 3, 4, 8, 9)
            )
            == (0,) * 5
        )
    calls = _apply_records(records)
    assert len(calls) == 2
    assert facts.submitted_joint_action.move[1] == int(calls[0]["action"].move[0, 1])
    assert facts.submitted_joint_action.move[5] == int(calls[1]["action"].move[0, 0])
    assert session.system_memory is not None
    trace = np.asarray(session.system_memory.policy_trace.policy_ids)[0]
    assert tuple(trace[i] for i in (0, 2, 3, 4, 6, 8, 9)) == (-1,) * 7
    assert all(trace[i] >= 0 for i in (1, 5, 7))


def test_dead_manual_slot_remains_owned_and_its_submitted_intent_is_recorded() -> None:
    loaded = load_tdm_scenario(6)
    dead = next(i for i in range(5) if not bool(loaded.initial_state.alive_mask[i]))
    method = _jax()
    controllers = tuple("manual" if i == dead else method for i in range(5))
    initial = _session(prepare_battle_setup(scenario_id=6, team_a=controllers))
    assert initial.manual_slots == (dead,)
    assert initial.slot_labels[dead] == "Manual"
    assert all(row.configured_active for row in initial.evaluation_context.roster)
    _same_tree(initial.state, loaded.initial_state)
    session = control.submit_interactive(_draft(initial, dead, 1))
    assert session.incoming_evaluation_view is not None
    facts = session.incoming_evaluation_view.transition.facts.action_acceptance_facts
    assert facts.submitted_joint_action.move[dead] == 1
    assert facts.accepted_joint_action.move[dead] == 0
    assert session.manual_slots == (dead,)
    assert session.system_memory is not None
    assert int(session.system_memory.policy_trace.policy_ids[0, dead]) == -1


def test_opaque_method_with_eight_components_executes_without_slot_guessing(
    ordinary: BattleSetup,
) -> None:
    records: list[dict[str, Any]] = []
    method = _host(records, "Opaque", parts=8)
    initial = _session(replace(ordinary, team_a=method, team_b="manual"))
    assert initial.systems is not None and initial.systems[0] is method
    session = control.submit_interactive(initial)
    assert len(_apply_records(records)) == 1
    assert session.system_memory is not None
    np.testing.assert_array_equal(
        session.system_memory.policy_trace.policy_ids, [[7] * 5 + [-1] * 5]
    )
    assert session.system_memory.team_a == [1]
    records.clear()
    compiled = _jax()
    mixed = _session(
        replace(
            ordinary,
            team_a=("manual", method, compiled, method, compiled),
            team_b=method,
        )
    )
    mixed = control.submit_interactive(mixed)
    assert len(_apply_records(records)) == 2
    assert mixed.systems is not None and mixed.system_memory is not None
    components = mixed.systems[0].components
    assert components is not None and len(components) > 5
    trace = np.asarray(mixed.system_memory.policy_trace.policy_ids)[0]
    assert trace[0] == -1
    for slot in (1, 3):
        assert components[int(trace[slot])]["name"] == "Opaque part 7"
    np.testing.assert_array_equal(trace[5:], [7] * 5)


def test_composed_exact_scenario_matches_public_evaluator(tmp_path: Path) -> None:
    live_records: list[dict[str, Any]] = []
    live_host = _host(live_records)
    live_jax = _jax()
    setup = prepare_battle_setup(
        scenario_id=6, team_a=(live_host, live_jax, live_host, live_jax, live_host)
    )
    initial = _session(setup)
    eval_records: list[dict[str, Any]] = []
    eval_host = _host(eval_records)
    eval_jax = _jax()
    resolved = resolve_controllers(
        replace(setup, team_a=(eval_host, eval_jax, eval_host, eval_jax, eval_host))
    )
    assert resolved.systems is not None and resolved.systems[0] is not None
    result = evaluate_episodes(
        resolved.systems[0],
        "tdm-alpha",
        [
            EpisodeSpec(
                initial.numeric_episode_id,
                initial.config,
                seed_id=0,
                initial_state=initial.state,
            )
        ],
        seed=0,
        num_envs=1,
        metrics="none",
        chunk_size=1,
        replay_episodes=[initial.numeric_episode_id],
        output_dir=tmp_path / "evaluation",
    )
    assert result.status == "complete"
    replay = load_replay(result.replay_paths[0]).replay
    _same_tree(live_records[0]["keys"], eval_records[0]["keys"])
    session = initial
    for transition, frame in zip(replay.transitions, replay.frames[1:], strict=True):
        session = control.submit_interactive(session)
        assert session.incoming_evaluation_view is not None
        assert session.incoming_evaluation_view.transition.facts == transition.facts
        actual = session.current_evaluation_frame
        expected = cast(EvaluationFrameV3, frame)
        assert actual.frame_index == expected.frame_index
        assert actual.simulator_step_count == expected.simulator_step_count
        assert actual.snapshot == expected.snapshot
        assert actual.base_observation == expected.base_observation
        assert actual.action_mask == expected.action_mask
        sources_name = (
            "shared_obs_information_availability_by_recipient_and_sensor_source"
        )
        assert getattr(actual, sources_name) == getattr(expected, sources_name)
    assert session.episode_sealed
    actual_calls, expected_calls = (
        _apply_records(live_records),
        _apply_records(eval_records),
    )
    assert len(actual_calls) == len(expected_calls) == len(replay.transitions)
    assert len(actual_calls) > 0
    for actual, expected in zip(actual_calls, expected_calls, strict=True):
        _same_tree(actual["keys"], expected["keys"])
        _same_tree(actual["inputs"], expected["inputs"])
        _same_tree(actual["action"], expected["action"])
        assert actual["memory"] == expected["memory"]


def test_saved_mixed_replay_keeps_manual_identity_and_actual_action_rows(
    ordinary: BattleSetup, tmp_path: Path
) -> None:
    records: list[dict[str, Any]] = []
    method = _host(records)
    setup = replace(
        ordinary, team_a=("manual", method, method, _jax(), method), team_b=method
    )
    session = _session(setup, capture=True)
    destination = tmp_path / "mixed.marlbg-replay.json"
    recorder = DebuggerReplayRecorder(
        specification=build_debugger_recording_specification_v1(
            action_source_kind=cast(
                DebuggerActionSourceKindV1,
                recording_action_source(session.evaluation_context),
            ),
            runtime_provenance=capture_runtime_provenance(
                "0.1.0",
                policy_execution_included=recording_policy_execution_included(
                    session.evaluation_context
                ),
            ),
        ),
        destination=preflight_replay_destination(destination),
        context=session.evaluation_context,
        initial_frame=session.current_evaluation_frame,
    )
    actual_frames = [session.current_evaluation_frame]
    actual_transitions: list[EvaluationTransitionV1] = []
    for move in (1, 3):
        session = control.submit_interactive(_draft(session, 0, move))
        assert session.incoming_evaluation_view is not None
        transition = session.incoming_evaluation_view.transition
        assert (
            transition.facts.action_acceptance_facts.submitted_joint_action.move[0]
            == move
        )
        recorder.append(transition, session.current_evaluation_frame)
        actual_frames.append(session.current_evaluation_frame)
        actual_transitions.append(transition)
    assert len(_apply_records(records)) == 4
    assert recorder.finalize_and_save("finish_and_review") == "saved"
    replay = load_replay(destination).replay
    assert replay.frames == tuple(actual_frames)
    assert replay.transitions == tuple(actual_transitions)
    assert replay.header.context == session.evaluation_context
    assert recording_action_source(replay.header.context) == "mixed"
    assignments = replay.header.context.policy_assignments
    assert isinstance(assignments[0], AssignedPolicySlotV2)
    assert assignments[0].policy_kind == "manual"
    assert all(
        isinstance(row, AssignedPolicySlotV2) and row.policy_kind == "system"
        for row in assignments[1:]
    )
    restarted = control.reset_session(session)
    assert restarted.current_evaluation_frame.frame_index == 0
    assert restarted.systems is session.systems
    assert restarted.manual_slots == (0,)
    assert restarted.system_memory is not None
    assert restarted.system_memory.team_b == [0]


def test_manual_claim_on_policy_row_fails_before_any_method_call(
    ordinary: BattleSetup,
) -> None:
    records: list[dict[str, Any]] = []
    method = _host(records)
    session = _session(
        replace(
            ordinary, team_a=("manual", method, method, method, method), team_b=method
        )
    )
    with pytest.raises(control.DebuggerTransitionFailureV1) as failure:
        control.submit_interactive(session, actor_global_slots=(1,))
    assert failure.value.stable_code == "interactive_action_build_failed"
    assert not _apply_records(records)
    assert session.current_evaluation_frame.frame_index == 0


def test_mixed_member_failure_keeps_accepted_memory_and_manual_draft(
    ordinary: BattleSetup,
) -> None:
    records: list[dict[str, Any]] = []
    method = _host(records)

    def fail(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: jax.Array
    ) -> SystemOutput:
        raise RuntimeError("Declared mixed member failure")

    failing = System("Failure", fail, execution="host")
    session = _session(
        replace(
            ordinary, team_a=("manual", method, failing, method, method), team_b=_jax()
        )
    )
    session = _draft(session, 0, 3)
    memory = session.system_memory
    assert memory is not None
    before = jax.tree.map(np.copy, memory.team_a)
    with pytest.raises(control.DebuggerTransitionFailureV1) as failure:
        control.submit_interactive(session)
    assert failure.value.stable_code == "policy_action_build_failed"
    assert len(_apply_records(records)) == 1
    assert session.system_memory is memory
    _same_tree(memory.team_a, before)
    assert session.pending_action.move_action == 3
    assert session.current_evaluation_frame.frame_index == 0
    assert session.environment_state is not None
    assert int(session.environment_state.cumulative_transition_count) == 0


def test_host_memory_survives_death_respawn_and_only_restart_clears_it() -> None:
    scripted = get_scenario("death_respawn_cycle")
    scenario = replace(scripted, mode="interactive", frames=())
    records: list[dict[str, Any]] = []
    original = _host(records)
    fixture = _session(BattleSetup(scenario, 0, "manual", original))
    tape = tuple(
        control.build_scripted_joint_action(fixture.evaluation_context, frame)
        for frame in scripted.frames
    )

    def choose(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: jax.Array
    ) -> SystemOutput:
        result = original.apply(variables, memory, inputs, keys)
        action = tape[int(memory[0])]
        return replace(
            result, actions=ActorAction(*(head[None, 5:10] for head in action))
        )

    method = replace(original, apply=choose)
    session = _session(BattleSetup(scenario, 0, "manual", method))
    records.clear()
    alive = [bool(session.state.alive_mask[5])]
    for turn, frame in enumerate(scripted.frames):
        for command in frame.commands:
            if command.actor_global_slot >= 5:
                continue
            session = _draft(session, command.actor_global_slot, command.move_action)
            session = control.select_clicked_target(
                session, cast(int, command.target_global_slot)
            )
            session = control.arm_basic(session)
        session = control.submit_interactive(session)
        assert session.system_memory is not None
        assert session.system_memory.team_b == [turn + 1]
        alive.append(bool(session.state.alive_mask[5]))
    assert alive == [True, False, False, True, True, True, True, True]
    calls = _apply_records(records)
    assert len(calls) == len(scripted.frames)
    assert [record["memory"] for record in calls] == [
        [turn] for turn in range(len(scripted.frames))
    ]
    assert [record["inputs"].episode_start.tolist() for record in calls] == [[True]] + [
        [False]
    ] * (len(scripted.frames) - 1)
    assert not any(record["phase"] == "init" for record in records)
    restarted = control.reset_session(session)
    assert len([record for record in records if record["phase"] == "init"]) == 1
    assert restarted.system_memory is not None
    assert restarted.system_memory.team_b == [0]
    assert bool(restarted.state.alive_mask[5])


@pytest.mark.parametrize("manual", [False, True])
def test_ten_distinct_controllers_keep_each_physical_assignment(
    ordinary: BattleSetup, manual: bool
) -> None:
    records: list[list[dict[str, Any]]] = [[] for _ in range(10)]
    methods = tuple(
        _host(records[slot], f"Controller {slot}")
        if slot % 2 == 0
        else _jax(f"Controller {slot}")
        for slot in range(10)
    )
    sources = ("manual", *methods[1:]) if manual else methods
    initial = _session(replace(ordinary, team_a=sources[:5], team_b=sources[5:]))
    assert initial.slot_labels == tuple(
        "Manual" if manual and slot == 0 else f"Controller {slot}" for slot in range(10)
    )
    session = control.submit_interactive(_draft(initial, 0, 2) if manual else initial)
    assert session.incoming_evaluation_view is not None
    assert session.system_memory is not None and session.systems is not None
    facts = session.incoming_evaluation_view.transition.facts.action_acceptance_facts
    submitted = facts.submitted_joint_action
    trace = np.asarray(session.system_memory.policy_trace.policy_ids)[0]
    for slot in range(10):
        if manual and slot == 0:
            assert submitted.move[slot] == 2
            assert trace[slot] == -1
            assert records[slot] == []
            continue
        local = slot % 5
        team = slot // 5
        components = session.systems[team].components
        assert components is not None
        component = components[int(trace[slot])]
        expected_name = f"Controller {slot}" + (" part 0" if slot % 2 == 0 else "")
        assert component["name"] == expected_name
        if slot % 2 == 0:
            calls = _apply_records(records[slot])
            assert len(calls) == 1
            _assert_permitted(calls[0], initial, team)
            np.testing.assert_array_equal(
                calls[0]["inputs"].controlled_mask,
                [[index == local for index in range(5)]],
            )
            assert submitted.move[slot] == int(calls[0]["action"].move[0, local])
        else:
            wrapped = initial.environment_state
            assert wrapped is not None
            inputs = system_inputs(
                Observations(wrapped.observation, wrapped.source_availability),
                wrapped,
                team=team,
            )
            expected = _jax_choose(
                (), jnp.zeros(1, jnp.int32), inputs, jax.random.key(0)
            )
            assert submitted.move[slot] == int(expected.actions.move[0, local])


@pytest.mark.parametrize("simple", [True, False])
def test_tutorial_group_shapes_call_each_source_once_per_team(
    ordinary: BattleSetup, simple: bool
) -> None:
    records: list[list[dict[str, Any]]] = [[] for _ in range(4)]
    first, second, third, fourth = tuple(
        _host(record, f"Method {index}") for index, record in enumerate(records)
    )
    sources_a = (
        ("manual", first, first, first, first)
        if simple
        else (first, first, first, second, second)
    )
    sources_b = (third,) * 5 if simple else (third, third, third, fourth, fourth)
    session = _session(replace(ordinary, team_a=sources_a, team_b=sources_b))
    session = control.submit_interactive(_draft(session, 0, 2) if simple else session)
    assert session.current_evaluation_frame.frame_index == 1
    assert [len(_apply_records(record)) for record in records] == (
        [1, 0, 1, 0] if simple else [1, 1, 1, 1]
    )
    expected_masks = (
        ((False, True, True, True, True), None, (True,) * 5, None)
        if simple
        else (
            (True, True, True, False, False),
            (False, False, False, True, True),
            (True, True, True, False, False),
            (False, False, False, True, True),
        )
    )
    for record, expected in zip(records, expected_masks, strict=True):
        if expected is not None:
            np.testing.assert_array_equal(
                _apply_records(record)[0]["inputs"].controlled_mask, [expected]
            )
