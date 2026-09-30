"""Check physical ownership validation and trusted setup replacement boundaries.

Normalized Manual rows stay distinct from focus, dead state and padding. Session
continuations reject ownership changes inside a match. Loaded setup replacement
uses the existing constructor and exact fixed-scenario checks before callbacks.
"""

from dataclasses import replace
from typing import cast

import jax
import numpy as np
import pytest
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client import control
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import (
    DebuggerSession,
    normalize_slot_ownership,
)
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.policies.actor import ActorAction


def test_normalization_keeps_physical_holes_and_preserves_normalized_tuples() -> None:
    labels = ("Manual", None, "Planner", None, None, "Manual", None, None, None, None)
    manual, result = normalize_slot_ownership(
        active_slots=(0, 2, 5),
        controllers=("system:first", "manual"),
        manual_slots=(5, 0),
        slot_labels=labels,
    )
    assert manual == (0, 5)
    assert result is labels
    repeated, repeated_labels = normalize_slot_ownership(
        active_slots=(0, 2, 5),
        controllers=("system:first", "manual"),
        manual_slots=manual,
        slot_labels=labels,
    )
    assert repeated is manual and repeated_labels is labels


@pytest.mark.parametrize(
    "manual,labels",
    [
        ((0, 0), None),
        ((True,), None),
        ((1,), None),
        ((), ("Method",) * 9),
        ((), ("Method",) * 10),
        ((), ("", None, "Method", None, None, "Method", None, None, None, None)),
        (
            (0,),
            ("Method", None, "Method", None, None, "Method", None, None, None, None),
        ),
    ],
)
def test_invalid_ownership_is_rejected(
    manual: tuple[int, ...], labels: tuple[str | None, ...] | None
) -> None:
    with pytest.raises(ValueError):
        normalize_slot_ownership(
            active_slots=(0, 2, 5),
            controllers=("system:first", "system:second"),
            manual_slots=manual,
            slot_labels=labels,
        )


def test_scripted_defaults_do_not_claim_human_action_ownership() -> None:
    manual, labels = normalize_slot_ownership(
        active_slots=(0, 5),
        controllers=("manual", "manual"),
        manual_slots=None,
        slot_labels=None,
        scripted=True,
    )
    assert manual == ()
    assert labels == (
        "Scripted",
        None,
        None,
        None,
        None,
        "Scripted",
        None,
        None,
        None,
        None,
    )


@pytest.fixture(scope="module")
def initial() -> DebuggerSession:
    setup = prepare_battle_setup(
        map_id=0,
        seed=9,
        team_a_roster=("hunter",),
        team_b_roster=("priest",),
    )
    return control.create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=debugger_test_launch_specification(setup.seed),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        execution_information_mode="shared_obs",
    )


@pytest.mark.parametrize("field", ["manual_slots", "slot_labels"])
def test_session_rejects_rebound_ownership_tuple(
    initial: DebuggerSession, field: str
) -> None:
    current = getattr(initial, field)
    copied = tuple(list(current))
    assert copied == current and copied is not current
    with pytest.raises(ValueError, match="continuation identity"):
        replace(initial, **{field: copied})


def test_trusted_replacement_uses_fixed_root_and_new_generation(
    initial: DebuggerSession,
) -> None:
    setup = prepare_battle_setup(scenario_id=1)
    resolved = resolve_controllers(setup)
    replaced = control.replace_loaded_setup(
        initial,
        scenario=setup.scenario,
        seed=setup.seed,
        team_a_controller=resolved.controllers[0],
        team_b_controller=resolved.controllers[1],
        systems=resolved.systems,
        manual_slots=resolved.manual_slots,
        slot_labels=resolved.slot_labels,
    )
    assert initial.seed == 9 and initial.run_generation == 0
    assert replaced.seed == 0 and replaced.run_generation == 1
    assert (
        replaced.evaluation_context.code_revision
        == initial.evaluation_context.code_revision
    )
    assert (
        replaced.evaluation_context.capture_profile
        == initial.evaluation_context.capture_profile
    )
    assert replaced.evaluation_context.seed_protocol.root_seed == 0
    assert replaced.current_evaluation_frame.frame_index == 0
    assert replaced.scenario is setup.scenario
    assert replaced.team_b_controller == setup.team_b
    restarted = control.reset_session(replaced)
    assert restarted.manual_slots is replaced.manual_slots
    assert restarted.slot_labels is replaced.slot_labels
    assert restarted.run_generation == 2


@pytest.mark.parametrize("override", ["system", "manual"])
def test_fixed_opponent_overrides_fail_before_snapshot_or_init(
    initial: DebuggerSession, monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    setup = prepare_battle_setup(scenario_id=1)
    resolved = resolve_controllers(setup)

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("Rejected fixed setup must not call this boundary")

    monkeypatch.setattr(control, "_fresh_snapshot", unexpected)
    with pytest.raises(ValueError, match="Scenario opponents are fixed"):
        control.replace_loaded_setup(
            initial,
            scenario=setup.scenario,
            seed=setup.seed,
            team_a_controller=resolved.controllers[0],
            team_b_controller=resolved.controllers[1],
            systems=(None, cast(System, object()))
            if override == "system"
            else resolved.systems,
            manual_slots=(*resolved.manual_slots, 5)
            if override == "manual"
            else resolved.manual_slots,
            slot_labels=resolved.slot_labels,
        )


def _opaque_manual_setup() -> tuple[DebuggerSession, list[str]]:
    calls: list[str] = []

    def choose(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, key: jax.Array
    ) -> SystemOutput:
        calls.append("apply")
        move = np.ones(inputs.active_mask.shape, np.int32)
        zero = np.zeros_like(move)
        return SystemOutput(
            ActorAction(*(cast(jax.Array, value) for value in (move, zero, zero))),
            memory,
        )

    method = System("Opaque proposal", choose, execution="host")
    setup = prepare_battle_setup(
        map_id=0, team_a_roster=("hunter",) * 3, team_b_roster=("priest",)
    )
    session = control.create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=debugger_test_launch_specification(setup.seed),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="system:opaque",
        team_b_controller="manual",
        execution_information_mode="shared_obs",
        systems=(method, None),
        borrow_systems=True,
        manual_slots=(0, 1, 5),
        slot_labels=(
            "Manual",
            "Manual",
            "Opaque proposal",
            None,
            None,
            "Manual",
            None,
            None,
            None,
            None,
        ),
    )
    for slot, move in ((0, 3), (1, 7), (5, 2)):
        session = control.set_pending_movement(
            control.select_controlled_actor(session, slot), move
        )
    return session, calls


def test_partial_manual_submission_neutralizes_other_owned_rows() -> None:
    session, calls = _opaque_manual_setup()
    result = control.submit_interactive(session, actor_global_slots=(0,))
    assert calls == ["apply"]
    assert result.incoming_evaluation_view is not None
    acceptance = (
        result.incoming_evaluation_view.transition.facts.action_acceptance_facts
    )
    requested = acceptance.submitted_joint_action
    assert requested.move[0] == 3
    assert requested.move[1] == requested.move[5] == 0
    assert requested.move[2] == 1
    assert result.system_memory is not None
    np.testing.assert_array_equal(
        result.system_memory.policy_trace.policy_ids[:, (0, 1, 5)], -1
    )


@pytest.mark.parametrize("slots", [(2,), ()])
def test_policy_or_empty_subset_rejects_before_opaque_call(
    slots: tuple[int, ...],
) -> None:
    session, calls = _opaque_manual_setup()
    with pytest.raises(control.DebuggerTransitionFailureV1) as error:
        control.submit_interactive(session, actor_global_slots=slots)
    assert error.value.stable_code == "interactive_action_build_failed"
    assert calls == []
    assert session.current_evaluation_frame.frame_index == 0
