"""Check whose eyes the BattleClient's Agent POV shows: the recipient rule.

control.agent_pov_recipient_slot keeps the Agent POV recipient apart from the
focused champion. With Manual champions it is the last Manual champion
focused, so Tab, clicks and roster picks onto a policy champion (a visible
enemy included) never change the viewing team; with Manual champions on both
teams the view follows the one focused last; in Policies Only it follows the
focus. In Agent POV, Tab and Shift+Tab skip champions the viewing team cannot
see (Manual champions and teammates are always reachable), a roster pick of an
unseen enemy is refused without naming where it is, clicks hit only seen
bodies, and orders go to the recipient. Every Agent POV frame describes the
recipient; Oracle View keeps describing the focus, and Oracle -> Agent POV ->
Oracle returns the same Oracle frame. Open Replay opens the Replay Viewer in
Oracle View with the recipient as its Agent POV choice.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from tests.helpers.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client import input as input_module
from marl_battlegrounds._battle_client.control import (
    agent_pov_recipient_slot,
    create_session,
)
from marl_battlegrounds._battle_client.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
    debugger_action_source_kind_v1,
)
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.protocol import (
    BattlefieldPointerCommandV1,
    CommandRequestV1,
    CommandResponseV2,
    FinishAndReviewCommandV1,
    KeyboardCommandV1,
    ResearcherLiveDebuggerFrameV2,
    ReviewReplayCommandV1,
    RosterSelectionCommandV1,
    SetViewCommandV1,
    SharedObsAgentPovLiveDebuggerFrameV2,
    ViewMode,
)
from marl_battlegrounds._battle_client.recording import (
    build_debugger_recording_specification_v1,
)
from marl_battlegrounds._battle_client.replay_recorder import DebuggerReplayRecorder
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds.evaluation._replays.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_io import preflight_replay_destination
from marl_battlegrounds.rendering.vocabulary import class_token_from_id

# Fixed scenario 1 at frame zero: Team A (slots 0-4) sees Team B's 5, 8 and 9;
# 6 and 7 are out of sight. Team B is the built-in ALPHA policy.
_SEEN_ENEMIES = (5, 8, 9)
_UNSEEN_ENEMIES = (6, 7)


def _scenario_session(
    *, team_a: str = "manual", launch: object | None = None
) -> DebuggerSession:
    setup = prepare_battle_setup(scenario_id=1, team_a=team_a)  # pyright: ignore[reportArgumentType]
    resolved = resolve_controllers(setup)
    return create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=launch  # pyright: ignore[reportArgumentType]
        or debugger_test_launch_specification(setup.seed),
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


def _hot_seat_session() -> DebuggerSession:
    setup = prepare_battle_setup(map_id=0)
    resolved = resolve_controllers(setup)
    return create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller=resolved.controllers[0],
        team_b_controller=resolved.controllers[1],
        execution_information_mode="shared_obs",
        manual_slots=resolved.manual_slots,
        slot_labels=resolved.slot_labels,
    )


def _service(session: DebuggerSession, view_mode: ViewMode = "pov") -> DebuggerService:
    return DebuggerService(
        session,
        view_mode=view_mode,
        preset="analysis",
        include_stress=False,
        session_id="view-recipient-test",
    )


def _send(service: DebuggerService, command: object, name: str) -> CommandResponseV2:
    result = service.apply_command(
        CommandRequestV1(
            client_id="recipient-client",
            command_id=name,
            base_revision=service.revision,
            command=command,  # pyright: ignore[reportArgumentType]
        )
    )
    assert isinstance(result.payload, CommandResponseV2), result.payload
    return result.payload


def _frame_recipient(service: DebuggerService) -> str:
    frame = service.current_frame()
    assert isinstance(frame, SharedObsAgentPovLiveDebuggerFrameV2)
    return frame.recipient_public_agent_id


def _public_id(service: DebuggerService, slot: int) -> str:
    return service.session.evaluation_context.roster[slot].public_agent_id


def _stand_in(focus: int, manual: tuple[int, ...]) -> DebuggerSession:
    roster = tuple(
        SimpleNamespace(global_slot=slot, configured_active=True) for slot in range(10)
    )
    return cast(
        DebuggerSession,
        SimpleNamespace(
            controlled_global_slot=focus,
            manual_slots=manual,
            evaluation_context=SimpleNamespace(roster=roster),
        ),
    )


@pytest.mark.parametrize(
    ("focus", "manual", "previous", "expected"),
    (
        pytest.param(2, (0, 1, 2, 3, 4), 0, 2, id="manual-focus-moves-it"),
        pytest.param(6, (0, 1, 2, 3, 4), 3, 3, id="enemy-policy-keeps-it"),
        pytest.param(6, (0, 1, 2, 3, 4), None, 0, id="launch-policy-focus-lowest"),
        pytest.param(1, (2, 3), 3, 3, id="ally-policy-keeps-it"),
        pytest.param(5, (), 0, 5, id="policies-only-follows-focus"),
        pytest.param(7, tuple(range(10)), 1, 7, id="hot-seat-follows-manual"),
        pytest.param(6, (1, 2), 9, 1, id="stale-previous-falls-back"),
    ),
)
def test_recipient_rule(
    focus: int, manual: tuple[int, ...], previous: int | None, expected: int
) -> None:
    session = _stand_in(focus, manual)
    assert agent_pov_recipient_slot(session, previous) == expected
    # Recomputing with its own result never drifts.
    assert agent_pov_recipient_slot(session, expected) == expected


def test_tab_skips_unseen_and_a_visible_enemy_never_moves_the_view() -> None:
    service = _service(_scenario_session())
    try:
        assert service.session.controlled_global_slot == 0
        assert service.pov_global_slot == 0
        visited: list[tuple[int, int, str]] = []
        for index in range(8):
            _send(service, KeyboardCommandV1(key="Tab"), f"tab-{index}")
            visited.append(
                (
                    service.session.controlled_global_slot,
                    service.pov_global_slot,
                    _frame_recipient(service),
                )
            )
        assert [focus for focus, _, _ in visited] == [1, 2, 3, 4, 5, 8, 9, 0]
        for focus, recipient, frame_recipient in visited:
            assert focus not in _UNSEEN_ENEMIES
            # Enemies are policies: the view stays with the last Manual champion.
            expected = 4 if focus in _SEEN_ENEMIES else focus
            assert recipient == expected
            assert frame_recipient == _public_id(service, expected)
        back = _send(service, KeyboardCommandV1(key="Tab", shift_key=True), "back")
        assert service.session.controlled_global_slot == 9
        assert service.pov_global_slot == 0
        assert back.notice is not None and "played by a policy" in back.notice
        assert _frame_recipient(service) == _public_id(service, 0)
    finally:
        service.close()


def test_oracle_tab_visits_everyone_without_moving_the_view() -> None:
    service = _service(_scenario_session(), view_mode="researcher")
    try:
        _send(service, RosterSelectionCommandV1(role="control", global_slot=4), "m4")
        steps: list[tuple[int, int]] = []
        for index in range(6):
            _send(service, KeyboardCommandV1(key="Tab"), f"oracle-tab-{index}")
            steps.append(
                (service.session.controlled_global_slot, service.pov_global_slot)
            )
        # Oracle View may focus every enemy; only a Manual focus moves the view.
        assert steps == [(5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (0, 0)]
    finally:
        service.close()


def test_roster_picks_and_clicks_respect_sight_and_never_move_the_view() -> None:
    service = _service(_scenario_session())
    try:
        before = service.current_frame()
        refused = _send(
            service, RosterSelectionCommandV1(role="control", global_slot=6), "pick-6"
        )
        assert service.session.controlled_global_slot == 0
        class_name = class_token_from_id(
            service.session.evaluation_context.roster[6].class_id
        ).label
        assert refused.notice == (
            f"{class_name} · ID 6 is out of sight. "
            "Agent POV hides where it is and how it fares."
        )
        assert service.current_frame() == before

        _send(
            service, RosterSelectionCommandV1(role="control", global_slot=5), "pick-5"
        )
        assert service.session.controlled_global_slot == 5
        assert service.pov_global_slot == 0
        assert _frame_recipient(service) == _public_id(service, 0)

        positions = service.session.current_evaluation_frame.snapshot.agent_positions
        x, y = positions[7]
        _send(
            service,
            BattlefieldPointerCommandV1(world_x=x, world_y=y, button="primary"),
            "click-unseen-7",
        )
        assert service.session.controlled_global_slot == 5
        x, y = positions[8]
        _send(
            service,
            BattlefieldPointerCommandV1(world_x=x, world_y=y, button="primary"),
            "click-seen-8",
        )
        assert service.session.controlled_global_slot == 8
        assert service.pov_global_slot == 0
    finally:
        service.close()


def test_orders_in_agent_pov_go_to_the_recipient() -> None:
    service = _service(_scenario_session())
    try:
        _send(service, RosterSelectionCommandV1(role="control", global_slot=2), "m2")
        _send(service, RosterSelectionCommandV1(role="control", global_slot=9), "e9")
        assert service.session.controlled_global_slot == 9
        assert service.pov_global_slot == 2
        mask = service.session.current_evaluation_frame.action_mask.move_mask[2]
        key, move = next(
            (key, move)
            for key, move in input_module._MOVEMENT_KEYS.items()
            if move != 0 and bool(mask[move])
        )
        enemy_before = service.session.pending_actions[9]
        _send(service, KeyboardCommandV1(key=key), "order")
        assert service.session.controlled_global_slot == 2
        assert service.pov_global_slot == 2
        assert service.session.pending_actions[2].move_action == move
        assert service.session.pending_actions[9] == enemy_before
    finally:
        service.close()


def test_policies_only_view_follows_the_focus() -> None:
    service = _service(_scenario_session(team_a="random"))
    try:
        assert service.session.manual_slots == ()
        for index in range(5):
            _send(service, KeyboardCommandV1(key="Tab"), f"watch-{index}")
            assert service.pov_global_slot == service.session.controlled_global_slot
        # Tab reached the first seen enemy, so the view is now Team B's.
        assert service.session.controlled_global_slot == 5
        assert _frame_recipient(service) == _public_id(service, 5)
    finally:
        service.close()


def test_manual_champions_on_both_teams_view_follows_the_last_focused() -> None:
    service = _service(_hot_seat_session())
    try:
        assert service.session.manual_slots == tuple(range(10))
        _send(service, RosterSelectionCommandV1(role="control", global_slot=1), "a1")
        assert service.pov_global_slot == 1
        assert _frame_recipient(service) == _public_id(service, 1)
        # A Manual enemy is always reachable, seen or not, and takes the view.
        _send(service, RosterSelectionCommandV1(role="control", global_slot=7), "b7")
        assert service.session.controlled_global_slot == 7
        assert service.pov_global_slot == 7
        assert _frame_recipient(service) == _public_id(service, 7)
        _send(service, KeyboardCommandV1(key="Tab"), "b8")
        assert service.pov_global_slot == 8
    finally:
        service.close()


def test_oracle_to_agent_pov_to_oracle_returns_the_same_oracle_frame() -> None:
    service = _service(_scenario_session(), view_mode="researcher")
    try:
        _send(service, RosterSelectionCommandV1(role="control", global_slot=8), "e8")
        oracle = service.current_frame()
        _send(service, SetViewCommandV1(view_mode="pov"), "to-pov")
        assert _frame_recipient(service) == _public_id(service, 0)
        pov_presentation = service.current_presentation().payload
        assert pov_presentation.authority.authority_kind == "agent_pov"  # pyright: ignore[reportAttributeAccessIssue]
        _send(service, SetViewCommandV1(view_mode="researcher"), "to-oracle")
        again = service.current_frame()
        assert isinstance(again, ResearcherLiveDebuggerFrameV2)
        assert isinstance(oracle, ResearcherLiveDebuggerFrameV2)
        assert again.projection == oracle.projection
        # Only the service revision (also echoed in the diagnostics) moved on.
        assert (
            again.model_copy(update={"revision": oracle.revision, "hud": oracle.hud})
            == oracle
        )
        assert (
            again.hud.model_copy(update={"diagnostics": oracle.hud.diagnostics})
            == oracle.hud
        )
        assert [
            fact for fact in again.hud.diagnostics if fact.fact_id != "revision"
        ] == [fact for fact in oracle.hud.diagnostics if fact.fact_id != "revision"]
        assert service.session.controlled_global_slot == 8
    finally:
        service.close()


def test_open_replay_opens_the_replay_viewer_in_oracle_view(tmp_path: Path) -> None:
    debug_launch = debugger_test_launch_specification()
    launch = build_debugger_evaluation_launch_specification_v1(
        root_seed=debug_launch.root_seed,
        code_revision=debug_launch.code_revision,
        capture_profile="evaluation_metric_complete",
    )
    session = _scenario_session(launch=launch)
    recorder = DebuggerReplayRecorder(
        specification=build_debugger_recording_specification_v1(
            action_source_kind=debugger_action_source_kind_v1(
                session.scenario,
                session.team_a_controller,
                session.team_b_controller,
            ),
            runtime_provenance=RuntimeProvenanceV1(
                python_version="3.14.0",
                package_version="0.1.0",
                jax_version="0.7.0",
                jaxlib_version="0.7.0",
                numpy_version="2.3.0",
                pydantic_version="2.11.0",
                platform="linux",
                machine="x86_64",
                backend="cpu",
                device="test-cpu",
                precision="float32",
                environment_count=1,
                batch_shape=(1,),
                policy_execution_included=True,
            ),
        ),
        destination=preflight_replay_destination(
            tmp_path / "recipient.marlbg-replay.json"
        ),
        context=session.evaluation_context,
        initial_frame=session.current_evaluation_frame,
        scenario_name=session.scenario.name,
    )
    service = DebuggerService(
        session,
        view_mode="pov",
        preset="analysis",
        include_stress=False,
        session_id="open-replay-test",
        recorder=recorder,
    )
    try:
        _send(service, RosterSelectionCommandV1(role="control", global_slot=3), "m3")
        _send(service, RosterSelectionCommandV1(role="control", global_slot=8), "e8")
        _send(service, FinishAndReviewCommandV1(), "finish")
        result = service.apply_command(
            CommandRequestV1(
                client_id="recipient-client",
                command_id="open-replay",
                base_revision=service.revision,
                command=ReviewReplayCommandV1(),
            )
        )
        viewer = result.replay_handoff
        assert viewer is not None
        assert viewer.current_frame().view_mode == "researcher"
        presentation = viewer.current_presentation().payload
        assert getattr(presentation, "presentation_kind", None) == "replay_oracle"
        assert viewer._pov_global_slot == 3
        assert viewer._reference_global_slot == 8
        assert service.current_frame().view_mode == "pov"
    finally:
        service.close()
