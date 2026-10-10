"""Check the Agent POV focus fact that lets the BattleClient show Watch Only.

An Agent POV live frame carries ``focused_public_agent_id`` only while the
player looks at a champion other than the POV recipient (a policy's champion
or an enemy the viewing team sees now); it names that champion's public ID
and leaves the recipient, the Agent POV view and the orders unchanged. When
the focus is the recipient the field is absent from the frame and from its
JSON. Oracle View frames never carry it, and a refused focus on an unseen
enemy never names that enemy. A match where you play one champion and
policies play the scenario's default one starts focused on your champion
(control._default_focus_slot), so it does not open in Watch Only.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from tests.helpers.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client.control import create_session
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.protocol import (
    CommandRequestV1,
    CommandResponseV2,
    RosterSelectionCommandV1,
    SharedObsAgentPovLiveDebuggerFrameV2,
    ViewMode,
)
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import BattleSetup, prepare_battle_setup

# Fixed scenario 1 at frame zero: Team A (slots 0-4, Manual) sees Team B's 5,
# 8 and 9; 6 and 7 are out of sight. Team B is the built-in ALPHA policy.
_SEEN_ENEMY = 5
_UNSEEN_ENEMY = 6


def _session(setup: BattleSetup) -> DebuggerSession:
    resolved = resolve_controllers(setup)
    session: DebuggerSession = create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=debugger_test_launch_specification(setup.seed),
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
    return session


def _service(view_mode: ViewMode) -> DebuggerService:
    session = _session(prepare_battle_setup(scenario_id=1, team_a="manual"))
    return DebuggerService(
        session,
        view_mode=view_mode,
        preset="analysis",
        include_stress=False,
        session_id="pov-focus-test",
    )


def _focus(service: DebuggerService, slot: int, name: str) -> CommandResponseV2:
    result = service.apply_command(
        CommandRequestV1(
            client_id="pov-focus-client",
            command_id=name,
            base_revision=service.revision,
            command=RosterSelectionCommandV1(role="control", global_slot=slot),
        )
    )
    assert isinstance(result.payload, CommandResponseV2), result.payload
    return result.payload


def _public_id(service: DebuggerService, slot: int) -> str:
    return service.session.evaluation_context.roster[slot].public_agent_id


def test_pov_frame_names_a_seen_focus_only_while_it_is_not_the_recipient() -> None:
    service = _service("pov")
    try:
        frame = service.current_frame()
        assert isinstance(frame, SharedObsAgentPovLiveDebuggerFrameV2)
        assert frame.focused_public_agent_id is None
        assert "focused_public_agent_id" not in frame.model_dump(mode="json")

        _focus(service, _SEEN_ENEMY, "seen-enemy")
        frame = service.current_frame()
        assert isinstance(frame, SharedObsAgentPovLiveDebuggerFrameV2)
        assert service.session.controlled_global_slot == _SEEN_ENEMY
        assert frame.recipient_public_agent_id == _public_id(service, 0)
        assert frame.focused_public_agent_id == _public_id(service, _SEEN_ENEMY)
        assert frame.model_dump(mode="json")["focused_public_agent_id"] == (
            _public_id(service, _SEEN_ENEMY)
        )
        # The page's presentation for the same frame still builds.
        assert service.current_presentation() is not None

        _focus(service, 0, "own")
        frame = service.current_frame()
        assert isinstance(frame, SharedObsAgentPovLiveDebuggerFrameV2)
        assert frame.focused_public_agent_id is None
        assert "focused_public_agent_id" not in frame.model_dump(mode="json")
    finally:
        service.close()


def test_unseen_focus_is_never_named_and_oracle_frames_never_carry_it() -> None:
    service = _service("pov")
    try:
        refused = _focus(service, _UNSEEN_ENEMY, "unseen-enemy")
        assert refused.notice is not None and "out of sight" in refused.notice
        frame = service.current_frame()
        assert frame.focused_public_agent_id is None
        assert _public_id(service, _UNSEEN_ENEMY) not in frame.model_dump_json()
    finally:
        service.close()
    oracle = _service("researcher")
    try:
        _focus(oracle, _SEEN_ENEMY, "oracle-enemy")
        assert oracle.session.controlled_global_slot == _SEEN_ENEMY
        assert oracle.current_frame().focused_public_agent_id is None
    finally:
        oracle.close()


def test_a_match_with_one_manual_champion_starts_focused_on_it() -> None:
    session = _session(
        prepare_battle_setup(
            map_id=0,
            max_steps=6,
            team_a=("tdm-alpha", "manual", "tdm-alpha", "tdm-alpha", "tdm-alpha"),
            team_b="tdm-beta",
        )
    )
    assert session.manual_slots == (1,)
    assert session.scenario.default_controlled_slot == 0
    assert session.controlled_global_slot == 1
