"""Check truthful Manual drafts and read-only policy inspection in every live view.

Real Python frames pass the browser's raw/authorized join and display adapters.
Mixed Systems, NoSharedObs Random, several drafts, dead configured Manual rows,
and malformed ownership joins keep the same action and information boundaries.
Drawing must not call a method or change its memory, draft, epoch, or accepted action.
Oracle View inspects the focused champion. Agent POV inspects its recipient: with
Manual champions in the game, focusing a policy champion keeps the view (and the
editable draft) on a Manual champion; without any, the policy focus is the
recipient and reads as policy inspection.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.battle_client.test_battle_client_mixed_control import _draft, _host, _session
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client import control
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.protocol import ResearcherLiveDebuggerFrameV2
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds.tasks import load_tdm_scenario

_ROOT = Path(__file__).resolve().parents[2]


def _pair(session: DebuggerSession, *, pov: bool) -> dict[str, Any]:
    service = DebuggerService(
        session,
        view_mode="pov" if pov else "researcher",
        preset="analysis",
        include_stress=False,
        session_id="pending-proof",
    )
    initial_memory = session.system_memory
    raw = service.current_frame()
    start = time.perf_counter()
    result = service.current_presentation()
    assert result.outcome == "response"
    first = result.payload.model_dump(mode="json")
    assert service.current_presentation().payload.model_dump(mode="json") == first
    elapsed = time.perf_counter() - start
    assert service.session is session
    assert service.session.system_memory is initial_memory
    assert service.current_frame() is raw
    service.close()
    return {
        "raw": raw.model_dump(mode="json"),
        "presentation": first,
        "manual_slots": list(session.manual_slots),
        "manual_moves": [
            session.pending_actions[slot].move_action for slot in session.manual_slots
        ],
        "build_seconds": elapsed,
    }


def _browser(pairs: list[dict[str, Any]], tmp_path: Path) -> None:
    node = shutil.which("node")
    assert node is not None, "Node is required for the live presentation join proof"
    saved = tmp_path / "pending-pairs.json"
    saved.write_text(json.dumps(pairs, allow_nan=False))
    result = subprocess.run(
        [
            node,
            str(_ROOT / "web/visual_debugger/tests/battle-client-pending-contract.js"),
            str(saved),
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mixed_system_drafts_policy_focus_and_last_actions(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    method = _host(calls, "Planner")
    session = _session(
        prepare_battle_setup(
            map_id=0,
            max_steps=4,
            team_a=("manual", method, "manual", method, method),
            team_b=(method, "manual", method, method, method),
        )
    )
    for slot, move in ((0, 1), (2, 3), (6, 5)):
        session = _draft(session, slot, move)
    initial_drafts = session.pending_actions
    initialized_calls = len(calls)
    pairs: list[dict[str, Any]] = []
    for slot in (0, 1, 2, 6, 7):
        focused = control.select_controlled_actor(session, slot)
        for pov in (False, True):
            pair = _pair(focused, pov=pov)
            wrapper = pair["presentation"]["live_inspection"]["inspection"]
            # A new service's Agent POV recipient is the focus when it is Manual,
            # else the lowest Manual slot.
            subject = (
                slot
                if not pov or slot in session.manual_slots
                else min(session.manual_slots)
            )
            if pov:
                assert pair["presentation"]["authority"][
                    "recipient_public_agent_id"
                ] == (session.evaluation_context.roster[subject].public_agent_id)
            assert wrapper["inspection_kind"] == (
                "editable_live_draft"
                if subject in session.manual_slots
                else "policy_read_only"
            )
            if subject not in session.manual_slots:
                assert "draft" not in wrapper and "draft_action" not in wrapper
                assert (
                    wrapper["decision_mask"]["owner_public_agent_id"]
                    == wrapper["actor_public_agent_id"]
                )
            pairs.append(pair)
    assert len(calls) == initialized_calls
    assert session.pending_actions == initial_drafts
    accepted = control.submit_interactive(session)
    accepted_count = len(calls)
    policy_focus = control.select_controlled_actor(accepted, 1)
    for pov in (False, True):
        pair = _pair(policy_focus, pov=pov)
        assert pair["presentation"]["latest_transition"] is not None
        pairs.append(pair)
    assert len(calls) == accepted_count
    _browser(pairs, tmp_path)


def test_no_shared_policy_focus_uses_only_own_masks(tmp_path: Path) -> None:
    session = control.create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="random_valid",
        team_b_controller="manual",
        execution_information_mode="no_shared_obs",
    )
    session = _draft(_draft(session, 5, 2), 6, 4)
    pairs: list[dict[str, Any]] = []
    for slot in (0, 5, 6):
        focused = control.select_controlled_actor(session, slot)
        for pov in (False, True):
            pairs.append(_pair(focused, pov=pov))
    _browser(pairs, tmp_path)


def test_dead_manual_row_stays_pending_and_empty_manual_set_is_valid(
    tmp_path: Path,
) -> None:
    loaded = load_tdm_scenario(6)
    dead = next(
        slot for slot in range(5) if not bool(loaded.initial_state.alive_mask[slot])
    )
    calls: list[dict[str, Any]] = []
    method = _host(calls)
    session = _session(
        prepare_battle_setup(
            scenario_id=6,
            team_a=tuple("manual" if slot == dead else method for slot in range(5)),
        )
    )
    session = _draft(session, dead, 1)
    pairs = [_pair(session, pov=pov) for pov in (False, True)]
    for pair in pairs:
        pending = (
            pair["presentation"].get("pending_joint_action")
            or pair["presentation"]["researcher_space"]["pending_joint_action"]
        )
        assert len(pending["action_rows"]) == 1
        assert pending["action_rows"][0]["pending_action"]["move_action"] == 1
    autonomous = _session(prepare_battle_setup(scenario_id=6, team_a=method))
    autonomous = control.select_controlled_actor(autonomous, dead)
    pairs.extend(_pair(autonomous, pov=pov) for pov in (False, True))
    _browser(pairs, tmp_path)


def test_raw_pending_rows_cannot_omit_a_configured_manual_actor() -> None:
    session = _session(prepare_battle_setup(map_id=0))
    pair = _pair(session, pov=False)
    raw = pair["raw"]
    raw["hud"]["pending_actions"].pop()
    with pytest.raises(ValueError, match="Manual slots"):
        ResearcherLiveDebuggerFrameV2.model_validate_json(json.dumps(raw))
