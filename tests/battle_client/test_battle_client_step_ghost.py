"""Check that the BattleClient's Step Ghost lands where Core plans the step.

The Step Ghost (``web/visual_debugger/src/looks/hotbar.js``) is the dotted
circle where your champion's planned step ends if nothing blocks it. The page
draws it from host facts only: your champion's place and its
``effective_movement_speed`` in the authorized arena scene, and the step's
compass direction from MOVE_DIRECTIONS, a copy of Core's fixed table. These
tests pin the two links that make that picture true:

- MOVE_DIRECTIONS in hotbar.js equals Core's
  UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION, entry for entry.
- On the Vision Moment (map 47, tick 91) with a slow on the Team A Warrior and
  a stun on the Team A Hunter added, served through the real BattleClient
  service, every champion the authorized scene shows (Agent POV for Team A,
  and Oracle View) has place + effective speed x direction equal to Core's own
  planned movement (``_build_intended_movement_deltas``) for each of the nine
  steps: the slowed Warrior moves less, the stunned Hunter does not move, the
  shielded Rogue moves at the spawn-shield speed. Agent POV shows no champion
  the scene leaves out, so the ghost can only ever be drawn for a body the
  team sees (the page draws it for your own champion alone).
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.scenarios import (
    build_vision_moment,
    vision_moment_scenario,
)
from tests.helpers.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client.control import create_session
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds._battle_client.system_worker import SystemWorker
from marl_battlegrounds.core.axis_mappings import (
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION,
)
from marl_battlegrounds.core.env import _build_intended_movement_deltas
from marl_battlegrounds.core.types import Action, EnvConfig, EnvState
from marl_battlegrounds.rendering.authorized_presentation import AuthorizedAgentV1
from marl_battlegrounds.viewer.presentation_protocol import (
    NoSharedObsAuthorizedCurrentEndpointV1,
    OracleAuthorizedCurrentEndpointV1,
    PresentationApiErrorV1,
    SharedObsAuthorizedCurrentEndpointV1,
)

_HOTBAR = (
    Path(__file__).resolve().parents[2]
    / "web"
    / "visual_debugger"
    / "src"
    / "looks"
    / "hotbar.js"
)
_SLOWED_WARRIOR = 1
_STUNNED_HUNTER = 2
_SHIELDED_ROGUE = 3


def _browser_directions() -> list[tuple[float, float]]:
    text = _HOTBAR.read_text(encoding="utf-8")
    block = text.split("export const MOVE_DIRECTIONS", 1)[1].split("]);", 1)[0]
    pairs = re.findall(r"Object\.freeze\(\[(-?[0-9.]+), (-?[0-9.]+)\]\)", block)
    return [(float(x), float(y)) for x, y in pairs]


def test_browser_step_directions_equal_cores_table() -> None:
    assert _browser_directions() == [
        (float(x), float(y)) for x, y in UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION
    ]


def _slowed_and_stunned_moment() -> tuple[EnvConfig, EnvState]:
    config, state = build_vision_moment()
    slow = np.asarray(state.slow_durations).copy()
    slow[_SLOWED_WARRIOR, 0] = 2
    stun = np.asarray(state.stun_durations).copy()
    stun[_STUNNED_HUNTER, 0] = 1
    return config, state._replace(
        slow_durations=jnp.asarray(slow, dtype=state.slow_durations.dtype),
        stun_durations=jnp.asarray(stun, dtype=state.stun_durations.dtype),
    )


type _Served = tuple[dict[str, DebuggerService], DebuggerSession]


@pytest.fixture(scope="module")
def service_and_session() -> Iterator[_Served]:
    policy = "tdm-alpha"
    setup = prepare_battle_setup(
        map_id=47,
        team_a=[policy, "manual", policy, policy, policy],
        team_b="tdm-beta",
    )
    scenario = replace(
        vision_moment_scenario(), build_scenario=_slowed_and_stunned_moment
    )
    setup = replace(setup, scenario=scenario)
    resolved = resolve_controllers(setup)
    session = create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=debugger_test_launch_specification(setup.seed),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
        team_a_controller=resolved.controllers[0],
        team_b_controller=resolved.controllers[1],
        execution_information_mode="shared_obs",
        systems=resolved.systems,
        borrow_systems=True,
        manual_slots=resolved.manual_slots,
        slot_labels=resolved.slot_labels,
    )
    worker = SystemWorker(caller_thread=True)
    worker.install(session, directory=None)
    services = {
        view: DebuggerService(
            session,
            view_mode=view,
            preset="analysis",
            include_stress=False,
            system_worker=worker,
        )
        for view in ("pov", "researcher")
    }
    try:
        yield services, session
    finally:
        for service in services.values():
            service.close()
        worker.close()
        while worker.drain():
            pass


def _scene_agents(service: DebuggerService) -> list[AuthorizedAgentV1]:
    result = service.current_presentation()
    assert result.outcome == "response"
    payload = result.payload
    assert not isinstance(payload, PresentationApiErrorV1)
    endpoint = payload.current_endpoint
    if isinstance(endpoint, OracleAuthorizedCurrentEndpointV1):
        return list(endpoint.scene.agents)
    assert isinstance(
        endpoint,
        SharedObsAuthorizedCurrentEndpointV1 | NoSharedObsAuthorizedCurrentEndpointV1,
    )
    return list(endpoint.parts.scene.agents)


def _global_slot(session: DebuggerSession, public_agent_id: str) -> int:
    for slot, row in enumerate(session.evaluation_context.roster):
        if row.public_agent_id == public_agent_id:
            return slot
    raise AssertionError(f"unknown public agent {public_agent_id}")


@pytest.mark.parametrize("view", ["pov", "researcher"])
def test_step_end_from_scene_facts_equals_cores_planned_step(
    service_and_session: _Served, view: str
) -> None:
    services, session = service_and_session
    agents = _scene_agents(services[view])
    assert agents, "the authorized scene shows champions"
    shown = {_global_slot(session, row.public_agent_id): row for row in agents}
    if view == "researcher":
        assert sorted(shown) == list(range(10))
    else:
        assert _SLOWED_WARRIOR in shown
    for move, (dx, dy) in enumerate(UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION):
        moves = jnp.full((10,), move, dtype=jnp.int32)
        zeros = jnp.zeros((10,), dtype=jnp.int32)
        deltas = np.asarray(
            _build_intended_movement_deltas(
                session.state,
                session.config,
                Action(move=moves, select_target=zeros, use_ultimate=zeros),
            )
        )
        positions = np.asarray(session.state.agent_positions)
        for slot, row in shown.items():
            speed = row.effective_movement_speed
            x, y = row.position
            ghost = (x + speed * dx, y + speed * dy)
            core = positions[slot] + deltas[slot]
            assert ghost == pytest.approx(tuple(core), abs=1e-5), (view, slot, move)
    if _STUNNED_HUNTER in shown:
        assert shown[_STUNNED_HUNTER].effective_movement_speed == 0
    warrior = shown[_SLOWED_WARRIOR]
    assert 0 < warrior.effective_movement_speed < warrior.base_movement_speed
    if _SHIELDED_ROGUE in shown:
        assert shown[_SHIELDED_ROGUE].effective_movement_speed == pytest.approx(
            float(session.config.spawn_shield_movement_speed)
        )
