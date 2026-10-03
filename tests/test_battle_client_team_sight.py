"""Check the merged team sight that the BattleClient's Agent POV Night Veil paints.

The live Agent POV scene carries ``team_sight``
(``rendering.authorized_presentation.AuthorizedTeamSightV1``), computed by
``_battle_client.live_presentation`` from the POV scene's own sensors and
stones. These tests pin, on the Vision Moment (map 47, tick 91) served through
the real BattleClient service in Agent POV for Team A:

- The scene carries a team sight whose rings and Reach Edges lie on the floor.
- Lit means what Core says: for a grid of floor points away from any ring
  edge, a point is inside an odd number of lit rings exactly when some living
  Team A champion has it within its sight reach and Core's own
  ``has_clear_line_of_sight`` finds no stone between them. Stone shadows are
  dark; champions never block sight.
- The sight uses the team's own champions only: moving every Team B champion
  in the scene changes nothing.
- Oracle View carries no team sight, and importing the live builder still
  loads no array runtime (numpy is imported only while a frame is built).
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.scenarios import vision_moment_scenario
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client.control import create_session
from marl_battlegrounds._battle_client.live_presentation import (
    _merged_team_sight_v1,
)
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds._battle_client.system_worker import SystemWorker
from marl_battlegrounds.core.geometry import has_clear_line_of_sight
from marl_battlegrounds.rendering.authorized_presentation import (
    AuthorizedBattlefieldSceneV1,
    AuthorizedTeamSightV1,
)
from marl_battlegrounds.viewer.presentation_protocol import (
    NoSharedObsAuthorizedCurrentEndpointV1,
    OracleAuthorizedCurrentEndpointV1,
    PresentationApiErrorV1,
    SharedObsAuthorizedCurrentEndpointV1,
)

type _Served = tuple[dict[str, DebuggerService], DebuggerSession]
_EDGE_MARGIN = 0.05


@pytest.fixture(scope="module")
def served() -> Iterator[_Served]:
    policy = "tdm-alpha"
    setup = prepare_battle_setup(
        map_id=47,
        team_a=[policy, "manual", policy, policy, policy],
        team_b="tdm-beta",
    )
    setup = replace(setup, scenario=vision_moment_scenario())
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


def _scene(service: DebuggerService) -> AuthorizedBattlefieldSceneV1:
    result = service.current_presentation()
    assert result.outcome == "response"
    payload = result.payload
    assert not isinstance(payload, PresentationApiErrorV1)
    endpoint = payload.current_endpoint
    if isinstance(endpoint, OracleAuthorizedCurrentEndpointV1):
        return endpoint.scene
    assert isinstance(
        endpoint,
        SharedObsAuthorizedCurrentEndpointV1 | NoSharedObsAuthorizedCurrentEndpointV1,
    )
    return endpoint.parts.scene


def _ring_count(point: tuple[float, float], sight: AuthorizedTeamSightV1) -> int:
    x, y = point
    count = 0
    for ring in sight.lit_rings:
        inside = False
        for (x0, y0), (x1, y1) in zip(ring, (*ring[1:], ring[0]), strict=True):
            if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
                inside = not inside
        count += inside
    return count


def _near_ring_edge(point: tuple[float, float], sight: AuthorizedTeamSightV1) -> bool:
    px, py = point
    for ring in sight.lit_rings:
        a = np.array(ring)
        b = np.roll(a, -1, axis=0)
        d = b - a
        length = np.maximum((d**2).sum(axis=1), 1e-18)
        t = np.clip(
            ((px - a[:, 0]) * d[:, 0] + (py - a[:, 1]) * d[:, 1]) / length, 0, 1
        )
        nearest = a + t[:, None] * d
        if np.hypot(nearest[:, 0] - px, nearest[:, 1] - py).min() < _EDGE_MARGIN:
            return True
    return False


def test_pov_scene_carries_team_sight_on_the_floor(served: _Served) -> None:
    services, _ = served
    scene = _scene(services["pov"])
    sight = scene.team_sight
    assert isinstance(sight, AuthorizedTeamSightV1)
    assert sight.lit_rings and (sight.own_edges or sight.allied_edges)
    for run in (*sight.lit_rings, *sight.own_edges, *sight.allied_edges):
        for x, y in run:
            assert -1e-6 <= x <= scene.map.width + 1e-6
            assert -1e-6 <= y <= scene.map.height + 1e-6
    assert _scene(services["researcher"]).team_sight is None


def test_lit_floor_matches_cores_line_of_sight(served: _Served) -> None:
    services, session = served
    scene = _scene(services["pov"])
    sight = scene.team_sight
    assert sight is not None
    sensors = [
        (row.position, row.observation_radius)
        for row in scene.agents
        if row.team_id == 1 and row.life_state == "alive"
    ]
    obstacles = session.config.obstacles
    checked = shadowed = 0
    for gx in np.linspace(0.1, scene.map.width - 0.1, 41):
        for gy in np.linspace(0.1, scene.map.height - 0.1, 21):
            point = (float(gx), float(gy))
            if _near_ring_edge(point, sight):
                continue
            seen = False
            in_reach = False
            for (sx, sy), reach in sensors:
                if np.hypot(point[0] - sx, point[1] - sy) > reach:
                    continue
                in_reach = True
                if bool(
                    has_clear_line_of_sight(
                        jnp.asarray((sx, sy), dtype=jnp.float32),
                        jnp.asarray(point, dtype=jnp.float32),
                        obstacles,
                    )
                ):
                    seen = True
                    break
            assert (_ring_count(point, sight) % 2 == 1) == seen, point
            checked += 1
            shadowed += in_reach and not seen
    assert checked > 600
    assert shadowed > 10, "stone shadows inside the reach are dark"


def test_team_sight_ignores_the_other_team(served: _Served) -> None:
    services, _ = served
    scene = _scene(services["pov"])
    sensors = [row.public_agent_id for row in scene.agents if row.team_id == 1]
    own = next(row.public_agent_id for row in scene.agents if row.relation == "self")
    # Aura fields join their source's place, so the moved copy drops them;
    # the sight reads neither.
    moved = replace(
        scene,
        aura_fields=(),
        agents=tuple(
            replace(row, position=(1.0, 1.0)) if row.team_id == 2 else row
            for row in scene.agents
        ),
    )
    first = _merged_team_sight_v1(
        scene, sensor_public_agent_ids=sensors, own_public_agent_id=own
    )
    second = _merged_team_sight_v1(
        moved, sensor_public_agent_ids=sensors, own_public_agent_id=own
    )
    assert first is not None and first == second
