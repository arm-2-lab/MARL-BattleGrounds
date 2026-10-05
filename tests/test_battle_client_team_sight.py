"""Check the merged team sight that the Agent POV Night Veil paints.

Agent POV scenes carry ``team_sight``
(``rendering.authorized_presentation.AuthorizedTeamSightV1``), computed by
``rendering.team_sight`` (the one owner for the BattleClient, DevClient and
Replay Viewer) from the POV scene's own sensors and stones. These tests pin,
on the Vision Moment (map 47, tick 91) served through the real BattleClient
service in Agent POV for Team A:

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
- The turn shapes (``with_team_sight_v1`` with a turn-start scene): one
  turn_start shape from the start scene's living sensors at their start
  spots, then the glide (fallers still lit) and the set left after each
  Fall beat, each equal to the merged sight of exactly those champions (a
  faller lights from where it fell); the frame's own set is not repeated; a
  team wipe gives one shape per Fall beat (six, not 32); a first frame has
  no turn shapes.
- Only the viewing team's own champions give light: naming enemy IDs as
  sensors changes nothing.
- Overlapping stones (a thin rail across a wall, a pillar touching it, as
  on map 51): the sweep aims at the points where their outlines cross, and
  near them the lit area matches Core's line of sight point for point.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.scenarios import vision_moment_scenario
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds._battle_client.control import create_session
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds._battle_client.system_worker import SystemWorker
from marl_battlegrounds.core.geometry import has_clear_line_of_sight
from marl_battlegrounds.rendering.authorized_presentation import (
    AuthorizedBattlefieldSceneV1,
    AuthorizedTeamSightTurnShapeV1,
    AuthorizedTeamSightV1,
)
from marl_battlegrounds.rendering.team_sight import (
    _sight_polygon,
    _Stone,
    _stone_crossings,
    merged_team_sight_v1,
    with_team_sight_v1,
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


def _ring_count(
    point: tuple[float, float],
    sight: AuthorizedTeamSightV1 | AuthorizedTeamSightTurnShapeV1,
) -> int:
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
    first = merged_team_sight_v1(
        scene, sensor_public_agent_ids=sensors, own_public_agent_id=own
    )
    second = merged_team_sight_v1(
        moved, sensor_public_agent_ids=sensors, own_public_agent_id=own
    )
    assert first is not None and first == second


def _same_lit_floor(
    a: AuthorizedTeamSightV1 | AuthorizedTeamSightTurnShapeV1,
    b: AuthorizedTeamSightV1,
    width: float,
    height: float,
) -> bool:
    for gx in np.linspace(0.1, width - 0.1, 41):
        for gy in np.linspace(0.1, height - 0.1, 21):
            point = (float(gx), float(gy))
            if _near_ring_edge(point, b):
                continue
            if _ring_count(point, a) % 2 != _ring_count(point, b) % 2:
                return False
    return True


def test_turn_shapes_cover_every_beat_of_the_turn(served: _Served) -> None:
    services, _ = served
    scene = _scene(services["pov"])
    width, height = scene.map.width, scene.map.height
    own = next(row.public_agent_id for row in scene.agents if row.relation == "self")
    team = sorted(row.public_agent_id for row in scene.agents if row.team_id == 1)
    assert len(team) == 5 and own in team
    # A turn where one champion falls (a sensor at the start only) and another
    # returns (a sensor at the end only), while the recipient moved 2 units.
    faller, returner = [key for key in reversed(team) if key != own][:2]
    start_scene = replace(
        scene,
        aura_fields=(),
        agents=tuple(
            replace(row, position=(row.position[0] - 2.0, row.position[1]))
            if row.public_agent_id == own
            else row
            for row in scene.agents
        ),
    )
    end_ids = [key for key in team if key != faller]
    start_ids = [key for key in team if key != returner]
    sight = with_team_sight_v1(
        scene,
        sensor_public_agent_ids=end_ids,
        own_public_agent_id=own,
        turn_start_scene=start_scene,
        turn_start_sensor_public_agent_ids=start_ids,
    ).team_sight
    assert sight is not None
    main = merged_team_sight_v1(
        scene, sensor_public_agent_ids=end_ids, own_public_agent_id=own
    )
    assert main is not None
    assert sight.lit_rings == main.lit_rings
    stays = [key for key in team if key not in (faller, returner)]
    # The glide (the faller still lit), then the set left after its Fall beat;
    # the returner joins only in the frame's own (main) shape.
    expected_sets = [[*stays, faller], stays]
    assert [
        (shape.stage, shape.lit_public_agent_ids) for shape in sight.turn_shapes
    ] == [
        ("turn_start", tuple(sorted(start_ids))),
        *(("turn_end", tuple(sorted(lit))) for lit in expected_sets),
    ]
    start = merged_team_sight_v1(
        start_scene, sensor_public_agent_ids=start_ids, own_public_agent_id=own
    )
    assert start is not None
    assert _same_lit_floor(sight.turn_shapes[0], start, width, height)
    assert not _same_lit_floor(sight.turn_shapes[0], main, width, height)
    # Each turn_end shape is the merged sight of exactly its champions at
    # their end spots (the faller lights from its own row in the end scene,
    # where it fell).
    for shape, lit in zip(sight.turn_shapes[1:], expected_sets, strict=True):
        expected = merged_team_sight_v1(
            scene, sensor_public_agent_ids=lit, own_public_agent_id=own
        )
        assert expected is not None
        assert _same_lit_floor(shape, expected, width, height)
    first_frame = with_team_sight_v1(
        scene, sensor_public_agent_ids=end_ids, own_public_agent_id=own
    ).team_sight
    assert first_frame is not None and first_frame.turn_shapes == ()
    # A whole team falling builds one shape per Fall beat, not one per
    # combination: the glide plus the set left after each fall.
    wiped = with_team_sight_v1(
        scene,
        sensor_public_agent_ids=[],
        own_public_agent_id=own,
        turn_start_scene=scene,
        turn_start_sensor_public_agent_ids=team,
    ).team_sight
    assert wiped is not None and wiped.lit_rings == ()
    assert [len(shape.lit_public_agent_ids) for shape in wiped.turn_shapes] == [
        5,
        5,
        4,
        3,
        2,
        1,
    ]


def test_only_the_viewing_teams_champions_give_light(served: _Served) -> None:
    services, _ = served
    scene = _scene(services["pov"])
    own = next(row.public_agent_id for row in scene.agents if row.relation == "self")
    team = [row.public_agent_id for row in scene.agents if row.team_id == 1]
    enemies = [row.public_agent_id for row in scene.agents if row.team_id == 2]
    assert enemies, "the Vision Moment shows some enemies"
    with_enemies = merged_team_sight_v1(
        scene, sensor_public_agent_ids=[*team, *enemies], own_public_agent_id=own
    )
    alone = merged_team_sight_v1(
        scene, sensor_public_agent_ids=team, own_public_agent_id=own
    )
    assert with_enemies == alone


def test_overlapping_stones_cast_exact_shadows() -> None:
    # Map 51's thin diagonal rail laid across a 1 x 1 wall: the rail sticks
    # out past the wall's corners, and the lit outline turns where the two
    # outlines cross. Seen from a sensor below-left (a real map 51 spot).
    theta = 0.7853981852531433
    rail = _Stone("wall", (14.5, 5.5), None, 2.0, 0.009999999776482582, theta)
    wall = _Stone("wall", (14.5, 5.5), None, 1.0, 1.0, 0.0)
    stones = (rail, wall)
    assert len(_stone_crossings(stones)) >= 4
    table = jnp.asarray(
        [
            [2.0, 14.5, 5.5, 0.0, 2.0, 0.009999999776482582, theta, 1.0],
            [2.0, 14.5, 5.5, 0.0, 1.0, 1.0, 0.0, 1.0],
            *([[0.0] * 8] * 30),
        ],
        dtype=jnp.float32,
    )
    sensor = (11.652600288391113, 2.481600046157837)
    points, _ = _sight_polygon(*sensor, 6.0, 20.0, 10.0, stones)
    ring = np.asarray(points)
    grid = np.stack(
        np.meshgrid(np.linspace(13.6, 14.4, 81), np.linspace(4.6, 5.4, 81)), axis=-1
    ).reshape(-1, 2)
    # Lit: inside the shape (even-odd ray test).
    a, b = ring, np.roll(ring, -1, axis=0)
    spans = (a[None, :, 1] > grid[:, 1:2]) != (b[None, :, 1] > grid[:, 1:2])
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = (b[None, :, 0] - a[None, :, 0]) * (grid[:, 1:2] - a[None, :, 1]) / (
            b[None, :, 1] - a[None, :, 1]
        ) + a[None, :, 0]
    lit = (np.count_nonzero(spans & (grid[:, 0:1] < cross), axis=1) % 2) == 1
    # Core: in reach and a clear line from the sensor.
    clear = np.asarray(
        jax.vmap(has_clear_line_of_sight, in_axes=(None, 0, None))(
            jnp.asarray(sensor, dtype=jnp.float32),
            jnp.asarray(grid, dtype=jnp.float32),
            table,
        )
    )
    seen = clear & (np.hypot(grid[:, 0] - sensor[0], grid[:, 1] - sensor[1]) <= 6.0)
    # Leave out points within 0.002 of the drawn edge and points inside or
    # within 0.001 of a stone (no champion can be there).
    d = b - a
    t = np.clip(
        (
            (grid[:, None, 0] - a[None, :, 0]) * d[None, :, 0]
            + (grid[:, None, 1] - a[None, :, 1]) * d[None, :, 1]
        )
        / np.maximum((d**2).sum(axis=1), 1e-18)[None, :],
        0,
        1,
    )
    edge = np.hypot(
        a[None, :, 0] + t * d[None, :, 0] - grid[:, None, 0],
        a[None, :, 1] + t * d[None, :, 1] - grid[:, None, 1],
    ).min(axis=1)
    lx = (grid[:, 0] - 14.5) * np.cos(theta) + (grid[:, 1] - 5.5) * np.sin(theta)
    ly = -(grid[:, 0] - 14.5) * np.sin(theta) + (grid[:, 1] - 5.5) * np.cos(theta)
    in_stone = (
        (np.abs(grid[:, 0] - 14.5) <= 0.501) & (np.abs(grid[:, 1] - 5.5) <= 0.501)
    ) | ((np.abs(lx) <= 1.001) & (np.abs(ly) <= 0.006))
    checked = (edge >= 0.002) & ~in_stone
    assert checked.sum() > 4000
    wrong = grid[checked & (lit != seen)]
    assert wrong.size == 0, wrong[:5]
