"""Build the floor an Agent POV team can see: the Night Veil's exact shape.

This module owns the Agent POV sight shape for all three clients: the
BattleClient and DevClient (``_battle_client.live_presentation``) and the
Replay Viewer (``viewer.presentation``). Each living sensor (a champion whose
sight the team shares) lights a star-shaped area: rays from its center run
until they meet a stone, the floor edge or its sight reach, the same test
Core's line of sight makes between two centers (``core.geometry
.has_clear_line_of_sight``: stones block sight, champions do not). The lit
areas of the team's sensors are merged into closed rings, so an even-odd fill
of the floor plus the rings darkens exactly the team's blind spots: beyond
reach and behind stone. The browser only paints the rings (amendment A17
keeps line of sight out of the browser).

Entry points:

- ``merged_team_sight_v1`` returns the merged shape of a scene's living
  sensors (AuthorizedTeamSightV1, no turn shapes).
- ``with_team_sight_v1`` returns an Agent POV scene with its ``team_sight``
  attached; given the turn's start scene it also adds the turn shapes the veil
  passes through while that turn plays (AuthorizedTeamSightTurnShapeV1).

Precision: rays every 1.5 degrees plus a pair either side of every stone
corner, pillar tangent, crossing of two stones' outlines, floor corner and
reach crossing, so stone faces and shadow edges are exact and a reach arc is
drawn as chords at most 0.0006 world units inside the true circle (far below
a pixel); points are rounded to 4 decimals. The front of a round stone is
drawn as chords too, so a sliver inside the stone (under its picture, where
no champion can be) may count as lit.

Boundary: only the named sensors' own rows (place, life, sight reach), and
only rows of the viewing team itself, and the public map are read, so nothing
the team cannot see is used, and the result names no enemy. Host display
work only: no simulator step, no JAX, no file or network access. numpy is
imported inside the functions that need it, so importing this module loads no
array runtime. A small in-process cache keeps recent shapes (the start of one
turn is the end of the one before).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np

from marl_battlegrounds.rendering.authorized_presentation import (
    AuthorizedBattlefieldSceneV1,
    AuthorizedTeamSightTurnShapeV1,
    AuthorizedTeamSightV1,
    Point2D,
)

# The team-sight sweep (Agent POV Night Veil). One ray every 1.5 degrees plus a
# pair of rays just either side of every stone corner, pillar tangent and point
# where a reach circle meets a stone or the floor edge, as the approved vision
# page's sweep does (vision_build.py, sight_shape).
_SIGHT_RAY_STEP_DEGREES = 1.5
_SIGHT_CRITICAL_NUDGE = 1e-4
# Each sensor's shape is pulled toward its sensor by a different tiny fraction:
# the step times the square root of (its place in the list + 2). Two shapes'
# copies of one stone face or floor edge then never fall on one line: that
# would need the ratio of two pull factors (an irrational number such as
# sqrt(3 / 2)) to equal a ratio of map distances, which are plain decimals.
# (Whole-number factors 1, 2, 3 failed: pads 1.5 and 0.5 below the top edge,
# pulled by 1 and 3 steps, both landed 1.5 steps below it.) The first step is
# tried first; when an outline still does not close, the next one. 1e-4 times
# sqrt(6) of a 6.5-unit reach is 0.0016 world units: well under a pixel.
_SIGHT_SHRINK_STEPS = (1e-7, 1e-6, 1e-5, 1e-4)
_SIGHT_DECIMALS = 4
# Outline points closer than this (world units) are one point when the merged
# outline is joined.
_SNAP = 1e-9

type _Polygon = tuple[np.ndarray, np.ndarray]
type _Run = tuple[Point2D, ...]
type _Geometry = tuple[tuple[_Run, ...], tuple[_Run, ...], tuple[_Run, ...]]
type _Sensor = tuple[str, float, float, float]


class _Stone(NamedTuple):
    """One map stone as plain values, so a sight request can be cached.

    Attributes
    ----------
    kind : str
        "pillar" or "wall" (AuthorizedObstacleV1.kind).
    center : tuple[float, float]
        Center in world units (y up).
    radius : float | None
        Pillar radius in world units; None for a wall.
    width, height : float | None
        Wall size in world units along its rotated x and y axes; None for a
        pillar.
    theta : float
        Wall rotation in radians (counterclockwise).
    """

    kind: str
    center: tuple[float, float]
    radius: float | None
    width: float | None
    height: float | None
    theta: float


def _stone_outline(
    stone: _Stone,
) -> tuple[list[tuple[Point2D, Point2D]], tuple[Point2D, float] | None]:
    """Return one stone's outline as wall sides or a pillar circle.

    Parameters
    ----------
    stone : _Stone
        A pillar or a wall.

    Returns
    -------
    tuple
        (sides, circle): a wall gives its four sides as point pairs and no
        circle; a pillar gives no sides and (center, radius). A stone with
        missing sizes gives neither.
    """
    cx, cy = stone.center
    if stone.kind == "pillar" and stone.radius is not None:
        return [], ((cx, cy), stone.radius)
    if stone.width is None or stone.height is None:
        return [], None
    c, sn = math.cos(stone.theta), math.sin(stone.theta)
    hw, hh = stone.width / 2, stone.height / 2
    corners = [
        (cx + sx * hw * c - sy * hh * sn, cy + sx * hw * sn + sy * hh * c)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return [(corners[i], corners[(i + 1) % 4]) for i in range(4)], None


@lru_cache(maxsize=64)
def _stone_crossings(stones: tuple[_Stone, ...]) -> tuple[Point2D, ...]:
    """Return every point where the outlines of two different stones cross.

    Where stones overlap (a thin rail across a wall or a pillar, for
    example), the lit outline can turn at such a point, so the sweep aims a
    pair of rays at each one, as it does at stone corners.

    Parameters
    ----------
    stones : tuple[_Stone, ...]
        The map's stones.

    Returns
    -------
    tuple[Point2D, ...]
        Crossing points in world units, in no particular order. Empty when no
        two stones touch. Cached per map (64 entries).
    """
    outlines = [_stone_outline(stone) for stone in stones]
    points: list[Point2D] = []

    def segment_circle(
        a: Point2D, b: Point2D, center: Point2D, radius: float
    ) -> list[Point2D]:
        """Return where segment a-b meets a circle."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        fx, fy = a[0] - center[0], a[1] - center[1]
        qa = dx * dx + dy * dy
        qb = 2 * (fx * dx + fy * dy)
        qc = fx * fx + fy * fy - radius * radius
        disc = qb * qb - 4 * qa * qc
        if qa < 1e-18 or disc < 0:
            return []
        root = math.sqrt(disc)
        return [
            (a[0] + t * dx, a[1] + t * dy)
            for t in ((-qb - root) / (2 * qa), (-qb + root) / (2 * qa))
            if 0 <= t <= 1
        ]

    for i, (sides_i, circle_i) in enumerate(outlines):
        for sides_j, circle_j in outlines[i + 1 :]:
            for a, b in sides_i:
                for c, d in sides_j:
                    rx, ry = b[0] - a[0], b[1] - a[1]
                    sx, sy = d[0] - c[0], d[1] - c[1]
                    denom = rx * sy - ry * sx
                    if abs(denom) < 1e-15:
                        continue
                    t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / denom
                    u = ((c[0] - a[0]) * ry - (c[1] - a[1]) * rx) / denom
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        points.append((a[0] + t * rx, a[1] + t * ry))
                if circle_j is not None:
                    points += segment_circle(a, b, *circle_j)
            if circle_i is not None:
                for c, d in sides_j:
                    points += segment_circle(c, d, *circle_i)
                if circle_j is not None:
                    (x0, y0), r0 = circle_i
                    (x1, y1), r1 = circle_j
                    gap = math.hypot(x1 - x0, y1 - y0)
                    if abs(r0 - r1) < gap < r0 + r1:
                        along = (r0 * r0 - r1 * r1 + gap * gap) / (2 * gap)
                        across = math.sqrt(max(r0 * r0 - along * along, 0.0))
                        mx = x0 + along * (x1 - x0) / gap
                        my = y0 + along * (y1 - y0) / gap
                        points += [
                            (
                                mx + across * (y1 - y0) / gap,
                                my - across * (x1 - x0) / gap,
                            ),
                            (
                                mx - across * (y1 - y0) / gap,
                                my + across * (x1 - x0) / gap,
                            ),
                        ]
    return tuple(points)


def _sight_critical_angles(
    ox: float,
    oy: float,
    reach: float,
    width: float,
    height: float,
    obstacles: Sequence[_Stone],
) -> list[float]:
    """Return the ray angles (radians) where one sensor's lit shape changes.

    Parameters
    ----------
    ox, oy : float
        The sensor's place in world units (y up).
    reach : float
        Its sight reach in world units.
    width, height : float
        The floor size in world units; the floor spans [0, width] x [0, height].
    obstacles : Sequence[_Stone]
        The map's stones: pillars (center, radius) and walls (center, width,
        height, theta in radians, width along the rotated x axis).

    Returns
    -------
    list[float]
        Angles of pillar tangents, wall corners, points where two stones'
        outlines cross (_stone_crossings), floor corners within reach, and
        points where the reach circle meets a stone edge or the floor edge.
        Unsorted, may repeat.
    """
    angles: list[float] = []
    for obstacle in obstacles:
        cx, cy = obstacle.center
        if obstacle.kind == "pillar" and obstacle.radius is not None:
            r = obstacle.radius
            dx, dy = cx - ox, cy - oy
            distance = math.hypot(dx, dy)
            if distance > r:
                bearing, half = math.atan2(dy, dx), math.asin(r / distance)
                angles += [bearing - half, bearing + half]
            if abs(reach - r) < distance < reach + r and distance > 1e-9:
                a = (reach * reach - r * r + distance * distance) / (2 * distance)
                base = math.atan2(dy, dx)
                spread = math.acos(max(-1.0, min(1.0, a / reach)))
                angles += [base - spread, base + spread]
        elif obstacle.width is not None and obstacle.height is not None:
            c, sn = math.cos(obstacle.theta), math.sin(obstacle.theta)
            hw, hh = obstacle.width / 2, obstacle.height / 2
            corners = [
                (cx + sx * hw * c - sy * hh * sn, cy + sx * hw * sn + sy * hh * c)
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
            for index, (px, py) in enumerate(corners):
                qx, qy = corners[(index + 1) % 4]
                angles.append(math.atan2(py - oy, px - ox))
                ex, ey, fx, fy = qx - px, qy - py, px - ox, py - oy
                a2 = ex * ex + ey * ey
                b2 = 2 * (fx * ex + fy * ey)
                c2 = fx * fx + fy * fy - reach * reach
                disc = b2 * b2 - 4 * a2 * c2
                if a2 > 1e-12 and disc > 0:
                    for t in (
                        (-b2 - math.sqrt(disc)) / (2 * a2),
                        (-b2 + math.sqrt(disc)) / (2 * a2),
                    ):
                        if 0 < t < 1:
                            angles.append(
                                math.atan2(py + ey * t - oy, px + ex * t - ox)
                            )
    # Where two stones' outlines cross (a rail across a wall, say), the lit
    # outline can turn too.
    for x, y in _stone_crossings(tuple(obstacles)):
        if math.hypot(x - ox, y - oy) < reach:
            angles.append(math.atan2(y - oy, x - ox))
    # The floor's own corners: without a ray at each, the outline would cut
    # a small lit triangle off every corner within reach.
    for x, y in ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height)):
        if math.hypot(x - ox, y - oy) < reach:
            angles.append(math.atan2(y - oy, x - ox))
    for x in (0.0, width):
        dx = x - ox
        if abs(dx) < reach:
            dy = math.sqrt(reach * reach - dx * dx)
            angles += [
                math.atan2(y - oy, dx) for y in (oy - dy, oy + dy) if 0 <= y <= height
            ]
    for y in (0.0, height):
        dy = y - oy
        if abs(dy) < reach:
            dx = math.sqrt(reach * reach - dy * dy)
            angles += [
                math.atan2(dy, x - ox) for x in (ox - dx, ox + dx) if 0 <= x <= width
            ]
    return angles


def _sight_polygon(
    ox: float,
    oy: float,
    reach: float,
    width: float,
    height: float,
    obstacles: Sequence[_Stone],
) -> _Polygon:
    """Return one sensor's lit shape as a star polygon around the sensor.

    Each ray runs from the sensor until it meets a stone, the floor edge or its
    reach, the same test Core's line of sight makes between two centers
    (stones block sight; champions do not). A point between two neighbors
    on one straight line (a stone face or floor edge, not the reach circle)
    is dropped, so faces are drawn with their corner points only; corners,
    however close their neighbors, are kept.

    Parameters
    ----------
    ox, oy, reach, width, height, obstacles
        As in _sight_critical_angles.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Points (n, 2) float64 in world units, sorted by angle (counterclockwise),
        and full (n,) bool: True where the ray ran its whole reach (the point
        lies on the reach circle).
    """
    import numpy as np  # lazy: importing this module loads no array runtime

    tau = 2 * math.pi
    base = [math.radians(i * _SIGHT_RAY_STEP_DEGREES) for i in range(240)]
    critical = _sight_critical_angles(ox, oy, reach, width, height, obstacles)
    raw = np.array(
        base
        + [a - _SIGHT_CRITICAL_NUDGE for a in critical]
        + [a + _SIGHT_CRITICAL_NUDGE for a in critical]
    )
    raw = np.sort(np.mod(raw, tau))
    keep = np.concatenate(([True], np.diff(raw) >= 1e-7))
    angles = raw[keep]
    dx, dy = np.cos(angles), np.sin(angles)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(
            dx > 1e-12, (width - ox) / dx, np.where(dx < -1e-12, -ox / dx, np.inf)
        )
        ty = np.where(
            dy > 1e-12, (height - oy) / dy, np.where(dy < -1e-12, -oy / dy, np.inf)
        )
    floor = np.minimum(tx, ty)
    hit = np.full(angles.shape, np.inf)
    for obstacle in obstacles:
        cx, cy = obstacle.center
        if obstacle.kind == "pillar" and obstacle.radius is not None:
            px, py = cx - ox, cy - oy
            b = px * dx + py * dy
            cc = px * px + py * py - obstacle.radius**2
            disc = b * b - cc
            with np.errstate(invalid="ignore"):
                t = np.where(cc > 0, b - np.sqrt(np.maximum(disc, 0.0)), 0.0)
            t = np.where((disc >= 0) & (t >= 0), t, np.inf)
        elif obstacle.width is not None and obstacle.height is not None:
            c, sn = math.cos(obstacle.theta), math.sin(obstacle.theta)
            lx = (ox - cx) * c + (oy - cy) * sn
            ly = -(ox - cx) * sn + (oy - cy) * c
            ldx, ldy = dx * c + dy * sn, -dx * sn + dy * c
            low = np.full(angles.shape, -np.inf)
            high = np.full(angles.shape, np.inf)
            for p0, dp, half in (
                (lx, ldx, obstacle.width / 2),
                (ly, ldy, obstacle.height / 2),
            ):
                parallel = np.abs(dp) < 1e-12
                safe = np.where(parallel, 1.0, dp)
                t1, t2 = (-half - p0) / safe, (half - p0) / safe
                outside = parallel & (abs(p0) > half)
                low = np.where(
                    parallel,
                    np.where(outside, np.inf, low),
                    np.maximum(low, np.minimum(t1, t2)),
                )
                high = np.where(
                    parallel,
                    np.where(outside, -np.inf, high),
                    np.minimum(high, np.maximum(t1, t2)),
                )
            t = np.where((low <= high) & (high >= 0), np.maximum(low, 0.0), np.inf)
        else:
            continue
        hit = np.minimum(hit, t)
    best = np.minimum(np.minimum(floor, hit), reach)
    full = np.minimum(floor, hit) >= reach
    points = np.stack((ox + dx * best, oy + dy * best), axis=1)
    previous, following = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
    # A point is dropped only when it lies on the straight line through its
    # neighbors: the sine of the turn there is below 1e-9. The test is relative
    # to the neighbors' distances, so a stone corner seen head-on, where the
    # rays either side of it land only 0.0002 units apart, is never mistaken
    # for a straight face (an absolute area test cut through such stones).
    to_point = points - previous
    to_next = following - points
    turn = np.abs(to_point[:, 0] * to_next[:, 1] - to_point[:, 1] * to_next[:, 0])
    flat = (
        ~full
        & ~np.roll(full, 1)
        & ~np.roll(full, -1)
        & (turn <= 1e-9 * np.hypot(*to_point.T) * np.hypot(*to_next.T))
        & (np.hypot(*(following - previous).T) >= 1e-9)
    )
    points, full = points[~flat], full[~flat]
    # Two rays may end on the same point (for example both sides of a corner
    # on the floor edge); keep one, so every outline piece has a length.
    distinct = np.hypot(*(points - np.roll(points, 1, axis=0)).T) > 1e-9
    if not distinct.any():
        distinct[0] = True
    return points[distinct], full[distinct]


def _inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Return which points lie inside a polygon (even-odd ray test).

    Parameters
    ----------
    points : numpy.ndarray
        (k, 2) float64 query points.
    polygon : numpy.ndarray
        (n, 2) float64 closed polygon (last point joins the first).

    Returns
    -------
    numpy.ndarray
        (k,) bool.
    """
    import numpy as np  # lazy: importing this module loads no array runtime

    x0, y0 = polygon[:, 0][None, :], polygon[:, 1][None, :]
    x1 = np.roll(polygon[:, 0], -1)[None, :]
    y1 = np.roll(polygon[:, 1], -1)[None, :]
    px, py = points[:, 0][:, None], points[:, 1][:, None]
    spans = (y0 > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_x = (x1 - x0) * (py - y0) / (y1 - y0) + x0
    return (np.count_nonzero(spans & (px < cross_x), axis=1) % 2) == 1


@lru_cache(maxsize=512)
def _merged_sight(
    width: float,
    height: float,
    stones: tuple[_Stone, ...],
    sensors: tuple[_Sensor, ...],
    own_id: str,
    shrink_step: float,
) -> _Geometry | None:
    """Return the merged lit rings and Reach Edges of some sensors.

    Parameters
    ----------
    width, height : float
        The floor size in world units; the floor spans [0, width] x [0, height].
    stones : tuple[_Stone, ...]
        The map's stones.
    sensors : tuple[tuple[str, float, float, float], ...]
        (public ID, x, y, sight reach) of each sensor giving light, in world
        units. Order only sets each shape's tiny pull toward its sensor.
        Sensors within 1e-6 units of each other count once, with the longest
        reach (the recipient's on a tie): the shorter shape lies inside it.
    own_id : str
        The recipient's public ID; its Reach Edges are the "own" ones.
    shrink_step : float
        The pull step (one of _SIGHT_SHRINK_STEPS): shape number i (from 0)
        is pulled toward its sensor by shrink_step * sqrt(i + 2) of its size.

    Returns
    -------
    tuple | None
        (lit_rings, own_edges, allied_edges), each a tuple of point runs in
        world units rounded to 4 decimals; all empty when there are no
        sensors (the whole floor is dark). None when the merged outline does
        not close (the caller then tries the next shrink step).

    Notes
    -----
    Each sensor's shape is the ray sweep; the union keeps each shape's edge
    pieces that lie inside no other shape, snaps their ends within 1e-9 units
    and joins them into closed rings (where two lit areas touch at a point,
    the rings may be grouped either way; the even-odd fill is the same).
    Cost: about 11 ms per call on CPU with five sensors. Results are cached
    (512 entries); the arguments are plain hashable values.
    """
    import numpy as np  # lazy: importing this module loads no array runtime

    polygons: list[np.ndarray] = []
    fulls: list[np.ndarray] = []
    owners: list[str] = []
    # Two sensors on the same spot give star shapes from one center, so the
    # one with the shorter reach lies inside the other; keep only the larger
    # (the recipient's on a tie), or the union would meet two copies of one
    # edge and give up.
    kept_sensors: list[_Sensor] = []
    for sensor in sorted(sensors, key=lambda row: (-row[3], row[0] != own_id)):
        if all(
            math.hypot(sensor[1] - other[1], sensor[2] - other[2]) > 1e-6
            for other in kept_sensors
        ):
            kept_sensors.append(sensor)
    sensors = tuple(row for row in sensors if row in kept_sensors)
    for index, (public_agent_id, ox, oy, reach) in enumerate(sensors):
        points, full = _sight_polygon(ox, oy, reach, width, height, stones)
        shrink = 1 - shrink_step * math.sqrt(index + 2)
        polygons.append(np.array((ox, oy)) + (points - np.array((ox, oy))) * shrink)
        fulls.append(full)
        owners.append(public_agent_id)
    # Split every edge where it crosses an edge of another shape.
    splits: list[list[list[tuple[float, tuple[float, float]]]]] = [
        [[] for _ in polygon] for polygon in polygons
    ]
    for i in range(len(polygons)):
        a0 = polygons[i]
        d1 = np.roll(a0, -1, axis=0) - a0
        for j in range(i + 1, len(polygons)):
            b0 = polygons[j]
            d2 = np.roll(b0, -1, axis=0) - b0
            denom = (
                d1[:, 0][:, None] * d2[:, 1][None, :]
                - d1[:, 1][:, None] * d2[:, 0][None, :]
            )
            ex = b0[:, 0][None, :] - a0[:, 0][:, None]
            ey = b0[:, 1][None, :] - a0[:, 1][:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (ex * d2[:, 1][None, :] - ey * d2[:, 0][None, :]) / denom
                u = (ex * d1[:, 1][:, None] - ey * d1[:, 0][:, None]) / denom
            crossing = (np.abs(denom) > 1e-15) & (t > 0) & (t < 1) & (u > 0) & (u < 1)
            for ia, jb in zip(*np.nonzero(crossing), strict=True):
                ta, ub = float(t[ia, jb]), float(u[ia, jb])
                point = (
                    float(a0[ia, 0] + ta * d1[ia, 0]),
                    float(a0[ia, 1] + ta * d1[ia, 1]),
                )
                splits[i][int(ia)].append((ta, point))
                splits[j][int(jb)].append((ub, point))
    # Cut each shape's edges into pieces and keep the pieces inside no other shape.
    pieces: list[tuple[int, int, tuple[float, float], tuple[float, float]]] = []
    for i, polygon in enumerate(polygons):
        count = len(polygon)
        for k in range(count):
            start = (float(polygon[k, 0]), float(polygon[k, 1]))
            end_index = (k + 1) % count
            end = (float(polygon[end_index, 0]), float(polygon[end_index, 1]))
            stops = [start, *(point for _, point in sorted(splits[i][k])), end]
            pieces.extend(
                (i, k, stops[n], stops[n + 1])
                for n in range(len(stops) - 1)
                if stops[n] != stops[n + 1]
            )
    if not pieces:
        return ((), (), ())
    middles = np.array(
        [((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) for _, _, p, q in pieces]
    )
    owner_index = np.array([piece[0] for piece in pieces])
    covered = np.zeros(len(pieces), dtype=bool)
    for j, polygon in enumerate(polygons):
        others = owner_index != j
        covered[others] |= _inside_polygon(middles[others], polygon)
    kept = [piece for piece, hidden in zip(pieces, covered, strict=True) if not hidden]
    # Snap piece ends within 1e-9 units of each other to one point (a crossing
    # found from both shapes, or near a floor corner where two shapes' edges
    # meet at a shallow angle, can differ by rounding) and drop pieces that
    # shrink to nothing, so every outline point has as many pieces arriving
    # as leaving.
    snapped: dict[tuple[int, int], tuple[float, float]] = {}

    def snap(point: tuple[float, float]) -> tuple[float, float]:
        """Return the first point seen within 1e-9 units, or this one."""
        gx, gy = round(point[0] / _SNAP), round(point[1] / _SNAP)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near = snapped.get((gx + dx, gy + dy))
                if (
                    near is not None
                    and abs(near[0] - point[0]) <= _SNAP
                    and abs(near[1] - point[1]) <= _SNAP
                ):
                    return near
        snapped[(gx, gy)] = point
        return point

    kept = [(i, k, snap(start), snap(end)) for i, k, start, end in kept]
    kept = [piece for piece in kept if piece[2] != piece[3]]
    # Join the kept pieces into closed rings. Where two lit areas touch at one
    # point, more than one piece leaves it; any unused one is taken, because an
    # even-odd fill depends only on the set of pieces, not on how they are
    # grouped into rings.
    outgoing: dict[tuple[float, float], list[int]] = {}
    for index, (_, _, start, _) in enumerate(kept):
        outgoing.setdefault(start, []).append(index)
    used = [False] * len(kept)
    rings: list[tuple[tuple[float, float], ...]] = []
    for first in range(len(kept)):
        if used[first]:
            continue
        ring: list[tuple[float, float]] = []
        index = first
        while True:
            used[index] = True
            _, _, start, end = kept[index]
            ring.append(start)
            if end == kept[first][2]:
                break
            following = [n for n in outgoing.get(end, ()) if not used[n]]
            if not following:
                return None  # an open outline: refuse rather than guess
            index = following[0]
        if len(ring) >= 3:
            rings.append(tuple(ring))
    # Reach Edges: kept pieces of reach-circle edges, joined along each shape.
    own: list[tuple[tuple[float, float], ...]] = []
    allied: list[tuple[tuple[float, float], ...]] = []
    run: list[tuple[float, float]] = []
    run_owner = -1
    for i, k, start, end in [*kept, (-1, -1, (0.0, 0.0), (0.0, 0.0))]:
        arc = i >= 0 and bool(fulls[i][k]) and bool(fulls[i][(k + 1) % len(fulls[i])])
        if run and (not arc or i != run_owner or start != run[-1]):
            (own if owners[run_owner] == own_id else allied).append(tuple(run))
            run = []
        if arc:
            if not run:
                run = [start]
                run_owner = i
            run.append(end)

    def rounded(
        runs: list[tuple[tuple[float, float], ...]],
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Round every point to _SIGHT_DECIMALS decimals (world units)."""
        return tuple(
            tuple(
                (round(x, _SIGHT_DECIMALS), round(y, _SIGHT_DECIMALS))
                for x, y in points
            )
            for points in runs
        )

    return rounded(rings), rounded(own), rounded(allied)


def _stones(scene: AuthorizedBattlefieldSceneV1) -> tuple[_Stone, ...]:
    """Return the scene map's stones as cacheable plain values.

    Parameters
    ----------
    scene : AuthorizedBattlefieldSceneV1
        Any authorized scene; only ``scene.map.obstacles`` is read.

    Returns
    -------
    tuple[_Stone, ...]
        One row per stone, in map order.
    """
    return tuple(
        _Stone(
            kind=obstacle.kind,
            center=(float(obstacle.center[0]), float(obstacle.center[1])),
            radius=obstacle.radius,
            width=obstacle.width,
            height=obstacle.height,
            theta=float(obstacle.theta),
        )
        for obstacle in scene.map.obstacles
    )


def _sight(
    scene: AuthorizedBattlefieldSceneV1,
    sensors: Sequence[_Sensor],
    own_public_agent_id: str,
) -> _Geometry | None:
    """Return the merged geometry of sensors on a scene's map (cached).

    Parameters
    ----------
    scene : AuthorizedBattlefieldSceneV1
        Supplies the floor size and stones.
    sensors : Sequence[tuple[str, float, float, float]]
        (public ID, x, y, sight reach) of each sensor giving light.
    own_public_agent_id : str
        The recipient's public ID.

    Returns
    -------
    tuple | None
        As in _merged_sight, from the first shrink step whose outline closes;
        None only when none of _SIGHT_SHRINK_STEPS gives a closed outline.
    """
    for step in _SIGHT_SHRINK_STEPS:
        geometry = _merged_sight(
            float(scene.map.width),
            float(scene.map.height),
            _stones(scene),
            tuple(sensors),
            own_public_agent_id,
            step,
        )
        if geometry is not None:
            return geometry
    return None


def _living_sensors(
    scene: AuthorizedBattlefieldSceneV1, sensor_public_agent_ids: Sequence[str]
) -> list[_Sensor]:
    """Return the named sensors that are alive in a scene, in scene order.

    Parameters
    ----------
    scene : AuthorizedBattlefieldSceneV1
        An Agent POV scene.
    sensor_public_agent_ids : Sequence[str]
        Public IDs whose sight the team shares. Missing rows, rows that are
        not alive, and rows that are not the viewing team's own (relation
        other than "self" or "ally") give no light, so a wrong ID list can
        never light the floor around an enemy.

    Returns
    -------
    list[tuple[str, float, float, float]]
        (public ID, x, y, sight reach) per living own-team sensor, in scene
        order.
    """
    wanted = set(sensor_public_agent_ids)
    return [
        (
            row.public_agent_id,
            float(row.position[0]),
            float(row.position[1]),
            float(row.observation_radius),
        )
        for row in scene.agents
        if row.public_agent_id in wanted
        and row.life_state == "alive"
        and row.relation in ("self", "ally")
    ]


def merged_team_sight_v1(
    scene: AuthorizedBattlefieldSceneV1,
    *,
    sensor_public_agent_ids: Sequence[str],
    own_public_agent_id: str,
) -> AuthorizedTeamSightV1 | None:
    """Return the floor the viewing team sees now, merged, for the Night Veil.

    Parameters
    ----------
    scene : AuthorizedBattlefieldSceneV1
        The Agent POV scene. Only its map (size and stones) and the rows of
        the named sensors (place, life, ``observation_radius``) are read, so
        nothing the team cannot see is used.
    sensor_public_agent_ids : Sequence[str]
        Public IDs whose sight the team shares: every SharedObs sensor source,
        or the recipient alone under NoSharedObs. Rows that are missing or not
        alive give no light.
    own_public_agent_id : str
        The recipient (the champion whose view this is); its Reach Edge is
        the solid one.

    Returns
    -------
    AuthorizedTeamSightV1 | None
        The lit rings and Reach Edges in world units, rounded to 4 decimals,
        with no turn shapes. Empty rings when no sensor is alive (the whole
        floor is dark). None when the merged outline cannot be traced cleanly
        (a degenerate touch); the page then keeps its reach-circle veil.

    Notes
    -----
    Host display work only; see the module description for the method,
    precision and cost.
    """
    geometry = _sight(
        scene, _living_sensors(scene, sensor_public_agent_ids), own_public_agent_id
    )
    if geometry is None:
        return None
    rings, own, allied = geometry
    return AuthorizedTeamSightV1(
        lit_rings=rings, own_edges=own, allied_edges=allied, turn_shapes=()
    )


def _natural_key(public_agent_id: str) -> tuple[tuple[int, int | str], ...]:
    """Return a sort key that orders public IDs as the browser's beats do.

    Digits compare as numbers ("agent-slot-2" before "agent-slot-10"), as
    ``comparePublicAgentIds`` in web/visual_debugger/src/choreography-plan.js.

    Parameters
    ----------
    public_agent_id : str
        A public agent ID such as "agent-slot-5" or "3".

    Returns
    -------
    tuple
        One (0, number) or (1, lowercase text) item per digit or text run.
    """
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", public_agent_id)
        if part
    )


def _turn_shapes(
    scene: AuthorizedBattlefieldSceneV1,
    *,
    sensor_public_agent_ids: Sequence[str],
    own_public_agent_id: str,
    turn_start_scene: AuthorizedBattlefieldSceneV1,
    turn_start_sensor_public_agent_ids: Sequence[str],
) -> tuple[AuthorizedTeamSightTurnShapeV1, ...]:
    """Return the shapes the veil passes through while one turn plays.

    Parameters
    ----------
    scene : AuthorizedBattlefieldSceneV1
        The Agent POV scene at the end of the turn (this frame).
    sensor_public_agent_ids : Sequence[str]
        The team's sensors at the end of the turn.
    own_public_agent_id : str
        The recipient's public ID.
    turn_start_scene : AuthorizedBattlefieldSceneV1
        The same recipient's Agent POV scene at the start of the turn (the
        frame before).
    turn_start_sensor_public_agent_ids : Sequence[str]
        The team's sensors at the start of the turn.

    Returns
    -------
    tuple[AuthorizedTeamSightTurnShapeV1, ...]
        First the turn_start shape (the sensors alive at the start, at their
        start spots). Then one turn_end shape per moment the browser's veil
        can hold while the turn plays, at end-of-turn spots: the glide (every
        sensor alive at the start still lit, the fallers too), then the set
        left after each Fall beat in turn. The frame's own set (survivors plus
        every returner) is the main shape and is not repeated. A faller lights
        from its row in the end scene (where it fell), or from its start spot
        when the end scene has no row for it. Shapes whose outline cannot be
        traced are left out; the browser then uses the nearest shape.

    Notes
    -----
    The browser's beat schedule (choreography-plan.js ``beatSchedule``) puts
    the Fall beats in public-ID order (digits as numbers), 60 ms apart, and
    every Return at one later beat. So with f fallers this gives f + 1 shapes
    (f when nobody falls or returns), not one per combination: a team wipe
    costs six shapes, not 32. Moments that last no time (one Return of
    several at the same instant) are not built; the browser takes the
    nearest shape for them. Only own-team sensors give light (see
    _living_sensors).
    """
    start = _living_sensors(turn_start_scene, turn_start_sensor_public_agent_ids)
    start_ids = {row[0] for row in start}
    end = _living_sensors(scene, sensor_public_agent_ids)
    end_alive = {row[0] for row in end}
    fallers = sorted(start_ids - end_alive, key=_natural_key)
    returners = end_alive - start_ids
    stays = start_ids & end_alive
    # A faller lights from where it fell: its own row in the end scene, which
    # the team may see (own team only), else its start spot.
    start_spot = {row[0]: row for row in start}
    fell_at = {
        row.public_agent_id: (
            row.public_agent_id,
            float(row.position[0]),
            float(row.position[1]),
            float(row.observation_radius),
        )
        for row in scene.agents
        if row.public_agent_id in start_spot and row.relation in ("self", "ally")
    }
    end_spot = {row[0]: row for row in end}

    def lit_sensors(lit: set[str]) -> list[_Sensor]:
        """Return the lit sensors at their end-of-turn spots, in a fixed order."""
        return [
            end_spot.get(key) or fell_at.get(key) or start_spot[key]
            for key in sorted(lit, key=_natural_key)
        ]

    shapes: list[AuthorizedTeamSightTurnShapeV1] = []
    geometry = _sight(scene, start, own_public_agent_id)
    if geometry is not None:
        shapes.append(
            AuthorizedTeamSightTurnShapeV1(
                stage="turn_start",
                lit_public_agent_ids=tuple(sorted(start_ids)),
                lit_rings=geometry[0],
                own_edges=geometry[1],
                allied_edges=geometry[2],
            )
        )
    # The glide, then the set left after each Fall beat, in the browser's order.
    for fallen in range(len(fallers) + 1):
        lit = stays | set(fallers[fallen:])
        if lit == stays | returners:
            continue  # this frame's own set: the main shape
        geometry = _sight(scene, lit_sensors(lit), own_public_agent_id)
        if geometry is None:
            continue
        shapes.append(
            AuthorizedTeamSightTurnShapeV1(
                stage="turn_end",
                lit_public_agent_ids=tuple(sorted(lit)),
                lit_rings=geometry[0],
                own_edges=geometry[1],
                allied_edges=geometry[2],
            )
        )
    return tuple(shapes)


def with_team_sight_v1(
    scene: AuthorizedBattlefieldSceneV1,
    *,
    sensor_public_agent_ids: Sequence[str],
    own_public_agent_id: str,
    turn_start_scene: AuthorizedBattlefieldSceneV1 | None = None,
    turn_start_sensor_public_agent_ids: Sequence[str] = (),
) -> AuthorizedBattlefieldSceneV1:
    """Return an Agent POV scene with its merged team sight attached.

    Parameters
    ----------
    scene : AuthorizedBattlefieldSceneV1
        The Agent POV scene just built (``team_sight`` None).
    sensor_public_agent_ids : Sequence[str]
        As in merged_team_sight_v1.
    own_public_agent_id : str
        As in merged_team_sight_v1.
    turn_start_scene : AuthorizedBattlefieldSceneV1 | None, optional,
        keyword-only
        The same recipient's Agent POV scene at the start of the turn that led
        here. Given, the sight also carries the turn's shapes (see
        _turn_shapes), so the veil changes in step with the turn's beats.
        None (the default; a first frame) means no turn shapes.
    turn_start_sensor_public_agent_ids : Sequence[str], optional,
        keyword-only
        The team's sensors at the start of the turn; read only with
        turn_start_scene. Default empty.

    Returns
    -------
    AuthorizedBattlefieldSceneV1
        A copy with ``team_sight`` set, or the same scene when this frame's
        outline could not be traced (see merged_team_sight_v1).
    """
    sight = merged_team_sight_v1(
        scene,
        sensor_public_agent_ids=sensor_public_agent_ids,
        own_public_agent_id=own_public_agent_id,
    )
    if sight is None:
        return scene
    if turn_start_scene is not None:
        sight = replace(
            sight,
            turn_shapes=_turn_shapes(
                scene,
                sensor_public_agent_ids=sensor_public_agent_ids,
                own_public_agent_id=own_public_agent_id,
                turn_start_scene=turn_start_scene,
                turn_start_sensor_public_agent_ids=turn_start_sensor_public_agent_ids,
            ),
        )
    return replace(scene, team_sight=sight)


__all__ = ["merged_team_sight_v1", "with_team_sight_v1"]
