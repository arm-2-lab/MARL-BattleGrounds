"""Find where a Capture the Flag bot should walk next, from tables built once per map.

A bot always walks toward a goal point: a flag home, a flag wherever it lies, a
carrier, an enemy or a fixed spot. When a body can slide straight to the goal, by
Core's own movement test, the bot heads straight there. Otherwise it heads for a
node of the map's corner graph:

- The nodes sit just outside the obstacles: three around each wall corner and
  eight around each pillar, kept only where a body fits.
- Two nodes are joined when a body can slide straight from one to the other, and
  the shortest walk between every pair of nodes is stored.
- Every 0.1-unit cell keeps the 16 nearest nodes a body can slide to straight from
  anywhere in the cell (its center and each corner where a body fits), and every
  node keeps its walk to each flag home.

A route is then the best of at most 16 by 16 sums: from the body to a node it
sees, along the stored walk, to a node the goal sees, to the goal. Flag homes use
the nodes' stored walks home instead of the goal's 16, so walks home are as exact
as the whole graph.

build_navigation_tables runs on the host, about 1.5 seconds a map on a GPU, and
returns NavigationTables: a tree of arrays the bots receive as their variables, so
a new set of maps needs no new compile. load_navigation_tables keeps a built set
in the user's cache, named by its inputs (navigation_tables_identity), so a later
process loads it instead of building it. find_map and route run inside compiled
code, one actor at a time; they read only the actor's own position, its goal and
the public obstacle table every observation carries, and keep no memory.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import ndimage  # pyright: ignore[reportMissingTypeStubs]

from marl_battlegrounds.core.geometry import (
    _swept_fraction,  # pyright: ignore[reportPrivateUsage]
    disc_overlaps_obstacle,
)
from marl_battlegrounds.core.types import (
    OBSTACLE_TYPE_PILLAR,
    EnvConfig,
)
from marl_battlegrounds.policies.route_field import (
    BODY_RADIUS,
    FINE_CELL_SIZE,
    blocked_cells,
)

NODES_PER_CELL = 16
"""Nodes each cell keeps: its nearest ones a body can slide to straight."""
NODE_MARGIN = 0.02
"""Extra clearance, in map units, between a node's body and the obstacle it rounds."""
PASSED_NODE_DISTANCE = 0.25
"""A node nearer the body than this, in map units, counts as reached, so the bot
heads for the next node of the walk instead."""
CLEAR_FRACTION = 0.9999
"""A body that can slide at least this share of a segment counts as sliding all of
it. It absorbs float32 rounding where a goal sits exactly one body radius from the
map edge, as the flag homes do; at 20 map units it is 2 millimeters."""
TABLE_BUILDER_VERSION = 1
"""The version of build_navigation_tables' rules; part of a table set's identity."""
_PILLAR_SIDES = 8
_SEGMENTS_PER_CALL = 65536


class NavigationTables(NamedTuple):
    """Hold the corner graph of every map a bot can recognize.

    Attributes
    ----------
    layouts : Array
        Float32 (M, 32, 8): each map's obstacle table, EnvConfig.obstacles rows.
        find_map compares an observed table with these, exactly.
    homes : Array
        Float32 (M, 2, 2): each map's flag homes in map units, Team A's first.
    nodes : Array
        Float32 (M, N, 2): node positions in map units. N is the largest node
        count among the maps; a map's unused rows are zero and never reached.
    walks : Array
        Float32 (M, N, N): the shortest walk between two nodes along the graph,
        in map units; infinity where none exists, unused rows included.
    next_nodes : Array
        Int16 (M, N, N): the node after the first one on the shortest walk from
        one node to another (the other node itself when they see each other);
        -1 where no walk exists.
    home_walks : Array
        Float32 (M, N, 2): each node's shortest walk to each flag home, through
        any node that sees the home; infinity where none exists.
    next_home_nodes : Array
        Int16 (M, N, 2): the node after a node on its shortest walk to each flag
        home; -1 when the node sees the home itself or no walk exists.
    cell_nodes : Array
        Int16 (M, rows, columns, NODES_PER_CELL): for each 0.1-unit cell (rows
        along y, columns along x, as route_field's fine grid), the indices of the
        nearest nodes to the cell's center that a body can slide to straight
        from that center and from each corner of the cell where a body fits
        (from the center alone when no node passes that test), nearest first;
        -1 fills the slots when it sees fewer. A cell whose
        center no body fits in takes the list of the nearest cell where one does.
    cell_home_nodes : Array
        Int16 (M, rows, columns, 2): for each cell and each flag home, the node
        that starts the shortest walk home, chosen from every node the cell sees
        by the same rule (not only its 16); -1 when none does. So walks home,
        the bots' most common trips, are as exact as the whole graph.

    Notes
    -----
    A NamedTuple of arrays is a JAX tree, so the tables pass as a bot's
    variables. All maps in one set share one size.
    """

    layouts: Array
    homes: Array
    nodes: Array
    walks: Array
    next_nodes: Array
    home_walks: Array
    next_home_nodes: Array
    cell_nodes: Array
    cell_home_nodes: Array


def _corner_nodes(obstacles: np.ndarray) -> np.ndarray:
    """Return candidate node positions around every active obstacle.

    obstacles is float32 (32, 8), EnvConfig.obstacles rows. Each wall corner
    gets three nodes on the rays at 0, 45 and 90 degrees from the corner within
    its outward quadrant (the wall's own axes), at (BODY_RADIUS + NODE_MARGIN)
    / cos(22.5 degrees), so the edges between them stay outside the corner's
    rounded clearance. Each pillar gets _PILLAR_SIDES nodes on the polygon whose
    sides lie BODY_RADIUS + NODE_MARGIN outside its circle. Returns float32
    (nodes, 2) world points, before blocked ones are dropped.
    """
    points: list[tuple[float, float]] = []
    corner_reach = (BODY_RADIUS + NODE_MARGIN) / math.cos(math.pi / 8)
    for kind, x, y, radius, width, height, theta, active in obstacles.tolist():
        if active != 1.0:
            continue
        if kind == OBSTACLE_TYPE_PILLAR:
            reach = (radius + BODY_RADIUS + NODE_MARGIN) / math.cos(
                math.pi / _PILLAR_SIDES
            )
            for k in range(_PILLAR_SIDES):
                angle = math.pi / _PILLAR_SIDES * (1 + 2 * k)
                points.append(
                    (x + reach * math.cos(angle), y + reach * math.sin(angle))
                )
            continue
        cosine, sine = math.cos(theta), math.sin(theta)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for angle in (0.0, math.pi / 4, math.pi / 2):
                    u = sx * (width / 2 + corner_reach * math.cos(angle))
                    v = sy * (height / 2 + corner_reach * math.sin(angle))
                    points.append(
                        (x + cosine * u - sine * v, y + sine * u + cosine * v)
                    )
    return np.asarray(points, np.float32).reshape(-1, 2)


def _blocked_points(
    points: np.ndarray, obstacles: np.ndarray, width: float, height: float
) -> np.ndarray:
    """Return True where no body fits at a point.

    points is float32 (P, 2) in map units; obstacles is float32 (32, 8). A body
    does not fit where it overlaps an obstacle (Core's disc_overlaps_obstacle)
    or reaches past the map edge, as route_field.blocked_cells decides for cell
    centers. Returns bool (P,).
    """
    table = jnp.asarray(obstacles)

    def overlaps_any(point: Array) -> Array:
        """Return whether a body at one point overlaps any obstacle row."""

        def overlaps(row: Array) -> Array:
            """Return whether a body at the point overlaps one obstacle row."""
            return disc_overlaps_obstacle(point, BODY_RADIUS, row)

        return jnp.any(jax.vmap(overlaps)(table))

    overlaps = jax.vmap(overlaps_any)(jnp.asarray(points))
    inside = (
        (points[:, 0] >= BODY_RADIUS)
        & (points[:, 0] <= width - BODY_RADIUS)
        & (points[:, 1] >= BODY_RADIUS)
        & (points[:, 1] <= height - BODY_RADIUS)
    )
    return np.asarray(overlaps) | ~inside


@jax.jit
def _slides_clear(
    starts: Array, ends: Array, obstacles: Array, width: Array, height: Array
) -> Array:
    """Return True where a body slides the whole way from start to end.

    starts and ends are float32 (S, 2) in map units; obstacles is float32
    (32, 8); width and height are the map size. It is Core's own movement test
    (geometry._swept_fraction, the first contact along the segment with an
    obstacle or the map edge), so a route the bot takes is one Core lets a body
    walk; CLEAR_FRACTION absorbs rounding at the end. Returns bool (S,).
    """
    radii = jnp.full(starts.shape[:1], BODY_RADIUS, jnp.float32)
    fraction = _swept_fraction(starts, ends, radii, width, height, obstacles)
    return fraction >= CLEAR_FRACTION


def _clear_segments(
    starts: np.ndarray,
    ends: np.ndarray,
    obstacles: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    """Apply _slides_clear to any number of segments on the host.

    starts and ends are float32 (S, 2). Segments run in calls of one fixed
    size, the last one padded, so the test compiles once. Returns bool (S,).
    """
    table = jnp.asarray(obstacles)
    size = (jnp.float32(width), jnp.float32(height))
    results: list[np.ndarray] = []
    for begin in range(0, len(starts), _SEGMENTS_PER_CALL):
        a = starts[begin : begin + _SEGMENTS_PER_CALL]
        b = ends[begin : begin + _SEGMENTS_PER_CALL]
        pad = ((0, _SEGMENTS_PER_CALL - len(a)), (0, 0))
        clear = cast(
            Array,
            _slides_clear(
                jnp.asarray(np.pad(a, pad)), jnp.asarray(np.pad(b, pad)), table, *size
            ),
        )
        results.append(np.asarray(clear)[: len(a)])
    return np.concatenate(results) if results else np.zeros(0, np.bool_)


def _sight_lengths(
    points: np.ndarray,
    targets: np.ndarray,
    obstacles: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    """Return each point's straight length to each target it can slide to.

    points is float32 (P, 2) and targets float32 (T, 2). Returns float32 (P, T):
    the straight length in map units, or infinity where a body cannot slide
    straight from the point to the target.
    """
    starts = np.repeat(points, len(targets), axis=0)
    ends = np.tile(targets, (len(points), 1))
    clear = _clear_segments(starts, ends, obstacles, width, height)
    lengths = np.linalg.norm(points[:, None] - targets[None], axis=-1)
    return np.where(clear.reshape(len(points), len(targets)), lengths, np.inf).astype(
        np.float32
    )


def _map_tables(
    config: EnvConfig,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Build one map's nodes, node walks, walks home and cell lists.

    config is one scalar configuration. Returns, as NavigationTables describes
    them for one map: float32 nodes (n, 2), float32 walks (n, n), int16 next
    nodes (n, n), float32 home walks (n, 2), int16 next home nodes (n, 2),
    int16 cell lists (rows, columns, NODES_PER_CELL) and int16 cell home nodes
    (rows, columns, 2).
    """
    obstacles = np.asarray(config.obstacles, np.float32)
    width, height = float(config.map_width), float(config.map_height)
    homes = np.asarray(config.capture_the_flag_flag_home_positions, np.float32)
    nodes = _corner_nodes(obstacles)
    nodes = nodes[~_blocked_points(nodes, obstacles, width, height)]
    walks = _sight_lengths(nodes, nodes, obstacles, width, height)
    np.fill_diagonal(walks, 0.0)
    count = len(nodes)
    targets = np.broadcast_to(np.arange(count), (count, count))
    next_nodes = np.where(np.isfinite(walks), targets, -1)
    for k in range(count):  # Floyd-Warshall, keeping each walk's next node.
        through = walks[:, k, None] + walks[None, k, :]
        shorter = through < walks
        walks = np.where(shorter, through, walks)
        next_nodes = np.where(shorter, next_nodes[:, k, None], next_nodes)
    to_home = _sight_lengths(nodes, homes, obstacles, width, height)
    via = walks[:, :, None] + to_home[None, :, :]  # (n, last node, home)
    home_walks = np.min(via, axis=1, initial=np.inf)
    last = np.argmin(via, axis=1) if count else np.zeros((0, 2), np.int64)
    rows_index = np.arange(count)[:, None]
    next_home_nodes = np.where(
        np.isfinite(home_walks) & (last != rows_index),
        next_nodes[rows_index, last],
        -1,
    )

    blocked = np.asarray(
        blocked_cells(
            jnp.asarray(obstacles),
            map_width=width,
            map_height=height,
            cell_size=FINE_CELL_SIZE,
        )
    )
    rows, columns = blocked.shape
    # A node counts for a cell only if a body slides to it straight from the
    # cell's center and from each of the cell's corners where a body fits, so a
    # body anywhere in the cell can head for it. Neighboring cells share corners.
    corner_y, corner_x = np.indices((rows + 1, columns + 1))
    corners = (
        np.stack([corner_x.ravel(), corner_y.ravel()], 1) * FINE_CELL_SIZE
    ).astype(np.float32)
    corner_open = ~_blocked_points(corners, obstacles, width, height)
    corner_sees = np.ones((len(corners), len(nodes)), np.bool_)
    corner_sees[corner_open] = np.isfinite(
        _sight_lengths(corners[corner_open], nodes, obstacles, width, height)
    )
    corner_sees = corner_sees.reshape(rows + 1, columns + 1, len(nodes))
    cell_corners_see = (
        corner_sees[:-1, :-1]
        & corner_sees[1:, :-1]
        & corner_sees[:-1, 1:]
        & corner_sees[1:, 1:]
    )
    row_index, column_index = np.nonzero(~blocked)
    open_centers = np.stack(
        [(column_index + 0.5) * FINE_CELL_SIZE, (row_index + 0.5) * FINE_CELL_SIZE], 1
    ).astype(np.float32)
    center_sight = _sight_lengths(open_centers, nodes, obstacles, width, height)
    sight = np.where(cell_corners_see[row_index, column_index], center_sight, np.inf)
    # A cell from all of whose corners no node is seen (a pocket) keeps the nodes
    # its center sees, so it still has a walk.
    pocket = ~np.isfinite(sight).any(axis=1)
    sight[pocket] = center_sight[pocket]
    padded = np.pad(
        sight,
        ((0, 0), (0, max(0, NODES_PER_CELL - len(nodes)))),
        constant_values=np.inf,
    )
    nearest = np.argsort(padded, axis=1, kind="stable")[:, :NODES_PER_CELL]
    seen = np.isfinite(np.take_along_axis(padded, nearest, axis=1))
    cells = np.full((rows, columns, NODES_PER_CELL), -1, np.int16)
    cells[row_index, column_index] = np.where(seen, nearest, -1)
    home_via = sight[:, :, None] + home_walks[None, :, :]  # (open cells, n, home)
    home_first = (
        np.argmin(home_via, axis=1)
        if len(nodes)
        else np.zeros((len(sight), 2), np.int64)
    )
    cell_homes = np.full((rows, columns, 2), -1, np.int16)
    cell_homes[row_index, column_index] = np.where(
        np.isfinite(np.min(home_via, axis=1, initial=np.inf)), home_first, -1
    )
    if blocked.any() and not blocked.all():
        # A body near an obstacle can stand in a cell whose center it does not
        # fit in; that cell takes the list of the nearest open cell.
        source = cast(
            np.ndarray,
            ndimage.distance_transform_edt(
                blocked, return_distances=False, return_indices=True
            ),
        )
        cells = cells[source[0], source[1]]
        cell_homes = cell_homes[source[0], source[1]]
    return (
        nodes,
        walks.astype(np.float32),
        next_nodes.astype(np.int16),
        home_walks.astype(np.float32),
        next_home_nodes.astype(np.int16),
        cells,
        cell_homes,
    )


def build_navigation_tables(configs: Sequence[EnvConfig]) -> NavigationTables:
    """Build the corner graph of each map a bot should recognize, on the host.

    Parameters
    ----------
    configs : sequence of EnvConfig
        One scalar configuration per map, with distinct obstacle tables and one
        map size. Only each one's obstacles, map size and flag homes are read.

    Returns
    -------
    NavigationTables
        JAX arrays on the default device, maps in the given order, padded to
        the largest node count.

    Raises
    ------
    ValueError
        configs is empty, the maps differ in size, or two maps share an
        obstacle table (find_map could not tell them apart).

    Notes
    -----
    Host-only; it compiles Core's movement test once and runs it in fixed-size
    calls, on the default JAX device. A map with no obstacles has no nodes: the
    bot always walks straight there. Writes no files.
    """
    if not configs:
        raise ValueError("Navigation tables need at least one map")
    sizes = {(float(c.map_width), float(c.map_height)) for c in configs}
    if len(sizes) != 1:
        raise ValueError("Navigation tables need maps of one size")
    layouts = np.stack([np.asarray(c.obstacles, np.float32) for c in configs])
    if len({layout.tobytes() for layout in layouts}) != len(layouts):
        raise ValueError("Two navigation maps share an obstacle table")
    built = [_map_tables(config) for config in configs]
    count = max(1, *(len(parts[0]) for parts in built))
    nodes = np.zeros((len(built), count, 2), np.float32)
    walks = np.full((len(built), count, count), np.inf, np.float32)
    next_nodes = np.full((len(built), count, count), -1, np.int16)
    home_walks = np.full((len(built), count, 2), np.inf, np.float32)
    next_home_nodes = np.full((len(built), count, 2), -1, np.int16)
    for index, (
        map_nodes,
        map_walks,
        map_next,
        map_home,
        map_next_home,
        *_,
    ) in enumerate(built):
        size = len(map_nodes)
        nodes[index, :size] = map_nodes
        walks[index, :size, :size] = map_walks
        next_nodes[index, :size, :size] = map_next
        home_walks[index, :size] = map_home
        next_home_nodes[index, :size] = map_next_home
    return NavigationTables(
        layouts=jnp.asarray(layouts),
        homes=jnp.asarray(
            [
                np.asarray(c.capture_the_flag_flag_home_positions, np.float32)
                for c in configs
            ]
        ),
        nodes=jnp.asarray(nodes),
        walks=jnp.asarray(walks),
        next_nodes=jnp.asarray(next_nodes),
        home_walks=jnp.asarray(home_walks),
        next_home_nodes=jnp.asarray(next_home_nodes),
        cell_nodes=jnp.asarray(np.stack([parts[5] for parts in built])),
        cell_home_nodes=jnp.asarray(np.stack([parts[6] for parts in built])),
    )


def navigation_tables_identity(configs: Sequence[EnvConfig]) -> str:
    """Name a table set by what it is built from.

    Parameters
    ----------
    configs : sequence of EnvConfig
        The maps, in table order, as build_navigation_tables takes them.

    Returns
    -------
    str
        A SHA-256 hex digest of TABLE_BUILDER_VERSION and, per map, its obstacle
        table, flag homes and size as float32 bytes. The same rules and the same
        maps give the same name on any machine, even where the built floats
        differ in their last digits.
    """
    digest = hashlib.sha256(f"ctf-bot-navigation:{TABLE_BUILDER_VERSION}".encode())
    for config in configs:
        for part in (
            config.obstacles,
            config.capture_the_flag_flag_home_positions,
            (config.map_width, config.map_height),
        ):
            digest.update(np.asarray(part, np.float32).tobytes())
    return digest.hexdigest()


def _arrays_digest(arrays: Sequence[np.ndarray]) -> str:
    """Return the SHA-256 hex digest of arrays' dtypes, shapes and bytes, in order."""
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_navigation_tables(
    configs: Sequence[EnvConfig], cache_dir: Path | None = None
) -> NavigationTables:
    """Load a table set from the user's cache, building and saving it once.

    Parameters
    ----------
    configs : sequence of EnvConfig
        The maps, as build_navigation_tables takes them.
    cache_dir : pathlib.Path or None, default=None
        The folder that holds built sets. None means the package's user cache
        (evaluation.tournament_assets.default_cache_dir) plus
        "ctf-bot-navigation".

    Returns
    -------
    NavigationTables
        The same arrays build_navigation_tables gives.

    Raises
    ------
    ValueError
        As build_navigation_tables, or a cached file whose arrays no longer match
        the digest saved beside them (a damaged file; delete it to rebuild).

    Notes
    -----
    Host-only. The file is "<navigation_tables_identity>.npz", written to a
    temporary name and then renamed, so a reader never sees half a file; it holds
    the arrays and their digest (_arrays_digest), checked on every load. A set
    for the 52 Capture the Flag maps is about 55 MB.
    """
    if cache_dir is None:
        from marl_battlegrounds.evaluation.tournament_assets import default_cache_dir

        cache_dir = default_cache_dir() / "ctf-bot-navigation"
    path = cache_dir / f"{navigation_tables_identity(configs)}.npz"
    if path.exists():
        with np.load(path) as stored:
            arrays = [stored[name] for name in NavigationTables._fields]
            saved = str(stored["arrays_digest"])
        if _arrays_digest(arrays) != saved:
            raise ValueError(f"Cached navigation tables are damaged: {path}")
        return NavigationTables(*(jnp.asarray(array) for array in arrays))
    tables = build_navigation_tables(configs)
    arrays = [np.asarray(array) for array in tables]
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.tmp.npz")
    named = dict(zip(NavigationTables._fields, arrays, strict=True))
    named["arrays_digest"] = np.asarray(_arrays_digest(arrays))
    np.savez(partial, **named)  # pyright: ignore[reportArgumentType]
    partial.replace(path)
    return tables


def find_map(tables: NavigationTables, obstacles: Array) -> tuple[Array, Array]:
    """Find which of the tables' maps an observed obstacle table belongs to.

    Parameters
    ----------
    tables : NavigationTables
        The bot's tables.
    obstacles : Array
        Float32 (32, 8), the observation's map_obstacle_features row for this
        actor.

    Returns
    -------
    tuple of Array
        Int32 scalar map row, and a bool scalar that is True when a map's table
        equals the observed one exactly. On no match the row is 0, a valid row
        whose graph route never uses: it answers with the straight line.
    """
    same = jnp.all(tables.layouts == obstacles, axis=(1, 2))
    return jnp.argmax(same).astype(jnp.int32), jnp.any(same)


def _length(vectors: Array) -> Array:
    """Return the Euclidean length of vectors along the last axis, in map units."""
    return jnp.sqrt(jnp.sum(jnp.square(vectors), axis=-1))


def _cell_entry(table: Array, row: Array, point: Array) -> Array:
    """Return a per-cell table's entry for the cell holding a point.

    table is (M, rows, columns, ...); the point's cell is clipped to the grid.
    """
    rows, columns = table.shape[1:3]
    cell = jnp.floor(point / FINE_CELL_SIZE).astype(jnp.int32)
    return table[row, jnp.clip(cell[1], 0, rows - 1), jnp.clip(cell[0], 0, columns - 1)]


class Route(NamedTuple):
    """Hold one actor's walk toward one goal.

    Attributes
    ----------
    length : Array
        Float32 scalar, the bot's walk distance in map units: the straight
        length when a body slides straight to the goal or the map is not in the
        tables; otherwise the graph's shortest walk, or infinity when the graph
        finds none. It is the bots' own measure, separate from the reward's
        carrier field (training.reward_recipes), which walks a 0.1-unit grid.
    waypoint : Array
        Float32 (2,), the point to head for next: the goal itself when the walk
        is straight, the map is unknown or the graph finds no walk (the bot's
        ordinary wall steering then takes over), else the walk's first node.
    """

    length: Array
    waypoint: Array


def route(
    tables: NavigationTables,
    map_row: Array,
    known_map: Array,
    position: Array,
    goal: Array,
    obstacles: Array,
    map_size: Array,
) -> Route:
    """Return an actor's walk distance to a goal and the point to head for next.

    Parameters
    ----------
    tables : NavigationTables
        The bot's tables.
    map_row, known_map : Array
        find_map's answer for this game's obstacle table.
    position, goal : Array
        Float32 (2,) in map units: the actor's center and where it wants to be.
    obstacles : Array
        Float32 (32, 8), the observation's obstacle table.
    map_size : Array
        Float32 (2,), the map's width and height.

    Returns
    -------
    Route
        The walk's length and its next point. A graph walk runs from the body to
        a node its cell sees, along the graph, then to the goal through a node
        the goal's cell sees; when the goal is one of the map's flag homes, from
        the body to its cell's home node, then along that node's walk home (as
        exact as the whole graph). When the walk's first node is
        nearer than PASSED_NODE_DISTANCE, the next point is the node after it
        (or the goal, when that node sees it), so the bot never stops at a node.

    Notes
    -----
    Pure JAX for one actor; vmap it over actors. The per-actor cost is one
    straight-slide test against 32 obstacles and one 16 by 16 sum.
    """
    straight = cast(
        Array,
        _slides_clear(position[None], goal[None], obstacles, map_size[0], map_size[1]),
    )[0]
    nodes = tables.nodes[map_row]
    near = _cell_entry(tables.cell_nodes, map_row, position)
    near_index = jnp.maximum(near, 0)
    to_near = jnp.where(near >= 0, _length(nodes[near_index] - position), jnp.inf)
    far = _cell_entry(tables.cell_nodes, map_row, goal)
    far_index = jnp.maximum(far, 0)
    from_far = jnp.where(far >= 0, _length(nodes[far_index] - goal), jnp.inf)
    via = (
        tables.walks[map_row][near_index[:, None], far_index[None, :]]
        + from_far[None, :]
    )
    exit_choice = jnp.argmin(via, axis=1)  # each near node's best goal-side node
    general = to_near + jnp.min(via, axis=1)
    best = jnp.argmin(general)
    exit_node = far_index[exit_choice[best]]
    is_home = jnp.all(jnp.abs(tables.homes[map_row] - goal) < 1e-3, axis=-1)
    going_home = jnp.any(is_home)
    home = jnp.argmax(is_home)
    home_entry = _cell_entry(tables.cell_home_nodes, map_row, position)[home]
    home_node = jnp.maximum(home_entry, 0)
    home_length = jnp.where(
        home_entry >= 0,
        _length(nodes[home_node] - position)
        + tables.home_walks[map_row][home_node, home],
        jnp.inf,
    )
    entry = jnp.where(going_home, home_node, near_index[best])
    walk_length = jnp.where(going_home, home_length, general[best])
    # At a node already: head for the walk's next node, or for the goal itself
    # when this node sees it.
    after = jnp.where(
        going_home,
        tables.next_home_nodes[map_row][entry, home],
        jnp.where(entry == exit_node, -1, tables.next_nodes[map_row][entry, exit_node]),
    )
    onward = jnp.where(after >= 0, nodes[jnp.maximum(after, 0)], goal)
    passed = _length(nodes[entry] - position) < PASSED_NODE_DISTANCE
    direct = straight | ~known_map
    use_graph = ~direct & jnp.isfinite(walk_length)
    return Route(
        length=jnp.where(direct, _length(goal - position), walk_length),
        waypoint=jnp.where(use_graph, jnp.where(passed, onward, nodes[entry]), goal),
    )
