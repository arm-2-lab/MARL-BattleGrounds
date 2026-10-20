"""Measure how far a body must walk around obstacles: the route field.

A route field covers the map with a regular grid of square cells. Each cell
holds the length of the shortest walk, in map units, from that cell's center to
a target, going around obstacles. Capture the Flag uses it to check that every
flag home can be reached from both ends of each map, for reward shaping on a
carrier's walk home, and in the scripted bots.

One definition serves every use, so all of them agree on what is walkable:

- blocked_cells marks a cell blocked when a body centered there would overlap
  an obstacle (Core's disc_overlaps_obstacle) or reach past the map edge.
- route_distances spreads distances from the targets to the 8 neighboring
  cells. A step costs its length, cell_size straight or cell_size * sqrt(2)
  diagonally, and a diagonal step may not cut the corner of a blocked cell.
- route_distance_at reads a field at any position.

Two grid sizes are used. The fine grid (FINE_CELL_SIZE, 0.1 map units: 100 by
200 cells on a 20 by 10 map) is computed once per map, before training or a
check, and runs until no distance changes. The coarse grid (COARSE_CELL_SIZE,
1.0: 10 by 20 cells) runs a fixed number of passes inside compiled bot code.

Limits of a grid: a walk can only follow 8 directions, so a straight line that
is neither straight across nor diagonal reads up to about 8 percent long. A gap
counts as open only when cell centers fall inside it, so a gap less than about
one body width plus one cell diagonal wide (1.14 map units on the fine grid)
may read as closed. All functions are pure JAX: callers compose them under jit
and vmap, marking the float and iteration arguments static.
"""

from __future__ import annotations

import math
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.geometry import disc_overlaps_obstacle

FINE_CELL_SIZE = 0.1
COARSE_CELL_SIZE = 1.0
# Every class's body radius in core.combat.BODY_RADIUS_BY_CLASS, in map units.
BODY_RADIUS = 0.5
# The 8 neighbor steps as (row, column) offsets; rows run along y, columns along x.
_STEPS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _grid_shape(
    map_width: float, map_height: float, cell_size: float
) -> tuple[int, int]:
    """Return the (rows, columns) count for a map and cell size.

    All three are positive map-unit floats known when the program is built.
    Raise ValueError unless the map width and height are whole multiples of
    cell_size, so every cell lies fully inside the map.
    """
    rows, columns = round(map_height / cell_size), round(map_width / cell_size)
    if (
        cell_size <= 0
        or rows < 1
        or columns < 1
        or not math.isclose(rows * cell_size, map_height)
        or not math.isclose(columns * cell_size, map_width)
    ):
        raise ValueError("map width and height must be whole multiples of cell_size")
    return rows, columns


def _cell_centers(rows: int, columns: int, cell_size: float) -> Array:
    """Return every cell center as float32 (rows, columns, 2) [x, y] in map units."""
    x = (jnp.arange(columns, dtype=jnp.float32) + 0.5) * cell_size
    y = (jnp.arange(rows, dtype=jnp.float32) + 0.5) * cell_size
    return jnp.stack(jnp.meshgrid(x, y), axis=-1)


def _neighborhood(
    position: Array, rows: int, columns: int, cell_size: float
) -> tuple[Array, Array, Array]:
    """Return the 3 by 3 block of cells around the cell that holds a position.

    position is float32 (2,) [x, y] in map units. Return the block's row and
    column indices, each int32 (9,), clipped to the grid (an edge cell can
    repeat), and the float32 (9,) distance from position to each cell center.
    """
    row = jnp.floor(position[1] / cell_size).astype(jnp.int32)
    column = jnp.floor(position[0] / cell_size).astype(jnp.int32)
    offsets = jnp.arange(-1, 2, dtype=jnp.int32)
    block_rows = jnp.clip(row + offsets[:, None], 0, rows - 1).repeat(3, axis=1).ravel()
    block_columns = (
        jnp.clip(column + offsets[None, :], 0, columns - 1).repeat(3, axis=0).ravel()
    )
    centers = (jnp.stack((block_columns, block_rows), axis=-1) + 0.5) * cell_size
    offsets_to_centers = centers - position
    distances = jnp.hypot(offsets_to_centers[:, 0], offsets_to_centers[:, 1])
    return block_rows, block_columns, distances


def blocked_cells(
    obstacles: Array,
    *,
    map_width: float,
    map_height: float,
    cell_size: float,
    body_radius: float = BODY_RADIUS,
) -> Array:
    """Mark the grid cells where a body's center cannot stand.

    Parameters
    ----------
    obstacles : jax.Array
        Float32 (obstacle slots, 8) rows in EnvConfig.obstacles columns, such
        as a config's ``obstacles``. Inactive rows block nothing.
    map_width, map_height : float
        Map size in map units, 20.0 and 10.0 on every installed map. Each must
        be a whole multiple of cell_size.
    cell_size : float
        Side of one square cell in map units: FINE_CELL_SIZE or
        COARSE_CELL_SIZE.
    body_radius : float, default=BODY_RADIUS
        Radius of the walking body in map units (0.5 for every class).

    Returns
    -------
    jax.Array
        Bool (rows, columns), rows = map_height / cell_size along y and
        columns = map_width / cell_size along x; cell (r, c) is centered at
        ((c + 0.5) * cell_size, (r + 0.5) * cell_size). True where a body
        centered on the cell would overlap an active obstacle or stand closer
        than body_radius to a map edge. Touching an obstacle or an edge
        exactly is allowed, as in the game.

    Raises
    ------
    ValueError
        If the map size is not a whole multiple of cell_size.

    Notes
    -----
    Overlap is Core's disc_overlaps_obstacle, the rule the simulator uses. The
    four floats set the grid's shape, so mark them static under jit (each new
    value builds a new program); obstacle values stay dynamic.
    """
    rows, columns = _grid_shape(map_width, map_height, cell_size)
    centers = _cell_centers(rows, columns, cell_size).reshape(-1, 2)

    def overlaps_any(center: Array) -> Array:
        return jax.vmap(disc_overlaps_obstacle, in_axes=(None, None, 0))(
            center, body_radius, obstacles
        ).any()

    x, y = centers[:, 0], centers[:, 1]
    off_map = (
        (x < body_radius)
        | (x > map_width - body_radius)
        | (y < body_radius)
        | (y > map_height - body_radius)
    )
    return (jax.vmap(overlaps_any)(centers) | off_map).reshape(rows, columns)


def route_distances(
    blocked: Array,
    targets: Array,
    *,
    cell_size: float,
    iterations: int | None = None,
) -> Array:
    """Compute each cell's walking distance to each target.

    Parameters
    ----------
    blocked : jax.Array
        Bool (rows, columns) from blocked_cells, built with the same cell_size.
    targets : jax.Array
        Float32 (targets, 2) target positions [x, y] in map units.
    cell_size : float
        The cell side blocked was built with, in map units.
    iterations : int or None, default=None
        None runs until no distance changes, which gives exact grid distances;
        use it for the fine field computed once per map (on a GPU, each pass
        then reads one value back to the host). An int runs exactly that many
        passes, for compiled code that needs a fixed cost: each pass carries
        distances one cell further from the 3 by 3 block around each target
        (see Notes), so a cell whose shortest walk needs more steps keeps a
        longer distance or infinity.

    Returns
    -------
    jax.Array
        Float32 (targets, rows, columns) walking distances in map units, from
        each cell's center to each target. Infinity marks a blocked cell or a
        cell no walk reaches.

    Notes
    -----
    A target starts the walk from the open cells in the 3 by 3 block around
    its cell, each at its straight-line distance from the target, so a target
    standing tangent to an obstacle still reaches the grid. On the coarse grid
    that block spans 3 map units and could reach across a wall thinner than
    one cell; the fine grid's block spans 0.3. Under jit, mark cell_size and
    iterations static; blocked and targets stay dynamic.
    """
    rows, columns = blocked.shape
    is_open = ~blocked

    def start(target: Array) -> Array:
        block_rows, block_columns, distance = _neighborhood(
            target, rows, columns, cell_size
        )
        distance = jnp.where(is_open[block_rows, block_columns], distance, jnp.inf)
        return (
            jnp.full((rows, columns), jnp.inf)
            .at[block_rows, block_columns]
            .min(distance)
        )

    padded_open = jnp.pad(is_open, 1)

    def neighbor(values: Array, row_step: int, column_step: int) -> Array:
        return values[
            ...,
            1 + row_step : 1 + row_step + rows,
            1 + column_step : 1 + column_step + columns,
        ]

    # A step is allowed from an open cell to an open neighbor; a diagonal step
    # also needs both cells it passes between to be open.
    allowed = [
        is_open
        & neighbor(padded_open, row_step, column_step)
        & neighbor(padded_open, row_step, 0)
        & neighbor(padded_open, 0, column_step)
        for row_step, column_step in _STEPS
    ]
    costs = [
        cell_size * math.hypot(row_step, column_step)
        for row_step, column_step in _STEPS
    ]

    def relax(distances: Array) -> Array:
        padded = jnp.pad(distances, ((0, 0), (1, 1), (1, 1)), constant_values=jnp.inf)
        for (row_step, column_step), step_allowed, cost in zip(
            _STEPS, allowed, costs, strict=True
        ):
            through = neighbor(padded, row_step, column_step) + cost
            distances = jnp.minimum(
                distances, jnp.where(step_allowed, through, jnp.inf)
            )
        return distances

    distances = jax.vmap(start)(targets)
    if iterations is not None:

        def relax_pass(_: int, values: Array) -> Array:
            return relax(values)

        return cast(Array, jax.lax.fori_loop(0, iterations, relax_pass, distances))

    def changed(carry: tuple[Array, Array]) -> Array:
        return carry[1]

    def step(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        relaxed = relax(carry[0])
        return relaxed, jnp.any(relaxed < carry[0])

    return jax.lax.while_loop(changed, step, (distances, jnp.array(True)))[0]


def route_distance_at(distances: Array, position: Array, *, cell_size: float) -> Array:
    """Read one route field at one position.

    Parameters
    ----------
    distances : jax.Array
        Float32 (rows, columns): one target's field from route_distances.
    position : jax.Array
        Float32 (2,) [x, y] in map units, such as an agent's position.
    cell_size : float
        The field's cell side, in map units.

    Returns
    -------
    jax.Array
        Float32 scalar: the walking distance from position to the field's
        target, in map units, through the best of the 3 by 3 cells around
        position (that cell's distance plus the straight line to its center).
        Infinity when none of those cells reaches the target.

    Notes
    -----
    Reading through the neighbors keeps a body standing tangent to an
    obstacle, whose own cell can be blocked, on the field. For many fields or
    positions, use vmap; under jit, mark cell_size static.
    """
    rows, columns = distances.shape
    block_rows, block_columns, offset = _neighborhood(
        position, rows, columns, cell_size
    )
    return jnp.min(distances[block_rows, block_columns] + offset)
