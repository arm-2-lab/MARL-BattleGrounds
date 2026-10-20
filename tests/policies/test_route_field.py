"""Check the route field: which grid cells are blocked and how far a body walks.

Contracts: blocked_cells blocks the band within one body radius of each map
edge and exactly the cells where Core's disc_overlaps_obstacle says a body
would overlap an obstacle, while a body touching an edge exactly stands;
route_distances gives exact lengths along grid lines and diagonals, walks
around a wall rather than through it, never cuts the corner between two
blocked cells, leaves a sealed region and blocked cells at infinity, and with
a fixed number of passes reaches exactly as far as those passes carry it;
route_distance_at reads a field at a body position whose own cell is blocked;
the field builds under jit with traced obstacles and targets.
"""

import math
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.core.types import (
    MAX_OBSTACLE_SLOTS,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
)
from marl_battlegrounds.policies.route_field import (
    COARSE_CELL_SIZE,
    FINE_CELL_SIZE,
    blocked_cells,
    route_distance_at,
    route_distances,
)

MAP = {"map_width": 20.0, "map_height": 10.0}


def _obstacles(*rows: tuple[float, ...]) -> Array:
    # Each row: (type, x, y, radius, width, height, theta); this sets active.
    table = np.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=np.float32)
    for index, row in enumerate(rows):
        table[index, :7] = row
        table[index, 7] = 1.0
    return jnp.asarray(table)


def _field(
    obstacles: Array, target: tuple[float, float], cell_size: float, **kwargs: int
) -> np.ndarray:
    blocked = blocked_cells(obstacles, cell_size=cell_size, **MAP)
    return np.asarray(
        route_distances(blocked, jnp.asarray([target]), cell_size=cell_size, **kwargs)[
            0
        ]
    )


def test_edges_and_obstacles_block_exactly_what_core_forbids() -> None:
    empty = _obstacles()
    assert not np.asarray(blocked_cells(empty, cell_size=COARSE_CELL_SIZE, **MAP)).any()
    fine = np.asarray(blocked_cells(empty, cell_size=FINE_CELL_SIZE, **MAP))
    x = (np.arange(200) + 0.5) * FINE_CELL_SIZE
    assert np.array_equal(fine[50], (x < 0.5) | (x > 19.5))
    pillar = np.asarray(
        blocked_cells(
            _obstacles((OBSTACLE_TYPE_PILLAR, 10, 5, 1, 0, 0, 0)),
            cell_size=FINE_CELL_SIZE,
            **MAP,
        )
    )
    centers_x, centers_y = np.meshgrid(x, (np.arange(100) + 0.5) * FINE_CELL_SIZE)
    assert np.array_equal(
        pillar, fine | (np.hypot(centers_x - 10, centers_y - 5) < 1.5)
    )


def test_open_ground_lengths_are_exact_on_lines_and_diagonals() -> None:
    field = _field(_obstacles(), (3.5, 2.5), COARSE_CELL_SIZE)
    assert field[2, 3] == pytest.approx(0.0)
    assert field[2, 13] == pytest.approx(10.0)
    assert field[7, 8] == pytest.approx(5 * math.sqrt(2))


def test_a_walk_goes_around_a_wall_and_never_cuts_a_blocked_corner() -> None:
    wall = _obstacles((OBSTACLE_TYPE_WALL, 10, 4, 0, 0.2, 8, 0))
    detour = _field(wall, (5.5, 4.5), FINE_CELL_SIZE)[45, 145]
    # The walk must pass above the wall's end, grown to y = 8.5.
    shortest = 2 * math.hypot(4.5, 4.0)
    assert shortest <= detour <= shortest * 1.09
    # Small pillars block only the cells (row 4, column 6) and (row 5, column 5);
    # cutting their corner would reach (row 5, column 6) at 2 + sqrt(2).
    pinch = _obstacles(
        (OBSTACLE_TYPE_PILLAR, 6.5, 4.5, 0.1, 0, 0, 0),
        (OBSTACLE_TYPE_PILLAR, 5.5, 5.5, 0.1, 0, 0, 0),
    )
    around = _field(pinch, (3.5, 4.5), COARSE_CELL_SIZE)[5, 6]
    assert 2 + math.sqrt(2) + 1e-6 < around < math.inf


def test_sealed_cells_and_blocked_cells_stay_unreachable() -> None:
    box = _obstacles(
        (OBSTACLE_TYPE_WALL, 10, 3, 0, 4.2, 0.2, 0),
        (OBSTACLE_TYPE_WALL, 10, 7, 0, 4.2, 0.2, 0),
        (OBSTACLE_TYPE_WALL, 8, 5, 0, 0.2, 4.2, 0),
        (OBSTACLE_TYPE_WALL, 12, 5, 0, 0.2, 4.2, 0),
    )
    field = _field(box, (2.5, 2.5), FINE_CELL_SIZE)
    assert np.isinf(field[50, 100])
    assert np.isinf(field[30, 100])
    assert np.isfinite(field[90, 100])


def test_fixed_passes_reach_exactly_as_far_as_they_carry_the_walk() -> None:
    empty = _obstacles()
    five = _field(empty, (0.5, 0.5), COARSE_CELL_SIZE, iterations=5)
    # The target seeds its 3 by 3 block, so 5 passes reach column 6.
    assert five[0, 6] == pytest.approx(6.0)
    assert np.isinf(five[0, 7])
    converged = _field(empty, (0.5, 0.5), COARSE_CELL_SIZE)
    assert np.array_equal(
        _field(empty, (0.5, 0.5), COARSE_CELL_SIZE, iterations=19), converged
    )


def test_a_body_tangent_to_a_pillar_reads_the_field() -> None:
    pillar = _obstacles((OBSTACLE_TYPE_PILLAR, 10, 5, 1, 0, 0, 0))
    blocked = blocked_cells(pillar, cell_size=FINE_CELL_SIZE, **MAP)
    offset = 1.501 / math.sqrt(2)  # just outside the pillar's 1.5 reach
    x, y = 10 + offset, 5 + offset
    position = jnp.asarray([x, y])
    assert bool(blocked[60, 110])  # the body's own cell
    field = route_distances(
        blocked, jnp.asarray([[15.05, 6.05]]), cell_size=FINE_CELL_SIZE
    )[0]
    walked = float(route_distance_at(field, position, cell_size=FINE_CELL_SIZE))
    assert walked == pytest.approx(math.hypot(15.05 - x, 6.05 - y), abs=0.2)


def test_the_field_builds_under_jit_with_traced_obstacles_and_targets() -> None:
    def field(obstacles: Array, targets: Array) -> Array:
        blocked = blocked_cells(obstacles, cell_size=COARSE_CELL_SIZE, **MAP)
        return route_distances(
            blocked, targets, cell_size=COARSE_CELL_SIZE, iterations=25
        )

    pillar = _obstacles((OBSTACLE_TYPE_PILLAR, 4, 4, 1, 0, 0, 0))
    targets = jnp.asarray([[9.5, 3.5]])
    compiled = cast(Array, jax.jit(field)(pillar, targets))
    assert np.array_equal(np.asarray(compiled), np.asarray(field(pillar, targets)))
