"""Check the Capture the Flag bots' navigation tables and routes.

Contracts: a clear line goes straight; a wall between body and goal sends the bot to
a node beside the wall, with a walk length close to the true shortest walk around it;
a body already at a node heads for the next node of its walk, never back to the node;
a walk to a flag home through the stored home walks equals the general walk to the
same point; an obstacle table outside the tables (an unknown map) walks straight and
reports itself unknown; a map with no obstacles has no nodes and always walks straight;
on real maps the bots' walk distance home agrees with the reward's carrier field within
that field's own grid error, except where the field closes a passage just over a body
wide, which on map 11 makes the bots' walk shorter (the reward's field is a separate
measure); and the builder refuses an empty set, maps of different sizes and two maps
with one obstacle table.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.modes.capture_the_flag.bots import navigation
from marl_battlegrounds.policies.route_field import FINE_CELL_SIZE, blocked_cells
from marl_battlegrounds.tasks import make_standard_capture_the_flag_config
from marl_battlegrounds.training import reward_recipes

_SIZE = jnp.asarray([20.0, 10.0], jnp.float32)
_Walled = tuple[navigation.NavigationTables, EnvConfig]


def _map(map_id: int) -> EnvConfig:
    return make_standard_capture_the_flag_config(
        map_id=map_id, team_a_roster=("warrior",), team_b_roster=("warrior",)
    )


def _with_wall(x: float) -> EnvConfig:
    # Map 0 is an open arena; one wall 1 wide and 6 tall stands at x, from y 2 to 8.
    config = _map(0)
    obstacles = np.asarray(config.obstacles, np.float32).copy()
    obstacles[0] = (2, x, 5.0, 0, 1.0, 6.0, 0, 1)
    return config._replace(obstacles=jnp.asarray(obstacles))


@pytest.fixture(scope="module")
def walled() -> _Walled:
    config = _with_wall(10.0)
    return navigation.build_navigation_tables([config]), config


def _route(
    tables: navigation.NavigationTables,
    config: EnvConfig,
    position: tuple[float, ...],
    goal: tuple[float, ...],
) -> navigation.Route:
    row, known = navigation.find_map(tables, jnp.asarray(config.obstacles))
    return navigation.route(
        tables,
        row,
        known,
        jnp.asarray(position, jnp.float32),
        jnp.asarray(goal, jnp.float32),
        jnp.asarray(config.obstacles),
        _SIZE,
    )


def test_a_clear_line_goes_straight(walled: _Walled) -> None:
    tables, config = walled
    walk = _route(tables, config, (5.0, 9.2), (15.0, 9.2))
    np.testing.assert_allclose(np.asarray(walk.waypoint), (15.0, 9.2))
    assert float(walk.length) == pytest.approx(10.0)


def test_a_wall_sends_the_bot_to_a_node_beside_it(walled: _Walled) -> None:
    tables, config = walled
    walk = _route(tables, config, (5.0, 5.0), (15.0, 5.0))
    x, y = np.asarray(walk.waypoint)
    # A node just past one of the wall's ends (y 8 or 2), rounding its corner.
    assert 9.0 < x < 11.0
    assert y > 8.0 or y < 2.0
    # The true walk: to the wall's end, around its rounded clearance, and back down.
    corner = np.hypot(4.5, 3.5)
    assert 2 * corner < float(walk.length) < 2 * corner + 1.5


def test_a_body_at_a_node_heads_for_the_next_one(walled: _Walled) -> None:
    tables, config = walled
    first = _route(tables, config, (5.0, 5.0), (15.0, 5.0))
    node = tuple(float(v) for v in np.asarray(first.waypoint))
    second = _route(tables, config, node, (15.0, 5.0))
    assert not np.allclose(np.asarray(second.waypoint), node)
    assert float(second.length) < float(first.length)


def test_a_walk_home_equals_the_general_walk_to_the_same_point(walled: _Walled) -> None:
    tables, config = walled
    home = tuple(
        float(v) for v in np.asarray(config.capture_the_flag_flag_home_positions[0])
    )
    near_home = (home[0] + 0.01, home[1])  # not a home: the general route
    to_home = _route(tables, config, (15.0, 5.0), home)
    to_point = _route(tables, config, (15.0, 5.0), near_home)
    assert float(to_home.length) == pytest.approx(float(to_point.length), abs=0.05)


def test_an_unknown_map_walks_straight(walled: _Walled) -> None:
    tables, _ = walled
    other = _with_wall(12.0)
    row, known = navigation.find_map(tables, jnp.asarray(other.obstacles))
    assert not bool(known)
    walk = _route(tables, other, (5.0, 5.0), (15.0, 5.0))
    np.testing.assert_allclose(np.asarray(walk.waypoint), (15.0, 5.0))
    assert float(walk.length) == pytest.approx(10.0)
    assert int(row) == 0


def test_a_map_with_no_obstacles_has_no_nodes_and_walks_straight() -> None:
    config = _map(0)
    tables = navigation.build_navigation_tables([config])
    assert not np.isfinite(np.asarray(tables.walks)).any()
    walk = _route(tables, config, (2.0, 2.0), (18.0, 8.0))
    np.testing.assert_allclose(np.asarray(walk.waypoint), (18.0, 8.0))


@pytest.mark.parametrize(("map_id", "shorter"), [(36, False), (11, True)])
def test_walks_home_agree_with_the_reward_field_except_where_it_closes_a_passage(
    map_id: int, shorter: bool
) -> None:
    config = _map(map_id)
    tables = navigation.build_navigation_tables([config])
    obstacles = jnp.asarray(config.obstacles)
    blocked = np.asarray(
        blocked_cells(
            obstacles, map_width=20.0, map_height=10.0, cell_size=FINE_CELL_SIZE
        )
    )
    rows, columns = np.nonzero(~blocked)
    spots = np.stack(
        [(columns + 0.5) * FINE_CELL_SIZE, (rows + 0.5) * FINE_CELL_SIZE], 1
    )
    layouts, fields, _ = reward_recipes._route_bank()  # pyright: ignore[reportPrivateUsage]
    bank = int(np.argmax(np.all(layouts == np.asarray(config.obstacles), axis=(1, 2))))
    home = np.asarray(config.capture_the_flag_flag_home_positions[0])
    field = fields[bank, 0][rows, columns]
    reached = np.isfinite(field)
    row, known = navigation.find_map(tables, obstacles)

    def walk_home(point: jax.Array) -> jax.Array:
        return navigation.route(
            tables, row, known, point, jnp.asarray(home), obstacles, _SIZE
        ).length

    lengths = jax.vmap(walk_home)(jnp.asarray(spots, jnp.float32))
    walk = np.asarray(lengths)[reached]
    field = field[reached]
    # The field walks 8 directions on a 0.1 grid, so it reads up to about 8% long.
    assert np.mean(walk > 1.09 * field + 0.2) < 0.001
    assert (np.sum((walk < 0.85 * field) & (field - walk > 1.0)) > 0) == shorter


def test_the_builder_refuses_sets_it_cannot_tell_apart_or_size() -> None:
    config = _map(36)
    with pytest.raises(ValueError, match="at least one map"):
        navigation.build_navigation_tables([])
    with pytest.raises(ValueError, match="share an obstacle table"):
        navigation.build_navigation_tables([config, config])
    with pytest.raises(ValueError, match="one size"):
        navigation.build_navigation_tables(
            [config, _map(1)._replace(map_width=jnp.float32(24.0))]
        )


def test_the_cache_names_sets_by_their_maps_and_loads_what_it_saved(
    tmp_path: Path,
) -> None:
    config = _with_wall(10.0)
    name = navigation.navigation_tables_identity([config])
    assert name == navigation.navigation_tables_identity([_with_wall(10.0)])
    assert name != navigation.navigation_tables_identity([_with_wall(12.0)])
    built = navigation.load_navigation_tables([config], cache_dir=tmp_path)
    saved = tmp_path / f"{name}.npz"
    assert saved.exists() and len(list(tmp_path.iterdir())) == 1
    loaded = navigation.load_navigation_tables([config], cache_dir=tmp_path)
    for first, second in zip(built, loaded, strict=True):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    with np.load(saved) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays["walks"] = arrays["walks"] + 1
    np.savez(saved, **arrays)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="damaged"):
        navigation.load_navigation_tables([config], cache_dir=tmp_path)
