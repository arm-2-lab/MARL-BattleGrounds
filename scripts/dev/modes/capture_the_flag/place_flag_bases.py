"""Place and check the Capture the Flag flag homes on every map, and draw their atlas.

Each team's flag home is its spawn center, unless an obstacle comes closer than
the home clearance (1.25 map units); then Team A's home moves along its spawn
edge to the nearest clear point and Team B's mirrors it. The moves live in
tasks._CAPTURE_THE_FLAG_MOVED_HOME_Y. This tool:

- proposes that table from the rule (the nearest clear y in 0.25-unit steps,
  the lower one on a tie) and lists where tasks.py differs; a home moved by
  hand differs on purpose;
- checks the homes tasks.py serves on every map: both keep the clearance
  (Core's disc_overlaps_obstacle), all ten spawn pads can walk to both homes on
  the fine route field, and each team's walk from its spawn center to the enemy
  home matches the other team's within half a fine cell;
- with --atlas PATH, draws all maps and their homes in one PNG.

Run from the repository root on the CPU (about 5 seconds; matplotlib only for
the atlas):

    JAX_PLATFORMS=cpu uv run --no-sync python \\
        scripts/dev/modes/capture_the_flag/place_flag_bases.py --atlas OUT.png

It reads the installed maps and writes only the atlas. It exits 1 when a check
fails. tests/environment/test_capture_the_flag_tasks.py runs the same check.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from marl_battlegrounds.core.geometry import disc_overlaps_obstacle
from marl_battlegrounds.core.types import (
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_TYPE_PILLAR,
)
from marl_battlegrounds.modes.team_deathmatch._assets import MapGeometry, map_geometry
from marl_battlegrounds.policies.route_field import (
    BODY_RADIUS,
    FINE_CELL_SIZE,
    blocked_cells,
    route_distance_at,
    route_distances,
)
from marl_battlegrounds.tasks import (
    _CAPTURE_THE_FLAG_HOME_CLEARANCE,  # pyright: ignore[reportPrivateUsage]
    _CAPTURE_THE_FLAG_MOVED_HOME_Y,  # pyright: ignore[reportPrivateUsage]
    _capture_the_flag_flag_homes,  # pyright: ignore[reportPrivateUsage]
    list_tdm_maps,
)

STEP = 0.25
_TEAM_COLORS = ("#2f6fbf", "#c8452f")
_SPLIT_COLORS = {"training": "#555555", "validation": "#8a5a00", "test": "#6b2e8f"}
# Words AP heading case keeps lowercase inside a title.
_SMALL_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "but",
    "or",
    "nor",
    "as",
    "at",
    "by",
    "for",
    "in",
    "of",
    "on",
    "per",
    "to",
    "via",
    "vs",
    "so",
    "yet",
}


@dataclass(frozen=True)
class FlagHomeCheck:
    """One map's flag homes and what the check found.

    Attributes
    ----------
    map_id : int
        Current map ID, 0 through 51.
    homes : numpy.ndarray
        Float32 (2, 2): Team A's and Team B's home [x, y] in map units.
    clear : tuple[bool, bool]
        Whether each home keeps the clearance from every obstacle.
    reachable : bool
        Whether all ten spawn pads can walk to both homes.
    attack_walks : tuple[float, float]
        Team A's walk from its spawn center to Team B's home, then Team B's to
        Team A's, in map units on the fine route field; infinity if none.
    """

    map_id: int
    homes: np.ndarray
    clear: tuple[bool, bool]
    reachable: bool
    attack_walks: tuple[float, float]

    def problems(self) -> list[str]:
        """Return one plain sentence per failed check; empty when all pass."""
        found = [
            f"map {self.map_id}: Team {team}'s home {tuple(home.tolist())} is closer "
            f"than {_CAPTURE_THE_FLAG_HOME_CLEARANCE} to an obstacle"
            for team, home, ok in zip("AB", self.homes, self.clear, strict=True)
            if not ok
        ]
        if not self.reachable:
            found.append(f"map {self.map_id}: a spawn pad cannot walk to a flag home")
        if not abs(self.attack_walks[0] - self.attack_walks[1]) <= FINE_CELL_SIZE / 2:
            found.append(
                f"map {self.map_id}: the teams' walks to the enemy home differ "
                f"({self.attack_walks[0]:.2f} and {self.attack_walks[1]:.2f})"
            )
        return found


def is_clear(obstacles: jax.Array, point: tuple[float, float] | np.ndarray) -> bool:
    """Return whether a disc of the home clearance at point touches no obstacle.

    obstacles is float32 (slots, 8) in EnvConfig.obstacles columns; point is
    [x, y] in map units. Tangency is clear, as in Core's disc_overlaps_obstacle.
    """
    overlaps = jax.vmap(disc_overlaps_obstacle, in_axes=(None, None, 0))(
        jnp.asarray(point, dtype=jnp.float32),
        _CAPTURE_THE_FLAG_HOME_CLEARANCE,
        obstacles,
    )
    return not bool(overlaps.any())


def placed_home_y(geometry: MapGeometry) -> float:
    """Return Team A's home y by the placement rule.

    Start at the spawn center's y and try y - k * STEP, then y + k * STEP, for
    k = 0, 1, 2, ..., keeping a body's center on the map; return the first y
    whose point on the spawn edge is clear. Raise ValueError when no point on
    the edge is clear.
    """
    obstacles = jnp.asarray(geometry.obstacles, dtype=jnp.float32)
    x, y = np.mean(np.asarray(geometry.team_spawn_pad_positions[0]), axis=0).tolist()
    for k in range(math.ceil(geometry.map_height / STEP) + 1):
        for candidate in (y - k * STEP, y + k * STEP):
            on_map = BODY_RADIUS <= candidate <= geometry.map_height - BODY_RADIUS
            if on_map and is_clear(obstacles, (x, candidate)):
                return candidate
    raise ValueError("no clear point on the spawn edge")


def check_map(map_id: int) -> FlagHomeCheck:
    """Check the flag homes tasks.py serves on one map (see the file description)."""
    geometry = map_geometry(list_tdm_maps()[map_id])
    obstacles = jnp.asarray(geometry.obstacles, dtype=jnp.float32)
    homes = np.asarray(_capture_the_flag_flag_homes(map_id), dtype=np.float32)
    pads = np.asarray(geometry.team_spawn_pad_positions, dtype=np.float32)
    blocked = blocked_cells(
        obstacles,
        map_width=geometry.map_width,
        map_height=geometry.map_height,
        cell_size=FINE_CELL_SIZE,
    )
    fields = route_distances(blocked, jnp.asarray(homes), cell_size=FINE_CELL_SIZE)

    def walk(home: int, start: np.ndarray) -> float:
        return float(
            route_distance_at(
                fields[home], jnp.asarray(start), cell_size=FINE_CELL_SIZE
            )
        )

    centers = pads.mean(axis=1)
    return FlagHomeCheck(
        map_id=map_id,
        homes=homes,
        clear=(is_clear(obstacles, homes[0]), is_clear(obstacles, homes[1])),
        reachable=all(
            math.isfinite(walk(h, pad)) for h in (0, 1) for pad in pads.reshape(-1, 2)
        ),
        attack_walks=(walk(1, centers[0]), walk(0, centers[1])),
    )


def draw_atlas(checks: list[FlagHomeCheck], path: Path) -> None:
    """Draw every checked map and its flag homes into one PNG at path.

    Each panel shows the obstacles, both spawn banks, each home with its
    clearance circle, an arrow where a home left its spawn center, and the
    map's ID, name, split and the teams' walks to the enemy home. Creates
    path's folder; overwrites path. Needs matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    maps = list_tdm_maps()
    columns = 8
    rows = math.ceil(len(checks) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.2 * rows))
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, check in zip(axes.flat, checks, strict=False):
        info = maps[check.map_id]
        geometry = map_geometry(info)
        axis.set_xlim(0, geometry.map_width)
        axis.set_ylim(0, geometry.map_height)
        axis.set_aspect("equal")
        axis.add_patch(
            patches.Rectangle(
                (0, 0), geometry.map_width, geometry.map_height, fill=False, lw=0.8
            )
        )
        for row in np.asarray(geometry.obstacles):
            if row[OBSTACLE_FEATURE_ACTIVE] != 1.0:
                continue
            center = (row[OBSTACLE_FEATURE_X], row[OBSTACLE_FEATURE_Y])
            if row[OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_PILLAR:
                shape = patches.Circle(center, row[OBSTACLE_FEATURE_RADIUS])
            else:
                width, height = (
                    row[OBSTACLE_FEATURE_WIDTH],
                    row[OBSTACLE_FEATURE_HEIGHT],
                )
                shape = patches.Rectangle(
                    (center[0] - width / 2, center[1] - height / 2),
                    width,
                    height,
                    angle=math.degrees(row[OBSTACLE_FEATURE_THETA]),
                    rotation_point="center",
                )
            shape.set_color("#9a9a9a")
            axis.add_patch(shape)
        pads = np.asarray(geometry.team_spawn_pad_positions)
        for team in (0, 1):
            color = _TEAM_COLORS[team]
            axis.scatter(
                pads[team, :, 0],
                pads[team, :, 1],
                s=6,
                marker="s",
                color=color,
                alpha=0.5,
            )
            home = check.homes[team]
            axis.add_patch(
                patches.Circle(
                    home,
                    _CAPTURE_THE_FLAG_HOME_CLEARANCE,
                    fill=False,
                    ls="--",
                    lw=0.8,
                    color=color,
                )
            )
            axis.scatter(*home, s=30, marker="^", color=color, zorder=3)
            spawn = pads[team].mean(axis=0)
            if not np.allclose(spawn, home):
                axis.annotate(
                    "",
                    xy=home,
                    xytext=spawn,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
                )
        # Catalog names read "tdm_map_id_<id>_<words>_<split>", with "_" or "-".
        words = re.split(r"[_-]", info.name)[4:-1]
        name = " ".join(
            word
            if 0 < i < len(words) - 1 and word in _SMALL_WORDS
            else word.capitalize()
            for i, word in enumerate(words)
        )
        moved = check.homes[0][1] - pads[0].mean(axis=0)[1]
        note = (
            f"home moved {abs(moved):g} {'down' if moved < 0 else 'up'}"
            if moved
            else "spawn center"
        )
        walks = f"walk {check.attack_walks[0]:.1f} / {check.attack_walks[1]:.1f}"
        axis.set_title(
            f"{info.map_id} · {name} · {info.split}\n{note} · {walks}",
            fontsize=7,
            color=_SPLIT_COLORS.get(info.split, "black"),
            fontweight="bold" if moved else "normal",
        )
    figure.suptitle(
        "Capture the Flag Flag Homes: Spawn Layout\n"
        "Triangle: a flag home. Dashed circle: the "
        f"{_CAPTURE_THE_FLAG_HOME_CLEARANCE} touch and capture distance. Arrow: a "
        "home moved off its spawn center. Walk: Team A's and Team B's walk to the "
        "enemy home, in map units",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))  # keep the heading clear of the panels
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> int:
    """Run the placement proposal and the check; draw the atlas when asked.

    Prints the proposed move table, each difference from tasks.py, and every
    problem. Returns 0 when every map passes, otherwise 1.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--atlas", type=Path, help="write the atlas PNG to this path")
    args = parser.parse_args()
    maps = list_tdm_maps()
    proposed: dict[int, float] = {}
    for info in maps:
        geometry = map_geometry(info)
        spawn_y = float(
            np.mean(np.asarray(geometry.team_spawn_pad_positions[0]), axis=0)[1]
        )
        y = placed_home_y(geometry)
        if y != spawn_y:
            proposed[info.map_id] = y
    print(f"Proposed moved homes: {proposed}")
    for map_id in sorted(proposed.keys() | _CAPTURE_THE_FLAG_MOVED_HOME_Y.keys()):
        if proposed.get(map_id) != _CAPTURE_THE_FLAG_MOVED_HOME_Y.get(map_id):
            print(
                f"map {map_id}: rule gives {proposed.get(map_id)}, tasks.py has "
                f"{_CAPTURE_THE_FLAG_MOVED_HOME_Y.get(map_id)}"
            )
    checks = [check_map(info.map_id) for info in maps]
    problems = [problem for check in checks for problem in check.problems()]
    print("\n".join(problems) or f"All {len(checks)} maps pass the flag-home check.")
    if args.atlas is not None:
        draw_atlas(checks, args.atlas)
        print(f"Wrote {args.atlas}.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
