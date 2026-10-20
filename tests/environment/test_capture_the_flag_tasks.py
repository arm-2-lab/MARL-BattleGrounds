"""Check the Capture the Flag flag homes on every installed map.

Contracts: each team's flag home is its spawn center, or a point along the
spawn edge on the maps listed in tasks.py, and Team B's home is always Team
A's mirror image; on all 52 maps both homes keep the 1.25 clearance from every
obstacle (Core's disc_overlaps_obstacle), all ten spawn pads can walk to both
homes on the fine route field, and the two teams' walks to the enemy home
match within half a fine cell (the placement tool's check, run here); a home
left inside an obstacle's clearance fails that check; a map ID outside 0
through 51 is refused.
"""

import pytest
from scripts.dev.modes.capture_the_flag.place_flag_bases import check_map

from marl_battlegrounds import tasks
from marl_battlegrounds.tasks import (
    _capture_the_flag_flag_homes,  # pyright: ignore[reportPrivateUsage]
    list_tdm_maps,
)


def test_every_map_flag_home_passes_the_check() -> None:
    assert [
        problem
        for info in list_tdm_maps()
        for problem in check_map(info.map_id).problems()
    ] == []


def test_homes_sit_at_the_spawn_centers_and_mirror_each_other() -> None:
    assert _capture_the_flag_flag_homes(0) == ((0.5, 5.0), (19.5, 5.0))
    assert _capture_the_flag_flag_homes(29) == ((0.5, 3.75), (19.5, 3.75))
    for info in list_tdm_maps():
        (a_x, a_y), (b_x, b_y) = _capture_the_flag_flag_homes(info.map_id)
        assert (b_x, b_y) == (20.0 - a_x, a_y)


def test_a_home_inside_an_obstacle_clearance_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(tasks._CAPTURE_THE_FLAG_MOVED_HOME_Y, 29)  # pyright: ignore[reportPrivateUsage]
    assert any("closer than 1.25" in problem for problem in check_map(29).problems())


@pytest.mark.parametrize("map_id", [-1, 52, True])
def test_a_map_id_outside_the_catalog_is_refused(map_id: int) -> None:
    with pytest.raises(ValueError, match="map_id"):
        _capture_the_flag_flag_homes(map_id)
