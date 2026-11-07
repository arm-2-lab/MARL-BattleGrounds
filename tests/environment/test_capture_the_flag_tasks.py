"""Check the Capture the Flag flag homes and the spawn exchange.

Contracts: on all 52 Capture the Flag maps the default flag homes are the two
spawn centers, every spawn pad of both teams can walk to both homes on the fine
route field, and each team's walk from its spawn center to the enemy home
matches the other team's within half a fine cell. A spawn exchange swaps the
two flag homes with the pads: balanced_spawn_configs keeps them in the first
half of the batch and swaps them in the second, and spawn_locations_for_source
names an exchanged Capture the Flag config choice 1 but refuses one whose pads
were exchanged without its homes. team_scores_for_mode, outcome_for_mode and
score_threshold_for_mode read each mode's own scores, result and winning score,
so a handle reports its own mode's (captures and the Capture the Flag result on
a Capture the Flag handle, points and the Team Deathmatch result otherwise),
and its training facts keep only its own mode's objective record: Core's
Capture the Flag record on a Capture the Flag handle, nothing on a Team
Deathmatch handle. make_standard_capture_the_flag_config builds the same config
as converting the Team Deathmatch one on Capture the Flag's own map geometry,
with its settings passed through; list_ctf_maps and maps_for_task give each
mode's 52 maps, and scenarios_for_task each mode's scenarios (Team Deathmatch's
eight; none yet for Capture the Flag). Capture the Flag has its own 52 maps:
ordered IDs, "ctf_map_id_<id>_" names ending in their split, Team Deathmatch's
splits and curriculum flags, and its own entries; the factory reads its own map
files and never Team Deathmatch's, listed_map_id accepts only the task's own
entries, and TASK_MODES equals Core's mode constants. make("ctf") plays Capture
the Flag on a listed map with the mode's default threshold, refuses a Red Zone
depth, and a task name and its exact config must share a mode; the error names
the task to pass. Each handle's task property names the task make fixed, "tdm"
also for a neutral config. One dispatcher, make_standard_config, builds each
task's standard config with the task's defaults (DEFAULT_SCORE_THRESHOLDS and
DEFAULT_RED_ZONE_DEPTHS), takes only exactly 0.0 as a Capture the Flag depth,
and is what normalize_episode_specs builds from map IDs; task_for_mode names
the task each Core mode plays.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.helpers.capture_the_flag_fixtures import capture_the_flag_config

from marl_battlegrounds import environment, make
from marl_battlegrounds._task_settings import (
    DEFAULT_CTF_FLAG_HOME_POSITIONS,
    DEFAULT_RED_ZONE_DEPTHS,
    DEFAULT_SCORE_THRESHOLDS,
)
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import TASK_MODE_CTF, TASK_MODE_KOTH, TASK_MODE_TDM
from marl_battlegrounds.environment import (
    outcome_for_mode,
    score_threshold_for_mode,
    team_scores_for_mode,
)
from marl_battlegrounds.evaluation.evaluate import normalize_episode_specs
from marl_battlegrounds.modes.capture_the_flag import _assets as ctf_assets
from marl_battlegrounds.modes.team_deathmatch import _assets as tdm_assets
from marl_battlegrounds.policies.route_field import (
    FINE_CELL_SIZE,
    blocked_cells,
    route_distance_at,
    route_distances,
)
from marl_battlegrounds.tasks import (
    TASK_MODES,
    AgentClassName,
    balanced_spawn_configs,
    list_ctf_maps,
    list_tdm_maps,
    list_tdm_scenarios,
    listed_map_id,
    make_standard_capture_the_flag_config,
    make_standard_config,
    make_standard_team_deathmatch_config,
    maps_for_task,
    scenarios_for_task,
    spawn_locations_for_source,
    task_for_mode,
)

_HOMES = np.asarray(DEFAULT_CTF_FLAG_HOME_POSITIONS, dtype=np.float32)
_training_facts = environment._training_facts  # pyright: ignore[reportPrivateUsage]
_TEAM_A: tuple[AgentClassName, ...] = ("mage", "rogue")
_TEAM_B: tuple[AgentClassName, ...] = ("priest",)
_CONFIG = make_standard_capture_the_flag_config(
    map_id=29, team_a_roster=_TEAM_A, team_b_roster=_TEAM_B
)


def _walk(field: jax.Array, start: np.ndarray) -> float:
    return float(route_distance_at(field, jnp.asarray(start), cell_size=FINE_CELL_SIZE))


def test_flag_homes_are_the_spawn_centers_and_reachable_on_every_map() -> None:
    for info in list_ctf_maps():
        geometry = ctf_assets.map_geometry(info)
        pads = np.asarray(geometry.team_spawn_pad_positions, dtype=np.float32)
        centers = pads.mean(axis=1)
        np.testing.assert_array_equal(centers, _HOMES)
        blocked = blocked_cells(
            jnp.asarray(geometry.obstacles, dtype=jnp.float32),
            map_width=geometry.map_width,
            map_height=geometry.map_height,
            cell_size=FINE_CELL_SIZE,
        )
        fields = route_distances(blocked, jnp.asarray(_HOMES), cell_size=FINE_CELL_SIZE)
        for field in fields:
            for pad in pads.reshape(-1, 2):
                assert math.isfinite(_walk(field, pad)), (info.map_id, pad)
        attack_a, attack_b = _walk(fields[1], centers[0]), _walk(fields[0], centers[1])
        assert abs(attack_a - attack_b) <= FINE_CELL_SIZE / 2, info.map_id


def test_the_balanced_batch_swaps_flag_homes_with_the_pads() -> None:
    batch = balanced_spawn_configs(_CONFIG, num_envs=4)
    homes = np.asarray(batch.capture_the_flag_flag_home_positions)
    pads = np.asarray(batch.team_spawn_pad_positions)
    np.testing.assert_array_equal(homes, [_HOMES, _HOMES, _HOMES[::-1], _HOMES[::-1]])
    np.testing.assert_array_equal(pads[2:], pads[:2, ::-1])


def test_a_source_match_needs_the_homes_exchanged_with_the_pads() -> None:
    batch = balanced_spawn_configs(_CONFIG, num_envs=4)
    matches, choices = spawn_locations_for_source(batch, _CONFIG)
    assert matches.tolist() == [True] * 4
    assert choices.tolist() == [0, 0, 1, 1]
    pads_only = _CONFIG._replace(
        team_spawn_pad_positions=_CONFIG.team_spawn_pad_positions[::-1]
    )
    matches, choice = spawn_locations_for_source(pads_only, _CONFIG)
    assert not bool(matches)
    assert int(choice) == -1


def test_each_mode_reads_its_own_scores_outcome_and_facts() -> None:
    state, _, _, info = core.reset(_CONFIG, jax.random.key(0))
    state = state._replace(
        team_deathmatch_scores=jnp.asarray([1, 2], jnp.int32),
        capture_the_flag_scores=jnp.asarray([3, 0], jnp.int32),
        king_of_the_hills_scores=jnp.asarray([5, 6], jnp.int32),
    )
    facts = info.transition_facts
    info = info._replace(
        transition_facts=facts._replace(
            team_deathmatch_facts=facts.team_deathmatch_facts._replace(
                outcome=jnp.asarray(1, jnp.int32)
            ),
            capture_the_flag_facts=facts.capture_the_flag_facts._replace(
                outcome=jnp.asarray(2, jnp.int32)
            ),
            king_of_the_hills_facts=facts.king_of_the_hills_facts._replace(
                outcome=jnp.asarray(3, jnp.int32)
            ),
        )
    )
    config = _CONFIG._replace(
        team_deathmatch_score_threshold=7, king_of_the_hills_score_threshold=9
    )
    for mode, scores, outcome, threshold in (
        (TASK_MODE_TDM, [1, 2], 1, 7),
        (TASK_MODE_CTF, [3, 0], 2, 3),
        (TASK_MODE_KOTH, [5, 6], 3, 9),
    ):
        assert team_scores_for_mode(state, mode).tolist() == scores
        assert int(outcome_for_mode(info, mode)) == outcome
        assert int(score_threshold_for_mode(config, mode)) == threshold
    ctf = _training_facts(info, TASK_MODE_CTF)
    assert ctf.capture_the_flag_facts is info.transition_facts.capture_the_flag_facts
    assert ctf.king_of_the_hills_facts is None
    tdm = _training_facts(info, TASK_MODE_TDM)
    assert tdm.capture_the_flag_facts is None and tdm.king_of_the_hills_facts is None


def test_only_a_capture_the_flag_handle_returns_capture_the_flag_training_facts() -> (
    None
):
    tdm_config = make_standard_team_deathmatch_config(
        map_id=29, team_a_roster=("mage", "rogue"), team_b_roster=("priest",)
    )
    for task, config, expect_record in (
        ("ctf", _CONFIG, True),
        ("tdm", tdm_config, False),
    ):
        env = make(
            task, env_config=config, num_envs=2, metrics="none", training_facts=True
        )
        _, state = env.reset(jax.random.key(0))
        key = jax.random.key(1)
        *_, info = env.step(key, state, env.sample_actions(key, state))
        facts = info.training_facts
        assert facts is not None and facts.king_of_the_hills_facts is None
        if expect_record:
            assert facts.capture_the_flag_facts is not None
            assert facts.capture_the_flag_facts.flag_was_picked_up_by_agent.shape == (
                2,
                10,
            )
        else:
            assert facts.capture_the_flag_facts is None


def test_the_factory_matches_the_converted_team_deathmatch_config() -> None:
    geometry = ctf_assets.map_geometry(list_ctf_maps()[29])
    converted = capture_the_flag_config(
        make_standard_team_deathmatch_config(
            map_id=29, team_a_roster=_TEAM_A, team_b_roster=_TEAM_B
        )
    )._replace(
        map_width=geometry.map_width,
        map_height=geometry.map_height,
        obstacles=jnp.asarray(geometry.obstacles, jnp.float32),
        team_spawn_pad_positions=jnp.asarray(
            geometry.team_spawn_pad_positions, jnp.float32
        ),
    )
    leaves = jax.tree.leaves(jax.tree.map(np.array_equal, _CONFIG, converted))
    assert all(leaves)
    changed = make_standard_capture_the_flag_config(
        map_id=29,
        score_threshold=5,
        max_steps=200,
        flag_interaction_distance=0.5,
        flag_capture_distance=1.0,
        dropped_flag_auto_return_steps=10,
        team_a_roster=_TEAM_A,
        team_b_roster=_TEAM_B,
    )
    assert (
        int(changed.capture_the_flag_score_threshold),
        int(changed.max_steps),
        float(changed.capture_the_flag_flag_interaction_distance),
        float(changed.capture_the_flag_flag_capture_distance),
        int(changed.capture_the_flag_dropped_flag_auto_return_steps),
    ) == (5, 200, 0.5, 1.0, 10)


def test_each_mode_lists_its_maps_through_one_dispatcher() -> None:
    ctf, tdm = list_ctf_maps(), list_tdm_maps()
    assert [info.map_id for info in ctf] == list(range(52))
    assert all(info.name.startswith(f"ctf_map_id_{info.map_id}_") for info in ctf)
    assert all(info.name.endswith(f"_{info.split}") for info in ctf)
    assert [(c.split, c.curriculum) for c in ctf] == [
        (t.split, t.curriculum) for t in tdm
    ]
    assert ctf != tdm
    assert maps_for_task("ctf") == list_ctf_maps()
    assert maps_for_task("tdm") == list_tdm_maps()
    with pytest.raises(ValueError, match=r"^task must be one of 'tdm', 'ctf'"):
        maps_for_task("koth")  # pyright: ignore[reportArgumentType]
    assert scenarios_for_task("tdm") == list_tdm_scenarios()
    assert scenarios_for_task("ctf") == ()
    with pytest.raises(ValueError, match=r"^task must be one of 'tdm', 'ctf'"):
        scenarios_for_task("koth")  # pyright: ignore[reportArgumentType]


def test_make_ctf_plays_capture_the_flag_on_a_listed_map() -> None:
    env = make(
        "ctf",
        map_id=29,
        num_envs=2,
        metrics="none",
        team_a_roster=_TEAM_A,
        team_b_roster=_TEAM_B,
    )
    _, state = env.reset(jax.random.key(0))
    config = state.config
    np.testing.assert_array_equal(np.asarray(config.task_mode), [TASK_MODE_CTF] * 2)
    np.testing.assert_array_equal(
        np.asarray(config.capture_the_flag_score_threshold), [3, 3]
    )
    homes = np.asarray(config.capture_the_flag_flag_home_positions)
    np.testing.assert_array_equal(homes[0], _HOMES)
    np.testing.assert_array_equal(homes[1], _HOMES[::-1])
    five = make("ctf", map_id=29, score_threshold=5, num_envs=2, metrics="none")
    _, state = five.reset(jax.random.key(0))
    np.testing.assert_array_equal(
        np.asarray(state.config.capture_the_flag_score_threshold), [5, 5]
    )


def test_a_task_and_its_exact_config_share_one_mode() -> None:
    with pytest.raises(ValueError, match=r"^red_zone_depth is a Team Deathmatch"):
        make("ctf", map_id=29, red_zone_depth=0.0, num_envs=2)
    tdm_config = make_standard_team_deathmatch_config(
        map_id=29, team_a_roster=_TEAM_A, team_b_roster=_TEAM_B
    )
    with pytest.raises(
        ValueError, match=r"which task 'ctf' does not play; pass task='tdm'$"
    ):
        make("ctf", env_config=tdm_config)
    with pytest.raises(
        ValueError, match=r"which task 'tdm' does not play; pass task='ctf'$"
    ):
        make("tdm", env_config=_CONFIG)
    with pytest.raises(ValueError, match=r"this environment plays task_mode 3$"):
        make("ctf").reset(jax.random.key(0), tdm_config)
    neutral = make_standard_team_deathmatch_config(
        map_id=29, team_a_roster=_TEAM_A, team_b_roster=_TEAM_B, red_zone_depth=0.0
    )._replace(task_mode=0, team_deathmatch_score_threshold=0)
    handles = (make("ctf"), make("tdm"), make("tdm", env_config=neutral))
    assert [env.task for env in handles] == ["ctf", "tdm", "tdm"]


def _same(left: object, right: object) -> bool:
    return all(jax.tree.leaves(jax.tree.map(np.array_equal, left, right)))


def test_one_dispatcher_builds_each_tasks_standard_config() -> None:
    assert DEFAULT_SCORE_THRESHOLDS == {"tdm": 20, "ctf": 3}
    assert DEFAULT_RED_ZONE_DEPTHS == {"tdm": 5.0, "ctf": 0.0}
    assert [task_for_mode(mode) for mode in range(5)] == [
        None,
        "tdm",
        None,
        "ctf",
        None,
    ]
    assert _same(
        make_standard_config(
            "ctf", map_id=29, team_a_roster=_TEAM_A, team_b_roster=_TEAM_B
        ),
        _CONFIG,
    )
    assert _same(
        make_standard_config(
            "ctf",
            map_id=29,
            red_zone_depth=0.0,
            team_a_roster=_TEAM_A,
            team_b_roster=_TEAM_B,
        ),
        _CONFIG,
    )
    assert _same(
        make_standard_config(
            "tdm", map_id=29, team_a_roster=_TEAM_A, team_b_roster=_TEAM_B
        ),
        make_standard_team_deathmatch_config(
            map_id=29, team_a_roster=_TEAM_A, team_b_roster=_TEAM_B
        ),
    )
    for depth in (5.0, -0.0):
        with pytest.raises(ValueError, match=r"ctf takes only 0.0$"):
            make_standard_config(
                "ctf",
                map_id=29,
                red_zone_depth=depth,
                team_a_roster=_TEAM_A,
                team_b_roster=_TEAM_B,
            )
    with pytest.raises(ValueError, match=r"^task must be one of 'tdm', 'ctf'"):
        make_standard_config(
            "koth",  # pyright: ignore[reportArgumentType]
            map_id=29,
            team_a_roster=_TEAM_A,
            team_b_roster=_TEAM_B,
        )
    (spec,) = normalize_episode_specs(
        [29], 1, _TEAM_A, _TEAM_B, 3, 300, red_zone_depth=0.0, task="ctf"
    )
    assert _same(spec.env_config, _CONFIG)
    defaults = normalize_episode_specs(
        None, 5, _TEAM_A, _TEAM_B, 3, 300, red_zone_depth=0.0, task="ctf"
    )
    assert [spec.map_id for spec in defaults] == [47, 48, 49, 50, 51]


def test_the_capture_the_flag_factory_reads_its_own_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[str] = []
    own = ctf_assets._resource_bytes  # pyright: ignore[reportPrivateUsage]

    def recorded(relative_path: str) -> bytes:
        reads.append(relative_path)
        return own(relative_path)

    def refused(relative_path: str) -> bytes:
        raise AssertionError(f"read Team Deathmatch's {relative_path}")

    monkeypatch.setattr(ctf_assets, "_resource_bytes", recorded)
    monkeypatch.setattr(tdm_assets, "_resource_bytes", refused)
    make_standard_capture_the_flag_config(
        map_id=29, team_a_roster=_TEAM_A, team_b_roster=_TEAM_B
    )
    assert "maps/29.json" in reads


def test_listed_entries_belong_to_one_task_and_modes_match_core() -> None:
    assert TASK_MODES == {"tdm": TASK_MODE_TDM, "ctf": TASK_MODE_CTF}
    assert listed_map_id(list_ctf_maps()[42], "ctf") == 42
    assert listed_map_id(list_tdm_maps()[42], "tdm") == 42
    with pytest.raises(ValueError, match=r"with list_ctf_maps\(\) from"):
        listed_map_id(list_tdm_maps()[42], "ctf")
