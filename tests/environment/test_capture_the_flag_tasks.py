"""Check the Capture the Flag flag homes and the spawn exchange.

Contracts: on all 52 maps the default flag homes are the two spawn centers,
every spawn pad of both teams can walk to both homes on the fine route field,
and each team's walk from its spawn center to the enemy home matches the other
team's within half a fine cell. A spawn exchange swaps the two flag homes with
the pads: balanced_spawn_configs keeps them in the first half of the batch and
swaps them in the second, and spawn_locations_for_source names an exchanged
Capture the Flag config choice 1 but refuses one whose pads were exchanged
without its homes. A handle reports its own mode's scores and outcome (captures
and the Capture the Flag result on a Capture the Flag handle, points and the
Team Deathmatch result otherwise), and its training facts keep only its own
mode's objective record: Core's Capture the Flag record on a Capture the Flag
handle, nothing on a Team Deathmatch handle.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from tests.helpers.capture_the_flag_fixtures import capture_the_flag_config

from marl_battlegrounds import environment, make
from marl_battlegrounds._task_settings import DEFAULT_CTF_FLAG_HOME_POSITIONS
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import TASK_MODE_CTF, TASK_MODE_KOTH, TASK_MODE_TDM
from marl_battlegrounds.modes.team_deathmatch._assets import map_geometry
from marl_battlegrounds.policies.route_field import (
    FINE_CELL_SIZE,
    blocked_cells,
    route_distance_at,
    route_distances,
)
from marl_battlegrounds.tasks import (
    balanced_spawn_configs,
    list_tdm_maps,
    make_standard_team_deathmatch_config,
    spawn_locations_for_source,
)

_HOMES = np.asarray(DEFAULT_CTF_FLAG_HOME_POSITIONS, dtype=np.float32)
_outcome = environment._outcome  # pyright: ignore[reportPrivateUsage]
_team_scores = environment._team_scores  # pyright: ignore[reportPrivateUsage]
_training_facts = environment._training_facts  # pyright: ignore[reportPrivateUsage]
_CONFIG = capture_the_flag_config(
    make_standard_team_deathmatch_config(
        map_id=29, team_a_roster=("mage", "rogue"), team_b_roster=("priest",)
    )
)


def _walk(field: jax.Array, start: np.ndarray) -> float:
    return float(route_distance_at(field, jnp.asarray(start), cell_size=FINE_CELL_SIZE))


def test_flag_homes_are_the_spawn_centers_and_reachable_on_every_map() -> None:
    for info in list_tdm_maps():
        geometry = map_geometry(info)
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
    for mode, scores, outcome in (
        (TASK_MODE_TDM, [1, 2], 1),
        (TASK_MODE_CTF, [3, 0], 2),
        (TASK_MODE_KOTH, [5, 6], 3),
    ):
        assert _team_scores(state, mode).tolist() == scores
        assert int(_outcome(info, mode)) == outcome
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
    for config, expect_record in ((_CONFIG, True), (tdm_config, False)):
        env = make(
            "tdm", env_config=config, num_envs=2, metrics="none", training_facts=True
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
