"""Check the Capture the Flag and King of the Hills fields at the host boundary.

Contracts: neutral_objective_fields gives one game's neutral value for every
objective field of EnvConfig and EnvState (carrier slots -1, contested False,
everything else 0), keeps one record's fields when asked, and returns NumPy
arrays equal to its JAX ones with host=True. Outside its own mode each field
must hold that value bit for bit, one test per field in a Team Deathmatch
config and snapshot, -0.0 included; wrong storage is refused. A Capture the
Flag config with the default settings is accepted. It refuses a nonzero Team
Deathmatch threshold or Red Zone depth, a non-neutral King of the Hills
setting, an empty team, homes no farther apart than the touch plus capture
distance, and each out-of-range Capture the Flag setting. King of
the Hills stays reserved. The two resolved config records refuse a Capture the
Flag config, whose identity records every field, while a Team Deathmatch
identity keeps its 13 keys. A Capture the Flag snapshot accepts each flag at
home, carried by a living enemy at that enemy's position, or dropped on the map
with a countdown in [1, return time], and captures up to the threshold; it
refuses every other combination, and an authored start must stay below the
threshold.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.helpers.capture_the_flag_fixtures import capture_the_flag_config

from marl_battlegrounds.core.config import (
    neutral_objective_fields,
    resolve_agent_profile,
    validate_env_config,
    validate_env_state,
    validate_scenario_initial_state,
)
from marl_battlegrounds.core.types import (
    TASK_MODE_KOTH,
    TEAM_A_ID,
    TEAM_B_ID,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.evaluation.catalog import (
    build_resolved_env_config_v1,
    build_resolved_env_config_v2,
)
from marl_battlegrounds.evaluation.evaluation_conditions import (
    config_record,
    restore_config,
)
from marl_battlegrounds.tasks import (
    load_tdm_scenario,
    make_standard_team_deathmatch_config,
)

_SCENARIO = load_tdm_scenario(1)
_TDM = make_standard_team_deathmatch_config(
    map_id=0, team_a_roster=("mage",), team_b_roster=("priest",)
)
_CTF = capture_the_flag_config(_TDM)


def _changed(value: int | float | jax.Array) -> int | float | jax.Array:
    if isinstance(value, jax.Array):
        return jnp.ones_like(value)
    return type(value)(1)


def test_neutral_values_split_by_record_and_match_on_the_host() -> None:
    config_fields = neutral_objective_fields(EnvConfig)
    state_fields = neutral_objective_fields(EnvState)
    host_fields = neutral_objective_fields(host=True)
    assert set(config_fields) <= set(EnvConfig._fields)
    assert set(state_fields) <= set(EnvState._fields)
    assert {**config_fields, **state_fields}.keys() == host_fields.keys()
    for name, value in {**config_fields, **state_fields}.items():
        assert not isinstance(host_fields[name], jax.Array)
        np.testing.assert_array_equal(np.asarray(value), host_fields[name])
    assert state_fields["capture_the_flag_flag_carrier_slots"].tolist() == [-1, -1]
    assert state_fields["king_of_the_hills_hill_is_contested"].dtype == jnp.bool_


@pytest.mark.parametrize("name", sorted(neutral_objective_fields(EnvConfig)))
def test_each_setting_must_be_neutral_outside_its_mode(name: str) -> None:
    changed = _TDM._replace(**{name: _changed(getattr(_TDM, name))})
    with pytest.raises(ValueError, match=f"^{name} must be .* outside "):
        validate_env_config(changed)


@pytest.mark.parametrize("name", sorted(neutral_objective_fields(EnvState)))
def test_each_state_field_must_be_neutral_outside_its_mode(name: str) -> None:
    state = _SCENARIO.initial_state
    changed = state._replace(**{name: _changed(getattr(state, name))})
    with pytest.raises(ValueError, match=f"^{name} must be .* outside "):
        validate_env_state(_SCENARIO.config, changed)


def test_negative_zero_and_wrong_storage_are_refused() -> None:
    with pytest.raises(ValueError, match=r"must be 0\.0 outside Capture the Flag"):
        validate_env_config(_TDM._replace(capture_the_flag_flag_capture_distance=-0.0))
    with pytest.raises(TypeError, match="must be an int, not float"):
        validate_env_config(_TDM._replace(king_of_the_hills_score_threshold=0.0))
    with pytest.raises(ValueError, match="must have shape"):
        validate_env_config(
            _TDM._replace(king_of_the_hills_hill_radii=jnp.zeros(2, jnp.float32))
        )


def test_a_capture_the_flag_config_is_accepted() -> None:
    assert validate_env_config(_CTF) is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"team_deathmatch_score_threshold": 20}, "zero outside Team Deathmatch"),
        ({"team_deathmatch_red_zone_depth": 5.0}, r"0\.0 outside Team Deathmatch"),
        ({"king_of_the_hills_score_threshold": 1}, "outside King of the Hills"),
        ({"capture_the_flag_score_threshold": 0}, "threshold must be in"),
        ({"capture_the_flag_score_threshold": 2**24 + 1}, "threshold must be in"),
        (
            {"capture_the_flag_flag_home_positions": jnp.full((2, 2), -1.0)},
            "must lie on the map",
        ),
        (
            {"capture_the_flag_flag_home_positions": jnp.full((2, 2), jnp.nan)},
            "must lie on the map",
        ),
        (
            {
                "capture_the_flag_flag_home_positions": jnp.asarray(
                    ((0.5, 5.0), (2.5, 5.0)), jnp.float32
                )
            },
            "farther apart than",
        ),
        ({"capture_the_flag_flag_interaction_distance": 0.0}, "stay finite"),
        ({"capture_the_flag_flag_capture_distance": 1e-40}, "stay finite"),
        ({"capture_the_flag_dropped_flag_auto_return_steps": 0}, "in \\[1, "),
    ],
    ids=[
        "tdm-threshold",
        "red-zone",
        "koth-setting",
        "zero-threshold",
        "huge-threshold",
        "home-off-map",
        "home-nan",
        "homes-too-close",
        "zero-reach",
        "subnormal-capture",
        "zero-return",
    ],
)
def test_capture_the_flag_refuses_each_bad_setting(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_env_config(_CTF._replace(**changes))


def test_capture_the_flag_needs_an_agent_on_each_team() -> None:
    profile = resolve_agent_profile(
        jnp.asarray((1, 0, 0, 0, 0, 0, 0, 0, 0, 0), jnp.int32),
        jnp.asarray((1, 0), jnp.int32),
    )
    with pytest.raises(ValueError, match="at least one configured active member"):
        validate_env_config(_CTF._replace(agent_profile=profile))


def test_king_of_the_hills_stays_reserved() -> None:
    with pytest.raises(ValueError, match="reserved but is not implemented"):
        validate_env_config(_TDM._replace(task_mode=TASK_MODE_KOTH))


def test_old_records_refuse_capture_the_flag_and_identities_keep_tdm_keys() -> None:
    for build in (build_resolved_env_config_v1, build_resolved_env_config_v2):
        with pytest.raises(ValueError, match="record only Team Deathmatch"):
            build(_CTF)
    tdm_id, tdm_content = config_record(_TDM)
    ctf_id, ctf_content = config_record(_CTF)
    assert len(tdm_content) == 13
    assert set(ctf_content) == set(EnvConfig._fields)
    assert config_record(restore_config(tdm_content))[0] == tdm_id
    assert config_record(restore_config(ctf_content))[0] == ctf_id


_CTF_START = capture_the_flag_config(_SCENARIO.config)
_HOME_STATE = _SCENARIO.initial_state._replace(
    team_deathmatch_scores=jnp.zeros(2, jnp.int32),
    capture_the_flag_flag_positions=_CTF_START.capture_the_flag_flag_home_positions,
)
_LIVING = np.asarray(_CTF_START.agent_profile.active_mask) & np.asarray(
    _HOME_STATE.alive_mask
)
_TEAMS = np.asarray(_CTF_START.agent_profile.team_ids)
_A_SLOT = int(np.flatnonzero(_LIVING & (_TEAMS == TEAM_A_ID))[0])
_B_SLOT = int(np.flatnonzero(_LIVING & (_TEAMS == TEAM_B_ID))[0])


def _flags(
    carriers: tuple[int, int] = (-1, -1),
    countdowns: tuple[int, int] = (0, 0),
    positions: jax.Array | None = None,
    scores: tuple[int, int] = (0, 0),
) -> EnvState:
    return _HOME_STATE._replace(
        capture_the_flag_flag_carrier_slots=jnp.asarray(carriers, jnp.int32),
        capture_the_flag_dropped_flag_countdowns=jnp.asarray(countdowns, jnp.int32),
        capture_the_flag_flag_positions=(
            _HOME_STATE.capture_the_flag_flag_positions
            if positions is None
            else positions
        ),
        capture_the_flag_scores=jnp.asarray(scores, jnp.int32),
    )


def _carried_by(slot: int) -> EnvState:
    positions = _HOME_STATE.capture_the_flag_flag_positions.at[0].set(
        _HOME_STATE.agent_positions[slot]
    )
    return _flags(carriers=(slot, -1), positions=positions)


def test_capture_the_flag_accepts_home_carried_and_dropped_flags() -> None:
    dropped = _HOME_STATE.capture_the_flag_flag_positions.at[1].set(
        jnp.asarray((10.0, 5.0))
    )
    for state in (
        _HOME_STATE,
        _carried_by(_B_SLOT),
        _flags(countdowns=(0, 30), positions=dropped),
        _flags(scores=(3, 0)),
    ):
        assert validate_env_state(_CTF_START, state) is None
    assert validate_scenario_initial_state(_CTF_START, _HOME_STATE) is None


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (_carried_by(_A_SLOT), "carrier must be -1 or an active, living"),
        (_flags(carriers=(10, -1)), "carrier must be -1 or an active, living"),
        (_flags(carriers=(_B_SLOT, -1)), "must sit at its carrier's position"),
        (_flags(countdowns=(0, 31)), r"countdown must be in \[1, 30\]"),
        (
            _flags(
                countdowns=(5, 0),
                positions=jnp.asarray(((-1.0, 5.0), (19.5, 5.0)), jnp.float32),
            ),
            "dropped flag must lie on the map",
        ),
        (
            _flags(positions=jnp.asarray(((1.0, 5.0), (19.5, 5.0)), jnp.float32)),
            "must be at its home",
        ),
        (_flags(scores=(4, 0)), r"capture_the_flag_scores must be in \[0, 3\]"),
    ],
    ids=[
        "own-team-carrier",
        "slot-out-of-range",
        "not-at-carrier",
        "countdown-too-long",
        "dropped-off-map",
        "home-elsewhere",
        "score-above-threshold",
    ],
)
def test_capture_the_flag_refuses_each_bad_flag_or_score(
    state: EnvState, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_env_state(_CTF_START, state)


def test_an_authored_capture_the_flag_start_stays_below_the_threshold() -> None:
    with pytest.raises(ValueError, match="Capture the Flag scenario scores"):
        validate_scenario_initial_state(_CTF_START, _flags(scores=(0, 3)))
