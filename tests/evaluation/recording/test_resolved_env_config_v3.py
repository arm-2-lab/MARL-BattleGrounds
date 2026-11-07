"""Check that resolved config V3 records Capture the Flag configs, mirrored with V2.

Contracts: build_resolved_env_config_v3 records a Capture the Flag config as
V2's shared settings, with the same values a Team Deathmatch V2 record of the
same map holds, plus the five Capture the Flag settings under EnvConfig's
names, and no Team Deathmatch or King of the Hills field. The record's Capture
the Flag fields are exactly EnvConfig's, so a setting added to Core fails here
until the record follows. Changing any one Capture the Flag setting, or
exchanging the flag homes, changes the digest. The record reloads from its
JSON, and in a saved record a wrong digest, a wrong task_mode or a Team
Deathmatch field is refused. build_resolved_env_config returns V2, the same
record build_resolved_env_config_v2 builds, for Team Deathmatch and neutral
configs and V3 for Capture the Flag; the V3 builder refuses Team Deathmatch,
neutral and Capture the Flag configs with a stray Team Deathmatch setting. Host
only, on CPU.
"""

from typing import Any

import pytest

from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.evaluation.catalog import (
    build_resolved_env_config,
    build_resolved_env_config_v2,
    build_resolved_env_config_v3,
)
from marl_battlegrounds.evaluation.models import (
    ResolvedCaptureTheFlagConfigV3,
    ResolvedEnvConfigV2,
)
from marl_battlegrounds.tasks import (
    AgentClassName,
    make_standard_capture_the_flag_config,
    make_standard_team_deathmatch_config,
)

_A: tuple[AgentClassName, ...] = ("mage", "warrior")
_B: tuple[AgentClassName, ...] = ("rogue",)
_CTF = make_standard_capture_the_flag_config(
    map_id=29, team_a_roster=_A, team_b_roster=_B
)
_TDM = make_standard_team_deathmatch_config(
    map_id=29, team_a_roster=_A, team_b_roster=_B
)
_NEUTRAL = make_standard_team_deathmatch_config(
    map_id=29, team_a_roster=_A, team_b_roster=_B, red_zone_depth=0.0
)._replace(task_mode=0, team_deathmatch_score_threshold=0)
_CTF_FIELDS = {
    name for name in EnvConfig._fields if name.startswith("capture_the_flag")
}
_TDM_FIELDS = {"team_deathmatch_score_threshold", "team_deathmatch_red_zone_depth"}


def test_a_record_holds_the_shared_settings_and_only_its_own() -> None:
    saved = build_resolved_env_config_v3(_CTF)
    assert (
        ResolvedCaptureTheFlagConfigV3.model_validate_json(saved.model_dump_json())
        == saved
    )
    record = saved.model_dump()
    shared = set(ResolvedEnvConfigV2.model_fields) - _TDM_FIELDS
    assert set(ResolvedCaptureTheFlagConfigV3.model_fields) == shared | _CTF_FIELDS
    assert set(record) == shared | _CTF_FIELDS
    assert (record["schema_version"], record["task_mode"]) == (3, 3)
    assert {name: record[name] for name in _CTF_FIELDS} == {
        "capture_the_flag_score_threshold": 3,
        "capture_the_flag_flag_home_positions": ((0.5, 5.0), (19.5, 5.0)),
        "capture_the_flag_flag_interaction_distance": 0.75,
        "capture_the_flag_flag_capture_distance": 1.25,
        "capture_the_flag_dropped_flag_auto_return_steps": 30,
    }
    tdm = build_resolved_env_config_v2(_TDM).model_dump()
    same = shared - {"schema_version", "task_mode", "canonical_digest_sha256"}
    assert {name: record[name] for name in same} == {name: tdm[name] for name in same}


def test_each_capture_the_flag_setting_moves_the_digest() -> None:
    homes = _CTF.capture_the_flag_flag_home_positions
    variants = [
        _CTF,
        _CTF._replace(capture_the_flag_score_threshold=5),
        _CTF._replace(capture_the_flag_flag_home_positions=homes[::-1]),
        _CTF._replace(capture_the_flag_flag_interaction_distance=0.5),
        _CTF._replace(capture_the_flag_flag_capture_distance=1.0),
        _CTF._replace(capture_the_flag_dropped_flag_auto_return_steps=31),
    ]
    digests = {
        build_resolved_env_config_v3(config).canonical_digest_sha256
        for config in variants
    }
    assert len(digests) == len(variants)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"canonical_digest_sha256": "0" * 64}, "canonical digest mismatch"),
        ({"task_mode": 1}, "task_mode"),
        ({"team_deathmatch_score_threshold": 0}, "Extra inputs are not permitted"),
    ],
)
def test_the_record_reloads_and_refuses_tampering(
    change: dict[str, Any], message: str
) -> None:
    record = build_resolved_env_config_v3(_CTF)
    with pytest.raises(ValueError, match=message):
        ResolvedCaptureTheFlagConfigV3.model_validate({**record.model_dump(), **change})


def test_the_router_picks_each_modes_current_version() -> None:
    for config in (_TDM, _NEUTRAL):
        record = build_resolved_env_config(config)
        assert type(record) is ResolvedEnvConfigV2
        assert record == build_resolved_env_config_v2(config)
        with pytest.raises(ValueError, match=r"record ResolvedEnvConfigV2, not V3$"):
            build_resolved_env_config_v3(config)
    assert build_resolved_env_config(_CTF) == build_resolved_env_config_v3(_CTF)
    with pytest.raises(ValueError, match=r"^team_deathmatch_score_threshold must be"):
        build_resolved_env_config_v3(_CTF._replace(team_deathmatch_score_threshold=20))
