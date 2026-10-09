"""Check resolved controller ownership at the browser transport boundary.

Current frames identify every configured physical row, including dead agents.
Browser commands cannot grant themselves Manual control by submitting display
metadata. Older fixtures without metadata remain readable as older fixtures.
"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from marl_battlegrounds._battle_client.protocol import (
    CombatSetupV1,
    SetCombatConfigurationCommandV1,
)


def _setup() -> dict[str, Any]:
    return {
        "team_a_controller": "system:team_a",
        "team_b_controller": "system:team_b",
        "execution_information_mode": "shared_obs",
        "manual_slots": [0, 2, 8],
        "slot_labels": [
            "Manual",
            "Actor A",
            "Manual",
            None,
            None,
            "Actor B",
            "Actor B",
            "Actor C",
            "Manual",
            None,
        ],
    }


def test_ownership_round_trip_preserves_physical_rows() -> None:
    original = CombatSetupV1.model_validate_json(json.dumps(_setup()))
    restored = CombatSetupV1.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.manual_slots == (0, 2, 8)
    assert restored.slot_labels is not None
    assert restored.slot_labels[3] is None
    assert restored.slot_labels[8] == "Manual"


@pytest.mark.parametrize(
    "changes",
    [
        {"manual_slots": [2, 0]},
        {"manual_slots": [0, 0]},
        {"manual_slots": [True]},
        {"manual_slots": [10]},
        {"manual_slots": [-1]},
        {"manual_slots": [3]},
        {"manual_slots": [1]},
        {"manual_slots": None},
        {"slot_labels": None},
        {"slot_labels": ["Manual"] * 9},
        {"slot_labels": [" "] * 10},
    ],
)
def test_invalid_ownership_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CombatSetupV1.model_validate_json(json.dumps({**_setup(), **changes}))


def test_display_metadata_cannot_be_used_as_a_setup_command() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SetCombatConfigurationCommandV1.model_validate(
            {**_setup(), "command_type": "set_combat_configuration"}
        )


def test_older_fixture_omits_both_new_fields() -> None:
    payload = _setup()
    del payload["manual_slots"]
    del payload["slot_labels"]
    restored = CombatSetupV1.model_validate(payload)
    assert restored.model_dump(mode="json") == payload
