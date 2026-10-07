"""Check the writer's handling of finished games' member summaries.

Contracts: a staged summary that no completion consumed refuses the next one;
checking a summary keeps only active slots of a side whose System has two or
more components, returns None when nothing is staged or no side has members,
and refuses arrays that are not int32 with ten slots, an inactive slot with an
ID other than -1, an ID outside the component table and a change decision
outside the game. No game is played.
"""

# pyright: reportPrivateUsage=false
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marl_battlegrounds.evaluation.run_writer import RunWriter

ACTIVE = np.array([True, True, False, False, False, True, True, True, False, False])
ENTRY = {"system_ids": {"team_a": "pool", "team_b": "solo"}}


def _writer(tmp_path: Path) -> RunWriter:
    writer = RunWriter(tmp_path)
    writer._details["systems"].update(
        pool={"components": [{"name": "a"}, {"name": "b"}]},
        solo={"components": [{"name": "x"}]},
    )
    return writer


def _record(
    **changes: tuple[int, int],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    first = np.where(ACTIVE, 0, -1).astype(np.int32)
    changed = np.full(10, -1, np.int32)
    for name, (slot, value) in changes.items():
        (first if name == "first" else changed)[slot] = value
    return first, changed


def test_unconsumed_summaries_refuse_the_next_ones(tmp_path: Path) -> None:
    with _writer(tmp_path) as writer:
        writer._stage_members({1: _record()})
        with pytest.raises(RuntimeError, match="lost their games"):
            writer._stage_members({2: _record()})
        writer._staged_members.clear()


def test_summaries_keep_member_sides_only(tmp_path: Path) -> None:
    with _writer(tmp_path) as writer:
        assert writer._prepare_members(1, 5, ACTIVE, ENTRY, {}) is None
        writer._stage_members({1: _record(changed=(1, 3))})
        record = writer._prepare_members(1, 5, ACTIVE, ENTRY, {})
        assert record is not None
        assert record.slots == (0, 1) and record.length == 5
        assert int(record.changed[1]) == 3
        solo = {"system_ids": {"team_a": "solo", "team_b": "solo"}}
        assert writer._prepare_members(1, 5, ACTIVE, solo, {}) is None
        writer._staged_members.clear()


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ((np.zeros(9, np.int32), np.full(10, -1, np.int32)), "int32 with ten slots"),
        ((np.zeros(10, np.int64), np.full(10, -1, np.int32)), "int32 with ten slots"),
        (_record(first=(3, 0)), "inactive slots"),
        (_record(first=(0, 2)), "outside its component table"),
        (_record(changed=(0, 5)), "outside its game"),
        (_record(changed=(0, 0)), "outside its game"),
    ],
)
def test_summaries_refuse_impossible_records(
    tmp_path: Path,
    record: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
    message: str,
) -> None:
    with _writer(tmp_path) as writer:
        writer._stage_members({1: record})
        with pytest.raises(ValueError, match=message):
            writer._prepare_members(1, 5, ACTIVE, ENTRY, {})
        writer._staged_members.clear()
