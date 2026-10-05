"""Check what a training summary says when a finished run selected no actor.

Contracts: ``_no_selection_reason`` puts the runner's saved
``selection_unavailable_reason`` first; a run with no validation results says
validation was off and names ``validation_opponents`` and ``validation_panel``;
a run with validation but no saved selection says no selection was saved. Every
case points to the final actor. ``_summary`` copies the saved reason from
``status.json`` and withholds it during checkpoint recovery.
"""

from __future__ import annotations

import json
from pathlib import Path

from marl_battlegrounds.training.analysis import (
    _no_selection_reason,  # pyright: ignore[reportPrivateUsage]
    _summary,  # pyright: ignore[reportPrivateUsage]
)


def test_saved_reason_comes_first() -> None:
    text = _no_selection_reason(
        {
            "selection_unavailable_reason": "No eligible trained checkpoint",
            "validation_checkpoints": 0,
        }
    )
    assert text == (
        "No eligible trained checkpoint, so no best checkpoint was chosen. "
        "Load the final actor."
    )


def test_run_without_validation_names_the_settings_that_add_it() -> None:
    text = _no_selection_reason(
        {"selection_unavailable_reason": None, "validation_checkpoints": 0}
    )
    assert text.startswith("This run had no validation, so no best checkpoint")
    assert "Load the final actor." in text
    assert "validation_opponents or validation_panel" in text


def test_run_with_validation_but_no_saved_selection() -> None:
    text = _no_selection_reason({"validation_checkpoints": 3})
    assert text == "No selection was saved. Load the final actor."


def test_summary_copies_the_saved_reason_except_during_recovery(
    tmp_path: Path,
) -> None:
    (tmp_path / "run_details.json").write_text(json.dumps({"config": {}}))
    (tmp_path / "status.json").write_text(
        json.dumps({"selection_unavailable_reason": "No eligible trained checkpoint"})
    )
    assert (
        _summary(tmp_path)["selection_unavailable_reason"]
        == "No eligible trained checkpoint"
    )
    (tmp_path / "checkpoint_recovery.json").write_text("{}")
    assert _summary(tmp_path)["selection_unavailable_reason"] is None
