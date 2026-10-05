"""Write a named checkpoint decision and optionally run its missing confirmations.

The public selection and validation owners check evidence and play games. This
Hydra stage only forwards their exact requests, then writes a linked decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
# isort: on


@hydra.main(version_base="1.3", config_path="configs", config_name="select")
def main(config: DictConfig) -> dict[str, Any]:
    """Return the saved decision, optionally completing its exact missing tasks.

    Fresh calls need run_dirs, declaration and a new output_dir. resume_from
    names an existing selection folder; omit those three inputs on resume.
    confirm=False only reports requests. True runs ready requests through
    validate_checkpoint, including expected_task, and writes a linked decision
    under confirmed/. Existing tasks resume through validation's owner. A saved
    confirmed decision is read and verified instead of replaced. Live bindings
    require the Python API and fail clearly here. Errors print the original
    decision folder; original run selections and decisions are never rewritten.
    """
    from marl_battlegrounds import training
    from marl_battlegrounds.training.selection import read_selection_decision

    if type(config.confirm) is not bool:
        raise TypeError("confirm must be true or false")
    if not config.resume_from and not config.output_dir:
        raise ValueError("Selection needs output_dir or resume_from")
    root = Path(
        to_absolute_path(
            str(Path(config.resume_from or config.output_dir).expanduser())
        )
    )
    try:
        if config.resume_from:
            if config.run_dirs or config.declaration or config.output_dir:
                raise ValueError(
                    "Selection resume inherits run_dirs/declaration/output_dir"
                )
            decision = read_selection_decision(root / "selection_decision.json")
            if decision["status"] == "complete":
                decision = read_selection_decision(
                    root / "selection_decision.json", verify_evidence=True
                )
        else:
            if not config.output_dir or not config.declaration:
                raise ValueError("Selection needs declaration and output_dir")
            decision = training.reselect_checkpoint(
                [
                    to_absolute_path(str(Path(path).expanduser()))
                    for path in config.run_dirs
                ],
                declaration=to_absolute_path(
                    str(Path(config.declaration).expanduser())
                ),
                output_dir=root,
            )
        if config.confirm and decision["status"] == "needs_confirmation":
            completed = root / "confirmed/selection_decision.json"
            if completed.exists():
                child = read_selection_decision(completed, verify_evidence=True)
                if child.get("previous_decision_id") != decision["decision_id"]:
                    raise ValueError(
                        f"Confirmed decision belongs to another selection: {completed}"
                    )
                decision = child
            else:
                if completed.parent.exists() and any(completed.parent.iterdir()):
                    raise ValueError(
                        "Confirmation output contains unfinished or unrelated files: "
                        f"{completed.parent}"
                    )
                requests = [
                    request
                    for run in decision["runs"]
                    for request in run["needs_confirmation"]
                ]
                if any(not row["ready_to_run"] for row in requests):
                    raise ValueError(
                        "Confirmation needs live Python bindings; use the saved "
                        "requests with training.validate_checkpoint"
                    )
                summaries: list[str] = list(
                    decision["declaration"].get("confirmation_results", [])
                )
                for request in requests:
                    training.validate_checkpoint(**request["arguments"])
                    summaries.append(
                        str(
                            Path(request["arguments"]["output_dir"])
                            / "validation_summary.json"
                        )
                    )
                declaration = {
                    **decision["declaration"],
                    "previous_decision": str(root / "selection_decision.json"),
                    "confirmation_results": summaries,
                }
                decision = training.reselect_checkpoint(
                    [run["run_dir"] for run in decision["runs"]],
                    declaration=declaration,
                    output_dir=root / "confirmed",
                )
        print(json.dumps(decision, sort_keys=True, allow_nan=False))
        return decision
    except Exception:
        print(f"Selection output: {root}", file=sys.stderr)
        raise


if __name__ == "__main__":
    prepare_hydra()
    main()
