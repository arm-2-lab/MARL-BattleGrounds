"""Run the small declared study through the public Python interface.

From the repository root, run::

    python examples/training_study.py output_dir=artifacts/example-study

The default JSON declares two short FF-IPPO runs. It checks the workflow, not
learning quality. Pass another config for a real declared experiment. To
resume, use resume_from instead of output_dir; the first launch's deadline
still applies. Use background=true for a detached launch. The package CLI provides
matching study run/start/status/stop commands. No GPU is selected by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

from marl_battlegrounds import training


def run(
    config: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    background: bool = False,
) -> int:
    """Run or detach a study through its public owner and print the saved report.

    config names a study JSON file. Fresh omission uses the shipped two-seed
    software example; resume omission inherits the original declaration and
    deadline. Choose exactly one output_dir or resume_from. background=False
    waits and returns zero only for complete work; True starts a detached owner
    and returns zero. Errors propagate without changing budgets or retrying.
    """
    if type(background) is not bool:
        raise TypeError("background must be true or false")
    if (output_dir is None) == (resume_from is None):
        raise ValueError("Choose exactly one output_dir or resume_from")
    if config is None and resume_from is None:
        config = Path(__file__).with_suffix(".json")
    function = training.start_study if background else training.run_study
    report = function(config, output_dir=output_dir, resume_from=resume_from)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if background or report["status"] == "complete" else 1


def main() -> int:
    """Compose packaged YAML, call the Python lesson and return its exit status."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("training_study", run)


if __name__ == "__main__":
    raise SystemExit(main())
