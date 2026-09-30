"""Run or preview a prepared study through the public study API.

Use python -m marl_battlegrounds.experiments.study declaration=/path/study.yaml
output_dir=/path/study. This stage adds no scheduler, seed expansion or selection.
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


@hydra.main(version_base="1.3", config_path="configs", config_name="study")
def main(config: DictConfig) -> dict[str, Any]:
    """Call run_study or preview_study once and return its unchanged report.

    config supplies declaration, output_dir, resume_from and preview. Paths use
    Hydra's original working directory. Fresh execution needs declaration/output;
    resume inherits omitted settings. Preview starts no numerical backend and
    writes no study files. Execution errors print the exact durable study path
    before propagating. The API owns input checks, attempts and recovery.
    """
    from marl_battlegrounds import training

    if type(config.preview) is not bool:
        raise TypeError("preview must be true or false")
    declaration = (
        to_absolute_path(str(Path(config.declaration).expanduser()))
        if config.declaration
        else None
    )
    output = (
        to_absolute_path(str(Path(config.output_dir).expanduser()))
        if config.output_dir
        else None
    )
    resume = (
        to_absolute_path(str(Path(config.resume_from).expanduser()))
        if config.resume_from
        else None
    )
    owner = training.preview_study if config.preview else training.run_study
    try:
        result = owner(declaration, output_dir=output, resume_from=resume)
    except Exception:
        print(f"Study output: {resume or output}", file=sys.stderr)
        raise
    print(
        json.dumps(result, sort_keys=True, allow_nan=False)
        if config.preview
        else str(Path(result["study_dir"]) / "report.json")
    )
    return result


if __name__ == "__main__":
    prepare_hydra()
    main()
