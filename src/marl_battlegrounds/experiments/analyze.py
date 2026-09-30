"""Build reports and manuscript tables from explicit saved study directories.

Run python -m marl_battlegrounds.experiments.analyze study_dirs=[/path/study]
output_dir=/path/reports. Add grouping=/path/grouping.yaml, selection=/path/
selection_decision.json and evaluation_dirs=[/path/results] when available.
This entry calls the public analysis/export APIs and launches no experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Import the optional adapter first for its clear installation error.
# isort: off
from marl_battlegrounds import training
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
# isort: on


def run_analysis(config: DictConfig) -> dict[str, Any]:
    """Read declared inputs, call shared report owners once, and print outputs.

    config comes from packaged YAML. Paths use the original working directory;
    paths inside grouping/protocol files use each declaration's directory.
    analysis_data_dir defaults to output_dir/analysis-data. Saved run discovery
    includes only readable run folders for plots; export still includes every
    planned case and attempt. Errors preserve raw inputs and name chosen outputs.
    """
    roots = [
        Path(to_absolute_path(str(Path(value).expanduser())))
        for value in config.study_dirs
    ]
    output = Path(to_absolute_path(str(Path(config.output_dir).expanduser())))
    data = (
        Path(to_absolute_path(str(Path(config.analysis_data_dir).expanduser())))
        if config.analysis_data_dir
        else output / "analysis-data"
    )
    run_dirs = [
        str(row["run_dir"])
        for root in roots
        for row in training.list_runs(root)
        if (Path(str(row["run_dir"])) / "run_details.json").is_file()
    ]
    grouping = (
        to_absolute_path(str(Path(config.grouping).expanduser()))
        if config.grouping
        else None
    )
    protocol = (
        to_absolute_path(str(Path(config.protocol).expanduser()))
        if config.protocol
        else None
    )
    # Resolve required authored values before either output owner writes files.
    if protocol:
        from marl_battlegrounds.training._launch import read_declaration

        read_declaration(Path(protocol))
    reports: list[dict[str, Any]] = []
    if run_dirs or grouping:
        reports.append(
            training.analyze(
                None if grouping else run_dirs,
                output_dir=output,
                grouping=grouping,
                selection=to_absolute_path(str(Path(config.selection).expanduser()))
                if config.selection
                else None,
            )
        )
    paths = training.export_study_data(
        roots,
        output_dir=data,
        reports=reports,
        evaluation_dirs=[
            to_absolute_path(str(Path(value).expanduser()))
            for value in config.evaluation_dirs
        ],
        protocol=protocol,
    )
    result = {
        "reports": [report["artifacts"] for report in reports],
        "data_index": str(paths["data_index.json"]),
    }
    print(json.dumps(result, sort_keys=True))
    return result


@hydra.main(version_base="1.3", config_path="configs", config_name="analyze")
def main(config: DictConfig) -> dict[str, Any]:
    """Return the shared analysis result and report chosen output paths on error.

    config uses the packaged analyze settings. run_analysis owns only stage
    wiring; the public analysis/export APIs own calculations and file writing.
    Errors propagate, leaving original results and completed derived files intact.
    """
    try:
        return run_analysis(config)
    except Exception:
        output = Path(to_absolute_path(str(Path(config.output_dir).expanduser())))
        data = (
            to_absolute_path(str(Path(config.analysis_data_dir).expanduser()))
            if config.analysis_data_dir
            else str(output / "analysis-data")
        )
        print(f"Analysis reports: {output}\nAnalysis tables: {data}", file=sys.stderr)
        raise


if __name__ == "__main__":
    prepare_hydra()
    main()
