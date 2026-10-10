r"""Declare a population selection before games, then test its frozen members.

Install the hydra extra and use a new output folder::

    python examples/training/population_selection.py \
        output_dir=artifacts/population-demo

This software lesson compares Random, ALPHA and BETA on validation maps 42--46.
It saves the declared rule before games, selects with the public owner, and uses
only the selected frozen members for the separate test. Incomplete selection
returns status 1 and starts no test games. No short run proves learned skill.
Use --help or --cfg job for settings; import run for plain Python.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def run(*, output_dir: str | Path) -> int:
    """Declare a tournament, select its population and test the frozen members.

    Parameters
    ----------
    output_dir : str | Path
        Parent for the example's declaration, selection games and held-out test. An
        incomplete selection does not start the test.

    Returns
    -------
    int
        Zero after complete selection/test, or one if selection is incomplete.

    Notes
    -----
    API errors propagate without an automatic retry. Python callers use current
    working-directory paths; Hydra preserves its original launch directory.
    """
    output_dir = Path(output_dir)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import marl_battlegrounds as marl_bgs

    entrants = ["random", "tdm-alpha", "tdm-beta"]
    result = marl_bgs.run_tournament(
        config={
            "entrants": entrants,
            "maps": [42, 43, 44, 45, 46],
            "games_per_opponent": 10,
            "max_steps": 2,
            "seed": 17,
            "selection": {
                "size": 2,
                "entrant_order": entrants,
                "failure_policy": "require-complete-field",
            },
        },
        output_dir=output_dir / "selection",
        num_envs=2,
        chunk_size=2,
    )
    decision = marl_bgs.select_initial_population(
        result, output_dir=output_dir / "decision"
    )
    print(json.dumps(decision, indent=2, allow_nan=False))
    if decision["status"] != "complete":
        return 1
    selected = [row["controller"]["reference"] for row in decision["members"]]
    tested = marl_bgs.run_tournament(
        config={
            "entrants": selected,
            "maps": [47, 48, 49, 50, 51],
            "games_per_opponent": 10,
            "max_steps": 2,
            "seed": 29,
            "selection": {
                "stage": "test",
                "population": str(output_dir / "decision" / "population.json"),
            },
        },
        output_dir=output_dir / "test",
        num_envs=2,
        chunk_size=2,
    )
    print(f"Test Status: {tested.status.capitalize()}")
    print(f"Population: {decision['population_id']}")
    return 0 if tested.status == "complete" else 1


def main() -> int:
    """Compose packaged YAML and call run; use --help or --cfg job to inspect."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("population_selection", run)


if __name__ == "__main__":
    raise SystemExit(main())
