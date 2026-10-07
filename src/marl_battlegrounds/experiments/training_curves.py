"""Plot point margin, kills and deaths over training from saved validation checks.

Use python -m marl_battlegrounds.experiments.training_curves
+training_runs=season_0 runs_dir=/path/hyperparameter-search
output_dir=/path/curves for the Season 0 baselines, or give your own
'lineages=[{name: Mine, runs: [/path/run]}]'. A lineage is one training
history: a run, then any runs that continued it, in training order (for
example a 40M run and its continuation to 250M). The stage reads every routine
validation check of every run through marl_battlegrounds.training.list_runs
with progress=True and writes curves.csv, curves.png and summary.md. It plays no
games and needs no GPU; the plot needs matplotlib (the viz extra).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
# isort: on


def read_lineage(name: str, runs: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read every routine validation check of one training history.

    Parameters
    ----------
    name : str
        Display name written into every row, for example ``"Recurrent IPPO"``.
    runs : sequence of str or pathlib.Path
        Exact training run folders in training order: the first run, then each
        run that continued the one before it. One folder is a complete history.

    Returns
    -------
    list of dict
        One row per routine check in step order: ``system``, ``run`` (1 for the
        first folder, 2 for the next, and so on), ``env_steps`` (game steps
        trained when the checkpoint was saved; step 0 is the untrained policy),
        ``games`` (validation games played at that check), and per game
        ``margin`` (points minus the opponents' points), ``kills`` and
        ``deaths``. Margin, kills and deaths average the check's map and
        opponent cells equally, as the trainer's checkpoint choice does.
        Confirmation games are not included.

    Raises
    ------
    ValueError
        A folder is not exactly one readable training run with validation
        progress, or the checks do not rise strictly in steps from one folder
        to the next (folders out of order, or two that overlap).
    OSError
        Raised by ``training.list_runs`` for a missing folder.
    """
    from marl_battlegrounds import training

    rows: list[dict[str, Any]] = []
    for number, run in enumerate(runs, start=1):
        found = [
            row for row in training.list_runs(run, progress=True) if "progress" in row
        ]
        if len(found) != 1:
            raise ValueError(f"{run} is not exactly one readable training run")
        progress: dict[str, Any] = found[0]["progress"]  # pyright: ignore[reportAssignmentType] - list_runs documents this mapping.
        for check in progress["checks"]:
            steps = int(check["env_steps"])
            if rows and steps <= rows[-1]["env_steps"]:
                raise ValueError(
                    f"{run} has a check at step {steps:,}, not after "
                    f"{rows[-1]['env_steps']:,}; list runs in training order"
                )
            rows.append(
                {
                    "system": name,
                    "run": number,
                    "env_steps": steps,
                    "games": sum(int(cell["games"]) for cell in check["opponents"]),
                    "margin": round(float(check["margin"]), 4),
                    "kills": round(float(check["kills_for"]), 4),
                    "deaths": round(float(check["kills_against"]), 4),
                }
            )
    return rows


def plot_curves(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Draw point margin and kills per game against game steps, one line each.

    Parameters
    ----------
    rows : sequence of dict
        Nonempty rows from ``read_lineage``, one or more systems.
    path : pathlib.Path
        PNG file to create or replace.

    Notes
    -----
    A dotted vertical line marks each step at which a continuing run starts:
    the last check of the run it continues. The left panel has a solid line at
    a margin of 0, where a team scores as many points as its validation
    opponents.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    systems = list(dict.fromkeys(row["system"] for row in rows))
    # A continuing run starts from the last checkpoint of the run before it.
    starts = sorted(
        {
            rows[index - 1]["env_steps"]
            for index, row in enumerate(rows)
            if row["run"] > 1 and rows[index - 1]["run"] != row["run"]
        }
    )
    for axis, key, title in (
        (axes[0], "margin", "Point margin per game"),
        (axes[1], "kills", "Kills per game"),
    ):
        for system in systems:
            mine = [row for row in rows if row["system"] == system]
            axis.plot(
                [row["env_steps"] / 1e6 for row in mine],
                [row[key] for row in mine],
                marker=".",
                label=system,
            )
        for step in starts:
            axis.axvline(step / 1e6, color="gray", linestyle=":", linewidth=1)
        if key == "margin":
            axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Game steps trained (millions)")
        axis.grid(alpha=0.3)
    axes[1].legend(fontsize="small")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_summary(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write summary.md: each system's first, best and last routine check.

    Parameters
    ----------
    rows : sequence of dict
        Rows from ``read_lineage`` for one or more systems.
    path : pathlib.Path
        Markdown file to create or replace.

    Notes
    -----
    The best check has the highest point margin; the earliest wins a tie. A
    routine check plays fewer games than the trainer's confirmation games, so
    this best check can differ from the checkpoint the trainer selected.
    """
    lines = [
        "# Training Curves",
        "",
        "Point margin / kills / deaths per game at routine validation checks.",
        "",
        "| System | Checks | First | Best (step) | Last (step) |",
        "| --- | ---: | --- | --- | --- |",
    ]

    def cell(row: dict[str, Any]) -> str:
        return f"{row['margin']:+.1f} / {row['kills']:.1f} / {row['deaths']:.1f}"

    for system in dict.fromkeys(row["system"] for row in rows):
        mine = [row for row in rows if row["system"] == system]
        best = max(mine, key=lambda row: (row["margin"], -row["env_steps"]))
        last = mine[-1]
        lines.append(
            f"| {system} | {len(mine)} | {cell(mine[0])} | {cell(best)} "
            f"({best['env_steps']:,}) | {cell(last)} ({last['env_steps']:,}) |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra.main(version_base="1.3", config_path="configs", config_name="training_curves")
def main(config: DictConfig) -> dict[str, int]:
    """Read every lineage's validation checks and write the curve files.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``lineages`` (a list of ``{name, runs}``; ``+training_runs=season_0``
        fills the six Season 0 baselines and needs ``runs_dir``, the
        hyperparameter-search root) and ``output_dir``. Relative paths use the
        folder the command was started from.

    Returns
    -------
    dict of str to int
        Checks read per system name, in the order given.

    Raises
    ------
    ValueError
        No lineage is given, the runs hold no routine checks, or
        ``read_lineage`` rejects a run folder.
    """
    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    if not settings["lineages"]:
        raise ValueError("give at least one lineage")
    output = Path(to_absolute_path(str(Path(settings["output_dir"]).expanduser())))
    rows: list[dict[str, Any]] = []
    for lineage in settings["lineages"]:
        runs = [
            to_absolute_path(str(Path(run).expanduser())) for run in lineage["runs"]
        ]
        rows += read_lineage(lineage["name"], runs)
    if not rows:
        raise ValueError("the runs hold no routine validation checks")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "curves.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, output / "summary.md")
    plot_curves(rows, output / "curves.png")
    print(output / "summary.md")
    return {
        lineage["name"]: sum(row["system"] == lineage["name"] for row in rows)
        for lineage in settings["lineages"]
    }


if __name__ == "__main__":
    prepare_hydra()
    main()
