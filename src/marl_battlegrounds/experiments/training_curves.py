"""Plot point margin, kills and deaths over training from saved validation checks.

Use python -m marl_battlegrounds.experiments.training_curves
+training_runs=season_0 runs_dir=/path/hyperparameter-search
output_dir=/path/curves for the Season 0 baselines, or give your own
'lineages=[{name: Mine, runs: [/path/run]}]'. A lineage is one training
history: a run, then any runs that continued it, in training order (for
example a 40M run and its continuation to 250M). A lineage may name a released
model repo instead of runs, '{name: QMIX, repo:
MARL-BattleGrounds/tdm-season-0-qmix}': the stage then downloads only that
repo's small training records (run details, status and validation results,
not the update logs) into the Hugging Face cache and reads them in training
order. The stage reads every routine
validation check of every run through marl_battlegrounds.training.list_runs
with progress=True and writes curves.csv, crossings.csv (the first check with
a point margin above zero and the first that won half its games), curves.png
and summary.md. Add
'opponents=[tdm-alpha,tdm-beta,tdm-gamma]' to draw the curves against those
validation opponents only, for example to see how long a method takes to beat
the scripted bots. It plays no games and needs no GPU; the plot needs
matplotlib (the viz extra).
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


def read_lineage(
    name: str,
    runs: Sequence[str | Path],
    opponents: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Read every routine validation check of one training history.

    Parameters
    ----------
    name : str
        Display name written into every row, for example ``"Recurrent IPPO"``.
    runs : sequence of str or pathlib.Path
        Exact training run folders in training order: the first run, then each
        run that continued the one before it. One folder is a complete history.
    opponents : sequence of str or None, default=None
        Validation opponent names as the run saved them (for example
        ``"tdm-alpha"``). None keeps every opponent of each check. Given names
        restrict every value to those opponents, each opponent weighted
        equally.

    Returns
    -------
    list of dict
        One row per routine check in step order: ``system``, ``run`` (1 for the
        first folder, 2 for the next, and so on), ``env_steps`` (game steps
        trained when the checkpoint was saved; step 0 is the untrained policy),
        ``games`` (validation games played at that check against the chosen
        opponents), per game ``margin`` (points minus the opponents' points),
        ``kills`` and ``deaths``, and ``win_share`` (the share of those games
        won, 0 to 1). Margin, kills and deaths average the check's map and
        opponent cells equally, as the trainer's checkpoint choice does.
        Confirmation games are not included.

    Raises
    ------
    ValueError
        A folder is not exactly one readable training run with validation
        progress, or the checks do not rise strictly in steps from one folder
        to the next (folders out of order, or two that overlap), or a check
        lacks a named opponent or that opponent's saved points and kills.
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
            cells = check["opponents"]
            if opponents is None:
                margin, kills, deaths = (
                    check["margin"],
                    check["kills_for"],
                    check["kills_against"],
                )
            else:
                saved = {str(cell["name"]): cell for cell in cells}
                missing = [opponent for opponent in opponents if opponent not in saved]
                if missing:
                    raise ValueError(
                        f"{run} check at step {steps:,} has no opponent {missing[0]!r}"
                    )
                cells = [saved[opponent] for opponent in opponents]
                values = [
                    [cell[key] for cell in cells]
                    for key in ("margin", "kills_for", "kills_against")
                ]
                if any(value is None for group in values for value in group):
                    raise ValueError(
                        f"{run} check at step {steps:,} lacks saved points or kills"
                    )
                margin, kills, deaths = (sum(group) / len(group) for group in values)
            games = sum(int(cell["games"]) for cell in cells)
            rows.append(
                {
                    "system": name,
                    "run": number,
                    "env_steps": steps,
                    "games": games,
                    "margin": round(float(margin), 4),
                    "kills": round(float(kills), 4),
                    "deaths": round(float(deaths), 4),
                    "win_share": round(
                        sum(int(cell["wins"]) for cell in cells) / games, 4
                    ),
                }
            )
    return rows


def plot_curves(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Draw point margin, kills and the share of games won against game steps.

    Parameters
    ----------
    rows : sequence of dict
        Nonempty rows from ``read_lineage``, one or more systems.
    path : pathlib.Path
        PNG file to create or replace.

    Notes
    -----
    A dotted vertical line marks each step at which a continuing run starts:
    the last check of the run it continues. The margin panel has a solid line
    at 0, where a team scores as many points as its validation opponents, and
    the panel of games won has one at 50%.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
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
        (axes[2], "win_share", "Games won (%)"),
    ):
        scale = 100 if key == "win_share" else 1
        for system in systems:
            mine = [row for row in rows if row["system"] == system]
            axis.plot(
                [row["env_steps"] / 1e6 for row in mine],
                [row[key] * scale for row in mine],
                marker=".",
                label=system,
            )
        for step in starts:
            axis.axvline(step / 1e6, color="gray", linestyle=":", linewidth=1)
        if key in ("margin", "win_share"):
            axis.axhline(50 if key == "win_share" else 0, color="black", linewidth=0.8)
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


def first_crossings(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find when each system first beat its opponents, by margin and by games won.

    Parameters
    ----------
    rows : sequence of dict
        Rows from ``read_lineage`` for one or more systems, each system's
        rows in training order.

    Returns
    -------
    list of dict
        One row per system, in first-seen order: ``system``,
        ``margin_above_zero_steps`` (``env_steps`` of the first check with a
        point margin above zero) and ``half_won_steps`` (``env_steps`` of the
        first check that won at least half its games). None when no check
        reached it; the CSV writes None as an empty cell.
    """
    result: list[dict[str, Any]] = []
    for system in dict.fromkeys(row["system"] for row in rows):
        mine = [row for row in rows if row["system"] == system]
        result.append(
            {
                "system": system,
                "margin_above_zero_steps": next(
                    (row["env_steps"] for row in mine if row["margin"] > 0), None
                ),
                "half_won_steps": next(
                    (row["env_steps"] for row in mine if row["win_share"] >= 0.5),
                    None,
                ),
            }
        )
    return result


def _released_runs(repo: str) -> list[str]:
    """Fetch a released repo's training records and order them by training.

    repo is a released model repo id such as
    ``"MARL-BattleGrounds/tdm-season-0-qmix"``. Only the small record files
    are downloaded into the Hugging Face cache (released_models owns this).
    Returns the record folders, ordered by their first routine validation
    step, so a run and its continuation read in training order. Raises
    ValueError when the records cannot be fetched or a folder has no checks.
    """
    from marl_battlegrounds import training
    from marl_battlegrounds.released_models import (
        _training_records,  # pyright: ignore[reportPrivateUsage]
    )

    _, folders = _training_records(repo)

    def first_step(folder: Path) -> int:
        """Return the step of the folder's first routine validation check."""
        for row in training.list_runs(folder, progress=True):
            checks = row.get("progress", {}).get("checks", [])  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType] - list_runs documents this mapping.
            if checks:
                return min(int(check["env_steps"]) for check in checks)  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
        raise ValueError(f"{folder} holds no routine validation checks")

    return [str(folder) for folder in sorted(folders, key=first_step)]


@hydra.main(version_base="1.3", config_path="configs", config_name="training_curves")
def main(config: DictConfig) -> dict[str, int]:
    """Read every lineage's validation checks and write the curve files.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``lineages`` (a list of ``{name, runs}`` or ``{name, repo}``;
        ``+training_runs=season_0`` fills the six Season 0 baselines and needs
        ``runs_dir``, the hyperparameter-search root), ``opponents``
        (validation opponent names to keep, or null for all; see
        ``read_lineage``) and ``output_dir``. Relative paths use the folder
        the command was started from. ``repo`` names a released model repo,
        such as ``MARL-BattleGrounds/tdm-season-0-qmix``; its training records
        are downloaded at the repo's pinned commit (or its latest commit, with
        a warning), and its runs are read in the order of their first
        validation check.

    Returns
    -------
    dict of str to int
        Checks read per system name, in the order given.

    Raises
    ------
    ValueError
        No lineage is given, a lineage gives both or neither of runs and repo,
        a released repo's records cannot be fetched, the runs hold no routine
        checks, or ``read_lineage`` rejects a run folder or an opponent name.
    """
    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    if not settings["lineages"]:
        raise ValueError("give at least one lineage")
    output = Path(to_absolute_path(str(Path(settings["output_dir"]).expanduser())))
    rows: list[dict[str, Any]] = []
    for lineage in settings["lineages"]:
        if ("runs" in lineage) == ("repo" in lineage):
            raise ValueError(f"lineage {lineage['name']!r} needs runs or repo")
        runs = (
            _released_runs(lineage["repo"])
            if "repo" in lineage
            else [
                to_absolute_path(str(Path(run).expanduser())) for run in lineage["runs"]
            ]
        )
        rows += read_lineage(lineage["name"], runs, settings.get("opponents"))
    if not rows:
        raise ValueError("the runs hold no routine validation checks")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "curves.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    crossings = first_crossings(rows)
    with (output / "crossings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(crossings[0]))
        writer.writeheader()
        writer.writerows(crossings)
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
